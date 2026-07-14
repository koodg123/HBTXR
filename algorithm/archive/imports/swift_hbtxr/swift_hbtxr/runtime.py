from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .antiblink import AntiBlinkDetector
from .geometry import xyabuv_to_xywht


def ellipse_similarity(prev_state: torch.Tensor, cur_state: torch.Tensor) -> torch.Tensor:
    prev_center = prev_state[..., 0:2]
    cur_center = cur_state[..., 0:2]
    prev_axes = torch.clamp(prev_state[..., 2:4], min=1e-3)
    cur_axes = torch.clamp(cur_state[..., 2:4], min=1e-3)

    center_dist = torch.linalg.norm(prev_center - cur_center, dim=-1)
    center_scale = torch.clamp(prev_axes.mean(dim=-1) + cur_axes.mean(dim=-1), min=1.0)
    axes_ratio = torch.abs(torch.log(cur_axes / prev_axes)).mean(dim=-1)
    angle_delta = 1.0 - torch.sum(prev_state[..., 4:6] * cur_state[..., 4:6], dim=-1).clamp(-1.0, 1.0)

    score = 1.0 - torch.clamp(0.5 * (center_dist / center_scale) + 0.35 * axes_ratio + 0.15 * angle_delta, min=0.0, max=1.0)
    return score.clamp(0.0, 1.0)


@dataclass
class RuntimeTrackerState:
    prev_state: torch.Tensor | None = None
    last_valid_state: torch.Tensor | None = None
    antiblink_armed: bool = False
    mode: str = "search"
    session_key: str | None = None


def _as_bool_flag(value: torch.Tensor | None) -> bool:
    if value is None:
        return False
    return bool(value.detach().float().mean().item() > 0.5)


def _as_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().float().mean().item())


def _ellipse_is_plausible(ellipse_xywht: torch.Tensor, *, frame_hw: tuple[int, int]) -> torch.Tensor:
    height, width = int(frame_hw[0]), int(frame_hw[1])
    center_x = ellipse_xywht[..., 0]
    center_y = ellipse_xywht[..., 1]
    axis_w = ellipse_xywht[..., 2]
    axis_h = ellipse_xywht[..., 3]
    center_ok = (center_x >= 0.0) & (center_x < float(width)) & (center_y >= 0.0) & (center_y < float(height))
    size_ok = (axis_w >= 4.0) & (axis_h >= 4.0) & (axis_w <= float(width)) & (axis_h <= float(height))
    area_ok = (axis_w * axis_h) >= 32.0
    return center_ok & size_ok & area_ok


class RuntimeSwiftHBTXRTracker:
    def __init__(
        self,
        model,
        *,
        antiblink_detector: AntiBlinkDetector | None = None,
        hold_last_on_blink: bool = True,
    ) -> None:
        self.model = model
        self.antiblink_detector = antiblink_detector
        self.hold_last_on_blink = bool(hold_last_on_blink)
        self.state = RuntimeTrackerState()
        if self.antiblink_detector is not None:
            self.antiblink_detector.eval()

    def reset(self) -> None:
        self.state = RuntimeTrackerState()
        self.model.scheduler.reset()

    def _run_antiblink(self, *, frame: torch.Tensor, state_xyabuv: torch.Tensor) -> dict[str, Any]:
        if self.antiblink_detector is None:
            return {}
        detector = self.antiblink_detector.to(frame.device)
        ellipse_xywht = xyabuv_to_xywht(state_xyabuv)
        outputs = detector(frame, ellipse_xywht)
        outputs["reference_valid"] = _ellipse_is_plausible(
            ellipse_xywht,
            frame_hw=(int(frame.shape[-2]), int(frame.shape[-1])),
        ).to(dtype=torch.float32)
        return {f"antiblink/{key}": value for key, value in outputs.items()}

    @torch.no_grad()
    def step(
        self,
        *,
        frame: torch.Tensor,
        event: torch.Tensor,
        event_density: torch.Tensor | None = None,
        closed_eye_flag: torch.Tensor | None = None,
        session_key: str | None = None,
    ) -> dict[str, torch.Tensor | str]:
        needs_bootstrap = self.state.prev_state is None or self.state.prev_state.shape[0] != frame.shape[0]
        if session_key is not None and self.state.session_key is not None and session_key != self.state.session_key:
            needs_bootstrap = True

        if needs_bootstrap:
            self.model.scheduler.reset()
            batch = {"frame": frame, "event": event, "prev_state": torch.zeros(frame.shape[0], 6, device=frame.device)}
            outputs = self.model.forward_train(batch)
            outputs.update(self._run_antiblink(frame=frame, state_xyabuv=outputs["search/state"]))
            search_state = outputs["search/state"].detach()
            self.state.prev_state = search_state
            self.state.last_valid_state = None
            self.state.antiblink_armed = False
            self.state.mode = "search"
            self.state.session_key = session_key
            outputs["runtime/state"] = "search"
            outputs["runtime/fsm_reason"] = "bootstrap"
            outputs["runtime/reason"] = "bootstrap"
            outputs["runtime/output_mode"] = "search"
            outputs["runtime/ellipse_state"] = search_state
            return outputs

        batch = {"frame": frame, "event": event, "prev_state": self.state.prev_state}
        outputs = self.model.forward_train(batch)
        outputs.update(self._run_antiblink(frame=frame, state_xyabuv=outputs["search/state"]))

        similarity = ellipse_similarity(outputs["search/state"], outputs["track/state"])
        search_conf = torch.sigmoid(outputs["search/pupil"][..., 6]).mean().item()
        track_conf = torch.sigmoid(outputs["track/pupil"][..., 6]).mean().item()
        track_quality = torch.sigmoid(outputs["track/pupil"][..., 7]).mean().item()
        density_value = float(event_density.mean().item()) if event_density is not None else 1.0
        provided_closed_eye = _as_bool_flag(closed_eye_flag)
        detector_closed_eye = _as_bool_flag(outputs.get("antiblink/closed_eye_flag"))
        detector_hold_requested = _as_bool_flag(outputs.get("antiblink/should_hold"))
        reference_valid = _as_bool_flag(outputs.get("antiblink/reference_valid"))
        antiblink_armed = bool(self.state.antiblink_armed and self.state.last_valid_state is not None)
        detector_enabled = bool(antiblink_armed and reference_valid)
        hold_requested = provided_closed_eye or (detector_enabled and detector_hold_requested)
        effective_closed_eye = provided_closed_eye or (detector_enabled and detector_closed_eye)

        decision = self.model.scheduler.step(
            search_conf=search_conf,
            track_conf=track_conf,
            track_quality=track_quality,
            similarity=float(similarity.mean().item()),
            event_density=density_value,
            closed_eye_flag=effective_closed_eye,
        )
        selected_state = outputs["track/state"].detach() if decision.state == "track" else outputs["search/state"].detach()

        outputs["runtime/state"] = decision.state
        outputs["runtime/fsm_reason"] = decision.reason
        if self.hold_last_on_blink and hold_requested and self.state.last_valid_state is not None:
            outputs["runtime/reason"] = "hold_last"
            outputs["runtime/output_mode"] = "hold"
            outputs["runtime/ellipse_state"] = self.state.last_valid_state
            self.state.prev_state = self.state.last_valid_state
        else:
            outputs["runtime/reason"] = decision.reason
            outputs["runtime/output_mode"] = decision.state
            outputs["runtime/ellipse_state"] = selected_state
            self.state.prev_state = selected_state
            if not effective_closed_eye and decision.state == "track":
                self.state.last_valid_state = selected_state
                self.state.antiblink_armed = True

        self.state.mode = str(decision.state)
        self.state.session_key = session_key
        outputs["runtime/similarity"] = similarity.detach()
        return outputs


RuntimeSWIFTHBTXRTracker = RuntimeSwiftHBTXRTracker


def _to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return value.detach().cpu().tolist()
    return value


def _state_to_ellipse(state: torch.Tensor) -> list[list[float]]:
    ellipse = xyabuv_to_xywht(state)
    if isinstance(ellipse, torch.Tensor):
        return ellipse.detach().cpu().tolist()
    return ellipse.tolist()


def _slice_tensor_or_none(value: torch.Tensor | None, idx: int) -> torch.Tensor | None:
    if value is None:
        return None
    return value[idx:idx + 1]


@torch.no_grad()
def run_runtime_trace(
    model,
    loader,
    *,
    device: torch.device | str = "cpu",
    antiblink_detector: AntiBlinkDetector | None = None,
    hold_last_on_blink: bool = True,
) -> list[dict[str, Any]]:
    device = torch.device(device)
    tracker = RuntimeSwiftHBTXRTracker(
        model,
        antiblink_detector=antiblink_detector,
        hold_last_on_blink=hold_last_on_blink,
    )
    rows: list[dict[str, Any]] = []

    for batch in loader:
        frame = batch["frame"].to(device)
        event = batch["event"].to(device)
        event_density = batch.get("event_density")
        closed_eye_flag = batch.get("closed_eye_flag")
        if torch.is_tensor(event_density):
            event_density = event_density.to(device)
        if torch.is_tensor(closed_eye_flag):
            closed_eye_flag = closed_eye_flag.to(device)

        metas = batch.get("meta") or [{} for _ in range(frame.shape[0])]
        sample_ids = batch.get("sample_id") or [f"sample_{len(rows) + idx}" for idx in range(frame.shape[0])]

        for idx, sample_id in enumerate(sample_ids):
            meta = metas[idx] if isinstance(metas, list) else metas
            session_key = str(meta.get("session_key")) if isinstance(meta, dict) and meta.get("session_key") is not None else None
            outputs = tracker.step(
                frame=frame[idx:idx + 1],
                event=event[idx:idx + 1],
                event_density=_slice_tensor_or_none(event_density, idx),
                closed_eye_flag=_slice_tensor_or_none(closed_eye_flag, idx),
                session_key=session_key,
            )
            similarity = ellipse_similarity(outputs["search/state"], outputs["track/state"])
            search_conf = torch.sigmoid(outputs["search/pupil"][:, 6])
            track_conf = torch.sigmoid(outputs["track/pupil"][:, 6])
            track_quality = torch.sigmoid(outputs["track/pupil"][:, 7])
            output_ellipse = _state_to_ellipse(outputs["runtime/ellipse_state"])[0]
            search_ellipse = _state_to_ellipse(outputs["search/state"])[0]
            track_ellipse = _state_to_ellipse(outputs["track/state"])[0]
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "runtime_state": str(outputs["runtime/state"]),
                    "runtime_reason": str(outputs["runtime/reason"]),
                    "runtime_fsm_reason": str(outputs.get("runtime/fsm_reason", outputs["runtime/reason"])),
                    "output_mode": str(outputs.get("runtime/output_mode", outputs["runtime/state"])),
                    "search_confidence": float(search_conf[0].item()),
                    "track_confidence": float(track_conf[0].item()),
                    "track_quality": float(track_quality[0].item()),
                    "similarity": float(similarity[0].item()),
                    "event_density": None if event_density is None else float(event_density[idx].detach().cpu().item()),
                    "closed_eye_flag": None if closed_eye_flag is None else float(closed_eye_flag[idx].detach().cpu().item()),
                    "open_extent": _as_float(outputs.get("antiblink/open_extent")),
                    "should_hold": _as_bool_flag(outputs.get("antiblink/should_hold")),
                    "ellipse_xywht": output_ellipse,
                    "search_ellipse_xywht": search_ellipse,
                    "track_ellipse_xywht": track_ellipse,
                    "meta": _to_python(meta),
                }
            )
    return rows
