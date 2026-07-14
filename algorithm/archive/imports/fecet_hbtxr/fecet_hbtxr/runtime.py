from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .io import xyabuv_to_xywht


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
    mode: str = "search"
    session_key: str | None = None


class RuntimeFECETHBTXRTracker:
    def __init__(self, model) -> None:
        self.model = model
        self.state = RuntimeTrackerState()

    def reset(self) -> None:
        self.state = RuntimeTrackerState()
        self.model.scheduler.reset()

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
            self.state.prev_state = outputs["search/state"]
            self.state.mode = "search"
            self.state.session_key = session_key
            outputs["runtime/state"] = "search"
            outputs["runtime/reason"] = "bootstrap"
            return outputs

        batch = {"frame": frame, "event": event, "prev_state": self.state.prev_state}
        outputs = self.model.forward_train(batch)
        similarity = ellipse_similarity(outputs["search/state"], outputs["track/state"])
        search_conf = torch.sigmoid(outputs["search/pupil"][..., 6]).mean().item()
        track_conf = torch.sigmoid(outputs["track/pupil"][..., 6]).mean().item()
        track_quality = torch.sigmoid(outputs["track/pupil"][..., 7]).mean().item()
        density_value = float(event_density.mean().item()) if event_density is not None else 1.0
        closed_eye_value = bool(closed_eye_flag.mean().item() > 0.5) if closed_eye_flag is not None else False
        decision = self.model.scheduler.step(
            search_conf=search_conf,
            track_conf=track_conf,
            track_quality=track_quality,
            similarity=float(similarity.mean().item()),
            event_density=density_value,
            closed_eye_flag=closed_eye_value,
        )
        outputs["runtime/state"] = decision.state
        outputs["runtime/reason"] = decision.reason
        if decision.state == "track":
            self.state.prev_state = outputs["track/state"]
        else:
            self.state.prev_state = outputs["search/state"]
        self.state.mode = str(decision.state)
        self.state.session_key = session_key
        return outputs


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


@torch.no_grad()
def run_runtime_trace(model, loader, *, device: torch.device | str = "cpu") -> list[dict[str, Any]]:
    device = torch.device(device)
    tracker = RuntimeFECETHBTXRTracker(model)
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
        session_key = None
        if metas and isinstance(metas, list):
            session_key = str(metas[0].get("session_key")) if metas[0].get("session_key") is not None else None

        outputs = tracker.step(
            frame=frame,
            event=event,
            event_density=event_density,
            closed_eye_flag=closed_eye_flag,
            session_key=session_key,
        )
        similarity = ellipse_similarity(outputs["search/state"], outputs["track/state"])
        search_conf = torch.sigmoid(outputs["search/pupil"][:, 6])
        track_conf = torch.sigmoid(outputs["track/pupil"][:, 6])
        track_quality = torch.sigmoid(outputs["track/pupil"][:, 7])
        search_ellipse = _state_to_ellipse(outputs["search/state"])
        track_ellipse = _state_to_ellipse(outputs["track/state"])

        for idx, sample_id in enumerate(sample_ids):
            runtime_state = str(outputs["runtime/state"])
            runtime_reason = str(outputs["runtime/reason"])
            selected_ellipse = track_ellipse[idx] if runtime_state == "track" else search_ellipse[idx]
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "runtime_state": runtime_state,
                    "runtime_reason": runtime_reason,
                    "search_confidence": float(search_conf[idx].item()),
                    "track_confidence": float(track_conf[idx].item()),
                    "track_quality": float(track_quality[idx].item()),
                    "similarity": float(similarity[idx].item()),
                    "event_density": None if event_density is None else float(event_density[idx].detach().cpu().item()),
                    "closed_eye_flag": None if closed_eye_flag is None else float(closed_eye_flag[idx].detach().cpu().item()),
                    "ellipse_xywht": selected_ellipse,
                    "search_ellipse_xywht": search_ellipse[idx],
                    "track_ellipse_xywht": track_ellipse[idx],
                    "meta": _to_python(metas[idx] if isinstance(metas, list) else metas),
                }
            )
    return rows
