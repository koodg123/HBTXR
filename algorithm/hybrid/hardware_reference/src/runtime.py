from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def ellipse_similarity(prev_state: torch.Tensor, cur_state: torch.Tensor) -> torch.Tensor:
    center_dist = torch.linalg.norm(prev_state[..., :2] - cur_state[..., :2], dim=-1)
    scale = torch.clamp(prev_state[..., 2:4].abs().mean(dim=-1) + cur_state[..., 2:4].abs().mean(dim=-1), min=1.0)
    return (1.0 - torch.clamp(center_dist / scale, 0.0, 1.0)).clamp(0.0, 1.0)


@dataclass
class RuntimeState:
    prev_state: torch.Tensor | None = None
    last_valid_state: torch.Tensor | None = None
    anchor_state: torch.Tensor | None = None
    anchor_valid: bool = False
    session_key: str | None = None


class RuntimeHGTXRTracker:
    def __init__(self, model, *, antiblink_detector=None, hold_last_on_blink: bool = True) -> None:
        self.model = model
        self.antiblink_detector = antiblink_detector
        self.hold_last_on_blink = bool(hold_last_on_blink)
        self.state = RuntimeState()

    def reset(self) -> None:
        self.state = RuntimeState()
        self.model.scheduler.reset()

    @torch.no_grad()
    def step(self, *, frame: torch.Tensor, event: torch.Tensor, event_density: torch.Tensor | None = None, closed_eye_flag: torch.Tensor | None = None, session_key: str | None = None) -> dict[str, Any]:
        needs_bootstrap = self.state.prev_state is None or self.state.prev_state.shape[0] != frame.shape[0] or not self.state.anchor_valid
        if session_key is not None and self.state.session_key not in {None, session_key}:
            needs_bootstrap = True
        prev_state = torch.zeros(frame.shape[0], 6, device=frame.device) if needs_bootstrap else self.state.prev_state
        outputs = self.model.forward_train({"frame": frame, "event": event, "prev_state": prev_state})
        similarity = ellipse_similarity(outputs["search/state"], outputs["track/state"])
        search_conf = torch.sigmoid(outputs["search/pupil"][..., 6]).mean().item()
        track_conf = torch.sigmoid(outputs["track/pupil"][..., 6]).mean().item()
        track_quality = torch.sigmoid(outputs["track/pupil"][..., 7]).mean().item()
        density = float(event_density.mean().item()) if event_density is not None else 1.0
        closed = bool(closed_eye_flag.float().mean().item() > 0.5) if closed_eye_flag is not None else False
        if self.antiblink_detector is not None:
            blink = self.antiblink_detector(frame, outputs["search/state"])
            outputs.update({f"antiblink/{k}": v for k, v in blink.items()})
            closed = closed or bool(blink["closed_eye_flag"].float().mean().item() > 0.5)
        if needs_bootstrap:
            self.model.scheduler.reset()
            decision_state, reason = "search", "bootstrap"
        else:
            decision = self.model.scheduler.step(
                search_conf=search_conf,
                track_conf=track_conf,
                track_quality=track_quality,
                similarity=float(similarity.mean().item()),
                event_density=density,
                closed_eye_flag=closed,
            )
            decision_state, reason = decision.state, decision.reason
        selected = outputs["track/state"].detach() if decision_state == "track" else outputs["search/state"].detach()
        if closed and self.hold_last_on_blink and self.state.last_valid_state is not None:
            selected = self.state.last_valid_state
            reason = "hold_last"
        elif not closed and decision_state == "track":
            self.state.last_valid_state = selected
        if decision_state == "search":
            self.state.anchor_state = outputs["search/state"].detach()
            self.state.anchor_valid = True
        self.state.prev_state = selected
        self.state.session_key = session_key
        outputs["runtime/state"] = decision_state
        outputs["runtime/reason"] = reason
        outputs["runtime/ellipse_state"] = selected
        outputs["runtime/similarity"] = similarity
        return outputs

