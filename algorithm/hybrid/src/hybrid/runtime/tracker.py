from __future__ import annotations

from dataclasses import dataclass

import torch

from hybrid.models.patch_embeddings import PatchEmbedCache
from hybrid.runtime.similarity import ellipse_similarity


@dataclass
class RuntimeTrackerState:
    prev_state: torch.Tensor | None = None
    mode: str = "search"
    patch_cache: PatchEmbedCache | None = None


class RuntimeHBTXRTracker:
    def __init__(self, model) -> None:
        self.model = model
        self.state = RuntimeTrackerState()

    def reset(self) -> None:
        self.state = RuntimeTrackerState()
        self.model.scheduler.reset()

    @torch.no_grad()
    def step(self, *, frame: torch.Tensor, event: torch.Tensor, event_density: torch.Tensor | None = None, closed_eye_flag: torch.Tensor | None = None):
        if self.state.prev_state is None:
            batch = {
                "frame": frame,
                "event": event,
                "prev_state": torch.zeros(frame.shape[0], 6, device=frame.device),
                "cached_frame": frame,
                "patch_cache": self.state.patch_cache,
            }
            outputs = self.model.forward_train(batch)
            self.state.prev_state = outputs["search/state"]
            self.state.mode = "search"
            self.state.patch_cache = self.model.prime_patch_cache(frame)
            outputs["runtime/state"] = "search"
            outputs["runtime/reason"] = "bootstrap"
            return outputs

        batch = {
            "frame": frame,
            "event": event,
            "prev_state": self.state.prev_state,
            "cached_frame": frame,
            "patch_cache": self.state.patch_cache,
        }
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
        if outputs["runtime/state"] == "track":
            self.state.prev_state = outputs["track/state"]
            self.state.patch_cache = self.model.prime_patch_cache(frame)
        else:
            self.state.prev_state = outputs["search/state"]
            if closed_eye_value:
                self.state.patch_cache = self.model.invalidate_patch_cache()
            else:
                self.state.patch_cache = self.model.prime_patch_cache(frame)
        self.state.mode = str(outputs["runtime/state"])
        return outputs
