from __future__ import annotations

from collections.abc import Callable

import torch

from ..patch_embeddings import PatchEmbedCache


class RuntimeStepPolicy:
    def __init__(
        self,
        *,
        scheduler,
        prime_patch_cache: Callable[[torch.Tensor], PatchEmbedCache],
        invalidate_patch_cache: Callable[[], PatchEmbedCache],
    ) -> None:
        self.scheduler = scheduler
        self.prime_patch_cache = prime_patch_cache
        self.invalidate_patch_cache = invalidate_patch_cache

    def apply(
        self,
        *,
        frame: torch.Tensor,
        outputs: dict[str, torch.Tensor | str | PatchEmbedCache],
        similarity: torch.Tensor | None = None,
        event_density: torch.Tensor | None = None,
        closed_eye_flag: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | str | PatchEmbedCache]:
        search_conf = torch.sigmoid(outputs["search/pupil"][..., 6]).mean().item()
        track_conf = torch.sigmoid(outputs["track/pupil"][..., 6]).mean().item()
        track_quality = torch.sigmoid(outputs["track/pupil"][..., 7]).mean().item()
        similarity_value = float(similarity.mean().item()) if similarity is not None else 1.0
        density_value = float(event_density.mean().item()) if event_density is not None else 1.0
        closed_eye_value = bool(closed_eye_flag.mean().item() > 0.5) if closed_eye_flag is not None else False
        decision = self.scheduler.step(
            search_conf=search_conf,
            track_conf=track_conf,
            track_quality=track_quality,
            similarity=similarity_value,
            event_density=density_value,
            closed_eye_flag=closed_eye_value,
        )
        if closed_eye_value:
            next_cache = self.invalidate_patch_cache()
        else:
            next_cache = self.prime_patch_cache(frame)
        outputs["runtime/state"] = decision.state
        outputs["runtime/reason"] = decision.reason
        outputs["runtime/ellipse_state"] = outputs["track/state"] if decision.state == "track" else outputs["search/state"]
        outputs["runtime/search_conf"] = torch.full((frame.shape[0],), float(search_conf), device=frame.device)
        outputs["runtime/track_conf"] = torch.full((frame.shape[0],), float(track_conf), device=frame.device)
        outputs["runtime/track_quality"] = torch.full((frame.shape[0],), float(track_quality), device=frame.device)
        outputs["runtime/patch_cache"] = next_cache
        outputs["runtime/cache_valid"] = torch.full((frame.shape[0],), 1.0 if next_cache.valid else 0.0, device=frame.device)
        return outputs
