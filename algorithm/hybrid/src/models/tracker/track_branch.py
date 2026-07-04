from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class TrackStateBranch:
    def __init__(
        self,
        *,
        prev_state_encoder: nn.Module,
        track_head: nn.Module | None,
        apply_width_mask: Callable[[torch.Tensor, int], torch.Tensor],
        apply_track_mask: Callable[[torch.Tensor, int], torch.Tensor],
        decode_state: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self.prev_state_encoder = prev_state_encoder
        self.track_head = track_head
        self.apply_width_mask = apply_width_mask
        self.apply_track_mask = apply_track_mask
        self.decode_state = decode_state

    @property
    def enabled(self) -> bool:
        return self.track_head is not None

    def forward(
        self,
        *,
        pooled: torch.Tensor,
        prev_state: torch.Tensor,
        active_dim: int,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        prev_feat = self.prev_state_encoder(prev_state)
        prev_feat = self.apply_width_mask(prev_feat, active_dim)
        fused = torch.cat([pooled, prev_feat], dim=-1)
        fused = self.apply_track_mask(fused, active_dim)
        outputs = {
            "track/fused": fused,
            "track/event_pooled": pooled,
            "track/prev_feat": prev_feat,
            "pruning/active_dim": torch.full((batch_size,), float(active_dim), device=device),
        }
        if self.track_head is not None:
            track_logits = self.track_head(fused)
            outputs["track/pupil"] = track_logits
            outputs["track/state"] = self.decode_state(prev_state, track_logits)
        return outputs
