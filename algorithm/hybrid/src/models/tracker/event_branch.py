from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


class EventStateBranch:
    def __init__(
        self,
        *,
        event_head: nn.Module | None,
        event_bbox_aux_head: nn.Module | None,
        event_obb_aux_head: nn.Module | None,
        aux_head: nn.Module | None,
        state_from_branch: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self.event_head = event_head
        self.event_bbox_aux_head = event_bbox_aux_head
        self.event_obb_aux_head = event_obb_aux_head
        self.aux_head = aux_head
        self.state_from_branch = state_from_branch

    @property
    def enabled(self) -> bool:
        return self.event_head is not None

    def forward(
        self,
        *,
        pooled: torch.Tensor,
        active_dim: int,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        if self.event_head is None:
            return {}
        event_logits = self.event_head(pooled)
        outputs = {
            "event/pupil": event_logits,
            "event/state": self.state_from_branch(event_logits),
            "event/pooled": pooled,
            "pruning/active_dim": torch.full((batch_size,), float(active_dim), device=device),
        }
        if self.event_bbox_aux_head is not None:
            outputs["event/pupil_bbox"] = self.event_bbox_aux_head(pooled)
        if self.event_obb_aux_head is not None:
            outputs["event/pupil_obb"] = self.event_obb_aux_head(pooled)
        if self.aux_head is not None:
            outputs["event/aux"] = self.aux_head(pooled)
        return outputs
