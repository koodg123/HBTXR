from __future__ import annotations

import torch
from torch import nn


class StateFusion(nn.Module):
    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.prev_encoder = nn.Sequential(nn.LayerNorm(6), nn.Linear(6, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        self.fuse = nn.Sequential(nn.LayerNorm(embed_dim * 3), nn.Linear(embed_dim * 3, embed_dim * 2), nn.GELU())

    def forward(self, frame_pooled: torch.Tensor, event_pooled: torch.Tensor, prev_state: torch.Tensor) -> torch.Tensor:
        prev = self.prev_encoder(prev_state)
        return self.fuse(torch.cat([frame_pooled, event_pooled, prev], dim=-1))

