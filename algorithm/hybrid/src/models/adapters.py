from __future__ import annotations

import torch
from torch import nn


class ModalityAdapter(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else int(embed_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)
