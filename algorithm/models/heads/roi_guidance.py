"""Eye-region ROI guidance head g_eye (paper Eq 15).

    b_hat^eye_t = g_eye(GAP(U^f_t))

A coarse eye-region bounding box used only as a geometric cue for the frame-side
search branch (not a final output). Predicts an axis-aligned box (x, y, w, h).
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.common import mlp_head, pool_tokens

EYE_BOX_DIM = 4  # (x, y, w, h)


class EyeRegionHead(nn.Module):
    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.regressor = mlp_head(embed_dim, EYE_BOX_DIM, hidden_dim=hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens ``[B, N, D]`` -> coarse eye box ``[B, 4] = (x, y, w, h)``."""
        return self.regressor(pool_tokens(tokens))
