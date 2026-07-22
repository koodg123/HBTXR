"""Pupil Box Head — the paper's search-path localization head (Eq 2).

The search path predicts an ORIENTED pupil bounding box (the "pupil anchor") in
search-anchor space::

    u_hat = (x_hat, y_hat, w_hat, h_hat, r_hat) = f_search(x^f_t)

(x, y) is the anchor center, (w, h) the box extents, r the box orientation. The
box is later decoded to an ellipse state by the box-to-state decoder g() (see
models.geometry.codec). This is the paper's default search head (NOT a heatmap).
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.common import mlp_head, pool_tokens

BOX_DIM = 5  # (x, y, w, h, r)


class PupilBoxHead(nn.Module):
    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.regressor = mlp_head(embed_dim, BOX_DIM, hidden_dim=hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens ``[B, N, D]`` -> oriented box ``[B, 5] = (x, y, w, h, r)``."""
        return self.regressor(pool_tokens(tokens))
