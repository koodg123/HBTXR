"""Reliability heads (paper Eq 17-18) — confidence + IoU-quality per branch.

Each branch emits a pair used by the runtime scheduler::

    (c^s_t, q^s_t) = h_search(U^f_t, ...)     # search branch
    (c^e_t, q^e_t) = h_track(U^e_t, ...)      # track branch

``c in [0, 1]`` is confidence, ``q in [0, 1]`` is IoU quality in the common
geometric state space. The scheduler combines them into ``rho = w_c*c + w_q*q``
(models.hybrid.scheduler).
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.common import mlp_head, pool_tokens


class ReliabilityHead(nn.Module):
    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.estimator = mlp_head(embed_dim, 2, hidden_dim=hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens ``[B, N, D]`` -> ``[B, 2] = (confidence, iou_quality)`` in [0, 1]."""
        return torch.sigmoid(self.estimator(pool_tokens(tokens)))
