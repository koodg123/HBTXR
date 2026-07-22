"""Pupil Ellipse Head — the paper's track-path residual head (Eq 7).

The track path predicts an anchor-RELATIVE pupil-state residual::

    d_s_hat = (dx_hat, dy_hat, da_hat, db_hat, dtheta_hat) = f_track(x^e_t, z_t)

relative to the decoded anchor state ``z_t``. The candidate state is then
``s_cand = Pi(z_t + d_s_hat)`` (re-canonicalized; see models.geometry). Optionally
conditioned on the anchor state ``z`` so the residual is aware of its reference.
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.common import mlp_head, pool_tokens

STATE_DIM = 5  # (x, y, a, b, theta) residual


class PupilEllipseHead(nn.Module):
    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None, condition_on_state: bool = True) -> None:
        super().__init__()
        self.condition_on_state = bool(condition_on_state)
        in_dim = embed_dim + (STATE_DIM if self.condition_on_state else 0)
        self.regressor = mlp_head(in_dim, STATE_DIM, hidden_dim=hidden_dim)

    def forward(self, tokens: torch.Tensor, anchor_state: torch.Tensor | None = None) -> torch.Tensor:
        """tokens ``[B, N, D]`` (+ optional anchor_state ``[B, 5]``) -> residual ``[B, 5]``."""
        feat = pool_tokens(tokens)
        if self.condition_on_state:
            if anchor_state is None:
                anchor_state = feat.new_zeros(feat.shape[0], STATE_DIM)
            feat = torch.cat([feat, anchor_state], dim=-1)
        return self.regressor(feat)
