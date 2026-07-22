"""Shared head building blocks.

Small helpers reused by the pupil heads: a token-pooling op and a two-layer MLP
regression stem. Kept here so each head file stays one-concern.
"""
from __future__ import annotations

import torch
from torch import nn


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Global average over the token axis: ``[B, N, D] -> [B, D]``."""
    return tokens.mean(dim=1)


def mlp_head(in_dim: int, out_dim: int, *, hidden_dim: int | None = None) -> nn.Sequential:
    """Linear -> GELU -> Linear regression stem."""
    hidden = hidden_dim or in_dim
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def tokens_to_grid(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """``[B, N, D] -> [B, D, H, W]`` for dense (mask / heatmap) heads."""
    batch, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(f"token count {num_tokens} != {height}*{width}")
    return tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)
