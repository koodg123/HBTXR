"""Center-heatmap head (ALTERNATIVE to the paper's box/ellipse regression heads).

The paper localizes the pupil by direct oriented-box / ellipse-residual
regression; a center heatmap is NOT used. This head is retained as a swappable
alternative (config ``head.search: heatmap``) for ablation against the parked
heatmap implementation. Predicts a dense center-likelihood map; the center is
recovered by soft-argmax at decode time.
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.common import tokens_to_grid


class CenterHeatmapHead(nn.Module):
    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None, upsample: int = 16) -> None:
        super().__init__()
        hidden = hidden_dim or embed_dim // 2
        self.proj = nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.to_heat = nn.Conv2d(hidden, 1, kernel_size=1)
        self.upsample = int(upsample)

    def forward(self, tokens: torch.Tensor, *, grid_hw: tuple[int, int]) -> torch.Tensor:
        """tokens ``[B, N, D]`` -> center-heatmap logits ``[B, 1, H*up, W*up]``."""
        height, width = grid_hw
        grid = tokens_to_grid(tokens, height=height, width=width)
        heat = self.to_heat(self.act(self.proj(grid)))
        if self.upsample > 1:
            heat = nn.functional.interpolate(
                heat, scale_factor=self.upsample, mode="bilinear", align_corners=False
            )
        return heat
