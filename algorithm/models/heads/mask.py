"""Auxiliary Pupil Mask Head (paper Sec III-D.1, Fig 3 "Mask Head (Aux)").

Predicts a dense binary pupil mask from the backbone token grid. Used only as
stage-1 auxiliary supervision (segmentation), NOT as the runtime localization
output. Upsamples the coarse token grid back toward input resolution.
"""
from __future__ import annotations

import torch
from torch import nn

from models.heads.common import tokens_to_grid


class PupilMaskHead(nn.Module):
    def __init__(self, embed_dim: int, *, hidden_dim: int | None = None, upsample: int = 16) -> None:
        super().__init__()
        hidden = hidden_dim or embed_dim // 2
        self.proj = nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.to_logits = nn.Conv2d(hidden, 1, kernel_size=1)
        self.upsample = int(upsample)

    def forward(self, tokens: torch.Tensor, *, grid_hw: tuple[int, int]) -> torch.Tensor:
        """tokens ``[B, N, D]`` -> mask logits ``[B, 1, H*up, W*up]``."""
        height, width = grid_hw
        grid = tokens_to_grid(tokens, height=height, width=width)
        x = self.act(self.proj(grid))
        logits = self.to_logits(x)
        if self.upsample > 1:
            logits = nn.functional.interpolate(
                logits, scale_factor=self.upsample, mode="bilinear", align_corners=False
            )
        return logits
