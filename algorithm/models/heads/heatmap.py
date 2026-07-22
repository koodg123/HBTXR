"""Center-heatmap heads (ALTERNATIVE to the paper's box/ellipse regression).

The paper localizes the pupil by direct oriented-box (search) / ellipse-residual
(track) regression; a center heatmap is NOT used there. These heads are retained
as swappable alternatives (config ``head: heatmap*``) for ablation against the
parked heatmap / CenterNet implementation. Both operate on the ViT token grid.

- SingleCenterHeatmapHead: a minimal single-channel center-likelihood map; the
  center is recovered by soft-argmax at decode time.
- MultiCenterHeatmapHead: reproduces the parked HBTXR/EPNet head
  (tmp/heads/hbtxr_head.py == ep_head.py) — a CenterNet-style multi-head that is a
  full heatmap-based ELLIPSE detector:
      hm   (1)  center heatmap
      ab   (2)  ellipse semi-axes
      trig (2)  orientation as (cos, sin)
      reg  (2)  sub-pixel center offset
      mask (1)  pupil mask
  Each branch is Conv3x3 -> ReLU -> Conv1x1; the hm branch uses the CenterNet
  focal-loss bias init (-2.19).
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.heads.common import tokens_to_grid

# CenterNet-style multi-head layout (from the parked HBTXR/EPNet head).
DEFAULT_HEAD_DICT: dict[str, int] = {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1}


class SingleCenterHeatmapHead(nn.Module):
    """Minimal single-channel center-likelihood map."""

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
            heat = F.interpolate(heat, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        return heat


class MultiCenterHeatmapHead(nn.Module):
    """CenterNet-style multi-head: hm + ab + trig + reg + mask (parked HBTXR head)."""

    def __init__(
        self,
        embed_dim: int,
        *,
        head_conv: int = 256,
        head_dict: dict[str, int] | None = None,
        upsample: int = 1,
    ) -> None:
        super().__init__()
        self.heads = dict(head_dict or DEFAULT_HEAD_DICT)
        self.upsample = int(upsample)
        self.branches = nn.ModuleDict()
        for name, out_ch in self.heads.items():
            branch = nn.Sequential(
                nn.Conv2d(embed_dim, head_conv, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, out_ch, kernel_size=1, bias=True),
            )
            if name == "hm":
                nn.init.constant_(branch[-1].bias, -2.19)  # CenterNet focal-loss init
            else:
                nn.init.zeros_(branch[-1].bias)
            self.branches[name] = branch

    def forward(self, tokens: torch.Tensor, *, grid_hw: tuple[int, int]) -> dict[str, torch.Tensor]:
        """tokens ``[B, N, D]`` -> ``{hm, ab, trig, reg, mask}`` maps ``[B, c, H*up, W*up]``."""
        height, width = grid_hw
        grid = tokens_to_grid(tokens, height=height, width=width)
        out = {name: branch(grid) for name, branch in self.branches.items()}
        if self.upsample > 1:
            out = {
                name: F.interpolate(value, scale_factor=self.upsample, mode="bilinear", align_corners=False)
                for name, value in out.items()
            }
        return out
