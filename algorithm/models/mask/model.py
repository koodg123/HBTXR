"""Standalone pupil-mask segmentation model (UNet) — models.mask.

A separate task from the HBTXR ellipse-regression models (frame/event/hybrid):
dense pixel-wise pupil segmentation. This is a clean encoder-decoder UNet (the
reorganized DavisWithMask/UNet family), independent and independently trainable,
producing a 1-channel pupil-mask logit map at input resolution.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # pad to match skip spatial size if odd input sizes cause a mismatch
        dy = skip.shape[-2] - x.shape[-2]
        dx = skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([skip, x], dim=1))


@dataclass
class MaskModelConfig:
    in_chans: int = 1
    base_ch: int = 32
    num_classes: int = 1


class MaskModel(nn.Module):
    """UNet pupil-mask segmentation: image -> per-pixel mask logits."""

    def __init__(self, config: MaskModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or MaskModelConfig()
        self.config = cfg
        c = cfg.base_ch
        self.inc = DoubleConv(cfg.in_chans, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.up1 = Up(c * 8, c * 4, c * 4)
        self.up2 = Up(c * 4, c * 2, c * 2)
        self.up3 = Up(c * 2, c, c)
        self.outc = nn.Conv2d(c, cfg.num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(image)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)


def build_mask_model(config: MaskModelConfig | None = None) -> MaskModel:
    return MaskModel(config)
