from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import compute_open_extent_from_binary_mask


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None) -> None:
        super().__init__()
        mid = out_channels if mid_channels is None else mid_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class _OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AntiBlinkUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2, bilinear: bool = False) -> None:
        super().__init__()
        factor = 2 if bilinear else 1
        self.inc = _DoubleConv(in_channels, 64)
        self.down1 = _Down(64, 128)
        self.down2 = _Down(128, 256)
        self.down3 = _Down(256, 512)
        self.down4 = _Down(512, 1024 // factor)
        self.up1 = _Up(1024, 512 // factor, bilinear)
        self.up2 = _Up(512, 256 // factor, bilinear)
        self.up3 = _Up(256, 128 // factor, bilinear)
        self.up4 = _Up(128, 64, bilinear)
        self.outc = _OutConv(64, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


@dataclass
class AntiBlinkConfig:
    closed_threshold: float = 0.2
    hold_threshold: float = 0.35
    detection_threshold: float = 0.75
    template_update_threshold: float = 0.95


class AntiBlinkDetector(nn.Module):
    def __init__(self, model: AntiBlinkUNet | None = None, *, config: AntiBlinkConfig | None = None) -> None:
        super().__init__()
        self.model = AntiBlinkUNet() if model is None else model
        self.config = AntiBlinkConfig() if config is None else config

    def predict_mask_logits(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[1] != 1:
            raise ValueError("AntiBlinkDetector expects frame [B,1,H,W]")
        return self.model(frame)

    @torch.no_grad()
    def forward(self, frame: torch.Tensor, ellipse_xywht: torch.Tensor) -> dict[str, Any]:
        logits = self.predict_mask_logits(frame)
        probs = torch.softmax(logits, dim=1)[:, 1]
        masks = (probs >= 0.5).to(dtype=torch.uint8)
        extents = []
        for idx in range(frame.shape[0]):
            extents.append(
                compute_open_extent_from_binary_mask(
                    masks[idx].detach().cpu().numpy(),
                    ellipse_xywht[idx].detach().cpu().numpy(),
                )
            )
        open_extent = torch.tensor(extents, dtype=torch.float32, device=frame.device)
        closed_eye_flag = (open_extent < float(self.config.closed_threshold)).to(dtype=torch.float32)
        should_hold = open_extent < float(self.config.hold_threshold)
        return {
            "mask_logits": logits,
            "mask_probability": probs,
            "mask_binary": masks,
            "open_extent": open_extent,
            "closed_eye_flag": closed_eye_flag,
            "should_hold": should_hold,
        }
