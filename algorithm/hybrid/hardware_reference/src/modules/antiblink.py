from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiBlinkUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 2) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = F.interpolate(self.enc2(x1), size=x1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec(torch.cat([x1, x2], dim=1))


@dataclass
class AntiBlinkConfig:
    closed_threshold: float = 0.2
    hold_threshold: float = 0.35


class AntiBlinkDetector(nn.Module):
    def __init__(self, model: AntiBlinkUNet | None = None, *, config: AntiBlinkConfig | None = None) -> None:
        super().__init__()
        self.model = AntiBlinkUNet() if model is None else model
        self.config = AntiBlinkConfig() if config is None else config

    @torch.no_grad()
    def forward(self, frame: torch.Tensor, ellipse_state: torch.Tensor) -> dict[str, torch.Tensor]:
        del ellipse_state
        logits = self.model(frame)
        prob = torch.softmax(logits, dim=1)[:, 1]
        open_extent = prob.mean(dim=(1, 2))
        return {
            "mask_logits": logits,
            "open_extent": open_extent,
            "closed_eye_flag": (open_extent < self.config.closed_threshold).float(),
            "should_hold": open_extent < self.config.hold_threshold,
        }

