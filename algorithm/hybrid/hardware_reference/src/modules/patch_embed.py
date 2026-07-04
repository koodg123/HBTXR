from __future__ import annotations

import torch
from torch import nn


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


class FramePatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        y = self.proj(frame)
        return flatten_tokens(y), (y.shape[-2], y.shape[-1])


class EventPatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Conv2d(2, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, event: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        y = self.proj(event)
        return flatten_tokens(y), (y.shape[-2], y.shape[-1])

