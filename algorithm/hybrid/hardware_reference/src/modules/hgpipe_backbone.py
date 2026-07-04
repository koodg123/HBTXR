from __future__ import annotations

import math

import torch
from torch import nn


class AttentionStage(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(self.norm(x)).view(b, n, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        y = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return x + self.drop(self.out(y))


class MLPStage(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class HGPipeBackbone(nn.Module):
    def __init__(self, embed_dim: int = 192, depth: int = 6, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(AttentionStage(embed_dim, num_heads=num_heads, dropout=dropout))
            self.blocks.append(MLPStage(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, *, depth_limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = tokens
        limit = len(self.blocks) if depth_limit is None else max(0, min(len(self.blocks), int(depth_limit) * 2))
        for block in self.blocks[:limit]:
            x = block(x)
        x = self.norm(x)
        return x, x.mean(dim=1)

