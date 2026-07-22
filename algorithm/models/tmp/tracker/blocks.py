from __future__ import annotations

import math

import torch
from torch import nn


class HGPipeAttentionStage(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, h * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = self.dropout(torch.softmax(scores, dim=-1))
        attn = torch.matmul(probs, v)
        attn = self._merge_heads(attn)
        return residual + self.dropout(self.out_proj(attn))


class HGPipeMLPStage(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        hidden_dim: int | None = None,
    ) -> None:
        del num_heads
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)
