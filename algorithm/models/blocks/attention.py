"""Multi-head self-attention (MHA) — the attention sublayer of the ViT TRB.

Standard scaled dot-product multi-head attention (paper Fig 3): project tokens
to Q/K/V, compute ``softmax(Q Kᵀ / √d) · V`` per head, concatenate heads, and
project out. This is the SMU/RMU workload the HBTXR accelerator maps to hardware
(Q/K/V and output projection on the RMU, Q×Kᵀ and S×V on the SMU).
"""
from __future__ import annotations

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = (attn @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(out))
