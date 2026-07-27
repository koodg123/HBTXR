"""Transformer Block (TRB) — the repeating unit of the HBTXR ViT backbone.

Paper Fig 3 defines the block as two pre-norm residual sublayers:

    x = x + MHA(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

The shared backbone is a stack of L identical TRBs; the search path traverses all
L, the track path exits early at cut point c < L (see models.backbones.vit).
"""
from __future__ import annotations

import torch
from torch import nn

from models.blocks.attention import MultiHeadAttention
from models.blocks.mlp import Mlp
from models.blocks.seams import Add


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = MultiHeadAttention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        # The two residual joins as modules: this is where two differently-scaled
        # tensors meet, so it is exactly what an integer graph has to align — and an
        # inline `+` is invisible to a converter that swaps children. Parameter-free.
        self.attn_residual = Add()
        self.mlp_residual = Add()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_residual(x, self.attn(self.norm1(x)))
        x = self.mlp_residual(x, self.mlp(self.norm2(x)))
        return x
