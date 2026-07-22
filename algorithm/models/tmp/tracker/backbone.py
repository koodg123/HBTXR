from __future__ import annotations

import torch
from torch import nn

from .blocks import HGPipeAttentionStage, HGPipeMLPStage


class PartialDeiTTiny(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        mlp_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.attn_stages = nn.ModuleList([HGPipeAttentionStage(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)])
        self.mlp_stages = nn.ModuleList(
            [
                HGPipeMLPStage(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    hidden_dim=mlp_hidden_dim,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, *, depth_limit: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = tokens
        stage_count = len(self.attn_stages) if depth_limit is None else max(0, min(len(self.attn_stages), int(depth_limit)))
        for attn_stage, mlp_stage in zip(self.attn_stages[:stage_count], self.mlp_stages[:stage_count]):
            x = attn_stage(x)
            x = mlp_stage(x)
        x = self.norm(x)
        return x, x.mean(dim=1)
