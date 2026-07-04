from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def mlp_head(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, max(in_dim, out_dim)), nn.GELU(), nn.Linear(max(in_dim, out_dim), out_dim))


def normalize_state(branch_logits: torch.Tensor) -> torch.Tensor:
    state = branch_logits[..., :6]
    uv = F.normalize(state[..., 4:6], dim=-1, eps=1e-6)
    return torch.cat([state[..., :4], uv], dim=-1)


class SearchHead(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = mlp_head(embed_dim, 7)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class TrackHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = mlp_head(in_dim, 8)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class MaskHead(nn.Module):
    def __init__(self, embed_dim: int, output_size: tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size
        self.net = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1), nn.GELU(), nn.Conv2d(embed_dim // 2, 1, 1))

    def forward(self, tokens: torch.Tensor, grid_size: tuple[int, int]) -> torch.Tensor:
        b, n, c = tokens.shape
        if n != grid_size[0] * grid_size[1]:
            raise ValueError("Token/grid mismatch")
        x = tokens.transpose(1, 2).contiguous().view(b, c, grid_size[0], grid_size[1])
        return F.interpolate(self.net(x), size=self.output_size, mode="bilinear", align_corners=False)

