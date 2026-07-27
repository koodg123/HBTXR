"""Shared ViT backbone for HBTXR with early-exit at a cut point.

Paper Sec III-C / Fig 5: one transformer backbone ``B`` of depth ``L``. The
frame/search path traverses the full depth ``B_{1:L}``; the event/track path
exits early at cut point ``c < L`` (``B_{1:c}``), reusing the SAME early blocks:

    U^f_t = B_{1:L}(P_f(x^f_t))        # search / frame, full depth
    U^e_t = B_{1:c}(P_e(x^e_t)),  c<L  # track / event, early exit

Fig 5 depicts ``L = 8, c = 4`` (validation-selected). This module is a pure
Transformer-Block stack with a final norm.

**No positional embedding exists anywhere in this codebase** — not here, not in
the stems (``models/backbones/patch_embed.py``), not in any caller. An earlier
version of this docstring claimed the caller/stem applied one; it does not. The
frame and event grids have different token counts, so adding one would need a
per-modality table, and that is a modelling decision nobody has made yet rather
than a step that is silently happening elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from models.blocks import TransformerBlock


@dataclass(frozen=True)
class ViTConfig:
    embed_dim: int = 192
    depth: int = 8            # L: full (search) depth
    num_heads: int = 3
    mlp_ratio: float = 4.0
    cut_point: int = 4        # c < L: track early-exit depth
    drop: float = 0.0
    attn_drop: float = 0.0
    norm_eps: float = 1e-6


class ViTBackbone(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if not 0 < config.cut_point <= config.depth:
            raise ValueError(f"cut_point {config.cut_point} must be in (0, depth={config.depth}]")
        self.config = config
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.embed_dim,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    drop=config.drop,
                    attn_drop=config.attn_drop,
                    norm_eps=config.norm_eps,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim, eps=config.norm_eps)

    def forward(self, tokens: torch.Tensor, *, depth_limit: int | None = None) -> torch.Tensor:
        n = len(self.blocks) if depth_limit is None else max(0, min(len(self.blocks), int(depth_limit)))
        x = tokens
        for block in self.blocks[:n]:
            x = block(x)
        return self.norm(x)

    def forward_search(self, tokens: torch.Tensor) -> torch.Tensor:
        """Full-depth traversal (frame/search path, ``B_{1:L}``)."""
        return self.forward(tokens, depth_limit=None)

    def forward_track(self, tokens: torch.Tensor) -> torch.Tensor:
        """Early-exit traversal at the cut point (event/track path, ``B_{1:c}``)."""
        return self.forward(tokens, depth_limit=self.config.cut_point)
