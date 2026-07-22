"""models.backbones — shared ViT backbone + modality patch-embedding stems.

- ViTBackbone / ViTConfig: the shared transformer backbone with early-exit at a
  cut point (full depth for search, ``B_{1:c}`` for track).
- FramePatchEmbed (Conv-F) / EventPatchEmbed (Conv-E): modality-specific stems.
"""
from models.backbones.patch_embed import (
    EventPatchEmbed,
    FramePatchEmbed,
    build_patch_embed,
    flatten_tokens,
    tokens_to_grid,
)
from models.backbones.vit import ViTBackbone, ViTConfig

__all__ = [
    "ViTBackbone",
    "ViTConfig",
    "FramePatchEmbed",
    "EventPatchEmbed",
    "build_patch_embed",
    "flatten_tokens",
    "tokens_to_grid",
]
