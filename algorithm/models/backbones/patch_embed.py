"""Modality-specific patch-embedding stems (paper Sec IV-B.1).

The frame/search path uses **Conv-F** (single grayscale channel); the event/track
path uses **Conv-E** (two-channel event voxel, +/- polarity). Each is a strided
conv whose stride equals the patch size, followed by a flatten to token-major
order. Both project into the SAME embedding dim so the one shared backbone can
consume either stream ("modality-specific patch-embedding stems ... projected
into one shared transformer backbone", Sec III-C).

This is the plain default (the parked reference's ``legacy`` frontend). The
cached / split / mode-affine variants in tmp are hardware-efficiency options and
are not needed by the software model.
"""
from __future__ import annotations

import torch
from torch import nn


def flatten_tokens(feature: torch.Tensor) -> torch.Tensor:
    """``[B, D, H, W] -> [B, H*W, D]`` (token-major)."""
    return feature.flatten(2).transpose(1, 2).contiguous()


def tokens_to_grid(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """``[B, N, D] -> [B, D, H, W]`` (inverse of :func:`flatten_tokens`)."""
    batch, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(f"token count {num_tokens} != {height}*{width}")
    return tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)


class FramePatchEmbed(nn.Module):
    """Conv-F: grayscale frame ``[B, 1, H, W]`` -> tokens ``[B, N, D]``."""

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, in_chans: int = 1) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feat = self.proj(frame)
        return flatten_tokens(feat), (int(feat.shape[-2]), int(feat.shape[-1]))


class EventPatchEmbed(nn.Module):
    """Conv-E: two-channel event voxel ``[B, 2, H, W]`` -> tokens ``[B, N, D]``."""

    def __init__(self, embed_dim: int = 192, patch_size: int = 16, in_chans: int = 2) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, event: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feat = self.proj(event)
        return flatten_tokens(feat), (int(feat.shape[-2]), int(feat.shape[-1]))


def build_patch_embed(
    modality: str,
    *,
    embed_dim: int = 192,
    patch_size: int = 16,
    in_chans: int | None = None,
) -> nn.Module:
    """Factory: ``modality`` in {frame/search, event/track} -> the matching stem."""
    m = str(modality).strip().lower()
    if m in ("frame", "search"):
        return FramePatchEmbed(embed_dim=embed_dim, patch_size=patch_size, in_chans=in_chans or 1)
    if m in ("event", "track"):
        return EventPatchEmbed(embed_dim=embed_dim, patch_size=patch_size, in_chans=in_chans or 2)
    raise ValueError(f"unknown modality for patch embed: {modality!r}")
