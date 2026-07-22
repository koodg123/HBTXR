from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


Mode = Literal["search", "track"]


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def tokens_to_grid(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    batch, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(f"Token count mismatch: got {num_tokens}, expected {height * width}")
    return tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)


def _grid_size_from_feature(feature: torch.Tensor) -> tuple[int, int]:
    return int(feature.shape[-2]), int(feature.shape[-1])


@dataclass
class PatchEmbedCache:
    frame_psum: torch.Tensor | None = None
    frame_shape: tuple[int, int] | None = None
    valid: bool = False


def invalid_patch_cache() -> PatchEmbedCache:
    return PatchEmbedCache()


class FramePatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, frame: torch.Tensor, *, return_psum: bool = False) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        feature = self.proj(frame)
        if return_psum:
            return feature
        return flatten_tokens(feature), _grid_size_from_feature(feature)


class EventPatchEmbeddingFACET(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(2, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, event: torch.Tensor, *, return_psum: bool = False) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        feature = self.proj(event)
        if return_psum:
            return feature
        return flatten_tokens(feature), _grid_size_from_feature(feature)


class PseudoEdgeGenerator(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def forward(self, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[1] != 1:
            raise ValueError(f"Expected frame [B,1,H,W], got {tuple(frame.shape)}")
        gx = F.conv2d(frame, self.sobel_x, padding=1)
        gy = F.conv2d(frame, self.sobel_y, padding=1)
        signed = self.alpha * gx + self.beta * gy
        pos = F.relu(signed)
        neg = F.relu(-signed)
        return torch.cat([pos, neg], dim=1)


class ModeAffine(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(2, embed_dim))
        self.beta = nn.Parameter(torch.zeros(2, embed_dim))

    def forward(self, x: torch.Tensor, mode: Mode) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,D,H,W], got {tuple(x.shape)}")
        mode_id = 0 if mode == "search" else 1
        gamma = self.gamma[mode_id].view(1, -1, 1, 1)
        beta = self.beta[mode_id].view(1, -1, 1, 1)
        return x * gamma + beta


class LegacyPatchFrontend(nn.Module):
    def __init__(self, *, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.variant = "legacy"
        self.supports_cache = False
        self.frame_embed = FramePatchEmbedding(embed_dim=embed_dim, patch_size=patch_size)
        self.event_embed = EventPatchEmbeddingFACET(embed_dim=embed_dim, patch_size=patch_size)

    def forward_search(
        self,
        frame: torch.Tensor,
        *,
        use_pseudo_edges: bool = False,
        pseudo_edges: torch.Tensor | None = None,
        return_psum: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        del use_pseudo_edges, pseudo_edges
        return self.frame_embed(frame, return_psum=return_psum)

    def forward_event(self, event: torch.Tensor, *, return_psum: bool = False) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        return self.event_embed(event, return_psum=return_psum)

    def forward_track(
        self,
        event: torch.Tensor,
        *,
        cache: PatchEmbedCache | None = None,
        cached_frame: torch.Tensor | None = None,
        event_only: bool = False,
        return_psum: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        del cache, cached_frame, event_only
        return self.event_embed(event, return_psum=return_psum)

    def prime_cache(self, cached_frame: torch.Tensor) -> PatchEmbedCache:
        del cached_frame
        return invalid_patch_cache()

    def invalidate_cache(self) -> PatchEmbedCache:
        return invalid_patch_cache()


class Split1PatchFrontend(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        patch_size: int = 16,
        bias: bool = True,
        flatten_output: bool = True,
        use_mode_affine: bool = True,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.variant = "split1"
        self.supports_cache = True
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.flatten_output = bool(flatten_output)
        self.proj = nn.Conv2d(3, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=bias)
        self.mode_affine = ModeAffine(self.embed_dim) if use_mode_affine else nn.Identity()
        self.edge_generator = PseudoEdgeGenerator(alpha=alpha, beta=beta)

    @property
    def weight_frame(self) -> torch.Tensor:
        return self.proj.weight[:, 0:1, :, :]

    @property
    def weight_event(self) -> torch.Tensor:
        return self.proj.weight[:, 1:3, :, :]

    @property
    def bias_term(self) -> torch.Tensor | None:
        return self.proj.bias

    def _conv_frame_only(self, frame: torch.Tensor, *, add_bias: bool) -> torch.Tensor:
        return F.conv2d(frame, self.weight_frame, bias=self.bias_term if add_bias else None, stride=self.patch_size, padding=0)

    def _conv_event_only(self, event: torch.Tensor, *, add_bias: bool) -> torch.Tensor:
        return F.conv2d(event, self.weight_event, bias=self.bias_term if add_bias else None, stride=self.patch_size, padding=0)

    def _to_tokens(self, psum: torch.Tensor, *, mode: Mode) -> torch.Tensor:
        psum = self.mode_affine(psum, mode) if isinstance(self.mode_affine, ModeAffine) else psum
        if not self.flatten_output:
            return psum
        return flatten_tokens(psum)

    def build_search_3ch(
        self,
        frame: torch.Tensor,
        *,
        use_pseudo_edges: bool = False,
        pseudo_edges: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame.ndim != 4 or frame.shape[1] != 1:
            raise ValueError(f"Expected frame [B,1,H,W], got {tuple(frame.shape)}")
        if use_pseudo_edges:
            pseudo = self.edge_generator(frame) if pseudo_edges is None else pseudo_edges
            if pseudo.ndim != 4 or pseudo.shape[1] != 2:
                raise ValueError(f"Expected pseudo edges [B,2,H,W], got {tuple(pseudo.shape)}")
            return torch.cat([frame, pseudo], dim=1)
        zeros = frame.new_zeros(frame.shape[0], 2, frame.shape[2], frame.shape[3])
        return torch.cat([frame, zeros], dim=1)

    def build_track_3ch(self, cached_frame: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
        if cached_frame.ndim != 4 or cached_frame.shape[1] != 1:
            raise ValueError(f"Expected cached_frame [B,1,H,W], got {tuple(cached_frame.shape)}")
        if event.ndim != 4 or event.shape[1] != 2:
            raise ValueError(f"Expected event [B,2,H,W], got {tuple(event.shape)}")
        return torch.cat([cached_frame, event], dim=1)

    @torch.no_grad()
    def prime_cache(self, cached_frame: torch.Tensor) -> PatchEmbedCache:
        if cached_frame.ndim != 4 or cached_frame.shape[1] != 1:
            raise ValueError(f"Expected cached_frame [B,1,H,W], got {tuple(cached_frame.shape)}")
        psum = self._conv_frame_only(cached_frame, add_bias=False)
        return PatchEmbedCache(frame_psum=psum, frame_shape=(cached_frame.shape[-2], cached_frame.shape[-1]), valid=True)

    def invalidate_cache(self) -> PatchEmbedCache:
        return invalid_patch_cache()

    def forward_search(
        self,
        frame: torch.Tensor,
        *,
        use_pseudo_edges: bool = False,
        pseudo_edges: torch.Tensor | None = None,
        return_psum: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        if use_pseudo_edges:
            x3 = self.build_search_3ch(frame, use_pseudo_edges=True, pseudo_edges=pseudo_edges)
            psum = self.proj(x3)
        else:
            psum = self._conv_frame_only(frame, add_bias=True)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode="search"), _grid_size_from_feature(psum)

    def forward_event(self, event: torch.Tensor, *, return_psum: bool = False) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        psum = self._conv_event_only(event, add_bias=True)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode="track"), _grid_size_from_feature(psum)

    def forward_track(
        self,
        event: torch.Tensor,
        *,
        cache: PatchEmbedCache | None = None,
        cached_frame: torch.Tensor | None = None,
        event_only: bool = False,
        return_psum: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        if event.ndim != 4 or event.shape[1] != 2:
            raise ValueError(f"Expected event [B,2,H,W], got {tuple(event.shape)}")
        event_psum = self._conv_event_only(event, add_bias=False)
        if event_only:
            psum = event_psum
            if self.bias_term is not None:
                psum = psum + self.bias_term.view(1, -1, 1, 1)
            if return_psum:
                return psum
            return self._to_tokens(psum, mode="track"), _grid_size_from_feature(psum)
        if cache is None or not cache.valid or cache.frame_psum is None:
            if cached_frame is None:
                raise ValueError("split1 track path requires a valid cache or cached_frame")
            cache = self.prime_cache(cached_frame)
        psum = cache.frame_psum + event_psum
        if self.bias_term is not None:
            psum = psum + self.bias_term.view(1, -1, 1, 1)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode="track"), _grid_size_from_feature(psum)

    def forward_dense_reference(self, x3: torch.Tensor, *, mode: Mode, return_psum: bool = False) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        if x3.ndim != 4 or x3.shape[1] != 3:
            raise ValueError(f"Expected x3 [B,3,H,W], got {tuple(x3.shape)}")
        psum = self.proj(x3)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode=mode), _grid_size_from_feature(psum)


class Split2PatchFrontend(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        patch_size: int = 16,
        search_bias: bool = True,
        event_bias: bool = False,
        flatten_output: bool = True,
        use_mode_affine: bool = True,
    ) -> None:
        super().__init__()
        self.variant = "split2"
        self.supports_cache = True
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.flatten_output = bool(flatten_output)
        self.frame_proj = nn.Conv2d(1, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=search_bias)
        self.event_proj = nn.Conv2d(2, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=event_bias)
        self.track_bias = nn.Parameter(torch.zeros(self.embed_dim))
        self.mode_affine = ModeAffine(self.embed_dim) if use_mode_affine else nn.Identity()

    def _to_tokens(self, psum: torch.Tensor, *, mode: Mode) -> torch.Tensor:
        psum = self.mode_affine(psum, mode) if isinstance(self.mode_affine, ModeAffine) else psum
        if not self.flatten_output:
            return psum
        return flatten_tokens(psum)

    @torch.no_grad()
    def prime_cache(self, cached_frame: torch.Tensor) -> PatchEmbedCache:
        if cached_frame.ndim != 4 or cached_frame.shape[1] != 1:
            raise ValueError(f"Expected cached_frame [B,1,H,W], got {tuple(cached_frame.shape)}")
        psum = F.conv2d(cached_frame, self.frame_proj.weight, bias=None, stride=self.patch_size, padding=0)
        return PatchEmbedCache(frame_psum=psum, frame_shape=(cached_frame.shape[-2], cached_frame.shape[-1]), valid=True)

    def invalidate_cache(self) -> PatchEmbedCache:
        return invalid_patch_cache()

    def forward_search(
        self,
        frame: torch.Tensor,
        *,
        use_pseudo_edges: bool = False,
        pseudo_edges: torch.Tensor | None = None,
        return_psum: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        del use_pseudo_edges, pseudo_edges
        psum = self.frame_proj(frame)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode="search"), _grid_size_from_feature(psum)

    def forward_event(self, event: torch.Tensor, *, return_psum: bool = False) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        psum = self.event_proj(event) + self.track_bias.view(1, -1, 1, 1)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode="track"), _grid_size_from_feature(psum)

    def forward_track(
        self,
        event: torch.Tensor,
        *,
        cache: PatchEmbedCache | None = None,
        cached_frame: torch.Tensor | None = None,
        event_only: bool = False,
        return_psum: bool = False,
    ) -> tuple[torch.Tensor, tuple[int, int]] | torch.Tensor:
        if event.ndim != 4 or event.shape[1] != 2:
            raise ValueError(f"Expected event [B,2,H,W], got {tuple(event.shape)}")
        event_psum = self.event_proj(event)
        if event_only:
            psum = event_psum + self.track_bias.view(1, -1, 1, 1)
            if return_psum:
                return psum
            return self._to_tokens(psum, mode="track"), _grid_size_from_feature(psum)
        if cache is None or not cache.valid or cache.frame_psum is None:
            if cached_frame is None:
                raise ValueError("split2 track path requires a valid cache or cached_frame")
            cache = self.prime_cache(cached_frame)
        psum = cache.frame_psum + event_psum + self.track_bias.view(1, -1, 1, 1)
        if return_psum:
            return psum
        return self._to_tokens(psum, mode="track"), _grid_size_from_feature(psum)


def build_patch_frontend(
    *,
    variant: str = "legacy",
    embed_dim: int = 192,
    patch_size: int = 16,
    use_mode_affine: bool = True,
    flatten_tokens: bool = True,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> nn.Module:
    normalized = str(variant or "legacy").strip().lower()
    if normalized == "legacy":
        return LegacyPatchFrontend(embed_dim=embed_dim, patch_size=patch_size)
    if normalized == "split1":
        return Split1PatchFrontend(
            embed_dim=embed_dim,
            patch_size=patch_size,
            flatten_output=flatten_tokens,
            use_mode_affine=use_mode_affine,
            alpha=alpha,
            beta=beta,
        )
    if normalized == "split2":
        return Split2PatchFrontend(
            embed_dim=embed_dim,
            patch_size=patch_size,
            flatten_output=flatten_tokens,
            use_mode_affine=use_mode_affine,
        )
    raise ValueError(f"Unsupported patch-embed variant: {variant}")


__all__ = [
    "EventPatchEmbeddingFACET",
    "FramePatchEmbedding",
    "LegacyPatchFrontend",
    "ModeAffine",
    "PatchEmbedCache",
    "PseudoEdgeGenerator",
    "Split1PatchFrontend",
    "Split2PatchFrontend",
    "build_patch_frontend",
    "flatten_tokens",
    "invalid_patch_cache",
    "tokens_to_grid",
]
