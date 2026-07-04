from __future__ import annotations

import random

import torch
from torch import nn

from src.models.patch_embeddings import PatchEmbedCache, invalid_patch_cache


class TrackerTokenEncoder:
    def __init__(
        self,
        *,
        embed_dim: int,
        structural_width_ratio: float,
        pruning_cfg: dict[str, object],
        patch_embed_cfg: dict[str, object],
        patch_frontend: nn.Module,
        frame_adapter: nn.Module,
        event_adapter: nn.Module,
        backbone: nn.Module,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.structural_width_ratio = float(structural_width_ratio)
        self.pruning_cfg = dict(pruning_cfg or {})
        self.patch_embed_cfg = dict(patch_embed_cfg or {})
        self.patch_frontend = patch_frontend
        self.frame_adapter = frame_adapter
        self.event_adapter = event_adapter
        self.backbone = backbone

        pruning_mode = str(
            self.pruning_cfg.get("scheme")
            or self.pruning_cfg.get("mode")
            or self.pruning_cfg.get("method")
            or ""
        ).strip().lower()
        default_legacy_mode = not pruning_mode and (
            "width_candidates" in self.pruning_cfg
            or "student_width" in self.pruning_cfg
            or "sample_strategy" in self.pruning_cfg
        )
        self.legacy_width_masking = bool(self.pruning_cfg.get("enabled", False)) and pruning_mode in {
            "implicit_width_slicing",
            "implicit_masking_legacy",
            "legacy_masking",
        }
        self.legacy_width_masking = self.legacy_width_masking or (
            bool(self.pruning_cfg.get("enabled", False)) and default_legacy_mode
        )
        self.patch_embed_variant = str(self.patch_embed_cfg.get("variant", "legacy")).strip().lower()
        self.patch_warmup_epochs = int(self.patch_embed_cfg.get("warmup_epochs", 0))
        self.patch_search_use_pseudo_edges = bool(self.patch_embed_cfg.get("search_use_pseudo_edges", False))
        self.current_epoch = 0

    @staticmethod
    def collapse_patch_weight(weight: torch.Tensor) -> torch.Tensor:
        if weight.ndim != 4:
            return weight
        if weight.shape[1] == 1:
            return weight
        return weight.mean(dim=1, keepdim=True)

    def set_epoch_context(self, epoch: int | None) -> None:
        self.current_epoch = 0 if epoch is None else int(epoch)

    def use_search_pseudo_edges(self, *, training: bool) -> bool:
        if self.patch_embed_variant != "split1":
            return False
        if not self.patch_search_use_pseudo_edges:
            return False
        if self.patch_warmup_epochs <= 0:
            return False
        return training and 1 <= self.current_epoch <= self.patch_warmup_epochs

    def prime_patch_cache(self, frame: torch.Tensor) -> PatchEmbedCache:
        prime_cache = getattr(self.patch_frontend, "prime_cache", None)
        if callable(prime_cache):
            return prime_cache(frame)
        return invalid_patch_cache()

    def invalidate_patch_cache(self) -> PatchEmbedCache:
        invalidate_cache = getattr(self.patch_frontend, "invalidate_cache", None)
        if callable(invalidate_cache):
            return invalidate_cache()
        return invalid_patch_cache()

    def resolve_width_ratio(self, *, training: bool, requested: float | None = None) -> float:
        if not self.legacy_width_masking:
            return self.structural_width_ratio
        if requested is not None:
            return float(max(1.0 / max(1, self.embed_dim), min(1.0, requested)))
        if not bool(self.pruning_cfg.get("enabled", False)):
            return self.structural_width_ratio
        if training:
            candidates = [float(v) for v in self.pruning_cfg.get("width_candidates", [1.0]) if float(v) > 0.0]
            if not candidates:
                return self.structural_width_ratio
            strategy = str(self.pruning_cfg.get("sample_strategy", "random")).strip().lower()
            if strategy == "min":
                return float(min(candidates))
            if strategy == "max":
                return float(max(candidates))
            return float(random.choice(candidates))
        return float(max(1.0 / max(1, self.embed_dim), min(1.0, self.pruning_cfg.get("student_width", 1.0))))

    def active_dim(self, width_ratio: float) -> int:
        if not self.legacy_width_masking:
            return self.embed_dim
        return max(1, min(self.embed_dim, int(round(self.embed_dim * float(width_ratio)))))

    @staticmethod
    def apply_width_mask(tensor: torch.Tensor, active_dim: int) -> torch.Tensor:
        if tensor.shape[-1] <= active_dim:
            return tensor
        mask = tensor.new_zeros(tensor.shape[-1])
        mask[:active_dim] = 1.0
        return tensor * mask.view(*([1] * (tensor.ndim - 1)), -1)

    def apply_track_mask(self, fused: torch.Tensor, active_dim: int) -> torch.Tensor:
        if fused.shape[-1] <= active_dim * 2:
            return fused
        mask = fused.new_zeros(fused.shape[-1])
        mask[:active_dim] = 1.0
        mask[self.embed_dim : self.embed_dim + active_dim] = 1.0
        return fused * mask.view(*([1] * (fused.ndim - 1)), -1)

    def encode_tokens(
        self,
        tokens: torch.Tensor,
        *,
        adapter: nn.Module,
        training: bool,
        width_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        resolved_width = self.resolve_width_ratio(training=training, requested=width_ratio)
        active_dim = self.active_dim(resolved_width)
        tokens = adapter(tokens)
        tokens = self.apply_width_mask(tokens, active_dim)
        tokens, pooled = self.backbone(tokens)
        tokens = self.apply_width_mask(tokens, active_dim)
        pooled = self.apply_width_mask(pooled, active_dim)
        return tokens, pooled, active_dim

    def encode_frame(
        self,
        frame: torch.Tensor,
        *,
        training: bool,
        width_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], int]:
        tokens, grid_size = self.patch_frontend.forward_search(
            frame,
            use_pseudo_edges=self.use_search_pseudo_edges(training=training),
        )
        tokens, pooled, active_dim = self.encode_tokens(
            tokens,
            adapter=self.frame_adapter,
            training=training,
            width_ratio=width_ratio,
        )
        return tokens, pooled, grid_size, active_dim

    def encode_event(
        self,
        event: torch.Tensor,
        *,
        training: bool,
        width_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], int]:
        tokens, grid_size = self.patch_frontend.forward_event(event)
        tokens, pooled, active_dim = self.encode_tokens(
            tokens,
            adapter=self.event_adapter,
            training=training,
            width_ratio=width_ratio,
        )
        return tokens, pooled, grid_size, active_dim

    def encode_track(
        self,
        event: torch.Tensor,
        *,
        cached_frame: torch.Tensor | None,
        patch_cache: PatchEmbedCache | None,
        training: bool,
        width_ratio: float | None = None,
    ) -> tuple[torch.Tensor, int]:
        resolved_width = self.resolve_width_ratio(training=training, requested=width_ratio)
        active_dim = self.active_dim(resolved_width)
        use_event_only = bool(getattr(self.patch_frontend, "supports_cache", False)) and (
            patch_cache is None or not patch_cache.valid or patch_cache.frame_psum is None
        ) and cached_frame is None
        tokens, _ = self.patch_frontend.forward_track(
            event,
            cache=patch_cache,
            cached_frame=cached_frame,
            event_only=use_event_only,
        )
        tokens = self.event_adapter(tokens)
        tokens = self.apply_width_mask(tokens, active_dim)
        tokens, pooled = self.backbone(tokens)
        pooled = self.apply_width_mask(pooled, active_dim)
        return pooled, active_dim

    def map_pretrained_state_dict(self, source_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mapped = dict(source_state)
        patch_weight = None
        patch_bias = None
        for key in ("patch_embed.proj.weight", "patch_embed.weight", "frame_embed.proj.weight"):
            if key in source_state:
                patch_weight = source_state[key]
                break
        for key in ("patch_embed.proj.bias", "patch_embed.bias", "frame_embed.proj.bias"):
            if key in source_state:
                patch_bias = source_state[key]
                break

        if patch_weight is None:
            return mapped

        gray_weight = self.collapse_patch_weight(patch_weight)

        if self.patch_embed_variant == "legacy":
            mapped.setdefault("patch_frontend.frame_embed.proj.weight", gray_weight)
            if patch_bias is not None:
                mapped.setdefault("patch_frontend.frame_embed.proj.bias", patch_bias)
            if "event_embed.proj.weight" in source_state:
                mapped.setdefault("patch_frontend.event_embed.proj.weight", source_state["event_embed.proj.weight"])
            if "event_embed.proj.bias" in source_state:
                mapped.setdefault("patch_frontend.event_embed.proj.bias", source_state["event_embed.proj.bias"])
            return mapped

        if self.patch_embed_variant == "split1":
            mapped.setdefault("patch_frontend.proj.weight", gray_weight.repeat(1, 3, 1, 1))
            if patch_bias is not None:
                mapped.setdefault("patch_frontend.proj.bias", patch_bias)
            return mapped

        if self.patch_embed_variant == "split2":
            mapped.setdefault("patch_frontend.frame_proj.weight", gray_weight)
            if patch_bias is not None:
                mapped.setdefault("patch_frontend.frame_proj.bias", patch_bias)
            return mapped

        return mapped
