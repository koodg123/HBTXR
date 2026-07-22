from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrackerNormalizedConfig:
    pruning_cfg: dict[str, Any]
    patch_embed_cfg: dict[str, Any]
    search_cfg: dict[str, Any]
    mask_cfg: dict[str, Any]
    component_cfg: dict[str, Any]
    runtime_cfg: dict[str, Any]
    eye_detector_cfg: dict[str, Any]
    eye_detector_mode: str
    search_use_mask_centroid: bool
    roi_bbox_cfg: dict[str, Any]
    search_use_roi_bbox_head: bool
    roi_bbox_use_external_eye: bool
    mask_cascade_cfg: dict[str, Any]
    search_use_mask_cascade: bool
    search_mask_cascade_use_xy: bool
    search_mask_cascade_use_ab: bool
    search_mask_cascade_detach: bool
    search_mask_cascade_xy_scale: float
    search_mask_cascade_ab_scale: float
    search_mask_cascade_ab_mode: str
    search_mask_cascade_ab_limit: float
    search_mask_cascade_min_ab: float
    structural_width_ratio: float
    legacy_width_masking: bool
    patch_embed_variant: str
    patch_warmup_epochs: int
    patch_search_use_pseudo_edges: bool
    eye_head_variant: str
    eye_reg_max: int
    mask_variant: str


class TrackerConfigNormalizer:
    def __init__(
        self,
        *,
        embed_dim: int,
        structural_width_ratio: float,
        eye_head_variant: str,
        eye_reg_max: int,
        mask_variant: str,
        runtime_cfg: dict[str, Any] | None = None,
        pruning_cfg: dict[str, Any] | None = None,
        patch_embed_cfg: dict[str, Any] | None = None,
        search_cfg: dict[str, Any] | None = None,
        mask_cfg: dict[str, Any] | None = None,
        component_cfg: dict[str, Any] | None = None,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.structural_width_ratio = float(structural_width_ratio)
        self.eye_head_variant = str(eye_head_variant).strip().lower()
        self.eye_reg_max = max(1, int(eye_reg_max))
        self.mask_variant = str(mask_variant).strip().lower()
        self.runtime_cfg = dict(runtime_cfg or {})
        self.pruning_cfg = dict(pruning_cfg or {})
        self.patch_embed_cfg = dict(patch_embed_cfg or {})
        self.search_cfg = dict(search_cfg or {})
        self.mask_cfg = dict(mask_cfg or {})
        self.component_cfg = dict(component_cfg or {})

    @staticmethod
    def _normalize_eye_detector_cfg(search_cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
        raw_cfg = search_cfg.get("eye_detector") or {}
        if isinstance(raw_cfg, str):
            raw_cfg = {"mode": raw_cfg}
        elif isinstance(raw_cfg, bool):
            raw_cfg = {"mode": "legacy" if raw_cfg else "disabled"}
        elif not isinstance(raw_cfg, dict):
            raw_cfg = {}
        cfg = dict(raw_cfg)
        mode = str(
            cfg.get(
                "mode",
                "dense" if bool(search_cfg.get("dense_eye", False)) else "legacy",
            )
        ).strip().lower()
        if mode not in {"legacy", "pooled", "dense"}:
            mode = "legacy"
        return cfg, mode

    @staticmethod
    def _normalize_roi_bbox_cfg(search_cfg: dict[str, Any]) -> tuple[dict[str, Any], bool, bool]:
        raw_cfg = search_cfg.get("roi_bbox_head") or {}
        if isinstance(raw_cfg, bool):
            raw_cfg = {"enabled": bool(raw_cfg)}
        elif not isinstance(raw_cfg, dict):
            raw_cfg = {}
        cfg = dict(raw_cfg)
        return cfg, bool(cfg.get("enabled", False)), bool(cfg.get("use_external_eye", False))

    @staticmethod
    def _normalize_mask_cascade_cfg(search_cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_cfg = search_cfg.get("mask_cascade") or {}
        if isinstance(raw_cfg, bool):
            raw_cfg = {"enabled": bool(raw_cfg)}
        elif not isinstance(raw_cfg, dict):
            raw_cfg = {}
        cfg = dict(raw_cfg)
        resolved = {
            "search_use_mask_cascade": bool(cfg.get("enabled", False)),
            "search_mask_cascade_use_xy": bool(cfg.get("use_xy", True)),
            "search_mask_cascade_use_ab": bool(cfg.get("use_ab", True)),
            "search_mask_cascade_detach": bool(cfg.get("detach_mask_stats", True)),
            "search_mask_cascade_xy_scale": float(cfg.get("xy_residual_scale", 1.0)),
            "search_mask_cascade_ab_scale": float(cfg.get("ab_residual_scale", 0.25)),
            "search_mask_cascade_ab_mode": str(cfg.get("ab_residual_mode", "log_scale")).strip().lower(),
            "search_mask_cascade_ab_limit": float(cfg.get("ab_residual_limit", 2.0)),
            "search_mask_cascade_min_ab": float(cfg.get("min_ab", 1.0)),
        }
        return cfg, resolved

    def _normalize_pruning(self) -> tuple[float, bool]:
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
        legacy_width_masking = bool(self.pruning_cfg.get("enabled", False)) and pruning_mode in {
            "implicit_width_slicing",
            "implicit_masking_legacy",
            "legacy_masking",
        }
        legacy_width_masking = legacy_width_masking or (
            bool(self.pruning_cfg.get("enabled", False)) and default_legacy_mode
        )
        structural_width_ratio = float(max(1.0 / max(1, self.embed_dim), min(1.0, self.structural_width_ratio)))
        return structural_width_ratio, legacy_width_masking

    def _normalize_patch_embed(self) -> tuple[str, int, bool]:
        variant = str(self.patch_embed_cfg.get("variant", "legacy")).strip().lower()
        warmup_epochs = int(self.patch_embed_cfg.get("warmup_epochs", 0))
        search_use_pseudo_edges = bool(self.patch_embed_cfg.get("search_use_pseudo_edges", False))
        return variant, warmup_epochs, search_use_pseudo_edges

    def build(self) -> TrackerNormalizedConfig:
        eye_detector_cfg, eye_detector_mode = self._normalize_eye_detector_cfg(self.search_cfg)
        roi_bbox_cfg, search_use_roi_bbox_head, roi_bbox_use_external_eye = self._normalize_roi_bbox_cfg(self.search_cfg)
        mask_cascade_cfg, mask_cascade_state = self._normalize_mask_cascade_cfg(self.search_cfg)
        structural_width_ratio, legacy_width_masking = self._normalize_pruning()
        patch_embed_variant, patch_warmup_epochs, patch_search_use_pseudo_edges = self._normalize_patch_embed()
        search_use_mask_centroid = bool(
            self.search_cfg.get(
                "xy_from_mask_centroid",
                self.search_cfg.get("center_from_mask_centroid", False),
            )
        )
        return TrackerNormalizedConfig(
            pruning_cfg=self.pruning_cfg,
            patch_embed_cfg=self.patch_embed_cfg,
            search_cfg=self.search_cfg,
            mask_cfg=self.mask_cfg,
            component_cfg=self.component_cfg,
            runtime_cfg=self.runtime_cfg,
            eye_detector_cfg=eye_detector_cfg,
            eye_detector_mode=eye_detector_mode,
            search_use_mask_centroid=search_use_mask_centroid,
            roi_bbox_cfg=roi_bbox_cfg,
            search_use_roi_bbox_head=search_use_roi_bbox_head,
            roi_bbox_use_external_eye=roi_bbox_use_external_eye,
            mask_cascade_cfg=mask_cascade_cfg,
            search_use_mask_cascade=mask_cascade_state["search_use_mask_cascade"],
            search_mask_cascade_use_xy=mask_cascade_state["search_mask_cascade_use_xy"],
            search_mask_cascade_use_ab=mask_cascade_state["search_mask_cascade_use_ab"],
            search_mask_cascade_detach=mask_cascade_state["search_mask_cascade_detach"],
            search_mask_cascade_xy_scale=mask_cascade_state["search_mask_cascade_xy_scale"],
            search_mask_cascade_ab_scale=mask_cascade_state["search_mask_cascade_ab_scale"],
            search_mask_cascade_ab_mode=mask_cascade_state["search_mask_cascade_ab_mode"],
            search_mask_cascade_ab_limit=mask_cascade_state["search_mask_cascade_ab_limit"],
            search_mask_cascade_min_ab=mask_cascade_state["search_mask_cascade_min_ab"],
            structural_width_ratio=structural_width_ratio,
            legacy_width_masking=legacy_width_masking,
            patch_embed_variant=patch_embed_variant,
            patch_warmup_epochs=patch_warmup_epochs,
            patch_search_use_pseudo_edges=patch_search_use_pseudo_edges,
            eye_head_variant=self.eye_head_variant,
            eye_reg_max=self.eye_reg_max,
            mask_variant=self.mask_variant,
        )


__all__ = ["TrackerConfigNormalizer", "TrackerNormalizedConfig"]
