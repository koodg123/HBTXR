from __future__ import annotations

from typing import Any


DEFAULT_EVENT_BUILDER = {
    "policy": "fixed_count",
    "time_bin_us": 5000,
    "event_count_target": 5000,
    "accumulation": "fast_causal_linear",
    "causal_weight_power": 1.0,
    "fast_causal_limit": 25.0,
    "polarity_split": True,
    "generation_strategy": "raw_window",
    "source_pair_scope": "anchor_window",
    "average_weighting": "mean",
    "crop_policy": "manifest_roi",
    "crop_scope": "within_roi",
    "crop_min_events": 64,
    "crop_quantile_low": 0.05,
    "crop_quantile_high": 0.95,
    "crop_margin_px": 12.0,
    "crop_min_side_px": 96.0,
}


def mode_defaults(mode: str) -> dict[str, str]:
    normalized = str(mode).strip().lower()
    if normalized == "mode2":
        return {"mode": "mode2", "canonical_name": "canonical2", "manifest_name": "manifest2", "frame_source": "interpolated"}
    if normalized == "mode0":
        return {"mode": "mode0", "canonical_name": "canonical0", "manifest_name": "manifest0", "frame_source": "original"}
    return {"mode": "mode1", "canonical_name": "canonical1", "manifest_name": "manifest1", "frame_source": "original"}


def resolve_data_mode(data_cfg: dict[str, Any] | None) -> str:
    cfg = data_cfg or {}
    return mode_defaults(cfg.get("mode", "mode1"))["mode"]


def resolve_mode_block(data_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = data_cfg or {}
    mode = resolve_data_mode(cfg)
    return dict(cfg.get(mode) or {})


def resolve_mode_contract(data_cfg: dict[str, Any] | None) -> dict[str, str]:
    cfg = data_cfg or {}
    defaults = mode_defaults(resolve_data_mode(cfg))
    mode_cfg = resolve_mode_block(cfg)
    return {
        "mode": defaults["mode"],
        "canonical_name": str(mode_cfg.get("canonical_name", cfg.get("canonical_name", defaults["canonical_name"]))),
        "manifest_name": str(mode_cfg.get("manifest_name", cfg.get("manifest_name", defaults["manifest_name"]))),
        "frame_source": str(mode_cfg.get("frame_source", defaults["frame_source"])),
    }


def build_dataset_kwargs(data_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = data_cfg or {}
    mode_contract = resolve_mode_contract(cfg)
    mode = mode_contract["mode"]
    mode_cfg = dict(cfg.get(mode) or {})
    loader_cfg = dict(mode_cfg.get("loader") or {})
    event_builder_cfg = dict(
        mode_cfg.get("event_builder")
        or mode_cfg.get("synthetic_event_builder")
        or cfg.get("event_builder")
        or {}
    )
    component_cfg = {
        **dict(cfg.get("components") or {}),
        **dict(mode_cfg.get("components") or {}),
    }
    return {
        "input_size": tuple(mode_cfg.get("input_size", cfg.get("input_size", [256, 256]))),
        "resize_policy": str(mode_cfg.get("resize_policy", cfg.get("resize_policy", "facet_square_direct"))),
        "canonical_root": mode_cfg.get("canonical_root", cfg.get("canonical_root")),
        "cache_root": loader_cfg.get("cache_root", cfg.get("cache_root")),
        "use_cache": bool(loader_cfg.get("use_cache", cfg.get("use_cache", True))),
        "per_channel_normalize": bool(loader_cfg.get("per_channel_normalize", cfg.get("per_channel_normalize", True))),
        "event_builder": {**DEFAULT_EVENT_BUILDER, **event_builder_cfg},
        "data_mode": mode_contract["mode"],
        "canonical_name": mode_contract["canonical_name"],
        "manifest_name": mode_contract["manifest_name"],
        "frame_source": mode_contract["frame_source"],
        "mode2_execution": str(mode_cfg.get("execution", "materialized")),
        "component_cfg": component_cfg,
    }


__all__ = [
    "DEFAULT_EVENT_BUILDER",
    "build_dataset_kwargs",
    "mode_defaults",
    "resolve_data_mode",
    "resolve_mode_block",
    "resolve_mode_contract",
]
