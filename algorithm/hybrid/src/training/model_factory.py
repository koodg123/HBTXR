from __future__ import annotations

from typing import Any

from torch import nn

from src.models.hybrid_tracker import HBTXRTracker
from src.models.tracker import TrackerConfigNormalizer
from src.models.pruning import normalize_compression_cfg, resolve_model_role_cfg

from .checkpoints import load_model_state, unwrap_model


def build_model(cfg: dict, *, role: str = "student") -> HBTXRTracker:
    normalized_cfg = normalize_compression_cfg(cfg)
    model_cfg = resolve_model_role_cfg(normalized_cfg, role=role)
    head_cfg = model_cfg.get("heads") or {}
    runtime_cfg = normalized_cfg.get("runtime") or {}
    structural_width_ratio = float(model_cfg.get("structural_width_ratio", 1.0))
    eye_head_variant = str(head_cfg.get("eye_variant", "legacy"))
    eye_reg_max = int(head_cfg.get("eye_reg_max", 1))
    mask_variant = str((model_cfg.get("mask") or {}).get("variant", "legacy"))
    tracker_cfg = TrackerConfigNormalizer(
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        structural_width_ratio=structural_width_ratio,
        eye_head_variant=eye_head_variant,
        eye_reg_max=eye_reg_max,
        mask_variant=mask_variant,
        runtime_cfg=runtime_cfg,
        pruning_cfg=normalized_cfg.get("pruning") or {},
        patch_embed_cfg=model_cfg.get("patch_embed") or {},
        search_cfg=model_cfg.get("search") or {},
        mask_cfg=model_cfg.get("mask") or {},
        component_cfg=model_cfg.get("components") or {},
    ).build()
    return HBTXRTracker(
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim"),
        patch_size=int(model_cfg.get("patch_size", 16)),
        input_size=tuple(model_cfg.get("input_size", cfg.get("data", {}).get("input_size", [256, 256]))),
        dropout=float(model_cfg.get("dropout", 0.0)),
        aux_classes=int(model_cfg.get("aux_classes", 5)),
        adapter_hidden_dim=model_cfg.get("adapter_hidden_dim"),
        prev_state_hidden_dim=model_cfg.get("prev_state_hidden_dim"),
        head_hidden_dim=model_cfg.get("head_hidden_dim"),
        track_head_hidden_dim=model_cfg.get("track_head_hidden_dim"),
        mask_hidden_dim=model_cfg.get("mask_hidden_dim"),
        structural_width_ratio=tracker_cfg.structural_width_ratio,
        eye_head_variant=tracker_cfg.eye_head_variant,
        eye_reg_max=tracker_cfg.eye_reg_max,
        mask_variant=tracker_cfg.mask_variant,
        enable_eye_head=bool(head_cfg.get("eye", True)),
        enable_search_head=bool(head_cfg.get("search", True)),
        enable_event_head=bool(head_cfg.get("event", True)),
        enable_track_head=bool(head_cfg.get("track", True)),
        enable_mask_head=bool(head_cfg.get("mask", True)),
        enable_aux_head=bool(head_cfg.get("aux", True)),
        enable_search_bbox_aux_head=bool(head_cfg.get("search_bbox_aux", False)),
        enable_event_bbox_aux_head=bool(head_cfg.get("event_bbox_aux", False)),
        enable_search_obb_aux_head=bool(head_cfg.get("search_obb_aux", False)),
        enable_event_obb_aux_head=bool(head_cfg.get("event_obb_aux", False)),
        tracker_cfg=tracker_cfg,
    )


def create_teacher_model(student: nn.Module, cfg: dict[str, Any], device) -> nn.Module | None:
    normalized_cfg = normalize_compression_cfg(cfg)
    distillation_cfg = normalized_cfg.get("distillation") or {}
    if not bool(distillation_cfg.get("enabled", False)) or not bool(distillation_cfg.get("teacher_student", True)):
        return None
    teacher = build_model(normalized_cfg, role="teacher").to(device)
    load_model_state(model=teacher, state_dict=unwrap_model(student).state_dict(), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


__all__ = ["build_model", "create_teacher_model"]
