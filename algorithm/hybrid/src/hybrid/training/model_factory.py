from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from hybrid.models.hybrid_tracker import HBTXRTracker
from hybrid.models.tracker import TrackerConfigNormalizer
from hybrid.models.pruning import normalize_compression_cfg, resolve_model_role_cfg

from .checkpoints import load_model_state, unwrap_model
from .ensemble_teacher import EnsembleTeacherMember, WeightedOutputEnsembleTeacher


def build_model(cfg: dict, *, role: str = "student") -> HBTXRTracker:
    normalized_cfg = normalize_compression_cfg(cfg)
    model_cfg = resolve_model_role_cfg(normalized_cfg, role=role)
    head_cfg = model_cfg.get("heads") or {}
    runtime_cfg = normalized_cfg.get("runtime") or {}
    structural_width_ratio = float(model_cfg.get("structural_width_ratio", 1.0))
    eye_head_variant = str(head_cfg.get("eye_variant", "legacy"))
    search_head_variant = str(head_cfg.get("search_variant", "legacy"))
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
    input_size = tuple(model_cfg.get("input_size", cfg.get("data", {}).get("input_size", [256, 256])))
    return HBTXRTracker(
        embed_dim=int(model_cfg.get("embed_dim", 192)),
        depth=int(model_cfg.get("depth", 6)),
        num_heads=int(model_cfg.get("num_heads", 3)),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        mlp_hidden_dim=model_cfg.get("mlp_hidden_dim"),
        patch_size=int(model_cfg.get("patch_size", 16)),
        input_size=input_size,
        frame_input_size=tuple(model_cfg.get("frame_input_size", input_size)),
        event_input_size=tuple(model_cfg.get("event_input_size", input_size)),
        event_cut_depth=model_cfg.get("event_cut_depth", model_cfg.get("track_depth")),
        track_depth=model_cfg.get("track_depth", model_cfg.get("event_cut_depth")),
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
        search_head_variant=search_head_variant,
        search_head_residual_hidden_dim=head_cfg.get("search_residual_hidden_dim"),
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
        enable_search_center_candidate_head=bool(head_cfg.get("search_center_candidate", False)),
        enable_track_state_aux_head=bool(head_cfg.get("track_state_aux", False)),
        enable_track_state_simdr_head=bool(head_cfg.get("track_state_simdr", False)),
        enable_track_center_heatmap_head=bool(head_cfg.get("track_center_heatmap", False)),
        enable_track_center_refine_head=bool(head_cfg.get("track_center_refine", False)),
        enable_track_center_candidate_head=bool(head_cfg.get("track_center_candidate", False)),
        track_state_simdr_bins=int(head_cfg.get("track_state_simdr_bins", 64)),
        track_center_heatmap_grid=int(head_cfg.get("track_center_heatmap_grid", 32)),
        track_center_refine_max_delta_px=float(head_cfg.get("track_center_refine_max_delta_px", 4.0)),
        track_center_candidate_count=int(head_cfg.get("track_center_candidate_count", 4)),
        track_center_candidate_max_delta_px=float(head_cfg.get("track_center_candidate_max_delta_px", 8.0)),
        search_center_candidate_count=int(head_cfg.get("search_center_candidate_count", 4)),
        search_center_candidate_max_delta_px=float(head_cfg.get("search_center_candidate_max_delta_px", 8.0)),
        track_state_simdr_as_track_state=bool(head_cfg.get("track_state_simdr_as_track_state", False)),
        track_state_simdr_coordinate_max=float(head_cfg.get("track_state_simdr_coordinate_max", 255.0)),
        track_state_simdr_blend=float(head_cfg.get("track_state_simdr_blend", 1.0)),
        track_center_heatmap_as_track_state=bool(head_cfg.get("track_center_heatmap_as_track_state", False)),
        track_center_heatmap_coordinate_max=float(head_cfg.get("track_center_heatmap_coordinate_max", 255.0)),
        track_center_heatmap_blend=float(head_cfg.get("track_center_heatmap_blend", 1.0)),
        track_center_refine_as_track_state=bool(head_cfg.get("track_center_refine_as_track_state", False)),
        track_center_refine_blend=float(head_cfg.get("track_center_refine_blend", 1.0)),
        track_center_candidate_as_track_state=bool(head_cfg.get("track_center_candidate_as_track_state", False)),
        track_center_candidate_blend=float(head_cfg.get("track_center_candidate_blend", 1.0)),
        search_center_candidate_as_search_state=bool(head_cfg.get("search_center_candidate_as_search_state", False)),
        search_center_candidate_blend=float(head_cfg.get("search_center_candidate_blend", 1.0)),
        tracker_cfg=tracker_cfg,
    )


def create_teacher_model(student: nn.Module, cfg: dict[str, Any], device) -> nn.Module | None:
    normalized_cfg = normalize_compression_cfg(cfg)
    distillation_cfg = normalized_cfg.get("distillation") or {}
    if not bool(distillation_cfg.get("enabled", False)) or not bool(distillation_cfg.get("teacher_student", True)):
        return None
    ensemble_cfg = distillation_cfg.get("ensemble") or {}
    if bool(ensemble_cfg.get("enabled", False)):
        members: list[EnsembleTeacherMember] = []
        for idx, spec in enumerate(ensemble_cfg.get("checkpoints") or []):
            path = Path(str(spec.get("path") or "")).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"ensemble teacher checkpoint not found: {path}")
            state = torch.load(path, map_location="cpu")
            model_state = state.get("model") if isinstance(state, dict) else None
            if not isinstance(model_state, dict):
                raise ValueError(f"ensemble teacher checkpoint has no model state: {path}")
            teacher_member = build_model(normalized_cfg, role="teacher").to(device)
            load_model_state(model=teacher_member, state_dict=model_state, strict=False)
            teacher_member.eval()
            for param in teacher_member.parameters():
                param.requires_grad_(False)
            members.append(
                EnsembleTeacherMember(
                    label=str(spec.get("label") or f"teacher_{idx}"),
                    model=teacher_member,
                    weight=float(spec.get("weight", 1.0)),
                )
            )
        return WeightedOutputEnsembleTeacher(members).to(device)
    teacher = build_model(normalized_cfg, role="teacher").to(device)
    load_model_state(model=teacher, state_dict=unwrap_model(student).state_dict(), strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


__all__ = ["build_model", "create_teacher_model"]
