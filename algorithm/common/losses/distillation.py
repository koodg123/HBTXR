from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from common.losses.common import _geom_mask, _sample_quality, _track_mask, smooth_l1_with_mask
from common.losses.primitives import EventToFrameContrastiveLoss, FeatureConsistencyLoss, PredictionKDLoss, RelationalKDLoss


def _align_last_dim(student: torch.Tensor, teacher: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dim = min(student.size(-1), teacher.size(-1))
    return student[..., :dim], teacher[..., :dim]


def _cosine_feature_loss(student: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    student, teacher = _align_last_dim(student, teacher.detach())
    loss = 1.0 - F.cosine_similarity(student, teacher, dim=-1)
    if weights is None:
        return loss.mean()
    w = weights.to(device=loss.device, dtype=loss.dtype).view(-1)
    return (loss * w).sum() / w.sum().clamp_min(1e-6)


def _feature_distillation_loss(student: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    student, teacher = _align_last_dim(student, teacher.detach())
    cosine = _cosine_feature_loss(student, teacher, weights)
    l2 = smooth_l1_with_mask(student, teacher, weights)
    return 0.5 * (cosine + l2)


def _prediction_regression_distillation(student: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor, *, dims: int) -> torch.Tensor:
    student, teacher = _align_last_dim(student[..., :dims], teacher[..., :dims].detach())
    return smooth_l1_with_mask(student, teacher, weights)


def _logits_distillation(student: torch.Tensor, teacher: torch.Tensor, weights: torch.Tensor, *, temperature: float) -> torch.Tensor:
    if student.dim() == 1:
        student = student.unsqueeze(-1)
        teacher = teacher.unsqueeze(-1)
    student, teacher = _align_last_dim(student, teacher.detach())
    if student.size(-1) == 1:
        return smooth_l1_with_mask(student, teacher, weights)
    return PredictionKDLoss(temperature=temperature)(student, teacher, weights)


def _normalize_distillation_keys(keys: object) -> set[str] | None:
    if keys is None:
        return None
    if isinstance(keys, str):
        return {keys}
    try:
        return {str(key) for key in keys}
    except TypeError:
        return None


def _default_stage_distillation_targets(stage: str | None) -> dict[str, set[str]] | None:
    if stage is None:
        return None
    normalized = str(stage).strip().lower()
    if normalized == "stage1":
        return {
            "feature_keys": {"search/pooled"},
            "state_keys": {"search/state"},
            "prediction_keys": {"search/eye", "search/pupil"},
            "mask_keys": {"search/mask_logits"},
            "logits_keys": {"search/eye", "search/pupil", "search/aux"},
            "rkd_keys": {"search/pooled"},
        }
    if normalized == "stage2":
        return {
            "feature_keys": {"search/pooled", "event/pooled", "track/fused"},
            "state_keys": {"search/state", "event/state", "track/state"},
            "prediction_keys": {"search/eye", "search/pupil", "event/pupil", "track/pupil"},
            "mask_keys": set(),
            "logits_keys": {"search/eye", "search/pupil", "event/pupil", "track/pupil", "search/aux", "event/aux"},
            "rkd_keys": {"search/pooled", "event/pooled", "track/fused"},
        }
    return None


def _resolve_stage_distillation_targets(distillation_cfg: Dict, stage: str | None) -> dict[str, set[str]] | None:
    defaults = _default_stage_distillation_targets(stage)
    if defaults is None:
        return None
    targets_cfg = distillation_cfg.get("targets") or {}
    stage_cfg = targets_cfg.get(str(stage).strip().lower()) or {}
    resolved = dict(defaults)
    for key in ("feature_keys", "state_keys", "prediction_keys", "mask_keys", "logits_keys", "rkd_keys"):
        override = _normalize_distillation_keys(stage_cfg.get(key))
        if override is not None:
            resolved[key] = override
    return resolved


def _distillation_key_allowed(stage_targets: dict[str, set[str]] | None, group: str, key: str) -> bool:
    if stage_targets is None:
        return True
    allowed = stage_targets.get(group)
    if allowed is None:
        return True
    return key in allowed


def _default_stage_regularization_ssl_targets(stage: str | None) -> dict[str, set[str]] | None:
    if stage is None:
        return None
    normalized = str(stage).strip().lower()
    if normalized == "stage1":
        return {
            "feature_keys": set(),
            "state_keys": set(),
            "prediction_keys": set(),
            "contrastive_keys": set(),
            "self_keys": set(),
        }
    if normalized == "stage2":
        return {
            "feature_keys": {"search/pooled", "event/pooled"},
            "state_keys": {"search/state", "event/state", "track/state"},
            "prediction_keys": {"search/pupil", "event/pupil"},
            "contrastive_keys": {"search/pooled", "event/pooled"},
            "self_keys": {"search/state", "track/state"},
        }
    return None


def _resolve_stage_regularization_ssl_targets(ssl_cfg: Dict, stage: str | None) -> dict[str, set[str]] | None:
    defaults = _default_stage_regularization_ssl_targets(stage)
    if defaults is None:
        return None
    targets_cfg = ssl_cfg.get("targets") or {}
    stage_cfg = targets_cfg.get(str(stage).strip().lower()) or {}
    resolved = dict(defaults)
    for key in ("feature_keys", "state_keys", "prediction_keys", "contrastive_keys", "self_keys"):
        override = _normalize_distillation_keys(stage_cfg.get(key))
        if override is not None:
            resolved[key] = override
    return resolved


def compute_regularization_ssl_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    ssl_cfg: Dict,
    *,
    teacher_outputs: Dict[str, torch.Tensor] | None = None,
    stage: str | None = None,
) -> Dict[str, torch.Tensor]:
    if not bool(ssl_cfg.get("enabled", False)):
        return {}
    if teacher_outputs is not None and not bool(ssl_cfg.get("force_with_teacher", False)):
        return {}

    device = batch["frame"].device if "frame" in batch else next(iter(outputs.values())).device
    quality = _sample_quality(batch)
    geom = _geom_mask(batch)
    track_geom = _track_mask(batch)

    feature_loss_fn = FeatureConsistencyLoss()
    contrastive_loss_fn = EventToFrameContrastiveLoss(temperature=float((ssl_cfg.get("kd") or {}).get("temperature", 1.0)))
    kd_loss_fn = PredictionKDLoss(temperature=float((ssl_cfg.get("kd") or {}).get("temperature", 1.0)))
    stage_targets = _resolve_stage_regularization_ssl_targets(ssl_cfg, stage)

    logs: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device)

    if bool(ssl_cfg.get("feature_similarity", False)):
        losses = []
        if teacher_outputs is not None:
            for key, weights in (("search/pooled", quality * geom), ("event/pooled", quality * geom), ("track/fused", quality * track_geom)):
                if key in outputs and key in teacher_outputs:
                    losses.append(feature_loss_fn(*_align_last_dim(outputs[key], teacher_outputs[key]), weights))
        elif (
            _distillation_key_allowed(stage_targets, "feature_keys", "search/pooled")
            and _distillation_key_allowed(stage_targets, "feature_keys", "event/pooled")
            and "search/pooled" in outputs
            and "event/pooled" in outputs
        ):
            losses.append(feature_loss_fn(outputs["search/pooled"], outputs["event/pooled"], quality))
        if losses:
            logs["loss_regularization_ssl_feature"] = torch.stack(losses).mean() * float(ssl_cfg.get("feature_weight", 0.1))
            total = total + logs["loss_regularization_ssl_feature"]

    if (
        bool(ssl_cfg.get("cross_modal_contrastive", False))
        and _distillation_key_allowed(stage_targets, "contrastive_keys", "search/pooled")
        and _distillation_key_allowed(stage_targets, "contrastive_keys", "event/pooled")
        and "search/pooled" in outputs
        and "event/pooled" in outputs
    ):
        logs["loss_regularization_ssl_contrastive"] = contrastive_loss_fn(outputs["event/pooled"], outputs["search/pooled"], quality) * float(ssl_cfg.get("contrastive_weight", 0.05))
        total = total + logs["loss_regularization_ssl_contrastive"]

    if bool(ssl_cfg.get("state_similarity", False)):
        losses = []
        if teacher_outputs is not None:
            for key, weights in (("search/state", quality * geom), ("event/state", quality * geom), ("track/state", quality * track_geom)):
                if key in outputs and key in teacher_outputs:
                    losses.append(smooth_l1_with_mask(outputs[key], teacher_outputs[key].detach(), weights))
        else:
            if (
                _distillation_key_allowed(stage_targets, "state_keys", "search/state")
                and _distillation_key_allowed(stage_targets, "state_keys", "event/state")
                and "search/state" in outputs
                and "event/state" in outputs
            ):
                losses.append(smooth_l1_with_mask(outputs["search/state"], outputs["event/state"].detach(), quality * geom))
            if (
                _distillation_key_allowed(stage_targets, "state_keys", "track/state")
                and _distillation_key_allowed(stage_targets, "state_keys", "event/state")
                and "track/state" in outputs
                and "event/state" in outputs
            ):
                losses.append(smooth_l1_with_mask(outputs["track/state"], outputs["event/state"].detach(), quality * track_geom))
        if losses:
            logs["loss_regularization_ssl_state"] = torch.stack(losses).mean() * float(ssl_cfg.get("state_weight", 0.1))
            total = total + logs["loss_regularization_ssl_state"]

    if bool(ssl_cfg.get("prediction_similarity", False)):
        losses = []
        if teacher_outputs is not None:
            for key, weights in (
                ("search/pupil", quality * geom),
                ("event/pupil", quality * geom),
                ("track/pupil", quality * track_geom),
                ("search/eye", quality),
            ):
                if key in outputs and key in teacher_outputs:
                    losses.append(smooth_l1_with_mask(outputs[key], teacher_outputs[key].detach(), weights))
            if "search/aux" in outputs and "search/aux" in teacher_outputs:
                losses.append(kd_loss_fn(outputs["search/aux"], teacher_outputs["search/aux"], quality))
            if "event/aux" in outputs and "event/aux" in teacher_outputs:
                losses.append(kd_loss_fn(outputs["event/aux"], teacher_outputs["event/aux"], quality))
        else:
            if (
                _distillation_key_allowed(stage_targets, "prediction_keys", "search/pupil")
                and _distillation_key_allowed(stage_targets, "prediction_keys", "event/pupil")
                and "search/pupil" in outputs
                and "event/pupil" in outputs
            ):
                losses.append(smooth_l1_with_mask(outputs["search/pupil"], outputs["event/pupil"].detach(), quality * geom))
        if losses:
            logs["loss_regularization_ssl_prediction"] = torch.stack(losses).mean() * float(ssl_cfg.get("prediction_weight", 0.05))
            total = total + logs["loss_regularization_ssl_prediction"]

    if bool(ssl_cfg.get("self_consistency", False)):
        search_state = outputs.get("search/state")
        track_state = outputs.get("track/state")
        if (
            _distillation_key_allowed(stage_targets, "self_keys", "search/state")
            and _distillation_key_allowed(stage_targets, "self_keys", "track/state")
            and search_state is not None
            and track_state is not None
        ):
            logs["loss_regularization_ssl_self"] = smooth_l1_with_mask(search_state, track_state.detach(), quality * track_geom) * float(ssl_cfg.get("self_consistency_weight", 0.0))
            total = total + logs["loss_regularization_ssl_self"]

    logs["loss_regularization_ssl_total"] = total
    return logs


def compute_distillation_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor] | None,
    distillation_cfg: Dict,
    *,
    stage: str | None = None,
) -> Dict[str, torch.Tensor]:
    if teacher_outputs is None or not bool(distillation_cfg.get("enabled", False)):
        return {}

    device = batch["frame"].device if "frame" in batch else next(iter(outputs.values())).device
    quality = _sample_quality(batch)
    geom = _geom_mask(batch)
    track_geom = _track_mask(batch)
    temperature = float((distillation_cfg.get("kd") or {}).get("temperature", 1.0))
    rkd_loss_fn = RelationalKDLoss()
    stage_targets = _resolve_stage_distillation_targets(distillation_cfg, stage)
    logs: Dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=device)

    if bool(distillation_cfg.get("feature_similarity", True)):
        losses = []
        for key, weights in (("search/pooled", quality * geom), ("event/pooled", quality * geom), ("track/fused", quality * track_geom)):
            if _distillation_key_allowed(stage_targets, "feature_keys", key) and key in outputs and key in teacher_outputs:
                losses.append(_feature_distillation_loss(outputs[key], teacher_outputs[key], weights))
        if losses:
            logs["loss_distillation_feature"] = torch.stack(losses).mean() * float(distillation_cfg.get("feature_weight", 0.1))
            total = total + logs["loss_distillation_feature"]

    if bool(distillation_cfg.get("state_similarity", True)):
        losses = []
        for key, weights in (("search/state", quality * geom), ("event/state", quality * geom), ("track/state", quality * track_geom)):
            if _distillation_key_allowed(stage_targets, "state_keys", key) and key in outputs and key in teacher_outputs:
                losses.append(smooth_l1_with_mask(outputs[key], teacher_outputs[key].detach(), weights))
        if losses:
            logs["loss_distillation_state"] = torch.stack(losses).mean() * float(distillation_cfg.get("state_weight", 0.1))
            total = total + logs["loss_distillation_state"]

    if bool(distillation_cfg.get("prediction_similarity", True)):
        losses = []
        if _distillation_key_allowed(stage_targets, "prediction_keys", "search/eye") and "search/eye" in outputs and "search/eye" in teacher_outputs:
            losses.append(_prediction_regression_distillation(outputs["search/eye"], teacher_outputs["search/eye"], quality, dims=4))
        for key, dims, weights in (
            ("search/pupil", 6, quality * geom),
            ("event/pupil", 6, quality * geom),
            ("track/pupil", 6, quality * track_geom),
        ):
            if _distillation_key_allowed(stage_targets, "prediction_keys", key) and key in outputs and key in teacher_outputs:
                losses.append(_prediction_regression_distillation(outputs[key], teacher_outputs[key], weights, dims=dims))
        if losses:
            logs["loss_distillation_prediction"] = torch.stack(losses).mean() * float(distillation_cfg.get("prediction_weight", 0.05))
            total = total + logs["loss_distillation_prediction"]

    if (
        bool(distillation_cfg.get("mask_similarity", True))
        and _distillation_key_allowed(stage_targets, "mask_keys", "search/mask_logits")
        and "search/mask_logits" in outputs
        and "search/mask_logits" in teacher_outputs
    ):
        logs["loss_distillation_mask"] = smooth_l1_with_mask(outputs["search/mask_logits"], teacher_outputs["search/mask_logits"].detach(), quality * geom) * float(distillation_cfg.get("mask_weight", 0.05))
        total = total + logs["loss_distillation_mask"]

    kd_cfg = distillation_cfg.get("kd") or {}
    if bool(kd_cfg.get("enabled", False)):
        kd_losses = []
        if _distillation_key_allowed(stage_targets, "logits_keys", "search/eye") and "search/eye" in outputs and "search/eye" in teacher_outputs:
            kd_losses.append(_logits_distillation(outputs["search/eye"][:, 4:5], teacher_outputs["search/eye"][:, 4:5], quality, temperature=temperature))
        if _distillation_key_allowed(stage_targets, "logits_keys", "search/pupil") and "search/pupil" in outputs and "search/pupil" in teacher_outputs:
            kd_losses.append(_logits_distillation(outputs["search/pupil"][:, 6:7], teacher_outputs["search/pupil"][:, 6:7], quality * geom, temperature=temperature))
        if _distillation_key_allowed(stage_targets, "logits_keys", "event/pupil") and "event/pupil" in outputs and "event/pupil" in teacher_outputs:
            kd_losses.append(_logits_distillation(outputs["event/pupil"][:, 6:7], teacher_outputs["event/pupil"][:, 6:7], quality * geom, temperature=temperature))
        if _distillation_key_allowed(stage_targets, "logits_keys", "track/pupil") and "track/pupil" in outputs and "track/pupil" in teacher_outputs:
            kd_losses.append(_logits_distillation(outputs["track/pupil"][:, 6:8], teacher_outputs["track/pupil"][:, 6:8], quality * track_geom, temperature=temperature))
        if _distillation_key_allowed(stage_targets, "logits_keys", "search/aux") and "search/aux" in outputs and "search/aux" in teacher_outputs:
            kd_losses.append(_logits_distillation(outputs["search/aux"], teacher_outputs["search/aux"], quality, temperature=temperature))
        if _distillation_key_allowed(stage_targets, "logits_keys", "event/aux") and "event/aux" in outputs and "event/aux" in teacher_outputs:
            kd_losses.append(_logits_distillation(outputs["event/aux"], teacher_outputs["event/aux"], quality, temperature=temperature))
        if kd_losses:
            logs["loss_distillation_kd"] = torch.stack(kd_losses).mean() * float(kd_cfg.get("weight", 0.05))
            total = total + logs["loss_distillation_kd"]

    rkd_cfg = distillation_cfg.get("rkd") or {}
    if bool(rkd_cfg.get("enabled", False)):
        rkd_losses = []
        for key in ("search/pooled", "event/pooled", "track/fused"):
            if _distillation_key_allowed(stage_targets, "rkd_keys", key) and key in outputs and key in teacher_outputs:
                student_repr, teacher_repr = _align_last_dim(outputs[key], teacher_outputs[key])
                rkd_losses.append(rkd_loss_fn(student_repr, teacher_repr))
        if rkd_losses:
            logs["loss_distillation_rkd"] = torch.stack(rkd_losses).mean() * float(rkd_cfg.get("distance_weight", 0.05))
            total = total + logs["loss_distillation_rkd"]

    logs["loss_distillation_total"] = total
    return logs


def compute_ssl_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor] | None,
    ssl_cfg: Dict,
    *,
    stage: str | None = None,
) -> Dict[str, torch.Tensor]:
    if not bool(ssl_cfg.get("enabled", False)):
        return {}
    regularization_cfg = {
        "enabled": bool(ssl_cfg.get("enabled", False)) and not bool(ssl_cfg.get("teacher_student", True)),
        "force_with_teacher": False,
        "feature_similarity": bool(ssl_cfg.get("feature_similarity", False)),
        "feature_weight": float(ssl_cfg.get("feature_weight", 0.1)),
        "state_similarity": bool(ssl_cfg.get("state_similarity", True)),
        "state_weight": float(ssl_cfg.get("state_weight", 0.1)),
        "prediction_similarity": bool(ssl_cfg.get("prediction_similarity", False)),
        "prediction_weight": float(ssl_cfg.get("prediction_weight", 0.05)),
        "cross_modal_contrastive": bool(ssl_cfg.get("cross_modal_contrastive", True)),
        "contrastive_weight": float(ssl_cfg.get("contrastive_weight", 0.05)),
        "self_consistency": False,
        "self_consistency_weight": 0.0,
    }
    distillation_cfg = {
        "enabled": bool(ssl_cfg.get("enabled", False)) and bool(ssl_cfg.get("teacher_student", True)),
        "feature_similarity": bool(ssl_cfg.get("feature_similarity", True)),
        "feature_weight": float(ssl_cfg.get("feature_weight", 0.1)),
        "state_similarity": bool(ssl_cfg.get("state_similarity", True)),
        "state_weight": float(ssl_cfg.get("state_weight", 0.1)),
        "prediction_similarity": bool(ssl_cfg.get("prediction_similarity", True)),
        "prediction_weight": float(ssl_cfg.get("prediction_weight", 0.05)),
        "mask_similarity": True,
        "mask_weight": 0.05,
        "kd": ssl_cfg.get("kd") or {"enabled": False, "temperature": 1.0, "weight": 0.05},
        "rkd": ssl_cfg.get("rkd") or {"enabled": False, "distance_weight": 0.05},
    }
    logs = {}
    logs.update(compute_regularization_ssl_losses(batch, outputs, regularization_cfg, teacher_outputs=teacher_outputs, stage=stage))
    logs.update(compute_distillation_losses(batch, outputs, teacher_outputs, distillation_cfg, stage=stage))
    total = torch.zeros((), device=batch["frame"].device if "frame" in batch else next(iter(outputs.values())).device)
    if "loss_regularization_ssl_total" in logs:
        total = total + logs["loss_regularization_ssl_total"]
    if "loss_distillation_total" in logs:
        total = total + logs["loss_distillation_total"]
    logs["loss_ssl_total"] = total
    return logs


__all__ = [
    "compute_distillation_losses",
    "compute_regularization_ssl_losses",
    "compute_ssl_losses",
]
