from __future__ import annotations

from typing import Any, Dict

import torch

from src.loss.bundles import axis_aligned_bbox_loss as _axis_aligned_bbox_loss
from src.loss.bundles import eye_mask_loss as _eye_mask_loss
from src.loss.bundles import eye_region_loss as _eye_region_loss
from src.loss.bundles import pupil_bbox_aux_losses as _pupil_bbox_aux_losses
from src.loss.bundles import pupil_bbox_targets as _pupil_bbox_targets
from src.loss.bundles import pupil_branch_losses as _pupil_branch_losses
from src.loss.bundles import pupil_obb_aux_losses as _pupil_obb_aux_losses
from src.loss.bundles import pupil_obb_targets as _pupil_obb_targets
from src.loss.bundles import rotated_angle_loss as _rotated_angle_loss
from src.loss.bundles import rotated_overlap_loss as _rotated_overlap_loss
from src.loss.bundles import zero_loss as _zero_loss
from src.loss.common import _geom_mask, _sample_quality, _track_mask
from src.loss.primitives import ConstraintCenterLoss


def constraint_center_loss(
    pred_state: torch.Tensor,
    target_center: torch.Tensor,
    *,
    radius: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return ConstraintCenterLoss(radius=radius)(pred_state, target_center, weights)


def resolve_sample_masks(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quality = _sample_quality(batch)
    geom = _geom_mask(batch)
    track_geom = _track_mask(batch)
    return quality, geom, track_geom


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _batch_meta_text(batch: Dict[str, Any], index: int, field: str) -> str:
    meta = batch.get("meta")
    if isinstance(meta, list) and index < len(meta) and isinstance(meta[index], dict):
        value = meta[index].get(field)
        if value is not None:
            return str(value)
    values = batch.get(field)
    if isinstance(values, list) and index < len(values):
        return str(values[index])
    return ""


def _multiply_if_text_contains(weight: float, batch: Dict[str, Any], index: int, cfg: Dict[str, Any]) -> float:
    needle = str(cfg.get("contains", "")).strip()
    if not needle:
        return weight
    field = str(cfg.get("field", "session_key"))
    if needle in _batch_meta_text(batch, index, field):
        return weight * max(0.0, _as_float(cfg.get("multiplier"), 1.0))
    return weight


def build_loss_sample_weights(batch: Dict[str, Any], loss_cfg: Dict[str, Any], key: str = "track_sample_weight") -> torch.Tensor | None:
    cfg = loss_cfg.get(key) or (loss_cfg.get("sample_weight") if key == "track_sample_weight" else None) or {}
    if not isinstance(cfg, dict) or not _as_bool(cfg.get("enabled"), default=False):
        return None

    ref = batch.get("annotation_quality")
    if not torch.is_tensor(ref):
        raise ValueError(f"loss.{key} requires tensor batch['annotation_quality']")
    device = ref.device
    dtype = ref.dtype
    count = int(ref.view(-1).shape[0])
    min_weight = max(0.0, _as_float(cfg.get("min_weight"), 1.0e-6))
    weights = [max(min_weight, _as_float(cfg.get("base_weight"), 1.0)) for _ in range(count)]

    low_sim_cfg = cfg.get("low_similarity") or {}
    if isinstance(low_sim_cfg, dict) and _as_bool(low_sim_cfg.get("enabled"), default=False):
        field = str(low_sim_cfg.get("field", "similarity_target"))
        values = batch.get(field)
        if torch.is_tensor(values):
            sims = values.detach().view(-1).float().cpu().tolist()
            threshold = _as_float(low_sim_cfg.get("threshold"), 0.1)
            multiplier = max(0.0, _as_float(low_sim_cfg.get("multiplier"), 1.0))
            for idx, similarity in enumerate(sims[:count]):
                if similarity <= threshold:
                    weights[idx] *= multiplier

    session_cfg = cfg.get("session_contains")
    if isinstance(session_cfg, str):
        session_cfg = {"contains": session_cfg}
    if isinstance(session_cfg, dict):
        for idx in range(count):
            weights[idx] = _multiply_if_text_contains(weights[idx], batch, idx, session_cfg)

    for text_cfg in cfg.get("text_contains") or []:
        if isinstance(text_cfg, dict):
            for idx in range(count):
                weights[idx] = _multiply_if_text_contains(weights[idx], batch, idx, text_cfg)

    max_multiplier = cfg.get("max_multiplier")
    if max_multiplier is not None:
        cap = max(0.0, _as_float(max_multiplier, 0.0))
        weights = [min(weight, cap) for weight in weights]
    if not any(weight > 0.0 for weight in weights):
        raise ValueError(f"loss.{key} produced all-zero sample weights")
    return torch.as_tensor(weights, device=device, dtype=dtype)


def compute_eye_logs(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    quality: torch.Tensor,
    active_head: str = "all",
) -> Dict[str, torch.Tensor]:
    if "search/eye" not in outputs or active_head not in {"all", "eye"}:
        return {"loss_eye": _zero_loss(batch, outputs)}
    return _eye_region_loss(
        pred=outputs["search/eye"],
        target=batch["eye_target"],
        weights=quality,
        loss_cfg={**loss_cfg, "input_size": [int(v) for v in batch["frame"].shape[-2:]]},
        outputs=outputs,
    )


def compute_mask_losses(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    weights: torch.Tensor,
    active_head: str = "all",
) -> tuple[torch.Tensor, torch.Tensor]:
    mask_enabled = active_head in {"all", "mask"}
    mask_loss = _zero_loss(batch, outputs)
    if "search/mask_logits" in outputs and mask_enabled:
        mask_loss = _eye_mask_loss(
            outputs["search/mask_logits"],
            batch["mask_target"],
            weights,
            mode=str(loss_cfg.get("mask_mode", "bce_dice")),
        ) * float(loss_cfg.get("mask_weight", 1.0))
    mask_coarse_loss = _zero_loss(batch, outputs)
    if "search/mask_coarse_logits" in outputs and mask_enabled:
        mask_coarse_loss = _eye_mask_loss(
            outputs["search/mask_coarse_logits"],
            batch["mask_target"],
            weights,
            mode=str(loss_cfg.get("mask_mode", "bce_dice")),
        ) * float(loss_cfg.get("mask_coarse_weight", 0.25))
    return mask_loss, mask_coarse_loss


def compute_aux_loss(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    quality: torch.Tensor,
    aux_key: str | None,
) -> torch.Tensor:
    aux_weight = float(loss_cfg.get("aux_weight", 0.0))
    aux_loss = _zero_loss(batch, outputs)
    if aux_weight <= 0.0 or aux_key is None or aux_key not in outputs:
        return aux_loss
    aux_loss = torch.nn.functional.cross_entropy(outputs[aux_key], batch["aux_target"], reduction="none")
    aux_loss = (aux_loss * quality).sum() / quality.sum().clamp_min(1e-6)
    return aux_loss * aux_weight


def compute_pupil_branch_log_group(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    branch: str,
    state: torch.Tensor | None,
    weights: torch.Tensor,
    target_key: str = "pupil_search_target",
) -> Dict[str, torch.Tensor]:
    branch_losses: Dict[str, torch.Tensor] = {}
    pupil_key = f"{branch}/pupil"
    bbox_key = f"{branch}/pupil_bbox"
    obb_key = f"{branch}/pupil_obb"
    if pupil_key in outputs and state is not None:
        branch_losses = _pupil_branch_losses(
            pred=outputs[pupil_key],
            target=batch[target_key],
            state=state,
            target_state=batch["cur_state"],
            weights=weights,
            xy_weight=loss_cfg.get(f"{branch}_xy_weight", 1.0),
            ab_weight=loss_cfg.get(f"{branch}_ab_weight", 0.5),
            trig_weight=loss_cfg.get(f"{branch}_trig_weight", 1.0),
            geo_weight=loss_cfg.get(f"{branch}_geo_weight", 0.5),
            conf_weight=loss_cfg.get(f"{branch}_conf_weight", 0.1),
            prefix=branch,
        )
    branch_bbox_aux = (
        _pupil_bbox_aux_losses(
            pred=outputs[bbox_key],
            target_box=_pupil_bbox_targets(batch),
            weights=weights,
            weight=float(loss_cfg.get(f"{branch}_bbox_aux_weight", 0.0)),
            conf_weight=float(loss_cfg.get(f"{branch}_bbox_aux_conf_weight", 0.1)),
            mode=str(loss_cfg.get(f"{branch}_bbox_aux_mode", "ciou")),
            prefix=branch,
            axis_bbox_loss_fn=_axis_aligned_bbox_loss,
        )
        if bbox_key in outputs
        else {}
    )
    branch_obb_aux = (
        _pupil_obb_aux_losses(
            pred=outputs[obb_key],
            target_box=_pupil_obb_targets(batch),
            weights=weights,
            weight=float(loss_cfg.get(f"{branch}_obb_aux_weight", 0.0)),
            angle_weight=float(loss_cfg.get(f"{branch}_obb_aux_angle_weight", 0.5)),
            conf_weight=float(loss_cfg.get(f"{branch}_obb_aux_conf_weight", 0.1)),
            mode=str(loss_cfg.get(f"{branch}_obb_aux_mode", "rotated_bbox")),
            mgiou_fast_mode=bool(loss_cfg.get("mgiou_fast_mode", False)),
            prefix=branch,
            rotated_overlap_loss_fn=_rotated_overlap_loss,
            rotated_angle_loss_fn=_rotated_angle_loss,
        )
        if obb_key in outputs
        else {}
    )
    return {
        **{f"loss_{name}": value for name, value in branch_losses.items()},
        **{f"loss_{name}": value for name, value in branch_bbox_aux.items()},
        **{f"loss_{name}": value for name, value in branch_obb_aux.items()},
    }


def compute_constraint_center_log(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    loss_cfg: Dict,
    *,
    primary_state: torch.Tensor | None = None,
    primary_weights: torch.Tensor | None = None,
    fallback_state: torch.Tensor | None = None,
    fallback_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = float(loss_cfg.get("constraint_center_weight", 0.0))
    if weight <= 0.0:
        return _zero_loss(batch, outputs)
    if primary_state is not None and primary_weights is not None:
        target_state = primary_state
        target_weights = primary_weights
    elif fallback_state is not None and fallback_weights is not None:
        target_state = fallback_state
        target_weights = fallback_weights
    else:
        return _zero_loss(batch, outputs)
    return constraint_center_loss(
        target_state,
        batch["constraint_center"],
        radius=float(loss_cfg.get("constraint_center_radius", 24.0)),
        weights=target_weights,
    ) * weight


__all__ = [
    "compute_aux_loss",
    "compute_constraint_center_log",
    "compute_eye_logs",
    "compute_mask_losses",
    "compute_pupil_branch_log_group",
    "build_loss_sample_weights",
    "constraint_center_loss",
    "resolve_sample_masks",
]
