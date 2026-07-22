from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F

from common.losses.common import ellipse_gwd_loss, normalize_uv, sigmoid_bce_with_mask, smooth_l1_with_mask, weighted_reduce


def p10_boundary_loss(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    weights: torch.Tensor,
    *,
    margin_px: float = 10.0,
    band_px: float = 4.0,
    temperature_px: float = 1.0,
) -> torch.Tensor:
    center_error = torch.linalg.norm(pred_state[:, :2] - target_state[:, :2], dim=-1)
    margin = float(margin_px)
    band = max(float(band_px), 1.0e-6)
    temperature = max(float(temperature_px), 1.0e-6)
    boundary_weight = torch.exp(-0.5 * ((center_error.detach() - margin) / band).pow(2))
    boundary_penalty = F.softplus((center_error - margin) / temperature) * boundary_weight
    return weighted_reduce(boundary_penalty, weights)


def center_soft_threshold_loss(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    weights: torch.Tensor,
    *,
    margin_px: float,
    temperature_px: float = 1.0,
) -> torch.Tensor:
    margin = float(margin_px)
    temperature = max(float(temperature_px), 1.0e-6)
    center_error = torch.linalg.norm(pred_state[:, :2] - target_state[:, :2], dim=-1)
    # Positive-class BCE for "center error is inside metric threshold".
    return weighted_reduce(F.softplus((center_error - margin) / temperature), weights)


def track_branch_losses(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    target_state: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    valid_track: torch.Tensor,
    similarity_target: torch.Tensor | None = None,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    weights = quality * track_geom
    low_sim_scale = float(loss_cfg.get("track_low_similarity_weight_scale", 0.0))
    if similarity_target is not None and low_sim_scale > 0.0:
        threshold = max(float(loss_cfg.get("track_low_similarity_threshold", 0.3)), 1.0e-6)
        low_sim = ((threshold - similarity_target.view(-1).float()) / threshold).clamp(0.0, 1.0)
        multiplier = 1.0 + low_sim_scale * low_sim
        max_multiplier = float(loss_cfg.get("track_low_similarity_max_multiplier", 0.0))
        if max_multiplier > 0.0:
            multiplier = multiplier.clamp(max=max_multiplier)
        weights = weights * multiplier.to(device=weights.device, dtype=weights.dtype)
    center_error = torch.linalg.norm(state[:, :2] - target_state[:, :2], dim=-1)
    center_hinge_margin = float(loss_cfg.get("track_center_hinge_margin_px", 10.0))
    center_hinge = torch.relu(center_error - center_hinge_margin)
    p10_margin = float(loss_cfg.get("track_p10_boundary_margin_px", 10.0))
    p10_band = float(loss_cfg.get("track_p10_boundary_band_px", 4.0))
    p10_temperature = float(loss_cfg.get("track_p10_boundary_temperature_px", 1.0))
    p10_soft_margin = float(loss_cfg.get("track_p10_soft_threshold_margin_px", loss_cfg.get("track_p10_boundary_margin_px", 10.0)))
    p10_soft_temperature = float(loss_cfg.get("track_p10_soft_threshold_temperature_px", loss_cfg.get("track_p10_boundary_temperature_px", 1.0)))
    p5_soft_margin = float(loss_cfg.get("track_p5_soft_threshold_margin_px", 5.0))
    p5_soft_temperature = float(loss_cfg.get("track_p5_soft_threshold_temperature_px", loss_cfg.get("track_p10_soft_threshold_temperature_px", 1.0)))
    axis_log_pred = torch.log(state[:, 2:4].clamp_min(1.0e-3))
    axis_log_target = torch.log(target_state[:, 2:4].clamp_min(1.0e-3))
    angle_cos = (normalize_uv(state[:, 4:6]) * normalize_uv(target_state[:, 4:6])).sum(dim=-1).clamp(-1.0, 1.0)
    return {
        "track_xy": smooth_l1_with_mask(pred[:, 0:2], target[:, 0:2], weights) * float(loss_cfg.get("track_xy_weight", 1.0)),
        "track_center_l2": weighted_reduce(((state[:, :2] - target_state[:, :2]) ** 2).sum(dim=-1), weights)
        * float(loss_cfg.get("track_center_l2_weight", 0.0)),
        "track_center_hinge": weighted_reduce(center_hinge, weights)
        * float(loss_cfg.get("track_center_hinge_weight", 0.0)),
        "track_center_hinge_sq": weighted_reduce(center_hinge.pow(2), weights)
        * float(loss_cfg.get("track_center_hinge_sq_weight", 0.0)),
        "track_p10_boundary": p10_boundary_loss(
            state,
            target_state,
            weights,
            margin_px=p10_margin,
            band_px=p10_band,
            temperature_px=p10_temperature,
        )
        * float(loss_cfg.get("track_p10_boundary_weight", 0.0)),
        "track_p10_soft_threshold": center_soft_threshold_loss(
            state,
            target_state,
            weights,
            margin_px=p10_soft_margin,
            temperature_px=p10_soft_temperature,
        )
        * float(loss_cfg.get("track_p10_soft_threshold_weight", 0.0)),
        "track_p5_soft_threshold": center_soft_threshold_loss(
            state,
            target_state,
            weights,
            margin_px=p5_soft_margin,
            temperature_px=p5_soft_temperature,
        )
        * float(loss_cfg.get("track_p5_soft_threshold_weight", 0.0)),
        "track_axis_log": smooth_l1_with_mask(axis_log_pred, axis_log_target, weights)
        * float(loss_cfg.get("track_axis_log_weight", 0.0)),
        "track_angle_cos": weighted_reduce(1.0 - angle_cos, weights) * float(loss_cfg.get("track_angle_cos_weight", 0.0)),
        "track_ab": smooth_l1_with_mask(pred[:, 2:4], target[:, 2:4], weights) * float(loss_cfg.get("track_ab_weight", 0.5)),
        "track_trig": smooth_l1_with_mask(pred[:, 4:6], target[:, 4:6], weights) * float(loss_cfg.get("track_trig_weight", 1.0)),
        "track_geo": ellipse_gwd_loss(state, target_state, weights) * float(loss_cfg.get("track_geo_weight", 0.5)),
        "track_conf": sigmoid_bce_with_mask(pred[:, 6], target[:, 6], quality * valid_track.view(-1)) * float(loss_cfg.get("track_conf_weight", 0.1)),
        "track_quality": sigmoid_bce_with_mask(pred[:, 7], target[:, 7], quality) * float(loss_cfg.get("track_quality_weight", 0.1)),
    }


def track_state_aux_losses(
    *,
    state_aux: torch.Tensor,
    target_state: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    weights = quality * track_geom
    p10_margin = float(loss_cfg.get("track_p10_boundary_margin_px", 10.0))
    p10_band = float(loss_cfg.get("track_p10_boundary_band_px", 4.0))
    p10_temperature = float(loss_cfg.get("track_p10_boundary_temperature_px", 1.0))
    p10_soft_margin = float(loss_cfg.get("track_state_aux_p10_soft_threshold_margin_px", loss_cfg.get("track_p10_soft_threshold_margin_px", p10_margin)))
    p10_soft_temperature = float(
        loss_cfg.get("track_state_aux_p10_soft_threshold_temperature_px", loss_cfg.get("track_p10_soft_threshold_temperature_px", p10_temperature))
    )
    p5_soft_margin = float(loss_cfg.get("track_state_aux_p5_soft_threshold_margin_px", loss_cfg.get("track_p5_soft_threshold_margin_px", 5.0)))
    p5_soft_temperature = float(
        loss_cfg.get("track_state_aux_p5_soft_threshold_temperature_px", loss_cfg.get("track_p5_soft_threshold_temperature_px", 1.0))
    )
    axis_log_pred = torch.log(state_aux[:, 2:4].clamp_min(1.0e-3))
    axis_log_target = torch.log(target_state[:, 2:4].clamp_min(1.0e-3))
    angle_cos = (normalize_uv(state_aux[:, 4:6]) * normalize_uv(target_state[:, 4:6])).sum(dim=-1).clamp(-1.0, 1.0)
    return {
        "track_state_aux_center_l2": weighted_reduce(((state_aux[:, :2] - target_state[:, :2]) ** 2).sum(dim=-1), weights)
        * float(loss_cfg.get("track_state_aux_center_l2_weight", 0.0)),
        "track_state_aux_p10_boundary": p10_boundary_loss(
            state_aux,
            target_state,
            weights,
            margin_px=p10_margin,
            band_px=p10_band,
            temperature_px=p10_temperature,
        )
        * float(loss_cfg.get("track_state_aux_p10_boundary_weight", 0.0)),
        "track_state_aux_p10_soft_threshold": center_soft_threshold_loss(
            state_aux,
            target_state,
            weights,
            margin_px=p10_soft_margin,
            temperature_px=p10_soft_temperature,
        )
        * float(loss_cfg.get("track_state_aux_p10_soft_threshold_weight", 0.0)),
        "track_state_aux_p5_soft_threshold": center_soft_threshold_loss(
            state_aux,
            target_state,
            weights,
            margin_px=p5_soft_margin,
            temperature_px=p5_soft_temperature,
        )
        * float(loss_cfg.get("track_state_aux_p5_soft_threshold_weight", 0.0)),
        "track_state_aux_axis_log": smooth_l1_with_mask(axis_log_pred, axis_log_target, weights)
        * float(loss_cfg.get("track_state_aux_axis_log_weight", 0.0)),
        "track_state_aux_angle_cos": weighted_reduce(1.0 - angle_cos, weights)
        * float(loss_cfg.get("track_state_aux_angle_cos_weight", 0.0)),
    }


def track_target_override_losses(
    *,
    state: torch.Tensor,
    override_state: torch.Tensor,
    override_weight: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    loss_cfg: Dict,
    prefix: str = "track_target_override",
) -> Dict[str, torch.Tensor]:
    """Auxiliary loss toward split-isolated teacher/pseudo target state."""

    weights = quality * track_geom * override_weight.view(-1).to(device=quality.device, dtype=quality.dtype).clamp_min(0.0)
    axis_log_pred = torch.log(state[:, 2:4].clamp_min(1.0e-3))
    axis_log_target = torch.log(override_state[:, 2:4].clamp_min(1.0e-3))
    angle_cos = (normalize_uv(state[:, 4:6]) * normalize_uv(override_state[:, 4:6])).sum(dim=-1).clamp(-1.0, 1.0)
    return {
        f"{prefix}_center_l2": weighted_reduce(((state[:, :2] - override_state[:, :2]) ** 2).sum(dim=-1), weights)
        * float(loss_cfg.get(f"{prefix}_center_l2_weight", 0.0)),
        f"{prefix}_axis_log": smooth_l1_with_mask(axis_log_pred, axis_log_target, weights)
        * float(loss_cfg.get(f"{prefix}_axis_log_weight", 0.0)),
        f"{prefix}_angle_cos": weighted_reduce(1.0 - angle_cos, weights)
        * float(loss_cfg.get(f"{prefix}_angle_cos_weight", 0.0)),
    }


def track_state_simdr_target_positions(
    target_state: torch.Tensor,
    *,
    bins: int,
    coordinate_max: float = 255.0,
) -> torch.Tensor:
    bins = max(2, int(bins))
    coord_max = max(float(coordinate_max), 1.0e-6)
    return target_state[:, :2].float().clamp(0.0, coord_max) * float(bins - 1) / coord_max


def track_state_simdr_losses(
    *,
    logits: torch.Tensor,
    target_state: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    weight = float(loss_cfg.get("track_state_simdr_weight", 0.0))
    bins = int(loss_cfg.get("track_state_simdr_bins", logits.shape[-1]))
    coordinate_max = float(loss_cfg.get("track_state_simdr_coordinate_max", 255.0))
    sigma = float(loss_cfg.get("track_state_simdr_sigma", 1.5))
    weights = quality * track_geom
    positions = track_state_simdr_target_positions(target_state, bins=bins, coordinate_max=coordinate_max).to(device=logits.device, dtype=logits.dtype)
    if logits.shape[-1] != bins:
        raise ValueError(f"track_state_simdr_bins={bins} does not match logits bins={logits.shape[-1]}")
    if sigma > 0.0:
        grid = torch.arange(bins, device=logits.device, dtype=logits.dtype).view(1, 1, bins)
        target = torch.exp(-0.5 * ((grid - positions.unsqueeze(-1)) / max(sigma, 1.0e-6)).pow(2))
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
        per_axis = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    else:
        target_idx = positions.round().long().clamp(0, bins - 1)
        per_axis = F.cross_entropy(logits.reshape(-1, bins), target_idx.reshape(-1), reduction="none").view(logits.shape[0], 2)
    return {
        "track_state_simdr": weighted_reduce(per_axis, weights) * weight,
    }


def track_center_heatmap_losses(
    *,
    logits: torch.Tensor,
    offset: torch.Tensor,
    target_state: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    heatmap_weight = float(loss_cfg.get("track_center_heatmap_weight", 0.0))
    offset_weight = float(loss_cfg.get("track_center_heatmap_offset_weight", 0.0))
    coordinate_max = max(float(loss_cfg.get("track_center_heatmap_coordinate_max", 255.0)), 1.0e-6)
    weights = quality * track_geom
    batch, height, width = logits.shape
    xy = target_state[:, :2].float().clamp(0.0, coordinate_max)
    gx = (xy[:, 0] / coordinate_max * float(width)).clamp(0.0, float(width) - 1.0e-6)
    gy = (xy[:, 1] / coordinate_max * float(height)).clamp(0.0, float(height) - 1.0e-6)
    ix = gx.floor().long().clamp(0, width - 1)
    iy = gy.floor().long().clamp(0, height - 1)
    target_index = iy * width + ix

    ce = F.cross_entropy(logits.reshape(batch, height * width), target_index, reduction="none")
    pred_offset = offset.permute(0, 2, 3, 1).reshape(batch, height * width, 2)
    gather_index = target_index.view(batch, 1, 1).expand(batch, 1, 2)
    pred_offset = pred_offset.gather(1, gather_index).squeeze(1)
    target_offset = torch.stack([gx - (ix.float() + 0.5), gy - (iy.float() + 0.5)], dim=-1).to(device=offset.device, dtype=offset.dtype)
    offset_loss = F.smooth_l1_loss(pred_offset, target_offset, reduction="none").sum(dim=-1)

    return {
        "track_center_heatmap": weighted_reduce(ce, weights) * heatmap_weight,
        "track_center_heatmap_offset": weighted_reduce(offset_loss, weights) * offset_weight,
    }


def track_center_refine_losses(
    *,
    delta: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    weights = quality * track_geom
    delta_l2_weight = float(loss_cfg.get("track_center_refine_delta_l2_weight", 0.0))
    delta_l1_weight = float(loss_cfg.get("track_center_refine_delta_l1_weight", 0.0))
    delta_norm = torch.linalg.norm(delta, dim=-1)
    return {
        "track_center_refine_delta_l2": weighted_reduce(delta_norm.pow(2), weights) * delta_l2_weight,
        "track_center_refine_delta_l1": weighted_reduce(delta_norm, weights) * delta_l1_weight,
    }


def track_center_candidate_losses(
    *,
    candidate_xy: torch.Tensor,
    candidate_logits: torch.Tensor,
    candidate_delta: torch.Tensor,
    target_state: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    weights = quality * track_geom
    margin = float(loss_cfg.get("track_center_candidate_p10_margin_px", 10.0))
    temperature = max(float(loss_cfg.get("track_center_candidate_temperature_px", 1.0)), 1.0e-6)
    error = torch.linalg.norm(candidate_xy - target_state[:, None, :2], dim=-1)
    labels = (error <= margin).to(dtype=candidate_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(candidate_logits, labels, reduction="none")
    min_error = error.min(dim=1).values
    min_threshold = F.softplus((min_error - margin) / temperature)
    delta_norm = torch.linalg.norm(candidate_delta, dim=-1)
    return {
        "track_center_candidate_p10_bce": weighted_reduce(bce, weights) * float(loss_cfg.get("track_center_candidate_p10_bce_weight", 0.0)),
        "track_center_candidate_min_soft_threshold": weighted_reduce(min_threshold, weights)
        * float(loss_cfg.get("track_center_candidate_min_soft_threshold_weight", 0.0)),
        "track_center_candidate_delta_l2": weighted_reduce(delta_norm.pow(2), weights) * float(loss_cfg.get("track_center_candidate_delta_l2_weight", 0.0)),
    }
