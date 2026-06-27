from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from .io import xyabuv_to_xywht
from .metrics import geom_mask, resolve_event_state, resolve_search_state, resolve_track_state, sample_quality, track_mask, weighted_mean


def smooth_l1_with_mask(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    return weighted_mean(loss, weights)


def sigmoid_bce_with_mask(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return weighted_mean(loss, weights)


def trig_l2_loss(pred_uv: torch.Tensor, target_uv: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_uv = F.normalize(pred_uv, dim=-1, eps=1e-6)
    target_uv = F.normalize(target_uv, dim=-1, eps=1e-6)
    loss = torch.sum((pred_uv - target_uv) ** 2, dim=-1)
    return weighted_mean(loss, weights)


def state6_to_xywht(state6: torch.Tensor) -> torch.Tensor:
    return xyabuv_to_xywht(state6)


def _ellipse_covariance(state6: torch.Tensor) -> torch.Tensor:
    xywht = state6_to_xywht(state6)
    a = torch.clamp(xywht[..., 2], min=1e-4)
    b = torch.clamp(xywht[..., 3], min=1e-4)
    theta = xywht[..., 4]
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    aa = a * a
    bb = b * b
    sigma_xx = aa * cos_t * cos_t + bb * sin_t * sin_t
    sigma_yy = aa * sin_t * sin_t + bb * cos_t * cos_t
    sigma_xy = (aa - bb) * sin_t * cos_t
    row0 = torch.stack([sigma_xx, sigma_xy], dim=-1)
    row1 = torch.stack([sigma_xy, sigma_yy], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _matrix_sqrt_psd(mat: torch.Tensor) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(mat)
    evals = torch.clamp(evals, min=0.0)
    return evecs @ torch.diag_embed(torch.sqrt(evals)) @ evecs.transpose(-1, -2)


def ellipse_gwd_loss(pred_state6: torch.Tensor, target_state6: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    pred_xywht = state6_to_xywht(pred_state6)
    target_xywht = state6_to_xywht(target_state6)
    center_term = torch.sum((pred_xywht[:, :2] - target_xywht[:, :2]) ** 2, dim=-1)
    pred_cov = _ellipse_covariance(pred_state6)
    target_cov = _ellipse_covariance(target_state6)
    target_sqrt = _matrix_sqrt_psd(target_cov)
    cross = _matrix_sqrt_psd(target_sqrt @ pred_cov @ target_sqrt)
    cov_term = (
        torch.diagonal(pred_cov, dim1=-2, dim2=-1).sum(dim=-1)
        + torch.diagonal(target_cov, dim1=-2, dim2=-1).sum(dim=-1)
        - 2.0 * torch.diagonal(cross, dim1=-2, dim2=-1).sum(dim=-1)
    )
    loss = torch.sqrt(torch.clamp(center_term + cov_term, min=1e-9))
    return weighted_mean(loss, weights)


def mask_bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3))
    probs = torch.sigmoid(logits)
    intersection = torch.sum(probs * target, dim=(1, 2, 3))
    union = torch.sum(probs, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    dice = 1.0 - (2.0 * intersection + 1.0) / (union + 1.0)
    return weighted_mean(bce + dice, weights)


def constraint_center_loss(pred_state: torch.Tensor, target_center: torch.Tensor, *, radius: float, weights: torch.Tensor | None = None) -> torch.Tensor:
    distance = torch.linalg.norm(pred_state[:, :2] - target_center, dim=-1)
    overflow = F.relu(distance - float(radius))
    return weighted_mean(overflow, weights)


def _pupil_branch_losses(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    target_state: torch.Tensor,
    weights: torch.Tensor,
    xy_weight: float,
    ab_weight: float,
    trig_weight: float,
    geo_weight: float,
    conf_weight: float,
    prefix: str,
) -> Dict[str, torch.Tensor]:
    return {
        f"{prefix}_xy": smooth_l1_with_mask(pred[:, 0:2], target[:, 0:2], weights) * float(xy_weight),
        f"{prefix}_ab": smooth_l1_with_mask(pred[:, 2:4], target[:, 2:4], weights) * float(ab_weight),
        f"{prefix}_trig": trig_l2_loss(pred[:, 4:6], target[:, 4:6], weights) * float(trig_weight),
        f"{prefix}_geo": ellipse_gwd_loss(state, target_state, weights) * float(geo_weight),
        f"{prefix}_conf": sigmoid_bce_with_mask(pred[:, 6], target[:, 6], weights) * float(conf_weight),
    }


def compute_stage1_losses(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], loss_cfg: Dict) -> Dict[str, torch.Tensor]:
    quality = sample_quality(batch)
    geom = geom_mask(batch)
    weights = quality * geom
    search_state = resolve_search_state(outputs)
    search_losses = _pupil_branch_losses(
        pred=outputs["search/pupil"],
        target=batch["pupil_search_target"],
        state=search_state,
        target_state=batch["cur_state"],
        weights=weights,
        xy_weight=loss_cfg.get("search_xy_weight", 1.0),
        ab_weight=loss_cfg.get("search_ab_weight", 0.5),
        trig_weight=loss_cfg.get("search_trig_weight", 1.0),
        geo_weight=loss_cfg.get("search_geo_weight", 0.5),
        conf_weight=loss_cfg.get("search_conf_weight", 0.1),
        prefix="search",
    )
    eye_weights = quality
    eye_loss = (
        smooth_l1_with_mask(outputs["search/eye"][:, 0:4], batch["eye_target"][:, 0:4], eye_weights)
        + sigmoid_bce_with_mask(outputs["search/eye"][:, 4], batch["eye_target"][:, 4], eye_weights)
        * float(loss_cfg.get("eye_conf_weight", 0.1))
    ) * float(loss_cfg.get("eye_weight", 1.0))
    mask_loss = mask_bce_dice_loss(outputs["search/mask_logits"], batch["mask_target"], weights) * float(loss_cfg.get("mask_weight", 1.0))
    aux_weight = float(loss_cfg.get("aux_weight", 0.0))
    aux_loss = torch.zeros((), device=batch["frame"].device)
    if aux_weight > 0.0 and "search/aux" in outputs:
        aux_loss = F.cross_entropy(outputs["search/aux"], batch["aux_target"], reduction="none")
        aux_loss = weighted_mean(aux_loss, quality) * aux_weight
    center_loss = constraint_center_loss(
        search_state,
        batch["constraint_center"],
        radius=float(loss_cfg.get("constraint_center_radius", 24.0)),
        weights=weights,
    ) * float(loss_cfg.get("constraint_center_weight", 0.0))
    losses = {
        "loss_eye": eye_loss,
        "loss_mask": mask_loss,
        "loss_constraint_center": center_loss,
        "loss_aux": aux_loss,
        **{f"loss_{name}": value for name, value in search_losses.items()},
    }
    losses["loss_total"] = torch.stack(list(losses.values())).sum()
    return losses


def compute_stage2_losses(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], loss_cfg: Dict) -> Dict[str, torch.Tensor]:
    quality = sample_quality(batch)
    geom = geom_mask(batch)
    track_geom = track_mask(batch)
    search_state = resolve_search_state(outputs)
    event_state = resolve_event_state(outputs)
    track_state = resolve_track_state(batch, outputs)

    search_losses = _pupil_branch_losses(
        pred=outputs["search/pupil"],
        target=batch["pupil_search_target"],
        state=search_state,
        target_state=batch["cur_state"],
        weights=quality * geom,
        xy_weight=loss_cfg.get("search_xy_weight", 1.0),
        ab_weight=loss_cfg.get("search_ab_weight", 0.5),
        trig_weight=loss_cfg.get("search_trig_weight", 1.0),
        geo_weight=loss_cfg.get("search_geo_weight", 0.5),
        conf_weight=loss_cfg.get("search_conf_weight", 0.1),
        prefix="search",
    )
    event_losses = _pupil_branch_losses(
        pred=outputs["event/pupil"],
        target=batch["pupil_search_target"],
        state=event_state,
        target_state=batch["cur_state"],
        weights=quality * geom,
        xy_weight=loss_cfg.get("event_xy_weight", 1.0),
        ab_weight=loss_cfg.get("event_ab_weight", 0.5),
        trig_weight=loss_cfg.get("event_trig_weight", 1.0),
        geo_weight=loss_cfg.get("event_geo_weight", 0.5),
        conf_weight=loss_cfg.get("event_conf_weight", 0.1),
        prefix="event",
    )

    track_pred = outputs["track/pupil"]
    track_target = batch["pupil_track_target"]
    track_losses = {
        "track_xy": smooth_l1_with_mask(track_pred[:, 0:2], track_target[:, 0:2], quality * track_geom) * float(loss_cfg.get("track_xy_weight", 1.0)),
        "track_ab": smooth_l1_with_mask(track_pred[:, 2:4], track_target[:, 2:4], quality * track_geom) * float(loss_cfg.get("track_ab_weight", 0.5)),
        "track_trig": smooth_l1_with_mask(track_pred[:, 4:6], track_target[:, 4:6], quality * track_geom) * float(loss_cfg.get("track_trig_weight", 1.0)),
        "track_geo": ellipse_gwd_loss(track_state, batch["cur_state"], quality * track_geom) * float(loss_cfg.get("track_geo_weight", 0.5)),
        "track_conf": sigmoid_bce_with_mask(track_pred[:, 6], track_target[:, 6], quality * batch["valid_track"].view(-1))
        * float(loss_cfg.get("track_conf_weight", 0.1)),
        "track_quality": sigmoid_bce_with_mask(track_pred[:, 7], track_target[:, 7], quality) * float(loss_cfg.get("track_quality_weight", 0.1)),
    }
    consistency = ellipse_gwd_loss(search_state, track_state, quality * track_geom) * float(loss_cfg.get("consistency_weight", 0.0))
    center_loss = constraint_center_loss(
        track_state,
        batch["constraint_center"],
        radius=float(loss_cfg.get("constraint_center_radius", 24.0)),
        weights=quality * track_geom,
    ) * float(loss_cfg.get("constraint_center_weight", 0.0))
    aux_weight = float(loss_cfg.get("aux_weight", 0.0))
    aux_loss = torch.zeros((), device=batch["frame"].device)
    if aux_weight > 0.0 and "event/aux" in outputs:
        aux_loss = F.cross_entropy(outputs["event/aux"], batch["aux_target"], reduction="none")
        aux_loss = weighted_mean(aux_loss, quality) * aux_weight

    losses = {
        "loss_constraint_center": center_loss,
        "loss_consistency": consistency,
        "loss_aux": aux_loss,
        **{f"loss_{name}": value for name, value in search_losses.items()},
        **{f"loss_{name}": value for name, value in event_losses.items()},
        **{f"loss_{name}": value for name, value in track_losses.items()},
    }
    losses["loss_total"] = torch.stack(list(losses.values())).sum()
    return losses
