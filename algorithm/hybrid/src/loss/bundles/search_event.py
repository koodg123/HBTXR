from __future__ import annotations

from collections.abc import Callable
from typing import Dict

import torch

from src.loss.common import ellipse_gwd_loss, sigmoid_bce_with_mask, smooth_l1_with_mask, trig_l2_loss


def pupil_branch_losses(
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


def pupil_bbox_aux_losses(
    *,
    pred: torch.Tensor,
    target_box: torch.Tensor,
    weights: torch.Tensor,
    weight: float,
    conf_weight: float,
    mode: str,
    prefix: str,
    axis_bbox_loss_fn: Callable[..., torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if weight <= 0.0:
        zero = pred.new_zeros(())
        return {f"{prefix}_bbox_aux": zero, f"{prefix}_bbox_aux_conf": zero}
    box_loss = axis_bbox_loss_fn(pred[:, :4], target_box, weights, mode=mode) * float(weight)
    conf_loss = sigmoid_bce_with_mask(pred[:, 4], torch.ones_like(pred[:, 4]), weights) * float(conf_weight)
    return {f"{prefix}_bbox_aux": box_loss, f"{prefix}_bbox_aux_conf": conf_loss}


def pupil_obb_aux_losses(
    *,
    pred: torch.Tensor,
    target_box: torch.Tensor,
    weights: torch.Tensor,
    weight: float,
    angle_weight: float,
    conf_weight: float,
    mode: str,
    mgiou_fast_mode: bool,
    prefix: str,
    rotated_overlap_loss_fn: Callable[..., torch.Tensor],
    rotated_angle_loss_fn: Callable[..., torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if weight <= 0.0:
        zero = pred.new_zeros(())
        return {f"{prefix}_obb_aux": zero, f"{prefix}_obb_aux_angle": zero, f"{prefix}_obb_aux_conf": zero}
    overlap_loss = rotated_overlap_loss_fn(pred, target_box, weights, mode=mode, fast_mode=mgiou_fast_mode) * float(weight)
    angle_loss = rotated_angle_loss_fn(pred[:, 4], target_box[:, 4], weights) * float(angle_weight)
    conf_loss = sigmoid_bce_with_mask(pred[:, 5], torch.ones_like(pred[:, 5]), weights) * float(conf_weight)
    return {f"{prefix}_obb_aux": overlap_loss, f"{prefix}_obb_aux_angle": angle_loss, f"{prefix}_obb_aux_conf": conf_loss}
