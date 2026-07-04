from __future__ import annotations

from typing import Dict

import torch

from src.loss.common import ellipse_gwd_loss, sigmoid_bce_with_mask, smooth_l1_with_mask


def track_branch_losses(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    state: torch.Tensor,
    target_state: torch.Tensor,
    quality: torch.Tensor,
    track_geom: torch.Tensor,
    valid_track: torch.Tensor,
    loss_cfg: Dict,
) -> Dict[str, torch.Tensor]:
    weights = quality * track_geom
    return {
        "track_xy": smooth_l1_with_mask(pred[:, 0:2], target[:, 0:2], weights) * float(loss_cfg.get("track_xy_weight", 1.0)),
        "track_ab": smooth_l1_with_mask(pred[:, 2:4], target[:, 2:4], weights) * float(loss_cfg.get("track_ab_weight", 0.5)),
        "track_trig": smooth_l1_with_mask(pred[:, 4:6], target[:, 4:6], weights) * float(loss_cfg.get("track_trig_weight", 1.0)),
        "track_geo": ellipse_gwd_loss(state, target_state, weights) * float(loss_cfg.get("track_geo_weight", 0.5)),
        "track_conf": sigmoid_bce_with_mask(pred[:, 6], target[:, 6], quality * valid_track.view(-1)) * float(loss_cfg.get("track_conf_weight", 0.1)),
        "track_quality": sigmoid_bce_with_mask(pred[:, 7], target[:, 7], quality) * float(loss_cfg.get("track_quality_weight", 0.1)),
    }
