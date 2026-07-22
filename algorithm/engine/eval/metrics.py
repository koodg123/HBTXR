"""Evaluation metrics aligned to the reimplemented-model contract (P5.7).

Contract-native metrics that work for every modality without depending on the
old detector output format:

- ``center_distance``: mean L2 between predicted and GT pupil centers (state[:2]).
- ``mask_iou``: soft->hard IoU of the predicted pupil mask vs the GT mask.

Richer geometric metrics (ellipse IoU / p-acc / AP) live in utils.metrics and can
be layered on top once the eval output format is pinned per experiment.
"""
from __future__ import annotations

import torch


def center_distance(pred_state: torch.Tensor, gt_state: torch.Tensor) -> float:
    """Mean L2 distance between predicted and GT centers (state ``[..., :2]``)."""
    diff = pred_state[..., :2] - gt_state[..., :2]
    return float(torch.linalg.norm(diff, dim=-1).mean())


def mask_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, *, threshold: float = 0.5) -> float:
    """Mean IoU of the thresholded predicted mask vs the GT mask."""
    prob = torch.sigmoid(pred_mask) if pred_mask.dtype.is_floating_point else pred_mask.float()
    pred = (prob >= threshold).float()
    target = (gt_mask >= threshold).float()
    dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims)
    union = pred.sum(dim=dims) + target.sum(dim=dims) - inter
    iou = (inter + 1.0) / (union + 1.0)
    return float(iou.mean())


__all__ = ["center_distance", "mask_iou"]
