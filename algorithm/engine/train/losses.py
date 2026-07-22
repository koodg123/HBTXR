"""Training losses for the reimplemented HBTXR models (P5.4).

Loss terms are matched to the new heads (dict outputs of models.frame / event /
hybrid / mask):

- box         : oriented-box regression (smooth L1 on x, y, w, h, r) — search / direct head.
- ellipse     : ellipse-state / residual regression (smooth L1 on x, y, a, b, theta).
- mask        : BCE + soft-Dice on the dense pupil mask.
- reliability : BCE on (confidence, IoU-quality) vs targets.
- eye_box     : eye-region ROI-guidance regression (smooth L1 on x, y, w, h) — g_eye head.

``compute_losses(outputs, targets, weights)`` assembles the terms that are present
in both ``outputs`` and ``targets`` into a weighted total. Heavier geometric
losses (GWD, CIoU) live in common.losses and can be swapped in per term.
"""
from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

DEFAULT_WEIGHTS: dict[str, float] = {
    "box": 1.0,
    "ellipse": 1.0,
    "state": 0.0,
    "mask": 1.0,
    "reliability": 0.5,
    "eye_box": 0.5,
}


def box_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def ellipse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target)


def mask_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    inter = (prob * target).sum(dim=dims)
    union = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)
    return bce + dice.mean()


def reliability_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy(pred, target)


def compute_losses(
    outputs: dict[str, Any],
    targets: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Assemble the active loss terms and their weighted total."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    losses: dict[str, torch.Tensor] = {}
    if "box" in outputs and "box" in targets:
        losses["box"] = box_loss(outputs["box"], targets["box"])
    if "residual" in outputs and "residual" in targets:
        losses["ellipse"] = ellipse_loss(outputs["residual"], targets["residual"])
    if "state" in outputs and "state" in targets and w.get("state", 0.0) > 0:
        losses["state"] = ellipse_loss(outputs["state"], targets["state"])
    if "mask" in outputs and "mask" in targets:
        losses["mask"] = mask_loss(outputs["mask"], targets["mask"])
    if "reliability" in outputs and "reliability" in targets:
        losses["reliability"] = reliability_loss(outputs["reliability"], targets["reliability"])
    if "eye_box" in outputs and "eye_box" in targets:
        losses["eye_box"] = box_loss(outputs["eye_box"], targets["eye_box"])

    if losses:
        total = sum(w.get(name, 1.0) * value for name, value in losses.items())
    else:
        total = torch.zeros((), requires_grad=True)
    losses["total"] = total
    return losses
