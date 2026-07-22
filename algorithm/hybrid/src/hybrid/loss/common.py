from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from hybrid.utils.state6 import xyabuv_to_xywht


def weighted_reduce(loss: torch.Tensor, weight: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    if loss.ndim > 1:
        loss = loss.mean(dim=tuple(range(1, loss.ndim)))
    if weight is None:
        return loss.mean()
    w = weight.to(device=loss.device, dtype=loss.dtype).view(-1)
    return (loss * w).sum() / w.sum().clamp_min(eps)


def normalize_uv(uv: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.normalize(uv, dim=-1, eps=eps)


def uv_to_theta(uv: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    uv = normalize_uv(uv, eps)
    return 0.5 * torch.atan2(uv[..., 0], uv[..., 1])


def decode_track_state(prev_state: torch.Tensor, track_pred: torch.Tensor) -> torch.Tensor:
    x0, y0, a0, b0, u0, v0 = prev_state.unbind(dim=-1)
    dx, dy, dloga, dlogb, du, dv = track_pred[..., :6].unbind(dim=-1)
    xy = torch.stack([x0 + dx, y0 + dy], dim=-1)
    ab = torch.stack([a0 * torch.exp(dloga), b0 * torch.exp(dlogb)], dim=-1)
    uv = normalize_uv(torch.stack([u0 + du, v0 + dv], dim=-1))
    return torch.cat([xy, ab, uv], dim=-1)


def state_to_covariance(state: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    x, y, a, b, u, v = state.unbind(dim=-1)
    a = a.clamp_min(eps)
    b = b.clamp_min(eps)
    theta = uv_to_theta(torch.stack([u, v], dim=-1), eps=eps)
    c = torch.cos(theta)
    s = torch.sin(theta)
    a2 = a * a
    b2 = b * b
    s11 = c * c * a2 + s * s * b2
    s12 = c * s * (a2 - b2)
    s22 = s * s * a2 + c * c * b2
    mu = torch.stack([x, y], dim=-1)
    sigma = torch.stack([torch.stack([s11, s12], -1), torch.stack([s12, s22], -1)], -2)
    return mu, sigma


def mat_sqrt_2x2(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    a = mat[..., 0, 0]
    b = 0.5 * (mat[..., 0, 1] + mat[..., 1, 0])
    d = mat[..., 1, 1]
    det_sqrt = torch.sqrt((a * d - b * b).clamp_min(eps))
    denom = torch.sqrt((a + d + 2.0 * det_sqrt).clamp_min(eps))
    out00 = (a + det_sqrt) / denom
    out01 = b / denom
    out11 = (d + det_sqrt) / denom
    return torch.stack(
        [torch.stack([out00, out01], dim=-1), torch.stack([out01, out11], dim=-1)],
        dim=-2,
    )


def ellipse_gwd_like(pred_state: torch.Tensor, tgt_state: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mu1, sigma1 = state_to_covariance(pred_state, eps)
    mu2, sigma2 = state_to_covariance(tgt_state, eps)
    d_mu = ((mu1 - mu2) ** 2).sum(dim=-1)
    d_cov = ((mat_sqrt_2x2(sigma1, eps) - mat_sqrt_2x2(sigma2, eps)) ** 2).sum(dim=(-2, -1))
    return d_mu + d_cov


def box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def box_xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = x1 + w / 2
    cy = y1 + h / 2
    return torch.stack([cx, cy, w, h], dim=-1)


def box_xywht_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    return box_cxcywh_to_xyxy(box[..., :4])


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    x11, y11, x12, y12 = box1.unbind(-1)
    x21, y21, x22, y22 = box2.unbind(-1)
    xi1, yi1 = torch.max(x11, x21), torch.max(y11, y21)
    xi2, yi2 = torch.min(x12, x22), torch.min(y12, y22)
    inter = (xi2 - xi1).clamp_min(0) * (yi2 - yi1).clamp_min(0)
    area1 = (x12 - x11).clamp_min(0) * (y12 - y11).clamp_min(0)
    area2 = (x22 - x21).clamp_min(0) * (y22 - y21).clamp_min(0)
    union = area1 + area2 - inter + eps
    return inter / union, union


def build_sample_weight(
    annotation_quality: Optional[torch.Tensor] = None,
    closed_eye_flag: Optional[torch.Tensor] = None,
    valid_flag: Optional[torch.Tensor] = None,
    *,
    allow_closed_eye: bool = False,
) -> Optional[torch.Tensor]:
    weight = annotation_quality.float().clamp(0.0, 1.0) if annotation_quality is not None else None
    if closed_eye_flag is not None and not allow_closed_eye:
        closed = 1.0 - closed_eye_flag.float().clamp(0.0, 1.0)
        weight = closed if weight is None else weight * closed
    if valid_flag is not None:
        valid = valid_flag.float().clamp(0.0, 1.0)
        weight = valid if weight is None else weight * valid
    return weight


def _sample_quality(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["annotation_quality"].view(-1).to(dtype=torch.float32)


def _geom_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return batch["mask_valid"].view(-1).float() * (1.0 - batch["closed_eye_flag"].view(-1).float())


def _track_mask(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return _geom_mask(batch) * batch["valid_track"].view(-1).float()


def smooth_l1_with_mask(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_reduce(F.smooth_l1_loss(pred, target, reduction="none"), weights)


def sigmoid_bce_with_mask(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_reduce(F.binary_cross_entropy_with_logits(logits, target, reduction="none"), weights)


def trig_l2_loss(pred_uv: torch.Tensor, target_uv: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_reduce((normalize_uv(pred_uv) - normalize_uv(target_uv)) ** 2, weights)


def state6_to_xywht(state6: torch.Tensor) -> torch.Tensor:
    return xyabuv_to_xywht(state6)


def ellipse_gwd_loss(pred_state6: torch.Tensor, target_state6: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    return weighted_reduce(torch.sqrt(torch.clamp(ellipse_gwd_like(pred_state6, target_state6), min=1e-9)), weights)


def mask_bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(dim=(1, 2, 3))
    prob = torch.sigmoid(logits)
    inter = (prob * target).flatten(1).sum(dim=-1)
    denom = prob.flatten(1).sum(dim=-1) + target.flatten(1).sum(dim=-1)
    dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
    return weighted_reduce(bce + dice, weights)


__all__ = [
    "_geom_mask",
    "_sample_quality",
    "_track_mask",
    "box_cxcywh_to_xyxy",
    "box_xywht_to_xyxy",
    "box_xyxy_to_cxcywh",
    "box_iou_xyxy",
    "build_sample_weight",
    "decode_track_state",
    "ellipse_gwd_like",
    "ellipse_gwd_loss",
    "mask_bce_dice_loss",
    "mat_sqrt_2x2",
    "normalize_uv",
    "sigmoid_bce_with_mask",
    "smooth_l1_with_mask",
    "state6_to_xywht",
    "state_to_covariance",
    "trig_l2_loss",
    "uv_to_theta",
    "weighted_reduce",
]
