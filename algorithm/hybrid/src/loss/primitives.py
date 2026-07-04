from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss.common import build_sample_weight, box_cxcywh_to_xyxy, box_iou_xyxy, decode_track_state, ellipse_gwd_like, mask_bce_dice_loss, normalize_uv, weighted_reduce


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0, alpha: float = 2.1) -> None:
        super().__init__()
        self.omega, self.theta, self.epsilon, self.alpha = omega, theta, epsilon, alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = (target - pred).abs()
        alpha_t = self.alpha - target
        ratio = self.theta / self.epsilon
        a_term = self.omega * (1.0 / (1.0 + ratio**alpha_t)) * alpha_t * (ratio ** (alpha_t - 1.0)) / self.epsilon
        c_term = self.theta * a_term - self.omega * torch.log1p(ratio**alpha_t)
        loss = torch.where(delta < self.theta, self.omega * torch.log1p((delta / self.epsilon) ** alpha_t), a_term * delta - c_term)
        return weighted_reduce(loss, weight)


class ConfidenceWeightedMSELoss(nn.Module):
    def __init__(self, lambda_reg: float = 0.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.lambda_reg, self.eps = lambda_reg, eps

    def forward(self, pred_xy: torch.Tensor, target_xy: torch.Tensor, conf: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        mse = ((pred_xy - target_xy) ** 2).sum(dim=-1)
        conf_prob = conf if conf.min() >= 0 and conf.max() <= 1 else torch.sigmoid(conf)
        loss = conf_prob * mse
        if self.lambda_reg > 0:
            loss = loss - self.lambda_reg * torch.log(conf_prob.clamp_min(self.eps))
        return weighted_reduce(loss, weight)


class TemporalDisplacementLoss(nn.Module):
    def forward(self, prev_xy: torch.Tensor, delta_xy: torch.Tensor, target_xy: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(((prev_xy + delta_xy - target_xy) ** 2).sum(dim=-1), weight)


class EventDensityWeighting(nn.Module):
    def __init__(self, max_weight: Optional[float] = None) -> None:
        super().__init__()
        self.max_weight = max_weight

    def forward(self, event_density: torch.Tensor) -> torch.Tensor:
        weight = torch.log1p(event_density.float().clamp_min(0.0))
        return torch.clamp(weight, max=self.max_weight) if self.max_weight is not None else weight


class AnchorConsistencyLoss(nn.Module):
    def forward(self, event_state: torch.Tensor, anchor_state: torch.Tensor, anchor_decay: Optional[torch.Tensor] = None, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = (event_state - anchor_state).abs().mean(dim=-1)
        if anchor_decay is not None:
            loss = loss * anchor_decay.float()
        return weighted_reduce(loss, weight)


class CIoULoss(nn.Module):
    def forward(self, pred_box: torch.Tensor, target_box: torch.Tensor, xyxy: bool = True, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not xyxy:
            pred_box = box_cxcywh_to_xyxy(pred_box)
            target_box = box_cxcywh_to_xyxy(target_box)
        iou, _ = box_iou_xyxy(pred_box, target_box)
        px1, py1, px2, py2 = pred_box.unbind(-1)
        tx1, ty1, tx2, ty2 = target_box.unbind(-1)
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
        rho2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
        cx1, cy1 = torch.min(px1, tx1), torch.min(py1, ty1)
        cx2, cy2 = torch.max(px2, tx2), torch.max(py2, ty2)
        c2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2 + 1e-6
        pw, ph = (px2 - px1).clamp_min(1e-6), (py2 - py1).clamp_min(1e-6)
        tw, th = (tx2 - tx1).clamp_min(1e-6), (ty2 - ty1).clamp_min(1e-6)
        v = (4 / math.pi**2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
        alpha = v / (1 - iou + v + 1e-6)
        return weighted_reduce(1 - iou + rho2 / c2 + alpha * v, weight)


class GIoULoss(nn.Module):
    def forward(self, pred_box: torch.Tensor, target_box: torch.Tensor, xyxy: bool = True, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not xyxy:
            pred_box = box_cxcywh_to_xyxy(pred_box)
            target_box = box_cxcywh_to_xyxy(target_box)
        iou, union = box_iou_xyxy(pred_box, target_box)
        px1, py1, px2, py2 = pred_box.unbind(-1)
        tx1, ty1, tx2, ty2 = target_box.unbind(-1)
        cx1, cy1 = torch.min(px1, tx1), torch.min(py1, ty1)
        cx2, cy2 = torch.max(px2, tx2), torch.max(py2, ty2)
        area_c = (cx2 - cx1).clamp_min(0) * (cy2 - cy1).clamp_min(0) + 1e-6
        return weighted_reduce(1 - (iou - (area_c - union) / area_c), weight)


class MPDIoULoss(nn.Module):
    def forward(self, pred_box: torch.Tensor, target_box: torch.Tensor, xyxy: bool = True, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not xyxy:
            pred_box = box_cxcywh_to_xyxy(pred_box)
            target_box = box_cxcywh_to_xyxy(target_box)
        iou, _ = box_iou_xyxy(pred_box, target_box)
        px1, py1, px2, py2 = pred_box.unbind(-1)
        tx1, ty1, tx2, ty2 = target_box.unbind(-1)
        d1 = (px1 - tx1).pow(2) + (py1 - ty1).pow(2)
        d2 = (px2 - tx2).pow(2) + (py2 - ty2).pow(2)
        cx1, cy1 = torch.min(px1, tx1), torch.min(py1, ty1)
        cx2, cy2 = torch.max(px2, tx2), torch.max(py2, ty2)
        cw = (cx2 - cx1).clamp_min(1e-6)
        ch = (cy2 - cy1).clamp_min(1e-6)
        denom = cw.pow(2) + ch.pow(2) + 1e-6
        mpdiou = iou - d1 / denom - d2 / denom
        return weighted_reduce(1.0 - mpdiou, weight)


class MGIoU2DLoss(nn.Module):
    # Source: https://github.com/ldtho/MGIoU/blob/main/mgiou/losses.py
    # Upstream: main branch, MGIoU2D (arXiv:2504.16443)
    def __init__(self, representation: str = "rect", loss_weight: float = 1.0, fast_mode: bool = False) -> None:
        super().__init__()
        if representation not in {"rect", "corner"}:
            raise ValueError("representation must be 'rect' or 'corner'")
        self.representation = representation
        self.loss_weight = float(loss_weight)
        self.fast_mode = bool(fast_mode)
        self.register_buffer(
            "_unit_square",
            torch.tensor([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32),
        )

    def forward(self, pred_box: torch.Tensor, target_box: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.representation == "rect":
            if pred_box.shape != target_box.shape or pred_box.ndim != 2 or pred_box.shape[-1] != 5:
                raise ValueError(f"MGIoU2DLoss(rect) expects matching [B, 5] inputs, got {tuple(pred_box.shape)} and {tuple(target_box.shape)}")
            valid = target_box.abs().sum(dim=-1) > 0.0
            loss = pred_box.new_zeros(pred_box.shape[0])
            if (~valid).any():
                fallback = F.l1_loss(pred_box[~valid], target_box[~valid], reduction="none").sum(dim=-1)
                loss[~valid] = fallback
            if valid.any():
                pred_corners = self._rect_to_corners(pred_box[valid])
                target_corners = self._rect_to_corners(target_box[valid])
                loss[valid] = self._mgiou_boxes(pred_corners, target_corners)
        else:
            if pred_box.shape != target_box.shape or pred_box.ndim != 3 or pred_box.shape[-2:] != (4, 2):
                raise ValueError(f"MGIoU2DLoss(corner) expects matching [B, 4, 2] inputs, got {tuple(pred_box.shape)} and {tuple(target_box.shape)}")
            loss = self._mgiou_boxes(pred_box, target_box)
        return weighted_reduce(loss * self.loss_weight, weight)

    def _rect_to_corners(self, box: torch.Tensor) -> torch.Tensor:
        trans, wh, angle = box[:, :2], box[:, 2:4], box[:, 4]
        base = self._unit_square.unsqueeze(0).to(device=box.device, dtype=box.dtype) * (wh * 0.5).unsqueeze(1)
        cos_a, sin_a = angle.cos(), angle.sin()
        rot = torch.stack(
            (
                torch.stack((cos_a, -sin_a), dim=-1),
                torch.stack((sin_a, cos_a), dim=-1),
            ),
            dim=1,
        )
        return torch.bmm(base, rot) + trans.unsqueeze(1)

    @staticmethod
    def _rect_axes(corners: torch.Tensor) -> torch.Tensor:
        edge_x = corners[:, 1] - corners[:, 0]
        edge_y = corners[:, 3] - corners[:, 0]
        normal_x = torch.stack((-edge_x[:, 1], edge_x[:, 0]), dim=-1)
        normal_y = torch.stack((-edge_y[:, 1], edge_y[:, 0]), dim=-1)
        return torch.stack((normal_x, normal_y), dim=1)

    def _mgiou_boxes(self, corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
        axes = torch.cat((self._rect_axes(corners1), self._rect_axes(corners2)), dim=1)
        proj1 = corners1 @ axes.transpose(1, 2)
        proj2 = corners2 @ axes.transpose(1, 2)
        mn1, mx1 = proj1.min(dim=1).values, proj1.max(dim=1).values
        mn2, mx2 = proj2.min(dim=1).values, proj2.max(dim=1).values
        if self.fast_mode:
            numerator = torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)
            denominator = torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2) + 1e-6
            giou1d = numerator / denominator
        else:
            inter = (torch.minimum(mx1, mx2) - torch.maximum(mn1, mn2)).clamp(min=0.0)
            union = (mx1 - mn1) + (mx2 - mn2) - inter + 1e-6
            hull = torch.maximum(mx1, mx2) - torch.minimum(mn1, mn2) + 1e-6
            giou1d = inter / union - (hull - union) / hull
        return (1.0 - giou1d.mean(dim=-1)) * 0.5


class CenterLoss(nn.Module):
    def forward(self, pred_center: torch.Tensor, target_center: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(((pred_center - target_center) ** 2).sum(dim=-1), weight)


class CornerLoss(nn.Module):
    def __init__(self, mode: str = "heatmap") -> None:
        super().__init__()
        self.mode = mode

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = (pred - target) ** 2 if self.mode == "heatmap" else (pred - target).abs()
        return weighted_reduce(loss, weight)


class NLLHeatmapLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target_prob: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.dim() == 4 and logits.size(1) == 1:
            logits, target_prob = logits[:, 0], target_prob[:, 0]
        return weighted_reduce(-(target_prob.flatten(1) * F.log_softmax(logits.flatten(1), dim=-1)).sum(dim=-1), weight)


class GazeVelocityLoss(nn.Module):
    def forward(self, pred_prev: torch.Tensor, pred_cur: torch.Tensor, gt_prev: torch.Tensor, gt_cur: torch.Tensor, dt: torch.Tensor | float = 1.0, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=pred_cur.device, dtype=pred_cur.dtype)
        while dt.dim() < pred_cur.dim():
            dt = dt.unsqueeze(-1)
        return weighted_reduce((((pred_cur - pred_prev) / dt - (gt_cur - gt_prev) / dt) ** 2).sum(dim=-1), weight)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * target + (1 - prob) * (1 - target)
        alpha_factor = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_factor * (1 - pt).pow(self.gamma) * bce
        return weighted_reduce(loss, weight)


class VarifocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, target_score: torch.Tensor, label: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_prob = torch.sigmoid(logits)
        target_score = target_score.to(dtype=logits.dtype)
        label = label.to(dtype=logits.dtype)
        focal_weight = target_score * label + self.alpha * (pred_prob - target_score).abs().pow(self.gamma) * (1.0 - label)
        loss = F.binary_cross_entropy_with_logits(logits, target_score, reduction="none") * focal_weight
        return weighted_reduce(loss, weight)


class DistributionFocalLoss(nn.Module):
    def forward(self, pred_dist: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred_dist.ndim != 3:
            raise ValueError(f"DistributionFocalLoss expects [N, 4, reg_max], got {tuple(pred_dist.shape)}")
        reg_max = pred_dist.shape[-1]
        target = target.clamp(0.0, float(max(0, reg_max - 1)) - 1e-6)
        left = target.floor().long()
        right = (left + 1).clamp(max=reg_max - 1)
        wl = (right.to(dtype=target.dtype) - target).clamp_min(0.0)
        wr = (target - left.to(dtype=target.dtype)).clamp_min(0.0)
        logits = pred_dist.reshape(-1, reg_max)
        left_loss = F.cross_entropy(logits, left.reshape(-1), reduction="none").reshape_as(target)
        right_loss = F.cross_entropy(logits, right.reshape(-1), reduction="none").reshape_as(target)
        loss = left_loss * wl + right_loss * wr
        return weighted_reduce(loss, weight)


class CenterNetFocalLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 4.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        prob = torch.sigmoid(logits).clamp(self.eps, 1.0 - self.eps)
        pos_mask = (target >= 1.0 - 1e-6).to(dtype=prob.dtype)
        neg_mask = (target < 1.0 - 1e-6).to(dtype=prob.dtype)
        neg_weight = (1.0 - target).clamp_min(0.0).pow(self.beta)
        pos_loss = -(1.0 - prob).pow(self.alpha) * torch.log(prob) * pos_mask
        neg_loss = -(prob).pow(self.alpha) * torch.log(1.0 - prob) * neg_weight * neg_mask
        loss = pos_loss + neg_loss
        if logits.ndim > 2:
            loss = loss.flatten(1).sum(dim=-1)
            pos_count = pos_mask.flatten(1).sum(dim=-1)
            loss = loss / pos_count.clamp_min(1.0)
        return weighted_reduce(loss, weight)


class BCEDiceLoss(nn.Module):
    def forward(self, mask_logits: torch.Tensor, mask_target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return mask_bce_dice_loss(mask_logits, mask_target, weight)


class RotatedAngleLoss(nn.Module):
    def forward(self, pred_angle: torch.Tensor, target_angle: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        delta = torch.atan2(torch.sin(pred_angle - target_angle), torch.cos(pred_angle - target_angle)).abs()
        return weighted_reduce(delta, weight)


class RotatedBBoxLoss(nn.Module):
    def __init__(self, angle_weight: float = 0.5) -> None:
        super().__init__()
        self.angle_weight = float(angle_weight)
        self.angle_loss = RotatedAngleLoss()

    def forward(self, pred_box: torch.Tensor, target_box: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_state = torch.cat([pred_box[:, :4], torch.sin(2.0 * pred_box[:, 4:5]), torch.cos(2.0 * pred_box[:, 4:5])], dim=-1)
        target_state = torch.cat([target_box[:, :4], torch.sin(2.0 * target_box[:, 4:5]), torch.cos(2.0 * target_box[:, 4:5])], dim=-1)
        overlap = weighted_reduce(ellipse_gwd_like(pred_state, target_state), weight)
        angle = self.angle_loss(pred_box[:, 4], target_box[:, 4], weight)
        return overlap + angle * self.angle_weight


class FeatureConsistencyLoss(nn.Module):
    def forward(self, pred_feat: torch.Tensor, target_feat: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(1.0 - F.cosine_similarity(pred_feat, target_feat, dim=-1), weight)


class EventToFrameContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, event_feat: torch.Tensor, frame_feat: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = F.normalize(event_feat, dim=-1) @ F.normalize(frame_feat, dim=-1).T / self.temperature
        target = torch.arange(logits.size(0), device=logits.device)
        return weighted_reduce(F.cross_entropy(logits, target, reduction="none"), weight)


class GazeAngularLoss(nn.Module):
    def forward(self, pred_uv: torch.Tensor, target_uv: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        cos = (normalize_uv(pred_uv) * normalize_uv(target_uv)).sum(dim=-1).clamp(-1 + 1e-6, 1 - 1e-6)
        return weighted_reduce(torch.acos(cos), weight)


class PredictionKDLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student: torch.Tensor, teacher: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if student.dim() == 1:
            student = student.unsqueeze(-1)
            teacher = teacher.unsqueeze(-1)
        if student.size(-1) == 1:
            loss = F.binary_cross_entropy_with_logits(student.squeeze(-1), torch.sigmoid(teacher.squeeze(-1)), reduction="none")
        else:
            log_p = F.log_softmax(student / self.temperature, dim=-1)
            q = F.softmax(teacher / self.temperature, dim=-1)
            loss = F.kl_div(log_p, q, reduction="none").sum(dim=-1) * (self.temperature**2)
        return weighted_reduce(loss, weight)


class RelationalKDLoss(nn.Module):
    def forward(self, student_repr: torch.Tensor, teacher_repr: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if student_repr.size(0) < 2:
            return torch.zeros((), device=student_repr.device, dtype=student_repr.dtype)
        sdist = torch.pdist(student_repr, p=2)
        tdist = torch.pdist(teacher_repr, p=2)
        return weighted_reduce(F.smooth_l1_loss(sdist, tdist.detach(), reduction="none"), weight)


class CenterHeatmapLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target_heatmap: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if logits.dim() == 4 and logits.size(1) == 1:
            logits = logits[:, 0]
        if target_heatmap.dim() == 4 and target_heatmap.size(1) == 1:
            target_heatmap = target_heatmap[:, 0]
        return weighted_reduce(F.binary_cross_entropy_with_logits(logits, target_heatmap, reduction="none"), weight)


class CenterOffsetLoss(nn.Module):
    def forward(self, pred_offset: torch.Tensor, target_offset: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(F.smooth_l1_loss(pred_offset, target_offset, reduction="none"), weight)


class TrigRotationLoss(nn.Module):
    def forward(self, pred_uv: torch.Tensor, target_uv: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce((normalize_uv(pred_uv) - normalize_uv(target_uv)) ** 2, weight)


class EllipseOverlapLoss(nn.Module):
    def forward(self, pred_state: torch.Tensor, tgt_state: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(ellipse_gwd_like(pred_state, tgt_state), weight)


class ConfidenceAwareRelocalizationLoss(nn.Module):
    def __init__(self, relocalize_threshold: float = 0.5) -> None:
        super().__init__()
        self.relocalize_threshold = relocalize_threshold

    def forward(self, conf: torch.Tensor, target_flag: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        target = target_flag.float().clamp(0.0, 1.0)
        return weighted_reduce(F.binary_cross_entropy_with_logits(conf, target, reduction="none"), weight)


class DisplacementLoss(nn.Module):
    def forward(self, pred_delta: torch.Tensor, target_delta: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce((pred_delta - target_delta).abs(), weight)


class PropagatedStateLoss(nn.Module):
    def forward(self, prev_state: torch.Tensor, track_pred: torch.Tensor, cur_state: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce((decode_track_state(prev_state, track_pred) - cur_state).abs(), weight)


class SearchBranchLosses(nn.Module):
    def forward(self, search_pred: torch.Tensor, search_target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        pred_state, tgt_state = search_pred[:, :6], search_target[:, :6]
        return {
            "search_xy": weighted_reduce((pred_state[:, :2] - tgt_state[:, :2]).abs(), weight),
            "search_ab": weighted_reduce((pred_state[:, 2:4] - tgt_state[:, 2:4]).abs(), weight),
            "search_trig": weighted_reduce((normalize_uv(pred_state[:, 4:6]) - normalize_uv(tgt_state[:, 4:6])) ** 2, weight),
            "search_gwd": weighted_reduce(ellipse_gwd_like(pred_state, tgt_state), weight),
            "search_conf": weighted_reduce(F.binary_cross_entropy_with_logits(search_pred[:, 6], search_target[:, 6], reduction="none"), weight),
        }


class EventSearchBranchLosses(SearchBranchLosses):
    pass


class TrackBranchLosses(nn.Module):
    def forward(self, track_pred: torch.Tensor, track_target: torch.Tensor, prev_state: torch.Tensor, cur_state: torch.Tensor, weight: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        pred_res, tgt_res = track_pred[:, :6], track_target[:, :6]
        decoded = decode_track_state(prev_state, track_pred)
        return {
            "track_xy": weighted_reduce((pred_res[:, :2] - tgt_res[:, :2]).abs(), weight),
            "track_ab": weighted_reduce((pred_res[:, 2:4] - tgt_res[:, 2:4]).abs(), weight),
            "track_trig": weighted_reduce((pred_res[:, 4:6] - tgt_res[:, 4:6]) ** 2, weight),
            "track_gwd": weighted_reduce(ellipse_gwd_like(decoded, cur_state), weight),
            "track_conf": weighted_reduce(F.binary_cross_entropy_with_logits(track_pred[:, 6], track_target[:, 6], reduction="none"), weight),
            "track_quality": weighted_reduce(F.smooth_l1_loss(track_pred[:, 7], track_target[:, 7], reduction="none"), weight),
        }


class ConsistencyLoss(nn.Module):
    def forward(self, search_state: torch.Tensor, track_state: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce((search_state - track_state).abs(), weight)


class ConstraintCenterLoss(nn.Module):
    def __init__(self, radius: float = 24.0) -> None:
        super().__init__()
        self.radius = radius

    def forward(self, decoded_state: torch.Tensor, constraint_center: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(F.relu(torch.linalg.norm(decoded_state[:, :2] - constraint_center, dim=-1) - self.radius), weight)


class MaskBCELoss(nn.Module):
    def forward(self, mask_logits: torch.Tensor, mask_target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none").flatten(1), weight)


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, mask_logits: torch.Tensor, mask_target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        prob = torch.sigmoid(mask_logits)
        inter = (prob * mask_target).flatten(1).sum(dim=-1)
        denom = prob.flatten(1).sum(dim=-1) + mask_target.flatten(1).sum(dim=-1)
        return weighted_reduce(1.0 - (2.0 * inter + self.eps) / (denom + self.eps), weight)


class EyeRegionLoss(nn.Module):
    def forward(self, eye_pred: torch.Tensor, eye_target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
        return {
            "eye_bbox": weighted_reduce((eye_pred[:, :4] - eye_target[:, :4]).abs(), weight),
            "eye_conf": weighted_reduce(F.binary_cross_entropy_with_logits(eye_pred[:, 4], eye_target[:, 4], reduction="none"), weight),
        }


class AuxClassificationLoss(nn.Module):
    def forward(self, aux_logits: torch.Tensor, aux_target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        return weighted_reduce(F.cross_entropy(aux_logits, aux_target, reduction="none"), weight)


class HBTXRStageLoss(nn.Module):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights
        self.eye_loss = EyeRegionLoss()
        self.mask_bce = MaskBCELoss()
        self.dice = DiceLoss()
        self.search_losses = SearchBranchLosses()
        self.event_losses = EventSearchBranchLosses()
        self.track_losses = TrackBranchLosses()
        self.consistency = ConsistencyLoss()
        self.constraint = ConstraintCenterLoss(radius=float(weights.get("constraint_center_radius", 24.0)))
        self.aux = AuxClassificationLoss()

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], stage: str = "stage2") -> dict[str, torch.Tensor]:
        logs: dict[str, torch.Tensor] = {}
        total = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        w_search = build_sample_weight(batch.get("annotation_quality"), batch.get("closed_eye_flag"), None)
        w_mask = build_sample_weight(batch.get("annotation_quality"), batch.get("closed_eye_flag"), batch.get("mask_valid"))
        w_track = build_sample_weight(batch.get("annotation_quality"), batch.get("closed_eye_flag"), batch.get("valid_track"))
        if "eye_pred" in outputs and "eye_target" in batch:
            out = self.eye_loss(outputs["eye_pred"], batch["eye_target"], w_search)
            logs.update(out)
            total = total + self.weights.get("eye_bbox", 1.0) * out["eye_bbox"] + self.weights.get("eye_conf", 0.25) * out["eye_conf"]
        if "mask_logits" in outputs and "mask_target" in batch:
            logs["mask_bce"] = self.mask_bce(outputs["mask_logits"], batch["mask_target"], w_mask)
            logs["mask_dice"] = self.dice(outputs["mask_logits"], batch["mask_target"], w_mask)
            total = total + self.weights.get("mask_bce", 1.0) * logs["mask_bce"] + self.weights.get("mask_dice", 1.0) * logs["mask_dice"]
        search_state = None
        if "search_pred" in outputs and "pupil_search_target" in batch:
            out = self.search_losses(outputs["search_pred"], batch["pupil_search_target"], w_search)
            logs.update(out)
            for key, value in out.items():
                total = total + self.weights.get(key, 1.0) * value
            search_state = outputs["search_pred"][:, :6]
        track_state = None
        if stage == "stage2" and "track_pred" in outputs:
            out = self.track_losses(outputs["track_pred"], batch["pupil_track_target"], batch["prev_state"], batch["cur_state"], w_track)
            logs.update(out)
            for key, value in out.items():
                total = total + self.weights.get(key, 1.0) * value
            track_state = decode_track_state(batch["prev_state"], outputs["track_pred"])
        if stage == "stage2" and search_state is not None and track_state is not None:
            logs["consistency"] = self.consistency(search_state, track_state, w_track)
            total = total + self.weights.get("consistency", 0.25) * logs["consistency"]
        if "constraint_center" in batch:
            ref_state = track_state if track_state is not None else search_state
            ref_weight = w_track if track_state is not None else w_search
            if ref_state is not None:
                logs["constraint_center"] = self.constraint(ref_state, batch["constraint_center"], ref_weight)
                total = total + self.weights.get("constraint_center", 0.25) * logs["constraint_center"]
        if "aux_logits" in outputs and "aux_target" in batch:
            logs["aux"] = self.aux(outputs["aux_logits"], batch["aux_target"], w_search)
            total = total + self.weights.get("aux", 0.1) * logs["aux"]
        logs["total"] = total
        return logs


__all__ = [
    "AdaptiveWingLoss",
    "AnchorConsistencyLoss",
    "AuxClassificationLoss",
    "BCEDiceLoss",
    "BinaryFocalLoss",
    "CIoULoss",
    "CenterHeatmapLoss",
    "CenterNetFocalLoss",
    "CenterLoss",
    "CenterOffsetLoss",
    "ConfidenceAwareRelocalizationLoss",
    "ConfidenceWeightedMSELoss",
    "ConsistencyLoss",
    "ConstraintCenterLoss",
    "DiceLoss",
    "DisplacementLoss",
    "DistributionFocalLoss",
    "EllipseOverlapLoss",
    "EventDensityWeighting",
    "EventSearchBranchLosses",
    "EventToFrameContrastiveLoss",
    "EyeRegionLoss",
    "FeatureConsistencyLoss",
    "GIoULoss",
    "MGIoU2DLoss",
    "MPDIoULoss",
    "GazeAngularLoss",
    "GazeVelocityLoss",
    "HBTXRStageLoss",
    "MaskBCELoss",
    "NLLHeatmapLoss",
    "PredictionKDLoss",
    "PropagatedStateLoss",
    "RelationalKDLoss",
    "RotatedAngleLoss",
    "RotatedBBoxLoss",
    "SearchBranchLosses",
    "TemporalDisplacementLoss",
    "TrackBranchLosses",
    "TrigRotationLoss",
    "VarifocalLoss",
]
