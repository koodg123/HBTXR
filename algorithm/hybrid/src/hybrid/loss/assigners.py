from __future__ import annotations

from dataclasses import dataclass

import torch


def _flatten_dense_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 4:
        return logits[:, 0].flatten(1)
    if logits.ndim == 3:
        return logits.flatten(1)
    if logits.ndim == 2:
        return logits
    raise ValueError(f"Unsupported dense-logit shape: {tuple(logits.shape)}")


def _pairwise_iou_xyxy(pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if pred_boxes.ndim != 3 or pred_boxes.shape[-1] != 4:
        raise ValueError(f"pred_boxes must be [B, N, 4], got {tuple(pred_boxes.shape)}")
    if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
        raise ValueError(f"target_boxes must be [B, 4], got {tuple(target_boxes.shape)}")
    target = target_boxes[:, None, :]
    inter_x1 = torch.maximum(pred_boxes[..., 0], target[..., 0])
    inter_y1 = torch.maximum(pred_boxes[..., 1], target[..., 1])
    inter_x2 = torch.minimum(pred_boxes[..., 2], target[..., 2])
    inter_y2 = torch.minimum(pred_boxes[..., 3], target[..., 3])
    inter_w = (inter_x2 - inter_x1).clamp_min(0.0)
    inter_h = (inter_y2 - inter_y1).clamp_min(0.0)
    inter = inter_w * inter_h
    area_pred = (pred_boxes[..., 2] - pred_boxes[..., 0]).clamp_min(0.0) * (pred_boxes[..., 3] - pred_boxes[..., 1]).clamp_min(0.0)
    area_target = (target[..., 2] - target[..., 0]).clamp_min(0.0) * (target[..., 3] - target[..., 1]).clamp_min(0.0)
    union = area_pred + area_target - inter
    return inter / union.clamp_min(1.0e-6)


def _points_inside_xyxy(points: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
    if points.ndim == 2:
        points = points.unsqueeze(0).expand(target_boxes.shape[0], -1, -1)
    x = points[..., 0]
    y = points[..., 1]
    x1 = target_boxes[:, None, 0]
    y1 = target_boxes[:, None, 1]
    x2 = target_boxes[:, None, 2]
    y2 = target_boxes[:, None, 3]
    return (x >= x1) & (x <= x2) & (y >= y1) & (y <= y2)


@dataclass
class TaskAlignedAssignment:
    positive_mask: torch.Tensor
    target_scores: torch.Tensor
    target_boxes: torch.Tensor
    assigned_iou: torch.Tensor
    alignment_metric: torch.Tensor


class TaskAlignedAssigner:
    def __init__(
        self,
        *,
        topk: int = 10,
        alpha: float = 1.0,
        beta: float = 6.0,
        min_positive: int = 1,
        min_score: float = 0.0,
    ) -> None:
        self.topk = max(1, int(topk))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.min_positive = max(1, int(min_positive))
        self.min_score = float(min_score)

    def _alignment_metric(
        self,
        *,
        cls_logits: torch.Tensor,
        pred_boxes_xyxy: torch.Tensor,
        target_boxes_xyxy: torch.Tensor,
        inside_mask: torch.Tensor,
        quality_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_prob = torch.sigmoid(_flatten_dense_logits(cls_logits))
        iou = _pairwise_iou_xyxy(pred_boxes_xyxy, target_boxes_xyxy)
        metric = cls_prob.clamp_min(1.0e-6).pow(self.alpha) * iou.clamp_min(1.0e-6).pow(self.beta)
        if quality_logits is not None:
            quality_prob = torch.sigmoid(_flatten_dense_logits(quality_logits))
            metric = metric * quality_prob.clamp_min(1.0e-6)
        return metric, iou

    def _nearest_indices(self, points: torch.Tensor, target_box: torch.Tensor, *, topk: int) -> torch.Tensor:
        center = 0.5 * (target_box[:2] + target_box[2:4])
        dist = ((points - center.view(1, 2)) ** 2).sum(dim=-1)
        count = min(max(1, int(topk)), int(points.shape[0]))
        return dist.argsort(dim=0)[:count]

    def _augment_indices(self, *, points: torch.Tensor, target_box: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        del points, target_box
        return selected.unique(sorted=False)

    def assign(
        self,
        *,
        cls_logits: torch.Tensor,
        pred_boxes_xyxy: torch.Tensor,
        points: torch.Tensor,
        target_boxes_xyxy: torch.Tensor,
        quality_logits: torch.Tensor | None = None,
    ) -> TaskAlignedAssignment:
        if points.ndim == 2:
            points = points.unsqueeze(0).expand(target_boxes_xyxy.shape[0], -1, -1)
        inside_mask = _points_inside_xyxy(points, target_boxes_xyxy)
        metric, iou = self._alignment_metric(
            cls_logits=cls_logits,
            pred_boxes_xyxy=pred_boxes_xyxy,
            target_boxes_xyxy=target_boxes_xyxy,
            inside_mask=inside_mask,
            quality_logits=quality_logits,
        )

        batch, num_points = metric.shape
        positive_mask = torch.zeros((batch, num_points), dtype=torch.bool, device=metric.device)
        target_scores = torch.zeros((batch, num_points), dtype=metric.dtype, device=metric.device)
        for batch_index in range(batch):
            valid_inside = torch.nonzero(inside_mask[batch_index], as_tuple=False).flatten()
            k = min(self.topk, num_points)
            global_order = torch.argsort(metric[batch_index], descending=True)
            selected = global_order[:k]
            positive_floor = max(0.0, self.min_score)
            selected = selected[metric[batch_index, selected] > positive_floor]
            if selected.numel() < self.min_positive:
                anchor_candidates = valid_inside
                if anchor_candidates.numel() == 0:
                    anchor_candidates = self._nearest_indices(
                        points[batch_index],
                        target_boxes_xyxy[batch_index],
                        topk=self.min_positive,
                    )
                selected = torch.cat([selected, anchor_candidates[: self.min_positive]], dim=0)
            selected = self._augment_indices(points=points[batch_index], target_box=target_boxes_xyxy[batch_index], selected=selected)
            positive_mask[batch_index, selected] = True
            selected_scores = iou[batch_index, selected].clamp(0.0, 1.0)
            if selected_scores.numel() > 0:
                target_scores[batch_index, selected] = selected_scores.clamp_min(self.min_score)

        expanded_boxes = target_boxes_xyxy[:, None, :].expand(-1, num_points, -1)
        return TaskAlignedAssignment(
            positive_mask=positive_mask,
            target_scores=target_scores,
            target_boxes=expanded_boxes,
            assigned_iou=iou,
            alignment_metric=metric,
        )

    __call__ = assign


class STALTaskAlignedAssigner(TaskAlignedAssigner):
    def __init__(
        self,
        *,
        topk: int = 10,
        alpha: float = 1.0,
        beta: float = 6.0,
        min_positive: int = 1,
        min_score: float = 0.0,
        stal_enabled: bool = True,
        stal_small_box_px: float = 16.0,
        stal_min_points: int = 4,
    ) -> None:
        super().__init__(topk=topk, alpha=alpha, beta=beta, min_positive=min_positive, min_score=min_score)
        self.stal_enabled = bool(stal_enabled)
        self.stal_small_box_px = float(stal_small_box_px)
        self.stal_min_points = max(1, int(stal_min_points))

    def _augment_indices(self, *, points: torch.Tensor, target_box: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if not self.stal_enabled:
            return selected.unique(sorted=False)
        wh = (target_box[2:4] - target_box[:2]).clamp_min(1.0e-3)
        if float(torch.min(wh).item()) > self.stal_small_box_px:
            return selected.unique(sorted=False)
        nearest = self._nearest_indices(points, target_box, topk=self.stal_min_points)
        return torch.cat([selected, nearest], dim=0).unique(sorted=False)


__all__ = [
    "TaskAlignedAssignment",
    "TaskAlignedAssigner",
    "STALTaskAlignedAssigner",
]
