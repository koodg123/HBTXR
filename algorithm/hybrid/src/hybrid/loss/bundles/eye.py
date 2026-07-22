from __future__ import annotations

from typing import Dict

import torch
from torch.nn import functional as F

from hybrid.loss.assigners import STALTaskAlignedAssigner, TaskAlignedAssigner
from hybrid.loss.common import (
    box_cxcywh_to_xyxy,
    box_xyxy_to_cxcywh,
    mask_bce_dice_loss,
    sigmoid_bce_with_mask,
    smooth_l1_with_mask,
)
from hybrid.loss.primitives import (
    BinaryFocalLoss,
    CIoULoss,
    CenterNetFocalLoss,
    DistributionFocalLoss,
    GIoULoss,
    MGIoU2DLoss,
    MPDIoULoss,
    RotatedAngleLoss,
    RotatedBBoxLoss,
    VarifocalLoss,
)

_EYE_CIOU_LOSS = CIoULoss()
_GIOU_LOSS = GIoULoss()
_MPDIOU_LOSS = MPDIoULoss()
_YOLO_FOCAL_LOSS = BinaryFocalLoss()
_VARIFOCAL_LOSS = VarifocalLoss()
_DFL_LOSS = DistributionFocalLoss()
_CENTERNET_FOCAL_LOSS = CenterNetFocalLoss()
_ROTATED_BBOX_LOSS = RotatedBBoxLoss()
_ROTATED_ANGLE_LOSS = RotatedAngleLoss()
_MGIOU2D_LOSS = MGIoU2DLoss(representation="rect", fast_mode=False)
_MGIOU2D_FAST_LOSS = MGIoU2DLoss(representation="rect", fast_mode=True)


def _build_eye_assigner(loss_cfg: Dict) -> TaskAlignedAssigner:
    assigner_name = str(loss_cfg.get("eye_assigner", "stal_tal")).strip().lower()
    kwargs = {
        "topk": int(loss_cfg.get("eye_assigner_topk", 10)),
        "alpha": float(loss_cfg.get("eye_assigner_alpha", 1.0)),
        "beta": float(loss_cfg.get("eye_assigner_beta", 6.0)),
        "min_positive": int(loss_cfg.get("eye_assigner_min_positive", 1)),
        "min_score": float(loss_cfg.get("eye_assigner_min_score", 0.0)),
    }
    if assigner_name in {"stal", "stal_tal", "tal_stal"}:
        return STALTaskAlignedAssigner(
            **kwargs,
            stal_enabled=bool(loss_cfg.get("eye_stal_enabled", True)),
            stal_small_box_px=float(loss_cfg.get("eye_stal_small_box_px", 16.0)),
            stal_min_points=int(loss_cfg.get("eye_stal_min_points", 4)),
        )
    if assigner_name == "tal":
        return TaskAlignedAssigner(**kwargs)
    raise ValueError(f"Unsupported eye_assigner: {assigner_name!r}. Expected tal or stal_tal.")


def _grid_points(height: int, width: int, *, image_size: tuple[int, int], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    stride_y = float(image_size[0]) / max(1, height)
    stride_x = float(image_size[1]) / max(1, width)
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride_y
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    stride = torch.tensor([stride_x, stride_y], device=device, dtype=dtype)
    return grid, stride


def _gather_spatial(tensor: torch.Tensor, flat_index: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten(2).transpose(1, 2)
    gather_index = flat_index.view(-1, 1, 1).expand(-1, 1, flat.shape[-1])
    return flat.gather(1, gather_index).squeeze(1)


def _gaussian_heatmap(
    *,
    center_xy: torch.Tensor,
    wh: torch.Tensor,
    height: int,
    width: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    device = center_xy.device
    dtype = center_xy.dtype
    grid, stride = _grid_points(height, width, image_size=image_size, device=device, dtype=dtype)
    grid = grid.view(1, height, width, 2)
    center = center_xy.view(-1, 1, 1, 2)
    sigma = (wh / 6.0).clamp_min(stride * 0.5).view(-1, 1, 1, 2)
    norm = ((grid - center) / sigma).pow(2).sum(dim=-1)
    return torch.exp(-0.5 * norm)


def _corner_heatmap(
    *,
    xyxy: torch.Tensor,
    height: int,
    width: int,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    wh = (xyxy[:, 2:4] - xyxy[:, :2]).clamp_min(1.0e-3)
    tl = _gaussian_heatmap(center_xy=xyxy[:, :2], wh=wh, height=height, width=width, image_size=image_size)
    br = _gaussian_heatmap(center_xy=xyxy[:, 2:4], wh=wh, height=height, width=width, image_size=image_size)
    return tl, br


def _decode_point_distances(reg_logits: torch.Tensor, *, reg_max: int, stride_xy: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = reg_logits.shape
    if reg_max > 1:
        raw = reg_logits.view(batch, 4, reg_max, height, width)
        prob = raw.softmax(dim=2)
        bins = torch.arange(reg_max, device=reg_logits.device, dtype=reg_logits.dtype).view(1, 1, reg_max, 1, 1)
        dist = (prob * bins).sum(dim=2)
    else:
        dist = F.softplus(reg_logits)
    scale = torch.tensor([stride_xy[0], stride_xy[1], stride_xy[0], stride_xy[1]], device=reg_logits.device, dtype=reg_logits.dtype).view(1, 4, 1, 1)
    return dist * scale


def _point_classification_loss(logits: torch.Tensor, target: torch.Tensor, *, mode: str) -> torch.Tensor:
    if mode == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        prob = torch.sigmoid(logits)
        pt = prob * target + (1.0 - prob) * (1.0 - target)
        alpha = 0.25 * target + 0.75 * (1.0 - target)
        return alpha * (1.0 - pt).pow(2.0) * bce
    if mode == "varifocal":
        pred_prob = torch.sigmoid(logits)
        focal_weight = target + 0.75 * (pred_prob - target).abs().pow(2.0) * (1.0 - target)
        return F.binary_cross_entropy_with_logits(logits, target, reduction="none") * focal_weight
    return F.binary_cross_entropy_with_logits(logits, target)


def axis_aligned_bbox_loss(pred_box: torch.Tensor, target_box: torch.Tensor, weights: torch.Tensor, *, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode in {"mpdiou", "yolo26_mpdiou"}:
        return _MPDIOU_LOSS(pred_box, target_box, xyxy=False, weight=weights)
    return _EYE_CIOU_LOSS(pred_box, target_box, xyxy=False, weight=weights)


def rotated_overlap_loss(pred_box: torch.Tensor, target_box: torch.Tensor, weights: torch.Tensor, *, mode: str, fast_mode: bool) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "mgiou2d":
        loss_fn = _MGIOU2D_FAST_LOSS if fast_mode else _MGIOU2D_LOSS
        return loss_fn(pred_box[:, :5], target_box, weights)
    return _ROTATED_BBOX_LOSS(pred_box[:, :5], target_box, weights)


def rotated_angle_loss(pred_angle: torch.Tensor, target_angle: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return _ROTATED_ANGLE_LOSS(pred_angle, target_angle, weights)


def _dense_eye_detector_loss(
    *,
    cls_logits: torch.Tensor,
    reg_logits: torch.Tensor,
    target_box: torch.Tensor,
    weights: torch.Tensor,
    loss_cfg: Dict,
    image_size: tuple[int, int],
    quality_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, _, height, width = cls_logits.shape
    grid, stride_xy = _grid_points(height, width, image_size=image_size, device=cls_logits.device, dtype=cls_logits.dtype)
    points = grid.reshape(-1, 2)
    target_xyxy = box_cxcywh_to_xyxy(target_box)
    inferred_reg_max = max(1, int(reg_logits.shape[1] // 4))
    reg_max = max(1, int(loss_cfg.get("eye_reg_max", inferred_reg_max)))
    pred_dist_px = _decode_point_distances(reg_logits, reg_max=reg_max, stride_xy=stride_xy)
    pred_dist_flat = pred_dist_px.flatten(2).transpose(1, 2)
    point_grid = points.unsqueeze(0).expand(batch, -1, -1)
    pred_xyxy = torch.stack(
        [
            point_grid[..., 0] - pred_dist_flat[..., 0],
            point_grid[..., 1] - pred_dist_flat[..., 1],
            point_grid[..., 0] + pred_dist_flat[..., 2],
            point_grid[..., 1] + pred_dist_flat[..., 3],
        ],
        dim=-1,
    )
    assigner = _build_eye_assigner(loss_cfg)
    assignment = assigner(
        cls_logits=cls_logits,
        pred_boxes_xyxy=pred_xyxy,
        points=point_grid,
        target_boxes_xyxy=target_xyxy,
        quality_logits=quality_logits,
    )
    positive_mask = assignment.positive_mask
    if quality_logits is None:
        cls_target = assignment.target_scores.view(batch, height, width).unsqueeze(1)
    else:
        cls_target = positive_mask.to(dtype=cls_logits.dtype).view(batch, height, width).unsqueeze(1)
    cls_loss_per_sample = _point_classification_loss(
        cls_logits,
        cls_target,
        mode=str(loss_cfg.get("eye_point_cls_loss", "bce")).strip().lower(),
    ).flatten(1).mean(dim=-1)
    cls_loss = (cls_loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
    pos = positive_mask > 0.0
    quality_loss = torch.zeros((), device=cls_logits.device)
    if quality_logits is not None:
        quality_target = assignment.target_scores.view(batch, height, width).unsqueeze(1)
        quality_loss = F.binary_cross_entropy_with_logits(quality_logits, quality_target, reduction="none").flatten(1).mean(dim=-1)
        quality_loss = (quality_loss * weights).sum() / weights.sum().clamp_min(1e-6)
    if not pos.any():
        zero = torch.zeros((), device=cls_logits.device)
        return cls_loss, zero, zero, quality_loss

    pred_dist_pos = pred_dist_flat[pos]
    point_pos = points.unsqueeze(0).expand(batch, -1, -1)[pos]
    target_xyxy_pos = assignment.target_boxes[pos]
    xyxy = torch.stack(
        [
            point_pos[:, 0] - pred_dist_pos[:, 0],
            point_pos[:, 1] - pred_dist_pos[:, 1],
            point_pos[:, 0] + pred_dist_pos[:, 2],
            point_pos[:, 1] + pred_dist_pos[:, 3],
        ],
        dim=-1,
    )
    pos_scores = assignment.target_scores[pos].clamp_min(1.0e-6)
    pos_weights = weights[:, None].expand(-1, points.shape[0])[pos] * pos_scores
    box_loss = _EYE_CIOU_LOSS(xyxy, target_xyxy_pos, xyxy=True, weight=pos_weights)

    dfl_loss = torch.zeros((), device=cls_logits.device)
    if reg_max > 1:
        raw = reg_logits.view(batch, 4, reg_max, height, width).permute(0, 3, 4, 1, 2).reshape(batch, -1, 4, reg_max)
        raw_pos = raw[pos]
        stride = stride_xy.view(1, 2)
        target_dist = torch.stack(
            [
                (point_pos[:, 0] - target_xyxy_pos[:, 0]) / stride[:, 0],
                (point_pos[:, 1] - target_xyxy_pos[:, 1]) / stride[:, 1],
                (target_xyxy_pos[:, 2] - point_pos[:, 0]) / stride[:, 0],
                (target_xyxy_pos[:, 3] - point_pos[:, 1]) / stride[:, 1],
            ],
            dim=-1,
        ).clamp_min(0.0)
        dfl_loss = _DFL_LOSS(raw_pos, target_dist, pos_weights)
    return cls_loss, box_loss, dfl_loss, quality_loss


def _dense_eye_center_loss(
    *,
    pred_box: torch.Tensor,
    center_logits: torch.Tensor,
    size_map: torch.Tensor,
    offset_map: torch.Tensor,
    target_box: torch.Tensor,
    weights: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = center_logits.shape
    target_heatmap = _gaussian_heatmap(
        center_xy=target_box[:, :2],
        wh=target_box[:, 2:4],
        height=height,
        width=width,
        image_size=image_size,
    ).unsqueeze(1)
    focal_loss = _CENTERNET_FOCAL_LOSS(center_logits, target_heatmap, weights)

    grid, stride_xy = _grid_points(height, width, image_size=image_size, device=pred_box.device, dtype=pred_box.dtype)
    stride = stride_xy.view(1, 2)
    center_index_x = (target_box[:, 0] / stride[:, 0]).floor().clamp(0, width - 1).long()
    center_index_y = (target_box[:, 1] / stride[:, 1]).floor().clamp(0, height - 1).long()
    flat_index = center_index_y * width + center_index_x
    gathered_size = _gather_spatial(size_map, flat_index)
    gathered_offset = _gather_spatial(offset_map, flat_index)
    point_center = grid.reshape(-1, 2)[flat_index]
    target_offset = (target_box[:, :2] - point_center) / stride
    target_size = target_box[:, 2:4] / stride
    reg_loss = smooth_l1_with_mask(gathered_size, target_size, weights) + smooth_l1_with_mask(gathered_offset, target_offset, weights)
    box_loss = _GIOU_LOSS(pred_box[:, :4], target_box[:, :4], xyxy=False, weight=weights) + smooth_l1_with_mask(pred_box[:, :4], target_box[:, :4], weights)
    return focal_loss, reg_loss + box_loss


def _dense_eye_corner_loss(
    *,
    pred_box: torch.Tensor,
    tl_logits: torch.Tensor,
    br_logits: torch.Tensor,
    target_box: torch.Tensor,
    weights: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    _, _, height, width = tl_logits.shape
    target_xyxy = box_cxcywh_to_xyxy(target_box)
    tl_target, br_target = _corner_heatmap(xyxy=target_xyxy, height=height, width=width, image_size=image_size)
    focal_loss = _CENTERNET_FOCAL_LOSS(tl_logits, tl_target.unsqueeze(1), weights) + _CENTERNET_FOCAL_LOSS(br_logits, br_target.unsqueeze(1), weights)
    box_loss = _GIOU_LOSS(pred_box[:, :4], target_box[:, :4], xyxy=False, weight=weights) + smooth_l1_with_mask(pred_box[:, :4], target_box[:, :4], weights)
    return focal_loss, box_loss


def eye_mask_loss(mask_logits: torch.Tensor, mask_target: torch.Tensor, weights: torch.Tensor, *, mode: str) -> torch.Tensor:
    mode = str(mode).strip().lower()
    if mode == "bce":
        loss = F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none").mean(dim=(1, 2, 3))
        return (loss * weights).sum() / weights.sum().clamp_min(1e-6)
    if mode == "dice":
        prob = torch.sigmoid(mask_logits)
        inter = (prob * mask_target).flatten(1).sum(dim=-1)
        denom = prob.flatten(1).sum(dim=-1) + mask_target.flatten(1).sum(dim=-1)
        dice = 1.0 - (2.0 * inter + 1.0) / (denom + 1.0)
        return (dice * weights).sum() / weights.sum().clamp_min(1e-6)
    return mask_bce_dice_loss(mask_logits, mask_target, weights)


def eye_region_loss(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    loss_cfg: Dict,
    outputs: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, torch.Tensor]:
    eye_weight = float(loss_cfg.get("eye_weight", 1.0))
    eye_conf_weight = float(loss_cfg.get("eye_conf_weight", 0.1))
    if eye_weight <= 0.0:
        return {"loss_eye": pred.new_zeros(())}
    eye_box_mode = str(loss_cfg.get("eye_box_mode", "legacy_l1")).strip().lower()
    logs: Dict[str, torch.Tensor] = {}
    if outputs is not None and {"search/eye_boxes", "search/eye_obj_logits", "search/eye_anchor_points"} <= outputs.keys():
        assignment = _build_eye_assigner(loss_cfg)(
            cls_logits=outputs["search/eye_obj_logits"],
            pred_boxes_xyxy=outputs["search/eye_boxes"],
            points=outputs["search/eye_anchor_points"],
            target_boxes_xyxy=box_cxcywh_to_xyxy(target[:, :4]),
        )
        obj_loss_per_sample = _point_classification_loss(
            outputs["search/eye_obj_logits"],
            assignment.target_scores,
            mode=str(loss_cfg.get("eye_point_cls_loss", "bce")).strip().lower(),
        )
        if obj_loss_per_sample.ndim > 1:
            obj_loss_per_sample = obj_loss_per_sample.mean(dim=-1)
        obj_loss = (obj_loss_per_sample * weights).sum() / weights.sum().clamp_min(1e-6)
        pos = assignment.positive_mask
        box_loss = pred.new_zeros(())
        if pos.any():
            pos_weights = weights[:, None].expand_as(assignment.target_scores)[pos] * assignment.target_scores[pos].clamp_min(1.0e-6)
            box_loss = _EYE_CIOU_LOSS(
                outputs["search/eye_boxes"][pos],
                assignment.target_boxes[pos],
                xyxy=True,
                weight=pos_weights,
            )
        logs["loss_eye_obj"] = obj_loss * float(loss_cfg.get("eye_obj_weight", eye_conf_weight)) * eye_weight
        logs["loss_eye_box"] = box_loss * float(loss_cfg.get("eye_box_weight", 1.0)) * eye_weight
        logs["loss_eye"] = torch.stack(tuple(logs.values())).sum()
        return logs
    if eye_box_mode in {"yolo26_ciou", "yolo26_mpdiou"}:
        overlap_term = axis_aligned_bbox_loss(pred[:, 0:4], target[:, 0:4], weights, mode=eye_box_mode)
        l1_term = smooth_l1_with_mask(box_cxcywh_to_xyxy(pred[:, 0:4]), box_cxcywh_to_xyxy(target[:, 0:4]), weights)
        box_loss = overlap_term * float(loss_cfg.get("eye_box_iou_weight", 1.0)) + l1_term * float(loss_cfg.get("eye_box_l1_weight", 0.25))
        conf_mode = str(loss_cfg.get("eye_conf_mode", "bce")).strip().lower()
        if conf_mode == "focal":
            conf_loss = _YOLO_FOCAL_LOSS(pred[:, 4], target[:, 4], weights)
        elif conf_mode == "varifocal":
            conf_loss = _VARIFOCAL_LOSS(pred[:, 4], target[:, 4], target[:, 4], weights)
        else:
            conf_loss = sigmoid_bce_with_mask(pred[:, 4], target[:, 4], weights)
        logs["loss_eye_conf"] = conf_loss * eye_conf_weight
        logs["loss_eye"] = (box_loss + conf_loss * eye_conf_weight) * eye_weight
        return logs
    if eye_box_mode in {"yolo26_point", "yolo_detect"}:
        if outputs is None or "search/eye_det_cls_logits" not in outputs:
            raise KeyError(
                "eye_box_mode in {'yolo26_point','yolo_detect'} requires "
                "search/eye_det_cls_logits/search/eye_det_reg_logits"
            )
        cls_loss, box_loss, dfl_loss, quality_loss = _dense_eye_detector_loss(
            cls_logits=outputs["search/eye_det_cls_logits"],
            reg_logits=outputs["search/eye_det_reg_logits"],
            target_box=target[:, :4],
            weights=weights,
            loss_cfg=loss_cfg,
            image_size=tuple(int(v) for v in loss_cfg.get("input_size", [256, 256])),
            quality_logits=outputs.get("search/eye_det_quality_logits"),
        )
        prefix = "eye_point" if eye_box_mode == "yolo26_point" else "eye_det"
        logs[f"loss_{prefix}_cls"] = cls_loss * float(loss_cfg.get("eye_point_cls_weight", 1.0)) * eye_weight
        logs[f"loss_{prefix}_box"] = box_loss * float(loss_cfg.get("eye_point_box_weight", 1.0)) * eye_weight
        logs[f"loss_{prefix}_dfl"] = dfl_loss * float(loss_cfg.get("eye_point_dfl_weight", 0.25)) * eye_weight
        if "search/eye_det_quality_logits" in (outputs or {}):
            logs[f"loss_{prefix}_quality"] = quality_loss * float(loss_cfg.get("eye_quality_weight", 0.25)) * eye_weight
        logs["loss_eye"] = torch.stack(tuple(logs.values())).sum()
        return logs
    if eye_box_mode == "sot_center":
        if outputs is None or "search/eye_center_logits" not in outputs:
            raise KeyError("eye_box_mode='sot_center' requires search/eye_center_logits/search/eye_size_map/search/eye_offset_map")
        focal_loss, box_loss = _dense_eye_center_loss(
            pred_box=pred,
            center_logits=outputs["search/eye_center_logits"],
            size_map=outputs["search/eye_size_map"],
            offset_map=outputs["search/eye_offset_map"],
            target_box=target[:, :4],
            weights=weights,
            image_size=tuple(int(v) for v in loss_cfg.get("input_size", [256, 256])),
        )
        logs["loss_eye_center_focal"] = focal_loss * float(loss_cfg.get("eye_center_focal_weight", 1.0)) * eye_weight
        logs["loss_eye_center_box"] = box_loss * float(loss_cfg.get("eye_center_box_weight", 1.0)) * eye_weight
        logs["loss_eye"] = torch.stack(tuple(logs.values())).sum()
        return logs
    if eye_box_mode == "sot_corner":
        if outputs is None or "search/eye_tl_logits" not in outputs:
            raise KeyError("eye_box_mode='sot_corner' requires search/eye_tl_logits/search/eye_br_logits")
        focal_loss, box_loss = _dense_eye_corner_loss(
            pred_box=pred,
            tl_logits=outputs["search/eye_tl_logits"],
            br_logits=outputs["search/eye_br_logits"],
            target_box=target[:, :4],
            weights=weights,
            image_size=tuple(int(v) for v in loss_cfg.get("input_size", [256, 256])),
        )
        logs["loss_eye_corner_focal"] = focal_loss * float(loss_cfg.get("eye_corner_focal_weight", 1.0)) * eye_weight
        logs["loss_eye_corner_box"] = box_loss * float(loss_cfg.get("eye_corner_box_weight", 1.0)) * eye_weight
        logs["loss_eye"] = torch.stack(tuple(logs.values())).sum()
        return logs
    box_loss = smooth_l1_with_mask(pred[:, 0:4], target[:, 0:4], weights)
    conf_loss = sigmoid_bce_with_mask(pred[:, 4], target[:, 4], weights) * eye_conf_weight
    logs["loss_eye"] = (box_loss + conf_loss) * eye_weight
    return logs


__all__ = [
    "axis_aligned_bbox_loss",
    "eye_mask_loss",
    "eye_region_loss",
    "rotated_angle_loss",
    "rotated_overlap_loss",
]
