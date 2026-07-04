from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .patch_embeddings import tokens_to_grid


def _mlp_head(in_dim: int, out_dim: int, *, hidden_dim: int | None = None) -> nn.Sequential:
    hidden = int(hidden_dim) if hidden_dim is not None else max(in_dim, out_dim)
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


def _dense_stem(in_dim: int, hidden_dim: int | None = None) -> nn.Sequential:
    hidden = int(hidden_dim) if hidden_dim is not None else max(in_dim, 64)
    return nn.Sequential(
        nn.Conv2d(in_dim, hidden, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
        nn.GELU(),
    )


def _grid_points(
    *,
    height: int,
    width: int,
    image_size: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    stride_y = float(image_size[0]) / max(1, height)
    stride_x = float(image_size[1]) / max(1, width)
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride_y
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride_x
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)
    stride = torch.tensor([stride_x, stride_y], device=device, dtype=dtype)
    return grid, stride


def _gather_spatial(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = tensor.flatten(2).transpose(1, 2)
    gather_index = indices.view(-1, 1, 1).expand(-1, 1, flat.shape[-1])
    return flat.gather(1, gather_index).squeeze(1)


def _decode_dense_bbox(
    *,
    reg_logits: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    reg_max: int,
    score_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid, stride_xy = _grid_points(
        height=grid_size[0],
        width=grid_size[1],
        image_size=image_size,
        device=reg_logits.device,
        dtype=reg_logits.dtype,
    )
    batch, _, _, _ = reg_logits.shape
    if reg_max > 1:
        raw_reg = reg_logits.view(batch, 4, reg_max, grid_size[0], grid_size[1])
        prob = raw_reg.softmax(dim=2)
        bins = torch.arange(reg_max, device=reg_logits.device, dtype=reg_logits.dtype).view(1, 1, reg_max, 1, 1)
        dist = (prob * bins).sum(dim=2)
    else:
        dist = F.softplus(reg_logits)
    scale = torch.tensor(
        [stride_xy[0], stride_xy[1], stride_xy[0], stride_xy[1]],
        device=reg_logits.device,
        dtype=reg_logits.dtype,
    ).view(1, 4, 1, 1)
    dist_px = dist * scale
    scores = score_map.flatten(1)
    top_idx = scores.argmax(dim=-1)
    points = grid.reshape(-1, 2)[top_idx]
    point_dist = _gather_spatial(dist_px, top_idx)
    xyxy = torch.stack(
        [
            points[:, 0] - point_dist[:, 0],
            points[:, 1] - point_dist[:, 1],
            points[:, 0] + point_dist[:, 2],
            points[:, 1] + point_dist[:, 3],
        ],
        dim=-1,
    )
    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    w = (xyxy[:, 2] - xyxy[:, 0]).clamp_min(1.0e-3)
    h = (xyxy[:, 3] - xyxy[:, 1]).clamp_min(1.0e-3)
    conf = scores.gather(1, top_idx.view(-1, 1))
    bbox = torch.cat([torch.stack([cx, cy, w, h], dim=-1), conf], dim=-1)
    return bbox, dist_px


def _soft_argmax_2d(logits: torch.Tensor, *, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = logits.shape
    grid, _ = _grid_points(
        height=height,
        width=width,
        image_size=image_size,
        device=logits.device,
        dtype=logits.dtype,
    )
    flat_logits = logits[:, 0].flatten(1)
    prob = flat_logits.softmax(dim=-1)
    coords = grid.reshape(-1, 2)
    xy = prob @ coords
    peak = flat_logits.max(dim=-1).values.unsqueeze(-1)
    return xy, peak


def _box_xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, (x2 - x1).clamp_min(0.0), (y2 - y1).clamp_min(0.0)], dim=-1)


def _box_cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    half_w = w.clamp_min(1.0) * 0.5
    half_h = h.clamp_min(1.0) * 0.5
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def _clamp_xyxy(box: torch.Tensor, *, width: float, height: float, min_size: float = 4.0) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    x1 = x1.clamp(0.0, max(0.0, width - min_size))
    y1 = y1.clamp(0.0, max(0.0, height - min_size))
    x2 = x2.clamp(min_size, width)
    y2 = y2.clamp(min_size, height)
    x2 = torch.maximum(x2, x1 + min_size)
    y2 = torch.maximum(y2, y1 + min_size)
    x2 = x2.clamp(max=width)
    y2 = y2.clamp(max=height)
    x1 = torch.minimum(x1, x2 - min_size)
    y1 = torch.minimum(y1, y2 - min_size)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _roi_align_feature_map(feature: torch.Tensor, rois: torch.Tensor, *, output_size: tuple[int, int]) -> torch.Tensor:
    try:
        from torchvision.ops import roi_align
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("search.roi_bbox_head.use_roi_align=true requires torchvision.ops.roi_align") from exc
    return roi_align(feature, rois, output_size=output_size, spatial_scale=1.0, aligned=True)


class EyeRegionHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, 5, hidden_dim=hidden_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class _SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 2.0) -> None:
        super().__init__()
        hidden = max(embed_dim, int(round(embed_dim * float(mlp_ratio))))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_in = self.norm1(x)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class DenseEyeRegionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        output_size: tuple[int, int] = (256, 256),
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim // 2, 32)
        self.output_size = tuple(int(v) for v in output_size)
        self.trunk = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.obj_proj = nn.Conv2d(hidden, 1, kernel_size=1)
        self.box_proj = nn.Conv2d(hidden, 4, kernel_size=1)
        nn.init.constant_(self.obj_proj.bias, -4.0)
        nn.init.constant_(self.box_proj.bias, 1.0)

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        feature = self.trunk(feature)
        box_scale = torch.tensor(
            [
                self.output_size[1] / max(1, grid_size[1]),
                self.output_size[0] / max(1, grid_size[0]),
                self.output_size[1] / max(1, grid_size[1]),
                self.output_size[0] / max(1, grid_size[0]),
            ],
            device=feature.device,
            dtype=feature.dtype,
        ).view(1, 4, 1, 1)
        obj_logits = self.obj_proj(feature)
        ltrb = F.softplus(self.box_proj(feature)) * box_scale

        height, width = grid_size
        xs = (torch.arange(width, device=feature.device, dtype=feature.dtype) + 0.5) * (self.output_size[1] / max(1, width))
        ys = (torch.arange(height, device=feature.device, dtype=feature.dtype) + 0.5) * (self.output_size[0] / max(1, height))
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        anchor_points = torch.stack([xx, yy], dim=-1).view(1, height * width, 2).expand(feature.shape[0], -1, -1)

        obj_logits_flat = obj_logits.flatten(2).transpose(1, 2).squeeze(-1)
        ltrb_flat = ltrb.flatten(2).transpose(1, 2)
        left, top, right, bottom = ltrb_flat.unbind(-1)
        ax, ay = anchor_points.unbind(-1)
        boxes_xyxy = torch.stack([ax - left, ay - top, ax + right, ay + bottom], dim=-1)

        scores = obj_logits_flat.sigmoid()
        best_idx = scores.argmax(dim=1, keepdim=True)
        best_boxes = boxes_xyxy.gather(1, best_idx.unsqueeze(-1).expand(-1, -1, 4)).squeeze(1)
        best_logits = obj_logits_flat.gather(1, best_idx).squeeze(1)
        top1_eye = torch.cat([_box_xyxy_to_cxcywh(best_boxes), best_logits.unsqueeze(-1)], dim=-1)
        return {
            "eye": top1_eye,
            "eye_boxes": boxes_xyxy,
            "eye_obj_logits": obj_logits_flat,
            "eye_anchor_points": anchor_points,
        }


class RoiCascadeBBoxHead(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        output_size: tuple[int, int] = (256, 256),
        mlp0_hidden_dim: int | None = None,
        mlp1_hidden_dim: int | None = None,
        roi_size: tuple[int, int] = (4, 4),
        attn_heads: int = 4,
        attn_layers: int = 1,
        attn_mlp_ratio: float = 2.0,
        use_roi_align: bool = True,
        detach_eye_boxes: bool = True,
        min_box_size: float = 4.0,
    ) -> None:
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        self.roi_size = tuple(int(v) for v in roi_size)
        self.use_roi_align = bool(use_roi_align)
        self.detach_eye_boxes = bool(detach_eye_boxes)
        self.min_box_size = float(min_box_size)
        self.mlp0 = _mlp_head(embed_dim, 5, hidden_dim=mlp0_hidden_dim)
        self.mlp1 = _mlp_head(embed_dim, 5, hidden_dim=mlp1_hidden_dim)
        self.eye_token_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn_blocks = nn.ModuleList(
            [_SelfAttentionBlock(embed_dim=embed_dim, num_heads=int(attn_heads), mlp_ratio=float(attn_mlp_ratio)) for _ in range(max(1, int(attn_layers)))]
        )

    def forward(self, tokens: torch.Tensor, *, pooled: torch.Tensor, grid_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        eye_logits = self.mlp0(pooled)
        eye_boxes_xyxy = _clamp_xyxy(
            _box_cxcywh_to_xyxy(eye_logits[:, :4]),
            width=float(self.output_size[1]),
            height=float(self.output_size[0]),
            min_size=self.min_box_size,
        )
        if self.use_roi_align:
            roi_boxes_xyxy = eye_boxes_xyxy.detach() if self.detach_eye_boxes else eye_boxes_xyxy
            scale_x = float(grid_size[1]) / float(max(1, self.output_size[1]))
            scale_y = float(grid_size[0]) / float(max(1, self.output_size[0]))
            roi_boxes_feature = roi_boxes_xyxy.clone()
            roi_boxes_feature[:, [0, 2]] *= scale_x
            roi_boxes_feature[:, [1, 3]] *= scale_y
            batch_indices = torch.arange(tokens.shape[0], device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
            rois = torch.cat([batch_indices, roi_boxes_feature], dim=-1)
            roi_feature = _roi_align_feature_map(feature, rois, output_size=self.roi_size)
            roi_tokens = roi_feature.flatten(2).transpose(1, 2)
            pupil_context_idx = None
        else:
            eye_token = self.eye_token_proj(torch.cat([_box_xyxy_to_cxcywh(eye_boxes_xyxy), eye_logits[:, 4:5]], dim=-1)).unsqueeze(1)
            roi_tokens = torch.cat([eye_token, tokens], dim=1)
            pupil_context_idx = 0
        for block in self.attn_blocks:
            roi_tokens = block(roi_tokens)
        roi_pooled = roi_tokens[:, pupil_context_idx] if pupil_context_idx is not None else roi_tokens.mean(dim=1)
        pupil_logits = self.mlp1(roi_pooled)
        return {
            "eye": torch.cat([_box_xyxy_to_cxcywh(eye_boxes_xyxy), eye_logits[:, 4:5]], dim=-1),
            "pupil_bbox": pupil_logits,
            "roi_boxes_xyxy": eye_boxes_xyxy,
            "roi_tokens": roi_tokens,
        }


class EyeGuidedBBoxHead(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        output_size: tuple[int, int] = (256, 256),
        mlp1_hidden_dim: int | None = None,
        attn_heads: int = 4,
        attn_layers: int = 1,
        attn_mlp_ratio: float = 2.0,
        detach_eye_boxes: bool = True,
        min_box_size: float = 4.0,
    ) -> None:
        super().__init__()
        self.output_size = tuple(int(v) for v in output_size)
        self.detach_eye_boxes = bool(detach_eye_boxes)
        self.min_box_size = float(min_box_size)
        self.eye_token_proj = nn.Sequential(
            nn.LayerNorm(5),
            nn.Linear(5, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.mlp1 = _mlp_head(embed_dim, 5, hidden_dim=mlp1_hidden_dim)
        self.attn_blocks = nn.ModuleList(
            [_SelfAttentionBlock(embed_dim=embed_dim, num_heads=int(attn_heads), mlp_ratio=float(attn_mlp_ratio)) for _ in range(max(1, int(attn_layers)))]
        )

    def forward(self, tokens: torch.Tensor, *, eye_seed: torch.Tensor) -> dict[str, torch.Tensor]:
        seed = eye_seed.detach() if self.detach_eye_boxes else eye_seed
        eye_boxes_xyxy = _clamp_xyxy(
            _box_cxcywh_to_xyxy(seed[:, :4]),
            width=float(self.output_size[1]),
            height=float(self.output_size[0]),
            min_size=self.min_box_size,
        )
        eye_token = self.eye_token_proj(torch.cat([_box_xyxy_to_cxcywh(eye_boxes_xyxy), seed[:, 4:5]], dim=-1)).unsqueeze(1)
        roi_tokens = torch.cat([eye_token, tokens], dim=1)
        for block in self.attn_blocks:
            roi_tokens = block(roi_tokens)
        pupil_logits = self.mlp1(roi_tokens[:, 0])
        return {
            "eye": torch.cat([_box_xyxy_to_cxcywh(eye_boxes_xyxy), seed[:, 4:5]], dim=-1),
            "pupil_bbox": pupil_logits,
            "roi_boxes_xyxy": eye_boxes_xyxy,
            "roi_tokens": roi_tokens,
        }


class YOLO26EyeRegionHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 128)
        self.stem = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
        )
        self.box_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.conf_branch = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        feat = self.stem(pooled)
        raw_box = self.box_branch(feat)
        xy = raw_box[:, :2]
        # Keep width/height positive for CIoU-based eye-box regression.
        wh = F.softplus(raw_box[:, 2:4]) + 1.0e-3
        conf = self.conf_branch(feat)
        return torch.cat([xy, wh, conf], dim=-1)


class YOLOPointEyeRegionHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None, *, reg_max: int = 1) -> None:
        super().__init__()
        self.reg_max = max(1, int(reg_max))
        self.stem = _dense_stem(embed_dim, hidden_dim)
        reg_channels = self.reg_max if self.reg_max > 1 else 1
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 64)
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 4 * reg_channels, kernel_size=1),
        )

    def _decode_distances(self, raw_reg: torch.Tensor, *, stride_xy: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = raw_reg.shape
        if self.reg_max > 1:
            raw_reg = raw_reg.view(batch, 4, self.reg_max, height, width)
            prob = raw_reg.softmax(dim=2)
            bins = torch.arange(self.reg_max, device=raw_reg.device, dtype=raw_reg.dtype).view(1, 1, self.reg_max, 1, 1)
            dist = (prob * bins).sum(dim=2)
        else:
            dist = F.softplus(raw_reg)
        scale = torch.tensor(
            [stride_xy[0], stride_xy[1], stride_xy[0], stride_xy[1]],
            device=raw_reg.device,
            dtype=raw_reg.dtype,
        ).view(1, 4, 1, 1)
        return dist * scale

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int], image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        feat = self.stem(feature)
        cls_logits = self.cls_head(feat)
        reg_logits = self.reg_head(feat)
        score_map = torch.sigmoid(cls_logits[:, 0])
        bbox, _ = _decode_dense_bbox(
            reg_logits=reg_logits,
            grid_size=grid_size,
            image_size=image_size,
            reg_max=self.reg_max,
            score_map=score_map,
        )
        return {
            "bbox": bbox,
            "det_cls_logits": cls_logits,
            "det_reg_logits": reg_logits,
            "point_cls_logits": cls_logits,
            "point_reg_logits": reg_logits,
        }


class YOLODetectEyeRegionHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None, *, reg_max: int = 4) -> None:
        super().__init__()
        self.reg_max = max(1, int(reg_max))
        self.stem = _dense_stem(embed_dim, hidden_dim)
        reg_channels = self.reg_max if self.reg_max > 1 else 1
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 64)
        self.cls_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.reg_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 4 * reg_channels, kernel_size=1),
        )
        self.quality_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int], image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        feat = self.stem(feature)
        cls_logits = self.cls_head(feat)
        reg_logits = self.reg_head(feat)
        quality_logits = self.quality_head(feat)
        score_map = torch.sigmoid(cls_logits[:, 0]) * torch.sigmoid(quality_logits[:, 0])
        bbox, _ = _decode_dense_bbox(
            reg_logits=reg_logits,
            grid_size=grid_size,
            image_size=image_size,
            reg_max=self.reg_max,
            score_map=score_map,
        )
        return {
            "bbox": bbox,
            "det_cls_logits": cls_logits,
            "det_reg_logits": reg_logits,
            "det_quality_logits": quality_logits,
        }


class SOTCenterPredictor(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 64)
        self.stem = _dense_stem(embed_dim, hidden_dim)
        self.center_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int], image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        feat = self.stem(feature)
        center_logits = self.center_head(feat)
        size_map = F.softplus(self.size_head(feat)) + 1.0e-3
        offset_map = torch.tanh(self.offset_head(feat)) * 0.5
        grid, stride_xy = _grid_points(
            height=grid_size[0],
            width=grid_size[1],
            image_size=image_size,
            device=feat.device,
            dtype=feat.dtype,
        )
        scores = center_logits[:, 0].flatten(1)
        top_idx = scores.argmax(dim=-1)
        points = grid.reshape(-1, 2)[top_idx]
        offset = _gather_spatial(offset_map, top_idx) * stride_xy.view(1, 2)
        size = _gather_spatial(size_map, top_idx) * stride_xy.view(1, 2)
        center = points + offset
        conf = scores.gather(1, top_idx.view(-1, 1))
        bbox = torch.cat([center, size.clamp_min(1.0e-3), conf], dim=-1)
        return {
            "bbox": bbox,
            "center_logits": center_logits,
            "size_map": size_map,
            "offset_map": offset_map,
        }


class SOTCornerPredictor(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 64)
        self.stem = _dense_stem(embed_dim, hidden_dim)
        self.tl_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        self.br_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int], image_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        feat = self.stem(feature)
        tl_logits = self.tl_head(feat)
        br_logits = self.br_head(feat)
        tl_xy, tl_peak = _soft_argmax_2d(tl_logits, image_size=image_size)
        br_xy, br_peak = _soft_argmax_2d(br_logits, image_size=image_size)
        x1 = torch.min(tl_xy[:, 0], br_xy[:, 0])
        y1 = torch.min(tl_xy[:, 1], br_xy[:, 1])
        x2 = torch.max(tl_xy[:, 0], br_xy[:, 0])
        y2 = torch.max(tl_xy[:, 1], br_xy[:, 1])
        w = (x2 - x1).clamp_min(1.0e-3)
        h = (y2 - y1).clamp_min(1.0e-3)
        cx = x1 + w / 2
        cy = y1 + h / 2
        conf = 0.5 * (tl_peak + br_peak)
        bbox = torch.cat([torch.stack([cx, cy, w, h], dim=-1), conf], dim=-1)
        return {
            "bbox": bbox,
            "tl_logits": tl_logits,
            "br_logits": br_logits,
        }


class PupilSearchHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, 7, hidden_dim=hidden_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class PupilBBoxAuxHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 128)
        self.stem = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
        )
        self.box_branch = nn.Linear(hidden, 4)
        self.conf_branch = nn.Linear(hidden, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        feat = self.stem(pooled)
        raw_box = self.box_branch(feat)
        xy = raw_box[:, :2]
        wh = F.softplus(raw_box[:, 2:4]) + 1.0e-3
        conf = self.conf_branch(feat)
        return torch.cat([xy, wh, conf], dim=-1)


class PupilOBBAuxHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim, 128)
        self.stem = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
        )
        self.box_branch = nn.Linear(hidden, 5)
        self.conf_branch = nn.Linear(hidden, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        feat = self.stem(pooled)
        raw_box = self.box_branch(feat)
        xy = raw_box[:, :2]
        wh = F.softplus(raw_box[:, 2:4]) + 1.0e-3
        angle = torch.tanh(raw_box[:, 4:5]) * (torch.pi / 2.0)
        conf = self.conf_branch(feat)
        return torch.cat([xy, wh, angle, conf], dim=-1)


class EventSearchHead(nn.Module):
    def __init__(self, embed_dim: int = 192, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, 7, hidden_dim=hidden_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class PupilTrackHead(nn.Module):
    def __init__(self, in_dim: int = 384, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.net = _mlp_head(in_dim, 8, hidden_dim=hidden_dim)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class AuxStateHead(nn.Module):
    def __init__(self, embed_dim: int = 192, num_classes: int = 5, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, num_classes, hidden_dim=hidden_dim)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class SearchMaskHead(nn.Module):
    def __init__(self, embed_dim: int = 192, output_size: tuple[int, int] = (256, 256), hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim // 2, 32)
        self.output_size = tuple(int(v) for v in output_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int]) -> torch.Tensor:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        logits = self.decoder(feature)
        return F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)


class ProtoMaskHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        output_size: tuple[int, int] = (256, 256),
        hidden_dim: int | None = None,
        *,
        num_prototypes: int = 8,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim) if hidden_dim is not None else max(embed_dim // 2, 32)
        self.output_size = tuple(int(v) for v in output_size)
        self.num_prototypes = max(1, int(num_prototypes))
        self.prototype_decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_prototypes, kernel_size=1),
        )
        self.coefficient_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_prototypes),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        prototypes = self.prototype_decoder(feature)
        pooled = tokens.mean(dim=1)
        coeff = self.coefficient_head(pooled)
        coarse_logits = torch.einsum("bkhw,bk->bhw", prototypes, coeff).unsqueeze(1)
        coarse_logits = F.interpolate(coarse_logits, size=self.output_size, mode="bilinear", align_corners=False)
        mask_logits = self.refine(coarse_logits)
        return {
            "mask_logits": mask_logits,
            "mask_proto": prototypes,
            "mask_coeff": coeff,
            "mask_coarse_logits": coarse_logits,
        }
