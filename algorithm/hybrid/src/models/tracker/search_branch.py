from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SearchMaskGuidanceRefiner:
    def __init__(
        self,
        *,
        use_mask_centroid: bool,
        use_mask_cascade: bool,
        cascade_detach: bool,
        cascade_use_xy: bool,
        cascade_use_ab: bool,
        cascade_xy_scale: float,
        cascade_ab_scale: float,
        cascade_ab_mode: str,
        cascade_ab_limit: float,
        cascade_min_ab: float,
    ) -> None:
        self.use_mask_centroid = bool(use_mask_centroid)
        self.use_mask_cascade = bool(use_mask_cascade)
        self.cascade_detach = bool(cascade_detach)
        self.cascade_use_xy = bool(cascade_use_xy)
        self.cascade_use_ab = bool(cascade_use_ab)
        self.cascade_xy_scale = float(cascade_xy_scale)
        self.cascade_ab_scale = float(cascade_ab_scale)
        self.cascade_ab_mode = str(cascade_ab_mode).strip().lower()
        self.cascade_ab_limit = float(cascade_ab_limit)
        self.cascade_min_ab = float(cascade_min_ab)

    @staticmethod
    def soft_mask_statistics(mask_logits: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
        prob = torch.sigmoid(mask_logits)
        if prob.ndim == 4 and prob.shape[1] == 1:
            prob = prob[:, 0]
        height, width = prob.shape[-2:]
        xs = torch.arange(width, device=prob.device, dtype=prob.dtype).view(1, 1, width)
        ys = torch.arange(height, device=prob.device, dtype=prob.dtype).view(1, height, 1)
        mass = prob.sum(dim=(1, 2)).clamp_min(eps)
        cx = (prob * xs).sum(dim=(1, 2)) / mass
        cy = (prob * ys).sum(dim=(1, 2)) / mass
        center = torch.stack([cx, cy], dim=-1)
        dx = xs - cx.view(-1, 1, 1)
        dy = ys - cy.view(-1, 1, 1)
        cov_xx = (prob * dx * dx).sum(dim=(1, 2)) / mass
        cov_yy = (prob * dy * dy).sum(dim=(1, 2)) / mass
        cov_xy = (prob * dx * dy).sum(dim=(1, 2)) / mass
        trace = cov_xx + cov_yy
        delta = torch.sqrt(torch.clamp((cov_xx - cov_yy) ** 2 + 4.0 * (cov_xy ** 2), min=0.0))
        lambda_major = ((trace + delta) * 0.5).clamp_min(eps)
        lambda_minor = ((trace - delta) * 0.5).clamp_min(eps)
        axes = 2.0 * torch.sqrt(torch.stack([lambda_major, lambda_minor], dim=-1))
        axes, _ = torch.sort(axes, dim=-1, descending=True)
        return center, axes

    def apply_mask_cascade(self, search_logits: torch.Tensor, mask_center: torch.Tensor, mask_axes: torch.Tensor) -> torch.Tensor:
        cascade_center = mask_center.detach() if self.cascade_detach else mask_center
        cascade_axes = mask_axes.detach() if self.cascade_detach else mask_axes
        xy = search_logits[:, 0:2]
        ab = search_logits[:, 2:4]
        tail = search_logits[:, 4:]
        if self.cascade_use_xy:
            xy = cascade_center + xy * self.cascade_xy_scale
        if self.cascade_use_ab:
            if self.cascade_ab_mode == "additive":
                ab = (cascade_axes + ab * self.cascade_ab_scale).clamp_min(self.cascade_min_ab)
            else:
                scaled = torch.clamp(
                    ab * self.cascade_ab_scale,
                    min=-self.cascade_ab_limit,
                    max=self.cascade_ab_limit,
                )
                ab = (cascade_axes * torch.exp(scaled)).clamp_min(self.cascade_min_ab)
        return torch.cat([xy, ab, tail], dim=-1)

    def refine(
        self,
        *,
        search_logits: torch.Tensor | None,
        mask_logits: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, torch.Tensor]]:
        if search_logits is None or mask_logits is None:
            return search_logits, {}
        if not (self.use_mask_centroid or self.use_mask_cascade):
            return search_logits, {}
        mask_center, mask_axes = self.soft_mask_statistics(mask_logits)
        extras = {
            "search/mask_center": mask_center,
            "search/mask_axes": mask_axes,
        }
        if self.use_mask_cascade:
            return self.apply_mask_cascade(search_logits, mask_center, mask_axes), extras
        if self.use_mask_centroid:
            return torch.cat([mask_center, search_logits[:, 2:]], dim=-1), extras
        return search_logits, extras


class SearchBranch:
    def __init__(
        self,
        *,
        input_size: tuple[int, int],
        eye_head: nn.Module | None,
        search_head: nn.Module | None,
        search_bbox_aux_head: nn.Module | None,
        search_obb_aux_head: nn.Module | None,
        roi_bbox_head: nn.Module | None,
        mask_head: nn.Module | None,
        aux_head: nn.Module | None,
        eye_detector_mode: str,
        eye_head_variant: str,
        roi_bbox_use_external_eye: bool,
        refiner: SearchMaskGuidanceRefiner,
    ) -> None:
        self.input_size = tuple(int(v) for v in input_size)
        self.eye_head = eye_head
        self.search_head = search_head
        self.search_bbox_aux_head = search_bbox_aux_head
        self.search_obb_aux_head = search_obb_aux_head
        self.roi_bbox_head = roi_bbox_head
        self.mask_head = mask_head
        self.aux_head = aux_head
        self.eye_detector_mode = str(eye_detector_mode).strip().lower()
        self.eye_head_variant = str(eye_head_variant).strip().lower()
        self.roi_bbox_use_external_eye = bool(roi_bbox_use_external_eye)
        self.refiner = refiner

    @staticmethod
    def state_from_branch(branch_logits: torch.Tensor) -> torch.Tensor:
        return branch_logits[..., :6]

    def forward(
        self,
        *,
        tokens: torch.Tensor,
        pooled: torch.Tensor,
        grid_size: tuple[int, int],
        active_dim: int,
        device: torch.device,
        batch_size: int,
    ) -> dict[str, torch.Tensor]:
        outputs = {
            "search/pooled": pooled,
            "pruning/active_dim": torch.full((batch_size,), float(active_dim), device=device),
        }
        search_logits = None

        if self.eye_head is not None:
            if self.eye_detector_mode == "dense":
                eye_outputs = self.eye_head(tokens, grid_size=grid_size)
                outputs["search/eye"] = eye_outputs["eye"]
                outputs["search/eye_boxes"] = eye_outputs["eye_boxes"]
                outputs["search/eye_obj_logits"] = eye_outputs["eye_obj_logits"]
                outputs["search/eye_anchor_points"] = eye_outputs["eye_anchor_points"]
            elif self.eye_head_variant in {"yolo26_point", "yolo_detect", "sot_center", "sot_corner"}:
                eye_outputs = self.eye_head(tokens, grid_size=grid_size, image_size=self.input_size)
                outputs["search/eye"] = eye_outputs["bbox"]
                if "det_cls_logits" in eye_outputs:
                    outputs["search/eye_det_cls_logits"] = eye_outputs["det_cls_logits"]
                    outputs["search/eye_det_reg_logits"] = eye_outputs["det_reg_logits"]
                if "det_quality_logits" in eye_outputs:
                    outputs["search/eye_det_quality_logits"] = eye_outputs["det_quality_logits"]
                if "point_cls_logits" in eye_outputs:
                    outputs["search/eye_point_cls_logits"] = eye_outputs["point_cls_logits"]
                    outputs["search/eye_point_reg_logits"] = eye_outputs["point_reg_logits"]
                if "center_logits" in eye_outputs:
                    outputs["search/eye_center_logits"] = eye_outputs["center_logits"]
                    outputs["search/eye_size_map"] = eye_outputs["size_map"]
                    outputs["search/eye_offset_map"] = eye_outputs["offset_map"]
                if "tl_logits" in eye_outputs:
                    outputs["search/eye_tl_logits"] = eye_outputs["tl_logits"]
                    outputs["search/eye_br_logits"] = eye_outputs["br_logits"]
            else:
                outputs["search/eye"] = self.eye_head(pooled)

        if self.search_head is not None:
            search_logits = self.search_head(pooled)
        if self.search_bbox_aux_head is not None:
            outputs["search/pupil_bbox"] = self.search_bbox_aux_head(pooled)
        if self.search_obb_aux_head is not None:
            outputs["search/pupil_obb"] = self.search_obb_aux_head(pooled)

        if self.roi_bbox_head is not None:
            if self.roi_bbox_use_external_eye:
                eye_seed = outputs.get("search/eye")
                if eye_seed is None:
                    raise RuntimeError("roi_bbox_head.use_external_eye=true requires search/eye output from the eye head")
                roi_outputs = self.roi_bbox_head(tokens, eye_seed=eye_seed)
            else:
                roi_outputs = self.roi_bbox_head(tokens, pooled=pooled, grid_size=grid_size)
            outputs["search/eye"] = roi_outputs["eye"]
            outputs["search/pupil_bbox"] = roi_outputs["pupil_bbox"]
            outputs["search/roi_boxes_xyxy"] = roi_outputs["roi_boxes_xyxy"]
            outputs["search/roi_tokens"] = roi_outputs["roi_tokens"]

        if self.mask_head is not None:
            mask_outputs = self.mask_head(tokens, grid_size=grid_size)
            if torch.is_tensor(mask_outputs):
                outputs["search/mask_logits"] = mask_outputs
            else:
                outputs["search/mask_logits"] = mask_outputs["mask_logits"]
                if "mask_proto" in mask_outputs:
                    outputs["search/mask_proto"] = mask_outputs["mask_proto"]
                if "mask_coeff" in mask_outputs:
                    outputs["search/mask_coeff"] = mask_outputs["mask_coeff"]
                if "mask_coarse_logits" in mask_outputs:
                    outputs["search/mask_coarse_logits"] = mask_outputs["mask_coarse_logits"]
            search_logits, extra_outputs = self.refiner.refine(
                search_logits=search_logits,
                mask_logits=outputs.get("search/mask_logits"),
            )
            outputs.update(extra_outputs)

        if search_logits is not None:
            outputs["search/pupil"] = search_logits
            outputs["search/state"] = self.state_from_branch(search_logits)
        if self.aux_head is not None:
            outputs["search/aux"] = self.aux_head(pooled)
        return outputs
