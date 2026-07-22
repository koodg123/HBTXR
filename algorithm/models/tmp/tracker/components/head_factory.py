from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ..heads import (
    AuxStateHead,
    DenseEyeRegionHead,
    EventSearchHead,
    EyeGuidedBBoxHead,
    EyeRegionHead,
    ProtoMaskHead,
    PupilBBoxAuxHead,
    PupilOBBAuxHead,
    PupilSearchHead,
    PupilTrackHead,
    RoiCascadeBBoxHead,
    SearchCenterCandidateHead,
    SOTCenterPredictor,
    SOTCornerPredictor,
    SearchMaskHead,
    TrackCenterCandidateHead,
    TrackCenterRefineHead,
    TrackStateAuxHead,
    TrackCenterHeatmapHead,
    TrackStateSimDRHead,
    YOLO26EyeRegionHead,
    YOLODetectEyeRegionHead,
    YOLOPointEyeRegionHead,
)


@dataclass(frozen=True)
class TrackerHeadModules:
    eye_head: nn.Module | None
    search_head: nn.Module | None
    event_head: nn.Module | None
    search_bbox_aux_head: nn.Module | None
    event_bbox_aux_head: nn.Module | None
    search_obb_aux_head: nn.Module | None
    event_obb_aux_head: nn.Module | None
    search_center_candidate_head: nn.Module | None
    roi_bbox_head: nn.Module | None
    track_head: nn.Module | None
    track_state_aux_head: nn.Module | None
    track_state_simdr_head: nn.Module | None
    track_center_heatmap_head: nn.Module | None
    track_center_refine_head: nn.Module | None
    track_center_candidate_head: nn.Module | None
    mask_head: nn.Module | None
    aux_head: nn.Module | None


class TrackerHeadFactory:
    def __init__(
        self,
        *,
        embed_dim: int,
        input_size: tuple[int, int],
        aux_classes: int,
        head_hidden_dim: int | None,
        track_head_hidden_dim: int | None,
        mask_hidden_dim: int | None,
        eye_detector_mode: str,
        eye_head_variant: str,
        eye_reg_max: int,
        mask_variant: str,
        roi_bbox_cfg: dict[str, object],
        search_use_roi_bbox_head: bool,
        roi_bbox_use_external_eye: bool,
        mask_cfg: dict[str, object],
        search_head_variant: str = "legacy",
        search_head_residual_hidden_dim: int | None = None,
        search_center_candidate_count: int = 4,
        search_center_candidate_max_delta_px: float = 8.0,
        track_state_simdr_bins: int = 64,
        track_center_heatmap_grid: int = 32,
        track_center_refine_max_delta_px: float = 4.0,
        track_center_candidate_count: int = 4,
        track_center_candidate_max_delta_px: float = 8.0,
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.input_size = tuple(int(v) for v in input_size)
        self.aux_classes = int(aux_classes)
        self.head_hidden_dim = head_hidden_dim
        self.search_head_variant = str(search_head_variant or "legacy").strip().lower()
        self.search_head_residual_hidden_dim = search_head_residual_hidden_dim
        self.search_center_candidate_count = max(1, int(search_center_candidate_count))
        self.search_center_candidate_max_delta_px = max(float(search_center_candidate_max_delta_px), 1.0e-6)
        self.track_head_hidden_dim = track_head_hidden_dim
        self.mask_hidden_dim = mask_hidden_dim
        self.eye_detector_mode = str(eye_detector_mode).strip().lower()
        self.eye_head_variant = str(eye_head_variant).strip().lower()
        self.eye_reg_max = max(1, int(eye_reg_max))
        self.mask_variant = str(mask_variant).strip().lower()
        self.roi_bbox_cfg = dict(roi_bbox_cfg or {})
        self.search_use_roi_bbox_head = bool(search_use_roi_bbox_head)
        self.roi_bbox_use_external_eye = bool(roi_bbox_use_external_eye)
        self.mask_cfg = dict(mask_cfg or {})
        self.track_state_simdr_bins = max(2, int(track_state_simdr_bins))
        self.track_center_heatmap_grid = max(2, int(track_center_heatmap_grid))
        self.track_center_refine_max_delta_px = max(float(track_center_refine_max_delta_px), 1.0e-6)
        self.track_center_candidate_count = max(1, int(track_center_candidate_count))
        self.track_center_candidate_max_delta_px = max(float(track_center_candidate_max_delta_px), 1.0e-6)

    def build(
        self,
        *,
        enable_eye_head: bool,
        enable_search_head: bool,
        enable_event_head: bool,
        enable_track_head: bool,
        enable_mask_head: bool,
        enable_aux_head: bool,
        enable_search_bbox_aux_head: bool,
        enable_event_bbox_aux_head: bool,
        enable_search_obb_aux_head: bool,
        enable_event_obb_aux_head: bool,
        enable_track_state_aux_head: bool,
        enable_search_center_candidate_head: bool = False,
        enable_track_state_simdr_head: bool = False,
        enable_track_center_heatmap_head: bool = False,
        enable_track_center_refine_head: bool = False,
        enable_track_center_candidate_head: bool = False,
    ) -> TrackerHeadModules:
        return TrackerHeadModules(
            eye_head=self.build_eye_head(enable_eye_head),
            search_head=PupilSearchHead(
                embed_dim=self.embed_dim,
                hidden_dim=self.head_hidden_dim,
                variant=self.search_head_variant,
                residual_hidden_dim=self.search_head_residual_hidden_dim,
            )
            if enable_search_head
            else None,
            event_head=EventSearchHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim) if enable_event_head else None,
            search_bbox_aux_head=PupilBBoxAuxHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim) if enable_search_bbox_aux_head else None,
            event_bbox_aux_head=PupilBBoxAuxHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim) if enable_event_bbox_aux_head else None,
            search_obb_aux_head=PupilOBBAuxHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim) if enable_search_obb_aux_head else None,
            event_obb_aux_head=PupilOBBAuxHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim) if enable_event_obb_aux_head else None,
            search_center_candidate_head=SearchCenterCandidateHead(
                in_dim=self.embed_dim,
                hidden_dim=self.head_hidden_dim,
                num_candidates=self.search_center_candidate_count,
                max_delta_px=self.search_center_candidate_max_delta_px,
            )
            if enable_search_center_candidate_head
            else None,
            roi_bbox_head=self.build_roi_bbox_head(),
            track_head=PupilTrackHead(in_dim=self.embed_dim * 2, hidden_dim=self.track_head_hidden_dim) if enable_track_head else None,
            track_state_aux_head=TrackStateAuxHead(in_dim=self.embed_dim * 2, hidden_dim=self.track_head_hidden_dim) if enable_track_state_aux_head else None,
            track_state_simdr_head=TrackStateSimDRHead(in_dim=self.embed_dim * 2, hidden_dim=self.track_head_hidden_dim, bins=self.track_state_simdr_bins)
            if enable_track_state_simdr_head
            else None,
            track_center_heatmap_head=TrackCenterHeatmapHead(
                in_dim=self.embed_dim * 2,
                hidden_dim=self.track_head_hidden_dim,
                grid_size=self.track_center_heatmap_grid,
            )
            if enable_track_center_heatmap_head
            else None,
            track_center_refine_head=TrackCenterRefineHead(
                in_dim=self.embed_dim * 2,
                hidden_dim=self.track_head_hidden_dim,
                max_delta_px=self.track_center_refine_max_delta_px,
            )
            if enable_track_center_refine_head
            else None,
            track_center_candidate_head=TrackCenterCandidateHead(
                in_dim=self.embed_dim * 2,
                hidden_dim=self.track_head_hidden_dim,
                num_candidates=self.track_center_candidate_count,
                max_delta_px=self.track_center_candidate_max_delta_px,
            )
            if enable_track_center_candidate_head
            else None,
            mask_head=self.build_mask_head(enable_mask_head),
            aux_head=AuxStateHead(embed_dim=self.embed_dim, num_classes=self.aux_classes, hidden_dim=self.head_hidden_dim) if enable_aux_head else None,
        )

    def build_eye_head(self, enabled: bool) -> nn.Module | None:
        if not enabled:
            return None
        if self.eye_detector_mode == "dense":
            return DenseEyeRegionHead(embed_dim=self.embed_dim, output_size=self.input_size, hidden_dim=self.head_hidden_dim)
        if self.eye_head_variant == "yolo26_bbox":
            return YOLO26EyeRegionHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim)
        if self.eye_head_variant == "yolo26_point":
            return YOLOPointEyeRegionHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim, reg_max=self.eye_reg_max)
        if self.eye_head_variant == "yolo_detect":
            return YOLODetectEyeRegionHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim, reg_max=self.eye_reg_max)
        if self.eye_head_variant == "sot_center":
            return SOTCenterPredictor(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim)
        if self.eye_head_variant == "sot_corner":
            return SOTCornerPredictor(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim)
        if self.eye_head_variant in {"", "legacy"}:
            return EyeRegionHead(embed_dim=self.embed_dim, hidden_dim=self.head_hidden_dim)
        raise ValueError(
            f"Unsupported eye_head_variant: {self.eye_head_variant!r}. "
            "Expected one of: legacy, yolo26_bbox, yolo26_point, yolo_detect, sot_center, sot_corner."
        )

    def build_roi_bbox_head(self) -> nn.Module | None:
        if not self.search_use_roi_bbox_head:
            return None
        if self.roi_bbox_use_external_eye:
            return EyeGuidedBBoxHead(
                embed_dim=self.embed_dim,
                output_size=self.input_size,
                mlp1_hidden_dim=self.roi_bbox_cfg.get("mlp1_hidden_dim", self.head_hidden_dim),
                attn_heads=int(self.roi_bbox_cfg.get("attn_heads", max(1, self.embed_dim // 64))),
                attn_layers=int(self.roi_bbox_cfg.get("attn_layers", 1)),
                attn_mlp_ratio=float(self.roi_bbox_cfg.get("attn_mlp_ratio", 2.0)),
                detach_eye_boxes=bool(self.roi_bbox_cfg.get("detach_eye_boxes", True)),
                min_box_size=float(self.roi_bbox_cfg.get("min_box_size", 4.0)),
            )
        return RoiCascadeBBoxHead(
            embed_dim=self.embed_dim,
            output_size=self.input_size,
            mlp0_hidden_dim=self.roi_bbox_cfg.get("mlp0_hidden_dim", self.head_hidden_dim),
            mlp1_hidden_dim=self.roi_bbox_cfg.get("mlp1_hidden_dim", self.head_hidden_dim),
            roi_size=tuple(self.roi_bbox_cfg.get("roi_size", [4, 4])),
            attn_heads=int(self.roi_bbox_cfg.get("attn_heads", max(1, self.embed_dim // 64))),
            attn_layers=int(self.roi_bbox_cfg.get("attn_layers", 1)),
            attn_mlp_ratio=float(self.roi_bbox_cfg.get("attn_mlp_ratio", 2.0)),
            use_roi_align=bool(self.roi_bbox_cfg.get("use_roi_align", True)),
            detach_eye_boxes=bool(self.roi_bbox_cfg.get("detach_eye_boxes", True)),
            min_box_size=float(self.roi_bbox_cfg.get("min_box_size", 4.0)),
        )

    def build_mask_head(self, enabled: bool) -> nn.Module | None:
        if not enabled:
            return None
        if self.mask_variant in {"", "legacy"}:
            return SearchMaskHead(embed_dim=self.embed_dim, output_size=self.input_size, hidden_dim=self.mask_hidden_dim)
        if self.mask_variant in {"proto", "proto26"}:
            return ProtoMaskHead(
                embed_dim=self.embed_dim,
                output_size=self.input_size,
                hidden_dim=self.mask_hidden_dim,
                num_prototypes=int(self.mask_cfg.get("num_prototypes", 8)),
            )
        raise ValueError(
            f"Unsupported mask_variant: {self.mask_variant!r}. "
            "Expected one of: legacy, proto26."
        )
