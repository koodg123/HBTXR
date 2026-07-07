from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn

from src.models.heads import (
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
    SearchMaskHead,
    TrackCenterCandidateHead,
    TrackCenterHeatmapHead,
    TrackCenterRefineHead,
    TrackStateAuxHead,
    TrackStateSimDRHead,
    YOLO26EyeRegionHead,
    YOLODetectEyeRegionHead,
    YOLOPointEyeRegionHead,
)

HeadBuilder = type[nn.Module] | Callable[..., nn.Module]

_HEADS: dict[str, HeadBuilder] = {
    "eye_region": EyeRegionHead,
    "dense_eye_region": DenseEyeRegionHead,
    "roi_cascade_bbox": RoiCascadeBBoxHead,
    "eye_guided_bbox": EyeGuidedBBoxHead,
    "yolo26_bbox": YOLO26EyeRegionHead,
    "yolo26_point": YOLOPointEyeRegionHead,
    "yolo_detect": YOLODetectEyeRegionHead,
    "pupil_search": PupilSearchHead,
    "event_search": EventSearchHead,
    "pupil_track": PupilTrackHead,
    "pupil_bbox_aux": PupilBBoxAuxHead,
    "pupil_obb_aux": PupilOBBAuxHead,
    "track_state_aux": TrackStateAuxHead,
    "track_state_simdr": TrackStateSimDRHead,
    "track_center_heatmap": TrackCenterHeatmapHead,
    "track_center_refine": TrackCenterRefineHead,
    "track_center_candidate": TrackCenterCandidateHead,
    "search_center_candidate": SearchCenterCandidateHead,
    "aux_state": AuxStateHead,
    "search_mask": SearchMaskHead,
    "proto_mask": ProtoMaskHead,
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def list_head_names() -> list[str]:
    return sorted(_HEADS.keys())


def get_head_class(name: str) -> HeadBuilder:
    normalized = _normalize_name(name)
    try:
        return _HEADS[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown head: {normalized}") from exc


def build_head(name: str, **kwargs: Any) -> nn.Module:
    return get_head_class(name)(**kwargs)
