from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LoadedSampleAssets:
    annotation: dict[str, Any]
    prev_annotation: dict[str, Any]
    frame_sensor: np.ndarray
    resolved_frame_path: Path | None
    resolved_store_path: Path | None
    sensor_size_wh: tuple[int, int]
    end_timestamp_us: int


@dataclass(frozen=True)
class ResolvedRoi:
    roi_xywh: tuple[float, float, float, float]
    meta: dict[str, Any]


@dataclass(frozen=True)
class BuiltEventInput:
    event: np.ndarray
    selected_count: int
    meta: dict[str, Any]


@dataclass(frozen=True)
class TargetBundle:
    cur_state: np.ndarray
    prev_state: np.ndarray
    pupil_region_target: list[float]
    eye_target_box: list[float]
    annotation_quality: float
    similarity_target: float
    closed_eye_flag: float
    mask_valid: float
    valid_track: float
    confidence_target: float
    pupil_track_target: np.ndarray
    aux_target: int
    sample_timestamp_us: int
