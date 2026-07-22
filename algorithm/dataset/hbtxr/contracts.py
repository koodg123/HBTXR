from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    NDArray = np.ndarray[Any, Any]
else:
    NDArray = Any


@dataclass(frozen=True)
class LoadedSampleAssets:
    annotation: dict[str, Any]
    prev_annotation: dict[str, Any]
    frame_sensor: NDArray
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
    event: NDArray
    selected_count: int
    meta: dict[str, Any]


@dataclass(frozen=True)
class TargetBundle:
    cur_state: NDArray
    prev_state: NDArray
    pupil_region_target: list[float]
    eye_target_box: list[float]
    annotation_quality: float
    similarity_target: float
    closed_eye_flag: float
    mask_valid: float
    valid_track: float
    confidence_target: float
    pupil_track_target: NDArray
    aux_target: int
    sample_timestamp_us: int


class ContractError(ValueError):
    """Raised when HANDOVER data cannot satisfy an explicit HBTXR contract."""


@dataclass(frozen=True)
class EventTensorSpec:
    """Exact tensor semantics; resolution never mutates the payload."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    axes: tuple[str, ...]
    polarity_semantics: str
    temporal_semantics: str


_EVENT_TENSOR_SPECS = (
    EventTensorSpec(
        "polarity_count_volume_raw", (2, 64, 64), "uint16",
        ("polarity", "height", "width"),
        "separate-negative-positive", "single-window-count",
    ),
    EventTensorSpec(
        "polarity_count_volume_postprocessed", (2, 64, 64), "float32",
        ("polarity", "height", "width"),
        "separate-negative-positive", "single-window-count",
    ),
    EventTensorSpec(
        "sequence_polarity_50", (50, 2, 64, 64), "float32",
        ("sequence", "polarity", "height", "width"),
        "separate-negative-positive", "ordered-50-frame-sequence",
    ),
    EventTensorSpec(
        "tdtracker_sequence_polarity_100", (100, 2, 64, 64), "float32",
        ("sequence", "polarity", "height", "width"),
        "separate-negative-positive", "ordered-100-frame-sequence",
    ),
    EventTensorSpec(
        "ervt_signed_voxel_30", (30, 3, 64, 64), "float32",
        ("sequence", "voxel-time-bin", "height", "width"),
        "signed-voxel", "ordered-30-frame-signed-voxel-sequence",
    ),
)
_LOWERCASE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def resolve_event_tensor_spec(
    *,
    shape: tuple[int, ...],
    dtype: str,
    axes: tuple[str, ...],
    polarity_semantics: str,
    temporal_semantics: str,
) -> EventTensorSpec:
    """Resolve an exact known representation without implicit coercion."""

    for spec in _EVENT_TENSOR_SPECS:
        if (
            spec.shape == shape
            and spec.dtype == dtype
            and spec.axes == axes
            and spec.polarity_semantics == polarity_semantics
            and spec.temporal_semantics == temporal_semantics
        ):
            return spec
    raise ContractError(
        "unsupported event tensor contract: "
        f"shape={shape!r}, dtype={dtype!r}, axes={axes!r}, "
        f"polarity_semantics={polarity_semantics!r}, "
        f"temporal_semantics={temporal_semantics!r}"
    )


@dataclass(frozen=True)
class ContractedEventTensor:
    payload: Any
    spec: EventTensorSpec


def bind_event_tensor(
    payload: Any,
    *,
    shape: tuple[int, ...],
    dtype: str,
    axes: tuple[str, ...],
    polarity_semantics: str,
    temporal_semantics: str,
) -> ContractedEventTensor:
    """Bind metadata without copying, reshaping, transposing, or reordering."""

    spec = resolve_event_tensor_spec(
        shape=shape,
        dtype=dtype,
        axes=axes,
        polarity_semantics=polarity_semantics,
        temporal_semantics=temporal_semantics,
    )
    return ContractedEventTensor(payload=payload, spec=spec)


@dataclass(frozen=True)
class CanonicalSampleMetadata:
    event_ref: str
    frame_ref: str | None
    target_ref: str
    subject_id: str
    eye: str
    sequence_id: str
    start_timestamp_us: int
    end_timestamp_us: int
    dataset: str
    split: str
    manifest_sha256: str
    sample_id: str

    def __post_init__(self) -> None:
        identity_fields = (
            ("event_ref", self.event_ref),
            ("target_ref", self.target_ref),
            ("subject_id", self.subject_id),
            ("sequence_id", self.sequence_id),
            ("dataset", self.dataset),
            ("split", self.split),
            ("sample_id", self.sample_id),
        )
        for field_name, value in identity_fields:
            if not isinstance(value, str) or not value.strip():
                raise ContractError(f"{field_name} must be a non-empty string")
        if self.frame_ref is not None and (
            not isinstance(self.frame_ref, str) or not self.frame_ref.strip()
        ):
            raise ContractError("frame_ref must be None or a non-empty string")
        if self.eye not in {"left", "right"}:
            raise ContractError("eye must be exactly 'left' or 'right'")
        if (
            isinstance(self.start_timestamp_us, bool)
            or not isinstance(self.start_timestamp_us, int)
            or self.start_timestamp_us < 0
        ):
            raise ContractError("start_timestamp_us must be a non-negative integer")
        if (
            isinstance(self.end_timestamp_us, bool)
            or not isinstance(self.end_timestamp_us, int)
            or self.end_timestamp_us <= self.start_timestamp_us
        ):
            raise ContractError("end_timestamp_us must be greater than start_timestamp_us")
        if (
            not isinstance(self.manifest_sha256, str)
            or _LOWERCASE_SHA256.fullmatch(self.manifest_sha256) is None
        ):
            raise ContractError("manifest_sha256 must be a lowercase SHA-256 digest")
