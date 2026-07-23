from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .event_repr import build_event_frame
from .io import build_cache_file, legacy_track_delta_to_uv, load_or_build_npz_array, read_jsonl, resolve_stored_path, xywht_to_xyabuv
from .transforms import SpatialTransform, apply_transform_to_event, apply_transform_to_frame, apply_transform_to_mask, build_transform


DEFAULT_INPUT_SIZE = (256, 256)
DEFAULT_EVENT_BUILDER = {
    "policy": "fixed_count",
    "time_bin_us": 5000,
    "event_count_target": 5000,
    "accumulation": "causal_linear",
    "causal_weight_power": 1.0,
    "polarity_split": True,
}


def _to_tensor_image(arr: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(arr.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _resolve_xywh(row: dict[str, Any], *keys: str, default: list[float] | None = None) -> list[float]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return [float(v) for v in value]
    if default is None:
        raise KeyError(f"Missing any of keys: {keys}")
    return [float(v) for v in default]


def _ellipse_to_bbox(ellipse_xywht: list[float]) -> list[float]:
    x, y, a, b, _ = [float(v) for v in ellipse_xywht]
    return [x - 0.5 * a, y - 0.5 * b, a, b]


def _state6_from_annotation(annotation: dict[str, Any]) -> np.ndarray:
    if annotation.get("state_xyabuv") is not None:
        return np.asarray(annotation["state_xyabuv"], dtype=np.float32)
    ellipse = (
        annotation.get("pupil_ellipse_xywht_sensor")
        or annotation.get("ellipse_sensor_xywht")
        or annotation.get("ellipse_xywht")
        or annotation.get("ellipse_frame_xywht")
    )
    if ellipse is None:
        raise KeyError("Annotation is missing ellipse/state information")
    return np.asarray(xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32)), dtype=np.float32)


def _xywht_from_state6(state6: np.ndarray) -> np.ndarray:
    state6 = np.asarray(state6, dtype=np.float32)
    x, y, a, b, u, v = state6.tolist()
    theta = 0.5 * np.arctan2(u, v)
    return np.asarray([x, y, a, b, theta], dtype=np.float32)


def _normalize_uv(state6: np.ndarray) -> np.ndarray:
    state6 = np.asarray(state6, dtype=np.float32).copy()
    norm = np.linalg.norm(state6[4:6])
    if norm < 1e-6:
        state6[4:6] = np.asarray([0.0, 1.0], dtype=np.float32)
    else:
        state6[4:6] = state6[4:6] / norm
    return state6


class FECETHBTXRDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        *,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        resize_policy: str | None = "facet_square_direct",
        event_builder: dict[str, Any] | None = None,
        canonical_root: str | None = None,
        cache_root: str | None = None,
        use_cache: bool = True,
        per_channel_normalize: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.rows = read_jsonl(self.manifest_path)
        self.input_size = tuple(int(v) for v in input_size)
        self.resize_policy = None if resize_policy is None or not str(resize_policy).strip() else str(resize_policy)
        self.event_builder = {**DEFAULT_EVENT_BUILDER, **(event_builder or {})}
        self.canonical_root = None if canonical_root is None else Path(canonical_root)
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.use_cache = bool(use_cache)
        self.per_channel_normalize = bool(per_channel_normalize)
        self._annotation_store_cache: dict[Path, dict[str, dict[str, Any]]] = {}
        self._events_cache: dict[Path, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_path(self, raw_path: str | Path) -> Path:
        if self.canonical_root is not None:
            return resolve_stored_path(self.canonical_root, raw_path)
        return resolve_stored_path(self.manifest_path.parent, raw_path)

    def _load_annotation(self, ref: dict[str, Any]) -> dict[str, Any]:
        store_path = self._resolve_path(ref["annotation_store_path"])
        cache = self._annotation_store_cache.get(store_path)
        if cache is None:
            rows = read_jsonl(store_path)
            cache = {str(row["ann_id"]): row for row in rows}
            self._annotation_store_cache[store_path] = cache
        ann_id = str(ref["ann_id"])
        if ann_id not in cache:
            raise KeyError(f"Annotation id not found: {ann_id} @ {store_path}")
        return cache[ann_id]

    def _load_events(self, events_path: str | Path) -> dict[str, np.ndarray]:
        resolved = self._resolve_path(events_path)
        cached = self._events_cache.get(resolved)
        if cached is None:
            data = np.load(resolved)
            cached = {key: np.asarray(data[key]) for key in data.files}
            self._events_cache[resolved] = cached
        return cached

    def _load_mask(self, annotation: dict[str, Any], *, sensor_size_wh: tuple[int, int], roi_xywh: tuple[float, float, float, float]) -> np.ndarray:
        mask_path = annotation.get("pupil_mask_path") or annotation.get("mask_path")
        if not mask_path:
            return np.zeros((sensor_size_wh[1], sensor_size_wh[0]), dtype=np.uint8)
        mask = np.asarray(Image.open(self._resolve_path(mask_path)).convert("L"))
        if mask.shape[1] == sensor_size_wh[0] and mask.shape[0] == sensor_size_wh[1]:
            return mask
        if mask.shape[1] == int(round(roi_xywh[2])) and mask.shape[0] == int(round(roi_xywh[3])):
            full = np.zeros((sensor_size_wh[1], sensor_size_wh[0]), dtype=np.uint8)
            x, y, w, h = [int(round(v)) for v in roi_xywh]
            x1 = max(0, min(sensor_size_wh[0], x + w))
            y1 = max(0, min(sensor_size_wh[1], y + h))
            sx = max(0, -x)
            sy = max(0, -y)
            full[max(0, y):y1, max(0, x):x1] = mask[sy:sy + (y1 - max(0, y)), sx:sx + (x1 - max(0, x))]
            return full
        return np.zeros((sensor_size_wh[1], sensor_size_wh[0]), dtype=np.uint8)

    def _resolve_resize_policy(self, row: dict[str, Any]) -> str:
        if self.resize_policy is not None:
            return self.resize_policy
        return str(row.get("resize_policy", "facet_square_direct"))

    def _resolve_event_window(self, row: dict[str, Any], *, end_timestamp_us: int) -> dict[str, Any]:
        manifest_event_window = dict(row.get("event_window") or {})
        effective = {**manifest_event_window, **self.event_builder}
        effective_end = int(manifest_event_window.get("end_timestamp_us", end_timestamp_us))
        effective["end_timestamp_us"] = effective_end
        policy = str(effective.get("policy", DEFAULT_EVENT_BUILDER["policy"]))
        if policy == "time_bin":
            effective["start_timestamp_us"] = int(effective_end - int(effective.get("time_bin_us", DEFAULT_EVENT_BUILDER["time_bin_us"])))
        elif manifest_event_window.get("start_timestamp_us") is not None:
            effective["start_timestamp_us"] = int(manifest_event_window["start_timestamp_us"])
        return effective

    def _build_event_input(self, row: dict[str, Any], transform: SpatialTransform, *, sensor_size_wh: tuple[int, int], end_timestamp_us: int) -> tuple[np.ndarray, int]:
        effective_event_window = self._resolve_event_window(row, end_timestamp_us=end_timestamp_us)
        cache_payload = {
            "sample_id": row["sample_id"],
            "resize_policy": transform.policy,
            "target_size": list(self.input_size),
            "event_window": effective_event_window,
        }
        events_path = row["events_npz"]

        def _builder() -> np.ndarray:
            voxel_sensor, selected_count = build_event_frame(
                self._load_events(events_path),
                sensor_size_wh=sensor_size_wh,
                end_timestamp_us=end_timestamp_us,
                event_window=effective_event_window,
            )
            built = apply_transform_to_event(voxel_sensor, transform).astype(np.float32)
            if self.per_channel_normalize:
                for channel in range(built.shape[0]):
                    denom = float(np.max(np.abs(built[channel])))
                    if denom > 1e-6:
                        built[channel] /= denom
            return np.concatenate([built, np.asarray([[selected_count]], dtype=np.float32)], axis=None)

        if self.use_cache and self.cache_root is not None:
            cache_file = build_cache_file(self.cache_root, "event", row["sample_id"], cache_payload)
            packed, _ = load_or_build_npz_array(cache_file, "array", _builder)
        else:
            packed = _builder()
        event = np.asarray(packed[:-1], dtype=np.float32).reshape(2, self.input_size[1], self.input_size[0])
        selected_count = int(round(float(packed[-1])))
        return event, selected_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        annotation = self._load_annotation(row["annotation_ref"])
        prev_ref = row.get("prev_annotation_ref")
        prev_annotation = self._load_annotation(prev_ref) if prev_ref else annotation

        frame_path = self._resolve_path(row["frame_path"])
        frame_sensor = np.asarray(Image.open(frame_path).convert("L"))
        sensor_size_wh = tuple(int(v) for v in row.get("sensor_size_wh", [frame_sensor.shape[1], frame_sensor.shape[0]]))
        roi_xywh = tuple(_resolve_xywh(row, "roi_xywh", default=[0, 0, sensor_size_wh[0], sensor_size_wh[1]]))
        transform = build_transform(
            resize_policy=self._resolve_resize_policy(row),
            source_shape_hw=(sensor_size_wh[1], sensor_size_wh[0]),
            roi_xywh=roi_xywh,
            target_size_wh=self.input_size,
        )

        mask_sensor = self._load_mask(annotation, sensor_size_wh=sensor_size_wh, roi_xywh=roi_xywh)
        frame = apply_transform_to_frame(frame_sensor, transform).astype(np.float32) / 255.0
        mask = apply_transform_to_mask(mask_sensor, transform)
        mask = (mask > 127).astype(np.float32)

        end_timestamp_us = int((row.get("event_window") or {}).get("end_timestamp_us", annotation.get("timestamp_us", 0)))
        effective_event_window = self._resolve_event_window(row, end_timestamp_us=end_timestamp_us)
        event, selected_count = self._build_event_input(row, transform, sensor_size_wh=sensor_size_wh, end_timestamp_us=end_timestamp_us)

        cur_state_sensor = _normalize_uv(_state6_from_annotation(annotation))
        prev_state_sensor = _normalize_uv(_state6_from_annotation(prev_annotation))
        cur_state_xywht = transform.ellipse(_xywht_from_state6(cur_state_sensor).tolist())
        prev_state_xywht = transform.ellipse(_xywht_from_state6(prev_state_sensor).tolist())
        cur_state = _normalize_uv(np.asarray(xywht_to_xyabuv(np.asarray(cur_state_xywht, dtype=np.float32)), dtype=np.float32))
        prev_state = _normalize_uv(np.asarray(xywht_to_xyabuv(np.asarray(prev_state_xywht, dtype=np.float32)), dtype=np.float32))

        pupil_region_bbox = annotation.get("pupil_region_bbox_xywh_sensor")
        if pupil_region_bbox is None:
            pupil_ellipse_sensor = (
                annotation.get("pupil_ellipse_xywht_sensor")
                or annotation.get("ellipse_sensor_xywht")
                or annotation.get("ellipse_xywht")
                or _xywht_from_state6(cur_state_sensor).tolist()
            )
            pupil_region_bbox = _ellipse_to_bbox([float(v) for v in pupil_ellipse_sensor])
        pupil_region_target = transform.bbox(pupil_region_bbox)

        eye_region_bbox = (
            annotation.get("eye_region_bbox_xywh_sensor")
            or annotation.get("eye_region_xywh")
            or row.get("roi_xywh")
            or [0, 0, sensor_size_wh[0], sensor_size_wh[1]]
        )
        eye_target_box = transform.bbox(eye_region_bbox)

        annotation_quality = _safe_float(row.get("annotation_quality", annotation.get("annotation_quality", 1.0)), 1.0)
        closed_eye_flag = float(_safe_bool(row.get("closed_eye_flag", annotation.get("closed_eye_flag", False))))
        mask_valid = float(_safe_bool(annotation.get("mask_valid", row.get("mask_valid", True)), True))
        valid_track = float(_safe_bool(row.get("valid_track", True), True))
        similarity_target = _safe_float(row.get("similarity_target", annotation_quality), annotation_quality)
        confidence_target = 1.0 if (mask_valid > 0.5 and closed_eye_flag < 0.5) else 0.0
        track_delta = np.asarray(legacy_track_delta_to_uv(prev_state, cur_state), dtype=np.float32)
        track_conf = 1.0 if valid_track > 0.5 else 0.0
        pupil_track_target = np.concatenate([track_delta, np.asarray([track_conf, annotation_quality], dtype=np.float32)], axis=0)
        event_density = float(selected_count) / max(1.0, float(roi_xywh[2] * roi_xywh[3]))
        aux_target = int(row.get("aux_target", 0))

        return {
            "sample_id": str(row["sample_id"]),
            "frame": _to_tensor_image(frame),
            "event": torch.from_numpy(event.astype(np.float32)),
            "mask_target": _to_tensor_image(mask.astype(np.float32)),
            "eye_target": torch.tensor([eye_target_box[0], eye_target_box[1], eye_target_box[2], eye_target_box[3], 1.0], dtype=torch.float32),
            "prev_state": torch.from_numpy(prev_state.astype(np.float32)),
            "cur_state": torch.from_numpy(cur_state.astype(np.float32)),
            "pupil_search_target": torch.tensor([cur_state[0], cur_state[1], cur_state[2], cur_state[3], cur_state[4], cur_state[5], confidence_target], dtype=torch.float32),
            "pupil_track_target": torch.from_numpy(pupil_track_target.astype(np.float32)),
            "constraint_center": torch.tensor([eye_target_box[0], eye_target_box[1]], dtype=torch.float32),
            "annotation_quality": torch.tensor(annotation_quality, dtype=torch.float32),
            "similarity_target": torch.tensor(similarity_target, dtype=torch.float32),
            "event_density": torch.tensor(event_density, dtype=torch.float32),
            "closed_eye_flag": torch.tensor(closed_eye_flag, dtype=torch.float32),
            "mask_valid": torch.tensor(mask_valid, dtype=torch.float32),
            "valid_track": torch.tensor(valid_track, dtype=torch.float32),
            "aux_target": torch.tensor(aux_target, dtype=torch.long),
            "meta": {
                "frame_path": str(frame_path),
                "session_key": row.get("session_key"),
                "resize_policy": transform.policy,
                "manifest_resize_policy": row.get("resize_policy"),
                "annotation_source": annotation.get("annotation_source", row.get("annotation_source", "manual")),
                "transform": transform.as_dict(),
                "selected_event_count": selected_count,
                "event_window": effective_event_window,
                "manifest_event_window": row.get("event_window") or {},
                "pupil_region_target_xywh": pupil_region_target,
            },
        }
