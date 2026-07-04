from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.preprocess.interpolation import interpolate_linear_blend
from src.utils.io import read_jsonl
from src.utils.paths import resolve_stored_path

from .contracts import LoadedSampleAssets
from .utils import (
    ellipse_to_bbox,
    import_h5py,
    rasterize_bbox_mask,
    rasterize_ellipse_mask,
    safe_float,
    safe_text,
)


class SessionFrameEventReader:
    def __init__(
        self,
        *,
        manifest_path: Path,
        canonical_root: Path | None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.canonical_root = canonical_root
        self._annotation_store_cache: dict[Path, dict[str, dict[str, Any]]] = {}
        self._events_cache: dict[Path, dict[str, np.ndarray]] = {}
        self._session_store_cache: dict[Path, dict[str, np.ndarray]] = {}

    def resolve_path(self, raw_path: str | Path) -> Path:
        if self.canonical_root is not None:
            return resolve_stored_path(self.canonical_root, raw_path)
        return resolve_stored_path(self.manifest_path.parent, raw_path)

    def load_annotation(self, ref: dict[str, Any]) -> dict[str, Any]:
        store_path = self.resolve_path(ref["annotation_store_path"])
        cache = self._annotation_store_cache.get(store_path)
        if cache is None:
            rows = read_jsonl(store_path)
            cache = {str(row["ann_id"]): row for row in rows}
            self._annotation_store_cache[store_path] = cache
        ann_id = str(ref["ann_id"])
        if ann_id not in cache:
            raise KeyError(f"Annotation id not found: {ann_id} @ {store_path}")
        return cache[ann_id]

    def load_events(self, events_path: str | Path) -> dict[str, np.ndarray]:
        resolved = self.resolve_path(events_path)
        cached = self._events_cache.get(resolved)
        if cached is None:
            data = np.load(resolved)
            cached = {key: np.asarray(data[key]) for key in data.files}
            self._events_cache[resolved] = cached
        return cached

    def resolve_session_store_ref(self, row: dict[str, Any]) -> tuple[Path, str] | None:
        raw_path = row.get("session_store_path") or row.get("session_h5_path") or row.get("session_npz_path")
        if raw_path in (None, ""):
            return None
        resolved = self.resolve_path(raw_path)
        store_format = safe_text(row.get("session_store_format"), "")
        if not store_format:
            suffix = resolved.suffix.lower()
            store_format = "h5" if suffix == ".h5" else "npz"
        return resolved, store_format

    def load_session_store(self, row: dict[str, Any]) -> tuple[Path, str, dict[str, np.ndarray]]:
        resolved_ref = self.resolve_session_store_ref(row)
        if resolved_ref is None:
            raise KeyError("Manifest row is missing session-store references")
        resolved, store_format = resolved_ref
        cached = self._session_store_cache.get(resolved)
        if cached is None:
            if store_format == "npz":
                data = np.load(resolved)
                cached = {key: np.asarray(data[key]) for key in data.files}
            elif store_format == "h5":
                h5py = import_h5py(required=True)
                with h5py.File(resolved, "r") as handle:
                    cached = {
                        "frame_timestamps_us": np.asarray(handle["frames/timestamps_us"]),
                        "frame_event_ranges": np.asarray(handle["frames/event_ranges"]),
                        "event_t": np.asarray(handle["events/t"]),
                        "event_x": np.asarray(handle["events/x"]),
                        "event_y": np.asarray(handle["events/y"]),
                        "event_p": np.asarray(handle["events/p"]),
                    }
                    if "images" in handle["frames"]:
                        cached["frame_images"] = np.asarray(handle["frames/images"])
                    if "source_frames" in handle:
                        source_frames = handle["source_frames"]
                        if "images" in source_frames:
                            cached["source_frame_images"] = np.asarray(source_frames["images"])
                        if "timestamps_us" in source_frames:
                            cached["source_frame_timestamps_us"] = np.asarray(source_frames["timestamps_us"])
            else:
                raise ValueError(f"Unsupported session store format: {store_format}")
            self._session_store_cache[resolved] = cached
        return resolved, store_format, cached

    def load_frame_sensor_from_lazy_session_store(
        self,
        row: dict[str, Any],
        *,
        store: dict[str, np.ndarray],
        resolved_store_path: Path,
    ) -> np.ndarray:
        source_frames = np.asarray(store.get("source_frame_images")) if store.get("source_frame_images") is not None else None
        if source_frames is None or len(source_frames) == 0:
            raise KeyError(f"Lazy session store {resolved_store_path} is missing source_frame_images")

        source_kind = safe_text(row.get("source_kind"), "raw")
        matched_raw_frame_index = row.get("matched_raw_frame_index")
        if source_kind == "raw" and matched_raw_frame_index not in (None, ""):
            frame_index = int(matched_raw_frame_index)
            if frame_index < 0 or frame_index >= len(source_frames):
                raise IndexError(f"Matched raw frame index {frame_index} out of range for {resolved_store_path}")
            return np.asarray(source_frames[frame_index], dtype=np.uint8)

        pair = row.get("source_pair_index") or row.get("interp_source_pair_index")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise KeyError(f"Lazy synthetic frame row is missing source_pair_index: {row.get('sample_id')}")
        left_idx = int(pair[0])
        right_idx = int(pair[1])
        if left_idx < 0 or right_idx < 0 or left_idx >= len(source_frames) or right_idx >= len(source_frames):
            raise IndexError(f"Source pair index {pair} out of range for {resolved_store_path}")

        backend = safe_text(row.get("interp_model"), "linear_blend")
        if backend != "linear_blend":
            raise NotImplementedError(
                f"Lazy session-store frame synthesis currently supports only linear_blend, got {backend!r}"
            )
        alpha = float(np.clip(safe_float(row.get("interp_alpha", 0.0), 0.0), 0.0, 1.0))
        return interpolate_linear_blend(
            np.asarray(source_frames[left_idx], dtype=np.uint8),
            np.asarray(source_frames[right_idx], dtype=np.uint8),
            alpha=alpha,
        )

    def load_frame_sensor(self, row: dict[str, Any]) -> tuple[np.ndarray, Path | None, Path | None]:
        frame_path_value = row.get("frame_path")
        if frame_path_value not in (None, ""):
            frame_path = self.resolve_path(frame_path_value)
            return np.asarray(Image.open(frame_path).convert("L")), frame_path, None

        resolved_store_path, _, store = self.load_session_store(row)
        session_store_layout = safe_text(
            row.get("session_store_layout"),
            "materialized_target_frames" if store.get("frame_images") is not None else "lazy_source_frames",
        )
        frames = store.get("frame_images")
        if session_store_layout == "lazy_source_frames" or frames is None:
            frame = self.load_frame_sensor_from_lazy_session_store(
                row,
                store=store,
                resolved_store_path=resolved_store_path,
            )
            return np.asarray(frame, dtype=np.uint8), None, resolved_store_path

        frame_index = int(row.get("frame_index", row.get("frame_store_index", 0)))
        frames_np = np.asarray(frames)
        if frame_index < 0 or frame_index >= len(frames_np):
            raise IndexError(f"Frame index {frame_index} out of range for session store {resolved_store_path}")
        return np.asarray(frames_np[frame_index], dtype=np.uint8), None, resolved_store_path

    def load_mask(
        self,
        annotation: dict[str, Any],
        *,
        sensor_size_wh: tuple[int, int],
        roi_xywh: tuple[float, float, float, float],
    ) -> np.ndarray:
        mask_path = annotation.get("pupil_mask_path") or annotation.get("mask_path")
        if mask_path:
            mask = np.asarray(Image.open(self.resolve_path(mask_path)).convert("L"))
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

        ellipse = (
            annotation.get("pupil_ellipse_xywht_sensor")
            or annotation.get("ellipse_sensor_xywht")
            or annotation.get("ellipse_xywht")
            or annotation.get("ellipse_frame_xywht")
        )
        if ellipse is not None:
            return rasterize_ellipse_mask(tuple(float(v) for v in ellipse), sensor_size_wh=sensor_size_wh)
        bbox = annotation.get("pupil_region_bbox_xywh_sensor")
        if bbox is not None:
            return rasterize_bbox_mask(tuple(float(v) for v in bbox), sensor_size_wh=sensor_size_wh)
        return np.zeros((sensor_size_wh[1], sensor_size_wh[0]), dtype=np.uint8)

    def read_sample_assets(self, row: dict[str, Any]) -> LoadedSampleAssets:
        annotation = self.load_annotation(row["annotation_ref"])
        prev_ref = row.get("prev_annotation_ref")
        prev_annotation = self.load_annotation(prev_ref) if prev_ref else annotation
        frame_sensor, resolved_frame_path, resolved_store_path = self.load_frame_sensor(row)
        sensor_size_wh = tuple(int(v) for v in row.get("sensor_size_wh", [frame_sensor.shape[1], frame_sensor.shape[0]]))
        end_timestamp_us = int((row.get("event_window") or {}).get("end_timestamp_us", annotation.get("timestamp_us", 0)))
        return LoadedSampleAssets(
            annotation=annotation,
            prev_annotation=prev_annotation,
            frame_sensor=frame_sensor,
            resolved_frame_path=resolved_frame_path,
            resolved_store_path=resolved_store_path,
            sensor_size_wh=sensor_size_wh,
            end_timestamp_us=end_timestamp_us,
        )
