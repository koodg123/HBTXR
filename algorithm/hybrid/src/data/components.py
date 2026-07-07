from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.utils.state6 import legacy_track_delta_to_uv, xywht_to_xyabuv

from .contracts import BuiltEventInput, LoadedSampleAssets, ResolvedRoi, TargetBundle
from .transform import SpatialTransform
from .utils import (
    clip_xywh_to_sensor,
    constrain_xywh_to_parent,
    ellipse_to_bbox,
    normalize_uv,
    resolve_xywh,
    safe_bool,
    safe_float,
    safe_text,
    state6_from_annotation,
    xywht_from_state6,
    to_tensor_image,
)


class AdaptiveRoiResolver:
    def __init__(self, *, event_builder: Any) -> None:
        self.event_builder = event_builder

    def resolve_event_crop_policy(self) -> str:
        policy = safe_text(self.event_builder.event_builder.get("crop_policy"), "manifest_roi").lower()
        return policy if policy in {"manifest_roi", "density_adaptive", "prev_pupil_anchor"} else "manifest_roi"

    def resolve_event_crop_scope(self) -> str:
        scope = safe_text(self.event_builder.event_builder.get("crop_scope"), "within_roi").lower()
        return scope if scope in {"within_roi", "full_sensor"} else "within_roi"

    @staticmethod
    def _pupil_bbox_from_annotation(annotation: dict[str, Any]) -> tuple[float, float, float, float] | None:
        bbox = annotation.get("pupil_region_bbox_xywh_sensor")
        if bbox is not None:
            return tuple(float(v) for v in bbox)
        ellipse = (
            annotation.get("pupil_ellipse_xywht_sensor")
            or annotation.get("ellipse_sensor_xywht")
            or annotation.get("ellipse_xywht")
            or annotation.get("ellipse_frame_xywht")
        )
        if ellipse is None:
            return None
        return tuple(float(v) for v in ellipse_to_bbox([float(v) for v in ellipse]))

    def resolve_prev_pupil_anchor_roi(
        self,
        row: dict[str, Any],
        *,
        base_roi: tuple[float, float, float, float],
        crop_scope: str,
        sensor_size_wh: tuple[int, int],
    ) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
        meta: dict[str, Any] = {
            "anchor_roi_source": "prev_pupil_anchor",
            "anchor_roi_applied": False,
            "anchor_roi_xywh": [float(v) for v in base_roi],
        }
        ref = row.get("prev_annotation_ref") or row.get("annotation_ref")
        if not ref:
            meta["anchor_roi_reason"] = "missing_annotation_ref"
            return base_roi, meta
        try:
            annotation = self.event_builder.reader.load_annotation(ref)
        except (KeyError, TypeError, FileNotFoundError):
            meta["anchor_roi_reason"] = "annotation_load_failed"
            return base_roi, meta

        bbox = self._pupil_bbox_from_annotation(annotation)
        if bbox is None:
            meta["anchor_roi_reason"] = "missing_pupil_bbox"
            return base_roi, meta

        margin_px = max(0.0, safe_float(self.event_builder.event_builder.get("crop_margin_px"), 24.0))
        min_side_px = max(1.0, safe_float(self.event_builder.event_builder.get("crop_min_side_px"), 96.0))
        x, y, w, h = [float(v) for v in bbox]
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        w = max(w + 2.0 * margin_px, min_side_px)
        h = max(h + 2.0 * margin_px, min_side_px)
        anchor_roi = (cx - 0.5 * w, cy - 0.5 * h, w, h)
        anchor_roi = clip_xywh_to_sensor(anchor_roi, sensor_size_wh=sensor_size_wh)
        if crop_scope == "within_roi":
            anchor_roi = constrain_xywh_to_parent(anchor_roi, parent_xywh=base_roi)
            anchor_roi = clip_xywh_to_sensor(anchor_roi, sensor_size_wh=sensor_size_wh)

        meta["anchor_roi_applied"] = True
        meta["anchor_roi_xywh"] = [float(v) for v in anchor_roi]
        return anchor_roi, meta

    def collect_event_points_for_adaptive_crop(
        self,
        row: dict[str, Any],
        *,
        sensor_size_wh: tuple[int, int],
        end_timestamp_us: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        generation_strategy = self.event_builder.resolve_event_generation_strategy(row)
        if generation_strategy == "target_fps_session_store":
            _, _, store = self.event_builder.reader.load_session_store(row)
            event_index_range = row.get("event_index_range")
            if not isinstance(event_index_range, (list, tuple)) or len(event_index_range) != 2:
                return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
            events = {
                "x": np.asarray(store["event_x"]),
                "y": np.asarray(store["event_y"]),
            }
            return self.event_builder.extract_valid_event_points(
                events,
                selection=slice(max(0, int(event_index_range[0])), max(int(event_index_range[0]), int(event_index_range[1]))),
                sensor_size_wh=sensor_size_wh,
            )

        events_path = row.get("events_npz")
        if events_path in (None, ""):
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        events = self.event_builder.reader.load_events(events_path)
        timestamps = np.asarray(events["t"], dtype=np.int64)
        source_pair_timestamps = self.event_builder.resolve_source_pair_timestamps(row) if generation_strategy == "source_pair_average" else None

        if generation_strategy == "source_pair_average" and source_pair_timestamps is not None:
            prev_timestamp_us, next_timestamp_us = source_pair_timestamps
            if self.event_builder.resolve_source_pair_scope() == "pair_local":
                pair_start_us, pair_end_us = sorted((int(prev_timestamp_us), int(next_timestamp_us)))
                selection = self.event_builder.resolve_timestamp_selection(
                    timestamps,
                    start_timestamp_us=pair_start_us,
                    end_timestamp_us=pair_end_us,
                    include_start=True,
                )
                return self.event_builder.extract_valid_event_points(events, selection=selection, sensor_size_wh=sensor_size_wh)

            prev_window = self.event_builder.resolve_anchor_event_window(prev_timestamp_us)
            next_window = self.event_builder.resolve_anchor_event_window(next_timestamp_us)
            prev_selection = self.event_builder.event_indices_for_window(
                timestamps,
                end_timestamp_us=int(prev_timestamp_us),
                policy=str(prev_window.get("policy", self.event_builder.default_event_builder["policy"])),
                time_bin_us=int(prev_window.get("time_bin_us", self.event_builder.default_event_builder["time_bin_us"])),
                event_count_target=int(prev_window.get("event_count_target", self.event_builder.default_event_builder["event_count_target"])),
                start_timestamp_us=prev_window.get("start_timestamp_us"),
            )
            next_selection = self.event_builder.event_indices_for_window(
                timestamps,
                end_timestamp_us=int(next_timestamp_us),
                policy=str(next_window.get("policy", self.event_builder.default_event_builder["policy"])),
                time_bin_us=int(next_window.get("time_bin_us", self.event_builder.default_event_builder["time_bin_us"])),
                event_count_target=int(next_window.get("event_count_target", self.event_builder.default_event_builder["event_count_target"])),
                start_timestamp_us=next_window.get("start_timestamp_us"),
            )
            prev_x, prev_y = self.event_builder.extract_valid_event_points(events, selection=prev_selection, sensor_size_wh=sensor_size_wh)
            next_x, next_y = self.event_builder.extract_valid_event_points(events, selection=next_selection, sensor_size_wh=sensor_size_wh)
            return np.concatenate([prev_x, next_x]), np.concatenate([prev_y, next_y])

        effective_event_window = self.event_builder.resolve_event_window(row, end_timestamp_us=end_timestamp_us)
        selection = self.event_builder.event_indices_for_window(
            timestamps,
            end_timestamp_us=int(effective_event_window.get("end_timestamp_us", end_timestamp_us)),
            policy=str(effective_event_window.get("policy", self.event_builder.default_event_builder["policy"])),
            time_bin_us=int(effective_event_window.get("time_bin_us", self.event_builder.default_event_builder["time_bin_us"])),
            event_count_target=int(effective_event_window.get("event_count_target", self.event_builder.default_event_builder["event_count_target"])),
            start_timestamp_us=effective_event_window.get("start_timestamp_us"),
        )
        return self.event_builder.extract_valid_event_points(events, selection=selection, sensor_size_wh=sensor_size_wh)

    def resolve_effective_roi(
        self,
        row: dict[str, Any],
        *,
        sensor_size_wh: tuple[int, int],
        end_timestamp_us: int,
    ) -> ResolvedRoi:
        base_roi = tuple(resolve_xywh(row, "roi_xywh", default=[0, 0, sensor_size_wh[0], sensor_size_wh[1]]))
        base_roi = clip_xywh_to_sensor(base_roi, sensor_size_wh=sensor_size_wh)
        crop_policy = self.resolve_event_crop_policy()
        crop_scope = self.resolve_event_crop_scope()
        meta = {
            "event_crop_policy": crop_policy,
            "event_crop_scope": crop_scope,
            "base_roi_xywh": [float(v) for v in base_roi],
            "adaptive_roi_applied": False,
            "adaptive_roi_xywh": [float(v) for v in base_roi],
            "adaptive_roi_event_count": 0,
        }
        if crop_policy == "prev_pupil_anchor":
            anchor_roi, anchor_meta = self.resolve_prev_pupil_anchor_roi(
                row,
                base_roi=base_roi,
                crop_scope=crop_scope,
                sensor_size_wh=sensor_size_wh,
            )
            meta.update(anchor_meta)
            meta["adaptive_roi_applied"] = bool(anchor_meta.get("anchor_roi_applied", False))
            meta["adaptive_roi_xywh"] = [float(v) for v in anchor_roi]
            return ResolvedRoi(roi_xywh=anchor_roi, meta=meta)

        if crop_policy != "density_adaptive":
            return ResolvedRoi(roi_xywh=base_roi, meta=meta)

        xs, ys = self.collect_event_points_for_adaptive_crop(
            row,
            sensor_size_wh=sensor_size_wh,
            end_timestamp_us=end_timestamp_us,
        )
        if crop_scope == "within_roi":
            bx, by, bw, bh = base_roi
            inside = (xs >= bx) & (xs < bx + bw) & (ys >= by) & (ys < by + bh)
            xs = xs[inside]
            ys = ys[inside]

        meta["adaptive_roi_event_count"] = int(len(xs))
        min_events = max(1, int(safe_float(self.event_builder.event_builder.get("crop_min_events"), 64)))
        if len(xs) < min_events:
            meta["adaptive_roi_reason"] = "insufficient_events"
            return ResolvedRoi(roi_xywh=base_roi, meta=meta)

        low = float(np.clip(safe_float(self.event_builder.event_builder.get("crop_quantile_low"), 0.05), 0.0, 0.49))
        high = float(np.clip(safe_float(self.event_builder.event_builder.get("crop_quantile_high"), 0.95), 0.51, 1.0))
        if high <= low:
            low, high = 0.05, 0.95
        margin_px = max(0.0, safe_float(self.event_builder.event_builder.get("crop_margin_px"), 12.0))
        min_side_px = max(1.0, safe_float(self.event_builder.event_builder.get("crop_min_side_px"), 96.0))

        x0 = float(np.quantile(xs.astype(np.float32), low)) - margin_px
        x1 = float(np.quantile(xs.astype(np.float32), high)) + margin_px
        y0 = float(np.quantile(ys.astype(np.float32), low)) - margin_px
        y1 = float(np.quantile(ys.astype(np.float32), high)) + margin_px
        w = max(1.0, x1 - x0)
        h = max(1.0, y1 - y0)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        w = max(w, min_side_px)
        h = max(h, min_side_px)
        adaptive_roi = (cx - 0.5 * w, cy - 0.5 * h, w, h)
        adaptive_roi = clip_xywh_to_sensor(adaptive_roi, sensor_size_wh=sensor_size_wh)
        if crop_scope == "within_roi":
            adaptive_roi = constrain_xywh_to_parent(adaptive_roi, parent_xywh=base_roi)
            adaptive_roi = clip_xywh_to_sensor(adaptive_roi, sensor_size_wh=sensor_size_wh)

        meta["adaptive_roi_applied"] = True
        meta["adaptive_roi_xywh"] = [float(v) for v in adaptive_roi]
        return ResolvedRoi(roi_xywh=adaptive_roi, meta=meta)

    def resolve_effective_roi_xywh(
        self,
        row: dict[str, Any],
        *,
        sensor_size_wh: tuple[int, int],
        end_timestamp_us: int,
    ) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
        resolved = self.resolve_effective_roi(
            row,
            sensor_size_wh=sensor_size_wh,
            end_timestamp_us=end_timestamp_us,
        )
        return resolved.roi_xywh, resolved.meta


class AnnotationTargetBuilder:
    def build(
        self,
        row: dict[str, Any],
        *,
        assets: LoadedSampleAssets,
        transform: SpatialTransform,
    ) -> TargetBundle:
        cur_state_sensor = normalize_uv(state6_from_annotation(assets.annotation))
        prev_state_sensor = normalize_uv(state6_from_annotation(assets.prev_annotation))
        cur_state_xywht = transform.ellipse(xywht_from_state6(cur_state_sensor).tolist())
        prev_state_xywht = transform.ellipse(xywht_from_state6(prev_state_sensor).tolist())
        cur_state = normalize_uv(np.asarray(xywht_to_xyabuv(np.asarray(cur_state_xywht, dtype=np.float32)), dtype=np.float32))
        prev_state = normalize_uv(np.asarray(xywht_to_xyabuv(np.asarray(prev_state_xywht, dtype=np.float32)), dtype=np.float32))

        pupil_region_bbox = assets.annotation.get("pupil_region_bbox_xywh_sensor")
        if pupil_region_bbox is None:
            pupil_ellipse_sensor = (
                assets.annotation.get("pupil_ellipse_xywht_sensor")
                or assets.annotation.get("ellipse_sensor_xywht")
                or assets.annotation.get("ellipse_xywht")
                or xywht_from_state6(cur_state_sensor).tolist()
            )
            pupil_region_bbox = ellipse_to_bbox([float(v) for v in pupil_ellipse_sensor])
        pupil_region_target = transform.bbox(pupil_region_bbox)

        eye_region_bbox = (
            assets.annotation.get("eye_region_bbox_xywh_sensor")
            or assets.annotation.get("eye_region_xywh")
            or row.get("roi_xywh")
            or [0, 0, assets.sensor_size_wh[0], assets.sensor_size_wh[1]]
        )
        eye_target_box = transform.bbox(eye_region_bbox)

        annotation_quality = safe_float(row.get("annotation_quality", assets.annotation.get("annotation_quality", 1.0)), 1.0)
        closed_eye_flag = float(safe_bool(row.get("closed_eye_flag", assets.annotation.get("closed_eye_flag", False))))
        mask_valid = float(safe_bool(assets.annotation.get("mask_valid", row.get("mask_valid", True)), True))
        valid_track = float(safe_bool(row.get("valid_track", True), True))
        similarity_target = safe_float(row.get("similarity_target", annotation_quality), annotation_quality)
        confidence_target = 1.0 if (mask_valid > 0.5 and closed_eye_flag < 0.5) else 0.0
        track_delta = np.asarray(legacy_track_delta_to_uv(prev_state, cur_state), dtype=np.float32)
        track_conf = 1.0 if valid_track > 0.5 else 0.0
        pupil_track_target = np.concatenate([track_delta, np.asarray([track_conf, annotation_quality], dtype=np.float32)], axis=0)
        aux_target = int(row.get("aux_target", 0))
        sample_timestamp_us = int(row.get("sample_timestamp_us", assets.annotation.get("timestamp_us", assets.end_timestamp_us)))
        return TargetBundle(
            cur_state=cur_state.astype(np.float32),
            prev_state=prev_state.astype(np.float32),
            pupil_region_target=pupil_region_target,
            eye_target_box=eye_target_box,
            annotation_quality=annotation_quality,
            similarity_target=similarity_target,
            closed_eye_flag=closed_eye_flag,
            mask_valid=mask_valid,
            valid_track=valid_track,
            confidence_target=confidence_target,
            pupil_track_target=pupil_track_target.astype(np.float32),
            aux_target=aux_target,
            sample_timestamp_us=sample_timestamp_us,
        )


class SampleAssembler:
    def __init__(
        self,
        *,
        data_mode: str,
        canonical_name: str,
        manifest_name: str,
        frame_source: str,
    ) -> None:
        self.data_mode = safe_text(data_mode, "mode1")
        self.canonical_name = safe_text(canonical_name, "canonical1")
        self.manifest_name = safe_text(manifest_name, "manifest1")
        self.frame_source = safe_text(frame_source, "original")

    def assemble(
        self,
        *,
        row: dict[str, Any],
        assets: LoadedSampleAssets,
        transform: SpatialTransform,
        frame: np.ndarray,
        mask: np.ndarray,
        event_input: BuiltEventInput,
        resolved_roi: ResolvedRoi,
        targets: TargetBundle,
    ) -> dict[str, Any]:
        data_mode = safe_text(row.get("data_mode"), self.data_mode)
        canonical_name = safe_text(row.get("canonical_name"), self.canonical_name)
        manifest_name = safe_text(row.get("manifest_name"), self.manifest_name)
        frame_source = safe_text(row.get("frame_source"), self.frame_source)
        event_density = float(event_input.selected_count) / max(1.0, float(resolved_roi.roi_xywh[2] * resolved_roi.roi_xywh[3]))

        meta = {
            "data_mode": data_mode,
            "canonical_name": canonical_name,
            "manifest_name": manifest_name,
            "frame_source": frame_source,
            "session_key": row.get("session_key"),
            "sample_timestamp_us": targets.sample_timestamp_us,
            "frame_path": None if assets.resolved_frame_path is None else str(assets.resolved_frame_path),
            "session_store_path": None if assets.resolved_store_path is None else str(assets.resolved_store_path),
            "frame_load_source": "session_store" if assets.resolved_frame_path is None else "frame_path",
            "session_store_layout": row.get("session_store_layout"),
            "resize_policy": transform.policy,
            "manifest_resize_policy": row.get("resize_policy"),
            "annotation_source": assets.annotation.get("annotation_source", row.get("annotation_source", "manual")),
            "transform": transform.as_dict(),
            "selected_event_count": event_input.selected_count,
            "event_window": event_input.meta["effective_event_window"],
            "effective_event_window": event_input.meta["effective_event_window"],
            "manifest_event_window": row.get("event_window") or {},
            "event_generation_strategy": event_input.meta["event_generation_strategy"],
            "event_generation_source_pair_scope": event_input.meta["source_pair_scope"],
            "event_average_weighting": event_input.meta["average_weighting"],
            "event_generation_source_timestamps": event_input.meta["source_pair_timestamps"],
            "pupil_region_target_xywh": targets.pupil_region_target,
            **resolved_roi.meta,
            "effective_roi_xywh": [float(v) for v in resolved_roi.roi_xywh],
        }
        if data_mode == "mode2" or row.get("synthetic_frame_flag") is not None:
            meta.update(
                {
                    "synthetic_frame_flag": bool(row.get("synthetic_frame_flag", False)),
                    "interp_source_pair": row.get("interp_source_pair"),
                    "interp_alpha": row.get("interp_alpha"),
                    "interp_model": row.get("interp_model"),
                    "interp_quality": row.get("interp_quality"),
                    "interp_rank": row.get("interp_rank"),
                    "interp_insert_count": row.get("interp_insert_count"),
                    "interp_target_fps": row.get("interp_target_fps"),
                    "interp_count_policy": row.get("interp_count_policy"),
                    "synthetic_overlap_policy": row.get("synthetic_overlap_policy"),
                    "interpolation_ref": row.get("interpolation_ref"),
                }
            )

        return {
            "sample_id": str(row["sample_id"]),
            "frame": to_tensor_image(frame),
            "event": torch.from_numpy(event_input.event.astype(np.float32)),
            "mask_target": to_tensor_image(mask.astype(np.float32)),
            "eye_target": torch.tensor([targets.eye_target_box[0], targets.eye_target_box[1], targets.eye_target_box[2], targets.eye_target_box[3], 1.0], dtype=torch.float32),
            "pupil_region_target": torch.tensor(
                [targets.pupil_region_target[0], targets.pupil_region_target[1], targets.pupil_region_target[2], targets.pupil_region_target[3], 1.0],
                dtype=torch.float32,
            ),
            "prev_state": torch.from_numpy(targets.prev_state.astype(np.float32)),
            "cur_state": torch.from_numpy(targets.cur_state.astype(np.float32)),
            "pupil_search_target": torch.tensor(
                [
                    targets.cur_state[0],
                    targets.cur_state[1],
                    targets.cur_state[2],
                    targets.cur_state[3],
                    targets.cur_state[4],
                    targets.cur_state[5],
                    targets.confidence_target,
                ],
                dtype=torch.float32,
            ),
            "pupil_track_target": torch.from_numpy(targets.pupil_track_target.astype(np.float32)),
            "constraint_center": torch.tensor([targets.eye_target_box[0], targets.eye_target_box[1]], dtype=torch.float32),
            "annotation_quality": torch.tensor(targets.annotation_quality, dtype=torch.float32),
            "similarity_target": torch.tensor(targets.similarity_target, dtype=torch.float32),
            "event_density": torch.tensor(event_density, dtype=torch.float32),
            "closed_eye_flag": torch.tensor(targets.closed_eye_flag, dtype=torch.float32),
            "mask_valid": torch.tensor(targets.mask_valid, dtype=torch.float32),
            "valid_track": torch.tensor(targets.valid_track, dtype=torch.float32),
            "aux_target": torch.tensor(targets.aux_target, dtype=torch.long),
            "meta": meta,
        }
