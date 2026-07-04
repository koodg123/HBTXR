from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from src.preprocess.annotation_backends import (
    DEFAULT_ANNOTATION_BACKEND,
    SUPPORTED_ANNOTATION_BACKENDS,
    build_annotation_runtime,
)
from src.preprocess.annotation_groundedsam import (
    groundedsam_annotation_store_path,
    load_groundedsam_annotation_store,
    resolve_groundedsam_mask_path,
)
from src.preprocess.canonicalize_pipeline import (
    CanonicalSessionArtifactWriter,
    CanonicalizeDatasetRunner,
    CanonicalizeResolvedRequest,
    CanonicalSessionProcessor,
)
from src.preprocess.event_generation import (
    DEFAULT_EVENT_GENERATION_BACKEND,
    SUPPORTED_EVENT_GENERATION_BACKENDS,
)
from src.preprocess.interpolation import (
    build_alpha_schedule,
    compute_insert_count,
    interpolate_pair,
    resolve_interpolation_backend,
    synth_timestamp_us,
)
from src.preprocess.io_utils import (
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
    canonical_user_name,
    collect_frame_records,
    crop_mask,
    derive_eye_region,
    discover_session_layout,
    ellipse_sensor_to_roi,
    ensure_dir,
    FrameRecord,
    is_official_session_code,
    load_events_from_txt,
    maybe_link_or_copy,
    parse_via_csv_with_report,
    rasterize_ellipse_mask,
    save_events_npz,
    write_json,
    write_jsonl,
)
from src.preprocess.path_utils import add_common_path_args, relativize_to, resolve_paths
from src.preprocess.raw_ellipse_blink import (
    apply_groundedsam_store_blink_metadata,
    build_raw_ellipse_blink_metadata,
)
from src.preprocess.protocol import derive_protocol_session_index, derive_session_motion_regime_prior
from src.preprocess.session_package import build_session_package
from src.utils.paths import resolve_canonical_dataset_root, resolve_canonical_indexes_root
from src.utils.state6 import xywht_to_xyabuv

DEFAULT_FRAME_SIZE = (256, 160)
DEFAULT_EVENT_SIZE = (128, 80)


def _mode_defaults(data_mode: str) -> dict[str, str]:
    mode = str(data_mode).strip().lower()
    if mode == "mode0":
        return {
            "data_mode": "mode0",
            "canonical_name": "canonical0",
            "frame_source": "original",
            "annotation_mode": "manual_csv",
        }
    if mode == "mode2":
        return {
            "data_mode": "mode2",
            "canonical_name": "canonical2",
            "frame_source": "interpolated",
            "annotation_mode": "auto",
        }
    return {
        "data_mode": "mode1",
        "canonical_name": "canonical1",
        "frame_source": "original",
        "annotation_mode": "auto",
    }


def _annotation_sort_key(row: Dict) -> tuple[int, int, str]:
    frame_idx = row.get("frame_idx")
    normalized_idx = int(frame_idx) if frame_idx is not None else -1
    return (
        normalized_idx,
        int(row.get("timestamp_us", 0)),
        str(row.get("frame_filename", row.get("ann_id", ""))),
    )


def _interpolate_theta(theta0: float, theta1: float, alpha: float) -> float:
    sin_term = (1.0 - alpha) * math.sin(2.0 * theta0) + alpha * math.sin(2.0 * theta1)
    cos_term = (1.0 - alpha) * math.cos(2.0 * theta0) + alpha * math.cos(2.0 * theta1)
    if abs(sin_term) < 1.0e-8 and abs(cos_term) < 1.0e-8:
        return float(theta0)
    return float(0.5 * math.atan2(sin_term, cos_term))


def _interpolate_ellipse(ellipse0: list[float], ellipse1: list[float], alpha: float) -> list[float]:
    t = float(np.clip(alpha, 0.0, 1.0))
    x0, y0, a0, b0, theta0 = [float(v) for v in ellipse0]
    x1, y1, a1, b1, theta1 = [float(v) for v in ellipse1]
    return [
        float((1.0 - t) * x0 + t * x1),
        float((1.0 - t) * y0 + t * y1),
        float((1.0 - t) * a0 + t * a1),
        float((1.0 - t) * b0 + t * b1),
        _interpolate_theta(theta0, theta1, t),
    ]


def _interpolate_frame_quality(
    frame0: np.ndarray,
    frame1: np.ndarray,
    quality0: float,
    quality1: float,
) -> float:
    frame0_f = np.asarray(frame0, dtype=np.float32)
    frame1_f = np.asarray(frame1, dtype=np.float32)
    delta = float(np.mean(np.abs(frame1_f - frame0_f)) / 255.0)
    source_quality = 0.5 * (float(quality0) + float(quality1))
    return float(np.clip(source_quality * (1.0 - delta), 0.0, 1.0))


def _mode2_source_tag(annotation_source_kind: str) -> str:
    source = str(annotation_source_kind).strip().lower()
    if source == "groundedsam":
        return "groundedsam"
    if source == "manual_csv":
        return "manual"
    return source or "unknown"


def _build_mode2_interpolated_rows(
    *,
    label_rows: List[Dict],
    layout_frames_dir: Path,
    session_key: str,
    user_id: int,
    eye: str,
    session_code: str,
    canonical_root: Path,
    session_dir: Path,
    events_npz_path: Path | None,
    export_masks: bool,
    eye_region,
    protocol_session_index: int | None,
    session_motion_regime_prior: Dict,
    data_mode: str,
    canonical_name: str,
    frame_source: str,
    interpolation_alpha: float,
    interpolation_model: str,
    interpolation_backend: str | None,
    interpolation_target_fps: float | None,
    interpolation_fixed_insert: int | None,
    interpolation_count_policy: str,
    interpolation_max_insert: int | None,
    interpolation_timelens_root: Path | None,
    interpolation_timelens_checkpoint: Path | None,
    interpolation_timelens_device: str,
    interpolation_timelens_xl_root: Path | None,
    interpolation_timelens_xl_checkpoint: Path | None,
    interpolation_timelens_xl_device: str,
    synthetic_overlap_policy: str,
    annotation_source_kind: str,
) -> tuple[List[FrameRecord], List[Dict], Dict | None]:
    ordered_rows = sorted(label_rows, key=_annotation_sort_key)
    if len(ordered_rows) < 2:
        return [], [], None

    interpolation_path = session_dir / "labels" / "interpolation_index.json"
    frame_records: List[FrameRecord] = []
    synthetic_rows: List[Dict] = []
    interpolation_rows: List[Dict] = []
    source_tag = _mode2_source_tag(annotation_source_kind)
    resolved_backend = resolve_interpolation_backend(
        interpolation_backend=interpolation_backend,
        interpolation_model=interpolation_model,
    )
    global_synth_idx = 0

    for pair_idx, (row0, row1) in enumerate(zip(ordered_rows[:-1], ordered_rows[1:])):
        src0 = layout_frames_dir / str(row0["frame_filename"])
        src1 = layout_frames_dir / str(row1["frame_filename"])
        if not src0.exists() or not src1.exists():
            continue

        ellipse0 = row0.get("ellipse_sensor_xywht") or row0.get("pupil_ellipse_xywht_sensor")
        ellipse1 = row1.get("ellipse_sensor_xywht") or row1.get("pupil_ellipse_xywht_sensor")
        if ellipse0 is None or ellipse1 is None:
            continue

        frame0 = np.asarray(Image.open(src0).convert("L"))
        frame1 = np.asarray(Image.open(src1).convert("L"))

        ts0 = int(row0.get("timestamp_us", 0))
        ts1 = int(row1.get("timestamp_us", ts0))
        insert_count = compute_insert_count(
            timestamp0_us=ts0,
            timestamp1_us=ts1,
            target_fps=interpolation_target_fps,
            fixed_insert=interpolation_fixed_insert,
            count_policy=interpolation_count_policy,
            max_insert=interpolation_max_insert,
            legacy_default_insert=1,
        )
        alpha_schedule = build_alpha_schedule(insert_count, legacy_alpha=interpolation_alpha)
        if not alpha_schedule:
            continue

        previous_timestamp_us = ts0
        for rank, alpha in enumerate(alpha_schedule, start=1):
            blended = interpolate_pair(
                backend=resolved_backend,
                frame0=frame0,
                frame1=frame1,
                alpha=alpha,
                timestamp0_us=ts0,
                timestamp1_us=ts1,
                events_path=events_npz_path,
                timelens_root=interpolation_timelens_root,
                timelens_checkpoint=interpolation_timelens_checkpoint,
                timelens_device=interpolation_timelens_device,
                timelens_xl_root=interpolation_timelens_xl_root,
                timelens_xl_checkpoint=interpolation_timelens_xl_checkpoint,
                timelens_xl_device=interpolation_timelens_xl_device,
            )
            synth_ts = synth_timestamp_us(ts0, ts1, alpha)
            synth_name = f"{global_synth_idx:06d}_{synth_ts}.png"
            dst_frame_path = session_dir / "frames" / synth_name
            Image.fromarray(blended).save(dst_frame_path)

            ellipse_sensor = _interpolate_ellipse(list(ellipse0), list(ellipse1), alpha)
            ellipse_roi = ellipse_sensor_to_roi(ellipse_sensor, eye_region)
            state6 = [float(v) for v in xywht_to_xyabuv(np.asarray(ellipse_sensor, dtype=np.float32))]
            quality = _interpolate_frame_quality(
                frame0,
                frame1,
                float(row0.get("annotation_quality", 1.0)),
                float(row1.get("annotation_quality", 1.0)),
            )
            pupil_region_bbox = [
                float(ellipse_sensor[0] - 0.5 * ellipse_sensor[2]),
                float(ellipse_sensor[1] - 0.5 * ellipse_sensor[3]),
                float(ellipse_sensor[2]),
                float(ellipse_sensor[3]),
            ]
            synthetic_event_window = {
                "source_timestamp_pair_us": [ts0, ts1],
                "start_timestamp_us": int(previous_timestamp_us),
                "end_timestamp_us": int(synth_ts),
                "synthetic_timestamp_us": int(synth_ts),
                "anchor": f"{resolved_backend}_scheduled",
                "interp_rank": int(rank),
                "interp_insert_count": int(insert_count),
            }
            interpolation_item = {
                "synthetic_frame_filename": synth_name,
                "synthetic_timestamp_us": synth_ts,
                "frame_idx": global_synth_idx,
                "pair_idx": int(pair_idx),
                "interp_rank": int(rank),
                "interp_insert_count": int(insert_count),
                "interp_source_pair": [str(row0["frame_filename"]), str(row1["frame_filename"])],
                "source_annotation_pair": [str(row0["ann_id"]), str(row1["ann_id"])],
                "source_timestamp_pair_us": [ts0, ts1],
                "interp_alpha": alpha,
                "interp_model": str(resolved_backend),
                "interp_quality": quality,
                "interp_target_fps": None if interpolation_target_fps is None else float(interpolation_target_fps),
                "interp_count_policy": str(interpolation_count_policy),
            }
            interpolation_rows.append(interpolation_item)
            frame_records.append(FrameRecord(filename=synth_name, timestamp_us=synth_ts, frame_idx=global_synth_idx))

            synthetic_row = {
                "ann_id": f"{session_key.replace('/', '__')}__interp__{global_synth_idx:06d}",
                "session_key": session_key,
                "user_id": int(user_id),
                "subject_id": int(user_id),
                "eye": str(eye),
                "session_code": str(session_code),
                "data_mode": str(data_mode),
                "canonical_name": str(canonical_name),
                "frame_source": str(frame_source),
                "sample_timestamp_us": synth_ts,
                "session_official": bool(is_official_session_code(session_code)),
                "frame_filename": synth_name,
                "frame_idx": global_synth_idx,
                "timestamp_us": synth_ts,
                "frame_path": relativize_to(canonical_root, dst_frame_path),
                "eye_region_xywh": eye_region.to_list(),
                "eye_region_bbox_xywh_sensor": eye_region.to_list(),
                "roi_size_wh": [int(eye_region.w), int(eye_region.h)],
                "frame_source_size_wh": [int(eye_region.w), int(eye_region.h)],
                "event_source_size_wh": [int(eye_region.w), int(eye_region.h)],
                "pupil_region_bbox_xywh_sensor": pupil_region_bbox,
                "pupil_ellipse_xywht_sensor": ellipse_sensor,
                "ellipse_sensor_xywht": ellipse_sensor,
                "ellipse_roi_xywht": ellipse_roi,
                "ellipse_frame_xywht": ellipse_roi,
                "ellipse_event_xywht": ellipse_roi,
                "state_xyabuv": state6,
                "sensor_size_wh": [SENSOR_WIDTH, SENSOR_HEIGHT],
                "protocol_session_index": protocol_session_index,
                "session_motion_regime_prior": dict(session_motion_regime_prior),
                "annotation_source": f"synthetic_{resolved_backend}_{source_tag}",
                "annotation_quality": quality,
                "closed_eye_flag": bool(row0.get("closed_eye_flag", False) or row1.get("closed_eye_flag", False)),
                "mask_valid": bool(row0.get("mask_valid", True) and row1.get("mask_valid", True)),
                "synthetic_frame_flag": True,
                "interp_source_pair": interpolation_item["interp_source_pair"],
                "interp_alpha": alpha,
                "interp_model": str(resolved_backend),
                "interp_quality": quality,
                "interp_rank": int(rank),
                "interp_insert_count": int(insert_count),
                "interp_target_fps": None if interpolation_target_fps is None else float(interpolation_target_fps),
                "interp_count_policy": str(interpolation_count_policy),
                "synthetic_event_window": synthetic_event_window,
                "synthetic_overlap_policy": str(synthetic_overlap_policy),
                "interpolation_ref": relativize_to(canonical_root, interpolation_path),
            }

            if export_masks:
                mask_sensor = rasterize_ellipse_mask(ellipse_sensor, image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
                mask_roi = crop_mask(mask_sensor, eye_region)
                mask_name = Path(synth_name).with_suffix(".png").name
                mask_path = session_dir / "labels" / "masks" / mask_name
                ensure_dir(mask_path.parent)
                Image.fromarray((mask_roi * 255).astype("uint8")).save(mask_path)
                synthetic_row["mask_path"] = relativize_to(canonical_root, mask_path)
                synthetic_row["pupil_mask_path"] = synthetic_row["mask_path"]
            else:
                synthetic_row["mask_path"] = None
                synthetic_row["pupil_mask_path"] = None

            synthetic_rows.append(synthetic_row)
            previous_timestamp_us = synth_ts
            global_synth_idx += 1

    if not interpolation_rows:
        return [], [], None

    interpolation_summary = {
        "enabled": True,
        "model": str(resolved_backend),
        "backend": str(resolved_backend),
        "interp_alpha": float(np.clip(interpolation_alpha, 0.0, 1.0)),
        "interp_target_fps": None if interpolation_target_fps is None else float(interpolation_target_fps),
        "interp_fixed_insert": None if interpolation_fixed_insert is None else int(interpolation_fixed_insert),
        "interp_count_policy": str(interpolation_count_policy),
        "interp_max_insert": None if interpolation_max_insert is None else int(interpolation_max_insert),
        "synthetic_overlap_policy": str(synthetic_overlap_policy),
        "n_source_annotations": len(ordered_rows),
        "n_synthetic_frames": len(interpolation_rows),
        "items": interpolation_rows,
    }
    write_json(interpolation_summary, interpolation_path)
    return frame_records, synthetic_rows, interpolation_summary


def _build_mode2_groundedsam_runtime(
    *,
    annotation_backend: str,
    groundedsam_root: Path | None,
    ultralytics_root: Path | None = None,
    groundedsam2_root: Path | None = None,
    device: str | None = None,
):
    resolved_backend = str(annotation_backend).strip().lower()
    if resolved_backend == "groundedsam" and groundedsam_root is None:
        return None
    if resolved_backend == "ultralytics_sam3" and ultralytics_root is None:
        return None
    if resolved_backend == "groundedsam2" and groundedsam2_root is None:
        return None
    return build_annotation_runtime(
        annotation_backend=resolved_backend,
        groundedsam_root=groundedsam_root,
        ultralytics_root=ultralytics_root,
        groundedsam2_root=groundedsam2_root,
        device=device,
    )


def _rerun_mode2_groundedsam_rows(
    *,
    synthetic_rows: List[Dict],
    canonical_root: Path,
    session_dir: Path,
    export_masks: bool,
    annotation_backend: str,
    groundedsam_root: Path | None,
    ultralytics_root: Path | None = None,
    groundedsam2_root: Path | None = None,
    device: str | None = None,
) -> tuple[List[Dict], Dict | None]:
    runtime = _build_mode2_groundedsam_runtime(
        annotation_backend=annotation_backend,
        groundedsam_root=groundedsam_root,
        ultralytics_root=ultralytics_root,
        groundedsam2_root=groundedsam2_root,
        device=device,
    )
    if runtime is None:
        return synthetic_rows, None

    detections: list[tuple[Dict, Dict | None]] = []
    for row in synthetic_rows:
        frame_path = session_dir / "frames" / str(row["frame_filename"])
        image_rgb = np.asarray(Image.open(frame_path).convert("RGB"))
        result = runtime.annotate_image(image_rgb)
        detections.append((row, result))

    if not any(result is not None for _, result in detections):
        return synthetic_rows, {
            "enabled": True,
            "status": "no_valid_detections",
            "frames_attempted": len(synthetic_rows),
            "frames_updated": 0,
        }

    ellipse_seed_rows = []
    for row, result in detections:
        if result is None:
            ellipse_seed_rows.append(type("Ann", (), {"ellipse_xywht": tuple(row["ellipse_sensor_xywht"])})())
        else:
            ellipse_seed_rows.append(type("Ann", (), {"ellipse_xywht": tuple(result["ellipse_xywht"])})())
    eye_region = derive_eye_region(ellipse_seed_rows, sensor_size=(SENSOR_WIDTH, SENSOR_HEIGHT))

    updated_rows: List[Dict] = []
    frames_updated = 0
    for row, result in detections:
        updated = dict(row)
        updated["eye_region_xywh"] = eye_region.to_list()
        updated["eye_region_bbox_xywh_sensor"] = eye_region.to_list()
        updated["roi_size_wh"] = [int(eye_region.w), int(eye_region.h)]
        updated["frame_source_size_wh"] = [int(eye_region.w), int(eye_region.h)]
        updated["event_source_size_wh"] = [int(eye_region.w), int(eye_region.h)]

        if result is not None:
            ellipse_sensor = [float(v) for v in result["ellipse_xywht"]]
            updated["pupil_region_bbox_xywh_sensor"] = [float(v) for v in result["bbox_xywh"]]
            updated["pupil_ellipse_xywht_sensor"] = ellipse_sensor
            updated["ellipse_sensor_xywht"] = ellipse_sensor
            updated["ellipse_roi_xywht"] = ellipse_sensor_to_roi(ellipse_sensor, eye_region)
            updated["ellipse_frame_xywht"] = updated["ellipse_roi_xywht"]
            updated["ellipse_event_xywht"] = updated["ellipse_roi_xywht"]
            updated["state_xyabuv"] = [float(v) for v in xywht_to_xyabuv(np.asarray(ellipse_sensor, dtype=np.float32))]
            updated["annotation_source"] = "gsa_synthetic_rerun"
            updated["annotation_quality"] = float(max(0.0, min(1.0, 0.5 * result["box_confidence"] + 0.5 * result["mask_score"])))
            updated["mask_valid"] = True
            updated["closed_eye_flag"] = False
            frames_updated += 1
            if export_masks:
                mask_sensor = np.asarray(result["mask"], dtype=np.uint8)
                mask_roi = crop_mask(mask_sensor, eye_region)
                mask_path = session_dir / "labels" / "masks" / Path(str(row["frame_filename"])).with_suffix(".png").name
                ensure_dir(mask_path.parent)
                Image.fromarray((mask_roi * 255).astype("uint8")).save(mask_path)
                updated["mask_path"] = relativize_to(canonical_root, mask_path)
                updated["pupil_mask_path"] = updated["mask_path"]
        updated_rows.append(updated)

    return updated_rows, {
        "enabled": True,
        "status": "completed",
        "frames_attempted": len(synthetic_rows),
        "frames_updated": frames_updated,
    }


def _ellipse_from_row(row: Dict) -> list[float]:
    ellipse = row.get("pupil_ellipse_xywht_sensor") or row.get("ellipse_sensor_xywht") or row.get("ellipse_xywht")
    if ellipse is None:
        raise ValueError(f"Grounded-SAM row is missing ellipse fields: {row.get('ann_id')}")
    if len(ellipse) != 5:
        raise ValueError(f"Expected 5 ellipse values, got {ellipse}")
    return [float(v) for v in ellipse]


def _groundedsam_label_rows(
    *,
    rows: List[Dict],
    session_key: str,
    user_id: int,
    eye: str,
    session_code: str,
    annotation_root: Path,
    canonical_root: Path,
    session_dir: Path,
    export_masks: bool,
    link_mode: str,
    overwrite_links: bool,
    protocol_session_index: int | None,
    session_motion_regime_prior: Dict,
    data_mode: str,
    canonical_name: str,
    frame_source: str,
) -> List[Dict]:
    annotations = []
    for row in rows:
        ellipse = _ellipse_from_row(row)
        frame_filename = str(row.get("frame_filename") or "")
        frame_idx = row.get("frame_idx")
        timestamp_us = int(row.get("timestamp_us", 0))
        if frame_idx is None and frame_filename:
            parsed_idx = Path(frame_filename).stem.split("_", 1)[0]
            frame_idx = int(parsed_idx) if parsed_idx.isdigit() else None
        annotations.append(
            {
                "row": row,
                "frame_filename": frame_filename,
                "frame_idx": frame_idx,
                "timestamp_us": timestamp_us,
                "ellipse_sensor": ellipse,
            }
        )
    annotations.sort(key=lambda item: (int(item["frame_idx"]) if item["frame_idx"] is not None else -1, int(item["timestamp_us"]), str(item["frame_filename"])))
    eye_region = derive_eye_region(
        [type("Ann", (), {"ellipse_xywht": tuple(item["ellipse_sensor"])})() for item in annotations],
        sensor_size=(SENSOR_WIDTH, SENSOR_HEIGHT),
    )
    label_rows: List[Dict] = []
    for item in annotations:
        row = dict(item["row"])
        ellipse_sensor = [float(v) for v in item["ellipse_sensor"]]
        pupil_region_bbox = row.get("pupil_region_bbox_xywh_sensor")
        if pupil_region_bbox is None:
            pupil_region_bbox = [
                float(ellipse_sensor[0] - 0.5 * ellipse_sensor[2]),
                float(ellipse_sensor[1] - 0.5 * ellipse_sensor[3]),
                float(ellipse_sensor[2]),
                float(ellipse_sensor[3]),
            ]
        frame_filename = str(item["frame_filename"])
        frame_path = session_dir / "frames" / frame_filename
        normalized = {
            **row,
            "ann_id": str(row.get("ann_id") or f"{session_key.replace('/', '__')}__{Path(frame_filename).stem}"),
            "session_key": session_key,
            "user_id": int(user_id),
            "subject_id": int(user_id),
            "eye": str(eye),
            "session_code": str(session_code),
            "data_mode": str(data_mode),
            "canonical_name": str(canonical_name),
            "frame_source": str(frame_source),
            "sample_timestamp_us": int(item["timestamp_us"]),
            "session_official": bool(is_official_session_code(session_code)),
            "frame_filename": frame_filename,
            "frame_idx": item["frame_idx"],
            "timestamp_us": int(item["timestamp_us"]),
            "frame_path": relativize_to(canonical_root, frame_path),
            "eye_region_xywh": eye_region.to_list(),
            "eye_region_bbox_xywh_sensor": eye_region.to_list(),
            "roi_size_wh": [int(eye_region.w), int(eye_region.h)],
            "frame_source_size_wh": [int(eye_region.w), int(eye_region.h)],
            "event_source_size_wh": [int(eye_region.w), int(eye_region.h)],
            "pupil_region_bbox_xywh_sensor": [float(v) for v in pupil_region_bbox],
            "pupil_ellipse_xywht_sensor": ellipse_sensor,
            "ellipse_sensor_xywht": ellipse_sensor,
            "ellipse_roi_xywht": ellipse_sensor_to_roi(ellipse_sensor, eye_region),
            "ellipse_frame_xywht": ellipse_sensor_to_roi(ellipse_sensor, eye_region),
            "ellipse_event_xywht": ellipse_sensor_to_roi(ellipse_sensor, eye_region),
            "state_xyabuv": [float(v) for v in xywht_to_xyabuv(ellipse_sensor)],
            "sensor_size_wh": [SENSOR_WIDTH, SENSOR_HEIGHT],
            "protocol_session_index": protocol_session_index,
            "session_motion_regime_prior": dict(session_motion_regime_prior),
            "annotation_source": str(row.get("annotation_source", "gsa_auto")),
            "annotation_quality": float(row.get("annotation_quality", 1.0)),
            "closed_eye_flag": bool(row.get("closed_eye_flag", False)),
            "blink_candidate_score": float(row.get("blink_candidate_score", 0.0)),
            "blink_candidate_reasons": list(row.get("blink_candidate_reasons", [])),
            "blink_candidate_source": row.get("blink_candidate_source"),
            "mask_valid": bool(row.get("mask_valid", True)),
        }
        if export_masks:
            stored_mask_path = row.get("pupil_mask_path") or row.get("mask_path")
            src_mask_path = resolve_groundedsam_mask_path(annotation_root, stored_mask_path)
            if src_mask_path is not None and src_mask_path.exists():
                dst_mask_path = session_dir / "labels" / "masks" / Path(frame_filename).with_suffix(".png").name
                maybe_link_or_copy(src_mask_path, dst_mask_path, mode=link_mode, overwrite=overwrite_links)
                normalized["mask_path"] = relativize_to(canonical_root, dst_mask_path)
                normalized["pupil_mask_path"] = normalized["mask_path"]
            else:
                mask_sensor = rasterize_ellipse_mask(ellipse_sensor, image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
                mask_roi = crop_mask(mask_sensor, eye_region)
                dst_mask_path = session_dir / "labels" / "masks" / Path(frame_filename).with_suffix(".png").name
                ensure_dir(dst_mask_path.parent)
                from PIL import Image

                Image.fromarray((mask_roi * 255).astype("uint8")).save(dst_mask_path)
                normalized["mask_path"] = relativize_to(canonical_root, dst_mask_path)
                normalized["pupil_mask_path"] = normalized["mask_path"]
        else:
            normalized["mask_path"] = None
            normalized["pupil_mask_path"] = None
        label_rows.append(normalized)
    return label_rows


def _annotation_row(
    *,
    ann,
    session_key: str,
    user_id: int,
    eye: str,
    session_code: str,
    eye_region,
    canonical_root: Path,
    session_dir: Path,
    export_masks: bool,
    protocol_session_index: int | None,
    session_motion_regime_prior: Dict,
    data_mode: str,
    canonical_name: str,
    frame_source: str,
    blink_metadata: Optional[Dict] = None,
) -> Dict:
    roi_size = [int(eye_region.w), int(eye_region.h)]
    ellipse_sensor = [float(v) for v in ann.ellipse_xywht]
    ellipse_roi = ellipse_sensor_to_roi(ann.ellipse_xywht, eye_region)
    state6 = [float(v) for v in xywht_to_xyabuv(ellipse_sensor)]
    pupil_region_bbox = [
        float(ellipse_sensor[0] - 0.5 * ellipse_sensor[2]),
        float(ellipse_sensor[1] - 0.5 * ellipse_sensor[3]),
        float(ellipse_sensor[2]),
        float(ellipse_sensor[3]),
    ]
    frame_rel = relativize_to(canonical_root, session_dir / "frames" / ann.frame_filename)
    row = ann.to_dict()
    row["ann_id"] = f"{session_key.replace('/', '__')}__{Path(ann.frame_filename).stem}"
    row["session_key"] = session_key
    row["user_id"] = user_id
    row["subject_id"] = user_id
    row["eye"] = eye
    row["session_code"] = session_code
    row["data_mode"] = str(data_mode)
    row["canonical_name"] = str(canonical_name)
    row["frame_source"] = str(frame_source)
    row["sample_timestamp_us"] = int(getattr(ann, "timestamp_us", 0))
    row["session_official"] = bool(is_official_session_code(session_code))
    row["frame_path"] = frame_rel
    row["eye_region_xywh"] = eye_region.to_list()
    row["eye_region_bbox_xywh_sensor"] = eye_region.to_list()
    row["roi_size_wh"] = roi_size
    row["frame_source_size_wh"] = roi_size
    row["event_source_size_wh"] = roi_size
    row["pupil_region_bbox_xywh_sensor"] = pupil_region_bbox
    row["pupil_ellipse_xywht_sensor"] = ellipse_sensor
    row["ellipse_sensor_xywht"] = ellipse_sensor
    row["ellipse_roi_xywht"] = ellipse_roi
    row["ellipse_frame_xywht"] = ellipse_roi
    row["ellipse_event_xywht"] = ellipse_roi
    row["state_xyabuv"] = state6
    row["sensor_size_wh"] = [SENSOR_WIDTH, SENSOR_HEIGHT]
    row["protocol_session_index"] = protocol_session_index
    row["session_motion_regime_prior"] = dict(session_motion_regime_prior)
    row["annotation_source"] = "manual_csv"
    row["annotation_quality"] = 1.0
    blink_metadata = dict(blink_metadata or {})
    row["closed_eye_flag"] = bool(blink_metadata.get("closed_eye_flag", False))
    row["blink_candidate_score"] = float(blink_metadata.get("blink_candidate_score", 0.0))
    row["blink_candidate_reasons"] = list(blink_metadata.get("blink_candidate_reasons", []))
    row["blink_candidate_source"] = blink_metadata.get("blink_candidate_source")
    row["mask_valid"] = True

    if export_masks:
        mask_sensor = rasterize_ellipse_mask(ann.ellipse_xywht, image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
        mask_roi = crop_mask(mask_sensor, eye_region)
        mask_name = Path(ann.frame_filename).with_suffix(".png").name
        mask_path = session_dir / "labels" / "masks" / mask_name
        ensure_dir(mask_path.parent)
        from PIL import Image

        Image.fromarray((mask_roi * 255).astype("uint8")).save(mask_path)
        row["mask_path"] = relativize_to(canonical_root, mask_path)
        row["pupil_mask_path"] = row["mask_path"]
    else:
        row["mask_path"] = None
        row["pupil_mask_path"] = None
    return row


def canonicalize_session(
    *,
    raw_session_dir: Path,
    canonical_root: Path,
    user_id: int,
    eye: str,
    default_frame_size: tuple[int, int],
    default_event_size: tuple[int, int],
    export_masks: bool,
    link_mode: str,
    overwrite_links: bool,
    strict_layout: bool,
    skip_nonstandard_sessions: bool,
    annotation_mode: str,
    annotation_root: Path | None,
    use_raw_ellipse_blink_heuristic: bool,
    annotation_backend: str,
    groundedsam_root: Path | None,
    ultralytics_root: Path | None,
    groundedsam2_root: Path | None,
    data_mode: str,
    canonical_name: str,
    frame_source: str,
    interpolation_alpha: float,
    interpolation_model: str,
    interpolation_backend: str | None,
    interpolation_target_fps: float | None,
    interpolation_fixed_insert: int | None,
    interpolation_count_policy: str,
    interpolation_max_insert: int | None,
    interpolation_timelens_root: Path | None,
    interpolation_timelens_checkpoint: Path | None,
    interpolation_timelens_device: str,
    interpolation_timelens_xl_root: Path | None,
    interpolation_timelens_xl_checkpoint: Path | None,
    interpolation_timelens_xl_device: str,
    event_generation_backend: str,
    v2e_root: Path | None,
    v2e_device: str,
    synthetic_overlap_policy: str,
) -> Optional[Dict]:
    layout = discover_session_layout(raw_session_dir, user_id=user_id)
    session_code = layout.session_code
    protocol_session_index, _protocol_flags = derive_protocol_session_index(raw_session_dir.name, session_code)
    session_motion_regime_prior = derive_session_motion_regime_prior(protocol_session_index)

    if skip_nonstandard_sessions and not layout.is_official_session:
        return {
            "skipped": True,
            "skip_reason": "nonstandard_session_code",
            "user_id": user_id,
            "eye": eye,
            "session_code": session_code,
            "session_dir_name": raw_session_dir.name,
            "session_key": f"{canonical_user_name(user_id)}/{eye}/session_{session_code}",
            "layout_warnings": layout.warnings,
            "protocol_session_index": protocol_session_index,
            "session_motion_regime_prior": session_motion_regime_prior,
            "session_package_path": None,
            "data_mode": str(data_mode),
            "canonical_name": str(canonical_name),
            "frame_source": str(frame_source),
        }

    if strict_layout and any(x in layout.warnings for x in ("frames_dir_missing", "event_file_missing")):
        raise FileNotFoundError(f"Invalid session layout for {raw_session_dir}: {layout.warnings}")

    if layout.frames_dir is None:
        raise FileNotFoundError(f"Missing frames directory: {raw_session_dir}")
    if layout.event_file is None:
        raise FileNotFoundError(f"Missing event file: {raw_session_dir}")

    session_dir = canonical_root / "sessions" / canonical_user_name(user_id) / eye / f"session_{session_code}"
    ensure_dir(session_dir / "frames")
    ensure_dir(session_dir / "events")
    ensure_dir(session_dir / "labels")
    if export_masks:
        ensure_dir(session_dir / "labels" / "masks")

    frame_records = collect_frame_records(layout.frames_dir)
    frame_names = [rec.filename for rec in frame_records]
    frame_stems = [Path(rec.filename).stem for rec in frame_records]
    frame_index_path = session_dir / "labels" / "frame_index.jsonl"
    if not (str(data_mode).strip().lower() == "mode2" and str(frame_source).strip().lower() == "interpolated"):
        for rec in frame_records:
            maybe_link_or_copy(
                layout.frames_dir / rec.filename,
                session_dir / "frames" / rec.filename,
                mode=link_mode,
                overwrite=overwrite_links,
            )

    dst_event_npz = session_dir / "events" / "events.npz"
    if layout.event_file.suffix == ".npz":
        maybe_link_or_copy(layout.event_file, dst_event_npz, mode=link_mode, overwrite=overwrite_links)
        event_origin = "npz"
    else:
        events = load_events_from_txt(layout.event_file, eye=eye)
        save_events_npz(events, dst_event_npz)
        event_origin = "txt"

    gsam_rows: List[Dict] = []
    annotation_source_kind = "manual_csv"
    if annotation_mode in {"groundedsam", "auto"} and annotation_root is not None:
        store_path = groundedsam_annotation_store_path(annotation_root, user_id=user_id, eye=eye, session_code=session_code)
        if store_path.exists():
            gsam_rows = load_groundedsam_annotation_store(annotation_root, user_id=user_id, eye=eye, session_code=session_code)
            annotation_source_kind = "groundedsam"
        elif annotation_mode == "groundedsam":
            raise FileNotFoundError(f"Grounded-SAM annotation store not found: {store_path}")

    if gsam_rows:
        annotations = []
        annotation_parse_report = {
            "annotation_mode": annotation_mode,
            "annotation_source": "groundedsam",
            "store_path": str(groundedsam_annotation_store_path(annotation_root, user_id=user_id, eye=eye, session_code=session_code)),
            "annotations_kept": len(gsam_rows),
        }
    elif layout.annotation_csv:
        annotations, annotation_parse_report = parse_via_csv_with_report(
            layout.annotation_csv,
            known_frame_filenames=frame_names,
            known_frame_stems=frame_stems,
        )
    else:
        annotations, annotation_parse_report = [], {
            "csv_exists": 0,
            "rows_total": 0,
            "rows_region_positive": 0,
            "groups_total": 0,
            "annotations_kept": 0,
            "groups_skipped_bad_filename": 0,
            "groups_skipped_unknown_frame": 0,
            "groups_skipped_nonellipse": 0,
            "groups_skipped_bad_shape": 0,
            "groups_skipped_bad_values": 0,
        }

    blink_metadata_by_frame: Dict[str, Dict] = {}
    blink_candidate_report: Dict[str, object] = {
        "enabled": bool(use_raw_ellipse_blink_heuristic and (bool(gsam_rows) or layout.annotation_csv is not None)),
        "source": None,
        "n_annotations": 0,
        "n_blink_candidates": 0,
        "candidate_rate": 0.0,
    }
    if gsam_rows:
        if not use_raw_ellipse_blink_heuristic:
            blink_candidate_report["source"] = "disabled"
        else:
            gsam_rows, blink_candidate_report = apply_groundedsam_store_blink_metadata(gsam_rows, overwrite_existing=False)
            blink_candidate_report["annotation_source"] = "groundedsam_store"
    elif layout.annotation_csv is None:
        blink_candidate_report["source"] = "annotation_csv_missing"
    elif not use_raw_ellipse_blink_heuristic:
        blink_candidate_report["source"] = "disabled"
    else:
        blink_metadata_by_frame, blink_candidate_report = build_raw_ellipse_blink_metadata(annotations)
    annotation_parse_report["blink_candidate_report"] = blink_candidate_report

    session_key = f"{canonical_user_name(user_id)}/{eye}/session_{session_code}"
    if gsam_rows:
        label_rows = _groundedsam_label_rows(
            rows=gsam_rows,
            session_key=session_key,
            user_id=user_id,
            eye=eye,
            session_code=session_code,
            annotation_root=annotation_root,
            canonical_root=canonical_root,
            session_dir=session_dir,
            export_masks=export_masks,
            link_mode=link_mode,
            overwrite_links=overwrite_links,
            protocol_session_index=protocol_session_index,
            session_motion_regime_prior=session_motion_regime_prior,
            data_mode=data_mode,
            canonical_name=canonical_name,
            frame_source=frame_source,
        )
        eye_region = derive_eye_region(
            [type("Ann", (), {"ellipse_xywht": tuple(row["ellipse_sensor_xywht"])})() for row in label_rows],
            sensor_size=(SENSOR_WIDTH, SENSOR_HEIGHT),
        )
    else:
        eye_region = derive_eye_region(annotations, sensor_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
        label_rows = [
            _annotation_row(
                ann=ann,
                session_key=session_key,
                user_id=user_id,
                eye=eye,
                session_code=session_code,
                eye_region=eye_region,
                canonical_root=canonical_root,
                session_dir=session_dir,
                export_masks=export_masks,
                protocol_session_index=protocol_session_index,
                session_motion_regime_prior=session_motion_regime_prior,
                data_mode=data_mode,
                canonical_name=canonical_name,
                frame_source=frame_source,
                blink_metadata=blink_metadata_by_frame.get(str(ann.frame_filename), {}),
            )
            for ann in annotations
        ]

    interpolation_summary = None
    if str(data_mode).strip().lower() == "mode2" and str(frame_source).strip().lower() == "interpolated":
        synthetic_frame_records, synthetic_rows, interpolation_summary = _build_mode2_interpolated_rows(
            label_rows=label_rows,
            layout_frames_dir=layout.frames_dir,
            session_key=session_key,
            user_id=user_id,
            eye=eye,
            session_code=session_code,
            canonical_root=canonical_root,
            session_dir=session_dir,
            events_npz_path=dst_event_npz,
            export_masks=export_masks,
            eye_region=eye_region,
            protocol_session_index=protocol_session_index,
            session_motion_regime_prior=session_motion_regime_prior,
            data_mode=data_mode,
            canonical_name=canonical_name,
            frame_source=frame_source,
            interpolation_alpha=interpolation_alpha,
            interpolation_model=interpolation_model,
            interpolation_backend=interpolation_backend,
            interpolation_target_fps=interpolation_target_fps,
            interpolation_fixed_insert=interpolation_fixed_insert,
            interpolation_count_policy=interpolation_count_policy,
            interpolation_max_insert=interpolation_max_insert,
            interpolation_timelens_root=interpolation_timelens_root,
            interpolation_timelens_checkpoint=interpolation_timelens_checkpoint,
            interpolation_timelens_device=interpolation_timelens_device,
            interpolation_timelens_xl_root=interpolation_timelens_xl_root,
            interpolation_timelens_xl_checkpoint=interpolation_timelens_xl_checkpoint,
            interpolation_timelens_xl_device=interpolation_timelens_xl_device,
            synthetic_overlap_policy=synthetic_overlap_policy,
            annotation_source_kind=annotation_source_kind,
        )
        if interpolation_summary is not None:
            interpolation_summary["event_generation_backend"] = str(event_generation_backend)
            interpolation_summary["v2e_root"] = None if v2e_root is None else str(v2e_root.resolve())
            interpolation_summary["v2e_device"] = str(v2e_device)
        if synthetic_rows:
            frame_records = synthetic_frame_records
            label_rows = synthetic_rows
            frame_names = [rec.filename for rec in frame_records]
            frame_stems = [Path(rec.filename).stem for rec in frame_records]
        else:
            for rec in frame_records:
                maybe_link_or_copy(
                    layout.frames_dir / rec.filename,
                    session_dir / "frames" / rec.filename,
                    mode=link_mode,
                    overwrite=overwrite_links,
                )

        if synthetic_rows and annotation_mode in {"groundedsam", "auto"} and groundedsam_root is not None:
            rerun_rows, rerun_summary = _rerun_mode2_groundedsam_rows(
                synthetic_rows=label_rows,
                canonical_root=canonical_root,
                session_dir=session_dir,
                export_masks=export_masks,
                annotation_backend=annotation_backend,
                groundedsam_root=groundedsam_root,
                ultralytics_root=ultralytics_root,
                groundedsam2_root=groundedsam2_root,
            )
            label_rows = rerun_rows
            if interpolation_summary is None:
                interpolation_summary = {}
            interpolation_summary["groundedsam_rerun"] = rerun_summary
            if rerun_summary is not None and rerun_summary.get("frames_updated", 0) > 0:
                annotation_source_kind = "groundedsam_synthetic_rerun"

    artifact_writer = CanonicalSessionArtifactWriter(
        canonical_root=canonical_root,
        relativize_fn=relativize_to,
        session_package_builder=build_session_package,
    )
    return artifact_writer.write(
        session_key=session_key,
        user_id=user_id,
        eye=eye,
        session_code=session_code,
        raw_session_dir=raw_session_dir,
        session_dir=session_dir,
        frame_records=frame_records,
        label_rows=label_rows,
        eye_region=eye_region,
        dst_event_npz=dst_event_npz,
        event_origin=event_origin,
        layout=layout,
        annotation_mode=annotation_mode,
        annotation_source_kind=annotation_source_kind,
        annotation_parse_report=annotation_parse_report,
        blink_candidate_report=blink_candidate_report,
        data_mode=data_mode,
        canonical_name=canonical_name,
        frame_source=frame_source,
        protocol_session_index=protocol_session_index,
        session_motion_regime_prior=session_motion_regime_prior,
        use_raw_ellipse_blink_heuristic=use_raw_ellipse_blink_heuristic,
        interpolation_summary=interpolation_summary,
        sensor_width=SENSOR_WIDTH,
        sensor_height=SENSOR_HEIGHT,
    )


def _canonicalize_session_job(job: Dict) -> Optional[Dict]:
    return CanonicalSessionProcessor(session_callable=canonicalize_session).process_job(job)


def canonicalize_dataset(
    *,
    raw_root: Path,
    canonical_root: Path,
    indexes_root: Optional[Path] = None,
    default_frame_size: tuple[int, int] = DEFAULT_FRAME_SIZE,
    default_event_size: tuple[int, int] = DEFAULT_EVENT_SIZE,
    export_masks: bool = True,
    link_mode: str = "symlink",
    overwrite_links: bool = False,
    strict_layout: bool = False,
    skip_nonstandard_sessions: bool = True,
    annotation_mode: str = "auto",
    annotation_root: Optional[Path] = None,
    use_raw_ellipse_blink_heuristic: bool = True,
    annotation_backend: str = DEFAULT_ANNOTATION_BACKEND,
    groundedsam_root: Optional[Path] = None,
    ultralytics_root: Optional[Path] = None,
    groundedsam2_root: Optional[Path] = None,
    data_mode: str = "mode1",
    canonical_name: str = "canonical1",
    frame_source: str = "original",
    interpolation_alpha: float = 0.5,
    interpolation_model: str = "linear_blend",
    interpolation_backend: str | None = None,
    interpolation_target_fps: float | None = None,
    interpolation_fixed_insert: int | None = None,
    interpolation_count_policy: str = "round",
    interpolation_max_insert: int | None = None,
    interpolation_timelens_root: Path | None = None,
    interpolation_timelens_checkpoint: Path | None = None,
    interpolation_timelens_device: str = "cpu",
    interpolation_timelens_xl_root: Path | None = None,
    interpolation_timelens_xl_checkpoint: Path | None = None,
    interpolation_timelens_xl_device: str = "cpu",
    event_generation_backend: str = DEFAULT_EVENT_GENERATION_BACKEND,
    v2e_root: Path | None = None,
    v2e_device: str = "cpu",
    synthetic_overlap_policy: str = "reuse_event_window",
    num_workers: int = 1,
    log_every: int = 10,
) -> Dict:
    mode_defaults = _mode_defaults(data_mode)
    resolved_data_mode = mode_defaults["data_mode"]
    requested_annotation_mode = str(annotation_mode).strip().lower()
    if resolved_data_mode == "mode0" and requested_annotation_mode == "groundedsam":
        raise ValueError("mode0 requires raw CSV supervision and does not allow annotation_mode=groundedsam")
    resolved_annotation_mode = mode_defaults["annotation_mode"] if resolved_data_mode == "mode0" else str(annotation_mode)
    resolved_canonical_name = str(canonical_name)
    if resolved_data_mode == "mode0" and resolved_canonical_name == "canonical1":
        resolved_canonical_name = "canonical0"
    elif resolved_data_mode == "mode2" and resolved_canonical_name == "canonical1":
        resolved_canonical_name = "canonical2"
    resolved_frame_source = str(frame_source)
    if resolved_data_mode == "mode0":
        resolved_frame_source = "original"
    elif resolved_data_mode == "mode2" and resolved_frame_source == "original":
        resolved_frame_source = "interpolated"

    raw_root = raw_root.resolve()
    canonical_workspace_root = canonical_root.resolve()
    canonical_root = resolve_canonical_dataset_root(canonical_workspace_root, resolved_canonical_name, prefer_nested=True).resolve()
    indexes_root = resolve_canonical_indexes_root(
        canonical_workspace_root,
        canonical_name=resolved_canonical_name,
        indexes_root=indexes_root,
        prefer_nested=True,
    ).resolve()
    resolved_annotation_backend = str(annotation_backend).strip().lower()
    if resolved_annotation_mode == "groundedsam" and annotation_root is None:
        raise ValueError("annotation_root is required when annotation_mode=groundedsam")
    if resolved_annotation_mode == "groundedsam" and resolved_annotation_backend == "ultralytics_sam3" and ultralytics_root is None:
        raise ValueError("ultralytics_root is required when annotation_backend=ultralytics_sam3")
    if resolved_annotation_mode == "groundedsam" and resolved_annotation_backend == "groundedsam2" and groundedsam2_root is None:
        raise ValueError("groundedsam2_root is required when annotation_backend=groundedsam2")
    groundedsam_root = None if groundedsam_root is None else groundedsam_root.resolve()
    ultralytics_root = None if ultralytics_root is None else ultralytics_root.resolve()
    groundedsam2_root = None if groundedsam2_root is None else groundedsam2_root.resolve()
    v2e_root = None if v2e_root is None else Path(v2e_root).resolve()
    request = CanonicalizeResolvedRequest(
        raw_root=raw_root,
        canonical_root=canonical_root,
        canonical_workspace_root=canonical_workspace_root,
        indexes_root=indexes_root,
        default_frame_size=default_frame_size,
        default_event_size=default_event_size,
        export_masks=export_masks,
        link_mode=link_mode,
        overwrite_links=overwrite_links,
        strict_layout=strict_layout,
        skip_nonstandard_sessions=skip_nonstandard_sessions,
        resolved_annotation_mode=str(resolved_annotation_mode),
        annotation_root=None if annotation_root is None else annotation_root,
        use_raw_ellipse_blink_heuristic=bool(use_raw_ellipse_blink_heuristic),
        annotation_backend=resolved_annotation_backend,
        groundedsam_root=None if groundedsam_root is None else groundedsam_root,
        ultralytics_root=None if ultralytics_root is None else ultralytics_root,
        groundedsam2_root=None if groundedsam2_root is None else groundedsam2_root,
        resolved_data_mode=str(resolved_data_mode),
        resolved_canonical_name=str(resolved_canonical_name),
        resolved_frame_source=str(resolved_frame_source),
        interpolation_alpha=float(interpolation_alpha),
        interpolation_model=str(interpolation_model),
        interpolation_backend=None if interpolation_backend is None else str(interpolation_backend),
        interpolation_target_fps=None if interpolation_target_fps is None else float(interpolation_target_fps),
        interpolation_fixed_insert=None if interpolation_fixed_insert is None else int(interpolation_fixed_insert),
        interpolation_count_policy=str(interpolation_count_policy),
        interpolation_max_insert=None if interpolation_max_insert is None else int(interpolation_max_insert),
        interpolation_timelens_root=None if interpolation_timelens_root is None else Path(interpolation_timelens_root).resolve(),
        interpolation_timelens_checkpoint=None if interpolation_timelens_checkpoint is None else Path(interpolation_timelens_checkpoint).resolve(),
        interpolation_timelens_device=str(interpolation_timelens_device),
        interpolation_timelens_xl_root=None if interpolation_timelens_xl_root is None else Path(interpolation_timelens_xl_root).resolve(),
        interpolation_timelens_xl_checkpoint=None if interpolation_timelens_xl_checkpoint is None else Path(interpolation_timelens_xl_checkpoint).resolve(),
        interpolation_timelens_xl_device=str(interpolation_timelens_xl_device),
        event_generation_backend=str(event_generation_backend),
        v2e_root=v2e_root,
        v2e_device=str(v2e_device),
        synthetic_overlap_policy=str(synthetic_overlap_policy),
        effective_workers=max(int(num_workers), 1),
        log_stride=max(int(log_every), 1),
    )
    return CanonicalizeDatasetRunner(
        request=request,
        session_callable=canonicalize_session,
    ).run()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonicalize EV-Eye raw dataset for HBTXR")
    add_common_path_args(parser, need_raw=True, need_canonical=True)
    parser.add_argument("--frame-width", type=int, default=DEFAULT_FRAME_SIZE[0], help="Deprecated. Canonical stores source/sensor geometry and loader decides frame size.")
    parser.add_argument("--frame-height", type=int, default=DEFAULT_FRAME_SIZE[1], help="Deprecated. Canonical stores source/sensor geometry and loader decides frame size.")
    parser.add_argument("--event-width", type=int, default=DEFAULT_EVENT_SIZE[0], help="Deprecated. Canonical stores source/sensor geometry and loader decides event size.")
    parser.add_argument("--event-height", type=int, default=DEFAULT_EVENT_SIZE[1], help="Deprecated. Canonical stores source/sensor geometry and loader decides event size.")
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite-links", action="store_true", help="Replace existing frame/event links when re-running canonicalization")
    parser.add_argument("--no-export-masks", action="store_true")
    parser.add_argument("--strict-layout", action="store_true")
    parser.add_argument("--include-nonstandard-sessions", action="store_true")
    parser.add_argument("--annotation-mode", choices=["auto", "manual_csv", "groundedsam"], default="auto", help="Annotation source selection for canonicalization")
    parser.add_argument("--annotation-backend", choices=list(SUPPORTED_ANNOTATION_BACKENDS), default=DEFAULT_ANNOTATION_BACKEND, help="Annotation runtime backend used when annotation_mode=groundedsam")
    parser.add_argument("--disable-raw-ellipse-blink-heuristic", action="store_true", help="Do not derive closed_eye_flag from raw CSV or Grounded-SAM ellipse-collapse heuristics during canonicalization")
    parser.add_argument("--data-mode", choices=["mode0", "mode1", "mode2"], default="mode1")
    parser.add_argument("--canonical-name", type=str, default="canonical1")
    parser.add_argument("--frame-source", type=str, default="original")
    parser.add_argument("--interp-alpha", type=float, default=0.5, help="Alpha used for synthetic mode2 frame interpolation")
    parser.add_argument("--interp-model", type=str, default="linear_blend", help="Legacy interpolation backend/model tag recorded for mode2 synthetic frames")
    parser.add_argument("--interp-backend", type=str, default=None, choices=["linear_blend", "timelens", "timelens_xl"], help="Interpolation backend used for mode2 synthetic frame generation")
    parser.add_argument("--interp-target-fps", type=float, default=None, help="Automatically compute inserted synthetic frames so that source gaps match the target FPS")
    parser.add_argument("--interp-fixed-insert", type=int, default=None, help="Explicit number of synthetic frames inserted between two source frames")
    parser.add_argument("--interp-count-policy", type=str, default="round", choices=["round", "floor", "ceil"], help="Rounding policy used when target FPS is converted to an insert count")
    parser.add_argument("--interp-max-insert", type=int, default=None, help="Optional cap for inserted synthetic frames per source pair")
    parser.add_argument("--interp-timelens-root", type=str, default=None, help="Local TimeLens reference repo root used when --interp-backend timelens. Falls back to --timelens-root / paths-config / packages/timelens when available")
    parser.add_argument("--interp-timelens-checkpoint", type=str, default=None, help="TimeLens checkpoint path. Defaults to <timelens_root>/refined_model/attention.bin when present")
    parser.add_argument("--interp-timelens-device", type=str, default="cpu", help="Torch device used for TimeLens interpolation")
    parser.add_argument("--interp-timelens-xl-root", type=str, default=None, help="Local TimeLens-XL repo root used when --interp-backend timelens_xl. Falls back to --timelens-xl-root / paths-config / Third/FI when available")
    parser.add_argument("--interp-timelens-xl-checkpoint", type=str, default=None, help="TimeLens-XL checkpoint path. Defaults to the expected checkpoint under the TimeLens-XL root when present")
    parser.add_argument("--interp-timelens-xl-device", type=str, default="cpu", help="Torch device used for TimeLens-XL interpolation")
    parser.add_argument("--event-generation-backend", type=str, default=DEFAULT_EVENT_GENERATION_BACKEND, choices=list(SUPPORTED_EVENT_GENERATION_BACKENDS), help="Synthetic event-generation backend recorded for downstream mode2 target-FPS / event materialization workflows")
    parser.add_argument("--v2e-device", type=str, default="cpu", help="Torch device used for v2e synthetic event generation when event_generation_backend=v2e")
    parser.add_argument("--synthetic-overlap-policy", type=str, default="reuse_event_window", help="Provenance tag describing how synthetic event windows are aligned")
    parser.add_argument("--num-workers", type=int, default=1, help="Use ProcessPoolExecutor when > 1")
    parser.add_argument("--log-every", type=int, default=10, help="Print progress every N processed sessions")
    return parser


def run(args: argparse.Namespace) -> Dict:
    paths = resolve_paths(args, need_raw=True, need_canonical=True)
    resolved_timelens_root = paths.timelens_root if getattr(args, "interp_timelens_root", None) is None else Path(args.interp_timelens_root)
    resolved_timelens_xl_root = paths.timelens_xl_root if getattr(args, "interp_timelens_xl_root", None) is None else Path(args.interp_timelens_xl_root)
    return canonicalize_dataset(
        raw_root=paths.raw_root,
        canonical_root=paths.canonical_root,
        indexes_root=None if getattr(args, "indexes_root", None) is None else paths.indexes_root,
        default_frame_size=(int(args.frame_width), int(args.frame_height)),
        default_event_size=(int(args.event_width), int(args.event_height)),
        export_masks=not bool(args.no_export_masks),
        link_mode=str(args.link_mode),
        overwrite_links=bool(args.overwrite_links),
        strict_layout=bool(args.strict_layout),
        skip_nonstandard_sessions=not bool(args.include_nonstandard_sessions),
        annotation_mode=str(args.annotation_mode),
        annotation_root=paths.annotation_root,
        use_raw_ellipse_blink_heuristic=not bool(args.disable_raw_ellipse_blink_heuristic),
        annotation_backend=str(args.annotation_backend),
        groundedsam_root=paths.groundedsam_root,
        ultralytics_root=paths.ultralytics_root,
        groundedsam2_root=paths.groundedsam2_root,
        data_mode=str(args.data_mode),
        canonical_name=str(args.canonical_name),
        frame_source=str(args.frame_source),
        interpolation_alpha=float(args.interp_alpha),
        interpolation_model=str(args.interp_model),
        interpolation_backend=None if getattr(args, "interp_backend", None) is None else str(args.interp_backend),
        interpolation_target_fps=None if getattr(args, "interp_target_fps", None) is None else float(args.interp_target_fps),
        interpolation_fixed_insert=None if getattr(args, "interp_fixed_insert", None) is None else int(args.interp_fixed_insert),
        interpolation_count_policy=str(args.interp_count_policy),
        interpolation_max_insert=None if getattr(args, "interp_max_insert", None) is None else int(args.interp_max_insert),
        interpolation_timelens_root=resolved_timelens_root,
        interpolation_timelens_checkpoint=None if getattr(args, "interp_timelens_checkpoint", None) is None else Path(args.interp_timelens_checkpoint),
        interpolation_timelens_device=str(args.interp_timelens_device),
        interpolation_timelens_xl_root=resolved_timelens_xl_root,
        interpolation_timelens_xl_checkpoint=None if getattr(args, "interp_timelens_xl_checkpoint", None) is None else Path(args.interp_timelens_xl_checkpoint),
        interpolation_timelens_xl_device=str(args.interp_timelens_xl_device),
        event_generation_backend=str(args.event_generation_backend),
        v2e_root=paths.v2e_root,
        v2e_device=str(args.v2e_device),
        synthetic_overlap_policy=str(args.synthetic_overlap_policy),
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
    )


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(f"[DONE] canonicalized {summary['n_sessions_ok']} sessions -> {summary['canonical_root']}")
