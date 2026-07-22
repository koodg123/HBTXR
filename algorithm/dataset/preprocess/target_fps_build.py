from __future__ import annotations

import time
from bisect import bisect_left
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image

from dataset.annotation.annotation_groundedsam import (
    groundedsam_annotation_store_path,
    mask_bbox_xywh,
    mask_to_ellipse_xywht,
    resolve_groundedsam_mask_path,
)
from dataset.preprocess.event_generation import (
    DEFAULT_EVENT_GENERATION_BACKEND,
    generate_event_packets,
)
from dataset.preprocess.interpolation import DEFAULT_INTERPOLATION_BACKEND, interpolate_pair
from dataset.preprocess.io_utils import (
    FrameRecord,
    canonical_user_name,
    collect_frame_records,
    collect_users,
    discover_session_layout,
    ellipse_sensor_to_roi,
    ensure_dir,
    load_events_from_txt,
    parse_via_csv_with_report,
    rasterize_ellipse_mask,
)
from dataset.preprocess.progress import print_progress
from dataset.preprocess.raw_ellipse_blink import (
    apply_groundedsam_store_blink_metadata,
    build_raw_ellipse_blink_metadata,
)
from utils.io import read_jsonl, write_json, write_jsonl
from utils.state6 import xywht_to_xyabuv


def _import_h5py(required: bool = True):
    try:
        import h5py  # type: ignore
    except ImportError:
        if required:
            raise RuntimeError(
                "h5py is required for session_store_format='h5'. "
                "Install h5py, or use session_store_format='npz'/'auto'."
            ) from None
        return None
    return h5py


NO_CSV_GROUNDEDSAM_STEP_ORDER = [
    "eye_region_mask",
    "eye_region_bbox_from_eye_region_mask",
    "pupil_mask_inside_eye_roi_from_eye_region_bbox",
    "pupil_bbox_from_pupil_mask",
    "pupil_state_xywht_from_pupil_mask",
]


def normalize_target_fps_tag(target_fps: float) -> str:
    fps = float(target_fps)
    if fps <= 0.0:
        raise ValueError("target_fps must be > 0")
    text = f"{fps:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def compute_target_step_us(target_fps: float) -> int:
    fps = float(target_fps)
    if fps <= 0.0:
        raise ValueError("target_fps must be > 0")
    return max(int(round(1_000_000.0 / fps)), 1)


def resolve_target_fps_root(target_root: str | Path, target_fps: float) -> Path:
    return Path(target_root).resolve() / f"fps_{normalize_target_fps_tag(target_fps)}"


def normalize_session_code_filter(session_code: str) -> str:
    text = str(session_code).strip().lower()
    if not text:
        raise ValueError("session_code must not be empty")
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 3:
        return digits
    raise ValueError(f"Unsupported session_code filter: {session_code!r}")


def build_target_timestamp_grid(frame_timestamps_us: Sequence[int], target_fps: float) -> list[int]:
    timestamps = [int(ts) for ts in frame_timestamps_us]
    if not timestamps:
        return []
    start_ts = timestamps[0]
    end_ts = timestamps[-1]
    step_us = compute_target_step_us(target_fps)
    grid: list[int] = []
    current = int(start_ts)
    while current <= end_ts:
        grid.append(int(current))
        current += step_us
    return grid


def _target_frame_filename(frame_index: int, timestamp_us: int) -> str:
    return f"{int(frame_index):06d}_{int(timestamp_us)}.png"


def build_target_frame_plan(frame_records: Sequence[FrameRecord], target_fps: float) -> list[dict]:
    records = sorted(frame_records, key=lambda rec: (int(rec.timestamp_us), int(rec.frame_idx or -1), str(rec.filename)))
    if not records:
        return []
    raw_timestamps = [int(rec.timestamp_us) for rec in records]
    raw_index_by_timestamp = {int(rec.timestamp_us): idx for idx, rec in enumerate(records)}
    target_timestamps = build_target_timestamp_grid(raw_timestamps, target_fps)

    rows: list[dict] = []
    for target_index, target_ts in enumerate(target_timestamps):
        matched_idx = raw_index_by_timestamp.get(int(target_ts))
        row = {
            "frame_index": int(target_index),
            "target_frame_filename": _target_frame_filename(target_index, target_ts),
            "target_timestamp_us": int(target_ts),
        }
        if matched_idx is not None:
            matched = records[matched_idx]
            row.update(
                {
                    "source_kind": "raw",
                    "matched_raw_frame_index": int(matched_idx),
                    "matched_raw_frame_filename": str(matched.filename),
                    "matched_raw_timestamp_us": int(matched.timestamp_us),
                    "source_pair_index": [-1, -1],
                    "source_pair_frame_filenames": [],
                    "source_pair_timestamps_us": [],
                    "interp_alpha": 0.0,
                }
            )
            rows.append(row)
            continue

        right_idx = bisect_left(raw_timestamps, int(target_ts))
        left_idx = right_idx - 1
        if left_idx < 0 or right_idx >= len(records):
            raise ValueError(f"Target timestamp {target_ts} could not be bracketed by raw frames")
        left = records[left_idx]
        right = records[right_idx]
        denom = max(int(right.timestamp_us) - int(left.timestamp_us), 1)
        alpha = float(int(target_ts) - int(left.timestamp_us)) / float(denom)
        row.update(
            {
                "source_kind": "interpolated",
                "matched_raw_frame_index": None,
                "matched_raw_frame_filename": None,
                "matched_raw_timestamp_us": None,
                "source_pair_index": [int(left_idx), int(right_idx)],
                "source_pair_frame_filenames": [str(left.filename), str(right.filename)],
                "source_pair_timestamps_us": [int(left.timestamp_us), int(right.timestamp_us)],
                "interp_alpha": float(alpha),
            }
        )
        rows.append(row)
    return rows


def resolve_session_store_format(session_store_format: str | None) -> str:
    text = str(session_store_format or "auto").strip().lower()
    if text not in {"auto", "h5", "npz"}:
        raise ValueError(f"Unsupported session_store_format: {session_store_format!r}")
    if text == "auto":
        return "h5" if _import_h5py(required=False) is not None else "npz"
    return text


def _load_frame_image(frame_path: Path) -> np.ndarray:
    with Image.open(frame_path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _load_source_frame_stack(frame_records: Sequence[FrameRecord], frames_dir: Path) -> np.ndarray:
    if not frame_records:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    return np.stack([_load_frame_image(frames_dir / str(record.filename)) for record in frame_records], axis=0).astype(np.uint8)


def _load_event_arrays(event_file: Path) -> dict[str, np.ndarray]:
    suffix = event_file.suffix.lower()
    if suffix == ".npz":
        raw = np.load(event_file)
        timestamp_key = "t" if "t" in raw.files else "timestamp_us"
        timestamps = np.asarray(raw[timestamp_key], dtype=np.int64)
        xs = np.asarray(raw["x"], dtype=np.int16)
        ys = np.asarray(raw["y"], dtype=np.int16)
        ps = np.asarray(raw["p"], dtype=np.int8)
    elif suffix == ".txt":
        eye = None
        for part in event_file.parts:
            lowered = str(part).strip().lower()
            if lowered in {"left", "right"}:
                eye = lowered
                break
        raw = load_events_from_txt(event_file, eye=eye)
        timestamps = np.asarray(raw["t"], dtype=np.int64)
        xs = np.asarray(raw["x"], dtype=np.int16)
        ys = np.asarray(raw["y"], dtype=np.int16)
        ps = np.asarray(raw["p"], dtype=np.int8)
    else:
        raise ValueError(f"Unsupported event file format for v3.2 builder: {event_file}")
    order = np.argsort(timestamps, kind="stable")
    return {
        "t": timestamps[order],
        "x": xs[order],
        "y": ys[order],
        "p": ps[order],
    }


def _render_target_frames(
    *,
    frame_plan: Sequence[dict],
    frame_records: Sequence[FrameRecord],
    frames_dir: Path,
    event_file: Path,
    interpolation_backend: str,
    timelens_root: Path | None = None,
    timelens_checkpoint: Path | None = None,
    timelens_device: str = "cpu",
    timelens_xl_root: Path | None = None,
    timelens_xl_checkpoint: Path | None = None,
    timelens_xl_device: str = "cpu",
) -> np.ndarray:
    records = sorted(frame_records, key=lambda rec: (int(rec.timestamp_us), int(rec.frame_idx or -1), str(rec.filename)))
    image_cache: dict[str, np.ndarray] = {}

    def _cached_image(filename: str) -> np.ndarray:
        image = image_cache.get(filename)
        if image is None:
            image = _load_frame_image(frames_dir / filename)
            image_cache[filename] = image
        return image

    rendered: list[np.ndarray] = []
    for row in frame_plan:
        if row.get("source_kind") == "raw":
            rendered.append(_cached_image(str(row["matched_raw_frame_filename"])))
            continue

        left_idx, right_idx = row["source_pair_index"]
        left = records[int(left_idx)]
        right = records[int(right_idx)]
        frame = interpolate_pair(
            backend=interpolation_backend,
            frame0=_cached_image(str(left.filename)),
            frame1=_cached_image(str(right.filename)),
            alpha=float(row.get("interp_alpha", 0.0)),
            timestamp0_us=int(left.timestamp_us),
            timestamp1_us=int(right.timestamp_us),
            events_path=event_file,
            timelens_root=timelens_root,
            timelens_checkpoint=timelens_checkpoint,
            timelens_device=str(timelens_device),
            timelens_xl_root=timelens_xl_root,
            timelens_xl_checkpoint=timelens_xl_checkpoint,
            timelens_xl_device=str(timelens_xl_device),
        )
        rendered.append(np.asarray(frame, dtype=np.uint8))

    if not rendered:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    return np.stack(rendered, axis=0).astype(np.uint8)


def build_target_event_packets(raw_events: dict[str, np.ndarray], frame_plan: Sequence[dict]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    timestamps = np.asarray(raw_events["t"], dtype=np.int64)
    if timestamps.size == 0:
        empty = np.zeros((len(frame_plan), 2), dtype=np.int64)
        return {
            "t": timestamps,
            "x": np.asarray(raw_events["x"], dtype=np.int16),
            "y": np.asarray(raw_events["y"], dtype=np.int16),
            "p": np.asarray(raw_events["p"], dtype=np.int8),
        }, empty

    ranges: list[list[int]] = []
    start_idx = 0
    for row in frame_plan:
        end_idx = int(np.searchsorted(timestamps, int(row["target_timestamp_us"]), side="right"))
        ranges.append([int(start_idx), int(end_idx)])
        start_idx = int(end_idx)

    total_end = int(ranges[-1][1]) if ranges else 0
    trimmed = {
        "t": timestamps[:total_end],
        "x": np.asarray(raw_events["x"], dtype=np.int16)[:total_end],
        "y": np.asarray(raw_events["y"], dtype=np.int16)[:total_end],
        "p": np.asarray(raw_events["p"], dtype=np.int8)[:total_end],
    }
    return trimmed, np.asarray(ranges, dtype=np.int64)


def _interpolate_theta(theta0: float, theta1: float, alpha: float) -> float:
    sin_term = (1.0 - alpha) * float(np.sin(2.0 * theta0)) + alpha * float(np.sin(2.0 * theta1))
    cos_term = (1.0 - alpha) * float(np.cos(2.0 * theta0)) + alpha * float(np.cos(2.0 * theta1))
    if abs(sin_term) < 1.0e-8 and abs(cos_term) < 1.0e-8:
        return float(theta0)
    return float(0.5 * np.arctan2(sin_term, cos_term))


def _interpolate_ellipse_sensor(ellipse0: Sequence[float], ellipse1: Sequence[float], alpha: float) -> list[float]:
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


def _interpolate_xywh(box0: Sequence[float], box1: Sequence[float], alpha: float) -> list[float]:
    t = float(np.clip(alpha, 0.0, 1.0))
    values0 = [float(v) for v in box0]
    values1 = [float(v) for v in box1]
    return [float((1.0 - t) * v0 + t * v1) for v0, v1 in zip(values0, values1, strict=False)]


def _ellipse_to_bbox_xywh(ellipse_xywht: Sequence[float]) -> list[float]:
    x, y, w, h, _ = [float(v) for v in ellipse_xywht]
    return [float(x - 0.5 * w), float(y - 0.5 * h), float(w), float(h)]


def _rasterize_bbox_mask(bbox_xywh: Sequence[float], *, image_size_wh: tuple[int, int]) -> np.ndarray:
    image_w, image_h = [int(v) for v in image_size_wh]
    x, y, w, h = [float(v) for v in bbox_xywh]
    x0 = max(0, min(image_w, int(np.floor(x))))
    y0 = max(0, min(image_h, int(np.floor(y))))
    x1 = max(x0, min(image_w, int(np.ceil(x + w))))
    y1 = max(y0, min(image_h, int(np.ceil(y + h))))
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _save_binary_mask(mask: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=np.uint8) * 255).astype(np.uint8)).save(path)
    return path


def _load_binary_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)


def _interpolate_binary_mask(mask0: np.ndarray | None, mask1: np.ndarray | None, alpha: float) -> np.ndarray | None:
    if mask0 is None and mask1 is None:
        return None
    if mask0 is None:
        return np.asarray(mask1, dtype=np.uint8).copy()
    if mask1 is None:
        return np.asarray(mask0, dtype=np.uint8).copy()
    t = float(np.clip(alpha, 0.0, 1.0))
    blended = ((1.0 - t) * np.asarray(mask0, dtype=np.float32) + t * np.asarray(mask1, dtype=np.float32)) >= 0.5
    return blended.astype(np.uint8)


def _restrict_mask_to_eye_roi(mask: np.ndarray, eye_region_bbox_xywh_sensor: Sequence[float] | None) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if eye_region_bbox_xywh_sensor in (None, []):
        return mask
    eye_roi_mask = _rasterize_bbox_mask(
        eye_region_bbox_xywh_sensor,
        image_size_wh=(int(mask.shape[1]), int(mask.shape[0])),
    )
    return np.logical_and(mask > 0, eye_roi_mask > 0).astype(np.uint8)


def _load_groundedsam_source_mask(
    row: dict,
    *,
    source: dict,
    annotation_root: Path | None,
) -> np.ndarray | None:
    cache = source.setdefault("_mask_cache", {})
    cache_key = str(row.get("pupil_mask_path") or row.get("mask_path") or row.get("frame_filename") or row.get("timestamp_us"))
    cached = cache.get(cache_key)
    if cached is not None:
        return np.asarray(cached, dtype=np.uint8)

    mask = None
    stored_path = row.get("pupil_mask_path") or row.get("mask_path")
    if annotation_root is not None and stored_path not in (None, ""):
        resolved_path = resolve_groundedsam_mask_path(annotation_root, stored_path)
        if resolved_path is not None and resolved_path.exists():
            mask = _load_binary_mask(resolved_path)

    if mask is None:
        ellipse = row.get("pupil_ellipse_xywht_sensor") or row.get("ellipse_sensor_xywht")
        if ellipse not in (None, []):
            mask = rasterize_ellipse_mask(ellipse, image_size=(346, 240))
        else:
            bbox = row.get("pupil_region_bbox_xywh_sensor") or row.get("pupil_bbox_xywh_sensor")
            if bbox not in (None, []):
                mask = _rasterize_bbox_mask(bbox, image_size_wh=(346, 240))

    if mask is None:
        return None

    cache[cache_key] = np.asarray(mask, dtype=np.uint8)
    return np.asarray(cache[cache_key], dtype=np.uint8)


def _bbox_only_store_path(annotation_root: Path, *, user_id: int, eye: str, session_code: str) -> Path:
    return (
        annotation_root
        / "eye_region_bbox_only"
        / "sessions"
        / canonical_user_name(user_id)
        / str(eye)
        / f"session_{session_code}"
        / "eye_region_bboxes.jsonl"
    )


def _resolve_source_span(rows: Sequence[dict], target_timestamp_us: int, *, key: str) -> dict | None:
    filtered = [dict(row) for row in rows if row.get(key) not in (None, [])]
    if not filtered:
        return None
    filtered.sort(key=lambda row: (int(row.get("timestamp_us", 0)), int(row.get("frame_idx") or -1), str(row.get("frame_filename", ""))))
    timestamps = [int(row.get("timestamp_us", 0)) for row in filtered]
    idx = bisect_left(timestamps, int(target_timestamp_us))
    if idx < len(filtered) and timestamps[idx] == int(target_timestamp_us):
        return {"mode": "exact", "row0": filtered[idx], "row1": filtered[idx], "alpha": 0.0}
    if idx <= 0:
        return {"mode": "carry_backward", "row0": filtered[0], "row1": filtered[0], "alpha": 0.0}
    if idx >= len(filtered):
        return {"mode": "carry_forward", "row0": filtered[-1], "row1": filtered[-1], "alpha": 0.0}
    row0 = filtered[idx - 1]
    row1 = filtered[idx]
    ts0 = int(row0.get("timestamp_us", 0))
    ts1 = int(row1.get("timestamp_us", ts0))
    denom = max(ts1 - ts0, 1)
    alpha = float(int(target_timestamp_us) - ts0) / float(denom)
    return {"mode": "interpolated", "row0": row0, "row1": row1, "alpha": float(alpha)}


def _load_raw_csv_pupil_source(job: dict) -> dict | None:
    annotation_csv = job.get("annotation_csv")
    if annotation_csv in (None, ""):
        return None
    frames_dir = Path(job["frames_dir"])
    frame_records = collect_frame_records(frames_dir)
    annotations, report = parse_via_csv_with_report(
        Path(annotation_csv),
        known_frame_filenames=[rec.filename for rec in frame_records],
        known_frame_stems=[Path(rec.filename).stem for rec in frame_records],
    )
    if not annotations:
        return {
            "kind": "manual_csv",
            "rows": [],
            "eye_region_default": None,
            "report": dict(report),
        }
    blink_by_frame, blink_report = build_raw_ellipse_blink_metadata(annotations)
    rows: list[dict] = []
    for ann in annotations:
        blink = blink_by_frame.get(str(ann.frame_filename), {})
        rows.append(
            {
                "frame_filename": str(ann.frame_filename),
                "frame_idx": ann.frame_idx,
                "timestamp_us": int(ann.timestamp_us),
                "pupil_ellipse_xywht_sensor": [float(v) for v in ann.ellipse_xywht],
                "annotation_source": "manual_csv",
                "annotation_quality": 1.0,
                "closed_eye_flag": bool(blink.get("closed_eye_flag", False)),
                "blink_candidate_score": float(blink.get("blink_candidate_score", 0.0)),
                "blink_candidate_reasons": list(blink.get("blink_candidate_reasons", [])),
                "blink_candidate_source": blink.get("blink_candidate_source"),
            }
        )
    return {
        "kind": "manual_csv",
        "rows": rows,
        "report": {**dict(report), "blink": blink_report},
    }


def _load_groundedsam_full_source(annotation_root: Path | None, job: dict) -> dict | None:
    if annotation_root is None:
        return None
    store_path = groundedsam_annotation_store_path(
        annotation_root,
        user_id=int(job["user_id"]),
        eye=str(job["eye"]),
        session_code=str(job["session_code"]),
    )
    if not store_path.exists():
        return None
    rows = read_jsonl(store_path)
    rows, blink_report = apply_groundedsam_store_blink_metadata(rows)
    return {
        "kind": "groundedsam_full",
        "rows": [dict(row) for row in rows],
        "store_path": str(store_path),
        "report": dict(blink_report),
    }


def _load_groundedsam_bbox_source(annotation_root: Path | None, job: dict) -> dict | None:
    if annotation_root is None:
        return None
    store_path = _bbox_only_store_path(
        annotation_root,
        user_id=int(job["user_id"]),
        eye=str(job["eye"]),
        session_code=str(job["session_code"]),
    )
    if not store_path.exists():
        return None
    rows = read_jsonl(store_path)
    return {
        "kind": "groundedsam_eye_bbox_only",
        "rows": [dict(row) for row in rows],
        "store_path": str(store_path),
        "report": {"n_rows": int(len(rows))},
    }


def _resolve_eye_roi_annotation(target_timestamp_us: int, *, full_source: dict | None, bbox_source: dict | None) -> dict | None:
    for source in (bbox_source, full_source):
        if source is None:
            continue
        span = _resolve_source_span(source["rows"], int(target_timestamp_us), key="eye_region_bbox_xywh_sensor")
        if span is None:
            continue
        row0 = span["row0"]
        row1 = span["row1"]
        if span["mode"] == "interpolated":
            bbox = _interpolate_xywh(row0["eye_region_bbox_xywh_sensor"], row1["eye_region_bbox_xywh_sensor"], span["alpha"])
            closed_eye_flag = bool(row0.get("closed_eye_flag", False) or row1.get("closed_eye_flag", False))
            reasons = sorted(set(list(row0.get("blink_candidate_reasons", [])) + list(row1.get("blink_candidate_reasons", []))))
            blink_score = max(float(row0.get("blink_candidate_score", 0.0)), float(row1.get("blink_candidate_score", 0.0)))
            blink_source = row0.get("blink_candidate_source") or row1.get("blink_candidate_source")
        else:
            bbox = [float(v) for v in row0["eye_region_bbox_xywh_sensor"]]
            closed_eye_flag = bool(row0.get("closed_eye_flag", False))
            reasons = list(row0.get("blink_candidate_reasons", []))
            blink_score = float(row0.get("blink_candidate_score", 0.0))
            blink_source = row0.get("blink_candidate_source")
        return {
            "eye_region_bbox_xywh_sensor": [float(v) for v in bbox],
            "eye_region_source_bbox_xywh_sensor": [float(v) for v in bbox],
            "roi_status": f"{span['mode']}_{source['kind']}",
            "annotation_source": str(source["kind"]),
            "eye_region_stage_source": str(source["kind"]),
            "eye_region_stage_validation_basis": (
                "validated_eye_region_bbox_only_preview"
                if str(source["kind"]) == "groundedsam_eye_bbox_only"
                else "groundedsam_full_store_fallback"
            ),
            "eye_region_mask_stage": "rasterize_and_rebox_from_eye_roi_source",
            "closed_eye_flag": bool(closed_eye_flag),
            "blink_candidate_score": float(blink_score),
            "blink_candidate_reasons": reasons,
            "blink_candidate_source": blink_source,
        }
    return None


def _resolve_pupil_annotation(
    target_timestamp_us: int,
    *,
    source: dict | None,
    annotation_root: Path | None,
    eye_region_bbox_xywh_sensor: Sequence[float] | None,
) -> dict | None:
    if source is None:
        return None

    if str(source.get("kind")) == "groundedsam_full":
        span = _resolve_source_span(source["rows"], int(target_timestamp_us), key="pupil_mask_path")
        if span is None:
            span = _resolve_source_span(source["rows"], int(target_timestamp_us), key="pupil_ellipse_xywht_sensor")
        if span is None:
            return None

        row0 = span["row0"]
        row1 = span["row1"]
        mask0 = _load_groundedsam_source_mask(row0, source=source, annotation_root=annotation_root)
        mask1 = _load_groundedsam_source_mask(row1, source=source, annotation_root=annotation_root)
        if span["mode"] == "interpolated":
            mask = _interpolate_binary_mask(mask0, mask1, float(span["alpha"]))
            closed_eye_flag = bool(row0.get("closed_eye_flag", False) or row1.get("closed_eye_flag", False))
            reasons = sorted(set(list(row0.get("blink_candidate_reasons", [])) + list(row1.get("blink_candidate_reasons", []))))
            blink_score = max(float(row0.get("blink_candidate_score", 0.0)), float(row1.get("blink_candidate_score", 0.0)))
            blink_source = row0.get("blink_candidate_source") or row1.get("blink_candidate_source")
            quality = 0.9 * max(float(row0.get("annotation_quality", 1.0)), float(row1.get("annotation_quality", 1.0)))
        else:
            mask = mask0 if mask0 is not None else mask1
            closed_eye_flag = bool(row0.get("closed_eye_flag", False))
            reasons = list(row0.get("blink_candidate_reasons", []))
            blink_score = float(row0.get("blink_candidate_score", 0.0))
            blink_source = row0.get("blink_candidate_source")
            quality = float(row0.get("annotation_quality", 1.0))

        if mask is None:
            return None

        mask = _restrict_mask_to_eye_roi(mask, eye_region_bbox_xywh_sensor)
        bbox = mask_bbox_xywh(mask)
        ellipse = mask_to_ellipse_xywht(mask)
        if bbox is None or ellipse is None:
            return None

        return {
            "pupil_mask_sensor": np.asarray(mask, dtype=np.uint8),
            "pupil_mask_source": str(source["kind"]),
            "pupil_bbox_source": "derived_from_pupil_mask",
            "pupil_state_source": "derived_from_pupil_mask",
            "pupil_mask_stage": "groundedsam_pupil_mask_inside_eye_roi",
            "pupil_ellipse_xywht_sensor": [float(v) for v in ellipse],
            "pupil_region_bbox_xywh_sensor": [float(v) for v in bbox],
            "pupil_status": f"{span['mode']}_{source['kind']}",
            "annotation_source": str(source["kind"]),
            "annotation_quality": float(np.clip(quality, 0.0, 1.0)),
            "closed_eye_flag": bool(closed_eye_flag),
            "blink_candidate_score": float(blink_score),
            "blink_candidate_reasons": reasons,
            "blink_candidate_source": blink_source,
            "state_xyabuv": [float(v) for v in xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32)).tolist()],
        }

    span = _resolve_source_span(source["rows"], int(target_timestamp_us), key="pupil_ellipse_xywht_sensor")
    if span is None:
        return None
    row0 = span["row0"]
    row1 = span["row1"]
    if span["mode"] == "interpolated":
        ellipse = _interpolate_ellipse_sensor(row0["pupil_ellipse_xywht_sensor"], row1["pupil_ellipse_xywht_sensor"], span["alpha"])
        closed_eye_flag = bool(row0.get("closed_eye_flag", False) or row1.get("closed_eye_flag", False))
        reasons = sorted(set(list(row0.get("blink_candidate_reasons", [])) + list(row1.get("blink_candidate_reasons", []))))
        blink_score = max(float(row0.get("blink_candidate_score", 0.0)), float(row1.get("blink_candidate_score", 0.0)))
        blink_source = row0.get("blink_candidate_source") or row1.get("blink_candidate_source")
        quality = 0.9 * max(float(row0.get("annotation_quality", 1.0)), float(row1.get("annotation_quality", 1.0)))
    else:
        ellipse = [float(v) for v in row0["pupil_ellipse_xywht_sensor"]]
        closed_eye_flag = bool(row0.get("closed_eye_flag", False))
        reasons = list(row0.get("blink_candidate_reasons", []))
        blink_score = float(row0.get("blink_candidate_score", 0.0))
        blink_source = row0.get("blink_candidate_source")
        quality = float(row0.get("annotation_quality", 1.0))
    return {
        "pupil_mask_sensor": rasterize_ellipse_mask(ellipse, image_size=(346, 240)),
        "pupil_mask_source": "raw_csv_state",
        "pupil_bbox_source": "derived_from_raw_csv_state",
        "pupil_state_source": "raw_csv_state",
        "pupil_mask_stage": "rasterize_from_pupil_state",
        "pupil_ellipse_xywht_sensor": [float(v) for v in ellipse],
        "pupil_region_bbox_xywh_sensor": _ellipse_to_bbox_xywh(ellipse),
        "pupil_status": f"{span['mode']}_{source['kind']}",
        "annotation_source": str(source["kind"]),
        "annotation_quality": float(np.clip(quality, 0.0, 1.0)),
        "closed_eye_flag": bool(closed_eye_flag),
        "blink_candidate_score": float(blink_score),
        "blink_candidate_reasons": reasons,
        "blink_candidate_source": blink_source,
        "state_xyabuv": [float(v) for v in xywht_to_xyabuv(np.asarray(ellipse, dtype=np.float32)).tolist()],
    }


def _annotate_target_frame_label_rows(
    *,
    job: dict,
    frame_labels: Sequence[dict],
    target_session_dir: Path,
    annotation_root: Path | None,
) -> tuple[list[dict], dict[str, object]]:
    manual_source = _load_raw_csv_pupil_source(job)
    gsam_full_source = _load_groundedsam_full_source(annotation_root, job)
    gsam_bbox_source = _load_groundedsam_bbox_source(annotation_root, job)
    pupil_source = manual_source if manual_source and manual_source.get("rows") else gsam_full_source

    eye_mask_dir = target_session_dir / "labels" / "eye_region_masks"
    pupil_mask_dir = target_session_dir / "labels" / "pupil_masks"
    annotated_rows: list[dict] = []
    n_complete = 0
    n_eye_only = 0
    n_missing = 0

    for base_row in frame_labels:
        row = dict(base_row)
        target_timestamp_us = int(row["target_timestamp_us"])
        eye_ann = _resolve_eye_roi_annotation(
            target_timestamp_us,
            full_source=gsam_full_source,
            bbox_source=gsam_bbox_source,
        )

        if eye_ann is not None:
            source_eye_bbox = [
                float(v)
                for v in eye_ann.get("eye_region_source_bbox_xywh_sensor", eye_ann["eye_region_bbox_xywh_sensor"])
            ]
            eye_mask = _rasterize_bbox_mask(source_eye_bbox, image_size_wh=(346, 240))
            resolved_eye_bbox = mask_bbox_xywh(eye_mask)
            if resolved_eye_bbox is None:
                resolved_eye_bbox = list(source_eye_bbox)

            row["eye_region_bbox_xywh_sensor"] = [float(v) for v in resolved_eye_bbox]
            row["eye_region_xywh"] = [float(v) for v in resolved_eye_bbox]
            row["eye_region_source_bbox_xywh_sensor"] = [float(v) for v in source_eye_bbox]
            row["roi_size_wh"] = [
                int(round(float(resolved_eye_bbox[2]))),
                int(round(float(resolved_eye_bbox[3]))),
            ]
            eye_mask_path = _save_binary_mask(eye_mask, eye_mask_dir / Path(row["target_frame_filename"]).with_suffix(".png").name)
            row["eye_region_mask_path"] = str(eye_mask_path)
            row["eye_region_mask_valid"] = True
            row["eye_region_mask_source"] = str(eye_ann["annotation_source"])
            row["eye_region_bbox_source"] = "derived_from_eye_region_mask"
            row["eye_region_mask_stage"] = eye_ann.get("eye_region_mask_stage")
            row["eye_region_stage_source"] = eye_ann.get("eye_region_stage_source")
            row["eye_region_stage_validation_basis"] = eye_ann.get("eye_region_stage_validation_basis")
            row["roi_status"] = eye_ann["roi_status"]
            row["closed_eye_flag"] = bool(eye_ann.get("closed_eye_flag", False))
            row["blink_candidate_score"] = float(eye_ann.get("blink_candidate_score", 0.0))
            row["blink_candidate_reasons"] = list(eye_ann.get("blink_candidate_reasons", []))
            row["blink_candidate_source"] = eye_ann.get("blink_candidate_source")
        else:
            row["eye_region_bbox_xywh_sensor"] = None
            row["eye_region_xywh"] = None
            row["eye_region_source_bbox_xywh_sensor"] = None
            row["eye_region_mask_path"] = None
            row["eye_region_mask_valid"] = False
            row["eye_region_mask_source"] = None
            row["eye_region_bbox_source"] = None
            row["eye_region_mask_stage"] = None
            row["eye_region_stage_source"] = None
            row["eye_region_stage_validation_basis"] = None
            row["roi_status"] = "pending_groundedsam"

        pupil_ann = _resolve_pupil_annotation(
            target_timestamp_us,
            source=pupil_source,
            annotation_root=annotation_root,
            eye_region_bbox_xywh_sensor=row.get("eye_region_bbox_xywh_sensor"),
        )

        if pupil_ann is not None:
            ellipse = pupil_ann["pupil_ellipse_xywht_sensor"]
            row["pupil_ellipse_xywht_sensor"] = ellipse
            row["ellipse_sensor_xywht"] = ellipse
            row["pupil_region_bbox_xywh_sensor"] = pupil_ann["pupil_region_bbox_xywh_sensor"]
            row["pupil_bbox_xywh_sensor"] = pupil_ann["pupil_region_bbox_xywh_sensor"]
            row["state_xyabuv"] = pupil_ann["state_xyabuv"]
            row["annotation_quality"] = float(pupil_ann["annotation_quality"])
            row["annotation_source"] = str(pupil_ann["annotation_source"])
            row["pupil_status"] = pupil_ann["pupil_status"]
            pupil_mask = np.asarray(pupil_ann.get("pupil_mask_sensor"), dtype=np.uint8)
            pupil_mask_path = _save_binary_mask(pupil_mask, pupil_mask_dir / Path(row["target_frame_filename"]).with_suffix(".png").name)
            row["pupil_mask_path"] = str(pupil_mask_path)
            row["mask_path"] = row["pupil_mask_path"]
            row["pupil_mask_source"] = pupil_ann.get("pupil_mask_source")
            row["pupil_bbox_source"] = pupil_ann.get("pupil_bbox_source")
            row["pupil_state_source"] = pupil_ann.get("pupil_state_source")
            row["pupil_mask_stage"] = pupil_ann.get("pupil_mask_stage")
            if row.get("eye_region_bbox_xywh_sensor"):
                row["ellipse_roi_xywht"] = ellipse_sensor_to_roi(ellipse, type("EyeRegionProxy", (), {
                    "x": int(round(float(row["eye_region_bbox_xywh_sensor"][0]))),
                    "y": int(round(float(row["eye_region_bbox_xywh_sensor"][1]))),
                    "w": int(round(float(row["eye_region_bbox_xywh_sensor"][2]))),
                    "h": int(round(float(row["eye_region_bbox_xywh_sensor"][3]))),
                })())
            else:
                row["ellipse_roi_xywht"] = ellipse
            row["closed_eye_flag"] = bool(row.get("closed_eye_flag", False) or pupil_ann["closed_eye_flag"])
            row["blink_candidate_score"] = max(
                float(row.get("blink_candidate_score", 0.0)),
                float(pupil_ann["blink_candidate_score"]),
            )
            row["blink_candidate_reasons"] = sorted(
                set(list(row.get("blink_candidate_reasons", [])) + list(pupil_ann["blink_candidate_reasons"]))
            )
            row["blink_candidate_source"] = pupil_ann["blink_candidate_source"] or row.get("blink_candidate_source")
            row["mask_valid"] = True
        else:
            row["pupil_region_bbox_xywh_sensor"] = None
            row["pupil_bbox_xywh_sensor"] = None
            row["pupil_ellipse_xywht_sensor"] = None
            row["ellipse_sensor_xywht"] = None
            row["ellipse_roi_xywht"] = None
            row["state_xyabuv"] = None
            row["pupil_mask_path"] = None
            row["mask_path"] = None
            row["annotation_quality"] = 0.0
            if manual_source is None:
                row["pupil_status"] = "pending_groundedsam_roi_mask_pipeline"
            else:
                row["pupil_status"] = "pending_pupil_source"
            row["pupil_mask_source"] = None
            row["pupil_bbox_source"] = None
            row["pupil_state_source"] = None
            row["pupil_mask_stage"] = None
            row["mask_valid"] = False
            row.setdefault("closed_eye_flag", False)
            row.setdefault("blink_candidate_score", 0.0)
            row.setdefault("blink_candidate_reasons", [])
            row.setdefault("blink_candidate_source", None)

        if eye_ann is not None and pupil_ann is not None:
            row["label_status"] = "annotated_complete"
            row["annotation_reason"] = "raw_csv_or_groundedsam_fused"
            n_complete += 1
        elif eye_ann is not None:
            row["label_status"] = "annotated_eye_only"
            if manual_source is None:
                row["annotation_reason"] = "groundedsam_no_csv_pupil_pipeline_pending"
            else:
                row["annotation_reason"] = "pupil_source_missing"
            n_eye_only += 1
        else:
            row["label_status"] = "pending_annotation_sources"
            if manual_source is None:
                row["annotation_reason"] = "groundedsam_no_csv_eye_mask_pipeline_pending"
            else:
                row["annotation_reason"] = "groundedsam_eye_roi_missing"
            n_missing += 1

        row["eye_state_flag"] = "closed" if bool(row.get("closed_eye_flag", False)) else "open"
        row["annotation_sources_used"] = {
            "eye": None if eye_ann is None else eye_ann["annotation_source"],
            "pupil": None if pupil_ann is None else pupil_ann["annotation_source"],
        }
        row["sensor_size_wh"] = [346, 240]
        annotated_rows.append(row)

    report = {
        "manual_csv_available": bool(manual_source and manual_source.get("rows")),
        "groundedsam_full_available": bool(gsam_full_source and gsam_full_source.get("rows")),
        "groundedsam_eye_bbox_available": bool(gsam_bbox_source and gsam_bbox_source.get("rows")),
        "pupil_source_kind": None if pupil_source is None else pupil_source["kind"],
        "eye_region_source_contract": "groundedsam_only",
        "pupil_source_contract": "raw_csv_state_or_groundedsam_in_roi",
        "eye_region_preferred_source_order": ["groundedsam_eye_bbox_only", "groundedsam_full"],
        "no_csv_groundedsam_step_order": list(NO_CSV_GROUNDEDSAM_STEP_ORDER),
        "n_rows": int(len(annotated_rows)),
        "n_complete": int(n_complete),
        "n_eye_only": int(n_eye_only),
        "n_missing": int(n_missing),
    }
    return annotated_rows, report


def _build_frame_label_rows(
    *,
    job: dict,
    frame_plan: Sequence[dict],
    event_packets: np.ndarray,
    rebinned_events: dict[str, np.ndarray],
    session_store_format: str,
    session_store_path: Path,
) -> list[dict]:
    event_timestamps = np.asarray(rebinned_events["t"], dtype=np.int64)
    rows: list[dict] = []
    for row, (start_idx, end_idx) in zip(frame_plan, event_packets.tolist(), strict=False):
        out = dict(row)
        out.update(
            {
                "session_key": str(job["session_key"]),
                "target_fps": float(job["target_fps"]),
                "target_step_us": int(job["target_step_us"]),
                "session_store_path": str(session_store_path),
                "session_store_format": str(session_store_format),
                "frame_store_index": int(row["frame_index"]),
                "event_index_range": [int(start_idx), int(end_idx)],
                "event_count": int(end_idx - start_idx),
                "event_start_timestamp_us": None if start_idx >= end_idx else int(event_timestamps[start_idx]),
                "event_end_timestamp_us": None if start_idx >= end_idx else int(event_timestamps[end_idx - 1]),
                "label_status": "pending_annotation",
                "annotation_reason": "annotation_pending_v3_2",
                "eye_state_flag": "unknown",
                "roi_status": "pending_groundedsam",
                "pupil_status": "pending_raw_csv_or_groundedsam",
                "annotation_sources_expected": {
                    "pupil_raw_csv": bool(job.get("annotation_csv")),
                    "eye_groundedsam": True,
                    "pupil_groundedsam_in_roi": True,
                },
            }
        )
        rows.append(out)
    return rows


def _collect_annotation_failure_rows(
    *,
    frame_labels: Sequence[dict],
    job: dict,
    frame_labels_path: Path,
    session_store_path: Path,
) -> list[dict]:
    failures: list[dict] = []
    for row in frame_labels:
        if str(row.get("label_status", "")) == "annotated_complete":
            continue
        failures.append(
            {
                "session_key": str(job["session_key"]),
                "user_id": int(job["user_id"]),
                "eye": str(job["eye"]),
                "session_code": str(job["session_code"]),
                "frame_index": int(row.get("frame_store_index", row.get("frame_index", 0))),
                "frame_filename": str(row.get("target_frame_filename") or row.get("frame_filename") or ""),
                "target_timestamp_us": int(row.get("target_timestamp_us", row.get("frame_timestamp_us", 0))),
                "source_kind": row.get("source_kind"),
                "label_status": str(row.get("label_status", "unknown")),
                "annotation_reason": row.get("annotation_reason"),
                "roi_status": row.get("roi_status"),
                "pupil_status": row.get("pupil_status"),
                "eye_state_flag": row.get("eye_state_flag"),
                "closed_eye_flag": bool(row.get("closed_eye_flag", False)),
                "annotation_sources_used": row.get("annotation_sources_used"),
                "annotation_sources_expected": row.get("annotation_sources_expected"),
                "frame_labels_path": str(frame_labels_path),
                "session_store_path": str(session_store_path),
            }
        )
    return failures


def _write_npz_session_store(
    *,
    path: Path,
    source_frames: np.ndarray,
    source_frame_timestamps_us: np.ndarray,
    frames: np.ndarray | None,
    frame_timestamps_us: np.ndarray,
    event_packets: np.ndarray,
    rebinned_events: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_frame_images": np.asarray(source_frames, dtype=np.uint8),
        "source_frame_timestamps_us": np.asarray(source_frame_timestamps_us, dtype=np.int64),
        "frame_timestamps_us": np.asarray(frame_timestamps_us, dtype=np.int64),
        "frame_event_ranges": np.asarray(event_packets, dtype=np.int64),
        "event_t": np.asarray(rebinned_events["t"], dtype=np.int64),
        "event_x": np.asarray(rebinned_events["x"], dtype=np.int16),
        "event_y": np.asarray(rebinned_events["y"], dtype=np.int16),
        "event_p": np.asarray(rebinned_events["p"], dtype=np.int8),
    }
    if frames is not None:
        payload["frame_images"] = np.asarray(frames, dtype=np.uint8)
    np.savez_compressed(path, **payload)


def _write_h5_session_store(
    *,
    path: Path,
    source_frames: np.ndarray,
    source_frame_timestamps_us: np.ndarray,
    frames: np.ndarray | None,
    frame_timestamps_us: np.ndarray,
    event_packets: np.ndarray,
    rebinned_events: dict[str, np.ndarray],
) -> None:
    h5py = _import_h5py(required=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_source_frames = int(source_frames.shape[0])
    source_chunk_frames = max(1, min(n_source_frames, 32))
    with h5py.File(path, "w") as handle:
        frames_group = handle.create_group("frames")
        if frames is not None:
            n_frames = int(frames.shape[0])
            chunk_frames = max(1, min(n_frames, 32))
            frames_group.create_dataset(
                "images",
                data=np.asarray(frames, dtype=np.uint8),
                compression="gzip",
                chunks=(chunk_frames, int(frames.shape[1]), int(frames.shape[2])),
            )
        frames_group.create_dataset(
            "timestamps_us",
            data=np.asarray(frame_timestamps_us, dtype=np.int64),
            compression="gzip",
        )
        frames_group.create_dataset(
            "event_ranges",
            data=np.asarray(event_packets, dtype=np.int64),
            compression="gzip",
        )
        source_frames_group = handle.create_group("source_frames")
        source_frames_group.create_dataset(
            "images",
            data=np.asarray(source_frames, dtype=np.uint8),
            compression="gzip",
            chunks=(source_chunk_frames, int(source_frames.shape[1]), int(source_frames.shape[2])),
        )
        source_frames_group.create_dataset(
            "timestamps_us",
            data=np.asarray(source_frame_timestamps_us, dtype=np.int64),
            compression="gzip",
        )
        events_group = handle.create_group("events")
        events_group.create_dataset("t", data=np.asarray(rebinned_events["t"], dtype=np.int64), compression="gzip")
        events_group.create_dataset("x", data=np.asarray(rebinned_events["x"], dtype=np.int16), compression="gzip")
        events_group.create_dataset("y", data=np.asarray(rebinned_events["y"], dtype=np.int16), compression="gzip")
        events_group.create_dataset("p", data=np.asarray(rebinned_events["p"], dtype=np.int8), compression="gzip")


def materialize_target_fps_session(
    *,
    job: dict,
    overwrite: bool = False,
    annotation_root: Path | None = None,
    interpolation_backend: str = DEFAULT_INTERPOLATION_BACKEND,
    timelens_root: Path | None = None,
    timelens_checkpoint: Path | None = None,
    timelens_device: str = "cpu",
    timelens_xl_root: Path | None = None,
    timelens_xl_checkpoint: Path | None = None,
    timelens_xl_device: str = "cpu",
    event_generation_backend: str = DEFAULT_EVENT_GENERATION_BACKEND,
    v2e_root: Path | None = None,
    v2e_device: str = "cpu",
    v2e_kwargs: dict | None = None,
    session_store_format: str = "auto",
    frame_storage_mode: str = "materialized_target_frames",
) -> dict:
    target_session_dir = Path(job["target_session_dir"])
    ensure_dir(target_session_dir)

    session_store_kind = resolve_session_store_format(session_store_format)
    session_store_path = target_session_dir / ("session.h5" if session_store_kind == "h5" else "session_arrays.npz")
    frame_labels_path = target_session_dir / "frame_labels.jsonl"
    annotation_failures_path = target_session_dir / "annotation_failures.jsonl"
    session_meta_path = target_session_dir / "session_meta.json"
    if (not overwrite) and (session_store_path.exists() or frame_labels_path.exists() or annotation_failures_path.exists() or session_meta_path.exists()):
        raise FileExistsError(
            f"Session artifacts already exist under {target_session_dir}. Pass overwrite=True to replace them."
        )

    frames_dir = Path(job["frames_dir"])
    event_file = Path(job["event_file"])
    frame_records = collect_frame_records(frames_dir)
    frame_plan = build_target_frame_plan(frame_records, target_fps=float(job["target_fps"]))
    source_frames = _load_source_frame_stack(frame_records, frames_dir)
    source_frame_timestamps_us = np.asarray([int(record.timestamp_us) for record in frame_records], dtype=np.int64)
    if str(frame_storage_mode) == "lazy_source_frames":
        frames = None
    else:
        frames = _render_target_frames(
            frame_plan=frame_plan,
            frame_records=frame_records,
            frames_dir=frames_dir,
            event_file=event_file,
            interpolation_backend=interpolation_backend,
            timelens_root=timelens_root,
            timelens_checkpoint=timelens_checkpoint,
            timelens_device=timelens_device,
            timelens_xl_root=timelens_xl_root,
            timelens_xl_checkpoint=timelens_xl_checkpoint,
            timelens_xl_device=timelens_xl_device,
        )
    raw_events = _load_event_arrays(event_file)
    frame_timestamps_us = np.asarray([int(row["target_timestamp_us"]) for row in frame_plan], dtype=np.int64)
    if frames is None and str(event_generation_backend).strip().lower() != "none":
        raise ValueError("frame_storage_mode='lazy_source_frames' is incompatible with event_generation_backend != 'none'")
    rebinned_events, event_packets = generate_event_packets(
        event_generation_backend=str(event_generation_backend),
        frames=source_frames if frames is None else frames,
        frame_timestamps_us=frame_timestamps_us if frames is not None else source_frame_timestamps_us,
        raw_events=raw_events,
        frame_plan=list(frame_plan),
        v2e_root=v2e_root,
        v2e_device=str(v2e_device),
        v2e_kwargs=v2e_kwargs,
    )

    if session_store_kind == "h5":
        _write_h5_session_store(
            path=session_store_path,
            source_frames=source_frames,
            source_frame_timestamps_us=source_frame_timestamps_us,
            frames=frames,
            frame_timestamps_us=frame_timestamps_us,
            event_packets=event_packets,
            rebinned_events=rebinned_events,
        )
    else:
        _write_npz_session_store(
            path=session_store_path,
            source_frames=source_frames,
            source_frame_timestamps_us=source_frame_timestamps_us,
            frames=frames,
            frame_timestamps_us=frame_timestamps_us,
            event_packets=event_packets,
            rebinned_events=rebinned_events,
        )

    frame_labels = _build_frame_label_rows(
        job=job,
        frame_plan=frame_plan,
        event_packets=event_packets,
        rebinned_events=rebinned_events,
        session_store_format=session_store_kind,
        session_store_path=session_store_path,
    )
    frame_labels, annotation_report = _annotate_target_frame_label_rows(
        job=job,
        frame_labels=frame_labels,
        target_session_dir=target_session_dir,
        annotation_root=annotation_root,
    )
    annotation_failures = _collect_annotation_failure_rows(
        frame_labels=frame_labels,
        job=job,
        frame_labels_path=frame_labels_path,
        session_store_path=session_store_path,
    )
    write_jsonl(frame_labels, frame_labels_path)
    write_jsonl(annotation_failures, annotation_failures_path)

    session_meta = {
        "stage": "materialized_session",
        "session_key": str(job["session_key"]),
        "user_id": int(job["user_id"]),
        "eye": str(job["eye"]),
        "session_code": str(job["session_code"]),
        "raw_session_dir": str(job["raw_session_dir"]),
        "frames_dir": str(frames_dir),
        "event_file": str(event_file),
        "annotation_csv": job.get("annotation_csv"),
        "annotation_root": None if annotation_root is None else str(annotation_root),
        "target_fps": float(job["target_fps"]),
        "target_step_us": int(job["target_step_us"]),
        "session_store_format": str(session_store_kind),
        "session_store_path": str(session_store_path),
        "frame_storage_mode": str(frame_storage_mode),
        "frame_labels_path": str(frame_labels_path),
        "annotation_failures_path": str(annotation_failures_path),
        "frame_shape_hw": [int(source_frames.shape[1]), int(source_frames.shape[2])],
        "n_source_frames": int(len(frame_records)),
        "n_target_frames": int(len(frame_plan)),
        "n_raw_frame_reuse": int(sum(1 for row in frame_plan if row.get("source_kind") == "raw")),
        "n_interpolated_frames": int(sum(1 for row in frame_plan if row.get("source_kind") == "interpolated")),
        "n_rebinned_events": int(len(rebinned_events["t"])),
        "event_time_range_us": None
        if len(rebinned_events["t"]) == 0
        else [int(rebinned_events["t"][0]), int(rebinned_events["t"][-1])],
        "n_rows_annotated_complete": int(sum(1 for row in frame_labels if row.get("label_status") == "annotated_complete")),
        "n_rows_annotated_eye_only": int(sum(1 for row in frame_labels if row.get("label_status") == "annotated_eye_only")),
        "n_rows_pending_annotation": int(sum(1 for row in frame_labels if row.get("label_status") == "pending_annotation_sources")),
        "n_annotation_failures": int(len(annotation_failures)),
        "label_status_default": "annotated_complete" if annotation_report.get("n_complete", 0) else "pending_annotation",
        "eye_state_flag_default": "open",
        "annotation_report": annotation_report,
        "interpolation_backend": str(interpolation_backend),
        "event_generation_backend": str(event_generation_backend),
        "timelens_xl_root": None if timelens_xl_root is None else str(timelens_xl_root),
        "timelens_xl_checkpoint": None if timelens_xl_checkpoint is None else str(timelens_xl_checkpoint),
        "timelens_xl_device": str(timelens_xl_device),
        "v2e_root": None if v2e_root is None else str(v2e_root),
        "v2e_device": str(v2e_device),
        "v2e_kwargs": None if not v2e_kwargs else dict(v2e_kwargs),
    }
    write_json(session_meta, session_meta_path)
    return session_meta


def _normalize_eye_filter(eye: Optional[str]) -> tuple[str, ...]:
    text = str(eye or "both").strip().lower()
    if text in {"both", "all", "*"}:
        return ("left", "right")
    if text in {"left", "right"}:
        return (text,)
    raise ValueError(f"Unsupported eye filter: {eye!r}")


def _normalize_session_filters(session_codes: Optional[Iterable[str]]) -> set[str]:
    normalized: set[str] = set()
    for session_code in session_codes or []:
        normalized.add(normalize_session_code_filter(session_code))
    return normalized


def discover_target_fps_session_jobs(
    *,
    raw_root: str | Path,
    target_root: str | Path,
    target_fps: float,
    user_id: int | None = None,
    eye: str | None = None,
    session_codes: Optional[Iterable[str]] = None,
    include_nonstandard_sessions: bool = False,
    max_sessions: int | None = None,
) -> tuple[list[dict], list[dict]]:
    raw_root = Path(raw_root).resolve()
    target_fps_root = resolve_target_fps_root(target_root, target_fps)
    session_filter = _normalize_session_filters(session_codes)
    allowed_eyes = _normalize_eye_filter(eye)
    jobs: list[dict] = []
    skipped: list[dict] = []

    for user_dir in collect_users(raw_root):
        try:
            current_user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        if user_id is not None and int(user_id) != current_user_id:
            continue
        for current_eye in allowed_eyes:
            eye_dir = user_dir / current_eye
            if not eye_dir.exists():
                continue
            for session_dir in sorted(eye_dir.glob("session_*_*_*")):
                layout = discover_session_layout(session_dir, user_id=current_user_id)
                session_code = str(layout.session_code)
                if session_filter and session_code not in session_filter:
                    continue
                session_key = f"{canonical_user_name(current_user_id)}/{current_eye}/session_{session_code}"
                target_session_dir = target_fps_root / "sessions" / canonical_user_name(current_user_id) / current_eye / f"session_{session_code}"
                skip_reason = None
                if (not include_nonstandard_sessions) and not layout.is_official_session:
                    skip_reason = "nonstandard_session_code"
                elif layout.frames_dir is None:
                    skip_reason = "frames_dir_missing"
                elif layout.event_file is None:
                    skip_reason = "event_file_missing"

                if skip_reason is not None:
                    skipped.append(
                        {
                            "user_id": int(current_user_id),
                            "eye": str(current_eye),
                            "session_code": session_code,
                            "session_key": session_key,
                            "raw_session_dir": str(session_dir.resolve()),
                            "target_session_dir": str(target_session_dir),
                            "skip_reason": str(skip_reason),
                            "layout_warnings": list(layout.warnings),
                        }
                    )
                    continue

                frame_records = collect_frame_records(layout.frames_dir)
                if not frame_records:
                    skipped.append(
                        {
                            "user_id": int(current_user_id),
                            "eye": str(current_eye),
                            "session_code": session_code,
                            "session_key": session_key,
                            "raw_session_dir": str(session_dir.resolve()),
                            "target_session_dir": str(target_session_dir),
                            "skip_reason": "no_frames_found",
                            "layout_warnings": list(layout.warnings),
                        }
                    )
                    continue

                frame_plan = build_target_frame_plan(frame_records, target_fps=target_fps)
                n_raw_reuse = sum(1 for row in frame_plan if row.get("source_kind") == "raw")
                n_interpolated = sum(1 for row in frame_plan if row.get("source_kind") == "interpolated")
                jobs.append(
                    {
                        "user_id": int(current_user_id),
                        "eye": str(current_eye),
                        "session_code": session_code,
                        "session_key": session_key,
                        "raw_session_dir": str(session_dir.resolve()),
                        "frames_dir": str(layout.frames_dir.resolve()),
                        "event_file": str(layout.event_file.resolve()),
                        "annotation_csv": None if layout.annotation_csv is None else str(layout.annotation_csv.resolve()),
                        "target_fps_root": str(target_fps_root),
                        "target_session_dir": str(target_session_dir),
                        "target_fps": float(target_fps),
                        "target_step_us": int(compute_target_step_us(target_fps)),
                        "n_raw_frames": int(len(frame_records)),
                        "raw_frame_time_range_us": [int(frame_records[0].timestamp_us), int(frame_records[-1].timestamp_us)],
                        "n_target_frames": int(len(frame_plan)),
                        "n_raw_frame_reuse": int(n_raw_reuse),
                        "n_interpolated_frames": int(n_interpolated),
                        "layout_warnings": list(layout.warnings),
                        "planned_artifacts": {
                            "session_h5": str(target_session_dir / "session.h5"),
                            "session_npz": str(target_session_dir / "session_arrays.npz"),
                            "session_meta": str(target_session_dir / "session_meta.json"),
                            "frame_labels": str(target_session_dir / "frame_labels.jsonl"),
                        },
                    }
                )
                if max_sessions is not None and len(jobs) >= int(max_sessions):
                    return jobs, skipped
    return jobs, skipped


def build_target_fps_dataset_plan(
    *,
    raw_root: str | Path,
    target_root: str | Path,
    target_fps: float,
    user_id: int | None = None,
    eye: str | None = None,
    session_codes: Optional[Iterable[str]] = None,
    include_nonstandard_sessions: bool = False,
    max_sessions: int | None = None,
    overwrite: bool = False,
) -> dict:
    started_at = time.perf_counter()
    raw_root = Path(raw_root).resolve()
    target_fps_root = resolve_target_fps_root(target_root, target_fps)
    indexes_root = target_fps_root / "indexes"
    ensure_dir(indexes_root)

    print_progress(
        name="target_fps_build",
        status="start",
        started_at=started_at,
        total=0,
        message="target-fps build planning started",
        raw_root=str(raw_root),
        target_root=str(target_fps_root),
        target_fps=float(target_fps),
    )

    jobs, skipped = discover_target_fps_session_jobs(
        raw_root=raw_root,
        target_root=target_root,
        target_fps=target_fps,
        user_id=user_id,
        eye=eye,
        session_codes=session_codes,
        include_nonstandard_sessions=include_nonstandard_sessions,
        max_sessions=max_sessions,
    )
    for job in jobs:
        ensure_dir(Path(job["target_session_dir"]))

    sessions_index_path = indexes_root / "sessions.jsonl"
    skipped_index_path = indexes_root / "skipped_sessions.jsonl"
    annotation_failures_index_path = indexes_root / "annotation_failures.jsonl"
    if (not overwrite) and (sessions_index_path.exists() or skipped_index_path.exists() or annotation_failures_index_path.exists()):
        raise FileExistsError(
            f"Index outputs already exist under {indexes_root}. Pass overwrite=True to replace them."
        )

    write_jsonl(jobs, sessions_index_path)
    write_jsonl(skipped, skipped_index_path)
    write_jsonl([], annotation_failures_index_path)

    summary = {
        "stage": "scan_grid",
        "raw_root": str(raw_root),
        "target_root": str(target_fps_root),
        "target_fps": float(target_fps),
        "target_step_us": int(compute_target_step_us(target_fps)),
        "n_sessions_planned": int(len(jobs)),
        "n_sessions_skipped": int(len(skipped)),
        "n_target_frames_planned": int(sum(int(job.get("n_target_frames", 0)) for job in jobs)),
        "n_raw_frame_reuse_planned": int(sum(int(job.get("n_raw_frame_reuse", 0)) for job in jobs)),
        "n_interpolated_frames_planned": int(sum(int(job.get("n_interpolated_frames", 0)) for job in jobs)),
        "user_id_filter": None if user_id is None else int(user_id),
        "eye_filter": None if eye is None else str(eye),
        "session_code_filter": sorted(_normalize_session_filters(session_codes)),
        "include_nonstandard_sessions": bool(include_nonstandard_sessions),
        "max_sessions": None if max_sessions is None else int(max_sessions),
        "indexes": {
            "sessions": str(sessions_index_path),
            "skipped_sessions": str(skipped_index_path),
            "annotation_failures": str(annotation_failures_index_path),
        },
    }
    write_json(summary, target_fps_root / "build_summary.json")

    print_progress(
        name="target_fps_build",
        status="done",
        started_at=started_at,
        completed=int(len(jobs)),
        total=int(len(jobs) + len(skipped)),
        message="target-fps build planning complete",
        n_sessions_planned=summary["n_sessions_planned"],
        n_sessions_skipped=summary["n_sessions_skipped"],
        n_target_frames_planned=summary["n_target_frames_planned"],
    )
    return summary


def build_target_fps_dataset(
    *,
    raw_root: str | Path,
    target_root: str | Path,
    target_fps: float,
    annotation_root: str | Path | None = None,
    user_id: int | None = None,
    eye: str | None = None,
    session_codes: Optional[Iterable[str]] = None,
    include_nonstandard_sessions: bool = False,
    max_sessions: int | None = None,
    overwrite: bool = False,
    execute: bool = False,
    interpolation_backend: str = DEFAULT_INTERPOLATION_BACKEND,
    timelens_root: str | Path | None = None,
    timelens_checkpoint: str | Path | None = None,
    timelens_device: str = "cpu",
    timelens_xl_root: str | Path | None = None,
    timelens_xl_checkpoint: str | Path | None = None,
    timelens_xl_device: str = "cpu",
    event_generation_backend: str = DEFAULT_EVENT_GENERATION_BACKEND,
    v2e_root: str | Path | None = None,
    v2e_device: str = "cpu",
    v2e_kwargs: dict | None = None,
    session_store_format: str = "auto",
    frame_storage_mode: str = "materialized_target_frames",
) -> dict:
    plan_summary = build_target_fps_dataset_plan(
        raw_root=raw_root,
        target_root=target_root,
        target_fps=target_fps,
        user_id=user_id,
        eye=eye,
        session_codes=session_codes,
        include_nonstandard_sessions=include_nonstandard_sessions,
        max_sessions=max_sessions,
        overwrite=overwrite,
    )
    if not execute:
        return plan_summary

    started_at = time.perf_counter()
    jobs = read_jsonl(plan_summary["indexes"]["sessions"])
    annotation_failures_index_path = Path(plan_summary["indexes"]["annotation_failures"])
    materialized_sessions: list[dict] = []
    annotation_failures: list[dict] = []
    resolved_store_format = resolve_session_store_format(session_store_format)
    resolved_annotation_root = None if annotation_root is None else Path(annotation_root).resolve()
    resolved_timelens_root = None if timelens_root is None else Path(timelens_root).resolve()
    resolved_timelens_checkpoint = None if timelens_checkpoint is None else Path(timelens_checkpoint).resolve()
    resolved_timelens_xl_root = None if timelens_xl_root is None else Path(timelens_xl_root).resolve()
    resolved_timelens_xl_checkpoint = None if timelens_xl_checkpoint is None else Path(timelens_xl_checkpoint).resolve()
    resolved_v2e_root = None if v2e_root is None else Path(v2e_root).resolve()
    resolved_frame_storage_mode = str(frame_storage_mode).strip().lower()
    if resolved_frame_storage_mode not in {"materialized_target_frames", "lazy_source_frames"}:
        raise ValueError(f"Unsupported frame_storage_mode: {frame_storage_mode!r}")

    print_progress(
        name="target_fps_materialize",
        status="start",
        started_at=started_at,
        total=int(len(jobs)),
        message="target-fps dataset materialization started",
        interpolation_backend=str(interpolation_backend),
        event_generation_backend=str(event_generation_backend),
        session_store_format=str(resolved_store_format),
        frame_storage_mode=str(resolved_frame_storage_mode),
    )

    for idx, job in enumerate(jobs, start=1):
        session_meta = materialize_target_fps_session(
            job=job,
            overwrite=overwrite,
            annotation_root=resolved_annotation_root,
            interpolation_backend=interpolation_backend,
            timelens_root=resolved_timelens_root,
            timelens_checkpoint=resolved_timelens_checkpoint,
            timelens_device=timelens_device,
            timelens_xl_root=resolved_timelens_xl_root,
            timelens_xl_checkpoint=resolved_timelens_xl_checkpoint,
            timelens_xl_device=timelens_xl_device,
            event_generation_backend=str(event_generation_backend),
            v2e_root=resolved_v2e_root,
            v2e_device=str(v2e_device),
            v2e_kwargs=v2e_kwargs,
            session_store_format=resolved_store_format,
            frame_storage_mode=resolved_frame_storage_mode,
        )
        materialized_sessions.append(session_meta)
        if session_meta.get("annotation_failures_path"):
            annotation_failures.extend(read_jsonl(session_meta["annotation_failures_path"]))
        print_progress(
            name="target_fps_materialize",
            status="progress",
            started_at=started_at,
            completed=int(idx),
            total=int(len(jobs)),
            message="target-fps session materialized",
            session_key=str(job["session_key"]),
        )

    summary = dict(plan_summary)
    summary.update(
        {
            "stage": "materialized",
            "annotation_root": None if resolved_annotation_root is None else str(resolved_annotation_root),
            "interpolation_backend": str(interpolation_backend),
            "event_generation_backend": str(event_generation_backend),
            "session_store_format": str(resolved_store_format),
            "frame_storage_mode": str(resolved_frame_storage_mode),
            "timelens_root": None if resolved_timelens_root is None else str(resolved_timelens_root),
            "timelens_checkpoint": None if resolved_timelens_checkpoint is None else str(resolved_timelens_checkpoint),
            "timelens_device": str(timelens_device),
            "timelens_xl_root": None if resolved_timelens_xl_root is None else str(resolved_timelens_xl_root),
            "timelens_xl_checkpoint": None if resolved_timelens_xl_checkpoint is None else str(resolved_timelens_xl_checkpoint),
            "timelens_xl_device": str(timelens_xl_device),
            "v2e_root": None if resolved_v2e_root is None else str(resolved_v2e_root),
            "v2e_device": str(v2e_device),
            "v2e_kwargs": None if not v2e_kwargs else dict(v2e_kwargs),
            "n_sessions_materialized": int(len(materialized_sessions)),
            "n_rebinned_events_materialized": int(sum(int(item.get("n_rebinned_events", 0)) for item in materialized_sessions)),
            "n_rows_annotated_complete": int(sum(int(item.get("n_rows_annotated_complete", 0)) for item in materialized_sessions)),
            "n_rows_annotated_eye_only": int(sum(int(item.get("n_rows_annotated_eye_only", 0)) for item in materialized_sessions)),
            "n_rows_pending_annotation": int(sum(int(item.get("n_rows_pending_annotation", 0)) for item in materialized_sessions)),
            "n_annotation_failures": int(len(annotation_failures)),
            "materialized_sessions": [
                {
                    "session_key": item["session_key"],
                    "session_store_path": item["session_store_path"],
                    "session_store_format": item["session_store_format"],
                    "frame_labels_path": item["frame_labels_path"],
                    "annotation_failures_path": item.get("annotation_failures_path"),
                }
                for item in materialized_sessions
            ],
        }
    )
    write_jsonl(annotation_failures, annotation_failures_index_path)
    write_json(summary, Path(summary["target_root"]) / "build_summary.json")

    print_progress(
        name="target_fps_materialize",
        status="done",
        started_at=started_at,
        completed=int(len(materialized_sessions)),
        total=int(len(jobs)),
        message="target-fps dataset materialization complete",
        n_sessions_materialized=summary["n_sessions_materialized"],
        n_rebinned_events_materialized=summary["n_rebinned_events_materialized"],
    )
    return summary
