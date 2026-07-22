from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dataset.preprocess.io_utils import (
    EllipseAnnotation,
    canonical_user_name,
    code_to_session_dir,
    collect_frame_records,
    collect_users,
    discover_session_layout,
    parse_via_csv_with_report,
    session_dir_to_code,
    try_parse_frame_filename,
)
from utils.io import write_json, write_jsonl


RAW_ELLIPSE_BLINK_SOURCE = "raw_ellipse_heuristic"
GROUNDEDSAM_ELLIPSE_BLINK_SOURCE = "groundedsam_ellipse_heuristic"


@dataclass
class RawEllipseBlinkHeuristicConfig:
    baseline_minor_quantile: float = 0.9
    baseline_area_quantile: float = 0.9
    baseline_major_quantile: float = 0.5
    local_window: int = 2
    global_minor_ratio_threshold: float = 0.75
    global_area_ratio_threshold: float = 0.70
    aspect_ratio_threshold: float = 0.70
    local_minor_ratio_threshold: float = 0.80
    local_area_ratio_threshold: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_minor_quantile": float(self.baseline_minor_quantile),
            "baseline_area_quantile": float(self.baseline_area_quantile),
            "baseline_major_quantile": float(self.baseline_major_quantile),
            "local_window": int(self.local_window),
            "global_minor_ratio_threshold": float(self.global_minor_ratio_threshold),
            "global_area_ratio_threshold": float(self.global_area_ratio_threshold),
            "aspect_ratio_threshold": float(self.aspect_ratio_threshold),
            "local_minor_ratio_threshold": float(self.local_minor_ratio_threshold),
            "local_area_ratio_threshold": float(self.local_area_ratio_threshold),
        }


@dataclass
class RawEllipseBlinkCandidate:
    frame_filename: str
    frame_idx: int | None
    timestamp_us: int
    ellipse_xywht: list[float]
    ellipse_major_axis_px: float
    ellipse_minor_axis_px: float
    ellipse_area_px2: float
    aspect_ratio: float
    baseline_major_axis_px: float
    baseline_minor_axis_px: float
    baseline_area_px2: float
    global_major_axis_ratio: float
    global_minor_axis_ratio: float
    global_area_ratio: float
    local_neighbor_minor_axis_px: float
    local_neighbor_area_px2: float
    local_minor_axis_ratio: float
    local_area_ratio: float
    blink_score: float
    blink_candidate: bool
    blink_candidate_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_filename": str(self.frame_filename),
            "frame_idx": None if self.frame_idx is None else int(self.frame_idx),
            "timestamp_us": int(self.timestamp_us),
            "ellipse_xywht": [float(v) for v in self.ellipse_xywht],
            "ellipse_major_axis_px": float(self.ellipse_major_axis_px),
            "ellipse_minor_axis_px": float(self.ellipse_minor_axis_px),
            "ellipse_area_px2": float(self.ellipse_area_px2),
            "aspect_ratio": float(self.aspect_ratio),
            "baseline_major_axis_px": float(self.baseline_major_axis_px),
            "baseline_minor_axis_px": float(self.baseline_minor_axis_px),
            "baseline_area_px2": float(self.baseline_area_px2),
            "global_major_axis_ratio": float(self.global_major_axis_ratio),
            "global_minor_axis_ratio": float(self.global_minor_axis_ratio),
            "global_area_ratio": float(self.global_area_ratio),
            "local_neighbor_minor_axis_px": float(self.local_neighbor_minor_axis_px),
            "local_neighbor_area_px2": float(self.local_neighbor_area_px2),
            "local_minor_axis_ratio": float(self.local_minor_axis_ratio),
            "local_area_ratio": float(self.local_area_ratio),
            "blink_score": float(self.blink_score),
            "blink_candidate": bool(self.blink_candidate),
            "blink_candidate_reasons": list(self.blink_candidate_reasons),
        }


def _normalize_session_code(session_code: str) -> str:
    value = str(session_code).strip()
    if value.startswith("session_"):
        return session_dir_to_code(value)
    if value.startswith("session"):
        value = value[len("session") :]
    value = value.replace("_", "")
    if len(value) != 3 or not value.isdigit():
        raise ValueError(f"Unsupported session code: {session_code}")
    return value


def _resolve_raw_user_dir(raw_root: str | Path, user_id: int) -> Path:
    raw_root = Path(raw_root).resolve()
    for user_dir in collect_users(raw_root):
        try:
            parsed_user_id = int(user_dir.name.replace("user", ""))
        except ValueError:
            continue
        if parsed_user_id == int(user_id):
            return user_dir
    return raw_root / f"user{int(user_id):02d}"


def _ellipse_axes(ellipse_xywht: Sequence[float]) -> tuple[float, float]:
    _, _, width, height, _ = [float(v) for v in ellipse_xywht]
    return max(width, height), min(width, height)


def _safe_quantile(values: Sequence[float], quantile: float, *, default: float = 1.0) -> float:
    if not values:
        return float(default)
    clipped = min(max(float(quantile), 0.0), 1.0)
    return float(np.quantile(np.asarray(values, dtype=np.float32), clipped))


def _local_reference(values: Sequence[float], index: int, *, window: int) -> float:
    if not values:
        return 1.0
    radius = max(0, int(window))
    lo = max(0, index - radius)
    hi = min(len(values), index + radius + 1)
    neighbors = [float(values[idx]) for idx in range(lo, hi) if idx != index]
    if not neighbors:
        return float(values[index])
    return float(np.median(np.asarray(neighbors, dtype=np.float32)))


def _ratio(value: float, reference: float) -> float:
    denom = max(float(reference), 1.0e-6)
    return float(value) / denom


def _deficit(ratio: float, threshold: float) -> float:
    limit = max(float(threshold), 1.0e-6)
    return max(0.0, (limit - float(ratio)) / limit)


def analyze_raw_ellipse_blink_candidates(
    annotations: Sequence[EllipseAnnotation],
    *,
    heuristic: RawEllipseBlinkHeuristicConfig | None = None,
) -> dict[str, Any]:
    cfg = heuristic or RawEllipseBlinkHeuristicConfig()
    if not annotations:
        return {
            "rows": [],
            "references": {
                "baseline_major_axis_px": 0.0,
                "baseline_minor_axis_px": 0.0,
                "baseline_area_px2": 0.0,
            },
            "n_annotations": 0,
            "n_blink_candidates": 0,
            "candidate_rate": 0.0,
            "heuristic": cfg.to_dict(),
        }

    major_axes: list[float] = []
    minor_axes: list[float] = []
    areas: list[float] = []
    for ann in annotations:
        major_axis, minor_axis = _ellipse_axes(ann.ellipse_xywht)
        major_axes.append(major_axis)
        minor_axes.append(minor_axis)
        areas.append(major_axis * minor_axis)

    baseline_major = max(_safe_quantile(major_axes, cfg.baseline_major_quantile), 1.0e-6)
    baseline_minor = max(_safe_quantile(minor_axes, cfg.baseline_minor_quantile), 1.0e-6)
    baseline_area = max(_safe_quantile(areas, cfg.baseline_area_quantile), 1.0e-6)

    rows: list[RawEllipseBlinkCandidate] = []
    candidate_count = 0
    for index, ann in enumerate(annotations):
        major_axis = major_axes[index]
        minor_axis = minor_axes[index]
        area = areas[index]
        aspect_ratio = _ratio(minor_axis, major_axis)
        global_major_ratio = _ratio(major_axis, baseline_major)
        global_minor_ratio = _ratio(minor_axis, baseline_minor)
        global_area_ratio = _ratio(area, baseline_area)
        local_minor_ref = _local_reference(minor_axes, index, window=cfg.local_window)
        local_area_ref = _local_reference(areas, index, window=cfg.local_window)
        local_minor_ratio = _ratio(minor_axis, local_minor_ref)
        local_area_ratio = _ratio(area, local_area_ref)

        reasons: list[str] = []
        if global_minor_ratio <= float(cfg.global_minor_ratio_threshold):
            reasons.append("minor_axis_collapse")
        if global_area_ratio <= float(cfg.global_area_ratio_threshold):
            reasons.append("area_collapse")
        if aspect_ratio <= float(cfg.aspect_ratio_threshold):
            reasons.append("aspect_collapse")
        if local_minor_ratio <= float(cfg.local_minor_ratio_threshold):
            reasons.append("local_minor_drop")
        if local_area_ratio <= float(cfg.local_area_ratio_threshold):
            reasons.append("local_area_drop")

        blink_candidate = bool(
            global_minor_ratio <= float(cfg.global_minor_ratio_threshold)
            and global_area_ratio <= float(cfg.global_area_ratio_threshold)
            and aspect_ratio <= float(cfg.aspect_ratio_threshold)
            and (
                local_minor_ratio <= float(cfg.local_minor_ratio_threshold)
                or local_area_ratio <= float(cfg.local_area_ratio_threshold)
            )
        )
        blink_score = min(
            1.0,
            0.40 * _deficit(global_minor_ratio, cfg.global_minor_ratio_threshold)
            + 0.25 * _deficit(global_area_ratio, cfg.global_area_ratio_threshold)
            + 0.20 * _deficit(aspect_ratio, cfg.aspect_ratio_threshold)
            + 0.10 * _deficit(local_minor_ratio, cfg.local_minor_ratio_threshold)
            + 0.05 * _deficit(local_area_ratio, cfg.local_area_ratio_threshold),
        )
        if blink_candidate:
            candidate_count += 1

        rows.append(
            RawEllipseBlinkCandidate(
                frame_filename=str(ann.frame_filename),
                frame_idx=ann.frame_idx,
                timestamp_us=int(ann.timestamp_us),
                ellipse_xywht=[float(v) for v in ann.ellipse_xywht],
                ellipse_major_axis_px=float(major_axis),
                ellipse_minor_axis_px=float(minor_axis),
                ellipse_area_px2=float(area),
                aspect_ratio=float(aspect_ratio),
                baseline_major_axis_px=float(baseline_major),
                baseline_minor_axis_px=float(baseline_minor),
                baseline_area_px2=float(baseline_area),
                global_major_axis_ratio=float(global_major_ratio),
                global_minor_axis_ratio=float(global_minor_ratio),
                global_area_ratio=float(global_area_ratio),
                local_neighbor_minor_axis_px=float(local_minor_ref),
                local_neighbor_area_px2=float(local_area_ref),
                local_minor_axis_ratio=float(local_minor_ratio),
                local_area_ratio=float(local_area_ratio),
                blink_score=float(blink_score),
                blink_candidate=blink_candidate,
                blink_candidate_reasons=reasons,
            )
        )

    candidate_rate = float(candidate_count) / max(len(rows), 1)
    return {
        "rows": rows,
        "references": {
            "baseline_major_axis_px": float(baseline_major),
            "baseline_minor_axis_px": float(baseline_minor),
            "baseline_area_px2": float(baseline_area),
        },
        "n_annotations": len(rows),
        "n_blink_candidates": int(candidate_count),
        "candidate_rate": float(candidate_rate),
        "heuristic": cfg.to_dict(),
    }


def _analysis_to_blink_metadata(
    analysis: dict[str, Any],
    *,
    source: str,
    extra_report_fields: dict[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata_by_frame: dict[str, dict[str, Any]] = {}
    for row in analysis["rows"]:
        metadata_by_frame[str(row.frame_filename)] = {
            "closed_eye_flag": bool(row.blink_candidate),
            "blink_candidate_score": float(row.blink_score),
            "blink_candidate_reasons": list(row.blink_candidate_reasons),
            "blink_candidate_source": str(source),
        }
    report = {
        "enabled": True,
        "source": str(source),
        "n_annotations": int(analysis["n_annotations"]),
        "n_blink_candidates": int(analysis["n_blink_candidates"]),
        "candidate_rate": float(analysis["candidate_rate"]),
        "heuristic": dict(analysis["heuristic"]),
        "references": dict(analysis["references"]),
    }
    if extra_report_fields:
        report.update(dict(extra_report_fields))
    return metadata_by_frame, report


def _groundedsam_row_to_ellipse_annotation(row: dict[str, Any]) -> EllipseAnnotation | None:
    ellipse = row.get("pupil_ellipse_xywht_sensor") or row.get("ellipse_sensor_xywht") or row.get("ellipse_xywht")
    if ellipse is None or len(ellipse) != 5:
        return None

    frame_filename = str(row.get("frame_filename") or "")
    parsed = try_parse_frame_filename(frame_filename) if frame_filename else None
    frame_idx = row.get("frame_idx")
    if frame_idx is None and parsed is not None:
        frame_idx = parsed[0]
    timestamp_us = row.get("timestamp_us")
    if timestamp_us is None and parsed is not None:
        timestamp_us = parsed[1]
    if timestamp_us is None:
        timestamp_us = 0

    return EllipseAnnotation(
        frame_filename=frame_filename,
        frame_idx=None if frame_idx is None else int(frame_idx),
        timestamp_us=int(timestamp_us),
        ellipse_xywht=tuple(float(v) for v in ellipse),
    )


def build_raw_ellipse_blink_metadata(
    annotations: Sequence[EllipseAnnotation],
    *,
    heuristic: RawEllipseBlinkHeuristicConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    analysis = analyze_raw_ellipse_blink_candidates(annotations, heuristic=heuristic)
    return _analysis_to_blink_metadata(analysis, source=RAW_ELLIPSE_BLINK_SOURCE)


def build_groundedsam_store_blink_metadata(
    rows: Sequence[dict[str, Any]],
    *,
    heuristic: RawEllipseBlinkHeuristicConfig | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    annotations: list[EllipseAnnotation] = []
    for row in rows:
        annotation = _groundedsam_row_to_ellipse_annotation(dict(row))
        if annotation is not None:
            annotations.append(annotation)

    analysis = analyze_raw_ellipse_blink_candidates(annotations, heuristic=heuristic)
    return _analysis_to_blink_metadata(
        analysis,
        source=GROUNDEDSAM_ELLIPSE_BLINK_SOURCE,
        extra_report_fields={
            "n_rows_total": int(len(rows)),
            "n_rows_with_ellipse": int(len(annotations)),
            "n_rows_skipped": int(max(0, len(rows) - len(annotations))),
        },
    )


def apply_groundedsam_store_blink_metadata(
    rows: Sequence[dict[str, Any]],
    *,
    heuristic: RawEllipseBlinkHeuristicConfig | None = None,
    overwrite_existing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    updated_rows = [dict(row) for row in rows]
    metadata_by_frame, report = build_groundedsam_store_blink_metadata(updated_rows, heuristic=heuristic)
    rows_updated = 0

    for row in updated_rows:
        frame_filename = str(row.get("frame_filename") or "")
        metadata = metadata_by_frame.get(frame_filename)
        if metadata is None:
            continue

        has_existing_metadata = bool(
            row.get("blink_candidate_source") not in (None, "")
            or "blink_candidate_score" in row
            or "blink_candidate_reasons" in row
        )
        if has_existing_metadata and not overwrite_existing:
            continue

        previous_payload = (
            bool(row.get("closed_eye_flag", False)),
            float(row.get("blink_candidate_score", 0.0)),
            tuple(row.get("blink_candidate_reasons", [])),
            row.get("blink_candidate_source"),
        )
        row.update(metadata)
        new_payload = (
            bool(row.get("closed_eye_flag", False)),
            float(row.get("blink_candidate_score", 0.0)),
            tuple(row.get("blink_candidate_reasons", [])),
            row.get("blink_candidate_source"),
        )
        if new_payload != previous_payload:
            rows_updated += 1

    report = {
        **report,
        "overwrite_existing": bool(overwrite_existing),
        "n_rows_updated": int(rows_updated),
    }
    return updated_rows, report


def export_raw_ellipse_blink_candidates(
    *,
    raw_root: str | Path,
    output_root: str | Path,
    user_id: int,
    eye: str,
    session_code: str,
    heuristic: RawEllipseBlinkHeuristicConfig | None = None,
) -> dict[str, Any]:
    cfg = heuristic or RawEllipseBlinkHeuristicConfig()
    normalized_eye = str(eye).strip().lower()
    if normalized_eye not in {"left", "right"}:
        raise ValueError(f"Unsupported eye selector: {eye}")

    normalized_session_code = _normalize_session_code(session_code)
    raw_user_dir = _resolve_raw_user_dir(raw_root, int(user_id))
    raw_session_dir = raw_user_dir / normalized_eye / code_to_session_dir(normalized_session_code)
    if not raw_session_dir.exists():
        raise FileNotFoundError(f"Raw session not found: {raw_session_dir}")

    layout = discover_session_layout(raw_session_dir, user_id=int(user_id))
    if layout.annotation_csv is None:
        raise FileNotFoundError(
            f"No raw annotation CSV found for {raw_session_dir}. "
            f"Blink-candidate analysis needs a raw VIA CSV with ellipse annotations."
        )

    frame_records = collect_frame_records(layout.frames_dir) if layout.frames_dir else []
    annotations, parse_report = parse_via_csv_with_report(
        layout.annotation_csv,
        known_frame_filenames=[record.filename for record in frame_records],
        known_frame_stems=[Path(record.filename).stem for record in frame_records],
    )
    analysis = analyze_raw_ellipse_blink_candidates(annotations, heuristic=cfg)

    session_output_dir = (
        Path(output_root).resolve()
        / "sessions"
        / canonical_user_name(int(user_id))
        / normalized_eye
        / f"session_{normalized_session_code}"
    )
    analysis_rows_path = session_output_dir / "ellipse_blink_analysis.jsonl"
    candidate_rows_path = session_output_dir / "ellipse_blink_candidates.jsonl"
    summary_path = session_output_dir / "ellipse_blink_summary.json"

    row_dicts = [row.to_dict() for row in analysis["rows"]]
    candidate_dicts = [row for row in row_dicts if row.get("blink_candidate")]
    write_jsonl(row_dicts, analysis_rows_path)
    write_jsonl(candidate_dicts, candidate_rows_path)

    summary = {
        "user_id": int(user_id),
        "eye": normalized_eye,
        "session_code": str(normalized_session_code),
        "raw_session_dir": str(raw_session_dir),
        "annotation_csv": str(layout.annotation_csv),
        "frames_dir": str(layout.frames_dir) if layout.frames_dir else None,
        "n_frames": len(frame_records),
        "n_annotations": int(analysis["n_annotations"]),
        "n_blink_candidates": int(analysis["n_blink_candidates"]),
        "candidate_rate": float(analysis["candidate_rate"]),
        "parse_report": dict(parse_report),
        "heuristic": dict(analysis["heuristic"]),
        "references": dict(analysis["references"]),
        "analysis_rows_path": str(analysis_rows_path),
        "candidate_rows_path": str(candidate_rows_path),
        "output_dir": str(session_output_dir),
    }
    write_json(summary, summary_path)
    summary["summary_path"] = str(summary_path)
    return summary
