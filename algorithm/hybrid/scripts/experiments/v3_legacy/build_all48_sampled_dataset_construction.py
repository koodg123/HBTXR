#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import (
    discover_session_layout,
    parse_via_csv_with_report,
    rasterize_ellipse_mask,
)
from hbtxr.preprocess.raw_ellipse_blink import build_raw_ellipse_blink_metadata
from hbtxr.utils.io import read_json, write_json, write_jsonl


SENSOR_WIDTH = 346
SENSOR_HEIGHT = 240


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a sampled all-48 dataset-construction experiment on the exact same 8-frame-per-session "
            "slice used by the recent ROI-crop follow-up runs."
        ),
    )
    parser.add_argument(
        "--source-batch-root",
        type=str,
        default="workspace_session_samples_7/roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples",
        help="Existing ROI-crop batch root used as the source-of-truth for the exact sampled frame list.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="workspace_session_samples_10/all48_dataset_construction_same8samples",
        help="Output root for merged sampled annotations and analysis artifacts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _output_paths(output_root: Path, *, session_key: str, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    session_rel = Path("sessions") / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}"
    session_dir = output_root / session_rel
    return {
        "session_rel": session_rel,
        "session_dir": session_dir,
        "labels_dir": session_dir / "labels",
        "eye_mask_dir": session_dir / "labels" / "eye_region_masks",
        "pupil_mask_dir": session_dir / "labels" / "pupil_masks",
        "rows_path": session_dir / "frame_annotations.jsonl",
        "summary_path": session_dir / "session_summary.json",
    }


def _save_binary_mask(mask: np.ndarray, path: Path) -> str:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(((np.asarray(mask) > 0).astype(np.uint8) * 255), mode="L").save(path)
    return str(path.resolve())


def _rasterize_bbox_mask(bbox_xywh: list[float], *, image_size_wh: tuple[int, int]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox_xywh]
    image_w, image_h = [int(v) for v in image_size_wh]
    x0 = max(0, min(int(np.floor(x)), image_w - 1))
    y0 = max(0, min(int(np.floor(y)), image_h - 1))
    x1 = max(x0 + 1, min(int(np.ceil(x + w)), image_w))
    y1 = max(y0 + 1, min(int(np.ceil(y + h)), image_h))
    mask = np.zeros((image_h, image_w), dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return mask


def _ellipse_bbox_xywh(ellipse_xywht: list[float]) -> list[float]:
    cx, cy, w, h, _theta = [float(v) for v in ellipse_xywht]
    return [cx - w / 2.0, cy - h / 2.0, w, h]


def _int_from_row(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _load_csv_rows_by_filename(csv_path: Path | None) -> dict[str, list[dict[str, str]]]:
    rows_by_filename: dict[str, list[dict[str, str]]] = {}
    if csv_path is None or not csv_path.exists():
        return rows_by_filename
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            filename = str(row.get("filename", "")).strip()
            rows_by_filename.setdefault(filename, []).append(dict(row))
    return rows_by_filename


def _csv_status(rows: list[dict[str, str]], *, csv_exists: bool) -> tuple[str, int | None]:
    if not csv_exists:
        return "csv_file_missing", None
    if not rows:
        return "csv_row_missing", None
    max_region = max((_int_from_row(row, "region_count", 0) for row in rows), default=0)
    if max_region > 0:
        return "csv_positive", int(max_region)
    return "csv_region_zero", 0


def _write_markdown_summary(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# All-48 Sampled Dataset Construction Summary",
        "",
        "## Aggregate",
        f"- sessions: {summary['n_sessions']}",
        f"- sampled_frames: {summary['n_sampled_frames']}",
        f"- eye_roi_present: {summary['n_eye_roi_present']}",
        f"- eye_roi_missing: {summary['n_eye_roi_missing']}",
        f"- pupil_geometry_complete: {summary['n_pupil_geometry_complete']}",
        f"- pupil_state_only: {summary['n_pupil_state_only']}",
        f"- pupil_related_missing: {summary['n_pupil_related_missing']}",
        "",
        "## CSV Status Counts",
    ]
    for key, value in summary.get("csv_status_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Label Status Counts",
        ]
    )
    for key, value in summary.get("label_status_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Top Sessions By Missing Pupil Related Annotation",
        ]
    )
    for row in summary.get("top_sessions_by_missing_pupil_related", []):
        lines.append(
            f"- {row['session_key']}: missing_pupil_related={row['n_pupil_related_missing']} "
            f"state_only={row['n_pupil_state_only']} complete={row['n_pupil_geometry_complete']}/{row['n_sampled']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_missing_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Missing-Only Pupil Annotation Analysis",
        "",
        "## Aggregate",
        f"- rows: {summary['n_rows']}",
        "",
        "## Missing Reasons",
    ]
    for key, value in summary.get("reason_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Top Sessions",
        ]
    )
    for row in summary.get("top_sessions", []):
        lines.append(
            f"- {row['session_key']}: missing={row['n_missing']} "
            f"csv_status={row['dominant_csv_status']} eye_status={row['dominant_eye_status']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sorted_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter.keys())}


def main() -> None:
    args = build_argparser().parse_args()
    source_batch_root = (PROJECT_ROOT / args.source_batch_root).resolve() if not Path(args.source_batch_root).is_absolute() else Path(args.source_batch_root).resolve()
    output_root = (PROJECT_ROOT / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root).resolve()

    summary_path = output_root / "experiment_summary.json"
    if summary_path.exists() and not args.overwrite:
        summary = read_json(summary_path)
        print(f"[DONE] reused_existing_summary summary={summary_path} sampled_frames={summary['n_sampled_frames']}")
        return

    source_summaries = [read_json(path) for path in sorted((source_batch_root / "sessions").rglob("summary.json"))]
    if not source_summaries:
        raise FileNotFoundError(f"No source summary.json files found under {source_batch_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest_rows: list[dict[str, Any]] = []
    merged_rows_all: list[dict[str, Any]] = []
    missing_only_rows: list[dict[str, Any]] = []
    session_summaries_out: list[dict[str, Any]] = []

    csv_status_counts: Counter[str] = Counter()
    label_status_counts: Counter[str] = Counter()
    eye_detection_status_counts: Counter[str] = Counter()
    missing_reason_counts: Counter[str] = Counter()

    for session in source_summaries:
        session_key = str(session["session_key"])
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        raw_session_dir = Path(session["raw_session_dir"])
        layout = discover_session_layout(raw_session_dir, user_id=user_id)
        csv_exists = bool(layout.annotation_csv is not None and layout.annotation_csv.exists())
        csv_rows_by_filename = _load_csv_rows_by_filename(layout.annotation_csv)
        known_filenames = [str(frame["frame_filename"]) for frame in session.get("frames", [])]
        known_stems = [Path(name).stem for name in known_filenames]
        annotations, parse_report = parse_via_csv_with_report(
            layout.annotation_csv,
            known_frame_filenames=known_filenames,
            known_frame_stems=known_stems,
        ) if csv_exists else ([], {})
        annotation_by_filename = {str(ann.frame_filename): ann for ann in annotations}
        blink_metadata_by_frame, blink_report = build_raw_ellipse_blink_metadata(annotations) if annotations else ({}, {})

        paths = _output_paths(output_root, session_key=session_key, user_id=user_id, eye=eye, session_code=session_code)
        rows: list[dict[str, Any]] = []
        n_eye_present = 0
        n_pupil_geometry_complete = 0
        n_pupil_state_only = 0
        n_pupil_related_missing = 0

        for frame in session.get("frames", []):
            frame_filename = str(frame["frame_filename"])
            csv_rows = csv_rows_by_filename.get(frame_filename, [])
            csv_status, csv_max_region_count = _csv_status(csv_rows, csv_exists=csv_exists)
            csv_status_counts[csv_status] += 1

            row: dict[str, Any] = {
                "session_key": session_key,
                "user_id": int(user_id),
                "subject_id": int(user_id),
                "eye": eye,
                "session_code": session_code,
                "frame_filename": frame_filename,
                "frame_idx": int(frame["frame_idx"]),
                "timestamp_us": int(frame["timestamp_us"]),
                "raw_frame_path": str(frame["raw_frame_path"]),
                "sample_source_batch_root": str(source_batch_root),
                "sample_source_summary_path": str((source_batch_root / paths["session_rel"] / "summary.json").resolve()),
                "eye_detection_status": str(frame.get("eye_detection_status") or "unknown"),
                "eye_region_bbox_xywh_sensor": frame.get("eye_region_bbox_xywh_sensor"),
                "eye_region_mask_path": None,
                "eye_region_mask_valid": False,
                "eye_region_mask_source": None,
                "eye_region_bbox_source": None,
                "csv_status": csv_status,
                "csv_max_region_count": csv_max_region_count,
                "csv_rows_found": int(len(csv_rows)),
                "pupil_state_flag": "unknown",
                "eye_state_flag": "unknown",
                "closed_eye_flag": False,
                "pupil_ellipse_xywht_sensor": None,
                "pupil_region_bbox_xywh_sensor": None,
                "pupil_bbox_xywh_sensor": None,
                "pupil_mask_path": None,
                "mask_path": None,
                "mask_valid": False,
                "pupil_state_source": None,
                "pupil_bbox_source": None,
                "pupil_mask_source": None,
                "pupil_annotation_level": "missing",
                "label_status": None,
                "annotation_reason": None,
                "annotation_sources_used": {
                    "eye": None,
                    "pupil": None,
                },
            }

            eye_bbox = row["eye_region_bbox_xywh_sensor"]
            if eye_bbox is not None:
                if row["eye_detection_status"] == "unknown":
                    row["eye_detection_status"] = "ok"
                eye_mask = _rasterize_bbox_mask([float(v) for v in eye_bbox], image_size_wh=(SENSOR_WIDTH, SENSOR_HEIGHT))
                eye_mask_path = _save_binary_mask(
                    eye_mask,
                    paths["eye_mask_dir"] / Path(frame_filename).with_suffix(".png").name,
                )
                row["eye_region_mask_path"] = eye_mask_path
                row["eye_region_mask_valid"] = True
                row["eye_region_mask_source"] = "groundedsam_eye_bbox_rasterized_mask"
                row["eye_region_bbox_source"] = "groundedsam_v10_preview_eye_stage"
                row["annotation_sources_used"]["eye"] = "groundedsam_v10_preview_eye_stage"
                n_eye_present += 1
            eye_detection_status_counts[row["eye_detection_status"]] += 1

            blink_meta = blink_metadata_by_frame.get(frame_filename, {})
            if csv_status == "csv_positive" and frame_filename in annotation_by_filename:
                ann = annotation_by_filename[frame_filename]
                ellipse = [float(v) for v in ann.ellipse_xywht]
                pupil_mask = rasterize_ellipse_mask(ellipse, image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
                pupil_mask_path = _save_binary_mask(
                    pupil_mask,
                    paths["pupil_mask_dir"] / Path(frame_filename).with_suffix(".png").name,
                )
                closed_eye_flag = bool(blink_meta.get("closed_eye_flag", False))
                row["pupil_state_flag"] = "closed" if closed_eye_flag else "open"
                row["eye_state_flag"] = row["pupil_state_flag"]
                row["closed_eye_flag"] = closed_eye_flag
                row["pupil_ellipse_xywht_sensor"] = ellipse
                row["pupil_region_bbox_xywh_sensor"] = _ellipse_bbox_xywh(ellipse)
                row["pupil_bbox_xywh_sensor"] = row["pupil_region_bbox_xywh_sensor"]
                row["pupil_mask_path"] = pupil_mask_path
                row["mask_path"] = pupil_mask_path
                row["mask_valid"] = True
                row["pupil_state_source"] = "raw_csv_state"
                row["pupil_bbox_source"] = "derived_from_raw_csv_state"
                row["pupil_mask_source"] = "raw_csv_state"
                row["pupil_annotation_level"] = "geometry_complete"
                row["annotation_sources_used"]["pupil"] = "raw_csv_positive"
            elif csv_status == "csv_region_zero":
                row["pupil_state_flag"] = "closed"
                row["eye_state_flag"] = "closed"
                row["closed_eye_flag"] = True
                row["pupil_state_source"] = "csv_region_zero_proxy"
                row["pupil_annotation_level"] = "state_only"
                row["annotation_sources_used"]["pupil"] = "csv_region_zero_proxy"
            else:
                row["pupil_annotation_level"] = "missing"

            if row["eye_region_mask_valid"] and row["pupil_annotation_level"] == "geometry_complete":
                row["label_status"] = "annotated_complete"
                row["annotation_reason"] = "groundedsam_eye_and_raw_csv_pupil_geometry"
                n_pupil_geometry_complete += 1
            elif row["eye_region_mask_valid"] and row["pupil_annotation_level"] == "state_only":
                row["label_status"] = "annotated_state_only"
                row["annotation_reason"] = "raw_csv_state_only_no_pupil_geometry"
                n_pupil_state_only += 1
            elif row["eye_region_mask_valid"]:
                row["label_status"] = "annotated_eye_only"
                row["annotation_reason"] = {
                    "csv_file_missing": "raw_csv_file_missing",
                    "csv_row_missing": "raw_csv_row_missing_for_sampled_frame",
                }.get(csv_status, "pupil_related_annotation_missing")
                n_pupil_related_missing += 1
            elif row["pupil_annotation_level"] == "geometry_complete":
                row["label_status"] = "annotated_pupil_only_eye_missing"
                row["annotation_reason"] = "groundedsam_eye_missing_but_raw_csv_pupil_geometry_present"
                n_pupil_geometry_complete += 1
            elif row["pupil_annotation_level"] == "state_only":
                row["label_status"] = "annotated_pupil_state_only_eye_missing"
                row["annotation_reason"] = "groundedsam_eye_missing_but_raw_csv_state_only_present"
                n_pupil_state_only += 1
            else:
                row["label_status"] = "pending_annotation_sources"
                row["annotation_reason"] = "groundedsam_eye_and_pupil_related_annotation_missing"
                n_pupil_related_missing += 1

            label_status_counts[row["label_status"]] += 1
            sample_manifest_rows.append(
                {
                    "session_key": session_key,
                    "user_id": int(user_id),
                    "eye": eye,
                    "session_code": session_code,
                    "frame_filename": frame_filename,
                    "frame_idx": int(frame["frame_idx"]),
                    "timestamp_us": int(frame["timestamp_us"]),
                    "raw_frame_path": str(frame["raw_frame_path"]),
                    "sample_source_batch_root": str(source_batch_root),
                }
            )
            if row["pupil_annotation_level"] == "missing":
                missing_reason_counts[row["annotation_reason"]] += 1
                missing_only_rows.append(dict(row))
            rows.append(row)
            merged_rows_all.append(dict(row))

        write_jsonl(rows, paths["rows_path"])
        session_summary = {
            "session_key": session_key,
            "raw_session_dir": str(raw_session_dir),
            "rows_path": str(paths["rows_path"]),
            "n_sampled": int(len(rows)),
            "n_eye_present": int(n_eye_present),
            "n_eye_missing": int(len(rows) - n_eye_present),
            "n_pupil_geometry_complete": int(n_pupil_geometry_complete),
            "n_pupil_state_only": int(n_pupil_state_only),
            "n_pupil_related_missing": int(n_pupil_related_missing),
            "csv_exists": bool(csv_exists),
            "csv_parse_report": dict(parse_report),
            "blink_report": dict(blink_report),
        }
        write_json(session_summary, paths["summary_path"])
        session_summaries_out.append(session_summary)

    sample_manifest_path = output_root / "sample_manifest.jsonl"
    merged_rows_path = output_root / "merged_frame_annotations.jsonl"
    missing_only_rows_path = analysis_dir / "missing_only_records.jsonl"
    write_jsonl(sample_manifest_rows, sample_manifest_path)
    write_jsonl(merged_rows_all, merged_rows_path)
    write_jsonl(missing_only_rows, missing_only_rows_path)
    write_jsonl(session_summaries_out, output_root / "session_summaries.jsonl")

    top_sessions_by_missing = sorted(
        session_summaries_out,
        key=lambda row: (-row["n_pupil_related_missing"], -row["n_pupil_state_only"], row["session_key"]),
    )[:20]
    summary = {
        "source_batch_root": str(source_batch_root),
        "output_root": str(output_root),
        "n_sessions": int(len(session_summaries_out)),
        "n_sampled_frames": int(len(merged_rows_all)),
        "n_eye_roi_present": int(sum(row["n_eye_present"] for row in session_summaries_out)),
        "n_eye_roi_missing": int(sum(row["n_eye_missing"] for row in session_summaries_out)),
        "n_pupil_geometry_complete": int(sum(row["n_pupil_geometry_complete"] for row in session_summaries_out)),
        "n_pupil_state_only": int(sum(row["n_pupil_state_only"] for row in session_summaries_out)),
        "n_pupil_related_missing": int(sum(row["n_pupil_related_missing"] for row in session_summaries_out)),
        "csv_status_counts": _sorted_counter_dict(csv_status_counts),
        "label_status_counts": _sorted_counter_dict(label_status_counts),
        "eye_detection_status_counts": _sorted_counter_dict(eye_detection_status_counts),
        "top_sessions_by_missing_pupil_related": top_sessions_by_missing,
        "sample_manifest_path": str(sample_manifest_path),
        "merged_rows_path": str(merged_rows_path),
        "missing_only_rows_path": str(missing_only_rows_path),
    }
    write_json(summary, summary_path)
    _write_markdown_summary(summary, output_root / "experiment_summary.md")

    missing_session_counter: Counter[str] = Counter()
    missing_csv_status_counter: Counter[str] = Counter()
    missing_eye_status_counter: Counter[str] = Counter()
    for row in missing_only_rows:
        missing_session_counter[str(row["session_key"])] += 1
        missing_csv_status_counter[str(row["csv_status"])] += 1
        missing_eye_status_counter[str(row["eye_detection_status"])] += 1
    missing_summary = {
        "n_rows": int(len(missing_only_rows)),
        "reason_counts": _sorted_counter_dict(missing_reason_counts),
        "csv_status_counts": _sorted_counter_dict(missing_csv_status_counter),
        "eye_detection_status_counts": _sorted_counter_dict(missing_eye_status_counter),
        "top_sessions": [
            {
                "session_key": session_key,
                "n_missing": int(count),
                "dominant_csv_status": max(
                    (
                        (status, sum(1 for row in missing_only_rows if row["session_key"] == session_key and row["csv_status"] == status))
                        for status in missing_csv_status_counter.keys()
                    ),
                    key=lambda item: item[1],
                    default=("none", 0),
                )[0],
                "dominant_eye_status": max(
                    (
                        (status, sum(1 for row in missing_only_rows if row["session_key"] == session_key and row["eye_detection_status"] == status))
                        for status in missing_eye_status_counter.keys()
                    ),
                    key=lambda item: item[1],
                    default=("none", 0),
                )[0],
            }
            for session_key, count in missing_session_counter.most_common(20)
        ],
    }
    write_json(missing_summary, analysis_dir / "missing_only_summary.json")
    _write_missing_markdown(missing_summary, analysis_dir / "missing_only_summary.md")

    print(
        f"[DONE] sessions={summary['n_sessions']} sampled_frames={summary['n_sampled_frames']} "
        f"eye_present={summary['n_eye_roi_present']} pupil_geometry_complete={summary['n_pupil_geometry_complete']} "
        f"pupil_state_only={summary['n_pupil_state_only']} pupil_related_missing={summary['n_pupil_related_missing']} "
        f"summary={summary_path}"
    )


if __name__ == "__main__":
    main()
