#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import (
    discover_session_layout,
    maybe_link_or_copy,
    parse_via_csv_with_report,
    rasterize_ellipse_mask,
)
from hbtxr.utils.io import read_json, write_json, write_jsonl


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge tsgss sampled session summaries with raw CSV supervision to build a dense-annotation table."
        ),
    )
    parser.add_argument(
        "--batch-root",
        type=str,
        default="tsgss/workspace/roi_crop_prompted_all_sessions",
        help="Workspace root produced by run_roi_crop_prompted_preview_batch.py",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="tsgss/dense_annotation",
        help="Dense-annotation export root",
    )
    parser.add_argument("--link-mode", type=str, default="symlink", choices=["symlink", "copy", "skip"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_csv_rows_by_filename(csv_path: Path | None) -> dict[str, list[dict[str, str]]]:
    rows_by_filename: dict[str, list[dict[str, str]]] = defaultdict(list)
    if csv_path is None or not csv_path.exists():
        return rows_by_filename
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_filename[str(row.get("filename", "")).strip()].append(dict(row))
    return rows_by_filename


def _int_from_row(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _csv_status(rows: list[dict[str, str]], *, csv_exists: bool) -> tuple[str, int | None]:
    if not csv_exists:
        return "csv_file_missing", None
    if not rows:
        return "csv_row_missing", None
    max_region = max((_int_from_row(row, "region_count", 0) for row in rows), default=0)
    if max_region > 0:
        return "csv_positive", int(max_region)
    return "csv_region_zero", 0


def _ellipse_to_bbox_xywh(ellipse_xywht: list[float] | tuple[float, ...]) -> list[float]:
    cx, cy, w, h, _theta = [float(v) for v in ellipse_xywht]
    return [cx - 0.5 * w, cy - 0.5 * h, w, h]


def _resolve_output_root(path_text: str) -> Path:
    path = Path(path_text)
    return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _relative_to(root: Path, path: Path) -> str:
    root_abs = root.resolve()
    path_abs = path if path.is_absolute() else (PROJECT_ROOT / path)
    try:
        return str(path_abs.relative_to(root_abs))
    except ValueError:
        return str(path_abs)


def _session_label_dirs(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
    base = output_root / "sessions" / f"user{int(user_id):02d}" / str(eye) / f"session_{session_code}" / "labels"
    return {
        "base": base,
        "eye_masks": base / "eye_region_masks",
        "pupil_masks": base / "pupil_masks",
    }


def _copy_or_link_if_present(src_text: str | None, dst: Path, *, mode: str, overwrite: bool) -> str | None:
    if not src_text:
        return None
    src = Path(src_text).resolve()
    if not src.exists():
        return None
    maybe_link_or_copy(src, dst, mode=mode, overwrite=overwrite)
    return str(dst)


def _save_raw_csv_mask(mask: np.ndarray, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(dst)
    return str(dst)


def _read_session_summaries(batch_root: Path) -> list[dict[str, Any]]:
    return [read_json(path) for path in sorted((batch_root / "sessions").rglob("summary.json"))]


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TSGSS Dense Annotation Summary",
        "",
        "## Aggregate",
        f"- sessions: {summary['n_sessions']}",
        f"- sampled_frames: {summary['n_sampled_frames']}",
        f"- annotated_complete: {summary['label_status_counts'].get('annotated_complete', 0)}",
        f"- annotated_state_only: {summary['label_status_counts'].get('annotated_state_only', 0)}",
        f"- annotated_eye_only: {summary['label_status_counts'].get('annotated_eye_only', 0)}",
        f"- pending_annotation_sources: {summary['label_status_counts'].get('pending_annotation_sources', 0)}",
        "",
        "## Sources",
    ]
    for key, value in sorted((summary.get("annotation_source_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("## CSV Status")
    for key, value in sorted((summary.get("csv_status_counts") or {}).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = build_argparser().parse_args()
    batch_root = _resolve_output_root(args.batch_root)
    output_root = _resolve_output_root(args.output_root)
    if output_root.exists() and not args.overwrite:
        summary_path = output_root / "experiment_summary.json"
        if summary_path.exists():
            print(f"[REUSED] {summary_path}")
            print(summary_path.read_text(encoding="utf-8"))
            return
    output_root.mkdir(parents=True, exist_ok=True)

    session_summaries = _read_session_summaries(batch_root)
    sample_manifest_rows: list[dict[str, Any]] = []
    merged_rows: list[dict[str, Any]] = []
    missing_only_rows: list[dict[str, Any]] = []
    label_status_counts: Counter[str] = Counter()
    annotation_source_counts: Counter[str] = Counter()
    csv_status_counts: Counter[str] = Counter()

    for session in session_summaries:
        user_id = int(session["user_id"])
        eye = str(session["eye"])
        session_code = str(session["session_code"])
        label_dirs = _session_label_dirs(output_root, user_id=user_id, eye=eye, session_code=session_code)
        for dst in label_dirs.values():
            dst.mkdir(parents=True, exist_ok=True)

        raw_session_dir = Path(session["raw_session_dir"]).resolve()
        layout = discover_session_layout(raw_session_dir, user_id=user_id)
        csv_exists = layout.annotation_csv is not None and layout.annotation_csv.exists()
        csv_rows_by_filename = _load_csv_rows_by_filename(layout.annotation_csv)
        ellipse_annotations, _report = parse_via_csv_with_report(layout.annotation_csv) if csv_exists else ([], {})
        ellipse_by_filename = {str(item.frame_filename): item for item in ellipse_annotations}

        for frame in session.get("frames", []):
            frame_filename = str(frame["frame_filename"])
            csv_rows = csv_rows_by_filename.get(frame_filename, [])
            csv_status, csv_max_region_count = _csv_status(csv_rows, csv_exists=csv_exists)
            csv_status_counts[csv_status] += 1

            manifest_row = {
                "session_key": str(session["session_key"]),
                "user_id": user_id,
                "eye": eye,
                "session_code": session_code,
                "frame_filename": frame_filename,
                "frame_idx": int(frame["frame_idx"]),
                "timestamp_us": int(frame["timestamp_us"]),
                "sample_frame_path": str(frame.get("sample_frame_path")),
                "raw_frame_path": str(frame.get("raw_frame_path")),
                "sample_mode": str(session.get("options", {}).get("sample_mode", "unknown")),
                "csv_status": csv_status,
                "csv_max_region_count": csv_max_region_count,
            }
            sample_manifest_rows.append(manifest_row)

            eye_bbox = frame.get("eye_region_bbox_xywh_sensor")
            eye_mask_rel = None
            pupil_mask_rel = None
            pupil_bbox = None
            pupil_ellipse = None
            pupil_state_flag = None
            eye_state_flag = "unknown"
            closed_eye_flag = False
            annotation_quality = 0.0
            annotation_source = "unknown"
            annotation_reason = None
            label_status = "pending_annotation_sources"
            mask_valid = False
            valid_track = False
            fail_like = bool(frame.get("fail_like_union", False))
            review_required = bool(fail_like or str(frame.get("status")) in {"eye_failed", "pupil_failed"})

            if eye_bbox is None:
                label_status = "pending_annotation_sources"
                annotation_source = "groundedsam_eye_bbox"
                annotation_reason = "eye_roi_missing"
                if csv_status == "csv_region_zero":
                    pupil_state_flag = "closed"
                    eye_state_flag = "closed"
                    closed_eye_flag = True
            else:
                eye_mask_src = _copy_or_link_if_present(
                    frame.get("eye_region_mask_path"),
                    label_dirs["eye_masks"] / Path(frame_filename).with_suffix(".png").name,
                    mode=args.link_mode,
                    overwrite=bool(args.overwrite),
                )
                eye_mask_rel = None if eye_mask_src is None else _relative_to(output_root, Path(eye_mask_src))

                ellipse_ann = ellipse_by_filename.get(frame_filename)
                if ellipse_ann is not None:
                    raw_csv_mask = rasterize_ellipse_mask(ellipse_ann.ellipse_xywht, image_size=(346, 240)).astype(np.uint8) * 255
                    pupil_mask_abs = _save_raw_csv_mask(
                        raw_csv_mask,
                        label_dirs["pupil_masks"] / Path(frame_filename).with_suffix(".png").name,
                    )
                    pupil_mask_rel = _relative_to(output_root, Path(pupil_mask_abs))
                    pupil_ellipse = [float(v) for v in ellipse_ann.ellipse_xywht]
                    pupil_bbox = _ellipse_to_bbox_xywh(pupil_ellipse)
                    pupil_state_flag = "open"
                    eye_state_flag = "open"
                    closed_eye_flag = False
                    annotation_quality = 1.0
                    annotation_source = "raw_csv_positive"
                    annotation_reason = "raw_csv_positive"
                    label_status = "annotated_complete"
                    mask_valid = True
                    valid_track = True
                elif csv_status == "csv_region_zero":
                    pupil_state_flag = "closed"
                    eye_state_flag = "closed"
                    closed_eye_flag = True
                    annotation_quality = 0.25
                    annotation_source = "csv_region_zero_proxy"
                    annotation_reason = "csv_region_zero_proxy"
                    label_status = "annotated_state_only"
                elif str(frame.get("status")) == "completed" and frame.get("sensor_pupil_bbox_xywh") is not None:
                    pseudo_mask_src = _copy_or_link_if_present(
                        frame.get("pupil_mask_path"),
                        label_dirs["pupil_masks"] / Path(frame_filename).with_suffix(".png").name,
                        mode=args.link_mode,
                        overwrite=bool(args.overwrite),
                    )
                    pupil_mask_rel = None if pseudo_mask_src is None else _relative_to(output_root, Path(pseudo_mask_src))
                    pupil_bbox = [float(v) for v in frame["sensor_pupil_bbox_xywh"]]
                    pupil_ellipse = [float(v) for v in frame["sensor_pupil_ellipse_xywht"]]
                    pupil_state_flag = "open"
                    eye_state_flag = "open"
                    closed_eye_flag = False
                    predicted_conf = frame.get("predicted_box_confidence")
                    annotation_quality = float(predicted_conf) if predicted_conf is not None else 0.6
                    annotation_quality = max(0.05, min(annotation_quality, 0.75))
                    annotation_source = "pseudo_no_csv_completion"
                    annotation_reason = csv_status
                    label_status = "annotated_complete"
                    mask_valid = not fail_like
                    valid_track = not fail_like
                    if fail_like:
                        annotation_quality = min(annotation_quality, 0.25)
                else:
                    annotation_source = "groundedsam_eye_bbox"
                    annotation_reason = "pupil_source_missing" if csv_status == "csv_positive" else csv_status
                    label_status = "annotated_eye_only"
                    if csv_status == "csv_region_zero":
                        pupil_state_flag = "closed"
                        eye_state_flag = "closed"
                        closed_eye_flag = True

            merged_row = {
                "session_key": str(session["session_key"]),
                "user_id": user_id,
                "eye": eye,
                "session_code": session_code,
                "frame_filename": frame_filename,
                "frame_idx": int(frame["frame_idx"]),
                "timestamp_us": int(frame["timestamp_us"]),
                "sample_frame_path": str(frame.get("sample_frame_path")),
                "raw_frame_path": str(frame.get("raw_frame_path")),
                "eye_region_bbox_xywh_sensor": None if eye_bbox is None else [float(v) for v in eye_bbox],
                "eye_region_mask_path": eye_mask_rel,
                "eye_region_source": "groundedsam_eye_bbox",
                "pupil_mask_path": pupil_mask_rel,
                "pupil_bbox_xywh_sensor": pupil_bbox,
                "pupil_ellipse_xywht_sensor": pupil_ellipse,
                "pupil_state_flag": pupil_state_flag,
                "eye_state_flag": eye_state_flag,
                "closed_eye_flag": bool(closed_eye_flag),
                "annotation_source": annotation_source,
                "annotation_reason": annotation_reason,
                "annotation_quality": float(annotation_quality),
                "label_status": label_status,
                "csv_status": csv_status,
                "csv_max_region_count": csv_max_region_count,
                "predicted_class_name": frame.get("predicted_class_name"),
                "predicted_box_confidence": frame.get("predicted_box_confidence"),
                "fail_like": bool(fail_like),
                "review_required": bool(review_required),
                "mask_valid": bool(mask_valid),
                "valid_track": bool(valid_track),
                "source_status": str(frame.get("status")),
                "source_workspace_dir": str(session.get("workspace_dir")),
            }
            merged_rows.append(merged_row)
            label_status_counts[label_status] += 1
            annotation_source_counts[annotation_source] += 1

            if merged_row["pupil_bbox_xywh_sensor"] is None:
                missing_only_rows.append(merged_row)

    summary = {
        "experiment": "tsgss_dense_annotation",
        "batch_root": str(batch_root),
        "output_root": str(output_root),
        "n_sessions": len(session_summaries),
        "n_sampled_frames": len(merged_rows),
        "label_status_counts": dict(sorted(label_status_counts.items())),
        "annotation_source_counts": dict(sorted(annotation_source_counts.items())),
        "csv_status_counts": dict(sorted(csv_status_counts.items())),
        "n_missing_only": len(missing_only_rows),
        "sample_manifest_path": str(output_root / "sample_manifest.jsonl"),
        "merged_frame_annotations_path": str(output_root / "merged_frame_annotations.jsonl"),
        "missing_only_records_path": str(output_root / "missing_only_records.jsonl"),
    }

    write_jsonl(sample_manifest_rows, output_root / "sample_manifest.jsonl")
    write_jsonl(merged_rows, output_root / "merged_frame_annotations.jsonl")
    write_jsonl(missing_only_rows, output_root / "missing_only_records.jsonl")
    write_json(summary, output_root / "experiment_summary.json")
    (output_root / "experiment_summary.md").write_text(_summary_markdown(summary), encoding="utf-8")
    print(f"[DONE] sessions={summary['n_sessions']} sampled_frames={summary['n_sampled_frames']} output={output_root}")


if __name__ == "__main__":
    main()
