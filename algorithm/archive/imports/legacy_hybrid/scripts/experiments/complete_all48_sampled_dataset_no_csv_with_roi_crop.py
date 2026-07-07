#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import ensure_project_src_on_path

PROJECT_ROOT = ensure_project_src_on_path()

from hbtxr.preprocess.io_utils import rasterize_ellipse_mask
from hbtxr.utils.io import read_json, read_jsonl, write_json, write_jsonl


SENSOR_WIDTH = 346
SENSOR_HEIGHT = 240


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Complete sampled all-48 no-CSV pupil annotations by attaching ROI-crop Grounded-SAM "
            "pupil predictions from an existing preview batch on the exact same sampled frames."
        ),
    )
    parser.add_argument(
        "--base-root",
        type=str,
        default="workspace_session_samples_10/all48_dataset_construction_same8samples",
        help="Base sampled dataset-construction root.",
    )
    parser.add_argument(
        "--source-batch-root",
        type=str,
        default="workspace_session_samples_7/roi_crop_prompted_all48_v10_native_then_crop128_rescue_same8samples",
        help="Existing ROI-crop preview batch root used as the completion source.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10",
        help="Output root for the completed sampled dataset and analysis.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _abs_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _output_paths(output_root: Path, *, user_id: int, eye: str, session_code: str) -> dict[str, Path]:
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


def _session_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row["user_id"]), str(row["eye"]), str(row["session_code"]))


def _row_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    return (int(row["frame_idx"]), str(row["frame_filename"]))


def _load_source_frame_index(source_batch_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    frame_index: dict[tuple[str, str], dict[str, Any]] = {}
    for summary_path in sorted((source_batch_root / "sessions").rglob("summary.json")):
        summary = read_json(summary_path)
        session_key = str(summary["session_key"])
        for frame in summary.get("frames", []):
            frame_index[(session_key, str(frame["frame_filename"]))] = dict(frame)
    if not frame_index:
        raise FileNotFoundError(f"No frame summaries found under {source_batch_root}")
    return frame_index


def _write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# All-48 Sampled No-CSV Completion Summary",
        "",
        "## Aggregate",
        f"- sessions: {summary['n_sessions']}",
        f"- sampled_frames: {summary['n_sampled_frames']}",
        f"- eye_roi_present: {summary['n_eye_roi_present']}",
        f"- eye_roi_missing: {summary['n_eye_roi_missing']}",
        f"- pupil_geometry_complete_total: {summary['n_pupil_geometry_complete_total']}",
        f"- pupil_geometry_complete_raw_csv: {summary['n_pupil_geometry_complete_raw_csv']}",
        f"- pupil_geometry_complete_no_csv_completion: {summary['n_pupil_geometry_complete_no_csv_completion']}",
        f"- pupil_state_only: {summary['n_pupil_state_only']}",
        f"- pupil_related_missing: {summary['n_pupil_related_missing']}",
        "",
        "## No-CSV Completion",
        f"- attempted_rows: {summary['no_csv_completion']['n_attempted']}",
        f"- completed_rows: {summary['no_csv_completion']['n_completed']}",
        f"- clean_completed_rows: {summary['no_csv_completion']['n_completed_clean']}",
        f"- fail_like_completed_rows: {summary['no_csv_completion']['n_completed_fail_like']}",
        f"- unresolved_rows: {summary['no_csv_completion']['n_unresolved']}",
        "",
        "## No-CSV Completion Stage Counts",
    ]
    for key, value in summary["no_csv_completion"].get("stage_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Label Status Counts"])
    for key, value in summary.get("label_status_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Sessions By No-CSV Completion Fail-Like"])
    for row in summary.get("top_sessions_by_no_csv_completion_fail_like", []):
        lines.append(
            f"- {row['session_key']}: fail_like={row['n_no_csv_completion_fail_like']} "
            f"completed={row['n_no_csv_completion_completed']} unresolved={row['n_no_csv_completion_unresolved']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_analysis_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# No-CSV Completion Analysis",
        "",
        "## Aggregate",
        f"- attempted_rows: {summary['n_attempted']}",
        f"- completed_rows: {summary['n_completed']}",
        f"- clean_completed_rows: {summary['n_completed_clean']}",
        f"- fail_like_completed_rows: {summary['n_completed_fail_like']}",
        f"- unresolved_rows: {summary['n_unresolved']}",
        "",
        "## Status Counts",
    ]
    for key, value in summary.get("status_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Stage Counts"])
    for key, value in summary.get("stage_counts", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Top Fail-Like Sessions"])
    for row in summary.get("top_fail_like_sessions", []):
        lines.append(
            f"- {row['session_key']}: fail_like={row['n_fail_like']} clean={row['n_clean']} unresolved={row['n_unresolved']}"
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _copy_and_materialize_existing_annotations(
    row: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    updated = dict(row)
    filename = Path(str(row["frame_filename"])).with_suffix(".png").name

    eye_bbox = row.get("eye_region_bbox_xywh_sensor")
    if eye_bbox is not None:
        eye_mask = _rasterize_bbox_mask([float(v) for v in eye_bbox], image_size_wh=(SENSOR_WIDTH, SENSOR_HEIGHT))
        eye_mask_path = _save_binary_mask(eye_mask, paths["eye_mask_dir"] / filename)
        updated["eye_region_mask_path"] = eye_mask_path
        updated["eye_region_mask_valid"] = True
    else:
        updated["eye_region_mask_path"] = None
        updated["eye_region_mask_valid"] = False

    ellipse = row.get("pupil_ellipse_xywht_sensor")
    if ellipse is not None:
        pupil_mask = rasterize_ellipse_mask([float(v) for v in ellipse], image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
        pupil_mask_path = _save_binary_mask(pupil_mask, paths["pupil_mask_dir"] / filename)
        updated["pupil_mask_path"] = pupil_mask_path
        updated["mask_path"] = pupil_mask_path
        updated["mask_valid"] = True
    else:
        updated["pupil_mask_path"] = None
        updated["mask_path"] = None
        updated["mask_valid"] = False

    return updated


def _complete_missing_row(
    row: dict[str, Any],
    source_frame: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[dict[str, Any], str]:
    updated = _copy_and_materialize_existing_annotations(row, paths)
    filename = Path(str(row["frame_filename"])).with_suffix(".png").name
    completion_status = str(source_frame.get("status") or "missing_source_status")
    updated["no_csv_completion_attempted"] = True
    updated["no_csv_completion_source"] = "groundedsam_roi_crop_v10"
    updated["no_csv_completion_model_root"] = str(paths["source_batch_root"])
    updated["no_csv_completion_status"] = completion_status
    updated["no_csv_completion_fail_like_union"] = bool(source_frame.get("fail_like_union", False))
    updated["no_csv_completion_predicted_class_name"] = source_frame.get("predicted_class_name")
    updated["no_csv_completion_predicted_box_confidence"] = source_frame.get("predicted_box_confidence")
    updated["no_csv_completion_predicted_pupil_stage_name"] = source_frame.get("predicted_pupil_stage_name")
    updated["no_csv_completion_predicted_candidate_rank"] = source_frame.get("predicted_candidate_rank")
    updated["no_csv_completion_predicted_max_candidate_boxes"] = source_frame.get("predicted_max_candidate_boxes")

    if completion_status == "completed":
        ellipse = source_frame.get("sensor_pupil_ellipse_xywht")
        bbox = source_frame.get("sensor_pupil_bbox_xywh")
        if ellipse is None or bbox is None:
            completion_status = "completed_missing_geometry_payload"
            updated["no_csv_completion_status"] = completion_status
        else:
            pupil_mask = rasterize_ellipse_mask([float(v) for v in ellipse], image_size=(SENSOR_WIDTH, SENSOR_HEIGHT))
            pupil_mask_path = _save_binary_mask(pupil_mask, paths["pupil_mask_dir"] / filename)
            updated["pupil_state_flag"] = "open"
            updated["eye_state_flag"] = "open"
            updated["closed_eye_flag"] = False
            updated["pupil_ellipse_xywht_sensor"] = [float(v) for v in ellipse]
            updated["pupil_region_bbox_xywh_sensor"] = [float(v) for v in bbox]
            updated["pupil_bbox_xywh_sensor"] = [float(v) for v in bbox]
            updated["pupil_mask_path"] = pupil_mask_path
            updated["mask_path"] = pupil_mask_path
            updated["mask_valid"] = True
            updated["pupil_state_source"] = "groundedsam_roi_crop_v10_open_proxy"
            updated["pupil_bbox_source"] = "groundedsam_roi_crop_v10"
            updated["pupil_mask_source"] = "rasterized_from_groundedsam_roi_crop_v10_ellipse"
            updated["pupil_annotation_level"] = "geometry_complete"
            updated["label_status"] = "annotated_complete"
            updated["annotation_reason"] = "groundedsam_eye_and_groundedsam_roi_crop_pupil_geometry_no_csv_completion"
            updated["annotation_sources_used"] = dict(updated.get("annotation_sources_used") or {})
            updated["annotation_sources_used"]["pupil"] = "groundedsam_roi_crop_v10_no_csv_completion"
            updated["groundedsam_pupil_fail_like_union"] = bool(source_frame.get("fail_like_union", False))
            updated["groundedsam_pupil_predicted_stage_name"] = source_frame.get("predicted_pupil_stage_name")
            updated["groundedsam_pupil_predicted_class_name"] = source_frame.get("predicted_class_name")
            updated["groundedsam_pupil_predicted_box_confidence"] = source_frame.get("predicted_box_confidence")
            updated["groundedsam_pupil_predicted_candidate_rank"] = source_frame.get("predicted_candidate_rank")
            updated["groundedsam_pupil_predicted_max_candidate_boxes"] = source_frame.get("predicted_max_candidate_boxes")
            return updated, "completed_fail_like" if bool(source_frame.get("fail_like_union", False)) else "completed_clean"

    updated["pupil_annotation_level"] = "missing"
    updated["label_status"] = "annotated_eye_only"
    updated["annotation_reason"] = f"no_csv_completion_unresolved:{completion_status}"
    updated["annotation_sources_used"] = dict(updated.get("annotation_sources_used") or {})
    updated["annotation_sources_used"]["pupil"] = "groundedsam_roi_crop_v10_no_csv_completion_unresolved"
    updated["groundedsam_pupil_fail_like_union"] = bool(source_frame.get("fail_like_union", False))
    updated["groundedsam_pupil_predicted_stage_name"] = source_frame.get("predicted_pupil_stage_name")
    updated["groundedsam_pupil_predicted_class_name"] = source_frame.get("predicted_class_name")
    updated["groundedsam_pupil_predicted_box_confidence"] = source_frame.get("predicted_box_confidence")
    updated["groundedsam_pupil_predicted_candidate_rank"] = source_frame.get("predicted_candidate_rank")
    updated["groundedsam_pupil_predicted_max_candidate_boxes"] = source_frame.get("predicted_max_candidate_boxes")
    return updated, "unresolved"


def main() -> None:
    args = build_argparser().parse_args()
    base_root = _abs_path(args.base_root)
    source_batch_root = _abs_path(args.source_batch_root)
    output_root = _abs_path(args.output_root)
    summary_path = output_root / "experiment_summary.json"
    if summary_path.exists() and not args.overwrite:
        summary = read_json(summary_path)
        print(
            f"[DONE] reused_existing_summary summary={summary_path} "
            f"sampled_frames={summary['n_sampled_frames']} "
            f"completed_no_csv={summary['no_csv_completion']['n_completed']}"
        )
        return

    base_rows = read_jsonl(base_root / "merged_frame_annotations.jsonl")
    source_frame_index = _load_source_frame_index(source_batch_root)

    output_root.mkdir(parents=True, exist_ok=True)
    analysis_dir = output_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows_by_session: dict[str, list[dict[str, Any]]] = {}
    session_meta: dict[str, dict[str, Any]] = {}

    label_status_counts: Counter[str] = Counter()
    annotation_reason_counts: Counter[str] = Counter()
    no_csv_status_counts: Counter[str] = Counter()
    no_csv_stage_counts: Counter[str] = Counter()
    no_csv_fail_like_session_counts: Counter[str] = Counter()
    no_csv_clean_session_counts: Counter[str] = Counter()
    no_csv_unresolved_session_counts: Counter[str] = Counter()

    completion_records: list[dict[str, Any]] = []
    fail_like_records: list[dict[str, Any]] = []
    unresolved_records: list[dict[str, Any]] = []
    merged_rows_all: list[dict[str, Any]] = []

    for base_row in sorted(base_rows, key=lambda row: (_session_sort_key(row), _row_sort_key(row))):
        session_key = str(base_row["session_key"])
        user_id = int(base_row["user_id"])
        eye = str(base_row["eye"])
        session_code = str(base_row["session_code"])
        paths = _output_paths(output_root, user_id=user_id, eye=eye, session_code=session_code)
        paths["source_batch_root"] = source_batch_root
        session_meta.setdefault(
            session_key,
            {
                "session_key": session_key,
                "user_id": user_id,
                "eye": eye,
                "session_code": session_code,
                "rows_path": str(paths["rows_path"]),
                "summary_path": str(paths["summary_path"]),
                "n_sampled": 0,
                "n_eye_present": 0,
                "n_eye_missing": 0,
                "n_pupil_geometry_complete": 0,
                "n_pupil_geometry_complete_raw_csv": 0,
                "n_pupil_geometry_complete_no_csv_completion": 0,
                "n_pupil_state_only": 0,
                "n_pupil_related_missing": 0,
                "n_no_csv_completion_completed": 0,
                "n_no_csv_completion_fail_like": 0,
                "n_no_csv_completion_unresolved": 0,
            },
        )

        if str(base_row.get("label_status")) == "annotated_eye_only" and str(base_row.get("csv_status")) == "csv_file_missing":
            source_frame = source_frame_index[(session_key, str(base_row["frame_filename"]))]
            updated_row, outcome = _complete_missing_row(base_row, source_frame, paths)
            completion_record = {
                "session_key": session_key,
                "frame_filename": str(base_row["frame_filename"]),
                "frame_idx": int(base_row["frame_idx"]),
                "raw_frame_path": str(base_row["raw_frame_path"]),
                "outcome": outcome,
                "source_status": updated_row["no_csv_completion_status"],
                "fail_like_union": bool(updated_row.get("no_csv_completion_fail_like_union", False)),
                "predicted_pupil_stage_name": updated_row.get("no_csv_completion_predicted_pupil_stage_name"),
                "predicted_class_name": updated_row.get("no_csv_completion_predicted_class_name"),
                "predicted_box_confidence": updated_row.get("no_csv_completion_predicted_box_confidence"),
            }
            completion_records.append(completion_record)
            if outcome == "completed_fail_like":
                fail_like_records.append(dict(completion_record))
                no_csv_fail_like_session_counts[session_key] += 1
            elif outcome == "completed_clean":
                no_csv_clean_session_counts[session_key] += 1
            else:
                unresolved_records.append(dict(completion_record))
                no_csv_unresolved_session_counts[session_key] += 1
            no_csv_status_counts[outcome] += 1
            stage_key = str(updated_row.get("no_csv_completion_predicted_pupil_stage_name") or "none")
            no_csv_stage_counts[stage_key] += 1
        else:
            updated_row = _copy_and_materialize_existing_annotations(base_row, paths)

        label_status_counts[str(updated_row["label_status"])] += 1
        annotation_reason_counts[str(updated_row["annotation_reason"])] += 1

        meta = session_meta[session_key]
        meta["n_sampled"] += 1
        if updated_row.get("eye_region_mask_valid"):
            meta["n_eye_present"] += 1
        else:
            meta["n_eye_missing"] += 1
        if str(updated_row.get("pupil_annotation_level")) == "geometry_complete":
            meta["n_pupil_geometry_complete"] += 1
            if str(updated_row.get("pupil_bbox_source")) == "groundedsam_roi_crop_v10":
                meta["n_pupil_geometry_complete_no_csv_completion"] += 1
            else:
                meta["n_pupil_geometry_complete_raw_csv"] += 1
        elif str(updated_row.get("pupil_annotation_level")) == "state_only":
            meta["n_pupil_state_only"] += 1
        else:
            meta["n_pupil_related_missing"] += 1

        completion_outcome = str(updated_row.get("no_csv_completion_status") or "")
        if completion_outcome:
            if str(updated_row.get("pupil_annotation_level")) == "geometry_complete":
                meta["n_no_csv_completion_completed"] += 1
                if bool(updated_row.get("no_csv_completion_fail_like_union", False)):
                    meta["n_no_csv_completion_fail_like"] += 1
            else:
                meta["n_no_csv_completion_unresolved"] += 1

        rows_by_session.setdefault(session_key, []).append(updated_row)
        merged_rows_all.append(updated_row)

    session_summaries = []
    for session_key, rows in sorted(rows_by_session.items(), key=lambda item: _session_sort_key(item[1][0])):
        first = rows[0]
        meta = session_meta[session_key]
        paths = _output_paths(output_root, user_id=int(first["user_id"]), eye=str(first["eye"]), session_code=str(first["session_code"]))
        rows_sorted = sorted(rows, key=_row_sort_key)
        write_jsonl(rows_sorted, paths["rows_path"])
        session_summary = {
            **meta,
            "rows_path": str(paths["rows_path"]),
            "summary_path": str(paths["summary_path"]),
            "source_base_root": str(base_root),
            "source_batch_root": str(source_batch_root),
        }
        write_json(session_summary, paths["summary_path"])
        session_summaries.append(session_summary)

    sample_manifest_rows = read_jsonl(base_root / "sample_manifest.jsonl")
    sample_manifest_path = output_root / "sample_manifest.jsonl"
    merged_rows_path = output_root / "merged_frame_annotations.jsonl"
    session_summaries_path = output_root / "session_summaries.jsonl"
    completion_records_path = analysis_dir / "no_csv_completion_records.jsonl"
    fail_like_records_path = analysis_dir / "no_csv_completion_fail_like_records.jsonl"
    unresolved_records_path = analysis_dir / "no_csv_completion_unresolved_records.jsonl"

    write_jsonl(sample_manifest_rows, sample_manifest_path)
    write_jsonl(merged_rows_all, merged_rows_path)
    write_jsonl(session_summaries, session_summaries_path)
    write_jsonl(completion_records, completion_records_path)
    write_jsonl(fail_like_records, fail_like_records_path)
    write_jsonl(unresolved_records, unresolved_records_path)

    analysis_summary = {
        "n_attempted": int(sum(no_csv_status_counts.values())),
        "n_completed": int(no_csv_status_counts["completed_clean"] + no_csv_status_counts["completed_fail_like"]),
        "n_completed_clean": int(no_csv_status_counts["completed_clean"]),
        "n_completed_fail_like": int(no_csv_status_counts["completed_fail_like"]),
        "n_unresolved": int(no_csv_status_counts["unresolved"]),
        "status_counts": {key: int(no_csv_status_counts[key]) for key in sorted(no_csv_status_counts.keys())},
        "stage_counts": {key: int(no_csv_stage_counts[key]) for key in sorted(no_csv_stage_counts.keys())},
        "top_fail_like_sessions": [
            {
                "session_key": session_key,
                "n_fail_like": int(count),
                "n_clean": int(no_csv_clean_session_counts[session_key]),
                "n_unresolved": int(no_csv_unresolved_session_counts[session_key]),
            }
            for session_key, count in no_csv_fail_like_session_counts.most_common(20)
        ],
        "records_path": str(completion_records_path),
        "fail_like_records_path": str(fail_like_records_path),
        "unresolved_records_path": str(unresolved_records_path),
    }
    write_json(analysis_summary, analysis_dir / "no_csv_completion_summary.json")
    _write_analysis_markdown(analysis_summary, analysis_dir / "no_csv_completion_summary.md")

    summary = {
        "source_base_root": str(base_root),
        "source_batch_root": str(source_batch_root),
        "output_root": str(output_root),
        "n_sessions": int(len(session_summaries)),
        "n_sampled_frames": int(len(merged_rows_all)),
        "n_eye_roi_present": int(sum(item["n_eye_present"] for item in session_summaries)),
        "n_eye_roi_missing": int(sum(item["n_eye_missing"] for item in session_summaries)),
        "n_pupil_geometry_complete_total": int(sum(item["n_pupil_geometry_complete"] for item in session_summaries)),
        "n_pupil_geometry_complete_raw_csv": int(sum(item["n_pupil_geometry_complete_raw_csv"] for item in session_summaries)),
        "n_pupil_geometry_complete_no_csv_completion": int(sum(item["n_pupil_geometry_complete_no_csv_completion"] for item in session_summaries)),
        "n_pupil_state_only": int(sum(item["n_pupil_state_only"] for item in session_summaries)),
        "n_pupil_related_missing": int(sum(item["n_pupil_related_missing"] for item in session_summaries)),
        "label_status_counts": {key: int(label_status_counts[key]) for key in sorted(label_status_counts.keys())},
        "annotation_reason_counts": {key: int(annotation_reason_counts[key]) for key in sorted(annotation_reason_counts.keys())},
        "no_csv_completion": {
            "n_attempted": int(analysis_summary["n_attempted"]),
            "n_completed": int(analysis_summary["n_completed"]),
            "n_completed_clean": int(analysis_summary["n_completed_clean"]),
            "n_completed_fail_like": int(analysis_summary["n_completed_fail_like"]),
            "n_unresolved": int(analysis_summary["n_unresolved"]),
            "stage_counts": dict(analysis_summary["stage_counts"]),
        },
        "sample_manifest_path": str(sample_manifest_path),
        "merged_rows_path": str(merged_rows_path),
        "session_summaries_path": str(session_summaries_path),
        "analysis_summary_path": str((analysis_dir / "no_csv_completion_summary.json").resolve()),
        "top_sessions_by_no_csv_completion_fail_like": [
            {
                "session_key": session_key,
                "n_no_csv_completion_fail_like": int(count),
                "n_no_csv_completion_completed": int(no_csv_clean_session_counts[session_key] + no_csv_fail_like_session_counts[session_key]),
                "n_no_csv_completion_unresolved": int(no_csv_unresolved_session_counts[session_key]),
            }
            for session_key, count in no_csv_fail_like_session_counts.most_common(20)
        ],
    }
    write_json(summary, summary_path)
    _write_summary_markdown(summary, output_root / "experiment_summary.md")

    print(
        f"[DONE] sampled_frames={summary['n_sampled_frames']} "
        f"no_csv_attempted={summary['no_csv_completion']['n_attempted']} "
        f"no_csv_completed={summary['no_csv_completion']['n_completed']} "
        f"no_csv_fail_like={summary['no_csv_completion']['n_completed_fail_like']} "
        f"remaining_missing={summary['n_pupil_related_missing']} summary={summary_path}"
    )


if __name__ == "__main__":
    main()
