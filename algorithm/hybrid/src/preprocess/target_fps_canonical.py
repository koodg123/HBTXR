from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from src.preprocess.io_utils import canonical_user_name, ensure_dir, maybe_link_or_copy
from src.preprocess.path_utils import add_common_path_args, relativize_to, resolve_paths
from src.preprocess.progress import print_progress
from src.preprocess.protocol import derive_protocol_session_index, derive_session_motion_regime_prior
from src.preprocess.target_fps_build import resolve_target_fps_root
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl
from src.utils.paths import normalize_user_path, resolve_canonical_dataset_root, resolve_canonical_indexes_root


def _mode_defaults(data_mode: str) -> dict[str, str]:
    mode = str(data_mode).strip().lower()
    if mode == "mode2":
        return {
            "data_mode": "mode2",
            "canonical_name": "canonical2",
            "frame_source": "target_fps_grid",
        }
    return {
        "data_mode": "mode1",
        "canonical_name": "canonical1",
        "frame_source": "target_fps_grid",
    }


def _session_sort_key(row: dict[str, Any]) -> tuple[int, str, str, str]:
    return (
        int(row.get("user_id", -1)),
        str(row.get("eye", "")),
        str(row.get("session_code", "")),
        str(row.get("session_key", "")),
    )


def _annotation_sort_key(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("frame_store_index", row.get("frame_index", -1))),
        int(row.get("target_timestamp_us", row.get("frame_timestamp_us", 0))),
        str(row.get("target_frame_filename", row.get("frame_filename", row.get("ann_id", "")))),
    )


def _relativize_optional(root: Path, path_value: str | Path | None) -> str | None:
    if path_value in (None, ""):
        return None
    return relativize_to(root, Path(path_value))


def _union_eye_region_xywh(rows: list[dict[str, Any]], sensor_size_wh: list[int]) -> list[int]:
    boxes = [row.get("eye_region_bbox_xywh_sensor") for row in rows if row.get("eye_region_bbox_xywh_sensor") not in (None, [])]
    if not boxes:
        return [0, 0, int(sensor_size_wh[0]), int(sensor_size_wh[1])]
    x0 = min(float(box[0]) for box in boxes)
    y0 = min(float(box[1]) for box in boxes)
    x1 = max(float(box[0]) + float(box[2]) for box in boxes)
    y1 = max(float(box[1]) + float(box[3]) for box in boxes)
    width = max(1.0, x1 - x0)
    height = max(1.0, y1 - y0)
    return [int(round(x0)), int(round(y0)), int(round(width)), int(round(height))]


def _link_optional_dir(src: Path, dst: Path, *, link_mode: str, overwrite: bool) -> None:
    if not src.exists():
        return
    maybe_link_or_copy(src, dst, mode=link_mode, overwrite=overwrite)


def _remap_mask_path(
    stored_path: str | Path | None,
    *,
    source_dir: Path,
    canonical_dir: Path,
    canonical_root: Path,
) -> str | None:
    if stored_path in (None, ""):
        return None
    src = Path(stored_path)
    try:
        rel = src.resolve().relative_to(source_dir.resolve())
        return relativize_to(canonical_root, canonical_dir / rel)
    except Exception:
        return relativize_to(canonical_root, src)


def _build_canonical_row(
    *,
    row: dict[str, Any],
    canonical_root: Path,
    linked_session_store_path: Path,
    session_store_format: str,
    session_eye_mask_dir: Path,
    session_pupil_mask_dir: Path,
    source_eye_mask_dir: Path,
    source_pupil_mask_dir: Path,
    data_mode: str,
    canonical_name: str,
    frame_source: str,
    interpolation_backend: str,
    target_fps: float,
    session_store_layout: str,
) -> dict[str, Any]:
    current = dict(row)
    frame_filename = str(current.get("target_frame_filename") or current.get("frame_filename") or f"{int(current.get('frame_store_index', 0)):06d}.png")
    sample_timestamp_us = int(current.get("target_timestamp_us", current.get("frame_timestamp_us", 0)))
    ann_id = current.get("ann_id") or f"{str(current['session_key']).replace('/', '__')}__{Path(frame_filename).stem}"

    current["ann_id"] = str(ann_id)
    current["frame_filename"] = frame_filename
    current["frame_index"] = int(current.get("frame_store_index", current.get("frame_index", 0)))
    current["frame_timestamp_us"] = sample_timestamp_us
    current["sample_timestamp_us"] = sample_timestamp_us
    current["data_mode"] = str(data_mode)
    current["canonical_name"] = str(canonical_name)
    current["frame_source"] = str(frame_source)
    current["annotation_status"] = str(current.get("label_status", "pending_annotation"))
    current["session_store_path"] = relativize_to(canonical_root, linked_session_store_path)
    current["session_store_format"] = str(session_store_format)
    current["session_store_layout"] = str(session_store_layout)
    current["session_h5_path"] = relativize_to(canonical_root, linked_session_store_path) if str(session_store_format) == "h5" else None
    current["session_npz_path"] = relativize_to(canonical_root, linked_session_store_path) if str(session_store_format) == "npz" else None
    current["frame_path"] = None
    current["eye_region_mask_path"] = _remap_mask_path(
        current.get("eye_region_mask_path"),
        source_dir=source_eye_mask_dir,
        canonical_dir=session_eye_mask_dir,
        canonical_root=canonical_root,
    )
    current["pupil_mask_path"] = _remap_mask_path(
        current.get("pupil_mask_path"),
        source_dir=source_pupil_mask_dir,
        canonical_dir=session_pupil_mask_dir,
        canonical_root=canonical_root,
    )
    current["mask_path"] = current.get("pupil_mask_path")

    if str(data_mode) == "mode2":
        current["synthetic_frame_flag"] = bool(str(current.get("source_kind", "")).strip().lower() == "interpolated")
        current["interp_source_pair"] = current.get("source_pair_frame_filenames")
        current["interp_source_pair_index"] = current.get("source_pair_index")
        current["interp_source_timestamps_us"] = current.get("source_pair_timestamps_us")
        current["interp_model"] = str(interpolation_backend)
        current["interp_target_fps"] = float(target_fps)
        current["synthetic_event_window"] = {
            "event_index_range": list(current.get("event_index_range", [])),
            "event_count": int(current.get("event_count", 0)),
            "event_end_timestamp_us": current.get("event_end_timestamp_us"),
        }
        current["synthetic_overlap_policy"] = "target_fps_grid_reuse"

    return current


def canonicalize_target_fps_dataset(
    *,
    target_root: str | Path,
    target_fps: float,
    canonical_root: str | Path | None = None,
    indexes_root: str | Path | None = None,
    data_mode: str = "mode1",
    canonical_name: str | None = None,
    frame_source: str | None = None,
    overwrite: bool = False,
    link_mode: str = "symlink",
) -> dict[str, Any]:
    started_at = time.perf_counter()
    target_fps_root = resolve_target_fps_root(target_root, target_fps)
    mode_defaults = _mode_defaults(data_mode)
    resolved_data_mode = mode_defaults["data_mode"]
    resolved_canonical_name = canonical_name or mode_defaults["canonical_name"]
    resolved_frame_source = frame_source or mode_defaults["frame_source"]

    canonical_workspace_root = Path(canonical_root).resolve() if canonical_root is not None else target_fps_root
    if canonical_workspace_root == target_fps_root:
        canonical_root = (canonical_workspace_root / resolved_canonical_name).resolve()
        indexes_root = (
            normalize_user_path(indexes_root).resolve()
            if indexes_root is not None
            else (canonical_root / "indexes").resolve()
        )
    else:
        canonical_root = resolve_canonical_dataset_root(canonical_workspace_root, resolved_canonical_name, prefer_nested=True).resolve()
        indexes_root = resolve_canonical_indexes_root(
            canonical_workspace_root,
            canonical_name=resolved_canonical_name,
            indexes_root=indexes_root,
            prefer_nested=True,
        ).resolve()
    ensure_dir(canonical_root / "sessions")
    ensure_dir(indexes_root)

    source_indexes_root = target_fps_root / "indexes"
    sessions_index_path = source_indexes_root / "sessions.jsonl"
    skipped_index_path = source_indexes_root / "skipped_sessions.jsonl"
    jobs = read_jsonl(sessions_index_path) if sessions_index_path.exists() else []
    source_skipped = read_jsonl(skipped_index_path) if skipped_index_path.exists() else []

    print_progress(
        name="target_fps_canonical",
        status="start",
        started_at=started_at,
        total=int(len(jobs)),
        message="target-fps canonical bridge started",
        target_fps_root=str(target_fps_root),
        canonical_root=str(canonical_root),
        indexes_root=str(indexes_root),
        data_mode=str(resolved_data_mode),
        canonical_name=str(resolved_canonical_name),
        frame_source=str(resolved_frame_source),
    )

    session_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    annotation_failure_rows: list[dict[str, Any]] = []
    for row in source_skipped:
        skipped_rows.append(
            {
                **dict(row),
                "skipped": True,
                "data_mode": str(resolved_data_mode),
                "canonical_name": str(resolved_canonical_name),
                "frame_source": str(resolved_frame_source),
                "skip_stage": "target_fps_build",
            }
        )

    n_rows_written = 0
    for idx, job in enumerate(jobs, start=1):
        target_session_dir = Path(job["target_session_dir"])
        session_meta_path = target_session_dir / "session_meta.json"
        frame_labels_path = target_session_dir / "frame_labels.jsonl"
        if not session_meta_path.exists() or not frame_labels_path.exists():
            skipped_rows.append(
                {
                    "skipped": True,
                    "skip_reason": "target_fps_session_not_materialized",
                    "skip_stage": "target_fps_canonical",
                    "session_key": str(job["session_key"]),
                    "user_id": int(job["user_id"]),
                    "eye": str(job["eye"]),
                    "session_code": str(job["session_code"]),
                    "target_session_dir": str(target_session_dir),
                    "data_mode": str(resolved_data_mode),
                    "canonical_name": str(resolved_canonical_name),
                    "frame_source": str(resolved_frame_source),
                }
            )
            continue

        session_meta = read_json(session_meta_path)
        frame_labels = sorted(read_jsonl(frame_labels_path), key=_annotation_sort_key)
        if not frame_labels:
            skipped_rows.append(
                {
                    "skipped": True,
                    "skip_reason": "target_fps_frame_labels_empty",
                    "skip_stage": "target_fps_canonical",
                    "session_key": str(job["session_key"]),
                    "user_id": int(job["user_id"]),
                    "eye": str(job["eye"]),
                    "session_code": str(job["session_code"]),
                    "target_session_dir": str(target_session_dir),
                    "data_mode": str(resolved_data_mode),
                    "canonical_name": str(resolved_canonical_name),
                    "frame_source": str(resolved_frame_source),
                }
            )
            continue

        user_name = canonical_user_name(int(session_meta["user_id"]))
        session_dir = canonical_root / "sessions" / user_name / str(session_meta["eye"]) / f"session_{session_meta['session_code']}"
        labels_dir = session_dir / "labels"
        store_dir = session_dir / "store"
        ensure_dir(labels_dir)
        ensure_dir(store_dir)

        source_session_store_path = Path(session_meta["session_store_path"])
        linked_session_store_path = store_dir / source_session_store_path.name
        maybe_link_or_copy(source_session_store_path, linked_session_store_path, mode=link_mode, overwrite=overwrite)

        source_eye_mask_dir = target_session_dir / "labels" / "eye_region_masks"
        source_pupil_mask_dir = target_session_dir / "labels" / "pupil_masks"
        session_eye_mask_dir = labels_dir / "eye_region_masks"
        session_pupil_mask_dir = labels_dir / "pupil_masks"
        _link_optional_dir(source_eye_mask_dir, session_eye_mask_dir, link_mode=link_mode, overwrite=overwrite)
        _link_optional_dir(source_pupil_mask_dir, session_pupil_mask_dir, link_mode=link_mode, overwrite=overwrite)

        canonical_rows = [
            _build_canonical_row(
                row=current,
                canonical_root=canonical_root,
                linked_session_store_path=linked_session_store_path,
                session_store_format=str(session_meta["session_store_format"]),
                session_eye_mask_dir=session_eye_mask_dir,
                session_pupil_mask_dir=session_pupil_mask_dir,
                source_eye_mask_dir=source_eye_mask_dir,
                source_pupil_mask_dir=source_pupil_mask_dir,
                data_mode=resolved_data_mode,
                canonical_name=resolved_canonical_name,
                frame_source=resolved_frame_source,
                interpolation_backend=str(session_meta.get("interpolation_backend", "linear_blend")),
                target_fps=float(session_meta["target_fps"]),
                session_store_layout=str(session_meta.get("frame_storage_mode", "materialized_target_frames")),
            )
            for current in frame_labels
        ]
        frame_index_rows = [
            {
                "frame_index": int(row["frame_index"]),
                "frame_filename": str(row["frame_filename"]),
                "frame_timestamp_us": int(row["frame_timestamp_us"]),
                "frame_source": str(row.get("frame_source", resolved_frame_source)),
                "source_kind": row.get("source_kind"),
                "session_store_path": str(row["session_store_path"]),
                "session_store_format": str(row["session_store_format"]),
            }
            for row in canonical_rows
        ]

        ann_path = labels_dir / "frame_annotations.jsonl"
        frame_index_path = labels_dir / "frame_index.jsonl"
        session_package_path = labels_dir / "session_package.json"
        meta_path = session_dir / "meta.json"
        write_jsonl(canonical_rows, ann_path)
        write_jsonl(frame_index_rows, frame_index_path)
        annotation_failure_rows.extend(
            [
                {
                    "session_key": str(session_meta["session_key"]),
                    "user_id": int(session_meta["user_id"]),
                    "eye": str(session_meta["eye"]),
                    "session_code": str(session_meta["session_code"]),
                    "frame_index": int(row["frame_index"]),
                    "frame_filename": str(row["frame_filename"]),
                    "sample_timestamp_us": int(row["sample_timestamp_us"]),
                    "data_mode": str(resolved_data_mode),
                    "canonical_name": str(resolved_canonical_name),
                    "label_status": str(row.get("label_status", "unknown")),
                    "annotation_status": str(row.get("annotation_status", row.get("label_status", "unknown"))),
                    "annotation_reason": row.get("annotation_reason"),
                    "roi_status": row.get("roi_status"),
                    "pupil_status": row.get("pupil_status"),
                    "eye_state_flag": row.get("eye_state_flag"),
                    "closed_eye_flag": bool(row.get("closed_eye_flag", False)),
                    "annotation_store_path": relativize_to(canonical_root, ann_path),
                    "frame_index_path": relativize_to(canonical_root, frame_index_path),
                }
                for row in canonical_rows
                if str(row.get("annotation_status", row.get("label_status", ""))) != "annotated_complete"
            ]
        )

        raw_session_dir = Path(session_meta.get("raw_session_dir", target_session_dir))
        protocol_session_index, _ = derive_protocol_session_index(raw_session_dir.name, str(session_meta["session_code"]))
        session_motion_regime_prior = derive_session_motion_regime_prior(protocol_session_index)
        sensor_size_wh = list(session_meta.get("frame_shape_hw", [240, 346]))
        sensor_size_wh = [int(sensor_size_wh[1]), int(sensor_size_wh[0])] if len(sensor_size_wh) == 2 else [346, 240]
        eye_region_xywh = _union_eye_region_xywh(canonical_rows, sensor_size_wh)

        session_package = {
            "session_key": str(session_meta["session_key"]),
            "user_id": int(session_meta["user_id"]),
            "subject_id": int(session_meta["user_id"]),
            "eye": str(session_meta["eye"]),
            "session_code": str(session_meta["session_code"]),
            "data_mode": str(resolved_data_mode),
            "canonical_name": str(resolved_canonical_name),
            "frame_source": str(resolved_frame_source),
            "protocol_session_index": protocol_session_index,
            "sensor_size_wh": list(sensor_size_wh),
            "eye_region_xywh": list(eye_region_xywh),
            "frame_source_size_wh": list(sensor_size_wh),
            "event_source_size_wh": list(sensor_size_wh),
            "events_npz": None,
            "frame_index_path": relativize_to(canonical_root, frame_index_path),
            "annotation_store_path": relativize_to(canonical_root, ann_path),
            "session_store_path": relativize_to(canonical_root, linked_session_store_path),
            "session_h5_path": relativize_to(canonical_root, linked_session_store_path) if str(session_meta["session_store_format"]) == "h5" else None,
            "session_store_format": str(session_meta["session_store_format"]),
            "session_store_layout": str(session_meta.get("frame_storage_mode", "materialized_target_frames")),
            "split_policy": "subject_independent_exgaze_with_val",
            "session_motion_regime_prior": dict(session_motion_regime_prior),
            "n_frames": int(len(canonical_rows)),
            "n_labelled_frames": int(len(canonical_rows)),
            "target_fps": float(session_meta["target_fps"]),
            "target_step_us": int(session_meta["target_step_us"]),
        }
        write_json(session_package, session_package_path)

        meta = {
            "user_id": int(session_meta["user_id"]),
            "subject_id": int(session_meta["user_id"]),
            "eye": str(session_meta["eye"]),
            "session_code": str(session_meta["session_code"]),
            "session_dir_name": raw_session_dir.name,
            "session_key": str(session_meta["session_key"]),
            "protocol_session_index": protocol_session_index,
            "session_motion_regime_prior": dict(session_motion_regime_prior),
            "session_package_path": relativize_to(canonical_root, session_package_path),
            "sensor_size_wh": list(sensor_size_wh),
            "frame_source_size_wh": list(sensor_size_wh),
            "event_source_size_wh": list(sensor_size_wh),
            "eye_region_xywh": list(eye_region_xywh),
            "n_frames": int(len(canonical_rows)),
            "n_labelled_frames": int(len(canonical_rows)),
            "annotation_store_path": relativize_to(canonical_root, ann_path),
            "events_npz": None,
            "raw_session_dir": str(raw_session_dir),
            "target_session_dir": str(target_session_dir),
            "source_target_fps_root": str(target_fps_root),
            "target_fps": float(session_meta["target_fps"]),
            "target_step_us": int(session_meta["target_step_us"]),
            "annotation_report": dict(session_meta.get("annotation_report", {})),
            "data_mode": str(resolved_data_mode),
            "canonical_name": str(resolved_canonical_name),
            "frame_source": str(resolved_frame_source),
            "session_store_path": relativize_to(canonical_root, linked_session_store_path),
            "session_h5_path": relativize_to(canonical_root, linked_session_store_path) if str(session_meta["session_store_format"]) == "h5" else None,
            "session_store_format": str(session_meta["session_store_format"]),
            "session_store_layout": str(session_meta.get("frame_storage_mode", "materialized_target_frames")),
            "interpolation_backend": str(session_meta.get("interpolation_backend", "linear_blend")),
            "n_rows_annotated_complete": int(session_meta.get("n_rows_annotated_complete", 0)),
            "n_rows_annotated_eye_only": int(session_meta.get("n_rows_annotated_eye_only", 0)),
            "n_rows_pending_annotation": int(session_meta.get("n_rows_pending_annotation", 0)),
        }
        write_json(meta, meta_path)

        session_rows.append(
            {
                "skipped": False,
                "user_id": int(session_meta["user_id"]),
                "subject_id": int(session_meta["user_id"]),
                "eye": str(session_meta["eye"]),
                "session_code": str(session_meta["session_code"]),
                "session_dir_name": raw_session_dir.name,
                "session_key": str(session_meta["session_key"]),
                "protocol_session_index": protocol_session_index,
                "session_motion_regime_prior": dict(session_motion_regime_prior),
                "session_package_path": relativize_to(canonical_root, session_package_path),
                "sensor_size_wh": list(sensor_size_wh),
                "frame_source_size_wh": list(sensor_size_wh),
                "event_source_size_wh": list(sensor_size_wh),
                "eye_region_xywh": list(eye_region_xywh),
                "n_frames": int(len(canonical_rows)),
                "n_labelled_frames": int(len(canonical_rows)),
                "annotation_store_path": relativize_to(canonical_root, ann_path),
                "events_npz": None,
                "session_store_path": relativize_to(canonical_root, linked_session_store_path),
                "session_h5_path": relativize_to(canonical_root, linked_session_store_path) if str(session_meta["session_store_format"]) == "h5" else None,
                "session_store_format": str(session_meta["session_store_format"]),
                "session_store_layout": str(session_meta.get("frame_storage_mode", "materialized_target_frames")),
                "frame_index_path": relativize_to(canonical_root, frame_index_path),
                "raw_session_dir": str(raw_session_dir),
                "target_session_dir": str(target_session_dir),
                "data_mode": str(resolved_data_mode),
                "canonical_name": str(resolved_canonical_name),
                "frame_source": str(resolved_frame_source),
                "target_fps": float(session_meta["target_fps"]),
                "target_step_us": int(session_meta["target_step_us"]),
                "annotation_report": dict(session_meta.get("annotation_report", {})),
            }
        )
        n_rows_written += len(canonical_rows)
        print_progress(
            name="target_fps_canonical",
            status="progress",
            started_at=started_at,
            completed=int(idx),
            total=int(len(jobs)),
            message="target-fps session bridged into canonical rows",
            session_key=str(session_meta["session_key"]),
        )

    session_rows.sort(key=_session_sort_key)
    skipped_rows.sort(key=lambda row: _session_sort_key(row) if not row.get("skipped") else (int(row.get("user_id", -1)), str(row.get("eye", "")), str(row.get("session_code", "")), str(row.get("session_key", ""))))
    annotation_failure_rows.sort(key=lambda row: (int(row.get("user_id", -1)), str(row.get("eye", "")), str(row.get("session_code", "")), int(row.get("frame_index", -1))))
    write_jsonl(session_rows, indexes_root / "sessions.jsonl")
    write_jsonl(skipped_rows, indexes_root / "canonical_skipped_sessions.jsonl")
    write_jsonl(annotation_failure_rows, indexes_root / "canonical_annotation_failures.jsonl")

    summary = {
        "stage": "target_fps_canonicalized",
        "target_fps_root": str(target_fps_root),
        "canonical_root": str(canonical_root),
        "indexes_root": str(indexes_root),
        "data_mode": str(resolved_data_mode),
        "canonical_name": str(resolved_canonical_name),
        "frame_source": str(resolved_frame_source),
        "n_sessions_canonicalized": int(len(session_rows)),
        "n_sessions_skipped": int(len(skipped_rows)),
        "n_rows_written": int(n_rows_written),
        "n_annotation_failures": int(len(annotation_failure_rows)),
        "target_fps": float(target_fps),
    }
    write_json(summary, canonical_root / "canonical_summary.json")

    print_progress(
        name="target_fps_canonical",
        status="done",
        started_at=started_at,
        completed=int(len(session_rows)),
        total=int(len(jobs)),
        message="target-fps canonical bridge complete",
        n_sessions_canonicalized=summary["n_sessions_canonicalized"],
        n_rows_written=summary["n_rows_written"],
        n_sessions_skipped=summary["n_sessions_skipped"],
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge target-fps session stores into canonical rows")
    add_common_path_args(parser, need_canonical=False)
    parser.add_argument("--target-root", type=str, required=True, help="Target dataset workspace root used by build_target_fps_dataset")
    parser.add_argument("--target-fps", type=float, required=True)
    parser.add_argument("--data-mode", choices=["mode1", "mode2"], default="mode1")
    parser.add_argument("--canonical-name", type=str, default=None)
    parser.add_argument("--frame-source", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link-mode", choices=["symlink", "copy", "skip"], default="symlink")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(args, need_canonical=False)
    canonical_root = args.canonical_root if getattr(args, "canonical_root", None) is not None else None
    if canonical_root is None and paths.canonical_root is not None:
        canonical_root = str(paths.canonical_root)
    return canonicalize_target_fps_dataset(
        target_root=str(args.target_root),
        target_fps=float(args.target_fps),
        canonical_root=canonical_root,
        indexes_root=getattr(args, "indexes_root", None),
        data_mode=str(args.data_mode),
        canonical_name=args.canonical_name,
        frame_source=args.frame_source,
        overwrite=bool(args.overwrite),
        link_mode=str(args.link_mode),
    )


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(summary)
