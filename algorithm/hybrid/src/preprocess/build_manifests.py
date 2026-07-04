from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.preprocess.path_utils import add_common_path_args, resolve_paths
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl
from src.utils.paths import resolve_canonical_dataset_root, resolve_canonical_indexes_root, resolve_stored_path


def _mode_defaults(data_mode: str) -> dict[str, str]:
    mode = str(data_mode).strip().lower()
    if mode == "mode0":
        return {
            "data_mode": "mode0",
            "canonical_name": "canonical0",
            "manifest_name": "manifest0",
            "frame_source": "original",
        }
    if mode == "mode2":
        return {
            "data_mode": "mode2",
            "canonical_name": "canonical2",
            "manifest_name": "manifest2",
            "frame_source": "interpolated",
        }
    return {
        "data_mode": "mode1",
        "canonical_name": "canonical1",
        "manifest_name": "manifest1",
        "frame_source": "original",
    }


def split_users_random(user_ids: list[int], train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, list[int]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")
    user_ids = sorted(user_ids)
    n = len(user_ids)
    n_train = max(1, int(round(n * train_ratio))) if n > 0 else 0
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    return {
        "train": user_ids[:n_train],
        "val": user_ids[n_train:n_train + n_val],
        "test": user_ids[n_train + n_val:],
    }


def split_users_exgaze(user_ids: list[int], with_val: bool = False) -> dict[str, list[int]]:
    user_ids = sorted(user_ids)
    train = [u for u in user_ids if 1 <= u <= 36]
    test = [u for u in user_ids if 37 <= u <= 48]
    if with_val:
        val = [u for u in train if 33 <= u <= 36]
        train = [u for u in train if u not in val]
    else:
        val = []
    return {"train": train, "val": val, "test": test}


def split_of_user(user_id: int, split_map: dict[str, list[int]]) -> str:
    for split, ids in split_map.items():
        if user_id in ids:
            return split
    return "train"


def _state6(annotation: dict[str, Any]) -> np.ndarray:
    if annotation.get("state_xyabuv") is not None:
        return np.asarray(annotation["state_xyabuv"], dtype=np.float32)
    ellipse = annotation.get("pupil_ellipse_xywht_sensor") or annotation.get("ellipse_sensor_xywht") or annotation.get("ellipse_xywht")
    if ellipse is None:
        return np.zeros((6,), dtype=np.float32)
    x, y, a, b, theta = [float(v) for v in ellipse]
    return np.asarray([x, y, a, b, math.sin(2.0 * theta), math.cos(2.0 * theta)], dtype=np.float32)


def _similarity_target(prev_ann: dict[str, Any], cur_ann: dict[str, Any]) -> float:
    prev_state = _state6(prev_ann)
    cur_state = _state6(cur_ann)
    center_dist = np.linalg.norm(prev_state[:2] - cur_state[:2])
    scale = max(1.0, 0.5 * (float(prev_state[2] + prev_state[3] + cur_state[2] + cur_state[3]) / 2.0))
    axes_term = np.mean(np.abs(np.log(np.maximum(cur_state[2:4], 1e-3) / np.maximum(prev_state[2:4], 1e-3))))
    angle_term = 1.0 - float(np.clip(np.dot(prev_state[4:6], cur_state[4:6]), -1.0, 1.0))
    score = 1.0 - np.clip(0.5 * (center_dist / scale) + 0.35 * axes_term + 0.15 * angle_term, 0.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def _annotation_ref(annotation: dict[str, Any], annotation_store_path: str) -> dict[str, str]:
    return {
        "ann_id": str(annotation["ann_id"]),
        "annotation_store_path": str(annotation_store_path),
    }


def _optional_str(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _annotation_sort_key(annotation: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(annotation.get("frame_index", annotation.get("frame_idx", -1))),
        int(annotation.get("sample_timestamp_us", annotation.get("frame_timestamp_us", annotation.get("timestamp_us", 0)))),
        str(annotation.get("frame_filename", annotation.get("ann_id", ""))),
    )


def _row_event_window(
    annotation: dict[str, Any],
    *,
    previous_annotation: dict[str, Any] | None,
    event_policy: str,
    time_bin_us: int,
    event_count_target: int,
    accumulation: str,
    causal_weight_power: float,
    fast_causal_limit: float,
) -> dict[str, Any]:
    end_timestamp_us = int(annotation.get("sample_timestamp_us", annotation.get("timestamp_us", 0)))
    if str(event_policy) == "interval_all":
        previous_effective = annotation if previous_annotation is None else previous_annotation
        start_timestamp_us = int(previous_effective.get("sample_timestamp_us", previous_effective.get("timestamp_us", end_timestamp_us)))
    else:
        start_timestamp_us = end_timestamp_us - int(time_bin_us)
    return {
        "policy": str(event_policy),
        "time_bin_us": int(time_bin_us),
        "event_count_target": int(event_count_target),
        "accumulation": str(accumulation),
        "causal_weight_power": float(causal_weight_power),
        "fast_causal_limit": float(fast_causal_limit),
        "start_timestamp_us": int(start_timestamp_us),
        "end_timestamp_us": int(end_timestamp_us),
    }


def _manifest_row(
    *,
    session: dict[str, Any],
    session_package: dict[str, Any],
    annotation_store_path: str,
    current: dict[str, Any],
    previous: dict[str, Any] | None,
    split: str,
    resize_policy: str,
    target_size_wh: tuple[int, int],
    event_policy: str,
    time_bin_us: int,
    event_count_target: int,
    accumulation: str,
    causal_weight_power: float,
    fast_causal_limit: float,
    data_mode: str,
    canonical_name: str,
    manifest_name: str,
    frame_source: str,
    synthetic_overlap_policy: str,
) -> dict[str, Any]:
    prev = current if previous is None else previous
    effective_data_mode = str(current.get("data_mode", session.get("data_mode", session_package.get("data_mode", data_mode))))
    effective_canonical_name = str(current.get("canonical_name", session.get("canonical_name", session_package.get("canonical_name", canonical_name))))
    effective_frame_source = str(current.get("frame_source", session.get("frame_source", session_package.get("frame_source", frame_source))))
    effective_timestamp_us = int(current.get("sample_timestamp_us", current.get("timestamp_us", 0)))
    effective_roi_xywh = (
        current.get("eye_region_bbox_xywh_sensor")
        or current.get("eye_region_xywh")
        or session.get("eye_region_xywh")
        or session_package.get("eye_region_xywh")
        or [0, 0, 346, 240]
    )
    row = {
        "sample_id": f"{session['session_key'].replace('/', '__')}__{Path(str(current.get('frame_filename', current['ann_id']))).stem}",
        "split": split,
        "subject_id": int(session.get("subject_id", session.get("user_id", -1))),
        "data_mode": effective_data_mode,
        "canonical_name": effective_canonical_name,
        "manifest_name": str(manifest_name),
        "frame_source": effective_frame_source,
        "sample_timestamp_us": effective_timestamp_us,
        "prev_sample_timestamp_us": int(prev.get("sample_timestamp_us", prev.get("timestamp_us", effective_timestamp_us))),
        "eye": str(session_package.get("eye", session.get("eye", "left"))),
        "session_key": str(session["session_key"]),
        "frame_path": _optional_str(current.get("frame_path")),
        "events_npz": _optional_str(session.get("events_npz")),
        "session_store_path": _optional_str(current.get("session_store_path", session.get("session_store_path", session_package.get("session_store_path")))),
        "session_store_format": _optional_str(current.get("session_store_format", session.get("session_store_format", session_package.get("session_store_format")))),
        "session_h5_path": _optional_str(current.get("session_h5_path", session.get("session_h5_path", session_package.get("session_h5_path")))),
        "session_npz_path": _optional_str(current.get("session_npz_path", session.get("session_npz_path", session_package.get("session_npz_path")))),
        "frame_index": current.get("frame_index"),
        "event_index_range": current.get("event_index_range"),
        "annotation_ref": _annotation_ref(current, annotation_store_path),
        "prev_annotation_ref": _annotation_ref(prev, annotation_store_path),
        "sensor_size_wh": list(session.get("sensor_size_wh", session_package.get("sensor_size_wh", [346, 240]))),
        "roi_xywh": list(effective_roi_xywh),
        "resize_policy": str(resize_policy),
        "target_size_wh": [int(target_size_wh[0]), int(target_size_wh[1])],
        "event_window": _row_event_window(
            current,
            previous_annotation=previous,
            event_policy=event_policy,
            time_bin_us=time_bin_us,
            event_count_target=event_count_target,
            accumulation=accumulation,
            causal_weight_power=causal_weight_power,
            fast_causal_limit=fast_causal_limit,
        ),
        "annotation_source": str(current.get("annotation_source", "manual")),
        "annotation_quality": float(current.get("annotation_quality", 1.0)),
        "label_status": str(current.get("label_status", current.get("annotation_status", "unknown"))),
        "annotation_status": str(current.get("annotation_status", current.get("label_status", "unknown"))),
        "annotation_reason": current.get("annotation_reason"),
        "roi_status": current.get("roi_status"),
        "pupil_status": current.get("pupil_status"),
        "eye_state_flag": current.get("eye_state_flag"),
        "similarity_target": _similarity_target(prev, current),
        "closed_eye_flag": bool(current.get("closed_eye_flag", False)),
        "blink_candidate_score": float(current.get("blink_candidate_score", 0.0)),
        "blink_candidate_source": current.get("blink_candidate_source"),
        "mask_valid": bool(current.get("mask_valid", True)),
        "valid_track": bool(previous is not None and not current.get("closed_eye_flag", False) and current.get("mask_valid", True)),
        "source_kind": current.get("source_kind"),
        "matched_raw_frame_index": current.get("matched_raw_frame_index"),
        "matched_raw_frame_filename": current.get("matched_raw_frame_filename"),
        "matched_raw_timestamp_us": current.get("matched_raw_timestamp_us"),
        "source_pair_index": current.get("source_pair_index") or current.get("interp_source_pair_index"),
        "source_pair_timestamps_us": current.get("source_pair_timestamps_us", current.get("interp_source_timestamps_us")),
        "session_store_layout": _optional_str(current.get("session_store_layout", session.get("session_store_layout", session_package.get("session_store_layout")))),
    }
    if effective_data_mode == "mode2" or current.get("synthetic_frame_flag") is not None:
        row["synthetic_frame_flag"] = bool(current.get("synthetic_frame_flag", False))
        row["interp_source_pair"] = current.get("interp_source_pair")
        row["interp_alpha"] = current.get("interp_alpha")
        row["interp_model"] = current.get("interp_model")
        row["interp_quality"] = current.get("interp_quality")
        row["interp_rank"] = current.get("interp_rank")
        row["interp_insert_count"] = current.get("interp_insert_count")
        row["interp_target_fps"] = current.get("interp_target_fps")
        row["interp_count_policy"] = current.get("interp_count_policy")
        row["synthetic_event_window"] = dict(current.get("synthetic_event_window") or row["event_window"])
        row["synthetic_overlap_policy"] = str(current.get("synthetic_overlap_policy", synthetic_overlap_policy))
        interpolation_ref = current.get("interpolation_ref") or session.get("interpolation_ref") or session_package.get("interpolation_ref")
        row["interpolation_ref"] = interpolation_ref
    return row


def build_manifests(
    *,
    canonical_root: str | Path,
    indexes_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    split_scheme: str = "exgaze_with_val",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    resize_policy: str = "facet_square_direct",
    target_size_wh: tuple[int, int] = (256, 256),
    event_policy: str = "fixed_count",
    time_bin_us: int = 5000,
    event_count_target: int = 5000,
    accumulation: str = "fast_causal_linear",
    causal_weight_power: float = 1.0,
    fast_causal_limit: float = 25.0,
    data_mode: str = "mode1",
    canonical_name: str | None = None,
    manifest_name: str | None = None,
    frame_source: str | None = None,
    synthetic_overlap_policy: str = "reuse_event_window",
) -> dict[str, Any]:
    mode_defaults = _mode_defaults(data_mode)
    resolved_data_mode = mode_defaults["data_mode"]
    resolved_canonical_name = canonical_name or mode_defaults["canonical_name"]
    resolved_manifest_name = manifest_name or mode_defaults["manifest_name"]
    resolved_frame_source = frame_source or mode_defaults["frame_source"]
    canonical_workspace_root = Path(canonical_root).resolve()
    canonical_root = resolve_canonical_dataset_root(canonical_workspace_root, resolved_canonical_name, prefer_nested=False).resolve()
    indexes_root = resolve_canonical_indexes_root(
        canonical_workspace_root,
        canonical_name=resolved_canonical_name,
        indexes_root=indexes_root,
        prefer_nested=False,
    ).resolve()
    manifests_root = (Path(manifests_root).resolve() if manifests_root is not None else Path.cwd() / "manifests")
    manifests_root = manifests_root / resolved_manifest_name
    manifests_root.mkdir(parents=True, exist_ok=True)

    session_rows = [row for row in read_jsonl(indexes_root / "sessions.jsonl") if not row.get("skipped")]
    skipped_session_rows = read_jsonl(indexes_root / "canonical_skipped_sessions.jsonl") if (indexes_root / "canonical_skipped_sessions.jsonl").exists() else []
    user_ids = sorted({int(row.get("subject_id", row.get("user_id", -1))) for row in session_rows})
    if split_scheme == "exgaze_with_val":
        split_map = split_users_exgaze(user_ids, with_val=True)
    elif split_scheme == "exgaze":
        split_map = split_users_exgaze(user_ids, with_val=False)
    elif split_scheme == "random":
        split_map = split_users_random(user_ids, train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    else:
        raise ValueError(f"Unsupported split_scheme: {split_scheme}")

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    annotation_failure_rows: list[dict[str, Any]] = []
    for session in session_rows:
        session_package = read_json(resolve_stored_path(canonical_root, session["session_package_path"]))
        annotation_store_path = str(session["annotation_store_path"])
        annotations = read_jsonl(resolve_stored_path(canonical_root, annotation_store_path))
        annotations = sorted(annotations, key=_annotation_sort_key)
        split = split_of_user(int(session.get("subject_id", session.get("user_id", -1))), split_map)
        previous = None
        for current in annotations:
            row = _manifest_row(
                session=session,
                session_package=session_package,
                annotation_store_path=annotation_store_path,
                current=current,
                previous=previous,
                split=split,
                resize_policy=resize_policy,
                target_size_wh=target_size_wh,
                event_policy=event_policy,
                time_bin_us=time_bin_us,
                event_count_target=event_count_target,
                accumulation=accumulation,
                causal_weight_power=causal_weight_power,
                fast_causal_limit=fast_causal_limit,
                data_mode=resolved_data_mode,
                canonical_name=resolved_canonical_name,
                manifest_name=resolved_manifest_name,
                frame_source=resolved_frame_source,
                synthetic_overlap_policy=synthetic_overlap_policy,
            )
            split_rows[split].append(row)
            if str(row.get("annotation_status", "")) != "annotated_complete":
                annotation_failure_rows.append(dict(row))
            previous = current

    for split, rows in split_rows.items():
        rows.sort(key=lambda row: (int(row["subject_id"]), str(row["session_key"]), str(row["sample_id"])))
        write_jsonl(rows, manifests_root / f"{split}_manifest.jsonl")
    annotation_failure_rows.sort(key=lambda row: (str(row.get("split", "")), int(row.get("subject_id", -1)), str(row.get("session_key", "")), str(row.get("sample_id", ""))))
    write_jsonl(skipped_session_rows, manifests_root / "skipped_sessions.jsonl")
    write_jsonl(annotation_failure_rows, manifests_root / "annotation_failures.jsonl")

    summary = {
        "canonical_root": str(canonical_root),
        "canonical_workspace_root": str(canonical_workspace_root),
        "indexes_root": str(indexes_root),
        "manifests_root": str(manifests_root),
        "data_mode": resolved_data_mode,
        "canonical_name": resolved_canonical_name,
        "manifest_name": resolved_manifest_name,
        "frame_source": resolved_frame_source,
        "split_scheme": split_scheme,
        "resize_policy": resize_policy,
        "target_size_wh": [int(target_size_wh[0]), int(target_size_wh[1])],
        "event_policy": event_policy,
        "time_bin_us": int(time_bin_us),
        "event_count_target": int(event_count_target),
        "accumulation": accumulation,
        "causal_weight_power": float(causal_weight_power),
        "fast_causal_limit": float(fast_causal_limit),
        "synthetic_overlap_policy": str(synthetic_overlap_policy),
        "n_skipped_sessions": int(len(skipped_session_rows)),
        "n_annotation_failures": int(len(annotation_failure_rows)),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
    }
    write_json(summary, manifests_root / "manifest_summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build HBTXR v3 manifests from canonical sessions")
    add_common_path_args(parser, need_canonical=True)
    parser.add_argument("--split-scheme", choices=["exgaze_with_val", "exgaze", "random"], default="exgaze_with_val")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--resize-policy", choices=["facet_square_direct", "letterbox_square", "sensor_full_square"], default="facet_square_direct")
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--event-policy", choices=["fixed_count", "time_bin", "interval_all"], default="fixed_count")
    parser.add_argument("--time-bin-us", type=int, default=5000)
    parser.add_argument("--event-count-target", type=int, default=5000)
    parser.add_argument("--accumulation", choices=["plain", "causal_linear", "fast_causal_linear"], default="fast_causal_linear")
    parser.add_argument("--causal-weight-power", type=float, default=1.0)
    parser.add_argument("--fast-causal-limit", type=float, default=25.0)
    parser.add_argument("--data-mode", choices=["mode0", "mode1", "mode2"], default="mode1")
    parser.add_argument("--canonical-name", type=str, default=None)
    parser.add_argument("--manifest-name", type=str, default=None)
    parser.add_argument("--frame-source", type=str, default=None)
    parser.add_argument("--synthetic-overlap-policy", type=str, default="reuse_event_window")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = resolve_paths(args, need_canonical=True)
    return build_manifests(
        canonical_root=paths.canonical_root,
        indexes_root=None if getattr(args, "indexes_root", None) is None else paths.indexes_root,
        manifests_root=paths.manifests_root,
        split_scheme=str(args.split_scheme),
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        resize_policy=str(args.resize_policy),
        target_size_wh=(int(args.input_width), int(args.input_height)),
        event_policy=str(args.event_policy),
        time_bin_us=int(args.time_bin_us),
        event_count_target=int(args.event_count_target),
        accumulation=str(args.accumulation),
        causal_weight_power=float(args.causal_weight_power),
        fast_causal_limit=float(args.fast_causal_limit),
        data_mode=str(args.data_mode),
        canonical_name=args.canonical_name,
        manifest_name=args.manifest_name,
        frame_source=args.frame_source,
        synthetic_overlap_policy=str(args.synthetic_overlap_policy),
    )


if __name__ == "__main__":
    summary = run(build_argparser().parse_args())
    print(summary)
