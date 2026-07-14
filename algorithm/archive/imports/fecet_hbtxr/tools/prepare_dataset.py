from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT

from fecet_hbtxr.io import read_json, read_jsonl, relativize_to, resolve_stored_path, write_json, write_jsonl, xywht_to_xyabuv


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


def _state6(annotation: dict[str, Any]) -> list[float]:
    if annotation.get("state_xyabuv") is not None:
        return [float(v) for v in annotation["state_xyabuv"]]
    ellipse = annotation.get("pupil_ellipse_xywht_sensor") or annotation.get("ellipse_sensor_xywht") or annotation.get("ellipse_xywht")
    if ellipse is None:
        return [0.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    return xywht_to_xyabuv(ellipse).tolist()


def _similarity_target(prev_ann: dict[str, Any], cur_ann: dict[str, Any]) -> float:
    prev_state = _state6(prev_ann)
    cur_state = _state6(cur_ann)
    center_dist = math.sqrt((prev_state[0] - cur_state[0]) ** 2 + (prev_state[1] - cur_state[1]) ** 2)
    scale = max(1.0, 0.5 * (float(prev_state[2] + prev_state[3] + cur_state[2] + cur_state[3]) / 2.0))
    axes_term = 0.5 * (
        abs(math.log(max(cur_state[2], 1e-3) / max(prev_state[2], 1e-3)))
        + abs(math.log(max(cur_state[3], 1e-3) / max(prev_state[3], 1e-3)))
    )
    angle_term = 1.0 - max(-1.0, min(1.0, prev_state[4] * cur_state[4] + prev_state[5] * cur_state[5]))
    score = 1.0 - max(0.0, min(1.0, 0.5 * (center_dist / scale) + 0.35 * axes_term + 0.15 * angle_term))
    return max(0.0, min(1.0, score))


def _annotation_ref(annotation: dict[str, Any], annotation_store_path: str) -> dict[str, str]:
    return {
        "ann_id": str(annotation["ann_id"]),
        "annotation_store_path": str(annotation_store_path),
    }


def _row_event_window(annotation: dict[str, Any], *, event_policy: str, time_bin_us: int, event_count_target: int, accumulation: str, causal_weight_power: float) -> dict[str, Any]:
    end_timestamp_us = int(annotation.get("timestamp_us", 0))
    start_timestamp_us = end_timestamp_us - int(time_bin_us)
    return {
        "policy": str(event_policy),
        "time_bin_us": int(time_bin_us),
        "event_count_target": int(event_count_target),
        "accumulation": str(accumulation),
        "causal_weight_power": float(causal_weight_power),
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
) -> dict[str, Any]:
    prev = current if previous is None else previous
    return {
        "sample_id": f"{session['session_key'].replace('/', '__')}__{Path(str(current.get('frame_filename', current['ann_id']))).stem}",
        "split": split,
        "subject_id": int(session.get("subject_id", session.get("user_id", -1))),
        "eye": str(session_package.get("eye", session.get("eye", "left"))),
        "session_key": str(session["session_key"]),
        "frame_path": str(current["frame_path"]),
        "events_npz": str(session["events_npz"]),
        "annotation_ref": _annotation_ref(current, annotation_store_path),
        "prev_annotation_ref": _annotation_ref(prev, annotation_store_path),
        "sensor_size_wh": list(session.get("sensor_size_wh", session_package.get("sensor_size_wh", [346, 240]))),
        "roi_xywh": list(current.get("eye_region_bbox_xywh_sensor", current.get("eye_region_xywh", session.get("eye_region_xywh", [0, 0, 346, 240])))),
        "resize_policy": str(resize_policy),
        "target_size_wh": [int(target_size_wh[0]), int(target_size_wh[1])],
        "event_window": _row_event_window(
            current,
            event_policy=event_policy,
            time_bin_us=time_bin_us,
            event_count_target=event_count_target,
            accumulation=accumulation,
            causal_weight_power=causal_weight_power,
        ),
        "annotation_source": str(current.get("annotation_source", "manual")),
        "annotation_quality": float(current.get("annotation_quality", 1.0)),
        "similarity_target": _similarity_target(prev, current),
        "closed_eye_flag": bool(current.get("closed_eye_flag", False)),
        "mask_valid": bool(current.get("mask_valid", True)),
        "valid_track": bool(previous is not None and not current.get("closed_eye_flag", False) and current.get("mask_valid", True)),
    }


def _discover_session_package_paths(root: Path) -> list[Path]:
    return sorted(root.glob("**/labels/session_package.json"))


def _resolve_known_path(primary_root: Path, secondary_root: Path, raw_path: str | Path) -> Path:
    primary = resolve_stored_path(primary_root, raw_path)
    if primary.exists():
        return primary
    return resolve_stored_path(secondary_root, raw_path)


def ingest_source_tree(source_root: str | Path, canonical_root: str | Path, *, mode: str = "none") -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    canonical_root = Path(canonical_root).resolve()
    canonical_root.mkdir(parents=True, exist_ok=True)
    if mode == "none" or source_root == canonical_root:
        return {"ingest_mode": mode, "mirrored_sessions": 0}

    mirrored = 0
    for package_path in _discover_session_package_paths(source_root):
        session_root = package_path.parent.parent
        rel_session = session_root.relative_to(source_root)
        dst_session = canonical_root / rel_session
        if dst_session.exists():
            continue
        dst_session.parent.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copytree(session_root, dst_session)
        elif mode == "symlink":
            try:
                dst_session.symlink_to(session_root, target_is_directory=True)
            except OSError:
                shutil.copytree(session_root, dst_session)
                mode = "copy-fallback"
        else:
            raise ValueError(f"Unsupported ingest mode: {mode}")
        mirrored += 1
    return {"ingest_mode": mode, "mirrored_sessions": mirrored}


def build_session_index(*, canonical_root: str | Path, indexes_root: str | Path | None = None) -> dict[str, Any]:
    canonical_root = Path(canonical_root).resolve()
    indexes_root = canonical_root / "indexes" if indexes_root is None else Path(indexes_root).resolve()
    indexes_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for package_path in _discover_session_package_paths(canonical_root):
        session_root = package_path.parent.parent
        package = read_json(package_path)
        annotation_store_abs = _resolve_known_path(canonical_root, package_path.parent, package.get("annotation_store_path", "frame_annotations.jsonl"))
        events_abs = _resolve_known_path(canonical_root, session_root, package.get("events_npz", "events/events.npz"))
        subject_id = int(package.get("subject_id", package.get("user_id", -1)))
        row = {
            "session_key": str(package.get("session_key") or "/".join(session_root.relative_to(canonical_root).parts)),
            "subject_id": subject_id,
            "user_id": int(package.get("user_id", subject_id)),
            "eye": str(package.get("eye", session_root.parent.name if session_root.parent.name else "left")),
            "session_code": str(package.get("session_code", session_root.name)),
            "session_package_path": relativize_to(canonical_root, package_path),
            "annotation_store_path": relativize_to(canonical_root, annotation_store_abs),
            "events_npz": relativize_to(canonical_root, events_abs),
            "sensor_size_wh": list(package.get("sensor_size_wh", [346, 240])),
            "eye_region_xywh": list(package.get("eye_region_xywh", [0, 0, 346, 240])),
            "skipped": not annotation_store_abs.exists() or not events_abs.exists(),
        }
        rows.append(row)

    rows.sort(key=lambda row: (int(row["subject_id"]), str(row["session_key"])))
    write_jsonl(rows, indexes_root / "sessions.jsonl")
    summary = {"canonical_root": str(canonical_root), "indexes_root": str(indexes_root), "session_count": len(rows)}
    write_json(summary, indexes_root / "sessions_summary.json")
    return summary


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
    accumulation: str = "causal_linear",
    causal_weight_power: float = 1.0,
) -> dict[str, Any]:
    canonical_root = Path(canonical_root).resolve()
    indexes_root = canonical_root / "indexes" if indexes_root is None else Path(indexes_root).resolve()
    manifests_root = Path.cwd() / "manifests" if manifests_root is None else Path(manifests_root).resolve()
    manifests_root.mkdir(parents=True, exist_ok=True)

    session_rows = [row for row in read_jsonl(indexes_root / "sessions.jsonl") if not row.get("skipped")]
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
    for session in session_rows:
        session_package = read_json(resolve_stored_path(canonical_root, session["session_package_path"]))
        annotation_store_path = str(session["annotation_store_path"])
        annotations = read_jsonl(resolve_stored_path(canonical_root, annotation_store_path))
        annotations = sorted(annotations, key=lambda row: (int(row.get("frame_idx") or -1), int(row.get("timestamp_us") or 0), str(row["ann_id"])))
        split = split_of_user(int(session.get("subject_id", session.get("user_id", -1))), split_map)
        previous = None
        for current in annotations:
            split_rows[split].append(
                _manifest_row(
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
                )
            )
            previous = current

    for split, rows in split_rows.items():
        rows.sort(key=lambda row: (int(row["subject_id"]), str(row["session_key"]), str(row["sample_id"])))
        write_jsonl(rows, manifests_root / f"{split}_manifest.jsonl")

    summary = {
        "canonical_root": str(canonical_root),
        "indexes_root": str(indexes_root),
        "manifests_root": str(manifests_root),
        "split_scheme": split_scheme,
        "resize_policy": resize_policy,
        "target_size_wh": [int(target_size_wh[0]), int(target_size_wh[1])],
        "event_policy": event_policy,
        "time_bin_us": int(time_bin_us),
        "event_count_target": int(event_count_target),
        "accumulation": accumulation,
        "causal_weight_power": float(causal_weight_power),
        "counts": {split: len(rows) for split, rows in split_rows.items()},
    }
    write_json(summary, manifests_root / "manifest_summary.json")
    return summary


def build_split_views(*, manifests_root: str | Path, splits_root: str | Path) -> dict[str, Any]:
    manifests_root = Path(manifests_root).resolve()
    splits_root = Path(splits_root).resolve()
    splits_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for split in ("train", "val", "test"):
        split_dir = splits_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifests_root / f"{split}_manifest.jsonl"
        link_path = split_dir / "manifest.jsonl"
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        linked = False
        try:
            link_path.symlink_to(manifest_path)
            linked = True
        except OSError:
            linked = False
        rows = read_jsonl(manifest_path) if manifest_path.exists() else []
        index_payload = {
            "split": split,
            "manifest_path": str(manifest_path),
            "is_symlink_view": linked,
            "count": len(rows),
        }
        write_json(index_payload, split_dir / "index.json")
        summary[split] = index_payload
    return summary


def prepare_dataset(
    *,
    source_root: str | Path | None = None,
    canonical_root: str | Path,
    indexes_root: str | Path | None = None,
    manifests_root: str | Path | None = None,
    splits_root: str | Path | None = None,
    ingest_mode: str = "none",
    split_scheme: str = "exgaze_with_val",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    resize_policy: str = "facet_square_direct",
    target_size_wh: tuple[int, int] = (256, 256),
    event_policy: str = "fixed_count",
    time_bin_us: int = 5000,
    event_count_target: int = 5000,
    accumulation: str = "causal_linear",
    causal_weight_power: float = 1.0,
) -> dict[str, Any]:
    canonical_root = Path(canonical_root).resolve()
    indexes_root = canonical_root / "indexes" if indexes_root is None else Path(indexes_root).resolve()
    manifests_root = PROJECT_ROOT / "data" / "_internal" / "manifests" if manifests_root is None else Path(manifests_root).resolve()
    splits_root = PROJECT_ROOT / "data" / "splits" if splits_root is None else Path(splits_root).resolve()

    if source_root is not None:
        ingest_summary = ingest_source_tree(source_root, canonical_root, mode=ingest_mode)
    else:
        ingest_summary = {"ingest_mode": "none", "mirrored_sessions": 0}

    index_summary = build_session_index(canonical_root=canonical_root, indexes_root=indexes_root)
    manifest_summary = build_manifests(
        canonical_root=canonical_root,
        indexes_root=indexes_root,
        manifests_root=manifests_root,
        split_scheme=split_scheme,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        resize_policy=resize_policy,
        target_size_wh=target_size_wh,
        event_policy=event_policy,
        time_bin_us=time_bin_us,
        event_count_target=event_count_target,
        accumulation=accumulation,
        causal_weight_power=causal_weight_power,
    )
    split_summary = build_split_views(manifests_root=manifests_root, splits_root=splits_root)
    summary = {
        "ingest": ingest_summary,
        "index": index_summary,
        "manifests": manifest_summary,
        "split_views": split_summary,
    }
    write_json(summary, manifests_root / "prepare_summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FACET-style event/ellipse data for the simplified FECET-HBTXR project")
    parser.add_argument("--source-root", type=str, default=None, help="Optional raw/canonical source root to ingest")
    parser.add_argument("--canonical-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "canonical"))
    parser.add_argument("--indexes-root", type=str, default=None)
    parser.add_argument("--manifests-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "manifests"))
    parser.add_argument("--splits-root", type=str, default=str(PROJECT_ROOT / "data" / "splits"))
    parser.add_argument("--ingest-mode", choices=["none", "copy", "symlink"], default="none")
    parser.add_argument("--split-scheme", choices=["exgaze_with_val", "exgaze", "random"], default="exgaze_with_val")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--resize-policy", choices=["facet_square_direct", "letterbox_square", "sensor_full_square"], default="facet_square_direct")
    parser.add_argument("--input-width", type=int, default=256)
    parser.add_argument("--input-height", type=int, default=256)
    parser.add_argument("--event-policy", choices=["fixed_count", "time_bin"], default="fixed_count")
    parser.add_argument("--time-bin-us", type=int, default=5000)
    parser.add_argument("--event-count-target", type=int, default=5000)
    parser.add_argument("--accumulation", choices=["plain", "causal_linear"], default="causal_linear")
    parser.add_argument("--causal-weight-power", type=float, default=1.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_dataset(
        source_root=args.source_root,
        canonical_root=args.canonical_root,
        indexes_root=args.indexes_root,
        manifests_root=args.manifests_root,
        splits_root=args.splits_root,
        ingest_mode=args.ingest_mode,
        split_scheme=args.split_scheme,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        resize_policy=args.resize_policy,
        target_size_wh=(args.input_width, args.input_height),
        event_policy=args.event_policy,
        time_bin_us=args.time_bin_us,
        event_count_target=args.event_count_target,
        accumulation=args.accumulation,
        causal_weight_power=args.causal_weight_power,
    )


if __name__ == "__main__":
    print(run(build_argparser().parse_args()))
