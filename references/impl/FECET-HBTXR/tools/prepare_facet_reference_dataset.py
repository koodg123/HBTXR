from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT

from fecet_hbtxr.io import ensure_dir, read_jsonl, resolve_stored_path, write_json


RAW_EVENT_DTYPE = np.dtype([("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int64)])
RAW_ELLIPSE_DTYPE = np.dtype([("t", np.int64), ("x", np.float64), ("y", np.float64), ("a", np.float64), ("b", np.float64), ("ang", np.float64)])


def _natural_key(path: Path) -> list[Any]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _event_selection(
    timestamps_us: np.ndarray,
    *,
    end_timestamp_us: int,
    policy: str,
    time_bin_us: int,
    event_count_target: int,
    start_timestamp_us: int | None,
) -> slice:
    end_idx = int(np.searchsorted(timestamps_us, end_timestamp_us, side="right"))
    if policy == "fixed_count":
        start_idx = max(0, end_idx - int(event_count_target))
        return slice(start_idx, end_idx)
    effective_start = int(end_timestamp_us) - int(time_bin_us) if start_timestamp_us is None else int(start_timestamp_us)
    start_idx = int(np.searchsorted(timestamps_us, effective_start, side="left"))
    return slice(start_idx, end_idx)


def _load_annotation_store(canonical_root: Path, rows: list[dict[str, Any]]) -> dict[Path, dict[str, dict[str, Any]]]:
    stores: dict[Path, dict[str, dict[str, Any]]] = {}
    for row in rows:
        ref = row["annotation_ref"]
        store_path = resolve_stored_path(canonical_root, ref["annotation_store_path"])
        if store_path not in stores:
            store_rows = read_jsonl(store_path)
            stores[store_path] = {str(item["ann_id"]): item for item in store_rows}
    return stores


def _load_events_npz(cache: dict[Path, dict[str, np.ndarray]], canonical_root: Path, events_npz: str | Path) -> dict[str, np.ndarray]:
    events_path = resolve_stored_path(canonical_root, events_npz)
    loaded = cache.get(events_path)
    if loaded is None:
        data = np.load(events_path)
        loaded = {key: np.asarray(data[key]) for key in data.files}
        cache[events_path] = loaded
    return loaded


def _write_structured_events(segment: dict[str, np.ndarray], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for t, x, y, p in zip(segment["t"], segment["x"], segment["y"], segment["p"]):
            handle.write(f"{int(t)} {int(x)} {int(y)} {int(p)}\n")


def _write_structured_center(annotation: dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    ellipse = annotation.get("ellipse_sensor_xywht") or annotation.get("pupil_ellipse_xywht_sensor")
    close = 1 if bool(annotation.get("closed_eye_flag", False)) else 0
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{int(annotation['timestamp_us'])},{float(ellipse[0]):.2f},{float(ellipse[1]):.2f},{close}\n")


def _write_structured_ellipse(annotation: dict[str, Any], output_path: Path) -> None:
    ensure_dir(output_path.parent)
    ellipse = annotation.get("ellipse_sensor_xywht") or annotation.get("pupil_ellipse_xywht_sensor")
    theta_deg = float(np.rad2deg(float(ellipse[4])))
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"{int(annotation['timestamp_us'])} {float(ellipse[0]):.2f} {float(ellipse[1]):.2f} "
            f"{float(ellipse[2]):.2f} {float(ellipse[3]):.2f} {theta_deg:.2f}\n"
        )


def _load_events_txt(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.int64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    out = np.zeros((raw.shape[0],), dtype=RAW_EVENT_DTYPE)
    out["t"] = raw[:, 0]
    out["x"] = raw[:, 1]
    out["y"] = raw[:, 2]
    out["p"] = raw[:, 3]
    return out


def _load_ellipse_txt(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    out = np.zeros((raw.shape[0],), dtype=RAW_ELLIPSE_DTYPE)
    out["t"] = raw[:, 0].astype(np.int64)
    out["x"] = raw[:, 1]
    out["y"] = raw[:, 2]
    out["a"] = raw[:, 3]
    out["b"] = raw[:, 4]
    out["ang"] = raw[:, 5]
    return out


def _merge_structured_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    total_length = sum(int(array.shape[0]) for array in arrays)
    merged = np.zeros((total_length,), dtype=arrays[0].dtype)
    cursor = 0
    for array in arrays:
        end = cursor + int(array.shape[0])
        merged[cursor:end] = array
        cursor = end
    return merged


def _indices_for_arrays(arrays: list[np.ndarray]) -> np.ndarray:
    indices = np.zeros((len(arrays), 2), dtype=np.int32)
    cursor = 0
    for idx, array in enumerate(arrays):
        end = cursor + int(array.shape[0])
        indices[idx] = [cursor, end]
        cursor = end
    return indices


def _create_memmap(data: np.ndarray, data_file: Path, info_file: Path) -> None:
    ensure_dir(data_file.parent)
    mmap = np.memmap(data_file, dtype=data.dtype, mode="w+", shape=data.shape)
    mmap[:] = data
    mmap.flush()
    info_file.write_text(f"Data shape: {data.shape}\nData dtype: {data.dtype}\n", encoding="utf-8")


def _cache_split(split_root: Path, *, file_batch_size: int, overwrite: bool) -> dict[str, Any]:
    data_paths = sorted((split_root / "data").glob("*.txt"), key=_natural_key)
    ellipse_paths = sorted((split_root / "ellipse").glob("*.txt"), key=_natural_key)
    if len(data_paths) != len(ellipse_paths):
        raise ValueError(f"FACET reference split mismatch: {split_root} has {len(data_paths)} data files and {len(ellipse_paths)} ellipse files")
    if not data_paths:
        return {"sample_files": 0, "event_batches": 0, "ellipse_batches": 0}

    if [path.stem for path in data_paths] != [path.stem for path in ellipse_paths]:
        raise ValueError(f"FACET reference split file ordering mismatch in {split_root}")

    cached_data_root = ensure_dir(split_root / "cached_data")
    cached_ellipse_root = ensure_dir(split_root / "cached_ellipse")
    if overwrite:
        for directory in (cached_data_root, cached_ellipse_root):
            for path in directory.glob("*"):
                if path.is_file():
                    path.unlink()

    batch_index = 0
    for start in range(0, len(data_paths), int(file_batch_size)):
        batch_data_paths = data_paths[start:start + int(file_batch_size)]
        batch_ellipse_paths = ellipse_paths[start:start + int(file_batch_size)]
        events_arrays = [_load_events_txt(path) for path in batch_data_paths]
        ellipse_arrays = [_load_ellipse_txt(path) for path in batch_ellipse_paths]
        merged_events = _merge_structured_arrays(events_arrays)
        merged_ellipses = _merge_structured_arrays(ellipse_arrays)
        event_indices = _indices_for_arrays(events_arrays)
        ellipse_indices = _indices_for_arrays(ellipse_arrays)

        _create_memmap(merged_events, cached_data_root / f"events_batch_{batch_index}.memmap", cached_data_root / f"events_batch_info_{batch_index}.txt")
        np.save(cached_data_root / f"events_indices_{batch_index}.npy", event_indices)
        _create_memmap(merged_ellipses, cached_ellipse_root / f"ellipses_batch_{batch_index}.memmap", cached_ellipse_root / f"ellipses_batch_info_{batch_index}.txt")
        np.save(cached_ellipse_root / f"ellipses_indices_{batch_index}.npy", ellipse_indices)
        batch_index += 1

    return {"sample_files": len(data_paths), "event_batches": batch_index, "ellipse_batches": batch_index}


def prepare_facet_reference_dataset(
    *,
    canonical_root: str | Path,
    manifests_root: str | Path,
    output_root: str | Path,
    splits: list[str],
    file_batch_size: int,
    overwrite: bool,
) -> dict[str, Any]:
    canonical_root = Path(canonical_root).resolve()
    manifests_root = Path(manifests_root).resolve()
    output_root = Path(output_root).resolve()
    ensure_dir(output_root)

    event_cache: dict[Path, dict[str, np.ndarray]] = {}
    split_summary: dict[str, Any] = {}

    for split in splits:
        manifest_path = manifests_root / f"{split}_manifest.jsonl"
        if not manifest_path.exists():
            split_summary[split] = {"status": "missing_manifest"}
            continue
        rows = read_jsonl(manifest_path)
        if not rows:
            split_summary[split] = {"status": "empty_manifest"}
            continue
        annotation_stores = _load_annotation_store(canonical_root, rows)
        split_root = Path(ensure_dir(output_root / split))
        if overwrite:
            for relative in ("data", "ellipse", "label"):
                directory = split_root / relative
                if directory.exists():
                    for path in directory.glob("*.txt"):
                        path.unlink()
        ensure_dir(split_root / "data")
        ensure_dir(split_root / "ellipse")
        ensure_dir(split_root / "label")

        sample_count = 0
        for row in rows:
            ref = row["annotation_ref"]
            store_path = resolve_stored_path(canonical_root, ref["annotation_store_path"])
            annotation = annotation_stores[store_path][str(ref["ann_id"])]
            events = _load_events_npz(event_cache, canonical_root, row["events_npz"])
            event_window = row.get("event_window") or {}
            end_timestamp_us = int(event_window.get("end_timestamp_us", annotation.get("timestamp_us", 0)))
            selection = _event_selection(
                np.asarray(events["t"], dtype=np.int64),
                end_timestamp_us=end_timestamp_us,
                policy=str(event_window.get("policy", "fixed_count")),
                time_bin_us=int(event_window.get("time_bin_us", 5000)),
                event_count_target=int(event_window.get("event_count_target", 5000)),
                start_timestamp_us=None if event_window.get("start_timestamp_us") is None else int(event_window["start_timestamp_us"]),
            )
            segment = {
                "t": np.asarray(events["t"][selection], dtype=np.int64),
                "x": np.asarray(events["x"][selection], dtype=np.int64),
                "y": np.asarray(events["y"][selection], dtype=np.int64),
                "p": np.asarray(events["p"][selection], dtype=np.int64),
            }
            sample_id = str(row["sample_id"])
            _write_structured_events(segment, split_root / "data" / f"{sample_id}.txt")
            _write_structured_ellipse(annotation, split_root / "ellipse" / f"{sample_id}.txt")
            _write_structured_center(annotation, split_root / "label" / f"{sample_id}.txt")
            sample_count += 1

        cache_summary = _cache_split(split_root, file_batch_size=file_batch_size, overwrite=overwrite)
        split_summary[split] = {"status": "prepared", "sample_count": sample_count, **cache_summary}

    summary = {
        "canonical_root": str(canonical_root),
        "manifests_root": str(manifests_root),
        "output_root": str(output_root),
        "splits": split_summary,
        "file_batch_size": int(file_batch_size),
    }
    write_json(summary, output_root / "facet_reference_summary.json")
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare FACET reference-compatible sample files and caches from FECET-HBTXR manifests")
    parser.add_argument("--canonical-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "canonical"))
    parser.add_argument("--manifests-root", type=str, default=str(PROJECT_ROOT / "data" / "_internal" / "manifests"))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "data" / "facet_reference"))
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], default=[])
    parser.add_argument("--file-batch-size", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    return prepare_facet_reference_dataset(
        canonical_root=args.canonical_root,
        manifests_root=args.manifests_root,
        output_root=args.output_root,
        splits=args.split or ["train", "val", "test"],
        file_batch_size=args.file_batch_size,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    print(json.dumps(run(build_argparser().parse_args()), indent=2, ensure_ascii=False))
