#!/usr/bin/env python3
"""Re-split an existing DeanDataset_full_unet cache by subject.

The full U-Net expansion is expensive. This script reuses the existing cached
event segments and ellipse records, reconstructs session-contiguous sample
ranges from progress_state.json, and writes a new DavisEyeEllipseDataset root
with train/val/test splits.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from build_dean_dataset_from_ev_eye import (
    ELLIPSE_DTYPE,
    EVENT_DTYPE,
    create_memmap,
    merge_structured,
)


LEAK_FREE_SPLIT = {
    "train": set(range(1, 33)),
    "val": set(range(33, 37)),
    "test": set(range(37, 49)),
}

LITERAL_OVERLAP_SPLIT = {
    "train": set(range(1, 37)),
    "val": set(range(33, 37)),
    "test": set(range(37, 49)),
}


def parse_shape_dtype(info_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with info_path.open("r", encoding="utf-8") as f:
        shape_line = f.readline().strip()
        dtype_line = f.readline().strip()
    shape = tuple(int(x) for x in shape_line.split(": ")[1].strip("()").split(",") if x.strip())
    dtype = eval(dtype_line.split(": ")[1], {"np": np, "numpy": np})
    return shape, dtype


def batch_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


class CachedSplitReader:
    def __init__(self, split_root: Path):
        self.split_root = split_root
        self.data_root = split_root / "cached_data"
        self.ellipse_root = split_root / "cached_ellipse"
        self.ellipses = self._load_ellipses()
        self.event_index_paths = sorted(self.data_root.glob("events_indices_*.npy"), key=batch_id)
        self.event_counts = []
        self.event_indices = []
        for path in self.event_index_paths:
            indices = np.load(path)
            self.event_indices.append(indices)
            self.event_counts.append(indices.shape[0])
        self.cumulative = np.cumsum([0] + self.event_counts)
        self._event_memmaps: dict[int, np.memmap] = {}

    def _load_ellipses(self) -> np.memmap:
        data_path = self.ellipse_root / "ellipses_batch_0.memmap"
        info_path = self.ellipse_root / "ellipses_batch_info_0.txt"
        shape, dtype = parse_shape_dtype(info_path)
        return np.memmap(data_path, dtype=dtype, mode="r", shape=shape)

    def __len__(self) -> int:
        return int(len(self.ellipses))

    def _load_event_batch(self, batch: int) -> np.memmap:
        if batch in self._event_memmaps:
            return self._event_memmaps[batch]
        data_path = self.data_root / f"events_batch_{batch}.memmap"
        info_path = self.data_root / f"events_batch_info_{batch}.txt"
        shape, dtype = parse_shape_dtype(info_path)
        mmap = np.memmap(data_path, dtype=dtype, mode="r", shape=shape)
        self._event_memmaps[batch] = mmap
        return mmap

    def event_segment(self, index: int) -> np.ndarray:
        batch_pos = int(np.searchsorted(self.cumulative, index, side="right") - 1)
        local = index - int(self.cumulative[batch_pos])
        source_batch = batch_id(self.event_index_paths[batch_pos])
        start, end = self.event_indices[batch_pos][local]
        events = self._load_event_batch(source_batch)
        return np.asarray(events[start:end], dtype=EVENT_DTYPE)

    def ellipse(self, index: int) -> np.ndarray:
        return np.asarray(self.ellipses[index], dtype=ELLIPSE_DTYPE)


class SplitWriter:
    def __init__(self, output_root: Path, split: str, batch_size: int):
        self.output_root = output_root
        self.split = split
        self.batch_size = batch_size
        self.data_root = output_root / split / "cached_data"
        self.ellipse_root = output_root / split / "cached_ellipse"
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.ellipse_root.mkdir(parents=True, exist_ok=True)
        self.event_batch: list[np.ndarray] = []
        self.ellipses: list[np.ndarray] = []
        self.batch_id = 0

    @property
    def count(self) -> int:
        return len(self.ellipses)

    def add(self, events: np.ndarray, ellipse: np.ndarray) -> None:
        self.event_batch.append(events)
        self.ellipses.append(ellipse)
        if len(self.event_batch) >= self.batch_size:
            self.flush_events()

    def flush_events(self) -> None:
        if not self.event_batch:
            return
        events_merged, event_indices = merge_structured(self.event_batch)
        create_memmap(
            events_merged,
            self.data_root / f"events_batch_{self.batch_id}.memmap",
            self.data_root / f"events_batch_info_{self.batch_id}.txt",
        )
        np.save(self.data_root / f"events_indices_{self.batch_id}.npy", event_indices)
        self.event_batch = []
        self.batch_id += 1

    def close(self) -> None:
        self.flush_events()
        ellipses = np.zeros(len(self.ellipses), dtype=ELLIPSE_DTYPE)
        if self.ellipses:
            ellipses[:] = self.ellipses
        ellipse_indices = np.array([[i, i + 1] for i in range(len(ellipses))], dtype=np.int32)
        create_memmap(
            ellipses,
            self.ellipse_root / "ellipses_batch_0.memmap",
            self.ellipse_root / "ellipses_batch_info_0.txt",
        )
        np.save(self.ellipse_root / "ellipses_indices_0.npy", ellipse_indices)
        np.save(self.ellipse_root / "ellipse_records.npy", ellipses)


@dataclass
class SessionRange:
    old_split: str
    old_start: int
    old_end: int
    target_split: str
    user: int
    session: str
    valid: int
    summary: dict


def user_from_session(session_path: str) -> int:
    match = re.search(r"/user(\d+)/", session_path)
    if not match:
        raise ValueError(f"Cannot parse user from session path: {session_path}")
    return int(match.group(1))


def target_split_for_user(user: int, policy: dict[str, set[int]]) -> str:
    matches = [split for split, users in policy.items() if user in users]
    if not matches:
        raise ValueError(f"No split for user{user}")
    if len(matches) > 1:
        raise ValueError(f"Ambiguous overlapping split for user{user}: {matches}")
    return matches[0]


def build_session_ranges(input_root: Path, policy: dict[str, set[int]]) -> list[SessionRange]:
    progress_path = input_root / "progress_state.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    old_offsets = {"train": 0, "val": 0}
    ranges = []
    for summary in progress["session_summaries"]:
        old_split = summary["split"]
        valid = int(summary["valid"])
        user = user_from_session(summary["session"])
        target = target_split_for_user(user, policy)
        start = old_offsets[old_split]
        end = start + valid
        old_offsets[old_split] = end
        ranges.append(
            SessionRange(
                old_split=old_split,
                old_start=start,
                old_end=end,
                target_split=target,
                user=user,
                session=summary["session"],
                valid=valid,
                summary=summary,
            )
        )
    return ranges


def validate_readers(input_root: Path, ranges: list[SessionRange]) -> None:
    expected = {"train": 0, "val": 0}
    for item in ranges:
        expected[item.old_split] += item.valid
    for split, count in expected.items():
        reader = CachedSplitReader(input_root / split)
        if len(reader) != count:
            raise RuntimeError(f"{split} reader length {len(reader)} != progress count {count}")


def write_manifest(
    input_root: Path,
    output_root: Path,
    policy_name: str,
    ranges: list[SessionRange],
    writers: dict[str, SplitWriter],
) -> None:
    counts = {split: writer.count for split, writer in writers.items()}
    subject_map = {}
    for split in writers:
        users = sorted({item.user for item in ranges if item.target_split == split})
        subject_map[split] = users

    manifest = {
        "source": "Resplit from DeanDataset_full_unet cached U-Net pseudo-label/event data",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "split_policy": policy_name,
        "split_subjects": subject_map,
        "split_counts": counts,
        "num_sessions": len(ranges),
        "note": "Leak-free policy uses train 1-32, val 33-36, test 37-48.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    session_summaries = []
    for item in ranges:
        record = dict(item.summary)
        record["source_split"] = item.old_split
        record["split"] = item.target_split
        record["user"] = item.user
        session_summaries.append(record)
    progress = {
        "source_progress": str(input_root / "progress_state.json"),
        "split_policy": policy_name,
        "session_summaries": session_summaries,
        "writer_counts": counts,
    }
    (output_root / "progress_state.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent"),
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument(
        "--policy",
        choices=["leak_free_32_36_48", "literal_overlap_36_36_48"],
        default="leak_free_32_36_48",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policy = LEAK_FREE_SPLIT if args.policy == "leak_free_32_36_48" else LITERAL_OVERLAP_SPLIT
    if args.policy == "literal_overlap_36_36_48":
        raise SystemExit(
            "literal_overlap_36_36_48 intentionally duplicates subjects 33-36 and is not "
            "implemented as a default cache writer. Use leak_free_32_36_48 for the planned experiment."
        )

    ranges = build_session_ranges(args.input_root, policy)
    validate_readers(args.input_root, ranges)

    counts = {}
    subjects = {}
    for split in ("train", "val", "test"):
        selected = [item for item in ranges if item.target_split == split]
        counts[split] = sum(item.valid for item in selected)
        subjects[split] = sorted({item.user for item in selected})
    print("Split subjects:", subjects)
    print("Split sample counts:", counts)
    if args.dry_run:
        return

    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{args.output_root} is not empty. Re-run with --overwrite.")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    readers = {
        "train": CachedSplitReader(args.input_root / "train"),
        "val": CachedSplitReader(args.input_root / "val"),
    }
    writers = {split: SplitWriter(args.output_root, split, args.batch_size) for split in ("train", "val", "test")}

    for item in tqdm(ranges, desc="Re-splitting sessions"):
        reader = readers[item.old_split]
        writer = writers[item.target_split]
        for old_idx in range(item.old_start, item.old_end):
            writer.add(reader.event_segment(old_idx), reader.ellipse(old_idx))

    for writer in writers.values():
        writer.close()
    write_manifest(args.input_root, args.output_root, args.policy, ranges, writers)
    print("Wrote", args.output_root)


if __name__ == "__main__":
    main()
