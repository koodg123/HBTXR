from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
from EvEye.dataset.DavisEyeEllipse.utils import cal_ellipse_area, convert_to_ellipse
from EvEye.utils.cache.MemmapCacheStructedEvents import load_ellipse, load_event_segment


SPLIT_TO_SUBDIR = {"train": "train", "val": "train", "test": "test"}


def _session_id(session_path: str, fallback: int) -> str:
    match = re.search(r"user(\d+).*/session_([0-9_]+)$", session_path)
    if not match:
        return f"record_{fallback:05d}"
    user, session = match.groups()
    return f"user{int(user):02d}_{session}"


def _session_ranges(root_path: Path) -> dict[str, list[tuple[str, int, int]]]:
    with (root_path / "progress_state.json").open() as f:
        state = json.load(f)

    ranges: dict[str, list[tuple[str, int, int]]] = {"train": [], "val": [], "test": []}
    offsets = {split: 0 for split in ranges}
    for idx, session in enumerate(state["session_summaries"]):
        split = session["split"]
        valid = int(session["valid"])
        start = offsets[split]
        end = start + valid
        offsets[split] = end
        ranges[split].append((_session_id(session["session"], idx), start, end))
    return ranges


def _make_dataset(root_path: Path, split: str) -> DavisEyeEllipseDataset:
    return DavisEyeEllipseDataset(
        root_path=root_path,
        split=split,
        accumulate_mode="fixed_count",
        sensor_size=(346, 260, 2),
        events_interpolation="causal_linear_ori",
        pupil_area=200,
        num_classes=1,
        default_resolution=[64, 64],
    )


def _center64(sample: dict) -> np.ndarray:
    center = np.asarray(sample["center"], dtype=np.float32) * 4.0
    if int(sample["close"]) != 0 or not np.isfinite(center).all():
        return np.zeros(2, dtype=np.float32)
    return np.clip(center, 0.0, 63.0).astype(np.float32)


def _raw_center64(index: int, ellipse_path: Path, pupil_area: float = 200.0) -> tuple[np.ndarray, int]:
    ellipse = convert_to_ellipse(load_ellipse(index, ellipse_path))
    center, axes, _ = ellipse
    valid = center != (0, 0) and cal_ellipse_area(axes[0], axes[1]) > pupil_area
    if not valid:
        return np.zeros(2, dtype=np.float32), 1
    x = float(center[0]) * 64.0 / 346.0
    y = float(center[1]) * 64.0 / 260.0
    return np.asarray([np.clip(x, 0.0, 63.0), np.clip(y, 0.0, 63.0)], dtype=np.float32), 0


def _iter_session_samples(dataset: DavisEyeEllipseDataset, start: int, end: int):
    for index in range(start, end):
        sample = dataset[index]
        frame = np.asarray(sample["input"], dtype=np.float16)
        label = _center64(sample) / 64.0
        yield frame, label.astype(np.float32)


def export_tdtracker_h5(
    root_path: Path,
    output_dir: Path,
    sequence_length: int,
    stride: int,
    max_sequences_per_split: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranges = _session_ranges(root_path)

    for split in ("train", "val", "test"):
        dataset = _make_dataset(root_path, split)
        seq_frames = []
        seq_labels = []
        for _, start, end in tqdm(ranges[split], desc=f"TDTracker {split} sessions"):
            frames = []
            labels = []
            for frame, label in _iter_session_samples(dataset, start, end):
                frames.append(frame)
                labels.append(label)
            if len(frames) < sequence_length:
                continue
            frames_arr = np.stack(frames)
            labels_arr = np.stack(labels)
            for seq_start in range(0, len(frames_arr) - sequence_length + 1, stride):
                seq_end = seq_start + sequence_length
                seq_frames.append(frames_arr[seq_start:seq_end])
                seq_labels.append(labels_arr[seq_start:seq_end])
                if max_sequences_per_split and len(seq_frames) >= max_sequences_per_split:
                    break
            if max_sequences_per_split and len(seq_frames) >= max_sequences_per_split:
                break

        out_path = output_dir / f"{split}_hbtxr_img64_seq{sequence_length}.h5"
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("frames", data=np.stack(seq_frames), compression="gzip")
            h5.create_dataset("label", data=np.stack(seq_labels), compression="gzip")


def _resize_events_to_64(events: np.ndarray, frame_index: int, frame_dt_us: int) -> np.ndarray:
    resized = np.zeros(events.shape, dtype=[("t", "<i8"), ("x", "<i8"), ("y", "<i8"), ("p", "<i8")])
    if len(events) == 0:
        return resized
    t = events["t"].astype(np.float64)
    span = max(float(t[-1] - t[0]), 1.0)
    resized["t"] = (frame_index * frame_dt_us + ((t - t[0]) / span) * (frame_dt_us - 1)).astype(np.int64)
    resized["x"] = np.clip((events["x"].astype(np.float32) * 64.0 / 346.0).astype(np.int64), 0, 63)
    resized["y"] = np.clip((events["y"].astype(np.float32) * 64.0 / 260.0).astype(np.int64), 0, 63)
    resized["p"] = (events["p"].astype(np.int64) > 0).astype(np.int64)
    return resized


def export_threeet_tree(
    root_path: Path,
    output_dir: Path,
    frame_dt_us: int,
    max_frames_per_split: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(exist_ok=True)
    ranges = _session_ranges(root_path)

    list_files = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        dataset = _make_dataset(root_path, split)
        data_path = dataset.data_path
        ellipse_path = dataset.ellipse_path
        emitted = 0
        for session_id, start, end in tqdm(ranges[split], desc=f"3ET {split} sessions"):
            if max_frames_per_split and emitted >= max_frames_per_split:
                break
            out_subdir = output_dir / SPLIT_TO_SUBDIR[split] / session_id
            out_subdir.mkdir(parents=True, exist_ok=True)
            events_all = []
            labels = []
            local_count = 0
            for local_idx, global_idx in enumerate(range(start, end)):
                if max_frames_per_split and emitted >= max_frames_per_split:
                    break
                center, close = _raw_center64(global_idx, ellipse_path)
                labels.append(tuple(center.tolist() + [float(close)]))
                event_segment = load_event_segment(global_idx, data_path)
                events_all.append(_resize_events_to_64(event_segment, local_idx, frame_dt_us))
                emitted += 1
                local_count += 1
            if local_count == 0:
                continue
            with h5py.File(out_subdir / f"{session_id}.h5", "w") as h5:
                h5.create_dataset("events", data=np.concatenate(events_all), compression="gzip")
            label_name = "label_zeros.txt" if split == "test" else "label.txt"
            with (out_subdir / label_name).open("w") as f:
                for label in labels:
                    f.write(f"({label[0]:.6f}, {label[1]:.6f}, {label[2]:.1f})\n")
            if split == "test":
                with (out_subdir / "label.txt").open("w") as f:
                    for label in labels:
                        f.write(f"({label[0]:.6f}, {label[1]:.6f}, {label[2]:.1f})\n")
            list_files[split].append(session_id)

    for split, records in list_files.items():
        with (dataset_dir / f"{split}_files.txt").open("w") as f:
            f.write("\n".join(records))
            if records:
                f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path",
        default="/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--format", choices=["tdtracker-h5", "threeet-tree"], required=True)
    parser.add_argument("--sequence-length", type=int, default=100)
    parser.add_argument("--stride", type=int, default=100)
    parser.add_argument("--frame-dt-us", type=int, default=10000)
    parser.add_argument("--max-sequences-per-split", type=int)
    parser.add_argument("--max-frames-per-split", type=int)
    return parser.parse_args()


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    args = parse_args()
    root_path = Path(args.root_path)
    output_dir = Path(args.output_dir)
    if args.format == "tdtracker-h5":
        export_tdtracker_h5(
            root_path,
            output_dir,
            args.sequence_length,
            args.stride,
            args.max_sequences_per_split,
        )
    else:
        export_threeet_tree(
            root_path,
            output_dir,
            args.frame_dt_us,
            args.max_frames_per_split,
        )


if __name__ == "__main__":
    main()
