from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEFAULT_SOURCE_ROOT = Path(
    "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/"
    "DeanDataset_full_unet_subject_independent"
)
DEFAULT_RAW_DAVIS_ROOT = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis")
DEFAULT_TARGET_BASE = Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data")
SENSOR_W = 346.0
SENSOR_H = 260.0
FRAME_SIZE = 128
EVENT_SIZE = 64


@dataclass(frozen=True)
class SessionRange:
    split: str
    start: int
    end: int
    session_path: Path
    subject: int
    eye: str
    session_name: str
    session_code: int


def parse_shape_dtype(info_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    lines = info_path.read_text().splitlines()
    shape = tuple(int(x) for x in lines[0].split(": ")[1].strip("()").split(",") if x.strip())
    dtype = eval(lines[1].split(": ")[1], {"np": np, "numpy": np})
    return shape, dtype


def load_ellipses(source_root: Path, split: str) -> np.memmap:
    root = source_root / split / "cached_ellipse"
    shape, dtype = parse_shape_dtype(root / "ellipses_batch_info_0.txt")
    return np.memmap(root / "ellipses_batch_0.memmap", dtype=dtype, mode="r", shape=shape)


def session_code(name: str) -> int:
    match = re.match(r"session_(\d)_(\d)_(\d)", name)
    if not match:
        raise ValueError(f"Cannot parse session name: {name}")
    return int("".join(match.groups()))


def parse_session_path(path: str) -> tuple[int, str, str, int]:
    subject = None
    eye = None
    session = None
    for part in Path(path).parts:
        if part.startswith("user") and part[4:].isdigit():
            subject = int(part[4:])
        elif part in {"left", "right"}:
            eye = part
        elif part.startswith("session_"):
            session = part
    if subject is None or eye is None or session is None:
        raise ValueError(f"Cannot parse session path: {path}")
    return subject, eye, session, session_code(session)


def build_ranges(source_root: Path) -> list[SessionRange]:
    state = json.loads((source_root / "progress_state.json").read_text())
    offsets = {"train": 0, "val": 0, "test": 0}
    ranges = []
    for summary in state["session_summaries"]:
        split = summary["split"]
        start = offsets[split]
        end = start + int(summary["valid"])
        offsets[split] = end
        subject, eye, session_name, code = parse_session_path(summary["session"])
        ranges.append(
            SessionRange(
                split=split,
                start=start,
                end=end,
                session_path=Path(summary["session"]),
                subject=subject,
                eye=eye,
                session_name=session_name,
                session_code=code,
            )
        )
    return ranges


def frame_index(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def frame_timestamp(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def frame_map(raw_davis_root: Path, session: SessionRange) -> dict[int, tuple[int, Path]]:
    frames_dir = raw_davis_root / f"user{session.subject}" / session.eye / session.session_name / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    return {frame_timestamp(path): (frame_index(path), path) for path in sorted(frames_dir.glob("*.png"))}


def scaled(value: float, src: float, dst: float) -> float:
    return value * dst / src


def row_for_sample(
    split: str,
    source_index: int,
    split_sample_idx: int,
    session: SessionRange,
    ellipse: np.void,
    frame_idx: int,
    frame_path: Path,
) -> dict:
    x = float(ellipse["x"])
    y = float(ellipse["y"])
    a = float(ellipse["a"])
    b = float(ellipse["b"])
    ang = float(ellipse["ang"])
    return {
        "sample_idx": split_sample_idx,
        "source_global_idx": source_index,
        "split": split,
        "subject": session.subject,
        "eye": session.eye,
        "session_name": session.session_name,
        "session_code": session.session_code,
        "frame_idx": frame_idx,
        "timestamp": int(ellipse["t"]),
        "source_frame": str(frame_path),
        "gt_x_orig": f"{x:.8f}",
        "gt_y_orig": f"{y:.8f}",
        "gt_a_orig": f"{a:.8f}",
        "gt_b_orig": f"{b:.8f}",
        "gt_ang": f"{ang:.8f}",
        "gt_x_128": f"{scaled(x, SENSOR_W, FRAME_SIZE):.8f}",
        "gt_y_128": f"{scaled(y, SENSOR_H, FRAME_SIZE):.8f}",
        "gt_a_128": f"{scaled(a, SENSOR_W, FRAME_SIZE):.8f}",
        "gt_b_128": f"{scaled(b, SENSOR_H, FRAME_SIZE):.8f}",
        "gt_x_64": f"{scaled(x, SENSOR_W, EVENT_SIZE):.8f}",
        "gt_y_64": f"{scaled(y, SENSOR_H, EVENT_SIZE):.8f}",
        "gt_a_64": f"{scaled(a, SENSOR_W, EVENT_SIZE):.8f}",
        "gt_b_64": f"{scaled(b, SENSOR_H, EVENT_SIZE):.8f}",
    }


def write_csv(path: Path, rows_iter, fieldnames: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_iter:
            writer.writerow(row)
            count += 1
    return count


def iter_manifest_rows(
    source_root: Path,
    raw_davis_root: Path,
    split: str,
    ranges: list[SessionRange],
    max_samples: int | None,
):
    ellipses = load_ellipses(source_root, split)
    emitted = 0
    for session in tqdm(ranges, desc=f"manifest {split}"):
        fmap = frame_map(raw_davis_root, session)
        for source_index in range(session.start, session.end):
            if max_samples is not None and emitted >= max_samples:
                return
            ellipse = ellipses[source_index]
            timestamp = int(ellipse["t"])
            if timestamp not in fmap:
                raise RuntimeError(f"No frame timestamp {timestamp} for {session.session_path}")
            frame_idx, frame_path = fmap[timestamp]
            yield row_for_sample(split, source_index, emitted, session, ellipse, frame_idx, frame_path)
            emitted += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-davis-root", type=Path, default=DEFAULT_RAW_DAVIS_ROOT)
    parser.add_argument("--target-base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument("--tag", default="train_ready")
    parser.add_argument("--max-samples-per-split", type=int, default=0, help="0 means all samples")
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_samples = args.max_samples_per_split if args.max_samples_per_split > 0 else None
    ev_root = args.target_base / "EV_Eye_Hybrid_frame128_event64_subject_independent" / args.tag
    ex_root = args.target_base / "EX_Gaze_Hybrid_frame128_event64_subject_independent" / args.tag
    ranges_by_split = defaultdict(list)
    for item in build_ranges(args.source_root):
        ranges_by_split[item.split].append(item)

    fieldnames = [
        "sample_idx",
        "source_global_idx",
        "split",
        "subject",
        "eye",
        "session_name",
        "session_code",
        "frame_idx",
        "timestamp",
        "source_frame",
        "gt_x_orig",
        "gt_y_orig",
        "gt_a_orig",
        "gt_b_orig",
        "gt_ang",
        "gt_x_128",
        "gt_y_128",
        "gt_a_128",
        "gt_b_128",
        "gt_x_64",
        "gt_y_64",
        "gt_a_64",
        "gt_b_64",
    ]
    summary = {
        "source_root": str(args.source_root),
        "raw_davis_root": str(args.raw_davis_root),
        "ev_eye_root": str(ev_root),
        "ex_gaze_root": str(ex_root),
        "tag": args.tag,
        "max_samples_per_split": max_samples,
        "splits": args.splits,
        "split_counts": {},
        "storage_mode": "compact_manifest_live_loader",
        "contracts": {
            "frame": "read source_frame live and resize to 1 x 128 x 128",
            "event": "read source cached_data live and bin to 2 x 64 x 64",
            "mask": "rasterize scaled ellipse to 1 x 128 x 128",
            "exgaze_patches": "derive 8 x 2 x 16 x 16 patches live from event volume",
        },
    }
    for split in args.splits:
        rows = iter_manifest_rows(args.source_root, args.raw_davis_root, split, ranges_by_split[split], max_samples)
        ev_count = write_csv(ev_root / split / "labels.csv", rows, fieldnames)
        # Re-read from EV-Eye CSV rather than generating twice to keep both model roots byte-compatible.
        with (ev_root / split / "labels.csv").open() as src:
            ex_root.joinpath(split).mkdir(parents=True, exist_ok=True)
            ex_root.joinpath(split, "labels.csv").write_text(src.read())
        summary["split_counts"][split] = ev_count

    ev_root.mkdir(parents=True, exist_ok=True)
    ex_root.mkdir(parents=True, exist_ok=True)
    (ev_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ex_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
