import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from dataset.DavisEyeEllipse.DavisEyeEllipseFrameDataset import (
    natural_key,
    parse_frame_timestamp,
)
from utils.cache.MemmapCacheStructedEvents import load_memmap


def remap_session_path(session_path: Path, raw_root: Path) -> Path:
    parts = session_path.parts
    if "Data_davis" in parts:
        rel = Path(*parts[parts.index("Data_davis") + 1 :])
        session_path = raw_root / rel
    frames_dir = session_path / "frames"
    if not frames_dir.exists():
        raise FileNotFoundError(f"missing frames directory: {frames_dir}")
    return session_path


def load_ellipses(root_path: Path, split: str):
    ellipse_root = root_path / split / "cached_ellipse"
    return load_memmap(
        ellipse_root / "ellipses_batch_0.memmap",
        ellipse_root / "ellipses_batch_info_0.txt",
    )


def load_session_records(root_path: Path, raw_root: Path, split: str):
    with (root_path / "progress_state.json").open("r", encoding="utf-8") as fp:
        progress = json.load(fp)
    records = []
    for summary in progress.get("session_summaries", []):
        if summary.get("split") != split:
            continue
        count = int(summary.get("valid", 0))
        if count <= 0:
            continue
        records.append(
            {
                "session_path": remap_session_path(Path(summary["session"]), raw_root),
                "count": count,
            }
        )
    return records


def write_info(path: Path, shape: tuple[int, ...]):
    with path.open("w", encoding="utf-8") as fp:
        fp.write(f"Data shape: {shape}\n")
        fp.write("Data dtype: np.dtype('uint8')\n")


def frame_index(session_path: Path):
    frames = sorted((session_path / "frames").glob("*.png"), key=natural_key)
    return {parse_frame_timestamp(frame_path): frame_path for frame_path in frames}


def read_resize_frame(frame_path: Path, resolution: tuple[int, int]):
    height, width = resolution
    frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        raise FileNotFoundError(f"failed to read frame {frame_path}")
    if frame.shape != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    return frame


def build_split(
    root_path: Path,
    raw_root: Path,
    split: str,
    resolution: tuple[int, int],
    force: bool,
    decode_workers: int,
):
    height, width = resolution
    out_root = root_path / split / "cached_frame"
    out_root.mkdir(parents=True, exist_ok=True)
    frames_path = out_root / "frames_batch_0.memmap"
    info_path = out_root / "frames_batch_info_0.txt"
    meta_path = out_root / "metadata.json"

    ellipses = load_ellipses(root_path, split)
    expected_shape = (len(ellipses), height, width)
    if frames_path.exists() and info_path.exists() and not force:
        existing = load_memmap(frames_path, info_path)
        if existing.shape == expected_shape and existing.dtype == np.dtype("uint8"):
            print(f"{split}: cached_frame already exists with shape={existing.shape}")
            return
        raise ValueError(
            f"{split}: existing cached_frame shape/dtype mismatch: "
            f"shape={existing.shape}, dtype={existing.dtype}, expected={expected_shape}"
        )

    records = load_session_records(root_path, raw_root, split)
    total_records = sum(record["count"] for record in records)
    if total_records != len(ellipses):
        raise ValueError(
            f"{split}: session count mismatch: sessions={total_records}, ellipses={len(ellipses)}"
        )

    mmap = np.memmap(frames_path, dtype=np.uint8, mode="w+", shape=expected_shape)
    out_index = 0
    missing = 0
    for record in tqdm(records, desc=f"Building {split} cached_frame"):
        index = frame_index(record["session_path"])
        frame_paths = []
        for _ in range(record["count"]):
            timestamp = int(ellipses[out_index]["t"])
            frame_path = index.get(timestamp)
            if frame_path is None:
                missing += 1
                raise FileNotFoundError(
                    f"{split}: no frame timestamp={timestamp} in {record['session_path']}"
                )
            frame_paths.append(frame_path)
            out_index += 1
        start = out_index - len(frame_paths)
        if decode_workers > 1:
            with ThreadPoolExecutor(max_workers=decode_workers) as executor:
                frames = list(
                    executor.map(
                        lambda path: read_resize_frame(path, resolution),
                        frame_paths,
                    )
                )
            mmap[start:out_index] = np.stack(frames, axis=0)
        else:
            for offset, frame_path in enumerate(frame_paths):
                mmap[start + offset] = read_resize_frame(frame_path, resolution)
    mmap.flush()
    write_info(info_path, expected_shape)
    meta = {
        "format": "uint8 memmap",
        "shape": list(expected_shape),
        "resolution": [height, width],
        "split": split,
        "source_raw_root": str(raw_root),
        "root_path": str(root_path),
        "missing": missing,
    }
    with meta_path.open("w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)
        fp.write("\n")
    print(f"{split}: wrote {frames_path} shape={expected_shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Build split cached_frame memmaps for DeanDataset-style roots."
    )
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--decode-workers", type=int, default=8)
    args = parser.parse_args()

    for split in args.splits:
        build_split(
            root_path=args.root_path,
            raw_root=args.raw_root,
            split=split,
            resolution=(args.height, args.width),
            force=args.force,
            decode_workers=args.decode_workers,
        )


if __name__ == "__main__":
    main()
