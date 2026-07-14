from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = Path(
    "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/"
    "DeanDataset_full_unet_subject_independent"
)
DEFAULT_RAW_DAVIS_ROOT = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis")
DEFAULT_TARGET_BASE = Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data")

EVENT_DTYPE = np.dtype([("t", "<i8"), ("x", "<i8"), ("y", "<i8"), ("p", "<i8")])
ELLIPSE_DTYPE = np.dtype(
    [("t", "<i8"), ("x", "<f8"), ("y", "<f8"), ("a", "<f8"), ("b", "<f8"), ("ang", "<f8")]
)
SENSOR_W = 346.0
SENSOR_H = 260.0
FRAME_SIZE = 128
EVENT_SIZE = 64
PATCH_SIZE = 16
PATCH_NUM = 8


def _parse_shape_dtype(info_path: Path) -> tuple[tuple[int, ...], np.dtype]:
    lines = info_path.read_text().splitlines()
    shape = tuple(int(x) for x in lines[0].split(": ")[1].strip("()").split(",") if x.strip())
    dtype = eval(lines[1].split(": ")[1], {"np": np, "numpy": np})
    return shape, dtype


def _batch_id(path: Path) -> int:
    return int(path.stem.split("_")[-1])


class CachedSplit:
    def __init__(self, split_root: Path):
        self.split_root = split_root
        self.data_root = split_root / "cached_data"
        self.ellipse_root = split_root / "cached_ellipse"
        ellipse_shape, ellipse_dtype = _parse_shape_dtype(self.ellipse_root / "ellipses_batch_info_0.txt")
        self.ellipses = np.memmap(
            self.ellipse_root / "ellipses_batch_0.memmap",
            dtype=ellipse_dtype,
            mode="r",
            shape=ellipse_shape,
        )
        self.index_paths = sorted(self.data_root.glob("events_indices_*.npy"), key=_batch_id)
        self.indices = [np.load(path, mmap_mode="r") for path in self.index_paths]
        self.counts = [int(idx.shape[0]) for idx in self.indices]
        self.cumulative = np.cumsum([0] + self.counts)
        self._event_maps: dict[int, np.memmap] = {}

    def event_segment(self, global_index: int) -> np.ndarray:
        batch_pos = int(np.searchsorted(self.cumulative, global_index, side="right") - 1)
        local_index = global_index - int(self.cumulative[batch_pos])
        batch = _batch_id(self.index_paths[batch_pos])
        start, end = self.indices[batch_pos][local_index]
        if batch not in self._event_maps:
            shape, dtype = _parse_shape_dtype(self.data_root / f"events_batch_info_{batch}.txt")
            self._event_maps[batch] = np.memmap(
                self.data_root / f"events_batch_{batch}.memmap",
                dtype=dtype,
                mode="r",
                shape=shape,
            )
        return np.asarray(self._event_maps[batch][start:end], dtype=EVENT_DTYPE)

    def ellipse(self, global_index: int) -> np.void:
        return self.ellipses[global_index]


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


def _session_code(name: str) -> int:
    match = re.match(r"session_(\d)_(\d)_(\d)", name)
    if not match:
        raise ValueError(f"Cannot parse session name: {name}")
    return int("".join(match.groups()))


def _parse_session_path(path: str) -> tuple[int, str, str, int]:
    p = Path(path)
    subject = None
    eye = None
    session = None
    for part in p.parts:
        if part.startswith("user") and part[4:].isdigit():
            subject = int(part[4:])
        elif part in {"left", "right"}:
            eye = part
        elif part.startswith("session_"):
            session = part
    if subject is None or eye is None or session is None:
        raise ValueError(f"Cannot parse session path: {path}")
    return subject, eye, session, _session_code(session)


def build_ranges(source_root: Path) -> list[SessionRange]:
    state = json.loads((source_root / "progress_state.json").read_text())
    offsets = {"train": 0, "val": 0, "test": 0}
    ranges = []
    for summary in state["session_summaries"]:
        split = summary["split"]
        valid = int(summary["valid"])
        start = offsets[split]
        end = start + valid
        offsets[split] = end
        subject, eye, session_name, session_code = _parse_session_path(summary["session"])
        ranges.append(
            SessionRange(
                split=split,
                start=start,
                end=end,
                session_path=Path(summary["session"]),
                subject=subject,
                eye=eye,
                session_name=session_name,
                session_code=session_code,
            )
        )
    return ranges


def frame_index(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def frame_timestamp(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def frame_map(raw_davis_root: Path, session: SessionRange) -> dict[int, Path]:
    frames_dir = (
        raw_davis_root
        / f"user{session.subject}"
        / session.eye
        / session.session_name
        / "frames"
    )
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    return {frame_timestamp(path): path for path in sorted(frames_dir.glob("*.png"))}


def ellipse_values(ellipse: np.void) -> dict[str, float | int]:
    return {
        "t": int(ellipse["t"]),
        "x": float(ellipse["x"]),
        "y": float(ellipse["y"]),
        "a": float(ellipse["a"]),
        "b": float(ellipse["b"]),
        "ang": float(ellipse["ang"]),
    }


def scale_ellipse(e: dict[str, float | int], width: int, height: int) -> list[float]:
    return [
        float(e["x"]) * width / SENSOR_W,
        float(e["y"]) * height / SENSOR_H,
        float(e["a"]) * width / SENSOR_W,
        float(e["b"]) * height / SENSOR_H,
        float(e["ang"]),
    ]


def make_mask128(ellipse128: list[float]) -> np.ndarray:
    mask = np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8)
    x, y, a, b, angle = ellipse128
    center = (int(round(x)), int(round(y)))
    axes = (max(1, int(round(a / 2.0))), max(1, int(round(b / 2.0))))
    cv2.ellipse(mask, center, axes, float(angle), 0, 360, 255, -1)
    return mask


def make_event64(events: np.ndarray) -> np.ndarray:
    volume = np.zeros((2, EVENT_SIZE, EVENT_SIZE), dtype=np.uint16)
    if len(events) == 0:
        return volume
    xs = np.clip((events["x"].astype(np.float32) * EVENT_SIZE / SENSOR_W).astype(np.int64), 0, EVENT_SIZE - 1)
    ys = np.clip((events["y"].astype(np.float32) * EVENT_SIZE / SENSOR_H).astype(np.int64), 0, EVENT_SIZE - 1)
    ps = (events["p"].astype(np.int64) > 0).astype(np.int64)
    np.add.at(volume, (ps, ys, xs), 1)
    return volume


def sample_regions_from_ellipse64(ellipse64: list[float]) -> list[list[int]]:
    cx, cy, a, b, angle = ellipse64
    rad = np.deg2rad(angle)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    regions = []
    for i in range(PATCH_NUM):
        theta = 2.0 * np.pi * i / PATCH_NUM
        px = cx + (a / 2.0) * np.cos(theta) * cos_a - (b / 2.0) * np.sin(theta) * sin_a
        py = cy + (a / 2.0) * np.cos(theta) * sin_a + (b / 2.0) * np.sin(theta) * cos_a
        x1 = int(np.ceil(px)) - PATCH_SIZE // 2
        y1 = int(np.ceil(py)) - PATCH_SIZE // 2
        x1 = int(np.clip(x1, 0, EVENT_SIZE - PATCH_SIZE))
        y1 = int(np.clip(y1, 0, EVENT_SIZE - PATCH_SIZE))
        regions.append([x1, y1, x1 + PATCH_SIZE, y1 + PATCH_SIZE])
    return regions


def event_patches(volume: np.ndarray, regions: list[list[int]]) -> np.ndarray:
    patches = []
    for x1, y1, x2, y2 in regions:
        patches.append(volume[:, y1:y2, x1:x2])
    return np.stack(patches).astype(np.uint16)


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def export_split(
    *,
    split: str,
    ranges: list[SessionRange],
    cache: CachedSplit,
    raw_davis_root: Path,
    ev_root: Path,
    ex_root: Path,
    max_samples: int | None,
) -> tuple[list[dict], list[dict]]:
    metadata_rows = []
    ex_data_list = []
    emitted = 0

    ev_frame_dir = ev_root / split / "frames_128"
    ev_mask_dir = ev_root / split / "masks_128"
    ev_event_dir = ev_root / split / "events_64"
    ex_frame_dir = ex_root / split / "frames_128"
    ex_event_dir = ex_root / split / "events_64"
    ex_patch_dir = ex_root / split / "event_patches_8x2x16x16"
    for path in (ev_frame_dir, ev_mask_dir, ev_event_dir, ex_frame_dir, ex_event_dir, ex_patch_dir):
        path.mkdir(parents=True, exist_ok=True)

    for session in tqdm(ranges, desc=f"export {split}"):
        fmap = frame_map(raw_davis_root, session)
        for global_idx in range(session.start, session.end):
            if max_samples is not None and emitted >= max_samples:
                return metadata_rows, ex_data_list
            ellipse = ellipse_values(cache.ellipse(global_idx))
            src_frame = fmap.get(int(ellipse["t"]))
            if src_frame is None:
                raise RuntimeError(
                    f"No frame with timestamp {ellipse['t']} for {session.session_path}"
                )
            frame = cv2.imread(str(src_frame), cv2.IMREAD_GRAYSCALE)
            if frame is None:
                raise RuntimeError(f"Cannot read frame: {src_frame}")
            frame128 = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
            ellipse128 = scale_ellipse(ellipse, FRAME_SIZE, FRAME_SIZE)
            ellipse64 = scale_ellipse(ellipse, EVENT_SIZE, EVENT_SIZE)
            mask128 = make_mask128(ellipse128)
            events = cache.event_segment(global_idx)
            event64 = make_event64(events)
            regions = sample_regions_from_ellipse64(ellipse64)
            patches = event_patches(event64, regions)

            sample_name = (
                f"{split}_{emitted:07d}_s{session.subject:02d}_{session.eye}_"
                f"{session.session_code}_{frame_index(src_frame):06d}"
            )
            frame_name = f"{sample_name}.png"
            mask_name = f"{sample_name}_mask.png"
            event_name = f"{sample_name}.npy"
            patch_name = f"{sample_name}.npy"

            ev_frame_path = ev_frame_dir / frame_name
            ev_mask_path = ev_mask_dir / mask_name
            ev_event_path = ev_event_dir / event_name
            ex_frame_path = ex_frame_dir / frame_name
            ex_event_path = ex_event_dir / event_name
            ex_patch_path = ex_patch_dir / patch_name

            cv2.imwrite(str(ev_frame_path), frame128)
            cv2.imwrite(str(ev_mask_path), mask128)
            np.save(ev_event_path, event64)
            cv2.imwrite(str(ex_frame_path), frame128)
            np.save(ex_event_path, event64)
            np.save(ex_patch_path, patches)

            row = {
                "sample_idx": emitted,
                "source_global_idx": global_idx,
                "split": split,
                "subject": session.subject,
                "eye": session.eye,
                "session_name": session.session_name,
                "session_code": session.session_code,
                "frame_idx": frame_index(src_frame),
                "timestamp": int(ellipse["t"]),
                "source_frame": str(src_frame),
                "frame128_path": relative(ev_frame_path, ev_root),
                "mask128_path": relative(ev_mask_path, ev_root),
                "event64_path": relative(ev_event_path, ev_root),
                "gt_x_orig": f"{ellipse['x']:.8f}",
                "gt_y_orig": f"{ellipse['y']:.8f}",
                "gt_a_orig": f"{ellipse['a']:.8f}",
                "gt_b_orig": f"{ellipse['b']:.8f}",
                "gt_ang": f"{ellipse['ang']:.8f}",
                "gt_x_128": f"{ellipse128[0]:.8f}",
                "gt_y_128": f"{ellipse128[1]:.8f}",
                "gt_a_128": f"{ellipse128[2]:.8f}",
                "gt_b_128": f"{ellipse128[3]:.8f}",
                "gt_x_64": f"{ellipse64[0]:.8f}",
                "gt_y_64": f"{ellipse64[1]:.8f}",
                "gt_a_64": f"{ellipse64[2]:.8f}",
                "gt_b_64": f"{ellipse64[3]:.8f}",
                "event_count": int(len(events)),
            }
            metadata_rows.append(row)
            ex_data_list.append(
                {
                    "img_shape": [FRAME_SIZE, FRAME_SIZE],
                    "img_id": emitted,
                    "timestamp": int(ellipse["t"]),
                    "img_filename": relative(ex_frame_path, ex_root),
                    "input_volume": relative(ex_event_path, ex_root),
                    "event_patches": relative(ex_patch_path, ex_root),
                    "sample_regions": regions,
                    "pre_state": ellipse64,
                    "eye_region": [0, 0, FRAME_SIZE, FRAME_SIZE],
                    "pupil": ellipse128,
                    "subject": session.subject,
                    "eye": session.eye,
                    "session": session.session_code,
                    "frame_idx": frame_index(src_frame),
                    "source_global_idx": global_idx,
                }
            )
            emitted += 1
    return metadata_rows, ex_data_list


def write_exgaze_json(path: Path, split: str, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metainfo": {
            "dataset_name": "HBTXR hybrid EX-Gaze adaptation",
            "task_name": "near eye pupil detection and event displacement",
            "classes": ["eye_region", "pupil"],
            "frame_size": [FRAME_SIZE, FRAME_SIZE],
            "event_size": [EVENT_SIZE, EVENT_SIZE],
            "frame_rate": 25,
            "split": split,
        },
        "data_list": rows,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--raw-davis-root", type=Path, default=DEFAULT_RAW_DAVIS_ROOT)
    parser.add_argument("--target-base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument("--tag", default="smoke_32")
    parser.add_argument("--max-samples-per-split", type=int, default=32)
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ev_root = args.target_base / "EV_Eye_Hybrid_frame128_event64_subject_independent" / args.tag
    ex_root = args.target_base / "EX_Gaze_Hybrid_frame128_event64_subject_independent" / args.tag
    ranges = build_ranges(args.source_root)
    ranges_by_split = defaultdict(list)
    for item in ranges:
        ranges_by_split[item.split].append(item)

    summary = {
        "source_root": str(args.source_root),
        "raw_davis_root": str(args.raw_davis_root),
        "ev_eye_root": str(ev_root),
        "ex_gaze_root": str(ex_root),
        "tag": args.tag,
        "max_samples_per_split": args.max_samples_per_split,
        "splits": args.splits,
        "split_counts": {},
        "contracts": {
            "frame": "1 x 128 x 128 PNG",
            "event": "2 x 64 x 64 uint16 NPY count volume",
            "mask": "1 x 128 x 128 PNG mask from scaled ellipse",
            "exgaze_patches": "8 x 2 x 16 x 16 uint16 NPY",
        },
    }

    for split in args.splits:
        cache = CachedSplit(args.source_root / split)
        metadata_rows, ex_rows = export_split(
            split=split,
            ranges=ranges_by_split[split],
            cache=cache,
            raw_davis_root=args.raw_davis_root,
            ev_root=ev_root,
            ex_root=ex_root,
            max_samples=args.max_samples_per_split,
        )
        write_rows_csv(ev_root / split / "labels.csv", metadata_rows)
        write_rows_csv(ex_root / split / "labels.csv", metadata_rows)
        write_exgaze_json(ex_root / split / "annotations.json", split, ex_rows)
        summary["split_counts"][split] = len(metadata_rows)

    ev_root.mkdir(parents=True, exist_ok=True)
    ex_root.mkdir(parents=True, exist_ok=True)
    (ev_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    (ex_root / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
