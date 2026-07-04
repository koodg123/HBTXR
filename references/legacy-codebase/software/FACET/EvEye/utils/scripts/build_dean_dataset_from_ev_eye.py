import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: h5py. Install it in the active Python environment "
        "before running this script."
    ) from exc


EVENT_DTYPE = np.dtype(
    [("t", np.int64), ("x", np.int64), ("y", np.int64), ("p", np.int64)]
)
ELLIPSE_DTYPE = np.dtype(
    [
        ("t", np.int64),
        ("x", np.float64),
        ("y", np.float64),
        ("a", np.float64),
        ("b", np.float64),
        ("ang", np.float64),
    ]
)


def natural_key(path: Path):
    parts = []
    for part in path.as_posix().replace("_", "/").replace(".", "/").split("/"):
        parts.append(int(part) if part.isdigit() else part)
    return parts


def parse_frame_timestamp(frame_path: Path) -> int:
    return int(frame_path.stem.split("_")[-1])


def load_events(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=EVENT_DTYPE)


def ellipse_from_mask(mask: np.ndarray, timestamp: int, min_pixels: int):
    yy, xx = np.nonzero(mask > 0)
    if xx.size < min_pixels:
        return None

    x_mean = float(xx.mean())
    y_mean = float(yy.mean())
    coords = np.stack([xx - x_mean, yy - y_mean], axis=0).astype(np.float64)
    cov = coords @ coords.T / max(coords.shape[1], 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    major = 4.0 * math.sqrt(max(float(eigvals[0]), 0.0))
    minor = 4.0 * math.sqrt(max(float(eigvals[1]), 0.0))
    if major < min_pixels or minor < 1:
        return None

    vec = eigvecs[:, 0]
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0])))
    while angle > 90:
        angle -= 180
    while angle < -90:
        angle += 180

    return (timestamp, x_mean, y_mean, major, minor, angle)


def iter_h5_samples(labelled_root: Path, data_root: Path, min_pixels: int):
    h5_paths = sorted(labelled_root.glob("*/*.h5"), key=natural_key)
    for h5_path in tqdm(h5_paths, desc="Reading labelled h5 files"):
        side = h5_path.parent.name
        user, session = h5_path.stem.split("_", 1)
        session_root = data_root / user / side / session
        events_path = session_root / "events" / "events.txt"
        frames_dir = session_root / "frames"
        if not events_path.exists() or not frames_dir.exists():
            yield {
                "kind": "missing_source",
                "h5": str(h5_path),
                "events_path": str(events_path),
                "frames_dir": str(frames_dir),
            }
            continue

        frame_paths = sorted(frames_dir.glob("*.png"), key=natural_key)
        with h5py.File(h5_path, "r") as h5_file:
            if "label" not in h5_file:
                yield {"kind": "missing_label", "h5": str(h5_path)}
                continue
            labels = np.asarray(h5_file["label"]).transpose(1, 0, 2)

        count = min(labels.shape[-1], len(frame_paths))
        if count == 0:
            yield {"kind": "empty_source", "h5": str(h5_path)}
            continue

        events = load_events(events_path)
        for index in range(count):
            timestamp = parse_frame_timestamp(frame_paths[index])
            ellipse = ellipse_from_mask(labels[..., index], timestamp, min_pixels)
            if ellipse is None:
                continue
            end = np.searchsorted(events["t"], timestamp)
            start = max(0, end - 5000)
            event_segment = events[start:end]
            if event_segment.shape[0] == 0:
                continue
            yield {
                "kind": "sample",
                "source": f"{user}/{side}/{session}",
                "frame": frame_paths[index].name,
                "events": event_segment,
                "ellipse": np.array(ellipse, dtype=ELLIPSE_DTYPE),
            }


def create_memmap(data: np.ndarray, data_file: Path, info_file: Path):
    mmap = np.memmap(data_file, dtype=data.dtype, mode="w+", shape=data.shape)
    mmap[:] = data
    mmap.flush()
    with info_file.open("w", encoding="utf-8") as info:
        info.write(f"Data shape: {data.shape}\n")
        info.write(f"Data dtype: {data.dtype}\n")


def merge_structured(arrays):
    total = sum(array.shape[0] if array.shape else 1 for array in arrays)
    merged = np.zeros(total, dtype=arrays[0].dtype)
    indices = np.zeros((len(arrays), 2), dtype=np.int32)
    offset = 0
    for index, array in enumerate(arrays):
        length = array.shape[0] if array.shape else 1
        merged[offset : offset + length] = array
        indices[index] = [offset, offset + length]
        offset += length
    return merged, indices


def write_split(split: str, samples, output_root: Path, batch_size: int):
    split_root = output_root / split
    data_root = split_root / "cached_data"
    ellipse_root = split_root / "cached_ellipse"
    data_root.mkdir(parents=True, exist_ok=True)
    ellipse_root.mkdir(parents=True, exist_ok=True)

    ellipses = np.zeros(len(samples), dtype=ELLIPSE_DTYPE)
    event_batch = []
    batch_id = 0
    for index, sample in enumerate(tqdm(samples, desc=f"Writing {split} events")):
        event_batch.append(sample["events"])
        ellipses[index] = sample["ellipse"]
        is_last = index == len(samples) - 1
        if len(event_batch) == batch_size or is_last:
            events_merged, event_indices = merge_structured(event_batch)
            create_memmap(
                events_merged,
                data_root / f"events_batch_{batch_id}.memmap",
                data_root / f"events_batch_info_{batch_id}.txt",
            )
            np.save(data_root / f"events_indices_{batch_id}.npy", event_indices)
            event_batch = []
            batch_id += 1

    ellipse_indices = np.array(
        [[index, index + 1] for index in range(len(samples))], dtype=np.int32
    )
    create_memmap(
        ellipses,
        ellipse_root / "ellipses_batch_0.memmap",
        ellipse_root / "ellipses_batch_info_0.txt",
    )
    np.save(ellipse_root / "ellipses_indices_0.npy", ellipse_indices)


def main():
    parser = argparse.ArgumentParser(
        description="Build a FACET DavisEyeEllipseDataset-compatible DeanDataset."
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=5000)
    parser.add_argument("--min-mask-pixels", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    labelled_root = args.raw_root / "Data_davis_labelled_with_mask"
    data_root = args.raw_root / "Data_davis"
    if not labelled_root.exists():
        raise FileNotFoundError(labelled_root)
    if not data_root.exists():
        raise FileNotFoundError(data_root)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"{args.output_root} is not empty. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    samples = []
    skipped = []
    for item in iter_h5_samples(labelled_root, data_root, args.min_mask_pixels):
        if item["kind"] == "sample":
            samples.append(item)
        else:
            skipped.append(item)

    if not samples:
        raise SystemExit("No usable samples were produced.")

    split_index = int(len(samples) * args.train_ratio)
    train_samples = samples[:split_index]
    val_samples = samples[split_index:]
    write_split("train", train_samples, args.output_root, args.batch_size)
    write_split("val", val_samples, args.output_root, args.batch_size)

    manifest = {
        "source": "Data_davis_labelled_with_mask h5 masks + Data_davis events",
        "note": (
            "This builds the locally reproducible labelled subset. The paper's "
            "U-Net-expanded DeanDataset requires generated segmentation masks or "
            "a compatible U-Net checkpoint, which is not embedded in this script."
        ),
        "raw_root": str(args.raw_root),
        "output_root": str(args.output_root),
        "train_ratio": args.train_ratio,
        "batch_size": args.batch_size,
        "num_samples": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_skipped_sources": len(skipped),
        "skipped_sources_preview": skipped[:20],
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)
        out.write("\n")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
