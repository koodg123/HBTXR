import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: h5py. Install it in the active Python environment "
        "before running this script."
    ) from exc


def natural_key(path: Path):
    parts = []
    for part in path.as_posix().replace("_", "/").replace(".", "/").split("/"):
        parts.append(int(part) if part.isdigit() else part)
    return parts


def read_h5_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        if "data" not in h5_file or "label" not in h5_file:
            raise KeyError(f"{path} must contain 'data' and 'label' datasets")
        data = np.asarray(h5_file["data"]).transpose(1, 0, 2)
        label = np.asarray(h5_file["label"]).transpose(1, 0, 2)
    if data.shape != label.shape:
        raise ValueError(f"data/label shape mismatch in {path}: {data.shape} vs {label.shape}")
    if data.ndim != 3:
        raise ValueError(f"expected HxWxN arrays in {path}, got {data.shape}")
    return data, label


def assign_split(index: int, count: int, train_ratio: float) -> str:
    return "train" if index < int(count * train_ratio) else "val"


def count_usable_samples(h5_paths: list[Path], min_mask_pixels: int) -> int:
    usable = 0
    for h5_path in tqdm(h5_paths, desc="Counting usable masks"):
        _, label = read_h5_arrays(h5_path)
        for frame_index in range(label.shape[-1]):
            if np.count_nonzero(label[..., frame_index] > 0) >= min_mask_pixels:
                usable += 1
    return usable


def save_png(array: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def build_dataset(args):
    h5_paths = sorted(args.labelled_root.glob("*/*.h5"), key=natural_key)
    if args.max_files:
        h5_paths = h5_paths[: args.max_files]
    if not h5_paths:
        raise FileNotFoundError(f"No h5 files found under {args.labelled_root}")

    if args.output_root.exists() and any(args.output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"{args.output_root} is not empty. Re-run with --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "labelled_root": str(args.labelled_root),
                    "output_root": str(args.output_root),
                    "num_h5_files": len(h5_paths),
                    "split_unit": args.split_unit,
                    "train_ratio": args.train_ratio,
                    "preview": [str(path) for path in h5_paths[:10]],
                },
                indent=2,
            )
        )
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    split_counts = {"train": 0, "val": 0}
    skipped_empty_masks = 0
    skipped_shape_errors = []
    sample_records = []
    total_frames_seen = 0
    files_for_split = {
        path: assign_split(index, len(h5_paths), args.train_ratio)
        for index, path in enumerate(h5_paths)
    }
    usable_sample_index = 0
    sample_split_index = None
    total_usable_for_sample_split = None
    if args.split_unit == "sample":
        total_usable_for_sample_split = count_usable_samples(
            h5_paths, args.min_mask_pixels
        )
        sample_split_index = int(total_usable_for_sample_split * args.train_ratio)

    for file_index, h5_path in enumerate(tqdm(h5_paths, desc="Converting h5 masks")):
        try:
            data, label = read_h5_arrays(h5_path)
        except Exception as exc:
            skipped_shape_errors.append({"path": str(h5_path), "error": str(exc)})
            continue

        side = h5_path.parent.name
        stem = h5_path.stem
        for frame_index in range(data.shape[-1]):
            total_frames_seen += 1
            mask = label[..., frame_index]
            if np.count_nonzero(mask > 0) < args.min_mask_pixels:
                skipped_empty_masks += 1
                continue

            split = (
                files_for_split[h5_path]
                if args.split_unit == "file"
                else (
                    "train"
                    if usable_sample_index < sample_split_index
                    else "val"
                )
            )
            usable_sample_index += 1
            basename = f"{side}_{stem}_{frame_index:04d}.png"
            image_out = args.output_root / split / "data" / basename
            mask_out = args.output_root / split / "label" / basename
            image = data[..., frame_index]
            binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            save_png(image, image_out)
            save_png(binary_mask, mask_out)
            split_counts[split] += 1
            if len(sample_records) < args.sample_count:
                sample_records.append(
                    {
                        "split": split,
                        "image": str(image_out),
                        "mask": str(mask_out),
                        "source_h5": str(h5_path),
                        "frame_index": frame_index,
                    }
                )

    manifest = {
        "source": "Data_davis_labelled_with_mask h5 data/label arrays",
        "labelled_root": str(args.labelled_root),
        "output_root": str(args.output_root),
        "format": "DavisWithMaskDataset PNG layout: split/data/*.png and split/label/*.png",
        "split_unit": args.split_unit,
        "train_ratio": args.train_ratio,
        "min_mask_pixels": args.min_mask_pixels,
        "num_h5_files": len(h5_paths),
        "total_frames_seen": total_frames_seen,
        "num_train": split_counts["train"],
        "num_val": split_counts["val"],
        "num_samples": split_counts["train"] + split_counts["val"],
        "total_usable_for_sample_split": total_usable_for_sample_split,
        "skipped_empty_masks": skipped_empty_masks,
        "skipped_shape_errors": skipped_shape_errors[:20],
        "sample_records": sample_records,
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as out:
        json.dump(manifest, out, indent=2)
        out.write("\n")
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Build a FACET DavisWithMaskDataset-compatible U-Net PNG dataset."
    )
    parser.add_argument(
        "--labelled-root",
        type=Path,
        default=Path(
            "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis_labelled_with_mask"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset"
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-unit", choices=["file", "sample"], default="file")
    parser.add_argument("--min-mask-pixels", type=int, default=20)
    parser.add_argument("--sample-count", type=int, default=20)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(args)


if __name__ == "__main__":
    main()
