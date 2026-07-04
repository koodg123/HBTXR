from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_TARGET_BASE = Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data")


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def validate_split(root: Path, split: str, expected_count: int) -> dict:
    labels_path = root / split / "labels.csv"
    rows = read_rows(labels_path)
    if len(rows) != expected_count:
        raise AssertionError(f"{labels_path}: expected {expected_count}, got {len(rows)}")
    first = rows[0]
    frame = cv2.imread(str(root / first["frame128_path"]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(str(root / first["mask128_path"]), cv2.IMREAD_GRAYSCALE)
    event = np.load(root / first["event64_path"])
    if frame.shape != (128, 128) or frame.dtype != np.uint8:
        raise AssertionError(f"bad frame shape/dtype: {frame.shape} {frame.dtype}")
    if mask.shape != (128, 128) or mask.dtype != np.uint8:
        raise AssertionError(f"bad mask shape/dtype: {mask.shape} {mask.dtype}")
    if event.shape != (2, 64, 64) or event.dtype != np.uint16:
        raise AssertionError(f"bad event shape/dtype: {event.shape} {event.dtype}")
    return {
        "rows": len(rows),
        "frame_shape": list(frame.shape),
        "frame_dtype": str(frame.dtype),
        "mask_shape": list(mask.shape),
        "mask_dtype": str(mask.dtype),
        "event_shape": list(event.shape),
        "event_dtype": str(event.dtype),
        "first_event_sum": int(event.sum()),
    }


def validate_exgaze_split(root: Path, split: str, expected_count: int) -> dict:
    ann_path = root / split / "annotations.json"
    ann = json.loads(ann_path.read_text())
    data_list = ann["data_list"]
    if len(data_list) != expected_count:
        raise AssertionError(f"{ann_path}: expected {expected_count}, got {len(data_list)}")
    first = data_list[0]
    event = np.load(root / first["input_volume"])
    patches = np.load(root / first["event_patches"])
    if event.shape != (2, 64, 64) or event.dtype != np.uint16:
        raise AssertionError(f"bad EX-Gaze event shape/dtype: {event.shape} {event.dtype}")
    if patches.shape != (8, 2, 16, 16) or patches.dtype != np.uint16:
        raise AssertionError(f"bad EX-Gaze patch shape/dtype: {patches.shape} {patches.dtype}")
    regions = np.asarray(first["sample_regions"])
    if regions.shape != (8, 4):
        raise AssertionError(f"bad sample region shape: {regions.shape}")
    if regions.min() < 0 or regions.max() > 64:
        raise AssertionError(f"sample regions out of 64x64 bounds: min={regions.min()} max={regions.max()}")
    return {
        "annotation_rows": len(data_list),
        "event_shape": list(event.shape),
        "event_dtype": str(event.dtype),
        "patch_shape": list(patches.shape),
        "patch_dtype": str(patches.dtype),
        "sample_regions_shape": list(regions.shape),
        "sample_regions_min": int(regions.min()),
        "sample_regions_max": int(regions.max()),
    }


def validate(tag: str, expected_count: int, target_base: Path) -> dict:
    ev_root = target_base / "EV_Eye_Hybrid_frame128_event64_subject_independent" / tag
    ex_root = target_base / "EX_Gaze_Hybrid_frame128_event64_subject_independent" / tag
    if not ev_root.exists():
        raise FileNotFoundError(ev_root)
    if not ex_root.exists():
        raise FileNotFoundError(ex_root)

    result = {
        "tag": tag,
        "expected_count_per_split": expected_count,
        "ev_eye_root": str(ev_root),
        "ex_gaze_root": str(ex_root),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        result["splits"][split] = {
            "ev_eye": validate_split(ev_root, split, expected_count),
            "ex_gaze": validate_exgaze_split(ex_root, split, expected_count),
        }
    return result


def render_markdown(result: dict) -> str:
    lines = [
        "# EV-Eye and EX-Gaze Hybrid Export Validation",
        "",
        "Date: 2026-07-04",
        "",
        f"- Tag: `{result['tag']}`",
        f"- Expected count per split: `{result['expected_count_per_split']}`",
        f"- EV-Eye root: `{result['ev_eye_root']}`",
        f"- EX-Gaze root: `{result['ex_gaze_root']}`",
        "",
        "| Split | EV Rows | Frame | Mask | Event | EX Ann Rows | EX Event | EX Patch | Region Range |",
        "|---|---:|---|---|---|---:|---|---|---|",
    ]
    for split, data in result["splits"].items():
        ev = data["ev_eye"]
        ex = data["ex_gaze"]
        lines.append(
            f"| {split} | {ev['rows']} | `{ev['frame_shape']} {ev['frame_dtype']}` | "
            f"`{ev['mask_shape']} {ev['mask_dtype']}` | "
            f"`{ev['event_shape']} {ev['event_dtype']}` | {ex['annotation_rows']} | "
            f"`{ex['event_shape']} {ex['event_dtype']}` | "
            f"`{ex['patch_shape']} {ex['patch_dtype']}` | "
            f"{ex['sample_regions_min']}..{ex['sample_regions_max']} |"
        )
    lines.extend(
        [
            "",
            "## Judgment",
            "",
            "- Validation passed for row counts, frame shape, mask shape, event shape, EX-Gaze patch shape, and patch-region bounds.",
            "- This validates the smoke package only. Full export still needs a separate run and validation pass.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="smoke_32")
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--target-base", type=Path, default=DEFAULT_TARGET_BASE)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_smoke_validation_2026-07-04.json"),
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("references/report/EV-Eye/EV_Eye_EX_Gaze_hybrid_smoke_validation_2026-07-04.md"),
    )
    parser.add_argument(
        "--mirror-markdown-output",
        type=Path,
        default=Path("references/report/EX-Gaze/EV_Eye_EX_Gaze_hybrid_smoke_validation_2026-07-04.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate(args.tag, args.expected_count, args.target_base)
    markdown = render_markdown(result)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2) + "\n")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(markdown + "\n")
    if args.mirror_markdown_output:
        args.mirror_markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.mirror_markdown_output.write_text(markdown + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
