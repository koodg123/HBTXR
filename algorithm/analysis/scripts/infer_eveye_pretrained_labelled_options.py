from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
EV_EYE_CODE = ROOT / "references/codebase/software/EV-Eye"
ANNOTATION_EV_EYE_CODE = ROOT.parent / "HBTXR-Annotation/HBTXR/references/codebase/software/EV-Eye"
if EV_EYE_CODE.exists():
    sys.path.insert(0, str(EV_EYE_CODE))
elif ANNOTATION_EV_EYE_CODE.exists():
    sys.path.insert(0, str(ANNOTATION_EV_EYE_CODE))
else:
    raise RuntimeError("Cannot find EV-Eye codebase for original UNet import")

from unet import UNet  # noqa: E402


SENSOR_W = 346
SENSOR_H = 260
SESSIONS = ("1_0_2", "2_0_1", "2_0_2")
SESSION_CODES = {"1_0_2": "102", "2_0_1": "201", "2_0_2": "202"}
FIELDNAMES = [
    "sample_idx",
    "subject",
    "eye",
    "session_name",
    "session_code",
    "frame_idx",
    "timestamp",
    "option",
    "input_w",
    "input_h",
    "gt_x_input",
    "gt_y_input",
    "pred_x_input",
    "pred_y_input",
    "error_input_px",
    "gt_x_original",
    "gt_y_original",
    "pred_x_original",
    "pred_y_original",
    "error_original_px",
    "mask_iou",
    "motion_state",
]


def parse_filename(filename: str) -> tuple[int, int]:
    stem = Path(filename).stem
    frame_str, ts_str = stem.split("_", 1)
    return int(frame_str), int(ts_str)


def labelled_csv_rows(data_root: Path, subject: int, eye: str, session: str) -> list[dict]:
    csv_path = data_root / f"user{subject}" / eye / f"session_{session}" / f"user_{subject}.csv"
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["region_shape_attributes"] == "{}":
                continue
            attrs = json.loads(row["region_shape_attributes"])
            if attrs.get("name") != "ellipse":
                continue
            try:
                frame_idx, timestamp = parse_filename(row["filename"])
            except ValueError:
                continue
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "cx": float(attrs["cx"]),
                    "cy": float(attrs["cy"]),
                    "rx": float(attrs.get("rx", 0.0)),
                    "ry": float(attrs.get("ry", 0.0)),
                    "theta": float(attrs.get("theta", 0.0)),
                }
            )
    return rows


def load_h5_samples(labelled_root: Path, subject: int, eye: str, session: str) -> tuple[np.ndarray, np.ndarray]:
    path = labelled_root / eye / f"user{subject}_session_{session}.h5"
    with h5py.File(path, "r") as h5:
        frames = np.transpose(h5["data"][:], (2, 1, 0))
        masks = np.transpose(h5["label"][:], (2, 1, 0))
    return frames, (masks > 0).astype(np.uint8)


def resize_batch(batch: np.ndarray, width: int, height: int, mask: bool) -> np.ndarray:
    if batch.shape[1] == height and batch.shape[2] == width:
        return batch.astype(np.uint8, copy=False)
    interp = cv2.INTER_NEAREST if mask else cv2.INTER_AREA
    out = np.empty((batch.shape[0], height, width), dtype=np.uint8)
    for idx, item in enumerate(batch):
        out[idx] = cv2.resize(item.astype(np.uint8), (width, height), interpolation=interp)
    return out


def centroid_from_probs(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, w = probs.shape
    ys = torch.arange(h, device=probs.device, dtype=probs.dtype).view(1, h, 1)
    xs = torch.arange(w, device=probs.device, dtype=probs.dtype).view(1, 1, w)
    denom = probs.sum(dim=(1, 2)).clamp_min(1e-6)
    pred_x = (probs * xs).sum(dim=(1, 2)) / denom
    pred_y = (probs * ys).sum(dim=(1, 2)) / denom
    return pred_x, pred_y


def mask_iou(logits: torch.Tensor, masks: torch.Tensor) -> list[float]:
    pred = torch.softmax(logits, dim=1)[:, 1] > 0.5
    target = masks.bool()
    inter = (pred & target).sum(dim=(1, 2)).float()
    union = (pred | target).sum(dim=(1, 2)).float()
    return (inter / union.clamp_min(1.0)).detach().cpu().tolist()


def load_motion_labels(path: Path) -> dict[tuple[int, str, str, int], str]:
    labels = {}
    if not path.exists():
        return labels
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["subject"]), row["eye"], str(row["session_code"]), int(row["frame_idx"]))
            labels[key] = row["motion_state"]
    return labels


def describe(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": math.nan, "median": math.nan, "p95": math.nan, "p99": math.nan}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def write_stats(output_dir: Path, option: str, predictions: list[dict]) -> None:
    subject_errors: dict[int, list[float]] = defaultdict(list)
    motion_errors: dict[str, list[float]] = defaultdict(list)
    subject_motion_errors: dict[tuple[int, str], list[float]] = defaultdict(list)
    counts: dict[tuple[int, str], int] = defaultdict(int)
    for row in predictions:
        err = float(row["error_original_px"])
        subject = int(row["subject"])
        motion = row["motion_state"]
        subject_errors[subject].append(err)
        if motion:
            motion_errors[motion].append(err)
            subject_motion_errors[(subject, motion)].append(err)
            counts[(subject, motion)] += 1

    by_subject_motion = output_dir / f"EV-Eye_{option}_subject37_48_error_distribution_by_subject_motion.csv"
    with by_subject_motion.open("w", newline="") as f:
        fields = ["subject", "motion_state", "n", "mean", "median", "p95", "p99"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(subject_motion_errors):
            stats = describe(subject_motion_errors[key])
            writer.writerow({"subject": key[0], "motion_state": key[1], **stats})

    by_subject = output_dir / f"EV-Eye_{option}_subject37_48_subject_error_stats.csv"
    with by_subject.open("w", newline="") as f:
        fields = ["subject", "n", "mean", "median", "p95", "p99"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for subject in sorted(subject_errors):
            writer.writerow({"subject": subject, **describe(subject_errors[subject])})

    by_motion = output_dir / f"EV-Eye_{option}_subject37_48_motion_error_stats.csv"
    with by_motion.open("w", newline="") as f:
        fields = ["motion_state", "n", "mean", "median", "p95", "p99"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for motion in sorted(motion_errors):
            writer.writerow({"motion_state": motion, **describe(motion_errors[motion])})

    joined_counts = output_dir / f"EV-Eye_{option}_subject37_48_joined_motion_counts.csv"
    with joined_counts.open("w", newline="") as f:
        fields = ["subject", "motion_state", "count"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (subject, motion), count in sorted(counts.items()):
            writer.writerow({"subject": subject, "motion_state": motion, "count": count})

    report = output_dir / f"EV-Eye_{option}_subject37_48_error_distribution_report.md"
    overall = describe([float(row["error_original_px"]) for row in predictions])
    lines = [
        f"# EV-Eye {option} Subject 37-48 Error Distribution",
        "",
        "## Overall",
        "",
        "| n | mean | median | p95 | p99 |",
        "|---:|---:|---:|---:|---:|",
        f"| {overall['n']} | {overall['mean']:.6f} | {overall['median']:.6f} | {overall['p95']:.6f} | {overall['p99']:.6f} |",
        "",
        "Error unit is original DAVIS pixel coordinates after rescaling predictions back to 346x260.",
        "Motion labels use the NoBlink row-label package when a labelled frame matches the motion table.",
    ]
    report.write_text("\n".join(lines) + "\n")


def run_option(args: argparse.Namespace, option: str, width: int, height: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    motion_labels = load_motion_labels(args.motion_labels)
    pred_path = output_dir / f"EV-Eye_{option}_subject37_48_test_predictions_with_metadata.csv"
    predictions = []
    sample_idx = 0
    device = torch.device(args.device)
    model_cache: dict[tuple[str, int], UNet] = {}

    with pred_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for eye in ("left", "right"):
            for subject in range(37, 49):
                model = UNet(n_channels=1, n_classes=2, bilinear=False).to(device)
                state = torch.load(args.pretrained_root / eye / f"user{subject}.pth", map_location="cpu", weights_only=False)
                model.load_state_dict(state)
                model.eval()
                model_cache[(eye, subject)] = model
                for session in SESSIONS:
                    frames, masks = load_h5_samples(args.labelled_root, subject, eye, session)
                    labels = labelled_csv_rows(args.data_root, subject, eye, session)
                    n = min(len(frames), len(labels))
                    if len(frames) != len(labels):
                        print(f"warning: h5/csv count mismatch subject={subject} eye={eye} session={session}: {len(frames)} vs {len(labels)}", flush=True)
                    frames_r = resize_batch(frames[:n], width, height, mask=False)
                    masks_r = resize_batch(masks[:n], width, height, mask=True)
                    for start in range(0, n, args.batch_size):
                        end = min(start + args.batch_size, n)
                        image_t = torch.from_numpy(frames_r[start:end]).float().div(255.0).unsqueeze(1).to(device)
                        mask_t = torch.from_numpy(masks_r[start:end]).long().to(device)
                        with torch.no_grad():
                            logits = model(image_t)
                            probs = torch.softmax(logits, dim=1)[:, 1]
                            pred_x_t, pred_y_t = centroid_from_probs(probs)
                            ious = mask_iou(logits, mask_t)
                        pred_x = pred_x_t.detach().cpu().numpy()
                        pred_y = pred_y_t.detach().cpu().numpy()
                        for local_idx, label in enumerate(labels[start:end]):
                            idx = start + local_idx
                            gt_x = float(label["cx"]) * width / SENSOR_W
                            gt_y = float(label["cy"]) * height / SENSOR_H
                            px = float(pred_x[local_idx])
                            py = float(pred_y[local_idx])
                            px_orig = px * SENSOR_W / width
                            py_orig = py * SENSOR_H / height
                            error_input = math.hypot(px - gt_x, py - gt_y)
                            error_orig = math.hypot(px_orig - float(label["cx"]), py_orig - float(label["cy"]))
                            session_code = SESSION_CODES[session]
                            motion = motion_labels.get((subject, eye, session_code, int(label["frame_idx"])), "")
                            row = {
                                "sample_idx": sample_idx,
                                "subject": subject,
                                "eye": eye,
                                "session_name": f"session_{session}",
                                "session_code": session_code,
                                "frame_idx": int(label["frame_idx"]),
                                "timestamp": int(label["timestamp"]),
                                "option": option,
                                "input_w": width,
                                "input_h": height,
                                "gt_x_input": gt_x,
                                "gt_y_input": gt_y,
                                "pred_x_input": px,
                                "pred_y_input": py,
                                "error_input_px": error_input,
                                "gt_x_original": float(label["cx"]),
                                "gt_y_original": float(label["cy"]),
                                "pred_x_original": px_orig,
                                "pred_y_original": py_orig,
                                "error_original_px": error_orig,
                                "mask_iou": float(ious[local_idx]),
                                "motion_state": motion,
                            }
                            writer.writerow(row)
                            predictions.append(row)
                            sample_idx += 1
                del model_cache[(eye, subject)]
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    write_stats(output_dir, option, predictions)
    metadata = {
        "option": option,
        "input_width": width,
        "input_height": height,
        "event_shape": [2, 64, 64],
        "event_usage": "not consumed by EV-Eye pretrained UNet; retained as comparison-option metadata",
        "pretrained_root": str(args.pretrained_root),
        "labelled_root": str(args.labelled_root),
        "data_root": str(args.data_root),
        "motion_labels": str(args.motion_labels),
        "num_predictions": len(predictions),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labelled-root", type=Path, default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis_labelled_with_mask"))
    parser.add_argument("--data-root", type=Path, default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis"))
    parser.add_argument("--pretrained-root", type=Path, default=Path("/home/kjm26/project/dataset/XR/EV_Eye/processed_data/Pre-trained_models"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis/RESULTS/EV-Eye")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--motion-labels", type=Path, default=ROOT / "analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv")
    parser.add_argument("--options", nargs="+", choices=["original_resolution", "frame128_event64", "frame64_event64"], default=["original_resolution", "frame128_event64", "frame64_event64"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    option_shapes = {
        "original_resolution": (SENSOR_W, SENSOR_H),
        "frame128_event64": (128, 128),
        "frame64_event64": (64, 64),
    }
    for option in args.options:
        width, height = option_shapes[option]
        run_option(args, option, width, height, args.output_root / option)


if __name__ == "__main__":
    main()
