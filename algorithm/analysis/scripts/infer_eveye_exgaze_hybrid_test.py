#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.scripts.train_eveye_exgaze_hybrid import (  # noqa: E402
    EVENT_SIZE,
    FRAME_SIZE,
    build_tracking_pairs,
    build_model,
    ellipse128_from_row,
    ellipse64_from_row,
    event_patches,
    make_event64,
    make_mask128,
    read_rows,
    sample_regions_from_ellipse64,
)

FACET_ROOT = ROOT / "references/codebase/software/FACET"
if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.utils.cache.MemmapCacheStructedEvents import load_event_segment  # noqa: E402


FIELDNAMES = [
    "sample_idx",
    "source_global_idx",
    "split",
    "subject",
    "eye",
    "session_name",
    "session_code",
    "frame_idx",
    "timestamp",
    "gt_x_input64",
    "gt_y_input64",
    "pred_x_input64",
    "pred_y_input64",
    "error_input64_px",
    "gt_x_128",
    "gt_y_128",
    "pred_x_128",
    "pred_y_128",
    "gt_a_128",
    "gt_b_128",
    "gt_ang",
    "pred_a_128",
    "pred_b_128",
    "pred_ang",
    "mask_iou_128",
    "motion_state",
]


class HybridTestDataset(Dataset):
    def __init__(self, dataset_root: Path, source_root: Path, model_name: str, limit: int | None = None):
        self.dataset_root = dataset_root
        self.source_root = source_root
        self.model_name = model_name
        self.rows = read_rows(dataset_root / "test" / "labels.csv", limit=limit)
        self.pairs = build_tracking_pairs(self.rows) if model_name == "ex-gaze" else []
        self.data_root = source_root / "test" / "cached_data"

    def __len__(self) -> int:
        return len(self.pairs) if self.model_name == "ex-gaze" else len(self.rows)

    def __getitem__(self, index: int) -> dict:
        if self.model_name == "ex-gaze":
            row, pre_row = self.pairs[index]
        else:
            row = self.rows[index]
            pre_row = None
        frame = cv2.imread(row["source_frame"], cv2.IMREAD_GRAYSCALE)
        if frame is None:
            raise RuntimeError(f"Cannot read frame: {row['source_frame']}")
        frame128 = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        ellipse128 = ellipse128_from_row(row)
        event64 = make_event64(load_event_segment(int(row["source_global_idx"]), self.data_root))
        item = {
            "frame": torch.from_numpy(frame128.astype(np.float32) / 255.0).unsqueeze(0),
            "event": torch.from_numpy(event64.astype(np.float32)),
            "sample_idx": int(row["sample_idx"]),
            "source_global_idx": int(row["source_global_idx"]),
            "subject": int(row["subject"]),
            "eye": row["eye"],
            "session_name": row["session_name"],
            "session_code": str(row["session_code"]),
            "frame_idx": int(row["frame_idx"]),
            "timestamp": int(row["timestamp"]),
            "gt_x_input64": float(row["gt_x_64"]),
            "gt_y_input64": float(row["gt_y_64"]),
            "gt_x_128": ellipse128[0],
            "gt_y_128": ellipse128[1],
            "gt_a_128": ellipse128[2],
            "gt_b_128": ellipse128[3],
            "gt_ang": ellipse128[4],
        }
        if self.model_name == "ev-eye":
            item["mask"] = torch.from_numpy(make_mask128(ellipse128).astype(np.uint8))
            return item
        assert pre_row is not None
        pre_ellipse64 = ellipse64_from_row(pre_row)
        regions = sample_regions_from_ellipse64(pre_ellipse64)
        item["patches"] = torch.from_numpy(event_patches(event64, regions).astype(np.float32))
        item["pre_state"] = torch.tensor(pre_ellipse64, dtype=torch.float32)
        return item


def as_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)


def load_motion_labels(path: Path) -> dict[tuple[int, str, str, int], str]:
    labels = {}
    if not path.exists():
        return labels
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["subject"]), row["eye"], str(row["session_code"]), int(row["frame_idx"]))
            labels[key] = row["motion_state"]
    return labels


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def describe(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "p99": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }


def ev_eye_centers(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(logits, dim=1)[:, 1]
    b, h, w = probs.shape
    ys = torch.arange(h, device=probs.device, dtype=probs.dtype).view(1, h, 1)
    xs = torch.arange(w, device=probs.device, dtype=probs.dtype).view(1, 1, w)
    denom = probs.sum(dim=(1, 2)).clamp_min(1e-6)
    pred_x = (probs * xs).sum(dim=(1, 2)) / denom
    pred_y = (probs * ys).sum(dim=(1, 2)) / denom
    return pred_x, pred_y


def mask_iou(logits: torch.Tensor, mask: torch.Tensor) -> list[float]:
    pred = torch.softmax(logits, dim=1)[:, 1] > 0.5
    target = mask.bool()
    inter = (pred & target).sum(dim=(1, 2)).float()
    union = (pred | target).sum(dim=(1, 2)).float()
    return (inter / union.clamp_min(1.0)).detach().cpu().tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ev-eye", "ex-gaze"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit-samples", type=int)
    parser.add_argument(
        "--motion-labels",
        type=Path,
        default=ROOT / "analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = json.loads(args.config.read_text())
    dataset_root = Path(cfg["dataset"]["dataset_root"])
    source_root = Path(cfg["dataset"]["source_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    motion_labels = load_motion_labels(args.motion_labels)

    dataset = HybridTestDataset(dataset_root, source_root, args.model, args.limit_samples)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    model = build_model(args.model).to(device)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    model_tag = "EV-Eye" if args.model == "ev-eye" else "EX-Gaze"
    ckpt_epoch = int(ckpt.get("epoch", -1))
    ckpt_val_metric = ckpt.get("val_metric")
    pred_path = args.output_dir / f"{model_tag}_subject37_48_test_sample_predictions.csv"
    subject_errors: dict[int, list[float]] = defaultdict(list)
    motion_errors: dict[str, list[float]] = defaultdict(list)
    subject_motion_errors: dict[tuple[int, str], list[float]] = defaultdict(list)
    rows_written = 0
    rows_no_motion = 0

    with pred_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        with torch.no_grad():
            for batch in loader:
                frame = batch["frame"].to(device, non_blocking=True)
                subject = as_list(batch["subject"])
                eye = as_list(batch["eye"])
                session_name = as_list(batch["session_name"])
                session_code = [str(x) for x in as_list(batch["session_code"])]
                frame_idx = as_list(batch["frame_idx"])
                sample_idx = as_list(batch["sample_idx"])
                source_global_idx = as_list(batch["source_global_idx"])
                timestamp = as_list(batch["timestamp"])
                gt_x64 = as_list(batch["gt_x_input64"])
                gt_y64 = as_list(batch["gt_y_input64"])
                gt_x128 = as_list(batch["gt_x_128"])
                gt_y128 = as_list(batch["gt_y_128"])
                gt_a128 = as_list(batch["gt_a_128"])
                gt_b128 = as_list(batch["gt_b_128"])
                gt_ang = as_list(batch["gt_ang"])

                if args.model == "ev-eye":
                    event = batch["event"].to(device, non_blocking=True)
                    mask = batch["mask"].to(device, non_blocking=True)
                    logits = model(frame, event)
                    pred_x128_t, pred_y128_t = ev_eye_centers(logits)
                    pred_x128 = pred_x128_t.detach().cpu().tolist()
                    pred_y128 = pred_y128_t.detach().cpu().tolist()
                    pred_a128 = [float("nan")] * len(pred_x128)
                    pred_b128 = [float("nan")] * len(pred_x128)
                    pred_ang = [float("nan")] * len(pred_x128)
                    mask_ious = mask_iou(logits, mask)
                else:
                    patches = batch["patches"].to(device, non_blocking=True)
                    pre_state = batch["pre_state"].to(device, non_blocking=True)
                    pred = model(frame, patches, pre_state).detach().cpu().numpy()
                    pred_x128 = (pred[:, 0] * FRAME_SIZE).tolist()
                    pred_y128 = (pred[:, 1] * FRAME_SIZE).tolist()
                    pred_a128 = (pred[:, 2] * FRAME_SIZE).tolist()
                    pred_b128 = (pred[:, 3] * FRAME_SIZE).tolist()
                    pred_ang = (pred[:, 4] * 90.0).tolist()
                    mask_ious = [float("nan")] * len(pred_x128)

                for i in range(len(pred_x128)):
                    px64 = float(pred_x128[i]) * EVENT_SIZE / FRAME_SIZE
                    py64 = float(pred_y128[i]) * EVENT_SIZE / FRAME_SIZE
                    err = float(math.hypot(px64 - float(gt_x64[i]), py64 - float(gt_y64[i])))
                    key = (int(subject[i]), str(eye[i]), str(session_code[i]), int(frame_idx[i]))
                    motion_state = motion_labels.get(key, "")
                    if motion_state:
                        subject_errors[int(subject[i])].append(err)
                        motion_errors[motion_state].append(err)
                        subject_motion_errors[(int(subject[i]), motion_state)].append(err)
                    else:
                        rows_no_motion += 1
                    writer.writerow(
                        {
                            "sample_idx": int(sample_idx[i]),
                            "source_global_idx": int(source_global_idx[i]),
                            "split": "test",
                            "subject": int(subject[i]),
                            "eye": str(eye[i]),
                            "session_name": str(session_name[i]),
                            "session_code": str(session_code[i]),
                            "frame_idx": int(frame_idx[i]),
                            "timestamp": int(timestamp[i]),
                            "gt_x_input64": float(gt_x64[i]),
                            "gt_y_input64": float(gt_y64[i]),
                            "pred_x_input64": px64,
                            "pred_y_input64": py64,
                            "error_input64_px": err,
                            "gt_x_128": float(gt_x128[i]),
                            "gt_y_128": float(gt_y128[i]),
                            "pred_x_128": float(pred_x128[i]),
                            "pred_y_128": float(pred_y128[i]),
                            "gt_a_128": float(gt_a128[i]),
                            "gt_b_128": float(gt_b128[i]),
                            "gt_ang": float(gt_ang[i]),
                            "pred_a_128": float(pred_a128[i]),
                            "pred_b_128": float(pred_b128[i]),
                            "pred_ang": float(pred_ang[i]),
                            "mask_iou_128": float(mask_ious[i]),
                            "motion_state": motion_state,
                        }
                    )
                    rows_written += 1

    subject_path = args.output_dir / f"{model_tag}_subject37_48_subject_error_stats.csv"
    with subject_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "n", "mean", "median", "p95", "p99"])
        writer.writeheader()
        for subject, values in sorted(subject_errors.items()):
            writer.writerow({"subject": subject, **describe(values)})

    motion_path = args.output_dir / f"{model_tag}_subject37_48_motion_error_stats.csv"
    with motion_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["motion_state", "n", "mean", "median", "p95", "p99"])
        writer.writeheader()
        for motion, values in sorted(motion_errors.items()):
            writer.writerow({"motion_state": motion, **describe(values)})

    subject_motion_path = args.output_dir / f"{model_tag}_subject37_48_error_distribution_by_subject_motion.csv"
    with subject_motion_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "motion_state", "n", "mean", "median", "p95", "p99"])
        writer.writeheader()
        for (subject, motion), values in sorted(subject_motion_errors.items()):
            writer.writerow({"subject": subject, "motion_state": motion, **describe(values)})

    metadata = {
        "model": args.model,
        "model_tag": model_tag,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt_epoch,
        "checkpoint_val_metric": ckpt_val_metric,
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "source_root": str(source_root),
        "split": "test",
        "test_samples": len(dataset),
        "rows_written": rows_written,
        "rows_without_motion_label": rows_no_motion,
        "device": args.device,
        "torch": torch.__version__,
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "prediction_file": str(pred_path),
        "subject_stats": str(subject_path),
        "motion_stats": str(motion_path),
        "subject_motion_stats": str(subject_motion_path),
    }
    (args.output_dir / f"{model_tag}_test_inference_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    report = args.output_dir / f"{model_tag}_subject37_48_test_inference_report.md"
    report.write_text(
        "\n".join(
            [
                f"# {model_tag} Subject 37-48 Test Inference",
                "",
                f"- Model: `{args.model}`",
                f"- Checkpoint: `{args.checkpoint}`",
                f"- Checkpoint epoch: `{ckpt_epoch}`",
                f"- Checkpoint val metric: `{ckpt_val_metric}`",
                f"- Test samples: `{len(dataset)}`",
                f"- Rows written: `{rows_written}`",
                f"- Rows without non-Blink motion label: `{rows_no_motion}`",
                f"- Device: `{args.device}`",
                f"- Prediction file: `{pred_path.name}`",
                f"- Subject stats: `{subject_path.name}`",
                f"- Motion stats: `{motion_path.name}`",
                f"- Subject-motion stats: `{subject_motion_path.name}`",
            ]
        )
        + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
