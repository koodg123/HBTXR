#!/usr/bin/env python3
"""Build subject-independent test error tables for TDTracker."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = ROOT / "references/codebase/software/FACET"
TD_ROOT = ROOT / "references/codebase/software/ais2025/tdtracker"
HBTXR_EVAL_SCRIPT = FACET_ROOT / "EvEye/utils/scripts/evaluate_hbtxr_val_motion.py"

if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))
if str(TD_ROOT) not in sys.path:
    sys.path.insert(0, str(TD_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="TDTracker")
    parser.add_argument(
        "--h5",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/target_data/tdtracker_hbtxr_img64_seq100/test_hbtxr_img64_seq100.h5"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=TD_ROOT / "checkpoint/HBTXR_subject_independent_img64/last_checkpoint.pth",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=ROOT / "analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv",
    )
    parser.add_argument(
        "--workbook-template",
        type=Path,
        default=ROOT / "analysis/RESULTS/JETCAS_REPLY_TABLES (Error-Distributions).xlsx",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/RESULTS/TDTracker")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


class H5SequenceDataset(Dataset):
    def __init__(self, h5_path: Path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.length = int(f["frames"].shape[0])
            self.seq_len = int(f["frames"].shape[1])
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        f = self._open()
        return (
            torch.from_numpy(f["frames"][index].astype(np.float32)),
            torch.from_numpy(f["label"][index].astype(np.float32)),
        )


def load_hbtxr_eval_module():
    spec = importlib.util.spec_from_file_location("hbtxr_eval_for_tdtracker", HBTXR_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HBTXR_EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hbtxr_eval_for_tdtracker"] = module
    spec.loader.exec_module(module)
    return module


def load_sequence_package_module():
    script = ROOT / "analysis/scripts/fill_sequence_center_error_distribution.py"
    spec = importlib.util.spec_from_file_location("sequence_center_package_tdtracker", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sequence_center_package_tdtracker"] = module
    spec.loader.exec_module(module)
    return module


def build_segments(dataset_root: Path, split: str, seq_len: int) -> list[tuple[int, int]]:
    import json

    progress_path = dataset_root / "progress_state.json"
    with progress_path.open("r", encoding="utf-8") as f:
        progress = json.load(f)
    segments = []
    offset = 0
    for session in progress["session_summaries"]:
        if session["split"] != split:
            continue
        valid = int(session["valid"])
        for local_start in range(0, valid - seq_len + 1, seq_len):
            start = offset + local_start
            segments.append((start, start + seq_len))
        offset += valid
    return segments


def ellipse_iou_center_proxy(pred_x64: float, pred_y64: float, gt_row_heatmap64: np.ndarray) -> float:
    gt = gt_row_heatmap64.astype(float).copy()
    if not np.isfinite(gt).all() or gt[2] <= 0 or gt[3] <= 0:
        return float("nan")
    gt[:4] *= 4.0
    pred = gt.copy()
    pred[0] = pred_x64
    pred[1] = pred_y64
    canvas_p = np.zeros((64, 64), dtype=np.uint8)
    canvas_g = np.zeros((64, 64), dtype=np.uint8)
    for canvas, row in ((canvas_p, pred), (canvas_g, gt)):
        x, y, a, b, angle = row
        if not np.isfinite(row).all() or a <= 0 or b <= 0:
            continue
        center = (int(round(np.clip(x, 0, 63))), int(round(np.clip(y, 0, 63))))
        axes = (
            max(1, int(round(np.clip(a / 2.0, 1, 64)))),
            max(1, int(round(np.clip(b / 2.0, 1, 64)))),
        )
        cv2.ellipse(canvas, center, axes, float(angle), 0, 360, 1, -1)
    union = np.logical_or(canvas_p, canvas_g).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(canvas_p, canvas_g).sum() / union)


def decode_simdr(output_x: torch.Tensor, output_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    width = output_x.size(-1)
    height = output_y.size(-1)
    preds_x = output_x.argmax(dim=2)
    preds_y = output_y.argmax(dim=2)
    x_prob = torch.softmax(output_x, dim=2).max(dim=2).values
    y_prob = torch.softmax(output_y, dim=2).max(dim=2).values
    out = torch.stack([preds_x.float() / float(width), preds_y.float() / float(height)], dim=2)
    return out, x_prob + y_prob


def load_model(checkpoint: Path, device: torch.device):
    import argparse as _argparse
    from models.TDTracker import Model

    args = _argparse.Namespace(sensor_width=64, sensor_height=64, spatial_factor=1.0)
    model = Model(args)
    state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def run_or_load_predictions(args: argparse.Namespace, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_mod = load_hbtxr_eval_module()
    metadata = eval_mod.build_metadata(args.dataset_root, args.output_dir, "test", prefix)
    pred_path = args.output_dir / f"{prefix}_sample_predictions.csv"
    if args.skip_inference and pred_path.exists():
        return metadata, pd.read_csv(pred_path)
    if pred_path.exists() and not args.skip_inference:
        return metadata, pd.read_csv(pred_path)

    dataset = H5SequenceDataset(args.h5)
    segments = build_segments(args.dataset_root, "test", dataset.seq_len)
    if len(segments) != len(dataset):
        raise RuntimeError(f"H5 sequences ({len(dataset)}) != reconstructed segments ({len(segments)})")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.enabled = False
    model = load_model(args.checkpoint, device)
    gt_ellipse_heat = metadata.set_index("sample_idx")[["gt_x_orig", "gt_y_orig", "gt_a_orig", "gt_b_orig", "gt_ang"]].copy()
    gt_ellipse_heat["gt_x_orig"] = gt_ellipse_heat["gt_x_orig"] * (64.0 / 346.0) / 4.0
    gt_ellipse_heat["gt_y_orig"] = gt_ellipse_heat["gt_y_orig"] * (64.0 / 260.0) / 4.0
    gt_ellipse_heat["gt_a_orig"] = gt_ellipse_heat["gt_a_orig"] * (64.0 / 346.0) / 4.0
    gt_ellipse_heat["gt_b_orig"] = gt_ellipse_heat["gt_b_orig"] * (64.0 / 260.0) / 4.0

    rows: list[dict] = []
    with h5py.File(args.h5, "r") as h5f, torch.no_grad():
        frames_ds = h5f["frames"]
        labels_ds = h5f["label"]
        total_batches = int(np.ceil(len(dataset) / float(args.batch_size)))
        for batch_start in tqdm(range(0, len(dataset), args.batch_size), total=total_batches, desc=f"{args.model_name} test inference"):
            batch_end = min(batch_start + args.batch_size, len(dataset))
            frames_np = frames_ds[batch_start:batch_end].astype(np.float32)
            labels_np = labels_ds[batch_start:batch_end].astype(np.float32)
            frames = torch.from_numpy(frames_np).to(device, non_blocking=True).float()
            pred_x, pred_y = model(frames)
            pred_norm, prob = decode_simdr(pred_x.detach().cpu(), pred_y.detach().cpu())
            pred_np = pred_norm.numpy()
            prob_np = prob.numpy()
            for b in range(pred_np.shape[0]):
                start, end = segments[batch_start + b]
                for t, sample_idx in enumerate(range(start, end)):
                    gt_x64 = float(labels_np[b, t, 0] * 64.0)
                    gt_y64 = float(labels_np[b, t, 1] * 64.0)
                    pred_x64 = float(pred_np[b, t, 0] * 64.0)
                    pred_y64 = float(pred_np[b, t, 1] * 64.0)
                    valid = int(gt_x64 > 0 and gt_y64 > 0)
                    err = float(np.hypot(pred_x64 - gt_x64, pred_y64 - gt_y64)) if valid else np.nan
                    iou = np.nan
                    if valid and sample_idx in gt_ellipse_heat.index:
                        iou = ellipse_iou_center_proxy(pred_x64, pred_y64, gt_ellipse_heat.loc[sample_idx].to_numpy(float))
                    rows.append(
                        {
                            "sample_idx": int(sample_idx),
                            "segment_start": int(start),
                            "segment_frame_idx": int(t),
                            "valid": valid,
                            "gt_x_input64": gt_x64,
                            "gt_y_input64": gt_y64,
                            "pred_x_input64": pred_x64,
                            "pred_y_input64": pred_y64,
                            "error_input64_px": err,
                            "pred_score": float(prob_np[b, t]),
                            "iou_input64": iou,
                            "iou_note": "center_proxy_gt_axes_angle",
                        }
                    )
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_path, index=False)
    return metadata, pred_df


def main() -> None:
    args = parse_args()
    if not hasattr(args, "model"):
        args.model = args.model_name
    sequence_mod = load_sequence_package_module()
    prefix = f"{args.model_name}_subject_independent_img64_test"
    metadata, pred = run_or_load_predictions(args, prefix)
    pred_meta = metadata.merge(pred, on="sample_idx", how="inner", validate="one_to_one")
    pred_meta.to_csv(args.output_dir / f"{args.model_name}_subject37_48_test_predictions_with_metadata.csv", index=False)
    joined = sequence_mod.join_motion_labels(pred_meta, args.labels, args.output_dir, args.model_name)
    stats_df = sequence_mod.aggregate(joined, args.output_dir, args.model_name)
    workbook = sequence_mod.update_workbook(args.workbook_template, stats_df, args.output_dir, args.model_name)
    report = sequence_mod.write_report(args, args.h5, stats_df, joined, workbook)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.checkpoint, ckpt_dir / args.checkpoint.name)
    print(f"model={args.model_name}")
    print(f"joined_rows={len(joined)}")
    print(f"stats_rows={len(stats_df)}")
    print(f"workbook={workbook}")
    print(f"report={report}")
    print(f"checkpoint_copy={ckpt_dir / args.checkpoint.name}")


if __name__ == "__main__":
    main()
