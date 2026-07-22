#!/usr/bin/env python3
"""Evaluate HBTXR split predictions with subject and motion breakdowns.

This script is intentionally self-contained because the original Lightning
validation loop only stores epoch-level averages. It reconstructs split metadata
from DeanDataset_full_unet timestamps, runs per-sample inference, and writes the
tables/figures needed for reviewer-facing analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.dataset_factory import make_dataset
from models.detectors.hbtxr import HBTXR
from models.postprocess.hbtxr_predict import post_process


SESSION_TO_CODE = {
    "session_1_0_1": "101",
    "session_1_0_2": "102",
    "session_2_0_1": "201",
    "session_2_0_2": "202",
}
CODE_TO_REGIME = {
    "101": "saccade_fixation",
    "102": "smooth",
    "201": "saccade_fixation",
    "202": "smooth",
}
SACCADE_SPEED_THRESHOLD_PXPS = 493.0
























from engine.tools.evaluation._motion_common import (
    parse_args,
    output_prefix,
    natural_user_key,
    parse_frame_name,
    load_yaml,
    ellipse_iou_input64,
    discover_frame_map,
    load_split_ellipses,
    build_metadata,
    assign_velocity_states,
    load_model,
    describe,
    make_group_stats,
    build_manual_gt_lookup,
    compute_label_noise,
    fmt,
    write_report,
)

def run_inference(config: dict, checkpoint: Path, output_dir: Path, device_name: str, batch_size: int, num_workers: int, max_samples: int, split: str, prefix: str) -> pd.DataFrame:
    out_path = output_dir / f"{prefix}_sample_predictions.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    torch.backends.cudnn.enabled = False

    dataset_cfg = dict(config["dataloader"]["val"]["dataset"])
    dataset_cfg["split"] = split
    dataset = make_dataset(dataset_cfg)
    if max_samples > 0:
        indices = list(range(min(max_samples, len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = load_model(config, checkpoint, device)
    rows = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"HBTXR {split} inference"):
            inputs = batch["input"].to(device, non_blocking=True).float()
            close = batch["close"].cpu().numpy()
            center = batch["center"].cpu().numpy()
            gt_ellipse = batch["ellipse"].cpu().numpy()
            pred = model(inputs)
            dets = post_process(pred)

            pred_ellipse = dets["ellipse"].detach().cpu().numpy()
            scores = dets["scores"].detach().cpu().numpy()
            pred_center = np.stack(
                [
                    dets["xs"].detach().cpu().numpy().reshape(-1),
                    dets["ys"].detach().cpu().numpy().reshape(-1),
                ],
                axis=1,
            )
            bs = center.shape[0]
            for i in range(bs):
                sample_idx = offset + i
                valid = int(close[i]) == 0
                err_heat = float(np.linalg.norm(pred_center[i] - center[i])) if valid else np.nan
                gt_h = gt_ellipse[i]
                pred_h = pred_ellipse[i]
                iou64 = ellipse_iou_input64(pred_h, gt_h) if valid else np.nan
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "valid": int(valid),
                        "gt_x_heatmap": float(center[i, 0]),
                        "gt_y_heatmap": float(center[i, 1]),
                        "pred_x_heatmap": float(pred_center[i, 0]),
                        "pred_y_heatmap": float(pred_center[i, 1]),
                        "error_heatmap_px": err_heat,
                        "error_input64_px": err_heat * 4.0 if valid else np.nan,
                        "pred_score": float(np.asarray(scores[i]).reshape(-1)[0]),
                        "gt_a_heatmap": float(gt_h[2]),
                        "gt_b_heatmap": float(gt_h[3]),
                        "gt_ang": float(gt_h[4]),
                        "pred_a_heatmap": float(pred_h[2]),
                        "pred_b_heatmap": float(pred_h[3]),
                        "pred_ang": float(pred_h[4]),
                        "iou_input64": iou64,
                    }
                )
            offset += bs

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df










def write_tables_and_figures(metadata: pd.DataFrame, pred: pd.DataFrame, label_noise: pd.DataFrame, output_dir: Path, split: str, prefix: str, run_name: str) -> dict[str, Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    merged = metadata.merge(pred, on="sample_idx", how="inner")
    merged["subject"] = merged["user"].astype(str).map(lambda x: f"user{int(x):02d}")
    merged.to_csv(output_dir / f"{prefix}_predictions_with_metadata.csv", index=False)

    subject_err = make_group_stats(merged, ["subject"], "error_input64_px", "err")
    subject_iou = make_group_stats(merged, ["subject"], "iou_input64", "iou")
    subject_stats = subject_err.merge(subject_iou, on="subject", how="left")
    subject_stats.to_csv(output_dir / f"{prefix}_subject_error_iou_stats.csv", index=False)

    motion_counts = (
        merged.groupby(["subject", "motion_state"]).size().unstack(fill_value=0).reset_index()
    )
    for col in ["Fixation", "Saccade", "Smooth"]:
        if col not in motion_counts:
            motion_counts[col] = 0
    motion_counts["total"] = motion_counts[["Fixation", "Saccade", "Smooth"]].sum(axis=1)
    for col in ["Fixation", "Saccade", "Smooth"]:
        motion_counts[f"pct_{col.lower()}"] = 100.0 * motion_counts[col] / motion_counts["total"].replace(0, np.nan)
    motion_counts.to_csv(output_dir / f"{prefix}_subject_motion_counts.csv", index=False)

    subject_pixel = make_group_stats(merged, ["subject"], "error_input64_px", "err")
    subject_pixel.to_csv(output_dir / f"{prefix}_subject_pixel_error_stats.csv", index=False)

    motion_pixel = make_group_stats(merged, ["motion_state"], "error_input64_px", "err")
    all_row = {"motion_state": "All"}
    all_row.update({f"err_{k}": v for k, v in describe(merged["error_input64_px"]).items()})
    motion_pixel = pd.concat([motion_pixel, pd.DataFrame([all_row])], ignore_index=True)
    motion_pixel.to_csv(output_dir / f"{prefix}_motion_pixel_error_stats.csv", index=False)

    label_precision = pd.DataFrame(
        [
            {
                "source": "manual_gt_integer_quantization_floor",
                "n_annotations_reference": 9011,
                "integer_quantized_pct": 100.0,
                "per_axis_label_std_px": round(1 / math.sqrt(12), 3),
                "floor_mean_px": 0.383,
                "floor_median_px": 0.399,
                "floor_p95_px": 0.599,
            }
        ]
    )
    label_precision.to_csv(output_dir / f"{prefix}_label_precision_floor.csv", index=False)

    if len(label_noise):
        label_noise_stats = make_group_stats(label_noise, ["motion_state"], "center_noise_px", "center_noise")
        all_noise = {"motion_state": "All"}
        all_noise.update({f"center_noise_{k}": v for k, v in describe(label_noise["center_noise_px"]).items()})
        label_noise_stats = pd.concat([label_noise_stats, pd.DataFrame([all_noise])], ignore_index=True)
    else:
        label_noise_stats = pd.DataFrame()
    label_noise_stats.to_csv(output_dir / f"{prefix}_pseudolabel_noise_stats.csv", index=False)

    # Figures
    plt.figure(figsize=(11, 5))
    data = [merged.loc[merged["subject"] == s, "error_input64_px"].dropna().to_numpy() for s in sorted(merged["subject"].unique())]
    plt.boxplot(data, tick_labels=sorted(merged["subject"].unique()), showfliers=False)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("center error (input64 px)")
    plt.title(f"{run_name} {split} subject-wise pixel error distribution")
    plt.tight_layout()
    plt.savefig(figures / "fig_subject_pixel_error_box.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    data = [merged.loc[merged["subject"] == s, "iou_input64"].dropna().to_numpy() for s in sorted(merged["subject"].unique())]
    plt.boxplot(data, tick_labels=sorted(merged["subject"].unique()), showfliers=False)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("ellipse IoU (input64 raster)")
    plt.title(f"{run_name} {split} subject-wise IoU distribution")
    plt.tight_layout()
    plt.savefig(figures / "fig_subject_iou_box.png", dpi=160)
    plt.close()

    plot_counts = motion_counts.set_index("subject")[["Fixation", "Saccade", "Smooth"]]
    plot_counts.plot(kind="bar", stacked=True, figsize=(11, 5), color=["#4c78a8", "#e45756", "#72b7b2"])
    plt.ylabel("samples")
    plt.title(f"{run_name} {split} subject-wise motion-state distribution")
    plt.tight_layout()
    plt.savefig(figures / "fig_subject_motion_counts.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    order = ["Fixation", "Saccade", "Smooth"]
    data = [merged.loc[merged["motion_state"] == m, "error_input64_px"].dropna().to_numpy() for m in order]
    plt.boxplot(data, tick_labels=order, showfliers=False)
    plt.ylabel("center error (input64 px)")
    plt.title(f"{run_name} {split} motion-wise pixel error distribution")
    plt.tight_layout()
    plt.savefig(figures / "fig_motion_pixel_error_box.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(merged["error_input64_px"].dropna(), bins=120, range=(0, 20), alpha=0.6, label="HBTXR error")
    if len(label_noise):
        plt.hist(label_noise["center_noise_px"].dropna(), bins=120, range=(0, 20), alpha=0.5, label="pseudo-label center noise")
    plt.axvline(0.383, color="k", linestyle="--", linewidth=1, label="quantization floor mean")
    plt.xlabel("center distance (px)")
    plt.ylabel("count")
    plt.title("HBTXR error vs label uncertainty")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "fig_label_uncertainty_overlay.png", dpi=160)
    plt.close()

    return {
        "merged": output_dir / f"{prefix}_predictions_with_metadata.csv",
        "subject_stats": output_dir / f"{prefix}_subject_error_iou_stats.csv",
        "motion_counts": output_dir / f"{prefix}_subject_motion_counts.csv",
        "subject_pixel": output_dir / f"{prefix}_subject_pixel_error_stats.csv",
        "motion_pixel": output_dir / f"{prefix}_motion_pixel_error_stats.csv",
        "label_precision": output_dir / f"{prefix}_label_precision_floor.csv",
        "label_noise": output_dir / f"{prefix}_pseudolabel_noise_stats.csv",
    }






def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)

    config = load_yaml(args.config)
    prefix = output_prefix(args.run_name, args.split)
    dataset_root = Path(config["dataloader"]["val"]["dataset"]["root_path"])
    metadata = build_metadata(dataset_root, args.output_dir, args.split, prefix)
    if args.max_samples > 0:
        metadata = metadata.iloc[: args.max_samples].copy()

    if args.skip_inference:
        pred = pd.read_csv(args.output_dir / f"{prefix}_sample_predictions.csv")
    else:
        pred = run_inference(
            config=config,
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device_name=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            split=args.split,
            prefix=prefix,
        )
    label_noise = compute_label_noise(metadata, args.output_dir, dataset_root, prefix)
    tables = write_tables_and_figures(metadata, pred, label_noise, args.output_dir, args.split, prefix, args.run_name)
    report = write_report(args.output_dir, args.config, args.checkpoint, tables, args.split, args.run_name, args.dataset_label, args.report_name)
    print(f"wrote report: {report}")


if __name__ == "__main__":
    main()
