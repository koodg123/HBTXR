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

from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
from EvEye.model.DavisEyeEllipse.HBTXR.HBTXR import HBTXR
from EvEye.model.DavisEyeEllipse.HBTXR.Predict import post_process


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet_img64_patch4.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("references/codebase/software/FACET/runs/logs/HBTXR_full_unet_img64_patch4/version_0/checkpoints/epoch=67-val_mean_distance=0.4492.ckpt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("references/report/FACET/HBTXR_val_motion_eval"),
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--run-name", default="HBTXR_full_unet_img64_patch4")
    parser.add_argument("--dataset-label", default="")
    parser.add_argument("--report-name", default="")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def output_prefix(run_name: str, split: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")
    return f"{safe}_{split}"


def natural_user_key(path: str) -> tuple[int, str]:
    match = re.search(r"/user(\d+)/", path)
    return (int(match.group(1)) if match else 10**9, path)


def parse_frame_name(path: Path) -> tuple[int, int]:
    stem = path.stem
    idx, ts = stem.split("_")[-2:]
    return int(idx), int(ts)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ellipse_iou_input64(pred_row: np.ndarray, gt_row: np.ndarray) -> float:
    # Rows are heatmap-scale [x, y, a, b, angle]. Scale to input64 first.
    pred = pred_row.astype(float).copy()
    gt = gt_row.astype(float).copy()
    pred[:4] *= 4.0
    gt[:4] *= 4.0
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
    inter = np.logical_and(canvas_p, canvas_g).sum()
    return float(inter / union)


def discover_frame_map(session_dir: Path) -> dict[int, int]:
    frames_dir = session_dir / "frames"
    out = {}
    for frame_path in frames_dir.glob("*.png"):
        try:
            frame_idx, ts = parse_frame_name(frame_path)
        except Exception:
            continue
        out[ts] = frame_idx
    return out


def load_split_ellipses(dataset_root: Path, split: str) -> np.ndarray:
    ellipse_path = dataset_root / split / "cached_ellipse" / "ellipses_batch_0.memmap"
    info_path = dataset_root / split / "cached_ellipse" / "ellipses_batch_info_0.txt"
    with info_path.open("r", encoding="utf-8") as f:
        shape_line = f.readline().strip()
        dtype_line = f.readline().strip()
    shape = tuple(int(x) for x in shape_line.split(": ")[1].strip("()").split(",") if x.strip())
    dtype = eval(dtype_line.split(": ")[1], {"np": np, "numpy": np})
    return np.memmap(ellipse_path, dtype=dtype, mode="r", shape=shape)


def build_metadata(dataset_root: Path, output_dir: Path, split: str, prefix: str) -> pd.DataFrame:
    out_path = output_dir / f"{prefix}_sample_metadata.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    progress_path = dataset_root / "progress_state.json"
    with progress_path.open("r", encoding="utf-8") as f:
        progress = json.load(f)
    ellipses = load_split_ellipses(dataset_root, split)

    rows = []
    cursor = 0
    for summary in progress["session_summaries"]:
        if summary["split"] != split:
            continue
        session_dir = Path(summary["session"])
        session_name = session_dir.name
        code = SESSION_TO_CODE.get(session_name, "unknown")
        regime = CODE_TO_REGIME.get(code, "unknown")
        user_match = re.search(r"user(\d+)", session_dir.parts[-3])
        user = int(user_match.group(1)) if user_match else -1
        eye = session_dir.parts[-2]
        valid = int(summary["valid"])
        frame_by_ts = discover_frame_map(session_dir)

        session_slice = ellipses[cursor : cursor + valid]
        for local_idx, e in enumerate(session_slice):
            ts = int(e["t"])
            rows.append(
                {
                    "sample_idx": cursor + local_idx,
                    "session_local_idx": local_idx,
                    "split": split,
                    "user": user,
                    "eye": eye,
                    "session_name": session_name,
                    "session_code": code,
                    "session_regime": regime,
                    "timestamp": ts,
                    "frame_idx": frame_by_ts.get(ts, -1),
                    "gt_x_orig": float(e["x"]),
                    "gt_y_orig": float(e["y"]),
                    "gt_a_orig": float(e["a"]),
                    "gt_b_orig": float(e["b"]),
                    "gt_ang": float(e["ang"]),
                }
            )
        cursor += valid

    if cursor != len(ellipses):
        raise RuntimeError(f"Metadata cursor {cursor} != {split} ellipse count {len(ellipses)}")
    df = pd.DataFrame(rows)
    df = assign_velocity_states(df)
    df.to_csv(out_path, index=False)
    return df


def assign_velocity_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["speed_pxps"] = np.nan
    group_cols = ["user", "eye", "session_name"]
    for _, idx in out.groupby(group_cols, sort=False).groups.items():
        g = out.loc[idx].sort_values("session_local_idx")
        x = g["gt_x_orig"].to_numpy(float)
        y = g["gt_y_orig"].to_numpy(float)
        t = g["timestamp"].to_numpy(float)
        speed = np.full(len(g), np.nan, dtype=float)
        for shift in (2, 1):
            prev = np.arange(len(g)) - shift
            nxt = np.arange(len(g)) + shift
            valid = (prev >= 0) & (nxt < len(g)) & np.isnan(speed)
            dt = (t[nxt[valid]] - t[prev[valid]]) / 1e6
            dist = np.hypot(x[nxt[valid]] - x[prev[valid]], y[nxt[valid]] - y[prev[valid]])
            ok = dt > 0
            values = np.full(valid.sum(), np.nan, dtype=float)
            values[ok] = dist[ok] / dt[ok]
            speed[np.where(valid)[0]] = values
        out.loc[g.index, "speed_pxps"] = speed

    out["motion_state"] = np.where(
        out["speed_pxps"] > SACCADE_SPEED_THRESHOLD_PXPS,
        "Saccade",
        np.where(out["session_regime"] == "smooth", "Smooth", "Fixation"),
    )
    return out


def load_model(config: dict, checkpoint: Path, device: torch.device) -> HBTXR:
    model_cfg = dict(config["model"])
    model_cfg.pop("type", None)
    model = HBTXR(**model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_inference(config: dict, checkpoint: Path, output_dir: Path, device_name: str, batch_size: int, num_workers: int, max_samples: int, split: str, prefix: str) -> pd.DataFrame:
    out_path = output_dir / f"{prefix}_sample_predictions.csv"
    if out_path.exists():
        return pd.read_csv(out_path)

    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    torch.backends.cudnn.enabled = False

    dataset_cfg = dict(config["dataloader"]["val"]["dataset"])
    dataset_cfg.pop("type", None)
    dataset_cfg["split"] = split
    dataset = DavisEyeEllipseDataset(**dataset_cfg)
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


def describe(values: pd.Series) -> dict:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(v) == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan}
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "p99": float(np.percentile(v, 99)),
        "max": float(np.max(v)),
    }


def make_group_stats(df: pd.DataFrame, group_cols: list[str], value_col: str, prefix: str) -> pd.DataFrame:
    rows = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        stats = describe(group[value_col])
        row.update({f"{prefix}_{k}": v for k, v in stats.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def build_manual_gt_lookup(raw_root: Path, users: set[int]) -> dict[tuple[int, str, str, int], dict]:
    lookup = {}
    for user in users:
        for eye in ("left", "right"):
            for session_name, code in SESSION_TO_CODE.items():
                session_dir = raw_root / f"user{user}" / eye / session_name
                csv_path = session_dir / f"user_{user}.csv"
                if not csv_path.exists():
                    continue
                data = pd.read_csv(csv_path)
                for _, r in data.iterrows():
                    if int(r.get("region_count", 0)) <= 0:
                        continue
                    try:
                        shape = json.loads(r["region_shape_attributes"])
                    except Exception:
                        continue
                    if shape.get("name") != "ellipse":
                        continue
                    try:
                        frame_idx, _ = parse_frame_name(Path(str(r["filename"])))
                    except Exception:
                        continue
                    theta = float(shape.get("theta", 0.0))
                    angle_deg = math.degrees(theta) if abs(theta) <= (2 * math.pi + 0.1) else theta
                    lookup[(user, eye, code, frame_idx)] = {
                        "manual_cx": float(shape["cx"]),
                        "manual_cy": float(shape["cy"]),
                        "manual_rx": float(shape.get("rx", np.nan)),
                        "manual_ry": float(shape.get("ry", np.nan)),
                        "manual_angle_deg": angle_deg,
                    }
    return lookup


def compute_label_noise(metadata: pd.DataFrame, output_dir: Path, dataset_root: Path, prefix: str) -> pd.DataFrame:
    out_path = output_dir / f"{prefix}_pseudolabel_noise.csv"
    if out_path.exists():
        if out_path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(out_path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    raw_root = dataset_root.parent / "Data_davis"
    lookup = build_manual_gt_lookup(raw_root, set(metadata["user"].unique()))
    rows = []
    for _, r in metadata.iterrows():
        key = (int(r.user), str(r.eye), str(r.session_code), int(r.frame_idx))
        m = lookup.get(key)
        if m is None:
            continue
        center_noise = math.hypot(float(r.gt_x_orig) - m["manual_cx"], float(r.gt_y_orig) - m["manual_cy"])
        rows.append(
            {
                "sample_idx": int(r.sample_idx),
                "user": int(r.user),
                "eye": r.eye,
                "session_code": r.session_code,
                "motion_state": r.motion_state,
                "frame_idx": int(r.frame_idx),
                "pseudo_cx": float(r.gt_x_orig),
                "pseudo_cy": float(r.gt_y_orig),
                "manual_cx": m["manual_cx"],
                "manual_cy": m["manual_cy"],
                "center_noise_px": center_noise,
                "pseudo_major_diam": float(r.gt_a_orig),
                "pseudo_minor_diam": float(r.gt_b_orig),
                "manual_major_diam": 2.0 * max(m["manual_rx"], m["manual_ry"]),
                "manual_minor_diam": 2.0 * min(m["manual_rx"], m["manual_ry"]),
            }
        )
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
    plt.boxplot(data, labels=sorted(merged["subject"].unique()), showfliers=False)
    plt.xticks(rotation=90, fontsize=7)
    plt.ylabel("center error (input64 px)")
    plt.title(f"{run_name} {split} subject-wise pixel error distribution")
    plt.tight_layout()
    plt.savefig(figures / "fig_subject_pixel_error_box.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    data = [merged.loc[merged["subject"] == s, "iou_input64"].dropna().to_numpy() for s in sorted(merged["subject"].unique())]
    plt.boxplot(data, labels=sorted(merged["subject"].unique()), showfliers=False)
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
    plt.boxplot(data, labels=order, showfliers=False)
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


def fmt(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.3f}"


def write_report(output_dir: Path, config_path: Path, checkpoint: Path, tables: dict[str, Path], split: str, run_name: str, dataset_label: str, report_name: str) -> Path:
    subject = pd.read_csv(tables["subject_stats"])
    motion_counts = pd.read_csv(tables["motion_counts"])
    subject_pixel = pd.read_csv(tables["subject_pixel"])
    motion = pd.read_csv(tables["motion_pixel"])
    label_precision = pd.read_csv(tables["label_precision"])
    if tables["label_noise"].exists() and tables["label_noise"].stat().st_size > 0:
        try:
            label_noise = pd.read_csv(tables["label_noise"])
        except pd.errors.EmptyDataError:
            label_noise = pd.DataFrame()
    else:
        label_noise = pd.DataFrame()
    merged = pd.read_csv(tables["merged"], usecols=["sample_idx", "motion_state", "error_input64_px", "iou_input64"])

    report_file = report_name or f"{output_prefix(run_name, split)}_motion_eval_2026-06-28.md"
    report_path = output_dir.parent / report_file
    top_subjects = subject_pixel.sort_values("err_median", ascending=False).head(5)
    motion_table = motion[["motion_state", "err_n", "err_mean", "err_median", "err_p95", "err_p99", "err_max"]]

    lines = []
    lines.append(f"# {run_name} {split.upper()} Motion Evaluation Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Model: `{run_name}`")
    lines.append(f"- Config: `{config_path}`")
    lines.append(f"- Checkpoint: `{checkpoint}`")
    split_label = dataset_label or Path(config_path).stem
    lines.append(f"- Evaluation split: `{split_label}/{split}`.")
    lines.append("- Motion state: velocity-based `Fixation`, `Saccade`, `Smooth`; repeated-run CI section is intentionally excluded.")
    lines.append("")
    lines.append("## Motion-State Rule")
    lines.append("")
    lines.append("- `Saccade`: pseudo-label pupil-center speed > 493 px/s.")
    lines.append("- `Fixation`: speed <= 493 px/s and session code is `101` or `201`.")
    lines.append("- `Smooth`: speed <= 493 px/s and session code is `102` or `202`.")
    lines.append("- Session mapping follows `subject-motion-analysis`: `101/201` are saccade-fixation regime, `102/202` are smooth-pursuit regime.")
    lines.append("- Velocity is computed from dense U-Net pseudo-label centers in `DeanDataset_full_unet`, not from Tobii or repeated manual annotations.")
    lines.append("")
    lines.append("## Overall Summary")
    lines.append("")
    all_row = motion[motion["motion_state"] == "All"].iloc[0]
    lines.append(f"- Total {split} samples: {len(merged):,}")
    lines.append(f"- Valid error samples: {int(all_row.err_n):,}")
    lines.append(f"- Overall center error: mean {fmt(all_row.err_mean)} px, median {fmt(all_row.err_median)} px, P95 {fmt(all_row.err_p95)} px, P99 {fmt(all_row.err_p99)} px in input64 coordinates.")
    lines.append(f"- Overall IoU: mean {fmt(merged.iou_input64.mean())}, median {fmt(merged.iou_input64.median())}.")
    lines.append("")
    lines.append("## (1) Subject-wise Pixel Error / IoU Distribution")
    lines.append("")
    lines.append(f"- Table: `{tables['subject_stats'].name}`")
    lines.append("- Figures: `fig_subject_pixel_error_box.png`, `fig_subject_iou_box.png`")
    lines.append(f"- Median subject error range: {fmt(subject.err_median.min())} to {fmt(subject.err_median.max())} px.")
    lines.append(f"- Median subject IoU range: {fmt(subject.iou_median.min())} to {fmt(subject.iou_median.max())}.")
    lines.append("- Interpretation: subject-wise spread is visible in the tail metrics; subjects with high P95/P99 should be inspected for pseudo-label quality, blink/occlusion, or motion imbalance.")
    lines.append("")
    lines.append("## (2) Subject-wise Motion Distribution")
    lines.append("")
    lines.append(f"- Table: `{tables['motion_counts'].name}`")
    lines.append("- Figure: `fig_subject_motion_counts.png`")
    total_motion = motion_counts[["Fixation", "Saccade", "Smooth"]].sum()
    for state in ["Fixation", "Saccade", "Smooth"]:
        lines.append(f"- {state}: {int(total_motion[state]):,} samples ({100*total_motion[state]/total_motion.sum():.2f}%).")
    lines.append("- Interpretation: the velocity rule gives a dense 3-state split for the HBTXR pseudo-label val set. Saccade is expected to be a minority class because high-speed movements are brief.")
    lines.append("- Caution: the Saccade group is very small in this val split, so Saccade error statistics should be treated as descriptive rather than conclusive.")
    lines.append("")
    lines.append("## (3) Subject-wise Mean / Median / P95 / P99 Pixel Error")
    lines.append("")
    lines.append(f"- Table: `{tables['subject_pixel'].name}`")
    lines.append("- Figure: `fig_subject_pixel_error_box.png`")
    lines.append("- Highest-median subjects:")
    for _, r in top_subjects.iterrows():
        lines.append(f"  - {r.subject}: median {fmt(r.err_median)} px, P95 {fmt(r.err_p95)} px, P99 {fmt(r.err_p99)} px, n={int(r.err_n):,}")
    lines.append("- Interpretation: median captures typical localization quality, while P95/P99 exposes rare tracking or label failures. Report both because mean alone hides tail behavior.")
    lines.append("")
    lines.append("## (4) Motion-wise Mean / Median / P95 / P99 Pixel Error")
    lines.append("")
    lines.append(f"- Table: `{tables['motion_pixel'].name}`")
    lines.append("- Figure: `fig_motion_pixel_error_box.png`")
    lines.append("")
    lines.append("| motion | n | mean | median | P95 | P99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, r in motion_table.iterrows():
        lines.append(
            f"| {r.motion_state} | {int(r.err_n):,} | {fmt(r.err_mean)} | {fmt(r.err_median)} | {fmt(r.err_p95)} | {fmt(r.err_p99)} | {fmt(r.err_max)} |"
        )
    lines.append("")
    lines.append("- Interpretation: compare `Saccade` against `Fixation` and `Smooth` primarily via median and P95/P99. If Saccade has a larger tail, this supports the reviewer-facing statement that fast motion is harder.")
    lines.append("")
    lines.append("## (7) Annotation Precision, Label Noise")
    lines.append("")
    lines.append(f"- Table: `{tables['label_precision'].name}`")
    lines.append(f"- Table: `{tables['label_noise'].name}`")
    lines.append("- Figure: `fig_label_uncertainty_overlay.png`")
    lp = label_precision.iloc[0]
    lines.append(f"- Manual GT integer quantization floor: mean {fmt(lp.floor_mean_px)} px, median {fmt(lp.floor_median_px)} px, P95 {fmt(lp.floor_p95_px)} px.")
    if len(label_noise):
        all_noise = label_noise[label_noise["motion_state"] == "All"].iloc[0]
        lines.append(f"- HBTXR {split} pseudo-label center noise on manual-GT matched frames: mean {fmt(all_noise.center_noise_mean)} px, median {fmt(all_noise.center_noise_median)} px, P95 {fmt(all_noise.center_noise_p95)} px, n={int(all_noise.center_noise_n):,}.")
    else:
        lines.append("- No manual-GT matched pseudo-label samples were found in the val split.")
    lines.append("- Interpretation: HBTXR was trained/evaluated against U-Net pseudo-labels in `DeanDataset_full_unet`. Therefore, label uncertainty has two parts: manual annotation quantization floor and pseudo-label generation noise. If reported HBTXR errors are near these values, sub-pixel differences should be interpreted cautiously.")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    for key, path in tables.items():
        lines.append(f"- {key}: `{path}`")
    lines.append(f"- figures: `{output_dir / 'figures'}`")
    lines.append("")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


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
