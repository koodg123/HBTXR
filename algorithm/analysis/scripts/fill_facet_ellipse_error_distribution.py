#!/usr/bin/env python3
"""Build subject-independent test error tables for FACET ellipse models.

This is the generic version of the original HBTXR table filler. It supports
FACET models that output CenterNet-style ellipse heads, currently HBTXR and
EPNet, and writes the same package layout used by TennSt/TENNs-Eye packages.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
FACET_ROOT = ROOT / "references/codebase/software/FACET"
HBTXR_EVAL_SCRIPT = FACET_ROOT / "EvEye/utils/scripts/evaluate_hbtxr_val_motion.py"

if str(FACET_ROOT) not in sys.path:
    sys.path.insert(0, str(FACET_ROOT))

from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
from EvEye.model.model_factory import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root-override",
        type=Path,
        default=None,
        help="Use this dataset root for test inference without modifying the YAML.",
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-inference", action="store_true")
    return parser.parse_args()


def load_hbtxr_eval_module():
    spec = importlib.util.spec_from_file_location("hbtxr_eval_for_facet_ellipse", HBTXR_EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {HBTXR_EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hbtxr_eval_for_facet_ellipse"] = module
    spec.loader.exec_module(module)
    return module


def load_sequence_package_module():
    script = ROOT / "analysis/scripts/fill_sequence_center_error_distribution.py"
    spec = importlib.util.spec_from_file_location("sequence_center_package", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sequence_center_package"] = module
    spec.loader.exec_module(module)
    return module


def load_post_process(model_type: str):
    if model_type == "EPNet":
        from EvEye.model.DavisEyeEllipse.EPNet.Predict import post_process
    elif model_type == "HBTXR":
        from EvEye.model.DavisEyeEllipse.HBTXR.Predict import post_process
    elif model_type == "ElNet":
        from EvEye.model.DavisEyeEllipse.EPNet.Predict import post_process
    else:
        raise ValueError(f"Unsupported FACET ellipse model type: {model_type}")
    return post_process


def ellipse_iou_input64(pred_row: np.ndarray, gt_row: np.ndarray, heatmap_to_input64: float) -> float:
    pred = pred_row.astype(float).copy()
    gt = gt_row.astype(float).copy()
    pred[:4] *= heatmap_to_input64
    gt[:4] *= heatmap_to_input64
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


def load_model(config: dict, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = copy.deepcopy(config["model"])
    model = make_model(model_cfg)
    ckpt = torch.load(checkpoint, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_dataset_cfg(config: dict, split: str, dataset_root_override: Path | None) -> dict:
    dataset_cfg = copy.deepcopy(config["dataloader"]["val"]["dataset"])
    dataset_cfg.pop("type", None)
    dataset_cfg["split"] = split
    if dataset_root_override is not None:
        dataset_cfg["root_path"] = str(dataset_root_override)
    return dataset_cfg


def run_or_load_predictions(args: argparse.Namespace, config: dict, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    eval_mod = load_hbtxr_eval_module()
    dataset_cfg = build_dataset_cfg(config, "test", args.dataset_root_override)
    dataset_root = Path(dataset_cfg["root_path"])
    metadata = eval_mod.build_metadata(dataset_root, args.output_dir, "test", prefix)
    pred_path = args.output_dir / f"{prefix}_sample_predictions.csv"
    if args.skip_inference and pred_path.exists():
        return metadata, pd.read_csv(pred_path)
    if pred_path.exists() and not args.skip_inference:
        return metadata, pd.read_csv(pred_path)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    torch.backends.cudnn.enabled = False

    dataset = DavisEyeEllipseDataset(**dataset_cfg)
    if args.max_samples > 0:
        dataset = torch.utils.data.Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    model_type = config["model"]["type"]
    post_process = load_post_process(model_type)
    model = load_model(config, args.checkpoint, device)
    rows: list[dict] = []
    offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{args.model_name} test inference"):
            inputs = batch["input"].to(device, non_blocking=True).float()
            close = batch["close"].cpu().numpy()
            center = batch["center"].cpu().numpy()
            gt_ellipse = batch["ellipse"].cpu().numpy()
            pred = model(inputs)
            dets = post_process(pred)
            heatmap_size = int(pred["hm"].shape[-1])
            heatmap_to_input64 = 64.0 / float(heatmap_size)

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
                iou64 = ellipse_iou_input64(pred_h, gt_h, heatmap_to_input64) if valid else np.nan
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "valid": int(valid),
                        "gt_x_heatmap": float(center[i, 0]),
                        "gt_y_heatmap": float(center[i, 1]),
                        "pred_x_heatmap": float(pred_center[i, 0]),
                        "pred_y_heatmap": float(pred_center[i, 1]),
                        "error_heatmap_px": err_heat,
                        "error_input64_px": err_heat * heatmap_to_input64 if valid else np.nan,
                        "pred_score": float(np.asarray(scores[i]).reshape(-1)[0]),
                        "gt_a_heatmap": float(gt_h[2]),
                        "gt_b_heatmap": float(gt_h[3]),
                        "gt_ang": float(gt_h[4]),
                        "pred_a_heatmap": float(pred_h[2]),
                        "pred_b_heatmap": float(pred_h[3]),
                        "pred_ang": float(pred_h[4]),
                        "iou_input64": iou64,
                        "iou_note": "predicted_ellipse_input64",
                        "heatmap_size": heatmap_size,
                    }
                )
            offset += bs

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(pred_path, index=False)
    return metadata, pred_df


def main() -> None:
    args = parse_args()
    if not hasattr(args, "model"):
        args.model = args.model_name
    sequence_mod = load_sequence_package_module()
    eval_mod = load_hbtxr_eval_module()
    config = eval_mod.load_yaml(args.config)
    prefix = f"{args.model_name}_test"
    metadata, pred = run_or_load_predictions(args, config, prefix)
    pred_meta = metadata.merge(pred, on="sample_idx", how="inner", validate="one_to_one")
    pred_meta.to_csv(args.output_dir / f"{args.model_name}_subject37_48_test_predictions_with_metadata.csv", index=False)
    joined = sequence_mod.join_motion_labels(pred_meta, args.labels, args.output_dir, args.model_name)
    stats_df = sequence_mod.aggregate(joined, args.output_dir, args.model_name)
    workbook = sequence_mod.update_workbook(args.workbook_template, stats_df, args.output_dir, args.model_name)
    report = sequence_mod.write_report(args, args.config, stats_df, joined, workbook)
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
