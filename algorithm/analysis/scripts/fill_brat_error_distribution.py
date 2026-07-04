#!/usr/bin/env python3
"""Build subject-independent test error tables for BRAT outputs."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE_SCRIPT = ROOT / "analysis/scripts/fill_sequence_center_error_distribution.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="BRAT")
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/ckpt/submission_test.csv",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "analysis/RESULTS/HBTXR_full_unet_img128_patch4/HBTXR_full_unet_img128_patch4_test_sample_metadata.csv",
    )
    parser.add_argument(
        "--test-files",
        type=Path,
        default=ROOT / "references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/event_data_hbtxr_img64_fulltest/dataset/test_files.txt",
    )
    parser.add_argument(
        "--event-test-root",
        type=Path,
        default=ROOT / "references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/event_data_hbtxr_img64_fulltest/test",
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
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/logs_hbtxr_img64/BRAT_subject_independent_img64/perror_0.5878_tsf1.0.pth",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/configs/hbtxr_subject_independent_img64_fulltest.json",
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=ROOT / "references/report/FACET/operations/brat_subject_independent_img64_gpu1_2026-07-02.log",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/RESULTS/BRAT")
    return parser.parse_args()


def load_sequence_module():
    spec = importlib.util.spec_from_file_location("sequence_center_package_for_brat", SEQUENCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SEQUENCE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sequence_center_package_for_brat"] = module
    spec.loader.exec_module(module)
    return module


def parse_session_key(name: str) -> tuple[int, str, int]:
    # user37_L_1_0_1 -> (37, left, 101)
    parts = name.split("_")
    if len(parts) == 5 and parts[0].startswith("user") and parts[1] in {"L", "R"}:
        user = int(parts[0].replace("user", ""))
        eye = "left" if parts[1] == "L" else "right"
        session_code = int(f"{int(parts[2])}0{int(parts[4])}")
        return user, eye, session_code
    # Backward-compatible parser for the previous flawed export. It had no eye
    # token and corresponded to right-eye files after directory overwrite.
    if len(parts) != 4 or not parts[0].startswith("user"):
        raise ValueError(f"Unexpected BRAT session name: {name}")
    user = int(parts[0].replace("user", ""))
    session_code = int(f"{int(parts[1])}0{int(parts[3])}")
    return user, "right", session_code


def ellipse_iou_center_proxy(pred_x64: float, pred_y64: float, row: pd.Series) -> float:
    gt = np.array(
        [
            row.gt_x_orig * 64.0 / 346.0,
            row.gt_y_orig * 64.0 / 260.0,
            row.gt_a_orig * 64.0 / 346.0,
            row.gt_b_orig * 64.0 / 260.0,
            row.gt_ang,
        ],
        dtype=float,
    )
    if not np.isfinite(gt).all() or gt[2] <= 0 or gt[3] <= 0:
        return float("nan")
    pred = gt.copy()
    pred[0] = pred_x64
    pred[1] = pred_y64

    canvas_p = np.zeros((64, 64), dtype=np.uint8)
    canvas_g = np.zeros((64, 64), dtype=np.uint8)
    for canvas, values in ((canvas_p, pred), (canvas_g, gt)):
        x, y, a, b, angle = values
        if not np.isfinite(values).all() or a <= 0 or b <= 0:
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


def build_raw_mapping(args: argparse.Namespace, metadata: pd.DataFrame, submission: pd.DataFrame) -> pd.DataFrame:
    names = [line.strip() for line in args.test_files.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows: list[pd.DataFrame] = []
    offset = 0
    for occurrence, name in enumerate(names):
        user, eye, session_code = parse_session_key(name)
        label_rows = sum(1 for _ in (args.event_test_root / name / "label_zeros.txt").open("r", encoding="utf-8"))
        pred_part = submission.iloc[offset : offset + label_rows].copy()
        if len(pred_part) != label_rows:
            raise RuntimeError(f"Submission ended early at {name}: need {label_rows}, got {len(pred_part)}")
        meta_part = metadata[
            (metadata["user"] == user)
            & (metadata["eye"] == eye)
            & (metadata["session_code"].astype(int) == session_code)
        ].sort_values("frame_idx")
        if len(meta_part) != label_rows:
            raise RuntimeError(
                f"Metadata/session length mismatch for {name}: labels={label_rows}, metadata={len(meta_part)}"
            )
        out = meta_part.reset_index(drop=True).copy()
        out["brat_row_id"] = pred_part["row_id"].to_numpy()
        out["brat_session_name"] = name
        out["brat_occurrence"] = occurrence
        out["pred_x_input64"] = pred_part["x"].to_numpy(float)
        out["pred_y_input64"] = pred_part["y"].to_numpy(float)
        rows.append(out)
        offset += label_rows
    if offset != len(submission):
        raise RuntimeError(f"Unused submission rows: offset={offset}, submission={len(submission)}")
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_mod = load_sequence_module()

    metadata = pd.read_csv(args.metadata)
    submission = pd.read_csv(args.submission)
    raw = build_raw_mapping(args, metadata, submission)
    raw.to_csv(args.output_dir / f"{args.model}_subject37_48_raw_test_predictions.csv", index=False)

    dup = raw.duplicated("sample_idx", keep=False)
    if dup.any():
        raw.loc[dup].to_csv(args.output_dir / f"{args.model}_duplicate_sample_predictions.csv", index=False)
        raise RuntimeError(f"BRAT full-test mapping produced duplicate sample_idx rows: {int(dup.sum())}")
    pred = raw[["sample_idx", "pred_x_input64", "pred_y_input64"]].copy()
    pred_meta = metadata[metadata["sample_idx"].isin(pred["sample_idx"])].merge(
        pred, on="sample_idx", how="inner", validate="one_to_one"
    )
    gt_x64 = pred_meta["gt_x_orig"] * 64.0 / 346.0
    gt_y64 = pred_meta["gt_y_orig"] * 64.0 / 260.0
    pred_meta["valid"] = 1
    pred_meta["gt_x_input64"] = gt_x64
    pred_meta["gt_y_input64"] = gt_y64
    pred_meta["error_input64_px"] = np.hypot(
        pred_meta["pred_x_input64"] - gt_x64,
        pred_meta["pred_y_input64"] - gt_y64,
    )
    pred_meta["iou_input64"] = [
        ellipse_iou_center_proxy(r.pred_x_input64, r.pred_y_input64, r)
        for r in pred_meta.itertuples(index=False)
    ]
    pred_meta["iou_note"] = "center_proxy_gt_axes_angle"
    pred_meta.to_csv(args.output_dir / f"{args.model}_subject37_48_test_predictions_with_metadata.csv", index=False)

    joined = sequence_mod.join_motion_labels(pred_meta, args.labels, args.output_dir, args.model)
    stats_df = sequence_mod.aggregate(joined, args.output_dir, args.model)
    workbook = sequence_mod.update_workbook(args.workbook_template, stats_df, args.output_dir, args.model)

    report_args = argparse.Namespace(
        model=args.model,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        device="BRAT saved submission_test.csv",
    )
    report = sequence_mod.write_report(report_args, args.config, stats_df, joined, workbook)

    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for src in (args.checkpoint, args.config, args.submission, args.train_log):
        if src.exists():
            shutil.copy2(src, ckpt_dir / src.name)

    limitation = args.output_dir / f"{args.model}_package_notes.md"
    limitation.write_text(
        "\n".join(
            [
                f"# {args.model} Package Notes",
                "",
                "- Source prediction file: `submission_test.csv` with columns `row_id,x,y`.",
                "- Source test file list has 96 unique eye-session entries with explicit `L`/`R` eye tokens.",
                "- This package maps BRAT predictions to the same FACET subject-independent test metadata used by EIDet.",
                "- Left-eye and right-eye test rows are both represented.",
                "- Pixel error is reported in 64x64 input coordinates.",
                "- IoU is a center proxy using ground-truth ellipse axes/angle shifted to the predicted center.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"model={args.model}")
    print(f"raw_rows={len(raw)}")
    print(f"unique_prediction_rows={len(pred_meta)}")
    print(f"joined_rows={len(joined)}")
    print(f"stats_rows={len(stats_df)}")
    print(f"workbook={workbook}")
    print(f"report={report}")


if __name__ == "__main__":
    main()
