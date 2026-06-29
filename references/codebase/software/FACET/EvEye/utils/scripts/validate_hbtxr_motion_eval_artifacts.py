#!/usr/bin/env python3
"""Validate HBTXR subject/motion evaluation report artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REQUIRED_TABLE_SUFFIXES = [
    "predictions_with_metadata.csv",
    "subject_error_iou_stats.csv",
    "subject_motion_counts.csv",
    "subject_pixel_error_stats.csv",
    "motion_pixel_error_stats.csv",
    "label_precision_floor.csv",
    "pseudolabel_noise_stats.csv",
]

REQUIRED_FIGURES = [
    "fig_subject_pixel_error_box.png",
    "fig_subject_iou_box.png",
    "fig_subject_motion_counts.png",
    "fig_motion_pixel_error_box.png",
    "fig_label_uncertainty_overlay.png",
]

REQUIRED_REPORT_PHRASES = [
    "Subject-wise Pixel Error / IoU Distribution",
    "Subject-wise Motion Distribution",
    "Subject-wise Mean / Median / P95 / P99 Pixel Error",
    "Motion-wise Mean / Median / P95 / P99 Pixel Error",
    "Annotation Precision, Label Noise",
]

REQUIRED_PRED_COLUMNS = [
    "sample_idx",
    "subject",
    "motion_state",
    "error_input64_px",
    "iou_input64",
]

EXPECTED_COUNTS = {
    "val": 122776,
    "test": 366171,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=Path("references/report/FACET"))
    parser.add_argument("--run-name", default="HBTXR_subject_independent_img64_patch4")
    parser.add_argument("--date", default="2026-06-28")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def prefix(run_name: str, split: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_")
    return f"{safe}_{split}"


def require_path(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"missing: {path}")
    elif path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty file: {path}")


def validate_csv(path: Path, required_columns: list[str], errors: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"failed to read CSV {path}: {exc}")
        return pd.DataFrame()
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        errors.append(f"{path} missing columns: {missing}")
    if len(df) == 0:
        errors.append(f"{path} has zero rows")
    return df


def validate_split(report_root: Path, run_name: str, date: str, split: str) -> dict:
    split_prefix = prefix(run_name, split)
    output_dir = report_root / f"{run_name}_{split}_motion_eval"
    report_path = report_root / f"{run_name}_{split}_motion_eval_{date}.md"
    errors: list[str] = []
    details: dict = {
        "split": split,
        "output_dir": str(output_dir),
        "report": str(report_path),
    }

    require_path(output_dir, errors)
    require_path(report_path, errors)

    if report_path.exists() and report_path.stat().st_size > 0:
        text = report_path.read_text(encoding="utf-8", errors="replace")
        missing_phrases = [phrase for phrase in REQUIRED_REPORT_PHRASES if phrase not in text]
        if missing_phrases:
            errors.append(f"{report_path} missing report sections: {missing_phrases}")

    table_paths = {}
    for suffix in REQUIRED_TABLE_SUFFIXES:
        path = output_dir / f"{split_prefix}_{suffix}"
        table_paths[suffix] = str(path)
        require_path(path, errors)

    figure_paths = {}
    for figure in REQUIRED_FIGURES:
        path = output_dir / "figures" / figure
        figure_paths[figure] = str(path)
        require_path(path, errors)

    pred_path = output_dir / f"{split_prefix}_predictions_with_metadata.csv"
    pred = validate_csv(pred_path, REQUIRED_PRED_COLUMNS, errors)
    expected_count = EXPECTED_COUNTS.get(split)
    if expected_count is not None and len(pred) not in (0, expected_count):
        errors.append(f"{pred_path} row count {len(pred)} != expected {expected_count}")
    if len(pred):
        details["prediction_rows"] = int(len(pred))
        details["subjects"] = sorted(str(x) for x in pred["subject"].dropna().unique())
        details["motion_states"] = sorted(str(x) for x in pred["motion_state"].dropna().unique())
        for state in ["Fixation", "Saccade", "Smooth"]:
            if state not in set(details["motion_states"]):
                errors.append(f"{pred_path} missing motion state: {state}")

    subject_stats = validate_csv(
        output_dir / f"{split_prefix}_subject_error_iou_stats.csv",
        ["subject", "err_n", "err_mean", "err_median", "err_p95", "err_p99", "iou_mean", "iou_median"],
        errors,
    )
    if len(subject_stats):
        details["subject_rows"] = int(len(subject_stats))

    motion_stats = validate_csv(
        output_dir / f"{split_prefix}_motion_pixel_error_stats.csv",
        ["motion_state", "err_n", "err_mean", "err_median", "err_p95", "err_p99"],
        errors,
    )
    if len(motion_stats) and "All" not in set(motion_stats["motion_state"].astype(str)):
        errors.append(f"{split_prefix}_motion_pixel_error_stats.csv missing All row")

    details["tables"] = table_paths
    details["figures"] = figure_paths
    details["errors"] = errors
    details["ok"] = not errors
    return details


def main() -> None:
    args = parse_args()
    results = [validate_split(args.report_root, args.run_name, args.date, split) for split in args.splits]
    payload = {
        "run_name": args.run_name,
        "date": args.date,
        "ok": all(item["ok"] for item in results),
        "splits": results,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
