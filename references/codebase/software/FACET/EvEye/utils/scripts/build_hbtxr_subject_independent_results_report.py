#!/usr/bin/env python3
"""Build the final HBTXR subject-independent val/test results report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, default=Path("references/report/FACET"))
    parser.add_argument("--run-name", default="HBTXR_subject_independent_img64_patch4")
    parser.add_argument("--date", default="2026-06-28")
    parser.add_argument("--config", type=Path, default=Path("references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def safe_prefix(run_name: str, split: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', run_name).strip('_')}_{split}"


def fmt(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.3f}"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def split_paths(report_root: Path, run_name: str, split: str) -> dict[str, Path]:
    prefix = safe_prefix(run_name, split)
    output_dir = report_root / f"{run_name}_{split}_motion_eval"
    return {
        "output_dir": output_dir,
        "report": report_root / f"{run_name}_{split}_motion_eval_2026-06-28.md",
        "merged": output_dir / f"{prefix}_predictions_with_metadata.csv",
        "subject_stats": output_dir / f"{prefix}_subject_error_iou_stats.csv",
        "subject_motion": output_dir / f"{prefix}_subject_motion_counts.csv",
        "subject_pixel": output_dir / f"{prefix}_subject_pixel_error_stats.csv",
        "motion_pixel": output_dir / f"{prefix}_motion_pixel_error_stats.csv",
        "label_precision": output_dir / f"{prefix}_label_precision_floor.csv",
        "label_noise": output_dir / f"{prefix}_pseudolabel_noise_stats.csv",
        "figures": output_dir / "figures",
    }


def load_split_summary(report_root: Path, run_name: str, split: str) -> dict:
    paths = split_paths(report_root, run_name, split)
    merged = read_csv(paths["merged"], usecols=["sample_idx", "subject", "motion_state", "error_input64_px", "iou_input64"])
    subject = read_csv(paths["subject_stats"])
    subject_pixel = read_csv(paths["subject_pixel"])
    subject_motion = read_csv(paths["subject_motion"])
    motion = read_csv(paths["motion_pixel"])
    label_noise = read_csv(paths["label_noise"]) if paths["label_noise"].exists() and paths["label_noise"].stat().st_size else pd.DataFrame()
    all_row = motion[motion["motion_state"].astype(str) == "All"].iloc[0]
    top_subjects = subject_pixel.sort_values("err_median", ascending=False).head(5)
    motion_rows = motion[["motion_state", "err_n", "err_mean", "err_median", "err_p95", "err_p99", "err_max"]]
    return {
        "paths": paths,
        "merged": merged,
        "subject": subject,
        "subject_pixel": subject_pixel,
        "subject_motion": subject_motion,
        "motion": motion,
        "motion_rows": motion_rows,
        "label_noise": label_noise,
        "all_row": all_row,
        "top_subjects": top_subjects,
    }


def validation_summary(path: Path | None) -> tuple[str, dict | None]:
    if path is None:
        return "not provided", None
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "passed" if payload.get("ok") else "failed", payload


def write_split_section(lines: list[str], split: str, data: dict) -> None:
    merged = data["merged"]
    all_row = data["all_row"]
    subject = data["subject"]
    paths = data["paths"]
    lines.append(f"## {split.upper()} Results")
    lines.append("")
    lines.append(f"- Samples: {len(merged):,}")
    lines.append(f"- Valid error samples: {int(all_row.err_n):,}")
    lines.append(f"- Center error: mean {fmt(all_row.err_mean)} px, median {fmt(all_row.err_median)} px, P95 {fmt(all_row.err_p95)} px, P99 {fmt(all_row.err_p99)} px.")
    lines.append(f"- IoU: mean {fmt(merged.iou_input64.mean())}, median {fmt(merged.iou_input64.median())}.")
    lines.append(f"- Subject median error range: {fmt(subject.err_median.min())} to {fmt(subject.err_median.max())} px.")
    lines.append(f"- Subject median IoU range: {fmt(subject.iou_median.min())} to {fmt(subject.iou_median.max())}.")
    lines.append(f"- Detailed report: `{paths['report']}`")
    lines.append(f"- Artifact directory: `{paths['output_dir']}`")
    lines.append("")
    lines.append("### Motion-Wise Pixel Error")
    lines.append("")
    lines.append("| motion | n | mean | median | P95 | P99 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in data["motion_rows"].iterrows():
        lines.append(
            f"| {row.motion_state} | {int(row.err_n):,} | {fmt(row.err_mean)} | {fmt(row.err_median)} | {fmt(row.err_p95)} | {fmt(row.err_p99)} | {fmt(row.err_max)} |"
        )
    lines.append("")
    lines.append("### Highest-Median Subjects")
    lines.append("")
    lines.append("| subject | n | mean | median | P95 | P99 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, row in data["top_subjects"].iterrows():
        lines.append(f"| {row.subject} | {int(row.err_n):,} | {fmt(row.err_mean)} | {fmt(row.err_median)} | {fmt(row.err_p95)} | {fmt(row.err_p99)} |")
    lines.append("")


def main() -> None:
    args = parse_args()
    output = args.output or (args.report_root / f"{args.run_name}_results_{args.date}.md")
    validation_state, validation_payload = validation_summary(args.validation_json)
    val = load_split_summary(args.report_root, args.run_name, "val")
    test = load_split_summary(args.report_root, args.run_name, "test")

    lines: list[str] = []
    lines.append(f"# {args.run_name} Subject-Independent Results")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Config: `{args.config}`")
    lines.append(f"- Checkpoint: `{args.checkpoint}`")
    lines.append("- Dataset: `DeanDataset_full_unet_subject_independent`")
    lines.append("- Split: train subjects 1-32, val subjects 33-36, test subjects 37-48.")
    lines.append("- Model setting: `img_size=64`, `patch_size=4`, output heatmap `16x16`.")
    lines.append("- Motion states: velocity-based `Fixation`, `Saccade`, `Smooth` using the same rule as `HBTXR_val_motion_eval`.")
    lines.append("- Repeated-run confidence intervals are intentionally excluded from this single-seed run.")
    lines.append(f"- Artifact validation: `{validation_state}`.")
    if args.validation_json:
        lines.append(f"- Validation JSON: `{args.validation_json}`")
    lines.append("")
    lines.append("## Split Summary")
    lines.append("")
    lines.append("| split | subjects | samples | mean err | median err | P95 | P99 | mean IoU | median IoU |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, subjects, data in [
        ("val", "33-36", val),
        ("test", "37-48", test),
    ]:
        all_row = data["all_row"]
        merged = data["merged"]
        lines.append(
            f"| {split} | {subjects} | {len(merged):,} | {fmt(all_row.err_mean)} | {fmt(all_row.err_median)} | {fmt(all_row.err_p95)} | {fmt(all_row.err_p99)} | {fmt(merged.iou_input64.mean())} | {fmt(merged.iou_input64.median())} |"
        )
    lines.append("")
    write_split_section(lines, "val", val)
    write_split_section(lines, "test", test)
    lines.append("## Required Artifact Coverage")
    lines.append("")
    lines.append("The generated per-split reports cover:")
    lines.append("")
    lines.append("1. Subject-wise pixel error / IoU distribution.")
    lines.append("2. Subject-wise motion distribution.")
    lines.append("3. Subject-wise mean / median / P95 / P99 pixel error.")
    lines.append("4. Motion-wise mean / median / P95 / P99 pixel error.")
    lines.append("5. Annotation precision and pseudo-label noise.")
    lines.append("")
    if validation_payload is not None:
        for split_result in validation_payload.get("splits", []):
            status = "ok" if split_result.get("ok") else "failed"
            lines.append(f"- `{split_result.get('split')}` validation: {status}")
    lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote report: {output}")


if __name__ == "__main__":
    main()
