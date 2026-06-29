#!/usr/bin/env python3
"""Build a detailed HBTXR val motion-evaluation report without repeated-run CI."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPORT_ROOT = Path("references/report/FACET")
EVALUATION_ROOT = REPORT_ROOT / "evaluation"
EVAL_DIR = EVALUATION_ROOT / "HBTXR_val_motion_eval"
REPORT_PATH = EVALUATION_ROOT / "HBTXR_val_motion_eval_detailed_no_ci_2026-06-29.md"


def fmt(v: float, ndigits: int = 3) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{float(v):.{ndigits}f}"


def as_int(v: float) -> str:
    if pd.isna(v):
        return "n/a"
    return f"{int(v):,}"


def markdown_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None) -> list[str]:
    rename = rename or {}
    table = df.loc[:, columns].rename(columns=rename)
    lines = ["| " + " | ".join(table.columns) + " |"]
    lines.append("|" + "|".join(["---"] * len(table.columns)) + "|")
    for _, row in table.iterrows():
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(fmt(value))
            elif isinstance(value, int):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def image_link(name: str, alt: str) -> str:
    return f"![{alt}](HBTXR_val_motion_eval/figures/{name})"


def main() -> None:
    subject_stats = pd.read_csv(EVAL_DIR / "hbtxr_val_subject_error_iou_stats.csv")
    motion_counts = pd.read_csv(EVAL_DIR / "hbtxr_val_subject_motion_counts.csv")
    subject_pixel = pd.read_csv(EVAL_DIR / "hbtxr_val_subject_pixel_error_stats.csv")
    motion_pixel = pd.read_csv(EVAL_DIR / "hbtxr_val_motion_pixel_error_stats.csv")
    label_precision = pd.read_csv(EVAL_DIR / "hbtxr_label_precision_floor.csv")
    label_noise = pd.read_csv(EVAL_DIR / "hbtxr_pseudolabel_noise_stats.csv")
    merged = pd.read_csv(
        EVAL_DIR / "hbtxr_val_predictions_with_metadata.csv",
        usecols=["sample_idx", "valid", "subject", "motion_state", "error_input64_px", "iou_input64"],
    )

    valid = merged[merged["valid"] == 1].copy()
    all_motion = motion_pixel[motion_pixel["motion_state"] == "All"].iloc[0]
    all_noise = label_noise[label_noise["motion_state"] == "All"].iloc[0]
    label_floor = label_precision.iloc[0]
    total_motion = motion_counts[["Fixation", "Saccade", "Smooth"]].sum()
    total_motion_n = int(total_motion.sum())
    top_median = subject_pixel.sort_values("err_median", ascending=False).head(3)
    top_tail = subject_pixel.sort_values("err_p99", ascending=False).head(3)
    best_subject = subject_pixel.sort_values("err_median", ascending=True).iloc[0]
    worst_subject = subject_pixel.sort_values("err_median", ascending=False).iloc[0]

    lines: list[str] = []
    lines.append("# HBTXR Val Motion Evaluation Detailed Report")
    lines.append("")
    lines.append("## Technical Summary")
    lines.append("")
    lines.append(
        f"- `HBTXR_full_unet_img64_patch4` was evaluated on `DeanDataset_full_unet/val` with {len(merged):,} samples; "
        f"{len(valid):,} samples were valid for center-error and IoU statistics."
    )
    lines.append(
        f"- Overall center error in 64x64 input coordinates is mean {fmt(all_motion.err_mean)} px, "
        f"median {fmt(all_motion.err_median)} px, P95 {fmt(all_motion.err_p95)} px, and P99 {fmt(all_motion.err_p99)} px."
    )
    lines.append(
        f"- The subject-wise median error ranges from {fmt(subject_pixel.err_median.min())} px "
        f"({best_subject.subject}) to {fmt(subject_pixel.err_median.max())} px ({worst_subject.subject}); "
        f"tail risk is largest for {top_tail.iloc[0].subject} with P99 {fmt(top_tail.iloc[0].err_p99)} px."
    )
    lines.append(
        f"- Velocity-based motion labels are highly imbalanced: Fixation {int(total_motion.Fixation):,} "
        f"({100 * total_motion.Fixation / total_motion_n:.2f}%), Saccade {int(total_motion.Saccade):,} "
        f"({100 * total_motion.Saccade / total_motion_n:.2f}%), Smooth {int(total_motion.Smooth):,} "
        f"({100 * total_motion.Smooth / total_motion_n:.2f}%). Saccade results are therefore descriptive only."
    )
    lines.append(
        f"- Annotation uncertainty is not negligible at sub-pixel scale: integer-coordinate manual annotation floor is "
        f"mean {fmt(label_floor.floor_mean_px)} px and the matched U-Net pseudo-label center noise is "
        f"median {fmt(all_noise.center_noise_median)} px / P95 {fmt(all_noise.center_noise_p95)} px."
    )
    lines.append("- Repeated-run confidence intervals requested in item (6) are intentionally excluded from this report.")
    lines.append("")

    lines.append("## Scope, Data, And Metric Definitions")
    lines.append("")
    lines.append("- Model: `HBTXR_full_unet_img64_patch4`.")
    lines.append("- Checkpoint: `references/codebase/software/FACET/runs/logs/HBTXR_full_unet_img64_patch4/version_0/checkpoints/epoch=67-val_mean_distance=0.4492.ckpt`.")
    lines.append("- Evaluation split: `DeanDataset_full_unet/val`; this is validation data, not the final held-out test split.")
    lines.append("- Pixel error: Euclidean distance between predicted and target pupil center, reported in 64x64 input-image pixels.")
    lines.append("- IoU: rasterized predicted ellipse vs target ellipse on the 64x64 input canvas.")
    lines.append("- Motion state: velocity-based rule using pseudo-label pupil-center trajectory.")
    lines.append("- `Saccade`: center speed > 493 px/s. `Fixation`: speed <= 493 px/s in session regimes 101/201. `Smooth`: speed <= 493 px/s in session regimes 102/202.")
    lines.append("")

    lines.append("## (1) Subject-wise Pixel Error / IoU Distribution")
    lines.append("")
    lines.append(
        "Subject-level distributions show that typical center localization is mostly below 1 px, "
        "but IoU and tail errors vary materially by subject. The largest median-error subject is "
        f"{worst_subject.subject} ({fmt(worst_subject.err_median)} px), while the lowest median-error subject is "
        f"{best_subject.subject} ({fmt(best_subject.err_median)} px)."
    )
    lines.append("")
    lines.append(image_link("fig_subject_pixel_error_box.png", "Subject-wise pixel error boxplot"))
    lines.append("")
    lines.append(image_link("fig_subject_iou_box.png", "Subject-wise IoU boxplot"))
    lines.append("")
    lines.extend(
        markdown_table(
            subject_stats,
            ["subject", "err_n", "err_mean", "err_median", "err_p95", "err_p99", "iou_mean", "iou_median", "iou_p95", "iou_p99"],
            {"err_n": "n"},
        )
    )
    lines.append("")
    lines.append(
        "Interpretation: median error is the most stable indicator of normal tracking quality, while P95/P99 captures rare failure modes. "
        "IoU is systematically lower for subjects with poorer ellipse geometry even when center error remains modest; this is why center error and IoU should be reported together."
    )
    lines.append("")

    lines.append("## (2) Subject-wise Motion Distribution")
    lines.append("")
    lines.append(
        "The val split contains enough Fixation and Smooth samples for stable descriptive statistics, but Saccade is extremely sparse. "
        "This makes Saccade useful as a sanity-check slice, not as a robust performance claim."
    )
    lines.append("")
    lines.append(image_link("fig_subject_motion_counts.png", "Subject-wise motion count stacked bars"))
    lines.append("")
    lines.extend(
        markdown_table(
            motion_counts,
            ["subject", "Fixation", "Saccade", "Smooth", "total", "pct_fixation", "pct_saccade", "pct_smooth"],
        )
    )
    lines.append("")
    lines.append(
        f"Interpretation: Saccade has only {int(total_motion.Saccade):,} samples out of {total_motion_n:,} "
        f"({100 * total_motion.Saccade / total_motion_n:.2f}%). Any paper text should explicitly state this imbalance and avoid overclaiming saccade generalization from this val split."
    )
    lines.append("")

    lines.append("## (3) Subject-wise Mean / Median / P95 / P99 Pixel Error")
    lines.append("")
    lines.append(
        "Subject-wise summary statistics confirm that mean error is dominated by tail events for several subjects. "
        f"The three largest median-error subjects are {', '.join(top_median.subject.tolist())}; "
        f"the three largest P99 subjects are {', '.join(top_tail.subject.tolist())}."
    )
    lines.append("")
    lines.extend(markdown_table(subject_pixel, ["subject", "err_n", "err_mean", "err_median", "err_p95", "err_p99", "err_max"], {"err_n": "n"}))
    lines.append("")
    lines.append("Highest-median subjects:")
    for _, row in top_median.iterrows():
        lines.append(
            f"- {row.subject}: median {fmt(row.err_median)} px, mean {fmt(row.err_mean)} px, "
            f"P95 {fmt(row.err_p95)} px, P99 {fmt(row.err_p99)} px, n={as_int(row.err_n)}."
        )
    lines.append("")
    lines.append("Largest-tail subjects:")
    for _, row in top_tail.iterrows():
        lines.append(
            f"- {row.subject}: P99 {fmt(row.err_p99)} px, P95 {fmt(row.err_p95)} px, "
            f"median {fmt(row.err_median)} px, max {fmt(row.err_max)} px."
        )
    lines.append("")
    lines.append(
        "Interpretation: subject-specific tails should be inspected against blink, occlusion, poor pseudo-labels, and motion imbalance. "
        "For reporting, mean/median/P95/P99 are all needed because the mean alone under-describes rare but large center errors."
    )
    lines.append("")

    lines.append("## (4) Motion-wise Mean / Median / P95 / P99 Pixel Error")
    lines.append("")
    lines.append(
        "Motion-wise results show Fixation and Smooth have similar median-scale behavior, while Smooth has the larger P99 tail in this val split. "
        "Saccade appears low-error here, but its n=58 denominator is too small for a stable conclusion."
    )
    lines.append("")
    lines.append(image_link("fig_motion_pixel_error_box.png", "Motion-wise pixel error boxplot"))
    lines.append("")
    lines.extend(markdown_table(motion_pixel, ["motion_state", "err_n", "err_mean", "err_median", "err_p95", "err_p99", "err_max"], {"motion_state": "motion", "err_n": "n"}))
    lines.append("")
    lines.append(
        "Interpretation: report Fixation and Smooth as the main motion-regime comparison. "
        "For Saccade, report the number and mark it as underpowered; do not claim that saccades are easier simply because this sparse subset has lower errors."
    )
    lines.append("")

    lines.append("## (7) Annotation Precision And Label Noise")
    lines.append("")
    lines.append(
        "The evaluation target is a U-Net-generated pseudo-label, not a repeated independent human annotation. "
        "Two uncertainty levels therefore matter: the manual annotation coordinate floor and the pseudo-label deviation from matched manual-GT frames."
    )
    lines.append("")
    lines.append(image_link("fig_label_uncertainty_overlay.png", "HBTXR error and label uncertainty overlay"))
    lines.append("")
    lines.append("Manual annotation precision floor:")
    lines.extend(
        markdown_table(
            label_precision,
            ["source", "n_annotations_reference", "per_axis_label_std_px", "floor_mean_px", "floor_median_px", "floor_p95_px"],
            {"n_annotations_reference": "n_ref"},
        )
    )
    lines.append("")
    lines.append("Pseudo-label center noise against matched manual-GT frames:")
    lines.extend(
        markdown_table(
            label_noise,
            [
                "motion_state",
                "center_noise_n",
                "center_noise_mean",
                "center_noise_median",
                "center_noise_p95",
                "center_noise_p99",
                "center_noise_max",
            ],
            {"motion_state": "motion", "center_noise_n": "n"},
        )
    )
    lines.append("")
    lines.append(
        f"Interpretation: the model's overall median error ({fmt(all_motion.err_median)} px) is only moderately larger than the pseudo-label median noise "
        f"({fmt(all_noise.center_noise_median)} px), while its P95 error ({fmt(all_motion.err_p95)} px) is much larger than pseudo-label P95 noise "
        f"({fmt(all_noise.center_noise_p95)} px). This means typical sub-pixel improvements should be discussed cautiously, but the tail-error reduction problem remains larger than annotation uncertainty."
    )
    lines.append("")

    lines.append("## Limitations And Robustness Notes")
    lines.append("")
    lines.append("- This report uses validation data only; the final subject-independent test result should be reported separately after that training/evaluation finishes.")
    lines.append("- Motion labels are derived from pseudo-label trajectory velocity, so motion-class uncertainty is coupled to label quality.")
    lines.append("- Saccade is severely underrepresented in this val split; its statistics are descriptive and not suitable for strong claims.")
    lines.append("- Item (6), repeated-run confidence interval, requires multiple independently trained checkpoints and is excluded by request.")
    lines.append("")

    lines.append("## Recommended Next Steps")
    lines.append("")
    lines.append("1. Use this report for reviewer response sections that do not require repeated-run CI.")
    lines.append("2. Re-run the same report builder on the subject-independent val/test outputs once the current 70-epoch subject-independent HBTXR training finishes.")
    lines.append("3. If Saccade performance is important, construct or sample a larger saccade-focused evaluation subset before making a strong motion-type claim.")
    lines.append("")

    lines.append("## Source Artifacts")
    lines.append("")
    lines.append("- Main evaluator: `references/codebase/software/FACET/EvEye/utils/scripts/evaluate_hbtxr_val_motion.py`.")
    lines.append("- Detailed report builder: `references/codebase/software/FACET/EvEye/utils/scripts/build_hbtxr_val_motion_detailed_report.py`.")
    lines.append("- CSV output directory: `references/report/FACET/evaluation/HBTXR_val_motion_eval/`.")
    lines.append("- Figure output directory: `references/report/FACET/evaluation/HBTXR_val_motion_eval/figures/`.")
    lines.append("- Prior short report: `references/report/FACET/evaluation/HBTXR_val_motion_eval_2026-06-28.md`.")
    lines.append("")

    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
