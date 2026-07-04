#!/usr/bin/env python3
"""Build subject37-48 error-distribution CSVs for the epoch-8 HBTXR frame package."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "HBTXR-Frame-subjet_independent_imgsz64-epoch8"
REFERENCE_JOINED = (
    ROOT.parent
    / "HBTXR_subject37_48_error_distribution"
    / "HBTXR_subject37_48_test_joined_motion_error.csv"
)


def main() -> None:
    pred_path = ROOT / f"{PREFIX}_test_predictions_with_metadata.csv"
    pred = pd.read_csv(pred_path)
    pred_37_48 = pred[pred["user"].between(37, 48)].copy()
    pred_37_48.to_csv(
        ROOT / f"{PREFIX}_subject37_48_test_predictions_with_metadata.csv",
        index=False,
    )

    ref = pd.read_csv(
        REFERENCE_JOINED,
        usecols=["sample_idx", "motion_state", "motion_label_speed_pxps"],
    ).rename(columns={"motion_label_speed_pxps": "speed_pxps"})
    ref = ref.drop_duplicates("sample_idx")

    merged = pred_37_48.merge(
        ref,
        on="sample_idx",
        how="left",
        suffixes=("_x", "_y"),
        indicator=True,
    )
    dropped = merged[merged["_merge"] == "left_only"].copy()
    dropped.to_csv(ROOT / f"{PREFIX}_subject37_48_dropped_blink_predictions.csv", index=False)
    dropped.to_csv(ROOT / f"{PREFIX}_subject37_48_unmatched_predictions.csv", index=False)

    joined = merged[merged["_merge"] == "both"].copy()
    joined = joined.drop(columns=["speed_pxps_x", "motion_state_x", "_merge"])
    joined = joined.rename(
        columns={
            "speed_pxps_y": "motion_label_speed_pxps",
            "motion_state_y": "motion_state",
        }
    )
    joined.to_csv(ROOT / f"{PREFIX}_subject37_48_test_joined_motion_error.csv", index=False)

    counts = (
        joined.groupby(["user", "motion_state"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ["Fixation", "Saccade", "Smooth"]:
        if col not in counts:
            counts[col] = 0
    counts = counts[["user", "Fixation", "Saccade", "Smooth"]].sort_values("user")
    counts.to_csv(ROOT / f"{PREFIX}_subject37_48_joined_motion_counts.csv", index=False)

    valid = joined[
        (joined["valid"] == 1)
        & joined["error_input64_px"].notna()
        & joined["iou_input64"].notna()
    ].copy()
    rows = []
    for (user, motion), group in valid.groupby(["user", "motion_state"], sort=True):
        rows.append(
            {
                "Subject": int(user),
                "Split": "Test",
                "Motion": motion,
                "N": int(len(group)),
                "Mean": float(group["error_input64_px"].mean()),
                "Median": float(group["error_input64_px"].median()),
                "P95": float(group["error_input64_px"].quantile(0.95)),
                "P99": float(group["error_input64_px"].quantile(0.99)),
                "IoU_Mean": float(group["iou_input64"].mean()),
                "IoU_Median": float(group["iou_input64"].median()),
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(
        ROOT / f"{PREFIX}_subject37_48_error_distribution_by_subject_motion.csv",
        index=False,
    )

    weighted_mean = valid["error_input64_px"].mean()
    weighted_iou = valid["iou_input64"].mean()
    report = [
        f"# {PREFIX} Subject-Independent Test Error Distribution",
        "",
        "## Scope",
        "",
        f"- Model: `{PREFIX}`",
        "- Config: `configs/DavisEyeEllipse_HBTXR_frame_cached_si_img128_patch4_epoch8_eval.yaml`",
        "- Checkpoint: `checkpoints/epoch8-final-08-00544995.ckpt`",
        "- Split: `test`, subjects `37-48`.",
        "- Blink rows are excluded by reusing the non-Blink sample index set from `HBTXR_subject37_48_test_joined_motion_error.csv`.",
        "- Error column name is kept as `error_input64_px` for compatibility with existing tables; this package was generated from the epoch-8 frame config stored in `configs/`.",
        "",
        "## Summary",
        "",
        f"- Subject37-48 prediction rows: {len(pred_37_48):,}",
        f"- Joined non-Blink rows: {len(joined):,}",
        f"- Dropped/left-only rows: {len(dropped):,}",
        f"- Valid error rows used in statistics: {len(valid):,}",
        f"- Weighted mean pixel error: {weighted_mean:.4f}",
        f"- Weighted mean IoU: {weighted_iou:.4f}",
        "",
        "## Files",
        "",
        f"- `{PREFIX}_subject37_48_error_distribution_by_subject_motion.csv`",
        f"- `{PREFIX}_subject37_48_test_joined_motion_error.csv`",
        f"- `{PREFIX}_subject37_48_joined_motion_counts.csv`",
        f"- `{PREFIX}_subject37_48_dropped_blink_predictions.csv`",
        f"- `{PREFIX}_subject37_48_unmatched_predictions.csv`",
        f"- `{PREFIX}_subject37_48_test_predictions_with_metadata.csv`",
        "",
    ]
    (ROOT / f"{PREFIX}_subject37_48_error_distribution_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"wrote subject37-48 package CSVs to {ROOT}")
    print(f"prediction_rows={len(pred_37_48)} joined={len(joined)} dropped={len(dropped)} valid={len(valid)}")
    print(f"weighted_mean={weighted_mean:.6f} weighted_iou={weighted_iou:.6f}")


if __name__ == "__main__":
    main()
