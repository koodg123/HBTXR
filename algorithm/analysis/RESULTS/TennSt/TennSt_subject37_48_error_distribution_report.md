# TennSt Subject-Independent Test Error Distribution

## Scope

- Model: `TennSt`
- Config: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/configs/DavisEyeEllipse_TennSt_subject_independent_img64.yaml`
- Checkpoint: `references/codebase/software/FACET/runs/logs/TennSt_subject_independent_img64/version_0/checkpoints/epoch=67-val_distance=2.0011.ckpt`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `cuda:1`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 360,105
- Valid error rows used in statistics: 359,865
- Weighted mean pixel error: 1.6446
- Weighted mean IoU proxy: 0.5757
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_TennSt.xlsx`

## Files

- `TennSt_subject37_48_error_distribution_by_subject_motion.csv`
- `TennSt_subject37_48_test_joined_motion_error.csv`
- `TennSt_subject37_48_joined_motion_counts.csv`
- `TennSt_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_TennSt.xlsx`
