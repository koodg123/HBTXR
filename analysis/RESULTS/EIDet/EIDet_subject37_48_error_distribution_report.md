# EIDet Subject-Independent Test Error Distribution

## Scope

- Model: `EIDet`
- Config: `references/codebase/software/FACET/configs/DavisEyeEllipse_ElNet_subject_independent_img64.yaml`
- Checkpoint: `references/codebase/software/FACET/runs/logs/ElNet_subject_independent_img64/version_0/checkpoints/epoch=01-val_mean_distance=1.1483.ckpt`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `cuda:0`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 360,495
- Valid error rows used in statistics: 360,255
- Weighted mean pixel error: 3.8497
- Weighted mean IoU proxy: 0.3574
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_EIDet.xlsx`

## Files

- `EIDet_subject37_48_error_distribution_by_subject_motion.csv`
- `EIDet_subject37_48_test_joined_motion_error.csv`
- `EIDet_subject37_48_joined_motion_counts.csv`
- `EIDet_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_EIDet.xlsx`
