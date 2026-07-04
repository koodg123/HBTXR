# EPNet_subject_independent_img64 Subject-Independent Test Error Distribution

## Scope

- Model: `EPNet_subject_independent_img64`
- Config: `references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_subject_independent_img64.yaml`
- Checkpoint: `references/codebase/software/FACET/runs/logs/EPNet_subject_independent_img64/version_0/checkpoints/epoch=57-val_mean_distance=0.5780.ckpt`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `cuda:1`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 360,495
- Valid error rows used in statistics: 360,255
- Weighted mean pixel error: 1.4671
- Weighted mean IoU proxy: 0.5280
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_EPNet_subject_independent_img64.xlsx`

## Files

- `EPNet_subject_independent_img64_subject37_48_error_distribution_by_subject_motion.csv`
- `EPNet_subject_independent_img64_subject37_48_test_joined_motion_error.csv`
- `EPNet_subject_independent_img64_subject37_48_joined_motion_counts.csv`
- `EPNet_subject_independent_img64_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_EPNet_subject_independent_img64.xlsx`
