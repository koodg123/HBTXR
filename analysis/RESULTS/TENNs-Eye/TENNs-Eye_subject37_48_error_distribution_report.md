# TENNs-Eye Subject-Independent Test Error Distribution

## Scope

- Model: `TENNs-Eye`
- Config: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2024/eye_track_spatiotemporal/runs/TENNs_Eye_subject_independent_img64_20260630_010754/config.yaml`
- Checkpoint: `references/codebase/software/ais2024/eye_track_spatiotemporal/runs/TENNs_Eye_subject_independent_img64_20260630_010754/checkpoints/best_epoch068_val_distance_1.8929.pth`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `cuda:1`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 360,105
- Valid error rows used in statistics: 359,865
- Weighted mean pixel error: 1.5056
- Weighted mean IoU proxy: 0.6135
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_TENNs-Eye.xlsx`

## Files

- `TENNs-Eye_subject37_48_error_distribution_by_subject_motion.csv`
- `TENNs-Eye_subject37_48_test_joined_motion_error.csv`
- `TENNs-Eye_subject37_48_joined_motion_counts.csv`
- `TENNs-Eye_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_TENNs-Eye.xlsx`
