# TDTracker Subject-Independent Test Error Distribution

## Scope

- Model: `TDTracker`
- Config: `/home/kjm26/project/dataset/XR/EV_Eye/target_data/tdtracker_hbtxr_img64_seq100/test_hbtxr_img64_seq100.h5`
- Checkpoint: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2025/tdtracker/checkpoint/HBTXR_subject_independent_img64/last_checkpoint.pth`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `cuda:1`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 358,838
- Valid error rows used in statistics: 358,598
- Weighted mean pixel error: 6.4147
- Weighted mean IoU proxy: 0.2830
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_TDTracker.xlsx`

## Files

- `TDTracker_subject37_48_error_distribution_by_subject_motion.csv`
- `TDTracker_subject37_48_test_joined_motion_error.csv`
- `TDTracker_subject37_48_joined_motion_counts.csv`
- `TDTracker_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_TDTracker.xlsx`
