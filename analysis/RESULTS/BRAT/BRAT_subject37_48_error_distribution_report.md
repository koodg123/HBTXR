# BRAT Subject-Independent Test Error Distribution

## Scope

- Model: `BRAT`
- Config: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/configs/hbtxr_subject_independent_img64_fulltest.json`
- Checkpoint: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/logs_hbtxr_img64/BRAT_subject_independent_img64/perror_0.5878_tsf1.0.pth`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `BRAT saved submission_test.csv`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 360,495
- Valid error rows used in statistics: 360,495
- Weighted mean pixel error: 0.8791
- Weighted mean IoU proxy: 0.7708
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_BRAT.xlsx`

## Files

- `BRAT_subject37_48_error_distribution_by_subject_motion.csv`
- `BRAT_subject37_48_test_joined_motion_error.csv`
- `BRAT_subject37_48_joined_motion_counts.csv`
- `BRAT_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_BRAT.xlsx`
