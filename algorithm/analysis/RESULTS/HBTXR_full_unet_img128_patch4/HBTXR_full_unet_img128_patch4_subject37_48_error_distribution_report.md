# HBTXR_full_unet_img128_patch4 Subject-Independent Test Error Distribution

## Scope

- Model: `HBTXR_full_unet_img128_patch4`
- Config: `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet_img128_patch4.yaml`
- Checkpoint: `references/codebase/software/FACET/runs/logs/HBTXR_full_unet_img128_patch4/version_0/checkpoints/epoch=56-val_mean_distance=0.4277.ckpt`
- Split: `test`, subjects `37-48`.
- Device requested for inference: `cuda:1`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in 64x64 input coordinates.
- IoU is `center_proxy_gt_axes_angle`: GT ellipse axes/angle shifted to the predicted center.

## Summary

- Joined non-Blink rows: 360,495
- Valid error rows used in statistics: 360,456
- Weighted mean pixel error: 0.5486
- Weighted mean IoU proxy: 0.6105
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_HBTXR_full_unet_img128_patch4.xlsx`

## Files

- `HBTXR_full_unet_img128_patch4_subject37_48_error_distribution_by_subject_motion.csv`
- `HBTXR_full_unet_img128_patch4_subject37_48_test_joined_motion_error.csv`
- `HBTXR_full_unet_img128_patch4_subject37_48_joined_motion_counts.csv`
- `HBTXR_full_unet_img128_patch4_subject37_48_dropped_blink_predictions.csv`
- `JETCAS_REPLY_TABLES (Error-Distributions)_HBTXR_full_unet_img128_patch4.xlsx`
