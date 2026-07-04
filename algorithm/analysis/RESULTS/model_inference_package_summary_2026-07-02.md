# Model Inference Package Summary - 2026-07-02

## Scope

Generated Subject-Independent Test result packages under `analysis/RESULTS` for:

- `HBTXR_full_unet_img128_patch4`
- `EPNet_subject_independent_img64`
- `TDTracker`

All tables use test subjects `37-48` and join against `analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv`, so Blink rows are excluded from error-distribution statistics.

## Package Outputs

| Model | Output directory | Checkpoint copied | Workbook |
|---|---|---|---|
| HBTXR_full_unet_img128_patch4 | `analysis/RESULTS/HBTXR_full_unet_img128_patch4` | `checkpoints/epoch=56-val_mean_distance=0.4277.ckpt` | `JETCAS_REPLY_TABLES (Error-Distributions)_HBTXR_full_unet_img128_patch4.xlsx` |
| EPNet_subject_independent_img64 | `analysis/RESULTS/EPNet_subject_independent_img64` | `checkpoints/epoch=57-val_mean_distance=0.5780.ckpt` | `JETCAS_REPLY_TABLES (Error-Distributions)_EPNet_subject_independent_img64.xlsx` |
| TDTracker | `analysis/RESULTS/TDTracker` | `checkpoints/last_checkpoint.pth` | `JETCAS_REPLY_TABLES (Error-Distributions)_TDTracker.xlsx` |

## Weighted Test Summary

| Model | Valid N | Weighted Mean Error | Weighted Median Error | Weighted Mean IoU |
|---|---:|---:|---:|---:|
| HBTXR_full_unet_img128_patch4 | 360,456 | 0.548598 | 0.269351 | 0.610454 |
| EPNet_subject_independent_img64 | 360,255 | 1.467138 | 0.729539 | 0.527995 |
| TDTracker | 358,598 | 6.414729 | 4.427413 | 0.282954 |

## Motion-Wise Weighted Mean Error

| Model | Fixation | Saccade | Smooth |
|---|---:|---:|---:|
| HBTXR_full_unet_img128_patch4 | 0.625025 | 0.515089 | 0.414747 |
| EPNet_subject_independent_img64 | 1.618015 | 1.654855 | 1.196751 |
| TDTracker | 6.827188 | 7.119394 | 5.665703 |

## Notes

- Pixel error is reported in normalized `64x64` input coordinates for cross-model consistency.
- HBTXR/EPNet IoU uses predicted ellipse outputs.
- TDTracker predicts pupil center only, so IoU is `center_proxy_gt_axes_angle`: the ground-truth ellipse axes/angle shifted to the predicted center.
- `HBTXR_full_unet_img128_patch4` was trained on `DeanDataset_full_unet`; this package evaluates it on `DeanDataset_full_unet_subject_independent/test` by dataset-root override for comparability with the requested subject-independent test rows.
- Helper scripts used:
  - `analysis/scripts/fill_facet_ellipse_error_distribution.py`
  - `analysis/scripts/fill_tdtracker_error_distribution.py`
