# Model Inference Package Summary - 2026-07-03

## Scope

Generated Subject-Independent Test result packages under `analysis/RESULTS` for:

- `EIDet`
- `BRAT`

The package layout follows `analysis/RESULTS/HBTXR_full_unet_img128_patch4`: subject/motion summary CSVs, error-distribution workbooks, Markdown reports, and copied config provenance files. Large row-level prediction and joined-error CSVs are kept locally but ignored for GitHub push.

## Package Outputs

| Model | Output directory | Checkpoint copied | Workbook |
|---|---|---|---|
| EIDet | `analysis/RESULTS/EIDet` | `checkpoints/epoch=01-val_mean_distance=1.1483.ckpt` | `JETCAS_REPLY_TABLES (Error-Distributions)_EIDet.xlsx` |
| BRAT | `analysis/RESULTS/BRAT` | `checkpoints/perror_0.5878_tsf1.0.pth` | `JETCAS_REPLY_TABLES (Error-Distributions)_BRAT.xlsx` |

## Weighted Test Summary

| Model | Valid N | Weighted Mean Error | Weighted Median Error | Weighted Mean IoU |
|---|---:|---:|---:|---:|
| EIDet | 360,255 | 3.849677 | 2.386649 | 0.357419 |
| BRAT | 360,495 | 0.879130 | 0.477422 | 0.770778 |

## Motion-Wise Weighted Mean Error

| Model | Fixation | Saccade | Smooth |
|---|---:|---:|---:|
| EIDet | 4.032597 | 4.304071 | 3.516293 |
| BRAT | 1.001052 | 0.767060 | 0.667041 |

## Notes

- Pixel error is reported in normalized `64x64` input coordinates.
- EIDet is the local FACET `ElNet` implementation packaged under the corrected target name `EIDet`.
- EIDet predicts ellipse outputs, so IoU uses predicted ellipse shape.
- BRAT predicts pupil center only, so IoU is `center_proxy_gt_axes_angle`: the ground-truth ellipse axes/angle shifted to the predicted center.
- BRAT was re-run on the same Subject 37-48 full test set used by EIDet. The corrected BRAT `test_files.txt` contains 96 unique eye-session entries with explicit `L`/`R` eye tokens, and the raw prediction file contains 366,171 rows.
- Large row-level files such as `*_test_predictions_with_metadata.csv`, `*_test_joined_motion_error.csv`, `*_test_sample_predictions.csv`, `*_test_sample_metadata.csv`, and `*_raw_test_predictions.csv` are excluded from git tracking because they exceed GitHub's recommended or hard file-size limits.
- Blink rows are excluded by joining against `analysis/motion-label-packages/outputs/RowLabels_test37_48_motion_NoBlink.csv`.
- Helper scripts used:
  - `analysis/scripts/fill_facet_ellipse_error_distribution.py`
  - `analysis/scripts/fill_brat_error_distribution.py`
