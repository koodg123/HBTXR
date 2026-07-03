# Retina/ERVT Subject37-48 Error Distribution Spec

Date: 2026-07-03

## Inputs

- Metadata: `analysis/results/HBTXR/HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv`
- Motion label map: `analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv`
- Retina checkpoint: `analysis/results/Retina/checkpoints/epoch=66-val_loss=2.8817.ckpt`
- ERVT checkpoint: `analysis/results/ERVT/checkpoints/best_epoch012_val_distance_8.5869.pth`

## Generated CSV Package

Each model directory contains:

- `<Model>_subject37_48_test_sample_predictions.csv`
- `<Model>_subject37_48_test_predictions_with_metadata.csv`
- `<Model>_subject37_48_test_joined_motion_error.csv`
- `<Model>_subject37_48_unmatched_predictions.csv`
- `<Model>_subject37_48_dropped_blink_predictions.csv`
- `<Model>_subject37_48_joined_motion_counts.csv`
- `<Model>_subject37_48_error_distribution_by_subject_motion.csv`

## Excel Output

Each model has a JETCAS-style workbook:

- `analysis/results/Retina/JETCAS_REPLY_TABLES (Error-Distributions)_Retina_subject37_48.xlsx`
- `analysis/results/ERVT/JETCAS_REPLY_TABLES (Error-Distributions)_ERVT_subject37_48.xlsx`

The combined workbook is:

- `analysis/results/JETCAS_REPLY_TABLES (Error-Distributions)_Retina_ERVT_subject37_48.xlsx`

## Motion Label Rule

The package must not use raw metadata `motion_state` for the final motion distribution. It must use the HBTXR joined motion label map:

`sample_idx -> motion_state, motion_label_speed_pxps`

from `analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv`.

This is required because the raw metadata has near-zero Saccade counts for multiple subjects, while the HBTXR reference Excel uses joined labels.
