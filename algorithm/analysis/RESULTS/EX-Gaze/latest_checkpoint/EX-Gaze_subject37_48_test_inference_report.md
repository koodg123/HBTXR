# EX-Gaze Subject-Independent Test Error Distribution

## Scope

- Model: `EX-Gaze`
- Checkpoint: `checkpoints/EX-Gaze_epoch043_latest_last_snapshot.ckpt`
- Checkpoint epoch: `43`
- Split: `test`, subjects `37-48`.
- Blink rows are excluded by joining against `RowLabels_test37_48_motion_NoBlink.csv`.
- Pixel error is in `64x64` input coordinates.
- IoU is from the predicted ellipse parameters rendered on the 128 grid when available; center error remains reported in 64 grid coordinates.

## Summary

- Prediction rows: 366,075
- Joined non-Blink rows: 360,399
- Valid error rows used in statistics: 360,399
- Unmatched prediction rows: 5,676
- Dropped blink rows: 5,676
- Mean pixel error: 0.280069
- Median pixel error: 0.092996
- P95 pixel error: 1.266235
- P99 pixel error: 2.595933
- Mean IoU: n/a
- Workbook: `JETCAS_REPLY_TABLES (Error-Distributions)_EX-Gaze.xlsx`

## Files

- `EX-Gaze_latest_last_snapshot.ckpt`
- `EX-Gaze_subject37_48_dropped_blink_predictions.csv`
- `EX-Gaze_subject37_48_error_distribution_by_subject_motion.csv`
- `EX-Gaze_subject37_48_inference_metadata.json`
- `EX-Gaze_subject37_48_joined_motion_counts.csv`
- `EX-Gaze_subject37_48_motion_error_stats.csv`
- `EX-Gaze_subject37_48_subject_error_stats.csv`
- `EX-Gaze_subject37_48_test_inference_report.md`
- `EX-Gaze_subject37_48_test_joined_motion_error.csv`
- `EX-Gaze_subject37_48_test_sample_predictions.csv`
- `EX-Gaze_subject37_48_unmatched_predictions.csv`
- `EX-Gaze_test_inference_metadata.json`
- `JETCAS_REPLY_TABLES (Error-Distributions)_EX-Gaze.xlsx`
- `checkpoints/EX-Gaze_epoch043_latest_last_snapshot.ckpt`
