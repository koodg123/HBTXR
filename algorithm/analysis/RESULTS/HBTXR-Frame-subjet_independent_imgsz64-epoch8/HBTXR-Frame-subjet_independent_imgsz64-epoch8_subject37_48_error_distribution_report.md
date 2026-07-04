# HBTXR-Frame-subjet_independent_imgsz64-epoch8 Subject-Independent Test Error Distribution

## Scope

- Model: `HBTXR-Frame-subjet_independent_imgsz64-epoch8`
- Config: `configs/DavisEyeEllipse_HBTXR_frame_cached_si_img128_patch4_epoch8_eval.yaml`
- Checkpoint: `checkpoints/epoch8-final-08-00544995.ckpt`
- Split: `test`, subjects `37-48`.
- Blink rows are excluded by reusing the non-Blink sample index set from `HBTXR_subject37_48_test_joined_motion_error.csv`.
- Error column name is kept as `error_input64_px` for compatibility with existing tables; this package was generated from the epoch-8 frame config stored in `configs/`.

## Summary

- Subject37-48 prediction rows: 366,171
- Joined non-Blink rows: 360,495
- Dropped/left-only rows: 5,676
- Valid error rows used in statistics: 360,456
- Weighted mean pixel error: 0.2845
- Weighted mean IoU: 0.6547

## Files

- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_error_distribution_by_subject_motion.csv`
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_test_joined_motion_error.csv`
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_joined_motion_counts.csv`
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_dropped_blink_predictions.csv`
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_unmatched_predictions.csv`
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_test_predictions_with_metadata.csv`
