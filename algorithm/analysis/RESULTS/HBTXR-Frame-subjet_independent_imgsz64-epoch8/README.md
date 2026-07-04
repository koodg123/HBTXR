# HBTXR-Frame-subjet_independent_imgsz64-epoch8

## Package Scope

This package collects the epoch-8 HBTXR frame-model training and validation artifacts for the subject-independent EV-Eye split.

- Model family: HBTXR frame input variant
- Input representation: cached frame, nominal image size 64 in the requested package name
- Training stop point: epoch 8
- Training data split: `/mnt/e/DATASET/eveye/DeanDataset_full_unet_subject_independent`
- Package created under: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8`

## Included Artifacts

- `checkpoints/epoch8-final-08-00544995.ckpt`: final epoch-8 checkpoint
- `checkpoints/last.ckpt`: Lightning last checkpoint after epoch 8
- `configs/DavisEyeEllipse_HBTXR_frame_cached_si_img128_patch4_epoch8_stop.yaml`: training config used to stop at epoch 8
- `configs/DavisEyeEllipse_HBTXR_frame_cached_si_img128_patch4_epoch8_eval.yaml`: validation config using the epoch-8 checkpoint
- `logs/HBTXR_frame_cached_si_img128_patch4_epoch8_stop_train_20260703.log`: training log
- `logs/HBTXR_frame_cached_si_img128_patch4_epoch8_eval_20260703.log`: validation log
- `metrics/validation_metrics.json`: parsed validation summary
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_sample_metadata.csv`: full test sample metadata
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_sample_predictions.csv`: full test predictions
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_predictions_with_metadata.csv`: full test prediction-metadata join
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_test_predictions_with_metadata.csv`: subject37-48 prediction-metadata join
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_test_joined_motion_error.csv`: subject37-48 non-Blink motion/error join
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_error_distribution_by_subject_motion.csv`: subject/motion error table
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_joined_motion_counts.csv`: subject/motion count table
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_dropped_blink_predictions.csv`: rows excluded by non-Blink join
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_unmatched_predictions.csv`: same left-only rows retained for compatibility with existing package style
- `HBTXR-Frame-subjet_independent_imgsz64-epoch8_subject37_48_error_distribution_report.md`: subject37-48 summary report
- `scripts/`: reproduction scripts used for inference and subject37-48 CSV packaging
- `MANIFEST.sha256`: SHA-256 checksums for packaged files

## Validation Summary

Validation was run on the validation split after epoch 8.

| Metric | Value |
|---|---:|
| val_mean_distance | 0.08000335097312927 |
| val_IoU | 0.6456773281097412 |
| val_AP | 0.8394579291343689 |
| val_loss | 4.440107345581055 |
| val_p1_acc | 0.9999918937683105 |
| val_p3_acc | 1.0 |
| val_p5_acc | 1.0 |
| val_p10_acc | 1.0 |

## Subject37-48 Test Summary

- Prediction rows: 366,171
- Joined non-Blink rows: 360,495
- Dropped/left-only rows: 5,676
- Valid error rows: 360,456
- Weighted mean pixel error: 0.284478
- Weighted mean IoU: 0.654663
- Non-Blink sample selection reuses the sample-index set from the existing `HBTXR_subject37_48_test_joined_motion_error.csv`, because the original `RowLabels_test37_48_motion_NoBlink.csv` file is not present in this workspace.
- The requested package name says `imgsz64`; the epoch-8 frame config stored in `configs/` uses `default_resolution: [128, 128]`, `img_size: 128`, and `patch_size: 4`. CSV column names retain `error_input64_px` for compatibility with the existing HBTXR result folders.
