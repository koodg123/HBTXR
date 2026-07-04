# HBTXR-Frame-subjet_independent_imgsz64-epoch8 TEST Motion Evaluation Report

## Scope

- Model: `HBTXR-Frame-subjet_independent_imgsz64-epoch8`
- Config: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/configs/DavisEyeEllipse_HBTXR_frame_cached_si_img128_patch4_epoch8_eval.yaml`
- Checkpoint: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/checkpoints/epoch8-final-08-00544995.ckpt`
- Evaluation split: `DeanDataset_full_unet_subject_independent/test`.
- Motion state: velocity-based `Fixation`, `Saccade`, `Smooth`; repeated-run CI section is intentionally excluded.

## Motion-State Rule

- `Saccade`: pseudo-label pupil-center speed > 493 px/s.
- `Fixation`: speed <= 493 px/s and session code is `101` or `201`.
- `Smooth`: speed <= 493 px/s and session code is `102` or `202`.
- Session mapping follows `subject-motion-analysis`: `101/201` are saccade-fixation regime, `102/202` are smooth-pursuit regime.
- Velocity is computed from dense U-Net pseudo-label centers in `DeanDataset_full_unet`, not from Tobii or repeated manual annotations.

## Overall Summary

- Total test samples: 366,171
- Valid error samples: 366,130
- Overall center error: mean 0.285 px, median 0.252 px, P95 0.616 px, P99 0.837 px in input64 coordinates.
- Overall IoU: mean 0.654, median 0.670.

## (1) Subject-wise Pixel Error / IoU Distribution

- Table: `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_subject_error_iou_stats.csv`
- Figures: `fig_subject_pixel_error_box.png`, `fig_subject_iou_box.png`
- Median subject error range: 0.199 to 0.384 px.
- Median subject IoU range: 0.649 to 0.709.
- Interpretation: subject-wise spread is visible in the tail metrics; subjects with high P95/P99 should be inspected for pseudo-label quality, blink/occlusion, or motion imbalance.

## (2) Subject-wise Motion Distribution

- Table: `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_subject_motion_counts.csv`
- Figure: `fig_subject_motion_counts.png`
- Fixation: 234,799 samples (64.12%).
- Saccade: 55 samples (0.02%).
- Smooth: 131,317 samples (35.86%).
- Interpretation: the velocity rule gives a dense 3-state split for the HBTXR pseudo-label val set. Saccade is expected to be a minority class because high-speed movements are brief.
- Caution: the Saccade group is very small in this val split, so Saccade error statistics should be treated as descriptive rather than conclusive.

## (3) Subject-wise Mean / Median / P95 / P99 Pixel Error

- Table: `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_subject_pixel_error_stats.csv`
- Figure: `fig_subject_pixel_error_box.png`
- Highest-median subjects:
  - user41: median 0.384 px, P95 0.841 px, P99 1.175 px, n=30,684
  - user45: median 0.333 px, P95 0.685 px, P99 0.873 px, n=30,606
  - user40: median 0.309 px, P95 0.643 px, P99 0.809 px, n=30,754
  - user42: median 0.300 px, P95 0.669 px, P99 0.849 px, n=29,737
  - user39: median 0.251 px, P95 0.564 px, P99 0.771 px, n=30,366
- Interpretation: median captures typical localization quality, while P95/P99 exposes rare tracking or label failures. Report both because mean alone hides tail behavior.

## (4) Motion-wise Mean / Median / P95 / P99 Pixel Error

- Table: `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_motion_pixel_error_stats.csv`
- Figure: `fig_motion_pixel_error_box.png`

| motion | n | mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Fixation | 234,774 | 0.282 | 0.249 | 0.613 | 0.832 | 3.253 |
| Saccade | 55 | 0.403 | 0.317 | 0.888 | 1.221 | 1.494 |
| Smooth | 131,301 | 0.289 | 0.257 | 0.622 | 0.848 | 2.504 |
| All | 366,130 | 0.285 | 0.252 | 0.616 | 0.837 | 3.253 |

- Interpretation: compare `Saccade` against `Fixation` and `Smooth` primarily via median and P95/P99. If Saccade has a larger tail, this supports the reviewer-facing statement that fast motion is harder.

## (7) Annotation Precision, Label Noise

- Table: `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_label_precision_floor.csv`
- Table: `HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_pseudolabel_noise_stats.csv`
- Figure: `fig_label_uncertainty_overlay.png`
- Manual GT integer quantization floor: mean 0.383 px, median 0.399 px, P95 0.599 px.
- No manual-GT matched pseudo-label samples were found in the val split.
- Interpretation: HBTXR was trained/evaluated against U-Net pseudo-labels in `DeanDataset_full_unet`. Therefore, label uncertainty has two parts: manual annotation quantization floor and pseudo-label generation noise. If reported HBTXR errors are near these values, sub-pixel differences should be interpreted cautiously.

## Files

- merged: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_predictions_with_metadata.csv`
- subject_stats: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_subject_error_iou_stats.csv`
- motion_counts: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_subject_motion_counts.csv`
- subject_pixel: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_subject_pixel_error_stats.csv`
- motion_pixel: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_motion_pixel_error_stats.csv`
- label_precision: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_label_precision_floor.csv`
- label_noise: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/HBTXR-Frame-subjet_independent_imgsz64-epoch8_test_pseudolabel_noise_stats.csv`
- figures: `/home/user/project/PRJXR-HBTXR/HBTXR/analysis/results/HBTXR-Frame-subjet_independent_imgsz64-epoch8/figures`

