# HBTXR Val Motion Evaluation Report

## Scope

- Model: `HBTXR_full_unet_img64_patch4`
- Config: `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet_img64_patch4.yaml`
- Checkpoint: `references/codebase/software/FACET/runs/logs/HBTXR_full_unet_img64_patch4/version_0/checkpoints/epoch=67-val_mean_distance=0.4492.ckpt`
- Evaluation split: `DeanDataset_full_unet/val` held-out validation split, not a final test split.
- Motion state: velocity-based `Fixation`, `Saccade`, `Smooth`; repeated-run CI section is intentionally excluded.

## Motion-State Rule

- `Saccade`: pseudo-label pupil-center speed > 493 px/s.
- `Fixation`: speed <= 493 px/s and session code is `101` or `201`.
- `Smooth`: speed <= 493 px/s and session code is `102` or `202`.
- Session mapping follows `subject-motion-analysis`: `101/201` are saccade-fixation regime, `102/202` are smooth-pursuit regime.
- Velocity is computed from dense U-Net pseudo-label centers in `DeanDataset_full_unet`, not from Tobii or repeated manual annotations.

## Overall Summary

- Total val samples: 292,560
- Valid error samples: 291,889
- Overall center error: mean 1.795 px, median 0.773 px, P95 7.563 px, P99 18.022 px in input64 coordinates.
- Overall IoU: mean 0.486, median 0.519.

## (1) Subject-wise Pixel Error / IoU Distribution

- Table: `hbtxr_val_subject_error_iou_stats.csv`
- Figures: `fig_subject_pixel_error_box.png`, `fig_subject_iou_box.png`
- Median subject error range: 0.561 to 1.355 px.
- Median subject IoU range: 0.444 to 0.588.
- Interpretation: subject-wise spread is visible in the tail metrics; subjects with high P95/P99 should be inspected for pseudo-label quality, blink/occlusion, or motion imbalance.

## (2) Subject-wise Motion Distribution

- Table: `hbtxr_val_subject_motion_counts.csv`
- Figure: `fig_subject_motion_counts.png`
- Fixation: 187,308 samples (64.02%).
- Saccade: 58 samples (0.02%).
- Smooth: 105,194 samples (35.96%).
- Interpretation: the velocity rule gives a dense 3-state split for the HBTXR pseudo-label val set. Saccade is expected to be a minority class because high-speed movements are brief.
- Caution: the Saccade group is very small in this val split, so Saccade error statistics should be treated as descriptive rather than conclusive.

## (3) Subject-wise Mean / Median / P95 / P99 Pixel Error

- Table: `hbtxr_val_subject_pixel_error_stats.csv`
- Figure: `fig_subject_pixel_error_box.png`
- Highest-median subjects:
  - user09: median 1.355 px, P95 18.359 px, P99 25.405 px, n=30,940
  - user08: median 0.954 px, P95 9.566 px, P99 17.347 px, n=29,374
  - user06: median 0.925 px, P95 5.101 px, P99 13.370 px, n=30,524
  - user45: median 0.900 px, P95 4.400 px, P99 12.924 px, n=30,593
  - user47: median 0.735 px, P95 12.557 px, P99 31.733 px, n=30,709
- Interpretation: median captures typical localization quality, while P95/P99 exposes rare tracking or label failures. Report both because mean alone hides tail behavior.

## (4) Motion-wise Mean / Median / P95 / P99 Pixel Error

- Table: `hbtxr_val_motion_pixel_error_stats.csv`
- Figure: `fig_motion_pixel_error_box.png`

| motion | n | mean | median | P95 | P99 | max |
|---|---:|---:|---:|---:|---:|---:|
| Fixation | 186,810 | 1.852 | 0.839 | 7.664 | 15.996 | 43.880 |
| Saccade | 58 | 0.533 | 0.534 | 1.008 | 1.403 | 1.406 |
| Smooth | 105,021 | 1.695 | 0.678 | 6.862 | 21.236 | 43.402 |
| All | 291,889 | 1.795 | 0.773 | 7.563 | 18.022 | 43.880 |

- Interpretation: compare `Saccade` against `Fixation` and `Smooth` primarily via median and P95/P99. If Saccade has a larger tail, this supports the reviewer-facing statement that fast motion is harder.

## (7) Annotation Precision, Label Noise

- Table: `hbtxr_label_precision_floor.csv`
- Table: `hbtxr_pseudolabel_noise_stats.csv`
- Figure: `fig_label_uncertainty_overlay.png`
- Manual GT integer quantization floor: mean 0.383 px, median 0.399 px, P95 0.599 px.
- HBTXR val pseudo-label center noise on manual-GT matched frames: mean 0.689 px, median 0.626 px, P95 1.395 px, n=1,825.
- Interpretation: HBTXR was trained/evaluated against U-Net pseudo-labels in `DeanDataset_full_unet`. Therefore, label uncertainty has two parts: manual annotation quantization floor and pseudo-label generation noise. If reported HBTXR errors are near these values, sub-pixel differences should be interpreted cautiously.

## Files

- merged: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_val_predictions_with_metadata.csv`
- subject_stats: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_val_subject_error_iou_stats.csv`
- motion_counts: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_val_subject_motion_counts.csv`
- subject_pixel: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_val_subject_pixel_error_stats.csv`
- motion_pixel: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_val_motion_pixel_error_stats.csv`
- label_precision: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_label_precision_floor.csv`
- label_noise: `references/report/FACET/HBTXR_val_motion_eval/hbtxr_pseudolabel_noise_stats.csv`
- figures: `references/report/FACET/HBTXR_val_motion_eval/figures/`
