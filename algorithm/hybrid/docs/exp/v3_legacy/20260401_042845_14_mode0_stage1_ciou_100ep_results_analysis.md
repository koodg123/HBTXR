# Mode0 Stage1 CIoU 100-Epoch Results

## Scope

This document records the `mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411` run and summarizes:

- the final `100`-epoch Stage1 result with `sensor_full_letterbox`
- `dense eye + search + mask`
- `search_xy_weight=0.25`
- `search_ciou_weight=0.25`
- `search_geo_weight=0.0`
- `cosine` scheduler
- `early_stopping=false`

Run root:

- `runs/mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411`

Main artifacts:

- `train/history.json`
- `train/best.pt`
- `train/best_search_p10.pt`
- `analysis/loss_curves.png`
- `analysis/metric_curves.png`
- `analysis/summary.json`
- `analysis/overlay_best_search_p10_val8/contact_sheet.png`
- `analysis/overlay_best_search_p10_val8/summary.json`

## Run Summary

The run completed all `100` epochs without divergence.

Best validation metrics:

| Metric | Best Epoch | Value |
| --- | ---: | ---: |
| `metric_search_p10_pct` | 71 | `22.29` |
| `metric_search_p5_pct` | 55 | `6.42` |
| `metric_search_center_px` | 73 | `17.28` |
| `metric_eye_iou` | 39 | `0.7446` |
| `metric_eye_center_px` | 41 | `13.15` |
| `loss_total` | 78 | `5.0635` |

Final epoch (`100`) validation metrics:

| Metric | Value |
| --- | ---: |
| `loss_total` | `5.1953` |
| `loss_eye` | `0.3080` |
| `loss_mask` | `0.0452` |
| `loss_search_xy` | `2.7372` |
| `loss_search_ab` | `1.3326` |
| `loss_search_trig` | `0.2858` |
| `loss_search_ciou` | `0.1650` |
| `metric_eye_iou` | `0.7286` |
| `metric_eye_center_px` | `14.04` |
| `metric_search_center_px` | `17.96` |
| `metric_search_p10_pct` | `21.53` |
| `metric_search_p5_pct` | `4.92` |

## Comparison To Previous Tuned Stage1

Reference run:

- `runs/mode0_stage1_dense_eye30_tuned_nogeo_20260330_193647`

Best validation comparison:

| Metric | Previous Tuned Run | CIoU 100-Epoch Run |
| --- | ---: | ---: |
| `metric_search_p10_pct` | `11.46` | `22.29` |
| `metric_search_center_px` | `27.41` | `17.28` |
| `metric_eye_iou` | `0.7281` | `0.7446` |

Interpretation:

- the new run materially improved Stage1 pupil search quality
- `100`-epoch full cosine training was beneficial
- disabling early stopping was the correct choice for this recipe because the best `search` metrics arrived late (`epoch 71-73`)

## Loss-Curve Interpretation

Plots:

- `analysis/loss_curves.png`
- `analysis/metric_curves.png`

Observed pattern:

1. Training stayed numerically stable for the full `100` epochs.
2. There was no late `mask` or distillation-style blow-up.
3. `eye` and `mask` converged relatively early.
4. `search` kept improving slowly until the late middle of training.

This is materially different from the earlier unstable `distillation + pruning` run, where auxiliary losses eventually dominated optimization.

## What Still Trains Poorly

Even in the improved run, the dominant validation loss terms remained:

- `loss_search_xy`
- `loss_search_ab`

At the best `search_p10` epoch (`71`), validation loss fractions were:

| Loss Term | Fraction of `loss_total` |
| --- | ---: |
| `loss_search_xy` | `52.25%` |
| `loss_search_ab` | `26.58%` |
| `loss_search_trig` | `5.77%` |
| `loss_search_ciou` | `3.18%` |
| `loss_constraint_center` | `5.50%` |
| `loss_eye` | `5.86%` |
| `loss_mask` | `0.86%` |

Interpretation:

- the main bottleneck is still pupil center regression (`xy`)
- the second bottleneck is pupil size regression (`ab`)
- `CIoU` is active and stable, but it is not the dominant optimization driver
- `eye` and `mask` are no longer the main failure points in this recipe

## Generalization Gap

Final epoch train vs validation:

| Metric | Train | Val |
| --- | ---: | ---: |
| `metric_search_center_px` | `2.38` | `17.96` |
| `metric_search_p10_pct` | `99.04` | `21.53` |
| `metric_eye_iou` | `0.9495` | `0.7286` |

Interpretation:

- optimization itself is no longer failing
- the remaining problem is generalization, especially on `search/pupil`
- the biggest remaining gap is between train and validation pupil center quality

## Best-Checkpoint Overlay Analysis

Checkpoint used:

- `train/best_search_p10.pt`

Overlay artifact:

- `analysis/overlay_best_search_p10_val8/contact_sheet.png`

Overlay selection:

- split: `val`
- random seed: `42`
- number of samples: `8`

Overlay drawing:

- ground-truth eye box: green
- predicted eye box: red
- ground-truth pupil ellipse: cyan
- predicted pupil ellipse: yellow
- predicted pupil mask fill: red transparent overlay
- ground-truth mask points: green

Random-8 aggregate summary:

| Metric | Mean |
| --- | ---: |
| `eye_iou` | `0.6693` |
| `eye_center_px` | `18.08` |
| `search_center_px` | `18.00` |
| `mask_iou` | `0.6472` |

Interesting sample-level cases:

1. `user33__right__session_102__000210_1658286177931888`
   - `eye_iou=0.640`
   - `search_center_px=7.31`
   - `mask_iou=0.773`
   - This is a relatively balanced success case.

2. `user36__right__session_102__002210_1658302612348138`
   - `eye_iou=0.932`
   - `search_center_px=26.93`
   - `mask_iou=0.652`
   - Eye ROI localization is excellent, but pupil center is still poor.

3. `user36__left__session_102__002210_1658302617497420`
   - `eye_iou=0.819`
   - `search_center_px=23.51`
   - `mask_iou=0.871`
   - Even strong eye and mask quality does not guarantee strong pupil center regression.

4. `user34__left__session_201__001410_1658287279851893`
   - `eye_iou=0.561`
   - `search_center_px=4.33`
   - `mask_iou=0.417`
   - Pupil center can still be good even when auxiliary heads are weaker.

These random samples reinforce an important point:

- `eye` quality and `mask` quality help, but they do not fully determine `search` quality
- the remaining Stage1 bottleneck is inside the pupil search branch itself, especially center and size regression

## Current Decision

This run is the strongest stable Stage1 result so far.

Recommended Stage1 checkpoint:

- `train/best_search_p10.pt` at `epoch 71`

Current working interpretation:

- keep `sensor_full_letterbox`
- keep `dense eye + search + mask`
- keep `search_geo_weight=0.0`
- keep long `cosine` training without early stopping for this recipe
- next optimization target should focus on improving `search_xy` and `search_ab` generalization
