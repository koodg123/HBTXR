# Mode0 Stage1 Exp-A vs Exp-D Qualitative Analysis

## Scope

This document isolates the two strongest center-focused Stage1 follow-up runs:

- Exp-A:
  - `runs/mode0_stage1_expA_maskcentroid50_from_ciou100_20260401_044109`
- Exp-D:
  - `runs/mode0_stage1_expD_maskcascade50_from_ciou100_fixeig_20260401_215301`

Both runs start from the same reference checkpoint:

- `runs/mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411/train/best_search_p10.pt`

The goal here is not to repeat the full A/B/C/D table. It is to answer a narrower question:

- if Stage2 must be initialized from one of the mask-guided Stage1 variants, should we prefer the hard centroid route in Exp-A or the soft cascade route in Exp-D?

## Shared Reference Artifacts

The main comparison artifacts are:

- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/comparison_curves_abcd_vs_base.png`
- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/overlay_compare_abcd_best_val8.png`
- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/summary.json`

The shared validation sample indices are:

- `[25, 114, 142, 228, 250, 281, 654, 759]`

## Numeric Summary

### Best Validation Metrics

| Run | Best Epoch | `search_p10` | `search_p5` | `search_center_px` | `eye_iou` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp-A | 5 | `99.63` | `98.89` | `1.82` | `0.7594` |
| Exp-D | 15 | `98.75` | `88.98` | `3.02` | `0.7042` |

### Shared-Overlay Aggregate

| Run | Mean `eye_iou` | Mean `search_center_px` | Mean `mask_iou` |
| --- | ---: | ---: | ---: |
| Exp-A | `0.6272` | `1.79` | `0.6793` |
| Exp-D | `0.6353` | `2.68` | `0.5590` |

Immediate read:

1. Exp-A is better on the center metric both in training logs and in the shared-sample overlay aggregate.
2. Exp-D is still excellent on center, but the gap in `mask_iou` is too large to ignore.
3. Eye-box quality is not the deciding factor here. The real difference is what happens after the eye ROI has been localized.

## Structural Difference

### Exp-A

Mechanism:

- `search xy = mask centroid`
- `search ab/trig/conf = original search head`

Meaning:

- the mask branch directly determines the center
- the search head still owns shape and orientation

Strength:

- center quality becomes almost immediate
- the behavior is simple to reason about

Risk:

- if mask center is wrong, `xy` has no independent correction path

### Exp-D

Mechanism:

- `xy = mask_center + residual_xy`
- `ab = mask_axes * exp(residual_ab * scale)`
- `trig/conf = original search head`

Meaning:

- the mask branch supplies a geometric base state
- the search head becomes a residual corrector instead of a direct predictor

Strength:

- more expressive than Exp-A
- can recover from imperfect mask statistics in principle

Risk:

- shape quality is no longer as stable
- residual learning for `ab` is still noticeably noisy in the current run

## Sample-Level Reading

### Sample 1

- sample:
  - `user33__right__session_102__000210_1658286177931888`
- Exp-A:
  - `search_center_px = 0.98`
  - `mask_iou = 0.742`
- Exp-D:
  - `search_center_px = 4.97`
  - `mask_iou = 0.581`

Reading:

- both are usable
- Exp-A looks more settled
- Exp-D still lands near the pupil, but the region prior is visibly rougher

### Sample 2

- sample:
  - `user34__left__session_201__001410_1658287279851893`
- Exp-A:
  - `search_center_px = 2.07`
  - `mask_iou = 0.591`
- Exp-D:
  - `search_center_px = 4.40`
  - `mask_iou = 0.301`

Reading:

- this is the most cautionary example for Exp-D
- center remains acceptable, but the mask and shape prior degrade substantially

### Sample 3

- sample:
  - `user36__left__session_102__002210_1658302617497420`
- Exp-A:
  - `search_center_px = 2.04`
  - `mask_iou = 0.698`
- Exp-D:
  - `search_center_px = 0.86`
  - `mask_iou = 0.595`

Reading:

- this is the kind of sample that keeps Exp-D interesting
- it can beat Exp-A on center once the residual cascade locks in
- but it still does not produce the same level of shape support

## Why Exp-D Is Not Yet The Default

The key issue is not center.

The key issue is that Exp-D still looks too expensive in `ab`.

At best `search_p10`:

- Exp-A:
  - `loss_search_xy = 0.186`
  - `loss_search_ab = 1.306`
- Exp-D:
  - `loss_search_xy = 0.377`
  - `loss_search_ab = 5.854`

Interpretation:

1. Exp-D does solve the center problem.
2. But it does so while making ellipse-size prediction much less reliable.
3. For Stage2 bootstrap, that matters because Stage2 should inherit not only center alignment but also a usable search-state geometry prior.

## Stage2 Initialization Decision

### Primary recommendation

Use:

- `runs/mode0_stage1_expA_maskcentroid50_from_ciou100_20260401_044109/train/best_search_p10.pt`

Why:

1. It is the strongest checkpoint on the Stage1 metric that we actually optimize.
2. It is cleaner than Exp-D on the shared overlay sample set.
3. It keeps `ab` in a materially healthier range than Exp-D.

Important requirement:

- if this checkpoint is used, Stage2 should keep:
  - `model.search.xy_from_mask_centroid: true`

This is not optional bookkeeping. It is part of what the checkpoint has learned to rely on.

### Secondary recommendation

If we want a more conservative path later, the fallback is still:

- `runs/mode0_stage1_expB_maskcenter50_from_ciou100_20260401_054827/train/best_search_p10.pt`

This is easier to interpret than either Exp-A or Exp-D, but it is not the strongest raw initializer.

### Current non-recommendation

Do not use Exp-D as the default Stage2 initializer yet.

Reason:

1. Its center metrics are extremely attractive.
2. But the current evidence suggests that it pushes too much instability into `ab`.
3. That makes it a promising research branch, not yet the safest bootstrap checkpoint.

## Final Takeaway

If the question is:

- which Stage1 checkpoint should drive the next Stage2 run?

The answer is:

- Exp-A first

If the question is:

- which branch is more interesting to keep researching for future Stage1 design?

The answer is:

- Exp-D

That split captures the current state well:

- Exp-A is the practical winner
- Exp-D is the more ambitious but not yet fully controlled variant
