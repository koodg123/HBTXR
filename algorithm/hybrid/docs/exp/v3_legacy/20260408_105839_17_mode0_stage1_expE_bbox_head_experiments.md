# Mode0 Stage1 Exp-E BBox-Head Experiment Analysis

## Scope

This document records the `Exp-E` family of Stage1 experiments that replace the standard pupil-state head with a two-stage bbox cascade:

- stage-1 eye-region bbox prediction
- stage-2 pupil-region bbox prediction

The purpose of `Exp-E` was to test whether a bbox-first cascade can provide a cleaner bridge between eye ROI localization and pupil localization than the existing `search/state + mask` formulation.

## Target Structure

All `Exp-E` variants supervise two bbox outputs:

1. `search/eye`
   - target: `eye_target`
   - meaning: eye-region ROI bbox
2. `search/pupil_bbox`
   - target: `pupil_region_target`
   - meaning: pupil-region bbox

Loss composition:

- `loss_eye`
  - `SmoothL1` on `[cx, cy, w, h]`
  - `BCEWithLogits` on confidence
- `loss_pupil_bbox`
  - `SmoothL1` on `[cx, cy, w, h]`
  - `BCEWithLogits` on confidence

For these runs:

- standard `search_xy/search_ab/search_trig/search_ciou/search_conf` losses were disabled
- mask, constraint-center, and aux losses were disabled

So the effective total loss was:

- `loss_total = loss_eye + loss_pupil_bbox`

## Run Inventory

| Run | Main structure | Main idea | Outcome |
| --- | --- | --- | --- |
| `mode0_stage1_expE_roiattn_bbox50_gpu1_nw4_20260402_023825` | `MLP0 -> ROI Align -> Self-Attention -> MLP1` | use predicted eye box to crop ROI tokens and predict pupil bbox | failed to learn eye bbox |
| `mode0_stage1_expE_no_roiattn_bbox50_gpu1_nw4_20260402_084118` | `MLP0 -> eye-conditioned token -> Self-Attention -> MLP1` | remove ROI Align and keep full tokens | slightly better optimization, but eye bbox still failed |
| `mode0_stage1_expE_dense_eye_no_roiattn_bbox50_gpu01_nw8_20260403_010037` | `DenseEye -> eye-conditioned token -> Self-Attention -> MLP1` | reuse `Exp-A` dense eye detector as the eye ROI provider | first `Exp-E` variant that learned eye bbox properly |

## Variant A: ROI Align Cascade

Run:

- `runs/mode0_stage1_expE_roiattn_bbox50_gpu1_nw4_20260402_023825`

Best validation metrics:

- best `metric_eye_iou`: `0.0`
- best `metric_eye_center_px`: `183.004516`
- best `metric_pupil_bbox_iou`: `0.059949` at epoch `17`
- best `metric_pupil_bbox_center_px`: `32.073877` at epoch `30`
- best `loss_total`: `141.661330` at epoch `49`

Final validation metrics:

- `loss_eye`: `128.772098`
- `loss_pupil_bbox`: `12.892338`
- `loss_total`: `141.664435`
- `metric_eye_center_px`: `183.004516`
- `metric_eye_iou`: `0.0`
- `metric_pupil_bbox_center_px`: `32.584923`
- `metric_pupil_bbox_iou`: `0.055324`

Interpretation:

- the eye-stage MLP never learned anything useful
- because ROI Align depended on that failed eye box, the pupil-stage branch never received a good spatial crop
- the pupil bbox branch learned only a very weak fallback solution

Conclusion:

- this version is effectively a failure
- the main bottleneck is the first-stage eye bbox predictor

## Variant B: No-ROIAlign Cascade

Run:

- `runs/mode0_stage1_expE_no_roiattn_bbox50_gpu1_nw4_20260402_084118`

Best validation metrics:

- best `metric_eye_iou`: `0.0`
- best `metric_eye_center_px`: `132.896076` at epoch `21`
- best `metric_pupil_bbox_iou`: `0.059056` at epoch `14`
- best `metric_pupil_bbox_center_px`: `32.034957` at epoch `10`
- best `loss_total`: `112.988076` at epoch `21`

Final validation metrics:

- `loss_eye`: `100.467345`
- `loss_pupil_bbox`: `12.880743`
- `loss_total`: `113.348088`
- `metric_eye_center_px`: `133.205294`
- `metric_eye_iou`: `0.0`
- `metric_pupil_bbox_center_px`: `32.135295`
- `metric_pupil_bbox_iou`: `0.058653`

Interpretation:

- removing ROI Align made optimization easier
- total loss dropped substantially
- eye-center error improved from `183` to about `133`
- but `metric_eye_iou` still remained `0.0`, so the eye stage was still not learning usable boxes
- pupil bbox quality remained essentially unchanged

Conclusion:

- ROI Align itself was not the primary failure
- the real problem remained the weak first-stage eye predictor

## Variant C: Dense-Eye Guided Cascade

Run:

- `runs/mode0_stage1_expE_dense_eye_no_roiattn_bbox50_gpu01_nw8_20260403_010037`

Structure:

- use the `Exp-A` style dense eye detector as the stage-1 eye ROI provider
- convert the predicted eye box into an eye-conditioned token
- run self-attention over `[eye token + frame tokens]`
- predict `search/pupil_bbox` from the attended eye token

Best validation metrics:

- best `metric_eye_iou`: `0.731098` at epoch `29`
- best `metric_eye_center_px`: `14.358903` at epoch `29`
- best `metric_pupil_bbox_iou`: `0.100132` at epoch `46`
- best `metric_pupil_bbox_center_px`: `27.177097` at epoch `27`
- best `loss_eye`: `0.289296` at epoch `17`
- best `loss_pupil_bbox`: `11.197239` at epoch `46`
- best `loss_total`: `11.501235` at epoch `46`

Final validation metrics:

- `loss_eye`: `0.307584`
- `loss_pupil_bbox`: `11.350132`
- `loss_total`: `11.657716`
- `metric_eye_center_px`: `16.301368`
- `metric_eye_iou`: `0.710300`
- `metric_pupil_bbox_center_px`: `28.251786`
- `metric_pupil_bbox_iou`: `0.092451`

Interpretation:

- this is the first `Exp-E` variant where the eye stage clearly works
- replacing the failed `MLP0` eye predictor with a strong dense-eye provider solved the main upstream collapse
- once the eye ROI became usable, the pupil bbox branch improved materially
- however, `pupil_bbox_iou ≈ 0.10` is still far from a strong standalone localization result

Conclusion:

- the `Exp-E` idea is viable only when its eye-stage provider is strong
- dense eye turns the experiment from "failed cascade" into "usable but still weak pupil-refinement head"

## Cross-Variant Comparison

| Metric | ROI Align | No ROI Align | Dense Eye + No ROI Align |
| --- | ---: | ---: | ---: |
| best `eye_iou` | `0.0` | `0.0` | `0.7311` |
| best `eye_center_px` | `183.00` | `132.90` | `14.36` |
| best `pupil_bbox_iou` | `0.05995` | `0.05906` | `0.10013` |
| best `pupil_bbox_center_px` | `32.07` | `32.03` | `27.18` |
| best `loss_total` | `141.66` | `112.99` | `11.50` |

This table shows the main story clearly:

1. changing ROI Align to full-token attention did not solve the problem
2. dense eye supervision did solve the eye-stage problem
3. once the eye stage worked, pupil bbox quality improved
4. the remaining bottleneck is now the pupil bbox head itself, not the eye stage

## What We Learned

### 1. The original `Exp-E` design failed because `MLP0` was too weak

The first-stage eye ROI prediction did not train successfully under either:

- ROI Align cascade
- no-ROIAlign cascade

So the two-step structure never had a reliable first stage to build on.

### 2. The eye pseudo-label is not enough for a weak eye head

`mode0` eye supervision is heuristic pseudo-GT rather than manual eye-region annotation.
That is enough for a stronger dense detector, but it was not enough for the original pooled `MLP0` stage.

### 3. Dense eye is the key enabling piece

Once dense eye was used as the stage-1 eye provider:

- eye metrics became healthy
- total loss dropped by an order of magnitude
- pupil bbox refinement became meaningfully trainable

### 4. The new bottleneck is the pupil refinement stage

In the dense-eye variant, final validation losses were:

- `loss_eye = 0.307584`
- `loss_pupil_bbox = 11.350132`

So the dominant term is now clearly `loss_pupil_bbox`.

That means:

- eye ROI localization is no longer the main issue
- the main issue is how the second-stage head uses the eye-conditioned representation to refine pupil bbox

## Practical Recommendation

Current status:

- use the dense-eye `Exp-E` only as an exploratory branch
- do not replace the current `Exp-A`-based Stage1/Stage2 path with `Exp-E` yet

Reason:

1. the dense-eye variant is the first workable `Exp-E`, which is encouraging
2. but `pupil_bbox_iou ≈ 0.10` is still too low to justify replacing the more successful Stage1 mask-guided pipeline

Recommended next-step directions if `Exp-E` is continued:

1. keep dense eye as the fixed eye-stage provider
2. strengthen the pupil refinement stage:
   - add mask-guided token pooling
   - or add consistency between pupil bbox and pupil mask/moments
3. consider teacher-forcing or partial GT-eye warm-up for the pupil stage

## Final Takeaway

The `Exp-E` experiments show a clear progression:

- `ROI Align` version: failed
- `no-ROIAlign` version: slightly less failed
- `dense-eye no-ROIAlign` version: first genuinely working version

So the idea is not dead.
But the evidence now says:

- the hard part was not ROI Align
- the hard part was getting a strong eye-stage provider
- after solving that, the next research target becomes pupil bbox refinement itself
