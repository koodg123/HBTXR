# Mode0 Stage1 Mask-Guidance Ablation Analysis

## Scope

This document compares four Stage1 follow-up experiments that all start from the same reference checkpoint:

- reference checkpoint:
  - `runs/mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411/train/best_search_p10.pt`
- reference baseline document:
  - `14_mode0_stage1_ciou_100ep_results_analysis.md`

The goal of these experiments was to test how much the pupil mask branch can guide pupil detection.

## Experiments

### Exp-A: Hard `xy_from_mask_centroid`

- config:
  - `exps/configs/mode0_stage1_expA_maskcentroid.yaml`
- run:
  - `runs/mode0_stage1_expA_maskcentroid50_from_ciou100_20260401_044109`
- method:
  - `search xy` is directly replaced by the predicted mask centroid
  - `ab/trig/conf` remain from the original search head

### Exp-B: Soft Center Consistency

- config:
  - `exps/configs/mode0_stage1_expB_maskcenter.yaml`
- run:
  - `runs/mode0_stage1_expB_maskcenter50_from_ciou100_20260401_054827`
- method:
  - keep the original search head output
  - add `loss_search_mask_center`
  - this softly encourages `search_xy` to match the predicted mask centroid

### Exp-C: Soft Center + Axis Consistency

- config:
  - `exps/configs/mode0_stage1_expC_maskcenter_ab.yaml`
- run:
  - `runs/mode0_stage1_expC_maskcenter_ab50_from_ciou100_20260401_154059`
- method:
  - same as Exp-B
  - plus `loss_search_mask_ab`
  - this additionally encourages `search_ab` to match mask-derived axes

### Exp-D: Soft Mask Cascade

- config:
  - `exps/configs/mode0_stage1_expD_maskcascade.yaml`
- final run:
  - `runs/mode0_stage1_expD_maskcascade50_from_ciou100_fixeig_20260401_215301`
- failed first attempt:
  - `runs/mode0_stage1_expD_maskcascade50_from_ciou100_20260401_163618`
- method:
  - `xy = mask_center + residual_xy`
  - `ab = mask_axes * exp(residual_ab * scale)`
  - `trig/conf` remain from the original search head

The first Exp-D run failed immediately because `torch.linalg.eigvalsh()` inside `DataParallel` raised:

- `RuntimeError: lazy wrapper should be called at most once`

The final Exp-D result below uses the fixed closed-form covariance solution.

## Reference Baseline

The reference recipe was:

- `sensor_full_letterbox`
- `dense eye + search + mask`
- `search_xy_weight = 0.25`
- `search_ab_weight = 0.25`
- `search_trig_weight = 0.5`
- `search_ciou_weight = 0.25`
- `search_geo_weight = 0.0`
- cosine schedule
- `100` epochs

Reference run:

- `runs/mode0_stage1_dense_eye_search_mask_ciou_xy025_100ep_20260401_021411`

Reference best validation metrics:

| Metric | Best Epoch | Value |
| --- | ---: | ---: |
| `metric_search_p10_pct` | 71 | `22.29` |
| `metric_search_p5_pct` | 55 | `6.42` |
| `metric_search_center_px` | 73 | `17.28` |
| `metric_eye_iou` | 39 | `0.7446` |

## Headline Comparison

| Run | Best `search_p10` | Best `search_p5` | Best `search_center_px` | Best `eye_iou` |
| --- | ---: | ---: | ---: | ---: |
| Baseline | `22.29` | `6.42` | `17.28` | `0.7446` |
| Exp-A | `99.63` | `98.89` | `1.82` | `0.7594` |
| Exp-B | `27.25` | `8.86` | `16.58` | `0.7348` |
| Exp-C | `20.26` | `5.76` | `19.85` | `0.7517` |
| Exp-D | `98.75` | `88.98` | `3.02` | `0.7042` |

## Final-Epoch Validation Comparison

| Run | Last Epoch | `search_p10` | `search_p5` | `search_center_px` | `eye_iou` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 100 | `21.53` | `4.92` | `17.96` | `0.7286` |
| Exp-A | 50 | `93.67` | `91.83` | `2.47` | `0.7533` |
| Exp-B | 50 | `24.73` | `8.28` | `16.96` | `0.7163` |
| Exp-C | 50 | `17.61` | `3.75` | `21.04` | `0.7420` |
| Exp-D | 50 | `95.69` | `82.10` | `3.88` | `0.6797` |

## Training Dynamics

Selected validation snapshots:

| Run | Epoch 1 `p10` | Epoch 5 `p10` | Epoch 20 `p10` | Epoch 40 `p10` | Epoch 50 `p10` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exp-A | `98.36` | `99.63` | `92.07` | `93.92` | `93.67` |
| Exp-B | `0.00` | `6.04` | `14.67` | `23.36` | `24.73` |
| Exp-C | `0.00` | `7.07` | `9.15` | `14.35` | `17.61` |
| Exp-D | `8.64` | `0.00` | `96.29` | `95.43` | `95.69` |

Interpretation:

1. Exp-A improves immediately because it hard-replaces `xy` with mask centroid.
2. Exp-B improves gradually and is the cleanest soft-guidance result.
3. Exp-C improves more slowly than Exp-B and never catches up.
4. Exp-D has a rough early phase but then snaps into a very strong center solution once the cascade stabilizes.

## Comparison Plots

Artifacts:

- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/comparison_curves_abcd_vs_base.png`
- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/overlay_compare_abcd_best_val8.png`
- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/summary.json`

The curve plot compares:

- `val metric_search_p10_pct`
- `val metric_search_center_px`
- `val loss_search_xy`
- `val loss_search_ab`

across:

- baseline
- Exp-A
- Exp-B
- Exp-C
- Exp-D

Interpretation from the curve plot:

1. Exp-A solves the center metric immediately because `xy` is replaced by mask centroid.
2. Exp-B improves more slowly, but the curve is stable and monotonic in the useful direction.
3. Exp-C does not recover the gap introduced by the added `mask_ab` coupling.
4. Exp-D has a visibly unstable early phase, then rapidly collapses center error once the cascade begins to work.

## Best-Checkpoint Overlay Comparison

Artifact:

- `workspace/previews/mode0_stage1_mask_guidance_compare_20260401/overlay_compare_abcd_best_val8.png`

Selection:

- split: `val`
- seed: `42`
- count: `8`
- same sample indices for all runs:
  - `[25, 114, 142, 228, 250, 281, 654, 759]`

Overlay drawing:

- GT eye box: green
- predicted eye box: red
- GT pupil ellipse: cyan
- predicted pupil ellipse: yellow
- predicted mask overlay: red transparent fill

### Random-8 Aggregate Summary

| Run | Mean `eye_iou` | Mean `search_center_px` | Mean `mask_iou` |
| --- | ---: | ---: | ---: |
| Exp-A | `0.6272` | `1.79` | `0.6793` |
| Exp-B | `0.6154` | `16.77` | `0.6606` |
| Exp-C | `0.6907` | `23.06` | `0.6835` |
| Exp-D | `0.6353` | `2.68` | `0.5590` |

### Overlay Interpretation

1. Exp-A is the cleanest center winner in both the metric table and the visual panels.
2. Exp-D is also very strong on center, but its average `mask_iou` is clearly worse than Exp-A on this shared sample set.
3. Exp-B remains much more moderate visually, which is consistent with the idea that it is a soft regularizer rather than a hard routing rule.
4. Exp-C often keeps eye and mask quality reasonable, but it still leaves larger pupil-center error than Exp-B or the hard/cascade methods.

This overlay comparison reinforces the earlier numeric interpretation:

- Exp-A and Exp-D are primarily center-solvers
- Exp-B is the most conservative improvement
- Exp-C does not justify its extra complexity in the current form

## A-vs-D Qualitative Focus

Exp-A and Exp-D are the two runs that matter most if Stage1 is evaluated mainly by center quality.
They are not interchangeable, though, and the shared-sample overlay comparison makes the difference clearer.

### Shared-Sample Aggregate

| Run | Mean `eye_iou` | Mean `search_center_px` | Mean `mask_iou` |
| --- | ---: | ---: | ---: |
| Exp-A | `0.6272` | `1.79` | `0.6793` |
| Exp-D | `0.6353` | `2.68` | `0.5590` |

Interpretation:

1. Exp-A is the cleaner center winner on the shared eight-sample set.
2. Exp-D remains very strong on center, but it gives up more mask quality on average.
3. Eye-box quality is roughly comparable, so the main difference is not eye detection. It is how aggressively the mask signal is routed into pupil state prediction.

### Sample-Level Qualitative Differences

Representative examples from the shared overlay set:

1. `user33__right__session_102__000210_1658286177931888`
   - Exp-A:
     - `search_center_px = 0.98`
     - `mask_iou = 0.742`
   - Exp-D:
     - `search_center_px = 4.97`
     - `mask_iou = 0.581`
   - reading:
     - both runs are usable, but Exp-A is visibly more settled in both center alignment and mask fill.

2. `user34__left__session_201__001410_1658287279851893`
   - Exp-A:
     - `search_center_px = 2.07`
     - `mask_iou = 0.591`
   - Exp-D:
     - `search_center_px = 4.40`
     - `mask_iou = 0.301`
   - reading:
     - this is the clearest warning case for Exp-D. Center remains good enough, but the mask and shape support collapse much more sharply.

3. `user36__left__session_102__002210_1658302617497420`
   - Exp-A:
     - `search_center_px = 2.04`
     - `mask_iou = 0.698`
   - Exp-D:
     - `search_center_px = 0.86`
     - `mask_iou = 0.595`
   - reading:
     - Exp-D can beat Exp-A on center for individual samples, but it often does so with a rougher shape prior.

### Structural Interpretation

Exp-A:

- uses the predicted mask centroid as the final `xy`
- has very little ambiguity about where center performance comes from
- keeps `ab` under materially better control than Exp-D in the current runs

Exp-D:

- uses mask statistics as a base state and learns residual correction
- is more expressive and more ambitious
- solves center very well once it stabilizes
- still shows a real risk that `ab` becomes the residual branch's weak point

This is why Exp-D feels promising but not yet final: it is solving the right problem in a more general way, but it still pays too much for that flexibility in ellipse-size quality.

## Loss Interpretation At Best `search_p10`

### Exp-A

Best epoch: `5`

Key validation losses:

- `loss_search_xy = 0.186`
- `loss_search_ab = 1.306`
- `loss_search_trig = 0.405`
- `loss_search_ciou = 0.0889`
- `loss_mask = 0.0490`
- `loss_constraint_center = 0.536`
- `loss_eye = 0.287`

Interpretation:

- `xy` is almost solved because it is directly taken from mask centroid
- `ab` remains nontrivial, but not catastrophic

### Exp-B

Best epoch: `43`

Key validation losses:

- `loss_search_xy = 2.524`
- `loss_search_ab = 1.636`
- `loss_search_trig = 0.210`
- `loss_search_ciou = 0.161`
- `loss_mask = 0.0468`
- `loss_constraint_center = 0.355`
- `loss_eye = 0.316`
- `loss_search_mask_center = 1.034`

Interpretation:

- the model still learns its own `xy`
- mask guidance helps without hard replacement
- this is the most conservative and easiest-to-interpret improvement

### Exp-C

Best epoch: `35`

Key validation losses:

- `loss_search_xy = 3.030`
- `loss_search_ab = 1.553`
- `loss_search_trig = 0.289`
- `loss_search_ciou = 0.178`
- `loss_mask = 0.0431`
- `loss_constraint_center = 0.145`
- `loss_eye = 0.293`
- `loss_search_mask_center = 1.261`
- `loss_search_mask_ab = 0.585`

Interpretation:

- adding `mask_ab` guidance did not help
- the current mask-derived axis signal appears too noisy or too biased to improve Stage1 search axes

### Exp-D

Best epoch: `15`

Key validation losses:

- `loss_search_xy = 0.377`
- `loss_search_ab = 5.854`
- `loss_search_trig = 0.405`
- `loss_search_ciou = 0.250`
- `loss_mask = 0.0658`
- `loss_constraint_center = 0.634`
- `loss_eye = 0.348`

Interpretation:

- center is nearly solved
- but `search_ab` becomes very large
- this means the cascade is excellent for center metrics but may distort or destabilize ellipse-size prediction

## Overall Interpretation

### Exp-A

Pros:

- strongest immediate center performance
- simplest implementation
- very strong `p10/p5/center`

Cons:

- `xy` is no longer a learned search prediction in the usual sense
- result is tightly coupled to mask quality

### Exp-B

Pros:

- best soft-guidance result
- improves the baseline without replacing the search head output
- preserves interpretation of the search branch

Cons:

- gains are smaller than Exp-A or Exp-D on center-focused metrics

### Exp-C

Pros:

- tests the natural next hypothesis after Exp-B

Cons:

- worse than Exp-B
- current `mask_ab` guidance should not be kept as-is

### Exp-D

Pros:

- extremely strong center metric
- more expressive than Exp-A because it still allows residual correction

Cons:

- unstable at the beginning
- `ab` quality remains a concern
- should not be judged by center metrics alone

## Recommendation

Current recommendation depends on the goal.

### If the goal is maximum center accuracy

Use:

- Exp-A first
- Exp-D as the more ambitious follow-up

Reason:

- both dramatically improve `search_center_px` and `search_p10`

### If the goal is the safest production-facing Stage1 improvement

Use:

- Exp-B

Reason:

- it improves the baseline cleanly
- it preserves the original search prediction path
- it is easier to reason about than hard replacement or cascade coupling

### If the goal is full pupil-state quality rather than center alone

Do not promote Exp-D yet without more analysis of:

- ellipse-axis quality
- overlay comparison
- state-level downstream effects in Stage2

### Current non-recommendation

Do not continue Exp-C in its current form.

Reason:

- `mask_ab` guidance did not improve the result

## Current Decision

At this point:

1. Exp-A is the strongest center-accuracy shortcut.
2. Exp-B is the strongest conservative improvement.
3. Exp-C is currently not useful.
4. Exp-D is very promising, but it still needs additional validation on `ab` / full ellipse quality before it can replace Exp-B or Exp-A.

## Stage2 Initialization Recommendation

The Stage2 recommendation should separate:

1. the best pure Stage1 metric result
2. the safest initialization for downstream temporal heads

### Primary recommendation for the next Stage2 run

Use:

- `runs/mode0_stage1_expA_maskcentroid50_from_ciou100_20260401_044109/train/best_search_p10.pt`

Reason:

1. It is the strongest overall checkpoint on the metric that Stage1 is explicitly optimizing.
2. It also keeps `ab` in a much healthier range than Exp-D.
3. Its shared-sample overlay looks cleaner than Exp-D in both center and mask support.

Required note:

- Stage2 should keep the same mask-centroid routing assumption if this checkpoint is used.
- In practice, that means `model.search.xy_from_mask_centroid: true` should remain enabled when evaluating this initialization path.

### Conservative fallback recommendation

If the goal is to keep the original search path more interpretable, use:

- `runs/mode0_stage1_expB_maskcenter50_from_ciou100_20260401_054827/train/best_search_p10.pt`

Reason:

1. It improves the baseline without hard replacement.
2. It is easier to reason about if Stage2 debugging still depends on separating search-head learning from mask-head guidance.

### Current non-recommendation for Stage2 bootstrap

Do not use Exp-D as the default Stage2 initializer yet.

Reason:

1. Its center metrics are excellent, but that is not the full Stage1 objective.
2. `loss_search_ab` is still too unstable relative to Exp-A.
3. The shared-sample overlay suggests that some of Exp-D's gains come with a rougher mask/shape prior than we want to hand over to Stage2.
