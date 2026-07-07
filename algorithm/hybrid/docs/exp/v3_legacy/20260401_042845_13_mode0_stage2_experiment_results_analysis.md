# HBTXR_v3_0 Mode0 Stage2 Experiment Results Analysis

Last updated: 2026-04-08 KST

Role: `Experiment Analyst`, `Stage2 Debugger`, `Mode0 Training Reviewer`

## Summary

This document records the current `mode0-stage2` experiment sequence that followed the tuned Stage1 checkpoint:

- `mode0_stage1_dense_eye30_tuned_nogeo_20260330_193647/train/best_search_p10.pt`

The current Stage2 conclusions are:

- the previous multi-GPU `DataParallel` crash has been fixed
- turning `search/event/track_geo` weights to `0.0` is necessary to keep Stage2 numerically sane
- with `geo=0`, Stage2 becomes stable, but `track` accuracy still does not improve over the first epoch
- the earlier full Stage2 run was likely limited by loss composition and by an input-contract mismatch with Stage1
- even after aligning Stage2 to the `Exp-A` Stage1 bootstrap assumptions, the current Stage2 objective still fails to preserve strong search quality while improving tracking

## 1. Run Inventory

| Run | Main configuration | Purpose | Outcome |
|---|---|---|---|
| `mode0_stage2_multigpu_fixcheck_20260331_210333` | `cuda:0,cuda:1`, `geo on`, `distillation=false`, `pruning=false` | verify that the Stage2 `DataParallel` path no longer crashes | crash fixed, but geo losses dominate total loss |
| `mode0_stage2_singlegpu_geo0_sanity_20260331_210333` | single GPU, `geo=0`, `distillation=false`, `pruning=false` | test whether zeroing geometry terms stabilizes Stage2 numerically | stable and much lower total loss, but only short sanity evidence |
| `mode0_stage2_multigpu_geo0_subset64_20260331_222906` | `cuda:0,cuda:1`, subset manifests, `geo=0` | verify end-to-end multi-GPU execution with the new trainer fix and reduced loss recipe | completed one epoch successfully, proving the new path works |
| `mode0_stage2_50ep_nw4_geo0_denseeye_20260331_223948` | `cuda:0,cuda:1`, `num_workers=4`, full manifests, `geo=0`, `distillation=false`, `pruning=false` | main Stage2 run from the tuned Stage1 checkpoint | stable, but stopped at epoch 15 because `track_p10` never improved beyond epoch 1 |
| `mode0_stage2_from_expA_50ep_gpu0_nw4_20260402_023825` | single GPU `cuda:0`, `num_workers=4`, `geo=0`, `sensor_full_letterbox`, `xy_from_mask_centroid=true`, initialized from `Exp-A` best checkpoint | test whether an `Exp-A` bootstrap and aligned transform contract produce a stronger long Stage2 run | stable for all 50 epochs, but best `track_p10` remains at epoch 1 and search quality collapses over training |

## 2. Detailed Analysis By Run

### 2.1 `mode0_stage2_multigpu_fixcheck_20260331_210333`

Configuration characteristics:

- multi-GPU `cuda:0,cuda:1`
- full manifests
- `distillation.enabled: false`
- `pruning.enabled: false`
- geometry terms still active

Validation epoch-1 result:

- `loss_total`: `2795.529158`
- `metric_search_p10_pct`: `85.646900`
- `metric_event_p10_pct`: `3.026730`
- `metric_track_p10_pct`: `13.350180`
- `loss_search_geo`: `938.119263`
- `loss_event_geo`: `877.527842`
- `loss_track_geo`: `632.279325`

Interpretation:

- this run is important because it proves the earlier `DataParallel` crash is fixed
- however, the geometry terms still completely dominate the objective
- the large total loss makes it difficult to interpret whether the model is learning the intended hybrid tracking behavior

Conclusion:

- useful as a crash-fix validation run
- not a suitable Stage2 recipe because geometry terms overwhelm the rest of the signal

### 2.2 `mode0_stage2_singlegpu_geo0_sanity_20260331_210333`

Configuration characteristics:

- single GPU
- full manifests
- `loss.search_geo_weight = 0.0`
- `loss.event_geo_weight = 0.0`
- `loss.track_geo_weight = 0.0`
- `distillation.enabled: false`
- `pruning.enabled: false`

Short sanity result:

| Epoch | val `loss_total` | val `metric_search_p10_pct` | val `metric_event_p10_pct` | val `metric_track_p10_pct` |
|---|---:|---:|---:|---:|
| `1` | `96.488496` | `99.747305` | `5.065139` | `5.925427` |
| `2` | `97.781065` | `99.573225` | `2.161950` | `7.924529` |

Interpretation:

- zeroing the geometry terms immediately makes Stage2 numerically healthy
- the search branch is already very strong at initialization
- event and track metrics remain modest, but the run at least becomes interpretable

Conclusion:

- validates `geo=0` as a practical stabilization step
- still too short to judge final Stage2 quality

### 2.3 `mode0_stage2_multigpu_geo0_subset64_20260331_222906`

Configuration characteristics:

- multi-GPU `cuda:0,cuda:1`
- `geo=0`
- subset manifests with `64` train rows and `64` validation rows
- `distillation.enabled: false`
- `pruning.enabled: false`

Validation epoch-1 result:

- `loss_total`: `283.635380`
- `metric_search_p10_pct`: `78.601192`
- `metric_event_p10_pct`: `0.0`
- `metric_track_p10_pct`: `13.333333`
- `metric_search_center_px`: `6.501612`
- `metric_event_center_px`: `175.901217`
- `metric_track_center_px`: `46.804476`

Interpretation:

- this run is not a quality benchmark because the subset is too small and too biased
- its main value is operational: it proves that `geo=0 + DataParallel` completes end-to-end after the trainer fix

Conclusion:

- good infrastructure sanity check
- not a representative quality result

### 2.4 `mode0_stage2_50ep_nw4_geo0_denseeye_20260331_223948`

Configuration characteristics:

- full manifests
- `cuda:0,cuda:1`
- `training.num_workers: 4`
- `training.epochs: 50`
- `distillation.enabled: false`
- `pruning.enabled: false`
- `geo=0`
- initialized from the tuned dense-eye Stage1 checkpoint

Important resolved runtime properties:

- `best_metric_name: metric_track_p10_pct`
- `xy_from_mask_centroid: true`
- `mode0.resize_policy: facet_square_direct`

The last point is important because Stage1 had been tuned under `sensor_full_letterbox`, so Stage2 did not preserve the same input transform contract.

Observed result:

- the run did not reach epoch `50`
- it stopped at epoch `15`
- `best.pt` and `best_track_p10.pt` were both saved at epoch `1`
- `best_track_p5.pt` was saved at epoch `5`

Best validation points:

- best `metric_track_p10_pct`: epoch `1`, `9.177898`
- best `metric_track_p5_pct`: epoch `5`, `2.695418`
- best `metric_event_p10_pct`: epoch `5`, `5.587377`
- best `metric_search_p10_pct`: epoch `6`, `100.0`
- best `metric_search_p5_pct`: epoch `12`, `99.707996`
- best `metric_track_center_px`: epoch `13`, `43.080013`
- best `metric_event_center_px`: epoch `8`, `45.582834`
- best `metric_search_center_px`: epoch `12`, `1.472320`
- best `loss_total`: epoch `15`, `93.020398`

First-to-last validation change:

| Metric | Epoch 1 | Epoch 15 | Change |
|---|---:|---:|---:|
| `loss_total` | `96.651382` | `93.020398` | `-3.630984` |
| `metric_search_p10_pct` | `98.989218` | `100.0` | `+1.010782` |
| `metric_search_p5_pct` | `81.732929` | `99.455301` | `+17.722372` |
| `metric_event_p10_pct` | `5.536838` | `2.633648` | `-2.903190` |
| `metric_track_p10_pct` | `9.177898` | `6.038859` | `-3.139039` |
| `metric_track_p5_pct` | `1.791330` | `0.898473` | `-0.892857` |
| `metric_search_center_px` | `3.363966` | `1.538893` | `-1.825073` |
| `metric_event_center_px` | `45.802961` | `45.667497` | `-0.135463` |
| `metric_track_center_px` | `44.793847` | `43.103389` | `-1.690457` |

Interpretation:

- this run is numerically stable and does not exhibit the earlier geometry-loss explosion
- the search branch becomes almost perfect very quickly, which means Stage2 is not bottlenecked by search anymore
- event and track branches do not benefit from continued training; in fact, the main tracking metric is best immediately at epoch `1`
- the falling `loss_total` is misleading as a progress signal because it does not correspond to better `track_p10`
- this is exactly why the run stopped early despite a nominal `50`-epoch request

Conclusion:

- `geo=0` is a necessary stability fix
- but the current Stage2 objective still does not translate stable optimization into better tracking accuracy
- the most likely next bottlenecks are loss balancing and the Stage1/Stage2 transform mismatch

### 2.5 `mode0_stage2_from_expA_50ep_gpu0_nw4_20260402_023825`

Configuration characteristics:

- single GPU `cuda:0`
- full manifests
- `training.num_workers: 4`
- `training.epochs: 50`
- `distillation.enabled: false`
- `pruning.enabled: false`
- `geo=0`
- `mode0.resize_policy: sensor_full_letterbox`
- `model.search.xy_from_mask_centroid: true`
- initialized from:
  - `runs/mode0_stage1_expA_maskcentroid50_from_ciou100_20260401_044109/train/best_search_p10.pt`

Observed result:

- the run completed all `50` epochs
- `best_track_p10.pt` was saved at epoch `1`
- the run stayed numerically stable throughout, but tracking accuracy never exceeded the first validation point

Best validation points:

- best `metric_track_p10_pct`: epoch `1`, `27.899821`
- best `metric_track_p5_pct`: epoch `1`, `11.190476`
- best `metric_track_center_px`: epoch `24`, `24.886914`
- best `metric_event_p10_pct`: epoch `7`, `7.918913`
- best `metric_search_p10_pct`: epoch `1`, `96.209569`
- best `metric_search_center_px`: epoch `1`, `2.122257`
- best `loss_total`: epoch `2`, `55.926322`

First-to-last validation change:

| Metric | Epoch 1 | Epoch 50 | Change |
|---|---:|---:|---:|
| `loss_total` | `58.396747` | `75.273037` | `+16.876290` |
| `metric_search_p10_pct` | `96.209569` | `6.385894` | `-89.823675` |
| `metric_search_p5_pct` | `93.306379` | `0.811995` | `-92.494384` |
| `metric_search_center_px` | `2.122257` | `30.861686` | `+28.739429` |
| `metric_event_p10_pct` | `6.436433` | `6.020890` | `-0.415544` |
| `metric_track_p10_pct` | `27.899821` | `15.250450` | `-12.649371` |
| `metric_track_p5_pct` | `11.190476` | `3.507413` | `-7.683064` |
| `metric_track_center_px` | `25.595335` | `25.222216` | `-0.373118` |

Best-epoch (`track_p10`) validation loss profile:

- `loss_search_xy`: `0.997172`
- `loss_search_ab`: `8.776917`
- `loss_event_xy`: `19.435839`
- `loss_event_ab`: `2.946368`
- `loss_track_xy`: `15.276040`
- `loss_track_ab`: `0.443907`
- `loss_consistency`: `8.340515`
- `loss_total`: `58.396747`

Final-epoch validation loss profile:

- `loss_search_xy`: `19.012841`
- `loss_search_ab`: `5.783324`
- `loss_event_xy`: `19.362083`
- `loss_event_ab`: `3.029107`
- `loss_track_xy`: `15.051966`
- `loss_track_ab`: `0.085503`
- `loss_consistency`: `11.035588`
- `loss_total`: `75.273037`

Interpretation:

- this run rules out the earlier Stage1/Stage2 transform mismatch as the only explanation
- even with `sensor_full_letterbox` and the `Exp-A` bootstrap assumptions preserved, Stage2 still fails to improve `track_p10`
- the main pathology is that the search branch deteriorates sharply during Stage2 training, while event/track branches do not improve enough to justify that tradeoff
- the current Stage2 objective is therefore not preserving the useful Stage1 prior

Conclusion:

- `Exp-A` remains the best Stage1 initializer
- but current Stage2 optimization still destroys too much of its search quality
- the next Stage2 iteration should focus on preservation:
  - freezing or partially freezing the search backbone/head
  - reducing search-side drift during Stage2
  - or explicitly regularizing Stage2 against the loaded Stage1 search behavior

## 3. Cross-Run Conclusions

### 3.1 What is now fixed

- the previous Stage2 `DataParallel` crash is fixed
- Stage2 can run on `cuda:0,cuda:1` without the earlier final-small-batch failure

### 3.2 What clearly helps

- zeroing `search/event/track_geo` terms makes the objective numerically manageable
- this removes the most obvious source of loss domination

### 3.3 What is still not working

- the current Stage2 recipe does not improve `metric_track_p10_pct`
- the best tracking checkpoint is effectively the first validation point after loading Stage1 weights

### 3.4 What this implies

- the model is not failing because search is weak
- instead, search is already over-solved relative to the remaining Stage2 difficulty
- event and residual track learning are the unresolved parts

### 3.5 Important contract issue

- Stage1 was tuned under `sensor_full_letterbox`
- this Stage2 run resolved to `facet_square_direct`
- that means Stage2 did not inherit the same spatial contract as the Stage1 checkpoint it was initialized from

This mismatch is a strong candidate explanation for why Stage2 does not convert a strong Stage1 checkpoint into improving track metrics.

The later `mode0_stage2_from_expA_50ep_gpu0_nw4_20260402_023825` run weakens that explanation:

- once Stage2 was aligned to `sensor_full_letterbox` and initialized from `Exp-A`, the run still failed
- the transform mismatch was real, but it was not the full story
- the remaining bottleneck is now more clearly in Stage2 loss composition and optimization behavior

## 4. Decision Checklist

- [x] fix the Stage2 multi-GPU `DataParallel` crash
- [x] verify Stage2 on single GPU with `geo=0`
- [x] verify Stage2 on multi-GPU with `geo=0` using a small subset
- [x] run a full-manifest Stage2 training attempt with `num_workers=4`
- [x] confirm that the run is stable under `geo=0`
- [x] confirm that `track_p10` still fails to improve
- [x] align Stage2 input transform with the tuned Stage1 `sensor_full_letterbox` contract
- [ ] rebalance Stage2 losses so event/track learning matters more than already-solved search
- [ ] test Stage2 search-preservation strategies after `Exp-A` bootstrap

## 5. Practical Recommendation

The current evidence supports the following next steps:

1. keep `mode0_stage2` on the aligned `sensor_full_letterbox + Exp-A` bootstrap contract
2. keep `geo=0` for the next Stage2 validation cycle
3. add explicit search-preservation mechanisms so Stage2 cannot destroy the loaded Stage1 prior so quickly
4. reduce the influence of the already-solved search branch or raise the relative importance of event/track learning
5. validate the next Stage2 recipe with a short `5-10` epoch run before attempting another long training job
