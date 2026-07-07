# HBTXR_v3_0 Mode0 Stage1 Experiment Results Analysis

Last updated: 2026-03-29 KST

Role: `Experiment Analyst`, `Training Debugger`, `Mode0 Stage1 Reviewer`

## Summary

This document compares the four recent `mode0-stage1` experiments that were used to debug Stage1 behavior:

- the unstable `distillation + pruning` run
- the stable but weak `full stage1 without distillation/pruning` run
- the `manifest metadata control` rerun
- the `eye-only + sensor_full_letterbox` sanity run

The overall conclusion is:

- the original `distillation + pruning` configuration can produce the best short-term search accuracy, but it is not stable enough to run to completion
- turning off `distillation` and `pruning` removes catastrophic divergence, but `facet_square_direct` still makes the eye ROI task too degenerate to be a meaningful localization signal
- regenerating manifest metadata alone does not change runtime behavior
- `sensor_full_letterbox` successfully converts eye ROI regression into a real full-sensor localization task, and the eye-only run validates that this path can train stably

## 1. Run Inventory

| Run | Main configuration | Purpose | Outcome |
|---|---|---|---|
| `mode0_stage1_20260328_213706` | `facet_square_direct`, full Stage1, `distillation=true`, `pruning=true` | original full recipe | strong peak search accuracy, then severe late-stage divergence |
| `mode0_stage1_20260328_235641` | `facet_square_direct`, full Stage1, `distillation=false`, `pruning=false` | isolate full Stage1 behavior without KD/pruning | stable training, but search quality remains weak and eye loss is misleadingly easy |
| `mode0_stage1_20260329_004755` | same runtime behavior as previous run, manifest metadata regenerated | test whether manifest rewrite changed actual learning behavior | confirmed that manifest metadata change alone had no effect |
| `mode0_stage1_eye_only_letterbox30_20260329_013549` | `sensor_full_letterbox`, `eye-only`, `distillation=false`, `pruning=false` | validate real eye ROI regression under full-sensor aspect-preserving transform | stable and meaningful eye ROI regression |

## 2. Detailed Analysis By Run

### 2.1 `mode0_stage1_20260328_213706`

Configuration characteristics:

- full Stage1 supervision
- `facet_square_direct`
- `distillation.enabled: true`
- `pruning.enabled: true`

Peak validation points:

- best validation `loss_total`: `24.752356` at epoch `41`
- best validation `loss_eye`: `0.049190` at epoch `44`
- best validation `metric_search_p10_pct`: `29.159165` at epoch `58`
- best validation `metric_search_center_px`: `15.996359` at epoch `55`

Representative progression:

| Epoch | train `loss_total` | train `loss_distillation_mask` | val `loss_mask` | val `metric_search_p10_pct` | val `metric_search_center_px` |
|---|---:|---:|---:|---:|---:|
| `41` | `22.042754` | `1.469667` | `0.574724` | `25.483184` | `16.636629` |
| `55` | `24.804384` | `6.233085` | `2.552317` | `28.671299` | `15.996359` |
| `58` | `28.062032` | `8.947669` | `3.635782` | `29.159165` | `16.830697` |
| `80` | `143.569349` | `99.782758` | `22.218680` | `22.679074` | `20.599844` |
| `123` | `8126.062409` | `6412.669149` | `703.368967` | `4.566074` | `42.113495` |

Interpretation:

- this run clearly learned useful search behavior early on
- the main failure mode was not a simple NaN explosion but a runaway growth of `loss_distillation_mask` and related mask terms
- the search branch was strongest around epoch `55-58`, after which performance collapsed
- this run proves that the architecture can reach materially better search metrics than the later no-distillation run, but the recipe is operationally unsafe in its current form

Conclusion:

- useful as evidence that the search branch can learn
- not suitable as a stable default Stage1 recipe

### 2.2 `mode0_stage1_20260328_235641`

Configuration characteristics:

- full Stage1 supervision
- `facet_square_direct`
- `distillation.enabled: false`
- `pruning.enabled: false`

Best validation points:

- best validation `loss_total`: `33.520172` at epoch `72`
- best validation `loss_eye`: `0.008951` at epoch `18`
- best validation `metric_search_p10_pct`: `14.615495` at epoch `78`
- best validation `metric_search_center_px`: `24.190518` at epoch `72`

Representative progression:

| Epoch | train `loss_total` | train `loss_eye` | train `loss_search_xy` | train `loss_search_geo` | val `metric_search_p10_pct` | val `metric_search_center_px` |
|---|---:|---:|---:|---:|---:|---:|
| `1` | `391.413610` | `176.953420` | `105.761911` | `82.693153` | `0.000000` | `121.578792` |
| `18` | `52.491963` | `0.084122` | `26.610258` | `22.309689` | `5.473342` | `45.384160` |
| `30` | `48.810975` | `0.236973` | `24.313385` | `20.699312` | `4.790249` | `41.636769` |
| `72` | `27.252941` | `0.161051` | `12.178487` | `11.467521` | `13.482566` | `24.190518` |
| `78` | `26.164195` | `0.136772` | `11.610089` | `11.032477` | `14.615495` | `24.307915` |
| `80` | `26.007667` | `0.124813` | `11.542727` | `10.966445` | `13.520283` | `24.334083` |

Interpretation:

- this run is stable and does not show the catastrophic late-stage divergence of the previous one
- however, search quality remains clearly weaker than the earlier unstable run
- the eye loss collapses to near-zero very early, but that low number is misleading because under `facet_square_direct` the eye target is almost degenerate in the transformed input space
- the remaining difficulty is dominated by search-state regression rather than eye ROI loss

Conclusion:

- safer than the `distillation + pruning` recipe
- still not a good final Stage1 recipe, because the eye supervision is too easy and the search branch remains underpowered

### 2.3 `mode0_stage1_20260329_004755`

Configuration characteristics:

- intended as a control rerun after regenerating manifest metadata
- runtime settings remained effectively the same as the previous full Stage1 run

Observed result:

- the run was intentionally stopped after epoch `1`
- epoch-1 metrics match `mode0_stage1_20260328_235641` epoch `1` up to about `1e-6`

Direct epoch-1 comparison:

| Metric | `20260328_235641` epoch 1 | `20260329_004755` epoch 1 | absolute difference |
|---|---:|---:|---:|
| train `loss_total` | `391.41361015073716` | `391.41361129924815` | `1.15e-06` |
| train `loss_eye` | `176.9534202493647` | `176.95342065954722` | `4.10e-07` |
| train `loss_search_xy` | `105.76191133068454` | `105.76191145373929` | `1.23e-07` |
| train `loss_search_geo` | `82.69315270454653` | `82.69315290963777` | `2.05e-07` |
| val `loss_total` | `313.74928566261576` | `313.7492845323351` | `1.13e-06` |

Interpretation:

- regenerating manifest metadata did not change the actual runtime transform that the model saw
- this control run was important because it ruled out the manifest rewrite as the primary cause of Stage1 behavior
- the real issue was the runtime transform path, not the stored manifest metadata

Conclusion:

- useful as a negative control
- not an independent training result, but a decisive contract-validation experiment

### 2.4 `mode0_stage1_eye_only_letterbox30_20260329_013549`

Configuration characteristics:

- `sensor_full_letterbox`
- `eye-only` Stage1
- `distillation.enabled: false`
- `pruning.enabled: false`

Best validation points:

- best validation `loss_eye`: `10.081159` at epoch `10`
- epoch `30` validation `loss_eye`: `11.709246`

Representative progression:

| Epoch | train `loss_eye` | val `loss_eye` |
|---|---:|---:|
| `1` | `107.888227` | `92.395671` |
| `10` | `6.410475` | `10.081159` |
| `30` | `1.582119` | `11.709246` |

Overlay verification on 8 random validation samples:

- mean IoU in sensor coordinates: `0.663755`
- min IoU: `0.540969`
- max IoU: `0.725800`

Interpretation:

- this is the first run in the sequence that validates meaningful eye ROI regression under a full-sensor, aspect-preserving transform
- the loss values are numerically much larger than the near-zero eye loss in the `facet_square_direct` run, but this is expected because the problem is now a real localization task rather than a nearly constant target
- the overfitting signal after epoch `10` is mild and easy to manage with checkpoint selection or early stopping

Conclusion:

- validates the corrected eye ROI path
- provides the cleanest current evidence that the eye head itself can learn once the spatial contract is fixed

## 3. Cross-Run Conclusions

### 3.1 What caused the worst failure

- the catastrophic collapse in `20260328_213706` is best explained by unstable auxiliary supervision, especially mask distillation
- the late-stage explosion is not a generic optimizer failure because it disappears when `distillation` and `pruning` are disabled

### 3.2 What did not cause the problem

- simply regenerating manifest metadata did not change runtime learning behavior
- therefore the earlier `target_size_wh` mismatch was a documentation and contract issue, but not the immediate source of the observed metric changes

### 3.3 What the stable full-stage1 run taught us

- removing `distillation` and `pruning` makes training stable
- but stability alone does not make the eye ROI task meaningful if the transform policy keeps the eye target nearly degenerate

### 3.4 What is now validated

- `sensor_full_letterbox` fixes the aspect-ratio problem of `sensor_full_square`
- the eye ROI head can train meaningfully on full-sensor supervision under this policy

### 3.5 What remains unresolved

- the full `search + eye + mask` Stage1 recipe has not yet been revalidated after switching to the corrected full-sensor transform path
- `xy_from_mask_centroid` has supporting evidence from offline inspection but still needs an end-to-end Stage1 validation run

## 4. Decision Checklist

- [x] characterize the original divergent full Stage1 run
- [x] characterize the no-distillation/no-pruning full Stage1 run
- [x] run a control experiment to test whether manifest regeneration changes runtime behavior
- [x] validate full-sensor aspect-preserving eye ROI regression
- [x] generate qualitative overlay evidence for the best eye-only checkpoint
- [ ] rebuild the full Stage1 recipe on top of `sensor_full_letterbox`
- [ ] compare full Stage1 with and without `xy_from_mask_centroid`
- [ ] decide whether to keep `mode0` as `valid 384` or move toward `clean 366`

## 5. Practical Recommendation

The current evidence supports the following order of operations:

1. keep `sensor_full_letterbox` as the `mode0` eye ROI transform path
2. restore `search` and `mask` heads on top of that corrected transform
3. run a short full Stage1 sanity experiment before reintroducing any distillation or pruning
4. only after the new full Stage1 baseline is stable, test whether mask-centroid fusion improves search-center metrics
