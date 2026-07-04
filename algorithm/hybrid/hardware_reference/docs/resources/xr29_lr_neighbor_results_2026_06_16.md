# XR-29 LR Neighbor Results

Date: 2026-06-16

## Context

XR-29 followed the XR-28 LR `1.5e-4` promotion. The goal was to test bounded LR neighbors before changing heatmap loss, teacher quality, or optimizer.

Baseline gate before XR-29:

| Leader | Center | P10 | P5 |
|---|---:|---:|---:|
| XR-28 LR `1.5e-4` | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 |

Shared contract:

- Stage2 support-adaptive fixed-count event window: min/base/max `160000/255000/384000`.
- `reference_us=4000003`, `scale_power=0.5`.
- Head-only track-center heatmap-state representation.
- Heatmap grid `32`.
- Heatmap/offset/center-L2 weights `0.005/0.001/0.001`.
- AdamW.
- State-similarity distillation disabled.
- Init/teacher checkpoint: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`.

## Runs

| LR | GPU | Run root | Train exit | Validation best center | Validation best P10 | Validation best P5 |
|---:|---:|---|---:|---:|---:|---:|
| `1.25e-4` | 0 | `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr29_heatmap_centerselect_lr1p25e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_204403` | 0 | 19.9247 | 22.1900 | 7.1069 |
| `1.75e-4` | 1 | `runs/raw_mode1_stage2_count255000_adamw_lr1_75e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr29_heatmap_centerselect_lr1p75e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_204422` | 0 | 19.6459 | 23.2603 | 7.5393 |

## Test Results

All best-P10, best-P5, and best-center checkpoints resolved to the same test metrics within each LR lane.

| LR | Checkpoint lane | Center | P10 | P5 | Gate decision |
|---:|---|---:|---:|---:|---|
| `1.25e-4` | best-P10 | 17.179786903517588 | 32.04761978558132 | 10.53443912097386 | P5 only; center/P10 miss |
| `1.25e-4` | best-P5 | 17.179786903517588 | 32.04761978558132 | 10.53443912097386 | P5 only; center/P10 miss |
| `1.25e-4` | best-center | 17.179786903517588 | 32.04761978558132 | 10.53443912097386 | P5 only; center/P10 miss |
| `1.75e-4` | best-P10 | 17.04961508342198 | 32.56462665285383 | 11.50467722075326 | promotes all gates |
| `1.75e-4` | best-P5 | 17.04961508342198 | 32.56462665285383 | 11.50467722075326 | promotes all gates |
| `1.75e-4` | best-center | 17.04961508342198 | 32.56462665285383 | 11.50467722075326 | promotes all gates |

Delta versus XR-28 LR `1.5e-4`:

| Candidate | Center delta | P10 delta | P5 delta |
|---|---:|---:|---:|
| XR-29 LR `1.75e-4` | -0.035236889975411856 | +0.48341838972909557 | +1.0021258762904566 |

## Decision

XR-29 LR `1.75e-4` is the new active software leader.

New active gates:

- Center `<17.04961508342198`.
- P10 `>32.56462665285383`.
- P5 `>11.50467722075326`.

## Next Experiment

Do not run the longer-budget LR `1.5e-4` fallback as the immediate next step because XR-29 LR `1.75e-4` already promoted all gates.

Recommended next P0:

1. Run a bounded LR micro-bracket around the promoted LR: `1.625e-4` and `1.875e-4`, same 10-epoch contract, center-selected checkpoints.
2. If `1.875e-4` keeps improving validation and test gates, test `2.0e-4` as the upper bound.
3. Defer heatmap-weight, teacher-quality, and optimizer changes until the LR bracket saturates or regresses.

## Evidence

- `runs/_logs/xr29_heatmap_centerselect_lr1p25e-4_gpu0_20260616.log`.
- `runs/_logs/xr29_heatmap_centerselect_lr1p75e-4_gpu1_20260616.log`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr29_heatmap_centerselect_lr1p25e4_adamw_lr1_25e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu0_w0_20260616_205819/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr29_heatmap_centerselect_lr1p25e4_adamw_lr1_25e_4_g32_hm0_005_off0_001_c0_001_bestp5_test_gpu0_w0_20260616_210125/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr29_heatmap_centerselect_lr1p25e4_adamw_lr1_25e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu0_w0_20260616_210432/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr29_heatmap_centerselect_lr1p75e4_adamw_lr1_75e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu1_w0_20260616_205847/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr29_heatmap_centerselect_lr1p75e4_adamw_lr1_75e_4_g32_hm0_005_off0_001_c0_001_bestp5_test_gpu1_w0_20260616_210155/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr29_heatmap_centerselect_lr1p75e4_adamw_lr1_75e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu1_w0_20260616_210505/eval/test/eval_summary.json`.
