# XR-30 LR Micro-Bracket Results

Date: 2026-06-16

## Context

XR-30 tested LR micro-neighbors around the XR-29 LR `1.75e-4` promoted heatmap-state leader.

Active gates before XR-30:

| Source | Center | P10 | P5 |
|---|---:|---:|---:|
| XR-29 LR `1.75e-4` | 17.04961508342198 | 32.56462665285383 | 11.50467722075326 |

Shared contract:

- Support-adaptive fixed-count event window: min/base/max `160000/255000/384000`.
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
| `1.625e-4` | 0 | `runs/raw_mode1_stage2_count255000_adamw_lr1_625e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr30_heatmap_centerselect_lr1p625e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_211610` | 0 | 19.7079 | 23.2626 | 7.7504 |
| `1.875e-4` | 1 | `runs/raw_mode1_stage2_count255000_adamw_lr1_875e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr30_heatmap_centerselect_lr1p875e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_211629` | 0 | 19.6130 | 23.5748 | 7.3877 |

## Test Results

All best-P10, best-P5, and best-center checkpoints resolved to the same test metrics within each LR lane.

| LR | Checkpoint lanes | Center | P10 | P5 | Gate decision |
|---:|---|---:|---:|---:|---|
| `1.625e-4` | best-P10/best-P5/best-center | 17.067689692974092 | 32.420068802152365 | 10.866496937615532 | no active gate promoted |
| `1.875e-4` | best-P10/best-P5/best-center | 17.035534060001375 | 32.42474567549569 | 11.266581957680838 | center gate only |

Delta for LR `1.875e-4` versus XR-29 LR `1.75e-4`:

| Candidate | Center delta | P10 delta | P5 delta |
|---|---:|---:|---:|
| XR-30 LR `1.875e-4` | -0.014081023420605021 | -0.1398809773581391 | -0.23809526307242201 |

## Decision

XR-30 LR `1.875e-4` becomes the center leader only.

Active gates after XR-30:

- Center `<17.035534060001375` from XR-30 LR `1.875e-4`.
- P10 `>32.56462665285383` from XR-29 LR `1.75e-4`.
- P5 `>11.50467722075326` from XR-29 LR `1.75e-4`.

Best unified checkpoint remains XR-29 LR `1.75e-4`, because it preserves stronger P10/P5 while remaining close on center.

## Next Experiment

Recommended next P0:

1. Run a no-train checkpoint interpolation between XR-29 LR `1.75e-4` and XR-30 LR `1.875e-4` with small center-direction alphas, for example `0.25`, `0.50`, and `0.75` toward XR-30.
2. If interpolation fails to preserve P10/P5, run one trained LR midpoint `1.8125e-4` with the same contract.
3. Do not jump to `2.0e-4` yet: LR `1.875e-4` improved center but already regressed P10/P5.

## Evidence

- `runs/_logs/xr30_heatmap_centerselect_lr1p625e-4_gpu0_20260616.log`.
- `runs/_logs/xr30_heatmap_centerselect_lr1p875e-4_gpu1_20260616.log`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr30_heatmap_centerselect_lr1p625e4_adamw_lr1_625e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu0_w0_20260616_213020/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr30_heatmap_centerselect_lr1p625e4_adamw_lr1_625e_4_g32_hm0_005_off0_001_c0_001_bestp5_test_gpu0_w0_20260616_213327/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr30_heatmap_centerselect_lr1p625e4_adamw_lr1_625e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu0_w0_20260616_213635/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr30_heatmap_centerselect_lr1p875e4_adamw_lr1_875e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu1_w0_20260616_213055/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr30_heatmap_centerselect_lr1p875e4_adamw_lr1_875e_4_g32_hm0_005_off0_001_c0_001_bestp5_test_gpu1_w0_20260616_213401/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr27_trackheatmap_xr30_heatmap_centerselect_lr1p875e4_adamw_lr1_875e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu1_w0_20260616_213707/eval/test/eval_summary.json`.
