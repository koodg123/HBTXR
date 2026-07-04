# XR-32/XR-33 Heatmap LR Midpoint Results - 2026-06-16

## Purpose

Close the trained LR midpoint check after XR-31 no-train interpolation promoted P10 but missed the strict XR-29 P5 gate.

## Shared Contract

- Base runner: `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`
- Event-count target: `255000`
- Support-adaptive event count: `160000 / 255000 / 384000`
- Reference window: `4000003us`, power `0.5`
- Coordinate representation: track heatmap-state, grid `32`
- Loss weights: heatmap `0.005`, offset `0.001`, decoded center `0.001`
- Trainable scope: head-only heatmap path, full-width Stage2
- Distillation: state-similarity disabled
- Checkpoint selection: `metric_track_center_px`

## Runs

| Run | GPU | LR | Log | Run root |
|---|---:|---:|---|---|
| XR-32 | 0 | `1.8125e-4` | `runs/_logs/xr32_heatmap_centerselect_lr1p8125e-4_gpu0_20260616.log` | `runs/raw_mode1_stage2_count255000_adamw_lr1_8125e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr32_heatmap_centerselect_lr1p8125e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_221611` |
| XR-33 | 1 | `1.84375e-4` | `runs/_logs/xr33_heatmap_centerselect_lr1p84375e-4_gpu1_20260616.log` | `runs/raw_mode1_stage2_count255000_adamw_lr1_84375e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr33_heatmap_centerselect_lr1p84375e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_222133` |

Both runs completed epoch `10/10` with train exit `0`.

## Test Results

For both runs, best-P10, best-P5, and best-center checkpoints produced identical full-test metrics within each lane.

| Run | Center | P10 | P5 | Decision |
|---|---:|---:|---:|---|
| XR-32 LR `1.8125e-4` | 17.041867678506033 | 32.5463443006788 | 11.37500034059797 | no promotion |
| XR-33 LR `1.84375e-4` | 17.039179919447218 | 32.62074908529009 | 11.362245225906372 | P10 promoted |

## Eval Artifacts

- XR-32 best-P10: `runs/eval_fixed255k_xr27_trackheatmap_xr32_heatmap_centerselect_lr1p8125e4_adamw_lr1_8125e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu0_w0_20260616_222938/eval/test/eval_summary.json`
- XR-32 best-P5: `runs/eval_fixed255k_xr27_trackheatmap_xr32_heatmap_centerselect_lr1p8125e4_adamw_lr1_8125e_4_g32_hm0_005_off0_001_c0_001_bestp5_test_gpu0_w0_20260616_223248/eval/test/eval_summary.json`
- XR-32 best-center: `runs/eval_fixed255k_xr27_trackheatmap_xr32_heatmap_centerselect_lr1p8125e4_adamw_lr1_8125e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu0_w0_20260616_223559/eval/test/eval_summary.json`
- XR-33 best-P10: `runs/eval_fixed255k_xr27_trackheatmap_xr33_heatmap_centerselect_lr1p84375e4_adamw_lr1_84375e_4_g32_hm0_005_off0_001_c0_001_bestp10_test_gpu1_w0_20260616_223519/eval/test/eval_summary.json`
- XR-33 best-P5: `runs/eval_fixed255k_xr27_trackheatmap_xr33_heatmap_centerselect_lr1p84375e4_adamw_lr1_84375e_4_g32_hm0_005_off0_001_c0_001_bestp5_test_gpu1_w0_20260616_223825/eval/test/eval_summary.json`
- XR-33 best-center: `runs/eval_fixed255k_xr27_trackheatmap_xr33_heatmap_centerselect_lr1p84375e4_adamw_lr1_84375e_4_g32_hm0_005_off0_001_c0_001_bestcenter_test_gpu1_w0_20260616_224056/eval/test/eval_summary.json`

## Updated Active Gates

- Center `<17.035534060001375` from XR-30 LR `1.875e-4`.
- P10 `>32.62074908529009` from XR-33 LR `1.84375e-4`.
- P5 `>11.50467722075326` from XR-29 LR `1.75e-4`.

Best unified checkpoint remains XR-29 LR `1.75e-4` because XR-33 improves P10 but regresses center and P5.

## Next Experiment Direction

LR midpoint training has mostly saturated the heatmap LR axis. The next P0 should avoid another close LR-only replay unless it tests a specific mechanism. Prefer one of:

1. Heatmap loss-ratio refinement around XR-29/XR-33, preserving LR near `1.75e-4` to `1.84375e-4`.
2. Longer-budget reproduction of XR-29/XR-33 only if epoch-10 non-convergence evidence appears in histories.
3. A bounded teacher/soup branch using XR-29 as center/P5 anchor and XR-33 as P10 anchor.
