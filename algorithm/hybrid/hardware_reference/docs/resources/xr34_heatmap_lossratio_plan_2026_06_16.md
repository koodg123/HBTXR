# XR-34 Heatmap Loss-Ratio Refinement Plan - 2026-06-16

## Purpose

XR-32/XR-33 showed close LR-only refinement is saturated: XR-33 promoted P10, but center and P5 regressed. XR-34 changes the heatmap loss ratio and init anchor while keeping the same heatmap-state support-adaptive fixed255k contract.

## Active Gates Before XR-34

- Center `<17.035534060001375` from XR-30 LR `1.875e-4`.
- P10 `>32.62074908529009` from XR-33 LR `1.84375e-4`.
- P5 `>11.50467722075326` from XR-29 LR `1.75e-4`.

## Shared Controls

- Runner: `scripts/external/run_xr34_heatmap_lossratio_refinement.sh`
- Base runner: `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`
- Event count: `255000`
- Adaptive range: `160000 / 255000 / 384000`
- Reference window: `4000003us`, power `0.5`
- Model head: track heatmap-state, grid `32`
- Trainable scope: `track_center_heatmap_head.*`
- Best metric / scheduler metric: `metric_track_center_px`
- Distillation state-similarity: disabled

## Lanes

| Lane | GPU | Init anchor | LR | Heatmap | Offset | Center L2 | Objective |
|---|---:|---|---:|---:|---:|---:|---|
| XR-34A | 0 | XR-29 best-center | `1.75e-4` | `0.004` | `0.0015` | `0.0015` | preserve center/P5 while improving sub-cell localization |
| XR-34B | 1 | XR-33 best-center | `1.84375e-4` | `0.006` | `0.001` | `0.001` | sharpen current P10 leader direction |

## Launch Evidence

- XR-34A log: `runs/_logs/xr34a_heatmap_lossratio_gpu0_20260616.log`
- XR-34B log: `runs/_logs/xr34b_heatmap_lossratio_gpu1_20260616.log`
- XR-34A startup validation:
  - raw event-count contract passed.
  - init checkpoint: XR-29 LR `1.75e-4` best-center.
  - resolved device: `cuda:0`.
  - trainable filter: `6` tensors / `1,331,328` params.
  - entered epoch `1/10`.
- XR-34B startup validation:
  - raw event-count contract passed.
  - init checkpoint: XR-33 LR `1.84375e-4` best-center.
  - resolved device: `cuda:1`.
  - trainable filter: `6` tensors / `1,331,328` params.
  - entered epoch `1/10`.

## Decision Gate

- Promote center if center `<17.035534060001375`.
- Promote P10 if P10 `>32.62074908529009`.
- Promote P5 if P5 `>11.50467722075326`.
- Do not replace XR-29 unified checkpoint unless a single checkpoint improves the balanced tradeoff or clearly wins multiple active gates.

## Closeout Results

Both lanes completed training with exit `0` and all best-P10, best-P5, and best-center full-test eval summaries.

| Lane | Checkpoint | Epoch | Center px | P10 | P5 | Decision |
|---|---|---:|---:|---:|---:|---|
| XR-34A | best-P10 / best-P5 / best-center | 3 | 17.026217068944657 | 32.430698088237214 | 10.911139822006225 | center only |
| XR-34B | best-P10 | 7 | 16.53321223940168 | 33.77168447630746 | 11.276786088943481 | center and P10 leader |
| XR-34B | best-P5 / best-center | 3 | 17.0139569742339 | 32.49957566261291 | 11.226190853118897 | center only |

Evidence:

- XR-34A train log: `runs/_logs/xr34a_heatmap_lossratio_gpu0_20260616.log`
- XR-34B train log: `runs/_logs/xr34b_heatmap_lossratio_gpu1_20260616.log`
- XR-34B best-P10 summary: `runs/eval_fixed255k_xr27_trackheatmap_xr34b_heatmap_lossratio_p10_xr33init_hm0p006_off0p001_c0p001_lr1p84375e4_adamw_lr1_84375e_4_g32_hm0_006_off0_001_c0_001_bestp10_test_gpu1_w0_20260616_230438/eval/test/eval_summary.json`

Decision:

- Active center gate is now `<16.53321223940168` from XR-34B best-P10.
- Active P10 gate is now `>33.77168447630746` from XR-34B best-P10.
- Active P5 gate remains `>11.50467722075326` from XR-29 LR `1.75e-4`.
- XR-34B is the center/P10 leader, but XR-29 remains the strict P5/unified anchor until a single checkpoint restores P5.

Next experiment:

- XR-35 no-train checkpoint interpolation between XR-29 LR `1.75e-4` and XR-34B best-P10.
- Purpose: test whether a very small move toward XR-34B preserves XR-29 P5 while gaining center/P10.
- Runner: `scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh`.

## Sidecar Evaluator Note

GPT-5.5 sidecar audit agreed optimizer changes and teacher retraining are not current P0. It ranked bounded longer-budget continuation of XR-29/XR-33 as the strongest next fallback because both histories still improved at epoch `10` with `early_counter=0/4`. XR-34 remains valid as the active loss-ratio branch because it is already launched and changes a concrete mechanism, but if XR-34 fails, next P0 should be XR-35 longer-budget continuation at the original `0.005/0.001/0.001` loss ratio for LR `1.75e-4` and `1.84375e-4`.
