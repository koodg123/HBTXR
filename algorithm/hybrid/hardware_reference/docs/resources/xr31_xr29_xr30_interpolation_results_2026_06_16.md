# XR-31 XR-29/XR-30 Checkpoint Interpolation Results

Date: 2026-06-16

## Context

XR-31 tested no-train checkpoint interpolation between:

- Alpha `0.0`: XR-29 LR `1.75e-4` unified leader, `runs/raw_mode1_stage2_count255000_adamw_lr1_75e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr29_heatmap_centerselect_lr1p75e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_204422/train/best_metric_track_center_px.pt`.
- Alpha `1.0`: XR-30 LR `1.875e-4` center leader, `runs/raw_mode1_stage2_count255000_adamw_lr1_875e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr30_heatmap_centerselect_lr1p875e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_211629/train/best_metric_track_center_px.pt`.

Active gates before XR-31:

| Source | Center | P10 | P5 |
|---|---:|---:|---:|
| XR-30 center leader | 17.035534060001375 | 32.42474567549569 | 11.266581957680838 |
| XR-29 unified/P10/P5 leader | 17.04961508342198 | 32.56462665285383 | 11.50467722075326 |

## Method

- Runner: `scripts/external/run_xr31_xr29_xr30_heatmap_interp_eval.sh`.
- Interpolation utility: `scripts/external/interpolate_hbtxr_checkpoints.py`.
- Tensor compatibility: both checkpoints are epoch `10`, have `138` model tensors, and all keys match.
- Eval contract: same support-adaptive fixed255k heatmap-state contract as XR-29/XR-30.
- Evaluated alphas: `0.125`, `0.1875`, `0.25`, `0.50`, `0.75`.

## Test Results

| Alpha | Center | P10 | P5 | Decision |
|---:|---:|---:|---:|---|
| `0.125` | 17.04816469124385 | 32.57313004221235 | 11.488945926938738 | P10-only improvement |
| `0.1875` | 17.047394837651932 | 32.528487185069494 | 11.503826883860997 | no active gate promotion |
| `0.25` | 17.04659355367933 | 32.57950758934021 | 11.503826883860997 | P10 gate promoted |
| `0.50` | 17.043099636690958 | 32.50170144353594 | 11.37500034059797 | no active gate promotion |
| `0.75` | 17.039319237640925 | 32.5080789906638 | 11.362245225906372 | no active gate promotion |

Best XR-31 candidate:

- Alpha `0.25` improves P10 from `32.56462665285383` to `32.57950758934021`.
- Alpha `0.25` improves center relative to XR-29 from `17.04961508342198` to `17.04659355367933`.
- Alpha `0.25` does not preserve the strict P5 gate: `11.503826883860997` versus `11.50467722075326`.

## Decision

XR-31 promotes the P10 gate only. It does not replace the unified checkpoint because strict P5 is slightly lower than XR-29.

Active gates after XR-31:

- Center `<17.035534060001375` from XR-30 LR `1.875e-4`.
- P10 `>32.57950758934021` from XR-31 alpha `0.25`.
- P5 `>11.50467722075326` from XR-29 LR `1.75e-4`.

Best unified checkpoint remains XR-29 LR `1.75e-4`.

## Next Experiment

Recommended next P0:

1. Train LR midpoint `1.8125e-4` with the same XR-29/XR-30 heatmap contract, because no-train interpolation did not preserve P5.
2. Use the XR-31 alpha `0.25` checkpoint as a P10-friendly teacher or checkpoint-soup anchor only if the trained midpoint cannot improve P10.
3. Do not broaden heatmap weights or teacher architecture until the LR midpoint has been tested.

## Evidence

- `runs/_logs/xr31_xr29_xr30_heatmap_interp_alphaa0p125_gpu0_20260616_220118.log`.
- `runs/_logs/xr31_xr29_xr30_heatmap_interp_alphaa0p1875_gpu1_20260616_220122.log`.
- `runs/_logs/xr31_xr29_xr30_heatmap_interp_alphaa0p25_gpu0_20260616_215415.log`.
- `runs/_logs/xr31_xr29_xr30_heatmap_interp_alphaa0p5_gpu1_20260616_215414.log`.
- `runs/_logs/xr31_xr29_xr30_heatmap_interp_alphaa0p75_gpu1_20260616_215721.log`.
- `runs/eval_fixed255k_xr31_xr29_xr30_heatmap_interp_alphaa0p125_gpu0_w0_20260616_220119/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr31_xr29_xr30_heatmap_interp_alphaa0p1875_gpu1_w0_20260616_220124/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr31_xr29_xr30_heatmap_interp_alphaa0p25_gpu0_w0_20260616_215416/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr31_xr29_xr30_heatmap_interp_alphaa0p5_gpu1_w0_20260616_215415/eval/test/eval_summary.json`.
- `runs/eval_fixed255k_xr31_xr29_xr30_heatmap_interp_alphaa0p75_gpu1_w0_20260616_215723/eval/test/eval_summary.json`.
