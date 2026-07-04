# XR-28 Heatmap LR Refinement Results

Date: 2026-06-16

## Objective

Refine the XR-27 track-center heatmap-state representation after consolidation diagnostics showed strong low/mid-similarity gains but remaining high-similarity and subject `39` risks.

Active pre-XR-28 gates:

- Center: `<17.274589475563594`
- P10: `>31.915391901561193`
- P5: `>10.289966331209456`

## Setup

- Base config: `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- Runner: `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`
- Init/teacher: XR-06C best-P10 `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`
- Event contract: fixed255k with support-adaptive count enabled, min/base/max `160000/255000/384000`, reference `4000003us`, power `0.5`
- Heatmap losses: heatmap `0.005`, offset `0.001`, center L2 `0.001`
- Distillation correction: `distillation.state_similarity=false`
- Selection: `XR27_BEST_METRIC_NAME=metric_track_center_px`, `XR27_SCHEDULER_METRIC_NAME=metric_track_center_px`

## Commands

```bash
XR27_BEST_METRIC_NAME=metric_track_center_px \
XR27_SCHEDULER_METRIC_NAME=metric_track_center_px \
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh \
  255000 160000 384000 4000003 0.5 7e-5 cuda:0 \
  0.005 0.001 0.001 32 xr28_heatmap_centerselect_lr7e5
```

```bash
XR27_BEST_METRIC_NAME=metric_track_center_px \
XR27_SCHEDULER_METRIC_NAME=metric_track_center_px \
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh \
  255000 160000 384000 4000003 0.5 1.5e-4 cuda:1 \
  0.005 0.001 0.001 32 xr28_heatmap_centerselect_lr1p5e4
```

## Results

| Branch | Checkpoint | Val epoch | Test center px | Test P10 pct | Test P5 pct | Decision |
|---|---|---:|---:|---:|---:|---|
| LR `7e-5` | best-P10 | 7 | 17.935810264519283 | 28.53273880141122 | 10.105442489896502 | no promotion |
| LR `7e-5` | best-P5 | 2 | 19.03988778250558 | 23.74872510773795 | 7.972789362498692 | no promotion |
| LR `7e-5` | best-center | 10 | 17.55661221402032 | 31.095664044788904 | 9.788265630177088 | no promotion |
| LR `1.5e-4` | best-P10 | 10 | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 | promotes all gates |
| LR `1.5e-4` | best-P5 | 10 | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 | same promoted checkpoint |
| LR `1.5e-4` | best-center | 10 | 17.08485197339739 | 32.081208263124736 | 10.502551344462804 | same promoted checkpoint |

## Promoted Leader

XR-28 LR `1.5e-4` is the new active software leader:

```text
runs/raw_mode1_stage2_count255000_adamw_lr1_5e_4_weakdistill_trackonly_heatmapstate_g32_hm0_005_off0_001_c0_001_xr28_heatmap_centerselect_lr1p5e4_min160k_max384k_ref4000kus_pow0p5_headonly_fullwidth_20260616_200229/train/best_metric_track_center_px.pt
```

Equivalent test result is produced by `best_track_p10.pt`, `best_track_p5.pt`, and `best_metric_track_center_px.pt` because all three were saved at epoch `10`.

New active gates:

- Center: `<17.08485197339739`
- P10: `>32.081208263124736`
- P5: `>10.502551344462804`

## Interpretation

LR `1.5e-4` improves all three gates over XR-27A while retaining the same heatmap-state representation and disabled state distillation. LR `7e-5` under-trains relative to XR-27A/XR-28 LR `1.5e-4`, especially on P10.

Next bounded action should not return to scalar boundary/LR-only polish. Prefer XR-29:

1. Reproduce LR `1.5e-4` with a longer budget, using early checkpoint tracking.
2. Narrow around LR `1.25e-4` and `1.75e-4` if compute is available.
3. Run post-XR-28 failure buckets, especially high-similarity `>0.9` and subject `39`.
4. Only after bucket analysis, test heatmap weight or teacher-quality changes.
