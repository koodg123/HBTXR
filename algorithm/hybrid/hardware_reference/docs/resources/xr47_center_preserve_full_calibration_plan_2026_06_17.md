# XR-47 Center-Preserve Full-Manifest Calibration Plan

Date: 2026-06-17

## Goal

Run a trainable full-manifest follow-up after XR-46 exhausted useful no-train checkpoint-space sweeps.

Active gates:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

## Rationale

XR-45 showed that LR `1e-5` full-manifest heatmap refresh from XR-43 can approach P5 but drifts center too far. XR-46 showed no-train interpolation/soup cannot recover enough P10/P5 without center drift. XR-47 therefore keeps the XR-43 center leader as both init and teacher, uses a lower LR, and enables tiny state distillation as a center-preservation regularizer.

Subject `39` remains test-only in the current split, so no subject-specific training is used.

## Lanes

Shared settings:

- Init: `runs/interpolated_checkpoints/xr43_xr39center_xr42ca_center_interp_a0_5.pt`
- Teacher: same XR-43 center checkpoint
- Train/val: full manifest
- Event count: fixed/adaptive `255000`, min `160000`, max `384000`, reference `4000003us`, power `0.5`
- LR: `2e-6`
- Epochs: `8`
- Optimizer: AdamW
- Head: track center heatmap state, grid `32`
- Loss: heatmap `0.004`, offset `0.0015`, center L2 `0.0025`
- Distillation: state similarity enabled, state weight `0.0005`; feature/prediction/KD/RKD disabled

Lane A:

- Best metric: `metric_track_p10_pct`
- Purpose: P10 calibration while preserving center.

Lane B:

- Best metric: `metric_track_p5_pct`
- Purpose: P5 calibration while preserving center.

## Execution

```bash
bash scripts/external/run_xr47_center_preserve_full_calibration.sh a cuda:0
bash scripts/external/run_xr47_center_preserve_full_calibration.sh b cuda:1
```

## Promotion Rule

Promote only if full-test eval beats one active gate:

- Center lower than `16.491429926667895`
- P10 higher than `35.02295998845781`
- P5 higher than `11.868197652271816`

If XR-47 fails, next branch should change mechanism rather than only tune LR: e.g. explicit calibration head, teacher retraining, or loss term that directly optimizes the P10/P5 boundary while measuring center drift.
