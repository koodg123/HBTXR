# XR-46 Center/P10 Compatibility Soup Plan

Date: 2026-06-17

## Goal

Recover P10/P5 without losing the XR-43 center leader.

Active gates:

- Center: `<16.491429926667895`
- P10: `>35.02295998845781`
- P5: `>11.868197652271816`

## Rationale

XR-44 mixed XR-43 center, XR-39 P10, and XR-41A P5, but the P5 anchor was not checkpoint-compatible enough with the current center/P10 basin. XR-45B did not promote, but its best-P5 checkpoint came from full-manifest training initialized from XR-43, so it is a better compatibility anchor than subset-only XR-45A or the older XR-41A P5 checkpoint.

Subject `39` is test-only in the current manifest split; direct subject-specific training remains rejected as leakage risk.

## Lanes

### Lane A: XR43 center to XR39 P10 interpolation

Inputs:

- A: `runs/interpolated_checkpoints/xr43_xr39center_xr42ca_center_interp_a0_5.pt`
- B: `runs/interpolated_checkpoints/xr39_mixedleader_soup_c25p45f30.pt`

Alphas:

- `0.125`
- `0.25`
- `0.50`
- `0.75`
- `0.875`

Purpose:

- Check whether the XR43 center rescue and XR39 P10 leader have a P10 overshoot region.
- This is not expected to improve P5; it is a cheap P10 diagnostic.

### Lane B: compatibility P5 soup

Inputs:

- Center: XR43 alpha `0.5`
- P10: XR39 `c25p45f30`
- Compatible P5: XR45B best-P5
- Gate P5: XR41A best-P5

Weights:

- `0.70,0.20,0.05,0.05`
- `0.60,0.25,0.10,0.05`
- `0.55,0.30,0.10,0.05`
- `0.50,0.35,0.10,0.05`
- `0.45,0.40,0.10,0.05`

Purpose:

- Retain a small amount of XR41A P5 gate signal.
- Use XR45B best-P5 as a compatibility bridge to reduce the XR44-style mismatch.

## Validation

Script:

```bash
bash scripts/external/run_xr46_center_p10_compat_soup_eval.sh a cuda:0
bash scripts/external/run_xr46_center_p10_compat_soup_eval.sh b cuda:1
```

Promotion requires beating at least one active gate on full test eval:

- Center lower than `16.491429926667895`
- P10 higher than `35.02295998845781`
- P5 higher than `11.868197652271816`

If no gate promotes, mark checkpoint-space search around these anchors exhausted and move to full-manifest calibration/training.
