# XR-37 XR-36B/XR-36A Interpolation Plan - 2026-06-16

## Purpose

XR-36 split the leaders: XR-36B best-P10 owns center, while XR-36A best-P10 owns P5 and is the best practical P10/P5 checkpoint. XR-37 is a no-train interpolation preflight to test whether a single checkpoint can preserve XR-36B center while moving toward XR-36A P5/P10.

## Inputs

- Checkpoint A, alpha `0`: XR-36B best-P10, test `16.53305721793856 / 34.19387831687927 / 11.502551344462804`.
- Checkpoint B, alpha `1`: XR-36A best-P10, test `16.59279990025929 / 34.39710958344596 / 11.738095617294311`.
- Runner: `scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh`.
- Eval contract: fixed255k support-adaptive event count, `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap-state grid `32`.

## Initial Sweep

```bash
bash scripts/external/run_xr37_xr36b_xr36a_interp_eval.sh cuda:1 \
  0.10:a0p10 \
  0.20:a0p20 \
  0.35:a0p35 \
  0.50:a0p50
```

## Gate

- Center promotion before XR-37: `<16.53305721793856`
- P10 promotion: `>34.74064704350063`
- P5 promotion: `>11.738095617294311`

## Results

| Alpha | Test center px | Test P10 % | Test P5 % | Decision |
|---:|---:|---:|---:|---|
| 0.10 | 16.52253861086709 | 34.062075574057445 | 11.574830266407558 | promotes center only |
| 0.20 | 16.51475806917463 | 34.28826605933053 | 11.44387788772583 | promotes center only |
| 0.35 | 16.50803507396153 | 34.23086808749608 | 11.292517362322126 | promotes center only |
| 0.50 | 16.507612899371555 | 34.33205857958112 | 11.539116007941109 | promotes center only; best XR-37 center |

## Decision

- XR-37 alpha `0.50` is the new strict center leader.
- XR-36A best-P5 remains the strict P10 leader at `34.74064704350063`.
- XR-36A best-P10 remains the strict P5 leader at `11.738095617294311`.
- XR-37 alpha `0.50` is not a unified replacement because it improves center but remains below the strict P10/P5 gates.
