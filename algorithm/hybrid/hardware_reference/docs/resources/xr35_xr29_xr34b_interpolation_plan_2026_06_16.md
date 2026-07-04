# XR-35 XR-29/XR-34B Heatmap Interpolation Plan - 2026-06-16

## Purpose

XR-34B best-P10 is the new center/P10 leader, but it misses the XR-29 P5 gate. XR-35 is a no-train checkpoint interpolation preflight that tests whether a small move from the XR-29 P5/unified anchor toward XR-34B can preserve P5 while improving center and P10.

## Inputs

- Checkpoint A, alpha `0`: XR-29 LR `1.75e-4` best-center, test `17.04961508342198 / 32.56462665285383 / 11.50467722075326`.
- Checkpoint B, alpha `1`: XR-34B best-P10, test `16.53321223940168 / 33.77168447630746 / 11.276786088943481`.
- Runner: `scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh`.
- Eval contract: fixed255k support-adaptive event count, `160000/255000/384000`, `reference_us=4000003`, `scale_power=0.5`, heatmap-state grid `32`.

## Initial Sweep

Use GPU1 by default for the no-train eval sweep:

```bash
bash scripts/external/run_xr35_xr29_xr34b_heatmap_interp_eval.sh cuda:1 \
  0.03125:a0p03125 \
  0.0625:a0p0625 \
  0.09375:a0p09375 \
  0.125:a0p125
```

## Promotion Gate

- Center must beat `<16.53321223940168` for center promotion.
- P10 must beat `>33.77168447630746` for P10 promotion.
- P5 must beat or strictly preserve `>11.50467722075326` for P5/unified promotion.

## Decision Logic

- If a small alpha keeps P5 above XR-29 while improving center/P10, select it as the new unified checkpoint.
- If no alpha preserves P5, use XR-34B as center/P10 leader and keep XR-29 as P5/unified anchor.
- If all small alphas are monotonic P5 regressions, next trainable branch should use XR-34B center/P10 as teacher and XR-29 as P5 anchor instead of another no-train interpolation.

## Results

All four alpha evals completed with exit `0`; no active gate promoted.

| Alpha | Center px | P10 | P5 | Decision |
|---:|---:|---:|---:|---|
| `0.03125` | 17.016330581051964 | 32.47534093856812 | 11.334609195164271 | no promotion |
| `0.0625` | 16.983979083810535 | 32.468963384628296 | 11.391156809670585 | no promotion |
| `0.09375` | 16.952843945366997 | 32.42432054110936 | 11.420918709891183 | no promotion |
| `0.125` | 16.923010180677686 | 32.352041605540684 | 11.255102368763515 | no promotion |

Evidence:

- Logs: `runs/_logs/xr35_xr29_xr34b_heatmap_interp_alphaa*_gpu1_20260616_*.log`
- Summaries: `runs/eval_fixed255k_xr35_xr29_xr34b_heatmap_interp_alpha*/eval/test/eval_summary.json`

Decision:

- XR-35 improved center versus XR-29, but did not beat XR-34B center or P10.
- XR-35 did not restore the XR-29 P5 gate.
- Active gates remain center `<16.53321223940168`, P10 `>33.77168447630746`, P5 `>11.50467722075326`.
- Next branch should be trainable P5-anchor continuation, not another no-train interpolation sweep.
