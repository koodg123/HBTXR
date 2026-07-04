# XR-65 P10/P5 Recovery Results - 2026-06-22

## Purpose

Evaluate whether a constrained XR-65 continuation from the XR-64C center-improved checkpoint can recover P10/P5 while preserving the new center level.

## Baseline And Gates

- Stage1 frame-search baseline: Lane J self-distill center-polish, `best_search_p10.pt`.
- Stage2 center basis: XR-64C `best_track_p10.pt`.
- Current center best: `16.464661524977004`.
- Center preserve rejection threshold: `16.764661524977003`.
- P10 gate: `35.02295998845781`.
- P5 gate: `12.133503770828247`.

## Runs

| Lane | Run root | Teacher | Override | LR | Epoch result |
|---|---|---|---|---:|---|
| XR-65A | `runs/xr65_xr64c_p10init_xr39teacher_threshold_recovery_c255000_lr2e_7_g32_hm0_004_off0_0015_c0_0035_p100_008_p50_0030_to0_00045_s0_00025_m160k_M384k_r4000kus_p0p5_20260622_055744` | XR-39 P10 | XR-64B threshold train override | `2e-7` | early stop at `7/8` |
| XR-65B | `runs/xr65_xr64c_p10init_xr56bp5teacher_conservative_p5guard_c255000_lr2e_7_g32_hm0_004_off0_0015_c0_0040_p100_006_p50_0040_to0_00035_s0_00030_m160k_M384k_r4000kus_p0p5_20260622_055754` | XR-56B P5 | XR-64A conservative train override | `2e-7` | early stop at `7/8` |

Both runs completed train and all three test evals:

- `best_track_p10`
- `best_track_p5`
- `best_metric_track_center_px`

## Test Metrics

| Candidate | Center px | P10 pct | P5 pct | P1 pct |
|---|---:|---:|---:|---:|
| XR-65A best center | `16.46762662444796` | `34.123725230353216` | `12.08248336655753` | `0.9251700946262904` |
| XR-65A best P10 | `16.464880844524927` | `34.30229665211269` | `12.037840502602714` | `0.9047619342803955` |
| XR-65A best P5 | `16.467129351411547` | `34.31505176680429` | `12.08248336655753` | `0.9251700946262904` |
| XR-65B best center | `16.467705251489367` | `34.07908237321036` | `12.08248336655753` | `0.9251700946262904` |
| XR-65B best P10 | `16.46500872884478` | `34.30229665211269` | `12.037840502602714` | `0.9047619342803955` |
| XR-65B best P5 | `16.467217557770866` | `34.31505176680429` | `12.08248336655753` | `0.9251700946262904` |

## Decision

XR-65A/B are closed as no-promotion.

- Center preservation: pass relative to the `16.764661524977003` rejection threshold.
- Center promotion: fail; best XR-65 center `16.464880844524927` is worse than XR-64C `16.464661524977004`.
- P10 promotion: fail; best XR-65 P10 `34.31505176680429` is below `35.02295998845781`.
- P5 promotion: fail; best XR-65 P5 `12.08248336655753` is below `12.133503770828247`.

XR-65 did recover P5 versus XR-64C best-P10 (`12.044218056542533 -> 12.08248336655753`) but did not close the P5 gate and reduced P10.

## Interpretation

The low-LR head/adapter continuation is too conservative and the train-time target override pressure is not transferring to the held-out test P10/P5 gate. The training split metrics are strong, while test P10 remains low, so the next branch should not simply extend XR-65 at the same LR/loss recipe.

## Next Candidate Experiments

1. XR-66 confidence-gated diagnostic before training:
   - Build confidence/error buckets from XR-64C and XR-65 eval rows.
   - Identify whether P10/P5 misses concentrate in low-quality, blink, low-similarity, or subject/session buckets.
   - Do not train local fallback until the bucket report identifies a targetable failure mode.

2. XR-65C stronger but bounded recovery:
   - Init: XR-64C `best_track_p10.pt`.
   - Teacher: XR-39 P10 or no teacher for a no-distill control.
   - LR bracket: `5e-7` and `1e-6`.
   - Reduce target override weight or remove override to test whether train-target pressure is causing poor test transfer.
   - Keep center guard: reject if test center exceeds `16.764661524977003`.

3. XR-66 local-update branch only after diagnostics:
   - Train a local correction or confidence-gated fallback only for diagnosed failure buckets.
   - Require fallback-off baseline comparison and full-test eval.

## Validation Evidence

```bash
bash -n scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh
DRY_RUN=1 bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr65_p10p5_recovery_from_xr64c.sh b cuda:1
.venv/bin/python -m pytest -q tests/test_xr65_runner_contract.py tests/test_trainer_checkpoint_specs.py tests/test_xr64_runner_contract.py
```

Observed:

- Shell syntax check: passed.
- Dry-run lane A/B: passed.
- Focused pytest: `8 passed`.
- Runtime: both train runs reached `XR65_TRAIN_EXIT:0`.
- Runtime: both lanes reached `XR65_DONE` with `eval_count=3`.
- Post-run process check: no `run_xr65`, `train_hbtxr.py`, or `eval_hbtxr.py` process remained.
- Post-run GPU check: GPU0/GPU1 returned to idle.
