# XR-68 Fixed-LR Low-Similarity Weighted Recovery Results - 2026-06-22

## Purpose

XR-67 showed that full-manifest low-similarity weighting was executable, but the inherited plateau scheduler reduced LR too quickly and the lane failed to promote the full-test center/P10 gates. XR-68 tested the scheduler correction directly: keep full-manifest low-sim weighting, keep XR-39 P10 teacher supervision, but disable the scheduler and run fixed-LR lanes for at least 50 epochs.

## Setup

Common setup:

- Train manifest: `data/_internal/manifests/manifest1/train_manifest.jsonl` (`5929` rows)
- Val manifest: `data/_internal/manifests/manifest1/val_manifest.jsonl` (`844` rows)
- Full test manifest: `data/_internal/manifests/manifest1/test_manifest.jsonl` (`2238` rows)
- Low-sim test manifest: `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/test_manifest.jsonl` (`493` rows)
- Epochs: `50`
- Scheduler: `none`
- Eval-time sample weighting: disabled
- Script: `scripts/external/run_xr68_fixedlr_lowsim_weighted_recovery.sh`

Promotion gates:

- Center gate: `< 16.464661524977004`
- P10 gate: `> 35.02295998845781`
- P5 gate: `> 12.133503770828247`

Lanes:

| Lane | Init | Teacher | LR | Center L2 | Low-sim weight | Intent |
|---|---|---|---:|---:|---:|---|
| XR-68A | XR-65A best-P5 | XR-39 P10 `c25p45f30` | `3e-7` | `0.0050` | `4.0` | clean fixed-LR scheduler correction |
| XR-68B | XR-65A best-P5 | XR-39 P10 `c25p45f30` | `5e-7` | `0.0060` | `4.0` | higher-pressure fixed-LR bracket |

An earlier draft launch used XR-67B as the XR-68B seed. It was interrupted at the beginning after GPT-5.5 read-only review identified that seed as center-regressed and not suitable for the main clean bracket. The completed runs above are the corrected XR-65A-seeded A/B bracket.

## Run Evidence

- GPT-5.5 read-only sub-agent: `Huygens the 2nd`
- XR-68A log: `runs/_logs/xr68_full_lowsim_weighted_a_gpu0_20260622_090550.log`
- XR-68B log: `runs/_logs/xr68_full_lowsim_weighted_b_gpu1_20260622_090552.log`
- XR-68A train root: `runs/xr68_xr65ap5init_xr39p10teacher_fixedlr3e7_full_lowsimw4_clean_c255000_lr3e_7_g32_hm0_004_off0_0015_c0_0050_p100_010_p50_0020_s0_00030_lsw4_0_m160k_M384k_r4000kus_p0p5_20260622_090551`
- XR-68B train root: `runs/xr68_xr65ap5init_xr39p10teacher_fixedlr5e7_full_lowsimw4_highpressure_c255000_lr5e_7_g32_hm0_004_off0_0015_c0_0060_p100_010_p50_0020_s0_00030_lsw4_0_m160k_M384k_r4000kus_p0p5_20260622_090553`
- Both lanes reached `XR68_TRAIN_EXIT:0`.
- Both lanes reached `XR68_DONE` with `eval_count=6`.
- Validation P10 improved compared with XR-67:
  - XR-68A reached best validation P10 `25.0809` at epoch `43`.
  - XR-68B reached best validation P10 `25.0809` at epoch `24`.
  - XR-67A/B validation P10 had plateaued around `24.3980` and `24.3587`.
- LR stayed fixed:
  - XR-68A LR stayed `3.00e-07`.
  - XR-68B LR stayed `5.00e-07`.

## Full-Test Results

| Lane | Checkpoint | Center | P10 | P5 | P1 |
|---|---|---:|---:|---:|---:|
| XR-68A | `best_track_p10` | `16.558039666925158` | `34.47534091813224` | `12.11224525996617` | `0.602891172681536` |
| XR-68A | `best_track_p5` | `16.51572412933622` | `34.53528990745544` | `12.039966358457292` | `0.8018707615988595` |
| XR-68A | `best_metric_track_center_px` | `16.500933163506645` | `34.645834132603234` | `12.016581991740635` | `0.9039115837642124` |
| XR-68B | `best_track_p10` | `16.553961242948258` | `34.40901436805725` | `12.09736430304391` | `0.5454932076590402` |
| XR-68B | `best_track_p5` | `16.525990513392856` | `34.60331711769104` | `11.988945954186576` | `0.6857993330274309` |
| XR-68B | `best_metric_track_center_px` | `16.517828347001757` | `34.53528990745544` | `12.016581998552596` | `0.9039115837642124` |

Best full-test observations:

- Best center: XR-68A `best_metric_track_center_px`, center `16.500933163506645`, worse than the center gate by `0.03627163852964088`.
- Best P10: XR-68A `best_metric_track_center_px`, P10 `34.645834132603234`, below the P10 gate by `0.3771258558545796`.
- Best P5: XR-68A `best_track_p10`, P5 `12.11224525996617`, below the P5 gate by `0.02125851086207753`.
- The best-P5 checkpoint has center `16.558039666925158`, worse than the center gate by `0.09337814194815408`.

## Low-Similarity Test Results

| Lane | Checkpoint | Center | P10 | P5 | P1 |
|---|---|---:|---:|---:|---:|
| XR-68A | `best_track_p10` | `17.783315843151463` | `34.04185944218789` | `11.991167837573636` | `0.23041475972821635` |
| XR-68A | `best_track_p5` | `17.949691718624486` | `34.99232046065792` | `11.651306060052686` | `0.7296467442666331` |
| XR-68A | `best_metric_track_center_px` | `18.013677612427742` | `35.45315001087804` | `11.920123315626576` | `0.7296467442666331` |
| XR-68B | `best_track_p10` | `17.802674162772394` | `34.04185947295158` | `11.76075307784542` | `0.0` |
| XR-68B | `best_track_p5` | `17.91902965884055` | `34.99232046065792` | `11.651306060052686` | `0.4608295194564327` |
| XR-68B | `best_metric_track_center_px` | `17.97900842851208` | `35.45315001087804` | `11.920123346390262` | `0.7296467442666331` |

Best low-sim observations:

- Best low-sim center: XR-68A `best_track_p10`, center `17.783315843151463`.
- Best low-sim P10: XR-68A/B `best_metric_track_center_px`, P10 `35.45315001087804`.
- Best low-sim P5: XR-68A `best_track_p10`, P5 `11.991167837573636`.

## Decision

Close XR-68A/B as no-promotion.

Fixed LR solved the training-dynamics issue observed in XR-67: validation P10 improved from the XR-67 plateau to `25.0809`, and LR stayed constant through 50 epochs. However, the validation improvement did not transfer to full-test promotion. XR-68 does not beat the active center, P10, or P5 gates.

## Next Experiment

Do not extend XR-68 to 200 epochs as-is. The full-test result shows that fixed LR alone is insufficient.

Recommended next branch:

1. Stop repeating full-manifest low-sim weighting without a new signal.
2. Use XR-68 as evidence that scheduler was a real issue, but not the only issue.
3. Move to either:
   - center-preserving P5/P10 checkpoint soup using XR-64C/XR-67/XR-68 candidates, or
   - a new target construction branch that explicitly optimizes full-test-like P10/P5 without low-sim overemphasis.
4. If training continues, add a stronger center anchor or direct center-teacher target, because P5/P10 recovery still pushes center above the strict gate.
