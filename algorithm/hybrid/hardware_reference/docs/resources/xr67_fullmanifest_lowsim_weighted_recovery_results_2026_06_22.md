# XR-67 Full-Manifest Low-Similarity Weighted Recovery Results - 2026-06-22

## Purpose

XR-66 showed that hard low-similarity subset training can improve the targeted low-sim bucket, but does not transfer to the full test distribution. XR-67 tested the safer alternative: train on the full manifest while applying extra loss weight to low-similarity samples.

The goal was to recover P10/P5 while preserving the XR-64C center gate.

## Setup

Common setup:

- Train manifest: `data/_internal/manifests/manifest1/train_manifest.jsonl` (`5929` rows)
- Val manifest: `data/_internal/manifests/manifest1/val_manifest.jsonl` (`844` rows)
- Full test manifest: `data/_internal/manifests/manifest1/test_manifest.jsonl` (`2238` rows)
- Low-sim test manifest: `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/test_manifest.jsonl` (`493` rows)
- Epochs: `50`
- Required checkpoints: `best_track_p10`, `best_track_p5`, `best_metric_track_center_px`
- Eval-time sample weighting: disabled
- Script: `scripts/external/run_xr67_fullmanifest_lowsim_weighted_recovery.sh`

Promotion gates:

- Center gate: `< 16.464661524977004`
- P10 gate: `> 35.02295998845781`
- P5 gate: `> 12.133503770828247`

Lanes:

| Lane | Init | Teacher | LR | Low-sim weight | Intent |
|---|---|---|---:|---:|---|
| XR-67A | XR-64C best-P10 | XR-39 P10 `c25p45f30` | `2e-7` | `3.0` | preserve center while adding low-sim pressure |
| XR-67B | XR-65A best-P5 | XR-39 P10 `c25p45f30` | `3e-7` | `4.0` | recover P10/P5 from stronger P5 seed |

## Run Evidence

- XR-67A log: `runs/_logs/xr67_full_lowsim_weighted_a_gpu0_20260622_072844.log`
- XR-67B log: `runs/_logs/xr67_full_lowsim_weighted_b_gpu1_20260622_072853.log`
- XR-67A train root: `runs/xr67_xr64cp10init_xr39p10teacher_full_lowsimw3_centerguard_c255000_lr2e_7_g32_hm0_004_off0_0015_c0_0050_p100_008_p50_0025_s0_00035_lsw3_0_m160k_M384k_r4000kus_p0p5_20260622_072845`
- XR-67B train root: `runs/xr67_xr65ap5init_xr39p10teacher_full_lowsimw4_p10recover_c255000_lr3e_7_g32_hm0_004_off0_0015_c0_0045_p100_010_p50_0020_s0_00030_lsw4_0_m160k_M384k_r4000kus_p0p5_20260622_072854`
- Both lanes reached `XR67_TRAIN_EXIT:0`.
- Both lanes reached `XR67_DONE` with `eval_count=6`.
- Validation P10 plateaued early:
  - XR-67A best validation P10 stayed at `24.3980` from epoch `1`.
  - XR-67B best validation P10 reached `24.3587` at epoch `7`.
- The LR scheduler reduced LR aggressively:
  - XR-67A reached `1.25e-8` by the later training phase.
  - XR-67B reached `1.87e-8` by the later training phase.

## Full-Test Results

| Lane | Checkpoint | Center | P10 | P5 | P1 |
|---|---|---:|---:|---:|---:|
| XR-67A | `best_track_p10` | `16.48420093229839` | `34.39795995439802` | `12.118622813905988` | `1.0654762199946812` |
| XR-67A | `best_track_p5` | `16.48420093229839` | `34.39795995439802` | `12.118622813905988` | `1.0654762199946812` |
| XR-67A | `best_metric_track_center_px` | `16.48420093229839` | `34.39795995439802` | `12.118622813905988` | `1.0654762199946812` |
| XR-67B | `best_track_p10` | `16.545599697317396` | `34.47151440892901` | `12.191752079554966` | `0.6717687266213553` |
| XR-67B | `best_track_p5` | `16.515732836723327` | `34.53528990745544` | `12.039966358457292` | `0.8018707615988595` |
| XR-67B | `best_metric_track_center_px` | `16.50094587121691` | `34.645834132603234` | `12.016581991740635` | `0.9039115837642124` |

Best full-test observations:

- Best center: XR-67A, center `16.48420093229839`, worse than the center gate by `0.019539407321385482`.
- Best P10: XR-67B `best_metric_track_center_px`, P10 `34.645834132603234`, below the P10 gate by `0.3771258558545796`.
- Best P5: XR-67B `best_track_p10`, P5 `12.191752079554966`, above the P5 gate by `0.0582483087267196`.
- The P5-improving checkpoint has center `16.545599697317396`, worse than the center gate by `0.08093817234039236`.

## Low-Similarity Test Results

| Lane | Checkpoint | Center | P10 | P5 | P1 |
|---|---|---:|---:|---:|---:|
| XR-67A | `best_track_p10` | `18.072367114405477` | `34.982719882842034` | `11.920123346390262` | `0.9600615039948495` |
| XR-67A | `best_track_p5` | `18.072367114405477` | `34.982719882842034` | `11.920123346390262` | `0.9600615039948495` |
| XR-67A | `best_metric_track_center_px` | `18.072367114405477` | `34.982719882842034` | `11.920123346390262` | `0.9600615039948495` |
| XR-67B | `best_track_p10` | `17.875147019663164` | `34.81950938317083` | `11.76075307784542` | `0.691244279184649` |
| XR-67B | `best_track_p5` | `17.949554128031576` | `34.99232046065792` | `11.651306060052686` | `0.7296467442666331` |
| XR-67B | `best_metric_track_center_px` | `18.01349646814408` | `35.45315001087804` | `11.920123315626576` | `0.7296467442666331` |

Best low-sim observations:

- Best low-sim center: XR-67B `best_track_p10`, center `17.875147019663164`.
- Best low-sim P10: XR-67B `best_metric_track_center_px`, P10 `35.45315001087804`.
- Best low-sim P5: XR-67A, P5 `11.920123346390262`.
- XR-67 low-sim P10 is `0.9082180146248078` below XR-66A targeted best P10 `36.36136802550285`.
- XR-67 low-sim P5 is effectively tied with XR-66B targeted best P5.

## Decision

Close XR-67A/B as no overall baseline promotion.

XR-67 does improve the full-test P5 gate with XR-67B `best_track_p10`, but the improvement comes with a center regression large enough to reject it as the main baseline. XR-67 also fails to beat the active full-test center and P10 gates, and its best low-sim P10 is below the XR-66 targeted low-sim result.

## Next Experiment

Do not repeat the exact XR-67 recipe. The main issue is not just the low-sim weighting idea; the LR scheduler removed training pressure too early, and the teacher/anchor losses did not move validation P10 after the first few epochs.

Recommended XR-68 direction:

1. Keep full-manifest training and low-sim weighting.
2. Remove aggressive LR decay or set a much higher `min_lr`.
3. Start from XR-67B or XR-65A if targeting P5/P10, but require an explicit center guard.
4. Treat XR-67B `best_track_p10` as a P5-recovery candidate, not as a promoted baseline.
5. Run a small 50-epoch bracket before any 200-epoch extension:
   - one lane with fixed LR `3e-7`,
   - one lane with fixed LR `5e-7` and stronger center L2,
   - both with eval-time weighting disabled.
