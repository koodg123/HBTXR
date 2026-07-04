# XR-66 Low-Similarity Targeted Recovery Results - 2026-06-22

## Purpose

Run the next experiment after the XR-66 no-train bucket comparison. The goal was to test whether training only on the trainable low-similarity subset can recover P10/P5 while preserving the XR-64C full-test center gain.

## Setup

Common setup:

- Train manifest: `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/train_manifest.jsonl` (`1624` rows)
- Val manifest: `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/val_manifest.jsonl` (`254` rows)
- Low-sim test manifest: `data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/test_manifest.jsonl` (`493` rows)
- Full test manifest: `data/_internal/manifests/manifest1/test_manifest.jsonl` (`2238` rows)
- Epochs: `50`
- Test target override: cleared
- Required checkpoints: `best_track_p10`, `best_track_p5`, `best_metric_track_center_px`
- Script: `scripts/external/run_xr66_lowsim_targeted_recovery.sh`

Lanes:

| Lane | Init | Teacher | LR | Intent |
|---|---|---|---:|---|
| XR-66A | XR-65A best-P5 | XR-39 P10 `c25p45f30` | `3e-7` | low-sim P10 recovery from the bucket-center winner |
| XR-66B | XR-64C best-P10 | XR-39 P10 `c25p45f30` | `2e-7` | low-sim center-preserving recovery from the current center baseline |

## Run Evidence

- XR-66A log: `runs/_logs/xr66_lowsim_targeted_a_gpu0_20260622_063841.log`
- XR-66B log: `runs/_logs/xr66_lowsim_targeted_b_gpu1_20260622_063844.log`
- XR-66A train root: `runs/xr66_xr65ap5init_xr39p10teacher_lowsim_p10recovery_c255000_lr3e_7_g32_hm0_004_off0_0015_c0_0040_p100_010_p50_0020_s0_00030_m160k_M384k_r4000kus_p0p5_20260622_063841`
- XR-66B train root: `runs/xr66_xr64cp10init_xr39p10teacher_lowsim_centerguard_c255000_lr2e_7_g32_hm0_004_off0_0015_c0_0050_p100_008_p50_0025_s0_00035_m160k_M384k_r4000kus_p0p5_20260622_063844`
- Both lanes reached `XR66_TRAIN_EXIT:0`.
- Both lanes reached `XR66_DONE` with `eval_count=6`.

## Full-Test Results

Promotion gates:

- Center gate: `< 16.464661524977004`
- P10 gate: `> 35.02295998845781`
- P5 gate: `> 12.133503770828247`

| Lane | Checkpoint | Center | P10 | P5 | P1 |
|---|---|---:|---:|---:|---:|
| XR-66A | `best_track_p10` | `16.709318779196057` | `34.26360624858311` | `11.964286068507603` | `0.35501702172415595` |
| XR-66A | `best_track_p5` | `16.524617418221066` | `34.58631029129028` | `11.755952746527536` | `0.7368197441101074` |
| XR-66A | `best_metric_track_center_px` | `16.973667568819863` | `34.280612986428395` | `11.367772456577846` | `0.5314626012529645` |
| XR-66B | `best_track_p10` | `16.66681088890348` | `34.30824911253793` | `11.87074865613665` | `0.35501702172415595` |
| XR-66B | `best_track_p5` | `16.498384244101388` | `34.52040895053319` | `11.909013972963606` | `0.7878401551927839` |
| XR-66B | `best_metric_track_center_px` | `16.903116943155016` | `34.27125922611781` | `11.61862280028207` | `0.34013606480189734` |

Best full-test observations:

- Best center: XR-66B `best_track_p5`, center `16.498384244101388`.
- Best P10: XR-66A `best_track_p5`, P10 `34.58631029129028`.
- Best P5: XR-66A `best_track_p10`, P5 `11.964286068507603`.

Full-test decision:

- No promotion.
- Center is worse than XR-64C by at least `0.033722719124384`.
- P10 remains below XR-39 gate by at least `0.43664969716753`.
- P5 remains below the active P5 gate by at least `0.169217702744644`.

## Low-Similarity Test Results

| Lane | Checkpoint | Center | P10 | P5 | P1 |
|---|---|---:|---:|---:|---:|
| XR-66A | `best_track_p10` | `17.64143173156246` | `34.71390262726815` | `11.21927845862604` | `0.4608295194564327` |
| XR-66A | `best_track_p5` | `17.92218274454917` | `35.42434812361194` | `11.689708525134671` | `0.4608295194564327` |
| XR-66A | `best_metric_track_center_px` | `17.431977118215254` | `36.36136802550285` | `10.599078793679514` | `0.4608295194564327` |
| XR-66B | `best_track_p10` | `17.69112765404486` | `35.184332755304155` | `11.449693187590569` | `0.4608295194564327` |
| XR-66B | `best_track_p5` | `18.002039071052305` | `34.953917995575935` | `11.920123284862887` | `0.4608295194564327` |
| XR-66B | `best_metric_track_center_px` | `17.484289407730103` | `36.15975512227705` | `10.829493553407731` | `0.4608295194564327` |

Low-sim subset decision:

- XR-66A `best_metric_track_center_px` strongly improves low-sim P10 to `36.36136802550285`.
- XR-66B `best_track_p5` gives the best low-sim P5 at `11.920123284862887`.
- These gains do not transfer to full-test promotion.

## Decision

Close XR-66A/B as no-promotion.

Hard low-similarity subset training is useful diagnostically, but it over-specializes: it improves low-sim bucket P10 while worsening full-test center and failing the global P10/P5 gates. Do not repeat hard-subset low-sim training as the next step.

Recommended next experiment:

1. Use full-manifest training, not low-sim-only training.
2. Keep XR-39 P10 as the P10 teacher signal because it wins P10 across targeted buckets.
3. Add a low-similarity loss-weight/curriculum term instead of a hard subset.
4. Preserve full-test center with XR-64C/XR-65A anchor constraints.
5. Require full-test promotion; targeted bucket gain alone is not sufficient.
