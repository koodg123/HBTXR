# XR-66 Targeted Bucket Comparison - 2026-06-22

## Inputs

- Config: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- State key: `track_state`

Candidates:

- `XR-39_P10_c25p45f30`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/XR-39/eval_fixed255k_xr39_mixedleader_soup_c25p45f30_gpu1_w0_20260617_010154/eval/test/eval_rows.json`
- `XR-64C_best_track_p10`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr64_c_xr62a_minerror_teacher_target_adamw_lr3e_7_to0_0010_best_track_p10_test_gpu0_w0_20260622_053934/eval/test/eval_rows.json`
- `XR-65A_best_track_p5`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/runs/eval_fixed255k_xr65_a_xr64c_p10init_xr39teacher_threshold_recovery_adamw_lr2e_7_best_track_p5_test_gpu0_w0_20260622_061131/eval/test/eval_rows.json`

Buckets:

- `low_similarity_le0p1`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/manifests/manifest1/xr66_failure_buckets/low_similarity_le0p1/test_manifest.jsonl`
- `user45_right_session201_testonly`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/manifests/manifest1/xr66_failure_buckets/user45_right_session201_testonly/test_manifest.jsonl`
- `user42_left_session201_testonly`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/manifests/manifest1/xr66_failure_buckets/user42_left_session201_testonly/test_manifest.jsonl`
- `user42_right_session201_testonly`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software/data/_internal/manifests/manifest1/xr66_failure_buckets/user42_right_session201_testonly/test_manifest.jsonl`

## Results

### low_similarity_le0p1

| Candidate | Joined | Missing | Weighted center | Weighted P10 | Weighted P5 |
|---|---:|---:|---:|---:|---:|
| XR-65A_best_track_p5 | 493 | 0 | `18.235119` | `34.204276` | `11.638955` |
| XR-64C_best_track_p10 | 493 | 0 | `18.259654` | `34.204276` | `11.638955` |
| XR-39_P10_c25p45f30 | 493 | 0 | `18.291538` | `35.154394` | `9.976247` |

- Best weighted center: `XR-65A_best_track_p5`
- Best weighted P10: `XR-39_P10_c25p45f30`
- Best weighted P5: `XR-64C_best_track_p10`

### user45_right_session201_testonly

| Candidate | Joined | Missing | Weighted center | Weighted P10 | Weighted P5 |
|---|---:|---:|---:|---:|---:|
| XR-39_P10_c25p45f30 | 31 | 0 | `24.526918` | `6.896552` | `3.448276` |
| XR-65A_best_track_p5 | 31 | 0 | `24.610347` | `6.896552` | `3.448276` |
| XR-64C_best_track_p10 | 31 | 0 | `24.651339` | `6.896552` | `3.448276` |

- Best weighted center: `XR-39_P10_c25p45f30`
- Best weighted P10: `XR-39_P10_c25p45f30`
- Best weighted P5: `XR-39_P10_c25p45f30`

### user42_left_session201_testonly

| Candidate | Joined | Missing | Weighted center | Weighted P10 | Weighted P5 |
|---|---:|---:|---:|---:|---:|
| XR-65A_best_track_p5 | 46 | 0 | `24.245301` | `17.073171` | `7.317073` |
| XR-39_P10_c25p45f30 | 46 | 0 | `24.268204` | `17.073171` | `7.317073` |
| XR-64C_best_track_p10 | 46 | 0 | `24.288248` | `17.073171` | `7.317073` |

- Best weighted center: `XR-65A_best_track_p5`
- Best weighted P10: `XR-39_P10_c25p45f30`
- Best weighted P5: `XR-39_P10_c25p45f30`

### user42_right_session201_testonly

| Candidate | Joined | Missing | Weighted center | Weighted P10 | Weighted P5 |
|---|---:|---:|---:|---:|---:|
| XR-65A_best_track_p5 | 46 | 0 | `23.481124` | `17.948718` | `5.128205` |
| XR-64C_best_track_p10 | 46 | 0 | `23.519332` | `17.948718` | `5.128205` |
| XR-39_P10_c25p45f30 | 46 | 0 | `23.753485` | `17.948718` | `5.128205` |

- Best weighted center: `XR-65A_best_track_p5`
- Best weighted P10: `XR-39_P10_c25p45f30`
- Best weighted P5: `XR-39_P10_c25p45f30`

## Decision

- Recommended next action: `train_low_similarity_targeted_branch`
- Rationale: Low-similarity is the only trainable XR-66 targeted subset. Worst-session buckets are test-only and can only validate generalization.

Bucket win counts:

| Candidate | Center wins | P10 wins | P5 wins |
|---|---:|---:|---:|
| `XR-39_P10_c25p45f30` | 1 | 4 | 3 |
| `XR-64C_best_track_p10` | 0 | 0 | 1 |
| `XR-65A_best_track_p5` | 3 | 0 | 0 |

Execution rule:

- Do not train on the test-only worst-session manifests.
- If training is launched, use only the low-similarity train/val/test subset or a full-manifest low-similarity weighting scheme.
- Promotion still requires full-test gate improvement, not just targeted bucket improvement.
