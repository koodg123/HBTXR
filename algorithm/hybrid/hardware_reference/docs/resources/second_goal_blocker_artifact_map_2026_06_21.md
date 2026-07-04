# HGTXR-SW Second Goal Blocker Artifact Map

Date: 2026-06-21

This is a no-execute blocker map. It does not launch train, eval, or GPU jobs.

## Purpose

The current second-goal blocker is not experiment ideation. The blocker is missing
leakage-safe evidence required before XR-64A/B can run and before paper-target
comparison can be claimed. This artifact maps those blockers to exact missing
paths and validation contracts.

## Current State

| Field | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| allowed to run now | `false` |
| completion blocker count | `28` |
| XR-64 status | `generated_incomplete` |
| XR-64 ready to train | `false` |
| XR-64 can run lane | `false` |
| missing XR-64 eval rows | `8` |
| missing XR-64 override JSON files | `6` |
| XR-64 post-run candidates | `0` |
| PTB-1 payloads present | `0` |
| PTB-1 payloads missing | `5` |
| paper-target required future artifacts | `10` |
| existing required future artifacts | `3` |
| direct submission comparison allowed | `false` |

## XR-64 Missing Generated Artifacts

Teacher eval rows:

| Split | Teacher | Path |
|---|---|---|
| train | `xr62a` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json` |
| train | `xr39` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json` |
| train | `xr56b` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json` |
| train | `xr58a` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json` |
| val | `xr62a` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr62a/eval_rows.json` |
| val | `xr39` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr39/eval_rows.json` |
| val | `xr56b` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr56b/eval_rows.json` |
| val | `xr58a` | `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr58a/eval_rows.json` |

Override JSON files:

| Rule | Split | Path |
|---|---|---|
| `xr64a_conservative` | train | `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json` |
| `xr64a_conservative` | val | `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_val_overrides.json` |
| `xr64b_threshold` | train | `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json` |
| `xr64b_threshold` | val | `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_val_overrides.json` |
| `xr64c_minerror` | train | `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_train_overrides.json` |
| `xr64c_minerror` | val | `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_val_overrides.json` |

Blocked build command IDs:

- `XR64-BUILD-TRAIN`
- `XR64-BUILD-VAL`

Strict readiness still requires:

- `resume_status=ready_to_train`
- `ready_to_train=true`
- `can_run_lane=true`
- `missing_eval_rows=0`
- `missing_overrides=0`
- `leakage_risk=none`

## Paper-Target Bridge Missing Artifacts

PTB-1 evaluator payloads are schema-defined but absent while paused:

| ID | Path |
|---|---|
| `sensor_space_eval_rows` | `docs/resources/future/paper_target_bridge/sensor_space_eval_rows.jsonl` |
| `coordinate_transform_audit` | `docs/resources/future/paper_target_bridge/coordinate_transform_audit.json` |
| `hybrid_eval_rows` | `docs/resources/future/paper_target_bridge/hybrid_scheduler_eval_rows.jsonl` |
| `hybrid_metric_report` | `docs/resources/future/paper_target_bridge/hybrid_scheduler_metric_report.json` |
| `paper_frame_full_test_p1` | `docs/resources/future/paper_target_bridge/paper_frame_full_test_p1_report.json` |

PTB-2 trained-candidate payloads remain blocked until XR-64-or-later post-run
evidence exists:

- `docs/resources/future/paper_target_bridge/trained_candidate_test_eval_summary.json`
- `docs/resources/future/paper_target_bridge/trained_candidate_leakage_audit.json`

PTB-3 bridge decision remains blocked until PTB-1 and PTB-2 payloads exist.
This is a downstream decision payload, separate from the `10` required future
artifacts counted by the paper-target workplan:

- `docs/resources/future/paper_target_bridge/bridge_decision.json`

## Drift-Prone Mirrors

The checker treats these as explicit mirrors of status/contract fields:

- `completion_readiness_blocker_count=28`
- `completion_readiness_xr64_next_prep_selector_next_id=XR64-EVAL-TRAIN-XR62A`
- `xr64_command_manifest_expected_total_eval_row_records=27092`
- `xr64_command_manifest_expected_total_override_records=20319`
- `paper_target_required_future_artifact_count=10`
- `paper_target_existing_required_future_artifact_count=3`
- `paper_target_bridge_workplan_current_existing_future_artifact_count=3`
- `xr64_postrun_candidate_count=0`

## Ablation Matrix

| Axis | Current blocker | Required evidence |
|---|---|---|
| teacher model training | missing XR-64 train/val teacher eval rows and override targets | strict XR-64 readiness with no missing generated artifacts |
| head | XR-64A/B launch is gated | certified future XR-64A/B post-run eval summaries |
| loss | override loss cannot be evaluated without override JSON | future test eval summaries with overrides cleared |
| LR | low-LR XR-64A/B continuations are blocked by missing targets | future ablation LR manifest and certified summary |
| self-supervised distillation | deferred until XR-64 non-collapse | future XR-65/XR-68 evidence |
| optimizer | deferred until stronger full-width teacher | future XR-68 evidence |

## Validation

```bash
python3 -m py_compile scripts/external/check_second_goal_blocker_artifact_map.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_second_goal_blocker_artifact_map.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_second_goal_blocker_artifact_map.py tests/test_second_goal_status.py
```

Machine-readable source:
`docs/resources/second_goal_blocker_artifact_map_2026_06_21.json`.
