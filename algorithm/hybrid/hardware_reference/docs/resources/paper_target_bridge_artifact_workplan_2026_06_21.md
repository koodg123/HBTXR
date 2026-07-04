# Paper Target Bridge Artifact Workplan

Date: 2026-06-21

Experiments are paused by user directive. This workplan orders the future
paper-target bridge artifacts without launching train/eval/GPU jobs.

## Purpose

The paper-target evidence manifest defines `10` future artifacts required before
current HGTXR-SW metrics can be compared directly with the
`10_submission_initial` targets. This workplan separates them into safe
no-execute work and post-resume work.

## Current State

| Field | Value |
|---|---:|
| evidence state | `missing_required_evidence` |
| future required artifacts | `10` |
| current supporting artifacts | `4` |
| existing future artifacts | `3` |
| XR-64 ready to train | `false` |
| missing XR-64 eval rows | `8` |
| missing XR-64 overrides | `6` |
| XR-64 post-run candidates | `0` |
| direct submission comparison allowed | `false` |

## Work Packages

| ID | Priority | Allowed while paused | Requires train/eval | Artifacts | Role |
|---|---:|---|---|---|---|
| `PTB-0-DOC-PREFILL` | P0 | true | false | `split_protocol_audit`, `cur_state_mapping_audit`, `accuracy_latency_separation_note` | paper/manifest/source notes that can be drafted without execution |
| `PTB-1-EVALUATOR-SCHEMA` | P0 | true | false | `sensor_space_eval_rows`, `coordinate_transform_audit`, `hybrid_eval_rows`, `hybrid_metric_report`, `paper_frame_full_test_p1` | schema and rejection rules now; actual rows/reports after resume |
| `PTB-2-XR64-TRAINED-CANDIDATE` | P0 | false | true | `trained_candidate_eval_summary`, `trained_candidate_leakage_audit` | future leakage-safe XR-64-or-later result evidence; must pass `paper_target_bridge_trained_candidate_contract` |
| `PTB-3-BRIDGE-DECISION` | P1 | false | false | `bridge_decision` | final allow/block decision after all required evidence exists |

## Immediate Safe Actions

These are allowed while experiments remain paused:

1. Keep `split_protocol_audit.json` synchronized with manifest1 and the paper split claims.
2. Keep `cur_state_target_mapping_audit.json` synchronized with current dataset target conventions.
3. Keep `accuracy_latency_separation_note.json` synchronized to separate software accuracy and hardware latency evidence.
4. Keep sensor-space rows, hybrid eval rows, full-test P1 reports, and trained-candidate artifacts missing until explicit experiment/evaluation resume.

## Guardrails

- Do not set `execute_supported=true` while experiments are paused.
- Do not allow direct submission comparison while future artifacts are missing.
- Do not mark `PTB-2-XR64-TRAINED-CANDIDATE` as allowed while paused.
- Do not claim paper-level completion from schema-only artifacts.
- Do not use validation commands that launch train/eval/GPU jobs.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_bridge_artifact_workplan.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_paper_target_bridge_artifact_workplan.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_bridge_artifact_workplan.py tests/test_second_goal_status.py
```

Machine-readable source:
`docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json`.
