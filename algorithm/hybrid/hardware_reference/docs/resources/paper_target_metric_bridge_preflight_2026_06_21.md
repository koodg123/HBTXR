# Paper Target Metric Bridge Preflight

Date: 2026-06-21

This is a no-execute preflight contract. It freezes what must be true before HGTXR-SW metrics can be compared with the `10_submission_initial` hybrid accuracy table.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Objective | Convert metric/protocol blockers into future evaluator preflight gates. |
| Target | HGTXR-SW `manifest1` paper-target bridge evaluator |
| Verification | `.venv/bin/python scripts/external/check_paper_target_metric_bridge_preflight.py --format summary` |

## Current State

| Field | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| bridge status | `blocked_until_metric_frame_and_protocol_match` |
| metric frame | `post-transform input coordinate frame` |
| directly sensor px | `false` |
| hybrid match status | `not_proven` |
| current P1 metric available | `true` |
| current P1 full-test paper-frame evidence | `false` |
| required evidence count | `7` |
| PTB-1 evaluator schemas | `5` |
| PTB-1 existing payloads | `0` |
| XR-64 ready to train | `false` |

## Preflight Requirements

| ID | Status | Required outputs |
|---|---|---|
| coordinate_frame_match | missing | `sensor_space_eval_rows`, `coordinate_transform_audit` |
| hybrid_scheduler_eval | missing | `hybrid_eval_rows`, `hybrid_metric_report` |
| p1_full_test_metric | partial | `paper_frame_full_test_p1` |
| split_protocol_match | partial | `split_protocol_audit` |
| target_definition_match | partial | `cur_state_mapping_audit` |
| latency_separation | partial | `accuracy_latency_separation_note` |
| xr64_or_later_trained_result | missing | `trained_candidate_eval_summary`, `trained_candidate_leakage_audit` |

## Rejection Rules

- Direct comparison remains forbidden while any preflight requirement is missing or partial.
- Paper-level completion remains forbidden without a leakage-safe trained XR-64-or-later candidate.
- Coordinate-frame claims require paired transform-space and raw sensor-space rows.
- Hybrid claims require chosen-mode rows.
- P1 claims require full-test paper-frame evidence.
- XR-63 oracle cannot be promoted as a trained result.
- Validation commands must not launch train/eval/GPU work.

Machine-readable source: `docs/resources/paper_target_metric_bridge_preflight_2026_06_21.json`.
