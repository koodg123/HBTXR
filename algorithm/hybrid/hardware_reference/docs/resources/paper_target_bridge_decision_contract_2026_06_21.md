# Paper Target Bridge PTB-3 Bridge-Decision Contract

Date: 2026-06-21

This is a no-execute contract for the final paper-target bridge decision. It
does not create `bridge_decision.json`, does not run train/eval/GPU work, and
does not permit direct comparison with `10_submission_initial`.

## Purpose

PTB-3 is the final allow/block decision after PTB-0 source notes, PTB-1
paper-frame evaluator payloads, and PTB-2 trained-candidate evidence exist.
While experiments are paused, only the schema and rejection rules are allowed.

## Current State

| Item | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| schema allowed while paused | `true` |
| bridge-decision payload allowed while paused | `false` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| required evidence items | `7` |
| required future artifacts | `10` |
| existing required future artifacts | `3` |
| PTB-1 payloads present | `0` |
| PTB-2 payloads present | `0` |
| XR-64 post-run candidates | `0` |

## Future Output Schema

| Output | Format | Path | Current status |
|---|---|---|---|
| `bridge_decision` | JSON | `docs/resources/future/paper_target_bridge/bridge_decision.json` | absent until all required evidence exists |

Required fields:

- `direct_submission_comparison_allowed`
- `paper_level_completion_allowed`
- `required_evidence_complete`
- `metric_protocol_unblocked`
- `trained_candidate_valid`
- `remaining_blockers`
- `decision`

Required context fields:

- `paper_target_manifest_path`
- `metric_protocol_unblock_contract_path`
- `trained_candidate_eval_summary_path`
- `trained_candidate_leakage_audit_path`
- `ptb1_payload_paths`
- `source_status_report_path`

## Signoff Rules

- Direct submission comparison may be true only if every required evidence item is complete.
- Paper-level completion may be true only if direct submission comparison is true.
- `decision` must remain blocked while `remaining_blockers` is non-empty.
- Metric/protocol unblock contract must allow direct comparison before PTB-3 can allow direct comparison.
- Trained-candidate leakage audit must report `leakage_risk=none`.
- PTB-1 evaluator payloads and PTB-2 trained-candidate payloads must be present together before PTB-3 can allow direct comparison.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_bridge_decision_contract.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_paper_target_bridge_decision_contract.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_bridge_decision_contract.py tests/test_second_goal_status.py
```

Machine-readable source:
`docs/resources/paper_target_bridge_decision_contract_2026_06_21.json`.
