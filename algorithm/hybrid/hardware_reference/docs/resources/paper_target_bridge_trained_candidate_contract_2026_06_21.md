# Paper Target Bridge PTB-2 Trained-Candidate Evidence Contract

Date: 2026-06-21

Experiments are paused by user directive. This is a schema/preflight contract only.
It does not create trained-candidate evidence, run train/eval, or permit direct
comparison with `10_submission_initial`.

## Purpose

PTB-2 will eventually attach a leakage-safe XR-64-or-later trained candidate to
the paper-target bridge. While paused, the useful work is to freeze the required
fields, semantics, and rejection rules so future payloads cannot accidentally
become paper-level evidence.

## Current State

| Item | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| schema contract allowed while paused | `true` |
| trained-candidate payload allowed while paused | `false` |
| execute supported | `false` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| PTB-2 payload schemas | `2` |
| PTB-2 payloads present | `0` |
| XR-64 post-run candidates | `0` |

## Payload Schemas

| ID | Format | Path | Paused-state rule |
|---|---|---|---|
| `trained_candidate_eval_summary` | JSON | `docs/resources/future/paper_target_bridge/trained_candidate_test_eval_summary.json` | absent until XR-64-or-later full-test evidence |
| `trained_candidate_leakage_audit` | JSON | `docs/resources/future/paper_target_bridge/trained_candidate_leakage_audit.json` | absent until trained-candidate provenance audit |

## Required Semantics

- `trained_candidate_eval_summary.split` must be `test`.
- `trained_candidate_eval_summary.target_override_path` must be `null`.
- `trained_candidate_eval_summary.allow_test_target_override` must be `false`.
- `trained_candidate_eval_summary.checkpoint_kind` must be one of `best_track_p10`, `best_track_p5`, or `best_metric_track_center_px`.
- `trained_candidate_eval_summary.lane_id` must identify an XR-64-or-later trained lane.
- `metric_track_center_px`, `metric_track_p10_pct`, `metric_track_p5_pct`, and `metric_track_p1_pct` must be finite.
- P10/P5/P1 metrics must be in `[0, 100]`.
- `trained_candidate_leakage_audit.test_derived_targets_used` must be `false`.
- `trained_candidate_leakage_audit.test_override_paths` must be empty.
- `trained_candidate_leakage_audit.leakage_risk` must be `none`.
- `trained_candidate_leakage_audit.decision` must be `valid`.
- Eval summary and leakage audit must exist together in post-resume validation mode; one without the other is invalid.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_bridge_trained_candidate_contract.py
.venv/bin/python scripts/external/check_paper_target_bridge_trained_candidate_contract.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_bridge_trained_candidate_contract.py
```

Machine-readable source:
`docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.json`.
