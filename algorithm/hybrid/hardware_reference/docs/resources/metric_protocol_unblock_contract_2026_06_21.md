# HGTXR-SW Metric/Protocol Unblock Contract

Date: 2026-06-21

Experiments are paused by user directive. This document does not run training or evaluation. It defines the evidence required before current software metrics may be compared directly with the `10_submission_initial` paper target.

## Prompt Pack

| Field | Value |
|---|---|
| Objective | define the smallest verification contract that can unblock paper-target metric comparison |
| Source artifacts | `second_goal_metric_protocol_bridge_2026_06_20`, `second_goal_submission_target_gap_audit_2026_06_20`, `second_goal_current_result_synthesis_2026_06_20` |
| IDEA-Gen phase | P1 |
| Domain | Research Workflow |
| Parent skill | `paper-idea-generator` |
| Target model/dataset | HGTXR-SW mode1 raw event-count tracker, manifest1 |
| Required artifact | blocked unblock contract plus checker/test coverage |
| Concrete evidence inventory | `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md` |
| Verification evidence | JSON contract, checker, pytest, status reporter |

## Current Decision

| Item | State |
|---|---:|
| contract status | `blocked` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| software promotion metric space | current center/P10/P5 gates only |

The current software results can still be promoted against the active software gates, but they cannot be used as paper-level closure evidence until this contract is unblocked.

## Current Gates And Targets

| Metric | Current software gate | Submission target | Direct comparison |
|---|---:|---:|---|
| center | `<16.468481131962367` | `0.1812 px` | invalid |
| P10 | `>35.02295998845781%` | `99.97%` | invalid |
| P5 | `>12.133503770828247%` | `99.72%` | invalid |
| P1 | evaluator key available, paper-frame full-test evidence missing | `99.61%` | invalid |

Reasons:

- Current metric frame is `post-transform input coordinate frame`.
- Raw sensor-space prediction rows are missing.
- Full hybrid scheduler evaluation rows are missing.
- P1 evaluator implementation exists as `metric_track_p1_pct`, but paper-frame/full-test P1 evidence is missing.
- EV-Eye/EX-Gaze split-protocol provenance is not yet proven.
- XR-64 post-run trained evidence is missing.

## Required Evidence To Unblock

| ID | Required evidence | Current status | Required outputs |
|---|---|---|---|
| `coordinate_frame_match` | raw sensor-space and transformed prediction rows, or inverse-transform/paper evaluator | missing | sensor-space rows, evaluator, coordinate-frame decision |
| `hybrid_scheduler_eval` | rows with search/track mode, chosen state, center/P10/P5/P1 | missing | hybrid rows, mode provenance, metric report |
| `p1_metric_coverage` | P1 hit-rate in matched frame | partial | P1 metric, checker/test coverage, full-test P1 value |
| `split_protocol_match` | manifest provenance/counts matched to EX-Gaze-following split claim | missing | split audit, count report, protocol decision |
| `target_definition_match` | target definition matches paper table | missing | target note, `cur_state` mapping, sample sanity rows |
| `latency_separation` | 0.43 ms latency kept separate from software accuracy | missing | separate accuracy and latency evidence |
| `xr64_result_dependency` | trained leakage-safe candidate, not XR-63 oracle | missing | checkpoint evidence, override-cleared test eval, leakage audit |

## Required Output Schema

Future unblocking evidence must include:

- sensor-space eval rows: `sample_id`, `split`, transformed prediction/target, sensor prediction/target, chosen mode, center error, P10/P5/P1 hits.
- metric report: metric frame, center px, P10/P5/P1, sample count.
- protocol audit: manifest root, split counts, split provenance, target definition, scheduler policy.
- bridge decision: direct-comparison flag, paper-level-completion flag, remaining blockers.

Concrete file-level inventory is tracked in `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md`. That manifest currently lists 10 future required artifacts and 4 current supporting artifacts.

## Reject Conditions

The checker must reject:

- `direct_submission_comparison_allowed=true` while any required evidence is incomplete.
- `paper_level_completion_allowed=true` while direct comparison is false.
- missing or incomplete P1, hybrid scheduler, or coordinate-frame requirements.
- validation commands that launch train/eval experiments.
- test-derived target overrides used as training evidence.
- XR-63 oracle used as trained promotion evidence.

## Validation

```bash
python3 -m py_compile scripts/external/check_metric_protocol_unblock_contract.py
.venv/bin/python scripts/external/check_metric_protocol_unblock_contract.py --format summary
.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_hbtxr_metrics_p1.py tests/test_metric_protocol_unblock_contract.py tests/test_metric_protocol_bridge.py tests/test_submission_target_gap.py tests/test_paper_target_comparison_evidence_manifest.py
```

Machine-readable source: `docs/resources/metric_protocol_unblock_contract_2026_06_21.json`.
