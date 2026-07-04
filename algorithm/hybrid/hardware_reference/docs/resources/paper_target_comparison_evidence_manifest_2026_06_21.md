# HGTXR-SW Paper Target Comparison Evidence Manifest

Date: 2026-06-21

Experiments are paused by user directive. This manifest does not run training or evaluation.

## Purpose

The metric/protocol unblock contract defines the evidence classes required before HGTXR-SW metrics may be compared directly with the `10_submission_initial` target. This manifest makes those requirements concrete as files and artifacts that must exist before a paper-level comparison claim is allowed.

## Current Decision

| Item | State |
|---|---:|
| manifest ID | `paper_target_comparison_evidence_manifest_2026_06_21` |
| evidence pack version | `1` |
| evidence state | `missing_required_evidence` |
| required evidence complete | `false` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| current software promotion metric space | current center/P10/P5 gates only |

## Required Evidence Items

| ID | Status | Required artifacts |
|---|---|---:|
| `coordinate_frame_match` | missing | 2 |
| `hybrid_scheduler_eval` | missing | 2 |
| `p1_full_test_metric` | partial | 1 future artifact plus current P1 implementation/tests |
| `split_protocol_match` | partial | 1 doc-prefill artifact exists; exact paper split identity not proven |
| `target_definition_match` | partial | 1 future artifact plus current source trace |
| `latency_separation` | partial | 1 future artifact plus current source trace |
| `xr64_or_later_trained_result` | missing | 2 |

## Concrete Missing Artifacts

The current future artifact namespace is `docs/resources/future/paper_target_bridge/`.

Still missing before direct comparison:

- `sensor_space_eval_rows.jsonl`
- `coordinate_transform_audit.json`
- `hybrid_scheduler_eval_rows.jsonl`
- `hybrid_scheduler_metric_report.json`
- `paper_frame_full_test_p1_report.json`
- `trained_candidate_test_eval_summary.json`
- `trained_candidate_leakage_audit.json`

Paused-safe doc-prefill artifacts now present:

- `split_protocol_audit.json`
- `cur_state_target_mapping_audit.json`
- `accuracy_latency_separation_note.json`

These three files satisfy source-note schemas, but they do not prove coordinate-frame match, hybrid scheduler metrics, paper-frame full-test P1, or trained XR-64-or-later evidence. Direct paper-target comparison remains blocked.

## Future Artifact Readiness Class

| Artifact | Readiness class | Reason |
|---|---|---|
| `sensor_space_eval_rows.jsonl` | needs trained/eval rows | requires actual full-test rows in the target metric frame |
| `coordinate_transform_audit.json` | doc-prefillable plus eval evidence | transform policy can be drafted now, but sample-level proof needs eval rows |
| `hybrid_scheduler_eval_rows.jsonl` | needs hybrid eval rows | requires search/track scheduler-mode evidence, not track-only summaries |
| `hybrid_scheduler_metric_report.json` | needs hybrid eval rows | depends on hybrid scheduler rows and paper-frame metrics |
| `paper_frame_full_test_p1_report.json` | needs full-test metric evidence | P1 implementation exists, but paper-frame full-test evidence is missing |
| `split_protocol_audit.json` | doc-prefillable | split counts/provenance can be documented before new training, then rechecked after runs |
| `cur_state_target_mapping_audit.json` | doc-prefillable | target tensor and paper state-definition mapping can be documented before new training |
| `accuracy_latency_separation_note.json` | doc-prefillable | software accuracy and hardware latency separation can be documented now |
| `trained_candidate_test_eval_summary.json` | needs trained candidate | requires XR-64-or-later full-test result |
| `trained_candidate_leakage_audit.json` | needs trained candidate | requires final candidate provenance and override-path audit |

## Future Artifact Content Schema

The manifest also defines content schemas for all `10` future bridge artifacts. If any future artifact already exists, `scripts/external/check_paper_target_comparison_evidence_manifest.py` parses the payload and validates required fields instead of treating file existence alone as sufficient evidence.

Schema coverage:

- `sensor_space_eval_rows`: JSONL rows with transform-space and sensor-space centers plus sensor-pixel center error.
- `coordinate_transform_audit`: JSON object proving metric frame, transform policy, sensor-pixel status, sample count, and decision.
- `hybrid_eval_rows`: JSONL rows with chosen mode/state and center/P10/P5/P1 hit evidence.
- `hybrid_metric_report`: JSON object with hybrid center/P10/P5/P1 metrics.
- `paper_frame_full_test_p1`: JSON object proving paper-frame full-test P1 evidence.
- `split_protocol_audit`: JSON object proving split counts, provenance, and paper protocol match decision.
- `cur_state_mapping_audit`: JSON object proving current target tensor and paper state-definition mapping.
- `accuracy_latency_separation_note`: JSON object proving software accuracy and hardware latency are not conflated.
- `trained_candidate_eval_summary`: JSON object with XR-64-or-later test metrics including P1.
- `trained_candidate_leakage_audit`: JSON object proving train/val/test override provenance and leakage decision.

Current schema count is `10`. Current existing required future artifacts are `3`, all in the paused-safe doc-prefill class. No execution-dependent future payload exists yet, so direct comparison remains blocked.

## Current Partial Evidence

Already available:

- `src/hbtxr/loss/metrics.py`: `metric_track_p1_pct` and `metric_search_p1_pct`.
- `tests/test_hbtxr_metrics_p1.py`: focused P1 metric coverage.
- `docs/resources/submission_target_source_trace_2026_06_21.json`: paper-side target and latency source trace.
- `docs/resources/future/paper_target_bridge/split_protocol_audit.json`: manifest split-count/source-note artifact.
- `docs/resources/future/paper_target_bridge/cur_state_target_mapping_audit.json`: target-definition source-note artifact.
- `docs/resources/future/paper_target_bridge/accuracy_latency_separation_note.json`: accuracy/latency separation decision note.

These reduce ambiguity, but they do not prove paper-frame/full-test comparison validity.

## Rejection Rules

Reject any final comparison claim if:

- any required evidence item is missing or partial;
- `direct_submission_comparison_allowed=true` while evidence is incomplete;
- XR-63 oracle is used as a trained result;
- `metric_track_center_px` is compared directly to `0.1812 px` before coordinate-frame matching;
- hardware latency is used as software accuracy evidence;
- validation commands launch train/eval/GPU jobs while experiments remain paused.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_comparison_evidence_manifest.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_paper_target_comparison_evidence_manifest.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_comparison_evidence_manifest.py tests/test_second_goal_status.py
```

Machine-readable source: `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json`.
