# Paper Target Bridge Evaluator Schema Contract

Date: 2026-06-21

Experiments are paused by user directive. This contract freezes the PTB-1
evaluator output schemas without creating execution-dependent rows or reports.

## Current Decision

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| PTB-1 evaluator outputs | `5` |
| current PTB-1 output files | `0` |

## PTB-1 Outputs

| Output | Format | Required Future Path | Current State |
|---|---|---|---|
| `sensor_space_eval_rows` | JSONL | `docs/resources/future/paper_target_bridge/sensor_space_eval_rows.jsonl` | absent until evaluator execution |
| `coordinate_transform_audit` | JSON | `docs/resources/future/paper_target_bridge/coordinate_transform_audit.json` | absent until sample-level transform evidence |
| `hybrid_eval_rows` | JSONL | `docs/resources/future/paper_target_bridge/hybrid_scheduler_eval_rows.jsonl` | absent until hybrid scheduler evaluation |
| `hybrid_metric_report` | JSON | `docs/resources/future/paper_target_bridge/hybrid_scheduler_metric_report.json` | absent until hybrid rows exist |
| `paper_frame_full_test_p1` | JSON | `docs/resources/future/paper_target_bridge/paper_frame_full_test_p1_report.json` | absent until paper-frame full-test P1 evaluation |

## Guardrails

- Do not set `execute_supported=true` while experiments are paused.
- Do not create PTB-1 required future artifact files as placeholders.
- Do not allow direct submission comparison from schema-only evidence.
- Do not allow paper-level completion from schema-only evidence.
- Do not use validation commands that launch train/eval/GPU jobs.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_bridge_evaluator_schema_contract.py scripts/external/check_paper_target_bridge_payloads.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_paper_target_bridge_evaluator_schema_contract.py --format summary
.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_bridge_evaluator_schema_contract.py tests/test_paper_target_bridge_payloads.py tests/test_second_goal_status.py
```

Machine-readable source:
`docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json`.
