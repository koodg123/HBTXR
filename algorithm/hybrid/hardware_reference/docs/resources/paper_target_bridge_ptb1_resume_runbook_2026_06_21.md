# Paper Target Bridge PTB-1 Resume Runbook

Date: 2026-06-21

Experiments are paused by user directive. This runbook does not create PTB-1
payload files and does not run train/eval jobs.

## Current State

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| allowed while paused | `true` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| PTB-1 output schemas | `5` |
| PTB-1 payloads present | `0` |
| PTB-1 payloads missing | `5` |

## Resume Sequence

| Step | Output | Path |
|---|---|---|
| `PTB1-01-COORDINATE-TRANSFORM-AUDIT` | `coordinate_transform_audit` | `docs/resources/future/paper_target_bridge/coordinate_transform_audit.json` |
| `PTB1-02-SENSOR-SPACE-EVAL-ROWS` | `sensor_space_eval_rows` | `docs/resources/future/paper_target_bridge/sensor_space_eval_rows.jsonl` |
| `PTB1-03-PAPER-FRAME-FULL-TEST-P1` | `paper_frame_full_test_p1` | `docs/resources/future/paper_target_bridge/paper_frame_full_test_p1_report.json` |
| `PTB1-04-HYBRID-EVAL-ROWS` | `hybrid_eval_rows` | `docs/resources/future/paper_target_bridge/hybrid_scheduler_eval_rows.jsonl` |
| `PTB1-05-HYBRID-METRIC-REPORT` | `hybrid_metric_report` | `docs/resources/future/paper_target_bridge/hybrid_scheduler_metric_report.json` |

All five steps are blocked while paused. After explicit resume, generate these
payloads in order, then validate with:

```bash
.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --allow-present --require-all-present --format summary
```

## Rejection Rules

- Do not create PTB-1 payload files while execution state is paused.
- Do not treat this runbook as payload evidence.
- Do not allow direct paper-target comparison from schema or runbook evidence alone.
- Do not create PTB-2 trained-candidate or PTB-3 bridge-decision payloads from PTB-1 evidence alone.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py --format summary
.venv/bin/python scripts/external/check_paper_target_bridge_payloads.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_bridge_ptb1_resume_runbook.py tests/test_second_goal_status.py
```

Machine-readable source:
`docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json`.
