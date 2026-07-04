# Paper Target Bridge Resume Dependency Matrix

Date: 2026-06-21

Experiments are paused by user directive. This is a no-execute dependency
matrix for the paper-target bridge; it does not create evaluator payloads,
trained-candidate evidence, bridge decisions, or accuracy claims.

## Purpose

The second goal cannot compare current HGTXR software metrics directly with
`10_submission_initial` until the paper-target bridge is complete. This matrix
binds the separate PTB contracts into one resume order:

1. PTB-0 source-note artifacts.
2. PTB-1 evaluator schema and runbook.
3. PTB-1 execution payloads after explicit resume.
4. PTB-2 XR-64-or-later trained candidate evidence.
5. PTB-3 bridge decision.
6. Strict second-goal completion gate.

## Current State

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| PTB-0 doc-prefill existing artifacts | `3` |
| PTB-1 payloads existing / missing | `0 / 5` |
| PTB-2 payloads existing / required | `0 / 2` |
| PTB-3 payloads existing / required | `0 / 1` |
| execution-dependent payloads missing | `8` |
| XR-64 ready to train | `false` |
| XR-64 missing eval rows / overrides | `8 / 6` |
| XR-64 post-run candidates | `0` |
| completion allowed | `false` |
| blocker count | `28` |

## Dependency Gates

| Gate | Work package | Allowed while paused | Complete now | Role |
|---|---|---:|---:|---|
| `PBRD-0-DOC-PREFILL` | PTB-0 doc prefill | true | true | source-note artifacts only |
| `PBRD-1-SCHEMA-AND-RUNBOOK` | PTB-1 schema/runbook | true | true | schema/order ready; payloads absent |
| `PBRD-2-PTB1-PAYLOADS-AFTER-RESUME` | PTB-1 payloads | false | false | sensor-space/hybrid/P1 execution payloads |
| `PBRD-3-XR64-TRAINED-CANDIDATE` | PTB-2 trained candidate | false | false | override-cleared trained result and leakage audit |
| `PBRD-4-BRIDGE-DECISION` | PTB-3 bridge decision | false | false | direct-comparison decision only after PTB-1/2 |
| `PBRD-5-COMPLETION-GATE` | second-goal completion | false | false | strict final gate |

## Rejection Rules

- Do not set `execute_supported=true` while experiments are paused.
- Do not allow direct paper-target comparison while any dependency gate is incomplete.
- Do not mark PTB-1 payloads complete while payload files are absent.
- Do not mark PTB-2 complete while XR-64 post-run candidate count is `0`.
- Do not create PTB-3 `bridge_decision.json` before PTB-1 and PTB-2 both pass.
- Do not mark the second goal complete while `blocker_count` is nonzero.
- Do not put train/eval/GPU launch commands in this artifact's validation contract.

## Validation

```bash
python3 -m py_compile scripts/external/check_paper_target_bridge_resume_dependency_matrix.py scripts/external/report_second_goal_status.py
.venv/bin/python scripts/external/check_paper_target_bridge_resume_dependency_matrix.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_paper_target_bridge_resume_dependency_matrix.py tests/test_second_goal_status.py
```

Machine-readable source:
`docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json`.
