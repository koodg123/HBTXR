# XR-64 Prep Progress Ledger

Date: 2026-06-21

This is a no-execute progress ledger for XR-64-prep. It records current artifact readiness only.
It does not run train/eval jobs and does not authorize XR-64A/B launch.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Objective | Track XR-64-prep progress so the next worker can resume artifact generation safely. |
| Source authority | `docs/resources/xr64_resume_command_manifest_2026_06_20.json` and `scripts/v3/emit_xr64_next_prep_command.py` |
| Target | HGTXR-SW `manifest1` train/val teacher-target construction |
| Required artifact | machine-checkable progress ledger and validator |
| Verification | `.venv/bin/python scripts/v3/check_xr64_prep_progress_ledger.py --format summary` |

## Current State

| Field | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| allowed to run now | `false` |
| prep units | `10` |
| eval units | `8` |
| build units | `2` |
| ready after resume | `8` |
| completed | `0` |
| blocked | `2` |
| invalid | `0` |
| next prep ID | `XR64-EVAL-TRAIN-XR62A` |
| missing eval rows | `8` |
| missing overrides | `6` |
| eval inputs ready | `true` |
| build inputs ready | `false` |

## Expected Generated Evidence

| Artifact family | Expected count | Expected records |
|---|---:|---:|
| teacher eval rows | `8` files | `27092` rows |
| override JSON | `6` files | `20319` override rows |

Split cardinalities remain train `5929` and val `844`.

## Unit Matrix

| Unit | Action | Split | Teacher | Status | Missing outputs |
|---|---|---|---|---|---:|
| XR64-EVAL-TRAIN-XR62A | eval | train | xr62a | ready_after_resume | 1 |
| XR64-EVAL-TRAIN-XR39 | eval | train | xr39 | ready_after_resume | 1 |
| XR64-EVAL-TRAIN-XR56B | eval | train | xr56b | ready_after_resume | 1 |
| XR64-EVAL-TRAIN-XR58A | eval | train | xr58a | ready_after_resume | 1 |
| XR64-EVAL-VAL-XR62A | eval | val | xr62a | ready_after_resume | 1 |
| XR64-EVAL-VAL-XR39 | eval | val | xr39 | ready_after_resume | 1 |
| XR64-EVAL-VAL-XR56B | eval | val | xr56b | ready_after_resume | 1 |
| XR64-EVAL-VAL-XR58A | eval | val | xr58a | ready_after_resume | 1 |
| XR64-BUILD-TRAIN | build | train | n/a | blocked_missing_inputs | 3 |
| XR64-BUILD-VAL | build | val | n/a | blocked_missing_inputs | 3 |

## Guardrails

- Do not execute while `execution_state=paused_by_user_directive`.
- Do not use test split target overrides.
- Do not treat subset artifacts as `ready_to_train`.
- Do not launch XR-64A/B until strict XR-64 readiness reports `ready_to_train=true`.
- Do not claim direct paper-target comparison while the metric/protocol bridge remains blocked.

Machine-readable source: `docs/resources/xr64_prep_progress_ledger_2026_06_21.json`.
