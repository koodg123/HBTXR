# HGTXR-SW Post-XR64 Follow-Up Experiment Contract

Date: 2026-06-21

This is a no-execute contract for the paused second-goal program. It prevents XR-64C, XR-65, XR-66, XR-67, and XR-68 from being launched or promoted without the exact XR-64 result pattern and PAPER_REF-backed evidence each branch requires.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Objective | prevent post-XR64 follow-up experiments from bypassing XR-64 evidence and branch-specific controls |
| Target | HGTXR-SW raw event-count track model on `manifest1` |
| Required artifact | branch-level launch/promotion contract for XR-64C/XR-65/XR-66/XR-67/XR-68 |
| Verification evidence | `scripts/external/check_second_goal_post_xr64_followup_contract.py --format summary` |

## Current State

| Item | State |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| follow-up launch allowed now | `false` |
| XR-64 ready to train | `false` |
| XR-64 missing eval rows | `8` |
| XR-64 missing overrides | `6` |
| XR-64 post-run candidates | `0` |
| minimal next worker after resume | `XR-64-prep` |
| direct submission comparison allowed | `false` |

## Follow-Up Branch Contracts

| ID | Priority | Required result pattern | Mechanism | Launch gate summary | Promotion scope |
|---|---:|---|---|---|---|
| `XR-64C` | P1 | validation gain without full-test gate | min-error teacher-target pressure diagnostic | requires XR-64A/B evidence, validation signal, full-test failure, no test override | diagnostic unless full-test gate improves |
| `XR-65` | P1 | gate improvement with bounded tradeoff | temporal-lite residual/state regularization | requires non-collapsing XR-64 checkpoint, temporal bucket, no test override | software-promotion candidate |
| `XR-66` | P1 | center gain with threshold collapse | confidence-gated local update diagnostic | requires XR-64 tradeoff and confidence bucket signal; no train first | diagnostic first |
| `XR-67` | P2 | dense segments available | dense trajectory refinement/postprocess | requires dense/continuous segments and same-scene before/after evidence | postprocess diagnostic until coverage proven |
| `XR-68` | P2 | stable full-width teacher improves gates | optimizer/distillation/sparse/hardware branch | requires stronger stable teacher, tolerance budget, latency/resource plan | hardware-export candidate after software promotion |

## Global Rejection Rules

- Do not execute while `execution_state=paused_by_user_directive`.
- Do not launch follow-up experiments before XR-64-prep and XR-64A/B post-run evidence exist.
- Do not promote validation-only gains.
- Do not compare with `10_submission_initial` until the metric/protocol bridge allows direct comparison.
- Do not start XR-68 before software teacher quality improves.

## Validation

```bash
python3 -m py_compile scripts/external/check_second_goal_post_xr64_followup_contract.py
.venv/bin/python scripts/external/check_second_goal_post_xr64_followup_contract.py --format summary
.venv/bin/python -m pytest -q tests/test_second_goal_post_xr64_followup_contract.py
```

Machine-readable source: `docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json`.
