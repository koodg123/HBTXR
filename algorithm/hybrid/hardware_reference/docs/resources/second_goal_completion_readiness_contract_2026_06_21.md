# HGTXR-SW Second-Goal Completion Readiness Contract

Date: 2026-06-21

This contract records the current completion-readiness state for the HGTXR-SW second goal.
It is a no-execute verification artifact. It does not run training or evaluation.

## Prompt Pack

| Field | Value |
|---|---|
| Skill phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Objective | Make second-goal completion auditable without narrowing the objective or running train/eval while paused. |
| Target | HGTXR-SW raw event-count track model on `manifest1`; XR-64-first accuracy-improvement program. |

## Current Readiness

| Item | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| active goal complete | `false` |
| completion allowed | `false` |
| blocker count | `28` |
| XR-64 ready to train | `false` |
| XR-64 can run lane | `false` |
| missing eval rows | `8` |
| missing overrides | `6` |
| leakage risk | `none` |
| metric/protocol bridge | `blocked_until_metric_frame_and_protocol_match` |
| direct submission comparison allowed | `false` |
| paper-level completion allowed | `false` |
| paper-target evidence state | `missing_required_evidence` |
| paper-target required evidence complete | `false` |
| paper-target future bridge artifacts present | `3 / 10` |
| paper-target future artifact schemas present | `10 / 10` |
| PTB-1 evaluator schema execute supported | `false` |
| PTB-1 evaluator output schemas | `5` |
| PTB-1 evaluator output payloads present | `0` |
| PTB-2 trained-candidate execute supported | `false` |
| PTB-2 schema allowed while paused | `true` |
| PTB-2 payload allowed while paused | `false` |
| PTB-2 trained-candidate output schemas | `2` |
| PTB-2 trained-candidate output payloads present | `0` |
| PTB-3 bridge-decision execute supported | `false` |
| PTB-3 schema allowed while paused | `true` |
| PTB-3 payload allowed while paused | `false` |
| PTB-3 bridge-decision output schemas | `1` |
| PTB-3 bridge-decision output payloads present | `0` |
| XR-64 command manifest post-run commands | `2` |
| XR-64 full eval coverage required | `true` |
| XR-64 full override coverage required | `true` |
| XR-64 duplicate sample IDs allowed | `false` |
| XR-64 opposite split sample IDs allowed | `false` |
| XR-64 subset artifacts satisfy ready-to-train | `false` |
| XR-64 strict generated artifact checker | `scripts/external/check_xr64_resume_artifacts.py` |
| XR-64 next prep selector ok | `true` |
| XR-64 next prep ready-after-resume commands | `8` |
| XR-64 next prep completed commands | `0` |
| XR-64 next prep blocked commands | `2` |
| XR-64 next prep invalid commands | `0` |
| XR-64 next prep command | `XR64-EVAL-TRAIN-XR62A` |
| XR-64 next prep command status | `ready_after_resume` |
| XR-64 next prep allowed to run now | `false` |
| XR-64 next prep execute supported | `false` |
| XR-64 prelaunch phases | `6` |
| XR-64 prelaunch post-run commands | `2` |
| post-XR64 decision tree execute supported | `false` |
| post-XR64 decision tree claim levels | `4` |
| post-XR64 decision tree decision inputs | `3` |
| post-XR64 decision tree branches | `7` |
| post-XR64 decision tree minimal next worker | `XR-64-prep` |
| post-XR64 decision tree current next prep | `XR64-EVAL-TRAIN-XR62A` |
| post-XR64 decision tree XR-64 ready to train | `false` |
| post-XR64 decision tree direct submission comparison allowed | `false` |
| post-XR64 follow-up contract execute supported | `false` |
| post-XR64 follow-up contract experiments | `5` |
| post-XR64 follow-up blocked launches | `5` |
| post-XR64 follow-up launch allowed now | `false` |
| XR-64 post-run candidates | `0` |

## Required Resume Gate

Before XR-64A/B can be launched after explicit user resume:

- strict checker must report `ready_to_train`,
- `can_run_lane=true`,
- missing eval rows must be `0`,
- missing override JSON files must be `0`,
- leakage risk must remain `none`.

Strict checker:

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary
```

## Remaining Gap By Goal Layer

| Goal layer | Current evidence | Remaining blockers |
|---|---|---|
| PAPER_REF trace | per-paper analysis and ablation plan exist | no blocker for planning input, but future paper additions must be re-indexed |
| XR-64 execution readiness | command manifest, selector, and prelaunch packet exist | experiments paused, `8` eval rows missing, `6` overrides missing, `ready_to_train=false` |
| single-model trained result | current best gates are split across XR-62A, XR-39, and XR-56B | no XR-64A/B/C train runs, no post-run candidates, no axis-certified promotion |
| metric/protocol bridge | target source trace and bridge contracts exist | coordinate frame, hybrid scheduler, P1 full-test, split protocol, and trained-candidate evidence incomplete |
| submission-level claim | `10_submission_initial` target values are traced | direct paper-target comparison remains blocked and paper-level completion is not allowed |

## Completion Blockers

Current blocker IDs:

- `execution_paused`
- `active_goal_not_marked_complete`
- `objective_trace_incomplete`
- `experiment_execution_incomplete`
- `accuracy_closure_incomplete`
- `single_model_result_missing`
- `direct_submission_comparison_blocked`
- `current_result_xr64_not_ready`
- `metric_protocol_bridge_blocked`
- `metric_protocol_bridge_not_ready`
- `submission_target_gap_unclosed`
- `paper_target_required_evidence_incomplete`
- `paper_target_direct_comparison_blocked`
- `paper_target_future_artifacts_missing`
- `xr64_not_ready_to_train`
- `xr64_lane_blocked`
- `xr64_eval_rows_missing`
- `xr64_overrides_missing`
- `indexed_eval_rows_missing`
- `indexed_overrides_missing`
- `xr64_training_runs_missing`
- `experiment_queue_paused`
- `xr64_postrun_candidates_missing`
- `xr64_postrun_software_promotion_blocked`
- `xr64_postrun_single_model_claim_blocked`
- `xr64_postrun_axis_claims_blocked`
- `xr64_postrun_controls_unchecked`
- `xr64_postrun_p1_evidence_missing`

## Guards Passed

Current guard IDs:

- `oracle_not_promotable`
- `leakage_risk_none`
- `authority_links_resolve`
- `command_emitter_non_executing`
- `command_manifest_paused_guard`
- `xr64_postrun_promotion_decision_valid`

## Allowed Actions

Allowed while paused:

- read current status,
- validate experiment queue,
- validate ablation evidence checklist,
- validate completion gate with `--allow-incomplete`,
- validate XR-64 resume command manifest,
- validate XR-64 prelaunch packet,
- inspect command manifests.

Not allowed while paused:

- execute XR-64 prep,
- launch XR-64A/B/C,
- claim software promotion,
- claim direct paper-target comparison,
- mark the active second goal complete.

Machine-readable source: `docs/resources/second_goal_completion_readiness_contract_2026_06_21.json`.
