# HGTXR-SW Second Goal Artifact Index

Date: 2026-06-18

## Purpose

Single entry point for the active HGTXR-SW second goal:

- PAPER_REF and XR-eye-tracking analysis.
- Accuracy-improvement experiment plan.
- XR-63/XR-64 teacher-target path.
- Pause state, completion audit, resume runbook, and validation tools.

Experiments are paused by user directive. This index is documentation only.

## Current Authority

| Role | Artifact | Use |
|---|---|---|
| pause state | `docs/resources/experiment_pause_documentation_2026_06_18.md` | current no-execution state and missing XR-64 artifacts |
| current summary | `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md` | latest gates, branch decisions, next experiment list |
| current result synthesis | `docs/resources/second_goal_current_result_synthesis_2026_06_20.md` | validated interpretation of current best gates, XR-63 oracle status, paper-target blocker, and XR-64 resume gate |
| accuracy lift decision ladder | `docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.md` | no-execute ladder from current gates to XR-63 oracle, XR-64 transfer, post-XR64 branching, and paper-target bridge |
| objective trace | `docs/resources/second_goal_objective_trace_2026_06_20.md` | requirement-by-requirement second-goal trace and false-completion guardrails |
| resume readiness matrix | `docs/resources/second_goal_resume_readiness_matrix_2026_06_21.md` | no-execute matrix binding PAPER_REF axes, XR-64 resume gates, and paper-target blockers |
| XR-64 resume command manifest | `docs/resources/xr64_resume_command_manifest_2026_06_20.md` | future-only XR-64-prep command contract and launch gate |
| XR-64 experiment design contract | `docs/resources/xr64_experiment_design_contract_2026_06_21.md` | non-executing P0/P1 design contract for XR-64-prep/A/B/C lane parameters, GPU assignment, and promotion/rejection gates |
| XR-64 experiment design checker | `scripts/external/check_xr64_experiment_design_contract.py` | validates XR-64 design contract pause state, strict checker dependency, artifact counts, GPU assignment, hyperparameters, and post-run evidence requirements |
| XR-64 resume execution DAG | `docs/resources/xr64_resume_execution_dag_2026_06_21.md` | non-executing dependency DAG for precheck, prep eval/build, strict readiness, XR-64A/B launch, and post-run review |
| XR-64 resume execution DAG checker | `scripts/external/check_xr64_resume_execution_dag.py` | validates DAG node counts/order, dependencies, pause guards, no-test prep nodes, launch device mapping, and manifest/design-contract alignment |
| XR-64 resume command emitter | `scripts/external/emit_xr64_resume_commands.py` | prints manifest commands for review without executing experiments |
| XR-64 next prep command selector | `scripts/external/emit_xr64_next_prep_command.py` | selects the next ready-after-resume XR-64-prep command from current artifact readiness without executing experiments |
| XR-64 prelaunch packet | `docs/resources/xr64_prelaunch_packet_2026_06_20.md` | future-only resume packet tying command manifest, completion gate, and strict XR-64 readiness |
| XR-64 post-run evidence contract | `docs/resources/xr64_postrun_evidence_contract_2026_06_21.md` | required evidence and promotion rules after future XR-64A/B execution |
| XR-64 post-run promotion decision | `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md` | current no-evidence decision placeholder and future decision-helper instructions |
| post-XR64 follow-up experiment contract | `docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.md` | fixes no-execute launch/promotion gates for XR-64C/XR-65/XR-66/XR-67/XR-68 |
| metric/protocol unblock contract | `docs/resources/metric_protocol_unblock_contract_2026_06_21.md` | required evidence before current software metrics may be compared directly with paper-level targets |
| paper-target comparison evidence manifest | `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md` | concrete file/artifact inventory required before direct paper-target comparison can be allowed |
| paper-target bridge artifact workplan | `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.md` | orders the 10 future paper-target bridge artifacts into safe paused-state and post-resume work packages |
| paper-target bridge evaluator schema contract | `docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.md` | fixes PTB-1 evaluator output schemas without creating execution-dependent rows/reports |
| paper-target bridge PTB-1 resume runbook | `docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.md` | fixes post-resume PTB-1 payload generation order without creating payload files while paused |
| paper-target bridge trained-candidate contract | `docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.md` | fixes PTB-2 trained-candidate evidence schemas and leakage rejection rules without creating execution-dependent payloads |
| paper-target bridge decision contract | `docs/resources/paper_target_bridge_decision_contract_2026_06_21.md` | fixes PTB-3 final direct-comparison decision schema and signoff rules without creating a decision payload |
| paper-target bridge resume dependency matrix | `docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.md` | binds PTB-0/1/2/3, XR-64 readiness, and final completion into one no-execute dependency gate |
| current PAPER_REF ablation plan | `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md` | current PAPER_REF-to-XR-64/XR-65/XR-66/XR-67/XR-68 priority bridge |
| current PAPER_REF ablation plan checker | `scripts/external/check_second_goal_paper_ref_current_ablation_plan.py` | validates prompt-pack metadata, PAPER_REF signal coverage, ablation rows, axis coverage, lane trace, and paused-state guards |
| PAPER_REF experiment trace | `docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.md` | machine-readable trace from PAPER_REF paper groups to XR-64/XR-65/XR-66/XR-67/XR-68 experiments, axes, required evidence, and rejection rules |
| machine-readable experiment queue | `docs/resources/second_goal_experiment_queue_2026_06_20.json` | current XR-64-first execution queue and pause guard |
| experiment queue summary | `docs/resources/second_goal_experiment_queue_2026_06_20.md` | human-readable P0/P1/P2 queue |
| ablation evidence checklist | `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.md` | non-executing evidence checklist for post-resume ablation decisions |
| ablation evidence checklist checker | `scripts/external/check_second_goal_ablation_evidence_checklist.py` | validates experiment evidence requirements, P1 future evidence, and no-execute validation commands |
| completion readiness contract | `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md` | current blocker/readiness contract before any completion claim |
| completion readiness checker | `scripts/external/check_second_goal_completion_readiness_contract.py` | validates current readiness against status reporter and completion gate output |
| submission target gap audit | `docs/resources/second_goal_submission_target_gap_audit_2026_06_20.md` | gap between current software gates and `10_submission_initial` target claims |
| submission target source trace | `docs/resources/submission_target_source_trace_2026_06_21.md` | source-line trace from `10_submission_initial/main.tex` to target values and comparison guards |
| submission target source trace checker | `scripts/external/check_submission_target_source_trace.py` | validates source LaTeX values, target-definition markers, and no direct-comparison claim |
| metric/protocol bridge | `docs/resources/second_goal_metric_protocol_bridge_2026_06_20.md` | coordinate/protocol bridge required before comparing current metrics to `0.1812 px` |
| metric/protocol unblock contract checker | `scripts/external/check_metric_protocol_unblock_contract.py` | validates the explicit evidence contract for unblocking paper-target comparison |
| paper-target evidence manifest checker | `scripts/external/check_paper_target_comparison_evidence_manifest.py` | validates concrete future/current artifact inventory for direct paper-target comparison |
| paper-target bridge workplan checker | `scripts/external/check_paper_target_bridge_artifact_workplan.py` | validates no-execute work-package ordering for the future paper-target bridge artifacts |
| paper-target bridge evaluator schema checker | `scripts/external/check_paper_target_bridge_evaluator_schema_contract.py` | validates PTB-1 evaluator schemas and keeps output payloads absent while paused |
| paper-target bridge payload checker | `scripts/external/check_paper_target_bridge_payloads.py` | validates PTB-1 official payload absence while paused and validates future payloads only with explicit allow-present mode |
| paper-target bridge PTB-1 resume runbook checker | `scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py` | validates post-resume PTB-1 payload order, source-schema alignment, and paused-state payload absence |
| paper-target bridge resume dependency matrix checker | `scripts/external/check_paper_target_bridge_resume_dependency_matrix.py` | validates PTB-0/1/2/3 dependency order, current blocked counts, XR-64 readiness linkage, and no-execute validation commands |
| current result synthesis checker | `scripts/external/check_current_result_synthesis.py` | validates current best gates, XR-63 non-promotion, direct-comparison blocker, and XR-64 missing-artifact state |
| accuracy lift decision ladder checker | `scripts/external/check_second_goal_accuracy_lift_decision_ladder.py` | validates ladder stages, current best owners, XR-63 non-promotion, XR-64 blockers, follow-up triggers, controls, evidence, and no-execute validation commands |
| objective trace checker | `scripts/external/check_second_goal_objective_trace.py` | validates requirement status, axis coverage, guardrails, and incomplete completion judgment |
| resume readiness matrix checker | `scripts/external/check_second_goal_resume_readiness_matrix.py` | validates axis-to-resume-gate alignment, source drift, pause guards, and no-execute validation commands |
| completion gate checker | `scripts/external/check_second_goal_completion_gate.py` | validates whether the full second goal may be marked complete; currently reports completion blocked |
| XR-64 prelaunch packet checker | `scripts/external/check_xr64_prelaunch_packet.py` | validates future-only prelaunch packet, source links, manifest alignment, pause guard, and blocker state |
| XR-64 post-run evidence contract checker | `scripts/external/check_xr64_postrun_evidence_contract.py` | validates future post-run evidence requirements, override-cleared test eval contract, and promotion decision guardrails |
| XR-64 ablation provenance writer | `scripts/external/write_xr64_ablation_provenance.py` | creates future XR-64 train provenance artifacts and patches eval summaries with ablation axis certification |
| XR-64 post-run candidate collector | `scripts/external/collect_xr64_postrun_candidates.py` | discovers future patched XR-64 eval summaries and builds promotion-helper candidate inputs |
| XR-64 post-run promotion decision helper | `scripts/external/decide_xr64_postrun_promotion.py` | computes software promotion status from future XR-64A/B test `eval_summary.json` files |
| XR-64 command manifest checker | `scripts/external/check_xr64_resume_command_manifest.py` | validates future XR-64-prep command count, split/teacher coverage, generated outputs, and launch guard |
| XR-64 command emitter tests | `tests/test_emit_xr64_resume_commands.py` | validates non-executing command output modes |
| XR-64 next prep selector tests | `tests/test_emit_xr64_next_prep_command.py` | validates deterministic next-command selection from missing/complete XR-64 generated artifacts |
| completion audit | `docs/resources/second_goal_completion_audit_2026_06_18.md` | objective-by-objective completion status |
| paper-backed plan | `docs/Paper-Backed-Experiment-Plan.md` | long-form experiment history and paper-backed plan |
| resume runbook | `docs/resources/xr64_resume_runbook_2026_06_18.md` | exact future XR-64 command sequence and validation gates |
| validation evidence | `docs/Validation.md` | test/static-check evidence |
| active TODO | `docs/track/TODO.md` | next tasks after resume |
| progress ledger | `docs/track/PROGRESS.md` | plan-versus-progress checklist |
| changelog | `docs/track/CHANGELOG.md` | change-history ledger |
| conversation context | `docs/track/CONVERSATION.md` | durable conversation/project context |
| legacy results | `docs/track/EXTERNAL_HYBRID_PACKAGE_PAST_EXPERIMENT_RESULTS.md` | imported past experiment evidence |
| work log | `docs/track/log.md` | chronological evidence log |

## Analysis Artifacts

| Category | Artifact | Status |
|---|---|---|
| PAPER_REF map | `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md` | complete as source map; top note marks older priority rows as superseded by XR-64 plan |
| PAPER_REF per-paper index | `anlaysis/paper-ref/papers/index.md` | 31 PDF-level analysis files with priority, HGTXR mapping, ablation option, and rejection rule |
| PAPER_REF coverage checker | `scripts/external/check_paper_ref_analysis_coverage.py` | verifies PDF-to-analysis coverage, required sections, XR-64-first wording, active gates, and option matrix axes |
| current PAPER_REF ablation matrix | `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md` | current execution-oriented ablation matrix and stale-item ruling |
| current PAPER_REF ablation plan checker | `scripts/external/check_second_goal_paper_ref_current_ablation_plan.py` | validates prompt-pack metadata, PAPER_REF signal coverage, 8 queued experiment rows, P0 IDs, requested axes, and no-completion/no-comparison flags |
| second-goal experiment queue checker | `scripts/external/check_second_goal_experiment_queue.py` | verifies pause guard, XR-64-first order, P0 IDs, lane assignment, required axes, and stale P0 exclusion |
| second-goal ablation evidence checklist | `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.md` | binds each queued experiment to controls, evidence, promotion decisions, and rejection rules |
| second-goal ablation evidence checker | `scripts/external/check_second_goal_ablation_evidence_checklist.py` | verifies checklist source links, axis coverage, P0 IDs, future P1 evidence, and no train/eval validation commands |
| PAPER_REF experiment trace | `docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.md` | binds six paper evidence groups to eight queued experiments and current pause-state blockers |
| PAPER_REF experiment trace checker | `scripts/external/check_second_goal_paper_ref_experiment_trace.py` | validates paper analysis paths, experiment order, axis coverage, no-execute commands, and XR-64 blocker state |
| XR-64 experiment design contract | `docs/resources/xr64_experiment_design_contract_2026_06_21.md` | concrete non-executing contract for XR-64-prep/A/B/C hyperparameters and launch gates |
| XR-64 experiment design checker | `scripts/external/check_xr64_experiment_design_contract.py` | verifies current XR-64 lane order, P0 IDs, pause guard, GPU assignment, LR/loss coefficients, missing generated-artifact counts, and evidence contract |
| XR-64 resume execution DAG | `docs/resources/xr64_resume_execution_dag_2026_06_21.md` | dependency graph binding manifest commands and design contract into the future resume order |
| XR-64 resume execution DAG checker | `scripts/external/check_xr64_resume_execution_dag.py` | verifies the future resume DAG against manifest command IDs and the XR-64 design contract |
| accuracy lift decision ladder | `docs/resources/second_goal_accuracy_lift_decision_ladder_2026_06_21.md` | checked no-execute decision ladder preserving XR-64-first path and post-XR64 branch triggers |
| accuracy lift decision ladder checker | `scripts/external/check_second_goal_accuracy_lift_decision_ladder.py` | verifies the ladder against active gates, oracle status, PAPER_REF axes, blocked launches, and missing XR-64 evidence |
| XR-eye-tracking index | `anlaysis/xr-eye-tracking/index.md` | available |
| codebase synthesis | `anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md` | available |
| paper synthesis | `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md` | available |
| experiment integration | `anlaysis/xr-eye-tracking/experiment_integration.md` | available |
| per-codebase analyses | `anlaysis/xr-eye-tracking/codebases/*/analysis.md` | `18` files |
| per-paper analyses | `anlaysis/xr-eye-tracking/papers/*/analysis.md` | `21` files |

Verified inventory:

- `39` XR-eye-tracking per-item `analysis.md` files.
- `31` PAPER_REF per-paper `analysis.md` files under `anlaysis/paper-ref/papers`.
- `15590` total lines across per-item analysis files.

## Current Gates

| Metric | Gate | Source |
|---|---:|---|
| center | `<16.468481131962367` | XR-62A FACET geometry auxiliary refresh |
| P10 | `>35.02295998845781` | XR-39 mixed-leader soup `c25p45f30` |
| P5 | `>12.133503770828247` | XR-56B best-P5 |

## Current Result Synthesis

| Artifact | Role |
|---|---|
| `docs/resources/second_goal_current_result_synthesis_2026_06_20.md` | human-readable current result interpretation |
| `docs/resources/second_goal_current_result_synthesis_2026_06_20.json` | machine-readable current result interpretation |
| `scripts/external/check_current_result_synthesis.py` | verifies result interpretation invariants |
| `tests/test_current_result_synthesis.py` | unit tests for the result synthesis checker |

Current decisions:

- The best center, P10, and P5 gates come from different experiment owners, so no single-model best claim is allowed.
- XR-63 is useful but diagnostic only; it cannot be promoted as a trained result.
- Direct comparison with the `0.1812 px` submission target remains invalid until the metric/protocol bridge is unblocked.
- XR-64 remains `generated_incomplete` with `8` missing eval-row files and `6` missing override JSON files.

## Metric/Protocol Unblock Contract

| Artifact | Role |
|---|---|
| `docs/resources/metric_protocol_unblock_contract_2026_06_21.md` | human-readable required evidence contract for paper-target comparison |
| `docs/resources/metric_protocol_unblock_contract_2026_06_21.json` | machine-readable unblock contract |
| `scripts/external/check_metric_protocol_unblock_contract.py` | validates blocked state, required evidence IDs, source-artifact alignment, and no train/eval validation commands |
| `tests/test_metric_protocol_unblock_contract.py` | unit tests for current pass state and false-unblock rejection |

Current contract decision:

- `contract_status=blocked`.
- `direct_submission_comparison_allowed=false`.
- `paper_level_completion_allowed=false`.
- Required evidence count is `7`: coordinate-frame match, hybrid scheduler eval, P1 metric coverage, split protocol match, target definition match, latency separation, and trained XR-64-or-later result dependency.
- P1 metric implementation now exists as `metric_track_p1_pct`; paper-frame/full-test P1 evidence is still missing.
- Current blockers are metric-frame mismatch, missing sensor-space rows, missing hybrid scheduler rows, incomplete P1 evidence, incomplete split-protocol provenance, and missing XR-64 post-run evidence.

## Paper Target Comparison Evidence Manifest

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.md` | human-readable concrete evidence inventory for direct paper-target comparison |
| `docs/resources/paper_target_comparison_evidence_manifest_2026_06_21.json` | machine-readable evidence manifest |
| `scripts/external/check_paper_target_comparison_evidence_manifest.py` | validates required/current artifact inventory and false-direct-comparison guards |
| `tests/test_paper_target_comparison_evidence_manifest.py` | unit tests for manifest pass state and false-unblock rejection |

Current manifest decision:

- `evidence_state=missing_required_evidence`.
- `required_evidence_complete=false`.
- `direct_submission_comparison_allowed=false`.
- `paper_level_completion_allowed=false`.
- Required evidence count is `7`.
- Missing evidence count is `3`; partial evidence count is `4`.
- Future required artifact count is `10`; current supporting artifact count is `4`.
- Future artifact content schema count is `10`; future bridge files are parsed for required JSON/JSONL fields when present.
- `3` paused-safe future paper-target bridge artifacts now exist: split protocol audit, cur_state target mapping audit, and accuracy/latency separation note. They are source-note artifacts only; direct comparison remains blocked.

## Paper Target Bridge Artifact Workplan

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.md` | human-readable work-package ordering for paper-target bridge artifacts |
| `docs/resources/paper_target_bridge_artifact_workplan_2026_06_21.json` | machine-readable workplan |
| `scripts/external/check_paper_target_bridge_artifact_workplan.py` | validates paused-state, source alignment, artifact-to-package mapping, and no train/eval validation commands |
| `tests/test_paper_target_bridge_artifact_workplan.py` | unit tests for workplan pass state and false-unblock rejection |

Current workplan decision:

- `execute_supported=false`.
- `direct_submission_comparison_allowed=false`.
- Work package count is `4`.
- Future artifact count is `10`.
- Two packages are allowed while paused: doc prefill and evaluator schema planning.
- `3` artifacts are doc-prefillable without train/eval: split protocol audit, cur_state mapping audit, and accuracy/latency separation note.
- Current existing future bridge artifact count is `3`.
- `5` artifacts are evaluator/schema artifacts whose actual rows/reports remain missing until explicit resume.
- `2` artifacts require a future trained XR-64-or-later candidate and are blocked while paused.

## Paper Target Bridge Evaluator Schema Contract

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.md` | human-readable PTB-1 evaluator output schema contract |
| `docs/resources/paper_target_bridge_evaluator_schema_contract_2026_06_21.json` | machine-readable PTB-1 schema contract |
| `scripts/external/check_paper_target_bridge_evaluator_schema_contract.py` | validates schema alignment with the paper-target evidence manifest and rejects paused-state output placeholders |
| `tests/test_paper_target_bridge_evaluator_schema_contract.py` | unit tests for pass state and false-unblock rejection |
| `scripts/external/check_paper_target_bridge_payloads.py` | validates PTB-1 output payload absence while paused and future payload schema/type consistency when explicitly allowed |
| `tests/test_paper_target_bridge_payloads.py` | unit tests for paused absence, complete future payload validation, and malformed payload rejection |

Current evaluator schema decision:

- `execute_supported=false`.
- `direct_submission_comparison_allowed=false`.
- Evaluator output count is `5`.
- JSONL output schemas: `2`; JSON output schemas: `3`.
- Current PTB-1 output payload count is `0`; the required future output files remain absent until explicit evaluation resume.

## Paper Target Bridge PTB-1 Resume Runbook

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.md` | human-readable post-resume PTB-1 payload runbook |
| `docs/resources/paper_target_bridge_ptb1_resume_runbook_2026_06_21.json` | machine-readable PTB-1 resume sequence |
| `scripts/external/check_paper_target_bridge_ptb1_resume_runbook.py` | validates runbook/source-schema alignment and keeps payload files absent while paused |
| `tests/test_paper_target_bridge_ptb1_resume_runbook.py` | unit tests for pass state, sequence drift, paused payload creation, and unsafe command rejection |

Current PTB-1 resume decision:

- `execute_supported=false`.
- `allowed_while_paused=true`.
- Resume sequence count is `5`.
- Current PTB-1 payload count is `0`; missing payload count is `5`.
- Future payload validation must use `check_paper_target_bridge_payloads.py --allow-present --require-all-present`.

## Paper Target Bridge Trained-Candidate Contract

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.md` | human-readable PTB-2 trained-candidate evidence contract |
| `docs/resources/paper_target_bridge_trained_candidate_contract_2026_06_21.json` | machine-readable PTB-2 contract |
| `scripts/external/check_paper_target_bridge_trained_candidate_contract.py` | validates PTB-2 schema alignment, paused-state payload absence, and future leakage-safe payload pairs |
| `tests/test_paper_target_bridge_trained_candidate_contract.py` | unit tests for paused state, future payload validation, override leakage rejection, and run-id consistency |

Current trained-candidate contract decision:

- `execute_supported=false`.
- `direct_submission_comparison_allowed=false`.
- `paper_level_completion_allowed=false`.
- PTB-2 schema contract is allowed while paused; PTB-2 payload creation is not.
- Trained-candidate output schemas: `2`.
- Current PTB-2 output payload count is `0`; both required future files remain absent until XR-64-or-later trained evidence exists.

## Paper Target Bridge Decision Contract

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_bridge_decision_contract_2026_06_21.md` | human-readable PTB-3 bridge-decision contract |
| `docs/resources/paper_target_bridge_decision_contract_2026_06_21.json` | machine-readable PTB-3 contract |
| `scripts/external/check_paper_target_bridge_decision_contract.py` | validates final direct-comparison decision schema, source blockers, and paused-state payload absence |
| `tests/test_paper_target_bridge_decision_contract.py` | unit tests for paused state, future blocked payload validation, and false direct-comparison rejection |

Current bridge-decision contract:

- `execute_supported=false`.
- `direct_submission_comparison_allowed=false`.
- `paper_level_completion_allowed=false`.
- PTB-3 schema contract is allowed while paused; bridge-decision payload creation is not.
- Decision output schemas: `1`.
- Current PTB-3 output payload count is `0`; `bridge_decision.json` remains absent until all required evidence is complete.

## Paper Target Bridge Resume Dependency Matrix

| Artifact | Role |
|---|---|
| `docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.md` | human-readable PTB resume dependency matrix |
| `docs/resources/paper_target_bridge_resume_dependency_matrix_2026_06_21.json` | machine-readable PTB-0/1/2/3 and completion dependency state |
| `scripts/external/check_paper_target_bridge_resume_dependency_matrix.py` | validates dependency gate order, source-count drift, pause guards, and no train/eval validation commands |
| `tests/test_paper_target_bridge_resume_dependency_matrix.py` | unit tests for current pass state, gate ordering, false completion, source drift, and unsafe command rejection |

Current dependency decision:

- `execute_supported=false`.
- `allowed_while_paused=true`.
- Dependency gate count is `6`.
- Complete-now gates: `2` (`PTB-0-DOC-PREFILL`, `PTB-1-SCHEMA-AND-RUNBOOK`).
- Blocked gates: `4` (`PTB-1 payloads`, `PTB-2 trained candidate`, `PTB-3 bridge decision`, `completion gate`).
- Execution-dependent payloads missing: `8`.
- Completion blocker count remains `28`; direct paper-target comparison and paper-level completion remain blocked.

## Submission Target Source Trace

| Artifact | Role |
|---|---|
| `docs/resources/submission_target_source_trace_2026_06_21.md` | human-readable source trace for target values in `10_submission_initial/main.tex` |
| `docs/resources/submission_target_source_trace_2026_06_21.json` | machine-readable source trace |
| `scripts/external/check_submission_target_source_trace.py` | validates the source trace against the current LaTeX file |
| `tests/test_submission_target_source_trace.py` | unit tests for source-trace pass state and false direct-comparison rejection |

Current source-trace decision:

- Table `tab:mode` hybrid row provides P10 `99.97`, P5 `99.72`, P1 `99.61`, pixel error `0.1812`, and latency `0.43`.
- Table `tab:slim` Option A provides pixel error `0.1812`, power `1.34 W`, and latency `0.43 ms`.
- Table `tab:algo` HBTXR column repeats P10/P5/P1/error target claims.
- Table `tab:accel` reports an error range `0.1812--0.357 px` and latency range `0.26--0.43 ms`; this range is not a single software target.
- Direct software comparison remains blocked until coordinate frame, hybrid scheduler mode, paper-frame P1, split protocol, target definition, and trained XR-64-or-later evidence are complete.

## Objective Trace

| Artifact | Role |
|---|---|
| `docs/resources/second_goal_objective_trace_2026_06_20.md` | human-readable requirement trace |
| `docs/resources/second_goal_objective_trace_2026_06_20.json` | machine-readable requirement trace |
| `scripts/external/check_second_goal_objective_trace.py` | verifies trace invariants |
| `tests/test_second_goal_objective_trace.py` | unit tests for the trace checker |

Current trace decision:

- REQ-1 PAPER_REF analysis is `complete_as_planning_input`.
- REQ-2 experiment planning is `planned_not_fully_executed`.
- REQ-3 accuracy closure toward `10_submission_initial` is `incomplete`.
- Ablation matrix covers the full current queue: XR-64-prep, XR-64A, XR-64B, XR-64C, XR-65, XR-66, XR-67, and XR-68.
- Goal completion remains `false`.

## Completion Gate

| Artifact | Role |
|---|---|
| `scripts/external/check_second_goal_completion_gate.py` | evaluates whether the full second goal may be marked complete |
| `tests/test_second_goal_completion_gate.py` | unit tests for current blocker state, strict pass state, leakage rejection, and summary output |
| `docs/resources/second_goal_completion_readiness_contract_2026_06_21.md` | current completion-readiness contract tying blockers, XR-64 readiness, and next allowed actions |
| `scripts/external/check_second_goal_completion_readiness_contract.py` | validates the readiness contract against `report_second_goal_status.py` and the completion gate |
| `tests/test_second_goal_completion_readiness_contract.py` | unit tests for readiness pass state and false-ready rejection |

Current gate decision:

- `completion_allowed=false`.
- Current blocker count is `28`.
- Primary blockers are pause state, `active_goal_complete=false`, incomplete objective trace, missing single-model trained result claim, blocked metric/protocol bridge, invalid direct submission comparison, missing XR-64 eval rows, missing XR-64 override JSON files, missing XR-64 training run evidence, and missing XR-64 post-run promotion evidence.
- Paper-target evidence manifest blockers are also active: required evidence is incomplete, direct paper-target comparison is blocked, and future bridge artifacts are missing.
- Guarded facts that currently pass: XR-63 oracle is not promotable, XR-64 leakage risk is `none`, authority links resolve, and the XR-64 command emitter is non-executing.
- Completion readiness contract reports `xr64_ready_to_train=false`, `can_run_lane=false`, `missing_eval_rows=8`, `missing_overrides=6`, and `leakage_risk=none`.
- Completion readiness now also tracks the XR-64 next prep selector: `ok=true`, `ready_after_resume=8`, `completed=0`, `blocked=2`, `invalid=0`, `next_id=XR64-EVAL-TRAIN-XR62A`, `next_status=ready_after_resume`, `next_allowed_to_run_now=false`, and `execute_supported=false`.

## XR-64 Resume Command Manifest

| Artifact | Role |
|---|---|
| `docs/resources/xr64_resume_command_manifest_2026_06_20.md` | human-readable future command contract |
| `docs/resources/xr64_resume_command_manifest_2026_06_20.json` | machine-readable future command contract |
| `scripts/external/check_xr64_resume_command_manifest.py` | verifies command contract invariants |
| `tests/test_xr64_resume_command_manifest.py` | unit tests for the command manifest checker |

Current manifest decision:

- `prep_commands` contains `10` future-only commands.
- Eval coverage is exactly `train/val x xr62a/xr39/xr56b/xr58a`.
- Build coverage is exactly `train` and `val`.
- Expected generated artifacts are `8` eval-row files and `6` override JSON files.
- Eval required input readiness is `6 / 6`; eval static inputs are ready.
- Build required input readiness is `0 / 8`; build steps are not ready until eval-row files exist.
- Post-run command count is `2`: candidate summary collection and decision-command printing after future XR-64A/B eval summaries exist.
- Prep runner contract is checked with `17` static safety/restartability markers: test-split refusal, train/val-only defaults, override-cleared eval, row validation, foreground logging, skip-path validation, and build-input validation.
- `default_allowed_to_run=false` while experiments are paused.

Emitter:

- `scripts/external/emit_xr64_resume_commands.py --section summary --format summary` prints counts and guards.
- `--section prep --format commands` prints future prep commands for review.
- `--section launch --format script` prints a commented script, not executable commands.
- `--section postrun --format commands` prints candidate collector review commands only; it does not train or evaluate models.
- `execute_supported=false`; this is not a launcher.

Next prep selector:

- `scripts/external/emit_xr64_next_prep_command.py --format summary` reports the next XR-64-prep command whose static inputs are ready and generated output is still missing.
- Current state selects `XR64-EVAL-TRAIN-XR62A` as the first ready-after-resume command.
- Current counts are `ready_after_resume=8`, `blocked=2`, `completed=0`, and `invalid=0`.
- `allowed_to_run_now=false` and `execute_supported=false`; the selector is a review aid, not an execution tool.

## XR-64 Prelaunch Packet

| Artifact | Role |
|---|---|
| `docs/resources/xr64_prelaunch_packet_2026_06_20.md` | human-readable future-only prelaunch packet |
| `docs/resources/xr64_prelaunch_packet_2026_06_20.json` | machine-readable prelaunch packet |
| `scripts/external/check_xr64_prelaunch_packet.py` | validates packet/source/manifest alignment without executing experiments |
| `tests/test_xr64_prelaunch_packet.py` | unit tests for packet pass state and execution-safety rejection |

Current packet decision:

- `allowed_to_execute=false`.
- `completion_allowed=false`.
- Current blocker count is `28`.
- Eval static input readiness is `6 / 6`, so `eval_inputs_ready=true`.
- Build input readiness is `0 / 8`, so `build_inputs_ready=false`; build steps remain blocked until XR-64-prep eval rows exist.
- Prelaunch phases are status precheck, eval-row generation after resume, override generation after resume, strict readiness, XR-64A/B launch after strict readiness, and post-run candidate review.
- The packet now cross-checks latest readiness sources: resume matrix axes `6`, resume gates `5`, prep ledger ready-after-resume units `8`, prep ledger blocked units `2`, blocker-map blockers `28`, paper-target preflight evidence count `7`, and output schema count `10`.
- The packet references `emit_xr64_resume_commands.py` only for command review; it does not execute train/eval commands.

## XR-64 Post-Run Evidence Contract

| Artifact | Role |
|---|---|
| `docs/resources/xr64_postrun_evidence_contract_2026_06_21.md` | human-readable post-run evidence contract |
| `docs/resources/xr64_postrun_evidence_contract_2026_06_21.json` | machine-readable post-run evidence contract |
| `scripts/external/check_xr64_postrun_evidence_contract.py` | validates post-run evidence requirements without running experiments |
| `scripts/external/write_xr64_ablation_provenance.py` | writes required XR-64 ablation provenance and eval-summary certification after future train/eval runs |
| `scripts/external/collect_xr64_postrun_candidates.py` | scans future patched test eval summaries and prepares promotion-helper candidates |
| `tests/test_xr64_postrun_evidence_contract.py` | unit tests for pass state and false-promotion rejection |
| `tests/test_write_xr64_ablation_provenance.py` | unit tests for provenance writer artifacts and promotion-helper compatibility |
| `tests/test_collect_xr64_postrun_candidates.py` | unit tests for future candidate discovery, latest selection, and command output |

Current contract decision:

- `evidence_state=missing_postrun_evidence`.
- `allowed_to_claim_promotion=false`.
- `allowed_to_mark_second_goal_complete=false`.
- Required lanes are XR-64A and XR-64B.
- Required checkpoint kinds are `best_track_p10`, `best_track_p5`, and `best_metric_track_center_px`.
- Required metrics are `metric_track_center_px`, `metric_track_p10_pct`, `metric_track_p5_pct`, and `metric_track_p1_pct`.
- Required ablation axes are `head`, `loss`, `lr`, and `teacher_model_training`.
- XR-64A/B each require lane-level ablation design, LR/optimizer snapshots, teacher provenance, and attribution reports.
- `metric_track_p1_pct` is required as paper-target bridge evidence, but current software promotion remains center/P10/P5 only.
- Final test evaluation must clear target override paths before any promotion claim.

## XR-64 Post-Run Promotion Decision

| Artifact | Role |
|---|---|
| `docs/resources/xr64_postrun_promotion_decision_2026_06_21.md` | human-readable current decision placeholder |
| `docs/resources/xr64_postrun_promotion_decision_2026_06_21.json` | machine-readable current decision placeholder |
| `scripts/external/decide_xr64_postrun_promotion.py` | read-only helper that scores future XR-64A/B eval summaries against current gates and bounded tradeoff floors |
| `tests/test_decide_xr64_postrun_promotion.py` | unit tests for no-evidence, promotion, missing lane, non-test rejection, override-cleared evidence, and bounded/unbounded tradeoff |
| `scripts/external/check_xr64_postrun_promotion_decision.py` | artifact checker that validates the no-evidence placeholder against helper output, required candidate matrix, override-clear schema, and bounded tradeoff floors |
| `tests/test_xr64_postrun_promotion_decision.py` | unit tests for promotion-decision artifact drift |

Current decision:

- `decision_status=missing_postrun_evidence`.
- `candidate_count=0`.
- `bounded_tradeoff_candidate_count=0`.
- `unbounded_tradeoff_candidate_count=0`.
- `p1_evidence_candidate_count=0`.
- `axis_certification_required=true`.
- `axis_certified_candidate_count=0`.
- `axis_claims_allowed=false`.
- `control_variables_checked=false`.
- `rejection_reason=missing_postrun_evidence`.
- `software_promotion_allowed=false`.
- `single_model_sota_claim_allowed=false`.
- `paper_level_completion_allowed=false`.

## XR-63 / XR-64 Path

| Artifact | Role |
|---|---|
| `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.md` | no-train oracle report |
| `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json` | machine-readable oracle evidence |
| `scripts/external/analyze_p10_teacher_target_oracle.py` | XR-63 diagnostic generator |
| `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md` | XR-64 design plan |
| `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md` | XR-64 implementation report |
| `scripts/external/build_xr64_teacher_target_overrides.py` | train/val teacher-target override builder |
| `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` | future train/val eval-row and override prep helper |
| `scripts/external/run_xr64_teacher_target_construction.sh` | future XR-64A/B/C train/eval runner |
| `scripts/external/check_xr64_resume_artifacts.py` | read-only resume artifact checker |
| `scripts/external/report_second_goal_status.py` | read-only second-goal status reporter |
| `tests/test_xr64_resume_artifacts.py` | checker unit tests |
| `tests/test_second_goal_status.py` | status reporter unit tests |
| `tests/test_track_target_override.py` | target override dataset/builder guard tests |

Current XR-64 artifact state:

- no train/val teacher `eval_rows.json` files exist yet.
- no XR-64 train/val override JSON files exist yet.
- no XR-64A/B/C training run exists yet.
- resume checker reports `ready=true`, `generated_complete=false` with `--allow-missing-generated`.

Expected generated files after resume:

```text
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr62a/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr39/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr56b/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/xr58a/eval_rows.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_val_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_val_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_train_overrides.json
data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_val_overrides.json
```

## Submission Target Gap

Target extracted from `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial/main.tex`:

| Target metric | Submission claim | Current software gate | Direct comparison status |
|---|---:|---:|---|
| center error | `0.1812 px` | `<16.468481131962367` | invalid until coordinate/protocol frame is matched |
| P10 | `99.97%` | `>35.02295998845781%` | apparent gap `64.94704001154219` pct points if same frame |
| P5 | `99.72%` | `>12.133503770828247%` | apparent gap `87.58649622917176` pct points if same frame |

Current target-gap artifacts:

| Artifact | Role |
|---|---|
| `docs/resources/second_goal_submission_target_gap_audit_2026_06_20.md` | human-readable submission target and protocol-gap audit |
| `docs/resources/second_goal_submission_target_gap_audit_2026_06_20.json` | machine-readable target claims and apparent gap values |
| `scripts/external/check_submission_target_gap.py` | validates target claims, current gates, direct-comparison blocker, and XR-64-first implications |
| `tests/test_submission_target_gap.py` | unit tests for the target-gap checker |

## Metric / Protocol Bridge

Current bridge status: `blocked_until_metric_frame_and_protocol_match`.

| Artifact | Role |
|---|---|
| `docs/resources/second_goal_metric_protocol_bridge_2026_06_20.md` | human-readable metric/protocol bridge |
| `docs/resources/second_goal_metric_protocol_bridge_2026_06_20.json` | machine-readable bridge state |
| `scripts/external/check_metric_protocol_bridge.py` | validates coordinate frame, protocol contract, hybrid-mode mismatch, P1 absence, and XR-64 readiness blockers |
| `tests/test_metric_protocol_bridge.py` | unit tests for the bridge checker |
| `docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.md` | human-readable post-XR64 PAPER_REF-backed decision tree |
| `docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json` | machine-readable post-XR64 decision rules |
| `scripts/external/check_second_goal_post_xr64_decision_tree.py` | validates post-XR64 claim levels, branch routing, pause state, and XR-64-prep priority |
| `tests/test_second_goal_post_xr64_decision_tree.py` | unit tests for post-XR64 decision tree validation |
| `docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.md` | human-readable no-execute follow-up launch/promotion contract |
| `docs/resources/second_goal_post_xr64_followup_experiment_contract_2026_06_21.json` | machine-readable follow-up branch gates for XR-64C/XR-65/XR-66/XR-67/XR-68 |
| `scripts/external/check_second_goal_post_xr64_followup_contract.py` | validates branch-specific prerequisites, blocked launch state, PAPER_REF basis, and no-execute validation commands |
| `tests/test_second_goal_post_xr64_followup_contract.py` | unit tests for false launch/promotion and missing branch evidence rejection |

Current bridge decision:

- direct comparison with `0.1812 px` is not allowed.
- current metric frame is `post-transform input coordinate frame`.
- current software metrics are track-centered `metric_track_center_px/P10/P5`, not proven hybrid scheduler mode.
- `metric_track_p1_pct` is not currently part of the active metric set.
- use current center/P10/P5 gates for software promotion until bridge evidence exists.

## Experiment Plan Groups

| Group | Representative artifacts | Current decision |
|---|---|---|
| heatmap/state representation | `xr27` through `xr33` result/plan docs | historical promoted path; superseded by later gates |
| loss-ratio / P5/P10 recovery | `xr34` through `xr56` plan docs | many closed branches; P10 still hard gate |
| refine/candidate/teacher refresh | `xr57` through `xr60` plan docs | no gate promotion |
| geometry auxiliary | `xr62_facet_geometry_aux_refresh_plan_2026_06_17.md` | promoted center only |
| teacher-target construction | `xr63`, `xr64` docs/scripts | current P0 after experiments resume |
| temporal/local/hardware follow-up | XR-65/XR-66/XR-67/XR-68 in current summary/TODO | deferred until XR-64 evidence exists |
| post-XR64 decision routing | `second_goal_post_xr64_decision_tree_2026_06_21.md` | maps XR-64A/B result patterns to XR-64C/XR-65/XR-66/XR-67/XR-68 |
| post-XR64 follow-up contract | `second_goal_post_xr64_followup_experiment_contract_2026_06_21.md` | keeps all follow-up launch gates blocked until branch-specific evidence exists |

## Current Experiment Queue

| Priority | Experiments | State |
|---|---|---|
| P0 | XR-64-prep, XR-64A, XR-64B | blocked by pause and missing generated XR-64 artifacts |
| P1 | XR-64C, XR-65, XR-66 | conditional after XR-64A/B evidence |
| P2 | XR-67, XR-68 | blocked by dense-trajectory absence or missing stable full-width teacher |

The queue is machine-readable in `docs/resources/second_goal_experiment_queue_2026_06_20.json`.
It must pass `scripts/external/check_second_goal_experiment_queue.py` before being treated as the current execution contract.

## Validation Commands

Read-only current-state checks:

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
python3 -m py_compile scripts/external/check_xr64_resume_artifacts.py
python3 -m py_compile scripts/external/report_second_goal_status.py
python3 -m py_compile scripts/external/check_current_result_synthesis.py
python3 -m py_compile scripts/external/check_second_goal_objective_trace.py
python3 -m py_compile scripts/external/check_second_goal_completion_gate.py
python3 -m py_compile scripts/external/check_xr64_resume_command_manifest.py
python3 -m py_compile scripts/external/emit_xr64_resume_commands.py
python3 -m py_compile scripts/external/check_xr64_prelaunch_packet.py
python3 -m py_compile scripts/external/check_xr64_postrun_evidence_contract.py
python3 -m py_compile scripts/external/decide_xr64_postrun_promotion.py
python3 -m py_compile scripts/external/check_xr64_postrun_promotion_decision.py
python3 -m py_compile scripts/external/check_second_goal_post_xr64_followup_contract.py
.venv/bin/python scripts/external/check_paper_ref_analysis_coverage.py --format summary
.venv/bin/python scripts/external/check_second_goal_experiment_queue.py --format summary
.venv/bin/python scripts/external/check_submission_target_gap.py --format summary
.venv/bin/python scripts/external/check_metric_protocol_bridge.py --format summary
.venv/bin/python scripts/external/check_current_result_synthesis.py --format summary
.venv/bin/python scripts/external/check_second_goal_accuracy_lift_decision_ladder.py --format summary
.venv/bin/python scripts/external/check_second_goal_objective_trace.py --format summary
.venv/bin/python scripts/external/check_second_goal_completion_gate.py --allow-incomplete --format summary
.venv/bin/python scripts/external/check_xr64_resume_command_manifest.py --format summary
.venv/bin/python scripts/external/emit_xr64_resume_commands.py --section summary --format summary
.venv/bin/python scripts/external/check_xr64_prelaunch_packet.py --format summary
.venv/bin/python scripts/external/check_xr64_postrun_evidence_contract.py --format summary
.venv/bin/python scripts/external/check_xr64_postrun_promotion_decision.py --format summary
.venv/bin/python scripts/external/decide_xr64_postrun_promotion.py --format summary
.venv/bin/python scripts/external/check_second_goal_post_xr64_followup_contract.py --format summary
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
.venv/bin/python -m pytest -q tests/test_second_goal_status.py tests/test_xr64_resume_artifacts.py tests/test_track_target_override.py
.venv/bin/python -m pytest -q tests/test_paper_ref_analysis_coverage.py
.venv/bin/python -m pytest -q tests/test_second_goal_experiment_queue.py
.venv/bin/python -m pytest -q tests/test_submission_target_gap.py
.venv/bin/python -m pytest -q tests/test_metric_protocol_bridge.py
.venv/bin/python -m pytest -q tests/test_current_result_synthesis.py
.venv/bin/python -m pytest -q tests/test_second_goal_accuracy_lift_decision_ladder.py
.venv/bin/python -m pytest -q tests/test_second_goal_objective_trace.py
.venv/bin/python -m pytest -q tests/test_xr64_resume_command_manifest.py
.venv/bin/python -m pytest -q tests/test_emit_xr64_resume_commands.py
.venv/bin/python -m pytest -q tests/test_second_goal_post_xr64_followup_contract.py
bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh scripts/external/run_xr64_teacher_target_construction.sh
```

Expected pause-state status summary:

```text
xr64_resume_status: generated_incomplete
xr64_ready_to_train: false
xr64_can_run_lane: false
xr64_missing_eval_rows: 8
xr64_missing_overrides: 6
xr64_leakage_risk: none
```

## Resume Gate

Before launching XR-64A/B after user resumes experiments:

1. Generate all train/val teacher eval rows.
2. Build all train/val override JSON files.
3. Run strict checker:

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary
```

Required strict state:

```text
resume_status: ready_to_train
ready_to_train: true
can_run_lane: true
missing_eval_rows: 0
missing_overrides: 0
leakage_risk: none
```

## Completion Status

Active goal remains incomplete.

Completed:

- analysis and experiment-plan integration.
- broad head/loss/LR/optimizer/distillation/teacher branch exploration.
- XR-63 oracle diagnostic.
- XR-64 code path and resume validation tooling.

Missing:

- XR-64 generated train/val artifacts.
- XR-64A/B/C training and full-test validation.
- final accuracy closure toward the submission target.

## Link And Staleness Risks

- `anlaysis` is intentionally spelled as currently present in the repository. Do not normalize it to `analysis` in links.
- `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md` is a source map with historical gates; its 2026-06-18 supersession note points to the current XR-64 plan.
- `docs/Paper-Backed-Experiment-Plan.md` is long-form and historical. Use the pause/current-summary/audit/runbook files above as current execution authority.
- XR-64 generated artifact paths are indexed even though the files are currently missing. Missing state is expected while experiments are paused.
