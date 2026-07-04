# HGTXR-SW Post-XR64 Decision Tree

Date: 2026-06-21

This is a non-executing ablation decision artifact for the paused second-goal program.
It defines how XR-64A/B evidence should route the next PAPER_REF-backed experiment without launching train/eval jobs.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Objective | Select the smallest next experiment after XR-64A/B by isolating teacher-target transfer, temporal residual, local fallback, dense trajectory, and hardware/student effects. |
| Source artifacts | `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`, `docs/resources/second_goal_paper_ref_current_ablation_plan_2026_06_20.md`, `docs/resources/second_goal_experiment_queue_2026_06_20.md`, `docs/resources/xr64_postrun_evidence_contract_2026_06_21.md` |
| Target | HGTXR-SW raw event-count track model on `manifest1`; full-test metrics center/P10/P5 and future P1 evidence. |
| Required artifact | Post-XR64 decision tree with branch conditions, controls, evidence, and rejection rules. |
| Verification evidence | JSON parse, status reporter pause state, and link from the current PAPER_REF ablation plan. |

## Claim Levels

| Level | Meaning | Allowed evidence | Not sufficient |
|---|---|---|---|
| diagnostic improvement | useful signal for follow-up only | validation movement, bucket gain, oracle or diagnostic report | software promotion |
| software promotion | one current software gate improved by a leakage-safe trained checkpoint | full-test `eval_summary.json`, checkpoint, no test override, tradeoff report | paper-level completion |
| clean single-model promotion | one trained checkpoint owns all current software gates | full-test center/P10/P5 all beat current gates with controls | direct comparison to `10_submission_initial` |
| paper-level completion | current software result is comparable to submission target frame | metric/protocol bridge complete and required paper-target evidence present | current center/P10/P5 gates alone |

## Known Facts

- Experiments are paused by user directive.
- XR-64-prep has not generated the required `8` train/val teacher eval-row files or `6` override JSON files.
- XR-64A/B/C have no post-run evidence yet.
- Current active software gates are center `<16.468481131962367`, P10 `>35.02295998845781`, and P5 `>12.133503770828247`.
- XR-63 oracle is diagnostic only; it cannot be promoted because it is test-aligned.
- Paper-level comparison remains blocked until the metric/protocol bridge is unblocked.

## Assumptions

- `manifest1` remains the active software split for near-term experiments.
- XR-64A/B will be evaluated on full test without target override paths.
- A valid next experiment must isolate one dominant cause instead of mixing temporal, local, distillation, and hardware changes at once.

## Decision Inputs

| Input | Required source | Required state |
|---|---|---|
| XR-64 readiness | `scripts/external/check_xr64_resume_artifacts.py --format summary` | `ready_to_train=true`, `can_run_lane=true`, missing eval rows `0`, missing overrides `0`, leakage risk `none` |
| XR-64A/B full-test summaries | future `runs/XR-64*/.../eval_summary.json` | center/P10/P5/P1 if available, no test override |
| Post-run evidence contract | `docs/resources/xr64_postrun_evidence_contract_2026_06_21.md` | required lane/checkpoint/metric evidence present |
| Promotion decision | `scripts/external/decide_xr64_postrun_promotion.py --format summary` | candidate count and axis certification reported |

## Decision Tree

| Result pattern after XR-64A/B | Interpretation | Next experiment | PAPER_REF basis | Control variables | Required evidence | Reject if |
|---|---|---|---|---|---|---|
| At least one checkpoint beats center, P10, or P5 gate with bounded tradeoff | Teacher-target transfer works | Promote checkpoint and run XR-65 only if temporal failure buckets remain | 3ET, MambaPupil, TDTracker temporal robustness | best XR-64 checkpoint, same manifest/evaluator, freeze broad backbone first | full-test metrics, temporal bucket report, center drift `<0.3 px` | temporal branch worsens all gates or lacks same-scene comparison |
| P10/P5 improve on validation but fail full-test gates | Teacher pressure exists but may underfit or overfit | XR-64C min-error diagnostic | teacher/distillation literature and XR-63 oracle | train/val-only overrides, same teacher pool, LR `3e-7` | validation-transfer report, full-test metrics, overfit report | test-derived selection or validation-only promotion |
| Center improves but P10/P5 collapse | Geometry/center path dominates threshold behavior | XR-66 confidence-gated local update diagnostic before new training | EX-Gaze, Swift-Eye, E-BTS local/fallback evidence | no train first, confidence buckets, fallback route required | bucket-level local-vs-global delta and aggregate gate check | confidence is saturated or fallback state is worse |
| P10 improves while P5 collapses | Threshold tuning is asymmetric | bounded XR-64B continuation only if P5 guard can be tightened | FACET/E-Track geometry guard plus teacher selection | same init/teacher, lower LR, stronger P5 guard only | P10/P5 tradeoff table and leakage check | center or P5 loss is undocumented or unbounded |
| A/B both fail to move validation | Teacher-target construction did not transfer | stop XR-64 family and reopen data/protocol diagnostics | EyeTrAES support density, failure-bucket analysis | no new architecture; analyze eval rows and buckets first | split/subject/session failure report | launching temporal/local/hardware changes without diagnosis |
| Dense/continuous segments become available | Postprocess refinement becomes testable | XR-67 dense trajectory refinement | EyeLoRiN, TimeLens-XL | same scenes before/after, no training required first | dense segment count, jitter, center/P10/P5/P1 before/after | dense trajectory coverage remains sparse |
| Stable full-width teacher improves gates | Student/hardware path becomes meaningful | XR-68 optimizer/distillation/hardware branch | Retina, SEE/ESDA, JaneEye, Local-Global Distillation | teacher-student tolerance, fixed teacher, export budget | student accuracy, latency/resource report | teacher is not stronger than current gates |

## Minimal Next Worker After Resume

The first executable worker remains XR-64-prep, not any branch in this document.
After explicit user resume, the current no-execute selector points to `XR64-EVAL-TRAIN-XR62A`.

## Signoff Rules

- Do not use this decision tree to bypass XR-64-prep readiness.
- Do not promote XR-63 oracle or validation-only gains.
- Do not compare against `10_submission_initial` targets until direct submission comparison is allowed.
- Do not start XR-65, XR-66, XR-67, or XR-68 unless the corresponding result pattern and required evidence above exist.

Machine-readable source: `docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.json`.
