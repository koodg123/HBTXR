# HGTXR-SW Second Goal Objective Trace

Date: 2026-06-20

Experiments are paused by user directive. This trace records current evidence against the full second-goal objective; it does not narrow or complete the objective.

## Prompt Pack

| Field | Value |
|---|---|
| Skill | `ablation-study-designer` |
| IDEA-Gen phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Objective | Trace the user-provided second goal against current evidence without running experiments. |
| Target | HGTXR-SW raw event-count track model on manifest1 train/val/test splits |
| Required artifact | machine-readable objective trace, human summary, and validator |

Constraints:

- Experiments are paused until explicit user resume.
- Test-derived target overrides are forbidden.
- Submission-level accuracy cannot be claimed until metric/protocol bridge is unblocked.
- Current software center/P10/P5 gates remain the promotion frame until paper-frame evaluator evidence exists.

## Requirement Trace

| Requirement | Category | Status | Promotion allowed | Evidence | Remaining gap |
|---|---|---|---|---|---|
| Analyze PAPER_REF papers in detail | `paper_ref_analysis` | `complete_as_planning_input` | `false` | `anlaysis/paper-ref/papers/index.md`, 31 per-paper `analysis.md` files, `check_paper_ref_analysis_coverage.py` | future paper additions must be re-indexed |
| Plan experiments across Head/Loss/LR/Optimizer/Self-supervised Distillation/Teacher training | `experiment_planning` | `planned_not_fully_executed` | `false` | `second_goal_paper_ref_current_ablation_plan_2026_06_20.md`, `second_goal_experiment_queue_2026_06_20.json`, `second_goal_post_xr64_decision_tree_2026_06_21.json`, queue/tree checkers | XR-64-prep/A/B/C are planned but not executed after pause |
| Use experiments to maximize accuracy toward `10_submission_initial` | `accuracy_closure` | `incomplete` | `false` | current result synthesis, submission target audit, metric/protocol bridge | XR-64 artifacts missing; XR-64A/B/C not run; direct paper-target comparison blocked |

False-completion guards:

- PAPER_REF planning coverage does not prove final accuracy closure.
- The experiment queue is a paused execution contract, not completed training evidence.
- The post-XR64 decision tree is routing/planning evidence only, not XR-64 execution or promotion evidence.
- Current best gates are split across owners, so single-model best/SOTA claims are not allowed.
- XR-63 oracle is not promotable because it is a test-aligned diagnostic.
- Direct comparison with `0.1812 px` is blocked while `direct_submission_comparison_allowed=false`.

## Current Evidence

Facts:

- PAPER_REF per-paper analysis index covers `31` analysis files.
- Current queue contains XR-64-prep, XR-64A, XR-64B, XR-64C, XR-65, XR-66, XR-67, and XR-68.
- Post-XR64 decision tree contains `7` result-pattern branches and keeps `XR-64-prep` as the minimal next worker after resume.
- Current active gates are center `16.468481131962367`, P10 `35.02295998845781`, and P5 `12.133503770828247`.
- XR-63 oracle is diagnostic only and not promotable as a trained result.
- XR-64 generated artifacts are missing: `8` eval-row files and `6` override JSON files.
- Metric/protocol bridge status is `blocked_until_metric_frame_and_protocol_match`.

Assumptions:

- `manifest1` remains the active software evaluation substrate.
- XR-64 teacher-target construction remains the next valid accuracy-improvement route unless new evidence supersedes it.
- Hardware/export branches remain secondary until software teacher quality improves.

## Ablation Matrix

| ID | Priority | Isolated contribution | Controls | Expected evidence | Current status |
|---|---:|---|---|---|---|
| XR-64-prep | P0 | teacher target construction/data protocol | manifest1 split, train/val only, no test override | 8 eval rows, 6 override JSONs, strict checker ready | missing generated artifacts |
| XR-64A | P0 | head/loss/LR/teacher override | XR-62A init, XR-39 teacher, same evaluator | full-test center/P10/P5, provenance, checkpoint | waiting for XR-64-prep |
| XR-64B | P0 | threshold-oriented loss and teacher selection | XR-56B init, XR-39 teacher, same evaluator | full-test center/P10/P5, tradeoff report, checkpoint | waiting for XR-64-prep |
| XR-64C | P1 | teacher-target pressure and overfit boundary | XR-62A init, train/val-only overrides, diagnostic unless full-test gate improves | validation transfer report, full-test center/P10/P5/P1, overfit report | conditional after XR-64A/B |
| XR-65 | P1 | self-supervised distillation and temporal regularization | best XR-64 checkpoint, frozen broad backbone, center drift stop rule | temporal gain and same-scene comparison | conditional after XR-64 non-collapse |
| XR-66 | P1 | fallback-safe local update and failure-bucket routing | confidence bucket report first, fallback route required, coordinate transform audit | confidence bucket report, fallback comparison, local-update subset center delta | conditional after XR-64 diagnostics |
| XR-67 | P2 | postprocess/dense trajectory refinement | dense trajectories required, same-scene before/after comparison, raw tuple interpolation controlled | dense segment count, before/after center/P10/P5/P1, jitter report | blocked by dense trajectory absence |
| XR-68 | P2 | optimizer/distillation/hardware export tradeoff | stable full-width teacher, teacher-student tolerance | student accuracy and latency/resource report | blocked until stable full-width teacher |

## Post-XR64 Planning Evidence

The post-XR64 decision tree is linked to REQ-2 as `experiment_planning_only`
evidence. It does not permit execution or promotion while experiments remain
paused. Its current contract:

| Field | Value |
|---|---|
| execute supported | `false` |
| claim levels | `4` |
| decision inputs | `3` |
| result-pattern branches | `7` |
| minimal next worker after resume | `XR-64-prep` |
| current next prep ID | `XR64-EVAL-TRAIN-XR62A` |
| direct submission comparison allowed | `false` |
| conditional next experiments | `XR-64C`, `XR-65`, `XR-66`, `XR-67`, `XR-68` |

Checker:

```bash
.venv/bin/python scripts/external/check_second_goal_post_xr64_decision_tree.py --format summary
```

## Guardrails

- Do not mark the active goal complete while the status reporter says `active_goal_complete=false`.
- Do not execute train/eval commands while `execution_state=paused_by_user_directive`.
- Do not promote XR-63 oracle as a trained result.
- Do not compare current `metric_track_center_px` directly with `0.1812 px` until direct submission comparison is allowed.
- Do not launch XR-64A/B until XR-64-prep generated artifacts pass strict checking.

## Next Required Evidence

1. After explicit user resume, generate the 8 XR-64 train/val teacher eval rows.
2. Generate the 6 XR-64 train/val override JSON files.
3. Run strict `check_xr64_resume_artifacts.py` and require `ready_to_train`.
4. Run XR-64A/B and compare full-test summaries against current software gates.
5. Unblock metric/protocol bridge before claiming paper-level accuracy closure.

## Completion Judgment

The goal is not complete.

Reasons:

- Accuracy closure toward `10_submission_initial` is not proven.
- XR-64 generated artifacts are missing.
- XR-64A/B/C have not run.
- Metric/protocol bridge blocks direct paper-target comparison.
- Experiments remain paused by user directive.

Machine-readable source: `docs/resources/second_goal_objective_trace_2026_06_20.json`.
