# HGTXR-SW Second Goal Current PAPER_REF Ablation Plan

Date: 2026-06-20

## Scope

This document is the current execution-oriented bridge between:

- `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`
- the HGTXR-SW second-goal accuracy plan
- the latest validated experiment state under `software`

Experiments remain paused unless the user explicitly resumes train/eval execution.
This document is planning and documentation only.

## Prompt Pack

Goal:

- Convert PAPER_REF evidence into the next accuracy-improvement experiments without repeating closed branches.
- Preserve the HGTXR paper contract: hybrid search/track, event residual tracking, shared pupil-state supervision, teacher/distillation only when provenance-safe, and deployment-aware follow-up only after software accuracy improves.

Inputs and evidence sources:

- PAPER_REF corpus and extracted text under `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`.
- Source map: `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`.
- Per-paper traceability: `anlaysis/paper-ref/papers/index.md` and 31 generated `papers/*/analysis.md` files.
- Per-paper coverage checker: `scripts/external/check_paper_ref_analysis_coverage.py`.
- Current status: `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`.
- Completion audit: `docs/resources/second_goal_completion_audit_2026_06_18.md`.
- Resume runbook: `docs/resources/xr64_resume_runbook_2026_06_18.md`.
- Active backlog: `docs/track/TODO.md`.

IDEA-Gen phase: P1

Domain: Research Workflow

Parent skill: `paper-idea-generator`

Worker skill: `ablation-study-designer`

Caveman plugin: `caveman`

Assumptions:

- The current active gates remain center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- XR-63 oracle is diagnostic only because it uses test-aligned teacher selection.
- XR-64 is the first valid route to convert the oracle signal into trainable supervision.

Unknowns:

- Whether train/val teacher routing transfers to the test split without overfitting subject/session artifacts.
- Whether XR-64A/B will produce a non-collapsing checkpoint suitable for XR-65 temporal-lite residual training.
- Whether dense or continuous trajectories can be produced for EyeLoRiN-style refinement.

Acceptance criteria:

- Planned experiments are traceable to PAPER_REF methods.
- Current P0/P1/P2 priorities do not conflict with latest completed results.
- Closed branches are explicitly marked as low-return unless a new mechanism is introduced.
- Each experiment has controls, expected evidence, and rejection rules.

## Expert Council Summary

Role: Paper-to-experiment analyst / Computer-vision tracking engineer / Validation evaluator

- Paper-to-experiment analyst: PAPER_REF still supports geometry, adaptive event slicing, temporal robustness, confidence-gated local update, and distillation, but the latest evidence says these must be sequenced behind XR-64 teacher-target construction.
- Computer-vision tracking engineer: the active failure is not lack of another head-only tweak; it is sample-wise disagreement between complementary leaders. Teacher-target selection is the smallest credible mechanism that can exploit this.
- Validation evaluator: old P0 rows in historical documents must not be treated as current execution priority. Current execution authority is XR-64-first.

## Actual Sub-Agent Task Cards

```yaml
task_card:
  task_id: T-SEC-001
  sub_agent: "gpt5.3-codex-spark"
  role: "research analyst"
  objective: "Audit PAPER_REF analysis artifacts for representation in software/anlaysis/paper-ref and connection to accuracy-improvement experiments."
  file_ownership: []
  assigned_skill: ["ablation-study-designer"]
  inputs:
    - "anlaysis/paper-ref"
    - "docs/resources/second_goal_artifact_index_2026_06_18.md"
    - "docs/resources/second_goal_completion_audit_2026_06_18.md"
    - "docs/track/TODO.md"
  outputs:
    - "gap list"
    - "strongest paper-derived experiment axes"
    - "files/sections needing update"
  validation:
    - "cite local file evidence"
  dependencies: []
```

```yaml
task_card:
  task_id: T-SEC-002
  sub_agent: "gpt5.3-codex-spark"
  role: "evaluator"
  objective: "Audit current future experiment lists for consistency with latest result synthesis."
  file_ownership: []
  assigned_skill: ["computer-vision-expert"]
  inputs:
    - "docs/resources/current_work_summary_and_next_experiments_2026_06_18.md"
    - "docs/track/TODO.md"
    - "anlaysis/xr-eye-tracking/experiment_integration.md"
    - "anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md"
  outputs:
    - "prioritized experiment list"
    - "stale items to mark low priority"
  validation:
    - "cite local file evidence"
  dependencies: []
```

## PAPER_REF-To-Experiment Current Mapping

| PAPER_REF signal | Current HGTXR meaning | Current priority | Current action |
|---|---|---:|---|
| FACET, EllSeg, E-Track geometry | explicit pupil-state geometry is useful | P0 support | keep as center guard and override loss support, but do not run geometry-only replay |
| EX-Gaze local event update | local update must be confidence/relocalization gated | P1 conditional | XR-66 diagnostic after XR-64; no blind local crop |
| EyeTrAES adaptive slicing | event support density matters | P1/P2 conditional | revisit only if XR-64 indicates evidence-density failures |
| 3ET, MambaPupil, TDTracker, AIS survey temporal cues | temporal context can improve robustness | P1 conditional | XR-65 temporal-lite residual after XR-64 non-collapse |
| EyeLoRiN, TimeLens-XL | dense trajectory refinement may help | P2 blocked | reopen only when dense/continuous trajectories exist |
| Local-global distillation, DistillGaze | teacher/student transfer is useful after strong teacher exists | P1/P2 | XR-64 first, then Stage3/distillation if teacher improves |
| Retina, SEE/ESDA, JaneEye, EyeCoD | sparse/accelerator path matters | P2 | XR-68 only after stable full-width teacher |
| Swift-Eye, E-BTS | blink/occlusion buckets need special handling | P1 diagnostic | use failure-bucket analysis; train only with usable fallback |
| Grounded SAM | label preparation can help Stage1/search masks | P2 data prep | use only for offline label improvement, not runtime model |

## Current Ablation Matrix

| ID | Priority | Paper basis | Mechanism | Control variables | Expected evidence | Reject if |
|---|---:|---|---|---|---|---|
| XR-64-prep | P0 | XR-63 oracle plus teacher/distillation literature | generate train/val teacher eval rows and override JSONs | same manifest1 split, train/val only, no test override | 8 eval-row files and 6 override JSON files pass strict checker | any test-derived override is used or generated artifacts are incomplete |
| XR-64A | P0 | FACET geometry + teacher-target supervision | conservative teacher-target override | XR-62A init, XR-39 teacher, LR `3e-7`, center guard kept | full-test gate improvement without leakage | center/P10/P5 all fail gates or override provenance is invalid |
| XR-64B | P0 | threshold-oriented teacher selection | P10/P5 threshold-priority override | XR-56B init, XR-39 teacher, LR `5e-7`, P5 guard | P10 or P5 improves with documented center tradeoff | center/P5 collapse is not bounded or documented |
| XR-64C | P1 | oracle pressure diagnostic | pure min-error override | XR-62A init, train/val only, LR `3e-7` | diagnostic of transfer limit | overfits validation or uses test-derived selection |
| XR-65 | P1 | 3ET, MambaPupil, TDTracker | temporal-lite residual/state regularization | only after XR-64 non-collapse; freeze broad backbone | temporal gain without center drift `>0.3 px` | sparse manifest makes temporal signal untestable |
| XR-66 | P1 | EX-Gaze, Swift-Eye, E-BTS | confidence-gated local update diagnostic | no train first; bucketed confidence/fallback report | identifies subset where local update is better | confidence remains saturated or fallback state is worse |
| XR-67 | P2 | EyeLoRiN, TimeLens-XL | dense trajectory refinement | continuous/dense trajectories required | before/after center/P10/P5/jitter on same scenes | dense segment remains unavailable |
| XR-68 | P2 | Retina, SEE/ESDA, JaneEye | sparse/quant/distilled edge branch | stable full-width teacher required | accuracy within tolerance plus latency/resource report | software teacher is not stronger than current gates |

## Lane-To-PAPER_REF Evidence Trace

| Lane | Paper-derived evidence | Why it maps to this lane | Claim level before execution |
|---|---|---|---|
| XR-64-prep | teacher/distillation papers plus XR-63 oracle | converts complementary leader predictions into train/val-only pseudo-targets | readiness only |
| XR-64A | FACET, EllSeg, E-Track geometry supervision | conservative center-safe teacher-target transfer keeps geometry as a guard | planned software-promotion candidate |
| XR-64B | threshold-oriented teacher selection plus FACET/E-Track guards | tests P10/P5 recovery while explicitly bounding center/P5 collapse | planned software-promotion candidate |
| XR-64C | teacher-target pressure and distillation evidence | diagnostic for whether stronger pseudo-target pressure transfers or overfits | diagnostic unless full-test gate improves |
| XR-65 | 3ET, MambaPupil, TDTracker, AIS temporal robustness | only valid after XR-64 provides a non-collapsing checkpoint to regularize temporally | conditional diagnostic/promotion |
| XR-66 | EX-Gaze, Swift-Eye, E-BTS local/fallback evidence | confidence-gated local update is justified only after bucket evidence shows a reliable subset | diagnostic first |
| XR-67 | EyeLoRiN, TimeLens-XL dense refinement | postprocess smoothing requires dense or continuous trajectories, which are currently absent | blocked |
| XR-68 | Retina, SEE/ESDA, JaneEye, Local-Global Distillation | student/hardware path needs a stable improved full-width teacher first | blocked |

## Requested Axis Coverage

| Axis | Covered by | Current contract |
|---|---|---|
| Head modification | XR-64A/B, XR-65, XR-66 | use existing heatmap/state heads first; add temporal/local heads only after XR-64 evidence |
| Loss modification | XR-64A/B/C, XR-65, XR-66 | train/val-only override, center/P10/P5, state-distill, temporal/local losses remain guarded |
| LR modification | XR-64A/B/C, XR-65 | low-LR continuation around teacher-target transfer; LR-only replay is stale |
| Optimizer modification | XR-68 | optimizer/compression path waits for stronger full-width teacher |
| Self-supervised distillation | XR-65, XR-68 | deferred until XR-64 proves non-collapsing teacher-target transfer |
| Teacher model training | XR-64-prep, XR-64A/B/C | current highest-leverage path from XR-63 oracle to leakage-safe train/val supervision |
| Data/protocol | XR-64-prep, XR-66, XR-67 | split-safe override generation, confidence buckets, and dense-trajectory prerequisite checks |
| Hardware/export | XR-68 | blocked until software teacher quality improves |

## Stale Or Low-Return Items

The following are retained as historical evidence but are not current P0 execution items:

- XR-15 support-adaptive event-window as immediate priority.
- XR-60/XR-61 candidate-head-only continuation.
- geometry-only replay after XR-62.
- scalar P10-soft continuation after XR-56/XR-58/XR-59.
- low-similarity weighting/subset/sampler after XR-04/XR-09/XR-10.
- checkpoint soup/interpolation except bounded diagnostics.
- prev-pupil/local crop without confidence fallback.

## Current Execution Order After Resume

1. Run XR-64-prep split-by-split and teacher-by-teacher.
2. Run strict XR-64 artifact checker.
3. Launch XR-64A on GPU0 and XR-64B on GPU1 only after checker success.
4. Decide XR-64C only if A/B evidence shows a useful but underfit teacher-target signal.
5. Use XR-65 or XR-66 only after XR-64 evidence selects the failure mode.
6. Keep XR-67/XR-68 blocked until their prerequisites are proven.

## Post-XR64 Decision Artifact

Use `docs/resources/second_goal_post_xr64_decision_tree_2026_06_21.md` after XR-64A/B evidence exists.
It maps result patterns to the next smallest PAPER_REF-backed experiment:

- XR-65 only after a leakage-safe XR-64 checkpoint improves gates and temporal buckets still matter.
- XR-64C only when validation suggests teacher pressure but full-test transfer is weak.
- XR-66 only when center/threshold behavior indicates a confidence-gated local fallback diagnostic.
- XR-67 only when dense or continuous trajectory evidence exists.
- XR-68 only after a stable full-width teacher improves software gates.

The decision artifact is non-executing and does not override the current XR-64-prep readiness gate.

## Validation Evidence Required

- `bash -n` passes for XR-64 scripts.
- `scripts/external/check_xr64_resume_artifacts.py` returns strict ready state.
- Full-test summaries are produced with `data.track_target_override_path=null`.
- Results are compared against center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`.
- Any promoted result records checkpoint path, run root, config, teacher path, override path, and leakage check.

## Residual Risks

- The teacher-target oracle may not transfer from train/val to test.
- Current validation split may not expose the same subject/session failures as test.
- XR-64 could improve P10 while hurting P5 or center; promotion must remain metric-specific.
- Dense-trajectory and hardware-facing branches are blocked by missing prerequisite evidence, not by implementation preference.
