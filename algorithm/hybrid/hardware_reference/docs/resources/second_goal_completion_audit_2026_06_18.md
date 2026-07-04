# HGTXR-SW Second Goal Completion Audit

Date: 2026-06-18

## Scope

Objective being audited:

1. Analyze papers under `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`.
2. Build accuracy-improvement experiments across head, loss, LR, optimizer, self-supervised distillation, and teacher training.
3. Use the planned experiments to maximize accuracy toward `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial`.

Latest user constraint:

- experiments are paused.
- current work must be documented.
- no new training/evaluation should run until explicit resume.

## Evidence Inventory

| Requirement | Current evidence | Status |
|---|---|---|
| PAPER_REF paper analysis | `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`; source path recorded; coverage matrix and experiment-axis mapping present | complete as planning input, but numeric priority rows are partly historical |
| XR-eye-tracking codebase analysis | `18` per-codebase `analysis.md` files under `anlaysis/xr-eye-tracking/codebases` | complete for current reference pass |
| XR-eye-tracking paper analysis | `21` per-paper `analysis.md` files under `anlaysis/xr-eye-tracking/papers` | complete for current reference pass |
| Aggregate analysis volume | `39` `analysis.md` files, `15590` total lines | verified by filesystem count |
| Head experiments | state aux, SimDR, refine head, candidate head, geometry auxiliary, heatmap-state paths are implemented/evaluated in the experiment history | partially complete; current best next head/protocol is XR-64 teacher-target selector |
| Loss experiments | center L2, hinge, P10/P5 soft threshold, geometry/ellipse aux, override target losses are implemented/evaluated or prepared | partially complete; XR-64 override losses still need full target files and execution |
| LR experiments | fixed-count/LR sweeps, AdamW fixed255k bracket, ultra-low LR polish, teacher-refresh brackets documented | complete for explored branches; not final accuracy target |
| Optimizer experiments | AdamW main contract plus Lion/ADOPT/SOAP-style optimizer planning and selected probes documented | partially complete; optimizer-only path judged low-return after current evidence |
| Self-supervised distillation | weak-distill branches and Stage3/distillation planning documented | partially complete; full Stage3 should wait for stronger teacher |
| Teacher model training | XR-58/XR-59 P10 teacher refresh attempted; XR-63 oracle and XR-64 teacher-target plan prepared | partially complete; leakage-safe train/val teacher targets not generated yet |
| Accuracy maximization toward submission | current gates improved to center `<16.468481131962367`, P10 `>35.02295998845781`, P5 `>12.133503770828247`; still far from submission target | not complete |

## Current Best Evidence

Active gates:

| Metric | Gate | Source |
|---|---:|---|
| center | `<16.468481131962367` | XR-62A FACET geometry auxiliary refresh |
| P10 | `>35.02295998845781` | XR-39 mixed-leader soup `c25p45f30` |
| P5 | `>12.133503770828247` | XR-56B best-P5 |

Best current strategic signal:

- XR-63 no-train teacher-target oracle reaches `16.04023192701366/36.48596938775512/13.41751700680271`.
- This beats all active gates, but it is not a valid trained model because it selects among test-aligned predictions.
- XR-64 is the current leakage-safe path to convert the oracle signal into trainable supervision.

## Current Blockers To Completion

The active objective is not complete.

Reasons:

1. Submission-level accuracy target has not been reached.
2. XR-64 train/val teacher `eval_rows.json` files do not exist yet.
3. XR-64 train/val override JSON files do not exist yet.
4. XR-64A/B/C training/evaluation has not run.
5. XR-65/XR-66/XR-67/XR-68 remain planned follow-ups, not completed evidence.
6. Experiments are intentionally paused by user directive.

## Artifact State

Implemented or prepared:

- `scripts/external/build_xr64_teacher_target_overrides.py`
- `scripts/external/run_xr64_teacher_target_construction.sh`
- `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`
- `scripts/external/check_xr64_resume_artifacts.py`
- `docs/resources/xr64_teacher_target_construction_plan_2026_06_18.md`
- `docs/resources/xr64_teacher_target_construction_implementation_2026_06_18.md`
- `docs/resources/xr64_resume_runbook_2026_06_18.md`
- `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`
- `docs/resources/experiment_pause_documentation_2026_06_18.md`

Missing:

- full train/val `eval_rows.json` for `xr62a`, `xr39`, `xr56b`, `xr58a`.
- `xr64a_conservative_*_overrides.json`.
- `xr64b_threshold_*_overrides.json`.
- `xr64c_minerror_*_overrides.json`.
- XR-64A/B/C trained checkpoints.

## Stale Or Historical Documents

`anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md` contains older gates and older immediate-priority rows from the XR-01/XR-15 period. It now has a 2026-06-18 supersession note pointing to the current authoritative plan.

Use these as current authority:

1. `docs/resources/experiment_pause_documentation_2026_06_18.md`
2. `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`
3. `docs/Paper-Backed-Experiment-Plan.md`
4. `docs/track/TODO.md`
5. `docs/track/PROGRESS.md`

## Resume-Only Execution Plan

Do not execute while pause is active.

When resumed:

| Step | Action | Evidence required |
|---:|---|---|
| 1 | Static validate XR-64 helper and runner | `bash -n` passes |
| 2 | Generate train eval rows teacher-by-teacher | each `eval_rows.json` exists and validates nonzero rows |
| 3 | Generate val eval rows teacher-by-teacher | each `eval_rows.json` exists and validates nonzero rows |
| 4 | Build conservative/threshold/min-error train and val overrides | six override JSON files exist and validate |
| 5 | Run strict XR-64 artifact checker | `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py` returns `0` |
| 6 | Launch XR-64A and XR-64B | train/eval logs, checkpoints, full-test summaries |
| 7 | Decide XR-64C only if A/B evidence justifies | documented gate comparison |
| 8 | Consider XR-65/XR-66 only after XR-64 | no execution before XR-64 evidence |

Detailed command runbook:

- `docs/resources/xr64_resume_runbook_2026_06_18.md`

## Completion Judgment

Current goal status: incomplete.

Completed:

- reference analysis and experiment-plan integration are substantially complete.
- multiple head/loss/LR/optimizer/distillation/teacher branches have been implemented or evaluated.
- current best next experiment list is documented.
- experiment pause state is documented.

Not completed:

- final accuracy objective.
- XR-64 execution and validation.
- submission-level accuracy closure.

Decision:

- keep active goal open.
- do not mark complete.
- next concrete progress is documentation/review only until user resumes experiments.
