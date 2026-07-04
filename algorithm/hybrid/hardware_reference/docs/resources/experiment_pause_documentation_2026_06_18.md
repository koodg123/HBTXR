# Experiment Pause Documentation

Date: 2026-06-18

## User Directive

Experiments are paused for now. The current task is documentation only:

- stop train/eval execution.
- record all completed work and current state.
- keep the next experiment list documented, but do not launch it until the user explicitly resumes execution.

## Runtime State At Pause

| Item | State |
|---|---|
| HGTXR train/eval process search | no active process matched `train_hbtxr`, `eval_hbtxr`, `run_xr64`, `run_xr`, or `scripts/external/run_.*train` |
| GPU0 | `15 MiB` used, `15827 MiB` free, `0%` utilization |
| GPU1 | `15 MiB` used, `15827 MiB` free, `0%` utilization |

Conclusion: no experiment is currently running.

## Current Accuracy Gates

| Gate | Current value | Source |
|---|---:|---|
| center | `<16.468481131962367` | XR-62A FACET geometry auxiliary refresh |
| P10 | `>35.02295998845781` | XR-39 mixed-leader soup `c25p45f30` |
| P5 | `>12.133503770828247` | XR-56B best-P5 |

Interpretation:

- XR-62A is the current center leader.
- XR-39 remains the strict P10 leader.
- XR-56B remains the P5 leader.
- Candidate-only, geometry-only, scalar P10-soft, and low-similarity weighting branches are closed as low-return primary work unless paired with a new selector or pseudo-target mechanism.

## Completed Analysis Work

Reference and planning artifacts:

- `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`
- `anlaysis/xr-eye-tracking/DETAILED_CODEBASE_ANALYSIS.md`
- `anlaysis/xr-eye-tracking/DETAILED_PAPER_ANALYSIS.md`
- `anlaysis/xr-eye-tracking/experiment_integration.md`
- `anlaysis/xr-eye-tracking/codebases/*/analysis.md`
- `anlaysis/xr-eye-tracking/papers/*/analysis.md`
- `docs/Paper-Backed-Experiment-Plan.md`
- `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`

Current analysis depth:

- XR-eye-tracking codebase analysis was regenerated to reference-document depth across `18` codebase folders.
- XR-eye-tracking paper analysis was regenerated to reference-document depth across `21` paper folders.
- PAPER_REF analysis was integrated into second-goal experiment planning.

## Completed Experiment Track Summary

Key progression:

1. Raw event-count baseline and CUDA readiness were restored.
2. Event tensor zero-input bug was diagnosed and fixed.
3. Fixed-count and center-aware Stage2 sweeps established the early raw-mode baseline.
4. Fixed-count expansion, AdamW fixed255k, geometry auxiliary, state auxiliary, distillation, checkpoint soup/interpolation, candidate head, and teacher refresh branches were evaluated.
5. XR-62A promoted the center gate to `16.468481131962367`.
6. XR-63 no-train teacher-target oracle showed real headroom: `16.04023192701366/36.48596938775512/13.41751700680271`.

Current conclusion:

- The next valuable mechanism is not another standalone training replay.
- XR-63 shows complementary teacher predictions exist.
- XR-64 is the leakage-safe training protocol for converting that oracle headroom into trainable supervision.

## XR-64 Implementation State

Implemented:

- `scripts/external/build_xr64_teacher_target_overrides.py`
- `scripts/external/run_xr64_teacher_target_construction.sh`
- `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`
- `scripts/external/check_xr64_resume_artifacts.py`
- dataset target override fields:
  - `track_target_override_state`
  - `track_target_override_weight`
  - `meta.track_target_override_source`
- config guards:
  - `data.track_target_override_path`
  - `data.allow_test_target_override`
- additive Stage2 target override loss path.
- test-manifest override guard.

Validated before the pause:

- XR-64 scripts passed syntax/static validation.
- builder and related scripts passed `py_compile`.
- targeted tests covered target override and track-loss paths.
- A/B construction runner dry-runs passed.
- direct 8-row XR-62A smoke eval succeeded under `/tmp/xr64_eval_smoke_20260618_0149`.
- XR-64 resume checker tests cover missing-generated, complete-artifact, and test-split leakage paths.

Important limitation:

- the direct 8-row smoke eval is only a path validation.
- it is not a full train/val teacher-target generation result.
- it is not a promotion result.

## XR-64 Current Artifact State

No full XR-64 target artifacts exist yet.

Expected but missing:

- `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/*/eval_rows.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/val/*/eval_rows.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_val_overrides.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_train_overrides.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64b_threshold_val_overrides.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_train_overrides.json`
- `data/_internal/manifests/manifest1/xr64_teacher_targets/xr64c_minerror_val_overrides.json`

The first all-in-one XR-64 prep attempt stopped during `train/xr62a` without producing `eval_rows.json`.

## XR-64 Helper Hardening

`scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` was updated for future resumable execution:

- `XR64_ACTIONS` selects `eval`, `build`, or both.
- `XR64_SPLITS` selects `train`, `val`, or both.
- `XR64_TEACHERS` selects `xr62a`, `xr39`, `xr56b`, `xr58a`, or a subset.
- foreground logging uses `tee`.
- JSON output validation runs after each eval/build step.
- the helper refuses `test` split for target generation.
- failure trap records `XR64_PREP_ERROR`.

These changes were made for future execution only. The helper was not executed after the pause directive.

## Recorded Next Experiments

These are recorded, not currently running:

| Priority | ID | Purpose | Status |
|---|---|---|---|
| P0 | XR-64A | conservative teacher-target selector | paused until train/val target rows and overrides exist |
| P0 | XR-64B | threshold-priority teacher-target selector | paused until train/val target rows and overrides exist |
| P1 | XR-64C | min-error teacher-target diagnostic | run only after A/B or as bounded diagnostic |
| P1 | XR-65 | teacher-target temporal-lite residual | depends on XR-64 evidence |
| P1 | XR-66 | EX-Gaze confidence-gated local update diagnostic | depends on confidence bucket evidence |
| P2 | XR-67 | EyeLoRiN/dense-trajectory reopening | blocked until dense/continuous trajectories exist |
| P2 | XR-68 | sparse/quant/distilled edge branch | deferred until full-width teacher stabilizes |

## Resume Checklist

When the user resumes experiments:

1. Re-run static validation for XR-64 helper scripts.
2. Generate one split/teacher at a time if full helper execution is too opaque.
3. Verify each generated `eval_rows.json` has nonzero valid rows.
4. Build train/val-only override JSON files.
5. Run strict checker: `.venv/bin/python scripts/external/check_xr64_resume_artifacts.py`.
6. Launch XR-64A on GPU0 and XR-64B on GPU1 only after override files exist.
7. Keep test-derived target override generation disabled.

Detailed resume runbook:

- `docs/resources/xr64_resume_runbook_2026_06_18.md`

## Documentation Updated

- `docs/resources/experiment_pause_documentation_2026_06_18.md`
- `docs/resources/xr64_resume_runbook_2026_06_18.md`
- `docs/resources/current_work_summary_and_next_experiments_2026_06_18.md`
- `docs/Paper-Backed-Experiment-Plan.md`
- `docs/track/PROGRESS.md`
- `docs/track/TODO.md`
- `docs/track/log.md`
