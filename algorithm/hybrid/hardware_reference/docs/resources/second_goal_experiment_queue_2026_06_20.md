# HGTXR-SW Second Goal Experiment Queue

Date: 2026-06-20

Experiments are paused by user directive. This queue is the current execution contract for future resumption only.

## Current Gates

| Metric | Gate |
|---|---:|
| center | `<16.468481131962367` |
| P10 | `>35.02295998845781` |
| P5 | `>12.133503770828247` |

## Queue Summary

| ID | Priority | Status | Axes | Execute When |
|---|---:|---|---|---|
| XR-64-prep | P0 | blocked by pause and missing generated artifacts | teacher, data protocol, target override loss | after user resumes experiments |
| XR-64A | P0 | waiting for XR-64-prep | head, loss, LR, teacher | strict XR-64 checker reports `ready_to_train` |
| XR-64B | P0 | waiting for XR-64-prep | head, loss, LR, teacher | strict XR-64 checker reports `ready_to_train` |
| XR-64C | P1 | conditional after XR-64A/B | loss, LR, teacher | A/B underfit or fail to move validation |
| XR-65 | P1 | conditional after XR-64 non-collapse | head, loss, LR, self-supervised distillation | XR-64 yields a non-collapsing checkpoint |
| XR-66 | P1 | conditional after XR-64 diagnostics | head, loss, data protocol | confidence/fallback diagnostic shows a valid local-update subset |
| XR-67 | P2 | blocked by dense trajectory absence | data protocol, post-process | dense or continuous trajectories exist |
| XR-68 | P2 | blocked until stable full-width teacher | optimizer, distillation, hardware export | teacher improves software gates |

## Execution Order

1. Run XR-64-prep teacher/split units only after explicit resume.
2. Run `scripts/external/check_xr64_resume_artifacts.py` in strict mode.
3. Launch XR-64A on GPU0 and XR-64B on GPU1 only when strict checker passes.
4. Decide XR-64C only after A/B evidence.
5. Decide between XR-65 and XR-66 based on XR-64 failure mode.
6. Keep XR-67 and XR-68 blocked until their prerequisites are proven.

## Axis Coverage

| Axis | Covered By | Current State |
|---|---|---|
| Head modification | XR-64A/B, XR-65, XR-66 | P0 uses existing heatmap/state heads plus override supervision |
| Loss modification | XR-64A/B/C, XR-65, XR-66 | override/center/P10/P5/state-distill losses remain guarded |
| LR modification | XR-64A/B/C, XR-65 | low-LR continuation only; LR-only replay is stale |
| Optimizer modification | XR-68 only | optimizer-only probes are low-return until teacher improves |
| Self-supervised distillation | XR-65, XR-68 | deferred until XR-64 teacher-target evidence exists |
| Teacher model training | XR-64-prep/A/B/C | current highest leverage path |
| Data/protocol | XR-64-prep, XR-66, XR-67 | train/val-only target construction and dense trajectory blockers |
| Hardware/export | XR-68 | blocked until stable full-width teacher |

## Machine-Readable Queue

See `docs/resources/second_goal_experiment_queue_2026_06_20.json`.

## Rejection Rules

- Do not execute anything while the pause directive is active.
- Do not use test-derived target overrides.
- Do not reopen candidate-head-only, geometry-only, scalar P10-soft, low-similarity weighting, checkpoint soup, or blind local crop as P0 without a new mechanism.
- Do not start hardware/compression before software teacher quality improves.
