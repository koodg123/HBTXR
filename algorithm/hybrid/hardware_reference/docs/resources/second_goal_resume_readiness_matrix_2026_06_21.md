# HGTXR-SW Second-Goal Resume Readiness Matrix

Date: 2026-06-21

This is a no-execute readiness artifact. It binds the user objective, PAPER_REF-derived experiment axes, XR-64 resume gate, and paper-target bridge blockers into one matrix.

## Current State

| Item | Value |
|---|---:|
| execution state | `paused_by_user_directive` |
| execute supported | `false` |
| goal complete | `false` |
| completion allowed | `false` |
| blocker count | `28` |
| XR-64 ready to train | `false` |
| XR-64 can run lane | `false` |
| missing eval rows | `8` |
| missing overrides | `6` |
| next prep ID | `XR64-EVAL-TRAIN-XR62A` |
| postrun candidates | `0` |
| direct submission comparison allowed | `false` |

## Axis Matrix

| Axis | Experiments | Current gate |
|---|---|---|
| head | `XR-64A`, `XR-64B`, `XR-65`, `XR-66` | blocked until XR-64-prep is ready to train |
| loss | `XR-64A`, `XR-64B`, `XR-64C`, `XR-65`, `XR-66` | blocked until train/val override JSONs exist |
| LR | `XR-64A`, `XR-64B`, `XR-64C`, `XR-65` | blocked until training lanes exist |
| optimizer | `XR-68` | deferred until stable full-width teacher exists |
| self-supervised distillation | `XR-65`, `XR-68` | deferred until XR-64 non-collapse evidence exists |
| teacher model training | `XR-64-prep`, `XR-64A`, `XR-64B`, `XR-64C` | highest priority after explicit resume |

## Resume Gates

| Gate | Status | Required Before |
|---|---|---|
| RG-1 planning coverage | satisfied for planning only | future experiment selection |
| RG-2 XR-64 prep generation | blocked by pause and missing outputs | XR-64A/B training |
| RG-3 XR-64 strict readiness | blocked | XR-64A/B launch |
| RG-4 postrun promotion | blocked, missing postrun candidates | software promotion claim |
| RG-5 paper-target bridge | blocked, missing required evidence | direct `10_submission_initial` comparison |

## Rejection Rules

- Do not run train/eval/GPU jobs while paused.
- Do not treat PAPER_REF analysis, XR-63 oracle evidence, or command manifests as trained promotion evidence.
- Do not launch XR-64A/B until strict XR-64 readiness passes.
- Do not compare directly with `10_submission_initial` until paper-target bridge evidence is complete.

Machine-readable source: `docs/resources/second_goal_resume_readiness_matrix_2026_06_21.json`.
