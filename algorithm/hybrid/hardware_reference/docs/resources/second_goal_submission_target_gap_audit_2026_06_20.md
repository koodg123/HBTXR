# HGTXR-SW Submission Target Gap Audit

Date: 2026-06-20

Experiments are paused by user directive. This audit is documentation and planning only.

## Purpose

This document connects the active HGTXR-SW second goal to the target claims in:

- `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial/main.tex`

The objective is not to claim completion. The objective is to make the remaining accuracy gap explicit and to prevent unsafe direct comparison between current software metrics and the paper-reported pixel error.

## Submission Target Extracted From `main.tex`

The submission draft claims the following deployed hybrid result:

| Target | Value | Source |
|---|---:|---|
| hybrid pupil-center error | `0.1812 px` | abstract and Table I |
| hybrid latency | `0.43 ms` | abstract and Table I |
| hybrid P10 | `99.97%` | Table I |
| hybrid P5 | `99.72%` | Table I |
| hybrid P1 | `99.61%` | Table I |

The draft also defines the algorithmic contract:

- search/track split with frame-guided search and event-guided local track.
- shared geometric pupil state `(x, y, a, b, theta)`.
- time-synchronous and geometry-stable supervision.
- stage-1 frame-guided pretraining, stage-2 hybrid learning, and stage-3 self-supervised distillation.
- Stage 1: `100` epochs, AdamW, LR `1e-3`.
- Stage 2: `200` epochs, AdamW, LR `1e-4`.
- Stage 3: reduced-depth track-path model slimming with self-supervised distillation.

Evidence locations in `main.tex`:

- abstract target: lines 48-52.
- three-stage training: lines 367-413.
- mode-wise metric table: lines 638-655.
- EV-Eye setup and Stage 1/2 hyperparameters: lines 683-695.
- evaluation interpretation: lines 817-842.

## Current Software State

Current active software gates:

| Metric | Current gate | Owner |
|---|---:|---|
| center | `<16.468481131962367` | XR-62A FACET geometry auxiliary refresh |
| P10 | `>35.02295998845781` | XR-39 mixed-leader soup `c25p45f30` |
| P5 | `>12.133503770828247` | XR-56B best-P5 |

Current strategic signal:

- XR-63 oracle reaches `16.04023192701366 / 36.48596938775512 / 13.41751700680271`.
- XR-63 is not promotable because it uses test-aligned no-train teacher selection.
- XR-64 is the current leakage-safe route for turning that oracle signal into trainable supervision.

Current execution state:

- `paused_by_user_directive`.
- XR-64 generated artifacts are incomplete.
- Missing XR-64 train/val teacher eval rows: `8`.
- Missing XR-64 train/val override JSON files: `6`.
- XR-64A/B/C training has not run.

## Direct Numeric Gap

If the current software gates and the submission table were treated as the same metric frame, the apparent gap would be:

| Metric | Current | Submission target | Apparent gap |
|---|---:|---:|---:|
| center error | `16.468481131962367` | `0.1812` | `90.88565746116096x` current/target |
| center error | `16.468481131962367` | `0.1812` | `16.287281131962366 px` absolute |
| P10 | `35.02295998845781%` | `99.97%` | `64.94704001154219` pct points |
| P5 | `12.133503770828247%` | `99.72%` | `87.58649622917176` pct points |

This direct comparison is not currently valid as promotion evidence.

## Protocol Gap

Existing project progress records that `metric_track_center_px` is post-transform input-coordinate error, not confirmed sensor-pixel error. Therefore direct comparison with the `0.1812 px` target is invalid until protocol and coordinate frame are matched.

The remaining target-gap checks are:

1. Verify the coordinate frame and scale of `metric_track_center_px` against the paper's pixel error definition.
2. Verify EV-Eye split protocol and train/val/test sample counts against the paper's EX-Gaze-following split claim.
3. Verify whether current track-only event-count experiments cover the paper's hybrid scheduler mode, not only event-side tracking.
4. Verify whether P10/P5/P1 are computed on the same target points and coordinate frame as Table I.
5. Keep latency and hardware claims out of the software accuracy promotion path until a stronger full-width teacher exists.

## Experiment Implications

The target gap does not justify reopening older low-return branches as P0.

Current decision:

| Priority | Experiment | Why it is aligned with the submission target |
|---|---|---|
| P0 | XR-64-prep | generates leakage-safe train/val teacher targets needed for hybrid target supervision |
| P0 | XR-64A | center-safe teacher-target branch from XR-62A center leader and XR-39 P10 teacher |
| P0 | XR-64B | threshold-priority teacher-target branch from XR-56B P5 leader and XR-39 P10 teacher |
| P1 | XR-64C | diagnostic for min-error teacher-target transfer limit |
| P1 | XR-65 | temporal-lite/state regularization only after XR-64 non-collapse |
| P1 | XR-66 | confidence-gated local update only after XR-64 diagnostics identify a usable subset |
| P2 | XR-68 | stage-3/hardware/distillation branch only after a stronger full-width teacher exists |

Do not treat the following as current P0:

- XR-15 immediate replay.
- XR-60/XR-61 candidate-head-only continuation.
- geometry-only replay.
- scalar P10-soft continuation.
- low-similarity weighting/subset/sampler.
- checkpoint soup/interpolation as a primary mechanism.
- local crop/update without confidence fallback.

## Completion Judgment

The second goal is still incomplete.

The reason is specific:

- The PAPER_REF and XR-eye-tracking analysis work is now documented and traceable.
- The experiment queue is now machine-readable and XR-64-first.
- But the submission-level accuracy closure is not reached.
- Direct comparison to the paper's `0.1812 px` claim is blocked by metric/protocol mismatch.
- XR-64 train/val artifacts and XR-64A/B training evidence are still missing.

Next concrete progress after experiments are resumed:

1. Generate XR-64 train/val teacher eval rows.
2. Build XR-64 train/val override JSON files.
3. Run strict XR-64 artifact checker.
4. Run XR-64A/B.
5. Evaluate full-test center/P10/P5 and the metric-frame bridge before claiming progress toward `0.1812 px`.
