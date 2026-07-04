# HGTXR-SW PAPER_REF Experiment Trace

Date: 2026-06-21

This is a no-execute trace artifact. It binds PAPER_REF per-paper analyses to the current HGTXR-SW second-goal experiment queue without launching train/eval jobs.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Plugin | `caveman` |
| Objective | Bind PAPER_REF paper evidence to each current experiment ID and required validation evidence. |
| Target | HGTXR-SW raw event-count track model on `manifest1` |

Constraints:

- Experiments remain paused until explicit user resume.
- No test-derived target override may be used for training.
- Current promotion uses center/P10/P5 gates until paper metric/protocol bridge evidence is complete.
- XR-63 oracle remains diagnostic only.

## Current Gates

| Metric | Gate |
|---|---:|
| center | `<16.468481131962367` |
| P10 | `>35.02295998845781` |
| P5 | `>12.133503770828247` |

## Paper Evidence Groups

| Group | Key papers | Current use |
|---|---|---|
| geometry/state supervision | FACET, E-Track, Etracker, event-driven eye segmentation | XR-64A/B center and geometry guard; not a standalone replay |
| teacher/distillation | Local-Global Distillation, EV-Eye | XR-64 teacher-target construction and later XR-65/XR-68 distillation |
| temporal robustness | 3ET, AIS 2024/2025 surveys | XR-65 temporal-lite residual after XR-64 non-collapse |
| local fallback/blink | EX-Gaze, Swift-Eye, E-BTS, SCNN-AKF fusion | XR-66 confidence/fallback diagnostic |
| dense refinement | TimeLens-XL, EyeTrAES | XR-67 only after dense/continuous trajectories exist |
| deployment/distillation | Retina, SEE/ESDA, JaneEye, EyeCoD, low-power spiking gaze | XR-68 only after stable full-width teacher |

## Experiment Trace

| ID | Priority | Paper groups | Axes | Current status | Required evidence |
|---|---:|---|---|---|---|
| XR-64-prep | P0 | teacher/distillation, geometry/state | teacher, data protocol, loss | blocked by pause and missing artifacts | 8 train/val eval rows, 6 override JSONs, strict checker ready |
| XR-64A | P0 | geometry/state, teacher/distillation | head, loss, LR, teacher | waiting for XR-64-prep | trained checkpoint, full-test rows/summary, P1, provenance, attribution |
| XR-64B | P0 | teacher/distillation, geometry/state | head, loss, LR, teacher | waiting for XR-64-prep | trained checkpoint, full-test rows/summary, P1, center/P5 tradeoff, provenance |
| XR-64C | P1 | teacher/distillation | loss, LR, teacher | conditional after A/B | validation transfer, full-test metrics, overfit report |
| XR-65 | P1 | temporal robustness, teacher/distillation | head, loss, LR, self-supervised distillation | conditional after XR-64 non-collapse | temporal gain, same-scene comparison, center drift, full-test metrics |
| XR-66 | P1 | local fallback/blink | head, loss, data protocol | conditional after XR-64 diagnostics | confidence buckets, fallback comparison, local subset delta |
| XR-67 | P2 | dense refinement | data protocol, post-process | blocked by dense trajectory absence | dense segment count, same-scene before/after metrics, jitter proxy |
| XR-68 | P2 | deployment/distillation | optimizer, distillation, hardware export | blocked until stable teacher | student accuracy, teacher-student tolerance, latency/resource report |

## Rejection Rules

- Reject any trainable branch if `data.track_target_override_path` is non-null during test evaluation.
- Reject XR-64-prep readiness if any of the 8 eval-row files or 6 override JSON files are missing.
- Reject XR-64A/B promotion if no full-test software gate improves.
- Reject XR-64C as promotion unless a full-test gate improves without leakage.
- Reject XR-65/XR-66 launch before XR-64 post-run evidence selects the failure mode.
- Keep XR-67/XR-68 blocked until their prerequisite evidence exists.

## Current State

| Item | Value |
|---|---:|
| XR-64 ready to train | `false` |
| XR-64 can run lane | `false` |
| missing eval rows | `8` |
| missing overrides | `6` |
| XR-64 postrun candidates | `0` |
| completion allowed | `false` |
| blocker count | `28` |
| direct submission comparison allowed | `false` |

Machine-readable source: `docs/resources/second_goal_paper_ref_experiment_trace_2026_06_21.json`.
