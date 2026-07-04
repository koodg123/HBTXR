# Second Goal Accuracy Lift Decision Ladder

Experiments remain paused by user directive. This artifact is a no-execute decision ladder for converting current metric-specific leaders and XR-63 oracle headroom into the next leakage-safe accuracy-improvement sequence.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | P1 |
| Domain | Research Workflow |
| Parent skill | `paper-idea-generator` |
| Target | HGTXR raw event-count eye tracking on manifest1 |
| Required artifact | Machine-checkable decision ladder with ablation matrix, controls, expected evidence, and blocked follow-up triggers |

Known facts:

- Current best gates are center `<16.468481131962367`, P10 `>35.02295998845781`, and P5 `>12.133503770828247`.
- These gates have different owners, so a single-model SOTA claim is not allowed.
- XR-63 oracle reaches center `16.04023192701366`, P10 `36.48596938775512`, and P5 `13.41751700680271`, but it is not promotable because it is a no-train test-aligned diagnostic.
- XR-64 generated artifacts are incomplete: missing eval rows `8`, missing overrides `6`, and post-run candidates `0`.
- Direct comparison to `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial` is blocked until metric/protocol evidence exists.

Assumptions:

- XR-64 is the highest-value path because it is the only current plan that can test whether XR-63 sample-wise teacher routing transfers without test leakage.
- P10 remains the hardest active gate; P5 is strong but fragile under P10-focused changes.
- Geometry-only, candidate-only, scalar P10-soft, and low-similarity replay branches have low expected return unless paired with new teacher-target or selector evidence.

## Current Gates

| Metric | Gate | Owner |
|---|---:|---|
| center | `<16.468481131962367` | XR-62A FACET geometry auxiliary refresh |
| P10 | `>35.02295998845781` | XR-39 mixed-leader soup `c25p45f30` |
| P5 | `>12.133503770828247` | XR-56B best-P5 |

## Ladder

| Stage | Decision | Next |
|---|---|---|
| current_gates | Use current center/P10/P5 gates as software promotion criteria while paper bridge is blocked. | oracle_signal |
| oracle_signal | Treat XR-63 as headroom evidence only. Convert through leakage-safe XR-64 train/val teacher-target construction. | xr64_transfer |
| xr64_transfer | Run XR-64-prep after explicit resume, then launch XR-64A/B only after strict readiness passes. | post_xr64_branching |
| post_xr64_branching | Select XR-64C/XR-65/XR-66/XR-67/XR-68 only from full-test XR-64A/B evidence. | paper_target_bridge |
| paper_target_bridge | Do not claim submission-level accuracy until coordinate frame, split protocol, hybrid scheduler, P1, and trained-candidate evidence exist. | completion_gate |

## Ablation Matrix

| ID | Priority | Main axis | Launch status | Required evidence |
|---|---|---|---|---|
| XR-64-prep | P0 | teacher targets, distillation | blocked until explicit resume | 8 eval rows, 6 overrides, strict checker ready |
| XR-64A | P0 | head/loss/LR/teacher transfer | blocked until XR-64-prep complete | full-test center/P10/P5/P1, no test override, axis certification |
| XR-64B | P0 | head/loss/LR/optimizer/teacher transfer | blocked until XR-64-prep complete | full-test center/P10/P5/P1, bounded center tradeoff, axis certification |
| XR-64C | P1 | loss/teacher diagnostic | blocked until XR-64A/B evidence | full-test gate movement, validation-transfer check |
| XR-65 | P1 | temporal-lite loss/distillation | blocked until XR-64 non-collapse | same-scene temporal comparison, center drift guard |
| XR-66 | P1 | confidence-gated local head/loss | blocked until no-train diagnostic | accepted subset center gain, fallback preservation |
| XR-67 | P2 | dense trajectory smoothing | blocked until dense segments exist | dense trajectory coverage, sparse-only rejection |
| XR-68 | P2 | stronger teacher/student | blocked until stronger teacher exists | teacher gate gain, student tolerance, hardware claim separation |

## Control Variables

- Keep manifest1 split identity fixed.
- Do not apply target override on test split.
- Promote software only from full-test center/P10/P5 evidence.
- Track P1 as future evidence, not as current gate owner.
- Keep XR-64A/B GPU assignment aligned with the design contract.
- Compare XR-64A/B against current gates, not directly against paper hybrid targets.

## Signoff Evidence

- Strict XR-64 generated-artifact checker report.
- Train/val eval-row coverage for all required teachers.
- Train/val override JSON coverage.
- XR-64A/B full-test `eval_summary.json`.
- XR-64A/B full-test `eval_rows.json`.
- Axis-certified ablation provenance.
- Post-run promotion decision report.
- Paper-target metric/protocol bridge payloads.

## Current Decision

No training or eval should run while paused. After explicit resume, the next worker remains `XR-64-prep`, starting with `XR64-EVAL-TRAIN-XR62A`. Completion remains blocked until XR-64 generated artifacts, trained runs, post-run candidates, and paper-target bridge evidence exist.
