# HGTXR-SW Second-Goal Ablation Evidence Checklist

Date: 2026-06-21

This checklist is a non-executing evidence contract for the paused second-goal experiment program.
It connects the PAPER_REF-backed ablation plan, the current XR-64-first queue, and the post-run promotion rules.

## Prompt Pack

| Field | Value |
|---|---|
| Skill phase | `P1` |
| Domain | `Research Workflow` |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Objective | Convert PAPER_REF-derived experiments into verifiable ablation evidence requirements without running train/eval. |
| Target | HGTXR-SW raw event-count track model on `manifest1`; XR-64 teacher-target construction is the next valid branch. |

Constraints:

- Experiments remain paused until the user explicitly resumes.
- Test-derived target overrides are forbidden.
- Paper-level accuracy comparison is blocked until the metric/protocol bridge is unblocked.
- A single-model SOTA claim requires a leakage-safe trained checkpoint, not an oracle or command artifact.

## Promotion Policy

Current software promotion metrics:

- `metric_track_center_px`
- `metric_track_p10_pct`
- `metric_track_p5_pct`

Future paper-bridge evidence metric:

- `metric_track_p1_pct`

Claim levels:

| Level | Meaning | Required evidence |
|---|---|---|
| diagnostic improvement | signal that justifies the next experiment only | validation movement, oracle/bucket report, or failure-mode diagnostic |
| software promotion | one current software gate improves | leakage-safe trained checkpoint, full-test summary, no test override, tradeoff report |
| clean single-model promotion | one trained checkpoint owns all current software gates | center/P10/P5 all beat active gates with controls checked |
| paper-level completion | result can be compared to `10_submission_initial` | metric/protocol bridge unblocked and required paper-target evidence complete |

Current gates:

| Metric | Gate |
|---|---:|
| center | `<16.468481131962367` |
| P10 | `>35.02295998845781` |
| P5 | `>12.133503770828247` |

Promotion requires:

- trained leakage-safe checkpoint,
- full-test `eval_summary.json`,
- full-test `eval_rows.json`,
- cleared `data.track_target_override_path` for test evaluation,
- tradeoff report against all current software gates,
- P1 recorded as future bridge evidence when available.

Reject any result if:

- XR-63 oracle is used as a result,
- test-derived target override appears in training,
- only prelaunch or command artifacts exist,
- paper-level comparison is made while `direct_submission_comparison_allowed=false`.

## Ablation Matrix

| ID | Priority | Isolated Contribution | Required Evidence | Decision |
|---|---:|---|---|---|
| XR-64-prep | P0 | teacher-target data protocol and leakage-safe pseudo-target construction | 8 train/val `eval_rows.json`, 6 train/val override JSONs, strict checker `ready_to_train`, leakage risk `none` | readiness only; no model promotion |
| XR-64A | P0 | conservative teacher-target head/loss/LR branch | run root, trained checkpoint, LR manifest, optimizer snapshot, teacher provenance, attribution report, full-test center/P10/P5/P1, override provenance, tradeoff report | promote only if center, P10, or P5 beats the active gate |
| XR-64B | P0 | threshold-priority teacher-target branch | run root, trained checkpoint, LR manifest, optimizer snapshot, teacher provenance, attribution report, full-test center/P10/P5/P1, center/P5 tradeoff, override provenance | promote only if P10/P5 improves with bounded center tradeoff |
| XR-64C | P1 | min-error pseudo-target pressure diagnostic | validation transfer, full-test center/P10/P5/P1, overfit report, override provenance | diagnostic unless full-test gate improves without leakage |
| XR-65 | P1 | temporal-lite residual and self-supervised state distillation | temporal gain, center drift, same-scene comparison, full-test center/P10/P5/P1 | continue only after XR-64 non-collapse |
| XR-66 | P1 | confidence-gated local event update with fallback | confidence buckets, fallback comparison, local subset delta, full-test metrics if trained | train only if reliable subset improves center and aggregate gates hold |
| XR-67 | P2 | dense trajectory refinement and jitter control | dense segment count, before/after center/P10/P5/P1, jitter report | blocked until dense trajectories exist |
| XR-68 | P2 | optimizer/distillation/hardware-export tradeoff | student metrics, teacher-student tolerance, latency/resource report | blocked until stable improved full-width teacher exists |

## Current State

| Item | Value |
|---|---:|
| execution allowed | `false` |
| software promotion allowed | `false` |
| paper-level completion allowed | `false` |
| XR-64 ready to train | `false` |
| missing eval rows | `8` |
| missing overrides | `6` |

Machine-readable source: `docs/resources/second_goal_ablation_evidence_checklist_2026_06_21.json`.
