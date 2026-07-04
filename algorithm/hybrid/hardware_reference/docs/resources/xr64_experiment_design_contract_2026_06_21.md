# XR-64 Experiment Design Contract

Date: 2026-06-21

Experiments remain paused by user directive. This artifact is a non-executing design contract that turns the PAPER_REF-backed XR-64 queue into concrete launch and promotion conditions.

## Prompt Pack

| Field | Value |
|---|---|
| IDEA-Gen phase | P1 |
| Domain | Research Workflow |
| Parent skill | `paper-idea-generator` |
| Worker skill | `ablation-study-designer` |
| Caveman plugin | `caveman` |
| Objective | Convert XR-63 oracle headroom into leakage-safe train/val-only teacher-target supervision |
| Acceptance | XR-64A/B launch only after strict XR-64 artifact checker passes |

## Current Gates

| Metric | Gate |
|---|---:|
| center | `<16.468481131962367` |
| P10 | `>35.02295998845781` |
| P5 | `>12.133503770828247` |

XR-63 oracle has center `16.04023192701366`, P10 `36.48596938775512`, and P5 `13.41751700680271`, but it remains diagnostic only because it is a no-train test-aligned oracle.

## Global Controls

- Manifest root: `data/_internal/manifests/manifest1`.
- Split counts: train `5929`, val `844`, test `2238`.
- Test target override is not allowed.
- Direct submission comparison is not allowed.
- Strict checker: `scripts/external/check_xr64_resume_artifacts.py`.
- Required generated artifacts: 8 train/val eval-row files and 6 train/val override JSON files.
- Current state: missing eval rows `8`, missing overrides `6`, ready to train `false`, can run lane `false`, leakage risk `none`.

## Lane Contract

| Lane | Priority | GPU | Rule | LR | Launch Gate | Promotion Gate |
|---|---:|---|---|---:|---|---|
| XR-64-prep | P0 | none | train/val artifact generation | n/a | explicit resume only | readiness only |
| XR-64A | P0 | `cuda:0` | conservative | `3e-7` | strict checker ready | center, P10, or P5 active gate improvement |
| XR-64B | P0 | `cuda:1` | threshold-priority | `5e-7` | strict checker ready | P10/P5 gain with bounded center tradeoff |
| XR-64C | P1 | `cuda:0` | min-error diagnostic | `3e-7` | after XR-64A/B evidence | diagnostic unless full-test gate improves |

## Hyperparameter Commitments

### XR-64A

- Init: XR-62A center checkpoint.
- Teacher: XR-39 P10 soup.
- Override center L2: `0.0006`.
- Center guard: `0.0025`.
- P10 soft: `0.004`.
- P5 soft: `0.002`.
- State distill: `0.00035`.
- Trainable scope: heatmap head, state aux head, event adapter, event embed projection.

### XR-64B

- Init: XR-56B P5 checkpoint.
- Teacher: XR-39 P10 soup.
- Override center L2: `0.0008`.
- Center guard: `0.003`.
- P10 soft: `0.006`.
- P5 soft: `0.0025`.
- State distill: `0.0003`.

### XR-64C

- Init: XR-62A center checkpoint.
- Teacher: best available train/val selector.
- Override center L2: `0.001`.
- Purpose: transfer-pressure diagnostic, not default promotion lane.

## Rejection Rules

- Reject any result using test-derived target selection.
- Reject XR-64A/B launch before all generated artifacts pass the strict checker.
- Reject promotion if override provenance is missing.
- Reject metric claims that directly compare current track metrics to the submission pixel target before metric/protocol bridge evidence exists.

Machine-readable source: `docs/resources/xr64_experiment_design_contract_2026_06_21.json`.
