# Current Work Summary And Next Experiments

Date: 2026-06-18

## Scope

This note summarizes the current HGTXR-SW second-goal state:

- paper/reference analysis status,
- software experiment status,
- XR-63/XR-64 teacher-target branch status,
- next experiment list that should be run or implemented next.

The active objective remains accuracy improvement under the HGTXR paper scope:
hybrid search/track, event residual tracking, pupil-state supervision, teacher/distillation when provenance-safe, and deployment-aware follow-up only after software accuracy stabilizes.

## Experiment Pause Snapshot

User directive on 2026-06-18:

- stop experiments for now.
- document all completed work and current state.
- do not launch new training/evaluation until the user explicitly resumes execution.

Runtime check at the pause point:

| Check | Result |
|---|---|
| HGTXR train/eval process search | no active process matched `train_hbtxr`, `eval_hbtxr`, `run_xr64`, or `scripts/external/run_.*train` |
| GPU0 | `15 MiB` used, `15827 MiB` free, `0%` utilization |
| GPU1 | `15 MiB` used, `15827 MiB` free, `0%` utilization |

Current execution state:

- No experiment is currently running.
- `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` has been hardened for resumable split/teacher execution, foreground logging, JSON validation, and test-split refusal, but it has not been executed after the pause directive.
- `data/_internal/manifests/manifest1/xr64_teacher_targets/` currently contains no generated `eval_rows.json` or override JSON files.
- The earlier direct 8-row XR-62A smoke eval remains a path validation only, not a full experiment result.
- Completion audit: `docs/resources/second_goal_completion_audit_2026_06_18.md` marks the active second goal incomplete until XR-64 execution and final accuracy closure are verified.
- Resume runbook: `docs/resources/xr64_resume_runbook_2026_06_18.md` records the exact future command sequence and validation gates.

## Current Active Gates

Current gates from the latest validated plan state:

| Metric | Active gate | Owner |
|---|---:|---|
| center | `<16.468481131962367` | XR-62A FACET geometry auxiliary refresh |
| P10 | `>35.02295998845781` | XR-39 mixed-leader soup `c25p45f30` |
| P5 | `>12.133503770828247` | XR-56B best-P5 |

Interpretation:

- Center improved most recently through FACET-style geometry auxiliary refresh.
- P10 remains the hardest unresolved gate.
- P5 is strong but fragile; many P10-focused branches gave back P5.
- Repeating geometry-only, candidate-only, scalar P10-soft, or low-similarity weighting branches has low expected return unless paired with a new teacher-target or selector signal.

## Analysis Artifacts Completed

### PAPER_REF

Reference path:

- `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`

Existing corpus includes PDF files, extracted text under `tmp_pdf_text/`, and summary JSON files:

- `tmp_pdf_text/index.json`
- `tmp_pdf_text/keyword_snippets.json`
- `tmp_pdf_text/metric_mentions.json`
- `eye_tracking_document_analysis_ko.md`

Current use:

- PAPER_REF is mapped into HGTXR experiment axes in `anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md`.
- The high-value signals are FACET/EllSeg/E-Track geometry, EyeLoRiN inference-time refinement, EX-Gaze confidence/local update, 3ET/MambaPupil/TDTracker/AISSM temporal robustness, and DistillGaze/local-global distillation.

### XR-Eye-Tracking Codebase/Paper Analysis

Analysis root:

- `anlaysis/xr-eye-tracking`

Current state:

- `codebases/*/analysis.md`: 18 codebase analyses.
- `papers/*/analysis.md`: 21 paper analyses.
- `DETAILED_CODEBASE_ANALYSIS.md` and `DETAILED_PAPER_ANALYSIS.md` retain global synthesis.
- `experiment_integration.md` maps the reference analysis into HGTXR experiment ideas.
- `scripts/external/deepen_xr_eye_tracking_analysis.py` can regenerate source-grounded deep sections.

Recent quality upgrade:

- Each per-codebase document now includes source/layer structure, core files, README evidence, symbol-level entry points, dataflow, code-quality risk, HGTXR conversion options, and next line-by-line targets.
- Each per-paper document now includes problem/method/algorithm/experiment/result evidence, HGTXR primitive mapping, scope ruling, ablation options, and verification gates.

## Experiment Status Analysis

### Exhausted Or Low-Return Axes

These axes should not be repeated as primary P0 work without a new mechanism:

| Axis | Evidence | Decision |
|---|---|---|
| scalar P10-soft continuation | XR-56/XR-58/XR-59 narrowed but failed to beat XR-39 P10 | close as standalone branch |
| candidate-head only | XR-60 and XR-61 failed to promote | do not replay without teacher-target selector |
| geometry-only auxiliary replay | XR-62 improved center but not P10/P5 | useful as support signal, not next standalone branch |
| low-similarity weighting/subset/sampler | XR-04/XR-09/XR-10 failed to promote | keep diagnostics only |
| checkpoint soup/interpolation | XR-39 promoted, later soups often failed | use only bounded diagnostic |
| prev-pupil/local crop without confidence fallback | previous `prev_pupil_anchor` failed badly | require confidence/relocalization gate first |

### High-Value Current Branch

XR-63 oracle diagnostic proves sample-wise teacher routing has headroom:

| Diagnostic | Center | P10 | P5 |
|---|---:|---:|---:|
| XR-63 oracle | `16.04023192701366` | `36.48596938775512` | `13.41751700680271` |

This is not a valid training result because it is a no-train oracle over test-aligned predictions, but it shows the current leader set contains complementary predictions.

XR-64 implements the leakage-safe way to turn this into a trainable protocol:

- generate train/val eval rows for XR-62A, XR-39, XR-56B, XR-58A,
- build train/val-only target override JSON files,
- train with additive override losses,
- keep original `cur_state` as metric/guard target,
- reject test-derived override training.

## XR-64 Current State

Implemented:

- `scripts/external/build_xr64_teacher_target_overrides.py`
- `scripts/external/run_xr64_teacher_target_construction.sh`
- `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`
- dataset fields: `track_target_override_state`, `track_target_override_weight`, `meta.track_target_override_source`
- config guards: `data.track_target_override_path`, `data.allow_test_target_override`
- loss terms: `track_target_override_*` and aux override losses
- test split override guard

Validation already passed:

- `bash -n` for XR-64 scripts
- `py_compile` for builder/generator scripts
- targeted pytest for target override and track loss paths
- `DRY_RUN=1` for XR-64 A/B construction runner
- 8-row smoke eval using XR-62A checkpoint succeeded on `/tmp/xr64_train_smoke_manifest.jsonl`

Current incomplete item:

- Full `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` was started once, but stopped without producing eval rows. The 8-row direct eval smoke succeeded, so the failure is not a basic config/checkpoint/CUDA issue. Re-run with clearer logging or split-by-split commands.

## Next Experiment List

### P0. XR-64A Conservative Teacher-Target Construction

Goal:

- convert XR-63 oracle headroom into a center-safe trainable branch.

Preconditions:

- train/val eval rows exist for XR-62A, XR-39, XR-56B, XR-58A.
- `xr64a_conservative_train_overrides.json` exists.
- no test override file is used for training.

Knobs:

| Knob | Value |
|---|---|
| init | XR-62A center checkpoint |
| teacher | XR-39 P10 soup |
| override rule | conservative |
| LR | `3e-7` |
| override center L2 | `0.0006` |
| center guard | `0.0025` |
| P10 soft | `0.004` |
| P5 soft | `0.0020` |
| state distill | `0.00035` |
| trainable scope | heatmap head, state aux head, event adapter, event embed projection |

Run:

```bash
bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0
```

Gate:

- promote if full-test center `<16.468481131962367`, or P10 `>35.02295998845781`, or P5 `>12.133503770828247` with documented tradeoff.

### P0. XR-64B Threshold-Priority Teacher-Target Construction

Goal:

- recover P10/P5 threshold hits while preserving the XR-56B P5 behavior.

Knobs:

| Knob | Value |
|---|---|
| init | XR-56B P5 checkpoint |
| teacher | XR-39 P10 soup |
| override rule | threshold priority |
| LR | `5e-7` |
| override center L2 | `0.0008` |
| center guard | `0.0030` |
| P10 soft | `0.006` |
| P5 soft | `0.0025` |
| state distill | `0.00030` |

Run:

```bash
bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1
```

Gate:

- P10 recovery is primary, but center/P5 collapse invalidates promotion unless explicitly documented as diagnostic.

### P1. XR-64C Min-Error High-Pressure Diagnostic

Goal:

- test how much train/val pure min-error pseudo-target pressure transfers.

Use only after A/B complete or if A/B fail to move validation.

Knobs:

| Knob | Value |
|---|---|
| init | XR-62A |
| override rule | min error |
| LR | `3e-7` |
| override center L2 | `0.0010` |

Risk:

- highest overfit risk; never use test-derived target selection.

### P1. XR-65 Teacher-Target Temporal-Lite Residual

Goal:

- add paper-backed temporal regularization after XR-64 proves teacher-target utility.

References:

- 3ET, MambaPupil, TDTracker, AISSM.

Precondition:

- XR-64A/B yields at least one non-collapsing checkpoint or strong diagnostic.
- dense trajectory availability is checked; if current manifest remains sparse, temporal smoothing must be framed as regularization rather than trajectory filtering.

Initial knobs:

| Knob | Candidate |
|---|---|
| init | best XR-64 checkpoint |
| LR | `2e-7` to `3e-7` |
| trainable | event adapter and event embed projection only, plus head if needed |
| distill state | `0.00020` to `0.00035` |
| stop rule | val center worsens by `>0.3 px` from active gate |

### P1. XR-66 EX-Gaze Confidence-Gated Local Update Diagnostic

Goal:

- revisit local/event patch only with a confidence/fallback gate.

References:

- EX-Gaze, Swift-Eye.

Precondition:

- no-train confidence bucket analysis identifies accepted subset where local update can improve center by at least `0.2 px`.
- fallback route preserves aggregate center/P10/P5.

Rules:

- no fallback, no training.
- no latency/hardware claim until software gate improves.
- local crop/coordinate transform must be audited before training.

### P2. XR-67 Dense-Trajectory/EyeLoRiN Reopen

Goal:

- reopen EyeLoRiN-style inference-time refinement only if dense trajectories exist.

Current status:

- existing manifest has no dense segment under `50000us`; prior XR-02 returned no change.

Gate:

- require dense/continuous split or generated prediction trajectories before spending GPU time.

### P2. XR-68 Hardware/Distillation Compression Branch

Goal:

- after a stable full-width teacher exists, start Retina/SEE/ESDA/DistillGaze-style sparse/quant/distilled branch.

Status:

- deferred. Accuracy teacher is not stable enough for compression target.

## Immediate Next Commands

Execution is paused by user directive. The commands below are recorded for resumption only and should not be launched until the user explicitly resumes experiments.

Prepare XR-64 train/val eval rows and overrides:

```bash
bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh
```

If the all-in-one script stalls or exits without rows, run one teacher/split at a time using `scripts/external/eval_hbtxr.py`, then build overrides with:

```bash
.venv/bin/python scripts/external/build_xr64_teacher_target_overrides.py \
  --reference-config runs/XR-62/eval_fixed255k_xr62_a_xr56b_facetaux_p10_preserve_adamw_lr3e_7_g32_hm0_004_off0_0015_c0_0025_p10soft0_004_p5soft0_0020_auxc0_0006_auxa0_0125_auxt0_005_best_track_p10_test_gpu0_w0_20260618_004307/hypers/resolved_config.json \
  --manifest data/_internal/manifests/manifest1/train_manifest.jsonl \
  --split train \
  --baseline xr62a \
  --rule conservative \
  --output data/_internal/manifests/manifest1/xr64_teacher_targets/xr64a_conservative_train_overrides.json \
  --eval xr62a=data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr62a/eval_rows.json \
  --eval xr39=data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr39/eval_rows.json \
  --eval xr56b=data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr56b/eval_rows.json \
  --eval xr58a=data/_internal/manifests/manifest1/xr64_teacher_targets/eval_rows/train/xr58a/eval_rows.json
```

Then launch:

```bash
bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0
bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1
```

## Decision Summary

The next experiment should not be another geometry/candidate/soft-P10 replay. The current best evidence says:

1. XR-63 oracle has real headroom.
2. XR-64 provides a leakage-safe trainable path.
3. The missing work is execution preparation, not another architecture idea.
4. After XR-64, only then consider temporal-lite or confidence-gated local update.
