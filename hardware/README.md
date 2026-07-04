# HGTXR Hardware

This directory contains the hardware-oriented implementation of HGTXR.

The current code is a portable HLS skeleton that fixes the HW/SW interface and
module boundaries. It is intentionally conservative: numerical kernels are
simple C++ loops first, so C simulation and software reference matching can be
established before aggressive pragma tuning.

## Module Map

- `hls/include/`: fixed-point types, tensor dimensions, shared interfaces.
- `hls/src/hgtxr_top.cpp`: top-level pipeline.
- `hls/src/frame_patch_embed.cpp`: frame patch embedding stub.
- `hls/src/event_patch_embed.cpp`: event patch embedding stub.
- `hls/src/matmul.cpp`: reusable matrix multiply.
- `hls/src/attention.cpp`: attention stage placeholder.
- `hls/src/mlp.cpp`: MLP stage placeholder.
- `hls/src/fusion.cpp`: previous-state and modality fusion.
- `hls/src/search_head.cpp`: search/event pupil head.
- `hls/src/track_head.cpp`: track refinement head.
- `hls/src/runtime_fsm.cpp`: Search/Track runtime state policy.
- `hls/tb/`: C simulation testbenches.
- `vivado/`: board and flow Tcl skeletons.

## Directory Layout

- `docs/architecture/DIRECTORY_LAYOUT.md`: target hardware directory layout,
  cleanup policy, compatibility paths, and deferred migration gates.
- `artifacts/`: preserved bitstreams, HWH files, reports, manifests, and smoke
  evidence that are worth keeping as reproducibility artifacts.
- `generated/`: rebuildable HLS/Vivado/PYNQ/signoff outputs and logs.
- `experiments/`: active, completed, blocked, and template experiment
  definitions.
- `external/`: external papers, codebase summaries, and legacy references.
- `archive/`: deprecated or migration-only material.

## Legacy Evidence

- `docs/legacy/legacy_experiment_analysis_2026_06_12.md`: consolidated
  ViT_Accel, XR_Accel, and HG_PIPE_MERGE historical experiment analysis.
- `docs/legacy/legacy_artifact_index_2026_06_12.json`: source artifact index
  with evidence boundaries for current DSP/URAM/LUTRAM/parallelism decisions.

## ViT Accelerator Reference Analysis

- `analysis/vit-accel/README.md`: per-codebase and per-paper reference analysis
  index.
- `analysis/vit-accel/experiment_extensions_2026_06_15.md`: prioritized
  HGTXR experiment extensions derived from ViT accelerator papers/codebases.
- `analysis/vit-accel/third_goal_integration_2026_06_15.md`: integration policy
  for adding those experiments without replacing the current C3b board-smoke
  gate.

## Third-Goal Tracking

- `docs/Master-Plan.md`: third-goal scope, priority order, and non-regression
  gates.
- `docs/Sub-Plan.md`: task cards for C3b, P2-ViT PoT calibration, and ME-ViT
  buffer audit.
- `docs/Spec.md`: reference-integration scope and acceptance criteria.
- `docs/Execution.md`: active execution decision and next work.
- `docs/Validation.md`: validation gates and remaining evidence gaps.
- `docs/track/PROGRESS.md`: plan-versus-progress checklist.
