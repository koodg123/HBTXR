# HGTXR Pool Integration

## Scope

This integration brings the selectable HGTXR hybrid training components into
`algorithm/hybrid/src` and exposes them through `src.pools`.

## Source Provenance

- HGTXR software source:
  `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HGTXR/HGTXR-etri-server/software/src/hbtxr`
- HGTXR v3 config presets:
  `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HGTXR/HGTXR-etri-server/software/configs/v3`
- Target package:
  `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/algorithm/hybrid/src`
- Target config presets:
  `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR/algorithm/hybrid/configs/external`

All package imports were normalized from `hbtxr.*` to `src.*` to match the
HBTXR hybrid package layout.

## Selectable Pools

- `src.pools.heads`: head class lookup and construction.
- `src.pools.losses`: Stage 1 search and Stage 2 hybrid loss function selection.
- `src.pools.optimizers`: optimizer registry wrappers and optimizer pool helpers.
- `src.pools.lr_schedulers`: LR scheduler selection for `none`, `cosine`, `step`, and `plateau`.
- `src.pools.runtime_schedulers`: runtime track/search scheduler selection.

## Trainer Wiring

`src.training.trainer` now delegates LR scheduler construction to
`src.pools.lr_schedulers.build_lr_scheduler`. Optimizer construction still uses
the existing `src.optim.registry.build_optimizer`, which is re-exported through
`src.pools.optimizers` for direct pool usage.

## Notable HGTXR Features Reflected

- Additional search head variants: `legacy`, `residual_mlp`, `deep_residual_mlp`.
- Candidate-center heads for search and track branches.
- Track auxiliary heads: state auxiliary, SimDR, center heatmap, center refine,
  and center candidate heads.
- Ensemble-teacher support through `src.training.ensemble_teacher`.
- Training convenience options such as trainable parameter filters and
  max-batch debugging limits.
- External experiment presets for center-loss, track-state auxiliary, scheduler,
  distillation, and optimizer comparisons.

## Verification

- `PYTHONPATH=algorithm/hybrid python3 -m compileall -q algorithm/hybrid/src`
  passed in the current environment.
- Runtime import smoke requiring `torch` could not be completed with system
  `python3` because `torch` is not installed in that interpreter.
