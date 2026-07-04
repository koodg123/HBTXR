# Algorithm Modality Reorganization

## Goal

Split the algorithm workspace into clear modality surfaces:

- `frame`: APS/RGB/cached-frame only paths.
- `event`: event-stack, event-frame, and event-only paths.
- `hybrid`: frame-event fusion and search/track runtime paths.
- `common`: shared legacy compatibility, training, data, metrics, and utilities.

## Current Design

```text
algorithm/
  frame/
  event/
  hybrid/
  common/
  analysis/
  artifacts/
  docs/
  requirements/
  README.md
  pyproject.toml
```

The legacy `EvEye` package is intentionally kept as one compatibility package under `common/src/EvEye`. It currently contains frame, event, and hybrid code in one import tree. Splitting it physically before stabilizing adapters would break existing imports and training scripts.

## Source Boundaries

- `frame`: configs moved from the former `algorithm/configs` tree for cached APS, RGB/image, and frame-only experiments.
- `event`: configs moved from the former `algorithm/configs` tree for event-only and event-stack experiments.
- `hybrid`: legacy hybrid configs, the external hybrid package staged under `hybrid/src`, and HGTXR software reference under `hybrid/hardware_reference`.
- `common`: former `algorithm/src/EvEye`, former FACET scripts, former FACET tests, and leftover shared configs.

## Packaging

`pyproject.toml` now points package discovery at `common/src` so the legacy `EvEye` import remains available while modality folders are staged.
