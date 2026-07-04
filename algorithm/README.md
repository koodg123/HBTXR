# HBTXR Algorithm

The algorithm package contains the runnable training and evaluation stack.

## Main Paths

- `frame/`: frame-only and APS/RGB/cached-frame experiment surfaces.
- `event/`: event-only and event-stack experiment surfaces.
- `hybrid/`: frame-event fusion, search/track, and hardware-aware reference surfaces.
- `common/`: shared legacy package, training scripts, dataloaders, metrics, and utilities.
- `analysis/`: result packages, motion labels, reports, and packaging scripts.
- `docs/`: durable planning, specification, and validation records.

## Import Compatibility

The source package remains `EvEye` under `algorithm/common/src` to preserve existing
imports such as `from EvEye.model.model_factory import make_model`.
