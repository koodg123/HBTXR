# HBTXR Algorithm

The algorithm package contains the runnable training and evaluation stack,
organized by explicit ownership boundaries.

## Main Paths

- `common/`: modality-neutral contracts, shared models, and CLI launchers.
- `dataset/`: sample construction, dataset classes, and the dataset factory.
- `utils/`: low-level reusable helpers (cache, event representation, I/O, visualization).
- `engine/`: execution lifecycle — callback, logger, model factory, and orchestration tools.
- `configs/`: cross-modality/common configuration.
- `tests/`: shared ownership and compatibility verification.
- `frame/`: frame-only and APS/RGB/cached-frame experiment surfaces.
- `event/`: event-only models (EPNet, ElNet, TennSt) and event-stack surfaces.
- `hybrid/`: frame-event fusion, search/track, and hardware-aware reference surfaces.
- `analysis/`: result packages, motion labels, reports, and packaging scripts.
- `docs/`: durable planning, specification, and validation records.

## Package Ownership

Canonical imports use the `eveye.*` namespace, one package per owner:

| Package | Source root |
|---|---|
| `eveye.common` | `common/src/eveye/common` |
| `eveye.dataset` | `dataset/src/eveye/dataset` |
| `eveye.utils` | `utils/src/eveye/utils` |
| `eveye.engine` | `engine/src/eveye/engine` |
| `eveye.event` | `event/src/eveye/event` |

Example: `from eveye.engine.model_factory import make_model`.

## Import Compatibility

The legacy `EvEye` package under `algorithm/common/src` is retained only as
explicit alias wrappers for research code that still imports the old paths.
It is scheduled for retirement, after which `EvEye` imports stop working and
package discovery narrows to `eveye*`. New code must use the `eveye.*` owners.
