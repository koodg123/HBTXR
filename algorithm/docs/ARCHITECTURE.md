> **작성** 2026-07-23 · **갱신** 2026-07-23
> **상태** active
> **소유** algorithm

# HBTXR Architecture

Design overview of the flat-functional reimplementation. Paper reference: HBTXR
(IEEE JETCAS) — a hybrid event-camera pupil/gaze model.

## 1. Models (`models/`)

Two layers: **shared role dirs** (reused by every modality) and **modality
assemblies**.

### Shared

- `backbones/` — `ViTBackbone` (`ViTConfig`) with a full-depth `forward_search`
  and an early-exit `forward_track` (cut point `c < L`); `patch_embed` provides the
  Conv-F (1→D frame) and Conv-E (2→D event) stems. `model.embed_dim` is the single
  source of truth for the width (the backbone config is synced to it).
- `blocks/` — the transformer block (LN → MHA → LN → MLP).
- `heads/` — `PupilBoxHead` (oriented box, 5), `PupilEllipseHead` (anchor-relative
  residual, 5), `PupilMaskHead`, `ReliabilityHead` (2), `EyeRegionHead` (4), plus a
  `build_head` factory so the primary head is config-swappable.
- `geometry/` — the canonical state `s = (x, y, a, b, θ)` with `a ≥ b`, `θ ∈ [0, π)`
  (`canonicalize` = Π); `box_to_state` = `g()`; `state_to_box` = `g⁻¹()` (build a box
  target from a GT state); `apply_residual` = `Π(z + d_s)`.

### Modality assemblies

- `frame/`, `event/` — `DirectPupilDetector`: stem → full-depth ViT → box head →
  `g()` → ellipse state, plus aux mask/ROI/reliability heads. Both are standalone,
  independently trainable detectors (no anchor); they differ only in the stem.
- `hybrid/` — one shared backbone: **search** (frame, full depth, box head → anchor
  state) + **track** (event, early-exit, ellipse residual → `apply_residual`). A
  CPU-side `TrackSearchScheduler` FSM selects the branch per stream step at runtime
  (inference only). Training supervises each branch on its own step.
- `mask/` — a standalone UNet segmenter (separate from the HBTXR detector).

## 2. Data → model contract (`engine/data/`)

The HBTXR dataset (`dataset/hbtxr/`) emits a rich sample dict (frame, event,
`mask_target`, 6-D `state6` cur/prev, targets, meta). `adapt_batch` maps a collated
batch to the model I/O contract per modality:

| modality | input | targets |
|---|---|---|
| frame / event | `image` = frame / event | `box` = `state_to_box(state)`, `mask`, `reliability`, `eye_box` |
| hybrid | `frame` + `event` | `box` (search), `residual` = `cur − prev`, `anchor_state` = prev (GT teacher forcing), `reliability` |
| mask | `image` = frame | `mask` |

`build_dataloader` wraps the existing `dataset.hbtxr.loader` pipeline (mode
resolution, weighted sampler, collate) with this adapter — no loader duplication.

## 3. Training / eval / inference (`engine/`)

- `train/` — `compute_losses` scores only the terms present in both outputs and
  targets (box · ellipse · mask · reliability · eye_box), weighted. `Trainer` is a
  small modality-aware loop (direct forward; hybrid = search + track summed). No
  Lightning.
- `eval/` — `evaluate` mirrors the trainer forward under `no_grad` and averages
  per-term losses + contract metrics (`center_distance`, `mask_iou`).
- `infer/` — `predict_batch` (batched forward) and `stream_infer` (the hybrid FSM
  runtime via `run_step`).
- `distill/` — contract-native teacher→student agreement on the shared heads +
  the task loss (`total = task + w · distill`).
- `compress/` — native `torch.nn.utils.prune` (global L1 / Ln-structured) +
  sparsity report.
- `model_factory.make_model` builds any modality from a config `model:` block.
- `tools/` — `load_config` (YAML/JSON + `include:` merge, torch-free) and
  `checkpoint` (save/load).

## 4. Configs (`configs/`)

`modality/<m>/*.yaml` is the model fragment; `base/{data,training}.yaml` are shared;
`experiment/*.yaml` compose them via `include:` (deep-merge, local wins) and add the
`experiment` block. The entrypoints consume the merged `experiment / data / training
/ model` schema.

## 5. Notes

- Run from the `algorithm/` root; the packages are top-level (not pip-installed).
- Parked pre-rewrite reference code lives in `../archive/models-tmp` (outside
  `algorithm/`), imported by nothing live.
