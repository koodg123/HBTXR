# IMPL_REPOS And HANDOVER Algorithm Import - 2026-07-08

## Scope

This pass imported the algorithm-side materials identified during the
`IMPL_REPOS` and `HANDOVER` review. The import is intentionally selective: it
keeps experiment summaries, metadata, configs, analysis scripts, and small
reference code, while excluding raw checkpoints, generated caches, venvs, and
large row-level prediction CSVs.

## Imported Sources

### HANDOVER/HBTXR

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/HANDOVER/HBTXR`

Imported to:

- `algorithm/analysis/scripts/`
- `algorithm/analysis/configs/`
- `algorithm/analysis/RESULTS/`

New analysis scripts:

- `build_eveye_exgaze_training_manifests.py`
- `dryrun_eveye_exgaze_train_ready.py`
- `export_eveye_exgaze_hybrid_dataset.py`
- `infer_eveye_exgaze_hybrid_test.py`
- `infer_eveye_pretrained_labelled_options.py`
- `preflight_eveye_exgaze_hybrid.py`
- `smoke_eveye_exgaze_hybrid_loaders.py`
- `train_eveye_exgaze_hybrid.py`
- `train_eveye_labelled_mask_128.py`
- `validate_eveye_exgaze_hybrid_export.py`

New config inputs:

- `EV_Eye_Hybrid_frame128_event64_subject_independent_train_ready.json`
- `EX_Gaze_Hybrid_frame128_event64_subject_independent_train_ready.json`

New or extended result packages include:

- `EV-Eye/`
- `EV-Eye_Hybrid_frame128_event64_epoch5_test_inference/`
- `EV-Eye_Hybrid_frame128_event64_train_ready/`
- `EV-Eye_labelled_mask_frame128_event64_original_protocol/`
- `EX-Gaze/`
- `EX-Gaze_Hybrid_frame128_event64_best_test_inference/`
- `EX-Gaze_Hybrid_frame128_event64_prestate_train_ready/`
- `EX-Gaze_Hybrid_frame128_event64_train_ready/`
- additional small TENNs-Eye manifest/stat metadata from the handover result package

### IMPL_REPOS/SW/FECET-HBTXR

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/IMPL_REPOS/SW/FECET-HBTXR`

Imported to:

- `algorithm/archive/imports/fecet_hbtxr/`

Reason:

- Preserve FACET-style event accumulation, ellipse supervision, Grounded-SAM
  dataset preparation wrappers, FACET comparison docs, and focused tests as
  optional future porting material.

### IMPL_REPOS/SW/SWIFT-HBTXR

Source root:

- `/mnt/d/dataset/EV_Eye/paper_works/IMPL_REPOS/SW/SWIFT-HBTXR`

Imported to:

- `algorithm/archive/imports/swift_hbtxr/`

Reason:

- Preserve the anti-blink, TimeLens interpolation, runtime trace, and
  HBTXR-style ABI integration work as optional future porting material.

## Excluded

- Checkpoints and model weights: `*.pt`, `*.pth`, `*.ckpt`.
- Python/cache/runtime folders: `.venv`, `.pytest_cache`, `__pycache__`.
- Local data and generated run trees: `data/`, `runs/`.
- Large row-level prediction CSVs above 20 MB copied during the first result
  pass were removed from this import. Smaller reports, metadata JSON, motion
  stats, subject stats, history CSVs, and package manifests were retained.

## Follow-Up

- Adapt EV-Eye/EX-Gaze scripts to the current `algorithm/hybrid/src` package
  before treating them as executable active scripts.
- Decide whether large row-level prediction CSVs should live in external
  artifact storage or Git LFS instead of the repository.
- Promote FECET/SWIFT modules only after API adaptation and tests.
