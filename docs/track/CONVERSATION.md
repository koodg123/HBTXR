# Conversation Summary

Date: 2026-07-03

## User Decisions

- Use `/home/user/project/PRJXR/HBTXR` as the main working directory.
- Analyze and work on the `etri-desktop` branch after earlier branch confusion.
- Train/evaluate Retina and ERVT on the HBTXR subject-independent dataset.
- Use dataset path `/mnt/d/dataset/EV_Eye/target_data/DeanDataset_full_unet_subject_independent`.
- Generate HBTXR-like pixel-error distribution files for Retina and ERVT only.
- Generate JETCAS-style Excel aggregation files like the HBTXR reference workbook.
- Fix Saccade blank rows by using the HBTXR joined label map.
- Document the work and commit the changes.

## Work Performed

- Reviewed training readiness for several models.
- Prioritized ERVT and Retina training/evaluation work.
- Adjusted Retina IoU loss aggregation from `sum()` to `mean()` earlier in the workflow.
- Used worker/cudNN tuning for training and evaluation where applicable.
- Generated Retina and ERVT result packages from checkpoints.
- Rebuilt motion-specific results using HBTXR joined labels after detecting raw metadata label mismatch.

## Key Outcome

Retina and ERVT now have comparable subject37-48 pixel-error distribution outputs using the same joined motion label basis as HBTXR.

## 2026-07-04 Continuation

### User Decisions

- Continue work on branch `etri-desktop`.
- Compare model complexity, input/output contracts, head semantics, and final prediction semantics for all models represented under `references/report`.
- Add `EIDet / ElNet` to the same measured-target table format as the HBTXR comparison targets.
- Analyze FACET `ElNet` code directly instead of leaving the row as unknown.
- Measure `E-Track`, `EV-Eye`, `EX-Gaze`, and `Swift-Eye` using each original codebase input contract when they are not part of the HBTXR 64x64 CSV.
- Create additional isolated virtual environments when the main `.venv` cannot support legacy TensorFlow or OpenMMLab stacks.
- Document the full conversation/work state and commit the tracked changes.

### Work Performed

- Reviewed `references/report` model coverage and local source paths for HBTXR, EPNet/FECET, FACET TennSt, Retina, TDTracker, ERVT, TENNs-Eye, BRAT, EIDet/ElNet, E-Track, EV-Eye, EX-Gaze, and Swift-Eye.
- Created `references/report/HBTXR_model_io_complexity_prediction_summary_2026-07-04.md` with Params, MACs, FLOPs, measured input, raw output dimensions, head meaning, and final prediction semantics.
- Profiled FACET `EIDet / ElNet` with the FACET model-factory input condition `1x3x256x256`; because local `DCNv2` is missing, used a DCNv2-compatible shim and documented the caveat.
- Profiled `EV-Eye` with its original PyTorch U-Net input `1x1x260x346`.
- Created additional venvs under `tmp/venvs/`:
  - `tmp/venvs/etrack_py38` for TensorFlow 2.6.0 and external `jakeret/unet`.
  - `tmp/venvs/mmrotate_py38` for Swift-Eye's PyTorch 1.13.0, MMCV-full 1.7.2, MMDetection 2.28.2, and local MMRotate 0.x stack.
  - `tmp/venvs/exgaze_py310` for EX-Gaze's PyTorch 1.13.1, MMCV 2.0.1, MMEngine 0.8.5, MMDetection 3.1.0, and MMRotate 1.0.0rc1 stack.
- Profiled `E-Track`, `EX-Gaze`, and `Swift-Eye` after installing their isolated dependencies.
- Added `tmp/venvs/` to `.gitignore` so the local profiling environments do not enter git history.
- Removed generated tracked `__pycache__` noise after profiling imports.

### Current Complexity Measurements Added

- `E-Track`: Params `466,562`, MACs `9,864,740,864`, FLOPs `19,729,481,728`, input `B x 352 x 256 x 3`.
- `EX-Gaze`: Params `623,920`, MACs `99,114,840`, FLOPs `198,229,680`, input `B x 1 x 160 x 256`.
- `Swift-Eye`: Params `58,650,327`, MACs `26,359,593,567`, FLOPs `52,719,187,135`, input `B x 3 x 352 x 352`.

### Important Caveats

- `EIDet / ElNet` complexity is a reproducible proxy because the repository does not include compiled `DCNv2`.
- `E-Track` MACs were computed from the installed TensorFlow U-Net source because Keras custom block internals are not exposed as normal top-level layers.
- `EX-Gaze` used PyPI `mmrotate==1.0.0rc1` because the README references a separate develop checkout that is not present in this repository.
- `Swift-Eye` used a valid-proposal monkey patch for the official MMRotate `forward_dummy` path because the upstream dummy proposals are random and can violate ROIAlign assertions.
