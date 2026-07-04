# Progress

Date: 2026-07-03

## Completed

- Set working repository context to `/home/user/project/PRJXR/HBTXR`.
- Switched analysis context to `etri-desktop`.
- Packaged HBTXR reference outputs under `analysis/results/HBTXR`.
- Packaged Retina training config/checkpoint artifacts.
- Packaged ERVT training config/checkpoint artifacts.
- Generated Retina subject37-48 predictions and pixel-error distribution files.
- Generated ERVT subject37-48 predictions and pixel-error distribution files.
- Found and fixed the Saccade blank issue by using the HBTXR joined motion label map.
- Regenerated Retina/ERVT JETCAS-style Excel workbooks.
- Added validation and execution documentation.

## Current Artifacts

- `analysis/results/Retina`
- `analysis/results/ERVT`
- `analysis/results/HBTXR`
- `analysis/scripts`
- `docs`

## Remaining Notes

- ERVT has fewer joined rows than Retina because sequence segmentation does not cover every sample index.
- Large per-sample CSVs, the Retina log, and the large HBTXR checkpoint are excluded from git tracking for GitHub push compatibility.

## 2026-07-04 Completed

- Added the model IO, complexity, head, and final-prediction summary report at `references/report/HBTXR_model_io_complexity_prediction_summary_2026-07-04.md`.
- Added `EIDet / ElNet` to the same column format as the measured HBTXR comparison targets.
- Measured FACET `ElNet` with the FACET input condition and documented the missing-DCNv2 profiling shim.
- Measured `EV-Eye` with its original PyTorch U-Net input.
- Installed isolated profiling venvs for `E-Track`, `EX-Gaze`, and `Swift-Eye`.
- Measured `E-Track`, `EX-Gaze`, and `Swift-Eye` with each original codebase input contract.
- Added `tmp/venvs/` to `.gitignore` so generated profiling environments remain local.
- Cleaned profiling-generated tracked `__pycache__` changes from the worktree.

## Current 2026-07-04 Artifacts

- `references/report/HBTXR_model_io_complexity_prediction_summary_2026-07-04.md`
- `tmp/venvs/etrack_py38` local-only profiling environment
- `tmp/venvs/mmrotate_py38` local-only profiling environment
- `tmp/venvs/exgaze_py310` local-only profiling environment

## 2026-07-04 Remaining Notes

- The new venvs are intentionally not tracked by git.
- The model-complexity report includes caveats for DCNv2, TensorFlow custom blocks, EX-Gaze's missing develop checkout, and Swift-Eye dummy ROI proposals.

## 2026-07-04 Swift-Eye Preparation

- Prepared direct HBTXR cache training path for Swift-Eye without PNG/DOTA export.
- Added `analysis/scripts/swifteye_hbtxr_direct_train.py`.
- Changed Swift-Eye Swin backbone first projection to `2` input channels instead of using a `2ch -> 3ch` adapter.
- Made `64x64` the default direct HBTXR input resolution.
- Reduced detector RPN anchors and temporal tracking anchors for 64px pupil boxes.
- Reduced temporal fusion crops from original `33/13` to `15/7` so the search crop fits the `16x16` P2 feature map.
- Parameterized the custom Swift-Eye tracking anchor generator so its original hard-coded crop geometry can be overridden.
- Validated detector and temporal forward/backward smoke checks with the CPU MMRotate venv.
- Wrote preparation metadata under `analysis/results/Swift-Eye/direct_hbtxr_img64`.

## 2026-07-04 Swift-Eye Remaining Notes

- Full training is blocked by environment, not data/code wiring: `tmp/venvs/mmrotate_py38` has CPU-only PyTorch.
- GPU launch needs a GPU-enabled MMRotate 0.3.4/MMCV 1.x stack or an equivalent port into the main CUDA venv.
