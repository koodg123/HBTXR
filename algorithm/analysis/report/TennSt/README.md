# TennSt Porting Review

Date: 2026-06-29

## Local Mapping

Two TennSt implementations exist locally:

1. FACET implementation: `references/codebase/software/FACET/EvEye/model/DavisEyeCenter/TennSt.py`
2. AIS2024 TENNs-Eye implementation: `references/codebase/software/ais2024/eye_track_spatiotemporal/tenn_model.py`

This document focuses on the requested `TennSt` target as a model family and keeps `TENNs-Eye (AIS2024)` in a separate directory.

Evidence:

- `references/codebase/software/FACET/configs/MemmapDavisEyeCenter_TennSt.yaml:46-55` defines FACET `model.type: TennSt`.
- `references/codebase/software/FACET/EvEye/model/DavisEyeCenter/TennSt.py:30-88` defines its basic activation/norm/pointwise blocks.
- `references/codebase/software/FACET/EvEye/model/DavisEyeCenter/TennSt.py:90-168` defines spatial and temporal blocks with streaming support.
- `references/codebase/software/ais2024/eye_track_spatiotemporal/tenn_model.py:153-217` defines the AIS2024 `TennSt` model.

## Original FACET TennSt Protocol

FACET TennSt is not wired to `DavisEyeEllipseDataset`. It uses `MemmapDavisEyeCenterDataset`.

Original config details:

- Dataset type: `MemmapDavisEyeCenterDataset`.
- Root path: hardcoded `/mnt/data2T/junyuan/Datasets/datasets/MemmapDavisEyeCenterDataset`.
- Splits: `train`, `val`.
- Sensor size: `[346, 260, 2]`.
- Frames per segment: `50`.
- Time window: `40000`.
- Fixed-count events: `5000`.
- Optional spatial downsample: `true`.
- Model outputs detector-style center predictions, not full ellipse masks.

Evidence:

- `references/codebase/software/FACET/configs/MemmapDavisEyeCenter_TennSt.yaml:5-23`
- `references/codebase/software/FACET/configs/MemmapDavisEyeCenter_TennSt.yaml:25-44`
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeCenter/MemmapDavisEyeCenterDataset.py:103-135`
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeCenter/MemmapDavisEyeCenterDataset.py:157-220`

## HBTXR Contract Compatibility

Compatibility is medium.

Why it is not direct:

- HBTXR reference data uses `cached_ellipse`, while FACET TennSt expects `cached_label` center labels.
- HBTXR samples are single fixed-count event frames; TennSt expects 50-frame temporal segments.
- TennSt validation metric is p-accuracy/pixel error around centers, not ellipse IoU.

Required changes:

1. Create `DeanDataset_full_unet_subject_independent_center_seq` or a live dataset adapter.
2. Convert every valid ellipse to center label `(x, y, close)` in the 64x64 coordinate system.
3. Group samples into 50-frame contiguous temporal segments without crossing session boundaries.
4. Set `spatial_downsaple: false` if the exported data is already 64x64.
5. Update the model's expected input shape to `(B, 2, T, 64, 64)`.
6. Preserve the subject split: train 1-32, val 33-36, test 37-48.
7. Add a test dataloader path if test-set evaluation is required, because the FACET dataset asserts only `train` and `val`.

## Expected Output

Directly possible:

- Center pixel error.
- p3/p5/p10/p15 accuracy if metric code is reused.

Not direct:

- Ellipse IoU, unless an ellipse head is added or a fixed ellipse reconstruction heuristic is accepted.

## Readiness

Status: not ready for immediate training under HBTXR contract.

Blocking work: center-sequence dataset export or a new `DavisEyeEllipseTennStDataset`.

