# ERVT AIS2024 Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/ais2024/ERVT`

ERVT is an AIS2024 challenge solution using an efficient Recurrent Vision Transformer.

Evidence:

- `references/codebase/software/ais2024/ERVT/README.md:1-4` identifies the challenge solution.
- `references/codebase/software/ais2024/ERVT/README.md:15-25` summarizes the model and reported p10 accuracy.
- `references/codebase/software/ais2024/ERVT/README.md:32-49` documents train/test/benchmark commands.
- `references/codebase/software/ais2024/ERVT/README.md:53-54` points to pretrained weights.

## Original Protocol

Original data and training:

- Data format: 3ET-style event recording folders.
- Dataset class: `ThreeETplus_Eyetracking`.
- Raw events are loaded from H5 key `events`.
- Labels are loaded from `label.txt`.
- Train/val/test membership is controlled by `dataset/train_files.txt`, `val_files.txt`, and `test_files.txt`.
- Event representation: sliced event voxel grid.
- Original sensor size: 640x480.
- Default spatial factor: 0.125, producing 80x60.
- Default input channels/time bins: 3.
- Epochs: 150.
- Batch size: 1.

Evidence:

- `references/codebase/software/ais2024/ERVT/dataset/ThreeET_plus.py:36-39`
- `references/codebase/software/ais2024/ERVT/dataset/ThreeET_plus.py:55-75`
- `references/codebase/software/ais2024/ERVT/dataset/ThreeET_plus.py:81-101`
- `references/codebase/software/ais2024/ERVT/train.py:120-147`
- `references/codebase/software/ais2024/ERVT/train.py:149-161`
- `references/codebase/software/ais2024/ERVT/configs/rvt.json:28-53`

## HBTXR Contract Compatibility

Compatibility is medium.

Why:

- ERVT already consumes event streams and center labels.
- However, it expects 3ET-style H5 recordings, not FACET cached fixed-count samples.
- Default output resolution is 80x60, not 64x64.
- Default temporal sequence protocol does not match HBTXR sample indexing.

Required changes:

1. Export HBTXR split into 3ET-style recording folders with `.h5` and `label.txt`.
2. Generate train/val/test file lists from subjects 1-32, 33-36, and 37-48.
3. Set `sensor_width/sensor_height` and `spatial_factor` so the effective input is 64x64, or bypass the 640x480 assumption in dataset transforms.
4. Convert ellipse center labels to `(x, y, close)` format expected by the challenge code.
5. Keep temporal sequences session-contiguous.
6. Adjust metric scaling so reported pixel error is in 64x64 HBTXR coordinates.

## Expected Output

Possible after adapter:

- Center pixel error.
- p3/p5/p10/p15-style accuracy.

Not direct:

- Ellipse IoU.

## Readiness

Status: promising but not immediate.

Blocking work: 3ET-style export with 64x64 coordinate handling.

