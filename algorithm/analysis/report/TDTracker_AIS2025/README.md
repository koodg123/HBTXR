# TDTracker AIS2025 Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/ais2025/tdtracker`

TDTracker is a CVPR 2025 Event-based Eye Tracking Challenge third-place solution.

Evidence:

- `references/codebase/software/ais2025/tdtracker/README.md:1-4` identifies TDTracker.
- `references/codebase/software/ais2025/tdtracker/README.md:15-35` documents preprocessing, training, testing, and post-processing.
- `references/codebase/software/ais2025/tdtracker/provider_data.py:22-28` loads H5 `frames` and `label`.
- `references/codebase/software/ais2025/tdtracker/train.py:21-41` defines training arguments and default H5 paths.
- `references/codebase/software/ais2025/tdtracker/train.py:149-167` loads train/test H5 files and builds dataloaders.

## Original Protocol

Default training:

- Train H5 path: `./data/train_aug.h5`.
- Test H5 path: `./data/test_aug.h5`.
- Batch size: 16.
- Sensor width/height: 640x480.
- Spatial factor: 0.125.
- Effective coordinate scale: 80x60.
- Epochs: 1000.
- Model: `models.TDTracker.Model`.
- Validation uses the H5 file named `test_h5_path`.

Evidence:

- `references/codebase/software/ais2025/tdtracker/train.py:24-40`
- `references/codebase/software/ais2025/tdtracker/train.py:151-167`
- `references/codebase/software/ais2025/tdtracker/train.py:170-180`
- `references/codebase/software/ais2025/tdtracker/train.py:184-210`

Preprocessing:

- Reads 3ET+ H5 events and `label.txt`.
- Converts events to `(80,60,2)` frames with 4 bins/BinaRep.
- Groups frames into sequences of length 100.
- Writes H5 datasets `frames` and `label`.

Evidence:

- `references/codebase/software/ais2025/tdtracker/dataprocess/3etplus.py:42-72`
- `references/codebase/software/ais2025/tdtracker/dataprocess/3etplus.py:114-172`
- `references/codebase/software/ais2025/tdtracker/dataprocess/3etplus.py:262-282`

## HBTXR Contract Compatibility

Compatibility is medium.

Why:

- TDTracker already consumes H5 tensors and center labels.
- It expects `frames`/`label`, so exporting HBTXR to this format is feasible.
- It currently assumes 80x60 derived from 640x480 with `spatial_factor=0.125`.
- The requested HBTXR resolution is 64x64, so scaling assumptions must be changed.

Required changes:

1. Export HBTXR train/val/test into separate H5 files:
   - `train_hbtxr_img64.h5`
   - `val_hbtxr_img64.h5`
   - `test_hbtxr_img64.h5`
2. Store datasets:
   - `frames`: sequence tensor compatible with TDTracker.
   - `label`: center labels in normalized or 64x64 coordinate convention.
3. Modify arguments:
   - `sensor_width=64`
   - `sensor_height=64`
   - `spatial_factor=1.0`
4. Split validation and test explicitly. The original code uses `test_h5_path` as validation during training.
5. Avoid data augmentation crossing subject/session boundaries.
6. Update metric scale so pixel error is reported on 64x64.

## Expected Output

Possible after adapter:

- Center pixel error.
- p-threshold accuracy.
- Post-processed center predictions.

Not direct:

- Ellipse IoU.

## Readiness

Status: good candidate after H5 export.

Risk: medium. The main risk is sequence construction and coordinate scaling, not model architecture.

