# E-Track Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/E-Track`

E-Track combines an Event-to-Frame converter, pupil event U-Net, and event-based RoI mechanism.

Evidence:

- `references/codebase/software/E-Track/README.md:1-5` identifies E-Track.
- `references/codebase/software/E-Track/README.md:27-31` describes its original data subset.
- `references/codebase/software/E-Track/README.md:33-43` documents U-Net and full E-Track execution.
- `references/codebase/software/E-Track/dataset/e_track_dataset.py:7-14` defines original image sizes and padded target sizes.
- `references/codebase/software/E-Track/e_track_unet.py:28-31` defines U-Net input size and learning rate.

## Original Protocol

Original training:

- Framework: TensorFlow/Keras.
- Input: TFRecord files.
- Original image size: 346x260.
- Target padded/cropped size: 352x256.
- Channels: 3.
- Classes: 2.
- Subjects: users 4-27.
- Train/valid/test TFRecord folders: `data/tfrecord_0`, `data/tfrecord_1`, `data/tfrecord_2`.
- Epochs: 40.
- Batch size: 8.

Evidence:

- `references/codebase/software/E-Track/dataset/e_track_dataset.py:17-40`
- `references/codebase/software/E-Track/e_track_unet.py:50-65`
- `references/codebase/software/E-Track/e_track_unet.py:79-101`

Full algorithm:

- Operates on raw event buffers and fits/updates ellipse-style RoI logic.
- Uses sensor size 346x260.

Evidence:

- `references/codebase/software/E-Track/e_track.py:40-52`
- `references/codebase/software/E-Track/e_track.py:62-72`
- `references/codebase/software/E-Track/e_track.py:148-249`

## HBTXR Contract Compatibility

Compatibility is low-medium.

Why:

- Original data is TFRecord, not FACET memmap cache.
- Original split excludes subjects 1-3 and uses subjects 4-27.
- Model input size is 352x256, not 64x64.
- The pipeline is partly algorithmic and not a clean train/val/test PyTorch model.

Required changes:

1. Export HBTXR split to TFRecord with 64x64 event-frame images and binary labels.
2. Change U-Net `img_x_size` and `img_y_size` to 64.
3. Rebuild the TFRecord loader reshape/padding assumptions.
4. Replace users 4-27 with train 1-32, val 33-36, test 37-48.
5. Decide whether to train only the U-Net or run the full E-Track algorithm.
6. Add post-processing to derive center pixel error from masks.

## Expected Output

Direct output:

- Pupil event segmentation mask.

Derived output:

- Center pixel error after fitting/centroid extraction.
- Ellipse-like result only if the RoI/ellipse fitting stage is run.

## Readiness

Status: not ready for immediate training.

Blocking work: TFRecord export and TensorFlow model/data resizing.

