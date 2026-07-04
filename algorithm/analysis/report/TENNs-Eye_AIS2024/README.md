# TENNs-Eye AIS2024 Porting Review

Date: 2026-06-29

## Local Mapping

Source path: `references/codebase/software/ais2024/eye_track_spatiotemporal`

TENNs-Eye is a lightweight causal spatio-temporal network for online event-camera eye tracking.

Evidence:

- `references/codebase/software/ais2024/eye_track_spatiotemporal/README.md:1-8` identifies TENNs-Eye and its AIS2024 result.
- `references/codebase/software/ais2024/eye_track_spatiotemporal/README.md:18-32` documents dataset setup, submission generation, and train entrypoint.
- `references/codebase/software/ais2024/eye_track_spatiotemporal/config.yaml:1-9` defines dataset windowing and downsampling.
- `references/codebase/software/ais2024/eye_track_spatiotemporal/config.yaml:11-23` defines model and trainer settings.

## Original Protocol

Original config:

- `time_window: 10000`
- `frames_per_segment: 50`
- `spatial_downsample: [5, 5]`
- `events_interpolation: causal_linear`
- `epochs: 200`
- `batch_size: 32`

Original dataset:

- Uses 3ET-style train/test folders.
- Val files are hardcoded as `["1_6", "2_4", "4_4", "6_2", "7_4", "9_1", "10_3", "11_2", "12_3"]`.
- For `mode=train`, the dataset excludes those val files from the train folder.
- Converts event streams to `(T, C, H, W)` event frames.
- Uses center labels and close flags.
- Original coordinate system is based on 640x480 and downsampling.

Evidence:

- `references/codebase/software/ais2024/eye_track_spatiotemporal/eye_dataset.py:14-20`
- `references/codebase/software/ais2024/eye_track_spatiotemporal/eye_dataset.py:37-94`
- `references/codebase/software/ais2024/eye_track_spatiotemporal/eye_dataset.py:156-205`
- `references/codebase/software/ais2024/eye_track_spatiotemporal/eye_dataset.py:207-257`
- `references/codebase/software/ais2024/eye_track_spatiotemporal/tenn_model.py:153-217`

## HBTXR Contract Compatibility

Compatibility is medium.

Why:

- TENNs-Eye accepts event-frame sequences and center labels, which fits HBTXR event data conceptually.
- It does not accept FACET `cached_ellipse` directly.
- It expects 50-frame segments, not independent fixed-count single samples.
- Its split logic is file-name based, not subject-based.

Required changes:

1. Export HBTXR sessions to TENNs-Eye-compatible folders.
2. Replace `val_files` with subject-independent file lists.
3. Convert ellipse centers to `(x, y, close)`.
4. Choose 64x64 coordinate policy: set spatial downsample/export directly to 64x64, or edit hardcoded 640x480 assumptions.
5. Preserve temporal continuity when making 50-frame segments.
6. Add a test split path for subjects 37-48.

## Expected Output

Possible after adapter:

- Center pixel error.
- p-threshold accuracy.

Not direct:

- Ellipse IoU.

## Readiness

Status: good sequence-model candidate after dataset export.

Risk: medium because temporal segmentation must not mix subjects or sessions.

