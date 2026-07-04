# FACET Phase 4 Full Training Launch

Date: 2026-06-26

## Summary

`DeanDataset_full_unet` generation completed and full EPNet/FACET plus HBTXR-DeiT training was launched in parallel.

## Dataset Gate

Full expanded dataset:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
```

Manifest and progress state:

```text
completed_session_count: 384
num_sessions: 384
num_train: 1165260
num_val: 292560
total_frames_scanned: 1490548
valid_ellipse_count: 1457820
skipped_frame_count: 32728
skip_no_ellipse_count: 32681
skip_no_events_count: 47
```

The train/val dataset smoke check passed for first, boundary, and last sample indices after fixing event cache indexing.

## Loader Fix

Initial full training attempts failed with:

```text
IndexError: index <N> is out of bounds for axis 0 with size <M>
```

Root cause:

- `build_full_dean_dataset_with_unet.py` flushes event batches at session boundaries for resumability.
- Therefore `events_indices_*.npy` files are variable-length, not fixed at 5000 samples.
- `load_event_segment()` assumed `batch_id = index // 5000`, which is only valid for fixed-size event batches.

Fix:

```text
references/codebase/software/FACET/EvEye/utils/cache/MemmapCacheStructedEvents.py
```

`load_event_segment()` now builds cached metadata from actual `events_indices_*.npy` lengths and resolves the batch by cumulative sample counts.

Validation:

```text
train len: 1165260
val len: 292560
boundary indices around batch transitions load successfully
last train and last val sample load successfully
```

## Augmentation Guard

The next launch attempt exposed another training-time dataset issue:

```text
ValueError: No contours found
```

Root cause:

- Training augmentation can shift/rotate an otherwise valid ellipse mask fully out of the visible frame.
- `transform_ellipse()` raises `ValueError` when no transformed contour remains.

Fix:

```text
references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py
```

When ellipse transform fails after augmentation, the sample is now treated as a closed/invalid-eye sample:

```text
ellipse = ((0, 0), (0, 0), 0)
close = 1
```

Validation:

```text
aggressive train augmentation smoke over sampled indices passed
EPNet and HBTXR were restarted after this fix
```

## Parallel Training

EPNet/FACET baseline:

```text
tmux: facet_epnet_full_gpu0
GPU: 0
PID: 125368
config: DavisEyeEllipse_EPNet_full_unet.yaml
log: references/report/FACET/EPNet_full_unet_gpu0_train_2026-06-26.log
run/log root: references/codebase/software/FACET/runs/logs/EPNet_full_unet
```

HBTXR-DeiT:

```text
tmux: facet_hbtxr_full_gpu1
GPU: 1
PID: 125360
config: DavisEyeEllipse_HBTXR_full_unet.yaml
log: references/report/FACET/HBTXR_full_unet_gpu1_train_2026-06-26.log
run/log root: references/codebase/software/FACET/runs/logs/HBTXR_full_unet
```

Host GPU evidence:

```text
00000000:02:00.0, 125368, /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
00000000:03:00.0, 125360, /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 5920 MiB
```

Both current jobs passed dataset gate, train/val smoke check, Lightning sanity validation, and entered epoch 0 training after the loader and augmentation fixes.

## Remaining Gates

- Wait for EPNet full checkpoint.
- Wait for HBTXR full checkpoint.
- Run full validation/evaluation on both checkpoints.
- Produce FACET paper Table II comparison and EPNet-vs-HBTXR comparison reports.
