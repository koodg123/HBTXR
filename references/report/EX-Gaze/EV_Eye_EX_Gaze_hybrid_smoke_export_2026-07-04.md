# EV-Eye and EX-Gaze Hybrid Smoke Export

Date: 2026-07-04

## Scope

This document records the first sequential execution step after the EV-Eye and
EX-Gaze hybrid plan/preflight.

Targets:

- EV-Eye adapted dataset root:
  `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent/smoke_32`
- EX-Gaze adapted dataset root:
  `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent/smoke_32`

Source:

- `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Raw frame recovery from `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis`

## Exporter

Script:

- `analysis/scripts/export_eveye_exgaze_hybrid_dataset.py`

Command:

```bash
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
  analysis/scripts/export_eveye_exgaze_hybrid_dataset.py \
  --tag smoke_32 \
  --max-samples-per-split 32
```

## Export Contract

| Item | Shape / Format | Notes |
|---|---|---|
| Frame | `1 x 128 x 128` PNG | Grayscale frame recovered from `Data_davis` and resized to 128x128. |
| Mask | `1 x 128 x 128` PNG | Rasterized from scaled ellipse pseudo-label. |
| Event | `2 x 64 x 64` uint16 NPY | Polarity-separated event count volume. |
| EX-Gaze patches | `8 x 2 x 16 x 16` uint16 NPY | Eight ellipse-boundary patches sampled from the 64x64 event volume. |
| EX-Gaze annotation | JSON | Contains frame path, event path, patch path, sample regions, `pre_state`, and pupil ellipse. |

## Smoke Counts

| Split | EV-Eye Samples | EX-Gaze Samples |
|---|---:|---:|
| train | 32 | 32 |
| val | 32 | 32 |
| test | 32 | 32 |

File counts:

- EV-Eye smoke files: 292
- EX-Gaze smoke files: 295

The EX-Gaze count is larger by three files because it also stores one
`annotations.json` file per split.

## Shape Validation

The first sample in each split was loaded after export.

| Split | Frame | Mask | Event | EX-Gaze Patch | Annotation Rows |
|---|---|---|---|---|---:|
| train | `(128, 128)` uint8 | `(128, 128)` uint8 | `(2, 64, 64)` uint16 | `(8, 2, 16, 16)` uint16 | 32 |
| val | `(128, 128)` uint8 | `(128, 128)` uint8 | `(2, 64, 64)` uint16 | `(8, 2, 16, 16)` uint16 | 32 |
| test | `(128, 128)` uint8 | `(128, 128)` uint8 | `(2, 64, 64)` uint16 | `(8, 2, 16, 16)` uint16 | 32 |

## Current Progress Checklist

- [x] Existing DeanDataset generation and resplit logic inspected.
- [x] EV-Eye/EX-Gaze hybrid exporter implemented.
- [x] Smoke export generated with 32 samples per split.
- [x] Smoke shape and row-count validation passed.
- [ ] Full export has not been started.
- [ ] EV-Eye model adapter/config has not been implemented.
- [ ] EX-Gaze model config/loader adapter has not been implemented.
- [ ] Training has not been launched.

## Next Step

Run a model-loader smoke check against the exported `smoke_32` package:

1. EV-Eye: load frame/mask/event rows and verify U-Net path plus event branch input.
2. EX-Gaze: load `annotations.json`, event volume, and patch tensor, then verify the
   data preprocessor can consume `8 x 2 x 16 x 16` patches or define a local adapter.
