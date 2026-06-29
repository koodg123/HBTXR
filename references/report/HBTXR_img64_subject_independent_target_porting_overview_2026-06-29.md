# HBTXR Img64 Subject-Independent Target Porting Overview

Date: 2026-06-29

## Goal

Train additional eye-tracking models under the same experimental contract as `HBTXR_subject_independent_img64_patch4`.

Common target contract:

- Dataset root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Split: train subjects 1-32, val subjects 33-36, test subjects 37-48.
- Sample counts: train 968,873, val 122,776, test 366,171, all 1,457,820.
- Input resolution: 64x64.
- HBTXR reference config: `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml`.
- HBTXR reference model settings: `img_size=64`, `patch_size=4`, `input_channels=2`.
- Reference input representation: FACET `DavisEyeEllipseDataset` converts cached structured events into a 2-channel event frame, resizes to 64x64, and generates ellipse heatmap/axis/mask targets at down ratio 4.

## Common Evidence

- `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml:5-18` defines the train dataset root, split, event interpolation, `default_resolution: [64, 64]`, batch size 32, and 8 workers.
- `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml:21-37` defines the val split with the same root and 64x64 resolution.
- `references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml:39-51` defines HBTXR as a 2-channel 64x64 DeiT-style model with `patch_size: 4`.
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py:23-49` defines the cached event/ellipse dataset and split-specific `cached_data` and `cached_ellipse` paths.
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py:60-83` resizes event frames to `default_resolution`.
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py:190-253` loads a fixed-count event segment, converts it to a 2-channel event frame, applies augmentation/resizing, and normalizes the tensor.
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py:255-305` downsamples ellipse labels by 4 and builds heatmap, axes, trig, regression, and mask targets.
- `references/report/FACET/planning/HBTXR_subject_independent_img64_patch4_plan_2026-06-28.md:25-33` documents the leak-free subject split.
- `references/report/FACET/datasets/HBTXR_subject_independent_4state_counts_2026-06-29.md:73-80` documents the split totals.

## Target Mapping Summary

| Requested target | Local codebase | Local status | Same split/resolution readiness |
|---|---|---|---|
| FECET | `references/codebase/software/FACET` | Name `FECET` not found; treated as FACET alias/typo | High, because FACET already owns `DavisEyeEllipseDataset` and EPNet/HBTXR configs |
| TennSt | `FACET/EvEye/model/DavisEyeCenter/TennSt.py`, plus `ais2024/eye_track_spatiotemporal` | Two implementations exist | Medium, but requires sequence/center-label adapter |
| Retina | `references/codebase/software/retina` | Local repo exists | Medium, config already supports 64x64/2-channel, but dataset helper must be added |
| EX-Gaze | `references/codebase/software/EX-Gaze` | Local repo exists | Medium-low, requires MMEngine annotation and event representation export |
| EV-Eye | `references/codebase/software/EV-Eye` | Local repo exists | Low for HBTXR-equivalent training; original model is frame/mask U-Net |
| Swift-Eye | `references/codebase/software/Swift-Eye` | Local repo exists | Low-medium, requires DOTA/MMRotate rotated bbox export and 64x64 config changes |
| E-Track | `references/codebase/software/E-Track` | Local repo exists | Low-medium, requires TFRecord export and TensorFlow model resizing |
| ERVT (AIS2024) | `references/codebase/software/ais2024/ERVT` | Local repo exists | Medium, but requires 3ET-style H5/list export |
| TENNs-Eye (AIS2024) | `references/codebase/software/ais2024/eye_track_spatiotemporal` | Local repo exists | Medium, but requires 3ET-style sequence dataset adaptation |
| BRAT (AIS2025) | `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution` | User-confirmed mapping; local model name is `CNN_GRU_base` | Medium, but requires 3ET-style HBTXR export and 64x64 coordinate/config changes |
| TDTracker (AIS2025) | `references/codebase/software/ais2025/tdtracker` | Local repo exists | Medium, but requires train/val/test H5 export at 64x64 |

## Shared Porting Requirements

All non-FACET targets need at least one of these adapters:

1. Dataset adapter: read `DeanDataset_full_unet_subject_independent/{train,val,test}/cached_data` and `cached_ellipse`.
2. Representation adapter: export each cached event segment to the target model's expected representation.
3. Label adapter: convert FACET ellipse labels to each target's labels.
4. Split adapter: preserve subject-independent membership exactly: train 1-32, val 33-36, test 37-48.
5. Metric adapter: report pixel error in the same coordinate scale used by HBTXR.
6. Resolution adapter: force 64x64 input and update any output scaling, bbox normalization, sensor-size metadata, or `spatial_factor`.

## Recommended Implementation Order

1. FACET/FECET alias: create an EPNet subject-independent img64 config mirroring HBTXR. This is the lowest-risk baseline.
2. Retina: add a new dataset helper because the model config already uses 64x64, 2-channel inputs.
3. AIS2024 ERVT and TENNs-Eye: export HBTXR split to 3ET-style H5/list format.
4. TDTracker: export HBTXR split to `train.h5`, `val.h5`, `test.h5` with `frames` and `label` datasets.
5. EX-Gaze and Swift-Eye: generate image/annotation datasets for MMEngine/MMRotate.
6. E-Track and EV-Eye: treat as special-purpose segmentation/algorithmic pipelines, not direct HBTXR model competitors unless a segmentation target is explicitly chosen.
7. BRAT: treat `Event-based-Eye-Tracking-Challenge-Solution` as the BRAT target; export HBTXR split to its 3ET-style event/label format and adjust 64x64 scaling.

## Blocking Questions

- Does `FECET` mean `FACET`? No `FECET` directory or class was found locally.
- `BRAT (AIS2025)` is user-confirmed as `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution`; its internal model/config names do not contain `BRAT`, so use this mapping consistently in future documents.
- Should models that output only center coordinates be compared against HBTXR ellipse-center pixel error only, or should ellipse IoU also be required? Center-only models cannot produce ellipse IoU without an extra shape head or heuristic ellipse reconstruction.
