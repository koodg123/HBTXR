# BRAT AIS2025 Porting Review

Date: 2026-06-29

## Local Mapping

User-confirmed target mapping:

```text
BRAT (AIS2025) = references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution
```

Important naming note:

- The repository README calls it `Event-based Eye Tracking Challenge--1st Solution`.
- The local config/model names do not contain `BRAT`.
- The train config uses `model: CNN_GRU_base`.
- For this project, treat this repository as the BRAT target.

Evidence:

- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/README.md:1-2` identifies it as the CVPR 2025 challenge first solution.
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/README.md:13-16` documents pretrained model placement, `sliced_baseline.json`, and `test.sh`.
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/README.md:32-35` documents training through `./train.sh`.
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/configs/sliced_baseline.json:1-21` defines the main dataset/model/training config.
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/model/CNN_GRU_base.py:11-73` defines the local model.

## Original Protocol

Original config:

- Data root: `data_dir` points to a 3ET+ challenge `event_data` tree.
- Metadata/cache roots are separate disk cache directories.
- Model: `CNN_GRU_base`.
- Pixel tolerances: `[5, 10, 15]`.
- Sensor size: 640x480.
- Train sequence length: 30.
- Val/test sequence length: 30.
- Train stride: 15.
- Val/test stride: 30.
- Time bins: 4.
- Loss: weighted MSE.

Evidence:

- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/configs/sliced_baseline.json:2-21`

The training script:

- Dynamically imports `model.{args.model}`.
- Uses `ThreeETplus_Eyetracking`.
- Applies `ScaleLabel`, `TemporalSubsample`, and `NormalizeLabel`.
- Converts event slices to maps using `EventSlicesToMap`.
- Caches sliced datasets with `DiskCachedDataset`.
- Uses a train/val split from the 3ET-style dataset file lists.

Evidence:

- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/train.py:50-77`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/train.py:79-97`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/train.py:99-124`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/train.py:126-137`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/train.py:139-178`

The dataset class:

- Expects 3ET-style H5 recordings with `events`.
- Uses file lists under `dataset/*.txt`.
- Non-test data loads from `data_dir/train/<record>/<record>.h5` and `label.txt`.
- Test data loads from `data_dir/test/<record>/<record>.h5` and `label_zeros.txt`.
- Original class-level sensor size is `(640, 480, 2)`.

Evidence:

- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/dataset/ThreeET_plus.py:38-46`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/dataset/ThreeET_plus.py:66-92`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/dataset/ThreeET_plus.py:95-145`

The model:

- Input shape is `(batch, seq_len, channels, height, width)`.
- Uses 2-channel Conv2D stem, CNN blocks, adaptive 4x4 pooling, GRU, relative-attention Transformer, and a 2D coordinate head.
- Outputs center coordinates only, not ellipse axes or masks.

Evidence:

- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/model/CNN_GRU_base.py:15-29`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/model/CNN_GRU_base.py:38-49`
- `references/codebase/software/ais2025/Event-based-Eye-Tracking-Challenge-Solution/model/CNN_GRU_base.py:51-73`

## HBTXR Contract Compatibility

Compatibility is medium.

Why:

- BRAT/CNN_GRU_base is event-sequence based and predicts pupil centers, so it can be trained on HBTXR event data after export.
- It does not consume FACET `cached_data` / `cached_ellipse` directly.
- It is center-only, so ellipse IoU is not available without an extra ellipse reconstruction step.
- Original coordinate handling assumes 640x480 downsampled to challenge-scale coordinates, not 64x64 HBTXR coordinates.

Required changes:

1. Export `DeanDataset_full_unet_subject_independent` to the 3ET-style tree:
   - `event_data/train/<session_id>/<session_id>.h5`
   - `event_data/train/<session_id>/label.txt`
   - `event_data/test/<session_id>/<session_id>.h5`
   - `event_data/test/<session_id>/label_zeros.txt` or real labels for local evaluation.
2. Generate file lists:
   - train list: subjects 1-32.
   - val list: subjects 33-36.
   - test list: subjects 37-48.
3. Preserve session boundaries when slicing sequences.
4. Convert FACET ellipse labels to center labels `(x, y, close)` at 64x64 scale.
5. Set or patch config/transform scaling:
   - `sensor_width: 64`
   - `sensor_height: 64`
   - `spatial_factor: 1.0`
   - keep `n_time_bins: 4` unless ablation is desired.
6. Patch hardcoded `NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)` if needed so it uses `args.sensor_width * factor` and `args.sensor_height * factor`.
7. Patch test output scaling currently based on `(640*factor, 480*factor)` and `0.125 / factor` to report 64x64 pixel coordinates.
8. Add local validation/test reporting against HBTXR metrics:
   - mean/median/P95/P99 center pixel error.
   - per-subject error.
   - per-motion-type error if metadata is joined.

## Expected Output

Directly possible after adapter:

- Center pixel error.
- p5/p10/p15-style threshold accuracy.
- Sequence-level prediction CSV.

Not direct:

- Ellipse IoU.
- Ellipse axis/angle error.
- Mask IoU.

These require either:

- adding an ellipse head, or
- reconstructing a fixed/heuristic ellipse around the predicted center, which should be reported as an approximation rather than a native model output.

## Training Readiness

Status: not ready for immediate HBTXR training, but no longer blocked.

Main blocker: HBTXR-to-3ET-style export and 64x64 coordinate scaling patches.

Recommended first implementation step:

Create a small export smoke dataset from one train subject and one val subject, then run a 1-batch forward/training smoke test before exporting the full 1,457,820-sample dataset.

