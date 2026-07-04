# EV-Eye and EX-Gaze Hybrid Frame128/Event64 Experiment Plan

Date: 2026-07-04

## Purpose

This document records the execution plan for training and evaluating two
adapted hybrid baselines under the same experimental environment used for the
HBTXR subject-independent evaluation package:

- `EV-Eye_Hybrid_frame128_event64` on GPU0
- `EX-Gaze_Hybrid_frame128_event64` on GPU1

The goal is not to reuse the original EV-Eye or EX-Gaze reported numbers
directly. The goal is to adapt both methods to the local HBTXR hybrid protocol
so that they use the same subject split, the same input resolution contract, and
the same subject- and motion-wise evaluation package format.

## Reference Package

The target output format should match the level of:

```text
/home/kjm26/project/PRJXR/HBTXR/analysis/RESULTS/HBTXR_subject37_48_error_distribution
```

That package contains:

- subject/motion summary:
  - `HBTXR_subject37_48_error_distribution_by_subject_motion.csv`
  - `HBTXR_subject37_48_joined_motion_counts.csv`
- row-level evaluation files:
  - `HBTXR_subject37_48_test_predictions_with_metadata.csv`
  - `HBTXR_subject37_48_test_joined_motion_error.csv`
  - `HBTXR_subject_independent_img64_patch4_test_sample_predictions.csv`
  - `HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv`
- quality-control files:
  - `HBTXR_subject37_48_dropped_blink_predictions.csv`
  - `HBTXR_subject37_48_unmatched_predictions.csv`
- checkpoint copy:
  - `checkpoints/epoch=66-val_mean_distance=0.5401.ckpt`

Large row-level CSV files should remain local and should not be tracked in git
when they exceed GitHub size limits.

## Common HBTXR-Hybrid Protocol

Both adapted models must use the same common contract.

| Item | Required setting |
|---|---|
| Source dataset | `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent` |
| Frame input | `1 x 128 x 128` |
| Event input | `2 x 64 x 64` |
| Train split | subjects 1-32 |
| Validation split | subjects 33-36 |
| Test split | subjects 37-48 |
| Batch size | 32 |
| Workers | 4 |
| Epochs | 70 |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Weight decay | `1e-5` |
| Scheduler | HBTXR-compatible StepLR/StepLRScheduler policy |
| Checkpoint monitor | `val_mean_distance` |
| Evaluation metrics | pixel error, IoU, subject-wise and motion-wise distributions |
| Motion labels | Fixation, Saccade, Smooth; Blink handled as a separate exclusion/QC flag |

The recommended evaluation coordinate system is the HBTXR event grid
coordinate system, i.e. `64 x 64`, unless a specific table requires conversion
back to the original DAVIS coordinate system. If frame-derived predictions are
produced in `128 x 128`, they must be converted to the common evaluation
coordinate system before metrics are computed.

## Why Derived Datasets Are Required

No new raw data collection is required. However, model-specific derived datasets
or adapter datasets are required because the existing HBTXR dataset is built
around a direct event-to-ellipse contract, while EV-Eye and EX-Gaze expect
different tensors and metadata.

The existing HBTXR-style flow is:

```text
event voxel/image -> model -> center or ellipse
```

The adapted EV-Eye and EX-Gaze flows are:

```text
EV-Eye:
frame -> mask or center
event -> update
frame + event -> final pupil prediction

EX-Gaze:
frame -> initialization or relocalization
event map + pre_state -> 8 event patches -> offset -> current ellipse
```

Therefore, the raw samples can be shared, but the model-facing dataset schemas
must be different.

## Derived Dataset Roots

Create the following derived datasets under `target_data`:

```text
/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent
/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent
```

These should be generated from:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent
```

The derived datasets should preserve the source sample identity so that
predictions can be joined with the existing motion-label package and the
HBTXR subject 37-48 evaluation tables.

## Common Derived Dataset Schema

Both derived datasets should expose the following shared fields:

| Field | Purpose |
|---|---|
| `sample_id` | Stable join key for prediction, label, motion, and metadata tables |
| `split` | `train`, `val`, or `test` |
| `subject` | Subject id |
| `eye` | Left/right eye when available |
| `session` | Original session code |
| `frame_id` or `timestamp` | Temporal ordering and pre-state lookup |
| `frame_128` | `1 x 128 x 128` frame input |
| `event_64` | `2 x 64 x 64` event input |
| `ellipse_64` | Label in event-grid coordinates, `[x, y, a, b, angle]` |
| `ellipse_128` | Optional frame-grid label for frame branch training |
| `motion_label` | Fixation, Saccade, or Smooth |
| `blink_flag` | Blink/session exclusion flag |
| `source_path` | Provenance back to the source sample |

The dataset generator should also write a `manifest.json` containing:

- source dataset path
- generation date
- split definition
- coordinate-system definition
- frame resizing method
- event resizing or event-map generation method
- label scaling rules
- software/script version or git commit when available

## EV-Eye Derived Dataset

Recommended root:

```text
/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent
```

Recommended structure:

```text
EV_Eye_Hybrid_frame128_event64_subject_independent/
  manifest.json
  train_index.csv
  val_index.csv
  test_index.csv
  frames_128/
  masks_128/
  events_64/
  labels.csv
  motion_labels.csv
```

Required tensors and labels:

| Item | Shape / format | Notes |
|---|---|---|
| Frame input | `1 x 128 x 128` | Frame branch input |
| Frame mask | `1 x 128 x 128` | U-Net target; generate from real mask if present, otherwise rasterize ellipse |
| Event input | `2 x 64 x 64` | Event update branch input |
| Ellipse label | `[x, y, a, b, angle]` | Keep both 64-grid and 128-grid versions if possible |
| Metadata | CSV rows | Required for HBTXR-style result join |

### EV-Eye Model Adaptation

The original EV-Eye code is primarily a frame U-Net segmentation pipeline. Under
the HBTXR-Hybrid protocol, it should be treated as an adapted baseline:

```text
frame branch: 1 x 128 x 128 -> pupil mask or base ellipse/center
event branch: 2 x 64 x 64 -> center or ellipse update
fusion: frame prediction + event update -> final pupil prediction
```

Recommended output:

- minimum: `[x, y]`
- preferred: `[x, y, a, b, angle]`

If only a mask is predicted by the frame branch, fit/rasterize an ellipse from
the mask for IoU and center-error evaluation.

### EV-Eye Training Run

Target run name:

```text
EV-Eye_Hybrid_frame128_event64
```

Target device:

```text
CUDA_VISIBLE_DEVICES=0
```

Training should follow the common HBTXR-Hybrid protocol:

- batch size 32
- workers 4
- 70 epochs
- Adam, lr `1e-3`, weight decay `1e-5`
- validation every epoch
- checkpoint by `val_mean_distance`

The model should not be reported as the original EV-Eye number. It should be
reported as:

```text
EV-Eye adapted under HBTXR-Hybrid frame128/event64 protocol
```

## EX-Gaze Derived Dataset

Recommended root:

```text
/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent
```

Recommended structure:

```text
EX_Gaze_Hybrid_frame128_event64_subject_independent/
  manifest.json
  train_index.csv
  val_index.csv
  test_index.csv
  frames_128/
  events_64/
  pre_state.csv
  sample_regions.csv
  labels.csv
  motion_labels.csv
  event_patches_8x2x16x16/   # optional cache
```

Required tensors and labels:

| Item | Shape / format | Notes |
|---|---|---|
| Frame input | `1 x 128 x 128` | Initialization/relocalization branch |
| Event map | `2 x 64 x 64` | Source for event patch extraction |
| Event patches | `8 x 2 x 16 x 16` | EX-Gaze event tracker input |
| `pre_state` | `[x, y, a, b, angle]` | Previous ellipse state |
| Current label | `[x, y, a, b, angle]` | Regression/evaluation target |
| `sample_regions` | eight `[x1, y1, x2, y2]` boxes | Patch provenance and debugging |

### EX-Gaze Patch Contract

Original EX-Gaze creates 8 event patches from a full `2 x 260 x 346` event map.
For this adapted experiment, patch extraction must instead operate on the local
HBTXR event map:

```text
source event map: 2 x 64 x 64
patch count: 8
patch size: 16 x 16
event tracker input: B x 8 x 2 x 16 x 16
```

Ellipse coordinates used for patch sampling must be in the `64 x 64` event-grid
coordinate system. If `pre_state` is derived from a frame-space prediction, it
must be scaled before patch extraction.

### EX-Gaze Model Adaptation

The original EX-Gaze system has two branches:

```text
frame branch: initialization/relocalization
event branch: sparse event-patch transformer
```

For the HBTXR-Hybrid protocol:

- frame branch input should be changed from the original `1 x 160 x 256` crop to
  `1 x 128 x 128`
- event branch should keep the original `8 x 2 x 16 x 16` patch contract
- `event_based_model.pth` can be considered for partial initialization because
  its event tracker input contract matches `8 x 2 x 16 x 16`
- original event weights were trained in a `260 x 346` coordinate context, so
  fine-tuning on the local `64 x 64` coordinate system is required

### EX-Gaze Training Run

Target run name:

```text
EX-Gaze_Hybrid_frame128_event64
```

Target device:

```text
CUDA_VISIBLE_DEVICES=1
```

Training should follow the common HBTXR-Hybrid protocol:

- batch size 32
- workers 4
- 70 epochs
- Adam, lr `1e-3`, weight decay `1e-5`
- validation every epoch
- checkpoint by `val_mean_distance`

The model should be reported as:

```text
EX-Gaze adapted under HBTXR-Hybrid frame128/event64 protocol
```

## Fairness Rules

1. Do not compare original EX-Gaze, EV-Eye, or Swift-Eye paper numbers directly
   against local HBTXR numbers unless the protocol mismatch is explicitly stated.
2. Both adapted models must use the same subject split as HBTXR.
3. Both adapted models must use the same output evaluation coordinate system.
4. For EX-Gaze, using previous ground-truth ellipse during test for every step is
   an oracle setting and should not be used for the main comparison.
5. EX-Gaze test-time tracking should use:
   - first state from frame branch or first label initialization, then
   - previous prediction for subsequent event tracking steps.
6. If an oracle-prestate variant is useful for debugging, report it separately as
   `EX-Gaze-oracle-prestate`, not as the main baseline.
7. Blink handling must match the HBTXR package policy. Blink rows should be
   kept in QC files and excluded/included in final tables according to the same
   rule already used for HBTXR subject 37-48 evaluation.

## Execution DAG

### Phase 0: Preflight

1. Verify source dataset exists.
2. Verify train/val/test subject split counts.
3. Verify frame availability and frame shape.
4. Verify event tensor shape is `2 x 64 x 64`.
5. Verify ellipse labels and coordinate systems.
6. Verify motion labels and blink flags can be joined by sample id or metadata.

### Phase 1: Derived Dataset Generation

1. Generate EV-Eye derived dataset.
2. Generate EX-Gaze derived dataset.
3. Write manifests and split indices.
4. Validate random samples visually and numerically.
5. Check that all labels can be converted between 64-grid and 128-frame grids.

### Phase 2: Model Smoke Tests

For each model:

1. Load one training batch.
2. Run forward pass.
3. Compute loss.
4. Run backward pass.
5. Run one validation batch.
6. Emit prediction rows with the same schema expected by the HBTXR evaluator.

### Phase 3: Parallel Training

Run the two jobs in separate tmux sessions:

```text
GPU0: EV-Eye_Hybrid_frame128_event64
GPU1: EX-Gaze_Hybrid_frame128_event64
```

Recommended log locations:

```text
references/report/FACET/training/EV_Eye_hybrid_frame128_event64_gpu0_train_2026-07-04.log
references/report/FACET/training/EX_Gaze_hybrid_frame128_event64_gpu1_train_2026-07-04.log
```

### Phase 4: Best Checkpoint Selection

Select checkpoints by lowest validation mean distance:

```text
val_mean_distance
```

Copy the selected checkpoint into:

```text
analysis/RESULTS/EV-Eye_Hybrid_frame128_event64/checkpoints/
analysis/RESULTS/EX-Gaze_Hybrid_frame128_event64/checkpoints/
```

### Phase 5: Test Inference

Run inference on subjects 37-48 only.

Required prediction columns:

- sample id
- subject
- split
- session
- frame/timestamp
- predicted center
- predicted ellipse if available
- ground-truth center
- ground-truth ellipse
- pixel error
- IoU
- motion label
- blink flag

### Phase 6: Result Packaging

Create:

```text
analysis/RESULTS/EV-Eye_Hybrid_frame128_event64/
analysis/RESULTS/EX-Gaze_Hybrid_frame128_event64/
```

Each package should include:

```text
checkpoints/
<MODEL>_subject37_48_error_distribution_report.md
<MODEL>_subject37_48_error_distribution_by_subject_motion.csv
<MODEL>_subject37_48_joined_motion_counts.csv
<MODEL>_subject37_48_dropped_blink_predictions.csv
<MODEL>_subject37_48_unmatched_predictions.csv
<MODEL>_subject37_48_test_predictions_with_metadata.csv
<MODEL>_subject37_48_test_joined_motion_error.csv
```

Large CSV files should be excluded from git tracking.

## Acceptance Criteria

The plan is ready to execute only when all items below pass:

- EV-Eye derived dataset exists and has train/val/test split indices.
- EX-Gaze derived dataset exists and has train/val/test split indices.
- Frame tensors are `1 x 128 x 128`.
- Event tensors are `2 x 64 x 64`.
- EX-Gaze event patch tensors are `8 x 2 x 16 x 16`.
- Labels are available in the evaluation coordinate system.
- Motion labels can be joined for subjects 37-48.
- Blink handling matches the HBTXR package.
- One-batch forward/backward smoke test passes for EV-Eye.
- One-batch forward/backward smoke test passes for EX-Gaze.
- Prediction CSV schema matches the HBTXR evaluator input requirements.
- Best checkpoints can be copied into the model result packages.
- Final report tables include subject-wise and motion-wise pixel error and IoU.

## Main Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Frame/event coordinate mismatch | Invalid pixel error and IoU | Store `ellipse_64` and `ellipse_128`; define one evaluation coordinate system |
| EX-Gaze test-time oracle pre-state | Inflated accuracy | Use previous prediction for main evaluation; reserve GT pre-state only for debug |
| EV-Eye original method mismatch | Ambiguous baseline claim | Report as adapted EV-Eye under HBTXR-Hybrid protocol |
| EX-Gaze original weight coordinate mismatch | Poor fine-tuning stability | Partial load event branch, then fine-tune in `64 x 64` grid |
| Patch extraction boundary cases | Missing or clipped patches | Pad event map before patch extraction and record `sample_regions` |
| Large row-level CSV files | Git push failure | Keep large files local and git-ignored |

## Reporting Language

Recommended manuscript/report phrasing:

```text
We compare against adapted EV-Eye and EX-Gaze hybrid baselines under a unified
HBTXR-Hybrid protocol using 128x128 frame inputs and 2x64x64 event inputs.
Original EX-Gaze and EV-Eye reported numbers are not directly comparable because
their input resolutions, fusion strategies, coordinate systems, and evaluation
protocols differ.
```

