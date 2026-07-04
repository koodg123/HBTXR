# EV-Eye, EX-Gaze, and Swift-Eye HBTXR Porting Analysis

Date: 2026-07-04

## Goal

This note consolidates the current analysis for training and evaluating
EV-Eye, EX-Gaze, and Swift-Eye against the same experimental baseline used by
`HBTXR_subject_independent_img64_patch4`.

The target comparison contract is:

- Dataset root:
  `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Train subjects: 1-32
- Validation subjects: 33-36
- Test subjects: 37-48
- Input resolution: 64x64
- Batch size: 32
- Data workers: 4
- Epochs: 70
- Optimizer target: Adam
- Learning rate: 1e-3
- Weight decay: 1e-5
- Main metric: center pixel error on the subject-independent test set
- Secondary metric, when available: ellipse or rotated-box IoU

## Dataset Summary

The subject-independent DeanDataset is not a raw frame image folder. It stores:

1. frame-aligned pseudo ellipse labels;
2. cached event segments around each frame timestamp;
3. split metadata.

Observed split counts:

| Split | Subjects | Valid samples |
|---|---:|---:|
| train | 1-32 | 968,873 |
| val | 33-36 | 122,776 |
| test | 37-48 | 366,171 |

`cached_ellipse/ellipse_records.npy` contains structured records:

```text
[t, x, y, a, b, ang]
```

where `t` is the frame timestamp, `x,y` are the pupil center, `a,b` are the
ellipse axes, and `ang` is the ellipse angle in the original DAVIS coordinate
system.

`cached_data/events_batch_*.memmap` contains structured events:

```text
[t, x, y, p]
```

`events_indices_*.npy` maps each sample to the `[start, end]` range inside the
corresponding event memmap. The FACET loader converts each event segment into a
two-channel event frame before resizing to the configured input resolution.

No original PNG/JPG frame images are stored inside
`DeanDataset_full_unet_subject_independent`.

## Common Interpretation

For all three targets, the common source of truth should be the
subject-independent DeanDataset, not the original EV-Eye split. The original
`Data_davis`, `Data_davis_labelled_with_mask`, and `processed_data` folders are
still useful for checking the original protocols and pretrained weights, but
the fair HBTXR comparison should be generated from the leak-free
subject-independent split.

The exported data should be model-specific:

| Target | Export needed |
|---|---|
| EV-Eye | HDF5 event-image or frame-image tensors plus binary masks |
| EX-Gaze | MMEngine JSON annotation plus event representation HDF5 |
| Swift-Eye | MMRotate/DOTA-style images plus rotated bbox annotations |

## Modality and Input/Output Forms

| Model | Branch | Modality | Input form | Output form |
|---|---|---|---|---|
| EV-Eye | Original U-Net | frame | `B x 1 x 260 x 346` | `B x 2 x 260 x 346` mask logits |
| EV-Eye | HBTXR event U-Net | event-only | `B x 2 x 64 x 64` recommended | `B x 2 x 64 x 64` mask logits |
| EX-Gaze | Frame detector | frame | original image `260 x 346`, detector crop/resize `B x 1 x 160 x 256` | pupil ellipse / rotated bbox |
| EX-Gaze | Event tracker | event + previous state | event representation `B x 2 x 260 x 346`, internal patches `B x 8 x 2 x 16 x 16` | current ellipse or displacement-refined ellipse |
| Swift-Eye | Detector | event-derived image | event accumulation image, original config `346 x 346`, HBTXR target `64 x 64` | rotated bbox `[x, y, w, h, angle]` |

For the HBTXR comparison, the intended dimensions are:

| Target | HBTXR-compatible input |
|---|---|
| EV-Eye event U-Net | `B x 2 x 64 x 64` |
| EX-Gaze event tracker | event representation `B x 2 x 64 x 64`, patchified internally |
| Swift-Eye | event-derived image `64 x 64`, encoded as image files for MMRotate |

## EV-Eye Plan

### Original Finding

The EV-Eye Python U-Net is a frame segmentation model. In the original code,
HDF5 data are reshaped as:

```text
images: [N, 1, 260, 346]
masks : [N, 260, 346]
```

The pretrained weights under:

```text
/home/kjm26/project/dataset/XR/EV_Eye/processed_data/Pre-trained_models
```

are useful for the original frame U-Net baseline.

### User-Requested Direction

The current preferred direction is:

```text
Use trained EV-Eye U-Net parameters
-> change the input to 64x64 event-only data
-> fine-tune on the HBTXR subject-independent split
```

This should be treated as a new adapted baseline:

```text
EV-Eye_UNet_pretrained_frame_to_event_img64
```

### Frame-Side Handling

If EV-Eye is trained with 64x64 event input, the frame side should not be used
as model input. It should only provide the frame timestamp and frame-aligned
ellipse label:

```text
frame timestamp -> anchor for the event segment
ellipse [x, y, a, b, ang] -> 64x64 binary pupil mask target
```

Raw frame PNGs are not needed for this event-only fine-tuning path.

### Event Input Construction

Recommended input:

```text
channel 0: negative polarity event image
channel 1: positive polarity event image
shape: B x 2 x 64 x 64
```

Target:

```text
ellipse_records.npy -> scale from 346x260 to 64x64 -> rasterize binary mask
shape: B x 64 x 64
```

### Pretrained Weight Loading

The original EV-Eye U-Net first convolution expects one input channel. For
two-channel event input, adapt the first convolution only:

```text
old first conv: [out_channels, 1, k, k]
new first conv: [out_channels, 2, k, k]
```

Recommended initialization:

```text
new_weight[:, 0] = old_weight[:, 0] * 0.5
new_weight[:, 1] = old_weight[:, 0] * 0.5
```

All other compatible layers can load from the pretrained checkpoint.

### Required Code Changes

- Add an HBTXR event U-Net dataset/exporter.
- Generate `train.h5`, `val.h5`, and `test.h5` or implement direct memmap
  reading.
- Change `UNet(n_channels=1, ...)` to `UNet(n_channels=2, ...)`.
- Add partial pretrained loading with first-conv adaptation.
- Replace leave-one-subject-out training with fixed train/val/test splits.
- Set batch size 32, workers 4, epochs 70.
- Set Adam `lr=1e-3`, `weight_decay=1e-5`.
- Add mask-to-center and mask-to-ellipse evaluation.

### Estimated Cost

| Stage | Estimate |
|---|---:|
| Exporter and dataset loader | 0.5-1.5 days |
| Pretrained loading and train loop patch | 0.5 day |
| Evaluation packaging | 0.5-1 day |
| Training time | 10-30 hours/GPU |

## EX-Gaze Plan

### Original Finding

EX-Gaze contains both frame and event branches.

The frame detector uses grayscale frame input and pupil annotation:

```text
original frame metadata: 260 x 346
detector crop/resize path: 160 x 256
output: pupil ellipse / rotated bbox
```

The event tracker uses event representations plus a previous pupil state:

```text
event HDF5 sample: 2 x 260 x 346
pre_state: [x, y, w, h, theta]
target: current [x, y, w, h, theta]
```

The local sample HDF5 files contain polarity event-count representations with
shape:

```text
[2, 260, 346]
```

The event model then patchifies the event representation around the previous
ellipse. One inspected config uses:

```text
patch_num=8
patch_size=16
in_channels=2
internal patch tensor: B x 8 x 2 x 16 x 16
```

### Recommended HBTXR Variant

Use the event displacement tracker, not the frame-only detector, for HBTXR
event-model comparison.

Suggested name:

```text
EX-Gaze_event_tracker_hbtxr_img64
```

### Required Export

From `DeanDataset_full_unet_subject_independent`:

- export event representations as HDF5;
- export JSON annotation containing `pre_gt`, `cur_gt`, `events_between`, and
  `img_shape`;
- preserve train/val/test subjects exactly;
- scale labels to 64x64;
- handle session boundaries where no valid previous state exists.

### Required Code/Config Changes

- Replace current split values with train 1-32, val 33-36, test 37-48.
- Change frame size and evaluator mask size assumptions from `260x346` to
  `64x64`.
- Generate `pre_gt`/`cur_gt` pairs from consecutive samples inside the same
  session.
- Ensure event HDF5 keys match the generated `events_between` IDs.
- Set optimizer and training schedule to the HBTXR comparison contract.
- Add a subject/motion-aware evaluation exporter.

### Estimated Cost

| Stage | Estimate |
|---|---:|
| JSON/HDF5 exporter | 1-2 days |
| Config and loader normalization | 0.5-1 day |
| Evaluation packaging | 0.5-1 day |
| Training time | 50-110 hours/GPU |

## Swift-Eye Plan

### Original Finding

Swift-Eye is built on MMRotate. It behaves like an image detector at the code
level, but the intended input is an event-derived image rather than a DAVIS
grayscale frame.

Original config traits:

- dataset style: `images/` plus `annotations/`
- input resize: `346 x 346`
- model family: RoITransformer / Swin / FPN
- output: rotated bbox
- original optimizer: AdamW, not the HBTXR Adam contract

### Recommended HBTXR Variant

Use event-derived 64x64 images and rotated bbox labels generated from the
subject-independent DeanDataset.

Suggested name:

```text
Swift-Eye_hbtxr_event_img64
```

### Required Export

From `DeanDataset_full_unet_subject_independent`:

- convert each event segment into a 64x64 event image;
- save image files in the MMRotate/DOTA-compatible folder structure;
- convert each ellipse `[x, y, a, b, ang]` into a rotated bbox;
- preserve train/val/test subject membership;
- include metadata for subject/motion-wise post-evaluation.

### Required Code/Config Changes

- Replace data roots with generated HBTXR train/val/test roots.
- Change all resize/image scale assumptions to 64x64.
- Set batch size 32 and workers 4.
- Set epochs 70.
- Replace optimizer with Adam `lr=1e-3`, `weight_decay=1e-5`.
- Add HBTXR-compatible center error and IoU evaluation, because the default
  MMRotate mAP is not enough for the current comparison.

### Estimated Cost

| Stage | Estimate |
|---|---:|
| MMRotate/DOTA exporter | 1.5-3 days |
| Config normalization | 0.5-1 day |
| Evaluation packaging | 1 day |
| Training time | 70-180 hours/GPU |

## Recommended Implementation Order

1. EV-Eye event U-Net with pretrained frame U-Net initialization.
2. EX-Gaze event tracker with generated `pre_gt`/`cur_gt` JSON and event HDF5.
3. Swift-Eye event-derived image detector with MMRotate/DOTA export.

This order provides the fastest event-only baseline first while preserving the
more expensive detector/tracker baselines for later.

## Reporting Names

Use explicit names to avoid mixing original protocols and HBTXR-adapted
protocols:

| Name | Meaning |
|---|---|
| `EV-Eye_original_pretrained_frame` | Original pretrained frame U-Net baseline |
| `EV-Eye_UNet_pretrained_frame_to_event_img64` | Event-only 64x64 U-Net fine-tuned from frame pretrained weights |
| `EX-Gaze_event_tracker_hbtxr_img64` | Event tracker using HBTXR event representation and previous ellipse state |
| `Swift-Eye_hbtxr_event_img64` | Swift-Eye detector trained on 64x64 event-derived images |

## Risks and Decisions

- EV-Eye event fine-tuning is no longer the original EV-Eye frame-only U-Net;
  it should be reported as an adapted event U-Net baseline.
- EX-Gaze requires careful session-boundary handling because the event tracker
  needs a previous ellipse state.
- Swift-Eye is closest to ellipse/shape comparison among the three, but its
  MMRotate stack is expected to train much slower than EV-Eye U-Net.
- All three should use `DeanDataset_full_unet_subject_independent` for fair
  HBTXR comparison.
- Original `Data_davis` and `Data_davis_labelled_with_mask` should remain
  evidence/reference sources, not the final split source for the HBTXR
  comparison.

## Validation Criteria for Future Implementation

Before launching training, verify:

- exported train/val/test sample counts match 968,873 / 122,776 / 366,171
  unless a documented boundary policy drops first samples for `pre_gt`;
- all exported coordinates are in the 64x64 coordinate system;
- every test prediction can be joined to subject and motion metadata;
- EV-Eye event U-Net loads pretrained weights except the adapted first
  convolution;
- EX-Gaze HDF5 event tensors and JSON `events_between` keys match;
- Swift-Eye annotation conversion preserves center and angle conventions;
- evaluation outputs include subject-wise and motion-wise center pixel error.
