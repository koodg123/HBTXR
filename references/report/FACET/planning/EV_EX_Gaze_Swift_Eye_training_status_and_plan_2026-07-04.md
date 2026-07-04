# EV-Eye, EX-Gaze, Swift-Eye Training Status and Plan

Date: 2026-07-04 16:26 KST

## Summary

This document consolidates the current conversation decisions, implementation status, training launches, environment changes, and next plan for EV-Eye, EX-Gaze, and Swift-Eye under the HBTXR/FACET subject-independent comparison workflow.

The comparison target remains aligned to `HBTXR_subject_independent_img64_patch4` where feasible:

- Train subjects: `1-32`
- Validation subjects: `33-36`
- Test subjects: `37-48`
- Core split source: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent`
- Train samples: `968873`
- Validation samples: `122776`
- Test samples: `366171`
- Target optimizer contract: Adam, `lr=1e-3`, `weight_decay=1e-5`
- Target schedule contract: StepLR, step size `10`, gamma `0.7`
- Target batch/workers/epochs: batch `32`, workers `4`, epochs `70`

## Conversation Decisions

### PyTorch/CUDA/cuDNN

- The system driver and system cuDNN must not be modified.
- The previous `.facet-train-venv` used `torch 2.12.1+cu130`.
- That environment could see CUDA in unrestricted execution, but cuDNN Conv failed with:

```text
CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

- The selected fix was to leave system CUDA/cuDNN untouched and create a project-local environment:

```text
.facet-cu128-venv
```

- Installed PyTorch stack in the project-local venv:

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
torchaudio==2.11.0+cu128
nvidia-cudnn-cu12==9.19.0.56
opencv-python-headless
natsort
tqdm
```

- cuDNN-enabled Conv2D and Conv3D smoke tests passed in `.facet-cu128-venv`.
- Training launch scripts unset `LD_LIBRARY_PATH` and use only the project-local venv runtime.

### EV-Eye

- Original EV-Eye is primarily a frame/mask U-Net benchmark.
- For HBTXR comparison, an adapted hybrid/event-aware baseline was implemented rather than using the original leave-one-subject-out EV-Eye protocol.
- Current adapted input:

```text
frame: 1 x 128 x 128
event: 2 x 64 x 64
target mask: 128 x 128
```

- Output:

```text
segmentation logits: 2 x 128 x 128
```

- Metric currently logged during training:

```text
pixel mask accuracy
```

- Later evaluation still needs mask-to-center and mask-to-ellipse conversion for HBTXR-style pixel error and IoU.

### EX-Gaze

- EX-Gaze is a hybrid event-frame method.
- For HBTXR comparison, the adapted model uses:

```text
frame: 1 x 128 x 128
event volume: 2 x 64 x 64
event patches: 8 x 2 x 16 x 16
previous state: [x, y, a, b, angle]
target: [x, y, a, b, angle]
```

- Output:

```text
normalized ellipse state [x, y, a, b, angle]
```

- Training metric currently logged:

```text
center pixel error in 128 x 128 frame coordinates
```

### Swift-Eye

- Swift-Eye is event-only in modality, but its implementation consumes event-derived image representations rather than raw event streams.
- Original Swift-Eye is built on MMRotate and predicts rotated pupil boxes:

```text
[x, y, w, h, angle]
```

- Original image/data assumptions:

```text
DAVIS source sensor: 346 x 260 (W x H)
Swift-Eye detector resize: 346 x 346
Temporal fusion feature mapping: padded image around 352 x 288 with stride 4 feature maps
```

- HBTXR-compatible Swift-Eye should use:

```text
event-derived image: 64 x 64
annotation: DOTA-style polygon + rotated bbox from FACET ellipse
```

- Swift-Eye is not yet train-ready. It needs dataset export, MMRotate config normalization, hardcoded GPU/path cleanup, and temporal-pair generation.

## Completed Work

### Dataset Manifests

Compact train-ready manifest roots exist outside the git repository:

| Target | Dataset root |
|---|---|
| EV-Eye | `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EV_Eye_Hybrid_frame128_event64_subject_independent/train_ready` |
| EX-Gaze | `/home/kjm26/project/dataset/XR/EV_Eye/target_data/EX_Gaze_Hybrid_frame128_event64_subject_independent/train_ready` |

Validated split counts:

| Split | Subjects | Samples |
|---|---:|---:|
| train | 1-32 | 968873 |
| val | 33-36 | 122776 |
| test | 37-48 | 366171 |

### Training Entrypoint

Implemented:

```text
analysis/scripts/train_eveye_exgaze_hybrid.py
```

Supported modes:

```text
--model ev-eye
--model ex-gaze
```

The script records:

- config path
- model name
- device
- `CUDA_VISIBLE_DEVICES`
- cuDNN enabled flag
- cuDNN version
- dataset roots
- train/val lengths
- torch version

### Runtime Environment

Created project-local environment:

```text
.facet-cu128-venv
```

Validation:

- `torch 2.11.0+cu128`
- CUDA wheel `12.8`
- cuDNN `91900`
- `torch.backends.cudnn.enabled == True`
- Conv2D and Conv3D smoke tests passed
- EV-Eye dry-run passed
- EX-Gaze dry-run passed
- batch `32`, workers `4` dry-runs passed

### Launch Scripts

EV-Eye GPU0:

```text
references/report/EV-Eye/run_ev_eye_hybrid_train_ready_gpu0_2026-07-04.sh
```

EX-Gaze GPU1:

```text
references/report/EX-Gaze/run_ex_gaze_hybrid_train_ready_gpu1_2026-07-04.sh
```

Both scripts:

- unset `LD_LIBRARY_PATH`
- use `.facet-cu128-venv`
- keep system driver/cuDNN untouched
- write logs under model-specific report directories

## Current Training Status

Branch:

```text
etri-server
```

Latest launch commit:

```text
535a8d8 feat(training): launch EV-Eye EX-Gaze cu128 runs
```

Active tmux sessions:

| Session | Purpose |
|---|---|
| `ev_eye_hybrid_gpu0` | EV-Eye adapted training on physical GPU0 |
| `ex_gaze_hybrid_gpu1` | EX-Gaze adapted training on physical GPU1 |

Current GPU state at this note:

| GPU | Model | Memory | Utilization | Notes |
|---:|---|---:|---:|---|
| 0 | EV-Eye | about `3177 MiB` | about `99%` | First epoch not yet written to history |
| 1 | EX-Gaze | about `559 MiB` | about `10%` | Epoch 1 completed |

EX-Gaze epoch 1 result:

| Epoch | Train loss | Train metric | Val loss | Val metric | LR | Seconds |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0003505224 | 3.1063084 | 0.0000540485 | 0.9475575 | 0.001 | 494.6107 |

Interpretation:

- EX-Gaze epoch 1 took about `8.24 minutes`.
- EX-Gaze `val_metric` is center pixel error in 128x128 frame coordinates.
- EV-Eye is still processing epoch 1. It is expected to be slower because it trains dense `128x128` segmentation.

## Training Outputs

Ignored local output directories:

| Target | Output directory |
|---|---|
| EV-Eye | `analysis/RESULTS/EV-Eye_Hybrid_frame128_event64_train_ready` |
| EX-Gaze | `analysis/RESULTS/EX-Gaze_Hybrid_frame128_event64_train_ready` |

These directories are intentionally ignored to avoid pushing large checkpoints/results to GitHub.

Tracked report/log entrypoints:

| Target | Log path |
|---|---|
| EV-Eye | `references/report/EV-Eye/logs/EV_Eye_Hybrid_frame128_event64_train_ready_gpu0_2026-07-04.log` |
| EX-Gaze | `references/report/EX-Gaze/logs/EX_Gaze_Hybrid_frame128_event64_train_ready_gpu1_2026-07-04.log` |

Log files are ignored by the top-level `.gitignore`.

## Swift-Eye Plan

### Goal

Prepare Swift-Eye as an event-only baseline under the same HBTXR subject-independent split.

### Required Dataset Export

Create:

```text
/home/kjm26/project/dataset/XR/EV_Eye/target_data/Swift_Eye_HBTXR_subject_independent_img64
```

Recommended layout:

```text
detection/
  train/
    images/
    annotations/
  val/
    images/
    annotations/
  test/
    images/
    annotations/
temporal/
  tracking_train_dataset.pickle
  tracking_validation_dataset.pickle
  tracking_test_dataset.pickle
manifest.json
```

Image export:

```text
FACET cached events -> 2 x 64 x 64 event image -> Swift-Eye image representation
```

Annotation export:

```text
FACET ellipse [x, y, a, b, angle]
-> rotated bbox [x, y, w, h, angle]
-> polygon [x1,y1,...,x4,y4]
-> DOTA-style annotation line
```

### Required Code Changes

1. Remove hardcoded `CUDA_VISIBLE_DEVICES` values from Swift-Eye scripts.
2. Remove hardcoded train/val paths in `set_args`.
3. Add a config for `img_scale=(64,64)`.
4. Recompute or set normalization for exported event images.
5. Replace original batch/epoch/lr/worker values with HBTXR comparison values where stable.
6. Generate temporal template/search pairs within the same subject/session only.
7. Add inference adapter that writes HBTXR-compatible test prediction CSVs.

### Training Sequence

1. Export detection images and DOTA annotations.
2. Smoke test MMRotate dataset loading.
3. Train detector/backbone-neck.
4. Export temporal pair pickle files.
5. Train temporal fusion component.
6. Run subject 37-48 sequential inference.
7. Create `analysis/RESULTS/Swift-Eye_HBTXR_subject_independent_img64` result package.

### Risk

Swift-Eye depends on older MMRotate/MMCV/MMDUT stack. The current `.facet-cu128-venv` may not support it directly. If installation fails, the fallback is to implement a lightweight PyTorch adapter that reuses Swift-Eye concepts but avoids the old MMRotate runtime.

## Progress Checklist

- [x] Preserve system driver/cuDNN.
- [x] Create project-local PyTorch CUDA 12.8 environment.
- [x] Validate cuDNN enabled Conv2D/Conv3D.
- [x] Build EV-Eye/EX-Gaze train-ready compact manifests.
- [x] Implement EV-Eye/EX-Gaze adapted training entrypoint.
- [x] Launch EV-Eye on GPU0.
- [x] Launch EX-Gaze on GPU1.
- [x] Record EX-Gaze epoch 1 result.
- [x] Analyze Swift-Eye modality and original image size.
- [x] Draft Swift-Eye HBTXR porting plan.
- [ ] Wait for EV-Eye epoch 1 result.
- [ ] Continue EX-Gaze training monitoring.
- [ ] Implement Swift-Eye HBTXR exporter.
- [ ] Validate Swift-Eye MMRotate/cu128 environment.
- [ ] Launch Swift-Eye training after exporter and environment are ready.
- [ ] Package final subject-independent test results for all models.

## Verification Evidence

Commands and observations used:

- `nvidia-smi` showed both RTX 5080 GPUs active after launch.
- `tmux ls` showed `ev_eye_hybrid_gpu0` and `ex_gaze_hybrid_gpu1`.
- `analysis/RESULTS/EX-Gaze_Hybrid_frame128_event64_train_ready/history.csv` contained epoch 1.
- EV-Eye `history.csv` was still empty at this note.
- `run_metadata.json` for both active runs recorded `cudnn_enabled: true` and `torch: 2.11.0+cu128`.

## Next Concrete Actions

1. Monitor EV-Eye first epoch completion.
2. Continue EX-Gaze training and record epoch trend.
3. Implement Swift-Eye event-image and DOTA annotation exporter.
4. Create Swift-Eye 64x64 config and environment smoke test.
