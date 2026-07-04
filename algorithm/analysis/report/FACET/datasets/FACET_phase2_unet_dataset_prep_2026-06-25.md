# FACET Phase 2 U-Net Dataset Prep Log

Date: 2026-06-25

## Scope

이 문서는 `FACET_reproduction_plan_2026-06-25.md`의 Phase 2, 즉 `Data_davis_labelled_with_mask`로 U-Net segmentation model을 재학습하기 위한 데이터 변환과 smoke 검증 기록이다.

## Implemented Changes

### 1. H5 to DavisWithMaskDataset 변환 스크립트 추가

추가 파일:

- `EvEye/utils/scripts/build_unet_dataset_from_h5.py`

역할:

- `Data_davis_labelled_with_mask/*/*.h5`를 읽는다.
- 각 h5의 `data`와 `label` dataset을 `(260, 346, N)` 형태로 transpose한다.
- `data` frame을 grayscale PNG로 저장한다.
- `label > 0` mask를 binary PNG `0/255`로 저장한다.
- FACET `DavisWithMaskDataset`이 요구하는 구조를 만든다.

출력 구조:

```text
<output_root>/
  train/
    data/*.png
    label/*.png
  val/
    data/*.png
    label/*.png
  manifest.json
```

기본 출력 경로:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset
```

지원 옵션:

- `--dry-run`
- `--max-files`
- `--overwrite`
- `--train-ratio`
- `--split-unit file|sample`
- `--min-mask-pixels`

기본 split은 file 단위 80/20이다.

### 2. U-Net model robustness 수정

수정 파일:

- `EvEye/model/DavisEyeEllipse/UNet/UNet.py`

수정 내용:

- `configure_optimizers()` 내부의 불필요한 `super().configure_optimizers()` 호출 제거
- batch 전체가 empty mask일 때도 `training_step`과 `validation_step`이 crash 나지 않도록 guard 추가

현재 변환 스크립트는 `min_mask_pixels=20` 미만 mask를 기본적으로 제외하므로 일반 학습 batch는 유효 mask를 가진 sample 중심으로 구성된다.

### 3. Local-safe U-Net configs 추가

추가 파일:

- `configs/DavisEyeEllipse_RGBUNet_local_subset.yaml`
- `configs/DavisEyeEllipse_RGBUNet_local_train_smoke.yaml`

`DavisEyeEllipse_RGBUNet_local_subset.yaml`:

- 실제 labelled subset U-Net 재학습용 config
- root path:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset
```

- stale `/mnt/data2T/...` checkpoint path 제거
- logger path를 FACET repo-local `runs/logs` 아래로 변경
- trainer device는 config/env 기반으로 사용 가능

`DavisEyeEllipse_RGBUNet_local_train_smoke.yaml`:

- `/tmp/facet_unet_dataset_smoke`를 읽는 제한 batch smoke config
- train 2 batches + val 2 batches
- checkpoint/log 생성 검증용

## Source H5 Shape Check

Sample file:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis_labelled_with_mask/right/user40_session_2_0_2.h5
```

Observed datasets:

```text
data  shape=(346, 260, 26), dtype=uint8
label shape=(346, 260, 26), dtype=float64
```

변환 시 `transpose(1, 0, 2)`를 적용하여 PNG 저장 기준 `(260, 346)` image/mask로 맞춘다.

## Smoke Dataset Verification

Command:

```bash
PYTHONPATH=. \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/build_unet_dataset_from_h5.py \
  --max-files 4 \
  --output-root /tmp/facet_unet_dataset_smoke \
  --overwrite
```

Result:

- source h5 files: 4
- frames seen: 125
- train samples: 103
- val samples: 22
- skipped empty masks: 0
- output size: about 6.9 MB

`DavisWithMaskDataset` smoke:

- train length: 103
- val length: 22
- batch image shape: `[2, 1, 256, 256]`
- batch mask shape: `[2, 256, 256]`
- batch close shape: `[2]`

U-Net direct smoke:

- train loss: `0.6634`
- validation metrics:
  - `val_loss`: `0.6615`
  - `val_p10_acc`: `0.0`
  - `val_p5_acc`: `0.0`
  - `val_p3_acc`: `0.0`
  - `val_p1_acc`: `0.0`
  - `val_mean_distance`: `172.6809`
  - `val_IoU`: `0.00433`

## `tools/train.py` U-Net Smoke

Command:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_RGBUNet_local_train_smoke.yaml
```

Result:

- exit code: 0
- train: 2 batches
- val: 2 batches
- model params: about 17.3M trainable
- representative metrics:
  - `train_loss`: about `0.793`
  - `val_loss`: about `0.646`
  - `val_mean_distance`: about `175.7243`
  - `val_IoU`: about `0.0203`

Generated smoke artifacts:

```text
runs/logs/RGBUNet_local_train_smoke/version_0/checkpoints/epoch=00-val_mean_distance=175.7243.ckpt
runs/logs/RGBUNet_local_train_smoke/version_0/checkpoints/last.ckpt
runs/logs/RGBUNet_local_train_smoke/version_0/events.out.tfevents.1782373354.etrib.2.0
runs/logs/RGBUNet_local_train_smoke/version_0/hparams.yaml
```

Checkpoint sizes:

- best checkpoint: `207338689 bytes`
- last checkpoint: `207338817 bytes`

## Full Labelled Subset Dataset Generation

Command:

```bash
PYTHONPATH=. \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/build_unet_dataset_from_h5.py \
  --output-root /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset
```

Result manifest:

```text
num_h5_files:        288
total_frames_seen:  9011
num_train:          7201
num_val:            1810
num_samples:        9011
skipped_empty_masks: 0
skipped_shape_errors: []
```

Output size:

```text
470M /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset
```

File count check:

```text
train data:  7201
train label: 7201
val data:    1810
val label:   1810
```

Full dataset DataLoader check:

```text
train len: 7201, image [2, 1, 256, 256], mask [2, 256, 256], close [2]
val len:   1810, image [2, 1, 256, 256], mask [2, 256, 256], close [2]
```

## Note On 9011 vs 8911

U-Net PNG dataset has 9011 samples, while the previously generated EPNet `DeanDataset` subset has 8911 samples.

Reason:

- U-Net PNG dataset uses all h5 `data/label` frame-mask pairs that pass mask pixel threshold.
- EPNet `DeanDataset` generation additionally requires matching raw `Data_davis` frame/event source, extracts ellipse from mask, and requires a non-empty event segment before the label timestamp.
- Therefore the EPNet dataset can have fewer usable samples than the raw labelled mask frame count.

## Current Phase 2 Status

Completed:

- h5 to PNG conversion script
- local-safe U-Net config
- U-Net DataLoader smoke
- U-Net forward/loss/backward smoke
- U-Net `tools/train.py` limited-batch smoke
- full labelled subset PNG dataset generation

Not completed:

- full U-Net training
- U-Net checkpoint selection for full `Data_davis` label expansion
- validation mask visualization report
- full `Data_davis` inference/label expansion

Current blocker:

- 현재 세션에서는 `nvidia-smi`가 NVIDIA driver 통신 실패를 반환하므로 GPU full training은 검증하지 못했다.

Next action:

```bash
PYTHONPATH=. FACET_DEVICES=0 \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_RGBUNet_local_subset.yaml
```

GPU 환경이 복구되면 위 command로 Phase 2 full U-Net 재학습을 시작한다.

