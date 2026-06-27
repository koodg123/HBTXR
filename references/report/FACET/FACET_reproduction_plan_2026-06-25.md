# FACET Reproduction Plan

Date: 2026-06-25

## Summary

FACET 논문 재현은 단계적으로 진행한다. 1차 목표는 현재 생성된 8911개 `DeanDataset` subset으로 EPNet 학습/평가 파이프라인을 재현하는 것이다. 2차 목표는 `Data_davis_labelled_with_mask`로 U-Net을 재학습하고, 전체 `Data_davis` frame에 mask와 ellipse label을 확장하는 것이다. 최종 목표는 full expanded DeanDataset으로 FACET 논문 metric과 비교하는 것이다.

추가 목표로, full expanded DeanDataset이 준비되면 FACET/EPNet baseline과 HBTXR-DeiT를 같은 split에서 병렬 학습한다. EPNet/FACET은 GPU0에서 실행하고, HBTXR는 GPU1에서 실행하여 같은 데이터와 metric 기준으로 비교한다.

기본 문서 저장 위치:

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET
```

## Current State

현재 사용 가능한 로컬 데이터와 산출물:

```text
EV-Eye raw root:
/home/kjm26/project/dataset/XR/EV_Eye/raw_data

Mask-labeled subset:
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis_labelled_with_mask

Full raw frame/event tree:
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis

Generated subset DeanDataset:
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset
```

현재 생성된 subset DeanDataset:

```text
num_samples: 8911
num_train:   7128
num_val:     1783
source:      Data_davis_labelled_with_mask h5 masks + Data_davis events
```

이 dataset은 FACET EPNet 학습 코드가 읽을 수 있는 `cached_data/cached_ellipse` 구조를 가진다. 다만 논문 전체 dataset이 아니라, 현재 로컬에서 신뢰 가능한 mask label이 있는 subset 기반 baseline이다.

## Phase 1: Subset EPNet Baseline

목표:

```text
현재 8911개 DeanDataset subset으로 EPNet training/validation pipeline을 안정화한다.
```

작업:

- `DavisEyeEllipse_EPNet.yaml` 또는 별도 local config가 `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset`을 바라보게 한다.
- `tools/train.py`, `tools/validate.py`의 GPU device 하드코딩을 config 또는 환경변수 기반으로 바꾼다.
- `EPNet/Loss.py`의 portable issue를 수정한다.
  - `from predict import _topk`를 실제 존재하는 `Predict.topk` 경로로 교체한다.
  - `.cuda()` 하드코딩을 입력 tensor device 기반 생성으로 바꾼다.
- DataLoader smoke test, forward/loss backward smoke test, 1-epoch train을 순서대로 수행한다.
- subset full run 결과를 저장한다.

산출물:

```text
references/report/FACET/FACET_subset_smoke_test_<date>.md
references/report/FACET/FACET_subset_training_results_<date>.md
```

성공 기준:

- train/val DataLoader가 batch를 정상 반환한다.
- EPNet forward, `CtdetLoss`, backward가 1 batch 이상 통과한다.
- checkpoint와 metric log가 생성된다.
- `val_p10_acc`, `val_p5_acc`, `val_p1_acc`, `val_mean_distance`, `val_IoU`, `val_AP`가 기록된다.

## Phase 2: U-Net 재학습

목표:

```text
Data_davis_labelled_with_mask로 U-Net segmentation model을 재학습한다.
```

작업:

- `.h5` mask-labeled data를 U-Net 학습용 PNG `data/label` 구조로 변환한다.
- deterministic split manifest를 만든다.
- `DavisEyeEllipse_RGBUNet.yaml`을 local-safe config로 복제하거나 수정한다.
- U-Net 학습 smoke test와 full training을 수행한다.
- validation mask sample과 metric을 저장한다.

산출물:

```text
U-Net checkpoint
U-Net validation report
sample mask visualization
references/report/FACET/FACET_unet_training_results_<date>.md
```

성공 기준:

- U-Net 학습이 crash 없이 완료된다.
- mask prediction sample이 원본 frame과 정렬된다.
- 전체 `Data_davis` label expansion에 사용할 checkpoint가 확보된다.

## Phase 3: Full Data_davis Label Expansion

목표:

```text
전체 Data_davis frame에 mask와 ellipse label을 생성하고 full DeanDataset을 만든다.
```

작업:

- U-Net checkpoint로 `Data_davis/user*/left|right/session_*/frames/*.png` 전체에 segmentation mask를 생성한다.
- 생성된 mask에서 ellipse `(t, x, y, a, b, ang)`를 추출한다.
- 같은 session의 `events/events.txt`에서 각 ellipse timestamp 직전 최대 5000개 event를 자른다.
- full expanded DeanDataset을 생성한다.

권장 출력 경로:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
```

manifest에 기록할 항목:

```text
source raw root
U-Net checkpoint path
total frames scanned
valid ellipse count
skipped frame count
train/val/test split rule
event count rule
generation command
generation timestamp
```

성공 기준:

- full expanded dataset이 `DavisEyeEllipseDataset`으로 로드된다.
- train/val/test 또는 train/val split별 sample 수가 manifest에 기록된다.
- 시각 확인용 `samples`가 생성된다.

## Phase 4: Full FACET Reproduction

목표:

```text
full expanded DeanDataset으로 EPNet을 학습하고 FACET 논문 metric과 비교한다.
```

작업:

- full dataset용 EPNet config를 작성한다.
- 기본 실험은 현재 code config의 `mode: fpn_2d`로 수행한다.
- 논문 대응성을 위해 `mode: fpn_dw` ablation도 수행한다.
- validation metric과 checkpoint를 저장한다.
- 논문 Table II 수치와 현재 결과를 비교한다.

비교 metric:

```text
P10
P5
P3
P1
mean pixel error
IoU
AP
parameter count
FLOPs
inference latency
```

산출물:

```text
references/report/FACET/FACET_reproduction_results_<date>.md
references/report/FACET/FACET_table2_comparison_<date>.md
```

성공 기준:

- full train/validation이 완료된다.
- 논문 metric과 현재 metric의 차이를 표로 정리한다.
- 차이 원인을 dataset size, split, label generation, model mode, runtime/backend 차이로 분해한다.

## Phase 4B: HBTXR Parallel Comparison

목표:

```text
full expanded DeanDataset으로 HBTXR-DeiT를 EPNet/FACET baseline과 병렬 학습하고 같은 metric으로 비교한다.
```

작업:

- HBTXR 이름공간의 DeiT backbone, head, loss, predict, metric 구현을 사용한다.
- EPNet 직접 import 없이 `model.type: HBTXR`로 학습 가능해야 한다.
- `DavisEyeEllipse_HBTXR_full_unet.yaml`은 EPNet full config와 같은 `DeanDataset_full_unet` root를 사용한다.
- `DeanDataset_full_unet/manifest.json` 생성 후 EPNet/FACET은 GPU0, HBTXR는 GPU1에서 동시에 시작한다.
- 두 모델의 checkpoint, TensorBoard log, validation metric, params/FLOPs/latency를 별도 저장한다.

실행 배치:

```text
EPNet/FACET: GPU0, DavisEyeEllipse_EPNet_full_unet.yaml
HBTXR:       GPU1, DavisEyeEllipse_HBTXR_full_unet.yaml
```

산출물:

```text
references/report/FACET/FACET_parallel_epnet_hbtxr_training_<date>.md
references/report/FACET/FACET_hbtxr_reproduction_results_<date>.md
references/report/FACET/FACET_epnet_vs_hbtxr_comparison_<date>.md
```

성공 기준:

- HBTXR full train/validation이 완료된다.
- EPNet/FACET과 HBTXR가 같은 validation split과 metric code로 평가된다.
- HBTXR 결과가 논문 FACET metric 및 로컬 EPNet baseline과 함께 비교표로 정리된다.

## Test Plan

Preflight:

```bash
python -c "import torch, lightning, cv2, albumentations, timm, tonic, h5py"
nvidia-smi
```

Dataset smoke:

```text
DavisEyeEllipseDataset(train)[0]
DavisEyeEllipseDataset(val)[0]
DataLoader batch_size=2, two batches
```

Expected batch fields:

```text
input
hm
reg_mask
ind
ab
trig
mask
reg
center
close
ellipse
```

Model smoke:

```text
EPNet forward 1 batch
CtdetLoss forward 1 batch
loss.backward() 1 batch
validation_step 1 batch
```

Training smoke:

```text
1 batch overfit
1 epoch subset run
full subset run
U-Net 1 epoch run
full Data_davis expansion dry-run on 1 session
full Data_davis expansion
full EPNet run
full HBTXR run on GPU1 in parallel with EPNet/FACET
```

## Assumptions

- 재현 목표는 단계적 재현이다.
- 전체 `Data_davis` 확장 라벨은 U-Net 재학습으로 생성한다.
- 학습은 다중 GPU 사용 가능을 기본 가정으로 한다.
- full expanded dataset 학습 단계에서는 EPNet/FACET baseline은 GPU0, HBTXR-DeiT는 GPU1에 배치한다.
- 모든 smoke test는 단일 GPU에서도 실행 가능해야 한다.
- 현재 8911개 `DeanDataset`은 논문 전체 dataset이 아니라 subset baseline이다.
- 원저자 `ellipses.txt`, U-Net checkpoint, paper split manifest가 확보되면 Phase 2-3은 해당 산출물 사용 경로로 단축할 수 있다.
- FACET 관련 문서, 계획, 실행 로그, 결과는 모두 `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET`에 저장한다.
