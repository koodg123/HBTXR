# 세션 01: FACET 논문 및 코드 분석

날짜: 2026-03-26

## 목적

FACET 논문과 공개 코드를 함께 읽고, 데이터 처리와 dataloader가 실제로 어떻게 구성되는지 설명한다.

## 검토 대상

- 논문 PDF
  - `references/(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`
- 코드 루트
  - `references/FACET-main/FACET-main/`
- 주요 dataset 구현
  - `references/FACET-main/FACET-main/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`
- cache 유틸리티
  - `references/FACET-main/FACET-main/EvEye/utils/cache/MemmapCacheStructedEvents.py`
- loader factory
  - `references/FACET-main/FACET-main/EvEye/dataset/dataset_factory.py`
- 학습 entry
  - `references/FACET-main/FACET-main/tools/train.py`

## 핵심 발견

### 1. 과제 정의

- FACET은 event-based eye tracking을 직접적인 pupil ellipse detection 문제로 재정의합니다.
- 모델은 pupil ellipse의 중심, 장축/단축, 각도 표현, 보조 mask를 함께 예측합니다.
- 공개 코드에서는 이를 `hm`, `ab`, `trig`, `reg`, `mask` head로 구현합니다.

### 2. 데이터셋 확장 전략

- README 기준 label 생성은 2단계 이상으로 나뉩니다.
- 먼저 `.h5` RGB/mask 데이터를 PNG image/mask pair로 변환합니다.
- 그 다음 U-Net segmentation 모델을 학습하고, 이를 나머지 raw RGB frame에 적용해 pseudo mask를 만듭니다.
- 최종적으로 mask에서 ellipse label을 추출해 text 형태로 저장합니다.

### 3. FACET 학습 데이터 포맷

- README가 설명하는 최종 구조는 split별 text 데이터입니다.
  - `train/data/*.txt`
  - `train/ellipse/*.txt`
  - `val/data/*.txt`
  - `val/ellipse/*.txt`
- 하지만 실제 공개 학습 코드는 memmap cache 형태를 기대합니다.
  - `train/cached_data/`
  - `train/cached_ellipse/`
  - `val/cached_data/`
  - `val/cached_ellipse/`

### 4. fixed-count sample 구성

- FACET은 fixed-count event slicing을 사용합니다.
- 공개 코드 경로는 sample 하나당 직전 `5000`개 이벤트를 하드코딩 수준으로 사용합니다.
- 즉 학습 샘플 1개는 다음으로 구성됩니다.
  - 시점 `t`의 ellipse label 1개
  - 해당 시점 직전의 `5000` event slice 1개

### 5. event representation

- loader는 structured event slice를 2채널 event frame으로 변환합니다.
- active accumulation path는 다음 계열입니다.
  - `causal_linear_ori`
  - `causal_linear`
  - `bilinear`
- 공개 구현은 `to_frame_stack_numpy(..., n_time_bins=1)`를 사용하므로, 주 입력은 temporal stack이 아니라 단일 event image입니다.

### 6. augmentation 및 target 구성

- event dropout, area dropout이 사용됩니다.
- 공간 augmentation은 `Albumentations ReplayCompose`로 적용됩니다.
- ellipse label은 canvas에 rasterize한 뒤 동일 replay transform으로 warp되고, 다시 `cv2.fitEllipse`로 fit됩니다.
- 최종 supervision tensor는 detection style 구조입니다.
  - heatmap
  - axis regression
  - trig angle encoding
  - center offset
  - ellipse mask

### 7. DataLoader 구성

- FACET은 custom collate function을 쓰지 않습니다.
- `dataset_factory.make_dataloader()`가 일반 PyTorch `DataLoader`를 생성합니다.
- config가 dictionary sample의 default collation에 의존합니다.

## 데이터셋과 DataLoader가 만들어지는 방식

### 오프라인 준비 단계

1. `.h5` 기반 labeled frame을 PNG image/mask pair로 변환
2. segmentation U-Net 학습
3. unlabeled RGB frame에 U-Net을 적용해 mask 생성
4. mask에서 ellipse label 생성
5. ellipse timestamp에 맞춰 fixed-count event slice 생성
6. event slice와 ellipse label을 memmap cache로 저장

### 런타임 로딩 단계

1. `DataLoader`가 `DavisEyeEllipseDataset`을 선택
2. `__getitem__`이 cached event slice와 cached ellipse를 읽음
3. event slice를 2채널 event image로 변환
4. image와 ellipse를 함께 augmentation
5. detection target 생성
6. default PyTorch collation으로 batch tensor 생성

## 코드 수준 계약

입력 sample:

- `input`

supervision:

- `hm`
- `ab`
- `trig`
- `reg`
- `mask`
- `center`
- `close`
- `ellipse`

## 논문-코드 정렬 메모

논문과 코드에서 모두 확인되는 점:

- fixed-count event slicing
- event-only ellipse detection
- trig 기반 angle 표현
- geometry-aware ellipse supervision

공개 코드에서 다소 단순화되어 보이는 점:

- 논문은 fast causal event volume 설명이 더 강하지만, 체크한 공개 구현은 causal accumulation + clipping 쪽에 가깝습니다.

공개 코드만으로 완전 복원이 어려운 점:

- 논문에서 사용한 정확한 train/val/test split
- 논문 benchmark 설정과 체크인된 YAML의 완전 일치 여부

## 세션 결과

- FACET 데이터 처리 경로는 “offline cache build + lightweight runtime loader” 구조로 정리됩니다.
- 실질적 런타임 계약은 다음 한 줄로 요약됩니다.
  - cached fixed-count event slice -> event image -> ellipse-aware training targets
