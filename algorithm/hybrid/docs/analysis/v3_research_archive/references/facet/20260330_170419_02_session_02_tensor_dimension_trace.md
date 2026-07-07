# 세션 02: Tensor 차원 추적

날짜: 2026-03-26

## 목적

FACET의 raw dataset artifact부터 최종 모델 입력 tensor까지를 차원 중심으로 추적한다.

## 범위

- pseudo-label 생성 경로
- main FACET event-to-ellipse 학습 경로

## A. pseudo-label 생성 경로

### Raw labeled H5 데이터

- H5 dataset `data`, `label`은 `.transpose(1, 0, 2)`를 거칩니다.
- transpose 이후 레이아웃은 다음과 같습니다.
  - `(H, W, N_frames)`
- 따라서 단일 frame 추출 결과는:
  - image: `(H, W)`
  - mask: `(H, W)`

### U-Net 학습 데이터셋

- `DavisWithMaskDataset`은 grayscale PNG pair를 읽습니다.
- 샘플별 tensor shape:
  - channel 삽입 전 image: `(256, 256)`
  - channel 삽입 후 image: `[1, 256, 256]`
  - mask: `[256, 256]`

### U-Net 추론 데이터셋

- `TestDataset`은 raw grayscale frame을 segmentation inference용으로 준비합니다.
- 샘플별 tensor shape:
  - original image: `(H, W)`
  - transformed tensor: `[1, 240, 346]`

## B. main FACET 학습 경로

### Text label과 event stream

- `TxtProcessor`가 읽는 event text:
  - shape: `[N_events]`
  - fields: `(t, x, y, p)`
- `TxtProcessor`가 읽는 ellipse text:
  - shape: `[N_frames]`
  - fields: `(t, x, y, a, b, ang)`

### Fixed-count sample

- FACET은 ellipse timestamp 하나마다 직전 `5000` event를 잘라 샘플 1개를 만듭니다.
- event slice shape:
  - `[5000]` structured events

### Event image 구성

- `to_frame_stack_numpy(..., n_time_bins=1)` 반환:
  - `[1, 2, 260, 346]`
- `.squeeze(0)` 후:
  - `[2, 260, 346]`
- channel-last 변환 후:
  - `[260, 346, 2]`
- resize 후:
  - `[256, 256, 2]`
- normalization + channel-first 변환 후:
  - `[2, 256, 256]`

### ElNet 변형

- `model_type == "ElNet"`일 때 edge-derived extra channel이 추가됩니다.
- 이 경우 입력 shape:
  - `[7, 256, 256]`

## C. target tensor shape

FACET sample 1개에서 생성되는 supervision:

- `input`: `[2, 256, 256]`
- `hm`: `[1, 64, 64]`
- `ab`: `[100, 2]`
- `ang`: `[100, 1]`
- `trig`: `[100, 2]`
- `reg`: `[100, 2]`
- `ind`: `[100]`
- `reg_mask`: `[100]`
- `mask`: `[1, 64, 64]`
- `center`: `[2]`
- `ellipse`: `[5]`
- `close`: scalar

## D. default collate 이후 batch shape

batch size `B` 기준:

- `input`: `[B, 2, 256, 256]`
- `hm`: `[B, 1, 64, 64]`
- `ab`: `[B, 100, 2]`
- `trig`: `[B, 100, 2]`
- `reg`: `[B, 100, 2]`
- `mask`: `[B, 1, 64, 64]`
- `ind`: `[B, 100]`
- `reg_mask`: `[B, 100]`
- `center`: `[B, 2]`
- `ellipse`: `[B, 5]`
- `close`: `[B]`

## E. EPNet forward shape

입력 `[B, 2, 256, 256]` 기준:

- `out2`: `[B, 24, 64, 64]`
- `out3`: `[B, 40, 32, 32]`
- `out4`: `[B, 112, 16, 16]`
- `out5`: `[B, 160, 8, 8]`
- FPN 최종 `P2`: `[B, 64, 64, 64]`

head 출력:

- `hm`: `[B, 1, 64, 64]`
- `ab`: `[B, 2, 64, 64]`
- `trig`: `[B, 2, 64, 64]`
- `reg`: `[B, 2, 64, 64]`
- `mask`: `[B, 1, 64, 64]`

## 세션 결과

FACET의 핵심 tensor 계약은 다음 한 줄로 요약됩니다.

- raw structured events `[N]` -> fixed-count slice `[5000]` -> event image `[2,256,256]` -> batch `[B,2,256,256]` -> ellipse heads on `[B,*,64,64]`
