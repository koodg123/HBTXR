# FACET Conversation Summary

Date: 2026-06-25

## Scope

이번 대화의 작업 범위는 FACET 코드베이스와 논문을 함께 확인하고, 로컬 EV-Eye 데이터로 FACET 훈련용 dataset을 준비하는 것이었다.

작업 디렉토리:

```text
/home/kjm26/project/PRJXR/HBTXR
```

FACET 코드:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET
```

보고서 저장 위치:

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET
```

## 주요 확인 사항

### 1. FACET codebase와 논문 분석

기존 분석 문서:

```text
references/report/FACET/FACET_code_and_paper_analysis.md
```

핵심 요약:

- FACET의 중심 구현은 `DavisEyeEllipseDataset` + `EPNet` + `DavisEyeEllipse_EPNet.yaml` 경로이다.
- EPNet은 event segment를 2-channel frame representation으로 바꾸고 ellipse 관련 target을 예측한다.
- 논문 설명과 코드가 대부분 대응되지만, 코드에는 논문의 4-head 설명 외에 `mask` head/loss가 추가되어 있다.
- paper의 fast causal event volume limit `l=25`는 현재 active config/code path에서 완전히 명확하게 재현되지 않는다.
- 논문 Table II 수치를 바로 재현하는 완성 artifact는 코드 트리에서 확인되지 않았다.

### 2. split 기준

`split`은 dataset class 내부에서는 이미 존재하는 `train` 또는 `val` 폴더를 선택하는 값이다.

`split_train_val.py`는 segmentation용 PNG 데이터의 `data/*.png` 전체를 랜덤 셔플한 뒤 기본 90:10으로 train/val에 이동한다.

즉 `split_train_val.py`는 EPNet용 `DeanDataset/cached_data/cached_ellipse`를 직접 만드는 스크립트가 아니다.

### 3. 로컬 데이터 구조

로컬 EV-Eye raw root:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data
```

확인된 주요 폴더:

```text
Data_davis
Data_davis_labelled_with_mask
DeanDataset
```

`Data_davis_labelled_with_mask`:

```text
left h5: 144
right h5: 144
total h5: 288
```

이 데이터는 사람이 mask label을 제공한 subset이다.

`Data_davis`:

```text
events.txt: 388 sessions
centers.txt: 0 files
```

전체 frame/event는 있지만 모든 frame에 대한 정답 ellipse label은 없다.

### 4. 8911개 sample의 의미

생성된 8911개는 `Data_davis_labelled_with_mask` 안에 파일이 8911개 있다는 뜻이 아니다.

정확한 의미:

```text
288개 h5 내부의 frame/mask 중 유효 ellipse로 변환 가능한 sample 수 = 8911
```

이 값은 README의 "over 9,000 images" 설명과 대체로 일치한다.

### 5. `num_events`, `event_t_start`, `event_t_end`의 출처

이 값들은 `Data_davis_labelled_with_mask`의 `.h5`에 원래 들어 있는 값이 아니다.

생성 방식:

1. `.h5` mask에서 ellipse label을 만든다.
2. 대응되는 `Data_davis/.../frames/*.png` filename에서 timestamp를 읽는다.
3. 같은 session의 `Data_davis/.../events/events.txt`에서 해당 timestamp 직전 event segment를 자른다.
4. 그 결과에서 `num_events`, `event_t_start`, `event_t_end`를 계산한다.

### 6. 전체 `Data_davis`가 모두 포함되지 않은 이유

전체 `Data_davis`를 FACET EPNet 학습에 쓰려면 frame마다 ellipse label이 필요하다. 현재 로컬에는 전체 `Data_davis`에 대한 ellipse label 또는 segmentation output이 없다.

논문 흐름상 필요한 확장 절차:

```text
Data_davis_labelled_with_mask -> U-Net 학습
Data_davis 전체 frame -> U-Net으로 mask 예측
mask -> ellipse 변환
event stream + ellipse label -> DeanDataset 생성
```

현재 로컬에 없는 것:

- 전체 `Data_davis` segmentation output
- 전체 `Data_davis` ellipse labels
- 원저자 `DeanDataset/events.txt`, `DeanDataset/ellipses.txt`
- README가 참조하는 U-Net checkpoint
- paper split manifest

## 생성/수정된 산출물

### 1. Generated DeanDataset

경로:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset
```

결과:

```text
num_samples: 8911
num_train: 7128
num_val: 1783
size: about 1.2G
```

manifest:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/manifest.json
```

### 2. Sample visualization

경로:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/samples
```

구성:

```text
*_frame.png
*_events.png
*_label.png
*_overlay.png
*_label.txt
README.txt
```

생성된 sample:

```text
train: 0, 1, 10, 100, 1000, 7127
val:   0, 1, 10, 100, 1000, 1782
```

### 3. FACET config 수정

수정 파일:

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet.yaml
```

수정 내용:

```yaml
root_path: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset
```

### 4. Dataset generation script

추가 파일:

```text
references/codebase/software/FACET/EvEye/utils/scripts/build_dean_dataset_from_ev_eye.py
```

목적:

- 현재 로컬 데이터만으로 FACET EPNet이 읽을 수 있는 `DeanDataset` cache를 생성한다.
- 입력은 `Data_davis_labelled_with_mask`의 mask `.h5`와 `Data_davis`의 event stream이다.

## 검증 기록

memmap 정합성:

```text
train event_index_rows = 7128
train ellipse_index_rows = 7128
val event_index_rows = 1783
val ellipse_index_rows = 1783
```

대표 sample:

```text
train[0] events=61
train[7127] events=225
val[0] events=253
val[1782] events=5000
```

sample visualization:

```text
frame PNG count: 12
sample directory file count including README.txt: 61
frame size: 346x260
frame mode: grayscale L
```

## 앞으로의 문서 저장 규칙

FACET 관련 분석, 재현 절차, 데이터셋 생성 기록, 검증 로그는 모두 아래 경로에 저장한다.

```text
/home/kjm26/project/PRJXR/HBTXR/references/report/FACET
```

새 문서를 만들 때는 가능하면 다음 파일명 규칙을 따른다.

```text
FACET_<topic>_<YYYY-MM-DD>.md
```

예:

```text
FACET_training_smoke_test_2026-06-25.md
FACET_full_data_expansion_plan_2026-06-25.md
FACET_validation_results_2026-06-25.md
```

## 남은 작업

1. 현재 생성된 8911개 subset으로 EPNet DataLoader smoke test를 수행한다.
2. `tools/train.py`의 hardcoded GPU device 설정을 현재 시스템에 맞게 점검한다.
3. FACET EPNet 학습이 실제로 시작되는지 짧은 epoch 또는 batch-level smoke run을 수행한다.
4. 논문 전체 재현이 목표라면 U-Net checkpoint 또는 원저자 `ellipses.txt`를 확보한다.
5. 전체 `Data_davis` 확장 label을 생성한 뒤 새 `DeanDataset`을 manifest 기반으로 재생성한다.

