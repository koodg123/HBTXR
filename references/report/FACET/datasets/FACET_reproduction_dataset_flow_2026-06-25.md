# FACET Reproduction Dataset Flow

Date: 2026-06-25

## 1. Purpose

이 문서는 FACET 코드베이스로 논문 결과를 재현하기 위해 확인한 데이터 흐름, split 기준, 현재 로컬 데이터로 생성한 `DeanDataset`, 그리고 남은 한계를 정리한다.

관련 경로:

- FACET code: `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET`
- FACET report directory: `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET`
- EV-Eye raw root: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data`
- Mask-labeled subset: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis_labelled_with_mask`
- Full raw event/frame data: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis`
- Generated FACET training dataset: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset`

## 2. FACET EPNet이 기대하는 데이터 구조

`DavisEyeEllipseDataset`는 최종 EPNet/FACET 학습용 dataset class이다. 이 class는 RGB frame을 직접 학습 입력으로 쓰지 않고, 아래 구조의 event segment와 ellipse label cache를 읽는다.

```text
DeanDataset/
  train/
    cached_data/
      events_batch_*.memmap
      events_batch_info_*.txt
      events_indices_*.npy
    cached_ellipse/
      ellipses_batch_0.memmap
      ellipses_batch_info_0.txt
      ellipses_indices_0.npy
  val/
    cached_data/
    cached_ellipse/
```

각 sample은 다음 흐름으로 사용된다.

1. `cached_ellipse`에서 ellipse label 1개를 읽는다.
2. `cached_data`에서 해당 label 시점에 대응하는 event segment를 읽는다.
3. 코드상 `load_event_segment(..., 5000)` 경로가 사용되므로 fixed-count 5000 event segment가 기본이다.
4. event segment를 2-channel event frame representation으로 변환한다.
5. `hm`, `ab`, `trig`, `reg`, `mask` target을 생성한다.

## 3. split의 의미

FACET 코드에서 `split`은 두 가지 의미로 쓰인다.

### 3.1 Dataset class 내부의 `split`

`split="train"` 또는 `split="val"`은 이미 만들어진 폴더 중 어떤 것을 읽을지 선택하는 값이다.

예:

```text
root_path/train/cached_data
root_path/train/cached_ellipse
root_path/val/cached_data
root_path/val/cached_ellipse
```

또한 `train` split에서는 event/image augmentation이 켜지고, `val`에서는 대체로 resize 또는 no-op transform만 적용된다.

### 3.2 `split_train_val.py`의 split

`EvEye/utils/scripts/split_train_val.py`는 segmentation용 PNG 데이터에 대해:

1. `base_path/data/*.png` 전체를 읽고,
2. `shuffle(file_names)`로 무작위 셔플한 뒤,
3. 기본 `train_ratio=0.9`로 90% train, 10% val로 이동한다.

이 스크립트는 EPNet용 `cached_data/cached_ellipse`를 만드는 스크립트가 아니다. 또한 seed가 없어서 실행마다 split이 달라질 수 있다.

## 4. 원본 데이터의 역할

### 4.1 `Data_davis_labelled_with_mask`

이 경로는 사람이 라벨링한 mask가 있는 subset이다.

현재 로컬 확인 결과:

```text
left h5 files:  144
right h5 files: 144
total h5 files: 288
```

이 `.h5` 파일 내부에는 frame/mask label이 들어 있고, 이 mask에서 ellipse label을 만들 수 있다.

중요: `num_events`, `event_t_start`, `event_t_end`는 이 `.h5`에 원래 포함된 값이 아니다. 이 값들은 `Data_davis`의 `events.txt`에서 event segment를 잘라낸 뒤 계산한 metadata이다.

### 4.2 `Data_davis`

이 경로는 전체 EV-Eye 원본 frame/event tree이다.

현재 로컬 확인 결과:

```text
events.txt sessions: 388
centers.txt files:   0
```

즉 frame과 event stream은 있지만 모든 frame에 대해 정답 mask/ellipse label이 존재하지 않는다.

### 4.3 왜 전체 `Data_davis`를 모두 쓰지 못했는가

FACET 논문 흐름에서는:

```text
Data_davis_labelled_with_mask
  -> U-Net segmentation model 학습
Data_davis 전체 frame
  -> 학습된 U-Net으로 mask label 확장
확장된 mask
  -> ellipse label 변환
event stream + ellipse label
  -> EPNet/FACET 학습용 DeanDataset 생성
```

하지만 현재 로컬에는 다음 산출물이 없다.

- 전체 `Data_davis`에 대한 segmentation output
- 전체 `Data_davis`에 대한 `ellipses.txt`
- README가 참조하는 U-Net checkpoint
- paper split을 그대로 재현할 수 있는 manifest

따라서 현재 생성한 dataset은 "로컬에서 신뢰 가능한 라벨이 있는 subset"인 `Data_davis_labelled_with_mask` 기반이다.

## 5. 생성한 DeanDataset

생성 경로:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset
```

생성에 사용한 입력:

```text
Data_davis_labelled_with_mask/*.h5
Data_davis/user*/left|right/session_*/events/events.txt
Data_davis/user*/left|right/session_*/frames/*.png
```

생성 결과:

```text
num_samples: 8911
num_train:   7128
num_val:     1783
train_ratio: 0.8
batch_size:  5000
dataset size: about 1.2G
```

생성 manifest:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/manifest.json
```

manifest 핵심 내용:

```json
{
  "source": "Data_davis_labelled_with_mask h5 masks + Data_davis events",
  "raw_root": "/home/kjm26/project/dataset/XR/EV_Eye/raw_data",
  "output_root": "/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset",
  "train_ratio": 0.8,
  "batch_size": 5000,
  "num_samples": 8911,
  "num_train": 7128,
  "num_val": 1783,
  "num_skipped_sources": 0
}
```

## 6. 생성 스크립트와 코드 수정

추가한 생성 스크립트:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/EvEye/utils/scripts/build_dean_dataset_from_ev_eye.py
```

역할:

1. `Data_davis_labelled_with_mask`의 `.h5` mask를 읽는다.
2. mask에서 ellipse label `(t, x, y, a, b, ang)`를 추정한다.
3. frame filename timestamp를 기준으로 `Data_davis`의 `events.txt`에서 직전 최대 5000개 event를 자른다.
4. `DavisEyeEllipseDataset` 호환 `cached_data/cached_ellipse` memmap을 생성한다.

실행 환경:

```text
/home/kjm26/project/PRJXR/HBTXR/.facet-prep-venv
```

설치된 생성용 의존성:

```text
h5py
numpy
tqdm
```

실행 명령:

```bash
cd /home/kjm26/project/PRJXR/HBTXR
.facet-prep-venv/bin/python \
  references/codebase/software/FACET/EvEye/utils/scripts/build_dean_dataset_from_ev_eye.py \
  --raw-root /home/kjm26/project/dataset/XR/EV_Eye/raw_data \
  --output-root /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset
```

수정한 FACET config:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet.yaml
```

변경 내용:

```yaml
root_path: /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset
```

train과 val 양쪽 모두 같은 `DeanDataset` root를 바라보도록 수정했다.

## 7. 검증 결과

memmap index 정합성:

```text
train event_index_rows:   7128
train ellipse_index_rows: 7128
val event_index_rows:     1783
val ellipse_index_rows:   1783
```

대표 sample 접근 검증:

```text
train[0]    events=61,   ellipse_t=1657711084457716
train[7127] events=225,  ellipse_t=1658299675790864
val[0]      events=253,  ellipse_t=1658299675830864
val[1782]   events=5000, ellipse_t=1659091434253100
```

주의: 초기 frame에서는 해당 timestamp 이전 event가 5000개보다 적을 수 있으므로 `num_events`가 5000보다 작을 수 있다.

## 8. 시각 확인용 samples

생성 위치:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/samples
```

생성한 sample:

```text
train: 0, 1, 10, 100, 1000, 7127
val:   0, 1, 10, 100, 1000, 1782
```

각 sample 구성:

```text
*_frame.png    원본 EV-Eye frame
*_events.png   event segment 시각화
*_label.png    ellipse label 렌더링
*_overlay.png  event image 위에 회전 ellipse label 오버레이
*_label.txt    split, index, event range, ellipse tuple, 원본 경로 metadata
```

검증:

```text
frame PNG count: 12
total sample directory files including README.txt: 61
frame resolution: 346x260
frame mode: grayscale L
```

색상:

```text
events.png: p<=0 event는 red, p>0 event는 cyan 계열
label.png: ellipse label 렌더링
overlay.png: event visualization + rotated ellipse
```

## 9. 현재 dataset의 의미와 한계

현재 생성한 `DeanDataset`는 FACET EPNet 학습 코드가 읽을 수 있는 형태이다. 그러나 논문 전체 재현 dataset과 완전히 같다고 단정하면 안 된다.

현재 dataset:

```text
Data_davis_labelled_with_mask 기반 8911개 유효 ellipse sample
```

논문식 전체 확장 dataset:

```text
Data_davis 전체 frame/event
+ U-Net으로 생성한 segmentation mask
+ mask-to-ellipse 변환 결과
+ paper split 또는 고정 manifest
```

따라서 paper의 20k/5k/5k 또는 Table II 수치를 재현하려면 다음 중 하나가 추가로 필요하다.

1. 원 저자가 사용한 `DeanDataset/events.txt`와 `DeanDataset/ellipses.txt`.
2. 전체 `Data_davis`에 대한 segmentation output 또는 ellipse label.
3. README가 참조하는 U-Net checkpoint를 확보하고 전체 `Data_davis` frame에 추론을 수행하는 절차.
4. split manifest와 평가 프로토콜.

## 10. Next Actions

권장 순서:

1. 현재 생성된 8911개 subset으로 EPNet 학습 smoke run을 수행한다.
2. hardcoded GPU device 설정을 현재 머신에 맞게 수정한다.
3. 학습 전 `DavisEyeEllipseDataset`이 실제 batch를 정상 반환하는지 DataLoader smoke test를 수행한다.
4. 논문 재현이 목표라면 U-Net checkpoint 또는 원저자 `ellipses.txt`를 확보한다.
5. 확보 후 전체 `Data_davis` 확장 DeanDataset을 새 manifest 기반으로 재생성한다.

