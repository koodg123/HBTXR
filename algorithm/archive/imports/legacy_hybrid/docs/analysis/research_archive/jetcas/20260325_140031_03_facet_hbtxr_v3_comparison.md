# FACET vs HBTXR_v3_0 Comparison

이 문서는 지금까지 대화에서 정리한 `FACET`과 `HBTXR_v3_0`의 차이점을 한곳에 모아 기록한 문서입니다.
범위는 아래와 같습니다.

- 데이터 세트 생성
- 데이터 구조와 저장 포맷
- 데이터 처리와 전처리
- 이벤트 표현과 누적 방식
- 데이터 로딩과 sample contract
- 입출력 타깃 구조
- `utils` / `logger` 활용 여부
- `FACET`에서 가져온 것과 가져오지 않은 것

## Reviewed Sources

- `references_repo/FACET/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`
- `references_repo/FACET/FACET/EvEye/dataset/DavisEyeCenter/MemmapDavisEyeCenterDataset.py`
- `references_repo/FACET/FACET/EvEye/utils/tonic/functional/ToFrameStack.py`
- `references_repo/FACET/FACET/EvEye/utils/dvs_common_utils/representation/NumpyFrameStack.py`
- `references_repo/FACET/FACET/EvEye/logger/logger_factory.py`
- `HBTXR_v3_0/src/hbtxr/preprocess/canonicalize.py`
- `HBTXR_v3_0/src/hbtxr/preprocess/build_manifests.py`
- `HBTXR_v3_0/src/hbtxr/data/dataset.py`

## One-Line Summary

- `FACET`는 task-specific dataset이 학습 입력을 바로 만들어 주는 구조다.
- `HBTXR_v3_0`는 raw -> canonical -> manifest -> dataloader로 단계를 분리한 시스템형 구조다.

## High-Level Architecture Difference

### FACET

- dataset class가 데이터 포맷, event representation, augmentation, target encoding을 한 번에 처리한다.
- task별 dataset class가 다르다.
- 학습용 입력 구조가 모델과 강하게 결합돼 있다.
- 이미 준비된 `train/val/test` 데이터 디렉토리를 바로 읽는 편이다.

### HBTXR_v3_0

- 데이터 준비와 학습 입력 생성을 단계별로 분리한다.
- canonical truth를 먼저 만들고, 이후 manifest를 만들고, 마지막에 dataloader가 sample을 조합한다.
- 한 dataset contract 안에 frame/event/search/track supervision을 같이 담는다.
- 모델 실험보다도 데이터 provenance와 runtime contract 정합성을 함께 고려한 구조다.

## 1. 데이터 세트 생성 방식 차이

### FACET

- `FACET`는 기본적으로 학습 가능한 형태의 dataset root를 전제로 한다.
- 예를 들어 `DavisEyeEllipse_EPNet.yaml`에서는 바로 `root_path`, `split`, `dataset type`을 지정해 dataloader를 만든다.
- 즉, raw 데이터를 canonical truth로 표준화하는 별도 공식 단계가 약하다.
- `cached_data`, `cached_ellipse`, `cached_label`, memmap cache 같은 학습 최적화 포맷이 dataset 앞단에 이미 준비돼 있어야 한다.

### HBTXR_v3_0

- raw session을 먼저 canonical session으로 변환한다.
- canonicalization 단계에서 아래를 만든다.
  - `frames/`
  - `events/events.npz`
  - `labels/frame_annotations.jsonl`
  - `labels/masks/`
  - `labels/session_package.json`
  - `indexes/sessions.jsonl`
- 그 다음 manifest를 생성한다.
- manifest에는 split, ROI, resize policy, event window, annotation ref, prev annotation ref가 들어간다.

### 차이의 의미

- `FACET`는 “dataset이 이미 만들어져 있다”는 가정이 강하다.
- `HBTXR_v3_0`는 “dataset을 어떻게 만들었는지” 자체를 시스템에 포함한다.
- 그래서 `HBTXR_v3_0`가 provenance 추적과 재현성에는 유리하고, `FACET`는 빠른 연구 반복에는 유리하다.

## 2. 데이터 저장 구조 차이

### FACET

- split 중심 디렉토리 구조가 강하다.
- 예:
  - `train/cached_data`
  - `train/cached_ellipse`
  - `train/cached_label`
  - memmap, npy, cached structured events
- 저장 포맷은 task별로 달라질 수 있다.
- dataset이 어떤 캐시를 쓰는지에 따라 `MemmapCacheStructedEvents`, `load_event_segment`, cached labels 등 접근 방식이 달라진다.

### HBTXR_v3_0

- session 중심 canonical 구조를 먼저 가진다.
- 예:
  - `sessions/user01/left/session_101/frames`
  - `sessions/user01/left/session_101/events/events.npz`
  - `sessions/user01/left/session_101/labels/frame_annotations.jsonl`
  - `sessions/user01/left/session_101/labels/masks`
  - `sessions/user01/left/session_101/labels/session_package.json`
- split은 canonical 안에 박혀 있지 않고, manifest 생성 단계에서 결정된다.
- 즉 canonical truth와 experiment split이 분리돼 있다.

### 차이의 의미

- `FACET`는 split-ready cache dataset이다.
- `HBTXR_v3_0`는 split-agnostic canonical truth + experiment-specific manifest 구조다.

## 3. 데이터 처리와 전처리 차이

### FACET DavisEyeEllipse

- `__getitem__`에서 ellipse를 읽고, event segment를 읽고, 이벤트 dropout을 적용하고, event frame을 만들고, resize와 augmentation을 적용하고, ellipse를 다시 fitting한 뒤 detection target을 만든다.
- 전처리 흐름이 dataset 내부에 압축돼 있다.

주요 특징:

- event transform:
  - `DropEvent`
  - `DropEventByArea`
- image transform:
  - `A.Resize(256,256)`
  - `A.ShiftScaleRotate`
  - `A.HorizontalFlip`
- ellipse transform:
  - mask로 rasterize
  - replay transform 적용
  - contour 추출
  - `cv2.fitEllipse`로 다시 fitting

### FACET DavisEyeCenter Memmap

- time segment를 먼저 자르고, 필요 시 spatial downsample을 하고, `frames_per_segment` 길이의 event frame stack을 만든다.
- keypoint label에 ReplayCompose를 적용한다.
- train 시 temporal flip도 수행한다.

### HBTXR_v3_0

- dataset 내부 전처리는 크게 3단계다.
  - manifest row 해석
  - canonical annotation / mask / event 읽기
  - 동일한 `SpatialTransform`으로 frame, mask, event, bbox, ellipse, state를 함께 변환
- `HBTXR_v3_0`는 augmentation보다 geometry consistency를 우선한다.
- current implementation 기준으로 FACET식 heavy augmentation parity는 아직 일부만 반영됐다.

### 차이의 의미

- `FACET`는 dataset이 곧 preprocessing pipeline이다.
- `HBTXR_v3_0`는 preprocessing을 geometry-preserving transform 중심으로 구조화했다.
- `FACET`는 task-specific augmentation 강도가 높고, `HBTXR_v3_0`는 modality alignment와 target consistency가 더 중요하다.

## 4. 이벤트 표현 방식 차이

### FACET

- `to_frame_stack_numpy()`가 핵심 event representation 함수다.
- 지원 모드:
  - `nearest`
  - `bilinear`
  - `causal_linear`
- `n_time_bins`를 지원해 temporal frame stack 표현이 가능하다.
- `DavisEyeEllipseDataset`는 보통 `n_time_bins=1`로 single event frame을 사용한다.
- `MemmapDavisEyeCenterDataset`는 `frames_per_segment` 길이의 time sequence를 만든다.
- 일부 설정에서는 `causal_linear_ori`와 `causal_linear`를 구분하고, `cut_max_count(..., maxcount=255)`로 saturation clip을 적용한다.

### HBTXR_v3_0

- 최종 event 입력은 공식적으로 `single-window [2,H,W]`로 통일했다.
- 지원 정책:
  - `fixed_count`
  - `time_bin`
- 지원 누적 방식:
  - `plain`
  - `causal_linear`
- `causal_linear`는 FACET 취지에 맞게 timestamp-aware weighting으로 구현했다.
- 선택된 윈도우 안에서 최근 timestamp 이벤트가 더 크게 반영된다.
- 현재는 multi-bin stack ABI를 공식 surface로 열지 않았다.

### 차이의 의미

- `FACET`는 time stack 기반 실험을 자연스럽게 지원한다.
- `HBTXR_v3_0`는 accelerator와 hybrid tracker ABI를 고려해 single-window 2-channel 표현으로 정리했다.
- 즉, `FACET`는 표현 실험에 열려 있고, `HBTXR_v3_0`는 runtime-oriented contract에 더 맞춰져 있다.

## 5. 고정 이벤트 수 / 시간 구간 선택 차이

### FACET

- task별 dataset에서 `fixed_count`, `time_window`, `frames_per_segment` 등을 직접 조합한다.
- dataset class마다 이벤트 선택 로직이 조금씩 다를 수 있다.
- `DavisEyeCenter` 계열은 같은 segment 안에서 frame마다 fixed-count를 다시 slice하는 구조도 사용한다.

### HBTXR_v3_0

- manifest row에 `event_window`를 명시한다.
- dataloader는 `event_window`와 현재 config의 `event_builder`를 바탕으로 selection을 수행한다.
- 현재는 config / shell override가 실제 동작 policy를 결정하고, manifest 값은 provenance로 함께 남긴다.

### 차이의 의미

- `FACET`는 dataset 구현이 policy를 소유한다.
- `HBTXR_v3_0`는 manifest와 config가 policy를 소유하고 dataset은 이를 해석한다.

## 6. Resize와 geometry transform 차이

### FACET

- `Albumentations ReplayCompose`를 사용한다.
- `DavisEyeEllipseDataset`는 image transform 뒤 ellipse를 다시 fitting한다.
- `DavisEyeCenter` 계열은 keypoint transform을 사용한다.
- resize는 사실상 `A.Resize(256,256)` 기반 warp resize가 기본이다.

### HBTXR_v3_0

- `SpatialTransform`으로 affine matrix를 명시적으로 관리한다.
- 지원 policy:
  - `facet_square_direct`
  - `letterbox_square`
  - `sensor_full_square`
- frame / mask / event / bbox / ellipse / state6가 같은 transform을 공유한다.
- ellipse는 analytic covariance transform으로 옮긴다.

### 차이의 의미

- `FACET`는 replay-based augmentation 친화적이다.
- `HBTXR_v3_0`는 transform provenance와 analytic consistency에 더 강하다.

## 7. 데이터 로딩 방식 차이

### FACET

- dataset class가 cached structured events나 memmap을 직접 읽는다.
- task별 반환 형식이 다르다.
- 예:
  - ellipse task: dict 반환
  - center task: `(event_frames, labels, closes)` tuple 반환
- 일부 class는 `__getitem__`마다 캐시 파일을 다시 로드하는 구조라, 구현 방식에 따라 I/O 효율 차이가 생길 수 있다.

### HBTXR_v3_0

- `EVEyeHBTXRDataset` 하나가 canonical + manifest를 읽는다.
- annotation store와 `events.npz`는 dataset 내부 캐시를 둔다.
- event frame은 필요 시 `cache_root` 아래 npz 캐시로 저장할 수 있다.
- sample contract가 통일돼 있어 trainer 쪽 collate와 device 이동 로직이 단순하다.

### 차이의 의미

- `FACET`는 task 최적화는 좋지만 dataset마다 loader contract가 다르다.
- `HBTXR_v3_0`는 hybrid tracker 하나의 contract로 통일돼 있다.

## 8. 입력 / 출력 계약 차이

### FACET DavisEyeEllipse

입력:

- `input`: event frame `[2,256,256]`

주요 출력 target:

- `hm`
- `ab`
- `ang` 또는 `trig`
- `reg`
- `mask`
- `ellipse`
- `close`

성격:

- detection-style ellipse localization
- event-only single-task supervision

### FACET MemmapDavisEyeCenter

입력:

- `event_frames [2,T,H,W]`

주요 출력 target:

- `labels [2,T]`
- `closes [T]`

성격:

- temporal center tracking / sequence learning

### HBTXR_v3_0

입력:

- `frame [1,256,256]`
- `event [2,256,256]`
- `prev_state [6]`

출력 target:

- `mask_target [1,256,256]`
- `eye_target [5]`
- `cur_state [6]`
- `pupil_search_target [7]`
- `pupil_track_target [8]`
- `constraint_center [2]`
- `annotation_quality`
- `similarity_target`
- `event_density`
- `closed_eye_flag`
- `mask_valid`
- `valid_track`

성격:

- multimodal search + track supervision
- runtime transition과 quality gating까지 dataset 단계에서 준비

## 9. annotation truth 관점 차이

### FACET

- task dataset이 요구하는 label 형식이 바로 목표다.
- ellipse, center, mask 등 task별 truth가 분리돼 있다.
- canonical source of truth 개념은 상대적으로 약하다.

### HBTXR_v3_0

- canonical annotation row가 먼저 존재한다.
- 그 위에서 frame/event/search/track target이 파생된다.
- annotation source, quality, `closed_eye_flag`, `mask_valid`를 공식 필드로 둔다.

### 차이의 의미

- `FACET`는 task-first label organization이다.
- `HBTXR_v3_0`는 truth-first organization이다.

## 10. split 관리 차이

### FACET

- split이 dataset root와 캐시에 직접 반영돼 있는 경우가 많다.
- train/val/test를 바꾸려면 dataset 준비 단계나 config root 자체를 바꾸는 식이다.

### HBTXR_v3_0

- split은 manifest 생성 단계에서 계산된다.
- user split 정책:
  - `random`
  - `exgaze`
  - `exgaze_with_val`
- canonical truth는 split과 독립적이다.

### 차이의 의미

- `HBTXR_v3_0`가 split 실험을 더 반복하기 쉽다.
- `FACET`는 split-ready dataset 준비가 선행된다.

## 11. logger / utils 활용 차이

### FACET logger

- `logger_factory.py`는 Lightning `TensorBoardLogger` wrapper다.
- Lightning trainer와 callback factory 구조를 전제로 한다.

### HBTXR_v3_0 decision

- 현재 `HBTXR_v3_0`는 custom PyTorch trainer를 사용하므로 FACET logger를 그대로 가져오지 않았다.
- 현재 실험 기록은 아래로 유지한다.
  - `history.json`
  - `history.jsonl`
  - `train_log.txt`
  - `Hyperparameters/resolved_config.yaml`

### FACET utils

- `EvEye/utils`는 범위가 넓고 task-specific, Lightning-specific, tonic-specific, memmap-specific 코드가 많이 섞여 있다.
- 그래서 통째로 가져오지 않았다.

### 부분적으로 참고하거나 가져온 개념

- `fixed_count`
- `time_bin` 대응 개념
- `causal_linear`
- FACET 스타일 direct resize
- replay 가능한 transform metadata 필요성
- memmap cache 전략의 필요성

## 12. 실제로 FACET에서 가져온 것

### 이미 반영한 것

- timestamp-aware `causal_linear`
- `fixed_count | time_bin` 선택 가능 구조
- `facet_square_direct` resize policy
- event representation을 manifest/config에서 제어하는 구조

### 참고만 하고 그대로 가져오지 않은 것

- Lightning logger / callback factory
- FACET dataset class 자체
- EPNet용 heatmap target encoding 전체
- sequence stack dataloader ABI
- 전체 `EvEye/utils`

## 13. 현재도 남아 있는 차이

아직 `FACET`와 완전히 같지 않은 부분:

- FACET 수준의 replay-based strong augmentation parity
- event clipping / saturation 처리 세부값
- memmap-backed 대용량 dataset backend
- multi-bin / sequence event ABI
- EPNet 스타일 heatmap supervision
- tonic transform을 직접 쓰는 loader branch

즉, 지금 `HBTXR_v3_0`는 FACET를 “재구현”한 것이 아니라, FACET의 유용한 데이터 표현 개념을 hybrid tracking 구조에 맞게 흡수한 상태다.

## 14. 장단점 정리

### FACET 장점

- task dataset이 단순하고 빠르게 실험 가능
- event representation과 augmentation 실험이 편함
- EPNet류 task에는 매우 직접적임

### FACET 단점

- dataset, utils, cache 포맷, trainer 의존성이 강하게 결합돼 있음
- canonical provenance가 약함
- task가 바뀌면 dataset contract를 다시 만들어야 할 가능성이 큼

### HBTXR_v3_0 장점

- canonical truth와 manifest가 분리돼 재현성에 유리함
- hybrid search/track runtime에 바로 연결되는 input/output contract를 가짐
- shell/config override 기반 실험 관리가 쉬움

### HBTXR_v3_0 단점

- 초기 세팅과 단계 수가 많아 구조가 더 무겁다
- FACET식 강한 augmentation parity는 아직 덜 들어와 있다
- single-window ABI 때문에 일부 temporal stack 실험은 별도 확장이 필요하다

## 15. 결론

- `FACET`는 모델 중심 dataset 파이프라인이다.
- `HBTXR_v3_0`는 시스템 중심 dataset 파이프라인이다.
- `HBTXR_v3_0`는 FACET를 그대로 복제하는 방향보다, `FACET`의 event representation / resize / supervision 힌트를 hybrid tracker 계약에 맞게 흡수하는 방향이 맞다.
- 따라서 현재 방향은 적절하며, 이후 필요하면 아래 순서로 FACET parity를 확장하는 것이 좋다.
  - replay-based augmentation
  - memmap backend
  - optional time-stack loader
  - optional EPNet-style auxiliary target branch
