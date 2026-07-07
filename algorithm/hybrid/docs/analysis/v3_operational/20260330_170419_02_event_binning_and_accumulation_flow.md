# 이벤트 Binning 및 Accumulation 흐름

최종 갱신: 2026-03-30 KST

Role: `이벤트 파이프라인 분석가`, `데이터셋 런타임 아키텍트`, `텐서 인터페이스 엔지니어`

## 요약

- `HBTXR_v3_0`의 기본 이벤트 경로는 `fixed_count + fast_causal_linear`입니다.
- 이벤트 창 선택은 manifest에 기록되고, 실제 binning과 accumulation은 active dataset 계층에서 수행됩니다.
- 현재 활성 정책은 `fixed_count`, `time_bin`, `interval_all` 세 가지이며, 누적 방식은 `plain`, `causal_linear`, `causal_linear_ori`, `fast_causal_linear`을 지원합니다.
- 이벤트는 먼저 센서 해상도에서 `[2, H_sensor, W_sensor]` polarity-split voxel로 누적된 뒤, ROI crop/resize를 거쳐 `[2, 256, 256]` 모델 입력 텐서로 변환됩니다.

## 1. 기본 event builder 계약

기본 설정은 다음 파일에서 정의됩니다.

- `configs/base.yaml`
- `src/hbtxr/data/dataset.py`
- `src/hbtxr/preprocess/build_manifests.py`

기본 builder 값:

- `policy = fixed_count`
- `event_count_target = 5000`
- `time_bin_us = 5000`
- `accumulation = fast_causal_linear`
- `causal_weight_power = 1.0`
- `fast_causal_limit = 25.0`
- `polarity_split = true`

즉, 기본 동작은 “현재 시점 직전의 최근 5000개 이벤트를 뽑고, 시간적으로 뒤쪽 이벤트에 더 큰 가중치를 주되 각 polarity/pixel cell은 25.0에서 saturation”하는 구조입니다.

## 2. 이벤트 binning이 결정되는 위치

이벤트 binning은 두 단계로 나뉩니다.

### 2.1 Manifest 단계

`build_manifests.py`의 `build_manifests()`는 각 row에 `event_window` 블록을 기록합니다.

저장되는 핵심 필드:

- `policy`
- `time_bin_us`
- `event_count_target`
- `accumulation`
- `causal_weight_power`
- `fast_causal_limit`
- `start_timestamp_us`
- `end_timestamp_us`

여기서 중요한 점은, manifest는 “이 샘플이 어떤 이벤트 창을 써야 하는지”를 고정하는 계약 파일이라는 것입니다.

### 2.2 Dataset 단계

실제 raw event slice 선택은 `src/hbtxr/data/dataset.py`에서 수행됩니다.

핵심 흐름:

1. `EVEyeHBTXRDataset.__getitem__()`
2. `_resolve_event_window()`
3. `_event_indices_for_window()`

`_resolve_event_window()`는 다음 두 입력을 병합합니다.

- manifest row에 저장된 `event_window`
- runtime config의 `event_builder` override

그 뒤 `_event_indices_for_window()`가 실제 event list에서 slice를 뽑습니다.

## 3. 지원되는 event window 정책

`_event_indices_for_window()` 기준으로 현재 지원 정책은 아래와 같습니다.

### 3.1 `fixed_count`

- 의미: `end_timestamp_us` 직전의 최근 `N`개 이벤트 선택
- 실제 선택량:
  - `E_sel = min(event_count_target, end 이전에 존재하는 이벤트 수)`

이 방식은 샘플마다 이벤트 수를 거의 일정하게 맞출 수 있어 학습 안정성이 높습니다.

### 3.2 `time_bin`

- 의미: `[end_timestamp_us - time_bin_us, end_timestamp_us)` 구간의 이벤트 선택
- 특징:
  - 시간 길이는 일정
  - 이벤트 수는 가변

즉 sparse 구간과 dense 구간의 이벤트 개수가 달라질 수 있습니다.

### 3.3 `interval_all`

- 의미: `[start_timestamp_us, end_timestamp_us]` 사이의 전체 구간 이벤트 사용
- 전제:
  - manifest row 또는 runtime override에 `start_timestamp_us`가 있어야 함

이 정책은 synthetic interpolation 또는 session-store 기반 mode2 경로에서 “이전 annotation 시점부터 현재 시점까지의 전체 이벤트”를 그대로 쓰고 싶을 때 유용합니다.

## 4. 이벤트 accumulation이 일어나는 위치

이벤트 slice가 정해진 뒤, 실제 센서 그리드 누적은 `_build_event_frame()` 내부에서 수행됩니다.

핵심 함수:

- `_accumulation_weights()`
- `_accumulate_events()`
- `_build_event_frame()`

### 4.1 가중치 생성

`_accumulation_weights()`는 선택된 이벤트 timestamp를 per-event weight로 변환합니다.

지원 모드:

- `plain`
  - 선택된 모든 이벤트 가중치 = `1`
- `causal_linear`
  - 더 최근 이벤트일수록 더 큰 weight
- `causal_linear_ori`
  - `causal_linear`와 같은 계열의 legacy 호환 경로
- `fast_causal_linear`
  - `causal_linear`과 동일한 시간 가중치를 쓰되 누적값을 saturation

기본적으로 `causal_*` 계열은 선택 창 내부에서 상대적 시간 위치를 `0~1`로 정규화하고, `causal_weight_power`를 적용해 weight를 만듭니다.

### 4.2 `fast_causal_linear` saturation

`fast_causal_linear`은 `_accumulate_events()`에서 특별 취급됩니다.

동작 방식:

- polarity를 2채널로 분리
- 각 이벤트를 순차 누적
- 해당 `(channel, y, x)` cell이 `fast_causal_limit`에 도달하면 더 이상 증가하지 않음

즉 단순 합산이 아니라 per-cell clipping을 포함한 causal event volume 구현입니다.

## 5. 공간 변환은 어디서 적용되나

이벤트는 먼저 센서 좌표계에서 누적되고, 그 다음 ROI crop/resize가 적용됩니다.

실제 순서:

1. `_build_event_frame()`
   - polarity-split sensor voxel 생성
2. `_apply_transform_to_event()`
   - `resize_policy`에 맞춰 crop/resize 또는 letterbox 수행
3. `_build_event_input()`
   - optional normalization
   - 최종 tensor packing

중요한 점:

- resize는 raw event를 뽑기 전에 적용되지 않습니다.
- 먼저 센서 해상도에서 누적한 뒤, 그 결과를 ROI transform으로 변환합니다.

## 6. 텐서 차원 변화

이벤트 입력은 아래 순서로 형태가 바뀝니다.

### 6.1 Raw event list

`events.npz` 또는 session store에서 읽은 원시 이벤트:

- `t`: `[E_total]`
- `x`: `[E_total]`
- `y`: `[E_total]`
- `p`: `[E_total]`

아직 dense tensor가 아니라 event list입니다.

### 6.2 Window selection 이후

`_event_indices_for_window()` 이후:

- `t_sel`: `[E_sel]`
- `x_sel`: `[E_sel]`
- `y_sel`: `[E_sel]`
- `p_sel`: `[E_sel]`

여기서 `E_sel`은 정책에 따라 달라집니다.

### 6.3 Weight 생성 이후

`_accumulation_weights()` 출력:

- `weights`: `[E_sel]`

여전히 list 기반 표현입니다.

### 6.4 센서 그리드 누적 이후

`_build_event_frame()` 출력:

- `voxel_sensor`: `[2, H_sensor, W_sensor]`

의미:

- 채널 `0`: 음극성 polarity
- 채널 `1`: 양극성 polarity

### 6.5 ROI transform 이후

`_apply_transform_to_event()` 출력:

- `event_resized`: `[2, 256, 256]`

이 텐서가 실제 모델 입력으로 사용됩니다.

### 6.6 DataLoader batch 이후

collate 후 active ABI:

- `event`: `[B, 2, 256, 256]`
- `frame`: `[B, 1, 256, 256]`
- `prev_state`: `[B, 6]`

즉 event와 frame은 같은 공간 해상도로 정렬된 채 backbone에 입력됩니다.

## 7. End-to-End 흐름

현재 active event path는 아래 순서입니다.

1. `build_manifests()`
   - 샘플별 `event_window` 기록
2. `EVEyeHBTXRDataset.__getitem__()`
   - row와 runtime override를 병합
3. `_event_indices_for_window()`
   - 실제 event list slice 선택
4. `_accumulation_weights()`
   - timestamp-aware weight 계산
5. `_accumulate_events()`
   - polarity-split voxel 누적
6. `_apply_transform_to_event()`
   - ROI crop/resize
7. `trainer.collate_fn()`
   - batch tensor 구성

## 8. FACET 정렬 지점과 현재 구현 차이

현재 기본값은 FACET 계열의 paper-side 방향과 잘 맞습니다.

정렬되는 부분:

- `fixed_count = 5000`
- fast-causal accumulation
- polarity-split event voxel

현재 구현의 차이:

- FACET paper는 “fast causal event volume” 개념을 설명 중심으로 제시하지만,
- HBTXR는 이를 active dataset code에서 직접 구현하고,
- `fast_causal_limit = 25.0`을 명시적 파라미터로 관리합니다.

즉 HBTXR의 event path는 offline event-image cache가 아니라 dataset runtime contract 위에서 동작하는 쪽에 더 가깝습니다.

## 점검 체크리스트

- [x] 실제 binning 함수 위치 정리
- [x] 실제 accumulation 함수 위치 정리
- [x] 지원 정책 `fixed_count / time_bin / interval_all` 반영
- [x] 지원 accumulation `plain / causal_linear / causal_linear_ori / fast_causal_linear` 반영
- [x] 센서 누적 후 ROI transform 순서 정리
- [x] 최종 모델 입력 텐서 ABI 정리
