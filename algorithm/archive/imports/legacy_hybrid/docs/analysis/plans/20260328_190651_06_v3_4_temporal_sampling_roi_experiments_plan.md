# HBTXR v3.4 Temporal Sampling 및 ROI Detection 실험 계획

Last updated: 2026-03-28 KST

Role:
- `이벤트 비전 시스템 아키텍트`
- `데이터셋 / 어노테이션 엔지니어`
- `Detection 및 Tracking 연구 엔지니어`

## 요약

- 이 문서는 `HBTXR_v3_0`의 다음 실험 wave를 정의한다.
- 목표는 active baseline을 즉시 바꾸지 않고 아래 항목을 평가하는 것이다.
  - temporal sampling
  - annotation-space policy
  - event voxel scaling
  - value-range normalization
  - ROI / pupil detection head
- 제안 아이디어는 아래 세 부류로 나눈다.
  - `baseline-safe`
  - `controlled ablation`
  - `research-only`
- 핵심 원칙:
  - 한 번에 하나의 지배 변수만 바꾼다.
  - 각 ablation 동안 data contract, training stage, evaluation protocol은 고정한다.

## 문제 정의

현재 논의 중인 질문은 서로 다른 다섯 설계 계층에 섞여 있다.

1. temporal sample 정의
2. frame-event synchronization
3. annotation coordinate policy
4. event representation scale / normalization
5. ROI / pupil detector head 설계

이것들을 동시에 바꾸면 어떤 정확도 향상이나 저하가 어디서 왔는지 해석하기 어렵다.

따라서 이 문서는 아이디어를 독립적인 experiment track으로 분리한다.

## 검토할 source 아이디어

1. 특정 `TimeBin`을 정의하고, 그 안에 원하는 event count를 맞춘 뒤 시간상 가장 가까운 frame을 synchronized frame으로 사용
2. 원본 데이터를 먼저 resize하고 그 뒤 annotation 적용
3. frame interpolation을 `1 ms` 간격으로 생성
4. event voxel을 더 작은 resolution으로 축소
5. eye-region ROI bbox와 pupil bbox에 YOLO-style bbox regression / loss 재사용
6. event accumulation 대신 `1 ms` 간격 event value interpolation을 수행하되, 중간 frame 수는 제한

## 구현 전 평가 판정

### 우선 반영 가치가 높음

- `event voxel min/max 및 분포 점검`
- `YOLO-style ROI bbox head for eye region`
- `ROI crop 이후 smaller event voxel resolution`

### 제약 정리 후 반영

- `TimeBin` 기반 sampling
- `frame-event synchronization policy`
- `annotation transform policy`
- `auxiliary pupil bbox branch`

### 연구 전용

- `1 ms frame interpolation`
- `event accumulation` 대신 event interpolation
- 고정 개수 intermediate frame 삽입

## 잠금 실험 원칙

- 첫 ablation 결과를 보기 전까지 current active baseline은 수정하지 않는다.
- `fixed_time_bin`과 `fixed_event_count`를 같은 baseline에서 동시에 hard constraint로 두지 않는다.
- authoritative source of truth를 resized coordinate 기반으로 직접 생성하지 않는다.
- causal event accumulation보다 낫다는 증거 없이 synthetic event interpolation을 baseline path로 올리지 않는다.
- `mode1`은 baseline으로 유지하고, synthetic-time 실험은 `mode2` 또는 전용 experiment preset으로 분리한다.

## 고정할 baseline

이번 계획에서 frozen reference baseline은 아래와 같다.

- `mode1`
- causal frame sync
  - timestamp가 `<= bin_end`인 최신 frame 사용
- event window
  - fixed `time_bin`
  - event count는 관측 metadata로만 저장
- annotation authority
  - 원본 sensor / canonical source coordinate
- event representation
  - 현재 event voxel builder 유지
- pupil target
  - ellipse / state prediction이 primary target 유지

로그에서는 이 baseline을 아래 태그로 기록한다.

- `exp_baseline_timebin_causal_rawanno`

## 실험 트랙

## Track A. Temporal Sample Definition

### 질문

시간축에서 한 training sample을 어떻게 정의할 것인가?

### A0. Frozen Baseline

- 정책
  - fixed `time_bin_us`
  - event count는 observed metadata만 저장
  - causal frame sync

### A1. Fixed Time Bin Variants

- 변경 변수
  - `time_bin_us`
- 후보 값
  - `500 us`
  - `1000 us`
  - `2000 us`
  - `5000 us`

### A2. Fixed Event Count Variants

- 변경 변수
  - `event_count_target`
- 후보 값
  - `1000`
  - `2000`
  - `5000`
  - `10000`

### A3. Hybrid Guardrail Variant

- 정책
  - primary = fixed `time_bin_us`
  - secondary = min/max event-count acceptance range
- 후보 정책
  - event count가 너무 작으면 reject 또는 low-quality 표시
  - event count가 너무 크면 clip 또는 subsample

### 결정 규칙

- `A0`, `A1`, `A2`, `A3`를 분리 비교한다.
- baseline에서는 모든 time bin 안에 exact event count를 억지로 강제하지 않는다.
- `A2`가 명확히 이기면, 기존 baseline을 조용히 대체하지 말고 dedicated baseline branch로 승격한다.

## Track B. Frame-Event Synchronization Policy

### 질문

어떤 frame이 event window를 대표하거나 supervise해야 하는가?

### B0. Causal Sync

- 정책
  - `frame_ts <= bin_end`를 만족하는 최신 frame

### B1. Nearest Sync

- 정책
  - `bin_end`에 가장 가까운 frame

### B2. Previous-And-Current Pair Sync

- 정책
  - nearest previous frame를 main frame으로 유지
  - next frame은 auxiliary provenance로만 기록

### 결정 규칙

- `B0`가 runtime-safe 기본 정책이다.
- `B1`은 offline analysis에서 future leakage를 허용할 때만 연구용으로 사용한다.
- causal tracking 결과는 반드시 `B0`를 사용한다.

## Track C. Annotation Coordinate Policy

### 질문

annotation을 어느 단계에서 적용해야 하는가?

### C0. Raw-Space Annotation

- 정책
  - raw 또는 canonical source resolution에서 annotation
  - resize는 downstream transform으로 적용

### C1. Resized-Space Annotation

- 정책
  - 먼저 resize
  - resized image에 직접 annotation

### 결정 규칙

- `C0`가 authoritative path이다.
- `C1`은 annotation 비용 절감 목적일 때만 평가 가능하며 final label truth가 되어서는 안 된다.
- `C1`을 쓰더라도 label은 source coordinate로 역추적 가능해야 한다.

## Track D. Event Voxel Scale And Value Range

### 질문

어떤 event resolution과 normalization을 사용해야 하는가?

### D0. Baseline Resolution

- 현재 event voxel resolution 유지

### D1. ROI-First Smaller Voxel

- 정책
  - eye ROI를 먼저 crop
  - event voxel 생성
  - voxel을 downsample

### D2. Direct Global Smaller Voxel

- 정책
  - ROI-specific crop보다 earlier stage에서 downsample

### 반드시 기록할 통계

normalization을 고르기 전에 아래를 로그로 남긴다.

- global min / max
- `p1 / p99`
- mean / std
- nonzero ratio
- positive / negative polarity count
- window event-count distribution

### 결정 규칙

- `D2`보다 `D1`을 우선한다.
- 단순히 더 싸다는 이유만으로 smaller voxel을 고르지 않는다.
- 정확도 하락이 제한적이거나 정확도 개선이 있을 때만 채택한다.

## Track E. ROI And Pupil Detection Head

### 질문

YOLO-style box regression / loss를 재사용할 것인가?

### E0. Current Head

- 현재 `HBTXR` supervision path 유지

### E1. Eye ROI BBox Head

- 아래에 YOLO-style bbox regression / loss 적용
  - eye region ROI bbox

### E2. Eye ROI + Auxiliary Pupil BBox Head

- 아래에 YOLO-style bbox regression / loss 적용
  - eye region ROI bbox
  - auxiliary pupil bbox
- 단, pupil primary target은 ellipse / state 유지

### E3. BBox-Only Pupil Head

- pupil을 bbox만으로 예측

### 결정 규칙

- `E1`이 가장 유망한 near-term 실험이다.
- `E2`는 multi-task ablation으로 허용 가능하다.
- `E3`는 ellipse / state prediction을 대체하는 main path가 되어서는 안 된다.

## Track F. Synthetic Frame / Event Generation

### 질문

synthetic temporal densification을 `1 ms`까지 밀어붙일 것인가?

### F0. No New Densification

- 현재 경로 유지

### F1. Target-FPS Frame Interpolation

- 평가 대상
  - `500 FPS`
  - `1000 FPS`
  - `2000 FPS`

### F2. Strict `1 ms` Frame Grid

- 대략 `1000 FPS`
- synthetic frame 품질과 storage cost를 같이 측정해야 함

### F3. Event Interpolation

- accumulated real event 대신 event value 또는 pseudo-event tensor를 interpolation

### 결정 규칙

- `F1`은 `mode2` 연구에서 허용 가능
- `F2`는 storage/runtime 근거 없이는 채택하지 않는다.
- `F3`는 가장 낮은 우선순위이며 연구 전용으로 남긴다.

## 권장 실행 순서

1. `D`
   - voxel range 및 normalization 통계 점검
2. `E1`
   - YOLO-style eye ROI bbox ablation 추가
3. `A + B`
   - time-bin 및 sync policy 비교
4. `D1`
   - ROI-first smaller voxel
5. `E2`
   - auxiliary pupil bbox branch
6. `F1`
   - `mode2`에서 target-FPS synthetic frame 실험
7. `F3`
   - 위 항목 측정 이후 event interpolation 검토

## 지표

각 실험은 아래를 반드시 기록한다.

- pupil center error
- 가능하면 pupil ellipse error
- ROI bbox IoU
- enabled된 경우 pupil bbox IoU
- tracking offset error
- train throughput
- eval latency
- GPU memory
- event-count distribution summary
- voxel value distribution summary

## 로깅 계약

각 experiment run은 아래를 저장해야 한다.

- baseline tag
- changed variable name
- changed variable value
- data mode
- stage
- frame sync policy
- event window policy
- event voxel resolution
- normalization policy
- annotation coordinate policy
- detector head policy

권장 experiment-name 패턴:

- `exp_<track>_<main_variable>_<value>_<timestamp>`

예시:

- `exp_A_timebin_1000us_<timestamp>`
- `exp_B_sync_causal_<timestamp>`
- `exp_D_voxel_roi96_<timestamp>`
- `exp_E_roi_yolo_head_<timestamp>`

## 채택 기준

변경을 baseline candidate로 승격하려면 아래를 만족해야 한다.

- 최소 두 번 이상 rerun에서도 개선이 반복
- hidden future leakage 없음
- runtime / storage cost 수용 가능
- annotation-space inconsistency 때문에 생긴 가짜 개선이 아님

아래 경우는 reject 또는 postpone한다.

- 두 개 이상의 major track을 동시에 바꿔야 하는 경우
- synthetic setting에서만 좋아지고 causal runtime setting에서 나빠지는 경우
- 보조 metric은 좋아져도 primary pupil task를 깨뜨리는 경우

## 즉시 다음 액션

- event voxel range statistics용 dataset-analysis utility 추가
- `Track A / B / D / E`의 exact baseline config 정의
- dedicated experiment log template 준비
- `1 ms` synthetic interpolation은 ablation 근거가 나오기 전까지 baseline에서 제외

## 현재 판단 스냅샷

- baseline candidate
  - `fixed_time_bin + causal_frame_sync + raw-space_annotation + ROI-first smaller voxel + optional ROI bbox head`
- 아직 baseline candidate가 아님
  - `nearest-frame sync`
  - `annotation on resized images`
  - `strict 1 ms synthetic interpolation`
  - `event-value interpolation`
  - `bbox-only pupil prediction`

## 이 문서 범위 밖

- code implementation detail
- manuscript wording
- JETCAS table 작성
- publication용 최종 claim 문구

이 문서는 계획 문서일 뿐이며 구현 문서는 아니다.
