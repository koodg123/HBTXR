# Tracker Class Split Plan

## 대상

현재 가장 먼저 분할할 대상은 `HBTXRTracker`이다.

- 파일: `src/hbtxr/models/hybrid_tracker.py`
- 이유:
  - patch frontend, adapter, backbone, pruning, eye head variant, ROI bbox head, mask cascade, event/track branch, cache 처리까지 한 클래스에 몰려 있다.
  - 옵션이 늘어날수록 `__init__`와 `forward_search()`가 계속 비대해진다.
  - 새 실험 헤드를 추가할 때 기존 search/track 경로와 결합도가 너무 높다.

두 번째 후보는 `EVEyeHBTXRDataset`이다.

- 파일: `src/hbtxr/data/dataset.py`
- 이유:
  - frame/event 로딩, resize transform, adaptive crop, synthetic event 생성, annotation target 생성이 한 클래스에 같이 들어 있다.

## 권장 분할 구조

### 1. `TrackerTokenEncoder`

역할:

- `patch_frontend`
- `frame_adapter`, `event_adapter`
- `backbone`
- pruning width mask
- `encode_frame()`, `encode_event()`, `forward_track()`용 token encode 공통부

효과:

- 모델의 공통 feature extraction 경로와 head 조립 경로를 분리할 수 있다.

### 2. `SearchBranchHeads`

역할:

- eye detector 선택
- search head
- ROI bbox head
- mask head
- aux head
- search output postprocess

포함 권장:

- dense eye output
- `search/pupil_bbox`
- `mask_center`, `mask_axes`
- mask cascade 보정

효과:

- `forward_search()`의 대부분을 이 클래스로 이동할 수 있다.
- 새 head 실험이 search branch 내부에서만 닫히게 된다.

### 3. `TrackBranchHeads`

역할:

- `prev_state_encoder`
- `track_head`
- `decode_track_state()`

효과:

- track 전용 상태 해석과 residual decode를 search logic과 분리할 수 있다.

### 4. `TrackerRuntimePolicy`

역할:

- scheduler FSM
- patch warmup / pseudo-edge 사용 조건
- cache invalidation / cache prime 정책

효과:

- 모델 본체에서 runtime rule branching을 줄일 수 있다.

### 5. `TrackerHeadFactory`

역할:

- eye head variant 선택
- ROI bbox head variant 선택
- mask head variant 선택

효과:

- `__init__`에서 길게 이어지는 variant 분기문을 제거할 수 있다.

## 최소 리팩터링 순서

1. `TrackerTokenEncoder`를 먼저 분리한다.
2. `SearchBranchHeads`를 분리하고 `forward_search()`를 위임한다.
3. `TrackBranchHeads`를 분리한다.
4. `TrackerRuntimePolicy`와 `TrackerHeadFactory`를 마지막에 뺀다.
5. 최종적으로 `HBTXRTracker`는 composition만 담당하는 thin facade로 남긴다.

## 최종 형태 예시

`HBTXRTracker`

- `self.encoder: TrackerTokenEncoder`
- `self.search_branch: SearchBranchHeads`
- `self.track_branch: TrackBranchHeads`
- `self.runtime_policy: TrackerRuntimePolicy`

`HBTXRTracker.forward_search()`

- encoder 호출
- search branch 호출
- output 반환

`HBTXRTracker.forward_track()`

- encoder 호출
- track branch 호출
- output 반환

## dataset 쪽 다음 단계

`EVEyeHBTXRDataset`는 아래 4개로 나누는 것을 권장한다.

- `SessionFrameEventReader`
  - frame / event / session-store 로딩
- `SpatialTransformResolver`
  - ROI, resize policy, letterbox, adaptive crop
- `EventWindowBuilder`
  - raw window, source-pair average, target-fps session-store
- `AnnotationTargetBuilder`
  - eye target, pupil region target, mask target, state target

## 실무 기준 권고

한 번에 전부 쪼개기보다, 먼저 `HBTXRTracker`에서 search branch만 분리하는 것이 가장 효과가 크다.

- 이유:
  - 이번 병합으로 `dense eye`, `roi bbox`, `mask cascade` 옵션이 search 쪽에 더 붙었다.
  - 가장 빨리 복잡도가 줄어드는 구간이 search branch이기 때문이다.
