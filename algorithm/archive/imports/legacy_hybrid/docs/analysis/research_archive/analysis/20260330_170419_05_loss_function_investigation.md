# HBTXR 손실 함수 검토 및 설계 메모

최종 갱신: 2026-03-30 KST

Role: `하이브리드 추적 손실 설계자`, `이벤트 비전 연구자`, `학습 안정화 엔지니어`

## 요약

이 문서는 “현재 코드에 이미 구현된 손실 함수 중 무엇이 실제 active path인지”, “어떤 손실은 실험용 후보로 남아 있는지”, “향후 YOLO26 box loss나 pupil auxiliary bbox/OBB 실험을 할 때 무엇을 재사용할 수 있는지”를 정리한 검토 문서입니다.

핵심 결론:

- 현재 HBTXR의 기본 방향은 옳습니다.
- Frame / Event / Track / Consistency 역할 분담도 명확합니다.
- 다만 eye ROI는 아직 단순 bbox regression 중심이라, 향후 box loss 업그레이드 여지가 큽니다.
- `loss_primitives.py`에 있는 많은 손실은 후보 라이브러리이며, 현재 학습의 주 손실은 `loss_stage.py` 경로입니다.

## 1. 현재 active 손실 체계의 구조

현재 active 학습은 대략 아래 구조를 따릅니다.

\[
L_{\text{total}} =
L_{\text{stage}}
+ L_{\text{distillation}}
+ L_{\text{regularization\_ssl}}
\]

여기서 `L_stage`는 실제 Stage1/Stage2 supervised loss이고, 나머지는 opt-in 확장입니다.

### 1.1 Stage1

Stage1의 핵심 목적:

- eye ROI 추정
- frame 기반 pupil absolute state 추정
- mask regularization

대응 손실:

- `loss_eye`
- `loss_mask`
- `loss_search_xy`
- `loss_search_ab`
- `loss_search_trig`
- `loss_search_geo`
- `loss_search_conf`
- `loss_constraint_center`
- `loss_aux`

### 1.2 Stage2

Stage2의 핵심 목적:

- search anchor 유지
- event-only branch 보강
- residual track 정교화
- drift 억제

대응 손실:

- `loss_search_*`
- `loss_event_*`
- `loss_track_*`
- `loss_consistency`
- `loss_constraint_center`
- `loss_aux`

## 2. 현재 설계가 좋은 이유

### 2.1 상태 성분별 분해가 명확함

현재 pupil state는 하나의 벡터지만 loss는 성분별로 분리됩니다.

- `xy`: 중심 위치
- `ab`: 장축/단축 크기
- `trig`: 방향
- `geo`: ellipse 전체 기하
- `conf`: 신뢰도

이 분해는 다음 장점이 있습니다.

- 각 성분이 어떤 이유로 깨졌는지 로그로 추적 가능
- 특정 성분만 가중치 조정 가능
- track/search/event branch를 동일 패턴으로 비교 가능

### 2.2 ellipse-aware supervision이 포함돼 있음

단순 L1만 쓰는 구조가 아니라 `ellipse_gwd_loss`가 포함됩니다.

즉 현재 손실은:

- 위치만 맞추는 수준이 아니라
- ellipse의 shape와 orientation까지 반영하는 쪽입니다.

이 점은 pupil geometry 과제와 잘 맞습니다.

### 2.3 stage-aware masking 계약이 일관됨

active path는 다음 마스크를 일관되게 사용합니다.

- `annotation_quality`
- `closed_eye_flag`
- `mask_valid`
- `valid_track`

즉 품질이 낮거나 의미 없는 샘플을 무작정 loss에 넣지 않습니다.

## 3. 현재 active path에서 부족한 점

### 3.1 eye ROI는 아직 detector-style box loss가 아님

현재 `loss_eye`는 본질적으로:

- bbox 좌표: Smooth L1
- confidence: sigmoid BCE

구조입니다.

이건 단순하고 안정적이지만, 다음과 같은 detector-style 개선 여지가 있습니다.

- `CIoULoss`
- `GIoULoss`
- YOLO26 계열 box regression

즉 eye ROI는 이후 가장 먼저 업그레이드해볼 만한 손실 대상입니다.

### 3.2 event density는 아직 active weighting으로 직접 쓰이지 않음

`event_density`는 dataset ABI에 존재하고 primitive로 `EventDensityWeighting`도 있지만, 현재 stage loss에서는 직접 사용되지 않습니다.

따라서 event-rich / event-sparse 샘플에 대해 가중치를 다르게 주는 설계는 아직 실험 surface에 머물러 있습니다.

### 3.3 primitive library와 active trainer 사이에 간극이 있음

예를 들어 다음 손실들은 코드에 있지만, 현재 기본 trainer path에서 직접 쓰이지 않습니다.

- `AdaptiveWingLoss`
- `CIoULoss`
- `GIoULoss`
- `CornerLoss`
- `NLLHeatmapLoss`
- `BinaryFocalLoss`
- `EventToFrameContrastiveLoss`
- `GazeAngularLoss`

즉 “존재한다”와 “현재 기본 학습에 쓰인다”를 구분해서 봐야 합니다.

## 4. 손실을 목적별로 다시 분류하면

### 4.1 Search / absolute estimation

현재 가장 잘 맞는 손실:

- `smooth_l1_with_mask`
- `trig_l2_loss`
- `ellipse_gwd_loss`
- `sigmoid_bce_with_mask`

이 조합은 `search/pupil`에 적합합니다.

### 4.2 Event / transition support

현재 event branch는 absolute state를 다시 예측하지만, 해석상으론 “frame anchor를 보강하는 branch”에 가깝습니다.

그래서 적합한 손실은:

- search와 같은 absolute loss 조합
- 또는 향후 `TemporalDisplacementLoss`, `DisplacementLoss` 같은 transition-centric loss 실험

입니다.

### 4.3 Track / residual propagation

현재 residual track branch에는 다음 조합이 합리적입니다.

- residual 자체에 대한 Smooth L1
- decode 후 ellipse GWD
- conf / quality supervision
- consistency loss

이 구조는 “이전 상태에서 현재 상태로 얼마나 안정적으로 전파했는가”를 직접 봅니다.

### 4.4 Hybrid consistency

현재 `loss_consistency`는 search state와 track state 사이의 괴리를 억제합니다.

이는 event tracking의 drift를 frame anchor로 제어하는 가장 중요한 항목 중 하나입니다.

즉 HBTXR의 hybrid 성격은 단순히 frame loss와 event loss를 더하는 데 있지 않고, consistency 항목에 있습니다.

## 5. 향후 반영 가치가 높은 손실 후보

### 5.1 Eye ROI용 box loss

우선순위가 가장 높습니다.

후보:

- `CIoULoss`
- `GIoULoss`
- 이후 YOLO26-style box regression / confidence 설계

적용 위치:

- `search/eye`

### 5.2 Pupil bbox / OBB auxiliary

기존 pupil state를 버리는 게 아니라 auxiliary branch에 붙이는 쪽이 맞습니다.

후보 primitive:

- `CIoULoss`
- `GIoULoss`
- `TrigRotationLoss`
- `EllipseOverlapLoss`

즉 axis-aligned bbox보다 rotated box 또는 ellipse-aware auxiliary가 더 자연스럽습니다.

### 5.3 Cross-modal regularization

이미 코드에 있는 후보:

- `EventToFrameContrastiveLoss`
- `FeatureConsistencyLoss`
- `PredictionKDLoss`
- `RelationalKDLoss`

이는 optimizer / data contract가 안정된 뒤 representation quality를 끌어올릴 때 검토할 가치가 있습니다.

## 6. 현재 기준 비추천 또는 후순위 항목

### 6.1 full heatmap 기반 재설계

`CenterHeatmapLoss`, `NLLHeatmapLoss`, `CornerLoss`는 의미는 있지만, 현재 HBTXR head ABI는 pooled regression 구조입니다.

즉 heatmap detector로 바꾸려면 head 자체를 재설계해야 합니다.

그래서 손실만 먼저 넣는 방식은 적합하지 않습니다.

### 6.2 gaze angle 중심 손실

`GazeAngularLoss`는 시선 벡터 회귀에 가까운 손실이라, 현재 pupil ellipse tracking의 1차 목적과는 거리가 있습니다.

PoG 또는 gaze calibration 계층이 강화될 때 검토하는 편이 맞습니다.

## 7. 실무 추천

현재 구조를 기준으로 추천 순서는 아래입니다.

1. `search/eye`에 detector-style bbox loss 적용
2. `search/event pupil`은 기존 state loss 유지
3. `pupil_bbox` 또는 `pupil_obb` auxiliary branch 추가
4. 이후 contrastive / KD / density weighting 같은 확장 실험 수행

즉 “현재 active stage loss를 버리고 완전히 새 손실로 갈아엎기”보다, 현 구조를 유지한 채 눈에 띄는 약한 지점을 강화하는 편이 맞습니다.

## 8. 결론

현재 HBTXR의 손실 설계는 다음 점에서 타당합니다.

- stage별 역할이 분리돼 있음
- ellipse geometry를 직접 반영함
- consistency 항목으로 hybrid drift를 억제함
- distillation / SSL 확장 surface가 이미 있음

가장 큰 개선 여지는:

- eye ROI의 bbox loss 업그레이드
- pupil auxiliary bbox / OBB 도입
- event density weighting의 실제 활성화

입니다.

## 점검 체크리스트

- [x] 현재 active loss와 primitive 후보 loss 구분
- [x] stage1 / stage2 손실 역할 재정리
- [x] 현재 설계의 장점과 한계 정리
- [x] 향후 반영 가치가 높은 후보 손실 선별
- [x] 비추천 / 후순위 손실 방향 구분
