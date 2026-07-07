# HBTXR v3.5 YOLO26 Head 및 BBox 통합 계획

Last updated: 2026-03-30 KST

Role:
- `모델 아키텍처 설계자`
- `Detection Loss 통합 엔지니어`
- `이벤트 비전 추적 연구 엔지니어`

## 요약

- 이 문서는 `YOLO26`의 검출 아이디어 가운데 현재 `HBTXR_v3_0`에 맞는 요소만 선별 반영하기 위한 `v3.5` 계획이다.
- 범위는 전체 YOLO 모델 이식이 아니다.
- 이번 계획의 범위는 아래로 제한한다.
  - `Eye ROI`에 대한 `YOLO26` 스타일 box regression / loss 도입
  - 선택적 `Pupil BBox / OBB auxiliary branch` 추가
  - 현재 `HBTXR`의 search-tracking 구조와 충돌하지 않는 head 수준 메커니즘만 반영
- 핵심 원칙은 다음과 같다.
  - 현재 `HBTXR`의 `search / event / track` 계약을 유지한다.
  - dense detector 전체를 억지로 이식하지 않는다.
  - ROI 및 pupil localization을 개선하는 `Ultralytics` 요소만 제한적으로 가져온다.
  - 단일 box detection 문제에 대해 grid/point detector를 강제하지 않는다.
  - pooled feature head를 기본으로 유지하고, dense point head는 필요 시 후속 실험으로만 다룬다.

## 현재 상태 갱신

- 구현 완료
  - `Eye ROI`용 `YOLO26` 스타일 head variant
  - `Eye ROI`용 `YOLO26` 스타일 bbox loss
  - `Eye ROI`용 `MPDIoU` bbox loss mode
  - `Eye ROI`용 single-scale point head와 `STAL` 유사 small-box positive expansion
  - `Eye ROI`용 `SOT CenterPredictor / CornerPredictor`
  - training pipeline 통합
  - `active head` YAML 선택 옵션
  - 선택되지 않은 head loss weight 자동 `0.0` 처리
  - `search/event pupil_bbox` auxiliary branch
  - `search/event pupil_obb` auxiliary branch
  - `search/event bbox_aux`의 `MPDIoU` mode
  - `search/event obb_aux`의 `MGIoU2D` mode
  - `mask_mode: bce_dice | bce | dice`
- 아직 미구현
  - `Proto26` 스타일 mask decoder 연구
- 이번 문서의 남은 active scope는 사실상 `Proto26` 계열 mask 연구와 full dense detector 미적용 원칙 정리다.
- 소스 정리:
  - `ldtho/MGIoU`는 `arXiv:2504.16443` 대응으로 `MGIoU2D` OBB loss에 반영
  - 사용자가 함께 준 `arXiv:2307.07662`는 별도 `MPDIoU` bbox loss로 분리 반영

## 문제 정의

현재 `HBTXR` head 구조와 `Ultralytics YOLO26` head 구조는 애초에 해결하려는 문제가 다르다.

현재 `HBTXR`:

- shared patch frontend
- shared transformer encoder
- pooled-token regression heads
- 하나의 dense mask decoder
- pupil supervision의 중심은 ellipse / state 회귀
- backbone은 `tokens`와 `pooled`를 함께 반환하지만, 현재 bbox 회귀 head는 `pooled`를 주로 사용

현재 `YOLO26`:

- multi-scale dense detector
- one-to-many / one-to-one detection heads
- task-aligned box assignment
- IoU 기반 box loss
- `reg_max` 제어 box regression
- segmentation / OBB / pose branch 확장 가능

이 둘을 그대로 섞으면 현재 tracker의 장점이 훼손된다.

- causal state-aware tracking
- event-conditioned residual prediction
- compact pooled search heads

따라서 이 계획은 아래를 분리해서 다룬다.

1. 지금 당장 가져올 가치가 있는 `head-level` 메커니즘
2. 지금 당장 가져올 가치가 있는 `loss-level` 메커니즘
3. 아직 가져오면 안 되는 `dense detection` 메커니즘

## 현재 HBTXR Head 인벤토리

주 기준 코드:

- `src/hbtxr/models/heads.py`
- `src/hbtxr/models/hybrid_tracker.py`
- `src/hbtxr/loss/stage.py`

### 활성 Head

- `EyeRegionHead`
  - legacy eye head
  - 출력: `search/eye`
  - shape: `[B, 5]`
  - 의미: `cx, cy, w, h, conf`
- `YOLO26EyeRegionHead`
  - 현재 구현된 eye head variant
  - 출력: `search/eye`
  - shape: `[B, 5]`
  - 의미: `cx, cy, w, h, conf`
- `PupilSearchHead`
  - 출력: `search/pupil`
  - shape: `[B, 7]`
  - 의미: `x, y, a, b, u, v, conf`
- `EventSearchHead`
  - 출력: `event/pupil`
  - shape: `[B, 7]`
  - 의미: `x, y, a, b, u, v, conf`
- `PupilTrackHead`
  - 출력: `track/pupil`
  - shape: `[B, 8]`
  - 의미: `dx, dy, dloga, dlogb, du, dv, conf, quality`
- `SearchMaskHead`
  - 출력: `search/mask_logits`
  - shape: `[B, 1, 256, 256]`
- `AuxStateHead`
  - 출력: `search/aux`, `event/aux`

### 구조 해석

- `search/eye`만 현재 구조에서 진짜 bbox 출력이다.
- `search/pupil`, `event/pupil`은 bbox detector가 아니라 ellipse / state regressor다.
- `track/pupil`은 detector가 아니라 residual propagation head다.
- `mask_head`만 현재 구조에서 dense spatial head에 해당한다.
- 따라서 `Eye ROI`는 loss 강화만으로도 상당 부분 개선 여지가 있고, dense point head는 필수 조건이 아니다.

## 검토한 Ultralytics 요소

로컬 reference 기준:

- `references/ultralytics-main/ultralytics/cfg/models/26/yolo26.yaml`
- `references/ultralytics-main/ultralytics/nn/modules/head.py`
- `references/ultralytics-main/ultralytics/nn/modules/block.py`
- `references/ultralytics-main/ultralytics/utils/loss.py`
- `references/ultralytics-main/ultralytics/utils/tal.py`

### 반영 검토 가치가 있는 요소

- `Detect`
  - 직접 drop-in 대상은 아니고 reference 구현으로만 참고
- `BboxLoss`
  - `Eye ROI` 손실 대체 후보로 유력
- `BCEWithLogitsLoss`
  - eye confidence / auxiliary confidence supervision에 직접 활용 가능
- `RotatedBboxLoss`
  - `Pupil OBB auxiliary branch`에 가장 직접적으로 연결 가능
- `v8OBBLoss.calculate_angle_loss`
  - pupil orientation 보조 supervision으로 유력
- `BCEDiceLoss`
  - mask head 실험 경로에서 참고 가치 있음
- `OBB26`
  - `Pupil OBB auxiliary branch` 설계 참고용으로 유력
- `Segment26 / Proto26`
  - 장기적으로 mask decoder 강화 아이디어로 참고 가치 있음
- `Pose26`
  - 장기 geometry / uncertainty 연구 참고용

### loss.py 기준 세부 판단

- 현재 `YOLO26` 로컬 reference는 `reg_max: 1`, `end2end: True`다.
- 따라서 일반적인 설명처럼 `DFL`이 핵심이 아니라, 실제 bbox 회귀 경로는 아래에 가깝다.
  - `CIoU`
  - normalized `ltrb L1`
  - `BCE`
  - end-to-end one-to-many / one-to-one wrapper
- 이 중 현재 `HBTXR`에 직접 반영할 가치는 아래 순서가 적절하다.
  1. `BboxLoss`의 `CIoU + ltrb L1` box supervision
  2. `BCEWithLogitsLoss`
  3. `RotatedBboxLoss + angle loss`
  4. `BCEDiceLoss`
- 반대로 당장 직접 반영하지 않는 항목:
  - `v8DetectionLoss`
  - `v8OBBLoss`
  - `E2EDetectLoss`
  - `E2ELoss`
  - `TaskAlignedAssigner`
  - full `STAL / TAL` assigner stack

### 당장 피해야 할 요소

- `TaskAlignedAssigner`
- full `STAL`
- full `Detect` one-to-many / one-to-one 학습 경로
- `YOLOEDetect`
- `YOLOESegment`
- `v10Detect`

이 요소들은 모두 현재 `HBTXR`에 없는 dense anchor / grid candidate 체계를 전제로 한다.

## Point Assignment 관련 판단

- pooled feature 하나만으로는 point assignment 기반 head를 구성할 수 없다.
- point assignment가 필요하면 `pooled`가 아니라 backbone이 반환하는 `tokens`와 `grid_size`를 사용해야 한다.
- 다만 현재 문제는 `Eye ROI` 단일 box detection이다.
- 단일 box detection에서는 grid별 detection이 필수는 아니다.
- 따라서 현재 기본 방침은 아래와 같다.
  - `Eye ROI`는 pooled head를 유지한다.
  - `YOLO26` loss만 먼저 반영한다.
  - dense point head는 실제 failure case가 누적될 때 후속 실험으로 분리한다.
- 후속 dense point head 후보는 아래 조건이 명확할 때만 검토한다.
  - pooled regression이 반복적으로 위치 ambiguity를 보임
  - ROI가 작거나 배경 clutter가 커서 spatial evidence가 필요한 경우
  - confidence calibration이 지속적으로 불안정한 경우

## 핵심 기술 판단

### 1. Eye ROI

`Eye ROI`는 가장 깔끔한 삽입 지점이다.

현재 상태:

- `YOLO26EyeRegionHead` variant 구현 완료
- `loss.eye_box_mode: yolo26_ciou` 구현 완료
- `model.heads.active` 기반 active head 선택 및 inactive loss zeroing 구현 완료

잠긴 방향:

- `EyeRegionHead` 출력 ABI `[cx, cy, w, h, conf]`는 유지
- 단일 box 문제이므로 pooled head 기반을 유지
- dense point assignment head는 이번 범위에서 도입하지 않음
- `L1 + BCE` 중심 `loss_eye`를 `YOLO26` 영감의 box loss로 교체

권장 loss 구성:

- `CIoU` box term
- normalized `ltrb` 또는 box-regression 보조항
- confidence term

즉 dense detection head를 들여오지 않고도 `YOLO26` box supervision의 장점을 가져올 수 있다. 현재 구현도 이 방향을 따른다.

### 2. Pupil

`Pupil`은 bbox-only supervision으로 바로 바꾸지 않는 것이 맞다.

권장 방향:

- 현재 primary output은 유지
  - `search/pupil`
  - `event/pupil`
- 선택적 auxiliary output 추가
  - `search/pupil_bbox`
  - `event/pupil_bbox`
  - 또는 더 나은 장기형으로
  - `search/pupil_obb`
  - `event/pupil_obb`

이유:

- 현재 pupil supervision은 단순 bbox가 아니라
  - center
  - semi-axes
  - orientation
  를 포함한다.
- 이것을 bbox-only supervision으로 대체하면 기존 tracker의 핵심 geometry를 버리게 된다.

### 3. Track Head

이번 wave에서는 `track_head`를 유지한다.

이유:

- residual state propagator이다.
- detection head가 아니다.
- `YOLO26` box regression과 의미적으로 맞지 않는다.

### 4. Mask Head

`Segment26 / Proto26` 아이디어는 기록만 해두고, phase 1 범위에는 넣지 않는다.

이유:

- 현재 `HBTXR`는 multi-scale feature pyramid가 없다.
- `Proto26`은 전혀 다른 feature topology를 전제로 한다.

## 반영 우선순위 매트릭스

| 후보 | 지금 반영 | 후속 반영 | 이번 wave 제외 | 메모 |
|---|---|---|---|---|
| `YOLO26` eye bbox loss | 완료 |  |  | pooled head 유지, loss/head 통합 완료 |
| `Eye ROI` dense point head |  | 조건부 | x | 단일 box 문제라 필수 아님 |
| `Pupil bbox auxiliary branch` | x |  |  | 첫 구현은 axis-aligned로 충분 |
| `Pupil OBB auxiliary branch` |  | x |  | 장기적으로는 이쪽이 더 적합 |
| `Proto26` 스타일 mask 강화 |  | x |  | bbox branch 이후 검토 |
| `TaskAlignedAssigner / full STAL` |  |  | x | 현재 구조와 계층이 맞지 않음 |
| full `Detect` head replacement |  |  | x | 너무 침습적 |
| `YOLOE` prompt heads |  |  | x | 현재 과제와 무관 |

## 잠금 원칙

- 전체 `HBTXR` head stack을 dense detector로 바꾸지 않는다.
- pooled feature 기반 `Eye ROI` head를 기본 경로로 유지한다.
- pupil의 ellipse / state supervision을 1차 pass에서 제거하지 않는다.
- full `TAL / STAL` assigner stack은 도입하지 않는다.
- 현재 single-scale point eye head에는 `STAL` 유사 small-box positive expansion만 유지한다.
- `track_head`를 bbox detector로 바꾸지 않는다.
- 현재 stage 분리와 deployment 복잡도를 유지한다.

## 구현 단계

## Phase 1. Eye ROI Loss / Head 업그레이드

### 목표

`EyeRegionHead` supervision 경로를 `YOLO26` 스타일 box regression / loss로 업그레이드한다.

### 범위

- 유지
  - pooled-feature 기반 `search/eye` ABI
  - 출력 ABI
  - inference output contract
- 변경
  - `YOLO26EyeRegionHead` variant 추가
  - `loss_eye`
  - 필요 시 eye target conversion helper

### 변경 가능성이 높은 파일

- `src/hbtxr/models/heads.py`
- `src/hbtxr/loss/common.py`
- `src/hbtxr/loss/primitives.py`
- `src/hbtxr/loss/stage.py`
- `src/hbtxr/models/hybrid_tracker.py`
- `src/hbtxr/models/pruning.py`
- `src/hbtxr/training/trainer.py`
- `configs/*.yaml`

### 완료 기준

- `search/eye` 출력 ABI 유지
- `Stage1` 학습 정상 동작
- 기존 checkpoint load 경로 회귀 없음
- active head 선택 시 eye-only 경로도 정상 동작

### 현재 상태

- 구현 완료
- 현재 기본 경로:
  - `model.heads.eye_variant: yolo26_bbox`
  - `loss.eye_box_mode: yolo26_ciou`
  - `model.heads.active: all`

## Phase 2. Pupil BBox Auxiliary Branch

### 목표

현재 ellipse / state branch를 primary로 유지한 채, pupil localization용 보조 bbox branch를 추가한다.

### 범위

- 신규 optional head
  - `PupilBBoxHead`
  - 또는 `PupilOBBHead`
- 신규 output
  - `search/pupil_bbox`
  - `event/pupil_bbox`
- 신규 target
  - 현재 pupil ellipse annotation에서 파생

### 권장 1차 구현

- ellipse에서 유도한 axis-aligned bbox
- 이유
  - target conversion이 단순
  - validation이 빠름

### 권장 2차 구현

- rotated bbox / `xywht`
- 이유
  - 현재 pupil orientation supervision과 더 잘 맞음

### 완료 기준

- 기존 `search/pupil`, `event/pupil` loss 유지
- bbox auxiliary를 config로 on/off 가능
- terminal / history는 계속 stage-aware하게 유지

## Phase 3. OBB 업그레이드

### 목표

phase 2의 bbox auxiliary가 유효하면, axis-aligned bbox를 oriented box로 승격한다.

### 범위

- angle output 추가
- `OBB26` inspired angle branch 반영
- 아래 비교 실험 수행
  - axis-aligned bbox auxiliary
  - rotated bbox auxiliary

### 완료 기준

- rotated branch가 pupil localization을 개선하거나 최소한 안정화
- track-stage primary metric 회귀 없음

## Phase 4. Optional Mask Decoder Research

### 목표

`Proto26` 스타일 decoder 아이디어가 `search/mask_logits`를 개선하는지 연구한다.

### 범위

- 연구 전용
- 명확한 근거 없이는 baseline 대체 금지

### 완료 기준

- `Stage1` mask 품질 개선
- search localization 안정성 저하 없음

## 제안 config surface

후보 config 항목:

- `model.heads.active`
  - `all | eye | search | event | track | mask | aux`
- `model.heads.eye_variant`
  - `legacy`
  - `yolo26_bbox`
- `model.heads.pupil_bbox`
- `model.heads.pupil_obb`
- `loss.eye_box_mode`
  - `legacy_l1`
  - `yolo26_ciou`
  - `yolo26_mpdiou`
- `loss.eye_box_iou_weight`
- `loss.eye_box_l1_weight`
- `loss.search_bbox_aux_mode`
  - `ciou`
  - `mpdiou`
- `loss.event_bbox_aux_mode`
  - `ciou`
  - `mpdiou`
- `loss.search_obb_aux_mode`
  - `rotated_bbox`
  - `mgiou2d`
- `loss.event_obb_aux_mode`
  - `rotated_bbox`
  - `mgiou2d`
- `loss.mgiou_fast_mode`

현재 구현된 추가 계약:

- `active head`가 선택되면 선택되지 않은 head는 비활성화한다.
- 선택되지 않은 head의 loss weight는 자동 `0.0`으로 정규화한다.
- 비활성 head의 loss / metric은 terminal 및 history에서 숨긴다.

## 이번 wave의 비목표

- full `YOLO26` model port
- transformer backbone을 `Ultralytics` backbone block으로 교체
- end-to-end dual-head detection training path 도입
- dense point assignment head 기본 경로화
- 현재 구조에서 `STAL` 도입
- full `STAL / TAL` assigner stack 직접 도입
- track residual prediction을 bbox prediction으로 대체

## 검증 계획

### 필수 점검

- bbox target conversion unit test
- 신규 head output shape / ABI test
- stage-specific loss smoke test
- 신규 head가 없는 기존 config backward-compatibility test
- active head normalization test
- inactive head loss zeroing test
- training smoke
  - `mode1_stage1`
  - `mode1_stage2`

### 봐야 할 metric

- `metric_search_p10_pct`
- `metric_search_p5_pct`
- `metric_search_center_px`
- `metric_track_p10_pct`
- `metric_track_center_px`
- 필요 시 신규 eye bbox diagnostic
- pupil bbox auxiliary diagnostic

## 리스크

- bbox auxiliary가 ellipse geometry supervision과 경쟁할 수 있음
- rotated bbox target conversion이 closed-eye / 저품질 라벨에서 noisy할 수 있음
- eye box loss 변경이 Stage1 수렴 성질을 바꿀 수 있음
- `YOLO26` box semantics를 너무 그대로 가져오면 object-detection 전제를 과도하게 끌고 올 수 있음

## 최종 권고

권장 구현 순서:

1. `Eye ROI` loss / head 업그레이드 완료 상태 유지
2. `Pupil bbox auxiliary` head
3. `Pupil OBB auxiliary` head
4. 선택적 `Proto26` inspired mask 연구
5. dense point head는 필요 시 별도 실험 문서로 분리

이번 wave에서 제외할 항목:

- full `Detect` replacement
- point assignment 기반 dense eye detector의 기본화
- `TaskAlignedAssigner`
- `STAL`
- `YOLOE`

즉, 현재 event-tracking 아키텍처를 깨지 않으면서 ROI와 pupil localization만 개선하는 통제된 `YOLO26` 통합 경로를 채택한다.
