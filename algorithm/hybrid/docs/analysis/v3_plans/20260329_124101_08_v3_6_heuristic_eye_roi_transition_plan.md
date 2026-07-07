# 휴리스틱 Eye ROI 전환 계획

Last updated: 2026-03-26 KST

이 문서는 `Grounded-SAM` 기반 eye ROI annotation 실험 이후, `HBTXR_v3_0`를 휴리스틱 기반 `Eye Region ROI` 추출 경로로 전환하기 위한 현재 계획을 정리한다.

## 전환이 필요한 이유

- `Grounded-SAM`은 `user01/left` 샘플에서 `full-frame` 또는 과대 `ROI`가 반복적으로 발생했다.
- prompt를 `v1 -> v2 -> v3`로 개선하고 `mask_bbox_area_ratio` 필터를 추가해도, `session_201`처럼 어려운 세션에서는 안정성이 부족했다.
- dense annotation 생산 경로에서는 프레임 drop 없이 모든 프레임에 대해 `ROI bbox`를 남기는 것이 우선이다.

## 현재 결정

- `Grounded-SAM`은 `eye ROI GT` 또는 dense annotation 메인 생산 경로로 사용하지 않는다.
- 대신 raw frame만으로 동작하는 휴리스틱 `Eye Region ROI Detection` 경로를 active 후보로 구축한다.
- dense 모드에서는 모든 프레임에 대해 row를 유지하고, `accepted / interpolated / propagated / fallback` 상태를 함께 기록한다.

## 현재 프로젝트 안의 baseline

프로젝트에 이미 들어 있는 heuristic ROI는 `raw-only detector`가 아니다.

- 현재 구현 위치
  - `src/hbtxr/preprocess/io_utils.py`
  - `derive_eye_region()`
- 현재 성격
  - manual ellipse annotation들을 세션 단위로 집계하여 stable `eye_region`을 생성하는 규칙 기반 crop generator
- 현재 한계
  - raw frame만 보고 `eye ROI`를 찾지 않는다.
  - per-frame detector가 아니다.
  - dense raw annotation 생성기 역할을 하지 못한다.

즉, 기존 heuristic은 재사용 가치는 있지만 목표가 다르다.

## 목표 출력 계약

heuristic dense 경로의 frame row는 최소한 아래 필드를 가져야 한다.

```json
{
  "frame_filename": "...",
  "frame_idx": 123,
  "timestamp_us": 123456789,
  "eye_region_bbox_xywh_sensor": [x, y, w, h],
  "annotation_accepted": true,
  "annotation_status": "accepted",
  "annotation_source": "heuristic_eye_roi",
  "heuristic_score": 0.83,
  "used_temporal_prior": true,
  "used_fallback": false
}
```

실패 프레임도 row는 유지한다.

```json
{
  "frame_filename": "...",
  "frame_idx": 124,
  "eye_region_bbox_xywh_sensor": [x, y, w, h],
  "annotation_accepted": false,
  "annotation_status": "propagated",
  "annotation_source": "heuristic_eye_roi"
}
```

## 제안 파이프라인

### 1. 전처리

- grayscale normalize
- weak blur 또는 median blur
- `CLAHE` 같은 contrast boost
- 필요 시 center prior crop

### 2. 점수 맵(score map)

아래 요소를 결합해 `eye likelihood map`을 만든다.

- darkness prior
- horizontal elongation prior
- edge density prior
- center prior
- previous bbox 기반 temporal prior

### 3. 후보 추출

- threshold 또는 top-k region selection
- morphology open / close
- connected components
- component scoring 기준
  - area
  - aspect ratio
  - center distance
  - temporal consistency

### 4. 프레임 bbox 생성

- 최고 점수 component -> bbox
- margin expand
- boundary clamp
- min / max size constrain

### 5. 시간축 안정화

- EMA smoothing
- sudden jump suppression
- short-gap interpolation
- previous-bbox propagation

### 6. Dense 보장

프레임 drop 금지 원칙을 유지한다.

- `accepted`
- `interpolated`
- `propagated`
- `fallback`

즉 bbox는 항상 존재하되 품질 상태가 다르게 기록된다.

## 구현 단계

### Step 1. 문서화와 baseline 감사

- [x] `Grounded-SAM` eye ROI 실험 결과 정리
- [x] 현재 프로젝트 안의 기존 heuristic ROI (`derive_eye_region`) 구조 재정리
- [x] dense annotation에서는 프레임을 버리면 안 된다는 정책 명확화

### Step 2. Raw-only heuristic detector 프로토타입

- [ ] `src/hbtxr/preprocess/heuristic_eye_roi.py` 추가
- [ ] score map / candidate extraction / bbox selection 구현
- [ ] raw frame 한 장에 대한 standalone detector 함수 구현

### Step 3. Dense session builder

- [ ] `scripts/annotate_heuristic_eye_roi.py` 추가
- [ ] 모든 프레임에 대해 `frame_annotations.jsonl` row 유지
- [ ] `accepted / propagated / fallback` 상태 기록

### Step 4. 시각화

- [ ] raw + bbox overlay
- [ ] raw + score map overlay
- [ ] session contact sheet export

### Step 5. 검증

- [ ] `user01/left` 네 세션 랜덤 샘플 검토
- [ ] `101/102/201/202` 세션별 accepted ratio 비교
- [ ] `Grounded-SAM v3 dense` 결과와 휴리스틱 결과 비교

## 작업 원칙

- dense annotation에서는 프레임 drop 금지
- `Grounded-SAM`은 비교용 보조 경로로만 유지
- heuristic output은 `GT`가 아니라 `auto ROI`로 명시
- raw-only detector와 annotation-derived session ROI를 혼동하지 않는다

## 즉시 다음 단계

다음 구현 우선순위는 아래와 같다.

1. `heuristic_eye_roi.py` 프로토타입 생성
2. raw frame 기반 bbox-only overlay preview 생성
3. `keep-all-frames` JSONL contract와 연결
