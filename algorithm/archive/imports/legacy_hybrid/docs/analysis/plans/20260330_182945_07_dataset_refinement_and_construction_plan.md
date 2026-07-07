# Dataset Refinement And Construction Plan

Last updated: 2026-03-30 KST

Role:
- `데이터셋 아키텍트`
- `이벤트 비전 데이터 엔지니어`
- `학습 파이프라인 계약 감사자`

## 요약

이 문서는 sampled all-48 실험, no-CSV completion follow-up, event-policy 논의를 거친 뒤 `HBTXR_v3_0`에 권장하는 **production-facing dataset refinement 및 construction 계획**을 정리한다.

핵심 원칙:

- `raw frame`, `raw event`, `annotation`, `derived event feature`를 분리 저장한다.
- canonical event-time coordinate는 `fixed_time_bin` 개념을 사용하되, 현재 코드/설정 용어로는 `time_bin`으로 매핑한다.
- `fixed_event_count`는 canonical storage rule이 아니라 각 time bin 내부의 input sampling policy로만 사용한다.
- `fail_like` row는 dataset에서 삭제하지 않고, track-anchor construction에서만 제외한다.

## 현재 코드와의 용어 매핑

이 문서는 production 설계 관점의 용어를 사용한다. 현재 `HBTXR_v3_0` 코드와 설정에서의 대응 용어는 아래와 같다.

| 설계 문서 용어 | 현재 코드 / YAML 용어 | 비고 |
|---|---|---|
| `fixed_time_bin` | `time_bin` | canonical event policy |
| `fixed_event_count` | `fixed_count` | input construction policy |
| `causal_prev_frame` | `prev_frame_idx` 중심 causal sync | 학습/런타임 기본 |
| `nearest_frame_idx` | `nearest_frame_idx` | 분석/시각화 보조 저장 |

즉 이 문서의 기본 정책은 현재 코드 기준으로 아래와 같이 해석한다.

- canonical event policy: `time_bin`
- in-bin event sampling policy: `fixed_count`

## 1. Canonical Contracts

### A. Canonical event policy

권장 canonical event policy:

- `fixed_time_bin`

현재 코드 매핑:

- `time_bin`

해석:

- dataset은 안정적인 time grid로 index된다.
- 각 event sample row는 명시적인 `bin_start_us`, `bin_end_us`를 가진다.
- frame sync는 event-count window가 아니라 time grid 기준으로 정의된다.

중요한 보정:

- `fixed_event_count`는 여전히 허용한다.
- 다만 각 time bin 내부의 **derived sampling policy**로만 쓴다.
- second canonical storage rule로 쓰지 않는다.

### B. Frame sync policy

권장 기본값:

- training / runtime tracking
  - `causal_prev_frame`
- offline analysis / visualization
  - `nearest_frame_idx`도 함께 저장

이유:

- `nearest_frame`는 causal tracking setup에서 future-frame leakage를 만들 수 있다.
- 두 참조를 모두 저장하면 canonical store의 유연성을 유지하면서도 학습 시 causal 기본값을 지킬 수 있다.

### C. Provenance policy

모든 annotation과 derived event representation은 source를 보존해야 한다.

최소 source family:

- `raw_csv_positive`
- `csv_region_zero_proxy`
- `groundedsam_eye_bbox`
- `pseudo_no_csv_completion`
- `observed_event_feature`
- `interpolated_event_feature`

## 2. Annotation Construction

### Step 1. 공식 데이터 전체에 대한 Eye ROI annotation

`48 users x 4 official sessions x 2 eyes` 전체에 대해 아래를 구축한다.

- `eye_region_bbox_xywh_sensor`
- `eye_region_mask_path`
- `eye_region_source`

권장 정책:

- Grounded-SAM eye bbox를 primary eye ROI signal로 사용한다.
- bbox를 first-class stable source로 취급한다.
- 이후 더 강한 eye-mask producer가 검증되기 전까지는, 검증된 eye bbox 경로에서 dataset-contract eye ROI mask를 생성한다.

### Step 2. labeled frame에 대한 Raw-CSV pupil annotation

신뢰 가능한 Raw-CSV geometry가 있는 frame에 대해 아래를 구축한다.

- `pupil_state_flag`
- `pupil_mask_path`
- `pupil_bbox_xywh_sensor`
- `pupil_ellipse_xywht_sensor`

중요한 구분:

- `csv_region_count > 0`
  - full geometry
- `csv_region_count == 0`
  - state-only proxy

### Step 3. missing-frame pupil completion

신뢰 가능한 Raw-CSV pupil geometry가 없는 frame에 대해서는 eye ROI 내부에서 pupil completion을 수행한다.

생성 산출물:

- `pupil_mask_path`
- `pupil_bbox_xywh_sensor`
- `pupil_ellipse_xywht_sensor`
- derived `pupil_state_flag`

중요한 계약:

- 이 row를 Raw-CSV label과 동등 confidence로 취급하지 않는다.
- 아래 필드로 pseudo label임을 명시한다.
  - `annotation_source`
  - `annotation_reason`
  - `label_status`

## 3. Quality Flags

모든 frame-level annotation row는 최소한 아래 필드를 노출해야 한다.

- `annotation_quality`
- `pupil_mask_valid`
- `pupil_box_valid`
- `pupil_state_valid`
- `mask_valid`
- `valid_track`
- `fail_like`
- `review_required`

권장 해석:

- `annotation_quality`
  - search / detection supervision에 쓰는 scalar 또는 ordinal confidence
- `pupil_mask_valid`
  - pixel mask supervision 가능 여부
- `pupil_box_valid`
  - bbox / ellipse geometry 신뢰 가능 여부
- `pupil_state_valid`
  - open / closed state 신뢰 가능 여부
- `valid_track`
  - 이 row를 track anchor로 사용할 수 있는지 여부

## 4. Pair And Track Construction

### 권장 규칙

`fail_like` row를 dataset에서 삭제하지 않는다.

대신:

- row는 유지한다.
- frame / event linkage는 유지한다.
- 필요 시 `valid_track=false`로 둔다.
- pair link는 가장 가까운 이전 clean anchor를 기준으로 만든다.
  - `prev_clean_annotation_ref`
  - 또는 `prev_track_anchor_ref`

이유:

- row를 삭제하면 provenance와 review traceability가 깨진다.
- 그 frame 주변 event stream 자체는 여전히 유용하다.
- 제외해야 하는 것은 data row 전체가 아니라 track anchor 자격이다.

추가 권고:

- clean-to-clean gap이 설정된 threshold를 넘으면 `valid_track=false`를 강제한다.

## 5. Event-Time Grid And In-Bin Sampling

### Step 4. Canonical time grid

각 session에 대해 아래를 정의한다.

- `time_bin_us`
- `bin_index`
- `bin_start_us`
- `bin_end_us`

per-bin metadata:

- `event_start_idx`
- `event_end_idx`
- `n_events_raw`
- `prev_frame_idx`
- `nearest_frame_idx`

### Step 5. In-bin fixed-event-count sampling

각 canonical time bin 내부에서 model input용으로 최대 `K`개 event를 sampling한다.

권장 정책:

- `n_events_raw > K`
  - deterministic subsampling
- `n_events_raw < K`
  - zero-pad
  - `valid_event_mask` 노출

중요 규칙:

- `K`를 채우기 위해 raw event를 인위적으로 생성하지 않는다.
- 여기서의 `fixed_event_count`는 input tensor construction rule이지 raw event stream 재정의가 아니다.

## 6. Interpolation And Derived Event Features

### Step 6. Feature-level interpolation only

`0.5 ms` 같은 더 촘촘한 time resolution이 필요할 때도 **derived event feature만 interpolation**한다.

금지:

- raw event tuple을 실제 관측 event처럼 interpolation하는 것

허용되는 derived target:

- event voxel feature
- polarity-split count map
- 기타 dense event feature grid

### Step 7. Causal linear generation

직접 관측된 feature 값이 없는 timestamp에 대해, 필요할 때만 `causal_linear` derived feature를 생성한다.

필수 safeguard:

- 결과를 raw event data와 분리 저장한다.
- 아래 메타를 기록한다.
  - `event_feature_source`
  - `interp_gap_us`
  - `event_feature_valid`

권장 source 값:

- `observed`
- `interpolated`
- `empty`

추가 권고:

- `max_interp_gap_us`를 둔다.
- temporal gap이 너무 크면 hallucination하지 말고 비워둔다.

## 7. Storage Layout

### H5

무거운 numeric array는 session 단위 H5에 저장한다.

권장 내용:

- frame tensor 또는 frame path / frame index table
- raw event array
- canonical time-grid metadata
- per-bin derived event feature
- sampled event tensor와 `valid_event_mask`

### JSON / JSONL

annotation과 bookkeeping layer는 JSON / JSONL에 저장한다.

권장 내용:

- annotation geometry
- state label
- provenance
- quality flag
- pair / track reference
- label status
- review reason

중요 저장 권고:

- low-confidence 또는 review-only heavy artifact는 가능하면 core training store 밖에 둔다.

## 8. Final Build Order

권장 실행 순서:

1. canonical policy와 naming 정의
2. eye ROI annotation 구축
3. Raw-CSV pupil geometry 융합
4. missing-frame manifest 추출
5. eye ROI 내부 pseudo pupil completion 수행
6. annotation quality와 flag 검증
7. canonical event time grid 구축
8. in-bin fixed-event-count input 생성
9. optional interpolated event feature 구축
10. clean-anchor skipping 규칙으로 pair / track link 생성
11. H5 + JSON / JSONL export
12. strict / relaxed manifest 생성

## 9. Recommended Defaults

first production pass 권장 기본값:

- canonical event policy
  - `fixed_time_bin`
- model-input sampling
  - bin 내부 `fixed_event_count`
- frame sync
  - `causal_prev_frame`
- `fail_like` handling
  - row는 유지, track anchor에서 제외
- interpolation
  - derived event feature만 허용
- pseudo-label policy
  - provenance 유지
  - Raw-CSV geometry보다 낮은 confidence 부여

## 10. Success Criteria

아래 조건을 만족하면 dataset plan을 닫는다.

- 모든 official session에 eye ROI coverage가 있다.
- Raw-CSV-positive frame에는 신뢰 가능한 pupil geometry가 있다.
- missing frame은 pseudo-labeled 또는 명시적으로 unresolved 상태다.
- 모든 row에 provenance와 quality flag가 일관되게 기록된다.
- pair / track link는 `fail_like` anchor를 피하되 row는 삭제하지 않는다.
- canonical event H5와 annotation JSON / JSONL export를 training / evaluation code가 deterministic하게 소비할 수 있다.

## 11. 구현 전 잠금이 필요한 항목

이 문서를 실제 production contract로 쓰기 전 아래를 고정해야 한다.

1. 필드 스키마
- `annotation_quality`의 scale 또는 enum
- `label_status` enum
- `annotation_reason` enum
- `review_required` 기준

2. geometry convention
- `pupil_ellipse_xywht_sensor`의 angle unit
- axis order와 rotation convention

3. manifest inclusion rule
- strict manifest 포함 조건
- relaxed manifest 포함 조건
- `fail_like`, `review_required`, `valid_track`가 각 manifest에 미치는 영향

4. track gap rule
- clean anchor gap threshold 수치
- threshold 초과 시 `valid_track=false` 처리 규칙

5. H5 / JSON source-of-truth 분리
- 어떤 필드가 H5 authoritative인지
- 어떤 필드가 JSON / JSONL authoritative인지

## 12. 최종 권고

이 계획은 production dataset contract의 방향으로 적절하다. 다만 현재 `HBTXR_v3_0` 코드에 바로 연결하려면 아래 보정이 선행돼야 한다.

- 문서 용어를 현재 YAML / code 용어와 명시적으로 매핑
- annotation / quality / manifest schema를 enum 수준으로 고정
- interpolation 및 pseudo-label confidence 정책을 수치화

즉, 이 문서는 **좋은 상위 설계안**이고, 위 잠금 항목을 채운 뒤 production-facing canonical contract로 승격하는 것이 맞다.

## 계획 대비 진행상황 체크리스트

- [x] sampled all-48 / no-CSV completion / event-policy 논의 결과를 단일 계획으로 통합
- [x] canonical event policy와 input sampling policy를 분리
- [x] fail-like row 유지 / track-anchor 제외 원칙 명시
- [x] annotation provenance 보존 원칙 명시
- [x] H5 / JSON / JSONL 역할 분리 원칙 명시
- [x] 현재 코드 기준 용어 매핑 추가
- [ ] annotation / quality flag enum 확정
- [ ] manifest strict / relaxed inclusion 규칙 확정
- [ ] track gap threshold 수치 확정
- [ ] production canonical contract로 최종 승격
