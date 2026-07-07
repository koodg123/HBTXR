# All-48 Sampled Dataset 구축 실험 계획

Last updated: 2026-03-29 KST

## 요약

이 문서는 최근 ROI-crop follow-up run에서 사용한 동일한 all-48 comparison slice 위에서 수행할 새로운 **sampled dataset-construction experiment**를 정의한다.

대상 범위:

- `48 users`
- `4 official sessions`
- `2 eyes`
- `eye-session`당 `8 sampled frames`
- 총 sampled frames
  - `3072`

요청된 실험 목표:

1. 모든 sampled frame에 대해 Grounded-SAM으로 **Eye Region ROI Bounding Box**와 **Eye Region ROI Mask** annotation
2. **Raw CSV**가 있는 frame에는 아래를 모두 annotation
   - pupil state
   - pupil mask
   - pupil bounding box
3. **pupil 관련 annotation이 아직 없는 frame만 별도 추적 및 분석**

주 권고:

- 최근 `v9 / v10 / v11 / v12` 비교에 사용했던 동일한 stable sampled frame set을 그대로 고정한다.
- 첫 구축 pass에서는 이미 검증된 **bbox-only eye ROI path**를 재사용한다.
- geometry-positive CSV row에는 기존 **raw CSV ellipse -> pupil mask / bbox / state** 경로를 재사용한다.
- 최종 분석은 아래로 분리한다.
  - fully annotated pupil geometry
  - state-only pupil labeling
  - missing pupil annotation

## 1. 범위와 제약

이 계획은 아래 실행 계약을 전제로 한다.

- server default environment
  - `conda` env `hbtxr`
- GPU rule
  - `GPU0`, `GPU1`만 사용
- sample freeze
  - all-48 preview 비교에서 쓰인 stable seed prefix 유지
  - 현재 `v2 / v9 / v10 / v11 / v12` 결과와 직접 비교 가능성을 보존

코드 기준 해석:

- **raw CSV positive ellipse row**는 full pupil geometry를 제공 가능
- **raw CSV `region_count == 0` row**는 ellipse geometry를 직접 제공하지 못함
- 따라서 이런 frame은 아래까지만 지원 가능
  - state-only labeling
- 반면 추가 정책 없이는 아래는 직접 생성 불가
  - pupil mask
  - pupil bbox

따라서 이 실험은 아래 세 상태를 분리해야 한다.

- `pupil_geometry_complete`
- `pupil_state_only`
- `pupil_annotation_missing`

## 2. 현재 재사용 가능한 코드 surface

현재 저장소에는 필요한 building block이 상당수 이미 들어 있다.

### A. Stable sampled-frame selection

관련 surface:

- `scripts/run_roi_crop_prompted_preview_batch.py`

이유:

- all-48 sampled frame subset을 stable seed로 고정하는 기능이 이미 있다.
- `v2`, `v8`, `v9`, `v10`, `v11`, `v12`에서 재현성이 검증됐다.

### B. Grounded-SAM eye ROI source

관련 surface:

- `src/hbtxr/preprocess/groundedsam_eye_region_bbox.py`
- `scripts/groundedsam_eye_region_bbox_preview.py`

이미 있는 것:

- 검증된 bbox-only eye ROI detection
- 명시적인 rejection reason
  - `no_eye_detections`
  - `rejected_large_box(...)`
  - `rejected_border_touch(...)`
  - `no_valid_eye_box`
- 아래 경로에 저장되는 per-frame bbox row
  - `annotation_root/eye_region_bbox_only/sessions/.../eye_region_bboxes.jsonl`

중요한 한계:

- 현재 검증된 reusable 경로는 **bbox-only**
- 현재 `target_fps_build.py` 계약은 resolved eye ROI bbox를 **rasterize**해서 eye ROI mask를 materialize한다.
- 즉 checked-out code 기준으로는 dedicated true eye contour mask producer가 가장 강한 검증 경로가 아니다.

### C. Raw CSV pupil source

관련 surface:

- `src/hbtxr/preprocess/io_utils.py`
- `src/hbtxr/preprocess/raw_ellipse_blink.py`
- `src/hbtxr/preprocess/target_fps_build.py`

이미 있는 것:

- `parse_via_csv_with_report(...)`
  - ellipse-positive CSV row를 파싱
- `build_raw_ellipse_blink_metadata(...)`
  - closed-eye / blink metadata를 생성
- `_resolve_pupil_annotation(...)`
  - raw CSV ellipse row를 아래로 변환
    - pupil mask
    - pupil bbox
    - pupil ellipse state

### D. Failure / missing analysis pattern

관련 surface:

- `scripts/review_roi_crop_prompted_batch.py`
- `src/hbtxr/preprocess/target_fps_build.py`

이미 있는 것:

- CSV-positive vs no-CSV breakdown
- blink-labeled grouping
- failure status / reason field
- 아래 기반의 frame-level missing-row collection
  - `label_status`
  - `annotation_reason`
  - `roi_status`
  - `pupil_status`

## 3. 이 실험의 핵심 설계 선택

### 첫 pass의 eye-ROI 계약

첫 pass에서는 아래를 사용한다.

- Grounded-SAM **eye ROI bbox**를 primary eye source로 사용
- 저장된 eye bbox를 rasterize해서 eye ROI mask 생성

이것을 첫 pass의 권장 선택으로 두는 이유:

- 아래 경로에서 이미 검증된 방식이기 때문
  - `groundedsam_eye_region_bbox.py`
  - `target_fps_build.py`
- current production-facing label contract와 맞는다.
- 아래 문서에 기록된 불안정한 direct eye-mask route를 다시 열지 않아도 된다.
  - `docs/prj/20260330_170419_03_groundedsam_dataset_pipeline.md`

이 선택의 의미:

- 초기 실험은 **Eye Region ROI Mask** artifact를 생성한다.
- 하지만 그 mask는 **resolved eye ROI bbox에서 파생한 dataset-contract mask**이다.
- 즉 새로운 dense eye contour segmentation mask를 의미하지는 않는다.

만약 true contour-style eye mask가 나중에 필요하다면, 그것은 첫 baseline이 아니라 별도 실험 branch로 다뤄야 한다.

## 4. 권장 실험 산출물

권장 root naming:

- raw sampled frames
  - `dataset_session_samples_10/raw_samples/all48_dataset_construction_same8samples`
- workspace
  - `workspace_session_samples_10/all48_dataset_construction_same8samples`
- analysis
  - `workspace_session_samples_10/all48_dataset_construction_same8samples_analysis`

권장 logical output:

1. sampled frame manifest
   - 실험에 사용된 정확한 `3072` frame 목록
2. eye ROI source store
   - per-frame eye ROI bbox row
   - rasterized eye ROI mask
3. raw-CSV pupil geometry store
   - pupil state
   - pupil mask
   - pupil bbox
   - blink / closed-eye metadata
4. merged frame-level annotation table
   - sampled frame당 1 row
5. missing-only analysis report
   - 최종 pupil geometry가 없는 frame만 별도 분석

권장 merged-row field:

- sample identity
  - `session_key`
  - `frame_filename`
  - `frame_idx`
  - `timestamp_us`
- eye ROI
  - `eye_region_bbox_xywh_sensor`
  - `eye_region_mask_path`
  - `eye_region_mask_valid`
  - `eye_region_mask_source`
- pupil
  - `pupil_ellipse_xywht_sensor`
  - `pupil_region_bbox_xywh_sensor`
  - `pupil_mask_path`
  - `mask_valid`
  - `closed_eye_flag`
  - `eye_state_flag`
- bookkeeping
  - `label_status`
  - `annotation_reason`
  - `roi_status`
  - `pupil_status`
  - `annotation_sources_used`

## 5. 작업 패키지

## P0. Exact Sample Set 고정

목표:

- 최근 all-48 ROI-crop 비교에 사용한 정확히 같은 `3072` frame을 dataset-construction experiment에서 그대로 사용

작업:

- stable sample seed prefix 재사용
  - `roi-crop-prompted-preview-v1`
- session별 또는 global sample manifest materialize
- experiment root 안에 manifest 저장

완료 기준:

- 이후 모든 annotation 및 분석 단계가 동일 sampled frame list를 기준으로 동작

## P1. 모든 sampled frame에 Eye Region ROI annotation

목표:

- sampled frame 전체에 eye ROI bbox annotation
- eye ROI mask contract materialize

권장 구현:

- sampled frame에 대해서만 validated bbox-only Grounded-SAM eye detector 실행
- 아래 저장
  - `eye_region_bboxes.jsonl`
  - 저장된 bbox를 rasterize한 eye ROI mask

출력 해석:

- `eye_region_bbox_xywh_sensor`는 Grounded-SAM eye ROI detection 결과
- `eye_region_mask_path`는 resolved eye ROI bbox를 rasterize해서 만든 dataset-contract mask

권장 metric:

- total sampled frames
- eye ROI detected frames
- eye ROI missing frames
- rejection-status count
  - `no_eye_detections`
  - `rejected_large_box`
  - `rejected_border_touch`
  - `no_valid_eye_box`

완료 기준:

- `3072` sampled frame 전부에 eye ROI result row 존재
- detection coverage와 missing reason이 명시적으로 요약됨

## P2. CSV-Labeled Frame용 Raw-CSV Pupil Annotation

목표:

- usable raw CSV pupil geometry가 있는 frame에 대해 아래 생성
  - pupil state
  - pupil mask
  - pupil bbox

권장 구현:

- raw CSV 파싱
  - `parse_via_csv_with_report(...)`
- blink / closed-eye metadata 생성
  - `build_raw_ellipse_blink_metadata(...)`
- ellipse geometry를 아래로 변환
  - rasterized pupil mask
  - pupil bbox
  - pupil state representation

중요한 분기:

### `csv_positive`

조건:

- `region_count > 0`

기대 결과:

- full pupil geometry
  - mask
  - bbox
  - ellipse state

### `csv_region_zero`

조건:

- CSV row는 있지만 positive ellipse region이 없음

기대 결과:

- pupil geometry 없음
- state-only 해석은 가능
  - `closed_eye_flag = True`
  - `eye_state_flag = closed`

이 구분은 아래와 섞이면 안 된다.

- parser failure
- missing CSV file
- unknown-frame CSV row

완료 기준:

- CSV-positive sampled frame은 전부 pupil geometry를 가짐
- CSV-nonpositive frame도 silent drop 없이 명시적으로 분류됨

## P3. Merged Sampled Annotation Table 구축

목표:

- `3072` sampled frame용 단일 실험 테이블 생성

권장 row status:

- `annotated_complete`
  - eye ROI + pupil geometry 존재
- `annotated_state_only`
  - eye ROI 존재
  - pupil state 존재
  - pupil geometry 없음
- `annotated_eye_only`
  - eye ROI 존재
  - pupil annotation 없음
- `pending_annotation_sources`
  - eye ROI 없음

`annotated_state_only`가 필요한 이유:

- `csv_region_zero` 같은 state-only case를 true missing row와 구분할 수 있다.
- closed-eye state-only frame을 총 annotation failure로 과대 집계하지 않게 해준다.

완료 기준:

- `3072` frame이 explicit status bucket으로 완전히 분할됨

## P4. Missing-Only Tracking 및 분석

목표:

- 최종 pupil geometry가 없는 frame만 별도 분석

주 분석 대상:

- merge 이후에도 pupil geometry가 없는 row

권장 분석 bucket:

1. `state_only`
   - state는 존재
   - geometry는 정책상 없음
2. `csv_missing`
   - 해당 frame/session용 CSV source 없음
3. `csv_region_zero`
   - CSV는 있으나 positive ellipse region 없음
4. `csv_parse_or_mapping_skip`
   - CSV row는 있으나 parse/match 실패
5. `eye_roi_missing`
   - eye ROI annotation 실패

권장 출력:

- aggregate JSON summary
- markdown report
- missing pupil geometry가 많은 session 순위
- reason별 count 순위
- unresolved subset 전용 contact sheet 또는 grouped overlay page

완료 기준:

- pupil geometry가 없는 모든 frame이 명시적인 reason bucket을 가짐

## 6. 권장 실행 순서

1. exact sampled frame manifest 고정
2. 그 sampled frame에 대해서만 eye ROI bbox annotation 수행
3. 저장된 eye bbox로 rasterized eye ROI mask 생성
4. raw CSV 파싱 후 CSV-positive frame의 pupil geometry 구축
5. `csv_region_zero` frame을 silent missing이 아니라 state-only로 명시
6. 단일 frame-level table로 merge
7. 최종 pupil geometry가 없는 frame만 별도 분석

## 7. 기대 산출물

이 실험이 끝나면 저장소에는 아래가 존재해야 한다.

- 재현 가능한 `3072`-frame sampled manifest
- 모든 sampled frame에 대한 eye ROI bbox annotation
- 모든 sampled frame에 대한 eye ROI mask artifact
- CSV-positive sampled frame에 대한 raw-CSV-derived pupil geometry
- merged sampled annotation table
- pupil geometry가 없는 frame 전용 missing-only analysis report

## 8. 열어둘 질문

1. 이 실험에서 **bbox-rasterized eye ROI mask**를 공식 mask contract로 받아들일 것인가?

- 첫 pass 권장 답
  - yes

2. `csv_region_zero`는 아래 중 무엇으로 셀 것인가?

- missing pupil annotation
- state-only annotation

- 권장 답
  - state-only annotation

3. final merged output은 기존 `frame_annotations.jsonl` contract를 곧바로 재사용할 것인가, 아니면 sampled experiment 전용 merged table을 먼저 둘 것인가?

- 권장 답
  - 먼저 sampled-experiment-specific merged table로 시작
  - 실험이 깨끗하게 검증되면 production contract로 bridge

## 9. 즉시 다음 단계

`v13` 같은 새 model-tuning experiment보다 먼저 할 일:

- exact sampled-frame manifest 구축
- 그 뒤 eye ROI annotation producer와 raw-CSV pupil merge를 fixed `3072`-frame slice에 실행

즉 다음 실험의 초점은 **prompt sweep**가 아니라 **dataset construction quality**가 되어야 한다.

## 10. 2026-03-29 실행 결과 스냅샷

계획했던 sampled dataset-construction flow는 아래 두 층에서 실제 실행됐다.

1. all-48 sampled slice
2. focused `session_101` repeat

### All-48 sampled slice

Construction root:

- `workspace_session_samples_10/all48_dataset_construction_same8samples`

결과:

- `sampled_frames=3072`
- `eye_roi_present=3071`
- `eye_roi_missing=1`
- `pupil_geometry_complete=16`
- `pupil_state_only=2288`
- `pupil_related_missing=768`

Missing-only analysis:

- `768` unresolved row 전부
  - `raw_csv_file_missing`

No-CSV completion follow-up root:

- `workspace_session_samples_11/all48_dataset_construction_same8samples_no_csv_completion_v10`

Completion result:

- `attempted=768`
- `completed=768`
- `completed_clean=753`
- `completed_fail_like=15`
- `unresolved=0`

### Focused `session_101` repeat

Preview root:

- `workspace_session_samples_12/roi_crop_prompted_session101_v10_same8samples`

Construction root:

- `workspace_session_samples_13/session101_dataset_construction_same8samples`

Completion root:

- `workspace_session_samples_14/session101_dataset_construction_same8samples_no_csv_completion_v10`

결과:

- preview
  - `sampled_frames=768`
  - `completed=768`
  - `fail_like=15`
- construction
  - `eye_roi_present=768`
  - `pupil_related_missing=768`
  - `csv_file_missing=768`
- completion
  - `attempted=768`
  - `completed=768`
  - `completed_clean=753`
  - `completed_fail_like=15`
  - `unresolved=0`

## 11. 계획 대비 진행상황

### `P0` Exact Sample Set 고정

상태:

- completed

근거:

- all-48 construction root가 earlier `v10` preview batch의 fixed sampled slice를 그대로 재사용
- focused `session_101` repeat도 stable sampled `8`-frame-per-eye-session 계약을 유지

### `P1` Eye Region ROI Annotation For All Sampled Frames

상태:

- sampled experiment root 기준 완료

근거:

- all-48
  - `3071 / 3072` eye ROI present
- `session_101`
  - `768 / 768` eye ROI present

### `P2` Raw-CSV Pupil Annotation For CSV-Labeled Frames

상태:

- partially completed

근거:

- all-48
  - `csv_positive=16`
  - `csv_region_zero=2288`
- `session_101`
  - usable raw CSV geometry source가 없음

해석:

- source가 있는 곳에서는 code path가 정상 동작
- 주요 한계는 pipeline collapse가 아니라 source coverage

### `P3` Build A Merged Sampled Annotation Table

상태:

- completed

근거:

- all-48 merged root 구축
- `session_101` merged root 구축

### `P4` Missing-Only Tracking And Analysis

상태:

- completed

근거:

- all-48 missing-only analysis에서 unresolved row 전부가 아래로 분류됨
  - `raw_csv_file_missing`
- `session_101` repeat도 같은 구조를 focused subset에서 재확인

### 원 계획을 넘는 후속 실행

실행된 follow-up:

- current `v10` ROI-crop pupil path 기반 no-CSV completion

결과:

- all-48과 `session_101` 모두에서 missing bucket을 완전히 닫음
- 남은 trust 이슈는 아래로 축소
  - `fail_like=15`

### 현재 권장 다음 단계

- current sampled dataset-construction output은 유지
- clean no-CSV completion row는 usable pseudo-label candidate로 취급
- `fail_like=15` subset은 제외하거나 별도 검토
- 그 다음에야 선택된 정책을 production `target_fps` label path로 bridge

## 12. 실행 후 정책 보정

completion run과 follow-up discussion 이후, sampled-plan 해석은 아래와 같이 정리한다.

- fail-like row를 저장 artifact에서 hard-delete 하지 않는다.
- event continuity를 유지하기 위해 row-level sample metadata는 남긴다.
- supervision만 gate한다.
  - `annotation_quality`로 trust를 낮춤
  - `mask_valid`, `valid_track`으로 direct mask / track supervision 비활성화
- tracking pair를 만들 때는
  - fail-like row를 trusted previous anchor로 사용하지 않음
  - pair가 fail-like 또는 skipped gap을 가로지르면 `valid_track=false`

저장 정책 보정:

- 나중에 이 로직을 H5에 넣더라도 storage burden의 대부분은 frame / event array와 optional bitmap mask에서 온다.
- fail-like flag 자체는 storage 비용이 거의 없다.
- 실용적인 production 정책은 아래에 가깝다.
  - annotation metadata는 lightweight하게 유지
  - heavy bitmap mask는 accepted supervision row 위주로 archive
