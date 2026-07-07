# All-48 샘플링 데이터셋 구성 실험

이 문서는 최근 ROI-crop follow-up 실행에서 사용한 것과 동일한 all-48 sampled frame slice를 재사용한 sampled dataset-construction 실험을 정리한다.

## 1. 실험 범위

대상 slice:

- `48 users`
- `4 official sessions`
- `2 eyes`
- `8 sampled frames per eye-session`
- total sampled frames
  - `3072`

주요 목표:

1. 모든 sampled frame에 Eye Region ROI annotation 부여
2. 가능한 경우 raw CSV supervision으로 pupil 관련 annotation 생성
3. 여전히 pupil geometry가 없는 frame만 분리해서 분석

## 2. Grounded-SAM 기준 옵션

이 실험에서 사용한 정확한 Grounded-SAM 기준 옵션은 아래에 저장되어 있다.

- `configs/groundedsam/all48_sampled_dataset_construction_groundedsam_reference_v1.json`

중요 해석:

- 이 실험은 이전 `v10` ROI-crop 실행에서 검증된 eye-stage 계약을 그대로 재사용했다.
- 따라서 eye ROI의 소스는 다음과 같다.
  - Grounded-SAM eye-region bbox detection
- 이 실험이 저장한 eye ROI mask는 다음 의미다.
  - 최종 eye ROI bbox를 rasterize해서 만든 mask
- pupil geometry는 Grounded-SAM이 아니라 raw CSV ellipse supervision에서 생성했다.

## 3. 출력 루트

주 출력 루트:

- `workspace_session_samples_10/all48_dataset_construction_same8samples`

주요 산출물:

- sampled manifest
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/sample_manifest.jsonl`
- merged frame-level annotation table
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/merged_frame_annotations.jsonl`
- experiment summary
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/experiment_summary.json`
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/experiment_summary.md`
- missing-only analysis
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/analysis/missing_only_records.jsonl`
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/analysis/missing_only_summary.json`
  - `workspace_session_samples_10/all48_dataset_construction_same8samples/analysis/missing_only_summary.md`

## 4. 집계 결과

최종 결과:

- sessions: `384`
- sampled_frames: `3072`
- eye_roi_present: `3071`
- eye_roi_missing: `1`
- pupil_geometry_complete: `16`
- pupil_state_only: `2288`
- pupil_related_missing: `768`

CSV status 분해:

- `csv_positive = 16`
- `csv_region_zero = 2288`
- `csv_file_missing = 768`

최종 label-status 분해:

- `annotated_complete = 16`
- `annotated_state_only = 2287`
- `annotated_pupil_state_only_eye_missing = 1`
- `annotated_eye_only = 768`

## 5. 핵심 해석

### Eye ROI coverage

- sampled slice 기준 eye ROI annotation은 거의 완전하다.
  - `3071 / 3072`
- 유일한 eye-ROI miss는 다음 사유였다.
  - `rejected_large_box(area_ratio=0.993)`

즉 현재 sampled construction 결과의 병목은 eye ROI detection이 아니다.

### Pupil geometry coverage

- full raw-CSV pupil geometry를 가진 sampled frame은 `16`개뿐이었다.
- 이 frame들은 다음 상태가 됐다.
  - `annotated_complete`

### State-only frame

- `2288` frame은 다음으로 분류됐다.
  - `csv_region_zero`
- 이 frame들은 다음 상태로 저장됐다.
  - `pupil_state_only`

실무적 의미:

- 이 frame들은 state 정보는 유용하게 담고 있다.
- 하지만 pupil mask / bbox geometry는 현재 포함하지 않는다.

### Missing-only frame

- `768` frame은 pupil geometry 없이 남았다.
- missing-only 분석 결과, `768`개 전부의 원인은 다음이었다.
  - `raw_csv_file_missing`

지배적인 패턴:

- eye ROI는 존재
- raw CSV pupil source만 없음

즉 현재 남아 있는 결손의 주된 원인은 eye ROI 실패가 아니라 supervision source 부재다.

## 6. 운영적 결론

이 실험의 결론은 다음과 같다.

- sampled all-48 slice에 대해 eye ROI는 사실상 완전하게 복구할 수 있다.
- pupil geometry coverage는 raw CSV availability에 거의 전적으로 지배된다.
- 따라서 다음 우선순위는 eye ROI detector 개선이 아니라, pupil geometry source policy를 어떻게 가져갈지 정하는 것이다.

권장 방향:

1. `annotated_complete`
   - 그대로 supervised geometry row로 사용
2. `annotated_state_only`
   - state supervision 전용 row로 유지
3. `annotated_eye_only`
   - blink / close-eye / geometry-missing policy와 연결해서 별도 취급

## 7. 권장 다음 단계

- sampled all-48 slice를 production-facing manifest 정책과 연결
- `annotated_state_only`를 어떤 loss와 어떤 head에 연결할지 명시
- `annotated_eye_only` 행을 close-eye / unavailable geometry 상태로 분류할지 결정
- Grounded-SAM eye ROI와 raw CSV pupil supervision을 섞은 현재 계약을 문서화하고 target-FPS label path에 반영
