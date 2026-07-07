# TSGSS Dataset Refinement And Guided Event Feature Progress

Last updated: 2026-04-08 KST

이 문서는 `tsgss/` 아래에서 진행된 샘플 기반 dataset refinement, frame/event alignment, visualization, 그리고 guided event feature interpolation 작업을 한 곳에 정리한 진행 기록이다.

## Summary

- 전체 공식 샘플 세션을 대상으로 `8`장 연속 프레임 샘플링, dense annotation, frame interpolation, event alignment, visualization, guided feature interpolation이 순차적으로 수행되었다.
- `raw event tuple` 자체를 인위적으로 늘리는 선형보간 path는 artifact가 크다는 점을 확인했고, 이후 작업은 `derived event feature` 중심으로 전환되었다.
- `density-driven adaptive crop`은 dataset path와 `tsgss` preview path 양쪽에 반영되었다.
- 최종적으로 `guided_event_feature_interpolation`과 `guided_event_feature_interpolation_roi_first` 두 버전의 derived feature artifact가 생성되었다.

## 1. Sample Extraction And Dense Annotation

### Scope

- 대상: 전체 유저 / 전체 세션 샘플 세트
- 세션 수: `388`
- 샘플 프레임 수: `3104`
- 세션당 연속 프레임: `8`

### Implemented Surface

- sampled raw-frame batch 구축
- Grounded-SAM eye ROI preview batch 실행
- dense annotation merge
- overlay preview / failure gallery / failure analysis 문서화

### Main Outputs

- root: `tsgss/raw_samples`
- workspace: `tsgss/workspace/roi_crop_prompted_all_sessions`
- merged annotation: `tsgss/dense_annotation`
- overlay preview: `tsgss/overlay_preview_gallery.html`
- failure gallery: `tsgss/failure_overlay_gallery.html`
- failure analysis: `tsgss/failure_analysis.md`

### Aggregate Result

- `annotated_complete`: `820`
- `annotated_eye_only`: `16`
- `annotated_state_only`: `2267`
- `pending_annotation_sources`: `1`
- annotation source counts:
  - `csv_region_zero_proxy`: `2267`
  - `groundedsam_eye_bbox`: `17`
  - `pseudo_no_csv_completion`: `783`
  - `raw_csv_positive`: `37`

Reference:
- `tsgss/dense_annotation/experiment_summary.json`

## 2. Frame Interpolation

### Implemented Surface

- sampled raw sequence 위에 synthetic frame 삽입
- backend 선택 surface 추가:
  - `linear_blend`
  - `timelens`
- 현재 생성 artifact는 `linear_blend` 기준

### Aggregate Result

- raw frames: `3104`
- interpolated frames: `2716`
- total sequence frames: `5820`
- insert policy: `fixed_insert=1`

Reference:
- `tsgss/frame_interpolation/experiment_summary.json`

## 3. Event Alignment

### Implemented Surface

- adjacent frame interval마다 fixed `1000` event alignment
- `frame_to_frame_sync=true`
- interval duration은 `20ms` 기준으로 유지
- mode 분기:
  - `subsampled`
  - `interpolated_dense`
  - `interp_single`
  - `empty_pad`

### Aggregate Result

- sessions: `388`
- intervals: `5432`
- target event count: `1000`
- zero-raw intervals: `30`
- raw event count mean: `738.2852`
- alignment mode distribution:
  - `interpolated_dense`: `4532`
  - `subsampled`: `865`
  - `interp_single`: `5`
  - `empty_pad`: `30`

### Important Finding

- 현재 alignment script의 sparse path는 `x/y/t` raw event tuple에 대한 선형보간을 사용한다.
- 이 방식은 눈 주변에 방사형 / 직선형 synthetic trace artifact를 만들 수 있다.
- 이후 작업에서 이 문제를 직접 설명하고, `raw tuple` 보간 대신 `derived event feature` interpolation으로 방향을 전환했다.

Reference:
- `scripts/build_tsgss_event_alignment.py`
- `tsgss/event_alignment/experiment_summary.json`

## 4. Visualization And Review Assets

### Generated Visualization Families

- frame-event voxel triptych
- `time-x-y` event plot PNG / GIF
- sensor-plane event preview
- raw vs interpolated side-by-side GIF
- adaptive-crop raw vs interpolated GIF
- sampled frame global preview with TSGSS numbering
- top-level `tsgss` numbered folder aliases

### Main Output Roots

- `tsgss/frame_event_visualization`
- `tsgss/frame_event_animation`
- `tsgss/frame_event_animation_xyz`
- `tsgss/event_xyz_preview`
- `tsgss/event_plane_preview`
- `tsgss/frame_event_animation_real_vs_interpolated_side_by_side_example`
- `tsgss/frame_event_animation_real_vs_interpolated_adaptive_crop_example`
- `tsgss/sampled_frame_preview`
- `tsgss/folder_order_aliases.md`

### Aggregate Visualization Summary

- voxel visualization:
  - `388` sessions
  - `5432` intervals
  - `5` temporal bins
- xyz animation:
  - `388` sessions
  - `5432` intervals
  - `mp4_enabled=false`

Reference:
- `tsgss/frame_event_visualization/experiment_summary.json`
- `tsgss/frame_event_animation_xyz/experiment_summary.json`
- `tsgss/event_plane_preview/experiment_summary.json`

## 5. Density-Adaptive Crop

### Dataset Path

- `src/hbtxr/data/dataset.py`에 `event_builder.crop_policy=density_adaptive`가 추가되었다.
- interval/event density를 이용해 ROI를 정하고, 동일 transform을 frame / mask / event에 적용한다.
- sparse event에서는 fallback reason을 남긴다.

### Validation

- dataset adaptive crop unit/regression test 추가
- 관련 pytest 통과

Affected files:

- `src/hbtxr/data/dataset.py`
- `tests/test_dataset_v3.py`
- `tests/test_target_fps_dataset_v3.py`
- `tests/test_interpolation_v3.py`

### TSGSS Preview Path

- `user01/left/session_102` 예시에 대해 adaptive crop 기반 raw vs interpolated GIF가 추가되었다.
- crop metadata는 summary와 manifest에 남긴다.

Reference:
- `tsgss/frame_event_animation_real_vs_interpolated_adaptive_crop_example/experiment_summary.json`

## 6. Guided Event Feature Interpolation

### Motivation

- raw sparse interval을 `1000`개 tuple로 직접 보간하는 현재 path는 artifact가 크다.
- production-facing dataset plan도 `raw event tuple` 인위 생성 대신 `derived event feature` interpolation만 허용한다.

### Implemented Surface

- 새 스크립트:
  - `scripts/build_tsgss_guided_event_feature_interpolation.py`
- feature type:
  - `128x96` polarity-split count map
- source labels:
  - `observed_event_feature`
  - `interpolated_event_feature`
- guidance inputs:
  - observed raw-event feature
  - start/end frame grayscale difference map
  - nearest anchor interval feature

### Aggregate Result

- sessions: `388`
- intervals: `5432`
- `observed_event_feature`: `2982`
- `interpolated_event_feature`: `2450`
- guidance modes:
  - `observed_anchor`: `1120`
  - `bilateral_linear`: `344`
  - `carry_next`: `1089`
  - `carry_prev`: `1017`
  - `no_anchor_fallback`: `1862`

Main outputs:

- `tsgss/guided_event_feature_interpolation`
- `tsgss/guided_event_feature_interpolation_quick_preview`

Reference:
- `tsgss/guided_event_feature_interpolation/experiment_summary.json`

## 7. ROI-First Guided Event Feature Interpolation

### Implemented Surface

- 같은 guided feature pipeline에 `density_adaptive` crop을 추가했다.
- interval마다 먼저 ROI를 고르고, 그 crop 안에서 observed / guided feature를 만든다.
- crop source:
  - `raw_density`
  - `frame_guidance`

### Aggregate Result

- sessions: `388`
- intervals: `5432`
- `observed_event_feature`: `2996`
- `interpolated_event_feature`: `2436`
- adaptive crop source distribution:
  - `raw_density`: `3445`
  - `frame_guidance`: `1987`

Main outputs:

- `tsgss/guided_event_feature_interpolation_roi_first`
- `tsgss/guided_event_feature_interpolation_roi_first_quick_preview`

Reference:
- `tsgss/guided_event_feature_interpolation_roi_first/experiment_summary.json`

## 8. Added Or Updated Scripts

- `scripts/run_roi_crop_prompted_preview_batch.py`
- `scripts/build_tsgss_dense_annotation.py`
- `scripts/build_tsgss_frame_interpolation.py`
- `scripts/build_tsgss_event_alignment.py`
- `scripts/build_tsgss_frame_event_visualization.py`
- `scripts/build_tsgss_frame_event_animation.py`
- `scripts/build_tsgss_event_xyz_preview.py`
- `scripts/build_tsgss_event_plane_preview.py`
- `scripts/build_tsgss_real_vs_interpolated_event_animation.py`
- `scripts/build_tsgss_sampled_frame_preview.py`
- `scripts/build_tsgss_folder_order_aliases.py`
- `scripts/build_tsgss_guided_event_feature_interpolation.py`

## 9. Key Conclusions

- `raw tuple` 선형보간은 시각적으로 직관적이지만 event artifact가 크다.
- `frame interpolation + raw event` 조합은 `raw event tuple` 합성보다 `guided derived feature` 생성에 더 적합하다.
- `ROI-first` guided feature는 full-sensor guided feature보다 눈 주변 구조를 더 집중적으로 다룰 수 있다.
- 현재 `tsgss` artifact는 dense annotation부터 ROI-first guided feature까지 한 번에 재추적 가능한 상태다.

## 10. Current Artifact Map

현재 `tsgss` top-level 산출물은 생성 순서 alias가 같이 유지된다.

예:

- `38_guided_event_feature_interpolation`
- `39_guided_event_feature_interpolation_quick_preview`
- `42_guided_event_feature_interpolation_roi_first`
- `43_guided_event_feature_interpolation_roi_first_quick_preview`

Reference:

- `tsgss/folder_order_aliases.md`
