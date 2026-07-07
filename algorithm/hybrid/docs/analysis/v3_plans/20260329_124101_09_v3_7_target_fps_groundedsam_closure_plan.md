# Target-FPS + Grounded-SAM 마감 계획

Last updated: 2026-03-28 KST

## 요약

이 문서는 과거 plan snapshot만이 아니라 **현재 checkout된 코드 상태**를 기준으로 남아 있는 `target_fps + Grounded-SAM` 과제를 정리한다.

핵심 결론:

- `target_fps` builder / canonical bridge / manifest bridge / dataset runtime 경로는 이미 상당 부분 구현되어 있다.
- 현재 가장 큰 남은 공백은 storage bridge 자체가 아니라 **annotation source closure**이다.
- 가장 중요한 아키텍처 결정은 live `Grounded-SAM` inference를 `build_target_fps_dataset()` 안에 직접 넣을지, 아니면 별도의 upstream producer 단계로 분리할지이다.

권장 방향:

- `target_fps_build.py`는 precomputed annotation store를 **소비하고 융합하는 consumer / fuser**로 유지한다.
- builder가 이미 읽을 수 있는 `Grounded-SAM` store를 생성하는 explicit producer 단계를 별도로 추가한다.

## 1. 현재 구현 상태에서 확정된 부분

아래 항목은 source와 test에서 이미 확인된다.

- target timestamp grid 생성
- target-FPS frame rendering
- rebinned existing-event packet 저장
- `npz`, `h5` session-store writer
- status / reason 필드가 포함된 frame-level label row
- target-FPS canonical bridge
- target-FPS manifest bridge
- session store 기반 dataset loading
- synthetic `mode1`, `mode2`에 대한 `npz` 회귀 커버리지

주 코드 경로:

- `src/hbtxr/preprocess/target_fps_build.py`
- `src/hbtxr/preprocess/target_fps_canonical.py`
- `src/hbtxr/preprocess/build_manifests.py`
- `src/hbtxr/data/dataset.py`

주 회귀 테스트:

- `tests/test_target_fps_build_v3.py`
- `tests/test_target_fps_canonical_v3.py`
- `tests/test_target_fps_manifest_v3.py`
- `tests/test_target_fps_dataset_v3.py`

## 2. 코드 레벨에서 남아 있는 TODO 신호

현재 미완료 작업은 `target_fps_build.py`가 직접 만드는 row status / reason에서 명확히 드러난다.

### A. Eye ROI source 누락

관측되는 status / reason:

- `label_status = pending_annotation_sources`
- `annotation_reason = groundedsam_eye_roi_missing`

조건:

- manual CSV pupil state는 존재
- 하지만 Grounded-SAM eye ROI source는 없음

해석:

- pupil state만으로 eye ROI를 역보정하지 않도록 의도적으로 막아둔 상태다.
- 즉 빠른 row-level patch가 아니라, 실제 eye-ROI producer를 붙여야 이 경로가 닫힌다.

### B. No-CSV eye pipeline 미완료

관측되는 status / reason:

- `label_status = pending_annotation_sources`
- `annotation_reason = groundedsam_no_csv_eye_mask_pipeline_pending`

조건:

- manual CSV 없음
- usable한 Grounded-SAM eye source도 없음

해석:

- 현재 남아 있는 Grounded-SAM production gap 중 가장 명확한 항목이다.

### C. No-CSV pupil pipeline 미완료

관측되는 status / reason:

- `label_status = annotated_eye_only`
- `annotation_reason = groundedsam_no_csv_pupil_pipeline_pending`

조건:

- eye ROI는 `groundedsam_eye_bbox_only` 또는 fallback full store를 통해 존재
- 하지만 ROI-local pupil source는 아직 production path로 닫히지 않음

해석:

- `all-user ROI-crop v2` 작업을 실제 production label path에 다시 연결해야 하는 정확한 지점이다.

### D. H5 경로는 구현됐지만 테스트 closure는 부족

코드에서 확인된 사실:

- `target_fps_build.py`에 `_write_h5_session_store()` 존재
- `data/dataset.py`에 H5 reading 경로 존재
- `requirements.txt`에 `h5py` 포함

남은 공백:

- 현재 테스트는 주로 `npz` 경로를 많이 밟는다.
- H5 bridge / runtime 커버리지는 여전히 얕거나 거의 없다.

## 3. 핵심 아키텍처 결정

### 권장안

live `Grounded-SAM` runtime을 기본 경로로 `build_target_fps_dataset()` 안에 직접 집어넣지 않는다.

대신 아래 경계를 유지한다.

1. `build_target_fps_dataset()`의 책임
   - target grid 생성
   - frame / event materialization
   - source fusion
   - 명시적인 pending / failure bookkeeping
2. 별도의 reusable Grounded-SAM producer 추가
   - `annotation_root/sessions/.../frame_annotations.jsonl`
   - `annotation_root/eye_region_bbox_only/sessions/.../eye_region_bboxes.jsonl`
3. `target_fps_build.py`는 producer가 만든 store를 소비만 하도록 유지

### 이 경계가 더 나은 이유

- 재현성이 좋아진다.
- 고비용 vision inference를 dataset materialization에서 분리할 수 있다.
- 현재 코드 구조와 잘 맞는다.
- 같은 annotation store를 여러 번 재사용할 수 있다.

## 4. 마감 작업 패키지

## P0. 문서 현실 동기화

목표:

- 현재 코드 상태가 progress 문서에 정확히 반영되도록 정리

작업:

- 이미 구현된 `target_fps` bridge 항목을 progress 문서에서 완료로 표시
- 남은 항목은 annotation-production 및 validation bucket으로 재배치
- all-48 ROI-crop handoff 문서를 production pupil-path reference로 연결

대상 파일:

- `docs/chat/20260326_011438_01_v3_update_history.md`
- `docs/chat/20260326_011438_02_v3_progress_checklist.md`
- `docs/chat/20260326_011438_03_v3_conversation_and_decision_log.md`

완료 기준:

- 전체 `target_fps` stack이 아직 미완료인 것처럼 보이지 않음

## P1. Eye ROI producer 통합

목표:

- `groundedsam_eye_roi_missing` 제거
- Grounded-SAM runtime이 가능한 환경에서 `groundedsam_no_csv_eye_mask_pipeline_pending` 제거

현재 재사용 가능한 surface:

- `src/hbtxr/preprocess/groundedsam_eye_region_bbox.py`

필요 작업:

- bbox-only preview path에서 reusable한 non-preview producer API 추출
- 아래 canonical store 경로로 쓰도록 정리
  - `annotation_root/eye_region_bbox_only/sessions/.../eye_region_bboxes.jsonl`
- preview가 아니라 batch materialization용 CLI entrypoint 추가
- 출력 schema를 `target_fps_build.py`가 이미 소비하는 형태에 맞춤

권장 신규 surface:

- `src/hbtxr/preprocess/groundedsam_eye_bbox_store.py`
- `scripts/build_groundedsam_eye_bbox_store.py`

완료 기준:

- valid runtime / checkpoint가 있으면 runnable session에서 eye-missing row가 더 이상 생성되지 않음

## P2. ROI-local pupil producer 통합

목표:

- `groundedsam_no_csv_pupil_pipeline_pending` 제거
- 검증된 ROI-crop `v2` operating point를 production target-FPS label path에 연결

현재 참조 소스:

- `scripts/run_roi_crop_prompted_preview_batch.py`
- `docs/exps/20260330_170419_14_groundedsam_roi_crop_prompted_experiment_summary.md`
- `docs/exps/20260330_170419_15_all48_v2_baseline_failure_review.md`
- `docs/exps/20260330_170419_16_all48_v2_handoff.md`

필요 작업:

- preview / batch script에서 crop-local pupil inference 로직 추출
- 입력을 아래로 표준화
  - frame image
  - eye ROI bbox
  - Grounded-SAM runtime
  - 기본값은 `v2` option config
- `target_fps_build.py`가 `groundedsam_full`로 소비할 수 있는 stable store format으로 저장
- geometry flag와 blink metadata propagation 유지

권장 신규 surface:

- `src/hbtxr/preprocess/groundedsam_roi_crop_pupil.py`
- `scripts/build_groundedsam_roi_crop_pupil_store.py`

중요 정책:

- `v2`를 기본 strict production path로 유지
- fallback은 기본값으로 켜지 않음
- large mask 같은 geometry warning은 조용히 삼키지 않고 명시적으로 남김

완료 기준:

- no-CSV session이 valid eye ROI와 valid runtime / checkpoint를 가지면 `annotated_eye_only`에서 `annotated_complete`까지 진행 가능

## P3. Target-FPS builder orchestration

목표:

- annotation production 단계를 clean sequence로 실행할 수 있게 정리

필요 작업:

- 아래 순서를 호출하는 orchestration script 추가
  1. eye bbox store build
  2. 필요 시 ROI-local pupil store build
  3. target-FPS dataset build
  4. target-FPS canonical bridge
  5. manifest build
- 각 단계를 restartable / cache-friendly 하게 유지

권장 신규 surface:

- `scripts/build_target_fps_groundedsam_pipeline.py`

비목표:

- `build_target_fps_dataset.py`를 monolithic live-inference script로 바꾸지 않음

완료 기준:

- raw data + Grounded-SAM runtime dependency만으로 target-FPS label을 재생성하는 문서화된 명령 시퀀스가 존재

## P4. H5 검증 마감

목표:

- storage format 검증 공백 해소

필요 작업:

- H5 중심 테스트 추가
  - target-FPS session store materialization
  - canonical bridge metadata
  - manifest row
  - `session.h5` 기반 dataset loading
- `session_h5_path` propagation 확인
- H5 layout이 dataset reader 기대와 일치하는지 점검

주 파일:

- `tests/test_target_fps_build_v3.py`
- `tests/test_target_fps_canonical_v3.py`
- `tests/test_target_fps_manifest_v3.py`
- `tests/test_target_fps_dataset_v3.py`

완료 기준:

- 같은 synthetic target-FPS 경로가 `npz`, `h5` 모두에서 통과

## P5. 실데이터 검증

목표:

- synthetic fixture를 넘어 실제 세션에서도 integrated path가 동작하는지 확인

필요 작업:

- 최소 1개 이상의 small real-session target-FPS materialization 실행
- 아래 경우를 각각 검증
  - manual CSV 존재
  - no CSV + bbox-only
  - no CSV + full ROI-local pupil path
- 결과 `annotation_failures.jsonl` 점검
- 남은 failure가 진짜 source absence 또는 의도된 geometry rejection인지 확인

완료 기준:

- 지원되는 경로에서는 unexplained pipeline-pending case가 사라지고, 남은 failure가 해석 가능해야 함

## 5. 권장 실행 순서

1. `P0` 문서 동기화
2. `P1` eye ROI producer 추출
3. `P2` ROI-local pupil producer 추출
4. `P3` orchestration script
5. `P4` H5 회귀 커버리지
6. `P5` 실데이터 검증

이 순서를 권장하는 이유:

- 실제 blocker는 producer closure이다.
- H5 validation은 중요하지만 annotation source completion보다 후순위다.
- 실데이터 검증은 production path가 구조적으로 닫힌 뒤 수행해야 한다.

## 6. 완료 정의

이 `target_fps + Grounded-SAM` 마감 트랙은 아래가 모두 참일 때 완료로 본다.

- `target_fps` 관련 문서가 실제 구현 상태와 일치
- eye ROI production이 reusable store-building step으로 제공
- ROI-local pupil production이 reusable store-building step으로 제공
- target-FPS fusion이 이 store들을 ad hoc 개입 없이 소비 가능
- `H5`, `NPZ` 모두 synthetic regression coverage 확보
- remaining non-complete row가 진짜 data/runtime absence이며, 미연결 integration 작업이 아님

## 7. 즉시 다음 단계

다음 시작점은 `P1`이다. 추가적인 target-FPS bridge refactoring이 아니다.

이유:

- bridge / fusion path는 이미 존재한다.
- 가장 큰 기능 공백은 Grounded-SAM producer integration이다.
- eye ROI production을 먼저 닫아야 manual-CSV와 no-CSV target-FPS closure 양쪽이 동시에 풀린다.
