# Tracker Refactor Checklist

## 목표

- `HBTXRTracker`에서 인코더, search branch, 추후 track/runtime policy를 분리한다.
- loss 옵션마다 head를 늘리지 않고, 출력 계약 기준으로 구조를 나눈다.
- 현재 출력 key와 state dict 사용성을 최대한 유지한다.
- `stage.py`를 facade로 유지하면서 `stage1.py`, `stage2.py`, `metrics.py`로 orchestration을 최종 분리한다.
- `dataset -> model -> loss` 계약을 repo-tracked JSON snapshot으로 고정한다.

## 체크리스트

### 1. 출력 계약 고정

- [x] `search/eye`, `search/pupil`, `search/pupil_bbox`, `search/mask_logits`, `track/pupil`를 핵심 계약으로 유지한다.
- [x] dense eye, ROI bbox, mask cascade는 기존 출력 key를 유지한 채 내부 구현만 분리한다.
- [x] 분리 후 shape/output 계약을 문서로 별도 정리한다.

### 2. tracker 하위 모듈 도입

- [x] `src/hbtxr/models/tracker/` 패키지를 만든다.
- [x] `encoder.py`를 추가한다.
- [x] `search_branch.py`를 추가한다.
- [x] `track_branch.py`를 추가한다.
- [x] `runtime_policy.py`를 추가한다.
- [x] `head_factory.py`를 추가한다.
- [x] `event_branch.py`를 추가한다.

### 3. 인코더 분리

- [x] patch frontend / adapter / backbone / pruning width logic를 `TrackerTokenEncoder`로 이동한다.
- [x] `encode_frame()`, `encode_event()`, patch cache 관련 로직을 helper class로 위임한다.
- [x] `map_pretrained_state_dict()`의 patch-embed 매핑 로직을 인코더 helper로 위임한다.
- [x] track 경로를 `TrackStateBranch`로 위임한다.
- [x] `decode_track_state()` 자체를 tracker 외부 `TrackStateCodec`로 이동한다.

### 4. search branch 분리

- [x] eye head / search head / ROI bbox head / mask head / aux head를 `SearchBranch`에서 조립한다.
- [x] mask centroid / mask cascade는 `SearchMaskGuidanceRefiner`로 분리한다.
- [x] `forward_search()`는 encoder + branch 위임 구조로 바꾼다.
- [x] dense eye / ROI bbox / legacy eye variant 생성 분기를 `TrackerHeadFactory`로 이동한다.

### 5. track branch 분리

- [x] `prev_state_encoder`와 `track_head`를 `TrackStateBranch` 위임 구조로 연결한다.
- [x] `forward_track()`를 branch 위임 구조로 바꾼다.
- [x] `decode_track_state()`를 branch 외부 `TrackStateCodec` helper로 이동한다.

### 6. event/runtime 분리

- [x] `forward_event()`를 `EventStateBranch`로 위임한다.
- [x] `runtime_step()`의 scheduler/cache 처리를 `RuntimeStepPolicy`로 분리한다.

### 7. loss bundle 분리

- [x] eye loss bundle
- [x] pupil state loss bundle
- [x] pupil bbox loss bundle
- [x] mask loss bundle
- [x] track loss bundle
- [x] `stage.py` facade 유지
- [x] `stage1.py` 분리
- [x] `stage2.py` 분리
- [x] `metrics.py` 분리
- [x] `stage_common.py` 분리

### 8. shape / output 계약 고정

- [x] `tests/snapshots/output_contracts_v3.json` 추가
- [x] `tests/test_output_contract_snapshot_v3.py` 추가
- [x] mode1 dataset sample contract snapshot 고정
- [x] mode2 synthetic sample contract snapshot 고정
- [x] core tracker output contract snapshot 고정
- [x] extended tracker output contract snapshot 고정
- [x] stage1/stage2 loss log contract snapshot 고정

### 9. dataset 분리

- [x] `SessionFrameEventReader`
- [x] `SpatialTransformResolver`
- [x] `EventInputBuilder`
- [x] `AdaptiveRoiResolver`
- [x] `AnnotationTargetBuilder`
- [x] `SampleAssembler`
- [x] `EVEyeHBTXRDataset.__getitem__()`를 orchestration 중심으로 축소

### 10. YAML 연결

- [x] `data.components`로 dataset split component variant를 선택할 수 있다.
- [x] `model.components`로 tracker split component variant를 선택할 수 있다.
- [x] dataset/model selector를 테스트로 고정한다.
- [x] `base.yaml`에 components selector 예시를 반영한다.

### 11. 검증

- [x] Python 문법 검사
- [x] model smoke test
- [x] dataset smoke test
- [x] train pipeline smoke test
- [x] loss regression test
- [x] loss stage facade/split 동일성 테스트
- [x] output contract snapshot 테스트
- [x] 전체 pytest 통과

## 최근 검증 결과

- [x] 루트 가상환경 `.venv`를 `uv venv .venv`로 생성
- [x] `uv pip install --python .venv/bin/python -r requirements.txt -e .`로 의존성 설치
- [x] `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_model_v3.py tests/test_loss_catalog_v3.py tests/test_train_pipeline_v3.py`
- [x] `PYTHONPATH=src .venv/bin/python -m pytest -q`
- [x] `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_dataset_v3.py tests/test_model_v3.py`
- [x] `HBTXR_UPDATE_SNAPSHOTS=1 PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_output_contract_snapshot_v3.py`
- [x] `PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_output_contract_snapshot_v3.py tests/test_loss_stage_module_split_v3.py tests/test_dataset_v3.py tests/test_model_v3.py tests/test_loss_catalog_v3.py`
- [x] 전체 테스트 결과: `100 passed`

## 누적 반영 범위

- `TrackerTokenEncoder` 도입
- `SearchBranch` 도입
- `SearchMaskGuidanceRefiner` 도입
- `HBTXRTracker.forward_search()` 위임 구조로 변경
- `TrackStateBranch` 도입
- `TrackerHeadFactory` 도입
- `EventStateBranch` 도입
- `RuntimeStepPolicy` 도입
- `TrackStateCodec` 도입
- `loss/bundles` 패키지 도입
- `eye.py`, `search_event.py`, `track.py`, `shared.py` bundle 분리
- `stage1.py`, `stage2.py`, `metrics.py`, `stage_common.py` 분리
- `SessionFrameEventReader`, `SpatialTransformResolver`, `EventInputBuilder` 도입
- `AdaptiveRoiResolver`, `AnnotationTargetBuilder`, `SampleAssembler` 도입
- `data.components`, `model.components` YAML selector 연결
- `base.yaml` selector 예시 및 output contract 문서 추가
- `tests/snapshots/output_contracts_v3.json` ABI snapshot 추가
- `tests/test_output_contract_snapshot_v3.py`, `tests/test_loss_stage_module_split_v3.py` 추가
- `.venv` 구성 및 전체 pytest 통과 확인
