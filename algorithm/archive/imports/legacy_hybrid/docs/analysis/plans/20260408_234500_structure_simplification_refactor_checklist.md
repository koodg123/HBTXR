# 구조 단순화 후속 리팩터링 체크리스트

최종 갱신: 2026-04-08 KST

Role: `구조 단순화 설계자`, `리팩터링 실행 계획 작성자`

## 목적

이 문서는 현재 `HBTXR_v3_0` 구조에서 추가로 단순화 효과가 큰 후속 리팩터링 포인트를 우선순위별 체크리스트로 정리한다.

핵심 원칙은 아래와 같다.

- 이미 잘 분리된 `tracker`, `dataset`, `loss`는 무리하게 더 잘게 쪼개지 않는다.
- 대신 `중앙 허브`에 남아 있는 orchestration 과다 책임을 줄인다.
- `main surface`의 사용성은 유지하고, 내부 구현만 더 얇고 예측 가능하게 만든다.
- 리팩터링은 항상 `snapshot`, `pytest`, main CLI smoke를 통과하는 범위에서 진행한다.

## 현재 가장 큰 허브

현재 구조에서 다음 파일들이 후속 단순화의 우선 대상이다.

- [src/hbtxr/training/trainer.py](../../src/hbtxr/training/trainer.py)
  - 1589 lines
- [src/hbtxr/preprocess/canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
  - 1372 lines
- [src/hbtxr/preprocess/groundedsam_build.py](../../src/hbtxr/preprocess/groundedsam_build.py)
  - 990 lines
- [configs/base.yaml](../../configs/base.yaml)
  - 507 lines
- [src/hbtxr/data/dataset.py](../../src/hbtxr/data/dataset.py)
  - 427 lines
- [src/hbtxr/models/hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)
  - 411 lines

## 우선순위 요약

1. `trainer.py` 분리
2. config 해석 단일화
3. `base.yaml` 노이즈 축소
4. `dataset.py` façade 정리 + component registry 공통화
5. `canonicalize.py` / `groundedsam_build.py` pipeline object 분리
6. `loss` 반복 조합 함수 정리

## 1단계: trainer 허브 축소

목표는 [trainer.py](../../src/hbtxr/training/trainer.py)를 "실행 세션 orchestration"만 남도록 줄이는 것이다.

- [x] `ModelFactory` 도입
  - 대상: `build_model()`
  - 책임: config 정규화, role별 student/teacher model 구성, `HBTXRTracker` 생성
- [x] `TeacherManager` 1차 분리
  - 대상: `_create_teacher_model()`, `_ema_update()`
  - 책임: distillation teacher 생성, EMA 동기화, freeze 정책
- [x] `CheckpointManager` 1차 분리
  - 대상: `_save_checkpoint()`, `_save_teacher_checkpoint()`, `_load_checkpoint()`, `flatten_best_checkpoints_from_state()`, `checkpoint_summary_metadata()`
  - 책임: best/latest checkpoint 저장, state summary, resume metadata
- [x] `StepRunner` 계층 1차 도입
  - 대상: `_epoch_loop()` 내부 실행 분기
  - 분리 후보:
    - `StandardStepRunner`
    - `SampledParamStepRunner`
    - `ExactMarsStepRunner`
- [x] `_forward_once()`를 별도 callable object로 분리
  - 이름 후보: `ForwardAndLossRunner`
  - 책임: model forward, stage loss, distillation, regularization-ssl, pruning regularizer, metrics 계산
- [x] `TrainingSession` 도입
  - 대상: `train()`
  - 책임: loader/model/optimizer/scheduler 조립, epoch 루프, early stop, export

완료 기준:

- `trainer.py`는 `train()`, `make_loader()`, 최소 orchestration helper만 남는다.
- optimizer/teacher/checkpoint/step execution이 서로 독립 모듈에서 읽힌다.

## 2단계: config 해석 단일화

현재 dataset kwargs 해석은 두 군데에 중복된다.

- [scripts/_config.py](../../scripts/_config.py)
- [src/hbtxr/data/loader.py](../../src/hbtxr/data/loader.py)

이 중복은 drift 위험이 있으므로 가장 먼저 없애는 편이 좋다.

- [x] `src/hbtxr/config/runtime_config.py` 추가
- [x] `build_dataset_kwargs()`를 단일 구현으로 이동
- [x] `resolve_mode_contract()`와 dataset mode defaults를 한곳으로 모은다
- [x] run contract 관련 helper를 `RunContractResolver`로 묶는다
  - 대상: `resolve_run_contract()`, `write_run_artifacts()`, `resolve_training_entry()`
- [x] script layer는 공통 config resolver를 호출만 하게 줄인다

완료 기준:

- dataset kwargs 계산 로직이 한 곳에서만 유지된다.
- `train/eval/infer/vis/export` script가 동일 runtime config source를 사용한다.

## 3단계: base.yaml 표면 축소

현재 [base.yaml](../../configs/base.yaml)는 시스템 제어판 역할을 하지만, 사용자가 늘 건드리지 않는 noise도 많다.

- [x] component selector 기본값 정리
  - 현재 `split_v1`만 쓰는 항목은 main preset에서 제거 검토
- [ ] `model.student.*` / `model.patch_embed.*` / `data.mode*.*` 중 운영자 기본면과 실험면을 분리
- [ ] `base.yaml`을 운영 기본값 중심으로 단순화
- [ ] 상세 실험 옵션은 `exps/configs/` 또는 별도 schema reference 문서로 이동
- [ ] config 문서와 실제 YAML 구조의 1:1 mapping 표 정리

완료 기준:

- main 사용자는 `base.yaml + mode{1,2}_{stage}.yaml`만 읽어도 기본 실행이 가능하다.
- rarely touched 옵션은 main surface에서 덜 보이게 된다.

## 4단계: dataset façade 축소

현재 [dataset.py](../../src/hbtxr/data/dataset.py)는 1차 분리는 끝났지만, component pass-through 메서드가 많다.

- [x] `EVEyeHBTXRDataset`의 proxy 메서드 제거 검토
  - 예: `_load_annotation()`, `_resolve_event_generation_strategy()` 같은 단순 위임 메서드
- [x] `DatasetPipeline` 도입 검토
  - 책임: `reader -> roi -> transform -> event_builder -> targets -> assembler` orchestration
- [x] `Mode0Dataset`, `Mode1Dataset`, `Mode2Dataset`를 config-only specializer로 단순 유지
- [x] dataset component selector와 tracker component selector의 공통 registry 추출
  - 새 후보: `src/hbtxr/utils/component_registry.py`
- [x] `contracts.py` 확장
  - `LoadedSampleAssets`, `TargetBundle` 외에 `ResolvedRoi`, `BuiltEventInput` contract를 dataclass로 추가

완료 기준:

- dataset façade는 `__getitem__` orchestration만 담당한다.
- component registry 패턴이 dataset/tracker에서 통일된다.

## 5단계: tracker config 정규화

현재 [hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)는 실행 경로는 많이 정리됐지만, `__init__`에 config normalize 책임이 남아 있다.

- [x] `TrackerConfigNormalizer` 추가
  - 위치 후보: `src/hbtxr/models/tracker/config.py`
- [x] `eye_detector_cfg`, `roi_bbox_cfg`, `mask_cascade_cfg` 해석 이전
- [x] pruning / patch_embed / search / mask 하위 config를 dataclass로 정리
- [x] `build_model()`는 raw YAML dict 대신 normalized config를 사용하게 전환
- [x] `HBTXRTracker.__init__()`는 wiring 중심으로 축소

완료 기준:

- tracker 생성자는 config 정규화보다 module wiring이 중심이 된다.
- search/roi/mask/pruning 설정 해석이 별도 파일에서 읽힌다.

## 6단계: preprocess pipeline object 도입

offline preprocessing은 아직 orchestration과 session-level processing이 한 파일에 많이 몰려 있다.

### 6.1 canonicalize 분리

- [x] `SessionDiscovery` 추가
  - 대상: session job scan
- [x] `CanonicalizeJobPlanner` 추가
  - 대상: `session_jobs` 생성
- [x] `CanonicalSessionProcessor` 추가
  - 대상: `_canonicalize_session_job()`, `canonicalize_session()`
- [x] `CanonicalSessionArtifactWriter` 추가
  - 대상: `frame_index.jsonl`, `frame_annotations.jsonl`, `eye_region.json`, `session_package.json`, `meta.json`
- [x] `CanonicalizeSummaryWriter` 추가
  - 대상: `sessions.jsonl`, `canonical_summary.json`, skipped summary 작성
- [x] `canonicalize_dataset()`는 orchestration façade만 남긴다

### 6.2 Grounded-SAM 분리

- [x] `GroundedSamRuntimeFactory` 추가
  - 대상: runtime build / default checkpoint resolution
- [x] `GroundedSamSessionProcessor` 추가
  - 대상: session loop와 frame annotation loop
- [x] `GroundedSamShardRunner` 추가
  - 대상: multi-device shard orchestration
- [x] `GroundedSamSummaryWriter` 추가
  - 대상: session summary / shard aggregate summary

완료 기준:

- preprocess 대형 파일은 orchestration entry + session processor 중심으로 재구성된다.
- multi-device / multi-worker / session processing 책임이 분리된다.

## 7단계: loss 조합 반복 축소

현재 `loss`는 큰 위험 구간은 아니지만, `stage1/stage2`에 반복되는 조합 패턴이 남아 있다.

- [x] `BranchLossComposer` 또는 유사 helper 추가
  - search/event bbox aux / obb aux 반복 통합
- [x] stage1/stage2에서 prefix별 반복 파라미터 조합을 데이터 기반으로 줄이기
- [x] `stage_common.py`를 stage1/stage2 shared orchestration 집합으로 조금 더 확장

완료 기준:

- `stage1.py`, `stage2.py`는 branch 조합 표를 읽는 느낌으로 단순화된다.
- 계산 의미는 유지하되 중복 라인이 줄어든다.

## 검증 체크리스트

각 단계마다 아래 검증을 유지한다.

- [x] `python3 -m py_compile src`
- [x] `.venv`에서 targeted pytest
- [x] 전체 `pytest -q`
- [x] [tests/test_output_contract_snapshot_v3.py](../../tests/test_output_contract_snapshot_v3.py)
- [x] [tests/test_loss_stage_module_split_v3.py](../../tests/test_loss_stage_module_split_v3.py)
- [x] main CLI help smoke
  - `scripts/run_train.sh --help`
  - `scripts/hbtxr_mode_pipeline.sh --help`
  - `scripts/run_prepare_and_train.sh --help`

## 추천 실행 순서

1. `trainer.py` 분리
2. config 해석 단일화
3. `base.yaml` 정리
4. dataset façade/registry 정리
5. preprocess pipeline object 도입
6. loss 반복 축소

## 최종 판단

다음 리팩터링의 핵심은 “더 많은 파일 분리” 자체가 아니라, 아래 중앙 허브를 더 얇게 만드는 것이다.

- [src/hbtxr/training/trainer.py](../../src/hbtxr/training/trainer.py)
- [src/hbtxr/preprocess/canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
- [src/hbtxr/preprocess/groundedsam_build.py](../../src/hbtxr/preprocess/groundedsam_build.py)
- [configs/base.yaml](../../configs/base.yaml)

즉, 다음 단계는 `tracker/dataset/loss`를 다시 찢는 것이 아니라, `훈련 허브`, `전처리 허브`, `설정 허브`를 단순한 orchestration 계층으로 줄이는 작업이다.
