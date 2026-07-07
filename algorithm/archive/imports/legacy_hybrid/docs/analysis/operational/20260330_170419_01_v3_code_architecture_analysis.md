# HBTXR_v3_0 코드 아키텍처 분석

최종 갱신: 2026-03-30 KST

Role: `시스템 아키텍트`, `ML/비전 연구 엔지니어`, `XR 런타임 엔지니어`

## 요약

- 이 문서는 현재 active `v3` 표면만 분석한다.
  - `src/`, `scripts/`, `configs/`, `tests/`, `docs/` canonical 구조
- `legacy/`는 역사적 참고용으로만 취급하며, 주요 결론에서 제외한다.
- 분석 초점은 아키텍처와 데이터 흐름이다.
  - `raw -> canonical -> manifest -> dataset -> model -> loss/trainer -> runtime`
- 현재 브랜치는 optimizer pool, loss cluster 분리, mode2 lazy target-FPS, stage-aware stat filtering, canonical docs 구조까지 반영된 상태다.

## 분석 기준

주요 검토 표면:

- [scripts/_config.py](../../scripts/_config.py)
- [train_hbtxr.py](../../scripts/train_hbtxr.py)
- [eval_hbtxr.py](../../scripts/eval_hbtxr.py)
- [infer_hbtxr.py](../../scripts/infer_hbtxr.py)
- [canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)
- [build_manifests.py](../../src/hbtxr/preprocess/build_manifests.py)
- [target_fps_build.py](../../src/hbtxr/preprocess/target_fps_build.py)
- [target_fps_canonical.py](../../src/hbtxr/preprocess/target_fps_canonical.py)
- [dataset.py](../../src/hbtxr/data/dataset.py)
- [hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)
- [backbone.py](../../src/hbtxr/models/backbone.py)
- [heads.py](../../src/hbtxr/models/heads.py)
- [controller.py](../../src/hbtxr/models/controller.py)
- [registry.py](../../src/hbtxr/optim/registry.py)
- [trainer.py](../../src/hbtxr/training/trainer.py)
- [tracker.py](../../src/hbtxr/runtime/tracker.py)

## 1. End-to-end 아키텍처

현재 `v3` 시스템은 “하나의 모델 스크립트”가 아니라 “계약 중심 파이프라인”으로 조직되어 있다.

```text
raw EV-Eye sessions
  -> canonicalize_dataset() / target_fps_build()
  -> canonical session package + annotation store + events.npz / session store
  -> build_manifests()
  -> split-specific manifest rows
  -> EVEyeHBTXRDataset
  -> HBTXRTracker
  -> stage1/stage2 loss + trainer
  -> checkpoint / eval / infer outputs
  -> optional runtime FSM usage
```

핵심 설계 선택:

- preprocessing
  - source normalization과 provenance를 담당
- manifest
  - experiment-time selection policy를 담당
- dataset
  - sample assembly와 geometry-consistent transform을 담당
- model
  - multimodal encoding과 decoupled head를 담당
- trainer
  - stage-specific optimization policy를 담당
- runtime
  - host-side mode switching logic을 담당

즉 event slicing, resize policy, target ABI, runtime switching을 한 클래스에 몰아넣지 않고 계층별 계약으로 분리한 구조다.

## 2. Entry path와 실험 orchestration

### 2.1 Shell / CLI 진입 구조

현재 공식 진입은 shell wrapper와 Python script 조합이다.

- shell wrapper
  - 환경 변수, interpreter, 기본 config, 출력 루트 정리
- Python CLI
  - config merge
  - train / eval / infer 실행
  - run artifact 기록

현재 실험 재현성 계약에 포함되는 산출물:

- resolved config snapshot
- source config copy
- CLI args / override 목록
- device selection
- output layout metadata

즉 shell layer는 단순 glue가 아니라 실험 재현성의 일부다.

### 2.2 Config resolution 모델

[scripts/_config.py](../../scripts/_config.py)는 다음 우선순위를 구현한다.

1. `extends`로 이어지는 base YAML
2. child YAML
3. dotted CLI override
4. manifest / output / checkpoint override

experiment naming 우선순위도 명시적이다.

1. config application 시 명시적 override
2. `experiment.name`
3. config filename stem

이 부분은 코드베이스의 강점이다. 실험 정체성을 임시 폴더 이름이 아니라 1급 artifact로 다룬다.

### 2.3 Output 계약

활성 script는 대체로 다음 출력 계약을 따른다.

```text
runs/experiments/<EXPERIMENT_NAME>/
  Train/Stage1 or Train/Stage2
  Eval/<split>
  Inference/<split>
  Visualization/<kind>/<split>
  Hyperparameters/
```

optimizer pool은 여기에 추가로:

```text
runs/<exp>/pool/<candidate>/train
```

형태를 사용한다.

## 3. 데이터 경로

### 3.1 Canonicalization 계층

[canonicalize.py](../../src/hbtxr/preprocess/canonicalize.py)는 raw session 디렉토리를 canonical session package로 바꾼다.

세션별 대표 산출물:

- `frames/`
- `events/events.npz`
- `labels/frame_annotations.jsonl`
- `labels/frame_index.jsonl`
- `labels/session_package.json`
- `meta.json`

mode2 target-FPS 경로에서는 별도로:

- `session_store_path`
- `session_store_layout`
- `event_index_range`

같은 session-store 기반 메타를 bridge한다.

즉 canonicalization은 단순 파일 복사가 아니라 이후 단계가 의존하는 supervision contract를 정의한다.

### 3.2 Manifest 계층

[build_manifests.py](../../src/hbtxr/preprocess/build_manifests.py)는 canonical session package를 split별 sample row로 변환한다.

대표 row binding:

- data pointer
  - frame, event, session store
- annotation reference
  - current / previous annotation
- geometry policy
  - ROI, resize policy, target size
- event slicing policy
  - `event_window`
- quality / validity flag
  - `annotation_quality`, `closed_eye_flag`, `mask_valid`, `valid_track`
- transition metadata
  - `similarity_target`

중요한 설계 포인트는 event slicing과 resize policy가 loader 내부에 숨지 않고 manifest와 config 양쪽에 기록된다는 점이다.

### 3.3 Dataset 계층

[dataset.py](../../src/hbtxr/data/dataset.py)는 저장 truth와 training ABI를 연결하는 실제 브리지다.

주요 역할:

- canonical annotation row load
- `events.npz` 또는 session store 기반 event load
- `fixed_count`, `time_bin`, `interval_all`, `source_pair_average`, `target_fps_session_store` 경로 처리
- frame, mask, event, bbox, state에 일관된 spatial transform 적용
- `pupil_search_target`, `pupil_track_target`, `constraint_center` 등 stage target 생성
- event tensor cache 사용

### 3.4 Sample ABI

dataset가 반환하는 sample은 일반적인 dataset item보다 풍부하다.

```python
{
  "frame",
  "event",
  "mask_target",
  "eye_target",
  "prev_state",
  "cur_state",
  "pupil_search_target",
  "pupil_track_target",
  "constraint_center",
  "annotation_quality",
  "similarity_target",
  "event_density",
  "closed_eye_flag",
  "mask_valid",
  "valid_track",
  "aux_target",
  "meta",
}
```

이 sample contract가 사실상 학습 시스템의 중앙 ABI다.

### 3.5 Loader override precedence

loader는 두 축에서 현재 config를 manifest보다 우선시한다.

- resize policy
- event builder policy

즉 manifest는 일부 provenance, 일부 운영 기본값의 성격을 갖고 있으며, 항상 절대 실행 진실은 아니다.

## 4. 모델 경로

### 4.1 입력 ABI

현재 모델 입력 ABI는 단순하고 안정적이다.

- `frame [B,1,256,256]`
- `event [B,2,256,256]`
- `prev_state [B,6]`

README, config, model 코드, test 전반에서 일관된다.

### 4.2 Encoder 설계

[hybrid_tracker.py](../../src/hbtxr/models/hybrid_tracker.py)는 다음 구조를 사용한다.

- frame patch embedding
  - `Conv2d(1 -> embed_dim, kernel=patch, stride=patch)`
- event patch embedding
  - `Conv2d(2 -> embed_dim, kernel=patch, stride=patch)`
- modality별 residual adapter
- shared `PartialDeiTTiny` backbone

중요한 점은 `frame`, `event`가 transformer trunk를 공유한다는 것이다. search / track branch 분리 전에 토큰 처리 trunk를 공유하는 것이 현재 배포 지향 압축 포인트다.

### 4.3 Head 분해

head는 명시적으로 분리되어 있다.

- `EyeRegionHead`
- `PupilSearchHead`
- `EventSearchHead`
- `PupilTrackHead`
- `SearchMaskHead`
- `AuxStateHead`

또한 최근 정책 기준으로:

- Stage1
  - `mask_head` 사용
  - `event_head` 비활성
- Stage2
  - `event_head` 사용
  - `mask_head` 비활성
- deployment
  - `event_head`, `mask_head` 제거 가능

### 4.4 State 표현

시스템은 raw angle 대신 `xyabuv` 상태 인코딩을 사용한다.

- `x, y`
  - center
- `a, b`
  - ellipse axis
- `u, v`
  - orientation의 trig 표현

이 방식은 angle discontinuity를 피하고 geometry loss를 더 안정적으로 만든다.

### 4.5 Track 경로 설계

tracking path는 pure event encoding이 아니다. 아래 둘을 fusion한다.

- event pooled feature
- encoded `prev_state`

그 뒤 residual을 예측한다.

- `dx, dy`
- `dlog(a), dlog(b)`
- `du, dv`
- confidence
- quality

이 설계는 track를 full re-detection이 아니라 state refinement로 명시해 runtime 안정성에 유리하다.

## 5. 학습 경로

### 5.1 Stage 분리

[trainer.py](../../src/hbtxr/training/trainer.py)는 두 stage를 지원한다.

- `stage1`
  - search-centric pretraining
- `stage2`
  - event / residual tracking을 포함한 hybrid finetuning

이 분리는 config, 출력 디렉토리, loss 계산, checkpoint naming에 일관되게 반영되어 있다.

### 5.2 Loss 구조

현재 loss는 cluster 파일로 분리되었지만, 개념적으로는 geometry-aware multi-objective 구조다.

- eye box regression + confidence
- mask BCE + Dice
- search branch `xy / ab / trig / geo / conf`
- event branch `xy / ab / trig / geo / conf`
- track branch `xy / ab / trig / geo / conf / quality`
- search-track consistency
- `constraint_center`
- optional auxiliary classification
- distillation / SSL regularization

강점:

- ellipse covariance와 GWD 유사 거리 기반 geometry loss
- validity / quality gate를 반영한 masking
- Stage2에서도 search supervision을 유지

### 5.3 Optimizer / checkpoint 정책

현재 trainer는 `AdamW` 하드코딩을 제거했고 `src/hbtxr/optim` registry를 사용한다.

지원 범위:

- optimizer registry
- modifier
  - `cautious`
  - `schedule_free`
- pool sweep
- optimizer metadata 기록
- `best_metric_name` 기반 best checkpoint
- custom metric용 동적 `best_<metric>.pt`

### 5.4 Trainer 강점

trainer는 다음을 포함한다.

- scheduler 선택
  - `none`, `cosine`, `step`, `plateau`
- optional early stopping
- resume support
- stage2 init from stage1 checkpoint
- optimizer-specific path
  - `ivon`
  - `sophia_g`
  - `mars`
- stage-aware stat filtering
- checkpoint metadata 및 epoch history 기록

## 6. 런타임 경로

### 6.1 Scheduler FSM

[controller.py](../../src/hbtxr/models/controller.py)는 host-side `TrackSearchSchedulerFSM`을 정의한다.

전이 신호:

- `search_conf`
- `track_conf`
- `track_quality`
- `similarity`
- `event_density`
- `closed_eye_flag`

동작 요약:

- closed eye면 즉시 `search`
- `track`는 confidence / quality / similarity / density가 기준 이하이면 `search`로 하향
- `search`는 모든 gating이 기준 이상일 때만 `track`으로 승격
- relocalization cooldown으로 즉각적인 진동 방지

### 6.2 두 개의 런타임 entry shape

runtime 표면은 두 개다.

- `HBTXRTracker.runtime_step()`
- `RuntimeHBTXRTracker.step()`

둘 다 search / track output을 계산하고 control signal을 만들고 FSM을 질의한다.

### 6.3 중요한 경계 조건

runtime scheduler는 library 코드에는 존재하지만, 기본 `eval` / `infer` CLI는 이를 직접 사용하지 않는다. 현재 CLI는 대체로 `model(batch)`를 호출하고 branch output을 기록한다.

즉:

- runtime logic은 존재한다.
- 모듈 수준 테스트도 가능하다.
- 하지만 main `eval` / `infer` 표면에 완전히 닫힌 배포 경로로 연결된 것은 아니다.

이 지점이 현재 코드베이스의 큰 architecture-to-operation gap이다.

## 7. 배포 지향 설계 의도

### 7.1 단일-window event ABI

event 입력이 더 깊은 time-stack이 아니라 `[2,H,W]`로 고정된다.

이 선택은 다음을 단순화한다.

- accelerator input contract
- memory planning
- runtime scheduling
- shared frame / event trunk 통합

### 7.2 Shared backbone

frame과 event가 transformer trunk를 공유하므로 시스템 복잡도가 줄고, hardware partitioning도 더 단순해진다.

### 7.3 Host-side scheduler

search / track switching을 learned trunk 밖에 두어:

- failure mode inspection
- threshold tuning
- accelerator boundary 분리

를 더 쉽게 만든다.

### 7.4 실험 artifact discipline

shell workflow와 실험 디렉토리 구조는 재현성을 강하게 지원한다. 이것은 편의 스크립트가 아니라 아키텍처의 일부다.

## 8. 핵심 리스크와 갭

아래 항목은 코드 기준으로 직접 확인 가능한 리스크다.

### 8.1 `similarity_target`는 생성되지만 활성 사용이 약하다

manifest는 `similarity_target`을 만들고 dataset는 이를 sample에 실어 나르지만, 현재 핵심 loss / metric / runtime 경로에서 강하게 닫히진 않는다.

영향:

- manifest가 저장하는 supervision / control signal 일부가 실질적으로 약하게 연결됨

### 8.2 일부 config knob는 의미가 제한적이다

예:

- deprecated size 관련 인자
- 일부 legacy compatibility key

영향:

- 사용자 mental model과 실제 동작이 어긋날 수 있음

### 8.3 Runtime FSM은 main CLI 경로와 완전히 닫히지 않았다

영향:

- 표준 evaluation이 배포형 mode-switch behavior를 완전히 반영하지 않음

### 8.4 Scaffold와 active integration이 섞여 있다

예:

- legacy helper
- 일부 experimental surface
- compatibility layer

영향:

- 코드베이스를 처음 읽는 사용자가 active path와 보조 / 역사적 path를 혼동할 수 있다.

## 9. 테스트 커버리지 평가

현재 테스트는 유용하지만 배포급 완전성은 아니다.

현재 잘 덮는 영역:

- config resolution과 override
- dataset sample ABI
- mode / event selection 경로
- model output ABI
- optimizer pool
- minimal stage1 / stage2 smoke path
- import graph와 console stat filtering

여전히 약한 영역:

- main CLI와 runtime FSM의 완전 통합
- 실제 EV-Eye full dataset end-to-end
- hardware latency / throughput / scheduler transition benchmark
- 환경 차이까지 포함한 재현성 검증

따라서 현재 테스트 스위트는 다음으로 보는 것이 맞다.

- ABI 보호
- contract smoke coverage
- 개발 회귀 방지
- 아직 deployment-grade validation net은 아님

## 10. 종합 평가

### 강점

- preprocessing, manifest, data, model, training, runtime의 분리
- 강한 실험 artifact discipline
- 안정적인 multimodal ABI
- geometry-consistent transform 처리
- 배포 지향 search / track 분해
- optimizer / loss / docs 구조의 최근 정리

### 약점

- 일부 노출된 knob는 실제 영향이 제한적이다.
- runtime scheduler가 표준 CLI 경로에 완전히 닫히지 않았다.
- 일부 문서화된 개념은 구조로는 존재하지만 full closed-loop validation은 아직 약하다.

### 실무 결론

`HBTXR_v3_0`은 배포 의도가 명확한 구조화된 연구 baseline으로 이해하는 것이 가장 정확하다.  
가장 큰 강점은 contract-oriented architecture이고, 가장 큰 남은 갭은 runtime / evaluation 폐회로를 main CLI 수준에서 얼마나 더 닫을 것인가에 있다.

## 진행 체크리스트

- [x] active `v3` 표면 분석
- [x] entry / orchestration 경로 분석
- [x] data 경로 분석
- [x] model 경로 분석
- [x] training 경로 분석
- [x] runtime 경로 분석
- [x] public ABI 및 sample contract 정리
- [x] static risk 점검
- [x] 테스트 커버리지 갭 식별
- [ ] 배포형 runtime CLI 완전 통합 여부 재평가
