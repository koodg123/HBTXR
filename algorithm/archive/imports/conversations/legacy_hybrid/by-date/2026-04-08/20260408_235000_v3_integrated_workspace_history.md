# HBTXR_v3_0 통합 작업 HISTORY

최종 갱신: 2026-04-09 KST

## 문서 목적

이 문서는 `HBTXR_v3_0` 통합 작업본이 어떤 순서로 정리되었는지, 이번 대화에서 실제로 수행한 주요 작업을 기준으로 요약한 운영용 HISTORY 문서다.

## 1. 시작 상태

- 작업 루트는 `/home/kjm26/project/PRJXR/HBTXR_MERGE`였다.
- 루트 아래 각 디렉터리는 `HBTXR_v3_0`의 서로 다른 branch 작업본이었다.
- 초기 목표는 다음 3가지였다.
  - branch별 작업 내용 분석
  - 충돌 없는 통합 workspace 구성
  - 구조 복잡도 축소

## 2. branch 분석과 통합 전략 수립

- 각 branch의 역할, 고유 커밋, 변경 파일 범위를 비교 분석했다.
- `codex-exp-deepmicro2`를 통합 베이스로 보고, `codex/etri-wsl`, `codex-exp-ubeeslab`, `codex-exp-deepmicro`, `exp_wsl` 계열은 전체 merge 대신 선별 흡수 전략으로 정리했다.
- 분석 결과와 충돌 완화 방안은 아래 문서에 기록했다.
  - [BRANCH_ANALYSIS_AND_CONFLICT_FREE_MERGE_PLAN.md](../../../BRANCH_ANALYSIS_AND_CONFLICT_FREE_MERGE_PLAN.md)

## 3. 통합 작업본 생성

- 새 통합 작업본 디렉터리 `HBTXR_v3_0/`를 만들었다.
- 먼저 `docs/`를 병합하고 구조를 재편했다.
- 이후 `configs/`, `scripts/`, `src/`, `tests/`를 순차적으로 흡수했다.
- 오래된 실험 branch는 코드 회귀 위험이 커서, 전체 merge가 아니라 기능 단위 선별 이식으로 반영했다.

## 4. 문서 구조 재편

- `docs/`를 아래 5개 역할 폴더로 재구성했다.
  - `chat`
  - `exps`
  - `plan`
  - `prj`
  - `others`
- 문서 파일명에 timestamp prefix를 붙였다.
- 중복 문서는 병합 정리하고, 운영 문서와 기록성 문서를 분리했다.
- 이후 archive depth를 추가로 줄여 `docs/others/analysis`, `docs/others/optimizers`, `docs/others/references`, `docs/others/update` 구조로 정리했다.

## 5. main surface와 experimental surface 분리

- `scripts/`에는 공식 실행 표면만 남겼다.
- 실험용 Python/shell script는 `exps/scripts/`로 이동했다.
- `configs/`에는 공식 preset만 남기고, 실험용 YAML/JSON은 `exps/configs/`로 이동했다.
- experimental YAML 경로는 `scripts/_config.py`에서 한시적 alias를 지원하도록 정리했다.

## 6. tracker 구조 분리

- 기존 `HBTXRTracker` 안에 몰려 있던 로직을 아래 모듈로 분리했다.
  - `tracker/encoder.py`
  - `tracker/search_branch.py`
  - `tracker/event_branch.py`
  - `tracker/track_branch.py`
  - `tracker/head_factory.py`
  - `tracker/runtime_policy.py`
  - `tracker/codec.py`
  - `tracker/config.py`
- 이후 `TrackerConfigNormalizer`를 도입해 tracker 생성자의 config parsing 책임도 줄였다.

## 7. loss 구조 분리와 계약 고정

- `loss/stage.py`를 facade로 남기고 내부 구현을 아래로 분리했다.
  - `stage1.py`
  - `stage2.py`
  - `metrics.py`
  - `stage_common.py`
- 반복되던 branch loss 조합을 shared helper로 줄였다.
- `dataset -> model -> loss` 출력 계약을 JSON snapshot으로 고정했다.
  - `tests/snapshots/output_contracts_v3.json`

## 8. dataset 구조 분리

- dataset 내부 책임을 다음 계층으로 분리했다.
  - reader
  - transform
  - event builder
  - ROI resolver
  - target builder
  - sample assembler
  - orchestration pipeline
- 이후 component registry를 tracker와 공통화했다.
- 최근에는 dataset contract를 더 명확히 하기 위해 아래 dataclass도 추가했다.
  - `LoadedSampleAssets`
  - `ResolvedRoi`
  - `BuiltEventInput`
  - `TargetBundle`

## 9. training / config / preprocess 단순화

- `trainer.py`에서 아래 책임을 분리했다.
  - `model_factory.py`
  - `teacher.py`
  - `checkpoints.py`
  - `step_runner.py`
  - `TrainingSession`
- config 해석은 `src/hbtxr/config/` 아래 공통 모듈로 이동했다.
  - `runtime_config.py`
  - `run_contract.py`
- preprocess는 orchestration helper를 따로 뽑았다.
  - `canonicalize_pipeline.py`
  - `groundedsam_pipeline.py`

## 10. 테스트와 CI 안전장치

- `.venv`를 `uv` 기반으로 구성했다.
- output contract snapshot test, split consistency test, surface smoke test, CI guard test를 추가했다.
- 최근 전체 회귀 기준은 아래다.
  - `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 결과: `114 passed in 18.85s`

## 11. 현재 상태 해석

- 현재 `HBTXR_v3_0`는 “모든 branch를 literal하게 그대로 겹쳐 올린 저장소”가 아니라, 실사용 가능한 통합 workspace다.
- 공식 표면과 실험 표면이 분리되어 있고, 주요 구조 리팩터링이 완료돼 유지보수성이 높아진 상태다.
- 구조 단순화 체크리스트 기준으로 큰 리팩터링 항목은 사실상 닫힌 상태다.

## 12. 현재 원격 저장소

이번 정리 시점 기준 `origin`은 아래 저장소로 맞췄다.

- `https://github.com/koodg123/HBTXR_v3_0`
