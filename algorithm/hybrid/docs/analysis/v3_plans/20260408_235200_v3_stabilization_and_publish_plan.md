# HBTXR_v3_0 안정화 및 공개 PLAN

최종 갱신: 2026-04-09 KST

## 목적

이 문서는 현재 통합 작업본을 기준으로, 이후 안정화와 공개 운영을 어떤 순서로 진행하면 좋은지 정리한 후속 PLAN 문서다.

## 1. 바로 실행할 항목

1. 현재 미커밋 변경을 의미 있는 단위로 정리해 커밋한다.
2. 새 원격 저장소 `https://github.com/koodg123/HBTXR_v3_0` 기준으로 origin 상태를 유지한다.
3. README, `docs/prj`, `docs/plan`, `docs/chat`의 최신 운영 문서가 서로 연결되어 있는지 최종 점검한다.

## 2. 단기 안정화 계획

### 2.1 CI 안정화

- `pytest -q`를 기본 회귀 job으로 유지
- snapshot guard는 계약 문서 동반 변경을 계속 강제
- 필요 시 GPU가 없는 환경에서도 도는 smoke-only workflow를 유지

### 2.2 실행 표면 안정화

- `scripts/`는 공식 실행면만 유지
- 새 helper나 one-off script는 원칙적으로 `exps/scripts/`로만 추가
- experimental preset은 `exps/configs/`에만 추가

### 2.3 문서 안정화

- 운영 문서는 `docs/prj/`
- 실험 결과는 `docs/exps/`
- 히스토리와 업데이트 기록은 `docs/chat/` 또는 `docs/others/update/`
- 보조 분석은 `docs/others/analysis/`

## 3. 중기 개선 계획

### 3.1 계약 강화

- `data/contracts.py`를 중심으로 meta contract를 더 정형화
- 필요하면 runtime validator를 `src/`에 일부 승격
- snapshot 변경 규칙을 PR review 절차와 연결

### 3.2 trainer / preprocess 추가 정리

- 필요 시 `TrainingSession`을 더 잘게 나눠 log/report/export 훅을 분리
- `groundedsam_pipeline.py`와 `canonicalize_pipeline.py`에 progress/report contract를 더 명확히 부여

### 3.3 평가/보고 체계 강화

- `eval_summary.json` 포맷을 문서 기준으로 고정
- best metric, checkpoint naming, export artifact naming을 문서와 테스트 기준으로 관리

## 4. 공개 저장소 운영 계획

- 기본 브랜치 정책 정리
- issue / PR template 추가
- snapshot / contract / docs 변경에 대한 reviewer guideline 추가
- release note나 migration note를 `docs/others/update/`에 누적

## 5. 추천 우선순위

1. 현재 커밋 정리
2. origin 기준 push 및 백업
3. CI와 PR 운영 규칙 정리
4. 평가 결과 포맷과 계약 문서 추가 고정
5. 필요 시 공개 배포용 release 문서 작성
