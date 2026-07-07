# HBTXR_v3_0 현재 작업 CHECKLIST

최종 갱신: 2026-04-09 KST

## 목적

이 문서는 통합 작업과 구조 단순화 작업이 현재 어디까지 완료되었는지 한눈에 보기 위한 운영용 CHECKLIST 문서다.

## 1. branch 통합

- [x] branch별 역할 분석 완료
- [x] 충돌 완화 merge 전략 문서화 완료
- [x] `HBTXR_v3_0` 통합 작업본 생성 완료
- [x] `docs` 우선 병합 완료
- [x] `configs` 병합 완료
- [x] `scripts`, `src`, `tests` 병합 완료
- [x] 오래된 branch는 전면 merge 대신 선별 이식으로 정리 완료

## 2. 문서 구조 정리

- [x] `docs/chat`, `docs/exps`, `docs/plan`, `docs/prj`, `docs/others` 구조 정리
- [x] timestamp 기반 파일명 규칙 적용
- [x] 중복 문서 병합 정리
- [x] archive 문서를 `others` 계층으로 이관
- [x] `prj` 인덱스를 운영 중심으로 단순화

## 3. main / experimental surface 분리

- [x] `scripts/`를 공식 실행 표면만 남기도록 축소
- [x] 실험 script를 `exps/scripts/`로 이동
- [x] `configs/`를 공식 preset만 남기도록 축소
- [x] 실험 YAML/JSON을 `exps/configs/`로 이동
- [x] old experimental YAML path alias 추가
- [x] README와 `exps/README.md`에 분리 정책 반영

## 4. 구조 리팩터링

### tracker

- [x] encoder 분리
- [x] search/event/track branch 분리
- [x] head factory 분리
- [x] runtime policy 분리
- [x] track codec 분리
- [x] tracker config normalizer 도입
- [x] `build_model()`이 normalized tracker config를 직접 사용하도록 전환

### dataset

- [x] reader / transform / event builder / ROI resolver / target builder / assembler 분리
- [x] `DatasetPipeline` 도입
- [x] dataset/tracker component registry 공통화
- [x] dataset façade 축소
- [x] mode dataset specializer 단순화
- [x] `contracts.py` 확장

### loss

- [x] `stage.py` facade 유지
- [x] `stage1.py`, `stage2.py`, `metrics.py`, `stage_common.py` 분리
- [x] 반복 branch loss 조합 공통 helper화

### training / config / preprocess

- [x] `ModelFactory`, `TeacherManager`, `CheckpointManager`, `StepRunner`, `TrainingSession` 도입
- [x] runtime config / run contract 공통화
- [x] `canonicalize_pipeline.py` 분리
- [x] `groundedsam_pipeline.py` 분리
- [x] `canonicalize_dataset()`를 orchestration façade로 축소

## 5. 계약 고정과 검증

- [x] dataset / model / loss output contract snapshot 고정
- [x] split consistency test 추가
- [x] main surface smoke test 추가
- [x] snapshot CI guard 추가
- [x] `.venv` 기반 전체 pytest 통과

## 6. 최근 검증 상태

- [x] `python3 -m py_compile` 통과
- [x] model / dataset / loss targeted pytest 통과
- [x] 전체 회귀 테스트 통과
  - 기준 명령: `PYTHONPATH=src .venv/bin/python -m pytest -q`
  - 최근 결과: `114 passed in 18.85s`

## 7. 남은 운영 작업

- [ ] 현재 미커밋 변경 묶어서 커밋
- [ ] 필요 시 새 원격 저장소로 push
- [ ] 필요 시 GitHub Actions / PR 템플릿 / CODEOWNERS 운영 규칙 추가
