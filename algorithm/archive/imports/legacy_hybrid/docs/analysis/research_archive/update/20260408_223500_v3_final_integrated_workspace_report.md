# 최종 통합 작업 보고서

마지막 갱신: 2026-04-08 KST

## 요약

현재 `HBTXR_v3_0`는 여러 branch 작업 내용을 안정적으로 흡수한 통합 작업본으로 정리되었습니다.

핵심 결과:

- `docs`, `configs`, `scripts`, `src`, `tests` 통합 완료
- 실험 표면을 `exps/`로 분리해 메인 실행 표면 축소
- tracker / dataset / loss / training / preprocess 구조 분리 완료
- output contract snapshot 고정 완료
- 전체 회귀 테스트 기준 `114 passed`

## 1. 통합 원칙

- 오래된 branch를 순차적으로 전면 merge하지 않고, 충돌 위험이 큰 부분은 선별 이식으로 반영
- main runtime surface와 experimental surface를 분리
- 리팩터링 이후에도 `dataset -> model -> loss` 출력 계약은 snapshot으로 유지
- 기록성 문서는 `docs/prj/`, `docs/exps/`, `docs/others/`로 역할 분리

## 2. 최종 표면 구조

### main runtime surface

- `scripts/`
- `configs/base.yaml`
- `configs/mode1_stage1.yaml`
- `configs/mode1_stage2.yaml`
- `configs/mode2_stage1.yaml`
- `configs/mode2_stage2.yaml`
- `configs/paths/*`

### experimental surface

- `exps/scripts/`
- `exps/configs/`
- `exps/configs/groundedsam/*.json`

### docs surface

- `docs/chat/`
- `docs/exps/`
- `docs/plan/`
- `docs/prj/`
- `docs/others/`

## 3. 구조 리팩터링 결과

### tracker

- encoder / search / event / track / runtime / codec / head-factory 분리
- tracker config normalizer 도입
- component selector를 통해 YAML에서 variant 선택 가능

### loss

- bundle 분리
- `stage.py` facade 유지
- `stage1.py`, `stage2.py`, `metrics.py`, `stage_common.py` 분리

### dataset

- reader / transform / event builder / ROI resolver / target builder / sample assembler 분리
- `LoadedSampleAssets`, `ResolvedRoi`, `BuiltEventInput`, `TargetBundle` contract dataclass 정리
- `data.components` selector 추가

### training / preprocess

- `ModelFactory`, `TeacherManager`, `CheckpointManager`, `StepRunner`, `TrainingSession` 도입
- `runtime_config.py`, `run_contract.py`로 config 해석 공통화
- `canonicalize_pipeline.py`, `groundedsam_pipeline.py`로 preprocess orchestration 분리

## 4. 계약 고정과 검증

- snapshot source of truth
  - `tests/snapshots/output_contracts_v3.json`
- snapshot test
  - `tests/test_output_contract_snapshot_v3.py`
- split 동일성 test
  - `tests/test_loss_stage_module_split_v3.py`
- surface smoke test
  - `tests/test_surface_split_v3.py`

최근 전체 검증 결과:

- `PYTHONPATH=src .venv/bin/python -m pytest -q`
- 결과: `114 passed in 18.85s`

## 5. 최근 통합 커밋

| Commit | 목적 |
| --- | --- |
| `390cd83` | dataset component split + YAML selector 연결 |
| `872f26c` | output contract snapshot 고정 + loss stage 최종 분리 |
| `548ee5b` | main surface 축소 + `exps/` 분리 |
| `d699c06` | docs archive 정리 + `prj` 인덱스 단순화 |

## 6. 현재 원격 저장소

- `origin`
  - `https://github.com/koodg123/HBTXR_v3_0`

## 7. 현재 해석

이 작업본은 “모든 branch의 모든 파일을 그대로 전부 합친 literal merge”가 아니라, 실제 유지 가능한 코드베이스 통합본입니다.

즉 현재 상태는:

- 실사용 가능한 통합 완료
- 실험 자산은 분리 보관
- 구조 복잡도는 메인 표면 기준으로 크게 축소
- 향후 변경은 snapshot / 테스트 / CI guard를 기준으로 관리 가능

## 8. 참고 운영 문서

- [docs/chat/20260408_235000_v3_integrated_workspace_history.md](../../chat/20260408_235000_v3_integrated_workspace_history.md)
- [docs/plan/20260408_235100_v3_current_workspace_checklist.md](../../plan/20260408_235100_v3_current_workspace_checklist.md)
- [docs/plan/20260408_235200_v3_stabilization_and_publish_plan.md](../../plan/20260408_235200_v3_stabilization_and_publish_plan.md)
- [docs/prj/20260408_235400_v3_training_evaluation_metrics_and_deployment_guide.md](../../prj/20260408_235400_v3_training_evaluation_metrics_and_deployment_guide.md)

## 9. 후속 권장 작업

- GitHub Actions CI에서 snapshot guard를 상시 실행
- snapshot 변경 시 계약 문서 동반 갱신 규칙 유지
- archive 문서는 계속 `docs/others/`로만 누적하고, 운영 문서는 `docs/prj/`를 유지
