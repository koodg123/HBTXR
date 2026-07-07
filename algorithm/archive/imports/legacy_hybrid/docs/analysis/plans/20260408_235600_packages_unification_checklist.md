# 외부 패키지 통합 체크리스트

최종 갱신: 2026-04-09 KST

## 목표

`/home/kjm26/project/PRJXR/HBTXR_MERGE/packages`의 독립 저장소 분석 결과를 기준으로, `HBTXR_v3_0/packages`를 공식 package hub로 정리한다.

## 체크리스트

- [x] `packages/repositories.json`를 6개 runtime repo + `swift-eye` reference 기준으로 확장
- [x] package별 `category`, `runtime_key`, `default_enabled`, `used_by` 메타데이터 추가
- [x] `external_packages.py`가 6개 runtime repo를 auto-detect 하도록 확장
- [x] `path_utils.py`에 `ultralytics_root`, `groundedsam2_root`, `timelens_xl_root`, `v2e_root` 추가
- [x] `configs/paths.example.json`에 새 root 예시 추가
- [x] `sync_packages.py`에 `--category` 필터 추가
- [x] annotation backend selector 추가
- [x] `groundedsam`, `ultralytics_sam3`, `groundedsam2` 공통 façade 추가
- [x] frame interpolation backend selector를 `timelens_xl`까지 확장
- [x] `v2e` event-generation façade 추가
- [x] canonicalize / prepare CLI가 새 selector와 root를 인식하도록 확장
- [x] package hub 문서 [packages/README.md](../../packages/README.md) 갱신
- [x] 운영 가이드 [20260408_235500_external_package_operating_guide.md](../prj/20260408_235500_external_package_operating_guide.md) 추가
- [x] 통합 audit report 추가
- [ ] root `/packages`를 실제 project-local clone hub로 재동기화
- [ ] backend별 실제 환경 설치 및 smoke run
- [ ] `Grounded-SAM-2`, `ultralytics_sam3`, `timelens_xl`, `v2e` 실환경 회귀 테스트 추가

## 검증 기준

- `repositories.json`와 `external_packages.py`가 같은 package matrix를 설명해야 한다.
- help surface에서 새 selector가 보여야 한다.
- 기존 기본 동작은 유지되어야 한다.
- `groundedsam`과 `timelens` 경로는 기존처럼 계속 동작해야 한다.
