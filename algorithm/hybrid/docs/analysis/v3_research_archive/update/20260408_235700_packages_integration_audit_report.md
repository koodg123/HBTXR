# 외부 패키지 통합 감사 보고서

최종 갱신: 2026-04-09 KST

## 요약

`/home/kjm26/project/PRJXR/HBTXR_MERGE/packages`의 독립 저장소 6개를 분석한 결과, `HBTXR_v3_0/packages`를 공식 package hub로 지정하고 manifest-first 방식으로 통합하는 것이 가장 안전하다.

이번 정리에서 외부 코드를 `src/`로 vendor하지는 않았다. 대신 HBTXR 내부에서 아래를 통합했다.

- package manifest
- path auto-detect
- backend selector
- runtime façade
- 운영 문서

## 대상 저장소와 결론

### Annotation

- `Grounded-Segment-Anything`
  - 상태: main backend
  - selector: `groundedsam`
- `ultralytics`
  - 상태: SAM3 annotation backend
  - selector: `ultralytics_sam3`
- `Grounded-SAM-2`
  - 상태: main-tier parallel backend
  - selector: `groundedsam2`

### Frame interpolation

- `timelens`
  - 상태: 기본 backend
  - selector: `timelens`
- `TimeLens-XL`
  - 상태: 병렬 backend
  - selector: `timelens_xl`

### Event generation

- `v2e`
  - 상태: synthetic event backend
  - selector: `v2e`
  - 비고: frame interpolation backend가 아니라 별도 event-generation backend

### Inactive optional reference

- `Swift-Eye`
  - 상태: manifest-only inactive optional reference

## 구현 결과

- [packages/repositories.json](../../../packages/repositories.json)
  - package matrix의 source of truth
- [external_packages.py](../../../src/hbtxr/utils/external_packages.py)
  - manifest와 동일한 runtime key를 기준으로 auto-detect
- [path_utils.py](../../../src/hbtxr/preprocess/path_utils.py)
  - 새 root field 추가
- [annotation_backends.py](../../../src/hbtxr/preprocess/annotation_backends.py)
  - annotation façade
- [interpolation.py](../../../src/hbtxr/preprocess/interpolation.py)
  - `timelens_xl` 추가
- [event_generation.py](../../../src/hbtxr/preprocess/event_generation.py)
  - `v2e` façade 추가

## 남은 운영 작업

- 실제 clone/sync를 `HBTXR_v3_0/packages` 기준으로 재실행
- backend별 설치 가이드와 checkpoint provisioning 정리
- non-default backend의 실환경 smoke test 추가
