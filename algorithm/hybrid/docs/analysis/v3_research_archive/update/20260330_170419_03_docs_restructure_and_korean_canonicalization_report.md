# 문서 재구성 및 한국어 정본화 완료 보고서

Last updated: 2026-03-30 KST

## 요약

이 문서는 `HBTXR_v3_0` 문서 체계를 `docs/chat`, `docs/jetcas` 유지 조건 아래에서 재구성하고, 그 외 문서를 한국어 정본 체계로 전환한 최종 상태를 정리한다.

핵심 결과:

- `docs/chat`, `docs/jetcas`는 유지
- 나머지 문서는 canonical 기준으로 아래 다섯 축에 재배치
  - `docs/analysis`
  - `docs/experiments`
  - `docs/update`
  - `docs/plan`
  - `docs/progress`
- `docs/hbtxr`, `docs/references_analysis`, `docs/others`는 정본이 아니라 **호환 레이어**로 유지
- `docs/chat`, `docs/jetcas`를 제외한 canonical 본문은 한국어 정본으로 정리

## 1. 최종 문서 구조

### 유지 폴더

- `docs/chat`
  - 대화 로그, update history, progress checklist, decision log
- `docs/jetcas`
  - 논문 초안, reference 비교, manuscript-related 문서

### Canonical 폴더

- `docs/analysis`
  - 코드 구조, 데이터 파이프라인, 손실 함수, 설정, 외부 레퍼런스 분석
- `docs/experiments`
  - 실험 결과, failure review, handoff, sampled construction, pilot 문서
- `docs/update`
  - 문서 재구성, 마이그레이션, selective integration, 구조 변경 추적
- `docs/plan`
  - 초기 계획, 전환 계획, 실험 계획, 통합 계획
- `docs/progress`
  - 독립 실행 단위 기준의 진행 문서

### 호환 레이어

- `docs/hbtxr`
- `docs/references_analysis`
- `docs/others`

이 세 폴더는 과거 링크와 `docs/chat` 내부 참조를 깨뜨리지 않기 위한 **호환용 경로**로 유지한다.

## 2. 호환 레이어 처리 방식

### `docs/hbtxr`

- 기존 개별 문서는 한국어 호환 스텁으로 전환
- 각 스텁은 새 canonical 문서 경로를 안내
- `00_index.md`는 이 폴더가 정본이 아니라 호환 레이어라는 점을 명시

### `docs/references_analysis`

- FACET / Swift-Eye 세션 문서는 한국어 호환 스텁으로 유지
- 실제 정본은 `docs/analysis/references/...`

### `docs/others`

- legacy archive map과 기타 전환 문서는 호환 링크 역할만 유지
- 실제 정본은 `docs/update/...`

## 3. 한국어 정본화 범위

### 완료된 canonical 문서군

- `docs/analysis`
  - 본문 한국어화 완료
  - optimizer diff 문서군 포함
  - FACET / Swift-Eye reference session 문서군 포함
- `docs/experiments`
  - 본문 한국어화 완료
- `docs/update`
  - 마이그레이션/추적 보고서 한국어화 완료
- `docs/plan`
  - 본문 한국어화 완료
- `docs/progress`
  - 활성 정본 문서 한국어화 완료

### 제외된 문서군

- `docs/chat`
  - 구조 및 역할 유지
- `docs/jetcas`
  - 논문 작업 영역으로 유지

## 4. 핵심 인덱스 체계

상위 인덱스:

- `docs/prj/20260325_140031_00_docs_index.md`

하위 인덱스:

- `docs/analysis/00_index.md`
- `docs/experiments/00_index.md`
- `docs/update/00_index.md`
- `docs/plan/00_index.md`
- `docs/progress/00_index.md`

보조 인덱스:

- `docs/prj/optimizers/20260330_170419_00_index.md`
- `docs/prj/references/20260330_170419_00_index.md`
- `docs/prj/references/facet/20260330_170419_00_index.md`
- `docs/prj/references/swift-eye/20260330_170419_00_index.md`

## 5. 이번 재구성에서 정리된 대표 문서군

### 분석 문서

- `01` architecture
- `02` event binning / accumulation
- `03` Grounded-SAM dataset pipeline
- `04`, `05` loss catalog / investigation
- `06` repository tree / dependency map
- `07` dataset analysis
- `08` head / stage loss
- `09` mode data pipeline contracts
- `10` YAML config reference
- `11` optimizer pool reference
- `18` heuristic eye ROI algorithm

### 실험 문서

- `12`, `13` optimizer pool pilot
- `14` Grounded-SAM ROI crop summary
- `15` all48 failure review
- `16` all48 handoff
- `17` sampled dataset construction experiment

### 업데이트 문서

- `01` migration manifest
- `02` legacy archive map
- `20` selective branch integration trace report

### 계획 문서

- `00` index
- `01`~`10` 전부 한국어 정본화 완료

## 6. 문서 운영 원칙

- 앞으로 새 분석 문서는 `docs/analysis`에 먼저 추가한다.
- 새 실험 결과는 `docs/experiments`에 추가한다.
- 구조 변경이나 통합 보고서는 `docs/update`에 추가한다.
- 진행 상태를 별도 문서로 남길 필요가 있을 때만 `docs/progress`를 사용한다.
- 과거 `docs/hbtxr` 경로는 새 정본이 아니라 **호환 안내 경로**로만 취급한다.

## 7. 현재 남아 있는 후속 작업

문서 구조 재구성 자체는 사실상 완료 상태다. 남은 후속 작업은 운영성 항목이다.

- 필요 시 `docs/chat`의 update history / progress / conversation에 이번 완료 상태를 계속 동기화
- 필요 시 이번 대규모 문서 재구성 작업을 별도 commit으로 정리
- 필요 시 old compatibility layer를 더 축소할지 검토

## 결론

현재 `HBTXR_v3_0` 문서 체계는 다음 원칙으로 안정화되었다.

- `chat`, `jetcas`는 유지
- 나머지는 canonical 카테고리로 재편
- canonical 본문은 한국어 정본으로 통일
- old 경로는 호환 레이어로 유지

즉, 이제부터의 기본 인용 경로는 `docs/analysis`, `docs/experiments`, `docs/update`, `docs/plan`, `docs/progress`이다.
