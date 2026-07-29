> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** repo

# HBTXR STATUS — 저장소 전체

**"지금 무엇이 진행 중이고 무엇이 막혀 있나"의 최상위 단일 출처.**
서브시스템 상세는 각자의 STATUS를 봅니다.

| 서브시스템 | STATUS | 마지막 활동 |
|---|---|---|
| algorithm | [algorithm/docs/STATUS.md](../algorithm/docs/STATUS.md) | **2026-07-29** (활발) |
| hardware | [hardware/docs/STATUS.md](../hardware/docs/STATUS.md) | 2026-07-15 (휴지) |
| repo (거버넌스·구조) | 이 파일 | 2026-07-29 |

전체 문서 색인: [INDEX.md](INDEX.md) · 문서 규약: [governance/DOC-CONVENTIONS.md](governance/DOC-CONVENTIONS.md)

---

## Active

없음. docs 구조 개편 1~5단계 완료. 6단계(`contracts/NUMERICS-CONTRACT.md`)는 아래 Next.

## Blocked

**모든 Blocked 항목은 "누가 풀 수 있나"를 반드시 적습니다.** 이것이 없으면 blocked와
forgotten이 구분되지 않습니다 — 아래 백로그 41건이 정확히 그 상태였습니다.

| 항목 | 무엇이 막고 있나 | 누가 풀 수 있나 |
|---|---|---|
| **A2 — 학습 체크포인트** | manifest 빌드 + 실제 학습 실행 | **사용자 결정** |
| **보드 접근** (하드웨어 P0 전부) | ZCU104 물리 접근 | **사용자 승인** |
| 선존 백로그 41건 (T-/SI-/RC-/CL-) | 계획 문서가 *"every deferred task needs a new selection and its applicable gate"*라고 명시 | **사용자 선택** — 어느 묶음을 열지 |
| Spec-Kit 스캐폴딩 초기화 | 결정 게이트 | 사용자 |
| Git LFS vs 외부 아티팩트 저장소 | 결정 게이트 | 사용자 |
| 의존성 설치 (T-000 테스트 환경) | 결정 게이트 | 사용자 |

## Next (승인 대기)

| 항목 | 규모 | 의존 |
|---|---|---|
| docs 6단계 — `contracts/NUMERICS-CONTRACT.md` | 중간 | 조사 필요 |
| **RC-/CL-/SI-/T- 백로그 상태 감사** | 중간 | — |
| **`hardware/docs/` 재구조화** | 중간 | **하드웨어 소유** — `hardware/tools/`가 경로로 고정하고 테스트 10개+가 읽으므로 도구·테스트를 같이 고쳐야 함. [DOC-CONVENTIONS §11](governance/DOC-CONVENTIONS.md) |
| hardware 테스트 49건 실패 조사 | 중간 | 하드웨어 소유. 문서 이동 전부터 실패 중 |

> **RC-/CL-/SI-/T- 감사가 왜 필요한가**: AM 섹션이 6일간 "미착수"로 표기되어 있었는데
> 실제로는 AM-900까지 실행되고 대체된 상태였습니다 ([track/TODO.md](track/TODO.md) 참조).
> 같은 파일의 다른 섹션들도 실제와 대조된 적이 없습니다.

## Done — 최근

| 날짜 | 내용 |
|---|---|
| 2026-07-29 | docs 5단계 — 188개 `git mv`, `.agents/` 해소(트리 4→3), 얼어붙은 문서는 경로 미변경 |
| 2026-07-29 | docs 2~4단계 — STATUS×3, `INDEX.md` 자동생성, `DOC-CONVENTIONS.md` |
| 2026-07-29 | docs 1단계 — `algorithm/docs/track/` 신설·33커밋 소급, AM 상태 충돌 해소, README 정정 |
| 2026-07-29 | TSR_FPGA(외부 INT8 CNN 가속기) 전수 분석 |
| 2026-07-24~29 | 양자화 Part A~D · A1 · B1 · B2 · B6 · R1~R4 (33커밋, 594 tests) |
| 2026-07-22 | flat-functional 재작성 — AM 계획의 `eveye.*` 패키징을 대체 |
| 2026-07-21 | AM-000 ~ AM-900 실행 |

## 이 저장소의 구조적 사실 (자주 잊힘)

- **문서 트리는 3개**: `docs/` (횡단) · `algorithm/docs/` · `hardware/docs/`.
  `.agents/`는 네 번째였고 2026-07-29에 `docs/handoff/`·`docs/snapshots/`·`docs/plans/done/`로
  흡수되어 사라졌습니다.
- **`hardware/docs/`는 자기 툴체인에 고정**되어 있습니다 — `hardware/tools/`가 경로로 읽고
  테스트 10개+가 의존하므로 `docs/`처럼 재배치할 수 없습니다.
- **`hardware/`는 문서 트리가 아니라 코드 서브시스템**입니다 — HLS/RTL/Vivado/PYNQ.
  ViT 가속기(`hgtxr_e2e_vit.hpp`)가 있습니다.
- **같은 HG-PIPE 커널이 두 서브시스템에 각각 구현**되어 있고 교차 참조가 0건입니다.
  → [hardware/docs/STATUS.md](../hardware/docs/STATUS.md) 하단
