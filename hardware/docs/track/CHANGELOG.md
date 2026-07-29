> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** hardware

# hardware CHANGELOG

`hardware/**`의 변경 기록. 최신이 위.

> 구 트리에는 `CHANGELOG.md`가 없었습니다 — 변경 이력이 `PROGRESS.md`와 `log.md`에
> 섞여 있었고, 그래서 "무엇이 바뀌었나"와 "어디까지 왔나"가 구분되지 않았습니다.
> 2026-07-15 이전 이력은 [PROGRESS.md](PROGRESS.md)와 [log.md](log.md)를 보십시오.

---

## 2026-07-29 — references/vit-accel 정리

- **우리 문서 2개 삭제.** ViT_Accel 문서에 프로젝트명만 치환한 사본이었습니다.
  - `reports/2026-06-16-multi-board-validation.md` — 248줄 중 실질 차이 **2줄**(`ViT_Accel`→`XR_Accel`)
  - `plans/done/2026-07-15-xr-accel-execution-plan.md` — 128줄 중 **4줄**(저장소 URL 3곳)
  - 각 디렉토리 README가 `references/vit-accel/` 원본을 가리킵니다. 원본은 `archive/`에도 있습니다.
- **`MODULE-GUIDE.md`는 유지.** 같은 파생이지만 **우리가 갱신할 문서**입니다 —
  `hgtxr_e2e_vit.hpp` 4,503줄을 분해하면 모듈 구성이 갈라집니다. 헤더에 파생 관계와
  "갱신할 생각이 없어지면 삭제하라"는 조건을 명시했습니다.
- **`experiments/` 평탄화** — 파일 10개가 디렉토리 8개에 4단계로 흩어져 있었고 4개는
  디렉토리당 파일 1개였습니다. 디렉토리 **8 → 1**. 원래 경로 전체를 그쪽 README 표에 보존.
  baseline과 ooc-rerun의 step2 요약은 바이트 동일이지만 **합치지 않았습니다** — "재실행했는데
  안 바뀌었다"가 그 실험의 결과입니다.
- **진입점 표 신설** — 6,943줄에 대해 우리 트리의 링크가 STATUS의 카운트 하나뿐이었습니다.
  "무엇을 알고 싶을 때 무엇을 읽나" + **읽지 말 것**(그쪽 2026-04 시점 상태 5개)을 명시.
- 죽은 링크 10개(`../workspace/hardware/src/*.h`)를 백틱으로 강등 →
  **`hardware/` 전체 깨진 링크 0**.
- 내용 무편집 유지: **27/30이 archive 원본과 바이트 동일**. 나머지 3개의 diff는 링크 재작성과
  우리 주석 블록뿐입니다.

---

## 2026-07-29 — 문서 이름 규약 적용

- **87개 개명.** `hardware/docs/`만 규약 밖에 있었습니다 — `THIRD_GOAL_REQUIREMENTS_2026_06_16.md`
  (UPPER_SNAKE + 접미사 날짜) · `req1_environment_audit_2026_06_16.md` (lower_snake) ·
  `2026-06-HBTXR-arch-freeze.md` (달까지만) · `summary-p0-mode-profile.md` (kebab)가
  한 트리 안에 공존했습니다. 규칙을 새로 만들지 않고 `docs/`·`algorithm/docs/`가 이미
  쓰던 것을 성문화했습니다 → [규약 §3](../../../docs/governance/DOC-CONVENTIONS.md).
  - 링크 126개 자동 재작성. `hardware/docs/` 안 **깨진 링크 0** (`references/`는 예외, 아래).
  - **`data/`의 `.json`·`.csv`는 개명하지 않았습니다** — 도구가 쓴 이름이고 manifest·sha256이
    그 이름을 참조합니다.
  - 날짜는 파일명 토큰이 없으면 헤더 `작성`에서 가져왔습니다. **추정한 날짜는 없습니다.**
- `references/vit_accel/`의 리다이렉트 스텁 3개(`SRC_CASE_{ANALYSIS,DIAGRAMS,MICROARCH}.md`,
  각 15~17줄) **삭제**. 내용은 `SRC_CASE_MODULE_GUIDE.md`와 `MODULE-GUIDE.md`에 그대로 있습니다.
  그쪽 README·HISTORY·MERGE_STATUS에 죽은 링크 3개가 남지만 **남의 문서라 고치지 않았습니다**
  ([references/README.md](../references/README.md)에 기록).
- `2026-06-27-mode-profile/summary-p0-report-collection.md` 삭제 — 06-26 캠페인 사본과
  **바이트 단위로 동일**했습니다 (archive에도 `p0_report_collection_2026_06_26/summary.md`와
  `_2026_06_27/summary.md` 두 벌). 캠페인 README가 원본을 가리킵니다.

---

## 2026-07-29 — 재구성

- 구 트리를 `archive/hardware/`로 이동. **테스트 49 failed / 426 passed 불변** (이동 전과 동일).
- 새 `hardware/` 골격 생성 — `config/ module/ build/ deploy/ tools/ docs/ workspace/`.
  각 디렉토리 README가 **이관 체크리스트**입니다.
- 문서 이관 D1~D3:
  - **D1** `references/vit_accel/` 33개 — 선행 프로젝트 문서 세트, 무변경 복사
  - **D2** `experiments/` 13캠페인 — 문서 75 · 데이터 73.
    `hgtxr_final_evidence`(CRLF)와 `hgtxr_handover_additional`(LF)이 **36쌍 동일**임을
    확인하고 LF 쪽을 정본으로 채택
  - **D3** `track/` 21 → 6종. 실험 기록 12개는 `experiments/`로, 핸드오프 4개는
    `handoff/`로 만료 고지와 함께
- 전수 semantic census (642 파일 · 285,261줄) → [reports/2026-07-29-hardware-census.md](../reports/2026-07-29-hardware-census.md)
