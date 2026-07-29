> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** hardware

# hardware CHANGELOG

`hardware/**`의 변경 기록. 최신이 위.

> 구 트리에는 `CHANGELOG.md`가 없었습니다 — 변경 이력이 `PROGRESS.md`와 `log.md`에
> 섞여 있었고, 그래서 "무엇이 바뀌었나"와 "어디까지 왔나"가 구분되지 않았습니다.
> 2026-07-15 이전 이력은 [PROGRESS.md](PROGRESS.md)와 [log.md](log.md)를 보십시오.

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
