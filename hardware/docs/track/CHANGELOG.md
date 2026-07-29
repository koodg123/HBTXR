> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** hardware

# hardware CHANGELOG

`hardware/**`의 변경 기록. 최신이 위.

> 구 트리에는 `CHANGELOG.md`가 없었습니다 — 변경 이력이 `PROGRESS.md`와 `log.md`에
> 섞여 있었고, 그래서 "무엇이 바뀌었나"와 "어디까지 왔나"가 구분되지 않았습니다.
> 2026-07-15 이전 이력은 [PROGRESS.md](PROGRESS.md)와 [log.md](log.md)를 보십시오.

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
