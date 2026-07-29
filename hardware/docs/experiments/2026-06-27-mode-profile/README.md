> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** frozen — 실험 결과, 갱신하지 않음
> **소유** hardware

# 2026-06-27-mode-profile

모드별(search/track) 자원 프로파일 · Vivado 진행

## 문서

- [mode-profile-results.md](mode-profile-results.md)
- [runtime-mode-e2e-shared-top.md](runtime-mode-e2e-shared-top.md)
- [vivado-progress.md](vivado-progress.md)
- [work-progress-experiment-summary.md](work-progress-experiment-summary.md)
- [summary-p0-mode-profile.md](summary-p0-mode-profile.md)
- P0 리포트 수집의 **요약 문서**는 06-26 캠페인과 바이트 단위로 동일해서 사본을 두지 않고
  원본을 가리킵니다 →
  [2026-06-26-no-board-p0/summary-p0-report-collection.md](../2026-06-26-no-board-p0/summary-p0-report-collection.md)
  > **다만 요약만 같았고 데이터는 달랐습니다.** 06-27 재수집의
  > `report-collection-hls_modules.{csv,json}`·`hls_top_variants.csv`는 06-26 것과 내용이
  > 다릅니다 (`data/`에 둘 다 있습니다). 상위 요약 표가 안 바뀌었을 뿐 모듈 단위 수치는
  > 움직였다는 뜻입니다. 2026-07-29에 "재수집했지만 아무것도 안 바뀌었다"라고 적었던 것은
  > 요약만 보고 쓴 것이라 틀렸습니다.

## 출처와 주의

- `archive/hardware/docs/` 에서 2026-07-29 복사. **내용 무변경.**
- 문서 안의 `/home/kjm26/...` 절대 경로는 **당시 실행 환경 기록**이며 고치지 않습니다.
- 이 디렉토리는 **불변**입니다. 다시 측정하면 새 날짜로 새 캠페인을 만듭니다
  ([DOC-CONVENTIONS 규칙 E](../../../../docs/governance/DOC-CONVENTIONS.md)).

## 데이터 — `data/`

원본 도구 출력. **2026-07-29 추가** — D1~D5 이관이 `.md`만 대상으로 해서 빠져 있었습니다.
