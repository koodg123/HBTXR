> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 색인
> **소유** hardware

# experiments — 날짜·캠페인별 실험 기록

**한 캠페인 = 한 디렉토리 = 불변.** 다시 측정하면 새 날짜로 새 디렉토리를 만듭니다
([DOC-CONVENTIONS 규칙 E](../../../docs/governance/DOC-CONVENTIONS.md)).

각 디렉토리는 `README.md`(무엇을 했나 · 기록된 status) + 리포트 `.md` + `data/`(원본 도구
출력)를 담습니다.

## 시간순

| 날짜 | 캠페인 | 문서 | 데이터 | 내용 |
|---|---|---:|---:|---|
| 06-06 | [cyclic-weight-layout](2026-06-06-cyclic-weight-layout/) | 3 | 6 | cyclic 가중치 레이아웃·패킹 확정 |
| 06-08 | [hgpipe-reference](2026-06-08-hgpipe-reference/) | 1 | 2 | HG-PIPE 참조 분석 · Q4W8A S2 양자화 |
| 06-09 | [e2e-axis-baseline](2026-06-09-e2e-axis-baseline/) | 1 | 1 | AXIS end-to-end 베이스라인 |
| **06-10** | [**final-signoff**](2026-06-10-final-signoff/) | **31** | **47** | 3차 목표 최종 서명 — 하루에 78개 산출 |
| 06-12 | [legacy-experiment](2026-06-12-legacy-experiment/) | 1 | — | 레거시 실험 분석 |
| **06-16** | [**third-goal-audit**](2026-06-16-third-goal-audit/) | **15** | 11 | req1/5/6/9 · 소스·게이트 감사 |
| 06-16 | [vref-p0-successor](2026-06-16-vref-p0-successor/) | 6 | 6 | PoT 스케일 스윕 · 버퍼 수명 · QKV URAM |
| 06-23 | [c3b-resource-power](2026-06-23-c3b-resource-power/) | 3 | — | C3b 자원·전력·지연 |
| 06-26 | [no-board-p0](2026-06-26-no-board-p0/) | 3 | — | 보드 없는 P0 계획·1차 수집 |
| 06-27 | [mode-profile](2026-06-27-mode-profile/) | 5 | — | search/track 모드별 프로파일 |
| 06-28 | [eight-question](2026-06-28-eight-question/) | 3 | — | 8문항 — Vivado·하이브리드 런타임·full-AXI |
| 06-30 | [aq2-search-track](2026-06-30-aq2-search-track/) | 3 | — | AQ2 지연·대역폭·자원·전력 측정 |
| 07-01 | [zcu104-physical-plan](2026-07-01-zcu104-physical-plan/) | 2 | — | ZCU104 물리 실험 계획 (**보드 필요**) |

**총 13캠페인 · 문서 77 · 데이터 73.**

> 이 표의 숫자는 `python scripts/check_docs_counts.py`가 검증합니다. 손으로 세지 마십시오 —
> 이 표는 이미 두 번 틀렸습니다 (75·78 둘 다 실제와 달랐습니다).

## 이 기록이 말해주는 것

- **2026-06-10 하루에 78개**가 나왔습니다. 최종 서명 파이프라인
  (`run_third_goal_final_signoff.py`)이 한 번 돌면서 감사·검증·패키징 산출물을 한꺼번에
  만든 결과입니다. 사람이 하루에 쓴 문서가 아닙니다.
- **대부분은 통과했습니다.** 전 캠페인 status 83건 집계: `pass` **36** · `blocked` **13** ·
  `missing` 6 · `pending-*` 11 · 나머지는 한두 건씩. 감사 자체는 대체로 통과했고,
  막힌 13건이 보드 접근과 형제 저장소 부재에 걸려 있습니다.
  (초안에 "blocked가 지배적"이라고 썼는데 집계해 보니 반대였습니다 — 세어 보기 전에는
  쓰지 않는 편이 낫습니다.)
- 남은 상태값이 말해주는 것: `ready-for-board` 2 · `ready-for-physical-smoke` 2 ·
  `pending-board-run` 2 · `candidate-ready-needs-approval` 2 — **보드만 있으면 진행 가능한
  항목이 여럿 대기 중**입니다.
- 06-06 → 07-01 사이 **한 달간 13개 캠페인**(서로 다른 12개 날짜)이 돌았고, 06-10과 06-16 두 날짜에 집중돼
  있습니다.

## 출처

전부 `archive/hardware/docs/{resources,reports,analysis,track,legacy}/` 에서
2026-07-29에 복사했습니다. **내용 무변경.**

`resources/`는 원래 두 벌(`hgtxr_final_evidence` CRLF · `hgtxr_handover_additional` LF)로
보관돼 있었고 **36쌍이 줄바꿈만 다른 동일 파일**이었습니다. LF 쪽(상위집합, 126개)을
정본으로 택했습니다. CRLF 쪽에만 있던 `.npz`/`.f32bin` 바이너리 4개는 문서가 아니므로
`archive/`에 남습니다.
