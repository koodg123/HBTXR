> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** algorithm

# algorithm STATUS

**이 파일이 "지금 무엇이 진행 중인가"의 단일 출처입니다.** 항목의 상세 상태는
[track/TODO.md](track/TODO.md), 변경 이력은 [track/CHANGELOG.md](track/CHANGELOG.md).

---

## Active

없음. 양자화 서브시스템은 A2를 제외하고 완료 상태입니다.

## Blocked

**모든 Blocked 항목은 "누가 풀 수 있나"를 반드시 적습니다.** 이것이 없으면 blocked와
forgotten이 구분되지 않습니다.

| 항목 | 무엇이 막고 있나 | 누가 풀 수 있나 |
|---|---|---|
| **A2 — 학습 체크포인트** | manifest 빌드 + 실제 학습 실행. 레거시 HGTXR 체크포인트는 state-dict 키가 하나도 일치하지 않고 `load_checkpoint`가 `strict=True` | **사용자 결정** (학습 착수 승인) |
| 학습된 PoT 스케일 측정 | A2 | A2 해소 시 자동 |
| rsqrt 세그먼트 재검토 | A2 (현재 결론은 랜덤 가중치 기준) | A2 해소 시 자동 |
| 정확도 수치의 의미 | A2 — 현재 모든 수치가 랜덤 초기화 모델 기준 | A2 해소 시 자동 |

**A2 하나가 나머지 전부를 막고 있습니다.**

## Next (승인 대기)

- `i_ops` ↔ `archive/hardware/hls/include/hgtxr_cyclic_math.hpp` 대조 → `docs/contracts/NUMERICS-CONTRACT.md`
  (둘 다 HG-PIPE 커널인데 교차 참조 0건, 수치 규약 상이)
- 입력 활성 그리드 `2⁻⁸` 고정 (A2 무관, 즉시 가능)
- 첫/마지막 레이어 8비트를 기본 정책으로 (A2 무관)

## Done — 최근

전체는 [track/CHANGELOG.md](track/CHANGELOG.md).

| 날짜 | 내용 |
|---|---|
| 2026-07-29 | R1–R4 · 문서-코드 정합, 배포 범위 분리, 가드 배선 확인, rsqrt 세그먼트 종결 |
| 2026-07-29 | A1 remainder + B1 · 정수 그래프 전체 (stem → backbone → heads) |
| 2026-07-29 | B6 · conv 누산 int64 im2col |
| 2026-07-29 | B2 · 하이브리드 공유 스칼라 측정 |
| 2026-07-28 | A1 step 2 · 블록 datapath에 float 0개 |
| 2026-07-27 | A1 step 1 · 순수 파이썬 블록 오라클 |
| 2026-07-27 | Part D (D0–D6) · 도달 가능성·정확성·export |
| 2026-07-25 | Part A/B/C · 티어 재구조화, 완전지정 config, 정수 커널 |

## 검증 상태

```bash
cd algorithm && PYTHONIOENCODING=utf-8 ~/hbtxr-venv/Scripts/python.exe -m pytest tests -q
```
**594 passed** (2026-07-29 기준). 신규 테스트는 뮤테이션 게이트 필수 —
[track/ADR.md](track/ADR.md) ADR-002.
