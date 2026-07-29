> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** algorithm

# algorithm 인계 로그

다른 컴퓨터/세션이 작업을 이어받기 위한 상태 기록. 최신이 위.

---

## 2026-07-29 — 양자화 A2 외 전 항목 완료, docs 구조 개편 착수

**브랜치**: `rewrite/flat-functional` · **origin 동기화됨** (`9e65e14`까지 푸시)

**환경**
```bash
# venv: ~/hbtxr-venv (python 3.11, torch 2.13.0 CPU)
cd algorithm && PYTHONIOENCODING=utf-8 ~/hbtxr-venv/Scripts/python.exe -m pytest tests -q
# -> 594 passed. 셸이 cp949라 PYTHONIOENCODING=utf-8 필수.
```

**끝난 것**: 양자화 Part A~D, A1(정수 그래프 전체), B1, B2, B6, R1~R4.
자세히는 [CHANGELOG.md](CHANGELOG.md).

**막힌 것**: A2(학습 체크포인트) 하나. 이것이 다음 질문 전부를 막고 있습니다 —
학습된 PoT 측정, 실제 분산 분포에서의 rsqrt 재검토, 모든 정확도 수치의 의미.
**사용자 결정 필요.**

**진행 중**: docs 3-트리 구조 개편 1~4단계.
설계는 [DOC-CONVENTIONS.md](../../../docs/governance/DOC-CONVENTIONS.md).
5단계(실제 파일 이동)와 6단계(NUMERICS-CONTRACT)는 미착수.

**주의할 것**
- `algorithm/docs/track/`은 이번에 신설됐습니다. 2026-07-24~29의 33커밋은 **소급 기록**
  이며 근거는 커밋 메시지입니다. 앞으로는 커밋마다 갱신해야 합니다.
- `i_ops.py`와 `archive/hardware/hls/include/hgtxr_cyclic_math.hpp`가 **같은 HG-PIPE 커널의 두
  구현**인데 교차 참조가 0건이고 수치 규약이 다릅니다. 조사 필요 — [TODO.md](TODO.md).
- 테스트를 추가할 때는 **뮤테이션 게이트 필수**(ADR-002). 이번에도 뮤테이션 생존이
  4건 있었고 전부 테스트가 약한 것이었습니다.

---

## 2026-07-23 이전

저장소 수준 [../../../docs/track/log.md](../../../docs/track/log.md) 참조.
