> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** hardware

# hardware STATUS

**이 파일이 "지금 무엇이 진행 중인가"의 단일 출처입니다.**

> **트리가 둘입니다.** `hardware/`는 재구성 중이라 거의 비어 있고, 실제로 도는 코드는
> [`archive/hardware/`](../../archive/hardware/)에 있습니다. 이 STATUS는 **재구성 작업**의
> 상태이며, 구 트리의 상태는 [archive/hardware/docs/STATUS.md](../../archive/hardware/docs/STATUS.md)에
> 그대로 보존되어 있습니다.

---

## Active

**hardware 재구성** — 골격 완성, 이관 착수 전.
계획: [plans/active/2026-07-29-hardware-reconstruction.md](plans/active/2026-07-29-hardware-reconstruction.md)

| 디렉토리 | 상태 | 다음 |
|---|---|---|
| `config/` | ⬜ 비어 있음 | `archive/hardware/configs/` 33개 이관 |
| `module/` | ⬜ 비어 있음 | `archive/hardware/hls/` 67 + `refs/` 29 |
| `build/` | ⬜ 비어 있음 | `archive/hardware/vivado/scripts/` 31 + `scripts/run/` |
| `deploy/` | ⬜ 비어 있음 | `archive/hardware/pynq/*.py` 9 (비트스트림 제외) |
| `tools/` | ⬜ 비어 있음 | `archive/hardware/tools/` 87 + `tests/` 60 — `_lib/` 공용화 포함 |
| `docs/` | 🟨 계획·census만 | 나머지는 이관 순서에 따라 |
| `workspace/` | ⬜ 골격만 | gitignore 정책 적용됨 |

## Blocked

| 항목 | 무엇이 막고 있나 | 누가 풀 수 있나 |
|---|---|---|
| 보드 검증 (C3b smoke, AQ2 히스토그램) | ZCU104 물리 접근 | **사용자 승인** |
| Xilinx 툴 실행 | Vitis HLS / Vivado 2023.2 미설치 | **사용자** — 환경 구성 |
| `hgtxr_e2e_vit.hpp` 4,503줄 분해 | 합성 결과 검증 불가 (보드 없음) | 보드 접근 후 별도 과제 |

## Next

1. **P0** — 구 트리에서 죽은 것 확정 후 이관 대상에서 제외
   (`=318.empty` `=332.empty`, `common.h`의 미호출 선언 3개)
2. **P1** — `tools/_lib/` 공용화 (`load_json` 22곳 등)
3. 이후 계획 §6의 P2~P8

## Done

| 날짜 | 내용 |
|---|---|
| 2026-07-29 | 구 트리 → `archive/hardware/` 이동, 새 골격 생성 (테스트 49/426 불변) |
| 2026-07-29 | 재구성 계획 — 목표 구조 · 생사 대장 · docs 잠금 해제 순서 |
| 2026-07-29 | 전수 semantic census — 642 파일 · 285,261줄 |

## 이관 규칙

- **archive는 읽기 전용 기준입니다.** 수정하지 않습니다.
- 각 이관은 **archive의 무엇을 옮겼는지** 추적 가능해야 합니다.
- **`archive/hardware/tests`의 426 passed가 회귀 기준선입니다.** 이관 중 새 트리에서
  같은 테스트를 돌렸을 때 이 숫자가 줄면 멈춥니다.
