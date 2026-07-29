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

**hardware 재구성** — 문서 완료 · **코드 이관 M1/5 진행 중**.
계획: [plans/active/2026-07-29-hardware-reconstruction.md](plans/active/2026-07-29-hardware-reconstruction.md) ·
대장: [plans/active/2026-07-29-code-migration-manifest.md](plans/active/2026-07-29-code-migration-manifest.md)

| 디렉토리 | 상태 | 다음 |
|---|---|---|
| `config/` | ✅ **M1 완료** | 33/33. M5까지는 사본 — 도구는 아직 archive를 읽습니다 |
| `module/` | ✅ **M2 완료** | 96/96. **합성 검증 안 됨** — Vitis HLS 없음 |
| `build/` | ⬜ 비어 있음 | `archive/hardware/vivado/scripts/` 31 + `scripts/run/` |
| `deploy/` | 🟨 런타임 가이드만 | `archive/hardware/pynq/*.py` 9 (비트스트림 제외) |
| `tools/` | ⬜ 비어 있음 | `archive/hardware/tools/` 87 + `tests/` 60 — `_lib/` 공용화 포함 |
| `docs/` | ✅ **완료 (D1~D5)** | 159개 전수 대조 완료 |
| `workspace/` | ⬜ 골격만 | gitignore 정책 적용됨 |

### 문서 이관 결과 (D1~D5, 2026-07-29)

| | |
|---|---|
| `references/vit-accel/` | 30 — 선행 프로젝트. 이름·계층 정리, **내용 무편집**(27/30 바이트 동일) |
| `experiments/` | **13캠페인 · 문서 77 · 데이터 84** |
| `track/` | 6종 (구 21개에서 실험 12·핸드오프 4 분리) |
| `handoff/` | 5 — 전부 만료 고지 부착 |
| `plans/{active,done}/` | 9 (README 포함) — active 1 · done 7 |
| `reports/` · `architecture/` · `snapshots/` | 4 · 2 · 12 (census 원자료 9 포함) |
| `SPEC.md` `ARCHITECTURE.md` `MODULE-GUIDE.md` | 승격 |

> `reports/2026-06-16-multi-board-validation.md`와 `plans/done/2026-07-15-xr-accel-execution-plan.md`는
> **2026-07-29 삭제**했습니다. 우리 문서가 아니라 ViT_Accel 문서에 프로젝트명만 치환한
> 사본이었습니다 ([references/README.md](references/README.md)).

**검증**: archive md 159개를 내용 해시로 대조해 미이관 0건.

`ARCHITECTURE.md`는 유일하게 새로 쓴 문서입니다 — 이관 대상이던
`DIRECTORY_LAYOUT.md`가 **구 레이아웃**을 정의하고 있어 그대로 두면 즉시 거짓이 됩니다.

> **D1~D5는 `.md`만 대상으로 했습니다.** P0의 전수 대장이 **데이터 11개 미이관**을
> 찾아냈습니다 — 3개 캠페인이 `data/` 없이 문서만 있었고, census 원자료 9개는
> `archive/`에 남아 있었습니다. 2026-07-29 보완 완료.

## Blocked

| 항목 | 무엇이 막고 있나 | 누가 풀 수 있나 |
|---|---|---|
| 보드 검증 (C3b smoke, AQ2 히스토그램) | ZCU104 물리 접근 | **사용자 승인** |
| Xilinx 툴 실행 | Vitis HLS / Vivado 2023.2 미설치 | **사용자** — 환경 구성 |
| `hgtxr_e2e_vit.hpp` 4,503줄 분해 | 합성 결과 검증 불가 (보드 없음) | **사용자** — 위 두 행이 풀리면 자동으로 풀립니다. 그 전에는 착수 금지 |

## Next

**P0 완료** — 이관 대장 625행 전수 판정
([대장](plans/active/2026-07-29-code-migration-manifest.md) · `migrate` 322 · `done` 234 ·
`skip` 64 · `undecided` 5). 검증: `python scripts/check_migration_manifest.py --check`

디렉토리별 순차 이관:

| | 디렉토리 | 대상 | 상태 |
|---|---|---:|---|
| **M1** | `config/` | 33 | ✅ **완료** — 해시 대조 33/33 |
| **M2** | `module/` | 96 | ✅ **완료** — 해시 대조 96/96 |
| **M3** | `build/` | 42 | ⬜ **다음** |
| **M4** | `deploy/` | 9 | ⬜ |
| **M5** | `tools/` | 147 | ⬜ |

이관이 끝나면 계획 §6의 P1~P8.

## Done

| 날짜 | 내용 |
|---|---|
| 2026-07-30 | **M2** `module/` 이관 — 96/96 해시 대조, 미결 5건 판정, 테스트 426 불변 |
| 2026-07-29 | **M1** `config/` 이관 — 33/33 해시 대조, 테스트 426 불변 |
| 2026-07-29 | **P0** 이관 대장 625행 전수 판정 + 문서 데이터 11개 구멍 발견·보완 |
| 2026-07-29 | 문서 이름 규약 적용 — 87개 개명, 링크 126개 재작성, 중복·스텁 4개 삭제 |
| 2026-07-29 | 큰 캠페인 디렉토리 분할 (최대 34 → 12항목) |
| 2026-07-29 | 구 트리 → `archive/hardware/` 이동, 새 골격 생성 (테스트 49/426 불변) |
| 2026-07-29 | 재구성 계획 — 목표 구조 · 생사 대장 · docs 잠금 해제 순서 |
| 2026-07-29 | 전수 semantic census — 642 파일 · 285,261줄 |

## 이관 규칙

- **archive는 읽기 전용 기준입니다.** 수정하지 않습니다.
- 각 이관은 **archive의 무엇을 옮겼는지** 추적 가능해야 합니다.
- **`archive/hardware/tests`의 426 passed가 회귀 기준선입니다.** 이관 중 새 트리에서
  같은 테스트를 돌렸을 때 이 숫자가 줄면 멈춥니다.
