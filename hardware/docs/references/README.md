> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active
> **소유** hardware

# references — 선행·자매 프로젝트 문서

**우리 코드가 바뀌어도 틀려지지 않는 문서**를 둡니다. 그래서 `reports/`나 `experiments/`가
아니라 여기입니다 ([DOC-CONVENTIONS 배치 규칙](../../../docs/governance/DOC-CONVENTIONS.md)).

## 저장소 수준 `docs/reference/` 와 무엇이 다른가

| | 담는 것 |
|---|---|
| `docs/reference/external/` (저장소) | 남의 논문·코드베이스 **서베이** — 읽고 요약한 것 |
| `hardware/docs/references/` (여기) | **선행/자매 프로젝트의 문서 세트 전체** — 우리 설계의 직접 조상 |

> 저장소 쪽은 아직 단수 `reference/`입니다. 복수로 통일하기로 했고 `docs/` 정리 때
> 개명합니다 ([규약 §11](../../../docs/governance/DOC-CONVENTIONS.md)).

## `vit-accel/` — 30개 · 6,943줄

`ViT_Accel` 프로젝트의 **문서 세트 전체**입니다. HBTXR 가속기의 직접 선행이고,
`hardware/docs/MODULE-GUIDE.md`가 여기서 파생했습니다.

### 무엇을 알고 싶을 때 무엇을 읽나

**이 표가 진입점입니다.** 6,943줄을 다 읽을 일은 없습니다.

| 알고 싶은 것 | 읽을 것 | 줄 |
|---|---|---:|
| **모듈이 어떻게 도는가** | [vit-accel/SRC-CASE-MODULE-GUIDE.md](vit-accel/SRC-CASE-MODULE-GUIDE.md) — 최대 문서 | 1,273 |
| 병렬도 knob를 어떤 순서로 만지나 | [SRC-CASE-PARALLELISM-TUNING.md](vit-accel/SRC-CASE-PARALLELISM-TUNING.md) | 488 |
| 보드에서 어떻게 빌드·실행하나 | [BOARD-RUNBOOKS.md](vit-accel/BOARD-RUNBOOKS.md) · [DEPLOYMENT-RUN-SCRIPTS.md](vit-accel/DEPLOYMENT-RUN-SCRIPTS.md) | 574 · 547 |
| Vivado만으로 도는 흐름 | [VIVADO-ONLY-FLOW.md](vit-accel/VIVADO-ONLY-FLOW.md) · [VCK190-DOCKER-BASELINE.md](vit-accel/VCK190-DOCKER-BASELINE.md) | 281 · 374 |
| VCK190 vs ZU15EG 파라미터 차이 | [VCK190-ZU15EG-PARAMETER-COMPARISON.md](vit-accel/VCK190-ZU15EG-PARAMETER-COMPARISON.md) | 348 |
| 함수 호출 관계 · 파이프라인 | [FUNCTION-CALL-STACK.md](vit-accel/FUNCTION-CALL-STACK.md) · [PIPELINE-OVERVIEW.md](vit-accel/PIPELINE-OVERVIEW.md) | 328 · 261 |
| 다중 보드 검증 결과 | [MULTI-BOARD-VALIDATION.md](vit-accel/MULTI-BOARD-VALIDATION.md) | 242 |
| 학습·평가·지표 | [TRAINING-EVALUATION-AND-METRICS.md](vit-accel/TRAINING-EVALUATION-AND-METRICS.md) | 176 |
| 그쪽 저장소 구조 · 로컬 셋업 | [PROJECT-STRUCTURE.md](vit-accel/PROJECT-STRUCTURE.md) · [LOCAL-SETUP.md](vit-accel/LOCAL-SETUP.md) | 213 · 204 |
| 보드별 P&R 실측 (2026-04-08·04-10) | [experiments/](vit-accel/experiments/) — 10개 | 800 |

**읽지 않아도 되는 것**: `CHECKLIST.md` · `MERGE-STATUS.md` · `2026-04-15-progress.md` ·
`PLAN.md` · `HISTORY.md`. **남의 프로젝트의 2026-04 시점 상태**이고 영구히 낡아 있습니다.
우리 상태는 [../STATUS.md](../STATUS.md)입니다.

```
vit-accel/
├── README.md  PROJECT-STRUCTURE.md  LOCAL-SETUP.md      시작점
├── PIPELINE-OVERVIEW.md  FUNCTION-CALL-STACK.md          설계
├── TRAINING-EVALUATION-AND-METRICS.md                    학습·평가·지표
├── SRC-CASE-{MODULE-GUIDE,PARALLELISM-TUNING}.md
├── BOARD-RUNBOOKS.md  DEPLOYMENT-RUN-SCRIPTS.md  VIVADO-ONLY-FLOW.md
├── VCK190-DOCKER-BASELINE.md  VCK190-ZU15EG-PARAMETER-COMPARISON.md
├── MULTI-BOARD-VALIDATION.md  CHECKLIST.md  MERGE-STATUS.md
├── HISTORY.md  PLAN.md  2026-04-15-progress.md        ← 그쪽 2026-04 상태
└── experiments/  (평면 10개)                          ★ 남의 실험 결과
    ├── 2026-04-08-*-pnr.md  2026-04-10-*-pnr.md       DeiT-Tiny 다중 보드 P&R
    └── {deit-tiny-baseline,zu15eg,vck190}-*.md        step2 HLS 자원 · step5 P&R
```

### 우리 트리와의 관계

| 우리 문서 | 여기 원본 | 왜 |
|---|---|---|
| [../MODULE-GUIDE.md](../MODULE-GUIDE.md) | `SRC-CASE-MODULE-GUIDE.md` | **우리가 갱신할 문서**라 우리 트리에 둡니다. `hgtxr_e2e_vit.hpp` 4,503줄을 분해하면 모듈 구성이 갈라지고 그때 고치는 쪽이 우리 사본입니다 |
| ~~`reports/2026-06-16-multi-board-validation.md`~~ | `MULTI-BOARD-VALIDATION.md` | **2026-07-29 삭제.** `ViT_Accel`→`XR_Accel` 치환본이었고 우리가 한 검증이 아니었습니다 |
| ~~`plans/done/2026-07-15-xr-accel-execution-plan.md`~~ | `PLAN.md` | **2026-07-29 삭제.** 저장소 이름 3곳만 치환. 실행 기록 없음 |

## 규칙 — 이름은 우리 것, 내용은 그쪽 것

| | |
|---|---|
| **파일·디렉토리 이름** | 우리 규약을 따릅니다 (2026-07-29 적용, 아래 대조표) |
| **문서 내용** | **편집하지 않습니다.** 남의 프로젝트 기록입니다 |
| **링크 대상·표시명** | 개명에 맞춰 재작성합니다 — 안 하면 클릭이 깨집니다 |
| **본문 산문의 경로** | 그대로 둡니다. `docs/SRC_CASE_MODULE_GUIDE.md` 같은 것은 **그쪽 저장소 기준** 서술이지 우리 트리 경로가 아닙니다 |
| `../workspace/hardware/src/*.h` | **백틱으로 강등**(2026-07-29). 원본 프로젝트 소스라 이관 대상이 아니었고, 클릭하면 죽는 링크로 두는 것보다 낫습니다. 텍스트는 그대로 |

**원본과 대조하려면 `archive/hardware/docs/references/vit_accel/` 를 보십시오.**
개명 전 이름 그대로 33개가 있습니다. 여기가 참조 자료로서의 대조 기준입니다 —
이름을 우리 규약으로 바꾼 대가로, 그 역할이 `archive/`로 넘어갔습니다.

### 개명 대조표 (2026-07-29)

디렉토리 9개 + 파일 25개. 규칙은 [규약 §3](../../../docs/governance/DOC-CONVENTIONS.md)
그대로입니다 — 역할 이름은 `UPPER-KEBAB.md`, 기록은 `YYYY-MM-DD-lower-kebab.md`.

| 개명 전 | 개명 후 |
|---|---|
| `vit_accel/` | `vit-accel/` |
| `BOARD_RUNBOOKS.md` 외 UPPER_SNAKE 12개 | `BOARD-RUNBOOKS.md` — 언더바만 하이픈으로 |
| `PROGRESS_2026_04_15.md` | `2026-04-15-progress.md` — 날짜가 접미사→접두사 |
| `experiments/pnr/*_20260410.md` (6) | `experiments/pnr/2026-04-10-*.md` — `20260410`은 규약 밖 |
| `deit_tiny_baseline/full_deit_tiny_fit/step2_…` | `deit-tiny-baseline-step2-…` — 평탄화, 아래 |

### `experiments/` 평탄화 (2026-07-29)

파일 **10개가 디렉토리 8개에 4단계**로 흩어져 있었고, 그중 4개는 디렉토리 하나에 파일
하나였습니다 — `experiments/vck190/deit-tiny-baseline/full-deit-tiny-fit/step5-….md`.
경로가 담던 정보를 파일명으로 옮겨 **디렉토리 8 → 1**로 만들었습니다.
**원래 경로 전체는 [experiments/README.md](vit-accel/experiments/README.md) 표에 보존**돼
있고, 그쪽 원문 설명도 그대로 아래 붙어 있습니다.

> **이건 "계층은 재배치하지 않는다"([규약 §11](../../../docs/governance/DOC-CONVENTIONS.md))의
> 예외입니다.** 그 규칙의 취지는 *조직 방식 자체가 정보다*인데, 파일 1개짜리 4단계 경로는
> 조직이 아니라 빌드 산출물 경로(`workspace/artifacts/reports/`)가 그대로 굳은 것입니다 —
> 그쪽 README가 스스로 그렇게 적고 있습니다. 조직 정보는 대조표로 보존됩니다.

**우리 캠페인 형식(`YYYY-MM-DD-slug/`)으로는 옮기지 않았습니다.** 남의 실험을 우리
`experiments/` 규격에 맞추면 우리가 돌린 것처럼 보입니다.

## 삭제된 것 — 리다이렉트 스텁 3개 (2026-07-29)

`SRC_CASE_{ANALYSIS,DIAGRAMS,MICROARCH}.md`. 셋 다 15~17줄이었고 자기 본문에서
리다이렉트라고 밝히고 있었습니다 — 예: *"이 문서는 index 역할만 담당하며, 실제 요약
다이어그램은 통합 가이드 안에 포함되어 있다."* 내용은
[vit-accel/SRC-CASE-MODULE-GUIDE.md](vit-accel/SRC-CASE-MODULE-GUIDE.md)와
승격본 [MODULE-GUIDE.md](../MODULE-GUIDE.md)에 그대로 있습니다.

`vit-accel/README.md`의 해당 항목은 취소선과 함께 남겨 뒀고, `HISTORY.md`·`MERGE-STATUS.md`의
백틱 언급은 **그쪽 저장소에 그 파일이 있었다는 기록**이므로 그대로입니다.

출처: `archive/hardware/docs/references/vit_accel/` (2026-07-29 복사).
