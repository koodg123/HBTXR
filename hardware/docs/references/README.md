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

## `vit-accel/` — 30개

`ViT_Accel` 프로젝트의 **문서 세트 전체**입니다. 자체 README·PROJECT-STRUCTURE·
BOARD-RUNBOOKS·SRC-CASE-* 를 갖춘 독립 프로젝트이고, HBTXR 가속기의 직접 선행입니다.

```
vit-accel/
├── README.md  PROJECT-STRUCTURE.md  LOCAL-SETUP.md      시작점
├── PIPELINE-OVERVIEW.md  FUNCTION-CALL-STACK.md          설계
├── SRC-CASE-{MODULE-GUIDE,PARALLELISM-TUNING}.md
├── BOARD-RUNBOOKS.md  DEPLOYMENT-RUN-SCRIPTS.md  VIVADO-ONLY-FLOW.md
├── VCK190-DOCKER-BASELINE.md  VCK190-ZU15EG-PARAMETER-COMPARISON.md
├── MULTI-BOARD-VALIDATION.md  CHECKLIST.md  MERGE-STATUS.md
├── HISTORY.md  PLAN.md  2026-04-15-progress.md
└── experiments/                                          ★ 남의 실험 결과
    ├── pnr/                    DeiT-Tiny 다중 보드 P&R (2026-04-08 · 04-10)
    ├── deit-tiny-baseline{,-ooc-rerun}/  step2 HLS 자원 요약
    └── {vck190,zu15eg}/                  보드별 P&R·자원
```

## 규칙 — 이름은 우리 것, 내용은 그쪽 것

| | |
|---|---|
| **파일·디렉토리 이름** | 우리 규약을 따릅니다 (2026-07-29 적용, 아래 대조표) |
| **문서 내용** | **편집하지 않습니다.** 남의 프로젝트 기록입니다 |
| **링크 대상·표시명** | 개명에 맞춰 재작성합니다 — 안 하면 클릭이 깨집니다 |
| **본문 산문의 경로** | 그대로 둡니다. `docs/SRC_CASE_MODULE_GUIDE.md` 같은 것은 **그쪽 저장소 기준** 서술이지 우리 트리 경로가 아닙니다 |
| `../workspace/hardware/src/*.h` 링크 | 깨진 채로 둡니다. 원본 프로젝트의 소스 트리이고 이관 대상이 아니었습니다 |

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
| `deit_tiny_baseline/full_deit_tiny_fit/` | `deit-tiny-baseline/full-deit-tiny-fit/` |
| `step2_hls_resource_summary.md` (3) | `step2-hls-resource-summary.md` |

**디렉토리 계층은 건드리지 않았습니다.** `experiments/`를 우리 캠페인 형식
(`YYYY-MM-DD-slug/`)으로 재배치하면 그건 개명이 아니라 재구성이고, 그쪽 프로젝트가
실험을 어떻게 조직했는지가 사라집니다.

## 삭제된 것 — 리다이렉트 스텁 3개 (2026-07-29)

`SRC_CASE_{ANALYSIS,DIAGRAMS,MICROARCH}.md`. 셋 다 15~17줄이었고 자기 본문에서
리다이렉트라고 밝히고 있었습니다 — 예: *"이 문서는 index 역할만 담당하며, 실제 요약
다이어그램은 통합 가이드 안에 포함되어 있다."* 내용은
[vit-accel/SRC-CASE-MODULE-GUIDE.md](vit-accel/SRC-CASE-MODULE-GUIDE.md)와
승격본 [MODULE-GUIDE.md](../MODULE-GUIDE.md)에 그대로 있습니다.

`vit-accel/README.md`의 해당 항목은 취소선과 함께 남겨 뒀고, `HISTORY.md`·`MERGE-STATUS.md`의
백틱 언급은 **그쪽 저장소에 그 파일이 있었다는 기록**이므로 그대로입니다.

출처: `archive/hardware/docs/references/vit_accel/` (2026-07-29 복사).
