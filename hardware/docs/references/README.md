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

## `vit_accel/` — 33개

`ViT_Accel` 프로젝트의 **문서 세트 전체**입니다. 자체 README·PROJECT_STRUCTURE·
BOARD_RUNBOOKS·SRC_CASE_* 를 갖춘 독립 프로젝트이고, HBTXR 가속기의 직접 선행입니다.

```
vit_accel/
├── README.md  PROJECT_STRUCTURE.md  LOCAL_SETUP.md      시작점
├── PIPELINE_OVERVIEW.md  FUNCTION_CALL_STACK.md          설계
├── SRC_CASE_{ANALYSIS,DIAGRAMS,MICROARCH,MODULE_GUIDE,PARALLELISM_TUNING}.md
├── BOARD_RUNBOOKS.md  DEPLOYMENT_RUN_SCRIPTS.md  VIVADO_ONLY_FLOW.md
├── VCK190_DOCKER_BASELINE.md  VCK190_ZU15EG_PARAMETER_COMPARISON.md
├── MULTI_BOARD_VALIDATION.md  CHECKLIST.md  MERGE_STATUS.md
├── HISTORY.md  PLAN.md  PROGRESS_2026_04_15.md
└── experiments/                                          ★ 남의 실험 결과
    ├── pnr/            DeiT-Tiny 다중 보드 P&R (2026-04-08 · 04-10)
    ├── deit_tiny_baseline{,_ooc_rerun}/  step2 HLS 자원 요약
    └── {vck190,zu15eg}/                  보드별 P&R·자원
```

## 규칙

- **내용을 편집하지 않습니다.** 남의 프로젝트 기록입니다. 안의 경로가 우리 트리에서
  깨져 보여도 그대로 둡니다 — `../workspace/hardware/src/*.h` 같은 링크는 원본 프로젝트
  기준입니다.
- `experiments/`도 **남의 실험**입니다. 우리 실험은 `hardware/docs/experiments/`로 갑니다.
- 출처: `archive/hardware/docs/references/vit_accel/` (2026-07-29 복사, 무변경).
