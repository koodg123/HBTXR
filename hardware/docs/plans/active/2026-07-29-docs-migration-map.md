> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 분석 완료, 이관 대기
> **소유** hardware

# archive/hardware 문서 → 새 `hardware/docs/` 이관 지도

대상: `archive/hardware/**` 의 **md 159개 전수** (`docs/` 밖 3개 포함).
상위 계획: [2026-07-29-hardware-reconstruction.md](2026-07-29-hardware-reconstruction.md)

---

## 1. 가장 중요한 축 — 손으로 쓴 것 vs 도구가 생성한 것

| | 개수 | 판정 근거 |
|---|---:|---|
| **도구 생성** (`resources/**`) | **68** | 문서 안에 `root: /home/kjm26/project/PRJXR/XR-VIT/HGTXR` 가 박혀 있음. `write_*_audit.py` / `run_third_goal_final_signoff.py` 가 `.json` + `.md` 쌍으로 출력 |
| **선행 프로젝트 문서** (`references/vit_accel/**`) | **22** | `ViT_Accel` 이라는 **별도 프로젝트의 문서 세트 전체** (README·PROJECT_STRUCTURE·BOARD_RUNBOOKS·SRC_CASE_*) |
| **손으로 쓴 것** | **69** | 나머지 |

**이 축이 먼저입니다.** 도구 출력은 `docs/`가 아니라 **실험 결과**이고, 선행 프로젝트
문서는 우리 코드가 바꿀 수 없으므로 `references/`입니다.

### 중복 — `final_evidence` 40개 중 **36개가 `handover_additional`과 동일 파일명**

```
final_evidence/       40개
handover_additional/ 126개
겹치는 파일명         36개  (e2e_resource_matrix · final_blocker_closure_readiness ·
                            final_evidence_manifest · third_goal_* …)
```

**같은 감사 산출물을 두 번 보관하고 있습니다.** 이관 시 한 벌만 남깁니다.

---

## 2. 날짜·캠페인 분류

파일명에 날짜가 있는 것 **99개**, 없는 것 **60개**.

| 날짜 | 문서 | 캠페인 | 무슨 일이 있었나 |
|---|---:|---|---|
| 2026-04-15 | 1 | `vit_accel` | 선행 프로젝트 진행 기록 |
| 2026-04-08~10 | 6 | `vit_accel/pnr` | DeiT-Tiny 다중 보드 P&R (vck190·zcu102·zu15eg) |
| 2026-06-06 | 2 | **cyclic** | cyclic weight layout·packing 도입 |
| 2026-06-08 | 1 | hgpipe | HG-PIPE 참조 분석 |
| 2026-06-09 | 1 | e2e | AXIS 베이스라인 |
| **2026-06-10** | **44** | **final_signoff** | 최종 증거·서명 대량 생성 (`final_*` 22 · `third_goal_*` 5 · `e2e_*` 4) |
| 2026-06-12 | 1 | legacy | 레거시 실험 분석 |
| **2026-06-16** | **23** | **third_goal + vref_p0** | 3차 목표 완료 감사 · VREF-P0 successor · req1/5/6/9 감사 |
| 2026-06-17 | 1 | cleanup | |
| 2026-06-23 | 3 | **c3b** | C3b attention/QKV/MLP 자원·전력·지연 |
| 2026-06-26~27 | 7 | **no_board_p0** | 보드 없는 실험 계획·수집·mode profile |
| 2026-06-28 | 3 | **eight_question** | 8문항 Vivado 실험 · 하이브리드 런타임 |
| 2026-06-30~07-01 | 4 | **AQ2** | search/track 측정·체크리스트 · ZCU104 물리 실험 계획 |
| 2026-07-05 | 1 | session | 세션 진행·대화 기록 |
| 2026-07-08 | 1 | impl_repos | 구현 저장소 인계 |
| 2026-07-29 | 2 | (신규) | census · 재구성 계획 |

**캠페인이 여덟 개**입니다: `cyclic` · `third_goal` · `final_signoff` · `c3b` ·
`vref_p0` · `no_board_p0` · `eight_question` · `AQ2`.

---

## 3. 목표 매핑

### 3.1 `docs/experiments/<날짜>-<캠페인>/` — **도구 생성 결과 68개 + 실험 리포트**

날짜·캠페인별로 묶습니다. 각 디렉토리는 **불변**이며 `results.md` + 원본을 담습니다
([DOC-CONVENTIONS 규칙 E](../../../../docs/governance/DOC-CONVENTIONS.md)).

```
docs/experiments/
├── 2026-06-06-cyclic-weight-layout/        cyclic_weight_{layout,packing}
├── 2026-06-09-e2e-axis-baseline/           e2e_axis_baseline
├── 2026-06-10-final-signoff/               ★ 44개 — final_* · third_goal_* · e2e_resource_*
│   ├── README.md                           무엇을 실행했나 · 커밋 SHA · 재현 명령
│   └── ...                                 감사 .md/.json (중복 제거 후)
├── 2026-06-16-third-goal-audit/            ★ 23개 — req1/5/6/9 · xr_vits_gate · hgpipe_operator
├── 2026-06-16-vref-p0-successor/           vref_p0_* 6개 (pot_scale · buffer_lifetime · qkv_uram)
├── 2026-06-23-c3b-resource-power/          c3b_* 3개
├── 2026-06-26-no-board-p0/                 P0_PROGRESS · p0_report_collection
├── 2026-06-27-mode-profile/                MODE_PROFILE_RESULTS · VIVADO_PROGRESS · p0_mode_profile
├── 2026-06-28-eight-question/              EIGHT_QUESTION_* · RUNTIME_MODE_FULL_AXI
└── 2026-06-30-aq2-search-track/            AQ2_* 3개
```

**왜 `docs/`이지 `workspace/`가 아닌가**: 이들은 *해석된 결과*(표·결론 포함)이고 이미
커밋되어 있습니다. 앞으로 생성되는 raw 출력은 `workspace/`로 가고, 승격된 것만
`experiments/`에 들어옵니다.

### 3.2 `docs/references/` — 선행/자매 프로젝트 22개

```
docs/references/vit_accel/       ← archive/hardware/docs/references/vit_accel/ 그대로
```

`ViT_Accel`은 **별도 프로젝트**입니다. 우리 코드가 바뀌어도 이 문서는 틀려지지 않으므로
`references/`가 맞습니다. 저장소 수준 `docs/reference/external/`(외부 서베이 140개)과는
다릅니다 — 이쪽은 **직접 선행 프로젝트의 전체 문서 세트**입니다.

단, 그 안의 `experiments/`(P&R 결과 10개)는 `references/vit_accel/experiments/`에
그대로 둡니다. 남의 실험이지 우리 실험이 아닙니다.

### 3.3 `docs/track/` — 6종만

| archive | → |
|---|---|
| `track/{PROGRESS,TODO,ADR,log,CONVERSATION}.md` | `track/` 그대로 |
| `track/HANDOVER.md` `HANDOVER_2026_06_27.md` | `docs/handoff/` (일회성, 만료 고지) |
| `track/{AQ2_*,EIGHT_QUESTION_*,RUNTIME_MODE_*,ZCU104_*,CLEANUP_*,SESSION_*,WORK_PROGRESS_*,IMPL_REPOS_*}` (12) | **`docs/experiments/`** — 날짜 박힌 실험 기록이 track에 섞여 있었음 |

**`track/` 21개 중 12개가 실험 기록이었습니다.** 그래서 track이 지저분했습니다.

### 3.4 `docs/plans/` — 계획

| archive | → |
|---|---|
| `Master-Plan.md` `Sub-Plan.md` `Execution.md` `Validation.md` | `plans/done/2026-06-16-third-goal/` |
| `Spec.md` | **`docs/SPEC.md`로 승격** (서브시스템 계약, Scope 섹션 포함) |
| `THIRD_GOAL_{FUTURE_EXPERIMENTS,PROGRESS_EXPERIMENT_REPORT}_2026_06_16.md` | `plans/done/` + `experiments/` 분리 |
| `analysis/NO_BOARD_PERFORMANCE_EXPERIMENT_PLAN_2026_06_26.md` | `experiments/2026-06-26-no-board-p0/plan.md` |
| `status/xr_accel/PLAN.md` `ZCU104_CYCLIC_MAXPERF_*` | `plans/` + `experiments/` |

### 3.5 `docs/reports/` — 분석 (개정 가능)

| archive | → |
|---|---|
| `reports/c3b_*` 3개 | `experiments/2026-06-23-c3b-resource-power/` (결과이므로) |
| `legacy/legacy_experiment_analysis_2026_06_12.md` | `reports/2026-06-12-legacy-experiment-analysis.md` |
| `analysis/no-board-results/*.md` | `experiments/` |
| 신규 `2026-07-29-hardware-census.md` | `reports/` 유지 |

### 3.6 `docs/` 루트 — 소수만

| archive | → | 이유 |
|---|---|---|
| `architecture/DIRECTORY_LAYOUT.md` · `architecture/xr_accel/HBTXR_{ARCH_FREEZE,CYCLIC_IMPLEMENTATION}.md` | **`docs/ARCHITECTURE.md`로 통합** | Spec 역할 |
| `Spec.md` | `docs/SPEC.md` | |
| `STATUS.md` | 신규 `docs/STATUS.md`가 대체 | |
| `CHOICE.md` | `track/ADR.md`로 흡수 | 결정 기록 |
| `SRC_CASE_MODULE_GUIDE.md` | `references/vit_accel/`에 동명 파일 존재 → **중복 확인 후 하나만** | |
| `HARDWARE_IMPORT_MANIFEST.md` | `docs/snapshots/` | 이관 시점 기록 |

### 3.7 `docs/` 밖 3개

| archive | → |
|---|---|
| `README.md` | 신규 `hardware/README.md`가 대체 (이미 작성) |
| `pynq/hgtxr/README.md` | `deploy/README.md`에 흡수 |
| `rtl/README.md` | **폐기** — "여기 온다"는데 온 적 없음 (census §3 확인) |

---

## 4. 이관 순서

| 단계 | 내용 | 규모 |
|---|---|---|
| **D1** | `references/vit_accel/` 그대로 복사 — 손댈 것 없음 | 22 |
| **D2** | `experiments/` 캠페인 10개 생성 + 도구 출력 배치 (**중복 36개 제거**) | 68 → 32 |
| **D3** | `track/` 6종만 남기고 실험 12개를 `experiments/`로 | 21 → 6 |
| **D4** | `plans/done/` + `SPEC.md` + `ARCHITECTURE.md` 승격 | 12 |
| **D5** | `reports/` · `snapshots/` · `handoff/` 잔여 | 8 |

**D1이 가장 안전합니다** — 남의 문서라 내용 판단이 필요 없습니다.
**D2가 가장 값이 큽니다** — 68개가 32개로 줄고, 8개 캠페인이 날짜별로 정렬됩니다.

---

## 5. 이관하지 않는 것

- `resources/**` 의 `.json` 원본 — `.md`와 쌍이고 도구가 재생성 가능. **`workspace/`로.**
  단 재생성에 Xilinx 환경이 필요하므로, 실제 삭제는 환경 확보 후 판단합니다.
- `final_evidence/` ↔ `handover_additional/` **중복 36건** — 한 벌만.
- 문서 안의 `/home/kjm26/...` 절대 경로 — 이관 시 그대로 둡니다. **당시 기록이고,
  고쳐 쓰면 그 실험이 어디서 돌았는지가 사라집니다** (규칙 G).

---

## 6. 이 지도가 답하지 못하는 것

- **`SRC_CASE_MODULE_GUIDE.md` 2벌이 같은 내용인가** — `docs/` 루트와
  `references/vit_accel/` 양쪽에 있습니다. 내용 대조 필요.
- **`resources/` 68개 중 실제로 참조 가치가 있는 것이 몇 개인가** — 전부 도구 출력이고
  대부분 `status: blocked`입니다. 캠페인별 대표 문서만 남기는 안도 가능하나, 그건
  **정보 삭제**이므로 별도 승인이 필요합니다.
