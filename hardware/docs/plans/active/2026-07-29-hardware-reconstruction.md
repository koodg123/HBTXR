> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 승인됨. **문서(§4) 완료 · 코드(§6 P0~P8) 미착수**
> **소유** hardware

# hardware/ 재구성 계획

근거: [2026-07-29-hardware-census.md](../../reports/2026-07-29-hardware-census.md) ·
원자료 [snapshots/2026-07-29-semantic-census/](../../../../archive/hardware/docs/snapshots/2026-07-29-semantic-census/)

목표: **실험·확장·분석이 쉽고 사람이 이해할 수 있는** 연구용 코드베이스.

해결할 세 가지 불만:
1. 뭐가 뭔지 모르겠다 → §2 목표 구조
2. 뭐가 살고 뭐가 죽었는지 모르겠다 → §3 생사 대장
3. docs가 지저분하다 → §4 (툴 결합을 먼저 끊어야 함)

---

## 1. 지금 왜 헷갈리는가 — 원인 셋

### 원인 A. 같은 역할이 여러 곳에 있습니다

| 역할 | 흩어진 곳 |
|---|---|
| **소스** | `hls/` (실제 67) · `src/{hls,host,rtl}` (**빈**) · `rtl/` (README만) |
| **스크립트** | `tools/` (87) · `scripts/{run,report}` (6) · `vivado/scripts/` (31) |
| **생성물** | `pynq/hgtxr/*.bit` (**9개**) · `reports/` (1) · `docs/analysis/no-board-results/` · `artifacts/*` (**빈**) |
| **골든/참조** | `refs/` (29) · `hls/tb/*_golden.hpp` (17) |

`src/hls`가 비어 있고 소스는 `hls/`에 있습니다. `artifacts/bitstreams`가 비어 있고
비트스트림 9개는 런타임 코드와 같은 디렉토리(`pynq/hgtxr/`)에 있습니다.

### 원인 B. 설계된 구조 24개가 비어 있습니다

`.gitkeep`만 든 디렉토리 24개 — 그중 `experiments/{active,blocked,completed,templates}`와
`tests/{unit,integration,hls,pynq,fixtures}`는 **우리가 원하는 구조 그 자체**입니다.

### 원인 C. 정본이 가장 읽기 어렵습니다

`hgtxr_e2e_vit.hpp` **4,503줄** 단일 헤더가 정본이고, 같은 일을 하는 106줄
`cyclic_transformer_block.hpp`는 비정본 경로가 씁니다. → §5

---

## 2. 목표 구조

> ### ⚠️ 아래 제안은 채택되지 않았습니다 (2026-07-29)
>
> 사용자가 **7개 디렉토리**로 지정했고 그것이 실제로 지어졌습니다:
>
> ```
> hardware/
> ├── config/     보드 config · 하드웨어 설계 파라미터
> ├── module/     HLS 소스 + 테스트벤치
> ├── build/      Vivado 통합 · 합성 · P&R · 비트스트림 export
> ├── deploy/     PYNQ · 호스트 런타임
> ├── tools/      스크립트 (구 tools/ 87 + tests/ 60)
> ├── docs/       ✅ 완료
> └── workspace/  작업 산출물 (gitignore, `release/`만 예외)
> ```
>
> **현재 레이아웃의 정의는 [hardware/README.md](../../../README.md)입니다.**
> 아래 §2 본문은 census 직후의 제안 기록으로 남겨 둡니다 — 무엇을 왜 제안했는지가
> §3 생사 대장과 §6 P0~P8의 근거이기 때문입니다. **목표로 읽지 마십시오.**
>
> 주요 차이: 제안의 `hls/`→`module/`, `runtime/`→`deploy/`, `configs/`→`config/`,
> `golden/`·`artifacts/`·`tests/`는 별도 최상위가 아니라 각각 `module/`·`workspace/`·
> `tools/` 안으로 들어갑니다.

**원칙: 한 역할은 한 디렉토리.** 새로 발명하지 않고 이미 있는 것을 채웁니다.

```
hardware/
├── hls/                    ⬅ 하드웨어 소스의 유일한 위치
│   ├── include/            템플릿 라이브러리 + 공통 헤더
│   ├── src/                top + 모듈 구현
│   └── tb/                 테스트벤치 (골든 데이터는 분리 → golden/)
├── runtime/                ⬅ 現 pynq/ — 오버레이·스모크 러너 (코드만)
├── build/                  ⬅ 現 vivado/scripts/ + scripts/run/ — 빌드 진입점 단일화
│   ├── hls/                csim·csynth tcl
│   ├── vivado/             비트스트림 tcl
│   └── no_board/           보드 없이 도는 실행 스크립트
├── tools/                  ⬅ 87개를 역할별로. 이름이 이미 분류를 말함
│   ├── _lib/               load_json·sha256_file·normalize_roots 공용화
│   ├── audit/              write_*_audit (37 중 대부분)
│   ├── validate/           validate_* (9)
│   ├── check/              check_* (9)
│   ├── package/            package_* export_* (6)
│   └── discover/           discover_* run_* update_* (8)
├── tests/                  ⬅ 빈 구조를 채움
│   ├── unit/  integration/  hls/  runtime/  fixtures/
├── configs/                보드·양자화·define 헤더
├── golden/                 ⬅ 現 refs/ + hls/tb/*_golden.hpp — 골든 벡터·가중치 매니페스트
├── artifacts/              ⬅ 생성물. .bit/.hwh/리포트. gitignore 정책 별도 결정
└── docs/                   §4
```

**사라지는 것**: `src/` (빈, `hls/`가 실체) · `rtl/` (README만) · `scripts/` (build로 흡수) ·
`archive/` (§3에서 판정) · `reports/` (artifacts로)

---

## 3. 생사 대장 — 근거와 함께

### ✅ 확실히 살아 있음

| 대상 | 근거 |
|---|---|
| `hls/include/hgtxr_e2e_vit.hpp` + `src/hgtxr_e2e_axis_top.cpp` | **정본**. 비트스트림 6개, master 빌드 스크립트, 정적 참조 54 |
| `hls/include/hgtxr_cyclic_*.hpp` (10) | `hgtxr_top`과 `tb_cyclic_*`가 사용. 순환 없는 의존 트리 |
| `hls/src/*.cpp` 15개 (attention·mlp·fusion·heads·noc·…) | `hgtxr_top.cpp`가 **실제로 호출**(호출 횟수 확인 완료) |
| `hls/src/hgtxr_mode_profile_top.cpp` + search/track_head + fusion | 비트스트림 **2개** 산출 (`mode_search_par32`, `mode_track_par32`) |
| `pynq/hgtxr/*.py` 9개 | AXIS/M-AXI 오버레이 + 스모크 러너 |
| `vivado/scripts/run_e2e_q4w8a_{csim,csynth}.tcl` | master. m_axi 변형이 여기서 regsub 파생 |
| `tools/` 87개 | 전부 argparse 진입점. 59개는 테스트도 있음 |

### ❌ 확실히 죽었음 — 삭제 가능

| 대상 | 근거 |
|---|---|
| `archive/migration-logs/=318.empty` `=332.empty` | **0바이트, 깨진 셸 리다이렉트 잔해** |
| `common.h`의 `matmul` / `layernorm_stage` / `softmax_stage` 선언 | `hgtxr_top.cpp` 호출 **0회**. 구현은 있으나 아무도 안 부름 |
| `tools/check_xsim_snapshot_smoke.py::run_command_with_input` | census 유일한 참조 0 심볼 |
| 빈 `.gitkeep` 디렉토리 중 §2가 채우지 않는 것 | `src/{hls,host,rtl}` `archive/{deprecated,old-runs}` `scripts/{build,maintenance,validate}` `docs/{decisions,experiments}` |

> `matmul.cpp` 자체는 남깁니다 — 선언이 죽었을 뿐 구현이 다른 곳에서 쓰일 수 있는지
> 컴파일러로 확인한 뒤 판단합니다. **census는 삭제 근거가 아닙니다.**

### ⚠️ 판정 보류 — 조사 필요 (결정 아님)

| 대상 | 무엇을 확인해야 하나 |
|---|---|
| `refs/weights/cyclic_weights_s2_block_*_manifest.json` (17,542줄 ×2 + 4,420) | `hgtxr_cyclic_weight_layout.hpp`와 레이아웃 대조 |
| `hardware/rtl/` | README는 "HLS 패키징 결과가 여기 온다"는데 실제로 온 적 없음. 정책인지 사문인지 |
| `docs/legacy/` (1개 370줄) | 내용 확인 |
| 테스트 없는 도구 28개 | 쓰이는지 / 일회성이었는지 |

---

## 4. `archive/hardware/docs/` — **툴 결합을 먼저 끊어야 합니다**

지저분한 건 맞지만 **먼저 옮기면 깨집니다.**

```python
# hardware/tools/write_spec_plan_conformance_audit.py
DOCS = {"master_plan": "docs/Master-Plan.md", "spec": "docs/Spec.md", ...}
current_doc_base = root/"hardware" if (root/"hardware"/"docs"/"Spec.md").exists() else root
```

도구가 문서를 **경로로 읽고 내용을 grep 해서 spec 준수를 판정**합니다. 테스트 10개 이상이
의존합니다. 그래서 2026-07-29 저장소 문서 재배치에서 `archive/hardware/docs/`만 손대지 못했습니다.

### 순서

**4-1. 결합을 설정으로 뺀다** (先)

```yaml
# configs/doc_contract.yaml
spec:        docs/SPEC.md
master_plan: docs/plans/done/2026-06-16-third-goal.md
progress:    docs/track/PROGRESS.md
...
```

도구는 이 파일을 읽습니다. **문서를 옮겨도 설정만 고치면 됩니다.** 감사가 무엇을
검사하는지도 한 곳에서 보입니다.

**4-2. 그 다음 재배치** — 저장소 `docs/`와 같은 규약
([DOC-CONVENTIONS](../../../../docs/governance/DOC-CONVENTIONS.md))

| 현재 | → |
|---|---|
| `{Master-Plan,Sub-Plan,Spec,Execution,Validation}.md` | `plans/done/2026-06-16-third-goal/` + `SPEC.md` 승격 |
| `THIRD_GOAL_*.md` (루트 2개) | `plans/done/` |
| `track/` 21개 중 날짜 박힌 12개 (`AQ2_*_2026_07_01` 등) | `experiments/` |
| `track/` 나머지 (PROGRESS·TODO·ADR·log·CONVERSATION·HANDOVER) | `track/` 유지 |
| `analysis/no-board-results/` `reports/` | `reports/` 통합 |
| `analysis/integrated-2026-06-26/` | **이미 `docs/reference/external/`로 이동 완료** |
| `resources/hgtxr_{final_evidence,handover_additional}/` 68개 | `snapshots/` (불변) |
| `references/vit_accel/experiments/` | `experiments/` (날짜 규약) |
| `status/xr_accel/` `validation/xr_accel/` `architecture/xr_accel/` | `STATUS.md` · `reports/` · `ARCHITECTURE.md`로 흡수 |
| `decisions/` `experiments/` (빈) | `track/ADR.md` · `experiments/`로 |

---

## 5. 정본이 가장 읽기 어려운 문제 — 세 가지 선택지

`hgtxr_e2e_vit.hpp` 4,503줄이 정본이고, 106줄 `cyclic_transformer_block.hpp`가 같은 일을
조립형으로 표현합니다. **`e2e_vit.hpp`는 그것을 0회 씁니다.**

| 선택지 | 내용 | 위험 | 비용 |
|---|---|---|---|
| **S1 그대로 두기** | 구조 정리만, HLS는 손대지 않음 | 0 | 소 |
| **S2 분해** | `e2e_vit.hpp`를 `cyclic_*` 조립 위로 재작성 | **높음** — 합성 결과(타이밍·자원)가 바뀔 수 있음 | 대 |
| **S3 문서화만** | 대응 관계를 표로 적고 코드는 유지 | 0 | 소 |

**S1 + S3을 권합니다.** HLS 합성 결과는 코드 형태에 민감하고, 검증할 보드가 없는
지금 S2는 되돌리기 어려운 위험입니다. 구조·도구·문서를 먼저 정리하고, 보드 접근이
생긴 뒤 S2를 별도 과제로 다룹니다.

---

## 6. 실행 순서

| 단계 | 내용 | 되돌리기 | 검증 |
|---|---|---|---|
| **P0** | 죽은 것 삭제 (`=318.empty` 등), 빈 디렉토리 정리 | 쉬움 | 테스트 49/426 유지 |
| **P1** | `tools/_lib/` 공용화 — `load_json`(22) `sha256_file`(11) `normalize_roots`(10) `write_outputs`(7) | 중간 | 테스트 통과 수 **증가하지 않아야 함**(동작 불변) |
| **P2** | `tools/` 역할별 재배치 + `sys.path` 조작 66곳 제거 (패키지화) | 중간 | 동일 |
| **P3** | **문서-도구 결합을 `configs/doc_contract.yaml`로** | 중간 | 감사 도구가 같은 판정 |
| **P4** | `archive/hardware/docs/` 재배치 (P3 이후에만) | 중간 | 링크 검사 + INDEX |
| **P5** | 소스/생성물 분리 — `hls/` 단일화, `artifacts/`로 비트스트림, `golden/` 분리 | 중간 | 빌드 스크립트 경로 갱신 |
| **P6** | `tests/` 하위 구조 채우기 (unit/integration/hls/runtime) | 쉬움 | 동일 |
| **P7** | 거대 함수 분해 — 1,954 / 1,719 / 1,176줄 | **큼** | 테스트 |
| **P8** | 환경 결합 제거 — Xilinx 경로·형제 저장소 단언을 설정으로 | 중간 | **49건 실패 감소 확인** |

**P0~P2가 가장 값싸고 효과가 큽니다.** P7은 마지막입니다 — 테스트가 49건 실패 중인
상태에서 거대 함수를 건드리면 회귀를 구분할 수 없습니다.

### 각 단계의 성공 기준

- **P1~P2**: `archive/hardware/tests` 통과 수가 **426에서 변하지 않을 것**. 늘어나면 동작을 바꾼
  것이고, 줄면 깨뜨린 것입니다.
- **P3~P4**: 감사 도구가 재배치 전후로 **같은 판정**을 낼 것.
- **P8**: 49건 중 몇 건이 환경 설정만으로 통과하는지 **수치로** 보고.

---

## 7. 이 계획이 전제하는 것

- **보드/Xilinx 접근 없음.** 따라서 합성 결과가 바뀌는 변경(S2, P7 일부)은 검증할 수
  없고 뒤로 미룹니다.
- **census는 삭제 근거가 아닙니다.** §3의 "확실히 죽었음"도 P0에서 컴파일/테스트로
  재확인한 뒤 지웁니다.
- **`archive/hardware/tests` 49건 실패는 기준선입니다.** 재구성 중 이 숫자가 늘면 즉시 멈춥니다.
