> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 승인됨. **문서(§4) 완료 · 코드(§6 P0~P8) 미착수**
> **소유** hardware

# hardware/ 재구성 계획

근거: [2026-07-29-hardware-census.md](../../reports/2026-07-29-hardware-census.md) ·
원자료 [snapshots/2026-07-29-semantic-census/](../../snapshots/2026-07-29-semantic-census/)

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
| `references/vit_accel/experiments/` | ~~`experiments/`~~ → **`references/vit-accel/experiments/` 유지**. 남의 실험을 우리 `experiments/`에 섞으면 우리가 돌린 것처럼 보입니다. 이름만 규약 적용 |
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

> ### P0 재정의 (2026-07-29) — "삭제"가 아니라 "대장"
>
> 원래 P0는 **죽은 것 삭제**였습니다. 그건 `hardware/`를 **제자리에서** 리팩터링한다는
> 전제로 쓰였고, 그 뒤 전체를 `archive/`로 옮기고 **새 트리를 채우는** 방식으로 바뀌면서
> 무효가 됐습니다. `archive/`는 읽기 전용 기준이라 거기서 지우지 않고, **안 옮기면 그만**입니다.
>
> **그런데 안 옮긴 것은 눈에 보이지 않습니다.** "판단해서 뺐다"와 "빠뜨렸다"가
> 구분되지 않습니다. 문서 이관 때 159개를 **내용 해시로 대조**한 이유가 이것이고, 실제로
> 그 대조가 파일명 충돌로 소실된 문서 1개를 잡아냈습니다.
>
> **그래서 P0는 삭제가 아니라 이관 대장(manifest)입니다.**
>
> 다만 "안 옮기면 끝"이 **통하지 않는 것도 있습니다** — 살아 있는 파일 **안**의 죽은 코드:
>
> | 판정 | 파일 통째? | 조치 |
> |---|---|---|
> | `=318.empty` `=332.empty` | ✅ | 안 옮김 |
> | 빈 `.gitkeep` 디렉토리 | ✅ | 안 만듦 |
> | `common.h`의 `matmul`/`layernorm_stage`/`softmax_stage` **선언** | ❌ | `common.h`는 살아 있음. **옮기면서** 3줄 제거 |
> | `check_xsim_snapshot_smoke.py::run_command_with_input` | ❌ | 파일은 살아 있고 함수만 죽음 |

### 실제 결정 대상은 864개가 아니라 466개입니다

```
864  archive/hardware 전체
-210  .pyc       컴파일 산물
-159  .md        이미 docs/로 이관 완료
- 29  .gitkeep   빈 디렉토리 표식
────
 466  py 157 · json 140 · tcl 31 · cpp 31 · hpp 28 · h 17 · csv 16 · sh 10 · hwh 9 · 기타
```

### P0 — 이관 대장

466건 각각에 `migrate` / `skip` / `undecided` + **근거 한 줄**. 산출물은
`hardware/docs/plans/active/2026-07-29-code-migration-manifest.md` 와
각 디렉토리 README의 체크리스트입니다. 함께 **판정 보류 4건**을 조사합니다
(§3 하단: `refs/weights/*.json` · `rtl/` · `docs/legacy/` · 테스트 없는 도구 28개).

**검증**: 이관이 끝나면 대장 ↔ 새 트리를 대조해 **`migrate`인데 없는 것 0건**,
**대장에 없는데 트리에 있는 것 0건**.

### M1~M5 — 디렉토리별 순차 이관

대장이 서면 **한 번에 한 디렉토리씩** 옮깁니다. 순서는 **의존이 적은 것부터**입니다.

| | 디렉토리 | archive에서 오는 것 | 왜 이 순서 |
|---|---|---|---|
| **M1** | `config/` | `configs/` 33 | 아무것도 참조하지 않음. 다른 것들이 여기를 참조 |
| **M2** | `module/` | `hls/` 67 + `refs/` 29 → `golden/` | 본체. 경로가 확정돼야 build가 가리킬 수 있음 |
| **M3** | `build/` | `vivado/scripts/` 31 + `scripts/` 16 | M2의 경로를 소비. **필수: tcl에 `-I ../golden` 추가** — M2의 tb/golden 분리로 무수식 include가 깨졌습니다. 그리고 `tb_cyclic_{head_attention,s2_projection,primitives}.cpp` 러너를 만드는 것이 가장 값싼 품질 개선입니다 ([분석](../../reports/2026-07-30-module-code-analysis.md)) |
| **M4** | `deploy/` | `pynq/*.py` 9 (비트스트림 `.bit`/`.hwh` 제외 → `workspace/`) | M2·M3와 독립이지만 보드 없이는 검증 불가 |
| **M5** | `tools/` | `tools/` 87 + `tests/` 60 | **마지막.** 87개가 전방위를 읽음(`generated` 339회·`docs` 208·`pynq` 206·`hls` 128). 대상 경로가 다 정해진 뒤라야 함 |

`workspace/`는 이관 대상이 없습니다 (gitignore, 산출물만).

**각 M단계의 검증**: `archive/hardware/tests` 통과 수가 **426에서 변하지 않을 것.**
늘면 동작을 바꾼 것이고, 줄면 깨뜨린 것입니다. 단계마다 커밋합니다.

### 이관 후 — P1~P8

| 단계 | 내용 | 되돌리기 | 검증 |
|---|---|---|---|
| **P1** | `tools/_lib/` 공용화 — `load_json`(**26**) `sha256_file`(**12**) `normalize_roots`(**11**) `write_outputs`(**15**) | 중간 | 통과 수 불변 |
| **P2** | `sys.path` 조작 **70곳** 제거 (패키지화) | 중간 | 동일 |
| **P3** | 문서-도구 결합을 `config/doc_contract.yaml`로 | 중간 | 감사 도구가 같은 판정 |
| **P4** | `archive/hardware/docs/` 재배치 (P3 이후에만) | 중간 | 링크 검사 + INDEX |
| **P5** | 소스/생성물 분리 — 비트스트림 → `workspace/`. **골든 분리는 M2에서 선취 완료** | 중간 | 빌드 스크립트 경로 갱신 |
| **P6** | `tools/tests/` 하위 구조 채우기 | 쉬움 | 동일 |
| **P7** | 거대 함수 분해 — `hgtxr_e2e_vit.hpp` **4,503줄** | **큼** | 테스트 |
| **P8** | 환경 결합 제거 — 절대 경로·형제 저장소 단언을 설정으로 | 중간 | **49건 실패 감소를 수치로** |

> **P1·P2의 숫자가 초안보다 큽니다.** 특히 `write_outputs`는 7 → **15**로 두 배였습니다.
> 착수 전에 다시 세십시오.

**P7은 마지막입니다** — 테스트가 49건 실패 중인 상태에서 거대 함수를 건드리면 회귀를
구분할 수 없습니다.

> **P8 착수 전 확인할 것**: 이동으로 도구들의 `parents[2]`가 `HBTXR` → `HBTXR/archive`로
> 바뀌었습니다. **개수가 49로 불변인 것만 측정했고 같은 49건인지는 검증하지 않았습니다.**
> 이동 때문에 새로 깨진 것이 섞여 있으면 P8의 목표치가 달라집니다.

---

## 7. 이 계획이 전제하는 것

- **보드/Xilinx 접근 없음.** 따라서 합성 결과가 바뀌는 변경(S2, P7 일부)은 검증할 수
  없고 뒤로 미룹니다.
- **census는 삭제 근거가 아닙니다.** §3의 "확실히 죽었음"도 P0에서 컴파일/테스트로
  재확인한 뒤 지웁니다.
- **`archive/hardware/tests` 49건 실패는 기준선입니다.** 재구성 중 이 숫자가 늘면 즉시 멈춥니다.
