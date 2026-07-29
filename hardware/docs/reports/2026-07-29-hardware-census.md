> **작성** 2026-07-29 · **갱신** 2026-07-29
> **상태** active — 재작성 착수 전 기준선
> **소유** hardware

# hardware/ 전수 semantic 검사 — 분석

원자료: [snapshots/2026-07-29-semantic-census/](../snapshots/2026-07-29-semantic-census/)
(커밋 `8a9093e`, committed blobs 기준)

목적: **Refactoring / Rewrite / Cleanup / Reconstruction 착수 전 기준선.**
2026-07-15의 algorithm 트리 census와 **같은 도구·같은 스키마**이므로 두 서브시스템을
같은 잣대로 비교할 수 있습니다.

---

## 1. 규모

| | |
|---|---:|
| 추적 파일 | **642** |
| 총 라인 | **285,261** |
| Python 심볼 | **1,873** (함수 1,784 · 클래스 89) |
| C++/Tcl 선언 후보 | **545** |
| import 간선 | 1,247 |

| 언어 | 파일 |
|---|---:|
| Python | 157 |
| Markdown | 156 |
| JSON | 139 |
| C++ / 헤더 | 76 |
| Tcl | 31 |
| Shell · YAML · 기타 | 83 |

**JSON 139개가 파일의 22%입니다.** 최대 파일 상위 10개 중 9개가 JSON이고, 그중
`hls_modules.json` 3개만 67,428줄입니다 — 생성된 리포트가 소스와 같은 트리에 섞여 있습니다.

---

## 2. 디렉토리 — **설계된 구조와 실제 구조가 다릅니다**

### 비어 있는 설계 구조 24개

`.gitkeep` 하나씩만 든 디렉토리:

```
archive/{deprecated,old-runs}          artifacts/{bitstreams,hwh,manifests,reports,smoke}
docs/{decisions,experiments}           experiments/{active,blocked,completed,templates}
scripts/{build,maintenance,validate}   src/{hls,host,rtl}
tests/{fixtures,hls,integration,pynq,unit}
```

**누군가 이 구조를 설계했고 아무것도 들어가지 않았습니다.** 그리고 그 구조는 지금 우리가
원하는 것과 거의 같습니다 — `experiments/{active,blocked,completed,templates}`는
연구용 실험 관리 구조 그 자체입니다.

### 실제 코드가 있는 곳 (전부 평면)

| 디렉토리 | 파일 | 실체 |
|---|---:|---|
| `tools/` | 236 (py 87) | **평면 스크립트 87개** |
| `tests/` | 126 (py 60) | **평면 테스트 60개** — `tests/{unit,integration,hls,pynq}`는 비어 있음 |
| `hls/` | 67 | `include/` 19 · `src/` 19 · `tb/` 29 — **`src/hls`는 비어 있음** |
| `vivado/` | 31 | tcl — **`scripts/build`는 비어 있음** |
| `configs/` | 33 | |
| `pynq/` | 28 (py 9) | 보드 런타임 |
| `refs/` | 29 | 가중치·골든 매니페스트 |
| `docs/` | 267 | 별도 분석: [DOC-CONVENTIONS §11](../../../docs/governance/DOC-CONVENTIONS.md) |

**두 구조가 공존합니다** — 비어 있는 설계본과 평면인 실제본. 재작성의 첫 결정은
"설계본을 채울 것인가, 폐기할 것인가"입니다.

---

## 3. `hls/` — 연구 산출물

가속기 본체. **두 구현 계열이 공존합니다.**

### (a) 모듈별 `src/*.cpp` 19개 — 논문 구조를 그대로 반영

```
attention  mlp  nonlinear  matmul  fusion  noc  controller  runtime_fsm  rmu_smu
event_patch_embed  frame_patch_embed  search_head  track_head
global_buffer  weight_prefetcher
+ top 4개: hgtxr_top(955줄) · hgtxr_e2e_axis_top · hgtxr_e2e_m_axi_top · hgtxr_mode_profile_top
```

**이름이 논문 Fig와 대응됩니다.** 사람이 읽기 좋은 구조이고, 연구용 코드베이스의
출발점으로 적합합니다.

### (b) 템플릿 기반 `include/hgtxr_cyclic_*.hpp`

```
 4,503줄  hgtxr_e2e_vit.hpp            ← 최대
 2,723줄  hgtxr_cyclic_math.hpp        ← HG-PIPE 정수 LUT
   315줄  hgtxr_cyclic_transformer_params.hpp
   149줄  hgtxr_cyclic_s2_projection.hpp
   140줄  hgtxr_cyclic_scheduler.hpp
   114줄  hgtxr_cyclic_attention.hpp / weight_layout.hpp
   106줄  hgtxr_cyclic_transformer_block.hpp
   100줄  hgtxr_cyclic_norm.hpp
```

**두 헤더가 hls/ 전체 10,778줄의 67%를 차지합니다.** `hgtxr_e2e_vit.hpp` 4,503줄은
단일 파일로서는 사람이 따라가기 어려운 크기입니다.

### top 4개의 실제 사용도

| top | 참조 |
|---|---:|
| `hgtxr_e2e_axis_top` | **54** |
| `hgtxr_e2e_m_axi_top` | 32 |
| `hgtxr_top` | 17 |
| `hgtxr_mode_profile_top` | **3** |

`mode_profile_top`(507줄)은 3곳에서만 참조됩니다 — 폐기 후보이거나, 살릴 이유를 적어야
할 대상입니다.

### `tb/` 29개 중 17개가 골든 벡터 헤더

`e2e_axis_vector_*_golden.hpp` — **생성된 데이터**이지 테스트 코드가 아닙니다.
소스와 같은 디렉토리에 섞여 있어 "테스트벤치 29개"라는 인상을 주지만 실제 tb는 12개입니다.

---

## 4. `tools/` — 87개 평면 스크립트, 공유 계층 없음

### 상호 import 0건, `sys.path` 조작 66곳

```
tools 내부 상호 import :  0
sys.path 조작 파일     : 66 (tools + tests)
_bootstrap.py          : sys.path.insert 만 하는 shim, 5곳이 import
```

**패키지가 아닙니다.** 각 도구가 독립 스크립트이고, 서로를 쓰려면 `sys.path`를 조작하거나
(`import create_xr_vits_replacement_policy` 10곳) 코드를 복사합니다.

### 그래서 헬퍼가 복제됩니다

| 함수 | 복제 |
|---|---:|
| `load_json` | **22곳** |
| `sha256_file` | **11곳** |
| `normalize_roots` | **10곳** |
| `write_outputs` | 7곳 |
| `read_json` | 3곳 |

**중복 함수 그룹 39개.** 공용 유틸 모듈 하나가 없어서 생긴 것이며, 재작성에서 가장 값싸게
회수할 수 있는 항목입니다.

### 이름 패턴이 이미 분류를 말하고 있습니다

```
write_*    37    audit/report 생성
validate_*  9    검증
check_*     9    사전 점검
export_*    4    아티팩트 추출
run_*       3    파이프라인 실행
discover_*  2 · package_* 2 · update_* 3 · audit_* 3 · 기타 3
```

87개를 평면으로 두었을 뿐, **분류는 이름에 이미 들어 있습니다.**

---

## 5. 함수 크기 — 읽기 어려움의 실체

```
함수 1,784개 · 중앙값 12줄 · 평균 26.8 · p90 47
50줄 초과  158개
100줄 초과  62개
200줄 초과  20개
```

중앙값 12줄은 건강합니다. 문제는 **꼬리**입니다:

| 줄 | 함수 | 파일 |
|---:|---|---|
| **1,954** | `build_consistency_checks` | `tools/write_final_evidence_manifest.py` |
| **1,719** | `run_pipeline` | `tools/run_third_goal_final_signoff.py` |
| **1,176** | `validate_handoff` | `tools/validate_final_operator_handoff.py` |
| 908 | `validate_bundle` | `tools/validate_final_signoff_bundle.py` |
| 764 | `make_tools` | `tests/test_run_third_goal_final_signoff.py` |
| 478 | `build_audit` | `tools/write_third_goal_current_audit.py` |

**1,954줄 함수 하나를 이해하려면 그 파일 전체를 읽어야 합니다.** 상위 4개가 전부
audit/signoff 계층이라는 점이 중요합니다 — 가속기 커널이 아니라 **주변 도구**입니다.

---

## 6. 테스트 — 49개 실패는 로직이 아니라 **머신 경로**입니다

```
hardware/tests : 475개 중 49 실패 / 426 통과   (문서 이동 전후 동일)
tools 87개 중 테스트 있음 59 / 없음 28
```

실패 원인을 확인했습니다:

```
[fail] Vitis HLS 2023.2: not found :: \tools\Xilinx\Vitis_HLS\2023.2\bin\vitis_hls
[fail] Vivado 2023.2: not found    :: \tools\Xilinx\Vivado\2023.2\bin\vivado
[warn] HGPIPE root: missing        :: .../XR-VIT/HGPIPE
       python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR
```

**감사 도구가 특정 개발자 머신의 레이아웃을 단언합니다** — Xilinx 설치 경로, 형제 저장소
`XR-VIT/HGPIPE`, 절대 경로 `/home/kjm26/...`. 다른 어떤 머신에서도 실패합니다.

실패 분포도 한 계층에 몰려 있습니다: `check_third_goal_preflight` 8건,
`write_*_audit` 계열이 대부분. **가속기 커널 테스트는 실패하지 않습니다.**

---

## 7. 문서 트리가 자기 툴체인에 고정돼 있습니다

`hardware/tools/write_spec_plan_conformance_audit.py`가 `hardware/docs/{Master-Plan,
Sub-Plan,Spec,Execution,Validation}.md`와 `docs/track/**`, `docs/resources/**`를 **경로로**
읽고, 테스트 10개 이상이 의존합니다. 그래서 2026-07-29 문서 재배치에서 `hardware/docs/`만
손대지 못했습니다 — 자세히는
[DOC-CONVENTIONS §11](../../../docs/governance/DOC-CONVENTIONS.md).

**도구가 문서를 grep 해서 spec 준수를 판정하는 구조**이므로, 문서를 고치면 감사가 깨지고
감사를 고치면 무엇을 감사하는지가 바뀝니다. 재작성에서 정면으로 다뤄야 할 결합입니다.

---

## 8. 사망 코드 후보 248개 — **삭제 근거가 아닙니다**

`tools` 241 · `scripts` 5 · `pynq` 2. 이름 참조 횟수 휴리스틱이고, 동적 디스패치와 CLI
진입점을 해석하지 못합니다. 87개가 전부 `argparse` 스크립트이므로 **참조 1회는 정상**입니다.

실제로 쓸모 있는 신호는 `ref=0` 한 건뿐입니다:
`run_command_with_input` (`tools/check_xsim_snapshot_smoke.py`).

---

## 9. 재작성이 풀어야 할 것 — 우선순위

| # | 문제 | 근거 | 규모 |
|---|---|---|---|
| **H1** | **감사 도구의 머신 결합** — 49 테스트 실패의 원인 | §6 | 중 |
| **H2** | **`tools/`에 공유 계층 없음** — 헬퍼 39그룹 복제, sys.path 66곳 | §4 | 중 |
| **H3** | **거대 함수** — 1,954 / 1,719 / 1,176줄 | §5 | 중 |
| **H4** | **빈 설계 구조 24개 vs 평면 실제** — 결정 필요 | §2 | 소(결정) |
| **H5** | **HLS 두 구현 계열 + top 4개** — 무엇이 정본인가 | §3 | **결정 필요** |
| **H6** | **생성물이 소스와 같은 트리** — JSON 139개, 골든 헤더 17개 | §1, §3 | 소 |
| **H7** | **문서-도구 결합** | §7 | 중 |
| **H8** | 테스트 없는 도구 28개 | §6 | 소 |

### 연구용 코드베이스라는 목표에 비춰

사용자가 원하는 것은 **실험·확장·분석이 쉽고 사람이 이해하기 쉬운** 코드베이스입니다.
census가 말하는 현재 상태는 그 반대에 가깝습니다:

- **실험하기 어려움** — `experiments/`가 비어 있고, 실행은 머신 경로를 단언하는 87개
  스크립트를 통합니다.
- **확장하기 어려움** — 공유 계층이 없어 새 도구는 `load_json`을 23번째로 복사합니다.
- **분석하기 어려움** — 결과 JSON이 소스와 섞여 있고, 어느 top이 정본인지 적힌 곳이 없습니다.
- **이해하기 어려움** — 1,954줄 함수와 4,503줄 헤더.

**단, 좋은 재료가 있습니다.** `hls/src/*.cpp` 19개는 논문 구조 그대로 이름 붙어 있고,
`tools/`의 이름 패턴은 이미 분류를 담고 있으며, 비어 있는 `experiments/{active,blocked,
completed,templates}`는 우리가 원하는 구조 그 자체입니다. **재작성은 새 구조를 발명하는
일이 아니라, 이미 암시된 구조를 실현하는 일입니다.**

---

## 10. 이 census가 답하지 못하는 것

- **HLS 두 계열 중 무엇이 정본인가** — 정적 참조 수만으로는 판정 불가. 빌드 스크립트와
  최근 실험 이력을 함께 봐야 합니다.
- **`hgtxr_mode_profile_top`을 버려도 되는가** — 참조 3곳이지만 실험 목적일 수 있습니다.
- **감사 도구 49건이 "고쳐야 할 것"인가 "환경만 갖추면 되는 것"인가** — 보드/Xilinx 접근
  가능한 환경에서 한 번 돌려봐야 압니다.
- **`refs/weights/*.json`(17,542줄 ×2)이 현재 설계와 일치하는가** — 내용 검증 미실시.

이 넷은 **사용자 결정 또는 환경**이 필요하며, census로는 결론 낼 수 없습니다.
