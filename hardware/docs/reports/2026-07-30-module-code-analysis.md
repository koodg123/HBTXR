> **작성** 2026-07-30 · **갱신** 2026-07-30
> **상태** active — 조사 결과. 조치는 "지금 가능"만 실행, 나머지는 목록
> **소유** hardware

# `module/` 코드 전수 분석 — 96 파일 · 12,175줄

5개 관점 병렬 분석 → **45건 제기 · 21 확정 · 8 기각 · 16 미검증**.

> **미검증 16건을 기각으로 섞지 않았습니다.** 세션 한도로 검증 에이전트가 중단됐습니다.
> 그중 가장 중요한 5건은 직접 소스를 읽어 확인했고, 아래 표시했습니다.

---

## 0. 가장 중요한 사실 — **컴파일할 수 없습니다**

Vitis HLS도 보드도 없습니다. 그래서 결론이 두 갈래입니다:

| | 근거로 충분한가 |
|---|---|
| **소스를 읽어 확인되는 것** | 참조 0건 · 타입 범위 불일치 · 어서션 없는 테스트벤치 · 빌드 스크립트 결선 |
| **컴파일러가 필요한 것** | 합성되는가 · QoR · 실제 정확도 영향 |

**전자만 조치했습니다.** 합성되는 코드를 검증 없이 고치면 깨졌는지 알 방법이 없습니다.

---

## 1. 규모의 실체 — 2개 파일이 59%

```
include/hgtxr_e2e_vit.hpp     4,503   ← 정본 top의 전부
include/hgtxr_cyclic_math.hpp 2,723
  ─────────────────────────  7,226 / 12,175 = 59%

나머지 17 include + 19 src + 12 tb =  4,949줄
그중 7~13줄 파일이 10개
```

"파일이 너무 많다"는 체감은 정확합니다. 다만 **줄 수 문제가 아니라 분포 문제**입니다 —
거대 파일 2개와 파편 10개가 공존하고, 중간이 비어 있습니다.

---

## 2. 실제로 안 쓰이는 것 — 직접 확인

### 2-1. 테스트벤치 12개 중 **6개만 빌드에 걸려 있습니다**

| tb 파일 | tcl 참조 | 어서션 | 판정 |
|---|---:|---|---|
| `tb_hgtxr_e2e_axis_top.cpp` | **5** | golden 비교 | 실질 게이트 |
| `tb_hgtxr_e2e_m_axi_top.cpp` | 3 | golden 비교 | 실질 게이트 |
| `tb_hgtxr_mode_profile_top.cpp` | 2 | 2건 (약함) | |
| `tb_hgtxr_top.cpp` | 1 | **0** | **아무것도 확인 안 함** |
| `tb_cyclic_s2_block_vector.cpp` (+`_hls_main`) | 1 | 파일 비교 | **입력 바이너리 부재로 실행 불가** |
| `tb_cyclic_head_attention.cpp` | **0** | 독립 참조모델 + 1e-4 + 음성대조 | ⚠️ **가장 좋은데 안 돌아감** |
| `tb_cyclic_s2_projection.cpp` | **0** | 다중 케이스 | ⚠️ 같음 |
| `tb_cyclic_primitives.cpp` | **0** | 있음 | ⚠️ 같음 |
| `tb_runtime_fsm.cpp` | 0 | 1건 | 약한 스모크 |
| `tb_mlp.cpp` · `tb_attention.cpp` | 0 | **0** | **테스트가 아님** |

**테스트벤치의 품질과 실행 여부가 반비례합니다.** 가장 잘 만든 셋이 안 돌고, 도는 것 중
하나(`tb_hgtxr_top.cpp`)는 아무것도 확인하지 않습니다.

```c
// tb_mlp.cpp 전문
int main() {
  static token_buffer_t tokens;   // 초기화 안 됨
  mlp_stage(tokens);              // 결과 미확인
  return 0;                       // DUT가 `{}` 여도 통과
}
```

**없는 것보다 나쁩니다** — 커버리지가 있는 것처럼 보이게 합니다.

### 2-2. cyclic 템플릿 라이브러리 8개 헤더가 **어떤 러너도 빌드하지 않습니다**

`src/hgtxr_top.cpp:57`의 `#if HGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP` 안에만 들어 있고,
그 매크로를 세우는 `run_cyclic_*.tcl` 9개를 **호출하는 셸 스크립트가 없습니다.**
실제로 도는 `run_hls_csim.sh`/`run_hls_csynth.sh`는 플래그를 넘기지 않아 기본값 0입니다.

대상: `hgtxr_cyclic_{transformer_block, attention, mlp, norm, mac, scheduler, s2_projection, weight_layout}.hpp`
(`hgtxr_cyclic_math.hpp`와 `_transformer_params.hpp`는 **아닙니다** — 정본 헤더가 씁니다.)

> **"증거 0"이라고 쓰지 않습니다.** 마지막 기록은 2026-06-08 수동 실행(csim·csynth 통과,
> `hgpipe_reference_analysis_2026_06_08.md:146`)이고, **자동으로 재현할 방법이 없다**는 것이
> 정확한 서술입니다.

### 2-3. `hgtxr_cyclic_math.hpp`의 LUT 48개 — 호출 0건, 그래도 지우면 안 됩니다

`hgtxr_hgpipe_attn{0..11}_{q,k,v,a}_quant64_int` (192~756행, 약 565줄). 정의 48개, 호출 0.
형제 계열(`_lnq_requant_int`, `_softmax_requant_uint3_int`)은 정본이 **호출합니다** —
즉 **LayerNorm·softmax에는 레이어별 캘리브레이션이 적용되고 Q/K/V·attention 출력에는 안 됩니다.**

**삭제 금지 이유**: `golden/hgpipe_lut_math_contract.json`이
`acceptance.must_match_hgtxr_header = true`로 이 테이블을 미러링하고
`validate_hgpipe_lut_math.py`가 강제합니다. 지우면 그 검사가 깨집니다.

---

## 3. HLS 적합성 — 직접 확인한 결함 셋

### 3-1. ⚠️ **csim 기본 데이터패스가 `float`입니다**

```c
// include/fixed_types.h
#if defined(__SYNTHESIS__) || defined(HGTXR_HLS_FIXED_CSIM)
  typedef ap_fixed<HGTXR_BIT_WIDTH, HGTXR_DATA_I> hgtxr_data_t;
#else
  typedef float hgtxr_data_t;      // ← csim 기본값
  typedef float hgtxr_acc_t;
#endif
```

`HGTXR_HLS_FIXED_CSIM`을 세우지 않으면 **csim은 float 참조모델을 검증합니다.**
"csim이 golden을 통과했다"는 기록은 **양자화 오차·포화·비트폭에 대해 아무것도 말하지 않습니다.**

이게 이 트리에서 가장 중요한 발견입니다. 성능·정확도 주장의 근거를 흔듭니다.

### 3-2. ⚠️ `quant.h`의 클램프가 배포 설정에서 **발동할 수 없습니다**

```c
if (x > (hgtxr_acc_t)31.0)  return  31.0;    // ap_fixed<16,6> 범위
if (x < (hgtxr_acc_t)-32.0) return -32.0;
```

그런데 배포 설정 `zcu104_e2e_q4w8a_defines.h:14`는 `HGTXR_BIT_WIDTH 8`이고,
`fixed_types.h:9-12`가 그때 `HGTXR_DATA_I 4`를 씁니다 → **`ap_fixed<8,4>`, 범위 ±8**.

**±8에서 이미 넘친 값을 ±31에서 막습니다.** 클램프는 죽은 코드이고, 타입 자체의
오버플로 동작(기본 wrap)이 먼저 일어납니다.

### 3-3. ⚠️ `hgtxr_data_to_axis()` — 8비트에서 부호확장 없음

```c
#if HGTXR_BIT_WIDTH >= 16
  packed.range(15, 0) = raw_bits.range(15, 0);
#else
  packed.range(HGTXR_BIT_WIDTH - 1, 0) = raw_bits.range(HGTXR_BIT_WIDTH - 1, 0);
  //  ← 상위 비트가 0으로 남습니다
#endif
```

호스트는 16비트 부호값으로 디코드합니다. `BIT_WIDTH=8`에서 음수 `0xF8`(-8)이
`0x00F8`(+248)로 읽힙니다. **모든 음수 출력이 틀립니다.**

### 3-4. 미검증 — 확인 필요

- `HGTXR_E2E_ACTIVE_TOKENS`가 64로 상한 걸려 active128/196 스윕이 무효라는 주장:
  **제가 확인하지 못했습니다.** `hgtxr_e2e_vit.hpp:18-19`의 기본값은 `HGTXR_TOKENS`(=256)이고
  64 상한을 찾지 못했습니다. 재조사 필요.
- 패치 임베딩이 AXIS가 쓰지 않은 `frame[][]` 열을 읽는다는 주장 — 미검증.
- `nonlinear.cpp`의 상수가 `hgtxr_data_t` 범위 초과 — 컴파일러 필요.

---

## 4. 제가 M2에서 만든 부채 — golden 분리의 대가

테스트벤치는 golden을 **상대 경로 무수식 include**로 참조합니다:

```c
#include "e2e_axis_vector_active8_golden.hpp"     // tb/ 기준으로 해석
```

M2에서 `tb/`와 `golden/`을 분리했으므로 **새 트리에서는 이 include가 실패합니다.**
archive는 둘이 같은 디렉토리라 영향 없고, 빌드는 아직 archive를 읽으니 지금은 무해합니다.

**분리 자체는 옳습니다**(데이터와 코드 분리). 다만 공짜가 아니었고,
**M3에서 tcl에 `-I ../golden`을 추가해 갚아야 합니다.** 계획에 적었습니다.

---

## 5. 조치 — 지금 한 것과 미룬 것

### 지금 실행 (컴파일러 불필요, 위험 0)

| | 대상 | 근거 |
|---|---|---|
| **삭제** | `tb/tb_mlp.cpp` · `tb/tb_attention.cpp` | 8줄, 어서션 0, **어떤 tcl·셸·py도 참조 안 함**(전수 확인). 항상 통과하므로 없는 것보다 나쁨 |
| **정정** | `module/README.md`의 top 표 | `hgtxr_mode_profile_top`은 **심볼이 아니라 파일명**입니다. 실제 top은 `hgtxr_search_profile_top`·`hgtxr_track_profile_top` 둘이고 `HGTXR_MODE_PROFILE_TOP` 환경변수가 고릅니다 |
| **기록** | 위 §2·§3·§4 전부 | README와 이 리포트 |

### 미룸 — 컴파일러 없이는 못 합니다

| 우선순위 | 대상 | 왜 미루나 |
|---|---|---|
| **1** | `fixed_types.h` float csim | 고치면 **기존 golden이 전부 실패할 수 있습니다.** 그게 정상이지만, 실패를 판정할 방법이 없으면 되돌릴 수도 없습니다 |
| **2** | `quant.h` 클램프 경계 | 비트폭에서 파생시켜야 하는데, 파생 후 수치가 바뀌는지 확인 불가 |
| **3** | `hgtxr_data_to_axis` 부호확장 | 호스트 디코더도 같이 봐야 합니다 (`deploy/`, M4) |
| **4** | LUT 48개 배선 여부 | 정확도 회귀 측정 필요 |
| **5** | 4,503줄 분해 (P7) | 합성 결과가 코드 형태에 민감 |
| — | `tb_hgtxr_top.cpp` 어서션 추가 | golden이 없어 무엇과 비교할지부터 정해야 함 |

### 살려야 할 것 — 지우는 게 아니라 **연결**

`tb_cyclic_head_attention.cpp` · `tb_cyclic_s2_projection.cpp` · `tb_cyclic_primitives.cpp`.
이 트리에서 **유일하게 제대로 된 테스트**이고 아무도 안 돌립니다.
**M3에서 `build/`에 러너를 만들 때 이 셋을 물리는 것이 가장 값싼 품질 개선입니다.**

---

## 6. 이 분석의 한계

1. **컴파일 불가** — 합성 가능성·QoR·정확도 영향은 전부 미확인.
2. **미검증 16건** — 세션 한도로 검증이 중단됐습니다. 위 §3-4가 그 일부입니다.
3. **안전 분류기 부재 6건** — 해당 검증 결과는 교차 확인 없이 채택하지 않았습니다.
4. 기각된 8건 중에는 **사실은 맞지만 진단이 틀린** 것이 여럿입니다 — 예: "tcl이 존재하지
   않는 `hardware/hls/`를 참조한다"는 사실이지만 **의도된 이관 중간 상태**입니다.
