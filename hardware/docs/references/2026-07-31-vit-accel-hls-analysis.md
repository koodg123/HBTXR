> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active — HBTXR 구현의 기법 참조
> **소유** hardware

# ViT_Accel HLS 참조 구현 분석

대상: `backup/ViT_Accel/workspace/hardware/{src,case}` — 헤더 15개 4,607줄 + 생성 인스턴스 1,049 파일.
HG-PIPE 계열이고, **HBTXR이 기법을 가져올 원본**입니다.

> 5개 관점 병렬 분석 → **35건 제기 · 26 확정 · 5 기각 · 4 미검증**(세션 한도).
> 미검증 4건은 기각으로 섞지 않았고 §7에 따로 적었습니다.

**파일 통째 복사는 하지 않습니다.** 아래는 전부 *기법*이고, 가져올 때 출처 주석을 답니다.

---

## 1. 파라미터화 — 세 층으로 나뉘어 있습니다

```cpp
Attn< H, T, TP, C, CAP, RESI_CP,          // ① shape
      MATMUL_QKV_CIP, MATMUL_QKV_COP, ..., // ② 병렬도
      __attn_q_t, __attn_k_t, ...,         // ③ 신호별 ap_int<W>
      Q_WEIGHT_RAM_STYLE, Q_USE_DSP,       // ④ 블록별 RAM·DSP
      Q_ADPT_FIFO_DEPTH, ...,              // ⑤ FIFO 깊이 개별
      LN_ENTRIES_RSQRT, ... > attn_inst(...);   // ⑥ LUT 엔트리
```

**파라미터 126개 중 76개가 `int`입니다.**

| 층 | 무엇이 들어가나 | 판단 기준 |
|---|---|---|
| 매크로 | **선언되는 멤버 집합을 바꾸는 것** (head serial/parallel 등) | 템플릿 파라미터는 멤버를 지울 수 없습니다 |
| 템플릿 | shape · 병렬도 · 타입 · RAM 스타일 · FIFO 깊이 | 값이면 타입 시스템으로 |
| `.cpp` const | 그 값들의 실제 수 | 인스턴스별로 생성 |

### HBTXR은 매크로 층이 **필요 없습니다**

참조가 매크로를 쓴 이유는 **12개 레이어를 각각 독립 하드웨어로 펼쳤기** 때문입니다 — 한 레이어가
배치에 실패하면 그 레이어만 serial로 떨어뜨리는 탈출구가 필요했습니다. 실제로 저장소 전체에서
`HGPIPE_ATTN_HEAD_SERIAL`을 켜는 파일은 **`case_zu15eg/ATTN8.cpp` 하나**뿐이고, 그 탈출구를 위해
`attn.h`의 **약 1,400줄(999~1960, 파일의 60%)**이 존재합니다.

**HBTXR은 코어 4개를 재사용합니다 — 공유가 아키텍처이지 성능 저하가 아닙니다.** 지울 멤버가 없고,
replicate-vs-share는 최상위 dataflow 조립 문제이지 클래스 멤버 선언 문제가 아닙니다.

### 위치 인자 126개 → **traits 클래스**

```cpp
struct HbtxrCfgBase {
  static constexpr int N = 64, D = 192, H = 3, F = 768;
  static constexpr int TP = 4, QKV_CIP = 8, QKV_COP = 8;
  using act_t = ap_int<4>;  using acc_t = ap_int<20>;
};
struct HbtxrCfgTrack : HbtxrCfgBase { static constexpr int N = 16; };

template <class CFG> class HbtxrMhaCore { ... };
```

**HLS 제약 2개 — 가정하지 말고 확인해야 합니다:**

1. pragma 인자는 **integral constant expression**이어야 합니다. 클래스 스코프
   `static constexpr int D = CFG::FIFO_DEPTH;`로 한 번 받아서 `#pragma HLS stream depth=D`.
2. **C++20 class-type non-type 템플릿 파라미터를 HLS 프론트엔드가 받지 않습니다.**
   config는 반드시 **타입**으로 넘깁니다. `constexpr` 객체 값으로 넘기면 안 됩니다.

위치 인자의 실제 위험은 호출부가 아니라 **`attn.h:288-305`의 내부 전달**입니다 — 이름을 버리고
18칸 위치 인자로 넘기고, 헤더 주석에 `WGHT_FIFO_DEPHT` 오타까지 있습니다. `int` 슬롯끼리 바뀌면
**타입이 맞아서 컴파일도 csim도 통과**하고, `#pragma HLS stream depth=`만 달라져 cosim 행이나
조용한 처리량 손실로 몇 시간 뒤에 나타납니다.

---

## 2. 채택 — 그대로 가져올 것

### 2-1. `CIP/COP` = **unroll 인자 == array_reshape cyclic 인자 == `hls::vector` 폭**

`matmul.h:100-169`. **PE 배열은 어디에도 기술돼 있지 않습니다** — 세 숫자가 일치하는 것으로
암시됩니다. 이게 언롤된 MAC 큐브를 II=1로 합법화하는 규칙입니다.

```cpp
weight_arr[cot*COP + cop][cit*CIP + cip]   // cop/cip 이 언롤 인덱스
#pragma HLS array_reshape variable=weight_arr cyclic factor=COP dim=1
#pragma HLS array_reshape variable=weight_arr cyclic factor=CIP dim=2
```

**`array_partition`이 아니라 `array_reshape`입니다** — 언롤 타일 전체가 **한 개의 넓은 워드**로
접힙니다. 4-bit에서 `COP*CIP=64` 레인 × 4b = **256b 워드** → BRAM18 8개 또는 URAM 4개 병렬.

### 2-2. 리덕션을 **최내곽**에 두고 누산기는 레지스터

`matmul.h:118-167`. **루프 순서가 설계 결정이고 pragma가 아닙니다.** `CI` 리덕션을 최내곽에 두면
부분합이 `TP*COP`개 플립플롭이 되고 루프 캐리 정수 덧셈 1사이클 → **II=1이 공짜로 나옵니다.**
부분합용 메모리 포트가 필요 없습니다.

**HBTXR 누산기 폭**: `4 + 4 + ceil(log2(CI))`. 최악 `CI = F = 768` (MLP FC2) → 18비트 → **`ap_int<20>`**.

### 2-3. Adapter = CIP/COP 임피던스 정합 (**RAM 없이 시프트 레지스터**)

`adapter.h:40-127`. 이것 덕분에 `MATMUL_R_COP=7` 옆에 `MATMUL_A_CIP=7`, `MATMUL_QKV_COP=12`를
**이웃을 모르고** 놓을 수 있습니다. 각 블록 양끝에 폭 변환기가 붙습니다.

**단, `non_divisible` LCM 경로는 피합니다** — HBTXR의 모든 shape가 2의 거듭제곱 인수를 가지므로
(`D=192=64·3`, `F=768=256·3`, `N=64/16`, `CH=64`) 모든 `*P`를 2의 거듭제곱으로 잡으면 값싼 경로만 씁니다.

### 2-4. **전역 TP · 스테이지별 CP**

`attn.h:2007-2061`의 스트림 40개가 전부 `hls::vector<T, TP*CP>`이고 **TP는 전 설계에서 하나**입니다.
물리적으로 한 beat = `TP`행 × `CP`열. TP를 고정하면 스테이지 간 폭 불일치가 **항상 채널 불일치**라
Adapter 하나로 해결됩니다.

### 2-5. Requant = **add → shift → clamp → ROM** (곱셈기 없음)

`quant.h:50-79`. **포화가 테이블에 구워져 있습니다.** 원소당 덧셈 1 · 상수 시프트 1 · 비교 2 ·
LUTRAM 읽기 1. DSP 0개, II=1, `TP*CP` 레인 전체 언롤.

**이게 HBTXR 4개 테이블(RSQRT 64 · EXP 32 · RECIP 128 · GeLU 32) 전부의 데이터패스입니다.**

### 2-6. 정수 LayerNorm

`layernorm.h:107-191`. 행 버퍼 하나 위에서 단일 패스 3번, 각각 II=1. `1/C`가 dyadic 상수
(`43691/2^23 = 1/192`), 분산은 정확 정수, rsqrt는 테이블, affine은 후행 시프트 1개로 융합.

`algorithm/quantization/i_ops.py:157-191`의 `layernorm_quantize`가 **줄 단위로 같은 구조**입니다
(`mean_tmp += 1 << (c_1_s - 1)` 포함) — 즉 정합성이 이미 보장돼 있습니다.

### 2-7. `check_stream` — dataflow 중간 프로브

`utils.h:26-73` (45줄), 호출 30곳(`attn.h:2312-2362`). 세 가지를 합니다:

1. 레인별 **값 일치**
2. **비었는지 단언** — 과생산/beat 수 오류를 잡습니다 (값 비교로는 못 잡음)
3. 골든으로 **재충전** — 앞 단계 오류가 뒤로 전파되는 것을 끊습니다

**SPEC §9의 "스테이지별 골든" 요구가 정확히 이것입니다.**

### 2-8. 텍스트 템플릿 + `${}` — **템플릿으로 불가능한 것만**

`#include` 경로는 템플릿 파라미터로 계산할 수 없습니다. HBTXR이 생성할 것은 **둘뿐**입니다:
코어 인덱스별 `refs/*.txt` include, 그리고 4-bit 캘리브레이션이 내는 requant 폭·테이블.
나머지는 헤더 + traits 구조체.

---

## 3. 회피 — 참조가 틀린 것

### 3-1. ⚠️ `#pragma HLS bind_storage`를 **런타임 `if`로 감싸기**

`matmul.h:105-113` (같은 블록이 3곳에 복사돼 있음).

```cpp
if (WEIGHT_RAM_STYLE == BRAM_STYLE) {
  #pragma HLS bind_storage variable=weight_arr type=ram_2p impl=bram
} else if (WEIGHT_RAM_STYLE == LRAM_STYLE) { ... }
// URAM_STYLE 케이스 없음 — 가드도 주석 처리
```

**pragma는 감싸는 스코프의 변수에 컴파일 타임 바인딩됩니다.** 런타임 `if`가 적용 여부를
고르지 않습니다 — **두 바인딩이 모두 적용되거나 충돌합니다.** 그리고 `URAM_STYLE`(`common.h:35`의
합법값)을 넘기면 **바인딩이 아예 없어** 툴이 알아서 고릅니다.

**같은 저장소가 다른 곳에서는 제대로 합니다** — `reshaper.h:67-73`, `attn.h:1190-1196`은 `#if/#elif`.

> **HBTXR 규칙**: 저장소 선택은 전부 `#if/#elif`, fallthrough는 주석이 아니라 **`#error`**.

### 3-2. ⚠️ `TRANSPOSED`를 **런타임 bool**로 받아 `#pragma HLS dataflow` 안에서 분기

`matmul.h:389-421`. Vitis HLS dataflow는 영역 내 모든 프로세스가 **정확히 한 번** 실행돼야 하고
생산자-소비자 사이에 조건문이 없어야 합니다. `if/else`로 어느 프로세스가 `weight_sm`을 구동할지
고르는 것은 그 규칙 위반입니다.

**고침**: `template<bool TRANSPOSED>` + `if constexpr`로 루프 순서만 선택. RAM 절반, 정규 dataflow,
**약 60줄 감소.**

### 3-3. ⚠️ 커서 타입이 **클램프 전 인덱스를 담지 못함**

`gelu.h:67-68` / `quant.h:67-68`이 `clamp()`를 **이미 좁혀진 타입 위에서** 실행합니다.
범위 밖 인덱스가 포화되지 않고 **유효해 보이는 값으로 앨리어싱**돼 엉뚱한 테이블 엔트리를 읽습니다.
Vitis는 경고하지 않습니다.

그리고 `softmax.h:171-181` / `layernorm.h:166-167`은 **커서가 unsigned인데 `b` 스칼라가 큰 음수**입니다.
합이 `|b|` 아래로 떨어지면 시프트 결과가 음수 → 큰 unsigned로 절단 → `clamp(.,0,bound)`가 **무의미**해집니다.

> **HBTXR 규칙**: 커서 폭은 **도달 가능 범위에서 유도**하고 **signed**로 둡니다.
> `i_ops.table_quantize`가 임의정밀 `table[clamp((x+b)>>s, 0, bound)]`라 wrap은 곧 비트 불일치입니다.

### 3-4. ⚠️ 중간 폭을 **캘리브레이션에서 베껴 쓰기**

`layernorm.h:155-158, 185-186`. 가드 없는 오버플로 지점 3곳. 예: `__var_t`가 `D=192`에서
**36비트 필요한데 27비트**입니다.

> **HBTXR 규칙**: `diff = width(if_t)+1`, `var = 2·width(diff) + ceil(log2(D))`,
> `affine = width(diff)+width(rsqrt)+width(lnw)+1`. **유도하고 `static_assert`로 고정합니다.**

---

## 4. HBTXR에 그대로 못 오는 것 — **RMU가 ROM입니다**

`matmul.h:38-55`. 참조의 RMU는 **생성자에서 가중치를 ROM 초기화**합니다.

**이것이 `ATTN0..ATTN11`이 12개 독립 하드웨어 인스턴스가 된 이유입니다.**

HBTXR은 **코어 4개를 8/4 블록에 재사용**하므로 가중치가 **런타임 적재**돼야 합니다.
참조에 없는 **제3의 모드**가 필요합니다:

| 모드 | 가중치 | 참조 | HBTXR |
|---|---|---|---|
| RMU-ROM | 컴파일 타임 상수 | ✅ | ❌ 못 씀 |
| **RMU-resident** | **런타임 적재, 블록 간 상주** | ❌ | ✅ **track 경로** |
| SMU | 런타임 생성 (Q×Kᵀ, S×V) | ✅ | ✅ |

그리고 논문의 **Weight Prefetcher**는 참조에 아예 없습니다 — 모든 가중치가 온칩 상수라 필요가
없었습니다. HBTXR의 search 경로는 DRAM에서 가져옵니다.

### DSP 패킹 — 가장 큰 미개척 여지

참조는 `USE_DSP=false`가 전부입니다(`ATTN.cpp.template:73/79/85/108/125/131`). W4A4에서
**DSP48E2 하나에 4×4 MAC을 2개(pre-adder) 또는 4개(`(a<<k)+b` 패킹) 실을 수 있습니다.**
그대로 포팅하면 이 여지를 통째로 버립니다.

---

## 5. 실제 수치 — HBTXR에서 다시 계산해야 하는 것

| | 참조 (T=196) | HBTXR (N=64) |
|---|---|---|
| residual FIFO | `4096*3` = **3 URAM** | `N·D/(TP·RESI_CP)` = 64·192/16 = **768 beat → BRAM36 1개** |
| `QQ_HEAD_FIFO_DEPTH` | **8000** | 약 1/3 |
| 누산기 | 케이스별 전사 | `4+4+ceil(log2(CI))` → **`ap_int<20>`** |
| LayerNorm `var` | 27비트 (**부족**) | `2·14+8` = **36비트** |

**residual FIFO 깊이 규칙**: 메인 경로에 **토큰 전역 연산**이 있으면 텐서 전체를 버퍼링해야
합니다 — attention의 `Q×Kᵀ`는 `K` 전체가 저장되기 전에 `R`의 0행을 못 냅니다. MLP는 없으므로
**파이프라인 지연만** 덮으면 됩니다 (참조: `4096*3` vs `mlp.h:243`의 `300`).

---

## 6. 검증 — 참조에 **없는 것**

`check_stream`은 훌륭하지만:

- **음성 대조가 없습니다.** DUT가 틀렸을 때 실패하는지 확인하는 검사가 없습니다.
- `TEST_ROUND`가 같은 텐서를 4번 밀지만, `y_dut`가 **버퍼 하나**라 읽기 루프가 매 라운드
  덮어씁니다. 비교 루프는 그 **한 버퍼를 4번 비교**합니다 — 라운드 간 차이를 못 잡습니다.
- 여러 스테이지 참조 비교 오버로드가 `case/ATTN.cpp.template`에서 **주석 처리**돼 있습니다.

> **HBTXR은 SPEC §9로 이 셋을 다 막습니다** — 원소별 비교 + **음성 대조 필수** + 러너 연결.

---

## 7. 미검증 4건 — 세션 한도로 검증이 중단됐습니다

기각이 아닙니다. 쓰기 전에 확인해야 합니다.

- softmax의 **recip 테이블 2개**(`_one`/`_two`)가 왜 둘인지 — 수치 문제 추정, 미확인
- 테스트벤치 3개의 음성 대조 부재 주장
- 골든 벡터 `#include` 방식의 컴파일 비용
- 병렬도-shape 행렬

---

## 8. 기각된 5건 — 사실은 맞지만 진단이 틀린 것들

- "블록마다 dataflow 영역 하나" — 인용 9곳 중 **5곳에 그 pragma가 없습니다**
- head-wise reorder의 **전치 피연산자를 잘못 지목**
- 깊은 per-head FIFO의 **원인 설명이 틀렸고**, 증거 하나가 소스와 모순
- FIFO 깊이 60개 관련 — 구조 관찰은 맞지만 **핵심 증거가 틀림**
- 스테이지별 ref 명명 — 인용은 전부 실재하나 결론이 다름
