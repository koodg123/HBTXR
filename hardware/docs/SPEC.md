> **작성** 2026-07-31 · **갱신** 2026-07-31
> **상태** active — HLS 재작성의 구현 계약
> **소유** hardware

# HBTXR 가속기 사양

논문 `PAPERS/JETCAS_REVISION1_FINAL_MAIN.pdf` §IV 구조를 HLS로 구현하기 위한 계약입니다.
[논문 대응표](architecture/2026-07-31-paper-to-hardware-mapping.md) ·
[재작성 계획](plans/active/2026-07-31-hls-rewrite-plan.md)

> 구 `HGTXR Third-Goal Reference Integration Spec`은 다른 문서입니다.
> `archive/hardware/docs/Spec.md`와 `rewrite/flat-functional` 브랜치에 있습니다.

---

## 0. 이름 규칙 — `hbtxr`

**새 코드에서 `hgtxr`는 금지입니다.**

| | |
|---|---|
| 함수·파일 | `hbtxr_<블록>_<역할>` — `hbtxr_mha_core`, `hbtxr_rmu_project` |
| 매크로·상수 | `HBTXR_<이름>` — `HBTXR_EMBED_DIM` |
| 타입 | `hbtxr_<의미>_t` — `hbtxr_acc_t` |
| 네임스페이스 | `hbtxr::` |

`docs/` 안의 **역사 기록(실험·핸드오프·리포트) 180개는 `hgtxr`를 유지**합니다. 그때
산출물의 실제 이름이 그것이고, 바꾸면 기록이 거짓이 됩니다.

---

## 1. 파라미터 계약

논문이 `D`·`H`·`F`를 기호로 두고 slimming이 정한다고 하므로, 실값의 정본은 **학습 모델**입니다.

| 기호 | 이름 | 값 | 출처 |
|---|---|---:|---|
| — | `HBTXR_SEARCH_H/W` | **128 × 128** | 논문 Table IV |
| — | `HBTXR_TRACK_H/W` | **64 × 64** | 논문 Table IV |
| — | `HBTXR_EVENT_CH` | **2** (극성) | 논문 Table IV `2×64×64` |
| `P` | `HBTXR_PATCH` | **16** | `algorithm/models/frame/model.py:29` |
| `N` | `HBTXR_SEARCH_TOKENS` | **64** (8×8) | 128/16 |
| `N` | `HBTXR_TRACK_TOKENS` | **16** (4×4) | 64/16 |
| `D` | `HBTXR_EMBED_DIM` | **192** | `model.py:28` |
| `H` | `HBTXR_HEADS` | **3** | `config/board/vitis_hls.yaml` |
| `d` | `HBTXR_HEAD_DIM` | **64** | `D/H` |
| `F` | `HBTXR_FF_DIM` | **768** | `mlp_ratio 4.0` × `D` (`backbones/vit.py:35`) |
| `L` | `HBTXR_SEARCH_DEPTH` | **8** TRB | 논문 Fig. 5 |
| `c` | `HBTXR_TRACK_DEPTH` | **4** TRB | 논문 Fig. 5 |

### 정합성 확인

MHA 1회 ≈ 11.0 M MAC · MLP 1회 ≈ 18.9 M MAC → TRB 1회 ≈ 29.9 M MAC ≈ **59.8 M op**.
search 8 TRB ≈ **478 M op**. 논문 search latency 1.342 ms × 326 GOPS ≈ **437 M op**.
**같은 자릿수입니다** — 파라미터 집합이 논문 처리량 주장과 모순되지 않습니다.

---

## 2. 4개 코어와 순회

논문은 코어 **4개**(`MHA Core0` `MLP Core0` `MHA Core1` `MLP Core1`)를 정의하고 TRB가
그 위를 **순환**합니다.

```
TRB_i  →  ( MHA Core_{i mod 2},  MLP Core_{i mod 2} )

search  TRB0..TRB7 = 4 순환        track  TRB0..TRB3 = 2 순환
```

코어는 **가중치만 바꿔 재사용**합니다. `HBTXR_CORE_PAIRS = 2`.

구현 수단: `template <int CORE_ID>` + `#pragma HLS INLINE off`.
`CORE_ID`는 **값으로 쓰지 않습니다** — HLS가 하드웨어를 2벌 만들게 하는 것이 목적입니다.

---

## 2-A. 파라미터화 — traits 클래스 하나, 매크로 0개

근거: [ViT_Accel 참조 분석](references/2026-07-31-vit-accel-hls-analysis.md)

```cpp
struct HbtxrCfgBase {                       // config/design/hbtxr_config.hpp
  static constexpr int N = 64, D = 192, H = 3, HD = 64, F = 768;
  static constexpr int TP = 4;              // 토큰 병렬도 — 전 설계에서 하나
  static constexpr int QKV_CIP = 8, QKV_COP = 8;   // 스테이지별 채널 병렬도
  static constexpr int R_CIP = 8,  R_COP  = 8;
  static constexpr int A_CIP = 8,  A_COP  = 8;
  static constexpr int O_CIP = 8,  O_COP  = 8;
  using act_t = ap_int<4>;  using acc_t = ap_int<20>;
};
struct HbtxrCfgTrack : HbtxrCfgBase { static constexpr int N = 16; };

template <class CFG> class HbtxrMhaCore { /* ... */ };
```

| 규칙 | 이유 |
|---|---|
| **매크로 0개** | 참조는 12 레이어를 각각 펼쳐서 한 레이어 배치 실패용 탈출구가 필요했고, 그 탈출구 하나에 `attn.h` 1,400줄(60%)이 딸려 있습니다. **HBTXR은 코어 공유가 아키텍처**라 지울 멤버가 없습니다 |
| **config 는 타입으로 전달** | **C++20 class-type non-type 템플릿 파라미터를 HLS 프론트엔드가 받지 않습니다.** `constexpr` 객체 값으로 넘기면 안 됩니다 |
| pragma 인자는 클래스 스코프 `static constexpr` 로 한 번 받기 | pragma 인자는 integral constant expression 이어야 합니다 |
| 모든 `*P` 는 **2의 거듭제곱** | Adapter 의 LCM 비분할 경로를 피합니다. `D=192=64·3`, `F=768=256·3`, `N=64/16`, `HD=64` 전부 가능 |
| 위치 인자 금지 | 참조는 파라미터 126개 중 **76개가 `int`** 라 슬롯이 바뀌어도 **컴파일·csim 통과**하고 cosim 행으로만 나타납니다 |

### PE 배열은 **세 숫자가 일치**해서 정의됩니다

```cpp
hls::vector<act_t, TP*CIP>                                   // 스트림 폭
#pragma HLS unroll factor=CIP                                 // 언롤 인자
#pragma HLS array_reshape variable=w cyclic factor=CIP dim=2  // reshape 인자
```

**`array_partition` 이 아니라 `array_reshape`** 입니다 — 언롤 타일이 한 개의 넓은 워드로 접힙니다.
4-bit 에서 `COP*CIP=64` 레인 = **256b 워드**.

**리덕션(`CI`)은 최내곽**에 둡니다. 부분합이 `TP*COP` 플립플롭이 되어 **II=1 이 공짜로** 나오고,
부분합용 메모리 포트가 필요 없습니다.

## 3. 수 체계

| 타입 | 정의 | 쓰는 곳 |
|---|---|---|
| `hbtxr_a8_t` `hbtxr_w8_t` | `ap_int<8>` | Patch Embedding · Head |
| `hbtxr_a4_t` `hbtxr_w4_t` | `ap_int<4>` | MHA · MLP |
| `hbtxr_nl_t` | `ap_int<16>` | 비선형 LUT 입출력 |
| `hbtxr_acc_t` | `ap_int<20>` | 모든 MAC 누산 — 아래에서 **유도** |

### 중간 폭은 전사하지 않고 **유도하고 `static_assert` 로 고정합니다**

참조가 캘리브레이션 값을 베껴 써서 `LayerNorm` 의 `var` 가 `D=192` 에서 **36비트 필요한데 27비트**
였습니다. 오버플로 가드가 없어 조용히 wrap 합니다.

| 폭 | 식 | HBTXR 값 |
|---|---|---|
| MAC 누산 | `W + A + ceil(log2(CI))` | `4+4+10` = 18 → **`ap_int<20>`** (최악 `CI=F=768`) |
| LN `diff` | `width(if_t) + 1` | |
| LN `var` | `2·width(diff) + ceil(log2(D))` | `2·14+8` = **36** |
| LN `affine` | `width(diff)+width(rsqrt)+width(lnw)+1` | |
| **LUT 커서** | **클램프 전** 인덱스의 도달 가능 범위. **반드시 signed** | |

> **커서를 좁게 잡으면 클램프가 무의미해집니다.** 좁혀진 타입 위에서 `clamp()` 를 돌리면 범위 밖
> 인덱스가 포화 대신 **유효해 보이는 값으로 앨리어싱**돼 엉뚱한 엔트리를 읽고, Vitis 는 경고하지
> 않습니다. unsigned 커서는 `clamp(.,0,bound)` 자체를 무의미하게 만듭니다.
> `i_ops.table_quantize` 가 임의정밀이라 **wrap 은 곧 비트 불일치**입니다.

### requant — `algorithm/quantization`과 **비트 단위로 같아야 합니다**

```
out = clamp( (acc * M + (1 << (n-1))) >> n , qmin, qmax )
```

`(M, n)`은 채널별 dyadic 파라미터입니다. **이 식은 `algorithm/quantization/i_ops.py`의
정의이고 하드웨어가 그것을 따릅니다** — 반대가 아닙니다. 원소별 `==`로 검증합니다.

#### `M` 은 33비트까지 옵니다 — S1 실측

`dyadic_params(scale, shift_max=31)` 은 **오차가 계속 줄기 때문에 거의 항상 `n=31`을 고릅니다.**
그러면 `M = round(scale·2³¹)` 이라 **비율이 1을 넘는 엣지에서 `M` 이 2³¹을 넘습니다.**
S1 골든의 requant 쌍 1,932개 실측:

| 프리셋 | `M` 최대 | 폭 | `n` 범위 | `acc·M` |
|---|---:|---:|---|---:|
| search-a4 | 2,468,372,009 | **32b** | 2 – 31 | **52b** |
| search-a8 | 4,461,753,799 | **33b** | 20 – 31 | **53b** |

> **`ap_int<20>` 누산기에 33비트 승수를 곱하면 53비트 곱입니다.** DSP48E2 한 개로 안 됩니다.
> 데이터패스를 넓히는 것이 답이 아니라 **`shift_max` 를 조이는 것**이 답입니다 — `shift_max=17`
> 이면 `M` 이 18비트에 들어가고, 상대오차 `2⁻¹⁷` 는 4비트 출력 격자에서 무의미합니다.
> **다만 `shift_max` 는 `i_block.rescale` 이 정하므로 이건 algorithm 쪽 변경입니다.**
> S2 착수 전 결정해야 합니다: 조이든지, 53비트 곱을 감수하든지.

### 비선형 LUT — 연산자별 크기 (논문 §V-C)

| | 엔트리 | 폭 |
|---|---:|---|
| RSQRT | **64** | 16-bit |
| EXP | **32** | 16-bit |
| RECIP | **128** | 16-bit |
| GeLU | **32** | 16-bit |

인덱싱 `idx = clamp((u - u_min) >> s, 0, K-1)`. **테이블 값의 정본은
`algorithm/quantization` export**이고 HLS는 헤더 배열로 받습니다.

> **LUT 의 입출력은 `hbtxr_nl_t`(16b)이지 matmul 폭이 아닙니다.** GeLU 테이블 출력을
> `dtype.bits` 로 묶었더니 4비트에서 **서로 다른 값이 2개**로 붕괴했습니다 (S1 `check()` 가 잡음).
> 좁히는 것은 테이블이 아니라 **그 다음 엣지의 requant** 입니다.

#### 혼합정밀은 현재 오라클이 표현하지 못합니다 — S1 미결

논문은 **matmul 피연산자 4비트 · 비선형 16비트**입니다. 그런데 `i_block.BlockSpec` 은 `dtype`
**하나**를 모든 엣지에 씁니다. 그래서 `fc1 → GeLU` 엣지가 4비트로 클램프되고, **GeLU 의 입력
알파벳이 16개**가 됩니다 — 32엔트리 테이블의 절반이 영영 안 닿습니다.

S1 은 이걸 우회하지 않고 **`--bits 4` 와 `--bits 8` 두 벌을 냅니다**: 4비트 벌은 RMU/SMU/qkv
(진짜 W4A4 피연산자)용이고, 8비트 벌은 LUT 입력에 여유가 있는 벌입니다.
**S4·S5 전에 결정해야 합니다** — `BlockSpec` 에 엣지별 dtype 을 넣을지, 하드웨어를 8비트 LUT
입력으로 갈지.

> **`HBTXR_FIXED_CSIM`은 항상 켭니다.** float 폴백을 두지 않습니다 — 구 구현이 기본
> float csim으로 양자화를 전혀 검증하지 못했던 것이 이 규칙의 이유입니다.

---

## 3-A. 금지 관용구 — 참조에서 확인된 것

| | 왜 |
|---|---|
| **런타임 `if` 로 감싼 `#pragma HLS bind_storage`** | pragma 는 스코프 변수에 **컴파일 타임** 바인딩됩니다. `if` 가 적용 여부를 고르지 않아 두 바인딩이 모두 걸리거나 충돌하고, 해당 케이스가 없으면 **바인딩이 아예 없습니다.** 저장소 선택은 전부 `#if/#elif`, fallthrough 는 **`#error`** |
| **`dataflow` 영역 안의 런타임 조건 분기** | dataflow 는 영역 내 프로세스가 정확히 한 번 실행돼야 하고 생산자–소비자 사이에 조건문이 없어야 합니다. `template<bool>` + `if constexpr` 로 |
| 위치 인자 템플릿 | §2-A |
| 캘리브레이션 폭 전사 | §3 |

## 4. 프리미티브 — RMU · SMU

논문이 트랜스포머 본체를 이 둘로 정의합니다.

| | 용도 | 피연산자 |
|---|---|---|
| **RMU** Resident Matmul Unit | Q/K/V 생성 · output projection · FC1 · FC2 | 가중치 **resident** |
| **SMU** Stream Matmul Unit | `Q×Kᵀ` · `S×V` | 둘 다 **런타임 생성** |

```cpp
template <class CFG> void hbtxr_rmu_project(...);   // 가중치 resident
template <class CFG> void hbtxr_smu_relate(...);    // 둘 다 런타임 스트림
```

### RMU 는 **런타임 적재**여야 합니다 — 참조에 없는 모드입니다

참조의 RMU 는 **생성자에서 가중치를 ROM 초기화**합니다. **그래서 `ATTN0..ATTN11` 이 12개 독립
하드웨어 인스턴스가 됐습니다.** HBTXR 은 코어 4개를 8/4 블록에 재사용하므로 그 방식을 쓸 수 없습니다.

| 모드 | 가중치 | 참조 | HBTXR |
|---|---|---|---|
| RMU-ROM | 컴파일 타임 상수 | ✅ | ❌ 코어 재사용과 양립 불가 |
| **RMU-resident** | **런타임 적재, 블록 간 상주** | ❌ 없음 | ✅ track |
| SMU | 런타임 생성 (`Q×Kᵀ`, `S×V`) | ✅ | ✅ |

참조에 **Weight Prefetcher 가 없는 이유도 같습니다** — 모든 가중치가 온칩 상수였습니다.

### DSP 패킹

곱셈은 `#pragma HLS bind_op ... impl=dsp|fabric` 으로 지정하고, **compound `+=` 가 아니라 이름 있는
임시 변수**에 붙여야 안정적으로 바인딩됩니다.

참조는 `USE_DSP=false` 가 전부입니다. **W4A4 에서는 DSP48E2 하나에 4×4 MAC 을 2개(pre-adder) 또는
4개(`(a<<k)+b` 패킹) 실을 수 있습니다** — 그대로 포팅하면 이 여지를 버립니다. S2 에서 두 방식을
모두 측정하고 기록합니다.

---

## 5. 스테이지 체인

### 5-1. MHA Core — 9단계

```
residual split → LayerNorm → Q/K/V(RMU) → head-wise reorder
  → Q×Kᵀ(SMU) → Softmax → S×V(SMU) → output proj(RMU) → residual merge
```

| 버퍼 | 크기 | 이유 |
|---|---|---|
| 지연 residual FIFO | **`N·D/(TP·RESI_CP)`** = 64·192/16 = **768 beat** | 아래 |
| head-local FIFO (Q) | `Nt` | |
| reorder buffer (K) | `Nt × d` | relation 단계가 자기 순서로 소비 |
| score buffer | `Nt × Nt` | Softmax 입력 |

### 5-2. MLP Core — 6단계

```
residual split → LayerNorm → FC1(RMU) → GeLU → FC2(RMU) → residual merge
```

**SMU · score buffer · head reorder 가 없습니다.** 그게 두 코어의 정의적 차이입니다.

### residual FIFO 깊이 규칙

**메인 경로에 토큰 전역 연산이 있으면 텐서 전체를 버퍼링해야 합니다.** attention 의 `Q×Kᵀ` 는
`K` 전체가 저장되기 전에 `R` 의 0행을 낼 수 없습니다 → MHA 는 `N·D/(TP·RESI_CP)`.
MLP 는 전역 연산이 없으므로 **파이프라인 지연만** 덮으면 됩니다.

참조에서 이 대비가 `4096*3`(3 URAM) vs `300` 으로 나타납니다. **베끼지 말고 다시 계산합니다** —
`N=64`, `TP·RESI_CP=16` 이면 768 beat 로 **BRAM36 하나**면 됩니다.

### 5-3. Patch Embedding Core

```
line buffer (K-1 행) → 모달리티별 conv (Conv-F | Conv-E) → shuffler → 토큰 스트림
```

PE = `Do` 개, PE당 `K×K×Ui` 곱셈기 + 누산기 1. **출력 토큰 프로토콜은 두 모드가 공유**하고
모달리티 차이는 conv stem 안에만 있습니다.

---

## 6. 구조 블록

| 블록 | 책임 |
|---|---|
| **Global Buffer** | 토큰·가중치 스테이징. track resident 영역과 search 스트리밍 영역 분리 |
| **on-chip interconnect** | GB ↔ 코어 로컬 버퍼 분배 |
| **Weight Prefetcher** | search 가중치 on-demand fetch, **초기 레이어 실행과 중첩** |
| **Controller** | 모드 선택 · 소스 라우팅 · 실행 시퀀스 · 상태 노출 |

메모리 정책: **track 상주 가중치와 anchor/state 는 on-chip**, **search 전용 가중치는
DRAM 에서 on-demand**. 첫 search 레이어 가중치가 도착하면 **즉시 실행 시작**하고 나머지는
계속 스트리밍됩니다.

---

## 7. 모드 계약

| | search | track |
|---|---|---|
| 입력 | frame 128×128 | event 2×64×64 |
| stem | **Conv-F** | **Conv-E** |
| 토큰 | 64 | 16 |
| 깊이 | 8 TRB | 4 TRB (cut `c`) |
| terminal decode | **Pupil Box** | **Pupil Ellipse** |
| 가중치 | on-demand + prefetch | resident |

**한 설계 안에서 라우팅합니다.** 모드별 비트스트림을 따로 만들지 않습니다 — 구 구현이
`search_profile_top` / `track_profile_top` 두 개로 나뉘어 있던 것이 논문 구조와 다릅니다.

---

## 8. 최상위 인터페이스

```cpp
void hbtxr_top(
    hls::stream<hbtxr_axis_t> &in_stream,     // AXIS   ← 입력 DMA
    hls::stream<hbtxr_axis_t> &out_stream,    // AXIS   → 출력 DMA
    const volatile hbtxr_axi_word_t *weights, // M-AXI  ← DRAM (search 가중치)
    int  mode,                                // s_axilite  0=search 1=track
    int *status);                             // s_axilite
```

제어는 `s_axilite`, 데이터는 AXIS, 가중치는 M-AXI. PS–PL 경계는 이 셋입니다.

---

## 9. 검증 계약 — **완료 조건은 "컴파일된다"가 아닙니다**

| 층 | 무엇 | 도구 |
|---|---|---|
| **V1** | 블록 출력 == 골든, **원소별** | g++ + `ap_int` (WSL) |
| **V2** | top == 모델 골든 | Vitis HLS csim |
| **V3** | 합성 가능 · 자원 · 타이밍 | Vitis HLS csynth |

**골든의 정본은 `algorithm/quantization`** 입니다 — 순수 파이썬 임의정밀 정수 오라클이고
허용오차가 아니라 `==` 로 비교합니다. `tools/export_hls_golden.py`(S1)가 블록별 벡터를 냅니다.

### 골든 파일 계약 — S1 산출

```bash
sh hardware/build/make_golden.sh          # 프리셋 6벌 -> hardware/workspace/golden/<mode>-a<bits>/
```

**생성물이라 git 에 없습니다.** seed 로 재현되고 6벌이 9 MB 라 커밋할 이유가 없습니다.
한 디렉토리에 84개 `.txt`(정수 CSV, 후행 콤마) + `index.tsv`(파일·개수·모양·설명).

| 무엇 | 파일 |
|---|---|
| 스테이지 벡터 | `<stage>_x` · `<stage>_y` — `patch` `ln1` `softmax` `gelu` `rmu` `smu` `qkv` `mha` `mlp` `block` |
| 상주 가중치 | `{rmu,qkv,fc1,fc2}_weight` + `_bias_acc` (**누산기 격자**에 있습니다) |
| LUT 페이로드 | `ln{1,2}_{scalars,lnw,lnb,rsqrt_table}` · `softmax_{scalars,exp_table,recip_table_one,recip_table_two}` · `gelu_{scalars,table}` |
| **모든** requant | `e01`~`e14` 각각 `_mult`/`_shift`. 순전파 순서로 번호를 매겼습니다 |

> **requant 를 전수로 내는 이유**: 재현 못 하는 골든은 골든이 아닙니다. 18개 엣지 중
> 흥미로운 몇 개만 내면 tb 가 `block_y` 를 만들 수 없습니다.

생성기는 매 실행마다 **파일만 읽어서** `rmu_y`·`gelu_y`·`smu_y` 를 스펙 객체 없이 재구성하고
`==` 로 확인합니다. **tb 가 할 일과 정확히 같은 일**이라, 통과하면 파일이 tb 에 충분합니다.

> **음성 대조로 확인한 한계**: 출력만 비교하면 `bias_acc` 오차 **±8 까지는 안 보입니다**
> (`n`이 27~31이라 누산기 LSB가 출력 LSB 한참 아래). ±64부터 잡힙니다. bias 를 LSB 단위로
> 검증하려면 tb 가 **누산기 자체**를 봐야 합니다.

### 스테이지 프로브 — `hbtxr_check_stream`

ViT_Accel 의 `utils.h:26-73` 기법을 가져옵니다 (45줄). dataflow 중간에 끼워 셋을 합니다:

1. 레인별 **값 일치**
2. **비었는지 단언** — 과생산·beat 수 오류를 잡습니다. 값 비교로는 못 잡습니다
3. 골든으로 **재충전** — 앞 단계 오류가 뒤로 전파되는 것을 끊어 첫 실패 지점을 특정합니다

`#ifndef __SYNTHESIS__` 오버로드로 두어 합성 경로에는 pragma 도 FIFO 도 남기지 않습니다.

### 모든 테스트벤치가 지켜야 할 것

1. 골든과 **원소별 비교**. 허용오차 금지
2. **음성 대조** — 일부러 틀린 입력이 실패하는지 확인. 없으면 통과가 무의미합니다
3. **러너에 연결** — `build/hls/` tcl 하나 또는 `build/run_*_tb.sh`. 연결 안 된 tb 는 만들지 않습니다

> 참조는 1·3 은 잘 하지만 **2 가 없습니다.** 그리고 `TEST_ROUND=4` 로 같은 텐서를 4번 밀면서
> `y_dut` 는 버퍼 하나라 매 라운드 덮어쓰고 **같은 버퍼를 4번 비교**합니다 — 라운드 간 차이를
> 못 잡습니다. 라운드별 버퍼를 쓰거나 라운드마다 즉시 비교합니다.

> 구 구현은 테스트벤치 12개 중 6개만 빌드에 걸렸고, 걸린 것 중 하나는 어서션이 0 이었으며,
> 가장 잘 만든 3개는 아무 빌드에도 없었습니다. 위 3번이 그것을 막습니다.

---

## 10. 범위 밖

- **Table III 자원·전력·지연 재현** — 구조가 목표입니다. 수치는 구조가 선 뒤 별도 과제
- 300 MHz 타이밍 클로저 — S9 에서 측정만 하고 기록
- 보드 실측 — ZCU104 물리 접근 필요
- 학습·양자화 캘리브레이션 — `algorithm/` 이 담당
