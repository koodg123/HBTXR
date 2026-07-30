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

## 3. 수 체계

| 타입 | 정의 | 쓰는 곳 |
|---|---|---|
| `hbtxr_a8_t` `hbtxr_w8_t` | `ap_int<8>` | Patch Embedding · Head |
| `hbtxr_a4_t` `hbtxr_w4_t` | `ap_int<4>` | MHA · MLP |
| `hbtxr_nl_t` | `ap_int<16>` | 비선형 LUT 입출력 |
| `hbtxr_acc_t` | `ap_int<32>` | 모든 누산 |

### requant — `algorithm/quantization`과 **비트 단위로 같아야 합니다**

```
out = clamp( (acc * M + (1 << (n-1))) >> n , qmin, qmax )
```

`(M, n)`은 채널별 dyadic 파라미터입니다. **이 식은 `algorithm/quantization/i_ops.py`의
정의이고 하드웨어가 그것을 따릅니다** — 반대가 아닙니다. 원소별 `==`로 검증합니다.

### 비선형 LUT — 연산자별 크기 (논문 §V-C)

| | 엔트리 | 폭 |
|---|---:|---|
| RSQRT | **64** | 16-bit |
| EXP | **32** | 16-bit |
| RECIP | **128** | 16-bit |
| GeLU | **32** | 16-bit |

인덱싱 `idx = clamp((u - u_min) >> s, 0, K-1)`. **테이블 값의 정본은
`algorithm/quantization` export**이고 HLS는 헤더 배열로 받습니다.

> **`HBTXR_FIXED_CSIM`은 항상 켭니다.** float 폴백을 두지 않습니다 — 구 구현이 기본
> float csim으로 양자화를 전혀 검증하지 못했던 것이 이 규칙의 이유입니다.

---

## 4. 프리미티브 — RMU · SMU

논문이 트랜스포머 본체를 이 둘로 정의합니다.

| | 용도 | 피연산자 |
|---|---|---|
| **RMU** Resident Matmul Unit | Q/K/V 생성 · output projection · FC1 · FC2 | 가중치 **resident** |
| **SMU** Stream Matmul Unit | `Q×Kᵀ` · `S×V` | 둘 다 **런타임 생성** |

```cpp
// RMU: 입력 채널 그룹 Di 를 소비해 출력 채널 그룹 Do 생성. PE = Nt × Do, PE당 Di 곱셈기
template <int Di, int Do, int Nt> void hbtxr_rmu_project(...);
// SMU: 런타임 스트림 둘을 상관. PE당 di 곱셈기
template <int Di, int Nt>          void hbtxr_smu_relate(...);
```

곱셈은 `#pragma HLS bind_op ... impl=dsp`로 **DSP에 고정**합니다.

---

## 5. 스테이지 체인

### 5-1. MHA Core — 9단계

```
residual split → LayerNorm → Q/K/V(RMU) → head-wise reorder
  → Q×Kᵀ(SMU) → Softmax → S×V(SMU) → output proj(RMU) → residual merge
```

| 버퍼 | 크기 | 이유 |
|---|---|---|
| 지연 residual FIFO | `Br` | attention 경로가 reorder·비선형을 포함해 bypass 보다 훨씬 깁니다 |
| head-local FIFO (Q) | `Nt` | |
| reorder buffer (K) | `Nt × d` | relation 단계가 자기 순서로 소비 |
| score buffer | `Nt × Nt` | Softmax 입력 |

### 5-2. MLP Core — 6단계

```
residual split → LayerNorm → FC1(RMU) → GeLU → FC2(RMU) → residual merge
```

**SMU · score buffer · head reorder 가 없습니다.** 그게 두 코어의 정의적 차이이고,
지연 residual 깊이도 `FC1→GeLU→FC2` 만 덮으면 됩니다.

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

### 모든 테스트벤치가 지켜야 할 것

1. 골든과 **원소별 비교**. 허용오차 금지
2. **음성 대조** — 일부러 틀린 입력이 실패하는지 확인. 없으면 통과가 무의미합니다
3. **러너에 연결** — `build/hls/` tcl 하나 또는 `build/run_*_tb.sh`. 연결 안 된 tb 는 만들지 않습니다

> 구 구현은 테스트벤치 12개 중 6개만 빌드에 걸렸고, 걸린 것 중 하나는 어서션이 0 이었으며,
> 가장 잘 만든 3개는 아무 빌드에도 없었습니다. 위 3번이 그것을 막습니다.

---

## 10. 범위 밖

- **Table III 자원·전력·지연 재현** — 구조가 목표입니다. 수치는 구조가 선 뒤 별도 과제
- 300 MHz 타이밍 클로저 — S9 에서 측정만 하고 기록
- 보드 실측 — ZCU104 물리 접근 필요
- 학습·양자화 캘리브레이션 — `algorithm/` 이 담당
