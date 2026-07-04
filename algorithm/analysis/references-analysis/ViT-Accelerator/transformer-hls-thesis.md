# transformer-hls-thesis 코드베이스 정밀 분석

> 대상 경로: `REF/ViT-Accelerator/transformer-hls-thesis`
> 분석 일자: 2026-06-20 / 분석 방식: 실제 소스 Read 후 라인 근거(파일명:라인) 기반

---

## 1. 개요

- **무엇인가**: Helsinki-NLP/opus-mt-en-es(MarianMT 계열, 영→스페인어 번역) 모델의 **Transformer 인코더 레이어 1개**를 Vitis/Vivado HLS C++로 구현한 석사 학위논문 프로젝트. Python(가중치 추출 + 임베딩 생성 + COMET/BLEU/BERTScore 평가)과 HLS C++(MHA, FFN, softmax, layernorm)가 결합된 HW/SW 혼합 코드베이스.
- **한 줄 요약**: "사전학습된 Opus-MT 인코더 레이어를 `ap_fixed` 고정소수점으로 양자화하여 FPGA(HLS)로 합성하고, COMET 점수로 양자화 비트폭 스윕(sweep)의 번역 품질 영향을 정량 평가하는 프로젝트".
- **논문 성격**: Master's thesis (README.md:3 "Master's thesis project"). 핵심 연구 질문은 "어느 텐서(attn/ffn1/ffn2/bias/acc)를 몇 비트로 줄여야 번역 품질(COMET)을 유지하는가"의 양자화 민감도 분석.
- **타깃 디바이스 (근거 추론)**:
  - `config/hls_config.cfg:1` → `part=xczu7ev-ffvc1156-2-e` (**Zynq UltraScale+ MPSoC, ZCU104 보드**).
  - `config/hls_config.cfg:4` → `flow_target=vivado`, `config/hls_config.cfg:32` → `syn.top=transformer_layer` (Top 함수 = `transformer_encoder_layer`).
  - README.md:40 도 동일하게 "Xilinx UltraScale+ (ZCU104: xczu7ev-ffvc1156-2-e)" 명시.
- **규모**: 모델 차원 D_MODEL=512, 헤드 8개(HEAD_DIM=64), FFN hidden=2048, 시퀀스 길이 SEQ_LEN=16 (attention_types.h:19-27). 테스트벤치는 동일 레이어를 6번 재사용해 6-layer 인코더를 흉내냄 (newDebugTb.cpp:340 `for(l=0; l<NUM_LAYERS; l++)`).

---

## 2. 디렉토리 구조

```
transformer-hls-thesis/
├── hls/src/                         # HLS C++ 커널 + 헤더
│   ├── transformer_layer.cpp/.h     # Top: 인코더 레이어 1개 (MHA→Add&Norm→FFN→Add&Norm)
│   ├── multi_head_attention.cpp/.h  # MHA: QKV projection→head split→attn→concat→Wo
│   ├── attention_scores2.cpp        # Q·Kᵀ/√d_k + 패딩 마스킹 (attention_scores)
│   ├── attn_output.cpp              # softmax(weights)·V (attention_output)
│   ├── matmul_qkv.cpp               # ★ 레거시/미사용(아래 §3.7 참조)
│   ├── softmax2.cpp/.h              # softmax_hybrid (piecewise exp 근사)
│   ├── layer_norm_cl.cpp + layer_norm.h  # LayerNorm (single-pass + hls::rsqrt)
│   ├── ffn_linear1.cpp/.h           # FFN 1단(512→2048), tiled
│   ├── ffn_linear2.cpp/.h           # FFN 2단(2048→512) + fused ReLU, tiled
│   ├── attention_types.h            # ★ 모든 ap_fixed 타입/차원/프로파일 정의
│   ├── lut_values.h                 # [생성된 LUT] rsqrt LUT (LayerNorm용)
│   ├── exp_lut_values.h             # [생성된 LUT] exp LUT (softmax용)
│   ├── opus_mt_weights.h            # [생성된 가중치] (내부 수치 분석 제외)
│   ├── opus_mt_embeddings.h         # [생성된 임베딩] (내부 수치 분석 제외)
│   └── newDebugTb.cpp               # C-sim 테스트벤치 (6-layer 루프, 통계/NaN 점검)
├── python/
│   ├── weightExtraction.py          # PyTorch → .bin 가중치 추출 (전치 포함)
│   ├── embedding.py                 # 문장 → 임베딩(.bin) 생성 (scale + pos emb)
│   ├── fetchTestSentence.py         # UN 병렬코퍼스에서 깨끗한 테스트 문장 채집
│   ├── comet2.py                    # HLS 출력 .bin → 번역 생성 → COMET/BLEU/BERT 평가
│   └── plotCometResults.py          # 양자화 스윕 결과(CSV) 시각화
├── config/hls_config.cfg            # Vitis HLS 프로젝트 설정(part/top/file list)
└── README.md
```

**중요한 구조적 사실 (config/hls_config.cfg:11-31)**: 합성 대상 파일 목록(`syn.file`)에 **`matmul_qkv.cpp`와 `attn_output.cpp`는 명시적으로 포함되지 않음** (단 `attn_output.cpp`는 attention_scores/softmax/MHA가 외부 참조하므로 실질 빌드에 필요. 반면 `matmul_qkv.cpp`는 §3.7처럼 정의 타입이 현재 프로파일에 없어 죽은 코드). Top은 `transformer_layer` 하나이며 나머지 커널은 일반 C++ 함수로 인라인 호출됨(데이터플로우 분리된 별도 IP가 아님).

---

## 3. 핵심 모듈 정밀 분석 ★

### 3.0 데이터 타입과 양자화 프로파일 (attention_types.h)

이 헤더가 전체 정밀도를 좌우한다. 활성 프로파일은 `#define SWEEP` (attention_types.h:10).

- 차원: `D_MODEL 512`, `NUM_HEADS 8`, `HEAD_DIM 64`, `FFN_HIDDEN_DIM 2048`, `SEQ_LEN 16`, `TILE_SIZE 64` (attention_types.h:19-27).
- SWEEP 프로파일 타입 (attention_types.h:34-44):
  - `w_attn_t = w_ffn1_t = w_ffn2_t = ap_fixed<16,8>` (가중치, 16비트·정수8비트)
  - `a_attn_t = a_ffn_t = ap_fixed<16,8>` (활성/바이어스)
  - `acc_t = ap_fixed<32,20>` (누산기, 32비트·정수20비트)
- 파생 타입: `proj_weight_t=w_attn_t`, `ffn1_weight_t=w_ffn1_t`, `qkv_output_t=a_attn_t`, `head_output_t=a_attn_t`, `ln_gamma_t=ln_beta_t=ap_fixed<16,8>` (attention_types.h:52-65, 71-74).
- softmax 전용 고정밀 타입 (attention_types.h:101-124):
  - `score_t = ap_fixed<22,10,AP_RND,AP_SAT>` (pre-softmax 점수)
  - `softmax_out_t = ap_fixed<12,2,AP_RND,AP_SAT>` (post-softmax 확률, 0~1 범위라 정수 2비트로 충분)
  - `safe_sum_t = ap_fixed<32,16,AP_RND,AP_SAT>` (exp 합 누산기), `inv_t = ap_fixed<32,4>` (1/합)
- 헤더에는 `ss1_1`~`ss6_10` 등 score/softmax 비트폭 조합이 다수 주석 처리되어 있어 (attention_types.h:85-119) **수동 양자화 스윕의 실험 흔적**임을 보여줌. `AP_RND`(반올림)/`AP_SAT`(포화)는 오버플로 안전성을 위한 선택.

> ⚠ **README와의 불일치(중요)**: README.md:52-56은 `ap_fixed<10,2> w_attn_t`, `ap_fixed<31,20> acc_t` 등을 예시로 들지만, 실제 활성 `SWEEP` 프로파일은 `<16,8>`/`<32,20>` 임. README는 baseline 스윕 시점의 옛 값으로 보이며 **실코드 기준은 attention_types.h**. plotCometResults.py:29-35의 `BASELINE_CONFIG`도 `<10,2>`/`<31,20>`을 baseline으로 적어, README 값은 "스윕의 한 지점"이지 현재 합성값이 아님.

### 3.1 Top: transformer_encoder_layer (transformer_layer.cpp)

- **역할**: Pre-norm이 아닌 **Post-norm(=Add&Norm)** 인코더 레이어. 파이프라인 순서 (transformer_layer.cpp:95-124):
  1. `multi_head_attention(input,...,attn_out)` (96)
  2. residual_1 = input + attn_out (99-104, `#pragma HLS PIPELINE II=1`)
  3. `layer_norm_newton_raphson(residual_1, ln1_out, ln1_gamma, ln1_beta)` (107)
  4. `ffn_linear1(ln1_out, ffn_w1, ffn_b1, ffn_hidden)` (110)
  5. `ffn_linear2_with_relu(ffn_hidden, ffn_w2, ffn_b2, ffn_out)` (113)
  6. residual_2 = ln1_out + ffn_out (116-121)  ← 주의: 잔차 기준이 input이 아니라 ln1_out
  7. `layer_norm_newton_raphson(residual_2, output, ...)` (124)
  8. 패딩 행 마스킹: `for(i=current_len; i<SEQ_LEN) output[i][j]=0` (127-132)
- **입출력**: `acc_t input[16][512]` → `acc_t output[16][512]`, 가중치 16종을 모두 인자로 받음 (transformer_layer.cpp:48-77). `current_len`으로 가변 시퀀스 길이 지원.
- **HLS pragma**: 인터페이스를 전부 `m_axi`로 두고 **gmem0~gmem8 9개 번들**로 분산 (transformer_layer.cpp:69-86) → 여러 DDR 포트로 동시 버스트 가능. 잔차 덧셈 루프만 `PIPELINE II=1`.
- **중간 버퍼**: `static head_output_t attn_out[16][512]` 등 모두 `static` 온칩 배열 (transformer_layer.cpp:88-93) → 단계 간 BRAM 재사용.
- **특이점**: `g_layer_counter` static 변수로 6레이어 통과를 추적해 6에서 리셋 (transformer_layer.cpp:46,134-135). 하나의 레이어 IP를 SW(테스트벤치)가 6번 호출하는 "weight-streaming" 패턴.

### 3.2 Multi-Head Attention (multi_head_attention.cpp)

- **역할**: 표준 MHA. QKV 전체 투영(512×512) → 8개 head로 슬라이스 → head별 attention → concat → 출력 투영 Wo.
- **3단계 구조 (multi_head_attention.cpp:140-194)**:
  - STEP1: `matmul_full_projection(input, W_q/W_k/W_v, b, Q_full/K_full/V_full)` — Q,K,V 각각 [16×512]@[512×512]+bias (149-151).
  - STEP2: head 루프 `for h in 0..7` (167): `extract_head_from_projection`으로 64차원 슬라이스 추출(offset=h*64, 169-171) → `attention_scores`(175) → `softmax_hybrid`(178) → `attention_output`(182) → `insert_head_output`로 concat 버퍼에 복원(185).
  - STEP3: `matmul_output_proj(concat_buffer, W_o, b_o, output)` (194).
- **핵심 연산 함수**:
  - `matmul_full_projection` (30-51): `sum_t sum=bias[j]; for k: sum += input[i][k]*weights[k][j]` 표준 행렬곱, `#pragma HLS PIPELINE II=1` (38). 가중치 레이아웃 `[in][out]` (44-46 주석).
  - `matmul_output_proj` (57-76): 동일 패턴.
  - `extract/insert_head` (82-111): 단순 슬라이스 복사, 각 `PIPELINE II=1`.
- **HLS pragma**: 인터페이스 m_axi gmem0~gmem5 (129-138). **head 루프(167)는 unroll/dataflow 없이 순차** → 8 head 직렬 처리(병렬화 여지로 남음, §8 참조).
- **입출력**: `acc_t input[16][512]` → `head_output_t output[16][512]`.

### 3.3 Attention Scores: Q·Kᵀ/√d_k + 마스킹 (attention_scores2.cpp)

- **역할**: head별 [16×64] Q,K로 [16×16] 점수 행렬 계산.
- **흐름 (attention_scores2.cpp)**:
  - Read: Q/K를 로컬 BRAM으로 복사 (22-38), `#pragma HLS ARRAY_PARTITION variable=Q_local/K_local dim=2 complete` (18-19) → 64차원(=dim2)을 완전 분할해 dot-product 한 사이클 읽기.
  - Compute (44-65): `if (j >= seq_len) scores=-1000.0` (패딩 마스킹, 51-53), else `dot_product += Q[i][k]*K[j][k]` 후 `scale_factor=0.125` 곱 (41,62). **0.125 = 1/8 = 1/√64** (HEAD_DIM=64의 역제곱근을 상수화). 각 루프 `PIPELINE II=1`.
  - mask 값 `-1000.0`은 softmax에서 exp(-1000)≈0이 되도록 한 안전값 (42, 50 주석).
- **데이터타입**: 입력 `qkv_output_t(=ap_fixed<16,8>)`, 누산 `sum_t(=acc_t=ap_fixed<32,20>)`, 출력 `score_t(=ap_fixed<22,10>)`.

### 3.4 Softmax: piecewise exp 근사 (softmax2.cpp)

- **역할**: [16×16] 점수 → [16×16] 확률. 함수명은 `softmax_hybrid`.
- **핵심: 실제 구현은 LUT가 아니라 piecewise 다항/선형 근사** (softmax2.cpp:47-60). `exp_lut_values.h`가 존재하지만 `softmax_hybrid`는 이를 사용하지 않음(아래 §3.8).
- **3단계 (softmax2.cpp:22-91)**:
  1. **Max 찾기** (27-33, `PIPELINE II=1`): 수치 안정성용 행 최댓값.
  2. **exp + 누산** (41-67): `x = scores - row_max` 후 piecewise (47-60):
     - `x <= -8.0` → 0 (꼬리 절단)
     - `-8.0 < x < -2.3` → 선형 램프 `0.1 + (x+2.3)*0.08`
     - `-2.3 <= x <= 0` → 2차 테일러 `1 + x + 0.5x²`
     - 음수 클램프 후 `exp_sum_hybrid` 누산 (63-66).
  3. **정규화** (74-89): `inv_t inverse = 1.0/final_sum`을 **행당 1회만** 계산(나눗셈 1개 IP, 79 주석), 이후 곱셈으로 확률 산출 → 나눗셈을 곱셈으로 치환한 HW 최적화. 0-합 방어 (`if final_sum==0 final_sum=1.0`, 75).
- **데이터타입**: `score_t<22,10>` 입력, `safe_sum_t<32,16>` 누산, `inv_t<32,4>` 역수, `softmax_out_t<12,2>` 출력.
- **부수 함수**: `softmax_lut/softmax_piecewise/softmax_taylor_improved`는 모두 빈 스텁(96-99) — 링커 만족용. 즉 piecewise hybrid 하나만 실사용.

### 3.5 Attention Output: weights·V (attn_output.cpp)

- **역할**: [16×16] attn_weights × [16×64] V → [16×64] head 출력. "Read-Compute-Write" 패턴 (attn_output.cpp:3-8 주석).
- **흐름**: V/weights 로컬 복사(33-50) → 컴퓨트(54-71): `output[i][d] = Σ_j weights[i][j]*V[j][d]`.
  - `#pragma HLS ARRAY_PARTITION variable=V_local dim=2 complete` (30) → V의 64차원 완전 분할.
  - 외부 루프(i,d) `PIPELINE II=1` (59), 내부 j 루프 `#pragma HLS UNROLL` (66) → SEQ_LEN=16 누산을 완전 언롤.
- **데이터타입**: `softmax_out_t<12,2>` × `qkv_output_t<16,8>` → `sum_t(=acc_t)` 누산 → `head_output_t<16,8>` 저장.

### 3.6 Layer Normalization (layer_norm_cl.cpp + layer_norm.h)

- **역할**: 행(토큰)별 LayerNorm. 함수명 `layer_norm_newton_raphson` (이름은 N-R이지만 실제론 `hls::rsqrt` IP 사용).
- **최적화 3가지 (layer_norm_cl.cpp:4-9 주석)**:
  1. **로컬 버퍼**: `row_buffer[D_MODEL]`에 행을 한 번만 읽어 저장 (24, 61), `#pragma HLS BIND_STORAGE ... type=ram_2p impl=bram` (25). gamma/beta도 로컬 복사 (30-39).
  2. **단일 패스 통계**: `sum_x`와 `sum_sq_x`를 한 루프에서 동시 누산 (47-66, `PIPELINE II=1`) → Var=E[x²]−E[x]² (76).
  3. **`hls::rsqrt`**: `inv_std = hls::rsqrt(variance + epsilon)` (81), `#include "hls_math.h"` (2). 분산 음수 클램프 (77), `epsilon=1e-5` (41).
- **정규화** (86-101): `normalized=(x-mean)*inv_std; result = normalized*gamma + beta`.
- **데이터타입**: 전부 `acc_t<32,20>` 도메인 + `ln_gamma/beta<16,8>`.
- **주의**: 파일이 `hls::rsqrt`를 쓰므로 `lut_values.h`(rsqrt LUT)는 **현 버전에서 미사용**일 가능성이 큼(과거 수동 N-R/LUT 구현 흔적). §3.8 참조.

### 3.7 FFN (ffn_linear1.cpp / ffn_linear2.cpp + 헤더)

**FFN1 (512→2048, ffn_linear1.cpp)**:
- **타일링**: hidden 2048을 `H_TILE_SIZE=512`로 나눠 `H_NUM_TILES=4` 타일 (ffn_linear1.h:9-15). `D_UNROLL_FACTOR=64`.
- **흐름**: 입력 로컬 복사(110-117) → 타일 루프(120-135): `read_weights_tile`/`read_bias_tile`/`compute_tile`/`write_output_tile`.
- **HLS pragma**: `input_local`/`weights_local`에 `ARRAY_PARTITION ... cyclic factor=D_UNROLL_FACTOR(64)` (101,104) → reduction 차원 64-way 병렬. `compute_tile`에 `#pragma HLS ALLOCATION operation instances=mul limit=64` (46) → 곱셈기(DSP) 64개로 제한해 자원 통제. 컴퓨트 루프는 1D 평탄화 후 `PIPELINE II=1` (48-64).

**FFN2 (2048→512, ffn_linear2.cpp) + fused ReLU**:
- **fused ReLU**: 입력을 읽으면서 `if(val<0) val=0` (ffn_linear2.cpp:8-26) → 별도 활성 패스 제거. (FFN의 비선형성은 GELU가 아니라 **ReLU**로 구현됨 — README:45는 GELU라 했으나 코드는 ReLU. §10 한계 참조.)
- **타일링**: 입력 2048을 `FFN2_INPUT_TILE_SIZE=512`로 4타일 (ffn_linear2.h:13-19), `FFN2_UNROLL_FACTOR=64`.
- **누산기 기반**: `sum_t output_accum[16][512]`를 bias로 초기화 후(94-100) 타일마다 부분합 누적 (compute_tile_ffn2, 47-68), 내부 reduction `#pragma HLS UNROLL factor=FFN2_UNROLL_FACTOR(64)` (61). `output_accum`에 `ARRAY_PARTITION dim=2 cyclic factor=64` (86).
- `ffn_linear2`는 `ffn_linear2_with_relu`로 위임하는 래퍼 (121-127).

### 3.8 LUT 헤더 (생성 데이터 — 용도/크기만)

- **`lut_values.h`**: "RSQRT Lookup Table for Layer Normalization", **256 엔트리**, 1/√x (x∈[0.01,10]) (lut_values.h:1-3). LayerNorm 분산 정규화용. 단 현 `layer_norm_cl.cpp`는 `hls::rsqrt`를 직접 호출하므로 이 LUT는 과거/대체 구현 흔적.
- **`exp_lut_values.h`**: "exp LUT", **256 엔트리**, exp(x) (x∈[-7.96875, 0]), 매핑 `x = -index/32.0` (exp_lut_values.h:1-3). softmax exp용으로 생성됐으나 현 `softmax_hybrid`는 piecewise 근사를 쓰므로 LUT 미사용. (LUT 기반 softmax는 `softmax_lut` 스텁만 존재.)
- **`opus_mt_weights.h` / `opus_mt_embeddings.h`**: 사전학습 가중치/임베딩을 C 배열로 박은 [생성된 데이터] — 내부 수치 분석 제외. 단 실제 테스트벤치는 이 헤더 대신 `weights_bin/*.bin` 파일을 런타임 로드함(newDebugTb.cpp:156-229).

### 3.9 레거시/미사용 모듈: matmul_qkv.cpp ★

- 이 파일은 `qkv_weight_t`/`qkv_bias_t` 타입을 사용(matmul_qkv.cpp:5,19,88-89)하지만 **현 `SWEEP` 프로파일의 attention_types.h에는 해당 typedef가 존재하지 않음**(grep 확인: 매치 0건). 또한 `config/hls_config.cfg`의 syn.file 목록에 없음 → **현재 빌드에서 컴파일/합성되지 않는 죽은 코드**.
- 다만 설계 관점에서 가치 있음: 이 파일은 **`#pragma HLS DATAFLOW`** 기반의 producer-consumer 분리 패턴(read_weights→read_bias→read_input→compute→write를 dataflow로 연결, matmul_qkv.cpp:106-112)과 `ALLOCATION mul limit=32`(52), `ARRAY_PARTITION cyclic factor=16`(103-104)을 보여줌. 실사용 MHA(§3.2)는 이 dataflow 패턴을 쓰지 않고 순차 호출함.

---

## 4. 데이터플로우 / 실행 흐름

**전체 파이프라인 (SW→HW→SW)**:

```
[SW/Python] embedding.py
   토큰화(MarianTokenizer) → embed_tokens → ×√d_model 스케일(embed_scale)
   → + sinusoidal positional emb → input_embeddings.bin, sequence_lengths.bin
[SW/Python] weightExtraction.py
   PyTorch encoder.layers[0] → W_q/k/v/o, b, ln, ffn → 전치(.T) → weights_bin/*.bin
        │
        ▼  (newDebugTb.cpp가 .bin 로드)
[HW/HLS] transformer_encoder_layer  (한 레이어를 6번 호출 = 6-layer 인코더 모사)
   input[16][512]
     → MHA: (QKV 512×512 투영) → head 8개 분할
              → scores=Q·Kᵀ·(1/8) + mask → softmax(piecewise exp) → ·V
              → concat → Wo 투영
     → Add(residual) → LayerNorm1(single-pass + hls::rsqrt)
     → FFN1(512→2048) → fused ReLU → FFN2(2048→512)
     → Add(residual) → LayerNorm2
     → 패딩 행 0 마스킹
   output[16][512]
        │
        ▼  (all_results → hls_output_full_encoder*.bin 저장, newDebugTb.cpp:451-452)
[SW/Python] comet2.py
   .bin → encoder_outputs로 주입 → model.generate(beam=4) → 스페인어 번역
   → COMET/BLEU/BERTScore vs PyTorch reference → master_experiment_comparison.csv
[SW/Python] plotCometResults.py  → 비트폭 vs COMET 히트맵/곡선
```

- **양자화/데이터타입 경계**: SW는 float32(.bin)로 전달 → 테스트벤치 `load_bin`이 `(T)temp_buf[i]`로 `ap_fixed`로 캐스팅(newDebugTb.cpp:170-172). 즉 양자화는 **C++ 로드 시점과 연산 중간**에서 ap_fixed 잘림으로 발생.
- **병렬화 요약**: (1) dim=2(임베딩/헤드차원) `ARRAY_PARTITION complete/cyclic`로 dot-product 병렬, (2) reduction 루프 `UNROLL factor=64`, (3) 거의 모든 출력 루프 `PIPELINE II=1`, (4) `ALLOCATION mul limit`로 DSP 캡. **head 루프와 6-layer 루프는 순차** (병렬화 안 됨).
- **가변 길이**: `current_len`으로 attention 마스킹(attention_scores2.cpp:51) + 출력 패딩 제로화(transformer_layer.cpp:127). 테스트벤치는 ping-pong 버퍼(buf_A/buf_B)로 레이어 간 입출력 교대(newDebugTb.cpp:345-346).

---

## 5. HW / SW 매핑

| Python (SW) | 산출물 | 대응 HLS C++ (HW) | 근거 |
|---|---|---|---|
| `weightExtraction.py` mha.q_proj/k/v/out + fc1/fc2 + layer_norm 추출, `.T` 전치 | `layerX_W_q.bin` 등 16종 | `EncoderWeights` 구조 / `transformer_encoder_layer` 인자, `matmul_full_projection`이 `[in][out]` 레이아웃 기대 | weightExtraction.py:26-99, transformer_layer.h:20-44, multi_head_attention.cpp:44 |
| `embedding.py` 토큰 임베딩 ×√d_model + sinusoidal pos | `input_embeddings.bin`, `sequence_lengths.bin` | `input_embeddings[][16][512]`, `sequence_lengths[]` 로 로드 후 `current_len` 사용 | embedding.py:104-134, newDebugTb.cpp:38-39,227-229 |
| `embedding.py` `embed_scale=√D_MODEL` | 스케일 적용된 임베딩 | 테스트벤치 "INPUT SCALING SKIPPED (already scaled in embedding.py)" | embedding.py:87, newDebugTb.cpp:336 |
| (PyTorch self_attn) | — | `multi_head_attention` (QKV→head→softmax→Wo) | multi_head_attention.cpp:116-194 |
| (PyTorch layer_norm) | ln gamma/beta | `layer_norm_newton_raphson` | layer_norm_cl.cpp:11 |
| (PyTorch fc1/ReLU/fc2) | ffn w1/b1/w2/b2 | `ffn_linear1` + `ffn_linear2_with_relu` | ffn_linear1.cpp:88, ffn_linear2.cpp:73 |
| `comet2.py` HLS .bin → `BaseModelOutput`로 디코더에 주입 → generate | 번역문 | `all_results`→`hls_output_full_encoder*.bin` | comet2.py:225-237, newDebugTb.cpp:451-452 |
| `comet2.py` COMET/BLEU/BERT 계산 | `master_experiment_comparison.csv` | (HLS 양자화 정밀도 평가 루프백) | comet2.py:256-282 |
| `plotCometResults.py` 실험명 파싱 `{n}sent_{attr}_{W}_{I}` | 히트맵/곡선 PNG | `ap_fixed<W,I>` 비트폭 ↔ COMET | plotCometResults.py:44-75, 187-266 |
| `fetchTestSentence.py` UN-PC 코퍼스 필터링 | 테스트 문장 리스트 | (embedding.py/comet2.py의 `test_sentences` 소스) | fetchTestSentence.py:5-110 |

핵심: **HW는 인코더 레이어만 담당**하고, 디코더+토크나이저+빔서치는 모두 PyTorch(SW)가 처리한다. HLS 인코더 출력을 PyTorch 디코더에 `encoder_outputs`로 끼워넣어 end-to-end 번역 품질을 측정하는 hybrid 평가 구조(comet2.py:225-235).

---

## 6. 빌드 / 실행

**HLS (README.md:75-97, config/hls_config.cfg)**:
```bash
cd config
vitis_hls -f hls_config.cfg     # part=xczu7ev-ffvc1156-2-e, top=transformer_layer
```
- 요구: Vitis HLS 2022.2+ (또는 Vivado HLS) (README.md:100). C-sim 테스트벤치는 `newDebugTb.cpp` (config:9).
- C-sim 실행 전 `weights_bin/`에 `.bin`들이 있어야 함(테스트벤치가 `weights_bin/` 상대경로 로드, newDebugTb.cpp:158,177).

**워크플로 (README.md:90-96)**:
1. `python python/weightExtraction.py` → 가중치 .bin
2. `python python/embedding.py` → 임베딩 .bin
3. Vitis HLS C-sim → `hls_output_full_encoder.bin`
4. `python python/comet2.py` → COMET 평가 CSV
5. `python python/plotCometResults.py` → 시각화

**양자화 스윕 절차(추론)**: attention_types.h의 타입을 수동 변경(예: `ss1_6`→`ss1_7` 주석 토글) → C-sim 재실행 → 출력 파일명을 `hls_output_full_encoder_50sent_attn_9_2.bin`처럼 바꿔 저장 → comet2.py의 `EXPERIMENT_NAMES`(comet2.py:51-70)가 일괄 평가. 즉 **빌드 자동화는 없고 수동 비트폭 토글 + 파일명 규약** 기반.

---

## 7. 의존성

**HLS 측**:
- Xilinx `ap_fixed.h` (attention_types.h:4), `hls_math.h` (layer_norm_cl.cpp:2, attention_scores2.cpp:2 — `hls::rsqrt`).
- 표준 C++ `<iostream><cmath><fstream>` 등(테스트벤치/디버그 출력용).

**Python 측 (README.md:71-73)**:
- `torch`, `transformers`(MarianTokenizer/MarianMTModel), `numpy`, `pandas`
- 평가: `sacrebleu`(BLEU), `bert-score`, `unbabel-comet`(COMET wmt22-comet-da, comet2.py:141)
- 데이터: `datasets`(Helsinki-NLP/un_pc, fetchTestSentence.py:5), 시각화 `matplotlib`
- 모델: `Helsinki-NLP/opus-mt-en-es` (weightExtraction.py:7, embedding.py:8, comet2.py:45)

**런타임 데이터 의존**: `weights_bin/` 디렉토리(가중치/임베딩/길이 .bin), HLS 출력 .bin, `master_experiment_comparison.csv`.

---

## 8. 강점 / 한계 / 리스크

**강점**:
- 실제 사전학습 모델(Opus-MT)을 그대로 양자화→HW 매핑하고, **번역 품질(COMET)로 양자화 영향을 정량 검증**하는 end-to-end 루프가 완성형. HW 정확도 검증을 task-level 지표로 닫음.
- HLS 최적화 정석을 충실히 적용: `ARRAY_PARTITION`(dot-product 병렬), `UNROLL factor`(reduction), `PIPELINE II=1`, `ALLOCATION mul limit`(DSP 캡), `BIND_STORAGE`(BRAM 지정), 타일링(FFN), fused ReLU, 나눗셈→곱셈 치환(softmax).
- 수치 안정성 처리가 견고: softmax max-subtraction, score 마스킹(-1000), 분산 음수 클램프, 0-합 방어, `AP_RND/AP_SAT`.
- 테스트벤치가 레이어별 통계/NaN/Inf/패딩 누수/explode·vanish 경고를 출력해 양자화 디버깅에 강함(newDebugTb.cpp:82-98,374-411).
- 비트폭 스윕을 위한 attention_types.h 프로파일 시스템 + 파일명 규약 + 자동 평가/플롯 인프라.

**한계 / 리스크**:
- **단일 레이어 IP를 SW가 6번 호출** — 진짜 6-layer 파이프라인이 아니라 weight-streaming 모사. HW 처리량/지연은 레이어 직렬 호출 + DDR 재로드에 묶임.
- **head 루프(8) 및 레이어 루프 미병렬** (multi_head_attention.cpp:167 순차) → 명백한 처리량 병목, dataflow/unroll 미적용.
- **죽은 코드 다수**: `matmul_qkv.cpp`(타입 미정의·빌드 제외), softmax LUT/스텁 3종, `lut_values.h`/`exp_lut_values.h`(현 구현 미사용), `opus_mt_weights/embeddings.h`(런타임 .bin로 대체). 코드 리딩 시 실제 활성 경로 판별 필요.
- **문서 불일치**: README의 ap_fixed 예시값(§3.0)·"GELU"(실제 ReLU, ffn_linear2.cpp:21)가 코드와 어긋남.
- **수동 스윕**: 비트폭 변경이 주석 토글 + 수동 파일명 → 재현성/자동화 취약, 합성 자원(LUT/DSP/BRAM) 리포트가 repo에 부재(품질 vs 자원 트레이드오프 데이터 없음).
- exp piecewise 근사(2차 테일러 + 선형 램프)는 정확도 손실 가능 — softmax_out_t<12,2>의 좁은 정수폭과 결합 시 확률 분포 왜곡 리스크.
- 잔차 기준이 LayerNorm 표준 구현에 따라 달라질 수 있음(transformer_layer.cpp:119는 residual_2=ln1_out+ffn_out; 모델 정의와의 일치 검증 필요).

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

우리는 **고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적**을 목표로 한다. 이 repo는 "단일 레이어·저처리량·품질검증 중심"이라 처리량 측면은 반면교사지만, **재사용 가능한 빌딩블록과 검증 인프라**가 분명히 있다.

**즉시 재사용/참고 가치 높은 것**:

1. **Attention 분해 패턴 (QKV→scores→softmax→·V→concat→Wo)**:
   - multi_head_attention.cpp의 4-스텝 분해와 head 슬라이싱(extract/insert, offset=h*HEAD_DIM)은 ViT MHSA에 그대로 매핑 가능. 단 **우리는 head 루프를 반드시 dataflow/unroll로 병렬화**해야 함(이 repo의 가장 큰 미흡점). matmul_qkv.cpp의 `#pragma HLS DATAFLOW` producer-consumer 분리 패턴(read→compute→write)을 head 레벨로 끌어올리면 HG-PIPE식 스테이지 파이프라인의 출발점이 됨.
   - `attention_scores2.cpp`의 1/√d_k를 상수(0.125)로 박아 DSP 절약하는 기법, `ARRAY_PARTITION dim=2 complete`로 dot-product 한 사이클 읽기는 우리 PE 설계에 직접 차용.

2. **LUT/근사 softmax·exp**:
   - `exp_lut_values.h`(256엔트리, x=-idx/32, x∈[-8,0]) 매핑 규약은 우리 LUT-기반 exp 유닛 설계 템플릿으로 좋음(인덱싱 수식이 명확). 다만 이 repo는 LUT를 실제로 안 쓰고 piecewise로 갔으므로, **우리는 LUT vs piecewise를 자원(BRAM 1개 vs DSP)·정확도로 비교 선택**해야 함.
   - softmax2.cpp의 **나눗셈→역수 곱셈 치환(행당 divider 1개)** + max-subtraction은 우리 streaming softmax(II=1 목표)에 필수 패턴. score_t<22,10>/softmax_out_t<12,2> 같은 비대칭 비트폭(확률은 정수폭 작게)도 좋은 출발점.

3. **LayerNorm HLS**:
   - layer_norm_cl.cpp의 **single-pass(sum_x + sum_sq_x 동시 누산) + `hls::rsqrt`** 구조는 그대로 쓸 만한 정석. `BIND_STORAGE ram_2p`로 행 버퍼를 BRAM 고정, gamma/beta 로컬 캐시도 차용. 대안으로 `lut_values.h`(rsqrt 256엔트리)를 살리면 `hls::rsqrt` IP 대신 BRAM-LUT로 갈 수 있어 자원 트레이드오프 카드가 됨.

4. **가중치 추출 → HW 흐름 (weightExtraction.py)**:
   - PyTorch `q_proj.weight` → `.T` 전치 → raw float .bin → C++ 로드 시 `(ap_fixed)` 캐스팅으로 PTQ. 이 "전치 + 레이아웃 `[in][out]` 합의 + .bin 컨벤션"은 우리 ViT 가중치(timm/torch) 추출 파이프라인에 그대로 적용 가능. **레이아웃 불일치가 가장 흔한 버그**임을 weightExtraction.py:55-62와 newDebugTb.cpp:267 주석이 경고.

5. **Task-level 양자화 검증 루프 (comet2/plotCometResults)**:
   - 우리도 시선추적 정확도(예: gaze MAE/픽셀오차)를 COMET 자리에 넣어 **"비트폭 vs end-task 정확도" 히트맵**을 만드는 평가 인프라를 차용. plotCometResults.py의 `{attr}_{W}_{I}` 파일명 규약 + 히트맵/Pareto 시각화는 거의 그대로 재사용 가능. 단 우리는 **여기에 없는 자원/지연 리포트(LUT/DSP/BRAM, latency)를 축에 추가**해 진짜 Pareto(품질 vs 자원/처리량)를 그려야 함.

6. **테스트벤치 디버깅 패턴**:
   - newDebugTb.cpp의 레이어별 mean/std/min/max + NaN/Inf + explode/vanish 경고는 양자화 가속기 br-up 단계의 표준 도구. XR 실시간 추론에서 수치 폭주는 치명적이므로 동일 가드를 우리 TB에 이식.

**경계/주의**: 이 repo는 SEQ_LEN=16·단일 레이어·저처리량이라 **HG-PIPE의 핵심인 레이어 간 파이프라이닝/이중버퍼/처리량 최적화는 전무**하다. 빌딩블록(softmax/layernorm/attention 분해/PTQ 흐름/평가 인프라)만 취하고, dataflow 아키텍처와 head/layer 병렬화는 우리가 새로 설계해야 한다.

---

## 10. 근거 / 한계 표기

**분석 근거**: 아래 파일을 직접 Read 후 라인 인용.
- HLS: `transformer_layer.cpp/.h`, `multi_head_attention.cpp/.h`, `attention_scores2.cpp`, `attn_output.cpp`, `matmul_qkv.cpp`, `softmax2.cpp/.h`, `layer_norm_cl.cpp` + `layer_norm.h`, `ffn_linear1.cpp/.h`, `ffn_linear2.cpp/.h`, `attention_types.h`, `newDebugTb.cpp`, `lut_values.h`(상단 25줄), `exp_lut_values.h`(상단 25줄).
- 설정: `config/hls_config.cfg` (전체).
- Python: `weightExtraction.py`, `embedding.py`, `comet2.py`, `fetchTestSentence.py`, `plotCometResults.py` (전체).
- `README.md` (전체).

**분석 제외/미완 부분과 이유**:
1. **`opus_mt_weights.h` / `opus_mt_embeddings.h`**: 제약에 따라 [생성된 가중치/임베딩 데이터]로 이름·용도만 언급, 내부 수치 미분석. 추가로 실제 빌드는 이 헤더 대신 `weights_bin/*.bin`(런타임 로드)을 사용하므로 헤더의 실사용 여부는 미확정.
2. **`lut_values.h`/`exp_lut_values.h`**: 제약에 따라 어떤 LUT/크기(256엔트리)/용도(rsqrt, exp)만 기술, 개별 수치 미분석. 현 활성 코드(`hls::rsqrt`, piecewise softmax)는 이 LUT를 사용하지 않는 것으로 보이나, 빌드 그래프 전체 추적은 미수행.
3. **합성 자원/지연 리포트 부재**: repo에 LUT/FF/DSP/BRAM/latency 합성 결과 파일이 없어 실제 처리량·자원 수치는 분석 불가(코드/pragma 기반 정성 추정만). 실측은 Vitis HLS 합성 필요.
4. **`matmul_qkv.cpp`**: 현 프로파일에 타입 미정의·syn.file 제외로 죽은 코드로 판단했으나(grep 0건), 다른 프로파일(`PROFILE_*`) 활성화 시 별도 typedef가 있을 가능성은 배제 못 함(주석 처리된 프로파일들의 전체 typedef 본문까지는 미추적).
5. **README vs 코드 불일치**(ap_fixed 예시값, GELU vs ReLU)는 본문에 명시했으며, 코드(attention_types.h, ffn_linear2.cpp)를 기준으로 삼음.
6. **6-layer 가중치 동일성**: 테스트벤치가 layer0~5 가중치를 각각 로드(newDebugTb.cpp:196-215)하나, weightExtraction.py는 `encoder.layers[0]`만 추출(weightExtraction.py:15)함 → 6레이어용 .bin이 어떻게 생성되는지는 repo 스크립트만으로는 불완전(별도 추출 스크립트/수동 작업 추정). 이 불일치는 미해소.
