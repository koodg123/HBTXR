# Transformer-Accel / `transformer-hls-thesis` 정밀 분석

> 대상 repo: `REF/Transformer-Accel/transformer-hls-thesis`
> 분석 방식: 자체 소스(.cpp/.h/.py/.cfg) 전수 Read 후 라인 근거 기반. 생성물/데이터 테이블은 이름만 언급.
> 근거 표기 규칙: 파일명:라인 = 코드 직접 확인 / "추정" = 코드로 단정 불가한 해석 / "확인 불가" = repo 내 증거 없음.

---

## 1. 개요

- **정체**: 석사 학위논문 코드. Vivado/Vitis HLS로 **Transformer 인코더 레이어 1개**를 FPGA에 합성하기 위한 HLS C++ 커널 + 가중치 추출/평가용 Python 파이프라인. (`README.md:1-3`)
- **타깃 모델**: HuggingFace **Helsinki-NLP/opus-mt-en-es** (MarianMT, 영→스페인어 번역) 인코더. (`README.md:40`, `python/embedding.py:8`, `python/weightExtraction.py:7`)
- **타깃 디바이스**: Xilinx UltraScale+ **ZCU104, xczu7ev-ffvc1156-2-e**. (`config/hls_config.cfg:1`, `README.md:40`)
- **모델 차원**: `D_MODEL=512`, `NUM_HEADS=8`, `HEAD_DIM=64`, `FFN_HIDDEN_DIM=2048`, `SEQ_LEN=16`. (`attention_types.h:19-27`)
- **핵심 성격**: 본 repo의 실질 기여는 "고처리량 가속기 마이크로아키텍처"가 아니라 **고정소수점(ap_fixed) 양자화 비트폭 스윕(sweep) 실험 프레임워크**다. 동일 RTL을 비트폭만 바꿔 합성하고, HLS C-시뮬레이션 출력을 PyTorch 디코더에 다시 먹여 **번역 품질(COMET/BLEU/BERTScore) vs 비트폭** 곡선을 그리는 것이 워크플로우의 중심. (`python/comet2.py:51-70`, `python/plotCometResults.py:1-13`)
- **중요 단서**: 코드 곳곳에 `ReLU`가 쓰이지만(`ffn_linear2.cpp:21`) README는 GELU라고 적어(`README.md:45`) **README와 실제 구현이 불일치**. 또한 README의 비트폭 예시(`ap_fixed<10,2>` 등)와 실제 활성 프로파일(`ap_fixed<16,8>`)도 불일치. (아래 9절 참조)

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상, 전수 Read 완료)

```
transformer-hls-thesis/
├── hls/src/
│   ├── transformer_layer.cpp / .h        # 톱레벨 인코더 레이어 (오케스트레이션)
│   ├── multi_head_attention.cpp / .h     # MHA: QKV 투영 + 8-head 루프 + 출력 투영
│   ├── attention_scores2.cpp             # Q·Kᵀ/√d + 패딩 마스킹
│   ├── softmax2.cpp / .h                  # piecewise-exp 기반 softmax
│   ├── attn_output.cpp                    # attn_weights · V
│   ├── matmul_qkv.cpp                     # (미사용) head 단위 matmul, DATAFLOW 예제
│   ├── ffn_linear1.cpp / .h              # FFN 1층 (512→2048), 타일링
│   ├── ffn_linear2.cpp / .h              # FFN 2층 (2048→512), ReLU 융합
│   ├── layer_norm_cl.cpp / layer_norm.h # LayerNorm (단일패스 + hls::rsqrt)
│   ├── attention_types.h                 # 모든 ap_fixed 타입/차원 정의 (핵심)
│   └── newDebugTb.cpp                     # C-sim 테스트벤치 (6-layer 루프, bin I/O)
├── python/
│   ├── weightExtraction.py               # opus-mt 인코더 가중치 → .bin
│   ├── embedding.py                       # 입력 문장 → 임베딩 .bin
│   ├── comet2.py                          # HLS 출력 vs PyTorch, COMET/BLEU/BERT
│   ├── plotCometResults.py               # 비트폭 스윕 결과 시각화
│   └── fetchTestSentence.py              # UN 병렬코퍼스에서 깨끗한 테스트 문장 추출
├── config/hls_config.cfg                  # Vitis HLS 프로젝트 정의
└── README.md
```

### 2.2 생성물 / 데이터 테이블 (이름만 언급, 분석 제외)

- `hls/src/opus_mt_weights.h` — opus-mt 가중치를 C 배열 리터럴로 박은 **자동생성 데이터 파일**(수천 행, `weight_t opus_W_q[512][512]={...}` 등). 런타임 경로는 이 헤더가 아니라 `.bin` 파일 로딩을 사용하므로 사실상 미사용 잔재로 추정. (`opus_mt_weights.h:10-15` 확인, 활성 경로 비사용은 추정)
- `hls/src/opus_mt_embeddings.h` — 임베딩 데이터 테이블(자동생성). 동일 사유로 미사용 추정.
- `hls/src/exp_lut_values.h` — `exp(x)` 256-엔트리 LUT, "Auto-generated with numpy.exp()". (`exp_lut_values.h:1-3`)
- `hls/src/lut_values.h` — `1/sqrt(x)` 256-엔트리 RSQRT LUT, 자동생성. (`lut_values.h:1-3`)
  - 두 LUT 모두 `softmax_lut` / Newton-Raphson 변형용이나 **활성 커널은 LUT 대신 piecewise-Taylor(softmax)와 `hls::rsqrt`(layernorm)를 사용** → LUT 헤더는 비활성 잔재로 추정. (`softmax2.cpp:47-60`, `layer_norm_cl.cpp:81`)
- `.git/` — 버전관리 메타데이터. 제외.

> 합성 리포트/비트스트림/`.ip_user_files` 등 빌드 생성물은 repo에 **존재하지 않음**(`.gitignore`로 제외된 것으로 추정).

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.0 타입 시스템 — `attention_types.h` (양자화 실험의 중심)

이 헤더가 전체 repo의 "다이얼"이다. `#define SWEEP`로 활성 프로파일을 선택하면(`attention_types.h:10`) 한 묶음의 ap_fixed 타입이 결정된다.

- **차원 상수**: `D_MODEL=512`, `NUM_HEADS=8`, `HEAD_DIM=64`, `FFN_HIDDEN_DIM=2048`, `SEQ_LEN=16`, `TILE_SIZE=64`. (`attention_types.h:19-27`)
- **활성 프로파일(SWEEP)**: 가중치/활성을 전부 `ap_fixed<16,8>`, 누산기 `acc_t=ap_fixed<32,20>`로 둠. (`attention_types.h:32-45`)
  - `w_attn_t / w_ffn1_t / w_ffn2_t = ap_fixed<16,8>` (`:34-36`)
  - `a_attn_t / a_ffn_t = ap_fixed<16,8>` (`:40-41`)
  - `acc_t = ap_fixed<32,20>` (`:44`)
- **파생 타입**: `proj_weight_t`(W_q/k/v/o), `proj_bias_t`, `ffn1/2_weight_t`, `ln_gamma_t/ln_beta_t=ap_fixed<16,8>`, 활성 `qkv_output_t/head_output_t/ffn_output_t`, `sum_t=acc_t`. (`:53-80`)
- **softmax 전용 타입(독립, 고정밀 유지)**: 활성 설정 `score_t=ap_fixed<22,10,AP_RND,AP_SAT>`, `softmax_out_t=ap_fixed<12,2,AP_RND,AP_SAT>`. (`:101-102`) 그 위/아래로 ss1_1~ss6_10 등 **주석 처리된 대안 비트폭 세트가 다수** 남아 스윕 흔적을 보여줌. (`:85-119`)
- **softmax 내부 누산**: `safe_sum_t=ap_fixed<32,16>`, `inv_t=ap_fixed<32,4>`. (`:123-124`)
- **근거 표기 주의**: README의 예시 타입(`ap_fixed<10,2>` 등, `README.md:52-56`)과 `plotCometResults.py`의 `BASELINE_CONFIG`(attn=(10,2), ffn1/2=(8,2), bias=(10,8), acc=(31,20))(`plotCometResults.py:29-35`)는 **활성 SWEEP 값과 다르다**. 즉 "baseline" 프로파일과 "sweep" 프로파일은 서로 다른 실험 시점의 설정이며, 활성 코드는 SWEEP(16비트). 비트폭 수치를 인용할 때 어느 프로파일인지 구분 필요.

---

### 3.1 톱레벨 — `transformer_encoder_layer()` (`transformer_layer.cpp:48-136`)

Pre-norm이 아니라 **Post-norm(Add→LayerNorm)** 구조로 구현됨(README는 "Pre-norm"이라 표기, `transformer_layer.h:13` → 불일치).

실행 순서 (`transformer_layer.cpp:95-124`):
1. `multi_head_attention(input,...,attn_out)` (`:96`)
2. **Residual Add**: `residual_1 = input + attn_out`, `#pragma HLS PIPELINE II=1` (`:99-104`)
3. **LayerNorm1**: `layer_norm_newton_raphson(residual_1, ln1_out, ln1_gamma, ln1_beta)` (`:107`)
4. **FFN1**: `ffn_linear1(ln1_out, ffn_w1, ffn_b1, ffn_hidden)` (`:110`)
5. **FFN2+ReLU**: `ffn_linear2_with_relu(ffn_hidden, ffn_w2, ffn_b2, ffn_out)` (`:113`)
6. **Residual Add**: `residual_2 = ln1_out + ffn_out` (`:116-121`)
   - 주의: 두 번째 잔차의 입력이 원래 인코더 입력이 아니라 **`ln1_out`** (= LayerNorm1 출력). 표준 BART/Marian Post-norm 잔차와 동일한 토폴로지로 추정.
7. **LayerNorm2** → `output` (`:124`)
8. **패딩 마스킹**: `current_len` 이후 행을 0으로 강제(`:127-132`). 가변 길이 시퀀스 처리.

특징/근거:
- **인터페이스**: 모든 포트가 `m_axi`, 가중치마다 별도 `gmem` 번들(gmem0~gmem8)로 분리해 HBM/DDR 동시 접근 대역폭 확보. (`:69-86`)
- **중간 버퍼는 `static`**: `attn_out, residual_1, ln1_out, ffn_hidden, ffn2_out, residual_2` (`:88-93`). on-chip 유지로 재로딩 방지.
- **레이어 카운터 잔재**: `g_layer_counter`를 증가시켜 6으로 리셋(`:46`, `:134-135`)하나 실제 분기에 쓰이지 않음 → 디버그 잔재로 추정.
- `transformer_encoder_layer_no_ln`은 헤더에만 선언, .cpp에 정의 없음 → 미사용 스텁 추정. (`transformer_layer.h:80-90`)

---

### 3.2 멀티헤드 어텐션 — `multi_head_attention.cpp` (`:116-195`)

데이터플로우: **"풀 512×512 투영 후 헤드 슬라이싱"** 방식 (헤드별 부분행렬을 따로 곱하지 않음).

1. **QKV 풀 투영** (`:143-151`): `matmul_full_projection`을 Q/K/V 각각 호출. `[16][512]@[512][512]+[512]=[16][512]`.
   - 내부 (`:30-51`): 출력 `[i][j]`마다 `sum=bias[j]` 시작, `k`에 대해 `input[i][k]*weights[k][j]` 누산. `#pragma HLS PIPELINE II=1`은 `j` 루프에 적용(`:38`), 내부 `k`(512)는 펼치지 않음 → **j당 ~512 사이클의 직렬 MAC**. 처리량보다 면적 절약 지향(추정).
2. **헤드 루프** (`:167-186`, 8회): 각 head h에 대해
   - `extract_head_from_projection`로 512차원 출력에서 `[h*64 : h*64+64]` 슬라이스 추출 (`:82-94`).
   - `attention_scores(Q_head,K_head,scores,current_len)` (`:175`)
   - `softmax_hybrid(scores,attn_weights)` (`:178`)
   - `attention_output(attn_weights,V_head,head_out)` (`:182`)
   - `insert_head_output`로 concat 버퍼 `[16][512]`에 되끼움 (`:99-111`, `:185`).
   - **주의**: 8개 head를 순차 루프로 처리 → 헤드 병렬화 없음. 면적 최소화 의도(추정).
3. **출력 투영** (`:194`): `matmul_output_proj(concat_buffer, W_o, b_o, output)` (`:57-76`).

근거/한계:
- 헤드 슬라이스/삽입에도 `m_axi` 포트가 톱에 선언(`:129-138`)되나, 실제로는 톱레벨에서 직접 호출되어 함수 경계의 인터페이스 pragma는 dead일 가능성(서브함수로 호출될 때 무시) — 합성 시 인라인 여부에 따라 다름(추정).
- 헤드 슬라이스/concat이 별도 루프로 BRAM round-trip을 발생 → 면적/지연 측면 비효율(한계).

---

### 3.3 어텐션 스코어 — `attention_scores2.cpp` (`:4-76`)

`scores[i][j] = (Qᵢ·Kⱼ) * scale`, 패딩 마스킹 포함.

- **로컬 캐싱**: `Q_global/K_global`을 `Q_local/K_local`로 복사, `dim=2 complete` 파티션으로 64차원 동시 접근. (`:14-19`, READ 루프 `:22-38`)
- **스케일/마스크 상수**: `scale_factor=0.125`(= 1/8 = 1/√64, HEAD_DIM=64이므로 1/√d_k 정확), `mask_val=-1000.0`. (`:41-42`)
- **컴퓨트** (`:44-65`): `j>=seq_len`이면 `mask_val`로 채우고(패딩 마스킹), 아니면 `dot_product += Q_local[i][k]*K_local[j][k]`(k=0..63) 후 `*scale_factor`. PIPELINE II=1은 `j` 루프(`:48`).
- **근거 주의**: 주석에 "Keep your existing read loops"(`:21`) 등 개발 흔적 다수 — 리팩토링 미완 상태. 내부 `k` 루프는 unroll/partition 없어 j당 64 MAC 직렬(추정).

---

### 3.4 Softmax — `softmax2.cpp::softmax_hybrid` (`:12-93`)

행 단위 안정화 softmax를 **piecewise 근사**로 구현(`exp()`/LUT 미사용, 활성 경로).

행(i)마다 3단계 (`:22-91`):
1. **row_max 찾기** (`:27-33`): PIPELINE II=1.
2. **exp 근사 + 누산** (`:41-67`): `x = scores[i][j]-row_max`에 대해
   - `x <= -8.0` → 0 (tail cutoff) (`:49-50`)
   - `-8.0 < x < -2.3` → 선형 램프 `0.1 + (x+2.3)*0.08` (`:53-54`)
   - `x >= -2.3` → 2차 Taylor `1 + x + 0.5x²` (`:57-60`)
   - 음수 클램프 후 `exp_sum_hybrid`에 `safe_sum_t`로 누산 (`:63-66`)
3. **정규화** (`:74-89`): sum==0 가드(`:75`), **나눗셈 1회**로 `inverse_val=1/final_sum`(`inv_t`, `:79`), 이후 곱셈으로 `attn_weights[i][j]=exp*inverse_val`(`:81-88`). "divider per row"로 자원 절약 명시(`:77-78`).

스텁: `softmax_lut/piecewise/taylor_improved`는 링커용 빈 함수 (`:96-99`). 헤더(`softmax2.h:7-28`)에 4종 변형 선언만 존재 → 실제 활성은 hybrid 1종.

품질 함의: 2차 Taylor는 `x∈[-2.3,0]`에서만 정확, 선형 램프 구간은 거친 근사 → softmax 분포 왜곡 가능(한계, 그래서 `score_t/softmax_out_t`를 별도 비트폭으로 스윕한 것으로 추정).

---

### 3.5 어텐션 출력 — `attn_output.cpp::attention_output` (`:10-82`)

`output[i][d] = Σⱼ attn_weights[i][j] * V[j][d]`, "Read-Compute-Write" 패턴.

- Read V (dim=2 complete 파티션, `:30`), Read attn_weights를 로컬로 복사 (`:33-50`).
- **컴퓨트** (`:54-71`): `(i,d)`마다 PIPELINE II=1(`:59`), 내부 `j`(SEQ_LEN=16) 루프는 **`#pragma HLS UNROLL`**(`:66`) → 16개 곱셈 병렬. attn 출력은 어텐션에서 가장 병렬화된 부분.
- Write back (`:74-81`).

---

### 3.6 LayerNorm — `layer_norm_cl.cpp::layer_norm_newton_raphson` (`:11-103`)

이름은 Newton-Raphson이나 **실제로는 `hls::rsqrt` IP 사용**(`:81`). 단일패스 통계.

- **파라미터 로컬화**: gamma/beta를 BRAM 로컬로 버스트 로드 (`:30-39`), `row_buffer`는 `ram_2p`/`bram` 바인딩 (`:24-25`).
- **단일패스 통계** (`:53-66`): 행을 한 번 읽으며 `sum_x`와 `sum_sq_x` 동시 누산, `row_buffer`에 저장. PIPELINE II=1.
- **평균/분산** (`:72-81`): `var = E[x²] - (E[x])²`, 음수 클램프(`:77`), `inv_std = hls::rsqrt(var + 1e-5)`(`:81`).
- **정규화** (`:86-101`): `(x-mean)*inv_std*gamma + beta`, 로컬 버퍼에서 읽어 PIPELINE II=1.
- 근거: `epsilon=1e-5f`(`:41`). 분산 음수 가드는 고정소수점 오차 방어(`:75-77`).

---

### 3.7 FFN 1층 — `ffn_linear1.cpp` (`:88-136`)

`[16][512]@[512][2048]+[2048]` 을 **hidden 차원 타일링**으로 처리.

- **타일 상수** (`ffn_linear1.h:9-15`): `H_TILE_SIZE=512`, `D_UNROLL_FACTOR=64`, `H_NUM_TILES=2048/512=4`.
- **입력 파티션**: `input_local` dim=2 cyclic factor=64, `weights_local` dim=1 cyclic factor=64 → reduction(D_MODEL) 64-way 병렬 의도. (`:101-104`)
- **타일 루프** (`:120-135`, 4회): read_weights_tile → read_bias_tile → compute_tile → write_output_tile.
- **compute_tile** (`:39-65`): `#pragma HLS ALLOCATION operation instances=mul limit=64`로 **DSP 64개 상한**(`:46`). 1D 평탄화 루프 `iter=SEQ_LEN*H_TILE_SIZE`에 PIPELINE II=1(`:48-51`), 내부 d(512) 누산. DSP 절약과 처리량 사이 타협(근거: limit=64).

---

### 3.8 FFN 2층 — `ffn_linear2.cpp::ffn_linear2_with_relu` (`:73-118`)

`[16][2048]@[2048][512]+[512]`, **ReLU 융합**(GELU 아님).

- **타일 상수** (`ffn_linear2.h:13-19`): `FFN2_INPUT_TILE_SIZE=512`, `FFN2_UNROLL_FACTOR=64`, `FFN2_NUM_TILES=2048/512=4`.
- **융합 ReLU**: `read_input_tile_with_relu`가 입력 읽으며 `if(val<0)val=0` (`:8-26`, 특히 `:21`). 별도 활성화 단계 없이 메모리 read에 융합.
- **누산기 초기화 = bias** (`:94-100`), 타일별 부분합을 `output_accum`에 누적 (`:65`).
- **compute_tile_ffn2** (`:47-68`): 내부 h 루프 `#pragma HLS UNROLL factor=64`(`:61`)로 reduction 64-way. `output_accum` dim=2 cyclic factor=64 파티션(`:86`).
- `ffn_linear2`는 with_relu로 위임하는 래퍼 (`:121-128`).

---

### 3.9 (미사용) `matmul_qkv.cpp` — DATAFLOW 예제 (`:87-113`)

톱레벨 합성에 **포함되지 않은** head 단위 matmul(`[16][64]@[64][64]`). 그러나 본 repo에서 유일하게 **클래식 DATAFLOW 분해**를 보여주는 교과서적 예제라 분석 가치 있음.

- `read_weights / read_bias / read_input / compute_matmul / write_output` 5단계 분리 (`:4-84`).
- 톱(`:87-112`): `#pragma HLS DATAFLOW`(`:106`)로 5단계를 스트리밍 파이프라인화, 로컬 파티션 cyclic factor=16(`:103-104`), `compute_matmul`에 `mul limit=32`(`:52`).
- `config/hls_config.cfg`의 `syn.file` 목록에 **없음** → 미합성. 다른 커널들이 DATAFLOW를 안 쓰는 것과 대조적이라, 폐기된 초기 설계로 추정.

---

## 4. 데이터 플로우 (엔드투엔드)

```
[SW: PyTorch]                              [HW: HLS C-sim]                        [SW: 평가]
weightExtraction.py ──(transpose Wᵀ)──▶ weights_bin/*.bin ──load_bin──▶ newDebugTb.cpp
embedding.py ──(scale·√d_model + pos)──▶ input_embeddings.bin ──┘   │  6-layer 루프
fetchTestSentence.py ──(테스트문장)──▶ embedding/comet 입력            │  transformer_encoder_layer ×6
                                                                     ▼
                                       hls_output_full_encoder[_exp].bin
                                                                     │
                              comet2.py: BaseModelOutput으로 PyTorch 디코더에 주입
                              ──▶ model.generate(num_beams=4) ──▶ 번역문
                              ──▶ COMET / BLEU / BERTScore ──▶ master_experiment_comparison.csv
                                                                     │
                              plotCometResults.py: 비트폭 vs COMET 히트맵/곡선 → plots/*.png
```

레이어 내부 데이터플로우 (`transformer_layer.cpp:95-124`):
```
input ─▶ MHA ─▶ (+input) ─▶ LN1 ─▶ FFN1 ─▶ ReLU+FFN2 ─▶ (+LN1out) ─▶ LN2 ─▶ output
                                                                          └▶ pad rows=0
```
MHA 내부 (`multi_head_attention.cpp:143-194`):
```
input ─▶ [Wq,Wk,Wv 풀투영] ─▶ {8× head: slice→scores(Q·Kᵀ/8)→softmax→·V→concat} ─▶ Wo 투영 ─▶ output
```

근거: SW↔HW 인터페이스는 **순수 binary 파일**(float32). `load_bin`이 `weights_bin/` 경로에서 읽음(`newDebugTb.cpp:156-173`), Python은 동일 폴더에 저장(`weightExtraction.py:73-99`, `embedding.py:124-137`). **가중치 전치**가 SW에서 일어남: PyTorch `[out][in]` → HLS `[in][out]`(`weightExtraction.py:55-62`).

---

## 5. HW/SW 매핑

| 단계 | 실행 위치 | 파일 근거 |
|---|---|---|
| 가중치 추출·전치·양자화 준비 | SW(PyTorch CPU) | `weightExtraction.py:26-99` |
| 토큰 임베딩 + √d_model 스케일 + 위치임베딩 | SW(PyTorch) | `embedding.py:87-114` |
| 테스트 문장 큐레이션(UN corpus) | SW | `fetchTestSentence.py:1-95` |
| **인코더 레이어 연산(MHA/FFN/LN/Softmax)** | **HW(HLS, C-sim)** | `transformer_layer.cpp`, `multi_head_attention.cpp` 등 |
| 6-layer 반복 오케스트레이션 | SW(테스트벤치) | `newDebugTb.cpp:340-412` (ping-pong buf_A/buf_B `:345-346`) |
| 디코더(generate, beam search) | SW(PyTorch) | `comet2.py:228-237` |
| 품질 평가(COMET/BLEU/BERT) | SW | `comet2.py:256-271` |

핵심 매핑 함의:
- **인코더만 HW**. 디코더는 PyTorch에 그대로 둠 → HW 인코더 출력을 `BaseModelOutput.last_hidden_state`로 디코더에 주입하는 **하이브리드 검증** 방식(`comet2.py:225-235`). XR/실시간 추론용 완결 SoC가 아니라 **인코더 양자화 영향 평가용**.
- **레이어 반복은 SW 루프**: HW는 1개 레이어만 합성, 6번 호출은 테스트벤치가 담당(`newDebugTb.cpp:340`, `NUM_LAYERS=6` `:12`). 즉 단일 레이어 재사용(weight reload) 구조. 처리량 관점에서 레이어 파이프라이닝 없음(한계).
- **가변 시퀀스**: `current_len`을 끝까지 전파해 마스킹(`attention_scores2.cpp:51`, `transformer_layer.cpp:127`).

---

## 6. 빌드·실행

### 6.1 HLS (`config/hls_config.cfg`)
- `part=xczu7ev-ffvc1156-2-e` (`:1`), `flow_target=vivado`, `package.output.format=ip_catalog`, `syn=false`(합성 비활성, C-sim 위주). (`:3-6`)
- 톱 함수: **`syn.top=transformer_layer`** (`:32`). 단, 실제 정의된 함수명은 `transformer_encoder_layer`(`transformer_layer.cpp:48`) → **cfg의 top 이름과 코드 함수명 불일치**(확인된 불일치; HLS가 못 찾을 가능성 또는 cfg가 구버전일 가능성, 추정).
- TB: `tb.file=../hls/src/newDebugTb.cpp` (`:9`). syn.file에 전 커널 + 데이터 헤더 나열 (`:11-30`).
- 실행: `cd config && vitis_hls -f hls_config.cfg` (`README.md:78-80`). 요구: Vitis HLS 2022.2+ (`README.md:100`).

### 6.2 Python 워크플로우 (`README.md:90-96`)
1. `python python/weightExtraction.py` → `weights_bin/*.bin`
2. `python python/embedding.py` → `input_embeddings.bin`, `sequence_lengths.bin`
3. Vitis HLS C-sim 실행(testbench) → `hls_output_full_encoder*.bin`
4. `python python/comet2.py` → `master_experiment_comparison.csv`
5. `python python/plotCometResults.py` → `plots/*.png`

> 주의: `weightExtraction.py`는 **layer 0 가중치만** 추출(`:14-15`)하나, TB는 **layer0~5 6개**를 `layerN_*.bin`으로 로드(`newDebugTb.cpp:196-215`). 즉 6-layer 실행을 위해선 6개 레이어분 bin이 필요한데 추출 스크립트는 1개분만 생성 → **스크립트 누락/외부 생성분 의존**으로 추정(확인된 갭).

---

## 7. 의존성

- **HLS C++**: `ap_fixed.h`(`attention_types.h:4`), `hls_math.h`(`attention_scores2.cpp:2`, `layer_norm_cl.cpp:2` → `hls::rsqrt`), `<iostream>/<cmath>/<fstream>` (TB·일부 커널).
- **Python**: `torch`, `transformers`(MarianTokenizer/MarianMTModel), `numpy`, `pandas`, `sacrebleu`(BLEU), `bert_score`, `comet`(Unbabel wmt22-comet-da), `datasets`, `matplotlib`. (`comet2.py:11-18`, `README.md:72`)
- **모델/데이터 의존**: `Helsinki-NLP/opus-mt-en-es`(`embedding.py:8`), `Helsinki-NLP/un_pc` en-es 스트리밍(`fetchTestSentence.py:5`), `Unbabel/wmt22-comet-da`(`comet2.py:141`).
- **파일 인터페이스 의존**: `weights_bin/` 디렉토리 규약(layer 프리픽스 명명, float32 raw)이 SW↔HW 계약. 변경 시 양쪽 동기 필요.

---

## 8. 강점·한계

### 강점
- **체계적 양자화 DSE 프레임워크**: 단일 `attention_types.h` 토글로 비트폭을 바꾸고(`:10`, `:101-119`의 다수 후보), 번역 품질(COMET)로 정량 평가하는 **완결된 PPA-품질 트레이드오프 루프**. attn/ffn1/ffn2/bias/acc 5개 그룹을 분리 스윕(`comet2.py:51-70`, `plotCometResults.py:29-35`).
- **HW-친화 비선형 근사**: softmax를 piecewise-Taylor로(`softmax2.cpp:47-60`), layernorm을 `hls::rsqrt`로(`:81`) — exp/div 회피. 나눗셈을 행당 1회로 축소(`softmax2.cpp:79`).
- **실모델 검증**: opus-mt 실가중치 + HW 인코더 출력을 PyTorch 디코더에 재주입하는 end-to-end 품질 평가(`comet2.py:225-271`). 합성 정확도가 아니라 **태스크 정확도**로 평가.
- **가변 길이 + 패딩 마스킹** 일관 처리(`current_len` 전파).
- **메모리 계층 위생**: read-compute-write 분리, 로컬 BRAM 캐싱, gmem 번들 분리(`transformer_layer.cpp:69-86`).

### 한계
- **처리량 최적화 부족**: 8 head 순차 루프(병렬 없음, `multi_head_attention.cpp:167`), 풀 512 투영의 내부 reduction 직렬, 6 layer를 SW 루프로 반복(HW 레이어 파이프라인 없음). → 고처리량 가속기라기보다 **기능/정확도 프로토타입**.
- **DATAFLOW 미적용**: 유일한 DATAFLOW 예제(`matmul_qkv.cpp:106`)는 미합성. 톱레벨 커널 간 스트리밍 없음 → 커널 경계마다 BRAM round-trip.
- **문서 vs 코드 불일치 다수**: GELU(README) vs ReLU(코드), Pre-norm(README/헤더) vs Post-norm(코드), README 비트폭 예시(`<10,2>`) vs 활성 SWEEP(`<16,8>`), cfg top명(`transformer_layer`) vs 함수명(`transformer_encoder_layer`).
- **재현성 갭**: `weightExtraction.py`는 layer0만 추출하나 TB는 6 layer 요구(6절 참조).
- **잔재 코드**: 미사용 LUT 헤더, opus_mt_weights/embeddings 헤더, no_ln 스텁, g_layer_counter, 개발용 주석 다수 → 리팩토링 미완.

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

> 우리 프로젝트는 "고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 추정. 그 관점에서:

- **직접 재사용 가능**: 
  - softmax piecewise 근사(`softmax2.cpp:47-89`)와 layernorm `hls::rsqrt` 단일패스(`layer_norm_cl.cpp:53-101`)는 ViT 인코더에도 그대로 이식 가능한 **HW-친화 비선형 블록**. exp/div 회피 패턴이 핵심.
  - **양자화 DSE 방법론**: `attention_types.h` 단일 토글 + COMET류 태스크 메트릭 스윕(`comet2.py`/`plotCometResults.py`)은 우리 ViT 양자화(별도 ViT-Quantization repo 존재)에서 **품질-비트폭 곡선** 생성 템플릿으로 차용 가능. 우리 메트릭은 mAP/시선 정확도(픽셀 오차)로 치환.
- **반면교사(우리가 넘어야 할 점)**:
  - HG-PIPE의 핵심인 **레이어 파이프라이닝/헤드 병렬화/DATAFLOW**가 본 repo엔 사실상 없음. 우리는 head 병렬 + 레이어 간 DATAFLOW로 처리량을 확보해야 함(본 repo는 그 부재가 명확한 대조군).
  - **인코더만 HW + 디코더 SW** 하이브리드는 XR 실시간(저지연) 요구와 맞지 않음. 우리 시선추적 ViT는 **완결 온칩 추론**이 목표여야 함(추정).
- **검증 패턴 차용**: HW 출력 .bin을 SW 상위 모델에 재주입해 **태스크 정확도로 검증**하는 방식(`comet2.py:225-235`)은, 우리 XR 파이프라인에서 "FPGA 시선추정 출력 → 다운스트림 시선 보정/렌더링 정확도"로 평가하는 데 동일하게 적용 가능.
- **주의/리스크**: 본 repo의 문서-코드 불일치가 심하므로, 인용 시 **반드시 코드(라인) 기준**으로 확인할 것. 특히 활성 비트폭은 `attention_types.h:32-45`(16비트)이지 README가 아님.

---

### 부록: 라인 근거 인덱스 (요약)

| 주제 | 핵심 근거 |
|---|---|
| 차원·타입 | `attention_types.h:19-27`, `:32-45`, `:101-102` |
| 톱레벨 흐름 | `transformer_layer.cpp:95-132` |
| MHA 구조 | `multi_head_attention.cpp:143-194` |
| 스코어/스케일/마스크 | `attention_scores2.cpp:41-65` |
| softmax 근사 | `softmax2.cpp:47-89` |
| attn 출력 unroll | `attn_output.cpp:54-71` |
| layernorm rsqrt | `layer_norm_cl.cpp:53-101` |
| FFN1 타일/DSP한도 | `ffn_linear1.cpp:46-65`, `.h:9-15` |
| FFN2 ReLU 융합 | `ffn_linear2.cpp:8-68` |
| DATAFLOW(미사용) | `matmul_qkv.cpp:106` |
| SW 가중치 전치 | `weightExtraction.py:55-62` |
| SW 임베딩 스케일 | `embedding.py:87-114` |
| 양자화 스윕 평가 | `comet2.py:51-70`, `:225-271` |
| 비트폭 시각화 | `plotCometResults.py:29-75` |
| 빌드 cfg | `config/hls_config.cfg:1-32` |
| TB 6-layer 루프 | `newDebugTb.cpp:340-412` |
