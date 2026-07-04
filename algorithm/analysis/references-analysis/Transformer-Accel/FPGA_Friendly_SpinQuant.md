# FPGA_Friendly_SpinQuant 코드베이스 정밀 분석

> 대상 repo: `REF/Transformer-Accel/FPGA_Friendly_SpinQuant`
> 분석 방식: 자체 소스(.h/.cpp/.py/Makefile)를 실제 Read하여 함수/HLS커널/TAPA 태스크 단위로 분석. 라인 근거는 `파일명:라인` 형식. 미확인 항목은 "추정"/"확인 불가"로 표기.

---

## 1. 개요

LLaMA 3.2 **1B** LLM을 **SpinQuant**(회전 기반 양자화) 기법으로 양자화하여 AMD/Xilinx FPGA(U280 검증, V80 추정)에 end-to-end로 매핑한 추론 가속기다. 핵심 특징:

- **HLS 커널 라이브러리 + TAPA 데이터플로우**: 모든 트랜스포머 연산(RMSNorm, QKVO/FFN Linear, RoPE, MHA, Softmax, Swish-Gate, Residual, **FHT R4 회전**)을 streaming 커널로 작성하고, `tapa::task().invoke(...)`로 한 디코더 블록을 하나의 거대한 dataflow 그래프로 묶는다 (`SpinQuant_Prefilling_mem_opt.h:945-1054`).
- **W4A4 + W8A8 혼합 양자화**: Linear 레이어는 가중치/활성 모두 **INT4**(`pref_Linear_Layer_i4xi4`), MHA의 QK^T·AV는 **INT8**(`MHA_i8xi8.h`)로 계산. DSP 1개에 INT4 2x2 또는 INT8 1x2를 packing하여 처리량을 높임 (`PE.h:416-583`).
- **SpinQuant의 핵심 = Fast Hadamard Transform(FHT)**: FFN down-proj 직전에 R4 Hadamard 회전을 Omega-network 형태의 in-place 버터플라이로 구현 (`FHT.h:7-129`).
- **Prefilling / Decoding 2-커널 분리**: 프리필(병렬 시퀀스)과 디코딩(단일 토큰 autoregressive)을 별도 xclbin으로 빌드. 디코딩은 QKVO+FFN 가중치를 하나의 weight-stationary GEMM으로 융합하고, **on-FPGA Top-K 샘플링 + embedding lookup**까지 포함 (`SpinQuant_Decoding_mem_opt.h:1049-1123`, `Logits.h`).
- **호스트(SW)**: TAPA C++ 호스트가 토큰 임베딩 조회 → 프리필 커널 invoke → KV-cache 재배치 → 디코딩 커널 invoke → 샘플 토큰 저장의 흐름을 담당 (`SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp:755-931`).

이 repo는 "FPGA에서 LLM 자기회귀 추론 전체 파이프라인을 INT4/INT8 양자화 + Hadamard 회전으로 구현"한 완결형 참조 설계로, HG-PIPE 계열 ViT 가속기 대비 (a) 가변 시퀀스 KV-cache, (b) 회전 양자화, (c) 자기회귀 디코딩이라는 추가 도전을 다룬다.

---

## 2. 디렉토리 구조

### 자체 소스 (분석 대상)

```
FPGA_Friendly_SpinQuant/
├─ README.md                       # 빌드/실행 가이드 (TAPA, RapidStream, U280)
├─ src/
│  ├─ config.h                     # 모델/병렬도 하이퍼파라미터 (가장 중요한 설정)
│  ├─ PE.h                         # MAC Processing Element (fp32/int4/int8 DSP packing)
│  ├─ Linear_Layer.h               # systolic array GEMM + weight loader 군
│  ├─ Linear_Layer_flatten.h       # (flatten 변형)
│  ├─ quant.h                      # 동적/정적 양자화·역양자화 커널 군
│  ├─ FHT.h                        # Fast Hadamard Transform (SpinQuant R4 회전)
│  ├─ LayerNorm.h                  # RMSNorm (gamma/beta loader 포함)
│  ├─ Softmax.h                    # attention softmax
│  ├─ RoPE.h                       # Rotary Position Embedding
│  ├─ Swish.h                      # SiLU/Swish 활성
│  ├─ Residual_Layer.h            # residual add + store/load
│  ├─ MHA.h / MHA_i8xi8.h / MHA_flatten.h  # multi-head attention (INT8 QK/AV)
│  ├─ Logits.h                     # Top-K 로짓 + 샘플링
│  ├─ data_io.h                    # mmap loader/drainer, io_buffer/discard/register
│  ├─ SpinQuant_Prefilling*.h      # 프리필 top-level (mem_opt / test / v80 변형)
│  ├─ SpinQuant_Decoding*.h        # 디코딩 top-level (mem_opt / flatten / v80 변형)
│  ├─ SpinQuant_*_tb.cpp           # 각 단계별 testbench (csim/cosim 진입점)
│  ├─ SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp  # ★메인 호스트(main)
│  ├─ Linear_Layer_*_tb.cpp, MHA_*_tb.cpp           # 단위 testbench
│  ├─ encode_prompt.py / decode_answer.py           # HF 토크나이저 (프롬프트↔토큰ID)
│  └─ Makefile                     # tapa g++ / tapa compile / rapidstream-tapaopt 흐름
└─ run/
   └─ llama32_1b_latency_bench.py  # GPU(PyTorch/HF) baseline 벤치마크 (비교용)
```

### 제외 항목 (생성물/대용량 바이너리 — 이름만 언급, 분석 안 함)

- `src/parameters/*.h`, `run/parameters/*.h` : **생성된** 스케일/RoPE 테이블 헤더 (`A_s.h`, `K_s.h`, `Q_s.h`, `V_s.h`, `w_*_s_sum.h`, `RoPE_sin_cos.h`, `w_rmsnorm.h`, `w_lm_head_lm_head.h`) — 코드 include 대상이지만 자동생성 상수.
- `run/parameters/*.bin` : **양자화 가중치 바이너리** (`q_proj_L00..15.bin`, `k/v/o_proj`, `gate/up/down_proj` 등 INT4 weight, 레이어×16). 대용량 데이터.
- `run/bitstreams/*.xclbin` : 컴파일된 **비트스트림** (README 명시, repo에 동봉 추정).
- `run/U280_test_result/*.txt`, `power_log*.txt` : 실측 결과/전력 로그.
- `lm_head.bin`, `model_embed_tokens_fp32.bin` : README에 따르면 **초대형이라 미동봉**, 외부 드라이브 다운로드 필요 (`README.md:30-39`).
- `figs/` : 그림.
- third_party/vendor/node_modules/.ip_user_files : **존재하지 않음**(확인됨).

---

## 3. 핵심 모듈 정밀 분석

### 3.1 `config.h` — 설계 파라미터 중앙 집중 (가장 먼저 읽어야 할 파일)

LLaMA 3.2 1B 형상을 컴파일 타임 상수로 박아둔다 (`config.h:18-45`):
- `DECODER_LAYER_NUM 16`, `HIDDEN_DIM 2048`, `KV_HIDDEN_DIM 512`(GQA), `HEAD_DIM 64`, `INTER_DIM 8192`, `VOCAB_SIZE 128256`.
- `Q_HEAD_NUM = 2048/64 = 32`, `KV_HEAD_NUM = 512/64 = 8`, `ATTN_GROUP_NUM = 2048/512 = 4` → **GQA(group=4)** 명확 (`config.h:36-38`).
- `MAX_PRE_SEQ_LEN 1024`, `MAX_DEC_SEQ_LEN 1024`.

**병렬도 노브**(설계공간 핵심):
- 프리필 (`config.h:66-71`, "big config"): `TOKEN_PARALLEL=8`, `PRE_QKVO_W_PARALLEL=24`, `PRE_K/V_PARALLEL=16`, `PRE_FFN_W_PARALLEL=96`, `PRE_FFN_W_BLOCK_NUM=3`.
- 디코딩 (`config.h:95-107`, "mem_opt big"): `T_BLOCK_PARALLEL=4`, `T_QKVO_FFN_BLOCK_PARALLEL=16`, `DEC_HEAD_PARALLEL=8`, `DEC_QKVO_FFN_W_PARALLEL=64`, `DEC_K/V_PARALLEL=32`.
- V80 설정/toy test/small/test는 모두 주석 처리되어 보존됨 → **DSE(설계공간탐색) 흔적**이 그대로 코드에 남아있다 (`config.h:47-189`).
- `PRE_QKVO_W_PARALLEL_READ`/`PRE_FFN_W_PARALLEL_READ`는 병렬도를 2의 거듭제곱으로 올림하여 mmap 포트폭을 정렬 (`config.h:192-200`).

### 3.2 `PE.h` — MAC Processing Element & DSP Packing (저수준 데이터패스 핵심)

DSP48E2(A27·B18·C48)/DSP58(A27·B24·C58)를 명시 주석으로 두고 (`PE.h:6-7`), 데이터타입별·packing별로 다수의 PE 템플릿을 제공한다.

- **`PE_fp32xfp32_2D` (`PE.h:9-84`)**: fp32 내적 PE. U280+Vitis 2022.1 경로는 `II=4`로 4-way 누산기(`p_sum[4]`)를 펼쳐 fadd 의존 지연을 숨기고 마지막에 tree adder로 합산. Versal/U250 경로는 주석으로 보존 → **타깃 보드별 PE 변형을 한 파일에 캡슐화**.
- **`PE_fp32xfp32_pack_1x2_2D` / `pack_2x2_2D` (`PE.h:92-334`)**: 64-bit 워드에 fp32 2개를 packing해 한 PE가 1x2 또는 2x2 출력을 동시 계산. systolic array 대역폭 절감.
- **`PE_i4xi4_pack_2x2_1xDSP_2D` (`PE.h:416-479`)**: ★INT4 2x2 출력을 **단일 DSP**로. a를 `(a1<<16 | a0)`, b를 `(b1<<8 | b0)` 형태로 packing한 뒤 한 번의 `pack_a*pack_b`(`#pragma HLS bind_op ... impl=dsp`, `PE.h:461`)로 4개 부분곱을 동시 산출, 32-bit 결과를 4개 8-bit 필드로 분할하고 carry 보정(`PE.h:466-471`). **W4A4 처리량 4배의 근거**.
- **`PE_i8xi8_pack_1x2_1xDSP_2D` / `pack_2x2_2xDSP_2D` (`PE.h:483-583`)**: INT8용 1x2/2x2 packing. b를 `(b1, sign_ext, b0)` 25-bit로 부호확장 packing하여 한 DSP로 2개 부분곱.
- 1D 변형(`PE_*_1D`, `PE.h:587-830`): 디코딩(단일 토큰, M=1)용으로 A_out broadcast 제거.

### 3.3 `Linear_Layer.h` — Systolic Array GEMM (연산량 대부분 담당)

- **`systolic_array_i4xi4_pack_2x2_2D` (`Linear_Layer.h:8-84`)**: A/B를 2개씩 묶어 `block_size/2 × block_size/2` PE 격자를 `#pragma HLS DATAFLOW`로 구성. A_fifo/B_fifo는 SRL FIFO(`BIND_STORAGE ... impl=srl`), 모서리 PE에 `is_last_A/B` 템플릿 플래그로 broadcast 종단 처리(`Linear_Layer.h:45-56`). C는 2개씩 packing 드레인(`Linear_Layer.h:70-83`).
- **`pref_Linear_Layer_i4xi4` (`Linear_Layer.h:293-341`)**: 입력 시퀀스를 URAM 버퍼 `A[max_input_dim]`에 적재 후, 출력 채널 블록(`weight_parallel` 단위)마다 systolic array를 dataflow로 재호출. weight-stationary가 아니라 **input(activation)-stationary** GEMM 타일링.
- **`pref_Linear_Layer_i4xi4_blocked` (`Linear_Layer.h:344-406`)**: `block_num`개 systolic array를 UNROLL하여 FFN처럼 큰 output_dim을 여러 HBM 포트로 분산(`PRE_FFN_W_BLOCK_NUM=3` 대응). 출력은 block 인터리브 순서로 드레인(`Linear_Layer.h:396-403`).
- **weight loader 군 (`Linear_Layer.h:86-292`)**: `pref_weight_loader_int4_pack_2` 등은 HBM에 INT8(=INT4 2개 packing)로 저장된 가중치를 읽어 nibble 분리하여 스트리밍(`Linear_Layer.h:148-155`). `*_discard` 변형은 output_dim이 weight_parallel 배수가 아닐 때 padding 채널을 건너뜀.
- fp32 systolic(`Linear_Layer.h:409-630`)과 디코딩용 1D systolic(`dec_Linear_Layer_i4xi4*`, `Linear_Layer.h:829-955`)도 동일 패턴으로 제공.

### 3.4 `quant.h` — 양자화/역양자화 (정밀도 변환 허브)

SpinQuant 특성상 양자화 모드가 다양하여 8종 이상의 템플릿이 있다:
- **`pref_quant_layer_fp32_qint` (`quant.h:6-112`)**: 동적 per-token 양자화. io_parallel개 토큰을 URAM에 버퍼링하며 min/max 추적, asym(`ap_uint`)/sym(`ap_int`) 분기로 scale `s`·zero-point `b` 산출 후 라운딩 (`quant.h:53-109`). FFN/QKVO 입력 활성에 사용.
- **`pref_dequant_layer_qint_fp32` (`quant.h:115-164`)**: 활성 scale × 가중치 scale 역양자화. asym일 때 `b*weight_sum` 보정항 추가(`quant.h:147-148`) → **asymmetric GEMM 보정** 명확.
- **`pref_static_sym_per_tensor_quant_layer_fp32_qint` (`quant.h:252-293`)**: MHA용 정적 per-head scale. `input_s[block_id][H]`와 `mha_scale_factor`(=`sqrt_HEAD_DIM`)로 양자화하며 saturate 클램프 (`quant.h:272-282`). Q/K/V/A scale 테이블(`Q_s.h` 등)이 여기로 주입.
- 디코딩 변형(`dec_quant_layer_*`, `dec_MHA_*`, `quant.h:332-806`): block_parallel 차원으로 단일 토큰을 분할 양자화. `dec_dequant_layer_qint_fp32_bandwidth`(`quant.h:471-525`)는 weight scale을 묶어 읽어 대역폭 최적화.

### 3.5 `FHT.h` — Fast Hadamard Transform (SpinQuant의 정체성)

- **`pref_FHT` (`FHT.h:7-67`)**: Hadamard 변환을 **Omega network**(perfect-shuffle + butterfly)로 in-place 구현. `log2_io_hidden_dim` 스테이지(`FHT.h:30-31`) 반복하며, 각 스테이지에서 perfect shuffle 인덱스 `((i<<1)&(N-1))|(i>>(m-1))`(`FHT.h:36`) 적용 후 add/sub 버터플라이(`FHT.h:47-48`). 마지막에 `scale_factor`(=`sqrt(dim)`)로 나눠 정규화 직교성 유지(`FHT.h:62`).
- top-level에서 **FFN gate*up 직후, down-proj 직전**에 `pref_FHT_R4`로 호출되어 INTER_DIM(8192) 차원 회전 적용 (`SpinQuant_Prefilling_mem_opt.h:672-683`, 호출은 `:1037`). 이것이 SpinQuant의 "online Hadamard rotation R4" — 활성의 outlier를 분산시켜 INT4 양자화 손실을 줄이는 핵심.
- **`dec_FHT` (`FHT.h:69-129`)**: 디코딩용. `block_parallel`로 차원을 나눠 저장하므로 shuffle 인덱스를 2D(`buffer[idx%N][idx/N]`)로 재계산(`FHT.h:98-100`) — 단일 토큰의 메모리 분할 레이아웃에 맞춘 비자명한 변형.

### 3.6 `LayerNorm.h` — RMSNorm

- **`pref_Layer_Norm` (`LayerNorm.h:28-180`)**: LLaMA는 RMSNorm이므로 평균 차감 없이 제곱합만. U280 경로는 `A_square_sum[io_parallel][4]` 4-way 누산(`II=4` 언롤, `LayerNorm.h:81-107`)으로 fadd 의존성 회피, `1/sqrt(meanSq+eps)` 역수 곱으로 정규화 후 `gamma` 곱(`enble_beta`면 `beta` 가산, `LayerNorm.h:116-130`). Versal 단순 경로는 주석 보존.
- gamma/beta는 별도 loader 태스크(`pref_Layer_Norm_gamma_beta_loader`, `LayerNorm.h:6-24`)가 mmap에서 스트리밍 — TAPA에서 mmap 읽기와 계산을 분리하는 패턴.

### 3.7 `MHA_i8xi8.h` + `MHA.h` — Multi-Head Attention (INT8)

- **Q/K/V 양자화**: `pref_quant_layer_q/k/v_fp32_int8`이 정적 per-head scale로 INT8 변환. Q는 `sqrt_HEAD_DIM`을 미리 곱해 1/√d 스케일링 내장(`MHA_i8xi8.h:40-42`).
- **KV-cache**: `pref_K_buffer`→`pref_K_cache_manager`, `pref_V_buffer_transpose`→`pref_V_cache_manager`로 INT8 K/V를 HBM `k_cache`/`v_cache`에 적재. V는 AV 곱을 위해 **transpose 저장**(`MHA_i8xi8.h:78-110` 및 호출 `Prefilling_mem_opt.h:982-987`).
- **`pref_MHA_i8xi8_qxk` (`MHA_i8xi8.h:120-132`)**: Q·K^T를 INT8 systolic로 계산, 출력은 `ap_int<log2_HEAD_DIM+16>`. 이어 역양자화→`pref_causal_mask`(`MHA_i8xi8.h:213-223`)→`pref_Softmax_MHA`(`:225-236`)→재양자화→`pref_MHA_i8xi8_axv`(`:317-329`)로 AV 곱.
- 인과 마스킹·softmax·재양자화가 모두 streaming 태스크로 파이프라인에 직렬 삽입됨.

### 3.8 `SpinQuant_Prefilling_mem_opt.h` — 프리필 Top-Level (★dataflow 그래프)

한 디코더 블록 = 약 50개 streaming 태스크. 각 wrapper(`pref_*`)는 `for block_id in 0..15`로 **16개 레이어를 시간 다중화**(temporal reuse, 가중치만 HBM에서 교체)한다 (예 `:665-668`). top-level `SpinQuant_Prefilling`(`:806-1055`):

1. `pref_block_input_loader_sync` ← io_mmap에서 임베딩 로드(`:946`)
2. `pref_Layer_Norm_0` (RMSNorm) → distributor (`:954-956`)
3. **QKVO**: quant_int4 → weight loader → `pref_Linear_Layer_i4xi4_kq` → discard → dequant → `pref_RoPE_layer_kq` → q/k 분리 (`:959-967`); v/o는 시간 병합(`pref_vo_temporal_merger`)으로 같은 GEMM 재사용 (`:969-977`).
4. **MHA**: K/V INT8 양자화+캐시 → `pref_MHA_i8xi8_qxk` → dequant → causal mask → softmax → quant → `pref_MHA_i8xi8_axv` → dequant (`:981-998`).
5. `pref_Residual_Layer_0` (`:1002`) → RMSNorm_1 (`:1010`).
6. **FFN**: gate/up INT4 GEMM + Swish + Gate(elementwise mul) (`:1015-1034`) → **`pref_FHT_R4`** (`:1037`) → down-proj INT4 GEMM (`:1040-1046`).
7. `pref_Residual_Layer_1` → `pref_block_output_drainer_sync`로 io_mmap에 기록(다음 레이어/블록 입력으로 순환) (`:1051-1052`).

FFN weight loader는 `.invoke<tapa::detach, PRE_FFN_W_BLOCK_NUM>`로 다중 HBM 포트 병렬 로드(`:1016,1025,1041`). 스트림 깊이를 `HIDDEN_DIM`/`INTER_DIM` 등으로 명시해 백프레셔 데드락 방지 (`:842,846,886,923`).

### 3.9 `SpinQuant_Decoding_mem_opt.h` — 디코딩 Top-Level (자기회귀)

`SpinQuant_Decoding`(`:946-1124`)은 프리필과 구조는 같지만 단일 토큰(M=1) 최적화:
- **QKVO+FFN 가중치 융합 GEMM**: `dec_Linear_Layer_i4xi4_qkvo_FFN_half`를 2-half로 나눠 두 HBM 그룹(`w_qkvo_FFN_mmaps_half_0/1`)에서 병렬 로드 후 merge (`:1076-1081`). 즉 q/k/v/o/gate/up/down + **lm_head(vocab logits)** 가중치를 한 weight-stationary 패스로 처리.
- KV-cache는 디코딩 전용 레이아웃(`dec_K/V_cache_buffer`)로 매 스텝 새 토큰을 append (`:1089-1103`).
- FFN 경로에 `dec_FHT_R4` 회전 (`:1116`).
- **on-FPGA 샘플링**: `dec_Top_K_Sampling_Embedding_Layer`(`:1123`)가 vocab logits에서 Top-K(`Logits.h:5-` `dec_Logits_Max_K_Layer`) → softmax → 난수(`rand_seeds_mmap`) 기반 샘플 → 즉시 `vocab_lib`에서 다음 토큰 임베딩 조회 → `new_embedding_stream`으로 피드백(`dec_block_io_sync`, `:1051`). **autoregressive 루프가 FPGA 내부에서 닫힌다**.

### 3.10 호스트 `SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp` (SW 오케스트레이션)

- **가중치 적재**: `prefilling_read_int4_bin_as_int8_weight_mmap`(`:20-73`)·`*_blocked_*`(`:76-128`)가 `.bin`(INT4, [-8..7])을 두 출력채널씩 nibble packing하여 mmap에 배치. tile/lane 인덱싱으로 systolic 격자 레이아웃에 맞춤(`:57-71`).
- **임베딩**: `prefilling_read_embedding_from_bin`이 토큰 ID로 `model_embed_tokens_fp32`에서 임베딩 조회(`:771`).
- **메인 루프**(`SpinQuant_Prefilling_Decoding_test`, `:755-931`): 프롬프트 토큰 로드 → 임베딩 → `tapa::invoke(SpinQuant_Prefilling, ...)`(`:787-809`, seq_len을 `pad_factor`로 패딩 `:776-777`) → **프리필 KV-cache를 디코딩 레이아웃으로 재배치**(`:840-873`, PRE_K_PARALLEL→DEC_K_PARALLEL 인덱스 변환) → 마지막 토큰 임베딩을 디코딩 입력에 적재(`:834-836`) → `tapa::invoke(SpinQuant_Decoding, ...)`(`:880-899`) → 샘플 토큰 파일 저장(`:910-922`). 3회 반복 평균 latency 측정.
- `main()`은 단순 진입점(`:933-936`).

### 3.11 Python 유틸 & 벤치마크

- `encode_prompt.py`/`decode_answer.py`: HF `AutoTokenizer("meta-llama/Llama-3.2-1B")`로 프롬프트↔토큰ID 변환 (FPGA 입출력 전후처리).
- `run/llama32_1b_latency_bench.py`: **GPU(PyTorch/HF) baseline** — TTFT/throughput + NVML 전력 측정 (`:31-50`). FPGA 설계와 무관한 **비교 기준**.

---

## 4. 데이터 플로우

### 프리필 (배치 시퀀스, fp32 io ↔ INT4/INT8 내부)
```
io_mmap(fp32 emb)
  → RMSNorm0 → [quant i4 → i4xi4 GEMM(QKVO) → dequant fp32] → RoPE
  → INT8 quant → KV-cache(HBM) → QK^T(i8) → dequant → causal mask
  → softmax → INT8 quant → AV(i8) → dequant → +Residual0
  → RMSNorm1 → [FFN gate/up i4 GEMM → Swish×Gate] → FHT(R4 회전)
  → [FFN down i4 GEMM] → +Residual1 → io_mmap (다음 레이어 입력)
```
16 레이어를 `block_id` 루프로 시간 다중화. 가중치만 HBM 교체, 연산 PE는 재사용.

### 디코딩 (단일 토큰 autoregressive, FPGA 내부 폐루프)
```
new token emb → RMSNorm → 융합 GEMM(QKVO+FFN+lm_head) 
  → RoPE/INT8 → KV-cache append → QK^T·softmax·AV → +Res 
  → Swish·Gate·FHT·down → +Res → vocab logits 
  → Top-K → softmax → 난수 샘플 → vocab_lib 임베딩 조회 → 다음 토큰 (피드백)
```

### 호스트 경계
SW는 (1) bin→mmap 가중치 packing, (2) 토큰 임베딩 조회, (3) 프리필↔디코딩 KV-cache 레이아웃 변환, (4) 두 커널 invoke, (5) 샘플 토큰 저장만 담당. 실제 연산·샘플링은 전부 FPGA.

---

## 5. HW/SW 매핑

| 기능 | 위치 | 근거 |
|---|---|---|
| 가중치 INT4→INT8 packing, mmap 레이아웃 | **SW(호스트)** | `..._tb.cpp:20-128` |
| 토큰 임베딩 조회 | **SW** | `..._tb.cpp:771, 834-836` |
| RMSNorm/Linear/RoPE/MHA/Softmax/Swish/Residual | **HW(HLS)** | `LayerNorm.h, Linear_Layer.h, RoPE.h, MHA_i8xi8.h, Softmax.h, Swish.h, Residual_Layer.h` |
| W4A4 INT4 GEMM (DSP 2x2 packing) | **HW** | `PE.h:416-479`, `Linear_Layer.h:8-84` |
| W8A8 MHA QK/AV | **HW** | `MHA_i8xi8.h:120-329` |
| SpinQuant R4 Hadamard 회전 | **HW** | `FHT.h`, 호출 `Prefilling_mem_opt.h:1037` |
| KV-cache (HBM) | **HW**(저장), **SW**(프리필↔디코딩 재배치) | `MHA_i8xi8.h:105-`, `..._tb.cpp:840-873` |
| Top-K 샘플링 + 다음 토큰 임베딩 | **HW(FPGA 내부 폐루프)** | `Logits.h`, `Decoding_mem_opt.h:1123` |
| 토크나이즈/디토크나이즈 | **SW(Python/HF)** | `encode_prompt.py`, `decode_answer.py` |
| 보드/주파수: U280 @300MHz(period 3.33ns), V80 추정 | **빌드** | `Makefile:6,53`, `README.md:112-114` |
| 빌드 흐름: TAPA → RapidStream floorplan/pipeline → impl | **빌드** | `Makefile:1-145` |

병렬도(`config.h`)가 HW 자원 매핑 노브: `TOKEN_PARALLEL`(시퀀스 병렬), `*_W_PARALLEL`(출력채널 병렬), `*_BLOCK_NUM`(HBM 포트 분산), `T_*_PARALLEL`(디코딩 블록 병렬).

---

## 6. 빌드·실행

- **호스트 csim**: `tapa g++ -- SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp -o ...` (`README.md:74-77`). bitstream 인자 비우면 csim, 채우면 on-board (`..._tb.cpp:13-14`).
- **합성/구현** (`Makefile`): `tapa compile --top SpinQuant_Prefilling --platform xilinx_u280_gen3x16_xdma_1_202211_1 --clock-period 3.33`(`:6,53`) → `rapidstream-tapaopt`로 floorplan/pipeline/impl (`:11-20,58-67`) → connectivity `.ini`로 HBM 채널 배치.
- **실행** (`README.md:87-92`): `./SpinQuant_Prefilling_Decoding_mem_opt --bitstream_pref ...xclbin --bitstream_dec ...xclbin`.
- **프롬프트/디코드 길이 변경**: `config.h`의 `MAX_PRE_SEQ_LEN`/`MAX_DEC_SEQ_LEN` 수정 후 재빌드 (README는 tb.cpp 상수라 하나 실제 정의는 `config.h:20-21` — 약간의 문서/코드 불일치).
- 실측 예시: prompt=1024/decode=1024 시 prefill ~1.66s, decode ~9.86s (`README.md:100-106`, `run/U280_test_result/*` 추정).

---

## 7. 의존성

- **HLS/툴**: Vitis HLS 2022.2 (U280), TAPA CLI(`tapa g++`/`tapa compile`), RapidStream(`rapidstream-tapaopt`), XRT (`README.md:41-50`). 헤더 `tapa.h`, `ap_int.h`, `ap_fixed.h`, `hls_vector.h`, `hls_math.h`, `hls_stream.h` (`config.h:4-9`).
- **호스트**: `gflags`, `tapa::aligned_allocator`, C++ STL (`..._tb.cpp:1-11`).
- **Python**: `transformers`(AutoTokenizer/AutoModelForCausalLM), `torch`, NVML/`nvidia-smi` (벤치마크 전력) — `encode_prompt.py:2`, `llama32_1b_latency_bench.py:23-24`.
- **외부 데이터**: HF `meta-llama/Llama-3.2-1B` 토크나이저, 미동봉 `lm_head.bin`/`model_embed_tokens_fp32.bin` (`README.md:30-39`).
- **타깃 보드**: AMD U280(검증), V80(추정/미완).

---

## 8. 강점·한계

### 강점
- **완결형 end-to-end LLM 추론**: 프리필+디코딩+on-FPGA 샘플링까지 FPGA 내부 폐루프로 닫음. 호스트 개입 최소.
- **DSP packing 극대화**: INT4 2x2/1xDSP(`PE.h:416-479`), INT8 1x2·2x2(`:483-583`)로 DSP당 처리량 2~4배.
- **SpinQuant R4 회전을 streaming FHT로 우아하게 구현**: Omega network in-place 버터플라이로 추가 곱셈 없이(±만) outlier 완화 (`FHT.h`).
- **TAPA dataflow로 한 블록 전체를 파이프라인화**: 약 50 태스크가 백프레셔로 자동 동기화, 명시적 스트림 깊이로 데드락 방지.
- **이식성 설계**: PE/LayerNorm/Softmax마다 U280/U250/Versal 경로를 주석으로 보존 → 보드 포팅 용이.
- **GQA(group=4) 네이티브 지원**: KV head 8, Q head 32 정확 반영.

### 한계
- **레이어 시간 다중화로 가중치 HBM 대역폭이 병목** (16 레이어 매번 HBM 재로드). 디코딩 9.86s는 메모리 바운드 신호 (`README.md:105`).
- **fp32 중간 계산 다수**: dequant→fp32 softmax/Swish/LayerNorm/FHT가 fp32 PE를 점유, 자원·전력 비효율 가능 (추정).
- **V80 경로 미완**: 추정치만, on-board 미지원 (`README.md:112-114`).
- **하드코딩된 형상**: `config.h`가 LLaMA 3.2 1B 전용. 다른 모델은 매크로·scale 헤더 재생성 필요.
- **문서/코드 사소한 불일치**: README의 seq_len 설정 위치(tb.cpp vs config.h), 다수 주석 코드/실험 변형이 혼재해 가독성 저하.
- **scale/가중치 생성 파이프라인 부재**: `*_s_sum.h`, `*.bin` 생성 스크립트(SpinQuant 학습·캘리브레이션)는 repo에 없음 — 양자화 전처리는 외부 의존 (확인 불가).

---

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 시사점

1. **DSP INT4/INT8 packing PE 직접 재사용 가능**: `PE_i4xi4_pack_2x2_1xDSP_2D`/`PE_i8xi8_pack_*`(`PE.h:416-583`)는 모델 독립적 저수준 빌딩블록. ViT의 W4A4/W8A8 GEMM에 그대로 이식하면 DSP 효율을 끌어올릴 수 있다. 시선추적용 경량 ViT라면 INT4 2x2 packing이 latency/전력에 직접 기여.

2. **TAPA dataflow + RapidStream floorplan 흐름이 HG-PIPE 파이프라인과 정합**: 한 블록 전체를 streaming 태스크 그래프로 묶고 명시적 스트림 깊이로 데드락을 피하는 패턴(`Prefilling_mem_opt.h:945-1054`)은 HG-PIPE의 "전 레이어 파이프라인"과 동일 철학. 빌드 레시피(`Makefile`)를 템플릿으로 채택 가능.

3. **SpinQuant R4 회전(FHT)은 ViT 양자화에도 적용 가치**: ViT의 LayerNorm/GELU 후 활성 outlier 문제에 online Hadamard 회전을 적용하면 INT4 정확도 회복 가능. `FHT.h`의 Omega-network 구현은 비용이 ±덧셈뿐이라 시선추적의 저지연 요구와 잘 맞는다. 단, ViT는 dim이 다르므로 `log2_io_hidden_dim` 재파라미터화 필요.

4. **XR 시선추적과의 차이 = 우리에게 유리**: 시선추적은 (a) **자기회귀 디코딩·KV-cache 불필요**(고정 입력 프레임), (b) **vocab/샘플링 불필요**, (c) 짧은 고정 시퀀스 → 이 repo의 가장 무거운 부분(디코딩 폐루프, KV-cache 재배치, Top-K)을 **제거**하고 프리필류 dataflow + FHT + INT4 GEMM만 차용하면 훨씬 단순·고처리량 설계가 된다.

5. **레이어 시간 다중화 vs 전 레이어 펼치기**: 이 repo는 16 레이어를 시간 다중화해 HBM 바운드가 됐다(한계 참조). HG-PIPE류 고처리량/저지연 목표라면 시선추적 ViT의 적은 레이어 수를 활용해 **전 레이어를 공간 펼침(가중치 on-chip 상주)**하는 방향이 유리 — 이 repo의 병목이 우리 설계 결정의 반례 데이터가 된다.

6. **per-head 정적 scale + per-token 동적 scale 혼합 전략**(`quant.h`)은 ViT 패치-토큰 양자화에 응용 가능. MHA는 정적(`Q_s/K_s/V_s/A_s`), Linear 활성은 동적이라는 분리는 정확도/하드웨어 균형의 참고 설계.

7. **호스트 측 가중치 nibble packing 코드**(`..._tb.cpp:20-128`)는 INT4 weight를 systolic 격자 tile/lane에 맞춰 배치하는 실전 레퍼런스 — 우리 INT4 ViT 가중치 적재 호스트 코드의 출발점.

---

### 근거 표기 요약
- **라인 근거로 확인**: 모델 형상/병렬도(`config.h`), DSP packing(`PE.h`), GEMM/systolic(`Linear_Layer.h`), 양자화(`quant.h`), FHT(`FHT.h`), RMSNorm/Softmax/RoPE/Swish/Residual, 프리필·디코딩 dataflow 그래프, 호스트 오케스트레이션·KV-cache 재배치·on-FPGA Top-K.
- **추정**: V80 성능, fp32 중간 연산의 자원/전력 영향, HBM 대역폭 병목 원인, `run/U280_test_result` 내용.
- **확인 불가**: SpinQuant 학습/캘리브레이션·scale·.bin 생성 파이프라인(repo 외부), 실제 비트스트림 자원 사용량 리포트.
