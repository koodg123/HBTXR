# FlexLLM 정밀 분석 (REF/ViT-Accelerator/FlexLLM)

> 분석 대상 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Accelerator\FlexLLM`
> 분석 도구: Glob/Grep/Read (라인 근거 기반). 생성물/바이너리(*.xclbin, run/ 내 실행파일, *.gguf, parameters/*.h 등)는 제외하고 이름만 언급.
> 표기 규약: 코드/주석에서 직접 확인한 사실은 단정, 코드 외 배경지식 추론은 "추정", 미확인은 "확인 불가"로 명시.

---

## 1. 개요

- **무엇인가**: FlexLLM은 LLaMA-3.2-1B를 FPGA(AMD Alveo U280)에 매핑하는 **HLS 기반 LLM 가속기 라이브러리**다. README 라인 4에서 스스로를 "a composable High-Level Synthesis (HLS) library for rapidly building hybrid temporal–spatial accelerators for Large Language Models"로 정의한다.
- **한줄요약**: SpinQuant 양자화(가중치/활성화 INT4, KV/Attention INT8)된 LLaMA-3.2-1B의 prefill·decode 파이프라인을 TAPA dataflow 태스크 그래프로 기술하고, RapidStream으로 U280 SLR floorplan/pipeline을 자동화하는, 모듈 재사용형 transformer 가속기.
- **원논문 추정**: 저장소명/README 제목이 **"FlexLLM: A Composable HLS Library for Rapid LLM Accelerator Design"**(README 라인 1). Zenodo DOI 배지(README 라인 2: `10.5281/zenodo.18793354`) 존재. 정식 학술 논문 venue는 코드만으로는 **확인 불가**(추정: FPGA/DAC 계열 시스템 논문).
- **타깃 하드웨어**: AMD Alveo **U280**, 플랫폼 `xilinx_u280_gen3x16_xdma_1_202211_1`(Tapa_Command 라인 9, impl.py 라인 33). 클럭 주기 3.33ns(≈300MHz, Tapa_Command 라인 9 `--clock-period 3.33`). README는 V80(7nm) 추정치도 언급(라인 28-31, "estimates").
- **양자화**: SpinQuant 계열. 회전(rotation) 적용을 위한 **FHT(Fast Hadamard Transform) R4** 모듈 존재(`Modules/FHT.h`), 가중치/활성화는 INT4(`ap_int<4>`), KV 캐시·attention score는 INT8(`ap_int<8>`).
- **HMT 변형**: `HMT_SpinQuant_Llama_32_1B/`는 동일 코어에 **Hierarchical Memory Transformer**(추정: README 라인 16 "Hierarchical Memory Transformer (HMT) Plug-In"의 약어 확장) 메모리 토큰 메커니즘을 plug-in 한 long-context 변형. (질의서의 "Hybrid?/Hierarchical Memory Tiling 추정"은 코드 근거상 **Hierarchical Memory Transformer**가 정확.)

---

## 2. 디렉토리 구조

### 2.1 자체 소스 트리 (핵심)

```
FlexLLM/
├─ README.md                              # 라이브러리 소개/빌드/실행
├─ Modules/                               # 공통 HLS 모듈 라이브러리 (-I$FLEXLLM_HOME/Modules)
│  ├─ config.h                            # 모듈측 기본 파라미터
│  ├─ PE.h                                # DSP 패킹 PE (fp32/i8xi8/i4xi4)
│  ├─ Linear_Layer.h                      # 시스톨릭 i4xi4 GEMM 템플릿
│  ├─ Linear_Layer_flatten.h
│  ├─ MHA.h / MHA_flatten.h               # i8xi8 QxK, AxV 시스톨릭 어텐션
│  ├─ Softmax.h / Softmax_backup.h        # 온라인 softmax
│  ├─ quant.h                             # 양자화/역양자화 (sym/asym, per-tensor)
│  ├─ RoPE.h                              # 회전 위치 임베딩
│  ├─ LayerNorm.h / Residual_Layer.h     # RMSNorm/LayerNorm, residual add
│  ├─ Swish.h                             # SiLU/Swish (FFN gate)
│  ├─ FHT.h                               # Fast Hadamard Transform (SpinQuant R4 회전)
│  ├─ Logits.h                            # Top-K 샘플링/logits
│  ├─ data_io.h                           # mmap loader/drainer/buffer/distributor
│  └─ HMT.h                               # 계층적 메모리 토큰 모듈
│
├─ SpinQuant_Llama_32_1B/                 # ★ 기본 SpinQuant 변형 (분석 주대상)
│  ├─ config_u280.h / config_u280_mem_opt.h / config_u280_mem_opt_new.h
│  ├─ SpinQuant_Prefilling.h              # prefill 톱레벨 TAPA 태스크 그래프
│  ├─ SpinQuant_Prefilling_mem_opt.h
│  ├─ SpinQuant_Decoding.h                # decode 톱레벨 TAPA 태스크 그래프
│  ├─ SpinQuant_Decoding_mem_opt.h / _new.h
│  ├─ MHA_i8xi8.h                         # prefill용 INT8 MHA 래퍼(블록 루프)
│  ├─ Linear_Layer_test.h                # 단위 Linear Layer 테스트 래퍼
│  ├─ llama_tokenizer.h                   # llama.cpp gguf 토크나이저 연동
│  ├─ *_tb.cpp                            # 테스트벤치 (csim)
│  ├─ SpinQuant_Prefilling_Decoding_mem_opt_demo.cpp  # end-to-end 호스트 데모
│  ├─ *.ini                               # TAPA link config (HBM/DDR 매핑)
│  ├─ Tapa_Command                        # 빌드 레시피 (make 스타일)
│  ├─ RapidStream_pref_u280/*.py          # prefill floorplan/pipeline/device/impl 생성
│  ├─ RapidStream_dec_u280/*.py           # decode  동일
│  └─ run/                                # 호스트/벤치/파워로거 스크립트 (실행물 제외)
│
├─ SpinQuant_Llama_32_1B_Ins/            # Instruct 모델 변형 (구조 동일, run/ 포함)
│
└─ HMT_SpinQuant_Llama_32_1B/            # ★ HMT(long-context) 변형
   ├─ HMT_SpinQuant_Unit.h               # HMT 세그먼트/메모리 토큰 유닛
   ├─ HMT_SpinQuant_Prefilling.h         # HMT prefill 톱레벨 (seg_len 스트림 제어)
   ├─ HMT_SpinQuant_Prefilling_mem_opt.h / _test.h
   ├─ SpinQuant_*.h (Prefilling/Decoding) # 기본 변형과 동일 코어 재사용
   ├─ config_u280*.h                     # HMT_SEG_LEN/MEM_NUM 등 추가 정의
   ├─ hmt_*.ini / *.ini                   # HMT 전용 link config
   ├─ Tapa_Command
   └─ Rapidstream_pref_u280/*.py
```

### 2.2 두 변형 비교 (SpinQuant vs HMT)

| 항목 | SpinQuant_Llama_32_1B | HMT_SpinQuant_Llama_32_1B |
|---|---|---|
| 목적 | 표준 prefill+decode 추론 | 동일 코어 + 장문맥(long-context) 처리 |
| 톱레벨 prefill | `SpinQuant_Prefilling`(SpinQuant_Prefilling.h:807) | `HMT_SpinQuant_Prefilling`(HMT_SpinQuant_Prefilling.h) |
| 추가 제어 | 없음 | `hmt_pref_control`(seg_len 스트림 fan-out, HMT_SpinQuant_Prefilling.h:10-26) |
| 세그먼트 처리 | 단일 시퀀스 | 세그먼트 루프 `for(seg_len=...; seg_len!=0; ...)` (라인 14, 36, 50) |
| 핵심 config | DECODER_LAYER_NUM=16, HIDDEN_DIM=2048 | 동일 + `HMT_SEG_LEN 990`, `HMT_SUM_SEG_LEN 495`, `HMT_REC_SEG_LEN 32`, `MEM_NUM 64`, `HMT_MAX_SEG_NUM 64` (config_u280.h:65-72) |
| 모듈 | 동일 Modules/ 재사용 | + `Modules/HMT.h` 사용 |

### 2.3 제외물 (이름만 언급, 분석 제외)
- `run/bitstreams/*.xclbin` (FPGA 비트스트림), `run/SpinQuant_Prefilling_Decoding_mem_opt*`(컴파일된 호스트 실행파일), `parameters/*.h`(다운로드 양자화 가중치 헤더), `*.gguf`(토크나이저), `work.out/`·`RapidStream/build/`(합성 산출물). README 라인 45-56이 이 배치를 설명.

---

## 3. 핵심 HLS 커널·헤더별 정밀 분석 (가장 중요)

FlexLLM의 설계 철학은 **"Modules/의 파라미터화 템플릿 + 각 변형의 얇은 래퍼"**다. `SpinQuant_*.h`의 함수들은 대부분 `decoder_block_loop`(16 레이어 반복) 안에서 `Modules/`의 `*_template`을 호출하는 1~10줄짜리 래퍼이고, 실제 연산·pragma는 Modules/에 있다. 따라서 아래 분석은 (A) Modules 코어 템플릿과 (B) 변형 래퍼/태스크 그래프를 함께 다룬다.

### 3.0 하드웨어 파라미터 (`config_u280.h`)

모델 상수 (LLaMA-3.2-1B):
- `DECODER_LAYER_NUM 16`, `HIDDEN_DIM 2048`, `KV_HIDDEN_DIM 512`(GQA), `HEAD_DIM 64`, `INTER_DIM 8192`, `VOCAB_SIZE 128256` (config_u280.h:19-46).
- 파생: `Q_HEAD_NUM = 2048/64 = 32`, `KV_HEAD_NUM = 512/64 = 8`, `ATTN_GROUP_NUM = 2048/512 = 4` → **GQA 4:1** (config_u280.h:36-38). 이는 LLaMA-3.2-1B의 grouped-query attention 구조와 일치.
- `MAX_PRE_SEQ_LEN 1024`, `MAX_DEC_SEQ_LEN 1024`, `MAX_SUM_SEQ_LEN 2048` (라인 20-22).

병렬도/타일 상수 (U280):
- **Prefill**(시퀀스 차원 병렬): `TOKEN_PARALLEL 8`, `PRE_QKVO_W_PARALLEL 16`, `PRE_K_PARALLEL 16`, `PRE_V_PARALLEL 16`, `PRE_FFN_W_PARALLEL 64`(블록 2분할, `PRE_FFN_W_BLOCK_NUM 2`) (라인 49-54).
- **Decode**(배치/블록 차원 병렬): `T_BLOCK_PARALLEL 4`, `T_QKVO_FFN_BLOCK_PARALLEL 8`, `DEC_HEAD_PARALLEL 4`, `DEC_QKVO_FFN_W_PARALLEL 64`, `DEC_K_PARALLEL 32`, `DEC_V_PARALLEL 32` (라인 56-62).
- `*_W_PARALLEL_READ`: INT4 가중치가 1바이트에 2개 패킹되어 HBM에서 읽히므로 read parallelism은 절반(`/2` mmap, 예: `wk_wq_mmap`은 `ap_int<8>, PRE_QKVO_W_PARALLEL_READ/2`)이 됨 (라인 65-73, SpinQuant_Prefilling.h:809).

### 3.1 INT4xINT4 / INT8xINT8 PE — DSP 패킹의 핵심 (`Modules/PE.h`)

FlexLLM의 효율성 근원은 **하나의 DSP48E2로 2개의 저비트 곱을 동시에 처리하는 패킹 기법**이다.
- 헤더 주석에 자원 폭 명시: `DSP48E2: A27 B18 C48`, `DSP58: A27 B24 C58` (PE.h:6-7).
- `PE_i4xi4_pack_2x2_2D`(PE.h:373-412): 한 PE가 입력 a(8비트=INT4 2개 a0,a1)와 가중치 b(8비트=INT4 2개 b0,b1)를 받아 **b0,b1을 하나의 와이드 워드 `pack_b`로 합성**(라인 391-393: `pack_b = (b1_temp, b0_sign_ex, b0)`), 단일 누산기에서 `pack_c += a0*pack_b`, `+= a1*pack_b`로 곱한 뒤 상·하위 비트 구간을 분리해 4개 부분합 `c_00,c_01,c_10,c_11`를 복원(라인 406-412). sign 보정(`c_01 += c_00[msb]`, 라인 410-411)으로 부호 있는 INT4 정확도 유지.
- `PE_i4xi4_pack_1x2_2D`(PE.h:338-368)는 1×2 변형(가중치만 2-패킹). `is_uint_A` 템플릿 인자로 비대칭 양자화(활성화 unsigned) 분기(라인 357-360).
- 동일 패턴의 `PE_i8xi8_pack_2x2_2xDSP_2D`(MHA.h:152-158에서 호출)와 `PE_fp32xfp32_pack_*`(PE.h:93,200)도 존재 → INT4(Linear), INT8(MHA), fp32(검증경로)를 동일 시스톨릭 골격으로 통일.

### 3.2 Linear Layer — 시스톨릭 INT4 GEMM (`Modules/Linear_Layer.h`)

- `systolic_array_i4xi4_pack_2x2_2D`(Linear_Layer.h:9-): 2D 시스톨릭 어레이. `A_fifo`/`B_fifo`/`C_fifo`는 SRL FIFO로 바인딩(`#pragma HLS BIND_STORAGE ... impl=srl`, 라인 17-26), 어레이 전체를 `#pragma HLS DATAFLOW`로 감싸 PE를 공간 전개(라인 26). PE 인스턴스화는 `block_size_a/2 × ...`를 `#pragma HLS UNROLL`로 펼침(라인 41-46), 모서리/내부 PE를 `is_last_A/is_last_B` 템플릿 플래그로 구분(라인 46-55). 이는 전형적인 **output-stationary 시스톨릭 어레이**(추정: 부분합 누산이 PE 로컬 `pack_c`에 머무름).
- `pref_Linear_Layer_i4xi4`(Linear_Layer.h:294-341): 한 GEMM 타일의 dataflow 본체.
  - 입력 활성화를 URAM 버퍼 `A[max_input_dim]`에 적재(`#pragma HLS bind_storage variable=A type=ram_2p impl=uram`, 라인 303) → **가중치 재사용을 위해 활성화 on-chip 상주**.
  - `io_block_loop`(M = seq/io_parallel)로 시퀀스 타일링(라인 315-316), `weight_block_loop`(N = out_dim/weight_parallel)로 출력채널 타일링(라인 322), 그 안을 `#pragma HLS DATAFLOW`(라인 323)로 load→systolic→drain 파이프라인 구성.
  - 출력 비트폭은 `ap_int<max_log2_input_dim + 8>`로 누산 폭 자동 산정(라인 297).
- 변형 래퍼 예 `pref_Linear_Layer_i4xi4_q`(Linear_Layer_test.h:34-45), `pref_weight_loader_int4_pack_2_discard`(Linear_Layer_test.h:27)에서 INT4 2-패킹 가중치 로딩을 처리.

### 3.3 MHA INT8 어텐션 커널 (`Modules/MHA.h`, `SpinQuant_Llama_32_1B/MHA_i8xi8.h`)

prefill과 decode가 **서로 다른 시스톨릭 형상**을 쓴다.

**(a) QxK (score = Q·Kᵀ)**
- `pref_MHA_i8xi8_qxk_template`(MHA.h:190-243): head 루프(`attn_head_loop`, 라인 215) 안에서 Q 한 head(`mha_head_dim=64`)를 버퍼 A에 적재(라인 216-219), K를 `K_parallel`(=16) 단위로 스트리밍하며 `systolic_array_i8xi8_pack_2x2`로 곱(라인 231-233). 출력은 `ap_int<log2_HEAD_DIM + 16>`(=22비트) score(라인 193). `#pragma HLS DATAFLOW`(라인 223)로 load/compute/drain 오버랩.
- decode 경로는 `dec_MHA_i8xi8_qxk_template`(MHA.h:616-)에서 `systolic_array_i8xi8_pack_1x2_1D`(MHA.h:565-)를 사용 → **decode는 토큰 1개씩이라 1D 시스톨릭**으로 자원 절약.
- 래퍼: `pref_MHA_i8xi8_qxk`(MHA_i8xi8.h:120-132)가 16블록 루프로 템플릿 호출.

**(b) AxV (out = softmax(score)·V)**
- `pref_MHA_i8xi8_axv_template`(MHA.h:340-): 동일 시스톨릭 골격, `V_parallel`(=16) 단위, 누산폭 `log2_max_seq_len`(라인 377).

**(c) Softmax/Mask/양자화 (prefill MHA_i8xi8.h)**
- `pref_causal_mask`(MHA_i8xi8.h:213-223) → `pref_causal_mask_template`(MHA.h:247-)에서 `k <= M*io_parallel + i` 조건으로 인과 마스킹(MHA.h:262).
- `pref_Softmax_MHA`(MHA_i8xi8.h:225-236) → `Modules/Softmax.h`.
- score는 INT8로 재양자화(`pref_quant_layer_sfm_a_fp32_int8`, MHA_i8xi8.h:239-) 후 AxV에 투입 → **attention 전체 INT8 경로**.
- 입출력 데이터타입 요약: Q/K/V int8(`ap_int<8>`), score int(`ap_int<22>`)→dequant fp32→softmax fp32→requant int8→AxV→int→dequant fp32.

**(d) KV 캐시 관리**
- `pref_K_buffer`/`pref_K_cache_manager`(MHA_i8xi8.h:78-116): INT8 K를 `PRE_K_PARALLEL` 폭으로 재패킹 후 `k_cache` mmap에 적재/로드(`pref_K_cache_manager_template`, GQA `ATTN_GROUP_NUM` 인지). decode는 head별 캐시 mmap을 `DEC_HEAD_PARALLEL`개로 분산(`k_caches`, `v_caches`, SpinQuant_Decoding.h:955-956).

### 3.4 Prefilling 톱레벨 태스크 그래프 (`SpinQuant_Prefilling.h:807-1047`)

`SpinQuant_Prefilling`는 단일 `tapa::task()`(라인 940) 안에서 **약 60개의 `.invoke`**로 한 decoder block을 dataflow로 구성하고, 각 invoke 내부가 16-layer 루프를 돈다(temporal 재사용). 시그니처(라인 807-825)에 HBM mmap 다수와 `int seq_len`.

데이터 경로 순서(라인 941-1047 근거):
1. **입력/LN0**: `pref_block_input_loader_sync`(941) → `pref_iembed_distributor`(945, residual용 분기) → `pref_Layer_Norm_0`(950, RMSNorm). 잔차는 깊은 스트림 `iembed_stream_res0`(depth `8*HIDDEN_DIM`, 라인 887)에 캐싱.
2. **QKV Linear (INT4)**: `pref_quant_layer_kq_fp32_int4`(954) → 가중치 로더(955) + scale 로더(956) → `pref_Linear_Layer_i4xi4_kq`(957) → `pref_dequant_layer_kq_int_fp32`(960) → `pref_RoPE_layer_kq`(961, RoPE). V/O도 동일 패턴(964-972).
3. **K/V 양자화 + 캐시 (INT8)**: 975-982.
4. **어텐션**: `pref_MHA_i8xi8_qxk`(985) → dequant(986) → `pref_causal_mask`(988) → `pref_Softmax_MHA`(989) → requant int8(990) → `pref_MHA_i8xi8_axv`(992) → dequant(993).
5. **Residual0/LN1**: 997, 1000, 1005.
6. **FFN (SwiGLU, INT4)**: gate(1010-1017, `pref_Swish_Layer_ffn`로 SiLU) + up(1019-1025) → `pref_Gate_Layer_ffn`(1029, 원소곱).
7. **SpinQuant R4 회전**: `pref_FHT_R4`(1032) — FFN down 입력에 **Fast Hadamard Transform** 적용(`Modules/FHT.h`의 omega-network 버전, FHT.h:7-67). 이것이 SpinQuant의 in-network rotation 핵심.
8. **FFN down (INT4)**: 1035-1041 → **Residual1**(1046) → `pref_block_output_drainer_sync`(1047, io_mmap 기록).

데이터타입 흐름: fp32 활성화 ↔ INT4 양자화(Linear) / INT8(MHA) ↔ per-tensor scale(`hls::vector<float,2>` = scale+sum, 라인 810 등). dequant마다 weight scale `*_s_sum_mmap` 사용.

### 3.5 Decoding 톱레벨 태스크 그래프 (`SpinQuant_Decoding.h:946-1125`)

decode는 **single-token 자기회귀**라 병렬화 축이 다르다(시퀀스 대신 hidden 블록/head 병렬).
- 시그니처(946-961): `vocab_lib`(임베딩/언임베딩 룩업), `io_mmap`, **head별 가중치/캐시 mmap 2분할** `w_qkvo_FFN_mmaps_half_0/1`(`T_QKVO_FFN_BLOCK_PARALLEL/2`개씩, 라인 952-953), head별 `k_caches`/`v_caches`(`DEC_HEAD_PARALLEL`개, 955-956), `rand_seeds_mmap`/`sampled_token_idx_mmap`(샘플링), `pre_seq_len`+`dec_seq_len`.
- 핵심 차이:
  - 가중치 로더를 `.invoke<tapa::detach, T_QKVO_FFN_BLOCK_PARALLEL/2>`(1066-1067)로 **공간 복제(detach)** → HBM 대역폭 병렬화.
  - QKVO+FFN을 하나의 융합 GEMM 스트림(`input_qkvo_ffn_stream`)으로 합쳐 처리(`dec_qkvo_FFN_input_merger`, 1062). Linear Layer를 두 half로 broadcast/merge(1073-1076).
  - KV 캐시 매니저도 `.invoke<tapa::detach, DEC_HEAD_PARALLEL>`(1087, 1098)로 head 병렬.
  - QxK/AxV는 `dec_MHA_*`(1089, 1100, 1D 시스톨릭).
  - **출력단**: `dec_block_output_broadcastor`(1122) + `dec_Top_K_Sampling_Embedding_Layer`(1125) — logits→Top-K 샘플링→다음 토큰 임베딩까지 **온칩 폐루프**. `LOGITS_MAX_K 5`(config_u280.h:75). FFN down 입력에 `dec_FHT_R4`(1118)로 동일 R4 회전.
- 잔차/LN을 broadcast/merge 패턴(`dec_residual_broadcastor`, `dec_Layer_Norm_merger`, 1054-1059)으로 3개 LN을 단일 LN 엔진에 시분할(temporal reuse).

### 3.6 mem_opt 변형 (`*_mem_opt.h`, `config_u280_mem_opt.h`)

- prefill FFN 병렬도를 `PRE_FFN_W_PARALLEL 64→96`, 블록 `2→3`으로 상향(config_u280_mem_opt.h:53-54) → 동일 자원 내 FFN throughput↑.
- decode KV 병렬을 `DEC_K_PARALLEL = DEC_QKVO_FFN_W_PARALLEL/2 = 32`로 **가중치 병렬에 종속**시켜(라인 60-61) HBM 채널을 가중치와 KV가 공유. 실제 link config에서 한 HBM 채널에 가중치+캐시를 묶음: `w_qkvo_FFN_mmaps_half_0_k_caches_0:HBM[5]` 같이 **합성 포트명**으로 묶임(dec_link_config_u280_mem_opt.ini:8-27) → HBM 채널 수 절감이 mem_opt의 본질(추정: 32채널 한계 내 배치).
- demo 호스트(`SpinQuant_Prefilling_Decoding_mem_opt_demo.cpp`)는 mem_opt 비트스트림 2개(`bitstream_pref`/`bitstream_dec`, 라인 13-14)를 `tapa::invoke`로 순차 구동(라인 832, 932).

### 3.7 RapidStream floorplan 자동화 스크립트

`RapidStream_pref_u280/`·`RapidStream_dec_u280/` 각 4개 파이썬이 RapidStream `tapaopt`용 JSON 4종을 생성:
- `gen_device_u280.py`(전체 6줄): `get_u280_vitis_device_factory(...)` → `factory.generate_virtual_device("u280_device.json")` (라인 1-6). U280 가상 디바이스(SLR/HBM 자원) 모델 생성.
- `gen_floorplan_config.py`: `FloorplanConfig(port_pre_assignments={".*": "SLOT_X0Y0:SLOT_X0Y0"})` → `floorplan_config.json` (라인 1-6). 모든 포트를 SLOT_X0Y0(SLR0)에 사전 배치(추정: 단일 SLR 우선, mem_opt에선 `floorplan_config_mem_opt.json` 별도 사용 — Tapa_Command:43).
- `gen_pipeline_config.py`: `PipelineConfig(pp_scheme="single")` → `pipeline_config.json` (라인 1-6). SLR-crossing 신호 자동 파이프라이닝 스킴.
- `impl.py`: `ImplConfig(vitis_platform="xilinx_u280_gen3x16_xdma_1_202211_1", placement_strategy="Explore", max_workers=4, max_synth_jobs=16)` → `impl_config.json` (라인 14-41). 클럭 `ap_clk: 3.33ns`(라인 19-20), Vivado `Explore` 배치 전략.

→ RapidStream은 TAPA가 생성한 `.xo`를 받아 **floorplan(SLR 슬롯 배치) + cross-SLR pipeline 삽입 + Vitis impl 실행**을 자동화(Tapa_Command:15-24의 `rapidstream-tapaopt` 호출이 4 JSON + `--connectivity-ini`를 인자로 받음).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 prefill vs decode 경로
- **prefill**: 입력 프롬프트 전체(`seq_len`)를 시퀀스 차원 병렬(`TOKEN_PARALLEL=8`)로 한 번에 처리. 16 layer를 invoke마다 시분할 반복. 시스톨릭은 2D(`pack_2x2`). KV 캐시 생성. (SpinQuant_Prefilling.h)
- **decode**: 토큰 1개씩 자기회귀. hidden/head 차원 병렬, 가중치·KV mmap을 detach 복제. 1D 시스톨릭. logits→Top-K 샘플링→다음 토큰을 온칩 폐루프로 생성(SpinQuant_Decoding.h:1125). KV 캐시 누적 사용(`pre_seq_len + dec_seq_len`).
- 호스트는 prefill 커널 → decode 커널 순으로 두 비트스트림을 구동(demo cpp:832, 932). 둘 사이 KV/io는 HBM mmap을 통해 전달(추정: 동일 HBM 영역 공유는 link config로 결정).

### 4.2 mem_opt 최적화 차이
3.6 참조. 핵심은 (1) FFN 병렬도 상향, (2) **가중치+KV 캐시를 동일 HBM 채널에 합성 포트로 묶어 채널 수 절감**, (3) decode KV 병렬을 가중치 병렬에 결속.

### 4.3 TAPA dataflow
- 모든 커널 본체가 `tapa::task().invoke(...)` 그래프(SpinQuant_Prefilling.h:940, Decoding.h:1049). 스트림은 `tapa::stream<T, depth>`로 깊이 지정(예: `quant_a_stream` depth `MAX_PRE_SEQ_LEN`, Prefilling.h:871; `iembed_stream_res0` depth `8*HIDDEN_DIM`, 라인 887). `tapa::detach`로 무한 실행 PE 복제(Decoding.h:1066,1087,1098). `tapa::mmaps<...,N>`로 다채널 HBM 묶음(Decoding.h:952-956).
- 모듈 내부는 HLS `#pragma HLS DATAFLOW`(Linear_Layer.h:323, MHA.h:223), `PIPELINE II=1`(다수), `UNROLL`(PE 전개), `BIND_STORAGE impl=srl/uram`(FIFO/버퍼)로 구현.

### 4.4 HBM 계층
- prefill link(`pref_link_config_u280.ini`): io_mmap=HBM[16], LN gamma/beta=HBM[18]/[2], QKVO 가중치/scale=HBM[19-22], k/v_cache=HBM[23]/[24], FFN gate/up/down 가중치+scale=HBM[3-8].
- decode link(`dec_link_config_u280_mem_opt.ini`): vocab/seed/sampled=DDR[1], io=HBM[0], 가중치+KV 합성 포트=HBM[5-12](half_0), HBM[16-23](half_1), gamma/beta=HBM[14], w_scale=HBM[24]. → **DDR은 작은 제어/샘플링 데이터, HBM은 대용량 가중치/캐시**.

### 4.5 양자화 (SpinQuant)
- Linear: **INT4×INT4**(`ap_int<4>`), per-tensor symmetric/asymmetric (`pref_quant_layer_fp32_qint<4, true, ...>`, Linear_Layer_test.h:99). scale은 `hls::vector<float,2>`(scale+zero/sum).
- MHA: **INT8×INT8**(Q,K,V,score). score 누산 후 dequant→softmax(fp32)→requant int8→AxV.
- **회전(rotation)**: SpinQuant의 핵심인 학습된 직교회전을 FFN 경로에서 **FHT R4**(Fast Walsh-Hadamard, `Modules/FHT.h`)로 근사/구현. FHT는 omega-network shuffle+butterfly로 O(N log N) 구현(FHT.h:30-53), `log2_HIDDEN_DIM`/`log2_INTER_DIM` 단계.
- KV 캐시 INT8 저장으로 대역폭/용량 절감.

---

## 5. HW/SW 매핑 (HLS C++ ↔ TAPA ↔ RapidStream ↔ U280)

| 계층 | 표현 | 근거 |
|---|---|---|
| 알고리즘 | LLaMA-3.2-1B (16층, GQA 32/8 head, SwiGLU) | config_u280.h:19-43 |
| HLS 연산 커널 | `Modules/*.h` 템플릿 (PE/시스톨릭/softmax/FHT/quant) | PE.h, Linear_Layer.h, MHA.h |
| 변형 래퍼 | `SpinQuant_*.h`의 16-layer 루프 invoke | SpinQuant_Prefilling.h:24, 940 |
| TAPA 태스크 그래프 | `tapa::task().invoke(...)`, `tapa::mmaps`, `tapa::detach` | Prefilling.h:940-1047, Decoding.h:1049-1125 |
| HBM/DDR 매핑 | `*.ini` `sp=Kernel.port:HBM[n]` | pref/dec_link_config_u280*.ini |
| 합성/배치 | TAPA `compile`(.xo) → RapidStream `tapaopt`(floorplan/pipeline/impl) | Tapa_Command:9,15-24 |
| 디바이스 | U280 (SLR0 우선 배치, 32 HBM 채널, 300MHz) | gen_*_u280.py, impl.py:33, floorplan SLOT_X0Y0 |
| 호스트 | `tapa::invoke(bitstream, mmaps...)` + llama.cpp 토크나이저 | demo cpp:832,932; llama_tokenizer.h |

---

## 6. 빌드·실행 방법

`Tapa_Command`(make 타겟 스타일)와 README 기준:

1. **csim(소프트웨어 검증)**: `tapa g++ -- SpinQuant_Prefilling_tb.cpp -o SpinQuant_Prefilling -I$FLEXLLM_HOME/Modules` 후 실행(Tapa_Command:7-8).
2. **HLS/XO 합성**: `tapa compile -j 128 --top SpinQuant_Prefilling --platform xilinx_u280_gen3x16_xdma_1_202211_1 --clock-period 3.33 -f SpinQuant_Prefilling.h -c -I"$FLEXLLM_HOME/Modules" -o SpinQuant_Prefilling.xo`(Tapa_Command:9).
3. **RapidStream floorplan+impl**: `rapidstream-tapaopt -j 32 --work-dir ./RapidStream/build --tapa-xo-path *.xo --device-config u280_device.json --floorplan-config floorplan_config.json --pipeline-config pipeline_config.json --run-impl --implementation-config impl_config.json --connectivity-ini pref_link_config_u280.ini`(Tapa_Command:15-24). JSON 4종은 `RapidStream_pref_u280/*.py` 실행으로 생성.
4. **decode/mem_opt/HMT**: 동일 패턴의 `sq_dec_u280`, `sq_*_mem_opt`, `hmt_sq_pref_u280*` 타겟(Tapa_Command 각 섹션). mem_opt는 `floorplan_config_mem_opt.json` + `*_mem_opt.ini` 사용.
5. **end-to-end 호스트 데모**: `tapa g++ -- SpinQuant_Prefilling_Decoding_mem_opt_demo.cpp -I$FLEXLLM_HOME/Modules -I$LLAMA_CPP_ROOT ... libllama.so ... -o run/...demo`(README:99-104, Tapa_Command:104-113). 실행: `./...demo --bitstream_pref ...xclbin --bitstream_dec ...xclbin llama-3.2-1b-f16.gguf my_prompt.txt my_answer.txt`(README:111).
6. **link config(`*.ini`)**: `[connectivity]` 섹션에서 커널 mmap 포트 ↔ HBM/DDR 채널 바인딩(섹션 4.4).

요구사항(README:81-87): Ubuntu 20.04/22.04, XRT, **Vitis 2022.2**, TAPA CLI, U280 보드. 가중치/gguf는 별도 Google Drive 다운로드(README:65-77).

---

## 7. 의존성
- **TAPA**(task-parallel HLS frontend; `tapa.h`, `tapa::task/stream/mmap/mmaps/detach/invoke`) — config_u280.h:4, 전 커널.
- **RapidStream**(`from rapidstream import FloorplanConfig/PipelineConfig/get_u280_vitis_device_factory`) — gen_*.py, impl.py. `rapidstream-tapaopt` CLI.
- **Vitis HLS 2022.2 / Vivado**(`ap_int.h`, `ap_fixed.h`, `hls_vector.h`, `hls_math.h`, `hls_stream.h`) — config_u280.h:5-9. impl이 Vivado `Explore` 전략 사용.
- **XRT**(런타임, `xbutil examine`) — README:84,92.
- **llama.cpp**(`libllama.so`, gguf 토크나이저) — README:100-103, llama_tokenizer.h, demo cpp:19.
- 플랫폼 shell: `xilinx_u280_gen3x16_xdma_1_202211_1`.

---

## 8. 강점 / 한계 / 리스크

**강점**
- **모듈 재사용성**: PE/시스톨릭/quant/softmax/FHT를 템플릿화하여 prefill·decode·HMT·Instruct 4변형이 동일 Modules/를 공유. README 주장(~1K LOC로 2개월 내 구현, 라인 7)과 부합하는 구조.
- **DSP 패킹**(INT4 2-pack, INT8 2xDSP)로 자원 효율↑. 단일 DSP에 2곱(PE.h:373-412).
- **TAPA+RapidStream 자동화**로 수동 floorplan 부담 제거(JSON 4종 + ini만으로 SLR 배치/파이프라인).
- prefill(2D)·decode(1D) **워크로드별 시스톨릭 형상 분리**로 자원 적정화.
- attention INT8 + Linear INT4 + KV INT8의 **혼합정밀 + SpinQuant 회전(FHT)** 통합.

**한계**
- `MAX_PRE_SEQ_LEN/MAX_DEC_SEQ_LEN = 1024`로 표준 변형은 짧은 문맥. 장문맥은 HMT 필요(README:33-35는 HMT로 64× context 주장 — 코드상 `HMT_MAX_SEG_NUM 64`와 정합, 단 성능 수치 자체는 확인 불가).
- 단일 모델(1B)·단일 보드(U280) 하드코딩 다수(layer 16, hidden 2048 등 #define). V80은 추정치이며 비트스트림 미제공(README:118).
- floorplan이 `SLOT_X0Y0` 단일 슬롯 사전배치(gen_floorplan_config.py:4) — 대형 설계 확장 시 multi-SLR 수동 조정 필요 가능성(추정).
- 다수의 `*_redundant`/discard 경로, 주석 처리된 대안 invoke가 코드에 잔존(가독성/유지보수 리스크).

**리스크**
- Vitis 2022.2 / 특정 shell 버전 고정 — 툴체인 마이그레이션 시 pragma/플랫폼 비호환 가능.
- 가중치/gguf 외부 다운로드 의존(재현성). 양자화 scale은 `parameters/*_s.h`에 정적 포함(MHA_i8xi8.h:11-14) → 모델 교체 시 재생성 필요.
- 정확도(quant) 검증은 tb의 fp32 reference 비교에 의존 — 실제 perplexity 영향은 코드만으로 **확인 불가**.

---

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 관점 시사점

1. **TAPA dataflow + RapidStream floorplan 자동화 도입**: HG-PIPE류 ViT 가속기는 보통 거대한 단일 dataflow로 SLR-crossing 타이밍 클로저가 병목인데, FlexLLM은 `tapa::task().invoke` 그래프 + RapidStream 4-JSON(`device/floorplan/pipeline/impl`)으로 이를 자동화한다. 우리 ViT 파이프라인의 stage들을 `tapa::stream`으로 연결하고 RapidStream `pp_scheme` 자동 파이프라이닝(gen_pipeline_config.py)을 적용하면, XR 실시간 추론에 필요한 **고주파(300MHz+) 타이밍 클로저를 수작업 없이** 달성할 여지가 크다.

2. **INT4/INT8 DSP 패킹 PE의 ViT 이식**: `PE_i4xi4_pack_2x2_2D`(PE.h:373)의 "1 DSP = 2 저비트곱" 기법은 ViT의 patch-embedding/QKV/MLP GEMM에 직접 적용 가능. 특히 시선추적은 작은 입력 해상도·실시간성 요구가 강해 **INT4 가중치로 LUT/DSP 절감 → 더 큰 배치/더 빠른 프레임레이트**를 얻을 수 있다. `is_uint_A` 분기로 비대칭 활성화(ReLU/GELU 후 양수)도 수용.

3. **INT8 시스톨릭 MHA 커널 재사용**: ViT의 self-attention은 LLM과 동일한 QxK/softmax/AxV 구조다. `Modules/MHA.h`의 prefill 2D / decode 1D 분리 전략 중, ViT는 전 토큰 병렬(prefill형)에 해당하므로 **prefill 2D 시스톨릭 MHA를 그대로 채택**하고, causal mask 대신 full attention으로 마스크 단계만 제거하면 된다. score INT8 재양자화 경로(softmax 전후)도 ViT attention에 이식 가능.

4. **mem_opt 기법 = HBM 채널 압축**: 가중치+KV(우리 경우 가중치+중간 feature map)를 **동일 HBM 채널에 합성 포트로 묶는** 전략(dec_link_config_u280_mem_opt.ini)은, HG-PIPE처럼 layer-pipelined 설계에서 HBM 채널이 부족할 때 유용. ViT의 다단 weight buffer를 채널별로 묶어 32채널 한계 내 배치하는 데 응용.

5. **모듈 라이브러리화 방법론**: `Modules/`처럼 PE/GEMM/softmax/norm/quant를 파라미터 템플릿으로 분리하고 변형은 얇은 래퍼로 두는 구조는, 우리 ViT/시선추적 가속기를 **여러 백본(ViT-Tiny/Small, 다른 해상도)**으로 빠르게 재타깃하는 데 직접 차용할 가치가 있다.

6. **온칩 폐루프(decode)·FHT 회전 모듈**: 시선추적의 시계열/순차 후처리(예: gaze regression head의 autoregressive 요소가 있다면)에 decode형 온칩 폐루프 패턴이, 그리고 양자화 정확도 보전이 필요할 때 `Modules/FHT.h`의 Hadamard 회전 모듈이 재사용 후보(추정: ViT에 SpinQuant류 회전을 적용할 경우).

> 단, FlexLLM은 **batch=1, autoregressive LLM** 특화라 ViT의 고처리량 batched/streaming 요구와는 워크로드가 다르다. prefill 경로(2D 시스톨릭, 토큰 병렬)가 ViT에 가깝고 decode 경로는 참고도가 낮다.

---

## 10. 근거 / 한계 표기

**라인 근거로 직접 확인한 사실**
- 모델/하드웨어 상수: config_u280.h:19-75 (16층, hidden 2048, GQA 32/8, INTER 8192, U280 병렬도).
- prefill 태스크 그래프 60+ invoke: SpinQuant_Prefilling.h:807-1047.
- decode 태스크 그래프 + Top-K 샘플링 폐루프: SpinQuant_Decoding.h:946-1125.
- DSP 패킹 PE: PE.h:6-7(자원폭 주석), 338-412(i4xi4 pack).
- 시스톨릭 Linear/MHA + pragma: Linear_Layer.h:9-341, MHA.h:114-653.
- FHT(SpinQuant 회전): FHT.h:7-67.
- mem_opt 차이: config_u280_mem_opt.h:53-61, dec_link_config_u280_mem_opt.ini:8-30.
- RapidStream 4-JSON 생성: gen_device/gen_floorplan/gen_pipeline/impl.py(각 파일 전체).
- 빌드/실행: Tapa_Command(SpinQuant·HMT), README.md:81-118, demo cpp:13-19,832,932.
- HMT 추가 상수·세그먼트 제어: config_u280.h:65-72, HMT_SpinQuant_Prefilling.h:10-53.

**추정(코드 외 배경지식 기반, 단정 회피)**
- "HMT = Hierarchical Memory Transformer"(README:16 문구 + config의 segment/memory-token 상수로 강하게 시사되나 풀네임 정의 라인은 미발견).
- floorplan `SLOT_X0Y0` 단일 슬롯 = SLR0 우선 배치 의도.
- mem_opt의 합성 포트 묶음 = HBM 32채널 한계 내 배치 목적.
- Linear systolic이 output-stationary인 점(누산 PE 로컬화로 추론).

**확인 불가**
- FlexLLM 정식 학술 venue/저자(코드 외). README는 제목·DOI·AMD 감사만 명시.
- README의 성능 수치(1.29× speedup, 23.23× prefill 등) — 측정 스크립트는 run/에 있으나 결과 자체는 비트스트림·하드웨어 실행 산출물이라 정적 분석으로 검증 불가.
- 양자화 정확도(perplexity) 실제 손실치.
- `Modules/HMT.h` 내부 메모리 토큰 알고리즘의 의미론적 정확성(템플릿 시그니처/상수만 확인, 세부 알고리즘 라인별 검증은 범위 외).
