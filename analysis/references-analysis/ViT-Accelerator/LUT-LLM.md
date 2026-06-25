# LUT-LLM 코드베이스 정밀 분석

> 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Accelerator\LUT-LLM`
> 분석 도구: Glob/Grep/Read 만 사용(실제 소스 라인 근거). 생성물(*.xo/*.pdi/*.bit/*.rpt/*.pwr/*.png)은 이름만 언급.

---

## 1. 개요

- **목적**: LUT-LLM은 "**FPGA에서 1B+ 규모 LLM(Qwen)을 메모리 기반 연산(memory-based computation)으로 추론**"하는 가속기다. 핵심은 곱셈(MAC)을 대부분 제거하고, **벡터 양자화(Vector Quantization, VQ) + 룩업 테이블(LUT) + 누산(accumulation)** 으로 선형 계층(Linear/GEMM)을 대체하는 것이다.
- **한줄요약**: 활성치를 센트로이드 인덱스로 양자화(CCU) → 사전계산된 (활성센트로이드×가중치센트로이드) 부분곱을 LUT에서 인덱싱(IMM) → SIMD 정수 덧셈으로 누산 → dequant. 즉 **"곱셈 → 표 조회 + 덧셈"** 치환.
- **원논문 추정**: README L1 `# LUT-LLM: Efficient Language Model Inference with Memory-based Computations on FPGAs` (README.md:1). Zenodo DOI 뱃지 존재(README.md:3). 따라서 **LUT-LLM 논문 본체에 동봉된 artifact 저장소**로 판단(추정). 정확한 학회/연도는 README에 명시 없음(확인 불가).
- **대상 모델**: **Qwen3 1.7B**(README.md:107; `qwen_lut_model/qwen3_1.7b.json`: hidden 2048, intermediate 6144, head_dim 128, num_attention_heads 16, num_key_value_heads 8, num_hidden_layers 28, hidden_act silu, rms_norm). `config/config.h`도 동일 파라미터로 고정(HIDDEN_DIM 2048, FFN/INTERM_DIM 6144, NUM_HEADS 16, NUM_GROUPS 8, HEAD_DIM 128).
- **타깃 디바이스**:
  - 실제 빌드/비트스트림 타깃 = **AMD V80** (`xcv80-lsva4737-2MHP-e-S`). qwen_block/Makefile:14, rapidstream_script/device.py:3, README.md:25에서 part 명시. `qwen_block/qwen_v80.pdi`(비트스트림, 이름만), `timing.rpt`, `example.pwr` 동봉(README.md:106-108, 이름만).
  - lut-dla 단독 합성 타깃 = **VPK180** (`lut-dla/bitgen_vpk180.sh`: Versal vpk180 플랫폼, 300MHz).
  - 성능모델 비교 타깃 = **V80 / VPK180 / U280 / VHK158** (`qwen_lut_model/{v80,vpk180,u280,vhk158}.json`).
- **프레임워크**: HLS는 **TAPA**(task-parallel HLS, `tapa::task().invoke<join>` 데이터플로우) 기반. 빌드는 `tapa g++`(csim) / `tapa compile`(xo). RapidStream(`rapidstream-tapaopt`)로 floorplan/pipeline.

---

## 2. 디렉토리 구조

루트(`LUT-LLM/`) 기준. (소스 트리는 좁은 Glob `**/*.h|*.cpp|*.py|*.tcl|Makefile`로 확인)

| 경로 | 역할 | 핵심 파일 |
|---|---|---|
| `config/` | 전역 파라미터/데이터타입 | `config.h` (모델 차원 상수, `repeater` 보조 함수) |
| `ccu/` | **BPCSU(Bandwidth-aware Parallel Centroid Search Unit)** = 활성치 VQ | `ccu_fp32.h`(distance_pe/treeccu_fp32/ccu_fp32), `ccu_fp32_tb.cpp` |
| `imm/` | **2D 테이블 룩업 엔진** = 가중치 VQ 매처 | `imm.h`(memory_matcher_w_vq 계열, lut/weight_idx reader), `imm_tb.cpp` |
| `lut-dla/` | **LUTLinear 엔진**(CCU+IMM 통합 단일 GEMM 코어) — 핵심 | `lut_dla.h`(lut_dla_core 데이터플로우), `lut_dla_tb.cpp`(레퍼런스), `Makefile`, `bitgen_vpk180.sh` |
| `gqa/` | Grouped-Query Attention(QK^T, softmax, AV), KV cache | `gqa.h`(gemm_gqa_qk/softmax/gemm_gqa_av/kv_cache_*), `gqa_tb.cpp` |
| `rope/` | Rotary Position Embedding | `rope.h`(apply_rotary_pos_emb), `rope_tb.cpp` |
| `rms_norm/` | RMSNorm(논문에서는 LayerNorm로 모델링) | `rms_norm.h`(rms_norm/rms_norm_cache), `rms_norm_tb.cpp` |
| `silu/` | SiLU 활성(piecewise-linear 근사) | `silu.h`(silu), `silu_tb.cpp` |
| `ffn/` | FFN(SwiGLU) 단독 모듈 | `ffn.h`(splitter/combiner/element_wise_mul), `ffn_tb.cpp` |
| `attention_block/` | Attention 블록 단독(LUTLinear+RoPE+GQA) | `attention_block.h`(transpose_head/transpose_vq/apply_rope), `attention_block_tb.cpp` |
| `qwen_block/` | **전체 Transformer 블록 통합(top)** | `qwen_block.h`(qwen_block top), `qwen_block_tb.cpp`(prefill), `qwen_block_decode_tb.cpp`(decode), `e2e_latency.cpp`, `Makefile` |
| `qwen_lut_model/` | 성능/리소스 분석 모델 | `model.py`(LUTLinear/QwenModel/roofline), `{v80,vpk180,u280,vhk158}.json`, `fpga_resource_config.json`, `{qwen2.5_0.5b,qwen3_1.7b}.json`, `param_settings/setting_*.json` |
| `rapidstream_script/` | RapidStream floorplan 자동화 | `device.py`(virtual device), `floorplan.py`, `pipeline.py`, `run_rs.sh`, `*_config.json`, `v80_device.json` |
| `custom_design/` | HBM 연결 Vivado 블록디자인 | `arm_bd.tcl`, `arm_bd_wide.tcl`, `constraint.tcl`, `run.tcl` |
| 루트 | 최종 BD | `create_bd_design_final.tcl`, `README.md`(`figs/`는 png — 제외) |

**제외물(이름만)**: `qwen_block/qwen_v80.pdi`(V80 비트스트림), `qwen_block/timing.rpt`, `qwen_block/example.pwr`, `figs/*.png`(lutlinear/lutlm/gpu_lat/roofline 등 그림). model.py가 생성하는 `roofline_model.png`, `param_comparison_*.png`, `throughput_vs_seqlen_final.png`(생성물, 이름만).

---

## 3. 핵심 HLS 커널별 정밀 분석 (가장 중요)

### 3.0 데이터타입/파라미터 기반 (`config/config.h`)

모든 차원이 컴파일타임 상수로 고정(Qwen3 1.7B). 핵심:

- `MAX_SEQ_LEN=128`, `MAX_KV_LEN=256` (config.h:12-13) — prefill 시퀀스/KV cache 상한.
- `HIDDEN_DIM=2048`, `INTERM_DIM=FFN_DIM=6144`, `HEAD_DIM=128` (config.h:31,41-42,27).
- `NUM_GROUPS=8`(KV head 수), `HEAD_PER_GROUP=2`, `NUM_HEADS=16`(=8×2) (config.h:22,34,35) → **GQA 그룹당 Q head 2개**.
- `n_cent=64`(활성 센트로이드 수), `w_n_cent=16`(가중치 센트로이드 수) (config.h:20-21) → **활성=64-centroid VQ, 가중치=16-centroid VQ**(co-quantization 핵심 두 축).
- `QKV_DIM=2048+128*8*2`(config.h:37), `QK_DIM`, `V_DIM`, `KV_CACHE_DIM=128*8` (config.h:23-28).
- LUT/centroid 메모리 크기 산식: `FFN_LUT_SIZE`, `ATTN_LUT_SIZE`, `TOTAL_LUT_WEIGHT_SIZE`, `TOTAL_CENTROID_SIZE` (config.h:49-58) — 온칩 버퍼 용량 산정에 사용.
- 데이터타입: 활성=`float`(fp32) 스트림 `tapa::vec_t<float,16/2/32>`; LUT/weight_idx=`ap_uint<8>` 패킹 `vec_t<ap_uint<8>,64>`; 누산=`ap_uint<44>`/`ap_uint<48>`/`ap_uint<32>`/`ap_uint<72>` SIMD 패킹; 센트로이드=fp32. (config.h:4-5 tapa/ap_int 포함)

---

### 3.1 LUTLinear 엔진 — `lut-dla/lut_dla.h` (`lut_dla_core`) ★핵심

`lut_dla_core`(lut_dla.h:20-70)는 **CCU(활성 VQ) + IMM(가중치 VQ 2D LUT) + 누산**을 하나의 TAPA 데이터플로우로 묶은 단일 GEMM 코어다. `tapa::task()` invoke 체인(lut_dla.h:50-68):

1. `input_reader_wide`(activation), `centroid_reader_split`(센트로이드), `lut_reader`×8 / `weight_idx_reader`×8(LUT·가중치 인덱스), `scale_zero_reader`(dequant 파라미터) — off-chip → FIFO 적재.
2. `input_splitter` → `ccu_fp32`×8 : 활성치 2-element 서브벡터를 8병렬로 **최근접 센트로이드 인덱스** 산출 (lut_dla.h:55-57).
3. `memory_matcher_w_vq_head` → `memory_matcher_w_vq`×7 → `memory_matcher_tail_acc` : **16-stage(head+7중간... 실제 8개 체인) systolic 누산 파이프라인**. psum_0..psum_7 FIFO로 부분합을 다음 매처에 전달(lut_dla.h:58-66).
4. `linear_out_writer` → `measure_cycle` : 결과 dequant 후 off-chip 기록 + 사이클 측정(lut_dla.h:67-68).

**LUT 메커니즘 실체(가장 중요)** — `imm/imm.h`의 `memory_matcher_w_vq`(imm.h:229-326)와 테스트벤치 레퍼런스(lut_dla_tb.cpp:81-120)로 확정:

- 온칩 LUT: `ap_uint<8> linear_lut[n_cent=64][w_n_cent=16][...]`(imm.h:245). **인덱스 = (활성센트로이드 64, 가중치센트로이드 16)** 의 2D 테이블. 값은 **(활성센트로이드 벡터 · 가중치센트로이드 벡터)의 내적을 uint8로 양자화한 부분곱**.
- 가중치 인덱스: `ap_uint<4> weight_idx[MAX_OUT_SIZE]`(imm.h:249) — 출력열마다 어떤 16-centroid를 쓰는지(4비트, 16개)를 LUTRAM에 저장(imm.h:250-251).
- 매칭 루프(imm.h:277-323): 활성 인덱스 `idx`(=CCU 결과)로 `linear_lut[idx][k]` 한 행을 레지스터로 뽑고(`lut_reg`), 출력열별 `weight_idx`로 그 행에서 다시 골라(`linear_out_reg[k] = lut_reg[...][w_idx]`, imm.h:299-303) **곱셈 없이** 부분곱을 얻는다. 이후 `simd_b`에 12비트씩 4개 패킹 후 `simd_out = simd_a + simd_b`(imm.h:317) — **순수 SIMD 정수 덧셈 누산**.
- 즉 `lut_dla_tb.cpp:106-120` 레퍼런스가 명시: ① `act_idx = find_closest_centroid(...)`(Chebyshev), ② `weight_idx = weight_indices[...]`, ③ `dot_product = Σ act_centroids[act_idx]·weight_centroids[weight_idx]`를 누산. **곱셈은 오프라인 LUT 생성 시 1회, 런타임은 조회+덧셈만**.
- DSP 절감 핵심: 누산 덧셈만 DSP에 `#pragma HLS bind_op ... impl=dsp`(예: imm.h:712, qwen_block.h:147/282)로 매핑, 곱셈기는 GEMM(QK/AV)·비선형부에만 잔존.

**여러 매처 변종**(imm.h): `memory_matcher`(초기 fp32 LUT 누산판, imm.h:145-227, URAM), `memory_matcher_w_vq`(uint8 12bit 누산, imm.h:229), `memory_matcher_w_vq_half`(16-lane, imm.h:329), `_half_final`(4-round QKV/out/up-gate/down 통합, imm.h:423), `_half_final_v2`(8bit 누산, ap_uint<32> SIMD, imm.h:528), `_dsp`/`_half_dsp`/`_half_dsp_final`/`_half_dsp_final_v2`(누산을 DSP로 강제 매핑한 변종, imm.h:631~1032), `memory_matcher_w_vq_head`(체인 시작단, 입력 누산 없이 부분합 생성, imm.h:1035-). qwen_block에서는 `_half_final_v2`(LUTRAM 누산)와 `_half_dsp_final_v2`(DSP 누산)를 **교대로** 16단 연결(아래 3.2).

**파이프라인/병렬화**: `#pragma HLS pipeline II=1`(매칭 루프), `array_partition complete`(lut_reg/linear_out_reg 완전분할로 1024/512-way 병렬 조회, imm.h:287,299), `weight_idx` LUTRAM `cyclic factor=512~1024`(imm.h:250,349), `linear_lut` `cyclic dim=1/dim=3`(imm.h:247-248). 라운드 루프(round 0/1)로 in/out bound를 바꿔 같은 하드웨어를 재사용.

---

### 3.2 전체 블록 통합 — `qwen_block/qwen_block.h` (`qwen_block`) ★top

`qwen_block`(qwen_block.h:517-633)이 **Qwen3 한 layer 전체**를 단일 TAPA 데이터플로우로 fuse한 top 함수. 인자: KV cache(k/v) mmap, input/centroid/lut_weight_idx/scale_zero/sin/cos/rms_norm_weight/out 버퍼(qwen_block.h:518-528).

**Temporal-Spatial Hybrid Execution**(README L19): 동일 LUTLinear 하드웨어를 **4 round**로 시분할(temporal) — round0=QKV proj, round1=out proj, round2=up/gate proj, round3=down proj. 각 round 내부는 dataflow(spatial). round별 in_size/out_size를 분기(imm.h:436-438, `memory_matcher_*_final*`).

**핵심 invoke 체인(qwen_block.h:591-631)**:
- 입력: `input_reader_wide`×2, `kv_cache_readwriter`×2(k/v), `rms_weight_reader`.
- `residual_bank`(qwen_block.h:19-70): **residual 누산 버퍼**(`residual_buf[128][2048]`), 입력 저장 + 2회 linear 결과를 residual에 += 누산(out_proj/down_proj 출력).
- `rms_norm_cache`(rms_norm.h:138): RMSNorm + 3회(attention/ffn/out) 재방출.
- `lut_weight_idx_reader`×16, `scale_zero_reader_final`(TOTAL_HEADS+4개 dequant 파라미터, qwen_block.h:510-515).
- `centroid_reader_split`×2 → `treeccu_fp32`×16(활성 VQ, 4×16 트리, 아래 3.3).
- **memory_matcher 16단 systolic 체인**(qwen_block.h:605-620): `memory_matcher_w_vq_head_half_final_v2` → `_half_final_v2`(LUTRAM) ↔ `_half_dsp_final_v2`(DSP) **교대** 15회. psum_0→…→psum_15로 부분합 전파. **LUTRAM/DSP 교대 배치로 단일 자원 포화 방지**.
- `memory_matcher_acc_overlay_half_v2`(qwen_block.h:207-347): 최종 누산(URAM `ap_uint<72>` 2-way 36bit SIMD, qwen_block.h:215,279-283) + **scale/zeropoint dequant**(qwen_block.h:292-345, `val*scale - zeropoint`). round별로 QKV head/INTERM/HIDDEN 출력 형식 분기.
- Attention 경로: `distributor`(v/rope 분기) → `apply_rope` → `kv_cache_transmitter`(KV cache 기록) → `gemm_gqa_qk` → `softmax` → `gemm_gqa_av` → `attn_cache`.
- FFN 경로: `silu` → `element_wise_mul`(up·SiLU(gate), qwen_block.h:397-452, URAM).
- 출력: `linear_out_writer`×2.

**L 인스트럭션 인코딩**: `ap_uint<10> L_inst`에서 bit[8:0]=prefill 길이, bit[9]=decode 플래그(1이면 L=1). 거의 모든 커널이 `const int L=(L_inst[9]==1)?1:L_prefill`로 prefill/decode 분기(qwen_block.h:32-33, gqa.h:121-122 등). FIFO로 L을 단계 간 전파(`L_*_fifo`, qwen_block.h:584-589).

---

### 3.3 활성 VQ(BPCSU) — `ccu/ccu_fp32.h`

활성치를 센트로이드 인덱스로 양자화하는 **Bandwidth-aware Parallel Centroid Search Unit**(README L17,97).

- `distance_pe<vec_len=2>`(ccu_fp32.h:18-69): 2-element 입력 서브벡터와 한 센트로이드의 **Chebyshev(L∞) 거리**(절댓값 후 tree-max, ccu_fp32.h:42-60), 현재 최소거리/인덱스를 갱신해 다음 PE로 carry(systolic). 거리 감산은 `bind_op fsub impl=primitivedsp`(ccu_fp32.h:45).
- `ccu_fp32`(ccu_fp32.h:245-371): **64개 distance_pe를 1열 systolic chain**으로 연결(ccu_fp32.h:296-359), input/distance/index carry FIFO(SRL). `loop_fill_inp`에서 거리 초기값 3.0e20f(ccu_fp32.h:291). epilogue에서 최종 인덱스 방출.
- `treeccu_fp32`(ccu_fp32.h:374-532): **4-branch × 16-PE 트리** 변종(README "Bandwidth-aware Parallel Centroid Search"의 resource↔latency tradeoff). 4갈래 각 16 PE → `argmin_pe_l1`×2 → `argmin_pe_l2`로 reduce(ccu_fp32.h:520-522). qwen_block에서는 **treeccu_fp32×16**을 사용(qwen_block.h:604) — 병렬 검색으로 decode 파이프라인 전파 지연 단축.
- 입력 분배: `input_splitter`(16→8×2, ccu_fp32.h:131), `input_splitter_final`(4-round용, ccu_fp32.h:209), `centroid_reader_split`(ccu_fp32.h:598).

---

### 3.4 GQA — `gqa/gqa.h`

순수 floating-point GEMM(여기는 LUT 미적용, **곱셈 사용**). Q head 16 / KV head 8, 그룹당 Q 2개.

- `gemm_gqa_qk`(gqa.h:209-344): KV cache를 `k_buf[256][1024]`(BRAM)에 적재, 그룹 루프(NUM_GROUPS=8)×HEAD_PER_GROUP=2로 **QK^T MACC**(16×32 PE 레지스터 어레이, qk_reg_row, gqa.h:309-317) 후 tree-reduction(gqa.h:320-337). decode 시 과거 토큰 KV cache 로드(gqa.h:229-241).
- `softmax`(gqa.h:482-555): exp(scale=0.0883883476=1/√128, gqa.h:514) + causal mask(상삼각 0, prefill 한정, gqa.h:515) + 합의 역수 곱. dataflow.
- `gemm_gqa_av`(gqa.h:346-480): softmax×V MACC(32×16 PE, gqa.h:445-451) + reduction.
- `kv_cache_readwriter`(gqa.h:115-166): prefill=전체 기록 / decode=과거 읽기+1토큰 기록. `kv_cache_transmitter`(gqa.h:168-207): RoPE 후 K, raw V를 cache로 분기 기록.

---

### 3.5 RoPE — `rope/rope.h`

`apply_rotary_pos_emb<iter>`(rope.h:65-139): sin/cos 테이블 prefetch(rope.h:87-98), 입력에 cos/sin 곱(rope.h:115-116) 후 **half-rotate**(앞 절반 = cos−sin(뒤절반), 뒤 절반 = cos+sin(앞절반), rope.h:123-132). `apply_rope`(qwen_block.h:454-465)가 `NUM_ROPE_HEADS=NUM_HEADS+NUM_GROUPS=24`회 적용(Q 16 + K 8 head). II=1 파이프라인.

---

### 3.6 SiLU — `silu/silu.h`

`silu`(silu.h:61-125): **9-구간 piecewise-linear 근사**(silu.h:84-119, 경계 ±1/±2/±4/±8, 각 구간 slope·intercept 상수)로 `slope*x+intercept` 계산(silu.h:120). exp 미사용 → DSP/BRAM 절약. 32-lane unroll, II=1. FFN gate 경로에 사용(qwen_block.h:629). (model.py의 SwiGLU는 exp 기반으로 모델링하나, 실제 HLS는 PWL — 모델↔구현 미세 불일치, "추정")

---

### 3.7 RMSNorm — `rms_norm/rms_norm.h`

- `rms_norm`(rms_norm.h:80-136): variance(=Σx², rms_norm.h:115) → `1/√(var/D+ε)`(rms_norm.h:121, R_HIDDEN_DIM·EPSILON config.h:39-40) → `x·rstd·weight`.
- `rms_norm_cache`(rms_norm.h:138-246): qwen_block용. 가중치 prefetch + **3회 round**(r<2는 linear FIFO로, r==2는 out FIFO로) 재방출, binary reduction으로 variance 계산(rms_norm.h:190-210), URAM `input_buf[128][2048]`.

---

### 3.8 IMM 보조/리더 — `imm/imm.h`

- `index_reader`/`lut_reader`/`weight_idx_reader`/`lut_weight_idx_reader`/`scale_zero_reader`: off-chip→FIFO, async_mmap read_addr/read_data handshake, II=1(imm.h:20-119).
- `linear_out_writer`(imm.h:121-143): 결과 write_addr/write_data/write_resp handshake로 off-chip 기록.
- LUT 비트폭: `lut_bit=8`(setting_w_vq) → uint8 부분곱 LUT. 누산 SIMD 폭은 변종별로 12bit×4(ap_uint<48>)/8bit×4(ap_uint<32>)/18bit×4(ap_uint<72>).

---

### 3.9 FFN / Attention 단독 모듈

- `ffn/ffn.h`: `splitter`/`combiner`(16↔2 vec 변환, ffn.h:16-68), `element_wise_mul`(up·gate, URAM, ffn.h:70-)로 **SwiGLU의 곱셈부**만 담당. CCU/IMM은 imm.h·ccu_fp32.h include로 재사용(ffn.h:11-12). qwen_block 미사용 단위 검증용(추정).
- `attention_block/attention_block.h`: `transpose_head`/`transpose_vq`(head 단위 전치, attn:29-62/64-), `apply_rope` 등 — attention 단독 검증용. qwen_block과 별개의 standalone TB.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 LUT 기반 GEMM 메커니즘(요약)

```
[off-chip] activation(fp32) ─▶ input_reader ─▶ input_splitter ─▶ CCU(treeccu_fp32)
                                                                       │  활성 최근접 센트로이드 idx (uint8, n_cent=64)
[off-chip] act/weight centroid, 2D-LUT(uint8), weight_idx(uint4) ─▶ lut_weight_idx_reader
                                                                       ▼
   memory_matcher_w_vq:  lut_reg = linear_lut[ idx ][k]                (활성축 선택)
                         out      = lut_reg[ weight_idx[col] ]         (가중치축 선택, 곱셈 0회)
                         psum    += SIMD_add(out)                      (DSP/LUTRAM 덧셈)
                                                                       ▼
   16-stage systolic chain (psum_0→…→psum_15)  ─▶ acc_overlay(dequant: val*scale - zeropoint)
                                                                       ▼
                                            QKV / out / up-gate / down 결과(fp32)
```

곱셈은 **오프라인 LUT 생성 시 (활성센트로이드·가중치센트로이드) 내적 1회**만 발생(lut_dla_tb.cpp:117-119가 그 레퍼런스). 런타임 선형계층은 **표 조회 + 정수 덧셈**뿐 → DSP 사용을 누산기로 한정.

### 4.2 Activation-Weight Co-quantization(README "Activation-weight Co-quantization", L16)

- **활성 VQ**: 2-element 서브벡터 → 64 센트로이드 중 최근접(Chebyshev) → uint8 idx. (CCU)
- **가중치 VQ**: 가중치 열을 16 센트로이드로 → uint4 weight_idx. (config: w_n_cent=16)
- **2D LUT**: 64×16 항목 = 활성·가중치 센트로이드 쌍의 부분곱을 uint8로 사전양자화 → 단일 scalar quant 대비 표 크기 축소(README "shrinked lookup tables"). `setting_w_vq.json`: `vec_len=4, n_centroids=64, act_bit=8, lut_bit=8, weight_vq=true`.
- dequant: 최종 누산값 × scale − zeropoint(qwen_block.h:166-167,302-305; lut_dla_tb.cpp:67-70이 scale/zeropoint 산출).

### 4.3 prefill vs decode

- **prefill**: L=실제 시퀀스(32 또는 128, README L49). QK^T causal mask 적용, KV cache 전체 기록. `qwen_block_tb.cpp`.
- **decode**: L=1(L_inst[9]=1). 과거 KV cache 로드 후 1토큰만 처리, mask 미적용. `qwen_block_decode_tb.cpp`. README 표(L82-89)는 [batch, seqlen] 조합별 ms 제시(LUT-LLM이 A100 INT4/MI210 대비 긴 seq에서 우위).

### 4.4 dataflow/pipeline 정리

- TAPA `invoke<tapa::join>` = 모든 sub-task 병렬 실행, FIFO로 연결(spatial dataflow). 단계 간 `#pragma HLS pipeline II=1`.
- temporal: LUTLinear는 4-round 시분할(QKV/out/up-gate/down) — 하드웨어 1벌 재사용. attention/FFN 비선형은 별도 engine으로 round 사이 실행(README "Temporal-Spatial Hybrid", L19).

### 4.5 메모리 계층

- **off-chip(HBM, V80)**: activation/centroid/LUT/weight_idx/KV cache/scale_zero/sin/cos/out. `custom_design/arm_bd*.tcl`로 HBM 연결.
- **URAM**: residual_buf, linear_out 누산(`bind_storage impl=URAM`, qwen_block.h:83,217; ffn.h:78), rms input_buf.
- **LUTRAM**: weight_idx(`impl=LUTRAM`, imm.h:251,350) — VQ 인덱스 저장.
- **BRAM**: k_buf/v_buf(gqa.h:227,364).
- **레지스터/SRL**: lut_reg/linear_out_reg(완전분할), CCU carry FIFO(`impl=srl`, ccu_fp32.h:259).

---

## 5. HW/SW 매핑 (성능모델 ↔ HLS ↔ RapidStream ↔ 디바이스)

| 성능모델 함수(model.py) | 대응 HLS 커널 | 비고 |
|---|---|---|
| `LUTLinear`(model.py:30-104) | `lut_dla_core`/`memory_matcher_w_vq*` | dPE_dsp/dPE_lut(센트로이드 검색), psum_lut(누산 DSP), centroid/LUT 메모리 산식. `fixed=True`면 순수 SLICEL CLB LUT 구현 분기(model.py:88-99) |
| `GroupQueryQK`/`GroupQueryAV`(model.py:106-128) | `gemm_gqa_qk`/`gemm_gqa_av` | pea_x·y·z = PE 어레이(setting: 16×16×16) |
| `RoPE`(model.py:130) | `apply_rotary_pos_emb` | parallel_factor=32 |
| `LayerNorm`(model.py:141) | `rms_norm_cache` | 모델은 LayerNorm 명칭이나 구현은 RMSNorm |
| `SwiGLU`(model.py:152) | `silu`+`element_wise_mul` | 모델은 exp 기반, 구현은 PWL(불일치) |
| `QwenModel`(model.py:207-364) | `qwen_block` 전체 | fuse_op/overlay_op_list/parallel_op_list로 temporal-spatial 조합. `breakdown`로 단계별 latency |
| `check_resources`(model.py:366) | — | DSP/LUT/메모리 한도 대비 검증, spill 산출 |

- **디바이스 JSON**(qwen_lut_model): V80(lut 2,574,208 / dsp 10,848 / bram 7,334 / uram 1,878 / off_chip_bw 820 / port 32 / dsp58), VPK180(lut 3,349,120 / dsp 14,352 / bw 102.4), U280/VHK158 별도. `fpga_resource_config.json`: dsp58는 fp 곱셈 1 DSP, dsp48e2는 3 DSP — **fp32 곱셈 비용이 LUT 치환 동기**.
- **하이퍼파라미터**(setting_w_vq.json): vec_len 4, n_centroids 64, parallel_acc 8, CCU_reload_factor 14, act/lut_bit 8, weight_vq true. `setting_1.json`(Act VQ만) vs `setting_w_vq.json`(Act+Weight VQ) 비교(model.py:767).
- **RapidStream floorplan**:
  - `device.py`: V80을 3행×1열 가상 슬롯(`xcv80-...`), 슬롯별 pblock(CLOCKREGION), SLR 경계 capacity(north/south 20000) → `v80_device.json`(device.py:3-23).
  - `floorplan.py`: `FloorplanConfig` port를 SLOT_X0Y2에 고정, `memory_matcher_w_vq_half_dsp_final_int4_0`을 SLOT_X1Y0에 고정, DSE 범위 0.65~0.90 → `floorplan_config.json`(floorplan.py:3-10).
  - `pipeline.py`: `PipelineConfig(pp_scheme="double")` → SLR 횡단 더블 파이프라인(pipeline.py:3-5).
  - `run_rs.sh`: `rapidstream-tapaopt -j6 --tapa-xo-path qwen_block.xo --device-config v80_device.json --floorplan-config ... --pipeline-config ...`(run_rs.sh:3-7).
- **합성/구현**:
  - V80: `qwen_block/Makefile` `tapa compile --part xcv80-... --clock-period 3.33`(=300MHz) → qwen_block.xo (Makefile:14).
  - VPK180: `lut-dla/bitgen_vpk180.sh` v++ link, 300MHz, Explore strategy → xsa(bitgen_vpk180.sh:20-38).
  - 최종 BD: `create_bd_design_final.tcl`, `custom_design/*.tcl`(ARM+HBM).

---

## 6. 빌드 · 실행 방법 (README + Makefile 근거)

1. **TAPA 설치**: tar 해제 + PATH(README L27-37).
2. **호스트 빌드(csim)**: `cd qwen_block; make csim; make csim_decode` → `tapa g++` (Makefile:4-7). 입력 길이는 `*_tb.cpp`의 `const int L`을 32/128로 수정(README L49).
3. **C-시뮬레이션**: `./qwen_block`, `./qwen_block_decode`(README L52-55).
4. **HLS(xo 생성)**: `make hls` → `tapa compile --top qwen_block --part xcv80-... --clock-period 3.33` (Makefile:13-14, README L57-60).
5. **RTL 시뮬레이션**: `./qwen_block --bitstream=qwen_block.xo -xosim_save_waveform -xosim_work_dir=waveform/`(README L62-67). Vivado에서 ap_start/ap_done 간 latency를 4로 나눠 사이클 측정(README L72).
6. **e2e latency**: `make e2e_latency; ./e2e_latency <prefill_cycle> <decode_cycle> <input_len> <output_len>`(README L74-78, Makefile:10).
7. **RapidStream floorplan**: `rapidstream_script`에서 `python device.py / floorplan.py / pipeline.py`로 config 생성 후 `./run_rs.sh`(run_rs.sh).
8. **VPK180 비트젠**: `lut-dla/bitgen_vpk180.sh`(PLATFORM 경로 수정 필요).
9. **성능모델**: `cd qwen_lut_model; python model.py --model_config qwen3_1.7b.json --fpga_config v80.json --hyperparams param_settings/setting_w_vq.json [--roofline] [--compare_params]`(model.py:1084-1207).
- 개별 op(`rms_norm/`, `gqa/` 등)도 각 Makefile + `*_tb.cpp`로 단독 csim 가능(top 함수는 주석 처리되어 있어 일부는 수정 필요 — "추정").

---

## 7. 의존성

- **Vitis HLS / Vivado 2024.2**(README L25), 합성·구현 라이선스 part `xcv80-lsva4737-2MHP-e-S`.
- **TAPA**(rapidstream-tapa): `tapa::task/stream/mmap/vec_t`, `tapa g++`, `tapa compile`. include `<tapa.h>`(전 헤더).
- **RapidStream**(`rapidstream-tapaopt`, `from rapidstream import FloorplanConfig/PipelineConfig/DeviceFactory`). **Gurobi**(floorplan ILP, README L25).
- **AMD HLS 라이브러리**: `ap_int.h`, `ap_fixed.h`, `hls_vector.h`, `hls_math.h`, `hls_stream.h`.
- **호스트**: gflags(`DEFINE_string(bitstream...)`, lut_dla_tb.cpp:13), C++17.
- **모델 스크립트**: Python3 + numpy + matplotlib + json(model.py:1-7).

---

## 8. 강점 / 한계 / 리스크

**강점**
- **DSP 대폭 절감**: 선형계층 곱셈을 LUT 조회+덧셈으로 치환(런타임 곱셈 ≈ 0). fp32 곱셈 1 DSP(dsp58) 비용을 누산기로 한정.
- **메모리 효율 co-quantization**: 2D(64×16) LUT로 scalar quant 대비 표 축소(README L16,18). uint8 LUT + uint4 weight_idx로 온칩 LUTRAM 절약.
- **1B+ LLM 실증**: Qwen3 1.7B(28 layer) 전체 블록 통합 + V80 비트스트림(`qwen_v80.pdi`) 존재 — end-to-end 검증.
- **temporal-spatial 하이브리드**로 1벌 LUTLinear를 4 proj에 재사용 → 면적 효율.
- **자동화 파이프라인**: TAPA dataflow + RapidStream floorplan/pipeline + 성능모델(DSE) 일관.
- **LUTRAM/DSP 교대 누산**(qwen_block.h:605-620)으로 단일 자원 포화 방지.

**한계/리스크**
- **GQA(QK/AV)는 여전히 fp32 곱셈**(LUT 미적용) — attention 비중 큰 워크로드에서 DSP·latency 병목(model.py도 attention을 별도 PE로 모델).
- **정확도 영향 불명**: VQ(특히 act 64 / weight 16 센트로이드)의 perplexity/정확도 손실 수치가 저장소에 없음(레퍼런스 검증 TB만; 정확도 곡선은 figs png — 확인 불가).
- **차원 하드코딩**: config.h가 Qwen3 1.7B 전용 상수로 고정 → 다른 모델/차원 이식 시 광범위 수정 필요.
- **모델↔구현 불일치**: model.py의 SwiGLU(exp) vs 구현(PWL silu), LayerNorm 명칭 vs RMSNorm 구현 — 성능모델은 근사("추정").
- **빌드 환경 의존성 강함**: 특정 part 라이선스, Gurobi, 절대경로(bitgen의 `/scratch/...`, `/home/oswaldhe/...`) 하드코딩 → 재현성 저하.
- **decode 파이프라인 전파 지연**: CCU reload(factor 14)·체인 깊이로 decode latency 모델에 reload 패널티 반영(model.py:64,73).
- 사이클 측정이 RTL 파형 수작업(README L72) — 자동화 부재.

---

## 9. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 고처리량 ViT/Transformer + XR 시선추적)

1. **LUT 기반 곱셈 치환의 ViT 적용(DSP 절감)**:
   - ViT의 patch-embedding/QKV/MLP linear는 동일하게 **활성 VQ + 가중치 VQ + 2D-LUT 부분곱**으로 치환 가능. HG-PIPE가 추구하는 고처리량(파이프라인 stall 최소)에서 **DSP를 LUT/LUTRAM/덧셈으로 대체**하면 PE 밀도를 높여 더 깊은 stage-parallel 파이프라인을 같은 칩에 적재 가능.
   - 단, **attention(QK/AV)는 본 저장소도 fp32 곱셈 유지** → ViT의 self-attention은 LUT화 효과가 제한적. HG-PIPE의 윈도/로컬 attention이나 토큰 수가 적은 시선추적용 경량 ViT에서는 linear 비중이 커 LUT 치환 ROI가 높음(추정).
   - **co-quantization 채택 시**: act 64 / weight 16 센트로이드(`n_cent`/`w_n_cent`)를 우리 비트버짓에 맞춰 튜닝, 2D LUT(uint8)·weight_idx(uint4) 구조 그대로 차용 가능.

2. **모듈식 HLS 커널 구성**:
   - `imm.h`(매처)·`ccu_fp32.h`(VQ)·`rope/silu/rms_norm`을 **include로 조립**하고 top(qwen_block)에서 TAPA invoke 체인으로 fuse하는 패턴은 HG-PIPE의 layer-by-layer 파이프라인과 정합. 각 op에 standalone TB(`*_tb.cpp`)+top 주석 패턴을 그대로 채택해 **단위검증→통합** 워크플로 재사용.
   - `L_inst` 한 워드로 prefill/decode를 분기·전파하는 기법은 ViT의 **가변 토큰 수/해상도** 런타임 구성에 응용 가능.

3. **RapidStream floorplan 재사용**:
   - 우리도 V80/Versal 멀티-SLR 타깃이면 `device.py`(슬롯/pblock/SLR capacity) + `floorplan.py`(port 고정·핵심 셀 슬롯 고정·DSE 0.65~0.90) + `pipeline.py`(double pp) + `run_rs.sh` 템플릿을 거의 그대로 적용. **SLR 횡단 더블 파이프라인**은 HG-PIPE 고클럭(300MHz) 타이밍 클로저에 유효.
   - 핵심 GEMM/매처 셀을 특정 슬롯에 고정(`cell_pre_assignments`)하는 방식은 우리 핵심 PE 어레이 배치 가이드로 차용.

4. **성능모델(model.py) 재사용**:
   - `LUTLinear`/`GroupQueryQK,AV`/`RoPE`/`LayerNorm`/`SwiGLU` op-level 모델 + `fuse_op`/`overlay_op_list`/`parallel_op_list` 조합기 + `check_resources`(DSP/LUT/메모리 한도·spill) + roofline은 **ViT 가속기 DSE 프레임워크로 직접 이식 가능**. ViT op(conv/patchify/MHSA/MLP)로 op 함수만 교체하면 됨.
   - `fpga_resource_config.json`(dsp58 vs dsp48e2 연산당 DSP 수)·디바이스 JSON 체계는 **타깃(U280/VHK158/VPK180/V80) 교차 비교**에 그대로 활용 — 우리 XR 보드(저전력 Versal/엣지) 후보 비교에 응용.
   - **roofline + param_settings 스윕**(Act VQ vs Act+Weight VQ)으로 정확도-자원-처리량 trade-off를 사전 탐색하는 방법론을 시선추적 모델 경량화 결정에 적용.

5. **시선추적 특성 고려**: XR eye-tracking은 **저지연·소토큰** 추론이 핵심 → decode-스타일(L=1) 경로 최적화(본 저장소의 treeccu 병렬검색·CCU reload tradeoff)가 직접적으로 유익. 단, 입력이 이미지/이벤트라면 patch embedding(conv) 단계는 LUT-LLM에 없으므로 별도 설계 필요(확인 불가/우리 추가 영역).

---

## 10. 근거 / 한계 표기

- **확정(실제 코드 라인 근거)**: LUT 메커니즘(ccu_fp32.h:18-371 VQ, imm.h:229-326 2D-LUT 매칭, lut_dla_tb.cpp:81-120 레퍼런스), top dataflow(qwen_block.h:517-633), 모델 차원(config.h, qwen3_1.7b.json), 디바이스/타깃(device.py, Makefile, bitgen_vpk180.sh, *.json), 성능모델 구조(model.py), floorplan(floorplan.py/pipeline.py/device.py/run_rs.sh).
- **추정(코드 정황 기반, 단정 불가)**:
  - 원논문 학회/연도 — README에 DOI만, 본문 표기 없음.
  - LUT 부분곱이 "오프라인 곱셈 1회"라는 해석 — lut_dla_tb.cpp 레퍼런스(host 전처리)와 런타임 커널(곱셈 없음)을 종합한 추론.
  - SwiGLU/LayerNorm의 모델↔구현 불일치는 근사 모델링으로 판단.
  - ffn.h/attention_block.h가 qwen_block 비사용 "단위검증용"이라는 점.
  - 일부 단독 op top 함수가 주석 처리되어 단독 합성 시 수정 필요.
  - 우리 프로젝트 적용 ROI(특히 attention LUT화 한계, patch-embed 부재) 관련 판단.
- **확인 불가(저장소 내 부재 또는 제외물)**:
  - VQ 적용 시 정확도/perplexity 수치(figs png·논문 본문 — png 제외 대상).
  - 실측 LUT/DSP/전력 수치(timing.rpt/example.pwr/qwen_v80.pdi — 생성물 제외, 미열람).
  - U280/VHK158 실제 빌드 스크립트(성능모델 json만 존재).
- **미접근/미상세 파일(존재 확인, 본 분석서 라인 인용 일부 생략)**: imm.h 1117행 이후(memory_matcher_tail_acc/head 종료부), ffn.h 90행 이후, attention_block.h 90행 이후, qwen_lut_model/param_settings/setting_2~12.json, custom_design/*.tcl 상세, create_bd_design_final.tcl 상세, e2e_latency.cpp 본문 — 구조·역할은 파악, 전 라인 인용은 미수행.
