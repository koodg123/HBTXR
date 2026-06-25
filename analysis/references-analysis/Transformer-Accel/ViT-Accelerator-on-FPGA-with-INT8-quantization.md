# ViT-Accelerator-on-FPGA-with-INT8-quantization 정밀 분석

> 분석 대상: `REF/Transformer-Accel/ViT-Accelerator-on-FPGA-with-INT8-quantization`
> 작성일: 2026-06-20
> 근거 표기 원칙: 코드로 확인한 것은 `파일:라인` 근거를 명시. 코드에 없는 추론은 "추정", 판단 불가는 "확인 불가"로 표기.

---

## 1. 개요

이 repo는 **DeiT-Tiny급 ViT 인코더(12-layer, FEATURE_DIM=192, NUM_HEADS=3)** 를 **INT8 양자화** 한 뒤, **Vitis HLS C++** 로 작성한 단일 가속기 커널(`compute_vit`)을 합성하여 Zynq UltraScale+ MPSoC(ZCU102/ZCU104)의 PL에 올리고 PS(PYNQ)에서 구동하는 **HW/SW 혼합 ViT 추론 가속기**다.

핵심 특징(코드 근거):
- **레이어별 시분할(time-multiplexed) 단일 데이터패스**: 12개 Transformer 블록을 하드웨어로 펼치지 않고 하나의 데이터패스로 반복 실행. `compute_vit` 의 `layer_no` 루프가 매 레이어마다 가중치를 DRAM에서 재로딩 (`src/vit.cc:117-177`).
- **INT8 비대칭(asymmetric) 양자화**: weight zero-point(`*_zp`), activation zero-point(`*_zero`/`quant_zp`), per-channel/per-tensor delta(스케일) 분리. matmul 시 4-항 zero-point 전개식을 직접 계산 (`src/attention.cc:130`, `src/linear.cc:317`).
- **FlashAttention 계열 온라인 softmax**: attention score를 streaming하며 running max(bias)·running sum을 갱신하는 numerically-stable online softmax (`src/attention.cc:191-218`).
- **HLS DATAFLOW 기반 producer-consumer 파이프라인**: 각 연산(linear, attention)을 read→compute→write 스테이지로 분할하고 `hls::stream` 으로 연결 (`src/linear.cc:441-448`, `src/attention.cc:287-300`).
- **고정소수점(ap_fixed) feature map + INT8 weight 혼합 정밀도**: feature는 `ap_fixed<32,10>`, weight/activation은 `ap_ufixed<8,8>`(=INT8) (`include/datatypes.hh:8-14`).

목표 보드: 합성 타깃은 `xczu9eg-ffvb1156-2-e`(ZCU102), 100MHz 클럭 (`vitis_hls_proj/run_hls.tcl:11-12`). PYNQ 비트스트림은 ZCU104용 (`README.md:9`).

> 참고: repo 디렉토리명에는 "ViT-Accelerator"가 들어가지만 코드 변수/주석/경로가 `ViT.Acc.HW`(`testbench/vit_tb.cc:137`)로 일관되며, `REF/Transformer-Accel` 하위에 배치되어 있다. 본 문서에서는 이 repo를 "Transformer-Accel/ViT-Acc"로 칭한다.

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상)

```
ViT-Accelerator-on-FPGA-with-INT8-quantization/
├── README.md                       # 빌드 워크플로우(중국어), Vitis HLS→Vivado→Vitis 단계별 설명
├── include/                        # HLS 헤더 (모델/하드웨어 파라미터 + 함수 선언)
│   ├── dcl.hh                      # 통합 include 헤더
│   ├── model.hh                    # ViT 모델 상수 (FEATURE_DIM, NUM_LAYERS 등)
│   ├── datatypes.hh                # ap_fixed/ap_ufixed 타입 정의 (정밀도 정책)
│   ├── hardware.hh                 # AXI 폭, 블록 크기, hls::vector 타입 정의
│   ├── util.hh                     # ceildiv/roundup_p2/clamp/relu 등 constexpr 유틸
│   ├── kernel.hh                   # compute_vit 최상위 커널 시그니처
│   ├── attention.hh               # attention 커널 선언 + enum AttentionLinear
│   ├── linear.hh                   # linear(matmul) 커널 선언 + ping/pong 버퍼
│   ├── layernorm.hh               # layernorm 커널 선언 + enum LayerNorm
│   ├── gelu.hh                     # gelu 커널 선언
│   └── add.hh                      # residual add 커널 선언
├── src/                            # HLS 커널 구현
│   ├── vit.cc                      # 최상위 커널 compute_vit (전체 인코더 오케스트레이션)
│   ├── attention.cc                # Q·Kᵀ matmul + online softmax + attn·V matmul (가장 복잡)
│   ├── linear.cc                   # INT8 양자화 GEMM (가중치 로딩 + streaming matmul + dequant/GELU)
│   ├── layernorm.cc                # LayerNorm (2-pass mean/var)
│   ├── gelu.cc                     # GELU(LUT 기반 delta 근사)
│   └── add.cc                      # residual element-wise add
├── testbench/
│   └── vit_tb.cc                   # C-sim 테스트벤치 (fp32 입력/파라미터 로딩 후 compute_vit 호출)
├── tools/
│   └── convert_all_params_to_apfixed.cc  # int32/fp32 파라미터 → ap_fixed 바이너리 변환기
├── vitis_hls_proj/run_hls.tcl      # Vitis HLS 프로젝트 생성/합성/IP export 스크립트
├── vivado_proj/run_vivado.tcl      # Vivado Block Design 자동화 스크립트 (PS-PL AXI 연결)
└── pynq/
    ├── main2.0.ipynb               # PYNQ 구동 노트북
    └── vit_acc_on_zcu104.ipynb     # ZCU104 구동 노트북
```

### 2.2 제외 항목 (third-party/생성물/바이너리 — 이름만 언급, 미분석)

- `.git/` — git 메타데이터 (pack, hooks, logs).
- `pynq/vit_acc.bit` — Vivado 생성 **비트스트림** (바이너리 산출물).
- `pynq/vit_acc.hwh` — 하드웨어 핸드오프 파일 (생성물).
- `pynq/params/*.apfixed.bin` (30+개) — `tools/convert_all_params_to_apfixed.cc` 가 생성한 **양자화 파라미터 바이너리** (산출물).
- `doc/step1.png ~ step15.png` — README 첨부 빌드 스크린샷 (이미지).
- `data/.gitkeep`, `params/.gitkeep` — 빈 디렉토리 placeholder (데이터셋/가중치 미업로드, `README.md:3`).
- `testbench.ipynb` — 루트의 노트북 (확인 불가: 미열람, 보조 검증용으로 추정).

---

## 3. 핵심 모듈 정밀 분석

### 3.0 모델·하드웨어 파라미터 (설계의 전제)

`include/model.hh`:
- `INPUT 3×224×224`, `PATCH 16×16` → 패치 수 14×14=196, +CLS = **NUM_PATCHES=197** (`model.hh:12`).
- `FEATURE_DIM=192`, `HIDDEN_DIM=4×192=768`(`model.hh:9,13`), `NUM_HEADS=3`, `NUM_LAYERS=12`(`model.hh:15-16`).
- DIM_PER_HEAD = 192/3 = 64 (`src/attention.cc:7`).

`include/hardware.hh` (데이터 패킹 정책 — 성능의 핵심):
- `AXI_XFER_BIT_WIDTH=256` (`hardware.hh:11`).
- `FEATURE_BLOCK_SIZE = 256 / 32 = 8` → feature 8개를 한 AXI 비트 트랜잭션으로 묶음 (`hardware.hh:12`). 즉 `fm_t`(32bit) × 8 = 256bit.
- `NUM_FEATURE_BLOCKS = ceil(192/8) = 24` (`hardware.hh:13`).
- `LINEAR_IN_SIZE = LINEAR_OUT_SIZE = 16`(=2×8) → linear의 입력/출력 병렬도 (`hardware.hh:15-16`).
- `ATTN_MATMUL_PARALLEL = 4` → attention에서 동시에 처리하는 query patch 수 (`hardware.hh:18`).
- `roundup_p2(NUM_HEADS)=roundup_p2(3)=4` → head를 2의 거듭제곱(4)로 패딩한 `heads_t` 벡터 (`hardware.hh:23`, `util.hh:18-21`).

`include/datatypes.hh` (정밀도 정책):
- `fm_t = ap_fixed<32,10>` — feature map (정수부 10bit, 소수부 22bit) (`datatypes.hh:8`).
- `fm_half_t = ap_fixed<16,5>` — LayerNorm weight/bias (절반 폭) (`datatypes.hh:10`).
- `quant8_t = ap_ufixed<8,8>` — **INT8 (0~255, 비대칭 unsigned)** (`datatypes.hh:13`).
- `quant32_t = ap_fixed<32,32>` — INT32 누산기 (matmul 부분합·bias) (`datatypes.hh:14`).
- `quant_delta1_t = ap_ufixed<16,2>`, `quant_delta2_t = ap_ufixed<16,7>` — 두 종류의 양자화 스케일(delta) (`datatypes.hh:11-12`).

> 평가: INT8 weight + INT32 누산 + ap_fixed feature의 혼합 정밀도(mixed-precision)는 HG-PIPE류 고처리량 ViT 가속기 표준 구성과 일치. `quant8_t`를 unsigned(`ap_ufixed`)로 두고 zero-point로 부호를 흡수하는 점이 특징 (asymmetric quantization, `clamp()`가 0~255로 saturate, `util.hh:35-38`).

---

### 3.1 `compute_vit` — 최상위 오케스트레이터 (`src/vit.cc:8-181`)

전체 12-layer 인코더의 데이터플로우를 조립하는 단일 HLS top 함수. `extern "C"` 로 export 되어 IP 핵심이 된다 (`include/kernel.hh:12-67`).

**인터페이스 (`src/vit.cc:60-107`)**:
- 제어: `s_axilite port=return` (`vit.cc:60`).
- 5개의 `m_axi` 번들: `inout1~inout4` (중간버퍼/입출력)과 `weights` (전체 양자화 파라미터). `max_widen_bitwidth=256` 으로 AXI 폭 자동 확장 (`vit.cc:62-107`).
- Vivado 스크립트에서 이 5개 마스터 포트가 ZCU102의 5개 HP/HPC 슬레이브에 연결됨: `m_axi_inout1→S_AXI_HP0`, `…inout4→S_AXI_HP3`, `m_axi_weights→S_AXI_HPC0` (`vivado_proj/run_vivado.tcl:19-23`, `README.md:35`).

**리소스 공유 (`src/vit.cc:109-110`)**:
```
#pragma HLS ALLOCATION function instances=quant_compute_linear_one_layer limit=1
#pragma HLS ALLOCATION function instances=quant_compute_linear limit=1
```
→ linear 엔진을 **각 1개만 인스턴스화**하여 면적을 절감하고 모든 레이어가 재사용. 이것이 "레이어 시분할 단일 데이터패스" 설계의 직접 증거.

**레이어 루프 본문 (`src/vit.cc:117-177`) — 한 Transformer 블록의 12단계**:
1. `load_norms` — 해당 레이어의 norm1/norm2 weight·bias를 on-chip으로 적재 (`vit.cc:122`).
2. `compute_norm1(x → tmp1)` — pre-LN (`vit.cc:124`).
3. **Q projection**: ping 버퍼에 Q weight/bias/zp 로딩 → `quant_compute_linear(tmp1 → tmp3)` (`vit.cc:126-129`).
4. **K projection**: pong 버퍼에 K 로딩 → `quant_compute_linear(tmp1 → tmp4)` (`vit.cc:131-134`).
5. **Q·Kᵀ + softmax**: `quant_compute_q_matmul_k(tmp3=Q, tmp4=K → attn, attn_softmax_info)` (`vit.cc:136-140`).
6. **V projection**: ping에 V 로딩 → `quant_compute_linear(tmp1 → tmp3)` (`vit.cc:142-145`).
7. **attn·V**: `quant_compute_attn_matmul_v(tmp3=V, attn, softmax_info → tmp1)` (`vit.cc:148-150`).
8. **output projection (proj)**: pong에 PROJ 로딩 + dequant delta 로딩 → `quant_compute_linear_one_layer(tmp1 → tmp2, use_gelu=false)` (`vit.cc:152-156`).
9. **residual #1**: `compute_add(x, tmp2 → x)` (`vit.cc:158`).
10. `compute_norm2(x → tmp1)` (`vit.cc:160`).
11. **MLP fc1 (+GELU)**: ping에 fc1 로딩 → `quant_compute_linear_one_layer(tmp1 → tmp_hidden, use_gelu=true)` HIDDEN_DIM=768 (`vit.cc:162-166`).
12. **MLP fc2**: pong에 fc2 로딩 → `quant_compute_linear_one_layer(tmp_hidden → tmp2, use_gelu=false)` (`vit.cc:169-173`).
13. **residual #2**: `compute_add(x, tmp2 → x)` (`vit.cc:175`).

> ping/pong 버퍼(`linear_*_ping`/`_pong`, `src/linear.cc:3-11`)는 단순히 두 개의 가중치 버퍼를 번갈아 쓰는 것으로, Q/K, V/PROJ, fc1/fc2 쌍이 서로 다른 버퍼를 사용. 다만 `quant_compute_linear` 인스턴스는 1개로 제한되므로(`vit.cc:110`) 이는 **연산 중첩이 아닌 버퍼 분리**로 보는 것이 정확하다(추정: 코드 가독성/leftover 가중치 분리 목적).

**조기 종료**: `last_layer_no` 로 부분 실행 가능 (`vit.cc:119`, tb에서 12로 설정 = 전체 실행, `vit_tb.cc:9`). 디버깅/레이어별 검증용으로 추정.

> 주의(잠재 이슈): `image_id` 루프(`vit.cc:112`)는 `x[image_id]`를 in-place로 갱신하지만, `last_layer_no` 도달 시 `return` 으로 함수 전체를 종료하므로 `num_images>1` 과 `last_layer_no<12` 를 함께 쓰면 두 번째 이미지는 처리되지 않는다. tb는 단일 이미지·full layer라 문제 없음 (확인: `vit_tb.cc:8-9`).

---

### 3.2 `linear.cc` — INT8 양자화 GEMM 엔진 (`src/linear.cc`, 가장 재사용도 높음)

ViT의 모든 fully-connected 연산(Q/K/V/proj, MLP fc1/fc2)을 담당. DATAFLOW로 3-스테이지 파이프라인.

#### (a) 가중치 로딩 함수군 — DRAM 레이아웃 → on-chip 블록 재배치
- `quant_load_linear_weights` (`linear.cc:14-89`): DRAM의 행우선 INT8 weight를 16개(WEIGHT_BLOCK_SIZE) 단위로 읽어, `[LINEAR_OUT_SIZE][...][...]` 형태의 on-chip 캐시로 transpose/타일링. `array_partition complete dim=1,2` 로 병렬 접근 (`linear.cc:36-37`). 복잡한 next-index 상태머신으로 II=1 파이프라인 유지 (`linear.cc:39-87`).
- `quant_load_linear_bias` (`linear.cc:92-134`): INT32 bias를 8개 단위로 블록화.
- `quant_load_linear_weights_zp` (`linear.cc:137-179`): weight zero-point(INT8)를 16개 단위로 블록화.
- `dequant_load_delta` (`linear.cc:182-224`): per-channel dequant scale(`fm_t`)을 8개 단위로 블록화.

> 이 4개 로더는 거의 동일한 "src 블록 → dst 블록 재패킹" 패턴의 복붙. 블록 크기만 다름(16/8/16/8). 코드 중복이 크다(약점).

#### (b) Streaming matmul 코어
- `quant_read_in_stream` (`linear.cc:227-262`): 입력 feature(`fm_block_t`, 8개)를 16개(LINEAR_IN_SIZE) 단위로 모아 **그 자리에서 INT8로 재양자화** 후 stream에 push. 핵심 식: `clamp(blocks[..] * quant_delta + quant_zp)` (`linear.cc:256`). 즉 activation을 fixed-point → INT8 로 동적 양자화.
- `quant_compute_linear_on_stream` (`linear.cc:265-327`) **— 진짜 MAC 코어**:
  - bias로 누산기 초기화 (`linear.cc:307-312`).
  - 16×16 타일 MAC: `addend[out] = in_block[in] * (weights[..][out][in] - weights_zp[..][out])` (`linear.cc:317`). weight zero-point를 weight에서 직접 빼고, activation zero-point(`quant_zp`)는 read 스테이지에서 이미 더해진 형태로 흡수.
  - `out_block += addend` 누산 후 in_dim 끝에서 stream out (`linear.cc:320-325`).
  - `#pragma HLS aggregate` 로 weight/bias/zp 벡터를 단일 워드로 묶어 BRAM 접근 최적화 (`linear.cc:275-277`).
- `dequant_write_out_stream` (`linear.cc:329-381`): INT32 결과에 per-channel delta 곱(`stream_block[j]*delta_block_cache[j]`, `linear.cc:373`) → `fm_t` 복원. `use_gelu` 면 `gelu_block` 적용 (`linear.cc:375-376`).
- `direct_write_out_stream` (`linear.cc:383-425`): dequant 없이 INT32 그대로 출력 (Q/K/V projection 결과는 attention matmul에서 정수 상태로 다시 쓰이므로).

#### (c) Top wrapper
- `quant_compute_linear_one_layer` (`linear.cc:427-448`): read→compute→**dequant_write** 3-스테이지 DATAFLOW (proj/MLP용, fm_t 출력).
- `quant_compute_linear` (`linear.cc:450-469`): read→compute→**direct_write** (Q/K/V용, INT32 출력).

> 핵심 근거: `linear.cc:317`(zero-point 포함 MAC), `linear.cc:256/373`(quant/dequant 경계), `linear.cc:441-447`(DATAFLOW 파이프라인).

---

### 3.3 `attention.cc` — Q·Kᵀ + Online Softmax + attn·V (`src/attention.cc`, 가장 복잡)

두 개의 DATAFLOW 함수로 구성: `quant_compute_q_matmul_k` 와 `quant_compute_attn_matmul_v`. `ATTN_MATMUL_PARALLEL=4` 개 query patch를 한 번에 처리.

#### (a) `quant_compute_q_matmul_k` (`attention.cc:272-300`) — 6-스테이지 DATAFLOW
파이프라인: `read_q → read_kv → compute_q_matmul_k_inner → finalize_attn → write_attn / write_attn_softmax_info`.

- `read_q` (`attention.cc:35-54`): INT32 Q를 per-channel delta·zero-point로 다시 INT8 양자화 (`clamp(q*delta + zero)`, `attention.cc:48`) 후 stream.
- `read_kv` (`attention.cc:56-81`): K(또는 V) 양자화 + stream. q_patch를 4개씩 타일링하며 K 전체를 반복 공급. overflow 가드(`attention.cc:71`)로 패딩된 마지막 타일 처리.
- `compute_q_matmul_k_inner` (`attention.cc:83-148`) **— QKᵀ MAC 코어**:
  - 4-항 zero-point 전개로 INT8 내적: `q*k - q_zero*k - q*k_zero + q_zero*k_zero` (`attention.cc:130`).
  - `head = dim_block / (DIM_PER_HEAD/FEATURE_BLOCK_SIZE)` 로 dim block을 head에 매핑 (`attention.cc:124`). head별로 누산(`attn_blocks[q_patch][head]`).
  - `DIM_PER_HEAD % FEATURE_BLOCK_SIZE == 0` static_assert (64%8=0, `attention.cc:123`).
- `finalize_attn` (`attention.cc:151-230`) **— FlashAttention 온라인 softmax**:
  - 정수 attention score를 `*dequant_delta*attn_scale` 로 dequant (attn_scale=0.125=1/√64, `attention.cc:9,186`).
  - running max(bias)·running sum 갱신: 새 max가 들어오면 기존 sum을 `exp(old_bias-new)` 로 rescale (`attention.cc:201-211`). `softmax_sums`/`softmax_biases` 를 4×NUM_HEADS로 partition하여 병렬 갱신 (`attention.cc:158-163`).
  - 출력: `softmax_info_row[head*2]=1/sum`, `[head*2+1]=bias` 로 sum의 역수와 max를 함께 저장 (`attention.cc:216-217`).
  - 동시에 dequant된 raw attention score(`attn_blocks`)는 그대로 `attn_stream` 으로 흘려보냄(softmax 미적용 상태, `attention.cc:226`).
- `write_attn`/`write_attn_softmax_info` (`attention.cc:232-269`): 결과를 DRAM(`attn`, `attn_softmax_info`)에 저장.

> 설계 포인트: QKᵀ 단계에서는 softmax 정규화를 적용하지 않고 **(max, 1/sum)만 계산해 저장**한다. 실제 `exp((score-max))/sum` 정규화와 V 가중합은 다음 함수에서 수행 → softmax를 두 패스(통계 수집 / 적용)로 분리한 streaming softmax.

#### (b) `quant_compute_attn_matmul_v` (`attention.cc:466-489`) — 6-스테이지 DATAFLOW
파이프라인: `read_kv(V) → read_attn → read_attn_softmax_info → prepare_attn → compute_attn_matmul_v_inner → write_attn_matmul_v`.

- `read_attn`/`read_attn_softmax_info` (`attention.cc:302-337`): 앞서 저장한 raw attention과 softmax 통계를 다시 stream으로 로드.
- `prepare_attn` (`attention.cc:340-369`): 4개 patch를 `attn_parallel_t` 로 재패킹.
- `compute_attn_matmul_v_inner` (`attention.cc:371-446`) **— softmax 적용 + V 가중합**:
  - dim_block==0 시점에 softmax 적용: `attn = exp(score - bias) * (1/sum)` (`attention.cc:411-414`).
  - V matmul (zero-point 보정 포함): `acc = v*attn - v_zero*attn` (`attention.cc:428`).
  - head 매핑은 QKᵀ와 동일(`attention.cc:422`).
  - 4-patch 순환 누산 후 V patch 끝에서 출력 (`attention.cc:437-441`).
- `write_attn_matmul_v` (`attention.cc:449-463`): 결과를 `attn_matmul_v`(=tmp1) 로 저장 → proj projection 입력.

> 근거: `attention.cc:130`(QKᵀ 4-항 정수 내적), `attention.cc:201-211`(online max-rescale), `attention.cc:411-414`(softmax 적용), `attention.cc:428`(attn·V).
> 복잡성 원인: q_patch를 4-stride로 타일링하면서 패딩 타일(NUM_PATCHES=197이 4의 배수가 아님)을 overflow 가드로 처리. `q_patch_limit = 197+3 = 200`, 200/4=50 타일 (`attention.cc:98,237-240`). 이 인덱싱이 코드를 가장 어렵게 만든다.

---

### 3.4 `layernorm.cc` — LayerNorm (`src/layernorm.cc`)

- 전역 on-chip 버퍼 `norm1/2_weights/bias[FEATURE_DIM]` (`layernorm.cc:3-6`). `norm_eps=1e-6` (`layernorm.cc:8`).
- `load_norms` (`layernorm.cc:10-80`): DRAM의 norm weight/bias(NORM_1/NORM_2)를 8개 단위 블록으로 on-chip 적재. 4개의 거의 동일한 블록(중복).
- `layernorm_accumulate` (`layernorm.cc:83-110`): **1-pass로 mean과 mean_sq 동시 계산**. `partial_mean += x/D`, `partial_mean_sq += x*(x/D)` (`layernorm.cc:100-102`). 입력을 `x_patch`에 캐시(`:106`).
- `layernorm_output` (`layernorm.cc:112-146`): `var = mean_sq - mean² + eps`, `rstddev = 1/√var` (`layernorm.cc:124-126`), 정규화 후 affine(`(x-mean)*rstddev*w + b`, `layernorm.cc:137-140`). `array_reshape cyclic factor=8` 로 weight/bias 병렬 접근 (`layernorm.cc:121-122`).
- `compute_norm` (`layernorm.cc:149-165`): patch별 DATAFLOW(accumulate→output). `compute_norm1/2` 는 inline wrapper (`layernorm.cc:168-176`).

> 평가: mean과 mean-of-square를 한 패스로 모아 variance를 `E[x²]-E[x]²` 로 계산하는 single-pass LayerNorm. 수치적으로는 catastrophic cancellation 위험이 있으나 ap_fixed<32,10> 정밀도로 흡수하는 설계(추정). `1/D`를 미리 곱해 partial sum의 오버플로를 줄임.

### 3.5 `gelu.cc` — LUT 기반 GELU (`src/gelu.cc`)

- `gelu(x) ≈ relu(x) - delta(x)` 근사. `delta(x)`는 우함수이고 0≤delta<1 (`gelu.cc:190-192`).
- `GELU_DELTA_TABLE[177]` (`gelu.cc:3-181`): step=0.03125, 범위 0~5.5의 delta LUT. `ROM_NP` 로 바인딩 (`gelu.cc:186-187,196`).
- `gelu()` (`gelu.cc:194-210`): `|x|≥5.5` 면 relu 반환, 아니면 LUT 인덱싱 + **선형 보간**(`a + t*(b-a)`, `gelu.cc:208`) 후 `relu - delta`.
- `gelu_block()` (`gelu.cc:212-219`): 8-element 블록에 UNROLL 적용.

> 평가: GELU를 "ReLU에서 빼는 보정값(delta)"으로 재정의하여 LUT 범위를 [0,1)로 압축 — 정밀도 효율적인 트릭. MLP fc1 출력에만 적용 (`linear.cc:375`).

### 3.6 `add.cc` — Residual Add (`src/add.cc`)

- `compute_add` (`add.cc:3-16`): `out = x + y` element-wise(블록 단위), `dependence inter false` 로 II=1 (`add.cc:11-14`). residual connection 2회/레이어.

---

### 3.7 보조 도구 (HW 아님)

- `tools/convert_all_params_to_apfixed.cc` (`tools/...:1-1325`): PyTorch에서 export된 int32/fp32 `.bin` 파라미터(레이어×30+종)를 읽어 ap_fixed 바이너리(`*.apfixed.bin`)로 변환. 경로 `/home/jack/Desktop/ViT.Acc.HW/...` 하드코딩 (`:137,152` 등). weight 0~255 범위 검사 포함 (`:159`).
- `testbench/vit_tb.cc` (`testbench/...:1-1011`): 동일 파라미터 로더 + `compute_vit` 호출 + 출력 `vit.out.fp32.bin` 저장. C-sim/cosim 검증용 (`:939-1002`). 경로 하드코딩 동일.

---

## 4. 데이터 플로우

```
[PyTorch 양자화 모델]
   │ export int32/fp32 .bin
   ▼
tools/convert_all_params_to_apfixed.cc  →  *.apfixed.bin (pynq/params)
   │
   ▼ (PS: PYNQ가 DRAM에 적재)
┌──────────────────────── compute_vit (PL, single datapath) ────────────────────────┐
│ for layer in 0..11:                                                                │
│   load_norms ──► compute_norm1(x→tmp1)                                             │
│   ┌─ Q: load wt → quant_compute_linear(tmp1→tmp3, INT32)                           │
│   ├─ K: load wt → quant_compute_linear(tmp1→tmp4, INT32)                           │
│   ├─ QKᵀ+softmax통계: quant_compute_q_matmul_k(tmp3,tmp4→attn,softmax_info)        │
│   ├─ V: load wt → quant_compute_linear(tmp1→tmp3, INT32)                           │
│   ├─ attn·V: quant_compute_attn_matmul_v(tmp3,attn,info→tmp1)                      │
│   └─ proj: quant_compute_linear_one_layer(tmp1→tmp2, fm_t)                         │
│   x += tmp2  (residual #1)                                                          │
│   compute_norm2(x→tmp1)                                                            │
│   fc1+GELU: quant_compute_linear_one_layer(tmp1→tmp_hidden 768, gelu)             │
│   fc2:      quant_compute_linear_one_layer(tmp_hidden→tmp2)                        │
│   x += tmp2  (residual #2)                                                          │
└────────────────────────────────────────────────────────────────────────────────┘
   ▼
x[image] (DRAM) ──► PS가 classification head 처리 (PL 외부, 추정)
```

각 연산 내부 데이터플로우(예: linear): `DRAM weight → 로더(블록 재배치) → on-chip BRAM | DRAM feature → read(재양자화)+stream → MAC(16×16 타일)+stream → dequant/GELU+stream → DRAM`.

정밀도 경계:
- feature는 `fm_t`(ap_fixed<32,10>)로 DRAM 왕복.
- linear/attention 진입 시 INT8(`quant8_t`)로 동적 재양자화 → INT32(`quant32_t`) 누산 → delta 곱으로 fm_t 복원.
- Q/K/V projection 결과는 INT32 상태로 DRAM 저장(`patch_quant32_blocks_t tmp3/tmp4`)되어 attention에 재공급.

---

## 5. HW/SW 매핑

| 계층 | 구성요소 | 근거 |
|------|----------|------|
| **PL (FPGA, HLS 합성)** | `compute_vit` 단일 IP — 모든 Transformer 연산 | `kernel.hh:12-67`, `run_hls.tcl:2` set_top |
| PL 내부 엔진 | linear GEMM ×1, attention QKᵀ/AV, layernorm, gelu, add | `vit.cc:109-110` ALLOCATION limit=1 |
| PL↔DRAM | 5개 AXI master(inout1~4, weights), 256bit | `vit.cc:62-107` |
| **PS (ARM, PYNQ)** | 파라미터 DRAM 적재, 커널 기동, classification head | `pynq/*.ipynb` (추정: 노트북 미열람) |
| PS↔PL 제어 | AXI-Lite (`s_axilite`) | `vit.cc:60` |
| PS-PL 연결 | HP0~3→inout1~4, HPC0→weights | `run_vivado.tcl:18-23`, `README.md:35` |
| **오프라인 SW** | PyTorch→int8 양자화, ap_fixed 변환기, 테스트벤치 | `tools/convert...cc`, `testbench/vit_tb.cc` |
| 보드 | 합성 xczu9eg(ZCU102) @100MHz / 배포 ZCU104 | `run_hls.tcl:11-12`, `README.md:9` |

> PS측 전처리(patch embedding)·후처리(head)는 repo에 코드 없음. `x[]` 입력이 이미 패치 임베딩된 `[197][192]` feature이므로(`vit_tb.cc:137` `vit.x.fp32.bin`), **patch embedding과 최종 분류 head는 PS/오프라인에서 처리**되는 것으로 추정(확인 불가).

---

## 6. 빌드·실행

`README.md`(중국어) + tcl 스크립트 기준 워크플로우:

1. **파라미터 변환(SW)**: `convert_all_params_to_apfixed.cc` 컴파일·실행 → `*.apfixed.bin` 생성 (`README.md:5`).
2. **Vitis HLS(PL 합성)**: `run_hls.tcl` 실행 (`README.md:13`).
   - `set_top compute_vit`, src 6개 + tb 추가 (`run_hls.tcl:2-9`).
   - part `xczu9eg-ffvb1156-2-e`, clock 10ns(100MHz) (`run_hls.tcl:11-12`).
   - `csim → csynth → cosim → export_design(ip_catalog)` (`run_hls.tcl:14-17`).
   - tb cflags에 `-mbig-obj`(거대 오브젝트), `-Wno-unknown-pragmas` (`run_hls.tcl:9`).
3. **Vivado(시스템 통합)**: `run_vivado.tcl` (`README.md:16`).
   - ZCU102 board preset, HLS IP import, Zynq UltraScale+ PS 추가 (`run_vivado.tcl:1-16`).
   - 5개 AXI master를 HP/HPC 슬레이브에 자동 연결, wrapper 생성, bitstream write, XSA export (`run_vivado.tcl:18-28`).
4. **Vitis(앱) / PYNQ**: XSA로 플랫폼 생성, 앱 빌드(xilffs/lfn, stack/heap 설정) 또는 PYNQ 노트북(`pynq/*.ipynb`)으로 `.bit`+`.hwh` 로드 후 구동 (`README.md:51-71`).

> 주의: README는 ZCU102(tcl)와 ZCU104(pynq) 보드 불일치를 명시 (`README.md:9`). tb/converter는 절대경로 하드코딩이라 재현 시 수정 필요.

---

## 7. 의존성

- **Vitis HLS 라이브러리** (Xilinx):
  - `ap_fixed.h` (고정소수점, `datatypes.hh:6`).
  - `hls_vector.h` (SIMD 벡터 패킹, `hardware.hh:8`).
  - `hls_stream.h` (DATAFLOW FIFO, `linear.hh:11`).
  - `hls_math.h` (`exp`, `recip`, `sqrt`, `signbit`, `sqrt`, `gelu.cc`/`attention.cc`/`layernorm.cc`).
  - `ap_uint`/`ap_int` (간접, gelu 인덱스 `gelu.cc:184`).
- **표준 C++**: `<fstream>`, `<iostream>`, `<sstream>` (tb/tools only).
- **외부 ML 프레임워크**: PyTorch (양자화·export, repo 외부, 파일명 규약 `block_*_attn_qkv_*.int32.bin` 으로 추정).
- **하드웨어 플랫폼**: Zynq UltraScale+ MPSoC, Vivado, PYNQ.
- third-party 소스 vendoring 없음 (전부 자체 작성 + Xilinx 헤더).

---

## 8. 강점·한계

### 강점
1. **완성도 높은 INT8 양자화 데이터패스**: asymmetric quant(weight/activation zero-point 분리), per-channel delta, 4-항 zero-point 정수 내적을 모두 정확히 구현 (`attention.cc:130`, `linear.cc:317`).
2. **FlashAttention 스타일 streaming softmax**: running max·sum으로 전체 score 행을 메모리에 담지 않고 numerically-stable softmax 처리 (`attention.cc:201-218`). 긴 시퀀스(197 patch)에 메모리 효율적.
3. **철저한 DATAFLOW/PIPELINE 적용**: 거의 모든 루프에 `#pragma HLS PIPELINE`, 연산 단위로 producer-consumer 분할 → II=1 지향.
4. **데이터 패킹 정교함**: 256bit AXI에 맞춘 8-feature 블록, hls::vector aggregate, array_partition으로 메모리 대역폭 활용 극대화.
5. **면적 절약형 단일 데이터패스**: linear 엔진 1개로 12레이어×6 linear를 시분할 (`vit.cc:109-110`).
6. **효율적 GELU LUT**: relu-delta 분해로 LUT 범위 압축 (`gelu.cc:190-192`).

### 한계
1. **레이어마다 전체 가중치를 DRAM에서 재로딩**: 가중치가 on-chip에 상주하지 않아 매 레이어 DRAM 트래픽이 크다 (`vit.cc:126-173`). 처리량 병목 가능성(추정).
2. **대규모 코드 중복**: 로더 함수 4종(linear) + 4종(layernorm)이 거의 동일. 유지보수성 저하.
3. **단일 linear 인스턴스(limit=1)**: 연산 병렬도가 낮음. Q/K/V를 순차 계산하므로 attention latency가 길다(`vit.cc:110`).
4. **하드코딩 절대경로**: tb/converter가 `/home/jack/Desktop/...` 고정 → 재현성 낮음.
5. **patch embedding/head 부재**: 전체 end-to-end가 아닌 인코더 블록만 가속. 입출력 정의가 PS/오프라인에 의존(확인 불가).
6. **문서 부족**: README는 빌드 단계만 다루고 아키텍처/성능 수치 없음. 합성 결과(LUT/DSP/BRAM/latency) 미공개.
7. **num_images>1 + 부분레이어 동시 사용 시 버그 소지** (`vit.cc:119` early return, 3.1 참고).
8. **단일 배치/단일 head 폭(NUM_HEADS=3→4 패딩)**: head를 2의 거듭제곱으로 패딩(`hardware.hh:23`)하여 25% 연산 낭비(head 4개 중 1개는 미사용, 추정).

---

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 시사점

> 우리 프로젝트 추정 전제: HG-PIPE류 **레이어 파이프라인 펼침(fully-pipelined) 고처리량** ViT 가속기 + XR 시선추적(eye-tracking) 저지연 요구.

### 직접 차용 가능한 자산
1. **INT8 비대칭 양자화 MAC 패턴** (`linear.cc:317`, `attention.cc:130`): 4-항 zero-point 전개식은 그대로 재사용 가능. weight zero-point를 weight에서 미리 빼고 activation zero-point를 read 단계에서 흡수하는 분리 전략이 깔끔.
2. **Streaming/FlashAttention softmax** (`attention.cc:151-230`): XR 시선추적처럼 토큰 수가 가변·다수일 때 메모리 footprint를 줄이는 핵심 기법. running max-rescale 로직(`attention.cc:201-211`)을 RTL/HLS로 이식 가능.
3. **GELU relu-delta LUT** (`gelu.cc`): 작은 LUT+선형보간으로 GELU를 구현하는 면적 효율적 방법. 우리 GELU/활성함수 블록에 적용 가능.
4. **single-pass LayerNorm** (`layernorm.cc:83-110`): `E[x²]-E[x]²` 한 패스 통계로 LN latency 단축.
5. **256bit AXI 블록 패킹 정책** (`hardware.hh:11-37`): feature를 AXI 폭에 맞춰 hls::vector로 묶는 정석. 우리 DMA/대역폭 설계 참고.

### 우리 설계에서 개선/차별화할 점 (HG-PIPE 방향)
1. **레이어 파이프라인 펼침 vs 시분할**: 이 repo는 단일 데이터패스 시분할(면적↓, 처리량↓). HG-PIPE류 고처리량 목표라면 **레이어를 펼쳐(unroll) on-chip 가중치 상주 + 레이어 간 파이프라인** 으로 가야 한다. 이 repo의 "매 레이어 DRAM 재로딩"(`vit.cc:126-173`)은 정확히 우리가 피해야 할 안티패턴.
2. **연산 병렬도 상향**: `ALLOCATION limit=1`(`vit.cc:110`)을 제거하고 Q/K/V·multi-head를 공간 병렬화. NUM_HEADS 패딩 낭비(`hardware.hh:23`)도 제거 대상.
3. **XR 저지연**: 이 repo는 throughput보다 면적 최적화. XR 시선추적은 **per-frame latency**가 핵심이므로, 레이어 펼침 + 더블버퍼링으로 frame-to-frame 파이프라인을 채워야 함. 이 repo의 ping/pong은 진정한 더블버퍼링이 아님(3.1 참고) → 우리는 실제 가중치 prefetch 더블버퍼링 필요.
4. **코드 생성/제너릭화**: 로더·MAC의 반복 패턴(중복 8종)을 템플릿/코드젠으로 통합하면 우리 DSE(design space exploration)에 유리.
5. **end-to-end 통합**: 이 repo가 빠뜨린 patch embedding·head를 우리는 PL에 포함하거나 명확히 PS 분담을 정의해야 함.

### 정량 비교 시 주의
- 이 repo는 합성 리포트(LUT/DSP/BRAM/latency/fps)를 공개하지 않음(확인 불가). 우리 가속기와의 PPA 비교 시 직접 재합성 필요.
- 타깃이 DeiT-Tiny급(dim=192)이라 우리 모델 규모와 다를 수 있음. 파라미터화(`model.hh`)는 잘 되어 있어 dim 변경은 용이.

---

## 부록: 핵심 파일·라인 근거 요약

| 기능 | 파일:라인 |
|------|-----------|
| 모델 상수(dim/layers/patches) | `include/model.hh:5-16` |
| 정밀도 타입(INT8/INT32/fm_t) | `include/datatypes.hh:8-14` |
| AXI/블록 패킹 상수 | `include/hardware.hh:11-37` |
| 최상위 커널 인터페이스 | `src/vit.cc:60-110` |
| 12-layer 오케스트레이션 | `src/vit.cc:117-177` |
| 단일 데이터패스(ALLOCATION limit=1) | `src/vit.cc:109-110` |
| INT8 MAC(zero-point) | `src/linear.cc:317` |
| activation 동적 양자화 | `src/linear.cc:256` |
| per-channel dequant + GELU | `src/linear.cc:373-376` |
| linear DATAFLOW 3-stage | `src/linear.cc:441-447` |
| QKᵀ 4-항 정수 내적 | `src/attention.cc:130` |
| online softmax max-rescale | `src/attention.cc:201-211` |
| softmax 통계 저장(1/sum,bias) | `src/attention.cc:216-217` |
| attn·V softmax 적용 | `src/attention.cc:411-414` |
| attn·V zero-point 보정 | `src/attention.cc:428` |
| single-pass LayerNorm 통계 | `src/layernorm.cc:100-102` |
| LayerNorm var/affine | `src/layernorm.cc:124-140` |
| GELU relu-delta LUT | `src/gelu.cc:190-210` |
| residual add | `src/add.cc:14` |
| 합성 타깃/클럭 | `vitis_hls_proj/run_hls.tcl:11-12` |
| PS-PL AXI 연결 | `vivado_proj/run_vivado.tcl:18-23` |
