# trans-fat — BERT/Transformer FPGA HLS 가속기 정밀 분석

> 분석 대상: `REF/Transformer-Accel/trans-fat`
> 분석 방식: 실제 소스 Read 직접 확인 (Glob/Grep/Read만 사용, bash 미사용)
> 작성일: 2026-06-20

---

## 1. 개요

- **무엇인가**: Transformer(BERT/RoBERTa) **인코더 레이어 1개**를 **Xilinx Vitis HLS**로 가속한 코드베이스. INT8 양자화 추론을 두 장의 FPGA에 4개 파이프라인 스테이지로 분할(stage1~stage4)하여 구현하고, 세 단계(v1/v2/v3) 최적화를 적용했다. SW 측에는 BERT/RoBERTa의 정수 양자화 연산을 PyTorch로 구현한 소프트웨어 모델(`bert_sw`)이 동봉되어 HLS 커널의 "골든 모델" 역할을 한다.
- **한 줄 요약**: "I-BERT 스타일 정수 전용(integer-only) 양자화 BERT 레이어를, 2-FPGA / 4-stage 파이프라인으로 분할한 Vitis HLS 가속기 + PyTorch 양자화 골든 모델."
- **출처·논문**: 저장소 자체에 논문 인용은 없음(README는 학생/연구 프로젝트성 톤). 코드 주석에 근거가 명시: GELU는 **"Copied this from I-BERT implementation"** (`bert_sw/src/quant_ops.py:86`), softmax/layernorm도 I-BERT 계열의 정수 근사를 차용. 타깃 클러스터는 **Pitt CRC fpga-n0 노드**라고 README가 명시(`README.md:7`).
- **타깃 디바이스·보드**: **Xilinx Alveo U200** (`xilinx_u200_xdma_201830_2`), 호스트 노드에 최소 **U200 2장** 필요(`README.md:7`, Makefile `DEVICE := xilinx_u200_xdma_201830_2` @ `fpga/Makefile:60`). 빌드 산출물 디렉토리 이름도 `..._u200_xdma_201830_2_...`(예: `builds/v3.fpga1/impl_1_xilinx_u200_xdma_201830_2_bb_locked_timing_summary_routed.rpt`).
- **모델 구성(BERT-base)**: `seqlen=128, nhead=12, dhead=64, dmodel=768, ffdim=3072, eps=1e-5` (`src/v3/config.hpp:6-11`). README는 "standard BERT size, sequence length 128(수정 가능)"이라고 명시(`README.md:4`).
- **SW 대상 모델**: HuggingFace **`textattack/roberta-base-MRPC`** (GLUE MRPC 파인튜닝된 RoBERTa-base) (`bert_sw/src/quant_roberta.py:10`, `bert_sw/src/utils.py:14`).

---

## 2. 디렉토리 구조 (자체 소스 트리)

```
trans-fat/
├── README.md                      # 프로젝트 설명/빌드법/v0~v3 최적화/결과 표
├── fpga/
│   ├── README.md                  # "reference top level README" (1줄)
│   ├── Makefile                   # Vitis v++ 빌드/실행/테스트 흐름 (자체 작성, Xilinx 템플릿 기반)
│   └── utils.mk                   # check-vitis/check-xrt/device2xsa 등 헬퍼 (Xilinx 템플릿)
├── src/
│   ├── v0/  v1/  v2/  v3/         # 4개 최적화 버전 (동일 인터페이스, 구현만 진화)
│   │   ├── config.hpp             # CFG 네임스페이스 (BERT 하이퍼파라미터)
│   │   ├── host_fpga1.cpp         # FPGA1(stage1+2) XRT/OpenCL 호스트
│   │   ├── host_fpga2.cpp         # FPGA2(stage3+4) XRT/OpenCL 호스트
│   │   ├── host_all.cpp           # 2-FPGA P2P 오케스트레이션 호스트
│   │   └── stages/
│   │       ├── pipeline.cpp/.hpp  # fpga1()/fpga2() 최상위 HLS 커널 + GT 래퍼 + m_axi/s_axilite 인터페이스
│   │       ├── pipeline_test.cpp  # 전체 파이프라인 C-sim 테스트벤치
│   │       ├── Makefile           # 스테이지별 csim 빌드
│   │       ├── stage1/ stage1.cpp/.hpp  stage1_test.cpp   # QKV projection
│   │       ├── stage2/ stage2.cpp/.hpp  stage2_test.cpp   # attention+softmax+dense+LN
│   │       ├── stage3/ stage3.cpp/.hpp  stage3_test.cpp   # FFN intermediate + I-GELU
│   │       └── stage4/ stage4.cpp/.hpp  stage4_test.cpp   # FFN output + residual + LN
└── bert_sw/                       # PyTorch 양자화 소프트웨어 모델 (HLS 골든 모델)
    ├── src/
    │   ├── quant_ops.py           # I-BERT 스타일 정수 양자화 연산 라이브러리
    │   ├── quant_kernels.py       # HLS로 "거의 그대로" 매핑되는 정수 커널 (linear/matmul/requant)
    │   ├── quant_layer.py         # 4-stage 양자화 레이어 + M_* 스케일 산출 (HLS 골든 모델)
    │   ├── quant_roberta.py       # QuantRoberta: HF RoBERTa에 양자화 encoder 주입
    │   ├── layer.py               # 비양자화 readable BERT 레이어 참조 구현
    │   ├── roberta.py             # (RoBERTa 래퍼)
    │   └── utils.py               # MRPC 데이터셋 로드 + 평가 루프
    ├── dynamic_quant_ops.py       # quant_ops.py의 동적 양자화 변형 (거의 동일)
    ├── dynamic_quant_roberta.py
    └── *.ipynb                    # 실험 노트북 다수 (1-layer_bert, 2-quant_matmul_numpy, 3-quant_roberta, test_softmax/gelu/layernorm 등)
```

### 제외 항목 (third-party/vendor/생성물 — 이름만 언급, 분석 제외)
- `.git/` — git 메타데이터.
- `builds/v0..v3.{fpga1,fpga2}/` — **합성 산출물**: `*.xclbin`(비트스트림), `*_csynth.rpt`, `impl_1_kernel_util_routed.rpt`, `impl_1_..._timing_summary_routed.rpt`, 컴파일된 `host_fpga*` 실행 파일. (Makefile `all` 타깃이 자동 복사 — `fpga/Makefile:152-158`.)
- `common/includes/` — **Xilinx Vitis 벤더 유틸리티**: `xcl2/`, `oclHelper/`, `opencl/`, `cmdparser/`, `logger/`, `bitmap/`, `simplebmp/`. 호스트 코드가 `#include "xcl2.hpp"`, `cmdlineparser.h`로 사용(`src/v3/host_fpga1.cpp:1`, `host_all.cpp:2`).
- `common/utility/`, `common/data/` — Xilinx 예제 빌드 스크립트/README 생성기/로고 BMP. 자체 가속기와 무관.
- `bert_sw/**/__pycache__/`, `.ipynb_checkpoints/` — 생성물(`.gitignore:1-2`).
- `src/*/stages/*/stage#_test` (확장자 없는 파일) — 컴파일된 csim 바이너리(생성물).

---

## 3. 핵심 모듈 정밀 분석 (★)

전체 설계의 골격은 다음 두 최상위 HLS 커널이다(`src/v3/stages/pipeline.cpp`):

| 커널 | 담당 FPGA | 포함 스테이지 | 의미 |
|---|---|---|---|
| `fpga1(...)` (`pipeline.cpp:45-82`) | U200 #1 | stage1 + stage2 | QKV projection → Self-Attention → Dense → Residual → LayerNorm |
| `fpga2(...)` (`pipeline.cpp:84-108`) | U200 #2 | stage3 + stage4 | FFN intermediate(+I-GELU) → FFN output → Residual → LayerNorm |

각 `extern "C"` 커널 옆에는 동일 시그니처의 `*_gt`(ground truth) SW 함수가 존재하여(`fpga1_gt`/`fpga2_gt` @ `pipeline.cpp:10-41`) 호스트가 동일 입력으로 SW 결과와 HW 결과를 비교한다.

### 3.0 자료형/양자화 컨벤션 (전 스테이지 공통)
- 활성/가중치: **`int8_t`**, 바이어스: **`int32_t`**, 누산: **`int32_t`**, LayerNorm 중간값: **`int16_t`**, 정규화 weight/bias: **`int16_t`** (`pipeline.hpp:4-32`의 args 구조체에서 직접 확인).
- 재양자화(requantize): 정수 누산값 × 부동소수 스케일 `M_*` → `int8_t` 캐스팅. 예: `out[..] = int8_t(in[..] * M_scale)` (`stage1.cpp:29`). `M_*`는 호스트/Python에서 사전 계산한 fixed-point 스케일(곧 `acc_scale / out_scale`).

---

### 3.1 Stage 1 — QKV Projection (`src/v3/stages/stage1/stage1.cpp`)

**역할**: 입력 활성 `in <seqlen,dmodel>`에 대해 Query/Key/Value 세 개의 선형층(각각 `<dmodel,dmodel>` weight)을 동시 수행 후 INT8 재양자화.

**(a) SW 골든 (`linear_sw1` @ stage1.cpp:13-24, `stage1_gt` @ 35-52)**
- 표준 3중 루프 GEMM: `out[i*M+j] += A[i*K+k] * B[k*M+j]`, bias로 초기화(`stage1.cpp:17-21`).
- `stage1_gt`는 q/k/v를 int32로 누산 후 `requantize1`로 int8 변환(`stage1.cpp:41-47`).

**(b) HLS 커널 (`stage1` @ stage1.cpp:196-246) — DATAFLOW 스트리밍 아키텍처**
- `#pragma HLS dataflow` (`stage1.cpp:199`): read→compute→write를 task-level 병렬 파이프라인으로 동시 실행. Q/K/V 3계열을 완전히 병렬화("Can run all linear layers in parallel" 주석 @ 198).
- **producer/consumer 분리**: `read_input1`(입력을 3개 스트림으로 fan-out, `stage1.cpp:58-78`), `read_weights1`(가중치 스트림, 80-96), `read_bias1`(98-113), `linear_fused1`(연산, 138-193), `write_out1`(124-131). 모두 `hls::stream<>`로 연결.
- 스트림 깊이: `#pragma HLS stream variable=... depth=128` × 9개(in/weight/bias 각 q/k/v) (`stage1.cpp:216-226`).
- **타일링**: `TILE_SIZE1 = 128` (`stage1.hpp:4`). seqlen=128, dmodel=768 → it=1, jt=6, kt=6 타일.
- **핵심 연산 루프 (`linear_fused1` @ 138-193)**:
  - 출력 타일 버퍼 `int32_t out_block[128][128]` + `#pragma HLS array_partition dim=2 complete` (`stage1.cpp:145`) → dim2(j) 완전 분할로 128-way 병렬 누산 가능.
  - `int8_t B_line[128]` + `#pragma HLS array_partition dim=1 complete` (`stage1.cpp:146`) → weight 라인 완전 분할.
  - 내부 곱셈누산: `#pragma HLS PIPELINE II=1` (외부 i루프 @ 173) + `#pragma HLS unroll`(내부 j루프 @ 176) → **i당 j=128 MAC을 1 cycle에** 발행하는 출력-스테이셔너리 패턴.
  - 재양자화 융합: 타일 완료 후 `out_stream.write(int8_t(out_block[i][j] * M_scale))` (`stage1.cpp:186`) — linear+requant을 한 커널에 융합.
- **입력 전치(transpose) 트릭(v3)**: `read_input1`이 `in[(kt*TILE+k)*seqlen + it*TILE+i]`로 읽음(`stage1.cpp:69`) — 즉 호스트가 입력을 `<dmodel,seqlen>`(전치) 레이아웃으로 미리 적재. 호스트에서 `stage1_in[i*seqlen+j] = s1_args.in[j*dmodel+i]`로 전치(`host_fpga1.cpp:91-95`). 이는 A 행 연속 접근으로 DDR burst 효율을 올리는 v2/v3 최적화("Transpose A matmul input", README v2).

---

### 3.2 Stage 2 — Self-Attention + Dense + Residual + LayerNorm (`src/v3/stages/stage2/stage2.cpp`, 가장 복잡)

stage2는 attention 전체 + 출력 dense + skip + layernorm을 하나의 HLS 커널로 묶는다. `stage2` @ 561-577이 4개 서브블록을 순차 호출.

**(a) Attention Scores = QK^T / sqrt(d) + Softmax**
- SW GT: `attention_scores`(reshape/transpose 인덱싱으로 `<seqlen,dmodel>`→`<nhead,seqlen,dhead>` view, `stage2.cpp:202-237`) → `scale`(÷√dmodel, 36-42) → `softmax`(48-69) → `attention_values`(probs·V, 239-267).
- **HLS 융합 `attention_scores_fused` (`stage2.cpp:307-339)**:
  - head별로 query_row/key_row를 레지스터 버퍼에 적재(`array_partition complete` @ 314-315) 후 dhead=64 내적을 `#pragma HLS unroll`(330)로 완전 병렬화.
  - `rowbuff[j] = accum / divisor` 후 한 row(seqlen=128) 완성되면 즉시 `softmax_fused` 호출(336) → **scores 한 줄마다 softmax를 흘려보내는 row-streaming**.
- **`softmax_fused` (`stage2.cpp:71-93)**: 수치안정 softmax. (1) max 탐색, (2) `sum += exp(x-m)`, (3) `constant = m + log(sum)`, (4) `out = int8_t(exp(x - constant) * M_softmax)`. **부동소수 `exp`/`log` 사용**(double) — 정수 전용이 아님(주의점, 9절).

**(b) Attention Values = probs·V**
- **HLS 융합 `attention_values_fused` (`stage2.cpp:341-388)**: probs_row(seqlen) + value_row(dhead) 버퍼, dhead 누산을 `unroll`(378)로 병렬화, 출력 시 `int8_t(row_buf[j] * M_attention_out)` 재양자화 융합(384). 출력 레이아웃은 `<seqlen,dmodel>`로 transpose-back(인덱싱 `i*nhead*dhead + n*dhead + j`).

**(c) Dense + Residual (tiled streaming)**
- **`linear_dataflow2` (`stage2.cpp:541-558)**: `#pragma HLS dataflow`로 `read_weight2`(405-423)/`read_bias2`(425-437)/`read_skip2`(439-452)/`linear_fused2`(454-504)를 동시 실행. 스트림 깊이 128(547-549).
- `read_skip2`가 skip을 **전치 인덱싱** `skip[(jt*TILE+j)*seqlen + it*TILE+i]`로 읽음(447) — residual을 전치 레이아웃으로 적재.
- `linear_fused2`: stage1과 동일한 출력-스테이셔너리(`out_block[128][128]` dim2 complete @ 462, `B_line` dim1 complete @ 463, PIPELINE II=1 @ 473/490, unroll @ 493). 출력 시 `write_output`(396-403)이 `requant(ob,sb,Md,Mr)`(390-394)로 **dense 재양자화 + skip 가산 + residual 재양자화**를 한 번에 수행: `int16_t(( int8_t(ob*Md)+sb )*Mr)`.

**(d) LayerNorm 융합 (`layernorm_fused2` @ stage2.cpp:507-539)**
- SW GT(`layernorm_sw2` @ 141-181)는 mean/diff/제곱/var/std/div/affine을 별도 버퍼로 분리. HLS 융합본은:
  - 1차 패스: row mean 계산 후 `act -= m`(514-523).
  - 2차 패스: `acc16 += act²/dmodel` → `stdev = sqrt(acc16 + C)` (C = eps/scaling_factor) → `act /= stdev` → affine `act*norm_weight[j]+norm_bias[j]` → `out[j*seqlen+i] = int8_t(... * M_stage)` (525-537).
  - **출력이 전치 레이아웃**(`out[j*seqlen+i]`, 535) — 다음 스테이지(또는 FPGA2)가 전치 입력을 기대하기 때문.
- 주석에 "for some reason rn if I fuse this int the next loops it doesn't work"(512) — 완전 파이프라인화 미완 상태임을 저자가 인정.
- `int16_t` 산술 사용(누산 오버플로 위험은 9절 한계 참조).

---

### 3.3 Stage 3 — FFN Intermediate + I-GELU (`src/v3/stages/stage3/stage3.cpp`)

**역할**: `<seqlen,dmodel> × <dmodel,ffdim>` (768→3072) 선형층 + GELU + INT8 재양자화. 가장 큰 GEMM(seqlen·dmodel·ffdim = 128·768·3072).

**(a) I-GELU 정수 근사 (`gelu_sw` @ 28-71, `gelu_fused` @ 107-130)** — **I-BERT 차용**
- 상수: `k=1.4142(√2), coef_0=-0.2888, coef_1=-1.769, coef_2=1/coef_0, constant=14`(`stage3.cpp:38-42`).
- int_erf 다항 근사: `int_erf_scaling = scale/k`, `b_int=coef_1/int_erf_scaling`, `c_int=coef_2/int_erf_scaling²`, `sigmoid_scaling = int_erf_scaling²·coef_0·2^14`, `shift_int=1/sigmoid_scaling`(46-54).
- 코어: `sign=sgn(x)`, `abs_int=min(|x|,-b_int)`, `y_int=sign·((abs_int+b_int)²+c_int)`, `sigmoid_int=y_int>>14`, `x·(sigmoid_int+shift_int)` → 재양자화(58-68). **부동소수 exp/erf 호출 없음 → 정수 다항식 근사**(softmax와 대조적).
- `gelu_fused`(107-130)는 동일 로직을 스칼라 단위로 만들어 `int8_t(gelu_in * M_stage3)` 재양자화까지 융합.

**(b) HLS 커널 (`stage3` @ 280-307)**
- `#pragma HLS dataflow`(299) + `read_A`/`read_B`/`read_bias`/`linear_fused`/`write_out` 5-stage. 스트림 깊이 8(294-297).
- 타일: `TILE_SIZE=128, TILE_SIZE_J=128` (`stage3.hpp:5-6`). `read_A`가 A를 전치로 읽음(`A[(kt*TILE+k)*seqlen + it*TILE+i]` @ 144).
- `linear_fused`(205-277): `out_block[128][128]`(dim2 complete @ 228), `B_line[128]`(dim1 complete @ 230), PIPELINE(258) + unroll(261). 타일 완료 시 `gelu_fused`로 GELU+requant 융합(272). `write_out`은 출력 전치 적재(`out_T[(jt*TILE_J+j)*seqlen + it*TILE+i]` @ 198).

---

### 3.4 Stage 4 — FFN Output + Residual + LayerNorm (`src/v3/stages/stage4/stage4.cpp`)

**역할**: `<seqlen,ffdim> × <ffdim,dmodel>` (3072→768) 선형층 + skip + INT8/INT16 재양자화 + LayerNorm.

**(a) SW GT (`stage4_gt` @ 118-136)**: linear_sw4 → requantize4(M_dense_acc) → add_skip → requantize4(M_residual)→int16 → layernorm_sw(75-116) → requantize4(M_stage4)→int8.
**(b) HLS (`stage4` @ 387-396)**:
- `linear_dataflow4`(362-383): `#pragma HLS dataflow`(376) + read_A4/read_B4/read_bias4/read_skip4/linear_fused4. 스트림 깊이 128(371-374). 주석(364): "linear가 끝난 뒤 layernorm 시작 — 향후 스트리밍 융합 가능"하나 우선순위 낮아 미적용.
- `linear_fused4`(226-278): TILE_SIZE4=TILE_SIZE4_J=128(`stage4.hpp:5-6`). out_block dim2 complete(234), B_line dim1 complete(236), PIPELINE(266)+unroll(269). `write_out`(148-155)이 `requant_out`(142-146)로 dense·skip·residual 재양자화 융합.
- `layernorm_fused4`(328-360): stage2와 동일 2-패스 LayerNorm. 단 여기서는 출력이 **비전치** `out[i*dmodel+j]`(356) — 레이어 최종 출력이라 표준 레이아웃.
- 주석 처리된 대안 `layernorm_fused`(281-326)에 norm_weight/bias 사전 버퍼링 + `array_partition cyclic factor=32` 시도 흔적이 남아있음(미사용).

---

### 3.5 HLS pragma 정량 근거 (Grep `#pragma HLS`, src/*.cpp, 총 217건)

버전별 진화가 **정량적으로** 확인됨 — pragma 수가 곧 최적화 강도:

| 파일 | pragma 수 |
|---|---|
| `v0/stages/stage1~4 + pipeline` | **0** (전부 순수 SW 루프, pragma 없음 — `v0/stages/stage1.cpp` 직접 확인) |
| `v1/stages/pipeline.cpp` | 24 |
| `v1/stages/stage1/2/3/4.cpp` | 6 / 6 / 8 / 7 |
| `v2/stages/pipeline.cpp` | 39 |
| `v2/stages/stage1/2/3/4.cpp` | 4 / 11 / 6 / 11 |
| `v3/stages/pipeline.cpp` | 39 |
| `v3/stages/stage1/2/3/4.cpp` | 14 / 15 / 11 / 16 |

- **v0**: pragma 0개. `stage1.cpp`의 `linear_fused`는 단순 3중 루프(`v0/stages/stage1/stage1.cpp:63-75`). → README 결과표 fpga1=4723.71ms(`README.md:68`).
- **v3 pipeline.cpp의 39개**: 대부분 `#pragma HLS interface m_axi port=... bundle=gmemN`(fpga1 15개 gmem 번들 @ `pipeline.cpp:51-72`, fpga2 9개 @ 88-96) + `s_axilite ... bundle=control`(스칼라 M_* + return). → DDR 다중 뱅크 동시 접근을 위한 분리 번들링.
- 연산부 핵심 pragma 4종: `dataflow`, `stream depth=`, `array_partition complete`, `PIPELINE II=1` + `unroll`. 출력-스테이셔너리 타일 GEMM이 모든 스테이지 공통 패턴.

---

### 3.6 호스트 코드 (XRT/OpenCL)

**(a) 단일 FPGA 호스트 (`src/v3/host_fpga1.cpp`)**
- 표준 Vitis 흐름: `xcl::get_xil_devices`(130) → `read_binary_file`(133) → device 루프에서 `cl::Program`/`cl::Kernel(program, "fpga1")`(149) → `aligned_allocator` 벡터로 입출력 버퍼(59-69) → `cl::Buffer(..., CL_MEM_USE_HOST_PTR|CL_MEM_READ_ONLY, ...)`(162-221) → `krnl.setArg(0..22)`(224-246) → `enqueueMigrateMemObjects(...,0)`(249) → `enqueueTask(krnl)`(271) → `q.finish()` → 결과 migrate back(281) → SW GT(`fpga1_gt` @ 125)와 `check`(308) 비교 + `chrono` 타이밍(268-289).
- 입력 데이터는 `genmat`(가짜 modular 패턴, 22-32)으로 생성 — **실제 RoBERTa 가중치가 아닌 합성 검증 데이터**. 즉 호스트는 **수치 검증·타이밍용 testbench**이지 end-to-end 추론 파이프라인이 아니다.

**(b) 2-FPGA P2P 호스트 (`src/v3/host_all.cpp`)** — 핵심 차별점
- `xcl::P2P::getMemObjectFd`/`getMemObjectFromFd`를 함수 포인터로 바인딩(`host_all.cpp:14-15`).
- 두 xclbin을 cmdparser로 받음: `-x1`(fpga1), `-x2`(fpga2) (`host_all.cpp:183-189`). Makefile `test PART=all`이 이를 호출(`fpga/Makefile:219`).
- 두 디바이스에 각각 `krnl_fpga1`/`krnl_fpga2` 생성(285/289), `xcl::P2P::init`(291).
- **실행 시퀀스(450-503)**: ① `enqueueTask(krnl_fpga1)` + finish(453-455) → ② **P2P 전송**: `getMemObjectFd(buffer_stage3_fc_in)`로 FD 추출(475) → `getMemObjectFromFd`로 FPGA1의 stage2_out을 FPGA2 주소공간에 import(481) → `clEnqueueCopyBuffer(queue0, buffer_stage2_out, exported_buf, ...)`로 **호스트 DRAM 우회 직접 FPGA→FPGA 복사**(483-486) → ③ `enqueueTask(krnl_fpga2)`(491) → ④ 결과 migrate back(494-497).
- 주석에 비-P2P 폴백(호스트 memcpy 경유) 코드가 보존되어 있음(459-468) — P2P 대비 성능 비교용.

---

## 4. 데이터플로우 (BERT 인코더 레이어 추론 흐름)

**SW 측 (PyTorch 골든, `bert_sw/src/quant_layer.py`)**
```
hidden_states <1,128,768>
  └─ stage1: Q/K/V = linear_kernel(act_int, W_qkv^T, bias) → requantize_kernel(M_q/k/v)
  └─ stage2: scores=matmul(Q,K^T) → /√dhead → tensor_quant_softmax → requant(M_probs)
             attn=matmul(probs,V) → requant(M_attn_out) → dense linear → +skip
             → tensor_quant_layernorm → requant(M_stage2)
  └─ stage3: linear(768→3072) → tensor_quant_gelu(I-GELU) → requant(M_stage3)
  └─ stage4: linear(3072→768) → +skip → tensor_quant_layernorm → requant(M_stage4)
```
- `layer_kernel_gt`(quant_layer.py:141-289)가 **각 M_* 스케일을 사전 계산**한다: 예 `M_query = query_acc_scale / query_out_scale`(169), `M_attention_out = attn_prob_scale·value_out_scale / attn_out_scale`(223), `M_stage2 = 1/attention_out_scale`(250). 이 M_* 값들이 곧 HLS 커널에 `float` 인자로 전달된다.

**HW 측 (HLS, `src/v3/stages/pipeline.cpp`)**
```
[U200 #1: fpga1 커널]
  stage1(in_T, →Q,K,V, W_qkv, bias_qkv, M_q,M_k,M_v)        # dataflow QKV GEMM
  stage2(Q,K,V, skip=in, →stage2_out, W_dense, bias, M_*,   # attn+softmax+dense+LN
         norm_w, norm_b, M_stage2)
        │
        ▼  P2P (clEnqueueCopyBuffer, 호스트 DRAM 우회)
[U200 #2: fpga2 커널]
  stage3(fc_in=stage2_out, W_fc1, bias, →fc3_to_fc4_buff,   # FFN1 + I-GELU
         dense_acc_scale, M_stage3)
  stage4(fc3_to_fc4_buff, skip=stage3_fc_in, W_fc2, bias,   # FFN2 + residual + LN
         →dense_out, norm_w, norm_b, M_*, M_stage4)
```
- 스테이지 간 중간 텐서는 대부분 **전치 레이아웃**으로 주고받음(각 stage write가 `[col*seqlen+row]`로 적재 → 다음 stage read가 그대로 행연속 소비). 이 전치 규약이 v2/v3 burst 최적화의 핵심.

---

## 5. HW/SW 매핑표 (Python 양자화 모델 ↔ HLS 커널 ↔ 호스트)

| 연산 | Python (bert_sw) | HLS 커널 함수 (v3) | 호스트 인자 |
|---|---|---|---|
| INT8 대칭 양자화 스케일 | `tensor_quant_scale` (`quant_ops.py:11-33`) | (호스트/오프라인 사전계산) | `M_*` 부동소수 |
| INT8 linear (W^T·a+bias) | `linear_kernel` (`quant_kernels.py:8-25`) | `linear_sw#`/`linear_fused#` (각 stage) | weight_t, bias 버퍼 |
| INT8 matmul (QK^T, probs·V) | `matmul_kernel` (`quant_kernels.py:28-44`) | `attention_scores_fused`/`attention_values_fused` (`stage2.cpp:307,341`) | — |
| INT32→INT8 재양자화 | `requantize_kernel` (`quant_kernels.py:47-56`) | `requantize#`/`requant`/`requant_out` (`stage1.cpp:26`, `stage2.cpp:390`, `stage4.cpp:142`) | `M_*` |
| Softmax | `tensor_quant_softmax` (I-Softmax, `quant_ops.py:123-173`) | `softmax`/`softmax_fused` (`stage2.cpp:44,71`) ※HLS는 float exp/log | `M_attention_probs` |
| GELU | `tensor_quant_gelu` (I-GELU, `quant_ops.py:83-115`) | `gelu_sw`/`gelu_fused` (`stage3.cpp:28,107`) | `dense_acc_scale`(=M_gelu), `M_stage3` |
| LayerNorm | `tensor_quant_layernorm` (I-LayerNorm, `quant_ops.py:176-228`) | `layernorm_sw#`/`layernorm_fused#` (`stage2.cpp:141,507`, `stage4.cpp:75,328`) | `norm_weight/bias`, `M_residual`, `M_stage#` |
| Stage 조립 | `stage1~4`(`quant_layer.py:18,37,105,116`) | `fpga1`/`fpga2`(`pipeline.cpp:45,84`) | args 구조체(`pipeline.hpp:4-32`) |
| 레이어 골든 + M_* 산출 | `layer_kernel_gt` (`quant_layer.py:141-289`) | `fpga1_gt`/`fpga2_gt` (`pipeline.cpp:10,31`) | genmat 합성 데이터 |
| 전체 모델 | `QuantRoberta`/`encoder` (`quant_roberta.py:8`, `quant_layer.py:377`) | (레이어 단위 가속, 인코더 루프는 SW) | — |

- **설계 의도 명문화**: `quant_kernels.py:4-6` "These kernels ... operate mainly on integer types. **These will be mapped, almost exactly as they are, to HLS kernels.**" → Python↔HLS 1:1 대응이 의도적.

---

## 6. 빌드·실행 (Makefile / utils.mk / Vitis 흐름)

**의존성 로드** (`README.md:12-16`): `module load xilinx/vitis/2020.2`, `module load libfaketime`, `source /opt/xilinx/xrt/setup.sh`.

**빌드** (`fpga/`에서, `README.md:22`):
```
faketime 'last year' make all TARGET=<hw|hw_emu|sw_emu> VERSION=<0..3> PART=<fpga1|fpga2|all> JOBS=<N>
```
- `faketime`은 라이선스 날짜 우회용(libfaketime). `--save-temps --hls.jobs/vivado.*.jobs $(JOBS)` 병렬 합성(`Makefile:111`).
- 커널 빌드: `v++ -c -k $(PART)` → `.xo`(`Makefile:172-174`), `v++ -l`로 링크 → `.link.xclbin` → `v++ -p`로 패키지 → `.xclbin`(`Makefile:175-182`). 컴파일 단위는 `pipeline.cpp + stage1~4.cpp`(`Makefile:116-120`).
- 호스트 빌드: `g++` + xcl2/cmdparser/logger + `host_$(PART).cpp`(`Makefile:99-100, 185-186`).
- `TARGET=hw`이면 산출물(`host`, `.xclbin`, `*_csynth.rpt`, `*_kernel_util_routed.rpt`, timing rpt)을 `builds/v$(VERSION).$(PART)/`로 자동 복사(`Makefile:151-159`).

**실행/테스트** (`README.md:28`, `Makefile:208-226`):
```
make test VERSION=<0..3> PART=all        # host_all -x1 ...fpga1.xclbin -x2 ...fpga2.xclbin (P2P)
make test VERSION=<0..3> PART=fpga1      # host_fpga1 <xclbin>
```
- sw_emu/hw_emu는 `XCL_EMULATION_MODE` 환경변수로 실행(`Makefile:210-214`). utils.mk가 `check-vitis`(XILINX_VITIS), `check-xrt`(XILINX_XRT), `device2xsa` 등 가드 제공(`utils.mk:42-59`).

**결과(README.md:51-91, latency ms)**:
| Version | fpga1 | fpga2 | all | 핵심 최적화 |
|---|---|---|---|---|
| v0 | 4723.71 | 10950.90 | 15676.30 | 없음 |
| v1 | 274.98 | 120.91 | 397.45 | 타일링, 입출력 버퍼링, 내부곱 unroll (~17×~39×) |
| v2 | 48.36 | 95.60 | 145.27 | A 전치, A.T 캐시라인, j타일 확대, attention head unroll |
| v3 | **35.03** | **71.76** | **110.99** | linear 층 DDR 입출력 스트리밍 |
- v0→v3: fpga1 약 **135×**, 전체 약 **141×** 속도 향상(확실, README 직접).

---

## 7. 의존성

- **HW 툴체인**(확실, README/Makefile): Xilinx **Vitis 2020.2**, **XRT**, **libfaketime**, 타깃 **Alveo U200**(`xilinx_u200_xdma_201830_2`). P2P를 위해 xdma/nodma 디바이스 필요(`host_all.cpp:254` 경고문).
- **벤더 라이브러리**(common/includes): xcl2, oclHelper, cmdparser, logger (Xilinx 예제 유틸).
- **SW(Python, bert_sw)**: `torch`, `numpy`, `transformers`(`RobertaForSequenceClassification`), `datasets`(GLUE MRPC), `tqdm` (`quant_roberta.py:1-4`, `utils.py:1-6`). requirements.txt는 발견되지 않음(확인 불가 — 노트북 환경 가정).

---

## 8. 강점

1. **명확한 SW↔HW 공동설계(co-design)**: 모든 HLS 스테이지가 동일 시그니처의 `*_gt` SW 함수를 동봉하고, Python `quant_kernels.py`가 "HLS로 거의 그대로 매핑" 의도를 명시. 검증·디버깅이 체계적(호스트가 매 실행 SW vs HW `check`).
2. **버전별 점진 최적화의 교본**: v0(pragma 0) → v3(타일링+dataflow+streaming)까지 **동일 알고리즘의 4단계 진화**가 한 저장소에 보존되어 HLS 최적화 효과(135×)를 직접 대조 가능.
3. **정수 전용 양자화 연산 차용(I-BERT)**: I-GELU(정수 다항 근사), I-Softmax, I-LayerNorm을 PyTorch+HLS 양쪽에 구현 → FPGA에서 DSP 절약형 비선형 연산 레시피 확보.
4. **2-FPGA P2P 파이프라인**: stage2_out을 호스트 DRAM 왕복 없이 `clEnqueueCopyBuffer`로 FPGA1→FPGA2 직접 전달(`host_all.cpp:483`) — 멀티-디바이스 스케일아웃의 실전 예시.
5. **출력-스테이셔너리 타일 GEMM + 전치 레이아웃 규약**: `out_block[T][T]` dim2 complete + PIPELINE II=1 + unroll 조합이 모든 선형층에 일관 적용 → 재사용 가능한 패턴.

## 9. 한계 / 주의점

1. **레이어 1개만 가속**: 전체 BERT(12 레이어) end-to-end가 아니라 **단일 인코더 레이어**. 인코더 루프(`quant_layer.py:377-386`)는 SW에 남아있고, HW는 1 레이어 단위. (README:4 "a transformer layer".)
2. **호스트가 합성 데이터 사용**: `genmat`(modular 패턴)으로 가중치/입력 생성(`host_fpga1.cpp:83-89`) — 실제 RoBERTa 가중치 적재 경로 없음. HW는 정확도 검증이 아닌 **수치 일치(bit-exact vs SW GT) + 타이밍** 측정용.
3. **Softmax는 부동소수**: HLS `softmax_fused`가 `double exp/log` 사용(`stage2.cpp:60-65,85-91`) — GELU(정수 근사)와 달리 "정수 전용"이 아니며 DSP/리소스·정확도 양쪽에 영향. Python I-Softmax(`quant_ops.py:123`)는 정수형이나 HLS엔 미반영.
4. **LayerNorm int16 누산 오버플로 위험**: `acc16 += int16_t(act²/dmodel)`(`stage2.cpp:528`, `stage4.cpp:349`)는 int16 누산 — 큰 분산에서 오버플로 가능(SW GT는 int32 경유). 저자도 "완전 융합 시 동작 안 함"을 주석에 명시(`stage2.cpp:512`).
5. **완전 파이프라인화 미완**: stage4 linear↔layernorm 스트리밍 미융합(`stage4.cpp:364` 주석), 주석 처리된 cyclic partition 시도 흔적(`stage4.cpp:281-326`) — 추가 최적화 여지.
6. **TILE_SIZE=128 고정**: 모든 타일이 seqlen(128)에 묶임 → seqlen 변경 시 타일/array_partition 재설계 필요. dmodel/ffdim 타일도 128 단위 가정.
7. **Vitis 2020.2 / U200 특정**: 최신 Vitis·디바이스 이식 시 P2P API(`xclGetMemObjectFd`)·번들링 재검토 필요.

---

## 10. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA + XR 시선추적) 시사점

**재사용 가능한 패턴**
1. **SW 골든 모델 동봉 구조**: 각 HLS 커널에 `*_gt` 짝 + 호스트 bit-exact `check`. ViT 가속기에서도 PyTorch 골든 ↔ HLS를 1:1로 두면 회귀 검증이 견고해진다(`quant_kernels.py` 패턴 차용).
2. **버전 분기(v0~v3) 디렉토리 전략**: 동일 인터페이스를 유지한 채 최적화 단계를 별 디렉토리로 보존 → DSE/논문 ablation에 직결. 우리 HG-PIPE 파이프라인 단계별 비교에 그대로 적용 가능.
3. **출력-스테이셔너리 타일 GEMM 템플릿**: `out_block[T][T]`(dim2 complete) + `B_line`(dim1 complete) + `PIPELINE II=1`+`unroll`. ViT의 patch-embed/QKV/MLP 선형층에 동일 적용 가능. 단, HG-PIPE가 추구하는 **무중단(stall-free) 전층 파이프라인**과 비교하면 trans-fat는 스테이지 내부만 dataflow이고 스테이지 간은 순차(버퍼 경유)라 처리량 한계가 있다 → 우리는 스테이지 경계까지 스트림으로 잇는 설계가 차별점.
4. **I-GELU 정수 다항 근사**: `stage3.cpp:107-130`의 b_int/c_int/shift_int 레시피는 DSP 절약형 GELU LUT/근사 설계에 직접 재사용. XR 저지연(LayerNorm/GELU가 latency-critical) 환경에서 유용.
5. **P2P 멀티-FPGA 전달**: 만약 시선추적 파이프라인(인코더 + 디코더/헤드)을 2칩으로 나눌 경우 `getMemObjectFromFd`+`clEnqueueCopyBuffer` 패턴이 호스트 왕복 제거 레퍼런스.

**주의점(우리가 피해야 할 것)**
- trans-fat의 **레이어 1개·합성데이터·float softmax·int16 LN**는 프로토타입 수준. 실시간 XR(저지연·실가중치·전체 모델)에는 ①전체 인코더 온칩 또는 가중치 스트리밍, ②정수 softmax(I-Softmax HLS화), ③LN 누산 비트폭 확대(int32), ④seqlen-가변 타일링이 필수.
- trans-fat는 **처리량보다 검증·이식성** 우선 설계 → HG-PIPE식 초당 프레임/이미지 처리량 목표와는 출발점이 다름. 타일 GEMM은 가져오되, 파이프라인 토폴로지는 재설계 권장.

---

## 11. 근거/한계 표기

- **확실(코드 직접 확인)**: 디렉토리 구조, config 값(`config.hpp`), 4-stage 분할 및 fpga1/fpga2 매핑(`pipeline.cpp`), 각 스테이지 알고리즘·pragma(stage1~4.cpp v3 + v0 baseline 직접 Read), pragma 정량 카운트(Grep 217건), 호스트 XRT/P2P 흐름(`host_fpga1.cpp`/`host_all.cpp`), Python 양자화 연산/레이어/모델(`quant_ops/kernels/layer/roberta/utils.py`), 빌드 흐름(`Makefile`/`utils.mk`), README 성능표·디바이스·의존성.
- **추정**: I-BERT 논문이 출처라는 점은 코드 주석("Copied this from I-BERT implementation", `quant_ops.py:86`)에 근거한 강한 정황이나 논문 메타데이터(citation/DOI)는 저장소에 없음. v1/v2 HLS 세부는 v3·README 기반으로 추론(v2 stage2 attention 일부만 직접 확인).
- **확인 불가**: Python 의존성 정확한 버전(requirements/environment 파일 미발견), 실제 RoBERTa 가중치를 HW에 적재한 end-to-end 정확도(호스트는 genmat 합성데이터만 사용), builds/ 합성 리포트의 LUT/DSP/BRAM 실수치(제외 대상이라 미분석).
