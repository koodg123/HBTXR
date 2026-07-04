# Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280 정밀 분석

> 분석 대상: `REF/ViT-Accelerator/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280`
> 분석 범위: **자체 추가 코드만**(Tiny-GPT 커널/호스트/학습 스크립트). Vortex 베이스(RTL/runtime/sim)는 매핑 근거용으로만 참조.
> 작성 기준: 실제 소스 라인 근거. "추정"/"확인 불가"는 본문에 명시.

---

## 1. 개요

- **목적**: 오픈소스 RISC-V GPGPU인 **Vortex**(Apache-2.0, MICRO'21)를 AMD/Xilinx **Alveo U280** FPGA에 올리고, 그 위에서 소형 GPT 스타일 텍스트 생성을 OpenCL 커널로 추론·시연하는 프로젝트.
- **한줄 요약**: "Vortex GPGPU 베이스에, 2-layer mat-vec 형태의 초소형 언어모델(자칭 Tiny-GPT)을 OpenCL 커널 + 호스트 루프로 얹어 simx/RTL/XRT(U280)에서 토큰 생성을 돌리는 데모."
- **베이스 = Vortex GPGPU**: RTL 코어 파이프라인, 캐시/메모리 서브시스템, FPU, OpenCL 런타임/드라이버(simx, rtlsim, opae, xrt), `ci/blackbox.sh` 실행 스크립트는 전부 upstream Vortex 그대로다. README L26-30이 이를 명시한다("This work builds on Vortex GPGPU ... Major changes here: TinyGPT kernels, host flow, U280 configs, and build scripts").
- **타깃**: AMD Alveo U280 (`xilinx_u280_gen3x16_xdma_1_202211_1`, README L150-152). 시뮬레이션은 simx(C++ cycle-approx)/Verilator RTL.
- **대회 이력**: AMD Open Hardware Competition 2025 (Adaptive Computing, Student) 수상작 (README L5-7), University of Essex 팀 AOHW25_616.
- **중요 정정(마케팅 vs 실제)**: README/소개문은 "GPT-style", "int8", "quant"를 언급하나, **실제 모델은 어텐션이 없는 2-layer MLP**이며 **양자화도 없고 전부 FP32**다(근거는 §3·§4·§10). "int8"/"quant"는 모델이 출력하는 어휘 토큰 이름일 뿐 연산 정밀도가 아니다.

---

## 2. 디렉토리 구조

### 2.1 자체 추가 코드 (분석 대상 — 본 repo의 실질 기여)

```
tests/opencl/
├── tinygptv1/                 ← Tiny-GPT v1 (가장 단순)
│   ├── kernel.cl              매트벡 2-layer 커널 (matvec2layer, phase 0/1)
│   ├── main.cc               OpenCL 호스트: 가중치 로드 + 토큰 생성 루프 + top-k 샘플링
│   ├── Makefile              PROJECT=tinygpt, cnpy.cpp 링크, -lz
│   └── scripts/train_model.py  NumPy로 모델 학습 후 .npy 5개 export
├── tinygptv2/                 ← Tiny-GPT v2 (fused/multi-core 확장)
│   ├── kernel.cl              fused 커널 2종 (tinygpt_persist_fused, ffn_to_logits_slice)
│   ├── main.cc               2개 엔진(persist/slice) + 호스트 softmax/topk + repetition penalty
│   ├── Makefile              tinygptv1과 동일 패턴
│   └── scripts/train_model.py  v1과 사실상 동일한 학습 스크립트
└── (cnpy.cpp / cnpy.h)        .npy 로더 (오픈소스 cnpy 차용, 빌드 시 동봉)
```

(추가 자료: `docs/xilinx_fpga_guide.md`는 U280/U50 비트스트림 빌드 절차 일부 수정. `docs/codebase.md`, `docs/simulation.md`는 upstream Vortex 문서 거의 그대로 — 후자는 기본 FPGA가 여전히 "Arria10"로 적혀 있어 U280용으로 갱신되지 않음.)

### 2.2 Vortex 베이스 (매핑 근거로만 참조, 자체 수정 없음으로 확인)

```
hw/rtl/
├── core/    VX_core.sv, VX_schedule.sv, VX_fetch.sv, VX_issue.sv,
│            VX_alu_int.sv, VX_alu_muldiv.sv, VX_alu_unit.sv,
│            VX_lsu_unit.sv/_slice.sv/_adapter.sv 등 (SIMT 코어 파이프라인)
├── cache/   L1/L2/L3 캐시 서브시스템
├── mem/     메모리 서브시스템 (HBM/DDR 인터페이스)
├── fpu/     VX_fpu_fpnew.sv, VX_fpu_fma.sv, VX_fpu_dsp.sv, VX_fpu_div/sqrt/cvt/ncp.sv
│            VX_fcvt_unit.sv, VX_fp_rounding.sv, VX_fp_classifier.sv (RV32F/RV64D FPU)
├── interfaces/, libs/
hw/syn/xilinx/xrt/  ← U280/U50 비트스트림 합성 (PREFIX/PLATFORM/TARGET/NUM_CORES make)
runtime/  include/ stub/ opae/ xrt/(vortex.cpp) rtlsim/ simx/  ← 드라이버 (tinygpt 흔적 0건)
kernel/   include/ linker/ src/  ← 디바이스 측 런타임 API
sim/      simX/(cycle-approx) rtlsim/ opaesim/
```

### 2.3 제외 항목 (이름만 명시, 내부 분석 안 함)

- `third_party/fpnew` — PULP FPNew 부동소수점 IP (FPU가 `VX_fpu_fpnew.sv`로 래핑해 사용).
- `third_party/ramulator` — DRAM/HBM 타이밍 모델(simx에서 사용).
- `third_party/softfloat` — Berkeley SoftFloat (FPU DPI 참조 모델).
- `third_party/openc910`, `third_party/opene906` — T-Head RISC-V 코어(본 분석과 무관).
- `tests/opencl`의 **그 외 벤치마크**(vecadd, sgemm, dogfood 등) — Vortex 동봉 벤치마크. **단, `tinygptv1`/`tinygptv2`는 본 프로젝트 전용 커널이므로 분석 대상에 포함**.
- `.git`, build 산출물(`*.depend` 등 생성물).

---

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

이 프로젝트의 실질 기여는 **`tests/opencl/tinygptv1`과 `tinygptv2`** 두 디렉토리에 100% 집중되어 있다. 따라서 본 절은 Tiny-GPT 커널·호스트·학습 스크립트를 라인 단위로 분석하고, 그것이 Vortex의 어떤 RTL/런타임에 매핑되는지를 다룬다.

### 3.0 모델 아키텍처 (학습 스크립트 기준 — 실제 모델 정의)

`tinygptv2/scripts/train_model.py`(v1도 동일)가 모델의 진짜 정의다.

- 어휘 `VOCAB` 41개 토큰(L4-11), `VOCAB_SIZE=41`, `HIDDEN_DIM=16`, `LR=0.05`, `EPOCHS=1000`(L13-16).
- 파라미터 초기화(L66-71):
  - `embedding`: `[41 x 16]` FP32
  - `W1`: `[16 x 16]`, `b1`: `[16]`
  - `W2`: `[41 x 16]`, `b2`: `[41]`
- **순전파(L77-79)**: `h1 = tanh(W1 · x_embed + b1)` → `logits = W2 · h1 + b2`.
  → 즉 **임베딩 → 은닉층 1개(tanh) → 출력층** 의 **2-layer MLP**. **self-attention, multi-head, layernorm, positional encoding, KV-cache가 전혀 없다.** "GPT/Transformer"라는 명칭과 달리 구조적으로는 next-token bigram-ish MLP 분류기다.
- 학습: cross-entropy + 수기 backprop + SGD(L81-105). 학습 데이터는 31개의 고정 단어 시퀀스(L22-54)에서 (현재토큰→다음토큰) 쌍을 만든다(L57-61) — 문맥 길이 1의 마르코프 근사.
- export(L111-115): `embedding.npy / weights1.npy / bias1.npy / weights2.npy / bias2.npy` 5개를 호스트가 읽는다.

> **핵심 시사**: 추론 시 입력은 "직전 토큰 1개의 임베딩 벡터(길이 16)"뿐이다. 그래서 디바이스 커널의 핵심 연산은 **행렬-벡터 곱(GEMV) 2회**이며, GEMM/어텐션이 아니다.

### 3.1 Tiny-GPT v1 디바이스 커널 — `tinygptv1/kernel.cl`

전체 37줄, 단일 커널 `matvec2layer`(L1-12). 파라미터: `W1[HxH], B1[H], W2[VxH], B2[V], x[H], h1[H], out[V], H, V, phase`.

- **SIMT 매핑**: `int gid = get_global_id(0)`(L13) — **출력 1개 = work-item 1개**. 호스트가 phase 0은 `global=H(=16)`, phase 1은 `global=V(=41)`로 두 번 enqueue(main.cc L211-217).
- **phase 0 (은닉층, L15-26)**: `gid < H`인 work-item이 `W1`의 `gid`번째 행과 입력 `x`의 내적(L19-20)에 `B1[gid]`를 더하고(L21), `tanh`를 **`(1-e^{-2a})/(1+e^{-2a})` 수식으로 직접 계산**(L23-25). → 16개 work-item이 각자 16-길이 내적 = warp 내 16-lane SIMT로 정확히 맞아떨어짐.
- **phase 1 (출력층, L27-34)**: `gid < V`인 work-item이 `W2`의 `gid`번째 행과 `h1`의 내적(L31-32) + `B2[gid]`(L33). 41개 출력 → 41 work-item.
- 특징: `__local`/배리어/벡터화 없음. 가장 단순·이식성 높은 버전. 정밀도 전부 `float`(FP32).

### 3.2 Tiny-GPT v2 디바이스 커널 — `tinygptv2/kernel.cl` (핵심 확장)

172줄, "fused" 설계. 헤더에 `D_MAX 64`, `VOCAB_MAX 128`(L4-5)로 컴파일타임 상한.

#### 공용 헬퍼
- `dot_row_vec4`(L10-25): `__global` 가중치 행 × `__local` 입력 벡터를 **float4 SIMD로 4개씩 누산**(L17-22), 나머지는 스칼라 처리(L23). → DSP-friendly FMA 4-way.
- `reduce_max_local`(L27-31): softmax 수치안정용 max 리덕션.
- `softmax_sample_top1`(L33-43): `exp(logit-max)` 정규화 후 argmax(top-1) — **디바이스 측 샘플링**.

#### 커널 1: `tinygpt_persist_fused` (1-core 최적, L46-115) — 가장 중요
하나의 work-group이 **전체 토큰 루프 `for t in 0..T`(L70)를 디바이스 안에서 통째로 돈다**(persistent kernel). 호스트 재호출이 토큰마다 일어나지 않음.

- 가드: `H>D_MAX || V>VOCAB_MAX` 또는 work-group이 1개가 아니면 return(L57-59). 즉 단일 WG 전제.
- `__local x_loc / hidden_loc / logits_loc`(L64-66)에 활성/로짓을 온칩(local memory)에 상주.
- 루프 본문(토큰당 4단계):
  1. **Embed**(L72-80): lid==0이 `E + token*H` 행을 `vload4/vstore4`로 `x_loc`에 복사. → 임베딩 룩업.
  2. **hidden = tanh(W1·x + B1)**(L84-88): 행을 `for r=lid; r<H; r+=lsize`로 work-item에 분배(스트라이드 분할) → 각자 `dot_row_vec4` + bias + `fast_tanh`(=`tanh`, L8).
  3. **logits = W2·hidden + B2**(L92-104): vocab 행을 lid로 분배, float4 누산(L96-100).
  4. **softmax + top-1 샘플**(L108-112): lid==0이 `softmax_sample_top1`로 다음 토큰 결정, `io_tokens[t+1]`에 기록하고 `token` 갱신.
  - 각 단계 사이 `barrier(CLK_LOCAL_MEM_FENCE)`(L81/89/105/113)로 work-item 동기화.
- **SIMT 매핑**: lsize(=16) work-item = warp lane들. 행/로짓을 lane에 스트라이드 분배 → GEMV의 출력 차원을 SIMT 병렬화. local memory로 중간 활성 재사용.

#### 커널 2: `ffn_to_logits_slice` (멀티코어 스케일, L118-171)
토큰당 1회 launch하되 **vocab(V)을 work-group(=core) 수 G로 분할**.

- `G=get_num_groups, gid=get_group_id`(L127-128).
- Embed + hidden 계산은 모든 WG가 중복 수행(L137-152) — hidden(16)이 작아 중복 비용이 작다는 설계 판단(추정).
- **vocab 슬라이스**(L154-170): `chunk=(V+G-1)/G`, `[begin,end)` 구간만 각 WG가 담당(L155-157), float4 누산으로 `logits[v]` 기록. 디바이스는 logits만 내보내고 **softmax/샘플링은 호스트가 수행**(persist 커널과의 핵심 차이).
- **SIMT 매핑**: WG↔코어, lane↔vocab 행. V=41이 작아 4-core면 코어당 ~11 vocab → 부하 미세, 멀티코어 스케일링 시연용.

### 3.3 호스트 진입점 — `tinygptv2/main.cc` (런타임 흐름)

표준 OpenCL 1.2 호스트(`CL_TARGET_OPENCL_VERSION 120`, L2). 이 호스트가 **Vortex의 OpenCL 런타임(POCL/XRT 드라이버)** 위에서 돈다.

- 모델 차원 하드코딩 `VOCAB_SIZE 41 / HIDDEN_DIM 16`(L18-19).
- **가중치 로드**(L106-116): `cnpy::npy_load`로 5개 .npy를 읽어 `std::vector<float>`로 평탄화. **전부 float — 양자화/int8 없음**(중요).
- **플랫폼/디바이스**(L119-127): `CL_DEVICE_TYPE_ACCELERATOR` 우선, 실패 시 `DEFAULT`. → FPGA(XRT)에서는 ACCELERATOR, simx/pocl에서는 DEFAULT로 잡힘.
- **커널 빌드**(L130-146): `kernel.cl` 파일을 읽어 `clCreateProgramWithSource` + `clBuildProgram` 옵션 `-cl-std=CL1.2 -cl-fast-relaxed-math`(L138). 두 커널 핸들 생성(L148-149).
- **버퍼**(L152-156): W1/B1/W2/B2/E를 `CL_MEM_READ_ONLY|COPY_HOST_PTR`로 디바이스 글로벌 메모리에 올림.
- **프롬프트**(L161-168): 표준입력에서 단어들을 받아 **마지막 단어**만 토큰화(L167-168) — 문맥 1 토큰만 사용(모델 구조와 일치).
- **엔진 분기**:
  - `persist`(L174-210): `io_tokens` 버퍼 1개에 prompt 토큰 넣고 `tinygpt_persist_fused`를 **딱 1회 launch**(global=local=16, L193-200), 디바이스가 전체 시퀀스를 채워 돌려줌(L203). 호스트 후처리 없음.
  - `slice`(L211-256): 토큰마다 `ffn_to_logits_slice`를 launch(global=local×groups, L220), logits 읽어와(L241) **호스트에서** repetition penalty(L244-246) → `softmax_host`(temperature, L248) → `topk_sample_host`(L249) 적용. 단, `topk_sample_host`는 정렬 후 **top-1 결정적 반환**(L74-76)이라 실제론 greedy.
- **호스트 샘플링**(L63-76): `softmax_host`(max-shift + temperature), `topk_sample_host`(partial_sort 후 idx[0]). temperature/top-k/penalty 인자는 받지만 v2 호스트 topk는 결정적.

#### v1 호스트와의 차이 — `tinygptv1/main.cc`
- `CL_TARGET_OPENCL_VERSION 300`(L2), POCL 플랫폼 명시 선택(`pick_pocl_platform`, L42-59).
- 토큰 루프가 **항상 호스트 주도**(L207-233): 매 스텝 임베딩을 호스트가 만들어 `x_buf`에 write(L208-209) → phase0 launch(global=H) → phase1 launch(global=V)(L211-217) → out 읽기(L221) → penalty(L223) → `softmax`(L225) → `sample_top_k`(L227).
- `sample_top_k`(L80-94)는 **`std::discrete_distribution`로 확률적 샘플링**(L90-92) — v2의 결정적 top-1과 달리 v1은 진짜 stochastic top-k. (분석상 v1이 샘플링 다양성 측면에서 더 "GPT스럽다".)
- `VX_FAST_EXIT` 환경변수로 atexit/소멸자 우회 `_exit(0)`(L237-241) — FPGA(XRT) 종료 시 double-free 회피용 실전 팁.

### 3.4 빌드 글루 — `Makefile` (v1·v2 동일 패턴)
- `PROJECT := tinygpt`(L4), `SRCS := main.cc cnpy.cpp`(L8), `kernel.cl`을 SRC에서 복사(L11-12), `LDFLAGS += -lz`(zlib, npy 압축 해제용, L18), `OPTS ?= -n32`. `../common.mk` 포함으로 Vortex 공통 빌드 흐름 재사용.
- 주의: 두 Makefile 모두 `PROJECT := tinygpt`로 동일하고 `SRC_DIR=$(VORTEX_HOME)/tests/opencl/tinygpt`를 가리킨다 → **빌드 시 `tinygpt`라는 정규 디렉토리를 기대**. 실제 repo엔 `tinygptv1`/`tinygptv2`만 존재(`tests/opencl/tinygpt` 커널 검색 0건). 즉 사용자는 v1 또는 v2를 `tinygpt`로 복사/심링크해 빌드하는 흐름으로 **추정**(확인 불가).

### 3.5 자체 RTL/런타임 수정 여부 (결론: 없음 = Vortex 베이스 그대로)
- `runtime/` 전체에서 `tinygpt` 문자열 검색 0건 → **런타임은 미수정 Vortex 드라이버**(`runtime/xrt/vortex.cpp` 등)를 그대로 사용. Tiny-GPT는 표준 OpenCL 앱일 뿐 런타임 패치 불필요.
- `hw/rtl/core`·`hw/rtl/fpu`는 표준 Vortex 모듈(`VX_core`, `VX_schedule`, `VX_alu_*`, `VX_lsu_*`, `VX_fpu_fpnew/fma/dsp/...`)이며 Tiny-GPT 전용 RTL 변경 흔적 없음. FP32 matvec의 곱셈-누산은 `VX_fpu_fma.sv`(FMA)·`VX_fpu_fpnew.sv`(FPNew 래퍼)·`VX_fpu_dsp.sv`(DSP 매핑)를 통해 실행됨. RV32IMA**F**의 F 확장이 활성화되어야 커널의 `float` 연산이 하드웨어 FPU로 내려간다(README L34 "RV32IMAF").
- `ci/`에도 tinygpt 문자열 0건 → CI에 Tiny-GPT 테스트 미등록(README 예제 명령으로 수동 실행).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 SIMT 실행 모델 (Vortex)
- 계층: cluster → core → warp → thread(lane). `blackbox.sh --clusters/--cores/--warps/--threads`로 구성(simulation.md L19-22). 기본 예시는 1 cluster, 1~4 core, 4 warp, 4 thread.
- OpenCL work-item↔SIMT thread(lane), work-group↔core에 매핑됨(Vortex 일반 규칙).

### 4.2 GPT 토큰 추론 파이프라인
1. (오프라인) Python 학습 → 5개 .npy export(§3.0).
2. 호스트가 .npy를 FP32로 로드 → 디바이스 글로벌 메모리 버퍼 생성.
3. 프롬프트의 **마지막 단어 1개**를 시작 토큰으로.
4. 토큰 t마다: **임베딩 룩업(E행)** → **GEMV1 + tanh** → **GEMV2(logits)** → **softmax → 샘플**.
   - v1: 매 토큰 호스트↔디바이스 왕복 2회 launch + 호스트 stochastic top-k.
   - v2 persist: 디바이스가 전 토큰 루프를 한 번에(왕복 최소화) + 디바이스 top-1.
   - v2 slice: 토큰당 1 launch, vocab을 코어로 분할 + 호스트 softmax/topk.
5. T개 토큰까지 반복, 토큰을 어휘로 디코드해 출력.

### 4.3 메모리 계층
- 가중치/임베딩 → 디바이스 글로벌 메모리(FPGA에선 HBM/DDR, simx에선 ramulator 모델). U280은 HBM2 8GB.
- v2는 중간 활성(`x_loc/hidden_loc/logits_loc`)을 **local memory(온칩)** 에 두고 배리어로 재사용 → 글로벌 트래픽 절감(L64-66). v1은 `h1`을 글로벌 버퍼로 둠(왕복 많음).
- L1/L2/L3 캐시는 Vortex 구성 옵션(`--l2cache/--l3cache`).

### 4.4 FPU / 부동소수점
- 연산 전부 **FP32**. 디바이스 커널의 `float`/`float4` MAC는 RV-F FPU(FMA)로 내려감. v2 `dot_row_vec4`의 float4 누산은 4-way FMA로 DSP에 친화적.
- `-cl-fast-relaxed-math`로 컴파일(v2) → 정밀도 완화 대신 속도.

### 4.5 양자화 여부 — **없음 (확정)**
- 학습 스크립트가 `np.float32`로 저장(train_model.py L67-71), 호스트가 `data<float>()`로 로드, 커널이 `float`로 연산. "int8"/"quant"는 어휘 토큰(main.cc L53-54)일 뿐. **INT8/양자화 추론 경로는 코드상 존재하지 않음.**

---

## 5. HW/SW 매핑 (OpenCL/C ↔ C++ 런타임 ↔ Vortex RTL ↔ U280)

| 계층 | 구성요소 | 자체 vs 베이스 |
|---|---|---|
| 모델 정의/학습 | `scripts/train_model.py` (NumPy 2-layer MLP) | **자체** |
| 디바이스 커널 | `kernel.cl` (matvec2layer / persist_fused / slice) | **자체** |
| 호스트 앱 | `main.cc` (OpenCL 1.2 host loop, 샘플링) | **자체** |
| .npy 로더 | `cnpy.cpp/.h` | 오픈소스 차용(동봉) |
| OpenCL 런타임/컴파일 | POCL + Vortex `kernel/` API | Vortex 베이스 |
| 디바이스 드라이버 | `runtime/{simx,rtlsim,xrt}` (`vortex.cpp`) | Vortex 베이스 (미수정) |
| 시뮬레이터 | `sim/simX`, `sim/rtlsim` | Vortex 베이스 |
| RTL 코어 | `hw/rtl/core/VX_*` (SIMT 파이프라인) | Vortex 베이스 |
| RTL FPU | `hw/rtl/fpu/VX_fpu_fpnew/fma/dsp` (FP32 MAC) | Vortex 베이스 (FPNew=third_party) |
| 합성/비트스트림 | `hw/syn/xilinx/xrt` (PLATFORM=u280...) | **U280 구성 = 자체 조정** |
| 보드 | AMD Alveo U280 (HBM2, XDMA) | 타깃 HW |

흐름: `blackbox.sh`가 driver(simx/rtlsim/xrt) 선택 → 해당 `runtime/<driver>` 빌드 → 호스트 `main.cc` 실행 → OpenCL로 `kernel.cl`을 Vortex 디바이스에 올려 SIMT 실행 → (xrt) U280 비트스트림(`vortex_afu.xclbin`) 상에서 동작.

---

## 6. 빌드·실행 방법 (docs/README 근거)

### 시뮬레이션(simx)
```sh
# 빌드
cd tests/opencl/tinygptv2 && make clean && make      # (또는 tinygptv1)
# 실행 (README L125-143)
./ci/blackbox.sh --clusters=1 --cores=1 --warps=4 --threads=4 \
  --driver=simx --app=tinygptv2 \
  --args="-engine persist -tokens 15" --perf=1
# 멀티코어 slice
./ci/blackbox.sh --clusters=1 --cores=4 --warps=4 --threads=4 \
  --driver=simx --app=tinygptv2 \
  --args="-engine slice -groups 4 -tokens 15 -temp 0.8 -topk 5 -penalty 1.1" --perf=1
```
- 실행 시 표준입력으로 프롬프트(어휘 내 단어) 입력 필요(main.cc L162-163).
- `--perf=1`로 코어별 instrs/cycles/IPC 출력(simulation.md L45-49 형식).

### FPGA(U280)
```sh
# 1) 비트스트림 합성 (xilinx_fpga_guide.md L19-24)
cd hw/syn/xilinx/xrt
PREFIX=test1 PLATFORM=xilinx_u280_gen3x16_xdma_1_202211_1 TARGET=hw NUM_CORES=1 make
#   → <BUILD_DIR>/bin/vortex_afu.xclbin
# 2) XRT 런타임 빌드
make -C runtime/xrt clean && TARGET=hw make -C runtime/xrt
# 3) 실행 (README L150)
VX_FAST_EXIT=1 TARGET=hw FPGA_BIN_DIR=<BUILD_DIR>/bin \
  ./ci/blackbox.sh --cores=1 --driver=xrt --app=tinygpt \
  --args="-temp 0.7 -topk 5 -penalty 1.1 -tokens 15"
```
- XRT 환경: `source .../Vitis/2023.1/settings64.sh`, `source /opt/xilinx/xrt/setup.sh`(guide L4-6).
- **주의**: FPGA 실행은 `--app=tinygpt`(정규명)를 쓰므로 §3.4의 디렉토리 복사 이슈가 적용됨(추정).

---

## 7. 의존성

- **Vortex 툴체인**: POCL(OpenCL), LLVM, RISCV-GNU-Toolchain(RV32/64), Verilator, Yosys, Sv2v(README L65-73).
- **FPGA/Xilinx**: Vitis 2023.1, XRT, U280 플랫폼 `xilinx_u280_gen3x16_xdma_1_202211_1`.
- **앱 의존**: zlib(`-lz`, npy 압축), cnpy(.npy 로더, 동봉), NumPy(학습).
- **제외(이름만)**: third_party의 fpnew(FPU IP), softfloat(FP 참조), ramulator(DRAM 모델), openc910/opene906(타 RISC-V 코어).

---

## 8. 강점 / 한계 / 리스크

### 강점
- **재현 가능한 오픈 AI/HW 스택 시연**: NumPy 학습 → .npy → OpenCL → simx/RTL/FPGA(U280)까지 end-to-end가 한 repo에. 교육·온보딩용으로 명확.
- **3가지 실행 전략 비교**: v1(호스트 주도 2-phase), v2 persist(디바이스 융합 루프), v2 slice(멀티코어 vocab 분할) — 동일 워크로드에서 SIMT 매핑 전략을 비교할 수 있는 좋은 케이스 스터디.
- **온칩 재사용·SIMD 활용**: v2의 local memory + float4 누산 + persistent kernel은 GPGPU 상 GEMV 최적화의 정석을 보여줌.
- **실전 디테일**: `VX_FAST_EXIT`(FPGA 종료 안정화), 빌드 옵션 등 실제 보드 경험 반영.

### 한계
- **사실상 GPT/Transformer가 아님**: 어텐션·layernorm·positional·멀티레이어·KV-cache 전무. 2-layer MLP next-token 분류기. 모델 표현력이 toy 수준(어휘 41, hidden 16, 학습 시퀀스 31개).
- **양자화/int8 없음**: 명칭과 달리 전부 FP32.
- **워크로드가 작아 가속 효과 미미**: GEMV 16×16, 41×16 규모는 메모리/런치 오버헤드가 지배적. 멀티코어 slice는 V=41로 코어당 부하가 너무 작아 스케일링 시연용에 가까움.
- **문맥 길이 1**: 프롬프트의 마지막 단어만 사용 → 진짜 언어모델 동작과 거리.
- **문서 정합성**: docs/simulation.md가 여전히 Arria10 기준 등 upstream 잔재.

### 리스크
- 빌드 시 `tinygpt`(정규명) vs `tinygptv1/v2` 디렉토리 불일치로 `make`가 바로 안 될 수 있음(수동 복사 필요, 추정).
- v2 `slice` 엔진의 `topk_sample_host`가 top-1 결정적이라 temperature/top-k 인자가 실효 없음(샘플링 다양성 무효).

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

### 9.1 GPGPU 기반 Transformer 추론 vs 전용 가속기
- **본 프로젝트의 위치**: "유연성(임의 OpenCL 커널)·재현성"을 얻는 대신 **처리량/효율은 전용 가속기에 크게 못 미침**. GEMV/GEMM을 SIMT lane에 분배하는 방식은 systolic/output-stationary MAC array(HG-PIPE류)가 주는 데이터 재사용·파이프라인 깊이를 따라갈 수 없다. 우리 XR 시선추적(저지연·고프레임)에는 GPGPU 경로는 부적합하고 전용 데이터패스가 정답이라는 점을 **반례로 확증**해 주는 자료.
- **그러나 "유연한 프론트엔드"로서의 가치**: 우리 전용 가속기와 호스트 사이 토큰 루프/샘플링/스케줄링 같은 **제어부**는 본 repo의 호스트 패턴(엔진 분기, persistent vs per-step launch)을 차용할 수 있다.

### 9.2 재사용 가능한 아이디어 (구체)
1. **Persistent-kernel 루프 패턴(v2)**: 토큰 루프를 디바이스(또는 가속기 FSM) 내부로 끌어들여 호스트 왕복을 없애는 설계. 우리 ViT 디코더/디톤화 루프에서 호스트-가속기 핸드셰이크 횟수를 줄이는 마이크로아키 모티프로 직접 차용 가능(특히 XR 실시간 추론의 per-frame 오버헤드 절감).
2. **출력차원 슬라이싱(slice 엔진)**: vocab/출력 채널을 PE 그룹에 분할(`chunk=(V+G-1)/G`)하는 깔끔한 부하분할식 — 우리 멀티-PE 타일 가속기의 classifier head 분할에 그대로 응용.
3. **온칩 활성 재사용 + float4 MAC**: `local memory + 4-way 누산` 구조는 우리 HLS/RTL에서 BRAM/URAM 활성 버퍼 + DSP 4-lane MAC 패턴으로 1:1 변환 가능. 본 코드는 그 SW 레퍼런스로 유용.
4. **end-to-end 검증 하네스**: Python(골든) → .npy → 디바이스 커널 결과 비교 흐름은 우리 algo2fpga 류 co-sim(파이썬 vs RTL) 검증 파이프라인의 경량 템플릿으로 재사용 가능.
5. **샘플링/후처리 분리**: softmax/top-k/repetition penalty를 호스트에 두는 분리(v2 slice)는, 우리 가속기가 무거운 GEMM만 맡고 가벼운 비선형/샘플링은 호스트/소프트 코어에 두는 분업 설계의 선례.

### 9.3 차별화 포인트(우리가 더 잘할 부분)
- **진짜 어텐션/layernorm/softmax 데이터패스**를 전용 reduction 회로로 구현(LUT 기반 exp/gelu, 스트리밍 softmax) → 본 repo가 비워둔 지점.
- **양자화(INT8/INT4) 실재 구현** → 본 repo는 명칭만 양자화. 우리 ViT-Quantization 자산과 결합 시 명확한 우위.
- **고처리량 파이프라인(HG-PIPE)**: GEMV가 아닌 GEMM/conv를 깊은 파이프라인으로 처리해 utilization을 끌어올림.

---

## 10. 근거 / 한계 표기 (Vortex 베이스 vs 자체추가 구분)

### 자체추가로 **확인된** 범위 (라인 근거 있음)
- `tests/opencl/tinygptv1/{kernel.cl, main.cc, Makefile, scripts/train_model.py}` — Tiny-GPT v1 전부.
- `tests/opencl/tinygptv2/{kernel.cl, main.cc, Makefile, scripts/train_model.py}` — Tiny-GPT v2 전부.
- README의 Tiny-GPT 빌드/실행 섹션, `docs/xilinx_fpga_guide.md`의 U280/U50 절차 일부.
- (보조) `cnpy.cpp/.h`는 오픈소스 cnpy를 동봉한 것으로, 본 팀 순수 창작은 아님.

### Vortex 베이스로 **확인된** 범위 (미수정)
- `runtime/*` 전체(`tinygpt` 문자열 0건), `sim/*`, `hw/rtl/{core,fpu,cache,mem}`(표준 `VX_*` 모듈), `ci/*`(tinygpt 0건), `docs/{codebase.md, simulation.md}`(upstream 잔재 — simulation.md는 기본 FPGA가 Arria10로 남아 있음).

### "추정" / "확인 불가"
- **추정**: 빌드 시 `tinygpt` 정규 디렉토리는 v1/v2 중 하나를 복사/심링크해 만드는 흐름(Makefile이 `PROJECT:=tinygpt`, `tests/opencl/tinygpt`를 가리키나 해당 커널 디렉토리 부재).
- **추정**: v2 slice 커널이 모든 WG에서 embed+hidden을 중복 계산하는 것은 hidden=16이 작다는 비용 판단.
- **확인 불가**: 실제 U280 비트스트림/성능 수치(instrs/cycles/IPC) — 본 repo 소스에는 합성 결과·로그가 동봉되어 있지 않음(README는 명령만 제시). 대회 영상/외부 결과로만 존재.
- **확인 불가**: `hw/syn/xilinx/xrt`에 U280 전용 핀/플랫폼 커스터마이즈가 어디까지 자체 수정인지 — README는 "U280 configs"를 언급하나, 본 분석에서 합성 스크립트 차이를 라인 단위로 대조하진 않음(상위 README/test tcl만 U280 토큰 확인).

### 한 줄 결론
**이 repo의 실질 기여는 `tests/opencl/tinygptv1`·`tinygptv2`에 있는 2-layer MLP(자칭 Tiny-GPT)의 OpenCL 커널·호스트·학습 스크립트뿐이며, 코어/캐시/FPU/런타임은 미수정 Vortex 베이스다. 어텐션도 양자화도 없는 toy 규모지만, persistent-kernel 루프·출력차원 슬라이싱·온칩 float4 MAC·end-to-end 골든 검증 패턴은 우리 전용 가속기 설계에 재사용 가치가 있다.**
