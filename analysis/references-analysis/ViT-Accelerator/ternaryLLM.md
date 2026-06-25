# ternaryLLM 정밀 분석

> 대상 repo: `REF/ViT-Accelerator/ternaryLLM`
> 분석 도구: Glob/Grep/Read 기반 라인 단위 직접 확인. third-party/vendor/생성물 제외.
> 작성일: 2026-06-20

---

## 1. 개요

- **목적**: 삼진(ternary, {-1, 0, +1}) 가중치 LLM 추론을 CPU/GPU/FPGA 세 플랫폼에서 가속하기 위한 통합 연구 코드베이스. 삼진 가중치는 곱셈을 단순 덧셈/뺄셈으로 치환할 수 있고, 50~90% 희소성을 가지므로, 이를 활용하는 전용 희소 GEMM(SpMM) 알고리즘과 압축 포맷(TCSC, Ternary Compressed Sparse Column)을 핵심으로 한다.
- **한줄요약**: "삼진 가중치의 부호(+/-)별 비제로 행 인덱스만 저장(TCSC)하고, 입력 X를 인덱싱하여 누산 덧셈/뺄셈으로 GEMM을 수행"하는 알고리즘을, AVX2/AVX-512(CPU) · CUDA warp/tile 커널(GPU) · SpinalHDL PE 어레이(FPGA)로 각각 구현한 것.
- **원논문 추정**: 루트 `README.md` L8-11, L22-28 근거.
  - **SSR: Sparse Segment Reduction for Ternary GEMM Acceleration** (DATE 2026, Pittet/Zhu/Verdan/Alonso) — 단, SSR 폴더는 코드 없이 README만 존재(아래 §10 참고).
  - **Fast Ternary LLM Inference with Addition-Based Sparse GEMM on Edge Devices** — ternaryLLM_CPU 대응 (추정).
  - **An Accelerator for Ternary Language Models based on FPGA** — ternaryLLM_FPGA 대응 (석사 thesis 추정, FPGA README L3 "this thesis/repository").
  - 기여자(루트 README L14-17): SSR=Pittet/Verdan/Zhu, CPU=Kjoseva/Zhu, GPU=Fu(`fuguan@ethz.ch`), FPGA=Giacone. 소속은 ETH Zurich(Coyote/HACC 사용)로 추정.
- **타깃 디바이스**:
  - CPU: AMD Ryzen 7 8845HS / Intel Core i9-11900H (AVX2 + AVX-512), Windows 11/10, MSVC 2022 (ternaryLLM_CPU/README L15-17).
  - GPU: NVIDIA CUDA 12.9, PyTorch CUDA extension (setup.py L6).
  - FPGA: **Xilinx Alveo U55C**, ETH **Coyote** shell, HACC 클러스터 (ternaryLLM_FPGA/README L5).

---

## 2. 디렉토리 구조

### 자체 소스 트리 (분석 대상)

```
ternaryLLM/
├── README.md                          # 전체 개요 + 인용
├── ternaryLLM_CPU/                     # C++ INT8/FP32 GEMM, TCSC
│   ├── README.md
│   └── src/
│       ├── TCSC.hpp                    # ★ TCSC/MergedTCSC/Grouped TCSC 포맷 변환 + naive sparseGEMM
│       ├── GEMM_CPU_INT8.cpp/.hpp      # ★ AVX2/AVX-512 INT8 커널 (Merged GroupMin / Uniform)
│       ├── GEMM_CPU_FP32.cpp/.hpp      # FP32 baseline/naive/unroll/AVX 커널
│       ├── LlamaModel.hpp              # libtorch Llama 레이어 + TernaryMLP<int32_t> 벤치 래퍼
│       ├── initData.cpp/.hpp           # X/W 초기화(랜덤 삼진 weight 생성)
│       ├── main.cpp                    # 벤치마크 드라이버
│       └── SIMD_Generator.ipynb        # SIMD 코드 자동 생성(노트북, 미독)
├── ternaryLLM_GPU/                     # PyTorch + CUDA ter_spmm 커널
│   ├── README.md, setup.py             # CUDAExtension 빌드
│   ├── csrc/
│   │   ├── ter_spmm_kernel.cuh         # ★ spmv/spmm CUDA 커널 + dispatcher
│   │   ├── ter_spmm_wrapper.cu         # ★ TCSC 변환(host) + pybind11 모듈
│   │   └── utils.cuh                   # 디스패치 매크로, 벡터타입, warp_reduce
│   ├── TernaryLLM/                     # nn.Module 레이어
│   │   ├── TernaryLinear.py            # ★ ter_spmm 호출 Linear + weight 생성/변환
│   │   ├── TernaryCSC.py               # 순수 파이썬 참조 TCSC 변환
│   │   ├── TernaryMLP.py / TernaryAttn.py / TernaryLlama.py
│   │   ├── configuration_ternary.py    # LlamaConfig 확장 (sparsity 등)
│   │   └── __init__.py
│   └── benchmark*.py, benchmark/       # 벤치마크
├── ternaryLLM_FPGA/                    # SpinalHDL/Scala Ternary GEMM 가속기 + Coyote
│   ├── README.md, build.sc, build.sbt  # Mill/sbt 빌드
│   ├── hw/spinal/gemmacc/
│   │   ├── src/
│   │   │   ├── Config.scala            # ★ ConfigSys: AXI/유닛 파라미터 (S=64, K_slice=128, UNROLL_M=4)
│   │   │   ├── Generator.scala         # ★ Verilog 생성 엔트리(TernaryGEMM/PE/TestLogic)
│   │   │   └── design/
│   │   │       ├── PE.scala            # ★ Processing Element (인덱싱 누산)
│   │   │       ├── DataFSM.scala       # ★ 데이터 이동 FSM(AXI burst, BRAM buffer)
│   │   │       ├── TopLevel.scala      # ★ PE 어레이 + FSM 결선
│   │   │       └── WrapGEMM.scala      # ★ AxiLite4 컨트롤 레지스터 래퍼
│   │   ├── coyote/                     # Coyote 인터페이스(AXI4SR, ReqT, AxiCoyote) — 통합용
│   │   ├── util/                       # AxiMemorySim, InputAndCheck (시뮬용)
│   │   ├── sim/ternaryGEMM/TopLevelSim.scala  # 시뮬레이션 testbench
│   │   └── src/old/...                 # 구버전 (Base / AXI_version / coyote_v2) — 미분석
│   ├── sw/gemmacc/main.cpp             # Coyote 호스트 드라이버 코드
│   └── coyote_files/                   # Coyote shell 통합 산출물(.bit 포함) — 미분석
└── SSR/
    └── README.md                       # ★ 코드 없음 (README만)
```

### 제외 항목 (이름만 언급)

- `ternaryLLM_FPGA/coyote_files/` — ETH Coyote shell 통합 파일(`cyt_top.bit` 비트스트림, CMakeLists, vfpga_top.svh). 자체코드 아님.
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/` — 구버전 RTL(Base/AXI_version/coyote_v2). 한 줄 언급만: `coyote_v2/PE.scala`는 현재 design/PE.scala의 직전 버전(enable_X 8bit, done_acc 신호 보유, L16-20)으로 현재는 미사용.
- `ternaryLLM_FPGA/hw/spinal/MatrixAdd/` — 테스트용 로직(Wrap_logic/logic), 본 GEMM과 무관.
- `*.bit`(비트스트림), `*.odt/.ods`(벤치 결과 문서), `*.ipynb`(SIMD_Generator/Benchmark) — 생성물/문서.
- libtorch / Eigen / transformers / pybind11 — 외부 라이브러리.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 CPU: `ternaryLLM_CPU/src/TCSC.hpp` (TCSC 포맷의 정의)

이 파일이 전체 repo의 알고리즘적 핵심이다. 삼진 행렬을 "부호별 비제로 행 인덱스 + 컬럼 오프셋"으로 변환하는 4개 클래스/함수를 정의한다.

**(a) `class SparseFormat` (L6-34) — 기본(naive) TCSC**
- 입력: `const int8_t* matrix`(K행×N열, row-major: `matrix[k*N+n]`), K, N (L13 주석 "K ROWS, N COLUMNS").
- 멤버(L8-11): `col_start_pos`, `col_start_neg`(컬럼별 누적 시작 오프셋, CSC의 col_ptr 역할), `row_index_pos`, `row_index_neg`(+1/-1 값을 가진 행 인덱스, int16_t).
- 알고리즘(L17-30): 각 컬럼 n에 대해 모든 행 k를 순회하며 `>= 1`이면 pos 인덱스에, `<= -1`이면 neg 인덱스에 행 번호 push. 값 자체(±1)는 저장하지 않음 — 부호는 pos/neg 배열 소속으로 암묵 표현. **이것이 표준 CSC 대비 TCSC의 절약점**: 값 배열이 불필요.
- 즉 가중치 W를 **양(+1) 인덱스 리스트와 음(-1) 인덱스 리스트로 분리** 저장.

**(b) `class MergedTCSC` (L36-90) — pos/neg 병합(interleaving) 단일 컬럼**
- 입력: `SparseFormat naiveTCSC`, K, N.
- 멤버(L38-39): `metadata`(컬럼당 4개: align_start, align_end, remain_end, remain_value), `row_index`(병합 인덱스).
- 알고리즘(L48-88): 컬럼별로 pos 개수(`col_pos`)와 neg 개수(`col_neg`)를 비교. 둘 중 작은 쪽 길이만큼 **pos/neg 인덱스를 번갈아(interleave) 배치**(L58-63 또는 L72-77) → 이 구간은 "+X[pos] - X[neg]" 쌍 연산으로 한 번에 처리 가능. 나머지(remainder)는 더 많은 쪽 부호로만 채우고(L64-67/78-81), `remain_val`(+1 또는 -1, L54-56/70)에 그 부호를 기록. 같은 개수면 remain_val=0(L55-57).
- 목적: 덧셈/뺄셈을 쌍으로 묶어 SIMD/벡터화 효율을 높이고, 잔여항은 부호 한 개로 일괄 처리.

**(c) `class MergedTCSC_Group` (L92-409) — 그룹 단위 병합 (가장 정교)**
- 생성자 2개: 일반(L99), SIMD_SIZE 인자 포함(L248). 핵심 파라미터: `group_size`, `group_method`("min"/"mid"/"max"/기타), `interleaved`, (옵션)`SIMD_SIZE`.
- 그룹 내 컬럼들을 묶어 정렬 정도를 3가지로 측정(L116-118):
  - `align_min` = 그룹 내 `min(pos,neg)`의 **최솟값** (모든 컬럼이 공통으로 가지는 interleave 길이).
  - `align_mid` = 그룹 내 `min(pos,neg)`의 **최댓값**.
  - `align_max` = 그룹 내 `max(pos,neg)`의 최댓값.
- **"min" 그룹(L146-174)**: 공통 align_min 만큼만 그룹 인터리브로 묶고, 각 컬럼의 잔여(pos/neg 각각)는 별도 루프로 처리 → metadata에 컬럼별 `remain_end_pos`, `remain_end_neg` 기록.
- **"mid" 그룹(L175-219)**: align_min~align_mid 구간에서 짧은 컬럼은 "같은 인덱스를 두 번 push"(L184-186, `a-a=0` 트릭)하여 **0을 만들어 정렬을 맞춤**. 이렇게 하면 잔여 루프가 컬럼당 1개로 줄어듦. metadata에 `remain_end`, `remain_val` 저장.
- **"max" 그룹(L220-238)**: semi-structured 희소성. 각 컬럼이 동일 개수 +1/-1을 갖도록 정렬, 잔여 루프 없음.
- **기타(else, L239-243)**: 완전 정렬(uniform TCSC). group_size==N으로 호출하며 metadata에 컬럼당 비제로 수(`align_min*2`)만 기록.
- SIMD 버전(L248-408): `interleaved` 시 SIMD_SIZE 보폭으로 컬럼 인덱스를 묶어 배치(L280-307) → AVX 레지스터 폭에 맞춘 메모리 레이아웃.
- **의의**: CPU 가속의 핵심 트레이드오프(잔여 처리 분기 수 vs 더미 0 저장 오버헤드)를 그룹 정렬 정책으로 조절. GPU의 tiled-MCSC, FPGA의 K_slice 균일화와 같은 사상.

**(d) `template sparseGEMM` (L413-430) — 참조용 naive 희소 GEMM**
- 입력: X(M×K), `SparseFormat W`, bias, Y(M×N), M/N/K.
- 알고리즘(L415-428): m,n 이중 루프. 컬럼 n의 pos 인덱스 구간을 순회하며 `y += X[m*K + row_index_pos[k]]`(L420-422), neg 구간은 `y -= X[...]`(L424-426). **곱셈 0개** — 전부 인덱싱 + 덧셈/뺄셈. `#pragma omp parallel for`(L417) + `#pragma omp simd`(L419,423).
- 이것이 "Addition-Based Sparse GEMM"의 정수: ±1 가중치이므로 X 원소를 부호에 따라 누산만 하면 됨.

### 3.2 CPU: `GEMM_CPU_INT8.cpp/.hpp` (AVX 가속 커널)

- 헤더(`.hpp` L11-29)는 함수 시그니처 카탈로그. 명명 규칙: `..._TCSC_{Merged_GroupMin|Uniform}_{MxGn}_{AVX2|AVX512}_OpenMP`. M=행 타일(64/128), Gn=그룹 크기(G1/G4/G8/G16/G32).
- **`..._Merged_GroupMin_64xG4_AVX2_OpenMP` (.cpp L3-84)**:
  - 입력: X(int8), metadata(int32, MergedTCSC_Group "min"의 출력), row_index(int16), result(int8), M_ROW/N_COL/K.
  - 구조(L4-7): `#pragma omp parallel for`로 컬럼 그룹(N/4) 분배, 내부에서 M_ROW를 64씩 타일링. 4개 컬럼 × (상위32/하위32) = 8개 `__m256i` 누산기(res00..res31, L8-15).
  - **인터리브 구간(L16-41)**: metadata `groupData[0]~[1]`. row_index에서 8개씩(pos4+neg4) 인덱스를 읽어 `X + row_index[k]*M_ROW + i`에서 256bit 로드 후 `_mm256_add_epi8(res, _mm256_sub_epi8(pos, neg))` — **pos는 더하고 neg는 빼는 융합 연산**.
  - **잔여 구간(L42-73)**: 컬럼별로 pos 잔여(add)와 neg 잔여(sub)를 metadata 경계(`groupData[1..9]`)로 순회.
  - 결과 store(L74-81). column-major 출력(`result + col*M_ROW + i`).
- **`..._Uniform_64xG4_AVX2` (L86-134)**: metadata 불필요. 컬럼당 비제로 수가 **균일(`NonZeroPerCol`)**하다고 가정 → 잔여 루프 자체가 없음(L98 단일 루프), 가장 단순/빠름. 인덱스 범위 `[j*4*NonZeroPerCol, (j+1)*4*NonZeroPerCol)`.
- **AVX-512 버전(L136-164)**: `__m512i`로 폭 2배, 누산기 절반(res00/10/20/30). 64개 int8을 한 레지스터에 처리.
- 핵심: 모든 연산이 8비트 정수 add/sub. **곱셈기 없음**. 메모리 접근 패턴은 `row_index`로 X를 gather하는 형태이나, X가 column-major(`*M_ROW`)로 저장돼 연속 256/512bit 로드가 가능하도록 설계.

### 3.3 CPU: `GEMM_CPU_FP32.cpp` (FP32 baseline/참조)

- `..._Direct_OpenMP`(L8-18): 표준 dense GEMM, 비교 baseline(곱셈 포함).
- `..._TCSC_Naive`(L22-46): pos는 `res +=`, neg는 `res -=`. 별도 pos/neg col_ptr·row_ind 사용(병합 전 형태).
- `..._TCSC_Naive_oneFor`(L48-79): `min(neg,pos)` 만큼 pos/neg를 한 루프에서 동시 처리 후 잔여 분리 — MergedTCSC의 FP32 버전.
- `..._oneFor_4x4_Unroll`(L82~): 컬럼 4개 × 행 4개 언롤(L85,117), 레지스터 블로킹. FP32 SIMD 수동 최적화 baseline.

### 3.4 CPU: `LlamaModel.hpp` / `main.cpp`

- `LlamaModel.hpp`:
  - `LlamaLayer`(L14-62): libtorch(`torch::nn`)로 q/k/v/o_proj, MHA, up/gate/down_proj 구성(L16-23). `forward`(L44-61)는 표준 Llama 블록(RoPE 없음, SiLU gate, residual). **이건 정확도/구조 비교용 dense 참조**.
  - `TernaryMLP<int32_t>`(L92-): WU/WG/WD metadata + rowindex 보유(L95-100) → TCSC 기반 삼진 MLP를 정수로 벤치마크하는 래퍼.
- `main.cpp`: 벤치 드라이버. `Config_MKNSV`(L81-114)에 (M,K,N,sparsity,variation) 조합. 활성 케이스(L99-113)는 **M=1**(디코드/single-token), K∈{1024,2048,4096}, N=4K, sparsity 0.5~0.9. `record_time`/`print_ms_speedup`(L17-52)로 baseline 대비 speedup 산출. libtorch/Eigen을 정확도/속도 비교 기준으로 포함(L12-14).

### 3.5 GPU: `csrc/ter_spmm_kernel.cuh` (CUDA 커널)

**(a) `ter_tiled_mcsc_spmv` (L24-78) — M=1 (벡터) 커널**
- 템플릿: `<int TILE_K, bool UNIFORMED, DATA_TYPE, IDX_TYPE>`.
- 매핑(L36-39): **1 warp = 1 출력 컬럼**. `warp_id`=col, `lane_id`=warp 내 스레드.
- K를 TILE_K로 타일링(L40). 각 타일마다 4-원소 오프셋 벡터(`OffsetVec`, L42,47-50) 로드: [interleaved_start, remain_start, common_end, sign].
- 인터리브 루프(L53-57): lane가 `i = start + lane*2` 보폭으로 (pos,neg) 쌍을 읽어 `sum += X[pos] - X[neg]`. `__ldca`로 read-only 캐시 경유.
- 잔여(common) 루프(L59-67, `!UNIFORMED`): 같은 부호 인덱스를 모아 partial_sum 후 `sign(±1)`을 곱해 가산(L66).
- warp 내 리덕션(L72, `warp_reduce_sum` = `__shfl_down_sync` 트리, utils.cuh L237-241), lane0이 result 기록(L75-77).

**(b) `ter_tiled_mcsc_spmm` (L106-308) — M>1 (행렬) 커널, 가장 복잡**
- 템플릿(L85-86): `TILE_M, TILE_N, TILE_K, FRAGMENT_SIZE, UNIFORMED, PADDED, VEC_TYPE_M, DATA_TYPE, IDX_TYPE`.
- 매핑(L133-135): blockDim=(TILE_N,1). col=x-index, row=y-index. 스레드당 `TILE_M`개 결과(L149 `res[TILE_M]`).
- **공유메모리 X 타일**(L142): `tile_X[TILE_M*TILE_K]` row-major. 글로벌에서 column-major X를 벡터(`float4` 등)로 로드해 smem에 전치 저장(L162-181, `StoreFPVectorToArray` utils L208-222).
- **레지스터 프래그먼트**(L145): `fragment_row_indices[FRAGMENT_SIZE]`, 16바이트 정렬, 벡터 로드(L146,199-201).
- 인터리브 계산(L197-246): FRAGMENT_SIZE/quarterFragment/residual 3단계로 인덱스를 벡터 로드, 각 TILE_M 결과에 `res[j] += tile_X[pos]; res[j] -= tile_X[neg]`(L207-210). PADDED가 아니면 4-원소 단위 잔여 처리(L235-245).
- 공통(common) 계산(L249-300, `!UNIFORMED`): 같은 부호 인덱스를 partial에 누적 후 `res += sign * partial`(L296-298).
- 결과 store(L305-307): row-major 출력 `result[(row*TILE_M+j)*columns + col]`.
- 최적화 요소: 벡터화 로드(float4/short4), `#pragma unroll` 다수, smem 타일, 레지스터 누산, read-only 캐시(`__ldca`).

**(c) 커널 caller / dispatcher (L316-531)**
- `ter_spmm_kernel_caller`(L318-375): batch를 **2개 CUDA stream**으로 핑퐁 처리(L345-372)하여 배치 병렬성/오버랩 확보. 홀수 배치는 먼저 1개 처리(L330).
- `kernel_dispatcher`(L454-496): `assert(inners%256==0)`, `assert(columns%8==0)`(L459-460). rows>1이면 spmm, 아니면 spmv 경로. `DISPATCH_M/K/BOOL` 매크로(utils.cuh)로 TILE 크기·VEC 타입·uniformed/padded를 **컴파일타임 상수**로 분기 → 템플릿 특수화.
- `ter_spmm`(L318-348, wrapper.cu): torch::Tensor 입출력, device=CUDA assert, `ter_spmm_wrapper<float>` 호출.

### 3.6 GPU: `csrc/ter_spmm_wrapper.cu` (포맷 변환 + pybind)

- `convert_to_ter_csc`(L39-78): W(K×N, row-major int32)에서 `==-1`/`==1`을 분류해 pos/neg row_indice + col_offset 생성(L60-77). CPU의 `SparseFormat`과 동치.
- `convert_to_ter_tiled_csc`(L92-152): K를 TILE_K로 타일링하여 **타일별 col_offset** 생성. 행 인덱스에서 `tile_offset`을 빼서 **타일-로컬 인덱스**로 변환(L139,142) → smem 인덱싱과 정합.
- `convert_to_ter_tiled_mcsc`(L156-315): 타일별로 pos/neg를 인터리브 병합. 타일당 4-오프셋 [start, interleave_end, common_end, sign](L294-297). **padding 옵션**(L198-230): interleave/remain을 padding_size 배수로 정렬(여분 슬롯에 `pos[start]` 채움, L259-261) → 메모리는 늘지만 잔여 분기 제거. 끝에서 `inter_tile_swizzle`(L9-27) 호출: 타일을 비제로 개수 오름차순 정렬(L14-23)하여 **워프 간 로드 밸런싱**.
- `PYBIND11_MODULE(ter_spmm, ...)`(L350-355): `ter_spmm`, `convert_to_ter_csc/tiled_csc/tiled_mcsc`를 파이썬에 노출.

### 3.7 GPU: `TernaryLLM/TernaryLinear.py` (PyTorch 연동)

- `random_ternary_weights`(L23-85): 두 모드 — (1) `uniform=True`: 블록(512)당 +1/-1 개수를 균등·균형 분배(L39-77, max/uniform 희소성과 정합). (2) 일반: rand 텐서로 sparsity 비율만큼 0, 나머지 절반씩 ±1(L79-84).
- `convert_ternary_weights`(L87-113): `ter_spmm.convert_to_ter_tiled_mcsc` 호출(L107). `MapKSizeToTileSize`(L9-16)로 K→TILE_K 매핑(8192/4096→512, 2048/1024→256, 512→64). 결과를 GPU 텐서로(row_indices는 int16, L110).
- `class TernaryLinear(nn.Module)`(L115-184): `nn.Linear` 대체물. 생성 시 삼진 weight 생성·변환을 buffer로 등록(L142-146). `forward`(L152-182)는 입력 텐서·압축 weight·메타를 `ter_spmm.ter_spmm`에 전달. batch_size/rows/columns/inners + uniform/padding 플래그.
- 즉 **추론 시 weight는 이미 TCSC 압축 상태로 GPU 상주**, X만 GEMM 시 흐른다.

### 3.8 GPU: MLP / Attn / Llama / Config

- `TernaryMLP.py`(L11-36): `LlamaTernaryMLP`. gate/up/down_proj를 `TernaryLinear`로 구성, `down(act(gate(x))*up(x))`(L34) — Llama SwiGLU MLP를 삼진화.
- `TernaryAttn.py`(L15-127): `LlamaTernaryAttention`. q/k/v/o_proj를 `TernaryLinear`로 교체(L39-54). RoPE(L57,92), GQA repeat_kv(L99-100), scaled-dot softmax(L101-108)는 표준 dense 유지. **즉 선형투영만 삼진 SpMM, attention score 계산은 dense**. `collect_attn_linear_layers`/`replace_linear_layers`(L134-147)로 기존 HF 모델의 nn.Linear를 in-place 교체.
- `TernaryLlama.py`(L10-): `collect_linear_layers`/`replace_linear_layers`(L10-37)로 attn+mlp의 Linear를 전역 교체. seq padding 유틸(L39-60+).
- `configuration_ternary.py`(L3-49): `TernaryConfig(LlamaConfig)`. 추가 필드 `ternary_attn_linear`, `ternary_mlp`, `sparsity`(기본 0.8), `uniform_sparsity`, `uniform_sparsity_block_size`(512), `padding`, `padding_size`(4).

### 3.9 FPGA: `src/Config.scala` (`trait ConfigSys`) — 모든 파라미터의 원천

- AXI4 config(L9-22): addressWidth=64, **dataWidth=512**, idWidth=6, burst/len 사용.
- Coyote v2 상수(L25-29): VADDR_BITS=48, LOCAL_READ/WRITE, STRM_CARD/HOST.
- 비트폭(L31-35): **BIT_WIDTH_X_Y=16**(X/Y 데이터), **BIT_WIDTH_INDEX=8**(W 인덱스), **BIT_WIDTH_X=8**(X 입력), DATA_WIDTH=512.
- 비트당 엔트리(L46-48): `ENTRIES_PER_BEAT_X_Y=512/16=32`, `ENTRIES_PER_BEAT_W=512/8=64`, `ENTRIES_PER_BEAT_X=512/8=64`.
- **언롤/병렬 파라미터(L51-59)**: `S=64`(컬럼 병렬도, 출력 S/2=32개), `K_slice=128`(K 방향 타일), `UNROLL_M=4`(행 병렬도). → PE가 한 번에 S개 weight 인덱스를 받아 32개 출력 누산기 차분(±) 생성.
- 버퍼(L64-75): `TOTAL_ENTRIES_BUFFER=16384`, `BUFFERSIZE = UNROLL_M*16384/64`, BRAM `Buffer_X` 크기. `TOTAL_BEATS_KSLICE = K_slice*8/512 = 2`(K_slice 한 개를 2 beat로 로드).
- 비트당 데이터(L78-86): Row_Data_X/Y/W를 바이트로, AXI burst beat 수 계산.

### 3.10 FPGA: `src/design/PE.scala` (Processing Element) — ★ HW 핵심

- IO(L12-20): `x`(512bit 입력, DataFSM가 BRAM에서 읽어 공급), `w`(slave Stream 512bit, weight 인덱스 묶음), `enable_X`(2bit one-hot, x_reg 갱신용), `reset_acc`, `y`(512bit 출력).
- 레지스터:
  - `x_reg`(L26): `Vec(SInt(16bit), K_slice=128)` — 한 K_slice(128개) 활성값을 보관. 128×16bit = 2048bit = 4×512bit. (주석은 256이라 하나 코드상 K_slice=128.)
  - `acc`(L27): `Vec(SInt(BIT_WIDTH_X_Y=16bit), S=64)` — 64개 누산기.
- **핵심 연산(L30-40)**: 각 누산기 i에 대해 `io.w.fire`일 때 `acc(i) := acc(i) + x_reg(w_subdivided(i))`(L32-34). 즉 **weight 비트(8bit 인덱스)를 x_reg의 인덱스로 사용해 활성값을 게더, 누산**. 곱셈 없음 — 순수 indexed-accumulate.
- **차분 출력(L36-39)**: i가 홀수일 때 `io.y[i/2] := (acc(i-1) - acc(i))`. **짝수 누산기(pos)에서 홀수 누산기(neg)를 빼서** 최종 ±1 GEMM 결과 산출 → S=64 누산기 → 32개 출력. TCSC의 pos/neg 인터리브 구조가 HW에 그대로 매핑됨.
- x_reg 로드(L43-49): `enable_X(i)` one-hot으로 512bit beat를 32개 16bit로 subdivide하여 x_reg에 기록(TOTAL_BEATS_KSLICE=2 beat).
- `io.w.ready := True`(L97) — 항상 수신 준비. 주석 처리된 FSM 버전(L57-95)은 순차 처리 구버전(현재는 병렬 unrolled).
- **요약**: PE는 "8bit weight 인덱스 스트림으로 BRAM 상주 X를 인덱싱→64-way 병렬 누산→pos-neg 차분"하는 **곱셈기 없는 sparse MAC 유닛**.

### 3.11 FPGA: `src/design/DataFSM.scala` (데이터 이동 컨트롤러)

- IO(L13-56): Coyote v2 큐(sq_rd/sq_wr 요청, cq_rd/cq_wr 완료), `axis_card_recv`(데이터 수신), `x`/`w` 출력 스트림, 런타임 입력(M/N/K, base_addr_X/W/Y, Non_zero_per_K_slice, expected_beats_X), 제어(enable_X, reset_acc, select_Y, enable_Y_write, start/done, cnt_cycles).
- `Buffer_X`(L93-103): `Mem(512bit, BUFFERSIZE)` 온칩 BRAM. write port(L96-100), async read port(L103) → PE의 `io.x`로 공급.
- 다수 카운터(L106-120): cnt_N/M/K, cnt_entries, cnt_beats_W, cnt_unroll_Y/x, k_c(K_slice), cnt_s, cnt_row, cnt_read_4K 등.
- 주소 계산(L122-145): X는 **4K(4096B) 단위 burst**로 읽음(`X_4k_base`, L126, `four_K=4096`). remainder 처리(L135-141). Y 쓰기 주소(L130-132).
- **FSM 상태(L181-183)**: IDLE → LOAD_X_4K → LOAD_X_DATA → SET_ADDR → LOAD_SLICE → READ_W_AR → LOAD_W_DATA → WAIT_ACC → WAIT → MEM_ADDR_Y → MEM_WRITE_Y → NEXT_4ROWS.
  - `IDLE`(L188-227): 카운터 리셋, Buffer_X 초기화, start 대기 후 cnt_time 시작.
  - `LOAD_X_4K`(L252-278): X를 4K burst 또는 remainder 길이로 sq_rd 요청.
  - `LOAD_X_DATA`(L280-323): axis_card_recv에서 512bit씩 BRAM에 기록(L288-296), expected_beats 도달 시 SET_ADDR.
  - `SET_ADDR`(L325-328): BRAM read 주소 1cycle 선설정(async read 지연 회피).
  - `LOAD_SLICE`(L331-357): cnt_s(0/1)에 따라 `enable_X` one-hot 설정(L334-341)으로 PE x_reg에 K_slice 적재. UNROLL_M(4행) × TOTAL_BEATS_KSLICE 순회 후 READ_W_AR.
  - `READ_W_AR`/`LOAD_W_DATA`(L360-388): weight 인덱스를 Row_Data_W(=S개) 단위로 읽어 `io.w.valid`로 PE에 스트림.
  - `WAIT_ACC`(L390-410): `Non_zero_per_K_slice`회 만큼 S 인덱스 반복(K_slice당 비제로 수). K 소진 시 cnt_N += S_2(32), WAIT로.
  - `MEM_ADDR_Y`/`MEM_WRITE_Y`(L422-469): UNROLL_M개 행의 Y(S/2=32값)를 차례로 sq_wr로 기록(`select_Y`로 PE 선택, L441). N 소진까지 반복, 끝나면 cnt_M += UNROLL_M, NEXT_4ROWS.
- **요약**: 4행(UNROLL_M)을 한 묶음으로, K를 K_slice(128) 단위로 타일링하며 X는 4K burst로 BRAM 적재, W는 비제로 인덱스만 스트림. **uniform TCSC 가정**(컬럼당 비제로 수 = Non_zero_per_K_slice로 고정)으로 제어 단순화.

### 3.12 FPGA: `src/design/TopLevel.scala` / `WrapGEMM.scala`

- `TopLevel`(L12-132): DataFSM 1개 + **PE 어레이 `UNROLL_M(=4)`개**(L85) 인스턴스화. 모든 PE에 동일한 `x`/`w`를 broadcast(L89-95), `enable_X(i)`로 행별 x_reg 분리 적재. 출력은 `select_Y`로 PE를 선택해 `axis_card_send`로 송신(L99-105). 사이클 카운터로 성능 측정(L102-105).
- `WrapGEMM`(L9-103): **AxiLite4 컨트롤 레지스터 인터페이스**. M/N/K/Non_zero_per_K_slice/start/base_addr_X/W/Y/expected_beats를 host가 쓰고(L43-52), done/cnt_cycles를 읽음(L51,53). Coyote 스트림(sq/cq, axis_card)을 TopLevel로 패스스루(L31-38). 이것이 Coyote vFPGA의 최상위 모듈(`setDefinitionName("ternaryGEMM")`, Generator.scala L26).

### 3.13 FPGA: `src/Generator.scala` / 빌드 / 시뮬

- `Generator.scala`: `object TernaryGEMM`(L21-30)이 `WrapGEMM`을 `hw/gen/ternaryGEMM.v`로 Verilog 생성. PE/TestLogic도 개별 생성 가능. `MySpinalConfig`(L12-18): 동기 리셋, active-LOW.
- 빌드: `build.sc`(Mill, L5-16) — SpinalHDL 1.12.0, Scala 2.13.14. `build.sbt`도 병존.
- `sim/ternaryGEMM/TopLevelSim.scala`(L10-): 테스트 케이스(L14-26, 예 (M=4,N=32,K=256)), `AxiCoyote` 변환기 + `AxiMemorySim`으로 메모리 모델링(L52-80). entries=0.5*K/2(50% 희소 가정, L44).
- 호스트(`sw/gemmacc/main.cpp` L1-90+): Coyote `cThread`로 X/W/Y 메모리 할당, CSR 레지스터 설정, start→done 폴링, Y 읽기. X 생성(L54-72), uniform weight 생성(L75-).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 TCSC 포맷 (3 플랫폼 공통 사상)
1. 삼진 W(K×N)를 컬럼별로 +1 행 인덱스 리스트와 -1 행 인덱스 리스트로 분리(값 미저장).
2. (Merged) pos/neg를 인터리브하여 "+a−b" 쌍으로 묶고, 잔여는 단일 부호로 + remain_val 메타.
3. (Grouped/Tiled) 그룹/타일 단위로 정렬도(min/mid/max)를 맞춰 분기/잔여 루프를 최소화. 메모리(더미 0/padding) vs 제어 단순화 트레이드오프.

### 4.2 삼진 SpMM 알고리즘 (핵심)
- `Y[m,n] = Σ_{k∈pos(n)} X[m,k] − Σ_{k∈neg(n)} X[m,k]`. **곱셈 0개**, 인덱싱+덧셈/뺄셈만.
- CPU: row_index로 X(col-major)를 256/512bit 로드 → `add(res, sub(pos,neg))`. uniform이면 잔여 루프 제거.
- GPU: 1 warp/컬럼(spmv) 또는 (TILE_M,TILE_N) 블록(spmm). smem에 X 타일 적재 → 레지스터 인덱스 프래그먼트로 `res += X[pos]; res -= X[neg]`. 잔여는 sign 곱.
- FPGA: 8bit weight 인덱스로 BRAM 상주 x_reg를 게더 → 64-way 누산 → 짝(pos)−홀(neg) 차분.

### 4.3 메모리 계층 / 병렬화 / 파이프라이닝
- CPU: row-major→col-major X 레이아웃으로 연속 SIMD 로드. OpenMP로 컬럼(그룹) 병렬, AVX2/512로 데이터 병렬.
- GPU: 글로벌(압축 weight 상주) → smem(X 타일) → 레지스터(누산/인덱스 프래그먼트). warp shuffle 리덕션, **2-stream 배치 핑퐁**(kernel caller L345-372), `inter_tile_swizzle`로 워프 부하 균형, padding으로 분기 제거.
- FPGA: HBM(Alveo U55C, Coyote) → 4K burst → 온칩 `Buffer_X` BRAM → PE x_reg. UNROLL_M=4(행) × S=64(컬럼) 공간 병렬. FSM이 X 적재/W 스트림/Y 기록을 K_slice 단위로 시퀀싱(파이프라인은 명시적 단계 FSM 형태).

### 4.4 양자화
- weight는 {-1,0,+1} 삼진. X/Y는 CPU INT8(8bit) 또는 FP32, GPU FP32, FPGA는 X 입력 8bit·누산/출력 16bit(Config L31-35). 가중치 값 자체는 저장 안 하고 부호=배열소속, 크기=1로 고정 → "곱셈→누산" 치환.

---

## 5. HW/SW 매핑 (동일 알고리즘의 3중 구현)

| 개념 | CPU (C++) | GPU (CUDA) | FPGA (SpinalHDL) |
|---|---|---|---|
| 포맷 변환 | `TCSC.hpp` SparseFormat/MergedTCSC_Group | `ter_spmm_wrapper.cu` convert_to_ter_(tiled)_mcsc / `TernaryCSC.py` | (host) sw/main.cpp generateW + uniform 가정 |
| 비제로 분리 | pos/neg row_index | pos/neg row_indice | 8bit 인덱스 스트림 io.w |
| 인터리브 +a−b | AVX `add(res,sub(pos,neg))` | `res+=X[pos]; res-=X[neg]` | `acc(i-1)-acc(i)` 차분 (L37) |
| 타일/그룹 | group_size, K(컬럼) | TILE_M/N/K, FRAGMENT | K_slice=128, S=64, UNROLL_M=4 |
| 병렬 단위 | OpenMP 컬럼 + AVX lane | warp/컬럼, block/타일 | PE×4(행) × acc×64(컬럼) |
| 균일화/잔여제거 | "uniform" 커널 | UNIFORMED/PADDED 템플릿 | Non_zero_per_K_slice 고정 |
| 상위 통합 | main.cpp 벤치 / libtorch Llama | TernaryLinear→TernaryMLP/Attn/Llama (HF) | WrapGEMM AxiLite4 + Coyote sw |

- **공통 추상**: "컬럼별 pos/neg 인덱스 → 인터리브 누산 + 잔여" 구조가 SIMD lane / warp / acc 레지스터로 각각 사상.
- Python ↔ CUDA: `TernaryLinear.forward` → `ter_spmm.ter_spmm`(pybind) → `kernel_dispatcher` → 템플릿 커널.
- Scala ↔ Verilog: `Generator.TernaryGEMM` → `WrapGEMM`(AxiLite4) → `TopLevel`(PE×4 + DataFSM) → `hw/gen/ternaryGEMM.v` → Coyote vFPGA.

---

## 6. 빌드 · 실행 방법

- **CPU**(README L19-34): libtorch + Eigen 3.4.0 include/lib 설정. MSVC 플래그 `/Ox /Ot /Oi /arch:AVX2(또는 AVX512) /openmp /fp:fast /GL`, 링커 `/LTCG /OPT:REF /OPT:ICF`, `/openmp:experimental`, `_CRT_SECURE_NO_WARNINGS`. main.cpp 실행하여 벤치마크.
- **GPU**(README L1-7 + setup.py): `setup.py`에서 `cuda_home_path`(예 CUDA v12.9) 설정 → `python setup.py bdist_wheel` → `pip install --force dist/ter_spmm-1.0-...whl`. CUDAExtension, nvcc 플래그 `-use_fast_math -Xptxas -O3`(setup L25). 이후 `benchmark_*.py`/Benchmark.ipynb 실행.
- **FPGA**(README L10-30 + build.sc):
  - 툴: JDK8, sbt 1.11.4 / Scala 2.13.14 / SpinalHDL 1.12.0, **Mill 0.11.13**, Verilator 5.034, GTKWave.
  - 컴파일: `mill gemmacc.compile`. 시뮬: `mill gemmacc.runMain gemmacc.sim.ternaryGEMM.TopLevelSim`(L27).
  - HDL 생성: `mill gemmacc.runMain gemmacc.src.TernaryGEMM` → `hw/gen/ternaryGEMM.v`. 생성 후 AXI ctrl 주소 마스크 수동 수정(README L47-56).
  - 배포: Coyote(tutorial 브랜치) `examples/11_perf_GEMM`에 v 파일/svh/CMake 배치 → `cmake -DFDEV_NAME=u55c` → `make project` → `make -j32 bitgen` → `program_hacc_local.sh`로 U55C 프로그래밍 → sw `bin/test` 실행. 사전빌드 `coyote_files/hw/cyt_top.bit` 제공.

---

## 7. 의존성

- **CPU**: C++17, MSVC 2022, OpenMP, AVX2/AVX-512 intrinsics(`immintrin.h`), **libtorch(PyTorch C++)**, **Eigen 3.4.0**(비교 baseline).
- **GPU**: CUDA 12.9, **PyTorch**(torch::extension, CUDAExtension), **pybind11**, numpy, **HuggingFace transformers**(LlamaConfig/Attention/RoPE 재사용).
- **FPGA**: **SpinalHDL 1.12.0**(Scala 2.13.14), Mill/sbt, Verilator(sim), **ETH Coyote**(AXI4SR/ReqT/ack_t 인터페이스, vFPGA shell), Xilinx Vivado(bitgen), Alveo U55C/HACC.

---

## 8. 강점 / 한계 / 리스크

**강점**
- 동일 알고리즘(TCSC + addition-based SpMM)을 CPU/GPU/FPGA에 일관 사상한 **풀스택 레퍼런스** — 포맷 변환 코드가 3곳에서 동치라 교차검증 용이.
- 곱셈 완전 제거(±1) → FPGA에서 DSP 거의 불필요(PE는 adder/subtractor만), 에너지·자원 효율.
- TCSC의 pos/neg 분리·인터리브·그룹/타일 정렬은 SIMD/warp/PE 어느 폭에도 맞출 수 있는 일반적 설계.
- GPU 측의 padding/swizzle/2-stream, CPU 측 uniform 커널 등 부하 균형·분기 제거 기법이 구체적.

**한계 / 리스크**
- **FPGA는 uniform TCSC 가정**(컬럼당 비제로 수 고정, DataFSM `Non_zero_per_K_slice`): 비균일 희소성 처리 미지원 → 실제 학습된 가중치엔 pre-balancing/padding 필요(정확도 영향 가능).
- FPGA `BIT_WIDTH_X_Y=16` 누산 폭 → 큰 K/희소도 낮을 때 오버플로 리스크(확인 불가, 검증 코드 미독).
- weight 생성이 대부분 **랜덤 삼진**(CPU initData, GPU random_ternary_weights) — 실제 ternary-trained LLM 정확도 평가는 이 repo 범위 밖(추정).
- **SSR 폴더는 코드 부재**(README만) — DATE 2026 핵심 논문 구현이 비어 있음.
- CPU 코드 일부 비활성/주석(SIMD_Generator로 생성 추정), FPGA `old/`에 다수 중복 버전 → 유지보수 복잡.
- 플랫폼 간 정확도 일치 검증(bit-exact)은 명시적으로 확인 불가.

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA + XR 시선추적) 관점 시사점

**(A) 재사용 가능한 핵심 아이디어 — 삼진/희소 GEMM의 ViT 적용**
- ViT의 연산 대부분은 **선형 투영(QKV/proj, MLP fc1/fc2)** = GEMM. 본 repo의 `TernaryLinear`처럼 ViT의 nn.Linear를 삼진 SpMM으로 치환하면, HG-PIPE의 파이프라인 PE에서 **곱셈기→누산기**로 대체 가능. XR 시선추적처럼 **저지연·저전력**이 중요한 엣지 추론에 특히 부합.
- TCSC의 "값 미저장(부호=배열소속)" 압축은 ViT weight도 그대로 적용 가능 → on-chip weight 상주량 대폭 감소(HG-PIPE의 전체 weight on-chip 전략과 시너지).

**(B) 직접 차용 가능한 설계/코드**
- **`PE.scala`(L30-40)의 indexed-accumulate + pos/neg 차분 구조**: HG-PIPE의 MAC PE를 삼진 모드로 확장하는 청사진. `acc(i-1)-acc(i)` 차분(L37)으로 부호 처리, weight 인덱스 스트림으로 X 게더하는 패턴은 ViT FFN에 직접 이식 가능. 곱셈기 제거로 동일 LUT/DSP 예산에서 PE 수를 늘려 처리량↑.
- **`DataFSM.scala`의 K_slice 타일링 + 4K burst + BRAM double-region(UNROLL_M)**: ViT의 token×dim GEMM을 타일 단위로 흘리는 데이터무브 컨트롤러 템플릿. HG-PIPE의 레이어별 파이프라인 버퍼 관리에 참고.
- **`Config.scala`의 파라미터화(S/K_slice/UNROLL_M/DATA_WIDTH)**: SpinalHDL trait로 언롤·타일을 한 곳에서 조절하는 방식은 ViT 헤드 수/패치 수에 맞춘 DSE에 유용.
- **GPU `inter_tile_swizzle` + padding**(wrapper.cu L9-27, L198-230): 비균일 희소도를 균형화하는 전처리 — ViT weight를 FPGA용으로 컴파일타임에 정렬/패딩하는 toolflow 아이디어로 차용.
- **CPU `MergedTCSC_Group`의 min/mid/max 정렬 정책**(TCSC.hpp L146-238): "더미 0 삽입으로 분기 제거 vs 메모리" 트레이드오프는 FPGA에서 **고정 길이 inner-loop**(파이프라인 II=1) 확보에 직결 — semi-structured(N:M) 희소성 강제와 동일 사상.

**(C) XR 시선추적 특화 시사점**
- 시선추적 ViT는 보통 작은 입력/저latency 요구 → 본 repo의 **M=1(single-token) 최적화 경로**(CPU main.cpp L99-113, GPU spmv 커널)가 프레임당 1회 추론에 적합한 참조.
- 삼진화로 모델 메모리가 작아지면 HG-PIPE식 **all-on-chip** 구현 가능성↑ → 외부 메모리 접근 제거로 결정론적 저지연(XR 필수) 달성.

**(D) 주의**
- 본 FPGA 설계는 Coyote/U55C(데이터센터) 타깃·uniform 가정 → HG-PIPE의 임베디드/스트리밍 파이프라인에 그대로 쓰기보다, **PE 연산 코어와 TCSC 포맷 개념만 추출**하고 데이터무브/제어는 HG-PIPE 구조에 맞게 재설계 권장. 정확도(삼진 ViT)는 별도 학습/검증 필요(이 repo 미제공).

---

## 10. 근거 / 한계 표기

- **라인 근거**: 본 분석의 모든 코드 주장은 위 파일을 직접 Read하여 라인 번호로 표기함(예: PE.scala L37, DataFSM.scala L391, ter_spmm_kernel.cuh L207-210, TCSC.hpp L146-238).
- **추정 항목**(원문 명시 없음, 정황 근거):
  - 소속 ETH Zurich, CPU/GPU 논문 제목과 코드의 1:1 대응, "이 thesis"가 FPGA repo = 석사논문이라는 점은 README 문맥 기반 **추정**.
  - PE.scala 주석의 x_reg=256 vs 코드 K_slice=128 불일치 — 코드값(128) 우선 채택, 주석은 구버전 잔재로 **추정**.
  - FPGA 16bit 누산 오버플로/정확도, 플랫폼 간 bit-exact 일치는 **확인 불가**(검증/테스트 코드 InputAndCheck.scala, benchmark 미정독).
- **미존재/빈/제외**:
  - `SSR/`는 **README만 존재, 소스 코드 없음** — DATE 2026 핵심 논문 구현 부재(확인됨).
  - `src/old/`(Base/AXI_version/coyote_v2), `coyote_files/`, MatrixAdd, *.bit/*.ipynb/*.odt/*.ods는 분석 제외(이름만 언급).
  - `initData.cpp`, `SIMD_Generator.ipynb`, GPU `benchmark*.py`, `util/InputAndCheck.scala`, `util/AxiMemorySim.scala`, `coyote/Types.scala`/`AxiCoyote.scala`는 정독하지 않음(보조/생성/통합 코드로 판단) — 필요 시 추가 분석 권장.
- 외부 라이브러리(libtorch/Eigen/transformers/pybind11/Coyote) 내부는 분석하지 않음.
