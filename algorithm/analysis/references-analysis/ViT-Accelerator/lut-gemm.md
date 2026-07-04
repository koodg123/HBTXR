# LUT-GEMM 정밀 분석

> 분석 대상 repo: `REF/ViT-Accelerator/lut-gemm`
> 분석 일자: 2026-06-20
> 근거: 실제 소스 Read 기반 (파일명:라인 표기). `thirdparty/googletest/**`, build 산출물은 분석 제외.

---

## 1. 개요

### 한 줄 요약
**LUT-GEMM**은 sub-4-bit로 양자화된 LLM 가중치(BCQ / binary-coding quantization, 그리고 RTN→BCQ 변환)를 **디퀀트(dequantization) 없이** LUT(Look-Up Table) 기반 부분합 누적으로 곱하는 W4A16(가중치 4-bit, activation FP16) GEMV/GEMM CUDA 커널 라이브러리다. 곱셈 대부분을 "activation 부분합을 미리 계산한 테이블 조회 + 덧셈"으로 치환하는 것이 핵심이다.

### 원논문 / 저자 / 소속
- 논문: **"LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models"** (`README.md:5`).
- 저자: Gunho Park, Baeseong Park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, Dongsoo Lee (`README.md:7`).
- arXiv: **2206.09557**, primaryClass `cs.DC` (`README.md:9`, `README.md:31-38`). citation year 2023.
- 저작권/라이선스: **Copyright (c) 2024-present NAVER Cloud Corp.**, Apache License 2.0 (`README.md:44-46`, 모든 소스 헤더 주석). → 소속은 **NAVER Cloud Corp.** 으로 확정. (질문에서 추정한 "NAVER/Samsung" 중 NAVER가 맞고, Samsung 표기는 소스에서 확인되지 않음. 저자 다수는 NAVER 소속 추정.)

### 타깃 플랫폼
- **NVIDIA GPU** (CUDA). `CMakeLists.txt:5` `find_package(CUDA 10.1 REQUIRED)`, `CMakeLists.txt:9` 기본 아키텍처 `CMAKE_CUDA_ARCHITECTURES 80` (Ampere, A100). README Quick Start도 `cmake -DCMAKE_CUDA_ARCHITECTURES=80` (`README.md:23`).
- **연산 정밀도: FP16 (`__half`)** 가중치 alpha·activation·output. 비교 baseline은 cuBLAS FP16 GEMM(`cublas.h:37-39`, `CUBLAS_COMPUTE_16F`).
- 의존: cuBLAS, cuda_fp16, curand, gomp (`lutGEMM/CMakeLists.txt:13`).

### 목적
LLM 추론 단계의 **메모리-바운드** GEMV(배치 1, 토큰 1개 생성 시 `M=1`)를 가속한다. 가중치를 1~8-bit BCQ로 압축해 메모리 대역폭을 줄이고, LUT 기법으로 dequant·곱셈 비용을 제거한다. 테스트는 "175B 모델의 한 레이어"(`H=7168`, `N=4H`, `K=H`)를 대상으로 cuBLAS / OPTQ(GPTQ) / LUT-GEMM의 레이턴시를 비교한다(`tests/opt/fp16/int3_col_wise_matmul_fp16.cu:292-314`).

---

## 2. 디렉토리 구조

```
lut-gemm/
├── README.md                      # 논문/citation/quick start (Table 1 재현)
├── CMakeLists.txt                 # 최상위: CUDA enable, arch=80, 3개 서브디렉토리
├── docs/overview.png              # (이미지, 본 분석 제외)
├── lutGEMM/                       # ★ 라이브러리 본체
│   ├── CMakeLists.txt             # STATIC lib "lutgemm" = nQWeight_fp16.cu + kernels.cu
│   ├── include/
│   │   ├── lutGEMM               # umbrella 헤더 (nQWeight_fp16.h + kernels.h include)
│   │   ├── nQWeight_fp16.h       # nQWeight_fp16 클래스 (양자 가중치 컨테이너)
│   │   └── kernels.h            # public API: matmul / matmul_gptq / matmul_gptq_faster
│   └── src/
│       ├── nQWeight_fp16.cu      # 가중치 parsing(호스트→GPU), dequant CPU/GPU
│       ├── kernels.cu            # matmul 디스패치 (GEMV vs cuBLAS), gptq 래퍼
│       └── cuda/
│           ├── tmpWeight.hpp     # dequant용 임시 GPU 버퍼 싱글톤
│           └── kernels/
│               ├── mv.hpp                  # (legacy) FP32 LUT GEMV 커널 _nqmv
│               ├── mv_fp16.hpp             # ★ FP16 LUT GEMV 커널 _nqmv
│               ├── mv_fp16_bias.hpp        # ★ FP16 LUT GEMV + q_bias (offset) 커널
│               ├── mm_t.hpp                # (미연결) LUT GEMM(다중행) 커널 _nqmm_t
│               ├── dequant.hpp             # (legacy FP32) BCQ→dense 디퀀트
│               ├── dequant_fp16.hpp        # ★ FP16 BCQ→dense 디퀀트 (cuBLAS 경로용)
│               ├── gptq_fp16_bias.hpp      # GPTQ 3-bit GEMV (참조 baseline)
│               ├── gptq_faster_fp16_bias.hpp # GPTQ 3-bit GEMV faster (half2, deq2 LUT)
│               └── cublas.h / cublas.cu    # cublasGemmEx 래퍼
├── examples/                      # 양자화(BCQ/RTN) Python 파이프라인
│   ├── README.md
│   ├── bcq.py                     # ★ BCQ 양자화 알고리즘 (greedy/alternating/BST)
│   ├── bcq_parameter.py           # BCQParameter: compress/decompress, packing 호출
│   ├── rtn_parameter.py           # ★ RTN(uniform) 양자화 + INT→BCQ 변환
│   ├── utils.py                   # Quantizer(RTN), Packer(비트패킹), Compression base
│   ├── quant_model_bcq.py         # HF OPT 모델 전체 BCQ 양자화 드라이버
│   └── quant_model_rtn.py         # HF OPT 모델 전체 RTN→BCQ 양자화 드라이버
├── tests/                         # 자체 벤치마크 (googletest 매크로 사용)
│   ├── CMakeLists.txt
│   ├── main.cc                    # InitGoogleTest / RUN_ALL_TESTS
│   ├── include/{tests.h,timer.h,custom_random.h,_cublas.h}
│   ├── src/custom_random.cpp
│   └── opt/
│       ├── fp16/int3_col_wise_matmul_fp16.cu  # ★ Table1 레이턴시 비교 테스트
│       └── {_cublas.cc,_cublas.cu}
└── thirdparty/googletest/**       # 외부 의존성 — 분석 제외 (이름만 언급)
```

핵심 빌드 산출물은 두 개의 `.cu`(`nQWeight_fp16.cu`, `kernels.cu`)이며, 나머지 커널은 모두 `.hpp`로 `namespace lutGEMM { #include ... }` 형태로 **소스 내 인라인 포함**된다(`kernels.cu:24-28`, `nQWeight_fp16.cu:26-27`). 즉 헤더-온리 스타일 커널 모음을 2개 TU에서 컴파일하는 구조.

---

## 3. 핵심 모듈 정밀 분석 ★

### 3.1 BCQ(Binary-Coding Quantization)의 수학

BCQ는 가중치 벡터(또는 group)를 q개의 부호 행렬과 스케일의 합으로 근사한다:

```
w ≈ Σ_{i=0}^{q-1}  α_i · b_i ,   b_i ∈ {-1, +1}
```

- `α_i` : i번째 비트-플레인의 스케일링 팩터(FP16). nQWeight에서 `alpha[group][nb][m]`로 보관(`nQWeight_fp16.h:33`, 주석 `alpha[num_alpha_groups][nb][mSize]`).
- `b_i` : i번째 비트-플레인의 부호 비트. 32개 비트를 `uint32_t`로 패킹(`nQWeight_fp16.h:29`, `bWeight Weight[kSize/32][nb][mSize]`).
- `nb` = num_bits = q (1~8 지원). `group_size = kSize / num_alpha_groups` (`nQWeight_fp16.cu:63`)로 K축을 그룹 단위로 나눠 α를 공유(그룹별 스케일).

이 분해는 **uniform 양자화(RTN/INT)도 표현 가능**하다. INT4 한 값을 `(b_{q-1}...b_0)` 비트로 보면 `Σ 2^i b_i`이므로, 스케일을 `α_i = scale·2^i/2`로 두면 BCQ 형식으로 변환된다(`rtn_parameter.py:77-96` `convert_bcq_format`: `upack=[2^i]`, `scale = scale/2 · upack`, asymmetric zero는 `offset`으로 흡수). 따라서 LUT 커널 하나로 비균일(BCQ)·균일(RTN) 둘 다 처리한다.

#### BCQ 학습(Python, bcq.py)
- `quantize()` (`bcq.py:22-98`): weighted PTQ. group_size로 reshape 후
  - `greedy_mean_torch` (`bcq.py:100-125`): 잔차 `r`의 부호 `b=r.sign()`, 스케일 `α = mean(|r|)`(가중치 wf 있으면 가중평균), 잔차 갱신 `r -= b·α`을 q회 반복 — 탐욕적 비트-플레인 분해.
  - `refine_mean_torch` (`bcq.py:127-153`): α는 `B^T B α = B^T w` 정규방정식을 **batch conjugate gradient**(`batch_cg_torch`, `bcq.py:187-206`)로 재추정, B는 **이진 탐색**(`find_B_torch`, `bcq.py:161-185`, `2^q` 후보 중 nearest) 또는 greedy로 재배정. `rounds`회 교대 최적화(alternating, `bcq.py:80-83`).
  - 출력 텐서 shape: `B[out,in//gs,gs,q]`, `alpha[out,in//gs,q]` (`bcq.py:91-92`).

### 3.2 LUT 사전계산 — 핵심 아이디어

GEMV `y_m = Σ_k W[m,k]·x[k]` 에서 W를 BCQ로 풀면
`y_m = Σ_b α_{b,m} · ( Σ_k b_{b,m,k}·x[k] )`.
내부 합 `Σ_k b·x`은 **b ∈ {-1,+1}만으로 결정**되므로, k를 8개씩 묶으면 8-bit 부호 패턴(256가지)에 대한 부분합 `Σ_{j=0}^{7} (±1)·x[j]`를 **단 한 번 미리 계산**해 LUT[256]에 저장할 수 있다. 이후 가중치는 "8-bit 패턴 인덱스"로 LUT를 조회만 하면 곱셈 없이 부분합을 얻는다. 이것이 dequant-free의 본질이다.

#### LUT 구성 코드 (`mv_fp16.hpp:26-62`, FP16 `_nqmv`)
1. shared LUT: `__shared__ __half lut[K_TILE_SIZE/8][256]` (`mv_fp16.hpp:28`). K_TILE_SIZE/8개의 sub-LUT(각 8개 activation 담당), 각 256 엔트리.
2. base 계산 (`mv_fp16.hpp:36-45`): 스레드의 `lut_x`(8-bit 패턴)에 대해 8개 activation의 부호합
   `base = Σ_{j} (2·((lut_x>>j)&1) - 1) · _inp[j]` 을 직접 계산해 `lut[lut_y][lut_x]` 채움. `(2·bit-1)`로 0/1 → -1/+1 매핑.
3. 점화식 확장 (`mv_fp16.hpp:56-61`): 스레드가 부족해 패턴 256개를 다 못 채우면, `lut[i+x] = lut[i+x-(1<<s)] + 2·_inp[s]`로 한 비트씩 켜진 항만 더해 나머지 엔트리를 누적 생성(prefix 식 확장). 곱셈 없이 덧셈만으로 256개 완성.
4. `__syncthreads()` (`mv_fp16.hpp:62`): LUT 공유 후 누적 단계로 진입.

> 핵심: LUT는 **shared memory**에 둔다(블록 내 모든 출력 행 m이 같은 activation tile을 재사용). global memory가 아님. 이로써 activation을 한 번만 읽고, 256개 부분합을 블록 전체가 공유.

### 3.3 비트단위 누적 (LUT 조회 + alpha 가중합)

`mv_fp16.hpp:71-96`:
```
for m (출력행, 스레드당 2행씩 half2):
  for b in [0..NUM_BITS):
    reg_a = alpha[group_idx*nb*M + b*M + m]
    for kt in [0..K_TILE_SIZE/32):           # 32-bit word 단위
      reg_w = bW[kt*nb*M + b*M + m]           # 가중치 비트워드 1개
      reg_w0 = reg_w & 0xFF; reg_t_o += lut[kt*4+0][reg_w0]   # 8-bit씩 4번
      reg_w1 = (reg_w>>8)&0xFF; reg_t_o += lut[kt*4+1][reg_w1]
      ... (총 4개의 8-bit 인덱스로 LUT 조회·누적)
    reg_o += reg_a * reg_t_o                  # 비트-플레인 스케일 곱은 단 1회
  atomicAdd((half2*)&output[m], {reg_o0,reg_o1})
```
- **인덱싱**: 32-bit `reg_w`를 8-bit 4조각으로 쪼개 4개의 sub-LUT(`lut[kt*4+0..3]`)에서 부분합을 끌어와 누적. 곱셈은 비트-플레인당 `α·(부분합)` 한 번뿐 → 32개 weight 처리에 곱셈 4개가 아니라 사실상 0 (LUT 덕분), 비트-플레인 스케일 곱만 잔존.
- **부분합 누적**: K 타일 전체를 K_TILE_SIZE/32 워드 루프로 합산 → `reg_t_o`. 여러 K-블록(grid.y)이 같은 출력 m을 쓰므로 최종은 `atomicAdd`로 합침(`mv_fp16.hpp:95`).
- **half2 처리**: 출력 두 행을 `__halves2half2`로 묶어 한 atomicAdd(`mv_fp16.hpp:95`), 스레드당 2행(`threadIdx.x*2`, `mv_fp16.hpp:64`).

### 3.4 스레드/워프·블록 매핑 (`mv_fp16.hpp:99-128`)
- grid = `(ceil(M/m_tile), ceil(K/k_tile))` (`mv_fp16.hpp:100-102`). blockIdx.x = 출력행 타일, blockIdx.y = K 타일.
- block = `num_threads = 256`, `m_tile_size = 2048` (`mv_fp16.hpp:122-123`). 256 스레드가 한 LUT를 만들고 2048개 출력행을 stride 처리.
- LUT 채우기에서 `lut_x_size = blockDim.x/(K_TILE_SIZE/8)` (`mv_fp16.hpp:29`)로 스레드를 (sub-LUT y) × (패턴 x)로 2D 분할. `lut_y = tid/lut_x_size`, `lut_x = tid%lut_x_size`.
- K_TILE_SIZE는 컴파일타임 템플릿(`_excute_nqmv<32*1..32*8>`)로 8종 인스턴스화, 런타임 함수포인터 테이블로 선택(`mv_fp16.hpp:107-118`). `nqmv()`는 현재 `k_tile_idx=1`(=64) 고정(`mv_fp16.hpp:122`).

### 3.5 q_bias(offset) 경로 — RTN asymmetric 지원 (`mv_fp16_bias.hpp`)
RTN의 asymmetric zero-point를 처리하기 위한 변형. `_nqmv_bias`는 일반 비트-플레인 누적 전에 **offset 항**을 추가한다(`mv_fp16_bias.hpp:74-92`):
- `lut[..][255]`는 모든 비트가 1인 패턴(=`Σ x[j]`, 전부 +1). 이를 K 타일 전체에 대해 합해 `reg_t_o`로 만들고 `q_bias`(=offset 스케일)와 곱해 누적. 즉 `offset · Σx` 항을 LUT 재사용으로 계산(`mv_fp16_bias.hpp:80-91`).
- 이후 일반 BCQ 누적(`mv_fp16_bias.hpp:94-115`)은 `_nqmv`와 동일.
- 디스패치: `matmul()`에서 `nqW.q_bias == nullptr`이면 `nqmv`, 아니면 `nqmv_bias`(`kernels.cu:59-60`, `kernels.cu:67-68`). RTN→BCQ 변환이 `offset`을 만들기 때문(`rtn_parameter.py:85`).

### 3.6 RTN / GPTQ 경로 차이
1. **LUT-GEMM 정규 경로 (BCQ·RTN 공통)**: 위 3.1~3.5. RTN은 Python에서 `convert_bcq_format`으로 BCQ화(`α=scale·2^i/2`, offset) → 같은 `_nqmv_bias` 커널 사용. **dequant 없이** LUT로 직접 GEMV.
2. **GPTQ baseline (`gptq_fp16_bias.hpp`)**: 비교용 참조 구현. `VecQuant3MatMulKernel`(`gptq_fp16_bias.hpp:35-105`)은 **전형적 dequant-then-MAC** 방식 — 3-bit 패킹 워드를 비트 시프트로 풀어 `scale·q - zero`로 매 weight를 실수 복원 후 `× blockvec`(activation). LUT 없음, 곱셈 그대로. BLOCKHEIGHT=24, BLOCKWIDTH=256(`gptq_fp16_bias.hpp:27-28`), 11+11+10=32개 weight를 32-bit 3개에서 언패킹하는 비트 경계 처리(`gptq_fp16_bias.hpp:70-71,86-87`).
3. **GPTQ faster (`gptq_faster_fp16_bias.hpp`)**: half2 벡터화 + `deq2[64][32]` 소형 디퀀트 LUT(`gptq_faster_fp16_bias.hpp:45-52`: 2개 3-bit 값을 half2로 미리 디코드)와 `__hfma2` FMA로 가속. 여전히 GPTQ식(dequant 기반)이며 LUT-GEMM의 "activation 부분합 LUT"와는 다른 개념(weight 코드→값 LUT).

> 즉 같은 repo 안에 **(a) LUT-GEMM 본 기법**과 **(b) GPTQ 두 변형(비교군)**이 공존한다. 본 논문 기법은 (a). 테스트에서 OPTQ=GPTQ_faster로 측정(`int3_col_wise_matmul_fp16.cu:80,304-305`).

### 3.7 FP16 처리
- 모든 누적 변수(`reg_o`, `reg_t_o`, base, LUT)가 `__half`(`mv_fp16.hpp:36-43,71-78`). `__float2half((2·bit-1))`로 부호 상수를 FP16화. 출력 atomicAdd는 `half2` 단위.
- alpha/q_bias는 host float을 `__float2half`로 변환해 managed memory에 저장(`nQWeight_fp16.cu:84-85,80-81`).
- FP16 누적은 정밀도 손실 위험이 있으나(특히 K가 큰 경우 reg_t_o 누산), 본 구현은 속도 우선. (GPTQ faster는 res를 float로 누산: `gptq_faster_fp16_bias.hpp:60,100`로 정밀도 보완 — LUT 경로는 그렇지 않음 → 한계 8장 참조.)

### 3.8 nQWeight_fp16 클래스 구조 (`nQWeight_fp16.h:27-52`, `nQWeight_fp16.cu`)
멤버 (`nQWeight_fp16.h:29-37`):
- `unsigned int* bWeight` : 패킹된 부호 비트 `[K/32][nb][M]`.
- `void* alpha` : `__half` 스케일 `[num_groups][nb][M]`.
- `void* q_bias` : `__half` offset `[num_groups][M]` (nullptr 가능).
- `num_groups, group_size, mSize(=출력 M), kSize(=입력 K), nb(비트수), is_row_wise_quantize`.

`parsing()` (`nQWeight_fp16.cu:60-91`):
- `is_row_wise_quantize`로 M/K를 row/col에 매핑(`nQWeight_fp16.cu:69-76`). `group_size = kSize/num_alpha_groups`.
- alpha/q_bias: `cudaMallocManaged` 후 float→`__half` 변환 복사(`nQWeight_fp16.cu:80-85`).
- bWeight: `cudaMallocManaged` 후 `cudaMemcpy(H2D)` (`nQWeight_fp16.cu:87-88`). 크기 `K·M·nb/32` words.

`getDequantiedWeight()` (`nQWeight_fp16.cu:93-99`): cuBLAS 경로(M>1)에서만 사용. `tmpWeight` 싱글톤(`tmpWeight.hpp:21-51`)이 재사용 가능한 GPU 버퍼를 주고 `dequantize_gpu`로 dense FP16 W를 복원한 뒤 cuBLAS GEMM.

소멸자 버그 노트: `~nQWeight_fp16()`가 `cudaFree(alpha)`를 **두 번** 호출(`nQWeight_fp16.cu:102-103`) — bWeight free 누락 + alpha 이중 free(8장 리스크).

---

## 4. 데이터 플로우 / 실행 흐름

### 4.1 오프라인 양자화 (Python, 호스트)
```
HF 사전학습 모델(OPT 등)
  └ quant_model_bcq.py / quant_model_rtn.py : 대상 레이어(q/k/v/out_proj/fc1/fc2) 순회
       ├ [BCQ]  BCQParameter.compress → bcq.quantize (greedy+alternating) → (alpha, binary{-1,+1}, shape)
       └ [RTN]  RTNParameter.compress (uniform INT) → convert_bcq_format → (scale=α, binary, offset)
  └ PACKER.pack (utils.py:162-172) : {-1,+1}→{0,1}→8개를 uint8 1바이트로 비트패킹
```
- BCQ: `bcq.py:79-83` greedy로 초기 (B,α), `rounds=15`회 alternating 정제(`quant_model_bcq.py:77-79`).
- RTN: `utils.py:Quantizer` scale/zero 산출(`utils.py:52-133`) → `convert_bcq_format`에서 `α_i=scale/2·2^i`, `offset=Σα-zero`(`rtn_parameter.py:80-85`).
- Packing: `(b+1)/2` → `[8,-1]` → `·[1,2,4,..,128]` 합 → uint8 (`utils.py:167-172`). 이것이 커널의 `uint32_t bWeight` 비트레이아웃에 대응.

### 4.2 온라인 추론 (CUDA, 디바이스)
```
nQWeight_fp16.parsing(bW, A, K, N, num_bits, ...)   # bWeight H2D, alpha/q_bias managed FP16
        │
matmul(output, input, nqW, m)   (kernels.cu:64-71)
        ├ m==1 (GEMV, LLM 토큰 생성):
        │     cudaMemset(output,0)  →  q_bias 없으면 kernel::nqmv, 있으면 kernel::nqmv_bias
        │        └ _nqmv<<<grid,block>>> : (1)shared LUT 구성 →(2)__syncthreads →(3)LUT조회·α누적 →(4)atomicAdd
        └ m>1  (GEMM):
              matmul_useCublas → nqW.getDequantiedWeight(true)  # dequant_fp16로 dense W 복원
                 → kernel::cublas_gemm_ex (cublasGemmEx FP16)   # 일반 GEMM으로 처리
```
- **LUT-GEMM의 빠른 경로는 GEMV(M=1)** — LLM 디코딩(토큰 1개씩 생성)의 지배적 연산. M>1이면 dequant+cuBLAS로 폴백(`kernels.cu:62,70`, `matmul_useCublas` `kernels.cu:73-79`).

### 4.3 메모리 계층 / 병렬화 요약
| 데이터 | 위치 | 근거 |
|---|---|---|
| activation 부분합 LUT[K/8][256] | **shared memory** (블록 재사용) | `mv_fp16.hpp:28` |
| bWeight (비트패킹) | global (managed) | `nQWeight_fp16.cu:87` |
| alpha / q_bias | global (managed, FP16) | `nQWeight_fp16.cu:80-85` |
| 부분합 누적 reg_t_o, reg_o | 레지스터 | `mv_fp16.hpp:71-93` |
| output | global, half2 atomicAdd | `mv_fp16.hpp:95` |
- 병렬화: grid.y(K타일)×grid.x(M타일), 블록당 256스레드가 LUT 협력 구성 후 각자 출력행 stride 누적. activation read는 블록당 1회(LUT화)로 메모리 대역폭 절감.

---

## 5. HW/SW 매핑 (GPU/CUDA ↔ Python ↔ cuBLAS)

| 레이어 | 구성요소 | 역할 | 파일 |
|---|---|---|---|
| **SW (Python, 오프라인)** | `bcq.py quantize` | 비균일 BCQ 비트-플레인 분해 (greedy+CG+BST) | `examples/bcq.py` |
| | `utils.py Quantizer` | 균일 RTN(scale/zero, MSE 옵션) | `examples/utils.py:28-144` |
| | `rtn_parameter.convert_bcq_format` | INT→BCQ(α,offset) 변환 | `examples/rtn_parameter.py:77-96` |
| | `utils.py Packer` | {-1,+1}→uint8 비트패킹(커널 레이아웃) | `examples/utils.py:146-181` |
| **SW/HW 경계** | `nQWeight_fp16.parsing` | host 파라미터 → GPU(managed) 업로드, FP16 변환 | `src/nQWeight_fp16.cu:60-91` |
| **HW (CUDA 커널)** | `_nqmv` / `_nqmv_bias` | ★ LUT 기반 dequant-free GEMV (논문 기법) | `mv_fp16.hpp`, `mv_fp16_bias.hpp` |
| | `_dequantize(_t)` | BCQ→dense FP16 복원 (GEMM 폴백용) | `dequant_fp16.hpp` |
| | `VecQuant3MatMulKernel(Faster)` | GPTQ baseline (dequant-then-MAC) | `gptq_*_fp16_bias.hpp` |
| **HW (vendor)** | `cublas_gemm_ex` | FP16 GEMM 기준선 / M>1 폴백 | `cublas.h`, `cublas.cu` |
| **검증/벤치** | `int3_col_wise_matmul_fp16` | cuBLAS vs GPTQ vs LUT-GEMM 레이턴시 | `tests/opt/fp16/...cu` |

대응 관계 핵심: **Python이 만든 (alpha, packed-binary, offset)** ↔ **커널의 (alpha, bWeight, q_bias)** ↔ 정확도 기준은 **cuBLAS FP16 dense GEMM**. `checkErr()`(`int3_..cu:155-161`)가 LUT-GEMM 출력과 cuBLAS 출력의 mean abs error로 정확도 검증.

---

## 6. 빌드 / 실행

### CUDA 라이브러리 + 벤치 (`README.md:20-26`)
```sh
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=80 ..   # 기본 80 (A100). 다른 GPU면 변경
make -j8
./tests/tests                            # Table 1 레이턴시 재현
```
- 요구: CMake ≥3.18(`CMakeLists.txt:1`), CUDA ≥10.1(`CMakeLists.txt:5`). 링크: cublas/cublasLt/curand/cudart/cuda/gomp(`tests/CMakeLists.txt:29`).
- 빌드 단위: static lib `lutgemm`(`nQWeight_fp16.cu`+`kernels.cu`) (`lutGEMM/CMakeLists.txt:10`), 테스트 실행파일 `tests`.

### Python 양자화 (`examples/README.md:5-16`)
```sh
python quant_model_bcq.py --model_name_or_path facebook/opt-125m --qbits 4 --group_size 128
python quant_model_rtn.py --model_name_or_path facebook/opt-125m --qbits 4 --group_size 128
```
- 의존: torch, transformers, numpy, tqdm. 단독 검증: `python bcq_parameter.py`, `python rtn_parameter.py`(`__main__`에서 reconstruction error 출력).

> 주의: Python(가중치 양자화 산출)과 CUDA 커널 사이의 **자동 직렬화/로딩 글루 코드는 repo에 없음**. Python은 파라미터 텐서를 만들고 크기만 출력(`quant_model_bcq.py:81-83`); 커널은 별도 `nQWeight_fp16::parsing`으로 raw 포인터를 받음. 즉 두 측은 동일 데이터 레이아웃을 공유하지만 end-to-end 연결 스크립트는 사용자 몫(8장 한계).

---

## 7. 의존성

| 종류 | 항목 | 용도 | 근거 |
|---|---|---|---|
| CUDA toolkit | cuda, cuda_fp16, cuda_runtime, cuda_profiler_api | 커널/FP16/타이밍 | `kernels.cu:18-20`, `cublas.h:17-21` |
| vendor lib | cuBLAS / cublasLt | dense GEMM 기준선·폴백 | `cublas.h`, `tests/CMakeLists.txt:29` |
| vendor lib | curand | 테스트 난수 | `lutGEMM/CMakeLists.txt:13` |
| OpenMP | gomp | 테스트 보조 | `lutGEMM/CMakeLists.txt:13` |
| build | CMake ≥3.18 | 빌드 | `CMakeLists.txt:1` |
| 외부(제외) | googletest (`thirdparty/`) | 테스트 매크로(TEST/RUN_ALL_TESTS) | `main.cc`, `tests.h:8` (분석 제외) |
| Python | torch, transformers, numpy, tqdm | 오프라인 양자화 | `examples/*.py import` |

내부 결합: `kernels.cu`/`nQWeight_fp16.cu`가 `cuda/kernels/*.hpp`를 namespace 안에서 `#include`(`kernels.cu:24-28`). public API는 `include/kernels.h`(`matmul` 4종) + `include/nQWeight_fp16.h`.

---

## 8. 강점 / 한계 / 리스크

### 강점
- **dequant-free GEMV**: activation 부분합을 shared LUT에 1회 선계산, 가중치는 8-bit 인덱스 조회만 → 곱셈을 비트-플레인 스케일 곱(α·부분합) 수준으로 최소화. 메모리-바운드 LLM 디코딩에 적합.
- **비트수 일반성**: 1~8-bit 임의 q를 같은 커널 구조로 지원(`nb` 루프). INT8(8-bit)~INT3까지 테스트(`int3_..cu:307-313`).
- **균일/비균일 통합**: BCQ(비균일) 네이티브, RTN(균일)은 `convert_bcq_format`로 같은 커널 재사용. asymmetric은 q_bias/offset 항으로 흡수.
- **그룹별 스케일**(group_size) 지원으로 정확도-압축 트레이드오프 조절.
- LUT prefix-확장(`mv_fp16.hpp:56-61`)으로 스레드 부족 시에도 곱셈 없이 256엔트리 완성 — 효율적.

### 한계
- **빠른 경로는 M=1 GEMV 한정**. M>1은 dequant+cuBLAS 폴백(`kernels.cu:62,70`)이라 prefill/배치 추론에선 본 기법 이점 없음. `mm_t.hpp`(다중행 LUT GEMM)는 정의만 있고 **public matmul에서 호출되지 않음**(미연결, dead path).
- **FP16 누산 정밀도**: LUT 경로의 `reg_t_o`가 `__half` 누산(`mv_fp16.hpp:77`)이라 K가 큰 레이어에서 오차 누적 우려. (대조적으로 GPTQ faster는 float 누산.)
- **End-to-end 글루 부재**: Python 산출물 직렬화/로더, 모델 전체 추론 통합 코드 없음. 검증용 raw-pointer API와 합성 난수 벤치 위주.
- LUT 인덱싱이 `[kt*4+0..3]` 하드코딩(8-bit×4=32-bit word 가정), atomicAdd 의존(K-블록 충돌) — 큰 K에서 atomic 경합 가능.

### 리스크 (코드 결함)
- `~nQWeight_fp16()`: `cudaFree(alpha)` **두 번**, `bWeight` free 누락(`nQWeight_fp16.cu:102-103`) → 이중 free + 메모리 릭.
- `kernels.h:17` `#pragma ones` 오타(정상은 `#pragma once`) — include guard는 `#ifndef KERNELS_H`로 동작하므로 무해하나 의도와 다름.
- `mv.hpp:87` `nQWeight`(비-fp16) 클래스가 헤더에 정의되어 있지 않음(`nQWeight_fp16`만 존재) → `mv.hpp`/`dequant.hpp`(FP32 legacy)는 현재 빌드 경로(`kernels.cu`)에서 include되지 않아 컴파일 안 됨. legacy 잔재.
- 변수 오타 다수(`num_thraeds`, `Qantized`) — 기능 무해.

---

## 9. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT/Transformer FPGA + XR 시선추적)

우리 타깃은 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**이다. LUT-GEMM은 GPU 코드지만, 그 **핵심 아이디어("곱셈을 LUT 부분합 조회+덧셈으로 치환")는 FPGA에 오히려 GPU보다 더 잘 맞는다.** FPGA는 LUT(룩업테이블)와 BRAM이 1차 자원이기 때문이다.

### 9.1 곱셈-없는 GEMM을 FPGA LUT/BRAM으로 이식
- LUT-GEMM의 shared-memory LUT[256]은 FPGA에서 **BRAM/URAM 또는 distributed LUTRAM**에 직접 대응된다. activation 8개에 대한 부호합 256-엔트리 테이블을 **온칩 BRAM에 1회 적재**하고, 양자화 가중치 비트패턴(8-bit)을 **BRAM read 주소**로 쓰면 DSP 곱셈기 없이 부분합을 얻는다.
- HG-PIPE식 **완전 파이프라인(데이터플로우)** 구조에서, 이 LUT 조회 → 누산(adder tree)은 II=1 파이프라인으로 합성하기 자연스럽다. DSP를 GEMV의 ±1 곱셈에서 해방시켜 **DSP를 LayerNorm/Softmax/GELU 등 비선형부에 재배치**할 수 있다(우리 다른 분석들의 reduction 데이터패스와 정합).

### 9.2 BCQ 양자화의 FPGA 친화성
- BCQ의 `b∈{-1,+1}` 곱은 FPGA에서 **부호 반전(2's complement)+선택**으로 곱셈기 0개. `Σα_i·b_i`의 비트-플레인 누적은 q개의 shift-add(α_i가 RTN처럼 2^i 배수면 더더욱 shift만)로 구현 가능 → **DSP-free GEMV PE** 설계의 직접적 청사진.
- `convert_bcq_format`(RTN→BCQ) 경로는 우리가 이미 가진 INT4/INT8 RTN/GPTQ 양자화 자산을 **BCQ 비트-플레인 표현으로 재해석**해 동일 LUT 하드웨어로 태우는 길을 제공한다(균일·비균일 통합 PE).

### 9.3 부분합 LUT 재사용 = 입력 스테이셔너리 데이터플로우
- LUT-GEMM은 "activation을 한 번 읽어 LUT화 → 여러 출력행이 공유"한다. 이는 FPGA의 **input/activation-stationary** 데이터플로우와 동형. ViT의 한 토큰(또는 patch) activation으로 LUT을 만들고, 그 LUT을 **출력 채널 전체에 broadcast**하면 activation 재로드 0, 가중치 스트리밍만 남아 **off-chip 대역폭을 가중치(압축된 비트)로 한정** — HBM/DDR 대역폭이 병목인 XR용 임베디드 FPGA에 결정적 이점.
- group_size 개념은 BRAM에 올리는 **타일 크기/α 공유 단위**로 직결, K-타일×M-타일 grid는 FPGA의 PE 어레이 타일링과 1:1 매핑 가능.

### 9.4 XR 시선추적 맥락
- 시선추적 추론은 소형 모델·저지연·저전력이 핵심. BCQ 1~4bit + LUT-GEMV는 (a) 모델 메모리를 온칩에 상주시키고(작은 ViT/MLP의 가중치를 BRAM에 fit), (b) DSP-free로 전력을 낮추며, (c) 토큰-단위(M=1) GEMV가 지배적인 워크로드와 정확히 맞물린다. LUT-GEMM이 "M=1에서 최강"이라는 한계가 XR 시선추적에선 오히려 **이상적 적합점**.

### 9.5 주의/차이
- LUT-GEMM의 M>1 폴백(cuBLAS)은 FPGA엔 없음 → ViT prefill/멀티토큰엔 `mm_t.hpp`식 **다중-출력 LUT 재사용**(LUT을 z축으로 확장, `mm_t.hpp:26` `lut[..][..][N_TILE_SIZE]`) 아이디어를 직접 RTL/HLS로 구현해야 우리 처리량 목표를 만족. 이 미연결 커널이 우리에겐 오히려 **가장 참고할 출발점**.
- FP16 누산 정밀도 한계는 FPGA에서 **고정소수 누산 폭을 설계자가 제어**(예: 32-bit 누산)해 회피 가능 — GPU 대비 우리 쪽 이점.

> 결론: LUT-GEMM은 우리에게 "런타임 라이브러리"가 아니라 **알고리즘-아키텍처 청사진**으로서 가치가 크다. 특히 (1) BCQ 비트-플레인 = DSP-free PE, (2) activation 부분합 LUT = BRAM 기반 input-stationary 데이터플로우, (3) `mm_t.hpp`의 다중행 LUT 확장 = ViT 멀티토큰 처리량 설계의 세 가지를 HG-PIPE 파이프라인에 접목할 것.

---

## 10. 근거 / 한계 표기

### 분석에 실제 Read한 파일 (라인 근거 확보)
- `README.md`, `examples/README.md`
- `lutGEMM/include/{lutGEMM, nQWeight_fp16.h, kernels.h}`
- `lutGEMM/src/{nQWeight_fp16.cu, kernels.cu}`
- `lutGEMM/src/cuda/tmpWeight.hpp`
- `lutGEMM/src/cuda/kernels/{mv.hpp, mv_fp16.hpp, mv_fp16_bias.hpp, mm_t.hpp, dequant.hpp, dequant_fp16.hpp, gptq_fp16_bias.hpp, gptq_faster_fp16_bias.hpp, cublas.h}`
- `examples/{bcq.py, bcq_parameter.py, rtn_parameter.py, utils.py, quant_model_bcq.py, quant_model_rtn.py}`
- `tests/{main.cc, include/tests.h, opt/fp16/int3_col_wise_matmul_fp16.cu}`
- `CMakeLists.txt`(root), `lutGEMM/CMakeLists.txt`, `tests/CMakeLists.txt`

### 본 분석에서 다루지 않은 부분 (한계)
- `thirdparty/googletest/**` : 외부 의존성, 의도적 제외(이름만 언급).
- `cublas.cu`, `tests/opt/{_cublas.cc,_cublas.cu}`, `tests/include/{timer.h, custom_random.h, _cublas.h}`, `tests/src/custom_random.cpp` : 벤치 보조/타이머/난수. 기능적 핵심 아님이라 본문에서 라인-인용은 생략(역할만 언급). cublas.cu는 cublas.h와 동일 래퍼로 추정.
- `docs/overview.png` : 바이너리 이미지, 미열람.
- **정량 성능 수치 미확보**: README가 "Table 1"을 언급하나 실제 측정 ms/속도향상 표는 repo에 텍스트로 없고 `./tests/tests` 실행 결과로만 생성됨(bash 도구 미사용 제약 + 빌드/GPU 부재로 미실행). 따라서 본 분석은 코드 구조·알고리즘 근거 중심이며, 절대 성능 배수는 미검증.
- **Python↔CUDA end-to-end 글루**: repo에 직렬화/로더가 없어 두 측 데이터 레이아웃 일치는 코드 정황(uint8 packing ↔ uint32 bWeight, alpha/offset 형상)으로 추론. 런타임 통합 실증은 불가.
- `mv.hpp`/`dequant.hpp`(FP32 legacy): `nQWeight`(비-fp16) 클래스 정의가 repo에 없어 현재 빌드 경로에서 미사용으로 판단(legacy 잔재로 표기). 과거 버전 흔적일 가능성.
