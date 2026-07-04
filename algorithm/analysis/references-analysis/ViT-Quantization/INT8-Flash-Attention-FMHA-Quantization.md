# INT8-Flash-Attention-FMHA-Quantization 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\INT8-Flash-Attention-FMHA-Quantization`
> 분석 방식: 실제 소스(README.md, `inc/fmha_i8.cuh`, `fmha_param_i8.h`, `fmha_i8_quant_deviation.py`) 라인 단위 정독. CUDA 커널은 **자체 작성**(WMMA INT8 Tensor Core).

---

## 1. 개요 (목적/원논문/핵심 아이디어)

- **목적**: Transformer 추론의 Fused Multi-Head Attention(FMHA) 및 Flash-Attention 전체를 8-bit 정수(INT8/UINT8)로 양자화하여 GPGPU에서 가속. FP32 대비 저장공간 4배 절감, 최대 6배 속도(README:62).
- **핵심 아이디어**: Softmax 출력 P의 통계적 성질(0~1 범위, 합=1)을 이용하여 **사전 데이터 지식 없이** Softmax 결과를 UINT8로 양자화. 두 GEMM(QKᵀ, PV)을 모두 INT8 Tensor Core로 수행하고, 그 사이의 Softmax만 FP32로 처리하는 혼합정밀 fused kernel.
- **원논문/정체**: 별도 논문 게재 repo가 아니라 구현 중심 repo로 보임("In this work, we quantize..." README:59). FlashAttention[DFE+22, arXiv:2205.14135] 및 Attention[VSP+17]을 참조로 명시(README:226-237). 정확도 결과는 BERT BASE/LARGE 384(F1 87.4~89.8) 기준 제시(README:206-220). **자체 연구·구현물로 추정**, 정식 논문 ID는 확인 불가.
- **저자 표기**: WMMA 사용법 참조 링크로 `jundaf2/CUDA-INT8-GEMM` 명시(`fmha_i8.cuh`:46).

---

## 2. 디렉토리 구조 (자체 + 제외, 커널 자체/외부 구분)

```
INT8-Flash-Attention-FMHA-Quantization/
├── README.md                      # 이론·수식·결과 (핵심 문서)
├── CMakeLists.txt                 # 빌드
├── test_fmha_i8.cpp               # 테스트 드라이버(미정독, 호스트 측)
├── src/
│   └── fmha_i8.cu                 # 커널 launcher 진입(.cu, 자체)
├── inc/
│   ├── fmha_i8.cuh                # ★ 핵심: INT8 FMHA/Flash 커널(자체 WMMA)
│   ├── fmha_i8.h                  # 호스트 API 선언
│   ├── fmha_param_i8.h            # ★ FMHAParamI8 / AttnDataDescriptor 구조체(자체)
│   ├── cpuGEMM.hpp                # CPU 참조 GEMM(검증용)
│   ├── cpuSoftmax.hpp             # CPU 참조 Softmax(검증용)
│   └── utils.hpp                  # 유틸
├── fmha_i8_quant_deviation.py     # ★ Python 시뮬레이션: 양자화 오차 분석(자체)
├── fmha_i8_quant_error_seqlen.py  # seqlen별 오차 합 시뮬레이션(자체)
└── fig/                           # 결과 그림(error_sum.png 등)
─ 제외: .git/
```

- **커널 구분**: `inc/fmha_i8.cuh`, `src/fmha_i8.cu`는 **자체 작성 CUDA 커널**이다. NVIDIA WMMA(`mma.h`, `wmma::fragment`)와 PTX `cp.async`를 직접 사용하며, cutlass 등 외부 vendor 라이브러리는 사용하지 않음(README:17에서 "using cutlass"는 향후 계획으로만 언급). 외부 vendor 디렉토리 없음.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 양자화 파라미터 구조체 — `inc/fmha_param_i8.h`

```c
struct FMHAParamI8 {       // (fmha_param_i8.h:10-17)
  float q_amax = 0.0f;     // Q의 절대 최댓값(α_q)
  float k_amax = 0.0f;     // K의 절대 최댓값(α_k)
  float v_amax = 0.0f;     // V의 절대 최댓값(α_v)
  float o_amax = 1.0f;     // O의 절대 최댓값(α_o)
  float s_max  = 1.0f;     // softmax 결과 최댓값(이 fused kernel에서는 미사용, README:27)
};
```
- **스케일 흐름의 근간**: Q/K/V/O 각각의 amax(절대 최댓값)를 외부에서 미리 정하여(static quantization) scale = amax/127로 사용. P(softmax 출력)는 amax를 1로 가정하는 정적 방식이거나, 동적(dynamic)으로 row별 amax를 쓸 수 있음(아래 시뮬레이션 참조).

### 3.2 INT8 Flash-Attention 커널 — `inc/fmha_i8.cuh` `FMHAInferKernel<...,USE_TCU=true>` (43-334)

블록당 1개 head를 처리(`blockIdx.x=batch, blockIdx.y=head`, 59-61). Tensor Core 단위 TC_SIZE=16, INT8×INT8→INT32 WMMA `[16,16,16]`(46). NUM_WARPS는 4 또는 8(49), 워프 레이아웃 [NUM_WARPS,1,1].

**스케일 상수 미리 계산** (71-72):
```c
constexpr float row_scale = 1.0f/sqrt(HEAD_DIM);            // 1/√d
const float scale_qk_out = q_amax * k_amax / (127.f*127.f); // (α_q/127)(α_k/127)
```

**(a) GEMM1: S = Q·Kᵀ (INT8 → INT32)** (120-172)
- `cp.async.ca.shared.global`(PTX)로 Q,K를 INT8로 글로벌→공유메모리 더블버퍼 비동기 로드(137-138, 152-153).
- WMMA fragment: `q_int8_frag`(row_major), `k_int8_frag`(col_major)로 로드 후 `wmma::mma_sync`로 INT32 누산(`s_o_int32_frag`)(155-171). HEAD_DIM/TC_SIZE 만큼 K 차원 누적.
- **결과 S는 INT32로 유지**(dequant은 Softmax 단계로 지연).

**(b) Softmax (FP32 혼합 처리)** (174-253) — *정수 GEMM 결과를 FP32로 dequant하여 처리하는 핵심 구간*
- padding mask 로드(177-179).
- **row-wise max**: INT32 누산값에 `row_scale * (·) * scale_qk_out`을 곱해 **FP32 S로 dequant**한 뒤 thread/warp shuffle(`__shfl_xor_sync`, 4-lane)로 reduce max(183-206). 즉 `S_FP32 = S_INT32 · (1/√d) · (α_q/127)(α_k/127)`(README:125와 일치).
- **row-wise exp & sum**: `__expf(row_scale*S_INT32*scale_qk_out - max)`로 P_FP32 계산, mask 위치는 0(218-225), warp reduce sum(229-235).
- **P를 UINT8(실질 INT8)로 양자화**: `softmax_out_scale = -128.f`를 곱해 `p_int8_frag`에 저장(241-250). 부호 트릭으로 `[-128,0]` 범위 사용(`__float2int_rn(max(min(-128*p, 0), -128))`). README의 `P_UINT8 = [255·P]₀²⁵⁵` 수식(137)의 커널 구현 버전. **scale = 1/128(=softmax_out_scale 역수)을 GEMM2 이후에 보정**.

**(c) GEMM2: O = P·V (UINT8/INT8 → INT32)** (255-296)
- V를 INT8로 비동기 로드, `p_int8_frag × v_int8_frag → s_o_int32_frag`(INT32 누산)(283-295).

**(d) Online-softmax rescale (Flash-Attention 누적)** (298-315) — *Flash의 핵심: 블록 간 max/sum 재정규화*
```c
thread_max_new = max(thread_max_old, thread_max);
exp_max_old = __expf(thread_max_old - thread_max_new);
exp_max     = __expf(thread_max     - thread_max_new);
thread_sum_new = exp_max_old*thread_sum_old + exp_max*thread_sum;
o_fp32 = (1/thread_sum_new) * (thread_sum_old*exp_max_old*o_fp32_old
                               + exp_max*(scale_o_fp32 * s_o_int32));  // scale_o_fp32=-1
```
- `scale_o_fp32 = -1.f`(298): P가 음수(-128 스케일)로 양자화되었으므로 -1을 곱해 부호 복원.
- README의 Flash 누적식(180)의 thread-local 구현. O는 FP32 누산기(`o_fp32_frag`)에 유지.

**(e) 최종 dequant & INT8 저장** (318-332)
```c
scale_v_out = v_amax / (o_amax * 128.f);   // (α_v/127)·(1/128)·(127/α_o)에 해당
O_INT8 = clamp(round(scale_v_out * O_FP32), -127, 127);  // char2로 벡터 저장
```
- README의 `O_INT8 = [127/α_o · O_FP32]`(143)에 P의 1/128 보정과 V의 α_v/127을 합친 형태.

**USE_TCU=false 분기는 빈 함수**(337-342) — CUDA Core 구현은 미구현(README "Planning").

### 3.3 Python 양자화 시뮬레이션 — `fmha_i8_quant_deviation.py`

3가지 P 양자화 케이스의 출력 편차를 ground truth(FP32)와 비교(124-126):
- **C1 worst case** (63-76): `p_amax=1` 고정 정적 양자화. `i8_p = uint8(P, clip=1, range=127)`(70), 출력 `f_o1 = i32_o · (1/127) · (v_amax/127)`(73).
- **C2 static quantization** (85-95): `p_amax = max|P|`(전체 최댓값) 정적. 
- **C3 dynamic quantization** (106-118): **row별** `p_amax[row]=max(P[row])`로 per-token(per-row) 동적 양자화, `f_o3 = diag(p_amax/127) · (P_uint8·V_int8) · (v_amax/127)`(118). → **per-token scale의 dequant 시점**: GEMM2 직후 row별 scale을 곱함.
- INT8 GEMM은 `quantize_to_int8(scale=127/clip_max)`(31-37), UINT8은 `[0,255]` 클립(39-44).

---

## 4. 알고리즘 / 수식 (스케일 전파·online softmax 정수화)

README(112-186)의 8-bit Flash-Attention 수식 + 커널 구현 대응:

1. **GEMM1**: `S_INT32 = Q_INT8 · K_INT8ᵀ` → `S_FP32 = S_INT32 · (1/√d) · (α_q/127)(α_k/127)` (커널 71-72,193,218).
2. **블록 Softmax**: `m̃ = rowmax(S_FP32)`, `P_FP32 = exp(S_FP32 - m̃)`, `l̃ = rowsum(P_FP32)` (커널 183-235).
3. **P 양자화**: `P_UINT8 = [255·P_FP32]₀²⁵⁵` (커널은 -128 스케일 사용, 241-250). **Softmax 합 정규화(1/l)를 양자화 단계에서 생략**하고 GEMM2 이후 `1/l`로 한 번에 나눔 — UINT8 범위 최대 활용.
4. **GEMM2**: `Õ_INT32 = P_UINT8 · V_INT8`, `Õ_FP32 = (l̃⁻¹/255)(α_v/127)·Õ_INT32` (커널 283-296,318).
5. **Online rescale (Flash 누적)**: `m_i = max(m_{i-1}, m̃_i)`, `l_i = exp(m_{i-1}-m_i)l_{i-1} + exp(m̃_i-m_i)l̃_i`, O 누적 재정규화(README:168-180, 커널 298-315).
6. **최종**: `O_INT8 = clamp(round((α_v/(α_o·127·128)) · O_FP32), -127, 127)` (커널 318-328).

**핵심 통찰**: P를 UINT8로 양자화할 때 `1/l`(softmax 정규화)을 dequant로 미루면 양자화 범위 손실을 막을 수 있다(README:186 — "Without which one shall lose half of the quantization range before the second GEMM"). 서로 다른 dtype(uchar×char) TCU 사용 시 UINT8 full range 활용 가능(향후 계획, README:15).

---

## 5. 학습/평가 파이프라인 (데이터셋/벤치/명령어)

- **학습 없음(추론·양자화 전용)**. 호스트 측 검증은 `test_fmha_i8.cpp` + `cpuGEMM.hpp`/`cpuSoftmax.hpp`로 CPU 참조와 대조(빌드는 `CMakeLists.txt`).
- **Python 시뮬레이션**: `python fmha_i8_quant_deviation.py`(SEQLEN=512, HEAD_DIM=64, 랜덤 가우시안 입력) → 토큰별 편차 그래프, `fmha_i8_quant_error_seqlen.py` → seqlen별 오차 합.
- **실모델 평가**: BERT BASE/LARGE 384, SQuAD F1/EM(README:201-222). Static vs Dynamic 8-bit 비교.
- **API 사용**: `FMHAInferI8(stream, fmha_param, attn_desc, q,k,v,padding_mask,o, use_tcu=true)`(README:30-47), Q/K/V/O dtype=int8, shape=[batch,head,seq,dim].

---

## 6. 의존성

- CUDA(`cuda.h`, `cuda/barrier`, `cooperative_groups`, `mma.h`), SM에서 INT8 WMMA + `cp.async` 지원 필요(Ampere급, sm_80+ 추정).
- HEAD_DIM ∈ {64,128}, SEQ_LEN은 64의 배수 지원(README:6-12).
- Python 시뮬레이션: numpy, matplotlib.
- 빌드: CMake.

---

## 7. 강점 / 한계 / 리스크

**강점**
- 두 GEMM 모두 INT8 Tensor Core로 fused 처리하면서 Softmax 정규화를 dequant로 지연 → 양자화 범위 손실 최소화(정확도 약 2배 개선, README:59).
- online-softmax rescale을 thread-local FP32 누산기로 구현 → Flash-Attention IO 효율 유지.
- static/dynamic(per-token) 양자화를 시뮬레이션으로 정량 비교 제공.

**한계 / 리스크**
- USE_TCU=false(CUDA Core) 미구현. uchar×char hybrid TCU도 계획 단계.
- Seq len SRC≠DST 미지원, seqlen 64배수 제약.
- P 양자화에 `-128` 부호 트릭 사용 → UINT8 0~255 full range가 아니라 사실상 7-bit 활용(README:186이 지적하는 한계와 동일, hybrid TCU 미적용 시).
- amax(α_q,α_k,α_v,α_o)를 외부에서 정해야 함(calibration 필요). 동적 모드는 row reduce 오버헤드.
- 오차가 seqlen 증가에 따라 누적 증가(README:196 — error sum increases with seq len).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적)

- **FPGA 어텐션 데이터패스에 직접적**: 두 GEMM을 INT8 정수 누산(INT32)으로 유지하고, **Softmax만 부분적으로 FP/고정소수점으로 처리**하는 구조는 FPGA에서 systolic INT8 PE 배열 + 별도 Softmax 유닛으로 매핑하기 좋은 데이터플로우. `scale_qk_out = (α_q/127)(α_k/127)·(1/√d)`를 **상수 곱(shift+mul)으로 미리 합쳐** 1회만 적용하는 방식은 FPGA에서 scale 전파를 단순화(DSP 절감).
- **Online-softmax 정수화/재정규화**: `m_new=max`, `l_new=exp(Δm)·l_old + ...`의 블록 간 재정규화 패턴은 FPGA 스트리밍 어텐션(HG-PIPE의 layer-by-layer 파이프라인)에서 행 단위 max/sum 레지스터로 구현 가능. exp는 LUT/PWL로 근사.
- **P를 1/l 지연 양자화**: softmax 정규화를 마지막에 한 번 곱하는 트릭은 FPGA에서 중간 P를 저비트로 유지해 PV GEMM 대역폭·BRAM을 절감하는 직접적 설계 지침.
- **XR 시선추적 관점**: 저지연·저전력 추론이 핵심인 XR eye-tracking ViT에서 INT8 어텐션 + 정적 amax calibration은 실시간성 확보에 유리. 단 seqlen 증가 오차 누적 특성은 짧은 토큰(작은 패치 그리드) 시선추적에 오히려 적합.

---

## 9. 근거 표기 / 불명확 사항

- CUDA 커널은 **자체 작성 확인**(WMMA/PTX 직접 사용, cutlass 미사용). 외부 vendor 없음.
- 정식 논문/저자명은 **확인 불가**(README에 저자 명시 없음, jundaf2 참조 링크만 존재 — 동일 저자 추정이나 단정 불가).
- `s_max` 필드는 정의되어 있으나 fused kernel에서 미사용(README:27 명시) — P 동적 양자화 확장용으로 **추정**.
- `test_fmha_i8.cpp`/`src/fmha_i8.cu` launcher 본문은 미정독(커널 헤더 `fmha_i8.cuh`로 알고리즘 전모 파악 완료).
