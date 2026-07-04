# int-flashattention (INT-FlashAttention, 2024) 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\int-flashattention`
> 분석 방식: README.md, Triton 커널 3종(`flash_atten_int8.py`, `flash_atten_full_int8.py`, `flash_atten_fp.py`), `benchmark.py`, `configs.py`, `csrc/README.md`, CUDA 커널 헤더 라인 정독.
> **정체**: arXiv 2409.16997 "INT-FlashAttention: Enabling Flash Attention for INT8 Quantization" 공식 구현(README:1-2). 핵심 커널 언어 = **Triton(.py, 자체 핵심)** + CUDA(.cu/.h, 대부분 Tri Dao flash-attention vendor 파생).

---

## 1. 개요 (목적/원논문/핵심 아이디어)

- **목적**: FlashAttention을 INT8 양자화와 결합. Q/K(및 full 버전은 V)를 INT8로, 두 GEMM(QKᵀ, PV)을 정수 연산으로 수행하여 Ampere INT8 Tensor Core 활용 + IO 효율 유지.
- **원논문**: INT-FlashAttention(2024). 공식 구현임을 README가 명시("official implementation", README:2).
- **핵심 아이디어**:
  - **per-token(행별) scale**: Q·scale, K·scale을 토큰 단위 벡터로 두어 INT8 GEMM 결과(INT32)에 dequant.
  - **online-softmax는 FP32에서 수행**(numerically stable), V는 두 가지 모드 — (a) `int8` 버전: Q/K만 INT8, V는 FP16; (b) `full_int8` 버전: V까지 INT8로 P를 INT8 양자화하여 PV도 정수 GEMM.

---

## 2. 디렉토리 구조 (자체 + 외부 구분)

```
int-flashattention/
├── README.md                       # 개요(자체)
├── flash_atten_fp.py               # FP16/FP8 기준 FlashAttention(Triton, OpenAI tutorial 파생)
├── flash_atten_int8.py             # ★ INT8(Q,K) FlashAttention(Triton, 자체 핵심)
├── flash_atten_full_int8.py        # ★ full INT8(Q,K,V) FlashAttention(Triton, 자체 핵심)
├── benchmark.py                    # 성능/정확도 벤치(자체) — quant_pertoken/pertensor 포함
├── configs.py                      # Triton autotune config(자체)
└── csrc/                           # CUDA 버전 (대부분 외부 vendor 파생)
    ├── README.md                   # "Code comes from Dao-AILab/flash-attention v2.5.7"(외부 명시)
    ├── flash_fwd_kernel.h, softmax.h, mask.h, dropout.h, rotary.h, alibi.h,
    │   kernel_traits*.h, block_info.h, utils.h, philox.cuh, static_switch.h, flash.h
    │   flash_fwd_hdim*_{fp16,bf16}_sm80*.cu  # ← 전부 Tri Dao vendor 코드(cutlass/cute 의존)
    ├── flash_fwd_kernel_full_qi8.h # ★ INT8 변형 커널(Tri Dao 베이스에 자체 INT8 추가)
    ├── flash_fwd_kernel_half_qi8.h # ★ half-INT8 변형(동일)
    └── flash_fwd_hdim128_*_{full,half}_qi8.cu  # 위 변형의 인스턴스화
─ 제외: .git/
```

- **커널 자체/외부 구분**:
  - **자체 핵심 = Triton .py 3종**(`flash_atten_int8.py`, `flash_atten_full_int8.py` 및 FP 기준). 정밀 분석 대상.
  - `csrc/`의 대다수 .cu/.h는 **외부 vendor(Tri Dao flash-attention v2.5.7, commit 85881f5)에서 복사**(`csrc/README.md`:1, 헤더 `Copyright (c) 2024, Tri Dao`). cutlass/cute(`cute/algorithm/copy.hpp`, `cutlass/cutlass.h`) 의존.
  - `*_qi8.h`(`flash_fwd_kernel_full_qi8.h`, `_half_qi8.h`)는 Tri Dao 베이스 위에 **자체 INT8 경로(`Element_Int8`, `compute_attn_1rowblock_full_qi8`)를 추가한 파생물**. cutlass/cute 의존(vendor) + 자체 INT8 로직 혼재.

---

## 3. 핵심 모듈·파일별 정밀 분석 (Triton 자체 커널)

### 3.1 `flash_atten_int8.py` — Q,K만 INT8, V는 FP16

**inner loop `_attn_fwd_inner_int8`** (9-66):
- K 블록과 **per-token K scale** 로드(33-34): `k = load(K_block_ptr)`, `k_scale = load(K_block_scale_ptr)`.
- **INT8 GEMM → dequant**(35-37):
  ```python
  qk = tl.dot(q, k).to(tl.float32)   # INT8×INT8 = INT32 누산 후 FP32 캐스트
  qk = qk * q_scale[:, None]          # per-token(행) Q scale 곱
  qk = qk * k_scale                   # per-token(열, K) scale 곱
  ```
  → **dequant 시점 = GEMM 직후, online-softmax 직전**. scale은 토큰 단위 외적(q_scale[:,None] × k_scale).
- **online-softmax(FP32)**(38-50): exp2 기반.
  - causal(STAGE==2): `qk = qk*qk_scale + mask`, `m_ij = max(m_i, max(qk,1))`, `qk -= m_ij`.
  - 비causal: `m_ij = max(m_i, max(qk,1)*qk_scale)`, `qk = qk*qk_scale - m_ij`.
  - `p = exp2(qk)`, `l_ij = sum(p,1)`, `alpha = exp2(m_i - m_ij)`, `l_i = l_i*alpha + l_ij`, `acc = acc*alpha`(rescale).
- **PV(혼합)**(54-59): `p`를 FP16(또는 fp8e5)로 캐스트 후 `acc = tl.dot(p, v, acc)` — **V는 FP16이므로 PV는 정수 GEMM 아님**.
- epilogue(171-176): `m_i += log2(l_i)`, `acc = acc / l_i`. M(LSE) 저장.

**`qk_scale` 처리**(148-149): `qk_scale = sm_scale * 1.44269504`(=1/ln2) → exp 대신 **exp2 사용을 위한 base 변환**. (FPGA에서 exp2가 shift로 분해되는 점과 연결.)

**호스트 `_attention_int8.forward`** (178-215): q,k=int8, v=fp16, q_scale,k_scale=fp16 입력. grid=(cdiv(N, BLOCK_M), Z*H). HEAD_DIM ∈ {16,32,64,128,256}(187).

### 3.2 `flash_atten_full_int8.py` — Q,K,V 전부 INT8 (★ full 정수 경로)

**inner `_attn_fwd_inner_full_int8`** (9-71): QKᵀ dequant·softmax는 위와 동일(34-48). 차이는 **P→INT8 양자화 후 PV 정수 GEMM**:
```python
p = p.to(tl.float16)
p = p * 127                  # (49-51) softmax 출력 P(0~1)를 [0,127]로
p = (p + 0.5).to(tl.int8)    # round → INT8
...
v = tl.load(V_block_ptr)     # INT8 V
tmp = tl.dot(p, v)           # (59) INT8×INT8 = INT32 PV GEMM
tmp = tmp.to(tl.float32)
tmp = tmp * v_scale / 127    # (61) per-(batch,head) V scale + P의 1/127 보정으로 dequant
acc = acc + tmp              # (62) online 누적
```
- **P 양자화 방식**: scale = 1/127 고정(P∈[0,1] 가정). zero-point 없음(대칭, 사실상 unsigned 7-bit 사용). **dequant 시점 = PV GEMM 직후**, `v_scale/127` 곱.
- **V scale은 per-(batch,head)** (텐서 단위, benchmark의 `quant_pertensor`와 일치, 아래 3.4).
- epilogue(180-182): `acc = acc / l_i`(여기선 m_i 보정 저장 생략).

### 3.3 `flash_atten_fp.py` — FP16/FP8 기준 구현
- OpenAI Triton FlashAttention tutorial 구조의 FP 기준(벤치 비교용). int8 버전들의 베이스. (정수화 없음, 상세 생략.)

### 3.4 `benchmark.py` — 양자화 헬퍼 & 정확도 검증

- **per-token 양자화** `quant_pertoken(X)`(96-100): `X_max = max|X|(dim=-1)`(마지막 dim), `scale = X_max/127`, `round(X/scale).int8`. → Q/K에 사용.
- **per-tensor 양자화** `quant_pertensor(X)`(102-107): head_dim·seq 양축 max, `scale=max/127`. → V에 사용.
- 정확도 테스트 `acc_test`(133-187): q8,k8=pertoken, v8=pertensor → `attention_int8`/`attention_full_int8` 출력과 FP ref의 MRE(mean relative error) 비교(176-185). FP8과도 비교.
- 성능 벤치(48-94): N_CTX=2¹⁰~2¹⁴, BATCH=4,H=32,HEAD_DIM=64. provider: triton-fp16/fp8/int8/full-int8/flash(공식).

### 3.5 `configs.py` — Triton autotune
- BLOCK_M,BLOCK_N ∈ {32,64,128,256}, num_stages ∈ {3,4,7}(HIP은 1), num_warps ∈ {4,8}(13-19). `keep` 필터로 BLOCK_M·BLOCK_N<128² & num_warps==8 조합 제외(30-35).

### 3.6 CUDA 변형 `csrc/flash_fwd_kernel_full_qi8.h` (외부+자체 혼재)
- `compute_attn_1rowblock_full_qi8`(28): Tri Dao의 `compute_attn_1rowblock` 구조에 `Element_Int8`, `kGmemScaleElemsPerLoad`, `kThrsScaleUsed{M,N,H}`(45-48) 등 **INT8 scale 로딩 멤버 추가**. cute/cutlass `make_tensor`/`local_tile` 사용(vendor 인프라). 빌드는 공식 flash-attention 참조 필요(README:12).

---

## 4. 알고리즘 / 수식 (scale 전파·online-softmax·per-token/tensor scale)

INT8 FlashAttention 1행블록(per query block):

1. **QKᵀ dequant**: `S = (Q_int8 · K_int8ᵀ)_int32`, `S_fp = S · s_q[i] · s_k[j]` (per-token 외적; int8.py:35-37).
   - 추가로 `qk_scale = sm_scale / ln2`를 곱해 exp2 도메인으로 변환(148-149).
2. **online-softmax(FP32)**: `m_ij = max(m_i, max_j S_fp·qk_scale)`, `P = 2^(S_fp·qk_scale - m_ij)`, `l_ij = Σ P`, `α = 2^(m_i - m_ij)`, `l_i ← l_i·α + l_ij`, `acc ← acc·α`(38-52).
3. **PV**:
   - **half-int8**(int8.py): `P_fp16`, `acc ← P·V_fp16`(정수 아님).
   - **full-int8**(full_int8.py): `P_int8 = round(P·127)`, `O_int32 = P_int8 · V_int8`, `O_fp = O_int32 · s_v / 127`, `acc += O_fp`(49-62).
4. **정규화**: `O = acc / l_i`(epilogue). int8.py는 LSE `m_i += log2(l_i)` 저장.

**Scale 종류 요약**:
| 텐서 | scale 단위 | 적용 시점 |
|---|---|---|
| Q | per-token(행), s_q[:,None] | QKᵀ 직후 |
| K | per-token(열), s_k | QKᵀ 직후 |
| P | 고정 1/127 | PV 직후(full-int8만) |
| V | per-(batch,head) per-tensor, s_v | PV 직후(full-int8만) |

---

## 5. 학습/평가 파이프라인 (데이터셋/벤치/명령어)

- **학습 없음(커널·추론 벤치 전용)**.
- 실행: `python benchmark.py` → 기본 `acc_test(BATCH=1,H=1,N_CTX=32,HEAD_DIM=32)` 정확도 출력(212). 성능 벤치는 `bench_flash_attention.run(...)` 주석 해제(193).
- pytest: `test_op`(214-250) — causal FlashAttention vs torch ref allclose(FP 경로 정합성).
- 비교 대상: Triton FP16/FP8/int8/full-int8 + 공식 Flash-2(`flash_attn_qkvpacked_func`, 가능 시).
- 입력: 랜덤(`torch.randint(-128,127, int8)` for q/k/v, `randn` for scale).

---

## 6. 의존성

- **Triton**(핵심), PyTorch, Ampere+ GPU(INT8 Tensor Core). pandas/numpy(벤치).
- 선택: `flash_attn`(공식 Flash-2 비교용), `torch.float8_e5m2`(FP8 비교).
- CUDA 변형 빌드: cutlass/cute + 공식 Dao-AILab flash-attention 빌드 시스템(외부).

---

## 7. 강점 / 한계 / 리스크

**강점**
- per-token Q/K scale + per-tensor V scale로 INT8 GEMM 정밀도 유지. exp2 기반 softmax로 효율적.
- full-int8 경로는 PV까지 정수화 → INT8 Tensor Core 최대 활용(메모리·연산 모두 절감).
- 공식 Flash-2/FP8과 직접 MRE·속도 비교 가능한 벤치 제공.

**한계 / 리스크**
- P 양자화가 scale=1/127 고정(P 최댓값<1 가정) → P 분포가 sharp하면 작은 값 절단(저비트 손실). per-row P scale은 미적용(앞 repo의 dynamic case와 대조).
- online-softmax는 여전히 FP32(정수-only 아님) — 어텐션의 비선형부는 부동소수.
- CUDA 경로는 vendor(cutlass) 강결합 → FPGA 직접 이식 어려움. Triton 경로가 알고리즘 참조용으로 더 유용.
- `benchmark.py`의 V scale을 `randn`(음수 포함)으로 생성 — 실제는 abs-max scale이어야(MRE 검증 시 주의).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적)

- **per-token scale 외적의 FPGA 매핑**: `S_fp = S_int32 · s_q[i] · s_k[j]`는 INT8 systolic 어레이 출력에 **행/열 scale 벡터를 곱하는 후처리 스테이지**로 자연스럽게 구현. HG-PIPE류 파이프라인에서 각 토큰 scale을 BRAM 벡터로 보관하면 됨.
- **exp2 도메인 변환(`·1/ln2`)**: FPGA에서 exp는 `2^x = 2^⌊x⌋ · 2^frac`로 분해해 정수 시프트 + 작은 LUT로 구현 가능 → 본 repo가 exp2를 쓰는 것은 HW 친화적 설계와 정확히 일치. **softmax 비선형부 LUT 비트 설계의 직접 참조**.
- **full-int8 PV의 P=int8(1/127) 양자화**: P를 7-bit로 두고 PV를 INT8 GEMM으로 처리하는 패턴은 FPGA 어텐션의 두 번째 GEMM도 INT8 PE로 통일 가능함을 보여줌(데이터패스 동질화 → 자원 재사용).
- **XR 시선추적**: 짧은 시퀀스·작은 head_dim(32/64)에서 full-int8 정확도 손실이 작다면 XR ViT 실시간 추론에 적합. 단 P 고정 scale의 분포 민감성은 시선추적의 sharp attention(특정 패치 집중)에서 검증 필요 — per-row P scale(앞 FMHA repo) 채택을 비교 고려.

---

## 9. 근거 표기 / 불명확 사항

- **정체 확인**: INT-FlashAttention 공식 구현(README:1-2). Triton .py가 자체 핵심.
- **커널 출처 확인**: `csrc/`의 비-qi8 파일은 Tri Dao flash-attention v2.5.7 vendor 코드(`csrc/README.md`:1, 헤더 copyright). `*_qi8.h`는 vendor 베이스 + 자체 INT8 추가(혼재) — cutlass/cute 의존.
- CUDA `*_qi8.h`의 INT8 dequant 세부 수식은 헤더 상단(scale 멤버)까지만 정독, 본문 mainloop 전체는 vendor 구조와 동일 가정(추정). 알고리즘 전모는 Triton 커널로 충분히 파악됨.
- `benchmark.py`의 V scale을 `randn`으로 생성하는 것은 벤치 편의(정확도보다 shape/속도 측정 목적)로 **추정**.
