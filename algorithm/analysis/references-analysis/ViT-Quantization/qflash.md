# QFlash 정밀 분석 (Integer-only FlashAttention, Triton)

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\qflash`
> 작성일: 2026-06-20 / 실제 소스 코드 기반. 라인 근거(파일:라인) 표기.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: Vision Transformer 어텐션을 **정수(INT8) 전용 FlashAttention 커널**로 가속하면서, FlashAttention의 메모리 효율(타일링 + online softmax)을 유지하는 것. (README.md:1-6)
- **원논문**: *QFlash: Bridging Quantization and Memory Efficiency in Vision Transformer Attention*, Sehyeon Oh, Yongin Kwon, Jemin Lee, **IJCAI-ECAI 2026** (공식 구현). (README.md:1-6, 58-64)
- **핵심 아이디어**:
  1. Q·K·V를 per-tensor symmetric INT8로 양자화하고, **attention 전 구간(QK^T → softmax → P·V)을 정수 도메인에서** 수행.
  2. softmax의 지수 함수를 **부동소수 없이** `exp2`의 정수 근사(`exp2_i32`)로 구현.
  3. online softmax의 스케일 보정(`alpha`, running max/sum/acc)을 모두 **고정소수점(fixed-point) shift 연산**으로 처리.
  4. 모든 dequant 스케일은 호스트(Python)에서 `(1<<SHIFT)` 곱으로 정수 상수화해 커널에 전달 → 커널 내부는 정수 곱·시프트만. (kernel.py:300-307)
- 정확도: ViT/DeiT(A2) SQNR 32.50 dB, Swin(A7) 31.02 dB. ImageNet Top-1은 FP32 대비 동등 또는 상회(예: ViT-B 85.10→86.84). (README.md:27-53)

---

## 2. 디렉토리 구조 (자체 소스)

```
qflash/
├── qflash/
│   ├── __init__.py        # 공개 API: qflash, QAttention, QWindowAttention, patch_attention
│   ├── kernel.py          # ★ Triton 정수 FlashAttention 커널 + 호스트 양자화 로직
│   └── attention.py       # ★ nn.Module 래퍼(QAttention/QWindowAttention) + timm 모델 패칭
├── benchmark.py           # 커널 단위 SQNR/지연시간 벤치 (A1~A7 워크로드)
├── validate.py            # ImageNet Top-1 평가(timm 모델에 패칭 후)
├── pyproject.toml         # torch>=2.7.1, timm 의존
├── README.md
├── LICENSE (Apache-2.0)
└── docs/benchmark.png
```
*제외*: `.git/` (분석 제외).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `qflash/kernel.py` — 정수 FlashAttention 커널 (가장 중요)

#### (a) 환경변수/오토튠 설정 (kernel.py:30-37)
- `QFLASH_AUTOTUNE`(default/fast), `QFLASH_CUDA_GRAPH`, `EXP_USE_SHIFT`/`L_USE_SHIFT`/`ACC_USE_SHIFT` 토글, `L_SHIFT`/`ACC_SHIFT`(기본 30).
- `get_fast_config`(kernel.py:12-23): BLOCK_M∈{32,64}, BLOCK_N∈{32,64}, `BM<=BN` 제외(즉 BM>BN), warps∈{2,4}, stages∈{2,4} 조합으로 autotune config 생성. `get_default_config`는 32/32, warps=1, stages=1 단일.

#### (b) 정수 지수함수 `exp2_i32` (kernel.py:40-59) — 양자화 online-softmax의 심장
- `exp2_i32_shift(x, x_rscale, exp_M, SHIFT=16)` (kernel.py:41-44):
  - `q = (x * exp_M) >> SHIFT` : 입력 x(정수 logit)에 exp_M(=`-sm_scale*(1<<SHIFT)`)를 곱하고 시프트 → 2의 지수부 정수.
  - `r = x + q * x_rscale` : 나머지(소수부 보정).
  - 반환 `(((r >> 1) + x_rscale) >> q)` : 2^(-q) 형태로 우시프트하여 정수 도메인 exp2 근사.
- `exp2_i32_idiv` (kernel.py:47-51): shift 대신 정수 나눗셈(`//`) 사용하는 대안 경로(USE_SHIFT=False).
- 즉 softmax의 `exp` 계산이 **부동소수 없이** 정수 곱셈 + 우시프트로 수행됨 → FPGA/ASIC softmax 매핑에 직접 시사.

#### (c) 재양자화/스케일 릴리스 헬퍼 (kernel.py:62-82)
- `requantize(x, req_M, SHIFT=16)` (kernel.py:62-64): `((x*req_M + (1<<(SHIFT-1))) >> SHIFT).to(int8)` — 라운딩 후 INT8로 재양자화(p, 즉 확률값을 int8로).
- `scale_release_shift(x, alpha, M, SHIFT)` (kernel.py:67-69): `x.to(int64)*alpha*M` 후 라운드 시프트하여 int32 — running sum/acc를 alpha로 보정하는 곱셈 경로.
- `scale_release_idiv` (kernel.py:72-74): `(x*alpha)//sm_rscale` 정수 나눗셈 경로.

#### (d) 메인 커널 `qflash_kernel` (kernel.py:85-180)
- 데코레이터: `@triton.autotune(key=['N_CTX','HEAD_DIM'])` + `@triton.jit`.
- 인자: Q,K,V,Out(모두 INT8), 정수 스케일 상수 `sm_rscale, exp_M, req_M, l_M, acc_M`, 각 텐서 strides, Z(배치×?), H(헤드). (kernel.py:87-104)
- 블록 포인터 구성: Q/K/V/Out을 `make_block_ptr`로 타일링. **K는 (HEAD_DIM, N_CTX) 전치 레이아웃**으로 로드(QK^T를 dot로 바로). (kernel.py:114-133)
- online softmax 상태 초기화 (kernel.py:135-140):
  - `m_i = NEG_INF(-(1<<20))` (running max, int32)
  - `l_i = 1` (running sum, int32 — 0이 아니라 1로 시작, 후술 나눗셈 안정화)
  - `acc = 0` (int32 누적기)
- **메인 루프** (kernel.py:147-174):
  1. `qk = tl.dot(q, k, out_dtype=tl.int32)` — INT8×INT8 → INT32 MAC(QK^T). (kernel.py:153)
  2. (마스크 없으면) 패딩 처리(kernel.py:154-156).
  3. `m_ij = max(m_i, max(qk,1))`, `qk -= m_ij` — online max 안정화. (kernel.py:158-159)
  4. `p = exp2_i32(qk, sm_rscale, exp_M)` — **정수 exp2로 softmax 분자**. (kernel.py:160)
  5. `p = requantize(p, req_M)` — p를 INT8로 재양자화. (kernel.py:161)
  6. `l_ij = sum(p,1)` — 블록 내 확률 합. (kernel.py:162)
  7. `alpha = exp2_i32(m_i - m_ij, ...)` — 이전 블록 보정계수. (kernel.py:163)
  8. `l_i = scale_release(l_i, alpha, ...) + l_ij` — running sum 갱신. (kernel.py:164)
  9. `acc = scale_release(acc, alpha[:,None], ...)` — running acc 보정. (kernel.py:165)
  10. `p = p.to(int8); acc = tl.dot(p, v, acc, out_dtype=int32)` — **INT8 P × INT8 V → INT32 누적(P·V)**. (kernel.py:170-171)
- 종료: `acc = (acc / l_i[:,None]).to(int32)` — 정수 나눗셈으로 softmax 정규화 후 출력 dtype으로 저장. (kernel.py:176-180)

#### (e) 마스크 버전 `qflash_masked_kernel` (kernel.py:183-272)
- 위와 동일하나 `attn_bias`(INT32)를 블록 단위로 로드(kernel.py:250)하여 `qk = where(padding, qk + bias, NEG_INF)`로 더함(kernel.py:253). Swin의 relative position bias / window mask 지원.

#### (f) 호스트 진입점 `qflash_forward` (kernel.py:275-351) — scale/dequant 위치 핵심
- HEAD_DIM 검증(16/32/64/128/256), `scale = 1/sqrt(D) * LOG2E`(log2e=1.44269504로 exp→exp2 변환). (kernel.py:283-292)
- **per-tensor amax 기반 symmetric 스케일** (kernel.py:294-297):
  - `q_scale = query.abs().amax() / 127`, `k_scale`, `o_scale`(=value의 amax/127) 각각.
  - `qk_scale = q_scale * k_scale`, `sm_scale = scale * qk_scale`.
- **고정소수점 상수화** (kernel.py:300-307):
  - `SHIFT = 16`.
  - `sm_rscale = round(1/sm_scale)` (역스케일, exp2 나눗셈용).
  - `exp_M = round(-sm_scale * (1<<16))`.
  - `req_M = round(sm_scale * 127 * (1<<16))`.
  - `l_M = round((1<<L_SHIFT)/sm_rscale)`, `acc_M = round((1<<ACC_SHIFT)/sm_rscale)`.
- **양자화** (kernel.py:309-311): `query = clamp(round(query/q_scale), -128,127).to(int8)` 동일하게 key, value.
- 출력: `return o * o_scale` (kernel.py:351) — **최종 dequant는 호스트에서 value 스케일(o_scale)만 곱**(softmax 분모로 이미 정규화되었기 때문에 V 스케일만 복원). → dequant 위치: 입력 양자화는 호스트, 누적/softmax는 커널 정수, 최종 V-스케일 복원은 호스트.
- `return_mode='latency'`면 `triton.testing.do_bench`로 median 지연 반환. (kernel.py:348-349)
- 클래스 `qflash` (kernel.py:354-356): `kernel`(2개 커널), `forward`(=qflash_forward) 묶음.

### 3.2 `qflash/attention.py` — nn.Module 래퍼 / 모델 패칭

- `QAttention(nn.Module)` (attention.py:12-36): timm `Attention` 대체. `qkv = Linear(dim, dim*3)`, `proj = Linear(dim,dim)`. forward에서 qkv → (q,k,v) 분리 후 `qflash.forward(q,k,v,attn_mask)` 호출, proj. `from_orig`로 기존 모듈의 state_dict 그대로 로드(가중치는 FP 유지, 어텐션 내부만 정수). (attention.py:20-36)
- `QWindowAttention(nn.Module)` (attention.py:39-86): Swin window attention 대체. relative_position_bias_table + window mask를 `attn_mask`로 합산하여 masked 커널에 전달. (attention.py:63-75)
- `patch_attention(model)` (attention.py:89-108): timm 모델을 순회하며 `timm.layers.attention.Attention`→QAttention, `swin.WindowAttention`→QWindowAttention으로 in-place 교체. 교체 카운트 반환. → **PTQ 방식의 어텐션-only 정수화**(가중치 재학습 없음, attention 연산만 INT8 커널로 치환).

### 3.3 `benchmark.py` — 커널 SQNR/지연 벤치
- 워크로드 A1~A7(ViT-T/S/B, Swin-1~4)의 (Z,H,N,D) 정의(benchmark.py:6-14). FP16 임의 입력으로 `qflash.forward` vs `F.scaled_dot_product_attention` 비교, SQNR(dB) = `20*log10(||ref||/||out-ref||)` 계산(benchmark.py:23-28), batch∈{1,8} 지연 측정(benchmark.py:31-34).

### 3.4 `validate.py` — ImageNet Top-1 평가
- timm 모델 7종(ViT-S/B, DeiT-T/S/B, Swin-T/S) 생성 → `patch_attention`으로 정수 어텐션 패칭 → ImageNet val Top-1/Top-5 측정(validate.py:17-25, 57-126). `--num-samples`로 서브셋 sanity 가능. 결과 JSON/CSV 저장.

---

## 4. 알고리즘 / 수식 — Quantized online-softmax FlashAttention

표준 FlashAttention online softmax(블록 j 갱신):
```
m_ij = max(m_i, rowmax(S_j))          # S_j = Q_i K_j^T
P_j  = exp(S_j - m_ij)
alpha = exp(m_i - m_ij)
l_i  = alpha * l_i + rowsum(P_j)
O_i  = alpha * O_i + P_j V_j
```
QFlash의 정수화 매핑:
1. **양자화**: `Q=round(q/sq)`, `K=round(k/sk)`, `V=round(v/sv)` (INT8, per-tensor amax/127). (kernel.py:294-311)
2. **S_j(QK^T)**: `tl.dot(Q,K) → INT32`. dequant 스케일 `sm_scale = sq·sk·(1/√D)·log2e`는 정수 상수 `exp_M, sm_rscale`로 흡수. (kernel.py:153, 300-303)
3. **exp → exp2 정수근사**: `P = exp2_i32(S - m_ij)` (kernel.py:160). exp2 선택 이유: `exp(x)=2^(x·log2e)`이므로 log2e를 sm_scale에 미리 곱해 2의 거듭제곱 = 우시프트로 구현 가능 → 부동소수 exp 제거.
4. **P 재양자화**: `P = requantize(P, req_M) → INT8` (kernel.py:161).
5. **alpha 보정**: `alpha = exp2_i32(m_i - m_ij)`, `l_i`/`acc`를 `scale_release`(고정소수점 곱·시프트)로 보정. (kernel.py:163-165)
6. **P·V**: `tl.dot(P_int8, V_int8) → INT32` 누적. (kernel.py:171)
7. **정규화/dequant**: `O = acc / l_i` (정수 나눗셈), 호스트에서 `O * o_scale` 복원. (kernel.py:176, 351)

핵심: softmax의 max-subtraction(수치 안정), exp, running rescale가 **전부 정수+시프트**. `l_i` 초기값 1(kernel.py:139)은 0 나눗셈 회피용 안정화.

---

## 5. 학습/평가 파이프라인

- **데이터셋**: ImageNet-1k val (`~/dataset/imagenet` 기본). (validate.py:27)
- **명령어**:
  - 커널 벤치: `python benchmark.py` (SQNR + 지연). (README.md:18-22)
  - 정확도: `python validate.py --all` 또는 `python validate.py -m DeiT-T --num-samples 1000`. (README.md:38-41)
- **학습 불필요**: 사전학습 timm 가중치에 어텐션만 패칭하는 **PTQ(training-free)**. 별도 calibration 데이터셋 없이 forward 시 입력 텐서의 amax로 동적 per-tensor 스케일 산출.
- **배포(TRT/ONNX 등)**: 본 repo에는 없음(Triton/CUDA 커널 전용). ONNX/TFLite 익스포트 코드 **확인 불가(없음)**.

---

## 6. 의존성 (pyproject.toml:9-12)
- `torch>=2.7.1` (CUDA 12.8, RTX 5090 검증; README.md:14), `timm`.
- 런타임 `triton`(kernel.py:5-6, torch에 동봉). Python>=3.10. 라이선스 Apache-2.0.

---

## 7. 강점 / 한계 / 리스크

**강점**
- 어텐션 전 구간(QK^T, softmax, P·V)을 **완전 정수**로 수행 — softmax까지 정수화한 사례는 드묾.
- FlashAttention 타일링 유지 → 메모리 효율 + 정수 연산 동시 확보.
- 사전학습 모델에 in-place 패칭만으로 적용(training-free). SQNR 31~32 dB, Top-1 유지/상회.
- 모든 dequant를 고정소수점 정수 상수로 흡수 → 부동소수 unit 불필요.

**한계 / 리스크**
- **per-tensor amax 동적 스케일**: 매 forward마다 `abs().amax()` 계산(kernel.py:294-296) → 호스트 측 추가 연산, outlier에 민감(per-channel/percentile 아님).
- HEAD_DIM 제한(16/32/64/128/256), `BLOCK_N <= HEAD_DIM` 정적 단언(kernel.py:105). 작은 head_dim에서만 검증된 ViT/Swin 형상 중심.
- 가중치(qkv/proj Linear)는 FP 유지 — **어텐션 행렬곱만 정수**(Linear GEMM은 비양자화). 전체 모델 정수화 아님.
- exp2 정수 근사 오차가 SHIFT(=16/30) 비트폭에 의존 → 비트폭 축소 시 정확도 저하 가능.
- Triton/특정 GPU(RTX 5090, CUDA 12.8) 의존. CPU/엣지/FPGA 직접 이식 코드는 없음.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적 — 추정)

- **softmax 정수화 직접 시사**: `exp2_i32`(kernel.py:40-59)는 **FPGA에서 부동소수 exp 없이 softmax를 시프트+정수 곱으로** 구현하는 청사진. log2e를 스케일에 흡수 → exp를 2^x = barrel shift로 처리하는 HW LUT/시프터 설계에 직결.
- **MAC 양자화**: QK^T·P·V를 INT8×INT8→INT32 dot(kernel.py:153,171)로 처리 → systolic/MAC array의 INT8 데이터패스 + INT32 누적기 폭 설계 근거.
- **online softmax 고정소수점 rescale**(`scale_release`, kernel.py:67-82): 누적기 보정을 곱·시프트로 한 점은 HW 파이프라인의 rescale 스테이지 비트폭(L_SHIFT/ACC_SHIFT=30) 결정에 참고.
- **스케일 상수화 패턴**(kernel.py:300-307): 모든 dequant를 호스트에서 정수 상수로 사전계산 → 가속기에 상수 레지스터로 주입하는 방식과 동형.
- XR 시선추적과의 직접 연관은 낮으나(영상 분류·어텐션 벤치 중심), 저지연 ViT 백본을 시선추적 인코더로 쓸 경우 어텐션 정수화 지연/정확도 트레이드오프 데이터로 활용 가능.

---

## 9. 근거 표기 규칙
- 본 문서의 모든 기술 주장은 (파일:라인) 근거. **"추정"** 표기: §8의 FPGA/XR 적용 해석(우리 프로젝트 맥락 연결은 추정).
- **확인 불가(없음)**: ONNX/TFLite/TRT 익스포트 코드, per-channel/percentile observer, QAT 경로 — repo에 미존재.
