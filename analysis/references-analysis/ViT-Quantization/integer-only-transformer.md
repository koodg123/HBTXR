# integer-only-transformer (I-ViT) 정밀 분석

> 분석 대상: `REF/ViT-Quantization/integer-only-transformer`
> 작성 기준: 실제 소스 코드 (파일:라인 근거 표기). 추정/확인 불가는 명시.

---

## 1. 개요

- **정체 (확인)**: 이 repo는 **I-ViT** — *"I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference"* (Zhikai Li, Qingyi Gu, **ICCV 2023**)의 공식 구현이다. README.md:5-8, 인용 정보 README.md:62-68.
- **목적**: ViT/DeiT/Swin 추론을 **정수 전용(integer-only)** 연산으로 수행. LayerNorm·Softmax·GELU 등 비선형 연산까지 부동소수점 없이 정수+비트시프트(**dyadic**)로 근사하여, INT8 정수 파이프라인으로 FP32에 근접한 정확도를 달성(README.md:47-55, 예: ViT-S 81.39→81.27).
- **핵심 아이디어**:
  1. **Dyadic arithmetic**: 스케일 재조정을 부동소수점 곱이 아닌 정수 곱 + 비트시프트 `round(q·m / 2^e)`로 수행 (`fixedpoint_mul`, `batch_frexp`).
  2. **Shiftmax / ShiftGELU**: exp를 `2^x` shift 근사로 계산하는 정수 softmax/GELU (`int_exp_shift`).
  3. **I-LayerNorm**: 정수 비트시프트 반복으로 표준편차(`sqrt`)를 계산하는 정수 LayerNorm.
- **학습 방식**: **QAT (Quantization-Aware Training)**. 사전학습 FP 모델을 INT8로 fine-tune (`quant_train.py`).

---

## 2. 디렉토리 구조 (자체 + 제외)

```
integer-only-transformer/
├── quant_train.py              # QAT 학습 엔트리 (★)
├── evaluate_layer.py           # 레이어 단위 latency 측정
├── ir_analysis.py
├── models/
│   ├── vit_quant.py            # 정수 ViT/DeiT (Attention/Block/VisionTransformer) (★)
│   ├── swin_quant.py           # 정수 Swin
│   ├── layers_quant.py         # PatchEmbed, Mlp, DropPath 등 정수 레이어
│   ├── model_utils.py / utils.py
│   └── quantization_utils/
│       ├── quant_utils.py      # dyadic/STE/대칭양자화 함수 (★)
│       └── quant_modules.py    # QuantLinear/QuantAct/IntLayerNorm/IntGELU/IntSoftmax/QuantMatMul (★)
├── utils/                      # 데이터/학습 보조
├── ibert/                      # I-BERT 참조 코드 (ibert_layers.py, ibert_quant_utils.py)
├── TVM_benchmark/              # TVM 배포·latency 재현 (별도 프로젝트)
└── experiments/                # latency/accuracy 결과 로그
```

**제외**: `.git/`, `__pycache__`, `experiments/*.json/.tsv/.out`(결과 로그), `overview.png`. `TVM_benchmark/`와 `ibert/`는 보조이므로 이름만 언급, 핵심 분석은 `models/quantization_utils/`에 집중.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `models/quantization_utils/quant_utils.py` — 정수 산술 기반

#### `linear_quantize(input, scale, zero_point, is_weight)` (12-48)
- 대칭 양자화 핵심: `round(1/scale · input + zero_point)` (48). weight는 출력채널축(`view(-1,1,...)`, 23-33), activation은 마지막 채널축(34-45)으로 scale/zp를 reshape → **per-channel** 지원.

#### `symmetric_linear_quantization_params(num_bits, min_val, max_val)` (51-69)
- 대칭 스케일: `n = 2^(b-1)-1`, `scale = max(|min|,|max|)/n` (62-66). `torch.no_grad()`로 grad 차단(61). eps clamp(67).

#### `SymmetricQuantFunction` (72-119)
- `forward`(77-96): zero_point=0인 순수 대칭 양자화. `clamp(-n-1, n)`로 INT 범위 포화(92).
- `backward`(98-119): **STE** — grad를 scale로 나눠 통과(119). dequant 경로 미분.

#### `floor_ste` / `round_ste` (122-147)
- `floor`/`round`의 **Straight-Through Estimator**. forward는 정수화, backward는 grad 그대로 통과. 정수 비선형 연산 학습 가능하게 하는 핵심.

#### `batch_frexp(inputs, max_bit=31)` (150-175)
- **Dyadic 분해**: scale을 `np.frexp`로 가수(mantissa) `m`과 지수(exponent) `e`로 분해 → `scale ≈ m / 2^e`, m은 `max_bit=31`비트 정수로 양자화(167-170). 하드웨어 정수곱+시프트로 스케일 재조정을 구현하기 위한 사전 분해.

#### `fixedpoint_mul` (178-261) — Dyadic requantization 핵심
- `forward`(192-253):
  - `z_int = round(pre_act / pre_act_scaling_factor)` 입력을 정수화(220).
  - `new_scale = pre_act_scale / z_scale`(이전 스케일/목표 스케일, 221-223) → `batch_frexp`로 `(m,e)` 분해(228).
  - `output = round( z_int · m / 2^e )` (229-230): **부동소수점 곱 없이** 정수곱+시프트로 requant.
  - residual `identity` 경로도 동일 dyadic 처리 후 가산(232-245) → skip-connection 정수 합산.
  - INT 범위 clamp (247-251).
- `backward`(255-261): STE로 grad / z_scale 통과.

### 3.2 `models/quantization_utils/quant_modules.py` — 정수 레이어

#### `QuantLinear(nn.Linear)` (12-97)
- 대칭 + **per-channel weight** 강제(70-77; per_channel 아니면 Exception). `weight_function = SymmetricQuantFunction`(43-44).
- `forward`(67-97): weight scale 계산(79) → weight 정수화(82) → `bias_scale = weight_scale · prev_act_scale`(85) → bias 정수화(88, **bias_bit=32**) → 입력 정수화 `x_int = x / prev_act_scale`(94) → `F.linear(x_int, w_int, b_int) · bias_scale` 반환(96). 즉 정수 GEMM 후 스케일 복원.

#### `QuantAct(nn.Module)` (100-206)
- **Observer 역할**: activation 양자화 범위 추정. running min/max를 momentum(`act_range_momentum=0.95`)으로 갱신(172-189). `fix()/unfix()`로 running_stat 토글(153-163) → calibration 종료/재개.
- `forward`(165-206): 입력 양자화는 `act_function`(196), 이전 스케일 있으면 **`fixedpoint_mul`로 dyadic requant**(198-202) → 정수 파이프라인 유지. residual `identity` 전달도 여기서 처리.

#### `QuantMatMul(nn.Module)` (209-228)
- Q·Kᵀ, attn·V 등 정수 행렬곱. `A_int = A/scale_A`, `B_int = B/scale_B`, 출력 스케일 `scale_A·scale_B`(224-227). `(A_int @ B_int)·scale` 반환 → 정수곱.

#### `QuantConv2d(nn.Conv2d)` (231-330)
- PatchEmbed용 정수 conv. per-channel weight, bias_bit=32, dyadic 출력 스케일(321-330). QuantLinear와 동형 구조.

#### `IntLayerNorm(nn.LayerNorm)` — I-LayerNorm (333-387) ★
- `forward`(354-387):
  - 입력 정수화 `x_int = x/scale`(360), 정수 평균 `round_ste(mean)`(361), 편차 `y_int = x_int - mean_int`(362), 분산 `var_int = Σ y²`(363-364).
  - **정수 표준편차(루트) 계산**(366-371): 부동소수 `sqrt` 대신 **정수 뉴턴/비트시프트 반복** — `k_{i+1} = floor((k + floor(var/k))/2)`를 10회 반복(367-371). 초기값 `k=2^16`. → 정수 √var.
  - `factor = floor((2^31-1)/std_int)`(373)로 정규화, `y_int = floor(y_int·factor/2)`(374), 스케일 `dim_sqrt/2^30`(375).
  - affine: bias를 weight로 나눠 정수화(378-381), `y_int + bias_int`(383), `scale·weight`(384).

#### `IntGELU(nn.Module)` — ShiftGELU (390-446) ★
- `n=23` (충분히 큰 정수, 401).
- `int_exp_shift(x_int, scale)`(411-424): exp의 정수 근사.
  - `x_int += floor(x/2) - floor(x/2^4)` → ≈ `x·1.4375 ≈ x·log2(e)` 근사로 `e^x → 2^x` 변환(412).
  - `x0_int = floor(-1/scale)`(415), `q = floor(x/x0)`, `r = x - x0·q`(418-419) → 정수부/소수부 분리.
  - `exp_int = (r/2 - x0)`를 `2^(n-q)` 시프트(420-421) → `2^x` 시프트 근사.
- `forward`(426-446): GELU(x)≈x·σ(1.702x). `scale_sig = scale·1.702`(428). `int_exp_shift`로 `e^(x-xmax)`(433)와 `e^(-xmax)`(435) 계산 → sigmoid_int(440) → `x_int · sigmoid_int`(443)로 GELU 정수 근사. 출력 스케일 dyadic.

#### `IntSoftmax(nn.Module)` — Shiftmax (449-498) ★
- `n=15`(459).
- `int_exp_shift`(470-482): IntGELU와 동일한 `2^x` shift exp 근사.
- `forward`(484-498): `x_int - x_max`(486-487, 안정화) → `exp_int`(489) → `exp_int_sum`(490) → `factor = floor((2^31-1)/sum)`(493) → 정규화 `exp_int·factor/2^(31-out_bit+1)`(494) → 출력 스케일 `1/2^(out_bit-1)`(495). **softmax 출력은 16-bit**(int_softmax는 vit_quant.py:54에서 `IntSoftmax(16)`).

### 3.3 `models/vit_quant.py` — 정수 ViT 모델 ★

#### `Attention` (23-88)
- 모든 연산을 정수 모듈로 구성: `QuantLinear qkv`(38), `QuantAct` 다수(43-51), `IntSoftmax(16)`(54), `QuantMatMul ×2`(56-57).
- `forward`(59-88): qkv 정수 linear→QuantAct(61-62) → reshape → **matmul_1**(Q·Kᵀ, 70) → `·scale`와 동시에 스케일도 곱(72-73) → QuantAct(74) → **int_softmax**(76) → dropout → **matmul_2**(attn·V, 79) → proj QuantLinear(84) → `qact3(16-bit)`(85). 스케일 인자(`act_scaling_factor`)가 레이어 사이를 명시적으로 흐름.

#### `Block` (91-143)
- `norm1`(IntLayerNorm)→qact1→attn→`qact2(16)`(residual 합산, 135)→norm2→qact3→mlp→`qact4(16)`(residual 합산, 141). residual 가산은 `QuantAct`의 identity 경로(dyadic)로 정수 합산.

#### `VisionTransformer` (146-282)
- `qact_input`(176)부터 출력까지 전부 정수 흐름. `IntGELU`를 act_layer로 주입(209), `IntLayerNorm`을 norm_layer로 주입(deit_*: 293 등). DeiT-T/S/B, ViT-B/L 팩토리(285-381), pretrained 로딩(npz/torch.hub).

### 3.4 `quant_train.py` — QAT 파이프라인 (요약)
- README.md:24-41: `python quant_train.py --model deit_tiny --data <DIR> --epochs 30 --lr 5e-7`. 매우 낮은 lr(2e-7~2e-6)로 정수화된 모델 fine-tune. `fix()/unfix()`로 activation range 동결 제어.

---

## 4. 알고리즘 / 수식

### 4.1 Dyadic requantization
스케일 변환 `s_in/s_out`을 정수 가수·시프트로:
$$\frac{s_{in}}{s_{out}} \approx \frac{m}{2^{e}},\quad m\in\mathbb{Z}_{31bit},\; q_{out}=\text{round}\!\left(\frac{q_{in}\cdot m}{2^{e}}\right)$$
근거: `batch_frexp`(quant_utils.py:150-175), `fixedpoint_mul`(178-253).

### 4.2 Shiftmax/ShiftGELU의 정수 exp
`e^x = 2^{x·log2(e)}`, `log2(e)≈1.4375 ≈ 1 + 1/2 - 1/16`:
$$x' = x + \lfloor x/2\rfloor - \lfloor x/2^4\rfloor,\quad x'=x_0\cdot q + r,\quad e^{x}\approx (r/2 - x_0)\cdot 2^{\,n-q}$$
근거: `int_exp_shift` (quant_modules.py:411-424, 470-482).

### 4.3 I-LayerNorm 정수 √
$$k_{i+1} = \left\lfloor \tfrac{1}{2}\left(k_i + \left\lfloor \tfrac{\text{var}}{k_i}\right\rfloor\right)\right\rfloor,\; i=0..9,\quad k_0=2^{16}$$
정수 뉴턴법으로 `std = √var` 근사. 근거: quant_modules.py:366-371.

### 4.4 정수 GELU
`GELU(x) ≈ x·σ(1.702x)`, σ는 shift-exp 기반. 근거: quant_modules.py:426-446.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet (README.md:32, 45).
- **모델**: deit_tiny/small/base, swin_tiny/small/base (README.md:30-31).
- **QAT 명령**: `python quant_train.py --model deit_tiny --data <DIR> --epochs 30 --lr 5e-7` (README.md:40). epochs∈{30,60,90}, lr∈{2e-7..2e-6}.
- **결과**: INT8에서 FP32 대비 ±0.3% 이내 (README.md:47-55).
- **Latency**: `TVM_benchmark/`로 GPU(2080Ti/3090) latency 재현 (experiments/latency/*).

---

## 6. 의존성

- PyTorch (CUDA `.cuda()` 강제, quant_utils.py:88 등 — **GPU 필수**), NumPy, `decimal`/`fractions`(dyadic 정밀 분해), timm 계열 ViT 구조. 배포: TVM 0.9.dev0 (README.md:14).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **완전 정수 전용 추론**: LayerNorm/Softmax/GELU까지 정수+시프트로 근사 → FP 유닛 없는 HW(정수 ASIC/FPGA)에 직결.
- Dyadic requant로 스케일 재조정도 정수곱+시프트 → MAC array 뒤단에서 lookup/FP 불필요.
- INT8에서 정확도 손실 거의 없음(QAT).

**한계 / 리스크**
- **QAT 필요**: ImageNet 재학습(저lr fine-tune) 비용. PTQ보다 무겁다.
- 코드가 `.cuda()` 하드코딩 → CPU/이식성 제약 (quant_utils.py:88, quant_modules.py:357 등).
- `int_exp_shift`의 `n`(GELU=23, Softmax=15)은 모델별 정확도에 민감(주석 401, 459 "varies depending on models").
- INT8 위주, 4bit 이하 저비트 activation은 별도 검증 필요(SymmetricQuantFunction은 4/8/16/32 분기, quant_utils.py:247).

---

## 8. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적 — 추정)

- **정수 전용 = FPGA 직결**: I-ViT의 dyadic requant(`fixedpoint_mul`)와 Shiftmax/ShiftGELU/I-LayerNorm은 **FP 유닛 없이** ViT 전체를 정수 datapath로 구현하는 정확한 청사진. HG-PIPE류 파이프라인 가속기에서 비선형 블록(Softmax/GELU/LN)을 LUT+시프트로 구현할 때 **레퍼런스 알고리즘**으로 직접 활용 가능.
- **시프트 기반 exp**: `x + x>>1 - x>>4` (log2(e) 근사)와 `2^x` 배럴시프터 구현은 FPGA에서 DSP 소모 없이 exp를 근사 → Softmax/GELU 하드웨어 비용 절감.
- **정수 √ (뉴턴 10-iter)**: LayerNorm을 정수 반복으로 — 다만 10회 반복은 latency 비용이므로, 우리 가속기에서는 반복수 축소 또는 LUT 대체 trade-off 검토 필요(추정).
- **XR 시선추적 적용 시**: 저지연·저전력 정수 datapath는 XR 엣지(배터리·열) 제약에 부합. 단, I-ViT는 QAT 기반이라 시선추적 데이터셋 재학습 파이프라인이 필요(추정).
- **scale 흐름 설계**: `act_scaling_factor`를 레이어 간 명시적으로 전달하는 구조는 HW 파이프라인 스테이지 간 스케일 메타데이터 전달 설계의 참고 모델.

---

## 9. 근거 표기

- repo 정체(I-ViT/ICCV2023): **확인** (README.md:5-8, 62-68).
- 정수 비선형/dyadic 구현: **확인** (quant_modules.py, quant_utils.py 라인 근거).
- "HG-PIPE 계열 + XR 시선추적" 프로젝트 맥락 및 적용 방안: **추정** (지시문 기반, repo 내 직접 근거 없음).
- `quant_train.py` 내부 루프 상세: README 명령어 기준 요약 (본문 미정독 — 필요시 추가 정독 권장, **부분 확인**).
