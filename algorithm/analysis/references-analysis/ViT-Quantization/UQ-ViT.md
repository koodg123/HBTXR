# UQ-ViT 정밀 분석

> 분석 대상: `REF/ViT-Quantization/UQ-ViT`
> 작성 기준: 실제 소스 코드 (파일:라인 근거). 추정/확인 불가 명시.

---

## 1. 개요

- **정체 (확인 — README 명시)**: *"**UQ-ViT: Harmonizing Extreme Activations with Hardware-Friendly Uniform Quantization in Vision Transformers**"*. (classification/README.md:1). 즉 약어 UQ = **Uniform Quantization**(uncertainty 아님). 부제가 핵심: "**극단(extreme) activation을 하드웨어 친화적 균등 양자화로 조화시킨다**".
  - 논문 출처/연도는 README/코드에 명기되어 있지 않음 → **확인 불가** (제목·결과표만 존재).
- **목적**: ViT의 **극단 activation(outlier)** 을, log-quant 등 비균등 양자화 없이 **순수 uniform(균등) 양자화**로 처리하면서 정확도를 유지 → FPGA/ASIC 친화적인 균등 정수 datapath 유지.
- **핵심 아이디어 (코드 기반)**:
  1. **DeMax quantizer** (`UniformQuantizer_DeMax`): post-softmax activation에서 각 행의 **최댓값을 분리(빼고)** 나머지를 균등 양자화 → softmax의 큰 값(극단)을 따로 처리, 균등 양자화 해상도 확보.
  2. **NormQuant + reparam(`alph` 탐색)**: LayerNorm 직후 Linear 입력의 채널별 극단을, weight/activation 스케일을 `alph`로 보간한 등가변환(`r`, `b`)으로 흡수 (SmoothQuant 계열) → 균등 per-tensor 양자화 가능.
- **양자화 방식**: **PTQ** (calibration + per-layer `alph` 탐색). 균등(uniform) asymmetric/symmetric.

---

## 2. 디렉토리 구조 (자체 + 제외)

```
UQ-ViT/
├── classification/                  # ★ 분류 (자체 핵심)
│   ├── test.py                      # 엔트리: quant_model + alph search + validate (★)
│   ├── README.md                    # UQ-ViT 정체/결과 (★)
│   ├── quant/
│   │   ├── quantizer.py             # UniformQuantizer, UniformQuantizer_DeMax (★★)
│   │   ├── quant_modules.py         # QuantConv2d/Linear/Linear_no_b/MatMul (NormQuant) (★★)
│   │   ├── quant_model.py           # quant_model() 모듈 치환 (★)
│   │   └── __init__.py
│   └── utils/
│       ├── build_model.py           # MatMul 래퍼 포함 ViT 빌드
│       ├── build_dataset.py / utils.py
└── detection/                       # mmdet 기반 (외부 프레임워크 — 제외, 이름만)
    └── mmdet/...                     # MMDetection 외부 코드 (분석 제외)
```

**제외 (지시 준수)**: `detection/mmdet/*` 전체는 외부 MMDetection 프레임워크 → **이름만** 언급, 분석 제외. `.git/`, `__pycache__`.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `classification/quant/quantizer.py` — 균등 양자화기 ★★

#### `UniformQuantizer(nn.Module)` (84-235)
- **균등 affine 양자화** (비대칭 default, 대칭 옵션). `n_levels = 2^n_bits`(97). 2~8bit 지원(95).
- `forward`(109-117): 비대칭이면 `x_dequant = (clamp(round(x/Δ)+zp, 0, n_levels-1) - zp)·Δ`(116) — fake-quant.
- `init_quantization_scale`(120-171): 
  - channel_wise면 채널별 abs-max로 Δ/zp(122-152).
  - per-tensor면 **percentile 후보 [0.999, 0.9999, 0.99999]** 로 quantile 계산(158-160) → 각 후보로 양자화 후 **MSE 최소 Δ/zp 선택**(165-171). **outlier robust calibration**.
  - GELU 출력 특수 처리: 범위가 거의 0이면 `min=GELU_MIN(-0.16997)`로 고정(161-163) — GELU 음수 하한.
- `forward_symmetric`/`init_quantization_scale_symmetric`(179-230): 대칭 버전. `quantile→abs_max→2·abs_m/(2^b-1)` MSE 탐색.
- `quantize`(173-177): 임의 max/min으로 즉석 양자화(후보 평가용).

#### `UniformQuantizer_DeMax(nn.Module)` (239-268) ★★ — 극단값 분리 양자화
- post-softmax(=`matmul2`의 A입력)용. `forward`→`init_quantization_uniform`(255-268):
  - 입력 shape `(B,H,N,_)`(256). **각 행의 max를 찾아(258) 해당 위치만 빼냄**: `sub_x = mask·max`(263), `x = x - sub_x`(264).
  - 남은 텐서의 **두 번째 max**로 Δ 산정(265-266) → `Δ = max₂/(n_levels-1)`.
  - 양자화 후 **빼냈던 max를 다시 더함**: `x_dequant = (clamp(round(x/Δ),0,n-1))·Δ + sub_x`(267).
  - 의미: softmax 출력은 한 토큰에 큰 확률 1개 + 나머지 작은 값들 → **큰 값을 따로 빼고** 나머지를 작은 Δ로 균등 양자화 → 작은 attention 가중치의 해상도 보존(극단값에 Δ가 끌려가지 않게).

#### percentile 헬퍼 (6-73)
- `calculate_quantiles`(6-31): `torch.quantile` 실패 시 `topk`/`np.percentile` fallback(대용량 텐서 대응). max/min percentile 추출.

### 3.2 `classification/quant/quant_modules.py` — NormQuant 레이어 ★★

#### `QuantLinear(nn.Linear)` (79-177) — NormQuant 핵심 ★
- `norm_quant=True`일 때(94-98) **observer 3종**: `input_quantizer_obs`, `observer`(weight, 대칭), `input_quantizer`.
- `forward`의 `norm_quant` 경로(130-161):
  - 최초 1회(`inited==False`): weight Δ(`observer`)와 activation Δ/zp(`input_quantizer_obs`)를 측정(135-140).
  - **등가 스케일 `r` 계산**(146): `r = (act_Δ/target_Δ)^(1-alph) / (weight_Δ)^alph` → **SmoothQuant식 채널 균형**, `alph`로 act↔weight 사이 분담 보간.
  - `alph==-1`이면 `r=1`(균형 끔, 148).
  - bias 보정 `b = act_min/r - target_min`(149), `bias_x = (weight·r)·b`(153).
  - **적용**: `x = input_quantizer(x/r - b)`, `w = weight_quantizer(weight·r)`(160-161) → **채널 outlier를 weight로 이동**시켜 activation을 per-tensor 균등 양자화하기 쉽게 만듦.
- 결과 `F.linear(x, w, bias + bias_x)`(176).

#### `QuantLinear_no_b(nn.Linear)` (179-282)
- bias 없는 Linear(proj, fc2)용. `alph=-1` 기본(207). `r`을 `clamp(1, None)`로 1 이상만(252) — **단방향(activation→weight) 이동**. act_delta를 채널 abs-max로 직접(243-246). `b=0`(254) — bias 보정 없이 스케일만 이동.

#### `QuantConv2d(nn.Conv2d)` (9-76)
- PatchEmbed conv. input은 8bit 고정(34), weight/input 균등 양자화. `set_quant_state`로 on/off(46-48).

#### `QuantMatMul(nn.Module)` (284-322) — Attention 행렬곱 ★
- `input_quant_params`에 `demax_quant` 키 있으면 **A입력을 `UniformQuantizer_DeMax`로**(292-294), 아니면 일반 UniformQuantizer(296). B는 항상 일반(297).
- `forward`(313-322): `use_quantizer_A`면 A,B 둘 다 양자화, 아니면 B만(315-320). `A @ B`(321).
- → **post-softmax matmul(matmul2)의 A입력(=attention prob)에 DeMax 적용**.

### 3.3 `classification/quant/quant_model.py` — 모듈 치환 ★
- `quant_model(model, input_quant_params, weight_quant_params)`(7-68):
  - `matmul2`(post-softmax)에 **`demax_quant=True`** 주입(11-12) → DeMax 양자화기.
  - **channel_wise input** params 별도 구성(`SimQuant` 주석, 14-15).
  - Linear 치환 규칙(47-58):
    - `qkv / fc1 / reduction` → `QuantLinear(norm_quant=True)` (LayerNorm 직후, bias 있음)(50-51).
    - `fc2 / proj` → `QuantLinear_no_b(norm_quant=True)` (51-53).
    - 그 외 → 일반 `QuantLinear`(54-55).
  - `MatMul`(build_model 래퍼) → matmul2는 DeMax, 나머지는 일반(59-66).
- `set_quant_state`/`set_initquant_state`(71-81): 전역 on/off, 재초기화.

### 3.4 `classification/test.py` — PTQ + alph 탐색 파이프라인 ★

#### `search_alph(...)` (43-77) ★
- 후보 `alph` 값들에 대해 블록 출력의 MSE(`lp_loss`)를 최소화하는 `alph` 선택(50-60). 양자화 상태를 켜고 각 alph로 재초기화→forward→score 비교(53-58). 선택 후 best_alph 고정(60).

#### `recon_model(...)` (119-176)
- 블록(Block/SwinBlock)별로:
  - calibration 입력/FP 출력 수집(`save_inp_oup_data`, 130-131).
  - **레이어별 alph 탐색** 순서(140-160):
    - `qkv`: `alph∈[0,0.05,...,0.25]` (matmul2 A 끄고 탐색)(142-145).
    - `proj`: `alph∈[-1,0,0.1,...,1]` (150-152).
    - `fc1`: `[0,...,0.25]`(155-156), `fc2`: `[-1,0,...,1]`(159-160).
  - PatchMerging의 reduction도 alph 탐색(164-170).
- 즉 **레이어별 최적 act↔weight 균형(alph)을 PTQ로 탐색** → 균등 양자화 정확도 최대화.

#### `main` (78-188)
- timm 모델 빌드→`quant_model` 치환(98-115). wq `channel_wise=True`, aq `channel_wise=False`(113-114).
- `recon_model`로 alph 탐색→ calibration 1배치 forward(178-182)→ `validate`(186).

---

## 4. 알고리즘 / 수식

### 4.1 DeMax (극단값 분리 균등 양자화)
행별 최댓값 `m = max_j x_{ij}` 분리:
$$\tilde{x}_{ij} = x_{ij} - m_i\cdot\mathbb{1}[j=\arg\max],\quad \Delta_i=\frac{\max_2(\tilde{x}_i)}{2^b-1},\quad \hat{x}_{ij}=\text{clamp}(\text{round}(\tilde{x}_{ij}/\Delta_i),0,n{-}1)\Delta_i + m_i\mathbb{1}[\cdot]$$
극단(=softmax 지배값)을 따로 보존하고 나머지를 작은 Δ로 균등 양자화 → 작은 attention 가중치 해상도 확보. 근거: quantizer.py:255-268.

### 4.2 NormQuant 등가변환 (SmoothQuant 계열)
채널별 균형 인자 `r`로 activation→weight 이동:
$$r = \frac{(\,s^{act}/s^{target}\,)^{\,1-\alpha}}{(\,s^{w}\,)^{\alpha}},\quad x\leftarrow x/r - b,\quad W\leftarrow W\cdot r,\quad b_{bias}=(W\cdot r)\,b$$
`α(=alph)`로 act/weight 분담을 보간, per-layer로 MSE 최소 α 탐색. 근거: quant_modules.py:146-161, test.py:43-77.

### 4.3 Percentile + MSE scale 선택
후보 percentile {0.999, 0.9999, 0.99999}로 범위 산정 → 각 후보 양자화 MSE 최소 Δ/zp 선택(outlier robust). 근거: quantizer.py:158-171.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet (README.md:15).
- **모델**: vit_small/base, deit_tiny/small/base, swin_tiny/small (README.md:13-14; test.py:82-94, timm).
- **명령** (README.md:9-23):
  `python test.py --model deit_small --dataset <DIR> --w_bit 4 --a_bit 4` (default W4/A4).
- **PTQ 흐름**: timm 빌드 → `quant_model` 치환 → calibration 데이터로 레이어별 `alph` 탐색(`recon_model`) → 양자화 forward → `validate`.
- **결과**(README.md:30-38): W4/A4에서 DeiT-S 72.13%, DeiT-B 76.59%, Swin-B 81.96%; W6/A6는 FP 근접.

---

## 6. 의존성

- PyTorch + **timm**(vision_transformer/swin_transformer Block·PatchMerging·PatchEmbed에 강결합, test.py:9-10). NumPy. CUDA(`.cuda()` 다수). detection은 MMDetection(외부, 제외).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **하드웨어 친화 균등 양자화 유지**: log-quant/비균등 없이 uniform만으로 극단 activation 처리 → FPGA/ASIC의 단순 정수 datapath와 직결.
- **DeMax**: post-softmax 극단값을 비파괴적으로 분리 → 작은 attention 가중치 해상도 보존(저비트에서 중요).
- **NormQuant+alph 탐색**: SmoothQuant식 등가변환을 per-layer로 자동 튜닝 → activation을 per-tensor 균등으로 압축.
- 순수 PTQ(재학습 불필요), timm 기반 다중 모델 지원.

**한계 / 리스크**
- **논문 메타데이터 부재**: venue/연도 코드/README에 없음 → **확인 불가**.
- timm 구조에 강결합(`qkv/proj/fc1/fc2/reduction` 이름 매칭, quant_model.py:50-53) → 커스텀 ViT엔 이름 매핑 수정 필요.
- DeMax는 "max 1개 분리"를 가정 → softmax가 아닌 일반 텐서엔 부적합. matmul2 전용.
- fake-quant PTQ — 정수/dyadic 실행 코드는 없음(HW 매핑 별도).
- `alph` 그리드 탐색이 레이어×후보 만큼 forward 반복 → calibration 비용.

---

## 8. 우리 프로젝트 관점 시사점 (ViT FPGA 가속기 + XR 시선추적 — 추정)

- **"hardware-friendly uniform"이 정확히 우리 타깃**: HG-PIPE류 가속기는 균등 정수 MAC array가 본체 → UQ-ViT의 핵심 가치(비균등 양자화 없이 outlier 처리)는 우리 HW 제약과 **직접 정합**. log-quant를 피하면서 저비트 정확도를 얻는 경로.
- **DeMax의 HW 매핑**: post-softmax에서 "max 분리 + 잔여 균등 양자화"는 attention·V 행렬곱 앞단에서 **max 추출기 + 작은 Δ 균등 양자화** 하드웨어로 구현 가능. softmax 출력의 dynamic range 문제를 균등 datapath로 해결 → I-ViT의 Shiftmax(정수 softmax)와 결합 검토 가치(추정).
- **NormQuant(alph) = 오프라인 등가변환**: `r`을 weight에 흡수하므로 **추론 시 추가 연산 0** (오프라인에서 weight 미리 곱함). FPGA에 weight를 굽기 전 적용하면 per-tensor activation 양자화로 단순화 → 우리 datapath에 이상적.
- **시선추적 적용**: 순수 PTQ라 시선추적 fine-tuned ViT에 재학습 없이 적용 가능. 단 timm 레이어 이름 매칭/회귀 head 처리 수정 필요(추정).
- **조합 전략(추정)**: UQ-ViT(균등 PTQ로 정확한 scale/alph 결정) → I-ViT식 dyadic 정수 datapath로 실행 → outlier-free식 학습 단계 억제(가능시). 세 repo가 상보적.

---

## 9. 근거 표기

- repo 정체(UQ = Uniform Quantization, "Harmonizing Extreme Activations..."): **확인** (classification/README.md:1). 지시문의 "uncertainty quantization 추정"은 **오류** — 실제는 Uniform Quantization (README 명시).
- 논문 venue/연도: **확인 불가** (코드/README에 없음).
- DeMax, NormQuant(alph), uniform quantizer, alph 탐색: **확인** (quantizer.py, quant_modules.py, quant_model.py, test.py 라인 근거).
- `detection/`(MMDetection): 외부 프레임워크 → 분석 제외(이름만), **지시 준수**.
- "HG-PIPE + XR 시선추적" 적용 방안 및 조합 전략: **추정** (지시문 기반).
