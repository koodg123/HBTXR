# I-ViT 정밀 분석 (Integer-only Quantization for Efficient Vision Transformer Inference)

> 분석 대상 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\I-ViT`
> 분석 방식: 자체 핵심 소스 라인 단위 정밀 분석 (Glob/Grep/Read). bash 미사용.
> 근거 표기 규칙: `파일:라인`. 직접 코드 확인분과 "추정"/"확인 불가"를 구분 표기.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문**: Zhikai Li, Qingyi Gu, *"I-ViT: Integer-only Quantization for Efficient Vision Transformer Inference"*, ICCV 2023 (arXiv:2207.01405). (`README.md:5-11, 63-69`)
- **목적**: ViT 추론을 **부동소수점(FP) 연산 없이 정수(integer)·비트시프트(bit-shift)만으로 수행**. 논문 저자 표현상 "ViT 최초의 정수 전용(integer-only) 양자화" (`README.md:8`).
- **핵심 아이디어 (코드로 확인)**:
  1. **Dyadic(이진 유리수) requantization**: 스케일 재조정 `M = b/2^c` 형태로 분해하여 정수 곱 + 산술 시프트만으로 스케일 변환 (`quant_utils.py:150-261`, `batch_frexp` + `fixedpoint_mul`).
  2. **ShiftGELU (`IntGELU`)**: GELU≈x·σ(1.702x)를 시프트 기반 정수 지수함수 `int_exp_shift`로 근사 (`quant_modules.py:389-445`).
  3. **Shiftmax (`IntSoftmax`)**: Softmax의 지수항을 동일한 `int_exp_shift`로 정수 근사 (`quant_modules.py:448-497`).
  4. **I-LayerNorm (`IntLayerNorm`)**: 표준편차를 정수 뉴턴-반복(integer iteration)으로 계산 (`quant_modules.py:333-386`).
  5. 학습 방식은 **QAT(양자화 인식 학습)** — fake-quant 모듈로 ImageNet fine-tuning (`quant_train.py`, `README.md:23-42`). INT8 기준 FP32 대비 정확도 손실 ±0.2%대 (`README.md:48-56`).

본 repo는 세 분석 대상 중 **FPGA 비선형 함수(GELU/Softmax/LayerNorm) 하드웨어 구현에 가장 직접적**이다. 모든 비선형 연산이 LUT/시프트/정수 산술로 환원되어 있기 때문이다(8장).

---

## 2. 디렉토리 구조 (자체 + 제외)

### 자체 핵심 소스 (분석 대상)
```
I-ViT/
├── quant_train.py                         # QAT 학습/검증 엔트리포인트 (timm 기반)
├── models/
│   ├── vit_quant.py                       # 정수 ViT/DeiT 모델 정의 (Attention/Block/VisionTransformer)
│   ├── swin_quant.py                      # 정수 Swin 모델 정의
│   ├── layers_quant.py                    # Mlp, PatchEmbed, DropPath, trunc_normal_ 등 빌딩블록
│   ├── model_utils.py / utils.py          # npz 가중치 로딩 등 보조
│   └── quantization_utils/
│       ├── quant_modules.py    ★ 핵심     # QuantLinear/QuantAct/QuantConv2d/QuantMatMul
│       │                                  #   + IntLayerNorm/IntGELU/IntSoftmax (정수 비선형)
│       └── quant_utils.py      ★ 핵심     # SymmetricQuantFunction, dyadic(batch_frexp/fixedpoint_mul),
│                                          #   floor_ste/round_ste, symmetric_linear_quantization_params
├── utils/
│   ├── train_utils.py / utils.py          # freeze/unfreeze, AverageMeter 등
│   ├── data_utils.py / samplers.py        # ImageNet DataLoader
└── README.md
```

### 제외 (지시에 따라 이름만 표기, 미분석)
- `TVM_benchmark/` — TVM 배포·지연시간 커널 (제외 대상: "TVM_benchmark 커널"). `evaluate_latency.py`, `convert_model.py`, `models/quantized_vit.py` 등 포함.
- `.git/`, `__pycache__/` — 버전관리/캐시.
- `overview.png`, `LICENSE` — 비소스.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 양자화 기반 함수 — `quant_utils.py`

#### (a) 대칭 양자화 스케일 산출
- `symmetric_linear_quantization_params(num_bits, min_val, max_val)` (`quant_utils.py:51-69`):
  - `n = 2**(num_bits-1) - 1` (`:62`), `max_val = max(-min_val, max_val)` (`:65`), `scale = max_val / n` (`:66`), `scale.clamp_(eps)` (`:67`).
  - **대칭 양자화이며 zero-point는 0** (아래 `SymmetricQuantFunction`에서 `zero_point=0`).
- `linear_quantize(input, scale, zero_point, is_weight)` (`quant_utils.py:12-48`): `round(input/scale + zero_point)`. weight/activation 차원에 따라 scale/zp를 reshape (`:22-45`). 핵심 식: `torch.round(1./scale * input + zero_point)` (`:48`).

#### (b) `SymmetricQuantFunction` (autograd Function) — `quant_utils.py:72-119`
- forward (`:77-96`): zero_point=0 고정(`:88`), `n=2^(k-1)-1`(`:90`), `linear_quantize` 후 `clamp(-n-1, n)`(`:92`) → 부호있는 정수 범위 `[-2^(k-1), 2^(k-1)-1]`.
- backward (`:98-119`): **STE** — `grad_output / scale` 그대로 전달 (양자화기를 통한 그래디언트 통과).

#### (c) **Dyadic requantization — 가장 중요** : `batch_frexp` + `fixedpoint_mul`
- `batch_frexp(inputs, max_bit=31)` (`quant_utils.py:150-175`): 스케일을 **가수(mantissa) m과 2의 지수 e로 분해**.
  - `np.frexp`로 분해 후 `m * 2^max_bit`를 반올림하여 정수 가수 생성 (`:164-169`), `output_e = max_bit - exponent` (`:172`).
  - 즉 임의 실수 스케일 `S ≈ m / 2^e` (m은 정수, e는 정수 시프트량). **이것이 dyadic 수 `M = b/2^c`의 구현체**.
- `fixedpoint_mul` (autograd Function) (`quant_utils.py:178-261`): 정수 입력에 새 스케일을 적용하는 **재양자화 핵심 루틴**.
  - 새 스케일 `new_scale = A/B` (입력 스케일 / 출력 스케일) 계산 (`:221-223`).
  - `m, e = batch_frexp(new_scale)` (`:228`) → 정수 가수·시프트.
  - `output = round( z_int * m / 2^e )` (`:229-230`): **정수 곱 + 우측 산술 시프트(2^e 나눗셈)** 만으로 스케일 변환. FP 나눗셈 없음.
  - identity(residual) 경로도 동일한 dyadic 곱으로 더함 (`:232-245`) → **residual add를 정수 도메인에서 정렬**.
  - bit_num∈{4,8,16,32}이면 대칭은 `clamp(-n-1, n)`, 비대칭은 `clamp(0, n)` (`:247-251`).
- **시사**: 이 `m·z >> e` 패턴이 곧 FPGA에서 "정수 MAC 결과를 다음 레이어 스케일로 맞추는 시프트-가수 재양자화"로 그대로 매핑된다.

#### (d) STE 헬퍼
- `floor_ste` (`quant_utils.py:122-133`), `round_ste` (`:136-147`): forward는 floor/round, backward는 identity. IntLayerNorm/IntGELU/IntSoftmax 내부 정수 연산이 학습 가능하도록 함.

### 3.2 양자화 레이어 — `quant_modules.py`

#### (a) `QuantLinear(nn.Linear)` — `quant_modules.py:12-97`
- per-channel 대칭 weight 양자화만 지원(`:70-77`, per_channel=False면 예외 `:77`).
- forward (`:67-97`):
  - weight min/max → `fc_scaling_factor`(per-out-channel) 산출 (`:79-80`).
  - `weight_integer = SymmetricQuantFunction(weight, weight_bit, fc_scaling_factor, is_weight=True)` (`:82-83`).
  - **bias 스케일 = weight_scale × 입력_scale** (`:85`) → `bias_integer`를 32bit로 양자화 (`:87-89`).
  - 입력은 이미 정수배 형태로 들어오므로 `x_int = x / prev_act_scaling_factor`로 정수 복원(`:94`) 후 `F.linear(x_int, weight_integer, bias_integer)` (정수 MAC) → 출력에 `bias_scaling_factor` 곱해 dequant (`:96-97`).
  - 반환: `(out, bias_scaling_factor)` — 스케일을 다음 레이어로 전파.

#### (b) `QuantAct(nn.Module)` — `quant_modules.py:100-206`
- activation 양자화 + **running min/max 통계** (momentum=0.95) (`:172-189`).
- forward (`:165-206`):
  - residual(identity) 합산 지원: `x_act = identity + x` (`:171`).
  - running stat로 min/max 갱신 후 `act_scaling_factor` 산출 (`:191-192`).
  - 입력 스케일이 없으면(모델 입력) 단순 `SymmetricQuantFunction`(`:196`), 있으면 **`fixedpoint_mul.apply(...)`로 dyadic 재양자화**(`:198-202`) → residual·스케일 정렬을 정수 도메인에서 수행.
  - `fix()`/`unfix()`로 running_stat on/off (`:153-163`) — 평가시 통계 고정.

#### (c) `QuantMatMul` — `quant_modules.py:209-228`
- Q·Kᵀ, attn·V용 정수 행렬곱. `A_int=A/scaleA, B_int=B/scaleB`, 출력 스케일=`scaleA*scaleB` (`:223-228`). 정수 누산 후 dequant.

#### (d) `QuantConv2d(nn.Conv2d)` — `quant_modules.py:231-330`
- PatchEmbed projection용. per-channel 대칭 weight, bias 32bit, dyadic 미사용(직접 정수 conv 후 스케일 곱) (`:297-330`).

#### (e) **`IntLayerNorm(nn.LayerNorm)` — I-LayerNorm** : `quant_modules.py:333-386`
- forward (`:353-386`):
  - `x_int = x/scale`, `mean_int = round_ste(x_int.mean)` (`:359-360`), 분산 `var_int = Σ(x_int-mean)^2` (`:361-363`).
  - **정수 표준편차 = 뉴턴 반복(Newton iteration)**: `k = 2^16`에서 시작, `k = floor((k + floor(var/k))/2)`를 10회 반복 (`:366-369`) → `std_int = k` (정수 sqrt 근사). **FP sqrt 제거**.
  - `factor = floor((2^31-1)/std_int)` (`:372`), `y_int = floor(y_int*factor/2)` (`:373`), `scaling_factor = dim_sqrt / 2^30` (`:374`).
  - affine: `bias/weight` 비를 정수 bias로 변환 후 더함 (`:377-383`).
- **시사**: 정수 division/sqrt가 시프트+반복으로 환원 → FPGA에서 반복 회로 또는 LUT-sqrt로 구현 가능.

#### (f) **`IntGELU(nn.Module)` — ShiftGELU** : `quant_modules.py:389-445`
- `self.n = 23` (지수 정확도용 큰 정수) (`:399`).
- `int_exp_shift(x_int, scaling_factor)` (`:410-423`):
  - `x_int = x_int + floor(x_int/2) - floor(x_int/2^4)` (`:411`) → **`exp(x)≈2^(1.5x)` 류 근사를 시프트로** (x + x/2 - x/16 ≈ 1.4375·x ≈ x·log2(e)).
  - `x0_int = floor(-1/scale)` (`:414`), `q = floor(x_int/x0_int)`, `r = x_int - x0*q` (`:417-418`).
  - `exp_int = (r/2 - x0) << (n-q)` 후 clamp≥0 (`:419-420`), 출력 스케일 `scale/2^n` (`:421`). **2의 거듭제곱 시프트로 지수항 구성**.
- forward (`:425-445`): GELU≈`x·σ(1.702x)` 사용.
  - `scale_sig = scale*1.702` (`:427`), max 빼서 안정화(`:429-430`).
  - `exp_int = int_exp_shift(x-x_max)`, `exp_int_max = int_exp_shift(-x_max)` (`:432-434`) → 합으로 σ 분모 구성 (`:435`).
  - `factor = floor((2^31-1)/exp_sum)` (`:438`), `sigmoid_int = floor(exp_int*factor/2^(31-out_bit+1))` (`:439`).
  - `x_int = pre_x_int * sigmoid_int` (`:442`) → GELU 정수 결과. 출력 스케일 `scale * (1/2^(out_bit-1))` (`:440,443`).
- **시사**: GELU 전체가 시프트+정수곱+한 번의 정수 나눗셈(reciprocal)으로 환원 → FPGA에서 LUT-free 또는 소형 LUT 구현 가능. **본 프로젝트의 비선형 가속에 직접 활용 1순위**.

#### (g) **`IntSoftmax(nn.Module)` — Shiftmax** : `quant_modules.py:448-497`
- `self.n = 15` (`:458`). `int_exp_shift`는 IntGELU와 동일 구조(`:469-481`).
- forward (`:483-497`):
  - max 차감 안정화(`:485-486`), `exp_int = int_exp_shift(x-x_max)` (`:488`), `exp_sum = Σ exp_int` (`:489`).
  - `factor = floor((2^31-1)/exp_sum)` (`:492`), `exp_int = floor(exp_int*factor/2^(31-out_bit+1))` (`:493`).
  - 출력 스케일 `1/2^(out_bit-1)` (`:494`). Attention에서는 `IntSoftmax(16)`로 16bit 출력 사용(아래 3.3).
- **시사**: Softmax도 지수(시프트) + 합 + 한 번의 정수 reciprocal로 환원. **본 프로젝트 Attention 정규화 가속에 직접 활용**.

### 3.3 모델 정의 — `vit_quant.py`

- `Attention` (`vit_quant.py:23-88`):
  - `qkv=QuantLinear`, 다수 `QuantAct`, `proj=QuantLinear`, `int_softmax=IntSoftmax(16)`, `matmul_1/matmul_2=QuantMatMul` (`:38-57`).
  - forward (`:59-88`): qkv→정수 분할→`matmul_1(q,kᵀ)`→`*scale`→`qact_attn1`→**`int_softmax`**(`:76`)→`matmul_2(attn,v)`→`proj`. **스케일 인자(`act_scaling_factor`)를 단계마다 명시적으로 전파**.
- `Block` (`vit_quant.py:91-143`): `norm1=IntLayerNorm`→qact1→attn→**residual을 qact2(16bit)에서 fixedpoint_mul로 정렬**(`:135`)→`norm2=IntLayerNorm`→mlp→qact4 residual(`:141`).
- `VisionTransformer` (`vit_quant.py:146-282`): patch_embed→cls_token cat→pos_embed를 별도 QuantAct로 양자화 후 정수 합(`:264-265`)→blocks→norm→head. **전 경로 정수 + 스케일 전파**.
- 팩토리: `deit_tiny/small/base_patch16_224`, `vit_base/large_patch16_224` (`:285-381`), 모두 `norm_layer=partial(IntLayerNorm, eps=1e-6)`로 지정 (`:293, 314, 334, ...`).

### 3.4 빌딩블록 — `layers_quant.py`
- `Mlp` (`layers_quant.py:116-153`): `fc1=QuantLinear`→`qact_gelu`→`act=IntGELU`→`qact1`→`fc2=QuantLinear`→`qact2(16)` (`:144-153`). GELU 전후로 QuantAct 삽입.
- `PatchEmbed` (`:156-196`): `proj=QuantConv2d`로 패치 투영, flatten/transpose 후 QuantAct(16) (`:172-195`).

### 3.5 학습 파이프라인 — `quant_train.py`
- timm 기반: `create_model`/`create_optimizer`/`create_scheduler`/Mixup/EMA (`quant_train.py:12-17`).
- `str2model(args.model)(pretrained=True, ...)`로 **FP 사전학습 가중치 로드 후 QAT** (`:187-190`).
- 학습 루프(`train`, `:266-311`): `unfreeze_model`(running_stat on) → AdamW + NativeScaler(AMP) → EMA 갱신.
- 검증(`validate`, `:314-`): `freeze_model`(running_stat off로 통계 고정) 후 평가.
- 옵티마이저 기본: AdamW, lr 기본 1e-6(`:80`), epochs 기본 90(`:46`), cosine 스케줄(`:78`), `min_lr = lr/15` (`:202`).

---

## 4. 알고리즘 / 수식

### 4.1 Dyadic requantization `M = b/2^c`
재양자화 시 새 스케일 `S = S_in / S_out`를 정수 가수 m과 시프트 e로 분해(`batch_frexp`, `quant_utils.py:150-175`):

```
S ≈ m / 2^e ,  m = round(frexp_mantissa(S) · 2^31),  e = 31 - frexp_exponent(S)
out = round( z_int · m / 2^e )      # 정수곱 + 우측 산술시프트 (FP 나눗셈 없음)
```
(`fixedpoint_mul.forward`, `quant_utils.py:221-230`). residual 경로도 동일 dyadic 곱으로 정렬 후 정수 덧셈(`:232-245`). → 지시문의 `M = 2^{-c}·b` 형태와 동일.

### 4.2 ShiftGELU 정수 근사 (`int_exp_shift`)
지수 입력 변환 `x ← x + ⌊x/2⌋ − ⌊x/16⌋ ≈ x·log₂(e)` (`quant_modules.py:411`), 정수 몫/나머지 분해 후
`exp_int = (r/2 − x0) << (n − q)` (`:419-420`). GELU는 `x·σ(1.702x)`, σ는 `exp/(exp+exp(−x_max))`를 정수 reciprocal `factor=⌊(2³¹−1)/Σexp⌋`로 계산(`:438-443`).

### 4.3 Shiftmax 정수 근사
동일 `int_exp_shift`(`quant_modules.py:469-481`)로 `exp_int`, 합 `Σexp_int`, reciprocal `factor=⌊(2³¹−1)/Σexp⌋`, 출력 `⌊exp·factor/2^(31−b+1)⌋`, 스케일 `1/2^(b−1)` (`:488-494`).

### 4.4 I-LayerNorm 정수 표준편차 (뉴턴 반복)
`k₀=2¹⁶`, `k_{t+1}=⌊(k_t + ⌊var/k_t⌋)/2⌋`를 10회 → `std_int≈√var` (`quant_modules.py:366-370`). 정수 sqrt를 반복으로 근사.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet (ILSVRC), `--data <YOUR_DATA_DIR>` (`README.md:33, quant_train.py:29`).
- **QAT 명령 예시** (`README.md:28-42`):
  ```bash
  python quant_train.py --model deit_tiny --data <DATA_DIR> --epochs 30 --lr 5e-7
  ```
  - `--model`: deit_tiny/small/base, swin_tiny/small/base (`quant_train.py:25-28`).
  - `--epochs`: 30/60/90 권장 (`README.md:34`), `--lr`: 2e-7~2e-6 권장(`README.md:35`).
- **결과(README.md:44-56)**: INT8(I-ViT) Top-1이 FP32 대비 −0.19~+0.27%. 예: DeiT-S 79.85→80.12(+0.27), ViT-B 84.53→84.76(+0.23).
- 지연시간 평가는 별도 `TVM_benchmark/`(제외 대상)에서 수행.

---

## 6. 의존성

- PyTorch, **timm 0.4.12 권장** (`README.md:15`), numpy, (지연 측정용) **TVM 0.9.dev0 권장** (`README.md:14`).
- timm 컴포넌트 사용: `Mixup, create_model, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, create_scheduler, create_optimizer, NativeScaler, ModelEma, accuracy` (`quant_train.py:12-17`).
- GPU(CUDA) 전제: `IntLayerNorm.forward`에서 `.cuda()` 하드코딩(`quant_modules.py:356`), 스케일 텐서 `.cuda()`(`:440, 494`). → CPU 단독 실행 불가(추정: 코드상 명시적 cuda 호출 근거).

---

## 7. 강점 / 한계 / 리스크

**강점**
- 비선형(GELU/Softmax/LayerNorm)을 포함한 **전 경로 정수·시프트화** → 하드웨어 친화 최고 수준.
- Dyadic requantization으로 스케일 변환을 정수 MAC + 시프트로 환원, FP 유닛 불필요.
- INT8에서 FP 대비 정확도 손실 사실상 0 (`README.md:48-56`).

**한계 / 리스크**
- **QAT 기반** → 전체 ImageNet fine-tuning 필요(수십 epoch). PTQ(AdaLog/APHQ-ViT) 대비 비용 큼.
- 공개 코드의 비선형은 **고정 비트(주로 INT8/INT16)** 중심. 초저비트(W4A4) 검증은 본 repo 범위 밖(추정: README가 INT8만 보고).
- `int_exp_shift`의 정확도 상수 `n`(GELU=23, Softmax=15)이 모델 의존적이라고 주석 명시(`quant_modules.py:399-400, 458-459`) → 모델별 튜닝 필요.
- CUDA 하드코딩으로 이식성 제약(6장).
- weight 양자화가 per-channel만 지원(`quant_modules.py:77, 314`) → per-tensor 필요시 코드 수정.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적; 프로젝트 성격은 추정)

- **(최우선) 정수 전용 비선형 = FPGA LUT/시프트 구현의 직접 청사진**:
  - `IntGELU`/`IntSoftmax`의 `int_exp_shift`(`quant_modules.py:410-423, 469-481`)는 지수항을 **시프트 + 정수곱 + 1회 reciprocal**로 환원 → DSP 없는 소형 LUT(reciprocal) + 시프트 회로로 합성 가능. HG-PIPE류 파이프라인에 GELU/Softmax 블록을 그대로 이식하는 설계 기준이 됨.
  - `IntLayerNorm`의 정수 sqrt 뉴턴 반복(`:366-370`)은 반복형 LN 데이터패스(또는 LUT-sqrt) 설계의 참조 구현.
- **Dyadic requantization(`fixedpoint_mul`/`batch_frexp`, `quant_utils.py:150-261`)**: 레이어 간 스케일 정렬을 `m·z >> e`로 처리 → FPGA에서 "정수 누산 후 가수곱+산술시프트"라는 표준 재양자화 PE의 레퍼런스. residual을 정수 도메인에서 정렬하는 패턴(`quant_modules.py:198-202`)도 파이프라인 누적오차 관리에 유용.
- **스케일 전파 규약**: 모델 전 구간 `(tensor, act_scaling_factor)` 페어 전파(`vit_quant.py` 전반)는 하드웨어 스케일 버스/메타데이터 설계와 1:1 대응 → RTL/HLS 인터페이스 정의에 참고.
- **XR 시선추적 적용(추정)**: 시선추적은 저지연·저전력이 관건이며, I-ViT의 INT8 무손실 + 시프트 비선형은 경량 ViT 백본을 FPGA에서 실시간 구동하기에 적합. 단, QAT 재학습 비용이 있으므로 PTQ(APHQ-ViT)와의 절충 필요.
- **주의**: 본 repo의 비선형 정수 상수(`n=23/15`)·16bit 중간정밀도는 모델별로 재튜닝 대상(추정 근거: 코드 주석 `quant_modules.py:399-400, 458-459`).

---

## 9. 근거 표기 / 확인 불가 항목

- **직접 코드 확인**: §3~§4의 모든 라인 인용(`quant_modules.py`, `quant_utils.py`, `vit_quant.py`, `layers_quant.py`, `quant_train.py`), §2 구조(Glob), §5 명령/결과(`README.md`).
- **추정**: 프로젝트 성격(FPGA 가속기+XR), CPU 실행 불가(코드상 cuda 근거는 확인됨, 실행 실패는 미실행), 초저비트 미검증(README 범위 기반).
- **확인 불가(미열람)**: `swin_quant.py` 세부, `model_utils.py`/`utils.py` 전체, `utils/` 데이터 파이프라인 세부 — Swin은 ViT와 동일 quant 모듈 재사용 구조로 추정되나 라인 단위 미확인. `TVM_benchmark/`는 제외 지시로 미분석.
