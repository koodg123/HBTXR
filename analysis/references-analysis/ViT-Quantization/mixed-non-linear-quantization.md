# mixed-non-linear-quantization 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\mixed-non-linear-quantization`
> 분석 방식: `quant_test.py`, `models/mivit/{vit_quant.py, vit_layers_quant.py, utils.py, model_utils.py}`, `models/mivit/quantization_utils/{quant_modules.py, quant_utils.py, fqvit_quant_modules_wrapper.py}`, `models/ivit/quantization_utils/quant_modules.py`(IntGELU/IntSoftmax/IntLayerNorm) 라인 정독.
> **정체**: 정수-only ViT 양자화(I-ViT/I-BERT/FQ-ViT) 3종 비선형 구현을 **레이어별로 혼합(mixed)** 선택하는 자체 실험 프레임워크. 커널 언어 = **순수 PyTorch(.py)**, CUDA/Triton 커스텀 커널 없음.

---

## 1. 개요 (목적/원논문/핵심 아이디어)

- **목적**: ViT(DeiT/Swin)의 **비선형 연산(Softmax, GELU, LayerNorm)**을 정수-only로 근사하되, 세 가지 선행연구 구현(I-ViT, I-BERT, FQ-ViT)을 **레이어 단위로 자유 조합(mixed non-linear quantization)**하여 정확도/하드웨어 비용 trade-off를 탐색.
- **원논문/기반**:
  - **I-ViT**(Li & Gu, ICCV 2023): ShiftGELU / Shiftmax / I-LayerNorm — bit-shift 기반 정수-only.
  - **I-BERT**(Kim et al., ICML 2021): 2차 다항식 기반 정수 GELU/Softmax.
  - **FQ-ViT**(Lin et al., IJCAI 2022): Log2 softmax + Power-of-Two Factor(PTF) LayerNorm.
  - `mivit`("Mixed ViT", `models/mivit/README.md`:1 "> Mixed ViT")가 자체 기여 = 세 구현의 레이어별 혼합.
- **핵심 아이디어**: YAML config로 각 블록의 norm1/norm2/attn.softmax/mlp.gelu에 `I-ViT|I-BERT|FQ-ViT`를 지정 → 런타임에 해당 정수 모듈을 동적 바인딩.

---

## 2. 디렉토리 구조 (자체 + 제외)

```
mixed-non-linear-quantization/
├── quant_test.py                 # ★ 메인 평가 진입(자체) — I-ViT/I-BERT/MI-ViT 선택, ImageNet val
├── manager.sh                    # 실행 스크립트
├── models/
│   ├── ivit/                     # I-ViT 정수-only 모듈(선행연구 기반)
│   │   ├── quantization_utils/quant_modules.py  # ★ IntGELU/IntSoftmax/IntLayerNorm/QuantAct/...
│   │   ├── quantization_utils/quant_utils.py    # STE, fixedpoint_mul, batch_frexp
│   │   ├── vit_quant.py, swin_quant.py, layers_quant.py, model_utils.py
│   ├── ibert/                    # I-BERT 정수 모듈(선행연구 기반)
│   │   └── quantization_utils/{quant_modules.py, quant_utils.py, quantize_model.py}
│   ├── fqvit/                    # FQ-ViT(PTQ) — observer/quantizer 포함
│   │   ├── ptq/observer/{minmax,ema,omse,percentile,ptf}.py
│   │   ├── ptq/quantizer/{uniform,log2}.py, bit_type.py, layers.py
│   │   └── vit_quant.py, swin_quant.py, layers_quant.py, config.py
│   └── mivit/                    # ★★ 자체 핵심: Mixed ViT
│       ├── README.md             # "Mixed ViT"
│       ├── vit_quant.py          # ★ Attention/Block — 레이어별 non_linear 선택
│       ├── vit_layers_quant.py   # ★ Mlp — gelu I-BERT/그외 분기
│       ├── swin_layers_quant.py, swin_quant.py
│       ├── utils.py              # ★ generate_non_linear_layers_pack(yaml→class)
│       ├── model_utils.py        # freeze/unfreeze
│       └── quantization_utils/
│           ├── quant_modules.py            # ★ FQ-ViT 스타일 IntLayerNorm/IntSoftmax(log2)
│           ├── fqvit_quant_modules_wrapper.py # ★ 동일(FQ-ViT non-linear wrapper)
│           └── quant_utils.py              # STE, fixedpoint_mul, batch_frexp(I-ViT 동일 계열)
├── utils.py                      # dataloader/build_dataset 등(미정독)
─ 제외: .git/, **/__pycache__/(.pyc)
```

- **커스텀 HW 커널 없음**: 전부 PyTorch autograd Function(STE). FPGA/CUDA 커널 미포함. 정수 연산은 float 텐서 위에서 round/floor로 시뮬레이션.
- `mivit/quantization_utils/quant_modules.py`와 `fqvit_quant_modules_wrapper.py`는 **내용 동일**(FQ-ViT 스타일 IntLayerNorm/IntSoftmax) — wrapper로 재노출.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 Mixed 선택 메커니즘 — `models/mivit/utils.py::generate_non_linear_layers_pack` (24-44)

```python
softmax_pack = {'I-ViT': ivit.IntSoftmax, 'FQ-ViT': fqvit.IntSoftmax, 'I-BERT': ibert.IntSoftmax}  # (7-11)
norm_pack    = {'I-ViT': ivit.IntLayerNorm,'FQ-ViT': fqvit.IntLayerNorm,'I-BERT': ibert.IntLayerNorm} # (12-16)
gelu_pack    = {'I-ViT': ivit.IntGELU,     'FQ-ViT': nn.GELU,          'I-BERT': ibert.IntGELU}      # (17-21)

def generate_non_linear_layers_pack(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))           # (25-26)
    for layer_name, model in cfg.items():          # 예: 'blocks[0].attn.softmax': ['I-ViT']
        if "softmax" in layer_name: layer = softmax_pack[model]   # (33-34)
        elif "gelu" in layer_name: layer = gelu_pack[model]       # (35-36)
        elif "norm" in layer_name: layer = norm_pack[model]       # (37-38)
        non_linear_layers_pack[layer_name] = layer; non_linear_cfg[layer_name] = model
    return non_linear_layers_pack, non_linear_cfg
```
- **이것이 "mixed"의 본질**: YAML로 레이어별 비선형 구현을 지정 → 클래스 dict로 매핑. `--mivit-cfg` 인자(`quant_test.py`:31)로 경로 전달.
- 주의: **gelu의 'FQ-ViT'는 정수 GELU가 아니라 일반 `nn.GELU`(FP)** — FQ-ViT가 GELU를 정수화하지 않는 점 반영(17-21). 즉 mixed bit/precision 할당이 op마다 정수↔FP까지 포함.

### 3.2 mixed Attention/Block — `models/mivit/vit_quant.py`

**Attention.__init__** (18-61):
- softmax 출력 bit가 cfg에 따라 달라짐(55-58):
  ```python
  if non_linear_cfg['blocks[idx].attn.softmax'] == 'FQ-ViT':
      self.int_softmax = non_linear_pack[...](6)    # FQ-ViT softmax는 output_bit=6
  else:
      self.int_softmax = non_linear_pack[...](16)   # I-ViT/I-BERT는 16
  ```
- QKᵀ·PV는 `QuantMatMul`(정수 matmul), scale 전파(`act_scaling_factor`)로 연결(74-83). `attn = attn*self.scale`, `act_scaling_factor *= scale`(76-77) — 1/√d를 scale에 합산.

**Block.__init__/forward** (94-160+):
- norm2 bit가 cfg 따라 32(I-BERT) vs 16(그외)(129-132).
- **FQ-ViT LayerNorm은 out_scale 입력이 필요** → forward 분기(156-160):
  ```python
  if cfg['blocks[idx].norm1'] == 'FQ-ViT':
      x, asf = self.norm1(x_1, asf_1, self.qact1.act_scaling_factor)  # out_scale 전달
  else:
      x, asf = self.norm1(x_1, asf_1)
  ```
  FQ-ViT 경로는 사전에 `qact1.act_scaling_factor += 1`로 초기화(148-152).

### 3.3 mixed Mlp(GELU 분기) — `models/mivit/vit_layers_quant.py` (7-54)
- `is_ibert_gelu = (cfg['blocks[idx].mlp.gelu']=='I-BERT')`(28-29).
- forward(44-53): I-BERT GELU가 아니면 GELU 전에 `qact_gelu`로 한 번 더 양자화(46-48). I-BERT GELU는 자체 내부 양자화 포함이라 생략.

### 3.4 정수 비선형 구현 (핵심 알고리즘)

#### (A) I-ViT 계열 — `models/ivit/quantization_utils/quant_modules.py`

**IntLayerNorm (I-LayerNorm)** (333-387):
- 정수 평균/분산 후 **정수 뉴턴 반복으로 √var 근사**(366-371):
  ```python
  k = 2**16
  for _ in range(10):
      k = floor((k + floor(var_int/k))/2)   # Newton iteration for sqrt
  std_int = k
  factor = floor((2**31-1)/std_int)         # (373)
  y_int = floor(y_int*factor/2); scaling_factor = dim_sqrt/2**30  # (374-375)
  ```
  → 나눗셈·sqrt 없이 정수-only LayerNorm. bias도 정수화(379-383).

**IntGELU (ShiftGELU)** (390-446):
- `int_exp_shift`(411-424): exp를 **bit-shift로 근사** — `x_int += floor(x/2) - floor(x/2**4)`(≈×1.4375 보정), `q=floor(x/x0_int)`, `r=x-x0·q`, `exp ≈ (r/2 - x0)·2^(n-q)`(420-421). n=23.
- forward(426-446): GELU≈x·sigmoid(1.702x) → sigmoid를 exp_shift로 정수 근사, `sigmoid_int = floor(exp_int·factor/2^(31-bit+1))`, scale=`1/2^(bit-1)`(440-441).

**IntSoftmax (Shiftmax)** (449-498):
- `int_exp_shift` 동일(470-482), n=15. forward(484-498): `x_int -= max`, `exp_int = exp_shift(x_int)`, `exp_int_sum` clamp 2³¹-1, `factor=floor((2³¹-1)/sum)`, `exp_int = floor(exp_int·factor/2^(31-bit+1))`, scale=`1/2^(bit-1)`(492-495). **나눗셈을 factor 곱+shift로 대체**.

#### (B) FQ-ViT 계열 — `models/mivit/quantization_utils/quant_modules.py`

**IntLayerNorm(PTF 기반)** (10-76):
- Power-of-Two Factor: `in_scale_mask = round(in_scale/in_scale.min())`로 채널별 scale을 2의 거듭제곱 정수배로 정렬(56-58).
- `get_MN`(21-25): scale A를 `M·2^-N`(고정소수점 mantissa M[8bit]/exponent N)으로 분해 → `x_q = round((A_sign·M·x_q + B)/2^N)`(67-73). **정수 곱+shift LayerNorm**.

**IntSoftmax(Log2 quantization)** (79-151):
- I-BERT식 2차 다항 exp(`int_polynomial` coef=[0.358,0.970,1.0])(102-122).
- **핵심: log2 양자화**(132-151): `softmax_out = round(exp_sum/exp_int)`(역수), `rounds = log_round(softmax_out)`(I-log2, 94-100), `deq_softmax = 2^(-qlog)`(143). → softmax 출력을 **log2 도메인 정수**로 저장, dequant은 2의 거듭제곱. output_bit로 clamp.

#### (C) 공통 양자화 유틸 — `quant_utils.py`(ivit/mivit 동일 계열)
- `SymmetricQuantFunction`(102-149): 대칭 양자화 `round(x/scale)` clamp `[-n-1, n]`, STE backward.
- `fixedpoint_mul`(208-287): scale을 `m·2^-e`(`batch_frexp`, 179-205, `np.frexp` + 31bit mantissa)로 분해 → **정수 requantization을 HW 정수연산과 일치**시킴(`output = round(z_int·m/2^e)`)(254-256). identity(residual) 가산 지원(258-271).
- `floor_ste`/`round_ste`(153-176): STE.

### 3.5 평가 진입 — `quant_test.py`
- `--type {I-ViT, I-BERT, MI-ViT}`(33), `--model {vit/deit/swin _tiny/small/base}`(34-38), `--bit`(32), `--mivit-cfg`(31).
- `str2model`(154-191): type별 모델 팩토리. MI-ViT는 `mivit_cfg` 전달(234-240).
- `validate`(352-428): ImageNet val top-1/5, `ivit.freeze_model`(367)로 activation range 고정. heatmap(attention rollout) 생성 옵션(400-428).
- **device `cuda:1` 하드코딩**(214, quant_utils 곳곳 `.cuda('cuda:1')`) — 재현 시 주의.

---

## 4. 알고리즘 / 수식 (mixed nonlinear bit 할당)

**Mixed 할당 모델**: 각 블록 b의 비선형 연산 집합 `{norm1, norm2, attn.softmax, mlp.gelu}`에 대해 구현 `m ∈ {I-ViT, I-BERT, FQ-ViT}`와 그에 따른 output_bit를 YAML로 지정.

- **bit 할당 규칙**(코드 근거):
  - attn.softmax: FQ-ViT→6bit(log2), I-ViT/I-BERT→16bit(`vit_quant.py`:55-58).
  - norm2 후 qact: I-BERT→32bit, 그외→16bit(`vit_quant.py`:129-132).
  - mlp.gelu: I-ViT→ShiftGELU(8bit), I-BERT→다항 GELU, FQ-ViT→FP GELU(`utils.py`:17-21).
- **정수 비선형 수식 요약**:
  - I-ViT Softmax: `exp ≈ shift-based`, `out = floor(exp·⌊(2³¹-1)/Σexp⌋ / 2^(31-bit+1))`, scale=`2^-(bit-1)`.
  - FQ-ViT Softmax: `out = 2^(-Ilog2(Σexp/exp))`, scale=`2^-(2^bit-1)`(log2 도메인).
  - I-ViT LayerNorm: `std`=정수 Newton-sqrt, `y = floor(y·⌊(2³¹-1)/std⌋/2)`.
  - FQ-ViT LayerNorm: PTF + `M·2^-N` 고정소수점.
- **scale 전파**: `act_scaling_factor`가 레이어 체인을 따라 흐르며 `fixedpoint_mul`로 정수 requant(scale=`m·2^-e`). 1/√d는 scale에 흡수(`vit_quant.py`:76-77).

---

## 5. 학습/평가 파이프라인 (데이터셋/벤치/명령어)

- **데이터셋**: ImageNet(`--data-set IMNET`, 1000 classes, 224px) 또는 CIFAR(39-43). val 평가 위주(`validate`).
- **모델**: timm 사전학습 ViT/DeiT/Swin tiny~base(`create_model`, `--resume` 체크포인트 로드, 270-294).
- **명령어**(추정):
  ```bash
  python quant_test.py --type MI-ViT --model deit_tiny --bit 8 \
      --mivit-cfg <layer_config.yaml> --resume <ckpt> --data /dataset/imagenet/
  ```
- 평가 산출: top-1/top-5 정확도, (옵션) attention heatmap PNG(top-k 이미지).
- 학습 루프는 main에서 호출 비활성(epoch 0 validate만, 296-301) — **평가/분석 중심**.

---

## 6. 의존성

- PyTorch, timm(data/models/optim/scheduler/utils), numpy, yaml, (heatmap) opencv/PIL.
- GPU `cuda:1` 하드코딩(다수). 별도 CUDA/Triton 빌드 불필요(순수 PyTorch 시뮬레이션).

---

## 7. 강점 / 한계 / 리스크

**강점**
- 정수-only 비선형 3종(I-ViT/I-BERT/FQ-ViT)을 **레이어 단위로 혼합** 가능한 통합 프레임워크 — 어느 레이어에 어느 근사가 정확/저비용인지 탐색에 최적.
- 모든 비선형이 정수 연산(shift/다항/log2)으로 근사 → HW 매핑 직접성 높음. `fixedpoint_mul`로 HW 정수 requant 정확 모사.
- Softmax 출력 6bit(FQ-ViT log2)~16bit(I-ViT) 등 op별 비트 차등 → mixed precision 비트 할당 실험 토대.

**한계 / 리스크**
- 자동 bit/구현 할당 탐색기(search) 부재 — YAML 수동 지정(NAS/sensitivity 자동화 없음).
- `cuda:1` 하드코딩, FQ-ViT 경로의 `act_scaling_factor += 1` 같은 ad-hoc 보정 → 재현/이식성 저하.
- 커스텀 HW 커널 없음(전부 float 시뮬레이션) — 실제 정수 처리량/지연은 별도 RTL/HLS 필요.
- 정확도 결과 표가 코드에 없음(README "Mixed ViT" 한 줄) — 성능 수치 **확인 불가**.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적)

- **비선형 LUT 비트 설계의 직접 청사진**: I-ViT ShiftGELU/Shiftmax는 exp를 `floor(x/2)-floor(x/2⁴)` + shift로 근사 — FPGA에서 **소형 LUT + barrel shifter**로 그대로 구현 가능. FQ-ViT Log2 softmax는 출력을 `2^-qlog`(6bit)로 두어 후속 PV GEMM을 **shift-only**로 만들 수 있어 DSP를 크게 절감(HG-PIPE 어텐션 후단 설계에 매우 유리).
- **Mixed precision 비트 할당 → FPGA 자원 배분**: 레이어/op별로 softmax 6bit, norm 16/32bit를 차등하는 구조는 FPGA에서 LUT 폭·고정소수점 비트를 op별로 다르게 합성하는 전략과 1:1 대응. 정확도 민감 레이어만 고비트로 두어 LUT/BRAM 예산 절감.
- **fixedpoint_mul(`m·2^-e`) requant**: scale을 mantissa·shift로 분해하는 방식은 FPGA requantization 유닛(정수 곱 + 우측 시프트) 설계의 정확한 레퍼런스. 1/√d, layer scale을 모두 이 형태로 흡수 가능.
- **I-LayerNorm 정수 Newton-sqrt**: FPGA에서 reciprocal-sqrt LUT 대신 반복 회로(10 iter)로 대체 가능 — 면적/지연 trade-off 설계 선택지 제공.
- **XR 시선추적**: 시선추적 ViT는 작은 모델(deit_tiny/swin_tiny)이 주력 — 이 repo가 다루는 정확히 그 크기대. 비선형을 정수-only로 두면 XR SoC/FPGA에서 부동소수 유닛 없이 어텐션 전체를 정수 파이프라인으로 통합 가능(전력·면적 이득).

---

## 9. 근거 표기 / 불명확 사항

- **정체**: I-ViT/I-BERT/FQ-ViT 정수 비선형을 레이어별 혼합하는 자체 실험 프레임("Mixed ViT", `mivit/README.md`:1). 커널 = 순수 PyTorch, 커스텀 CUDA/Triton **없음(확인)**.
- 세 비선형 모듈(ivit/ibert/fqvit)은 **선행연구 구현 기반**(I-ViT/I-BERT/FQ-ViT 논문) — 본 repo 고유 기여는 `mivit`의 혼합 메커니즘. ibert/fqvit 모듈 내부는 mivit 경유 호출만 정독, 전체 라인은 **부분 확인**.
- 정확도/벤치 수치는 코드·README에 부재 → **확인 불가**.
- `mivit/quantization_utils/quant_modules.py` ≡ `fqvit_quant_modules_wrapper.py`(동일 내용) — wrapper 재노출로 **확인**.
- `cuda:1` 하드코딩은 원저자 실험 환경 흔적으로 **추정**.
