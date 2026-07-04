# PSAQ-ViT 정밀 분석

> 분석 대상: `REF/ViT-Quantization/psaq-vit`
> 작성일: 2026-06-20 / 근거: 실제 소스 코드 (파일:라인 표기)

---

## 1. 개요

- **목적**: 실제 학습/검증 데이터(real data) 없이 ViT/DeiT/Swin을 PTQ(Post-Training Quantization)하는 **data-free PTQ** 기법.
- **원논문**: *"Patch Similarity Aware Data-Free Quantization for Vision Transformers"*, Zhikai Li et al., **ECCV 2022** (arXiv:2203.02250). README.md:5-8 명시. 저자들 주장으로는 ViT 최초의 data-free quantization 연구.
- **핵심 아이디어**:
  1. ViT의 **self-attention map(Multi-Head Self-Attention 출력)에서 패치 간 유사도(cosine similarity)** 분포가 실데이터 입력 vs Gaussian noise 입력에서 뚜렷이 다르다는 관찰을 이용.
  2. Gaussian noise를 시작점으로, 합성 이미지를 **패치 유사도 분포의 미분 엔트로피(differential entropy)를 최대화**하도록 최적화 → 실데이터와 유사한 attention 통계를 갖는 합성 캘리브레이션 샘플 생성.
  3. 이 합성 샘플로 activation range를 한 번 통과시켜 캘리브레이션 → 가중치/활성 양자화.
- 캘리브레이션 입력만 합성으로 대체하고, **양자화 자체는 표준 min/max 기반 uniform PTQ** (대칭 가중치 + 비대칭 활성).

---

## 2. 디렉토리 구조 (자체 소스 + 제외 항목)

### 자체 핵심 소스
```
psaq-vit/
├── test_quant.py                 # 진입점: 모델 빌드 → 캘리브레이션 → 검증
├── generate_data.py              # ★ 합성 데이터 생성 (patch similarity entropy 손실)
├── models/
│   ├── vit_quant.py              # DeiT/ViT 양자화 모델 (Attention, Block, VisionTransformer)
│   ├── swin_quant.py             # Swin 양자화 모델
│   ├── layers_quant.py           # Mlp, PatchEmbed, HybridEmbed, DropPath, trunc_normal_
│   ├── utils.py
│   └── quantization_utils/
│       ├── quant_modules.py      # ★ QuantConv2d / QuantLinear / QuantAct
│       └── quant_utils.py        # ★ 대칭/비대칭 양자화 함수, scale/zero-point 계산
└── utils/
    ├── kde.py                    # ★ Kernel Density Estimator (엔트로피 추정용)
    ├── build_model.py            # timm 사전학습 모델 로딩
    ├── data_utils.py             # ImageNet dataloader
    └── __init__.py
```

### 제외 (분석 대상 아님)
- `.git/`, `__pycache__/` (관례적 제외)
- `overview.png` (논문 figure, README.md:2 참조)
- timm / torch / torchvision 등 외부 프레임워크 (의존성만, 6장 참조)

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 양자화 연산자 — `models/quantization_utils/quant_modules.py`

세 가지 양자화 모듈을 정의한다. 모두 `quant` 플래그로 FP↔양자화를 토글한다.

#### `QuantConv2d(nn.Conv2d)` (quant_modules.py:12-72)
- 패치 임베딩의 conv를 양자화 (가중치만).
- `self.weight_function = SymmetricQuantFunction.apply` (quant_modules.py:36) — **가중치는 대칭 양자화**.
- forward (quant_modules.py:43-72): `self.quant`가 False면 일반 conv. True면:
  - 가중치를 `(out_channels, -1)`로 reshape 후 행(=출력 채널)별 max/min 추출 (quant_modules.py:58-61) → **per-output-channel 양자화**.
  - `self.weight_function(self.weight, self.weight_bit, v_min, v_max)`로 양자화.

#### `QuantLinear(nn.Linear)` (quant_modules.py:75-115)
- QKV, proj, MLP fc1/fc2, head 등 모든 Linear를 양자화.
- 동일하게 `SymmetricQuantFunction.apply`, 행별(per-channel) min/max (quant_modules.py:105-109).

#### `QuantAct(nn.Module)` (quant_modules.py:118-170)
- **활성(activation) 양자화 + running min/max observer** 역할.
- `self.act_function = AsymmetricQuantFunction.apply` (quant_modules.py:129) — **활성은 비대칭 양자화**.
- buffer로 `x_min`, `x_max` 등록 (quant_modules.py:131-132).
- `forward` (quant_modules.py:151-170):
  - `running_stat`가 True일 때 입력의 `x.data.max()/min()`을 누적하여 `x_max`/`x_min` 갱신 (quant_modules.py:155-163) — **이것이 observer/calibration의 실체**. 단순 글로벌 min/max 추적(percentile 없음).
  - `quant`가 False면 통과만 (range 수집만).
  - `quant`가 True면 누적된 `x_min/x_max`로 비대칭 양자화 수행.
- `fix()`(quant_modules.py:139-143) / `unfix()`(quant_modules.py:145-149): `running_stat`을 끄고/켜서 range를 동결/해제.

> **관찰**: observer가 "단 한 번의 캘리브레이션 forward"에서 본 min/max를 그대로 쓰는 매우 단순한 방식. percentile clipping이나 MSE 최적화가 없다(이 점은 7장 한계 참조).

### 3.2 양자화 수식 구현 — `models/quantization_utils/quant_utils.py`

#### 대칭 가중치 양자화 — `symmetric_linear_quantization_params` (quant_utils.py:31-51)
```
qmax = 2^(num_bits-1) - 1 ,  qmin = -(2^(num_bits-1))
max_val = max(-min_val, max_val)                 # 대칭화
scale   = max_val / ((qmax - qmin)/2)            # = max_val / qmax
zero_point = 0                                    # 대칭이므로 0
```
(quant_utils.py:42-49) — zero-point=0인 전형적 대칭 양자화.

#### 비대칭 활성 양자화 — `asymmetric_linear_quantization_params` (quant_utils.py:54-71)
```
qmax = 2^num_bits - 1 ,  qmin = 0
scale = (max_val - min_val) / (qmax - qmin)
zero_point = qmin - round(min_val / scale)       # clamp(qmin, qmax)
```
(quant_utils.py:66-69) — 0~(2^b-1) 그리드의 표준 비대칭(affine) 양자화.

#### `SymmetricQuantFunction` (quant_utils.py:74-101) / `AsymmetricQuantFunction` (quant_utils.py:104-131)
- `torch.autograd.Function` 서브클래스. forward에서:
  - `quant_x = x/scale + zero_point` → `round().clamp(qmin, qmax)` → `(quant_x - zero_point)*scale` (quant_utils.py:91-95, 121-125) — **fake quantization (quantize-dequantize)**.
- `backward`는 `raise NotImplementedError` (quant_utils.py:100-101, 130-131) → **추론 전용, QAT 불가**. STE 없음.
- `reshape_tensor` (quant_utils.py:7-28): weight는 `(-1,1,1,1)` 또는 `(-1,1)`로, activation은 `(1,1,-1)` 등으로 scale/zero_point를 broadcast → weight는 per-channel(차원0), activation은 per-tensor 형태.

### 3.3 합성 데이터 생성 — `generate_data.py` (★ 핵심 기여)

#### `AttentionMap` 훅 (generate_data.py:20-28)
- `register_forward_hook`으로 특정 모듈의 출력을 캡처. attention map을 가로채는 데 사용.

#### `generate_data(args)` (generate_data.py:31-123)
1. **사전학습 FP 모델 로드**: `build_model(model_zoo[...], Pretrained=True)` (generate_data.py:35).
2. **attention 훅 등록**: Swin은 `m.blocks[n].attn.matmul2`, ViT/DeiT는 `m.attn.matmul2`에 훅 (generate_data.py:39-45). `matmul2`는 `attn @ v` 출력(= attention의 context). vit_quant.py에서는 명시적 `matmul2` 모듈이 없으므로(아래 주석), 이 훅 대상은 모델 정의에 의존 — **확인 필요 지점**(아래).
3. **합성 입력 초기화**: `img = torch.randn((B,3,224,224))`, `requires_grad=True` (generate_data.py:48-49) — Gaussian noise에서 출발.
4. **Adam optimizer로 img 자체를 최적화** (generate_data.py:53). lr=0.20(ViT)/0.25(Swin).
5. **pseudo label**: 무작위 클래스 라벨 (generate_data.py:56), TV(total variation) 목표값 `var_pred` 무작위 (generate_data.py:57).
6. **2 epoch × 500 iter** 최적화 (generate_data.py:62-121). DeepInversion[Yin et al. CVPR'20] 기법 차용:
   - random jitter roll (generate_data.py:81-82), random flip (generate_data.py:84-86).
7. **손실 구성** (generate_data.py:94-114):
   - `loss_oh = CrossEntropy(output, pseudo_label)` — 클래스 사전(one-hot) 손실.
   - `loss_tv = ||TV(img) - var_pred||` — total variation 정규화 (generate_data.py:95, `get_image_prior_losses` 135-143).
   - `loss_entropy`: **각 attention 훅마다**
     - context feature를 head 평균 후 `[:,1:,:]`로 cls 토큰 제외 (patch 토큰만) (generate_data.py:101).
     - 패치쌍 cosine similarity 행렬 `sims` 계산 (generate_data.py:102).
     - `KernelDensityEstimator(sims)`로 유사도 분포 추정 후 (generate_data.py:105-109)
     - `differential_entropy`로 미분 엔트로피 추정, **`loss_entropy -= entropy`** (엔트로피 최대화) (generate_data.py:110-111).
   - 총손실: `loss_entropy + 1.0*loss_oh + 0.05*loss_tv` (generate_data.py:114).
8. **업데이트 후 색 outlier clip** (`clip`, generate_data.py:121, 146-158): ImageNet mean/std 기준 `[-m/s, (1-m)/s]`로 채널별 clamp(픽셀 정규화 범위 유지).
9. 반환: `img.detach()` (generate_data.py:123).

#### `differential_entropy(pdf, x_pdf)` (generate_data.py:126-132)
- `f = -pdf*log(pdf)`를 사다리꼴 적분(`torch.trapz`)으로 수치 적분 → 연속 분포의 미분 엔트로피 추정 (generate_data.py:128-131).

> **확인 불가/주의**: `vit_quant.py`의 `Attention.forward`(vit_quant.py:54-74)에는 `matmul1/matmul2`라는 명시적 서브모듈이 **없고** `attn @ v`가 인라인이다(vit_quant.py:69). `generate_data.py`는 `m.attn.matmul2`(generate_data.py:45)에 훅을 단다. 즉 generate_data는 **별도 모델 빌드 경로**(`build_model`, utils/build_model.py)를 쓰며, 그 경로의 모델은 `matmul1/matmul2`를 명시 모듈로 가진 것으로 **추정**된다(timm 기반 별도 정의일 가능성). 이 불일치는 `utils/build_model.py` 미독으로 **확인 불가**.

### 3.4 KDE — `utils/kde.py`

- `GaussianKernel.forward` (kde.py:90-96): test 점과 train 점의 차이에 Gaussian 커널 적용, 평균 → 밀도 추정. bandwidth 기본 0.01 (kde.py:47).
- `KernelDensityEstimator` (kde.py:104-128): train_Xs(=cosine sim 값들)에 커널을 얹어 test 점(`x_plot`, 10개 linspace)에서 밀도 평가. `differential_entropy`에 입력으로 들어감.

### 3.5 모델 구조 — `models/vit_quant.py`

- `Attention` (vit_quant.py:21-74): `qkv`(QuantLinear) → `qact1`(QuantAct) → QKV split → `attn = q@k^T*scale` → `qact_attn1`로 attention score 양자화 (vit_quant.py:66) → softmax → `attn@v` → `qact2` → `proj`(QuantLinear) → `qact3`.
  - **attention score는 양자화하지만 softmax 출력(확률)은 양자화하지 않음** (vit_quant.py:66-69). softmax/GELU/LayerNorm 자체는 FP로 둠.
- `Block` (vit_quant.py:77-123): `norm1`(FP LayerNorm) → `qact1` → attn → residual `qact2` → `norm2` → `qact3` → MLP → `qact4`. **LayerNorm은 양자화하지 않고, 그 입출력만 QuantAct로 감쌈**.
- `Mlp` (layers_quant.py:116-146): `fc1`(QuantLinear) → GELU(FP) → `QuantAct1` → `fc2` → `QuantAct2`. **GELU는 FP, 입출력만 양자화**.
- `VisionTransformer.model_quant()` (vit_quant.py:254-257): 모든 QuantLinear/QuantConv2d/QuantAct의 `quant=True`. `model_freeze()` (vit_quant.py:259-262): QuantAct의 `running_stat=False`로 range 동결.

### 3.6 진입점 — `test_quant.py`

- `main()` (test_quant.py:73-126):
  - `Config(w_bit, a_bit)` (test_quant.py:44-47, 79)로 비트폭 전달.
  - 모델 빌드 후 `--mode`로 캘리브레이션 데이터 선택 (test_quant.py:94-116):
    - **mode 0 (PSAQ-ViT)**: `generate_data(args)`로 합성 → `model(calibrate_data)` 1회 forward로 range 수집 (test_quant.py:94-99).
    - mode 1: Gaussian noise.
    - mode 2: real data 1 배치.
  - `model.model_quant()` + `model.model_freeze()` (test_quant.py:119-120)로 양자화 활성화 & range 동결.
  - `validate(...)`로 ImageNet top-1/top-5 평가 (test_quant.py:124, 129-177).

---

## 4. 알고리즘 / 수식

### 4.1 합성 캘리브레이션 데이터 최적화
입력 이미지 `x`(처음엔 Gaussian noise)에 대해:

```
min_x   L(x) = L_entropy(x) + λ_oh · L_oh(x) + λ_tv · L_tv(x)
        λ_oh = 1.0,  λ_tv = 0.05            (generate_data.py:114)
```

- **패치 유사도 엔트로피 손실** (핵심):
  - layer l의 attention context `A_l`(head 평균, patch 토큰만) (generate_data.py:101)
  - 패치 i,j 코사인 유사도: `s_{ij} = cos(A_l[i], A_l[j])` (generate_data.py:102)
  - KDE로 `{s_{ij}}` 분포의 pdf `p(s)` 추정 (generate_data.py:105-109)
  - 미분 엔트로피 `H(p) = -∫ p(s) log p(s) ds` (사다리꼴 적분, generate_data.py:128-131)
  - **`L_entropy = -Σ_l H(p_l)`** → 엔트로피 **최대화** (generate_data.py:110-111).
  - 직관: 실데이터의 attention은 패치 유사도가 다양(높은 엔트로피)하나, noise 입력은 유사도가 단조로움(낮은 엔트로피). 엔트로피를 키워 실데이터 통계에 근접시킴.
- `L_oh = CE(f(x), ŷ)` (pseudo label) — 클래스 사전 정보 주입.
- `L_tv = ||TV(x) - var_pred||` — 자연 이미지 평활성 prior.

### 4.2 양자화 수식 (표준 uniform PTQ)
- **가중치(대칭, per-channel)**: `s = max(|w|)/qmax`, `w_q = clamp(round(w/s), -2^{b-1}, 2^{b-1}-1)·s`, zp=0.
- **활성(비대칭, per-tensor)**: `s = (x_max - x_min)/(2^b - 1)`, `z = clamp(round(-x_min/s),0,2^b-1)`, `x_q = (clamp(round(x/s)+z, 0, 2^b-1) - z)·s`.
- range(`x_min/x_max`)는 **합성 데이터 1회 forward**로 수집한 글로벌 min/max (quant_modules.py:155-163).

---

## 5. 학습/평가 파이프라인

- **데이터셋**: 평가는 ImageNet val. **캘리브레이션은 데이터 불필요(mode 0)** — 합성 32장(`calib_batchsize=32`, test_quant.py:23, generate_data.py:32).
- **모델**: DeiT-T/S/B, Swin-T/S (timm 사전학습 가중치, test_quant.py:50-58, vit_quant.py:315-322 등 fbaipublicfiles checkpoint URL).
- **명령어** (README.md:24-54):
  ```bash
  # PSAQ-ViT (data-free)
  python test_quant.py --model deit_base --dataset <DATA_DIR> --mode 0
  # Gaussian noise 비교
  python test_quant.py --model deit_base --dataset <DATA_DIR> --mode 1
  # Real data 비교
  python test_quant.py --model deit_base --dataset <DATA_DIR> --mode 2
  # 비트폭: --w_bit 8 --a_bit 8 (기본), --w_bit 4 등
  ```
- **보고 정확도** (README.md:60-66): DeiT-B W8A8 79.10%, W4A8 77.05%; Swin-S W8A8 76.64% 등.
- 평가 루프: `validate()` (test_quant.py:129-177)에서 top-1/5 측정.

---

## 6. 의존성

- `torch`, `torch.nn`, `torch.optim`, `torch.nn.functional` — 핵심.
- `timm` — 사전학습 모델/데이터 로딩(추정; build_model.py 경유).
- `numpy`, `tqdm`.
- 양자화는 **순수 PyTorch fake-quant**(외부 양자화 라이브러리 불필요).
- 외부 기법 차용: DeepInversion(Yin et al. CVPR'20) jitter/TV (generate_data.py:79-80 주석).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **완전 data-free 캘리브레이션**: 실데이터/라벨 접근 불필요 → 프라이버시·데이터 부재 환경에 적합.
- 합성 손실이 ViT 고유 구조(attention patch similarity)를 직접 겨냥 → Gaussian noise 대비 분명한 정확도 우위(README.md mode 0 vs 1 비교).
- 양자화 코어가 단순·투명(표준 대칭/비대칭 uniform).

### 한계
- **observer가 단순 min/max** (quant_modules.py:155-163): percentile/MSE clipping 없음 → outlier에 취약. activation outlier가 큰 ViT에서 저비트(≤4bit activation) 시 정확도 저하 위험.
- **추론 전용**: `backward`가 `NotImplementedError` (quant_utils.py:100-101) → QAT/range learning 불가, 합성+PTQ만.
- **합성 데이터 생성 비용**: 2 epoch × 500 iter Adam 최적화 + 매 iter 다층 KDE → GPU 시간 소요(추정: 캘리브레이션은 가벼우나 데이터 생성이 무겁다).
- attention 훅 대상(`matmul2`)이 모델 정의에 강결합 (generate_data.py:45) — 모델 변경 시 훅 경로 수정 필요(3.3 확인불가 항목).
- softmax/GELU/LayerNorm을 FP로 유지 → 완전 정수 추론(integer-only) 아님(HW 가속 시 부동소수 유닛 필요).

### 리스크
- W4A8 등 저비트에서 min/max observer + FP softmax 가정이 깨질 경우 정확도 보장 어려움.
- 합성 데이터 품질이 hyperparameter(lr, λ, iteration)에 민감할 수 있음(추정).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적, 추정)

> 우리 프로젝트 성격은 "HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적"으로 **추정**됨.

- **data-free PTQ의 가치**: XR 시선추적은 사용자별·기기별 도메인 편차가 크고 라벨 데이터 확보가 어렵다. PSAQ-ViT식 합성 캘리브레이션은 **현장 캘리브레이션 데이터 없이** 양자화 모델을 배포할 수 있게 해줌 → 온디바이스/엣지 배포 부담 감소.
- **단, HW 관점 한계**: 본 repo의 observer는 단순 min/max라 FPGA 저비트(W4/A4) 가속에는 outlier 대응이 부족. 우리 가속기에는 **percentile/MSE clipping 또는 outlier 완화(NoisyQuant 류)** 를 결합해야 안전(상호보완).
- **integer-only 미지원**: FPGA 가속에서 softmax/GELU/LayerNorm은 별도 처리(LUT/근사)가 필요한데 PSAQ-ViT는 이들을 FP로 둠 → I-ViT 같은 integer-only 기법과 조합 검토 권장(추정).
- **재사용 가능 자산**: `quant_utils.py`의 대칭/비대칭 scale 공식, per-channel weight 양자화 로직은 우리 HW 비트 그리드 설계의 검증용 reference로 직접 활용 가능.
- **합성 데이터 생성기**: 시선추적용 ViT 캘리브레이션 셋을 합성으로 만들 때 patch-similarity entropy 손실 아이디어를 차용 가능(단, 시선추적 입력 분포에 맞춘 prior 재설계 필요 — 추정).

---

## 9. 근거 표기 정리

- 모든 코드 동작은 파일:라인으로 표기(예: generate_data.py:114).
- **추정** 항목: (a) `build_model` 경로 모델이 `matmul2` 모듈을 명시 보유 — utils/build_model.py 미독으로 확인 불가; (b) 합성 데이터 생성 시간 비용; (c) 우리 프로젝트 성격 및 적용 방향.
- **확인 불가**: `utils/build_model.py`, `models/swin_quant.py` 세부, `utils/data_utils.py`는 본 분석에서 정독하지 않음(핵심 흐름은 test_quant.py/generate_data.py/quant_modules.py로 충분히 파악).
