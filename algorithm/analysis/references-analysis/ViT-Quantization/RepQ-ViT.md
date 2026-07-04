# RepQ-ViT 정밀 분석

> 대상 경로: `REF/ViT-Quantization/RepQ-ViT`
> 분석 범위: `classification/` 자체 quant 코드에 집중. `detection/`은 mmdet(외부 객체검출 프레임워크)이므로 **제외(이름만 언급)**.
> 분석 방식: Read한 라인 근거 기반. 추측은 "추정", 미확인은 "확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: 비전 트랜스포머(ViT/DeiT/Swin)에 대한 **극저비트 PTQ(W4/A4, W6/A6)**. 학습 없이 calibration 1배치만으로 4-bit 양자화에서도 정확도를 유지하는 것이 목표.
- **원논문**: Li, Zhikai et al., *"RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers"*, ICCV 2023, pp.17227–17236. (`classification/README.md` L45~L51)
- **핵심 아이디어 2가지 (코드로 확인)**:
  1. **Scale Reparameterization (LayerNorm → 다음 Linear)**: LayerNorm 직후 활성은 채널별 분포 편차가 커서 per-channel scale이 정확하지만 HW에서 비효율적. **calibration은 per-channel로 정확히 하고, 추론 직전에 LayerNorm의 affine(γ,β)과 다음 Linear의 weight/bias로 스케일을 흡수(reparameterize)하여 per-tensor(layer-wise) scale로 변환**한다. 정확도는 per-channel급, 추론은 per-tensor급. (`test_quant.py` L96~L143)
  2. **Log√2 Quantization (post-Softmax)**: softmax 출력의 멱급수적(power-law) 분포를 log 도메인에서 양자화. (`quantizer.py::LogSqrt2Quantizer` L125~L183)
- **양자화 형식**: **asymmetric uniform(affine, zero-point 사용)** 이 기본 — Q-HyViT의 대칭 방식과 대조됨. (`quantizer.py::UniformQuantizer` L18~L122)

---

## 2. 디렉토리 구조 (자체 소스 + 제외 표기)

```
RepQ-ViT/
├── classification/                 [자체 핵심 - 본 분석 대상]
│   ├── test_quant.py               엔트리포인트 + Scale Reparameterization 본체
│   ├── quant/
│   │   ├── quantizer.py            UniformQuantizer, LogSqrt2Quantizer
│   │   ├── quant_modules.py        QuantConv2d/Linear/MatMul
│   │   ├── quant_model.py          모델 치환 + 양자화기 배치 정책
│   │   └── __init__.py
│   └── utils/
│       ├── build_model.py          timm 모델 로드 + attention forward 재정의
│       ├── build_dataset.py        ImageNet 로더/transform
│       └── __init__.py
└── detection/                      [제외 - mmdet 외부 프레임워크]
    ├── mmdet/                       (외부 객체검출 라이브러리)
    ├── configs/, tests/, docs/      (mmdet 부속)
```

- **제외 대상(이름만 언급)**: `detection/mmdet/*` 및 그 configs/tests/docs — open-mmlab mmdetection 기반 외부 프레임워크. VOCdevkit 등 테스트 데이터 포함. 본 분석 미포함.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.0 양자화 공통 패러다임

- **방식: PTQ (무학습)**. calibration 1배치(`calib-batchsize=32`)로 forward 1~2회만 수행. QAT 없음. (`test_quant.py` L72~L93, L146~L148)
- **양자화 트리거: lazy init**. `UniformQuantizer.forward`가 처음 호출될 때 `inited=False`이면 scale/zero-point를 입력 통계로 산출하고 `inited=True`로 고정(L42~L53). 즉 첫 calibration forward가 곧 calibration.
- **bit 설정**: 기본 W4/A4 (`test_quant.py` L34~L37), W6/A6도 지원. weight는 channel-wise, activation은 기본 layer-wise (`test_quant.py` L83~L84).

### 3.1 `quant/quantizer.py` — 양자화기 (핵심)

#### `UniformQuantizer` (L18~L122) — asymmetric affine
- **양자화 식 (forward, L42~L53)**:
  ```
  x_int     = round(x / delta) + zero_point
  x_quant   = clamp(x_int, 0, n_levels-1)        # n_levels = 2^n_bits
  x_dequant = (x_quant - zero_point) * delta
  ```
  → `delta`가 scale, `zero_point`가 offset인 **비대칭(affine)** 양자화. 출력 범위 `[0, 2^b-1]` (unsigned 격자).
- **scale/zero-point 산출 (`init_quantization_scale`, L55~L113)**:
  - **channel-wise (L57~L87)**: weight 전용. tensor 차원에 따라 채널 축 결정 (4D conv: dim0, 2D linear: dim0, 3D activation: 마지막 dim). 채널별로 재귀 호출하여 per-channel delta/zero_point 산출, shape를 broadcast 형태로 reshape (conv `-1,1,1,1` / linear `-1,1` / 3D `1,1,-1`).
  - **layer-wise (L88~L113)**: percentile 탐색. `pct ∈ {0.999, 0.9999, 0.99999}` 각각에 대해 `new_max/new_min`을 quantile로 잡고, L2 loss(`lp_loss`, L8~L15)를 최소화하는 clipping 범위 선택. `delta=(max-min)/(2^b-1)`, `zero_point=round(-min/delta)` (L110~L111). → outlier-robust 비대칭 양자화.
- **`quantize`(L115~L122)**: 주어진 max/min으로 fake-quant(loss 평가용).

#### `LogSqrt2Quantizer` (L125~L183) — post-Softmax 전용 (FPGA 친화 핵심)
- **forward (L143~L151)**: lazy init 후 `quantize`.
- **scale 산출 (L153~L172)**: layer-wise percentile 탐색(L2 loss 최소). softmax 출력은 항상 양수(0~1)이므로 max만 사용.
- **양자화 식 (`quantize`, L174~L183)** — **log base √2 (즉 log2의 2배 해상도)**:
  ```
  x_int   = round( -2 · log2(x/delta) )          # √2 간격 = log2 ×2
  x_quant = clamp(x_int, 0, n_levels-1)
  odd_mask = (x_quant % 2)·(√2 - 1) + 1          # 홀수 레벨 보정(√2 가중)
  x_float_q = 2^(-ceil(x_quant/2)) · odd_mask · delta
  (x_int >= n_levels 인 위치는 0으로)
  ```
  → 지수가 정수(`2^-k`)인 레벨 + √2 보정 레벨이 번갈아 나오는 **log√2 격자**. 곱셈 대신 시프트로 dequant 가능(2의 거듭제곱).

### 3.2 `quant/quant_modules.py` — 양자화 레이어 래퍼

| 클래스 | 라인 | 역할 |
|---|---|---|
| `QuantConv2d` | L13~L76 | conv. input/weight 각각 `UniformQuantizer`. **input 양자화는 항상 8bit 강제**(L37~L39, embedding/patchify 보호). `set_quant_state`로 on/off (L50~L52). |
| `QuantLinear` | L79~L120 | linear. input/weight `UniformQuantizer`. forward에서 plug-in 방식으로 양자화 적용(L105~L120). |
| `QuantMatMul` | L123~L155 | attention 행렬곱. **A/B 비대칭 처리**: `input_quant_params`에 `log_quant` 키가 있으면 A를 `LogSqrt2Quantizer`로(=post-softmax score), 없으면 `UniformQuantizer`. B는 항상 `UniformQuantizer` (L131~L137). |

- **핵심: `QuantMatMul`의 quantizer 분기 (L131~L137)** — matmul2(score@V)에서 A(=softmax score)만 log√2, B(=V)는 uniform. matmul1(Q@Kᵀ)은 둘 다 uniform.

### 3.3 `quant/quant_model.py` — 모델 치환 + 양자화기 배치 정책 (핵심)

`quant_model`(L9~L67):
- 3가지 파라미터 세트 준비(L10~L16):
  - `input_quant_params_matmul2`: `log_quant=True` 추가 → post-softmax용(L11~L12).
  - `input_quant_params_channel`: `channel_wise=True` 추가 → **LogN 후 activation을 per-channel로**(L15~L16).
  - 기본 `input_quant_params`: layer-wise.
- **치환 규칙 (L19~L65)**:
  - `nn.Conv2d` → `QuantConv2d` (patch embedding, L30~L47).
  - `nn.Linear`: 이름에 **`qkv`/`fc1`/`reduction`** 포함 시 **per-channel activation** quantizer 사용(`input_quant_params_channel`, L51~L52), 그 외(proj/fc2/head)는 layer-wise(L54). → **이들이 바로 LayerNorm 직후 레이어** = reparameterization 대상.
  - `MatMul`: 이름에 `matmul2` 포함 시 `QuantMatMul(input_quant_params_matmul2)`(log√2), 그 외 `matmul1`은 uniform (L58~L65).
- `set_quant_state`(L70~L73): 전 모듈 양자화 on/off 일괄 제어.

### 3.4 `test_quant.py` — Scale Reparameterization 본체 (RepQ의 정체성)

- **3단계 실행 (`main`, L51~L157)**:
  1. **Initial quantization (L89~L93)**: per-channel로 정밀 calibration (`fc1/qkv/reduction` activation은 channel-wise). calib forward 1회로 delta/zero_point 산출.
  2. **Scale Reparameterization (L96~L143)** ← 핵심:
     - LayerNorm 모듈(`norm1/norm2/norm`)을 찾고, 그 다음 Linear(`attn.qkv` / `mlp.fc1` / `reduction`)를 `next_module`로 지정(L111~L117).
     - 다음 Linear의 **per-channel** activation scale을 가져옴: `act_delta`, `act_zero_point`, `act_min` (L119~L121).
     - 목표 per-tensor scale = 채널 평균: `target_delta=mean(act_delta)`, `target_zero_point=mean(act_zero_point)` (L123~L125).
     - 보정 계수: `r = act_delta/target_delta`, `b = act_min/r - target_min` (L127~L128).
     - **LayerNorm 흡수**: `LN.weight /= r`, `LN.bias = LN.bias/r - b` (L130~L131).
     - **다음 Linear 흡수**: `Linear.weight *= r`, `Linear.bias += Linear.weight·b` (bias 없으면 생성) (L133~L138).
     - 다음 Linear의 input quantizer를 **layer-wise로 전환**(`channel_wise=False`, delta/zp = target) + weight quantizer 재초기화(`inited=False`) (L140~L143).
  3. **Re-calibration (L146~L148)**: 변환된 모델로 calib forward 1회 → weight 재양자화 + 검증.
- **평가**: `validate`(L160~L208) — ImageNet val top-1/top-5.

> **수학적 불변성**: `LN_out * Linear_w` 결과가 reparameterize 전후 동일해야 함. `(LN_w/r)·(Linear_w·r)`로 r이 상쇄되고 bias 항도 b로 보정 → 출력 동치 유지하면서 activation 분포만 per-tensor 친화적으로 평탄화. (코드 L130~L138로 확인되는 항등 변환)

### 3.5 `utils/build_model.py` / `build_dataset.py`

- `build_model`(L66~L92): timm 모델 로드 후 `Attention`/`WindowAttention`의 forward를 monkey-patch하여 `matmul1=Q@Kᵀ`, `matmul2=score@V`를 `MatMul()` 모듈로 노출(Q-HyViT와 동일 패턴, L11~L58). attention 구조를 양자화 가능하게 분해.
- `build_dataset`(L9~L52): 모델별 정규화(deit/vit/swin) + crop_pct 다르게 적용. val/train(calib) 로더 반환.

---

## 4. 알고리즘 / 수식

### 4.1 Asymmetric Uniform Quantization
```
delta = (max - min) / (2^b - 1)          # scale
zero_point = round(-min / delta)
x_q = clamp(round(x/delta) + zp, 0, 2^b-1)
x̂  = (x_q - zp)·delta
```
clipping(max,min)은 percentile {0.999,0.9999,0.99999} 중 L2 loss 최소값. (`quantizer.py` L106~L122)

### 4.2 Scale Reparameterization (LayerNorm → Linear)
per-channel scale `s_c`(=act_delta_c), 목표 `s̃=mean(s_c)`에 대해:
```
r_c = s_c / s̃,    b_c = act_min_c/r_c - target_min
LN:     γ'_c = γ_c / r_c,    β'_c = β_c/r_c - b_c
Linear: W'_{·,c} = W_{·,c}·r_c,   bias' = bias + W'·b
```
→ 출력 동치, activation scale을 per-tensor `s̃`로 통일. (`test_quant.py` L127~L138)

### 4.3 Log√2 Quantization (post-Softmax)
```
q = clamp( round(-2·log2(x/Δ)), 0, 2^b-1 )
x̂ = 2^(-ceil(q/2)) · [ (q mod 2)(√2-1)+1 ] · Δ
```
log2 대비 2배 해상도(√2 간격), 짝수 레벨은 순수 2의 거듭제곱(시프트), 홀수 레벨은 √2 보정. (`quantizer.py` L174~L183)

### 4.4 양자화기 배치 정책 요약
| 위치 | activation quantizer | 비고 |
|---|---|---|
| patch embed conv | Uniform(8bit, asym) | input 항상 8bit |
| qkv / fc1 / reduction (LayerNorm 직후) | Uniform **channel-wise** → reparam 후 layer-wise | RepQ 핵심 |
| proj / fc2 / head | Uniform layer-wise | |
| weight (전체) | Uniform **channel-wise** | |
| matmul1 (Q@Kᵀ) A,B | Uniform | |
| matmul2 (score@V) A | **Log√2** | post-softmax |
| matmul2 B (V) | Uniform | |
(`quant_model.py` L51~L65, `quant_modules.py` L131~L137)

---

## 5. 학습 / 평가 파이프라인

엔트리포인트 `classification/test_quant.py` (CLI: `--model --dataset --w_bits --a_bits`, README L9~L24):
1. seed 고정 → dataloader(`build_dataset`) → calib 1배치 추출(L72~L75).
2. `build_model` (timm + attention 분해).
3. `quant_model` 치환(wq channel-wise, aq layer-wise).
4. `set_quant_state(True,True)` → calib forward 1회(initial quant).
5. **Scale Reparameterization** (LN↔Linear 흡수).
6. Re-calibration forward 1회.
7. `validate` → ImageNet top-1/5.
- **QAT 없음**, gradient/backprop 없음(Q-HyViT의 hessian backward와 대조). calibration 비용이 매우 낮음.
- 결과(README L30~L38): DeiT-S W4/A4 69.03%, W6/A6 78.90% 등.

---

## 6. 의존성

- `torch`, `timm`(모델 로드 + Attention/WindowAttention), `torchvision`(ImageNet ImageFolder), `numpy`, `PIL`.
- 데이터: ImageNet(train=calib, val=eval).
- 버전 명시는 classification README에 없음(확인 불가). detection은 mmdet 의존(제외).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **W4/A4 극저비트에서 무학습으로 동작** — calibration 1배치, backprop 없음 → 매우 빠르고 가벼운 PTQ.
- **Scale Reparameterization이 HW 친화의 정석**: per-channel 정확도를 얻으면서 추론은 per-tensor(layer-wise) scale → 채널별 scale 곱셈기/저장 불필요. **추론 오버헤드 0(항등 변환)**.
- **Log√2 softmax 양자화는 시프트 기반 dequant 가능** → 곱셈기 절감(FPGA/ASIC 친화).
- 코드가 간결(quantizer 184줄, modules 156줄)하고 정책이 명확 → 이식·이해 용이.

**한계 / 리스크**
- fake-quant 시뮬레이션(float 복원). 실제 정수 커널/HW 매핑 코드는 없음(`integer.py` 같은 추출기 부재).
- weight 양자화가 항상 channel-wise이며 reparam 대상이 아님 → weight per-channel scale은 HW에서 그대로 비용으로 남음(추정).
- percentile 탐색이 고정 3개 값({.999,.9999,.99999})으로 단순 → 일부 분포에서 suboptimal 가능.
- LayerNorm이 없는 구조(BN 기반 하이브리드 등)에는 reparameterization 적용 불가 — Q-HyViT가 다룬 하이브리드/grouped-conv 모델에서 RepQ가 붕괴하는 이유와 일치(Q-HyViT README 결과표에서 Mobile-Former RepQ ≈ 0%대).
- Swin은 `blocks` 대신 `layers` 슬라이싱 등 모델별 분기 하드코딩(`test_quant.py` L99).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + HG-PIPE 계열 + XR 시선추적)

> 전제: HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적(추정).

- **Scale Reparameterization = FPGA 가속기의 정석 패턴**: LayerNorm 직후 activation은 채널 편차가 커서 단일 per-tensor scale로는 정확도 손실이 크다. RepQ는 이를 **컴파일타임에 LN 파라미터로 흡수**해 런타임 per-tensor로 만든다. 우리 가속기에서 **requantization 유닛을 per-tensor(scalar) 한 개로 단순화**할 수 있는 강력한 근거. HG-PIPE 류 파이프라인에서 채널별 scale broadcast 회로 제거 가능.
- **Log√2 softmax 양자화 → 시프트 기반 dequant**: `2^(-ceil(q/2))·odd_mask` 형태는 짝수 레벨에서 순수 비트 시프트, 홀수 레벨에서만 √2 상수 곱. softmax/attention 분모 처리를 **곱셈기 대신 barrel shifter + 1개 √2 상수 mul**로 구현 가능 → DSP 절감. attention이 병목인 ViT 가속기에 직접 이득.
- **비대칭(zero-point) 양자화의 HW 비용**: RepQ는 affine(zero_point)을 쓰므로 정수 MAC 후 zero-point 보정항(`-zp·Σw` 등)이 필요. FPGA에서 zero-point는 bias로 사전 folding 가능 — RepQ의 reparam이 이미 bias로 흡수하는 패턴(`test_quant.py` L133~L138)을 zero-point folding으로 확장 적용 검토 가치.
- **dyadic scale 관점**: RepQ의 uniform delta는 임의 float이지만, log√2 부분은 본질적으로 2의 거듭제곱(dyadic)이라 HW 친화. 우리 가속기에서 **weight/act uniform scale을 dyadic(2^-n)으로 근사 + softmax는 log√2** 조합이 면적-정확도 최적점일 수 있음(추정).
- **Q-HyViT와의 상호보완**: RepQ는 표준 ViT(DeiT/Swin)·LayerNorm 구조에서 극저비트(W4A4)에 강하지만 BN 기반 경량 하이브리드에서 붕괴. Q-HyViT는 하이브리드에 강하나 dyadic이 아님. **표준 ViT 백본 → RepQ, 경량 하이브리드 백본 → Q-HyViT** 의 백본별 양자화 전략 분기가 우리 멀티-백본 가속기 설계에 합리적(추정).
- **XR 시선추적**: 저지연/저전력이 핵심. RepQ의 W4A4는 메모리·대역폭을 1/8로 줄여 XR 엣지에 유리. 단 시선추적 백본이 LayerNorm 기반 ViT일 때만 reparam 이득이 직접 발생(BN 기반이면 Q-HyViT 쪽 참조).
- **주의**: 본 repo도 정수 커널/latency 정보 없음 → HW 성능은 합성·구현으로 별도 검증 필요(확인 불가).

---

## 9. 근거 표기 규칙

- **[코드 확인]**: 본문의 모든 `파일.py Lxx~Lyy`는 실제 Read 라인 근거. asymmetric uniform 식, channel-wise/layer-wise 분기, log√2 양자화, scale reparameterization 변환식, matmul2 분기, 양자화기 배치 정책은 모두 코드로 직접 확인.
- **[추정]**: (a) weight per-channel scale의 HW 잔여 비용, (b) dyadic+log√2 혼합 최적점, (c) 백본별 RepQ/Q-HyViT 전략 분기, (d) zero-point→bias folding 확장 — 코드 근거 기반 합리적 추론이나 직접 명시 없음.
- **[확인 불가]**: (a) classification 의존성 버전, (b) 실제 FPGA latency/면적/전력, (c) 정수 end-to-end 커널(이 repo는 fake-quant 시뮬레이터까지만), (d) 원논문 PDF(이 repo에는 PDF 미동봉, README 인용만 존재) 내부 수식 전체.
