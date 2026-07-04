# Q-HyViT 정밀 분석

> 대상 경로: `REF/ViT-Quantization/q-hyvit`
> 분석 방식: 자체 핵심 소스(quant_layers / utils / configs / example)를 함수·클래스 단위로 Read하여 라인 근거 기반 작성. 추측은 "추정", 미확인은 "확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: 효율적 하이브리드 비전 트랜스포머(MobileViTv1/v2, Mobile-Former, EfficientFormerV1/V2)에 대한 **Post-Training Quantization(PTQ)**. CNN+Transformer 혼합 구조 + 선형복잡도 attention을 가진 모바일/IoT용 경량 모델을 양자화 대상으로 한다. (`README.md` L1~L5)
- **원논문**: Lee, Jemin et al., *"Q-HyViT: Post-Training Quantization for Hybrid Vision Transformer with Bridge Block Reconstruction for IoT Systems"*, arXiv:2303.12557, 2024. (`README.md` L76~L81). 동봉 PDF: `q-hyvit-paper.pdf` (존재 확인, 본 분석은 코드 기반).
- **핵심 아이디어 (README L4~L5 기준)**: 기존 ViT용 PTQ(EasyQuant, FQ-ViT, PTQ4ViT, RepQ-ViT)를 하이브리드 트랜스포머에 적용하면 4가지 문제로 정확도가 급락한다 — (i) highly dynamic ranges, (ii) zero-point overflow, (iii) diverse normalization, (iv) limited model parameters(<5M). 이를 해결하기 위해 **하이브리드 양자화 입도(granularity) 자동 선택 + Bridge Block Reconstruction(블록 단위 재구성)**을 제안. 8-bit에서 평균 17.73%, 6-bit에서 29.75% 개선을 보고.
- **코드 계보**: 코드 베이스는 **PTQ4ViT**와 **FQ-ViT**에서 차용 (`README.md` L68~L69). 따라서 `quant_layers/`의 PTQSL(Power-of-Two/Sub-Layerwise) 구조, twin-uniform softmax 양자화, search 기반 interval 탐색은 PTQ4ViT 유산이며, Q-HyViT 고유 기여는 주로 `conv.py::hyvit_Conv2d`, `quant_calib.py::batching_hybrid_calib`, configs의 hybrid 설정에 들어 있다.

---

## 2. 디렉토리 구조 (자체 소스 + 제외 표기)

```
q-hyvit/
├── quant_layers/              [자체 핵심]
│   ├── matmul.py              attention 내부 QK / score·V 행렬곱 양자화
│   ├── conv.py                Conv2d 양자화 (hyvit_Conv2d 포함)
│   └── linear.py              Linear 양자화 (QKV/proj/MLP/classifier)
├── utils/                     [자체 핵심]
│   ├── quant_calib.py         calibration 엔진 (Hessian, hybrid, bridge block)
│   ├── net_wrap.py            timm 모델 → quant 모듈 치환
│   ├── integer.py             int8/uint8 추출(배포용)
│   ├── models.py              attention forward 재정의, get_net
│   ├── datasets.py            ImageNet 로더 (보조)
│   └── mobileformer/          [제외 - Mobile-Former 외부 모델 정의/체크포인트]
├── configs/                   [자체 핵심]
│   ├── BasePTQ.py             baseline (cosine, a_bit=32)
│   ├── PTQ4ViT.py             PTQ4ViT 재현 (twin softmax/gelu ON)
│   └── Block4Hybrid.py        Q-HyViT 메인 설정 (hessian, hyvit_Conv2d)
└── example/
    └── test_qhyvit.py         메인 실험 엔트리포인트
```

- **제외 대상(이름만 언급)**: `utils/mobileformer/*` (외부 Mobile-Former 모델 정의·registry·체크포인트), timm 라이브러리(외부), ImageNet 체크포인트.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.0 양자화 공통 패러다임 (전 모듈 공통)

- **방식: PTQ (학습 없음)**. 모든 양자화 모듈은 `mode` 상태 머신을 가진다: `raw` → `calibration_step1`(FP32 입출력 수집) → `calibration_step2`(interval 탐색) → `quant_forward`(시뮬레이션 추론). (`linear.py` L36~L47, `conv.py` L43~L54, `matmul.py` L25~L36)
- **양자화 형식: symmetric, signed, fake-quant**. scale에 해당하는 변수는 `interval`이며 zero-point는 사용하지 않음(대칭). 양자화 식: `round(x/interval).clamp(-qmax, qmax-1) * interval`. `qmax = 2**(bit-1)` (예: 8bit → 128). (`linear.py` L29~L30 L60~L63; `conv.py` L36~L37 L67~L70; `matmul.py` L16~L17 L38~L41)
- **scale(interval) 산출**: MinMax 초기값은 `abs().max()/(qmax-0.5)` 형태(0.5 여유). (`linear.py` L99~L100; `conv.py` L88~L89; `matmul.py` L67~L68)
- **fake-quant 구조**: 정수로 내림 후 다시 interval을 곱해 float로 복원 → 실제 정수연산이 아니라 정확도 시뮬레이션. 실제 int8 추출은 `utils/integer.py`가 담당.

### 3.1 `quant_layers/linear.py` — Linear 양자화

| 클래스 | 라인 | 역할 |
|---|---|---|
| `MinMaxQuantLinear` | L6~L103 | 기본 대칭 PTQ Linear. weight/activation 각각 layer-wise MinMax interval. `bias_correction` 옵션(L80~L88) 지원. |
| `PTQSLQuantLinear` | L105~L271 | **Sub-Layerwise** weight 양자화 + search 기반 interval 탐색. weight를 `n_V × n_H` 블록(L128~L129), activation을 `n_a` 그룹(L130)으로 나눠 블록별 interval. |
| `PostGeluPTQSLQuantLinear` | L273~L358 | GELU 직후 활성(비대칭 분포) 전용. 양/음 분리 양자화: 양수는 `a_interval[0]`(per-group), 음수는 고정 음수 interval `0.16997.../qmax`(L331). |
| `PTQSLBatchingQuantLinear` | L360~L567 | 위의 메모리 절약형(batched calibration). GPU 메모리 한도(3GB)에 맞춰 `parallel_eq_n` 자동 산정(L382~L389). pearson 유사도 별도 구현(L438~L465). |
| `PostGeluPTQSLBatchingQuantLinear` | L569~L654 | PostGelu + batching 결합. `a_neg_interval` 고정(L586). |

- **scale 계산 핵심**: weight interval은 `self.weight.view(n_V,crb_rows,n_H,crb_cols).abs().amax([1,3])/(w_qmax-0.5)` (L243). activation interval은 `n_a` 그룹별 amax (L244).
- **search 알고리즘 (EasyQuant 류)**: interval 후보를 `[eq_alpha, eq_beta]` 구간을 `eq_n` 등분한 배율로 생성(L258~L259), 각 후보에 대해 `_get_similarity`(L135~L161)로 raw 출력과의 유사도(L2/cosine/pearson/hessian 등)를 최대화하는 interval을 argmax 선택(L182~L236).
- **bias correction**: `_bias_correction_quant_forward`(L80~L88)에서 양자화 오차의 평균을 bias에서 차감.

### 3.2 `quant_layers/conv.py` — Conv 양자화 (Q-HyViT 고유 부분 포함)

| 클래스 | 라인 | 역할 |
|---|---|---|
| `MinMaxQuantConv2d` | L9~L92 | 기본 대칭 PTQ Conv. |
| `QuantileQuantConv2d` | L94~L127 | quantile(0.9999) 기반 interval(outlier 완화). 대용량 텐서는 청크 quantile(L114~L119). |
| `PTQSLQuantConv2d` | L129~L281 | weight를 `(oc,ic·kw·kh)`로 펼쳐 `n_V×n_H` 블록 분할 후 search. |
| `BatchingEasyQuantConv2d` | L283~L445 | layer-wise EasyQuant batched 버전. |
| `ChannelwiseBatchingQuantConv2d` | L448~L657 | **per-channel weight** (`n_V = out_channels`, L472). weight interval shape `oc,1,1,1` (L498). `a_bit>=32`이면 activation 양자화 OFF(L558 L647). |
| **`hyvit_Conv2d`** | **L659~L891** | **Q-HyViT 고유 클래스**. per-channel weight + **activation을 layer-wise/channel-wise 자동 선택**. |

- **`hyvit_Conv2d._initialize_intervals` (L699~L734) — 핵심 기여**:
  - weight: per-channel (`amax([1,2,3])`, L704).
  - activation: `init_activation_layerwise` 플래그로 layer-wise(L726) vs channel-wise(`amax([0,2,3])`, L723, L728) 전환. **이것이 "diverse normalization / dynamic range" 대응**.
  - **zero-channel 안전장치**(L731~L734): interval이 0인 채널을 `float32 eps`로 치환해 division NaN 방지 → README의 "zero-point overflow / 극단적 동적범위" 대응 코드 근거.
- **per-channel search 지원**: `_search_best_a_interval`(L810~L854)이 layer-wise면 단일 스칼라, channel-wise면 채널별 best index를 선택(L845~L851). grouped conv(`groups>1`)도 청크 루프로 처리(L790~L797).

### 3.3 `quant_layers/matmul.py` — Attention 행렬곱 양자화 (구조 핵심)

| 클래스 | 라인 | 역할 |
|---|---|---|
| `MinMaxQuantMatMul` | L8~L71 | `A@B` 대칭 PTQ. A/B 각각 interval(`A_interval`,`B_interval`). |
| `PTQSLQuantMatMul` | L73~L293 | head-wise / 블록(`n_G,n_V,n_H`) padding 후 search. hessian 지원. |
| `SoSPTQSLQuantMatMul` | L295~L399 | **Split-of-Softmax(SoS)** — softmax 출력(score) 전용 twin-uniform. |
| `PTQSLBatchingQuantMatMul` | L401~L588 | head-wise batched (`n_G_A=heads`, L426~L427). GPU 메모리 한도(3GB) 기반 batch 산정(L407~L420). |
| `SoSPTQSLBatchingQuantMatMul` | L590~L656 | SoS + batching. |

- **Attention 양자화 구조 (2개 matmul)**: attention은 `matmul1 = Q@Kᵀ`, `matmul2 = score@V` 두 단계로 명시적 분리(`models.py` L16, L22). 각각 별도 quant 모듈로 치환된다(net_wrap의 `qmatmul_qk`, `qmatmul_scorev`).
- **Split-of-Softmax(SoS) (L296~L327)**: softmax 후 score는 (0,1) 구간에 극단적으로 편향 → 단일 uniform으로는 표현 불가. 구간을 `split` 기준 high/low 두 영역으로 나눠 각각 uniform 양자화(L324~L327). split 후보는 `2^(-i)` (L380), search로 최적값 선택(L329~L355). 주석 L307: "with proper hardware implementation, we don't need to use a sign bit anymore" → 양수 전용 8bit(부호비트 불필요) 의도. **FPGA 친화 시사점 직결**.
- **head-wise 양자화**: `PTQSLBatchingQuantMatMul._get_padding_parameters`가 `n_G_A=A.shape[1]`(=heads)로 설정(L426) → head별 독립 scale.
- **유사도 metric (`_get_similarity`, L158~L186 / L453~L493)**: cosine, pearson, L1/L2_norm, linear/square_weighted_L2, **hessian**(grad²·오차², L180~L182 L485~L489) 지원.

### 3.4 `utils/net_wrap.py` — 모델 치환

- `wrap_modules_in_net`(L39~L82): timm 모델의 모든 모듈을 순회하며 `nn.Conv2d`→qconv, `nn.Linear`→이름 기반 매핑(`module_types` dict L42~L43), `MatMul`→qmatmul로 치환.
- **모듈 이름 매핑(L42~L43)**: `qkv→qlinear_qkv`, `proj→qlinear_proj`, `fc1/fc2→qlinear_MLP_1/2`, `head/fc→qlinear_classifier`, `matmul1/2→qmatmul_qk/scorev`, `reduction→qlinear_reduction`. 하이브리드 모델용 키(`0/2/q/k/channel_mlp→qlinear_feature_hyper`)가 추가됨(L43) → Mobile-Former 등의 비표준 레이어 이름 대응(Q-HyViT 확장).
- `fold_bn_into_conv`(L8~L36): BN을 conv에 fold(추론 시 BN 제거). CNN 부분의 양자화 친화 전처리.

### 3.5 `utils/quant_calib.py` — Calibration 엔진 (Bridge Block 핵심)

| 클래스/메서드 | 라인 | 역할 |
|---|---|---|
| `QuantCalibrator.sequential_quant_calib` | L28~L55 | 순차 2-step calibration(메모리 친화). |
| `QuantCalibrator.parallel_quant_calib` | L57~L93 | 병렬(메모리 다소비). |
| `HessianQuantCalibrator.quant_minmax_calib` | L217~L280 | hook으로 raw 수집 후 MinMax interval만 초기화(빠른 초기값). |
| `HessianQuantCalibrator.quant_calib` | L283~L365 | **Hessian PTQ**: raw softmax를 타깃으로 KL-div loss backward → grad 수집(`grad_hook` L173~L177) → hessian metric으로 interval search. |
| **`batching_hybrid_calib`** | **L367~L454** | **Q-HyViT 메인 calibration**. |
| `batching_quant_calib` | L456~L545 | hybrid 없는 batched hessian. |

- **Hessian 기반 search**: `loss = KL(log_softmax(pred), raw_pred_softmax)` (L325, L403) 의 grad를 `raw_grad`로 저장(L340~L341). search 시 hessian metric은 `-(grad·(raw-sim))²` (matmul.py L485~L489) → 출력 민감도 가중 양자화.
- **`batching_hybrid_calib` 의 hybrid 입도 자동 선택 (L420~L435)**: Conv 모듈에 대해 `groups==1`이면 activation **layer-wise**(`init_activation_layerwise=True`, L426~L427), grouped conv면 **channel-wise**(L429~L431). → README의 "highly dynamic ranges / diverse normalization" 대응의 구체적 코드.
- **Bridge Block Reconstruction (L443~L448, L534~L539)**: 인자 `index=[...]`로 지정된 블록 인덱스(q.n)는 calibration 후에도 `quant_forward`로 유지, 나머지는 `raw`로 되돌림(주석 "Bridge Block (v2)"). 즉 블록 단위로 양자화 영향을 순차 누적·재구성하는 구조. block_list는 `test_qhyvit.py` L126~L150에서 모델별로 하드코딩(예: efficientformer_l1 → [26,27,28]).
- **loss surface 측정**(L397~L410): 블록별 cumulative KL loss 출력(디버그/분석용).

### 3.6 `utils/integer.py` — 실제 정수 추출 (배포용, FPGA 관점 중요)

- `quantize_int_weight`(L8~L18): weight를 **int8**로 추출(`round(w/interval).clamp(-128,127).to(int8)`).
- `quantize_int_activation`(L44~L110): activation을 int8/uint8로 추출하는 pre-forward hook.
  - PostGelu: 양수→uint8+128, 음수→abs uint8, 합쳐 **twin uint8** (L61~L67). MSB가 sign 역할(주석 L53~L54).
  - SoS softmax: high→uint8+128, low→uint8, MSB가 "큰/작은 interval 선택" 역할(주석 L51~L52, L85~L96).
  - 일반 matmul/linear: int8 (L74~L75, L104~L110).
- `quantize_matmul_input`(L27~L41): sublayerwise padding 고려한 matmul 입력 정수화.

### 3.7 `utils/models.py` — Attention forward 재정의

- `attention_forward`(L10~L26): timm `Attention`의 forward를 monkey-patch. `q@kᵀ`를 `self.matmul1(q, k.transpose(-2,-1))`로(L16), `attn@v`를 `self.matmul2(attn,v)`로(L22) 치환 → matmul을 양자화 가능 모듈로 노출.
- `window_attention_forward`(L28~L56): Swin WindowAttention용(상대위치 bias 포함).
- `get_net`(L62~L107): timm/Mobile-Former 모델 로드 후 Attention/WindowAttention에 `MatMul()` 주입 + forward 교체.

---

## 4. 알고리즘 / 수식

### 4.1 기본 대칭 양자화 (uniform symmetric)
```
qmax = 2^(b-1)
interval(scale) s = max(|x|) / (qmax - 0.5)
x_q = clamp(round(x / s), -qmax, qmax-1)
x_dequant = x_q * s        (zero-point = 0)
```
(`linear.py` L60~L63, L99~L100)

### 4.2 Search 기반 interval 최적화 (EasyQuant/PTQ4ViT 류)
후보 `s_i = (eq_alpha + i·(eq_beta-eq_alpha)/eq_n) · s_init`, i=0..eq_n. 각 `s_i`로 양자화한 출력과 raw 출력의 유사도 최대화:
```
s* = argmax_{s_i}  Similarity(raw_out, quant_out(s_i))
```
(`linear.py` L258~L264; `conv.py` L268~L274; `matmul.py` L276~L283)

### 4.3 Hessian-aware metric
```
Similarity = - Σ (∂L/∂out)² · (out_raw - out_sim)²,   L = KL(softmax(pred_q) || softmax(pred_fp))
```
(grad 수집: `quant_calib.py` L325 L340; metric: `matmul.py` L485~L489) — PTQ4ViT의 hessian-guided 양자화 계승.

### 4.4 Split-of-Softmax (twin-uniform for softmax)
```
x_high = round(clamp(x, split, 1)·(qmax-1)) / (qmax-1)          # 큰 값 구간
x_low  = round(clamp(x, 0, split)/a_interval)·a_interval        # 작은 값 구간 (a_interval = split/(qmax-1))
x_q = x_high + x_low,   split ∈ {2^-i}
```
(`matmul.py` L324~L327, L341~L355, L380)

### 4.5 Twin-uniform for post-GELU (양/음 분리)
```
x_pos = round(clamp(x, 0, ·)/a_interval_pos)·a_interval_pos
x_neg = round(clamp(x, ·, 0)/a_interval_neg)·a_interval_neg     # a_interval_neg = 0.16997.../qmax (고정)
x_q = x_pos + x_neg
```
(`linear.py` L613~L619, 음수 interval 상수 L586/L331)

### 4.6 Hybrid quantization (Q-HyViT 고유)
- weight: per-channel.
- activation: `groups==1` → layer-wise, `groups>1`(depthwise/grouped) → channel-wise 자동 선택. (`quant_calib.py` L420~L435; `conv.py::hyvit_Conv2d._initialize_intervals` L699~L734)

### 4.7 Bridge Block Reconstruction
지정된 블록 인덱스 집합 `index`에 대해서만 양자화를 활성 유지, 나머지는 raw로 되돌려 블록 단위로 점진 재구성. (`quant_calib.py` L443~L448, `test_qhyvit.py` block_list L126~L150)

> **주의(확인 불가/추정)**: README가 말하는 "Bridge Block Reconstruction"의 완전한 수식적 정의(블록 경계에서의 reconstruction loss 형태)는 코드상 `index` 기반 모드 토글과 cumulative KL loss 출력까지만 확인됨. 정확한 reconstruction objective는 `q-hyvit-paper.pdf` 참조 필요(코드만으로는 부분 확인).

---

## 5. 학습 / 평가 파이프라인

엔트리포인트 `example/test_qhyvit.py::experiment_basic` (L100~L157):
1. config 로드(`init_config`, L89~L97) — 기본 `Block4Hybrid`.
2. `get_net(net)` (models.py) — timm/Mobile-Former 로드 + MatMul 주입.
3. `net_wrap.wrap_modules_in_net` — quant 모듈 치환.
4. calib_loader = ImageNet 32장(`g.calib_loader(num=32)`, L111).
5. `HessianQuantCalibrator(sequential=False, batch_size=4)` (L113).
6. `quant_minmax_calib()` — MinMax 초기값(L116).
7. 모든 모듈 `mode="raw"`, grad 리셋(L120~L122).
8. 모델별 block_list 결정(L126~L150).
9. **`batching_hybrid_calib(block_list)`** — hybrid+hessian+bridge calibration(L152).
10. `test_classification` — ImageNet val top-1 평가(L31~L50, L156).
- 다중 GPU 멀티프로세스 실험 지원(L53~L86, `--multiprocess`).
- **QAT 없음**: 전 과정 무학습 PTQ, calibration 32장만 사용.

---

## 6. 의존성

- `torch==1.13.1`, `timm==0.9.2` (`README.md` L9~L11).
- `pandas`, `tqdm`, `numpy`.
- 모델: timm(대부분 자동 다운로드) + Mobile-Former 사전 weight 동봉(`utils/mobileformer/`).
- 데이터: ImageNet(ILSVRC12).
- 코드 계보: PTQ4ViT + FQ-ViT (`README.md` L68~L69).

---

## 7. 강점 / 한계 / 리스크

**강점**
- 하이브리드 모델(MobileViT, Mobile-Former, EfficientFormer)을 실제로 다룬 드문 PTQ. README 결과표 기준 W8A8/W6A6에서 PTQ4ViT/RepQ-ViT 대비 큰 격차(특히 Mobile-Former에서 타 방법은 사실상 붕괴, Q-HyViT는 50~75% 유지).
- per-channel/per-layer hybrid 자동 선택 + zero-channel 안전장치로 극단적 동적범위 견딤.
- twin-uniform(softmax/GELU)로 비대칭·편향 분포 대응. 부호비트 절감 의도(HW 친화).
- int8/uint8 실제 추출 경로(`integer.py`) 제공 → 배포/HW 연계 가능.

**한계 / 리스크**
- **fake-quant 시뮬레이션**: `quant_forward`는 float로 복원해 추론(실제 정수 행렬곱 아님). 정확도 검증용이며, 실제 정수 파이프라인은 별도 구현 필요.
- bias는 미양자화(`bias_bit is None` assert, `linear.py` L22). HW에서 bias 처리 별도 고려.
- search 비용 큼(eq_n=100, search_round=3, hessian backward) → calibration 시간/메모리 부담.
- block_list가 모델별 하드코딩(`test_qhyvit.py`) → 신규 모델 적용 시 수작업 필요.
- 코드 내 다수 `#jemin` 주석/주석 처리된 실험 코드 → 연구용 프로토타입 성격, 정리 미흡.
- Bridge Block reconstruction의 정밀 정의는 코드만으로 부분 확인(논문 의존).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적)

> 전제: 우리 프로젝트는 HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적(추정).

- **정수연산 추출 경로(`utils/integer.py`)가 직접 유용**: int8 weight/activation, twin-uint8(softmax/GELU)를 텐서로 뽑는 코드가 이미 있음 → FPGA용 정수 데이터 패킹의 출발점. MSB를 sign/interval-select로 쓰는 패킹(L51~L54)은 HW LUT/mux 설계와 직결.
- **Split-of-Softmax(SoS)의 "부호비트 제거"(matmul.py L307, L601)**: softmax 출력은 항상 0~1 양수이므로 unsigned 8bit로 표현 → FPGA에서 sign 처리 회로 절감. 단, twin-interval(split high/low)은 2-region이라 HW에서 region 판정 비교기 + 2개 scale mux 필요(비용 vs 정확도 트레이드오프 검토 대상).
- **per-channel vs per-tensor 선택 로직(hyvit_Conv2d, hybrid calib)**: FPGA에서 per-channel scale은 채널별 scale 저장/곱셈기 필요(면적↑) vs per-tensor는 단일 scale(면적↓, 정확도↓). Q-HyViT가 `groups==1→layer-wise, grouped→channel-wise`로 나눈 휴리스틱은 우리 가속기에서 **레이어 타입별 scale 입도 정책**의 참고가 됨.
- **dyadic/shift 양자화 관점에서의 갭**: 이 repo는 scale이 **임의 float interval**(dyadic 아님)이라 FPGA에서는 requantization 시 부동소수 곱 또는 fixed-point multiplier+shift 근사가 필요. RepQ-ViT의 log2/dyadic 접근이 HW에 더 친화적(RepQ-ViT.md 참조). Q-HyViT의 정확도 우위 + RepQ의 dyadic scale을 **혼합**하는 방향이 유망(추정).
- **Hessian-aware bit/granularity 선택**: 출력 민감도 기반 가중은 우리가 "어느 레이어를 더 높은 비트/정밀 scale로 둘지" 결정하는 HW 자원 배분 정책에 응용 가능.
- **XR 시선추적 관점**: 시선추적 백본은 보통 경량 CNN/하이브리드 → Q-HyViT가 다루는 <5M 파라미터 모바일 하이브리드와 타깃이 정확히 겹침. 경량 모델에서 PTQ 정확도 유지 노하우(zero-channel 안전장치, hybrid 입도)가 직접 이식 가치 있음.
- **주의**: 본 repo는 정확도 시뮬레이터일 뿐 latency/throughput/면적 정보 없음 → HW 성능은 별도 합성·구현으로 검증 필요(확인 불가).

---

## 9. 근거 표기 규칙

- **[코드 확인]**: 위 본문의 모든 `파일.py Lxx~Lyy` 표기는 실제 Read한 라인 근거. 양자화 식, search 구조, hybrid 선택, twin-uniform, int 추출, attention matmul 분리는 모두 코드로 직접 확인.
- **[추정]**: (a) Bridge Block의 정확한 reconstruction objective, (b) Q-HyViT + RepQ dyadic 혼합 유망성, (c) XR 시선추적 백본과의 타깃 일치 — 코드/README 기반 합리적 추론이나 직접 명시 없음.
- **[확인 불가]**: (a) `q-hyvit-paper.pdf` 내부 수식 전체(본 분석은 코드 우선), (b) 실제 FPGA latency/면적/전력, (c) 정수 파이프라인 end-to-end 동작 여부(시뮬레이터까지만 코드 존재).
