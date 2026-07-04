# transformer-quantization (Qualcomm) 정밀 분석

> 분석 대상: `REF/ViT-Quantization/transformer-quantization`
> 작성일: 2026-06-20 / 근거: 실제 소스 코드 (파일:라인 표기)

---

## 1. 개요

- **목적**: BERT 등 Transformer의 양자화 challenge(특히 **activation의 높은 동적 범위와 residual 연결의 구조적 outlier**)를 분석하고, PTQ/QAT 양면으로 극복하는 종합 프레임워크.
- **원논문**: *"Understanding and Overcoming the Challenges of Efficient Transformer Quantization"*, Y. Bondarenko, M. Nagel, T. Blankevoort, **EMNLP 2021** (Qualcomm AI Research). README.md:3-5, ACL Anthology 2021.emnlp-main.627.
- **핵심 기여** (README.md:25 abstract):
  1. Transformer activation에 **residual connection 기반 구조적 outlier**가 존재(특히 [SEP] 토큰 attend 패턴 유발)함을 규명.
  2. 3가지 해법: ① mixed-precision PTQ(특정 텐서만 16-bit), ② **per-embedding-group(PEG) quantization**(임베딩 차원을 그룹화하여 그룹별 양자화), ③ QAT + range learning.
  3. GLUE 벤치마크에서 PTQ SOTA. 가중치/임베딩을 ultra-low bit까지 양자화.
- **양자화 방식**: PTQ(min/max·MSE·cross-entropy range estimator, AdaRound) + **QAT(range learning, learnable scale)** 모두 지원. 대칭/비대칭 uniform quantizer를 STE로 구현.

---

## 2. 디렉토리 구조 (자체 소스 + 제외 항목)

### 자체 핵심 소스
```
transformer-quantization/
├── main.py                          # click CLI 진입점 (train/validate × baseline/quantized)
├── quantization/                    # ★ 양자화 코어
│   ├── quantizers.py                # ★ Asymmetric/Symmetric Uniform Quantizer (STE)
│   ├── range_estimators.py          # ★ MinMax/Running/MSE/CrossEntropy + per-group 추정
│   ├── quantization_manager.py      # ★ Quantizer+RangeEstimator 결합, range 상태머신
│   ├── base_quantized_classes.py    # QuantizedModule/Activation, FP↔양자화 토글
│   ├── base_quantized_model.py      # QuantizedModel 베이스
│   ├── hijacker.py                  # ★ QuantizationHijacker (Linear/Conv forward 가로채기)
│   ├── autoquant_utils.py           # quantize_model: nn.Module→양자화 모듈 자동 변환
│   ├── utils.py
│   └── adaround/                    # AdaRound (가중치 rounding 학습)
│       ├── adaround.py, quantizer.py, utils.py, config.py
├── models/
│   ├── quantized_bert.py            # ★ Quantized BERT (embedding/attention/FFN/LN/GELU)
│   ├── quantized_roberta.py
│   └── quantized_mobilebert.py
└── utils/
    ├── per_embd_quant_utils.py      # ★ per-embedding-group 축/그룹 설정(hijack)
    ├── quant_click_options.py       # 양자화 CLI 옵션
    ├── transformer_click_options.py # 모델/태스크 CLI 옵션
    ├── qat_utils.py, adaround_utils.py, glue_tasks.py, hf_models.py, tb_utils.py
```

### 제외
- `.git/`, `__pycache__/`.
- HuggingFace `transformers` 라이브러리 자체(외부 의존) — `quantized_bert.py`가 import하는 `BertLayer` 등(quantized_bert.py:10-17).
- 체크포인트/모델 산출물(`pytorch_model.bin` 등, 이름만 — README.md:91-134).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 양자화기 — `quantization/quantizers.py`

#### STE 라운딩 (quantizers.py:12-33)
- `RoundStraightThrough`/`FloorStraightThrough`: forward는 round/floor, **backward는 gradient 그대로 통과**(STE) → **QAT 가능**(PSAQ-ViT와 결정적 차이).

#### `AsymmetricUniformQuantizer(QuantizerBase)` (quantizers.py:81-289)
- 비대칭 affine 양자화. 핵심 파라미터: `_delta`(=scale), `_zero_float`(=실수 zero-point) buffer (quantizers.py:101-102).
- `scale` property (quantizers.py:142-147): `scale_domain='linear'`면 `clamp(delta, eps)`, `'log'`면 `exp(delta)` → **log-domain scale 학습 지원**.
- `zero_point` (quantizers.py:149-153): `round_ste(zero_float)` clamp[0, 2^b-1].
- `to_integer_forward` (quantizers.py:172-187): `x_int = round_ste(x/scale) + zero_point`, clamp[int_min, int_max].
- `forward` (quantizers.py:189-211): `x_int` 계산 후 `x_quant = scale*(x_int - zero_point)` (fake-quant). per-axis/per-channel scale broadcast (quantizers.py:213-232).
- `set_quant_range(x_min, x_max)` (quantizers.py:263-282): `delta = (x_max-x_min)/int_max`, `zero_float = -x_min/delta`. x_min은 0과 min, x_max는 eps와 max로 보정(quantizers.py:258-259).
- **`make_range_trainable()`** (quantizers.py:284-288): `_delta`, `_zero_float`를 `nn.Parameter`로 승격 → **range learning(QAT) 핵심**.

#### `SymmetricUniformQuantizer` (quantizers.py:291-349)
- 비대칭을 상속, zero_point=0 (quantizers.py:331-332).
- `signed` 여부를 데이터로 결정(`x_min.min()<0`) (quantizers.py:336). int_min/max가 signed/unsigned에 따라 분기 (quantizers.py:321-328).
- `set_quant_range` (quantizers.py:334-344): `delta = max(|x_min|, x_max)/int_max` (absmax 대칭).
- **가중치=symmetric, 활성=asymmetric**이 권장 설정(README.md:152-153 `--qmethod symmetric_uniform --qmethod-act asymmetric_uniform`).

### 3.2 Range Estimator (observer) — `quantization/range_estimators.py`
양자화 range(x_min/x_max)를 데이터로 추정. 5종:

| 클래스 | 방식 | 라인 |
|---|---|---|
| `CurrentMinMaxEstimator` | 현재 배치 min/max(옵션 percentile, per-group) | 62-145 |
| `AllMinMaxEstimator` | 전 배치 누적 min/max | 148-169 |
| `RunningMinMaxEstimator` | momentum EMA min/max | 172-216 |
| `MSE_Estimator` | 양자화 재구성 **MSE 최소화** clipping(grid/golden-section, 1D/2D) | 228-490 |
| `CrossEntropyEstimator` | 출력 분포 **cross-entropy 최소화**(logits용) | 493-502 |

- **percentile clipping**: `CurrentMinMaxEstimator`에서 `np.percentile(data, (p, 100-p))` (range_estimators.py:121-140) → outlier 절단.
- **MSE 2D search**: 비대칭일 때 clipping range + skew(asymmetry)를 2차원 grid로 탐색 (range_estimators.py:378-420). golden-section(scipy `minimize_scalar`) 옵션 (range_estimators.py:422-470).
- **per-group(PEG) 추정**: `CurrentMinMaxEstimator.forward`에서 `axis`/`n_groups` 지정 시 임베딩 차원을 그룹으로 나눠 그룹별 min/max (range_estimators.py:82-116). **range 기반 permutation**도 지원: `ranges`로 채널을 정렬(argsort)해 유사 range끼리 묶음(`per_group_range_estimation`) (range_estimators.py:68-79, 93-108).

### 3.3 Quantization Manager — `quantization/quantization_manager.py`
- `QuantizationManager(nn.Module)` (quantization_manager.py:19-113): **Quantizer + RangeEstimator를 묶고 range 상태를 관리**.
- 상태(`Qstates`, quantization_manager.py:12-16): `estimate_ranges`(추정), `fix_ranges`(동결), `learn_ranges`(nn.Parameter 학습), `estimate_ranges_train`(train만 추정).
- `forward(x)` (quantization_manager.py:94-106):
  - `per_group_range_estimation`이면 range만 수집하고 x 반환(permutation 사전계산용) (quantization_manager.py:95-97).
  - `estimate_ranges` 상태면 `range_estimator(x)`로 cur min/max 받아 `set_quant_range` (quantization_manager.py:99-104).
  - 그 후 `self.quantizer(x)`로 실제 양자화.
- `learn_ranges()` (quantization_manager.py:82-84): `quantizer.make_range_trainable()` 호출 → scale/zp가 학습 파라미터화.

### 3.4 Hijacker (Linear/Conv 양자화 forward) — `quantization/hijacker.py`
- `QuantizationHijacker(QuantizedModule)` (hijacker.py:18-117): mixin. `class QuantLinear(QuantizationHijacker, nn.Linear)` 식으로 사용(hijacker.py:23-26).
- 생성 시 **weight_quantizer + activation_quantizer**(둘 다 QuantizationManager) 구성 (hijacker.py:44-62). weight range method가 current_minmax면 percentile 옵션 전달(hijacker.py:52-53).
- `forward(x)` (hijacker.py:66-70): `get_params()`(가중치 양자화) → `run_forward`(실제 linear/conv) → `quantize_activations`(출력 양자화).
- `get_params` (hijacker.py:72-86): `_quant_w`면 weight 양자화. 추론 시 양자화 가중치 캐시(`cached_params`).
- `quantize_activations` (hijacker.py:98-116): **활성화 함수(GELU/ReLU 등)를 먼저 적용한 뒤 양자화** (hijacker.py:102-109) → 즉 GELU 출력에 양자화 적용. `activations_list`에 `nn.GELU` 포함(hijacker.py:15) → **GELU는 FP로 계산하되 그 출력을 양자화**.

### 3.5 양자화 BERT 모델 — `models/quantized_bert.py` (★ 구조별 양자화)

#### 임베딩 — `QuantizedBertEmbeddings` (quantized_bert.py:26-88)
- word/position/token_type embedding을 `quantize_model`로 양자화 (quantized_bert.py:36-39).
- **임베딩 가중치 ultra-low bit**: `quant_dict`에 `'Et'`가 있으면 weight range를 **MSE+golden_section**으로 설정(quantized_bert.py:33-35) — 2-bit 임베딩 등(README.md:200 `--quant-dict "{'Et':2}"`).
- **임베딩 합산 지점마다 activation quantizer**: `sum_input_token_type_embd_act_quantizer`, `sum_pos_embd_act_quantizer` (quantized_bert.py:52-53, 79, 84) → residual-합 outlier 양자화 제어.
- **LayerNorm도 양자화**: `self.LayerNorm = quantize_model(org_model.LayerNorm)` (quantized_bert.py:57, 86). → PSAQ-ViT/NoisyQuant와 달리 **LayerNorm을 양자화 모듈로 감쌈**.

#### Self-Attention — `QuantizedBertSelfAttention` (quantized_bert.py:91-218)
- query/key/value Linear 양자화 (quantized_bert.py:110-112).
- **attention score / probs / context 각각 별도 activation quantizer**:
  - `attn_scores_act_quantizer` (q@kᵀ 직후, quantized_bert.py:153-154),
  - `attn_probs_act_quantizer` (softmax 출력, quantized_bert.py:197-198),
  - `context_act_quantizer` (attn@v 후, quantized_bert.py:208-213).
- **softmax는 FP**로 계산하되 그 출력(probs)을 양자화. scale factor `1/√d`는 "이전 act quant delta에 흡수 가능"이라 주석(quantized_bert.py:189).

#### FFN 출력 / residual — `QuantizedBertOutput` (quantized_bert.py:251-280)
- `dense`(Linear) → dropout → **residual 합** → `res_act_quantizer` → LayerNorm (quantized_bert.py:264-277).
- residual 합 직전/직후를 TB 히스토그램으로 기록(`_tb_hist 'res_output_x/h/x_h'`, quantized_bert.py:268-274) → **논문이 지목한 residual outlier를 실제로 관측하는 코드**.

#### GELU(intermediate) — `quantize_intermediate` (quantized_bert.py:283-291)
- `nn.Sequential(dense, gelu)`를 한 번에 양자화 (quantized_bert.py:291). `F.gelu`면 `nn.GELU()`로 변환(quantized_bert.py:287-288). hijacker가 GELU 적용 후 출력 양자화(3.4).

#### mixed precision 진입 — `quant_dict`
- `QuantizedBertForSequenceClassification` (quantized_bert.py:525-555): `quant_setup`으로 분류기 처리 분기:
  - `'MSE_logits'`: logits에 MSE+golden_section range (quantized_bert.py:539-542),
  - `'FP_logits'`: 분류기 출력은 양자화 안 함(`FP32Acts`) (quantized_bert.py:544-549),
  - `'all'`: 전부 양자화 (quantized_bert.py:551-552).

### 3.6 per-embedding-group 설정 — `utils/per_embd_quant_utils.py`
- `_hijack_act_quant(module, value)` (per_embd_quant_utils.py:7-22): `quant_dict` 값에 따라
  - `int`: 비트수 변경(mixed precision) (per_embd_quant_utils.py:11-12),
  - `'fp32'`: 활성 양자화 해제 (per_embd_quant_utils.py:13-14),
  - `'per_embd'`: axis=2, n_groups=None → **임베딩 차원(축2)별 양자화** (per_embd_quant_utils.py:15-16),
  - `'ng{N}'`/`'ngp{N}'`: axis=2로 N개 그룹, permute 여부 (per_embd_quant_utils.py:17-20).
- `set_act_quant_axis_and_groups` (per_embd_quant_utils.py:54-68): quantizer/range_estimator의 `axis`, `n_groups`, `per_group_range_estimation`(permute) 설정.

### 3.7 base 토글 — `quantization/base_quantized_classes.py`
- `QuantizedModule` (base_quantized_classes.py:35-126): `quantized_weights/acts`, `full_precision_*`, `learn_ranges/fix_ranges/estimate_ranges` 등 **전 모델 일괄 상태 전환** API (base_quantized_classes.py:79-111).
- `QuantizedActivation` (base_quantized_classes.py:129-147): 독립 activation quantizer 모듈(임베딩 합·attention 등 중간 지점에 삽입).
- `FP32Acts` (base_quantized_classes.py:150-155): 양자화 우회(특정 텐서 16/32-bit 유지용).

---

## 4. 알고리즘 / 수식

### 4.1 Uniform quantize (STE)
- **비대칭**: `x_q = s·(clamp(round(x/s)+z, 0, 2^b-1) - z)`, `s=(x_max-x_min)/(2^b-1)`, `z=round(-x_min/s)` (quantizers.py:184-211, 263-277).
- **대칭(가중치)**: `s = max(|x_min|, x_max)/int_max`, `z=0`, signed면 int∈[-2^{b-1}, 2^{b-1}-1] (quantizers.py:321-344).
- backward는 STE(round의 grad=1) → **scale/zp/weight 모두 학습 가능**.

### 4.2 Range learning (QAT 핵심)
- `learn_ranges` 상태에서 `_delta`(scale), `_zero_float`(zp)를 `nn.Parameter`로 만들어 **task loss로 직접 최적화**(quantizers.py:284-288, quantization_manager.py:82-84). log-domain scale도 가능(quantizers.py:146-147, 279-281, 341-342).

### 4.3 Range estimation 최적화
- **MSE**: `(x_min,x_max)* = argmin || x - Quant_{x_min,x_max}(x) ||²` — 1D(대칭/단측) 또는 2D(비대칭 range+skew) grid, 또는 golden-section (range_estimators.py:248-470).
- **Cross-entropy(logits)**: `argmin Σ -softmax(x)·log_softmax(Quant(x))` (range_estimators.py:498-502).

### 4.4 Mixed precision (HW 비트할당 직접 매핑)
- `quant_dict` 키별 비트/방식 (README.md:159-173):
  - `'x':16` FFN 입력 16-bit, `'h':16` FFN 출력, `'y':16` FFN residual 합,
  - `'P','C':16` pooler/classifier 16-bit,
  - `'Et':2` 임베딩 2-bit.
- `'ng6'`/`'ngp6'`: 6-group PEG(permutation 유무). → **텐서·차원별 비트/그룹을 선언적으로 지정**.

### 4.5 Per-Embedding-Group (PEG) quantization (논문 신규 기여)
- 임베딩 차원 D를 G개 그룹으로 나눠 **그룹별 독립 scale/zp** (axis=2, n_groups=G) (per_embd_quant_utils.py:15-20, range_estimators.py:87-116).
- **range 기반 permutation**: 유사 range를 갖는 채널끼리 묶도록 argsort permutation 적용(`per_group_range_estimation=True`) → outlier 채널을 한 그룹에 격리, 다른 그룹의 해상도 보존 (range_estimators.py:68-108).

### 4.6 AdaRound (W4A32)
- `quantization/adaround/`로 가중치 rounding(up/down)을 학습(`learned_hard_sigmoid`) (README.md:176-184). 본 분석에서 adaround 내부 코드는 미정독(7장/9장 참조).

---

## 5. 학습/평가 파이프라인

- **데이터셋/태스크**: GLUE (RTE, CoLA, MNLI, MRPC, QNLI, QQP, SST-2, STS-B) (README.md:104-135).
- **모델**: `bert_base_uncased`(주력), RoBERTa, MobileBERT (models/). HuggingFace 체크포인트 사용.
- **진입점**: `main.py` (click), 4개 커맨드 `train-baseline / train-quantized / validate-baseline / validate-quantized` (README.md:57-71).
- **명령어 예시** (README.md):
  - FP fine-tune: `python main.py train-baseline --model-name bert_base_uncased --task rte ...` (README.md:78-81).
  - **W8A8 PTQ**: `validate-quantized --act-quant --weight-quant --n-bits 8 --n-bits-act 8 --qmethod symmetric_uniform --qmethod-act asymmetric_uniform --weight-quant-method MSE --act-quant-method current_minmax --quant-setup all` (README.md:150-156).
  - **Mixed W8A{8,16}**: `--quant-dict "{'y':16,'h':16,'x':16}"` (README.md:159-164).
  - **PEG**: `--per-groups 6 --per-groups-permute` 또는 `--quant-dict "{'x':'ngp6',...}"` (README.md:168-173).
  - **W4A32 AdaRound**: `--adaround all --adaround-mode learned_hard_sigmoid ...` (README.md:176-184).
  - **QAT(W4A8)**: `train-quantized --learn-ranges --n-bits 4 --n-bits-act 8 ...`, 2-bit 임베딩은 `--quant-dict "{'Et':2}"` (README.md:188-200).

---

## 6. 의존성

- `torch==1.4.0`(테스트), `torchvision==0.5.0` (README.md:42-44).
- **HuggingFace `transformers`** — BERT/RoBERTa/MobileBERT 원본 모듈(quantized_bert.py:10-17).
- `scipy`(golden-section `minimize_scalar`, range_estimators.py:10), `numpy`.
- `click`(CLI), TensorBoard(tb_utils). Python ≥3.6 (README.md:36).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **가장 종합적**: PTQ(5종 range estimator + AdaRound) + QAT(range learning) + mixed precision + PEG를 모두 제공.
- **outlier를 정면 분석/처리**: residual outlier를 TB로 관측(quantized_bert.py:268-274)하고 PEG/permutation/mixed-precision으로 대응 — 본 분석 3개 repo 중 outlier 처리가 가장 정교.
- **선언적 비트할당(`quant_dict`)**: 텐서·차원·그룹 단위 비트/방식을 한 줄로 지정 → HW 비트할당 탐색에 직접 매핑(4.4).
- **GELU/LayerNorm/softmax 주변까지 모두 양자화 지점 정의**(quantized_bert.py 전반) → end-to-end 양자화 설계에 근접.
- 가중치 캐시(hijacker.py:81-86)로 추론 효율 고려.

### 한계
- **NLP(BERT) 전용 모델 구현**: ViT 모델은 미포함. ViT 적용은 양자화 코어(quantizers/range_estimators/manager/hijacker) 재사용 + 모델 래퍼 신규 작성 필요(추정).
- 구버전 의존(torch 1.4.0, README.md:42) → 현대 환경 포팅 부담.
- AdaRound 등 일부 모듈은 본 분석에서 미정독(9장).
- PEG permutation/그룹 양자화는 **HW에서 그룹별 scale·채널 재배열을 지원**해야 이득 → 단순 가속기에선 구현 복잡도 증가.
- LayerNorm/softmax는 양자화 입출력만 다루고 연산 자체는 FP → 완전 정수 추론은 아님(I-ViT류와 차이).

### 리스크
- mixed precision 설정이 태스크마다 다름(README.md:157 "settings slightly different per task") → 일반화/튜닝 비용.
- 매우 풍부한 옵션 → 설정 공간이 커서 재현/검증 난이도 높음.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적, 추정)

> 프로젝트 성격은 "HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적"으로 **추정**.

- **HW 비트할당 설계의 직접 참고서**: `quant_dict` 기반 텐서/차원/그룹별 비트 지정(4.4)은 우리 FPGA 가속기의 **layer별·텐서별 mixed-precision 비트 플랜**을 설계할 때 그대로 차용 가능. 특히 "어떤 텐서를 16-bit로 올려야 정확도가 사는가"(x/h/y, P/C, Et)에 대한 경험적 가이드를 제공.
- **outlier 대응 전략 메뉴**: residual outlier가 ViT에도 존재(공통 현상). PEG + range permutation + percentile/MSE clipping은 **저비트 FPGA 정확도 확보의 핵심 도구상자**. NoisyQuant(noisy bias)와 상호보완: 둘 다 outlier 완화이나 접근(그룹화 vs 디더링)이 달라 비교/조합 가치.
- **range estimator 라이브러리 재사용**: MinMax/Running/MSE/CrossEntropy 5종은 우리 캘리브레이션 도구로 거의 그대로 이식 가능(순수 PyTorch). FPGA scale 결정에 MSE/percentile estimator가 특히 유용.
- **QAT/range learning**: XR 시선추적처럼 정확도 요구가 높고 저비트가 필요하면 PTQ만으론 부족할 수 있음 → 본 repo의 range learning(scale/zp 학습)으로 W4A8급 QAT 파이프라인 구성 가능(HG-PIPE 타깃 비트에 맞춤).
- **주의(추정)**: 본 repo는 BERT용이라 ViT/시선추적에 쓰려면 (a) 양자화 코어는 재사용하되 (b) ViT 모델 래퍼와 (c) LayerNorm/GELU/softmax의 정수화(또는 LUT 근사, HG-PIPE의 HW 유닛에 맞춤)를 추가해야 함. integer-only가 목표라면 I-ViT류 결합 필요.
- **PEG의 HW 함의**: 그룹별 scale은 가속기에서 채널 그룹 단위 dequant 지원이 필요 → HG-PIPE 데이터패스에 group-scale 브로드캐스트 경로를 추가하면 정확도/면적 트레이드오프 개선 가능(추정).

---

## 9. 근거 표기 정리

- 모든 동작은 파일:라인 표기(예: quantizers.py:284-288, quantized_bert.py:268-277).
- **추정**: (a) ViT/시선추적 적용 시 필요한 추가 작업; (b) PEG의 HW 구현 함의; (c) 우리 프로젝트 성격 및 비트할당 적용 방향.
- **확인 불가/미정독**: `quantization/adaround/*`(AdaRound 내부), `quantization/autoquant_utils.py`(quantize_model 세부 매핑 — 호출 시그니처는 quantized_bert.py에서 확인), `models/quantized_roberta.py`·`quantized_mobilebert.py`, `main.py`/click 옵션 파일들의 세부. 핵심 양자화 메커니즘은 quantizers/range_estimators/manager/hijacker/quantized_bert로 충분히 파악.
