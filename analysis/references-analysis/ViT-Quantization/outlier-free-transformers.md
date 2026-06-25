# outlier-free-transformers 정밀 분석

> 분석 대상: `REF/ViT-Quantization/outlier-free-transformers`
> 작성 기준: 실제 소스 코드 (파일:라인 근거). 추정/확인 불가 명시.

---

## 1. 개요

- **정체 (확인)**: Qualcomm AI Research, *"Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing"* (Bondarenko, Nagel, Blankevoort, **NeurIPS 2023**)의 공식 구현. README.md:1-19 (arXiv 2306.12929).
- **목적**: Transformer activation의 **strong outlier 발생 원인 자체를 제거**하여 추가 노력 없이 **full INT8 양자화**가 가능하도록 만든다.
- **핵심 통찰**: outlier는 attention head가 "no-op(잔차 보존)"을 하려고 softmax 출력에서 **정확한 0**을 만들려 할 때, softmax 입력을 점점 크게 밀어붙이면서 다른 곳에 outlier가 생긴다는 것. README.md:26-35.
- **두 가지 독립 수정 (핵심 기여)**:
  1. **Clipped Softmax**: softmax 출력을 `[γ, η]`로 stretch 후 `[0,1]`로 clip → 입력을 무한대로 키우지 않고도 정확한 0/1 달성.
  2. **Gated Attention**: head별/토큰별 sigmoid gate를 attention 출력에 곱해 "no-op"을 gate로 처리 → softmax를 극단으로 밀 필요 제거.
- **양자화 방식**: 사전학습(pre-training) 시 위 두 기법 적용 → 학습된 모델을 **PTQ(INT8)**로 검증. 양자화 엔진은 자체 uniform quantizer + range estimator.
- **대상 모델**: BERT(MLM), OPT(CLM) — **언어모델**(ViT 아님). ViT-Quantization 폴더에 있으나 LLM repo임에 유의.

---

## 2. 디렉토리 구조 (자체 + 제외)

```
outlier-free-transformers/
├── run_mlm.py / run_clm.py            # BERT(MLM)/OPT(CLM) 사전학습
├── validate_mlm.py / validate_clm.py  # FP/INT8 검증 (kurtosis·inf-norm 측정)
├── transformers_language/             # ★ 핵심 기법
│   ├── models/
│   │   ├── softmax.py                 # clipped_softmax (★)
│   │   ├── bert_attention.py          # BertSelfAttentionWithExtras: gated attn (★)
│   │   ├── opt_attention.py           # OPT용 gated attn
│   │   ├── quantized_bert.py / quantized_opt.py  # 양자화 래핑
│   │   └── __init__.py
│   ├── quant_configs.py / args.py / dataset_setups.py / utils.py
├── quantization/                      # ★ 양자화 엔진
│   ├── quantizers/
│   │   ├── uniform_quantizers.py      # 균등 양자화 (Asym/Sym)
│   │   ├── base_quantizers.py / quantizer_utils.py
│   ├── range_estimators.py            # min/max·MSE·percentile observer
│   ├── quantization_manager.py / hijacker.py / autoquant_utils.py
│   ├── base_quantized_classes.py / base_quantized_model.py / qstates.py
├── model_configs/  *.yaml             # 모델 크기 설정
├── scripts/  *.sh                     # vanilla/clipped/gated 사전학습 스크립트
├── accelerate_configs/  *.yaml        # HF accelerate
└── docker/                            # Dockerfile, requirements
```

**제외**: `.git/`, `img/`, `__pycache__`. HuggingFace transformers 라이브러리는 외부 의존(이름만).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `transformers_language/models/softmax.py` — Clipped Softmax ★

#### `clipped_softmax(data, dim, eta=1.1, gamma=-0.1)` (8-11)
```python
sm_out = softmax(data, dim)
stretched_out = sm_out * (eta - gamma) + gamma
return torch.clip(stretched_out, 0, 1)
```
- 표준 softmax 결과를 선형으로 `[γ, η]`로 늘린 뒤 `[0,1]`로 clip(9-11).
- `γ<0`이면 작은 확률은 **정확히 0**으로 clip됨 → head가 "no-op"하려 softmax 입력을 +∞로 밀 필요가 사라짐. `η>1`이면 큰 확률을 1로 clip.
- `SOFTMAX_MAPPING`(14-51): `clipped(γ:η)` 문자열 → `partial(clipped_softmax,...)` 다양한 프리셋. 예: `clipped(-.025:1)`은 γ=-0.025, η=1.0 (41).

### 3.2 `transformers_language/models/bert_attention.py` — Gated Attention ★

#### `AttentionGateType` (21-26)
- `none / unconditional_per_head / conditional_per_head / conditional_per_token` 4가지 gate 종류.

#### `BertSelfAttentionWithExtras.__init__` (28-162)
- **clipped softmax 연동**(89-94): `alpha`가 주어지면 `gamma = -alpha/max_seq_length`로 clipped softmax 자동 구성(91-92). 즉 시퀀스 길이에 비례해 clip 강도 설정.
- **gate 정의**(119-162):
  - `unconditional_per_head`: head 수만큼 학습 파라미터 `alpha`(121-122). gate = sigmoid(alpha).
  - `conditional_per_head/token`: head별 작은 FC/MLP 예측기로 입력 의존 gate 생성(124-162). `attn_gate_mlp`이면 2층 MLP(135-141), 아니면 단일 Linear(151). gate 초기 bias를 `logit(attn_gate_init)`로 설정(153-155) → 초기 gate 확률 제어.
- `gate_fn = torch.sigmoid`(109), `pooling_fn = mean(dim=1)`(110).
- `logit(p)`(16-18): gate 초기화용 역시그모이드.

#### `forward` (169-343)
- 표준 BERT self-attention(query/key/value, Q·Kᵀ, scale, mask) (183-272).
- **softmax 교체**(276): `attention_probs = self.softmax_fn(...)` — vanilla 또는 clipped softmax (config에 따라).
- `context_layer = attn_probs @ value`(292) 이후 **gating 적용**(294-333):
  - `unconditional_per_head`(295-299): `context *= sigmoid(alpha).view(-1,1,1)` — head별 스칼라 gate.
  - `conditional_*`(301-331): 입력 hidden state로부터 head/token별 gate를 예측 → `context *= gate · gate_scaling_factor`(327). `last_gate_*`로 gate 분포 로깅(329-331).
- gate가 곧 "head가 얼마나 업데이트할지" → no-op은 gate≈0으로 표현, softmax 극단화 불필요.

> `opt_attention.py`는 OPT(decoder)용으로 동일 패턴(clipped softmax + gated attention) 적용(미정독, **추정** — bert_attention과 동형 구조로 판단).

### 3.3 `quantization/quantizers/uniform_quantizers.py` — 균등 양자화
- (파일 미정독, README/구조 기반) Asymmetric/Symmetric uniform quantizer 제공. validate 명령에서 `--qmethod_acts asymmetric_uniform`, `--ranges_acts running_minmax`, `--percentile 99.999` 사용(README.md:204-210) → activation은 비대칭 균등 + percentile range.

### 3.4 `quantization/range_estimators.py` — Observer
- (구조 기반) running_minmax / MSE / percentile 기반 range 추정. INT8 PTQ calibration용. `--est_num_batches`로 calibration 배치 수 제어(README.md:147).

### 3.5 검증 스크립트 — outlier 정량화
- `validate_mlm.py`/`validate_clm.py`: perplexity뿐 아니라 **inf-norm**과 **Avg/Max Kurtosis**(outlier 지표)를 측정·로깅(README.md:133-139, 187-197). clipped/gated 적용 시 kurtosis가 줄어드는지로 효과 검증.

---

## 4. 알고리즘 / 수식

### 4.1 Clipped Softmax
$$\text{clip-sm}(x)=\text{clip}\Big(\big(\eta-\gamma\big)\cdot\text{softmax}(x)+\gamma,\;0,\;1\Big)$$
- `γ<0`: softmax 확률 `p < -γ/(η-γ)`면 출력 0 (정확한 zero attention) → 입력 logit을 무한대로 키우지 않고도 0 달성. 근거 softmax.py:8-11.
- 논문 설정: `γ = -α / L` (L=max_seq_length), η=1.0. 근거 bert_attention.py:91-92.

### 4.2 Gated Attention
$$\text{out}_h = G_h \odot (\text{Attn}_h \cdot V),\quad G_h=\sigma(\alpha_h)\ \text{or}\ \sigma(f_h(x))$$
- `unconditional`: `G_h=σ(α_h)` head별 학습 스칼라.
- `conditional_per_token`: `G_{h,t}=σ(f_h(x_t))` 토큰별 입력 의존 gate. 근거 bert_attention.py:295-331.

### 4.3 Outlier 억제 메커니즘 (논문 논리)
no-op을 하려는 head는 softmax→0이 필요 → 기존엔 logit→±∞로 밀며 LayerNorm/FFN에 outlier 유발. clipped softmax는 유한 logit으로 0 달성, gated attention은 gate≈0로 no-op을 직접 표현 → 두 경우 모두 logit 극단화 불필요 → activation outlier↓ → INT8 양자화 용이. 근거 README.md:26-35.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: BookCorpus + Wikipedia (`--dataset_setup bookcorpus_and_wiki`, README.md:120).
- **사전학습**: `scripts/*.sh` — bert_base/opt_125m/350m/1.3b × {vanilla, clipped_softmax, gated_attention} (README.md:97-108). A100 80GB 1장 기준.
  - 예: clipped → `--attn_softmax "clipped(-.025:1)"`, `--alpha 12`; gated → `--attn_gate_type conditional_per_token --attn_gate_init 0.25` (README.md:113).
- **검증**: `accelerate launch validate_mlm.py [--quantize] ...` (README.md:115-223). FP/INT8 모두, perplexity + kurtosis/inf-norm 출력.
- **INT8 설정**: `--quant_setup fp32_head --ranges_acts running_minmax --qmethod_acts asymmetric_uniform --percentile 99.999` (README.md:204-210).

---

## 6. 의존성

- PyTorch 1.11 / CUDA 11.3 (README.md:79-81), HuggingFace `transformers` + `accelerate`, `datasets`. docker/requirements.txt. Python ≥3.8.

---

## 7. 강점 / 한계 / 리스크

**강점**
- **outlier를 사후 처리가 아닌 사전(학습 단계) 억제** → INT8에서 perplexity 손실 미미(README.md:133 vs 162: 4.54→4.66).
- 두 기법 모두 **추론 추가 비용 거의 0**(clipped softmax는 곱+clip, gate는 작은 FC). 양자화기와 독립.
- kurtosis/inf-norm 정량 측정 도구 내장 → outlier 효과 검증 재현 가능.

**한계 / 리스크**
- **사전학습 단계 개입 필요**: 이미 학습된 체크포인트에 clipped/gated를 PTQ만으로 적용 불가 (재학습/계속학습 필요).
- **LLM(BERT/OPT) 대상**. ViT/시선추적 직접 적용 코드는 없음 (개념 이식은 가능, **추정**).
- gated attention(conditional)은 head별 FC 추가 → 파라미터·연산 소량 증가.

---

## 8. 우리 프로젝트 관점 시사점 (ViT FPGA 가속기 + XR 시선추적 — 추정)

- **저비트 activation 양자화 정확도의 핵심**: HG-PIPE류 가속기가 INT8 이하 activation을 쓸 때, **outlier가 activation 양자화 정확도의 1차 병목**. 이 repo는 outlier를 만들지 않는 모델을 학습하는 정공법 → 우리 ViT 백본을 학습/파인튜닝할 때 **clipped softmax/gated attention을 도입하면 저비트 activation에서 정확도 방어** 가능(개념 이식, 추정).
- **HW 친화성**: clipped softmax는 기존 softmax datapath에 `×(η-γ)+γ` 와 `clip` 한 단계만 추가 → FPGA에서 거의 무비용. gate는 작은 MAC + sigmoid LUT.
- **ViT로의 이식 주의**: 본 코드는 BERT/OPT 어텐션을 직접 patch한 형태이므로, ViT(timm)용으로 `clipped_softmax`와 gate를 다시 구현해야 함 — softmax.py의 `clipped_softmax`는 그대로 재사용 가능(프레임워크 독립).
- **검증 도구 재사용**: kurtosis/inf-norm 측정 로직은 우리 ViT activation의 outlier 진단에 그대로 응용 가능.

---

## 9. 근거 표기

- repo 정체(Qualcomm/NeurIPS2023): **확인** (README.md:1-19).
- clipped softmax / gated attention 구현: **확인** (softmax.py:8-11, bert_attention.py:89-331).
- uniform_quantizers/range_estimators 내부: **부분 확인** (구조+README 기반, 파일 본문 미정독).
- opt_attention.py 상세: **추정** (bert_attention과 동형으로 판단, 미정독).
- 본 repo는 ViT가 아닌 **LLM(BERT/OPT)** 대상: **확인**. ViT/시선추적 적용 방안: **추정**.
