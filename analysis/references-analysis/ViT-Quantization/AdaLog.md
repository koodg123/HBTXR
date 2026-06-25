# AdaLog 정밀 분석 (Post-Training Quantization for ViT with Adaptive Logarithm Quantizer)

> 분석 대상 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\AdaLog`
> 분석 방식: 자체 핵심 소스 라인 단위 정밀 분석 (Glob/Grep/Read). bash 미사용.
> 근거 표기 규칙: `파일:라인`. 직접 확인분과 "추정"/"확인 불가" 구분.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문**: Zhuguanyu Wu et al., *"AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer"*, ECCV 2024 (arXiv:2407.12951). (`README.md:1-3, 91-97`)
- **목적**: **PTQ(학습 없는 사후 양자화)** 로 ViT를 저비트(W3/W4/W6) 양자화. 재학습 없이 캘리브레이션만으로 정확도 회복.
- **핵심 아이디어 (코드로 확인)**:
  1. **Adaptive Logarithm Quantizer (AdaLog)**: log₂ 기반 비균등 양자화의 **밑(base)을 적응적으로 변경**. log 밑 `= 1/k`, `k = r/q` (r=37 고정, q 탐색) (`logarithm.py:68-102`, `matmul.py:286-290`).
  2. **Post-Softmax / Post-GELU 활성에 적용**: 멱법칙(power-law)·long-tail 분포에 log 양자화가 적합 (`README.md:3`).
  3. **Bias Reparameterization (shift)**: 음수/비대칭 분포를 shift로 옮겨 log 양자화 적용 가능하게 함 (`logarithm.py:105-136`).
  4. **FPCS (Fast Progressive Combining Search)**: scale/zero-point/log-base를 점진적 후보 좁히기로 탐색 (`matmul.py:243-262`, `linear.py:483-523`).
  5. **하드웨어 친화**: log 양자화는 dequant가 **시프트(2의 거듭제곱)** 로 환원 → AdaLog는 LUT(table1/table2)로 비정수 밑까지 시프트화 (`logarithm.py:77-99`).
  6. 선택적 **BRECQ 최적화(`--optimize`)** 로 AdaRound 미세조정 (`README.md:47`, `quantizers/adaround.py`).

본 repo는 **log2 기반 시프트 친화 양자화**의 레퍼런스로, 본 프로젝트의 Softmax/GELU 후단 활성 양자화를 시프트 회로로 구현하는 데 직접적 참고가 된다(8장).

---

## 2. 디렉토리 구조 (자체 + 제외)

### 자체 핵심 소스
```
AdaLog/
├── test_quant.py                  # PTQ 양자화·평가 엔트리포인트 (calibrate/optimize)
├── configs/
│   ├── 3bit.py / 4bit.py / 6bit.py  # 비트폭·탐색·최적화 설정 (Config 클래스)
├── quantizers/                    ★ 양자화기 핵심
│   ├── logarithm.py    ★         # Log2/LogSqrt2/AdaLog + Shift(bias-reparam) 변형
│   ├── uniform.py      ★         # Uniform/ShiftUniform/TwinUniform 양자화기
│   ├── adaround.py               # AdaRound(학습형 반올림) 양자화기
│   └── _ste.py                   # round_ste/floor_ste/ceil_ste
├── quant_layers/                 ★ 양자화 레이어 핵심
│   ├── linear.py       ★         # MinMax→PTQSL→Asymmetric→PostGeLU(Log/Twin) Linear 계층
│   ├── matmul.py       ★         # MinMax→PTQSL→Asymmetric→PostSoftmax(Log) MatMul 계층
│   └── conv.py                   # QuantConv2d (PatchEmbed)
├── utils/
│   ├── wrap_net.py     ★         # timm 모델을 양자화 모듈로 치환·결선
│   ├── calibrator.py   ★         # 캘리브레이션(hook으로 raw I/O 수집) + hyperparameter_searching 구동
│   ├── block_recon.py            # 블록 단위 BRECQ 재구성(선택)
│   ├── test_utils.py / datasets.py  # ImageNet 평가/로더
└── README.md
```

### 제외 (이름만)
- `.git/`, `__pycache__/`, `assets/framework.png`, `LICENSE` — 비소스.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 로그 양자화기 — `quantizers/logarithm.py`

#### (a) `Log2Quantizer` — `logarithm.py:8-38`
- `n_levels = 2^(n_bits-1)` (`:13`).
- forward (`:25-35`): `scaled_x = (x/scale).clamp(1e-15,1)` (`:29`), `x_quant = round(-log2(scaled_x))` (`:30`), `clamp(0, 2·n_levels-1)` (`:32`), **dequant = `2^(-x_quant) · scale`** (`:33`). → 양자화 인덱스가 곧 **2의 거듭제곱 지수** = 시프트량.
- mask로 범위 초과(언더플로우)는 0 처리 (`:31, 34`).

#### (b) `LogSqrt2Quantizer` — `logarithm.py:41-65`
- 밑 √2 사용: `x_quant=round(-log2(x)·2)` (`:51,56`), dequant은 홀짝 마스크로 `2^(-⌈q/2⌉)·(√2 보정)·scale` (`:59-60`). 더 촘촘한 log 그리드.

#### (c) **`AdaLogQuantizer` (핵심)** — `logarithm.py:68-102`
- `self.r = 37.0` 고정, `self.q` 버퍼(초기 37) (`:71-72`). **적응적 밑 = `2^(q/r)`** 형태.
- **LUT 사전계산** `update_table()` (`:77-81`):
  - `table1[i] = floor(i·q/r)` (정수 시프트량) (`:80`).
  - `table2[i] = round(2^(-((q·i) mod r)/r) · (4·n_levels-2)) / (4·n_levels-2)` (잔여 비정수 밑을 LUT 값으로 양자화) (`:79, 81`).
- forward 평가모드 (`:93-97`): `x_quant = round(-log2(scaled_x)·r/q)` (`:94`), **dequant = `2^(-table1[x_quant]) · table2[x_quant] · scale`** (`:97`).
  - → **정수 시프트(`2^(-table1)`) + 작은 LUT 곱(`table2`)** 으로 임의 log 밑을 하드웨어 친화적으로 근사. 비정수 밑을 "시프트 + LUT"로 분해한 것이 AdaLog의 핵심.
- 학습모드(`:88-92`)는 round_ste로 미분가능.

#### (d) Shift 변형 (bias reparameterization) — `logarithm.py:105-136`
- `ShiftLog2/ShiftLogSqrt2/ShiftAdaLogQuantizer`: `shift`(학습 파라미터)와 `bias_reparamed` 버퍼.
- forward: `result = base.forward(x + shift)`; reparam 안 됐으면 `result - shift` (`:111-113, 122-124, 133-135`). → **음수/비대칭 활성을 양수 영역으로 옮겨 log 양자화 적용**, shift는 이전 레이어 bias로 흡수 가능(reparam).

### 3.2 균등 양자화기 — `quantizers/uniform.py`
- `UniformQuantizer` (`:7-39`): 대칭/비대칭 모두. 비대칭은 `(round(x/scale)+round(zp)).clamp(0,2·n_levels-1)`, dequant `(x_quant-zp)·scale` (`:33-35`).
- `ShiftUniformQuantizer` (`:42-50`): shift + bias_reparam.
- `TwinUniformQuantizer` (`:53-68`): **양/음 두 스케일** (PTQ4ViT식). `scale[0]`(양), `scale[1]`(음)로 분리 양자화 후 합산 (`:62-67`) → post-GELU의 양·음 비대칭 분포 대응.

### 3.3 AdaRound 양자화기 — `quantizers/adaround.py`
- `AdaRoundQuantizer` (`:7-77`): 학습형 반올림(Up/Down). `alpha` 파라미터, `get_soft_targets()=clamp(sigmoid(alpha)(ζ-γ)+γ,0,1)` (`:59-60`), hard는 `floor(x/scale)+(alpha≥0)` (`:48, 71-73`). BRECQ/블록 재구성용.

### 3.4 양자화 Linear 계층 — `quant_layers/linear.py` (1007 lines, 핵심 부분)
계층 상속: `MinMaxQuantLinear` → `PTQSLQuantLinear` → `PTQSLBatchingQuantLinear` → `AsymmetricallyBatchingQuantLinear` → {`AsymmetricallyChannelWiseBatchingQuantLinear`, `PostGeluTwinUniformBatchingQuantLinear`, `PostGeluLogBasedBatchingQuantLinear`}.

- `MinMaxQuantLinear` (`linear.py:8-61`): `mode∈{raw, quant_forward, debug_only_*}` (`:26-37`). raw=FP, quant_forward=가중치/활성 모두 양자화 (`:46-51`).
- `PTQSLQuantLinear` (`:64-92`): per-channel weight 양자화 + 유사도 기반 탐색. `_get_similarity = -(raw-sim)^2` (`:87-88`). `n_V`로 weight 행 그룹화(`:82-91`).
- `PTQSLBatchingQuantLinear` (`:95-235`): GPU 메모리로 병렬 후보 수(`parallel_eq_n`) 자동 산정 (`:111-121`). `_search_best_w_scale`/`_search_best_a_scale`로 scale 후보 탐색(MSE 최대화) (`:141-208`). `hyperparameter_searching`에서 weight·activation scale 순차 탐색 (`:210-235`), 후보 범위 `eq_alpha=0.01, eq_beta=1.2` (`:216`).
- `AsymmetricallyBatchingQuantLinear` (`:238-621`): **비대칭(zero-point 포함)** weight·activation. percentile 후보 생성(`calculate_percentile_*_candidates`, `:432-481`), **FPCS** 탐색(`weight_fpcs`/`activation_fpcs`, `:483-523`).
  - `AsymmetricallyChannelWiseBatchingQuantLinear` (`:548-621`): **per-channel(채널별) 활성 양자화 + reparam**. `reparam_step1`(`:596-612`)에서 채널별 scale을 평균 scale로 통일하며 그 비율 `r`을 **이전 레이어(LayerNorm) weight/bias로 흡수**(`:604-605`), 현재 레이어 weight를 보정(`:606-611`) → LayerNorm-Linear fusion식 채널 재파라미터화.
- `PostGeluTwinUniformBatchingQuantLinear` (`:624-721`): GELU 출력의 음수 꼬리(`-0.17`)를 `TwinUniformQuantizer`로 처리, 음 스케일 초기값 `0.16997.../n_levels` (`:653-657`).
- **`PostGeluLogBasedBatchingQuantLinear`** (`:724-1007 일부`): post-GELU 활성에 `ShiftAdaLogQuantizer` 적용 (`:747`), shift 초기 `0.16997...`(`:749`). LUT `table` 사전계산(`:750-752`). 양수 percentile 후보(`positive_percentile`, `:763-798`)로 scale 탐색.
  - 주석: "log_2 base = 1/k, k=r/q, q=37 고정, r 탐색" (`:725-728`) — (matmul과 r/q 역할 표기가 반대이나 실질 동일 메커니즘. 확인: matmul은 r=37 고정·q 탐색 `:288`).

### 3.5 양자화 MatMul 계층 — `quant_layers/matmul.py`
계층: `MinMaxQuantMatMul` → `PTQSLQuantMatMul` → `PTQSLBatchingQuantMatMul` → `AsymmetricallyBatchingQuantMatMul` → `PostSoftmaxAsymmetricallyBatchingQuantMatMul`.

- `MinMaxQuantMatMul` (`:13-45`): Q@K, attn@V용. A/B 각각 UniformQuantizer (`:20-21`).
- `PTQSLQuantMatMul` (`:48-79`): **head-channel-wise** 양자화 옵션(`head_channel_wise`, `:71-76`).
- `AsymmetricallyBatchingQuantMatMul` (`:109-283`): 비대칭 A/B + FPCS(`_fpcs`, `:243-262`). percentile 후보 + zero-point 후보 격자 생성(`calculate_percentile_candidates`, `:211-240`).
- **`PostSoftmaxAsymmetricallyBatchingQuantMatMul`** (`:286-378`): attn@V의 A(=softmax 출력)에 **AdaLog/Log2/LogSqrt2** 적용 (`:307-317`). 
  - log 밑 탐색 `_search_best_A_log_base`(`:321-358`): q 후보 `range(10, 11+eq_n)`(`:323`)에 대해 `A_quant=round(-log2(A)·r/q)`(`:337`), LUT 인덱스 `col_index=(A_quant·q) mod r`(`:340`), `A_sim=2^(-floor(A_quant·q/r))·table[col_index]`(`:341`)로 유사도 최대 q 선정 → `q.update_table()` (`:355-357`). **이것이 "적응적 log 밑 탐색"의 실체**.

### 3.6 모델 결선 — `utils/wrap_net.py`
- timm `Attention`/`WindowAttention`에 `MatMul()` 주입 + forward 메서드 교체 (`:19-64`).
- `wrap_modules_in_net(model, cfg, reparam)` (`:55-172`): 모듈 순회하며 치환.
  - `nn.Conv2d`→`AsymmetricallyBatchingQuantConv2d` (`:78-96`).
  - `MatMul`: **`matmul2`(=attn@V)는 `PostSoftmax...`**(post-softmax log 양자화), 그 외는 `Asymmetric...` (`:97-121`).
  - `nn.Linear`: **`fc2`(=post-GELU)는 `PostGeluLogBased...` 또는 `PostGeluTwin...`** (`:154-163`); reparam 대상(`qkv/reduction/fc1`)은 `AsymmetricallyChannelWise...`로 prev_layer(norm)를 연결(`:139-153`); 그 외는 `Asymmetric...`(`:164-167`).
  - `head`는 `qhead_a_bit` 별도 적용 (`:123`).

### 3.7 캘리브레이션 — `utils/calibrator.py`
- `QuantCalibrator.batching_quant_calib()` (`:30-67`): 미캘리브 모듈마다 forward hook으로 **raw_input/raw_out 수집**(`:38-56`) → `module.hyperparameter_searching()` 실행(`:59`) → prev_layer 있으면 `reparam()`(`:60-62`). 끝나면 전 모듈 `mode='quant_forward'`(`:65-67`).

### 3.8 설정 — `configs/4bit.py`
- `Config`(`4bit.py:2-24`): `w_bit=a_bit=s_bit=4`, `qconv_a_bit=8`, `qhead_a_bit=4` (`:9-13`), `post_softmax_quantizer='adalog'`, `post_gelu_quantizer='adalog'` (`:15-16`), `eq_n=128, search_round=3, fpcs=True, steps=6` (`:18-20`), `train_act=True`(BRECQ시) (`:24`).

---

## 4. 알고리즘 / 수식

### 4.1 Log2 비균등 양자화
```
q = round( -log2( clamp(x/scale, 1e-15, 1) ) ),  q∈[0, 2·n_levels-1]
dequant = 2^(-q) · scale                      # 인덱스 q = 우측 시프트량
```
(`logarithm.py:30-33`). dequant이 순수 시프트 → 하드웨어 친화.

### 4.2 AdaLog 적응적 밑 + LUT
log 밑 `= 2^(q/r)` (r=37 고정, q는 캘리브레이션으로 탐색):
```
idx = round( -log2(x/scale) · r/q )
dequant = 2^( -⌊idx·q/r⌋ )  ·  table2[idx]  ·  scale
table1[i] = ⌊i·q/r⌋ (시프트량),  table2[i] = round(2^(-((q·i) mod r)/r)·(4n_lvl-2))/(4n_lvl-2)
```
(`logarithm.py:79-81, 94-97`). 정수 시프트 + 소형 LUT로 임의(비정수) log 밑 실현. q 탐색은 유사도(`-MSE`) 최대화(`matmul.py:321-358`).

### 4.3 Bias Reparameterization (shift)
`AdaLog(x) = base(x + shift) (− shift if not reparamed)` (`logarithm.py:133-135`). 음수 활성을 양수로 이동해 log 양자화 적용; shift는 이전 레이어 bias로 흡수.

### 4.4 FPCS (Fast Progressive Combining Search)
percentile로 초기 scale/zp 후보 생성 → topk(fpcs_width=16) 선택 → 선택 후보 주변을 더 촘촘히 재샘플(`delta_scale` 축소) 하며 steps 회 반복 → 최종 1개 선택 (`matmul.py:243-262`, `linear.py:483-523`).

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet (ILSVRC), `--dataset <DATA_DIR>` (`README.md:43`).
- **PTQ 명령** (`README.md:33-71`):
  ```bash
  # 캘리브레이션
  python test_quant.py --model deit_tiny --config ./configs/3bit.py --dataset <DIR> --val-batch-size 500 --calibrate
  # 캘리브레이션 + BRECQ 최적화
  python test_quant.py ... --calibrate --optimize
  # 캘리브된 체크포인트 로드 후 최적화
  python test_quant.py ... --load-calibrate-checkpoint <CKPT> --optimize
  ```
  - `--model`: deit_tiny/small/base, vit_small/base, swin_small/base (`README.md:39`).
  - `--calibrate` vs `--load-calibrate-checkpoint`: mutually exclusive (`README.md:45`).
- **결과(README.md:77-85)**: 예 — ViT-S W4A4 캘리브 72.75 / 재학습 77.25 / W6A6 80.91 (FP 81.39). 저비트(W3A3)는 재학습으로 큰 회복.

---

## 6. 의존성

- PyTorch (README 기준 1.10.0), **timm 0.9.2 권장** (`README.md:16-21`), numpy, tqdm.
- timm 사용: `timm.models.vision_transformer.Attention`, `timm.models.swin_transformer.WindowAttention` (`wrap_net.py:7-9`).
- CUDA 전제: `quant_layers`의 메모리 계산이 `torch.cuda.get_device_properties` 의존(`matmul.py:97-101`, `linear.py:113-117`), 다수 `.cuda()`. → GPU 필수(추정 근거: 코드상 명시).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **PTQ만으로 저비트 양자화** → I-ViT 대비 재학습 비용 대폭 절감.
- **log2/AdaLog로 Softmax·GELU 후단을 시프트+소형 LUT로** → 하드웨어 친화(특히 시프트 기반 dequant).
- FPCS·twin/channel-wise/reparam 등 정확도 회복 기법 다층 제공.

**한계 / 리스크**
- 핵심은 **fake-quant(시뮬레이션)** — 실제 정수 추론 커널/배포는 본 repo 범위 밖(I-ViT의 dyadic 정수 연산 같은 구현은 없음). 하드웨어 실연산 매핑은 별도 필요.
- AdaLog LUT(`table2`)는 비정수 밑을 근사하는 추가 곱이 필요 → 순수 시프트만은 아님(소형 LUT 곱 동반).
- GPU 메모리 기반 `parallel_eq_n` 산정으로 환경 의존(`matmul.py:95-106`).
- `r=37` 고정 등 하이퍼파라미터가 경험적(`logarithm.py:71`).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적; 프로젝트 성격 추정)

- **log2/AdaLog = 시프트 친화 양자화의 직접 참조**: post-Softmax(`matmul.py:286-378`)·post-GELU(`linear.py:724-`) 활성을 `2^(-q)` 시프트 + 소형 LUT(`table1/table2`, `logarithm.py:77-99`)로 dequant → FPGA에서 **곱셈기 대신 배럴 시프터 + 작은 LUT**로 후단 활성 처리 가능. HG-PIPE류 파이프라인의 Softmax/GELU 후단 데이터패스 경량화에 적합.
- **재학습 불요(PTQ)**: 시선추적 모델을 빠르게 양자화·배포해야 하는 상황에서 I-ViT(QAT) 대비 turnaround가 짧음 → 프로토타이핑·다모델 대응에 유리(추정).
- **Bias reparameterization(shift)**(`logarithm.py:105-136`) + **channel-wise reparam**(`linear.py:596-621`): 비대칭/채널별 분포를 이전 레이어로 흡수 → 하드웨어에서 추가 zero-point 보정 로직 없이 균일 파이프라인 유지에 도움.
- **한계 보완 필요**: 본 repo는 정수 실연산 커널이 없으므로, **AdaLog의 log 인덱스/LUT를 I-ViT의 dyadic 정수 연산과 결합**하는 것이 본 프로젝트 하드웨어화의 현실적 경로(추정). 즉 "AdaLog의 분포 적합 + I-ViT의 정수 실행"의 하이브리드.
- **XR 시선추적 적용(추정)**: 저비트(W4A4) PTQ로 메모리/대역폭을 줄이되, log 양자화의 long-tail 대응이 attention 분포가 뾰족한 시선추적류 태스크에서 정확도 유지에 기여할 수 있음.

---

## 9. 근거 표기 / 확인 불가 항목

- **직접 코드 확인**: §3~§4의 라인 인용(`logarithm.py`, `uniform.py`, `adaround.py`, `linear.py`(1-817행), `matmul.py` 전체, `wrap_net.py`, `calibrator.py`, `configs/4bit.py`), §2 구조(Glob), §5 명령/결과(`README.md`).
- **추정**: 프로젝트 성격, GPU 필수, PTQ turnaround 이점, 하이브리드 하드웨어화 경로.
- **확인 불가(부분 열람/미열람)**: `linear.py`의 818-1007행(`PostGeluLogBased...`의 `_search_best_a_scale` 등 일부) — `__init__`·percentile 후보 로직까지는 확인, 탐색 루프 후반부 라인은 미열람. `block_recon.py`(AdaLog판), `test_quant.py`, `conv.py`, `configs/3bit.py`/`6bit.py`, `utils/test_utils.py`/`datasets.py` — 미열람(APHQ-ViT가 AdaLog 기반이므로 구조는 APHQ.md의 block_recon과 유사할 것으로 추정).
