# P2-ViT 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/P2-ViT`
> 분석 일자: 2026-06-20
> 근거 표기 규칙: 코드로 확인한 사실은 `(파일:라인)`. 직접 확인 못한 항목은 "확인 불가", 추론은 "추정".

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문**: *P²-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer*, Huihong Shi, Xin Cheng, Wendong Mao, Zhongfeng Wang, **IEEE TVLSI 2024** (arXiv:2405.19915) (`README.md:1-3,48-53`).
- **기반 코드**: **FQ-ViT** (megvii-research) 포크 (`README.md:57`). 파일 헤더 다수 "Copyright (c) MEGVII Inc." (`bit_type.py:1` 등).
- **목적**: ViT를 **완전 양자화(fully quantized)** 하되, 기존 ViT 양자화가 유지하던 **부동소수점 스케일 팩터를 2의 거듭제곱(Power-of-Two, PoT)으로 대체**하여 재양자화(re-quantization) 오버헤드를 제거 → 하드웨어 효율 극대화 (`README.md:8-17`).
- **핵심 아이디어 (코드 기반 확인)**:
  1. **PoT 스케일 팩터**: scale = `2^α` 로 강제. requant 시 부동소수점 곱셈 대신 **시프트**로 처리 가능 (`minmax.py:175-177`, `ptf.py`).
  2. **PTQ (Post-Training Quantization)**: 학습 없이 calibration 데이터로 scale/zero-point 결정 (`test_quant.py:203-253`).
  3. **Fully Quantized**: weight/activation뿐 아니라 **LayerNorm(QIntLayerNorm), Softmax(Log-Int-Softmax)** 까지 정수 연산화 (`layers.py:225-413`). FQ-ViT의 PTF·LIS 계승.
  4. **Coarse-to-fine 자동 mixed-precision**: Hessian trace 민감도 + 양자화 거리(distance)로 레이어별 비트(4/8) 할당, Pareto/진화탐색 (`test_quant.py:152-385`).
- **백본**: DeiT/ViT (`vit_quant.py`), Swin (`swin_quant.py`). DeiT-tiny/small/base, ViT-base/large, Swin-tiny/small/base (`test_quant.py:61-73`).

---

## 2. 디렉토리 구조 (자체 소스 + 제외 표기)

```
P2-ViT/
├── README.md                       # 논문/사용법 (분석함)
├── test_quant.py                   # ★ PTQ calibration+평가+mixed-precision 탐색 엔트리 (분석함)
├── config.py                       # ★ 양자화 설정 Config (분석함)
├── test.py / generate_data.py      # 보조(PSAQ-ViT calib 데이터 생성)
├── models/
│   ├── vit_quant.py                # ★ 양자화 ViT(DeiT/ViT) 모델 (분석함)
│   ├── vit_fquant.py               # fully-quant ViT 변종
│   ├── swin_quant.py               # Swin 양자화 모델
│   ├── layers_quant.py             # ★ smoothquant/공통 레이어 유틸 (분석함)
│   ├── utils.py / plot_distrib.py  # npz 로드, 분포 시각화
│   └── ptq/                        # ★ 양자화 핵심 패키지
│       ├── layers.py               # ★ QConv2d/QLinear/QAct/QIntLayerNorm/QIntSoftmax (분석함)
│       ├── bit_type.py             # ★ BitType (uint/int 비트 정의) (분석함)
│       ├── quantizer/
│       │   ├── base.py             # ★ BaseQuantizer (분석함)
│       │   ├── uniform.py          # ★ UniformQuantizer (분석함)
│       │   ├── log2.py             # ★ Log2Quantizer (softmax용) (분석함)
│       │   └── build.py            # ★ 빌더 (분석함)
│       └── observer/
│           ├── base.py             # ★ BaseObserver (분석함)
│           ├── minmax.py           # ★ MinmaxObserver(+PoT round) (분석함)
│           ├── ema.py              # ★ EmaObserver (분석함)
│           ├── omse.py             # ★ OmseObserver(LAPQ Lp-norm) (분석함)
│           ├── percentile.py       # ★ PercentileObserver (분석함)
│           ├── ptf.py              # ★ PtfObserver(Power-of-Two Factor) (분석함)
│           ├── utils.py            # ★ lp_loss (분석함)
│           └── build.py            # ★ 빌더 (분석함)
├── pyhessian/                      # Hessian trace 계산(mixed-precision 민감도)
│   └── {hessian.py, utils.py}
├── utils/                          # build_model, data_utils, kde
└── figures/                        # ❌ 제외: PoT.png/Overview.png 이미지
```

### 제외 항목
- `figures/` : 논문 그림 이미지.
- `pyhessian/` : mixed-precision 민감도용 외부 유틸(ML 분석 핵심 아님, 호출부만 `test_quant.py:153`에서 확인).
- 사전학습 가중치: `torch.hub`/`npz` URL로 런타임 다운로드 (`vit_quant.py:448-558`) — 로컬 대용량 체크포인트 미발견, **이름만 언급**.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `ptq/bit_type.py` — 비트 타입 정의

- **`BitType(bits, signed, name)` (`bit_type.py:7-39`)**: `upper_bound`/`lower_bound`/`range` 프로퍼티 제공. signed면 `[-2^(b-1), 2^(b-1)-1]`, unsigned면 `[0, 2^b-1]` (`bit_type.py:17-31`).
- **`BIT_TYPE_LIST` / `BIT_TYPE_DICT` (`bit_type.py:42-57`)**: 활성 비트 후보 = `uint3, uint4, int4, int8, uint8`. (int5/6/7/10 등은 주석 처리 → 현재 4/8 위주.)

### 3.2 `ptq/observer/*` — 통계 수집 및 scale/zero-point 산출

모든 observer는 `BaseObserver` 상속. `update(v)`로 calibration 중 min/max 누적, `get_quantization_params()`로 (scale, zero_point) 반환.

- **`BaseObserver` (`observer/base.py:5-37`)**: `reshape_tensor`로 weight는 `[out, -1]`, activation은 `[..., C]→[C, -1]` 재배열 (`base.py:16-29`). `eps=finfo.eps`.

- **`MinmaxObserver` (`observer/minmax.py:9-198`)** — **P2-ViT의 PoT 핵심 로직 포함**:
  - `update` (`minmax.py:16-33`): cur_max/cur_min 누적, layer_wise면 전체 max/min으로 축약.
  - `get_quantization_params` (`minmax.py:36-198`):
    - **`round_ln(x, type)` (`minmax.py:50-58`)**: `log2(x)`의 ceil/floor/round. round 모드는 `floor(log2 x)`와 다음 칸 중 가까운 쪽 선택 (`minmax.py:56-58`) → **scale을 2의 거듭제곱 지수로 변환**.
    - **`round_x(scale, x, zero_point)` (`minmax.py:122-168`)**: 후보 지수 `α_floor-1, α_floor, α_floor+1, α_floor+2` 4개로 각각 양자화(`weight/2^α` round-clamp-dequant)한 뒤, 실제 conv/linear/attn 출력과의 `lp_loss(p=2)`를 비교해 **최소 오차 지수 α 선택** (`minmax.py:142-167`). → **출력 인지(output-aware) PoT scale 최적화**.
    - `get_out` (`minmax.py:75-118`): module_type별 실제 연산(`F.conv2d`/`F.linear`/attention `get_attn`)을 수행해 양자화 전/후 출력 비교 (`minmax.py:90-118`). attention의 경우 q@k softmax @v 직접 계산 (`minmax.py:62-73`).
    - symmetric(가중치): `scale = max_val/((qmax-qmin)/2)` → `α_x=round_x(scale)` → **`scale = 2^α_x`** (`minmax.py:170-187`).
    - asymmetric(활성): `scale=(max-min)/(qmax-qmin)`, `zero_point=qmin-round(min/scale)`, 이후 PoT화 (`minmax.py:188-197`).

- **`PtfObserver` (`observer/ptf.py:8-135`)** — **Power-of-Two Factor (FQ-ViT의 LayerNorm용)**:
  - symmetric 기준 base scale `scale8 = 2·max_val_t/(qmax-qmin)` (`ptf.py:49-51`).
  - `scale4=scale8/2, scale2/2, scale1/2` 의 PoT 사다리 생성 (`ptf.py:97-99`).
  - 각 채널 j에 대해 `scale1·2^k (k∈{0,1,2,3})` 중 `lp_loss` 최소인 factor 선택 → **채널별 PoT factor `scale_mask`** (`ptf.py:109-133`). 최종 `scale = scale1 * scale_mask` (`ptf.py:133`).
  - → LayerNorm 입력처럼 채널간 분산이 큰 경우, **채널별 2^k 시프트 factor**로 표현해 정수 LN 가능케 함.

- **`OmseObserver` (`observer/omse.py:8-56`)**: max/min을 0~89% 축소하며 LAPQ식 **Lp-norm(L2) 최소화**로 clipping range 탐색 (`omse.py:38-50`, LAPQ arXiv:1911.07190 주석). PoT화는 안 함(순수 uniform clip 최적).
- **`EmaObserver` (`observer/ema.py:7-58`)**: min/max를 지수이동평균(`ema_sigma=0.01`)으로 누적 (`ema.py:25-32`).
- **`PercentileObserver` (`observer/percentile.py:9-71`)**: 0.99999 분위수로 outlier 제거 (`percentile.py:28-49`), layer_wise 전용 (`percentile.py:25`).
- **`lp_loss(pred,tgt,p,reduction)` (`observer/utils.py:2-9`)**: `|pred-tgt|^p`의 평균/합 — 모든 PoT/clip 탐색의 목적함수.

### 3.3 `ptq/quantizer/*` — 양자화/역양자화 실행

- **`BaseQuantizer` (`quantizer/base.py:6-45`)**: `forward = dequantize(quant(x))` (fake-quant) (`base.py:42-45`). `get_reshape_range`로 scale broadcast shape 결정 (`base.py:14-31`).
- **`UniformQuantizer` (`quantizer/uniform.py:8-127`)**:
  - `update_quantization_params` (`uniform.py:25-47`): observer로부터 (scale,zp) 받아 activation은 `self.scale`, weight는 비트별 dict(`dic_scale[bit_name]`)에 저장 → **per-bit 스케일 보관(mixed-precision 지원)** (`uniform.py:45-47`).
  - **`quant` (`uniform.py:50-88`)**: `outputs = round(x/scale + zp).clamp(lower, upper)` (`uniform.py:85-87`) — **표준 affine 양자화**. scale/zp을 `get_reshape_range`로 broadcast (`uniform.py:82-84`).
  - **`dequantize` (`uniform.py:90-127`)**: `(x - zp) * scale` (`uniform.py:126`).
  - 주의: scale 자체가 observer 단계에서 이미 `2^α`(PoT)로 산출되므로, uniform quantizer가 PoT-aware affine이 됨.
- **`Log2Quantizer` (`quantizer/log2.py:7-26`)** — **Softmax(Attention 확률) 전용 로그 양자화**:
  - `quant` (`log2.py:17-21`): `round(-log2(x))` 후 `[0, 2^bits-1]` 클램프 (`log2.py:18-20`). `softmax_mask`로 너무 작은 확률(=0 처리) 마스킹 (`log2.py:19`).
  - `dequantize` (`log2.py:23-26`): `2^(-x)`, 마스크 위치는 0 (`log2.py:24-26`). → softmax 출력을 2의 거듭제곱 역수로 표현해 **확률·V 곱을 시프트로** 대체.

### 3.4 `ptq/layers.py` — 양자화 레이어 (모델 빌딩 블록)

- **`QConv2d` (`layers.py:12-100`)**: patch embed conv 양자화. forward에서 `calibrate` 시 비트 후보별 observer.update + (last_calibrate면) `update_quantization_params` (`layers.py:54-70`). int8은 layer_wise, 그 외는 channel_wise calibration (`layers.py:63-67`). `quant` 시 `weight=quantizer(weight)` 후 `F.conv2d` (`layers.py:85-86`).
- **`QLinear` (`layers.py:103-177`)**: 핵심 선형 양자화. `calibrate` 루프에서 비트별로 weight 양자화 거리 `lp_loss(weight, weight_q)`를 `global_distance`에 누적 (`layers.py:147-169`) → **mixed-precision용 레이어 민감도**. `bit_config` 인자로 추론 시 비트 동적 지정 (`layers.py:172-176`). `weight_smoothed`로 SmoothQuant 가중치 입력 가능 (`layers.py:141-142`).
- **`QAct` (`layers.py:180-222`)**: 활성 양자화. `asymmetric` 옵션 시 uint8/비대칭 (`layers.py:207-211`). attention 입력엔 `attn`/`attn_para` 전달 (`layers.py:215`).
- **`QIntLayerNorm` (`layers.py:225-291`)** — **정수 LayerNorm (INT_NORM)**:
  - `get_MN(x)` (`layers.py:233-237`): 스케일을 `M·2^-N` 형태(M:정수 가수, N:시프트량)로 분해 (`layers.py:235-236`). bit=7.
  - `mode=='int'` forward (`layers.py:254-288`): 입력을 `x_q = round(x/in_scale)`로 정수화, 채널 스케일 마스크 `in_scale/in_scale_min` round (`layers.py:268-272`), 정수 평균/표준편차 계산 (`layers.py:274-277`), 어파인 계수 A를 `A_sign·M·x_q`로 정수 연산 후 `>>N`(`/2^N`) (`layers.py:281-287`). → **부동소수점 나눗셈/제곱근 없이 정수+시프트로 LN 근사**.
- **`QIntSoftmax` (`layers.py:294-413`)** — **Log-Int-Softmax (LIS)**:
  - `int_softmax` (`layers.py:330-364`): 2차 다항식 근사 `int_polynomial`(`layers.py:333-343`)와 정수 지수 `int_exp`(x0=-ln2, `layers.py:345-357`)로 **정수 softmax**. exp_int와 exp_int_sum 반환.
  - `log_round` (`layers.py:322-328`): softmax 결과를 `floor(log2)` + 보정으로 로그 양자화.
  - forward (`layers.py:366-394`): `log_i_softmax` 활성 시 `2^(-qlog)` 역양자화 (`layers.py:373-375`). → **softmax 확률을 PoT로 표현, attn@V를 시프트로**.

### 3.5 `config.py` — 양자화 설정

- **`Config(ptf, lis, quant_method)` (`config.py:4-52`)**:
  - 가중치 `BIT_TYPE_W = int4` (기본; int8 주석) (`config.py:13`), 활성 `BIT_TYPE_A = int8` (`config.py:17`).
  - `OBSERVER_W='minmax'`, `OBSERVER_A=quant_method`(minmax/ema/omse/percentile) (`config.py:19-20`).
  - `QUANTIZER_W/A='uniform'` (`config.py:22-23`).
  - `CALIBRATION_MODE_W='channel_wise'`, `CALIBRATION_MODE_A='layer_wise'` (`config.py:27-29`).
  - **`lis=True`** → `INT_SOFTMAX=True`, `BIT_TYPE_S=uint4`, `QUANTIZER_S='log2'` (`config.py:32-38`).
  - **`ptf=True`** → `INT_NORM=True`, `OBSERVER_A_LN='ptf'`, LN은 channel_wise (`config.py:44-47`).

### 3.6 `models/vit_quant.py` — 양자화 ViT/DeiT 모델

- **`Attention` (`vit_quant.py:25-117`)**: 각 연산 경계마다 양자화 노드 삽입.
  - `qkv = QLinear`, 출력 `qact1`(QAct) (`vit_quant.py:43-57`). `q@k^T·scale` 후 `qact_attn1` (`vit_quant.py:106-107`).
  - `log_int_softmax = QIntSoftmax(log_i_softmax=cfg.INT_SOFTMAX)` (`vit_quant.py:86-93`). 단 현재 forward는 `attn.softmax(dim=-1)` 직접 호출, LIS 경로는 TODO 주석 (`vit_quant.py:108-110`).
  - `attn@v` 후 `qact2`→`proj(QLinear)`→`qact3` (`vit_quant.py:112-116`).
- **`Block` (`vit_quant.py:120-196`)**: `norm1→qact1→attn`, `norm2→qact3→mlp`, residual 후 qact2/qact4(LN observer=ptf) (`vit_quant.py:193-195`). 정수 LN 연계용 `last_quantizer` 인자 보유(현재 정수 경로 TODO, `vit_quant.py:182-192`).
- **`VisionTransformer` (`vit_quant.py:199-425`)**: patch_embed/cls_token/pos_embed 후 각 단계 QAct. `model_quant()`/`model_open_calibrate()`/`model_open_last_calibrate()`/`model_close_calibrate()`로 **calibrate↔quant 모드 전환** (`vit_quant.py:364-390`). `INT_NORM`이면 QIntLayerNorm.mode='int' (`vit_quant.py:368-370`).
- **모델 팩토리** (`vit_quant.py:428-559`): deit_tiny/small/base(embed 192/384/768), vit_base/large. pretrained는 fbaipublicfiles(DeiT)·googleapis(ViT npz)에서 로드 (`vit_quant.py:448-558`).

### 3.7 `test_quant.py` — PTQ 파이프라인 + mixed-precision 탐색

- **calibration** (`test_quant.py:203-253`): mode 0(실데이터)/1(가우시안)/2(PSAQ-ViT 생성데이터) 중 선택 (`test_quant.py:207-248`). `model_open_calibrate()` → forward로 통계 수집 → `model_open_last_calibrate()` → `model_quant()` (`test_quant.py:237-253`).
- **mixed-precision** (`--mixed`, `test_quant.py:152-385`):
  - `pyhessian.hessian`으로 레이어별 Hessian trace 민감도 계산(주석 처리, 사전계산 `mean_hessian` 사용, `test_quant.py:153-200`).
  - **omega = Hessian × 양자화거리** 로 비트구성 점수화, 모델 크기 제약하 Pareto 후보 50개 생성 (`test_quant.py:264-304`).
  - omega 정렬 후 상위 검증 → **진화탐색(mutate/crossover, pop=25, iter=8)** 으로 최적 비트구성 탐색 (`test_quant.py:328-385`).
  - 비-mixed면 전 레이어 4비트 `bit_config=[4]*50` (`test_quant.py:389`).
- **평가** `validate` (`test_quant.py:395-443`): ImageNet val top-1/5.

---

## 4. 알고리즘 / 수식 (PoT / log2 양자화)

### 4.1 표준 affine 양자화 (Uniform)
```
q  = clamp( round(x/s + z), q_min, q_max )      # (uniform.py:85-87)
x̂ = (q - z) · s                                 # (uniform.py:126)
```
- weight symmetric: `s = max_val / ((q_max-q_min)/2)`, `z=0` (`minmax.py:173`).
- activation asymmetric: `s = (max-min)/(q_max-q_min)`, `z = q_min - round(min/s)` (`minmax.py:190-191`).

### 4.2 Power-of-Two (PoT) 스케일 — P2-ViT 핵심
부동소수점 scale `s`를 2의 거듭제곱으로:
```
α = round_ln(s)              # round(log2 s), floor와 다음 칸 중 근접 선택 (minmax.py:56-58)
s_PoT = 2^α                  # (minmax.py:177, 195)
```
**출력 인지 최적화** (`minmax.py:142-167`): α 후보 {α_f-1, α_f, α_f+1, α_f+2}로 각각 양자화한 출력과 원 출력의 L2 오차(`lp_loss`) 최소인 α 선택.

**HW 의미**: requantization `y = (acc · s_in · s_w) / s_out` 에서 모든 s가 `2^α`이면 → `y = acc << (α_in+α_w-α_out)` (시프트). **곱셈기/나눗셈기 불필요**.

### 4.3 Power-of-Two Factor (PTF, LayerNorm 활성)
채널 c별로 base scale `s1`에 `2^k` (k∈{0,1,2,3}) factor:
```
s_c = s1 · 2^(k_c),   k_c = argmin_k  lp_loss(x_c, dequant(quant(x_c; s1·2^k)))   # (ptf.py:109-133)
```
채널간 분산이 큰 LN 입력을 단일 정수 데이터패스로 표현.

### 4.4 Log2 양자화 (Softmax 확률)
```
q  = clamp( round(-log2 x), 0, 2^b-1 )          # (log2.py:18-20)
x̂ = 2^(-q)   (너무 작으면 0)                     # (log2.py:24-26)
```
attention `P·V` 에서 P가 `2^(-q)`이므로 `P·V = V >> q` (시프트).

### 4.5 정수 LayerNorm (QIntLayerNorm, mode='int')
스케일 A를 `M·2^-N` 분해 (`get_MN`, `layers.py:233-237`):
```
x_q = ( A_sign·M·x_q + B ) >> N                 # (layers.py:287)
```
부동 나눗셈·sqrt 없이 정수 가산·시프트로 LN.

### 4.6 정수 Softmax (LIS)
`exp(x) ≈ 2차 다항식·2^n` 정수 근사 (`int_exp`/`int_polynomial`, `layers.py:333-357`), x0=-ln2 분해로 정수 지수 계산.

---

## 5. 학습 / 평가 파이프라인

- **양자화 방식**: **PTQ (학습 불필요)**. calibration 데이터로 scale/zp 결정 후 평가.
- **데이터셋**: ImageNet (train/val ImageFolder, `test_quant.py:124-149`). calib batch=100, calib_iter=10, val batch=200 (`test_quant.py:35-48`).
- **calibration 데이터 mode** (`--mode`): 0=실데이터(기본), 1=가우시안, 2=PSAQ-ViT 생성(`generate_data`) (`test_quant.py:39-41,207-248`).
- **실행 명령어** (`README.md:31-39`):
  ```bash
  python test_quant.py deit_small <DATA_DIR> --quant --quant-method minmax
  ```
  - 모델: deit_tiny/small/base, vit_base/large, swin_tiny/small/base (`README.md:35`, `test_quant.py:20-25`).
  - `--quant-method`: minmax/ema/omse/percentile (활성 observer, `test_quant.py:30-32`).
  - `--mixed`: Hessian 기반 자동 mixed-precision 탐색 (`test_quant.py:33`).
  - `--ptf`/`--lis` 기본 True (PTF LN, Log-Int-Softmax) (`test_quant.py:28-29`).
- **전처리**: 모델별 mean/std/crop_pct 차등 (deit 0.875, vit/swin 0.9, `test_quant.py:104-117`), bicubic resize+centercrop (`test_quant.py:481-511`).
- **재현성**: seed 고정 + cudnn.deterministic (`test_quant.py:76-91`).

---

## 6. 의존성

- **PyTorch**, **torchvision** (datasets/transforms, `test_quant.py:9-10`).
- **timm** 계열 ViT 구조 (DropPath/PatchEmbed 등은 `layers_quant.py`에 자체 포함).
- **numpy**, **PIL** (`test_quant.py:11,16`).
- **pyhessian** (자체 포함, mixed-precision 민감도, `test_quant.py:153`).
- 사전학습 가중치: torch.hub(DeiT .pth), googleapis(ViT augreg .npz) 런타임 다운로드 (`vit_quant.py:448-558`).
- 원천: **FQ-ViT** 기반 (`README.md:57`).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **완전 정수 데이터패스**: weight/act뿐 아니라 LayerNorm(정수)·Softmax(LIS)까지 정수+시프트화 (`layers.py:225-413`) → ViT 가속기에서 부동소수점 유닛 제거 가능.
- **PoT 스케일로 requant=시프트**: requantization 곱셈기 제거 — **HW 효율의 핵심 기여** (`minmax.py:177`, `ptf.py`).
- **출력 인지 PoT 최적화**(`minmax.py:142-167`)로 단순 `round(log2 s)` 대비 정확도 보존.
- mixed-precision(Hessian×distance + 진화탐색)으로 정확도-효율 trade-off 자동화 (`test_quant.py:264-385`).
- PTQ라 재학습 비용 없음.

**한계 / 리스크**
- 코드 곳곳 TODO/주석 처리: `vit_quant.py`의 LIS forward 미연결(softmax 직접 호출, `vit_quant.py:108-110`), 정수 LN 경로 일부 주석(`vit_quant.py:182-192`) → **일부 정수 경로가 실평가에서 비활성일 수 있음**(확인 필요).
- mixed-precision `bit_config=[4]*50`, `mean_hessian` 길이 49 등 **모델 의존 하드코딩**(deit-base용) — 다른 모델엔 수동 조정 필요 (`test_quant.py:198-200,389`).
- 일부 텐서 `.cuda()` 하드코딩(`uniform.py:85-86`, `minmax.py:52`) → CPU/디바이스 유연성 낮음.
- 가속기 RTL/HLS 코드는 본 repo에 **미포함**(논문의 chunk-based accelerator는 별도) — **확인 불가**.

---

## 8. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적, 추정)

> 우리 프로젝트는 "Transformer/ViT FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 **추정**. P2-ViT는 ShiftAddViT보다 **FPGA 정수 데이터패스 정합성이 더 직접적**.

1. **PoT 스케일 → requantization 시프트화**가 가장 큰 시사점. HG-PIPE 류 파이프라인에서 레이어 간 requant는 보통 부동소수점 곱셈(scale 조정)인데, P2-ViT처럼 모든 scale을 `2^α`로 두면 **requant 블록을 배럴 시프터로 대체** → DSP/LUT 절감, 파이프라인 단순화(추정). 근거 `minmax.py:177`, `layers.py:235-237,287`.
2. **정수 LayerNorm(QIntLayerNorm)·Log-Int-Softmax**는 ViT 특유의 비선형 블록(LN/Softmax/GELU)을 FPGA에서 LUT+시프트로 구현하는 레퍼런스 — 시선추적 ViT의 LN/Softmax 데이터패스 설계에 직접 이식 가능 (`layers.py:225-413`).
3. **채널별 PTF**(`ptf.py:109-133`)는 LN 입력의 채널 분산 문제 해결책 — 시선추적 입력(저조도·고대비 눈 영상) 활성 분포가 채널별로 치우칠 때 유용(추정).
4. **W4A8 + mixed-precision**(`config.py:13,17`, `test_quant.py`)은 시선추적용 경량 ViT의 정확도-자원 균형점 탐색에 활용. 단 Hessian/bit_config 하드코딩은 우리 모델에 맞춰 재산출 필요.
5. **PTQ 특성**: 재학습 없이 calibration만으로 양자화 → 시선추적 도메인 데이터로 빠른 calibration 후 FPGA 배포 워크플로우에 적합.
6. **ShiftAddViT와 대비**: ShiftAddViT는 **가중치 자체를 PoT(곱셈→시프트)**, P2-ViT는 **스케일 팩터를 PoT(requant→시프트) + 전 블록 정수화**. 두 접근은 **상보적** — 우리 가속기에 "PoT 가중치 PE(ShiftAddViT) + PoT-scale requant(P2-ViT) + 정수 LN/Softmax(P2-ViT)"를 결합하면 곱셈기 최소화 ViT 데이터패스 구성 가능(추정).
7. **차이/주의**: 본 repo의 가속기 HW는 미포함(확인 불가). 알고리즘/양자화 레퍼런스로서 가치가 크며, 일부 정수 경로가 forward에서 TODO로 비활성일 수 있어 RTL 매핑 전 실제 활성 경로 검증 필요.

---

## 9. 근거/불확실성 정리
- **코드 확인**: PoT scale(2^α), 출력 인지 α 탐색, PTF, Log2 softmax, 정수 LN/Softmax, uniform affine, mixed-precision 탐색 — 모두 (파일:라인)으로 확인.
- **추정**: 프로젝트(HG-PIPE/XR) 연관 시사점, FPGA requant 시프트화 효과, 두 repo 결합 전략 — "추정" 명시.
- **확인 불가**: 논문의 chunk-based accelerator RTL(미포함), 실제 정확도/속도 수치(논문 본문 미열람), 일부 TODO 경로의 실험 시 실제 활성 여부.
