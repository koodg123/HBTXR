# Quantformer 코드베이스 정밀 분석

> 대상: `REF/ViT-Quantization/Quantformer`
> 원논문: *Quantformer: Learning Extremely Low-precision Vision Transformers*
> 분석 방식: 실제 소스 코드 직접 판독(Glob/Grep/Read). 모든 핵심 주장에 `파일:라인` 근거 표기.
> 근거 규칙: "코드 확인" = 해당 라인에서 직접 확인됨 / "추정" = 코드 정황상 합리적 추론 / "확인 불가" = 코드만으로 단정 불가.

---

## 1. 개요

Quantformer는 **극저비트(주로 4-bit) Vision Transformer**를 학습(QAT, Quantization-Aware Training)으로 얻기 위한 PyTorch 구현이다. 핵심 아이디어는 README에 명시되어 있다(`README.md:3`):

> "differentiable searching and finetuning process for **patch group assignment** in **group-wise quantization**"

즉, 활성값(activation)을 단일 스케일로 양자화하지 않고 **그룹 단위(group-wise) 양자화**를 적용하되, 어떤 채널/토큰이 어느 그룹에 속할지(group assignment)를 **미분 가능(differentiable) 탐색**으로 학습하고, 이후 finetuning으로 정확도를 회복한다.

파이프라인은 3단계다(`README.md:26-47`):
1. **Pretraining**: shared quantization(`--group-num 1`)으로 저비트 양자화 모델을 먼저 만든다.
2. **Searching**(`--search True`): group 수를 늘려(`--group-num 8` 등) 미분 탐색으로 최적 group assignment를 학습한다.
3. **Finetuning**(`--search False`): 탐색된 assignment를 고정/이용해 모델을 미세조정한다.

기반 백본: **DeiT**(vision_transformer) 및 **Swin Transformer**(`README.md:20,53-54`). 학습/평가 데이터셋은 ImageNet ILSVRC2012(`README.md:16`).

**중요 사전 경고 — 어떤 코드가 "실제로 동작하는" 코드인가:**
이 저장소에는 동일 이름의 quant 유틸이 **두 군데** 존재한다.
- 최상위 `lib/utils/quantize_utils.py` (그룹 양자화 로직 `sw`/`alpha_activ` 포함) — **실제 활성 경로**
- `custom_timm/models/layers/lib/utils/quantize_utils.py` (HAQ 원본, 그룹 로직 없음) — **벤더링된 잔재**

`main.py:27`, `engine.py:20`, `swin_transformer.py:29`, `vision_transformer.py:35`, `qmlp.py:6`가 모두 **최상위 `lib.utils.quantize_utils`**를 import한다(코드 확인). 따라서 본 분석은 최상위 `lib/`를 기준으로 한다. `custom_timm/.../layers/lib/`는 사용되지 않는 사본으로 판단(추정).

---

## 2. 디렉토리 구조

### 2.1 자체 quant 추가분 (분석 핵심)

```
Quantformer/
├── main.py                         # 학습/평가 엔트리. quant 설정·calibrate·이중 optimizer
├── engine.py                       # train_one_epoch / evaluate. aux_loss·dm_loss(탐색 손실)
├── models.py                       # DeiT/fpDeiT 모델 등록 (register_model)
├── losses.py                       # DistillationLoss (DeiT 원본)
├── deit_run.sh / swin_run.sh       # 실행 예시 스크립트 (실제 하이퍼파라미터)
├── lib/                            # ★ 실제 활성 quant 라이브러리
│   ├── utils/
│   │   ├── quantize_utils.py            # ★ QModule/QLinear/QConv2d + 그룹 양자화(sw, alpha_activ, QuantGroup, Quant, two_groups)
│   │   ├── quantize_utils_activation.py # ★ QActvation/QModule1 (활성 전용 PACT형 양자화, half_wave='Q')
│   │   ├── quantize_utils_train.py      # (변형 사본, grep만 확인)
│   │   ├── quantize_utils_search.py     # (변형 사본, grep만 확인)
│   │   ├── binary_utils.py              # BinarizeConv2d (resnet용)
│   │   └── utils.py, data_utils.py …
│   ├── rl/  ddpg.py, memory.py     # HAQ RL (DDPG) — 본 파이프라인에서 미사용(아래 §3.6)
│   └── env/ quantize_env.py, linear_quantize_env.py  # HAQ mixed-precision env — 미사용
└── custom_timm/
    └── models/
        ├── vision_transformer.py   # ★ Attention/QAttention/QBlock/QPatchEmbed (DeiT 경로)
        ├── swin_transformer.py     # ★ WindowAttention/QWindowAttention (Swin 경로, 탐색이 실제로 도는 경로)
        └── layers/
            ├── qmlp.py             # ★ QMlp (QLinear 2개로 구성된 양자화 MLP)
            └── lib/ …              # custom_timm 내부의 lib 사본 (미사용 추정)
```

### 2.2 벤더링 timm (제외, 이름만)

`custom_timm/data/` 전체(loader, mixup, auto_augment, parsers …), 그리고 `custom_timm/models/` 내 **비-Transformer CNN**: `resnet.py`*, `efficientnet*.py`, `densenet.py`, `vgg.py`, `regnet.py`, `nfnet.py`, `resnest.py`, `dpn.py`, `inception_*.py`, `nasnet.py`, `pnasnet.py`, `mobilenetv3.py`, `ghostnet.py`, `hrnet.py`, `cspnet.py`, `byobnet.py`, `byoanet.py`, `dla.py`, `senet.py`, `sknet.py`, `vovnet.py`, `selecsls.py`, `tresnet.py`, `xception*.py`, `gluon_*.py`, `res2net.py`, `cait.py`, `pit.py`, `mlp_mixer.py`, `vision_transformer_hybrid.py`, `hardcorenas.py` 등. `custom_timm/models/layers/*`의 표준 timm 레이어(activations, padding, drop, weight_init 등)도 제외.
(*`resnet.py`는 `QConv2d`/`BinarizeConv2d`를 쓰지만 본 ViT 파이프라인의 핵심이 아니므로 본문에서 다루지 않음.)

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 QMlp — 양자화 MLP (`custom_timm/models/layers/qmlp.py`)

표준 ViT MLP의 두 Linear를 `QLinear`로 교체한 것이 전부다(`qmlp.py:8-26`).

```python
self.fc1 = QLinear(in_features, hidden_features, half_wave='Q', **kwargs)   # qmlp.py:15
self.act = act_layer()                                                       # GELU
self.fc2 = QLinear(hidden_features, out_features, half_wave='A', **kwargs)  # qmlp.py:17
```

- `fc1`은 `half_wave='Q'`(대칭/양방향 클립, 일반 분포용), `fc2`는 `half_wave='A'`(GELU 출력 → 음수 일부만 있는 분포용)로 **활성 분포 특성에 맞춰 양자화 모드를 다르게** 준다(코드 확인). `**kwargs`로 `args`(=group-num 등)를 전달받아 그룹 양자화가 작동한다(추정: QLinear가 `kwargs["args"]`를 요구, `quantize_utils.py:66`).

### 3.2 QAttention / QWindowAttention — 양자화 어텐션

#### (a) DeiT 경로 — `vision_transformer.py:166-200`
```python
bitwidth = self.args.bitwidth                                                # :170
self.qkv  = QLinear(dim, dim*3, ..., w_bit=bitwidth, a_bit=bitwidth, half_wave='Q')  # :175
self.qkv_actvation = QActvation(dim*3, a_bit=8, half_wave='Q')               # :177
self.attn_actvation = QActvation(dim,  a_bit=8, half_wave='Q')               # :178
self.proj = QLinear(dim, dim, w_bit=bitwidth, a_bit=bitwidth, half_wave='Q') # :181
```
forward(`:185-200`): `qkv → qkv_actvation`로 qkv 출력 활성을 8-bit 양자화한 뒤 head 분해, `attn = softmax(q@kᵀ·scale)` 후 `attn_actvation`으로 attention map도 양자화하고, `proj`로 출력. 즉 **(1) qkv/proj 가중치+활성 = bitwidth-bit, (2) qkv 출력·softmax 출력 = 8-bit** 두 종류 양자화가 공존한다(코드 확인).

#### (b) Swin 경로 — `swin_transformer.py:204-280` (탐색이 실제로 도는 경로)
구조는 DeiT와 동일하나 활성 양자화 비트가 `bitwidth`로 통일된다:
```python
self.qkv_actvation = QActvation(dim*3, a_bit=bitwidth, half_wave='Q')   # :236
self.attn_actvation = QActvation(dim,  a_bit=bitwidth, half_wave='Q')   # :237
```
+ Swin 고유의 relative position bias / window mask 유지(`:261-272`).

> **⚠ 코드 정합성 경고 1 (engine과 forward 불일치):**
> `engine.py:61,65`는 탐색 시 `layer.attent`(어텐션 맵)를 읽어 aux_loss를 계산한다. 그러나 `QWindowAttention.forward`(`swin:246-280`)와 `WindowAttention.forward`(`swin:171-202`) 어디에도 **`self.attent = attn` 대입이 존재하지 않는다**(grep `attent` 결과: 정의 없이 engine에서 읽기만 함). REF에 담긴 이 사본은 aux_loss 경로가 그대로는 실행되지 않는(버그/미완) 상태로 보인다 — **확인 불가**(원 저장소에는 대입이 있었으나 이 사본에서 누락되었을 가능성, 추정).

> **⚠ 코드 정합성 경고 2 (`bitwidth` vs `bit_width`):**
> argparse는 `--bit-width` → `args.bit_width`로 받는다(`main.py:178`). 그런데 어텐션 생성자는 `args.bitwidth`(언더스코어 없음)를 읽는다(`vit:170`, `swin:209`). 두 속성명이 **다르다**. 실제 비트폭은 생성 후 `main.py:319-324`에서 `layer.w_bit/layer.a_bit = quantize_layer_bit_dict[i]`로 **재설정**되므로 동작은 가능하나, 생성 시점의 `args.bitwidth` 의존부는 추가 주입 코드가 있어야 정상(확인 불가). 본 사본만으로는 `args.bitwidth` 설정 경로가 보이지 않음.

### 3.3 group-wise quantizer — `lib/utils/quantize_utils.py` (양자화의 심장)

#### QModule (`quantize_utils.py:62-110`) — 그룹 파라미터 정의
```python
self.group_n = self.args.group_num                                  # :67  그룹 수
self.groups_range1 = nn.Parameter(torch.zeros(self.group_n))        # :79  그룹별 클립 범위(학습)
self.alpha_activ  = nn.Parameter(torch.Tensor(group_n, dim))        # :88  그룹 배정 로짓(학습)
self.alpha_activ.data.fill_(0.01)                                   # :89
self.sw = torch.Tensor(group_n, dim)                                # :90  softmax(alpha)=배정 가중치
```
- `dim`은 채널 차원(=`in_features`, `QLinear`에서 주입, `:404,407`). 즉 **그룹 배정 단위는 "채널(임베딩 차원)"** 이다(코드 확인). README의 "patch group assignment"라는 표현과 달리, **코드상 배정은 채널 축(dim) 단위 softmax**로 구현되어 있다 — 용어와 구현의 미묘한 차이에 주의(코드 확인).

#### QuantGroup (`quantize_utils.py:93-110`) — 미분 가능한 그룹 혼합
```python
for group in self.groups_range1.data:
    self.mix_activ_mark1.append(Quant(range=group, dim=self.dim))   # 그룹 수만큼 양자화기 생성
self.sw = F.softmax(self.alpha_activ, dim=0)                        # :100  그룹축 softmax → 배정 가중치
for i, branch in enumerate(self.mix_activ_mark1):
    ori_x, x = branch(input, half_wave, a_bit)                      # i번째 그룹 클립범위로 양자화
    outs_x   += (x     * self.sw[i])                                # :104-107  sw로 가중합
activ = torch.sum(outs_x, dim=0)                                    # :108  모든 그룹의 가중합 = 최종 양자화값
```
즉, **각 그룹마다 독립 클립 범위(`groups_range1[i]`)를 가진 양자화기를 만들고, 입력을 모든 그룹 양자화기로 양자화한 뒤 `softmax(alpha)`로 가중 합산**한다. 이는 EdMIPS류의 "여러 후보를 soft-weight로 섞고 alpha를 미분 학습"하는 **continuous relaxation** 방식이다(코드 확인, `README.md:56` EdMIPS 인용과 일치).

#### Quant (`quantize_utils.py:38-60`) — 채널별 비대칭 클립+라운드
```python
activation_r = self.range.repeat(C)                                # 채널마다 동일 range 복제
scaling_factor1 = activation_r / (2.**a_bit - 1.)                  # :51  스케일 = range/(2^b - 1)
if half_wave == 'A':  ori_x = 0.5*(|x| - |x-r| + r)                # :54  한쪽(>=0) 클립
elif half_wave == 'Q': ori_x = 0.5*(|-x+lw| - |x-rw| + lw + rw)    # :56  양방향(좌/우) 클립
x = round(ori_x / b) * b                                           # :58-59  균일 양자화
```
- 비대칭(asymmetric) 균일 양자화. `half_wave='A'`는 비음수 활성용, `'Q'`는 양/음 모두 있는 활성용(코드 확인).
- forward에서 `ori_x`(클립만)와 `x`(라운드)를 둘 다 반환해, 상위에서 `STE.apply(ori_x, x)`로 **Straight-Through Estimator** 역전파를 적용(`:252-254`, `STE` 정의 `:339-347`).

#### two_groups (`quantize_utils.py:483-524`) — 캘리브레이션용 초기 그룹 분할
캘리브레이션 시(`_quantize_activation` 의 `_calibrate` 분기, `:237-249`) 채널별 활성 범위를 구한 뒤(`find_scale_by_percentile`, `:476-480`), `[min,max]`를 `group_n` 균등 구간으로 나눠 각 채널을 그룹에 배정하고 그룹별 max를 대표값으로 잡는다(코드 확인). 학습 중에는 이 초기 배정이 `alpha_activ`(soft) 학습으로 갱신됨(추정).

#### _quantize_weight (`quantize_utils.py:259-303`)
가중치는 그룹 양자화가 아니라 **레이어 단위 대칭 양자화**: `threshold=|w|.max()`, `scale=threshold/(2^(b-1)-1)`, clamp→round, `b=1`이면 1-bit(부호×평균절댓값, BNN형). STE 적용(`:301`). (코드 확인)

### 3.4 QActvation — 활성 전용 양자화기 (`lib/utils/quantize_utils_activation.py`)

`QModule1`(`:15-205`) + `QActvation`(`:218-224`). `half_wave='Q'`일 때 학습 파라미터 `cw_1`(중심), `dw_1`(폭)을 두고(`:31-32`), **PACT/LSQ류의 학습 가능한 클립 경계**로 활성을 양자화한다:
```python
ori_x = 0.5*((-x + cw_1 - dw_1).abs() - (x - (cw_1+dw_1)).abs() + 2*cw_1)   # :169
scaling_factor = dw_1 / (2.**a_bit - 1.)                                     # :181
x = round(ori_x/scale)*scale                                                 # :183
```
1-bit일 때는 `BinActiveBiReal`로 이진화(`:190`, 단 해당 클래스 정의는 이 파일에 없음 → import 누락 가능, 확인 불가). QActvation은 **그룹 없이 텐서 전체** 단일 클립으로 동작(QModule의 그룹 양자화와 별개)이며, 어텐션에서 qkv/attn-map 양자화에 쓰인다(§3.2). (코드 확인)

### 3.5 differentiable search 메커니즘 — `engine.py` finetune/search loss

`train_one_epoch`(`engine.py:28-103`)의 손실 구성:
```python
loss = criterion(samples, outputs, targets)                 # :51  분류(distill) 손실
if args.group_num > 1:                                        # :53  그룹>1일 때만 추가 손실
    # (1) aux_loss: 양자화 모델 attn ↔ FP 모델 attn 정렬
    fpattn = layer.attent.detach(); fpattn = pow(fpattn, pnorm)   # :65-66
    aux_loss += cal_l2loss(fpattn, attn[j])                       # :67
    # (2) dm_loss: 그룹 배정(alpha softmax)의 엔트로피 (탐색 시에만)
    if args.search == True:
        alpha = layer.sw                                         # :73  softmax(alpha_activ), shape=(group_n,dim)
        for k in range(group_n): dm_loss_t += cal_entropy(alpha[k])  # :77
        dm_loss += dm_loss_t / (group_n*dim)                        # :78
loss = loss + dm_weight*dm_loss + aux_weight*aux_loss        # :80
```
- **aux_loss**(`aux_weight`): FP(full-precision) 교사 모델과 양자화 학생 모델의 **어텐션 맵을 L2로 정렬**한다(`cal_l2loss` `:25-26`, 정규화 후 차이의 제곱평균). `pnorm`은 FP 어텐션을 거듭제곱해 분포를 sharpen/약화하는 지수(`:66`). → "self-information / 어텐션 보존" 효과(추정: 어텐션 분포 보존 = 정보 보존).
- **dm_loss**(`dm_weight`): 그룹 배정 가중치 `sw`(softmax 결과)의 **엔트로피**(`cal_entropy` `:22-23`, `-Σ p·log p`). 엔트로피를 손실에 더하면(최소화) 배정이 **한 그룹으로 뾰족하게(near one-hot) 수렴**하도록 유도 → 미분 탐색을 이산 배정으로 수렴시키는 정규화(코드 확인, EdMIPS의 cost/entropy 정규화와 유사).
- 두 항 모두 `group_num>1`에서만 활성, dm_loss는 `search=True`에서만 활성(코드 확인 → searching/finetuning 단계 구분의 실제 구현).

### 3.6 RL env (HAQ DDPG / quantize_env) — 본 파이프라인 미사용

`lib/rl/ddpg.py`(`DDPG` actor-critic, `:56-115`)와 `lib/env/quantize_env.py`, `lib/env/linear_quantize_env.py`(`LinearQuantizeEnv`, `min_bit/max_bit` 액션공간, `:20-80`)는 **HAQ 원본 mixed-precision 비트 탐색 RL 코드**다(파일 헤더 `:1-3`가 HAQ 명시).

그러나 **`main.py`/`engine.py`는 이 RL/env를 전혀 import하지 않는다**(grep 확인: main/engine은 `lib.utils.quantize_utils`의 `calibrate, QLinear`만 import). 비트폭도 `main.py:304-310`에서 **모든 레이어 동일(`args.bit_width`)** 로 고정한다.
→ **결론: Quantformer 실제 파이프라인에서 mixed-precision RL 비트 탐색은 사용되지 않는다(코드 확인). RL/env는 HAQ에서 가져온 미사용 잔재(dead code).** "Quantformer가 RL로 비트를 탐색한다"는 서술은 **확인 불가/사실 아님**.

---

## 4. 알고리즘 / 수식

### 4.1 균일 양자화 (가중치, 레이어 단위, `quantize_utils.py:282-286`)
threshold `t = max|W|`, 비트 `b`에 대해
```
s = t / (2^(b-1) - 1)
W_q = round( clip(W, -t, t) / s ) · s
```
역전파는 STE: `∂W_q/∂W ≈ 1`.

### 4.2 그룹 양자화 + 미분 배정 (활성, `quantize_utils.py:43-110`)
그룹 수 `G`, 채널 `c`, 그룹 `i`의 학습 클립폭 `r_i`(=`groups_range1[i]`), 배정 로짓 `α∈ℝ^{G×dim}`:
```
배정 가중치:  sw_{i,c} = softmax_i(α)_{i,c}             (그룹축 softmax)
그룹별 양자화: Q_i(x) = round( clip_i(x) / s_i ) · s_i ,  s_i = r_i/(2^b - 1)
최종 활성:    x_q[c] = Σ_{i=1}^{G}  sw_{i,c} · Q_i(x)[c]   (soft mixture)
```
탐색이 끝나면 `sw`가 (엔트로피 정규화로) one-hot에 가까워져 각 채널이 단일 그룹으로 배정됨(추정).

### 4.3 Finetune/Search 총손실 (`engine.py:80`)
```
L = L_cls(distill)  +  aux_weight · Σ_layers ‖ Â_fp^{pnorm} − Â_q ‖_2²   +  dm_weight · (1/G·dim) Σ_layer Σ_i H(sw_i)
```
- `Â`: 정규화된 어텐션 맵, `H`: 엔트로피.
- searching 단계: dm_loss 활성(배정 학습). finetuning 단계(`search=False`): dm_loss 비활성, aux_loss·분류손실로 미세조정(코드 확인).

### 4.4 비트폭 설정 (`main.py:304-324`)
- `--bit-width 4`가 기본(`main.py:178`). 모든 양자화 `QLinear`에 동일 비트 적용(`:307,310,323-324`).
- 어텐션의 qkv/attn-map 활성은 DeiT에서 8-bit 고정(`vit:177-178`), Swin에서 `bitwidth`(`swin:236-237`).
- patch embed conv와 classifier head는 8-bit(`vit:253,336`, `swin:649,744`).

---

## 5. 학습 / 평가 파이프라인

엔트리: `main.py`(`DeiT training script`). 분산학습(`torch.distributed.launch`)을 전제.

1. **모델 생성**(`main.py:255-272`): `create_model(args.model)`(양자화 학생) + `create_model(args.fpmodel)`(FP 교사). 둘 다 `args=args`를 넘겨 group-num 등 주입.
2. **체크포인트 로드**(`:274-298`): `--finetune`(양자화 사전학습), `--fpfinetune`(FP 사전학습). shape 매칭 기반 부분 로드.
3. **비트 설정 + 캘리브레이션**(`:304-327`): 레이어별 `w_bit/a_bit` 설정 후 `calibrate(model, loader, device)`로 클립 범위 초기화(8배치 사용, `quantize_utils.py:446-451`).
4. **이중 옵티마이저**(`:374-383`): 이름에 `alpha`가 든 파라미터(=`alpha_activ`, 그룹 배정)는 `arch_optimizer`(AdamW, lr=1e-4), 나머지는 `optimizer`(AdamW, args.lr). → **DARTS/EdMIPS형 양수준(bi-level) 최적화**(코드 확인).
5. **학습 루프**(`:447-484`): `train_one_epoch` 매 step에서 두 옵티마이저 모두 `step()`(`engine.py:88-92`). FP 모델은 `eval()`로 어텐션 교사 제공(`engine.py:54-56`).
6. **평가**(`engine.py:106-137`): top-1/top-5 정확도, AMP autocast.

**3단계 명령(README/run.sh 기준):**
- Pretraining: `--group-num 1 --bit-width 4 --epoch 120 --lr 1e-4` (shared quant)
- Searching: `--group-num 8 --aux-weight 20 --dm-weight 0.025 --pnorm 2 --search True --epoch 1 --lr 2e-6` (`README.md:39`)
- Finetuning: 같은 group-num/weights, `--search False --epoch 5 --lr 5e-6` (`README.md:46`)
- 실제 스크립트: `deit_run.sh`는 deit_tiny + group-num 2 + search False, `swin_run.sh`는 swin_small + group-num 8 + search True(코드 확인).
- 데이터: ImageNet ILSVRC2012 (`*_run.sh` data-path, `README.md:16`).

---

## 6. 의존성 (출처)

`README.md:49-56` 및 코드 헤더로 확인:
- **DeiT** (facebookresearch/deit): `main.py`/`engine.py`/`losses.py`/`models.py`의 골격, DistillationLoss, 학생-교사 distill 구조.
- **Swin Transformer** (microsoft): `swin_transformer.py` 백본.
- **HAQ** (mit-han-lab/haq): `lib/utils/quantize_utils.py`·`quantize_utils_activation.py`·`rl/ddpg.py`·`env/*.py`의 헤더가 HAQ 명시(`quantize_utils.py:1-3`). 단 RL/env는 본 파이프라인 미사용(§3.6).
- **EdMIPS** (zhaoweicai/EdMIPS): 미분 가능한 후보 혼합(soft-weight + alpha 학습) 양자화 탐색 방식의 기반(`QuantGroup`/`alpha_activ`/이중 옵티마이저 구조가 EdMIPS류, 추정+README 인용).
- timm(custom_timm): 데이터·증강·레지스트리·기타 백본.

---

## 7. 강점 / 한계 / 리스크

**강점**
- 그룹 양자화로 채널별 분포 이질성을 흡수 → 단일 스케일 4-bit 대비 정확도 유리(추정).
- 그룹 배정을 **미분 탐색**으로 자동 학습(수동 그룹 분할 불필요), 엔트로피 정규화로 이산화 수렴.
- FP 어텐션 정렬(aux_loss)로 저비트에서 어텐션 구조 보존 → 정확도 회복.

**한계 / 리스크**
- **QAT라 학습 비용이 큼**: FP 교사+양자화 학생 동시 forward, ImageNet 전체 학습(pretraining 120 epoch). searching 단계는 group마다 양자화기를 forward해 메모리/연산 추가(`QuantGroup`이 그룹 수만큼 반복, `:101-107`).
- **`QuantGroup`이 매 forward마다 `Quant` 모듈을 새로 생성**(`:97-99`)·텐서 cat 반복 → 비효율(추정, 성능 리스크).
- **코드 정합성 이슈**(§3.2 경고 1·2): 이 REF 사본은 `self.attent` 대입 누락·`args.bitwidth` 속성 불일치가 있어, 그대로 실행 시 aux_loss/일부 경로가 동작하지 않을 수 있음(확인 불가).
- **그룹 배정 단위가 README("patch group")와 코드(채널/dim축)가 어긋남** — 재현 시 의미 해석 주의.
- 가중치는 그룹 양자화가 아님(레이어 단위) → 그룹 이득은 활성에 한정(코드 확인).

---

## 8. 우리 프로젝트(HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적) 관점 시사점

> 전제: 우리 프로젝트가 ViT/Transformer FPGA 데이터플로우 가속기(HG-PIPE 류)와 XR 시선추적이라는 점은 **추정**.

1. **극저비트 ViT QAT의 정확도 확보 전략으로 유효**: 4-bit ViT를 정확도 손실 적게 얻으려면 PTQ보다 QAT가 유리하다는 점을 코드로 확인. 다만 ImageNet 풀 학습 비용이 크므로, 우리 시선추적 도메인에서는 **소형 데이터셋 + 짧은 finetuning**(README의 search 1 epoch / finetune 5 epoch처럼)로 적용 가능성 검토.
2. **group-wise(채널 그룹) 양자화 ↔ FPGA 매핑**: 채널을 G개 그룹으로 나눠 그룹마다 다른 스케일을 쓰는 구조는, FPGA에서 **채널 타일 단위로 서로 다른 스케일 레지스터/시프트**를 두는 데이터플로우와 잘 맞는다(추정). HG-PIPE류 파이프라인에서 채널 그룹별 dequant 상수를 BRAM에 두는 설계로 직접 매핑 가능.
3. **soft-mixture 탐색은 학습 전용, 추론은 단일 그룹**: `dm_loss` 엔트로피 정규화로 배정이 one-hot에 수렴하므로, **배포(FPGA) 시점에는 채널→그룹 lookup 1개**만 남는다 → 추론 하드웨어는 mixture 연산 불필요(softmax 합산은 학습 그래프에만 존재). 이는 우리 가속기에 부담 없는 좋은 성질(코드 확인 기반 추정).
4. **mixed-precision RL search는 본 코드에 실제 미사용** → "Quantformer가 RL 비트할당을 한다"고 인용하면 오류. FPGA 비트할당 설계 참고가 필요하면 **HAQ 원본**을 직접 참조해야 하며, Quantformer에서 가져올 것은 **균일 4-bit + 채널 그룹 스케일** 개념이다(코드 확인).
5. **XR 어텐션 보존**: aux_loss(FP 어텐션 L2 정렬)는 시선추적처럼 **공간적 attention 정확도가 중요한 태스크**에서 저비트화 후 attention drift를 막는 정규화로 차용할 가치가 있음(추정).

---

## 9. 근거 표기 요약

| 주장 | 근거 | 상태 |
|---|---|---|
| group-wise 활성 양자화 = softmax(alpha) soft-mixture | `quantize_utils.py:88-110` | 코드 확인 |
| 그룹 배정 단위 = 채널/dim (patch 아님) | `quantize_utils.py:88`(`group_n,dim`), `QLinear` dim=in_features `:404,407` | 코드 확인 |
| 이중 옵티마이저(arch=alpha) = 미분 탐색 | `main.py:374-383` | 코드 확인 |
| dm_loss = sw 엔트로피, search 시만 | `engine.py:70-78` | 코드 확인 |
| aux_loss = FP/Q 어텐션 L2 정렬 | `engine.py:58-67` | 코드 확인 |
| 비트폭 전 레이어 균일(mixed X) | `main.py:304-324` | 코드 확인 |
| RL/DDPG/env 미사용(HAQ 잔재) | main/engine import 부재(grep), `ddpg.py:1-3` | 코드 확인 |
| `self.attent` 대입 누락(aux 경로 미완) | engine만 읽음, swin/vit forward에 대입 없음 | 확인 불가(버그/누락 추정) |
| `args.bitwidth` vs `args.bit_width` 불일치 | `main.py:178` vs `swin:209`/`vit:170` | 확인 불가(주의) |
| 우리 프로젝트 성격(FPGA ViT+XR) | 외부 전제 | 추정 |
