# CLAMP-ViT 코드베이스 정밀 분석

> 대상 경로: `REF/ViT-Quantization/CLAMP-ViT/`
> 원논문: **CLAMP-ViT: Contrastive Data-Free Learning for Adaptive Post-Training Quantization of ViTs** (ECCV 2024)
> 분석 방식: 저장소 내 실제 소스코드 직접 판독 (Glob/Grep/Read). 모든 주장은 `파일:라인`으로 근거 표기.

---

## 0. 핵심 결론 먼저 (가장 중요한 발견)

**이 공개 저장소는 "평가(evaluation) 전용" 코드이며, 논문의 두 핵심 기여(① contrastive data-free 합성데이터 생성, ② evolutionary mixed-precision search)의 구현 코드는 저장소에 포함되어 있지 않다.**

- 저장소 전체를 `contrastive | InfoNCE | nce_loss | evol | mutation | crossover | population | generate_data | synthe | optimize image` 등으로 전수 검색한 결과, 위 메커니즘을 구현한 코드는 **단 한 줄도 존재하지 않는다** (Grep 전수 검색 결과: 매치된 곳은 `README.md`의 논문 제목/출처 표기와 `test_clamp_vit.py:27`의 argparse 문자열 선택지뿐).
- `test_clamp_vit.py`는 사전에 양자화·MPQ(Mixed-Precision Quantization)가 완료된 체크포인트(`quantized/deit_tiny.pth`, `quantized/deit_small.pth`)를 `torch.load`로 그대로 불러와 ImageNet val 정확도만 측정한다 (`test_clamp_vit.py:224-235`).
- `models/ptq/` 트리는 **전부 FQ-ViT 코드를 그대로 가져온 것**이다. 모든 파일 헤더에 `# Adapted from MEGVII Inc. https://github.com/megvii-research/FQ-ViT/tree/main`가 명시되어 있다 (예: `models/ptq/quantizer/base.py:1-2`, `models/ptq/observer/build.py:1-2`, `models/vit_quant.py:1-2`).
- 따라서 본 문서의 3·4절에서 분석하는 "양자화 엔진"은 **FQ-ViT 기반 PTQ 인프라**이고, 논문의 contrastive/evol 알고리즘은 **코드로 확인 불가(미공개)**이며 논문 본문 기준으로만 서술한다(해당 부분은 "추정/논문기준"으로 명시).

---

## 1. 개요

- **목적**: ViT(Vision Transformer) 계열 모델을 **데이터 없이(data-free)** 사후 양자화(PTQ)하고, **레이어별로 비트폭을 다르게 할당(layer-wise mixed precision)**하여 정확도-압축 트레이드오프를 최적화한다.
- **원논문 핵심 아이디어(논문 기준)**:
  1. **Contrastive data-free 합성데이터 생성** — 원본 학습 데이터 접근 없이, contrastive learning 목적함수로 의미 있는(semantic) 합성 캘리브레이션 이미지를 생성한다. 프라이버시/배포 제약이 있는 환경(예: 의료, 사용자 디바이스)을 겨냥.
  2. **Layer-wise mixed-precision (MPQ) evolutionary search** — 각 레이어의 비트폭을 진화 탐색(evolutionary search)으로 결정하여 평균 비트(예: W4.9/A6.2)에서 정확도 손실을 최소화.
- **기반 코드**: README.md:56 에 명시 — **FQ-ViT**(양자화 엔진·관찰자·정수 LayerNorm/Softmax)와 **Evol-Q**(진화 기반 양자화 탐색)에서 파생.
- **공개 저장소의 실제 범위**: 위 0절대로 **평가 전용**. 사전 양자화된 MPQ 체크포인트의 ImageNet 정확도 재현이 목적 (README.md:39-54).

---

## 2. 디렉토리 구조

```
CLAMP-ViT/
├── test_clamp_vit.py        # 평가 진입점: ImageNet val 정확도 측정 (양자화 X, 사전 .pth 로드)
├── config.py                # Config 클래스: 비트타입/관찰자/양자화기/캘리브레이션 모드 설정
├── README.md                # 사용법, 결과표(W4.9/A6.2 등), 출처(FQ-ViT+Evol-Q)
├── LICENSE
├── quantized/               # [제외-체크포인트] 사전 양자화 MPQ 모델
│   ├── deit_tiny.pth        #   (이름만 — torch.load로 통째로 로드되는 nn.Module)
│   └── deit_small.pth
├── models/
│   ├── __init__.py          # from .ptq import *, from .vit_quant import *
│   ├── vit_quant.py         # 양자화 ViT 구조 (Attention/Block/VisionTransformer, DeiT/ViT 팩토리)
│   ├── layers_quant.py      # Mlp/PatchEmbed/HybridEmbed/DropPath/trunc_normal_ (양자화 빌딩블록)
│   ├── utils.py             # .npz(Google Brain Flax) 가중치 로더
│   └── ptq/                 # ★ 자체 양자화 핵심 (전부 FQ-ViT에서 파생)
│       ├── __init__.py
│       ├── bit_type.py      # BitType 정의 (int8/uint8/uint7..uint2)
│       ├── layers.py        # QConv2d/QLinear/QAct/QLayerNorm/QSoftmax 양자화 모듈
│       ├── quantizer/       # 양자화기 (값→정수 매핑)
│       │   ├── base.py      #   BaseQuantizer (reshape range, quant/dequant 인터페이스)
│       │   ├── uniform.py   #   UniformQuantizer (균일 양자화: scale·zero_point)
│       │   ├── log2.py      #   Log2Quantizer (로그2 양자화: 시프트 친화적)
│       │   └── build.py     #   str2quantizer 팩토리
│       └── observer/        # 관찰자 (캘리브레이션: min/max·scale 추정)
│           ├── base.py      #   BaseObserver (reshape_tensor, update 인터페이스)
│           ├── minmax.py    #   MinmaxObserver
│           ├── ema.py       #   EmaObserver (지수이동평균)
│           ├── percentile.py#   PercentileObserver (분위수 클리핑)
│           ├── omse.py      #   OmseObserver (Lp-norm 최소화, LAPQ식)
│           ├── ptf.py       #   PtfObserver (Power-of-Two Factor, LayerNorm용)
│           ├── utils.py     #   lp_loss
│           └── build.py     #   str2observer 팩토리
└── .git/, __pycache__/      # [제외-이름만]
```

> **제외(이름만)**: `.git/`, 모든 `__pycache__/`, `quantized/*.pth`(사전 양자화 체크포인트), `LICENSE`.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `config.py` — 전역 양자화 설정

`Config` 클래스 하나로 가중치(W)/활성화(A)의 양자화 정책을 일괄 설정한다 (`config.py:4-19`).

| 항목 | 값 | 근거 |
|---|---|---|
| 가중치 비트타입 | `int8` (대칭, signed) | `config.py:9` |
| 활성화 비트타입 | `uint8` (비대칭, unsigned) | `config.py:10` |
| 가중치 관찰자 | `minmax` | `config.py:12` |
| 활성화 관찰자 | `quant_method` 인자(기본 `minmax`) | `config.py:13` |
| 양자화기 (W/A) | `uniform` / `uniform` | `config.py:15-16` |
| 캘리브레이션 모드 | W=`channel_wise`, A=`layer_wise` | `config.py:18-19` |

- **주의점 1**: `Config.__init__`은 `INT_NORM` 속성을 설정하지 않는다. 그런데 `vit_quant.py:358`의 `model_quant()`에서 `self.cfg.INT_NORM`을 참조한다 → 기본 Config로 `model_quant()` 호출 시 `AttributeError` 발생. 이는 이 저장소에서 `model_quant()`가 실행 경로에 없음(체크포인트를 통째 로드)을 방증한다. (코드 확인: `config.py` 전체에 `INT_NORM` 없음 / `vit_quant.py:358`에서 참조)
- **주의점 2**: `--quant-method`의 선택지는 `['minmax', 'contrastive']` (`test_clamp_vit.py:27`)이지만, `'contrastive'`라는 이름의 **관찰자(observer)는 `str2observer`에 등록되어 있지 않다** (`models/ptq/observer/build.py:10-16` — minmax/ema/omse/percentile/ptf만 존재). 즉 `--quant-method contrastive`를 실제로 주면 `build_observer`에서 `KeyError`가 난다. 이는 contrastive 로직이 코드에 부재함을 다시 확인시켜 준다.

### 3.2 `models/ptq/bit_type.py` — 비트 정의

- `BitType(bits, signed, name)` 클래스가 정수 양자화의 표현 범위를 정의 (`bit_type.py:6-38`).
  - `upper_bound`: signed면 `2^(bits-1)-1`, unsigned면 `2^bits-1` (`bit_type.py:16-21`).
  - `lower_bound`: signed면 `-2^(bits-1)`, unsigned면 `0` (`bit_type.py:22-26`).
  - `range`: `2^bits` (`bit_type.py:28-30`).
- 사전 등록 비트타입: `int8, uint8, uint7, uint6, uint5, uint4, uint3, uint2` (`bit_type.py:41-49`). → **uint2~uint8**까지 표현 가능 → mixed precision 비트 후보군의 기반이 되는 자료구조(단, 어떤 레이어에 어떤 비트를 할당하는 탐색 로직 자체는 부재).
- `BIT_TYPE_DICT`로 이름→객체 매핑 (`bit_type.py:50`).

### 3.3 `models/ptq/quantizer/` — 양자화기

#### `base.py` — `BaseQuantizer`
- `get_reshape_range(inputs)`: 모듈 타입별 broadcast용 reshape 형상 반환 (`quantizer/base.py:15-32`).
  - `conv_weight` → `(-1,1,1,1)`, `linear_weight` → `(-1,1)` (per-output-channel scale)
  - `activation` → 2D `(1,-1)`, 3D `(1,1,-1)`, 4D `(1,-1,1,1)`
- `forward`: `quant` → `dequantize` 순으로 **fake quantization**(시뮬레이션 양자화) 수행 (`quantizer/base.py:43-46`).

#### `uniform.py` — `UniformQuantizer` (균일 양자화 — 핵심)
- `update_quantization_params`: 관찰자로부터 `scale, zero_point`를 받아 저장 (`quantizer/uniform.py:16-18`).
- `quant` (양자화): `q = round(x / scale + zero_point)` 후 `[lower_bound, upper_bound]` 클램프 (`quantizer/uniform.py:20-31`).
- `dequantize` (역양자화): `x̂ = (q - zero_point) * scale` (`quantizer/uniform.py:33-42`).
- scale/zero_point는 `get_reshape_range`로 reshape 후 broadcast (`quantizer/uniform.py:25-27, 38-40`) → **W는 채널별, A는 레이어별** 적용.

#### `log2.py` — `Log2Quantizer` (로그2 양자화 — 하드웨어 친화적)
- `quant`: `round(-log2(x))` → 비트 범위 밖 값은 `softmax_mask`로 마킹, `[0, 2^bits-1]` 클램프 (`quantizer/log2.py:18-22`).
- `dequantize`: `x̂ = 2^(-q)`, 마스크된 위치는 0 (`quantizer/log2.py:24-27`).
- **의미**: 어텐션 softmax 출력처럼 [0,1] 범위에 long-tail 분포를 갖는 값에 적합. 곱셈이 비트 시프트로 치환되어 **FPGA/ASIC에서 곱셈기 없이 구현 가능**. (단, `config.py`에서는 기본적으로 W/A 모두 `uniform`을 사용 — log2는 옵션으로만 존재).
- `build.py`: `str2quantizer = {'uniform':..., 'log2':...}` 팩토리 (`quantizer/build.py:6-11`).

### 3.4 `models/ptq/observer/` — 관찰자(캘리브레이션)

모든 관찰자는 `update(v)`로 calibration 데이터의 min/max를 누적하고, `get_quantization_params()`로 scale/zero_point를 산출한다. **대칭(signed)/비대칭(unsigned) 분기 공식은 minmax/ema/percentile에서 동일**하다.

#### `base.py` — `BaseObserver`
- `reshape_tensor(v)`: 가중치는 `(out_ch, -1)`로, 활성화는 마지막 채널 기준 `(C, N)`으로 정렬 (`observer/base.py:17-30`).
- `eps = finfo(float32).eps` (scale 0 방지) (`observer/base.py:15`).

#### `minmax.py` — `MinmaxObserver` (기본값)
- `update`: 채널별 max/min 누적, `layer_wise`면 전체 스칼라로 축약 (`observer/minmax.py:15-30`).
- `get_quantization_params` (`observer/minmax.py:32-52`):
  - **대칭(signed, 가중치)**: `max_val = max(-min, max)`, `scale = max_val / ((qmax-qmin)/2)`, `zero_point = 0`.
  - **비대칭(unsigned, 활성화)**: `scale = (max-min)/(qmax-qmin)`, `zero_point = qmin - round(min/scale)` 후 클램프.

#### `ema.py` — `EmaObserver`
- min/max를 **지수이동평균**으로 갱신: `max_val += ema_sigma·(cur_max - max_val)`, `ema_sigma=0.01` (`observer/ema.py:14, 20-37`). scale 공식은 minmax와 동일 (`observer/ema.py:39-59`).

#### `percentile.py` — `PercentileObserver`
- `layer_wise`에서만 동작 (`observer/percentile.py:26`). `torch.quantile`로 상위/하위 분위수(`alpha=0.99999`) 클리핑하여 outlier 영향 완화 (`observer/percentile.py:24-50`). EMA식 갱신(`sigma=0.01`).

#### `omse.py` — `OmseObserver` (Lp-norm 최소화)
- max/min을 0~89% 범위로 0.01씩 줄여가며(`for i in range(90)`), 각 후보 scale로 양자화→복원 후 **Lp(p=2) 손실이 최소가 되는 clipping range를 탐색** (`observer/omse.py:32-57`). LAPQ(arXiv 1911.07190) 방식 (`observer/omse.py:48-49`).

#### `ptf.py` — `PtfObserver` (Power-of-Two Factor, LayerNorm/활성화용 — 핵심)
- FQ-ViT의 핵심 기법. LayerNorm 입력처럼 채널 간 분포 분산이 큰 경우, **채널마다 2의 거듭제곱(power-of-two) 스케일 팩터**를 곱해 채널별 동적 범위 차이를 흡수한다.
- 절차 (`observer/ptf.py:32-67`):
  1. 전체 min/max로 기준 `scale8` 산출 (`ptf.py:42-43`).
  2. `scale4 = scale8/2`, `scale2 = scale4/2`, `scale1 = scale2/2` — 2의 거듭제곱 후보 (`ptf.py:44-46`).
  3. zero_point는 공유 (`ptf.py:47-48`).
  4. **채널 j마다** 4개 후보 scale로 양자화→복원, `lp_loss`가 최소인 거듭제곱 지수를 선택: `scale_mask[j] *= 2^(argmin)` (`ptf.py:50-65`).
  5. 최종 `scale = scale1 * scale_mask` (채널별 2^k 배율) (`ptf.py:66`).
- **하드웨어 의미**: 채널별 배율이 모두 **2의 거듭제곱** → 정수 연산 후 **비트 시프트만으로 채널별 재정규화** 가능. (단, 이 저장소 `config.py`는 기본 관찰자로 ptf를 쓰지 않음 — minmax가 기본).
- `build.py`: `str2observer = {minmax, ema, omse, percentile, ptf}` (`observer/build.py:10-16`) — **`contrastive` 키 없음**(3.1 주의점 2 참조).

### 3.5 `models/ptq/layers.py` — 양자화 연산 모듈

#### `QConv2d` (`layers.py:10-69`), `QLinear` (`layers.py:72-109`)
- `nn.Conv2d`/`nn.Linear` 상속. 내부에 `observer`+`quantizer`를 빌드 (`layers.py:47-50, 96-99`).
- forward 흐름 (`layers.py:52-69, 101-109`):
  - `calibrate=True`면 가중치로 observer 갱신; `last_calibrate=True`면 `update_quantization_params(x)` 호출.
  - `quant=False`면 원본 conv/linear, `quant=True`면 `weight = quantizer(self.weight)`로 가중치를 fake-quant 후 연산.

#### `QAct` (`layers.py:112-147`)
- 활성화 양자화 전용 모듈. calibrate 시 입력 x로 observer 갱신 (`layers.py:138-143`), quant 시 `x = quantizer(x)`.

#### `QLayerNorm` (`layers.py:150-205`) — 정수 LayerNorm (FQ-ViT의 핵심)
- `mode='ln'`: 일반 부동소수 `F.layer_norm` (`layers.py:169-171`).
- `mode='int'`: **정수 LayerNorm** (`layers.py:172-202`):
  - 입력을 in_scale로 정수화 (`x_q = round(x/in_scale)`), 채널별 in_scale 차이를 정수 마스크로 반영 (`layers.py:182-186`).
  - 정수 평균/표준편차 계산 (`layers.py:188-190`).
  - affine 계수 A를 `A_sign·M·2^(-N)` 형태(고정소수점)로 분해 — `get_MN`이 `M, N`(8비트 가수, 시프트량)을 산출 (`layers.py:158-162, 192-201`).
  - **의미**: LayerNorm을 부동소수 없이 정수 곱+시프트로 근사 → 하드웨어 친화적. (이 저장소에선 `INT_NORM` 미설정으로 비활성, 3.1 주의점 1).

#### `QSoftmax` (`layers.py:208-271`) — (부분적으로) 정수 Softmax
- `int_softmax`/`int_exp`/`int_polynomial` 정적 메서드로 **정수 지수함수 근사**(2차 다항식, `-ln2` 기반 range reduction)를 구현 (`layers.py:234-266`). I-BERT/FQ-ViT 계열 기법.
- **단, 실제 `forward`는 `x.softmax(dim=-1)`로 부동소수 softmax를 그대로 사용**하고 `int_softmax`를 호출하지 않는다 (`layers.py:268-270`). 즉 정수 softmax 코드는 존재하나 평가 경로에서는 미사용(원본 FQ-ViT에서 가져왔으나 비활성화).

### 3.6 `models/vit_quant.py` — 양자화 ViT 구조

#### `Attention` (`vit_quant.py:27-115`)
- `qkv`(QLinear), `proj`(QLinear) + 활성화 양자화기 `qact1/qact2/qact3/qact_attn1`(QAct) + `softmax`(QSoftmax) (`vit_quant.py:44-93`).
- forward (`vit_quant.py:95-115`):
  - `x = qkv(x); x = qact1(x)` → QKV 분리.
  - `attn = (q @ kᵀ)·scale; attn = qact_attn1(attn)` — 어텐션 스코어 양자화 (`vit_quant.py:106-107`).
  - `attn = softmax(attn, qact_attn1.quantizer.scale)` — softmax에 스케일 전달 (`vit_quant.py:108`).
  - `x = (attn @ v); x = qact2(x); x = proj(x); x = qact3(x)` (`vit_quant.py:110-113`).
  - → **QKV·어텐션맵·proj 전부 양자화 경로에 포함**.

#### `Block` (`vit_quant.py:118-190`)
- `norm1/norm2`(QLayerNorm) + `attn`(Attention) + `mlp`(Mlp) + `qact1~qact4`(QAct) (`vit_quant.py:135-178`).
- forward에서 **이전 모듈의 quantizer를 다음 LayerNorm에 전달**(`last_quantizer`, `qact*.quantizer`) — 정수 LayerNorm 모드를 위한 in/out scale 연결 (`vit_quant.py:180-189`).

#### `VisionTransformer` (`vit_quant.py:193-414`)
- 입력 양자화(`qact_input`), patch embed, cls token + pos embed 양자화(`qact_embed/qact_pos/qact1`), blocks, 최종 norm/head/act_out 전부 양자화 모듈로 구성 (`vit_quant.py:229-328`).
- forward에서도 인접 quantizer를 LayerNorm에 체이닝 (`vit_quant.py:399-405`).
- **양자화 상태 토글 메서드**: `model_quant`/`model_dequant`/`model_open_calibrate`/`model_open_last_calibrate`/`model_close_calibrate` (`vit_quant.py:354-380`). → calibration 파이프라인 인프라는 갖춰져 있으나, `test_clamp_vit.py`는 이를 호출하지 않고 사전 양자화 모델을 로드만 한다.
- **모델 팩토리**: `deit_tiny/small/base_patch16_224`, `vit_base/large_patch16_224` (`vit_quant.py:417-548`). DeiT는 facebookresearch deit pth, ViT는 Google Brain npz에서 사전학습 가중치 로드.
- **주의**: `test_clamp_vit.py:160-167`의 `str2model`은 deit/vit 5종만 매핑하며 **Swin은 없다**. README 결과표(Swin-T/S)는 있으나 코드 경로엔 swin 모델 정의/매핑이 부재 → Swin은 코드로 재현 불가(확인 불가).

### 3.7 `models/layers_quant.py` — 양자화 빌딩블록
- `Mlp` (QLinear×2 + QAct×2 + GELU) (`layers_quant.py:118-173`), `PatchEmbed`(QConv2d + 선택적 QLayerNorm) (`layers_quant.py:176-247`), `HybridEmbed`(CNN 백본 임베딩) (`layers_quant.py:250-297`), `DropPath`/`trunc_normal_` (`layers_quant.py:66-115`). 전부 FQ-ViT 파생.

### 3.8 `models/utils.py` — 가중치 로더
- `load_weights_from_npz`: Google Brain Flax `.npz` 체크포인트를 PyTorch ViT로 변환(축 transpose, pos embed 보간) (`utils.py:12-198`). ViT-B/L 사전학습 로딩용.

### 3.9 contrastive data-free 합성데이터 생성 / evolutionary mixed-precision search 로직
- **코드 확인 결과: 저장소에 부재(미공개).**
  - `contrastive`: `README.md:4`(제목), `test_clamp_vit.py:27`(argparse 문자열)에서만 등장. 손실 함수/데이터 생성 루프 없음.
  - `evol/evolution/mutation/crossover/population/search`: **소스 매치 0건**. (Evol-Q 출처는 README.md:56에만 언급).
  - 합성 이미지 최적화(이미지를 학습 파라미터로 backprop), 가우시안 초기화, augmentation 기반 contrastive view 생성 코드 모두 없음.
- 즉 **MPQ 비트할당 결과와 contrastive 합성데이터로 캘리브레이션된 scale은 이미 `quantized/*.pth`에 "구워져(baked-in)" 있고**, 이를 산출한 학습/탐색 코드는 공개되지 않았다.

---

## 4. 알고리즘 / 수식

> 4.1~4.3은 **코드로 확인**된 양자화 수식. 4.4~4.5는 **코드 부재 → 논문 기준 서술(추정)**.

### 4.1 균일 양자화 (Uniform Quant) — 코드 확인
양자화 (`quantizer/uniform.py:28-30`):
```
q = clamp( round(x / s + z),  q_min,  q_max )
```
역양자화 (`quantizer/uniform.py:41`):
```
x̂ = (q − z) · s
```
스케일/제로포인트 산출:
- 대칭(가중치, signed) (`observer/minmax.py:42-46`): `s = max(|min|,|max|) / ((q_max−q_min)/2)`,  `z = 0`
- 비대칭(활성화, unsigned) (`observer/minmax.py:47-51`): `s = (max−min)/(q_max−q_min)`,  `z = clamp( q_min − round(min/s), q_min, q_max )`

### 4.2 로그2 양자화 (Log2 Quant) — 코드 확인 (`quantizer/log2.py`)
```
q = clamp( round( −log2(x) ),  0,  2^b − 1 )      # 양자화
x̂ = 2^(−q)        (단, q ≥ 2^b 인 위치는 0)         # 역양자화
```
→ 곱셈이 시프트로 환원. softmax 같은 [0,1] long-tail 분포용.

### 4.3 PTF (Power-of-Two Factor, LayerNorm 채널별) — 코드 확인 (`observer/ptf.py:42-66`)
기준 스케일 `s8 = (max−min)/(q_max−q_min)`, 후보 `s8, s4=s8/2, s2=s4/2, s1=s2/2`.
채널 j마다:
```
k_j = argmin_{k∈{0,1,2,3}}  || x_j − dequant(quant(x_j; s1·2^k)) ||₂
s_j = s1 · 2^{k_j}
```
→ 채널별 스케일이 공통 베이스 `s1`에 **2의 거듭제곱 배율**만 곱한 형태 → 비트 시프트로 채널 재정규화.

정수 LayerNorm의 affine 분해 (`layers.py:158-162`): 부동소수 계수 `A`를
```
A ≈ sign(A) · M · 2^(−N),   N = clamp(8−1−⌊log2 A⌋, 0, 31),  M = clamp(⌊A·2^N⌋, 0, 255)
```
로 8비트 가수 M과 시프트량 N으로 표현.

### 4.4 Contrastive Loss / data-free 합성 이미지 최적화 — 코드 부재(논문 기준)
- 논문 기준: 합성 입력 이미지를 학습 가능한 텐서로 두고, 동일 이미지의 두 증강 뷰(view)에 대해 모델 특징의 **contrastive(예: InfoNCE 류) 목적함수**를 최소화하여 의미 있는 캘리브레이션 샘플을 생성한다고 기술. **그러나 본 저장소에는 손실 정의·생성 루프가 없어 정확한 수식/온도(temperature)/뷰 구성은 코드로 확인 불가**.

### 4.5 Layer-wise Mixed Precision Evolutionary Search — 코드 부재(논문 기준)
- 논문 기준: 각 레이어의 비트폭을 유전자로 하는 개체군을 두고, **합성데이터에 대한 성능(블록 출력 오차/정확도)을 적합도(fitness)로 진화 탐색**, 평균 비트 제약(예: W4.9) 하에서 정확도 최대화. **본 저장소에는 탐색 루프·mutation/crossover·적합도 함수가 없어 코드로 확인 불가**. 결과(비트 배분)는 `quantized/*.pth`에 내장.

---

## 5. 학습 / 평가 파이프라인

- **진입점**: `test_clamp_vit.py main()` (`test_clamp_vit.py:190-235`).
  1. seed 고정(결정론적) (`test_clamp_vit.py:172-187, 192`).
  2. `Config(quant_method)` 생성, `str2model(model)(pretrained=True, cfg)`로 모델 골격 생성 (`test_clamp_vit.py:195-197`). — **단, 이 모델은 곧 폐기됨**.
  3. 모델별 전처리(mean/std/crop_pct) 분기: deit vs vit (`test_clamp_vit.py:200-210`).
  4. ImageNet **val** 폴더만 로드(`ImageFolder(valdir)`) — **train 불필요(data-free 결과 평가)** (`test_clamp_vit.py:214-223`).
  5. **`model = torch.load(args.weight_path)`** — 사전 양자화 MPQ 모델(`quantized/*.pth`)을 통째로 덮어쓰기 로드 (`test_clamp_vit.py:225`).
  6. `validate()`로 top-1/top-5 측정 (`test_clamp_vit.py:45-90, 233`).
- **실행 예** (README.md:44): `python test_clamp_vit.py --model deit_tiny --weight-path ./quantized/deit_tiny.pth --data <IMAGENET>`
- **보고된 결과** (README.md:48-54): DeiT-T(W4.9/A6.2) 71.69, DeiT-S(W4.7/A5.9) 79.43, Swin-T(W5.5/A6.9) 81.78, Swin-S(W5.1/A6.3) 82.86.
  - 단, `quantized/`에는 **deit_tiny.pth, deit_small.pth 2개만** 존재(Glob 확인). Swin/DeiT-base 등 나머지 체크포인트와 Swin 모델 코드는 부재 → 표의 일부만 재현 가능.
- **양자화/캘리브레이션 자체는 이 저장소에서 수행되지 않음** — calibration 토글 메서드(`vit_quant.py:354-380`)는 존재하나 호출되지 않음.

---

## 6. 의존성

- **FQ-ViT** (megvii-research): `models/ptq/` 전체, `vit_quant.py`, `layers_quant.py`, `utils.py`의 직접 원천. 모든 파일 헤더에 명시 (`README.md:56`, 각 파일 1-2행).
- **Evol-Q** (enyac-group): evolutionary 양자화 탐색의 개념적 원천 (`README.md:56`). **코드는 본 저장소에 미포함**.
- **timm**: README 요구사항(`README.md:27`). 단 코드 직접 `import timm`은 핵심 경로에 없음(자체 ViT 구현 사용). `HybridEmbed`가 백본 인터페이스만 가정.
- **기타**: PyTorch(≥1.5), torchvision(데이터 로딩/전처리), PIL, numpy, matplotlib/pandas (`README.md:23-28`).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **Data-free 평가/배포**: 원본 학습 데이터 없이 양자화 모델을 검증 가능(개념). 프라이버시 민감 도메인에 유리(논문 주장).
- **하드웨어 친화 기법 내장**: PTF(채널별 2^k 스케일), Log2 양자화, 정수 LayerNorm(M·2^−N 분해), 정수 Softmax 근사 — 모두 **시프트 기반**으로 FPGA/ASIC 매핑에 직접 유용.
- **모듈식 PTQ 인프라**: observer/quantizer/layer가 팩토리 패턴으로 분리되어 비트폭·관찰자 교체가 쉽다(`build_*` 함수).

**한계 / 리스크**
- **핵심 알고리즘 미공개(가장 큰 리스크)**: contrastive 합성데이터 생성·evolutionary MPQ 탐색 코드가 없어 **논문의 핵심 기여를 코드로 재현 불가**. 본 저장소만으로는 "결과 정확도 측정"만 가능.
- **양자화 품질이 사전 체크포인트에 종속**: scale/zero_point/비트배분이 `*.pth`에 내장 → 새 모델·새 비트제약으로 재탐색 불가.
- **합성데이터 품질 의존(논문 일반론)**: data-free PTQ는 생성 합성데이터가 실분포를 얼마나 대표하느냐에 정확도가 좌우됨. 본 저장소로는 이 품질을 검증/조정할 수단이 없음.
- **불완전/비활성 코드**: `Config.INT_NORM` 미정의(`model_quant` 호출 시 에러), `QSoftmax.forward`가 정수 softmax 미사용(부동소수로 폴백), `--quant-method contrastive`는 observer 부재로 동작 불가, Swin 모델 코드/일부 체크포인트 부재.

---

## 8. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 FPGA ViT 가속기 + XR 시선추적 — 추정)

> 아래는 코드에서 확인된 기법을 우리 과제(ViT/Transformer FPGA 가속기 + XR eye-tracking)에 연결한 해석. 우리 과제 정의 자체는 본 저장소 외부의 추정.

1. **Data-free PTQ ⇒ XR 프라이버시·배포 이점**: XR 디바이스에서 수집되는 시선/안구 영상은 민감 정보다. data-free 캘리브레이션(논문 컨셉)은 **사용자 데이터 없이 양자화**가 가능해 온디바이스 배포·프라이버시 측면에서 직접적 이점. 단, 본 저장소엔 생성기 코드가 없으므로 **FQ-ViT/Evol-Q 원본 또는 논문 재현이 필요**.

2. **Layer-wise mixed precision ⇒ FPGA 레이어별 비트 할당 설계 참고**: `bit_type.py`의 uint2~uint8 후보군과 W4.9/A6.2 같은 평균 비트 결과는, **HG-PIPE식 레이어별/스테이지별 PE 비트폭 차등 설계**에 그대로 매핑되는 사고틀. 단 탐색 알고리즘은 직접 가져올 수 없고(미공개) Evol-Q 또는 자체 탐색기로 대체 필요.

3. **시프트 기반 연산자 ⇒ DSP-free 하드웨어**: 
   - **PTF(`observer/ptf.py`)**: 채널별 재정규화를 2^k 시프트로 → LayerNorm 전후 채널 스케일링을 곱셈기 없이 구현.
   - **Log2 양자화(`quantizer/log2.py`)**: 어텐션맵 dequant를 시프트로.
   - **정수 LayerNorm(`layers.py:158-202`)**: `A ≈ M·2^−N` 분해 → 고정소수점 곱+시프트.
   - **정수 Softmax(`layers.py:234-266`)**: 2차 다항식 + range reduction으로 exp 근사(현재 비활성이지만 RTL화 시 곱셈·LUT 최소화에 유용).
   → 이들은 FPGA에서 **DSP 사용량/지연을 줄이는 직접적 후보 연산자**.

4. **양자화 ViT 구조 참고**: `vit_quant.py`의 어텐션 양자화 지점(QKV/attn-score/proj/softmax 입력 모두 양자화)은, 우리 가속기의 **양자화 삽입 위치 설계 체크리스트**로 활용 가능.

5. **활용 전략**: 본 저장소는 "참고 평가 코드"로 한정하고, 실제 양자화 파이프라인은 **FQ-ViT(엔진) + Evol-Q(탐색)** 원본을 결합하거나 논문 재현으로 보강해야 한다.

---

## 9. 근거 표기 규칙 / 확인-vs-추정 요약

| 항목 | 상태 | 근거 |
|---|---|---|
| Uniform/Log2 양자화 수식 | **코드 확인** | `quantizer/uniform.py:20-42`, `quantizer/log2.py:18-27` |
| minmax/ema/percentile/omse/ptf 관찰자 | **코드 확인** | `observer/*.py` |
| PTF 채널별 2^k 스케일 | **코드 확인** | `observer/ptf.py:42-66` |
| 정수 LayerNorm(M·2^−N) | **코드 확인** (단 기본 비활성) | `layers.py:150-205`, `config.py`에 `INT_NORM` 부재 |
| 정수 Softmax 근사 | **코드 존재하나 비활성** | `layers.py:234-266` vs 실제 `forward` `layers.py:268-270` |
| QConv/QLinear/QAct/어텐션 양자화 구조 | **코드 확인** | `layers.py`, `vit_quant.py:27-115` |
| 비트 후보 uint2~uint8 | **코드 확인** | `bit_type.py:41-49` |
| **Contrastive 합성데이터 생성** | **코드 부재 (미공개)** | 전수 Grep 0건; `README.md:4`, `test_clamp_vit.py:27`만 문자열 |
| **Evolutionary MPQ 탐색** | **코드 부재 (미공개)** | 전수 Grep 0건; `README.md:56`만 출처 언급 |
| `--quant-method contrastive` 실행 | **동작 불가(확인)** | observer 미등록 `observer/build.py:10-16` |
| MPQ 비트배분(W4.9/A6.2 등) | **결과만 체크포인트 내장** | `quantized/*.pth`, `README.md:48-54` |
| Swin-T/S 코드 | **부재(확인 불가)** | `test_clamp_vit.py:160-167` deit/vit만 매핑 |
| 본 저장소 성격 | **평가(eval) 전용 (확인)** | `test_clamp_vit.py:225` `torch.load`, calibration 토글 미호출 |

**총평**: 이 공개 저장소는 **CLAMP-ViT 논문의 "결과 검증용 평가 코드 + FQ-ViT 기반 양자화 추론 엔진"**이며, 논문 제목이 내세우는 **contrastive data-free 합성데이터 생성과 evolutionary mixed-precision search의 구현은 포함되어 있지 않다(미공개)**. 우리 프로젝트에는 PTF/Log2/정수 LayerNorm·Softmax 등 **시프트 기반 하드웨어 친화 양자화 연산자**와 **양자화 ViT 구조 설계**가 직접적 참고 가치를 가진다.
