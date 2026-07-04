# ViDiT-Q 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/ViDiT-Q`
> 핵심 양자화 패키지: `quant_utils/qdiff/`
> 원논문: ViDiT-Q, *Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation*, ICLR'25, arXiv 2406.02540
> 작성 기준: 실제 소스코드(Glob/Grep/Read)로 확인한 내용만 기술. 코드로 직접 확인한 사실과 "추정"/"확인 불가"를 구분.

---

## 1. 개요

### 1.1 목적
ViDiT-Q는 **Diffusion Transformer(DiT)** 계열 모델(이미지/비디오 생성: OpenSora, Latte, Pixart-α/Σ, DiT-XL/2 등)에 특화된 **PTQ(Post-Training Quantization, 학습후 양자화)** 방법론이다. README에 따르면(`README.md:23`):
- **W8A8**(weight 8bit / activation 8bit) 양자화를 metric 저하 없이 달성
- **W4A8 mixed precision**에서 눈에 띄는 화질 저하 없이 달성

즉 LLM 양자화에서 발전한 기법들(SmoothQuant, QuaRot)을 **diffusion transformer의 특수성**(timestep마다 활성값 분포가 변함, 토큰별 분포 차이가 큼)에 맞게 재구성한 것이 핵심이다.

### 1.2 핵심 아이디어 (코드로 확인된 범위)
1. **Dynamic per-token activation quantization** — 활성값을 정적(offline) 통계가 아니라 forward 시점에 토큰 그룹마다 online으로 scale을 계산해 양자화한다. diffusion의 timestep별 활성 분포 변화에 robust하다. (`base_quantizer.py:100-161`, `quant_layer.py:64-71`)
2. **Static weight quantization** — 가중치는 모델 초기화 시 1회 양자화 후 고정. (`quant_layer.py:40-41`)
3. **SmoothQuant migration** — 활성의 outlier 채널을 가중치로 "이전(migration)"시켜 활성 양자화 난이도를 낮춤. migration 강도 α로 제어. (`sq_quant_layer.py:24-44`)
4. **QuaRot(Hadamard rotation)** — 무작위 Hadamard 회전 행렬로 활성/가중치를 회전시켜 outlier를 분산. (`quarot_quant_layer.py`, `quarot_utils.py:186-208`)
5. **ViDiT-Q layer** — SmoothQuant scaling + QuaRot rotation을 **결합**한 레이어. (`viditq_quant_layer.py:40-73`)
6. **Mixed precision** — layer별(정규식 매칭) 및 timestep/attention-block별 bit-width 차등 할당. (`mixed_precision_quantizer.py`, `quant_model.py:77-106`, `quant_attn.py`)
7. **Attention map quantization** — softmax 후 attention map을 column/block 단위로 양자화 (CogVideoX/OpenSora용). (`quant_attn.py`)
8. **CUDA kernel 추론 경로** — 알고리즘 시뮬레이션과 별개로 실제 INT8 가속 커널(`viditq_extension`) 경로 제공. (`quant_dit.py:31-33,315-346`)

---

## 2. 디렉토리 구조

### 2.1 분석 대상 (자체 quant_utils 핵심)
```
ViDiT-Q/
├── quant_utils/                         # 독립 python 패키지 "qdiff" (pip install -e .)
│   ├── setup.py                         # name='qdiff', find_packages()  (setup.py:1-9)
│   └── qdiff/
│       ├── utils.py                     # apply_func_to_submodules, seed_everything, setup_logging
│       ├── base/
│       │   ├── base_quantizer.py        # StaticQuantizer / DynamicQuantizer (핵심 양자화 수식)
│       │   ├── quant_layer.py           # QuantizedLinear (기본 양자화 Linear)
│       │   ├── quant_attn.py            # QuantizedAttentionMap(+OpenSORA) attention map 양자화
│       │   ├── quant_model.py           # QuantModel 베이스 + layer refactor 로직
│       │   └── mixed_precision_quantizer.py  # MixedPrecision Static/Dynamic Quantizer
│       ├── smooth_quant/
│       │   └── sq_quant_layer.py        # SQQuantizedLinear (SmoothQuant migration)
│       ├── quarot/
│       │   ├── quarot_quant_layer.py    # QuarotQuantizedLinear (Hadamard rotation)
│       │   ├── quarot_utils.py          # Hadamard 행렬 생성/변환 (1.9MB, 상수 행렬 포함)
│       │   └── hadamard_utils/          # 사전 계산 Hadamard 행렬(.pth) + 변환 스크립트
│       └── viditq/
│           └── viditq_quant_layer.py    # ViDiTQuantizedLinear (SmoothQuant + QuaRot 결합)
├── examples/
│   └── dit/                             # DiT-XL/2 예제 (ImageNet class-conditional)
│       ├── ptq.py                       # PTQ 실행 (calib 적용 → quant_params 저장)
│       ├── quant_inference.py           # 양자화 모델 추론(시뮬/하드웨어)
│       ├── get_calib_data.py            # 채널별 활성 통계 수집(calib data)
│       ├── sweep_alpha.py               # SmoothQuant α 스윕
│       ├── fp_inference.py              # FP baseline 추론
│       ├── main.sh                      # 전체 파이프라인 스크립트
│       ├── configs/                     # config.yaml, sq.yaml, quarot.yaml, mixed_precision.yaml
│       ├── models/                      # models.py(DiT), quant_dit.py(QuantDiT), download.py
│       └── diffusion/                   # diffusion 샘플러(원본 DiT에서 차용)
└── config.yaml, README.md
```

### 2.2 분석 제외 항목 (이름만, 분석 안 함)
- `eval/` — Vbench, RAFT, PickScore, ImageReward, align_sd 등 평가 metric 및 third_party
- `examples/opensora1.2/Open-Sora` — OpenSora 본체(외부 모델 코드)
- `kernels/` — CUDA 커널 (C++/CUDA), `viditq_extension`
- `.git`, `__pycache__`, 대용량 체크포인트, `assets/`(데모 이미지/비디오)

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.0 PTQ 전체 흐름 개요
DiT의 모든 `nn.Linear`를 양자화 Linear로 교체한 뒤, 가중치는 초기화 시 정적 양자화, 활성은 추론 시 동적 양자화한다. 메서드(SmoothQuant/QuaRot/ViDiT-Q)는 config의 정규식으로 특정 레이어에만 적용한다.

핵심 진입점 `apply_func_to_submodules`(`utils.py:15-50`)는 모듈 트리를 재귀 순회하면서, 지정 `class_type`에 매칭되는 서브모듈마다 `function`을 적용한다. `full_name`(점 표기 전체 경로), `name`, `parent_module`을 kwargs로 주입한다. 이 단일 유틸이 layer 교체, quant_param 저장/로드, init_done 설정, hook 부착 등 거의 모든 모델 변형 작업의 기반이다.

### 3.1 base_quantizer.py — 양자화의 기본 수식 (핵심)

#### BaseQuantizer (`base_quantizer.py:13-41`)
- `n_bits`, `sym`(symmetric 여부)을 config에서 읽음 (`:19-21`)
- `n_levels` 계산: 비대칭이면 `2**n_bits`, 대칭이면 `2**(n_bits-1)-1` (`:32`)
- `delta`(scale), `zero_point`을 buffer로 등록 (`:27-28`)
- `n_bits`가 list이면 단독 사용 불가, MixedPrecision 계열로 분기하라고 assert (`:23-24`)

#### StaticQuantizer (`base_quantizer.py:43-98`) — 가중치용
- `init_quant_params(x)`로 calib/weight 통계를 **오프라인 1회** 계산해 delta/zero_point 저장, `init_done=True`면 재계산 안 함 (`:65-66`)
- 입력은 반드시 2D `[N_group, -1]` (`:73`). group 차원(dim=0)마다 독립 scale → per-channel 또는 per-tensor를 입력 reshape로 표현.
- **비대칭(asymmetric) 수식** (`:79-90`):
  - `x_max = max(x, dim=1)`, 음수면 0으로 clip; `x_min = min(x, dim=1)`, 양수면 0으로 clip → 0을 항상 포함하는 affine 양자화
  - `delta = (x_max - x_min)/(n_levels-1)`
  - `zero_point = round(x_min/delta) + n_levels/2`
- **대칭(symmetric) 수식** (`:74-78`): `delta = |x|_max / n_levels`, `zero_point=0`
- 양자화: `x_int = round(x/delta) - zero_point`, clamp `[-n_levels-1, n_levels]` (`:67-68`)
- 역양자화: `(x_quant + zero_point) * delta` (`:60`)
- 다중 호출 시 max/min을 누적 업데이트(`torch.max(self.x_max, ...)`)하여 여러 배치 calib 통계를 결합 (`:83-87`)

#### DynamicQuantizer (`base_quantizer.py:100-205`) — 활성용 (핵심)
- Static과 수식은 동일하나 **delta/zero_point를 forward마다 online 계산**한다 (`quantize()` 내부에서 매번 계산: `:109-156`). 이것이 ViDiT-Q의 **"per-token dynamic activation quantization"**의 실제 구현이다.
- 입력 형태 `[N_group, -1]` (`:111`). `QuantizedLinear`가 `[B,N_token,C]`를 `[B*N_token, C]`로 reshape해 넘기므로(`quant_layer.py:65-66`), **group=각 토큰** → 토큰마다 독립 scale = **per-token quantization**.
- delta가 너무 작으면(0 근처) nan 방지를 위해 eps로 치환 (`:121-127`, `:140-147`).
- `forward_with_quant_params(x, delta, mixed_precision=None)` (`:163-205`): attention map처럼 **미리 계산된 delta**를 외부에서 받아 양자화. mixed_precision 텐서가 주어지면 원소별로 `n_levels = 2**bit - 1`을 적용하고 0-bit 원소는 마스킹(0으로) 처리 (`:174-203`). attention map의 block별/원소별 mixed precision을 구현하는 통로.

> 정리: **가중치=정적, 활성=동적**이라는 비대칭 구조가 ViDiT-Q의 기본 골격. group을 입력 reshape로 표현하므로 같은 quantizer 코드로 per-tensor/per-channel/per-token을 모두 표현한다.

### 3.2 quant_layer.py — QuantizedLinear (기본 양자화 Linear)

`QuantizedLinear(torch.nn.Linear)` (`quant_layer.py:8-76`):
- 생성자에서 원본 FP 모듈(`fp_module`)을 받음 (`:21`)
- weight quantizer 선택: `weight.n_bits`가 `ListConfig`이면 `MixedPrecisionStaticQuantizer`, 아니면 `StaticQuantizer` (`:32-36`)
- **가중치는 생성 즉시 정적 양자화** 후 `init_done=True` (`:40-41`). bias는 FP로 유지 (`:46`)
- act quantizer: list면 `MixedPrecisionDynamicQuantizer`, 아니면 `DynamicQuantizer` (`:48-52`). act는 미리 양자화하지 않음(동적).
- `forward` (`:57-76`):
  1. `quant_mode=False`면 원본 FP 모듈로 우회 (`:61-62`) — FP fallback / mixed precision의 FP16 레이어 처리에 사용
  2. `[B,N_token,C] → [B*N_token, -1]` reshape (`:65-66`)
  3. act quantizer로 활성 동적 양자화 (`:70`)
  4. `F.linear(x, self.weight, self.bias)` — **dequant된 weight·activation으로 FP 연산**(알고리즘 시뮬레이션) (`:74`)
- `use_kernel`, `quant_mode` 플래그로 실제 커널/시뮬/FP 전환 (`:54-55`)

> 즉 시뮬레이션 모드에서는 "양자화→역양자화"를 거친 값으로 FP matmul을 수행해 양자화 오차만 모사한다. 실제 INT 가속은 `quant_dit.py`의 CUDA kernel 경로에서 수행.

### 3.3 quant_attn.py — Attention Map 양자화

`QuantizedAttentionMap`(CogVideoX용, `:8-116`)와 `QuantizedAttentionMapOpenSORA`(OpenSora용, `:118-241`)는 거의 동일 구조. **softmax 직후 attention map을 양자화**한다.

- quantizer: `DynamicQuantizer` 사용 (`:20`). attention map은 항상 sym 가정(`forward_with_quant_params`에서 `assert self.sym`, `base_quantizer.py:166`).
- `group='column'`(또는 `'row'`): attention map을 전치 후 `[-1, N_token]`로 펼쳐 **행(또는 열)마다 동일 scale** → per-row/column 동적 양자화 (`:48-51`, `:168-174`).
- `group='block'`: 가장 정교한 경로 (`:52-113`, `:176-238`)
  - text token / image token 분리. **text-text self-attn, text-image cross-attn은 FP 유지**, image-image 부분만 양자화 (`:54-62`).
  - 사전 계산된 `optimal_reorder`(reorder 파일)에서 head별 chunk 수(`chunk_num_table`)를 읽어 attention map을 정사각 블록으로 `unfold` (`:71-81`)
  - 블록별 최댓값으로 delta 산출 (`:83`)
  - `int8_scale` 옵션: delta(scale) 자체를 INT8로 양자화해 저장(scale의 추가 압축) (`:86-92`)
  - `mixed_precision_cfg`가 있으면 블록별 bit-width를 적용(level-2 fine-grained block 전용) (`:95-101`)
  - 최종적으로 `forward_with_quant_params(attn_map, delta, mixed_precision)`로 양자화 (`:108-110`)

> attention map 양자화는 토큰 위치(공간/시간) 구조를 활용한 **블록 단위 mixed precision**까지 지원한다. 단 `F=13,H=30,W=45` 등 비디오 해상도가 하드코딩(`:56-59`)되어 OpenSora 특정 설정에 묶여 있음.

### 3.4 quant_model.py — 모델 레벨 refactor 로직 (핵심 디스패처)

`QuantModel`(`:182-234`)은 베이스(서브클래스에서 실제 모델 상속 구현, 예: `quant_dit.py`의 `QuantDiT`). 실제 핵심은 모듈 단위로 적용되는 함수들이다.

#### quant_layer_refactor_ (`:15-75`) — 레이어 교체 디스패처
`full_name`(레이어 경로)에 대해 config 정규식을 순서대로 매칭하여 양자화 레이어 타입을 결정:
- `smooth_quant.layer_name_regex` 매칭 → `SQQuantizedLinear` (`:21-29`)
- `quarot.layer_name_regex` 매칭 → `QuarotQuantizedLinear` (`:33-41`)
- `viditq.layer_name_regex` 매칭 → `ViDiTQuantizedLinear`(QuaRot+SmoothQuant 동시) (`:45-53`)
- `remain_fp_regex` 매칭 시 양자화 건너뛰고 FP 유지 (`:57-62`) — config에서 `t_embedder|adaLN_modulation|final_layer`를 FP로 유지(`config.yaml:6`)
- `setattr(parent_module, name, quant_layer_type(...))`로 in-place 교체 (`:68`)
- 교체 후 `module_name`을 각 레이어/quantizer에 부착 (`:70-75`)

#### bitwidth_refactor_ (`:77-106`) — layer별 mixed precision 비트 할당
- config의 `mixed_precision.weight.layer_name_regex`, `...act.layer_name_regex`는 **bit-width index별 정규식 리스트** (`:79-80`)
- 리스트 idx 0은 FP16(`quant_mode=False`로 우회), idx≥1은 `bitwidth_list[idx-1]` bit로 설정 (`:88-93`, `:101-106`)
- 즉 `mixed_precision.yaml`의 `n_bits:[2,4,8]`과 `layer_name_regex:['mlp','','','attn']`을 결합해, 정규식에 매칭되는 레이어를 해당 인덱스 비트로 강제. (예: 'mlp' 매칭=idx0=FP16, 'attn' 매칭=idx3=8bit)

#### load/save quant_param_dict (`:139-172`)
- `save_quant_param_dict_`: 각 quantizer의 delta/zero_point 저장. parent가 channel_mask(SmoothQuant/ViDiT-Q) 보유 시 함께 저장. rotation_matrix는 레이어마다 동일·대용량이라 저장 생략(`None`) (`:171-172`)
- `load_quant_param_dict_`: delta/zero_point 로드 후 **메서드별로 가중치 재구성**:
  - ViDiT-Q(channel_mask+rotation 둘 다): `get_rotation_matrix()` → channel_mask 로드 → `update_quantized_weight_rotated_and_scaled()` (`:144-148`)
  - QuaRot(rotation만): `get_rotation_matrix()` → `update_quantized_weight_rotated()` (`:149-152`)
  - SmoothQuant(channel_mask만): `update_quantized_weight_scaled()` (`:153-157`)

> rotation_matrix를 저장하지 않고 로드 시 재생성한다는 점은 재현성 측면에서 주의(아래 7장 리스크 참조).

### 3.5 mixed_precision_quantizer.py — 멀티 비트 quantizer

`MixedPrecisionBaseQuantizer`(`:15-54`): `n_bits`가 리스트(`[2,4,8]`)이고 현재 활성 인덱스 `i_bitwidth`를 가짐 (`:29-31`). `delta_list`/`zero_point_list`로 비트별 파라미터를 모두 보관 (`:36-39`). `bitwidth_refactor(i)`로 현재 비트와 delta/zp를 전환 (`:50-54`).

- `MixedPrecisionStaticQuantizer`(`:56-125`): `init_quant_params`에서 **모든 비트폭에 대해** delta/zero_point를 한꺼번에 계산해 `delta_list`/`zero_point_list`에 stack (`:81-119`). 이후 `bitwidth_refactor`로 원하는 비트만 선택 → 재계산 없이 비트 전환 가능.
- `MixedPrecisionDynamicQuantizer`(`:127-187`): 동적이므로 delta_list를 만들지 않고, forward마다 현재 `n_bits`로 online 계산 (`:140-174`). `bitwidth_refactor`는 `n_bits`만 바꿈(delta는 매번 재계산되므로) (`:181-186`).

### 3.6 smooth_quant/sq_quant_layer.py — SmoothQuant migration

`SQQuantizedLinear(QuantizedLinear)`(`:6-68`):
- `alpha = config.smooth_quant.alpha`(migration 강도) (`:24`)
- `get_channel_mask(act_mask)` (`:27-34`): 활성의 채널별 최대값(`act_mask`)과 가중치의 채널별 최대값(`weight_mask = |W|.max(dim=0)`, `[C_in]`)으로 채널 스케일 계산:
  ```
  channel_mask = |weight_mask|^alpha / |act_mask|^(1-alpha)
  ```
  (`:30`) — 이것이 SmoothQuant의 smoothing factor s.
- `update_quantized_weight_scaled()` (`:36-44`): 가중치를 `W / channel_mask`로 나눈 뒤 정적 양자화(`init_done`을 잠시 풀고 재양자화) (`:40-41`). → outlier를 활성에서 가중치로 이전.
- `forward` (`:46-68`): 활성에 `x * channel_mask`를 곱해 smoothing 적용 후 동적 양자화 (`:55-62`). 가중치엔 이미 `1/channel_mask`가 fuse돼 있어 수학적으로 등가.

> 코드의 `mlp.fc2`에만 적용(`sq.yaml:8`)되는 점으로 보아, ViDiT-Q에서 가장 양자화가 어려운 특정 레이어를 타겟팅한다. **확인된 수식**: `s = |W|^α / |X|^(1-α)`.

### 3.7 viditq/viditq_quant_layer.py — ViDiT-Q 핵심 레이어 (SmoothQuant + QuaRot 결합)

`ViDiTQuantizedLinear(QuantizedLinear)`(`:8-73`):
- `channel_mask`(SmoothQuant scale) + `rotation_matrix`(QuaRot Hadamard) **둘 다** 보유 (`:27-28`)
- `get_channel_mask`: SmoothQuant와 동일 수식 (`:30-35`)
- `get_rotation_matrix`: `random_hadamard_matrix(in_features, "cuda")` (`:37-38`)
- `update_quantized_weight_rotated_and_scaled()` (`:40-50`): **순서 = 먼저 scaling, 그 다음 rotation**
  ```
  W_scaled  = quant(W / channel_mask)
  W_rotated = quant(W_scaled @ rotation_matrix)   # double 정밀도로 회전 후 재양자화
  ```
  (`:47-48`)
- `forward` (`:52-73`): 활성도 동일 순서 — `x * channel_mask`(scale) → `x @ rotation_matrix`(rotate, double 연산) → 동적 양자화 → `F.linear` (`:62-71`)

> ViDiT-Q = SmoothQuant(channel-wise scaling) + QuaRot(Hadamard rotation)의 **합성**. 두 기법은 outlier를 (1) 가중치로 이전, (2) 회전으로 분산이라는 상보적 방식으로 처리한다.

### 3.8 quarot/quarot_quant_layer.py + quarot_utils.py — Hadamard rotation

`QuarotQuantizedLinear`(`quarot_quant_layer.py:7-53`):
- `update_quantized_weight_rotated()`: `quant(W @ rotation_matrix)` (`:30-33`)
- `forward`: 활성을 `x @ rotation_matrix`로 회전(double) 후 동적 양자화 (`:43-51`)

`quarot_utils.py` (1.9MB, 대부분 사전계산 Hadamard 상수 행렬):
- `random_hadamard_matrix(size, device)` (`:186-192`): 랜덤 부호 대각행렬(±1) `Q`를 만들고 `matmul_hadU(Q)`로 Hadamard 변환 → **randomized Hadamard 행렬**. (QuIP# 참조 주석, `:187-188`)
- `matmul_hadU(X)` (`:158-179`): 재귀적 butterfly 합/차 연산으로 빠른 Hadamard 변환, 마지막에 `/sqrt(n)`로 정규화(직교성 보존) (`:166-179`)
- `get_hadK(n)` (`:100-...`): n의 약수(172/156/144/140/108…)에 맞는 사전계산 Hadamard 행렬 선택(2의 거듭제곱이 아닌 hidden size 처리). 144는 pixart/opensora hidden 4608용 주석 (`:110-113`)
- `matmul_hadU_cuda`: `fast_hadamard_transform` 라이브러리 사용(설치 안 되면 경고만, `:87-93`) (`:195-208`)
- `apply_exact_had_to_linear`, `fuse_ln_fcs`, `rotate_pre/post_layers` 등 LayerNorm fuse 및 회전 적용 유틸 다수 포함(`:14-80`, `:215-254`) — QuaRot 원본의 전처리 루틴.

> Hadamard 회전은 직교 변환이라 출력 불변이면서 채널 outlier를 균등 분산시켜 양자화 오차를 줄인다. `hadamard_utils/hadamard_mat.pth`에서 행렬 로드(`:262`).

---

## 4. 알고리즘 / 수식 정리

### 4.1 Per-token dynamic activation quantization (확인된 수식)
입력 활성 `X ∈ R^{B×N×C}`를 `[B·N, C]`로 reshape(`quant_layer.py:65-66`). 각 토큰(행) g마다 (`base_quantizer.py:130-156`):

비대칭:
```
x_max_g = max(X_g), 0)          # 음수면 0으로 clip
x_min_g = min(X_g), 0)          # 양수면 0으로 clip
delta_g = (x_max_g - x_min_g) / (n_levels - 1)
zp_g    = round(x_min_g / delta_g) + n_levels/2
X_q_g   = clamp( round(X_g / delta_g) - zp_g , -n_levels-1, n_levels)
X_dq_g  = (X_q_g + zp_g) * delta_g
```
대칭(attention map 등):
```
delta_g = |X_g|_max / n_levels ,  zp_g = 0
```
→ delta/zp가 **토큰마다, forward마다** 산출되므로 timestep별 분포 변화에 robust. (정적 통계 의존 없음)

### 4.2 Timestep / layer별 mixed precision 비트 할당 (확인된 로직)
- **layer별**: `bitwidth_refactor_`(`quant_model.py:77-106`)이 config의 비트별 정규식 리스트로 매칭. idx0=FP16, idx k≥1 → `bitwidth_list[k-1]` bit. `mixed_precision.yaml`에서 weight/act 모두 `n_bits:[2,4,8]`, `layer_name_regex:['mlp','','','attn']`.
- **timestep별**: 알고리즘 시뮬레이션에서 활성은 매 timestep마다 동적으로 재양자화되므로(4.1) timestep-aware가 자동 성립. 또한 attention map의 `mixed_precision_cfg`(`quant_attn.py:67-101`)는 `[i_block][i_head]` 단위 비트맵을 적용. **단, "timestep마다 다른 비트폭을 명시적으로 스케줄링"하는 별도 테이블은 본 minimum 버전 코드에서 직접 확인되지 않음** → timestep-aware bit allocation의 **명시적 스케줄러는 확인 불가**(원논문 본문 기법이나 이 minimum repo에는 layer/block 단위까지만 노출된 것으로 보임. 추정).
- 0-bit 처리: `forward_with_quant_params`에서 `n_levels=2^bit-1`로 원소별 비트 적용, 0-bit는 마스킹(0) (`base_quantizer.py:174-203`).

### 4.3 SmoothQuant migration 강도 α (확인된 수식)
(`sq_quant_layer.py:30`, `viditq_quant_layer.py:33`)
```
s_c = |W_c|^α / |X_c|^(1-α)         # 채널 c별 smoothing factor (channel_mask)
W' = W / s   (열 c마다)              # 가중치로 outlier 이전
X' = X * s                          # 활성 smoothing
Y = (X*s) @ (W/s)ᵀ = X @ Wᵀ        # 수학적 등가
```
α는 활성↔가중치 사이의 양자화 난이도 분배. config 기본 `alpha=0.88`(`sq.yaml:7`). `sweep_alpha.py`는 `np.arange(0.1,0.9,0.2)`로 스윕(`:21`).

### 4.4 QuaRot Hadamard rotation (확인된 수식)
(`quarot_utils.py:158-192`, `quarot_quant_layer.py`)
```
H : randomized Hadamard 행렬 (직교, HᵀH=I)
W_rot = W @ H ,  X_rot = X @ H
Y = (X@H) @ (W@H)ᵀ = X @ (H Hᵀ) @ Wᵀ = X @ Wᵀ   # 직교성으로 출력 불변
```
회전은 채널 간 outlier를 섞어 분산을 균등화 → 양자화에 유리. butterfly 알고리즘으로 O(n log n) 변환(`matmul_hadU`, `:163-169`).

---

## 5. 학습/평가 파이프라인 (PTQ는 무학습)

ViDiT-Q는 PTQ라 gradient 학습 없음(`ptq.py:34` `torch.set_grad_enabled(False)`). 전체 흐름(`examples/dit` 기준):

### 5.1 단계
1. **Calib data 생성** — `get_calib_data.py`
   - FP DiT에 모든 `nn.Linear`에 forward hook 부착(`:92-97`)
   - hook이 입력의 **채널별 absmax** `[C]`만 저장(메모리 절약), timestep마다 누적 → `[N_timestep, C]` (`:34-38`, `:124`)
   - diffusion p_sample_loop 실행 후 `calib_data.pth` 저장(`:128`). 클래스 라벨 8개 고정(`[217,363,...]`, `:102`)
2. **PTQ** — `ptq.py`
   - `QuantDiT` 생성 시 모든 Linear → 양자화 Linear 교체 + 가중치 정적 양자화(`ptq.py:57-65`)
   - SmoothQuant: calib에서 각 레이어 채널 mask 계산(`timestep 평균`, `:91-95`)→`update_quantized_weight_scaled()`
   - QuaRot: rotation matrix 생성·적용(`:116-131`)
   - `set_init_done()` → `save_quant_param_dict()` → `quant_params.pth` 저장(`:133-135`)
3. **양자화 추론** — `quant_inference.py`
   - `QuantDiT` 생성 후 `quant_params.pth` 로드(`:86-87`)
   - mixed precision config면 `bitwidth_refactor()` 호출(`:69-71`)
   - `--hardware` 시 INT weight 저장 후 CUDA kernel forward(`:73-83`); 아니면 알고리즘 시뮬레이션
   - `--profile` 시 FP vs 양자화 e2e latency 측정(`:106-151`)
   - diffusion 샘플링 → VAE decode → 이미지 저장(`:155-162`)
4. **α 스윕** — `sweep_alpha.py`: config의 `smooth_quant.alpha`를 바꿔가며 `main.sh` 반복 실행(`:22-26`)

### 5.2 비트 설정 (config로 확인)
| config | weight | act | 메서드 |
|---|---|---|---|
| `config.yaml` | 8bit/tensor/asym | 8bit/tensor/sym | 기본 W8A8 |
| `sq.yaml` | 8bit/asym | 8bit/asym | SmoothQuant(α=0.88, `mlp.fc2`) |
| `quarot.yaml` | 8bit/asym | 8bit/asym | QuaRot(`mlp.fc2`) |
| `mixed_precision.yaml` | `[2,4,8]`,i=1 | `[2,4,8]`,i=2 | layer 정규식 기반 mixed |

- `remain_fp_regex: t_embedder|adaLN_modulation|final_layer` — 민감 레이어 FP 유지(`config.yaml:6`)
- CUDA kernel은 **W8A8 only, mixed precision 미지원**(`quant_inference.py:74`, `quant_dit.py:48` `# kernel supports W8A8 only`)

### 5.3 데이터셋
- DiT 예제: ImageNet class-conditional 생성(DiT-XL/2, 256/512, 1000 classes). (`ptq.py:148-151`)
- OpenSora/Pixart 예제 및 비디오 평가는 `examples/opensora1.2`, `eval/`(분석 제외).

### 5.4 명령어 (README/main.sh 확인)
```
# 설치
cd quant_utils && pip install -e .
# (커널) cd viditq-extension && pip install -e .

# 파이프라인
python get_calib_data.py --log ./logs/EXP --ptq-config ./configs/CFG.yaml
python ptq.py --image-size 256 --ptq-config ./configs/CFG.yaml --log ./logs/EXP
python quant_inference.py --image-size 256 --ptq-config ./configs/CFG.yaml --log ./logs/EXP [--hardware] [--profile]
```
(`README.md:1-21`, `examples/dit/main.sh:1-19`, `examples/dit/README.md`)

---

## 6. 의존성

- **핵심 런타임**: `torch, torchvision, timm, omegaconf`(`README.md:5`). qdiff 패키지 자체는 외부 의존성 명시 없음(`setup.py:6-8`).
- **OmegaConf/ListConfig**: config 로드 및 mixed precision 비트리스트 판별(`quant_layer.py:33,49`)
- **diffusers**: `AutoencoderKL`(VAE decode) (`ptq.py:14`)
- **diffusion** 서브패키지: DiT 원본 샘플러(`examples/dit/diffusion/`)
- **fast_hadamard_transform**(선택): CUDA Hadamard, 미설치 시 경고만(`quarot_utils.py:87-93`)
- **viditq_extension**(선택, CUDA kernel): `W8A8OF16LinearDynamicInputScale`, `LayerNormGeneral`, fused kernels(`quant_dit.py:31-33`) — `kernels/` 빌드 필요(분석 제외)
- **ipdb**(디버그용, `base_quantizer.py:95,144` 등)

---

## 7. 강점 / 한계 / 리스크

### 7.1 강점
- **모듈성**: qdiff가 독립 pip 패키지. 모델별로 `QuantModel` 상속 + `apply_func_to_submodules`만으로 적용 가능(`README.md:52`).
- **정규식 기반 선택적 양자화**: layer별로 메서드/비트/FP 유지를 config 정규식으로 유연 제어(`quant_model.py:15-106`).
- **diffusion 특화 동적 양자화**: per-token + per-forward 동적 scale로 timestep 분포 변화 흡수(`base_quantizer.py:130-156`).
- **상보적 outlier 처리 결합**: SmoothQuant(이전) + QuaRot(회전)를 ViDiT-Q 레이어로 합성(`viditq_quant_layer.py`).
- **시뮬↔실측 분리**: 알고리즘 시뮬(역양자화 후 FP matmul)과 실제 INT8 CUDA kernel 경로를 모두 제공, 정확도/속도 모두 검증 가능(`quant_dit.py:315-346`).

### 7.2 한계
- **CUDA kernel은 W8A8 전용**: mixed precision/W4A8은 알고리즘 시뮬에서만, 실제 가속 미지원(`quant_inference.py:74`).
- **attention map block 경로의 하드코딩**: `F=13,H=30,W=45`, 텍스트 토큰 분리 등 OpenSora/CogVideoX 특정 설정에 결합(`quant_attn.py:56-59`).
- **`module_name` 전제 위반 가능**: DynamicQuantizer가 `self.module_name`를 로깅에 사용하나(`base_quantizer.py:127`) 직접 인스턴스화 시 미설정 가능(quant_model.py:74에서만 부착).
- **rotation_matrix 미저장**: 저장 시 None, 로드 시 `random_hadamard_matrix` 재생성(`quant_model.py:172`, `:146,151`). 재생성이 학습/검증 시점과 다른 랜덤이면 결과 불일치 위험(아래 리스크).
- 코드 곳곳 `import ipdb; ipdb.set_trace()`가 예외 핸들러에 남아 있어(`base_quantizer.py:95,144`; `mixed_precision_quantizer.py:163`) 프로덕션에서 hang 가능.

### 7.3 리스크
- **랜덤 Hadamard 재현성**: `random_hadamard_matrix`는 `torch.randint`로 매번 새 ±1 대각을 생성(`quarot_utils.py:189`). seed 고정(`seed_everything`)으로 동일성을 보장하려는 의도로 보이나, save 시 rotation을 None으로 둔 채 load에서 재생성하므로 **PTQ와 추론의 호출 순서/난수 소비가 다르면 행렬이 달라질 수 있음** → 정확도 저하 위험(확인 필요. **추정**).
- **per-token 동적 양자화 비용**: forward마다 max/min/round를 토큰별로 계산 → 시뮬 오버헤드 큼(`base_quantizer.py:130-156`). FPGA/엣지로 옮길 때 online scale 계산 비용 고려 필요.
- **double 정밀도 회전**: 활성/가중치 회전을 `.double()`로 수행(`viditq_quant_layer.py:48,63`) → 정확하지만 무겁다.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적)

> 우리 프로젝트 추정: HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적. diffusion transformer 자체는 무겁지만, ViDiT-Q의 **양자화 기법은 ViT로 전이 가능**.

### 8.1 전이 가능한 기법
1. **Per-token dynamic activation quantization → ViT 토큰 양자화**: ViT도 토큰별 활성 분포 차이가 큼. 단 FPGA에서는 online max/min 계산이 비용. **시선추적용 ViT처럼 입력 분포가 비교적 안정적이면 정적(per-tensor/per-channel) calib로 대체**해 하드웨어 단순화 가능(diffusion만큼 timestep 변동이 없으므로). (`base_quantizer.py` Static vs Dynamic 비교가 좋은 참고)
2. **SmoothQuant migration (s=|W|^α/|X|^(1-α))**: 가중치는 정적 양자화이므로 outlier 이전을 **오프라인에 fuse**하면 추론 시 추가 연산 없음(`sq_quant_layer.py:36-44`). FPGA 친화적 — 활성 스케일을 미리 가중치에 흡수해 INT MAC만 수행. **권장 1순위 전이 기법.**
3. **QuaRot Hadamard rotation**: 직교 변환이라 정확도 이득은 크나, FPGA에서는 추론 시 `x@H` 회전(O(n log n) butterfly)이 추가 데이터패스 필요(`matmul_hadU`). butterfly는 FPGA에 매핑하기 좋은 구조이지만 latency/area 트레이드오프 평가 필요. **시선추적의 실시간 제약 하에선 비용-효익 검토 대상.**
4. **Mixed precision layer 할당(정규식 기반)**: 레이어별 민감도에 따라 비트 차등(`mixed_precision.yaml`, `quant_model.py:77-106`). FPGA에서 **레이어별 PE 비트폭/리소스 차등 설계**의 직접적 참고: 민감 레이어(embedder, 출력층)는 고비트/FP, 내부 attention/mlp는 저비트. `remain_fp_regex`로 FP 유지하는 패턴이 곧 "FPGA에서 어떤 레이어를 FP/고정밀 블록으로 둘지" 설계 지침.

### 8.2 FPGA 비트할당 설계 참고점 (구체)
- **W8A8을 안전 baseline으로**: kernel도 W8A8만 지원(`quant_inference.py:74`)할 만큼 검증됨. FPGA INT8 MAC 어레이의 1차 타겟.
- **W4A8 mixed precision**: weight 4bit + activation 8bit. FPGA에서 weight 메모리(BRAM/URAM) 절반, activation 데이터패스는 8bit 유지 → **메모리 대역폭 절감 + 정확도 유지**의 균형점. HG-PIPE 같은 파이프라인 가속기의 weight 버퍼 설계에 유리.
- **민감 레이어 FP 유지 패턴**: embedding/정규화 조건부/출력층은 양자화 제외(`config.yaml:6`). FPGA에선 해당 레이어를 별도 FP16/고정밀 경로로 분리.
- **scale의 추가 양자화(int8_scale)**: attention map의 scale 자체를 INT8로 저장(`quant_attn.py:86-92`) → FPGA에서 scale 저장 비용까지 줄이는 아이디어로 응용 가능.

### 8.3 XR 시선추적 관점 (추정)
- 시선추적은 **저지연·저전력**이 핵심. ViDiT-Q의 동적 양자화는 정확하나 online scale 계산 비용 → **정적 양자화 + SmoothQuant(오프라인 fuse)** 조합이 XR 엣지에 더 적합할 것(추정).
- mixed precision으로 시선추적 ViT의 비핵심 레이어를 저비트화하면 FPGA 리소스/전력 절감 → 프레임률 향상 기대(추정).
- **주의**: ViDiT-Q는 생성(품질) 태스크 기준 검증. 시선추적의 회귀/분류 정확도(시선각 오차 등)에 동일 비트 설정이 안전한지는 **별도 검증 필요(확인 불가)**.

---

## 9. 근거표기 규칙 및 확인 상태

### 9.1 코드로 직접 확인한 사실 (file:line 근거 보유)
- 가중치=정적/활성=동적 양자화 구조 (`quant_layer.py:40-41,57-76`, `base_quantizer.py:43-205`)
- per-token 동적 양자화(입력 reshape + 행별 scale) (`quant_layer.py:65-66`, `base_quantizer.py:130-156`)
- 비대칭/대칭 양자화 수식, clamp 범위, zero_point 계산 (`base_quantizer.py:67-90`)
- SmoothQuant 수식 `s=|W|^α/|X|^(1-α)`, fuse 방식 (`sq_quant_layer.py:30,36-68`)
- QuaRot randomized Hadamard 생성/적용 (`quarot_utils.py:158-208`, `quarot_quant_layer.py:30-51`)
- ViDiT-Q = scaling→rotation 결합 (`viditq_quant_layer.py:40-73`)
- mixed precision layer 정규식 비트 할당, FP16 우회 (`quant_model.py:77-106`, `mixed_precision_quantizer.py`)
- attention map column/block 양자화, text/image 분리, int8_scale (`quant_attn.py`)
- PTQ/calib/추론 파이프라인, 명령어, 비트 config (`ptq.py`, `get_calib_data.py`, `quant_inference.py`, `configs/*.yaml`, `main.sh`)
- CUDA kernel W8A8 전용 제약 (`quant_inference.py:74`, `quant_dit.py:48`)
- W8A8 무손실 / W4A8 mixed precision 목표 (`README.md:23`)

### 9.2 추정 (코드로 직접 검증 못 함, 합리적 추론)
- **Timestep마다 비트폭을 바꾸는 명시적 스케줄러**: minimum 버전 코드에선 layer/attention-block 단위까지만 확인. timestep-aware는 동적 양자화로 "자동" 성립하나, 원논문에서 강조하는 timestep별 명시적 비트 스케줄 테이블은 이 repo에서 직접 확인 불가.
- 랜덤 Hadamard 재현성 위험(저장 생략+재생성)의 실제 영향도.
- XR 시선추적에의 적합성/정확도 영향 — 우리 프로젝트 정의 자체가 추정.

### 9.3 확인 불가
- 원논문 본문의 정량 수치(FID/CLIPScore 등 metric 표) — `eval/`(분석 제외)·논문 PDF 미확인.
- `viditq_extension` CUDA 커널 내부 구현(`kernels/`, 분석 제외).
- `optimal_reorder`/reorder 파일 생성 절차(파일 경로만 참조, 생성 스크립트 미확인).

---

*분석 파일 경로*: `REF/Analysis/ViT-Quantization/ViDiT-Q.md`
