# Q-VDiT 코드베이스 정밀 분석

> 대상: `REF/ViT-Quantization/Q-VDiT`
> 분석 방식: 실제 소스(Read) 기반. 모든 핵심 근거는 `파일:라인`으로 표기.
> 근거 표기 규칙: 코드로 확인된 사실은 단정, 코드 외 추론은 "추정", 미확인은 "확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **정식 명칭**: *Q-VDiT: Towards Accurate Quantization and Distillation of Video-Generation Diffusion Transformers* (ICML 2025, arXiv:2505.22167) (`README.md:1~3`, `:173~178`).
- **목적**: **비디오 생성용 Diffusion Transformer**(OpenSora v1.0의 **STDiT** = Spatio-Temporal DiT)를 **W4A8/W6A6/W3A8** 등 저비트로 PTQ하면서, 양자화 + **증류(distillation)** 를 결합해 품질을 복원.
- **핵심 아이디어 (코드 확인)**:
  1. **Timestep-aware quantization**: diffusion timestep(0~1000)에 따라 act 양자화 파라미터/smooth-quant α를 구간별로 다르게 적용 (`base_quantizer.py:38`, `:59~62`; `quant_layer.py:90~99`, `:118`).
  2. **Token-aware / per-token dynamic act quantization**: 활성을 토큰축 그룹으로 동적 양자화(클리핑 에러 0) (`dynamic_quantizer.py:11~45`, configs `per_group: 'token'`, `dynamic: True`).
  3. **Learnable codebook(LoRA) 보정 + 증류**: 각 QuantLayer에 LoRA형 보정항(loraA/B, loraA_out/B_out)을 두고, **block reconstruction(BRECQ류)** 으로 weight LoRA + act delta를 학습 (`quant_layer.py:55~64`, `:182~197`; `block_recon.py:178~188`).
  4. **Smooth-quant (timestep-범위별 채널 스케일)**: 활성-가중치 outlier를 채널 스케일로 흡수 (`quant_layer.py:116~148`).
- 코드베이스는 **open-sora v1.0 + ViDiT-Q** 기반 (`README.md:166`). 양자화 엔진은 `qdiff/`(Q-Diffusion/BRECQ 계열).

---

## 2. 디렉토리 구조 (자체 소스 / 제외)

```
Q-VDiT/
├── qdiff/                                  # ★자체 양자화 엔진
│   ├── quantizer/
│   │   ├── base_quantizer.py               # ★Base/Weight/Act Quantizer, rounding, lp_loss
│   │   └── dynamic_quantizer.py            # ★DynamicActQuantizer (online 동적 양자화)
│   ├── models/
│   │   ├── quant_layer.py                  # ★QuantLayer (LoRA 보정 + smooth_quant)
│   │   ├── quant_block.py                  # QuantTransformerBlock/Attention (diffusers)
│   │   ├── quant_model.py                  # ★QuantModel (refactor, timestep set, mixed-prec)
│   │   ├── stdit_quant_layer.py            # ★STDiT Spatial/Temporal/CrossAttn Linear
│   │   ├── dit_quant_layer.py              # PixArt(DiT) AttnLinearImg/CrossAttnLinearImg
│   │   ├── quant_layer_pixart.py           # PixArt 변형 (확인 불가: 본문 미열람)
│   │   └── stdit_quant_layer_pixart.py     # PixArt-STDiT 변형 (확인 불가)
│   ├── optimization/
│   │   ├── block_recon.py                  # ★블록 재구성 (BRECQ류, LoRA+delta opt)
│   │   ├── layer_recon.py                  # 레이어 재구성
│   │   └── model_recon.py                  # 모델 단위 래퍼
│   └── utils.py                            # ★LossFunction, save_in_out_data, calib data
├── t2v/configs/quant/opensora/            # ★비트 설정 yaml (W4A8/W6A6/W8A8 등)
│   ├── w4a8_ours.yaml / w6a6_ours.yaml / w8a8_ours.yaml
│   ├── w4a8_timestep_aware_cb.yaml / w4a8_naive_cb.yaml
│   ├── w8a8_dynamic.yaml / viditq_w*.yaml (비교군)
│   └── mixed_precision/{weight,act}_*_mp.yaml  # 레이어별 비트맵
├── t2v/configs/{opensora,latte}/...        # 백본 inference/train 설정
├── README.md / LICENSE
└── assets/forest/*.gif                     # 데모(이름만, 대용량 제외)
```

**제외**: `.git/`, `__pycache__/`, `assets/*.gif`(대용량 데모), `t2v/configs/latte`(외부 백본 설정), Open-Sora 본체(별도 설치, repo 외부).
**미열람(확인 불가)**: `quant_layer_pixart.py`, `stdit_quant_layer_pixart.py`, `t2v/scripts/*`(calib.py 등, repo Glob 외).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `qdiff/quantizer/base_quantizer.py` — 양자화기 기반

**`BaseQuantizer(nn.Module)`** (`:13~369`): asymmetric uniform affine 양자화 + AdaRound(learned rounding) 지원.

- **설정 언팩** (`:25~81`): `n_bits`, `mixed_precision`(비트 리스트), `timestep_wise`(1000구간), `per_group`(`channel`/`token`/False), `channel_dim`, `scale_method`(`min_max`/`grid_search_lp`), `round_mode`, `sym`, `running_stat`(EMA), `always_zero`(softmax용).
- **`n_levels`** (`:51~52`): `2^bits` (asym) / `2^(bits-1)-1` (sym).
- **AdaRound 파라미터** (`:75~81`): `round_mode='learned_hard_sigmoid'`면 `alpha`(학습형 rounding), `gamma=-0.1, zeta=1.1, beta=2/3`.
- **`rounding(x)`** (`:83~116`): 5가지 round 모드.
  - `nearest`/`nearest_ste`(STE)/`stochastic`/`learned_hard_sigmoid`.
  - **AdaRound**: `soft_targets = clamp(sigmoid(alpha)*(ζ-γ)+γ, 0,1)`, `x_int = floor(x/δ)+soft_targets` (`:95~105`).
  - clamp: sym `[-n-1, n]`, asym `[0, n_levels-1]` with `+zero_point` (`:111~115`).
- **`forward(x)`** (`:122~166`): 미초기화면 `init_quant_params`로 δ/zp 추정 → `rounding` → dequant `(q - zp)·δ` (sym은 `q·δ`).
- **`init_quant_params(x, per_group, ...)`** (`:168~325`): scale/zp 추정 핵심.
  - per_group `channel`(채널축 reshape), `token`(act 전용, `[n_token, BS*C]`로 reshape) (`:190~207`).
  - x_min/x_max 산출, running_stat면 momentum(0.95) EMA (`:213~229`).
  - **`min_max`**: asym `δ=(max-min)/(n-1)`, `zp=round(-min/δ)`; sym `δ=absmax/n_levels` (`:235~250`).
  - **`grid_search_lp`**: scale을 0~1 그리드(step 0.02)로 스윕하며 `lp_loss(L2)` 최소화하는 δ/zp 탐색 (`:252~281`) — outlier 강건한 weight 양자화.
  - AdaRound alpha 초기화: `α = -log((ζ-γ)/(rest-γ) - 1)` → `sigmoid(α)=rest` (`:287~296`).
- **`quantize(x, x_max, x_min, n_bits)`** (`:327~352`): 주어진 max/min으로 즉석 fake-quant (grid search 내부용).
- **`bitwidth_refactor(bit)`** (`:355~364`): mixed-precision 비트 변경.
- **`WeightQuantizer`/`ActQuantizer`** (`:371~389`): BaseQuantizer 상속(동작 동일).
- **유틸**: `round_ste`(`:400~404`), **`lp_loss(pred, tgt, p)`** (`:406~438`) — 재구성 손실의 L_p norm.

### 3.2 `qdiff/quantizer/dynamic_quantizer.py` — 동적 활성 양자화

**`DynamicActQuantizer(ActQuantizer)`** (`:11~45`): **매 forward마다 입력으로부터 δ/zp를 온라인 계산**(클리핑 에러 없음).
- `init_done=True`, `running_stat=False` 강제 (`:17~18`).
- `init_quant_params`를 forward 내부에서 호출(per-token min-max) → `x_int = round_ste(x/δ)+zp → clamp → (q-zp)·δ` (`:22~42`).
- text_embed 등 가변 shape 입력 대응 위해 delta_list 캐시 무효화 (`:28~29`).

### 3.3 `qdiff/models/quant_layer.py` — ★QuantLayer (LoRA codebook + smooth quant)

**`QuantLayer(nn.Module)`** (`:24~232`): Conv2d/Conv1d/Linear을 양자화 레이어로 래핑.

- **LoRA 보정항(=learnable codebook)** (`:55~64`):
  - `loraA: Linear(in, r=32)`, `loraB: Linear(r, out)` → weight 보정용 저랭크 항.
  - `loraA_out: Linear(in, r_out=1)`, `loraB_out: Linear(1, out)` → 추가 출력 보정.
  - loraB/loraB_out은 **0 초기화**(초기엔 항등) (`:62~64`).
- 양자화기 선택 (`:73~79`): weight는 `WeightQuantizer`, act는 `dynamic` 플래그면 `DynamicActQuantizer` else `ActQuantizer`.
- **smooth_quant 설정** (`:87~105`): timerange(구간), `channel_wise_scale_type`(`dynamic`/`momentum_act_max`), momentum, alpha(구간별 리스트 가능). act_scale 버퍼 등록.
- **`forward(input, scale, split, smooth_quant_enable)`** (`:107~219`):
  - smooth_quant 시 현재 timestep으로 timerange 선택 → α 선택 → **채널 스케일** `s = act_max^α / weight_max^(1-α)` (dynamic) 또는 momentum act_scale 기반 (`:118~146`). `input = input / s` (`:148`).
  - act_quant on이면 `act_quantizer(input)` (split이면 2분할 양자화) (`:166~174`).
  - **weight_quant on**: 항등행렬로 LoRA 가중치 추출 `lora_weight = (loraB(loraA(E)))ᵀ`, `lora_weight_out` 동일 (`:182~187`).
    - smooth_quant면 timestep-wise weight quantizer 활성 → `weight = weight_quantizer(W·s + lora_weight) + lora_weight_out` (`:188~197`).
    - 즉 **양자화 대상에 LoRA 보정을 합산** → 양자화 오차를 학습형 codebook으로 흡수.
  - `F.linear/conv`로 출력 (`:211`).
- `set_quant_state(weight_quant, act_quant)` (`:221~223`), `set_split` (`:228~232`).

### 3.4 `qdiff/models/stdit_quant_layer.py` — ★STDiT(OpenSora) 어텐션 Linear

OpenSora STDiT의 **공간/시간/교차 어텐션**별 Linear을 토큰 reshape에 맞춰 양자화.

- **`QuantSpatialAttnLinear`** (`:14~124`): 입력 `[BS·T, S, C]` → 양자화 시 `[BS, T·S, C]`로 reshape 후 per-token 양자화, 다시 복원 (`:25~29`, `:76~82`). LoRA 보정 동일.
- **`QuantTemporalAttnLinear`** (`:126~237`): 시간축 어텐션. `mask = Parameter(ones[1,T,1])` 보유 (`:130~131`).
  - act 양자화 시 `[BS, S·T, C]`로 reshape (`:186~191`).
  - **출력에 LoRA-out를 마스크 가중 합산**: `out += (input @ lora_weight_outᵀ)·mask` (`:227~230`) → 시간 프레임별 보정 강도 학습. **Q-VDiT의 temporal 보정 핵심**.
- **`QuantCrossAttnLinear`** (`:239~370`): Q(`[BS, T·S, C]`)/KV(`[1, BS·n_prompt, C]`) shape 분기 (`:254~262`). KV는 tensor-wise/per-group/dynamic에 따라 reshape 처리 (`:316~328`). prompt 토큰(n_prompt=120) 단위 양자화.
- 세 클래스 모두 smooth_quant(timerange α) + LoRA 보정 구조를 공유.

### 3.5 `qdiff/models/dit_quant_layer.py` — PixArt(이미지 DiT)

- **`QuantAttnLinearImg`** (`:9~32`): 단순 self-attn Linear (토큰 reshape 불필요, 이미지 단일프레임).
- **`QuantCrossAttnLinearImg`** (`:34~79`): Q/KV 분기 + prompt reshape. STDiT 대비 시간축 없음.

### 3.6 `qdiff/models/quant_block.py` — diffusers용 Quant 블록

- `BaseQuantBlock` (`:41~64`): 블록 내 QuantLayer에 quant_state 전파.
- `QuantResnetBlock2D` (`:66~177`), `QuantTransformerBlock` (`:181~403`): diffusers `BasicTransformerBlock`을 래핑, **attn QKV/softmax용 ActQuantizer 부착**(`:221~236`), `QuantAttnProcessor`로 attention forward 교체 (`:242~249`).
- `QuantAttnProcessor.__call__` (`:543~650`): Q/K/V projection → `head_to_batch_dim` → `get_attention_scores`(softmax) → `bmm(attn, V)` → out proj. (현재 QK/softmax act_quant은 디버그상 비활성 주석 처리 `:617~632`.)
- `get_specials(model_type)` (`:653~673`): opensora/pixart는 특수블록 없음(레이어 단위 양자화), diffusers만 블록 치환.
- **주의**: OpenSora 경로는 `quant_block`이 아닌 **레이어 단위(quant_model.quant_layer_refactor)** 로 양자화 (3.7 참조).

### 3.7 `qdiff/models/quant_model.py` — ★QuantModel

**`QuantModel(nn.Module)`** (`:38~629`): FP 모델을 받아 양자화 모델로 변환·제어.

- **`quant_layer_refactor`** (`:63~103`): 재귀적으로 Linear/Conv를 양자화 레이어로 치환. **이름 기반 분기**(`:78~97`):
  - `.attn.` → `QuantSpatialAttnLinear`(opensora)/`QuantAttnLinearImg`(pixart).
  - `cross_attn` → `QuantCrossAttnLinear`/`QuantCrossAttnLinearImg`.
  - `attn_temp` → `QuantTemporalAttnLinear`.
  - 그 외 → `QuantLayer`.
- **timestep 제어** (`:160~199`):
  - `set_timestep_for_quantizer(t)` (`:160~168`): 모든 quantizer에 `cur_timestep_id` 설정.
  - `set_timestep_id_for_quantlayer(t)` (`:171~183`): QuantLayer에 timestep 설정(smooth α 선택용).
  - `repeat_timestep_wise_quant_params(ts)` (`:186~199`): timestep-wise δ/zp를 1000 구간으로 복제.
- **mixed-precision** (`:529~622`): `set_layer_bit`/`load_bitwidth_config`로 레이어별 비트 적용(mp yaml 로드).
- `set_quant_state` (`:130~141`): fp_layer_list는 FP 유지.
- `forward(x, t, y)` (`:371~394`): timestep_wise면 timestep 설정 후 STDiT 백본 호출.
- LoRA/quant 파라미터 직렬화: `get_quant_params_dict`/`set_quant_params_dict` (`:222~303`) — loraA/B/out, mask, δ/zp 버퍼 저장·로드.

### 3.8 `qdiff/optimization/block_recon.py` — ★블록 재구성 (BRECQ류)

**`block_reconstruction(model, block, calib_data, config, param_types, opt_target)`** (`:35~372`):
- calib 입출력 캐시: `save_in_out_data`로 **FP 출력(target)과 양자화 입력** 저장 (`:65~73`).
- **최적화 대상 파라미터 그룹** (`:98~190`):
  - weight δ/alpha, act δ 등 (param_type별 독립 lr).
  - **LoRA 파라미터**: `('lora' in name and 'minus' not in name)`, temporal은 `mask` 포함 (`:178~181`).
- optimizer AdamW + CosineAnnealing (`:194~203`).
- **손실**: `LossFunction`(재구성 + AdaRound round_loss) (`:206~218`).
- grad checkpoint로 27개 블록 메모리 절감 (`:247~248`).
- 반복 루프 (`:250~357`): 캐시 입력으로 양자화 블록 forward → `loss_func(out_quant, cur_out)` → backward → step. 입력 tuple 길이(2/3/5)별 STDiT 다중입력 처리 (`:310~327`).
- 종료 시 AdaRound `soft_targets=False`(hard rounding 고정) (`:363~366`).
- `requires_grad`는 lora/delta/mask만 True (`:237~242`) → **양자화 step + 학습형 codebook만 최적화**(원 weight 동결).

### 3.9 `qdiff/optimization/layer_recon.py` — 레이어 재구성
- `layer_reconstruction` (`:15~204`): block_recon의 단일 레이어 버전. weight/act/joint 그룹별 δ·alpha 최적화 (`:83~128`), Adam, lp_loss 재구성 (`:130~151`). 종료 시 hard rounding (`:200~202`).

### 3.10 `qdiff/optimization/model_recon.py`
- `our_model_reconstruction` (`:17~27`): 단순히 `block_reconstruction` 호출 래퍼(모델 전체를 블록 단위로 재구성하기 위한 진입점).

### 3.11 `qdiff/utils.py` — 손실/캘리브레이션
- **`LossFunction`** (`:127~224`): `total = reconstruction_loss + round_loss`.
  - reconstruction: `mse`(lp_loss), **`relation`**(lp + `get_time_relation_loss`×100), `fisher_diag/full` (`:174~192`).
  - **`get_time_relation_loss(pred, tgt)`** (`:100~123`): 프레임축 정규화 후 **frame-frame 유사도 맵의 KL divergence** → **시간 일관성(temporal relation) 증류** (Q-VDiT 비디오 특화 손실).
  - round_loss(AdaRound): `λ·Σ(1 - |2·soft_target-1|^b)` (`:200~211`), b는 LinearTempDecay로 annealing (`:227~244`).
- **`get_quant_calib_data`** (`:17~63`): diffusion sampling 궤적에서 timestep subsample하여 calib 데이터(xs/ts/cond/mask) 구성 — **timestep별 활성 분포 수집**.
- **`save_in_out_data` / `GetLayerInOut`** (`:268~663`): hook으로 레이어 입출력 저장. **`previous_layer_quantized=True`**: 이전 레이어는 양자화, 대상 레이어 출력은 FP target — BRECQ식 누적 양자화 재구성 (`:606~618`).

---

## 4. 알고리즘 / 수식

### 4.1 양자화 (asymmetric uniform, fake-quant)
```
δ = (x_max - x_min)/(2^b - 1),   zp = round(-x_min/δ)
q = clamp(round(x/δ) + zp, 0, 2^b-1)
x̂ = (q - zp)·δ
```
(`base_quantizer.py:235~250`, `:122~166`). sym은 `δ=absmax/(2^(b-1)-1)`, zp=0.

### 4.2 Timestep-aware smooth quant (per-range channel scale)
timestep t가 속한 구간 r에 대해:
```
s_c = (act_max_c)^{α_r} / (|W|_max,c)^{1-α_r}
x' = x / s_c ,   W' = W · s_c   (수학적 등가, outlier 이동)
```
- α_r는 구간별 값(예: timerange `[[0,500],[501,1000]]`, α `[0.11,0.11]`) (`quant_layer.py:124~148`, `w4a8_timestep_aware_cb.yaml:34~39`).
- `channel_wise_scale_type=momentum_act_max`면 act_max를 momentum(0.95) EMA로 추정 (`:126~144`).

### 4.3 Learnable codebook (LoRA) 보정 + 양자화
```
W_lora   = (loraB · loraA)(I)ᵀ              # rank=32 weight 보정
W_lora_out = (loraB_out · loraA_out)(I)ᵀ    # rank=1 출력 보정
W_q = Quant(W·s_c + W_lora) + W_lora_out     # 양자화 + 보정
```
(`quant_layer.py:182~197`). temporal layer는 출력에 `out += (x·W_lora_outᵀ)·mask` 추가 (`stdit_quant_layer.py:227~230`).
> "codebook(cb)" 명칭은 yaml 파일명(`*_cb.yaml`)과 LoRA 보정항을 지칭하는 것으로 **추정**(코드에 'codebook' 식별자 직접 없음; LoRA 보정이 양자화 격자 보정 역할 = 사실상 learnable codebook).

### 4.4 Block reconstruction (BRECQ류) 목적함수
```
min_{δ, α(AdaRound), W_lora, mask}  ‖ Block_q(x_q) - Block_fp(x_fp) ‖_p^p
                                     + λ·Σ(1 - |2·sigmoid(α)-1|^b)        # round_loss
                                     + 100·KL(simmap_q ‖ simmap_fp)        # relation loss (옵션)
```
- 원 weight는 동결, **양자화 step·learnable rounding·LoRA codebook·temporal mask만 학습** (`block_recon.py:237~242`, `utils.py:174~211`).
- `relation` 손실로 프레임 간 시간 일관성 보존 (`utils.py:100~123`, `w4a8_ours.yaml:22`).

### 4.5 비트 설정 (W4A8 등)
- **W4A8 (ours)** (`w4a8_ours.yaml`): weight n_bits=4, per-channel, `grad_search_lp`, nearest_ste; act n_bits=8, per-token, **dynamic=True**, min_max, smooth_quant enable α=0.11; weight 최적화 iters=10000(`relation` loss), act iters=100 (`:13~64`).
- **W8A8 dynamic** (`w8a8_dynamic.yaml`): mixed_precision `[4,6,8]`, weight learned_hard_sigmoid(AdaRound), act per-token dynamic (`:8~41`).
- **W4A8 timestep_aware_cb** (`w4a8_timestep_aware_cb.yaml`): smooth_quant timerange 2구간, α `[0.11,0.11]` (`:34~39`).
- **mixed precision** (`weight_4_mp.yaml`): attn/cross_attn/temp Linear=4bit, **mlp.fc1/fc2=8bit** (계산량 큰 MLP는 8bit 유지) (`:1~12` 패턴, 전 28블록).
- 비교군: `viditq_w{3,4,6}a*.yaml`, `*_naive_cb.yaml`(grid search 대신 min_max, smooth α만 다름).

---

## 5. 학습 / 평가 파이프라인 (README 기준)

데이터셋: OpenSora 예제 프롬프트 10개 + 사전계산 text_embeds(`README.md:74`). calib data는 FP 추론 궤적에서 활성 저장.

1. **체크포인트 변환** (`split_ckpt.py`): OpenSora QKV merged linear을 3개로 분리(양자화용) (`README.md:62~67`).
2. **FP16 inference** (`fp16_inference.sh`): 기준 영상 생성 (`:71~84`).
3. **calib data 생성** (`get_calib_data.py`): 활성 저장 → `calib_data.pt` (`:88~101`).
4. **Calibration(양자화 + 재구성)** (`calib.py`): Q_CFG(yaml) + MP weight/act config + `--part_fp`(소수 레이어 FP 유지) (`:103~131`).
5. **Quantized inference** (`quant_txt2video.py`): 양자화 ckpt로 영상 생성 (`:133~156`).

비트 yaml은 4장 참조. shell 스크립트는 `t2v/shell_scripts/`(repo Glob 외, **확인 불가**).

---

## 6. 의존성 (`qdiff/requirements.txt`)
- `diffusers==0.24.0`(attention/block 래핑 핵심, `quant_block.py` import), `transformers==4.32.1`, `einops==0.3.0`, `omegaconf`(config), `torchmetrics`, `torch-fidelity`(평가), `kornia`, `pytorch_lightning==1.4.2`.
- README상 `torch==2.1.1`, `xformers==0.0.23`, `flash-attn`(opensora 요구), Python 3.10 (`README.md:24~44`).
- Open-Sora v1.0 본체 별도 설치(`pip install -e t2v`).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **비디오 DiT 특화**: spatial/temporal/cross attention을 토큰 레이아웃에 맞춰 정밀 양자화(`stdit_quant_layer.py`), temporal mask + relation loss로 **시간 일관성 보존**.
- **양자화 + 증류 통합**: AdaRound(learned rounding) + LoRA codebook + smooth-quant + block reconstruction을 한 파이프라인에 결합 → W4A8에서 품질 복원.
- **dynamic per-token act quant**: 클리핑 에러 0, 토큰별 분포 적응 (`dynamic_quantizer.py`).
- mixed-precision(MLP 8bit, attn 4bit)과 fp_layer 보호로 정확도-효율 균형.

**한계 / 리스크**
- **연산 무게**: STDiT 28블록, weight recon 10000 iter, calib 데이터/궤적 필요 → PTQ 비용 큼(데이터-free 아님, calib 필수).
- **dynamic act + per-token + smooth scale**은 추론 시 **온라인 scale 계산** 필요 → 정수 전용 하드웨어에 직접 매핑 어려움(추정).
- 코드에 디버그(`import ipdb`), 주석 처리, `bak/`, PixArt 미열람 파일 다수 → 성숙도/가독성 낮음.
- softmax/QK act 양자화는 현재 비활성(주석) (`quant_block.py:617~632`) → 완전 정수 추론 아님.
- 평가 지표/수치는 이미지(`imgs/result.png`)로만 제시, 코드상 정량 미확인.

---

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 관점 시사점

> 아래는 프로젝트 맥락 **추정**. 코드 사실과 분리.

- **Diffusion 본체는 우리 타깃과 무관**(비디오 생성, 무겁고 multi-step). XR 시선추적은 경량 단일패스 ViT가 적합 → Q-VDiT 모델 자체 이식은 비현실적(추정).
- **그러나 양자화 기법 자체는 ViT로 전이 가능**:
  - **per-channel weight + grid_search_lp(L2 최소 scale)** (`base_quantizer.py:252~281`)는 outlier 강건 → FPGA용 ViT weight 양자화 scale 결정에 직접 차용 가능.
  - **AdaRound(learned_hard_sigmoid) + block reconstruction(BRECQ)** (`base_quantizer.py:95~105`, `block_recon.py`)는 캘리브레이션 데이터만으로 W4 정확도 복원 → HG-PIPE 정수 데이터패스용 weight 준비 단계에 적용 가능(추정).
  - **smooth-quant 채널 스케일**(`quant_layer.py:124~148`)은 W에 흡수(`W·s`)되면 추론 시 추가 연산 없음 → **오프라인으로 fold하면 FPGA 친화적**. 단, dynamic/per-token act scale은 온라인 계산이라 HG-PIPE식 고정소수점 파이프라인에는 부적합 → **static per-tensor act 양자화로 단순화 필요**(추정).
- **mixed-precision 정책**(attn 4bit, MLP 8bit) (`weight_4_mp.yaml`)은 FPGA에서 레이어별 PE 비트폭 차등 설계 시 참고 가능.
- **LoRA codebook 보정**은 추론 시 weight에 fold 가능(`W_q = Quant(W)+W_lora`) → 양자화 정확도 복원을 추론 비용 없이 얻는 기법으로 ViT 가속기에 유용(추정).
- **주의**: softmax/LayerNorm 정수화는 본 repo가 다루지 않음(확인됨) → HG-PIPE 완전 양자화 파이프라인엔 별도 정수 비선형 모듈 필요.

---

## 9. 근거·불확실성 표기 요약
- 단정 서술은 모두 `파일:라인` 근거 보유.
- `quant_layer_pixart.py`, `stdit_quant_layer_pixart.py`, `t2v/scripts/*`, `t2v/shell_scripts/*` 미열람 → 관련 동작은 **확인 불가** 또는 README 기준.
- "codebook" 용어 해석(LoRA 보정=codebook)은 yaml 파일명+코드 구조 기반 **추정**.
- 8장 시사점은 프로젝트 맥락 **추정**이며 코드 직접 근거 없음.
