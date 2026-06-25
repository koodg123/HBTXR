# MBQ 코드베이스 정밀 분석

> 대상 경로: `REF/ViT-Quantization/MBQ`
> 분석 도구: Glob/Grep/Read (bash 미사용)
> 근거 표기: `[코드]` = 실제 코드 라인으로 확인 / `[추정]` = 코드 정황상 추정 / `[확인불가]` = 저장소 내 근거 없음

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **목적**: 대형 Vision-Language Model(VLM/MLLM)을 PTQ(Post-Training Quantization)로 저비트 양자화하면서, 비전 토큰과 텍스트(캡션) 토큰 간 양자화 민감도 불균형(modality imbalance)을 보정하는 것. `[코드]` README.md:1
- **원논문**: *MBQ: Modality-Balanced Quantization for Large Vision-Language Models*, arXiv:2412.19509 (Shiyao Li 외, Tsinghua NICS-EFC / Infinigence-AI). `[코드]` README.md:1, 157-165 (citation 블록)
  - 저장소 내 PDF는 없음. `[확인불가]` (Glob `**/*.pdf` 결과 없음)
- **핵심 아이디어** (코드로 확인되는 범위):
  1. AWQ식 "activation-aware weight scaling" 그리드 서치를 기반(`auto_scale_block`)으로 함. `[코드]` methods/mbq/quantize/auto_scale.py:88-199
  2. 여기에 **modality 분리 손실**을 추가: 출력 오차를 비전 마스크(`vis_mask`)와 캡션/정답 마스크(`ans_mask`)로 분리해 가중합. `[코드]` auto_scale.py:144-183
  3. **gradient 기반 reweight**: 역전파 grad의 비전/캡션 평균 비율로 비전 토큰 손실에 가중치(`reweight_ratio`)를 부여. `[코드]` pre_quant.py:30-105, 295-305, 352-360
  4. **distort 입력**: 다음 레이어의 입력으로 "이미 양자화된 출력"을 사용해 누적 오차를 반영하는 블록 단위 재구성. `[코드]` pre_quant.py:308-311, 435-461, auto_scale_wa_distort.py:240-287
- 지원 방식: `mbq`, `awq`, `smoothquant`, `rtn` 4종. `[코드]` quant_wrapper.py:9-30

---

## 2. 디렉토리 구조 (자체 소스 + 제외 표기)

```
MBQ/
├── main_quant.py                  # 양자화 서치 진입점(CLI/YAML)            [정밀분석]
├── main.py                        # 평가 진입점(lmms-eval 연동)             [요약]
├── setup.py                       # 패키지 설치                              [요약]
├── README.md                      # 사용법/논문/citation                     [정밀분석]
└── qmllm/
    ├── quantization/              # ★ 자체 양자화 코어
    │   ├── quant_funcs.py         # pseudo_quantize_tensor 등 기본 양자화      [정밀분석]
    │   ├── qlinear.py             # WALinear (W·A 동시 양자화 Linear)         [정밀분석]
    │   └── quant_wrapper.py       # 메소드 디스패처                          [정밀분석]
    ├── methods/                   # ★ 양자화 알고리즘
    │   ├── mbq/                   # MBQ 본체                                  [정밀분석]
    │   │   ├── entry.py
    │   │   └── quantize/{pre_quant, auto_scale, auto_scale_wa,
    │   │       auto_scale_distort, auto_scale_wa_distort,
    │   │       quantizer, qmodule}.py
    │   ├── awq/                   # AWQ (비교 기준)                          [정밀분석]
    │   ├── smoothquant/           # SmoothQuant (smooth/gen_act_scales)       [정밀분석]
    │   └── rtn/                   # Round-To-Nearest                         [정밀분석]
    ├── sensitive/gen_grad.py      # (사실상 빈 파일, 1줄)                     [확인불가]
    ├── utils/search.py            # op 이름/모듈 탐색 유틸                    [정밀분석]
    ├── calibration/{coco_vl, pileval}.py  # 캘리브레이션 데이터 로더          [정밀분석]
    ├── datasets/                  # 멀티모달 데이터셋                        [요약]
    └── models/{internvl2, llava_onevision, llava_v15, vila, qwen2_vl}/
                                   # VLM 래퍼 — 양자화 연동부만 요약          [부분: 외부 모델정의 제외]
```

- **제외 대상**: `models/*` 내부의 HuggingFace 모델 정의(InternVL/LLaVA/Qwen2-VL 원본 forward, conversation, tokenizer 등)는 외부 이식 코드이므로 양자화 연동부만 요약. `3rdparty`(LLaVA-NeXT, lmms-eval)는 저장소에 미포함(README 설치 안내만). `[코드]` README.md:15-28

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 기본 양자화 함수 — `qmllm/quantization/quant_funcs.py`
- `pseudo_quantize_tensor(tensor, n_bits, zero_point, q_group_size, per_tensor, inplace)` — 모든 weight/activation/KV의 기본 fake-quant 함수. `[코드]` quant_funcs.py:3-45
  - `q_group_size>0`이면 `(-1, group)`으로 reshape → **per-group(그룹) 양자화**. `[코드]` 9-11
  - `per_tensor`이면 전체를 `(1,-1)`로 → **per-tensor**. `[코드]` 12-13
  - `zero_point=True`: 비대칭(asymmetric) `max_int=2^n-1`, scale=(max-min)/max_int, zero=round(-min/scale). `[코드]` 15-21
  - `zero_point=False`: 대칭(symmetric) absmax, `max_int=2^(n-1)-1`. `[코드]` 22-28
  - 양자화-역양자화를 한 번에 수행하는 **fake quantization**(실제 INT 저장 아님). `[코드]` 30-37
  - **주의**: scale/zero를 return하지 않고 dequant된 float 텐서만 반환(주석 처리됨). 실제 INT 패킹/커널 없음. `[코드]` 43-45
- `quantize_weight_per_channel_absmax`(per-channel, group=-1), `quantize_activation_per_token_absmax`(토큰별), per_tensor 변형들. 활성값은 토큰 단위로 reshape 후 absmax. `[코드]` 48-77

### 3.2 W·A 동시 양자화 Linear — `qmllm/quantization/qlinear.py`
- `class WALinear(nn.Module)` — weight-activation 양자화용 Linear. `[코드]` qlinear.py:6-77
  - 가중치는 `from_float`에서 미리 fake-quant 후 fp16 buffer로 저장. `[코드]` 14-15, 60-68
  - activation은 **forward 시 동적 양자화**: `per_token`(기본) 또는 `per_tensor`. `[코드]` 22-31, 48-52
  - `from_float`: weight를 `per_channel`/`per_tensor`/`per_group(weight_group=128)`로 양자화 선택. 기본 호출은 `weight_quant="per_channel", act_quant="per_token", w_bit=4, a_bit=8`. `[코드]` 54-74
  - 즉 MBQ의 W·A 경로는 **weight per-channel(대칭 absmax) + activation per-token(동적)** 조합. `[코드]` 61-62 + quant_funcs.py:49,57
  - `__repr__` = `W{w_bit}A{a_bit}Linear`. `[코드]` 76-77

### 3.3 메소드 디스패처 — `qmllm/quantization/quant_wrapper.py`
- `qwrapper(model, prompt_inputs, prompt_kwargs, args)`가 `args.method`로 분기. `[코드]` quant_wrapper.py:8-30
  - mbq/rtn은 `wa_quant = (w_bit<16 and a_bit<16)`로 weight-only vs weight-activation 자동 결정. `[코드]` 14, 27
  - mbq에 `reweight`, `distort`, `loss_mode` 등 MBQ 고유 인자 전달. `[코드]` 15-25

### 3.4 MBQ 진입점 — `qmllm/methods/mbq/entry.py`
- `mbq_entry(...)`: `scale_path` 캐시가 없고 `run_mbq_process`면 `run_mbq`로 스케일 서치 후 `torch.save`. `pseudo_quant`면 캐시 로드→`apply_mbq`(스케일 적용)→weight 또는 W·A pseudo-quant. `[코드]` entry.py:8-58
  - `model.to_cpu()/to_cuda()`로 메모리 관리. `[코드]` 25, 57
  - weight-only(`pseudo_quantize_model_weight`) vs W·A(`pseudo_quantize_model_weight_act`) 분기. `[코드]` 50-55

### 3.5 MBQ 본체 — `qmllm/methods/mbq/quantize/pre_quant.py` (가장 중요)
- `run_mbq(...)` 전체 흐름 `[코드]` pre_quant.py:199-477:
  1. **레이어0 Catcher 훅**으로 첫 디코더 블록 입력/kwargs를 가로채 캘리브레이션 입력 캐시(AWQ와 동일 패턴). `[코드]` 219-258
  2. `process_input`이 `vision_mask`/`caption_mask`를 입력 kwargs에서 분리. `[코드]` 190-196, 241
  3. **reweight 경로**: `GradCacheHook`을 down_proj/o_proj/w2/wo 등 Linear에 full_backward_hook으로 등록 → forward+`loss.backward()`로 출력 grad를 토큰별로 비전/캡션 마스크로 분리 누적. `[코드]` 30-105, 265-293
     - 각 레이어의 `vis_avg_grad / cap_avg_grad` 비율을 산출, attn/mlp별 중앙값(`attn_median`, `mlp_median`)을 baseline으로. `[코드]` 295-305
  4. **블록 단위 루프**: 각 디코더 블록마다 Linear 입력 feature를 forward hook으로 캐시 → wa_quant/distort 조합에 따라 4가지 auto_scale 함수 중 하나 호출. `[코드]` 317-430
     - reweight 시 현재 블록 인덱스에 해당하는 attn/mlp 비율을 `max(ratio, median)`으로 clip 후 전달. `[코드]` 352-360
  5. 서치된 스케일을 `apply_scale`로 즉시 반영하고, distort면 양자화 블록 출력(`inps_distort`)을 다음 블록 입력으로 갱신. `[코드]` 433-461
  6. 스케일 리스트에 글로벌 prefix를 붙여 `mbq_results["scale"]`에 누적, 최종 반환. `[코드]` 463-477
- `GradCacheHook` `[코드]` 30-105: backward 출력 grad를 `vis_mask`/`cap_mask`로 인덱싱해 `abs().mean()`을 batch별로 누적. → **modality별 민감도(grad 크기) 측정**이 MBQ의 핵심 차별점.
- `get_blocks`/`move_embed`: InternVL/LLaVA/Qwen2-VL 등 여러 VLM의 LLM 디코더 레이어 경로를 매핑. **VLM의 언어모델 디코더 블록만 대상**(비전 인코더는 대상 아님). `[코드]` 112-188 (예: InternVLChatModel→`model.language_model.model.layers` 123)
- `apply_mbq(model, mbq_results)` = `apply_scale(model, scale)`. `[코드]` 480-481

### 3.6 스케일 서치 — `qmllm/methods/mbq/quantize/auto_scale.py` (weight-only)
- `auto_scale_block(...)`: AWQ 구조 + MBQ 손실. `[코드]` auto_scale.py:88-630
  - `_search_module_scale`: `x_max=mean(|x|)` 채널 스케일을 `ratio∈[0,1)` 20-grid로 거듭제곱(`x_max.pow(ratio)`), `scales/sqrt(max*min)`로 정규화 → 가중치에 곱하고 fake-quant 후 다시 나눠 출력 오차 최소 ratio 선택. `[코드]` 110-199 (특히 129-135)
  - **MBQ 손실(차별점)**: `loss_mode`(mae/mse)와 `ans_mask`/`vis_mask`/`reweight_ratio`로 분기.
    - 둘 다 있으면 `(L_ans + reweight_ratio·L_vis)/(분모)`로 modality 가중합. `[코드]` 164-183 (mae), 144-163 (mse)
  - 블록별 prev_op→layers 매핑: LLaMA/Qwen2/InternLM2/Qwen2-VL 등 디코더 레이어 타입별로 qkv/o_proj/gate·up/down 스케일 쌍을 정의. reweight_ratio를 attn/mlp에 분배. `[코드]` 258-626
- `apply_scale`: prev_op이 LayerNorm/RMSNorm이면 `scale_ln_fcs`(LN weight÷scale, FC weight×scale), Linear면 `scale_fc_fc`, GELU면 `ScaledActivation`로 등가 변환(수학적 동치 유지). `[코드]` 36-86, 633-665

### 3.7 W·A / distort 변형 — `auto_scale_wa.py`, `auto_scale_distort.py`, `auto_scale_wa_distort.py`
- `auto_scale_block_wa_distort` `[코드]` auto_scale_wa_distort.py:93-807:
  - `_search_module_scale_wa_distort`: 스케일 후보마다 각 fc를 **`WALinear.from_float`로 W·A 양자화** 시킨 뒤, **양자화된 입력 `x_q`**(`x_q/scales`)로 forward해 오차를 측정 → 실제 W4A8 추론 오차를 직접 최적화. `[코드]` 99-220 (124-145)
  - `_auto_get_input_feat_distort`: 모듈을 deepcopy→이전까지의 스케일 적용→Linear들을 WALinear로 교체→양자화 입력으로 forward해 다음 sub-layer의 "양자화된 입력 feature"를 수집(누적 양자화 오차 반영). `[코드]` 240-287
  - LLaMA/Qwen2/InternLM2/Qwen2-VL에 대해 q/k/v → o_proj → gate/up → down 순서로 단계적으로 distort 입력을 재계산. `[코드]` 330-803
- 즉 distort = **순차적 블록 재구성(누적 오차 인지)**, wa = **W·A 양자화 인지 스케일**. 둘 다 MBQ가 AWQ 위에 추가한 부분. `[추정: 논문의 "quantization-aware" 항목과 대응되나 PDF 없어 명칭 매칭은 추정]`

### 3.8 quantizer / qmodule
- `quantizer.py`: `pseudo_quantize_model_weight`(weight-only 일괄), `pseudo_quantize_model_weight_act`(모든 Linear→WALinear 교체). `[코드]` quantizer.py:61-102
- `qmodule.py`: `ScaledActivation`(act 출력을 scale로 나눔), AWQ 호환용 zeros width 계산 유틸. `[코드]` qmodule.py:9-31

### 3.9 비교 메소드
- **AWQ** (`methods/awq/`): `auto_scale_block`이 MBQ와 거의 동일하나 **손실이 단일 mse, mask=ans_mask만, reweight/distort 없음**. `[코드]` awq/quantize/auto_scale.py:88-166 (144-151), entry.py:7-40. → MBQ는 AWQ + (modality mask + grad reweight + distort).
- **SmoothQuant** (`methods/smoothquant/`): `smooth_ln_fcs`가 `scale = act_scale^α / weight_scale^(1-α)` (α=0.5)로 활성→가중치 difficulty 이전(migration). LLM뿐 아니라 **ViT(CLIP/SigLIP) 인코더 레이어도 smooth** 지원. `[코드]` smooth.py:35-40, 207-272 (smooth_vit), entry.py:8-31
- **RTN** (`methods/rtn/`): 서치 없이 `pseudo_quantize_tensor` 직접 적용(weight 또는 W·A). `[코드]` rtn/quantizer.py:46-85

### 3.10 VLM 양자화 연동 범위 (모델 래퍼 — 요약)
- `models/base.py`: `fetch_vit/fetch_llm/fetch_proj` 추상 인터페이스. `[코드]` base.py:1-30
- InternVL2 래퍼 `generate_input`: 이미지 토큰(`img_context_token_id`) 위치로 **`vision_mask`**, label≠-100로 **`caption_mask`(answer_mask)** 생성 → MBQ의 modality 손실/grad 분리의 입력. `[코드]` internvl2.py:432-485 (469-482)
- **양자화 대상은 LLM 디코더의 Linear들**(qkv/o/gate/up/down). 비전 인코더 자체 양자화는 MBQ 경로에는 없음(SmoothQuant만 ViT smooth 지원). `[코드]` pre_quant.py:74, 112-142; smooth.py:207-272

---

## 4. 알고리즘 / 수식

### 4.1 AWQ activation-aware scaling (MBQ의 기반)
채널별 활성 크기 s_x = mean(|X|) (채널축). 후보 스케일:
```
s = clamp(s_x^ratio, 1e-4),  ratio ∈ {0, 1/20, ..., 19/20}
s = s / sqrt(max(s)·min(s))
W' = Q(W·diag(s)) · diag(1/s)        # 등가변환 후 weight만 양자화
ratio* = argmin_ratio  L(blockout(X) vs blockout_orig(X))
```
`[코드]` auto_scale.py:119, 129-135, 186-191

### 4.2 MBQ modality-balanced loss
```
L_mae = ( Σ|Δ|·M_ans + λ·Σ|Δ|·M_vis ) / ( ΣM_ans + ΣM_vis )
  Δ = org_out - quant_out,  M_ans=캡션마스크, M_vis=비전마스크, λ=reweight_ratio
```
mse는 |Δ|→Δ². `[코드]` auto_scale.py:164-183

### 4.3 gradient 기반 reweight λ
```
r_layer = mean(|grad_vis|) / mean(|grad_cap|)            # 레이어별 grad 비율
λ_attn = max(r_oproj/woproj, median_over_blocks(r_attn))  # 비전 grad가 큰 블록 강조
λ_mlp  = max(r_down/w2,      median_over_blocks(r_mlp))
```
`[코드]` pre_quant.py:61-65, 298-305, 352-360

### 4.4 SmoothQuant migration (비교)
```
s_j = (max_i|X_ij|)^α / (max_i|W_ij|)^(1-α),  α=0.5
LN.weight /= s ;  FC.weight *= s
```
`[코드]` smooth.py:35-46

> 비고: 위 수식 표기는 코드에서 직접 유도. 논문 원식 기호와의 1:1 대응은 PDF 부재로 `[추정]`.

---

## 5. 학습 / 평가(캘리브레이션) 파이프라인

- **진입점** `main_quant.py`: lmms-eval로 모델 로드 → `get_process_model`로 VLM 래퍼 생성 → 캘리브레이션 데이터(`pileval` 순수텍스트 / `coco` 멀티모달) 생성 → `qwrapper`로 양자화. `[코드]` main_quant.py:99-135
- **캘리브레이션 데이터** `calibration/coco_vl.py`: ShareGPT4V류 JSON(L)에서 n_samples 샘플링, 이미지 로드, 모델별 `preprocess_data`/`data_collator`로 배치화. `few_shot_format`(샘플 연결)·`interleave_format`(이미지-텍스트 사이 512 순수텍스트 토큰 삽입) 지원. `[코드]` coco_vl.py:14-97; README.md:48-52
- **양자화 절차**: `--run_process`로 스케일 서치→`scale_path` 저장, `--pseudo_quant`로 fake-quant 적용. weight-only는 `--w_bit 4 --w_group 128`, W·A는 `--w_bit 4 --a_bit 8`. `[코드]` README.md:67-153; entry.py:24-55
- **평가** `main.py`: lmms-eval 태스크(mmmu 등)로 fake-quant 모델 평가. `[코드]` README.md:113-153 (`[요약]`: main.py 본문 미정독)
- 학습(파인튜닝)은 없음 — 순수 PTQ. `[코드]` 전 메소드가 `@torch.no_grad`(reweight의 backward만 grad 계산, 가중치 갱신 없음) pre_quant.py:199, 272-288

---

## 6. 의존성

- PyTorch, `transformers`(bloom/opt/llama/qwen2/mistral/mixtral/falcon/clip 모델 정의 import). `[코드]` pre_quant.py:12-14, smooth.py:4-17
- `lmms-eval`(모델 로드/평가), `LLaVA-NeXT`(3rdparty). `[코드]` main_quant.py:18; README.md:15-28
- `datasets`(pile-val-backup, HuggingFace), `accelerate`(device dispatch), numpy, tqdm, yaml, PIL. `[코드]` coco_vl.py:7,72; utils/search.py:2; main_quant.py:11-12

---

## 7. 강점 / 한계 / 리스크

**강점**
- AWQ 호환 구조 위에 modality 분리 손실/grad reweight/distort를 모듈식으로 추가 — 4가지 auto_scale 조합으로 weight-only/W·A × distort 유무를 깔끔히 커버. `[코드]` pre_quant.py:377-430
- 다수 VLM 백본(InternVL2/LLaVA-OV/LLaVA1.5/VILA/Qwen2-VL) 디코더 경로를 일관 매핑. `[코드]` pre_quant.py:112-142
- 등가 스케일 변환(LN/FC/GELU 흡수)으로 추가 연산 없이 가중치에 스케일 흡수. `[코드]` auto_scale.py:633-665

**한계 / 리스크**
- **실제 INT 커널 없음**: `pseudo_quantize_tensor`는 dequant된 float 반환(fake quant). 실측 속도/메모리 이득은 별도 추론엔진 필요. `[코드]` quant_funcs.py:30-45
- **비전 인코더 미양자화**(MBQ 경로): 양자화는 LLM 디코더 Linear에 한정. ViT/projector는 SmoothQuant smooth_vit만 존재. `[코드]` pre_quant.py:74; smooth.py:207-272
- distort/wa 서치는 블록마다 deepcopy+재forward로 **비용이 큼**(메모리/시간). `[코드]` auto_scale_wa_distort.py:240-287, pre_quant.py:435-461
- `sensitive/gen_grad.py`가 사실상 빈 파일(1줄) — grad 로직은 `pre_quant.py`의 `GradCacheHook`에 인라인. `[코드]` gen_grad.py(1라인); pre_quant.py:30-105
- W·A 경로는 weight per-channel + act per-token으로 **고정**(group-wise act 미지원). `[코드]` qlinear.py:55, 98

---

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

- **FPGA 친화 양자화 요소**:
  - `pseudo_quantize_tensor`의 비대칭/대칭, per-channel/per-group 분기는 정수 scale+zero_point 정의가 명확 → HLS/RTL의 정수 MAC + 채널별 scale 곱셈 구조로 직접 매핑 가능. `[코드]` quant_funcs.py:15-37
  - per-channel weight scale은 출력채널마다 상수배 → FPGA에서 가중치 ROM에 사전 흡수(`apply_scale`의 등가변환과 동일 발상) 가능. `[코드]` auto_scale.py:633-665
  - W·A 양자화 시 activation **per-token 동적** 양자화는 FPGA에서 토큰마다 absmax 산출 회로가 필요 → 하드웨어 비용 큼. 우리는 **per-tensor 또는 정적 calibration scale**(SmoothQuant식 미리 고정 scale)이 더 친화적. `[코드]` qlinear.py:22-31 vs smooth.py:35-46
- **outlier 처리**: AWQ scaling은 채널별 outlier를 weight로 흡수해 activation 범위를 줄임 → activation 비트폭을 낮춰도 정확도 유지 → FPGA의 좁은 정수 폭(INT4/INT8) 친화. `[코드]` auto_scale.py:129-135
- **VLM 시선추적 응용**: MBQ의 modality-balanced 아이디어(비전 토큰 vs 텍스트 토큰 민감도 분리)는, XR 시선추적에서 "시각 입력(eye/scene patch) 경로"와 "메타/텍스트 경로"의 비트 예산을 차등 배분하는 설계 근거로 차용 가능. grad 기반 민감도 측정(`GradCacheHook`)은 어느 레이어/모달리티에 비트를 더 줄지 결정하는 **민감도 분석 도구**로 재사용 가치 높음. `[코드]` pre_quant.py:30-105
- **주의**: 본 repo는 LLM 디코더 중심이라 ViT 인코더 가속(우리 HG-PIPE 계열)에 직접 코드 재사용은 제한적. ViT 양자화는 SmoothQuant의 `smooth_vit`(CLIP/SigLIP) 경로가 더 직접적 참고점. `[코드]` smooth.py:207-272

---

## 9. 근거 표기 규칙 요약
- `[코드]`: 해당 파일·라인을 Read로 직접 확인.
- `[추정]`: 코드 구조·주석상 합리적 추론(논문 PDF 부재로 명칭/원식 매칭 등).
- `[확인불가]`: 저장소 내 근거 없음(예: 동봉 논문 PDF, gen_grad.py 의도).
