# Q-DiT 코드베이스 정밀 분석

> 대상 저장소: `REF/ViT-Quantization/Q-DiT`
> 논문: **Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers** (arXiv 2406.17343, CVPR'25 추정)
> 저자: Lei Chen, Yuan Meng, Chen Tang, Xinzhu Ma, Jingyan Jiang, Xin Wang, Zhi Wang, Wenwu Zhu (README.md:37-45 BibTeX 확인)
> 분석 범위: `qdit/` 핵심 양자화 패키지 + `scripts/` + `models/models.py`. `evaluations/`(FID), `diffusion/`(샘플링)은 이름만 언급.

---

## 1. 개요

Q-DiT는 **Diffusion Transformer(DiT)** 에 대한 **Post-Training Quantization(PTQ)** 프레임워크다. 학습 없이(calibration data만 사용) DiT의 weight/activation을 저비트(W4/A8 등)로 양자화한다.

- **원논문**: arXiv 2406.17343 (README.md:41-44에서 `eprint={2406.17343}` 확인).
- **핵심 아이디어 4가지**(코드로 확인된 범위):
  1. **Group-wise(fine-grained) quantization** — weight/activation을 채널 방향 그룹 단위로 분할하여 그룹마다 별도 scale/zero-point 적용 (`quant.py:8-47`, `quant.py:61-63`).
  2. **GPTQ 기반 weight 보정** — Hessian 기반 2차 정보로 weight를 순차 보정 (`gptq.py:197-318`).
  3. **Automatic granularity(자동 그룹 크기 할당)** — 블록·연산자별 group size를 evolutionary search(진화 탐색)로 자동 결정 (`scripts/evolution.py:74-319`).
  4. **MSE/Max 기반 scale 탐색** — clip ratio를 grid search로 최적화 (`quant.py:99-142`, `gptq.py:145-162`).
- **베이스 모델**: DiT-XL/2 (depth=28, hidden=1152, num_heads=16) (`models/models.py:346-347`).
- **데이터셋/태스크**: ImageNet 256×256 class-conditional image generation, FID 평가 (`scripts/quant_main.sh:8`, `scripts/evolution.sh:7-8`).

> **참고(코드 확인 vs 논문 주장)**: 코드는 README와 BibTeX 외에 논문 본문을 포함하지 않으므로, "automatic granularity"라는 명명은 논문 용어 추정이며 코드상으로는 evolutionary search로 group size 벡터를 탐색하는 형태로 구현되어 있다(아래 3.7절 참조).

---

## 2. 디렉토리 구조

```
Q-DiT/
├── qdit/                      # ★ 핵심 양자화 패키지
│   ├── quant.py               # Quantizer 클래스, per-group/per-channel int 양자화 커널
│   ├── gptq.py                # GPTQ Hessian 보정 (Quantizer_GPTQ, GPTQ.fasterquant)
│   ├── qLinearLayer.py        # 양자화 Linear 래퍼 (QLinearLayer)
│   ├── qBlock.py              # DiT Block/Attention/MLP 양자화 래퍼
│   ├── outlier.py             # (이름과 달리) activation 통계 수집 (get_act_scales/get_act_stats)
│   ├── modelutils.py          # 레이어 교체·모델 양자화 오케스트레이션
│   ├── datautils.py           # calibration 데이터 로더, gradient(KL) 수집 유틸
│   └── __init__.py            # (빈 파일, 1줄)
├── scripts/
│   ├── collect_cali_data.py   # DiT로 calibration 입력 [x,t,y] 샘플링·저장
│   ├── quant_main.py          # 메인 양자화·샘플링 엔트리포인트
│   ├── quant_main.sh          # 실행 예시 (W4A8 + GPTQ + group=128)
│   ├── evolution.py           # group size automatic granularity 진화 탐색
│   └── evolution.sh           # 진화 탐색 실행 예시
├── models/
│   ├── models.py              # DiT/DiTBlock/Attention/Mlp 정의 (Meta DiT 원본 기반)
│   └── __init__.py
├── utils/                     # download.py, logger_setup.py
├── setup.py                   # name='Q-DiT', find_packages
├── requirements.txt           # torch, torchvision, timm, diffusers, transformers, accelerate, pytorch_lightning
└── README.md

# 이름만 언급(분석 제외):
├── evaluations/               # FID/IS 평가기 (evaluator.py) — TensorFlow Inception 기반
├── diffusion/                 # gaussian_diffusion.py 등 DDIM/DDPM 샘플링 — 외부(ADM/DiT) 차용
├── .git / __pycache__         # 제외
```

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `qdit/quant.py` — 양자화 커널과 Quantizer 클래스

이 파일이 **integer uniform 양자화의 수치 핵심**이다.

#### (1) `quantize_tensor_channel_group(...)` (quant.py:7-47)
- weight 양자화 진입점. `group_size`에 따라 분기:
  - `group_size == 0` → per-channel 양자화 (quant.py:16-17).
  - `group_size > 0` → `W.shape[1]`(입력 채널 축)을 `group_size` 단위로 잘라 각 슬라이스를 독립 양자화 (quant.py:19-45). 입력은 `W.shape[-1] % group_size == 0`을 강제(quant.py:13).
- `channel_group > 1`이면 연속 채널 묶음이 동일 양자화 설정을 공유 — bitsandbytes 커널 효율 고려용 reshape (quant.py:25-26, 43-44). 주석상 "Continous number of channel_group channels share the same quantization setup"(quant.py:6).

#### (2) `quantize_tensor(...)` (quant.py:50-147) — uniform int 양자화 본체
- 입력을 `[num_groups, group_size]` 2D 레이아웃으로 reshape (quant.py:61-65).
- `tiling > 0` (16×16 block-wise)은 **폐기됨**(`assert False`, quant.py:58-59).
- **`quant_method == "max"`** (quant.py:75-97):
  - 대칭(sym): `w_max = |w|.amax`, `q_max = 2^(b-1)-1`, `scales = w_max/q_max`, `base=0` (quant.py:82-88).
  - 비대칭(asym): `scales = (w_max - w_min)/q_max`, `base = round(-w_min/scales)` clamp (quant.py:89-96). `q_max=2^b-1`, `q_min=0`.
  - dequant 포함 양자화: `w = (clamp(round(w/scales)+base, q_min, q_max) - base) * scales` (quant.py:97) → **fake quantization(시뮬레이션)**, 실제 정수 저장이 아님.
- **`quant_method == "mse"`** (quant.py:99-142):
  - clip ratio를 100단계 grid(`1.0 - i*0.001`)로 줄여가며 `lp_loss(p=2.4)` 최소화하는 scale/base 선택 (quant.py:107-130). LAPQ(arXiv 1911.07190) 방식 명시(quant.py:122-123).
- **부동소수점(fp) 경로**는 `quant_type == "int"`로 assert되어 `quantize_tensor`에서는 막혀 있음(quant.py:74). (fp4는 `gptq.py`의 `quantize_gptq`에만 별도 분기 존재 — 3.2절.)

#### (3) Activation 양자화 래퍼
- `quantize_activation_wrapper(x, args)` (quant.py:151-173): `abits>=16`이면 패스, 아니면 `act_group_size` 단위로 동적 양자화. 마지막 축 기준 reshape 후 `quantize_tensor` 호출.
- `quantize_attn_{q,k,v}_wrapper` (quant.py:176-210): attention의 Q/K/V를 **head_dim 축**으로 양자화. `head_dim == 72` 하드코딩 assert (quant.py:179, 191, 203) — DiT-XL은 1152/16=72이므로 모델 종속 상수.

#### (4) `Quantizer(nn.Module)` (quant.py:212-267)
- activation quantizer 모듈. `static`(quant.py:222)이면 미리 계산된 `scales`(scale, base 쌍)로 정적 양자화, 아니면 `act_quant` 함수(dynamic)로 처리.
- 정적 경로(quant.py:225-245): `act_group_size`에 따라 scale/base를 broadcast하여 `(clamp(round(x/scales)+base, q_min, q_max)-base)*scales` 적용. **비대칭만 지원**(`assert self.args.a_sym == False`, quant.py:227).
- `configure(func, scales)` (quant.py:253-264): dynamic이면 함수 등록, static이면 scales buffer 등록 및 q_min/q_max 설정.

### 3.2 `qdit/gptq.py` — Hessian 기반 weight 보정 (GPTQ)

원본 IST-DASLab GPTQ를 DiT용으로 수정(gptq.py:1-4, 19-25).

#### (1) `quantize_gptq(x, scale, zero, maxq, channel_group, quant_type)` (gptq.py:26-59)
- `int`: `q = clamp(round(x/scale)+zero, 0, maxq); q = scale*(q-zero)` (gptq.py:35-38) — uniform affine.
- `fp`: bitsandbytes FP4(`quantize_fp4`/`dequantize_fp4`)를 **rounding kernel로 차용**하는 분기(gptq.py:39-58). blocksize 64 미만은 ones 패딩으로 보정(gptq.py:42-53). 단, 코드 내 `quantize_fp4`/`dequantize_fp4` import는 보이지 않음 → fp 경로는 별도 의존 필요(확인 불가, 실제 실험은 int 경로 중심).

#### (2) `Quantizer_GPTQ(nn.Module)` (gptq.py:61-195)
- `configure(...)` (gptq.py:68-97): `bits`로 `maxq=2^bits-1`(int) 설정. `perchannel`, `channel_group`, `sym`, `mse`, `clip_ratio` 등 보유.
- `find_params(x, weight)` (gptq.py:99-184): per-channel min/max로 scale/zero 산출. 대칭이면 zero=`(maxq+1)/2`(gptq.py:140-141), 비대칭이면 `round(-xmin/scale)`(gptq.py:142-143). `clip_ratio`로 range 축소(gptq.py:139).
  - `mse=True`이면 `maxshrink*grid` 단계로 scale grid search, `norm=2.4` Lp 손실 최소화(gptq.py:145-162).
- `quantize(x)` (gptq.py:186-189): 준비되면 `quantize_gptq` 호출.

#### (3) `GPTQ` 클래스 — OBS/OBQ 기반 순차 보정 (gptq.py:197-323)
- `__init__` (gptq.py:198-214): layer weight `[rows, columns]` 추출, **Hessian `H = zeros(columns, columns)`** 누적용 버퍼 생성. `n_nonout = W.shape[1]`(전체 입력 채널 수)로 설정 — 즉 **outlier 분리 비활성**(전 채널을 "non-outlier"로 처리).
- `add_batch(inp, out)` (gptq.py:216-238): calibration 입력으로 **Hessian H ≈ Σ (2/n) x xᵀ** 온라인 누적 (gptq.py:235-238). running mean 형태로 nsamples 갱신.
- `fasterquant(blocksize=128, percdamp=.01, groupsize=-1)` (gptq.py:240-318) — **GPTQ 핵심 루프**:
  1. `actorder=False` 강제(gptq.py:243).
  2. quantizer 미준비 시 `find_params`로 초기 scale 산출(gptq.py:251-252).
  3. dead 채널(H 대각=0) 처리: `H[dead]=1, W[:,dead]=0` (gptq.py:257-259).
  4. **damping**: `H[diag]+= percdamp*mean(diag(H))` (gptq.py:264-266).
  5. **Cholesky 역행렬**로 `Hinv`(upper) 계산(gptq.py:268-271).
  6. blocksize(128) 단위 외부 루프 × 그룹/열 단위 내부 루프(gptq.py:273-304):
     - `groupsize > 0`이고 열 인덱스가 그룹 경계면 해당 그룹에 대해 `find_params` 재계산 → **그룹별 scale 갱신**(gptq.py:287-289).
     - 열 양자화 후 잔차 `err1 = (w-q)/d`를 `Hinv`로 **나머지 열에 전파**(OBQ weight update): `W1[:, i:] -= err1 · Hinv1[i, i:]` (gptq.py:290-299), 블록 간에도 `W[:, i2:] -= Err1 · Hinv[i1:i2, i2:]` (gptq.py:304).
  7. 양자화된 `Q`를 layer weight에 덮어씀(gptq.py:310-315).

### 3.3 `qdit/qLinearLayer.py` — 양자화 Linear 래퍼 (qLinearLayer.py:16-65)
- `QLinearLayer(originalLayer, args, enable_quant)`: 원본 `nn.Linear`의 weight/bias를 buffer로 복제(qLinearLayer.py:25-30).
- `forward`: 평범한 `F.linear`(qLinearLayer.py:33-36) — **weight는 forward 시점이 아니라 `quant()` 호출 시 in-place로 fake-quantize**.
- `quant()` (qLinearLayer.py:43-62): `wbits>=16`이면 패스, 아니면 `quantize_tensor_channel_group`로 weight를 양자화하고 `self.weight`를 교체. group_size/channel_group/clip_ratio/sym/quant_method 등 모든 weight 양자화 옵션을 args에서 받음.
- `find_qlinear_layers(module)` (qLinearLayer.py:5-14): 재귀적으로 `enable_quant`인 QLinearLayer를 수집 — GPTQ 대상 탐색용.

### 3.4 `qdit/qBlock.py` — DiT Block/Attention/MLP 양자화 래퍼 (qBlock.py)
- **`QuantDiTBlock(dit_block, args)`** (qBlock.py:12-61): 원본 `DiTBlock`을 받아 attn/mlp를 `QuantAttention`/`QuantMlp`로 교체. `adaLN_modulation`의 Linear도 `QLinearLayer`로 래핑(qBlock.py:26-29).
  - `quantize_bmm_input=True`이면 LayerNorm 출력·attention 출력·adaLN 출력에 `Quantizer`를 삽입하여 **activation도 양자화**(qBlock.py:30-35, 57-60).
- **`QuantAttention(attn, args)`** (qBlock.py:64-142):
  - `qkv`/`proj`를 `QLinearLayer`로, 입력·중간 activation을 `Quantizer`로 래핑(qBlock.py:84-91).
  - forward: `input_quant → qkv → (q/k/v_quant) → SDPA/manual attn → act_quant → proj`(qBlock.py:111-142). `reorder_index_*` buffer는 채널 재정렬용으로 존재하나 **기본 None**(미사용 — Atom 차용 흔적, qBlock.py:92-93, 113-114).
- **`QuantMlp(mlp, args)`** (qBlock.py:144-187): `fc1`/`fc2`를 `QLinearLayer`로, `input_quant`/`act_quant`로 activation 양자화. forward: `input_quant → fc1 → act(GELU) → norm → act_quant → fc2`(qBlock.py:176-187).

### 3.5 `qdit/outlier.py` — (실제로는) activation 통계 수집 (outlier.py)
- **파일명은 outlier지만 outlier 채널 분리 로직은 없음.** 내용은 activation scale/Hessian 통계 수집기다.
- `get_act_stats(model, dataloader, ..., metric='hessian')` (outlier.py:8-60): 각 Linear의 input/output에 hook을 걸어 Hessian 대각(`diag(x xᵀ)`) 또는 abs-mean 통계를 누적(outlier.py:18-32).
- `get_act_scales(model, diffusion, dataloader, ..., args)` (outlier.py:63-124): **static activation 양자화용 scale/base** 산출. group_size 단위로 max/min을 EMA(0.9/0.1)로 누적(outlier.py:78-87) 후 비대칭 scale `(max-min)/q_max`, base `round(-min/scale)` 계산하여 `torch.stack([scales, base])` 반환(outlier.py:114-121).

> **근거 기반 정정**: 작업 지시의 "outlier 채널 처리"는 **코드에 구현되어 있지 않다**. `gptq.py`의 `n_nonout = W.shape[1]`(gptq.py:213)은 전 채널을 처리하므로 outlier/non-outlier 분리(Atom 스타일)는 사실상 비활성화됨. 이는 Atom 차용 흔적이 남아 있으나 Q-DiT에서는 group-wise 양자화로 대체되었음을 시사(추정).

### 3.6 `qdit/modelutils.py` — 레이어 교체·양자화 오케스트레이션 (modelutils.py)
- `add_act_quant_wrapper(model, device, args, scales)` (modelutils.py:15-71): `model.blocks`의 각 `DiTBlock`을 `QuantDiTBlock`으로 교체하고, 블록별 `weight_group_size[i]`/`act_group_size[i]`를 주입(modelutils.py:18-27). 각 activation Quantizer를 `configure`로 설정하고 scale 연결(modelutils.py:37-66).
- `quantize_model(model, device, args)` (modelutils.py:73-99): GPTQ 미사용 시. 블록을 GPU로 올려 `mlp.fc1/fc2`, `attn.qkv/proj`의 `.quant()`를 순차 호출(modelutils.py:92-95) → max/mse 기반 weight 양자화.
- `quantize_model_gptq(model, device, args, dataloader)` (modelutils.py:134-204): **GPTQ 경로**.
  - 블록별로 `find_qlinear_layers`로 대상 Linear 수집(modelutils.py:155).
  - 각 Linear에 `GPTQ` + `Quantizer_GPTQ` 생성, `perchannel=True`, `channel_group`, `clip_ratio`, `quant_type` 등 configure(modelutils.py:163-171).
  - forward hook으로 `add_batch`(Hessian 누적) 후 calibration 데이터 통과(modelutils.py:173-185).
  - `fasterquant(percdamp, groupsize=weight_group_size[0])` 호출로 보정(modelutils.py:190-191) → **groupsize는 첫 블록 값(`[0]`)을 일괄 사용** (블록별 다른 group을 GPTQ에 반영하지는 않는 한계, 추정).
- `quantize_layer`/`quantize_block` (modelutils.py:101-132): 단일 레이어/블록 양자화 헬퍼(주로 search에서 활용).

### 3.7 `scripts/evolution.py` — automatic granularity (진화 탐색) (evolution.py)
- **목적**: 블록·연산자별 **group size 조합(granularity)** 을 진화 알고리즘으로 자동 탐색하여 FID를 최소화.
- `EvolutionSearcher` (evolution.py:74-315):
  - 탐색 차원: `len(blocks)*4` (DiT-XL=28블록 → 112), 즉 블록당 4개 연산자(attn-qkv, attn-proj, mlp-fc1, mlp-fc2)에 각각 group size 부여(evolution.py:107, 125-143).
  - **search space**: `[32, 64, 128, 192, 288]` (evolution.py:484). argparse에는 더 넓은 후보 리스트도 존재(evolution.py:381, 389).
  - `configure_group_size(model, group_size)` (evolution.py:125-143): 후보 벡터를 블록의 각 quantizer/Linear의 `act_group_size`/`weight_group_size`에 매핑.
  - **목적함수(loss) = FID** (evolution.py:145-202): 후보로 양자화된 모델로 이미지 샘플링 → reference batch 대비 `frechet_distance`(FID) 계산(evolution.py:193-200). 즉 적합도는 생성 품질(FID).
  - **제약(constraint)**: `efficiency_predictor(cand) = sum(cand)/len(cand)` (평균 group size, evolution.py:317-319)가 `constraint`(기본 128) 이상이어야 legal — **평균 bit-budget(평균 그룹 크기) 제약** 하에서 FID 최소화(evolution.py:110-111).
  - **EA 연산**: `get_random`/`get_cross`(uniform crossover)/`get_mutation`(`m_prob`로 변이)로 population 진화, top-k 유지(evolution.py:204-315). 초기 preset으로 `[128]*112`(전부 128) 포함(evolution.py:290-292).

### 3.8 `scripts/collect_cali_data.py` — calibration 데이터 생성 (collect_cali_data.py)
- DiT-XL/2를 `record_inputs=True`로 로드(collect_cali_data.py:53-57) → forward 중 DiT 내부 입력 `[x, t, y]`를 기록(`models.py:273-274`의 `inputs_list.append`).
- DDIM 샘플링 루프를 돌며 timestep별 입력을 수집, batch당 timestep을 무작위 분산 샘플링(`sample_cali_data_per_batch`, collect_cali_data.py:26-38) → 다양한 timestep 커버.
- 결과를 `cali_data_{image_size}.pth`로 저장(`[x_t, t, y]` 텐서 리스트, collect_cali_data.py:99-106). 기본 256 샘플(collect_cali_data.py:116).

### 3.9 `models/models.py` — DiT 정의 (models.py)
- `DiTBlock` (models.py:101-122): LayerNorm(affine=False) + `timm` `Attention` + `Mlp` + adaLN-Zero(`SiLU→Linear(hidden→6*hidden)`)(models.py:107-116). forward는 adaLN modulation으로 shift/scale/gate 6분할(models.py:118-121).
- `DiT` (models.py:145-284): PatchEmbed + timestep/label embedder + `blocks`(ModuleList) + FinalLayer. `forward(x,t,y)`에서 c = t_emb + y_emb를 각 block에 전달(models.py:249-264).
- DiT-XL/2: depth=28, hidden=1152, heads=16 → head_dim=72 (quant.py의 head_dim=72 assert와 일치, models.py:346-347).

---

## 4. 알고리즘/수식

### 4.1 Per-group (fine-grained) uniform quantization
weight 행 `W ∈ R^{out × in}`을 입력 채널축에서 group_size `g`로 분할. 그룹 `G`마다(quant.py:78-97):

- 비대칭(asym, 기본):
  - `s = (max(W_G) − min(W_G)) / (2^b − 1)`
  - `z = round(−min(W_G) / s)`  (zero-point/base)
  - `Q = clamp(round(W_G / s) + z, 0, 2^b−1)`
  - `Ŵ_G = (Q − z) · s`  (dequant)
- 대칭(sym):
  - `s = max(|W_G|) / (2^{b−1} − 1)`, `z = 0`
  - `Ŵ_G = clamp(round(W_G/s), −2^{b−1}, 2^{b−1}−1) · s`

그룹이 작을수록 scale 해상도가 높아 정확도↑, 대신 그룹당 (s, z) 메타데이터 저장 비용↑.

### 4.2 GPTQ Hessian 기반 weight update (OBS/OBQ)
calibration 입력 `x`로 layer Hessian 근사(gptq.py:235-238):

- `H ≈ (2/N) Σ_n x_n x_nᵀ`  (입력 채널 간 2차 상관)

damping과 Cholesky 역행렬(gptq.py:264-271):

- `H ← H + λ·mean(diag(H))·I`,  `λ = percdamp`(기본 0.01)
- `H_inv = Cholesky⁻¹(H)`

열(입력 채널) `i`를 순차 양자화하며 잔차를 나머지 가중치로 전파(OBQ, gptq.py:283-299):

- `q_i = Quant(w_i)`
- `err_i = (w_i − q_i) / [H_inv]_{ii}`
- `W_{:, i+1:} ← W_{:, i+1:} − err_i · [H_inv]_{i, i+1:}`
- 손실 누적: `L_i = (w_i − q_i)² / [H_inv]_{ii}²`

group-wise와 결합: 열 인덱스가 그룹 경계(`(i1+i) % groupsize == 0`)에 도달하면 해당 그룹에 대해 scale/zero 재계산(gptq.py:287-289) → **그룹별 양자화 파라미터를 GPTQ 루프 내에서 갱신**.

### 4.3 Automatic granularity의 evolutionary search 목적함수
group size 벡터 `c ∈ S^{4·L}` (S=`{32,64,128,192,288}`, L=블록 수)에 대해:

- 적합도: `minimize FID(Quant(model; c), reference)` (evolution.py:199-200)
- 제약: `mean(c) ≥ constraint` (기본 128) (evolution.py:110-111, 317-319) — 평균 그룹 크기 하한(=메모리/연산 budget) 유지.
- EA: population 50, crossover/mutation/random 혼합, top-k(=select_num) 유지, max_epochs 반복(evolution.sh:10-11, evolution.py:286-315).

> 직관: 작은 group(예: 32)을 일부 민감한 연산자에 할당해 정확도를 확보하되, 평균 그룹 크기 제약으로 전체 비용을 통제 → **mixed-granularity** 할당.

### 4.4 MSE scale 탐색 (clip ratio)
`quant.py:99-130` 및 `gptq.py:145-162`에서 clip 비율 `p = 1 − i·Δ`를 grid로 줄여가며 `‖W − Quant(W)‖_p^p` (p=2.4, LAPQ) 최소화하는 scale을 선택.

---

## 5. 학습/평가 파이프라인

PTQ이므로 재학습은 없고 calibration → 양자화 → 샘플링/평가 순서다.

1. **Calibration 데이터 생성** (`collect_cali_data.py`):
   `python collect_cali_data.py` → DiT-XL/2로 DDIM 샘플링하며 `[x_t, t, y]` 256개 수집 → `../cali_data/cali_data_256.pth` 저장.

2. **양자화 메인** (`quant_main.py` / `quant_main.sh`):
   - `quant_main.sh`(quant_main.sh:3-6) 설정: **W4A8**, `weight_group_size=128`, `act_group_size=128`, `--use_gptq`, `quant_method=max`.
   - 흐름(quant_main.py:165-209):
     a. DiT-XL/2 + 사전학습 ckpt 로드(quant_main.py:167-175).
     b. `static`이면 `get_act_scales`로 activation scale 수집(quant_main.py:184-189).
     c. `add_act_quant_wrapper`로 블록을 `QuantDiTBlock`으로 교체 + activation quantizer 삽입(quant_main.py:192).
     d. `--use_gptq`면 `quantize_model_gptq`(calib 256샘플), 아니면 `quantize_model`(quant_main.py:195-199).
     e. `validate_model`로 샘플 이미지 생성(quant_main.py:209). FID용 `sample_fid`는 주석 처리(quant_main.py:211).

3. **Automatic granularity 탐색** (`evolution.py` / `evolution.sh`):
   - `evolution.sh`(evolution.sh:3-11): **W4A8 sym**, group=128 시작, `ref_batch=VIRTUAL_imagenet256_labeled.npz`, EA(max_epochs=10, pop=50, constraint=128).
   - 후보별로 양자화→샘플링→FID 계산→EA 진화(evolution.py:294-315).

4. **평가**(이름만): `evaluations/evaluator.py`(TF Inception 기반 FID/IS). 분석 제외.

- **비트 설정**: argparse choices `[2,3,4,5,6,8,16]`(quant_main.py:218-223). 기본 실험은 W4/A8.
- **모델**: DiT-XL/2, ImageNet 256×256, 1000 classes, cfg_scale=1.5, 50 sampling steps(quant_main.sh:8).

---

## 6. 의존성

- **외부 알고리즘 기반**(README.md:49-50 명시):
  - **GPTQ** (IST-DASLab) — `gptq.py` 전체가 수정본(gptq.py:1-4).
  - **Atom** (efeslab) — group/channel_group, reorder_index, static/dynamic, `n_nonout` 등 흔적(quant.py 주석, qBlock.py:92-93). 단 outlier 분리·reorder는 실제 비활성.
  - **ADM / guided-diffusion** (OpenAI) — `evaluations/`, `diffusion/` 평가·샘플링.
- **모델 기반**: Meta **DiT** 원본 (`models/models.py` 헤더 Meta 저작권, models.py:1-9).
- **패키지**(requirements.txt): torch, torchvision, timm(`Attention`/`Mlp`/`PatchEmbed`), diffusers(`AutoencoderKL`), transformers(`Conv1D` 호환), accelerate, pytorch_lightning(`seed_everything`). evolution.py는 추가로 `tensorflow.compat.v1`(Inception FID) 필요(evolution.py:50).
- Python 3.8 환경(README.md:14).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **Fine-grained group-wise 양자화 + GPTQ + 자동 granularity**를 하나의 PTQ 파이프라인으로 결합 — DiT 같은 양자화 민감 모델에서 W4A8까지 비교적 안정적.
- group size를 블록·연산자별로 다르게 줄 수 있는 유연한 구조(`weight_group_size`/`act_group_size`가 블록 리스트, modelutils.py:18-27).
- GPTQ가 weight-only 보정이라 calibration 데이터만으로 빠르게 적용(재학습 불필요).
- evolutionary search가 FID를 직접 목적함수로 사용 → end-task에 정렬된 granularity 할당.

### 한계
- **outlier 처리 미구현**: 파일명과 달리 outlier 분리 없음(`n_nonout=전체`, gptq.py:213). Atom의 강점인 outlier-aware 양자화는 빠져 있음.
- **GPTQ groupsize 일괄화**: `fasterquant`에 `groupsize=weight_group_size[0]`(첫 블록 값)만 전달(modelutils.py:191) → automatic granularity로 찾은 블록별 weight group이 GPTQ 경로에 완전히 반영되지 않을 가능성(추정/확인 필요).
- **fake quantization만 구현**: 모든 경로가 dequant까지 포함한 시뮬레이션(quant.py:97). 실제 정수 커널/실속도 이득은 별도 구현 필요.
- **하드코딩 상수**: `head_dim==72` assert(quant.py:179 등), search 차원 112(evolution.py:107) 등 DiT-XL/2 종속.
- **evolution 비용**: 후보마다 전체 샘플링+FID 계산 → 매우 비쌈(GPU-시간 多).
- **fp4 경로 미완**: `quantize_fp4`/`dequantize_fp4` import 부재로 `quant_type=fp`는 실행 불가 가능(확인 불가).

### 리스크
- TF1 의존(evolution.py:50)으로 환경 구성 까다로움.
- static activation 양자화는 비대칭만 지원(quant.py:227) — 대칭 static 경로는 미지원.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적, 추정)

> 우리 프로젝트 맥락은 "HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적"으로 **추정**한다. 아래는 그 가정 하의 시사점이다.

### (1) Group-wise quantization의 하드웨어 매핑 비용
- Q-DiT의 group-wise는 그룹마다 `(scale, zero)` 메타데이터를 저장/적용한다(quant.py:78-97). FPGA 매핑 시 **그룹 scale을 어디에 저장(BRAM/URAM)하고 언제 곱할지**가 PE 데이터패스 비용을 좌우한다. group_size가 작을수록(예: 32) scale 테이블이 커지고 dequant 곱셈 빈도가 증가 → **BRAM 점유·DSP 사용량 트레이드오프**. HG-PIPE의 파이프라인 단계에 group dequant를 어떻게 융합할지(예: accumulator 출력단에서 그룹 scale 곱) 설계 포인트.
- 그룹 경계가 입력 채널 축이므로(quant.py:19), 가속기의 채널 타일링 크기를 group_size의 배수로 맞추면 scale 재로드가 단순해짐 (참고 가치).

### (2) GPTQ weight-only 보정의 ViT 전이성
- GPTQ는 **weight만** 보정하고 activation 분포를 건드리지 않으므로(gptq.py), ViT/시선추적 백본(예: DeiT-Tiny, RITnet류)에도 그대로 전이 가능. FPGA에서는 **오프라인에서 GPTQ로 weight를 보정한 뒤 정수 weight를 그대로 ROM/BRAM에 적재**하면 되어 추론 측 추가 비용이 없음 → 가속기 친화적.
- 다만 GPTQ는 Hessian 역행렬·Cholesky가 필요(gptq.py:268-271)하지만 이는 **오프라인 양자화 단계**에서만 수행되므로 FPGA 자원과 무관.

### (3) Automatic granularity → FPGA 비트/그룹 할당 설계
- evolution.py의 "평균 그룹 크기 제약 하 FID 최소화"(evolution.py:110-111, 317-319) 발상은 **FPGA 자원 제약(DSP/BRAM/LUT budget) 하 정확도 최대화** 문제로 직접 치환 가능.
  - 목적함수 FID → (우리 태스크) gaze 오차(angular error)/IoU로 교체.
  - `efficiency_predictor`(평균 group)를 **실제 하드웨어 비용 모델**(레이어별 cycle/DSP)로 교체하면 hardware-aware NAS-style 그룹/비트 할당 탐색 프레임으로 재사용 가능(우리 algo2fpga DSE 흐름과 정합).
- 단, Q-DiT처럼 후보마다 full inference를 돌리면 비용이 큼 → FPGA용으로는 proxy metric(소수 샘플 FID/error)나 surrogate predictor 도입 권장(설계 제안).

### (4) 주의점
- Q-DiT는 fake-quant 시뮬레이션이라 실속도/자원 수치를 직접 주지 않음 → 우리 가속기 비교 시 **실제 정수 데이터패스로 재구현** 필요.
- group dequant 곱셈을 FPGA에서 줄이려면 **per-group scale을 power-of-two로 제약**(shift로 대체)하는 변형을 고려할 수 있음(Q-DiT 코드엔 없음, 우리 측 확장 아이디어).

---

## 9. 근거 표기 규칙

- **코드 확인**: 본 문서의 `file:line` 인용은 모두 실제 소스에서 직접 확인한 사실이다.
  - 예: GPTQ Hessian 누적(gptq.py:235-238), group-wise 분기(quant.py:19-45), 진화 탐색 목적함수=FID(evolution.py:193-200), W4A8 기본 설정(quant_main.sh:3-6).
- **추정**: 명시적으로 "추정"으로 표기.
  - 예: GPTQ groupsize 일괄 전달이 블록별 group을 완전 반영 못 할 가능성(7절 한계), outlier 흔적이 Atom 차용 잔재라는 해석(3.5절), 우리 프로젝트 맥락 가정(8절).
- **확인 불가**: 코드에 근거가 없어 검증 못 한 항목.
  - 예: `quant_type=fp` 경로의 `quantize_fp4`/`dequantize_fp4` import 부재로 실제 동작 여부(3.2절), 논문 본문 수치/표(코드에 미포함).
- **명시적 정정**: 작업 지시의 "outlier 채널 처리"는 코드상 미구현 — `outlier.py`는 activation 통계 수집기이며 outlier 분리 로직은 없다(3.5절).
