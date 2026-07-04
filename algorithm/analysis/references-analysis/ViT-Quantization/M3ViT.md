# M3ViT 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/M3ViT`
> 분석 방법: Glob/Grep/Read 기반 자체 핵심 소스 정독, 라인 근거(파일:라인) 표기.
> **중요 사전 결론**: 이 repo도 "ViT-Quantization" 디렉토리에 위치하지만 **모델 양자화(PTQ/QAT, INT8, fake-quant, observer/qconfig) 코드는 자체 소스에 존재하지 않는다.** 핵심은 **MoE(Mixture-of-Experts) ViT 기반 멀티태스크 학습 + 모델-가속기 코디자인(연산 reordering)**. (근거: 아래 "양자화 유무 확인" 절. Grep의 `fp16` 매칭은 mmcv `auto_fp16` 데코레이터로 양자화 아님 — `models/decoder_head.py:6,163`.)

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문**: *M³ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design* (NeurIPS 2022). (근거: `README.md:1`)
- **목적**: 멀티태스크 학습(MTL)의 두 병목 해결 — (i) 학습 시 태스크 간 gradient 충돌, (ii) 추론 시 단일 태스크 실행에도 전체 모델을 활성화하는 비효율. (근거: `README.md:8`)
- **핵심 아이디어**:
  1. ViT 백본의 MLP를 **MoE 레이어**로 교체, 태스크별 전문가(expert)를 sparse하게 활성화 → 파라미터 공간을 분리해 태스크 충돌 완화. (README.md:10)
  2. 추론 시 관심 태스크의 sparse "expert pathway"만 활성화 → "all tasks activated"를 피함. (README.md:8,10)
  3. **하드웨어 코디자인**: MoE gating이 토큰을 expert별 큐로 라우팅하고, **double-buffered 연산 흐름**(한 expert 계산 중 다음 expert 파라미터 로드)으로 zero-overhead 태스크 전환. (README.md:19-24)
- **두 가지 MoE gate 설계** (README.md:12-13):
  - **Multi-gate**: 태스크마다 자신의 router(gate)를 가짐.
  - **Task-conditioned**: 모든 태스크가 router를 공유하되, task-specific embedding을 token embedding에 concat하여 router 입력으로 사용.

---

## 2. 디렉토리 구조 (자체 소스 + 제외)

### 자체 핵심 소스 (분석 대상)
```
M3ViT/
├── train_fastmoe.py          # ★ ViT/ViT-MoE 분산 학습 엔트리 (README 권장 경로)
├── train_vit.py              # ViT 학습 변형
├── main.py                   # baseline(resnet/hrnet) 학습 엔트리
├── models/
│   ├── vision_transformer_moe.py  # ★ MoE를 끼운 ViT 백본 (Block/Attention/VisionTransformerMoE)
│   ├── custom_moe_layer.py        # ★ FMoETransformerMLP (fastmoe FMoE 래핑, expert+gate 조립/forward)
│   ├── gate_funs/
│   │   ├── noisy_gate.py           # ★ NoisyGate (GShard/Switch식 noisy top-k + load-balancing loss)
│   │   └── noisy_gate_vmoe.py      # ★ NoisyGate_VMoE (V-MoE식, sem/subimage 정규화 옵션)
│   ├── vit_up_head.py             # ★ ViT 디코더 업샘플 헤드(분할용 PUP head)
│   ├── vit.py / vits_gate.py      # ViT 변형/게이트 ViT
│   ├── decoder_head.py            # BaseDecodeHead (mmcv 기반)
│   ├── models.py / model_utils.py # MTL 래퍼/유틸
│   └── (baseline) resnet, hrnet, mti_net, padnet, papnet, mtan, nddr_cnn, cross_stitch ...
├── utils/
│   ├── moe_utils.py          # ★ load-balancing loss 수집, expert pruning, MoE 체크포인트 I/O
│   ├── common_config.py, config.py, mypath.py, helpers.py ...
├── losses/
│   ├── loss_functions.py     # 태스크별 loss (SoftMaxwithLoss 등)
│   └── loss_schemes.py       # 멀티태스크 loss 결합
├── data/                     # NYUD, PASCAL-Context, Cityscapes 데이터셋
├── evaluation/               # 태스크별 평가 (semseg/depth/normals/edge/sal/human_parts)
├── train/train_utils.py      # 학습 루프 유틸
└── configs/                  # YAML 설정 (nyud/vit, nyud/vit_moe, cityscapes ...)
```

### 제외 (지침에 따라 이름만)
- `.git/`, `**/__pycache__/`, `*.pyc` : 버전관리/캐시
- `evaluation/seism/` : 외부 edge 평가 repo (seism)
- `models/pretrained_models/`, `*.pth` : 사전학습 가중치(대용량, 외부)
- `data/db_info/*.npy`, `*.json` : 데이터셋 메타(소형이나 코드 아님)
- `fmoe` 패키지 자체 : 외부 라이브러리(fastmoe), import만 함 (`from fmoe...`)

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `models/gate_funs/noisy_gate.py` — NoisyGate (GShard/Switch식) ★ 가장 중요

`fmoe.gates.base_gate.BaseGate` 상속. (noisy_gate.py:4,14)

#### 구성 (noisy_gate.py:14-46)
- `w_gate`(d_model × tot_expert): gating logits 가중치. `w_noise`: 학습 가능한 노이즈 표준편차 가중치. 둘 다 0 초기화 후 kaiming. (noisy_gate.py:18-23, 48-53)
- `top_k`(기본 2), `softplus`, `softmax(dim=1)`, `noise_epsilon=1e-2`. (noisy_gate.py:34-39)
- `return_decoupled_activation` 시 보조 게이트 `w_gate_aux`/`w_noise_aux` 추가 (noisy_gate.py:25-32) — pruning용 활성화 분리.

#### `forward` (noisy_gate.py:136-218) — Noisy Top-K Gating
1. `clean_logits = inp @ w_gate` (noisy_gate.py:142)
2. `noise_stddev = (softplus(inp @ w_noise) + eps) * self.training` — 학습 중에만 노이즈. (noisy_gate.py:143-144)
3. `noisy_logits = clean_logits + randn * noise_stddev` (noisy_gate.py:149)
4. top-(k+1) 추출 → 상위 k를 softmax하여 `top_k_gates` (noisy_gate.py:178-184)
5. `gates = scatter(top_k_indices, top_k_gates)` — sparse one-hot 가중치 (noisy_gate.py:186-187)
6. **load-balancing loss** (학습 시, noisy_gate.py:189-200):
   - `importance = gates.sum(0)` (각 expert의 게이트 가중치 합)
   - `load = _prob_in_top_k(...)` (노이즈 하에서 top-k에 들 확률의 합, 미분 가능; noisy_gate.py:69-112)
   - `loss = cv_squared(importance) + cv_squared(load)` (noisy_gate.py:200)
   - `cv_squared(x) = var(x)/(mean(x)^2 + eps)` — 분포 균등화 유도 (noisy_gate.py:114-128)
7. `set_loss(loss)`로 누적, `activation` 저장(pruning/분석용) → `(top_k_indices, top_k_gates)` 반환 (noisy_gate.py:204-218).

#### 보조 메서드
- `_prob_in_top_k` (noisy_gate.py:69-112): clean/noisy logits와 threshold로, 정규분포 CDF를 써서 "top-k 포함 확률"을 미분 가능하게 계산 → load를 backprop 가능하게.
- `get_activation`/`has_activation` (noisy_gate.py:220-228): 게이트 활성화 캐시 접근(expert pruning에서 사용).

### 3.2 `models/gate_funs/noisy_gate_vmoe.py` — NoisyGate_VMoE ★

V-MoE 스타일. NoisyGate와 골격은 같으나 차이점:
- 노이즈 std가 학습 파라미터가 아니라 **상수 비율** `noise_std / tot_expert` (noisy_gate_vmoe.py:182-184). `w_noise` 없음.
- **softmax를 top-k 추출 전에 logits에 적용** (noisy_gate_vmoe.py:258) — gate score가 softmax 확률.
- 동일한 load-balancing loss(`cv_squared(importance)+cv_squared(load)`, noisy_gate_vmoe.py:270-282).
- **추가 정규화 옵션** (M3ViT 고유):
  - `regu_sem` (noisy_gate_vmoe.py:186-197): gating logits로부터 semantic class를 예측하는 보조 head(`self.head`, noisy_gate_vmoe.py:49)를 두고 GT semseg와 `SoftMaxwithLoss`로 `semregu_loss` 계산 → expert 선택을 의미정보와 정렬.
  - `regu_subimage` (noisy_gate_vmoe.py:199-222): 서브이미지(5×5 토큰 블록) 내 top-2 expert 분포에 KL-divergence를 걸어 공간적으로 일관된 라우팅 유도.
  - `regu_experts_fromtask` (noisy_gate_vmoe.py:54-60, 178-180): 태스크별로 expert 부분집합(`start_experts_id`~+num_experts_pertask)만 사용.

### 3.3 `models/custom_moe_layer.py` — FMoETransformerMLP ★

`fmoe.layers.FMoE` 상속, Transformer MLP를 MoE로 대체. (custom_moe_layer.py:6,66)

#### `_Expert` (custom_moe_layer.py:24-44)
- 2개의 `FMoELinear`(htoh4, h4toh)로 한 worker 내 여러 expert를 묶어 계산. forward: `h4toh(act(htoh4(x)))` (custom_moe_layer.py:41-44).

#### `FMoETransformerMLP.__init__` (custom_moe_layer.py:73-159)
- 다수 옵션: `num_expert`, `d_model`, `d_hidden`, `top_k`, `multi_gate`, `regu_experts_fromtask`, `num_experts_pertask`, `num_tasks`, `regu_sem`, `sem_force`, `regu_subimage`, `expert_prune`, `prune_threshold`. (custom_moe_layer.py:74-97)
- `sem_force` 시 고정 force_id 매핑 정의(semantic class→expert 강제, custom_moe_layer.py:112-113).
- gate 조립 (custom_moe_layer.py:132-158): `NoisyGate` 또는 `NoisyGate_VMoE`. `multi_gate`이면 `(d_gate - d_model)`개의 게이트를 `ModuleList`로(태스크별 router). 아니면 단일 게이트. `gate_task_specific_dim>=0`이면 task embedding concat을 위해 `d_gate = d_model + gate_task_specific_dim`.

#### `forward` (custom_moe_layer.py:161-181)
- `inp`를 `(-1, d_model)`로 reshape, gate 입력 별도. (custom_moe_layer.py:170-174)
- task-conditioned 모드: `task_specific_feature`를 gate_inp에 concat (custom_moe_layer.py:176-179).
- `forward_moe(...)` 호출 후 원형 복원.

#### `forward_moe` (custom_moe_layer.py:184-314) — MoE 라우팅 본체
1. 분산 통신(world_size>1) / slice 처리 (custom_moe_layer.py:197-211).
2. gate 호출: multi_gate면 `self.gate[task_id](gate_inp)`, 아니면 `self.gate(gate_inp, task_id, sem)` → `(gate_top_k_idx, gate_score)` (custom_moe_layer.py:213-217).
3. `expert_prune` 시 `gate_score>threshold`만 유지 (custom_moe_layer.py:219-222).
4. `sem_force` 시 semantic GT로 expert 인덱스 강제 치환 (custom_moe_layer.py:223-233).
5. `regu_experts_fromtask` 시 인덱스에 `start_experts_id[task_id]` 오프셋 (custom_moe_layer.py:236-238).
6. **`_fmoe_general_global_forward`** (fastmoe): 토큰을 expert별로 scatter→expert 계산→gather (custom_moe_layer.py:255-257).
7. 출력 `(B*L, top_k, dim)` 뷰 후 **gate_score로 가중합**: `bmm(gate_score, expert_outputs)` (custom_moe_layer.py:283-297).

### 3.4 `models/vision_transformer_moe.py` — MoE ViT 백본 ★

#### `Attention` (vision_transformer_moe.py:194-224)
- 표준 MHSA: `qkv = Linear(dim, 3*dim)` → reshape/permute → `attn = (q@k^T)*scale` → softmax → `attn@v` → proj. (vision_transformer_moe.py:205-223). LayerNorm은 Block에서 처리.

#### `Block` (vision_transformer_moe.py:226-283) ★
- `norm1→attn→residual`, `norm2→mlp→residual` 표준 ViT 블록.
- **MoE 토글** (vision_transformer_moe.py:247-271): `moe=True`이면 MLP를 `FMoETransformerMLP`로 교체(gate type은 noisy/noisy_vmoe). 아니면 일반 `Mlp`.
- forward에서 MoE면 `mlp(norm2(x), gate_inp, task_id, task_specific_feature, sem)` 호출 (vision_transformer_moe.py:275-283).

#### `VisionTransformerMoE` (vision_transformer_moe.py:350-566) ★
- patch_embed(Conv2d patchify) + cls_token + pos_embed (vision_transformer_moe.py:400-412).
- **블록 배치 규칙** (vision_transformer_moe.py:425-437): `i % 2 == 0`이면 일반 Block, 홀수 인덱스이면 **MoE Block** → **격층(every other layer)으로 MoE 삽입**. (V-MoE 관행)
- task-conditioned: `gate_task_represent`(new_Mlp)로 task one-hot→embedding 생성 (vision_transformer_moe.py:420-423, 546-550).
- `forward_features` (vision_transformer_moe.py:537-560): 블록 순회, MoE 블록엔 `(gate_inp, task_id, task_specific_feature, sem)` 전달, 각 stage feature를 `outs`에 누적 → tuple 반환(다단계 디코더용).
- `regu_sem/sem_force` 시 `get_groundtruth_sem`으로 patch 단위 GT semantic 생성(다수결, vision_transformer_moe.py:519-535).
- 사전학습 pos_embed/가중치 로딩(`load_pretrained*`, vision_transformer_moe.py:481-490).

### 3.5 `models/vit_up_head.py` — VisionTransformerUpHead (분할 디코더)
- `BaseDecodeHead`(mmcv 기반) 상속, ViT 토큰 시퀀스를 2D로 reshape 후 **PUP(Progressive UPsampling)** 식으로 conv+bilinear 업샘플 (vit_up_head.py:149-224).
- `num_conv=4 / num_upsampe_layer=4`: 4단계 conv+syncbn+relu+×2 bilinear로 점진 업샘플 (vit_up_head.py:182-218).
- `multi_level`/`tam`(task affinity) 옵션으로 중간 레벨 출력 반환 (vit_up_head.py:189-218).

### 3.6 `utils/moe_utils.py` — MoE 손실 수집 / pruning / 체크포인트 ★
- **`collect_noisy_gating_loss(model, weight)`** (moe_utils.py:105-111): 모델 내 모든 NoisyGate/NoisyGate_VMoE의 `get_loss()`를 합산×weight → 전체 loss에 더해질 **load-balancing 항** 집계.
- `collect_semregu_loss` / `collect_regu_subimage_loss` (moe_utils.py:114-128): semantic/subimage 정규화 손실 집계.
- `collect_moe_activation` (moe_utils.py:130-152): 게이트 활성화 수집(pruning 통계).
- `prune_moe_experts` (moe_utils.py:174-204): train_loader로 게이트 활성화 평균을 누적(`feature_avger`)→ top-`prune_num` expert 선택→`set_moe_mask`로 게이트 `select_idx` 설정 → **expert pruning**(추론 시 일부 expert만).
- 분산 expert 체크포인트 I/O: `collect_moe_model_state_dict`/`save_moe_model_to_dir`/`read_specific_group_experts` (moe_utils.py:21-102) — rank별로 expert 파라미터를 분산 저장/수집.

### 3.7 `train_fastmoe.py` — 학습 엔트리
- 풍부한 CLI: `--moe_gate_type`, `--moe_experts`, `--moe_top_k`, `--multi_gate`, `--gate_task_specific_dim`, `--backbone_random_init`, `--vmoe_noisy_std`, `--moe_mlp_ratio` 등으로 MoE 설정을 config에 주입 (train_fastmoe.py:160-190). 분산학습(`torch.distributed.launch`, train_fastmoe.py:192-197).

### 3.8 양자화 유무 확인 (코드 근거)
- `quant|int8|fake_quant|observer|qconfig` Grep: 자체 모델/학습 소스에 **매칭 없음**.
- `fp16` 매칭은 `models/decoder_head.py:6`(`from mmcv.runner import auto_fp16, force_fp32`), `:85`(`self.fp16_enabled=False`), `:163`(`@auto_fp16()`) — 이는 **mmcv의 혼합정밀(AMP) 데코레이터**로 모델 양자화가 아님. 기타 매칭(`custom_transforms.py`, `evaluate_utils.py`, `custom_collate.py`, `helpers.py`)도 이미지 quantize/형변환 등 비양자화.
- → **결론: 본 repo에 모델 양자화 코드 없음. 효율 핵심은 MoE sparse activation + 하드웨어 연산 reordering(코디자인).**

---

## 4. 알고리즘 / 수식

### 4.1 Noisy Top-K Gating (noisy_gate.py:142-187)
```
H(x)        = x·W_gate                              # clean logits
σ(x)        = softplus(x·W_noise) + ε               # 학습 시 노이즈 std (NoisyGate)
G_noisy(x)  = H(x) + N(0,1) ⊙ σ(x)                  # noisy logits
TopK_idx, TopK_val = top_k( G_noisy(x) )
g           = softmax( TopK_val )                   # 선택된 k expert의 게이트 가중치
y           = Σ_{i∈TopK} g_i · Expert_i(x)          # 가중합 (custom_moe_layer.py:283-297)
```
V-MoE 변형(noisy_gate_vmoe.py): `σ = noise_std / tot_expert` 상수, softmax를 top-k 추출 전 적용.

### 4.2 Load-Balancing Loss (noisy_gate.py:189-200)
```
importance_e = Σ_tokens g_e            # expert e의 게이트 가중치 합
load_e       = Σ_tokens P(expert e ∈ top-k)   # _prob_in_top_k, 미분 가능
cv_squared(z)= Var(z) / (Mean(z)^2 + ε)
L_balance    = cv_squared(importance) + cv_squared(load)
```
전체 손실: `L = Σ_task L_task + λ · Σ_gates L_balance` (집계는 `collect_noisy_gating_loss`, moe_utils.py:105-111). `cv_squared`는 expert 사용량 분포를 균등하게 만들어 특정 expert 쏠림(라우팅 collapse)을 방지.

### 4.3 Top-K MoE 연산량
한 토큰당 활성화되는 expert는 `top_k`개(전체 `num_expert`가 아님) → 추론 FLOPs는 expert 수와 무관하게 top_k에 비례. 이것이 "sparse activation"의 효율 근거. (custom_moe_layer.py:255-297)

### 4.4 모델-가속기 코디자인 (개념, README.md:19-24)
gating이 토큰을 expert 큐로 라우팅 → **double-buffering**: 현재 expert가 자기 큐 토큰을 계산하는 동안 다음 expert의 파라미터를 미리 로드 → 버퍼 스왑으로 expert 간 zero-overhead 전환, expert 수에 무관하게 스케일. (코드 자체에는 HW RTL 없음 — 논문/설명 기반, **추정**: 본 repo는 SW 측 학습/추론만 포함.)

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: NYUD-v2, PASCAL-Context (멀티태스크: semseg/depth/normals/edge/human_parts/saliency), Cityscapes. (`data/nyud.py`, `data/pascal_context.py`, `data/cityscapes.py`, configs)
- **baseline 학습** (README.md:67-69):
  ```
  python main.py --config_env configs/env.yml --config_exp configs/$DATASET/$MODEL.yml
  ```
- **ViT-MoE 학습** (README.md:72-77, 분산 2-GPU 예):
  ```
  python -m torch.distributed.launch --nproc_per_node=2 train_fastmoe.py \
    --config_env configs/env.yml --config_exp configs/$DATASET/vit_moe/$MODEL.yml \
    --moe_gate_type "noisy_vmoe" --moe_experts 16 --moe_top_k 4 \
    --multi_gate True --moe_mlp_ratio 1 --vmoe_noisy_std 0 ...
  ```
- **평가**: 학습 말기 best model 평가, `eval_final_10_epochs_only: True`로 마지막 10 epoch만 평가 가속 (README.md:78-83). 태스크별 평가는 `evaluation/eval_*.py`. edge 평가는 외부 `seism` 필요 (README.md:57).
- **의존 설치**: fastmoe(특정 커밋 `4edeccd`) 별도 빌드 필요 (README.md:42-50).

---

## 6. 의존성
- `torch`, `torch.distributed`(분산/expert 병렬)
- **fastmoe** (`fmoe.layers.FMoE`, `fmoe.linear.FMoELinear`, `fmoe.functions`, `fmoe.gates.base_gate.BaseGate`) — **핵심 강결합** (custom_moe_layer.py:6-17, noisy_gate.py:4).
- `dm-tree`(`tree.map_structure`로 중첩 텐서 처리, custom_moe_layer.py:9,190)
- `timm`(layers/init), `mmcv`(`build_norm_layer`, `auto_fp16` 등; vit_up_head.py:11, decoder_head.py:6)
- `thop`(FLOPs), `easydict/pyyaml`(config), `numpy`
- 코드 베이스 출처: ASTMT, Multi-Task-Learning-PyTorch (README.md:85-86).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **태스크 충돌 완화 + 추론 sparse 활성화**: MoE로 파라미터 공간 분리, top-k만 활성화 → MTL 확장성. (custom_moe_layer.py:255-297)
- **풍부한 라우팅 정규화**: load-balancing(cv_squared) + semantic/subimage 정규화 + task-from-task expert 분할 → collapse 방지·구조화. (noisy_gate*.py)
- 모델-가속기 코디자인 관점(reordering/double-buffer) 제시 — HW 친화 사고. (README.md:19-24)

### 한계 / 리스크
- **fastmoe 특정 커밋 강결합**: `fmoe` 내부 scatter/gather에 의존, 재현성/이식성 부담 (README.md:48). HW 매핑 시 fastmoe의 동적 디스패치를 직접 구현해야 함.
- **동적 라우팅(데이터 의존 top-k)**: 토큰마다 활성 expert가 달라짐 → 메모리 접근 패턴/연산량이 입력 의존적, 정적 스케줄 HW에 불리.
- 코드에 다수 디버그/하드코딩(`print`, force_id 상수, `set_trace`, magic numbers like 30×40 토큰) → 정리 필요 (noisy_gate_vmoe.py:200-222).
- 양자화 코드 부재 → 저비트 배포는 별도 작업.
- semseg/subimage 정규화는 입력 해상도/태스크에 하드코딩된 가정 존재(예: 패치 30×40, noisy_gate_vmoe.py:203).

---

## 8. 우리 프로젝트 관점 시사점 (FPGA 가속기 HG-PIPE 계열 + XR 시선추적)

> 전제: "ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 **추정**. FPGA 친화도 관점.

- **MoE 동적 라우팅은 FPGA(특히 HG-PIPE식 정적 파이프라인)에 부담.** top-k expert가 토큰별로 달라지는 data-dependent dispatch는, 레이어별 고정 데이터패스를 가정하는 HG-PIPE 파이프라인과 상충. scatter/gather·가변 토큰 큐는 동적 버퍼링/제어 오버헤드를 만든다. (근거: custom_moe_layer.py:255-297의 동적 forward)
- **단, M3ViT의 코디자인 통찰(double-buffer expert 로딩, 연산 reordering)은 참고 가치.** expert 파라미터를 미리 로드하며 계산을 겹치는 기법은, FPGA의 weight prefetch / ping-pong BRAM 버퍼 설계와 직접 대응됨 → HG-PIPE에 "weight streaming + double buffer" 아이디어로 이식 가능 (추정). (README.md:19-24)
- **표준 MHSA/LayerNorm/GELU 데이터패스 존재**(vision_transformer_moe.py:194-224, 226-283): 이 부분은 HG-PIPE의 attention/softmax 엔진 검증 워크로드로 적합. MoE를 제외한 ViT backbone(짝수 레이어, vision_transformer_moe.py:425-428)만 떼어내면 일반 ViT 가속기 타깃으로 쓸 수 있음.
- **Expert pruning**(moe_utils.py:174-204)으로 추론 시 expert 수를 줄이면 동적성 일부 완화 가능 → "정적으로 top-N expert 고정" 변형이면 FPGA 매핑 난도 하락 (추정).
- **XR 시선추적 관점**: 시선추적은 보통 단일/소수 태스크·저지연. M3ViT의 MTL 전체 구조는 과해 보이나, "태스크 조건부 sparse pathway"는 XR에서 시선/제스처/장면 등 멀티모달 태스크를 한 백본으로 전환하며 처리할 때 개념적 시사점. 다만 실시간 FPGA 추론에는 동적 라우팅을 정적화하는 추가 설계가 필수 (추정).
- **양자화 적용 시**: 본 repo 미포함. MoE에서 expert별 weight 양자화는 가능하나, gating logits(softmax/topk)는 정밀도 민감 → gate는 고정밀, expert는 저비트 혼합 양자화 전략이 합리적일 것 (추정).

---

## 9. 근거 표기 / 확인 불가 항목
- **확인됨(코드 직접 근거)**: NoisyGate/NoisyGate_VMoE의 noisy top-k + load-balancing loss, FMoETransformerMLP의 라우팅 forward, MoE ViT의 격층 삽입, load-balancing/semantic/subimage 정규화 손실 수집, expert pruning, 양자화 코드 부재 — 모두 위 파일:라인 근거.
- **추정**: 모델-가속기 코디자인(double-buffer/reordering)은 README 설명 기반이며 **본 repo에 RTL/HW 코드는 없음**(SW 학습/추론만). FPGA 적합성/양자화 전략 평가는 우리 프로젝트 맥락 해석.
- **확인 불가**: fastmoe(`fmoe`) 내부 `_fmoe_general_global_forward`/`FMoELinear` 구현은 외부 패키지로 본 분석 범위 밖. 논문상 정확도/지연 수치는 로컬 재현 불가.
