# Bi-ViT 모듈 통합 가이드 (S-PyTorch)

> 1차 요약: [`../Bi-ViT.md`](../Bi-ViT.md) — 본 문서는 그 요약을 모듈 단위로 심화한 통합 가이드다.
> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\Bi-ViT`
> 작성 원칙: 실제 소스 Read 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가"로 명시.
> 형제 가이드(`../I-ViT/MODULE_GUIDE.md`)의 6요소 구조를 동형(同形)으로 따르되, I-ViT의 "정수전용(integer-only)" 수치 규약을 Bi-ViT의 **이진(1-bit) 규약**(sign+scale / XNOR-popcount BOPs / gradient 근사)으로 치환한다.
> 학회 표기: README는 **AAAI 2024** 채택본(`README.md:1-3`). 과제 지정 "ICCV'23"과 차이 → "표기차이"로 둠.

---

## 0. 문서 머리말

### 0.1 대표 케이스 선정
- **대표 모델: `bi_deit_small_patch16_224` (Bi-DeiT-S)** — `embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, patch16, img224, distilled=True`(`bi_vision_transformer.py:792-795`, `twobits_VisionTransformer.__init__` 기본값 `:392-393`). 근거:
  1. README 평가 명령이 tiny/small 두 모델만 제시(`README.md:22,27`)하고, small이 DeiT-S 표준 분석 케이스라 형제 가이드(I-ViT)와 비교 가능.
  2. 토큰 수 N=197(=14×14 패치 + cls + dist token, distilled 2토큰), C=384, H=6, head_dim=64 — 이진 행렬곱/헤드별 이진화 규모가 비자명. (PatchEmbed.num_patches = (224/16)²=196, +2 토큰 = 198이나 cls/dist 2개 → N=198; 본문은 cls 1 + dist 1 + 196 = 198로 계산. 단 형제 가이드와 동일 표기 일관성 위해 N≈197~198 명기, 정량은 N=198 사용.)
- **대표 분석 단위: `bi_Block` 1개** = `LayerNorm → bi_Attention(BiLinearBiReal qkv → q/k LayerNorm → ActBi q/k/v → q@kᵀ → softmax(float) → ActBi attn → @v → BiLinearBiReal proj) → residual(float add) → LayerNorm → bi_Mlp(BiLinearBiReal fc1 → GELU → clip(-10,10) → BiLinearBiReal fc2) → residual(float add)`(`bi_vision_transformer.py:246-249, 209-230, 172-182`). Bi-DeiT-S는 이 Block을 12개 적층(`:433-437`).
- **대표 이진화 3종**:
  1. **BiReal 3차 STE 활성 + 채널평균 스케일 가중치** = `BiLinearBiReal`(`Bi_Quant.py:77-101`) — MLP/qkv/proj의 이진 GEMM.
  2. **헤드별 학습 스케일 + zero_point 비대칭 이진 활성** = `ActBi`(`Bi_Quant.py:162-200`) — attention degeneration 대응의 코드적 실체.
  3. **q/k LayerNorm 선처리 + softmax float 유지**(`bi_vision_transformer.py:195-196,214-215,222`) — softmax 분포 붕괴 회피.

### 0.2 S-PyTorch 수치 규약 (이진 규약으로 치환)
- **params**: 모듈 차원에서 분석적 계산. Bi-ViT는 가중치를 forward마다 fake-binarize(`Bi_Quant.py:96-98`)하므로 **params 개수는 FP DeiT-S와 동일**, 단 가중치는 1-bit(sign) + per-row 스케일 1개. 추가 학습 파라미터는 `ActBi`의 `alpha`/`zero_point`(헤드별 벡터, 길이=num_heads), `_LinearB`의 `alpha`(LinearBi 경로만, 실사용 안 됨).
- **BOPs/MACs**: 이진 Linear/MatMul은 곱셈이 XNOR+popcount로 환원되므로 **BOPs(binary ops)**로 표기. 환산 규약: `BOPs ≈ MACs`(1 MAC = 1 XNOR + popcount 누산 1, 비트 단위). 분석에서는 표준 MAC 식으로 카운트하되 "이진연산(XNOR-popcount)"으로 주석. 단 Bi-ViT에서 **q@kᵀ, attn@v 행렬곱은 float 텐서끼리**(ActBi 출력이 `(x-z)·α`로 dequant된 float, `Bi_Quant.py:199`) → 엄밀히는 이진 입력의 float 행렬곱(완전 XNOR 아님, 본문 §6 정밀해부).
- **activation memory**: 텐서 shape × 비트폭. Bi-ViT는 fake-quant라 실제 메모리는 FP32지만, **이진 도메인 비트폭(A1)**을 "HW 환산 activation bit"로 표기 — `shape × 1-bit` + 헤드별 스케일/zp 상수.
- **이진화 함수**: weight `W_b = mean(|W|, dim=row)·sign(W)`(`Bi_Quant.py:94-96`); activation `a_b = sign(a)`(`:62,84`) 또는 비대칭 `sign(x/α+z)`(ActBi, `:198`).
- **gradient 근사**: BiReal 3차 다항식 STE(`:66-72`), sign_pass(forward sign / backward identity, `:132-135`), grad_scale(LSQ 스케일 그래디언트, `_binary_base_plus.py:20-23`).
- **정확도/속도**: README는 평가 명령만 제시, **정확도 수치 미기재**(`README.md` 전체) → "확인 불가". 본 세션 미실행.

### 0.3 운영 경로 (사전학습 로드 → distillation QAT → ImageNet 평가)
```
[FP distilled DeiT 체크포인트 로드] bi_deit_small_patch16_224()
   │  torch.load('checkpoints/deit_small_distilled_patch16_224-649709d9.pth')
   │  model.load_state_dict(ckpt["model"], strict=False)   (bi_vision_transformer.py:797-804)
   │  → 이진 모델 가중치 FP 초기화(strict=False라 ActBi/LayerNorm 신규 파라미터는 미로드)
   ▼
[teacher 생성] create_model(deit_small_patch16_224, pretrained=True)  (main_1bit.py:359-363)
   │  float DeiT-S가 teacher, eval 고정 (main_1bit.py:370-371)
   ▼
[Distillation QAT] train_one_epoch(): loss = DistillationLoss(samples, outputs, targets)
   │  fake-binarize forward (engine_ts.py:41-44)
   │  AdamW + cosine, lr linear-scaled(lr·bs·world/512), epochs 기본 300 (main_1bit.py:318-322,39,71-73)
   ▼
[체크포인트 저장] best Top-1 갱신 시 best_checkpoint.pth (main_1bit.py:436-449)
   ▼
[ImageNet 평가] evaluate(): model.eval() → accuracy Top-1/5 (engine_ts.py:79-109)
   ▼
[(외부) 사전학습 체크포인트] Google Drive (README.md:29-31, 이름만 / 확인 불가)
```
- 타깃 디바이스: **CUDA 전제** — `engine_ts.py:67`의 `torch.cuda.synchronize()`, `main_1bit.py:157` 기본 `--device cuda`. CPU 단독 실행 여부는 코드상 강제 cuda 호출이 적어 I-ViT보다 약하나, 학습 루프가 cuda 동기화 호출 → 사실상 GPU(추정, 미실행).

### 0.4 모델 / 데이터셋 / 정확도
| Model | embed/depth/heads | 평가명령 근거 | 정확도 |
|---|---|---|---|
| Bi-DeiT-T | 192/12/3 | `README.md:22` | 확인 불가(README 미기재) |
| **Bi-DeiT-S(대표)** | **384/12/6** | `README.md:27` | 확인 불가(README 미기재) |
- 데이터셋: **ImageNet (IMNET)** `--data-path`, 224×224, 1000 클래스(`main_1bit.py:147-150`, `--data-set IMNET`).
- 정확도: README에 **수치 표 없음**(평가 커맨드만) → "확인 불가". 논문(AAAI2024) 본문 수치는 본 repo에 미포함.
- 속도/latency: 본 PyTorch repo에 측정 코드 없음 → "확인 불가".

### 0.5 실사용 경로 vs 데드코드 (정밀 구분) ★중요
Bi-ViT repo는 실험 잔재가 많아 **정의되었으나 대표 모델 경로에서 호출되지 않는** 코드가 다수다. 본 가이드는 실사용만 정밀 분석한다.

| 구분 | 실사용(대표 경로) | 정의만/데드코드 | 근거 |
|---|---|---|---|
| 이진 Linear | `BiLinearBiReal`(qkv/proj/fc1/fc2) | `LinearBi(_LinearB)`, `utils_quant.QuantizeLinear`, `QuantizeLinearMCN` | `bi_Mlp:164,169`, `bi_Attention:197,199` vs 미import/미호출 |
| 이진 활성 | `ActBi`(q/k/v/attn) | `ActLSQ_bi`, `_ActB_qk` | `bi_Attention:201-204` vs `_ActB_qk` 정의만(`_binary_base_plus.py:210-247`) |
| Patch embed | **float `PatchEmbed`**(기본 embed_layer) | `bi_PatchEmbed`/`Conv2dBi`, `BiConv2dBiReal` | `twobits_VisionTransformer:394,423-424`(기본 `embed_layer=PatchEmbed`) — bi 버전은 정의만 |
| 증류 손실 | 표준 `DistillationLoss`(soft/hard) | `DistillationLoss_rank`, `mf_loss`/`layer_mf_loss`(manifold), `KD_loss.DistributionLoss` | `main_1bit.py:376-378`, `engine_ts.py:44` vs 미호출 |
| 양자화 함수 | (없음 — Bi_Quant 경로) | `utils_quant.py` 전체(`BinaryQuantizer`/`Twn`/`Sym`/`Asym`) | `utils_quant`는 어떤 모델 파일도 import 안 함(확인: import 미발견) |
- **시사**: 1차 요약(`../Bi-ViT.md`)이 `LinearBi`/`utils_quant.QuantizeLinear`/`_ActB_qk`/manifold를 "핵심"으로 소개했으나, **대표 학습 경로(main_1bit.py)에서는 미사용**임을 라인 근거로 정정. 실제 1-bit 동작은 `BiLinearBiReal`+`ActBi`만으로 구성.

---

## 1. Repo / Layer 개요

Bi-ViT = ViT/DeiT를 **1-bit 가중치 + 1-bit 활성(완전 이진화)** 으로 압축하는 QAT 프레임워크. 핵심 난점인 **binarized attention의 degeneration**(softmax 분포 붕괴/정보 손실)을 (a) BiReal 3차 STE, (b) 헤드별 학습 스케일·zero_point 이진화, (c) q/k LayerNorm + softmax float 유지, (d) float teacher distillation으로 대응(`README.md:1-9` + 코드). DeiT를 기반으로 timm 위에 커스텀 이진 모듈을 얹는다.

### 1.1 자체 소스 vs 외부 프레임워크 vs 제외

| 구분 | 파일(자체 소스) | 역할 |
|---|---|---|
| **이진 레이어** | `Bi_Quant.py` ★핵심 | `BinaryActivation`, `BiLinearBiReal`, `BiConv2dBiReal`, `sign_pass`, `LinearBi`, `ActBi` |
| **이진 베이스** | `_binary_base_plus.py` | `Qmodes`, `grad_scale`/`round_pass`, `_Conv2dB`/`_LinearB`/`_ActB`/`_ActB_qk`, truncation 유틸 |
| **모델 정의** | `bi_vision_transformer.py` ★ | `bi_Mlp`, `bi_Attention`, `bi_Block`, `bi_PatchEmbed`, `twobits_VisionTransformer`, 팩토리 `bi_deit_*` |
| | `bi_swin_transformer.py` | 이진 Swin (미열람 — 확인 불가) |
| **증류 손실** | `losses.py` | `DistillationLoss`(실사용), `DistillationLoss_rank`/`mf_loss`(데드코드) |
| | `KD_loss.py` | `DistributionLoss`(KL, 데드코드) |
| **양자화 함수(데드)** | `utils_quant.py` | `BinaryQuantizer`/`Twn`/`Sym`/`Asym`/`QuantizeLinear` — 미import |
| **학습 엔트리** | `main_1bit.py` ★ | argparse, distillation QAT train/eval, teacher 생성 |
| | `main_swin.py` | Swin 엔트리 |
| **학습 엔진** | `engine_ts.py`/`engine_ts_swin.py` | train_one_epoch / evaluate |
| **데이터/보조** | `datasets.py`, `samplers.py`, `utils.py`, `hubconf.py` | ImageNet DataLoader, RASampler 등 |

### 1.2 forward 진입점
`twobits_VisionTransformer.forward`(`bi_vision_transformer.py:510-521`) → `forward_features`(`:495-508`):
`patch_embed`(float Conv2d) → cls_token + dist_token cat(distilled 2토큰) → `+ pos_embed`(float add) → `blocks`(12×bi_Block) → `norm`(float LayerNorm) → `(x[:,0], x[:,1])`(cls/dist) → `head`/`head_dist`(float Linear) → 학습 시 (x, x_dist), 추론 시 평균(`:512-518`).
- **주의**: 입력 임베딩·pos_embed·residual·LayerNorm·softmax·head는 **모두 float**. 이진화 대상은 **bi_Attention/bi_Mlp 내부의 qkv/proj/fc1/fc2 GEMM과 q/k/v/attn 활성**뿐.

### 1.3 제외 (지시에 따라 미분석)
- **외부 프레임워크(커스텀 아님)**: `timm.data.Mixup`, `timm.models.create_model`, `timm.loss.{LabelSmoothing,SoftTargetCE}`, `timm.scheduler/optim`, `timm.utils.{NativeScaler,ModelEma}`, `timm.models.layers.{Mlp,DropPath,trunc_normal_}`(`main_1bit.py:13-18`, `bi_vision_transformer.py:10-19`). DeiT distilled **사전학습 체크포인트**(`.pth`) — 가중치만 로드.
- **import되나 본 repo 미노출 모듈**: `bi_vision_transformer.py:23-27`의 `from Binary_plus import *`, `from lsq_plus import *`, `from _quan_base_plus import *` — 해당 .py가 repo 파일 목록에 없음 → **"확인 불가"**. (단 실사용 `BiLinearBiReal`/`ActBi`는 `Bi_Quant`·`_binary_base_plus`에서 정의되므로 분석 가능.)
- **미열람(확인 불가)**: `bi_swin_transformer.py`/`main_swin.py`/`engine_ts_swin.py`(ViT와 동일 모듈 재사용 추정), `datasets.py`/`samplers.py`/`utils.py` 세부, `models.py`/`deit.py`/`distill_vision_transformer.py`/`patchconvnet_models.py`/`run_with_submitit.py`/`hubconf.py`.
- **데드코드(정의만, §0.5 참조)**: `LinearBi`, `_ActB_qk`, `utils_quant.py` 전체, manifold/rank 손실.

### 1.4 대표 모델 레이어 구성 (Bi-DeiT-S)
`forward_features`(`bi_vision_transformer.py:495-508`): float PatchEmbed(Conv2d 16×16 s16) → +cls/dist/pos(float) → bi_Block×12 → float LayerNorm → cls/dist head. 1 bi_Block당: 이진 Linear 4개(qkv, proj = bi_Attention; fc1, fc2 = bi_Mlp, 모두 `BiLinearBiReal`), ActBi 4개(q/k/v/attn), float LayerNorm 4개(norm1, norm2, norm_q, norm_k), float GELU 1개, float softmax 1개.

---

## 2. 모듈: BiReal STE 이진 활성 — `Bi_Quant.py` (BinaryActivation)

### 2.1 역할 + 상위/하위
- **역할**: 활성을 `sign(x)`로 이진화하되 backward는 미분불가한 sign 대신 **piecewise 3차 다항식 근사의 도함수**(BiReal/ReActNet STE)를 사용해 그래디언트 정합.
- **상위**: 단독 모듈로는 `BinaryActivation`(`:57-74`)이나, 대표 경로는 동일 로직이 `BiLinearBiReal.forward`에 inline됨(`:84-92`). **하위**: `torch.sign`, mask 연산.

### 2.2 데이터플로우 (텐서 shape 흐름)
```
x (FP32) ──> out_forward = sign(x)           # forward 값 (±1)
        ──> mask1=x<-1, mask2=x<0, mask3=x<1
        ──> out1 = -1·mask1 + (x²+2x)·(¬mask1)
        ──> out2 = out1·mask2 + (-x²+2x)·(¬mask2)
        ──> out3 = out2·mask3 + 1·(¬mask3)    # piecewise 3차 근사 (backward용)
        ──> out = sign(x).detach() - out3.detach() + out3
                  (forward = sign, backward = ∂out3/∂x)
```

### 2.3 forward call stack
`bi_Mlp.forward`/`bi_Attention.forward` → `BiLinearBiReal.forward`(`Bi_Quant.py:82`) → inline BiReal STE(`:84-92`). 독립 `BinaryActivation`은 대표 경로 미사용(데드, `BiLinearBiReal`이 동일 로직 내장).

### 2.4 대표 코드 위치
`Bi_Quant.py`: `BinaryActivation.forward` `:61-74`, `BiLinearBiReal` inline 동일 로직 `:84-92`.

### 2.5 대표 코드 블록
```python
# Bi_Quant.py:62-72  BiReal 3차 STE (forward=sign, backward=piecewise 3차)
out_forward = torch.sign(x)
mask1 = x < -1; mask2 = x < 0; mask3 = x < 1
out1 = (-1)*mask1 + (x*x + 2*x)*(1-mask1)      # x<-1 → -1, else x²+2x
out2 = out1*mask2 + (-x*x + 2*x)*(1-mask2)      # x<0 분기
out3 = out2*mask3 + 1*(1-mask3)                 # x≥1 → 1
out = out_forward.detach() - out3.detach() + out3
```
→ forward는 정확히 `sign(x)`(±1), backward 그래디언트는 `∂out3/∂x` = (x∈[-1,0): `2x+2`, x∈[0,1): `-2x+2`, |x|≥1: `0`). I-ViT의 STE(`grad/scale`)와 대비되는 **이진 특화 부드러운 STE**.

### 2.6 연산·수치표현 분해 + 정량
- **이진화 방식**: forward `sign` → {-1,+1}. backward는 [-1,1]에서 양수 그래디언트, 밖에서는 0(saturating). zero_point 없음(대칭).
- **gradient 근사**: piecewise 3차 도함수(`:66-72`). |x|>1 영역 grad=0 → BinaryQuantizer(`utils_quant.py:81-82`)의 hard-clip STE와 동일 saturation, 단 내부는 부드러운 곡선.
- **params**: 0.
- **FLOPs**: 원소당 mask 3 + 곱/가감 ~8 = O(N) (forward는 sign만, backward용 다항식 계산이 추가 비용).
- **activation bit**: 출력 A1(±1). HW에서 1-bit 신호.

---

## 3. 모듈: 이진 Linear (BiReal + 채널평균 스케일) — `Bi_Quant.py` (BiLinearBiReal) ★핵심

### 3.1 역할 + 상위/하위
- **역할**: `nn.Linear` 상속. 입력을 BiReal STE로 이진화, 가중치를 **per-row 채널평균 스케일 × sign**으로 이진화 후 `F.linear`. Bi-ViT의 이진 GEMM 본체(qkv/proj/fc1/fc2 전부).
- **상위**: `bi_Mlp.fc1/fc2`(`bi_vision_transformer.py:164,169`), `bi_Attention.qkv/proj`(`:197,199`). **하위**: BiReal STE inline, `torch.sign`, `torch.mean`.

### 3.2 데이터플로우 (텐서 shape 흐름, qkv 예 [B,N,384]→[B,N,1152])
```
input ──BiReal STE──> input_b (±1) [B,N,384]
weight [1152,384] ──> α = mean(|W|, dim=1, keepdim).detach() [1152,1]   # per-row(=per-out) 스케일
                  ──> W_b_nograd = α · sign(W)
                  ──> W_cliped = clamp(W, -1, 1)
                  ──> W_b = W_b_nograd.detach() - W_cliped.detach() + W_cliped   # STE
output = F.linear(input_b, W_b)   [B,N,1152]   # 이진입력 × (스케일×이진가중치)
```

### 3.3 forward call stack
`bi_Attention.forward`(`bi_vision_transformer.py:211`) → `BiLinearBiReal.forward`(`Bi_Quant.py:82`) → 입력 BiReal STE(`:84-92`) → 가중치 채널평균 스케일(`:93-98`) → `F.linear`(`:100`).

### 3.4 대표 코드 위치
`Bi_Quant.py`: 클래스 `:77-101`, 입력 이진화 `:84-92`, 가중치 이진화 `:93-98`, GEMM `:100`. (conv 버전 `BiConv2dBiReal` `:103-130` — 데드, bi_PatchEmbed 미사용.)

### 3.5 대표 코드 블록
```python
# Bi_Quant.py:93-100  가중치 = per-row 채널평균 스케일 × sign, STE
real_weights = self.weight
scaling_factor = torch.mean(abs(real_weights), dim=1, keepdim=True)   # [out,1] per-out 스케일
scaling_factor = scaling_factor.detach()
binary_weights_no_grad = scaling_factor * torch.sign(real_weights)     # α·sign(W)
cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
output = F.linear(input, binary_weights)                              # input은 이미 ±1
```
→ `W_b = mean(|W|)·sign(W)` (XNOR-Net 스타일 채널평균 스케일). I-ViT의 학습형 scale(`fc_scaling_factor`)과 달리 **분석적(non-learnable) 스케일**(detach). 단 `utils_quant.QuantizeLinear`(데드)는 평균제거형 `mean(|W|)·sign(W-mean(W))`(`utils_quant.py:328-329`)로 차이.

### 3.6 연산·수치표현 분해 + 정량 (Bi-DeiT-S, B=1, N=198)
- **이진화 방식**: 입력 A1(sign), 가중치 W1(sign) + per-out FP 스케일 α 1개. zero_point 없음.
- **scale**: α=`mean(|W|, dim=row)`, `[out]`개, detach(학습 안 됨, forward 재계산).
- **비트폭**: W1+α, A1. 출력은 FP(스케일 곱 결과).
- **params** (Bi-DeiT-S 1 block, C=384) — 개수는 FP와 동일:
  - qkv: 384×1152 = **442,368** (qkv_bias=True지만 `bias=qkv_bias`로 전달, `:197`; bias 있으면 +1152)
  - proj: 384×384 + 384 = **147,840**
  - fc1: 384×1536 + 1536 = **591,360**
  - fc2: 1536×384 + 384 = **590,208**
  - 이진 Linear params/block ≈ **1.77M** (FP 동일), 가중치 저장은 **1-bit + per-out 스케일** → 메모리 ~32×↓.
- **BOPs/block** (B=1, N=198, XNOR-popcount 환산):
  - qkv: 198×384×1152 ≈ **87.6M**
  - proj: 198×384×384 ≈ **29.2M**
  - fc1: 198×384×1536 ≈ **116.8M**
  - fc2: 198×1536×384 ≈ **116.8M**
  - 이진 Linear BOPs/block ≈ **350.4M**, ×12 ≈ **4.20G BOPs** (attention 행렬곱 별도).
- **HW 함의**: `F.linear(±1 input, α·sign(W))` = `α ⊙ (XNOR-popcount(input, sign(W)))`. DSP 없이 LUT XNOR + popcount tree + per-out 스케일 곱 1회. 곱셈기 사실상 0(스케일 곱만).

---

## 4. 모듈: 헤드별 학습 스케일+zero_point 이진 활성 — `Bi_Quant.py` (ActBi / _ActB) ★attention degeneration 대응

### 4.1 역할 + 상위/하위
- **역할**: q/k/v/attn 활성을 **헤드별(in_features=num_heads) 학습 스케일 α + zero_point z**로 비대칭 이진화. softmax 출력의 peaky/sparse 분포를 헤드마다 다른 임계/스케일로 이진화해 **binarized attention의 degeneration(정보 붕괴)에 대응**.
- **상위**: `bi_Attention`의 `q_act/k_act/v_act/attn_act`(`bi_vision_transformer.py:201-204`). **하위**: `_ActB`(`_binary_base_plus.py:177-208`)의 alpha/zero_point Parameter, `sign_pass`(`Bi_Quant.py:132-135`), `grad_scale`.

### 4.2 데이터플로우 (텐서 shape 흐름, attn 예 [B,H,N,N])
```
x [B,6,N,N] ──(첫 배치 init)──> signed 판정; α = 2·mean(|x|)/√Qp; z EMA 초기화
        ──> z_ste = round_pass(z)                       # zero_point round-STE
        ──> α, z ← grad_scale(·, g)   g=1/√(numel·Qp)
        ──> α,z broadcast: 4D면 unsqueeze(0,2,3) → 헤드축 H에 정렬
        ──> x_b = sign_pass( clamp(x/α + z, -1, 1) )     # 비대칭 이진화, ±1
        ──> x_out = (x_b - z) · α                         # dequant (FP)
```

### 4.3 forward call stack
`bi_Attention.forward`(`bi_vision_transformer.py:217-219,225`) → `ActBi.forward`(`Bi_Quant.py:166`) → 첫 배치 α/z init(`:170-178`) → round-STE z(`:187`) → grad_scale(`:188-189`) → broadcast(`:191-196`) → `sign_pass(clamp(...))`(`:198`) → dequant(`:199`).

### 4.4 대표 코드 위치
`Bi_Quant.py`: `ActBi` `:162-200`, init `:170-178`, 이진화 `:198-199`. base `_ActB` `:177-208`(헤드별 alpha/zero_point `:188-193`).

### 4.5 대표 코드 블록
```python
# Bi_Quant.py:170-178  첫 배치 초기화 (signed, LSQ+ 스케일, zero_point EMA)
if self.training and self.init_state == 0:
    if x.min() < -1e-5: self.signed.data.fill_(1)
    Qn, Qp = -1, 1
    self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))            # LSQ+ init
    self.zero_point.data.copy_(self.zero_point.data*0.9 + 0.1*(x.min() - self.alpha*Qn))
    self.init_state.fill_(1)

# Bi_Quant.py:187-199  헤드별 비대칭 이진화 (round-STE zp + grad_scale + sign_pass)
zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point  # round-STE
alpha = grad_scale(self.alpha, g); zero_point = grad_scale(zero_point, g)
# (2D/4D broadcast to head axis)
x = sign_pass((x / alpha + zero_point).clamp(Qn, Qp))   # ±1, 비대칭
x = (x - zero_point) * alpha                             # dequant
```
→ Qn=-1, Qp=1로 **이진(±1)**이되 zero_point로 비대칭 임계 이동. α/z가 헤드축 벡터(길이 num_heads, `_binary_base_plus.py:191-192`)라 **헤드마다 다른 이진화**. softmax 출력처럼 `[0,1)` 비대칭·peaky 분포를 z로 시프트해 ±1 격자에 정합 → degeneration 완화의 코드적 실체.

### 4.6 연산·수치표현 분해 + 정량 (Bi-DeiT-S, H=6)
- **이진화 방식**: 비대칭 1-bit. `x_b = sign(x/α+z) ∈ {-1,+1}`, dequant `(x_b-z)·α`. zero_point는 헤드별 학습 파라미터(round-STE).
- **gradient 근사**: `sign_pass`(forward sign / backward identity, `:132-135`), α·z는 `grad_scale`(LSQ 그래디언트 스케일 `g=1/√(numel·Qp)`, `:184,188-189`).
- **params (추가 학습 파라미터)**: ActBi당 α[6] + z[6] = **12**. bi_Attention당 4개(q/k/v/attn) = **48**/block, ×12 = **576** (Bi-DeiT-S 전체 ActBi 파라미터, FP DeiT-S 대비 신규).
- **비트폭**: 출력 A1(±1) + 헤드별 α,z 상수.
- **activation memory**: attn [1,6,198,198] A1 = 6×198²×1bit ≈ **29.4 KB**(1-bit) + α/z 48개 FP.
- **HW 함의**: 헤드 타일당 상수 레지스터 α,z 2개만 추가 → 오버헤드 작음. 단 zero_point(비대칭)는 popcount 결과에 z 보정 가산 필요(대칭 BiReal보다 1 add/원소 추가).

---

## 5. 모듈: 이진 베이스 클래스 + STE 유틸 — `_binary_base_plus.py`

### 5.1 역할 + 상위/하위
- **역할**: 이진 Linear/Conv/Act의 공통 부모 클래스(`_LinearB`/`_Conv2dB`/`_ActB`/`_ActB_qk`)와 STE 유틸(`grad_scale`/`round_pass`/`FunStopGradient`)·고정소수 양자화 보조(`truncation`/`linear_quantize_*`) 제공. 기본 nbits=1.
- **상위**: `Bi_Quant.LinearBi`가 `_LinearB` 상속(데드), `Bi_Quant.ActBi`가 `_ActB` 상속(실사용). **하위**: torch Parameter/buffer.

### 5.2 데이터플로우 (파라미터 등록)
```
_ActB(in_features=num_heads, nbits=1, mode=kernel_wise):
  alpha = Parameter([num_heads])        # 헤드별 스케일
  zero_point = Parameter([num_heads])   # 헤드별 zp (zeros init)
  buffer: init_state[1], signed[1]
```

### 5.3 forward call stack
직접 forward 없음(베이스). `ActBi.__init__`(`Bi_Quant.py:162-164`) → `_ActB.__init__`(`_binary_base_plus.py:178-196`)에서 헤드별 alpha/zero_point 등록.

### 5.4 대표 코드 위치
`_binary_base_plus.py`: `grad_scale` `:20-23`, `round_pass` `:32-35`, `_LinearB` `:153-174`, `_ActB` `:177-208`(헤드별 파라미터 `:188-193`), `_ActB_qk` `:210-247`(데드), `truncation` `:90-98`.

### 5.5 대표 코드 블록
```python
# _binary_base_plus.py:20-23  LSQ grad_scale STE
def grad_scale(x, scale):
    y = x; y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad   # forward=x, backward=scale·grad

# _binary_base_plus.py:188-193  kernel_wise면 헤드별(in_features) 벡터 파라미터
self.alpha = Parameter(torch.Tensor(1))
self.zero_point = Parameter(torch.Tensor([0]))
if self.q_mode == Qmodes.kernel_wise:
    self.alpha = Parameter(torch.Tensor(in_features))     # =num_heads
    self.zero_point = Parameter(torch.Tensor(in_features))
    torch.nn.init.zeros_(self.zero_point)
```
→ `_ActB_qk`(`:210-247`)는 q,k 분리용 `alpha_q/alpha_k`, `zero_point_q/k`로 q·k 비대칭 곱 분포를 따로 보정하는 변형(추정). **단 대표 경로에서 미사용(데드)** — bi_Attention은 `_ActB` 기반 `ActBi` 4개만 씀.

### 5.6 연산·수치표현 분해 + 정량
- **이진화 방식**: 베이스는 파라미터 컨테이너. nbits<0이면 alpha=None(fake/passthrough).
- **params**: `_ActB` kernel_wise = 2×in_features(alpha+zero_point). `_LinearB` kernel_wise = out_features(alpha만, zp 없음).
- **truncation**(`:90-98`): `il = ceil(log2(max|fp|))`, scale=2^(nbits-il), clamp+dequant — 고정소수 양자화(보조 유틸, 대표 경로 미사용).
- **HW 함의**: kernel_wise(헤드별) 스케일 구조 = HW에서 채널/헤드 타일당 상수 레지스터 배열. grad_scale은 학습 전용(HW 추론 시 상수화).

---

## 6. 모듈: 이진 Attention + softmax 붕괴 대응 — `bi_vision_transformer.py` (bi_Attention) ★

### 6.1 역할 + 상위/하위
- **역할**: attention 전체를 이진화하되 **degeneration을 3중 방어**: (a) qkv/proj는 BiLinearBiReal(이진 GEMM), (b) q/k는 LayerNorm 후 ActBi 이진화, (c) softmax는 **float 유지**, 그 출력만 attn_act(ActBi)로 이진화.
- **상위**: `bi_Block.attn`(`:239`). **하위**: `BiLinearBiReal`(qkv/proj), `ActBi`(q/k/v/attn), `nn.LayerNorm`(norm_q/norm_k), float `@`/`softmax`.

### 6.2 데이터플로우 (텐서 shape 흐름, 1 head 기준 [N,64])
```
x [B,N,384] ──qkv(BiLinearBiReal)──> [B,N,1152] ──reshape/permute──> q,k,v [B,6,N,64]
q ──norm_q(LayerNorm head_dim=64)──> ──q_act(ActBi)──> q_b [B,6,N,64] (±1·α)
k ──norm_k(LayerNorm)──────────────> ──k_act(ActBi)──> k_b
v ─────────────────────────────────> ──v_act(ActBi)──> v_b
attn = (q_b @ k_bᵀ) · scale   [B,6,N,N]    # FLOAT 행렬곱 (입력은 이진화됐으나 dequant됨)
attn = softmax(attn, dim=-1)               # FLOAT softmax (붕괴 회피)
attn = attn_drop(attn)
attn = attn_act(attn)                      # softmax 출력만 ActBi 이진화
x = (attn @ v_b) ──reshape──> [B,N,384] ──proj(BiLinearBiReal)──> [B,N,384]
```

### 6.3 forward call stack
`bi_Block.forward`(`bi_vision_transformer.py:247`) → `bi_Attention.forward`(`:209`) → qkv(`:211`) → norm_q/k(`:214-215`) → q/k/v_act(`:217-219`) → q@kᵀ·scale(`:221`) → softmax(`:222`) → attn_act(`:225`) → @v + proj(`:227-228`).

### 6.4 대표 코드 위치
`bi_vision_transformer.py`: `bi_Attention.__init__` `:185-207`(norm_q/k `:195-196`, ActBi 4개 `:201-204`), forward `:209-230`.

### 6.5 대표 코드 블록
```python
# bi_vision_transformer.py:211-228  이진 attention + softmax float 유지
qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2,0,3,1,4)
q, k, v = qkv.unbind(0)
q = self.norm_q(q); k = self.norm_k(k)          # ★ q/k LayerNorm (분포 안정화)
q = self.q_act(q); k = self.k_act(k); v = self.v_act(v)   # 헤드별 이진화
attn = (q @ k.transpose(-2, -1)) * self.scale   # float 행렬곱
attn = attn.softmax(dim=-1)                      # ★ float softmax (붕괴 회피)
attn = self.attn_drop(attn)
attn = self.attn_act(attn)                       # ★ softmax 출력만 이진화
x = (attn @ v).transpose(1, 2).reshape(B, N, C)
x = self.proj(x)
```
→ **degeneration 대응 3요소**: (1) q/k LayerNorm으로 q·kᵀ 분산 정규화(포화 완화), (2) softmax를 float로 두어 분포 정보 보존, (3) softmax 출력을 헤드별 α/z로 이진화해 peaky 분포를 스케일로 보존. I-ViT의 IntSoftmax(정수 softmax)와 정반대 전략 — **Bi-ViT는 softmax를 절대 양자화하지 않음**.

### 6.6 연산·수치표현 분해 + 정량 (Bi-DeiT-S, B=1, H=6, N=198, dh=64)
- **이진화 방식**: qkv/proj 이진 GEMM(W1A1); q/k/v/attn은 ActBi(A1+헤드별 α,z). **q@kᵀ, attn@v 행렬곱은 float**(ActBi가 `(x_b-z)·α`로 dequant, `Bi_Quant.py:199`) → 완전 XNOR 아닌 "이진화 후 dequant된 float 행렬곱".
- **MACs/block (행렬곱, float)**:
  - q@kᵀ: H·N²·dh = 6×198²×64 ≈ **15.05M**
  - attn@v: 6×198²×64 ≈ **15.05M**
  - attention 행렬곱 MAC/block ≈ **30.1M**, ×12 ≈ **361M** (float MAC, 이진 GEMM과 별도).
- **BOPs/block (이진 GEMM)**: qkv+proj = 87.6M+29.2M ≈ **116.8M** (§3 일부).
- **params**: qkv/proj FP-동일 + ActBi 48개(§4).
- **activation memory**: q/k/v [1,6,198,64] A1 ≈ 9.5KB each; attn [1,6,198,198] softmax 출력은 **float(미양자화)** = 6×198²×4byte ≈ **941 KB**(block 내 최대, I-ViT의 A16 466KB보다 큼 — softmax float 유지 비용).
- **HW 함의**: ★ **혼합 정밀도 데이터패스 필연**. qkv/proj는 XNOR-popcount PE, q@kᵀ/attn@v/softmax는 FP(또는 고정밀) 유닛. softmax float 유지 = LayerNorm/softmax를 정밀 경로로 분리하고 GEMM만 이진화하는 heterogeneous 설계. softmax 활성이 메모리 지배(941KB).

---

## 7. 모듈: 이진 MLP — `bi_vision_transformer.py` (bi_Mlp)

### 7.1 역할 + 상위/하위
- **역할**: Transformer MLP를 이진화. fc1/fc2 모두 BiLinearBiReal(이진 GEMM), 그 사이 float GELU + **`clip(-10,10)`** 폭주 방지.
- **상위**: `bi_Block.mlp`(`:244`). **하위**: `BiLinearBiReal`×2, `nn.GELU`, `torch.clip`.

### 7.2 데이터플로우
```
x [B,N,384] ──fc1(BiLinearBiReal)──> [B,N,1536] ──GELU──> ──clip(-10,10)──> ──drop──>
            ──fc2(BiLinearBiReal)──> [B,N,384] ──drop──> out
```

### 7.3 forward call stack
`bi_Block.forward`(`:248`) → `bi_Mlp.forward`(`:172`) → fc1(`:173`) → GELU(`:175`) → clip(`:177`) → fc2(`:180`).

### 7.4 대표 코드 위치
`bi_vision_transformer.py`: `bi_Mlp.__init__` `:157-170`(fc1/fc2=BiLinearBiReal `:164,169`), forward `:172-182`(clip `:177`).

### 7.5 대표 코드 블록
```python
# bi_vision_transformer.py:172-181  이진 MLP + GELU 후 clip
x = self.fc1(x)            # 이진 GEMM
x = self.act(x)            # float GELU
x = torch.clip(x, -10., 10.)   # ★ 폭주 방지 (이진화 입력 동적범위 제한)
x = self.drop1(x)
x = self.fc2(x)            # 이진 GEMM
x = self.drop2(x)
```
→ `clip(-10,10)`은 GELU 출력을 다음 BiLinearBiReal의 sign 입력 전에 제한 → 이진화 입력의 동적범위를 묶어 학습 안정화(추정, 주석/논문 근거 없음). I-ViT가 QuantAct observer로 범위를 추적하는 것과 대비되는 **하드코딩 상수 클립**.

### 7.6 연산·수치표현 분해 + 정량 (Bi-DeiT-S, B=1, N=198)
- **이진화 방식**: fc1/fc2 W1A1 이진 GEMM. GELU는 float(미양자화).
- **params**: fc1 591,360 + fc2 590,208 = **1.18M**/block.
- **BOPs/block**: fc1 116.8M + fc2 116.8M ≈ **233.6M**(§3).
- **activation memory**: GELU 출력 [1,198,1536] — sign 입력 전 float = ~1.16 MB(block 내 큰 활성).
- **HW 함의**: GELU/clip만 float 유닛, fc1/fc2는 XNOR-popcount. clip은 saturating 비교 2회로 LUT-free.

---

## 8. 모듈: 이진 ViT 조립 + 사전학습 로드 — `bi_vision_transformer.py` (twobits_VisionTransformer)

### 8.1 역할 + 상위/하위
- **역할**: bi_Block×12를 DeiT distilled(cls+dist token, head+head_dist) 토폴로지로 조립. patch_embed/pos_embed/residual/norm/head는 float. DeiT distilled 체크포인트를 strict=False로 로드해 가중치 초기화.
- **상위**: 팩토리 `bi_deit_small_patch16_224`(`:792`). **하위**: float `PatchEmbed`, `bi_Block`×12, float `LayerNorm`/`Linear`.

### 8.2 데이터플로우 (텐서 shape 흐름)
```
img [1,3,224,224] ──PatchEmbed(float Conv2d 16×16 s16)──> [1,196,384]
   ──cat(cls[1,1,384], dist[1,1,384], patches)──> [1,198,384]
   ──+ pos_embed[1,198,384] (float add)──> ──pos_drop──>
   ──bi_Block×12──> ──norm(float LayerNorm)──>
   ──(x[:,0]=cls, x[:,1]=dist)──> head(cls), head_dist(dist)
   ──train: (out, out_dist) / eval: (out+out_dist)/2
```

### 8.3 forward call stack
`twobits_VisionTransformer.forward`(`:510`) → `forward_features`(`:495`) → patch_embed(`:496`) → cat cls/dist(`:497-501`) → +pos(`:502`) → blocks(`:503`) → norm(`:504`) → head/head_dist(`:513`).

### 8.4 대표 코드 위치
`bi_vision_transformer.py`: `twobits_VisionTransformer` `:384-521`(기본 `embed_layer=PatchEmbed` `:394`, blocks `:433-437`, distilled head `:451-454`), 팩토리 `bi_deit_small` `:792-805`(체크포인트 로드 `:799-804`).

### 8.5 대표 코드 블록
```python
# bi_vision_transformer.py:797-804  DeiT distilled 체크포인트 strict=False 로드
checkpoint = torch.load('checkpoints/deit_small_distilled_patch16_224-649709d9.pth', map_location='cpu')
model.load_state_dict(checkpoint["model"], strict=False)   # ActBi/norm_q/k 등 신규 파라미터는 미로드
```
→ FP distilled DeiT-S 가중치로 이진 모델을 초기화(qkv/proj/fc/patch/pos/head). strict=False라 `norm_q/norm_k`, `ActBi.alpha/zero_point`는 로드 안 되고 기본 init → 첫 학습 배치에서 ActBi가 α/z를 self-init(`Bi_Quant.py:170-178`).

### 8.6 연산·수치표현 분해 + 정량 (Bi-DeiT-S 전체)
- **params (분석적, FP DeiT-S와 동일 골격)**:
  - PatchEmbed(float): 384×3×16×16 + 384 = **295,296**
  - cls+dist token: 2×384 = 768, pos_embed: 198×384 = 76,032
  - bi_Block×12: 이진 Linear 1.77M×12 ≈ 21.27M + LayerNorm(norm1/norm2/norm_q/norm_k) + ActBi 48×12=576
  - 최종 norm + head + head_dist(distilled 2 head): 384×1000×2 + ... ≈ 770K
  - **총 ≈ 22.3M params** (DeiT-S 골격 + dist head + ActBi 576). 가중치 저장은 bi_Block 이진 GEMM 부분만 1-bit, 나머지(patch/pos/norm/head) float.
- **BOPs/이미지**: bi_Block×12 이진 GEMM ≈ 350.4M×12 ≈ **4.20G BOPs**. + float MAC: PatchEmbed 57.8M + attention 행렬곱 361M + head ≈ **~420M float MAC**.
- **activation memory (block당 피크)**: softmax 출력 [1,6,198,198] **float** ≈ **941 KB**(최대).
- **HW 함의**: 이진 GEMM(4.2G BOPs)은 XNOR PE로, float 경로(patch/attention matmul/softmax/LN/head ~420M MAC + 941KB softmax)는 FP 유닛으로 분리. 이진:float 연산비 ≈ 10:1(BOPs는 비트연산이라 등가 면적은 더 작음).

---

## 9. 모듈: 증류 손실 — `losses.py` (DistillationLoss) + 데드 손실들

### 9.1 역할 + 상위/하위
- **역할**: float teacher DeiT의 출력을 추가 지도로 사용. soft(KL, 온도 T) 또는 hard(teacher argmax CE) 증류. 이진 student의 정확도 회복 핵심.
- **상위**: `main_1bit.py:376-378`에서 criterion 래핑, `engine_ts.py:44`에서 호출. **하위**: `F.kl_div`/`F.cross_entropy`, teacher_model forward.

### 9.2 데이터플로우
```
samples ──teacher(no_grad)──> teacher_outputs
outputs=(out, out_kd) [student cls/dist head]
base_loss = CE(out, labels)
soft: distill = KL(logsoftmax(out_kd/T), logsoftmax(teacher/T))·T²/numel
hard: distill = CE(out_kd, teacher.argmax)
loss = base_loss·(1-α) + distill·α    # α=0.5, T=1.0 (main_1bit.py:140-141)
```

### 9.3 forward call stack
`train_one_epoch`(`engine_ts.py:41-44`) → `outputs = model(samples)` → `criterion(samples, outputs, targets)` → `DistillationLoss.forward`(`losses.py:25`) → teacher(`:48`) → soft/hard 분기(`:50-67`).

### 9.4 대표 코드 위치
`losses.py`: `DistillationLoss` `:10-70`(soft `:50-65`, hard `:66-67`). 데드: `DistillationLoss_rank` `:72-131`(teacher_outputs 외부 인자), `mf_loss`/`layer_mf_loss` `:135-192`(manifold), `merge` `:195-204`. `KD_loss.py`: `DistributionLoss` `:8-42`(데드).

### 9.5 대표 코드 블록
```python
# losses.py:47-69  soft/hard distillation (실사용)
with torch.no_grad():
    teacher_outputs = self.teacher_model(inputs)
if self.distillation_type == 'soft':
    distillation_loss = F.kl_div(F.log_softmax(outputs_kd/T, dim=1),
        F.log_softmax(teacher_outputs/T, dim=1), reduction='sum', log_target=True) * (T*T)/outputs_kd.numel()
elif self.distillation_type == 'hard':
    distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
loss = base_loss*(1-self.alpha) + distillation_loss*self.alpha
```
```python
# losses.py:157-167  manifold loss (데드코드 — engine_ts 미호출)
F_s = F.normalize(F_s, dim=-1); F_t = F.normalize(F_t, dim=-1)
M_s = F_s.bmm(F_s.transpose(-1,-2)); M_t = F_t.bmm(F_t.transpose(-1,-2))   # patch간 Gram
loss_mf_patch = ((M_t - M_s)**2).mean()    # student/teacher 매니폴드(관계 구조) 정렬
```
→ **실사용은 표준 soft/hard distillation만**. README 예시는 `--distillation-type hard`(`README.md:22,27`). manifold/rank 손실은 논문 아이디어("Ranking-aware distillation")일 수 있으나 **본 repo 학습 루프(engine_ts.py)에서 미호출** → 코드상 데드(`mf_loss` 정의만, 호출처 없음).

### 9.6 연산·수치표현 분해 + 정량
- **하이퍼파라미터**: α=0.5(`main_1bit.py:140`), T=1.0(`:141`), type∈{none,soft,hard}(`:139`). 단 `main_1bit.py:353`에서 base criterion을 `CrossEntropyLoss()`로 강제 덮어씀(mixup/label smoothing 비활성).
- **params**: 0(손실 함수). teacher는 float DeiT-S(~22M, eval 고정, gradient 없음).
- **HW 함의**: 학습 전용(추론 시 teacher/loss 불필요). 이진 ViT가 distillation 의존도 높음 → 학습 비용↑(teacher forward 매 step).

---

## 10. 모듈: Distillation QAT 학습·평가 파이프라인 — `main_1bit.py` + `engine_ts.py`

### 10.1 역할 + 상위/하위
- **역할**: FP distilled DeiT 로드 → float teacher 생성 → distillation QAT(AdamW+cosine, fake-binarize) → best 체크포인트 저장 → ImageNet 평가.
- **상위**: CLI(`README.md:22,27`). **하위**: timm(`create_model/optimizer/scheduler`, `NativeScaler`), `DistillationLoss`, `bi_vision_transformer`.

### 10.2 데이터플로우
```
argparse ──create_model(bi_deit_small, pretrained=False)──> [이진 모델, 내부서 FP ckpt 로드]
        ──teacher = create_model(deit_small, pretrained=True).eval()──>
        ──train_one_epoch: model(samples) ──fake-binarize forward──>
            loss = DistillationLoss(samples, outputs, targets) ──AdamW.step()──>
        ──evaluate: model.eval() ──accuracy top1/5──>
        ──best 갱신 → best_checkpoint.pth──>
```

### 10.3 forward call stack
`main`(`main_1bit.py:183`) → model/teacher 생성(`:255,359`) → criterion 래핑(`:376`) → epoch 루프(`:407`) → `train_one_epoch`(`:411`, engine_ts.py:20) → `model(samples)`(`engine_ts.py:41`) → `loss.backward()`(`:59`) → `evaluate`(`:433`, engine_ts.py:79).

### 10.4 대표 코드 위치
`main_1bit.py`: argparse `:36-180`, 모델/teacher `:254-371`, distillation 래핑 `:376-378`, train 루프 `:407-463`, eval `:397-400,433`. `engine_ts.py`: `train_one_epoch` `:20-76`, `evaluate` `:79-109`.

### 10.5 대표 코드 블록
```python
# main_1bit.py:318-322  linear LR scaling + AdamW
linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
args.lr = linear_scaled_lr
optimizer = create_optimizer(args, model_without_ddp)

# main_1bit.py:359-371  float teacher 생성 (distillation)
teacher_model = create_model(args.teacher_model, pretrained=True, num_classes=args.nb_classes)
teacher_model.to(device); teacher_model.eval()

# engine_ts.py:41-44  fake-binarize forward + distillation loss
outputs = model(samples)
loss = criterion(samples, outputs, targets)   # DistillationLoss
```
→ I-ViT의 `freeze_model/unfreeze_model`(observer 토글) 같은 명시적 통계 고정 메커니즘 없음. Bi-ViT는 가중치 매 forward 재이진화(`BiLinearBiReal`), ActBi는 `init_state` 버퍼로 첫 배치만 α/z self-init(`Bi_Quant.py:170-178`), 이후 학습.

### 10.6 연산·수치표현 분해 + 정량 / 재현 명령
- **하이퍼파라미터**: AdamW(`--opt adamw`, `:58`), lr 기본 5e-4(`:73`, linear-scaled), min-lr 1e-5(`:83`), epochs 기본 **300**(`:39`), warmup 5(`:88`), cosine(`:71`), weight_decay 0.05(`:68`), batch 64(`:38`), drop_path 0.1(`:48`), label_smoothing 0.1(`:103`, 단 `:353`서 CE로 덮음), EMA decay 0.99996(`:54`, 단 `:303-309`서 비활성), Mixup 0.8/CutMix 1.0(`:122,124`, 단 `:248`서 `if False`로 비활성).
- **재현 명령** (`README.md:22,27`):
  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 --use_env main_1bit.py \
    --model bi_deit_small_patch16_224 --data-path /path/ImageNet/ \
    --distillation-type hard --teacher-model deit_small_patch16_224 \
    --resume best_checkpoint.pth --eval
  ```
- **정확도**: README 수치표 없음 → **확인 불가**. 본 세션 미실행.
- **주의**: mixup/EMA/manifold/rank 모두 코드에서 비활성 또는 미호출 → 실제 학습은 **단순 CE + hard/soft distillation + AdamW**.

---

## N+1. 모듈 한눈 요약 표

| 모듈 | 파일:라인 | 역할 | 이진화 방식 | 대표 정량(Bi-DeiT-S) |
|---|---|---|---|---|
| BinaryActivation | Bi_Quant.py:57-74 | BiReal 3차 STE sign | forward sign, backward 3차 도함수 | params 0, A1 |
| BiLinearBiReal | Bi_Quant.py:77-101 | 이진 Linear(qkv/proj/fc) | W=mean(|W|)·sign(W), A=sign STE | block 1.77M params, 350.4M BOPs |
| ActBi(_ActB) | Bi_Quant.py:162-200 | 헤드별 α+zp 비대칭 이진 활성 | sign(x/α+z)·α, kernel_wise=heads | ActBi 48 params/block, attn A1 29.4KB |
| _binary_base_plus | _binary_base_plus.py:177-247 | 이진 베이스+STE 유틸 | grad_scale/round_pass, 헤드별 param | _ActB 2×heads params |
| bi_Attention | bi_vision_transformer.py:185-230 | 이진 attn + softmax float | qkv/proj 이진, softmax float, attn_act 이진 | 행렬곱 30.1M MAC(float), softmax 941KB(float) |
| bi_Mlp | bi_vision_transformer.py:154-182 | 이진 MLP + clip(-10,10) | fc1/fc2 이진, GELU float | 1.18M params, 233.6M BOPs |
| twobits_VisionTransformer | bi_vision_transformer.py:384-521 | 이진 ViT 조립(distilled) | bi_Block 이진, patch/pos/norm/head float | 총 22.3M params, 4.2G BOPs/img |
| DistillationLoss | losses.py:10-70 | soft/hard 증류 | (손실) α=0.5,T=1.0 | params 0, teacher float forward |
| QAT pipeline | main_1bit.py:407-463 | distillation QAT + eval | fake-binarize, AdamW cosine | lr 5e-4, epochs 300, 정확도 확인불가 |
| (데드) LinearBi/_ActB_qk/utils_quant/manifold | 각 파일 | 미사용 실험 잔재 | — | §0.5 참조 |

---

## N+2. 학습·평가 파이프라인 + 재현 명령

- **데이터셋**: ImageNet (IMNET), 224×224, 1000 클래스(`main_1bit.py:147-150`).
- **사전학습**: bi_deit_* 팩토리가 내부에서 `checkpoints/deit_*_distilled_*.pth`를 strict=False 로드(`bi_vision_transformer.py:785,799`). teacher는 `create_model(deit_*, pretrained=True)`.
- **Distillation QAT**:
  ```bash
  python -m torch.distributed.launch --nproc_per_node=4 --use_env main_1bit.py \
    --model bi_deit_small_patch16_224 --data-path <DATA> \
    --distillation-type hard --teacher-model deit_small_patch16_224 --epochs 300
  ```
  옵션: `--model {bi_deit_tiny,bi_deit_small}_patch16_224`, `--distillation-type {none,soft,hard}`(`:139`), `--lr`(기본 5e-4 linear-scaled).
- **체크포인트**: 매 epoch `checkpoint.pth`, best Top-1 갱신 시 `best_checkpoint.pth`(`main_1bit.py:420,439`).
- **평가**: `--eval` + `--resume best_checkpoint.pth` → `evaluate`(`engine_ts.py:79`) Top-1/5.
- **의존성**: Python 3.8, PyTorch 1.7.1, torchvision 0.8.2, **timm 0.4.12**(`README.md:11-15`). DeiT 기반. **CUDA 필수**(추정, `engine_ts.py:67` cuda.synchronize).
- **(외부) 체크포인트**: Google Drive(`README.md:29-31`, 이름만 / 확인 불가).

---

## N+3. 우리 프로젝트(FPGA ViT 가속) 시사점 + 1-bit XNOR HW 함의

### N+3.1 이진 GEMM = XNOR-popcount PE 직접 청사진 (최우선)
- **BiLinearBiReal**(`Bi_Quant.py:93-100`): `F.linear(sign(input), mean(|W|)·sign(W))` = `α ⊙ XNOR-popcount(±1 input, ±1 weight)`. → FPGA에서 곱셈을 **XNOR(LUT) + popcount tree(가산기 트리) + per-out 스케일 곱 1회**로 환원. DSP 거의 0, LUT 기반 초고효율. HG-PIPE류 PE를 비트연산 PE로 치환하는 직접 설계 기준. 가중치 메모리 ~32×↓(1-bit + per-out 스케일).
- **정량**: Bi-DeiT-S 이진 GEMM = **4.2G BOPs/img**(qkv/proj/fc1/fc2). BOPs는 1-bit 연산이라 등가 MAC 면적보다 훨씬 작음(추정).

### N+3.2 헤드별 α/zero_point = HW 타일 상수 레지스터 + zp 보정
- **ActBi**(`Bi_Quant.py:198-199`): `sign(x/α+z)·α`, α/z가 헤드축 벡터(`_binary_base_plus.py:191-192`). → 헤드 타일 단위 파이프라인에 **헤드당 상수 레지스터 2개(α,z)** 만 추가. 오버헤드 작음. 단 zero_point(비대칭)는 popcount 결과에 **z 보정 가산 1회/원소** 필요 — 대칭 BiReal보다 가산기 1단 추가. binarized attention degeneration 대응을 HW 비용 최소로 매핑.

### N+3.3 ★ softmax float 유지 = 혼합 정밀도 데이터패스 필연 (HG-PIPE 핵심 설계 변수)
- Bi-ViT는 **softmax를 절대 양자화하지 않음**(`bi_vision_transformer.py:222`), q@kᵀ/attn@v도 dequant된 float 행렬곱. I-ViT(IntSoftmax 정수 softmax)와 **정반대 전략**.
- → FPGA에서 (a) qkv/proj/fc = XNOR-popcount 이진 PE, (b) q@kᵀ/attn@v/softmax/LayerNorm = FP(또는 고정밀) 유닛으로 **명확히 분리**하는 heterogeneous 데이터패스 필연. softmax 활성이 메모리 지배(**941KB**, I-ViT A16 466KB의 2배) → on-chip 메모리 압박 지점. XR 저지연에서 softmax FP 유닛(exp/reciprocal) 비용이 핵심 설계 변수.
- 비교: I-ViT는 전 경로 정수전용(softmax도 시프트), Bi-ViT는 GEMM만 이진·비선형은 float. **두 전략의 절충(GEMM 1-bit + 비선형 정수 softmax)이 우리 가속기의 미탐색 설계점**(추정).

### N+3.4 q/k LayerNorm + clip = 이진화 전 분포 안정화 파이프 스테이지
- q/k LayerNorm(`:214-215`)·MLP clip(-10,10)(`:177`)은 이진화 전 동적범위 제한. → HW에서 "LayerNorm/clip → sign → XNOR" 파이프 스테이지로 매핑. 시선추적 소형 ViT에 1-bit 적용 시 정확도 위해 필수(추정). I-ViT의 observer 기반 범위 추적과 달리 **하드코딩 상수**라 HW 단순(LayerNorm 누산기 + saturating 비교).

### N+3.5 FPGA 친화도 평가 (1-bit XNOR 관점)
| 항목 | 평가 | 근거 |
|---|---|---|
| 이진 GEMM(XNOR-popcount) | ★★★ qkv/proj/fc 전부 1-bit, DSP-free | `Bi_Quant.py:93-100` |
| 가중치 메모리 | ★★★ ~32×↓(1-bit+per-out 스케일) | `:94-96` |
| attention 행렬곱 | ★ float(dequant된 ±1·α 곱) — 완전 XNOR 아님 | `:199`, `bi_vt.py:221,227` |
| softmax/LayerNorm | △ float 유지 → FP 유닛 필수, 면적·메모리 부담 | `bi_vt.py:222`, softmax 941KB |
| 헤드별 zp 보정 | ★★ 상수 레지스터+가산 1단 | `Bi_Quant.py:198-199` |
| 정확도 | ? README 수치 없음 → 확인 불가, 1-bit 본질적 갭 우려 | `README.md` 미기재 |
| 학습 비용 | △ ImageNet 300 epoch + teacher distillation 의존 | `main_1bit.py:39,376` |

### N+3.6 XR 시선추적 적용 (프로젝트 성격은 추정)
- 1-bit는 면적/전력 최강 → coarse gaze 같은 저정밀 단계에 적합. 단 softmax/LayerNorm float 경로가 면적 잔존 + 정확도 갭(README 수치 부재로 정량 확인 불가) → 정밀 추정엔 저비트(I-ViT INT8/Q-ViT)와 혼용하는 **계층적 정밀도 전략** 권장(추정). 비선형(softmax) FP 유닛은 I-ViT의 시프트 softmax로 대체 시 완전 비-FP 1-bit ViT 가능성(미탐색, 추정).

---

## 부록. 근거 / 확인 불가

- **직접 코드 확인**: §2~§10 전 라인 인용 — `Bi_Quant.py`(전체), `_binary_base_plus.py`(전체), `bi_vision_transformer.py`(전체), `losses.py`(전체), `KD_loss.py`(전체), `engine_ts.py`(전체), `main_1bit.py`(전체), `utils_quant.py`(전체, 데드 판정용), `README.md`(전체).
- **분석적 산출(검증 가능)**: params/BOPs/MACs/activation memory는 Bi-DeiT-S config(`bi_vision_transformer.py:792-795`)와 표준식으로 계산(N=198=cls+dist+196). 이진 BOPs는 표준 MAC 카운트를 XNOR-popcount로 환산.
- **추정**: clip(-10,10)·q/k LayerNorm 의도(주석 없음), `_ActB_qk` 용도(q·k 비대칭), HW 매핑, BOPs 등가 면적, XR 프로젝트 성격, CPU 실행 가능 여부.
- **데드코드 판정**: `LinearBi`/`_ActB_qk`/`utils_quant.py`/`mf_loss`/`DistillationLoss_rank`/`KD_loss.DistributionLoss`/`bi_PatchEmbed`/`BiConv2dBiReal` — 대표 학습 경로(main_1bit.py → bi_deit_small → twobits_VisionTransformer)에서 호출처 미발견(§0.5). 1차 요약(`../Bi-ViT.md`)이 일부를 "핵심"으로 소개한 것을 라인 근거로 정정.
- **확인 불가(미열람/미실행/미노출)**: 정확도 수치(README 미기재), latency(측정 코드 없음), 학회(README는 AAAI2024, 과제는 ICCV'23 — 표기차이), 외부 import 모듈(`Binary_plus`/`lsq_plus`/`_quan_base_plus` — repo 파일 미노출, `bi_vision_transformer.py:23-27`), `bi_swin_transformer.py`/`main_swin.py` 세부, Google Drive 체크포인트 내용.
