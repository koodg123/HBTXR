# Bi-ViT (Binary Vision Transformer) 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\Bi-ViT`
> 분석 방식: 실제 소스 라인 단위(파일:라인). 추정/확인불가 명시.
> 참고: README는 ICCV'23이 아닌 **AAAI 2024** 채택본이라 명기(`README.md:2`). 과제 지정 "ICCV'23"과 차이 → "확인 불가/표기차이"로 둠.

---

## 1. 개요

- **목적**: ViT를 **1-bit 가중치/활성(완전 이진화)** 으로 압축. ViT 이진화의 최대 난점인 **어텐션(softmax 분포) 붕괴**를 다룸.
- **원논문**: *Bi-ViT: Pushing the Limit of Vision Transformer Quantization* (README는 AAAI2024 명기, `README.md:1-2`). 기반은 DeiT(`README.md:9`).
- **핵심 아이디어** (코드 확인 범위):
  1. **BiReal 스타일 이진 활성**: `sign(x)`에 **3차 다항식 근사(piecewise polynomial) STE**로 그래디언트 정합 (`Bi_Quant.py:57-74`).
  2. **채널/헤드별 학습 스케일 + zero_point**: Q/K/V/attention을 **헤드별(`in_features=num_heads`) α·zero_point**로 이진화 — softmax 출력 분포에 맞춘 비대칭 이진화 (`Bi_Quant.py:162-200`, `bi_vision_transformer.py:201-204`).
  3. **Q/K LayerNorm 선처리**: 이진화 직전 q,k에 LayerNorm을 걸어 분포를 안정화(softmax 붕괴 완화) (`bi_vision_transformer.py:195-196,214-215`).
  4. **이진 가중치 스케일**: `W_b = mean(|W|)·sign(W - mean(W))` 형태(평균제거 후 채널평균 스케일) (`Bi_Quant.py:94-98`, `utils_quant.py:326-331`).
  5. **증류 학습**: KL/hard distillation + manifold(다양체) 증류 손실 (`losses.py:135-204`, `KD_loss.py`).

---

## 2. 디렉토리 구조 (자체 핵심 / 제외)

```
Bi-ViT/
├── main_1bit.py                  # ★ 1-bit DeiT 학습/평가 엔트리
├── main_swin.py                  # Swin 1-bit 엔트리
├── engine_ts.py / engine_ts_swin.py  # teacher-student 학습 루프
├── bi_vision_transformer.py      # ★ 이진 ViT/DeiT 모델(bi_Attention/bi_Mlp)
├── bi_swin_transformer.py        # 이진 Swin
├── Bi_Quant.py                   # ★ 이진 양자화 레이어(BinaryActivation/LinearBi/ActBi)
├── _binary_base_plus.py          # ★ 이진 베이스(_LinearB/_ActB/_ActB_qk)
├── utils_quant.py                # ★ 양자화 함수군(BinaryQuantizer/TWN/Sym/Asym, QuantizeLinear)
├── losses.py                     # DistillationLoss(+rank), manifold mf_loss
├── KD_loss.py                    # DistributionLoss(KL 증류)
├── models.py / deit.py / distill_vision_transformer.py
├── datasets.py / samplers.py / utils.py / hubconf.py
└── patchconvnet_models.py / run_with_submitit.py
```
**제외**: `__pycache__`, 체크포인트(google drive, 이름만). `Binary_plus`/`lsq_plus`/`_quan_base_plus`는 import되나(`bi_vision_transformer.py:23-27`) 일부 파일이 본 repo 목록에 미노출 → "확인 불가" 표기.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `Bi_Quant.py` — 이진 양자화 핵심 ★

- **`BinaryActivation`** (`:57-74`): **BiReal 3차 근사 STE**.
  - forward 값: `out_forward = sign(x)` (`:62`).
  - STE 근사 `out3`: x<-1→-1, -1≤x<0→`x²+2x`, 0≤x<1→`-x²+2x`, x≥1→1 (`:66-71`).
  - 최종: `out = sign(x).detach() - out3.detach() + out3` → **forward는 sign, backward는 부드러운 3차 도함수**(ReActNet/BiReal STE) (`:72`).
- **`BiLinearBiReal`** (`:77-101`): 입력은 위 3차-STE 이진화, 가중치는
  `scaling_factor = mean(|W|, dim=1)` (행별, detach), `W_b = α·sign(W)`, STE clamp(-1,1) (`:93-98`). → **W_b = α·sign(W)**. MLP/QKV/proj에 사용.
- **`BiConv2dBiReal`** (`:103-130`): patch embed용 이진 conv(동일 방식).
- **`sign_pass(x)`** (`:132-135`): `sign`의 STE(forward sign, backward identity).
- **`LinearBi(_LinearB)`** (`:137-160`): LSQ식 **학습형 이진 스케일**.
  - α 초기화 `2·mean(|W|)/sqrt(Qp)` (LSQ+ init, `:149`), `g=1/sqrt(numel·Qp)` (`:151`).
  - `w_q = sign_pass(clamp(W/α, -1, 1))·α` (`:156`) — Qn=-1,Qp=1.
  - 내부에 `self.act = ActBi(...)`로 활성도 이진화 (`:141,158`).
- **`ActBi(_ActB)`** (`:162-200`) ★ **채널별 학습 스케일 + zero_point 이진 활성**:
  - 첫 배치 init: signed 판정, `α=2·mean(|x|)/sqrt(Qp)`, `zero_point` EMA 초기화 (`:170-178`).
  - `zero_point`는 round-STE, α·zp 모두 `grad_scale(·,g)` (`:187-189`).
  - 2D/4D 입력에 맞춰 α,zp broadcasting (`:191-196`).
  - **비대칭 이진화**: `x = sign_pass(clamp(x/α + zp, -1, 1)); x = (x - zp)·α` (`:198-199`).
  - → q/k/v/attn 각각에 `in_features=num_heads`로 생성되어 **헤드별 스케일** (`bi_vision_transformer.py:201-204`).

### 3.2 `_binary_base_plus.py` — 이진 베이스 클래스 ★

- **`Qmodes`** (`:15-17`), **`grad_scale`/`round_pass`** (`:20-35`): STE 유틸.
- **`_Conv2dB`/`_LinearB`** (`:123-174`): `nbits` 기본 1(`get_default_kwargs_q` `:101-104`), `q_mode=kernel_wise`면 `alpha=Parameter(out_channels/out_features)` → **채널별 스케일** (`:135,164`).
- **`_ActB`** (`:177-208`): `alpha`/`zero_point` Parameter, kernel_wise면 `in_features` 길이 벡터(헤드별), `signed` 버퍼 (`:188-196`).
- **`_ActB_qk`** (`:210-247`): q,k 분리용 (`alpha_q/alpha_k`, `zero_point_q/k`) — q·k 비대칭 이진화 변형(추정: qk 곱 분포 보정).
- `FunStopGradient`(`:38-49`), `truncation`/`linear_quantize_*`(`:71-98`): 보조 양자화.

### 3.3 `bi_vision_transformer.py` — 이진 ViT 모델 ★

- **`bi_Mlp`** (`:154-182`): `fc1,fc2 = BiLinearBiReal`. act 후 **`torch.clip(x, -10, 10)`** 로 폭주 방지 (`:177`) — 이진화 후 활성 안정화(추정).
- **`bi_Attention`** (`:185-230`) ★ **어텐션 이진화 + softmax 붕괴 대응**:
  - `qkv = BiLinearBiReal(dim, 3*dim)` (이진 가중치) (`:197`).
  - **q,k에 LayerNorm**: `norm_q,norm_k = LayerNorm(head_dim)`를 이진화 전에 적용 (`:195-196,214-215`) → q·kᵀ 분포 표준화로 softmax 포화/붕괴 완화.
  - `q_act,k_act,v_act,attn_act = ActBi(in_features=num_heads)` (`:201-204`) → **헤드별 학습 스케일/zero_point로 이진화**.
  - 흐름: q,k LayerNorm → ActBi 이진화 → `attn=(q@kᵀ)·scale` → **softmax(float)** → dropout → **`attn_act`(헤드별 이진화)** → `@v` → proj (`:214-228`).
  - **핵심**: softmax 자체는 float, 그 **출력(attention prob)을 헤드별 학습 스케일+zero_point로 이진화**하여 분포 정보 보존(이것이 "softmax 분포 붕괴" 대응의 코드적 실체).
- **`bi_PatchEmbed`** (`:298-319`): `Conv2dBi`(이진 conv) patch embed (`:310`).
- **`twobits_VisionTransformer`** (`:384-521`): DeiT distilled(cls+dist token, 이중 head) 구조 + `bi_Block` 스택 (`:433-437`). float `VisionTransformer`(`:524-661`)와 공존(teacher용).
- 모델 등록: `bi_deit_tiny/small_patch16_224`가 **distilled DeiT 체크포인트를 strict=False로 로드**(가중치 초기화) (`:773-805`).

### 3.4 `utils_quant.py` — 양자화 함수군

- **`BinaryQuantizer(autograd.Function)`** (`:70-83`): forward `sign(x)`, backward는 |input|>1 영역 그래디언트 0(클리핑 STE).
- **`BinaryQuantizerMCN`** (`:85-131`): MCN(Modulated Conv) 스타일, MFilter(채널 스케일) 학습 + 정규화 항 그래디언트 (`:124-128`).
- **`ZMeanBinaryQuantizer`**(`:133-147`, {0,1} 출력), **`SymQuantizer`/`AsymQuantizer`/`TwnQuantizer`**(`:150-290`, n-bit/3진 옵션).
- **`QuantizeLinear`** (`:293-347`): config 비트수별 분기. weight_bits==1이면 **평균제거 후 이진화**: `real_w = W - mean(W); W_b = mean(|W|)·sign(real_w)` (`:326-331`). input_bits==1이면 sign-STE (`:335-338`).
- `QuantizeLinearMCN`(`:349-399`), `QuantizeEmbedding`(`:402-429`).

### 3.5 손실 / 증류

- **`losses.py:DistillationLoss`** (`:10-70`): DeiT 표준 soft(KL,온도 T)/hard(teacher argmax CE) 증류 (`:50-67`).
- **`DistillationLoss_rank`** (`:72-131`): teacher_outputs를 외부 인자로 받는 변형(랭킹/사전계산 teacher).
- **`mf_loss`/`layer_mf_loss`** (`:135-204`) ★ **Manifold(다양체) 증류**: feature 정규화 후 patch간/sample간/random patch간 Gram 행렬 차이 MSE — student와 teacher의 **관계 구조(매니폴드)** 정렬 (`:157-192`). 이진 ViT의 표현력 회복용(추정).
- **`KD_loss.py:DistributionLoss`** (`:8-42`): teacher softmax vs student log-softmax KL(분포 증류).

### 3.6 학습 루프 `engine_ts.py`
- `train_one_epoch(model, teacher_model, criterion=DistillationLoss, ...)` (`:20-...`): teacher eval 고정, `loss = criterion(samples, outputs, targets)` (`:44`). teacher-student 증류 기반 QAT.

---

## 4. 알고리즘 / 수식

### 4.1 이진 활성 (BiReal STE)
- forward: `a_b = sign(a)`.
- backward 근사 `g(a)`: a<-1→0, -1≤a<0→`2+2a`, 0≤a<1→`2-2a`, a≥1→0 (3차 근사 `±a²+2a`의 도함수) (`Bi_Quant.py:66-72`).

### 4.2 이진 가중치
- `α = mean(|W|)` (행/채널별, detach), `W_b = α·sign(W)`; QuantizeLinear은 평균제거형 `W_b = mean(|W|)·sign(W-mean(W))` (`utils_quant.py:326-331`).

### 4.3 헤드별 학습 스케일 + zero_point 이진화 (ActBi) ★
- Qn=-1, Qp=1. `g = 1/sqrt(numel·Qp)`.
- `x_b = sign_pass(clamp(x/α + z, -1, 1))`, `x_out = (x_b - z)·α` (`Bi_Quant.py:198-199`).
- α,z는 헤드축(`in_features=num_heads`) 벡터 → **헤드마다 다른 임계/스케일**로 softmax 출력 이진화.

### 4.4 어텐션/Softmax 붕괴 대응 (코드적 정의)
- q,k에 LayerNorm → q·kᵀ 분산 정규화(포화 완화) (`bi_vision_transformer.py:214-215`).
- softmax는 float 유지, 그 출력을 헤드별 α·z 학습 이진화(`attn_act`)하여 **희소·peaky 분포 정보를 스케일로 보존** (`:222-225`).

### 4.5 증류
- 전체: `loss = (1-α_d)·CE(student,label) + α_d·distill` (soft KL 또는 hard CE) (`losses.py:69`).
- + manifold: patch/sample/random Gram 행렬 MSE (`losses.py:157-192`).

---

## 5. 학습 / 평가 파이프라인
- **데이터셋**: ImageNet-1k (`main_1bit.py --data-path .../ImageNet`).
- **평가 명령** (README):
  - Tiny: `python -m torch.distributed.launch --nproc_per_node=4 --use_env main_1bit.py --model bi_deit_tiny_patch16_224 --distillation-type hard --teacher-model deit_tiny_patch16_224 --resume best_checkpoint_tiny.pth --eval` (`README.md:22`).
  - Small: 동일 패턴, `bi_deit_small_patch16_224` (`README.md:27`).
- **teacher**: float DeiT가 teacher, 이진 모델이 student(`engine_ts.py`).
- 체크포인트: Google Drive(`README.md:29-31`, 이름만/확인 불가).

## 6. 의존성
- Python 3.8, PyTorch 1.7.1, torchvision 0.8.2, `timm 0.4.12` (`README.md:11-15`). DeiT 기반.

## 7. 강점 / 한계 / 리스크
- **강점**:
  - 1-bit W/A로 **메모리 32×↓, 곱셈→XNOR/popcount** 가능(극단 효율).
  - 헤드별 학습 스케일+zero_point로 어텐션 이진화의 정보 손실 완화 → 1-bit ViT의 정확도 한계 돌파 시도.
  - q/k LayerNorm + softmax float 유지로 **분포 붕괴를 구조적으로 회피**.
  - manifold 증류로 표현 구조 복원.
- **한계 / 리스크**:
  - 1-bit임에도 **softmax·LayerNorm·일부 스케일은 float** → 완전 이진 HW가 아니라 혼합. 이 부분이 HW 병목.
  - 정확도는 여전히 full-precision 대비 큰 갭(1-bit의 본질적 한계).
  - import 구조에 외부 모듈(`Binary_plus` 등) 의존 → 일부 경로 "확인 불가".
  - 증류 의존도 높음(teacher 필요, 학습 비용↑).

## 8. 우리 프로젝트(ViT FPGA 가속기 HG-PIPE + XR 시선추적) 관점 시사점 — 추정
- **XNOR-popcount 곱셈 대체**: `W_b=α·sign(W)`, `a_b=sign(a)` 구조는 FPGA에서 곱셈을 **XNOR + popcount + 스케일 1회 곱**으로 환원 → DSP 거의 0, LUT 기반 초고효율. HG-PIPE의 PE를 비트연산 PE로 대체 가능(추정).
- **헤드별 스케일/zero_point**: 헤드 타일 단위 파이프라인에 헤드당 상수 레지스터(α,z)만 추가하면 됨 → HW 오버헤드 작음. 단 zero_point(비대칭)는 popcount 결과 보정 가산이 필요.
- **softmax 붕괴 ↔ HW 비선형 처리**: softmax/LayerNorm을 float(또는 고정밀)로 두는 설계는, 가속기에서 **비선형 유닛(softmax LUT/exp 근사, LayerNorm 누산기)을 정밀 경로로 분리**하고 MAC만 이진화하는 헤테로지니어스 데이터패스를 시사. XR 저지연에서 softmax 근사 LUT의 비용이 핵심 설계 변수.
- **q/k LayerNorm 선처리**: 이진화 전 정규화는 HW에서 "정규화→이진화→XNOR" 파이프 스테이지로 매핑. 시선추적 소형 ViT에 1-bit 적용 시 정확도 위해 필수(추정).
- **트레이드오프**: 1-bit는 면적/전력 최강이나 정확도 갭 → 시선추적 정밀도 요구가 낮은 단계(coarse gaze)에는 1-bit, 정밀 추정엔 저비트(Q-ViT/OFQ)와 혼용하는 계층적 정밀도 전략 가능(추정).

## 9. 근거 표기
- 라인 근거: 본문 (파일:라인) 직접 확인.
- "추정": clip(-10,10) 의도, _ActB_qk 용도, manifold 증류 목적, HW 매핑.
- "확인 불가/표기차이": 학회(README는 AAAI2024, 과제는 ICCV'23), 외부 import 모듈(`Binary_plus` 등), 체크포인트 내용.
