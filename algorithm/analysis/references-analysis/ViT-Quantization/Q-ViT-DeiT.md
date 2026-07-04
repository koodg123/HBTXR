# Q-ViT (DeiT 구현) 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\Q-ViT-DeiT`
> 분석 방식: 실제 소스 라인 단위 검토(파일:라인 근거 표기). 추정은 "추정"으로 명시.

---

## 1. 개요

- **목적**: Vision Transformer(DeiT)를 **저비트 정수 양자화(QAT)** 하되, 양자화 파라미터(스케일·비트폭)를 **완전 미분 가능(fully differentiable)** 하게 학습하여 정확도 손실을 최소화.
- **원논문**: *Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer* (NeurIPS'22). README는 arXiv 2201.07703을 인용 (`README.md:2`).
- **기반 코드**: facebookresearch/DeiT + hustzxd의 LSQ 구현(`README.md:3-4`).
- **핵심 아이디어** (코드로 확인된 범위):
  1. **LSQ 기반 학습형 스케일(learnable scale α)** — 가중치/활성/어텐션을 모두 미분 가능 스케일로 양자화 (`lsq_layer.py:50-59`, `FunLSQ`).
  2. **Head-wise(헤드별) 양자화** — 멀티헤드 어텐션의 Q/K/V, attention map을 헤드별로 독립 스케일 양자화 (`lsq_layer.py:260-413`, `QuantMultiHeadAct`/`QuantMuitiHeadLinear`).
  3. **Switchable scale(스위처블 스케일)** — `alpha`를 `zeros(7)` 벡터로 두고 비트폭 `n`에 따라 `alpha[n-2]` 인덱싱하여 2~8bit 스케일을 한 텐서로 보관 (`_quan_base.py:34,56,84`, `lsq_layer.py:129,170,243`).
  4. **학습형 비트폭(mixed precision)** — `nbits`를 `Parameter`로 두고 `mixpre=True`면 미분 가능. BitOPs 예산 정규화로 비트 배분 (`_quan_base.py:24,51,76`, `lsq_layer.py:61-63 bit_pass`, `quantvit_mixpre.py:104-109`).
  5. **Information rectification(정보 정류) — 추정**: 논문 용어. 코드상으로는 헤드별 학습 스케일 + MSE 기반 스케일 초기화로 양자화 후 분포 정보를 보존하는 것이 이에 대응 (직접 명명된 함수는 확인 불가, "추정").

---

## 2. 디렉토리 구조 (자체 핵심 / 제외)

```
Q-ViT-DeiT/
├── main.py                       # 학습/평가 엔트리 (DeiT 기반 argparse)
├── engine.py                     # train_one_epoch / evaluate, BitOPs 예산 손실
├── losses.py                     # DistillationLoss (DeiT KD)
├── datasets.py / samplers.py     # ImageNet 로더
├── utils.py / hubconf.py
├── models/
│   ├── vit.py                    # float 베이스 ViT/DeiT
│   ├── distill_models.py         # 증류용 모델
│   └── __init__.py
└── quantization/                 # ★ 양자화 핵심
    ├── _quan_base.py             # 양자화 베이스 클래스(_Conv2dQ/_LinearQ/_ActQ/_MultiHead*)
    ├── lsq_layer.py              # ★ LSQ 양자화 레이어 (Conv/Linear/Act/MultiHead)
    ├── binary_layer.py           # 1-bit 경로(BinaryActivation/BinaryLinear, ReActNet)
    ├── quantvit.py               # 단일정밀 QAT용 QuantVisionTransformer
    └── quantvit_mixpre.py        # ★ mixed-precision + head-wise + BitOPs 모델
```

**제외**: `.github/`(행동강령), `__pycache__`, 체크포인트(이름만, 확인 불가).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `quantization/_quan_base.py` — 양자화 파라미터 컨테이너

- **`Qmodes`** (`_quan_base.py:12-14`): `layer_wise=1`, `kernel_wise=2` 양자화 입도.
- **`_Conv2dQ` / `_LinearQ`** (`:17-67`): `nn.Conv2d`/`nn.Linear` 상속.
  - `self.nbits = Parameter(tensor([nbits]), requires_grad=mixpre)` — **비트폭을 학습 파라미터화** (`:24,51`). `mixpre=True`면 비트폭 자체가 역전파 대상.
  - `self.alpha = Parameter(torch.zeros(7))` (`:34,56`) — **스위처블 스케일 벡터(2~8bit, 7개)**. `learned=True`면 학습.
  - `register_buffer('init_state', zeros(1))` — MSE 스케일 초기화 1회 수행 플래그.
- **`_ActQ`** (`:71-96`): 활성 양자화. `signed`, `offset`(비대칭용 β), `dim` 옵션. offset이면 `beta=Parameter(zeros(7))` (`:85-86`).
- **`_MultiHeadActQ` / `_MultiHeadLinearQ`** (`:98-150`): **head-wise** 전용. `self.nbits = Parameter(ones(num_head)*nbits)` — **헤드별 비트폭** (`:104,133`).
- **`get_default_kwargs_q`** (`:152-170`): 레이어 타입별 기본 kwargs(nbits=8, conv는 layer_wise, act는 signed=True).

### 3.2 `quantization/lsq_layer.py` — LSQ 양자화 핵심 ★

- **`FunLSQ(autograd.Function)`** (`:25-47`): LSQ의 정석 구현.
  - forward: `q_w = (w/α).round().clamp(Qn,Qp); w_q = q_w*α` (`:31-33`).
  - backward: 스케일 α의 그래디언트를 클리핑 구간별로 계산. 중간 영역은 `(-q_w + round(q_w))`, 포화 영역은 `Qn`/`Qp` (`:40-46`). `g` (gradient scale)로 정규화. STE로 가중치 그래디언트는 중간 구간만 통과(`grad_weight = indicate_middle*grad_weight`, `:46`).
- **STE 헬퍼**:
  - `grad_scale(x, scale)` (`:50-53`): forward는 x, backward는 x*scale (스케일 학습률 보정).
  - `round_pass(x)` (`:56-59`): `round`의 STE(forward round, backward identity).
  - `bit_pass(x)` (`:61-63`): 비트폭을 [2,8]로 clamp 후 round-STE → **학습형 비트폭의 정수화** 통로.
- **`QuantConv2d` / `QuantLinear`** (`:92-178`):
  - forward에서 `nbits = bit_pass(self.nbits)` → `Qn=-2^(n-1), Qp=2^(n-1)-1` (`:113-116,153-155`).
  - `g = 1/sqrt(numel*Qp)` (LSQ 논문의 스케일 그래디언트 정규화, `:124,165`).
  - **switchable**: `alpha = grad_scale(self.alpha[n-2], g)` — 현재 비트폭 슬롯 선택 (`:129,170`).
  - `w_q = round_pass(clamp(w/α, Qn, Qp)) * α` (`:132,173`).
- **`QuantAct`** (`:180-258`): 활성 양자화.
  - **초기화 캘리브레이션**: `init_state==0`이면 forward에서 입력 샘플을 `act_samples`에 누적만 하고 통과 (`:215-223`). 이후 `initialize_scale_offset()`가 MSE로 α(,β) 초기화 (`:186-208`).
  - signed/unsigned 분기, offset이면 `x_q=round_pass(clamp((x-β)/α,Qn,Qp)); x_out=x_q*α+β` (`:249-252`).
- **`QuantMultiHeadAct`** (`:260-337`) ★ **head-wise 활성 양자화**:
  - `g = 1/sqrt(numel/num_head * Qp)` — 헤드 단위 정규화 (`:315`).
  - 입력 `x.shape = B,H,N,D`. Qn/Qp/α를 `(1,-1,1,1)`로 reshape하여 **헤드축 broadcasting** (`:323-333`).
- **`QuantMuitiHeadLinear` / `_in`** (`:339-413`) ★ **head-wise 가중치 양자화**:
  - weight를 `(Cin, num_head, Cout//H)`(out 분할) 또는 `(num_head, Cin//H, Cout)`(in 분할)로 reshape 후 헤드별 α 적용 (`:360,398`).
- **`quantize_by_mse` / `_with_offset`** (`:415-470`): 2~8bit 각각에 대해 MSE 최소화 반복으로 α(,β) 초기값 산출 → `p_alpha.data[n-2]`에 저장(스위처블 슬롯 채움, `:438,469-470`). 알고리즘: `α = <x, x_q> / <x_q, x_q>` 고정점 반복 (`:433,463`).

### 3.3 `quantization/quantvit_mixpre.py` — Mixed-precision + head-wise 모델 ★

- **`Mlp`** (`:85-112`): 각 FC 앞에 `QuantAct`, 가중치는 `QuantLinear`. **BitOPs 누적**: `bitops += N * weight.numel() * bit_pass(act.nbits) * bit_pass(fc.nbits)` (`:104,109`) — 활성비트×가중치비트×연산수.
- **`Attention`** (`:117-...`): `headwise=True`면 `QuantMultiHeadAct`/`QuantMuitiHeadLinear`로 교체 (`:125-127`). Q/K/V 각각 헤드별 활성 양자화(`quant_q/k/v`), attention map은 `quant_attn`(unsigned) (`:139-145`). `abits==1`이면 `BinaryActivation` 사용(1-bit 경로 공존).
- **BitOPs 합산 로직**: Attention/Mlp/모델 전체에서 bitops를 누적해 반환 → engine에서 예산 손실 계산.

### 3.4 `quantization/quantvit.py` — 단일정밀 QAT 모델

- `Mlp`/`Attention`/`Block`/`QuantVisionTransformer` (`:84-337`): mixpre 없는 고정 비트 QAT. attention 경로는 `q@k.T*scale → softmax → quant_attn → @v`로 **softmax는 float, 그 출력만 양자화**(`:146-152`). `abits==1`이면 BinaryActivation/BinaryLinear로 1-bit 전환 (`:91-132`).
- DeiT 변형 등록: `deit_tiny/small/base_patch16_224` (`:349-391`).

### 3.5 `quantization/binary_layer.py` — 1-bit 경로(ReActNet 계열)

- **`BinaryActivation`** (`:42-60`): `scaling_factor = mean(|x|)`(detach), `x_b = scaling_factor*sign(x)`, STE는 `clamp(x,-1,1)` 경로 (`:51-54`). → **x_b = α·sign(x)** 형태.
- **`BinaryLinear`** (`:71-83`): `scaling_factor = mean(|W|)`(detach), `W_b = α*sign(W)`, STE clamp (`:76-81`).
- **`HardBinaryConv`** (`:85-105`), `LearnableBias`(`:62-69`), `BasicBlock`/`reactnet`(`:107-208`): ReActNet 스타일 CNN 이진화(ViT 본체와는 별개, 1-bit 비교용으로 추정).

### 3.6 `engine.py` — 학습/평가 루프 + BitOPs 예산 손실

- **BitOPs 예산 정규화** (`engine.py:128-135`): 모델이 `(outputs, bitops)` 반환 시
  `loss = criterion(...) + bitops_scaler * (clamp(bitops/1e9 - budget, min=0))**2` (`:135`).
  → 예산 `budget`(GBitOps) 초과분만 2차 페널티 → **비트폭이 예산 내에서 자동 배분**(mixed precision의 핵심 제어).
- evaluate에서 `bitops(G)` 메트릭 로깅 (`:96-97`).

---

## 4. 알고리즘 / 수식

### 4.1 LSQ 스케일 그래디언트
가중치 w, 스케일 α, 정수범위 [Qn,Qp]:

- 양자화: `w_q = round(clamp(w/α, Qn, Qp)) · α`
- α 그래디언트 (`lsq_layer.py:40-46`):
  - `∂w_q/∂α = -w/α + round(w/α)`  (중간 구간, |w/α|∈[Qn,Qp])
  - `= Qn` (w/α < Qn), `= Qp` (w/α > Qp)
  - 전체에 gradient scale `g = 1/sqrt(N·Qp)` 곱 (`:44`).
- w 그래디언트: STE, 중간 구간만 통과 (`grad_weight = indicate_middle·grad_weight`, `:46`).

### 4.2 Switchable scale (스위처블 스케일)
- α ∈ ℝ⁷, 비트폭 n에 대해 사용 슬롯 = α[n-2] (n=2..8) (`:129,170,243`).
- MSE 초기화로 각 슬롯을 `α_n = argmin_α ||x - Q_n(x;α)||²` 고정점 반복으로 채움 (`:415-438`).

### 4.3 Head-wise 양자화
- 멀티헤드 텐서 B×H×N×D에서 헤드축 H마다 독립 α, Qn, Qp (`:323-333`).
- gradient scale 정규화도 헤드 단위: `g = 1/sqrt(numel/H · Qp)` (`:315,362`).

### 4.4 학습형 비트폭 + BitOPs 예산
- 비트폭 b는 Parameter, `bit_pass`로 [2,8] clamp+round-STE 후 정수화 (`:61-63`).
- BitOPs = Σ (연산수 × b_act × b_weight) (`quantvit_mixpre.py:104`).
- 손실 = task loss + λ·(max(BitOPs − budget, 0))² (`engine.py:135`).

### 4.5 1-bit (binary)
- `x_b = mean(|x|)·sign(x)` (`binary_layer.py:52`), `W_b = mean(|W|)·sign(W)` (`:79`), STE는 `clamp(·,-1,1)`.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet-1k (`datasets.py`, DeiT 표준).
- **3단계 워크플로 (README)**:
  1. **Float 베이스라인**: `main.py --model deit_tiny_patch16_224_float` (`README.md:19-27`).
  2. **균일 QAT(finetune)**: `--model deit_tiny_patch16_224_mix --wbits 4 --abits 4 --finetune <float>` (`README.md:33-53`).
  3. **Q-ViT(mixed)**: `--wbits 5 --abits 5 --bitops-scaler 1e-1 --budget 21.455 --stage-ratio 0.9 --mixpre --head-wise` (`README.md:69-88`). wbits/abits는 **초기** 비트폭(`README.md:89`).
- 분산학습: `torch.distributed.launch --nproc_per_node=8`, `--dist-eval`.

## 6. 의존성
- PyTorch 1.7.0+, torchvision 0.8.1+, `timm==0.3.2` (`README.md:6-12`).
- DeiT 코드베이스, hustzxd LSQuantization 참조.

## 7. 강점 / 한계 / 리스크
- **강점**:
  - 완전 미분 가능 양자화(스케일+비트폭) → 저비트에서 SOTA급 정확도.
  - Head-wise 스케일로 어텐션 헤드 간 분포 차이 대응(정보 보존).
  - BitOPs 예산 손실로 정확도-효율 트레이드오프를 명시적으로 제어.
  - 스위처블 스케일로 단일 모델에서 다중 비트폭 지원 가능.
- **한계 / 리스크**:
  - QAT 비용이 큼(300 epoch 풀 ImageNet 재학습, `README.md:73`).
  - **Softmax는 양자화하지 않고 float 유지**(quantvit `:147-149`) → softmax 자체는 HW에서 별도 처리 필요.
  - 활성 캘리브레이션이 `act_samples`를 CPU numpy로 누적(`lsq_layer.py:218-222`) → 메모리·속도 부담(추정).
  - head-wise/switchable로 스케일 파라미터가 많아 HW 매핑 시 LUT/스케일 곱 증가.

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 관점 시사점 — 추정
- **저비트 정수(4~8bit) MAC**: LSQ의 `w_q=round(clamp(w/α))·α`는 FPGA에서 **정수 MAC + 출력단 스케일 곱(시프트/고정소수점)** 으로 직결. α는 레이어/헤드당 상수 → 추론 시 재양자화 테이블화 가능.
- **Head-wise 스케일**: 헤드별 스케일은 systolic array를 헤드 타일로 분할하는 HG-PIPE 파이프라인과 자연스럽게 매핑(헤드당 스케일 레지스터). 단, 스케일 다양성↑ → 재양자화 로직 복잡도 trade-off.
- **Switchable/learnable bit-width**: 시선추적의 latency 예산에 맞춰 layer별 비트 배분(BitOPs budget) → **동적 정밀도 가속기**(추정) 설계 근거.
- **Softmax float 유지**의 함의: 가속기에서 softmax는 별도 비선형 유닛으로 두고, 그 입출력 경계에서만 양자화(Q-ViT의 quant_attn 위치)하는 분할이 합리적. XR 저지연 요구 시 softmax 근사·LUT 처리와 연결.
- **QAT 비용**: 시선추적용 소형 ViT/DeiT-Tiny라면 QAT 부담이 상대적으로 작아 적용 현실성 높음(추정).

## 9. 근거 표기
- 라인 근거: 본문 (파일:라인). 코드에서 직접 확인된 사실.
- "추정": Information rectification 명명, ReActNet 경로 용도, 캘리브레이션 비용, HW 매핑 해석 등.
- "확인 불가": 체크포인트 내용, main.py 미정독 부분의 일부 argparse 디테일.
