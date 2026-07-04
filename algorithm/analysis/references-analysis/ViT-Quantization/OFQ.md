# OFQ (Oscillation-Free Quantization) 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\OFQ`
> 분석 방식: 실제 소스 라인 단위(파일:라인). 추정/확인불가 명시.

---

## 1. 개요

- **목적**: 저비트 ViT의 **QAT 중 가중치 진동(weight oscillation)** 문제를 제거하여 학습 안정성·정확도 향상. 2-bit DeiT-T/S에서 이전 SOTA 대비 +9.8%/+7.7% (`README.md:5,74`).
- **원논문**: *Oscillation-free Quantization for Low-bit Vision Transformers* (ICML 2023, Liu et al., HKUST) (`README.md:1-3,61-74`). 기반 DeiT(`README.md:55`).
- **핵심 3대 기법** (README abstract `:74`, 코드 확인):
  1. **StatsQ (Statistical weight Quantization)**: 학습형 스케일(LSQ) 대신 **통계 기반 비학습 스케일**(`α=2·mean(|W|)`)로 진동 억제 (`statsq.py:122-150`).
  2. **CGA (Confidence-Guided Annealing)**: 양자화 경계 근처에서 **진동 중인 가중치를 freeze**하고, 확신 높은(경계에서 먼) 가중치만 갱신 (`cga.py:450-469,956-1013`, `statsq.py:154-193`).
  3. **QKR (Query-Key Reparameterization)**: q·k의 상호의존 진동 해소 — `W_q`, `W_k`를 **단일 행렬 `W_qk=W_qᵀW_k`로 융합 후 양자화** (`attention.py:107-222`).
- **부가**: oscillation을 정수 영역에서 추적하는 `TrackOscillation`(EMA), 반복 freezing 양자화기 (`lsq.py:111-296`).

---

## 2. 디렉토리 구조 (자체 핵심 / 제외)

```
OFQ/
├── train.py                      # timm 기반 표준 QAT 학습 스크립트
├── cga.py                        # ★ CGA(weight-freezing) 통합 학습 스크립트
├── eval.py                       # 평가
├── src/
│   ├── deit.py / deit_vision_transformer.py   # DeiT 베이스
│   ├── swin.py                   # Swin 베이스
│   ├── registry.py / __init__.py
│   ├── quantization/
│   │   ├── utils.py              # replace_module_by_qmodule_* (모듈 교체)
│   │   ├── quantizer/
│   │   │   ├── statsq.py         # ★ StatsQ + CGA freezing + TrackOscillation
│   │   │   └── lsq.py            # ★ LSQ 양자화기군 + iterative freezing
│   │   └── modules/
│   │       ├── qlinear.py        # ★ QLinear/QMLP (StatsQ-weight + LSQ-act)
│   │       ├── attention.py      # ★ QAttention / QAttention_qkreparam(_cga) / _lsq
│   │       ├── swin_attention_and_mlp.py
│   │       ├── qbias.py          # LearnableBias (RPReLU의 move)
│   │       └── utils.py
│   └── utils/                    # tokenizer/transformers/embedder 등 보조
└── timm_fix_imagenet_loading_bugs/dataset_factory.py  # timm 버그픽스(외부, 핵심 아님)
```
**제외**: `imgs/`(그림), `train_scripts/`,`eval_scripts/`(셸), `__pycache__`, 체크포인트(SharePoint 링크, 이름만).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `src/quantization/quantizer/statsq.py` — StatsQ + CGA ★

- **STE 유틸** `round_pass`/`grad_scale`/`clip`/`modify_grad` (`:13-29`). `modify_grad`는 freeze 인덱스의 그래디언트를 0으로(`:27-29`).
- **`StatsQuantizer`** (`:122-150`) ★ StatsQ 본체:
  - **비학습 통계 스케일**: `clip_val`은 `requires_grad=False` (`:128`), `scaling_factor = 2·mean(|W|)`(2D는 dim=1, 3D는 dim=-1&0) (`:138-140`) — **detach** (`:142`).
  - 양자화: `cliped = clamp(W/α, -clip/2, clip/2-eps)`; `n=2^(b-1)`; `q = α·((round(cliped·n - 0.5) + 0.5)/n)` (`:145-147`) — **half-level(대칭 격자) 양자화**.
  - STE: `q.detach() - W.detach() + W` (`:148`).
  - **핵심**: 스케일을 학습하지 않고 |W| 통계로 결정 → LSQ의 학습 스케일이 유발하는 진동 제거.
- **`StatsQuantizer_specific_4_qkreparam_cga`** (`:154-193`) ★ CGA freezing:
  - `boundaryRange`(기본 0.005) (`:155,162`).
  - 학습 시 각 정수 레벨 `i`에 대해 `round` 경계(`0.5±boundaryRange`) 근처에 있는 가중치를 `within_boundary`로 마킹 → 그 외(경계에서 먼=확신 높은)는 freeze (`:181-188`).
  - `b4_round = b4_round.detach()·freeze_idx + b4_round·(1-freeze_idx)` → **freeze된 가중치는 그래디언트 차단** (`:188`).
- **`StatsQuantizer_4d`** (`:196-219`): B,H,N,C 4D용 StatsQ(헤드/채널 평균).
- **`TrackOscillation`** (`:32-120`) ★ 진동 추적/동결:
  - 정수값 `x_int`의 매 step 변화 `delta_x_int∈{-1,0,1}`, 부호 `switch_dir` (`:72-73`).
  - **진동 정의**: `oscillated = (prev_switch_dir · switch_dir) == -1` — 직전 변화 방향과 현재가 반대 = 진동 (`:77`).
  - EMA: `ema_oscillation = m·oscillated + (1-m)·ema` (`:78-80`).
  - **freeze**: `ema_oscillation > freeze_threshold`인 가중치를 `frozen=True`로 고정, 값은 `ema_x_int` 반올림으로 동결 (`:90-98`).

### 3.2 `src/quantization/quantizer/lsq.py` — LSQ 양자화기군 ★

- **`LsqQuantizerWeight`** (`:20-109`): 표준 LSQ 가중치 양자화. per_channel α(`s`), init `α=2·mean(|W|)/sqrt(Qp)` (`:54`), `s_grad_scale = 1/sqrt(Qp·N)` (`:87`), `x_q = round_pass(clamp(x/s,Qn,Qp))·s` (`:94-100`). 1-bit는 `sign` (`:96`).
- **`LsqQuantizerWeight_iterative_freezing`** (`:202-304`): 위에 `TrackOscillation`을 끼워 **정수영역에서 진동 가중치 동결** (`:289-293`). OFQ가 LSQ를 쓸 때의 진동 대응 버전.
- **활성/이미지/conv/head/v 전용 변형**: `LsqQuantizer`(`:515`), `LsqQuantizer4img`(`:306`, signed 자동감지 `:338-339`), `LsqQuantizer4Conv2d`(`:384`), `LsqQuantizer4head_input`(`:448`), `LsqQuantizer_only_headwise`(`:612`), `LsqQuantizer4v`(`:701`) — **per-channel/per-head LSQ 활성 양자화** 다양화.

### 3.3 `src/quantization/modules/qlinear.py` — 양자 Linear/MLP ★

- **`LSQ_input`** (`:12-26`): 입력 양자화 + **RPReLU식 LearnableBias(move_b4/aft)** 로 분포 시프트 (`:19-25`).
- **`QLinear`** (`:28-87`): **가중치=StatsQ, 활성=LSQ**가 기본 (`:44,51,61-69`). `move_b4 → LSQ act → move_aft` 후 linear (`:66-69`). 가중치는 `StatsQuantizer`로 양자화 (`:62`).
- **`QMLP`** (`:89-136`): fc1/fc2를 QLinear로. act_layer는 gelu/relu/prelu/`rprelu` 선택, rprelu면 `move1→act→move2`(RPReLU) (`:110-131`).
- **`LSQ_QConv2d`**(`:138`, patch embed), **`LSQ_QLinear4head`**(`:193`), **`LSQ_w_and_act_QLinear`/`QMLP`**(`:254-363`, 가중치도 LSQ로 쓰는 비교군).

### 3.4 `src/quantization/modules/attention.py` — 어텐션 양자화 + QKR ★

- **`QAttention`** (`:12-105`): 기본 양자 어텐션(qkv/proj 양자화, softmax 후 LSQ).
- **`QAttention_qkreparam`** (`:107-222`) ★ **QKR 핵심**:
  - qkv를 q,k,v 개별 Linear로 분리(사전학습 가중치 슬라이스 복사) (`:126-138`).
  - **W_qk 융합**: 헤드별 `multi_head_q_weight.transpose·multi_head_k_weight = W_qk`(num_heads, C, C) (`:190-193`).
  - **한 번에 StatsQ**: `qk_quant(multi_head_qk)` — q,k를 따로가 아니라 **융합 행렬을 단일 양자화**하여 q-k 상호의존 진동 제거 (`:140,195`).
  - 어텐션 계산: `xqkx = X·W_qk·Xᵀ`(einsum) → `*scale` → `softmax` → `quan_a_softmax_fn`(unsigned LSQ) → `@v` → proj (`:200-221`).
  - 활성 LSQ + LearnableBias(move_qkx/v) 다수 배치.
- **`QAttention_qkreparam_4_cga`** (`:224-336`): 위와 동일하되 `qk_quant = StatsQuantizer_specific_4_qkreparam_cga`(CGA freezing 적용) (`:257`). → **QKR + CGA 동시 적용**(파이프라인의 finetune 단계).
- **`QAttention_lsq`** (`:341-...`): LSQ 비교군.

### 3.5 학습 스크립트 `cga.py` / `train.py`

- **`get_qat_model`** (`cga.py:391-431`): qconfig(가중치 StatsQ/LSQ, 활성 LSQ, 비트폭, qk_reparam 여부) 생성 → `replace_module_by_qmodule_deit/swin`으로 모듈 교체 (`:424-429`).
- **`freeze_outside_boundary_weight_idx`** (`cga.py:450-469`) ★: 현재 가중치에 대해 StatsQ 격자 경계(`0.5±boundaryRange`) 안=진동 후보, 밖=동결 인덱스 산출 (`:465-467`).
- **`train_one_epoch`의 CGA 로직** (`cga.py:953-1013`):
  1. backward 후, fc1/fc2/v/proj(또는 qkv) 가중치의 **경계 밖 그래디언트를 0** 으로 (`:962,970,978`).
  2. optimizer.step() 후 **freeze된 부분을 직전 값으로 복원** (`:989-1013`) → 확신 높은 가중치 동결, 진동 가중치만 학습.
- **CGA = finetune 단계**: 사전학습 모델을 `freeze_for_n_epochs`(기본 15) 동안 CGA finetune (`cga.py:370,835`).
- **증류 손실**: KLLossSoft/KDLossSoftandHard(+qk/qkv 변형) (`cga.py:776-783`).
- `setup_alpha`(`cga.py:1076-1089`): 첫 배치로 LSQ α 초기화(forward 1회).
- `train.py`: timm 표준 학습(StatsQ/LSQ 적용, CGA freezing 없는 일반 QAT) — argparse는 cga.py와 유사.

---

## 4. 알고리즘 / 수식

### 4.1 가중치 진동(oscillation) 정의
- 정수표현 `w_int(t)`의 step간 차분 `Δ=round(w_int(t-1)-w_int(t))∈{-1,0,1}`.
- 진동 = `sign(Δ(t-1))·sign(Δ(t)) == -1` (직전과 반대 방향 점프) (`statsq.py:77`, `lsq.py:156`).
- 진동도 EMA: `o_ema(t) = m·o(t) + (1-m)·o_ema(t-1)`.

### 4.2 StatsQ (통계 스케일 양자화)
- `α = 2·mean(|W|)` (비학습, detach) (`statsq.py:138`).
- `W_q = α·( round( clamp(W/α, -c/2, c/2)·n − 0.5 ) + 0.5 ) / n`, `n=2^(b-1)` (`:145-147`).
- LSQ 대비 α를 학습하지 않으므로 **스케일-가중치 공진동 제거**.

### 4.3 CGA (Confidence-Guided Annealing / weight freezing)
- 경계근접도: `|frac(W/α·n − 0.5)| < boundaryRange` → 진동 후보(미동결).
- 경계에서 먼(=high confidence) 가중치는 `freeze_idx=1`로 그래디언트 0 + step 후 값 복원 (`cga.py:962,989-1013`).
- 효과: 진동 가중치만 천천히 수렴(annealing), 확신 가중치는 고정.

### 4.4 QKR (Query-Key Reparameterization)
- 원래 attention logit: `(XW_qᵀ)(XW_k)ᵀ = X (W_qᵀW_k) Xᵀ`.
- **재파라미터화**: `W_qk = W_qᵀW_k`(헤드별) 를 **단일 행렬로 양자화** `Q(W_qk)` (`attention.py:193-196`).
- logit = `X · Q(W_qk) · Xᵀ` (`:210`). → q,k를 개별 양자화할 때 생기는 **상호의존 진동·그래디언트 오추정 제거**(README `:74`).

### 4.5 LSQ 활성 (보조)
- `x_q = round_pass(clamp(x/s, Qn, Qp))·s`, `s_grad_scale=1/sqrt(Qp·N)` (`lsq.py:87-100`). per-channel/head 변형 다수.

---

## 5. 학습 / 평가 파이프라인
- **데이터셋**: ImageNet-1k(ILSVRC12) (`README.md:24-25`).
- **흐름**: (a) `train.py`/`cga.py`로 사전학습 가중치 초기화 → StatsQ+LSQ QAT → (b) `cga.py`로 `--qk_reparam --qk_reparam_type 1` + `--freeze_for_n_epochs 15 --boundaryRange 0.005/0.007` CGA finetune (`cga.py:365-370`).
- **스크립트**: `train_scripts/`, `eval_scripts/`(예: `eval_scripts/deit_t/w2a2.sh`). 동일 global batch 필수(`README.md:31`).
- **결과 예**: OFQ DeiT-T 2-2 64.33 / 3-3 72.72 / 4-4 75.46; DeiT-S 2-2 75.72; Swin-T 2-2 78.52 (`README.md:41-51`).

## 6. 의존성
- numpy 1.22.3, torch 2.0.0, torchvision 0.15.1, timm 0.5.4, pyyaml (`README.md:16-20`). timm dataset_factory 패치 필요(`README.md:22`).

## 7. 강점 / 한계 / 리스크
- **강점**:
  - 진동을 **정수영역에서 정량 측정·동결**하는 명확한 메커니즘 → 저비트(2-bit) 안정성 대폭 향상.
  - StatsQ는 비학습 스케일이라 추가 파라미터·학습 불안정 제거(추론 시에도 스케일 상수).
  - QKR로 q-k 융합 → **연산도 단순화(단일 W_qk)** + 양자화 안정.
- **한계 / 리스크**:
  - CGA는 매 step `named_modules` 순회·grad 마스킹·값 복원으로 **학습 오버헤드** 큼 (`cga.py:953-1013`).
  - softmax는 float 유지(`attention.py:214`), QKR로 attention이 `X·W_qk·Xᵀ`가 되어 **중간 차원 증가**(헤드별 C×C) — 메모리/연산 trade-off.
  - 모듈 교체(`replace_module_by_qmodule_*`) 기반이라 모델 구조 변경 시 적용 범위 확인 필요.

## 8. 우리 프로젝트(ViT FPGA 가속기 HG-PIPE + XR 시선추적) 관점 시사점 — 추정
- **StatsQ → HW 친화**: 비학습 통계 스케일은 추론 시 **레이어/채널당 상수 α** → FPGA에서 정수 MAC + 고정 시프트/곱 1회로 재양자화. LSQ 학습 스케일보다 HW 매핑·캘리브레이션 단순(추정).
- **진동 억제 = 양자화 모델 재현성**: QAT가 안정적이면 동일 비트폭에서 정확도 분산↓ → 가속기 검증/배포 신뢰성↑. 시선추적처럼 안정성이 중요한 응용에 유리(추정).
- **QKR의 HW 함의**: `W_qk=W_qᵀW_k`를 오프라인 융합·양자화하면 추론 시 **q,k 두 MatMul을 W_qk 단일 MatMul로 대체** 가능 → HG-PIPE 파이프라인의 QK 스테이지 단순화. 단 W_qk는 C×C로 커져 on-chip 저장/대역폭 부담 → 타일링 필요(추정).
- **softmax float 분리**: 다른 ViT 양자화와 동일하게 softmax는 정밀 비선형 유닛으로 분리하는 헤테로지니어스 설계 시사. XR 저지연에서 softmax 근사 비용이 병목.
- **2-bit 실현성**: OFQ가 2-bit DeiT를 실용 정확도로 끌어올림 → 시선추적용 초저비트 가속기의 정확도 근거 확보(추정). 이진(Bi-ViT)과 저비트(Q-ViT) 사이의 2~4bit 스윗스팟.

## 9. 근거 표기
- 라인 근거: 본문 (파일:라인) 직접 확인.
- "추정": HW 매핑 해석, QKR 메모리 trade-off, 재현성 효과.
- "확인 불가": SharePoint 체크포인트 내용, `utils.py`의 replace_module 세부(미정독), 일부 보조 모듈.
