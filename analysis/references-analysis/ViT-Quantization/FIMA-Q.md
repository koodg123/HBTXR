# FIMA-Q 정밀 분석

> 분석 대상: `REF/ViT-Quantization/FIMA-Q`
> 작성 기준: 실제 소스 코드 (파일:라인 근거). 추정/확인 불가 명시.

---

## 1. 개요

- **정체 (확인)**: *"FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation"* (Wu, Wang, Zhang, Chen, Wang, **CVPR 2025**)의 공식 PyTorch 구현. README.md:1-3, 86-92 (arXiv 2506.11543). (코드 내부엔 선행작 **APHQ-ViT** 흔적도 있음 — README.md:69, block reconstruction 골격 공유.)
- **목적**: ViT를 위한 **PTQ(Post-Training Quantization)**. 소량 calibration 데이터만으로 W4/A4, W3/A3 같은 저비트 양자화에서 정확도 손실 최소화.
- **핵심 아이디어**: 블록 단위 reconstruction의 손실을 단순 MSE가 아니라 **Fisher Information Matrix(FIM) 근사**로 가중. 특히 **FIM을 대각(diag) + 저랭크(low-rank)로 분해(DPLR: Diagonal Plus Low-Rank)** 하여 출력 오차의 중요도를 KL-divergence 기반 gradient로 가중 → quantization 손실을 task loss에 가깝게 정렬.
- **양자화 방식**: PTQ 2단계 —
  1. **Calibration**: PTQSL(병렬 스케일 탐색) + percentile 후보로 weight/act scale·zero-point 탐색.
  2. **Optimization(선택)**: **AdaRound**(학습형 반올림) + **Fisher 손실(fisher_dplr 등)** + QDrop로 블록 reconstruction.

---

## 2. 디렉토리 구조 (자체 + 제외)

```
FIMA-Q/
├── test_quant.py                # 엔트리: calibrate/optimize/test (★)
├── configs/                     # 비트별 설정 (Config 클래스)
│   ├── 3bit/{best.py, brecq.py}
│   ├── 4bit/best.py            # w_bit=4,a_bit=4, fisher_dplr, k=5 (★)
│   └── 6bit/best.py
├── quant_layers/                # 양자화 레이어 (★)
│   ├── linear.py               # MinMax/PTQSL/Asym Batching QuantLinear (★)
│   ├── matmul.py               # QuantMatMul (Q·K, attn·V) (★)
│   └── conv.py
├── quantizers/                  # 양자화기 (★)
│   ├── uniform.py              # UniformQuantizer
│   ├── adaround.py             # AdaRoundQuantizer (학습형 반올림) (★)
│   ├── logarithm.py            # 로그 양자화 (post-softmax용 추정)
│   └── _ste.py                 # round_ste
├── utils/
│   ├── block_recon.py          # BlockReconstructor + LossFunction (Fisher 손실) (★★)
│   ├── calibrator.py           # QuantCalibrator (hook 기반 입출력 수집)
│   ├── wrap_net.py             # 모듈 → 양자화 모듈 치환
│   ├── datasets.py / test_utils.py
└── assets/main_fig.png
```

**제외**: `.git/`, `__pycache__`, `assets/`, `checkpoint/`(대용량, 이름만).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `utils/block_recon.py` — Fisher 기반 블록 reconstruction ★★

#### `BlockReconstructor(QuantCalibrator)` (80-386)
- 블록 단위(timm의 PatchEmbed / ViT Block / SwinTransformerBlock / PatchMerging / head)로 reconstruction 수행(93-102).
- block forward를 MethodType으로 교체(112-119)하여 **perturb(랜덤 섭동)** 주입 가능하게 함(`vit_block_forward` 등 17-77; APHQ-ViT 계열 기법, 추정).

#### Fisher 정보 수집 — `init_block_brecq_hessian` (225-243)
- KL-div loss로 backward → block의 입력 grad 수집(229-238). `loss = KL(log_softmax(pred/T), raw_softmax)`(234) → grad의 절댓값을 reshape(238). **BRECQ식 Hessian/FIM 근사** (diag of FIM ≈ E[g²]).

#### `new_fisher_ro` (245-275) — 저랭크 FIM 갱신 ★
- 양자화 출력 hook + backward hook으로 `raw_grad`(KL grad)와 `delta_out = |q_out - raw_out|` 수집(257-261).
- 매 호출마다 `raw_grad`/`delta_out`을 행으로 누적 (k×N) (266-271).
- **핵심**: `inverse_B = inv(delta_out @ delta_outᵀ)` (k×k) 계산(272) → 저랭크 FIM의 역행렬 항. (DPLR의 low-rank 성분에 사용.)

#### `reconstruct_single_block` (277-363) — AdaRound + Fisher 최적화
- `wrap_quantizers_in_net`(281): weight quantizer를 **AdaRoundQuantizer**로 교체(137-149), `soft_targets=True`.
- weight는 AdaRound `alpha` 파라미터, activation은 scale을 학습 대상으로(289-307). w/a 별도 Adam optimizer + cosine LR(305-307).
- **QDrop**(286-287, 314-317): calibration 입력을 양자화입력/FP입력을 `drop_prob`로 확률적 혼합(317).
- Fisher 갱신 스케줄(324-334): `dis_mode='q'`면 `i_change=iters/k` 간격마다 `new_fisher_ro` 호출(327), `'qf'`면 처음 k회(331).
- 손실 `LossFunction(...)` 호출 → backward → step(342-351).
- 종료 후 hard rounding 고정(`soft_targets=False`, 353-360).

#### `reconstruct_model` (366-386)
- 블록 순차 reconstruction. 끝나면 weight를 AdaRound hard value로 복사하고 `alpha` 제거, round_mode='nearest'(382-385).

#### `LossFunction` (388-493) ★★ — Fisher 손실 정의
- `lp_loss`(413-421): Lp norm 손실.
- `__call__`(423-493): `rec_loss` 선택 분기 —
  - `'mse'`/`'mae'`(435-444): 기본 Lp.
  - `'fisher_lr'`(445-450): `loss = mean((Δ·|g|).mean²)` — low-rank 근사. Δ=|pred-tgt|.
  - `'fisher_diag'`(451-456): `loss = mean(Δ² · mean(|g|))` — **대각 FIM** 가중 MSE.
  - **`'fisher_dplr'`(457-465)** ★ (논문 핵심, configs default):
    - `A = Δ.unsqueeze(1) @ |g|ᵀ` (459)
    - `loss_1 = (A @ inverse_B @ Aᵀ).mean()` — **low-rank 성분**(460, `inverse_B`는 new_fisher_ro에서 계산).
    - `loss_2 = (Δ² · mean(|g|)).mean()` — **diagonal 성분**(461).
    - `rec_loss = p1·loss_1/init + p2·loss_2/init` (465) → **Diagonal + Low-Rank FIM 근사**.
  - `'fisher_brecq'`(466-471): `(Δ²·g²).mean()` — BRECQ식 diag Hessian.
  - `'kl_div'`(472-473): head 블록은 KL.
  - **round_loss**(477-487): AdaRound 정규화 `weight·Σ(1 - |2·(h-0.5)|^b)` (485), `b`는 cosine 온도 감쇠(`LinearTempDecay`, 496-513).

### 3.2 `quant_layers/linear.py` — PTQSL 양자화 Linear ★

#### `MinMaxQuantLinear(nn.Linear)` (8-61)
- 기본: 대칭 UniformQuantizer w/a(18-19), `mode`별 forward(raw/quant_forward/debug)(26-37). `quant_forward`는 weight·act 양자화 후 `F.linear`(46-51).

#### `PTQSLQuantLinear` (64-105)
- **PTQSL(Parallel Quantization Scale Learning)**: weight를 `n_V × crb_rows`로 그룹화(`crb_rows = out_features//n_V`, 88) → 그룹별 채널 scale(91). weight는 channel_wise, act는 per-tensor.
- `_get_similarity`(94-101): -MSE/-MAE 유사도.

#### `PTQSLBatchingQuantLinear` (108-253) — 스케일 탐색 본체
- `_initialize_calib_parameters`(126-139): GPU 메모리 기반 병렬 탐색 수 `parallel_eq_n` 자동 결정.
- `_initialize_weight_scale`(141-146)/`_initialize_activation_scale`(148-157): abs-max 기반 초기 scale.
- `_search_best_w_scale`(159-191): `eq_n`개 scale 후보를 **병렬로** 양자화→출력 유사도 최대 후보 선택(argmax)(167-190).
- `_search_best_a_scale`(193-226): activation scale 동일 방식 탐색.
- `hyperparameter_searching`(228-253): weight scale 후보(eq_alpha=0.01~eq_beta=1.2 사이, 234-237) × act scale 후보 생성 후 search_round 반복 탐색.

#### `AsymmetricallyBatchingQuantLinear` (256-524) — 비대칭 + percentile
- 비대칭 UniformQuantizer로 교체(274-280), scale+zero_point 모두 nn.Parameter.
- `calculate_percentile_weight_candidates`(449-468)/`..._activation_candidates`(470-501): **percentile(l=0.9, r=1.0) 기반 scale·zp 후보** 격자 생성 (outlier robust). num_zp×num_scale 조합.
- `_search_best_*_scale_self`(313-370): weight/act **자체 재구성 오차** 최소 후보 선택(소프트). `_search_best_*_scale`(372-447): **출력 오차** 기반 선택.
- `hyperparameter_searching`(503-524): self-search → output-search 반복. `token_channel_wise`면 토큰별 scale로 확장(516-520).

#### `AsymmetricallyChannelWiseBatchingQuantLinear` (527-597) — 채널별 + reparam ★
- activation을 **채널별(channel_wise)** 양자화(545-547).
- **`reparam_step1`(571-587)** ★: 채널별 scale을 평균 target scale로 정규화하면서, **이전 레이어(prev_layer)의 weight/bias에 역스케일 `r`을 흡수**(579-580), 현재 weight에 `r`을 곱해 흡수(581) → **수학적 등가 변환으로 채널 outlier를 흡수**(SmoothQuant/cross-layer equalization 계열). bias 보정(`b`)도 처리.
- `reparam`(589-597): raw_input도 `/r - b`로 갱신 후 per-tensor로 재탐색.

### 3.3 `quant_layers/matmul.py` — Attention 행렬곱 양자화 ★
- `MinMaxQuantMatMul`(12-43): Q·Kᵀ, attn·V 양자화. A/B 각각 quantizer.
- `PTQSLQuantMatMul`(45-83): **head_channel_wise** 양자화 — head 차원(dim-1)별 scale(`[1, num_heads, 1, 1]`, 72-74). attention의 head별 분포 차이 대응.
- `AsymmetricallyBatchingQuantMatMul`(109-282): 비대칭 + percentile(l=0.99, r=0.99999, 202-224) 후보로 A/B scale·zp 교대 탐색(241-267). `token_channel_wise`면 토큰별 scale 확장(269-277).

### 3.4 `quantizers/adaround.py` — AdaRound ★

#### `AdaRoundQuantizer` (7-77)
- **학습형 반올림** (Nakkiran AdaRound, 12). `round_mode='learned_hard_sigmoid'`(43-48): `x_int = floor(x/scale) + (soft_targets ? sigmoid_clip(alpha) : (alpha≥0))`.
- `get_soft_targets`(59-60): `clip(sigmoid(alpha)·(ζ-γ)+γ, 0, 1)`, ζ=1.1, γ=-0.1(34) — rectified sigmoid.
- `init_alpha`(62-67): `sigmoid(alpha)=rest`가 되도록 alpha 초기화 → 처음엔 nearest와 동일.
- `get_hard_value`(71-73): 최종 hard 반올림 값.

### 3.5 `test_quant.py` — 파이프라인 엔트리
- `wrap_modules_in_net`(224)로 양자화 모듈 치환.
- **Calibration**(235-242): `QuantCalibrator.batching_quant_calib()` → `wrap_reparamed_modules_in_net`(reparam 흡수).
- **Optimization**(246-253): `BlockReconstructor(... metric=fisher_dplr, k, dis_mode, p1, p2)` → `reconstruct_model(quant_act=True, mode='qdrop', drop_prob=0.5)`.
- `load_model`(129-150): calibrate/optimize 체크포인트 로딩(scale/zp 복원).
- README.md:52 권장: `--optim-metric fisher_dplr`.

### 3.6 설정 (`configs/4bit/best.py`)
- `w_bit=4, a_bit=4, qconv_a_bit=8, qhead_a_bit=4`(8-11), `calib_metric='mse'`, `eq_n=128, search_round=3`(15-16), `optim_metric='fisher_dplr', temp=20`(19-20), **Fisher rank `k=5`, p1=p2=1.0**(22-24), `optim_mode='qdrop', drop_prob=0.5`(27-28).

---

## 4. 알고리즘 / 수식

### 4.1 FIM 근사 reconstruction objective (DPLR)
출력 오차 `Δ = |y_q - y_fp|`, KL-div task-gradient `g`에 대해 Fisher 가중 손실:
$$\mathcal{L}_{rec} = p_1\cdot\frac{\Delta^\top (G\,B^{-1}G^\top)\,\Delta}{c_1} \;+\; p_2\cdot\frac{\sum_i \Delta_i^2\,\mathbb{E}[|g_i|]}{c_2}$$
- 첫 항 = **low-rank** 성분 (`A=Δ·|g|ᵀ`, `inverse_B = (Δ_outΔ_outᵀ)^{-1}`), 둘째 항 = **diagonal** 성분. → FIM ≈ Diagonal + Low-Rank. 근거: block_recon.py:457-465, 245-275.

### 4.2 FIM diag 근사
$$F_{ii}\approx \mathbb{E}\big[g_i^2\big],\quad g = \nabla\,\text{KL}\big(\text{softmax}(z_{fp}/T)\,\|\,\text{softmax}(z_q/T)\big)$$
근거: block_recon.py:234, 451-456, 466-471.

### 4.3 AdaRound
$$\hat{w} = s\cdot\Big(\lfloor w/s\rfloor + h(\alpha)\Big),\quad h(\alpha)=\text{clip}\big(\sigma(\alpha)(\zeta-\gamma)+\gamma,0,1\big)$$
정규화: `f_reg = Σ(1 - |2h(α)-1|^b)`. 근거 adaround.py:43-60, block_recon.py:480-485.

### 4.4 PTQSL 스케일 탐색 + percentile
eq_n개 scale 후보를 병렬 양자화하여 출력 MSE 최소 후보 선택(argmax of -MSE). percentile(0.99~0.99999)로 outlier robust 후보 생성. 근거 linear.py:159-253, 449-501; matmul.py:202-267.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet. calibration 소량(calib_size=128, optim_size=1024). README.md:36-43.
- **모델**: deit/vit/swin tiny~base (timm, README.md:39). pretrained timm/AdaLog 체크포인트.
- **명령** (README.md:52):
  `python test_quant.py --model vit_small --config ./configs/3bit/best.py --dataset <DIR> --calibrate --optimize --optim-metric fisher_dplr`
- **모드**: `--calibrate`(신규) vs `--load-calibrate-checkpoint`(기존 로드), `--optimize`(블록 recon) vs `--load-optimize-checkpoint`. README.md:45-64.
- **결과**: W4/A4 ViT-S 76.68%, DeiT-S 76.87%; W3/A3도 동작 (README.md:71-79).

---

## 6. 의존성

- PyTorch 2.2.2 + CUDA 12.1, **timm 0.9.2**(timm Block/Swin 구조에 강결합, README.md:18-21). NumPy, tqdm. **CUDA 필수**(메모리 기반 병렬 탐색, GPU 강제).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **PTQ만으로 저비트(W3/A3) 동작** — 재학습 불필요, calibration 1024장 수준.
- **Fisher-DPLR**: 출력 MSE를 task-aware로 가중 → 단순 BRECQ/MSE 대비 정확도↑ (특히 저비트).
- AdaRound + QDrop + percentile + PTSL이 잘 통합된 강한 PTQ 파이프라인.
- reparam(채널 equalization)으로 activation outlier 흡수.

**한계 / 리스크**
- **timm 구조 의존**: block forward를 MethodType으로 교체 → 커스텀 ViT엔 어댑터 필요.
- 블록 reconstruction(iters=20000/블록, optimize)은 calibration보다 **시간 비용 큼**.
- `inverse_B` 행렬역(블록당) 등 GPU 메모리·연산 부담. k(rank) 증가 시 비용↑.
- **모의 양자화(fake quant)** 기반 — dyadic/정수 전용 추론 코드는 아님(HW 직접 매핑은 별도 작업).

---

## 8. 우리 프로젝트 관점 시사점 (ViT FPGA 가속기 + XR 시선추적 — 추정)

- **PTQ 정확도 확보 경로**: HG-PIPE류 가속기에 ViT를 올릴 때 재학습(QAT) 없이 **저비트 정확도를 확보**하려면 FIMA-Q식 PTQ가 직접 후보. 특히 XR 시선추적처럼 **데이터·재학습 자원이 제한**된 상황에서 calibration 소량으로 W4/A4를 맞추는 전략에 부합.
- **Fisher-DPLR의 의미**: 양자화 손실을 "최종 task에 영향 큰 출력"에 집중 → 시선추적 정확도(시선 좌표 회귀/분류)에 민감한 채널을 우선 보호. 단, 본 코드는 분류(softmax/KL) 기준이라 **회귀 task(시선 좌표)엔 손실 정의 수정 필요**(추정).
- **reparam(채널 equalization)**: `AsymmetricallyChannelWiseBatchingQuantLinear.reparam`은 outlier를 인접 레이어로 흡수하는 등가변환 → FPGA에서 per-channel scale을 per-tensor로 줄여 **HW 단순화** 가능. HG-PIPE의 per-tensor datapath에 유리.
- **HW 매핑 갭**: FIMA-Q는 fake-quant PTQ이므로, FPGA 배포 전 I-ViT식 정수/dyadic datapath로의 변환(scale→dyadic)이 추가로 필요(추정). FIMA-Q로 **정확도 좋은 scale/round를 찾고**, I-ViT로 **정수 실행**하는 조합이 이상적(추정).

---

## 9. 근거 표기

- repo 정체(FIMA-Q/CVPR2025): **확인** (README.md:1-3, 86-92). (코드 내 APHQ-ViT 언급 — README.md:69, 선행작 골격 공유: **확인**.)
- Fisher 손실(fisher_dplr/diag/lr/brecq), AdaRound, PTQSL, reparam: **확인** (block_recon.py, linear.py, matmul.py, adaround.py 라인 근거).
- `logarithm.py`(로그 양자화, post-softmax 용도): **추정** (파일 미정독).
- block forward perturb의 정확한 의도: **추정** (APHQ-ViT 계열).
- "HG-PIPE + XR 시선추적" 적용 방안: **추정** (지시문 기반).
