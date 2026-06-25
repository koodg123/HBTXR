# APHQ-ViT 정밀 분석 (Average Perturbation Hessian Based Reconstruction PTQ for ViT)

> 분석 대상 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\APHQ-ViT`
> 분석 방식: 자체 핵심 소스 라인 단위 정밀 분석 (Glob/Grep/Read). bash 미사용.
> 근거 표기 규칙: `파일:라인`. 직접 확인분과 "추정"/"확인 불가" 구분.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문**: Zhuguanyu Wu et al., *"APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers"*, CVPR 2025 (arXiv:2504.02508). (`README.md:1-3, 93-99`)
- **코드 출처**: **AdaLog 기반으로 수정** ("modified based on AdaLog", `README.md:3`). → 양자화기/양자화 레이어/캘리브레이터 골격은 AdaLog와 공유, 차이는 **Hessian 기반 재구성(reconstruction)** 에 집중.
- **목적**: **PTQ**로 저비트(W3/W4) ViT 양자화 정확도 회복. 특히 **Average Perturbation Hessian(APH)** 라는 새 중요도 척도로 출력 reconstruction 손실을 가중.
- **핵심 아이디어 (코드로 확인)**:
  1. **Average Perturbation Hessian (APH)**: 출력에 ±미소 섭동(`±1e-6`)을 주고 KL-div loss의 그래디언트 차분으로 **Hessian(곡률) 근사**, 배치 평균 사용 (`utils/block_recon.py:228-248`).
  2. **MRECON / MLP Reconstruction**: GELU를 clamp 기반으로 대체하고 MLP를 재구성(가중치 미세조정)하여 양자화 친화적 활성 분포 확보 (`utils/mlp_recon.py:19-173`).
  3. **블록 단위 reconstruction + AdaRound**: 블록별로 weight 반올림(AdaRound)·activation scale을 Hessian-가중 손실로 최적화 (`utils/block_recon.py:271-372`).
  4. **QDrop**: 재구성 시 양자화/FP 입력을 확률적으로 섞어 일반화 (`block_recon.py:280-281, 308-311`).
- 양자화 자체(log/uniform/twin/FPCS)는 AdaLog와 동일 → 본 문서는 **APH·재구성** 위주로 정밀 분석하고, 공유 부분은 요약·교차참조(AdaLog.md).

---

## 2. 디렉토리 구조 (자체 + 제외)

### 자체 핵심 소스
```
APHQ-ViT/
├── test_quant.py                  # PTQ 양자화·재구성·평가 엔트리포인트
├── configs/{2,3,4}bit/            # best.py / brecq_*.py / qdrop_*.py / ablation_*.py
├── quantizers/                    ★ (AdaLog와 동일 골격)
│   ├── uniform.py / adaround.py / _ste.py
│   │   (주의: AdaLog의 logarithm.py에 해당하는 별도 파일은 미확인 — 4bit best는 uniform 중심, 9장)
├── quant_layers/                  ★ linear.py / matmul.py / conv.py  (AdaLog 기반)
├── utils/
│   ├── block_recon.py  ★ 핵심     # APH(Average Perturbation Hessian) + 블록 reconstruction + LossFunction
│   ├── mlp_recon.py    ★ 핵심     # MLP reconstruction(GELU 대체·재학습)
│   ├── calibrator.py   ★         # 캘리브레이션 + grad_hook(Hessian용 그래디언트 수집)
│   ├── wrap_net.py / datasets.py / test_utils.py
└── README.md
```

### 제외 (지시에 따라 이름만 표기, 미분석)
- `Object-Detection/` — **외부 mmdetection 프레임워크**(`configs/*/*.py` 다수: dcn/fcos/retinanet/mask_rcnn 등). 제외 지시("mmdetection/mmcv 외부", "Object-Detection\mmdetection 등 외부 프레임워크는 이름만") 대상. APHQ-ViT의 검출 적용용이나 본 분석에서 제외.
- `.git/`, `__pycache__/`, `assets/`, `LICENSE`, 대용량 체크포인트(언급만) — 비소스.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 캘리브레이터 + 그래디언트 수집 — `utils/calibrator.py`
- AdaLog판과 거의 동일하나 **`grad_hook` 추가** (`calibrator.py:30-33`): `register_full_backward_hook`으로 `grad_output[0]`을 `tmp_grad`에 누적 → Hessian 근사용 그래디언트 캐싱.
- `batching_quant_calib()` (`:35-83`): raw softmax 예측 사전계산(`:37-44`) 후 모듈별 raw I/O 수집 + `hyperparameter_searching()` + 필요시 `reparam()` (`:73-78`). 끝나면 `mode='quant_forward'`.

### 3.2 **Average Perturbation Hessian + 블록 재구성 — `utils/block_recon.py` (가장 중요)**

#### (a) 블록 forward에 섭동 주입
- 각 블록 타입(PatchEmbed/ViT Block/Swin Block/PatchMerging)의 forward를 교체하며 **출력 끝에 `±1e-6` 섭동** 삽입:
  - `vit_block_forward` (`:32-39`): 정상 forward 후 `perturb_u`면 `x + 1e-6`, `perturb_d`면 `x - 1e-6` (`:35-38`).
  - 동일 패턴이 `patch_embed_forward`(`:25-28`), `swin_block_forward`(`:63-66`), `swin_patchmerging_forward`(`:75-78`)에도 적용.
- → 출력에 미소 양/음 섭동을 주어 **그래디언트 차분으로 Hessian(곡률)을 수치 근사**하기 위한 장치.

#### (b) `BlockReconstructor` (QuantCalibrator 상속) — `:82-372`
- 생성자(`:82-106`): `metric="hessian_perturb"`, `use_mean_hessian=True`, `temp=20`. ViT/Swin 블록·head를 수집해 forward 교체·섭동 플래그 초기화 (`:93-122`).
- **`init_block_perturb_hessian` (APH 핵심) — `:228-248`**:
  - step 0: `perturb_u=True`(출력 +1e-6), step 1: `perturb_d=True`(−1e-6) (`:235`).
  - 각 step에서 **FP 모델 예측 `pred = full_model(inp)/temperature`**, **KL-div loss = `KL(log_softmax(pred) ‖ raw_pred_softmax)`** 계산 후 `loss.backward()` (`:239-241`).
  - `grad_hook`으로 블록 출력 그래디언트 수집 → `raw_grads[0]`(+섭동), `raw_grads[1]`(−섭동) (`:243`).
  - **`block.raw_grad = |raw_grads[0] − raw_grads[1]|`** (`:247`): 양/음 섭동 그래디언트의 차분 절댓값 = **Perturbation Hessian 근사**.
  - `use_mean_hessian`이면 배치 평균 → **Average** Perturbation Hessian (`:248`).
- **비교군 `init_block_brecq_hessian` — `:250-267`**: 기존 BRECQ식. 양자화 모델로 1회 backward → `raw_grad = grad.abs().pow(2)` (`:265`). (APH는 ±섭동 차분, BRECQ는 단일 제곱 — config `ablation_brecq_hessian.py`로 비교 가능.)
- **`reconstruct_single_block` — `:271-350`**:
  - 블록 weight 양자화기를 **AdaRoundQuantizer로 래핑**(`wrap_quantizers_in_net`, `:139-151`), soft_targets=True (`:151`).
  - QDrop 설정(`set_qdrop`, `:153-163, 280-281`): activation 양자화기 `drop_prob` 지정.
  - 최적화 대상: weight는 AdaRound `alpha`(`:286`), 옵션으로 activation scale(`:288-296`). Adam optimizer (`:299-301`).
  - 손실: `LossFunction(rec_loss=metric)` (`:302-304`). head는 `kl_div` 사용(`:303`).
  - 반복(`:306-338`): 입력을 QDrop으로 섞고(`:308-311`), `out_quant=block(cur_inp)`, `err=loss_func(out_quant, cur_out, cur_grad)` (`:329-333`), backward+step.
  - 종료시 hard rounding 고정(soft_targets=False) (`:341-347`).
- **`reconstruct_model` — `:352-372`**: 블록 순차로 raw 데이터 초기화→APH 계산→재구성→`quanted_blocks`에 추가(다음 블록 입력은 양자화된 출력 사용) (`:357-364`). 끝나면 전 모듈 `quant_forward`, weight를 hard value로 확정(`:365-371`).

#### (c) **`LossFunction` (Hessian-가중 reconstruction loss) — `:374-451`**
- `__call__`(`:408-451`):
  - rec_loss가 `hessian_perturb`/`hessian_brecq`면 **`rec_loss = mean( (pred−tgt)^2 · |grad| )`** (`:425-426`) → **출력 오차를 (Average Perturbation) Hessian으로 가중한 quadratic reconstruction loss**.
  - mse/mae/kl_div 옵션도 지원(`:421-429`).
  - round_loss(AdaRound 정규화): `weight·Σ(1 − |2(round_vals−0.5)|^b)` (`:436-441`), b는 LinearTempDecay로 감쇠(`:393, 433, 454-471`).
  - 첫 스텝 손실로 정규화 `rec_loss·2/init_loss` (`:444-446`).

### 3.3 **MLP Reconstruction — `utils/mlp_recon.py`**
- `mlp_forward`(`:19-30`): `fc1→act(GELU)→drop→norm→fc2` 후 ±1e-6 섭동(`:26-29`). (MLP 내부에 추가 norm이 있는 변형 구조.)
- `MLPReconstructor` (`:52-173`): 블록의 MLP만 대상으로 APH 계산·재구성.
  - `init_block_perturb_hessian`(`:110-128`): 블록 단위와 동일한 ±섭동 KL-div backward로 `raw_grad=|grad_u−grad_d|`(`:126`), 평균(`:127`), **정규화 `raw_grad·sqrt(numel/Σgrad²)`** (`:128`).
  - `reconstruct_single_block`(`:130-157`): fc1/fc2/norm2의 weight·bias 직접 학습(Adam) (`:133-137`).
    - **GELU 대체/clamp**: `fc2_inp = act(fc1(x))`, **`fc2_quant_inp = clamp(fc2_inp, 0, ub)`** (`:149-150`) — 상한 `ub`는 positive percentile(기본 0.98)로 산출(`:170`, `reconstruct_model`). 양자화시 GELU 출력의 꼬리를 clamp하여 양자화 친화적으로 만듦.
    - 손실(`LossFunction`, `:177-221`): `rec_loss + weight·quant_loss` (`:217`), 각각 Hessian-가중(`:211-213`). recon출력과 clamp양자화출력을 동시에 FP 타깃에 맞춤.
- → **MRECON**: GELU의 비양자화 친화적 분포를 clamp+재학습으로 교정하는 것이 정확도 회복의 핵심 기여(추정 근거: 코드 구조 + README "MLP Recon." 결과 컬럼 `README.md:78-86`).

### 3.4 양자화 레이어/양자화기 (AdaLog 공유) — 요약
- `quant_layers/linear.py`, `matmul.py`, `conv.py`, `quantizers/uniform.py`, `adaround.py`, `_ste.py`는 AdaLog와 동일 골격(상세는 AdaLog.md §3 참조). 
- `quantizers/adaround.py`의 `AdaRoundQuantizer`(`:7-77`)가 블록 재구성에서 weight 학습형 반올림에 사용됨(soft `sigmoid(alpha)` → hard `(alpha≥0)`, `:48, 59-60, 71-73`).
- `wrap_quantizers_in_net`(`block_recon.py:139-151`)에서 `MinMaxQuantLinear/Conv2d`의 `w_quantizer`를 `AdaRoundQuantizer`로 교체.

### 3.5 설정 — `configs/4bit/best.py`
- `Config`(`4bit/best.py:2-28`): `w_bit=a_bit=4`, `qconv_a_bit=8`, `qhead_a_bit=4` (`:9-12`), `calib_metric='mse'`(`:13`), **`optim_metric='hessian_perturb'`, `use_mean_hessian=True`, `temp=20`** (`:20-22`), **`recon_metric='hessian_perturb'`, `pct=0.98`(MLP clamp 상한)** (`:24-25`), **QDrop `optim_mode='qdrop', drop_prob=0.5`** (`:27-28`).
- 다른 config: `brecq_baseline.py`, `qdrop_baseline.py`, `brecq_qinp.py`, `ablation_brecq_hessian.py`, `ablation_avg.py`(3bit) — APH vs BRECQ vs avg Hessian 어블레이션용(Glob `configs/3bit/`).

---

## 4. 알고리즘 / 수식

### 4.1 Average Perturbation Hessian (APH) objective
출력 섭동 `±δ (δ=1e-6)`에 대한 KL-div loss 그래디언트 차분으로 곡률(Hessian) 근사:
```
g₊ = ∂/∂z  KL( softmax(z₊) ‖ p_FP ),  z₊ = block_out + δ        # perturb_u
g₋ = ∂/∂z  KL( softmax(z₋) ‖ p_FP ),  z₋ = block_out − δ        # perturb_d
H_perturb ≈ | g₊ − g₋ |                                          # (block_recon.py:247)
APH = mean_batch( H_perturb )                                    # use_mean_hessian (:248)
```
KL은 `KL(log_softmax(full_model(x)/T) ‖ raw_pred_softmax)`, T=20 (`block_recon.py:239-240`).

### 4.2 Hessian-가중 reconstruction loss
```
L_rec = mean( (out_quant − out_FP)² · |APH| )                    # (block_recon.py:426)
L_round = weight · Σ_w ( 1 − |2·(h(α) − 0.5)|^b )               # AdaRound 정규화 (:436-441)
L = L_rec / init_loss · 2 + L_round                              # (:444-447)
```
b는 LinearTempDecay로 start_b→end_b 감쇠 (`:454-471`).

### 4.3 MLP reconstruction (GELU clamp)
```
fc2_quant_inp = clamp( GELU(fc1(x)), 0, ub ),  ub = positive_percentile(fc2_in, 0.98)
L = L_rec(recon_out, FP) + weight · L_quant(quant_out, FP)       # 둘 다 |APH| 가중 (mlp_recon.py:204-217)
```

### 4.4 QDrop 입력 혼합
```
cur_inp = where( rand < drop_prob, quanted_input, fp_input )     # (block_recon.py:308-311)
```

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet (ILSVRC), `--dataset <DATA_DIR>` (`README.md:46`).
- **명령** (`README.md:38-71`):
  ```bash
  python test_quant.py --model <MODEL> --config <CONFIG> --dataset <DIR> \
      [--reconstruct-mlp] [--load-reconstruct-checkpoint <CKPT>] \
      [--calibrate] [--load-calibrate-checkpoint <CKPT>] [--optimize]
  # 예: MLP 재구성 + 캘리브 + 최적화
  python test_quant.py --model vit_small --config ./configs/3bit/best.py \
      --dataset <DIR> --val-batchsize 500 --reconstruct-mlp --calibrate --optimize
  ```
  - `--model`: deit/vit/swin {tiny,small,base} (`README.md:42`).
  - `--reconstruct-mlp`: MLP reconstruction 사용(§3.3) (`README.md:48`).
  - `--optimize`: 캘리브 후 AdaRound(블록 재구성) 최적화(§3.2) (`README.md:54`).
- **결과(README.md:78-86)**: W4/A4 — ViT-S 76.07, ViT-B 82.41, DeiT-B 80.21, Swin-B 83.42. W3/A3 — ViT-B 76.31, Swin-B 78.14. "MLP Recon." 단독 컬럼도 보고(FP 근접).
- 검출(Object-Detection)은 외부 mmdetection 경유(제외).

---

## 6. 의존성

- PyTorch (README 기준 1.10.0), **timm 0.9.2 권장** (`README.md:16-22`), numpy, tqdm.
- timm 구조 직접 참조: `timm.layers.patch_embed.PatchEmbed`, `timm.models.vision_transformer.Block`, `timm.models.swin_transformer.{SwinTransformerBlock, PatchMerging, window_partition, window_reverse}` (`block_recon.py:6-7, 93-98`).
- CUDA 전제(AdaLog와 동일, GPU 메모리 기반 탐색·`.cuda()` 다수). 검출 파이프라인은 mmdetection/mmcv 별도 필요(제외).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **APH**로 출력 중요도(곡률)를 ±섭동 차분으로 근사 → BRECQ 단일 제곱 Hessian 대비 더 안정적 가중(어블레이션 config 제공).
- **MLP reconstruction(GELU clamp+재학습)** 으로 post-GELU 분포를 양자화 친화적으로 교정 → 저비트 정확도 회복.
- AdaLog 기반이라 log/uniform/twin/FPCS 등 양자화 기법을 그대로 활용.
- W4A4에서 SOTA급 정확도(`README.md:78-86`).

**한계 / 리스크**
- **재구성(reconstruction) 비용**: 블록당 수만 iteration AdaRound 최적화(`block_recon.py:272 iters 기본 20000`) → 캘리브치곤 무겁다(여전히 PTQ지만 GPU 시간 큼).
- I-ViT 같은 **정수 실연산 커널 없음** — fake-quant 시뮬레이션. 하드웨어 실행 매핑 별도 필요.
- APH는 KL-div + temperature(T=20) 등 하이퍼파라미터 의존(`block_recon.py:92, 239`).
- 검출 적용은 외부 mmdetection 의존(이식성·버전 민감).

**확인 불가 리스크**: 본 repo에 AdaLog의 `logarithm.py` 대응 파일을 Glob에서 직접 못 봄(9장) → log 양자화 사용 여부는 config(`post_*_quantizer`)·linear/matmul 코드로 추정. 4bit best는 uniform 중심으로 보임(추정).

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적; 프로젝트 성격 추정)

- **APHQ-ViT의 역할 = PTQ 정확도 회복 기법**: 본 프로젝트 하드웨어(HG-PIPE류)에 직접 들어가는 정수 연산기는 I-ViT 쪽이지만, **저비트(W4A4) 배포 정확도를 끌어올리는 소프트웨어 단계**로 APHQ-ViT의 APH·MLP reconstruction을 채택하면 FPGA에 올릴 양자화 모델의 품질을 높일 수 있음.
- **MLP/GELU clamp(`mlp_recon.py:149-150`, `pct=0.98`)**: post-GELU 활성을 `[0, ub]`로 clamp하면 하드웨어에서 **활성 비트폭·동적범위가 축소**되어 GELU 후단 데이터패스(I-ViT의 IntGELU나 AdaLog의 log 양자화)와 결합 시 회로 단순화·정확도 동시 확보 가능(추정).
- **APH 중요도 맵의 하드웨어 함의(추정)**: 블록/채널별 곡률(`raw_grad`)은 어느 레이어가 양자화에 민감한지를 정량화 → **혼합정밀(mixed-precision) 비트할당** 또는 민감 블록만 고정밀 처리하는 FPGA 자원 배분 정책의 근거로 활용 가능.
- **권장 조합(추정)**: ① 분포 적합/저비트 회복 = APHQ-ViT(APH+MRECON) → ② 시프트 친화 후단 양자화 = AdaLog(log2) → ③ 정수 실연산·dyadic requant = I-ViT. 세 repo를 단계적으로 결합하는 것이 시선추적용 경량 ViT의 FPGA화에 가장 합리적.
- **XR 시선추적 적용(추정)**: 시선추적은 정확도-지연 트레이드오프가 첨예. APH로 민감 레이어를 식별·보호하면서 나머지를 공격적으로 저비트화하면 정확도 손실을 최소화한 채 FPGA 자원/지연을 줄일 수 있음.

---

## 9. 근거 표기 / 확인 불가 항목

- **직접 코드 확인**: §3~§4의 라인 인용(`utils/block_recon.py` 전체, `utils/mlp_recon.py` 전체, `utils/calibrator.py` 전체, `quantizers/adaround.py`, `configs/4bit/best.py`), §2 구조(Glob), §5 명령/결과(`README.md`).
- **추정**: 프로젝트 성격, MRECON이 정확도 회복의 핵심이라는 인과, 혼합정밀 활용 가능성, 3-repo 결합 경로, 4bit best의 uniform 중심 추정.
- **확인 불가(미열람/미발견)**:
  - `quantizers/logarithm.py` 대응 파일을 Glob 결과에서 직접 확인 못함 → APHQ-ViT의 log 양자화 사용 여부/구현은 미확정(linear.py/matmul.py 내부 import로 존재 가능, 미열람).
  - `quant_layers/{linear,matmul,conv}.py`, `quantizers/uniform.py`, `utils/wrap_net.py`, `test_quant.py` — APHQ판은 직접 미열람(AdaLog 동일 골격으로 추정, 상세는 AdaLog.md 교차참조).
  - `Object-Detection/` 전체 — 외부 mmdetection으로 제외(이름만).
