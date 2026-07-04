# MimiQ 코드베이스 정밀 분석

> 대상: `REF/ViT-Quantization/mimiq`
> 분석 방식: 실제 소스(Read) 기반. 모든 핵심 근거는 `파일:라인`으로 표기.
> 근거 표기 규칙: 코드로 확인된 사실은 단정, 코드 외 추론은 "추정", 미확인은 "확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **정식 명칭**: *MimiQ: Low-Bit Data-Free Quantization of Vision Transformer with Encouraging Inter-Head Attention Similarity* (`README.md:1`).
- **목적**: 실제 학습 데이터(ImageNet train set) 없이(**data-free**) ViT/DeiT/Swin을 저비트(예: W4A4)로 양자화하는 PTQ-기반 QAT 추정 프레임워크.
- **핵심 아이디어 2가지** (코드로 확인):
  1. **어텐션 맵 기반 합성 데이터 생성**: 사전학습 모델의 **inter-head attention map 유사도(SSIM)** 와 cross-entropy를 손실로 입력 이미지를 직접 최적화하여 합성 학습셋을 생성 (`hydra_image_gen_ssim_att_map.py:91~119`, `:189~252`).
  2. **Inter-Head Attention Similarity 증류**: 양자화 모델(student)과 FP 모델(teacher)의 헤드별 어텐션 출력(matmul2)을 SSIM 거리로 정렬하는 **head-wise distillation** (`trainer.py:67~104`, `:325~335`).
- **양자화 방식**: 가중치는 LSQ(학습형 step) 또는 min/max symmetric, 활성은 LSQ 또는 min/max asymmetric. KD(KL+CE) + head-distill로 fine-tune → 실질적으로 **합성데이터 기반 QAT** (`main.py:194~241`, `quant_utils/quant_modules.py:133~234`).
- 코드 헤더상 양자화 유틸은 **ZeroQ 저장소**에서 파생, 어텐션 양자화 아이디어는 **PTQ4ViT**에서 차용 (`quant_utils/quant_modules.py:21`, `:789`).

---

## 2. 디렉토리 구조 (자체 소스 / 제외)

```
mimiq/
├── main.py                          # 양자화+학습 엔트리 (ExperimentDesign)
├── trainer.py                       # 학습/평가 루프, head-wise SSIM 증류
├── options.py                       # .hocon 설정 파서 (확인 불가: 본문 미열람, main.py에서 Option import)
├── dataloader.py                    # 검증셋/합성셋 DataLoader (요지만 main.py에서 확인)
├── hydra_image_gen_ssim_att_map.py  # ★합성 데이터 생성 (어텐션 SSIM + CE + TV)
├── hydra_image_gen_merge.py         # 생성 .pt 병합 (확인 불가: 본문 미열람)
├── image_gen_aug_utils.py           # 생성용 증강 (ColorJitter, GaussianNoise, Zoom 등)
├── gaussian_blur.py                 # 학습 증강용 GaussianBlur (trainer.py에서 import)
├── quant_utils/
│   ├── quant_utils.py               # ★양자화 함수 (linear_quant/dequant, scale/zp, STE)
│   └── quant_modules.py             # ★Quant_Linear/Conv/Matmul/Act, quantize_model 등
├── utils/                           # LRPolicy, AverageMeter, arglist 등 보조
├── imagenet_{deit,vit,swin}_*.hocon # 모델별 설정 (8개)
├── train.sh / generate_dataset.sh / merge_dataset.sh
└── requirements.txt
```

**제외 대상**: `.git/`, `__pycache__/`, pack 파일. (third_party/대용량 체크포인트는 본 repo에 없음.)

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `quant_utils/quant_utils.py` — 양자화 수치 연산 핵심

저수준 양자화 함수 모음. `@torch.jit.script`로 JIT 컴파일.

- **`linear_quantize(input, scale, zero_point)`** (`:41~69`): `round(scale*input - zero_point)`. conv(4D)/linear(2D) shape에 따라 scale/zp를 채널축으로 reshape (`:52~64`). → **per-channel(가중치)** 지원.
- **`linear_dequantize`** (`:71~92`): `(input + zero_point) / scale`. 역연산.
- **`asymmetric_linear_quantization_params(num_bits, sat_min, sat_max)`** (`:94~116`):
  - `n = 2^bits - 1`, `scale = n / clamp(max-min, min=1e-8)`, `zero_point = round(scale*min)`, signed면 `zp += 2^(bits-1)` → **asymmetric uniform affine** (`:105~115`).
- **`symmetric_linear_quantization_params`** (`:118~134`): `max_max = max(-min, max)`, `scale = n / clamp(2*max_max)`, `zp = 0` → **symmetric** (가중치 기본).
- **`merged_quantization_internal*`** (`:136~171`): quant→clamp(`-n, n-1`)→dequant 일괄(fake-quant). sym/asym/act 3종.
- **Autograd Function**:
  - `SymmetricQuantFunction` / `AsymmetricQuantFunction` (`:173~239`): forward는 fake-quant, **backward는 STE(grad_output 그대로 통과)** (`:203~205`, `:237~239`).
  - `AsymmetricQuantFunctionAct` (`:241~273`): act 전용, scale/zp 직접 입력.
  - `AsymmetricQuantFunctionPerturb` / `PerturbNorm` (`:275~369`): **gradient 기반 양자화 섭동**(quant int에 ±1 perturbation 추가) — 실험용 기능. 본 메인 경로에서는 미사용 추정.
- **LSQ 헬퍼**: `grad_scale(x, scale)` (`:371~374`), `round_pass(x)` (`:377~380`) — STE 형태(`y.detach()-y_grad.detach()+y_grad`).

### 3.2 `quant_utils/quant_modules.py` — 양자화 레이어 & 모델 변환

#### Activation Quantizer
- **`QuantAct`** (`:38~130`): min/max asymmetric act 양자화. running stat로 x_min/x_max를 **beta=0.9 EMA + bias 보정**(`beta_t`)로 갱신 (`:121~124`). `fix()/unfix()`로 range 동결/해제 (`:74~84`). `activation_bit>=32`면 FP (`:65~66`).
- **`QuantAct_lsq`** (`:133~234`): **LSQ(Learned Step Size Quantization)**. 학습 파라미터 `alpha`(step), `zero_point` (`:151~152`). 초기화 시 입력 부호 판별(min<1e-5 → signed) (`:194~205`), `alpha = 2*mean(|x|)/sqrt(q_max)` (`:203`). forward에서 `grad_scale`로 step gradient 스케일링 후 `round_pass`로 STE quant (`:214~227`). `g_scale = lsq_g_scale/sqrt(numel*q_max)` (`:207`).

#### Weight Quantizer Layers
- **`Quant_Linear`** (`:240~426`): `weight_q_mode ∈ {lsq, minmax, minmax_asym}` (`:265~274`).
  - `_lsquant(w)` (`:286~292`): symmetric LSQ. `w_q = round_pass(clamp(w/alpha, q_min, q_max)) * alpha`. **per-output-channel alpha** (`out_features`개) (`:311~314`).
  - minmax 모드는 per-row(dim=1) min/max로 SymmetricQuantFunction 적용 (`:334~340`).
  - `set_param(linear)` (`:301~315`): FP weight/bias 복사, LSQ alpha 초기화.
  - `eval_mode`에서 quant weight 캐싱 (`:328~329`, `eval_mode_init` `:405~415`).
  - `update_bit()` (`:417~426`): 비트 변경 시 q_min/q_max/alpha 재초기화.
- **`Quant_Conv2d`** (`:615~786`): 위와 동일 구조, conv weight는 `view(out_channels,-1)` per-channel (`:708~712`, `:656~662`).
- **`Quant_Matmul`** (`:545~612`): 어텐션 내부 Q@K, attn@V 양자화용. **두 입력 각각 별도 act quantizer**(`quant_act1`, `quant_act2`) (`:570~575`). weight 양자화 없음(matmul은 weightless).
- `Quant_AvgPool2d`/`Quant_MaxPool2d` (`:429~543`): act만 양자화 (ViT 경로에서는 비활성 추정).

#### 어텐션 구조 처리 (PTQ4ViT 차용)
- **`MatMul`** (`:791~793`): `A @ B`를 nn.Module로 래핑 → 양자화 hook 부착 가능하게.
- **`attention_forward`** (`:795~811`): timm `Attention.forward`를 monkey-patch. `attn = matmul1(q, k^T)*scale → softmax → matmul2(attn, v)` (`:801~807`). **matmul1/matmul2가 합성데이터 생성·증류의 어텐션 맵 hook 지점**.
- **`window_attention_forward`** (`:813~841`): Swin WindowAttention용 동일 패치 (relative position bias 포함 `:822~825`).
- **`prepare_vit_model(module)`** (`:843~855`): 모든 `Attention`/`WindowAttention`에 matmul1/matmul2 부착 + forward 교체.

#### 모델 변환 함수
- **`quantize_model(module, qw, qa, weight_q_mode, act_q_mode, lsq_g_scale)`** (`:860~889`): 재귀적으로 `nn.Conv2d→Quant_Conv2d`, `nn.Linear→Quant_Linear`, `MatMul→Quant_Matmul` 치환.
- `quantize_model_only_attn` (`:891~899`): 어텐션 블록만 양자화.
- **`set_first_last_layer(model, fl8bit)`** (`:932~948`): **첫 conv/linear act는 FP 유지**, 옵션으로 첫/마지막 layer를 8bit로 (`:939~947`). → 양자화 민감 레이어 보호.
- `freeze_model`/`unfreeze_model` (`:950~980`): act range 동결/해제.
- `to_train_mode`/`to_eval_mode` (`:1009~1047`): weight quant 캐싱 on/off ('norm' 포함 모듈은 제외 — LayerNorm 비양자화).

### 3.3 `hydra_image_gen_ssim_att_map.py` — ★합성 데이터 생성

`torch.randn` 노이즈를 입력으로 두고, 이미지 픽셀 자체를 AdamW로 최적화한다(`img_train.requires_grad=True`, `:175~177`).

- **`get_image_prior_losses(inputs_jit)`** (`:48~56`): **Total Variation(TV) 정규화** — 인접 픽셀 4방향 차분의 L2 norm 합. 자연스러운 이미지 prior.
- **`ssim_loss_att_map(att_hook_output, ...)`** (`:91~118`): ViT/DeiT용. 어텐션 맵(matmul1 출력, q@k^T)을 헤드별로 min-max 정규화 후 14×14 spatial map으로 재배열 (`:93~98`), SSIM의 평균·분산·공분산을 계산하여 **헤드 i,j 간 SSIM 매트릭스**(L B H H) 산출 (`:99~110`). `mode='loss'`면 평균.
- **`ssim_loss_att_map_swin`** (`:58~87`): Swin은 윈도우(7×7) 단위로 동일 SSIM 계산.
- **`generate(args, save_dir, logger)`** (`:122~251`):
  - timm 사전학습 모델 로드 → eval → `prepare_vit_model`로 matmul hook 부착 (`:125~171`).
  - 증강: `GaussianNoise`, `ColorJitter` (`:139~148`).
  - 반복(`args.iter`, 기본 2000) (`:189~244`):
    - random jitter(roll) + flip + aug 적용 (`:200~209`).
    - `output = model(img_jit)` → SSIM loss + CE loss + TV loss 합산.
    - **총 손실** (`:229`): `total = ssim_coef*ssim_loss + class_coef*class_loss + tv_coef*loss_tv`.
    - **SSIM loss는 `(1 - ssim_map^2).mean()`** 형태로, 헤드 간 어텐션을 **유사하게(=서로 다르게 멀어지지 않게)** 유도 (`:220`). [추정: 헤드 다양성 확보를 위해 SSIM 제곱 패널티]
  - clamp_training 시 정규화 범위로 픽셀 클램프 (`:238~239`).
  - 결과 이미지/라벨을 `.pt`로 저장 (`:248~251`).

### 3.4 `trainer.py` — 학습/증류 루프

- **`ssim_dist_loss(teacher_attn, quant_attn, distance)`** (`:67~104`): **teacher↔student 어텐션 출력(matmul2=attn@v) 정렬**. SSIM 거리 `(2σ_tq + C2)/(σ_t² + σ_q² + C2)`, `loss = 1 - ssim` (`:74~94`). distance로 mse/kl/l1도 선택 가능 (`:99~104`). Swin은 `ssim_dist_loss_swin` (`:26~64`).
- **`Trainer.loss_fn_kd(output, labels, teacher_outputs)`** (`:212~232`): KD 손실 `kd_scale*KLDiv(logsoftmax(out/T), softmax(teacher/T))*α*T² + ce_scale*CE(out, labels)` (`:226~231`).
- **`Trainer.train(epoch)`** (`:263~387`):
  - teacher/student의 `matmul2`에 forward hook 부착 → 어텐션 출력 수집 (`:282~290`).
  - 합성/실데이터 이미지에 data_transforms(color jitter, grayscale, gaussian blur) 적용 (`:305`, init `:168~199`).
  - teacher forward(no_grad) → student forward → `loss_S`(KD) + `head_dist_coef*head_dist_loss`(SSIM 증류) (`:321~335`).
  - grad accumulation(`grad_acc`), warmup 이후 act range 동결 + optimizer step (`:342~348`).
  - `random_samples`면 `train_random`(순수 가우시안 노이즈 학습) (`:390~504`).

### 3.5 `main.py` — 엔트리 (`ExperimentDesign`)

- `_set_model` (`:150~155`): timm으로 student/teacher 동일 사전학습 모델 2개 생성.
- `_set_dataloader` (`:86~147`): `real_data`면 실제 ImageNet, 아니면 **합성 `.pt` 데이터셋 로드**(`{network}_ssim_{...}_merged.pt`) (`:113~123`). 검증셋은 항상 실제. 모델 depth/head 수 추출 (`:134~147`).
- `_replace` (`:184~191`): `prepare_vit_model` → `quantize_model` → `set_first_last_layer`.
- `run` (`:194~258`): warmup epoch 동안 act range unfreeze→train→freeze→eval 반복, best top-1 저장 (`:211~235`).
- main args (`:262~300`): `qw/qa`(비트), `head_dist_coef`(증류 계수, 기본 10.0), `head_dist_distance`(기본 ssim), `wq_mode`(기본 minmax), `aq_mode`(기본 lsq), `lsq_g_scale`(0.01).

---

## 4. 알고리즘 / 수식

### 4.1 합성 데이터 생성 손실 (data generation)
입력 이미지 `x`(초기 가우시안 노이즈)에 대해:

```
L_gen(x) = λ_ssim · mean(1 - SSIM_inter-head(A(x))²)
         + λ_cls  · CE(f(x), y_rand)
         + λ_tv   · TV(x)
```
- `A(x)`: q@kᵀ 어텐션 맵, SSIM_inter-head는 헤드 i,j 쌍 SSIM (`hydra_image_gen_ssim_att_map.py:91~118`, `:220`).
- `y_rand`: 무작위 클래스 라벨 (`:180~185`).
- 기본 계수: `ssim_coef=1.0`, `class_coef=1.0`, `tv_coef=2.5e-5` (`:262~266`).
- 최적화: AdamW(lr=0.1) + CosineAnnealing, 2000 iter (`:177~178`).

### 4.2 양자화 (fake-quant)
- **Asymmetric (act)**: `q = clamp(round(scale·x - zp), -n, n-1)`, `x̂ = (q + zp)/scale`, `scale = (2^b-1)/(max-min)` (`quant_utils.py:94~116`, `:149~160`).
- **LSQ (weight, symmetric)**: `ŵ = round_pass(clamp(w/α, q_min, q_max))·α`, step `α` 학습, gradient scale `g=1/sqrt(N·q_max)` (`quant_modules.py:286~292`, `:314`).
- STE backward로 grad 통과 (`quant_utils.py:203~205`, `:377~380`).

### 4.3 양자화 학습 손실 (QAT on synthetic data)
```
L = L_KD + γ · L_head-distill
L_KD = kd_scale · KL(σ(z_q/T) ‖ σ(z_t/T))·α·T² + ce_scale · CE(z_q, y)
L_head-distill = mean( 1 - SSIM(attn_v^teacher, attn_v^student) )
```
- `γ = head_dist_coef`(기본 10.0) (`main.py:277`, `trainer.py:335`).
- DeiT-Base 기본 비트 `qw=qa=4` (`imagenet_deit_b_16_224.hocon:30~31`).

---

## 5. 학습 / 평가 파이프라인

### 데이터셋
- **학습**: 합성 `.pt` 데이터셋(라벨 포함) 또는 순수 가우시안 노이즈(random_samples). 실데이터는 옵션(`--real_data`).
- **평가**: 항상 실제 ImageNet 검증셋 (`main.py:111`, `:125`).

### 명령어 (README + 스크립트)
1. **합성 데이터 생성** (`generate_dataset.sh:3`):
   ```
   python3 hydra_image_gen_ssim_att_map.py --model deit_base_patch16_224 \
       --num_images N --save_prefix P --save_path PATH --clamp_training
   ```
   이후 `merge_dataset.sh SAVE_PATH MODEL_NAME`으로 병합.
2. **양자화 학습** (`train.sh:3`):
   ```
   python3 main.py --conf_path CONF --id ID --lrs LR --qw QW --qa QA \
       --head_dist_coef GAMMA --dataset_path DATA --lr_policy POL --lr_step STEP --aq_mode {minmax|lsq} --bs 16
   ```

### 비트 설정 (.hocon 예: DeiT-Base)
- `qw=4, qa=4` (`imagenet_deit_b_16_224.hocon:30~31`), `nEpochs=200`, `batchSize=32`, `warmup_epochs=5`, `lr_S=1e-6`, opt=SGD (`:13~20`). 모델별 hocon 8개(tiny/small/base × vit/deit/swin).

---

## 6. 의존성 (`requirements.txt`)
- `torch==2.0.1`, `torchvision==0.16.1`, `timm==0.9.8` (모델 백본 timm 의존, `:47~49`).
- `einops==0.7.0`(rearrange), `pyhocon==0.3.60`(설정 파서), `numpy`, `Pillow`. CUDA 11/12 wheel 다수.
- Python 3.9.18 (`README.md:7`).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **완전 data-free**: 사전학습 가중치만으로 합성셋 생성 → 프라이버시/데이터 접근 제약 환경에 적합.
- **어텐션 특화**: ViT 고유의 inter-head attention 구조를 생성·증류 양면에서 활용 (단순 BN-statistics matching 대비 ViT에 정합적).
- LSQ + min/max 혼용, 첫/마지막 레이어 보호 등 실전 PTQ 노하우 반영 (`quant_modules.py:932~948`).

**한계 / 리스크**
- **합성 데이터 생성 비용**: 이미지당 2000 iter AdamW 최적화 → 대량 생성 시 GPU 시간 큼 (`:189`).
- **LayerNorm/Softmax 비양자화**: 'norm' 모듈은 양자화 제외(`:1025`, `:1045`), softmax도 FP — 완전 정수 추론 아님 (FPGA 관점에서 별도 처리 필요).
- 코드 내 디버그/주석 처리 코드, perturbation 실험 함수 다수 잔존 → 유지보수성 낮음.
- 검증은 ImageNet 분류만 확인. 다른 태스크 전이는 확인 불가.

---

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적) 관점 시사점

> 아래는 우리 프로젝트 맥락에 대한 **추정**이며, 코드 사실과 분리하여 기술.

- **data-free PTQ의 배포 이점 (추정)**: XR 시선추적 ViT를 FPGA에 올릴 때, 캘리브레이션용 실제 시선 데이터셋 확보가 어렵거나 프라이버시 이슈가 있을 수 있음. MimiQ식 합성 캘리브레이션은 **타깃 디바이스에서 보정 데이터 없이 W4A4 양자화**를 가능케 함 → HG-PIPE 류 정수 데이터패스에 직접 이식 가능한 양자화 산출물.
- **LSQ step(α) per-channel 구조**는 FPGA에서 **채널별 scale 곱셈(또는 shift)** 으로 매핑되며, symmetric weight(zp=0)는 zero-point 덧셈 회로를 절감 → 하드웨어 친화적 (`quant_modules.py:286~292`).
- **첫/마지막 레이어 8bit·act FP 보호 정책**(`:932~948`)은 FPGA에서 mixed-precision 데이터패스 설계 시 참조 가능 (정확도-자원 트레이드오프).
- **주의 (추정)**: LayerNorm/Softmax가 FP로 남으므로, HG-PIPE식 완전 양자화 파이프라인에 쓰려면 별도의 정수 LN/Softmax 근사가 필요. MimiQ 자체는 그 부분을 다루지 않음(확인됨).
- 합성데이터 생성기는 **학습 단계 오프라인 도구**로, 추론 가속기 RTL과 무관 → FPGA 측에 부담 없음(추정).

---

## 9. 근거·불확실성 표기 요약
- 단정 서술은 모두 `파일:라인` 근거 보유.
- `options.py`, `dataloader.py`, `hydra_image_gen_merge.py` 본문은 미열람 → 해당 동작은 main.py 호출부 기준 **추정** 또는 **확인 불가**로 표기.
- 8장 시사점은 프로젝트 맥락 **추정**이며 코드 직접 근거 없음.
