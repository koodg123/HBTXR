# PQV-Mobile 코드베이스 정밀 분석

> 분석 대상 repo: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\PQV-Mobile`
> 분석 방식: Glob/Grep/Read 기반 정적 분석 (bash 미사용). 모든 코드 근거는 `파일:라인` 으로 표기.
> 작성일: 2026-06-20

---

## 1. 개요

### 1.1 정체
- **PQV-Mobile** = *"Pruning and Quantization framework for mobile applications of Vision Transformers (ViTs)"*.
- 원논문: arXiv **2408.08437** (`README:2` 에 직접 링크). LLNL(Lawrence Livermore National Laboratory) 산하 작업이며 코드 릴리즈 번호 **LLNL-CODE-865374** (`NOTICE.md:1-3`). 라이선스는 MIT, Copyright 2024 kshitij11 (`LICENSE:1-3`).
- 모바일/엣지 배포를 겨냥해 사전학습된 **timm ViT(특히 Facebook DeiT-Base/16)** 를 **structured pruning + INT8 quantization** 으로 압축하고, 압축 전후 모델의 **CPU 추론 latency** 를 직접 측정하는 도구 모음이다.

### 1.2 핵심 아이디어 (코드 기반 확인)
1. **Structured Pruning** — `torch_pruning`(VainF/Torch-Pruning) 라이브러리를 그대로 사용해 ViT의 attention head-dim/head 수 및 FFN 채널을 구조적으로 제거 (`prune_timm_vit.py:7,155-167`). importance 기준으로 Taylor/Hessian/L1/L2/Random 선택 가능 (`prune_timm_vit.py:115-125`).
2. **Quantization** — PyTorch eager-mode **dynamic quantization** (`torch.quantization.quantize_dynamic`) 을 `nn.Linear` 에만 적용, INT8(`torch.qint8`), **x86 백엔드** (`quant.py:117-121,151-155`).
3. **Mobile 변환 + Latency 측정** — TorchScript 직렬화 → `optimize_for_mobile` → Lite Interpreter(`.ptl`) 로 변환 후, `torch.autograd.profiler.profile(use_cuda=False)` 의 `self_cpu_time_total` 으로 CPU latency(ms)를 측정 (`quant.py:102-135,165-176`).
4. **Finetune** — pruning 으로 떨어진 정확도를 ImageNet 분산 학습으로 회복 (`finetune.py`, `finetune_timm_deit_b_16_taylor_uniform.sh`). 단, README는 finetune이 latency에는 영향을 주지 않음을 명시 (`README:17`).

### 1.3 전형적 워크플로 (`README:13-17`)
```
prune_timm_vit.py   →  pruned/model_taylor_0.25.pth 생성 (구조적 프루닝)
finetune.py (sh)    →  pruned 모델 정확도 회복 (선택)
quant.py            →  unpruned/pruned 모델을 x86 INT8 양자화 + latency 측정
```

---

## 2. 디렉토리 구조

### 2.1 자체(핵심) Python 소스 — 4개 (Glob 확인)
| 파일 | 역할 | 라인 수 |
|---|---|---|
| `prune_timm_vit.py` | timm ViT 구조적 프루닝 (Taylor 등 importance) | 225 |
| `quant.py` | INT8 dynamic quantization + mobile 변환 + latency 측정 | 179 |
| `finetune.py` | pruned 모델 ImageNet 미세조정 (분산 학습 루프) | 605 |
| `presets.py` | 데이터 전처리(train/eval) preset 클래스 | 115 |

> 4개 파일 모두 헤더에 `## Modified from / Taken from Torch-Pruning Package` 주석을 달고 있음 (`prune_timm_vit.py:1`, `quant.py:1`, `finetune.py:1`, `presets.py:1`). 즉 VainF/Torch-Pruning의 예제 스크립트를 ViT mobile용으로 개조한 것.

### 2.2 스크립트/설정
| 파일 | 역할 |
|---|---|
| `finetune_timm_deit_b_16_taylor_uniform.sh` | DeiT-B/16 + Taylor uniform pruning 모델의 finetune 명령 (torchrun 4-GPU) |
| `requirments.txt` | 의존성 (오타 그대로; torch 2.0.0, torchvision 0.15.1, timm 0.8.17.dev0, torch-pruning) |
| `README` | 사용법 + timm 패치 안내 |

### 2.3 라이선스/고지 (간략)
- `LICENSE` : MIT License, Copyright (c) 2024 kshitij11 (`LICENSE:1-3`).
- `NOTICE.md` : 미 DOE/LLNL 산하 작업 고지, 코드 번호 LLNL-CODE-865374, 책임 면책 조항 (`NOTICE.md:1-18`).
- `.git/`, `__pycache__` 제외.

### 2.4 **누락 의존 모듈 주의 (확인됨)**
`finetune.py` 는 다음 모듈을 import 하지만 **repo 내에 존재하지 않음** (Glob 결과 `.py` 파일 4개뿐):
- `import utils` (`finetune.py:14`) — `MetricLogger`, `accuracy`, `init_distributed_mode`, `set_weight_decay`, `ExponentialMovingAverage` 등 사용.
- `from sampler import RASampler` (`finetune.py:15`).
- `from transforms import get_mixup_cutmix` (`finetune.py:20`).

이 셋은 **torchvision references/classification** 의 표준 유틸리티이며, 사용자가 별도로 가져와야 finetune.py가 동작함(추정). 또한 `evaluate()` 함수가 전역 `args`를 참조하는 버그성 코드가 있음 (`finetune.py:110` 의 `args.is_huggingface` — `evaluate`는 `args`를 인자로 받지 않음) → 단독 실행 시 `NameError` 발생 가능 (확인됨, 정적 분석 기준).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `quant.py` — 양자화 + 모바일 변환 + latency 측정 (가장 중요)

#### 3.1.1 양자화 방식: PyTorch eager-mode **Dynamic Quantization** (확인됨)
- 양자화 핵심 호출 (2회, unpruned/pruned 각각):
  ```python
  # quant.py:117-121 (unpruned), :151-155 (pruned) — 동일 패턴
  backend = "x86"
  model.qconfig = torch.quantization.get_default_qconfig(backend)
  torch.backends.quantized.engine = backend
  quantized_model = torch.quantization.quantize_dynamic(
      model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
  ```
- **방식 = Dynamic PTQ (Post-Training Dynamic Quantization)** 임을 명확히 확인:
  - `quantize_dynamic` 사용 → calibration 데이터셋이 필요 없는 동적 양자화.
  - `qconfig_spec={torch.nn.Linear}` → **오직 `nn.Linear` 레이어만** 양자화 대상. ViT의 거의 모든 연산량(QKV projection, attention output proj, MLP fc1/fc2, head)이 Linear 이므로 사실상 핵심 연산을 모두 커버.
  - `dtype=torch.qint8` → **가중치를 INT8(qint8)** 로 양자화. 활성값(activation)은 추론 시점에 동적으로 관찰되어 양자화됨(dynamic 특성).
- **주의: `model.qconfig`/`get_default_qconfig` 라인은 dynamic 경로에서 사실상 무의미하다.** `quantize_dynamic`은 자체 `qconfig_spec` 으로 동작하므로, `get_default_qconfig("x86")`(static PTQ용 observer 설정)와 `torch.backends.quantized.engine="x86"`(엔진 지정)은 실제 dynamic 양자화 동작에 직접 반영되지 않음. 이는 static PTQ 보일러플레이트가 dynamic 코드에 남은 흔적으로 추정. (정적 분석 기준 근거: `quantize_dynamic`은 별도 calibration/observer 삽입 단계 없이 호출됨.)

#### 3.1.2 backend: x86 vs qnnpack (확인됨)
- `backend = "x86"` 사용 (`quant.py:117,151`). x86은 서버/데스크톱 CPU(구 fbgemm 기반)용 quantized 커널.
- 주석에 `# replaced with ``qnnpack`` causing much worse inference speed for quantized model on this notebook` (`quant.py:117,151`) — 즉 ARM/모바일 백엔드인 **qnnpack** 으로 바꾸면 (이 측정 환경에서는) 더 느렸다는 실험 메모. README 제목도 "x86 backends" 로 명시 (`README:15`).
- **해석**: 이름은 "Mobile" 이지만 **양자화 latency 측정은 x86(데스크톱 CPU) 백엔드에서 수행**. 실제 ARM 모바일 디바이스 측정 코드는 repo에 없음(확인됨). Lite Interpreter(`.ptl`) 변환은 모바일 배포 포맷을 만들기 위함이나, 측정 자체는 호스트 CPU profiler.

#### 3.1.3 per-tensor / per-channel, observer (확인됨/추정)
- `quantize_dynamic`은 기본적으로 **가중치를 per-tensor 또는 per-channel symmetric INT8** 로 양자화하나, 본 코드는 dynamic 경로라 별도 observer/qscheme를 명시하지 않음. PyTorch 기본 dynamic quant은 가중치 per-tensor(레이어별 단일 scale)로 처리(추정 — 코드에 명시적 per_channel 설정 없음; Grep 결과 `per_channel`/`per_tensor`/`HistogramObserver`/`MinMax` 키워드 **부재**).
- static PTQ에서 쓰는 calibration loop / HistogramObserver / MinMaxObserver 는 **존재하지 않음**(확인됨, Grep). 즉 calibration 절차가 없는 가장 단순한 형태의 양자화.

#### 3.1.4 양자화 범위: attention/MLP
- `qconfig_spec={torch.nn.Linear}` 이므로 양자화 대상은 **모든 `nn.Linear`**:
  - Attention: `qkv` (in→3·dim), `proj` (out projection).
  - MLP: `fc1`, `fc2`.
  - Classification `head`.
- `nn.LayerNorm`, GELU, softmax, attention 내적(`q@k.T`, `attn@v`) 등 **non-Linear 연산은 양자화되지 않고 FP로 유지**(확인됨). ViT의 LayerNorm/Softmax 비선형은 FP32 그대로.

#### 3.1.5 모바일 변환 파이프라인 (확인됨)
- 공통 패턴 (unpruned: `quant.py:102-135`, pruned: `quant.py:146-176`):
  ```
  model.eval(); model.cpu()
  scripted = torch.jit.script(model)              # TorchScript
  optimized = optimize_for_mobile(scripted)        # 모바일 최적화 (operator fusion 등)
  optimized.save(".pt")
  optimized._save_for_lite_interpreter(".ptl")     # Lite Interpreter 포맷
  ptl = torch.jit.load(".ptl")
  ```
- 생성 산출물: `vit_unpruned_scripted.pt`, `..._optimized_scripted.pt`, `..._lite.ptl`, 그리고 양자화 버전 `vit_unpruned_optimized_scripted_quantized_lite.ptl`, `vit_pruned_optimized_scripted_quantized_lite.ptl` 등.

#### 3.1.6 Latency 측정 로직 (확인됨)
- **방법 = `torch.autograd.profiler.profile(use_cuda=False)` 의 `self_cpu_time_total`** (단위 µs → /1000 → ms):
  ```python
  # quant.py:109-111 (unpruned 비양자화 lite)
  with torch.autograd.profiler.profile(use_cuda=False) as prof1:
      out = ptl_unquant(img)
  print("Unpruned non-quantized lite model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
  ```
- 3가지 시나리오 측정:
  1. `prof1` : **Unpruned + non-quantized** lite 모델 (`quant.py:109-111`).
  2. `prof2` : **Unpruned + quantized** lite 모델 (`quant.py:132-135`).
  3. `prof3` : **Pruned + quantized** lite 모델 (`quant.py:173-176`).
- 입력은 단일 배치 `torch.randn(1,3,224,224)` (`quant.py:100,171`) — batch=1 단발 추론 latency.
- **한계(확인됨)**: 단일 forward 1회만 측정 → warm-up 없음, 반복 평균/표준편차 없음. `timeit`/다회 반복 미사용. 측정 노이즈 큼(추정).

#### 3.1.7 attention forward 패치 (확인됨)
- `quant.py:29-52` 에 `forward(self, x)` 정의 — timm `Attention.forward` 를 대체하기 위한 함수. `self.fused_attn = hasattr(F, 'scaled_dot_product_attention')` (`quant.py:35`) 로 SDPA 사용 가능 여부 동적 판단. README가 timm 소스에서 `fast_attn`→`fused_attn` 치환을 요구하는 이유와 연결 (`README:10-11`).
- 단, `quant.py:main()` 에서는 이 `forward`가 **실제로 모델에 주입되지는 않음**(prune_timm_vit/finetune와 달리 `m.forward = forward.__get__(...)` 호출이 quant.py main에 없음 — 확인됨). 즉 quant.py의 `forward`는 정의만 되어 있고 unpruned 모델 측정에는 timm 원본 forward가 쓰임(추정). pruned 모델은 `torch.load`로 통째 로드되므로 prune 시점에 이미 패치된 forward를 보유.

#### 3.1.8 정확도 평가 보조 함수
- `prepare_imagenet()` (`quant.py:54-69`) : val 폴더 ImageFolder + `presets.ClassificationPresetEval` (resize 256 / crop 224 / BILINEAR).
- `validate_model()` (`quant.py:71-85`) : top-1 정확도 + cross-entropy loss. `flag=1`이면 CPU에서 평가 (양자화 모델은 CPU 전용이므로). `--test_accuracy` 플래그 필요.
- **버그 주의(확인됨)**: `quant.py:92` 에서 `args.train_batch_size`, `args.use_imagenet_mean_std` 참조하지만 `quant.py:parse_args()`(`:19-27`)에는 해당 인자가 **정의되어 있지 않음** → `--test_accuracy` 사용 시 `AttributeError` 발생 가능.

---

### 3.2 `prune_timm_vit.py` — 구조적 프루닝 (Torch-Pruning 기반)

#### 3.2.1 Importance 기준 선택 (확인됨, `:115-125`)
| `--pruning_type` | Torch-Pruning importance 클래스 |
|---|---|
| `random` | `tp.importance.RandomImportance()` |
| `taylor` (기본) | `tp.importance.GroupTaylorImportance()` |
| `l2` | `tp.importance.GroupNormImportance(p=2)` |
| `l1` | `tp.importance.GroupNormImportance(p=1)` |
| `hessian` | `tp.importance.GroupHessianImportance()` |

#### 3.2.2 모델 로드 및 attention forward 패치 (확인됨, `:133-148`)
- `torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)` — **Facebook DeiT-Base/16** 로드 (`:133`). `--model_name` 인자(`:21`)는 있으나 main에서는 hardcode된 deit_base 사용.
- 모든 `timm...Attention` 모듈의 `forward`를 본 파일의 `forward`(`:38-61`)로 **monkey-patch** (`:145`) — head_dim 변경 후에도 reshape이 일반화되도록 `reshape(B, N, -1)` 사용(`:58`, 원본은 `reshape(B,N,C)`). 이게 head pruning 후 차원 불일치를 막는 핵심 트릭.
- `num_heads[m.qkv] = m.num_heads` (`:146`) 로 qkv Linear → head 수 매핑을 Torch-Pruning에 전달.
- `--bottleneck` 모드면 MLP의 `fc2`를 `ignored_layers`에 추가(`:147-148`) → FFN/Attention의 **내부 레이어만** 프루닝(병목 구조 유지).

#### 3.2.3 Pruner 구성 (확인됨, `:155-167`)
```python
pruner = tp.pruner.MetaPruner(
    model, example_inputs,
    global_pruning = args.global_pruning,   # False면 레이어별 uniform 비율
    importance = imp,
    pruning_ratio = args.pruning_ratio,     # 기본 0.25
    ignored_layers = ignored_layers,        # 최소한 model.head는 항상 보존
    num_heads = num_heads,
    prune_num_heads = args.prune_num_heads,            # head 개수 자체를 줄일지
    prune_head_dims = not args.prune_num_heads,        # head 차원(feature dim)을 줄일지 (기본 True)
    head_pruning_ratio = 0.5,                          # head 50% 제거 (prune_num_heads=True일 때만)
    round_to = 2,                                       # 채널 수를 2의 배수로 라운딩 (HW 친화)
)
```
- 두 가지 head 프루닝 전략 (상호 배타):
  - **head_dim pruning** (기본): 각 head의 feature 차원을 줄임 (`prune_head_dims=True`).
  - **num_heads pruning**: 전체 head를 통째 제거, `head_pruning_ratio=0.5` 하드코딩 (`:165`). 주의: `--head_pruning_ratio` 인자가 있으나 코드에서 0.5로 덮어씀(주석 처리됨, `:165`) → CLI 무시됨(확인됨).
- `round_to=2` (`:166`): 잔존 채널을 2의 배수로 맞춤 → 하드웨어/SIMD 친화 (FPGA/벡터 유닛 관점에서 유의미).

#### 3.2.4 Taylor/Hessian gradient 누적 (확인됨, `:169-187`)
- Taylor/Hessian importance는 **gradient 정보**가 필요 → ImageNet train 배치를 `--taylor_batchs`(기본 10) 만큼 forward/backward 해 grad 누적:
  ```python
  for k,(imgs,lbls) in enumerate(train_loader):
      if k>=args.taylor_batchs: break
      output = model(imgs)
      # Hessian: per-sample loss → l.backward(retain_graph=True) → imp.accumulate_grad(model)
      # Taylor : loss = CE(output,lbls); loss.backward()   (:185-187)
  ```
- Hessian은 per-sample 루프로 accumulate (`:179-184`), Taylor는 배치 평균 loss 1회 backward (`:185-187`).

#### 3.2.5 실제 프루닝 실행 + head 메타데이터 갱신 (확인됨, `:189-202`)
- `for g in pruner.step(interactive=True): g.prune()` (`:189-190`) — interactive 모드로 그룹별 프루닝.
- 프루닝 후 각 Attention 모듈의 `num_heads`/`head_dim`을 **수동 갱신** (`:193-202`):
  ```python
  m.num_heads = pruner.num_heads[m.qkv]
  m.head_dim  = m.qkv.out_features // (3 * m.num_heads)
  ```
  → 프루닝으로 qkv out_features가 줄었으니 head_dim 재계산. 이 갱신이 없으면 forward의 reshape이 깨짐.

#### 3.2.6 요약/저장 (확인됨, `:206-222`)
- `tp.utils.count_ops_and_params` 로 **Base vs Pruned MACs(G) / Params(M)** 출력 (`:215-216`).
- `torch.save(model, args.save_as)` 로 **모델 객체 전체**(state_dict 아님) 저장 (`:222`, 기본 `pruned/model_taylor_0.25.pth`). → 로드 시 torch_pruning이 변경한 구조가 그대로 복원되므로 객체 통째 저장이 필요.

---

### 3.3 `finetune.py` — pruned 모델 ImageNet 미세조정

> torchvision references/classification `train.py` 를 ViT/torch_pruning용으로 개조 (확인됨, `:1` 헤더).

#### 3.3.1 학습 루프 핵심
- `train_one_epoch()` (`:54-96`): AMP(`torch.cuda.amp.autocast`, `:64`), gradient clipping(`--clip-grad-norm`, `:73-82`), EMA 업데이트(`:85-89`), top-1/top-5 metric 로깅.
- `evaluate()` (`:99-141`): `torch.inference_mode()` 하 top-1/top-5. (앞서 언급한 `args` 전역 참조 버그 `:110`).
- `main()` (`:245-458`): 분산 학습(DDP), mixup/cutmix collate(`:270-279`), optimizer(SGD/RMSprop/AdamW, `:332-348`), LR scheduler(StepLR/CosineAnnealingLR/ExponentialLR + warmup, `:352-384`), EMA(`:391-402`), epoch마다 `torch.save(model.module, path)` (`:435-437`).

#### 3.3.2 모델 로드 분기 (확인됨, `:294-303`)
- `--model`이 파일 경로면 → `torch.load(... map_location='cpu')` (pruned 모델 객체 로드, `:296`).
- `--is_huggingface` → `ViTForImageClassification.from_pretrained` (`:299-300`).
- 그 외 → `timm.create_model(args.model, pretrained=True)` (`:303`).
- timm 모델이면 Attention forward 패치 적용(`:305-308`) — prune_timm_vit와 동일 트릭.

#### 3.3.3 정규화/전처리 일관성 (확인됨)
- mean/std: `is_huggingface` 또는 `--use_imagenet_mean_std` 면 ImageNet 통계(0.485.../0.229...), 아니면 **0.5/0.5** (`:180-181,213-214`). DeiT는 0.5/0.5 기본인데 finetune sh에서는 `--use_imagenet_mean_std` 사용(`*.sh:24`) → **prune/quant(0.5 기본) 과 finetune(ImageNet 통계) 사이 정규화 불일치 가능성**(주의, 추정).

---

### 3.4 `presets.py` — 데이터 전처리 preset (Torch-Pruning/torchvision 유래)

- `ClassificationPresetTrain` (`:19-76`): RandomResizedCrop → RandomHorizontalFlip → (선택)AutoAugment/RandAugment/TrivialAugmentWide/AugMix → PILToTensor → ConvertImageDtype → Normalize → (선택)RandomErasing.
- `ClassificationPresetEval` (`:79-115`): Resize(resize_size) → CenterCrop(crop_size) → PILToTensor → ConvertImageDtype → Normalize. ViT 표준 256-resize/224-crop.
- prune/quant은 항상 **Eval preset** 사용(추론 전용 전처리). Train preset은 finetune.py 경로에서만 사용.

---

## 4. 알고리즘 (수식)

### 4.1 Taylor Importance Pruning (Torch-Pruning `GroupTaylorImportance`)
- 직관: 파라미터/채널을 제거했을 때 손실 변화량 ΔL 의 1차 Taylor 근사로 중요도를 추정.
- 1차 Taylor 근사 (single weight w):
  ```
  ΔL ≈ |∂L/∂w · w|
  ```
- 채널/그룹 단위(structured) 에서는 그룹에 속한 파라미터들의 (gradient × weight) 합/제곱합을 importance로 사용 (구현 세부는 torch_pruning 내부). 본 repo는 `--taylor_batchs`(기본 10) 배치로 grad를 누적해 이 통계를 추정 (`prune_timm_vit.py:174-187`). 중요도가 낮은 그룹부터 `pruning_ratio` 만큼 제거.
- L1/L2 importance는 gradient 없이 가중치 노름(‖w‖₁ 또는 ‖w‖₂)만으로 중요도 산정 (`:121-122`).

### 4.2 INT8 양자화 수식 (PyTorch dynamic, qint8)
- 가중치 양자화 (per-tensor symmetric 가정, 추정):
  ```
  scale  = max(|W|) / 127          (qint8 범위 [-128,127], symmetric)
  zero_point = 0                   (symmetric)
  W_int8 = round(W / scale)        clamp to [-128,127]
  W_dequant = scale · W_int8
  ```
- 활성값(dynamic): 추론 시점에 입력 텐서의 min/max를 즉석 관찰해 affine 양자화:
  ```
  scale_x = (x_max - x_min)/255 ,  zp = round(-x_min/scale_x)   (asymmetric uint8 통상)
  ```
  → calibration 불필요. 본 repo는 이 dynamic 경로만 사용(static observer/calibration 부재, §3.1.3).

### 4.3 PTQ Calibration 절차
- **본 repo에는 static PTQ calibration 절차가 없음(확인됨)**. dynamic quantization이라 calibration 데이터 통과 단계가 생략됨. (`get_default_qconfig`/`prepare`/`convert` 의 정식 static 3-step 흐름 부재 — Grep으로 `prepare`/`convert`/observer 미검출.)

---

## 5. 학습/평가 파이프라인

### 5.1 데이터셋
- **ImageNet** (ImageFolder, `train/` + `val/`). 기본 경로 `/p/vast1/MLdata/james-imagenet/` (LLNL 클러스터 경로로 추정, `prune_timm_vit.py:22`, `*.sh:22`).
- 입력 224×224, resize 256, BILINEAR, mean/std 기본 0.5 (DeiT 관례), `--use_imagenet_mean_std` 시 ImageNet 통계.

### 5.2 모델
- **DeiT-Base/16 (deit_base_patch16_224)** — `torch.hub` Facebook DeiT (`prune_timm_vit.py:133`, `quant.py:95`). finetune은 `vit_base_patch16_224`/timm 모델도 지원 (`finetune.py:466,303`).

### 5.3 Finetune 명령 (`finetune_timm_deit_b_16_taylor_uniform.sh`)
```
torchrun --nproc_per_node=4 finetune.py \
  --model "pruned/model_taylor_0.25.pth" \
  --epochs 300 --batch-size 64 --opt adamw \
  --lr 0.000015 --wd 0.3 \
  --lr-scheduler cosineannealinglr --lr-warmup-method linear \
  --amp --label-smoothing 0.11 --mixup-alpha 0.2 --auto-augment ra \
  --clip-grad-norm 1 --ra-sampler --random-erase 0.25 --cutmix-alpha 1.0 \
  --data-path "/p/vast1/MLdata/james-imagenet/" \
  --output-dir finetuned_output/ --use_imagenet_mean_std \
  --path "./finetuned_output/model_finetune_taylor_0.0859375"
```
- **4-GPU DDP**, AdamW lr=1.5e-5, weight decay 0.3, cosine annealing, 300 epoch.
- 강한 augmentation: RandAugment + mixup(0.2) + cutmix(1.0) + random-erase(0.25) + label smoothing(0.11) + Repeated Augmentation sampler. DeiT 학습 레시피 계열(추정).
- `--path` 의 `0.0859375` 는 다른 pruning ratio(추정 — head_dim pruning 기반의 유효 비율). `--model` 의 `0.25` 와 불일치하는 라벨(주의).

### 5.4 Latency Benchmark
- §3.1.6 참조. CPU(x86) Lite Interpreter, batch=1, autograd profiler `self_cpu_time_total`. unpruned-FP / unpruned-INT8 / pruned-INT8 3종 비교.

---

## 6. 의존성 (`requirments.txt` — 오타 그대로)
```
torch==2.0.0
torchvision==0.15.1
timm==0.8.17.dev0
torch-pruning            # 버전 미고정
```
- 추가(코드상 import, requirements 미기재): `tqdm` (`prune_timm_vit.py:11`, `quant.py:11`), `transformers`(huggingface 경로, `finetune.py:299`).
- **timm 패치 필수**: `site-packages/timm/models/vision_transformer.py` 의 line 70/85 에서 `fast_attn` → `fused_attn` 치환 (`README:10-11`). timm 0.8.17.dev0 와 코드의 `fused_attn`/SDPA forward 사이 API 차이를 메우기 위함.
- Python 3.9 에서 테스트 (`README:4`).
- 설치 명령은 `pip install -r requirements.txt` 인데 실제 파일명은 `requirments.txt` (오타) → README와 파일명 불일치(주의, `README:6`).

---

## 7. 강점 / 한계 / 리스크

### 7.1 강점
- **pruning + quantization 통합 + latency 측정**을 end-to-end로 보여주는 간결한 레퍼런스. 압축 단계별(FP→INT8→pruned+INT8) latency 비교가 한 스크립트에 있음.
- Torch-Pruning 의 ViT-aware structured pruning(head/head_dim/FFN)을 그대로 활용 → MACs/Params 실측 감소 확인 가능 (`prune_timm_vit.py:214-216`).
- `round_to=2`, Lite Interpreter 변환 등 **배포 친화적 디테일** 존재.

### 7.2 한계
- **양자화가 dynamic + Linear-only** 로 매우 단순. static PTQ(calibration), QAT, per-channel weight, attention matmul/softmax/LayerNorm 양자화 미지원. 정확도-효율 trade-off 탐색 폭이 좁음.
- **"Mobile" 이지만 측정은 x86 호스트 CPU**. 실제 ARM(qnnpack) 디바이스 on-device 측정 코드 없음. qnnpack은 오히려 느렸다고 메모(§3.1.2).
- **latency 측정 신뢰성 낮음**: batch=1 단발 1회, warm-up/반복 평균 없음 (§3.1.6).
- **코드 견고성 이슈(정적 분석 확인)**: `quant.py` `parse_args`에 `train_batch_size`/`use_imagenet_mean_std` 미정의(`:92`), `finetune.py` `evaluate`의 전역 `args` 참조(`:110`), 누락 모듈 `utils`/`sampler`/`transforms`(§2.4). 그대로 실행 시 일부 경로 에러 가능.
- `get_default_qconfig`/`quantized.engine` 설정이 dynamic 경로에서 무효(§3.1.1) — 의도와 코드 불일치 가능.

### 7.3 리스크
- timm 버전 의존성 강함(0.8.17.dev0 + 수동 소스 패치). timm 업그레이드 시 forward 시그니처 불일치로 깨지기 쉬움.
- pruned 모델을 `torch.save(model)`(객체 전체)로 저장 → 로드 환경의 클래스/버전 의존성, 보안(역직렬화) 리스크.

---

## 8. 우리 프로젝트(PRJXR-HBTXR: ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> 아래는 PRJXR-HBTXR이 "HG-PIPE 계열 Transformer FPGA 가속기 + XR eye-tracking" 임을 전제로 한 해석(추정). PQV-Mobile은 SW(PyTorch) 압축 도구로, FPGA RTL/HLS는 포함하지 않음.

1. **INT8 PTQ 의 FPGA 양자화 추론 참고**: PQV-Mobile의 INT8 dynamic quant은 FPGA 가속기의 fixed-point/INT8 datapath 설계 시 정확도 baseline 확보용 reference로 활용 가능. 단, FPGA는 보통 **static per-tensor/per-channel symmetric INT8(+ LayerNorm/Softmax도 정수 근사)** 가 필요하므로, 본 repo의 Linear-only dynamic 방식은 그대로 쓰기 부족 → static PTQ/per-channel/activation 양자화로 **확장이 필요**(우리 쪽 quantization 스택은 더 공격적이어야 함).
2. **Structured pruning = 가속기 연산량 직접 절감**: head/head_dim/FFN 채널을 구조적으로 줄이면 systolic array/MAC 활용도와 BRAM/대역폭이 그대로 줄어듦. `round_to=2`(HW 친화 라운딩), MACs/Params 실측(`prune_timm_vit.py:214-216`) 은 우리 가속기의 PE 배열 크기·tiling 결정에 직접 입력 가능. **head pruning은 attention 병렬도(헤드 단위 PE) 매핑과 직결**.
3. **Latency 측정 방법론은 반면교사**: PQV-Mobile의 single-shot CPU profiler 측정은 신뢰도가 낮음 → 우리 쪽 벤치마크는 warm-up + 다회 반복 + 분산 보고, 그리고 FPGA는 **cycle-accurate (II/latency from HLS report)** 로 측정해야 함. 다만 "FP vs INT8 vs pruned" 3단 비교 프레이밍 자체는 우리 평가표 구성에 차용할 만함.
4. **DeiT-B/16 224 baseline 공유**: 우리 ViT 가속기 평가도 동일 DeiT-B/16, 224 입력, ImageNet top-1 을 공통 기준으로 쓰면 PQV-Mobile 수치(논문 2408.08437)와 직접 비교 가능.
5. **XR eye-tracking 연결(추정)**: eye-tracking용 경량 ViT를 엣지(FPGA)에서 저지연 추론하려면 pruning(연산량↓)+quantization(비트폭↓)이 핵심. PQV-Mobile은 **압축 파이프라인의 SW 레퍼런스**로서, 우리는 그 위에 (a) FPGA-friendly static INT8/INT4, (b) 비선형(softmax/LN) 정수화, (c) HLS 매핑까지 더해야 함.

---

## 9. 근거 표기 / 확인 불가 항목

### 9.1 코드로 직접 확인됨
- dynamic INT8 quantization, Linear-only, x86 backend (`quant.py:117-121,151-155`).
- latency = autograd profiler `self_cpu_time_total`, batch=1 (`quant.py:109-176`).
- Torch-Pruning MetaPruner, Taylor/Hessian/L1/L2/Random importance, head/head_dim/FFN pruning, round_to=2 (`prune_timm_vit.py:115-167`).
- DeiT-Base/16 torch.hub 로드 (`prune_timm_vit.py:133`, `quant.py:95`).
- 자체 .py 4개뿐; `utils`/`sampler`/`transforms` 누락 (Glob + `finetune.py:14-20`).
- timm fast_attn→fused_attn 패치 필요 (`README:10-11`).
- 코드 버그성 항목(quant.py:92 미정의 인자, finetune.py:110 전역 args) — 정적 분석 기준.

### 9.2 추정 (코드만으로 단정 불가)
- per-tensor weight 양자화 여부 (dynamic 기본값 가정; per_channel 명시 없음).
- quant.py의 `get_default_qconfig`/`quantized.engine` 이 dynamic 경로에서 무효라는 판단(라이브러리 동작 추정).
- finetune의 mean/std 불일치로 인한 실제 정확도 영향(추정).
- `--path` 의 0.0859375 라벨 의미(head_dim pruning 유효 비율 추정).
- PRJXR-HBTXR 의 구체 목표(HG-PIPE 계열 FPGA + XR eye-tracking) — 외부 컨텍스트 기반 추정.

### 9.3 확인 불가
- 논문(2408.08437) 본문 수치(정확도/속도/압축률) — 본 repo 코드에 미포함, PDF 미열람.
- 실제 ARM 모바일 디바이스에서의 latency — 측정 코드 자체가 repo에 없음.
- timm 0.8.17.dev0 의 정확한 vision_transformer.py line 70/85 내용 — 외부 패키지로 repo 외부.
