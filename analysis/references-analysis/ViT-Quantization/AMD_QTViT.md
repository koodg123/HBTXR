# AMD_QTViT (QT-ViT) 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/AMD_QTViT`
> 원논문: **QT-ViT: Improving Linear Attention in ViT with Quadratic Taylor Expansion** (Xu et al., NeurIPS 2024, AMD)
> 분석 방식: 실제 소스(Glob/Grep/Read) 기반. 라인 근거 표기. 외부/`.git`/`__pycache__`/대용량 로그 제외.

---

## 0. 핵심 결론 (먼저 읽기)

- **이 repo의 "QT"는 정수 양자화(Quantization)가 아니라 "Quadratic Taylor"(2차 테일러 전개)다.** 부모 디렉토리명이 `ViT-Quantization`이라 양자화로 오해하기 쉬우나, README.md:1, 62~66의 논문 제목/BibTeX이 "Quadratic Taylor Expansion"임을 명시한다.
- **정수/INT8/fake_quant/QAT/PTQ 양자화 코드는 존재하지 않는다.** Grep `quant|Quant|int8|fake_quant|fp8|INT8|qconfig|observer`로 전 repo를 검색했을 때 자체 소스에서의 유일한 매치는 `efficientvit/models/efficientvit/sam.py:81`의 주석 `"...numpy array ... in uint8 format"`(이미지 전처리 dtype 설명)뿐이다. → **양자화 코드 유무: 없음 (확인 완료)**.
- 학습은 fp16 AMP를 쓰되, 핵심 선형 어텐션 연산(`relu_linear_att`)은 **수치 안정성을 위해 `@autocast(enabled=False)`로 fp32 강제**한다 (`ops.py:401`, `ops.py:405-406`).
- 코드 베이스는 MIT의 **EfficientViT**(Han Cai 등) 구조를 AMD가 fork/수정한 것이다. 헤더 주석 `# Modifications Copyright © 2025 Advanced Micro Devices, Inc.` (예: `ops.py:1`, `backbone.py:1`)가 모든 자체 파일에 붙어 있다.
- QT-ViT의 본질적 기여는 **LiteMLA의 선형 어텐션 커널을 ReLU 기반에서 "L2 정규화 + 제곱(2차 Taylor 근사) + 상수항 augmentation + DWConv 보정"으로 교체**한 것이다 (`ops.py:402-462`).

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

### 목적
ViT의 표준 softmax self-attention은 시퀀스 길이 N에 대해 O(N²)이다. 선형 어텐션(linear attention)은 O(N)으로 줄이지만, softmax의 표현력을 단순 kernel feature map(예: ReLU)으로 근사하면 정확도가 떨어진다. QT-ViT는 **softmax exp(qᵀk)를 2차 테일러 전개(Quadratic Taylor)로 근사**하여 선형 복잡도를 유지하면서 표현력 손실을 줄인다.

### 핵심 아이디어 (코드로 확인된 것)
1. **2차 Taylor 근사**: q, k를 L2 정규화한 뒤 제곱(`q**2`, `k**2`)하고 다시 정규화 (`ops.py:432-437`). exp의 1차+2차 항을 모사하기 위한 feature map.
2. **상수항(ones) augmentation**: q, k에 학습 가능한 스케일 `ones_scale1`을 곱한 ones 채널을 concat (`ops.py:439-442`). Taylor 전개의 0차/저차 상수항 역할.
3. **선형 결합 순서**: `kv = kᵀ·v` 먼저 계산 후 `q·kv` → O(N·d²) (`ops.py:445-449`).
4. **정규화 트릭**: v에 1로 패딩된 추가 채널을 두어 분모(정규화 상수)를 동시에 계산, 마지막에 나눔 (`ops.py:447, 450`).
5. **DWConv 로컬 보정**: linear attention 출력에 v를 2D feature map으로 reshape→BN→GELU한 항을 더해 지역성 보강 (`ops.py:452-457`).
6. **절대 위치 임베딩**: 학습 가능한 positional encoding을 k에 더함, 해상도 변하면 bicubic interpolate (`ops.py:399, 425-430`).

### ImageNet 결과 (README.md:36-43)
| 모델 | Top1 | Top5 | Params | MACs |
|---|---|---|---|---|
| QT-ViT-1 | 79.6 | 94.7 | 9.4M | 0.52G |
| QT-ViT-3 | 83.9 | 96.7 | 49.7M | 3.97G |
| QT-ViT-6 | 86.0 | 97.3 | 246.8M | 27.60G |

---

## 2. 디렉토리 구조 (자체 + 제외)

```
AMD_QTViT/
├── README.md                         # 논문/설치/학습 명령 (QT-ViT 명시)
├── LICENSE
├── train_cls_model.py                # 분류 학습 진입점 (torchpack dist-run)
├── eval_cls_model.py                 # 단독 평가 스크립트 (DataParallel)
├── configs/cls/imagenet/             # b1~b3, l1~l3, default.yaml
└── efficientvit/                     # 자체 핵심 패키지
    ├── cls_model_zoo.py              # create_cls_model + checkpoint 레지스트리
    ├── seg_model_zoo.py / sam_model_zoo.py
    ├── models/
    │   ├── nn/
    │   │   ├── ops.py    ★★★         # LiteMLA(QT 선형 어텐션), Conv/MBConv/DSConv 등
    │   │   ├── norm.py               # LayerNorm2d, build_norm, reset_bn
    │   │   ├── act.py                # build_act (relu/relu6/hswish/silu/gelu-tanh)
    │   │   └── drop.py
    │   ├── efficientvit/
    │   │   ├── backbone.py ★         # EfficientViTBackbone/LargeBackbone + b0~b3,l0~l3
    │   │   ├── cls.py    ★           # EfficientViTCls, ClsHead, cls_b0~l3
    │   │   ├── seg.py / sam.py       # (분류 외 태스크, 본 분석 범위 밖)
    │   └── utils/                    # network.py, list.py, random.py
    ├── apps/                         # trainer/base, data_provider, utils(lr,ema,dist,opt)
    └── clscore/                      # cls 전용 trainer, data_provider/imagenet.py
```

### 제외 항목 (절대 제약 준수)
- `.git/**`, `__pycache__/**` — 분석 제외.
- `efficientvit/models/efficientvit/seg.py`, `sam.py` 및 `seg_model_zoo.py`, `sam_model_zoo.py` — 분류 태스크 범위 밖이라 정밀 분석 생략(존재만 기록).
- 외부 패키지 의존(torchpack, timm 등)은 vendoring되지 않음 → 외부.

---

## 3. 핵심 모듈·파일별 정밀 분석 ★ (가장 중요)

### 3.1 `efficientvit/models/nn/ops.py` — 모델 연산 핵심

#### (a) `LiteMLA` (ops.py:333-484) — **QT 선형 어텐션의 핵심**
EfficientViT 원본 "Lightweight Multi-Scale Linear Attention"을 QT-ViT 방식으로 개조한 클래스.

생성자 (`ops.py:336-399`):
- `self.qkv`: 1×1 ConvLayer로 `in_channels → 3*total_dim` 한 번에 Q,K,V 생성 (`ops.py:362-369`). `total_dim = heads*dim` (`ops.py:355`).
- `self.aggreg`: 각 scale마다 (depthwise sxs conv) + (groupwise 1×1 conv) 쌍을 ModuleList로 구성 → **multi-scale token aggregation** (`ops.py:370-385`). 기본 `scales=(5,)`.
- `self.proj`: `total_dim*(1+len(scales)) → out_channels` 1×1 Conv (`ops.py:387-394`).
- **QT 전용 추가 파라미터**: `self.bn = BatchNorm2d(dim)`, `self.act = GELU()` (DWConv 보정용), `self.ones_scale1 = Parameter(1.0)` (상수항 스케일), `self.positional_encoding`(학습형 절대위치, shape `[1, heads*dim*2, 224//downsample, 224//downsample]`) (`ops.py:396-399`).

`relu_linear_att(qkv)` (`ops.py:401-462`) — 본 repo의 알고리즘적 심장:
- `@autocast(enabled=False)` 데코레이터 + fp16이면 `.float()` → **fp32 강제** (`ops.py:401, 405-406`). 선형 어텐션의 분모/제곱 연산이 fp16에서 inf/nan 나는 것을 방지.
- qkv를 `(B, heads, 3*dim, H*W)`로 reshape 후 transpose, q/k/v 분리 (`ops.py:408-422`).
- 절대 위치 임베딩을 k에 더함; 해상도 불일치 시 `F.interpolate(..., mode='bicubic')` (`ops.py:425-430`).
- **2차 Taylor feature map** (`ops.py:432-437`):
  ```
  q = q / (||q|| + eps);  k = k / (||k|| + eps)   # 1차 L2 정규화
  q = q**2;               k = k**2                 # 제곱 (2차항)
  q = q / (||q|| + eps);  k = k / (||k|| + eps)   # 재정규화
  ```
  주석(`ops.py:432-433`)에 "학습 중 inf/nan 방지용, 추론 시 출력 변화 없이 삭제 가능"이라 명시.
- **상수항 concat** (`ops.py:439-442`): `ones * ones_scale1`을 q,k 마지막 차원에 붙여 Taylor 저차항 모사.
- **선형 matmul** (`ops.py:445-450`): `trans_k = kᵀ`; v에 1 패딩(`F.pad(v,(0,1),value=1)`)으로 정규화 분모 채널 추가; `kv = kᵀ·v`; `out = q·kv`; `out = out[...,:-1] / (out[...,-1:]+eps)` → 정규화. **O(N·d²) 선형 복잡도**.
- **DWConv 로컬 보정** (`ops.py:452-457`): v를 `(b·e, c, w, h)`로 rearrange → `act(bn(...))` → 다시 토큰 형태 → out에 가산.
- 출력 reshape `(B, -1, H, W)` (`ops.py:460-461`).

`forward` (`ops.py:464-475`): qkv 생성 → multi-scale aggreg concat → `relu_linear_att` → `proj`.

> **FPGA 관점 메모**: softmax가 전혀 없고(`exp` 없음), 핵심 연산이 1×1 conv + 행렬곱(kᵀv, q·kv) + L2 정규화 + 제곱 + DWConv로만 구성. 단 **L2 정규화의 1/sqrt(reduce)**와 **나눗셈(정규화 분모)**, fp32 강제는 하드웨어 매핑 시 검토 필요.

#### (b) `EfficientViTBlock` (ops.py:487-523)
- `context_module`: `ResidualBlock(LiteMLA(...), IdentityLayer())` — 선형 어텐션 + skip (`ops.py:499-509`).
- `local_module`: `ResidualBlock(MBConv(expand_ratio, use_bias=(T,T,F), act=(act,act,None)), Identity)` — 지역 정보 (`ops.py:510-518`).
- forward: context → local (`ops.py:520-523`). 즉 **[선형 어텐션] → [MBConv] 순서의 하이브리드 블록**.

#### (c) 기본 빌딩 블록
- `ConvLayer` (ops.py:38-79): Conv2d + (선택)Norm + (선택)Act, padding은 `get_same_padding`.
- `LinearLayer` (ops.py:101-132): Linear + Norm + Act, 2D 초과 입력 flatten.
- `DSConv` (ops.py:145-184): depthwise + pointwise (MobileNet 식).
- `MBConv` (ops.py:187-239): inverted(1×1↑) → depthwise(k×k) → point(1×1↓). expand_ratio=6 기본.
- `FusedMBConv` (ops.py:242-285): spatial conv + point conv (큰 stage용).
- `ResBlock` (ops.py:288-330): conv1+conv2.
- `ResidualBlock` (ops.py:531-561): main + shortcut(+post_act), shortcut None이면 단순 forward.
- `DAGBlock` (ops.py:563-597), `OpSequential` (ops.py:600-612).

### 3.2 `efficientvit/models/efficientvit/backbone.py`
- `EfficientViTBackbone` (backbone.py:34-160): input_stem(stride2 Conv + DSConv residual들) → stage1~2(MBConv/DSConv local) → stage3~4(첫 block은 stride2 down + 이후 `EfficientViTBlock` 반복). `downsample` 값을 EfficientViTBlock에 넘겨 위치임베딩 해상도 결정 (backbone.py:107-117).
- `build_local_block` (backbone.py:123-152): expand_ratio==1이면 DSConv, 아니면 MBConv. `fewer_norm` 옵션으로 norm 일부 제거.
- 변형 b0~b3 (backbone.py:163-200): width/depth/dim 차이. 예 b1=width[16,32,64,128,256], depth[1,2,3,3,4], dim16.
- `EfficientViTLargeBackbone` (backbone.py:203-341): l0~l3. stage별 ResBlock/FusedMBConv/MBConv 선택, act 기본 `gelu`, downsample 16부터 시작 (backbone.py:262).

### 3.3 `efficientvit/models/efficientvit/cls.py`
- `ClsHead` (cls.py:25-48): `ConvLayer(1×1) → AdaptiveAvgPool2d(1) → LinearLayer(ln+act) → LinearLayer(n_classes)`. `fid="stage_final"`로 backbone 출력 dict에서 feature 선택.
- `EfficientViTCls` (cls.py:51-60): backbone(dict 출력) → head.
- `efficientvit_cls_b0~b3, l1~l3` (cls.py:63-161): backbone + ClsHead(width_list, act) 조합. l 계열은 act_func="gelu".

### 3.4 `efficientvit/models/nn/norm.py`
- `LayerNorm2d` (norm.py:13-19): 채널축(dim=1) 평균/분산으로 정규화하는 2D LN.
- `build_norm` (norm.py:30-40): bn2d/ln/ln2d 레지스트리.
- `reset_bn` (norm.py:43-130): calibration 데이터로 BN running stat 재추정(분산 학습, distributed sync 지원). **추론 최적화/재현용**.
- `set_norm_eps` (norm.py:133-137): l 계열에서 eps=1e-7 설정(cls_model_zoo.py:68-69).

### 3.5 `efficientvit/models/nn/act.py`
- `build_act` (act.py:23-29): `relu/relu6/hswish/silu/gelu(tanh approx)` 레지스트리 (act.py:14-20). **gelu는 tanh 근사** → 하드웨어 LUT 친화적.

### 3.6 `eval_cls_model.py`
- 단독 평가 스크립트. torchvision `ImageFolder` + Resize(bicubic, crop_ratio 0.95)/CenterCrop/Normalize (eval:51-70).
- `create_cls_model(model, False)` 후 `checkpoint['state_dict']` 로드, `DataParallel`, `inference_mode` top1/top5 (eval:72-102).

### 3.7 `train_cls_model.py` (진입점, 라인 정밀분석 생략 — torchpack 의존)
- README.md:49-55 기준 `torchpack dist-run -np 8 python train_cls_model.py configs/cls/imagenet/b1.yaml --data_provider.image_size "[128,160,192,224]" --run_config.eval_image_size "[224]"`. **multi-resolution 학습**(여러 해상도 랜덤).

---

## 4. 알고리즘 / 수식

### 4.1 표준 softmax attention (근사 대상)
$$\text{Attn}(Q,K,V)_i = \frac{\sum_j \exp(q_i^\top k_j)\, v_j}{\sum_j \exp(q_i^\top k_j)}$$

### 4.2 QT-ViT의 2차 Taylor 선형 근사 (코드 `ops.py:432-450` 기준)
exp을 2차 테일러로 근사: $\exp(x) \approx 1 + x + \tfrac{1}{2}x^2$. 이를 feature map $\phi(\cdot)$으로 분리.

코드의 실제 feature map:
$$\hat q = \frac{q}{\|q\|},\quad \tilde q = \frac{\hat q^{2}}{\|\hat q^{2}\|},\quad \phi(q) = [\,\tilde q \,;\, c\,] \quad (c = \texttt{ones\_scale1})$$
(k도 동일, 단 k는 위치임베딩 가산 후). 그 다음 선형 결합:
$$\text{out}_i = \frac{\phi(q_i)^\top \big(\sum_j \phi(k_j) \,[v_j;1]\big)}{\big(\phi(q_i)^\top \sum_j \phi(k_j)\big)}$$
- 분자/분모를 v에 1을 패딩한 augmented matmul로 동시 계산 (`ops.py:447-450`).
- 복잡도: $\sum_j \phi(k_j)[v_j;1]$를 먼저 계산(=`kᵀv`)하므로 **O(N·d²)** (`ops.py:448-449`). softmax의 O(N²)와 대비.
- 추가 로컬항: $\text{out} \mathrel{+}= \text{GELU}(\text{BN}(\text{reshape}(v)))$ (`ops.py:452-457`).

### 4.3 위치 임베딩 보간
해상도 변경 시 학습된 PE를 `bicubic`으로 interpolate 후 k에 가산 (`ops.py:425-430`).

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet-1K. 디렉토리 `imagenet/{train,val}` (README.md:24-29). 평가 전처리: bicubic Resize → CenterCrop → Normalize(mean .485/.456/.406) (eval_cls_model.py:54-62).
- **학습**: `torchpack dist-run -np 8 python train_cls_model.py configs/cls/imagenet/b1.yaml ...` (README.md:49-55). multi-resolution(`image_size "[128,160,192,224]"`). config는 매우 얇음(b1.yaml: name/dropout/test_crop_ratio만, 1-6라인) — 대부분 하이퍼파라미터는 `efficientvit/apps`·`clscore` 코드 기본값/CLI override.
- **평가**: `python eval_cls_model.py --model b1 --weight_url ... --image_size 224` (eval_cls_model.py:30-42).
- **배포/양자화**: **TensorRT/CoreML/ONNX 배포 스크립트 없음**(Next-ViT와 대비되는 차이). `apps/utils/export.py`가 있으나 ONNX export 보조 수준(범위 밖). **PTQ/QAT 양자화 파이프라인 없음(확인 완료)**.

---

## 6. 의존성 (README.md:9-15)
- python 3.10, torch, einops, opencv-python, **timm==0.6.13**, tqdm, **torchprofile**, matplotlib, transformers, onnx/onnxsim/onnxruntime, pycocotools, mpi4py/openmpi.
- **torchpack** (zhijian-liu fork, 특정 커밋) — 분산 학습 런처. → 학습 재현에 외부 의존 강함.

---

## 7. 강점 / 한계 / 리스크

### 강점
- **softmax 제거 + 선형 복잡도**: 긴 토큰열/고해상도에서 연산·메모리 이득.
- 2차 Taylor로 단순 ReLU 선형 어텐션 대비 정확도 회복(README 결과 79.6~86.0).
- conv 중심(1×1 conv로 QKV, DWConv 보정) → CNN 가속기 인프라 재활용 가능.
- gelu가 tanh 근사라 LUT 구현 용이.

### 한계 / 리스크
- **양자화 코드 부재**: 이름과 달리 INT8/QAT/PTQ 없음. FPGA용 정수화는 별도 직접 구현 필요.
- **fp32 강제 구간**(`ops.py:401,405`): L2 정규화/제곱/나눗셈이 수치 민감 → 저정밀(INT8/FP16) 매핑 시 정확도/오버플로우 리스크. 양자화 시 가장 까다로운 부분.
- **나눗셈(정규화 분모)**과 **1/sqrt(L2 norm)**: FPGA에서 reciprocal/rsqrt LUT·Newton 반복 필요.
- 학습형 절대 위치 임베딩 + bicubic interpolate: 가변 해상도 추론 시 interpolation 비용/구현 부담.
- torchpack 등 외부 런처 의존으로 학습 재현 진입장벽.
- config가 얇아 실제 하이퍼파라미터가 코드에 흩어져 있음(가독성/이식성 저하).

---

## 8. 우리 프로젝트(HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적) 관점 시사점

> 아래는 부모 경로(`PRJXR-HBTXR`, HGTXR/HGPIPE)로부터의 **추정**. 본 repo 자체에는 FPGA/시선추적 언급 없음(확인 완료).

- **선형 어텐션 → softmax-free 데이터패스에 유리(추정)**: HG-PIPE류 파이프라인 ViT 가속기는 softmax의 전역 reduction/exp가 파이프라인 stall·LUT 비용의 주범인데, QT-ViT는 exp가 전혀 없고 `kᵀv → q·kv` 행렬곱 2개로 환원된다. 시선추적처럼 **저지연·스트리밍**이 중요한 XR에서 선형 어텐션의 O(N) 특성은 프레임당 latency 안정화에 유리.
- **conv-heavy 구조의 매핑 용이성**: QKV가 1×1 conv, multi-scale aggreg가 DW/PW conv, 보정항도 DWConv. 기존 CNN systolic array/conv 엔진을 재사용해 어텐션을 conv로 흡수 가능(추정). EfficientViTBlock = [선형어텐션]→[MBConv]는 conv 파이프라인과 자연스럽게 직렬화.
- **양자화는 우리가 직접 해야 함**: 이 repo에는 정수화가 없으므로, HG-PIPE 통합 시 (1) `relu_linear_att`의 fp32 구간(L2 norm, 제곱, 나눗셈)을 fixed-point/INT로 재설계, (2) `kᵀv` 누산기 비트폭 결정, (3) 정규화 분모 나눗셈을 reciprocal LUT로 치환하는 작업이 핵심 리스크 포인트.
- **HW 비친화 요소 주의**: 1/sqrt·나눗셈·bicubic interpolate. 시선추적이 고정 해상도(예: 눈 ROI 패치)라면 위치임베딩 interpolation을 컴파일타임에 제거해 데이터패스 단순화 가능(추정).
- **참고용 우선순위**: QT-ViT는 "선형 어텐션 알고리즘" 레퍼런스로서 가치가 크고, "양자화 레퍼런스"로는 부적합(코드 없음). 양자화는 동일 부모 폴더의 다른 repo(Next-ViT의 TensorRT int8 경로 등)나 별도 양자화 toolkit 참조 권장.

---

## 9. 근거 표기 (추정 / 확인 불가 구분)

- **확인 완료**:
  - 양자화 코드 부재: Grep 전수 검색 결과 자체 소스 매치는 sam.py:81 주석(uint8 전처리)뿐.
  - QT = Quadratic Taylor: README.md:1, 62-66.
  - 선형 어텐션 fp32 강제: ops.py:401, 405-406.
  - 2차 Taylor feature map / 상수항 / DWConv 보정: ops.py:432-457.
  - TensorRT/CoreML 배포 스크립트 부재: 디렉토리 Glob 결과(deployment 폴더 없음).
- **추정**: 8장 FPGA/시선추적 관련 시사점 전부(부모 경로명 기반 추론, 본 repo 무관).
- **확인 불가**: 실제 학습 하이퍼파라미터 전체값(config 얇음 + torchpack/apps 코드 기본값 의존), 사전학습 가중치 동작(외부 Google Drive 링크).
