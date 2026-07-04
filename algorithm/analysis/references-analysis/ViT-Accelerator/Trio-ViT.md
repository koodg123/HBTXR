# Trio-ViT 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Accelerator\Trio-ViT`
> 분석 범위: EfficientViT 모델 정의(`EfficientViT/`) + PTQ 양자화 엔진(`quant/`) + 비교용 CNN(`models/`) + 평가 엔트리(`main_imagenet*.py`). 체크포인트(`*.pt`/`*.pth`)와 `.git`은 이름만 언급하고 내부 미분석.
> 작성일: 2026-06-20
> **핵심 결론(선요약): 이 repo에는 FPGA RTL/HLS 소스가 전혀 없다(.v/.sv/.cpp/.hpp/.cl/.tcl = 0개, Glob 확인). 순수 PyTorch 알고리즘(소프트맥스-프리 선형 어텐션 EfficientViT + PTQ) repo이며, 논문이 말하는 "accelerator"는 본 repo에 미포함이다.**

---

## 1. 개요

- **목적**: Softmax를 제거하고 ReLU 기반 선형 어텐션을 쓰는 **효율적 ViT(EfficientViT)**를 대상으로, 양자화에 민감한 비선형 연산(특히 Softmax)을 회피한 상태에서 **사후 학습 양자화(Post-Training Quantization, PTQ)**를 적용해 정수 추론을 가능하게 하는 것. 논문 본문은 추가로 HW 가속기를 제안하지만, 이 코드 저장소는 **알고리즘 측(양자화 엔진)** 구현만 담고 있다.
- **한 줄 요약**: EfficientViT(Conv-Transformer 하이브리드, 선형 어텐션)에 **BRECQ/AdaRound 계열의 블록 단위 reconstruction PTQ**를 적용하는 PyTorch 코드베이스. Softmax-free `LiteMSA` 선형 어텐션, BN folding, 채널별 outlier 대응(Scaled/Shifted 양자화 모듈), 어텐션 내부 `K^T·V`/`Q·(KV)`/분모(log 양자화)까지 정수화하는 것이 특징.
- **원논문(추정/확인)**: README.md:3 가 직접 명시 — "Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer" (arXiv:2405.03882). 즉 사용자 제공 추정이 README로 **확인됨**.
- **타깃 성격(명확화)**:
  - **이 repo = 알고리즘(PTQ + 모델 정의) 연구 코드.** HW 가속 소스(RTL/HLS) **미포함**.
  - 근거: 본 디렉토리 전체에 대해 `**\*.v`, `**\*.sv`, `**\*.cpp`, `**\*.hpp`, `**\*.cl`, `**\*.tcl` Glob 결과 모두 **0개**. 존재하는 비표준 파일은 `.py`, `.sh`, `.md`, `LICENSE`, `.gitignore`, `.git/*` 뿐.
  - 논문 제목의 "Acceleration"/"accelerator"는 abstract(README.md:12 "at the hardware level, we build an accelerator")에 언급되나, **해당 RTL/HLS는 이 저장소에 공개되어 있지 않다**(또는 별도 미포함). → 본 분석은 알고리즘 측에 한정.

---

## 2. 디렉토리 구조

### 2.1 자체 소스 트리(분석 대상)

```
Trio-ViT/
├─ README.md                       # 논문 제목/abstract/실행 예시(arXiv:2405.03882)
├─ quant.sh                        # 실행 커맨드 모음(주석 처리된 다양한 설정)
├─ main_imagenet.py                # ★ 단일 GPU 평가/양자화 엔트리(전체 파이프라인)
├─ main_imagenet_dist.py           # 분산(멀티 GPU) 버전 엔트리
├─ hubconf.py                      # torch.hub 진입점(비교용 CNN 로딩)
├─ data/
│   └─ imagenet.py                 # ImageNet DataLoader(torchvision)
├─ EfficientViT/                   # ★ EfficientViT 모델 정의(소프트맥스-프리)
│   └─ models/
│       ├─ cls_model_zoo.py        # b1/b2/b3 cls 모델 팩토리 + 체크포인트 경로
│       ├─ seg_model_zoo.py        # seg 모델 zoo(분류 파이프라인에선 미사용)
│       ├─ efficientvit/
│       │   ├─ backbone.py         # ★ EfficientViTBackbone(input stem + 5 stage)
│       │   ├─ cls.py              # ★ EfficientViTCls + ClsHead
│       │   └─ seg.py              # 세그멘테이션 헤드(분류엔 미사용)
│       ├─ nn/
│       │   ├─ ops.py              # ★★ 핵심 연산: ConvLayer/MBConv/DSConv/LiteMSA/EfficientViTBlock
│       │   ├─ act.py              # 활성함수 레지스트리(relu/relu6/hswish)
│       │   └─ norm.py             # 정규화 레지스트리(bn2d/ln)
│       └─ utils/                  # metric/network/list 유틸
├─ quant/                          # ★★ PTQ 양자화 엔진(BRECQ/AdaRound 계열)
│   ├─ quant_layer.py              # ★★ 양자화 quantizer + QuantModule(평/Shifted/Scaled)
│   ├─ adaptive_rounding.py        # ★ AdaRound(learned rounding) 구현
│   ├─ quant_block.py              # ★ 블록별 양자화 래퍼(CNN 블록 + EfficientViT MB/LiteMSA)
│   ├─ quant_model.py              # ★ 모델 전체를 재귀적으로 QuantModule로 치환
│   ├─ block_recon.py             # ★ 블록 단위 reconstruction(BRECQ)
│   ├─ layer_recon.py             # 레이어 단위 reconstruction(첫/끝 레이어용)
│   ├─ fold_bn.py                 # ★ Conv+BN folding
│   └─ data_utils.py              # 캘리브레이션 입출력/그래디언트 캐싱
├─ models/                         # ★ 비교용 CNN(BRECQ 원본에서 계승)
│   ├─ mobilenetv2.py / resnet.py / mnasnet.py / regnet.py / utils.py
└─ linklink/                       # 분산 학습 헬퍼(allreduce/allaverage 등 stub)
```

- 참고: `ops.py:10`은 `from EfficientViT.plot import plot_distribution`을 import하나, `EfficientViT/plot.py`는 Glob 결과에 **존재하지 않음**(분포 시각화는 코드 곳곳에서 주석 처리됨). 즉 실제 실행 시 이 import는 누락될 수 있으나, plot 호출부가 모두 주석이라 정상 경로에서는 사용되지 않는다(확인: ops.py:238-239, quant_block.py:381-383 등 모두 `#` 처리).

### 2.2 HW(가속) 소스 유무 — 명시

| 항목 | 결과 |
|---|---|
| Verilog `.v` | 0개 |
| SystemVerilog `.sv` | 0개 |
| C/C++ HLS `.cpp/.hpp` | 0개 |
| OpenCL `.cl` | 0개 |
| Tcl 스크립트 `.tcl` | 0개 |
| 결론 | **HW 가속 코드 미포함. 순수 Python(PyTorch) 알고리즘 repo.** |

### 2.3 제외물(이름만 언급)

- `EfficientViT/checkpoints/cls/{b1,b2,b3}-r{224,256,288}.pt` — 사전학습 가중치(경로는 `cls_model_zoo.py:11-21`에 등록). 파일 실체는 분석 제외.
- `.git/*` — 버전관리 메타데이터. 분석 제외.

---

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 EfficientViT 모델: Softmax-free 선형 어텐션과 Conv-Transformer 하이브리드

#### 3.1.1 `LiteMSA` — Lightweight Multi-Scale Attention (핵심, `ops.py:252-354`)

이 클래스가 Trio-ViT가 "softmax-free"라고 부르는 선형 어텐션의 본체다.

- **QKV 생성(ops.py:277-284)**: `qkv = ConvLayer(in_ch, 3*total_dim, 1)` — 1×1 Conv 한 번으로 Q/K/V를 동시에 만든다(`total_dim = heads*dim`, ops.py:270). 어텐션 입력이 토큰 시퀀스가 아니라 **공간 feature map(B,C,H,W)** 형태로 유지된다는 점이 일반 ViT와 다르다.
- **멀티스케일 집계(ops.py:285-295)**: `aggreg`은 scale=5의 **depthwise 5×5 Conv + 1×1 Conv**로 QKV의 다중 스케일 토큰을 추가 생성(`groups=3*total_dim`인 depthwise, 이어 `groups=3*heads`인 그룹 conv). `scales=(5,)`(ops.py:265)이므로 원본 + 1개 스케일 = 총 2개 분기.
- **헤드 분해(ops.py:319-334)**: `(B,-1,3*dim,H*W)`로 reshape 후 transpose하여 마지막 축을 토큰(H·W)으로 두고, 채널을 Q/K/V로 슬라이스(`[..0:dim], [dim:2dim], [2dim:]`).
- **★ Softmax 대체 = ReLU 커널 + 선형 어텐션(ops.py:336-347)**:
  ```
  q = relu(q); k = relu(k)                # kernel_func = build_act("relu")  (ops.py:296)
  trans_k = k.transpose(-1,-2)
  v = pad(v, (0,1), value=1)              # V에 1열 추가 → 분모(정규화 합) 트릭
  kv = trans_k @ v                        # (dim) x (dim+1)  : K^T·V 먼저 계산(선형 복잡도)
  out = q @ kv                            # Q·(K^T·V)
  out = out[..,:-1] / (out[..,-1:] + 1e-15)   # 마지막 열이 ∑(k) → 정규화 분모
  ```
  - **의의**: 표준 어텐션의 `softmax(QK^T)V`(O(N^2)) 대신 `(Q)·((K)^T·V)` 결합법칙으로 **O(N) 선형 복잡도**를 달성하고, exp/softmax 없이 **ReLU만으로 양의 커널**을 만든다(ops.py:337-338). 분모는 V에 1을 패딩(ops.py:342)해 `K^T·V`의 마지막 열로 `∑k`를 흡수, 별도 softmax 정규화 없이 나눗셈 1회로 처리한다(ops.py:347).
  - `MatMul` 래퍼(ops.py:244-249)를 `kv_matmul`/`qkv_matmul`로 명시 분리(ops.py:306-307, 345-346) → 양자화 단계에서 두 matmul을 개별 모듈로 가로채기 위함(매우 의도적).
- **최종 투영(ops.py:349-352)**: reshape 복원 후 `proj`(1×1 Conv, BN)로 출력.

#### 3.1.2 `MBConv` / `DSConv` — Conv 측 빌딩블록(`ops.py:138-241`)

- **MBConv(ops.py:180-241)**: MobileNetV2식 inverted bottleneck = `inverted_conv(1×1 확장) → depth_conv(k×k DW) → point_conv(1×1 축소)`. 기본 `expand_ratio=6`, 활성 `(relu6, relu6, None)`. EfficientViTBlock 안에서 쓰일 땐 `act=hswish`로 교체되고 `use_bias=(True,True,False)`(ops.py:370-377).
- **DSConv(ops.py:138-177)**: depthwise separable conv. `expand_ratio==1`인 stage에서 사용(backbone.py:100-108).
- **forward에서 중간 activation을 리스트로 수집(ops.py:230-236)** — 분포 분석/양자화 디버깅 흔적(plot은 주석).

#### 3.1.3 `EfficientViTBlock` (`ops.py:357-383`)

- 두 개의 residual로 구성: ① `context_module = ResidualBlock(LiteMSA, Identity)` (어텐션), ② `local_module = ResidualBlock(MBConv, Identity)` (지역 conv). 즉 **어텐션 + MBConv가 각각 잔차로 직렬** → 이게 Conv-Transformer 하이브리드의 한 블록.

#### 3.1.4 `EfficientViTBackbone` / `Cls` (`backbone.py`, `cls.py`)

- **백본(backbone.py:18-127)**: `input_stem`(stride2 conv + DSConv 잔차들) → stage1~2(DSConv/MBConv 지역 블록) → stage3~4(다운샘플 MBConv + **EfficientViTBlock 반복**). 앞 stage는 순수 conv, 뒤 stage에 어텐션 삽입 — 전형적 하이브리드 토폴로지.
- **모델 폭/깊이(backbone.py:130-167)**: b0=[8,16,32,64,128], b1=[16,32,64,128,256], b2=[24,48,96,192,384], b3=[32,64,128,256,512]. dim(헤드 차원) b0/b1=16, b2/b3=32.
- **분류 헤드(cls.py:18-32)**: `1×1 Conv → AdaptiveAvgPool → LinearLayer(LN, hswish) → LinearLayer(n_classes)`. 헤드에서 **LayerNorm 1회** 사용(`norm="ln"`, cls.py:23) — 이 LN은 양자화에서 경계(quant_model.py:49-50에서 LN 만나면 prev 끊음)로 취급됨.
- **활성/정규화(act.py, norm.py)**: 등록된 활성은 relu/relu6/**hswish**(act.py:11-15), 정규화는 bn2d/ln(norm.py:11-14). **Softmax/GELU가 레지스트리에 아예 없음** → softmax-free 설계가 모듈 레벨에서 강제됨.

### 3.2 PTQ 양자화 엔진(`quant/`)

#### 3.2.1 `UniformAffineQuantizer` — 핵심 양자화기(`quant_layer.py:62-202`)

- **스킴: asymmetric uniform affine(비대칭 균등)**. `self.sym=False` 강제(quant_layer.py:79). 양자화 공식(quant_layer.py:102-105):
  ```
  x_int   = round_ste(x/delta) + zero_point
  x_quant = clamp(x_int, 0, n_levels-1)          # n_levels = 2^n_bits (quant_layer.py:82)
  x_dequant = (x_quant - zero_point) * delta      # fake-quant: dequant까지 수행
  ```
  즉 **fake quantization**(정수→실수 복원)으로 학습/평가하며, 실제 정수 텐서를 내보내진 않는다.
- **round_ste(quant_layer.py:18-22)**: Straight-Through Estimator(`(round(x)-x).detach()+x`)로 round의 미분 통과.
- **스케일/제로포인트 산정(`init_quantization_scale`, quant_layer.py:107-183)**:
  - `channel_wise=True`(가중치 기본): 채널별 absmax로 delta/zp 산출. 4D 가중치는 `dim=0`(출력채널) 기준, 4D **activation은 `is_act` 플래그 시 `dim=1`(채널) 기준**으로 별도 처리(quant_layer.py:114-141) — activation per-channel을 위한 분기.
  - `scale_method='mse'`(main에서 사용, quant_layer.py:165-179): max를 1%씩 80스텝 줄여가며 **L_2.4 norm(LAPQ식)** 최소화 지점 탐색 → outlier에 강인한 클리핑. (`'max'`/`'max_scale'`는 단순 absmax, quant_layer.py:146-163.)
- **비트폭 동적 변경**: `bitwidth_refactor`(quant_layer.py:194-197)로 첫/끝 레이어 8bit 승격 지원(`set_first_last_layer_to_8bit`, quant_model.py:63-73 — 단 main에선 주석 처리됨).

#### 3.2.2 `QuantModule` 3종 — outlier 대응 양자화 모듈(`quant_layer.py:230-609`)

EfficientViT의 채널별 activation outlier(특히 depthwise conv 후, MBConv 내부)를 다루기 위해 **세 변종**이 존재한다. 이것이 Trio-ViT PTQ 엔진의 차별점.

1. **`QuantModule`(평범, quant_layer.py:230-291)**: `weight_quantizer`+`act_quantizer`. fake-quant weight로 conv/linear 수행 후 act 양자화(quant_layer.py:268-287). `disable_act_quant`로 elementwise add 전 양자화를 미룰 수 있음(quant_layer.py:283-284).
2. **`QuantModule_Shifted`(quant_layer.py:294-427)**: **채널별 shift(z) 양자화**. 입력의 채널별 (max+min)/2를 `z`로 잡아(quant_layer.py:356-362) `x-z`로 분포를 0 중심화한 뒤 양자화, conv 출력에 `z`에 대한 보정항(`shifted_bias` 또는 `diff_out`)을 더해 수학적 등가 유지(quant_layer.py:396-407). depthwise(weight.shape[1]==1)와 일반 conv를 분기 처리.
3. **`QuantModule_Scaled`(quant_layer.py:430-609)**: **채널별 scale(div) 양자화**. 채널 동적범위 비율을 2의 거듭제곱 근방으로 클램프(`div>64→64`, `div<1/10→1/10`, quant_layer.py:520-522)해 `input/scale`, `weight*scale`로 재분배(quant_layer.py:548-552) — 활성의 채널 outlier를 가중치로 흡수(SmoothQuant식 아이디어). shift도 함께 적용 가능(`enbale_shift`, quant_layer.py:577-596).
   - `get_ratio`(quant_layer.py:474-533): 99 percentile 기반으로 채널별 스케일 결정. 주석에 자동 탐색(LAPQ L_2.4) 버전도 보존(미사용).

- **`LogQuantizer`(quant_layer.py:205-227)**: 로그 도메인(2^y) 양자화기. 어텐션 분모처럼 동적범위가 큰 양에 사용. `y`를 [-12,11] 클램프.

#### 3.2.3 `AdaRoundQuantizer` — Adaptive Rounding(`adaptive_rounding.py:6-72`)

- **알고리즘**: AdaRound(arXiv:2004.10568). round를 floor+학습가능 이진결정으로 대체. `learned_hard_sigmoid` 모드에서 `x_int = floor(x/delta) + (alpha>=0)` 혹은 soft target `floor + h(alpha)`를 사용(adaptive_rounding.py:46-51).
- **soft target(adaptive_rounding.py:60-61)**: `clamp(sigmoid(alpha)*(zeta-gamma)+gamma, 0,1)` (rectified sigmoid, gamma=-0.1, zeta=1.1, adaptive_rounding.py:32). reconstruction 중엔 soft, 종료 후 hard로 고정(block_recon.py:127-129).
- **alpha 초기화(adaptive_rounding.py:63-69)**: `sigmoid(alpha)=rest`가 되도록 역산 → 시작점은 nearest rounding과 동일.

#### 3.2.4 블록/레이어 Reconstruction = BRECQ(`block_recon.py`, `layer_recon.py`)

- **block_reconstruction(block_recon.py:10-169)**: BRECQ의 블록 단위 출력 재구성.
  - 단계 A(가중치, act_quant=False): 블록 내 모든 QuantModule의 weight_quantizer를 **AdaRoundQuantizer로 교체**(block_recon.py:47-51), `alpha`만 Adam 최적화(block_recon.py:54-58). 손실 = **MSE reconstruction + rounding regularization**(LossFunction, block_recon.py:172-235). rounding loss는 `weight*(1-(|2(h-0.5)|)^b)`로 alpha를 0/1로 밀어내며, **온도 b를 20→2로 코사인/선형 감쇠**(LinearTempDecay, block_recon.py:238-256).
  - 단계 B(활성, act_quant=True): weight는 고정, **activation의 delta(스케일)를 Adam+CosineLR로 학습**(block_recon.py:60-86). LiteMSA 블록은 `qkv` 발견 시 별도 `msa_params`(어텐션 내부 양자화기 delta)를 모아(block_recon.py:66-84) **2차 패스로 MSA 양자화 캘리브레이션**(block_recon.py:135-160, `msa_quant=True`).
  - 캘리브레이션 데이터는 `save_inp_oup_data`로 블록 입출력을 미리 캐싱(block_recon.py:95, data_utils.py:8-37). `asym=True`면 양자화 입력 + FP 출력으로 재구성(BRECQ asymmetric).
- **layer_reconstruction(layer_recon.py:10-99)**: 동일 로직의 단일 레이어 버전(첫/끝 레이어 등 블록화 불가 대상용).

#### 3.2.5 `QuantModel` — 모델 전체 치환(`quant_model.py:7-89`)

- 생성 시 ① `search_fold_and_remove_bn`로 BN folding(quant_model.py:11) ② `quant_module_refactor`로 재귀 치환(quant_model.py:13).
- **치환 규칙(quant_model.py:15-53)**: `specials` 등록 블록(`BasicBlock/Bottleneck/ResBottleneckBlock/InvertedResidual/ResidualBlock`)은 대응 Quant 블록으로(quant_block.py:458-464), 그 외 `Conv2d/Linear`는 `QuantModule`로, `ReLU/ReLU6/Hardswish`는 직전 모듈의 activation_function으로 흡수하고 자신은 StraightThrough로 치환. **LayerNorm을 만나면 prev_quantmodule을 끊어**(quant_model.py:49-50) 양자화 경계로 둔다.

#### 3.2.6 `QauntMBBlock` — EfficientViT 블록의 양자화(`quant_block.py:190-455`)

- `ResidualBlock`(EfficientViT의 모든 잔차 블록)이 이 클래스로 치환됨(quant_block.py:463). 내부 main 타입으로 분기:
  - **DSConv(quant_block.py:205-210)**: depth/point conv를 QuantModule 2개로.
  - **MBConv(quant_block.py:213-229)**: inverted=평 QuantModule, **depth_conv=`QuantModule_Scaled`(enbale_scale=True)**, **point_conv=`QuantModule_Shifted`(enbale_shift=True)** — depthwise outlier는 scale로, point conv 입력은 shift로 처리(Trio-ViT의 채널 outlier 대응 핵심 배치).
  - **LiteMSA(quant_block.py:232-279)**: 어텐션을 양자화 가능한 `forward_attn`(quant_block.py:294-433)으로 재구현. qkv/aggreg/proj는 QuantModule, **kernel_func=ReLU**(softmax-free 유지). 추가로 어텐션 내부 중간값 양자화기 4쌍을 둠: `k_v_quant`(K^T·V 본체), `k_sum_quant`(∑k 분모항), `q_kv_quant_N`(Q·KV 분자), `out_quant`(최종). **분모(q_kv_1)는 `LogQuant`로 로그 양자화**(quant_block.py:360, 282-291) — 동적범위가 큰 정규화 분모를 정수 LUT/shift로 처리하기 위함. 정수 경로(`msa_quant`/`enbale_quant`)와 FP 경로를 둘 다 보존(quant_block.py:297-433).
- **BaseQuantBlock(quant_block.py:13-37)**: 모든 양자화 블록의 베이스. elementwise add **후** 활성/양자화를 적용(분기 구조 때문)하도록 act_quantizer를 블록 레벨에 둠.

#### 3.2.7 BN Folding(`fold_bn.py`)

- `_fold_bn`(fold_bn.py:14-34): Conv weight/bias에 BN의 (gamma/std, beta)를 흡수하는 표준 공식. `search_fold_and_remove_bn`(fold_bn.py:68-80)이 conv→bn 인접 쌍을 찾아 fold하고 BN을 StraightThrough로 치환. **양자화 전에 BN을 없애 정수 추론 친화 그래프**로 만든다.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 전체 파이프라인(`main_imagenet.py:134-265`)

```
1) seed 고정 + ImageNet DataLoader 구성        (main:177-179, data/imagenet.py: torchvision)
2) FP 모델 생성: create_cls_model(b1/b2/b3)     (main:184, cls_model_zoo.py:25-46 → .pt 로드)
3) (옵션) FP 정확도 측정                          (main:191-192)
4) QuantModel(cnn) 생성                          (main:197)
     ├─ BN folding (fold_bn.search_fold_and_remove_bn)
     └─ 재귀 치환: Conv/Linear→QuantModule, 블록→Quant*Block
5) 캘리브레이션 데이터 1024장 추출                 (main:205, num_samples=1024)
6) 가중치 양자화 파라미터 초기화: set_quant_state(W=on,A=off) 후 64장 forward  (main:210-211)
7) ★ 가중치 reconstruction(BRECQ+AdaRound): recon_model(qnn)   (main:247)
     └─ 블록별 block_reconstruction / 단일 레이어 layer_reconstruction (iters_w=20000)
8) W-only 정확도 측정                            (main:248-249)
9) (act_quant 시) 활성 양자화:
     ├─ set_quant_state(W=on,A=on) + 64장 forward로 act delta 초기화  (main:253-255)
     ├─ 네트워크 출력단 양자화 비활성             (main:258)
     └─ ★ 활성 reconstruction(delta 학습, LiteMSA는 MSA 2차 패스)  (main:261, iters_a=5000)
10) 최종 W{n}A{m} 정확도 출력                    (main:262-264)
```

### 4.2 Softmax-free 어텐션 연산 흐름(추론 시)

`x(B,C,H,W)` → 1×1 Conv로 QKV → (멀티스케일 DW/PW Conv 분기) → 헤드 분해 → **q=ReLU(q), k=ReLU(k)** → V에 1패딩 → **kv = K^T·V** (양자화: k_v_quant/k_sum_quant) → **Q·(KV)** (양자화: q_kv_quant_N + 분모 LogQuant) → 분자/분모 나눗셈 → out_quant → reshape → proj(1×1 Conv). 핵심은 **exp/softmax가 없고**, 행렬곱 2개 + ReLU + 나눗셈 1회로 끝난다는 점.

### 4.3 양자화 비트폭/스킴 요약

| 항목 | 값/방식 | 근거 |
|---|---|---|
| 가중치 비트폭 | `--n_bits_w`(기본 4, README 예시 8) | main:148 |
| 활성 비트폭 | `--n_bits_a`(기본 4, README 예시 8) | main:150 |
| 가중치 양자화 입도 | per-channel(`--channel_wise`) | main:149, quant_layer.py:109-144 |
| 활성 양자화 입도 | per-tensor(aq_params channel_wise=False) | main:196 |
| 스케일 산정 | MSE(L_2.4 탐색) | main:195-196, quant_layer.py:165-179 |
| 양자화 스킴 | asymmetric uniform affine(zero_point 존재), fake-quant | quant_layer.py:79,102-105 |
| 라운딩 | AdaRound(learned, rectified sigmoid) | adaptive_rounding.py |
| 재구성 | BRECQ 블록/레이어 reconstruction(MSE + round reg, 온도 20→2) | block_recon.py:172-256 |
| 어텐션 내부 | K^T·V/Q·KV per-tensor 양자화, 분모는 Log 양자화 | quant_block.py:282-367 |
| BN 처리 | Conv에 fold 후 제거 | fold_bn.py |
| 첫/끝 8bit 승격 | 지원하나 main에선 주석 처리 | quant_model.py:63-73, main:201-203 |

---

## 5. HW/SW 매핑 (알고리즘 측 관점 — 이 연산들이 왜 HW 친화적인가)

이 repo에 RTL/HLS는 없지만, 코드가 구현한 연산 특성은 명확히 **FPGA/ASIC 가속 친화적**이며, 논문의 "accelerator"가 노릴 지점을 코드에서 역추적할 수 있다.

1. **Softmax 제거 = 비선형 LUT 제거.** 표준 ViT 가속의 최대 난점은 exp/softmax(지수·나눗셈·reduction)인데, `LiteMSA`는 ReLU 커널(ops.py:337-338)만 사용. ReLU는 `max(0,x)` 비교 1회 → DSP/LUT 거의 무비용. **양자화-친화적이며 HW에서 softmax 유닛이 통째로 사라짐.**
2. **선형 어텐션의 결합법칙 재배치 = 누산기 친화.** `K^T·V`를 먼저 계산(ops.py:345)하면 토큰 수 N에 선형이고, 두 matmul 모두 **고정 크기(dim×dim, dim×(dim+1))** GEMM이라 systolic array/MAC 타일에 그대로 매핑된다. 분모 정규화를 V의 1-패딩(ops.py:342)으로 같은 GEMM에 흡수 → 별도 reduction 트리 불필요.
3. **분모 Log 양자화 = shift 연산화.** `LogQuant`(quant_block.py:282-291)는 분모를 2^y로 표현 → 나눗셈을 **비트 시프트**로 근사 가능(HW에서 나눗셈기 회피).
4. **채널별 Scale/Shift의 가중치 흡수.** `QuantModule_Scaled`는 `input/s, weight*s`(quant_layer.py:548-552)로 outlier를 가중치로 옮기고, `div`를 2의 거듭제곱 근방으로 제한(quant_layer.py:520, 주석의 power-of-2 버전 존재) → **추론 시 추가 곱셈 없이 시프트/사전계산**으로 처리 가능. Shift의 `z` 보정은 bias로 흡수(quant_layer.py:404-407) → 런타임 오버헤드 0.
5. **BN folding(fold_bn.py)** → 추론 그래프에 BN 연산 없음, Conv 정수 MAC만 남음.
6. **Conv-Transformer 하이브리드** → 전반부 conv(공간 locality, DSP 재사용)와 후반부 어텐션이 **동일한 Conv/GEMM 데이터패스**로 통합 가능(LiteMSA의 QKV/proj가 전부 1×1 Conv). 논문 abstract(README.md:12)의 "Convolution-Transformer hybrid 전용 accelerator"가 이 점을 겨냥한 것으로 **추정**.

> 단, 위 매핑은 **코드 특성 기반 추론**이며 실제 RTL/HLS 구현·자원 수치는 본 repo에 없어 **확인 불가**.

---

## 6. 빌드 · 실행 방법

- **의존 환경**: Python + PyTorch + torchvision(+ CUDA). 설치 스크립트/requirements 파일은 repo에 **없음**(확인 불가, pip 수동 설치 가정).
- **데이터셋**: ImageNet(`--data_path`로 경로 지정, `data/imagenet.py`가 torchvision `ImageFolder`식 로딩).
- **사전학습 가중치**: `EfficientViT/checkpoints/cls/{model}.pt` 필요(cls_model_zoo.py:11-21). 없으면 `--weight_url`로 지정. (별도 다운로드 필요 — repo 미포함)
- **대표 실행(README.md:20)**:
  ```bash
  python main_imagenet.py --data_path PATH_TO_IMAGENET \
    --n_bits_w 8 --channel_wise --weight 0.5 --model b1-r224 \
    --disable_8bit_head_stem --n_bits_a 8 --act_quant \
    --input_size 224 --test_before_calibration
  ```
- **주요 인자**(main:140-174): `--model`(b1-r224 등), `--n_bits_w`/`--n_bits_a`(비트폭), `--channel_wise`(가중치 per-channel), `--act_quant`(활성 양자화 on), `--num_samples`(캘리브레이션 1024), `--iters_w`(20000)/`--iters_a`(5000), `--weight`(rounding reg 가중치), `--b_start`/`--b_end`(온도 20→2), `--lr`/`--p`(활성 delta 학습).
- **분산 실행**: `main_imagenet_dist.py` + `--dist-url`(quant.sh:17-20 예시). `linklink`가 allreduce 헬퍼 제공.
- **quant.sh**: 실제 실행은 대부분 주석이고, 활성 라인은 mobilenetv2 + b1-r224 W8A8 1건(quant.sh:15). `--arch`는 BRECQ 잔재(실제 모델은 `--model`로 결정, main:184).

---

## 7. 의존성

| 의존 | 용도 | 근거 |
|---|---|---|
| PyTorch(`torch`, `torch.nn`, `torch.nn.functional`, `torch.autograd`) | 전 코드 기반 | 거의 모든 파일 |
| torchvision | ImageNet 로딩/transform | data/imagenet.py:3-6 |
| numpy | percentile/log 계산 | quant_layer.py:7,343 |
| linklink(자체 stub) | 분산 allreduce/allaverage | quant/*recon.py, quant_model.py:83 |
| (누락) `EfficientViT.plot` | 분포 시각화(import만, 호출부 주석) | ops.py:10 — **파일 부재, 정상경로 영향 없음** |
| timm | **미사용/미import**(grep상 코드 내 직접 사용 흔적 없음) | — (사용자 추정과 달리 직접 의존 확인 안 됨) |

> 참고: 사용자가 추정한 "timm"은 이 repo의 핵심 경로에서 직접 import 흔적이 보이지 않음(EfficientViT 모델은 자체 구현). hubconf/비교용 CNN 경로에서 일부 쓰일 가능성은 있으나 **분류 양자화 메인 경로(main_imagenet.py)에는 불필요**.

---

## 8. 강점 / 한계 / 리스크

### 강점
- **Softmax-free 선형 어텐션**을 모듈 레벨에서 강제(act 레지스트리에 softmax/gelu 부재) → 양자화·HW 친화성이 설계에 내장.
- **EfficientViT 채널 outlier에 특화된 PTQ**(Scaled/Shifted 모듈, MBConv depth/point conv별 차등 적용) → 일반 PTQ보다 정확도 보존 기대.
- **어텐션 내부 중간값까지 정수화**(K^T·V, Q·KV, 분모 Log 양자화) → 어텐션 전체를 정수 경로로 닫음(quant_block.py:282-367).
- BRECQ+AdaRound라는 검증된 PTQ 백본 위에 구축 → 재현성/이해도 높음.

### 한계
- **HW 가속 코드(RTL/HLS) 미포함** — 논문의 핵심 절반(accelerator)이 이 repo로는 검증 불가.
- **연구용 미정제 코드**: 다수의 `TODO/FIXME`, 하드코딩 상수(`div>64`, `LogQuant clamp [-12,11]`/[-6,9] 등 quant_layer.py:225, quant_block.py:287), 주석 처리된 plot/대체 버전이 산재.
- `EfficientViT/plot.py` 부재로 ops.py:10 import는 잠재적 ImportError(다만 호출부 전부 주석).
- `LogQuantizer.forward`/`QauntMBBlock.LogQuant`가 `.cuda()` 하드코딩(quant_layer.py:221, quant_block.py:283) → CPU/다중 디바이스 비호환.
- requirements/환경 파일 부재 → 재현 환경 명세 불충분.
- fake-quant만 구현(실 정수 텐서/커널 없음) → 정확도 검증용이며 실제 정수 추론 속도 이득은 별도 백엔드 필요.

### 리스크
- 사전학습 `.pt` 가중치가 repo에 없어 즉시 실행 불가(외부 확보 필요).
- `--arch`(BRECQ 잔재)와 `--model`(실제 선택자) 혼재로 인자 혼동 가능.

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT FPGA 가속 + XR 시선추적) 관점 시사점

1. **Softmax-free 선형 어텐션의 FPGA 적합성(직접 차용 가치 높음)**: HG-PIPE 같은 **완전 파이프라인(fully-pipelined) FPGA 가속기**에서 softmax는 파이프라인 stall/비선형 LUT의 주범이다. `LiteMSA`(ops.py:336-347)처럼 **ReLU 커널 + K^T·V 선형 어텐션 + 1-패딩 정규화**를 채택하면 softmax 유닛을 제거하고 어텐션을 **고정 크기 GEMM 2개**로 환원 → II=1 파이프라인에 매우 유리. 우리 가속기의 어텐션 stage 재설계 시 1순위 참고 패턴.
2. **EfficientViT 백본의 경량 XR 적용**: b0/b1(width 8~256, dim 16)은 파라미터/연산이 작아 **XR 시선추적의 저지연·저전력 백본** 후보. Conv-Transformer 하이브리드라 전반부 conv는 우리 CNN 데이터패스를, 후반부 어텐션은 동일 GEMM 타일을 재사용 가능 → **단일 MAC 어레이로 conv+attention 통합 매핑**이 현실적. 시선추적은 입력 해상도가 작아(예: 눈 영역 crop) H·W가 작고, 선형 어텐션의 O(N) 이점이 그대로 살아남.
3. **PTQ 코드 재사용(구체적)**:
   - `UniformAffineQuantizer`/`AdaRoundQuantizer`/`block_reconstruction`은 **모델 비종속**이라 우리 ViT/Transformer 가속기의 W/A 양자화 캘리브레이션 파이프라인으로 거의 그대로 이식 가능(입력만 우리 모델로 교체).
   - **`QuantModule_Scaled`/`QuantModule_Shifted`의 채널 outlier 처리**(quant_layer.py:430-609)는 우리가 INT8/INT4 어텐션을 FPGA에 올릴 때 활성 outlier로 인한 정확도 붕괴를 막는 실전 레시피. 특히 scale을 power-of-2로 제한하면 **시프트로 흡수** 가능(우리 HW에서 곱셈기 절감).
   - **분모 Log 양자화**(quant_block.py:282-291)는 선형 어텐션 정규화 나눗셈을 **시프트화**하는 직접적 HW 트릭 → 우리 어텐션 stage에 적용 검토.
   - **BN folding**(fold_bn.py)은 우리 가속기 컴파일 단계 전처리로 채택(추론 그래프 단순화).
4. **검증 전략**: 이 repo는 fake-quant로 정확도 상한을 빠르게 확인하는 용도로 쓰고, 우리 RTL/HLS의 비트 정확(bit-accurate) 결과와 **cross-check 기준선(golden)**으로 활용. (HG-PIPE 측 RTL ↔ Trio-ViT 측 fake-quant 출력 비교.)
5. **주의**: 본 repo는 알고리즘만이므로 HG-PIPE식 처리량/자원 수치는 여기서 얻을 수 없다. **알고리즘(양자화·어텐션 구조)은 차용하되, 가속기 마이크로아키텍처는 우리 HG-PIPE/타 REF(ViT-FPGA-TPU 등)에서 가져오는 분업**이 적절.

---

## 10. 근거 / 한계 표기

- **확인된 사실**:
  - 논문 제목/arXiv = README.md:3 명시(확인). abstract의 algorithm-level PTQ + hardware-level accelerator 주장 = README.md:11-12.
  - **HW 소스 부재 = Glob 직접 확인**(`.v/.sv/.cpp/.hpp/.cl/.tcl` 전부 0개). → 이 repo는 알고리즘 측 단독.
  - LiteMSA의 softmax-free 선형 어텐션, ReLU 커널, K^T·V 선결합, 1-패딩 정규화 = ops.py:336-347(코드 직접 확인).
  - PTQ = asymmetric uniform affine + AdaRound + BRECQ block reconstruction(quant_layer.py / adaptive_rounding.py / block_recon.py 직접 확인).
  - 채널 outlier 대응 Scaled/Shifted 모듈 + MBConv 배치(quant_layer.py:430-609, quant_block.py:213-229), 어텐션 분모 Log 양자화(quant_block.py:282-360) = 코드 확인.
- **추정(코드 정황 기반, 단정 불가)**:
  - HW/SW 매핑(섹션5)의 "HW 친화성"은 연산 특성에서 도출한 **추론**이며, 실제 RTL 자원/지연 수치는 **확인 불가**(repo에 HW 없음).
  - 논문 accelerator가 Conv-Transformer 통합 데이터패스를 노린다는 해석 = abstract 문구 기반 **추정**.
- **확인 불가/부재**:
  - requirements/환경 명세, 사전학습 `.pt`(경로만 존재), `EfficientViT/plot.py`(import만), 실제 정수 커널/백엔드.
  - timm 의존: 메인 경로에서 직접 import 미확인(사용자 추정과 상이).
- **분석 제약 메모**: `.git` 및 `.pt` 체크포인트는 지시대로 내부 미분석(이름만 언급). 본 분석은 `.py`/`.sh`/`.md` 텍스트 소스 전수 읽기에 기반.
