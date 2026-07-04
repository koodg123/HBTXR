# Next-ViT 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/Next-ViT`
> 원논문: **Next-ViT: Next Generation Vision Transformer for Efficient Deployment in Realistic Industrial Scenarios** (Li et al., arXiv:2207.05501, ByteDance AutoML)
> 분석 방식: 실제 소스(Glob/Grep/Read) 기반. 라인 근거 표기. 외부 mmdet/mmcv·`.git`·`__pycache__`·대용량 로그(이름만) 제외.

---

## 0. 핵심 결론 (먼저 읽기)

- **Next-ViT는 "효율 하이브리드 CNN-Transformer" 백본**이다. 양자화는 모델 알고리즘이 아니라 **배포 단계의 옵션**으로만 등장한다.
- **양자화 코드 유무**: 모델 내부에 QAT/PTQ/fake_quant 코드는 **없음**. 단, **TensorRT 배포 시 `--datatype int8` 플래그 경로가 존재**한다 (`deployment/export_tensorrt_engine.py:33-39, 116-117`). 이는 `trtexec`에 위임하는 PTQ 수준(엔진 빌드 옵션)이며, repo 자체에 calibration 데이터/observer/QAT 학습 루프는 없다. Grep `int8` 매치는 (1) export 스크립트 인자, (2) mmdet detection config의 `img_norm_cfg` 무관 매치뿐 (확인 완료).
- **배포 친화성이 설계 목표**: softmax ViT가 TensorRT/CoreML에서 느린 문제를 해결하기 위해 **NCB(conv 블록)+NTB(transformer 블록)를 NHS 전략으로 하이브리드 스택**하고, **추론 전 BN 재파라미터화(`merge_bn`)** 로 latency를 줄인다 (`nextvit.py:101-102, 134-147, 175-183, 249-253, 341-345`; `utils.py:241-281`).
- 코드 출처는 ByteDance 공식 구현 (`nextvit.py:1` `# Copyright (c) ByteDance Inc.`). detection/segmentation은 mmdetection/mmsegmentation에 백본만 끼우는 형태.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

### 목적 (README.md:19-20)
대부분의 ViT는 복잡한 attention과 모델 설계 때문에 TensorRT/CoreML 같은 실제 산업 배포 환경에서 CNN만큼 빠르지 못하다. Next-ViT는 **"CNN처럼 빠르고 ViT처럼 강한"** 비전 네트워크를 목표로, latency/accuracy trade-off에서 CNN·ViT·기존 하이브리드를 모두 능가한다고 주장.

### 3가지 핵심 구성요소 (README.md:20)
1. **NCB (Next Convolution Block)**: 지역 정보를 deployment-friendly하게 포착 (`nextvit.py:113-147`).
2. **NTB (Next Transformer Block)**: 전역 정보 포착, conv+transformer 혼합 (`nextvit.py:216-275`).
3. **NHS (Next Hybrid Strategy)**: NCB/NTB를 효율적 하이브리드 패턴으로 스택 (`nextvit.py:291-294`).

### 성능 (README.md:58-65)
- Next-ViT-S @224: 5.8 GFLOPs, 31.7M, **TensorRT 7.7ms / CoreML 3.5ms, Acc@1 82.5%**.
- Next-ViT-L @384: 32.0 GFLOPs, 57.8M, TensorRT 36.0ms, Acc@1 84.7%.

---

## 2. 디렉토리 구조 (자체 + 제외)

```
Next-ViT/
├── README.md                         # 논문/벤치/학습·배포 명령
├── requirements.txt / LICENSE(Apache 2.0)
├── images/ (structure.png 등)
├── logs/*.log                        # 사전학습/세그/검출 로그 (대용량, 이름만 — 제외)
├── classification/                   # ★ 핵심 자체 소스
│   ├── nextvit.py    ★★★             # NCB/NTB/NHS/E_MHSA/MHCA, NextViT, S/B/L
│   ├── main.py                       # timm 기반 학습/평가 진입점
│   ├── engine.py                     # train_one_epoch / evaluate
│   ├── utils.py      ★               # merge_pre_bn(BN 재파라미터화), flops 계산
│   ├── datasets.py / samplers.py / losses.py(DistillationLoss)
│   └── train.sh
├── deployment/                       # ★ 배포(양자화 옵션 포함)
│   ├── export_tensorrt_engine.py ★   # onnx→trtexec, fp16/int8 선택
│   └── export_coreml_model.py    ★   # torch.jit.trace → coremltools
├── detection/                        # mmdetection 연동 (Mask R-CNN)
│   ├── nextvit.py                    # 백본(검출용, FPN out_indices)
│   ├── configs/mask_rcnn_nextvit_*.py
│   └── train.py/test.py (mmdet 래퍼)
└── segmentation/                     # mmsegmentation 연동 (FPN/UperNet)
    ├── nextvit.py / configs/ / train.py/test.py
```

### 제외 항목 (절대 제약 준수)
- `.git/**`, `__pycache__/**` — 제외.
- `logs/*.log` (약 20여 개 학습 로그) — 대용량, 이름만 기록.
- `detection/`·`segmentation/`의 train/test 스크립트는 **mmdetection/mmsegmentation 외부 프레임워크 래퍼** → 백본(`nextvit.py`) 외에는 정밀 분석 생략. configs는 mmdet/mmseg DSL.

---

## 3. 핵심 모듈·파일별 정밀 분석 ★ (가장 중요)

### 3.1 `classification/nextvit.py` — 모델 정의 핵심

#### (a) `ConvBNReLU` (nextvit.py:15-33)
Conv2d(bias=False) + BatchNorm2d(eps=1e-5) + ReLU. stem과 기본 conv에 사용. **배포 친화 기본 단위**(ReLU/conv/BN만).

#### (b) `PatchEmbed` (nextvit.py:46-67)
다운샘플/채널 변경 담당. stride==2면 `AvgPool2d(2) + 1×1Conv + BN`, 채널만 다르면 1×1Conv+BN, 동일하면 Identity (nextvit.py:53-64). **stride conv 대신 AvgPool+1×1**을 써서 배포 호환성 확보.

#### (c) `MHCA` — Multi-Head Convolutional Attention (nextvit.py:70-88)
- `group_conv3x3`: groups=`out_channels//head_dim`인 3×3 그룹 컨볼루션 → **헤드별 지역 어텐션을 conv로 구현** (nextvit.py:77-78).
- BN + ReLU + 1×1 projection (nextvit.py:79-87).
- **softmax 없는 "어텐션": 순수 conv 연산** → CNN 가속기 친화.

#### (d) `Mlp` (nextvit.py:91-110)
1×1Conv → ReLU → Dropout → 1×1Conv. hidden은 `_make_divisible(in*mlp_ratio, 32)`. **`merge_bn(pre_norm)`** 메서드로 앞단 BN을 conv1에 흡수 (nextvit.py:101-102).

#### (e) `NCB` — Next Convolution Block (nextvit.py:113-147) ★
구조: `PatchEmbed → x + DropPath(MHCA(x)) → [norm] → x + DropPath(Mlp(norm(x)))`.
- `patch_embed`(다운샘플) → MHCA residual (nextvit.py:140-141).
- `merge_bn()`: `self.mlp.merge_bn(self.norm)` 호출, `is_bn_merged=True` 플래그 (nextvit.py:134-137).
- **ONNX export 또는 BN merged 상태면 norm을 건너뜀**(`torch.onnx.is_in_onnx_export()` 분기, nextvit.py:142-145) → 추론 그래프 단순화.
- 본질: **conv 기반 지역 블록** (MHCA=conv attention + conv MLP).

#### (f) `E_MHSA` — Efficient Multi-Head Self Attention (nextvit.py:150-213) ★
- Q/K/V를 **별도 Linear**로 생성 (nextvit.py:161-163).
- **Spatial Reduction**: `sr_ratio>1`이면 K,V 계산 전 `AvgPool1d(kernel=sr²)`로 토큰 수를 줄임 → attention 비용 감소 (nextvit.py:168-172, 190-199). PVT/Twins 식 SRA.
- 실제 attention은 **표준 softmax**: `attn = (q@k)*scale; attn.softmax(-1); (attn@v)` (nextvit.py:205-210). → **NTB 내부에 유일하게 softmax가 존재**.
- `merge_bn(pre_bn)`: q에 BN 흡수, sr_ratio>1이면 k/v에 BN+norm 흡수 (nextvit.py:175-183).

#### (g) `NTB` — Next Transformer Block (nextvit.py:216-275) ★
**전역(E_MHSA) + 지역(MHCA)을 채널 분할로 혼합**하는 하이브리드 블록.
- `mix_block_ratio=0.75`: 출력 채널을 MHSA용(`mhsa_out_channels`)과 MHCA용(`mhca_out_channels`)으로 분할 (nextvit.py:230-231).
- 경로:
  1. `patch_embed` → `[norm1]` → `rearrange b c h w → b (h w) c` → `E_MHSA` → residual로 가산 (nextvit.py:256-264). **전역 self-attention(softmax) 경로**.
  2. `projection(PatchEmbed)` → `MHCA` residual (nextvit.py:266-267). **지역 conv-attention 경로**.
  3. 두 경로를 `torch.cat([x, out], dim=1)`로 채널 결합 (nextvit.py:268).
  4. `[norm2]` → `Mlp` residual (nextvit.py:270-274).
- `merge_bn()`: e_mhsa, mlp의 BN 흡수 (nextvit.py:249-253).
- 본질: **softmax transformer(전역) + conv attention(지역)을 한 블록 안에서 병합**.

#### (h) `NextViT` (nextvit.py:278-372) — NHS 조립
- `stage_out_channels` (nextvit.py:285-288): 4 stage 채널 스케줄.
- **NHS 패턴** (nextvit.py:291-294):
  ```
  stage0: [NCB]*depth0
  stage1: [NCB]*(depth1-1) + [NTB]
  stage2: ([NCB,NCB,NCB,NCB,NTB]) * (depth2//5)   # 4 NCB마다 1 NTB
  stage3: [NCB]*(depth3-1) + [NTB]
  ```
  → **지역(NCB)으로 정보를 모으고 stage 끝/주기적으로 전역(NTB)을 배치**하는 것이 NHS의 핵심.
- stem: `ConvBNReLU` 4개(stride 2,1,1,2) (nextvit.py:296-301).
- stage 빌드 루프: stride는 `strides[stage]==2 and block_id==0`일 때만 2 (nextvit.py:310-326).
- 마지막 BN → AvgPool → Linear head (nextvit.py:330-335).
- `merge_bn()` (nextvit.py:341-345): 모든 NCB/NTB의 BN 흡수 — **배포 전 1회 호출**.

#### (i) 모델 변형 (nextvit.py:375-390)
- `nextvit_small`: depths=[3,4,10,3], path_dropout 0.1.
- `nextvit_base`: depths=[3,4,20,3], 0.2.
- `nextvit_large`: depths=[3,4,30,3], 0.2.
- stem_chs=[64,32,64] 공통. timm `@register_model`로 등록.

### 3.2 `classification/utils.py` — `merge_pre_bn` (utils.py:241-281) ★
**BN을 직전 Linear/1×1 Conv에 수학적으로 흡수**하는 재파라미터화. 배포 latency의 핵심.
- 단일 BN: `scale = (var+eps)^{-0.5}`, `extra_weight = scale*γ`, `extra_bias = β - γ*mean*scale` (utils.py:253-255).
- 이중 BN(pre_bn_2): SR 경로에서 두 BN 연쇄 흡수 공식 (utils.py:263-267).
- Linear/1×1Conv weight·bias에 반영 (utils.py:269-281). 1×1 conv는 `weight.shape[2]==weight.shape[3]==1` assert.
- → **추론 그래프에서 BN 노드 제거 = TensorRT/CoreML에서 conv+BN fusion을 PyTorch 단에서 선수행**.

### 3.3 `deployment/export_tensorrt_engine.py` ★ (양자화 옵션 위치)
- `--datatype {fp16,int8}` 선택 (export:33-39).
- 흐름: `create_model` → `merge_bn()` 호출(export:99-101) → `torch.onnx.export`(opset 13) → `onnxsim.simplify` (export:104-111) → `trtexec --onnx ... --saveEngine ... --explicitBatch --{fp16|int8}` subprocess 호출 (export:116-117) → 선택적 profile(warmUp/iterations) (export:121-123).
- **int8 경로는 `trtexec`에 전적으로 위임**: repo 내 calibration cache/데이터/QAT 없음 → **TensorRT 빌트인 PTQ 수준**(엔진 옵션). int8 정확도 보정용 calibrator를 별도 제공하지 않음 (확인 완료).

### 3.4 `deployment/export_coreml_model.py` ★
- `create_model` → `merge_bn()` (coreml:39-41) → `torch.jit.trace` (coreml:44) → `coremltools.convert(inputs=[ImageType(...)])` → `.mlmodel` 저장 (coreml:47-51).
- **CoreML은 fp16/내장 양자화에 의존**(coremltools 변환), repo에 명시적 INT 양자화 코드 없음.

### 3.5 `classification/main.py` / `engine.py` (학습 파이프라인, 라인 정밀분석 요약)
- timm 생태계 기반: `create_model`, `Mixup`, `LabelSmoothing/SoftTargetCE`, `create_scheduler`/`create_optimizer`(adamw), `NativeScaler`(AMP), `ModelEma`, `DistillationLoss` (main.py:10-21).
- 기본 하이퍼: epochs 300, opt adamw, sched cosine, lr 5e-4, clip-grad 5, weight-decay 0.05, drop-path 0.1 (main.py:25-56).

### 3.6 `detection/`·`segmentation/nextvit.py` (요약)
- 분류 nextvit.py와 동일 블록 정의를 재사용하되 `forward`가 multi-stage feature(`out_indices`/FPN용)를 반환하도록 변형되고, mmdet/mmseg `@BACKBONES.register_module()`로 등록(외부 프레임워크 의존). 정밀 라인 분석은 외부 의존성으로 범위 밖.

---

## 4. 알고리즘 / 구조 (NCB / NTB / NHS / E_MHSA)

### 4.1 NCB (지역, 순수 conv)
$$x \leftarrow \text{PatchEmbed}(x);\quad x \leftarrow x + \text{MHCA}(x);\quad x \leftarrow x + \text{Mlp}(\text{Norm}(x))$$
- MHCA = group-3×3 conv + BN + ReLU + 1×1 conv. **softmax 없음**.

### 4.2 E_MHSA (전역, SR + softmax)
$$\text{SR}: K,V \leftarrow \text{AvgPool1d}_{sr^2}(K,V),\qquad
A = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right),\quad O = AV$$
- sr_ratios=[8,4,2,1] (stage별, nextvit.py:280). 초기 stage일수록 강한 토큰 축소 → cost↓.

### 4.3 NTB (전역+지역 채널 혼합)
$$x \leftarrow \text{PatchEmbed}(x)$$
$$x \leftarrow x + \text{E\_MHSA}(\text{Norm}_1(x)) \quad (\text{MHSA 채널})$$
$$o \leftarrow \text{Proj}(x);\quad o \leftarrow o + \text{MHCA}(o) \quad (\text{MHCA 채널})$$
$$x \leftarrow \text{Concat}[x, o];\quad x \leftarrow x + \text{Mlp}(\text{Norm}_2(x))$$
- mix_block_ratio 0.75 = MHSA:MHCA 채널 비중.

### 4.4 NHS (Next Hybrid Strategy)
- stage2를 `(NCB×4 + NTB×1)` 주기로 반복(nextvit.py:293). **"여러 conv 블록으로 지역 특징을 축적한 뒤 한 번 전역 transformer"** → 전역 attention 횟수를 줄여 latency 절감하면서 전역 표현 유지.

### 4.5 BN 재파라미터화 (배포 latency)
- 학습은 conv-BN 분리, 배포는 `merge_bn`으로 BN을 conv/linear weight·bias에 흡수 (4.x §3.2 공식). → 추론 op 수 감소.

---

## 5. 학습 / 평가 / 배포 파이프라인

- **데이터셋**: ImageNet-1K(분류, ImageFolder train/val, README.md:36-53), COCO(검출, mmdet), ADE20K(세그, mmseg).
- **학습(분류)**: `cd classification && bash train.sh 8 --model nextvit_small --batch-size 256 --lr 5e-4 --warmup-epochs 20 --weight-decay 0.1 --data-path <imagenet>` (README.md:84-87). 384 finetune은 step sched, lr 5e-6 (README.md:89-92).
- **평가**: 동일 train.sh에 `--resume <ckpt> --eval` (README.md:97-101).
- **검출/세그**: `dist_train.sh configs/mask_rcnn_nextvit_small_1x.py 8` 등 mmdet/mmseg 분산 스크립트 (README.md:115-164).
- **배포 (★)**:
  - CoreML: `python3 export_coreml_model.py --model nextvit_small --batch-size 1 --image-size 224` (README.md:170-172). coremltools==5.2.0.
  - TensorRT: `python3 export_tensorrt_engine.py --model nextvit_small --batch-size 8 --image-size 224 --datatype fp16 --profile True --trtexec-path /usr/bin/trtexec` (README.md:188-191). tensorrt==8.0.3.4. `--datatype int8`로 INT8 엔진 빌드 가능(단 calibration 미제공).
  - latency 벤치: CoreML은 iPhone12 Pro Max(iOS16, Xcode14), TensorRT는 trtexec profile (README.md:182, export:121-123).

---

## 6. 의존성 (README.md:31, requirements.txt)
- torch==1.10.0, **timm==0.4.9**, mmcv-full==1.5.0, mmdetection==2.23.0, mmsegmentation==0.23.0.
- 배포: onnx, onnxsim(onnx_simplifier), coremltools==5.2.0, tensorrt==8.0.3.4(+trtexec), fvcore(FLOPs).
- einops, timm DropPath/trunc_normal_/register_model.

---

## 7. 강점 / 한계 / 리스크

### 강점
- **배포 친화 설계가 일급 목표**: BN 재파라미터화(merge_bn), ONNX-aware forward 분기, AvgPool 기반 PatchEmbed → TensorRT/CoreML에서 빠름.
- **하이브리드 효율**: NCB(순수 conv)로 대부분 처리 + NTB(주기적 전역)로 표현력 확보 → latency/accuracy trade-off 우수(README 벤치).
- **표준 연산 위주**: conv/BN/ReLU/Linear/softmax-SRA → 대부분 가속기에서 지원되는 연산.
- TensorRT int8 / CoreML fp16 배포 경로 제공(엔진 레벨).

### 한계 / 리스크
- **NTB 내부 softmax 존재**(E_MHSA, nextvit.py:207): 전역 attention은 여전히 softmax+행렬곱 → FPGA에서 exp/정규화 LUT 필요. 단, SR로 토큰 수 축소되어 부담은 경감.
- **양자화는 frame work 위임**: int8 정확도 calibration·QAT가 repo에 없음 → INT8 정확도 보장 책임은 사용자. 자체 양자화 레퍼런스로는 빈약.
- 검출/세그는 mmdet/mmseg 강결합 → 해당 부분 재현·이식 부담.
- `merge_bn`은 1×1 conv/Linear 한정(utils.py:273 assert) → 일반 k×k conv-BN fusion은 별도(보통 추론엔진이 처리).

---

## 8. 우리 프로젝트(HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적) 관점 시사점

> 부모 경로(`PRJXR-HBTXR`, HGTXR/HGPIPE) 기반 **추정**. 본 repo에 FPGA/시선추적 언급 없음(확인 완료).

- **하이브리드 conv-transformer 매핑 모델로 적합(추정)**: HG-PIPE류 파이프라인 가속기는 conv 엔진이 잘 갖춰져 있는데, Next-ViT는 연산의 대부분이 NCB(group conv + 1×1 conv)로 구성되어 **기존 conv 데이터패스에 직접 매핑**할 수 있다. NTB만 transformer 경로로 분기하면 되어, "conv 위주 파이프라인 + 소수 attention 모듈" 구조와 잘 맞음.
- **NHS = attention 빈도 최소화 → 파이프라인 stall 절감(추정)**: stage2가 `NCB×4 + NTB×1`이라 전역 attention(stall 유발 reduction)이 드물게 등장. XR 시선추적의 저지연 스트리밍에서 NTB만 별도 처리하고 NCB를 연속 파이프라인으로 흘리면 throughput 안정화에 유리.
- **softmax는 NTB에만 국한**: HG-PIPE에서 softmax/exp LUT는 NTB의 E_MHSA에만 필요. 게다가 **SR(AvgPool)로 K/V 토큰 수가 줄어** softmax 입력 크기가 작아져 LUT/reduction 비용이 완화됨(특히 초기 stage sr=8). 시선추적용 작은 ROI 입력이면 토큰 수가 더 작아 유리(추정).
- **BN 재파라미터화는 FPGA 합성 전 그대로 활용 가능**: `merge_bn`을 PyTorch 단에서 선수행하면 하드웨어에 BN 모듈을 구현할 필요 없이 conv weight/bias에 흡수된 형태로 양자화·합성 가능 → HW 구현 단순화에 직접적 이득(확인된 코드 기반 + 적용은 추정).
- **양자화 출발점**: TensorRT int8 경로(export:116-117)는 calibration이 없어 그대로 FPGA INT8로 옮길 수 없음. HG-PIPE 통합 시 (1) NCB(group/1×1 conv)는 표준 conv INT8 양자화로 쉽게, (2) NTB의 E_MHSA softmax·SR·행렬곱은 별도 fixed-point 설계 + calibration 필요. **NCB 양자화는 저위험, NTB 양자화가 핵심 리스크**(추정).
- **AvgPool 기반 PatchEmbed**: stride-conv 대신 AvgPool+1×1라 다운샘플이 HW에서 단순(누산+시프트). 시선추적 다해상도 처리 시 유리(추정).

---

## 9. 근거 표기 (추정 / 확인 불가 구분)

- **확인 완료**:
  - 양자화: 모델 내부 QAT/PTQ 없음, TensorRT `--datatype int8` 빌드 옵션만 존재(export_tensorrt_engine.py:33-39, 116-117), calibration 미제공.
  - NCB/NTB/NHS/E_MHSA/MHCA 구조 및 라인: nextvit.py 해당 라인.
  - BN 재파라미터화: utils.py:241-281, 각 블록 merge_bn 호출부.
  - 배포 흐름(onnx→trtexec, jit.trace→coreml): deployment/*.py 라인.
  - NTB 내부에만 softmax 존재: nextvit.py:207.
- **추정**: 8장 FPGA/시선추적 시사점 전부(부모 경로명 기반, 본 repo 무관).
- **확인 불가**: 실측 FPGA latency/정확도, INT8 변환 후 정확도(외부 trtexec/하드웨어 의존), detection/segmentation 백본 forward 세부(외부 mmdet/mmseg 의존으로 정밀 분석 생략).
