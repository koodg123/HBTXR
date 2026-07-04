# RepViT 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/RepViT`
> 분석 방법: Glob/Grep/Read 기반 자체 핵심 소스 정독, 라인 근거(파일:라인) 표기.
> **중요 사전 결론**: 이 repo는 "ViT-Quantization" 디렉토리에 위치하지만, **모델 양자화(PTQ/QAT, INT8, fake-quant, observer/qconfig 등) 코드는 자체 소스에 존재하지 않는다.** 핵심은 **구조적 재파라미터화(Structural Reparameterization)** 기반 효율적 CNN-스타일 ViT 백본이다. (근거: 아래 "양자화 유무 확인" 절)

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문 1**: *RepViT: Revisiting Mobile CNN From ViT Perspective* (CVPR 2024). Ao Wang, Hui Chen, Zijia Lin, Jungong Han, Guiguang Ding. (근거: `README.md:3`, `README.md:147-153`)
- **원논문 2**: *RepViT-SAM: Towards Real-Time Segmenting Anything* (arXiv 2312.05760). (근거: `README.md:1`, `README.md:19-21`)
- **핵심 아이디어**: 경량 ViT(MobileViT/LeViT 등)의 효율적 아키텍처 설계 선택지를 표준 경량 CNN(MobileNetV3)에 점진적으로 이식하여, **순수 경량 CNN** 계열인 RepViT를 만든다. 즉 self-attention 없이도 ViT급 정확도/지연 트레이드오프를 달성. iPhone 12에서 1ms 지연으로 ImageNet top-1 80% 초과를 주장. (근거: `README.md:40`)
- **핵심 메커니즘**: 학습 시에는 multi-branch(3x3 DW conv + 1x1 conv + identity) 구조를 쓰고, **추론 시 단일 3x3 conv로 융합(fuse)** 하는 RepVGG식 재파라미터화. BatchNorm도 직전 conv/linear로 흡수. (근거: `model/repvit.py:36-48`, `83-121`, `utils.py:227-236`)
- **양자화 위상**: 효율성은 "양자화"가 아니라 "재파라미터화 + DW/PW 분리 + SE"로 달성. 양자화는 우리 측 디렉토리 분류명일 뿐 코드 근거 없음.

---

## 2. 디렉토리 구조 (자체 소스 + 제외)

### 자체 핵심 소스 (분석 대상)
```
RepViT/
├── model/
│   ├── repvit.py        # ★ 핵심: 재파라미터화 블록 + RepViT 모델 + 변형(m0_6~m2_3)
│   └── __init__.py
├── main.py              # ImageNet 학습/평가 엔트리 (DeiT 계열 학습 루프 기반)
├── utils.py             # ★ replace_batchnorm(재파라미터화 적용), 분산학습/로깅 유틸
├── losses.py            # distillation loss 등
├── flops.py             # FLOPs 계산
├── export_coreml.py     # CoreML 변환 (모바일 배포)
├── data/                # 데이터셋/샘플러/augmentation (datasets.py, samplers.py, threeaugment.py)
├── detection/           # 다운스트림 객체검출 (MMDetection 연동)
│   └── repvit.py        # ★ 검출용 백본(동일 재파라미터화 블록, BACKBONES 등록)
└── segmentation/        # 다운스트림 시맨틱분할 (MMSegmentation 연동)
    └── repvit.py        # ★ 분할용 백본
```

### 제외 (지침에 따라 이름만)
- `.git/`, `__pycache__/` : 버전관리/캐시
- `detection/mmdet_custom/`, `detection/mmcv_custom/`, `detection/configs/_base_/`, `segmentation/configs/_base_/`, `segmentation/tools/` : MMDetection/MMSegmentation 외부 프레임워크 커스텀/설정/스크립트 (third-party 성격)
- `sam/` : RepViT-SAM 서브프로젝트 (SAM 디코더/인코더 export, gradio app 등 — 별도 setup.py 패키지). 본 분석에서는 백본 재사용 사실만 언급.
- `logs/*.json`, `*.txt` : 학습 로그 (대용량, 이름만)
- 체크포인트(`*.pth`, `*.mlmodel`) : README 링크로만 제공, 로컬 없음 가정

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `model/repvit.py` — 재파라미터화 빌딩 블록 (가장 중요)

이 파일이 RepViT의 본질이다. 모든 블록은 "학습 시 multi-branch → 추론 시 single conv"가 가능하도록 `fuse()` 메서드를 가진다.

#### (a) `Conv2d_BN` (repvit.py:26-48)
- Conv2d(bias 없음) + BatchNorm2d를 묶은 `nn.Sequential`. (repvit.py:30-32)
- BN weight를 `bn_weight_init`(기본 1, 잔차 출력단은 0)으로 초기화. (repvit.py:33-34)
- **`fuse()` (repvit.py:36-48)**: BN을 conv에 흡수.
  - `w = bn.weight / sqrt(bn.running_var + eps)` (repvit.py:39)
  - `W_fused = conv.weight * w[:,None,None,None]` (repvit.py:40)
  - `b_fused = bn.bias - bn.running_mean * bn.weight / sqrt(var+eps)` (repvit.py:41-42)
  - 결과: bias 있는 단일 Conv2d 반환. **추론 시 BN 연산 완전 제거** → HW 매핑에 매우 유리.

#### (b) `Residual` (repvit.py:50-80)
- `x + m(x)` 잔차. 학습 시 drop>0이면 확률적 스킵(repvit.py:57-59).
- **`fuse()` (repvit.py:63-80)**: 잔차의 identity 경로를 conv weight에 직접 더해 흡수.
  - DW conv(`groups == in_channels`)인 경우, 1x1 identity 커널을 3x3으로 zero-pad 후 weight에 가산(repvit.py:67-71). 즉 `+x` 항이 conv 가중치 속으로 합쳐짐.

#### (c) `RepVGGDW` — RepVGG식 Depthwise 블록 (repvit.py:83-121) ★
학습 시 **3개 분기**: `conv`(3x3 DW Conv+BN) + `conv1`(1x1 DW Conv) + `identity(+x)`. (repvit.py:86-92)
- forward: `bn( (conv(x) + conv1(x)) + x )` (repvit.py:91-92)
- **`fuse()` (repvit.py:94-121)**: 세 분기를 하나의 3x3 DW conv로 병합.
  1. `conv`의 BN을 먼저 흡수 → bias 있는 3x3 conv (repvit.py:96)
  2. 1x1 커널 `conv1_w`를 3x3로 zero-pad (repvit.py:104)
  3. identity 커널을 3x3로 구성 (repvit.py:106)
  4. `final_conv_w = conv_w + conv1_w + identity`, `final_conv_b = conv_b + conv1_b` (repvit.py:108-109)
  5. 마지막 `self.bn`까지 한 번 더 흡수 (repvit.py:114-120)
  - 결과: **추론 시 단일 3x3 depthwise conv 하나**. 이것이 RepViT의 token-mixer 핵심.

#### (d) `RepViTBlock` (repvit.py:124-160) ★
MetaFormer식 구조: **token_mixer(공간 혼합) + channel_mixer(채널 혼합/MLP)**.
- `assert hidden_dim == 2*inp` (repvit.py:130): 확장비 t=2 고정.
- **stride==2 (다운샘플) 분기 (repvit.py:132-144)**: token_mixer = DW Conv2d_BN → (SE 옵션) → PW Conv2d_BN. channel_mixer = Residual(PW 확장→GELU→PW 축소).
- **stride==1 (identity) 분기 (repvit.py:145-157)**: token_mixer = `RepVGGDW` → (SE 옵션). channel_mixer = Residual(PW 1x1 → GELU → PW 1x1, 출력 BN init=0).
- `use_hs`(hard-swish) 플래그는 코드상 GELU로 통일됨 (repvit.py:141, `nn.GELU() if use_hs else nn.GELU()` — 양분기 동일, **사실상 항상 GELU**). 주의 포인트.
- SE(Squeeze-Excite)는 timm의 `SqueezeExcite(inp, 0.25)` 사용 (repvit.py:135,149).

#### (e) `BN_Linear` / `Classfier` (repvit.py:163-216)
- `BN_Linear`: BatchNorm1d + Linear, 역시 `fuse()`로 BN을 Linear에 흡수 (repvit.py:172-186).
- `Classfier`: distillation 시 `classifier`와 `classifier_dist` 2-head, 추론 시 두 출력 평균 (repvit.py:196-203). `fuse()`는 두 head 가중치를 평균하여 단일 Linear로 병합 (repvit.py:205-216). → DeiT식 hard distillation 토큰 헤드.

#### (f) `RepViT` 본체 (repvit.py:218-245)
- **patch_embed**: `Conv2d_BN(3, c/2, 3, s2) → GELU → Conv2d_BN(c/2, c, 3, s2)` — 두 번의 stride-2 conv로 4배 다운샘플 (repvit.py:226-227). ViT의 patchify를 conv stem으로 대체.
- cfg 리스트 `[k, t, c, use_se, use_hs, s]`를 순회하며 `RepViTBlock` 쌓음 (repvit.py:231-234). 채널은 `_make_divisible(c,8)`로 8의 배수 정렬 (repvit.py:232, `_make_divisible` 정의 repvit.py:3-20).
- forward: 블록 순회 → `adaptive_avg_pool2d(x,1)` → flatten → classifier (repvit.py:239-245).

#### (g) 모델 변형 (repvit.py:250-504)
`@register_model`로 timm에 등록되는 6종: `repvit_m0_6`(repvit.py:251), `m0_9`(277), `m1_0`(313), `m1_1`(350), `m1_5`(385), `m2_3`(439). cfg에서 채널폭/깊이만 다름. 모든 블록 t=2(확장비 2), kernel 3, SE/HS 토글로 구성. (각 cfg 표 repvit.py:255-503)

### 3.2 `utils.py` — 재파라미터화 적용기 (utils.py:227-236) ★
```python
def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            fused = child.fuse(); setattr(net, child_name, fused); replace_batchnorm(fused)
        elif isinstance(child, nn.BatchNorm2d):
            setattr(net, child_name, nn.Identity())
        else:
            replace_batchnorm(child)
```
- 모델 트리를 재귀 순회하며 `fuse()`가 있는 모듈은 융합 모듈로 치환, 잔여 BN2d는 Identity로 제거. (utils.py:228-235)
- README에서 추론 변환 절차로 명시: `model = create_model('repvit_m0_9'); utils.replace_batchnorm(model)` (README.md:73-80).
- 나머지 `utils.py`는 `SmoothedValue`/`MetricLogger`(로깅, utils.py:11-153), 분산학습 초기화(`init_distributed_mode`, utils.py:202-224), EMA 체크포인트(`_load_checkpoint_for_ema`, utils.py:155-162) 등 DeiT 표준 유틸.

### 3.3 `detection/repvit.py` / `segmentation/repvit.py` — 다운스트림 백본
- `model/repvit.py`와 **동일한 재파라미터화 블록**(`Conv2d_BN.fuse`, `Residual.fuse` 등)을 복제하되, MMDetection의 `@BACKBONES.register_module()`로 등록(`detection/repvit.py:5-6`)하고 다단계 feature를 반환하도록 변형. (detection/repvit.py:1-55에서 Conv2d_BN/Residual 동일 구조 확인)
- 즉 검출/분할에서도 추론 시 단일 conv 융합 이점을 그대로 가져감. (`_BatchNorm`, `_load_checkpoint` import: detection/repvit.py:7-8)

### 3.4 양자화 유무 확인 (코드 근거)
- `quant|int8|fake_quant|observer|qconfig|fbgemm` 등 Grep 결과: 매칭된 파일은 전부 **주변부**였음 — `sam/`(SAM 서브프로젝트), `segmentation/tools/onnx2tensorrt.py`, `segmentation/tools/pytorch2onnx.py`, `segmentation/tools/pytorch2torchscript.py`, `convert_datasets/*` 등. **자체 분류/모델 정의(`model/repvit.py`, `main.py`, `utils.py`)에는 양자화 코드 없음.**
- TensorRT/ONNX 경로(`segmentation/tools/onnx2tensorrt.py`)는 MMSegmentation 표준 배포 도구이며 RepViT 고유 양자화 알고리즘이 아니다. → **결론: 본 repo의 효율 핵심은 "양자화 코드 미존재, 재파라미터화·효율 구조 위주"**.

---

## 4. 알고리즘 / 수식 (재파라미터화 융합)

### 4.1 Conv + BatchNorm 융합 (Conv2d_BN.fuse, repvit.py:36-48)
BN: `BN(y) = gamma * (y - mu)/sqrt(sigma^2 + eps) + beta`, conv 출력 `y = W*x`.
융합 가중치/편향:
```
W_fused = W * ( gamma / sqrt(sigma^2 + eps) )      # 채널별 스케일
b_fused = beta - mu * gamma / sqrt(sigma^2 + eps)
```
→ 추론 시 `out = W_fused * x + b_fused` 단일 conv. (repvit.py:39-42)

### 4.2 RepVGGDW 3-분기 → 단일 3x3 DW conv (repvit.py:94-121)
학습 분기: `out = BN( Conv3x3_DW(x) + Conv1x1_DW(x) + x )`
1. `Conv3x3_DW`의 BN을 먼저 흡수 → `(W3, b3)`
2. 1x1 커널을 3x3로 zero-pad: `W1_pad = Pad_{1,1,1,1}(W1)`
3. identity 커널: 중앙 1만 1인 3x3 DW 커널 `I`
4. 합산: `W_merge = W3 + W1_pad + I`, `b_merge = b3 + b1`
5. 최종 BN(`self.bn`) 한 번 더 흡수:
```
w = gamma_bn / sqrt(var_bn + eps)
W_final = W_merge * w[:,None,None,None]
b_final = beta_bn + (b_merge - mean_bn) * gamma_bn / sqrt(var_bn + eps)
```
→ **추론 시 단일 3x3 depthwise conv** (repvit.py:108-120). multi-branch의 표현력은 학습에서 얻고, 추론 비용은 conv 1개로 수렴.

### 4.3 Distillation head 융합 (Classfier.fuse, repvit.py:205-216)
두 BN-Linear head를 fuse 후 `W = (W_cls + W_dist)/2`, `b = (b_cls + b_dist)/2` 단일 Linear로 병합 (추론 시 평균이 가중치에 흡수).

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet-1K (분류). 다운스트림: COCO(검출/인스턴스분할), ADE20K(시맨틱분할). (README.md:108-115, 131-133)
- **학습 (8-GPU)** (README.md:118-122):
  ```
  python -m torch.distributed.launch --nproc_per_node=8 --master_port 12346 --use_env main.py \
      --model repvit_m0_9 --data-path ~/imagenet --dist-eval
  ```
  학습 루프는 DeiT 계열(distillation, threeaugment, EMA). (`data/threeaugment.py`, `losses.py`, `utils.py`의 EMA)
- **평가** (README.md:126-129):
  ```
  python main.py --eval --model repvit_m0_9 --resume <ckpt>.pth --data-path ~/imagenet
  ```
- **추론 변환**: `utils.replace_batchnorm(model)`로 재파라미터화 (README.md:73-80).
- **CoreML export**: `python export_coreml.py --model repvit_m0_9 --ckpt <ckpt>` (README.md:90-92) — iPhone 배포/지연 측정.
- **다운스트림**: `detection/`은 MMDetection(`train.py`, `dist_train.sh`), `segmentation/`은 MMSegmentation(`tools/train.py`).

---

## 6. 의존성
- `torch`, `torch.nn` (코어)
- `timm`: `SqueezeExcite`(repvit.py:22), `register_model`/`create_model`(repvit.py:247), `trunc_normal_`(repvit.py:162) — **timm 강결합**.
- 분산: `torch.distributed` (utils.py:8)
- 다운스트림: MMDetection/MMCV(`detection/repvit.py:5-8`), MMSegmentation
- 배포: coremltools(export_coreml.py), onnx/tensorrt(segmentation/tools)
- (참고) 학습 코드베이스는 LeViT/PoolFormer/EfficientFormer에서 차용 (README.md:137).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **추론 그래프가 극단적으로 단순**: 모든 블록이 conv/linear로 융합 → BN/multi-branch/identity 분기가 추론 시 사라짐. (repvit.py:36-121)
- **연산 종류가 적음**: 3x3 DW conv, 1x1 PW conv, GELU, SE, avgpool, Linear뿐. attention/softmax/layernorm 없음.
- **8의 배수 채널 정렬**(`_make_divisible`, repvit.py:3-20) → 타일링/병렬화 친화.
- 모바일 검증(CoreML, iPhone 12 지연) 실측 기반.

### 한계 / 리스크
- `use_hs` 플래그가 코드상 무의미(양 분기 GELU 동일, repvit.py:141) → cfg의 HS 컬럼이 실제 activation을 바꾸지 않음. 논문 의도(hard-swish)와 코드 불일치 **추정**.
- 양자화 코드 부재 → INT8/저비트 배포는 별도 도구(외부) 필요.
- SE 모듈(글로벌 평균풀링 기반 채널 게이팅, repvit.py:135)은 전역 reduction을 포함해 스트리밍/파이프라인 HW에서 약간의 동기화 비용.
- timm/MM* 프레임워크 강결합으로 순수 추론 그래프 추출에 약간의 정리 필요.

---

## 8. 우리 프로젝트 관점 시사점 (FPGA 가속기 HG-PIPE 계열 + XR 시선추적)

> 전제: 본 프로젝트는 "ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 **추정**됨. 아래는 FPGA 친화도 관점.

- **재파라미터화 → 추론 시 단일 conv는 HW 매핑에 매우 유리.** 학습/추론 그래프 분리로 추론 데이터패스가 conv+activation만 남음 → 파이프라인/시스톨릭 어레이 매핑이 단순. HG-PIPE식 레이어별 파이프라이닝과 궁합이 좋다. (근거: repvit.py:94-121, utils.py:227-236)
- **BN 제거(흡수)**: 추론 시 BN의 채널별 affine이 conv weight/bias로 흡수되어 별도 정규화 유닛이 불필요 → DSP/LUT 절감. (repvit.py:36-48)
- **attention/softmax/layernorm 부재**: 순수 CNN이므로, ViT 가속기에서 가장 까다로운 softmax·LN 데이터패스가 없음. 다만 "Transformer 가속기"라는 우리 타겟과는 연산 성격이 달라(본질이 CNN), **HG-PIPE의 attention 엔진을 직접 검증할 워크로드로는 부적합**. 대신 "효율 백본/스템" 또는 경량 비교 baseline으로 활용 가치. (추정)
- **SE의 전역 reduction**과 **distillation 2-head**는 추론 시 각각 채널 게이팅/단일 Linear로 정리되므로 큰 부담 아님(Classfier.fuse, repvit.py:205-216).
- **XR 시선추적 관점**: 입력 해상도가 작고 저지연이 핵심인 시선추적에서, conv-stem(2× stride-2)으로 빠르게 다운샘플하는 RepViT의 stem 설계와, 3x3 DW + 1x1 PW의 저연산 블록은 FPGA 저지연 추론에 참고 가치가 큼. (repvit.py:226-227, 124-160)
- **양자화 적용 시**: 본 repo는 양자화 미포함이므로, INT8 PTQ/QAT는 우리 측에서 추가 적용해야 함. 융합 후 단일 conv 구조는 per-channel weight 양자화에 자연스럽게 부합(BN affine이 이미 weight에 들어감) → **양자화 친화적 출발점**으로 평가 (추정).

---

## 9. 근거 표기 / 확인 불가 항목
- **확인됨(코드 직접 근거)**: 재파라미터화 fuse 로직, RepViTBlock 구조, replace_batchnorm, 모델 변형 cfg, 양자화 코드 부재 — 모두 위 파일:라인 근거.
- **추정**: `use_hs` 플래그 무력화가 의도된 단순화인지 버그인지(코드상 GELU 고정), HG-PIPE 적합성/양자화 친화성 평가는 우리 프로젝트 맥락 기반 해석.
- **확인 불가**: 실제 iPhone 지연/정확도 수치(README 표 README.md:64-70)는 외부 체크포인트/디바이스 측정 결과로 로컬 재현 불가. `sam/` 서브프로젝트 내부는 본 분석 범위에서 제외(백본 재사용 사실만 확인).
