# Q-DETR (Quantized Detection Transformer) 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Quantization\Q-DETR`
> 분석 방식: 실제 소스 라인 단위(파일:라인). 추정/확인불가 명시.

---

## 1. 개요

- **목적**: Detection Transformer(여기선 **SMCA-DETR**)를 **저비트(2/3/4-bit) 정수 양자화** 하여 효율적 객체검출 수행. 4-bit 38.5 AP, 2-bit 32.4 AP(real 41.0) (`README.md:50-52`).
- **원논문**: *Q-DETR: An Efficient Low-Bit Quantized Detection Transformer* (CVPR 2023, Xu et al.) (`README.md:1,57-63`, arXiv 2304.00253). 기반 SMCA-DETR(`README.md:67`).
- **핵심 아이디어** (코드 확인 범위):
  1. **LSQ 기반 W/A 양자화 + per-channel/head 스케일**: Conv/Linear/Act를 LSQ로 양자화 (`lsq_plus.py:63-334`).
  2. **DistributionAlignment(분포 정렬)** ★: 어텐션 query(q)에 **정규화 + 학습형 per-head scale·bias**를 적용해 저비트화 시 정보 손실/분포 왜곡 보정 — 논문의 **distribution rectification** 의 코드적 실체 (`quant_attention_layer.py:19-34,426`).
  3. **양자 멀티헤드 어텐션**: q/k/v/attn을 헤드별 `ActLSQ`로 양자화, softmax는 float 유지 (`quant_attention_layer.py:333-368,434-437`).
  4. **GFFN/distribution rectification distillation(DRD) — 추정/부분확인**: 논문은 정보-병목 기반 distribution rectification distillation을 제시하나, 본 공개 코드의 학습 손실은 표준 DETR SetCriterion(focal/L1/GIoU)만 명시적으로 확인됨. 별도 teacher 분포 증류 모듈은 본 release에서 "확인 불가" (`detr.py:125-236`, `main.py`에 teacher 인자 없음).

---

## 2. 디렉토리 구조 (자체 핵심 / 제외)

```
Q-DETR/
├── main.py                       # 학습/평가 엔트리(--quant, --n_bit, --backbone_quant)
├── engine.py                     # train/eval (COCO)
├── models/
│   ├── __init__.py               # build_model / build_quant_model
│   ├── smca_detr/                # float SMCA-DETR (backbone/transformer/detr/attention)
│   └── quant_smca_detr/          # ★ 양자화 SMCA-DETR
│       ├── lsq_plus.py           # ★ LSQ 양자화기(Conv2dLSQ/LinearLSQ/ActLSQ/LinearMCN)
│       ├── _quan_base_plus.py    # 양자화 베이스(_Conv2dQ/_LinearQ/_ActQ)
│       ├── quant_attention_layer.py  # ★ QuantMultiheadAttention + DistributionAlignment
│       ├── attention_layer.py    # float MHA 참조
│       ├── transformer.py        # ★ 양자 Encoder/Decoder(FFN=LinearLSQ)
│       ├── detr.py               # DETR head + SetCriterion(focal/L1/GIoU)
│       ├── quant_resnet.py / quant_mobilenet.py  # 양자 backbone
│       ├── backbone.py / position_encoding.py / matcher.py / segmentation.py
├── datasets/                     # COCO 로더(coco.py/transforms.py/eval)
├── util/                         # box_ops/misc/plot
├── d2/                           # detectron2 연동(보조)
└── Adaptive_Cluster_Transformer/ # ★외부 프레임워크(ACT) — 핵심 아님, 제외 대상
```
**제외**: `Adaptive_Cluster_Transformer/`(외부 ACT, 핵심만 언급), `d2/`(detectron2 연동, 보조), `__pycache__`, 체크포인트(Google Drive, 이름만).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `models/quant_smca_detr/lsq_plus.py` — LSQ 양자화기 ★

- **`FunLSQ`** (`:23-48`): 표준 LSQ autograd(Q-ViT와 동일 형태). α 그래디언트 = 클리핑 구간별 `Qn/Qp/(-q_w+round(q_w))` (`:42-43`), STE로 가중치 중간구간만 통과 (`:44`).
- **STE 유틸** `grad_scale`/`round_pass` (`:51-60`).
- **`Conv2dLSQ`** (`:63-112`): kernel_wise(채널별) α. init `α=2·mean(|W|)/sqrt(Qp)` (`:81`), `g=1/sqrt(numel·Qp)` (`:94`), per-channel α broadcasting(`:100`), `w_q=round_pass(clamp(W/α,Qn,Qp))·α` (`:101`). 입력은 내부 `self.act=ActLSQ` (`:70,103`).
- **`LinearLSQ`** (`:153-189`) ★ FFN/proj 핵심: `qw()`로 가중치 LSQ 양자화(`:160-173`), forward에서 입력 `ActLSQ` 후 linear (`:179-189`).
- **`LinearLSQ_v2`** (`:115-150`): **이중 스케일 α⊗β**(outer product `scale=α@β`)로 더 미세한 per-element 스케일 (`:134-140`) — 추정: 가중치 분포 적응.
- **`LinearMCN` / `MCF_Function`** (`:192-252`): MCN(Modulated Conv) 스타일, MFilter 스케일 + 정규화 항 그래디언트(`:240-250`). 비교/대체 경로(추정).
- **`ActLSQ`** (`:255-334`) ★ 활성 양자화:
  - signed 자동감지(`x.min()<-1e-5`면 signed) (`:267-268`).
  - **per-channel/head α + zero_point**(비대칭): init `α=2·mean(|x|)/sqrt(Qp)`, `zero_point` EMA (`:276-277`).
  - 입력 차원(2/3/4D)별로 α,zp를 해당 축에 broadcasting하는 정교한 분기 (`:296-322`) — q/k/v(B,H,N,D)에서 **헤드축 매칭**.
  - `x = round_pass(clamp(x/α + zp, Qn, Qp)); x = (x - zp)·α` (`:326-327`) — **비대칭 LSQ**.

### 3.2 `models/quant_smca_detr/quant_attention_layer.py` — 양자 어텐션 ★

- **`DistributionAlignment`** (`:19-34`) ★★ **분포 정렬(distribution rectification 코드적 실체)**:
  - per-head `bias(head_dim)`, `scale(1,1,1,head_dim)` 학습 파라미터 (`:23-24`).
  - forward: **LayerNorm식 정규화** `(x - mean)/sqrt(var+eps)` 후 `scale·out + bias` (`:32-33`).
  - → 저비트 어텐션에서 q 분포를 표준화 + affine 재조정하여 양자화 정보 손실 보정.
- **`multi_head_attention_forward`** (`:36-380`): 커스텀 MHA. q/k/v를 in_proj(LinearLSQ)로 투영 후
  - q/k/v를 각각 `q_act/k_act/v_act`(ActLSQ, 헤드별)로 양자화.
  - `attn_output_weights = q@kᵀ·scaling` → mask → **`F.softmax`(float)** (`:333-355`) → dropout → **`attn_act`(ActLSQ로 attention map 양자화)** (`:360`) → `@v` → out_proj(float linear) (`:368-375`).
- **`QuantMultiheadAttention`** (`:385-...`):
  - encoder면 `norm_q = DistributionAlignment(head_dim)`, norm_k/v는 None (`:425-432`) → **q에만 분포정렬 적용**.
  - `q_act/k_act/v_act/attn_act = ActLSQ(nbits_a=8, in_features=num_heads)` (`:434-437`) — 헤드별 활성 양자화(여기선 8bit 기본, n_bit는 weight).
  - `in_proj = LinearLSQ(embed_dim, 3·embed_dim, nbits_w=n_bit)` (`:440`).

### 3.3 `models/quant_smca_detr/transformer.py` — 양자 인코더/디코더

- **`TransformerEncoderLayer`** (`:151-209`): `self_attn = QuantMultiheadAttention(n_bit)`, **FFN = LinearLSQ→activation→LinearLSQ** (`:156-161`). LayerNorm은 float.
- **`TransformerDecoderLayer`** (`:212-...`): self-attn + cross-attn(SMCA의 spatial modulated co-attention) + FFN(LinearLSQ) (`:221-223`). `dynamic_scale`/`smooth`로 SMCA 가우시안 prior 반영.
- → **FFN(=GFFN의 base) 전부 LinearLSQ 양자화**. 논문의 "GFFN(Gaussian-prior/distribution-aware FFN)" 명명 모듈은 본 코드에서 명시 클래스로 "확인 불가"이며, FFN 자체는 LSQ 양자화됨.

### 3.4 `models/quant_smca_detr/detr.py` — DETR head + 손실

- **`SetCriterion`** (`:107-236`): DETR 표준 손실.
  - `loss_labels`: `sigmoid_focal_loss`(box mode) (`:144`).
  - `loss_boxes`: L1 + GIoU (`:176-184`).
  - `loss_cardinality`, `loss_masks`(focal+dice) (`:153-211`).
  - Hungarian matcher(`matcher.py`)로 예측-GT 매칭.
- weight_dict: ce=2, bbox/giou coef (`:353-357`). → **distribution rectification distillation 손실은 이 criterion에 미포함**(teacher 인자 없음) → 본 release는 QAT 위주, DRD는 "확인 불가".

### 3.5 양자 backbone `quant_resnet.py` / `quant_mobilenet.py`
- `--backbone_quant` 시 ResNet/MobileNet conv를 `Conv2dLSQ`로 교체(추정: 동일 LSQ 적용).

### 3.6 외부 `Adaptive_Cluster_Transformer/` (ACT) — 핵심 아님
- 별도 DETR+ACT 실험 코드(`ACT/ada_clustering_attention.py` 등). Q-DETR 본체와 분리된 외부 프레임워크로 본 분석에서 제외(언급만).

---

## 4. 알고리즘 / 수식

### 4.1 LSQ W/A 양자화
- 가중치: `W_q = round_pass(clamp(W/α, Qn, Qp))·α`, per-channel α, `g=1/sqrt(N·Qp)` (`lsq_plus.py:101,167`).
- 활성(비대칭): `x_q = round_pass(clamp(x/α + z, Qn, Qp))`, `x = (x_q - z)·α` (`lsq_plus.py:326-327`).
- α 그래디언트(FunLSQ): 중간구간 `-q_w+round(q_w)`, 포화 `Qn/Qp` (`:42-43`).

### 4.2 DistributionAlignment (분포 정렬 / distribution rectification)
- `x̂ = (x − μ)/√(σ²+ε)` (head_dim 축 정규화) (`:32`).
- `out = γ·x̂ + β`, γ=per-head scale, β=per-head bias (`:33`).
- 인코더 q에 적용 → q·kᵀ 분포 표준화 후 affine 복원 → 저비트 양자화 강건성↑(추정: 논문 distribution rectification의 구현).

### 4.3 양자 어텐션 흐름
- `q,k,v = ActLSQ(in_proj_q/k/v)` (헤드별) → `A = softmax(q kᵀ·s)`(float) → `A_q = ActLSQ(A)`(attn 양자화) → `O = A_q · v` → out_proj.

### 4.4 검출 손실
- `L = λ_ce·FocalCE + λ_bbox·L1 + λ_giou·GIoU` (+mask/dice) (`detr.py:144-184,353-357`).

## 5. 학습 / 평가 파이프라인
- **데이터셋**: COCO 2017(train2017/val2017) (`README.md:24-34`).
- **양자 모델 빌드**: `main.py --quant` → `build_quant_model` (`main.py:131-132`). `--n_bit 2/3/4`, `--backbone_quant` (`main.py:40-51`).
- **평가**: `bash evaluate_coco.sh`, `--coco_path`, `--n_bit` 지정 (`README.md:37-43`).
- **백본/transformer 분리 lr**: backbone과 나머지에 다른 param group (`main.py:146-149`).
- **결과**: 4-bit 38.5 AP / 2-bit 32.4 AP, 50 epoch (`README.md:50-52`).

## 6. 의존성
- PyTorch 1.5+, torchvision 0.6+, pycocotools, scipy, (옵션) panopticapi (`README.md:8-22`). SMCA-DETR 기반.

## 7. 강점 / 한계 / 리스크
- **강점**:
  - 검출 transformer를 저비트로 양자화하면서 DistributionAlignment로 어텐션 분포 보정 → 2-bit에서도 실용 AP.
  - LSQ per-channel/head 비대칭 활성 양자화로 분포 다양성 대응.
  - 백본+transformer 통합 양자화 가능(`--backbone_quant`).
- **한계 / 리스크**:
  - **softmax/LayerNorm/out_proj 등 float 유지** → 완전 양자화 아님, HW에서 혼합정밀 처리 필요.
  - **distribution rectification distillation(DRD)** 가 본 공개 코드 학습 손실에 미확인 → 논문 핵심 일부가 release에 누락되었을 수 있음("확인 불가").
  - DETR 계열 학습은 50 epoch COCO로 비용 큼.
  - ActLSQ의 차원 매칭 분기(`:296-322`)가 복잡 → 비표준 텐서 형상에서 취약 가능(추정).

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 HG-PIPE + XR 시선추적) 관점 시사점 — 추정
- **검출-transformer 저비트 데이터패스**: in_proj/FFN/proj가 LSQ 정수화 → FPGA 정수 MAC + 출력 스케일(고정소수점) 매핑. 시선추적이 "동공/glint 검출" 같은 detection 헤드를 쓸 경우 Q-DETR식 저비트 검출 transformer가 직접 참조 가능(추정).
- **DistributionAlignment ↔ HW LayerNorm 유닛**: q 정규화+affine은 HG-PIPE의 LayerNorm 누산 유닛 + per-head scale/bias 레지스터로 매핑. 저비트 안정화를 위해 정규화 스테이지를 어텐션 앞단에 두는 설계가 합리적(추정).
- **softmax/attention 양자화 경계**: softmax는 float, 그 출력만 ActLSQ로 양자화 → 가속기에서 **softmax 비선형 유닛(exp/LUT) 분리 + 입출력 경계 양자화** 패턴(다른 ViT 양자화 repo와 일관). XR 저지연에서 softmax LUT 비용이 핵심.
- **백본+transformer 통합 저비트**: 시선추적 end-to-end 파이프(CNN backbone → transformer)를 단일 저비트 정밀도로 가속하는 근거.
- **트레이드오프**: 4-bit는 정확도 갭 작음(41.0→38.5), 2-bit는 갭 큼(→32.4) → 시선추적 정밀도 요구에 따라 3~4bit 권장(추정). Q-ViT/OFQ의 head-wise·진동억제 기법과 결합 시 저비트 검출 정확도 추가 개선 여지.

## 9. 근거 표기
- 라인 근거: 본문 (파일:라인) 직접 확인.
- "추정": GFFN 명명-FFN 대응, LinearLSQ_v2/MCN 용도, backbone 양자화 세부, HW 매핑.
- "확인 불가": distribution rectification distillation(DRD) 손실의 본 release 포함 여부(teacher 미발견), Google Drive 체크포인트, ACT 외부 모듈 세부, evaluate_coco.sh 내용.
