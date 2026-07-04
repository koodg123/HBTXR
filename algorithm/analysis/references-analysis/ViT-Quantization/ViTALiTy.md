# ViTALiTy 정밀 분석

> 분석 대상: `REF/ViT-Quantization/ViTALiTy`
> 작성 기준: `README.md`, `src/vision_transformer.py`(Attention), `src/quantize.py`(CPT식 학습형 양자화), `src/quant_utils.py`(Intel NLP-architect식 QAT) 직접 분석. 라인 근거 표기.

## 1. 개요 (목적 / 원논문 / 핵심아이디어)

- **원논문**: *ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention* (**HPCA 2023**), Dass, Wu, Shi 외 (GATECH-EIC + 난징대). (arXiv:2211.05109, `README.md:1-11`)
- **목적**: ViT attention을 **알고리즘-가속기 공동설계(co-design)** 로 가속. softmax attention을 **선형 복잡도 Taylor attention(저랭크) + 희소(sparse) 보정**으로 분해. (`README.md:17-19`)
- **핵심아이디어**:
  1. **Linear Taylor Attention (저랭크, low-rank)**: softmax를 **1차 Taylor 전개(m=1)** 로 근사 → `K^T V`로 **전역 컨텍스트 행렬 G**를 먼저 만들고(키×값) `Q·G`로 출력 → **선형 복잡도** `O(Nd^2)`. (`README.md:19,25-26,34`, `vision_transformer.py:122-126`)
  2. **저랭크 + 희소 통합 학습**: 고차 Taylor 항(m>1, =softmax)을 **희소 attention(SANGER식)** 으로 근사하여 학습 시 보강 → 저랭크는 전역, 희소는 지역 정보. **추론 시에는 저랭크 항만** 사용(하드웨어 효율). (`README.md:19,26`, `vision_transformer.py:115-128`)
  3. **전용 가속기**: chunk 기반, **systolic array(SA-Diag/SA-General 분할)** + pre/post-processor(누산/나눗셈/덧셈 어레이), 4-level 메모리 계층(DRAM/SRAM/NoC/Reg), **intra-layer 파이프라인 + down-forward 누산 데이터플로우**. (`README.md:37-44`)
- **양자화**: QAT(학습 인지) 기반. 저랭크 항은 16-bit, 희소 항의 q/k는 **4-bit**로 양자화(`vision_transformer.py:105-106,116`). `quantize.py`는 **학습형 정밀도(learnable precision, CPT식)**, `quant_utils.py`는 **대칭 EMA/dynamic QAT(Intel)**.

## 2. 디렉토리 구조

### 자체 소스
```
ViTALiTy/
├── README.md, requirement.txt, LICENSE
├── figures/                       # workflow, TaylorAttentionFlow, hardware_overall (이미지)
└── src/
    ├── main.py, engine.py         # 학습/평가 루프 (DeiT 기반)
    ├── vision_transformer.py      # ViT + ViTALiTy Attention (핵심 알고리즘)
    ├── quantize.py                # CPT식 학습형 정밀 양자화 (QConv2d/QLinear/RangeBN)
    ├── quant_utils.py             # Intel식 대칭 QAT (QuantizedLinear/MatMul/Embedding)
    ├── models.py, resmlp_models.py, patchconvnet_models.py
    ├── patch_embed.py, mlp.py, drop.py, losses.py
    ├── datasets.py, samplers.py, utils.py, hubconf.py
    └── run_with_submitit.py
```

### 제외 항목
- `.git/`, `.github/`, `figures/`(이미지) — 비코드.
- DeiT 코드베이스(facebookresearch/deit)에서 파생(`README.md:64`) → 모델 골격 일부는 **외부 기원**. **하드웨어 가속기 RTL은 본 repo에 없음**(논문/그림만) → 확인 불가.

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 `vision_transformer.py` — `Attention` (92-137) — 핵심 알고리즘
- 생성자(93-106): 표준 `qkv`, `proj`, `scale = head_dim^-0.5`. `vitality=True`면 **4-bit `QuantMeasure`(q,k용)** 와 16-bit `QuantMeasure`를 둠 (104-106).
- **forward (108-137)**:
  - `vitality=False`(130-133): 표준 softmax attention `softmax(QK^T/√d)V` — `O(N^2 d)`.
  - **`vitality=True` (115-128) — Taylor attention(Algorithm 1)**:
    1. **희소(sparse) 보정 항**: `quant_q,quant_k = quant_4bit(q,k)` → `quant_attn = softmax(quant_q quant_k^T · scale)` → **임계 0.002로 마스킹** → `sparse = mask·quant_attn` (116-120). 즉 softmax attention의 **4-bit 희소 잔차**.
    2. **저랭크(low-rank) Taylor 항**: `k = k - mean(k)`(중심화, 122) → **`kv = k^T @ v`(전역 컨텍스트 G, 124)** → `attn = (sparse@v + q@kv)·scale` (126). `q@kv = Q(K^T V)` → **선형 복잡도**.
  - 즉 출력 = **저랭크 전역(q·G)** + **희소 지역(sparse·v)**. 추론 가속 시 저랭크 항만 남기는 것이 논문 설계(코드는 학습용으로 둘 다 계산).

### 3.2 `quantize.py` — CPT식 학습형 정밀 양자화 (590L)
- `calculate_qparams` (32-51): min/max로 `range`, `zero_point` 산출(비대칭). `reduce_dim/type`으로 채널/레이어 단위 선택.
- `my_clamp_round` (54-72): round+clamp의 STE(역전파는 항등 통과, 71). 양자화 미분 가능화.
- `UniformQuantize.quantize` (128-177): `out = round((x - zp)/scale + qmin)` clamp 후 dequantize. **학습형 정밀도 `prec_sf`**: `prec = round(prec_sf·(max_bit-min_bit)+min_bit)` (149-150) → **비트폭 자체를 학습**(mixed-precision). stochastic rounding 옵션(166-168).
- `UniformQuantizeGrad` (180-209): **기울기도 양자화**(CPT, Cyclic Precision Training 류) — backward에서 grad를 extreme range로 양자화 (204-208).
- `QConv2d`(270-413)/`QLinear`(416-494): `prec_w = nn.Parameter`(학습형 비트폭, 291,431). 가중치/활성/기울기 모두 양자화. `fix_bit`로 고정 비트 평가 경로(380-413).
- `RangeBN`(499-564): 양자화 친화 BatchNorm(min/max 기반 scale, 학습 통계). LayerNorm 대체 옵션.

### 3.3 `quant_utils.py` — Intel NLP-architect식 대칭 QAT (483L)
- 대칭 양자화: `quantize = round(x·scale).clamp(-thresh,thresh)`, `thresh = 2^(bits-1)-1` (28-36). scale은 `2^(bits-1)-1 / max|x|` (23-25).
- `FakeLinearQuantizationWithSTE` (45-59): fake-quant + STE(backward 항등, 59).
- `QuantizedLayer` (71-206): QAT 베이스. `mode ∈ {NONE, DYNAMIC, EMA}`(62-65). 학습 시 fake-quant, 평가 시 정수 가중치 캐시(`_eval`→`quantized_weight`, 152-158). 8-bit export 훅(188-200).
- `QuantizedLinear` (209-350): **추론 시 정수 연산 시뮬레이션** — `quantized_input @ quantized_weight + quantized_bias` → `dequantize(out, w_scale·in_scale)` (268-283). bias는 **accumulation 32-bit**(238) — 누산기 폭 명시. EMA로 활성 threshold 추적(342-349).
- **`QuantizedMatMul` (437-482)**: attention의 `Q@K`, `score@V`를 2/4/6/8-bit 대칭 양자화. EMA threshold (456-466). `build_quant_matmul`(479-482). → **attention matmul 정수화 경로**.

## 4. 알고리즘 / 수식

**Taylor attention(1차, m=1)** — softmax `exp(qk)` ≈ `1 + qk` 전개에서:
- 전역 컨텍스트(저랭크): `G = K_c^T V` (`K_c = K - mean(K)`), 출력 `≈ Q·G·scale`. 복잡도 `O(Nd^2)` (vs softmax `O(N^2 d)`). (`vision_transformer.py:122-126`)
- 고차항(m>1) ≈ softmax → 학습 시 **희소 근사** `sparse = (softmax(q_4 k_4^T) > 0.002)·softmax(...)`로 보강 (116-120).
- 최종: `Out = (sparse·V + Q·G)·scale`.

**대칭 양자화**: `q = clamp(round(x·s), -(2^{b-1}-1), 2^{b-1}-1)`, `s = (2^{b-1}-1)/max|x|`. (quant_utils.py:23-36)

**학습형 비트폭(CPT)**: `b = round(prec_sf·(b_max-b_min)+b_min)`, `prec_sf`는 미분 가능 파라미터. 기울기까지 양자화(저정밀 학습). (quantize.py:149-150,180-209)

**복잡도 요약**: ViTALiTy attention = 선형 `O(Nd^2)`(저랭크) + 희소항(학습 시만). 가속기는 SA-Diag(대각/소규모 행렬)·SA-General(일반 행렬) 분할로 두 경로를 매핑.

## 5. 학습 / 평가 파이프라인

- DeiT 기반(`README.md:64`). ImageNet, 분산 학습.
- 학습(ViTALiTy): `python -m torch.distributed.launch --nproc_per_node=8 main.py --model deit_tiny_patch16_224 --lr 1e-4 --epochs 300 --batch-size 256 --data-path <IMAGENET> --vitality` (`README.md:53-55`).
- 평가: 위에 `--eval` 추가. `--vitality` 없으면 vanilla softmax. (`README.md:56-61`)
- 의존성: `pip install -r requirement.txt` (`README.md:49`).

## 6. 의존성

- `torch`, `torch.nn.functional`, `numpy` (소스 import). `requirement.txt` 존재(세부 미독). DeiT/timm 계열 백본 — **외부 의존성**.
- 가속기 RTL/HLS는 **본 repo에 없음**(그림·논문만) — 확인 불가.

## 7. 강점 / 한계 / 리스크

**강점**
- **선형 Taylor attention** — softmax를 저랭크로 근사, `K^T V` 전역 컨텍스트 → N에 선형. 긴 토큰열/고해상도에 강함.
- **저랭크+희소 분해** — 추론 시 저랭크만 → 하드웨어 단순, 학습 시 희소로 정확도 보강.
- **알고리즘-가속기 co-design**(systolic array 분할, intra-layer 파이프라인) → 우리 연구와 직접적으로 유사한 사고방식.
- **풍부한 양자화 옵션** — CPT식 학습형 비트폭 + 기울기 양자화(quantize.py), Intel식 대칭 EMA QAT + 정수 추론 시뮬레이션 + 32-bit 누산기 명시(quant_utils.py).

**한계 / 리스크**
- 두 개의 양자화 프레임워크(quantize.py vs quant_utils.py)가 공존 → 어느 경로가 본 실험에 쓰였는지 코드만으로는 모호(확인 불가, main.py 추적 필요).
- 희소항 임계(0.002)·중심화(k-mean) 등 하이퍼 민감 가능.
- 가속기 RTL 부재 → 하드웨어 성능 수치는 논문 의존(재현 불가).
- Taylor 1차 근사는 attention 표현력 손실 가능(희소항으로 보강하나 추론 시 제거).

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + HG-PIPE + XR 시선추적) 관점 시사점

> 전제: 본 연구는 ViT FPGA 가속기 + XR 시선추적으로 **추정**. ViTALiTy는 **우리와 거의 동일한 문제의식(attention 선형화 + 전용 가속기)** 을 가진 가장 직접적인 레퍼런스.

- **저랭크 전역 컨텍스트 `G = K^T V` → 시스톨릭 어레이 매핑의 모범**: 토큰 수 N에 무관한 `d×d` 컨텍스트 행렬 → HG-PIPE식 파이프라인에서 **고정 크기 버퍼**. ViTALiTy의 SA-Diag/SA-General 분할은 우리가 attention의 두 matmul(QK, QG)을 **이종(heterogeneous) PE 어레이**로 나눌 근거. XR 시선추적 저지연에 직접 부합.
- **추론 시 저랭크만 = 하드웨어 단순화**: 우리 가속기는 **저랭크 경로만 RTL로 구현**하고 희소 경로는 학습 단계(SW)로 분리 → 추론 데이터패스 최소화. Castling-ViT와 동일한 "switching" 사상 → 두 repo 교차 참고 가치.
- **intra-layer pipeline + down-forward 누산 데이터플로우**: HG-PIPE의 파이프라인 설계와 직접 비교·차용 가능한 데이터플로우 사례(추정). pre-processor(누산/나눗셈/덧셈 어레이)는 LayerNorm/정규화 전처리 회로 설계의 참조.
- **양자화 레시피**:
  - `QuantizedMatMul`(2/4/6/8-bit 대칭 + EMA, **누산 32-bit**, quant_utils.py:238,437-482)는 우리 attention matmul 정수화의 직접 청사진 — 특히 **누산기 폭 32-bit 명시**는 우리 RTL 누산기 사이징의 안전 기준.
  - **q/k만 4-bit, 저랭크는 16-bit** 혼합 정밀(vision_transformer.py:105-116)은 우리가 **경로별 차등 비트폭**(희소=저비트, 저랭크=중비트)을 채택할 근거.
  - CPT식 학습형 비트폭(quantize.py:prec_w)은 하드웨어 직접 적용은 어렵지만(SW 학습), 최종 결정된 per-layer 비트폭만 RTL에 반영하면 됨.
- **시선추적 적합성**: 선형 attention + 4-bit 희소는 토큰 많은 시선추적에 이상적. 단 본 repo는 DeiT/ImageNet 대상 → 시선추적 데이터셋으로의 재학습/적용은 우리가 수행해야 함(추정).
- **결합 전략**: ViTALiTy(저랭크 Taylor + co-design) ↔ Castling(angular 선형) ↔ FQ-ViT(LN/softmax 정수화) ↔ PTQ4ViT(분포 인지 PTQ)는 상호보완. 우리 가속기는 **ViTALiTy의 데이터플로우 골격 + FQ-ViT의 정수 비선형 + PTQ4ViT의 PTQ scale**을 합성하는 방향이 유망(추정).

## 9. 근거 표기

- **확인**: README.md(전수), vision_transformer.py Attention(92-137, Taylor/sparse/저랭크 라인), quantize.py(590L, 학습형 정밀·기울기 양자화), quant_utils.py(483L, 대칭 QAT·QuantizedMatMul·32-bit 누산). 라인 인용.
- **추정**: 우리 FPGA/XR 시사점(SA 분할 매핑, 경로별 차등 비트폭, 데이터플로우 차용)은 코드+README 기반 설계 추론.
- **확인 불가**: 가속기 RTL/HLS(본 repo 부재, 논문·그림만), 두 양자화 경로 중 실험 사용본(main.py 미독), requirement.txt 세부, 정확도 수치.
