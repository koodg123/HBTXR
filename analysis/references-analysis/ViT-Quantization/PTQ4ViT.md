# PTQ4ViT 정밀 분석

> 분석 대상: `REF/ViT-Quantization/PTQ4ViT`
> 작성 기준: `README.md`, `quant_layers/{linear,matmul,conv}.py`, `utils/quant_calib.py`, `configs/{PTQ4ViT,BasePTQ}.py` 직접 분석. 라인 근거 표기.

## 1. 개요 (목적 / 원논문 / 핵심아이디어)

- **원논문**: *PTQ4ViT: Post-Training Quantization Framework for Vision Transformers* (arXiv:2111.12293, ECCV 2022), Zhihang Yuan et al. (PKU). (`README.md:1-5, 206-211`)
- **목적**: **Post-Training Quantization(PTQ)** 만으로(재학습 없이) ViT/DeiT/Swin을 8-bit에서 무손실에 가깝게(<0.5% drop) 양자화. (`README.md:5`)
- **핵심아이디어 2가지**:
  1. **Twin Uniform Quantization (쌍둥이 균일 양자화)**: ViT의 두 문제 분포 — (a) **softmax 출력**(0~1, 극히 비대칭/롱테일), (b) **GELU 출력**(음수 영역이 작고 양수 영역이 넓은 비대칭) — 에 대해 양/음(또는 고/저) 영역을 **서로 다른 scale로 분리 양자화**. (`README.md:3`, matmul.py `SoSPTQSLQuantMatMul` 284-, linear.py `PostGeluPTQSLQuantLinear` 262-)
  2. **Hessian Guided Metric**: scale 후보 탐색 시 단순 MSE 대신 **출력에 대한 손실의 Hessian(기울기 제곱)으로 가중한 오차**를 최소화. 출력 기여도가 큰 값의 양자화 오차를 우선 줄임. (`quant_calib.py:HessianQuantCalibrator 203-`, linear.py `_get_similarity` metric=="hessian" 144-146)
- **속도 최적화**: layer별 출력/기울기를 **사전 계산(parallel/batching)** 하고, scale 후보들을 **배치로 병렬 평가**하여 양자화 시간을 분 단위로 단축. (`README.md:42-46`, linear.py `parallel_eq_n`)

## 2. 디렉토리 구조

### 자체 소스
```
PTQ4ViT/
├── README.md
├── quant_layers/          # 양자화 가능한 레이어 정의 (핵심)
│   ├── linear.py          # MinMax→PTQSL→Batching→PostGelu 계층 구조 (642라인)
│   ├── matmul.py          # QK^T, scores@V 양자화 + Split-of-Softmax (644라인)
│   └── conv.py            # 패치 임베딩 conv 양자화
├── configs/
│   ├── PTQ4ViT.py         # hessian + twin quant 설정
│   └── BasePTQ.py         # 베이스라인(L2, minmax) 설정
├── utils/
│   ├── quant_calib.py     # QuantCalibrator / HessianQuantCalibrator (calib 엔진)
│   ├── net_wrap.py        # FP 모델 모듈 → 양자화 모듈 치환(wrapping)
│   ├── models.py          # timm ViT/DeiT/Swin 로더
│   ├── datasets.py        # ViTImageNetLoaderGenerator
│   └── integer.py         # calib된 fp32 → int8 변환 / 활성 fetch 훅
└── example/
    ├── test_all.py, test_vit.py, test_ablation.py, get_int.py
```

### 제외 항목
- `.git/`(VCS 메타), 다운로드 체크포인트(Google Drive 링크, `README.md:80-92`) — **대용량/외부**.
- timm 모델 정의 자체는 외부 패키지(`utils/models.py`가 timm 로드) — **외부 의존성**.

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 양자화 방식 개요
- **PTQ (uniform, symmetric per-tensor 기본, 블록/채널 단위 확장)**. round-to-nearest + clamp.
- 비트: `w_bit`, `a_bit`(linear/conv), `A_bit`, `B_bit`(matmul). `qmax = 2^(bit-1)` (부호 있음). (linear.py:29-30, matmul.py:16-17)
- 4-state forward 모드: `raw`(FP) / `quant_forward` / `calibration_step1`(원본 수집) / `calibration_step2`(scale 탐색). (linear.py:33-44, matmul.py:22-33)

### 3.2 `quant_layers/linear.py`
**`MinMaxQuantLinear` (6-92) — 베이스**
- `quant_weight_bias` (46-55): `w = round(W/Δ_w).clamp(-qmax, qmax-1) · Δ_w`. **가중치 fake-quant**. bias는 양자화 안 함(주석 처리, 49-53).
- `quant_input` (57-60): 동일한 round-clamp-scale로 **활성 fake-quant**.
- `calibration_step1` (79-84): FP 입출력을 CPU에 캐시(`raw_input`, `raw_out`).
- `calibration_step2` (86-92): **MinMax scale 산출** — `Δ_w = max|W| / (qmax-0.5)`, `Δ_a = max|x| / (qmax-0.5)`. 베이스라인.
- `_bias_correction_quant_forward` (69-77): 양자화 오차의 평균을 bias에서 차감하는 **bias correction**(선택).

**`PTQSLQuantLinear` (94-260) — PTQSL(Powerof-Two-free Sub-Layerwise) 핵심**
- 블록 분할 파라미터 `n_V`(출력 행 분할), `n_H`(입력 열 분할), `n_a`(활성 채널 분할) → **sub-layerwise(블록별) scale** (107, 117-119). `crb_rows = out_features//n_V` 등.
- `_get_similarity` (124-150): 양자화 품질 메트릭. `cosine / pearson / L1 / L2 / linear_weighted_L2 / square_weighted_L2 / **hessian**` 지원. hessian은 `-(raw_grad·(raw-sim))^2` (144-146) — **출력 기울기로 가중된 양자화 오차**.
- `_search_best_w_interval` (171-200) / `_search_best_a_interval` (202-225): scale 후보를 격자 탐색. 후보는 `Δ·[α + i(β-α)/n]` (247-248). **블록 h마다 최적 scale을 argmax(유사도)로 선택**, `parallel_eq_n`개씩 병렬 평가.
- `_initialize_intervals` (227-233): minmax로 초기화(layerwise 또는 블록별 amax).
- `calibration_step2` (235-260): 초기화 → `search_round`회 W/A 교대 탐색 → bias correction → 캐시 삭제.

**`PostGeluPTQSLQuantLinear` (262-347) — GELU 후단 Twin 양자화**
- `quant_input` (276-285): 활성을 **양수부/음수부 분리**. `x_pos`는 `[0, qmax-1]`로 `a_interval[0]`(탐색), `x_neg`는 `[-qmax, 0]`로 `a_interval[1]`(고정값 `0.16997.../qmax`, GELU 음수 최소값 근사) 양자화 → 합. (283-285) **GELU Twin**.
- `_initialize_intervals` (313-320): 음수 interval을 GELU 하한 상수로 설정 (320).

**`PTQSLBatchingQuantLinear` (349-555)** / **`PostGeluPTQSLBatchingQuantLinear` (557-642)**
- 캐시된 raw 데이터를 GPU 메모리 한도(`3GB/4` 가정, 372-378)에 맞춰 **배치로 분할** 처리하는 메모리 친화 버전. `_get_pearson_w/a`(426-453)로 빠른 pearson 유사도. **실제 PTQ4ViT 설정이 사용하는 클래스**(configs/PTQ4ViT.py:60-).

### 3.3 `quant_layers/matmul.py`
**`MinMaxQuantMatMul` (8-60)**: `A@B`의 A, B를 각각 minmax scale로 fake-quant. QK^T, scores@V 모두 입력 양자화. (35-45)

**`PTQSLQuantMatMul` (62-282)**: matmul을 그룹(`n_G`, head 단위)·행(`n_V`)·열(`n_H`)로 분할(`_get_padding_parameters` 109-122)하여 **head-wise/블록별 scale** 탐색. `_search_best_A_interval`/`_search_best_B_interval` (177-241) — Hessian/L2 등 메트릭으로 격자 탐색. 배치 버전은 head-wise 분할 자동 적용(`n_G_A=A.shape[1]`, 415-417).

**`SoSPTQSLQuantMatMul` (284-388) — Split-of-Softmax (핵심)**
- **문제**: softmax 출력(0~1)은 극소수 큰 값 + 다수 작은 값 → 균일 양자화 시 작은 값들이 0으로 뭉개짐. (docstring 285-296)
- **해법**: 구간을 split point로 둘로 쪼갬. `x_high` = `[split,1]` 구간을 `(qmax-1)` 레벨로 균일 양자화, `x_low` = `[0,split]` 구간을 `split/(qmax-1)` scale로 별도 양자화 → 합. (`quant_input_A` 313-316)
- split point는 `2^(-i)` 후보(i=0..19) 중 유사도 최대값 탐색 (`_search_best_A_interval` 318-344, 369). **부호 비트 불필요**(non-negative) → 하드웨어 효율 주석(296).
- 배치판 `SoSPTQSLBatchingQuantMatMul` (578-644).

### 3.4 `utils/quant_calib.py` — 캘리브레이션 엔진
- `QuantCalibrator` (9-171): `sequential_quant_calib`(메모리 친화, 순차) / `parallel_quant_calib`(전부 한 번에) / `batching_quant_calib`(레이어별 훅으로 raw 수집 후 step2). (28-171)
- forward/backward 훅 (173-201): `linear_forward_hook`, `matmul_forward_hook`(입력 2개), `grad_hook`(출력 기울기 수집).
- **`HessianQuantCalibrator` (203-378)**: 핵심. (1) FP 모델로 `raw_pred_softmax` 목표분포 계산(229-233). (2) 레이어별로 fwd 훅(입출력) + **bwd 훅(grad)** 등록. (3) calib 데이터로 forward → **KL(quant_pred ‖ raw_pred) 손실 backward** (258-259, 338-339)로 출력 기울기 수집 → 이 기울기가 hessian 메트릭의 가중치(`raw_grad`)가 됨. (4) `calibration_step2`로 scale 확정. **즉 "Hessian"은 KL 손실의 1차 기울기 제곱으로 2차(Hessian) 영향을 근사**.

### 3.5 `configs/PTQ4ViT.py` — 실제 설정
- `bit=8`, conv/linear/matmul 모두 8-bit (8-14).
- 공통 kwargs: `metric="hessian"`, `eq_alpha=0.01`, `eq_beta=1.2`, `eq_n=100`, `search_round=3` (16-48). 즉 scale 후보 = `Δ·[0.01 → 1.2]` 100분할, 3라운드 탐색.
- `get_module` (51-80): 레이어 이름으로 클래스 선택. `qlinear_MLP_2` → **PostGelu** Twin, `qmatmul_scorev` → **SoS** Split-of-Softmax, `qlinear_qkv` → `n_V*=3`(q,k,v 분리). patch-embed conv는 **활성 양자화 끔(`a_bit=32`)** (54). `no_softmax`/`no_postgelu` 플래그로 ablation.
- `BasePTQ.py`: 베이스라인(L2 metric, minmax). (configs 비교용)

## 4. 알고리즘 / 수식

**균일 양자화(대칭)**: `Q(x) = round(x/Δ)·Δ`, `clamp[-2^(b-1), 2^(b-1)-1]`. scale `Δ = max|x|/(2^(b-1)-0.5)` (minmax 초기값).

**Hessian-guided scale 탐색**: 각 블록에서
`Δ* = argmin_Δ  Σ g^2 · (o_raw - Q_Δ(o))^2`,
여기서 `g = ∂KL(p_quant‖p_fp)/∂o`(출력 기울기, raw_grad). 코드상 `similarity = -(raw_grad·(raw-sim))^2` 최대화와 동치. (linear.py:144-146)

**GELU Twin (양/음 분리)**:
`Q(x) = round(clamp(x,0,·)/Δ_pos)·Δ_pos + round(clamp(x,·,0)/Δ_neg)·Δ_neg`, `Δ_neg = 0.16997/qmax` 고정. (linear.py:283-285)

**Split-of-Softmax**:
`Q(a) = round(clamp(a,s,1)·(qmax-1))/(qmax-1) + round(clamp(a,0,s)/Δ_low)·Δ_low`, `Δ_low = s/(qmax-1)`, split `s ∈ {2^-i}`. (matmul.py:313-316, 369)

**복잡도**: attention 자체는 표준 `O(N^2 d)`(양자화는 연산 구조 불변, 비트폭만 저감). 캘리브레이션은 scale 후보 `eq_n` × `search_round`회 forward로 분 단위.

## 5. 학습 / 평가 파이프라인

- **재학습 없음(PTQ)**. ImageNet2012를 `/datasets/imagenet`에 두고 `ViTImageNetLoaderGenerator`로 로드(32 또는 128 calib 이미지면 충분, `README.md:12-26,110-113`).
- 명령:
  - 전체 모델 테스트: `python example/test_all.py` (`--multigpu --n_gpu 6` 지원). (`README.md:118-132`)
  - ablation: `python example/test_ablation.py`.
  - int8 변환/활성 추출: `utils/integer.py` (`README.md:94-98`).
- 결과: ViT/DeiT/Swin 8-bit에서 원본 대비 <0.5% drop; W6A6에서도 우수. (`README.md:154-168`)

## 6. 의존성

- `python>=3.5`, `pytorch>=1.5`, `timm`, `matplotlib`, `pandas`. (`README.md:102-107`)
- `numpy`, `tqdm`(quant_calib.py:7). timm으로 ViT/DeiT/Swin 로드 — **외부 의존성**.

## 7. 강점 / 한계 / 리스크

**강점**
- **재학습 불필요 + 빠른 calib(분 단위)** → 실무 적용성 최상.
- ViT 고유 분포 문제(softmax/GELU)를 **분포 인지 twin 양자화**로 정밀 대응 → 8-bit 거의 무손실.
- Hessian 메트릭으로 중요 값 우선 보존.
- 레이어/블록/head-wise scale을 통합 프레임워크로 제공, ablation 플래그 풍부.

**한계 / 리스크**
- scale 탐색이 GPU 메모리·시간 소모(특히 W6A6, 큰 모델 — `README.md:33,40`). calib 데이터에 약간 민감(거의 무시 가능 수준, 74-75).
- **fake quantization(시뮬레이션)** 중심 — 실제 정수 추론 커널은 `integer.py` 일부만(완전한 INT 추론 그래프 아님). 하드웨어 매핑은 별도 작업.
- Twin/SoS는 **두 scale·구간 분리** → 하드웨어에서 분기/합산 로직 추가 필요.

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + HG-PIPE + XR 시선추적) 관점 시사점

> 전제: 본 연구는 ViT FPGA 가속기 + XR 시선추적으로 **추정**.

- **분포 인지 양자화는 그대로 FPGA에 이식 가치가 높음**:
  - **Split-of-Softmax(SoS)**: softmax 출력의 롱테일을 두 구간으로 나눠 양자화 → 우리 가속기의 softmax 후단 INT8 표현에 직접 채택 가능. 특히 코드 주석(matmul.py:296)대로 **부호 비트가 불필요**(non-negative) → 1비트 절약 + 단순 unsigned 데이터패스. HG-PIPE식 파이프라인에서 attention score 버퍼를 unsigned INT로 운용 가능.
  - **GELU Twin(PostGelu)**: 음수부 scale을 **상수(0.16997/qmax)로 고정** → 런타임 탐색 불필요, 하드웨어에서 음수부는 고정 LUT/시프트로 처리, 양수부만 일반 양자화 → **저비용 비대칭 양자화 회로**로 매핑 가능. 시선추적 모델의 MLP GELU 후단에 적용 시 INT8 정확도 확보.
- **Hessian/중요도 가중은 오프라인 calib 단계 기법** → 우리 흐름에서 **호스트(SW) 측 calib 툴**로 흡수하고, RTL은 결정된 per-block scale만 받으면 됨. 즉 하드웨어 부담 없이 정확도 이득.
- **per-block / head-wise scale**: `n_V/n_H/n_G` 블록 단위 scale은 **FPGA의 타일링 구조와 자연스럽게 정합**. 타일 경계 = scale 경계로 두면, 타일별 dequant 상수만 ROM에 저장하면 됨. 다만 **scale granularity ↑ = 상수 저장량/스위칭 ↑** → 우리 BRAM 예산과 trade-off 분석 필요(추정).
- **균일 대칭 양자화 기본형**(`round/Δ`, clamp)은 시프트+곱셈으로 단순 매핑 → HLS/RTL 기본 데이터패스.
- **시선추적 적합성**: PTQ4ViT는 재학습 없이 기존 가중치를 8-bit화 → 우리가 사전학습 시선추적 ViT를 빠르게 INT8 가속기로 내릴 때 **무손실 양자화 경로**로 활용(추정). 단, Castling류 선형 attention과 결합하면 연산량+비트폭 동시 절감(세트 내 교차 활용).
- **주의**: 본 repo는 fake-quant 시뮬레이터라 실제 정수 GEMM 커널/오버플로 관리(누산기 비트폭)는 우리가 RTL에서 별도 설계해야 함(확인: integer.py 외 정수 추론 그래프 부재).

## 9. 근거 표기

- **확인**: linear.py(642L)·matmul.py(644L)·quant_calib.py(378L)·configs/PTQ4ViT.py(80L)·README.md 전수 직접 분석. 라인 번호 인용.
- **추정**: 우리 FPGA/XR 시사점(SoS unsigned 데이터패스, GELU Twin 고정 LUT, 타일=scale 경계 등)은 코드 구조로부터의 설계 추론.
- **확인 불가**: 실제 정수 추론 커널 전체, conv.py 세부(미독; 패치 임베딩 conv 양자화로 추정), 정확도 재현(체크포인트 외부).
