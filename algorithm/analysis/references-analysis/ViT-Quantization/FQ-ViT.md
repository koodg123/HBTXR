# FQ-ViT 정밀 분석

> 분석 대상: `REF/ViT-Quantization/FQ-ViT`
> 작성 기준: `README.md`, `models/ptq/layers.py`, `models/ptq/quantizer/{base,uniform,log2}.py`, `models/ptq/observer/{base,ptf}.py`, `models/ptq/bit_type.py` 직접 분석. 라인 근거 표기.

## 1. 개요 (목적 / 원논문 / 핵심아이디어)

- **원논문**: *FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer* (IJCAI 2022), Lin et al. (MEGVII). (arXiv:2111.13824, `README.md:1-4,130-136`)
- **목적**: ViT를 **완전(fully) 양자화** — 기존 연구가 부동소수점으로 남겨두던 **LayerNorm과 Softmax까지 정수화**하여 최초의 fully quantized ViT 구현. (`README.md:24`)
- **핵심아이디어 2가지**:
  1. **PTF (Power-of-Two Factor) — LayerNorm 양자화** (`README.md:26-34`): ViT는 채널 간 분포 편차(inter-channel variation)가 CNN보다 심해 layer-wise 양자화 시 큰 오차. PTF는 **채널마다 서로 다른 2의 거듭제곱 인자**를 부여(scale 자체가 아니라 2^k 시프트)하여 channel-wise의 정밀함을 layer-wise의 단순함으로 근사. (observer `ptf.py`)
  2. **LIS (Log-Int-Softmax) — Softmax 양자화** (`README.md:36-50`): softmax 출력은 작은 값에 밀집 + 소수 큰 outlier → **Log2 양자화**가 균일 양자화보다 작은 값 구간에 더 많은 bin 할당. 여기에 I-BERT의 **i-exp(지수의 정수 다항 근사)**를 결합해 **정수 전용(integer-only) softmax**를 4-bit로 구현. (`layers.py:QIntSoftmax`, `quantizer/log2.py`)
- **양자화 방식**: **PTQ**(calibrate→quant 2단계). W8A8 + Attn4까지. 결과: DeiT/ViT/Swin에서 baseline(MinMax/EMA/Percentile/OMSE) 대비 큰 정확도 우위, 특히 ViT-B/L에서 MinMax가 붕괴(23%/3%)하는 반면 FQ-ViT는 83%/85% 유지. (`README.md:113-123`)

## 2. 디렉토리 구조

### 자체 소스
```
FQ-ViT/
├── README.md
├── test_quant.py              # 엔트리 (calibrate + evaluate)
├── config.py                  # 양자화 설정(비트/observer/quantizer 선택)
├── models/
│   ├── vit_quant.py           # 양자화 ViT
│   ├── swin_quant.py          # 양자화 Swin
│   ├── layers_quant.py        # 양자화 모델 구성 레이어(블록/attention)
│   ├── utils.py
│   └── ptq/                   # 양자화 엔진 (핵심)
│       ├── layers.py          # QConv2d/QLinear/QAct/QIntLayerNorm/QIntSoftmax (299L)
│       ├── bit_type.py        # BitType (uint4/int8/uint8)
│       ├── quantizer/         # base / uniform / log2 + build
│       └── observer/          # base / minmax / ema / percentile / omse / ptf + build
```

### 제외 항목
- `.git/`(VCS), `figures/`(이미지) — 비코드.
- timm 등 외부 백본 패키지 — **외부 의존성**.

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 양자화 추상화 (Observer + Quantizer 분리 설계)
FQ-ViT는 **Observer(통계 수집) ↔ Quantizer(scale/zp 적용)** 를 분리한 깔끔한 빌더 패턴. (`layers.py:48-51` build_observer/build_quantizer)

**`BitType` (bit_type.py:7-47)**: `bits`, `signed`로 상하한 계산. `uint4`(0~15), `int8`(-128~127), `uint8`(0~255). (42-46) → 활성/가중치별 부호 선택 가능.

**`BaseObserver` (observer/base.py:5-36)**: `max_val/min_val` 추적. `reshape_tensor`(15-28)로 모듈 타입별 reshape — weight는 `(out, -1)`, activation은 `(-1, last_dim)` 후 transpose → **마지막 차원(채널) 기준 통계** 수집(channel-wise 가능 구조).

**`BaseQuantizer` (quantizer/base.py:6-45)**: `get_reshape_range`(14-31)로 scale 브로드캐스트 shape 결정 — conv_weight `(-1,1,1,1)`, linear_weight `(-1,1)`, activation은 차원수별. `forward = quant→dequantize`(42-45) → **fake quantization**.

### 3.2 `UniformQuantizer` (quantizer/uniform.py:8-41)
- `quant` (19-30): `out = round(x/scale + zero_point).clamp(lower, upper)`. **비대칭 affine 양자화**(zero_point 사용).
- `dequantize` (32-41): `(x - zero_point)·scale`.
- scale/zp는 observer의 `get_quantization_params`에서 받음 (15-17). **per-channel/per-layer**는 observer가 결정.

### 3.3 `Log2Quantizer` (quantizer/log2.py:7-26) — LIS의 일부
- `quant` (17-21): `out = clamp(round(-log2(x)), 0, 2^bits-1)`. **로그 도메인 양자화** — 값이 작을수록 큰 정수. `softmax_mask`로 표현 범위 밖(너무 작은 값)을 기록 (19).
- `dequantize` (23-26): `2^(-x)`, mask된 곳은 0. → softmax처럼 0~1 밀집 분포에 적합.

### 3.4 `PtfObserver` (observer/ptf.py:8-66) — PTF의 핵심
- `update` (14-29): channel별 max/min 추적, layer_wise면 전체 max/min.
- `get_quantization_params` (31-66): **핵심 알고리즘**
  1. 전체 layer scale `scale8 = (max-min)/(qmax-qmin)` 계산 (41). 그리고 `scale4=scale8/2`, `scale2`, `scale1` (43-45) → **2의 거듭제곱으로 나눈 4단계 scale**.
  2. `zero_point = qmin - round(min/scale8)` (46).
  3. **채널 j마다** 4개 scale(1/2/4/8)로 양자화-복원한 뒤 L2 손실(`lp_loss`) 비교 → 최소 손실 scale을 `2^index`로 선택 (49-64). `scale_mask[j] *= 2^best`.
  4. 최종 `scale = scale1 * scale_mask` (65) → **채널마다 2^k 배율(=PTF)이 곱해진 scale**. zero_point는 공유.
- → layer-wise 한 개의 base scale + **채널별 power-of-two 시프트**로 channel-wise 효과. 하드웨어에서 채널별 scale 곱셈 대신 **시프트(2^k)** 로 구현 가능 → **핵심 하드웨어 친화 포인트**.

### 3.5 `models/ptq/layers.py` — 양자화 레이어 (299L)
**`QConv2d`(11-70) / `QLinear`(73-110) / `QAct`(113-148)**: 공통 패턴 — `calibrate` 모드면 observer 통계 갱신, `last_calibrate`면 scale/zp 확정 (54-57, 103-106, 140-144). `quant` 모드면 가중치/활성을 `self.quantizer()`로 fake-quant 후 연산. patch-embed conv부터 classifier까지 전부 교체 가능.

**`QIntLayerNorm` (151-206) — 정수 LayerNorm (PTF 적용처)**
- `get_MN` (159-163): 실수 A를 `M·2^(-N)`(고정소수점)으로 분해. `N = clamp(7 - floor(log2(x)), 0, 31)`, `M = clamp(floor(x·2^N), 0, 255)` → **8-bit 정수 곱 + 시프트로 실수 곱 근사**.
- `forward` int 모드 (173-203): 입력 scale로 정수화(`x_q = round(x/in_scale)`), `in_scale_mask = round(in_scale/in_scale.min())`(=PTF 채널 배율, 185), 정수 도메인에서 mean/std 계산(189-191), affine 파라미터 A·x_q+B를 `M,N`(고정소수점)으로 정수 연산 (196-202), 출력 scale로 복원. → **부동소수 LayerNorm을 정수 MAC + 시프트로 대체**.

**`QIntSoftmax` (209-298) — LIS (Log-Int-Softmax)**
- `int_softmax` (245-277): I-BERT식 정수 지수.
  - `int_polynomial` (248-258): `exp`를 2차 다항 `0.358x^2+0.970x+1`로 근사, 계수를 scale로 정수화.
  - `int_exp` (260-270): 입력을 `-ln2` 단위로 분해(`q,r`), `exp(r)`은 다항으로, `2^(n-q)` 시프트로 복원 → **정수 전용 지수**.
- `log_round` (237-243): log2 라운딩(반올림 보정).
- `forward` (279-298): `log_i_softmax`면 정수 exp → `round(sum/exp)` → `log_round` → clamp → `2^(-qlog)` 복원, 범위 밖 0 (280-288). 즉 **softmax = 정수 지수(i-exp) + 로그 양자화(Log2)**. → attention map을 4-bit 정수로, 부동소수 지수 연산 없이.

## 4. 알고리즘 / 수식

**균일 비대칭 양자화**: `q = clamp(round(x/s + z), q_min, q_max)`, `x̂ = (q - z)·s`. (uniform.py:27-40)

**PTF (LayerNorm)**: base scale `s` + 채널별 `α_c = 2^{k_c}` → 채널 c의 유효 scale `= s·α_c`, `k_c = argmin_{k∈{0,1,2,3}} ‖x_c - Q_{s/2^k}(x_c)‖_2`. (ptf.py:43-65) 하드웨어: 채널별 시프트로 구현.

**고정소수점 분해 (M, N)**: 실수 `A ≈ A_sign · M · 2^{-N}`, `M∈[0,255]`, `N∈[0,31]`. LayerNorm affine을 정수 곱+시프트로. (layers.py:159-202)

**LIS (Softmax)**:
- i-exp: `exp(x) = 2^q · exp(r)`, `x = -q·ln2 + r`, `exp(r) ≈ 0.358r^2+0.970r+1` (정수 계수). (layers.py:248-270)
- Log2 양자화: `q_attn = clamp(round(-log2(p)), 0, 2^b-1)`, `p̂ = 2^{-q_attn}` (작은 값에 bin 집중). (log2.py:17-26)
- attention bit를 4-bit까지. (`README.md:123`)

**복잡도**: attention 구조는 표준 `O(N^2 d)`. 기여는 **연산 정밀도(정수화)와 비선형(LN/softmax) 정수 대체**.

## 5. 학습 / 평가 파이프라인

- **재학습 없음(PTQ)**. ImageNet(train/val) 준비. (`README.md:76-85`)
- 명령(`README.md:90-105`):
  `python test_quant.py deit_small <DATA_DIR> --quant --ptf --lis --quant-method minmax`
  - 모델: `deit_{tiny,small,base}`, `vit_{base,large}`, `swin_{tiny,small,base}`.
  - `--ptf`(PTF LayerNorm), `--lis`(Log-Int-Softmax), `--quant-method ∈ {minmax, ema, percentile, omse}`(활성 observer).
- calibrate(observer 통계 수집) → last_calibrate(scale 확정) → quant 평가. (layers.py 패턴)

## 6. 의존성

- `python=3.7`, `pytorch=1.7.1`, `torchvision`, `cudatoolkit=10.1`. (`README.md:66-73`) `numpy`, `torch`. 외부 백본 정의는 timm류 — **외부 의존성**.

## 7. 강점 / 한계 / 리스크

**강점**
- **최초의 fully quantized ViT** — LayerNorm/Softmax까지 정수화 → **부동소수 유닛 없는 가속기**에 직접 적합.
- **PTF = 채널별 2^k 시프트** → channel-wise 정밀도를 시프트만으로 → 하드웨어 저비용.
- **LIS = i-exp + Log2** → 지수 연산 없이 정수만으로 softmax, attention map 4-bit.
- Observer/Quantizer 분리로 확장성·가독성 우수.

**한계 / 리스크**
- i-exp 다항 근사·log_round 등은 정확도-비트폭 trade-off 존재(작은 값 표현 한계, log2.py의 mask=0 처리).
- PTF의 power-of-two 제약은 표현력이 임의 scale보다 떨어질 수 있음(4단계만).
- fake-quant 시뮬레이션 중심 — 실제 정수 추론 커널/누산기 비트폭은 별도 구현 필요(확인: 정수 GEMM 커널 부재).
- Swin/ViT 변형별 모델 코드 수정 필요(레이어 교체 침투적).

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + HG-PIPE + XR 시선추적) 관점 시사점

> 전제: 본 연구는 ViT FPGA 가속기 + XR 시선추적으로 **추정**.

- **본 repo는 7개 중 FPGA 적합도가 가장 높은 후보 중 하나**(LN/Softmax 정수화 = 부동소수 회로 제거).
- **LIS(Log-Int-Softmax) → 우리 softmax 데이터패스의 직접 청사진**:
  - i-exp(2차 다항 + 2^q 시프트)는 **LUT 없이 곱셈/덧셈/시프트만으로 지수** → HG-PIPE 파이프라인에 비차단(non-blocking) softmax 유닛으로 삽입 가능. 시선추적의 저지연 요구에 부합.
  - Log2 attention(4-bit) → **attention map 저장량 2배 절감**(BRAM 절약) + dequant가 `2^{-q}`(시프트) → 곱셈 불필요. 우리 score 버퍼를 4-bit로 운용.
- **PTF → 우리 LayerNorm/스케일 회로의 핵심 기법**:
  - 채널별 scale을 임의 실수 대신 **2^k 시프트**로 → DSP 곱셈기 대신 배럴 시프터로 dequant. 채널별 k값만 작은 ROM에 저장 → **DSP/BRAM 동시 절감**.
  - `get_MN`의 고정소수점 분해(M·2^{-N})는 우리가 LN/affine을 **8-bit 정수 곱 + 시프트**로 RTL 구현할 때 그대로 채택 가능한 정량화 레시피.
- **inter-channel variation 인사이트**: ViT는 채널 편차가 크다는 관찰 → 우리 가속기에서 **per-channel(또는 per-channel-PTF) dequant**가 정확도에 중요함을 시사. layer-wise 단일 scale은 ViT-B/L에서 붕괴(README 표) → 시선추적 ViT도 채널 단위 보정 필요(추정).
- **결합 전략**: FQ-ViT(연산 정밀도/비선형 정수화) + PTQ4ViT(분포 인지 scale 탐색) + Castling(선형 attention) 3자 결합이 우리 가속기의 이상적 조합 — 비선형 제거 + 비트폭 저감 + 연산량 선형화.
- **주의**: 본 repo는 PyTorch fake-quant 검증 수준. 실제 정수 누산기 폭(LN의 `(x_q^2).sum`은 큰 비트 필요, layers.py:191)·오버플로는 우리 RTL에서 별도 설계 필요(확인).

## 9. 근거 표기

- **확인**: README.md(전수), layers.py(299L, QIntLayerNorm 151-206 / QIntSoftmax 209-298 직접 분석), uniform.py/log2.py/base(quantizer)/base(observer)/ptf.py/bit_type.py 전수. 라인 인용.
- **추정**: 우리 FPGA/XR 시사점(시프트 dequant, 4-bit score 버퍼, non-blocking softmax 유닛 등)은 코드 구조 기반 설계 추론.
- **확인 불가**: 실제 정수 추론 커널/누산기 비트폭(미구현), models/{vit,swin}_quant.py·layers_quant.py 세부(미독, 모델 조립부로 추정), config.py 세부.
