# AnyPackingNet-main (DeepBurning-MixQ) 코드베이스 정밀 분석

> 분석 대상: `REF/CNN-Accel/AnyPackingNet-main/`
> 분석 방식: 핵심 자체 소스 Read 후 (파일:라인) 근거 인용. 추론은 "추정", 확인 불가는 "확인 불가"로 명시.

---

## 1. 개요 (목적 / 원논문 / 타깃보드)

- **실제 프로젝트 정체성**: 디렉토리 이름은 `AnyPackingNet`이지만, `readme.md:1`의 제목은 **DeepBurning-MixQ**이다. 즉 디렉토리명과 프로젝트명이 불일치한다(원본 GitHub 명과 클론 폴더명이 다른 것으로 추정).
- **목적**: FPGA용 저비트 mixed-precision NN 가속기를 위한 **SW/HW 공동 최적화** 프레임워크. 두 축으로 구성된다(`readme.md:3`):
  1. **HW 축**: 다양한 저비트 conv 연산자의 **DSP 패킹**으로 단일 DSP가 가능한 한 많은 저비트 연산을 수용 → DSP 이용률 향상.
  2. **모델 축**: **differentiable NAS**로 모델별 mixed-precision 양자화 비트폭을 탐색하되, 양자화 모델의 HW 구현 효율(DSP/BRAM)까지 고려.
- **원논문(확인됨)**: ICCAD'23. "DeepBurning-MixQ: An Open Source Mixed-Precision Neural Network Accelerator Design Framework for FPGAs" (DOI 10.1109/ICCAD57390.2023.10323831) — `readme.md:5-7`. 소속: Institute of Computing Technology, Chinese Academy of Sciences (`readme.md:3`).
- **이 repo의 범위**: 학습 / 양자화 / HW-aware NAS / 가중치 export까지. **HLS 연산자 자체는 별도 repo** `MixQ_Gen_Accel`에 있다고 명시(`readme.md:12`). 즉 이 repo는 "**모델·양자화·export 파이프라인**"이며 HLS 커널 C++ 소스는 포함하지 않는다(확인됨, export가 `.h`/`.hpp`만 생성).
- **타깃 보드**: DAC-SDC 객체검출 예제는 **Ultra96 v2 FPGA** (`readme.md:57`). CIFAR 분류 예제는 보드 무관(C-sim/HLS export 대상).
- **베이스 모델**: UltraNet, UltraNet_iSmart, UltraNet_Bypass(SJTU), SkyNet(MobileNet 류), SkyNetk5 (`readme.md:59-64`). NAS 비트폭 탐색의 백본으로 사용.

---

## 2. 디렉토리 구조 (자체 + 제외 이유)

### 자체 핵심 (정밀 분석 대상)
```
anypacking/
  dsp_packing.py     # DSP 패킹 효율 테이블(factor) — 커널 크기별 (wbit,abit)→DSP packing factor
  quant_module.py    # ★핵심: 양자화 함수/모듈 + mixed-precision NAS supernet 빌딩블록 + 복잡도(DSP/BRAM) loss
cifar/               # CIFAR-10 분류 예제 (VGG_tiny 계열)
  models.py          # VGG_tiny / VGG_tiny_MixQ(supernet) / VGG_tiny_FixQ(고정비트)
  search_train.py    # differentiable NAS 학습 (weight + alpha 이중 옵티마이저)
  main_train.py      # 탐색된 비트폭으로 고정-비트 재학습
  export_hls.py      # .pt → config.h + weights.hpp (PE/SIMD 패킹, BN 흡수)
  simulate_hw.py     # 정수 도메인 HW 동작 비트-정확 시뮬레이션
dacsdc/              # DAC-SDC 객체검출 예제 (UltraNet/SkyNet 계열 + YOLO)
  mymodel.py         # UltraNet/UltraNetBypass/SkyNet/SkyNetk5 의 Float/MixQ/FixQ 변형
  quant_dorefa.py    # 구형 DoReFa 양자화(베이스 모델 호환용)
  export_hls.py      # UltraNet용 HLS export
  export_hls_skynet.py # SkyNet용(dwconv+pwconv, PEP/ACTP 추가 패킹축)
  search_train.py    # YOLO loss + complexity/bram decay NAS
  pareto_train.py    # 여러 complexity decay 값 batch 실행 → Pareto front
  simulate_hw.py     # HW 시뮬
  yolo_utils.py, datasets.py, test.py  # YOLOv3 학습 프레임워크 차용물
utils/               # torch_utils, view_pt(체크포인트 선택)
```

### 제외 (이름만 언급)
- `weights/*.pt`, `hls/<name>/weights.hpp`, `hls/<name>/config.h` — **생성물**(export 산출). 거대 가중치 배열.
- `dacsdc/yolo_utils.py`, `datasets.py`, `train_old.py` — ultralytics yolov3 학습 프레임워크 차용(third-party, AGPL; `readme.md:140`). NAS 흐름 이해에 필요한 `search_train.py`만 분석.
- `cifar/data/` (CIFAR 데이터셋), `__pycache__` — 데이터/캐시.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 DSP 패킹 테이블 — `dsp_packing.py`

- 파일 전체가 3개의 2D 리스트: `factors_k11`(1x1), `factors_k33`(3x3), `factors_k55`(5x5) (`dsp_packing.py:1,11,21`).
- 각 테이블은 **7x7 행렬**으로, 인덱스 `[wbit-2][abit-2]` (즉 wbit/abit ∈ {2..8})에 대응한다. 이 인덱싱은 `quant_module.py:400` `factors[wbits[i]-2][abits[j]-2]`에서 확인된다.
- 값의 의미: **한 개 DSP(또는 DSP 그룹)가 처리할 수 있는 (wbit,abit) 곱셈 연산 수**(packing factor). 비트폭이 작을수록 값이 큼(예: k3x3, w2/a2 → `18`; w8/a8 → `2` — `dsp_packing.py:12,18`). 즉 저비트일수록 한 DSP에 더 많은 MAC을 패킹.
- 소수값 존재(예: `7.5`, `6.67`, `4.5`, `3.33`, `2.25` — `dsp_packing.py:12,22`): packing이 DSP를 분수적으로 공유함을 모델링(추정: 여러 출력채널이 한 DSP 사이클을 나눠 쓰는 평균 효율).
- **DSP 비용 추정식**: `dsps = size_product / factor[wbit-2][abit-2]` (`quant_module.py:475`). size_product는 layer MAC 규모(아래 3.4). factor가 클수록(저비트) 필요 DSP 감소.

### 3.2 양자화 primitive — `quant_module.py`

두 가지 step 테이블(`quant_module.py:11-12`):
- `gaussian_steps` (weight용): bit→step (1bit:1.596 … 8bit:0.032). 가중치가 가우시안 분포라 가정한 최적 양자화 스텝.
- `hwgq_steps` (activation용): HWGQ(half-wave Gaussian quantizer) step.

**가중치 양자화 autograd 함수들**:
- `_gauss_quantize.forward` (`:46-54`): `step *= x.std()` 로 표준편차 스케일 → `round(x/step)` clamp `[-lvls, lvls-1]` (`lvls=2^bit/2`, 즉 부호있는 대칭). backward는 STE(straight-through, grad 그대로 통과 — `:57`).
- `_gauss_quantize_resclaed_step` (`:67-77`): std 곱셈 없이 외부에서 미리 스케일된 step을 받는 버전. mixed-precision 분기에서 std를 1회만 계산해 재사용하기 위함(`:307`).
- `_gauss_quantize_export` (`:60-65`): export용. 스케일 곱(`*step`) 없이 **정수 인덱스**와 step을 함께 반환(`:65`) → HLS 가중치 배열로 직접 사용.

**활성 양자화**:
- `HWGQ` (`:91-107`): `bit>=32`면 ReLU(`clamp(min=0)`), 아니면 `[0, step*(2^bit-1)]` clamp 후 `_hwgq`(round-to-step). 양수 단방향(half-wave) 양자화.
- `ImageInputQ` (`:109-122`): 입력 이미지를 `[0..255]/256` 이산값으로 가정, `floor(x/step)*step`, step=1/2^bit. **gradient 없음**(`:121` 주석). 첫 레이어 입력 전용.

**고정-비트 conv/linear** (재학습 후 사용):
- `QuantConv2d(nn.Conv2d)` (`:124-144`): forward에서 `_gauss_quantize`로 weight 양자화 후 `F.conv2d`. `export_quant()`이 정수+step 반환(`:143`).
- `QuantActivConv2d` (`:167-207`): `[ActQ] → QuantConv2d` 묶음 + **복잡도 버퍼**(`size_product`, `memory_size`, `in_width`)를 forward마다 갱신(`:194-201`). `param_size = inplane*outplane*ksize/groups *1e-6` (`:187`), `filter_size = param_size/stride^2` (`:188`).

### 3.3 Mixed-Precision NAS supernet 블록 (★)

- `MixQuantActiv` (`:235-252`): 후보 비트들 각각의 `HWGQ` 분기를 `ModuleList`로 두고, **softmax(alpha_activ) 가중합**으로 출력 (`:248-251`). alpha 초기값 0.01 (`:241`). 이것이 differentiable bit search의 핵심 — 비트 선택을 연속 완화.
- `MixQuantConv2d` (`:255-284`) / `SharedMixQuantConv2d` (`:287-316`): 후보 비트별로 가중치를 양자화→softmax(alpha_weight) 가중합한 **단일 합성 weight**로 conv 1회 수행 (`:280-283`).
  - **share_weight 차이**: `MixQuantConv2d`는 비트마다 별도 `nn.Conv2d`(weight 분리), `SharedMixQuantConv2d`는 conv 1개 weight를 모든 비트가 **공유**하고 std 계산을 1회만(`:307`) → 메모리/연산 절약. supernet 기본은 share=True (`models.py:114`, `mymodel.py:320` qspace).
- `MixActivConv2d` (`:319-487`): 위 둘을 결합한 **탐색 가능 conv 레이어**. wbits/abits 기본 `[1,2]`(`:324,328`)지만 실제 모델은 `[2,3,4,5,6,7,8]` 사용(`models.py:121`).

**복잡도 loss (HW-aware NAS의 정수)** — `MixActivConv2d` 내부:
- `complexity_loss_trivial` (`:369-381`): 단순 `size_product * mix_abit * mix_wbit` (bitops 비례). HW 무관.
- `complexity_loss` (`:383-402`): **DSP-packing 인식**. 모든 (wbit_i, abit_j) 쌍에 대해 `sw[i]*sa[j] / factors[wbit_i-2][abit_j-2]` 합산 → `size_product*64*mix_scale` (`:398-401`). factor가 큰(저비트) 조합일수록 loss 작아짐 → NAS가 DSP-효율 비트로 수렴하도록 유도. **이것이 "AnyPacking"의 본질** — 패킹 효율을 미분가능 loss로 직접 최적화.
- `bram_loss` (`:404-426`): weight BRAM(`param_size*mix_wbit`) + activation 캐시 BRAM(sliding-window: 1x1은 `2*in_width*inplane`, else `(k+1)*in_width*inplane`)을 mix-bit로 추정(`:411-425`). UltraNet pipeline 설계의 BRAM 제약 반영.

**아키텍처 추출** — `fetch_best_arch` (`:428-487`): softmax argmax로 레이어별 최적 비트 선택(`:433,440`), 그리고 best/expected(mix)에 대한 bitops/bita/bitw/**dsps**/**mixdsps**/bram을 모두 계산해 반환. dsps는 best-arch factor(`:475`), mixdsps는 확률가중 합(`:480-483`).

### 3.4 복잡도 버퍼 정의 (size_product / memory_size)

- `size_product = filter_size * in_H * in_W` (`:197-198`) → 레이어 총 MAC 규모(M 단위). DSP/bitops 추정의 기준량.
- `memory_size = in_C*in_H*in_W *1e-3` (`:195-196`) → activation 메모리(K 단위), bita 계산용.
- `in_width` (`:200`) → BRAM sliding-window 폭 계산용.

### 3.5 모델 정의 — `cifar/models.py`, `dacsdc/mymodel.py`

- **3종 변형 패턴**(전 모델 공통): `*Float`(부동소수 기준), `*_MixQ`(supernet, NAS 탐색용), `*_FixQ`(탐색된 비트폭으로 고정 재학습). 예: `VGG_tiny_MixQ`(`models.py:113`), `VGG_tiny_FixQ`(`models.py:201`).
- `VGG_tiny_MixQ`: 6개 `MixActivConv2d`(모두 3x3, qspace 2~8bit) + BN + maxpool, 마지막 `QuantActivLinear`는 8/8 고정(`models.py:146`). 첫 레이어만 `ActQ=ImageInputQ`(`models.py:124`).
- `UltraNet_MixQ`(`mymodel.py:313`): UltraNet(VGG류) 9-conv. `UltraNet_ismart`/`UltraNet`은 DoReFa 4bit 고정(`mymodel.py:169-170,244-245`) — 베이스 비교군.
- `UltraNetBypass_*`(`mymodel.py:547,623,704`): `ReorgLayer`(`:526-545`, stride2 space-to-depth)로 bypass branch를 reorg 후 concat(`:609`). bypass가 정확도↑이나 pipeline 가속기 설계 난이도↑(`readme.md:62`).
- `SkyNet_MixQ`/`SkyNetk5_MixQ`(`mymodel.py:840,896`): **depthwise(groups=inp) + pointwise(1x1)** 반복(`:854-863`), MobileNet류. k5는 5x5 dwconv(`:911`). dwconv가 연산 적어 큰 커널이 저비용 정확도↑(`readme.md:64`).
- `YOLOLayer`(`mymodel.py:25-58`): anchor 6개, no=6(=x,y,w,h,obj + 1?), inference시 sigmoid/exp 디코딩.

### 3.6 학습 흐름 — `search_train.py`

- **이중 옵티마이저**: 일반 파라미터(weight)는 `optimizer`(SGD), `alpha`(비트선택 logit)는 별도 `arch_optimizer`로 분리 (`cifar/search_train.py:43-50`). 이름에 'alpha' 포함 여부로 분류(`:44-46`). EdMIPS 방식(`readme.md:121`).
- **loss = task loss + complexity_decay·complexity_loss + complexity_decay_trivial·trivial** (`cifar/search_train.py:78-81`). `--cd`(complexity decay)가 클수록 저비트/저DSP로 압박.
- dacsdc 버전(`dacsdc/search_train.py:241-252`)은 추가로 `--bram-decay`(`bram_loss`)까지 합산 — BRAM 제약 NAS.
- 에폭마다 `fetch_best_arch()`로 현재 best 비트열 출력(`cifar/search_train.py:93-101`), `bestw/besta` 문자열을 체크포인트 extra에 저장(`:121`).
- `pareto_train.py`(`:4-17`): complexity decay 값 리스트(`cd`/`cdt`)를 순회하며 `search_train.py`/`main_train.py`를 `os.system`으로 반복 실행 → **Pareto front 스윕**.

### 3.7 HLS Export 파이프라인 — `cifar/export_hls.py` (★ HW 매핑의 핵심)

export는 `.pt` → `config.h` + `weights.hpp` 생성. 4단계:

1. **`extract_model`** (`:58-154`): `model.modules()`를 **상태머신**으로 순회([QAct]→[Pool]→Conv→[BN]→[Pool], `:64`). 각 conv의 k/s/p/ich/och/irow/icol/orow/ocol, abit/astep(ActQ에서), wbit/정수weight/wstep(`export_quant()`)을 수집. feature map shape를 전파 계산(`:92-96`). 이전 레이어 `obit/ostep`은 다음 레이어 abit/astep으로 설정(`:75-78`) — **레이어 간 양자화 스케일 체이닝**.

2. **`process_batchnorm`** (`:156-210`): **BN을 정수 inc/bias로 흡수**(가속기에서 BN을 별도 연산 안 하고 conv 누산기에 정수 곱/덧셈으로 처리). 수식 docstring(`:164-178`):
   - `out = MAC*BN_w + BN_b`, `outq = MACq*inc_raw + bias_raw`, `inc_raw = BN_w*MACstep/ostep`, `bias_raw = BN_b/ostep` (`:188-191`).
   - 정수화: `T = lshift + wbit + abit - 1` (lshift=16, `:181,195`), `inc = round(inc_raw*2^T)`, `bias = round(bias_raw*2^T)` (`:196-197`). HLS에서 `(MACq*inc + bias + 2^(T-1)) >> T`로 복원(`:170-172`).
   - `incbit/biasbit` = 비트길이(`:200-203`). **제약**: `MBIT+incbit<48`(DSP 누산폭, `:178`).
   - 마지막 레이어는 inc 없이 `div=1/(wstep*astep)`로 bias만(`:205-210`).

3. **`reorder_weight`** (`:212-258`): **PE/SIMD 병렬화에 맞춘 가중치 재배치**.
   - conv: `[och,ich,kr,kc]` → transpose `[och,kc,kr,ich]`(`:248`) → reshape `[och/pe, pe, k, k*ich/simd, simd]` → transpose `[pe, k, och/pe, k*ich/simd, simd]` → `[pe, k, -, simd]`(`:250-252`). 즉 **PE 차원(출력채널 병렬)과 SIMD 차원(입력채널×커널 병렬)을 분리**해 HLS 배열 인덱스로 매핑. k=1은 추가 reshape(`:254-255`).
   - PE/SIMD 값은 `hls/config_simd_pe.txt`에서 로드(`:384`) — 레이어별 수동 지정.
   - linear는 별도 reorder(`:218-228`, `[OUT/2PE, PE, 2, IN/IN_PE, IN_PE/SIMD, SIMD, H, W]` 다축).

4. **`write_hls_config` / `write_hls_weights`** (`:17-56`, `:305-366`):
   - config.h: name_mapping(`:18-35`)으로 `CONV_n_IFM_CH`, `..._W_BIT`, `..._SIMD`, `..._PE`, `..._L_SHIFT` 등 `#define` 생성.
   - weights.hpp: SIMD개 가중치를 **하나의 `ap_uint<wbit*simd>`로 패킹**(`:341,343`). `pack1d_str`(`:345-351`)가 **역순(reverse) 패킹**(`arr[::-1]`, `:347` — HLS 구현 unpack 순서와 연동). inc/bias도 `ap_int<incbit>[pe][och/pe]` 배열로 출력(`:357-362`).
   - **special_wa_bit 보정**(`adjust_weight`, `:368-374`): 특정 (wbit,abit) 패킹 조합(예 (4,2),(5,3)...,(7,3))은 `-2^(wbit-1)` 값을 표현 못 함 → `max(w, -2^(wbit-1)+1)`로 클립. **DSP 패킹 시 부호 최소값 충돌 회피** — 패킹 구현의 실질 제약을 드러냄.

- dacsdc export(`dacsdc/export_hls.py`)는 거의 동일하나 DoReFa(`Conv2d_Q`) 가중치도 처리(`:110-119`), maxpool 흡수(`max_pool=True`, `:142`), 첫 레이어 입력 8bit 기본(`:144-147`).
- **SkyNet export**(`export_hls_skynet.py`)는 dwconv/pwconv용 **추가 패킹축 PEP/ACTP** 도입(`:209-255`): 1x1 pwconv를 `[och/(pe*pep), pe, pep, ich/simd, simd]`로 재배치(`:234-237`), `ap_uint<wbit*pep*simd>` 패킹(`:301`). inc/bias는 actp 단위(`:317`). depthwise/pointwise 분리 가속을 위한 다중 병렬축.

### 3.8 비트-정확 HW 시뮬레이션 — `cifar/simulate_hw.py`

- `QConvLayer`(`:11-58`): export된 정수 weight/inc/bias로 **정수 도메인** conv 재현. `F.conv2d(int64)`(`:28`), `x*inc + bias`(`:35-39`), `x += 2^(T-1); x >>= T`(`:42-44`) — export의 BN 정수화와 **동일한 round/shift**. 출력 clip `[0, 2^obit-1]`(`:56-57`).
- `HWModel`(`:60-75`): 첫 레이어 입력을 `x >> (8-abit)`로 양자화(`:68-69`), 마지막 div로 float 복원(`:74`). HW 출력 == HLS C-sim 출력 검증용(`readme.md:48-49`).
- maxpool 순서 주의 주석(`:17`): BN.inc가 음수일 때 maxpool/BN 순서 중요.

---

## 4. 데이터플로우

### 학습/탐색 시간 (오프라인, GPU)
```
search_train.py: supernet(MixQ) → [task loss + cd·complexity_loss(DSP-packing) + bd·bram_loss]
   → weight optimizer + alpha arch_optimizer 동시 갱신 → fetch_best_arch() → bestw/besta 비트열
main_train.py: FixQ(고정 비트열) 재학습 → weights/*.pt
pareto_train.py: cd 스윕 → Pareto(정확도 vs DSP/BRAM)
```

### Export → 배포 (오프라인 CPU)
```
export_hls.py: .pt → extract_model(state machine) → adjust_weight(special pack 보정)
   → process_batchnorm(BN→정수 inc/bias, lshift=16) → reorder_weight(PE/SIMD 패킹)
   → config.h(#define) + weights.hpp(ap_uint<wbit*simd> 패킹)  → model_param.pkl
simulate_hw.py: model_param.pkl → 정수 비트-정확 추론(HLS C-sim 등가 검증)
```
HLS 커널 합성/실행은 외부 repo(MixQ_Gen_Accel)에서(`readme.md:12`). 이 repo 단독으로는 합성 불가(확인됨).

### 활성 스케일 체이닝
레이어 i의 출력 step(`ostep`) = 레이어 i+1의 입력 step(`astep`) (`export_hls.py:75-81`). conv 누산기는 `MACstep=wstep*astep` 도메인, BN inc/bias가 `MACstep/ostep`로 다음 레이어 도메인으로 정규화.

---

## 5. HW/SW 매핑

| 모델/알고리즘 개념 | HW(HLS) 매핑 | 근거 |
|---|---|---|
| 출력채널 병렬 | **PE** 차원 | `export_hls.py:250-252`, config `_PE` |
| 입력채널×커널 병렬 | **SIMD** 차원, `ap_uint<wbit*simd>` 1워드 패킹 | `export_hls.py:341` |
| 저비트 MAC을 1 DSP에 다중 수용 | **DSP packing factor** 테이블 | `dsp_packing.py`, `quant_module.py:400` |
| BatchNorm | conv 누산기 정수 `inc/bias` + shift(>>T) | `export_hls.py:188-203` |
| activation(LeakyReLU/HWGQ) | export 시 obit clip, 시뮬 clip `[0,2^obit-1]` | `simulate_hw.py:56-57` |
| depthwise/pointwise(SkyNet) | 추가 PEP/ACTP 패킹축 | `export_hls_skynet.py:234-237` |
| bypass/reorg | space-to-depth ReorgLayer | `mymodel.py:526-545` |
| BRAM 예산 | bram_loss(weight+sliding-window 캐시) | `quant_module.py:404-426` |

**핵심**: NAS가 비트폭을 정하면 → DSP packing factor가 그 비트폭의 DSP 비용을 결정 → complexity_loss로 다시 NAS에 피드백. **모델 탐색과 HW 비용이 폐루프**.

---

## 6. 빌드 · 실행

CIFAR (`readme.md:17-50`):
```
python search_train.py --cd 3e-5 --name mix_vggtiny_cifar_cd3e5   # NAS
python main_train.py --bitw 822222 --bita 833363 --name vggtiny_cifar_cd3e5  # 고정비트 재학습
python test_acc.py
vim hls/config_simd_pe.txt        # 레이어별 SIMD/PE 수동 설정
python export_hls.py              # config.h + weights.hpp
python simulate_hls.py            # 비트-정확 검증(=HLS C-sim)
```
DAC-SDC (`readme.md:70-117`): `search_train.py --cd 1e-5 --model [UltraNet_MixQ|SkyNet_MixQ|...]` → `main_train.py --bitw <> --bita <>` → `export_hls.py [--model SkyNetk5_FixQ]`.

> 주의: readme는 `simulate_hls.py`/`export_hls.py`로 표기하나 실제 파일은 `simulate_hw.py`(확인됨). 문서-코드 파일명 불일치.

---

## 7. 의존성

- PyTorch (nn, autograd.Function, F.conv2d/linear), torchvision(CIFAR), numpy, tqdm.
- DAC-SDC 예제: ultralytics yolov3 학습 프레임워크 차용(yolo_utils/datasets/test, AGPL — `readme.md:140`).
- export 출력은 `ap_int.h`(Xilinx HLS) 헤더 전제(`export_hls.py:320`). HLS 합성은 외부 repo 필요.
- 참고 baseline: EdMIPS(differentiable MP search), UltraNet/SkyNet/iSmart(DAC-SDC 수상작) — `readme.md:120-132`.

---

## 8. 강점 · 한계

**강점**
- DSP packing 효율을 **미분가능 NAS loss로 직접 통합**(`quant_module.py:383-402`) — 단순 bitops가 아닌 실제 HW 자원으로 탐색.
- BRAM까지 loss화(`bram_loss`) — pipeline 아키텍처의 실 제약 반영.
- 학습→양자화→BN흡수→PE/SIMD패킹→비트정확시뮬이 **하나의 파이프라인**으로 일관(스케일 체이닝, special-pack 보정까지).
- 모델별 3변형(Float/MixQ/FixQ)으로 탐색-재학습-배포 분리가 깔끔.

**한계**
- HLS 커널 C++가 이 repo에 없음(외부 repo 의존, `readme.md:12`) → 단독으로 합성/온보드 실행 불가.
- PE/SIMD가 **수동 지정**(`config_simd_pe.txt`) — 자동 자원배분 아님.
- DSP packing factor 테이블이 **고정 상수**(`dsp_packing.py`) — 특정 DSP 패킹 구현/타깃 가정. 다른 보드/패킹전략엔 재캘리브레이션 필요.
- "fully pipelined across FPGA"라 소형 모델/레이어 한정(`readme.md:10`); 범용 multi-core는 "곧"(미구현, 확인됨).
- 문서-코드 파일명/프로젝트명 불일치(AnyPackingNet vs DeepBurning-MixQ; simulate_hls vs simulate_hw).
- C-RTL co-sim 미구현 영역은 외부 repo 책임.

---

## 9. 우리 프로젝트(ViT/Transformer FPGA + XR 시선추적) 시사점

- **DSP packing의 ViT 전이(추정)**: ViT의 Q/K/V projection·FFN GEMM도 저비트(INT4/INT8) MAC가 많다. `dsp_packing.py`식 (wbit,abit)→packing-factor 테이블을 **GEMM tile**에 적용하면 systolic/dataflow PE에서 DSP 이용률을 끌어올릴 수 있다. 특히 INT4 attention에 유효(추정).
- **HW-aware mixed-precision NAS**: `complexity_loss`(`quant_module.py:383-402`)를 ViT의 layer-wise/head-wise 비트탐색에 차용 가능. attention(민감)과 FFN(둔감)의 비트를 DSP 예산 하에서 자동 배분 → HG-PIPE 계열 파이프라인 자원 최적화에 직결.
- **BN 정수 흡수 → LayerNorm/스케일 흡수(주의)**: `process_batchnorm`의 `inc/bias + >>T` 정수화 기법은 CNN BN 전용. ViT는 LayerNorm(채널별 통계, 동적)이라 **정적 흡수가 불가** — conv-BN fold 기법을 그대로 가져오면 안 됨(확인 가능한 차이). 대신 quant scale 체이닝(`export_hls.py:75-81`) 아이디어(레이어 간 step 전파)는 transformer 양자화 스케일 관리에 재사용 가능.
- **PE/SIMD 가중치 재배치**: `reorder_weight`의 `[pe, k, och/pe, ..., simd]` 패킹·`ap_uint<wbit*simd>` 1워드 패킹은 우리 weight loader(HGPIPE)에 직접 참고. 단 ViT는 커널 차원(k) 없이 GEMM이므로 `[PE, OUT/PE, IN/SIMD, SIMD]`로 단순화(추정).
- **special-pack 부호 보정**(`export_hls.py:368-374`)은 우리가 INT 패킹 구현 시 동일 함정(최소 음수값 표현 불가)을 만날 수 있음 — 양자화 export 단계 체크리스트로 채택 권장.
- **Pareto 스윕 워크플로**(`pareto_train.py`): XR 시선추적은 latency/power 예산이 빡빡하므로, complexity decay 스윕으로 정확도-DSP/BRAM Pareto를 뽑아 보드 예산에 맞는 비트구성을 고르는 절차가 그대로 유용.
- **비트-정확 시뮬**(`simulate_hw.py`)은 HLS 합성 전 디버깅 비용을 크게 줄이는 패턴 — 우리 transformer 가속기 검증에도 "PyTorch 정수 모델 == HLS C-sim" 골든 비교로 도입 권장.
