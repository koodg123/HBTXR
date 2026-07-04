# ShiftAddViT 코드베이스 정밀 분석

> 분석 대상: `REF/ViT-Quantization/ShiftAddViT`
> 분석 일자: 2026-06-20
> 근거 표기 규칙: 코드로 확인한 사실은 `(파일:라인)`으로 표기. 코드로 직접 확인 못한 항목은 "확인 불가", 추론은 "추정"으로 명시.

---

## 1. 개요 (목적 / 원논문 / 핵심 아이디어)

- **원논문**: *ShiftAddViT: Mixture of Multiplication Primitives Towards Efficient Vision Transformer*, Haoran You, Huihong Shi, Yipin Guo, Yingyan Lin, **NeurIPS 2023** (arXiv:2306.06446) (`README.md:1-8`).
- **목적**: Vision Transformer(ViT)의 핵심 연산인 **곱셈(MAC)을 비싼 부동소수점 곱셈 대신 시프트(shift)·덧셈(add) 등 "곱셈 기본 연산(multiplication primitives)"의 혼합으로 치환**하여 GPU/엣지 디바이스에서 추론 효율(latency/energy)을 높이는 것.
- **핵심 아이디어 (코드 기반 확인)**:
  1. **선형 계층(Linear) → Shift 계층**: 가중치를 `2^shift × sign` (2의 거듭제곱 × 부호) 형태로 표현하여, 곱셈을 비트 시프트로 대체 (`pvt/deepshift/modules.py:40,177`). 이것이 "Add" 계열 MLP/Attention의 곱셈 primitive.
  2. **Attention 행렬곱 → MatAdd 치환**: linear-attention 계열에서 `Q@K`, `attn@V`를 `MatAdd` 모듈로 추상화하여 add 연산으로 매핑 (`pvt/matkernel.py:6-11`, `pvt/pvt_v2.py:339-340,404,413`).
  3. **MoE(Mixture-of-Experts) 게이팅**: MLP를 "정규 곱셈 MLP"와 "Shift MLP" 두 expert로 구성, 게이트가 토큰별로 선택 (`pvt/fmoe_mlp.py:285-291`). 이것이 논문 제목의 "Mixture of Multiplication Primitives".
  4. **TVM 기반 커스텀 커널**: shift/add 연산을 TVM Ansor로 자동 튜닝하여 실측 throughput 비교 (`OPs_Speedups/Shift/latency_test.py`, `OPs_Speedups/README.md`).
- **백본**: PVT(Pyramid Vision Transformer) v1/v2 계열 (`pvt/pvt.py`, `pvt/pvt_v2.py`). DeepShift 라이브러리(GATECH-EIC)의 shift 모듈을 ViT에 이식한 구조.

---

## 2. 디렉토리 구조 (자체 소스 + 제외 표기)

```
ShiftAddViT/
├── README.md                         # 논문/사용법 (분석함)
├── OPs_Speedups/                     # TVM shift/add 커널 단위 테스트
│   ├── README.md                     (분석함)
│   ├── Shift/{Ansor_tune.py, latency_test.py}   (분석함: shift TVM 커널)
│   └── Add/{Ansor_tune.py, latency_test.py}     (구조 동일, add 커널)
└── pvt/                              # 메인 코드 (PVT + DeepShift 이식)
    ├── pvt_v2.py                     # ★ PVTv2 모델 + 다종 Attention (분석함)
    ├── pvt.py                        # PVTv1 모델
    ├── main.py / main_cpu.py         # 학습/평가 엔트리 (params 기반)
    ├── params.py                     # ★ argparse 전체 옵션 (분석함)
    ├── engine.py / losses.py         # train/eval 루프, loss
    ├── matkernel.py                  # ★ MatMul/MatAdd/Mul 추상화 (분석함)
    ├── fmoe_mlp.py                   # ★ Shift_Linear/Shift_Mlp + MoE MLP (분석함)
    ├── fmoe_fc.py / fmoe_new.py      # MoE FC, SparseDispatcher, Gate
    ├── performer.py / performer_new.py  # Performer/Ecoformer linear-attn
    ├── tvm_func.py / hw_utils.py / cal_energy.py  # TVM 컴파일·HW 비용
    ├── datasets.py / samplers.py / mcloader/      # ImageNet 데이터
    ├── configs/{pvt,pvt_v2}/*.py     # mmcv config (모델·attn_type 조합)
    ├── hashing/{ksh.py,...}          # Ecoformer 커널 해싱
    └── deepshift/                    # ★ Shift 양자화 핵심
        ├── modules.py               # ★ LinearShift/Conv2dShift (PS 모드) (분석함)
        ├── modules_q.py             # ★ LinearShiftQ/Conv2dShiftQ (Q 모드, APoT) (분석함)
        ├── ste.py                   # ★ STE(Straight-Through Estimator) Function들 (분석함)
        ├── utils.py                 # ★ round_to_fixed/get_shift_and_sign/APoT (분석함)
        ├── convert.py               # ★ nn.Linear→ShiftLinear 변환기 (분석함)
        └── kernels/                 # ❌ 제외: CUDA/CPU 외부 커널 (cuda/, cpu/)
```

### 제외 항목 (지시에 따라 분석 제외, 이름만 언급)
- `pvt/deepshift/kernels/cuda/`, `pvt/deepshift/kernels/cpu/` : DeepShift 외부 CUDA/CPU 커널 (compress_sign_and_shift 등 네이티브 op).
- `pvt/unoptimized/kernels/` : unoptimized CUDA 커널.
- `pvt/run_with_submitit.py`, `gpu_mem_track.py` : 인프라/프로파일 유틸 (분석 비핵심).
- 체크포인트/대용량 가중치: README상 "ToDos"로 미공개 상태 (`README.md:18-21`) — **확인 불가** (저장된 .pth 파일 미발견).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `deepshift/ste.py` — Straight-Through Estimator (양자화 미분 처리)

shift/sign 파라미터는 round/clamp/sign 등 미분 불가능 연산을 거치므로, forward는 양자화하되 backward는 gradient를 그대로 통과시키는 **STE** 패턴을 `torch.autograd.Function`으로 구현.

- `RoundPowerOf2` / `round_power_of_2()` (`ste.py:6-19`): forward는 `utils.round_power_of_2`, backward는 `grad_output` 그대로 통과 (`ste.py:11-13`).
- `RoundFixedPoint` / `round_fixed_point()` (`ste.py:21-34`): 활성값을 고정소수점(act_integer_bits/act_fraction_bits)으로 라운딩, gradient는 통과.
- `RoundFunction` / `round()` (`ste.py:36-49`): deterministic/stochastic 라운딩.
- `SignFunction` / `sign()` (`ste.py:51-64`): `torch.sign` forward, gradient 통과(STE).
- `ClampFunction`, `ClampAbsFunction` (`ste.py:66-108`): 범위 클램프, gradient 통과.
- `UnsymmetricGradMulFunction` / `unsym_grad_mul()` (`ste.py:125-140`): `2^shift × sign` 곱셈 시 **비대칭 gradient** 처리. forward `mul(input1,input2)`, backward는 `grad*input2, grad`로 sign에 대한 비대칭 미분 (`ste.py:131-134`) — shift와 sign을 독립적으로 학습시키기 위한 핵심 트릭.
- 모든 함수가 `args.tvm_tune or args.tvm_throughput`이면 autograd 우회(순수 텐서 연산)로 분기 — TVM 추론 모드에서는 STE 불필요 (`ste.py:16-19` 등 전반).

### 3.2 `deepshift/utils.py` — shift/sign 추출과 고정소수점

- `round_to_fixed(input, integer_bits=16, fraction_bits=16)` (`utils.py:7-21`): 고정소수점 양자화. `delta=2^-fraction_bits`로 floor 후 `[-2^(int-1), 2^(int-1)-1]` 클램프. **활성/바이어스를 16비트 고정소수점으로 표현** (FPGA 친화).
- `get_shift_and_sign(x)` (`utils.py:23-29`): 가중치 `x`로부터 `sign=sign(x)`, `shift=round(log2(|x|))` 추출 → `x ≈ 2^shift × sign`. **부동소수점 가중치를 2의 거듭제곱으로 근사**하는 핵심 함수.
- `round_power_of_2(x)` (`utils.py:51-54`): `(2^shift)*sign`으로 재구성.
- `get_shift_and_sign_SP2(x, weight_bits)` (`utils.py:32-48`): **Sum-of-Power-of-2 (SP2 / APoT)** — 가중치를 2개의 2-거듭제곱 항의 합 `2^shift_1 + 2^shift_2`으로 분해 (잔차 `x_diff = |x| - 2^shift_1`에 다시 log2 적용, `utils.py:43-45`). 표현력 ↑.
- `build_power_value(B=4, additive=True)` / `get_param_APoT(x)` (`utils.py:125-189`): APoT 양자화 레벨 집합 생성 및 가장 가까운 레벨로 투영(nearest projection, `utils.py:181`).
- `compress_bits(shift, sign)` (`utils.py:81-108`): shift/sign을 비트팩킹하여 CUDA 커널용 압축 가중치(`ConcWeight`) 생성. sign==0(ternary)일 때 shift=-32(=곱셈0 등가) 처리 (`utils.py:88-89`). → **외부 CUDA 커널(`deepshift.kernels.compress_sign_and_shift`) 호출, 분석 제외 대상**.
- `get_weight_from_ps(shift, sign, ...)` (`utils.py:111-118`): shift/sign 파라미터 → 실수 가중치 복원 (multi 변환용).

### 3.3 `deepshift/modules.py` — LinearShift / Conv2dShift (**PS 모드: shift·sign 직접 학습**)

ShiftAddViT의 가장 핵심. nn.Linear를 대체하는 shift 연산 계층.

- **`LinearShift` (`modules.py:79-197`)**:
  - 파라미터: 일반 모드는 `shift`, `sign` (각 `out×in` 형태, `modules.py:125-126`). SP2 모드는 `shift_1/2`, `sign_1/2`, `sign` (`modules.py:117-121`).
  - `shift_range = (-(2^(weight_bits-1)-2), 0)` — weight_bits=5면 `(-14, 0)` (`modules.py:102`). SP2면 `(-2,0)` (`modules.py:97`).
  - **forward (`modules.py:157-190`)**:
    - 일반: `shift` clamp→round, `sign`은 round 후 sign, **가중치 = `unsym_grad_mul(2^shift_rounded, sign_rounded_signed)`** (`modules.py:174-177`).
    - SP2/APoT: 두 shift 항을 결합 `weight_ps = (2^(2*sh1)*sgn1 + 2^(2*sh2-1)*sgn2)*2/3 * sgn` (`modules.py:169-171`).
    - 활성·바이어스를 `round_fixed_point`로 16비트 고정소수점화 (`modules.py:180-184`).
    - 출력: `F.linear(input_fixed_point, weight_ps, bias_fixed_point)` (kernel 미사용 시, `modules.py:190`). use_kernel 시 `LinearShiftFunction.apply` → 외부 커널.
  - **`LinearShiftFunction.forward` (`modules.py:18-47`)**: use_kernel=False 경로에서 `v = 2^shift.round() * sign.round().sign()`, `out = input.mm(v.t())` (`modules.py:40-41`). **즉 v는 2의 거듭제곱 가중치 → 곱셈이 시프트로 환원 가능**.
  - **backward (`modules.py:50-76`)**: shift gradient에 `log2` 인수 `grad_shift = grad_sign * v * log2` (`modules.py:70,72`) — `d(2^shift)/dshift = 2^shift·ln2` 의 chain rule 반영. `log2 = math.log(2)` (`modules.py:15`).

- **`Conv2dShift` / `_ConvNdShift` (`modules.py:200-446`)**: Conv2d 버전. forward에서 동일하게 `weight_ps = unsym_grad_mul(2^shift_rounded, sign)` 후 `F.conv2d` (`modules.py:417-445`). `Conv2dShiftFunction.backward`는 `torch.nn.grad.conv2d_*`로 shift/sign gradient 계산 (`modules.py:258-268`).

### 3.4 `deepshift/modules_q.py` — LinearShiftQ / Conv2dShiftQ (**Q 모드: weight를 실수로 두고 양자화**)

PS 모드와 달리 **실수 `weight`를 보관**하고, forward 시점에 2의 거듭제곱으로 양자화하는 방식 (QAT 친화).

- **`LinearShiftQ` (`modules_q.py:163-257`)**:
  - 실수 `weight` 파라미터 보관 (`modules_q.py:196`). `shift_range = (-(2^(weight_bits-1)-1), 0)` (`modules_q.py:174`).
  - **forward (`modules_q.py:215-250`)**: 일반 모드는 `clampabs`로 `[2^range_low, 2^range_high]` 클램프 후 `round_power_of_2(weight)`로 2의 거듭제곱 양자화 (`modules_q.py:224-225`). SP2면 `apot_quantization` 사용 (`modules_q.py:222`).
  - **`apot_quantization(tensor, alpha, proj_set, ...)` (`modules_q.py:124-160`)**: `data=tensor/alpha`를 `[-1,1]`(weight) 또는 `[0,1]`(act) 클램프 후, `power_quant`로 사전계산된 APoT 레벨 집합 `proj_set`에 nearest 투영 (`modules_q.py:133,141`). STE로 `(xhard-x).detach()+x` 미분 (`modules_q.py:144`). `alpha`는 학습 가능 스케일 (`modules_q.py:180`).
  - **`build_power_value(B, additive)` (`modules_q.py:71-116`)**: B=2/3/4/5/6별 additive PoT 레벨 생성. 예 B=2: `2^(-1),2^(-2),2^(-3)` (`modules_q.py:76-78`).
  - **`LinearShiftQFunction.forward` (`modules_q.py:16-44`)**: `shift,sign = get_shift_and_sign(weight)` → `weight_s=(2^shift)*sign` → `input.mm(weight_s.t())` (`modules_q.py:22,37-38`).
- **`Conv2dShiftQ` (`modules_q.py:381-431`)**: Conv 버전, 동일 패턴.

### 3.5 `deepshift/convert.py` — 모델 변환기 (nn.Linear → ShiftLinear)

- **`convert_to_shift(model, shift_depth, shift_type, ...)` (`convert.py:12-110`)**: 모델 트리를 재귀 순회하며 `nn.Linear`를 shift 계층으로 치환.
  - **선택적 치환**: attention 내 `to_qk`, `to_v`, `proj` 이름의 Linear만, 그리고 `linear_count>0` 조건에서 치환 (`convert.py:29`). 즉 **특정 attention projection만 shift로 변환** (모든 Linear가 아님).
  - `shift_type=='Q'` → `modules_q.LinearShiftQ`, weight/bias 복사 (`convert.py:31-42`).
  - `shift_type=='PS'` → `modules.LinearShift`, `convert_weights=True`면 `get_shift_and_sign(linear.weight)`로 초기화 (`convert.py:43-57`).
  - Conv2d 변환 블록은 전체 주석 처리됨 (`convert.py:65-108`) — 현재 Conv는 변환 안 함.
- **`round_shift_weights(model, ...)` (`convert.py:112-134`)**: 추론 전 shift/sign을 정수로 라운딩 고정.
- **`convert_to_multi(...)` (`convert.py:148-172`)**: shift 계층 → 일반 nn.Linear로 역변환 (가중치 = `2^shift*sign`).

### 3.6 `pvt_v2.py` — PVTv2 백본 + 다종 Attention

- **`PyramidVisionTransformerV2` (`pvt_v2.py:825-1025`)**: 4-stage 피라미드 구조. 각 stage = `OverlapPatchEmbed`(Conv 기반 패치, `pvt_v2.py:765-813`) + `Block` 리스트 + LayerNorm (`pvt_v2.py:873-914`). 모델 변종: `pvt_v2_b0~b5` (`pvt_v2.py:1052-1159`).
- **`Block` (`pvt_v2.py:658-762`)**: `attn_dict[args.attn_type]`로 attention 종류를 동적 선택 (`pvt_v2.py:692`). `moe_mlp=True`면 MLP를 `Mlp_FMoE`(MoE)로 (`pvt_v2.py:733-740`).
- **`attn_dict` (`pvt_v2.py:646-655`)** — 지원 attention:
  - `msa`: 표준 softmax MHSA `Attention` (`pvt_v2.py:92-136`). `attn=(q@k^T)*scale`, softmax (`pvt_v2.py:128-130`).
  - `sra`: Spatial-Reduction Attention (PVT 고유, conv로 K/V 다운샘플) (`pvt_v2.py:187-293`).
  - `linear`: linear attention `(k^T@v)` 먼저 계산 → `q@attn` (O(N) 복잡도) (`pvt_v2.py:636-638`).
  - `performer`/`ecoformer`: 커널화 linear attention.
  - `LinAngular` (= `LinAngularAttention_ksh`): **Ecoformer 양자화 + MatAdd 치환 attention** (`pvt_v2.py:422-529`).
- **`LinAngularAttention_ksh.forward` (`pvt_v2.py:482-529`)**: q/k를 L2 정규화 후 `qk_quant`(EcoformerQuant)로 양자화 (`pvt_v2.py:508-510`), **`attn = MatAdd(qk^T, v)`**, 출력 `x = 0.5*v + (1/π)*MatAdd(qk, attn)` (`pvt_v2.py:514,523`) + depthwise conv 보정 `dconv_v` (`pvt_v2.py:512,525`). 곱셈을 add 연산으로 대체.
- **`LinAngularAttention_binary` (`pvt_v2.py:296-419`)**: q/k를 **binary 양자화**(`gt(q-mean,0)`, STE) (`pvt_v2.py:384-400`) 후 MatAdd. → q,k가 binary이므로 `qk@v`가 사실상 add/누산으로 환원.

### 3.7 `matkernel.py` — 연산 추상화

- `MatAdd` / `MatMul` / `Mul` (`matkernel.py:6-27`): 셋 다 forward는 `a@b`(또는 `a*b`)지만 **이름으로 구분**하여 (1) FLOPs 계산 분기, (2) TVM 컴파일 시 add/shift 커널로 매핑하기 위한 마커 역할. 주석: "MatAdd op can be used to mask out the attention but we use multiply here for simplicity" (`matkernel.py:4-5`) — **PyTorch 레퍼런스 구현에서는 곱셈, 실제 HW/TVM에서는 add로 치환하는 의도**.

### 3.8 `fmoe_mlp.py` — Shift_Linear / Shift_Mlp / MoE MLP

- **`Shift_Linear(...)` (`fmoe_mlp.py:106-108`)**: `LinearShift` 래퍼. 기본 `weight_bits=5, act_integer_bits=16, act_fraction_bits=16` (`fmoe_mlp.py:14-16`).
- **`Shift_Mlp` (`fmoe_mlp.py:136-205`)**: fc1/fc2를 `Shift_Linear`로 구성한 MLP (`fmoe_mlp.py:157,160`).
- **`PyTorchFMoE_MLP` (`fmoe_mlp.py:257-332`)**: **2개 expert = [정규 곱셈 `Mlp`, `Shift_Mlp`]** (`fmoe_mlp.py:285-291`). `NaiveGate`가 토큰을 두 expert로 라우팅, `SparseDispatcher`로 dispatch/combine (`fmoe_mlp.py:325-328`). → 논문 핵심 "곱셈 primitive의 혼합".

### 3.9 `OPs_Speedups/Shift/latency_test.py` — TVM shift 커널 실측

- **`matshift(A,M,K,N)` (`latency_test.py:21-37`)**: TVM TE로 정의한 shift 행렬곱. 핵심: `te.sum(SIGN[j,k] * (INPUT[a,i,k] >> SHIFT[j,k]))` (`latency_test.py:32`) — **곱셈 없이 비트 시프트 `>>`와 sign 곱(±1)만으로 행렬곱**. INPUT은 int32, SHIFT/SIGN은 int8 (`latency_test.py:23-24`).
- `matmul` (`latency_test.py:41-52`): 비교용 float32 곱셈.
- `matshift_fake` (`latency_test.py:55-74`): `weight = pow(2,-SHIFT)*SIGN` 후 일반 곱셈 (가짜 shift, PyTorch 시뮬레이션 등가).
- PVT 실제 텐서 shape `shapeList` (`latency_test.py:14-17`)로 shift vs matmul latency 비교 (`latency_test.py:126-154`). → **논문 Fig.4/5 재현용** (`README.md:15`).

---

## 4. 알고리즘 / 수식

### 4.1 Shift 기반 가중치 표현 (PS 모드)
부동소수점 가중치 W를 2의 거듭제곱 × 부호로 근사:

```
sign  = sign(W)
shift = round(log2(|W|))      # (utils.py:27)
W ≈ 2^shift × sign            # (modules.py:40, 177)
```
shift_range = `(-(2^(b-1)-2), 0)`, b=weight_bits=5 → `[-14,0]` (`modules.py:102`).

**시프트로의 환원**: `Y = X · W`에서 `W=2^s` 이면 `X·2^s = X << s` (s≥0) 또는 `X >> (-s)`. 따라서 곱셈기 없이 시프터로 구현 가능. TVM 커널에서 `INPUT >> SHIFT` 로 실증 (`latency_test.py:32`).

backward: `∂(2^s)/∂s = 2^s·ln2` → `grad_shift = grad · 2^s·sign · ln2` (`modules.py:70-72`, `log2=ln2` `modules.py:15`).

### 4.2 SP2 (Sum-of-Power-of-2 / APoT) — 표현력 향상
단일 2-거듭제곱 대신 두 항의 합:
```
W ≈ (2^(2·s1)·sgn1 + 2^(2·s2-1)·sgn2) · (2/3) · sign     # (modules.py:169-171)
```
또는 잔차 분해 (`utils.py:40-47`):
```
shift_1 = floor(log2(|W|)),  잔차 = |W| - 2^shift_1
shift_2 = round(log2(잔차))
```
2개 shift+add로 곱셈 1회를 근사 → 시프트 2회 + 덧셈 1회.

### 4.3 Add 기반 Attention (linear/Ecoformer)
표준 attention `softmax(QK^T)V` 의 O(N²) 곱셈을, kernel-trick으로 `Q(K^TV)` 순서로 바꾸고(`pvt_v2.py:636-638`), Q/K를 binary 또는 양자화하여 `MatAdd`로 치환:
```
attn = MatAdd(K^T, V)                          # (pvt_v2.py:514)
X = 0.5·V + (1/π)·MatAdd(Q, attn) + dconv(V)   # (pvt_v2.py:523-525)
```

### 4.4 활성 고정소수점
```
delta = 2^(-frac_bits),  rounded = floor(x/delta)*delta,  clamp([-2^(int-1), 2^(int-1)-1])   # (utils.py:14-21)
```
기본 act_integer_bits=16, act_fraction_bits=16 (Q16.16) (`fmoe_mlp.py:14-15`).

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: ImageNet (1000-class). `datasets.py`, `mcloader/imagenet.py`. 기본 경로 `/datasets01/imagenet_full_size/...` (`params.py:267-275`).
- **엔트리**: `pvt/main.py` (GPU), `main_cpu.py`. mmcv config 필수 (`--config`, `params.py:11`).
- **핵심 실행 옵션** (`params.py`):
  - `--attn_type` (msa/sra/linear/performer/ecoformer/LinAngular 등, `params.py:345`)
  - `--shift_training` shift 계층 학습 (`params.py:308`)
  - `--shift_type` PS/Q (`params.py:314`), `--SP2` APoT 사용 (`params.py:313`)
  - `--moe_mlp` / `--moe_attn` MoE 치환 (`params.py:309-310`)
  - `--progressive_training` 점진적 shift 학습 (`params.py:312`)
  - `--use_kernel` 커스텀 커널 사용 (`params.py:315`)
  - `--tvm_tune` / `--tvm_throughput` TVM 컴파일·throughput (`params.py:349-350`)
  - `--nbits` 비트수 (기본 16, `params.py:346`)
- **학습 하이퍼파라미터(기본값)**: AdamW, lr=5e-5, weight_decay=0.05, cosine sched, epochs=30, batch=128, mixup/cutmix/AutoAugment 활성 (`params.py:9,41-253`).
- **TVM 커널 테스트**: `cd OPs_Speedups/Shift && python Ansor_tune.py && python latency_test.py` (`OPs_Speedups/README.md:17-36`). USE_CUDA ON으로 TVM 빌드 필요.

---

## 6. 의존성

- **PyTorch** + **timm** (DropPath, trunc_normal_, register_model, `pvt_v2.py:6-8`).
- **mmcv** (config 로딩, `params.py:356`).
- **thop** (FLOPs 프로파일, `pvt_v2.py:9`).
- **TVM** (auto_scheduler/Ansor, shift/add 커널, `latency_test.py:2-8`) — USE_CUDA 빌드 필수.
- **DeepShift** 외부 CUDA/CPU 커널 (`deepshift.kernels`, use_kernel=True 시) — **분석 제외**.
- `tornado, psutil, xgboost>=1.1.0, cloudpickle` (TVM 튜닝, `OPs_Speedups/README.md:12`).
- `tree` (fmoe_mlp), fmoe 관련 자체 모듈(`fmoe_new.py`).

---

## 7. 강점 / 한계 / 리스크

**강점**
- **FPGA/ASIC 친화도 최상**: 가중치가 `2^shift×sign` → 곱셈기(DSP) 없이 시프터+가산기로 GEMM 구현 가능. TVM 커널이 `>>`로 실증 (`latency_test.py:32`).
- 활성/바이어스가 16비트 고정소수점(`round_to_fixed`)으로 통일 — 정수 데이터패스 일관성.
- MoE로 정확도-효율 trade-off 조절 (곱셈 expert vs shift expert).
- SP2/APoT로 단일 PoT 대비 표현력 보강.

**한계 / 리스크**
- shift 치환이 **attention의 to_qk/to_v/proj 일부 Linear에만** 적용 (`convert.py:29`) — 전체 모델이 곱셈-free는 아님. patch embed Conv, dwconv 등은 부동소수점 유지.
- 코드에 다수의 TODO/FIXME/주석처리 (예 `convert.py`의 Conv 블록 전체 주석, `pvt_v2.py:108` softmax 양자화 TODO) — **실험적/미완 상태 흔적**.
- 사전학습 체크포인트 미공개("ToDos", `README.md:18-19`) → 정확도 재현 불가.
- use_kernel=True 경로는 외부 DeepShift CUDA 커널 의존 — 이식성 제약.
- act fixed-point가 16/16비트로 다소 넉넉 — 하드웨어 비용 관점에서 추가 축소 여지(논문 외 추정).

---

## 8. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적, 추정)

> 우리 프로젝트는 "Transformer/ViT FPGA 가속기(HG-PIPE 계열) + XR 시선추적"으로 **추정**됨. 아래는 FPGA 친화 양자화·연산치환 관점의 시사점.

1. **곱셈→시프트 치환이 DSP-bound 완화에 직결**: FPGA에서 ViT GEMM의 병목은 DSP48 수. ShiftAddViT의 `2^shift×sign` 가중치는 DSP 없이 LUT 기반 시프터+가산기로 PE를 구성 가능 → **HG-PIPE의 systolic/output-stationary PE를 shift-PE로 재설계 시 DSP 절감·동일 LUT로 더 많은 PE 배치** 가능 (추정). TVM 커널 `INPUT >> SHIFT`(`latency_test.py:32`)가 그대로 RTL 시프트로 매핑됨.
2. **고정소수점 활성(Q16.16)** (`utils.py:7-21`)은 FPGA 정수 데이터패스와 정합 — requantization 로직 단순화. 단 16/16은 과할 수 있어 시선추적용 경량 ViT엔 더 공격적 비트(예 8/8) 탐색 권장.
3. **SP2(2-term PoT)** (`modules.py:169-171`)는 "시프터 2 + 가산기 1"로 정확도 보강 — 정확도 민감한 시선추적(저오차 요구) 회귀 헤드에 선택 적용하는 mixed-PoT 전략에 활용 가능.
4. **MoE(곱셈 expert + shift expert)** (`fmoe_mlp.py:285-291`)는 HW에서 두 데이터패스 동시 운영을 요구해 FPGA 자원 부담 → 시선추적 실시간성 위해서는 단일 shift-path로 단순화하는 편이 현실적(추정).
5. **MatAdd 추상화 마커**(`matkernel.py`)는 알고리즘↔HW 매핑 분리 설계 — 우리 HLS/RTL 파이프라인에서도 "연산 종류 태깅 → 커널 선택" 패턴 차용 가능.
6. **차이점 주의**: 본 repo는 GPU/TVM 타깃 PTQ/QAT 시뮬레이션 중심이며 **FPGA RTL/HLS 미포함**(확인 불가). P2-ViT처럼 가속기 RTL은 없음. 우리 프로젝트엔 "양자화 알고리즘 레퍼런스"로서 가치가 크고, HW 구현은 자체 설계 필요.

---

## 9. 근거/불확실성 정리
- **코드 확인**: shift/sign 표현, STE, SP2/APoT, MoE expert 구성, MatAdd 치환, TVM shift 커널 — 모두 위 (파일:라인)으로 확인.
- **추정**: 프로젝트(HG-PIPE/XR) 연관 시사점, FPGA PE 재설계 효과 — "추정" 명시.
- **확인 불가**: 사전학습 체크포인트(미공개), 외부 CUDA 커널 내부 구현(분석 제외), 실제 정확도/속도 수치(논문 본문 미열람, .json 미공개).
