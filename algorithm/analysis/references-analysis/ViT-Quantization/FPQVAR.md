# FPQVAR 코드베이스 정밀 분석

> 분석 대상 repo: `REF/ViT-Quantization/FPQVAR`
> 분석 방식: Glob/Grep/Read 기반 정적 분석 (bash 미사용). CUDA 커널(`quant/*.cu`, `quant/quant.cpp`), build 산출물(`*.o`, `*.so`), 체크포인트(`*.pt`, `*.pth`, `*.npz`)는 이름·역할만 기술. 자체 양자화 핵심 `.py`만 함수/클래스 단위로 라인 근거(파일:라인)와 함께 분석.

---

## 1. 개요

### 1.1 정체 (확인됨)
- **논문**: "FPQVAR: Floating Point Quantization for Visual Autoregressive Model with FPGA Hardware Co-design", Wei, Renjie / Xu, Songqiang / Guo, Qingyu / Li, Meng, arXiv:2505.16335, 2025. (`README.md:1-4`, `README.md:66-71`의 BibTeX로 확인)
- 본 repo는 README에 따르면 **"알고리즘 파트의 공식 구현"**이다 (`README.md:2`). 즉 FPGA HW co-design 자체(RTL/HLS)는 포함되지 않고, **PTQ(Post-Training Quantization) 알고리즘 + 이미지 생성 + FID 평가 파이프라인**만 들어있다.

### 1.2 목적·핵심 아이디어
대상 모델은 **VAR (Visual AutoRegressive model)** — next-scale prediction 방식의 이미지 생성 트랜스포머다 (depth=30, VAR-d30 = 256x256; depth=36, VAR-d36 = 512x512, `README.md:49`, `evaluate_fp_quant_transform_rotate.py:54-60`). FPQVAR는 이 VAR를 **정수(INT) 대신 저비트 부동소수(FP4/FP6) 포맷**으로 PTQ 양자화한다. 핵심 4요소:

1. **FP4/FP6 floating-point 양자화** — 비균일(non-uniform) 양자화 그리드를 사용. exponent/mantissa 조합별로 그리드를 미리 정의하고, 입력을 가장 가까운 그리드 값으로 매핑(round-to-nearest-grid).
2. **비대칭(asymmetric) FP 포맷** — GELU 출력(fc2 입력)처럼 음/양 분포가 비대칭인 텐서에 대해 음수부·양수부에 서로 다른 FP 포맷을 적용 (`fp_e1m2_neg_e2m1_pos`).
3. **Hadamard rotation (block-diagonal rotation 포함)** — outlier를 채널 간에 분산시켜 저비트 양자화 정확도를 확보 (QuaRot/SpinQuant 계열 기법).
4. **GALT (GHT-Aware Learnable Transformation)** — 채널별 학습 가능한 smoothing factor `s`(λ에 해당)를 MSE로 학습하여, rotation과 함께 weight↔activation 간 양자화 난이도를 재배분.

이 모든 기법은 **하드웨어(FPGA) 친화적인 저비트 FP 포맷 선택**을 전제로 설계되어 있다(FP4 = 4비트, FP6 = 6비트; group_size=128 그룹 단위 scale).

---

## 2. 디렉토리 구조

```
FPQVAR/
├── models/                            # [원본] FP32 VAR 모델 (baseline, 양자화 없음)
├── models_quant/                      # [INT 양자화] 정수 PTQ baseline
├── models_fp_quant/                   # [FP 양자화, rotation 없음] FP4/FP6 PTQ
├── models_fp_quant_rotate/            # [FP 양자화 + rotation] rotation까지
├── models_fp_quant_transform_rotate/  # [FP + GALT transform + rotation]  ← FPQVAR 최종본
├── rotate_utils/                      # Hadamard / rotation 유틸
│   ├── hadamard_utils.py              # Hadamard 행렬 생성·적용 (대부분 하드코딩 행렬)
│   ├── rotation_utils.py              # weight rotation, block-diagonal Hadamard
│   └── block_rotation_utils.py        # block-diagonal Hadamard 생성
├── learnable_transformation/          # GALT (학습 가능 smoothing factor s 학습)
│   ├── learnable_transformation_fc1_fp4.py
│   ├── learnable_transformation_mat_qkv_fp4.py / _fp6.py / *_512x512.py
│   ├── transform_model_utils.py       # 학습된 s를 weight에 적용
│   └── best_lambda_var30/, best_lambda_var36/  # [체크포인트] *_best_s_fp4.pt (이름만)
├── search/                            # FP 포맷 탐색 (MSE 기반)
│   ├── search_fp4_format.py / search_fp6_format.py / search_fp_format_ada.py
│   ├── baseline/                      # 추가 탐색·플롯 스크립트
│   └── optimal_quantization_formats_*.json  # 블록별 최적 포맷 결과
├── quant/                             # [CUDA 커널 — 역할만] FP round-to-nearest 가속
│   ├── quant.cpp / quant_kernel.cu    # pybind11 → quant_cuda.quant(x, grid)
│   ├── setup.py                       # CUDAExtension 빌드
│   └── build/, *.egg-info/            # [build 산출물 — 이름만]
├── utils/                            # data/lr/arg 유틸 (VAR 원본 유래)
├── evaluate_fp_quant_transform_rotate.py        # 256x256 최종 평가 진입점
├── evaluate_fp_quant_transform_rotate_512x512.py# 512x512 최종 평가 진입점
├── evaluate*.py                       # 기타 평가 변형들
├── pack_figs.py                       # 생성 이미지 → npz 패킹
├── openai_evaluator.py               # FID 평가 (OpenAI toolkit)
├── run.sh                             # 전체 실행 커맨드 모음
└── README.md
```

### 2.1 자체 .py 정밀 분석 대상 (본 문서의 분석 범위)
- `models_quant/quant_utils.py` (INT 양자화 baseline)
- `models_fp_quant_rotate/quant_utils.py` (FP4/FP6 양자화 핵심)
- `models_fp_quant_transform_rotate/basic_var.py` (rotation+GALT 삽입 위치)
- `rotate_utils/rotation_utils.py`, `rotate_utils/block_rotation_utils.py`
- `rotate_utils/hadamard_utils.py` (함수 시그니처만 — 421KB의 대부분이 하드코딩 Hadamard 행렬이라 본문 분석 제외)
- `search/search_fp4_format.py`, `search_fp6_format.py`, `search_fp_format_ada.py`
- `learnable_transformation/learnable_transformation_fc1_fp4.py`, `transform_model_utils.py`
- `evaluate_fp_quant_transform_rotate.py`, `run.sh`

### 2.2 분석 제외 (이름·역할만)
- `quant/quant.cpp`, `quant/quant_kernel.cu` — FP round-to-nearest CUDA 커널 (csrc 제외 규칙).
- `quant/build/*.o`, `quant/build/*.so` — 빌드 산출물.
- `learnable_transformation/best_lambda_var30/*.pt`, `best_lambda_var36/*.pt` — 학습된 smoothing factor 체크포인트.
- VAE/VAR 가중치(`vae_ch160v4096z32.pth`, `var_d30.pth` 등)는 repo에 없음(HuggingFace 다운로드, `README.md:17`).

### 2.3 models / models_quant / models_fp_quant_rotate(및 transform_rotate) 세 버전 차이
| 디렉토리 | 양자화 | rotation | GALT | basic_var 비고 |
|---|---|---|---|---|
| `models/` | 없음 (FP32 원본) | 없음 | 없음 | 원본 VAR. |
| `models_quant/` | INT (per-channel/tensor/group, sym/asym) | 없음 | 없음 | `quant_utils.py`가 정수 quantizer. `quant.py`는 VQVAE의 `VectorQuantizer2`(코드북 VQ)로, **이 파일의 "quant"는 양자화 PTQ가 아니라 VQ-VAE 토큰화**임에 주의. |
| `models_fp_quant/` | FP4/FP6 | **없음**(주석 처리, `models_fp_quant/basic_var.py:112`에 `# x = torch.matmul(x, rotation_matrix.T) # debug`만 존재) | 없음 | FP 양자화만. |
| `models_fp_quant_rotate/` | FP4/FP6 | 있음(Hadamard rotation) | 없음 | rotation 추가. |
| `models_fp_quant_transform_rotate/` | FP4/FP6 | 있음 | **있음** | **FPQVAR 최종 구성**. `basic_var.py:263,266`에서 GALT smoothing×rotation을 LayerNorm 출력에 온라인 융합. |

> 즉 디렉토리들은 ablation 단계(원본 → INT → FP → FP+rotate → FP+rotate+GALT)를 코드 복제로 구현한 형태다. 최종 결과는 `*_transform_rotate`를 사용한다(`run.sh`의 모든 커맨드가 `evaluate_fp_quant_transform_rotate*.py` 호출).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 FP4 포맷 정의 (exponent/mantissa 비트 의미)
FP4 = 4비트 부동소수. 1 sign bit + exponent bits(E) + mantissa bits(M), E+M=3. FPQVAR는 세 가지 FP4 변형을 그리드(quant_grid)로 하드코딩한다. (`models_fp_quant_rotate/quant_utils.py`)

| 포맷 | 의미 | 그리드 (양수부 대표값) | 특징 | 라인 근거 |
|---|---|---|---|---|
| `fp_e1` (e1m2) | E=1, M=2 | `[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]` | 동적 범위 작음, 정밀도 높음(균일에 가까움) | `quant_utils.py:310`, `:323`, `:338` |
| `fp_e2` (e2m1) | E=2, M=1 | `[0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]` | 균형형. **FPQVAR 기본 act/weight 포맷** | `quant_utils.py:262`, `:275`, `:290` |
| `fp_e3` (e3m0) | E=3, M=0 | `[0, 0.25, 0.5, 1, 2, 4, 8, 16]` | 동적 범위 큼(outlier 친화), 정밀도 낮음 | `quant_utils.py:234`, `:247` (단 e3 그리드는 파일에 따라 e3m0 정의가 약간 다름; `search_fp4_format.py:479`의 `fp4_e3m0_grid`=`[-16,-8,-4,-2,-1,-0.5,-0.25,0,...]`가 정식) |

- 각 그리드는 총 15개 값(=2^4−1, 0이 두 번 나오지 않도록 부호 대칭 + 단일 0)으로 4비트 표현력에 대응.
- e3m0의 그리드가 파일마다 두 종류 존재: `[-64..64]`(`search_fp4_format.py:279`의 `fp_quant_e3`)와 `[-16..16]`(`:328`의 `fp_quant_e3_per_group`, `:479`의 `fp4_e3m0_grid`). 실제 모델 적용본(`models_fp_quant_rotate/quant_utils.py:234`)은 `[-16..16]`을 사용 → **추정**: 16 버전이 VAR 분포에 맞춰 채택된 정식 포맷.

### 3.2 FP6 포맷 정의
FP6 = 6비트. 64개 그리드 값. 두 변형 (`models_fp_quant_rotate/quant_utils.py:370-398`):
- `fp6_e2m3` (E=2, M=3): 그리드 범위 ±7.5, 촘촘함. **FP6의 기본 act/weight 포맷** (`run.sh:7,22`).
- `fp6_e3m2` (E=3, M=2): 그리드 범위 ±28, 동적 범위 큼.

### 3.3 비대칭 FP 포맷 — `fp_e1m2_neg_e2m1_pos` (FPQVAR의 핵심 기여 중 하나)
GELU(approximate='tanh') 출력은 음수부가 [-0.17, 0) 근방으로 작고, 양수부는 크게 퍼진 **비대칭** 분포다. 이를 단일 대칭 FP 포맷으로 양자화하면 손실이 크다. 그래서 **음수부와 양수부에 다른 FP 포맷·다른 scale**을 적용한다.

`fp_quant_e1m2_neg_e2m1_pos_per_group` (`models_fp_quant_rotate/quant_utils.py:335-366`):
- 음수부 그리드 `quant_grid_e1m2_neg = [-1.75,...,-0.25, 0.0]` (E1M2, 정밀형) — `:338`
- 양수부 그리드 `quant_grid_e2m1_pos = [0.0, 0.5,...,6.0]` (E2M1, 넓은 범위) — `:339`
- 음/양 분리(`torch.where(x<=0,...)`, `:347-348`), 각각 absmax로 별도 scale 계산(`scale_neg`, `scale_pos`, `:351-352`), 별도 양자화 후 합성(`:359-364`).
- 이 포맷은 **fc2 입력(GELU 출력)에만** 적용됨 — `QuantizedLinear_fc2` 분기에서만 `fp_e1m2_neg_e2m1_pos`를 받음 (`:778-779`). 일반 `QuantizedLinear`에는 이 분기가 없다(`:521-568`엔 없음).
- 최종 256x256 FP4 커맨드: `--fc2_fp_type fp_e1m2_neg_e2m1_pos` (`run.sh:4`).
- FP6에서는 `fp6_int_neg_e2m3_pos`를 fc2에 사용(`run.sh:7,10,22`)하나, 본 repo의 `quant_utils.py`에는 이 정확한 이름의 함수가 구현되어 있지 않음 → **확인 불가/추정**: 별도 구현 누락 또는 다른 이름으로 매핑.

### 3.4 양자화 기본 연산 — quantizer / observer / scale / zero-point
**Observer(통계)와 scale은 별도 클래스가 아니라 양자화 함수 내부에서 그때그때 absmax로 계산**한다(static observer 객체 없음, 즉 calibration이 함수 호출마다 즉석 계산되는 dynamic 방식).

#### (a) INT 양자화 baseline (`models_quant/quant_utils.py`)
- per-channel sym/asym, per-tensor, per-group, per-token 모두 구현.
- scale 계산식: `scale = absmax / q_max`, `q_max = 2^(n_bits-1)-1` (`:11-13`, `:33-35` 등).
- asym: `zero_point = round(q_min - t_min/scale)`, `t_hat = clamp(round(t/scale)+zp, q_min, q_max).sub(zp).mul(scale)` (`:60-65`).
- per_group group_size는 코드 내 **하드코딩 128** (`:205`, `:261`).

#### (b) FP 양자화 (`models_fp_quant_rotate/quant_utils.py`)
- 공통 패턴(예 `fp_quant_e2_per_group`, `:272-284`):
  1. (선택) clamp,
  2. `x` 를 `(-1, group_size)`로 reshape (group-wise),
  3. `scale = x.abs().max(dim=-1) / grid.abs().max()` — 그룹별 absmax를 그리드 최대값에 맞춤,
  4. `x /= scale` 로 그리드 표현 범위로 정규화,
  5. `quantize_to_nearest_grid(x, grid)` 로 가장 가까운 그리드 값 매핑,
  6. `output = quantized_x * scale` 로 역정규화(dequant).
- `quantize_to_nearest_grid` (`:207-228`): `argmin |x.unsqueeze(-1) - grid|` 로 nearest-neighbor 매핑. 순수 PyTorch 버전.
- `*_cuda` 버전(`fp_quant_e2_per_group_cuda`, `:287-304`; `fp6_quant_*_cuda`, `:400-471`)은 위 nearest-grid 매핑을 CUDA 커널 `quant_cuda.quant(quant_array, quant_grid)`로 대체(속도). 커널 자체는 분석 제외(quant/).
- **FP 양자화에는 zero-point가 없음** (부호 대칭 그리드 + per-group scale만). 비대칭은 §3.3의 음/양 분리 방식으로 처리.
- log2 양자화(`log2_quant_per_group_asym`, `:179-204`)도 구현되어 있으나 fc2 대안(실험용)으로 보임.

#### (c) `QuantizedLinear` / `QuantizedLinear_fc2` (`models_fp_quant_rotate/quant_utils.py:474-913`)
- `nn.Linear`를 대체하는 양자화 선형 레이어. weight는 buffer로 저장(`:502-510`), `from_float`로 원본 가중치를 즉시 양자화(`:596-685`).
- `forward` (`:588-594`): `q_x = act_quant(x)` → `F.linear(q_x, weight, bias)` → `output_quant(y)`. 즉 **activation은 동적 양자화, weight는 PTQ로 사전 양자화된 값을 그대로 사용**.
- 분기 트리: `act_quant`(per_token/per_tensor/per_group) × `activation_fp_quant`(FP vs INT) × `act_fp_type`(fp_e1/e2/e3, fp6_e2m3/e3m2) — `:521-571`.
- `QuantizedLinear_fc2`는 `fp_e1m2_neg_e2m1_pos` 비대칭 포맷 분기를 추가로 가짐 (`:778-779`).

### 3.5 VAR attention/FFN 구조에 양자화 삽입 (`quantize_VAR`, `:916-978`)
모듈 타입을 순회하며 교체:
- `FFN`: `m.fc1` → `QuantizedLinear`, `m.fc2` → `QuantizedLinear_fc2`(비대칭, `act_quant_sym=False`, `:939-947`).
- `SelfAttention`: `m.mat_qkv` → `QuantizedLinear`, `m.proj` → `QuantizedLinear` (`:950-966`).
- `AdaLNSelfAttn`: `m.ada_lin[1]` → `QuantizedLinear` (`:969-976`).
- INT baseline의 `quantize_VAR`(`models_quant/quant_utils.py:276-311`)는 fc2를 주석 처리(`:287-290`)하고 ada_lin도 미적용(`:300-304`) → INT에서는 더 보수적으로 양자화함을 의미.
- `quantize_VAR_use_different_datatype`(`:982-1067`): **블록별로 다른 FP 포맷**을 강제 지정하는 변형(mixed-precision per-block). 예: fc1 블록 6~20은 `fp_e2`, 그 외는 act `fp_e3`/weight `fp_e2`(`:997-1014`); mat_qkv 블록 24~25는 `fp_e2`, 그 외 `fp_e3`(`:1029-1046`). → **추정**: §3.7 search 결과를 코드로 굳힌 mixed-format 실험 경로.

### 3.6 Hadamard rotation / block rotation 적용 위치

#### (a) rotation 유틸 (`rotate_utils/rotation_utils.py`, `block_rotation_utils.py`)
- `random_hadamard_matrix(size, device, seed)` / `random_orthogonal_matrix` (`rotation_utils.py:38-64`): Hadamard 또는 QR 기반 직교행렬 생성. `get_orthogonal_matrix(mode='hadamard')`가 진입.
- `block_random_hadamard_matrix(total_size=1920, block_size=128, ...)` (`rotation_utils.py:69-104`, `block_rotation_utils.py:16-51`): **block-diagonal Hadamard**. 1920채널을 128 블록 15개로 나눠 각 블록에 독립 Hadamard 적용 → block_diag로 합성(`:108-126`). full 1920×1920 대신 블록 대각 → **HW 비용 절감(FPGA에서 작은 Hadamard 유닛 재사용)**.
- weight-side rotation (PTQ에서 가중치에 미리 흡수): `rotate_mat_qkv`(`:129-144`), `rotate_fc1`(`:147-154`), `rotate_fc2`(`:157-164`), `rotate_ada_lin`(`:167-207`). 각각 `W = W @ Q` (fp64 계산 후 캐스팅).
- `rotate_model(model, device, block_rotate)` (`:211-241`): `block_rotate=False`면 full Hadamard, `True`면 block-diagonal. 각 블록의 `mat_qkv`, `fc1` weight를 회전 (fc2/ada_lin은 주석 처리됨, `:221-222`, `:239`).

#### (b) activation-side rotation (온라인) — `models_fp_quant_transform_rotate/basic_var.py`
`AdaLNSelfAttn.forward` (`:253-269`)에서 LayerNorm 출력에 **GALT scale과 Hadamard rotation을 융합**:
```python
x_1 = torch.matmul(self.ln_wo_grad(x).mul(scale1.add(1)).add_(shift1).mul(mat_qkv_best_s), rotation_matrix)  # :263
x   = x + self.drop_path(self.attn(x_1, ...).mul_(gamma1))                                                    # :264
x_2 = torch.matmul(self.ln_wo_grad(x).mul(scale2.add(1)).add_(shift2).mul(fc1_best_s), rotation_matrix)       # :266
x   = x + self.drop_path(self.ffn(x_2, ...).mul(gamma2))                                                      # :267
```
- 즉 **activation은 `(LN출력 ∘ scale/shift) × s(GALT) @ Q(Hadamard)` 형태로 회전된 뒤** mat_qkv/fc1에 입력된다. weight 쪽은 사전에 `W/s` (transform) 후 `W @ Q` (rotate)로 흡수되어, 수학적으로 등가(`(x·s)Q · (W/s·Q)^T = x·W^T`)를 유지하면서 양자화 난이도만 재분배.
- KV cache 양자화: `SelfAttention.forward`(`:160-219`)에서 캐시된 k/v를 kv_bit에 따라 FP6(`fp6_quant_e2m3_per_token_cuda`) 또는 FP4(`fp_quant_e2_per_group_cuda`)로 양자화(`:192-209`).

### 3.7 FP format search 알고리즘 (`search/`)
블록별·레이어별로 최적 FP 포맷을 MSE 기반 grid search로 선택.

#### (a) FP4 search (`search/search_fp4_format.py`)
- 후보 `formats = ['e1m2','e2m1','e3m0']` (`:577`,`:612`,...).
- 절차(예 mat_qkv, `:600-654`): 30개 블록 각각에 대해, calibration activation(label 100개 × step 10개)과 weight를 로드 → weight_format×act_format 모든 조합(3×3=9)에 대해 `y_fp = x@W^T` vs `y_quant = x_quant@W_quant^T`의 **출력 MSE**(`compute_quant_error`, `:472-476`)를 계산 → 최소 MSE 조합 선택.
- **핵심**: 텐서 자체 MSE가 아니라 **GEMM 출력(y) MSE**를 기준으로 함 (가중치-활성화 결합 효과 반영).
- 결과를 `optimal_quantization_formats_{mat_qkv,fc1,fc2,proj}.json`에 저장(`:651-654`). JSON 포맷: `[{block_idx, weight_format, activation_format, loss}, ...]` (실제 파일 `optimal_quantization_formats_fc2.json:1-5`로 확인: block0 → weight `e3m0`, act `e2m1`).

#### (b) FP6 search (`search/search_fp6_format.py`)
- 후보 `['e2m3','e3m2']` (`:584`,...). FP4와 동일한 출력-MSE grid search. condition/mat_qkv/proj/fc1/fc2 각각 수행. 결과 `optimal_fp6_formats_*.pt` 저장(`:623`,`:682` 등).

#### (c) ada search (`search/search_fp_format_ada.py`)
- `ada` (AdaLN 조건부 입력) 텐서에 대한 포맷 탐색. 다양한 quantizer(INT per-token/group, fp_e1/e2/e3, log2, du_quantizer)의 출력 MSE를 비교 출력(`:415-470`). du_quantizer(`:127-202`)는 inner/outer 두 구간을 다른 step으로 양자화하는 mixed quantizer(실험용).

### 3.8 GALT — GHT-Aware Learnable Transformation (`learnable_transformation/`)
README 명시(`README.md:46-49`)의 핵심 기여. 채널별 학습 가능한 smoothing factor `s`를 MSE로 학습.

#### (a) 학습 스크립트 (`learnable_transformation_fc1_fp4.py`)
- 목적: fc1 레이어에 대한 best `s` (길이=1920=채널수, `:224`) 학습.
- 손실 함수 `compute_quant_error_v1(x, w, learnable_s, Q)` (`:117-133`):
  ```python
  fp_result   = x @ w.T
  x_2 = (x * learnable_s) @ Q ;  x_2_quant = FPQuant(x_2)   # act: smooth → rotate → FP4 quant
  w_2 = (w / learnable_s) @ Q ;  w_2_quant = FPQuant(w_2)   # weight: 역smooth → rotate → FP4 quant
  quant_result = x_2_quant @ w_2_quant.T
  return mean((fp_result - quant_result)^2)
  ```
  즉 `(x·s)Q`와 `(w/s)Q`가 곱해지면 `s`·`Q`가 상쇄되어 원래 결과와 등가지만, **양자화는 회전·smoothing된 표현에서 수행**되므로 MSE가 작아지는 `s`를 찾는다.
- `FPQuant`(`:70-95`)는 e2m1(fp_e2) 그리드 고정, group=128, backward는 STE(straight-through: `grad_input = grad_output`, `:86-95`).
- 최적화: `s = nn.Parameter(ones(1920))`, AdamW lr=0.01, epochs=50(기본), 30블록 각각 독립 학습(`:215-253`), best `s` 리스트를 `fc1_best_s_fp4.pt`로 저장(`:255`).
- `Q`는 §3.6의 block-diagonal Hadamard(1920, 128, seed=42)로 고정(`:163-168`) → 이름 그대로 **rotation-aware(=GHT-aware) learnable transform**.
- 동일 패턴의 mat_qkv용 / fp6용 / 512x512용 변형 다수(`learnable_transformation_mat_qkv_fp4.py` 등).

#### (b) 적용 (`transform_model_utils.py`)
- `transform_mat_qkv`(`:8-13`), `transform_fc1`(`:16-21`): `W_new = W / best_s` 로 weight에 역smoothing을 사전 흡수.
- `transform_model`(`:24-28`): 모든 블록에 학습된 `mat_qkv_best_s[idx]`, `fc1_best_s[idx]` 적용.
- activation 쪽 `× s`는 §3.6의 `basic_var.py:263,266`에서 온라인 적용.
- 학습된 s는 `best_lambda_var30/{mat_qkv,fc1}_best_s_fp4.pt`, `best_lambda_var36/...`에 제공(README:49, 체크포인트 — 이름만).

---

## 4. 알고리즘 / 수식

### 4.1 FP 양자화 수식 (round-to-FP-grid)
입력 텐서 `x`를 그룹 단위(g=128)로 나눈 뒤, 그룹 `G`에 대해:

1. **scale**: `Δ = max_{i∈G}|x_i| / max(grid)`  (그룹 absmax를 그리드 최대값에 정렬)
2. **정규화**: `x̃ = x / Δ`
3. **그리드 매핑**: `q = argmin_{v∈grid} |x̃ - v|`  →  `x̂_grid = v_q`
4. **dequant**: `x̂ = x̂_grid · Δ`

여기서 grid는 FP 포맷이 표현 가능한 비균일 값 집합(§3.1, §3.2). exponent 비트가 클수록 grid가 지수적으로 넓게 퍼지고(동적 범위↑), mantissa 비트가 클수록 인접 grid 간격이 촘촘(정밀도↑)하다. INT 양자화의 `round(x/Δ)`가 균일 격자인 것과 달리, FP는 **0 근방 촘촘 + 큰 값 성김**이라 정규분포·outlier가 섞인 트랜스포머 텐서에 유리.

### 4.2 비대칭(음/양 분리) FP — fc2/GELU 출력
`x⁻ = min(x,0)`, `x⁺ = max(x,0)` 로 분리 후 각각 다른 grid·scale:
- `Δ⁻ = max|x⁻| / max(grid_e1m2_neg)`, `Δ⁺ = max(x⁺) / max(grid_e2m1_pos)`
- `x̂ = nearest(x⁻/Δ⁻, grid_neg)·Δ⁻ + nearest(x⁺/Δ⁺, grid_pos)·Δ⁺`

(근거: `quant_utils.py:347-364`)

### 4.3 FP 포맷 후보 탐색 (출력-MSE 기반)
블록 b, 레이어 ℓ 마다:
```
(W_fmt*, A_fmt*) = argmin_{W_fmt, A_fmt}  E_x[ || x·Wᵀ − Q_A(x)·Q_W(W)ᵀ ||² ]
```
calibration 데이터에 대한 평균. FP4는 3×3, FP6는 2×2 조합 완전탐색. (근거: `search_fp4_format.py:617-636`)

### 4.4 GALT — learnable smoothing factor `s` (논문의 λ)
등가 변환 `x·Wᵀ = (x⊙s)·(W⊘s)ᵀ` 를 회전과 결합하여,
```
s* = argmin_s  E_x[ || x·Wᵀ − Q((x⊙s)Q_H) · Q((W⊘s)Q_H)ᵀ ||² ]
```
를 AdamW(STE)로 학습. `Q_H`는 block-diagonal Hadamard, `Q(·)`는 FP4(e2m1) 양자화. (근거: `learnable_transformation_fc1_fp4.py:98-133`, `:224-253`)

### 4.5 Hadamard rotation으로 outlier 완화
직교행렬 `Q_H`(Hadamard)는 채널 차원을 회전시켜 특정 채널에 집중된 outlier 에너지를 여러 채널에 고르게 분산 → 그룹 absmax(=scale)가 outlier에 끌려가지 않게 함. block-diagonal로 하여 FPGA에서 작은(128×128) Hadamard 곱 유닛만으로 처리 가능. 등가성: `(xQ)(WQ)ᵀ = xQQᵀWᵀ = xWᵀ` (Q 직교). (근거: `rotation_utils.py:211-241`, `basic_var.py:263,266`)

---

## 5. 학습/평가 파이프라인

### 5.1 모델·데이터
- 모델: VAR (FoundationVision/var). depth=30 → 256x256, depth=36 → 512x512 (`evaluate_fp_quant_transform_rotate.py:54-71`). VQVAE(`vae_ch160v4096z32.pth`) + VAR transformer.
- patch_nums = (1,2,3,4,5,6,8,10,13,16) — VAR의 multi-scale token map (`:63`).
- 평가 데이터: ImageNet reference npz (256: `VIRTUAL_imagenet256_labeled.npz`, 512: `VIRTUAL_imagenet512.npz`, `README.md:19-20`).

### 5.2 진입점 `evaluate_fp_quant_transform_rotate.py` (256x256)
순서(`:87-199`):
1. VAE/VAR 로드 (`:75-80`).
2. `--transform`이면 학습된 best_s 로드 후 `transform_model`로 weight에 `W/s` 흡수 (`:87-97`). 아니면 s=1 (`:99-100`).
3. `--rotate`면 `rotate_model`로 weight rotation (`:103-106`).
4. `--quant`면 `quantize_VAR`로 Linear 교체 + 즉시 양자화, 그 후 `var.half()` (`:112-131`).
5. rotation matrix `Q` 생성(block 여부에 따라 full/block, `:142-153`).
6. 1000 클래스 × 50 이미지 생성: `autoregressive_infer_cfg(..., rotation_matrix=Q, mat_qkv_best_s, fc1_best_s, quant_KV, kv_bit)` (cfg=1.5, top_k=900, top_p=0.96) (`:187-199`).
7. PNG로 저장 (`:203-207`).

### 5.3 평가 절차 (README §Quantize and Evaluate)
1. `python evaluate_fp_quant_transform_rotate.py ...` → 이미지 생성 (`README.md:33`).
2. `python pack_figs.py --file_path <생성폴더>` → npz 패킹 (`README.md:38`).
3. `python openai_evaluator.py <ref폴더> <생성폴더>` → **FID/IS 등 OpenAI evaluation toolkit** 평가 (`README.md:43`).

### 5.4 주요 실행 구성 (`run.sh`)
- **256 FP4 + KV**: `--w_bit 4 --a_bit 4 --weight_quant per_group --act_quant per_group --act_sym --activation_fp_quant --weight_fp_quant --act_fp_type fp_e2 --weight_fp_type fp_e2 --fc2_fp_type fp_e1m2_neg_e2m1_pos --rotate --block_rotate --transform --quant_kv --kv_bit 6` (`run.sh:4`).
- **256 FP6**: `--w_bit 6 --a_bit 6 --weight_quant per_channel --act_quant per_token --act_fp_type fp6_e2m3 --weight_fp_type fp6_e2m3 --fc2_fp_type fp6_int_neg_e2m3_pos --rotate --block_rotate` (`run.sh:7`).
- 512x512는 `evaluate_fp_quant_transform_rotate_512x512.py`로 동일 패턴 (`run.sh:16-25`).
- 관찰: **FP4는 per_group(그룹128), FP6는 per_channel/per_token** 으로 granularity가 다름 → FP4의 저비트를 group-wise scale로 보완.

---

## 6. 의존성

- `requirements.txt`는 **repo에 실제로 존재하지 않음**(Glob 검색 0건). README는 `pip install -r requirements.txt`를 안내하나(`README.md:11`) 파일이 누락된 상태 → **확인 불가** (정식 의존성 목록 없음).
- import로 확인되는 의존성:
  - `torch`, `torchvision` (전반), `numpy`, `PIL` (`evaluate_*.py:12-15`).
  - `transformers`, `tqdm` (`rotation_utils.py:3-5`, `transform_model_utils.py:2-5`).
  - `quant_cuda` — repo 내 CUDA 확장(`pip install ./quant`로 빌드, `README.md:13`; setup.py에서 `quant_cuda` 모듈명, `quant/setup.py:6`).
  - `dist` (VAR 분산 유틸, `models_quant/quant.py:8`).
  - flash-attn / xformers (옵션, basic_var의 `flash_attn_func`/`memory_efficient_attention` 사용 — 가용 시).
  - `matplotlib` (플롯 스크립트, `learnable_transformation_fc1_fp4.py:9`).
- CUDA 빌드: `quant/setup.py`가 `CUDAExtension`으로 `quant.cpp`+`quant_kernel.cu`를 컴파일(`quant/setup.py:6-17`). build 산출물은 `quant/build/lib.linux-x86_64-cpython-39/quant_cuda.cpython-39-...so` → **Python 3.9 / Linux x86_64 환경** (build 디렉토리명으로 확인).

---

## 7. 강점 / 한계 / 리스크

### 강점
- **저비트 FP 포맷 + 출력-MSE search**: 텐서별 분포에 맞춘 exponent/mantissa 자동 선택으로 INT 대비 outlier 강건성 확보.
- **GELU 비대칭 포맷(e1m2_neg / e2m1_pos)**: fc2 입력 양자화의 정밀도 손실을 구조적으로 줄임 — 단순 대칭 양자화 대비 명확한 동기.
- **block-diagonal Hadamard + GALT의 등가 변환**: 정확도를 유지(수학적 등가)하면서 양자화 난이도만 재분배. block 단위라 FPGA HW 비용 친화적.
- **CUDA 커널로 round-to-nearest-grid 가속**: PyTorch argmin 버전과 커널 버전 병존 → 실용성.

### 한계 / 리스크
- **코드 복제 기반 ablation**: `models/`, `models_quant/`, `models_fp_quant*/` 다섯 디렉토리가 거의 동일 코드 복제 → 유지보수·일관성 리스크(한 곳 수정 시 동기화 누락 가능).
- **하드코딩 경로**: calibration/weight/best_s 경로가 `/home/rjwei/...`, `/home/wrj/...`로 절대경로 하드코딩(`search_fp4_format.py:605`, `block_rotation_utils.py:2`, `transform_model_utils`는 상대적). 재현 시 수정 필수.
- **requirements.txt 부재**: 의존성 버전 고정 불가 (확인 불가).
- **fc2 FP6 포맷 `fp6_int_neg_e2m3_pos` 미구현**: `run.sh`가 참조하나 `quant_utils.py`에 동명 함수 없음 → 실행 시 에러 가능 또는 별도 브랜치 필요(추정).
- **observer 부재(완전 dynamic)**: activation scale을 추론 시마다 absmax로 계산 → HW에서는 online absmax 회로가 필요(논문의 HW co-design이 이를 다룰 것으로 추정, 본 repo엔 없음).
- **STE backward만 제공**: GALT 학습 외 양자화는 PTQ로 gradient 불필요하나, 양자화 grid 자체는 학습 대상이 아님(고정 grid).

---

## 8. 우리 프로젝트 관점 시사점 (PRJXR-HBTXR)

> 전제(추정): 본 상위 프로젝트는 ViT/Transformer FPGA 가속기(HG-PIPE 계열, `IMPL_REPOS/HGPIPE`)와 XR 시선추적(eye-tracking)을 결합하는 것으로 보임. 아래는 그 가정 하의 연관성이며 "추정"임을 명시한다.

1. **저비트 FP 포맷 선택의 HW 함의**: FP4(e2m1)/FP6(e2m3)의 exponent/mantissa 비트 배분은 **FPGA의 저비트 부동소수 PE(multiplier/adder) 설계 직접 입력**이 된다. 우리 가속기가 INT8/INT4 PE라면, FPQVAR식 FP4 PE(15-entry grid LUT + scale 곱)로 전환 시 outlier 강건성을 얻을 수 있음 — 단 grid LUT·정규화 회로 추가 비용 고려 필요(추정).
2. **group-wise scale(group=128)**: 우리 HG-PIPE 데이터플로우의 타일/채널 분할 단위(128)와 정합되면 scale 브로드캐스트 회로를 타일 경계와 일치시킬 수 있음(추정). FP4를 per_group, FP6를 per_channel/token으로 쓰는 분리 전략은 정확도-HW비용 트레이드오프의 참고 사례.
3. **Hadamard rotation을 block-diagonal로 한정**: full N×N Hadamard가 아니라 128×128 블록만 쓰므로, **작은 Hadamard 곱 유닛 1개를 재사용**하는 FPGA 구현이 가능 — 우리 가속기에서 outlier 완화를 HW에 통합할 때 비용 상한을 잡아주는 직접적 레퍼런스(`rotation_utils.py:69-104`).
4. **GALT(λ smoothing)의 사전 흡수**: smoothing은 weight에 `W/s`로 사전 흡수(`transform_model_utils.py`)되고 activation `×s`만 온라인이므로, **HW에는 채널별 scale 곱 1회만 추가**된다(LayerNorm 출력 직후). 이는 우리 LayerNorm 데이터패스에 곱셈기 한 단을 더하는 수준으로 통합 가능(추정).
5. **시선추적(저지연 추론)과의 적합성**: VAR는 생성 모델이라 우리 인식/추적 태스크와 직접 동일하진 않으나, **PTQ + rotation + 비대칭 FP 포맷**이라는 정확도 보존 레시피는 ViT 백본(시선추적용)에 그대로 이식 가능(추정). 특히 GELU 출력 비대칭 양자화는 모든 ViT FFN에 공통 적용 가능한 일반 기법.
6. **평가 메트릭 차이 유의**: 본 repo는 생성품질(FID, openai_evaluator)로 평가 → 시선추적 정확도(각도 오차)와 직접 비교 불가. 양자화 기법만 차용하고 평가는 우리 태스크 메트릭으로 별도 수행해야 함(확인 사항).

---

## 9. 근거 표기 / 미확인 사항

- **확인됨(코드/README 직접 근거)**: FP4/FP6 grid 정의, 비대칭 fc2 포맷, group=128, block-diagonal Hadamard(1920/128/seed42), GALT MSE 학습(AdamW lr0.01 ep50, STE), 출력-MSE format search, rotation+transform의 `basic_var.py:263,266` 온라인 융합, 평가 3단계 파이프라인, run.sh 커맨드.
- **추정**:
  - e3m0 grid의 [-16..16] 채택이 정식(파일별 [-64..64]도 존재).
  - `quantize_VAR_use_different_datatype`의 블록별 포맷이 search 결과를 굳힌 실험 경로라는 점.
  - FPGA HW 비용 절감 동기(block-diagonal, weight 사전 흡수) — 논문 주장과 부합하나 본 repo에 HW 코드는 없음.
  - 우리 프로젝트(HG-PIPE/XR eye-tracking)와의 연관성 전반(상위 프로젝트 구성 추정 기반).
- **확인 불가**:
  - `requirements.txt` (파일 부재) → 정확한 의존성 버전.
  - `fp6_int_neg_e2m3_pos`(run.sh 참조) 구현체 → `quant_utils.py`에 동명 함수 없음.
  - VAE/VAR 체크포인트 내용(외부 다운로드, repo 미포함).
  - CUDA 커널 `quant_kernel.cu` 내부 구현(분석 제외 규칙).
