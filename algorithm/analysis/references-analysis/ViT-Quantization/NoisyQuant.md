# NoisyQuant 정밀 분석

> 분석 대상: `REF/ViT-Quantization/NoisyQuant`
> 작성일: 2026-06-20 / 근거: 실제 소스 코드 (파일:라인 표기)

---

## 1. 개요

- **목적**: ViT의 activation에 존재하는 **무거운 outlier/비균등 분포**를, **고정된 noisy bias(무작위 노이즈 편향)** 를 더해 분포를 평탄화한 뒤 균등(uniform) 양자화하여 저비트(예: 6-bit) PTQ 정확도를 회복.
- **원논문**: *"NoisyQuant: Noisy Bias-Enhanced Post-Training Activation Quantization for Vision Transformers"*, **CVPR 2023** (arXiv:2211.16056). README.md:1-3 명시.
- **핵심 아이디어**:
  1. activation `x`에 layer마다 **고정된 noisy bias `N`** 를 더해 `x+N`을 양자화하면, outlier로 인한 양자화 오차가 입력에 무관한 노이즈로 분산되어 평균 양자화 오차가 줄어든다(논문 이론).
  2. 양자화 후 다시 `N`을 빼서 보정 → 선형(Linear) 연산이므로 `bias`로 흡수 가능.
  3. noisy bias의 **mean(이동량)** 과 **range(스케일)** 를 캘리브레이션 데이터로 grid search.
- **양자화 자체는 매우 단순한 균등 양자화** (가중치 대칭 per-channel, 활성 대칭). 핵심 기여는 양자화 전에 더하는 noisy bias.
- 구현 방식: timm 모델의 `nn.Linear.forward`를 **MethodType으로 monkey-patch**하여 양자화 forward로 교체 (fast_quant.py:163). 별도 모델 재정의 없음 → 매우 경량 구현.

---

## 2. 디렉토리 구조 (자체 소스 + 제외 항목)

### 자체 핵심 소스
```
NoisyQuant/
├── fast_quant.py      # ★ 전부의 핵심: 양자화 함수 + noisy bias 탐색 + forward 패치
├── validate.py        # timm 기반 ImageNet 검증 스크립트 (캘리브레이션 호출 포함)
├── run.sh             # 재현용 실행 스크립트
├── README.md
└── LICENSE
```
> 사실상 **`fast_quant.py` 한 파일이 양자화 로직 전부**. `validate.py`는 timm 표준 validate 스크립트(Ross Wightman 작성, validate.py:8)에 양자화 호출만 추가.

### 제외
- `.git/` (모든 hooks/objects 제외)
- `validate.py`의 timm 표준 보일러플레이트(argparse 등)는 양자화 관련 인자(155-164)만 분석.

---

## 3. 핵심 모듈·파일별 정밀 분석 — `fast_quant.py`

### 3.1 균등 양자화 함수

#### `quant_activation(x, bit, act_scale)` (fast_quant.py:9-13)
```python
n = 2**(bit-1) - 1
aint = (x/act_scale).round().clamp(-n-1, n)
x = aint * act_scale
```
- **대칭(symmetric) per-tensor 활성 양자화**. zero-point 없음(=0). `act_scale`은 외부에서 결정(percentile search 또는 absmax). qmin=-n-1, qmax=n.

#### `quant_weight(w, bit, mode="channel_wise", symmetric=True)` (fast_quant.py:15-24)
```python
n = 2**(bit-1) - 1
scale_channel_wise = w.abs().max(dim=1, keepdim=True)[0] / n   # 행(출력채널)별
wint = (w/scale_channel_wise).round().clamp(-n-1, n)
wq = wint * scale_channel_wise
```
- **대칭 per-output-channel 가중치 양자화** (dim=1 기준 max). symmetric=False 경로는 `NotImplementedError` (fast_quant.py:21-22).

### 3.2 activation scale(clipping) 탐색 — `percentile_search` (fast_quant.py:26-59)
- raw activation `x`에 대해 `clip_value = absmax/search_space*ii`를 큰 값→작은 값으로 훑으며(fast_quant.py:39-40):
  - `act_scale = clip_value/(2^{bit-1}-1)` (fast_quant.py:42)
  - clip한 입력을 양자화 후 `F.linear`로 출력 `z` 계산, raw 출력 `z0`과 **MSE** `((z-z0)^2).mean()` 비교 (fast_quant.py:43-44).
  - MSE 최소가 되는 `act_scale` 선택 (fast_quant.py:45-51).
- 즉 **출력 재구성 MSE를 최소화하는 clipping point**를 grid search(기본 search_space=1000, fast_quant.py:66). → outlier를 적절히 잘라내는 percentile clipping의 역할.

### 3.3 양자화 forward (monkey-patch) — `quant_forward(self, x)` (fast_quant.py:62-141)
`nn.Linear.forward`를 대체. 3가지 상태로 분기:

#### (A) `self.clip_search` — percentile 캘리브레이션 1회 (fast_quant.py:64-69)
- raw 출력 `z0 = F.linear(x, W, b)` 계산 후 `percentile_search`로 `act_scale` 결정(fast_quant.py:65-66), 플래그 끔. 이후 양자화 출력 반환.

#### (B) `self.noisy_search` — ★ noisy bias 탐색 1회 (fast_quant.py:70-134)
`with_noisy_quant`일 때:
- **noisy bias 초기 샘플** (fast_quant.py:74):
  ```python
  noisy_bias = (torch.randn_like(x[:1,:1,:])*2 - 1) * self.act_scale
  ```
  → 채널 차원(`[1,1,C]`) 길이의 **고정 무작위 벡터**, 스케일은 `act_scale` 단위. (`randn*2-1`은 가우시안을 대략 [-?] 범위로 변형; 핵심은 채널별 고정 노이즈.)
- **(B-1) mean 탐색** (`search_mean`, fast_quant.py:78-99):
  - candidate `= act_scale * ii/200`, `ii ∈ [-200, 200)` (fast_quant.py:84).
  - `xq = quant(x + candidate); xq -= candidate; zq = F.linear(xq, W, b)` vs `z = F.linear(x, W, b)`의 **MSE** 최소화 (fast_quant.py:85-97).
  - 최적 candidate를 `noisy_bias`(스칼라 mean)로 저장.
- **(B-2) range(노이즈 스케일) 탐색** (`search_noisy`, fast_quant.py:104-125):
  - candidate `= best_noisy_mean + noisy_bias * ii/1000`, `ii ∈ [0, 2000)` (fast_quant.py:110) → mean에 무작위 벡터를 점점 키워가며 더함.
  - 동일하게 `quant(x+candidate) - candidate` 후 출력 MSE 최소화 (fast_quant.py:111-124).
  - 최적 `candidate`(= mean + scaled noise vector)를 최종 `self.noisy_bias`로 확정.
- `self.add_noise=True` 설정 후, 확정된 noisy bias로 한 번 양자화 실행 (fast_quant.py:128-131):
  ```python
  x = quant_activation(x + self.noisy_bias, bit, act_scale)
  x -= self.noisy_bias
  ```
- `with_noisy_quant`가 아니면 **트릭 없는 vanilla 양자화** (fast_quant.py:132-134).

#### (C) 정상 추론 (fast_quant.py:135-141)
```python
if self.add_noise:  x = x + self.noisy_bias
x = quant_activation(x, bit, act_scale)
if self.add_noise:  x = x - self.noisy_bias
return F.linear(x, self.weight, self.bias)
```
- **핵심 동작**: `Quant(x + N) - N`. N은 고정이므로 선형성에 의해 `F.linear(Quant(x+N)-N, W, b) = F.linear(Quant(x+N), W, b) - F.linear(N, W, b)` → `F.linear(N,W,b)`는 상수이므로 추론 시 **bias로 흡수 가능**(논문 주장). 즉 추가 런타임 비용 없음.

### 3.4 모델 양자화 적용 — `fast_quant(model, ...)` (fast_quant.py:143-165)
- `model.named_modules()`를 돌며 `nn.Linear`이고 `name != "head"`인 모듈만 처리 (fast_quant.py:146):
  - `module.bit = bit`, 가중치를 `quant_weight`로 즉시 양자화하여 `module.weight.data`에 덮어씀(원본은 `original_weight`에 보존) (fast_quant.py:148-152).
  - `act_scale=None`, `add_noise=False` 초기화 (fast_quant.py:152-153).
  - `with_noisy_quant`면 `clip_search=percentile`, `noisy_search=(search_mean or search_noisy)` 등 플래그 세팅 (fast_quant.py:155-159).
  - **`module.forward = MethodType(quant_forward, module)`** — forward를 양자화 버전으로 교체 (fast_quant.py:163).
- **head(분류기)는 양자화 제외** (fast_quant.py:146).

> **주의**: 가중치는 `fast_quant` 호출 즉시 양자화되지만(per-channel 대칭), **conv(patch embed)·LayerNorm·GELU·softmax·attention의 matmul(q@k, attn@v)은 양자화 대상이 아님**. 오직 `nn.Linear`만 대상. → activation 양자화도 Linear 입력에 한정.

### 3.5 검증/캘리브레이션 흐름 — `validate.py`
- 양자화 인자 (validate.py:154-164): `--quant`, `--with_noisy_quant`, `--calib_root`, `--calib_num`, `--percentile`, `--search_mean`, `--search_noisy`, `--bitwidth`(기본 6).
- `model = create_model(...)`(timm) 후 `fast_quant(model, bit, with_noisy_quant, percentile, search_noisy, search_mean)` 적용 (validate.py:210-223).
- **캘리브레이션** (validate.py:341-367):
  1. calib 데이터 1 배치(`calib_num`장)를 로드/캐시(`calib_data_{n}.pt`) (validate.py:344-355).
  2. `--percentile`이면 먼저 `model(input)` 1회 → 각 Linear의 `clip_search` 경로로 act_scale 결정 (validate.py:358-362).
  3. `--search_noisy or --search_mean`이면 다시 `model(input)` 1회 → noisy bias 탐색 경로 실행 (validate.py:363-366).
- 이후 ImageNet val에서 top-1/5 평가.

---

## 4. 알고리즘 / 수식

### 4.1 noisy bias로 분포 변형 후 양자화
고정 noisy bias `N`(채널별 벡터, 캘리브레이션으로 결정)에 대해:

```
정상 추론:   y = W · ( Quant(x + N) - N ) + b
            = W · Quant(x + N)  -  W·N  +  b      (선형성)
            = W · Quant(x + N)  +  b'             (b' = b - W·N, 사전계산 가능)
```
(fast_quant.py:136-141)

- `Quant(·)`는 대칭 균등 양자화: `Quant(u) = round(u/s)·s, clamp[-2^{b-1}, 2^{b-1}-1]`.
- **직관**: outlier가 큰 `x`를 그대로 양자화하면 큰 scale 때문에 일반값의 해상도가 떨어진다. `x+N`을 양자화하면 양자화 오차 `Quant(x+N)-(x+N)`가 입력에 상관없는(decorrelated) 노이즈처럼 작용 → `N`을 빼면 평균적으로 `E[Quant(x+N)-N] ≈ x`이면서 분산이 감소(논문 정리). 즉 **dithering(디더링) 효과**.

### 4.2 캘리브레이션 목표 (모두 출력 재구성 MSE)
1. **clipping/act_scale**: `s* = argmin_s || F.linear(Quant_s(clip(x)), W) - F.linear(x, W) ||²` (fast_quant.py:43-44).
2. **noisy bias mean** `μ`: `μ* = argmin_μ || F.linear(Quant(x+μ)-μ, W) - F.linear(x, W) ||²` (fast_quant.py:85-89).
3. **noisy bias range** `α`: `N = μ* + α·n₀` (n₀=고정 무작위 벡터), `α* = argmin_α || F.linear(Quant(x+N)-N, W) - F.linear(x, W) ||²` (fast_quant.py:110-115).
- 모두 grid search (mean 400개, range 2000개 후보).

---

## 5. 학습/평가 파이프라인

- **데이터셋**: ImageNet val(평가). **캘리브레이션은 train의 소수 샘플**(`--calib_num`, 예: 256장) (README.md:76, validate.py:157-159).
- **모델**: timm의 `vit_base_patch16_224` 등 (README.md:30, 53). 사전학습 가중치 사용.
- **명령어** (README.md:28-107):
  ```bash
  # 1) FP baseline
  python validate.py /data/imagenet/val/ --model vit_base_patch16_224 --pretrained
  # 2) vanilla 6-bit (정확도 급락: 85.1→64.6)
  python validate.py ... --pretrained --quant
  # 3) 6-bit NoisyQuant (회복: 83.28)
  python validate.py ... --pretrained --quant --with_noisy_quant \
     --calib_root /data/imagenet/train --calib_num 256 \
     --percentile --search_mean --search_noisy --bitwidth 6
  # 재현 일괄: bash run.sh $GPU $MODEL $CALIB_NUM $IMAGENET_DIR
  ```
- **보고 정확도** (README.md:36-96): ViT-B FP 85.1% → vanilla 6-bit 64.6% → **NoisyQuant 6-bit 83.28%** (약 +18.7%p 회복).

---

## 6. 의존성

- `torch` (>=2.0.1 권장), `torch.nn.functional`.
- **`timm` (0.9.8 권장)** — 모델 생성/데이터 로딩/validate 보일러플레이트의 핵심 (README.md:14-16, validate.py:25-29).
- `tqdm`(진행바), (optional) `apex`/native AMP.
- 양자화는 순수 PyTorch fake-quant. 외부 양자화 라이브러리 없음.

---

## 7. 강점 / 한계 / 리스크

### 강점
- **극단적 단순·경량**: 핵심이 단일 파일, forward monkey-patch로 어떤 timm Linear 모델에도 즉시 적용. README.md:100 "very easy to understand".
- **런타임 오버헤드 0**: noisy bias가 고정 → bias로 흡수 가능. 추론 시 추가 연산 없음(논문/4.1).
- **저비트 outlier 완화에 효과적**: 6-bit에서 +18.7%p 정확도 회복(README.md). FPGA 저비트 양자화에 직접적으로 유익한 성질.
- 캘리브레이션이 출력 MSE 기반 grid search라 안정적.

### 한계
- **Linear만 양자화** (fast_quant.py:146): conv(patch embed), LayerNorm, GELU, softmax, attention의 두 matmul(q@kᵀ, attn@v)은 미양자화 → **end-to-end 정수 추론 아님**. 실제 HW 가속 시 matmul/softmax는 별도 처리 필요.
- 캘리브레이션 grid search 후보가 많음(mean 400 + range 2000회 × layer수 × forward) → **캘리브레이션 시간 비용 큼**(추정).
- 활성 양자화가 **per-tensor 대칭**(zero-point 없음) — 비대칭 분포에 비효율적일 수 있음(단 noisy bias mean이 이동을 일부 흡수).
- noisy bias 초기 샘플이 단일 무작위 시드에 의존(fast_quant.py:74) → 시드 민감성 가능(추정).

### 리스크
- 4-bit 이하에서의 효과는 본 repo 코드/README로 **확인 불가**(6-bit 위주).
- monkey-patch 방식은 모델 구조가 표준 `nn.Linear`가 아니면 적용 누락 위험.

---

## 8. 우리 프로젝트 관점 시사점 (ViT/Transformer FPGA 가속기 + XR 시선추적, 추정)

> 프로젝트 성격은 "HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적"으로 **추정**.

- **저비트 FPGA 정확도에 직결**: FPGA 가속기는 자원/전력 제약상 저비트(W6/A6 이하)를 선호. NoisyQuant의 outlier 완화는 **저비트에서 정확도 유지**라는 우리 핵심 요구와 정확히 부합.
- **추가 HW 비용 0이라는 점이 매력**: noisy bias를 사전계산 bias로 흡수하면, 가속기 데이터패스(MAC 어레이) 변경 없이 정확도만 개선 가능. 즉 **컴파일/캘리브레이션 단계에서 끝나는 SW 트릭** → HG-PIPE 같은 파이프라인 가속기에 비침투적으로 결합 가능.
- **결합 전략(추정)**: PSAQ-ViT(data-free 캘리브레이션) + NoisyQuant(outlier 완화)를 조합하면 "데이터 없이 + 저비트 정확도 유지"라는 XR 엣지 배포 시나리오에 강력. 단 둘 다 Linear 중심이므로 attention matmul/softmax/LayerNorm은 별도 정수화(I-ViT류) 필요.
- **HW 비트할당 참고**: percentile_search의 출력-MSE 기반 clipping은 우리 가속기의 layer별 scale 선정 알고리즘 reference로 재사용 가능.
- **주의**: 본 repo는 Linear만 양자화하므로, 우리가 attention/softmax까지 정수 가속하려면 noisy bias 아이디어를 **그 연산들로 확장**하는 추가 연구가 필요(논문 범위 밖, 추정).

---

## 9. 근거 표기 정리

- 모든 동작은 파일:라인 표기(예: fast_quant.py:136-141).
- **추정**: (a) 캘리브레이션 시간 비용; (b) noisy bias 시드 민감성; (c) 4-bit 효과; (d) 우리 프로젝트 적용 전략.
- **확인 불가**: 4-bit 이하 정확도(코드/README에 6-bit 위주). validate.py의 timm 표준 부분은 양자화 관련 인자만 정독.
