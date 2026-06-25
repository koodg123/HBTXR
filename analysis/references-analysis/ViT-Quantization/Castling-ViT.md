# Castling-ViT 정밀 분석

> 분석 대상: `REF/ViT-Quantization/Castling-ViT`
> 작성 기준: 실제 소스(`attention.py`) 및 `README.md` 라인 근거. 근거표기는 9절 참조.

## 1. 개요 (목적 / 원논문 / 핵심아이디어)

- **원논문**: *Castling-ViT: Compressing Self-Attention via Switching Towards Linear-Angular Attention During Vision Transformer Inference* (CVPR 2023), You et al. (Georgia Tech EIC). (`README.md:1-8`)
- **목적**: ViT의 self-attention 연산 복잡도 `O(N^2 d)`(N=토큰 수)를 추론 시 **선형 복잡도 `O(N d^2)`**로 낮추되, 학습 시에는 softmax attention의 표현력을 유지한다.
- **핵심아이디어 (Castling = "성을 바꾸다" 메타포)**:
  1. **Linear-Angular Attention**: softmax 대신 각도(angular) 유사도 기반 커널을 사용. Q, K를 L2-정규화하여 코사인 유사도를 근사하고, 결합법칙(associativity)으로 `K^T V`를 먼저 계산해 선형 복잡도를 달성.
  2. **Switching (학습→추론 전환)**: 학습 중에는 softmax 기반 sparse attention 항(`sparse_reg`)을 보조로 두고, 추론 시 이 항을 제거(switch off)하여 순수 선형 attention만 남긴다. 이것이 "Castling"의 핵심. (단, 본 미니어처 릴리스에서는 switching 스케줄 자체는 코드에 없고 `sparse_reg` 플래그로 두 경로만 노출됨 — `attention.py:61,73`.)
  3. **Depthwise Conv 잔차항(DWConv)**: linear attention이 놓치는 지역적(local) 정보를 보강하는 1D depthwise conv를 value에 적용.

> **중요**: 본 디렉토리는 **저자가 명시한 "비공식 미니어처 릴리스"**로, attention 블록 핵심만 담고 있다. 분류/세그멘테이션/검출 전체 코드베이스는 MobileVision / Mask2Former / PicoDet 위에 구축되어 별도 존재(미포함). (`README.md:12,24-31`)
- **양자화 관련**: 이 repo 자체에는 **명시적 양자화(PTQ/QAT) 코드가 없다**(확인: `attention.py`에 quantizer/observer/scale 없음). 본 repo는 "ViT-Quantization" 폴더에 속하지만, 그 가치는 **연산량 자체를 선형화하여 가속기 친화 구조를 제공**하는 데 있다(8절 참조).

## 2. 디렉토리 구조

### 자체 소스
```
Castling-ViT/
├── README.md          # 논문 메타 + 사용법 (python attention.py)
├── attention.py       # 유일한 핵심 소스. LinAngularAttention, MatMul 클래스
└── castling-vit.png   # 개념도 (이미지)
```
- `attention.py` 1개 파일(약 100라인)이 전부. (`Glob *.py` 결과 = `attention.py` 단일)

### 제외 항목
- 외부 코드베이스(분류=MobileVision@Meta, 세그=Mask2Former, 검출=PicoDet/Picodet_Pytorch, 사전학습=MAE/LeViT)는 본 repo에 미포함 → **외부 의존성**. (`README.md:24-31`)
- `.git` 등 VCS 메타데이터 제외.

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `MatMul` (attention.py:7-12)
- 단순 `a @ b` 래퍼 `nn.Module`. 주석(`attention.py:6`)에 "Q @ K를 쓰면 FLOPs 계산이 틀릴 수 있다"고 명시 — 즉 **FLOPs 프로파일러가 matmul을 모듈 단위로 후킹**하기 위한 래퍼. 가속기 관점에서 각 matmul을 독립 연산 노드로 분리·계측·치환하기 좋은 구조.

### 3.2 `LinAngularAttention` (attention.py:14-86) — 핵심 클래스

**생성자 (`__init__`, 14-50)**
- `num_heads`, `head_dim = in_channels // num_heads`, `self.scale = head_dim**-0.5` (`28-29`).
- `self.qkv = nn.Linear(in_channels, in_channels*3)` — Q/K/V 동시 투영 (`32`).
- 네 개의 `MatMul` 인스턴스: `kq_matmul`, `kqv_matmul` (선형 경로), 그리고 `sparse_reg`일 때만 `qk_matmul`, `sv_matmul` (softmax 보조 경로) (`37-41`).
- **DWConv 잔차 (`43-50`)**: `nn.Conv2d(num_heads, num_heads, kernel_size=(res_kernel_size,1), padding=(res_kernel_size//2,0), groups=num_heads, bias=False)`. head 차원을 conv 채널로 보고, 토큰 축(길이 L)에 대해 1D(9×1) depthwise conv를 수행 → **지역적 토큰 간 상호작용 보강**. `res_kernel_size=9` 기본.

**forward (`52-86`)**
- 입력 `x: (N, L, C)`. `qkv` 투영 후 `(3, N, num_heads, L, head_dim)`로 reshape·permute, `q,k,v` unbind (`53-59`).
- **Sparse 보조 경로 (학습용, `61-65`)**: `sparse_reg=True`일 때만 — `attn = qk_matmul(q*scale, k^T)` → `softmax` → 임계값 `0.02`로 마스크 → `sparse = mask * attn`. 즉 **softmax attention의 희소 잔차**만 추가 보정. (임계값은 코드베이스별 조정 필요 주석, `attention.py:64`)
- **Linear-Angular 핵심 (`67-80`)**:
  - `q = q / q.norm(dim=-1, keepdim=True)`, `k = k / k.norm(...)` — **L2 정규화** → 내적이 코사인(=각도) 유사도가 됨 (`67-68`).
  - `dconv_v = self.dconv(v)` — value에 DWConv 잔차 (`69`).
  - `attn = kq_matmul(k.transpose(-2,-1), v)` — **핵심 결합법칙 변경**: `K^T V`를 먼저 계산 (`(head_dim × L)·(L × head_dim) = head_dim × head_dim`), N(=L)에 대해 **선형**. (`71`)
  - 출력: `x = 0.5*v + (1/π) * kqv_matmul(q, attn)` (`80`). `q @ (K^T V)` 형태 → 선형 복잡도. `0.5*v`는 항등 근사항, `1/π` 계수는 angular kernel 근사 상수.
  - `sparse_reg`이면 `sv_matmul(sparse, v)`(softmax 경로) 추가 (`74-78`).
- **후처리 (`81-85`)**: `x = x / x.norm(...)` 정규화 → `x += dconv_v`(DWConv 잔차 합산) → reshape → `proj` 선형 → dropout.

### 3.3 양자화 관점
- **PTQ/QAT 없음**, observer/scale/zero-point 없음 (확인: 파일 전체에 해당 키워드 부재). 본 repo는 "효율화(연산 치환)" 카테고리지 양자화 기법 자체는 아니다.

## 4. 알고리즘 / 수식

**표준 softmax attention** (참고): `Attn(Q,K,V) = softmax(QK^T/√d) V`, 복잡도 `O(N^2 d)`.

**Linear-Angular attention (본 구현, attention.py:67-80)**:
- Q, K를 단위 정규화: `q̂ = q/‖q‖`, `k̂ = k/‖k‖`.
- 결합법칙 적용: `Out ≈ 0.5·V + (1/π)·Q̂ (K̂^T V)`.
  - 먼저 `S = K̂^T V ∈ R^{d×d}` 계산 → `O(N d^2)`.
  - 다음 `Q̂ S ∈ R^{N×d}` → `O(N d^2)`.
  - **전체 복잡도 `O(N d^2)`** (vs softmax `O(N^2 d)`). N≫d인 고해상도/긴 토큰열에서 큰 이득.
- DWConv 잔차: `Out += DWConv_{9×1}(V)` (지역 정보).
- (학습 전용) sparse 보정: `Out += (mask·softmax(QK^T/√d)) V`, mask = `attn > 0.02`.

**각도 유사도 동기**: 정규화 후 내적 `q̂·k̂ = cos θ`. 논문은 softmax(지수 커널)를 angular 커널 + 1차 근사로 대체하여 선형화함 (`0.5 + (1/π)·cosθ`류 근사; 코드의 `0.5*v`, `1/π` 상수가 그 흔적).

## 5. 학습 / 평가 파이프라인

- 본 미니어처: `python attention.py` 실행으로 shape 검증만 (`README.md:14-16`, `attention.py:88-99`). 입력 예시 `torch.randn(1,196,256)` (196=14×14 패치, 256 채널).
- 실제 태스크 파이프라인은 외부 코드베이스 사용:
  - 분류: MobileVision@Meta. 세그: Mask2Former(+MAE 사전학습). 검출: PicoDet/Picodet_Pytorch(+LeViT 사전학습). (`README.md:24-31`) → **데이터셋/명령어는 본 repo에 없음(확인 불가, 외부 의존)**.

## 6. 의존성

- `torch`, `torch.nn`, `torch.nn.functional`, `math` (`attention.py:1-4`). 그 외 없음.
- (실제 재현 시) 외부: MobileVision, Mask2Former, MAE, PicoDet, LeViT — **외부 의존성**.

## 7. 강점 / 한계 / 리스크

**강점**
- **선형 복잡도 attention**: N에 선형 → 긴 시퀀스/고해상도에서 메모리·연산 급감. FPGA·엣지에 매우 유리.
- `K^T V` 선계산은 **고정 크기 `d×d` 중간 텐서** → 토큰 수에 무관한 버퍼 → 하드웨어 버퍼 사이징 단순화.
- softmax 제거(추론 시) → 지수/정규화 비선형 회피, 하드웨어 친화.
- 코드가 100라인으로 명료, MatMul 모듈화로 연산 노드 분리 용이.

**한계 / 리스크**
- **미니어처 릴리스**: switching 스케줄, 학습 루프, 사전학습 가중치 미포함 → 단독으로 정확도 재현 불가(확인 불가).
- 양자화 코드 없음 → 본 repo만으로는 "정량화" 분석 대상이 아님.
- L2 norm / `x.norm` 연산이 다수(`67,68,81`) → 하드웨어에서 제곱근·역수 LUT 필요(softmax보단 가벼우나 비선형 잔존).
- DWConv 잔차(`9×1`)는 토큰 순서(공간 구조)를 가정 → 시퀀스 재배열 시 의미 달라짐.

## 8. 우리 프로젝트(ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> 전제: 본 연구는 HG-PIPE 계열 ViT FPGA 가속기 + XR 시선추적으로 **추정**.

- **선형 attention의 하드웨어 가치(핵심)**: `K^T V` 선계산 패턴은 **시스톨릭 어레이/누산기로 매핑하기 이상적**. 토큰 수 N에 무관한 `d×d` 누산 버퍼만 있으면 됨 → HG-PIPE식 파이프라인에서 **attention 버퍼를 토큰 길이와 독립적으로 고정** 설계 가능. XR 시선추적의 고프레임레이트(저지연) 요구에 직접 부합.
- **softmax 제거 = 가속기 단순화**: 지수 LUT/누적/정규화 제거 가능. 다만 본 구조는 `‖·‖`(L2 정규화) 비선형이 남으므로, 이를 **역제곱근(rsqrt) LUT 또는 고정점 근사**로 치환하는 것이 우리 RTL/HLS 단계의 과제(추정).
- **연산 치환 아이디어**: `0.5*v + (1/π)*q(k^T v)` 구조는 곱셈-누산(MAC)만으로 표현 → 양자화(INT8/INT4)와 결합 시 softmax 기반 대비 정밀도 손실 경로가 단순(비선형 부재). 우리 양자화 가속기에 **"softmax-free attention 데이터패스"** 옵션으로 채택 검토 가치 높음.
- **DWConv 잔차**: depthwise(`groups=num_heads`)라 곱셈량 적음 → 가속기에서 작은 1D conv 엔진 또는 line-buffer로 저비용 구현 가능. 지역 정보 보강을 저비용으로 얻는 패턴.
- **시선추적 적합성**: 시선추적은 토큰(패치) 수가 많고 저지연이 생명 → N-선형 attention은 본질적으로 적합. 단, 정확도-효율 trade-off(switching 스케줄)는 외부 코드 필요 → 우리가 학습 파이프라인을 재구성해야 함(추정).
- **양자화와의 결합 제안**: 이 repo는 양자화가 없으므로, **PTQ4ViT/FQ-ViT의 양자화 기법을 Castling의 선형 attention 위에 얹는** 조합이 우리 가속기에 가장 매력적(연산량 선형 + 비트폭 저감 동시). 이는 본 세트 내 repo 간 교차 활용 포인트.

## 9. 근거 표기

- **확인**: `attention.py` 전체 라인 직접 분석(MatMul 7-12, LinAngularAttention 14-86, 메인 88-99). README 메타·외부 의존 24-31. 양자화 코드 부재는 파일 전수 확인.
- **추정**: 우리 프로젝트(HG-PIPE/XR 시선추적) 관련 시사점은 사용자가 제시한 연구 맥락 기반 추정.
- **확인 불가**: switching 스케줄·학습 하이퍼·정확도 수치(외부 코드베이스 의존, 본 repo 미포함).
