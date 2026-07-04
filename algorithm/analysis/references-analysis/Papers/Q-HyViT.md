# Q-HyViT: Post-Training Quantization of Hybrid Vision Transformers with Bridge Block Reconstruction for IoT Systems

## 1. 서지정보
- **제목**: Q-HyViT: Post-Training Quantization of Hybrid Vision Transformers with Bridge Block Reconstruction for IoT Systems
- **저자**: Jemin Lee, Yongin Kwon, Misun Yu, Jeman Park (ETRI), Sihyeong Park (KETI), Hwanjun Song (KAIST, 교신저자)
- **연도**: 2024 (arXiv submit/5601736, 17 May 2024; "Journal of LaTeX Class Files" 템플릿 - IEEE 저널 투고본). 과제 지원: IITP RS-2023-00277060 (open edge AI SoC HW/SW 플랫폼).
- **학회/저널**: IEEE 저널 투고본 추정 (IoT Journal 계열 추정, 정확한 게재처 표기 없음) - **확인 불가**
- **원본파일명**: q-hyvit-paper.pdf

## 2. 문제정의·배경
- ViT는 CNN을 대체했으나 높은 연산량으로 모바일/IoT 배치 어려움. 이를 위해 **hybrid ViT**(conv + transformer 결합, 선형복잡도 attention)와 **양자화**가 제안됨.
- 모바일 최적 가속엔 양자화 + 효율 hybrid 구조 통합 필요하나, **기존 PTQ를 hybrid ViT에 적용하면 심각한 정확도 하락** - 4가지 도전과제(C1~C4) 때문:
  - **C1**: 매우 동적인 activation range (채널별 분포 상이)
  - **C2**: bridge block의 zero-point overflow (conv↔transformer 전이부, local↔global 표현 격차)
  - **C3**: 다양한 normalization (BatchNorm/LayerNorm/GroupNorm 혼용)
  - **C4**: <5M 파라미터 소형 모델 (residual 적어 양자화에 취약)
- hybrid ViT 양자화를 다룬 선행연구 부재 - **최초로** 효율 hybrid ViT(MobileViTv1/v2, Mobile-Former, EfficientFormerV1/V2) 양자화.

## 3. 핵심기여
- hybrid ViT 양자화의 **4대 고유 도전과제(C1~C4) 규명**.
- **Q-HyViT**: Hessian 기반 통합 PTQ. bridge/non-bridge 레이어별로 **granularity(channel/layer-wise), scheme(symmetric/asymmetric), scaling factor를 자동 선택**.
- 기존 PTQ(EasyQuant, FQ-ViT, PTQ4ViT, RepQ-ViT)를 5종 hybrid ViT에 확장 적용·직접 비교.
- 평균 정확도 향상: **8-bit +17.73%, 6-bit +29.75%** (vs 기존 PTQ). full quantization(softmax+norm 양자화) 시 8-bit FQ-ViT 대비 **+43.63%** SOTA.

## 4. 방법론
### Uniform 양자화 기본식
- $x_q = Q(x_r) = \text{clip}(\text{round}(\frac{x_r}{\Delta_x}) + zp, \min, \max)$, $\Delta_x$=scaling factor, $zp$=zero-point(asymmetric만).
- MHSA 양자화: $\bar{Q}=\bar{w}_Q\bar{E}$, $\bar{K}=\bar{w}_K\bar{E}$, $\bar{V}=\bar{w}_V\bar{E}$. quant-attention $= \text{quant-softmax}(\frac{\bar{Q}_h \bar{K}_h^T}{\sqrt{d_k}})\bar{V}_h$.

### Bridge Block (Table I)
- MobileViTv1/v2: conv+reshape로 Transformer 입력 차원 정렬 (3개). Mobile-Former: local↔global 양방향 전이 (15개). EfficientFormerV1: 4D→3D meta block 변환 (1개). EfficientFormerV2: 3,4 stage local/global 전이 (2개).

### Hybrid Reconstruction Error Minimization (핵심)
- 2차 Taylor 전개 기반 reconstruction(BRECQ 계열): $E[L(\hat{w})] - E[L(w)] \approx \epsilon^T \bar{g}^{(w)} + \frac{1}{2}\epsilon^T \bar{H}^{(w)}\epsilon$. well-converged 시 gradient 무시, $\epsilon^T \bar{H}^{(x)}\epsilon \approx \Delta O^T \bar{H}^{(O)}\Delta O$.
- **bridge 여부로 reconstruction 목적 분기**:
  $$O^{bb} = \begin{cases} w_n^{bb} w_{n-1}^{bb} \cdots w_1^{bb} x^{bb}, & \text{layer가 bridge block 내부} \\ w^\ell x^\ell, & \text{otherwise} \end{cases}$$
  (bridge면 선행 레이어 전부 포함하여 종속성 고려.)
- 최적화 목적: $\min_{\Delta,g,s} E[\Delta O^{(bb),T}, H^{O(bb)} \Delta O^{(bb)}] \approx \min E[\Delta O^{(bb),T}, \text{diag}((\frac{\partial L}{\partial O_1^{(bb)}})^2, \cdots) \Delta O^{(bb)}]$.
- 탐색: scaling factor $\Delta \in [1, 100]$, granularity $g \in [\text{layer, channel}]$, scheme $s \in [\text{asym, sym}]$. 후보 범위 $[\alpha\frac{\max|w|}{2^{k-1}}, \beta\frac{\max|w|}{2^{k-1}}]$.

### C2 (zero-point overflow) 핵심 분석
- $-128 > q_{min} - \frac{r_{min}}{s}$ → $0 < \frac{r_{min}}{s}$: $r_{min}>0$이면 zero-point가 8-bit 범위(-128~127) 초과·clamp → 정확도 급락. **layer-wise 양자화로 음수 포함시켜 해결**(전이 conv 출력이 모두 양수일 때 채널별 asym이 문제).

### Algorithm 1 (calibration → optimization)
- 3개 while loop: (1) FP forward로 $y^{fp32}$·중간출력 저장, (2) default 양자화 후 backprop으로 layer/bridge gradient 저장, (3) Eq.(9)로 granularity/scheme/scaling 3-iteration 최적화.

## 5. 실험
- **데이터셋/메트릭**: ImageNet-1K Top-1 accuracy. 5종 hybrid ViT(MobileViTv1/v2, Mobile-Former, EfficientFormerV1/V2). calibration 32 images(Q-HyViT). A100-80G.
- **Partial quant (Table II, softmax/LN은 FP)**:
  - MobileViTv1-xxs(1.3M) W8A8: FP32 69.0 → Q-HyViT **68.20** (PTQ4ViT 37.75, RepQ-ViT 1.85, EasyQuant 36.13).
  - sub-5M 모델(xxs/xs/050/075/100/26m/52m/96m/S0)에서 8-bit 정확도 손실 <1%.
  - 평균 개선: 8-bit EasyQuant 대비 +9.54%, PTQ4ViT 대비 +7.09%. 6-bit는 +43.39%, +8.65%.
- **Full quant (Table III, softmax+norm 양자화)**: MobileViTv1-xxs FP32 68.91 → Q-HyViT **67.20** (FQ-ViT 0.1). 평균 FQ-ViT 대비 **+43.63%**.
- **Ablation (Table IV)**: scaling factor만으론 PTQ4ViT 대비 미미, granularity 추가 시 큰 향상(MobileViTv1-xxs 44.37→59.50), scheme까지 추가 시 68.20. granularity가 핵심.
- **Running time (Fig.9)**: Q-HyViT 평균 523s(reconstruction 수행으로 최장), RepQ-ViT 159s(reconstruction 없음). PTQ는 offline 1회라 무시 가능.

## 6. 강점/한계
- **강점**: hybrid ViT 양자화 최초 체계화. bridge block 인지 자동 granularity/scheme/scaling 선택. sub-5M 모델에서 8-bit <1% 손실. full quantization(integer-only, softmax/norm 포함)로 **off-chip 데이터 이동 감소 → HW 설계 친화적**(C3 동기). MobileViTv1/v2, Mobile-Former, EfficientFormerV1/V2 광범위 검증.
- **한계**: reconstruction으로 양자화 시간 최장(523s, 단 offline). 극소형 모델(<5M)은 여전히 어려운 과제로 남음. 6-bit에서 일부 모델(Mobile-Former-26m 51.06 등) 손실 큼. QAT 미적용(PTQ만).

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **직접 연관 (매우 높은 가치)**: Q-HyViT는 우리 프로젝트의 **ViT 양자화 코어 기술** - 본 repo(q-hyvit) 코드가 함께 존재. hybrid ViT(MobileViT/EfficientFormer 등)를 INT8/INT6로 정확도 손실 최소화하며 양자화 → FPGA on-device ViT 가속의 핵심 전처리.
- **직접 연관**: **full quantization(softmax/LayerNorm/GroupNorm까지 정수화)**은 FPGA에서 off-chip dequant 데이터 이동을 제거 → 저지연·저전력 데이터패스에 직접 부합(논문 C3 동기와 정확히 일치). 우리 FPGA 가속기 설계 시 integer-only 데이터패스 정당화 근거.
- **직접 연관**: bridge block의 zero-point overflow(C2) 분석은 hybrid ViT를 FPGA INT 데이터패스에 매핑할 때 반드시 고려할 HW 이슈 - layer-wise vs channel-wise 선택이 정확도뿐 아니라 zero-point 회로 설계에 직결(추정).
- **추정**: granularity/scheme 자동선택은 호스트 측 calibration 단계 → FPGA는 결정된 per-layer 설정(channel/layer-wise, sym/asym, scaling factor)을 고정 회로/LUT로 구현. mixed granularity는 HW 제어 복잡도 증가 가능.
- **시선추적 연결 (추정)**: XR 시선추적에 ViT backbone(EX-Gaze, BRAT 등)을 쓸 경우 Q-HyViT로 양자화하여 FPGA 배치 → 시선추적 정확도 유지 + 저지연 가속. 본 프로젝트가 ViT 양자화와 이벤트 시선추적을 결합하려는 의도로 보임(추정).
- 저자가 ETRI/KETI/KAIST(한국)로 IITP "open edge AI SoC" 과제 지원 - 우리 FPGA on-device 가속 목표와 기관/과제 성격 정합(추정).

## 8. 근거표기
- 1~6장 수치/수식/구조/알고리즘은 본문(arXiv submit/5601736) 직접 확인.
- 정확한 게재 저널은 "Journal of LaTeX Class Files" 플레이스홀더로 표기되어 **확인 불가**(IEEE IoT Journal 계열 추정).
- 7장 FPGA 매핑·시선추적 결합은 **추정** (논문은 GPU(A100) PTQ 실험·ImageNet 정확도만 보고, FPGA 구현·시선추적 미포함. IoT HW 절은 8-bit 가속기 일반 언급 수준).
