# TDTracker: Exploring Temporal Dynamics in Event-based Eye Tracker

## 1. 서지정보
- **제목**: Exploring Temporal Dynamics in Event-based Eye Tracker (TDTracker)
- **저자**: Hongwei Ren*, Xiaopeng Lin*, Hongxiang Huang*, Yue Zhou, Bojun Cheng† (*동등기여, †교신저자)
- **소속**: The Hong Kong University of Science and Technology (Guangzhou), MICS Thrust
- **연도**: 2025 (arXiv:2503.23725v1, 31 Mar 2025)
- **학회/발표**: CVPRW 2025 Event-Based Eye Tracking Challenge 관련 논문 (챌린지 3위 수상)
- **원본파일명**: TDTracker.pdf

## 2. 문제정의·배경
- XR/AR/VR 웨어러블에서 고속·고정밀 시선추적은 필수이나, 프레임 기반 센서는 시간해상도 한계로 saccade(>300°/s), 급가속(최대 24,000°/s²) 등 빠른 안구 동역학을 정확히 포착하지 못함.
- 프레임 기반 시스템은 45~81 ms 추적 지연이 있어 kHz급 샘플링에 부적합하고, 높은 샘플링은 전력·대역폭 부담을 야기.
- 이벤트 카메라는 μs 단위 시간해상도, 저전력, 고동적범위, 비동기 동작으로 빠른 안구 움직임 포착에 적합하나, 이벤트 데이터의 **시간 동역학(temporal dynamics)** 추출·활용이 핵심 과제.
- 기존 RNN/LSTM/GRU는 장기 시퀀스에서 gradient 소실·폭주 및 정보 망각 문제, Mamba는 고정 hidden-state 차원 한계.

## 3. 핵심기여
- **암묵적(ITD) + 명시적(ETD) 두 관점에서 시간 동역학을 종합 모델링**하는 TDTracker 프레임워크 제안.
- **ITD**: 3D CNN으로 단기(short-term) 시공간 특징을 암묵적으로 추출.
- **ETD**: Frequency-aware Module(FFT) → GRU → Mamba의 **캐스케이드 구조**로 장기(long-term) 명시적 시간 특징 추출.
- 좌표를 직접 회귀하지 않고 **heatmap(1D heat vector) 예측 + KL divergence loss** 사용 → 확률분포 기반 후처리 가능.
- 합성 SEET 데이터셋 SOTA 달성(이전 SOTA EventMamba 대비 적은 FLOPs), CVPR 2025 챌린지 3위.
- 안구추적 태스크에 **FFT 기반 주파수 도메인 모듈을 처음 적용**.

## 4. 방법론
### 이벤트 표현
- 원시 이벤트: $E = \{e_i = (x_i, y_i, t_i, p_i)\}$ (2S-1T-1P).
- Event Frame: $\mathcal{F}(x,y) = \sum_{t_i \in (t_0, t_n)} p_i \text{ or } |p_i|$.
- Voxel: 시간축을 $b$개 bin으로 분할, $l_k = t_0 + k \cdot \frac{t_n - t_0}{b}$, $\mathcal{V}(x,y,k) = \sum_{t_i \in (l_k, l_{k+1})} p_i$.
- **Binary Map** (채택된 최적 표현): $b$개 이진 프레임을 $b$비트 정수로 인코딩(예: 8프레임→8bit). ablation에서 Event Frame/Voxel 대비 최고 성능(MSE 1.89 vs 2.06/2.10).

### ITD (Implicit Temporal Dynamic)
- 입력 텐서 $I \in \mathbb{R}^{C \times T \times H \times W}$.
- 공간 3D conv: $F_S = \text{Conv3D}(I; W_S) + B_S$, 커널 $(1, K_s, K_s)$.
- 시간 3D conv (Implicit-conv): $F_T = \text{Conv3D}(F_S; W_T)$, 커널 $(K_t, K_t, K_t)$ → 시간 차원 $T$를 시공간 특징으로 추상화.
- 3단계 순차 구조로 receptive field 점진 증가, 각 단계에서 공간 차원 average pooling 다운샘플링.

### ETD (Explicit Temporal Dynamic)
- **Frequency-aware Module**: 1D DFT $X[k] = \sum_{n=0}^{T-1} x[n] e^{-j\frac{2\pi}{T}kn}$, 학습가능 필터 $V$와 Hadamard 곱. $X = \text{FFT}(x)$, $\hat{X} = V \odot X$, $x = \sigma(\text{iFFT}(\hat{X}))$. 켤레대칭성으로 $T/2+1$ 길이만 필요.
- **GRU**: $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$, $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$, $\tilde{h}_t = \tanh(W_h x_t + U_h(r_t \odot h_{t-1}) + b_h)$, $h_t = (1-z_t)\odot\tilde{h}_t + z_t \odot h_{t-1}$.
- **Mamba** (SSMv6): $h_t = \bar{A} h_{t-1} + \bar{B} x_t$, $y_t = C h_t$, $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot\Delta B$ (zero-order hold). 선형복잡도로 100 시퀀스 장기 상관 처리.

### Loss
- 2D 라벨 $(x,y)$를 1D heat vector $X, Y$로 변환, Gaussian blur: $v_i = \frac{1}{\sqrt{2\pi}\sigma}\exp(-\frac{(i-\hat{v})^2}{2\sigma^2})$.
- KL divergence: $L_{KL}(P \| Q) = \sum_i P(i)\log\frac{P(i)}{Q(i)}$, 총 손실 $L_{total} = L_{KL}(X \| \hat{X}) + L_{KL}(Y \| \hat{Y})$.

## 5. 실험
- **데이터셋**: 합성 SEET (LPW에서 v2e로 생성, 240×180, ΔT=4.4ms), 실측 3ET+ 2025 (DVXplorer Mini, 13명, 9.2GB, 100Hz 라벨).
- **메트릭**: pixel error 임계 추적률 p3/p5/p10 (예측-GT 유클리드 거리 ≤ p px), MSE(px).
- **SEET 주요수치**: TDTracker 60×80, Param 3.248M, FLOPs 318M, p3=0.953, p5=0.996, p10=1.0, **MSE 1.30px** (SOTA).
  - baseline: MambaPupil(8.608M, p3=0.905, MSE 1.64), EventMamba(0.903M, 476M FLOPs, MSE 1.48), FAPNet(0.29M, MSE 1.56), PEPNet(MSE 1.57).
- **3ET+ 2025 수치**: p3=0.912, p5=0.972, p10=0.992, MSE 1.62px. 후처리 후 **MSE 1.4936**까지 개선. RTX 4090에서 inference 1.7923ms.
- **Ablation**: w/o Implicit-Conv 시 가장 큰 성능 저하(MSE 2.32) → ITD가 핵심. w/o FFT(MSE 1.63), w/o Mamba(MSE 1.70).
- **학습**: PyTorch, AdamW, lr 2e-3(Cosine decay), weight decay 1e-4, RTX 4090, batch 16.

## 6. 강점/한계
- **강점**: 암묵적+명시적 시간 동역학을 결합한 종합적 설계로 정확도/효율 균형 우수. Binary Map 표현 효과적. heatmap+KL로 확률기반 후처리 가능. 적은 FLOPs로 SOTA.
- **한계**: Param 3.248M으로 EventMamba/FAPNet 대비 큼. Frequency-aware Module 파라미터가 시퀀스 길이에 종속되어 챌린지에서는 제거함(유연성 한계). 개폐안(blink) 처리를 별도 휴리스틱(up/down 이벤트 비율 0.09)으로 대응. SEET는 합성 데이터 의존.

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **추정**: 3D CNN(ITD) + 캐스케이드 시계열(FFT/GRU/Mamba) 구조는 FPGA 매핑 시 모듈별 파이프라이닝/병렬화 여지가 큼. 특히 Binary Map 표현은 비트시프트 기반 인코딩으로 HW 친화적.
- **추정**: 1.79ms inference는 GPU 기준이며, FPGA로는 Mamba/GRU의 순환 구조가 지연 병목이 될 수 있음 → ITD(3D conv) 위주 경량화 또는 Mamba 선택적 사용이 저지연 on-device에 유리할 수 있음.
- **추정**: heatmap+KL 방식은 양자화 후 확률분포 안정성 측면에서 직접 회귀보다 강건할 가능성. 양자화 친화성은 별도 검증 필요(확인 불가).
- Frequency-aware Module의 시퀀스 길이 종속성은 고정 시퀀스 HW 설계에서는 오히려 장점일 수 있음(추정).

## 8. 근거표기
- 1~6장 수치/수식/구조는 본문(arXiv:2503.23725v1) 직접 확인.
- 7장 FPGA/on-device 시사점은 **추정** (논문은 GPU 실험만 보고, FPGA 구현·양자화는 다루지 않음).
