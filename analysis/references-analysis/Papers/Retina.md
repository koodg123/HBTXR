# Retina: Low-Power Eye Tracking with Event Camera and Spiking Hardware

## 1. 서지정보
- **제목**: Retina: Low-Power Eye Tracking with Event Camera and Spiking Hardware
- **저자**: Pietro Bonazzi, Sizhen Bian, Giovanni Lippolis, Yawei Li, Sadique Sheik, Michele Magno (ETH Zurich, Inivation AG, Synsense AG)
- **연도**: 2024 (arXiv:2312.00425v2, 17 Apr 2024)
- **학회/발표**: IEEE/CVF CVPR Workshops (CVPRW) 2024, pp. 5684-5692, DOI 10.1109/CVPRW63382.2024.00577 (다른 논문 참고문헌 교차확인)
- **원본파일명**: retina.pdf

## 2. 문제정의·배경
- 뉴로모픽 시스템(이벤트 카메라 + spike 연산)은 저전력·저복잡도 장점이 있으나, 기존 이벤트 시선추적은 (1) 프레임 기반 입력 의존 또는 end-to-end gaze(고정 화면좌표 학습), (2) subject-specific calibration 필요, (3) end-to-end 전력/지연 실측 부재의 한계.
- 3ET[Chen et al.]는 합성 이벤트를 32-bit 픽셀 프레임으로 누적·정규화 → 이벤트의 비동기 1-bit 특성 미활용으로 비효율.
- 목표: **순수 이벤트 입력 + SNN + 뉴로모픽 칩(Speck) 실배치**로 저전력·저지연 동공 추적.

## 3. 핵심기여
- **Ini-30 데이터셋**: 안경 프레임에 DVS 2대(DVXplorer, 640×480) 장착, 30명, 머리 고정 없이 자연스러운 안구 움직임. 화면좌표가 아닌 **센서 어레이 상 동공 위치 라벨**. 이벤트 수 기반 슬라이싱(고정 timestamp 아님)으로 DVS 간 domain gap 적응.
- **Retina 알고리즘**: IAF(Integrate-and-Fire) 뉴런 기반 SNN + non-spiking 1D temporal weighted-sum filter(회귀). voltage decay/recurrent 뉴런 없이(Speck 미지원) 시간정보 학습. **뉴로모픽 HW에 배치 가능한 최초의 시선추적 알고리즘** 주장. 3ET 대비 정밀도 -20% centroid error, 연산 -30× MAC.
- **뉴로모픽 HW(Speck) 실배치**: 온칩 추론 전력/지연 + 온보드 DVS end-to-end 평가 최초 제공.

## 4. 방법론
### 데이터 준비
- 원본 사각 해상도 → 512×512 정사각(y축 16px shift, x<96 & x>608 폐기) → sum pooling으로 Speck 호환 **64×64×2채널**.
- **dynamic event window**(이벤트 수 기반 슬라이싱)가 fixed time window보다 우수. 동일 픽셀 다중 이벤트 시 최다 polarity 유지 후 0/1 clip(1채널만 활성). 라벨 30ms주기 vs 이벤트 200μs → 인접 2라벨 가중보간. 64 timebin 학습.

### 네트워크 구조 (Table 5)
- 단일 SNN: spiking spatial conv + fusible BatchNorm + IAF 뉴런. 8 layer(BatchConv+IAF+Pool), Speck 코어 메모리 제약에 맞춤.
- 커널 메모리 $K_{MT} = c \cdot 2^{\log_2 k_x k_y} + \log_2 f$, 뉴런 메모리 $N_M = f f_x f_y$, $f_x = \frac{c_x - k_x + 2p_x}{s_x} + 1$.
- spike threshold=1, min voltage=-1, surrogate gradient(periodic exponential).
- 출력층: YOLO-식 4×4 cell, 각 cell에 2개 5차원 벡터(bbox 좌상/우하 + confidence) → NMS 후처리. 1px 라벨을 각 방향 2px 확장한 target box.

### IAF 뉴런 모델
- $\tau_m \frac{dV}{dt} = -V(t) + R_m I_{syn}(t)$, $V(th)=1$ 도달 시 발화·$V(reset)=0$.
- $I_{syn}(t) = \sum_j w_j \cdot I_j(t - t_j)$.

### Temporal weighted-sum filter (핵심: spike→연속값 변환)
- $y(t) = \sum_{i=1}^{N} w_i \cdot x(t-i)$.
- 가중치는 synaptic kernel $S(t) = \exp(-t/\tau_{syn})$ 와 membrane kernel $M(t) = \exp(-t/\tau_{mem})$ 의 convolution으로 초기화. (filter 없으면 error 24.46px → 있으면 3.24px로 핵심 역할.)

### Loss
- $L_{box} = \sum (p_i - t_i)^2$ (bbox MSE), $L_{conf} = \sum (c_i - g_i)^2$ (confidence MSE), $L_{syn}$ (layer별 MAC을 Speck 한계 1e6 내로 정규화하는 regularizer).
- 가중치 $\lambda_{box}=7.5$, $\lambda_{conf}=1.5$, $\lambda_{syn}=$1e-7. ADAM, 576 iter, RTX 4090 1시간 학습, 매 iter 뉴런 state reset.

## 5. 실험
- **메트릭**: centroid error (px), 전력 P(mW), 에너지 E(mJ), 지연 L(ms), Param, MAC.
- **정밀도 (Ini-30, 64×64×2)**: Retina **3.24px (±0.79)** vs 3ET 4.48px (±1.94). 합성 LPW: Retina 6.46px vs 3ET 5.33px.
- **복잡도**: Retina **63k param, 3.03M MAC** vs 3ET 418k param, 107M MAC (param 6.63×↓, MAC ~35×↓).
- **Speck 실측 (end-to-end)**: Fixed Window(3ms) → **전력 2.89mW, 지연 5.57ms, 에너지 16.10mJ**. Dynamic Window(300ev) → 4.80mW, 8.01ms, 38.40mJ.
- **Ablation**: dynamic window가 fixed보다 정밀(3.24 vs 3.71px @3ms) + firing rate 낮음(첫 layer 10% vs 20%). temporal filter $\tau_{mem}=\tau_{syn}=5$ 최적(3.24px). neuron reset 필수(없으면 7.99px), bbox 예측이 단일좌표보다 우수(w/o box 5.89px).

## 6. 강점/한계
- **강점 (FPGA/on-device 관점 매우 중요)**: 실제 뉴로모픽 칩(Speck)에 배치하여 **end-to-end 전력 2.89~4.8mW, 지연 5.57~8.01ms 실측**. 극경량(63k param, 3.03M MAC). 순수 이벤트 입력·spike 연산으로 sparsity 완전 활용. 8-bit weight + 16-bit neuron state. calibration-free.
- **한계**: 정밀도(3.24px)는 프레임 기반 고정밀(<0.5px glint 등) 대비 낮음. SNN 학습 난이도(surrogate gradient, neuron reset 의존). Speck 코어 메모리 제약으로 모델 크기 강제 제한. 합성 LPW에선 3ET보다 다소 떨어짐.

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **직접 연관 (높은 가치)**: Retina는 우리 목표("저지연 on-device 시선추적")의 거의 정확한 사례 - SNN을 뉴로모픽 SoC에 배치하여 mW급 전력·ms급 지연 달성. 우리가 **FPGA로 동일 목표를 추구할 때 정량적 비교 기준(baseline)**으로 직접 활용 가능(추정).
- **추정**: Retina는 ASIC형 뉴로모픽 칩(Speck)을 쓰나, 우리는 FPGA → SNN 대신 양자화 ANN(예: 경량 CNN/ViT) + 이벤트 표현으로 유사 전력/지연을 노릴 수 있음. 8-bit weight / 16-bit state 양자화 전략은 우리 INT8 FPGA 데이터패스와 정합(추정).
- **추정**: dynamic event-count slicing은 FPGA 입력 버퍼링 설계에 유용한 패턴(고정 window 대비 정밀+저firing). temporal weighted-sum filter(고정 가중치 1D conv)는 FPGA에서 단순 MAC 누산으로 매우 저비용 구현 가능 - spike→좌표 회귀의 핵심.
- **추정**: 출력층 grid(4×4)+NMS는 anchor-free에 가까워 우리 경량 헤드 설계의 참고 모델.

## 8. 근거표기
- 1~6장 수치/수식/구조/전력·지연 실측은 본문(arXiv:2312.00425v2) 직접 확인.
- 발표 학회(CVPRW 2024, pp.5684-5692)는 다른 논문(SynUnlabeled, AdaptiveSSM)의 참고문헌에서 **교차확인** (본 PDF는 arXiv 버전이라 게재정보 미표기였으나 DOI 확정).
- 7장 FPGA 매핑·양자화 ANN 대안은 **추정** (논문은 Speck 뉴로모픽 SNN 배치만 다룸).
