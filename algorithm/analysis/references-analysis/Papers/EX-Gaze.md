# EX-Gaze: High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality

## 1. 서지정보
- **제목**: EX-Gaze: High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality
- **저자**: Ning Chen, Yiran Shen(교신), Tongyu Zhang, Yanni Yang, Hongkai Wen (Shandong University / University of Warwick)
- **연도**: 2025 (TVCG Vol.31, No.5, May 2025; DOI 10.1109/TVCG.2025.3549565)
- **학회/저널**: IEEE Transactions on Visualization and Computer Graphics (TVCG)
- **원본 파일명**: `EX-Gaze_High-Frequency_and_Low-Latency_Gaze_Tracking_with_Hybrid_Event-Frame_Cameras_for_On-Device_Extended_Reality.pdf`
- **코드**: https://github.com/Ningreka/EX-Gaze

## 2. 문제정의·배경
- XR(VR/AR)에서 시선추적은 핵심 HCI 모달리티(포비티드 렌더링, 정신건강 진단, 사용자 인증 등). 이들 응용은 **고주파(수 kHz)·실시간·저지연** 추적을 요구.
- 인간 안구는 saccade 시 각속도 최대 700°/s, 가속도 24,000°/s²로 매우 빠름 → 높은 시간 해상도 필요.
- 전통 CCD/CMOS 카메라는 30~50Hz 수준, EyeLink 1000(1kHz)은 계산·메모리·대역폭 부담이 커 임베디드 실시간 처리 곤란.
- 이벤트 카메라는 픽셀단위 밝기 변화만 비동기·희소(sparse)하게 출력 → 수십 µs 시간해상도, 저전력·저대역폭으로 임베디드에 적합. 그러나 기존 이벤트 기반 방법들은 정확도/효율 균형이 부족(EBVEYE/EV-Eye는 데이터셋 위주, Swift-Eye는 정확하나 무거움, 3ET/Retina는 가벼우나 정확도 미흡, SNN은 전용 뉴로모픽 칩 필요).

## 3. 핵심 기여
- **하이브리드 이벤트-프레임 카메라(DAVIS346)** 기반, 임베디드(Jetson Orin Nano 8GB)에서 **2KHz 실시간 고주파** 시선추적 시스템 제안.
- 이벤트의 희소성을 활용하는 **Sparse Event-patch Representation**과 이를 처리하는 **Sparse Event-patch Transformer(SEPT)** 백본 설계로 계산시간 대폭 절감.
- 프레임/이벤트 접근을 효율적으로 전환하는 **gradient 기반 경량 approach-adaptation** 스킴(무거운 프레임 검출을 매 프레임 수행하지 않음).
- GPU/CPU 이종 컴퓨팅 자원 스케줄링·오프로딩으로 지연 누적 없이 2KHz 달성.

## 4. 방법론
### 시스템 구조 (4개 모듈)
1. **Frame-based Initialization & Relocalization**: MobileNetV3-small 백본 + (RetinaNet 유사) 2개 검출 헤드(eye region, pupil region). 초기화 시 1회, 그리고 이벤트 추적 실패 시에만 동작(저프레임 25fps).
2. **Event-based Pupil Tracking**: SEPT로 고주파(2KHz) 추적, GPU 처리시간 < 0.4ms.
3. **Approach-adaptation**: 추적 정확도를 gradient similarity로 추정해 프레임/이벤트 전환.
4. **Gaze Regression**: 다항 회귀로 동공 중심 → 화면상 시선점(PoG) 매핑.

### 안구 파라메트릭 모델
- eye region: $D = (x_D, y_D, w_D, h_D)$ (근안(near-eye) 설정에서 거의 고정 → 초기화 시 1회 crop)
- pupil region (타원): $P = (x_P, y_P, a_P, b_P, \theta_P)$

### 이벤트 표현 (Sparse Event-patch)
- 짧은 슬라이스(예: 5ms) 이벤트: $E = \{e_i\} = \{x_i, y_i, t_i, p_i\}_{i=1:N}$
- 좌표·극성별 누적으로 2채널 이벤트 이미지: $I(x,y,p)=\sum_{i\in N}\delta(x-x_i, y-y_i, p-p_i)$ (δ는 Kronecker)
- **핵심 아이디어**: 동공/홍채/속눈썹 등 edge 주변에만 이벤트가 집중 → 직전 추적 타원 경계 위 M=8개 점(반경간격 $2\pi/M$)을 중심으로 16×16 정사각 patch만 추출. 입력 크기 축소 + 노이즈/무관 이벤트 배제. (희소 CNN 대신 dense CNN을 patch에 적용 → CUDA 효율 활용)

### Adaptive Frequency
- 누적 이벤트 수 $e^c$와 누적 기간 $t^c$로 업데이트 트리거: $e^c_{(t-t^c)\to t} > TH_{e^c}\ \&\ t^c < TH_{t^c}$ (구현: $TH_{e^c}=50$, $TH_{t^c}=5ms$). 0.5ms마다 조건 확인 → fixation 시 불필요 갱신 회피, saccade 시 최고주파.

### SEPT 백본
- patch-shared CNN: conv 1층 + Fused-MBConvBlock 3개(파라미터 공유로 병렬화). + learnable positional embedding.
- Transformer encoder(경량 separable self-attention) → average pooling → FC로 오프셋 회귀.
- 출력 오프셋: $O_{t_0\to t_1} = (x_O, y_O, a_O, b_O, \theta_O)$
- 디코딩(mmrotate 유사): $x_P^{t_1}=x_P^{t_0}-x_O\cdot w_{anchor}$, $a_P^{t_1}=a_P^{t_0} e^{a_O}$, $\theta_P^{t_1}=\theta_P^{t_0}+\theta_O$ 등. 최고 시간해상도 = $t_0\to t_1$ 간격 = 0.5ms.

### Approach-adaptation (gradient template matching)
- 타원 경계 K점의 법선 $n_i$와 Sobel 이미지 gradient $g_i$의 유사도: $S=\frac{1}{K}\sum_{i=1}^K \frac{\langle n_i, g_i\rangle}{\|n_i\|\cdot\|g_i\|}$. $S<TH_s$(=0.8)이면 프레임 재지역화 트리거.
- 계산시간: sparse normal/gradient 0.02ms vs 프레임 검출 1.17ms (약 60배 빠름). glint 등 outlier gradient(평균±2σ 초과)는 제외.

### Gaze Regression
- n차 다항 회귀: $x_{PoG}=a_0+\sum_{u=1}^n\sum_{v=0}^u a_{(v,u)} x_P^{u-v} y_P^v$ (y도 동일). 사용자별 보정 단계에서 계수 추정.

### 손실/학습
- 학습 증강: 추정 동공 파라미터에 랜덤 translation/rotation/scaling 섭동 부여 → 노이즈/부정확성 강건화.

## 5. 실험
- **데이터셋**: (1) EV-Eye(DAVIS346 2대, 48명, 머리 고정, grayscale 25fps + 고시간해상도 이벤트). 기존 9,011장 라벨을 보강해 총 19,270장(384 연속 시퀀스) 추적 평가용 라벨링. (2) OpenEDS2020(VR HMD, 100fps near-eye 프레임) → Super-SloMo 500배 보간 후 v2e로 이벤트 합성, 200개 1초 시퀀스. EV-Eye 학습 모델을 OpenEDS에 fine-tuning 없이 적용(cross-dataset 일반화).
- **메트릭**: IoU, F1, Pixel error(Pe, 동공중심 유클리드 거리), 그리고 Jetson Orin Nano 8GB 실측 계산시간.
- **Ablation**: $TH_{e^c}=50$일 때 최적(IoU 0.880, F1 0.933, Pe 1.183). $TH_s=0.8$ 선택 시 약 92% 프레임 생략(재지역화 확률 8.1%, Pe 1.412). frame localization IoU≥0.7 유지가 매끄러운 전환에 중요(>99% 충족).
- **SOTA 비교(EV-Eye 데이터, Jetson 실측)**: EX-Gaze가 효율 1위. saccade 1초 데이터 처리 0.39s / Pe 1.33px, smooth pursuit 0.43s / Pe 1.49px. 3ET는 1.03s/24.58px, Swift-Eye는 정확하나 1초 데이터에 1,000s 이상 소요. → 정확도-효율 trade-off 최적.
- **OpenEDS**: fine-tuning 없이도 고정밀 방법에 근접, 경쟁 방법들은 2KHz 실시간 미달.
- **모듈 오버헤드(Jetson)**: 프레임 재지역화 1.17ms, 이벤트 추적 0.32ms, adaptation 0.02ms, gaze regression 0.00012ms → 평균 추적당 약 0.407ms (2KHz, 0.5ms 예산 충족).

## 6. 강점/한계
- **강점**: 임베디드에서 실제 2KHz 실시간 달성. 희소성을 dense CNN 친화 patch로 변환해 CUDA 효율 극대화. 경량 approach-adaptation으로 무거운 프레임 검출 호출을 ~8%로 억제. 정확도-효율 균형이 SOTA 대비 우수.
- **한계(저자 명시)**: 실제 프로토타입 미구축(합성 데이터 한계, 실환경 조명/움직임 미검증). 현재 2D gaze만, 3D 미지원. DAVIS346(이벤트+프레임) 하드웨어 의존.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)
- **추정**: 우리 목표(FPGA 저지연 가속)와 매우 유사한 문제설정. SEPT의 patch-shared CNN + 경량 트랜스포머는 0.5ms 예산 내 동작하도록 설계되어, **FPGA 파이프라인화 후보**로 적합. 특히 patch 단위 공유 가중치 CNN은 systolic/병렬 PE로 매핑이 자연스러움.
- **Sparse Event-patch Representation**은 입력 크기를 8×16×16 patch로 고정 → FPGA에서 결정적 latency(고정 데이터 레이아웃)와 작은 온칩 버퍼로 구현 유리(희소 CNN의 동적 레이아웃 문제 회피). 우리 가속기 설계 시 "edge 주변 고정 patch" 전략은 BRAM 사용량 예측가능성 측면에서 재사용 가치.
- **Adaptive frequency 트리거**($e^c, t^c$ 임계값)는 단순 카운터·비교기로 HW 구현 가능 → 저전력 게이팅(이벤트 없을 때 가속기 idle).
- **Gradient template matching**(법선·Sobel·코사인유사도)은 FPGA 친화 고정소수점 연산으로 이식 가능, 무거운 프레임 검출 호출 빈도를 줄이는 정책으로 활용.
- gradient/내적/exp 디코딩 등은 LUT/CORDIC 근사로 양자화 검토 필요(정확도 영향 평가 권장).

## 8. 근거표기
- 모델 구조/수식/실험수치/모듈 오버헤드는 본문(p.1~9)에서 직접 확인. Figure 8/9/10 그래프 수치는 본문 텍스트에 인용된 값 기준.
- "FPGA 적합성·이식 시사점"은 본 논문에 FPGA 언급 없음 → **추정**(논문은 Jetson Orin Nano GPU/CPU 대상).
