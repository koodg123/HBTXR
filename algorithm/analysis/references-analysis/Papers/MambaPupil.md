# MambaPupil: Bidirectional Selective Recurrent model for Event-based Eye tracking

## 1. 서지정보
- **제목**: MambaPupil: Bidirectional Selective Recurrent model for Event-based Eye tracking
- **저자**: Zhong Wang, Zengyu Wan, Han Han, Bohao Liao, Yuliang Wu, Wei Zhai(교신), Yang Cao, Zheng-jun Zha (University of Science and Technology of China, USTC)
- **연도**: 2024
- **학회/저널**: CVPR Workshop (AIS 2024 Event-based Eye Tracking Challenge 관련; CVPRW)
- **원본 파일명**: `MambaPupil Bidirectional Selective Recurrent model for Event-based Eye tracking.pdf`

## 2. 문제정의·배경
- 이벤트 카메라는 고시간해상도·저중복으로 시선추적에 유망하나, 안구 움직임 패턴이 다양·급변(blink, fixation, saccade, smooth pursuit)하여 동공 위치추정이 어려움.
- 세 가지 핵심 도전: (1) **blinking으로 인한 타깃 소실** + 다량의 무관 이벤트 → 단기 예측 변동, (2) **eye resting 시 이벤트 희소** → 예측 편차·jitter, (3) **glasses/속눈썹/홍채 반사 등 간섭** → 특히 안경은 장기 오정렬.
- 기존 RNN(LSTM)/SNN 기반은 시간 방향을 단방향으로만 보고 모든 timestamp를 동등 취급 → 모델이 단순하여 어려운 상황에 취약. → **문맥적 시간정보의 양방향·선택적 활용**이 핵심.

## 3. 핵심 기여
- **MambaPupil** 프레임워크: 시간 관계를 **양방향(bidirectional)·선택적(selective)** 으로 모델링.
- **Dual Recurrent Module**: Bi-GRU(양방향 문맥) + **LTV-SSM**(Linear Time-Varying State Space Module, 입력의존 가중치로 유효 모션 단계에 선택적 집중).
- **Bina-rep** 컴팩트 이벤트 표현 채택 + 맞춤형 데이터 증강 **Event-Cutout**(공간 랜덤 마스킹)으로 강건성 향상.
- ThreeET-plus(EET+) 벤치마크에서 SOTA 달성, 학습비용 저·추론 매우 빠름.

## 4. 방법론
### 이벤트 트리거 모델
- 로그 밝기 변화 임계 초과 시 이벤트 발생: $\log\mathcal{I}(x,y,t)-\log\mathcal{I}(x,y,t-\delta t)=p\,C$ ($p\in\{-1,1\}$ 극성, C 임계).

### 이벤트 표현: Bina-rep
- Barchid et al.의 이진화 양자화: 일정 시간 ∆t 누적 이벤트를 N-bit map으로 변환. 누적 이벤트 프레임을 공간 이진화 → 각 프레임을 temporal binarization mask로 인코딩 → 누적해 최종 Bina-rep $B_e$.
- 이점: (1) 저장·계산비용 절감(동일 시간길이에서 표준 이벤트프레임 대비 $1/n_{bins}$ 크기), (2) 고립 노이즈 이벤트 영향 감소.

### 데이터 증강
- spatial flip, spatial shift, time shift, 그리고 **Event-Cutout**: 랜덤 크기·위치의 사각 영역을 0으로 마스킹 → blinking/안경 등 외부 간섭 시뮬레이션, 전역 특징 학습 유도. 학습 데이터의 50%에 적용.

### 네트워크 구조
1. **Spatial Feature Extractor**: conv block 3개. $x_t = \text{Pool}(\text{ReLU}(\text{BN}(\text{Conv}(B_e))))$. 채널 32/128/512, 큰 커널(7/5/5)로 넓은 공간 상관 포착. 이후 global spatial pooling + spatial dropout.
2. **Dual Recurrent Module** (핵심):
   - **Bi-GRU** (식): $z_t=\sigma(W_z\cdot[h_{t\pm1},x_t])$, $r_t=\sigma(W_r\cdot[h_{t\pm1},x_t])$, $\tilde h_t=\tanh(W\cdot[r_t * h_{t-1}, x_t])$, $h_{t\{f,b\}}=(1-z_t)*h_{t\pm1}+z_t*\tilde h_t$. forward/backward hidden state를 concat → 양방향 문맥 $h_t$. 시퀀스 양끝 예측 정확도 향상.
   - **LTV-SSM** (Mamba/S6 기반, time-varying): 상태공간 $\Delta x = Ax+Bu$, $y=Cx+Du$. D는 고정 파라미터, A·B·C는 입력 의존. 이산화: $\Delta, B, C = \text{Linear}(x)$, $\overline{\Delta A}=\exp(\Delta * A)$, $\overline{\Delta B}=\Delta * B$. RMSNorm + residual 적용: $x_t'=\text{RMSNorm}(x_t)$, $h_t=\overline{\Delta A}*h_{t-1}+\overline{\Delta B}*x_t'$, $y_t=C*h_t+D*x_t'+x_t$. → 유효 단계(smooth pursuit)에 더 집중, resting/blinking에 덜 집중.
3. **FC layer**: 최종 동공 위치 (x,y) 회귀.

### 손실 함수
- 세그먼트 평균 RMSE: $Loss=\sqrt{\frac{1}{L}\sum_{i=1}^L (y_{i,pred}-y_{i,label})^2}$.

## 5. 실험
- **데이터셋**: EET+ (ThreeET-plus, AIS2024 CVPRW). 13개 시나리오, 각 2~6 세그먼트, 이벤트 해상도 640×480. 5종 모션(random/saccade/reading/blinking/smooth pursuit). 라벨 100Hz, 3/4 train·1/4 val, 예측은 20Hz. 학습 시 80×60으로 다운샘플.
- **메트릭**: p5/p10/p15 (n픽셀 이내 성공률, 높을수록 좋음), p_error(평균 유클리드 거리, 낮을수록 좋음).
- **구현**: PyTorch, RTX2080Ti/GTX1080Ti, 1000 epoch, batch 32, Adam, Cosine Annealing Warm Restart, lr 0.002. Bina-rep bit=4, 시퀀스 길이 45, train stride 5.
- **주요 결과(SOTA 비교, train stride 5)**: MambaPupil p5 0.937 / p10 0.984 / p15 0.990 / **p_error 2.03**. CNN-GRU 대비 p5 +9.0%, p10 +1.5%, p15 +0.1%, 오차 2px대. (CB-ConvLSTM[3ET] p_error 3.82, CNN-GRU 5.90)
- **모델 규모**: MambaPupil 8.59M params, FLOPs 2.61T, 1000ep 학습 1h31m (CNN-GRU 37.23M, CB-ConvLSTM 417.17K).
- **Ablation**: Bi-GRU+LTV-SSM가 최적(p5 0.935). Uni-GRU 대체 시 양끝 문맥 부족으로 오차↑, LTV-SSM 제거 시 5/10px 미스율 약 2배·오차 ~0.1px↑. 이벤트 표현은 Bina-rep(2.35) > Frame(2.42) > Voxel-grid(2.56). 증강 중 spatial shift/Event-Cutout가 가장 효과적.

## 6. 강점/한계
- **강점**: Bi-GRU의 양방향 문맥 + SSM의 선택적 시간 모델링으로 blink/rest/onset/fast-move 등 난조건에서 안정·정밀. 파라미터·학습시간 대비 우수, 추론 빠름.
- **한계**: 오프라인 양방향(Bi-GRU)은 미래 정보를 사용 → **실시간 인과적(causal) 스트리밍에는 부적합**(전체 세그먼트 필요). 20Hz 라벨 예측(저주파). 이벤트의 비동기성을 완전 활용하진 못함(저자 future work로 명시). FPGA/임베디드 실측 없음.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)
- **추정**: LTV-SSM(선택적 SSM)은 순환식이 단순한 elementwise 곱·누적 형태($h_t=\overline{\Delta A}h_{t-1}+\overline{\Delta B}x_t'$)라 **FPGA 파이프라인/누산기 매핑에 매우 유리**. 단, A·B·C 입력의존 생성(Linear)과 exp 이산화는 LUT/근사 양자화 필요.
- **Bina-rep**(N-bit 이진 표현)은 메모리·연산량을 $1/n_{bins}$로 줄여 **FPGA 온칩 메모리·비트연산(bitwise)** 친화 → 입력 표현으로 강력 후보. 비트맵 누적은 popcount/shift로 경량화 가능.
- **양방향성은 저지연 실시간과 상충** → 우리 시스템에서는 Bi-GRU를 단방향/지연 제한 형태로 변형하거나, LTV-SSM 단독(인과적) 사용 검토 필요(트레이드오프 평가 권장).
- Event-Cutout 같은 증강은 학습단계 기법이라 HW와 무관하나, 강건성 확보용으로 우리 학습 파이프라인에 채택 가치.

## 8. 근거표기
- 구조/수식(식 1~15)/실험 표(Table 1~5)/규모는 본문(p.5762~5770)에서 직접 확인.
- "FPGA 적합성, 양방향-실시간 상충" 분석은 본 논문에 FPGA·실시간 지연 분석 없음 → **추정**.
- LTV-SSM의 정확한 채널/상태차원(N) 등 세부 하이퍼파라미터는 본문에 미기재 → **확인 불가**.
