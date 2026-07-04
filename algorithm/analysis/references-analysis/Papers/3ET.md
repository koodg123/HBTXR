# 3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network

## 1. 서지정보
- **제목**: 3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network
- **저자**: Qinyu Chen*, Zuowen Wang*, Shih-Chii Liu*, Chang Gao† (*INI, Univ. Zurich & ETH Zurich; †TU Delft Microelectronics)
- **연도**: 2023 (arXiv:2308.11771v1, 22 Aug 2023)
- **학회**: 2023 IEEE Biomedical Circuits and Systems Conference (BioCAS), Toronto
- **원본파일명**: 3ET Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network.pdf

## 2. 문제정의·배경
- AR/VR 헤드셋의 차세대 웨어러블 헬스케어/foveated rendering에 시선추적이 필수이나, 고속·고해상 프레임 카메라는 비싸고 전력 소모 큼.
- 근안(near-eye) 추적에서 대부분 영역은 정적이고 안구 영역만 의미 있는 변화 → 프레임 데이터의 redundancy 큼.
- DVS(이벤트 카메라)는 밝기 변화만 sparse하게 포착 → 연산 부담 절감 + iris 정보 미수집으로 프라이버시 보호.
- 기존 CNN은 공간 특징만 추출하고 시간 맥락을 무시 → sparse 이벤트 프레임에서 정보 부족 문제.

## 3. 핵심기여
- 이벤트 스트림에서 sparse 시공간 특징 추출에 ConvLSTM 아키텍처 적용(CNN 대비 30%+ 정확도 향상).
- **Change-Based ConvLSTM (CB-ConvLSTM)** 제안: hidden path에 delta encoder를 적용해 시간 sparsity 유도 → 정확도 손실 없이 **연산량 약 4.7× 감소**.
- 기존 change-based 네트워크와 달리 hidden path에만 delta encoder 사용 → 이전 MVM 결과 누적 불필요, 연산·메모리 오버헤드 최소화.
- 코드/데이터셋 공개 (이후 3ET+ 챌린지 데이터셋의 기반).

## 4. 방법론
### 이벤트 표현
- 이벤트 $e_i = (x_i, y_i, t_i, p_i)$, $p_i = \pm 1$.
- constant time-bin count 표현: $V(x,y) = \sum p_i \ast I(x,y,t_i,x_i,y_i,T_1,T_1+\Delta T)$, 시간창 $\Delta T = 4.4$ms(원본 RGB 프레임율과 동기화).
- LPW(Labeled Pupils in the Wild) RGB를 v2e 시뮬레이터로 DVS 이벤트 변환, 640×480 → 240×180 → 80×60 리사이즈, 22개 비디오에서 11k 프레임.

### 모델 구조
- 4개 ConvLSTM layer (hidden node 8/16/32/64, 커널 3×3) + 2개 FC layer, 총 ~0.42M 파라미터.
- 각 ConvLSTM 출력 후 BatchNorm + ReLU + max pooling. FC1=128 neuron, FC2=2 출력(pupil x,y). FC는 시퀀스 길이 T번 실행.
- **ConvLSTM 갱신식**:
  - $i_t = \sigma(W_{xi} \ast X_t + W_{hi} \ast H_{t-1} + b_i)$
  - $f_t = \sigma(W_{xf} \ast X_t + W_{hf} \ast H_{t-1} + b_f)$
  - $g_t = \tanh(W_{xg} \ast X_t + W_{hg} \ast H_{t-1} + b_g)$
  - $o_t = \sigma(W_{xo} \ast X_t + W_{ho} \ast H_{t-1} + b_o)$
  - $C_t = f_t \odot C_{t-1} + i_t \odot g_t$, $H_t = o_t \odot \tanh(C_t)$

### CB-ConvLSTM (핵심)
- hidden state 변화량에 threshold $\theta$ 적용:
  $$\Delta H_{t-1} = \begin{cases} (H_{t-1} - H_{t-2}), & (H_{t-1}-H_{t-2}) \geq \theta \\ 0, & \text{otherwise} \end{cases}$$
- 갱신식에서 $H_{t-1}$ 대신 $\Delta H_{t-1}$ 사용 → recurrent conv에 높은 시간 sparsity 유도. $X_t$는 본래 sparse(이벤트+ReLU), $H_{t-1}$만 dense였던 점을 해소.

### 학습
- Loss: MSE(예측-GT 유클리드), SGD optimizer, lr 0.001, 30 epoch, batch 16, train/val 80/20. stride 1 overlapping clip augmentation.

## 5. 실험
- **메트릭**: 검출률 p3/p5/p10 (예측-GT 거리 ≤ p px).
- **시퀀스 길이 효과**: 길이 2→40 증가 시 p3 88.8%, p5 97.0%, p10 99.5% (각 +17.4/15.8/6.9%p).
- **주요수치 (CB-ConvLSTM θ=0.5)**: p3=88.50%, p5=96.70%, p10=99.20%, 0.42M param, **9.00M FLOPs**.
  - vanilla ConvLSTM(θ=0): p3=88.88%, FLOPs 18.86M(θ적용전) / 42.61M(원본). CNN baseline: p3=57.80%, 0.40M, 18.40M FLOPs.
- **sparsity**: θ 0→0.5에서 시간 sparsity 69.2%→85.3%, 정확도 손실 없음. CB-ConvLSTM은 vanilla 대비 ~3× sparsity, conv에서 8.8× 연산 절감 → 네트워크 전체 4.7× 절감.

## 6. 강점/한계
- **강점**: 매우 경량(0.42M), CNN 대비 30%+ 정확도, ConvLSTM의 시간 맥락 활용으로 sparse 프레임에서도 강건. delta encoder 방식이 단순하고 HW 친화적. 저지연 실시간 추적 지향.
- **한계**: 합성 데이터(v2e from LPW)만으로 평가, 실측 검증 부재. 80×60 저해상도로 정밀도 제한. blink/occlusion 등 특수 케이스 명시적 처리 없음. ConvLSTM의 순환 구조는 병렬화 제약.

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **명시적 시사점**: 논문 자체가 "CB-ConvLSTM은 시공간 sparsity를 활용하는 전용 하드웨어(예: Spartus FPGA LSTM 가속기)에 구현 가능"이라고 결론에서 언급 → **FPGA 가속 직접 연관**.
- delta encoding으로 유도된 시간 sparsity는 FPGA에서 zero-skipping/sparse MAC으로 직접 연산·에너지 절감 가능(추정 + 논문 시사).
- **추정**: 0.42M 초경량 모델은 on-device FPGA에 적합. ConvLSTM의 순환 의존성은 파이프라인 stall을 유발할 수 있어 hidden state 캐싱/스케줄링 설계 필요.
- **추정**: threshold θ는 정확도-sparsity trade-off 튜닝 노브 → HW 자원 예산에 맞춰 조정 가능. 양자화는 논문에서 다루지 않음(확인 불가).

## 8. 근거표기
- 1~6장 수치/수식/구조는 본문(arXiv:2308.11771v1) 직접 확인.
- 7장: 논문 결론의 "specialized hardware exploiting spatio-temporal sparsity [Spartus FPGA, ...]" 언급은 **직접 인용**, 양자화·구체적 FPGA 구현 세부는 **추정/확인 불가**.
