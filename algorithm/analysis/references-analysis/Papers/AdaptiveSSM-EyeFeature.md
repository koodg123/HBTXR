# AISSM: Enhancing Eye Feature Estimation from Event Data Streams through Adaptive Inference State Space Modeling

## 1. 서지정보
- **제목**: Enhancing Eye Feature Estimation from Event Data Streams through Adaptive Inference State Space Modeling
- **저자**: Viet Dung Nguyen, Mobina Ghorbaninejad, Chengyi Ma, Reynold Bailey, Gabriel J. Diaz, Alexander Ororbia (RIT) + Alexander Fix, Ryan J. Suess (Meta Reality Labs, advisors)
- **연도**: 2026 (arXiv:2603.14077v2, 31 Mar 2026; preprint)
- **학회/발표**: 미상(preprint) - **확인 불가** (arXiv preprint, 정식 게재처 명시 없음)
- **원본파일명**: ENHANCING EYE FEATURE ESTIMATION FROM EVENT DATA STREAMS THROUGH ADAPTIVE INFERENCE STATE SPACE MODELING.pdf

## 2. 문제정의·배경
- 이벤트 기반 안구 특징(동공중심 등) 추출은 저전력·고효율이나, **gaze behavior의 kinematics 변화에 따른 급격한 event density 변화**를 다루는 추출기가 거의 없어 성능 저하.
- saccade(빠른 ballistic 움직임)는 dense 이벤트·고 SNR, fixation/smooth pursuit/VOR은 sparse 이벤트·저 SNR. 시간창을 늘리면 지연↑·유효 샘플링↓, RNN으로 대체해도 유사 trade-off + 고SNR 구간에서 불필요한 과거맥락 bias.
- 가설: history-driven/recurrent 모델에만 의존하면 event density/SNR 급변 시 부정확. → Kalman filter 영감의 Bayesian 모델로 posterior(현재)와 prior(과거)를 동적 가중.

## 3. 핵심기여
- **AISSM(Adaptive Inference State Space Model)**: 현재 vs 최근 정보의 상대 가중치를 SNR·event density 추정에 따라 동적 조절하는 특징추출 아키텍처.
- **Dynamic confidence network**: 현재 관측의 신뢰도 α(SNR + event density)를 예측하는 보조 네트워크.
- **Long-horizon training**: replay buffer로 모델 상태(posterior/prior/RNN state)를 데이터 인스턴스에 부착·갱신하여 stateless training 문제 해결, 학습 효율·안정성 향상.
- 3ET+ 벤치마크에서 SOTA baseline(CB-ConvLSTM, MambaPupil 등) 능가.

## 4. 방법론
### 이벤트 표현
- 이벤트 $(x, y, p, t)$. **binarep**(binary event-frame, Barchid 2022) 채택 - 경량 2D 그리드. CB-ConvLSTM에서 높은 정확도 확인된 표현.

### AISSM 모델 구성 (Bayesian SSM)
- 구성요소 (모두 ANN):
  - Encoder posterior: $q_\theta(s_t | o_t)$ (CNN+MLP, 출력을 2D categorical 분포 행렬로 reshape)
  - Recurrent dynamics: $f_\omega(h_t | h_{t-1}, s_{t-1})$ (RNN)
  - Transition prior: $p_\nu(s_t | h_t)$ ($h_{t-1}$과 $s_{t-1}$ concat 후 MLP → categorical 분포)
  - Confidence: $f_\lambda(\hat{\alpha}_t | o_t)$
  - Task head: $f_\psi(\hat{y}_t | \bar{s}_t)$
- **동적 가중 결합 (핵심)**: $\bar{s}_t = \hat{\alpha}_t \times s_t^q + (1 - \hat{\alpha}_t) \times s_t^p$, $s_t^p \sim p_\nu(s_t|h_t)$, $s_t^q \sim q_\theta(s_t|o_t)$. ($\hat{\alpha} \in [0,1]$이 현재 관측 신뢰도.)
- 3ET+ 라벨이 동공중심이라 $\hat{y}_t \in \mathbb{R}^2$.

### 학습
- variational inference와 달리 posterior/prior 붕괴(KL penalty) 안 함 → posterior=현재 전용, prior=과거 전용 유지 (붕괴 시 RNN으로 퇴화). KL 불필요.
- Task head: **Huber loss** $\mathcal{U}(y,\hat{y}) = 0.5\min(|\hat{y}-y|,\delta)^2 + \delta(|\hat{y}-y| - \min(|\hat{y}-y|,\delta))$, $\delta=1.0$ (outlier 둔감·gradient clip).
- categorical 샘플링 비미분 → **straight-through gradient**: $s = \text{sg}(\text{sample}(z)) + (\sigma(z) - \text{sg}(\sigma(z)))$.

### Dynamic confidence network
- 라벨 주변 ROI $r_t$ (160×120 학습 해상도에서 h=40, w=70).
- $\text{SNR}_t = \sigma(\frac{\sum_{r_t} e}{\sum_{\notin r_t} e})$, $\text{ED}_t = \text{clip}(\frac{\sum_{r_t} e}{h \times w \times \tau}, 0, 1)$ ($\tau=0.1$), $\alpha_t = \beta \text{SNR}_t + (1-\beta)\text{ED}_t$ ($\beta=0.1$). Huber loss로 학습.

### Long-horizon training
- RNN 재초기화 시 시간정보 소실(stateless training) 문제 → 모델 상태를 각 학습 item에 부착·dataset에 저장, replay buffer로 uniform 샘플링·갱신(truncated BPTT와 차별).

## 5. 실험
- **메트릭**: P5/P10/P15 (예측-GT 거리 ≤ 5/10/15px 성공률), normalized distance ($[0,\sqrt{2}]$). P-metric은 해상도/광학 의존성으로 cross-dataset 비교 제한 명시.
- **데이터셋**: 3ET+ (단일 벤치마크). 학습 160×120, 평가 320×320(선행 파이프라인 정합).
- **주요수치 (Table 1, 320×320, ~500k param 제약 공정비교)**: AISSM **P5 46.85, P10 72.38, P15 85.41, distance 0.01** (모두 SOTA + 가장 낮은 표준편차).
  - baseline: CB-ConvLSTM(P5 29.86), CNN-GRU(25.33), CNN(17.67), CNN-Mamba(16.02), CNN-BiGRU(21.63), MambaPupil(11.35, distance 0.13).
- saccade→fixation 전환에서 RNN형(CNN-GRU)은 event density 하락 시 급락, AISSM은 과거가중 증가로 hit 우위. long-horizon training이 후반 발산/저하 방지.

## 6. 강점/한계
- **강점**: event density/SNR 급변에 강건한 적응적 가중. ~500k param으로 SOTA, 낮은 분산. Kalman 영감 Bayesian 설계로 해석성. binarep 경량 표현.
- **한계 (논문 명시)**: **inference 속도·지연 미고려** - 두 네트워크(AISSM + confidence) 존재로 지연 증가 가능. 단일 데이터셋(3ET+)만 평가로 일반화 미검증. categorical 분포 샘플링이 시간/연산 비용 큼.

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **주의 (논문 명시)**: 저자가 latency를 명시적으로 다루지 않았고 두 네트워크 구조로 지연 증가 가능 → 우리 저지연 on-device 목표와 직접 부합하지 않음. 아이디어 차용 대상.
- **추정**: binarep(이진 이벤트 프레임)는 FPGA 입력 대역폭/저장에 유리(1-bit). 우리 시스템 입력 표현 후보.
- **추정**: dynamic confidence 기반 가중(현재 vs 과거)은 FPGA에서 RNN/SSM 가동을 조건부로 생략(저SNR 시만 과거 경로 활성)하는 전력 절감 게이팅으로 응용 가능 - E-Track의 RoI 절감과 유사 철학.
- **추정**: categorical 분포 + straight-through gradient + replay buffer 학습은 호스트 학습 단계 기법 - on-device 추론과 무관. 양자화는 논문에서 다루지 않음(확인 불가).
- **추정**: SSM/Mamba 계열은 선형복잡도이나 순환 의존성으로 FPGA 파이프라인에 부담 → 우리는 경량 CNN+조건부 게이팅이 더 적합할 수 있음.

## 8. 근거표기
- 1~6장 수치/수식/구조는 본문(arXiv:2603.14077v2) 직접 확인.
- 게재처는 본문에 명시 없음 → **확인 불가** (preprint).
- 7장 FPGA 매핑·양자화·게이팅 응용은 **추정** (논문은 GPU 실험·정확도만 보고, latency/HW 미평가, 양자화 없음).
