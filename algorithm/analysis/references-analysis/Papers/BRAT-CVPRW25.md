# BRAT: Bidirectional Relative Positional Attention Transformer for Event-based Eye tracking

## 1. 서지정보
- **제목**: BRAT: Bidirectional Relative Positional Attention Transformer for Event-based Eye tracking
- **저자**: Yuliang Wu, Han Han, Jinze Chen, Wei Zhai*, Yang Cao, Zheng-jun Zha (USTC, 중국과기대)
- **연도**: 2025
- **학회/저널**: CVPRW 2025 (CVPR Event-based Vision Workshop, 3ET Challenge 2025 우승)
- **원본파일명**: Wu_BRAT_Bidirectional_Relative_Positional_Attention_Transformer_for_Event-based_Eye_tracking_CVPRW_2025_paper.pdf

## 2. 문제정의·배경
- 이벤트 카메라의 비동기 트리거는 고시간해상도·저전력을 주지만 양날의 검: (1) **정적 정보 손실**(응시 고정/휴식 시 이벤트 거의 미발생 → 동공 위치 추출 곤란), (2) **동적 신호 복잡도 증가**(깜빡임·눈 감음 시 대량 이벤트 발생하나 동공은 가려짐), (3) **이벤트의 불규칙 시공간 분포**.
- 기존 방법(RNN, SNN, Mamba)은 **동공 운동의 방향성(directional) 시간 의존성을 명시적으로 모델링하지 못함** → fine-grained 이벤트 동역학 포착 한계. 특히 2D 표현 기반은 local 시간 관계에 치우쳐 장기 응시/고빈도 깜빡임에 취약.

## 3. 핵심 기여
- 복잡·급격한 동공 운동 시퀀스의 **방향성 시간 의존성을 명시적으로 모델링**하는 BRAT 프레임워크 제안.
- **multi-time-step training** 전략: 서로 다른 time step의 이벤트 표현 시퀀스를 혼합 입력 → 강건성·일반화 향상.
- **3ET Challenge (CVPR EVW 2025) 우승**, ThreeET-plus에서 SOTA.

## 4. 방법론
### 이벤트 표현
- 이벤트 트리거 조건: $\left|\log\mathbf{I}(x_i,y_i,t_i) - \log\mathbf{I}(x_i,y_i,t_i-\Delta t_i)\right| \geq c$
- 이벤트 정의: $e_i := (x_i, y_i, t_i, p_i)$, $p_i \in \{-1, 1\}$ (극성)
- 고정 시간 구간 내 이벤트를 **2D feature 표현(binary representation)으로 적분**, 표현 시퀀스를 네트워크 입력으로 사용. (입력 해상도 80×60로 다운샘플)

### 모델 구조 (spatial encoder + temporal decoder)
- **Spatial Encoder (CNN)**: 입력→커널 3 conv(32ch)→**대형 커널 conv 3층(K=7,5,5)**. 큰 receptive field로 눈꼬리·눈썹 등 주변 특징과의 상관 활용. 앞 2개 conv 뒤 pooling, 3번째 뒤 **spatial dropout**(개별 픽셀이 아닌 전체 행 드롭 → 동공 일부 가림 시뮬레이션, 공간 구조 보존).
- **Temporal Decoder = Bi-GRU + BRAT**:
  - **Bi-GRU**: 양방향 처리로 local 단기 시간 의존성(LSTM보다 경량·학습 용이).
  - **BRAT (핵심)**: 표준 multi-head self-attention을 **bidirectional relative positional** 변형으로 대체. 장기(global) 시간 추론으로 smooth pursuit(점진 운동)·blink(과거+미래 문맥 추론) 보완.

### 핵심 수식 (Bidirectional Relative Positional Attention)
- 각 헤드 출력:
$$\text{Attention}^i = \text{softmax}\left(\frac{\mathbf{Q}^i \mathbf{K}^{i\top}}{\sqrt{d_k}} + \mathbf{B}^i\right)\mathbf{V}^i$$
- $\mathbf{B}^i \in \mathbb{R}^{T\times T}$는 시간 거리 기반 상대위치 bias, forward/backward로 분리:
$$\mathbf{B}^i_{forward} = \begin{cases} m^i\cdot(t-s), & t\ge s \\ 0, & t<s \end{cases}, \quad \mathbf{B}^i_{backward} = \begin{cases} 0, & t\ge s \\ m^i\cdot(s-t), & t<s \end{cases}$$
- $m^i$는 헤드 i의 상대위치 민감도(단조감소 선형 매핑 → 먼 step에 주의 감소). h개 헤드를 절반씩 forward/backward에 할당하여 과거/미래 문맥을 분리(disentangle).

### 학습법
- **Multi-time-step training**: stride 1 슬라이딩 윈도, $R_1,R_2,...,R_T$ 대신 $R_{1\cdot Step},R_{2\cdot Step},...,R_{T\cdot Step}$로 등간격 샘플링(실제 T=45, Step 1~5 혼합). 중간 정보 일부 생략 → 빠른 saccade 동역학 시뮬레이션 + 학습샘플 증대.
- **Data Augmentation**: spatial translation, horizontal flip, temporal shifting(순환 시프트), event cutout. (vertical flip 제외 — 실제 안구 운동과 불일치)
- **손실함수**: $Loss = \frac{1}{T}\sqrt{\sum_{t=1}^{T}(y_{t,pred}-y_{t,label})^2}$ (time축 평균)
- 구현: PyTorch, RTX2080Ti ×4, 800 epoch, batch 32, Adam, Cosine Annealing Warm Restart, 초기 lr 0.001.

## 5. 실험
- **데이터셋**: ThreeET-plus (13 시나리오, 각 2~6 세그먼트, 640×480, 100Hz 라벨; random/saccade/reading/blinking/smooth pursuit). 75% train / 25% val. 100Hz 예측.
- **메트릭**: p5/p10/p15 성공률, perror(평균 유클리드 거리).
- **주요 결과 (Table 1, 모두 동일 multi-time-step 재학습)**:
  - BRAT(Ours): p5 0.978, p10 0.995, p15 1.000, **perror 1.14px** (최고).
  - MambaPupil: p5 0.963, p10 0.990, perror 1.34.
  - CB-ConvLSTM: p5 0.869, p10 0.970, perror 3.41.
  - CNN-GRU: p5 0.802, p10 0.945, perror 4.46.
- **Ablation**:
  - 커널: (7,5,5)가 최적(p10 0.981 w/o BRAT). BRAT 추가 시 (7,5,5)에서 perror 2.12→1.99.
  - Sampling step: step=1은 p5 최고(0.984), step↑ 시 fine-grained 저하. **Mixed가 종합 최고**(p10 0.995, perror 1.14).
- 정성: normal/fast saccade/blinking/glasses/resting 5케이스에서 BRAT가 가장 정확·강건.

## 6. 강점/한계
- **강점**: 상대위치 인코딩의 forward/backward 분리로 깜빡임·응시 같은 정적/가림 상황에 강건; 대형 커널 + spatial dropout으로 가림 시뮬레이션; multi-time-step로 데이터 희소성 보완; 명확한 SOTA(perror 1.14).
- **한계**: Transformer self-attention은 $O(T^2)$ → 긴 시퀀스에서 연산/메모리 부담(엣지 배치 시 부담, 본 논문은 파라미터/FLOPs/지연 미보고 → 경량성 정량화 **확인 불가**); 2D 표현 적분 방식이라 raw 이벤트의 완전한 시간 정밀도는 미활용(저자도 future work로 SNN 직접 처리 언급); 양자화/FPGA 관련 내용 없음.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **상대위치 bias의 forward/backward 선형 매핑($m^i\cdot|t-s|$)** 은 사인/코사인 절대 인코딩보다 정수/고정점 구현에 친화적일 수 있어, FPGA에서 attention bias를 LUT/누적기로 단순화 가능(추정).
- 그러나 self-attention의 $T\times T$ 행렬·softmax는 FPGA 저지연 가속의 병목 → **시퀀스 길이 T 축소 / sliding window attention / Bi-GRU 단독 경량 변형** 등이 이식 시 재설계 포인트(추정).
- spatial dropout(행 단위 가림 시뮬레이션) 아이디어는 학습 단계 기법이라 추론 HW에는 영향 없음(채택 가능).
- **대형 커널 CNN encoder**는 FPGA conv 가속기에 적합하나 K=7×7은 PE 자원·라인버퍼 증가 → 양자화(INT8) 및 커널 분해 고려(추정).
- multi-time-step 학습은 HW 비용 없이 정확도 향상 → 우리 학습 파이프라인에 무료로 채택 가능.
- 직접 비교군: MambaPupil(같은 데이터셋, SSM 계열)과 함께 정확도-경량성 트레이드오프 벤치마크에 유용.

## 8. 근거표기
- 4-6섹션 수식·구조·수치는 PDF 본문(pp.5145-5153) 직접 근거.
- 파라미터/FLOPs/지연/전력은 논문 미보고 → 경량성 정량 "확인 불가".
- 7섹션 FPGA/양자화 매핑은 분석자 해석 "추정"(논문 직접 기술 없음).
