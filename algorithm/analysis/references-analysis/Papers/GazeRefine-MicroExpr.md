# Inference-Time Gaze Refinement for Micro-Expression Recognition (Motion-Aware Post-Processing)

## 1. 서지정보
- **제목**: Inference-Time Gaze Refinement for Micro-Expression Recognition: Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing
- **저자**: Nuwan Bandara, Thivya Kandappu, Archan Misra (Singapore Management University)
- **연도**: 2025
- **학회/저널**: IJCAI-W'25 (Workshop for 4D Micro-Expression Recognition for Mind Reading)
- **코드**: github.com/eye-tracking-for-physiological-sensing/EyeLoRiN
- **원본파일명**: Inference-Time Gaze Refinement for Micro-Expression Recognition Enhancing Event-Based Eye Tracking with Motion-Aware Post-Processing.pdf

## 2. 문제정의·배경
- 마이크로표정(미세표정) 기반 정신상태 추론에 안구 행동(동공 확장, 깜빡임율, 안구 운동)이 핵심 신호. 이벤트 카메라는 고시간해상도·저지연으로 적합.
- 그러나 기존 이벤트 기반 동공 추적 모델은 (1) **깜빡임 아티팩트**로 인한 오예측, (2) **시간적 불일치**(생리적으로 연속/유계인 동공 운동을 강제하지 못해 급격한 점프), (3) **local 이벤트 분포 미활용**(예측 오프셋) 문제. 이벤트 데이터셋의 라벨 희소성도 보편적 강건 모델 개발을 어렵게 함.

## 3. 핵심 기여
- **모델 비종속(model-agnostic) 추론 시점 후처리** 프레임워크 — 기존 모델 재학습/구조변경 없이 출력만 정제:
  - (1) **Motion-Aware Median Filtering (M2F)**: 깜빡임 유발 스파이크 억제 + 자연스러운 시선 동역학 보존.
  - (2) **Optical Flow-Based Local Refinement (OFE)**: 누적 이벤트 모션과 정렬하여 공간 jitter·시간 불연속 감소.
- **Jitter Metric** 신규 제안: p-accuracy/픽셀거리가 못 잡는 **시간적 부드러움(궤적 연속성)**을 속도 규칙성·국소 신호 복잡도 기반으로 측정.

## 4. 방법론
### (a) Motion-Aware Median Filtering (Algorithm 1)
- 예측 궤적 $\mathbf{p}(t)=[x(t),y(t)]^\top$ 의 국소 모션 분산을 시간 윈도 내에서 추정(0~2차 운동학·공분산·주파수 방식 중 선택):
  - 변위: $d_t = \sqrt{(x_t-x_{t-1})^2+(y_t-y_{t-1})^2}$, $\bar{d}_t = \frac{1}{w}\sum_{i=-w/2}^{w/2}\|d(t+i)\|$
  - 속도: $V^{vel}_t = \frac{1}{w}\sum \|v(t+i)\|$, 가속도: $V^{acc}_t = \frac{1}{w}\sum \|a(t+i)\|$
  - 공분산: $V^{cov}_t = \|\Sigma_t\|_F$ (Frobenius norm), 주파수: $V^{freq}_t = \sqrt{Var_f(P_x)+Var_f(P_y)}$ (STFT 파워스펙트럼)
- 분산 변동성에 맞춰 **적응적 median 커널 크기**($w_{min}$~$w_{max}$ 클리핑, percentile 기반)로 롤링 median 적용 → 이상치(깜빡임/불안정) 억제하며 시간 일관성 유지.

### (b) Optical Flow-Based Local Refinement (Algorithm 2)
- 필터링된 예측 주변 ROI(크기 $R$, 1차 미분으로 적응)를 잡고, ROI 내 이벤트 수가 임계 초과 시 누적 벡터 흐름 $(dx,dy)$ 계산 → 정규화된 흐름 방향으로 예측을 소폭 shift하여 공간 오프셋 보정.

### (c) Jitter Metric (핵심 수식)
$$JM(\text{pred}, \text{true}) = \lambda\cdot\frac{\left|SPE_{\frac{d(\text{pred})}{dt}} - SPE_{\frac{d(\text{true})}{dt}}\right|}{\left|SPE_{\frac{d(\text{true})}{dt}}\right|+\varepsilon} + (1-\lambda)\cdot\log\left(1 + D_{KL}\left(P_{f[\frac{d(\text{pred})}{dt}]} \| P_{f[\frac{d(\text{true})}{dt}]}\right)\right)$$
- (1) **전역**: 속도 히스토그램의 KL divergence(comparative velocity entropy). (2) **국소**: SPARC 유도 spectral entropy(SPE) 차이.
$$SPE\left(\frac{dg(x,y)}{dt}\right) = -\sum_{f>0}\log(f+\varepsilon)\cdot\left(\frac{|V_f|}{\sum_f|V_f|+\varepsilon}\right)$$
- 값이 낮을수록 예측-실제 궤적의 시간적 부드러움이 유사. jerk 기반 지표보다 노이즈에 덜 민감.

## 5. 실험
- **데이터셋**: 3ET+ (event-based eye tracking 표준 벤치마크). **베이스 모델**: CB-ConvLSTM, bigBrains(Pei et al.). 단일 V100 GPU.
- **하이퍼파라미터**: M2F($w_{min}=5, w_{max}=20$, percentile 75, 기본 covariance); OFE($\tau=8, c=5, \gamma=2$).
- **메트릭**: p10/p5/p1, $l_2$(유클리드), $l_1$(맨해튼), JM.
- **주요 결과 (3ET+ validation, Table 1)**:
  - Ours(bigBrains 기반): p10 **99.99**, p5 **99.84**, p1 **59.87**, $l_2$ **0.80**, $l_1$ **0.65** (전 항목 최고).
  - bigBrains(원본): p10 99.00, p5 97.79, p1 45.50, $l_2$ 1.44.
  - MambaPupil: p10 99.42, $l_2$ 1.67 / FreeEVs: p10 99.26 / EyeGraph: p10 91.45.
- **Ablation (Table 4, challenge test set)**: bigBrains $l_2$ 1.500(private)→1.466(M2F만)→**1.423**(M2F+OFE), 약 **5.13%** 개선. ConvGRU 7.922→7.504.
- **연산 오버헤드 (Table 2)**: M2F ≈172 FLOPs/frame, OFE ≈340 FLOPs/frame, 학습 파라미터 0. 베이스 모델 비용의 **0.00048% 미만** → 실시간 엣지 배치 실용적.
- **Jitter (Table 3)**: bigBrains JM 0.4936 → 0.4372.

## 6. 강점/한계
- **강점**: 완전 model-agnostic·재학습 불필요(블랙박스 모델에도 적용); 학습 파라미터 0·극저연산(엣지 친화); 시간 평활성 전용 평가지표(Jitter Metric) 제시; 일관된 정확도 향상.
- **한계 (저자 명시)**: ocular 단일 모달(멀티모달 통합 미흡); 통제된 실험실 데이터만 검증(실제 머리 움직임·조명변화·가림 미검증); **베이스 모델의 근본 오류는 교정 못하고 평활만 함**(이미 합리적 정확도 모델의 향상 레이어).

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **후처리 모듈이 학습 파라미터 0, 수백 FLOPs/frame** → FPGA에서 소형 고정점 데이터패스(median 필터 = 정렬/비교 트리, 흐름 누적 = 가산기)로 구현 매우 용이. 신경망 가속기 출력단 뒤에 **저비용 후처리 IP**로 붙이기 적합(추정).
- median 필터·롤링 통계는 FPGA에서 sliding window + 비교기로 잘 매핑되며, 메인 추론 가속기와 병렬 파이프라인 가능 → 추가 지연 거의 없음(추정).
- **Jitter Metric**은 추론 HW가 아닌 평가/검증용 → 우리 시스템의 시선 출력 품질(부드러움) 벤치마크 지표로 채택 가능(데모/UX 안정성 정량화).
- OFE의 ROI 이벤트 흐름 누적은 **이벤트 스트림 직접 접근**을 요구 → FPGA에서 이벤트 버퍼/주소 디코더와 연동 설계 필요(추정).
- 우리가 어떤 백본을 쓰든 이 후처리를 무료로 얹어 정확도·안정성 향상 가능(강한 재사용 포인트).

## 8. 근거표기
- 4-6섹션 수식·알고리즘·수치는 PDF 본문(pp.1-18) 직접 근거. 특히 Table 1/2/4 수치, FLOPs 0.00048% 등.
- 7섹션 FPGA 매핑은 분석자 해석 "추정"(논문에 FPGA/양자화 직접 기술 없음 → 해당 부분 "확인 불가").
