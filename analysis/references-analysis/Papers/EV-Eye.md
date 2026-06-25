# EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras

## 1. 서지정보
- **제목**: EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras
- **저자**: Guangrong Zhao, Yurun Yang, Jingwei Liu, Ning Chen, Yiran Shen*(교신), Hongkai Wen, Guohao Lan (Shandong Univ. / Univ. of Warwick / TU Delft)
- **연도**: 2023
- **학회/저널**: NeurIPS 2023 Track on Datasets and Benchmarks (37th)
- **코드/데이터**: github.com/Ningreka/EV-Eye (데이터셋 170GB+)
- **원본파일명**: 196_EV_Eye_Rethinking_High_fre.pdf

## 2. 문제정의·배경
- 기존 CCD/CMOS 카메라 기반 아이트래킹은 프레임율·대역폭 한계로 추적 주파수가 수백 Hz에 묶임(Tobii Pro Glasses 3=100Hz, Pupil Labs=200Hz). 그러나 정신질환 진단·시선 인증 등은 kHz급 필요. 인간 안구 saccade 각속도 최대 700°/s, 가속 24,000°/s².
- 고가 고속 카메라(EyeLink 1000, 1KHz)는 다운스트림 연산 부담 큼. 이벤트 카메라(DVS)는 sub-microsecond 지연·적응적 대역폭으로 대안이나, **대규모 데이터셋 부재**(기존 EBVEYE[Angelopoulos]는 fixation 위주·소수 피험자·sparse 라벨)와 **모델 기반 접근의 강건성 부족**이 한계.

## 3. 핵심 기여
- **EV-Eye 데이터셋**: 문헌상 최대·최다양 멀티모달 frame-event 고주파 아이트래킹 데이터셋(170GB+). 48명(남28/여20, 21~35세), DAVIS346 ×2로 **150만 near-eye grayscale + 27억 이벤트**, Tobii Pro Glasses 3로 **67.5만 scene 이미지 + 270만 gaze reference**. fixation/saccade/smooth pursuit 모두 포함.
- **하이브리드 frame-event 벤치마킹 방법**: near-eye grayscale(분할)+이벤트(고빈도 추적) 결합으로 **최대 38.4KHz** 동공 추적. 기존 EBVEYE 모델 기반 대비 동공·시선 추정 정확도 대폭 향상.

## 4. 방법론
### 이벤트 표현 / 센서
- 이벤트 트리거: $|\log I(x,y,t_{now}) - \log I(x,y,t_{previous})| < C$ (임계 $C$ 초과 시 즉시 이벤트 발생). 이벤트 = quadruplet $\{x,y,t,p\}$, $p\in\{+1,-1\}$.
- DAVIS346(346×240): 이벤트 스트림 + 25fps grayscale 동시 출력. Tobii(100Hz, FoV 95°×63°, 중심부 0.6° 오차)는 gaze reference 제공.

### 파이프라인 (하이브리드, Figure 3)
1. **Frame 기반 동공 분할 (저빈도)**: **U-Net**으로 binarized mask M 출력. 후처리(morphological closing으로 glint 노이즈 제거) → 중심 c, 경계 Q(에지 검출)를 동공 템플릿으로 추출.
2. **고빈도 이벤트 기반 동공 추적 (Template matching)**:
   - **Candidate Points Subset**: 동공 경계 Q의 평균 반경 $\bar\gamma$ 기준, 두 동심원 사이 이벤트만 선택(눈썹/눈꺼풀 노이즈 제거):
$$\lambda_1\bar\gamma < \|(x,y)-c\| < \lambda_2\bar\gamma, \quad (\lambda_1=0.8,\ \lambda_2=1.2)$$
   - **Points-to-edge Matching**: 후보 이벤트 집합 P를 경계 Q에 정렬하는 최적 translation T 탐색(ℓ2 최소화):
$$\min E(T) = \min \frac{1}{N}\sum_{i=1}^{N}\|q_i - (p_i + T)\|^2$$
     - 각 $p_i$의 최근접 $q_i'$(NN search) 찾고, 평균 변위 $\bar{\Delta T}_x,\bar{\Delta T}_y$로 이동 후 $T = T + \bar{\Delta T}$ 반복($\bar{\Delta T}/T<0.01$ 시 종료).
   - **Template Center Update**: $c^{t+1} = c^t - T$. 소수 이벤트(N=20)로 갱신 → 고빈도 추적.
3. **Gaze Estimation**: 동공 중심 $(x_c,y_c)$ → n차 다항 회귀로 PoG:
$$x_{Gaze} = a_0 + \sum_{u=1}^{n}\sum_{v=0}^{u} a_{(v,u)} x_c^{u-v} y_c^{v}$$ (피험자별 캘리브레이션으로 계수 획득)

### 학습 (분할 네트워크)
- PyTorch, Adam, ReduceLROnPlateau(dice 미개선 2 epoch 시 lr×0.1), 초기 lr 1e-3, batch 8, RTX 3090.

## 5. 실험
- **데이터셋/평가**: 9,011개 수동 라벨 frame(타원 라벨→inpolygon 마스크). user-independent(1명 test / 47명 train, leave-one-subject-out).
- **메트릭**: IoU, Dice(F1), PE(픽셀 오차, 동공 중심 유클리드 거리), DoD(시선 방향 차이, gaze 오차).
- **Frame 기반 분할 (DL-based vs Model-based[EBVEYE])**:
  - IoU 0.9187 vs 0.8360 (+8.27%), F1 0.9560 vs 0.9075 (+4.85%), **PE 0.64px vs 1.3px (50.7% 개선)**.
- **이벤트 기반 추적 (Matching vs Model-based)**: 평균 PE **7.7px → 1.2px (약 6.5× 감소)**. (이벤트당 갱신 N=20, EBVEYE와 동일)
- **추적 주파수**: 20 이벤트 누적마다 갱신, $F=1/T_{interval}$. CDF: 50%에서 2.6KHz, 95%에서 7.4KHz, **peak 38.4KHz**(saccade). 적응적(움직임 속도 비례) → 저속 시 저빈도로 연산·전력 절감.
- **Gaze 추적 (DoD)**: 평균 DoD **4.71° vs 9.72°**(model-based) — 상용 트래커에 더 근접.

## 6. 강점/한계
- **강점**: 최대 규모·다양 멀티모달(48명, 3종 안구운동, 270만 gaze ref) 공개; frame(강건 분할)+event(고빈도 추적) 결합으로 38.4KHz·고정확; user-independent 평가로 일반화 검증; 적응적 추적 빈도로 에너지 효율.
- **한계 (저자 명시)**: 라벨 부족(9,000장만 분할 라벨, 더 다양한 라벨 필요); gaze ground truth 부재(Tobii reference는 큰 시야각에서 부정확); 피험자 다양성 제한(단일 학술기관, 인종 편향); 안구운동 3종만(비디오 자극 등 미포함). **단, 양자화/FPGA 구현은 본 논문 범위 아님(확인 불가).**

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **EV-Eye는 우리 프로젝트의 핵심 데이터셋**: FACET·E-Track 등 후속 경량 검출기들이 EV-Eye를 학습/평가 기준으로 사용 → 우리 가속기의 정확도-효율 벤치마크 기준 데이터로 채택 적합(확인: FACET이 EV-Eye를 150만 라벨로 확장해 사용).
- **하이브리드 detection-tracking 구조**(무거운 분할 저빈도 + 경량 템플릿 매칭 고빈도)는 우리 시스템 설계 원형으로 매우 유의미: FPGA에서 "무거운 NN IP(저클럭/저빈도)"와 "경량 매칭 데이터패스(고빈도)"를 분리하면 저지연·저전력 달성 가능(추정).
- **Points-to-edge matching**(NN search + 평균 변위 반복)은 신경망 없이 비교/가산 위주라 **FPGA 고정점 데이터패스로 직접 구현 가능**(곱셈 거의 없음) → 초고빈도(kHz~38KHz) 추적단을 저비용으로 구현하는 강한 재사용 포인트(추정).
- **적응적 갱신(20 이벤트 누적 시 갱신)**은 이벤트 밀도 기반 클럭게이팅/연산 스킵으로 매핑 가능 → 정지/저속 구간 전력 절감(추정).
- **다항 회귀 gaze 매핑**은 소수 곱셈/덧셈으로 FPGA에서 LUT+MAC로 경량 구현 가능(추정). 단 피험자별 캘리브레이션 계수 저장 필요.
- U-Net 분할단은 무거우므로 RITnet/FACET 같은 경량 백본으로 대체 + INT8 양자화가 우리 가속기 적용 시 유리(추정).

## 8. 근거표기
- 4-6섹션 수식·구조·수치(0.64px/1.2px/38.4KHz/4.71° DoD 등)는 PDF 본문(pp.1-10) 직접 근거.
- 7섹션 FPGA/양자화 매핑은 분석자 해석 "추정"(논문에 FPGA/양자화 직접 기술 없음 → "확인 불가"). 단 "EV-Eye가 FACET 등의 학습 데이터로 쓰임"은 FACET 논문 교차 확인 근거.
