# FAPNet: An Effective Frequency Adaptive Point-based Eye Tracker

## 1. 서지정보
- **제목**: FAPNet: An Effective Frequency Adaptive Point-based Eye Tracker
- **저자**: Xiaopeng Lin*, Hongwei Ren*, Bojun Cheng† (HKUST-Guangzhou) (*공동1저자, †교신)
- **연도**: 2024 (arXiv:2406.03177v1, 2024-06-05)
- **학회/저널**: CVPRW 2024 (AI for Streaming Workshop, Event-based Eye Tracking Challenge 관련 논문으로 추정) — arXiv 프리프린트
- **원본파일명**: fapnet.pdf

## 2. 문제정의·배경
- 아이트래킹은 AR/XR, 의료, 심리 등에서 핵심이나 빠른 안구 운동(>300°/s, 가속 24,000°/s²)을 포착하려면 kHz급 샘플링이 필요. 기존 고속 프레임 카메라는 고가·고전력·동작 중복(near-eye에서는 동공만 움직이고 나머지는 정적)이라는 한계.
- 이벤트 카메라(DVS)는 픽셀별 밝기 변화 발생 시 비동기 이벤트만 발생 → 저전력·고동적범위·고시간해상도·저지연. near-eye 동공 추적에 이상적.
- 그러나 기존 이벤트 기반 아이트래킹 네트워크는 (1) 이벤트의 sparse·fine-grained 시간 정보를 무시하고 frame/voxel 표현으로 변환(포맷 변환 비용, 정확도-효율 trade-off), (2) 과도하게 복잡한 모델로 엣지 배치가 어려움.

## 3. 핵심 기여
- 아이트래킹 분야에 **Point Cloud(이벤트 클라우드) 표현**을 최초 도입 — 동공 운동에 직접 대응하는 fine-grained 시간 정보를 활용.
- **경량 모델 설계**: local-global 공간 정보 집계 + long-short 시간 정보 상관(고정확도 + 저전력).
- **주파수 적응(Frequency Adaptive) 메커니즘**: 동공 운동 속도(이벤트 밀도)에 따라 추적 주기를 동적 조정 → 정확도 향상 + 연산 절감.
- 파라미터/FLOPs가 **센서 공간 해상도와 무관**(프레임 기반 대비 핵심 장점). SEET에서 PEPNet 대비 약 10% 연산으로 SOTA급 달성.

## 4. 방법론
### 이벤트 표현
- raw event cloud를 직접 입력 (포맷 변환 불필요). (x, y, t)를 좌표로 사용, t를 z축처럼 처리.
- **전처리/정규화**: 각 sample에서 N개 점 랜덤 샘플링하여 점 수 통일. S개 sample을 하나의 sequence로 입력. 공간은 해상도 w,h로 정규화, 시간은 sample 내 최소 timestamp를 빼고 [start,end] 길이로 나눠 정규화.
$$S'_i = \left(\frac{x}{w}, \frac{y}{h}\right), \quad T'_i = \frac{T_i - \min(T)}{\max(T) - \min(T)}$$

### 데이터 처리 (3단계)
- **Sliding Window Split**: 모션을 비중첩 10ms 윈도로 분할(각 sample은 100Hz 라벨에 대응).
- **Frequency Adaptive Window Expanding**: 10ms 윈도 내 이벤트 수가 임계값 미만이면 윈도를 양쪽으로 점진 확장(최대 100ms)하여 이벤트 수 확보 → 느린 동공 운동 구간(저밀도)의 p3 오차 대폭 감소.
- **Trajectory Augmentation**: 궤적 반전(trajectory inversion) 등으로 과적합 방지·일반화 강화.

### 모델 구조 (4개 모듈, PEPNet 기반 경량화)
1. **Sampling and Grouping**: FPS(Farthest Point Sampling) + KNN으로 local 공간·시간 정보 추출, 그룹별 표준화.
2. **Intra Group Aggregation**: MLP+residual(extractor) + attention으로 local spatio-temporal 특징 집계.
3. **Inter Group Aggregation**: Bi-LSTM + attention으로 그룹 간 global 시간 정보(timestamp 차원) 통합.
4. **Inter Sample LSTM**: sequence 내 S개 sample 간 long-term 시간 의존성 포착.
$$h_t = f(W_h \cdot x_t + U_h \cdot h_{t-1} + b_h), \quad y_t = V \cdot h_t + b_y$$
- 마지막 regressor가 동공 좌표 회귀.

### 학습법
- **손실함수**: weighted MSE (WMSE)
$$\mathcal{L} = w_x \cdot \frac{1}{n}\sum_{i=1}^{n}(x_{pred,i}-x_{label,i})^2 + w_y \cdot \frac{1}{n}\sum_{i=1}^{n}(y_{pred,i}-y_{label,i})^2$$
- AdamW, 초기 lr 1e-3 (100/120 epoch에서 감쇠), weight decay 1e-4, batch 256, RTX 4090.
- 구조 하이퍼파라미터: extractor MLP 차원 PEPNet=[64,128,256], FAPNet=[32,64,128]; Bi-LSTM hidden PEPNet=128/FAPNet=64; FAPNet sequence length S=20, Inter Sample LSTM hidden=128.

## 5. 실험
- **데이터셋**: (실제) EET+ (CVPR 2024 Challenge, 640×480, 학습 40/테스트 12 모션, 100Hz 라벨·20Hz 평가); (합성) SEET (LPW에서 v2e로 생성, 240×180).
- **메트릭**: p3/p5/p10 (오차 ≤ p 픽셀 성공률), 평균 유클리드 거리(mse px), Param/FLOPs.
- **SEET 주요 결과 (180×240)**:
  - FAPNet: Param 0.29M, FLOPs 58.7M, p3 0.920, p5 0.991, p10 0.996, mse 1.56px.
  - PEPNet: 0.64M, 443M, p3 0.918, p10 0.998, mse 1.57 → FAPNet은 약 1/10 연산으로 동급 정확도.
  - PointNet++: 1.46M, 1099M, p3 0.607 / PointNet: 3.46M, p3 0.322 / PointMLP-elite: p3 0.840.
  - 프레임 기반 CB-ConvLSTM(60×80): p3 0.889 (FAPNet이 p3 +3%, p5 +2% 우위, 해상도 무관 장점).
- **Challenge 결과 (vanilla PEPNet 사용)**: p3 49.08%, p5 80.67%, p10 97.95%, 평균 유클리드 거리 3.51.
- **Ablation**: 주파수 적응 윈도 → 저밀도 구간 오차 감소; Trajectory Augmentation → 누적 오차 분포 개선.

## 6. 강점/한계
- **강점**: 포맷 변환 불필요(raw event cloud 직접 처리); **연산량이 센서 해상도와 독립** → 고해상도 엣지에 유리; 초경량(0.29M/58.7M); 주파수 적응으로 저밀도 구간 강건; CIM/SNN 하드웨어(참고문헌 42,43)와의 연계 언급.
- **한계**: point sampling(N개 랜덤 샘플링)에 따른 정보 손실 가능; LSTM 기반 순차 의존으로 완전 비동기 추론에는 제약; 챌린지 본선은 FAPNet이 아닌 vanilla PEPNet 사용(FAPNet의 실제 챌린지 성능 미보고); 양자화/FPGA 직접 구현 결과는 없음(확인 불가).

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **해상도 독립 연산**은 FPGA 자원 예산 설계에 매우 유리(센서 변경 시 가속기 재설계 부담 감소). point 기반 입력은 sparse하므로 FPGA에서 event-driven 데이터패스에 적합할 수 있음(추정).
- **Frequency Adaptive Window**는 저밀도 구간 연산 스킵/동적 클럭게이팅 등 전력 절감 기법과 자연스럽게 매핑 가능(추정).
- 모듈 구성(FPS/KNN, MLP, attention, Bi-LSTM)에서 **FPS/KNN은 FPGA에서 비정형 메모리 접근으로 구현 난이도가 높음** → 양자화·고정점화 시 KNN 정렬·재배열이 병목이 될 수 있어 재설계 포인트(추정).
- WMSE 회귀 헤드는 단순하여 INT8 양자화에 무난할 것으로 보임(추정).
- PEPNet/EventMamba 계열과 같은 저자 라인 → 후속 비교군으로 활용 가능.

## 8. 근거표기
- 본문 4-7섹션의 수치·수식·구조는 PDF 본문(pp.1-8) 직접 근거.
- "추정" 표기: 7섹션 FPGA/양자화 매핑은 본 논문에 직접 기술 없음(저자 일반론 + 분석자 해석). 양자화/FPGA 구현 결과는 본 논문에 없어 "확인 불가".
