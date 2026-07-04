# Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN (SEE)

## 1. 서지정보
- **제목**: Co-designing a Sub-millisecond Latency Event-based Eye Tracking System with Submanifold Sparse CNN
- **저자**: Baoheng Zhang*, Yizhao Gao*, Jingyuan Li, Hayden Kwok-Hay So (The University of Hong Kong) (*동등기여)
- **연도**: 2024
- **학회/저널**: CVPR Workshop (CVPRW, AIS 2024 Event-based Eye Tracking 관련)
- **원본 파일명**: `Zhang_Co-designing_a_Sub-millisecond_Latency_Event-based_Eye_Tracking_System_with_Submanifold_CVPRW_2024_paper.pdf`
- **시스템명**: SEE. 코드: https://github.com/CASR-HKU/ESDA/tree/eye_tracking

## 2. 문제정의·배경
- VR/AR 시선추적은 **저지연·저전력·고정밀** 세 가지를 동시에 요구하나 균형이 어려움. 프레임 카메라+dense DNN은 정확하나 지연 ~25ms.
- 이벤트 카메라는 픽셀단위 변화만 출력 → 희소·고시간해상도 → 저지연 잠재력. 그러나 GPU/CPU는 이벤트 희소성/고속성을 잘 활용 못함.
- 본 연구는 **HW/SW co-design**으로 이 격차 해소: 이벤트 공간 희소성을 중심으로 한 submanifold sparse CNN(SCNN)을 FPGA dataflow 가속기로 처리.

## 3. 핵심 기여
- **SEE**: SCNN(특징추출) + GRU(시간융합) + FC(눈 중심 회귀)로 구성된 HW/SW 공동최적화 시스템.
- **Submanifold Sparse CNN**을 voxel grid 이벤트 표현에서 비제로 활성만 처리 → dilation 효과 회피·희소성 보존.
- **이종(heterogeneous) FPGA SoC** 구현: SCNN은 FPGA PL의 sparse dataflow 가속기(Int8), GRU+FC는 Arm Cortex-A53 + NEON SIMD(float).
- **HW/SW co-optimization 탐색 프레임워크**: MobileNetV2 supernet + AGNA 하드웨어 시뮬레이터로 latency-accuracy Pareto frontier 모델 자동 선택.
- AIS2024 데이터셋에서 **0.7ms latency, p5 81%, p10 99.5%, Mean Euclidean Distance 3.71, 2.29mJ/inference** 달성.

## 4. 방법론
### SW 모델 구조
- 입력: 표준 **voxel grid** 표현(고정 시간구간 이벤트, 공간 희소). 출력: 정규화된 눈 중심 (0~1) → 입력 H,W 곱으로 픽셀좌표 복원.
- SCNN backbone → GRU(프레임 간 시간정보) → FC.

### Submanifold Sparse Convolution
- 표준 conv는 희소 입력에서 dilation으로 출력이 더 dense해짐. SCNN(Minkowski 계열)은 **입력 비제로 위치 = 출력 비제로 위치** 유지. 유효 위치에서는 표준 conv와 동일 계산 → 희소성 보존 + 불필요 연산 제거.

### 양자화 (HAWQv3, dyadic int8)
- $Y = S_y\hat Y = W\times X = S_w\hat W \times S_x\hat X$, $\hat Y = \frac{S_w S_x}{S_y}(\hat W\times\hat X)=\frac{\hat S}{2^n}(\hat W\times\hat X)$. 스케일 나눗셈을 정수곱+shift로 대체(고정소수점 유사) → 단순 HW 산술.

### HW 설계 (Xilinx Zynq UltraScale+ MPSoC, ZCU102)
- **FPGA SCNN 가속기**: ESDA[15]의 dynamic sparse dataflow 채택. 모든 레이어를 on-chip에 공간적으로 매핑·파이프라인. 통합 token-feature streaming 인터페이스, token `[.x, .y, .end]`로 비제로 좌표 표시.
  - dataflow 모듈 3원칙: (1) 다음 비제로 좌표 해소, (2) 해당 위치 특징 계산, (3) left-to-right/top-to-bottom 스트리밍 순서.
  - Submanifold conv 3x3: Sparse Line Buffer(SLB) + compute engine. 입출력 비제로 위치 동일 → token FIFO로 버퍼·재사용. kernel offset stream으로 3x3 내부 희소성 활용. 가중치는 on-chip BRAM에 정적 저장(off-chip 통신↓, 단 모델크기 제한).
- **GRU+FC는 CPU(NEON SIMD, Eigen C++)**: sigmoid 등 비선형이 FPGA 양자화 어려워 부동소수점으로 호스트 처리. PYNQ 통합.

### Co-optimization 탐색
- search space: (1) inverted bottleneck block 수, (2) 블록별 채널, (3) expansion ratio, (4) GRU hidden size. AGNA(geometric programming)로 latency 추정 → 저지연·feasible 모델 학습 → Pareto frontier 선택.

## 5. 실험
- **데이터셋**: Event-based Eye-Tracking-AIS2024 (13 subjects, 각 2~6 세션, 5종 활동). default split.
- **메트릭**: Mean Euclidean Distance(Dist.), p5/p10 accuracy.
- **구현**: Vitis HLS 2020.2 + Vivado 2020.2, ZCU102 보드.
- **Standard vs Submanifold**: p5/p10 정확도 유사(MobileNetV2 87.42→87.63 p5, SEE-B 84.87→85.21), 활성 희소성은 크게 증가.
- **HW 구현 상세(Table 2)**:
  | 모델 | p5 | p10 | Dist | #Param | Total Latency | Power | mJ/inf | 
  |---|---|---|---|---|---|---|---|
  | MobileNetV2 | 87.36 | 99.53 | 3.15 | 797K | 1.45ms | 4.36W | 3.23 |
  | SEE-A | 80.83 | 99.60 | 3.77 | 465K | 0.64ms | 4.05W | 1.99 |
  | SEE-B | 83.32 | 99.53 | 3.39 | 372K | 0.94ms | 4.17W | 3.28 |
  | SEE-C | 75.92 | 98.39 | 4.05 | 180K | 0.60ms | 3.86W | 1.88 |
  | SEE-D | 81.37 | 99.53 | 3.71 | 178K | 0.70ms | 3.86W | 2.29 |
  - SCNN(FPGA)와 GRU&FC(CPU) latency 분리 기록(예: SEE-D 0.59ms + 0.11ms). 자원: DSP/BRAM/FF/LUT 기재.
- **임베디드 GPU 대비(Jetson Xavier NX, batch=1)**: SEE가 표준 구현 대비 11.47~13.89배, submanifold GPU(MinkowskiEngine) 대비 57.4~72.6배 speedup. (GPU submanifold는 sparse 좌표 bookkeeping 오버헤드로 dense보다 느림)

## 6. 강점/한계
- **강점**: **실제 FPGA SoC end-to-end 구현 + sub-ms latency + mJ급 에너지** — 우리 프로젝트와 직접 정합. dataflow + 정적 on-chip 가중치 + Int8 dyadic 양자화로 결정적 저지연. Pareto 자동탐색으로 accuracy-latency-resource trade-off 제공.
- **한계**: GRU+FC가 FPGA가 아닌 CPU에서 실행(sigmoid 양자화·비선형 미지원) → 완전 FPGA화 미달. on-chip 버퍼 용량이 모델 크기를 제한. inter-batch 파이프라인 미지원(future work). submanifold conv 정확도가 standard보다 p5에서 다소 낮음.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)
- **가장 직접적인 참조 논문 중 하나** (FPGA 저지연 on-device 시선추적 co-design). 우리 가속기 설계의 baseline/비교군으로 활용 가치 높음.
- **재사용 포인트(추정)**:
  - **Sparse dataflow 아키텍처(ESDA 기반)**: token-feature streaming + SLB + kernel offset로 희소 conv를 결정적 파이프라인으로 구현 — 우리 RTL/HLS 설계의 골격 후보.
  - **Int8 dyadic 양자화(HAWQv3)**: 스케일 나눗셈을 곱+shift로 대체 → FPGA 산술 단순화. 우리 양자화 파이프라인에 채택 검토.
  - **정적 on-chip BRAM 가중치**: off-chip 대역폭 제거로 latency 결정성↑. 단 모델 크기 제약 → 경량 백본 필수.
  - **HW 시뮬레이터(AGNA, geometric programming) 기반 DSE + Pareto frontier**: 우리 algo2fpga DSE 흐름과 정합. 도입 가치.
- **우리가 개선 가능한 갭**: 이 논문은 GRU+FC를 CPU로 오프로딩 → **순환/어텐션 모듈의 FPGA 양자화·비선형 구현**이 미해결 과제(저자도 future work로 명시). 우리 프로젝트에서 GRU/SSM/sigmoid를 LUT/CORDIC로 FPGA에 온칩 통합하면 차별화 가능.

## 8. 근거표기
- 구조/수식/Table 1~2/speedup/자원수치는 본문(p.5771~5779)에서 직접 확인.
- "우리 프로젝트와의 정합·개선 갭" 해석은 **추정**(논문은 XR-FPGA 일반론, 우리 특정 목표 미언급). AGNA·ESDA 세부 내부동작은 참고문헌[10][15] 의존이라 본문만으로는 **부분 확인**.
