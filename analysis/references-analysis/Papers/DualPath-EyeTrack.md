# Dual-Path Enhancements in Event-Based Eye Tracking: Augmented Robustness and Adaptive Temporal Modeling

## 1. 서지정보
- **제목**: Dual-Path Enhancements in Event-Based Eye Tracking: Augmented Robustness and Adaptive Temporal Modeling
- **저자**: Hoang M. Truong, Vinh-Thuan Ly, Huy G. Tran, Thuan-Phat Nguyen, Tram T. Doan (University of Science, VNU-HCM, Vietnam)
- **연도**: 2025 (arXiv:2504.09960v1, 2025-04-14)
- **학회/저널**: Event-based Eye Tracking Challenge @ CVPR 2025 관련 (workshop submission, arXiv 프리프린트)
- **원본파일명**: Dual-Path Enhancements in Event-Based Eye Tracking Augmented Robustness and Adaptive Temporal Modeling.pdf

## 2. 문제정의·배경
- 이벤트 카메라는 µs 시간해상도·~10mW 초저전력(고속 카메라 대비 2자릿수 효율)이나 sparse·불규칙 출력이라 전통 CV 기법 부적합. AR/VR gaze-contingent 렌더링·신경질환 진단에 실시간·강건 시선추정 필요.
- 기존 한계: FACET(타원 피팅)은 가림/조명 변화 취약, 3ET(CB-ConvLSTM)는 순환구조로 엣지 지연, **Lightweight Spatiotemporal Network(Pei et al.)**는 causal·FIFO 효율적이나 **고정 시간 수용야(fixed temporal receptive field)** 때문에 가변 안구운동·실세계 노이즈(깜빡임 아티팩트, 안경 간섭)에 취약.

## 3. 핵심 기여
- **(1) Augmented Robustness**: Lightweight Spatiotemporal Network(baseline)에 **데이터 증강(temporal shift + spatial flip + event deletion)** 적용 → 효율 유지하며 강건성 향상. challenging 샘플에서 유클리드 오차 12% 감소(1.61 vs 1.70).
- **(2) KnightPupil (신규 하이브리드 아키텍처)**: EfficientNet-B3(공간) + Bi-GRU(시간 문맥) + **LTV-SSM(Linear Time-Varying State-Space Module, 적응적 상태 전이)** 결합으로 sparse 입력·노이즈 동적 대응.
- **(3) Dual-Strategy 프레임워크**: 배포가능 최적화 baseline + 탐구적 신규 구조를 동시 제시.

## 4. 방법론
### 데이터 증강 (이벤트 스트림 특화)
- **Temporal Shift**: $t_i' = t_i + \Delta t,\ \Delta t \sim \mathcal{U}(-200,200)$ms. 라벨 100Hz(10ms)이므로 라벨 인덱스 재계산 $L_j' = L_{j+\lfloor\Delta t/10\rfloor}$.
- **Spatial Flip**: $x'=W-x,\ y'=H-y$ (라벨도 동일 변환).
- **Event Deletion**: $P(\text{delete }e_i)=p=5\%$ (센서 노이즈·가림 시뮬레이션, 라벨 불변).

### 이벤트 표현 (KnightPupil)
- **Voxel Grid** $V\in\mathbb{R}^{H\times W\times T}$ (Tonic 구현, **3 bins**):
$$V(x,y,t) = \sum_{i=1}^{N} p_i \cdot \max\left(0,\ 1 - \left|t - \frac{T(t_i-t_{min})}{t_{max}-t_{min}}\right|\right)$$
- 선형보간으로 인접 bin에 분배, [-1,1] 정규화. 공간 다운샘플 0.125 → **80×60×3** voxel(0.3s window, 0.1s씩 3 subinterval).

### 모델 구조 (KnightPupil, Figure 5)
1. **EfficientNet-B3 (공간 백본)**: compound scaling ($\phi=1.8,\alpha=1.2,\beta=1.1,\gamma=1.15$), ImageNet 사전학습 + 전체 fine-tune. depthwise-separable conv로 연산↓. 출력 $F\in\mathbb{R}^{T\times d},\ d=1536$.
2. **Bi-GRU (시간 모델링)**: 2층, 방향당 hidden 128(총 256), dropout 0.3. 표준 GRU 게이트(reset $r_t$, update $z_t$):
$$h_t = (1-z_t)\odot h_{t-1} + z_t\odot\tilde{h}_t$$
3. **LTV-SSM (적응적 상태전이, 핵심)**: 고정 전이행렬 대신 GRU 출력 $h_t$에서 동적으로 학습:
$$\Delta A_t = \exp(\delta_t), \quad \Delta B_t = \delta_t\odot B_t, \quad h_t' = \Delta A_t h_t + \Delta B_t$$
$$y_t = C h_t' + h_t \mathbf{D}^T$$ ($\mathbf{D}$=identity 안정화). $\delta_t, B_t$는 $h_t$의 선형변환으로 학습.
   - **vs Mamba/GLA**: Mamba의 selective long-range conv·게이팅 없이 element-wise 곱셈 업데이트만 → 구조적으로 더 단순·경량. **parallel scan 미사용**(표준 순차 학습 루프).
4. **Final FC**: $Y = W_o H' + b_o$ (2D 좌표 회귀).

### 학습
- **baseline(Pei)**: batch 32(50 frame), 200 epoch, AdamW lr 0.002, wd 0.005, cosine+warmup(2.5%), AMP(FP16).
- **KnightPupil**: 600 epoch, batch 24, Adam lr 0.001, StepLR(200마다 ×0.5), Tonic DiskCachedDataset 캐싱. 단일 Tesla P100(Kaggle).

## 5. 실험
- **데이터셋**: 3ET+ 2024 & 2025 (13명, 2~6 세션, 640×480, 5클래스: random/saccade/reading/smooth pursuit/blink, 라벨 100Hz + blink indicator). 메트릭: 평균 유클리드 거리(Dist), p10(2024).
- **주요 결과 (Table 1)**:
  - **2024 p10**: Spatiotemporal w/aug **99.37** (w/o 99.16) — 증강이 명확 개선; KnightPupil 96.66→96.61(미세 하락).
  - **2025 Dist**: Spatiotemporal w/aug **1.61** (w/o 1.70, 약 12% 개선, **private test 최종 제출치**); KnightPupil 2.82→2.78.
- **증강 ablation (Table 2, Dist)**: Full aug = KnightPupil 3.08 / Spatiotemporal 1.61. 제거 시 — Temporal Shift 제거가 가장 큰 악화(3.28/1.67), 다음 Event Deletion(3.34/1.66), Spatial Flip(3.26/1.64).
- **핵심 관찰**: baseline(Lightweight Spatiotemporal)이 KnightPupil보다 정확도 우수(1.61 vs 2.78). 즉 신규 구조보다 **증강 적용 baseline이 실배포에 유리**.

## 6. 강점/한계
- **강점**: 이벤트 특화 증강(temporal shift/spatial flip/event deletion)이 HW 비용 0으로 강건성 향상; LTV-SSM은 Mamba보다 단순(element-wise, parallel scan 불요)해 경량; baseline+신규구조 dual 제시로 실용성·연구성 동시 확보.
- **한계**: **KnightPupil이 baseline보다 정확도 낮음**(EfficientNet-B3 백본이 d=1536로 무겁고 voxel/frame 변환으로 이벤트 시간정밀도 손실); 파라미터/FLOPs/지연/전력 정량 미보고(경량성 "확인 불가"); 단일 데이터셋(3ET+)만 검증; 양자화/FPGA 구현 없음.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **데이터 증강 3종(temporal shift/spatial flip/event deletion)** 은 학습 단계 기법으로 HW 무관 → 우리 학습 파이프라인에 무료로 채택해 강건성 확보 가능(특히 temporal shift 효과 최대). 강한 재사용 포인트.
- **Lightweight Spatiotemporal Network(Pei, baseline)** 가 KnightPupil보다 정확·경량 → **causal conv + FIFO 버퍼 + depthwise conv + sparse activation(L1)** 구조가 FPGA 저지연 스트리밍에 더 적합(추정). causal binning은 버퍼 최소화로 FPGA 파이프라인 친화적.
- **LTV-SSM의 element-wise 곱셈 업데이트(parallel scan 불요)** 는 FPGA에서 순차 누산기로 단순 구현 가능 → Mamba/GLA보다 HW 이식 비용 낮음(추정). 단 KnightPupil 정확도 열위라 채택 우선순위는 낮음.
- **EfficientNet-B3(d=1536)** 은 무거워 우리 저지연 목표엔 과함 → FACET의 MobileNetV3나 Pei의 경량 STN이 더 적합(추정). EfficientNet의 depthwise-separable conv 사상 자체는 INT8 양자화 친화적이나 B3 스케일은 부적합.
- voxel grid(3-bin) 변환은 FPGA에서 가산기+포화로 구현 가능하나 point/causal 표현 대비 메모리 큼 → trade-off 고려(추정).
- 직접 비교군: Pei Lightweight STN, FACET, 3ET(CB-ConvLSTM)와 동일 3ET+ 벤치마크 → 정확도-효율 곡선 작성에 활용.

## 8. 근거표기
- 4-6섹션 수식·구조·수치(Table 1/2, 1.61/1.70/99.37/d=1536/80×60×3 등)는 PDF 본문(pp.1-9) 직접 근거.
- 파라미터/FLOPs/지연/전력은 논문 미보고 → 경량성 정량 "확인 불가".
- 7섹션 FPGA/양자화 매핑은 분석자 해석 "추정"(논문 직접 기술 없음).
