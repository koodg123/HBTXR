# A Lightweight Spatiotemporal Network for Online Eye Tracking with Event Camera

## 1. 서지정보
- **제목**: A Lightweight Spatiotemporal Network for Online Eye Tracking with Event Camera
- **저자**: Yan Ru Pei, Sasskia Brüers, Sébastien Crouzet, Douglas McLelland, Olivier Coenen (Brainchip Inc.)
- **연도**: 2024
- **학회/저널**: CVPR Workshop (CVPRW, AIS 2024 Event-based Eye Tracking Challenge)
- **원본 파일명**: `Pei_A_Lightweight_Spatiotemporal_Network_for_Online_Eye_Tracking_with_Event_CVPRW_2024_paper.pdf`
- **코드**: https://github.com/PeaBrane/eye_track_spatiotemporal
- **성과**: AIS2024 챌린지 private testset p10 **0.9916** (벤치마크 모델 0.992)

## 2. 문제정의·배경
- 이벤트 데이터는 edge 환경(효율·저지연 중요)에서 흔함. 고시간해상도(sub-ms)로 미세 움직임을 포착.
- 그러나 보통 이벤트를 프레임으로 "binning"하면 시간정보가 크게 손실됨(너무 짧으면 정보부족, 너무 길면 저지연 이점 상실).
- 공간 CNN은 시간 연속성이 없어 스트리밍 온라인 추론에 비효율. ConvLSTM 같은 recurrent head는 깊은 곳에서만 시간모델링하고 학습이 어려움.
- → **완전 causal한 spatiotemporal CNN**으로 효율적 온라인 추론 제안.

## 3. 핵심 기여
- **경량 fully-causal spatiotemporal CNN**: FIFO 버퍼로 온라인 추론(모든 시간프레임 저장 불필요).
- **Causal event (volume) binning** 전략: 지연 최소화·과도한 버퍼링 회피.
- **L1 activation 정규화**로 레이어당 활성 희소성 >90% 달성 → 이벤트 프로세서에서 큰 효율 이득(최대 5배 속도).
- **이벤트 직접 affine 증강**(spatial/temporal) 전략으로 데이터 부족 완화.
- BatchNorm/GroupNorm 교대 정규화(causality 유지).

## 4. 방법론
### 이벤트 표현
- 이벤트 $E=(p,x,y,t)$. **Event volume binning**으로 (2, T, H, W) 텐서 생성(이벤트 cuboid가 인접 bin에 기여, bilinear 보간 유사):
$V_+ = \sum_{p_i=1} k(\frac{x_b-x_i}{\Delta x_b})k(\frac{y_b-y_i}{\Delta y_b})k(\frac{t_b-t_i}{\Delta t_b})$ (V_-도 동일), $k(\chi)=\max(|1-\chi|,0)$ (triangle filter).
- **Causal event volume binning**: 시간 필터를 half-triangle로: $k(\tau)=H(\tau)\max(|1-\tau|,0)$, H는 Heaviside step. 미래 이벤트($t_i>t_b$) 불필요 → bin이 $t_b$까지 이벤트만 받으면 즉시 스트리밍 → 전처리 지연↓.

### 데이터 증강 (이벤트에 직접)
- **Spatial affine**: homogeneous 좌표에서 $A=TRS$ (translation·rotation·scaling). $s_x,s_y\in[0.8,1.2]$, 회전 [-15°,15°], translation [-0.2,0.2]. 동공 라벨도 동일 행렬로 변환.
- **Temporal affine**: 타임스탬프 $at+b$ (a∈[0.8,1.2], b=0). binning 후 0.5확률로 시간축 반전(극성도 swap).

### 네트워크 구조 (Fig.1A, CenterNet 유사 head)
- backbone: spatiotemporal block 5개 스택. 각 block = **temporal conv($k_t\times1\times1$) → spatial conv($1\times k_x\times k_y$)** = (1+2)D conv (pseudo-3D).
  - **Causal 보장**: 입력을 $k_t-1$만큼 pre-pad → 현재 출력이 현재+과거 프레임에만 의존(지연 0). 덕분에 긴 temporal kernel $k_t=5$ 사용 가능.
  - **temporal을 spatial보다 먼저**: 첫 temporal layer가 전처리 이벤트에 직접 접근(시간특징 smear 방지) → (1+2)D.
  - **residual connection 미사용**(mobile HW에서 skip 버퍼 비용↑).
  - depthwise-separable 옵션으로 추가 경량화.
- **Causal Group Norm**: temporal conv 뒤 GroupNorm(통계를 (H,W)만으로 계산해 시간 비혼합 → causal), spatial conv 뒤 BatchNorm(고정통계). 혼합 정규화로 small/large batch 모두 안정.
- **온라인 추론(FIFO)**: 각 temporal layer 입력에 깊이=$k_t$ FIFO 버퍼. conv = 슬라이딩윈도우(FIFO) × kernel dot product. (Algorithm 1: CONCAT으로 FIFO 갱신 → einsum 시간축 축약 → GroupNorm → ReLU → Conv2D spatial(stride 1,2,2) → BatchNorm → ReLU)
- **Detector head(CenterNet 유사)**: backbone 출력 C×T×3×4 → temporal smoothing → 3×3 conv + ReLU + 1×1 conv + sigmoid → 3×T×3×4 (각 grid-cell: pupil 존재확률, x/y offset).

### 손실
- grid-cell별 focal loss + 회귀손실: $\ell=-(1-\hat p)^\gamma\log(\hat p)+\ell_{reg}$ (p=1), $\hat p^\gamma\log(1-\hat p)$ (p=0), $\gamma=2$.
- $\ell_{reg}$ = SmoothL1 ($\beta=0.11$). 모든 grid-cell·유효 프레임 평균.
- **Activity regularization**: ReLU 출력 L1 norm(출력 볼륨 정규화·가중치 스케일) → 희소성 유도.

## 5. 실험
- **데이터셋**: AIS2024 (13 subjects, 480×640, 라벨 100Hz; 평가 20Hz, 60×80 다운샘플 공간에서 p10).
- **학습**: batch 32(각 50 프레임), 200 epoch, AdamW lr 0.002 wd 0.005, cosine decay+warmup, AMP+torch.compile.
- **결과(Table 1)**: public test p10 0.988, private 0.992, validation 0.963 (전체 0.9898).
- **Ablation 증강(Table 2)**: spatial affine이 결정적(+0.375; 없으면 p10 0.588). temporal flip 소폭 향상, temporal scale은 오히려 악화.
- **Ablation 구조(Table 3)**: causal event volume binning은 정확도 영향 작으나 causality 보장. CenterNet head가 +0.027. temporal kernel 5가 3보다 우수. 정규화는 batch32에서 All BN이 약간 우수하나 mixed가 batch에 강건(Fig.2). Conv3D는 p10 0.969(최고)이나 MACs 267M(vs (1+2)D 55.2M)로 매우 비쌈.
- **효율-정확도(Fig.3)**: 다운샘플 8(60×80)에서도 성능 소폭 하락, 연산 1/3. DWS 레이어로 추가 절감. **L1 정규화로 >90% 활성 희소성 + 최소 성능손실 → 희소성 활용 프로세서에서 5배 추론 가속**. benchmark 809K params, 55.2M MACs/frame.

## 6. 강점/한계
- **강점**: **완전 causal** → 진짜 온라인/저지연 스트리밍에 적합(MambaPupil의 양방향과 대비되는 큰 장점). FIFO 버퍼 기반 추론은 HW 매핑이 명확. (1+2)D 분해 + DWS + >90% 희소성으로 매우 경량. 연산이 conv+ReLU로 단순.
- **한계**: 활성 희소성 이득은 "희소성 활용 가능한 프로세서(neuromorphic 등)"에서만 5배(일반 dense FPGA/GPU는 정적 희소성 활용이 어려움). 정확도 최고점은 Conv3D(비쌈). 특정 챌린지 데이터셋 위주 평가.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)
- **FIFO 버퍼 기반 causal temporal conv는 FPGA 스트리밍 데이터패스에 이상적** (추정): conv를 슬라이딩윈도우(shift register/FIFO) + dot product로 환원 → 우리 RTL/HLS에서 결정적 latency·작은 버퍼로 직접 구현 가능. temporal kernel=5는 깊이 5 shift register.
- **(1+2)D(temporal→spatial) 분해**는 3D conv 대비 MACs 1/5 → FPGA DSP/리소스 예산에 유리. 우리 가속기 백본 구조 후보.
- **Causal event volume binning**(half-triangle filter)은 전처리 지연을 제거 → 우리 입력 인터페이스 설계에 채택 가치. triangle 가중치는 곱+덧셈으로 HW 경량.
- **BatchNorm은 추론 시 고정통계 → conv에 fold 가능(양자화 친화)**. 단 GroupNorm은 동적통계라 FPGA에서 평균/분산 누적기 필요 → 비용 평가 필요(또는 All-BN 변형 검토).
- **주의(추정)**: >90% 희소성 5배 이득은 dynamic sparsity 처리 가능 HW 전제 → 일반 dense FPGA 파이프라인에서는 이득이 제한적. Zhang-Submanifold(SEE)의 sparse dataflow와 결합하면 우리 환경에서 희소성 활용 가능.

## 8. 근거표기
- 구조/수식/Algorithm 1/Table 1~3/Fig.2~3 수치는 본문(p.5780~5788)에서 직접 확인.
- "FPGA 매핑·희소성 이득 제약" 해석은 본 논문에 FPGA 실측 없음(Brainchip neuromorphic 지향) → **추정**.
- 정확한 채널 진행(8/16/32...256)은 Fig.1A에 도식화되어 있으나 일부 하이퍼파라미터(GroupNorm groups=4 외)는 코드 의존 → **부분 확인**.
