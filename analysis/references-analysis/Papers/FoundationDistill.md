# DistillGaze: Rapidly deploying on-device eye tracking by distilling visual foundation models

## 1. 서지정보
- **제목**: Rapidly deploying on-device eye tracking by distilling visual foundation models
- **저자**: Cheng Jiang(Meta Reality Labs/Univ. of Michigan, 인턴), Jogendra Kundu, David Colmenares, Fengting Yang, Joseph Robinson, Yatong An(교신), Ali Behrooz (Meta Reality Labs)
- **연도**: 2026 (arXiv:2604.02509v1, 2026-04-02)
- **학회/저널**: arXiv 프리프린트 (Meta Reality Labs 기술보고서)
- **원본 파일명**: `Rapidly deploying on-device eye tracking by distilling visual foundation models.pdf`
- **시스템명**: DistillGaze

## 2. 문제정의·배경
- AR/VR 시선추적(ET)은 포비티드 렌더링·시선상호작용에 핵심이나, **신제품마다 하드웨어 구성(카메라 위치/포즈/조명) 변경** → 데이터 수집·라벨링·재학습 반복 비용 막대.
- 시선 GT 라벨 자체가 불완전 fixation·미준수로 noisy → 라벨 신뢰성 문제.
- **Visual Foundation Model(VFM)** (DINOv3, SAM3)은 자연영상엔 강하나 **근적외선(IR) near-eye 영상에는 부적합**: t-SNE 분석 결과 DINOv3 임베딩이 시선이 아니라 **피험자(subject) ID로 클러스터링**되어 직접 전이 실패(linear probe gaze error >5°). 단, 클러스터 내부에는 시선과 상관된 부드러운 gradient 존재 → 도메인 적응으로 추출 가능.
- → **합성 라벨 + 비라벨 실데이터**로 VFM을 적응시키고 경량 student로 distill = 실라벨 없이 빠른 on-device 배포.

## 3. 핵심 기여
- off-the-shelf DINOv3 표현의 한계를 t-SNE로 규명: subject 클러스터 지배, 단 intra-subject에 시선 구조 존재.
- **DistillGaze 2단계 프레임워크**: (1) 합성 라벨 + 비라벨 실데이터로 VFM teacher 최적화(실 GT 불필요), (2) 경량 256K student로 distill.
- Project Aria 대규모 벤치마크에서 SOTA: 합성-supervised 대비 **E50U50 58.62%, E90U90 42.91% 감소**, 모델 크기 증가 없음(256K).

## 4. 방법론
*주의: 본 논문은 프레임 기반 IR near-eye 영상 시선회귀이며, 이벤트 카메라 기반이 아님. 우리 프로젝트와는 "경량 on-device + 합성/비라벨 학습" 관점에서 연결됨.*

### 기본 ET 회귀 모델 (Sec 3.1)
- 좌/우 눈 이미지 쌍 $X=(X^L, X^R)$ → 공유 CNN 백본 $f$로 per-eye 특징 $h^L=f(X^L), h^R=f(X^R)$ → concat $h=[h^L;h^R]$.
- FC 회귀 헤드: $\hat y=g(h)=[\hat\theta^L,\hat\varphi^L,\hat\theta^R,\hat\varphi^R]\in\mathbb{R}^4$ (θ=yaw, φ=pitch).
- **Smooth L1 + outlier rejection 손실**(라벨 노이즈 강건): $|y-\hat y|<\beta$ 이면 quadratic, $\beta\le|y-\hat y|<\gamma$ 이면 linear, $\ge\gamma$ 이면 $k(\cdot)$로 down-weight (k<1).

### Stage 1: VFM 표현 최적화 (Sec 3.2)
- teacher/student 백본 모두 DINOv3 초기화. teacher는 weak aug $X_w$, student는 strong aug $X_s$.
- projected embedding: $z_t=p_t(f_t(X_w))$, $z_s=p_s(f_s(X_s))$.
- self-distillation loss: $\mathcal{L}_{SD}=\|z_t-z_s\|_2^2$.
- 합성 supervised: $\mathcal{L}_{SynSup}=\mathcal{L}_{Gaze}(y,\hat y_s)$. 비라벨 실데이터는 teacher 예측을 pseudo-label로: $\mathcal{L}_{Pseudo}=\mathcal{L}_{Gaze}(\hat y_t, \hat y_s)$.
- 총 손실: $\mathcal{L}=\lambda_{SynSup}\sum\mathcal{L}_{SynSup}+\lambda_{SD}[\mathcal{L}_{SD}+\sum\mathcal{L}_{Pseudo}]$. **코사인 스케줄**로 초기엔 합성 supervision 강조 → 점차 self-distillation/pseudo로 전환.
- teacher는 EMA 업데이트: $\theta_t\leftarrow\alpha\theta_t+(1-\alpha)\theta_s$ (collapse 방지).

### Stage 2: On-device student로 distill (Sec 3.3)
- 3개 모델: teacher $f_t$(최적화된 VFM), student $f_s$(합성 사전학습 on-device 모델), EMA student $f_e$.
- Community KD 기반이나 차이점: (1) **feature space(VIC-KD)** + output space(pseudo-label) 동시 distill, (2) 한 student를 gradient 대신 EMA로 갱신.
- teacher-student feature distill(분산-불변-공분산 정규화): $\mathcal{L}_{KD}=\lambda_{inv}\|z_t-z_{s\to t}\|_2^2+\lambda_{var}v(z_{s\to t})+\lambda_{cov}c(z_{s\to t})$.
- teacher pseudo-label: $\mathcal{L}_{Pseudo\text{-}t}=\mathcal{L}_{Gaze}(\hat y_t,\hat y_s)$.
- EMA self-distill: $\mathcal{L}_{SD}=\|z_e-z_{s\to e}\|_2^2$, $\mathcal{L}_{Pseudo\text{-}e}=\mathcal{L}_{Gaze}(\hat y_e,\hat y_s)$.
- 총 손실: $\mathcal{L}=\lambda_t(\mathcal{L}_{KD}+\mathcal{L}_{Pseudo\text{-}t})+\lambda_e(\mathcal{L}_{SD}+\mathcal{L}_{Pseudo\text{-}e})$. λ_e 코사인 스케줄.
- 추론 시 student $f_s$만 배포.

### 구현
- on-device: **FBNet 백본 공유, 256K params**. teacher: DINOv3 ViT-B(86M). AdamW, batch 256, 50K iter, 4×A100.
- 증강: weak(gamma jitter, scaling) / strong(추가로 카메라 열화 MTF·motion blur·압축, 조명 brightness/contrast·glint inpainting·random shadow·coarse dropout, 각 p=0.3).

## 5. 실험
- **데이터셋**: Project Aria(2,222명, 6,299녹화, ET 카메라 640×480→320×240 다운샘플, 20fps). 학습셋 1,825명의 GT는 미사용(비라벨 모사). 합성: Blender 165K 프레임 998명.
- **메트릭**: 3D gaze 각오차(도). EU 테이블 - 사용자별 per-frame 분포를 E50/E75/E90 percentile로 요약 후 사용자 간 U50/U75/U90 집계. E50U50=중앙 사용자 중앙오차, E90U90=tail.
- **주요 결과(Table 2)**:
  | 방법 | Inf params | E50U50↓ | E90U90↓ |
  |---|---|---|---|
  | On-device 합성 supervised(baseline) | 256K | 3.48 | 14.84 |
  | DINOv3 linear probe | 86M | 5.47 | 18.74 |
  | DINOv3 합성 finetune | 86M | 2.01 | 12.11 |
  | **Optimized VFM(teacher)** | 86M | **1.33** | 8.32 |
  | **DistillGaze(student)** | 256K | **1.44** | 8.45 |
  | Fully supervised(상한, 참고) | 256K | 0.82 | 5.91 |
- DistillGaze student는 256K로 86M teacher에 근접(58.62% 개선 vs baseline), tail에선 pseudo-label only가 최강(E90U90 8.19).
- **Ablation**: DINO loss보다 **단순 MSE가 우수**(1.33 vs 1.50; 합성 supervision이 collapse 방지 역할). ViT-B가 ConvNeXt-S보다 일관 우수. 합성 supervision/스케줄러 제거 시 성능 급락(linear probe보다도 나빠짐).

## 6. 강점/한계
- **강점**: **실라벨 GT 없이** 합성+비라벨로 VFM 적응 → 하드웨어 변경 시 빠른 재배포. 256K 경량 student로 86M teacher 성능 거의 보존(지식전이 효과 입증). 라벨 노이즈 강건(Smooth L1+outlier reject).
- **한계**: fully-supervised 상한과 격차 잔존(특히 tail). 단일 디바이스 구성에서만 평가(다중 카메라 지오메트리 미검증). 프레임 기반(이벤트 카메라 아님), 연산/latency 실측·HW 가속 보고 없음.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)
- **직접적 HW 시사점은 약함**(이벤트 카메라 아님, FPGA 무관, latency 미보고). 그러나 **학습 파이프라인 관점에서 가치**.
- **데이터 부족 대응 레시피(추정)**: 우리 이벤트 기반 시스템도 합성 이벤트(v2e 등)+비라벨 실데이터로 경량 student를 distill하면 실 GT 수집 비용 절감 가능. SynUnlabeled-PupilTrack 논문과 동일 철학.
- **256K급 student로 대형 VFM teacher 성능 보존** = 우리 FPGA 타깃 초경량 모델(BRAM 예산 내)을 만들 때 **KD(VIC-KD/pseudo-label)** 채택의 강한 근거. teacher는 오프라인(GPU), student만 FPGA 배포.
- **Smooth L1 + outlier rejection**(분기형 손실)은 회귀 출력 헤드 학습 시 채택 가치(노이즈 라벨 강건). 추론 시점 연산엔 영향 없음.
- **단순 MSE > DINO loss** 교훈: 시선회귀 같은 연속 회귀에서는 soft-clustering 목적함수가 오히려 해 → 우리 회귀 헤드 학습 목적함수 설계 참고.
- **주의(추정)**: VFM teacher(ViT-B 86M)는 학습용일 뿐 배포 대상 아님 → 우리 FPGA 가속 대상은 student(FBNet급). 이벤트 입력에 맞게 백본은 재설계 필요.

## 8. 근거표기
- 구조/수식(식1~13)/Table 1~8/Figure 1~10 수치는 본문(arXiv p.1~22)에서 직접 확인.
- "우리 프로젝트(이벤트/FPGA) 적용" 해석은 본 논문이 프레임 IR·GPU 학습 중심이라 직접 근거 없음 → **추정**.
- student 백본 FBNet의 정확한 레이어 구성, 하이퍼파라미터(λ 값들, k/β/γ)는 본문 미기재 → **부분 확인/확인 불가**.
