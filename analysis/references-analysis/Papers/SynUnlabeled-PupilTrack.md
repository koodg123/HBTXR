# Overcoming Data Scarcity for Event-Based Pupil Tracking with Synthetic and Unlabeled Data

## 1. 서지정보
- **제목**: Overcoming Data Scarcity for Event-Based Pupil Tracking with Synthetic and Unlabeled Data
- **저자**: Aaron Tognoli, Andrea Simpsi, Andrea Aspesi, Marco Cannici, Luca Merigo, Matteo Matteucci, Simone Mentasti (Politecnico di Milano + EssilorLuxottica, Smart Eyewear Lab)
- **연도**: 2026 (Proc. ACM Hum.-Comput. Interact. Vol.10 No.3, Article ETRA018, May 2026; DOI 10.1145/3806032)
- **학회/저널**: ACM ETRA 2026 (Proc. ACM Human-Computer Interaction)
- **원본파일명**: Overcoming Data Scarcity for Event-Based Pupil Tracking with Synthetic and Unlabeled Data.pdf

## 2. 문제정의·배경
- 스마트 안경(always-on smart eyewear)에서 동공추적은 필수이나, 엄격한 전력·폼팩터·연산 제약과 **라벨 데이터 부족**이 큰 장애.
- 이벤트 카메라는 저전력·μs 지연·bandwidth-latency trade-off 회피로 근안 추적에 적합하나, 공개 이벤트 데이터셋은 소수·coarse 라벨(화면좌표 또는 저시간해상도)뿐. 최대 데이터셋도 66명 수준.
- 핵심 과제: 제한된 real 라벨 + 합성 이벤트 + unlabeled real을 결합해 **real-world 일반화**가 강한 이벤트 동공추적 학습.

## 3. 핵심기여
- **U2Eyes 렌더러 + v2e 이벤트 시뮬레이터** 결합으로 합성 이벤트 스트림 생성. U2Eyes를 확장해 corneal refraction 기반 **정확한 동공중심 라벨** 생성.
- **합성 이벤트로 학습한 네트워크가 합성 이미지로 학습한 것보다 sim-to-real gap이 작음**을 입증(이벤트는 texture/illumination 불변성).
- **3단계 학습 파이프라인**: (1) unlabeled real 이벤트 label-free pretraining → (2) 합성 이벤트 supervised → (3) 제한된 real 라벨 fine-tuning. 추가 수작업 라벨 없이 real 전이 일관 개선.
- 4개 공개 데이터셋(EV-Eye, 3ET+, INI-30, Angelopoulos/EB-NEG)에서 일관된 정확도 향상. real-only 대비 최대 **25%** 개선.

## 4. 방법론
### Problem formulation
- 이벤트: $\mathcal{E} = \{e_i = (u_i, v_i, t_i, p_i)\}_{i=1}^{N}$, $p_i \in \{+1, -1\}$. 도메인: $\mathcal{D}^{img}_{sim}, \mathcal{D}^{img}_{real}, \mathcal{D}^{evt}_{sim}, \mathcal{D}^{evt}_{real}$.
- 학습 regime: sim-only, real-only, joint, train+finetune, three-stage, zero-shot.

### 합성 이벤트 렌더링
- U2Eyes(UnityEyes 기반 3D 안구 모델, 눈꺼풀 애니메이션, 물리기반 안구 재질) 확장: corneal refraction을 명시 모델링하여 oblique 각도 동공중심 오차 해소. 굴절 동공 binary mask 렌더 후 centroid로 픽셀정확 라벨. IR 광원·corneal reflection 지원.
- S자 gaze trajectory를 1000Hz 고주파 렌더 → v2e로 이벤트 변환. v2e 파라미터(contrast threshold $\Theta_+, \Theta_-$, $\sigma_\Theta$) 수동 조정(렌더 이미지는 이벤트율 과대평가 경향).

### 모델: EETNet (on-device CNN)
- MAX78000 저전력 CNN 가속기 대상의 경량 CNN. 3개 conv block(각 2 conv+ReLU+maxpool) + 2 FC 회귀헤드 → 동공중심 $(\hat{u}, \hat{v})$.
- 이벤트 표현: 짧은 시간창에 polarity 합산 후 [-1,1] clip → **3치 영상 {-1, 0, +1}**. 눈 영역 crop·resize. flip/shift augmentation.

### Unsupervised pretraining (event-frame reconstruction)
- EETNet conv encoder + 대칭 transposed-conv decoder의 autoencoder로 unlabeled real 이벤트 재구성.
- {-1, 0, +1} 3-class segmentation으로 프레이밍, **Dice + Focal loss**(배경 0 클래스 불균형 대응):
  $$\mathcal{L}_{Dice} = 1 - \frac{1}{C}\sum_{c=0}^{C-1}\frac{2\sum_n p_{n,c}\hat{p}_{n,c}}{\sum_n (p_{n,c} + \hat{p}_{n,c})}, \quad C=3$$
  $$\mathcal{L}_{Focal} = \frac{1}{N}\sum_n [-\alpha_c (1-p_{n,c})^\gamma \log(p_{n,c})]$$
  $$\mathcal{L}_{Total} = \lambda_D \mathcal{L}_{Dice} + \lambda_F \mathcal{L}_{Focal}$$
- 학습된 encoder 가중치로 동공추적 backbone 초기화. (MSE autoencoder보다 sharp.)

### 학습 설정
- PyTorch, RTX A6000. supervised 200ep MSE, crop 157×90 → 78×45 downsample, Adam lr 1e-3, batch 100, Reduce-on-Plateau. autoencoder: $\lambda_D=\lambda_F=1$, $\alpha=[5,1,5]$, $\gamma=2$, 5ms 적분창. fine-tune 100ep cosine.

## 5. 실험
- **메트릭**: MAE(Mean Absolute Error, Manhattan px), MPD(Mean Pupil Distance, Euclidean px).
- **데이터셋**: EV-Eye(48명, DAVIS346, ~9000 타원 라벨), INI-30(30명, DVXplorer, 64×64 변형), EB-NEG/Angelopoulos(24명, DAVIS346b), 3ET+(13명, DVXplorer Mini, 100Hz, 60×80).
- **Domain shift (Table 1)**: 이벤트 sim-only가 image sim-only보다 gap 훨씬 작음. EB-NEG: 이벤트 sim-only MAE 2.45 (real-only 2.60보다 **오히려 우수**, ΔMAE -0.15) vs 이미지 sim-only +7.34. EV-Eye: 이벤트 +3.45 vs 이미지 +53.27.
- **Three-stage (Ours, Table 2 최고치)**: EB-NEG MAE 1.96/MPD 3.08, EV-Eye 2.36/3.78, INI-30 8.37/13.01, 3ET+ 6.76/10.91. real-only 대비 ↑25%, real-pre+fine ↑20%, sim-pretrain+fine ↑10%.
- **SOTA 비교 (Table 3)**: INI-30 64×64에서 Ours MPD 2.48(EETNet 2.77, Retina 3.24, GG-SSM 3.11). 원해상도 예측 후 rescale(Ours*) 시 1.63. 3ET+ 60×80에서 Ours 3.24, Ours* 1.36. **temporal/recurrent 없이 경량 유지하며 비교군 SOTA**.

## 6. 강점/한계
- **강점**: 데이터 부족 문제를 합성+unlabeled로 정면 해결. 이벤트 modality가 sim-to-real에 유리함을 정량 입증. **EETNet은 MAX78000 on-device 가속기 대상 경량 CNN** → 우리 목표와 직접 정합. temporal/attention 없이 경쟁력.
- **한계 (논문 명시)**: blink/강한 eyelid occlusion을 명시 모델링하지 않음(smooth motion 가정) → 빈번한 깜빡임 시 성능 저하. 미관측 인구특성 robustness 미평가. 합성 카메라 pose 수동 정렬 필요(pose 정렬 민감, Table 4).

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **직접 연관**: EETNet은 MAX78000(저전력 CNN 가속기) 대상 - 우리 FPGA on-device 가속 목표와 동일 계열. 경량 3-block CNN + FC 회귀헤드 구조는 FPGA에 매핑 용이(추정).
- **직접 연관**: {-1, 0, +1} 3치 이벤트 프레임 표현은 2-bit 데이터로 FPGA 입력 대역폭/저장 절감에 유리. 정수/비트시프트 친화적(추정).
- **데이터 전략**: 합성+unlabeled 3단계 학습은 우리 시스템의 데이터 확보·도메인적응 문제에 직접 적용 가능(호스트 측 학습). on-device 추론과 무관하게 모델 품질 향상 수단.
- **추정**: temporal/recurrent 없이 SOTA급 정확도를 내는 점은 FPGA 저지연 설계에 유리(순환 의존성 없음 → 파이프라인 stall 없음).
- 양자화는 논문에서 직접 다루지 않으나 EETNet의 MAX78000 타깃 특성상 INT 양자화 친화적(추정, MAX78000은 1/2/4/8-bit 가중치 지원).

## 8. 근거표기
- 1~6장 수치/수식/구조는 본문(ETRA018, ACM 2026) 직접 확인.
- 7장 FPGA 매핑·양자화 친화성은 **추정** (논문은 MAX78000 CNN 가속기 타깃만 언급, FPGA 구현 없음). MAX78000의 INT 지원은 일반 상식 기반 **추정**.
