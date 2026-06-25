# Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras

## 1. 서지정보
- **제목**: Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras
- **저자**: Tongyu Zhang, Yiran Shen*, Guangrong Zhao, Lin Wang, Xiaoming Chen, Lu Bai*, Yuanfeng Zhou (*교신저자; Shandong Univ., HKUST, Beijing Tech & Business Univ.)
- **연도**: 2024 (IEEE TVCG Vol.30 No.5, May 2024; DOI 10.1109/TVCG.2024.3372039)
- **학회/저널**: IEEE Transactions on Visualization and Computer Graphics (TVCG)
- **원본파일명**: Swift-Eye_Towards_Anti-blink_Pupil_Tracking_for_Precise_and_Robust_High-Frequency_Near-Eye_Movement_Analysis_with_Event_Cameras.pdf

## 2. 문제정의·배경
- VR/AR 시선추적은 mental health 진단, foveated rendering 등에 필수이나, CCD/CMOS 카메라 기반은 30~90Hz로 시간해상도가 낮아 고속 안구 움직임(saccade 최대 700°/s, 24,000°/s²) 분석 불가.
- 이벤트 카메라는 μs 시간해상도로 고주파(수십 kHz) 추적 가능하나, 기존 방법(EBVEYE, EV-Eye)은 noisy/비구조적 이벤트 스트림을 직접 처리 → 시각화 품질 낮고, **눈 깜빡임(blink)에 의한 동공 occlusion**에 취약.
- 핵심 목표: **부분 occlusion 상황에서도 정밀·강건한 동공 추정 + 초고주파 시각화** (단, offline 분석 지향).

## 3. 핵심기여
- **Swift-Eye 프레임워크**: 이벤트 카메라의 고시간해상도를 활용해 부분 occlusion 동공 추정/추적.
- **Temporal feature fusion 컴포넌트**: 이전 프레임(완전 개안)의 adaptive template로 현재 occluded 프레임의 동공을 예측 → 연속적 고시간해상도 추적.
- **Mask in-painting 기반 데이터 합성(LAMA)**: 완전 개안 이미지에서 부분 occlusion 합성 → 라벨링 노동 절감 + 강건성 향상(real fine-tuning 없이 일반화).
- EV-Eye 데이터셋 평가: occlusion >80% 시 2위 대비 **IoU +20%, F1 +12.5%**.

## 4. 방법론
### 시스템 구성 (3개 모듈)
1. **Event-based frame interpolation (Timelens)**: 25fps + 이벤트 스트림 → 5,000fps 고속 영상. self-supervised fine-tuning(5연속 프레임 중 양끝 입력, 중간 3프레임 GT). PSNR 37.46→41.16dB(2.34× 개선).
2. **Multi-scale spatial feature extraction & fusion**: Swin-Transformer(shifted window self-attention) backbone + FPN(top-down, 1×1 conv lateral, 2× upsample) 융합.
3. **Anti-blink pupil estimation & tracking**: temporal feature fusion + rotated pupil detector + occlusion-ratio adaptation + trace completion.

### Temporal feature fusion
- 이전 완전개안 프레임의 동공 중심 52×52px 영역 → 최저레벨 feature pyramid의 13×13 영역을 template로 사용.
- template와 현재 프레임 feature를 **depth-wise cross-correlation**(SiamRPN++ 방식)으로 융합.

### Rotated bounding box 동공 추정
- 동공 타원 5파라미터 $\{P_x, P_y, P_w, P_h, P_a\}$ (중심, 장/단축, 회전).
- anchor box(~10,000개) + class subnet(신뢰도) + box subnet(offset $\{d_x,d_y,d_w,d_h,d_a\}$). 디코딩:
  $$P_w = \max\{B_w e^{d_w}, B_h e^{d_h}\}, \quad P_h = \min\{B_w e^{d_w}, B_h e^{d_h}\}$$
  $$P_x = d_x B_w \cos(B_a) - d_y B_h \sin(B_a) + B_x$$
  $$P_y = d_x B_w \sin(B_a) + d_y B_h \cos(B_a) + B_y$$

### Occlusion-ratio 기반 approach adaptation
- U-Net 분할로 노출된 동공 픽셀 $P_s$, 완전개안 동공 면적 $P_l$ → $\alpha = \frac{P_l - P_s}{P_l}$.
- $\alpha < 0.25$: temporal fusion 미사용 / $0.25 \leq \alpha \leq 0.875$: full Swift-Eye / $\alpha > 0.875$: INVALID(스킵).
- (직전 프레임 기반으로 $\alpha$ 추정, 인접 프레임 차이 최소 가정.)

### Trace completion
- INVALID 프레임은 전후 valid 추정으로 선형 보간: $\Theta_i = \frac{t_k - t_i}{t_k - t_j}\Theta_j + \frac{t_i - t_j}{t_k - t_j}\Theta_k$.

### 학습
- RTX 3090, PyTorch 1.13, Adam, lr 1e-4, batch 8. EV-Eye 9,011 개안 이미지 + LAMA 합성 부분occlusion으로 학습, 직접 라벨링한 2,400 real occlusion 이미지로 테스트(학습 미사용→일반화 검증).

## 5. 실험
- **데이터셋**: EV-Eye (DAVIS346 2대, 25fps + 이벤트, 48명, 150만+ grayscale, 27억+ 이벤트). 추가로 occlusion 비율별 real 이미지 2,400장 직접 라벨링.
- **메트릭**: IoU $= |P\cap G|/|P\cup G|$, F1(Dice) $= 2|P\cap G|/(|P|+|G|)$.
- **주요수치**: occlusion >80%에서 2위(EllSeg) 대비 IoU +20% (0.58→0.702), F1 +12.5% (0.726→0.817).
- **baseline**: EV-Eye, DeepVOG, EllSeg (모두 합성 학습, real occlusion 테스트). EllSeg 2위(occlusion 고려), DeepVOG/EV-Eye는 U-Net 유사 backbone으로 비슷.
- **Ablation**: temporal fusion 임계 0.25/0.875 결정. temporal fusion + 데이터합성 둘 다 강건성에 크게 기여.
- 고시간해상도(5,000fps) trace가 25fps 대비 fast movement 국소 디테일 포착 우수.

## 6. 강점/한계
- **강점**: blink occlusion에 특화된 anti-blink 설계. rotated box로 타원 prior 활용(분할 방식보다 강건). LAMA 합성으로 라벨링 비용 절감 + real 일반화. 초고주파 시각화로 mental health/behavior-brain 분석 지원.
- **한계 (FPGA/on-device 관점에서 중요)**: 논문 명시 - **현재 work는 tracking latency를 고려하지 않음. offline 분석용**(mental health, behavior-brain). foveated rendering 같은 저지연 실시간엔 부적합. Swin-Transformer+FPN+Timelens+다중모듈로 무겁고 연산량 큼. anchor box 10,000개로 detection 비용 큼.

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **주의 (논문 명시)**: Swift-Eye는 offline·고정밀 지향이며 **저지연 실시간을 명시적으로 다루지 않음** → 우리 프로젝트(저지연 on-device)와 목표가 상충. 직접 채택보다는 **개념 차용** 대상.
- **추정**: rotated ellipse box 디코딩 수식(삼각함수 포함)은 FPGA에서 CORDIC 등으로 구현 가능하나 비용 고려 필요. anchor 10,000개는 HW에 부담 → anchor-free 경량화 필요.
- **추정**: anti-blink/occlusion 강건성 아이디어(adaptive template, occlusion-ratio gating)는 우리 시스템에 가치 있으나, Swin-Transformer backbone은 FPGA 양자화 대상으로 무거움. 우리 Q-HyViT 등 ViT 양자화와 결합 시 backbone 경량화가 관건(추정).
- **추정**: LAMA 합성 데이터 전략은 on-device 학습이 아닌 호스트 측 데이터 증강으로 활용 가능.

## 8. 근거표기
- 1~6장 수치/수식/구조는 본문(TVCG 2024) 직접 확인. "offline 지향, latency 미고려"는 논문 Conclusion **직접 인용**.
- 7장 FPGA/양자화 시사점은 **추정** (논문은 GPU offline 실험만 보고).
