# E-Track: Eye Tracking with Event Camera for Extended Reality (XR) Applications

## 1. 서지정보
- **제목**: E-Track: Eye Tracking with Event Camera for Extended Reality (XR) Applications
- **저자**: Nealson Li, Ashwin Bhat, Arijit Raychowdhury (Georgia Institute of Technology, School of ECE)
- **연도**: 2023 (IEEE AICAS 2023; DOI 10.1109/AICAS57966.2023.10168551)
- **학회**: 2023 IEEE 5th International Conference on Artificial Intelligence Circuits and Systems (AICAS)
- **원본파일명**: E-Track_Eye_Tracking_with_Event_Camera_for_Extended_Reality_XR_Applications.pdf

## 2. 문제정의·배경
- XR 헤드셋의 시선추적(foveated rendering, varifocal display, gaze UI)은 엄격한 지연·전력 제약 하에 동작해야 함.
- RGB 카메라는 고정 프레임율로 빠른 동공 움직임 추적 실패 + dense 데이터로 저전력/저지연에 불리. 정지 시 프레임 무의미.
- 이벤트 카메라는 움직임 트리거·고시간해상도·sparse로 적합하나, 프레임 기반 CV 기법 직접 적용 불가.
- 기존 이벤트 시선추적은 RGB+이벤트 동시 사용([10]) 또는 IR LED glint([11]) 등 **추가 하드웨어 필요** → 헤드셋 크기·전력 증가.

## 3. 핵심기여
- **이벤트 카메라만으로 동작하는 최초의 완전 이벤트 기반 시선추적 시스템**(추가 센서 불필요).
- **Event-to-frame 변환**: 이벤트 특징을 3채널 프레임으로 인코딩.
- 24명 데이터로 학습한 **저지연 segmentation CNN(U-Net)**으로 동공 이벤트 식별.
- **이벤트 기반 RoI 메커니즘**: 비동기 동공 추적으로 **CNN 추론 96% 감소**.
- 결과: 동공 위치 오차 **3.68px @ 160mW 시스템 전력**(edge SoC).

## 4. 방법론
### 이벤트 표현 / Event-to-Frame Converter
- 이벤트 $(x, y, t, p)$.
- **고정 시간창 대신 고정 이벤트 수(2000개) set** 누적 → 모든 set이 동일 motion량·명확한 eye feature 표현.
- 3채널 프레임 변환: ch1=positive polarity, ch2=negative polarity, ch3=전체 이벤트. 각 이벤트는 픽셀에 intensity 10 기여(max 255, 정규화 없음). 각 eye 부위(eyelash/eyelid/pupil)가 distinct 시공간 분포.

### Pupil Event U-Net
- U-Net encoder-decoder + skip connection. **17 layer, 466,562 param**. 입력 352×256×3, 출력 2채널(pupil/background) 352×256.
- 인코딩: (conv+ReLU+maxpool)×2 + conv+ReLU. 디코딩: (up-conv+concat+conv+ReLU)×2 + conv+softmax.
- 클래스 불균형(pupil 픽셀 희소) 대응: **weighted loss** (pupil 채널 0.9, background 0.1). 출력 >0.9 픽셀을 pupil로 분류 후 **타원 fitting**.

### Event-based RoI mechanism (핵심: 추론 절감)
- 첫 set에서 U-Net으로 동공 식별 후, 타원을 8px 확장한 toroidal band를 RoI로 마킹.
- 이후 RoI 내 **256 이벤트만** 수집 → 타원 fitting으로 동공 위치 갱신(추론 없이).
- CNN 추론은 (1) 최초 iteration, (2) blink로 동공 소실 시에만 수행.

### 학습
- 24명 70/15/15 split. weighted categorical cross-entropy, Adam, lr 0.001 (exp decay, step 4500, rate 0.5).
- **데이터셋**: [10](Angelopoulos) DAVIS-346(346×260), 24명, 각 330초, blink/saccade/smooth pursuit. 라벨은 gaze point라서 프레임 기반 분할([6] EllSeg)로 동공 타원 생성, 시공간 근접 이벤트를 pupil event GT로 라벨.

## 5. 실험
- **메트릭**: IoU, center distance error(px).
- **정확도**: 전체 median **center distance error 3.68px, median IoU 0.72**(대부분 subject IoU>0.5 목표 상회, median>0.7). 3분위 오차 대부분 <5px.
- **RoI 효과**: 330초 평균 7,130 프레임 / 20,669,767 이벤트. frame 기반 7,130 추론, 2000-event set 기반 10,334 추론 → **RoI로 404 추론(96.1% 감소)**. 99%+ 타원은 fitting만으로 추적. frame 대비 17.65× 적은 추론.
- **HW 성능 (Table I, U-Net 추론 / 시스템 전력)**:
  - CPU(Ryzen9 5900H): 69.5ms, 시스템 2.97W / 982J
  - GPU(RTX 3070): 4.1ms, 시스템 0.62W / 206J
  - **TF-Lite/Coral Dev Board(edgeTPU): 66.4ms, 시스템 0.16W(160mW) / 53.6J** (채택 플랫폼)
  - TF-Lite/Coral USB: 95.2ms, 0.23W / 76.9J
- CPU/GPU는 에너지 과다 → edgeTPU 등 전용 HW 필요성 입증.

## 6. 강점/한계
- **강점 (FPGA/on-device 관점 매우 중요)**: 이벤트 카메라 단독·추가 HW 불필요로 폼팩터/전력 절감. **RoI 메커니즘으로 추론 96% 감소** → 저지연·저전력의 핵심 아이디어. edge SoC(Coral) 실배치로 160mW 달성. XR의 저지연·전력제약을 정면 타겟.
- **한계**: U-Net 466k param·352×256 입력으로 단일 추론(66.4ms@edgeTPU)은 무거움 - RoI로 빈도를 줄여 보완. blink 시 재추론 필요. GT가 프레임 기반 분할에서 파생되어 라벨 noise 가능. 정확도(3.68px)는 중간 수준.

## 7. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)
- **직접 연관 (높은 가치)**: E-Track은 우리 목표("XR 저지연 on-device 시선추적")의 핵심 사례 - 저자가 칩 설계 그룹(Raychowdhury, Georgia Tech)으로 HW 가속 지향. edge accelerator 배치 + 전력/지연 실측 제공 → 우리 FPGA 비교 baseline으로 직접 활용 가능(추정).
- **직접 차용 가능**: **RoI 메커니즘**(추론을 96% 줄이고 대부분 타원 fitting으로 처리)은 FPGA에서 CNN 가속기 가동률을 낮춰 전력 절감하는 강력한 패턴. 우리 시스템에 이식 가치 높음(추정).
- **추정**: 고정 이벤트 수(2000/256) 누적 방식은 FPGA 입력 버퍼링·트리거 로직을 단순화. 3채널 프레임은 정수 카운트 기반으로 양자화 친화적.
- **추정**: U-Net을 FPGA에 올릴 경우 466k param·352×256은 BRAM/대역폭 부담 → 양자화(INT8)+pruning 또는 더 경량 backbone 필요. RoI로 추론 빈도를 낮추므로 처리량 요구는 완화됨.
- 타원 fitting(least-squares ellipse)은 FPGA에서 고정소수점 행렬연산으로 구현 가능하나 비용 검토 필요(추정).

## 8. 근거표기
- 1~6장 수치/수식/구조/전력·지연 실측은 본문(AICAS 2023) 직접 확인.
- 7장 FPGA 매핑·양자화·RoI 이식은 **추정** (논문은 edgeTPU/CPU/GPU 실험만 보고, FPGA 구현 없음). 저자 그룹의 HW 가속 지향성은 소속/주제 기반 판단.
