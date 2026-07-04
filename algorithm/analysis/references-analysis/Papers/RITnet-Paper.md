# RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking

## 1. 서지정보
- **제목**: RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking
- **저자**: Aayush K. Chaudhary*, Rakshit Kothari*, Manoj Acharya*, Shusil Dangi, Nitinraj Nair, Reynold Bailey, Christopher Kanan, Gabriel Diaz, Jeff B. Pelz (Rochester Institute of Technology)
- **연도**: 2019 (arXiv:1910.00694v1, 2019-10-01)
- **학회/저널**: 2019 OpenEDS Semantic Segmentation Challenge (ICCV 2019 Workshop 관련, OpenEDS challenge 우승작 — 추정)
- **코드**: bitbucket.org/eye-ush/ritnet
- **원본파일명**: RITnet Real-time Semantic Segmentation of the Eye for Gaze Tracking.pdf

## 2. 문제정의·배경
- 정확한 안구 분할(pupil/iris/sclera/skin)은 시선 추정 파이프라인의 핵심 단계이나, 기존 방법은 person-dependent·강건성 부족·실시간 불가.
- HMD용 시선추적 연구 활성화를 위해 Facebook Reality Labs가 OpenEDS Semantic Segmentation Challenge 개최. **프레임(RGB/grayscale) 기반** 안구 영역 분할 문제. (이벤트 카메라 논문이 아님 — 프레임 기반 분할 baseline 성격)

## 3. 핵심 기여
- **RITnet**: U-Net + DenseNet 결합 분할 아키텍처. **모델 크기 0.98MB**(<1MB), OpenEDS 95.3% mIoU로 SOTA, **301Hz (640×400, 1080Ti)** 실시간.
- **도메인 특화 증강**: starburst(안경 IR 반사), thin lines 등으로 까다로운 조건 일반화.
- **경계 인지 손실(boundary-aware loss) + loss scheduling**: 명확한 영역 경계 생성.

## 4. 방법론
### 모델 구조
- **DenseUNet-K 기반** encoder-decoder. 5개 Down-Block + 4개 Up-Block.
- 마지막 Down-Block = bottleneck(입력 해상도 1/16). 각 Down-Block = 5개 conv(LeakyReLU), DenseNet식 이전 층 연결 공유, 채널 수 **K=32 상수 유지**(파라미터 절감), 2×2 average pooling.
- Up-Block = 4개 conv(LeakyReLU), **nearest-neighbor 2배 업샘플**(bilinear보다 sclera 분할 우수), 대응 Down-Block의 skip connection.
- 총 **248,900 trainable parameters** (32-bit 기준 <1MB). 4클래스 출력(background/iris/sclera/pupil).

### 손실함수 (핵심)
- 클래스 불균형(pupil 픽셀 최소) 대응 위해 4개 손실 가중 결합:
$$\mathcal{L} = \mathcal{L}_{CEL}(\lambda_1 + \lambda_2\mathcal{L}_{BAL}) + \lambda_3\mathcal{L}_{GDL} + \lambda_4\mathcal{L}_{SL}$$
  - **CEL**(cross-entropy), **GDL**(Generalized Dice Loss, 클래스 빈도 제곱 역수 가중), **BAL**(Boundary Aware Loss, Canny edge 2px dilation으로 경계 픽셀 마스킹), **SL**(Surface Loss, 경계 거리 기반, 작은 영역 복원).
- **Loss scheduling**: $\lambda_1=1, \lambda_2=20, \lambda_3=(1-\alpha), \lambda_4=\alpha$, $\alpha=\text{epoch}/125$ (epoch<125, 이후 0). 초기엔 GDL 위주, 안정 후 SL이 stray patch 패널티.

### 전처리·증강
- **전처리**: gamma correction(exp 0.8) + CLAHE(8×8 grid, clip 1.5) → iris/pupil 구분 용이, train/val/test 밝기 분포 차이 완화.
- **증강**(반복마다 확률 0.2, 수평flip 0.5): 수직축 반사, Gaussian blur(7×7, σ 2~7), translation(0~20px), thin line(2~9개), **starburst 패턴**(안경 IR 반사 시뮬레이션, 0~40px translation).

### 학습
- Adam, lr 0.001, batch 8, 175 epoch, TITAN 1080Ti. val loss plateau(5 epoch) 시 lr ×0.1. best=151 epoch.

## 5. 실험
- **데이터셋**: OpenEDS (12,759 이미지: train 8,916/val 2,403/test 1,440, 4 라벨 수동 주석).
- **메트릭**: mIoU, mean F1, Model Size(S, MB), Overall Score = (mIoU + min(1/S, 1))/2.
- **주요 결과 (Table 1, test)**:
  - **RITnet(Ours): mean F1 99.3, mIoU 95.3, Size 0.98MB, 0.25M params, Overall Score 0.976** (전 항목 최고).
  - mSegNet w/SC(B): mIoU 89.5, 1.6MB, 0.4M, Score 0.762.
  - mSegNet w/BR: mIoU 91.4, 13.3MB, 3.5M, Score 0.495.
  - baseline 대비 mIoU ~6% 향상, 복잡도 ~38% 감소.
- **한계 관찰**: 심한 motion blur·defocus 시 분할 품질 저하(Fig.5). BAL 가중 ↑ 시 mIoU 94.8→95.3%.

## 6. 강점/한계
- **강점**: 초경량(0.98MB, 0.25M params)·실시간(301Hz); 다중 손실+scheduling으로 클래스 불균형·경계 처리; 도메인 특화 증강(starburst)으로 안경/마스카라/저조도 강건; OpenEDS SOTA.
- **한계**: **프레임 기반**(이벤트 카메라 아님 → near-eye 고속 운동 시 motion blur 취약, 실제 motion blur/defocus에서 성능 저하 명시); 분할 후 별도 타원 피팅/시선 회귀 필요(end-to-end gaze 아님); 양자화/FPGA 구현 없음.

## 7. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)
- **0.25M 파라미터·<1MB 모델**은 FPGA on-chip BRAM에 가중치 전체 적재 가능한 수준 → 외부 메모리 접근 최소화로 저지연·저전력에 매우 유리(추정).
- **K=32 상수 채널 + DenseNet skip**: 채널 폭이 일정해 FPGA conv PE 어레이 설계가 규칙적(타일링 단순) → 양자화(INT8) 시 가속기 매핑 용이(추정).
- nearest-neighbor 업샘플은 FPGA에서 단순 픽셀 복제로 구현(bilinear 곱셈 불필요) → HW 비용 절감(채택 적합, 추정).
- **다만 프레임 기반 분할**이므로 우리 이벤트 카메라 파이프라인과 직접 호환은 아님. RITnet은 (a) 분할 백본 설계·손실/증강 기법 참고, (b) FACET처럼 이벤트→분할/타원으로 라벨 생성하는 **반지도 라벨링 도구**로 활용 가능(추정, FACET이 실제로 U-Net으로 EV-Eye 라벨 확장한 전례).
- CLAHE/gamma 전처리는 FPGA 이미지 전처리 IP(히스토그램 LUT)로 구현 가능하나 이벤트 경로엔 불필요.
- 클래스 불균형 손실/scheduling은 학습 단계 기법 → HW 무관, 분할 기반 접근 채택 시 재사용 가능.

## 8. 근거표기
- 4-6섹션 수식·구조·수치(0.98MB/0.25M/95.3 mIoU/301Hz/Table 1)는 PDF 본문(pp.1-5) 직접 근거.
- 7섹션 FPGA/양자화 매핑은 분석자 해석 "추정"(논문엔 FPGA/양자화 직접 기술 없음 → "확인 불가").
- 본 논문은 프레임 기반이므로 이벤트 카테고리 내 baseline/참고용으로 분류함(분석자 판단).
