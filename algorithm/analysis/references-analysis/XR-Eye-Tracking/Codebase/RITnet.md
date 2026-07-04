# RITnet 코드베이스 정밀 분석

> 기준 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\RITnet\`
> 분석 도구: Glob/Grep/Read 만 사용 (bash 미사용). 라인 근거는 `파일명:라인` 형식.

---

## 1. 개요

- **목적**: 시선 추적(gaze tracking)을 위한 **실시간 안구 의미분할(semantic segmentation)**. 안구 영상을 4클래스(배경/공막/홍채/동공)로 분할하는 초경량 네트워크. (`README.md:6`, `densenet.py:83` `out_channels=4`)
- **원논문/챌린지**: Chaudhary et al., "RITnet: Real-time Semantic Segmentation of the Eye for Gaze Tracking", ICCVW 2019 (`README.md:1-15`). **OpenEDS 2019 Semantic Segmentation Challenge** 참가작(잠재 우승 모델 248,900 파라미터, `README.md:32, 52-53`). EllSeg의 직접적 전신(DenseElNet의 원형).
- **입력**: 단일 채널 grayscale 안구 프레임 `640x400` (`README.md:78` 입력 형태 `[-1,32,640,400]`, `train.py:88` `input_size=(1,640,400)`).
- **출력**: 4클래스 세그멘테이션 맵 (배경/공막/홍채/동공). 동공 좌표는 본 repo에서 직접 회귀하지 않음 — **분할만 수행**, 이후 외부 ellipse fitting으로 시선 산출(논문 파이프라인). 이벤트 카메라 아님 — **프레임 기반**.

---

## 2. 디렉토리 구조

### 자체 소스 (분석 대상)
```
RITnet/
├── densenet.py   # DenseNet2D 모델 (DenseUNet-K 단순화판)
├── models.py     # model_dict['densenet'] = DenseNet2D(dropout=True, prob=0.2)
├── dataset.py    # IrisDataset: 데이터로더 + 전처리(gamma/CLAHE) + 증강
├── utils.py      # 손실(CE2d/Focal/Surface/GDice), one_hot2dist, mIoU, Logger
├── train.py      # 학습 루프 + 검증 + 주기적 시각화
├── test.py       # 추론/평가
├── opt.py        # argparse
├── README.md
└── License.md
```
- 제외할 외부 프레임워크/vendor/build 디렉토리 없음(자체 코드만 존재 — 확인: Glob 결과 7개 .py 전부 자체 소스).

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 모델 — DenseNet2D (`densenet.py`)

**출처/성격**: ShusilDangi/DenseUNet-K의 단순화 2D 구현 (`densenet.py:8-11` 주석). DenseNet 연결 + U-Net 인코더-디코더.

**(a) Down block** `DenseNet2D_down_block` (`densenet.py:18-52`)
- 구성: `conv1`(3x3) → concat → `conv21`(1x1)+`conv22`(3x3) → concat → `conv31`(1x1)+`conv32`(3x3) → 마지막 BN (`:21-25, 36-52`).
- **dense 연결**: 각 단계 출력을 입력에 `torch.cat` (`:42, 44`) — feature reuse.
- 다운샘플: `AvgPool2d(kernel_size=down_size)`를 **블록 진입부에서** 적용 (`:26, 37-38`). `down_size=None`이면 첫 블록(다운샘플 없음).
- LeakyReLU 활성, dropout 옵션(`prob=0.2`, `models.py:13`).

**(b) Up block** `DenseNet2D_up_block_concat` (`densenet.py:55-80`)
- 업샘플: `F.interpolate(scale_factor, mode='nearest')` (`:70`) — **nearest** (EllSeg는 bilinear과 대조; HW 친화적).
- skip concat 후 `conv11`(1x1)+`conv12`(3x3) → concat → `conv21`(1x1)+`conv22`(3x3) (`:58-62, 71-79`).

**(c) 전체** `DenseNet2D` (`densenet.py:82-143`)
- 인코더: down_block1~5 (block1은 다운샘플 없음, 2~5는 (2,2) AvgPool) → 해상도 1/16까지 (`:86-95`).
- 디코더: up_block1~4 (각 (2,2) nearest 업샘플) + skip 연결 (`:97-104, 134-137`).
- 출력: `out_conv1`(1x1, channel_size→4) (`:106, 139-141`).
- **모든 단계 channel_size=32 고정** (`:83`) — 채널 폭이 일정해 파라미터가 248,900으로 매우 작음(`README.md:186`).
- forward에서 중간 feature를 `self.x1~x9`로 인스턴스 속성 저장 (`:129-137`) — 디버깅 편의지만 멀티스레드/추론 시 상태 공유 리스크(코드 스멜).
- 가중치 초기화: He-normal(Conv), 1/0(BN) (`:113-126`).

### 3.2 손실 함수 (`utils.py` + `train.py`)

**총손실** (`train.py:137-145`):
```
CE_loss = CrossEntropyLoss2d(output, target)
loss = mean( CE_loss * (1 + spatialWeights) )       # 경계 가중 CE (Canny edge*20)
loss = (1-alpha)*loss_sl + alpha*loss_dice + loss   # surface + dice + 가중CE
```
- **`CrossEntropyLoss2d`** (`utils.py:35-42`): `NLLLoss(log_softmax(·))`. `spatialWeights`로 경계 픽셀 가중(`dataset.py:186-187`, Canny→dilate→*20).
- **`GeneralizedDiceLoss`** (`utils.py:58-97`): 클래스별 역제곱 빈도 가중(`:88`)으로 클래스 불균형(동공이 작음) 보정.
- **`SurfaceLoss`** (`utils.py:44-55`): softmax 출력 × 거리맵(`one_hot2dist`) 평균 — 경계 정밀도 향상(저자 Rakshit Kothari, EllSeg와 동일 계열).
- **`FocalLoss2d`** (`utils.py:25-31`): 정의돼 있으나 train 루프에서는 미사용(확인: `train.py`는 CE/Dice/Surface만 호출).
- **alpha 커리큘럼** (`train.py:123-126`): 0~125 epoch에서 `1→0`으로 선형 감소 후 125 이후 1 고정 → 초기엔 surface loss 비중↑, 후기엔 dice 비중↑. (주석상 "surface↔dice 가중 함수", `:122`)

### 3.3 데이터셋/전처리 (`dataset.py`)

**`IrisDataset`** (`dataset.py:124-203`):
- **전처리(전 split 공통)** (`README.md:62-64`, `dataset.py:138-176`):
  1. **Gamma 보정** 계수 0.8 — LUT 테이블 `255*(linspace^0.8)` (`:153-155`).
  2. **CLAHE** (`clipLimit=1.5, tileGridSize=(8,8)`) (`:140, 175`) — 국소 대비 향상.
  3. **정규화** `Normalize([0.5],[0.5])` (`:36-37`).
- **학습 증강(train만)** (`dataset.py:164-180`):
  1. RandomHorizontalFlip 50% (`:40-45, 180`).
  2. **Starburst 패턴** 20% (`:47-65`) — 안경 다중반사 모사. 고정 패턴 `starburst_black.png`를 랜덤 평행이동 후 곱셈 합성.
  3. Line augment 20% (`:107-117`) — 랜덤 중심 주변 1~9개 흰 선 (반사/속눈썹 모사).
  4. Gaussian blur 20% (kernel 7x7, sigma 2~7) (`:74-77`).
  5. Translation 40% (랜덤 방향, factor<40) — 이미지·라벨 동시 이동 (`:79-105, 172-173`).
- **부가 출력** (`dataset.py:184-203`):
  - `spatialWeights`: Canny→dilate→*20 (경계 가중 CE용, `:186-187`).
  - `distMap`: 4클래스 각각 `one_hot2dist` (surface loss용, `:191-194`).
- 반환: `img, label, fileName, spatialWeights, distMap` (test split은 라벨/맵 0 placeholder, `:198-203`).

**`one_hot2dist`** (`utils.py:100-111`): signed distance transform (외부 `distance_transform_edt`), 최대거리로 정규화. LIVIAETS/surface-loss 차용 (`:99`).

### 3.4 학습/평가 루프 (`train.py`)

- 옵티마이저 Adam(`lr=1e-3`, `opt.py:15`), `ReduceLROnPlateau(mode='min', patience=5)` (`train.py:95`).
- 기본 epoch 250, bs 8 (`opt.py:10-11`).
- `lossandaccuracy` (`train.py:23-51`): 검증 손실·mIoU. `total_metric`(`utils.py:176-179`)으로 파라미터 수 페널티 결합 점수(챌린지 복잡도 메트릭 반영).
- 매 epoch 모델 저장, 5 epoch마다 테스트셋 시각화 저장 (`train.py:166-192`).

---

## 4. 알고리즘/데이터 표현

- **데이터 표현**: 단일 grayscale 프레임 (이벤트 아님). 시공간 모델 없음 — 프레임 독립.
- **분할 후 ellipse fitting**: 본 repo에는 ellipse fitting/동공좌표 회귀 코드 없음(분할까지만). 시선 산출은 논문 외부 파이프라인 또는 후속 EllSeg에서 수행 (확인: `densenet.py` 출력은 4채널 분할맵뿐).
- **핵심 설계**: 극경량(248,900 params) + 경계인식 손실(Canny 가중 CE + surface) + 강한 도메인 증강(starburst/line)으로 **실시간성 + 경계 정밀도** 동시 추구.

---

## 5. 학습/평가 파이프라인

- **데이터셋**: `Semantic_Segmentation_Dataset/` (OpenEDS 2019), split=train/validation/test 하위에 `images/`·`labels/`(.npy) (`dataset.py:127-160`).
- **메트릭** (`utils.py`):
  - `mIoU` (`:113-126`), `compute_mean_iou` (precision/recall/f1 포함, `:140-174`).
  - `total_metric` (`:176-179`): `0.5*(min(1,1/S) + mIoU)`, S=파라미터 크기(MB) — OpenEDS 챌린지 "정확도+모델크기" 통합 점수.
- **실행 명령어** (`README.md:21-27`):
  - 학습: `python train.py --model densenet --expname FINAL --bs 8 --useGPU True --dataset Semantic_Segmentation_Dataset/`
  - 테스트: `python test.py --model densenet --load best_model.pkl --bs 4 --dataset Semantic_Segmentation_Dataset/`
- **보고 성능** (`README.md:52-53`): Epoch 151 Val 95.78% / Test 95.28% (잠재 우승 모델).

---

## 6. 의존성

- PyTorch, torchvision(transforms), numpy, OpenCV(cv2), PIL, scipy(`distance_transform_edt`), scikit-learn(precision/recall/f1), torchsummary(옵션), matplotlib, tqdm (`dataset.py:23-33`, `utils.py:14-23`, `train.py:9-20`).
- 외부 vendor/extern 없음.

---

## 7. 강점 / 한계 / 리스크

**강점**
- **극경량**: 248,900 파라미터(0.95MB FP32, `README.md:186-192`) — 임베디드/FPGA 후보로 매우 우수.
- 채널 폭 32 고정 + 단순 연산(Conv/AvgPool/nearest upsample/LeakyReLU)으로 **HW 매핑 단순**.
- nearest 업샘플(`densenet.py:70`) — 보간 로직 없이 HW 구현 용이.
- 경계인식 복합 손실로 작은 동공/홍채 경계 정밀.

**한계 / 리스크**
- deprecated numpy(`np.bool` `utils.py:106`), CUDA 하드코딩(`utils.py:76,82-83` `.cuda()`) → 버전/디바이스 이식 수정 필요.
- `DenseNet2D`가 중간 feature를 `self.x1~x9` 인스턴스 속성에 저장(`densenet.py:129-137`) — 추론 재진입/멀티스레드 안전성 저하(코드 스멜).
- `GeneralizedDiceLoss.forward`가 numpy 왕복으로 one-hot 생성(`utils.py:75-76`) → GPU↔CPU 전송 오버헤드.
- 학습 루프 `ious` 리스트가 epoch 간 초기화 안 됨(`train.py:127` 루프 밖 선언) → train mIoU 누적 평균이 왜곡(버그 가능, 확인: `:127, 149, 157`).
- 동공 좌표/시선 출력 없음 — 분할 후 fitting을 별도 구현 필요.

---

## 8. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 가속, 추정)

> "XR 시선추적 + FPGA 가속" 프로젝트 맥락은 **추정**.

- **FPGA 1순위 후보 백본**: 248,900 파라미터·0.95MB는 중급 FPGA 온칩 BRAM에 weight 상주 가능 → **외부 DRAM 접근 최소화·저지연**. EllSeg(DenseElNet)보다 더 단순/작아 PoC에 적합.
- **연산 균질성**: 모든 블록 channel_size=32, 동일 conv 패턴(1x1+3x3) 반복 → **시스톨릭 MAC 어레이 재사용/타일링 용이**, 제어 로직 단순.
- **양자화**: AvgPool·LeakyReLU·Conv 위주 → INT8 PTQ 후보. BatchNorm은 conv에 fold 가능. Dice/Surface 손실은 학습 전용이라 추론 HW엔 영향 없음.
- **업샘플 이점**: nearest interpolation(`densenet.py:70`)은 라인버퍼 복제만으로 구현 — bilinear 대비 DSP/로직 절감. EllSeg를 HW화할 때도 nearest로 치환 검토 근거.
- **전처리 가속**: gamma(LUT)·CLAHE는 FPGA 전처리 IP로 흔히 구현됨 → 카메라→전처리→추론을 온칩 파이프라인화 가능.
- **재사용 포인트**: ① 경량 DenseUNet 백본 자체, ② 경계 가중(Canny) 학습 기법(정확도 보존하며 양자화), ③ `total_metric` 식 정확도-크기 trade-off 평가 지표(DSE 목표함수로 차용 가능).
- **한계 인지**: 분할만 제공하므로, 시선/동공좌표까지 가속하려면 EllSeg식 soft-argmax 헤드 또는 경량 ellipse fit을 추가 설계해야 함.

---

## 9. 근거 표기

- **확인(코드 라인 근거)**: 모델 전체(`densenet.py`), 손실 전부(`utils.py:25-111`, `train.py:137-145`), 전처리·증강(`dataset.py`), 학습/검증 루프(`train.py`), 인자(`opt.py`), 파라미터 수·성능(`README.md`).
- **추정**: 입력 해상도 640x400(README summary + train summary 기준), "FPGA 가속" 프로젝트 맥락(8장).
- **확인 불가(미열람)**: `test.py` 전체 로직(추론/저장 세부) — 본 분석 범위 외. starburst_black.png 생성 절차(README 문서 참조).
