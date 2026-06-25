# EllSeg 코드베이스 정밀 분석

> 기준 경로: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EllSeg\`
> 분석 도구: Glob/Grep/Read 만 사용 (bash 미사용). 라인 근거는 `파일명:라인` 형식.

---

## 1. 개요

- **목적**: 프레임 기반(grayscale 근적외선 안구 영상) 입력에서 동공(pupil)·홍채(iris)를 "타원 구조(elliptical structure) 통째로" 세그멘테이션하여, 가림(occlusion)에 강건한 시선/동공 추적을 수행하는 프레임워크. (`README.md:10-11`)
- **원논문/근거**: Kothari et al., "EllSeg: An Ellipse Segmentation Framework for Robust Gaze Tracking", arXiv:2007.09600 (`README.md:121-126`). 보조 데이터셋: RIT-Eyes (`README.md:130-136`).
- **입력**: 단일 채널(grayscale) 안구 프레임. 추론 시 `(240, 320)`(HxW)로 리사이즈 (`evaluate_ellseg.py:241`, `preprocess_frame`의 `op_shape=(240,320)`). 이벤트 카메라 입력 아님 — **프레임 기반**.
- **출력**:
  1. 3-클래스 세그멘테이션 맵 (배경/홍채/동공; `dec`의 `out_c=3`, `RITnet_v3.py:177`)
  2. 타원 파라미터 회귀값 10개 = 동공 5 + 홍채 5 `[cx, cy, a, b, θ]` (`utils.py:643-667`, `regressionModule`)
  3. 동공/홍채 중심 좌표 (세그멘테이션 center-of-mass 기반, `loss.py:16-37`)
  4. 후처리 타원 fitting 결과 (ElliFit + RANSAC, `evaluate_ellseg.py:134-166`)
- **핵심 아이디어**: 일반적인 "eye-parts 세그멘테이션 → 경계 edge → ellipse fit" 파이프라인은 눈꺼풀·속눈썹 가림에 취약하므로, CNN이 **가려진 부분까지 포함한 전체 타원 영역**을 직접 분할하도록 학습 (`README.md:11`).

---

## 2. 디렉토리 구조

### 자체 소스 (분석 대상)
```
EllSeg/
├── train.py                  # 학습 루프 (disentanglement 포함)
├── test.py                   # 체크포인트 평가
├── evaluate_ellseg.py        # 사용자 비디오 추론 + 타원 fitting (RANSAC/ElliFit)
├── loss.py                   # seg/center/ellipse/self-consistency/confusion loss
├── utils.py                  # regressionModule, convBlock, linStack, 메트릭, soft-argmax
├── helperfunctions.py        # my_ellipse, ElliFit, ransac, getValidPoints, 전처리
├── args.py                   # 학습 인자 파서
├── modelSummary.py           # model_dict (ritnet_v1~v7, deepvog)
├── CurriculumLib.py          # DataLoader_riteyes (H5 기반 데이터셋 로더)
├── data_augment.py           # 증강 (augment)
├── pytorchtools.py           # EarlyStopping, load_from_file
├── models/
│   ├── RITnet_v1.py ~ v7.py  # DenseElNet 변형 (v3가 기본/배포 모델)
│   └── deepvog_pytorch.py    # DeepVOG PyTorch 포팅
├── curObjects/               # 데이터로더 생성 스크립트 + 데이터셋 split 정의
│   ├── datasetSelections.py
│   ├── createDataloaders_*.py (baseline/leaveoneout/allvsone/random/pretrained)
├── dataset_generation/       # 각 공개 데이터셋 → 공통 H5 포맷 변환기 (Extract*.py)
└── analysis/                 # 실험·시각화 보조 스크립트
```

### 제외 항목 (분석 제외, 지침에 따름)
- `extern/locating-objects-without-bboxes/` — 외부 프레임워크 원본(Weighted Hausdorff Distance 구현체). EllSeg 자체 `loss.py:211-332`의 `WeightedHausdorffDistance`는 이 외부 코드에서 차용·복제된 것으로, 본 보고서는 자체 `loss.py` 내 사용 부분만 다룸.
- `figures/`, `maintainance/old_code/`, `Sandbox.py` — 산출물/구버전/실험 잔재.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 모델 아키텍처 — DenseElNet (`models/RITnet_v3.py`)

EllSeg의 기본/배포 모델은 `ritnet_v3` (`evaluate_ellseg.py:281`, `model = model_dict['ritnet_v3']`). 인코더-디코더(U-Net 유사) + 타원 회귀 헤드 구조.

**(a) 채널 크기 계산** — `getSizes()` (`RITnet_v3.py:14-28`)
- `growth=1.2`, `chz=32`, `blks=4` 기반으로 enc/dec 각 단계의 입·출력 채널 수를 산출. 디코더 skip 크기는 인코더 입력+intermediate를 역순으로 합산 (`:25`).

**(b) Backbone — Dense 인코더** `DenseNet_encoder` (`RITnet_v3.py:85-134`)
- `head`: `convBlock`(3x3 conv 2개 + BN) — 입력 1채널 → `chz` (`:93-96`).
- `down_block1~4`: `DenseNet2D_down_block` (`:43-62`). 각 블록은 dense 연결(`torch.cat`으로 입력·중간특징 반복 concat, `:57-61`) + `Transition_down`(1x1 conv + AvgPool 다운샘플, `:30-41`).
- `bottleneck`: down_size=0 (다운샘플 없음, `:121-126`).
- forward는 4개 skip + bottleneck 출력 반환 (`:127-134`).
- **주의**: 다운샘플이 `MaxPool`이 아니라 `AvgPool2d` (`:34`) — FPGA/양자화 시 average pooling이 더 단순.

**(c) Neck/디코더** `DenseNet_decoder` (`RITnet_v3.py:136-157`)
- `up_block4~1`: `DenseNet2D_up_block` (`:64-83`). 업샘플은 `F.interpolate(mode='bilinear')` (`:75-78`) 후 skip concat + dense conv. 최종 `convBlock`으로 `out_c=3` 출력 (`:149`).

**(d) 타원 회귀 헤드** `regressionModule` (`utils.py:616-667`)
- bottleneck 특징(`enc.op[-1]` 채널) → conv 3단(`c1`(2x3)·`c2`·`c3`) + AvgPool → flatten → FC 2단(`l1`:`32*3*5→256`, `l2`:`256→10`) (`utils.py:622-651`).
- 출력 10개를 분리: 중심은 `tanh`(범위 [-1,1]), 축 길이는 `sigmoid`(양수·[0,1]), 각도는 raw (`utils.py:653-666`). 동공/홍채 각각 `[cx,cy,a,b,θ]`.

**(e) 최상위 모델** `DenseNet2D` (`RITnet_v3.py:159-266`)
- 구성: `enc` + `dec`(out_c=3) + `elReg`(regressionModule) (`:176-178`).
- forward (`:194-251`): 인코딩 → `latent`(bottleneck GAP, `:207`) → `elReg`로 타원 회귀 → 디코딩 → `get_allLoss`로 손실 계산.
- **EllSeg 핵심 결합 전략** (`:226-227`): 최종 예측 타원 `elPred`는 **중심은 세그멘테이션 center-of-mass(`pred_c_seg`)에서, 축·각도는 회귀(`elOut`)에서** 가져와 결합. 즉 "분할이 잘 되는 중심 + 회귀가 안정적인 형태"의 하이브리드.
- 옵션: `selfCorr`(자기일관성 손실), `disentangle`(데이터셋 편향 제거 — `dsIdentify_lin` MLP로 confusion loss) (`:235-249`, `setDatasetInfo`:`183-192`).
- 가중치 초기화: Conv는 He-normal, BN은 1/0 (`:253-266`).

> v1~v7는 동일 골격의 변형으로 추정(확인: v3만 정밀 분석, 나머지는 `modelSummary.py:19-25`에서 등록만 확인). DeepVOG는 별도 비교 baseline (`deepvog_pytorch.py`, `modelSummary.py:26`).

### 3.2 손실 함수 (`loss.py` + `RITnet_v3.py:268-324`)

**총손실 구성** `get_allLoss` (`RITnet_v3.py:268-324`):
```
total_loss = l_seg2pt + 20*l_seg + 10*(l_pt + l_ellipse)         (:322)
loss += loss_seg2el (동공/홍채 분할-타원 일치)                    (:230-232)
loss += 10*get_selfConsistency (selfCorr 활성 시)                (:237)
```

구성 요소별:
- **`get_seg2ptLoss`** (`loss.py:16-37`): 단일 채널 분할맵에 `softmax(·*temperature)` 후 정규화 좌표 그리드와 가중합 → **미분가능 center-of-mass(soft-argmax)**로 중심 좌표 회귀, GT와 L1. temperature=4 (`RITnet_v3.py:284,293`).
- **`get_segLoss`** (`loss.py:39-60`): 샘플별로 마스크 존재 시 ① `SurfaceLoss`(거리맵 기반 경계 손실, `:77-83`) ② `wCE`(공간가중 cross-entropy, Canny edge 가중, `:114-127`) ③ `GDiceLoss`(generalized Dice, 클래스 불균형 보정, `:85-112`)를 `alpha*SL + (1-alpha)*GD + CE`로 합성. **alpha는 epoch에 따라 0→1 커리큘럼** (`train.py:179`, `linVal`).
- **`get_ptLoss`** (`loss.py:62-75`): 유효 샘플에 한해 L1 (정규화 좌표 가정).
- **`get_seg2elLoss`** (`loss.py:149-175`): 회귀 타원을 `soft_heaviside`(미분가능 계단함수, `utils.py:518-540`)로 내부/외부 마스크화 → 분할 결과와 binary CE. "회귀 타원이 분할과 겹칠수록 손실↓".
- **`get_selfConsistency`** (`loss.py:177-196`): logSoftmax 분할 채널과 타원 마스크 간 KL divergence.
- **`conf_Loss`** (`loss.py:129-147`): disentanglement용. flag=1이면 균등분포로 confusion(L1), flag=0이면 dataset 분류 CE. 논문 "Turning a Blind Eye" 전략 차용.
- **`WeightedHausdorffDistance`** (`loss.py:211-332`): generalized-mean 기반 WHD. 외부 코드 차용(extern). EllSeg 메인 손실에는 직접 호출 안 됨(보조).

### 3.3 데이터셋/전처리 (`CurriculumLib.py`, `data_augment.py`)

- **`DataLoader_riteyes`** (`CurriculumLib.py:34-120+`): H5 아카이브 기반. fold/condition(train/valid/test)별 imList, dataset ID 부여(`extract_datasets`, `:53`), `pad2Size`로 크기 정규화(`:106`), 증강(`augment`, `:114`).
- 반환 텐서: `img, label, spatialWeights, distMap, pupil_center, iris_center, elNorm, cond, imInfo` (`train.py:182`). `cond`는 4-flag: [pupil center 존재, mask 존재, pupil ellipse, iris ellipse] (`CurriculumLib.py:96-103`) — 데이터셋마다 라벨 가용성이 달라 손실 마스킹에 사용.
- 추론 전처리 `preprocess_frame` (`evaluate_ellseg.py:60-95`): width 정렬 리사이즈(LANCZOS4) → 세로 pad/crop → **per-image 표준화** `(img-mean)/std` (`:93`). RITnet의 gamma+CLAHE와 달리 EllSeg 추론은 단순 표준화만 사용(확인: `:93`).

### 3.4 학습 루프 (`train.py`)

- 옵티마이저 Adam(`lr=5e-4` 기본, `args.py:32`), `ReduceLROnPlateau`(mode='max', factor=0.1, `train.py:134-138`), `EarlyStopping`(`:140-145`).
- alpha 커리큘럼: `alpha = linVal(epoch, (0,epochs), (0,1), 0)` (`:179`) — surface loss↔Dice loss 비중을 epoch 따라 전환.
- **disentanglement 절차** (`:188-222`): dsIdentify 분기만 학습(secondary)하는 내부 while 루프 → 다시 본체 학습(primary+confusion). gradient reversal 대안.
- 조기종료 메트릭 `stopMetric` (`:390-393`): mIoU + 동공/홍채 중심오차(역가중) + 각도오차 결합 점수.

### 3.5 추론 + 후처리 (`evaluate_ellseg.py`)

- `evaluate_ellseg_on_image` (`:98-172`): enc→latent→elReg→dec를 `no_grad`로 실행. `--ellseg_ellipses` 플래그로 3-모드:
  1. **network 타원**(`=1`): 중심은 분할 soft-argmax, 축·각도는 회귀 → `my_ellipse().transform(H)`로 픽셀좌표 변환 (`:116-132`).
  2. **output 타원**(`=0`): 분할맵에서 유효 경계점 추출(`getValidPoints`) 후 **ElliFit + RANSAC**으로 타원 적합 (`:134-166`).
  3. **타원 없음**(`=-1`, 기본): 분할맵만 (`:168-170`).
- 비디오 단위 처리 `evaluate_ellseg_per_video` (`:197-272`): OpenCV로 프레임 읽기→grayscale→어두운 프레임 skip(`:235`)→전처리→추론→원해상도 복원(`rescale_to_original`, `:175-194`)→오버레이 비디오 + `_pred.npy` 저장.

---

## 4. 알고리즘/데이터 표현

- **데이터 표현**: 단일 grayscale 프레임 (이벤트/voxel 아님). 시공간 모델 없음 — 프레임 독립 처리(ConvLSTM/SSM/Transformer 미사용).
- **타원 표현**: 파라미터형 `[cx, cy, a, b, θ]` ↔ conic matrix ↔ quadratic form 상호변환을 `my_ellipse`가 제공 (`helperfunctions.py:13-129`). `param2mat`/`mat2param`로 affine 변환(H) 하에서 타원 보존.
- **미분가능 후처리**: soft-argmax(center-of-mass, `loss.py:21-34`), soft-heaviside(`utils.py:518-540`)로 타원↔분할을 end-to-end 학습 가능하게 연결 — 핵심 설계.
- **고전 후처리(추론용)**:
  - **ElliFit** (`helperfunctions.py:209-276`): 최소제곱 conic 적합 (Phi 계수 → 파라미터). 6*2개 미만 점이면 무효(`-1` 반환).
  - **RANSAC** (`helperfunctions.py:278-310`): n_min=15, mxIter=40, Thres=5e-3, n_good=15로 이상치 제거 후 best model (`evaluate_ellseg.py:147,157`).

---

## 5. 학습/평가 파이프라인

- **데이터셋**: OpenEDS, NVGaze, RITEyes, LPW, Fuhl(ElSe+ExCuSe), PupilNet (`README.md:18-23, 62-67`). 각 데이터셋을 `dataset_generation/Extract*.py`로 공통 H5 포맷 변환 후 사용.
- **메트릭** (`utils.py`):
  - 세그멘테이션 IoU/mIoU: `getSeg_metrics` (`:119-149`, sklearn jaccard, nanmean으로 미존재 클래스 무시).
  - 중심 거리 오차(pixel): `getPoint_metric` (`:151-161`, euclidean) — 논문의 p-error(detection rate within N-pixel)에 대응.
  - 각도 오차: `getAng_metric` (`:163-169`), 스케일 비율 (`train.py:273-280`).
- **실행 명령어** (`README.md`):
  - 학습: `./runLocal.sh` (curObjects/baseline의 train/test object 사용).
  - 테스트: `python test.py --expname=OpenEDS --model=ritnet_v3 --path2data=... --curObj=OpenEDS --loadfile=.../checkpoint.pt --disp=0` (`README.md:110`).
  - 자체 비디오 추론: `python evaluate_ellseg --path2data=${PATH_EYE_VIDEOS}` (`README.md:29`). 타원 모드/RANSAC off 플래그 지원(`:38-40`).

---

## 6. 의존성

- PyTorch, torchvision (`make_grid`), numpy, OpenCV(cv2), scikit-image(`draw`), scikit-learn(metrics, model_selection), scipy, h5py, tensorboardX, tqdm, matplotlib (`evaluate_ellseg.py:1-26`, `CurriculumLib.py:9-26`, `utils.py:11-23`).
- 외부 차용: WHD 손실은 `extern/locating-objects-without-bboxes` 기반.

---

## 7. 강점 / 한계 / 리스크

**강점**
- 가림에 강건: 전체 타원 분할 + 회귀/분할 하이브리드 중심으로 occlusion 내성 (`README.md:11`, `RITnet_v3.py:226-227`).
- end-to-end 미분가능: soft-argmax·soft-heaviside로 타원-분할 일관성을 손실에 직접 반영.
- 멀티 데이터셋 학습 + disentanglement으로 도메인 일반화 지향.
- 모델 자체가 경량 계열(DenseElNet, growth=1.2, chz=32). pooling이 AvgPool로 단순.

**한계 / 리스크**
- 코드 안정성 이슈: `np.int`(`RITnet_v3.py:21-22`), `np.bool`(`utils.py:129`) 등 **deprecated numpy API** → 최신 numpy에서 에러. 실행 시 버전 고정 필요(확인: 라인 근거 존재).
- CUDA 하드코딩 다수: `.cuda()` 직접 호출(`loss.py:29,93,143,160` 등), `train.py:37` `torch.device("cuda")` 고정 → CPU/타 디바이스 이식 시 수정 필요. (`evaluate_ellseg.py`만 `--eval_on_cpu` 지원)
- 손실 가중치(`20*l_seg`, `10*(...)`)가 하드코딩(`RITnet_v3.py:322`) — 튜닝 어려움.
- 추론 후처리 RANSAC+ElliFit은 Python 루프(`helperfunctions.py:289-310`) — 실시간/온디바이스에서 병목 가능.
- `disentangle` 학습 루프의 while(`train.py:198-217`)는 수렴 비결정성 위험.

---

## 8. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 가속, 추정)

> 본 프로젝트가 "XR 시선추적 + FPGA 가속(on-device/저지연/경량화)"이라는 가정은 **추정**임.

- **백본 이식성**: DenseElNet은 conv + AvgPool + bilinear upsample + dense concat 조합으로, MaxPool/복잡 attention이 없어 **HLS/RTL 매핑이 비교적 단순**. growth=1.2·chz=32로 파라미터가 작아 FPGA 온칩 메모리에 유리.
- **양자화 친화 요소**: AvgPool(`RITnet_v3.py:34`), LeakyReLU, conv 위주 → INT8 PTQ/QAT 적용 후보. 단, InstanceNorm(`RITnet_v3.py:165`)·BatchNorm 혼용은 HW 매핑 시 fold 전략 필요.
- **후처리 분리 전략**: soft-argmax 중심 추출(`loss.py:21-34`)은 단순 가중합이라 **FPGA에서 저비용 구현 가능**(softmax+MAC). 반면 RANSAC/ElliFit은 호스트 CPU에 남기거나 고정 반복수 파이프라인으로 단순화 권장.
- **저지연 경로**: `--ellseg_ellipses=1` 모드(분할 중심 + 회귀 축)는 RANSAC을 우회 → 결정적 지연(latency-deterministic). 실시간 XR에 적합. `=-1`(분할맵만)은 더 가벼움.
- **재사용 포인트**: ① `my_ellipse`의 conic↔param 변환(고정소수 변환 후보), ② soft-heaviside 타원 마스킹, ③ 경량 회귀 헤드(`regressionModule`)는 좌표 출력 가속기 IP로 재사용 가능.
- **주의**: bilinear interpolate 업샘플(`RITnet_v3.py:75`)은 HW에서 라인버퍼·보간 로직 필요 — nearest(RITnet 방식, `densenet.py:70`)로 대체 시 정확도 영향 평가 필요.

---

## 9. 근거 표기

- **확인(코드 라인 근거)**: 모델 구조(`RITnet_v3.py`), 손실 전체(`loss.py`, `RITnet_v3.py:268-324`), 추론·후처리(`evaluate_ellseg.py`), 타원 수학(`helperfunctions.py:13-310`), 메트릭(`utils.py:119-169`), 학습 루프(`train.py`), 데이터로더(`CurriculumLib.py:34-120`).
- **추정**: RITnet_v1·v2·v4~v7 세부 차이(v3만 정밀 분석, 나머지는 등록만 확인); "FPGA 가속" 프로젝트 맥락(8장은 일반 추론).
- **확인 불가(미열람)**: `data_augment.py` 증강 세부, `dataset_generation/Extract*.py` 변환 로직, `pytorchtools.py` EarlyStopping 세부, `deepvog_pytorch.py` 구조, `test.py` 전체 — 본 분석 범위에서 제외/요약 수준.
