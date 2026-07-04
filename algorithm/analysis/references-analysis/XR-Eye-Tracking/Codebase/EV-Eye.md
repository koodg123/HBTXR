# EV-Eye 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EV-Eye`
> 분석 방식: 실제 소스(Python 5개, MATLAB 핵심 7개)를 라인 단위로 정독. 모든 라인 근거는 절대경로+라인번호 표기.
> 근거 표기 규칙: 코드/README로 직접 확인된 사실은 "확인", 추론은 "추정", 코드만으로 알 수 없는 부분은 "확인 불가".

---

## 1. 개요

### 목적
EV-Eye는 **이벤트 카메라(DVS) + 프레임(APS)** 을 함께 쓰는 하이브리드 고주파 시선/동공 추적 데이터셋이자 벤치마크다. README.md(루트, 라인 21)에 따르면 동공을 **최대 38.4kHz**까지 추적하는 하이브리드 frame-event 벤치마킹 접근을 제안한다고 명시되어 있다(확인).

- 핵심 아이디어: **저주파(약 25Hz) 프레임에서 U-Net으로 동공을 세그먼트해 동공 중심을 절대 위치로 "리셋"하고, 그 사이를 고주파 이벤트 스트림으로 ICP 기반 평행이동(translation) 추정으로 보간/추적**한다(확인 — `matlab_processed/frame_event_pupil_track.m`).

### 원논문 / 챌린지
- 논문: *EV-Eye: Rethinking High-frequency Eye Tracking through the Lenses of Event Cameras*, Guangrong Zhao 외, **NeurIPS 2023 Datasets and Benchmarks Track**(README.md 라인 6~11, 확인).
- 데이터셋 규모(README.md 라인 43~49, 확인): 48명 참가자, 4세션, 두 대의 **DAVIS346** 이벤트 카메라, 150만+ 근안(near-eye) grayscale 이미지, 27억 이벤트 샘플, Tobii Pro Glasses 3로 수집한 270만 gaze reference(cross-modality 검증용).

### 입력 / 출력
- **입력(확인)**:
  - 프레임 분기: DAVIS346 APS grayscale 이미지 (해상도 **260×346**, 단일 채널). train.py 라인 92~93에서 `reshape(-1,1,260,346)`로 확인.
  - 이벤트 분기: `events.txt`의 `(timestamp, x, y, polarity)` 4-튜플 이벤트 스트림 (`frame_event_pupil_track.m` 라인 31, 확인).
  - gaze reference: Tobii `gazedata.txt`의 2D/3D 시선 (`d2x,d2y,d3x,d3y,d3z`) (`frame_event_pupil_track.m` 라인 37, 확인).
- **출력(확인)**:
  - Python: 동공 **이진 세그멘테이션 마스크**(2-class, 배경/동공). predict.py가 프레임마다 `*_mask.gif` 저장(라인 145~149).
  - MATLAB: **동공 중심 좌표 시계열** `[timeinterval, pixel_num, timestamp, center_x, center_y, frame_or_event]`(`frame_event_pupil_track.m` 라인 164, 확인) → 다항회귀로 화면상 **PoG(Point-of-Gaze)** 좌표 변환(`evaluation_on_gaze_tracking_with_polynomial_regression.m`, 확인).

---

## 2. 디렉토리 구조 (자체 코드 vs 제외 대상)

```
EV-Eye/
├─ train.py                         ← [자체] U-Net 학습 루프 (LOSO cross-validation)
├─ predict.py                       ← [자체] 마스크 추론/저장
├─ evaluate.py                      ← [자체] IoU / Dice 평가
├─ requirements.txt                 ← torch==1.9.0, numpy, tqdm, h5py, torchvision, argparse
├─ README.md                        ← [필독] 목적/데이터셋/실행법
├─ LICENSE                          ← CC BY-NC 4.0
├─ unet/
│  ├─ unet_model.py                 ← [자체] UNet 조립 (표준 U-Net)
│  ├─ unet_parts.py                 ← [자체] DoubleConv/Down/Up/OutConv
│  └─ __init__.py
├─ utils/
│  ├─ data_loading.py               ← [자체] BasicDataset/CarvanaDataset (predict 경로에서만 부분 사용)
│  ├─ dice_score.py                 ← [자체] dice_coeff/multiclass/dice_loss
│  └─ utils.py                      ← [자체] plot_img_and_mask (시각화)
├─ matlab_processed/                ← [자체, 알고리즘 핵심] 전처리·추적·평가 MATLAB
│  ├─ generate_pupil_mask.m         ← VGG 타원 라벨 → 이진 마스크(.h5 GT) 생성
│  ├─ frame_event_pupil_track.m     ← ★하이브리드 frame-event 동공 추적 메인★
│  ├─ calculate_optimal_translation.m ← ICP(kd-tree NN) 기반 평행이동 추정
│  ├─ pe_of_frame_based_pupil_track.m ← 프레임 기반 PE(픽셀오차)
│  ├─ pe_of_event_based_pupil_track.m ← 이벤트 기반 PE
│  ├─ evaluation_on_gaze_tracking_with_polynomial_regression.m ← DoD(시선 각오차)
│  ├─ frame_event_pupil_track_result_find_tobii_reference.m ← Tobii 시간정렬
│  ├─ read_csv.m / sort_nat.m       ← VGG csv 파싱 / 자연정렬 헬퍼
│  ├─ plot_*.m, *_plot.m            ← 결과 시각화
│  └─ Display Dot/                  ← 자극(dot) 디스플레이 스크립트(데이터수집용)
└─ pictures/                        ← [제외] 문서용 이미지
   .git/, .idea/, __pycache__/      ← [제외] VCS/IDE/캐시
```

### 제외 대상 (분석 비포함)
- `.git/`, `.idea/`, `**/__pycache__/*.pyc`: 버전관리·IDE·바이트코드 (제외).
- `pictures/*.png|jpg`: 문서 삽화 (제외).
- **대용량 데이터·체크포인트**: `EV_Eye_dataset/`(raw_data, processed_data, Pre-trained_models)는 repo에 미포함이며 외부 다운로드(README.md 라인 31, 142~167). 코드에서 경로만 참조 (분석 대상 아님, 확인).
- **외부 프레임워크 원본**: U-Net 구현은 잘 알려진 milesial/Pytorch-UNet 계열로 추정됨(unet_parts.py 라인 64~65 주석에 milesial 커밋 링크 존재). `CarvanaDataset`(data_loading.py 라인 84~86)도 해당 origin의 잔재로 보임(추정).

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.1 모델 — U-Net (unet/unet_model.py, unet/unet_parts.py)

**표준 U-Net**이며 동공 분야 특화 변형은 없다(확인).

- `UNet.__init__`(unet_model.py 라인 6~22):
  - `n_channels=1, n_classes=2`로 사용(train.py 라인 156; predict.py 라인 124, 확인). 즉 **grayscale 단일 채널 입력, 2-class(배경/동공) 출력**.
  - 인코더: `DoubleConv(1,64)` → `Down(64,128)` → `Down(128,256)` → `Down(256,512)` → `Down(512,1024)`(bilinear=False, factor=1).
  - 디코더: `Up(1024,512)`→`Up(512,256)`→`Up(256,128)`→`Up(128,64)` → `OutConv(64,2)`.
- `forward`(라인 24~36): 전형적 skip-connection U-Net. logits 반환(softmax 미적용).
- 부품(unet_parts.py):
  - `DoubleConv`(라인 8~25): `Conv3x3(bias=False)→BN→ReLU` ×2. **이중 컨볼루션 블록**.
  - `Down`(라인 28~38): `MaxPool2d(2)` + `DoubleConv`.
  - `Up`(라인 41~67): `ConvTranspose2d`(stride 2) 또는 `Upsample(bilinear)`. 라인 58~62에서 skip 텐서와 크기 불일치를 `F.pad`로 보정 → **260×346처럼 비2의거듭제곱 해상도에서도 동작**(확인). concat 후 DoubleConv.
  - `OutConv`(라인 70~76): `Conv1x1`로 클래스 수 매핑.

> 경량화·HW 관점(추정): 입력단 64채널 + 4단 다운샘플로 파라미터/연산량이 큰 풀-사이즈 U-Net이다. bottleneck 1024 채널. on-device/FPGA 이식 시 채널 축소·깊이 절감이 1차 후보(아래 7장).

### 3.2 Loss (train.py 라인 163, 186~189 + utils/dice_score.py)

복합 손실 = **CrossEntropy + Dice loss**(확인):
```
loss = CrossEntropyLoss(masks_pred, true_masks)
     + dice_loss(softmax(masks_pred), one_hot(true_masks), multiclass=True)
```
- `criterion = nn.CrossEntropyLoss()` (train.py 라인 163).
- `dice_loss`(dice_score.py 라인 35~39): `1 - multiclass_dice_coeff(...)`, `reduce_batch_first=True`.
- `multiclass_dice_coeff`(라인 25~32): 채널별 `dice_coeff` 평균.
- `dice_coeff`(라인 5~22): `(2·inter + ε)/(sets_sum + ε)`, ε=1e-6. 배치 차원 평균 처리.
- 클래스 불균형(동공이 화면의 작은 영역) 대응으로 CE+Dice 조합을 쓴 것으로 추정.

### 3.3 Dataset / DataLoader

두 갈래가 공존한다. **실제 학습 경로는 h5 직접 로딩**이고, `BasicDataset`은 predict.py 시점에만 import되며 학습에는 미사용에 가깝다(확인).

- **학습(train.py 라인 83~116, 125~154)**: HDF5(`Data_davis_labelled_with_mask/*.h5`)에서 `f['data']`, `f['label']`을 직접 읽음.
  - `data.T.reshape(-1,1,260,346)`, `label.T.reshape(-1,260,346)`(라인 92~93).
  - 정규화: `torch.from_numpy(train_data)/255`(라인 105) → **[0,1] 스케일**.
  - `Data.TensorDataset` + `DataLoader(batch_size=8, num_workers=4)`(라인 110~116). **하드코딩 batch_size=8**(인자 `batch_size`는 사실상 무시됨, 확인 — 잠재적 버그/혼선).
- **`BasicDataset`(utils/data_loading.py 라인 12~81)**: 이미지/마스크 디렉토리 페어링 일반 세그 데이터셋.
  - `preprocess`(라인 27~49): scale 리사이즈(mask는 NEAREST, image는 BICUBIC), `/255` 정규화, 채널축 추가/transpose.
  - `__getitem__`(라인 61~81): `{'image','mask'}` dict 반환.
  - `CarvanaDataset`(라인 84~86): `mask_suffix='_mask'`. **외부 origin 잔재**로 EV-Eye 본 파이프라인과 무관(추정).

> 주의: train.py는 `from utils.data_loading import BasicDataset, CarvanaDataset`(라인 15)를 import하지만 실제 학습 루프는 h5 TensorDataset만 사용. evaluate.py도 `batch['image']`가 아니라 `image, mask_true = batch`(라인 16)로 TensorDataset 형식을 기대 → **학습/평가 데이터 경로 = h5 텐서 경로로 일관**(확인).

### 3.4 학습 루프 — LOSO 교차검증 (train.py `train_net`, 라인 39~222)

핵심 설계는 **참가자 단위 Leave-One-Subject-Out(LOSO) 교차검증을 48회 반복**한다는 점이다(확인).

- 루프 구조(라인 68~80): `userlist = 1..48`. 각 `user`마다 자신을 제외(`userlist_remain.remove(user)`)한 나머지 47명으로 학습, 해당 user로 검증.
- 세션: `orders = ["1_0_2","2_0_1","2_0_2"]`(라인 69) — 라벨이 있는 3개 세션만 사용(확인).
- 학습 셋업(라인 156~164):
  - 모델 매 user마다 **새로 초기화**(`net = UNet(...)`, 라인 156) → **48개 사용자별 모델**을 각각 학습·저장(라인 221, `user{N}.pth`). README.md 라인 221에서도 "48명 좌/우 각각의 pre-trained" 명시(확인).
  - `optim.Adam(net.parameters())` (lr 기본값 사용; 인자 `learning_rate`는 전달되나 optimizer에 미반영 → **lr 인자 무시**, 확인).
  - `ReduceLROnPlateau(mode='max', patience=2)` — Dice 최대화 목표(라인 161).
  - `GradScaler(enabled=amp)` + `autocast` — **AMP 혼합정밀** 지원(라인 162, 183).
- 학습 스텝(라인 167~199): forward → `CE + dice_loss` → `grad_scaler.scale(loss).backward()` → step/update.
- 주기적 검증(라인 201~211): `division_step = n_train//(10*batch_size)`마다 `evaluate(net, val_loader, device)`로 Dice/mIoU 산출 후 `scheduler.step(val_score)`.
- 체크포인트(라인 213~222): `ui_result.txt`에 user별 dice/miou 기록 + `user{N}.pth` 저장.

> 코드 잔재 위험: 라인 89~90, 92, 129 등의 `f['data'].value`는 구버전 h5py API(현재는 `f['data'][()]`)로 **h5py>=3.x에서 동작 실패 가능**(확인 — 호환성 이슈).

### 3.5 추론 — predict.py (라인 18~150)

- `predict_img`(라인 18~49):
  - 입력 PIL → `np.asarray[np.newaxis]` → `/255` → `unsqueeze(0)` (배치/채널 추가). **단일 채널 grayscale 가정**(확인).
  - `net.eval()` + `torch.no_grad()`로 forward.
  - 2-class이므로 `softmax(dim=1)[0]` → `transforms.Resize`로 원본 크기 복원 → `argmax`로 one-hot 마스크 반환(라인 33~49).
- main(라인 91~150): user/eye 지정, 4세션(`'1_0_1','1_0_2','2_0_1','2_0_2'`) 전 프레임에 대해 추론, 사용자별 모델(`{whicheye}/user{N}.pth`) 로드(라인 129~131), `*_mask.gif`로 저장(라인 145~149).

> 버그성 경로: 라인 110 `glob.glob(os.path.join(origin_data_dir, '/frames/', '*.png'))` — `'/frames/'`의 선행 슬래시는 `os.path.join`을 절대경로로 리셋시켜 의도와 다른 경로가 됨(확인 — 잠재 버그).

### 3.6 평가 — evaluate.py (라인 8~72)

- `evaluate`(라인 8~52): 검증 로더 순회하며 **Dice score**(multiclass_dice_coeff, 배경 제외 `[:,1:]`)와 **mIoU**(`iou_mean`)를 동시에 계산(라인 33, 42).
- `iou_mean`(라인 55~72): 배경(class 0) 제외, class 1(동공)에 대한 IoU = `inter/union`. union=0이면 NaN 제외(확인).
- 즉 README.md 라인 136~138의 "IoU and F1 score" 중 IoU/Dice를 여기서 산출. (F1=Dice는 이진세그에서 동치, 추정.)

---

## 4. 알고리즘 · 데이터 표현 (이벤트 표현 / 프레임-이벤트 융합 / 후처리)

### 4.1 GT 마스크 생성 — 타원 라벨 → 이진 마스크 (generate_pupil_mask.m)
- VGG Image Annotator로 동공을 **타원(major/minor axis, tilt, center)** 으로 9,011장 라벨(README.md 라인 66, .m 라인 5~7, 확인).
- `read_csv.m`(라인 7~33)이 VGG csv의 `region_shape_attributes`를 파싱해 `[cx,cy,rx,ry,theta]` 추출(라인 17~21). 동공 미라벨 행은 0으로 채움(라인 24~28).
- 타원 파라메트릭 곡선 생성(generate_pupil_mask.m 라인 33~46) 후 `inpolygon`으로 260×346 그리드를 픽셀별 내부/외부 판정해 이진 마스크 `I_new` 생성(라인 48~56) → `/data`(원본), `/label`(마스크)로 h5 저장(라인 75, 주석 처리됨).

### 4.2 이벤트 표현
- EV-Eye는 이벤트를 **그리드/voxel/time-surface 등 텐서로 누적하지 않는다**. 대신 **raw 이벤트 포인트 `(t,x,y,polarity)` 를 직접 사용**한다(`frame_event_pupil_track.m` 라인 31, 확인).
- 추적 시 동공 경계 부근의 이벤트만 **공간 필터링**으로 골라 "candidate point set"을 만든다: 이벤트가 현재 동공 중심으로부터 거리 `radius*0.8 ~ radius*1.2` 환형(annulus) 안에 있을 때만 채택(라인 124, 확인). 여기서 `radius`는 동공 edge 픽셀들의 중심까지 평균거리(라인 111).
- polarity는 candidate 선별에서 사실상 미사용(필터 조건에 polarity 없음, 확인). `calculate_optimal_translation.m` 시그니처에 `polarity_list` 인자가 있으나 함수 본문에서 사용하지 않음(라인 2, 확인 — 미사용 잔재).

### 4.3 프레임-이벤트 융합 (★핵심, frame_event_pupil_track.m)
하이브리드 추적 메인 루프(라인 76~161). 패턴: **프레임에서 절대 위치 갱신 → 다음 프레임 전까지 이벤트로 상대 보간 갱신**.

1. **시간 정렬**(라인 39~62): Tobii TTL/send 시간과 DAVIS creation_time을 이용해 frame/event/tobii 타임라인을 공통 기준으로 정렬. 프레임 드롭 보정으로 역행 타임스탬프에 `+40ms`(=25fps) 보정(라인 23~27, 확인).
2. **프레임 기반 갱신**(라인 78~105):
   - 예측 마스크 `*.gif` 로드(라인 79).
   - `imclose(disk5)`로 **IR 반사광(pupil 내부 밝은 점) 메움**(라인 80, se 라인 66).
   - **k-means(k=1) 기반 노이즈 제거**: 중심까지 거리 `D>2.5·mean(D)`인 픽셀 제거(라인 84~88, 확인).
   - 동공 픽셀 → 중심 `(mean(xind),mean(yind))`, 에지 `edge(...,'Canny')`(라인 89~97).
   - 결과를 리스트에 push, `frame_or_event=1`로 표시(라인 99~104).
3. **이벤트 기반 갱신**(라인 110~158):
   - 현재 프레임 이후 첫 이벤트부터 다음 프레임 타임스탬프(`end_event_time`)까지 순회(라인 112~113).
   - annulus 필터(4.2)로 candidate 누적, **20개 모이면 1회 갱신**(라인 129).
   - candidate `P`(이벤트), `Q`(동공 edge)에 대해 `calculate_optimal_translation(P,Q)`로 최적 평행이동 `Ts` 추정(라인 132).
   - **update_factor=0.3 으로 동공 edge와 center를 점진 갱신**(EMA형, 라인 134~138):
     `center = center - Ts*0.3`. 갱신 타임스탬프 = candidate 이벤트 시각 평균(라인 140).
   - `frame_or_event=0`으로 표시(라인 144). 다음 프레임 도달 시 break(라인 153).
4. **결과 저장**(라인 164): `[timeinterval, pixel_num, timestamp, center_x, center_y, frame_or_event]`.

> 즉 융합은 네트워크 레벨 fusion이 아니라 **알고리즘 레벨(파이프라인) fusion**이다: U-Net(프레임)이 절대 앵커를, ICP(이벤트)가 고주파 보간을 담당(확인). 이벤트당 20개 누적·1갱신이 38.4kHz 추적의 근거 메커니즘으로 추정.

### 4.4 ICP 평행이동 추정 (calculate_optimal_translation.m, 라인 1~27)
- `createns(Q,'kdtree')`로 동공 edge에 kd-tree 구축(라인 3).
- 최대 50회 반복(라인 6) 동안: `knnsearch`로 각 이벤트 P의 최근접 edge 점 탐색(라인 8) → 평균 변위 `Ts_temp = mean(mapPoint - P)`(라인 12~13) → P를 그만큼 이동·누적(라인 19~23). 변위가 0.001 미만이면 수렴 종료(라인 15~17).
- **회전/스케일 없이 평행이동만 추정하는 단순화된 ICP**(확인). 동공이 짧은 시간 내 작게 움직인다는 가정에 적합(추정).

### 4.5 후처리 / 메트릭 산출
- **PE(프레임)** `pe_of_frame_based_pupil_track.m`: 예측 마스크 중심 vs VGG GT 타원 중심 간 **유클리드 픽셀오차**(라인 90, 확인).
- **PE(이벤트)** `pe_of_event_based_pupil_track.m`: 직전 프레임에서 초기화 후 이벤트로 추적한 중심 vs 다음 GT 타원 중심 간 유클리드오차(라인 154~156, 확인). 즉 **이벤트 추적이 프레임 간격 동안 얼마나 정확히 따라가는지** 측정.
- **DoD(시선 각오차)** `evaluation_on_gaze_tracking_with_polynomial_regression.m`:
  - blink 제거: 동공 픽셀수가 평균의 0.2배 미만이면 blink로 보고 전후 3프레임 구간 제거(라인 24~48, 확인).
  - Tobii reference 시간차 2ms 초과/0/NaN 제거(라인 53~61).
  - **다항회귀 보정**: 동공중심(정규화 x/346, y/260) → Tobii 화면좌표로 `poly53` 표면 적합(라인 71~77). 약 60~66개 샘플로 캘리브레이션(라인 73, 82).
  - 나머지로 PoG 추정 → reference와의 화면거리(라인 94~101) → **각오차 = atan(mean_dist/905)·180/π** (905=Tobii 가상스크린 깊이, 라인 113~114, 확인).

---

## 5. 학습 · 평가 (데이터셋 / 메트릭 / 명령어)

### 데이터셋
- `EV_Eye_dataset/raw_data/{Data_davis, Data_davis_labelled_with_mask, Data_tobii}` + `processed_data/{Data_davis_predict, Frame_event_pupil_track_result, Pixel_error_evaluation, Pre-trained}`(README.md 라인 156~167, 확인). repo에는 미포함, 외부 다운로드.

### 메트릭 (README.md 라인 136, 확인)
| 메트릭 | 구현 | 대상 |
|---|---|---|
| IoU / F1(Dice) | evaluate.py (`iou_mean`, `multiclass_dice_coeff`) | 프레임 동공 세그멘테이션 |
| PE (frame) | pe_of_frame_based_pupil_track.m | 프레임 동공 중심 픽셀오차 |
| PE (event) | pe_of_event_based_pupil_track.m | 이벤트 추적 중심 픽셀오차 |
| DoD (각오차) | evaluation_on_gaze_tracking_with_polynomial_regression.m | 시선방향 차이(deg) |

### 명령어 (README.md 라인 191~262, 확인)
```bash
# 학습 (48-fold LOSO, 사용자별 .pth 생성)
python train.py --whicheye R --batch-size 8 --epochs 5

# 추론(마스크 gif 생성)
python predict.py --whicheye R --user 1 --output ./EV_Eye_dataset/processed_data/Data_davis_predict

# IoU/Dice 평가
python evaluate.py
```
```matlab
% 픽셀오차
pe_of_frame_based_pupil_track.m ; pe_of_event_based_pupil_track.m
% 하이브리드 추적 → Tobii 정렬 → 시선 각오차
frame_event_pupil_track.m
frame_event_pupil_track_result_find_tobii_reference.m
evaluation_on_gaze_tracking_with_polynomial_regression.m
```
- 학습 하이퍼파라미터(확인): epochs 기본 5, batch 하드코딩 8, Adam(기본 lr), AMP optional, 입력 1×260×346, 2-class.

---

## 6. 의존성

- **Python(requirements.txt, 확인)**: `torch==1.9.0`, `numpy>=1.21.0`, `tqdm`, `h5py>=3.2.1`, `torchvision>=0.10.0`, `argparse`. matplotlib(utils.py)도 사실상 필요(추정 — requirements 미기재).
- **MATLAB(README.md 라인 224~229, 확인)**: `io`, `curvefit`(Octave forge 패키지). `kmeans`, `imclose`, `edge`, `createns/knnsearch`, `fit('poly53')` 등 Image Processing/Statistics/Curve Fitting Toolbox 사용(확인).
- **라이선스**: CC BY-NC 4.0 (비상업, README.md 라인 311~313, LICENSE, 확인).
- **호환성 리스크(확인)**: train.py의 `f['data'].value`는 h5py 3.x에서 제거됨 → requirements의 `h5py>=3.2.1`과 충돌. torch 1.9 고정도 최신 환경에서 빌드 마찰 가능.

---

## 7. 강점 · 한계

### 강점
- **하이브리드 설계의 명료함**: 프레임=절대 앵커(U-Net), 이벤트=고주파 보간(ICP)으로 역할 분리가 깔끔. 38.4kHz 추적의 메커니즘이 코드로 추적 가능(확인).
- **저주파 NN + 경량 고전 CV의 조합**: 이벤트 경로가 딥러닝 없이 kd-tree NN + 평행이동만으로 동작 → on-device/저지연 친화적(추정).
- **엄격한 평가 프로토콜**: 48-fold LOSO, 세션별 분리, blink/reference 정제, 화면-각 변환까지 end-to-end 메트릭 구비(확인).
- **재현성**: 데이터셋+사전학습+중간 결과(processed_data)까지 공개 명시(확인).

### 한계
- **모델 미경량화**: 풀-사이즈 U-Net(64→1024ch, ConvTranspose), 양자화/프루닝/distillation 코드 전무(확인). edge/FPGA 직접 이식엔 무겁다.
- **융합이 네트워크 레벨이 아님**: 학습 가능한 frame-event fusion이 없고, 이벤트 추적은 손튜닝 임계값(radius 0.8~1.2, 20점, update_factor 0.3, k-means 2.5·mean) 의존(확인). 일반화/적응성 제한 추정.
- **사용자별 모델 + 사용자별 다항 캘리브레이션**: 48개 모델 + per-user poly53 → 신규 사용자 zero-shot 대응이 어려움(확인).
- **코드 품질 이슈**: lr/batch 인자 무시(train.py 160·113), predict.py glob 경로 버그(라인 110), `.value` 구버전 API, polarity 미사용 잔재 등(모두 확인).
- **ICP 단순화**: 회전/스케일 미고려 평행이동만 → 큰 saccade/회전성 동공 변형에서 오차 누적 가능(추정).

---

## 8. 우리 프로젝트 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속 — 추정)

> 전제: 본 분석 컨텍스트상 우리 목표가 "XR 시선추적의 FPGA 기반 저지연 on-device 가속"이라는 것은 **추정**(코드/README에 FPGA 언급 없음, 확인 불가).

1. **이중 경로 가속 분할(권장)**: EV-Eye 구조는 우리에게 거의 그대로 매핑된다 —
   - *프레임 경로(U-Net, 저주파 25Hz)*: FPGA에서 **INT8 양자화 + 채널 프루닝**된 경량 U-Net으로 가속. 25Hz로만 돌리면 됨 → 연산 budget 여유. on-device-ai 관점에서 PTQ/QAT 후보.
   - *이벤트 경로(ICP/NN, 고주파 38.4kHz)*: 딥러닝이 아니므로 **전용 데이터패스(annulus 필터 + kd-tree NN + 평균 변위 누적)** 를 RTL/HLS로 구현하면 초저지연·초저전력 달성 가능. 38.4kHz 갱신을 FPGA 파이프라인으로 흡수하는 것이 핵심 차별화 포인트.
2. **경량화 타깃 수치**: 입력 1×260×346, 2-class는 작은 출력이므로 U-Net 채널을 32→512 또는 그 이하로 줄여도 동공 세그 정확도 유지 여지(추정). depthwise-separable conv / U-Net 변형(예: 본 디렉토리 내 RITnet, EllSeg 분석본과 비교)으로 파라미터 수십배 절감 가능.
3. **고정소수점 친화 알고리즘**: 이벤트 경로는 곱셈이 거의 없고(거리비교·평균·덧셈) **정수/고정소수점 산술 위주** → FPGA 자원(DSP) 사용 최소화 가능(추정). `calculate_optimal_translation`의 kd-tree는 HW에선 고정 동공 edge 점(수십~수백 개)에 대한 brute-force NN으로 치환하는 편이 파이프라이닝에 유리(추정).
4. **메모리 관점**: raw 이벤트 스트림을 텐서로 누적하지 않으므로 **이벤트당 스트리밍 처리**가 자연스럽다 → on-chip BRAM에 동공 edge 점 집합만 상주시키면 됨. EV-Eye의 "텐서화하지 않는" 표현이 FPGA 친화적이라는 점이 가장 큰 시사점.
5. **캘리브레이션 가속**: poly53 회귀는 학습이 아니라 작은 선형대수 → MCU/PS측에서 처리하고 PL은 추론만 담당하는 PS-PL 분할이 자연스러움(추정).
6. **비교 기준선**: 본 repo는 우리 시스템의 **정확도 상한(소프트웨어 baseline)** 으로 활용 가치가 높음. 양자화/프루닝 후 정확도 손실을 IoU/PE/DoD 메트릭으로 직접 측정 가능(확인 — 메트릭 코드 재사용).

---

## 9. 근거 표기 요약

- **확인(코드/README 직접 근거)**: U-Net 구조·2class·1ch·260×346, CE+Dice loss, 48-fold LOSO, h5 텐서 학습 경로, batch=8 하드코딩 / lr 무시, predict gif 출력, IoU/Dice/PE/DoD 메트릭 구현, 하이브리드 추적(프레임 앵커+이벤트 ICP 보간), annulus 필터·20점·update_factor 0.3, calculate_optimal_translation의 평행이동 ICP, poly53 캘리브레이션·905 각오차, CC BY-NC, 의존성, NeurIPS 2023.
- **추정**: U-Net이 milesial/Pytorch-UNet origin, F1=Dice 동치, FPGA 매핑 전략·경량화 수치, polarity 미활용 의미, ICP 단순화의 한계 영향.
- **확인 불가**: 실제 데이터셋 내용/사전학습 가중치(외부, 미포함), 학습 후 정량 성능 수치(논문/PDF 영역), FPGA 관련 사항(코드에 없음), `f['data'].value` 실행 성공 여부(환경 의존).
