# FACET (EvEye / eye-tracking) 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\FACET`
> 도구: Glob / Grep / Read (라인 근거 기반). 모든 경로는 절대경로.
> 근거 표기 규칙: 코드에서 직접 확인한 것은 `파일:라인`으로 명시. 코드 근거 없이 추론한 것은 **(추정)**, 코드만으로 판단 불가한 것은 **(확인 불가)** 로 표기.

---

## 1. 개요 (Overview)

### 1.1 목적
이벤트 카메라(DVS, DAVIS346) + RGB 프레임 기반의 **동공/시선 추적(pupil/eye tracking)** 벤치마크 프레임워크다. 내부 패키지명은 `EvEye`(Event-Eye)이며, README 기준 디렉토리명은 `eye-tracking`이다(`\\...\FACET\README.md:26-29`, `:72-82`). 두 종류의 출력 태스크를 한 코드베이스에서 모두 지원한다.

- **동공 중심 좌표(center) 회귀** — `DavisEyeCenter` 계열, 대표 모델 `TennSt`
- **동공 타원(ellipse) 5-파라미터 검출 + 세그멘테이션(mask)** — `DavisEyeEllipse` 계열, 대표 모델 `EPNet`

### 1.2 원논문/챌린지 (근거)
- **데이터셋**: EV-Eye. README에 원본 데이터셋 다운로드 및 `EV_Eye_dataset/raw_data/Data_davis_labelled_with_mask`(9000+ 마스크 라벨 이미지) 구조가 명시됨(`README.md:5-25`). GitHub `Ningreka/EV-Eye` 링크도 명시(`README.md:9`).
- **AIS 2024 Event-based Eye Tracking Challenge 연관**: `tools/inference.py:36-37`이 `submission.csv`를 `["row_id", "x", "y"]` 컬럼으로 생성 → 캐글식 챌린지 제출 포맷. 또한 좌표를 `*346`, `*260`(DAVIS346 해상도)으로 역정규화(`inference.py:29-30`). **(추정)** 이는 AIS2024 Eye Tracking Challenge 제출 파이프라인.
- **TennSt = TENNs(Temporal Neural Networks) 계열 모델 (추정)**: 스트리밍 인과(causal) 시공간 CNN 구조와 채널/네이밍이 BrainChip TENNs-Eye/AIS2024 우승 계열 코드와 동일 패턴. 코드 자체에 "TENNs" 문자열 출처 명시는 없음 **(확인 불가)**.
- **EPNet**: CenterNet(CtdetLoss, focal heatmap, offset reg) + 회전 타원용 GWD(Gaussian Wasserstein Distance) loss 조합 → 회전 객체 검출(mmrotate 계열) 아이디어를 동공 타원에 적용 **(추정)**.
- "FACET"이라는 명칭의 직접 정의는 repo 내에서 발견되지 않음. 디렉토리/배포 명칭으로 추정 **(확인 불가)**.

### 1.3 입력 / 출력
- **입력**:
  - 이벤트 스트림 `(t, x, y, p)` txt 포맷(`README.md:30-37`) → 프레임 스택(2채널: ON/OFF polarity)으로 누적.
  - 옵션으로 RGB 프레임(`DavisEyeEllipse_RGBUNet.yaml`, `UNet`/`DeepLabV3` 경로).
- **출력**:
  - EyeCenter: 정규화된 동공 좌표 `(x, y)` + openness(눈 깜빡임 여부)(`losses.py:189-218`).
  - EyeEllipse: 타원 5-파라미터 `(x, y, a, b, θ)` + 동공 마스크(`DavisEyeEllipseDataset.py:304-318`, `EPNet.py:43`의 head_dict).

---

## 2. 디렉토리 구조 (자체 소스 + 제외 대상 명시)

```
FACET/
├─ README.md                  # 데이터 준비 + 실행 가이드
├─ main.py                    # AWS SageMaker 학습 런처 (자체)
├─ requirements.txt           # 의존성
├─ configs/                   # YAML 설정 (자체, 핵심)
│   ├─ DavisEyeEllipse_EPNet.yaml
│   ├─ MemmapDavisEyeCenter_TennSt.yaml
│   ├─ DavisEyeEllipse_{RGBUNet,EventUNet,ElNet,EPNet}.yaml
│   └─ Test*.yaml, OutputGroundTruth.yaml
├─ tools/                     # 학습/검증/추론 진입점 (자체, 핵심)
│   ├─ train.py  validate.py  validate10times.py  inference.py
├─ EvEye/                     # 메인 패키지 (자체 전부)
│   ├─ model/
│   │   ├─ model_factory.py             # 모델 레지스트리
│   │   ├─ DavisEyeCenter/TennSt.py     # ★ 시공간 CNN (center)
│   │   ├─ DavisEyeEllipse/
│   │   │   ├─ EPNet/EPNet.py           # ★ CenterNet식 타원 검출 (LightningModule)
│   │   │   ├─ EPNet/Backbone/{MobileNetV3,ResNet18}Backbone.py
│   │   │   ├─ EPNet/Neck/{FPN,SSD}.py
│   │   │   ├─ EPNet/Head/EPHead.py
│   │   │   ├─ EPNet/{Loss,Metric,Predict,utils}.py
│   │   │   ├─ ElNet/ElNet.py           # DLA34+DCNv2 (CenterNet 원형, 대형)
│   │   │   ├─ UNet/UNet.py             # 세그멘테이션 학습용
│   │   │   └─ EllipseMobileNet.py
│   │   ├─ DavisWithMask/{UNet,DeepLabV3}.py
│   │   └─ CitiBike/ConvLSTM.py         # 참고/실험용 (eye-tracking과 무관)
│   ├─ dataset/
│   │   ├─ dataset_factory.py
│   │   ├─ DavisEyeCenter/{Memmap,Npy,Dat,...}DavisEyeCenterDataset.py, losses.py
│   │   └─ DavisEyeEllipse/{DavisEyeEllipseDataset,utils}.py
│   ├─ utils/
│   │   ├─ tonic/functional/ToFrameStack.py    # ★ 이벤트→프레임 표현
│   │   ├─ tonic/slicers/Slice*.py
│   │   ├─ cache/Memmap*.py                     # memmap 캐시
│   │   ├─ dvs_common_utils/representation/{TimeSurface,FrameStack,Histgram}.py
│   │   ├─ dvs_common_utils/processor/*Affine*.py  # 이벤트 증강
│   │   └─ scripts/exportONNX.ipynb            # ONNX 내보내기
│   ├─ logger/  callback/                       # Lightning 보조
│   └─ ...
└─ tests/                     # 시각화/스모크 테스트 (.py + .ipynb)
```

### 분석에서 제외한 대상 (자체 코드 아님 / 비핵심)
- `.git/` 전체 (hooks 샘플, 메타데이터).
- `EvEye/utils/tonic/` 중 `tonic` 라이브러리 미러 학습용(`tonicLearning.ipynb`) — 외부 `tonic` 패키지를 import하여 사용(`DavisEyeEllipseDataset.py:14-15`).
- `ElNet/ElNet.py`의 DLA34 + **DCNv2(deformable conv)**: CenterNet 원본(xingyizhou/CenterNet)에서 이식된 외부 코드 성격이 강하고, ONNX 변환 실패의 원인(`exportONNX.ipynb` cell-7: `No Op registered for DCNv2`). 본 분석에서는 핵심 자체모델(EPNet/TennSt) 대비 보조로만 다룸.
- 대용량 `.ckpt`/데이터셋(`/mnt/data2T/...` 경로로 외부 마운트, repo에 미포함).
- 다수의 `.ipynb`(시각화/전처리 실험용)는 핵심 알고리즘이 `.py`에 있어 보조 취급.

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.0 데이터 흐름 (거시)
```
이벤트 txt/캐시 ──(slicer)──> event_segment(t,x,y,p)
   └─ to_frame_stack_numpy ──> (T,2,H,W) 프레임스택(ON/OFF)
        ├─ EyeCenter:  (B,2,T,H,W) ──> TennSt(Conv3d) ──> (B,3,T,H',W') heatmap+offset ──> (x,y)
        └─ EyeEllipse: (B,2,256,256) ──> EPNet(MobileNetV3+FPN+multihead)
                                          ──> {hm, ab, trig, reg, mask} ──post_process──> (x,y,a,b,θ)
```

---

### 3.1 Backbone

#### (A) EPNet 백본 — MobileNetV3-Large
파일: `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\Backbone\MobileNetV3Backbone.py`
- `InvertedResidual`(`:67-123`): MobileNetV3 표준 블록. depthwise conv + Squeeze-Excite(`SELayer`, `:37-52`) + h-swish/h-sigmoid(`:19-35`). pw-linear로 채널 변환, `identity`(`:72`)로 residual.
- `MobileNetV3Backbone`(`:126-175`): `cfgs`(`:131-154`)는 입력 256×256 기준 stride 다운샘플 스테이지를 정의. 입력 2채널(이벤트 ON/OFF)을 받는 첫 conv(`:158`).
- `forward`(`:170-175`): 멀티스케일 4-feature 반환 — `out2(24ch), out3(40ch), out4(112ch), out5(160ch)` (네크에서 FPN 입력으로 사용). `in_filters=[24,40,112,160]`(`:129`).
- **주의**: `WEIGHT_PATH`(`:6`)에 사전학습 가중치 경로 하드코딩되어 있으나 클래스 내부에서 실제 로드 코드는 없음 → 사실상 미사용/외부 절대경로 의존 **(확인 불가, 실제 미로드)**.

#### (B) TennSt 백본 — 시공간(Spatiotemporal) Conv3d 스택
파일: `\\...\FACET\EvEye\model\DavisEyeCenter\TennSt.py`
- `SpatialBlock`(`:90-166`): `Conv3d` 커널 `(1,3,3)`, stride `(1,2,2)` → **공간만 다운샘플, 시간축 유지**. depthwise+pointwise 분리 지원(`:113-127`).
- `TemporalBlock`(`:169-259`): 커널 `(k,1,1)`(또는 full_conv3d 시 `(k,3,3)`). 시간축으로 **causal padding** `F.pad(input,(0,0,0,0,k-1,0))`(`:238-240`) → 미래 프레임 비참조(인과성 보장). 깜빡임 같은 시계열 정보 처리.
- `backbone` 구성(`:287-317`): `temporals=[True,False]*5`(`:285`)로 **시간블록과 공간블록을 번갈아** 10단 쌓음. depthwise는 마지막 `n_depthwise_layers`개만 적용(`:284`).
- 정규화: `CausalGroupNormBlock`(`:52-63`)은 GroupNorm을 시간통계 미사용 형태로 적용해 인과성 유지. `norms="mixed"`면 TemporalBlock 1층 BN + 2층 GN(`:191-193`).
- **스트리밍 추론 핵심**: `SpatialBlock._streaming_forward`/`TemporalBlock._streaming_forward`(`:158-166`, `:251-259`)는 FIFO 버퍼로 단일 프레임씩 처리 → **프레임당 inference** 가능. `streaming()`(`:357-364`)이 모든 서브모듈 streaming 토글, `reset_memory()`(`:366-369`)로 FIFO 초기화. → 저지연 on-device 추론 지향 설계의 직접 증거.

---

### 3.2 Neck

#### EPNet 네크 — FPN 변형 (4가지 mode)
파일: `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\EPNet.py`
- `EPNet.__init__`(`:133-201`)이 4가지 mode를 선택적으로 구성: `standard`, `light`, `fpn_2d`, `fpn_dw`.
- **`fpn_2d`(설정 기본값, `DavisEyeEllipse_EPNet.yaml:45`)**: `:178-186` — P5(160→512)→upsample, P4(112→256), P3(40→128), P2(24→64) top-down 결합. `forward` `:271-290`에서 각 단계 conv→upsample→concat→`final_conv(128→64)`.
- `fpn_dw`(`:188-196`): 동일 구조이나 `conv_dw`(depthwise separable)로 경량화 → **FLOPs 절감판**.
- `standard`(`:148-162`): SPP(SpatialPyramidPooling, `:92-104`) + 5-conv 반복(PANet식 더 무거움).
- 별도 `Neck/FPN.py`, `Neck/SSD.py` 파일도 존재하나 EPNet 본체는 네크를 내장(self-contained)으로 구현. 별도 파일은 실험적/대체 구현 **(추정)**.

> TennSt는 별도 neck 없음. 백본 출력 → head 직결(`forward` `:371-375`).

---

### 3.3 Head

#### (A) EPNet head — CenterNet식 멀티헤드
파일: `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\Head\EPHead.py`
- `EPHead`(`:10-44`): `head_dict`의 각 키마다 `Conv3x3(64→256)+ReLU+Conv1x1(256→classes)` 독립 헤드 생성(`:14-38`).
- head 종류(`:7`, `EPNet.py:34`): `hm`(heatmap 1ch), `ab`(장/단축 2), `trig`(각도 sin2θ/cos2θ 2), `reg`(서브픽셀 오프셋 2), `mask`(세그 1).
- heatmap bias 초기화 `-2.19`(`:35`): CenterNet의 focal loss 안정화 관행(sigmoid 후 낮은 prior).
- **각도 표현이 trig(2채널)**: 각도 불연속 문제를 회피하려 `sin2θ, cos2θ` 회귀(`DavisEyeEllipseDataset.py:141-146`의 `cal_trig`). 복원은 `restore_angle`(`Predict.py:236-244`)의 `atan2(sin2A,cos2A)/2`.

#### (B) TennSt head — detector / regressor
파일: `\\...\FACET\EvEye\model\DavisEyeCenter\TennSt.py:319-336`
- `detector_head=True`(설정 기본): TemporalBlock + Conv3d → **3채널 출력**(`:329`) = [pupil heatmap, x_mod, y_mod] (서브픽셀). 디코딩은 `process_detector_prediction`(`losses.py:153-186`)에서 argmax 위치 + 서브픽셀 오프셋.
- `detector_head=False`: 공간평균 후 Conv1d → 2채널 직접 좌표 회귀(`:332-336`, `forward` `:375`).

---

### 3.4 Loss

#### (A) EPNet — `CtdetLoss` (복합 손실)
파일: `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\Loss.py`
- `CtdetLoss`(`:402-507`): 7개 손실 가중합. `forward`에서 `close==0`(눈 뜬 프레임)만 마스킹해 손실 계산(`:435-485`).
  - `hm`: **Modified Focal Loss**(CornerNet/CenterNet식, `_neg_loss` `:9-29`, neg_weight=`(1-gt)^4`).
  - `ab`/`reg`: `RegL1Loss`(smooth L1, gather-by-index, `:61-69`).
  - `trig`: `TrigL2Loss`(MSE, `:97-105`).
  - `iou`: **GWDLoss**(`:323-377`) — 회전 타원을 2D 가우시안으로 보고 Gaussian Wasserstein Distance를 IoU 대용으로 사용. `gwd_loss`(`:258-320`)에 닫힌형 유도 주석 포함. 가중치 `iou_weight: 15`로 가장 큼(`yaml:53`).
  - `mask`: `BCEWithLogits`(`MaskLoss` `:380-386`).
- 가중치(`EPNet.py:24-32`, `yaml:46-54`): hm=1, ab=0.1, trig=1, reg=0.1, iou=15, mask=1, ang=0(비활성).
- **주의/스멜**: `GWDLoss.forward`가 `from predict import _topk`를 **함수 내부에서 import**(`:343`) — 모듈명이 `Predict`(대문자)와 불일치, 실제 학습 시 `iou_weight>0`이면 ImportError 위험 **(확인 불가: 별도 `predict.py` 존재 여부 미확인)**. 또한 `sqrt_newton_schulz_autograd`/`gwds_loss`가 `.cuda()` 하드코딩(`:116-124`, `:227`)되어 CPU/다중디바이스 비호환.

#### (B) TennSt — `tracking_loss` / `regression_loss`
파일: `\\...\FACET\EvEye\dataset\DavisEyeCenter\losses.py`
- `tracking_loss`(`:99-134`): heatmap focal(`(1-p)^gamma`) + 정답 픽셀 위치에서의 서브픽셀 smooth_l1(`center_mod`). `valid_mask`로 깜빡임/화면밖 제외(`:115`).
- `regression_loss`(`:84-96`): detector_head 미사용 시 좌표 직접 smooth_l1.
- `RegularizationLoss`(`:59-81`): ReLU 출력에 L1 패널티(activity regularization) — **희소성 유도**(MAC 절감/이벤트 친화적). `MacsEstimationHook`(`:15-57`)은 레이어별 MAC과 **입력 희소성 반영 MAC**(`macs_per_layer_with_sparsity` `:40-42`)까지 계측 → 효율/하드웨어 비용 의식의 직접 증거.

---

### 3.5 Dataset

#### (A) `DavisEyeEllipseDataset` (EPNet용)
파일: `\\...\FACET\EvEye\dataset\DavisEyeEllipse\DavisEyeEllipseDataset.py`
- `__getitem__`(`:190-320`) 파이프라인:
  1. 타원 라벨 + 5000개 고정-카운트 이벤트 세그먼트 로드(`:192-193`, `load_event_segment(..., 5000)`).
  2. 이벤트 증강(`DropEvent p=0.3`, `DropEventByArea`)(`:85-98`).
  3. `to_frame_stack_numpy`로 누적 → `causal_linear`/`bilinear`(`:202-223`), `cut_max_count(...,255)`로 클리핑(`:224-225`).
  4. Albumentations `ShiftScaleRotate`+`HorizontalFlip` 공간 증강(`:60-83`), 타원도 `transform_ellipse`(마스크 그려 cv2.fitEllipse 재추출, `:100-139`)로 동기 변환.
  5. **CenterNet GT 생성**(`:266-302`): `draw_umich_gaussian`으로 heatmap, `ab/ang/trig/reg/ind/reg_mask`와 다운샘플(`down_ratio=4`) mask 생성. `close` 플래그는 면적(`pupil_area=200`) 미만이면 1(깜빡임).

#### (B) `MemmapDavisEyeCenterDataset` (TennSt용)
파일: `\\...\FACET\EvEye\dataset\DavisEyeCenter\MemmapDavisEyeCenterDataset.py`
- 세그먼트 단위(`frames_per_segment=50`, `time_window=40000us`) memmap 로드(`:157-173`).
- `fixed_count`(예 5000)면 프레임마다 **타임스탬프+카운트 동시 슬라이싱**(`slice_events_by_timestamp_and_count` `:197-214`) → 카운트 고정 프레임화.
- `spatial_downsaple`(0.5배), `saptial_transform`(키포인트 동기 Albumentations, `DataAugmentation` `:28-100`), `temporal_transform`(시간 뒤집기 `temporal_flip` `:143-152`).
- 출력: `event_frames (2,T,H,W)`, `labels (2,T)` 정규화 좌표, `closes`=openness(`:236-242`).

> 캐싱: `EvEye/utils/cache/MemmapCacheStructedEvents.py` 등으로 memmap 사전 캐싱(README `:66-68`).

---

### 3.6 Train / Infer 루프

#### 학습 진입점 `tools/train.py`
파일: `\\...\FACET\tools\train.py`
- Lightning 기반. `make_dataloader → make_model → lightning.Trainer.fit`(`:31-55`).
- `seed_everything(42)`, `set_start_method("spawn")`, `float32_matmul_precision("medium")`(`:19-21`).
- **하드코딩**: `devices=[2]`(`:42`) — 특정 GPU 인덱스 고정. SageMaker용 S3 채널 분기(`:24-30`).
- 모델/데이터셋/로거/콜백 전부 **factory + YAML** 패턴(`model_factory.py:24-31`, registry는 `MODEL_CLASSES` dict `:12-19`).

#### 학습/검증 스텝 (모델 내장)
- EPNet `training_step`/`validation_step`(`EPNet.py:316-370`): val에서 `post_process`→`cal_batch_iou`/`cal_batch_ap`/`p_acc`/`cal_mean_distance` 계산, NaN 방어(`:353-354`), 에폭말 평균(`on_validation_epoch_end` `:359-370`).
- TennSt `training_step`/`validation_step`(`TennSt.py:401-447`): p1/p3/p5/p10 정확도 + distance 로깅, val에서 예측 프레임 시각화 후 텐서보드 image 기록.

#### 추론 진입점 `tools/inference.py`
- `streaming_inference`(`TennSt.py:338-355`)로 프레임 단위 추론 + **프레임별 wall-clock 시간 측정**(`:347-352`). 결과를 DAVIS346 해상도로 역정규화 후 `submission.csv` 생성(`inference.py:24-37`).
- EPNet 추론: `Predict.py`의 `predict`/`predict_txt`(`:267-391`) + `post_process`(`:141-192`) + `get_ellipse`(`:247-264`). `test_inference_time`(`:393-417`)로 평균 추론시간 측정.

---

## 4. 알고리즘 · 데이터 표현

### 4.1 이벤트 표현 (Event Representation)
파일: `\\...\FACET\EvEye\utils\tonic\functional\ToFrameStack.py`
- `to_frame_stack_numpy`(`:55-109`): `(t,x,y,p)` 이벤트를 `(n_time_bins, 2, H, W)` 텐서로 누적. 2채널 = polarity(ON/OFF) 분리. 시간은 `normalize`(`:4-23`)로 `[0, n_time_bins]` 정규화.
- 시간 보간 3종:
  - `nearest`(`:79-82`): 최근접 bin에 카운트 가산.
  - `bilinear`(`:84-96`): 인접 두 bin에 시간 가중 분배 — **voxel grid(시간축 선형 보간)** 표현.
  - `causal_linear`(`:98-104`): 미래 bin 비참조(인과). **스트리밍/저지연 추론과 정합**.
- 즉, 본 repo의 기본 표현은 **2-polarity frame-stack(= voxel grid의 polarity 분리형)**. TimeSurface(`dvs_common_utils/representation/TimeSurface.py`)와 Histgram도 보유하나 학습 파이프라인 기본은 frame-stack.
- 카운트 폭주 방지: `cut_max_count`로 상한 클리핑(`DavisEyeEllipseDataset.py:224-225`).

### 4.2 시계열 모델링 (ConvLSTM / SSM / Transformer / Conv3d)
- **Conv3d 시공간 모델**: TennSt가 채택(인과 TemporalBlock + SpatialBlock, §3.1B). SSM/Transformer는 **본 repo 미사용**.
- **ConvLSTM**: `EvEye/model/CitiBike/ConvLSTM.py`로 존재하나 `CitiBike`(자전거 수요 등) 도메인 예제로, eye-tracking 파이프라인과 무관 **(추정: 참고/이식 잔재)**.
- **SSM(Mamba)/Transformer**: 코드 내 import/구현 없음 **(확인: model_factory 등록 모델은 DeepLabV3/ConvLSTM/TennSt/EPNet/ElNet/UNet)**(`model_factory.py:12-19`).

### 4.3 후처리 (Post-processing)
- **EPNet (CenterNet decode)**: `post_process`(`Predict.py:141-192`) — `sigmoid → nms(maxpool, kernel=3, :103-111) → topk(K=100, :114-138) → gather offset/ab/trig`. 각도 복원 `restore_angle`(`:236-244`). 최종 타원은 `(64×64)` 출력 좌표를 `transform_ellipse`(`:218-233`)로 원해상도(260×346) 복원.
- **타원 피팅(ellipse fitting)**: 라벨 생성/증강 단계에서 `cv2.fitEllipse`(`DavisEyeEllipseDataset.py:122`) 사용. 추론 출력은 모델이 직접 5-파라미터 회귀 → 별도 외부 피팅 불필요.
- **TennSt decode**: `process_detector_prediction`(`losses.py:153-186`) — heatmap argmax 위치 + sigmoid 서브픽셀 오프셋으로 `(x,y)` 산출.

---

## 5. 학습 · 평가

### 5.1 데이터셋
- EV-Eye(DAVIS346, sensor_size `[346,260,2]`). 라벨: 이벤트 txt `(t,x,y,p)` + 타원 txt `(t,x,a,b,θ)`(`README.md:30-45`). 마스크는 `Data_davis_labelled_with_mask`의 `.h5`.
- split: `train/val` (일부 데이터셋은 `test`도). EyeCenter는 memmap 캐시 권장(`README.md:66-68`).

### 5.2 메트릭
- **EyeCenter (TennSt)**: `p_acc`(tolerance=1/3/5/10px) 픽셀 정확도, blink 제외 정확도, mean distance(`losses.py:189-218`). 체크포인트 모니터는 `val_p10_acc`(`MemmapDavisEyeCenter_TennSt.yaml:76-83`). 기존 best 가중치 파일명에 `0.9689`(p10=96.89%) 명시(`yaml:67`).
- **EyeEllipse (EPNet)**: `val_mean_distance`(중심 거리, 모니터), `p1/3/5/10_acc`, **IoU**(`cal_iou`: 두 타원 래스터화 후 교집합/합집합, `Metric.py:51-67`), **AP@0.5**(`cal_batch_ap` `:102-160`)(`EPNet.py:342-351`).

### 5.3 실행 명령어 (README 근거)
```bash
# 학습
python .../tools/train.py --config .../configs/DavisEyeEllipse_EPNet.yaml   # README:74-77
# 검증
python .../tools/validate.py --config .../configs/DavisEyeEllipse_EPNet.yaml # README:79-82
```
- `train.py`는 click 옵션 `-c/--config`(`train.py:16-17`). TennSt 학습은 `--config configs/MemmapDavisEyeCenter_TennSt.yaml`.
- 추론/제출: `tools/inference.py`(`submission.csv` 생성).
- 클라우드 학습: `main.py`(SageMaker `ml.g5.8xlarge`, `torch 2.2.0`, `:21-40`).

---

## 6. 의존성
파일: `\\...\FACET\requirements.txt`
- 학습: `torch==2.2`, `lightning`/`pytorch-lightning`, `lightning-bolts`.
- 이벤트/CV: `tonic`(이벤트 변환·슬라이서), `albumentations`(공간 증강), `opencv-python`, `scikit-image`.
- 모델: `timm`(StepLRScheduler, MobileNetV3 가중치).
- 효율 계측: `thop`(FLOPs/params 프로파일, `model_factory.py:91-98`).
- 데이터/클라우드: `h5py`, `natsort`, `pandas`(submission), `boto3`/`sagemaker`.
- 설정: `hydra-core`, `click`(실제는 click 사용). PyPI 미러는 칭화대(`requirements.txt:1`).
- ONNX: requirements에는 없으나 `exportONNX.ipynb`가 `onnx`/`torch.onnx` 사용.

---

## 7. 강점 · 한계

### 강점
- **멀티-태스크/멀티-모델 통합**: factory+YAML로 center/ellipse, 다양한 백본·네크 교체 용이(`model_factory.py`, `EPNet.py`의 4 mode).
- **저지연 추론 의식적 설계**: TennSt의 causal Conv3d + FIFO 스트리밍(`TennSt.py:158-166,251-259,338-369`), activity regularization과 sparsity-aware MAC 계측(`losses.py:40-42`).
- **회전 타원 검출의 정교함**: 각도 trig 표현 + GWD loss(`Loss.py:258-377`)로 각도 불연속/IoU 미분 문제 회피.
- **재현 가능 전처리**: Albumentations `ReplayCompose`로 이미지·라벨·타원 동기 증강(`DavisEyeEllipseDataset.py:100-139`).
- ONNX 내보내기 경로 존재(TennSt는 정상 export, `exportONNX.ipynb` cell-2/cell-5).

### 한계 / 코드 스멜
- **하드코딩 다수**: GPU `devices=[2]`(`train.py:42`), 절대경로 `/mnt/data2T/...`, `WEIGHT_PATH`(`MobileNetV3Backbone.py:6`), `.cuda()` 직접 호출(`Loss.py:116-124`).
- **GWDLoss 내부 `from predict import _topk`**(`Loss.py:343`): 모듈명 불일치로 잠재 ImportError. iou_weight>0 경로 신뢰성 의문 **(확인 불가)**.
- **ElNet ONNX 변환 실패**: DCNv2 커스텀 op 미등록(`exportONNX.ipynb` cell-7). 즉 ElNet 계열은 표준 배포 비친화.
- 다수 죽은 코드/주석 블록, 혼재된 중국어 주석, `(bug)` 표기 파일(`TorchEventFrameRandomAffine(bug).py`).
- 테스트는 시각화 스모크 위주, 단위 테스트 빈약.
- 양자화(PTQ/QAT) 코드 **부재** — 경량화는 depthwise/FPN-dw 수준에 그침.

---

## 8. 우리 프로젝트 시사점 (XR 시선추적 + FPGA 저지연 on-device 가속)

> 전제 **(추정)**: 본 분석의 상위 목표는 "XR 헤드셋용 이벤트 기반 시선추적을 FPGA에서 저지연·저전력으로 on-device 가속"으로 보임.

1. **TennSt = FPGA 이식 1순위 후보**.
   - causal Conv3d + 프레임별 FIFO 스트리밍(`TennSt.py:251-259`)은 **라인버퍼/시프트레지스터 기반 RTL·HLS dataflow와 1:1 매핑** 가능. 시간축 인과성이 보장되어 미래 프레임 버퍼링 불필요 → 지연(latency) 최소.
   - 입력이 2-polarity frame-stack(`ToFrameStack.py`)이라 이벤트→프레임 누적기(accumulator)를 FPGA 프런트엔드로 두면 호스트 부담 감소.
   - 연산이 Conv3d/Conv1d + GroupNorm/BN + ReLU로 단순 → **HLS 친화** (anthropic-skills의 algo2fpga로 Python→HLS C++ 변환 검토 권장).

2. **경량화·양자화 작업이 필요(현 repo 미제공)**.
   - 현재 FP32 학습만 존재. FPGA용으로는 **INT8 PTQ/QAT 추가**가 필수. BN/GroupNorm fold, h-swish/h-sigmoid(MobileNetV3) → FPGA 친화 활성화로 치환 필요.
   - `MacsEstimationHook`의 sparsity-aware MAC(`losses.py:15-57`)을 활용하면 **희소성 기반 PE 게이팅** 설계 근거로 재활용 가능.

3. **EPNet은 정확도(타원/세그) 강점이나 가속 난이도 높음**.
   - MobileNetV3의 SE-block(global avgpool→FC)과 FPN concat이 dataflow 파이프라인화 시 병목 **(추정)**. on-device용으로는 `fpn_dw` mode + SE 제거 변형을 검토.
   - CenterNet decode(nms maxpool + topk K=100)는 호스트 후처리로 분리하거나, K=1(동공 1개) 단순화로 FPGA 후처리 경량화 가능(`Predict.py:114-192`).

4. **이벤트 표현 가속**: `to_frame_stack_numpy`의 `causal_linear` 누적(`ToFrameStack.py:98-104`)은 단순 가산이라 **FPGA 이벤트 누적기 IP로 쉽게 구현**. 시간 정규화/보간만 고정소수점화하면 됨.

5. **배포 경로**: TennSt ONNX export가 동작하므로(`exportONNX.ipynb`), ONNX → (Vitis AI / FINN / 자체 HLS) 흐름이 현실적 출발점 **(추정)**. ElNet의 DCNv2는 FPGA 비친화이므로 제외 권장.

---

## 9. 근거 / 확인 불가 정리

| 항목 | 상태 | 근거 |
|---|---|---|
| EV-Eye 데이터셋 사용 | 확인 | `README.md:5-29` |
| AIS2024 챌린지 제출 포맷 | 추정 | `inference.py:36-37` submission.csv |
| TennSt = TENNs 계열 | 추정 | 구조 동일성; repo 내 "TENNs" 문자열 없음 → 출처 **확인 불가** |
| "FACET" 명칭 의미 | 확인 불가 | repo 내 정의 미발견 |
| 기본 이벤트 표현 = 2ch frame-stack | 확인 | `ToFrameStack.py:55-109`, dataset `__getitem__` |
| Conv3d 스트리밍/인과 설계 | 확인 | `TennSt.py:158-166,238-259,338-369` |
| GWD loss(회전 타원) | 확인 | `Loss.py:258-377` |
| GWDLoss `from predict import` 정상동작 | 확인 불가 | `Loss.py:343` 모듈명 불일치 |
| 양자화/FPGA 코드 | 확인(부재) | repo 전역 grep에 quantize/fpga/tensorrt 구현 없음, ONNX export만 존재 |
| MobileNetV3 사전학습 가중치 로드 | 확인 불가(미로드) | `MobileNetV3Backbone.py:6` 경로만 존재, 로드 코드 없음 |
| SSM/Transformer 사용 | 확인(미사용) | `model_factory.py:12-19` 등록 모델에 없음 |

---

### 참고 파일 목록 (핵심)
- `\\...\FACET\README.md`
- `\\...\FACET\tools\{train,validate,inference}.py`
- `\\...\FACET\EvEye\model\model_factory.py`
- `\\...\FACET\EvEye\model\DavisEyeCenter\TennSt.py`
- `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\{EPNet,Loss,Predict,Metric}.py`
- `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\Backbone\MobileNetV3Backbone.py`
- `\\...\FACET\EvEye\model\DavisEyeEllipse\EPNet\Head\EPHead.py`
- `\\...\FACET\EvEye\dataset\DavisEyeEllipse\{DavisEyeEllipseDataset,utils}.py`
- `\\...\FACET\EvEye\dataset\DavisEyeCenter\{MemmapDavisEyeCenterDataset,losses}.py`
- `\\...\FACET\EvEye\utils\tonic\functional\ToFrameStack.py`
- `\\...\FACET\EvEye\utils\scripts\exportONNX.ipynb`
- `\\...\FACET\configs\{DavisEyeEllipse_EPNet,MemmapDavisEyeCenter_TennSt}.yaml`
- `\\...\FACET\requirements.txt`
