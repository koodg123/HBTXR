# Swift-Eye 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\Swift-Eye`
> 분석 범위: Swift-Eye가 자체 커스텀한 소스만 정밀 분석. 외부 프레임워크 원본(`mmrotate/mmrotate/*`, `mmrotate/configs/*`, `mmrotate/tests/*`, `mmrotate/mmrotate.egg-info/*` 중 표준 부분)은 제외.
> 근거 표기 규칙: `파일:라인` 형식. 코드에 명시되지 않은 해석은 "추정", 코드/문서에서 확인되지 않으면 "확인 불가"로 표기.

---

## 1. 개요

### 1.1 목적
이벤트 카메라(event camera) 기반의 **anti-blink(눈 깜빡임에 강인한) 동공(pupil) 추적** 솔루션. 고주파(고FPS) near-eye 움직임 분석을 목표로 한다.
- 논문 제목(README.md:1): *"Swift-Eye: Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras"*
- 베이스 프레임워크: MMRotate(회전 객체 검출 벤치마크) 위에 구축됨(README.md:6). 동공을 **회전 bounding box(rotated bbox, OBB)** 로 검출/추적하고 이를 ellipse(타원)로 변환하는 구조.

### 1.2 원논문/챌린지
- 베이스 라이브러리: MMRotate (arXiv:2204.13317), mmdet 2.28.2, mmcv-full 1.7.2 (requirements.txt:96-98).
- 관련 데이터/도구: EV-Eye 데이터셋, timelens(이벤트 프레임 보간) 참조(README.md:28). 즉 이벤트 스트림을 **timelens로 보간하여 고FPS 회색조 프레임(interpolated_frames)** 을 생성한 뒤 그 위에서 동공을 추적하는 파이프라인.
- 성능 데모: 5000 FPS vs 25 FPS 시선 궤적 비교(README.md:48-50). "확인 불가" — 정량 메트릭(p-error 등) 수치는 코드 내 비포함, README에는 IoU/p-error 표 없음.

### 1.3 입력 / 출력
- **입력**: 단일 채널(grayscale)로 변환되는 PNG 프레임 시퀀스. test 단계는 `interpolated_frames/*.png`(test_interpolated.py:23) — timelens로 보간된 고FPS 이벤트 재구성 프레임으로 추정.
  - 검출/추적 헤드 입력: 정규화된 3채널 텐서(mean/std = 81.49/38.74, model_config.py:113-116). 회색조를 3채널로 복제하여 RGB 정규화 사용(추정 — to_rgb=True이나 평균이 3채널 동일).
  - UNet(세그멘테이션) 입력: 1채널 grayscale 그대로(model.py:263-267).
- **출력**:
  1. 동공 회전 bbox `[x_ctr, y_ctr, w, h, angle]`(rotated bbox, le90 표현). test_interpolated.py:32에서 `results[0][0][0][0:5]`로 추출.
  2. 이를 그대로 **타원(ellipse) 파라미터**로 해석하여 시각화: center=(x,y), 축=(w/2,h/2), 회전각=angle(test_interpolated.py:38, model.py:290).
  3. 동공 영역 **세그멘테이션 마스크**(UNet 2-class 출력, model.py:268-276) — open_extent(눈 뜬 정도) 계산에 사용.

---

## 2. 디렉토리 구조 (자체 커스텀 vs 외부 원본)

```
Swift-Eye/
├── README.md, requirements.txt           # 자체(설치/실행 안내, conda env spec)
└── mmrotate/                              # 외부 MMRotate 0.3.4 원본 fork
    ├── mmrotate/, configs/, tests/, ...   # [제외] 외부 원본 표준 코드
    │   └── mmrotate/models/detectors/correlation_head.py   # [자체 추가] CorrelationHead (커스텀)
    │   └── mmrotate/core/anchor/anchor_generator.py        # [자체 추가] RotatedAnchorGenerator_tracking 클래스만 커스텀
    └── train_swift_eye/                   # ★ Swift-Eye 자체 핵심 소스 전부 여기
        ├── swift_eye/                      # ★ 통합 추론 패키지 (최종 모델)
        │   ├── model.py                    # swift_eye nn.Module (detection+tracking+UNet 통합 상태머신)
        │   ├── model_config.py             # backbone/neck/2 heads/correlation/mask 설정
        │   ├── test_interpolated.py        # 추론 엔트리 (README 실행점)
        │   ├── utils.py                    # obb<->poly, obb->hbb, IoU(ellipse 기반)
        │   ├── assemble_model.ipynb        # 3개 학습 결과 가중치 조립 노트북
        │   └── unet/ (unet_model.py, unet_parts.py)  # 동공 세그멘테이션 UNet
        ├── train_backbone_and_neck/        # 1단계: Swin+FPN+RoITransformer 검출기 학습
        │   ├── train_backbone_and_neck.py  # GazeDataset 등록 + mmrotate train_detector
        │   └── swift_eye_config.py          # RoITransformer full config
        ├── train_with_temporal_fusion_component/      # 2단계: tracking(correlation) head 학습
        │   ├── train_with_temporal_fusion_component.py # 학습 루프
        │   ├── model.py                                # swift_eye_temporal_fusion_component
        │   ├── model_config.py                         # tracking_head + correlation_head 설정
        │   └── regress_classify_datasets_code/         # 시퀀스(template/search 쌍) 데이터셋
        │       ├── sequence_dataset.py, sequence_dataloader.py
        │       └── pipelines/ (loading, transforms, formatting, collate, data_container)
        └── train_without_temporal_fusion_component/   # 비교군: detection head만 단일 프레임 학습
            └── (동일 구조, dataset이 단일 이미지)
```

**제외 근거**: `mmrotate/mmrotate/`, `mmrotate/configs/`(s2anet, redet, roi_trans 등 DOTA 표준), `mmrotate/tests/`는 MMRotate 0.3.4 원본(requirements.txt:98 `mmrotate=0.3.4=dev_0`). 단, `correlation_head.py`와 `anchor_generator.py`의 `RotatedAnchorGenerator_tracking`은 Swift-Eye가 원본 트리에 끼워넣은 커스텀이므로 분석 포함.

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.1 통합 추론 모델 `swift_eye` (swift_eye/model.py:45-360)
세 갈래(검출/추적/보간)를 **open_extent(눈 뜬 정도)** 로 전환하는 상태머신(state machine).

**구성요소 빌드**(model.py:49-55):
- `self.backbone = MODELS.build(cfg.backbone)` → SwinTransformer
- `self.neck = MODELS.build(cfg.neck)` → FPN
- `self.tracking_head` / `self.detection_head` → 둘 다 `RotatedRetinaHead`(model_config.py:3-52, 127-176)
- `self.correlation_head` → 커스텀 `CorrelationHead`(model.py:54)
- `self.unet = UNet(n_channels=1, n_classes=2, bilinear=False)`(model.py:55) → 동공 세그멘테이션

**상태머신 임계값**(model.py:67-70):
- `tracking_threshold = 0`
- `detection_threshold = 0.75`
- `template_update_threshold = 0.95`
- 모드: `"detection" | "tracking" | "interpolation"`

**좌표/특징 규약**: feature map 크기 `feat_h=72, feat_w=88`(model.py:62-63, FPN stride 4 기준 346/4≈88), search ROI `33`, template ROI `13`(model.py:64-65). `get_top_left()`(model.py:96-125)는 이전 예측 중심을 4로 나눠 feature 좌표로 변환 후 search/template ROI 좌상단을 경계 클램핑하여 산출.

**get_first_pred()**(model.py:127-157, 첫 프레임 검출):
1. test_pipeline → backbone → neck → `roi_features = features[0]`(FPN 최고해상도 레벨).
2. `detection_head_simple_test()`로 전체 feature 검출(model.py:141).
3. 동공 존재 시: UNet 마스크 획득 → `get_open_extent()`로 눈 뜬 정도 산출(model.py:145-147).
4. open_extent > 0.95이면 template feature(13×13 crop) 저장(model.py:150-153) → 추적용 커널.

**predict()**(model.py:160-243, 핵심 분기 로직, `@torch.no_grad`):
- 이전 예측 중심으로 search ROI(33×33) 추출(model.py:174), 전체 feature도 보관(model.py:175), UNet 마스크 계산(model.py:176).
- **모드 선택**(model.py:177-188):
  - `last_open_extent >= 0.75` → **detection** 모드 (전체 feature에 detection head)
  - `0 < last_open_extent < 0.75` → **tracking** 모드 (correlation + tracking head)
  - 그 외(눈 거의 감김) → **interpolation** 모드: 직전 rbbox를 그대로 신뢰도 0으로 복사(model.py:184-188).
- tracking 결과는 ROI 좌상단 오프셋을 더해 전역 좌표 복원(model.py:196-197 `pred_ep[:2] += top_left*4`).
- 사후 재검증 단계(model.py:215-240): open_extent가 임계 구간을 넘나들면 모드를 tracking→detection으로 승격 재시도. 즉 **눈 뜬 정도에 따라 보간→추적→검출로 점진 복귀**하는 휴리스틱.
- interpolation 중 마스크가 있으면 마스크 무게중심으로 중심 갱신(model.py:206-211).

**get_pred_masks()**(model.py:261-276): grayscale 1채널 → UNet → softmax → resize(260×346) → argmax → 동공 마스크(0/1). `unet_ransform`(model.py:71-75)은 ToPILImage→Resize(260,346)→ToTensor.

**get_open_extent()**(model.py:277-300): 예측 ellipse를 채운 마스크(cv2.ellipse, -1=fill)와 UNet 세그 마스크의 교집합 비율을 계산.
- 비-interpolation: `교집합 / ellipse면적`(model.py:293-294) — 검출된 ellipse가 실제 동공 마스크와 얼마나 겹치는가 = 가림(occlusion)/뜬 정도.
- interpolation: `세그면적 / ellipse면적`(model.py:296-297).

**detection_head_simple_test()**(model.py:306-325) / **tracking_head_simple_test()**(model.py:327-357): mmrotate `RotatedRetinaHead.get_bboxes` + `rbbox2result` 표준 호출. tracking은 먼저 `correlation_head(kernel=template, search=roi)`로 상관맵 생성 후 헤드 적용(model.py:344-349).

### 3.2 검출/추적 헤드 (model_config.py)
둘 다 **`RotatedRetinaHead`**(1-stage anchor-based 회전 검출기):
- num_classes=1(pupil), in_channels=256, stacked_convs=4(model_config.py:5-8).
- **anchor**: scales=[6,8,10], ratios=[1.0,0.5,2.0], strides=[4] (model_config.py:10-14). detection은 `RotatedAnchorGenerator`, tracking은 커스텀 `RotatedAnchorGenerator_tracking`(model_config.py:11 vs 135).
- **bbox_coder**: `DeltaXYWHAOBBoxCoder`, angle_range='le90', edge_swap=True, proj_xy=True(model_config.py:15-22).
- **loss**: cls=`FocalLoss`(γ=2.0, α=0.25), bbox=`L1Loss`(model_config.py:23-29).
- **assigner**: `MaxIoUAssigner`(pos 0.5/neg 0.4) + `RBboxOverlaps2D`(model_config.py:31-37). sampler `RandomSampler`(num=64, pos_fraction=0.25).
- **test_cfg**: nms_pre=2000, score_thr=0.05, nms iou_thr=0.5(model_config.py:46-51).

### 3.3 CorrelationHead (커스텀, mmrotate/mmrotate/models/detectors/correlation_head.py:11-83)
SiamRPN++(arXiv:1812.11703)의 **depthwise cross-correlation**을 그대로 채택(correlation_head.py:14-16 docstring).
- `kernel_convs`, `search_convs`: 각각 ConvModule(3×3, BN, ReLU)(correlation_head.py:41-53).
- `depthwise_correlation()`(correlation_head.py:56-75): kernel을 그룹 컨볼루션 필터로, search를 입력으로 `F.conv2d(groups=batch*channel)` → 채널별 상관맵. 출력 `H_o = H_x - H_k + 1`.
- `forward(kernel, search)`: template(13×13) feature를 커널, search(33×33)를 입력으로 상관맵(약 25×25) 생성(correlation_head.py:78-82). 이 상관맵이 tracking head의 입력.
- **`@ROTATED_DETECTORS.register_module()`로 등록**(correlation_head.py:8-9) — 단 추론 시에는 `MODELS` 레지스트리로 빌드(model.py:54). 등록 레지스트리와 빌드 레지스트리 불일치는 mmrotate 내부 alias로 동작(추정).

### 3.4 RotatedAnchorGenerator_tracking (커스텀, anchor_generator.py:80-127)
표준 `AnchorGenerator`를 상속, search 영역(33×33×4=132px) 좌표계에 맞게 anchor 중심을 보정.
- `single_level_grid_priors()`(anchor_generator.py:88-127): super로 HBB anchor 생성 후 `[x,y,w,h,θ=0]` 회전 anchor로 변환(anchor_generator.py:113-117).
- 좌표 원점 보정(anchor_generator.py:118-122): `-10*4`(template 절반 오프셋) 후 `+33*4/2`(search 중심 정렬). 주석: "scaled feature map과 searched image의 중심을 일치"(anchor_generator.py:122-123). 즉 correlation 출력 크기 축소를 보정하는 핵심 커스텀.

### 3.5 UNet 세그멘테이션 (swift_eye/unet/)
표준 UNet(unet_model.py:6-48): n_channels=1, n_classes=2(동공/배경), bilinear=False(ConvTranspose 업샘플).
- inc(1→64), down1~4(64→1024), up1~4, outc(64→2)(unet_model.py:13-23).
- parts(unet_parts.py): DoubleConv(Conv-BN-ReLU ×2), Down(MaxPool+DoubleConv), Up(ConvTranspose+concat+DoubleConv), OutConv(1×1).
- 역할: open_extent 계산용 동공 픽셀 마스크 제공. blink(가림) 판정의 근거(occlusion-ratio estimator에 해당, README.md:36).

### 3.6 학습 루프 (3단계 분리 학습)
**1단계 backbone+neck (train_backbone_and_neck.py)**: mmrotate 표준 `train_detector`로 **RoITransformer(2-stage)** 검출기 전체를 학습(swift_eye_config.py:114-289). Swin-T 백본 + FPN + RotatedRPNHead + RoITransRoIHead(2 stage). `GazeDataset(DOTADataset)`로 등록(train_backbone_and_neck.py:28-31). optimizer AdamW lr=1e-4 wd=0.05(swift_eye_config.py:83-92), 30 epoch(swift_eye_config.py:100). 즉 **검출 backbone/neck는 RoITransformer로 사전학습 → 이후 RetinaHead 검출/추적 헤드로 전이**.

**2단계 tracking head (train_with_temporal_fusion_component/)**:
- `swift_eye_temporal_fusion_component`(model.py:33-230): backbone/neck **freeze**(requires_grad=False, model.py:42-45), correlation_head + tracking_head만 학습.
- `forward(search, kernel, ...)`(model.py:110-118): correlation → tracking_head.forward_train → loss.
- `get_corresponding_feature()`(train_*.py:141-211): GT bbox 주변에서 template(13×13)/search(33×33) feature를 **랜덤 시프트(jitter)** 하며 크롭(train_*.py:176, 198 `torch.randint`) → 추적 강건성 증강. obb 좌표도 크롭 오프셋만큼 재계산(train_*.py:207-210).
- 학습 데이터: `df['template_path']`(origin_poly) + `df['search_path']`(occlusion_poly) **쌍**(sequence_dataset.py:101-102). 즉 **가림이 없는 template과 가림이 있는 search 프레임 쌍**으로 추적기를 학습 = anti-blink의 핵심.
- optimizer Adam lr=1e-4(train_*.py:264, set_args), 30 epoch, val loss 최소 시 저장(train_*.py:297-299).

**비교군 without_temporal_fusion (train_without_temporal_fusion_component/)**:
- `swift_eye_without_temporal_fusion_component`(model.py:33-): **detection_head만** 단일 이미지로 학습(model.py:38). dataset도 단일 `df['image_path']`(poly)(sequence_dataset.py:102). 즉 temporal(template-search 상관) 없이 프레임 독립 검출.

### 3.7 데이터셋/파이프라인 (regress_classify_datasets_code/)
- `gazeSequenceDataset`(sequence_dataset.py:57-): CLASSES=("pupil",). df(pickle)에서 template/search 쌍 로드(sequence_dataset.py:90-104). `poly2obb_np`로 polygon→OBB 변환(sequence_dataset.py:122).
- **시퀀스 일관 증강**: `Compose.__call__`이 `seq_number`/`flip_direction`/`rotate_angle`를 받아 template(seq=0)에서 정한 flip/rotate를 search(seq=1)에 동일 적용(sequence_dataset.py:46-54, 177-187). 커스텀 `RRandomFlip`(transforms.py:564-, seq_number 분기 512-555), `PolyRandomRotate`(transforms.py:744-862, seq_number==0일 때 각도 결정 846-862) — **쌍 내 기하 변환 동기화**가 핵심 커스텀.
- collate: `collate_sequence`(partial, samples_per_gpu=2, sequence_dataset.py:82).

---

## 4. 알고리즘 / 데이터 표현

### 4.1 이벤트 표현
- **이벤트→프레임 보간**: 원시 이벤트 스트림은 timelens(README.md:28)로 보간되어 고FPS grayscale 프레임(`interpolated_frames/*.png`, test_interpolated.py:23)으로 변환된 뒤 입력. 코드 자체에는 voxel grid/time-surface/event-frame 누적 로직이 **없음**(저장소는 프레임화 이후 단계만 포함). "확인 불가" — voxel/time-surface 여부는 timelens 측 코드(외부)에 위임.
- 입력 정규화: mean/std = 81.49/38.74 단일값 3채널(model_config.py:113-116). grayscale 통계로 추정.

### 4.2 백본/넥/헤드/loss
- **백본**: SwinTransformer-Tiny(embed_dims=96, depths=[2,2,6,2], heads=[3,6,12,24], window=7)(model_config.py:62-78). → ConvLSTM/SSM/RNN 아님. **시간 정보는 Siamese cross-correlation으로 처리**(아래).
- **넥**: FPN(in=[96,192,384,768]→out=256, num_outs=5)(model_config.py:80-84). 추론은 최고해상도 레벨 features[0]만 사용(model.py:139, 174).
- **헤드**: RotatedRetinaHead ×2(검출/추적). FocalLoss(cls) + L1Loss(bbox).
- **마스크 헤드**: 추론 통합 모델은 `UNet`을 사용(model.py:55, 268). model_config.py:86-100의 `FCNMaskHead`+`RotatedSingleRoIExtractor`는 정의되어 있으나 추론 코드(`get_pred_masks`)는 UNet 경로를 사용 — `get_pred_masks_origin`(model.py:244-260)이 FCNMaskHead 경로지만 호출되지 않음(추정: 레거시/대체 경로).

### 4.3 시간 융합(temporal fusion) 메커니즘
- **ConvLSTM/SSM/Transformer-RNN 없음**(전체 train_swift_eye 검색 결과 LSTM/GRU/Mamba/SSM 키워드는 코드 본문에 미존재; 노트북 주석에서만 "temporal fusion component" 명명, assemble_model.ipynb:61).
- 실제 "temporal fusion"의 정체: **인접 프레임 간 Siamese depthwise cross-correlation 추적**(CorrelationHead). template(앞 프레임 동공 feature) ↔ search(현 프레임) 상관으로 시간적 연속성을 활용. blink로 검출이 끊기는 구간을 추적/보간으로 메우는 것이 anti-blink 핵심.
- **3-mode 상태머신**(검출↔추적↔보간)이 시간축 강건성의 주된 알고리즘(model.py:177-240).

### 4.4 회전 bbox → ellipse 후처리
- 회전 bbox `[x,y,w,h,θ]`(le90)를 **그대로 ellipse**로 해석: center=(x,y), 반축=(w/2,h/2), 회전각=θ(rad→deg)(test_interpolated.py:38, model.py:290, utils.py:184). 별도 최소자승 ellipse fitting은 **없음** — rotated bbox가 곧 동공 타원이라는 가정.
- 변환 유틸(utils.py): `obb2poly_le90`(9-34), `poly2obb_le90`(54-84), `obb2hbb_le90`(116-139), `hbb2xyxy`/`xyxy2hbb`(142-179).

### 4.5 메트릭
- **IoU(ellipse 기반)**: `calculate_iou`(utils.py:181-189) — 예측/GT ellipse를 260×346 마스크로 채워 교집합/합집합 픽셀 비율. mAP는 1단계 학습 평가에만 사용(swift_eye_config.py:80 `metric='mAP'`).
- p-error(동공 중심 픽셀 오차) 계산 코드는 저장소에 **미포함**("확인 불가").

---

## 5. 학습 / 평가

### 5.1 데이터셋
- 자체 GazeDataset(DOTADataset 상속, CLASSES=('pupil',))(train_backbone_and_neck.py:28-31).
- 출처: EV-Eye 기반 + timelens 보간(README.md:28). 학습 df(pickle)에 template/search/occlusion polygon 컬럼.
- 테스트 데이터/모델 가중치는 Google Drive 링크 제공(README.md:38, 42-43).

### 5.2 명령어 (README + 코드 기준)
- 설치(README.md:9-21): conda PyTorch 1.13.1 + cu11.6 → `mim install mmcv-full` → `mim install "mmdet<3.0.0"` → `cd mmrotate; pip install -v -e .`
- 추론 실행: `python /Swift-Eye/Swift-Eye/test_interpolated.py`(README.md:46; 실제 경로 `mmrotate/train_swift_eye/swift_eye/test_interpolated.py`). 결과는 `swift_eye/images_consequence/`에 ellipse가 그려진 프레임 저장(test_interpolated.py:13, 39).
- 학습 3단계: `train_backbone_and_neck.py` → `train_with_temporal_fusion_component.py` → `train_without_temporal_fusion_component.py`. 가중치 조립은 `assemble_model.ipynb`(temporal=epoch_29, w/o=epoch_24 로드, assemble_model.ipynb:61-78).

### 5.3 평가
- 1단계: mmrotate `evaluation=dict(interval=3, metric='mAP')`(swift_eye_config.py:80).
- 2단계: train/val loss(loss_bbox, loss_cls)만 추적, val loss 최소 epoch 저장(train_with_*.py:278-299).

---

## 6. 의존성
- PyTorch 1.13.0 + CUDA 11.7, torchvision 0.14.0(requirements.txt:142, 167).
- mmcv-full 1.7.2, mmdet 2.28.2, mmrotate 0.3.4(dev)(requirements.txt:96-98).
- opencv-python 4.8.1, numpy 1.24.3, pandas 2.0.3, scipy 1.10.1, e2cnn 0.2.3(ReDet 회전동변환 — 외부 mmrotate용)(requirements.txt:35, 104, 107, 117, 153).
- 단일 클래스(pupil) 회전검출 + UNet 세그 + Siamese 추적의 혼합 파이프라인. Python 3.8, conda 환경(requirements.txt:139).

---

## 7. 강점 / 한계

### 강점
- **3-mode 상태머신(검출/추적/보간)** 으로 blink 구간 강건성 확보 — 가림 시 추적·보간으로 끊김 방지(model.py:177-240).
- **occlusion-aware 학습**: 깨끗한 template ↔ 가려진 search 쌍으로 추적기 학습(sequence_dataset.py:101-102) → anti-blink 직접 최적화.
- 회전 bbox를 ellipse로 직결하여 동공 타원 추정 단순화. UNet open_extent로 모드 전환을 정량 제어(model.py:277-300).
- backbone/neck를 freeze하고 가벼운 correlation+head만 학습 → 학습 효율(model.py:42-45).

### 한계
- **무거운 백본(Swin-T)** + FPN + 두 RetinaHead + UNet 동시 추론 → 고FPS/저지연 on-device에는 부담. 실시간 5000FPS 데모는 사전 보간된 프레임 가정.
- 상태 전환 임계값(0.75/0.95 등)이 하드코딩(model.py:67-70) → 도메인 변화에 취약.
- ellipse fitting이 rotated-bbox=ellipse 단순 가정 → 비대칭/부분가림 동공에는 부정확 가능(추정).
- 이벤트→프레임 보간(timelens)이 외부 의존 → 진정한 이벤트-네이티브 처리(voxel/time-surface)는 미구현("확인 불가").
- 추론 코드가 단일 GPU 하드코딩(`CUDA_VISIBLE_DEVICES`), 경로 상대화 미흡 → 재현성/이식성 낮음.

---

## 8. 우리 프로젝트(PRJXR-HBTXR) 시사점 — "XR 시선추적 + FPGA 저지연 on-device 가속"(추정)

> 본 저장소는 GPU 학습/추론 PyTorch 코드로, FPGA/HW 가속 코드는 **미포함**(확인 불가). 아래는 우리 프로젝트 가속 관점의 활용 방향.

1. **알고리즘 단순화 여지(가속 친화)**:
   - 회전 bbox→ellipse 직결(test_interpolated.py:38)은 ellipse fitting HW IP 없이 곧바로 좌표 출력 가능 → FPGA 후처리 단순.
   - 상태머신 휴리스틱(model.py:177-240)은 제어로직(FSM)으로 RTL 이식이 자연스러움. 임계값 비교/모드전환은 경량 제어회로로 구현 가능.
   - depthwise cross-correlation(correlation_head.py:56-75)은 `F.conv2d(groups=...)`로 표현 → systolic array/PE 배열에 매핑 용이(소형 13×13 커널, 33×33 search → 연산량 작음, on-device 가속 1순위 후보).

2. **경량화/양자화 대상 분리**:
   - 무거운 Swin-T 백본+FPN은 사전계산/오프로드 또는 경량 CNN 대체 검토. Swift-Eye 구조상 backbone/neck은 freeze(model.py:42-45)되어 별도 가속 블록으로 분리 가능.
   - **correlation+tracking head(RetinaHead, in=256, 4 conv)** 가 프레임당 반복 호출되는 핵심 latency 경로 → INT8/INT4 PTQ/QAT 우선 적용 대상. anchor scales=[6,8,10] 단일 stride=4로 anchor 수가 적어 양자화 영향 분석 단순.
   - UNet(open_extent 산출)은 2-class 소형 세그 → 별도 가속 또는 임계연산 근사화 후보.

3. **FPGA 이식 로드맵(추정 제안)**:
   - (a) backbone/neck를 FP16/INT8로 호스트(또는 NPU) 처리 → feature를 FPGA로 전달.
   - (b) **correlation + RetinaHead + FSM(상태머신) + ellipse 변환**을 FPGA 저지연 파이프라인으로 묶어 프레임당 ms 이하 처리.
   - (c) 고정 해상도(72×88 feature, 33/13 ROI, model.py:62-65)·단일 클래스·소형 anchor는 HW 자원 산정이 결정적이라 DSE(파이프라인/병렬도 탐색)에 유리.
   - (d) timelens 보간을 이벤트-네이티브 누적(event frame/voxel) HW로 대체하면 진정한 on-device 저지연 달성 가능(연구 여지).

4. **참고로 가져올 자산**: ellipse-IoU 메트릭(utils.py:181-189), occlusion-aware template/search 쌍 학습 전략(sequence_dataset.py), open_extent 기반 모드 전환(model.py:277-300) — XR 시선추적의 blink 강건성 설계에 직접 차용 가능.

---

## 9. 근거 표기 요약
- **코드 라인 근거 확인 완료**: model.py(통합 추론·상태머신), model_config.py(아키텍처), correlation_head.py(Siamese 상관), anchor_generator.py:80-127(tracking anchor), unet_model.py/unet_parts.py(세그), train_*.py(3단계 학습), sequence_dataset.py/transforms.py(데이터·증강), utils.py(좌표변환·IoU), swift_eye_config.py(RoITransformer), requirements.txt(의존성).
- **"추정"**: timelens 보간 입력의 이벤트 표현 세부, FCNMaskHead 미사용 경위, correlation 레지스트리 alias 동작, rotated-bbox=ellipse 가정의 정확도 영향, FPGA 이식 제안 전반.
- **"확인 불가"**: 정량 메트릭(p-error/IoU 수치), 이벤트 voxel/time-surface 생성 코드(외부 timelens), FPGA/HW 가속 코드(저장소 미포함), 5000FPS 실시간 달성의 실제 latency 구성.
