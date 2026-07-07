# EX-Gaze, FACET, Swift-Eye 논문/코드 비교 분석

Role: 논문 분석가, 데이터 파이프라인 아키텍트, 딥러닝 학습 시스템 엔지니어

## 분석 범위

- 본 문서는 `references/` 아래 공개 논문과 공개 코드만 대상으로 한 정적 분석이다.
- `캐노니컬`은 두 가지를 함께 뜻한다.
  - 내부 표준 샘플 스펙: 입력 샘플, 라벨, 중간 표현, 타깃 구조
  - 좌표계/정규화: ellipse/OBB 정의, 해상도 변환, downsample 규칙, 정규화 규칙
- 근거가 없으면 채우지 않았고, 불명확한 부분은 `확실하지 않음`, `코드상 미노출`, `논문상 미기재`로 표기했다.

## 한눈 비교표

| 항목 | EX-Gaze | FACET | Swift-Eye |
| --- | --- | --- | --- |
| 데이터세트 구성방법 | 논문은 `EV-Eye + OpenEDS 합성 이벤트` 기반. 코드는 `EV-Eye` 세션 단위 JSON/HDF5 샘플 구조를 직접 사용하며, 공개 샘플 설정은 `user48/session201`로 축소됨 | 논문은 `EV-Eye`의 약 9k 마스크 라벨을 U-Net으로 1.5M 샘플까지 확장하고 그중 20k/5k/5k를 사용. 코드는 `DeanDataset/{train,val}/cached_*` 구조를 기대 | 논문은 `EV-Eye fully-open + LAMA 합성 가림 + 2400장 수동 평가셋 + Timelens 5000fps` 조합. 코드는 일부 학습 데이터/가중치를 외부 다운로드로 분리 |
| 캐노니컬 | `pre_gt -> pre_state -> 8개 event patch -> rotated bbox` 흐름. `le90` 회전 박스와 patch 기반 추적 | `event segment -> 2채널 256x256 event frame -> 64x64 dense heads + ellipse target` 흐름 | 단일 canonical이 아니라 3단계. 1단계는 `346x346 OBB detection`, 2/3단계는 `template/search pair + cropped feature` canonical |
| 메니페스트 | JSON annotation + HDF5 event representation + user/eye/session split config | 단일 manifest 파일 대신 `cached_data/*.memmap`, `cached_ellipse/*.memmap`, `*_indices.npy`가 사실상 manifest 역할 | 1단계는 MMRotate식 `annotations/ + images/`, 2/3단계는 `pickle DataFrame` manifest |
| 데이터 로더 | `WrappedEvPupilDataset(ConcatDataset)`가 세션별 dataset을 합침. MMEngine dataloader 사용 | 일반 `torch.utils.data.DataLoader`, Albumentations/tonic augment 적용 | 1단계는 MMRotate dataloader, 2/3단계는 custom `gazeSequenceDataset + GroupSampler + collate_batch` |
| 훈련 파이프라인 | MMEngine `Runner.from_cfg`. 최종 event tracker config는 80 epoch, random pre-state perturbation, warmup + cosine | Lightning `Trainer`. YAML 로딩 후 dataloader/model 생성, Adam + StepLR | 1단계 backbone/neck MMRotate 학습 후, 2단계 detection refinement, 3단계 temporal fusion refinement로 분리 |
| 손실함수 | 공개 tracker head는 `GDLoss(KLD)` 중심의 rotated bbox regression | heatmap focal + `ab/reg` L1 + trig L2 + GWD 계열 ellipse loss + mask loss | 1단계는 RPN/ROI의 CE + SmoothL1, 2/3단계는 `FocalLoss + L1Loss` |
| 모델 구조 | 시스템 논문은 `frame init/relocalization + event tracker + approach-adaptation + gaze regression`. 공개 코드는 주로 `EfficientTransVit` event tracker와 detector config를 제공 | `MobileNetV3 backbone + FPN-like fusion + 5-head ellipse predictor(heatmap/ab/trig/reg/mask)` | `Swin + FPN` 기반 detection backbone, 이후 `tracking_head + correlation_head`, 런타임 조립 시 `U-Net occlusion/open-extent estimator` 포함 |
| 입출력 차원 및 정의 | 입력은 실질적으로 `[B,8,2,16,16]` patch tensor, 타깃은 `le90` rotated bbox | 입력 `[B,2,256,256]`, 출력 dense maps는 `64x64`, 샘플 dict는 `input/hm/ab/trig/reg/mask/ellipse` 포함 | 1단계 입력 `346x346`, FPN 채널 `256`. 런타임 feature map `256x72x88`, template `13x13`, search `33x33` |
| 스케줄러 | 기본 schedule은 `ReduceOnPlateau + MultiStep`, 실제 주요 tracker config는 `LinearLR + CosineAnnealingLR`, optimizer는 AdamW | `StepLRScheduler(decay_t=10, decay_rate=0.7, warmup_t=5)` + Adam | 1단계만 `warmup + step` scheduler. 2/3단계는 Adam 고정 LR, 별도 scheduler 없음 |
| 런타임 | 논문은 Jetson Orin Nano에서 2kHz, event module 평균 0.32ms. 코드는 ONNX export와 `trtexec` 경로 제공 | 논문은 0.53ms inference, fast causal volume EPT 1.6493ms 보고. 코드는 직접 benchmark 스크립트가 주 경로로 노출되지는 않음 | 논문은 offline framework. 코드는 `test_interpolated.py`와 runtime state machine을 제공하지만 Timelens/LAMA/occlusion estimator 학습은 외부 의존 |

## EX-Gaze 상세

### 데이터 구성

- 논문 기준:
  - `DAVIS346` 하이브리드 event-frame camera를 사용해 `EV-Eye`를 평가 데이터로 사용한다.
  - `48 participants`, `4 sessions`가 언급되며, 연속 추적 평가를 위해 `19,270`장 이미지와 `384 continuous sequences`를 구성했다고 서술한다.
  - 추가로 `OpenEDS2020` 비디오를 보간/이벤트 합성해 VR 환경 일반화 평가에 사용한다.
- 코드 기준:
  - 공개 샘플 데이터는 `data/user48/left/session_2_0_1/` 아래에 `frames`, `events.npz`, `origin_landmark_tracking_dataset_ann.json`, `inter-2000_frame_endtime/...json` 등을 둔다.
  - 실제 학습 config는 `session_list=["201"]`, `eye_list=["left"]`로 샘플 설정이 좁혀져 있다.
  - `configs/_base_/data_split.py`에는 주석으로 `train_user_list = range(1, 37)`, `test_user_list = range(37, 49)`가 남아 있지만, 활성 값은 train/val/test 모두 `[48]`이다.
- 정리:
  - 논문은 full benchmark 관점의 데이터셋 구성이고, 공개 코드는 재현 가능한 축소 샘플과 세션 단위 파일 구조를 제공한다.
- paper-code gap:
  - 공개 repo의 기본 split은 논문 평가 split이 아니라 데모/샘플 실행용에 가깝다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/data/user48/left/session_2_0_1/events/`, `references/EX-Gaze-main/EX-Gaze-main/configs/_base_/data_split.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/dataset_config/pre_accum_pol_even_count_inter2000/patch_n8_s16/multi_max10_accum50_blink_exp5_ev_overlap_pupil_disp_with_rand_pre.py`

### 캐노니컬

- 내부 표준 샘플 스펙:
  - `EvEyeDataset`의 raw sample은 `pre_gt`, `cur_gt`, `events_between`, `img_shape`를 가진다.
  - dataset pipeline은 `cur_gt` ellipse를 rotated bbox target으로 바꾸고, `pre_gt.instances[0].ellipse`를 `pre_state`로 맵핑한다.
  - data preprocessor는 `pre_state = [cx, cy, w, h, t]`를 이용해 ellipse 둘레의 8개 점을 샘플링하고, 각 점 주변 patch를 잘라 `batch_inputs`로 변환한다.
- 좌표계/정규화:
  - 공개 코드의 회전 박스 표현은 `le90` 기반 `DeltaXYWHTRBBoxCoder`를 사용한다.
  - 주요 tracker config는 `patch_num=8`, `patch_size=16`, `in_channels=2`이며, 최종 입력 텐서는 `[B, 8, 2, 16, 16]`이다.
  - 학습 config에서 event patch 정규화 mean/std를 별도로 지정한다.
  - `bbox_head.ref_bbox_shape`는 최종 config에서 `[50, 50]`로 재설정된다.
- 정리:
  - EX-Gaze의 canonical은 "full image detector"가 아니라 "이전 ellipse 상태에 의해 조건부로 잘린 sparse event patch 집합"이다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/dataset/datasets.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/dataset_config/zhu_b8_u1_3/base_ev_pupil_displace.py`, `references/EX-Gaze-main/EX-Gaze-main/model/data_preprocessor/ev_pupil_patch_preprocessor.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/model_config/heads/displace_heads/trans_transformer_head.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre.py`

### 메니페스트

- 논문 기준:
  - 논문은 연속 추적 시퀀스와 OpenEDS 합성 데이터셋 구성을 설명하지만, 파일 수준 manifest 형식은 서술하지 않는다.
- 코드 기준:
  - 세션 단위 JSON annotation 파일과 HDF5 event representation 파일이 manifest 계층을 이룬다.
  - 대표 파일명은 `blink_seg_exp5_cont_frame_event_pre_accum_thr50_tracking_dataset.json`과 `event_accum_thr50_pol_event_count.hdf5`다.
  - `WrappedEvPupilDataset`는 `user/eye/session`을 순회하며 annotation path와 frame root를 결합해 `ConcatDataset`를 구성한다.
- 정리:
  - EX-Gaze manifest는 단일 중앙 테이블이 아니라, `split config + 세션별 JSON + 세션별 HDF5` 조합이다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/dataset/wrapped_dataset.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/dataset_config/pre_accum_pol_even_count_inter2000/patch_n8_s16/multi_max10_accum50_blink_exp5_ev_overlap_pupil_disp_with_rand_pre.py`

### 데이터로더

- 논문 기준:
  - 논문은 event patch 추적을 설명하지만 dataloader 구현 세부는 제시하지 않는다.
- 코드 기준:
  - MMEngine dataloader 설정을 사용한다.
  - `WrappedEvPupilDataset`가 여러 세션 dataset을 concat하고, pipeline 첫 단계에서 `LoadEventVolume`로 HDF5 event representation을 로딩한다.
  - `default_train.py`는 multiprocessing sharing strategy를 `file_system`으로 설정한다.
- 정리:
  - dataloader 수준에서는 "세션 concat + HDF5 volume load + sample packing"이 핵심이다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/dataset/wrapped_dataset.py`, `references/EX-Gaze-main/EX-Gaze-main/train/default_train.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/dataset_config/zhu_b8_u1_3/base_ev_pupil_displace.py`

### 훈련 파이프라인

- 논문 기준:
  - 시스템 전체는 `frame-based initialization/relocalization -> event-based pupil tracking -> approach-adaptation -> gaze regression` 순으로 설명된다.
- 코드 기준:
  - `default_train.py`가 config를 읽고 `Runner.from_cfg(cfg)`로 학습/평가를 실행한다.
  - 주요 tracker config는 이전 상태에 랜덤 perturbation을 주고, event patch 정규화, model init, optimizer/scheduler를 override한다.
  - config 파일명에 `pre_accum`, `event_count_inter2000`, `thr50`, `blink_exp5`가 들어 있어 데이터 생성/훈련 실험 조건을 파일명 레벨에서 관리한다.
- 정리:
  - 공개 코드의 학습 파이프라인은 전체 EX-Gaze 시스템 중 "event-based tracking module" 재현에 더 가깝다.
- paper-code gap:
  - 논문의 gaze regression 및 CPU/GPU scheduling은 공개 학습 경로에서 직접적인 훈련 모듈로 드러나지 않는다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/train/default_train.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre.py`

### 손실함수

- 논문 기준:
  - 시스템 수준으로는 tracking error estimation과 gaze regression까지 포함하지만, 공개 논문 본문에서 event tracker 내부 손실을 자세히 모두 나열하지는 않는다.
- 코드 기준:
  - 공개 tracker head는 `loss_decoded_bbox=True`와 `GDLoss(loss_type='kld', tau=1)`를 사용한다.
  - bbox coder는 `DeltaXYWHTRBBoxCoder`이며 rotated box regression이 핵심 supervision이다.
- 정리:
  - 공개 코드에서 확인되는 주 손실은 rotated ellipse/OBB 회귀 손실이다.
- paper-code gap:
  - 논문의 전체 시스템 목적함수와 공개 event tracker 코드의 손실 정의는 동일 범위가 아니다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/configs/model_config/heads/displace_heads/trans_transformer_head.py`

### 모델 구조

- 논문 기준:
  - EX-Gaze는 하이브리드 camera 기반 전체 gaze system이며, frame detector와 event tracker를 적응적으로 전환한다.
  - event-based tracker는 sparse rectangular event patch와 transformer를 사용한다.
- 코드 기준:
  - `EfficientTransVit`는 patch별 CNN encoder 후 transformer encoder를 통과시키고 patch 차원 평균을 취해 bbox head로 보낸다.
  - 현재 주요 config는 `cnn_encoder_config="CNNEncoderConfig_16_s1"`, transformer attention unit `48`, patch 수 `8`, bbox head hidden dim `48`이다.
- 정리:
  - 모델의 실질적 중심은 "local patch feature extractor + lightweight transformer + rotated bbox head" 조합이다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/model/detectors/efficient_trans_vit_v4.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/model_config/detectors/efficient_trans_v4/cnn_16_s1_transformer_f2_n4.py`

### 입출력 차원 및 정의

- 논문 기준:
  - 실시간 추적 주기는 `0.5ms`이며, 최근 `50`개 누적 이벤트의 시간폭이 `5ms` 이하일 때 event tracking을 수행한다.
  - 논문은 sparse event patches를 "typically 8"개 사용한다고 설명한다.
- 코드 기준:
  - 입력: `event_volume`와 `pre_state`.
  - 전처리 후 입력 텐서: `[B, 8, 2, 16, 16]`.
  - 출력: current pupil ellipse에 대응하는 `le90` rotated bbox 회귀 결과 `[cx, cy, w, h, angle]`.
- 정리:
  - 입력이 "전체 near-eye frame"이 아니라 "이전 상태로 조건부 샘플링된 patch tensor"라는 점이 가장 중요하다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/model/data_preprocessor/ev_pupil_patch_preprocessor.py`, `references/EX-Gaze-main/EX-Gaze-main/model/detectors/efficient_trans_vit_v4.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/model_config/heads/displace_heads/trans_transformer_head.py`

### 스케줄러

- 논문 기준:
  - 논문은 시스템 성능과 runtime을 중심으로 설명하며, 공개 코드 수준의 상세 scheduler는 논문 본문에서 핵심이 아니다.
- 코드 기준:
  - base schedule은 `ReduceOnPlateauLR + MultiStepLR`, optimizer는 Adam이다.
  - 실제 주요 tracker train config는 이를 override해 `LinearLR` warmup과 `CosineAnnealingLR`, optimizer `AdamW(lr=0.004/16, weight_decay=0.05)`를 사용한다.
  - epoch 수는 `80`으로 재설정된다.
- 정리:
  - EX-Gaze repo는 base schedule과 실험별 override를 분리하는 MMEngine 스타일이다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/configs/_base_/defualt_schedule.py`, `references/EX-Gaze-main/EX-Gaze-main/configs/train_config/eff_trans_vit_v4/in16_s1_f2_n4/patch_n8_s16/pre_accum/ev_pupil_dis_multi_max10_accum50_blink_exp5_overlap_pol_event_count_inter2000_with_rand_pre.py`

### 런타임

- 논문 기준:
  - `Jetson Orin Nano 8GB`에서 2kHz 실시간 tracking을 목표로 하고, 한 cycle이 `0.5ms` 이내여야 한다고 명시한다.
  - event-based module 평균 runtime은 `0.32ms`, frame-based relocalization은 `1.17ms`라고 보고한다.
  - CPU/GPU scheduling과 offloading이 시스템 설계의 일부다.
- 코드 기준:
  - README는 학습, end-to-end tracking, ONNX export, Jetson `trtexec` 테스트 절차를 제공한다.
  - 공개 repo는 deployment artifact 경로를 제공하지만, 논문과 동일한 full runtime scheduler를 코드상 한 파일로 노출하지는 않는다.
- 정리:
  - 논문은 embedded runtime system이고, 공개 코드는 그중 학습/추론/export 경로를 재현하는 연구 코드다.
- paper-code gap:
  - 논문 수준의 full scheduling/runtime orchestration은 공개 코드에서 부분적으로만 확인된다.
- 근거: 논문 `(EX-Gaze) High-frequency and Low-latency Gaze Tracking with Hybrid Event-frame Cameras for On-Device Extended Reality.pdf`; 코드 `references/EX-Gaze-main/EX-Gaze-main/README.md`

## FACET 상세

### 데이터 구성

- 논문 기준:
  - `EV-Eye`의 약 `9,000`개 마스크 라벨을 U-Net으로 전체 `1.5 million` 샘플까지 확장하고 ellipse로 fitting한다.
  - 이 중 `20,000`을 train, `5,000`을 val, `5,000`을 test로 랜덤 split한다.
- 코드 기준:
  - YAML은 dataset root를 `DeanDataset`으로 가리키며 `split=train/val`, `accumulate_mode=fixed_count`, `sensor_size=[346,260,2]`, `default_resolution=[256,256]`를 사용한다.
  - `DavisEyeEllipseDataset`는 `root/split/cached_data`, `root/split/cached_ellipse` 구조를 기대하고, 각 샘플에서 `5000` events를 읽는다.
- 정리:
  - 논문은 enhanced EV-Eye 구축 방법을 강조하고, 공개 코드는 이미 cache된 event/ellipse dataset 소비자 역할을 한다.
- paper-code gap:
  - 공개 코드에는 논문에서 설명한 `1.5M` 전체 라벨 확장 파이프라인과 정확한 `20k/5k/5k` split 생성 코드가 직접 묶여 있지 않다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/configs/DavisEyeEllipse_EPNet.yaml`, `references/FACET-main/FACET-main/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`, `references/FACET-main/FACET-main/EvEye/utils/cache/MemmapCacheStructedEvents.py`

### 캐노니컬

- 내부 표준 샘플 스펙:
  - `__getitem__` 반환값은 `input`, `hm`, `reg_mask`, `ind`, `ab`, `ang`, `trig`, `mask`, `reg`, `center`, `close`, `ellipse`다.
  - `close=1`은 closed eye 혹은 invalid ellipse를 뜻하고, 이 경우 유효한 ellipse supervision이 약화된다.
- 좌표계/정규화:
  - event segment는 2채널 frame으로 누적된 뒤 `256x256`으로 resize되고 `float32 / 255.0` 정규화된다.
  - `down_ratio=4`라서 target 공간은 `64x64`다.
  - ellipse는 `[x, y, a, b, angle]` 형식이며, angle은 `trig = [sin(2A), cos(2A)]`로도 encoding된다.
  - `mask`는 downsampled ellipse rasterization 결과다.
- 정리:
  - FACET의 canonical은 detection-style dense target과 ellipse geometry target이 함께 들어 있는 매우 명확한 supervision contract다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`

### 메니페스트

- 논문 기준:
  - 논문은 enhanced EV-Eye와 split만 설명하고 file manifest 포맷은 제시하지 않는다.
- 코드 기준:
  - `cached_data/events_batch_*.memmap`, `events_indices_*.npy`, `cached_ellipse/ellipses_batch_*.memmap`, `ellipses_indices_*.npy`가 manifest 역할을 한다.
  - cache utility는 원본 `.txt` event/ellipse를 읽어 batch별 memmap과 index 파일을 생성한다.
- 정리:
  - FACET의 manifest 계층은 JSON이나 CSV가 아니라 `memmap + info txt + npy index` 조합이다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/EvEye/utils/cache/MemmapCacheStructedEvents.py`

### 데이터로더

- 논문 기준:
  - 논문은 fixed-count binning, augmentation, fast causal volume을 설명한다.
- 코드 기준:
  - `dataset_factory.py`가 일반 PyTorch `DataLoader`를 생성한다.
  - YAML 기본값은 train/val 모두 `batch_size=2`, `num_workers=8`, `persistent_workers=true`다.
  - image augmentation은 `Resize`, `ShiftScaleRotate`, `HorizontalFlip`, event augmentation은 `DropEvent`, `DropEventByArea`다.
- 정리:
  - FACET dataloader는 dataset 내부에서 event-to-frame 변환과 target 생성을 완료한 뒤 일반 dict batch를 반환한다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/configs/DavisEyeEllipse_EPNet.yaml`, `references/FACET-main/FACET-main/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`, `references/FACET-main/FACET-main/EvEye/dataset/dataset_factory.py`

### 훈련 파이프라인

- 논문 기준:
  - 단일 GPU(`RTX3090`), batch size `32`, `70 epochs`, `Adam(lr=1e-3, wd=1e-5)`, warmup `5 epochs`, 이후 `10 epoch`마다 `0.7` decay라고 설명한다.
- 코드 기준:
  - `tools/train.py`가 config를 로드하고 dataloader/model을 만들며 optimizer config를 model에 주입한 뒤 `lightning.Trainer`로 학습한다.
  - Trainer는 `max_epochs=70`, `check_val_every_n_epoch=1`을 YAML에서 읽는다.
  - 공개 YAML의 batch size는 `2`이며, `devices=[2]`로 특정 GPU를 고정한다.
- 정리:
  - 학습 제어는 Lightning에 있고, 모델 내부 `configure_optimizers`가 실제 scheduler를 구성한다.
- paper-code gap:
  - 논문 batch size `32`와 공개 YAML batch size `2`는 다르다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/tools/train.py`, `references/FACET-main/FACET-main/configs/DavisEyeEllipse_EPNet.yaml`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`

### 손실함수

- 논문 기준:
  - 총 손실은 heatmap, offset, size, rotation, ellipse quality를 결합하며, 핵심 기여는 `Trigonometric Loss`다.
  - 논문은 angle discontinuity를 피하기 위해 `sin(2A), cos(2A)` 공간에서 회귀한다고 설명한다.
- 코드 기준:
  - `CtdetLoss`는 `FocalLoss`, `RegL1Loss(ab)`, `RegL1Loss(reg)`, `TrigL2Loss`, `GWD/GWDS` 계열 ellipse loss, mask loss를 결합한다.
  - `close == 0`일 때 유효 supervision이 적용된다.
  - YAML 기본 weight는 `hm=1`, `ab=0.1`, `trig=1`, `reg=0.1`, `iou=15`, `mask=1`이다.
- 정리:
  - FACET은 ellipse regression을 "center heatmap + geometry branch + mask regularization"으로 분해한 구조다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/Loss.py`, `references/FACET-main/FACET-main/configs/DavisEyeEllipse_EPNet.yaml`

### 모델 구조

- 논문 기준:
  - `MobileNetV3 backbone + FPN + 4 heads`로 설명되며, 최종 feature map `P2` 크기는 `(64, 64, 64)`다.
  - head는 heatmap, offset, size, rotation으로 설명된다.
- 코드 기준:
  - `EPNet`는 `MobileNetV3Backbone(input_channels=2)`와 `fpn_2d` fusion, `EPHead(in_channels=64)`를 사용한다.
  - 실제 head dict는 `hm:1`, `ab:2`, `trig:2`, `reg:2`, `mask:1`이어서 공개 코드는 논문 도식보다 mask head가 하나 더 있다.
  - Backbone 출력 채널은 `[24, 40, 112, 160]`, 최종 fused map은 64채널이다.
- 정리:
  - 논문 도식은 4-head지만, 공개 구현은 mask head까지 포함한 5-head predictor다.
- paper-code gap:
  - 코드 구현은 논문 설명보다 더 풍부한 supervision head를 공개하고 있다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/Backbone/MobileNetV3Backbone.py`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/Head/EPHead.py`

### 입출력 차원 및 정의

- 논문 기준:
  - 입력 events는 fixed count binning과 fast causal event volume을 거쳐 `256x256x2` representation이 된다.
  - 최종 feature map `P2`는 `(64,64,64)`이며 heatmap head는 `64x64` 중심 heatmap을 출력한다.
- 코드 기준:
  - 모델 입력: `[B, 2, 256, 256]`.
  - dense head 출력:
    - `hm`: `[B, 1, 64, 64]`
    - `ab`: `[B, 2, 64, 64]`
    - `trig`: `[B, 2, 64, 64]`
    - `reg`: `[B, 2, 64, 64]`
    - `mask`: `[B, 1, 64, 64]`
  - sample target dict:
    - `ind`, `reg_mask`: 최대 `100` object 슬롯
    - `ellipse`: downsampled ellipse `[x, y, a, b, angle]`
- 정리:
  - FACET은 dense map과 sparse index target을 동시에 사용하는 center-based ellipse detector다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/Head/EPHead.py`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/Predict.py`

### 스케줄러

- 논문 기준:
  - `Adam`, 초기 LR `1e-3`, WD `1e-5`, 첫 `5 epochs` warmup, 이후 `10 epochs`마다 `0.7` decay로 명시한다.
  - fast causal event volume의 limit `l=25`도 함께 언급한다.
- 코드 기준:
  - YAML optimizer는 `learning_rate=1e-3`, `weight_decay=1e-5`다.
  - `configure_optimizers()`는 `StepLRScheduler(decay_t=10, decay_rate=0.7, warmup_lr_init=1e-5, warmup_t=5)`를 사용한다.
- 정리:
  - FACET의 optimizer/scheduler는 논문과 코드가 가장 잘 정합되는 부분 중 하나다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/configs/DavisEyeEllipse_EPNet.yaml`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`

### 런타임

- 논문 기준:
  - abstract는 `0.53ms inference time`을 보고한다.
  - ablation에서는 fast causal event volume 사용 시 event processing time `1.6493ms`를 보고한다.
- 코드 기준:
  - 공개 코드의 주 경로는 학습과 예측이며, 논문 수치를 직접 재현하는 독립 benchmark entrypoint는 중심 경로로 노출되지 않는다.
  - `Predict.py`에는 event-to-frame, post-processing, FLOPs/latency 분석에 사용할 수 있는 함수가 있으나, 공식 README 수준의 runtime 스크립트는 제한적이다.
- 정리:
  - FACET은 논문상 매우 빠르지만, 공개 코드만으로 동일 수치를 즉시 재현하기 위한 완결 benchmark harness는 약하다.
- paper-code gap:
  - 논문은 `fast causal event volume(l=25)`를 전면에 내세우지만, 공개 YAML 기본값은 `events_interpolation: causal_linear_ori`다.
- 근거: 논문 `(FACET) Fast and Accurate Event-Based Eye Tracking Using Ellipse Modeling for Extended Reality.pdf`; 코드 `references/FACET-main/FACET-main/configs/DavisEyeEllipse_EPNet.yaml`, `references/FACET-main/FACET-main/EvEye/model/DavisEyeEllipse/EPNet/Predict.py`, `references/FACET-main/FACET-main/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`

## Swift-Eye 상세

### 데이터 구성

- 논문 기준:
  - `DAVIS346`의 저프레임 비디오(`25fps`)와 event stream을 `Timelens`로 `5000fps`로 보간한다.
  - `EV-Eye`의 `9,011` fully-open eye image를 기반으로 `LAMA`를 재학습해 부분 가림 이미지를 합성한다.
  - 별도 수동 라벨 평가셋 `2,400`장을 만들고, detector 학습은 10-fold split을 사용한다.
- 코드 기준:
  - README는 test dataset, `train_backbone_and_neck`, `train_with_temporal_fusion_component`, `train_without_temporal_fusion_component`, `train Occlusion-ratio estimator`를 모두 외부 다운로드 링크로 제공한다.
  - repo 내부에는 MMRotate 학습 코드와 sequence dataset loader는 있지만, Timelens/LAMA 자체 학습 코드는 포함되지 않는다.
- 정리:
  - Swift-Eye의 공개 repo는 self-contained dataset builder가 아니라, 논문 구성요소를 연결하는 runtime/training skeleton에 가깝다.
- paper-code gap:
  - Timelens fine-tuning과 LAMA synthesis는 논문 핵심이지만 공개 repo 내부에는 외부 의존으로 빠져 있다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/README.md`

### 캐노니컬

- 내부 표준 샘플 스펙:
  - Swift-Eye는 단일 canonical이 아니라 stage별 canonical이 다르다.
  - 1단계: full interpolated image + OBB label.
  - 2/3단계: `template image + search image` pair와 각 polygon/OBB label.
- 좌표계/정규화:
  - 모든 detection/tracking box는 `le90` oriented bbox로 통일된다.
  - sequence dataset은 polygon 8좌표를 `poly2obb_np(..., version='le90')`로 바꾼다.
  - runtime feature crop은 scale factor `4` 기준이며, image-space `52x52` 영역이 feature-space `13x13 template`, `132x132` 영역이 `33x33 search`에 대응한다.
  - runtime model은 full feature map 크기를 `feat_h=72`, `feat_w=88`로 가정한다.
- 정리:
  - Swift-Eye canonical의 핵심은 "이미지-기반 OBB detection"과 "특징맵 crop 기반 temporal fusion"이 분리되어 있다는 점이다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/model.py`

### 메니페스트

- 논문 기준:
  - 논문은 fully-open set, 합성 occlusion set, extra evaluation set을 설명하지만 file manifest 구조는 직접 명시하지 않는다.
- 코드 기준:
  - 1단계 backbone/neck 학습은 MMRotate/DOTA 스타일 `annotations`와 `images` root를 사용한다.
  - 2단계 detection refinement는 `detection_train.pickle`, `detection_validation.pickle`.
  - 3단계 temporal fusion refinement는 `tracking_train_dataset.pickle`, `tracking_validation_dataset.pickle`.
  - sequence dataset dataframe은 최소 `template_path`, `origin_poly`, `search_path`, `occlusion_poly`를 사용한다.
- 정리:
  - Swift-Eye manifest는 stage별로 완전히 다르며, 특히 refinement 단계는 pickle dataframe이 핵심이다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_without_temporal_fusion_component/train_without_temporal_fusion_component.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/train_with_temporal_fusion_component.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py`

### 데이터로더

- 논문 기준:
  - 논문은 adaptive template와 temporal fusion을 설명하지만 dataloader 세부는 직접 다루지 않는다.
- 코드 기준:
  - 1단계는 MMRotate `GazeDataset(DOTADataset)`를 사용한다.
  - 2/3단계는 `gazeSequenceDataset`가 template/search 두 샘플을 읽고 같은 flip/rotation augmentation을 공유한다.
  - `GroupSampler`, `collate_sequence`, `collate_batch`가 pair 데이터를 한 batch로 병합한다.
  - 결과적으로 model 입력 직전에는 template와 search가 합쳐진 `[2B, C, H, W]` 형태로 다뤄진다.
- 정리:
  - Swift-Eye dataloader의 차별점은 "pair-consistent augmentation + custom collate"다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataloader.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/pipelines/collate.py`

### 훈련 파이프라인

- 논문 기준:
  - 전체 framework는 `Timelens 보간 -> Swin/FPN spatial extraction -> temporal fusion + occlusion-aware adaptation -> interpolation` 순으로 설명된다.
- 코드 기준:
  - 1단계: `train_backbone_and_neck.py`가 MMRotate `RoITransformer`를 학습한다.
  - 2단계: `train_without_temporal_fusion_component.py`가 backbone+neck checkpoint를 로드하고 detection refinement를 별도 학습한다.
  - 3단계: `train_with_temporal_fusion_component.py`가 동일 checkpoint를 로드하고 temporal fusion component를 별도 학습한다.
  - 두 refinement 단계는 모두 `torch.optim.Adam(lr=1e-4)`, `30 epochs`의 수동 training loop다.
- 정리:
  - Swift-Eye는 end-to-end 단일 trainer가 아니라 3단계 분해형 학습 파이프라인이다.
- paper-code gap:
  - 논문은 하나의 일관된 framework로 설명하지만, 공개 코드는 실질적으로 stage-wise training recipes를 제공한다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_backbone_and_neck/train_backbone_and_neck.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_without_temporal_fusion_component/train_without_temporal_fusion_component.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/train_with_temporal_fusion_component.py`

### 손실함수

- 논문 기준:
  - 논문은 detector, temporal fusion, occlusion-aware adaptation의 결합 효과를 강조하지만 세부 loss 분해는 공개 코드만큼 상세하지 않다.
- 코드 기준:
  - 1단계 `RoITransformer`는 RPN/ROI head에서 `CrossEntropyLoss + SmoothL1Loss`를 사용한다.
  - 2단계/3단계 detection/tracking head는 `RotatedRetinaHead` 기반이며 `FocalLoss`와 `L1Loss`를 사용한다.
  - occlusion/open-extent estimator는 runtime에 U-Net으로 들어가지만 공개 학습 스크립트는 repo 내부에서 찾기 어렵다.
- 정리:
  - Swift-Eye의 공개 loss 설계는 detection/tracking head 중심이고, occlusion estimator loss는 외부 artifact에 의존한다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_without_temporal_fusion_component/model_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/model_config.py`

### 모델 구조

- 논문 기준:
  - Swin Transformer와 FPN으로 spatial feature를 만들고, adaptive template와 depth-wise cross correlation으로 temporal fusion을 한다.
  - occlusion-ratio estimator와 interpolation strategy까지 포함한 offline framework다.
- 코드 기준:
  - 1단계 config는 `RoITransformer + SwinTransformer + FPN`.
  - 2단계 config는 `detection_head(RotatedRetinaHead)` 중심 refinement.
  - 3단계 config는 `tracking_head + correlation_head`.
  - 통합 runtime 모델 `swift_eye`는 `backbone`, `neck`, `detection_head`, `tracking_head`, `correlation_head`, `UNet(n_channels=1, n_classes=2)`를 함께 갖는다.
- 정리:
  - 논문 개념도를 코드로 풀면 "MMRotate detector + correlation tracker + external occlusion estimator"의 조합이다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_without_temporal_fusion_component/model_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/model_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/model.py`

### 입출력 차원 및 정의

- 논문 기준:
  - 입력은 Timelens가 만든 고프레임 eye image이며, template은 image-space `52x52` pupil-centered crop에서 만들고 feature-space `13x13`으로 대응된다고 설명한다.
  - occlusion ratio에 따라 detection-only, temporal fusion, invalid/interpolation을 전환한다.
- 코드 기준:
  - 1단계 입력 image scale은 `(346, 346)`이다.
  - backbone/FPN 출력 채널은 `256`, 통합 runtime은 level-0 feature를 `1x256x72x88`로 사용한다.
  - runtime crop 크기:
    - `template_shape=13`
    - `search_shape=33`
    - `feat_h=72`, `feat_w=88`
  - sequence dataset은 polygon을 OBB `[x, y, w, h, a]`로 변환한다.
- 정리:
  - Swift-Eye는 최종적으로 full-image detector와 crop-based tracker가 같은 feature pyramid를 공유한다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/model.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/regress_classify_datasets_code/sequence_dataset.py`

### 스케줄러

- 논문 기준:
  - 논문은 threshold/ablation 중심이라 stage-wise training scheduler를 상세 공개하지 않는다.
- 코드 기준:
  - 1단계 backbone/neck 학습은 `AdamW(lr=1e-4, weight_decay=0.05)`와 `warmup + step([8,11])` scheduler를 사용한다.
  - 2단계/3단계 refinement는 `Adam(lr=1e-4)`만 사용하고 scheduler는 없다.
- 정리:
  - Swift-Eye의 training dynamics는 stage별로 크게 다르며, backbone 학습만 MMRotate 표준 scheduler를 사용한다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_backbone_and_neck/swift_eye_config.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_without_temporal_fusion_component/train_without_temporal_fusion_component.py`, `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/train_with_temporal_fusion_component/train_with_temporal_fusion_component.py`

### 런타임

- 논문 기준:
  - Swift-Eye는 low-latency online tracker가 아니라 `offline precise and robust` framework다.
  - occlusion ratio가 `<0.25`면 temporal fusion을 쓰지 않고, `0.25~0.875`면 full Swift-Eye, `>0.875`면 invalid로 보고 interpolation한다.
  - 고가림 구간에서 IoU와 F1이 각각 `20%`, `12.5%` 개선된다고 보고한다.
- 코드 기준:
  - `swift_eye/model.py`는 runtime mode를 `detection`, `tracking`, `interpolation`으로 관리한다.
  - 공개 코드 threshold는 `tracking_threshold=0`, `detection_threshold=0.75`, `template_update_threshold=0.95`다.
  - `test_interpolated.py`가 실행 entrypoint로 README에 언급된다.
- 정리:
  - 공개 runtime은 논문 개념을 구현하지만, occlusion threshold 정의가 논문과 문자 그대로 같지는 않다.
- paper-code gap:
  - 논문은 `0.25/0.875` occlusion-ratio 기반 규칙이고, 공개 runtime은 `open_extent` 기반 `0/0.75/0.95` 규칙을 사용한다.
- 근거: 논문 `(Swift-Eye) Towards Anti-blink Pupil Tracking for Precise and Robust High-Frequency Near-Eye Movement Analysis with Event Cameras.pdf`; 코드 `references/Swift-Eye-main/Swift-Eye-main/mmrotate/train_swift_eye/swift_eye/model.py`, `references/Swift-Eye-main/Swift-Eye-main/README.md`

## 공통점, 차이점, 재사용 가능 포인트

### 공통점

- 세 프로젝트 모두 pupil을 ellipse 또는 rotated box로 다루며, event/hybrid sensor의 시간 해상도를 적극 활용한다.
- 세 프로젝트 모두 "원시 이벤트를 그대로 넣지 않고" 중간 canonical representation을 만든다.
  - EX-Gaze: sparse local event patches
  - FACET: fixed-count event frame
  - Swift-Eye: interpolated image + OBB + cropped feature pair
- 세 프로젝트 모두 detector 구조만으로 끝나지 않고 상태 전이 또는 geometry prior를 암묵적으로 사용한다.

### 차이점

- EX-Gaze는 `이전 상태 조건부 local tracking`이 본질이다.
- FACET은 `단일 프레임 end-to-end ellipse detection`이 본질이다.
- Swift-Eye는 `offline interpolation + occlusion-aware temporal fusion`이 본질이다.
- manifest 관점에서도 성격이 다르다.
  - EX-Gaze: 세션별 JSON/HDF5
  - FACET: memmap/index cache
  - Swift-Eye: stage별 pickle/annotation root

### 재사용 가능 포인트

- EX-Gaze에서 가져올 포인트:
  - 이전 추정치를 이용해 입력 공간을 극단적으로 줄이는 patch canonical
  - local tracker를 위한 `pre_state -> patchify` 설계
- FACET에서 가져올 포인트:
  - canonical sample dict를 매우 명확하게 만드는 supervision contract
  - ellipse angle을 `sin(2A), cos(2A)`로 다루는 안정적 각도 표현
- Swift-Eye에서 가져올 포인트:
  - detector와 temporal refinement를 stage-wise로 분리하는 학습 전략
  - template/search pair dataloader와 pair-consistent augmentation 설계

## paper-code gap와 불확실성 목록

### EX-Gaze

- 논문은 full XR gaze system을 설명하지만, 공개 코드에서 직접 확인되는 주 경로는 event-based pupil tracking과 detector/export 경로다.
- `configs/_base_/data_split.py`의 주석 split과 활성 split이 다르므로, 공개 기본 config는 논문 평가 설정이 아니다.
- OpenEDS 합성 데이터 생성 전체 파이프라인은 논문에 비해 코드 노출이 제한적이다.

### FACET

- 논문은 fast causal event volume의 limit `l=25`를 전면에 내세우지만, 공개 YAML 기본값은 `causal_linear_ori`다.
- 논문 batch size `32`와 공개 YAML batch size `2`는 다르다.
- enhanced EV-Eye 전량 라벨링과 split 생성 절차는 논문 설명이 더 풍부하고, repo는 이미 cache된 데이터 소비에 초점이 있다.

### Swift-Eye

- Timelens fine-tuning, LAMA synthesis, occlusion estimator training은 논문 핵심이지만 공개 repo에서는 외부 다운로드/외부 코드 의존이 크다.
- 논문 threshold(`0.25`, `0.875`)와 공개 runtime threshold(`0`, `0.75`, `0.95`)는 직접 일치하지 않는다.
- 공개 repo는 offline runtime과 stage-wise training skeleton은 제공하지만, full self-contained reproduction package는 아니다.

### 공통 불확실성

- 논문에서 보고한 정확한 split 생성 스크립트, 실험 seed, full preprocessing artifact는 공개 코드만으로 모두 복원되지는 않는다.
- Swift-Eye와 EX-Gaze의 시스템 수준 runtime 수치는 논문에는 있으나, 공개 repo만으로 동일 수치를 즉시 재현할 수 있는 benchmark harness는 제한적이다.

## 진행 체크리스트

- [x] 세 프로젝트와 논문 위치 확인
- [x] 분석 형식 확정
- [x] `캐노니컬` 해석 범위 확정
- [x] 핵심 코드 근거 파일 식별
- [x] 논문 근거와 코드 근거를 항목별로 대조 정리
- [x] 비교표와 프로젝트별 상세 해설 작성
- [x] paper-code gap와 불확실성 표기 검수
