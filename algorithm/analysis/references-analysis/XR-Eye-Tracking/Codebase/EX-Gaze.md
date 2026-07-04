# EX-Gaze 코드베이스 정밀 분석

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\EX-Gaze`
> 분석 방법: 자체 핵심 소스(model / dataset / train / test / deploy / misc / configs)를 함수·클래스 단위로 라인 근거와 함께 정밀 분석. mmdet/mmrotate/mmcv/mmengine/torchvision 등 외부 프레임워크 원본과 `data/` 하위 대용량 샘플(npz/png/pickle/hdf5)은 제외.
> 근거 표기 규칙: 실제 코드 라인을 인용한 사실은 "확인", 코드에 직접 단언이 없어 합리적으로 추론한 부분은 "추정", 저장소 내에서 판단 불가한 부분은 "확인 불가"로 명시.

---

## 1. 개요

### 1.1 목적
EX-Gaze는 **이벤트 카메라(event camera) 기반 근안(near-eye) 동공(pupil) 추적** 시스템이다. 동공을 **회전 타원/회전 박스(rotated bbox, `[cx, cy, w, h, theta]`)** 로 표현하고, 두 개의 모델을 결합한 **프레임 검출 + 이벤트 추적 하이브리드 파이프라인**을 구성한다. (확인: `model/detectors/base_pupil_detector.py` L15-24 docstring "near eye pupil detection", `misc/ev_eye_dataset_utils.py` L26-35 `img_shape=(260,346)`, `task_name="ev pupil tracking"`)

- **프레임 기반 검출 모델**(저속, 정확): 그레이스케일 안구 이미지에서 RetinaNet 계열 회전 검출기로 동공 타원을 검출 → 추적 초기화/재초기화(re-localization)에 사용. (확인: `deploy/scripts/export.py` L21-26, `configs/.../full_eye_pupil_detector/...`)
- **이벤트 기반 추적 모델**(고속, 경량): 이전 동공 상태 주변에서 샘플링한 이벤트 패치들로부터 동공의 **변위(displacement)** 를 회귀하여 동공 상태를 갱신. (확인: `model/detectors/efficient_trans_vit_v4.py`, `model/heads/single_displacement_head.py`)

핵심 아이디어: 프레임은 25 fps로 느리지만 정확하고, 이벤트는 마이크로초 단위로 빠르다. **프레임 검출로 초기화한 뒤 이벤트 변위 추적으로 프레임 사이를 고속 보간**하고, 화면 유사도(similarity)가 임계 이하로 떨어지면 프레임으로 재검출하는 폐루프 구조다. (확인: `test/tracking_eval/end_to_end_tracking.py` L178-232 `frame_re_localization`, L262-328 `end_to_end_tracking`)

### 1.2 원논문 / 챌린지
- 저장소 자체에는 논문/챌린지를 직접 명시한 텍스트가 없다(README.md는 의존성·실행법만 기재). 따라서 원논문은 **확인 불가**. 다만 다음 근거로 추정한다.
  - 데이터셋은 **EV-Eye** 류로 추정. (근거: `misc/ev_eye_dataset_utils.py` L31-35 `"dataset_name": 'ev eye dataset'`, `origin_single_data_pattern`이 `user{id}/{eye}/session_*` 디렉토리 구조 사용, `session_col_idx = {'101','102','201','202'}` L38 — EV-Eye 데이터셋 구성과 일치 → **추정**)
  - 아키텍처는 **MobileViTv2의 separable self-attention** 과 **EfficientNetV2의 FusedMBConv** 를 차용. (확인: `model/blocks/seperable_self_attention.py` L17,177,322 docstring이 MobileViTv2 arxiv 2206.02680을 명시; `model/blocks/mbconv_v3.py` L135 `FusedMBConv`)
  - "EX-Gaze"라는 이름과 위 구성으로 보아 **이벤트 기반 저지연 XR 시선/동공 추적 논문의 공식/재현 구현**으로 추정. 정확한 출처는 **확인 불가**.

### 1.3 입력 / 출력
| 구분 | 프레임 검출 모델 | 이벤트 추적 모델 |
|---|---|---|
| 입력 | 그레이스케일 안구 프레임(eye_region 크롭 후 256×160 리사이즈), shape `(B,1,160,256)` (확인: `deploy/scripts/export.py` L22) | 이전 상태 주변에서 잘라낸 이벤트 패치 묶음 `(B, P=8, C=2, 16, 16)` + 이전 동공 상태 `pre_state` (확인: `export.py` L31, `model/data_preprocessor/ev_pupil_patch_preprocessor.py`) |
| 이벤트 표현 | - | **2채널 polarity event count 히스토그램**(pos/neg) (확인: `misc/event_representations/event_count.py` L7-20 `to_pol_event_count`) |
| 출력 | 동공 회전박스 `[cx,cy,w,h,theta]` + score (다중 객체 검출 후 max score 선택) (확인: `end_to_end_tracking.py` L115-124) | 동공 변위로부터 디코딩된 동공 회전박스(타원) `[cx,cy,w,h,theta]` (확인: `single_displacement_head.py` L14-29) |

> 세그멘테이션 헤드는 **없다**. 출력은 항상 회전 타원 파라미터(5-DoF)이며, IoU/F1 평가 시에만 `cv2.ellipse`로 마스크를 그려 비교한다. (확인: `misc/ev_eye_dataset_utils.py` L94-144 `ellipse_mask`/`ellipse_iou`)

---

## 2. 디렉토리 구조 (자체 코드 / 제외 대상 명시)

```
EX-Gaze/
├── README.md                         # 의존성 목록 + 실행 명령(확인: 논문/챌린지 언급 없음)
├── registry.py                       # EV_MODELS/EV_DATASETS/EV_TASK_UTILS 레지스트리 (자체, 미열람이나 전 모듈이 import)
├── model/                            # ★ 자체 핵심 (모델 정의)
│   ├── detectors/                    #   base_disp_detector / efficient_trans_vit_v4 / base_pupil_detector
│   ├── heads/                        #   single_detection_head / single_displacement_head / trans_transformer_head ...
│   ├── blocks/                       #   mbconv_v3(FusedMBConv) / seperable_self_attention(MobileViTv2 linear attn)
│   ├── backbones/                    #   mobilenet(MobileNetV3 wrapper)
│   ├── stems/ task_modules/          #   transformer_stem / displace_bbox_coder
│   ├── data_preprocessor/            #   ev_pupil_patch_preprocessor(타원 패치 샘플링)
│   └── img_pupil_similarity.py       #   shape_based_similarity (재초기화 판단)
├── dataset/                          # ★ 자체 (데이터셋/파이프라인)
│   ├── eye_pupil_dataset.py / eye_region_dataset.py / datasets.py
│   └── transforms/ (transform/formatting/utils)
├── train/default_train.py            # ★ 자체 (학습 엔트리; mmengine Runner)
├── test/                             # ★ 자체 (평가)
│   ├── default_test.py / test_utils.py
│   └── tracking_eval/                #   end_to_end_tracking(.py / _analysis / _model_cfg)
├── deploy/scripts/export.py          # ★ 자체 (ONNX export → Jetson/TensorRT). HLS/RTL 없음
├── misc/                             # ★ 자체 (이벤트 표현/데이터셋 유틸/전처리)
│   ├── event_representations/        #   event_count(폴라리티 히스토그램) / serialization
│   ├── ev_eye_dataset_utils.py       #   타원 샘플링·IoU·경로 패턴
│   └── gen_pre_accum_thres_dataset/, generate_event_threshold_dataset/
├── configs/                          # ★ 자체 (mmengine config; model/dataset/train/_base_)
├── result_plot/eye_tracking_video.py # 시각화
├── data/                             # ✗ 제외: 대용량 샘플(events.npz, frames/*.png 수백장, *.pickle, *.hdf5)
└── .git/                             # ✗ 제외
```

**제외 사유 요약**
- `data/` : 학습/추적 샘플 데이터(user48 한 명 분량의 frames PNG 수백 장, events.npz, end_to_end_tracking 결과 pickle). 코드가 아님 → 제외.
- 외부 프레임워크: `mmdet`/`mmrotate`/`mmcv`/`mmengine`/`torchvision`/`mmdeploy`는 pip 설치 의존성(README.md L14-18)이며 repo에 vendor되지 않음 → 분석 대상 외.
- **HW(HLS/RTL/Verilog) 코드는 저장소 내에 존재하지 않음**. `deploy/`에는 `scripts/export.py` 단일 파일만 존재(확인: Glob `deploy/**/*` 결과 1개). 하드웨어 가속은 **ONNX → TensorRT(trtexec) on Jetson** 경로로만 구현(확인: README.md L75-90).

---

## 3. 핵심 모듈 정밀 분석 (가장 중요)

### 3.1 이벤트 추적 백본: `EfficientTransVit` (`model/detectors/efficient_trans_vit_v4.py`)
이벤트 추적 모델의 본체. **"패치별 CNN 인코더 → 패치 시퀀스에 대한 선형 어텐션 트랜스포머 → 평균 풀링 → 변위 회귀 헤드"** 구조.

- **입력 형상**: `(B, P, C, H, W)` — P개 패치(기본 8), 각 패치 C=2(pos/neg), 16×16. (확인: L142-145 `extract_feat`의 `B,P,C,H,W = batch_inputs.shape`, `H==W==patch_size` assert)
- **CNNEncoder** (L57-107): 각 패치를 독립적으로 처리하기 위해 `(B*P, C, H, W)`로 reshape 후(L146) FusedMBConv 스택 통과 → `(B*P, D)`. 사용 구성은 `CNNEncoderConfig_16_s1`(L24-29): 입력 16×16에서 stride conv + FusedMBConv 3단 → 1×1, 출력 채널 48. (확인: 설정 파일 `cnn_16_s1_transformer_f2_n4.py` L7 `cnn_encoder_config="CNNEncoderConfig_16_s1"`, attn_unit_dim=48)
  - `pool_conv_feat=False`면 공간 차원을 flatten하여 `out_channels = C*h*w`로 잡고(L100), True면 AdaptiveAvgPool로 `D×1×1`. (확인: L96-100)
- **transformer_stage** (L132-137, 150-155): 패치 차원으로 reshape `(B, P, D)` 후 `LinearAttnEncoder` 적용. config에서 `patch_num==P` 및 `attn_unit_dim==conv_dim` 일치를 assert(L133-134). 출력 `(B, P, D)`를 **패치 축으로 평균**(`torch.mean(trans_feat, dim=1)`, L155)하여 `(B, D)` 단일 토큰 생성.
- **bbox_head** (L139-140): `hidden_dim == conv_dim(=48)` assert 후 헤드 빌드.

> 설계 함의(추정): 패치를 독립 CNN으로 인코딩 후 트랜스포머로 **패치 간 관계(동공 경계 둘레의 8방향)** 를 통합한다. 패치가 16×16로 작고 P=8로 적어 연산량이 매우 작다 → on-device 저지연 의도.

### 3.2 변위 추적기의 forward/loss 흐름: `BaseDispDetector` (`model/detectors/base_disp_detector.py`)
- mmdet `BaseDetector` 상속. `forward(batch_inputs, pre_state, data_samples, mode)`에서 mode별 분기(L12-25): `loss`/`predict`/`tensor`.
- `loss` (L27-30): `feat = extract_feat(...)` → `bbox_head.loss(feat, pre_state, ...)`.
- `predict` (L32-39): 동일 feat에서 `bbox_head.predict`, 결과를 datasample에 부착.
- `_forward` (L41-43): ONNX export용 순수 텐서 경로(`bbox_head(feat)` 직접 호출) — export.py가 이 경로를 탄다. (추정: `mode='tensor'`로 trace됨)

### 3.3 변위 회귀 헤드: `SingleDisplacementHead` / `TransTransformerHead`
**가장 핵심적인 추적 로직.** 절대 좌표가 아니라 **이전 상태(pre_state) 대비 변위(delta)** 를 예측한다.

- `SingleDisplacementHead.predict_bbox` (`single_displacement_head.py` L14-29):
  1. `displacement_preds = self.forward(x)` — 네트워크가 변위를 출력.
  2. `pre_encoded_box = bbox_coder.encode(decode_ref_bboxes, pre_state_boxes)` — 이전 상태를 기준박스 대비 인코딩.
  3. `box_preds = displacement_preds + pre_encoded_box` (L25) — **예측 변위를 이전 상태에 누적**.
  4. `decode_predict`로 디코딩(필요 시 크롭 복원). (확인: L27)
- `loss` (L31-46): GT 박스/가중치 추출 → ref/cropped bbox 파싱 → `predict_bbox(predict=False)` → `loss_bbox(box_preds.tensor, target_bboxes.tensor, weight=loss_weights)`. **가중 손실** 사용. (확인: L34-46)
- `predict` (L48-57): 동일 흐름, 결과를 `InstanceData(bboxes=...)`로 반환.

- `TransTransformerHead` (`model/heads/trans_transformer_head.py`):
  - `SingleDisplacementHead` 상속. `forward`(L63-69): (옵션) norm → (옵션 pool_dim) 평균 → `reg_bbox`(Linear, hidden_dim→encode_size). encode_size는 회전박스 5. (확인: L59 `self.reg_bbox = nn.Linear(hidden_dim, self.bbox_coder.encode_size)`)
  - `parse_ref_bboxes`(L71-88): `ref_bbox_shape`가 주어지면 기준 박스를 `[W/2, H/2, ref_w, ref_h, 0]`로 고정. 학습 config에서 `ref_bbox_shape=[50,50]`(확인: 학습 config L27).
  - `is_distil` 인자 존재(L24) → distillation 옵션을 염두에 둔 흔적이나 본 헤드 forward에서 직접 사용되진 않음. (확인 불가: distill 학습 경로 별도 존재 여부)

- 공통 베이스 `SingleDetectionHead` (`model/heads/single_detection_head.py`):
  - `_init_detection_cfg`(L117-154): 기본 손실 `GDLoss(loss_type='gwd')` (L132-133), 기본 bbox_coder `DeltaXYWHTRBBoxCoder(angle_version="le90")` (L134-142). → **회전 IoU 친화적 GWD(Gaussian Wasserstein Distance) 손실** 사용(확인). mmrotate의 GDLoss/coder를 빌드.
  - `get_target_bboxes`(L202-223): 샘플당 GT 1개만 허용(L213 `if target.shape[0]!=1: raise`), 가중치 cat. → **단일 동공 회귀 태스크**.
  - `decode_predict`(L225-237): `use_crop`이면 크롭 좌상단을 더해 원좌표 복원(L233-234).

### 3.4 프레임 검출 백본: `BasePupilDetector` + MobileNetV3 (`model/detectors/base_pupil_detector.py`, `model/backbones/mobilenet.py`)
- `BasePupilDetector` (L13-79): `stem → backbone → bbox_head` 단순 검출기. `extract_feat`(L74-79)에서 stem 입력 개수(1 또는 2)에 따라 분기 → backbone.
- `MobileNetBackbone` (`mobilenet.py` L17-86): torchvision의 `mobilenet_v3_small/large`/`v2` `.features`를 래핑. stem conv 첫 레이어를 그레이스케일(1ch) 입력용으로 교체(L69-76), 마지막 분류 헤드 제거(L77).
- `MobileNetBackbonePreX` (L89-106): **앞쪽 X개 레이어만 사용**(pre_x_layers). 검출 config에서 `v3_small`, `pre_x_layers=8`, 출력 48ch 사용(확인: `mbv3spreX_head_retina_img_pupil_detector.py` L7-14). → 백본을 잘라 경량화·다운샘플 x16 유지.
- 검출 헤드는 **RetinaNet 계열 anchor 기반 회전 검출**(확인: 설정 L21-29 `anchor_generator(scales=[1,1.5,2,3], base_sizes=[16], ratios=[1.0], strides=[16])`, assigner `pos_iou_thr=0.3/neg_iou_thr=0.1`). 헤드 구현 본체는 mmrotate 측에 위임(추정).

### 3.5 이벤트 패치 전처리: `EvPupilPatchPreprocessor` (`model/data_preprocessor/ev_pupil_patch_preprocessor.py`)
**이벤트 표현에서 패치를 추출하는 핵심 전처리.** 동공이 타원이라는 사전지식을 활용해 **타원 둘레 8방향에서 패치를 샘플링**한다.

- `__init__`(L16-43): `patch_size=16, patch_num=8`. `base_rad = 2π/patch_num`(L38), `sample_rads = base_rad * arange(patch_num)`(L39) → 8개 균등 각도. 입력 볼륨 가장자리 padding(L40 `pad_size=[8]*4`).
- `ellipse_patchify`(L45-79):
  - 각 샘플의 `pre_state(cx,cy,w,h,theta)`로부터 `ellipse_point_sample`(misc 유틸)로 8방향 타원 둘레 점 계산(L57).
  - 각 점 중심 16×16 영역을 crop하여 P개 패치 `(P, C, H, W)` 생성(L58-68).
  - 정규화 옵션(`mean/std`)(L77-78). 학습 config에서 `mean=[0.00118537,0.00101531], std=[0.03475869,0.03229738]` 적용(확인: 학습 config L16-18).
- `forward`(L81-105): `input_volume`(이벤트 폴라리티 히스토그램)와 `pre_state`를 받아 패치화하고 `batch_inputs`로 교체, 샘플 영역을 datasample에 기록.

### 3.6 선형 어텐션 트랜스포머: `LinearAttnEncoder` (`model/blocks/seperable_self_attention.py`)
- MobileViTv2의 **separable(linear) self-attention** 을 토큰 시퀀스에 적용한 인코더(L382-422).
- `LinearSelfAttention_v2`(L175-235): `(B,N,C)` 입력에서 qkv를 단일 Linear로 투영(L200-201, `out=1+2d`), query에 softmax(L223), context vector를 `bmm`으로 계산(L229) → **O(N) 선형 복잡도**. 표준 어텐션의 `O(N²)` 대비 경량.
- `LinearAttnFFN_v2`(L319-378): pre-norm + linear attn + FFN(Linear-act-Linear) residual 블록.
- `LinearAttnEncoder`(L381-422): 위치 임베딩(학습형 BERT식 또는 sinusoidal, L397-404) + N개 블록 + LayerNorm. config: `n_attn_blocks` f2_n4 명명상 패치/블록 수 의미(추정). attn_unit_dim=48.

> 함의(추정): O(N) 선형 어텐션 + 작은 패치/토큰(P=8) → 트랜스포머지만 연산·파라미터가 극히 작아 모바일/엣지에 적합. on-device 의도의 명확한 증거.

### 3.7 변위 코더: `DisplaceBBoxCoder` (`model/task_modules/displace_bbox_coder.py`)
- `encode`(L14-32): `displace = gt - pre` (단순 차분, L31).
- `decode`(L34-52): `pred = displace_pred + pre` (L47).
- → 추적을 **순수 좌표 변위 회귀**로 환원. 단, 회전박스 5-DoF용 mmrotate `DeltaXYWHTRBBoxCoder`(SingleDetectionHead 기본값)와 이 단순 코더 두 종류가 공존 → 설정에 따라 선택(추정: TransTransformerHead는 SingleDetectionHead 기본 coder 사용).

### 3.8 학습 루프 (`train/default_train.py`)
- mmengine `Runner.from_cfg(cfg)` → `runner.train()` → `runner.test()` (L29-32). 표준 mmengine 학습. `CUDA_VISIBLE_DEVICES='1'` 하드코딩(L5).
- 기본 config: `eff_trans_vit_v4/.../ev_pupil_dis_multi_max10_accum50_blink_exp5_..._rand_pre0.5.py` (L15).

### 3.9 추론/추적 루프 (`test/tracking_eval/end_to_end_tracking.py`) — ★ end-to-end 핵심
- **`BaseEndToEndTracker`** (L34-232): 프레임 검출 + 유사도 기반 재초기화.
  - `img_pupil_detect`(L129-163): eye_region 크롭→리사이즈→검출 모델→타원, 크롭 역변환으로 원좌표 복원(L157-162).
  - `frame_re_localization`(L178-193): 재초기화 필요 시 프레임 검출, 아니면 `shape_based_similarity`로 현 상태 검증(L191). 유사도 < 임계면 재초기화 플래그 set.
- **`EndToEndTracker`** (L235-328) / **`EndToEndTrackerWithPreAccum`** (L331-412): 이벤트 누적 추적.
  - 핵심 루프 `end_to_end_tracking`(L262-328): 프레임으로 초기화 → 이벤트 시간창 누적 → 동공 둘레 패치 영역(`parse_patch_region`)에 들어온 **이벤트 개수가 임계(`event_accum_num_threshold=50`) 이상**이면 `ev_single_pred`로 동공 상태 갱신(L293-303). 미달이면 시간창을 더 누적(`max_accum_frame_num=10`까지 슬라이딩). **이벤트 적응형 시간창**(밝기 변화/이동량에 따라 추론 빈도 자동 조절)이 특징. (확인: model_cfg L9-11)
  - WithPreAccum 버전(L349-412)은 누적 윈도우를 과거 방향으로 확장하며 임계 도달 시 1회 추론하고 break(L377-394). README가 권장(`--pre_accum_tracking`).
- `ev_single_pred`(`test/test_utils.py` L58-81): 이벤트 sub-stream → `to_pol_event_count`로 폴라리티 히스토그램 생성(L65) → `pre_state`와 함께 `model.test_step`(L75-79) → 예측 타원 반환.
- `build_eval_model`(test_utils.py L126-138): config로 모델 빌드 + checkpoint 로드 + eval().

### 3.10 평가 분석 (`test/tracking_eval/end_to_end_tracking_analysis.py`)
- 추적 결과 pickle을 GT와 시간 정렬(`argmin(|t_gt - t_pred|)`, L145)하여 **IoU / F1-score / pixel error(거리)** 계산(L150-156). `ellipse_iou`로 마스크 IoU·F1(L150), 중심 거리로 pixel error(L153).
- 사용자/눈/세션별·전체 통계 출력. pixel error ≥25/≥34 임계 기준 필터링 통계도 제공(L256-266). → **p-error(픽셀 오차)와 IoU가 주 메트릭**(확인).

---

## 4. 알고리즘 · 데이터 표현

### 4.1 이벤트 표현
- **2채널 polarity event count(히스토그램)** 가 본 추적 모델의 표준. `to_pol_event_count`(`event_count.py` L7-20): 시간창 내 이벤트를 pos/neg로 나눠 각각 `histogram2d`로 2D 누적 → `(2, H, W)`. (확인)
- 대안 표현도 구현: `abs_event_count`(1ch 누적), `pol_event_sum`(pos-neg, 1ch), `event_binary`(0/1) (L23-44). 선택은 `get_format_function`(L47-59).
- voxel grid나 time-surface는 사용하지 않음. **frame(누적 히스토그램) 표현**이며 시간정보는 "시간창 분할 + 적응형 누적"으로 간접 표현. (확인)

### 4.2 시퀀스 모델 종류
- **ConvLSTM / SSM 미사용**. 시간 의존성은 (1) 명시적 `pre_state` 변위 회귀(재귀적 상태 갱신)와 (2) 이벤트 적응형 누적으로 처리. (확인: 코드 전반에 RNN/LSTM/SSM 모듈 부재, displacement 코더 존재)
- **Transformer 사용** — 단, 시퀀스 차원이 "시간"이 아니라 "동공 둘레 8개 패치"다. MobileViTv2식 **선형(separable) self-attention** 으로 패치 간 공간 관계를 통합. (확인: §3.6)
- CNN: **FusedMBConv(EfficientNetV2)** 패치 인코더 + **MobileNetV3**(프레임 검출 백본). (확인: §3.1, §3.4)

### 4.3 후처리
- 프레임 검출: 다중 후보 중 max-score 선택(L115-124) → 크롭 역변환(rescale + translate, L159-162).
- 이벤트 추적: 변위 누적 후 회전박스 디코딩(`decode_predict`).
- 폐루프: `shape_based_similarity`(`model/img_pupil_similarity.py`)로 추적 품질을 모니터링, 임계 이하 시 프레임 재검출. (확인: end_to_end_tracking.py L191-228)

---

## 5. 학습 · 평가

### 5.1 데이터셋
- **EV-Eye 류 근안 이벤트-프레임 데이터셋**(추정, §1.2). 디렉토리: `user{id}/{eye}/session_{a}_{b}_{c}/events/{frames,events.npz,*.json}`. `img_shape=(260,346)`, frame_rate 25. (확인: dataset docstring `eye_pupil_dataset.py` L11-35, `ev_eye_dataset_utils.py` L26-38)
- 본 repo 동봉 샘플은 user48 / left / session 201 1종(확인: 데이터 Glob 결과 및 dataset config L14-16,29 `eye_list=["left"], session_list=["201"]`).
- 사용자 분할: `configs/_base_/data_split.py`의 train/val/test_user_list (확인: import 존재, 내용 미열람).
- 이벤트 추적 학습 입력은 사전계산된 패치/누적 데이터셋(hdf5/json): `blink_seg_exp5_..._thr50_tracking_dataset.json`, `event_accum_thr50_pol_event_count.hdf5` (확인: dataset config L7-8). blink(눈 깜빡임) 세그먼트 처리·임계 누적이 데이터셋 생성에 반영됨.
- `Dataset`: `EyePupilDataset`(mmengine `BaseDataset` 상속, `parse_data_info`로 eye_region/pupil 파싱, L62-87).

### 5.2 학습 설정 (이벤트 추적 모델)
(확인: `ev_pupil_dis_multi_max10_accum50_..._rand_pre.py`)
- Optimizer: **AdamW**, `base_lr = 0.004/16`, weight_decay 0.05, eps 1e-8 (L57-62).
- Scheduler: LinearLR warmup(0~4000 iter) + CosineAnnealingLR(40~80 epoch) (L39-54).
- max_epochs = 80 (L29).
- 데이터 증강: `random_dis_range=1.5/0.5`(pre_state 변위 섭동), `random_rot_range=0.3/0.1`, `random_scale_range=0.1/0` (L11-14, rand_pre0.5 변형은 0.5/0.1/0). → **이전 상태에 인위적 perturbation을 주어 변위 회귀 강건성 학습**(추정).
- Loss: `GDLoss(gwd)` (회전 박스용 Gaussian Wasserstein, §3.3).

### 5.3 메트릭
- **IoU**(타원 마스크), **F1-score**, **pixel error(중심 거리)**. (확인: analysis.py L150-156, `print_stats` L86-91 mean/std/max/min)

### 5.4 실행 명령 (확인: README.md)
```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH
# 학습
python train/default_train.py
# end-to-end 추적
python test/tracking_eval/end_to_end_tracking.py --device 0 --similarity_threshold 0.8 \
  --pre_accum_tracking --continuous_ann_track \
  --model_cfg_option config-eye_crop_mbv3spreX_multi_anchor_det-s1_f2_n4_trans-pre_accum10_50_blink_exp5_0.5_rand
# 결과 분석
python test/tracking_eval/end_to_end_tracking_analysis.py
# ONNX export → Jetson
python deploy/scripts/export.py
trtexec --onnx=ev_pupil_dis_...x2.onnx --useCudaGraph --useSpinWait
trtexec --onnx=mbv3spreX_head_retina_..._x2.onnx --useCudaGraph --useSpinWait
```
- 절대경로 수정 필요 위치: `test/tracking_eval/end_to_end_tracking_model_cfg.py`(체크포인트 경로 `/path/to/...` L4,8), `misc/ev_eye_dataset_utils.py`(데이터 루트 L14,19), `deploy/scripts/export.py`(가중치 L23,32). (확인: README L43-48)

---

## 6. 의존성
(확인: README.md L1-40, import 문)
- **PyTorch 1.13.1 + CUDA 11.7**, torchvision 0.14.1, Python 3.10.
- **OpenMMLab 스택**: mmcv 2.0.1, mmdet 3.1.0, mmengine 0.8.5, **mmrotate 1.0.0rc1(회전 검출 핵심)**, mmdeploy 1.3.1.
- 배포: onnx 1.13.1, onnxruntime/onnxsim/onnxoptimizer, **trtexec ≥8.6 (Jetson/TensorRT)**, thop(MACs/Params 프로파일, export.py L15-19).
- 기타: numpy, scipy, scikit-learn, pandas, opencv, h5py, matplotlib, tqdm.

---

## 7. 강점 · 한계

### 7.1 강점
1. **하이브리드 프레임+이벤트 폐루프**: 느린 정확 검출 + 빠른 경량 추적 + 유사도 재초기화로 정확도와 지연을 동시 관리. (확인: §3.9)
2. **극경량 추적 모델**: 16×16 패치 8개 + FusedMBConv + O(N) 선형 어텐션 → 파라미터·연산 최소화. on-device 명백히 지향. (확인: §3.1, §3.6)
3. **사전지식 활용 패치 샘플링**: 동공 타원 둘레 8방향만 처리해 입력 차원을 대폭 축소. (확인: §3.5)
4. **이벤트 적응형 추론 빈도**: 이벤트 개수 임계 기반 누적으로 정지 시 연산 절약, 급변 시 고속 반응. (확인: §3.9)
5. **배포 파이프라인 내장**: ONNX export + thop 프로파일 + TensorRT 명령까지 제공. (확인: §5.4)

### 7.2 한계
1. **HW(FPGA/HLS/RTL) 코드 부재**: 가속은 Jetson GPU + TensorRT에 국한. RTL/HLS 산출물 없음. (확인: §2)
2. **하드코딩·미완성 경로**: `/path/to/...` 다수, `CUDA_VISIBLE_DEVICES` 고정(train L5), 평가 엔트리의 user_list가 `[48]`로 고정(end_to_end_tracking.py L660). 재현 시 수작업 다수. (확인)
3. **단일 동공 가정**: 헤드가 샘플당 GT 1개만 허용(L213). 양안/다중 객체 일반화 제한.
4. **추적이 pre_state 품질에 의존**: 패치 샘플링이 직전 상태 기준이라 누적 드리프트/큰 점프(빠른 사케이드) 취약 → 유사도 재검출로 보완하나 프레임 의존. (추정)
5. **distillation/일부 분기 미사용**: `is_distil`, `LinearSelfAttention`(2D conv 버전) 등 사용되지 않는 변형 코드 잔존. (확인: §3.3, blocks 내 미사용 클래스)
6. **세그멘테이션 미지원**: 출력은 타원 5-DoF뿐. 픽셀 단위 동공 마스크가 필요한 응용에는 후처리 필요. (확인: §1.3)

---

## 8. 우리 프로젝트(XR 시선추적 + FPGA 저지연 on-device 가속) 시사점

> 전제: 본 repo는 GPU/Jetson 타깃까지만 구현. 우리 프로젝트 목표(FPGA 저지연 on-device)에 맞춰 **경량화·양자화·FPGA 이식** 관점으로 시사점을 정리. 이하 권고는 우리 목표에 대한 제언(추정)이며, repo가 직접 FPGA를 다루지는 않음(확인).

1. **연산 그래프가 FPGA 친화적이다.**
   - 추적 모델은 16×16 패치 × 8개라는 **고정·소형 입력**과 1×1/3×3 conv + 선형 어텐션으로 구성 → 시스톨릭 MAC 배열·라인버퍼 기반 HLS로 매핑 용이. (근거: §3.1, §3.6)
   - O(N) 어텐션은 softmax 1회 + 가중합으로, 대형 ViT 대비 FPGA에서 BRAM/DSP 압박이 작음. → **HLS C++ 변환 1차 후보**로 추적 모델 권고.

2. **양자화 적용 지점이 명확하다.**
   - FusedMBConv/Linear는 PTQ/QAT INT8 친화. SE/Hardsigmoid·SiLU 같은 비선형은 LUT 근사 필요. (근거: `mbconv_v3.py` SqueezeExcitation/Hardsigmoid L113-115, SiLU 활성)
   - 이미 ONNX export 경로(`_forward` 텐서 모드)가 있어 **ONNX→양자화→HLS/RTL** 흐름으로 연결 가능. (근거: §3.2, §5.4)

3. **프레임/이벤트 역할 분리를 HW에서도 유지하라(추정).**
   - 무거운 프레임 검출은 저빈도(재초기화 시에만)로 호출되므로 **소프트코어/저속 경로**, 고빈도 이벤트 추적은 **전용 가속기 데이터패스**로 분리하면 평균 전력·지연 최적. (근거: §3.9 적응형 호출 구조)

4. **이벤트 표현이 단순 히스토그램이라 전처리 HW가 가볍다.**
   - `to_pol_event_count`는 좌표 기반 2D 누적(histogram2d)뿐 → 스트리밍 이벤트를 BRAM 누적기로 직접 구현 가능. voxel/time-surface 대비 HW 비용 낮음. (근거: §4.1)

5. **적응형 누적 임계 로직은 컨트롤러로.**
   - 이벤트 개수 임계(50) 기반 추론 트리거는 간단한 카운터+비교기 FSM으로 구현 → "이벤트 충분히 모이면 1회 추론" 정책이 그대로 저전력 FPGA 컨트롤러가 됨. (근거: §3.9)

6. **재현·이식 시 주의:** mmrotate 의존 GWD 손실·회전박스 코더는 학습 측 의존성이며, **추론 그래프만 추출하면 mmrotate 없이도 HW 이식 가능**(forward는 Linear/Conv/attn 위주). 우리 파이프라인에서는 추론 그래프만 분리 권고. (추정)

---

## 9. 근거 표기 요약
- "확인" 표기: 인용한 파일·라인에 직접 근거가 있는 사실(아키텍처 구성, 입출력 형상, 손실/옵티마이저, 메트릭, 실행 명령, HW 코드 부재 등).
- "추정" 표기: 코드 정황상 합리적으로 도출했으나 직접 단언이 없는 부분(원논문·챌린지 정체, EV-Eye 데이터셋 동일성, 설계 의도, FPGA 이식 권고 등).
- "확인 불가" 표기: 저장소 내 정보만으로 판단 불가(정확한 원논문 출처, distillation 학습 경로 존재 여부, data_split 세부 내용).
- 외부 프레임워크(mmrotate의 GDLoss/anchor head/coder 내부 구현 등)는 본 repo 외부에 있어 라인 근거를 제시하지 않고 역할만 기술함.
