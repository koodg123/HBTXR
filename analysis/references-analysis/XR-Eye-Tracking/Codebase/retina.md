# Retina 코드베이스 정밀 분석 (XR-Eye-Tracking / Codebase / retina)

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\XR-Eye-Tracking\Codebase\retina`
> 분석 방법: README → 핵심 자체 소스(모델/loss/dataset/train·infer) 함수·클래스 단위 라인 근거 정밀 분석. 외부 프레임워크 원본(sinabs/tonic/dv_processing 등)은 import 경계까지만 확인하고 내부는 제외.
> 근거 표기 규칙: 코드에서 직접 확인한 사실은 `파일:라인` 인용. 코드로 단정 불가한 부분은 "추정" 또는 "확인 불가"로 명시.

---

## 1. 개요 (Overview)

### 1.1 목적
이벤트 카메라(event camera) 입력으로부터 **동공 중심 좌표(pupil center)를 회귀**하는 **저전력 시선/동공 추적(eye/pupil tracking)** 시스템. 일반 ANN, 양자화(INT8/이진), 그리고 뉴로모픽 칩(SynSense Speck/DYNAP-CNN) 배포용 SNN까지 하나의 코드베이스로 다룬다. (`README.md:1`, `engine\models\retina\retina.py`, `scripts\train.py`)

### 1.2 원논문 / 챌린지
- **Retina: Low-Power Eye Tracking with Event Camera and Spiking Hardware**, Bonazzi et al., **CVPR 2024 Workshops**, pp.5684-5692 (`README.md:38-45`). arXiv:2312.00425, IEEE 10678580.
- 소속: ETH Zurich + SynSense AG. 즉 알고리즘(ETH) + 뉴로모픽 HW(SynSense Speck) 협업. (`README.md:8-16`)
- 벤치마크: **Ini-30** 데이터셋의 "pupil detection"·"pupil tracking" SOTA 뱃지 (`README.md:2-3`). 비교 데이터셋으로 **3ET**(cb-convlstm-eyetracking) 사용 (`README.md:86-88`).

### 1.3 입력 / 출력
- **입력**: 이벤트 카메라 raw 이벤트(aedat4)를 이벤트 빈(event-count frame)으로 변환한 텐서.
  - 형상: `(num_bins, input_channel, img_width, img_height)` = 기본 `(1, 2, 64, 64)` (`configs\default.yaml:45-48`, `data\datasets\ini_30\ini_30_dataset.py:131`).
  - 채널 2 = ON/OFF 극성(polarity). Ini-30은 2채널, 그 외(3ET)는 1채널로 강제 (`scripts\train.py:75`).
- **출력 (arch에 따라 3가지)**:
  - `retina_ann`: 바운딩 박스 4좌표 `(x1,y1,x2,y2)` → 박스 중심이 동공 좌표 (`engine\models\utils.py:17-18`, `scripts\eval.py:28-30`).
  - `retina_snn`: YOLO 그리드 `S×S×(C+B*5)` 텐서 (`engine\models\utils.py:19-22`).
  - `3et` 베이스라인: 2D 좌표 `(x,y)` 직접 회귀 (`engine\models\utils.py:15-16`).
  - **세그멘테이션은 없음**. 출력은 동공 중심 좌표(점/박스)뿐. 라벨 CSV에 `axis_x/axis_y/angle`(타원) 컬럼 옵션이 있으나(`data\datasets\ini_30\ini_30_aeadat_processor.py:131-135`) 본 학습 파이프라인에서 ellipse fitting은 사용하지 않음(확인 불가 → 미사용). 따라서 "동공 좌표 회귀" 과제이며 "세그/타원 적합"은 미구현.

---

## 2. 디렉토리 구조 (자체 vs 제외)

```
retina/
├── README.md, requirements.txt, configs/default.yaml, .env.example   # 진입/설정
├── configs/
│   └── default.yaml                  [자체] 전체 하이퍼파라미터 (training/dataset/quant)
├── data/                             [자체] 데이터 파이프라인
│   ├── module.py                     LightningDataModule, 데이터셋 셀렉터
│   ├── utils.py                      yaml 로드, 통계(collect_statistics)
│   ├── transforms/
│   │   ├── transform.py              ★ FromPupilCenterToBoundingBox (라벨→박스/그리드)
│   │   └── helper.py                 tonic transform 파이프라인 조립
│   └── datasets/
│       ├── ini_30/                   ★ 메인 데이터셋(aedat4)
│       │   ├── ini_30_dataset.py     ★ 이벤트→프레임 변환(static/dynamic window)
│       │   ├── ini_30_aeadat_processor.py  dv_processing 래퍼(이벤트 슬라이싱)
│       │   └── helper.py             train/val 인덱스 분할, transform 주입
│       └── synthetic_3et/            3ET(h5) 비교 데이터셋
├── engine/                           [자체] 모델/학습 엔진
│   ├── module.py                     ★ LightningModule (train/val step, optim, loss)
│   ├── loss.py                       ★ BboxLoss / YoloLoss / Euclidian / Speck / IoU
│   ├── callbacks/logging.py          메트릭(distance/IoU)·GIF·wandb 로깅
│   └── models/
│       ├── retina/retina.py          ★ Retina 본체(config 기반 동적 nn.Sequential)
│       ├── retina/helper.py          ★ 레이어 구성표(64x64/128x128, ann/snn)
│       ├── baseline_3et.py           ConvLSTM 베이스라인
│       ├── utils.py                  ★ conv-bn fuse, DYNAP/N6 변환, synops 계산
│       ├── spiking/                  Decimation, LPF, residual, speck config
│       ├── binarization/            ★ DoReFa(이진) Conv/Linear
│       └── quantization/            ★ LSQ+ / DoReFa(저비트) 양자화
├── scripts/
│   ├── train.py                      ★ 학습 진입점(fire CLI) + ONNX export
│   └── eval.py                       ONNX(fp32/int8) 평가, Euclidean 분포
├── plots/                            시각화(GIF/애니메이션)
├── stm32ai_quantize_onnx_benchmark.ipynb   STM32 MCU 양자화/벤치 노트북
└── docs/*.gif                        결과 GIF
```

**제외 대상(외부/비핵심)**:
- `.git/` 일체.
- 외부 프레임워크(코드 미포함, import만): `sinabs`(SNN), `sinabs.exodus`, `tonic`(이벤트 변환), `dv_processing`(aedat4 디코딩), `pytorch_lightning`, `fire`, `wandb`, `ptflops/fvcore/thop`(FLOPs).
- `plots/`(시각화), `docs/`(GIF), `stm32ai_*.ipynb`(MCU 배포 노트북) — 본 분석에서는 보조로만 언급.
- `engine\models\quantization\*`의 LSQ/DoReFa 원본 알고리즘은 외부 양자화 라이브러리 이식분으로 보이며(추정), 본 분석에서는 진입점(`lsqplus_quantize_V2.prepare`) 호출 경계까지만 다룸.

---

## 3. 핵심 모듈 정밀 분석 (라인 근거)

### 3.1 Backbone — `Retina` (config-driven CNN) — `engine\models\retina\retina.py`
- **설계 철학**: 별도 backbone/neck/head 분리 대신, **레이어 딕셔너리 리스트를 순회하며 `nn.Sequential`을 동적 조립**한다 (`retina.py:57-194`). 동일 클래스가 ANN/SNN/이진 모두를 표현.
- 생성자 인자: `dataset_params, training_params, layers_config` (`retina.py:26`). 입력 해상도/채널은 `dataset_params`에서 받음 (`retina.py:30-33`).
- **SNN 활성 함수 분기**: `arch_name=="retina_snn"`일 때 sinabs 활성/리셋/대리기울기 구성 (`retina.py:39-54`). `spike_multi`면 `MultiSpike`+`PeriodicExponential`, 아니면 `SingleSpike`+`SingleExponential` (`retina.py:40-54`).
- **레이어 빌더(라인 근거)**:
  - `Conv`: bias 없는 `nn.Conv2d`, 출력 차원 수식으로 H/W 추적 (`retina.py:67-85`).
  - `DoReFaConv2d`/`DoReFaLinear`: 이진 연산자(아래 3.7) (`retina.py:87-114`).
  - `IAF`: `sinabs.layers.IAFSqueeze` — `num_timesteps=num_bins`, `tau_syn`, `spike_threshold`, `min_v_mem`, `record_states=True` (`retina.py:167-180`).
  - `Decimation`: 입력 이벤트 율 제어 spiking 전처리 (`retina.py:182-190`, 본체 3.5).
  - `SumPool`(sinabs) vs `AvgPool`(nn) — SNN은 SumPool, ANN은 AvgPool (`retina.py:139-159`).
  - `PredictionHead`: ANN 64x64의 헤드 (아래).
  - 미지원 레이어명은 `NotImplementedError` (`retina.py:191-192`).
- **forward**: 단순히 `self.seq(x)` (`retina.py:196-197`). 즉 시간 차원 처리(B,T 병합)는 상위 `EyeTrackingModelModule.forward`가 담당(3.3).
- **`__main__`**: ptflops/fvcore/thop 3종으로 MACs·Params 측정 코드 내장 (`retina.py:229-239`) → **경량 모델 복잡도 보고가 1급 관심사**임을 시사.

### 3.2 Head — `PredictionHead` — `engine\models\retina\retina.py:12-23`
- `Linear(hidden,128) → ReLU → Linear(128,4)` (`retina.py:15-19`). 출력 4 = 박스 4좌표.
- ANN 64x64 구성에서 마지막에 `PredictionHead` + `Sigmoid` 배치 (`helper.py:225-226`) → 출력 0~1 정규화 좌표.

### 3.3 학습/추론 루프 — `EyeTrackingModelModule` — `engine\module.py`
- PyTorch Lightning 모듈 (`module.py:11`).
- **loss 선택**: `3et→EuclidianLoss`, `retina_ann→BboxLoss`, `retina_snn→YoloLoss` (`module.py:44-49`).
- **SNN 전용**: LPF 온라인 필터(3.6), SpeckLoss(synops/firing 제약, 3.4) 구성 (`module.py:32-61`).
- **forward (시간축 처리)**: retina 계열은 `x.shape=(B,T,C,H,W)`를 `(-1,C,H,W)`로 병합 (`module.py:64-66`). SNN이면 `model.spiking_model(x)` 호출 (`module.py:67-68`).
- **training_step**: 배치 언팩 `data, labels, _, _` (`module.py:75`) → forward → (SNN이면 LPF) → `compute_loss` → `train_loss` 로깅 (`module.py:74-93`).
- **validation_step**: 동일 구조, `val_loss` 로깅 (`module.py:95-114`).
- **optimizer**: Adam/SGD 선택, SNN은 LPF의 `tau_mem/tau_syn/scale_factor`에 별도 lr 부여 (`module.py:116-141`). 스케줄러 `StepLR(step=1,gamma=0.8)` 또는 `ReduceLROnPlateau` (`module.py:134-137`).
- **compute_loss**: 각 loss dict 합산 → `total_loss = sum(values)` (`module.py:143-156`). SNN은 SpeckLoss를 추가 합산 (`module.py:151-152`).

### 3.4 Loss — `engine\loss.py`
- **IoU 함수**: `intersection_over_union` — corner 포맷 (x1,y1,x2,y2) 가정, `+1e-6` 안정화 (`loss.py:12-44`).
- **`EuclidianLoss`** (3ET용): `PairwiseDistance(p=2)` 평균 (`loss.py:47-67`). 동공 좌표 직접 거리.
- **`BboxLoss`** (retina_ann, 메인): (`loss.py:261-328`)
  - `square_results`: 예측 박스 중심을 구해 고정 폭 `bbox_w/img_width` 정사각형으로 재구성 (`loss.py:294-301`). 즉 **모델은 박스를 학습하지만 실질은 중심점 회귀 + 고정폭 박스**.
  - 손실 = `box_loss = 1 - IoU` + `distance_loss = 점 간 L2` (`loss.py:309-320`). 가중치 `w_box_loss=7.5`, `w_euclidian_loss=1` (`configs\default.yaml:31-32`).
- **`YoloLoss`** (retina_snn): YOLO v1식 (`loss.py:331-467`). 예측을 `(-1,S,S,C+B*5)`로 reshape (`loss.py:382`), `box_loss`(MSE)·`conf_loss`(MSE)·`distance_loss`(L2) (`loss.py:423-459`). B=2 분기로 best-box 선택 로직 포함 (`loss.py:396-415`).
- **`SpeckLoss`** (HW 제약): sinabs `SNNAnalyzer` 통계 기반. `synops/s`가 `synops_lim=[1e3,1e6]` 범위 밖이면 제곱 패널티(`loss.py:159-164`), 레이어별 `firing_rate`를 목표 선형 분포로 유도(`loss.py:166-172`), 입력이 임계 초과분에 `input_loss` (`loss.py:170-171`). **이것이 "뉴로모픽 HW 친화 학습"의 핵심** — 칩 자원/발화율을 loss로 직접 제약.
- 보조: `GaussianLoss`(미사용 추정), `FocalIoULoss`(미사용 추정), `encode_to_spike_pattern`(Poisson 스파이크 인코딩, 미사용 추정).

### 3.5 Dataset — `Ini30Dataset` — `data\datasets\ini_30\ini_30_dataset.py`
- aedat4에서 이벤트를 읽어 **이벤트-카운트 프레임**으로 변환하는 핵심 데이터로더.
- **라벨 로드** (`load_labels:73-93`): `annotations.csv`에서 `center_x/center_y` 읽고, 640x480→512x512 센터크롭 좌표 보정 (`:90-91`).
- **이벤트 로드** (`load_events:95-117`): `AedatProcessorLinear`로 이벤트 수집 → 좌표/타임스탬프/극성 분리 → 96~608 x범위 크롭 (`:107-111`) → `//= 512//img_width`로 다운샘플 (`:115`).
- **두 가지 윈도잉 방식**:
  - `load_static_window` (고정 시간창, `:119-202`): `fixed_window_dt`(기본 25000us, `default.yaml:53`) 기준 시간 슬라이스. 빈별 라벨은 타임스탬프 선형 보간 (`:153-167`).
  - `load_dynamic_window` (고정 이벤트수창, `:223-301`): **빈당 `events_per_frame=1000`개의 고유 이벤트가 모일 때까지** 역방향으로 슬라이스 (`:240-244`, `find_first_n_unique_pairs:204-221`). README는 이 동적 윈도가 `dv.Accumulator` 때문에 병목이라 명시 (`README.md:84`).
- **프레임 누적(이벤트 표현)**: `np.add.at(data[i,0], (xy[p==0,0], xy[p==0,1]), 1)`로 ON/OFF 극성별 히스토그램 누적 (`:177-182`), 두 채널 중 더 많은 극성만 남기는 마스킹 (`:183-188`), `clip(0,1)`로 이진화("no double events", `:190`). 최종 `rot90`+`permute`로 방향 정렬 (`:195-196`).
- **augmentation**: tonic의 `TimeJitter/UniformNoise/DropEvent`를 `__getitem__`에서 조건부 적용 (`:312-325`).
- 반환: `(event_tensor, labels_tensor, avg_dt, exp_id)` (`:346`). 1채널이면 `1 - event` 반전 (`:343-344`).
- **train/val 분할**: 30개 실험 중 `ini30_val_idx`(기본 `[1]`)만 val, 나머지 train (`helper.py:39-48`) → LOSO 유사 분할.

### 3.6 Data 표현 transform — `FromPupilCenterToBoundingBox` — `data\transforms\transform.py:47-113`
- **동공 중심점 → 학습 타깃 변환기**. arch별로 출력 형태가 갈림.
  - 좌표 정규화: ini-30은 `x/img_w, y/img_h` (`:78-79`).
  - `3et`: `[x_norm,y_norm]` 점 (`:83-84`).
  - `retina_ann`: 중심±`bbox_w/img` 정사각 박스 4좌표, `clip(0,1)` (`:87-94`).
  - `retina_snn`: `S×S×(C+5B)` 그리드에 obj conf=1과 박스좌표를 해당 셀에 기입 (`:96-109`).
- 입력 파이프라인은 tonic `Compose`로 `CenterCrop→Downscale→(decimate/denoise/flip/drop)→(MergePolarities)→ToFrame(n_event_bins=num_bins)` 조립 (`data\transforms\helper.py:28-51`).

### 3.7 양자화/이진화 — `engine\models\binarization\binary_operator.py`
- **`DoReFaConv2d`** (`:7-36`): 입력은 `sign` 이진화(STE: `binary - cliped.detach() + cliped`, `:24-26`), 가중치는 채널별 평균 스케일×`sign` (`:28-33`). XNOR-net/DoReFa 계열 1-bit conv.
- **`DoReFaLinear`** (`:87-104`): `BinaryQuantizer`(STE, grad는 |x|>1에서 0, `:71-84`)로 act/weight 이진화 + 가중치 스케일 (`:97-99`).
- 활성화 조건: `quant_params`에서 `a_bit==1 or w_bit==1`이면 레이어명을 DoReFa로 치환 (`engine\models\retina\helper.py:178-183`).
- 저비트(2~8bit)는 `lsqplus_quantize_V2.prepare`를 `train.py`에서 inplace 적용 (`scripts\train.py:133-143`).

### 3.8 학습 진입점 — `scripts\train.py`
- fire CLI `launch_fire` (`:30-178`). `.env`에서 데이터/출력 경로 로드 (`:23-28`).
- config 로드 → DataModule setup → 모델 생성(arch 분기) (`:86-130`).
- **SNN 경로**: ANN을 `sinabs.from_torch.from_model(..., synops=True)`로 SNN 변환 (`:109-113`), `verify_hardware_compatibility`면 `convert_to_dynap`로 Speck2f용 config 생성 (`:114-116`).
- **양자화**: 1<bit<32면 LSQ+ 적용 (`:133-143`).
- **Trainer**: PL `Trainer(accelerator="gpu")`, `LoggingCallback` 부착, `fit`+`validate` (`:156-168`).
- **배포 export**: `retina_ann`은 `convert_to_n6`(conv-bn fuse, `:170-171`) → **ONNX export(opset 11, dynamic batch)** (`:173-178`). README는 이후 onnxsim→onnx2tf→INT8 양자화 흐름 명시 (`README.md:114-120`).

---

## 4. 알고리즘 · 데이터 표현

### 4.1 이벤트 표현
- **방식**: voxel/time-surface가 아니라 **이벤트-카운트 2D 히스토그램(event-count frame)**. 극성별 2채널, 시간은 `num_bins`개 빈으로 분할 (`ini_30_dataset.py:131,177-190`; tonic `ToFrame(n_event_bins)` `transforms\helper.py:49-50`).
- **윈도 정의 2종**: 고정 시간창(`fixed_window_dt`) vs 고정 이벤트수창(`events_per_frame`) (`ini_30_dataset.py:334-338`). 기본은 동적(이벤트수) 윈도 (`default.yaml:52`).
- **이진화**: `clip(0,1)`로 픽셀당 단일 이벤트만 (`:190`).

### 4.2 시간 모델링 (ConvLSTM / SSM / Transformer)
- **Transformer 없음**(코드 전체에서 attention/transformer 모듈 미확인).
- **ConvLSTM**: `Baseline_3ET`에 직접 구현된 4단 ConvLSTM (`baseline_3et.py:19-269`). 이는 3ET 챌린지 비교 베이스라인이며 메인 모델 아님.
- **메인 Retina**: 시간축은 LSTM이 아니라 (a) 데이터단에서 빈으로 누적, (b) SNN의 IAF 막전위/LPF로 처리.
- **SSM 유사 요소(LPF)**: `LPFOnline` (`spiking\lpf.py:7-137`)은 시냅스·막전위 지수 커널을 1D conv로 적용하는 **저역통과 시간필터** — 상태공간(SSM)의 단순화 형태로 볼 수 있음(추정). SNN 출력의 시계열 디노이즈/적분 역할. 단, `forward`에 `pdb.set_trace()`가 남아 있어(`lpf.py:110`) **현재 그대로는 디버거에서 멈춤 → 미완성/주의** (4.6 한계).
- **Decimation**: 학습 불가 항등 1x1 conv + IAF로 입력 이벤트 율을 `decimation_rate` 임계로 솎아냄 (`spiking\decimation.py:6-47`).

### 4.3 후처리
- **ellipse fitting 없음**. 후처리는 단순히 **박스 중심 = 동공 좌표** 계산 (`scripts\eval.py:28-30`, `loss.py:313-314`). 평가도 중심점 Euclidean 거리.
- conv-bn fuse(`utils.py:193-213`)는 후처리라기보다 배포 최적화.

### 4.4 백본/넥/헤드 요약
- **백본**: 6단 Conv(+BN+ReLU/IAF) 스택. 64x64 ANN 기준 채널 16→64→16→16→8→16, 중간 AvgPool 2회 (`helper.py:185-227`).
- **넥**: 명시적 FPN/넥 없음. Flatten 후 FC.
- **헤드**: ANN은 `Linear(.,128)→ReLU→PredictionHead(.,4)→Sigmoid`; SNN은 `Linear→IAF`로 YOLO 그리드 출력 (`helper.py:108-114, 220-227`).

---

## 5. 학습 · 평가

### 5.1 데이터셋
- **Ini-30** (메인): aedat4 + annotations.csv, Zenodo 배포 (`README.md:70-82`). 30 실험, val 인덱스 1개 분리.
- **3ET** (비교): cb-convlstm-eyetracking의 h5/txt 포맷 (`README.md:86-88`, `synthetic_3et\synthetic_dataset.py`).

### 5.2 메트릭
- **거리(distance)**: 동공 중심 Euclidean(픽셀) — train/val 양쪽 로깅 (`callbacks\logging.py:91-103`). SNN은 LPF 커널 이후 구간만 평균 (`logging.py:101-102`).
- **IoU**: SNN val에서 박스 IoU (`logging.py:106-110`).
- **p-error**: 명칭 그대로의 "p10/p-error" 함수는 코드에 **명시적으로 없음**(확인 불가). 단 거리 분포 통계(`collect_statistics`, `eval.py:60-67`)로 유사 지표 산출 가능. → 논문상 p-error는 distance 분포에서 파생되는 것으로 추정.
- **MACs/Params**: ptflops/fvcore/thop 3종 (`retina.py:229-239`, `train.py:107-130`).
- **HW 통계**: synops, firing_rate (`loss.py:154-172`, `logging.py:56-73`).

### 5.3 실행 명령어 (README 근거)
- 학습: `python3 -m scripts.train --run_name="retina-ann"` (`README.md:102`).
- 배포(INT8): `onnxsim → onnx2tf → python3 -m scripts.quantize` (`README.md:116-120`). 단 `scripts\quantize.py`는 디렉토리에 **미존재**(확인 불가 → 누락/외부) — Glob 결과에 없음.
- 평가: `scripts\eval.py`가 fp32/int8 ONNX 거리 분포 비교 (`eval.py:45-79`).

### 5.4 설정 핵심값 (`configs\default.yaml`)
- arch_name=`retina_ann`, lr=1e-3, batch=32, Adam+StepLR (`:3-8`).
- 입력 64x64, 2채널, num_bins=1, events_per_frame=1000 (`:45-51`).
- loss 가중치 box=7.5, conf=7.5, euclidian=1, bbox_w=5 (`:31-37`).
- quant 기본 fp32(a/w=32) (`:69-70`).

---

## 6. 의존성 (`requirements.txt`)
- DL: `torch/torchvision`, `lightning`(PL), `tensorflow`(TFLite 배포).
- **이벤트/뉴로모픽**: `tonic`(이벤트 변환), `sinabs>=2.0`(SNN, Speck/DYNAP), `dv-processing`(aedat4).
- 배포/측정: `onnx/onnxruntime`, `thop`(+코드 내 ptflops/fvcore).
- 인프라: `fire`(CLI), `wandb`(로깅), `python-dotenv`, `pandas/tables`, `opencv-python`, `matplotlib`.
- 코드 추가 import(requirements 외): `ptflops`, `fvcore` (`retina.py:229,233`), `scipy`(median_filter). → requirements 불완전(경미).

---

## 7. 강점 · 한계

### 7.1 강점
- **단일 config로 ANN/SNN/이진/양자화/뉴로모픽 배포까지 일원화** (`retina.py:57-194`, `helper.py`). 레이어 딕셔너리만 바꿔 아키텍처 교체.
- **HW-aware 학습**: SpeckLoss로 synops·firing_rate를 loss에 직접 반영 (`loss.py:90-181`) → 칩 자원 제약을 학습에 내장.
- **명시적 복잡도 보고**: 3종 FLOPs 카운터로 경량성 정량화 (`retina.py:229-239`).
- **배포 파이프라인 완비**: conv-bn fuse → ONNX → TFLite INT8, DYNAP-CNN config 생성 (`utils.py:122-169`, `train.py:170-178`).
- **이벤트 윈도 2종**(시간/이벤트수)으로 데이터 표현 실험 유연 (`ini_30_dataset.py`).

### 7.2 한계 / 코드 리스크
- **LPF forward에 `pdb.set_trace()` 잔존** (`spiking\lpf.py:110`) → SNN+LPF 실행 시 디버거 멈춤. **버그/미완성**. (전반적으로 `pdb` import가 다수 파일에 잔존: retina.py, module.py, loss.py 등.)
- **`square_results`가 예측을 고정폭 박스로 덮어씀** (`loss.py:294-301`) → 박스 폭은 학습되지 않고 `bbox_w` 상수. 실질 task는 중심점 회귀에 가까움(의도된 단순화로 보이나 박스 IoU 의미 제한).
- **DataLoader 병목**: 동적 윈도 + `dv.Accumulator`가 느림(README 자인, `README.md:84`). 사전 저장 캐싱 미구현.
- **`scripts\quantize.py` 부재**: README가 호출하나 파일 없음(확인 불가) → 재현 시 작성 필요.
- **세그/타원 미지원**: 라벨에 타원 정보가 있으나 활용 안 함. 동공 형상이 필요한 응용엔 부족.
- **하드코딩 좌표 보정**(`min_x=96,max_x=608`, `+16` 등 `ini_30_dataset.py:59-62`)은 특정 센서/녹화 포맷 종속.

---

## 8. 우리 프로젝트 시사점 ("XR 시선추적 + FPGA 저지연 on-device 가속" 추정)

> 전제: 우리 프로젝트 방향은 README/경로 맥락상 "XR용 이벤트 기반 시선추적을 FPGA에서 저지연 on-device로 가속"으로 **추정**. 아래는 본 repo에서 직접 차용 가능한 요소.

1. **모델 경량성이 이미 검증됨**: 6단 작은 CNN(채널 8~64, 64x64 입력)으로 동공 추적. FPGA HLS/RTL 이식에 적합한 규모. `helper.py:185-227`의 64x64 ANN 구성을 **그대로 가속기 매핑 대상**으로 삼을 수 있음.
2. **양자화 자산 재사용**: DoReFa 1-bit conv/linear(`binary_operator.py`) + LSQ+ 저비트는 FPGA의 DSP/LUT 효율에 직결. **INT8/이진 가중치를 가정한 RTL MAC 어레이** 설계의 레퍼런스 정밀도 곡선을 `eval.py`(fp32 vs int8 거리 분포)로 확보 가능.
3. **conv-bn fuse**(`utils.py:193-213`)는 FPGA 추론 시 곱셈/덧셈 절감 — **HLS 전 단계 그래프 최적화로 필수 채택**.
4. **이벤트→프레임 변환을 HW로 이관**: `np.add.at` 히스토그램 누적(`ini_30_dataset.py:177-190`)은 단순 scatter-add — FPGA에서 이벤트 스트림을 직접 프레임 버퍼에 누적하는 전처리 IP로 구현하면 **DataLoader 병목(README:84)을 근본 해소**하고 저지연 달성. (우리 프로젝트의 차별화 포인트로 적합 — 추정)
5. **저지연 윈도 전략**: `events_per_frame` 고정 윈도(`load_dynamic_window`)는 활동량 적응형 — FPGA에서 이벤트 카운터 임계 도달 시 추론 트리거하는 **event-driven 파이프라인** 설계 근거.
6. **HW-aware loss 개념 차용**: SpeckLoss의 "자원/발화율을 loss로 제약"(`loss.py:90-181`)을 FPGA로 일반화하면 **리소스(LUT/DSP/BRAM)·지연 제약을 학습 목적함수에 반영하는 algo↔HW 공동최적화** 아이디어로 확장 가능(추정, 본 repo는 Speck 전용).
7. **ONNX가 이미 export 경로**(`train.py:173-178`): ONNX → HLS 프론트엔드(예: hls4ml/FINN)로 연결하면 FPGA 합성 흐름과 자연스럽게 접속.

**주의(차용 시)**: SNN 경로(IAF/LPF/Decimation)는 SynSense Speck(ASIC) 전용으로 설계됨 — FPGA로 가는 경우 **`retina_ann`(순수 CNN) 경로를 기반**으로 하고 SNN 요소는 선택적. LPF의 `pdb` 버그·`quantize.py` 부재 등 미완성 부분은 재구현 전제.

---

## 9. 근거 표기 / 미확인 항목 정리
- **확인됨(코드 직접 인용)**: 모델 동적 조립, 3종 loss, 이벤트 히스토그램 표현, 두 윈도 방식, DoReFa 이진화, conv-bn fuse, ONNX export, distance/IoU 메트릭, SpeckLoss.
- **추정**: LPF=SSM 유사 시간필터, p-error는 distance 분포 파생, SpeckLoss의 FPGA 일반화 가능성, 우리 프로젝트 방향(FPGA on-device).
- **확인 불가/미존재**: `scripts\quantize.py`(README 언급, 파일 없음), ellipse/세그 사용 여부(라벨 컬럼은 있으나 파이프라인 미사용), `GaussianLoss/FocalIoULoss/encode_to_spike_pattern` 실제 호출처(정의만 확인, 미사용 추정), Transformer/Attention(전무).
- **코드 품질 경고**: `lpf.py:110` `pdb.set_trace()` 잔존(실행 차단 버그), 다수 파일 `import pdb` 잔존, requirements에 ptflops/fvcore 누락.
