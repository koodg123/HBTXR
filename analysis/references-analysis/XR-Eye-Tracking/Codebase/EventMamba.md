# EventMamba 정밀 분석 (Point-based Event Camera Network with Mamba SSM)

> 분석 기준 경로: `REF/XR-Eye-Tracking/Codebase/EventMamba/`
> 분석 도구: Glob/Grep/Read (실제 코드 라인 근거). 추론은 "추정"/"확인 불가" 명기.
> 제외: `mamba-ssm`(외부 pip 패키지, Mamba 코어 커널), Pointnet2 CUDA 커널(FPS 가속, 외부).

---

## 1. 개요

- **목적**: 이벤트 카메라 데이터를 **포인트 클라우드(point cloud)**로 보고, PointNet++ 계열 계층적 다운샘플링 + **Mamba(SSM)** 기반 시퀀스 모델링을 결합해 분류·회귀를 효율적으로 수행.
- **원논문**: *Rethinking Efficient and Effective Point-based Networks for Event Camera Classification and Regression: EventMamba*, Ren et al., arXiv:2405.06116 (README L1). 이전 작업 TTPOINT(ACM MM'23), PEPNet(CVPR'24)의 확장(README L77).
- **태스크**: (a) 동작 인식(action recognition, `train_classification.py`), (b) 카메라 포즈 재측위(odometry, `train_odometry.py`), (c) **시선추적(eye tracking, `train_eye_tracking.py`)**.
- **시선추적 입출력**:
  - 입력 = **이벤트 포인트 클라우드** `[B, 3, N]` where 3 = (t, x, y), N=1024 (`train_eye_tracking.py` L38, `provider_data.load_h5_and_resample` L44). 즉 프레임/voxel이 아닌 **(t,x,y) 정규화 포인트 집합**.
  - 출력 = **동공 중심 좌표 (x, y)**, `num_category=2` (`train_eye_tracking.py` L33, classifier 최종 Linear→2). 좌표 회귀(분류 head를 회귀로 재사용).
  - 세그멘테이션 없음. 좌표 회귀만. 확인됨.
- **성능**(README L51): 3ET(+sigmoid) V1, dim[32,64,128], group[512,256,128] → accuracy 0.951.

---

## 2. 디렉토리 구조

```
EventMamba/
├── README.md
├── models/
│   ├── eventmamba_v1.py   ★ 3-stage, feature_list=[6,64,128,256], 시선추적 학습이 import (train_eye_tracking L166)
│   ├── eventmamba_v2.py   ★ 3-stage 일반화 버전 (분류용, feature_list 가변)
│   ├── mamba_layer.py     ★ MambaBlock (mamba_ssm.Mamba 래핑 + pre-norm residual)
│   └── modules.py         ★ LocalGrouper (FPS+kNN 그룹핑) + 포인트 유틸
├── train_classification.py   # 동작 인식 학습
├── train_odometry.py         # 포즈 재측위 학습
├── train_eye_tracking.py  ★ 시선추적 학습 (train/validate/main)
├── provider_data.py       ★ h5 로딩 + 리샘플링 (포인트 표현)
├── metrics.py             ★ p_acc / px_euclidean_dist / weighted_MSELoss
└── dataprocess/
    ├── generate_3et.py    ★ 3ET 이벤트→포인트 h5 변환 (frequency-adaptive sampling)
    ├── generate_{daily,action,dvsgesture,thu,ijrr}.py  # 각 데이터셋 전처리
    └── generate_by_{sliding_window,filename}.py
── mamba-ssm (외부) ── [제외] Mamba selective scan CUDA 커널
── pointnet2_ops (외부) ── [제외] FPS CUDA 가속 (modules.py L6, L127 주석처리)
```

★ = 정밀 분석 파일.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 포인트 그룹핑 — `models/modules.py`

PointNet++의 SA(Set Abstraction) 레이어를 PyTorch 순수 구현(CUDA FPS는 옵션).

- `square_distance` (L23-42): 두 포인트 집합 간 제곱 유클리드 거리 행렬 `[B,N,M]`.
- `index_points` (L44-60): 인덱스로 포인트 gather.
- **`furthest_point_sample` (L62-82)**: FPS(최원점 샘플링)로 npoint개 센트로이드 인덱스 선택. 순수 PyTorch 반복(L76-81) — **CUDA 가속(pointnet2_ops)은 L6, L127에서 주석으로 대체 가능**.
- `knn_point` (L84-95): topk로 k-최근접.
- **`LocalGrouper` (L97-151)** — 핵심:
  - 생성자(L98-119): groups(센트로이드 수), kneighbors(이웃 수), use_xyz, normalize("center"/"anchor"). normalize 시 `affine_alpha/beta`(L118-119) 학습 파라미터로 정규화 후 affine 변환.
  - forward(L121-151): FPS로 센트로이드 선택(L125) → **정렬**(L128, 시퀀스 순서 부여 = Mamba 입력 순서 결정) → kNN 그룹핑(`dists=square_distance(new_xyz, xyz)`, L133, **xyz 거리 사용**; feature 거리(L134)는 주석) → anchor 정규화(L141-149, std로 정규화 후 affine) → 센트로이드 피처와 concat(L150). 반환 `new_xyz[B,S,3], new_points[B,S,k,d]`.
  - **버전 차이**(README L66): v1은 `square_distance(new_xyz, xyz)`, v2 pretrained는 `square_distance(new_points, points)`(피처 거리). 이웃 선택 기준이 좌표냐 피처냐의 차이.

### 3.2 Mamba 블록 — `models/mamba_layer.py`

- `MambaBlock` (L9-84): 외부 `mamba_ssm.modules.mamba_simple.Mamba`를 mixer로 래핑(L28, L31). 구조는 **pre-norm + residual**: `residual ← residual + drop_path(hidden)`, `hidden ← norm(residual)`, `hidden ← mixer(hidden)`(L49-80). `bimamba_type='v2'`로 양방향 Mamba 사용(L28). fused_add_norm 옵션(RMSNorm/layer_norm_fn, L58-79).
- `allocate_inference_cache` (L83-84): 추론 시 상태 캐시(스트리밍 추론 지원) — Mamba 고유 기능.
- **SSM 상태 전이는 외부 mamba-ssm 커널 내부**(selective scan: `h_t = exp(Δ_t A) h_{t-1} + Δ_t B_t x_t`, `y_t = C_t h_t + D x_t`). 본 repo는 이를 호출만 함 → **커널 내부는 확인 불가(제외)**.

### 3.3 시선추적 모델 — `models/eventmamba_v1.py`

`train_eye_tracking.py`가 실제 import하는 모델(L166).

- **보조 레이어**:
  - `Attention` (L9-17): `Linear(hidden,1)` → softmax → 포인트별 attention 가중치(시퀀스 풀링용).
  - `Linear1Layer` (L19-30): `Conv1d+BN+ReLU` — 좌표 임베딩.
  - `Linear2Layer` (L32-50): residual bottleneck(`Conv1d C→C/2→C` + skip) — 피처 추상화.
- **`EventMamba` (L52-145)** — 3-stage 계층:
  - `feature_list=[6,64,128,256]`(L60), group 수 512→256→128(L62-64), 각 stage 24-NN.
  - 흐름(forward L91-140):
    1. `group`(FPS 512) → embed_dim(3?→64; 실제 6→64, L68) → conv1 → attention_1로 그룹 내 풀링(`bmm`, L102-103) → reshape `[b,n,d]` → **mamba1**(SSM) → conv1_1.
    2. `group_1`(256) → conv2 → attention_2 풀링 → **mamba2** → conv2_1.
    3. `group_2`(128) → conv3 → attention_3 풀링 → **mamba3** → conv3_1.
    4. 최종 attention_4로 전역 풀링(`bmm`, L138-139) → classifier(`Linear 256→BN→ReLU→Dropout0.5→Linear→num_classes`, L82-89) → 좌표 2개.
  - **구조 요약(backbone/neck/head)**: backbone = (LocalGrouper+conv+attention pooling+Mamba)×3 stage; neck = attention 기반 시퀀스 풀링; head = MLP classifier(회귀로 사용).
  - 주: classifier에 `nn.Sigmoid()`가 주석처리됨(L88) — README의 "3ET(+sigmoid)" 변형이 이 줄을 활성화한 것으로 **추정**.

### 3.4 분류 모델 — `models/eventmamba_v2.py`

- `EventMamba` (L48-114): v1과 동일 철학이나 `feature_list`, `group_number`를 데이터셋별로 주석 전환(L52-58), `local_grouper_list`/`mamba_list` 등을 `ModuleList`로 일반화(L63-81). stage=3, bimamba_type='v2'. forward(L90-114)는 v1과 동일 패턴(group→conv→attention pooling→mamba→global conv) 반복 후 최종 attention+classifier.
- fvcore FLOPs 측정 코드 포함(L116-122).

### 3.5 시선추적 학습 — `train_eye_tracking.py`

- **인자**(L19-41): batch_size=96, num_point=1024, sensor 640×480, spatial_factor=0.125(→입력 좌표 스케일 80×60), pixel_tolerances=[3,5,10,15], epoch=350, loss='weighted_mse', AdamW lr=1e-3 decay=1e-4, num_category=2.
- **`validate` (L49-79)** / **`train` (L81-113)**:
  - data permute `[B,N,3]→[B,3,N]`(L57, L88)로 모델 입력.
  - loss = criterion(outputs, `label[:,:2]`)(L60, L92) — 라벨 앞 2개(x,y)만 사용.
  - 메트릭: `p_acc`(pixel tolerance별 정답 수, width/height_scale = sensor·spatial_factor=80/60)(L62-65), `px_euclidean_dist`(평균 픽셀 오차)(L69-71).
- **`main` (L115-225)**:
  - 데이터: `provider_data.load_h5_and_resample(...sample_size=1024)`로 train/test 로드 → TensorDataset → DataLoader(L148-163).
  - 모델: `from models.eventmamba_v1 import EventMamba; EventMamba(num_classes=2)`(L166-167).
  - loss: `weighted_MSELoss(weights=(640/480, 1))`(L176) — x축 가중(종횡비 보정).
  - optimizer AdamW + MultiStepLR(milestones=[100,300])(L191-192).
  - best p3 기준 체크포인트 저장(L208-213). TensorBoard 로깅(p3/p5/p10/p15, p_error)(L202-219).

### 3.6 데이터 로딩 — `provider_data.py`

- **`load_h5_and_resample` (L21-48)**: h5의 각 그룹(sample)에서 x/y/t 읽음 → **포인트 수를 sample_size=1024로 고정**: 부족하면 tile 반복(L33-36), 많으면 random choice 후 argsort로 시간순 유지(L38-42). `np.stack((t,x,y), axis=-1)`로 **(t,x,y) 순서 포인트** 구성(L44). 1024개 이상 샘플만 사용(L43). 라벨은 `group.attrs['label']`(L46).

### 3.7 3ET 전처리 — `dataprocess/generate_3et.py`

- `process_labels` (L9-10): 라벨 좌표를 `/640, /480`로 정규화.
- **`process_h5_and_labels` (L12-69)**: frame_ts 사이 구간으로 이벤트 분할(L27-32) → **frequency-adaptive sampling**(L37-48): 한 GT 구간 이벤트 수가 6144 미만이면 인접 구간(최대 ±5)을 확장 병합. → x/240, y/180 정규화(L51-52), t는 [0,1] min-max 정규화(L61), 시간순 정렬(L57-62).
- `save_samples_to_hdf5` (L71-79): 그룹별 x/y/t/p 데이터셋 + `attrs['label']` 저장.

---

## 4. 알고리즘 / 데이터 표현

- **이벤트 표현**: **포인트 클라우드** `(t, x, y)` (극성 p는 전처리엔 저장하나 학습 입력은 3채널 t,x,y만). voxel/frame이 아닌 raw event sampling. 시간 t는 [0,1] 정규화로 순서 정보 제공.
- **시공간 모델링**: PointNet++식 계층 다운샘플(FPS+kNN 그룹핑)로 공간 구조 축약 → 각 stage에서 **Mamba SSM(bimamba v2)**로 센트로이드 시퀀스의 장거리 의존성 모델링. FPS 정렬 순서가 Mamba 스캔 순서를 정의(modules.py L128).
- **선택적 풀링**: attention(softmax) 가중 합으로 그룹/전역 풀링(고정 max-pool 대신 학습형).
- **후처리**: 정규화 좌표를 sensor·spatial_factor로 픽셀 환산 후 p_acc/p_error 계산.

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**: 3ET(시선), DVSGesture/DailyDVS/DVSAction/HMDB51-DVS/UCF101-DVS/THU-CHA/IJRR(분류·odometry). README 표(L39-64).
- **메트릭**:
  - 시선추적: **p-accuracy(p3/p5/p10/p15)**, **p-error(평균 유클리드 픽셀 오차)** — metrics.py `p_acc`(L6-27), `px_euclidean_dist`(L63-83).
  - `p_acc_wo_closed_eye`(L30-60): 감은 눈 프레임 제외 정확도(3ET 라벨에 eye_closed 플래그 있을 때). 본 학습루프는 `p_acc` 사용.
- **loss**: `weighted_MSELoss`(x축 종횡비 가중, metrics.py L86-100) 또는 MSE.
- **명령어**(README L25-36):
  - 전처리: `cd dataprocess && python generate_xxx.py` → train.h5/test.h5를 `./data/xxx/`.
  - 학습: `python train_classification.py` / `train_odometry.py` / `train_eye_tracking.py`.
- **설치**(README L5-13): conda py3.8, PyTorch 2.1+CUDA12.1, `pip install spikingjelly mamba-ssm`, Pointnet2 CUDA 커널.

---

## 6. 의존성

- **mamba-ssm**(필수, SSM selective scan 커널) — 외부, 제외.
- **Pointnet2_PyTorch**(FPS CUDA 가속) — 외부, 순수 PyTorch fallback 있음.
- torch 2.1, h5py, scikit-learn, tensorboard, timm(DropPath), einops(간접), fvcore(FLOPs), spikingjelly.

---

## 7. 강점 / 한계 / 리스크

- **강점**:
  - 포인트 표현으로 이벤트 희소성을 직접 활용(프레임 누적 불필요) → 낮은 연산/메모리.
  - 계층적 FPS로 포인트 수 급감(1024→512→256→128) → Mamba 시퀀스 짧아져 효율적.
  - Mamba의 선형 복잡도 + 추론 캐시(스트리밍 추론 가능).
- **한계/리스크**:
  - **mamba-ssm CUDA 커널 강결합** → CPU/FPGA 이식 시 selective scan 직접 재구현 필요.
  - **FPS가 순차 반복(O(N·npoint))** → 데이터 의존 제어흐름, HW 병렬화 난이도. CUDA FPS 별도 의존.
  - v1/v2 이웃 선택 기준 불일치(README L66) → pretrained 호환성 주의(코드 수정 요구).
  - 시선추적 head가 분류 classifier 재사용(BN+Dropout) — 회귀 안정성 측면에서 비최적일 수 있음 — **추정**.
  - generate_3et.py에 `print(samples)`(L65) 디버그 잔존 → 전처리 매우 느려질 수 있음.

---

## 8. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 가속, 추정)

- **포인트 기반의 HW 매핑**: FPS와 kNN 그룹핑은 불규칙·데이터 의존 연산으로 FPGA에 직접 친화적이지 않음. 다만 **포인트 수가 작아(1024) 짧은 시퀀스**라, Mamba selective scan만 떼어 가속하면 효율적일 수 있음 — **추정**.
- **Mamba SSM 가속**: selective scan(`h_t = A_bar h_{t-1} + B_bar x_t`)은 **선형 재귀**라 FPGA pipelined MAC/state-update에 매핑 적합. 단 입력 의존 Δ/B/C 생성(x_proj, dt_proj)을 함께 매핑해야 함. bimamba(양방향)는 2-pass 필요 → 단방향으로 단순화 검토.
- **on-device 후보**: 프레임 누적·voxelization 없이 raw event를 직접 처리 → 저지연·저메모리 XR 시선추적에 개념적으로 부합. 그러나 FPS/그룹핑 전처리를 SW(CPU/임베디드)에서 처리하고 SSM backbone만 가속하는 분할이 현실적.
- **gg_ssms와 대비**: gg_ssms는 frame+MST, EventMamba는 point+Mamba. FPGA 관점에서 EventMamba의 Mamba(정형 선형 재귀)가 gg_ssms의 MST refine(불규칙 트리)보다 가속 매핑이 단순할 가능성 — **추정(정량 확인 불가)**.
- **양자화·압축**: FLOPs 측정 코드(eventmamba_v2.py L116-122) 존재 → 경량화 baseline 출발점.

---

## 9. 근거 표기

- 모델/학습/전처리: 실제 코드 Read(파일:라인)로 확인.
- "추정": +sigmoid 변형(v1 L88), 회귀 head 비최적성, FPGA 매핑 우열, FPS의 HW 영향.
- "확인 불가": mamba-ssm selective scan 커널 내부, Pointnet2 CUDA FPS 내부(외부·제외). odometry/classification 세부(범위상 요약).
