# gg_ssms 정밀 분석 (Graph-Generating State Space Models, GG-SSM)

> 분석 기준 경로: `REF/XR-Eye-Tracking/Codebase/gg_ssms/`
> 분석 도구: Glob/Grep/Read (실제 코드 라인 근거). 추론 부분은 "추정"/"확인 불가"로 명기.
> 제외 대상: `core/.../third-party/TreeScan`, `third-party/TreeScanLan`(CUDA 커널 원본), `MambaTS`(시계열, 본 보고서 범위에서 요약만), 데이터/체크포인트.

---

## 1. 개요

- **목적**: SSM(State Space Model)의 고정된 1차원 시퀀셜 스캔 한계를 극복하기 위해, **피처 관계 기반으로 동적으로 그래프(최소신장트리, MST)를 구성**하고 그 트리 위에서 상태 전파(state propagation)를 수행하는 범용 시공간 모델 프레임워크.
- **원논문/챌린지**: *Graph-Generating State Space Models (GG-SSMs)*, Zubić & Scaramuzza, **CVPR 2025 Highlight** (arXiv:2412.12423). README L1-35 근거. 11개 데이터셋(ImageNet, KITTI optical flow, 6개 시계열, event-based eye-tracking 등) 검증.
- **핵심 아이디어**: Chazelle의 MST 알고리즘으로 동적 그래프 생성 → 그래프 BFS 순서로 SSM 상태 재귀(refine) 수행. Mamba/VMamba의 사전 정의된 스캔 경로 대신 데이터 구조 적응형 스캔(README L19-24).
- **본 분석의 핵심 관심사(시선·동공 추적)**:
  - 입력 = **이벤트 프레임**(LPW/SEET를 event frame으로 누적, 그레이스케일 1채널, `eye_tracking_lpw/graph_ssm_train.py`).
  - 출력 = **동공 중심 좌표 (x, y)** 정규화 회귀값, `[B, T, 2]` (`graph_ssm_train.py` L182, L209).
  - 세그멘테이션은 본 repo 시선추적 경로에서는 사용하지 않음(좌표 회귀만). 확인됨.
- **시선추적 두 갈래**:
  1. **LPW 데이터셋** (`eye_tracking_lpw/`) — ConvGraphSSM(공간) + TemporalGraphSSM(시간) 결합, 자체 학습코드.
  2. **Ini-30 데이터셋** (`retina/`) — Retina(pbonazzi) 코드 기반에 GG-SSM 백본 이식. `retina/training/models/baseline_3et.py`가 동일 구조 사용.

---

## 2. 디렉토리 구조 (자체 소스 / 제외 구분)

```
gg_ssms/
├── README.md                                   # 설치/사용/인용
├── core/
│   ├── convolutional_graph_ssm/                # [핵심] 공간(2D) Graph SSM
│   │   ├── classification/
│   │   │   ├── models/
│   │   │   │   ├── graph_ssm.py        ★ ConvGraphSSM 백본 (Stem/Block/Level)
│   │   │   │   ├── tree_scanning.py    ★ Tree_SSM 모듈 + tree_scanning_core (2D MST 스캔)
│   │   │   │   ├── build.py            ★ config→모델 빌더
│   │   │   │   ├── tree_scan_utils/tree_scan_core.py  ★ MinimumSpanningTree (격자 MST)
│   │   │   │   └── __init__.py
│   │   │   ├── config.py / logger.py / lr_scheduler.py / export.py / extract_feature.py / ddp_hooks.py
│   │   └── third-party/TreeScan/      ── [제외] CUDA 커널 (bfs/refine/mst). 역할만 언급.
│   └── graph_ssm/
│       ├── main.py                    ★ TemporalGraphSSM (1D 시퀀스용 Graph SSM)
│       └── third-party/TreeScanLan/   ── [제외] CUDA 커널 (mst/bfs/refine). 역할만 언급.
├── eye_tracking_lpw/                           # [핵심] LPW 시선추적 자체 학습코드
│   ├── graph_ssm_train.py             ★ GG-SSM 시선추적 학습 (Dataset/Model/Train/Val/Plot)
│   ├── convlstm.py / convlstm_cell.py / convlstm_delta.py / convlstm_sp.py / convlstmbak.py  # 비교용 ConvLSTM
│   ├── convlstm_train.py              # ConvLSTM 학습 스크립트
│   ├── graph_ssm_bak.py / process_event.py
├── retina/                                     # [자체+Retina 기반] Ini-30 시선추적
│   ├── training/models/baseline_3et.py ★ GG-SSM 이식 모델 (spatial+temporal)
│   ├── training/{loss.py, module.py, models/...}  # spiking/quantization/binarization (Speck HW)
│   ├── data/{datasets/, transforms/, module.py, speck_processor.py}
│   └── scripts/{train.py, eval.py, quantize.py}
└── MambaTS/                                    # [요약만] 시계열 예측. TemporalGraphSSM을 인코더로 교체
    └── models/MambaTS.py, layers/mamba_ssm/*, exp/*, data_provider/*
```

★ = 본 보고서에서 라인 단위 정밀 분석한 자체 핵심 파일.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 공간 백본 — `core/convolutional_graph_ssm/classification/models/graph_ssm.py`

ImageNet 분류용으로 설계된 4-stage 계층적 백본이며, 시선추적에서는 2-level 경량 버전으로 재사용된다.

- **보조 모듈**:
  - `to_channels_first/last` (L23-38): NCHW ↔ NHWC 변환. 이 백본은 내부적으로 **channels_last(NHWC)** 표현을 주로 사용.
  - `build_norm_layer` (L41-59): "BN" 또는 "LN" + 포맷 변환을 Sequential로 조립.
  - `StemLayer` (L73-104): `Conv3x3(in→C/2)+BN+GELU+Conv3x3(C/2→C)+LN`. stride=1이므로 해상도 유지하며 패치 임베딩 역할. 출력은 NHWC.
  - `DownsampleLayer` (L107-126): `Conv3x3 stride2 (C→2C)+LN`. stage 간 2× 다운샘플 + 채널 2배.
  - `MLPLayer` (L129-161): 표준 FFN(Linear-GELU-Drop-Linear-Drop).

- **`GraphSSMLayer` (L164-236)** — 핵심 블록:
  - `self.TreeSSM = Tree_SSM(d_model=channels, d_state=1, ssm_ratio=2, ssm_rank_ratio=2, dt_rank="auto", d_conv=3, conv_bias=False, dropout=0.0)` (L184-196). **d_state=1**로 매우 경량.
  - forward(L214-236): pre-norm 기본 `x = x + DropPath(TreeSSM(LN(x)))` → `x = x + DropPath(MLP(LN(x)))` (L221-222). post_norm/layer_scale/gradient-checkpoint(with_cp) 변형 지원.

- **`GraphSSMBlock` (L239-296)**: depth개의 GraphSSMLayer + LN + (선택)DownsampleLayer. `return_wo_downsample`로 다운샘플 전 피처 반환 가능(L284-295).
  - **버그 주의**: L287 `if not self.post_norm or self.center_feature_scale:` 에서 `self.center_feature_scale`는 정의되지 않은 속성 → post_norm=True 경로에서 AttributeError 발생 가능. **확인됨(코드상 미정의)**. 기본 경로(post_norm=False)에서는 단락평가로 회피됨.

- **`GraphSSM` (L299-470)** — 전체 백본:
  - config 또는 직접 인자(num_levels/depths/channels/mlp_ratio/drop_path_rate)로 구성(L320-331).
  - `num_features = channels * 2^(num_levels-1)` (L340).
  - `one_layer`/`two_layer` 플래그로 stage 수 축소 가능(L364-371). 시선추적은 `num_levels=2, depths=[2,2], channels=16`을 명시 전달.
  - `lr_decay_keywards` (L411-436): stage별 layer-wise LR decay(0.87) 키 생성. 단, 4-stage 가정 하드코딩(`for i in range(4)`)이라 2-level 사용 시 KeyError 가능 — **추정**(시선추적 학습루프에서는 호출 안 함, `graph_ssm_train.py`는 단일 Adam lr 사용).
  - forward(L464-470): `forward_features` 후 NHWC→NCHW permute하여 반환. **분류 head(avgpool/fc)는 forward에서 호출 안 함** → 백본 피처맵만 출력(시선추적에서 외부 pooling).

### 3.2 2D 트리 스캔 코어 — `core/.../models/tree_scanning.py`

GG-SSM의 공간 SSM 본체. Mamba의 selective scan을 **MST 트리 위 재귀**로 대체.

- **autograd 함수** (CUDA 커널 `tree_scan._C` 래핑, 커널 자체는 제외):
  - `_BFS` (L16-22): `_C.bfs_forward(edge_index, max_adj_per_vertex)` → (sorted_index, sorted_parent, sorted_child). 트리를 BFS 순회 순서로 직렬화.
  - `_Refine` (L25-97): `_C.tree_scan_refine_forward(...)`로 트리 위 피처 집계(aggregation, up/down sweep). backward는 feature·edge_weight gradient를 커널로 계산. → **트리 위의 associative scan**(SSM 상태 누적의 그래프 일반화)에 해당.
  - `batch_index_opr` (L100-105): BFS 정렬 인덱스로 edge_weight를 gather.

- **`tree_scanning_core` (L108-158)** — 상태 전이 핵심:
  - `dts = softplus(dts + delta_bias)` (L112) → Δ.
  - `deltaA = exp(dts * As)` (L114): **이산화된 상태 전이 행렬 A_bar** (b d l). 이것이 트리 간선 가중치(edge_weight)로 사용됨(L122).
  - `deltaB = dts * Bs`, `BX = deltaB * xs` (L115-116): 입력 항 B_bar·x → feat_in(L121).
  - `fea4tree_hw = rearrange(xs, "b d (h w) -> b d h w")` (L128) 로 2D 격자 복원 후 `MinimumSpanningTree("Cosine", torch.exp)`로 **코사인 거리 기반 MST 생성**(L129-130).
  - BFS로 직렬화(L131) → edge_weight를 트리 순서로 정렬(L132) → `_Refine`로 트리 위 상태 누적(L137-144).
  - 출력 정규화(h_norm) 후 C 행렬과 곱: `y = (h ⊗ C)` (L149-156), skip 연결 `y += Ds * xs` (L157). → 표준 SSM의 `y = C·h + D·x`를 트리 스캔으로 구현.

- **`tree_scanning` (L161-210)**: x_proj로 (Δ,B,C) 생성, dt_proj로 Δ 확장, A_logs로 A 복원, `force_fp32=True` 강제(L195) 후 core 호출, out_norm 적용. **K=1**(단일 스캔 방향, L175 영역에서 K 추출되지만 코어는 K=1).

- **`Tree_SSM` (L213-414)**:
  - 파라미터: in_proj(d_model→2·d_expand), dwconv(`Conv2d groups=d_expand`, depthwise, L264-273), x_proj_weight(L276-284), dt_projs(L292-311), A_logs(S4D 실수 초기화 L356-371), Ds(skip, L373-383).
  - **A_log_init** (L356-371): `A = arange(1, d_state+1)` 반복 → log. S4D 실초기화.
  - forward(L403-414): `in_proj` → chunk로 (x, z) gate 분리 → dwconv → SiLU → `forward_core`(tree_scanning) → `y = y * z`(gated) → out_proj. **Mamba 게이팅 구조 + 트리 스캔**.

### 3.3 격자 MST 생성 — `core/.../models/tree_scan_utils/tree_scan_core.py`

- `_MST` (L12-24): `_C.mst_forward(edge_index, edge_weight, vertex_index)` 래핑(`tree_scan._C`). backward는 None(미분 불가, no_grad 영역에서만 사용).
- 거리 함수: `norm2_distance`(L27-30), `norm1_distance`(L33-36).
- **`MinimumSpanningTree` (L39-102)**:
  - `_build_matrix_index` (L46-55): H×W 격자의 4-이웃(행/열 방향) 간선 인덱스 생성. 즉 **격자 그래프**에서 MST를 구함.
  - `_build_feature_weight_cosine` (L68-91): 인접 픽셀 간 코사인 유사도 기반 가중치. `max_tree` 플래그로 부호 반전(min-tree vs max-tree).
  - forward(L93-102): `no_grad` 하에 index+weight 계산 → `mst(...)` 호출하여 트리 간선 반환. 정점 수 = H·W.

> **요약**: 공간 경로는 (1) 픽셀 격자에서 코사인 유사도로 MST 생성 → (2) BFS 직렬화 → (3) A_bar/B_bar·x를 트리 위에서 refine(누적) → (4) C·h + D·x. 이것이 GG-SSM의 "동적 그래프 스캔".

### 3.4 시간 SSM — `core/graph_ssm/main.py` (TemporalGraphSSM)

1D 시퀀스(프레임 임베딩 시퀀스, 시계열)용 Graph SSM. `tree_scan_lan._C` 커널 사용(제외 대상이지만 인터페이스 분석).

- autograd: `_MST`(L10-20), `_BFS`(L22-28), `_Refine`(L31-60) — 위와 동일 패턴.
- **`tree_scanning_algorithm` (L82-205)** — 시간 스캔 핵심:
  - 표준 Mamba 전반부: in_proj→chunk(hidden, gate)(L87-90), conv1d+SiLU(L92-94), x_proj로 (Δ,B,C) split(L97-102), dt_proj+softplus(L103-106).
  - 이산화: `A = -exp(A_log)`(L108), `discrete_A = exp(A·Δ)`(L109-111), `discrete_B = Δ·B`, `deltaB_u = discrete_B·hidden`(L112-115).
  - **두 개의 트리 결합**(핵심 차별점):
    1. **사슬 트리(chain tree)**: `tree_[i,i+1]` 연속 간선(L127-132) → 표준 순차 SSM과 동일한 인과적 스캔. `feature_out1`(L176-178).
    2. **피처 기반 MST**: `generate_pairs`로 시간축 candidate 간선 생성(최근 context_len 구간은 i+1,i+2,i+3 다중 간선)(L141-152) → 코사인 거리로 가중치 → `_MST`로 트리 생성 → BFS → `feature_out2`(L154-183).
  - 결합: `feature_out = feature_out2 * 0.3 + feature_out1`(L184-186). 0.3은 하이퍼파라미터.
  - flip/roll로 인과성 처리(L119-120, L189) → C와 곱(L194-196) → skip `+ hidden·D`(L199) → gate `* silu(gate)`(L200) → out_proj(L202-204).
- **`GraphSSM` 클래스 (L208-304)**: 표준 Mamba 시그니처(d_model/d_state/d_conv/expand). forward는 `(input_states, context_len)` 받아 위 알고리즘 호출(L303-304).

> **요약**: 시간 경로는 **인과적 사슬 스캔 + 피처-MST 스캔의 가중합**으로, 비국소(non-local) 시간 의존성을 동적으로 포착.

### 3.5 LPW 시선추적 학습 — `eye_tracking_lpw/graph_ssm_train.py`

- **하이퍼파라미터**(L29-36): height=60, width=80, batch_size=16, seq=40, stride=1(train)/40(val), chunk_size=500, epochs=100.
- **`normalize_data` (L57-72)**: 프레임별 평균/표준편차 정규화(z-score).
- **`create_samples` (L75-96)**: chunk(500프레임) 단위로 길이 seq의 슬라이딩 윈도우 생성(stride). 라벨에도 동일 적용.
- **`EventDataset` (L99-146)**:
  - `__getitem__` (L113-131): h5(`file.root.vector`)에서 sample 읽어 `cv2.resize`로 80×60 리사이즈 + 정규화, `[seq, 1, H, W]` 반환.
  - 라벨(L127-129): `label1=x/M/8, label2=y/N/8` — **원본 좌표를 (M, N) 및 추가 8로 나눠 정규화**(8은 입력 다운샘플/스케일 보정 추정). `[seq, 2]`.
  - `_concatenate_files` (L133-146): 라벨 txt에서 `lines[3::4]`로 4줄마다 1개 추출(LPW 라벨 포맷). create_samples로 윈도우화.
- **`GraphSSMModel` (L154-210)** — 시공간 결합 모델(★):
  - `spatial_backbone = ConvGraphSSM(in_chans=1, num_levels=2, depths=[2,2], channels=16, mlp_ratio=4)` (L160-170) → `d_model = num_features = 16·2^(2-1) = 32` (L172-174).
  - `temporal_ssm = TemporalGraphSSM(d_model=32, d_state=16, d_conv=4, expand=2)` (L177-179).
  - `fc_out = Linear(32, 2)` (L182).
  - forward(L184-210): `[B,T,C,H,W]` → `view(B*T,C,H,W)`로 시간을 배치에 흡수 → 공간 백본 → `adaptive_avg_pool2d(1,1)`로 프레임당 벡터(L199) → `view(B,T,d)` → temporal_ssm(context_len=T) → fc_out → `[B,T,2]`.
- **학습 루프 (L256-348)**: criterion=`SmoothL1Loss`(L252), optimizer=Adam(lr=1e-3, L253). 100 epoch.
- **검증·메트릭 (L292-315)**: 예측-타깃 차이를 `dis[...,0]*=height, dis[...,1]*=width`로 픽셀화 후 `dist=norm`. `dist > {1,3,5,10}` 비율을 err_rate로 계산(p-error 보완). wandb 로깅(L317-327).
- **플롯 (L350-505)**: 좌표 시계열 비교, 프레임 위 예측/GT 점 오버레이 저장.

### 3.6 Ini-30 시선추적 — `retina/training/models/baseline_3et.py`

- `Baseline_3ET` (L15-71): `graph_ssm_train.py`의 `GraphSSMModel`과 **사실상 동일 구조**(ConvGraphSSM 2-level/channels=16 + TemporalGraphSSM(d_model=32) + Linear(32,2)). 입력 `[B,T,C,H,W]` → 출력 `[B,T,2]`. README L129-132에 따라 `retina/scripts/train.py --run_name=graph_ssm`로 학습, wandb 프로젝트 `eye_tracking_ini_30`.
- retina 패키지는 원 Retina(pbonazzi) 기반(spiking/quantization/binarization for Speck 뉴로모픽 칩) — 시선추적 회귀 head에 GG-SSM 백본을 끼워 넣은 형태.

### 3.7 비교 베이스라인 — `eye_tracking_lpw/convlstm.py` (ConvLSTM)

- `ConvLSTMCell` (L13-120): 표준 ConvLSTM 게이트(`conv(in+h)` → i/f/o/g split → `c=f·c+i·g, h=o·tanh(c)`)인데 **`combined = cat(input, delta=h_cur)`** 사용(L50-54)하고 **eval 모드에서 sparse_rate(0의 비율) 로깅**(L57-71). 이는 cb-convlstm의 change-based 희소성 분석을 위한 계측.
- `ConvLSTM` (L123-271): 다층 스택, batch_first 지원. (별도 convlstm_train.py에서 사용 — 본 repo 내 비교 실험용.)

---

## 4. 알고리즘 / 데이터 표현

- **이벤트 표현**: LPW/SEET 원시 이벤트를 누적해 **이벤트 프레임(event frame)**으로 변환(외부 전처리, `process_event.py` 계열). 학습 입력은 그레이스케일 1채널 프레임 시퀀스 `[T,1,60,80]`. graph 표현은 **모델 내부에서 동적 생성**(픽셀 격자 MST / 시간 candidate MST)되며 입력 자체는 frame.
- **공간 SSM 동작**: 픽셀 격자 → 코사인 유사도 가중 → MST → BFS 직렬화 → A_bar(=edge weight)로 트리 위 상태 누적(`refine`) → C·h+D·x. 표준 1D selective scan을 트리로 일반화.
- **시간 SSM 동작**: 프레임 임베딩 시퀀스에 (a) 인과 사슬 스캔 + (b) 피처 MST 스캔을 0.3 가중 결합.
- **후처리**: 정규화 좌표를 `*width/*height`로 픽셀 환산하여 거리(p-error) 계산. median_filter 등은 cb-convlstm 쪽에서 사용(본 repo 학습루프에는 미적용).

---

## 5. 학습 / 평가 파이프라인

- **데이터셋**:
  - LPW(SEET event frame) — `eye_tracking_lpw`, h5(`data_ts_pro/{train,val}`) + label txt. README L148-166.
  - Ini-30 — `retina`, tonic/sinabs 기반. README L60-64, L127-132.
- **메트릭**:
  - LPW: `dist > {1,3,5,10}px` 초과 비율(err_rate). 작을수록 좋음(검출 실패율).
  - Ini-30/3ET 계열: p-accuracy(p3/p5/p10/p15), p-error(평균 유클리드 픽셀 오차).
- **명령어**(README 근거):
  - LPW: `python eye_tracking_lpw/graph_ssm_train.py` (L164). 사전 `DATA_DIR_ROOT` 설정(graph_ssm_train.py L216).
  - Ini-30: `CUDA_VISIBLE_DEVICES=i python retina/scripts/train.py --run_name=graph_ssm --device=i` (README L131).
  - ImageNet 분류 forward: `python core/.../models/graph_ssm.py` (README L93).
- **설치**: conda py3.11 + PyTorch 2.5 + CUDA 12.4, TreeScan/TreeScanLan `pip install -e .`로 CUDA 확장 빌드 필수(README L40-56).

---

## 6. 의존성

- 핵심: PyTorch 2.5, einops, timm, easydict, yaml, **TreeScan/TreeScanLan CUDA 확장**(`tree_scan._C`, `tree_scan_lan._C`) — bfs/mst/refine forward·backward 커널.
- 시선추적: matplotlib, opencv-python, tqdm, tables(PyTables, h5), wandb, einops(README L65-69).
- Ini-30: dv-processing, sinabs, tonic, thop, samna, fire(README L60-64).
- MambaTS: mamba-ssm 계열(layers/mamba_ssm).

---

## 7. 강점 / 한계 / 리스크

- **강점**:
  - 동적 그래프(MST) 스캔으로 비국소 의존성 포착 → 적은 파라미터로 SOTA(README 주장: eye-tracking 검출율 +0.33%, 파라미터 감소).
  - 공간/시간 SSM 모듈화로 시공간 파이프라인 조립 용이(LPW/Ini-30가 동일 백본 재사용).
  - d_state=1(공간) 등 극경량 설정 가능.
- **한계/리스크**:
  - **CUDA 커널 의존성**: bfs/mst/refine가 커스텀 CUDA 확장. 다른 HW(특히 FPGA/CPU)로의 이식이 가장 큰 장벽. MST/BFS는 데이터 의존적 불규칙 제어흐름 → 하드웨어 매핑 난이도 높음.
  - 코드 품질: `GraphSSMBlock.center_feature_scale` 미정의(L287), `lr_decay_keywards` 4-stage 하드코딩(L417), 중복 import(tree_scanning.py L6-7) 등 정리 미흡.
  - LPW 라벨 정규화의 `/8`(graph_ssm_train.py L127-128) 근거가 코드 주석에 없음 — **추정**(스케일 보정).
  - `force_fp32=True` 강제(tree_scanning.py L195)로 저정밀 추론 경로 부재.

---

## 8. 우리 프로젝트 관점 시사점 (XR 시선추적 + FPGA 가속, 추정)

- **on-device/저지연 적합성**: 공간 SSM은 d_state=1, channels=16의 극경량 구성 가능 → 임베디드 후보. 그러나 **MST/BFS 동적 그래프 생성이 FPGA 친화적이지 않음**(불규칙 메모리 접근, 데이터 의존 트리 구조). FPGA 이식 시 정적 격자 스캔(예: 행/열 raster scan)으로 단순화하거나, 트리 구성을 호스트(CPU)에서 처리하고 refine만 가속하는 분할이 현실적 — **추정**.
- **시간 SSM**: 사슬 트리(`feature_out1`)만 쓰면 표준 인과 SSM이 되어 **선형 재귀(`h_t = A_bar·h_{t-1} + B_bar·x_t`)** 형태로 FPGA의 systolic/pipelined MAC에 매핑 용이. 피처-MST 분기(`feature_out2`, 0.3 가중)는 정확도 기여가 제한적일 수 있어 HW에서는 생략 검토 가치 있음 — **추정(정확도 영향 확인 불가)**.
- **대안 비교**: 같은 repo의 ConvLSTM(`convlstm.py`)은 고정 conv 게이트 구조로 FPGA 매핑이 훨씬 단순. GG-SSM의 정확도 이점 대비 HW 복잡도 trade-off를 정량 비교할 가치.
- **양자화**: `retina/training/models/quantization/`(lsq/dorefa) 자산이 있어, 시선추적 회귀 head + SSM 백본 양자화 실험의 출발점으로 활용 가능.
- **권장 이식 전략**: ① 공간 백본을 정적 스캔 CNN/SSM으로 치환 → ② 시간 SSM은 사슬-스캔 단일화 → ③ 좌표 회귀 head는 단순 FC로 FPGA 친화적.

---

## 9. 근거 표기

- 모델/알고리즘/학습 구조: 모두 실제 코드 Read(파일:라인 명시)로 확인.
- "추정" 항목: LPW 라벨 `/8` 의미, lr_decay 미사용, feature_out2 정확도 기여, FPGA 이식 전략.
- "확인 불가": TreeScan/TreeScanLan CUDA 커널 내부 구현(제외 대상, 인터페이스만 분석). MambaTS 세부(범위 외 요약).
- 코드 결함(미정의 속성 등)은 코드 라인 직접 인용으로 "확인됨" 표기.
