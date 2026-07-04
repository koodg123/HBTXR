# ViTCoD 코드베이스 정밀 분석

> 대상 repo: `REF/Transformer-Accel/ViTCoD`
> 논문: *ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design* (HPCA 2023, GATECH-EIC)
> 분석 기준: 실제 소스 Read 기반. 라인 근거 = `파일명:라인범위`. 미확인은 "추정"/"확인 불가" 명시.

---

## 1. 개요

ViTCoD는 Vision Transformer(ViT)의 **어텐션 맵 희소성**을 알고리즘-하드웨어 공동설계로 가속하는 프레임워크다. 핵심 아이디어는 두 가지다.

1. **알고리즘 레벨**: ViT 어텐션 맵을 학습된 임계값으로 프루닝(pruning)한 뒤, **denser/sparser 두 패턴으로 양극화(polarize)**하여 워크로드를 정규화한다. 추가로 **경량 학습형 auto-encoder(encoder/decoder Linear)**를 Q/K에 삽입해 고비용 데이터 이동을 저비용 연산으로 치환한다. (`README.md:31-32`)
2. **하드웨어 레벨**: denser 워크로드와 sparser 워크로드를 동시 처리하도록 **denser engine + sparser engine**으로 PE array를 동적 분할(dynamic PE allocation)하고, 온칩 encoder/decoder 엔진으로 Q/K 압축·복원을 수행해 데이터 이동을 줄인다. (`README.md:34`, `Hardware/Simulator/README.md:32`)

코드베이스는 (1) 알고리즘(DeiT/LeViT 위에 프루닝+auto-encoder를 얹은 학습 코드), (2) 사이클 추정 **하드웨어 시뮬레이터**(Python), (3) Nvidia TX2 EdgeGPU/GPU 프로파일링 — 세 부분으로 구성된다. 본 repo에는 **RTL(.v/.sv)이나 HLS 커널(.cpp)은 없으며**, 하드웨어는 전적으로 **분석적(cycle-accurate analytical) Python 시뮬레이터**로 모델링되어 있다. (확인: Glob 결과에 .v/.sv/.cpp/.cc/.h 없음)

---

## 2. 디렉토리 구조

### 2.1 ViTCoD 자체 코드 (분석 대상)

```
ViTCoD/
├── README.md, LICENSE                         # Apache 2.0
├── Figures/                                    # overview.png, Vision_vs_NLP.png (생성물/이미지)
├── Hardware/Simulator/                         # ★ 핵심: 하드웨어 사이클 시뮬레이터
│   ├── ViTCoD.py                               # 어텐션(SDDMM/SpMM) 사이클 시뮬레이터 (메인)
│   ├── ViT_FFN.py                              # 선형 투영 + FFN/MLP 사이클 시뮬레이터
│   ├── SRAM.py                                 # SRAM/HBM 대역폭·preload 사이클 모델
│   ├── PE.py                                   # 64x64 PE array 동작 모델
│   ├── reorder.py                              # 어텐션 맵 양극화(denser/sparser) 전처리
│   ├── README.md
│   ├── masks/deit_tiny_lowrank/               # info_*.npy(프루닝 마스크), reodered_*.npy, global_token_*.npy, 로그 .txt (생성물)
│   └── figs/                                   # arch.png, sparse_attention.jpg (이미지)
├── Profile/                                     # 프로파일링
│   ├── TX2_benchmark/ (benchmark.py, parse_json.py, *.sh, trtexec)  # Jetson TX2 + TensorRT 지연 측정
│   ├── GPU_benchmark/ (benchmark.py, *.sh)     # GPU torch profiler
│   └── models/ (vit.py, linformer/)           # 프로파일용 ViT/Linformer 모델 정의
└── Algorithm/
    ├── deit/   # DeiT 기반 (ViTCoD 자체 수정 + vendor 혼재, 아래 구분)
    └── levit/  # LeViT 기반 (DeiT와 동일 패턴, vendor 혼재)
```

**ViTCoD 자체 알고리즘 핵심 파일 (`Algorithm/deit/`, `Algorithm/levit/`에 동형):**
- `gen_mask.py` — 어텐션 맵 → 희소 마스크 생성 (info/ratio/random/std 4종)
- `mask_utils.py` — 마스크 로드/검증/FLOPs 감소량 계산
- `attnweights_utils.py` — 학습 중 평균 어텐션 맵 누적·저장
- `timm/vision_transformer.py` — **★ ViTCoD가 직접 개조한 timm**: 마스크 어텐션 + auto-encoder(encoder/decoder Linear) + SVD 저랭크. (vendor timm을 fork해 수정)
- `engine.py` / `main.py` — recon_loss를 학습 loss에 통합 (`engine.py:48-58`)
- `models.py` — DeiT 모델 등록 (대부분 stock DeiT, ViTCoD 수정 거의 없음)

### 2.2 제외(vendor/third-party) — 이름만 언급

- **DeiT 원본** (Facebook): `Algorithm/deit/{datasets.py, samplers.py, losses.py, utils.py, hubconf.py, run_with_submitit.py, resmlp_models.py, cait_models.py}` 등 — DeiT 배포 원본.
- **timm 라이브러리 fork**: `Algorithm/deit/timm/` — Ross Wightman의 timm을 가져온 뒤 `vision_transformer.py`만 ViTCoD용으로 개조. 나머지는 vendor.
- **LeViT 원본**: `Algorithm/levit/{levit.py, levit_c.py, ...}` — LeViT 배포 원본. ViTCoD 추가분(`gen_mask.py`, `mask_utils.py`, `attnweights_utils.py`)은 DeiT 쪽과 동형(추정).
- **Linformer**: `Profile/models/linformer/` — 프로파일 비교용 third-party 모델.
- **trtexec** (바이너리), `*.json/*.npy/*.txt 로그`, `.git/`, `.DS_Store` — 생성물/도구.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 어텐션 맵 양극화: `reorder.py` (denser/sparser 분리)

이 스크립트가 ViTCoD 알고리즘-HW 다리 역할의 핵심이다. 프루닝된 어텐션 마스크(`info_0.95.npy`)를 **denser(global token) 영역과 sparser 영역으로 재배열**하여 시뮬레이터 입력을 만든다.

- **`calc(graph, ax, threshold=90)`** (`reorder.py:8-59`)
  - 마스크의 0 위치(=살아남은 어텐션, 비프루닝)를 엣지로 보고 DGL 그래프를 구성 (`reorder.py:14-20`).
  - `out_degree > threshold`인 노드를 **고밀도(global) 토큰**으로 식별 (`reorder.py:28-32`). 즉 "다른 토큰 다수와 어텐션을 맺는 토큰"을 denser로 분리.
  - 고밀도 노드 인덱스를 앞쪽으로 **재배열(permute)**하여 denser 블록이 좌측에 모이도록 재정렬 (`reorder.py:42-49`). `total`(고밀도 노드 수) = 이후 `num_global_tokens`.
  - 반환: `(dense_cnt, total_cnt, new_graph(재배열된 마스크), total(global token 수))` (`reorder.py:59`).
- **메인 루프** (`reorder.py:69-104`): `masks/deit_tiny_lowrank/info_0.95.npy`를 layer×head별로 `calc(threshold=50)` 호출 (`reorder.py:93`), 결과를 누적해
  - `reodered_info_0.95.npy` (재배열 마스크, `reorder.py:102`)
  - `global_token_info_0.95.npy` (head별 global token 수, `reorder.py:103`) 로 저장. 전체 sparsity도 출력 (`reorder.py:100`).

> 주의: 입력 경로가 하드코딩됨 `'/home/sheminghao/ViTCoD/...'` (`reorder.py:69`) — 실행 시 수정 필요. **확인됨**.
> 의존성: `dgl`, `torch`, `matplotlib`.

**시사점**: denser/sparser 양극화는 단순 thresholding(out-degree)과 인덱스 재배열로 구현된다. "동적 sparse pattern 예측"이 아니라 **오프라인 정적 재배열**이다(ViT의 고정 패턴 가정 활용, `README.md:21`).

### 3.2 어텐션 사이클 시뮬레이터: `ViTCoD.py` (★ 가장 중요)

denser/sparser engine의 동적 PE 분할로 어텐션(QKᵀ=SDDMM, S·V=SpMM) 사이클을 추정한다. layer×head 루프(`ViTCoD.py:65-67`, head는 head-parallel 가정상 1개만 순회 `head//head`)로 진행.

**입력 준비** (`ViTCoD.py:41-50`):
- `reodered_info_p.npy`(마스크), `global_token_info_p.npy`(global 수) 로드.
- Q/K/V는 무작위 텐서로 대체(latency만 추정하므로 값 무관, `ViTCoD.py:45-47`). dim = (layer, head, token=197, feature_dim=64).

**(a) Sparser 영역 추출** (`ViTCoD.py:75-87`):
- `mask[:, global_tokens:]`의 0 위치를 `coo_matrix`로 추출해 (row, col) 좌표 리스트 `sparser` 생성 (`ViTCoD.py:78-79`). 이것이 **sparser engine이 처리할 비정형 비제로 좌표**.
- `sparse_ratio` = 비제로 수 / 영역 크기 (`ViTCoD.py:83`).

**(b) 데이터 preload + decoder 전처리 (Dense/Sparse QKᵀ)** (`ViTCoD.py:90-152`):
- Dense 패턴: global_tokens마다 K, decoder weight preload (`ViTCoD.py:95-100`), Q preload (`ViTCoD.py:110-112`). preprocessing(decoder 복원) 사이클은 `head*ratio*K.shape[1]`을 PE 처리량(`PE_width²/head`)으로 나눠 누적 (`ViTCoD.py:102-103, 114-115`). **encoder ratio(2/3)만큼 압축된 Q/K를 로드 후 on-chip decoder로 복원**하는 흐름을 모델링.
- Sparse 패턴: **K-stationary** 선택(주석 근거: global token 수가 head마다 크게 달라 score-stationary가 부적합, `ViTCoD.py:123`). K loop (`ViTCoD.py:128-134`), Q는 `reload_ratio`(SRAM 용량 대비 재적재 비율, `ViTCoD.py:138-148`)를 곱해 누적.

**(c) 동적 PE 분할 (denser↔sparser engine)** (`ViTCoD.py:159-161`):
```python
dense_ratio    = global_tokens*Q.shape[0] / (len(sparser) + global_tokens*Q.shape[0])
dense_PE_width = int(PE_width * dense_ratio)
sparse_PE_width = PE_width - dense_PE_width
```
- **핵심 co-design 로직**: 전체 PE 폭(64)을 dense/sparse 워크로드 **비율에 비례해 동적 할당** (`ViTCoD.py:159-161`). denser engine과 sparser engine이 PE를 나눠 갖고 **동시 실행**됨을 모델링.

**(d) SDDMM (QKᵀ) 사이클** (`ViTCoD.py:163-177`):
- Dense SDDMM: global_tokens × ⌈Q행/dense_PE_width⌉ × ⌈Q특징/(PE_width/head)⌉ 누적 (`ViTCoD.py:164-167`).
- Sparse SDDMM: ⌈sparser수/sparse_PE_width⌉ × ⌈특징/(PE_width/head)⌉ (`ViTCoD.py:172-174`).
- 두 엔진 동시 실행 → **`SDDMM_PE_cycles = max(dense, sparse)`** (`ViTCoD.py:176`). max 사용이 "동시 처리"의 핵심 모델링.

**(e) SpMM (S·V) 사이클** (`ViTCoD.py:182-228`):
- V preload(dense/sparse 각각, `ViTCoD.py:184-186, 216-219`).
- Dense SpMM: ⌈V.shape[0]·V.shape[1]·global_tokens / (dense_PE_width·PE_width/head)⌉ (`ViTCoD.py:190-191`).
- Sparse SpMM: sparser 좌표를 **row별로 그룹화(num_list)**하여 (`ViTCoD.py:197-212`) row당 비제로 수 × V특징을 합산, sparse_PE_width로 나눔 (`ViTCoD.py:223-225`). 이것이 **CSR-유사 row-wise 비정형 SpMM 처리**의 사이클 모델.
- `SpMM_PE_cycles = max(sparse, dense)` (`ViTCoD.py:227`).

**(f) 집계** (`ViTCoD.py:269-278`):
- `Total cycles = max(total_preload, total_PRE + total_SDDMM + total_SpMM)` (`ViTCoD.py:278`). → **메모리 bound(preload) vs 연산 bound(compute) 중 큰 쪽**으로 latency 결정 = roofline식 모델.

### 3.3 SRAM/대역폭 모델: `SRAM.py`

- **하드웨어 파라미터** (`SRAM.py:12-20`): Q/K/V SRAM 각 53KB, index 20KB, output 108KB; **HBM 대역폭 76.8 GB/s, clock 500 MHz**. → ViTCoD 논문 가속기 스펙(추정: HBM2 기반 엣지급).
- preload_* 류 메서드(`preload_Q/K/V/decoder/encoder/index/weight`, `SRAM.py:23-107`): 공통 패턴 — `latency = nums*bits / (bandwidth*bandwidth_ratio)`; `cycle = ceil(latency*clock)` (`SRAM.py:28-29` 등). 즉 **데이터 전송량 → 사이클** 환산. `bandwidth_ratio`로 채널/헤드 분할 대역폭을 표현.
- 용량 초과 시 에러 출력하나 `exit()`는 주석 처리되어 **실제 중단 안 함**(검증용, `SRAM.py:24-26`). **확인됨**.
- encoder/decoder preload는 Q SRAM 한도를 공유(`SRAM.py:23-41`) — encoder/decoder weight가 작아 별도 버퍼 미모델링(추정).

### 3.4 PE array 모델: `PE.py`

- `PE_array`: **64×64 결과 누산 레지스터** `res` (`PE.py:6`). 실제로는 시뮬레이터 메인이 사이클을 직접 카운트하므로 이 클래스는 **기능 검증용/보조**(메인 ViTCoD.py에서 `my_PE` 생성만 하고 cal_* 미사용 — `ViTCoD.py:50`).
- `cal_attn_map()` (`PE.py:23-36`): Q·K MAC을 element별 1 cycle로 카운트 → QKᵀ 1 output.
- `cal_V_update()` (`PE.py:38-53`): attn·V 삼중 루프 MAC, cycle 누적 → S·V.
- `store_res_V()` (`PE.py:62-64`): 64-way 병렬 저장 가정.
- **한계**: 메인 시뮬레이터는 PE.py의 cal_*를 호출하지 않고 자체적으로 ⌈워크/PE처리량⌉ 공식을 쓴다. PE.py는 사실상 사용되지 않는 reference 구현(확인: `ViTCoD.py`/`ViT_FFN.py`에 `my_PE.cal_` 호출 없음).

### 3.5 선형/FFN 사이클 시뮬레이터: `ViT_FFN.py`

어텐션 외 **QKV 선형 투영 + multi-head concat projection + FFN(MLP 2층)**의 사이클을 추정. (`ViT_FFN.py`)

- **Embedding(QKV 투영)** (`ViT_FFN.py:107-159`): Q/K/V 각각에 대해 weight preload(`preload_weight`, `ViT_FFN.py:111`) + 연산 사이클(embedding·feature / (PE_width·PE_height/head), `ViT_FFN.py:113-114`). PE_height(8)가 여기서 처음 활용됨 → **선형층은 PE_width×PE_height(64×8) systolic 사용**(추정).
- **on-chip encoder 적용** (`ViT_FFN.py:136-153`): Q/K를 encoder로 압축(`preload_encoder`, ratio=2/3)한 뒤 store_out. → **압축된 Q/K를 DRAM에 저장**해 이후 어텐션 단계 데이터 이동 절감. 이것이 auto-encoder의 HW상 이득 모델링.
- **Multi-head concat projection** (`ViT_FFN.py:161-166`).
- **FFN** (`ViT_FFN.py:180-194`): MLP 1층(dim→4·dim) + 2층(4·dim→dim) 각각 연산/preload 사이클. mlp_ratio=4 하드코딩 (`ViT_FFN.py:181, 187`).
- **집계** (`ViT_FFN.py:209-213`): `linear = max(compute+PRE, preload)`, `ffn = max(compute, preload)`, `total = linear + ffn`. 이를 `ViTCoD.py`의 어텐션 사이클과 **수동 합산**해 end-to-end latency 산출(`Simulator/README.md:38-50`).

### 3.6 마스크 생성: `gen_mask.py` (프루닝 정책)

어텐션 점수 npy(`attention_score.npy`)로부터 희소 마스크를 만든다. 4가지 정책:
- **`gen_info_based_mask` / `info_cutoff`** (`gen_mask.py:51-105`): ★ ViTCoD 기본 정책. 각 행(쿼리)의 어텐션 값을 내림차순 정렬해 **누적합이 info 임계(기본 0.186)에 도달**할 때까지의 상위 토큰만 유지, 나머지 마스킹 (`gen_mask.py:91-105`). = "정보량 기반(top-cumulative) 프루닝". cls 토큰 행/열은 보존 (`gen_mask.py:67-68`).
- `gen_ratio_based_mask` (`gen_mask.py:22-47`): 행별 rank가 cutoff 미만이면 유지(고정 비율 프루닝).
- `gen_random_mask` (`gen_mask.py:7-19`): scipy.sparse.random 무작위 마스크(baseline).
- `gen_std_based_mask` (`gen_mask.py:108-129`): mean+coef·std 임계 마스킹.
- CLI: `--method {info,ratio,random,std}`, `--info_cut 0.186` (`gen_mask.py:132-152`). sparsity는 모델 크기별(base/small/tiny) 분모로 산출 (`gen_mask.py:76-84`).

> 주의: 분모 `12*12*197*197`이 일부 경로에 하드코딩(`gen_mask.py:40, 42`) — base 기준. 모델별 분기는 info 함수에만 존재(`gen_mask.py:76-84`). **확인됨**.

### 3.7 학습형 auto-encoder + 마스크 어텐션: `timm/vision_transformer.py` (★)

ViTCoD가 timm을 fork해 `Attention` 클래스를 개조한 부분. 두 메커니즘이 결합:

**(A) 학습형 head-mixing auto-encoder** (`vision_transformer.py:235-249, 359-426`):
- `svd_type`에 따라 encoder/decoder Linear를 head 차원에 구성:
  - `mix_head_fc_qk`: `encoder_q/k: Linear(num_heads → hidden)`, `decoder_q/k: Linear(hidden → num_heads)`, `hidden = ceil(num_heads/2)` (`vision_transformer.py:235-240`). → **헤드 차원 압축(2/3 등)**.
- forward (`mix_head_fc_qk`, `vision_transformer.py:359-386`): Q/K를 `b×h×(n·d)`로 reshape 후 head축으로 permute → `decoder(encoder(q))`로 통과 (`vision_transformer.py:373-374`) → **헤드 간 정보를 저차원으로 압축·복원**. 이것이 논문의 "learnable auto-encoder"의 실제 구현. `recon_loss = dist(q,pre_q)+dist(k,pre_k)` (`vision_transformer.py:376`).
- `mix_head_fc_q`/`mix_head_fc_k`는 Q 또는 K만 처리(`vision_transformer.py:388-426`).

**(B) SVD 저랭크 변형(분석/비교용)** (`vision_transformer.py:260-357`):
- `single_head`: `torch.svd_lowrank(q, q=100)` 후 재구성 (`vision_transformer.py:277-285`).
- `mix_head`: `torch.pca_lowrank(q, q=6, niter=1)`을 **확률 3%로 학습 중 적용**(`vision_transformer.py:303-334`). → SVD/PCA 저랭크가 어텐션에 미치는 영향 실험용. auto-encoder(FC)가 이를 학습으로 대체(추정).

**(C) 마스크 적용 + 어텐션 통계** (`vision_transformer.py:432-450`):
- `attn = cal_attn(q, kᵀ)*scale` (`vision_transformer.py:433`).
- `attn_mask`가 있으면 batch로 확장 후 `masked_fill(..., -inf)` (`vision_transformer.py:434-438`) → softmax에서 0. = **고정 sparse pattern 강제**.
- `need_weight`면 `attention_sum`/`num_attention` 버퍼에 평균 어텐션 누적 (`vision_transformer.py:441-443`) → `attnweights_utils.save`가 npy로 덤프(`attnweights_utils.py:19-41`) → 이것이 `gen_mask.py` 입력.
- 마스크 로드: `_generate_patterns` (`vision_transformer.py:750-777`)가 `mask_utils.mask_read_files`로 npy를 블록별 tensor로 변환, FLOPs 감소량 계산(`mask_utils.cal_reduced_Gflops`).

**(D) 학습 통합**: `recon_loss`를 분류 loss에 `1e-4` 가중으로 합산 (`engine.py:57-58`). auto-encoder를 end-to-end로 공동 학습.

### 3.8 마스크 유틸: `mask_utils.py`

- `cal_reduced_Gflops` (`mask_utils.py:8-23`): 마스킹 비율 × 2·N²·C로 절감 GFLOPs 산출.
- `cal_num_elements_pattern` (`mask_utils.py:40-56`): 마스크 0/1 카운트(cls 포함/제외 둘 다).
- `mask_read_files` (`mask_utils.py:84-105`): npy 마스크를 layer별 `(1,heads,197,197)` tensor 리스트로 로드·검증.

### 3.9 프로파일링: `Profile/`

- `TX2_benchmark/benchmark.py` (`benchmark.py:38-80`): 모델→ONNX export→onnx-simplifier→**trtexec(TensorRT)**로 Jetson TX2 지연 측정, "end to end mean" 파싱 (`benchmark.py:71-79`). batch=1.
- `GPU_benchmark/benchmark.py`: torch profiler 기반(추정, 미정독).
- `Profile/models/vit.py`: einops 기반 표준 ViT(softmax 분해 분석용, `vit.py:41-60`). **ViTCoD 가속기와 무관한 baseline 측정용**.

---

## 4. 데이터 플로우

```
[학습/마스크 추출 단계 — Algorithm]
DeiT/LeViT 학습 (need_weight=True)
  → Attention.attention_sum 누적 (vision_transformer.py:441-443)
  → attnweights_utils.save → attention_score.npy (attnweights_utils.py:39)
  → gen_mask.py (info_cutoff 프루닝) → info_0.95.npy (gen_mask.py:86)
  → (auto-encoder 공동학습: recon_loss → engine.py:57-58)

[양극화 단계 — Hardware/Simulator]
info_0.95.npy
  → reorder.py (out-degree threshold로 global token 분리 + 재배열)
  → reodered_info_0.95.npy + global_token_info_0.95.npy (reorder.py:102-103)

[하드웨어 사이클 추정 단계 — Hardware/Simulator]
reodered_info + global_token
  → ViTCoD.py: dense_ratio로 PE 동적 분할 → SDDMM/SpMM = max(dense,sparse) → 어텐션 사이클
  → ViT_FFN.py: 선형 투영 + encoder 압축 + FFN → 선형/FFN 사이클
  → 수동 합산 = end-to-end latency (Simulator/README.md:38-50)

[검증 — Profile]
ViT → ONNX → trtexec(TX2) → 실측 지연 (benchmark.py)
```

---

## 5. HW/SW 매핑

| 알고리즘 개념 (SW) | 하드웨어 모델 (시뮬레이터) | 라인 근거 |
|---|---|---|
| info 기반 어텐션 프루닝 | sparse mask (희소 SDDMM/SpMM 대상) | `gen_mask.py:91-105`, `ViTCoD.py:78` |
| denser/sparser 양극화 | global token 분리 + 재배열 | `reorder.py:28-49`, `ViTCoD.py:76-79` |
| denser engine | dense_PE_width 분할 PE 묶음 | `ViTCoD.py:160, 164-167, 190-191` |
| sparser engine | sparse_PE_width 분할, row-wise CSR SpMM | `ViTCoD.py:161, 172-174, 223-225` |
| 동적 PE 할당(워크밸런싱) | dense_ratio 비례 분할 + max() 동시실행 | `ViTCoD.py:159-161, 176, 227` |
| 학습형 auto-encoder (head 압축) | on-chip encoder/decoder weight preload | `vision_transformer.py:237-240, 373-374` / `ViT_FFN.py:136-153`, `ViTCoD.py:98-115` |
| Q/K 압축 저장(데이터 이동 절감) | encoder 후 store_out (ratio=2/3) | `ViT_FFN.py:138-143`, `SRAM.py:33-41` |
| K-stationary 데이터플로우 | sparse QKᵀ에서 K 우선 적재 | `ViTCoD.py:123-148` |
| 메모리 vs 연산 bound | max(preload, compute) roofline | `ViTCoD.py:278`, `ViT_FFN.py:209-213` |
| PE array (64×64 / 64×8) | PE_width=64, PE_height=8 | `ViTCoD.py:18-19`, `PE.py:6`, `ViT_FFN.py:113` |
| HBM 76.8GB/s @500MHz | 대역폭→사이클 환산 | `SRAM.py:18-20` |

---

## 6. 빌드·실행

RTL/HLS 빌드 없음. 전부 Python 스크립트.

1. **마스크 양극화** (`Simulator/README.md:13-17`):
   ```
   cd Hardware/Simulator && python reorder.py   # reorder.py:69 경로 하드코딩 → 수정 필요
   ```
2. **어텐션 latency**:
   ```
   python ViTCoD.py --root masks/deit_tiny_lowrank --sparse 0.95 \
       --feature_dim 64 --ratio 0.667 --PE_width 64 --PE_height 8
   ```
   (`Simulator/README.md:22-32`)
3. **선형+FFN latency**:
   ```
   python ViT_FFN.py --root masks/deit_tiny_lowrank --sparse 0.95 \
       --feature_dim 64 --embedding 192 --ratio 0.667 --PE_width 64 --PE_height 8
   ```
   → 두 결과를 **수동 합산** (`Simulator/README.md:38-51`).
4. **알고리즘 학습/마스크 생성**: `Algorithm/deit/main.py` (DeiT 학습 CLI) + `gen_mask.py`. DeiT 의존성(`Algorithm/deit/requirements.txt`).
5. **프로파일**: `Profile/TX2_benchmark/benchmark.py` (trtexec 필요, Jetson 환경).

---

## 7. 의존성

- **시뮬레이터**: `numpy`, `scipy.sparse`(coo_matrix), `math`, `logging`, `argparse` (경량). `ViT_FFN.py`는 `torch` import하나 `embedding`만 사용(사실상 불필요, `ViT_FFN.py:4`).
- **reorder.py**: `dgl`(그래프), `torch`, `matplotlib` (무거움 — DGL 필요).
- **알고리즘**: `torch`, `timm`(fork 포함), DeiT/LeViT 스택 (`einops` 등).
- **프로파일**: `onnx`, `onnxsim`(onnx-simplifier), TensorRT `trtexec`, CUDA.

---

## 8. 강점 · 한계

**강점**
- **명확한 알고리즘-HW 공동설계 구현**: 프루닝(SW) → 양극화(reorder) → 동적 PE 분할(시뮬레이터)의 사슬이 코드로 일관되게 추적 가능.
- **dense_ratio 기반 동적 PE 할당 + max() 동시실행 모델**(`ViTCoD.py:159-176`)이 워크로드 밸런싱의 핵심을 간결히 포착.
- **학습형 auto-encoder가 실제 Linear+recon_loss로 end-to-end 구현**(SVD를 학습으로 대체)되어 재현 가능.
- 시뮬레이터가 경량(numpy)이라 sparsity/PE 파라미터 스윕에 적합.

**한계**
- **RTL/HLS 부재**: 실제 합성 가능한 하드웨어 기술(.v/.sv/.cpp)이 없고, **분석적 사이클 모델**만 존재. 면적/전력/실제 타이밍은 본 repo로 검증 불가(논문 별도 측정 추정).
- **시뮬레이터 정확도 가정**: SDDMM/SpMM을 ⌈work/throughput⌉ 단순 공식으로 추정. 파이프라인 stall, PE 채움 지연, 분할 경계 오버헤드 미반영(추정). `PE.py`의 cycle-accurate 경로는 **메인에서 미사용**(`ViTCoD.py:50`).
- **하드코딩 다수**: 경로(`reorder.py:69`), 차원 분모(`gen_mask.py:40-42`), mlp_ratio=4, 197 토큰, 8bit 고정 → DeiT-tiny 외 적용 시 수정 필요.
- **end-to-end가 수동 합산**: 어텐션/FFN을 따로 돌려 사람이 더함(자동 통합 스크립트 없음).
- **양자화 비트(8bit) 고정**(`SRAM.py` 전반 `bits=8`) — 정확도-비트 trade-off 미탐색.
- LeViT 쪽 ViTCoD 확장은 DeiT와 동형으로 **추정**(전 파일 정독 안 함).

---

## 9. 우리 프로젝트 시사점 (HG-PIPE 계열 고처리량 ViT FPGA 가속기 + XR 시선추적)

> 우리 프로젝트 추정: 고처리량 ViT/Transformer **FPGA(HLS/RTL) 가속기**(HG-PIPE 계열, 완전 파이프라인 지향) + XR 시선추적. ViTCoD는 **ASIC형 분석 시뮬레이터 + sparse 어텐션 co-design**으로 우리와 결이 다르지만 다음이 직접 유용하다.

1. **고정 sparse pattern 가정의 적합성** — XR 시선추적은 입력 토큰 수가 고정(고정 해상도 ROI)이라 ViTCoD의 "ViT는 고정 패턴으로 90% 프루닝 가능"(`README.md:15-21`) 전제가 그대로 성립할 가능성 높음. **오프라인 마스크 고정 → 런타임 예측기 불필요**는 FPGA 파이프라인에 이상적(동적 분기 회피, HG-PIPE의 정적 파이프라인과 궁합 좋음).

2. **denser/sparser 양극화 + 정적 재배열**(`reorder.py`) — 우리 RTL/HLS에서 동적 sparse를 다루기 어렵다면, ViTCoD처럼 **컴파일타임 재배열로 정형 dense 블록 + 작은 비정형 블록**으로 분리하면 HLS pragma(완전 언롤/파이프라인)를 dense 블록에 적용 가능. HG-PIPE의 완전 파이프라인에 **dense engine만 합성**하고 sparse는 별도 경로로 두는 분할 전략 차용 가능.

3. **동적 PE 분할은 우리에겐 부적합할 수 있음** — ViTCoD의 `dense_ratio` 런타임 PE 재할당(`ViTCoD.py:159-161`)은 reconfigurable PE를 전제. 완전 파이프라인 FPGA에서는 재구성 비용이 큼 → **고정 분할(dense:sparse 비율 컴파일타임 고정)**로 단순화 권장. ViTCoD 시뮬레이터를 우리 비율 탐색 도구로 재활용 가능(numpy라 이식 쉬움).

4. **auto-encoder(head 압축)로 데이터 이동 절감**(`vision_transformer.py:373-374`, `ViT_FFN.py:136-153`) — 우리 가속기가 대역폭 bound라면 Q/K를 2/3로 압축 저장하는 아이디어는 BRAM/HBM 트래픽 절감에 직접 적용 가능. 단 decoder Linear가 추가 연산이므로 FPGA DSP 예산과 trade-off 평가 필요.

5. **roofline식 max(preload, compute)** (`ViTCoD.py:278`) — 우리 HLS 설계에서 **II(initiation interval)와 메모리 대역폭 중 병목**을 같은 프레임으로 빠르게 추정하는 사전 모델로 활용. 우리 알고리즘-피드백 루프(Python→HLS)에서 합성 전 1차 스크리닝 도구로 좋음.

6. **차용하지 말 것 / 주의** — (a) ViTCoD엔 합성 가능한 RTL이 없으므로 **하드웨어 검증 근거로 인용 금지**(시뮬레이터 가정 기반). (b) 8bit 고정·DeiT-tiny 하드코딩은 우리 양자화/모델과 다름. (c) `PE.py` cycle-accurate 경로는 미사용이므로 신뢰 말 것.

7. **시선추적 특화 제안(추정)** — 시선추적은 latency-critical(저지연 요구). ViTCoD의 정적 마스크 + dense 우선 처리(diagonal 집중, `README.md:23`)는 **결정론적 저지연** 파이프라인에 부합. 우리는 sparse 경로를 아예 제거하고 **info_cutoff로 얻은 고정 마스크를 HLS에서 dense 블록 스케줄로 컴파일**하는 방향이 HG-PIPE 철학과 가장 잘 맞음(추정).

---

## 근거 표기 요약

- **확인됨(라인 근거)**: 양극화 로직(`reorder.py:28-49`), 동적 PE 분할(`ViTCoD.py:159-176`), max() 동시실행(`ViTCoD.py:176,227,278`), auto-encoder Linear(`vision_transformer.py:237-240,373-374`), recon_loss 통합(`engine.py:57-58`), info 프루닝(`gen_mask.py:91-105`), SRAM 스펙(`SRAM.py:12-20`), trtexec 프로파일(`benchmark.py:71-79`).
- **추정**: PE_height=8의 systolic 의미, encoder 버퍼 미모델링 이유, mix_head SVD가 auto-encoder로 대체된 의도, LeViT 확장 동형성, GPU_benchmark 내부.
- **확인 불가(본 repo 범위 밖)**: 실제 면적/전력/주파수 합성 결과, 가속기 RTL 구현(논문 별도 자료 추정).
