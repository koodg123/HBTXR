# ViTCoD 코드베이스 정밀 분석

> 분석 대상 repo: `REF/ViT-Accelerator/ViTCoD`
> 분석 일자 기준 코드베이스 스냅샷. 본 문서의 모든 라인 근거는 실제 소스 기준이며, 코드로 확인 불가한 사항은 "추정"/"확인 불가"로 명시한다.

---

## 1. 개요

- **프로젝트명**: ViTCoD — *Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design*
- **원논문**: You et al., **HPCA 2023** (IEEE International Symposium on High-Performance Computer Architecture, HPCA-29). arXiv:2210.09573. 공식 GitHub: GATECH-EIC/ViTCoD. (루트 `README.md` L1-L8, L59-L64에서 확정. "추정"이 아니라 코드베이스 README가 직접 명시)
- **한줄요약**: ViT의 어텐션 맵을 **고정 희소 패턴으로 프루닝 + 양극화(polarize)** 하여 "denser 영역(=global token)"과 "sparser 영역"으로 분할하고, 전용 **denser/sparser 엔진**을 동적으로 PE를 나눠 동시 처리하며, on-chip **encoder/decoder(auto-encoder)** 로 Q/K를 압축해 데이터 이동을 줄이는 ViT 전용 알고리즘+가속기 co-design 프레임워크.
- **목적**: ViT의 어텐션 맵은 입력 토큰 수가 고정적이어서 정확도 손실 거의 없이(예: 90% 프루닝에 <=1.5% 정확도 하락) 고정 희소 패턴으로 90%까지 프루닝 가능하다는 점(README L15)을 활용. 단, 높은 희소도가 야기하는 워크로드 불균형/저활용 문제(비영(non-zero) 원소가 대각선에 몰림, README L23)를 HW에서 해소.
- **타깃 — 핵심 판별**:
  - **하드웨어는 "사이클 시뮬레이터(analytical/cycle-estimation simulator)" 기반.** RTL(.v/.sv), HLS(.cpp/.h/.cl) 소스는 **존재하지 않음**(섹션 2의 Glob 결과로 확인). `Hardware/Simulator`는 순수 Python로 작성된 **사이클 카운트 추정 모델**이며, 실제 합성 가능한 ASIC/FPGA RTL이 아니다.
  - 논문상으로는 dedicated ASIC accelerator를 주장하지만(README L34: "develop a dedicated accelerator", "on-chip encoder and decoder engines"), **본 repo에 공개된 것은 그 가속기의 동작을 모사하는 cycle-level 분석 시뮬레이터뿐**이다. 따라서 본 repo는 "ASIC을 가정한 시뮬레이터 + ViT 알고리즘(PyTorch)" 조합으로 보는 것이 정확하다.
  - SRAM/HBM 파라미터(`SRAM.py`)는 ASIC급 메모리/대역폭(SRAM 53KB/108KB, HBM 76.8GB/s, 500MHz)을 가정 → 타깃은 **ASIC 가속기(시뮬레이션 상)** 로 추정.

---

## 2. 디렉토리 구조 (자체 소스 트리)

ViTCoD는 README L44-L50대로 3축으로 구성된다.

```
ViTCoD/
├── README.md                         # 프로젝트 개요 (HPCA 2023)
├── Algorithm/                        # [SW] ViT 모델 + sparse attention 프루닝/auto-encoder (PyTorch)
│   ├── deit/                         # DeiT 기반 (facebookresearch/deit fork)
│   │   ├── README.md                 # 4-step 워크플로(low-rank finetune→attn map→gen mask→sparse finetune)
│   │   ├── requirements.txt          # torch==1.7.0, torchvision==0.8.1, timm==0.3.2
│   │   ├── main.py                   # 학습/평가 엔트리 (torch.distributed.launch)
│   │   ├── models.py                 # deit_tiny/small/base_patch16_224 등 모델 팩토리
│   │   ├── engine.py                 # train/eval 루프
│   │   ├── gen_mask.py               # ★ 어텐션 마스크 생성(info/ratio/random/std)
│   │   ├── mask_utils.py             # ★ 마스크 로드/통계/FLOPs 절감 계산
│   │   ├── attnweights_utils.py      # 어텐션 가중치 누적/평균
│   │   ├── read_attn_map.py / plot_attn_map.py / run_visualize_sparsity.py
│   │   ├── cal_flops.py / test_calc_flops.py / vision_transformer_flop.py
│   │   ├── cait_models.py / resmlp_models.py  # 비교 모델
│   │   ├── run_deit_tiny_svd_sparse.sh # 실행 스크립트(info_0.5~0.95 sparse finetune)
│   │   └── timm/
│   │       ├── vision_transformer.py # ★★ 수정된 ViT: low-rank/SVD attention + encoder/decoder + mask
│   │       ├── mask_utils.py / utils.py
│   ├── levit/                        # LeViT 기반 (deit와 유사 구조; README는 미완성 "Will update ASAP")
│   │   ├── levit.py / levit_c.py / main.py / engine.py
│   │   ├── gen_mask.py / gen_test_masks.py / mask_utils.py
│   │   ├── attnweights_utils.py / read_attn_map.py / speed_test.py ...
├── Hardware/
│   └── Simulator/                    # ★★★ 가속기 사이클 시뮬레이터 (핵심, 순수 Python)
│       ├── README.md                 # reorder→ViTCoD(attn)→ViT_FFN 실행 절차
│       ├── ViTCoD.py                 # ★★★ 어텐션 사이클 시뮬 (denser/sparser 엔진 동적 분할)
│       ├── ViT_FFN.py                # ★★ 선형투영/MLP(FFN) 사이클 시뮬 (end-to-end 보완)
│       ├── reorder.py                # ★★ 어텐션 맵 양극화(reorder/polarize) - DGL 기반 global token 추출
│       ├── PE.py                     # ★ PE 어레이 모델(64×64, MAC 사이클)
│       ├── SRAM.py                   # ★ SRAM/HBM 메모리 모델(preload/store 사이클)
│       └── masks/deit_tiny_lowrank/  # info_0.95.npy, reodered_info_0.95.npy, global_token_info_0.95.npy
└── Profile/                          # [SW] FLOPs/latency 프로파일링 (GPU & Edge GPU)
    ├── README.md
    ├── GPU_benchmark/benchmark.py, op_profile.sh
    ├── TX2_benchmark/benchmark.py, parse_json.py, op_profile.sh, print_softmax_percentage.sh
    └── models/vit.py, models/linformer/  # 벤치마크용 ViT/Linformer 정의
```

**RTL/HLS 유무 (Glob 확인 결과)**:
- `**/*.v`, `**/*.sv`, `**/*.cpp`, `**/*.h`, `**/*.cl` → **모두 "No files found"**. → **RTL/HLS 소스 없음. 하드웨어는 시뮬레이터(Python) 기반으로 확정.**
- `Hardware/Simulator/figs/arch.png` 등 아키텍처 다이어그램 이미지가 README에서 참조됨(L3) → 실제 마이크로아키텍처 도식은 그림으로만 제공되며 RTL은 비공개("확인 불가").

**제외물(이름만 언급)**: `Algorithm/deit/`의 사전학습 체크포인트(*.pth, Google Drive 링크로 외부 호스팅, README L5-L6), ImageNet-2012 데이터셋(`--data-path`로 외부 참조), `masks/`의 npy 마스크/체크포인트류는 분석 대상에서 제외(이름만 언급).

---

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 Hardware/Simulator — 가속기 사이클 시뮬레이터 (최우선)

ViTCoD의 핵심 HW 기여가 응축된 부분. 4개 파일(`SRAM.py`, `PE.py`, `reorder.py`, `ViTCoD.py`)과 보조 `ViT_FFN.py`로 구성되며, **합성 가능한 RTL이 아니라 cycle-count를 산술적으로 누적하는 analytical model**이다.

#### 3.1.1 `SRAM.py` — 메모리 계층 사이클 모델

ASIC급 on-chip SRAM + off-chip HBM을 가정한 메모리 모델. 핵심 상수(L12-L19):

- `max_Q = max_K = max_V = 53 * 1024 * 8` (각 53KB, 비트 단위) — Q/K/V 전용 SRAM 뱅크.
- `max_index = 20 * 1024 * 8` (20KB) — 희소 인덱스(coo) 저장용.
- `max_output = 108 * 1024 * 8` (108KB) — 출력/중간결과 버퍼.
- `bandwidth = 76.8 * 1024^3 * 8` (76.8GB/s, HBM→SRAM).
- `clock_frequency = 500 * 1e6` (500MHz).

사이클 산식은 전 메서드 공통(L28-L29 등):
```
latency = nums * bits / (bandwidth * bandwidth_ratio)
cycle   = ceil(latency * clock_frequency)
```
즉 **데이터 양(원소수×비트수)을 유효 대역폭으로 나눠 시간을 구하고, 클럭을 곱해 사이클로 변환**. `bandwidth_ratio`(<1)로 멀티헤드 병렬/압축 효과를 모델링한다(예: `1/head`로 헤드 분할, `1/(head*ratio)`로 압축 후 대역 절감).

메서드별 역할:
- `preload_Q/K/V`(L43-L76): HBM→SRAM Q/K/V 적재. 각 max_* 초과 시 에러 출력(`exit()`는 주석처리되어 실제 종료는 안 함 — L24-L25 등, 즉 **용량 초과는 경고만 하고 무시되는 한계**).
- `preload_decoder/preload_encoder`(L23-L41): on-chip auto-encoder(decoder)/encoder 가중치 적재.
- `preload_index`(L78), `store_out`(L88), `preload_weight`(L99): 인덱스/출력/선형층 가중치.
- 모든 용량 체크가 `max_Q`를 잘못 참조하는 케이스 존재(`preload_decoder`/`preload_encoder`가 `self.max_Q` 검사, `store_out`/`preload_weight`가 `self.max_output` 검사) — 일부는 의도, 일부는 복붙으로 추정.

#### 3.1.2 `PE.py` — PE 어레이 (연산 사이클 모델)

`PE_array` 클래스(L3). **64×64 PE 어레이**를 `self.res = [[0]*64]*64`로 표현(L6 주석 "64 x 64 PE array").

- `cal_attn_map`(L23-L36): Q·K 내적(SDDMM의 1원소) — `for k in range(len(Q)): res[0][0] += Q[k]*K[k]; cycle += 1` → 내적 길이만큼 사이클.
- `cal_V_update`(L38-L53): attn·V (SpMM) — 3중 루프로 MAC당 1사이클.
- `store_res_V`(L62-L64): `len(attn)*len(V)//64` (64-way 병렬 가정).

**중요**: `PE.py`는 실제 시뮬레이션 메인 루프(`ViTCoD.py`)에서 import는 되나(`from PE import PE_array`, L5; `my_PE = PE_array()`, L50) **연산 메서드는 거의 호출되지 않는다**. 실제 사이클은 `ViTCoD.py` 내부에서 `PE_width`/`PE_height` 기반 산술 루프로 직접 카운트된다. 즉 `PE.py`는 **참조/명세용 모델**이고, 본 사이클 산정은 `ViTCoD.py`가 담당(이는 라인 근거로 확인 가능한 사실).

#### 3.1.3 `reorder.py` — 어텐션 맵 양극화(polarize/reorder)

ViTCoD 알고리즘-HW 접점의 핵심. 희소 어텐션 마스크(`info_0.95.npy`)를 입력받아 **denser 영역(global token)과 sparser 영역으로 재배열**하고 두 산출물을 저장.

- 그래프 기반 분석(`dgl` 사용, L3-L4). `calc(graph, ax, threshold)`(L8-L59):
  - 마스크의 0(=연결/non-mask)을 edge로 DGL 그래프 구성(L13-L20).
  - `out_deg = g.out_degrees()`로 각 토큰(노드)의 out-degree 계산(L28).
  - `high_density = out_deg[out_deg > threshold]` → **out-degree가 threshold를 넘는 "고밀도 토큰(=global token)" 식별**(L29-L32).
  - 고밀도 토큰을 행렬 앞쪽으로 **재배열(permute)** — `orig_a`/`orig_b` 인덱스 스왑(L42-L49)으로 dense 토큰을 컬럼 0..(total-1)로 모음.
  - 반환: `dense_cnt`(dense edge 수), `total_cnt`, `new_graph`(재배열된 마스크), `total`(= **global token 수**).
- 메인(L69-L104): `info_0.95.npy` 로드 → 각 (layer, head)마다 `calc(..., threshold=50)` 호출 → `reodered_info_0.95.npy`(재배열 마스크)와 `global_token_info_0.95.npy`(헤드별 global token 수)를 저장(L102-L103).
- **threshold=50**(L93)이 denser/sparser를 가르는 임계: out-degree>50인 토큰을 denser(global)로 분류.
- README L13-L17과 정합: "polarize ... to be either denser or sparser", 산출물 `global_token_info_*.npy`(denser 토큰 수)와 `reodered_info_*`(sparser 패턴).
- **한계 근거**: L69의 하드코딩 경로(`/home/sheminghao/ViTCoD/...`)는 원저자 로컬 경로 → 그대로 실행 불가(수정 필요). `dgl` 의존(L3).

#### 3.1.4 `ViTCoD.py` — 어텐션 사이클 시뮬레이터 (핵심 중 핵심)

denser/sparser 동시 처리, 동적 PE 분할, encoder/decoder 데이터 절감을 사이클로 모델링하는 메인.

**입력/초기화**(L41-L50):
- `reodered_info_*.npy`(재배열 마스크), `global_token_info_*.npy`(헤드별 global token 수) 로드.
- Q/K/V는 실제 값이 아니라 `np.random.random((layer, head, token, feature_dim))` — **사이클만 추정하므로 값은 무의미**(L45-L47). feature_dim 기본 64.
- 인자(L12-L20): `--ratio`(encoder/decoder 압축비, 기본 2/3), `--PE_width`(64), `--PE_height`(8). PE 어레이는 64×8 구성으로 시뮬(README 실행 예시 L29-L30).

**메인 루프(layer×head)**(L65-L67): `for _head in range(head//head)` = **head를 1개만 순회**(=head 병렬을 가정하고 대표 1헤드만 계산, 나머지는 병렬로 처리된다고 모델링). `head` 변수는 대역폭/PE 분할 식에서 곱셈 인자로 사용.

**(a) denser/sparser 분리**(L75-L87):
- `global_tokens = num_global_tokens[layer, head]` (reorder 산출).
- `sparser = coo_matrix(1 - mask[:, global_tokens:])` → global token 이후 컬럼만 떼어 **희소(sparser) 영역의 non-zero 좌표(row,col)** 추출(L78-L79).
- `sparse_ratio = len(sparser)/(...)` 로 sparser 영역 밀도 계산(L83).

**(b) 데이터 적재 + on-chip decoder 전처리 (denser, Q·K)**(L92-L120):
- global token마다 `preload_K`(압축비 `head*ratio` 반영, L98)와 첫 회 `preload_decoder`(L100) 적재.
- 전처리(decode) 사이클 `PRE_cycles`는 `ceil((head*ratio*K.shape[1]) / (PE_width*PE_width/head))`로 누적(L102-L103) → **압축된 K를 PE에서 decoder로 복원하는 비용**.
- Q도 동일 패턴(L104-L115). `reload_ratio`는 SRAM 용량 한계 시 재적재 비용(여기선 0으로 단순화, L109).

**(c) 데이터 적재 (sparser, Q·K) — K-stationary**(L122-L152):
- 주석 L123: "K-stationary (Why? Because the number of global tokens vary a lot → Score stationary is not best fit)" — **헤드마다 global token 수가 크게 달라 score-stationary 대신 K-stationary 데이터플로 채택**.
- sparser 영역 K를 토큰별 적재(L128-L134). Q의 `reload_ratio`는 SRAM(max_K) 대비 시퀀스 길이로 계산(L138-L141), 캐시 미스 재적재를 모델링.

**(d) ★ 동적 PE 분할 (denser/sparser 엔진)**(L159-L177):
```python
dense_ratio = global_tokens*Q.shape[0] / (len(sparser) + global_tokens*Q.shape[0])   # L159
dense_PE_width  = int(args.PE_width * dense_ratio)                                    # L160
sparse_PE_width = args.PE_width - dense_PE_width                                      # L161
```
→ **denser/sparser 워크로드 비율에 비례해 64-wide PE를 동적으로 두 엔진에 분배.** denser 작업량(global_tokens×토큰수)과 sparser 작업량(non-zero수)의 비로 PE 폭을 나눔 = README L32의 "dynamic PE allocation between the denser and sparser engines"를 직접 구현한 부분.
- denser SDDMM 사이클(L163-L168): `for global_tokens × ceil(Q.shape[0]/dense_PE_width) × ceil(Q.shape[1]/(PE_width/head))`.
- sparser SDDMM 사이클(L170-L175): `for ceil(len(sparser)/sparse_PE_width) × ceil(Q.shape[1]/(PE_width/head))`.
- **두 엔진은 동시 동작 → 사이클은 max()로 합산**(L176): `SDDMM_PE_cycles = max(dense_SDDMM_PE_cycles, sparse_SDDMM_PE_cycles)`. (병렬 실행 모델링)

**(e) S·V (SpMM) 단계**(L182-L228):
- denser SpMM(L189-L193): V 적재 후 `ceil((V원소수×global_tokens)/(dense_PE_width*PE_width/head))`.
- sparser SpMM(L196-L228): `sparser` 좌표를 **row별 non-zero 개수(num_list)로 압축**(L197-L212, run-length 유사 누적) → row마다 `row_num*V.shape[1]` 사이클, 최종 `sparse_PE_width`로 나눔(L223-L225). **CSR/row-wise 희소 연산 모델.**
- 역시 `SpMM_PE_cycles = max(sparse, dense)`(L227)로 두 엔진 병렬 합산.

**(f) 총합**(L269-L279):
- preload(데이터이동)/PRE(decoder전처리)/Computation(SDDMM+SpMM)을 각각 누적.
- 최종 `Total cycles = max(total_preload_cycles, total_PRE_cycles + total_SDDMM + total_SpMM)`(L278) → **데이터 이동과 연산을 오버랩(겹침)했다고 가정하고 더 큰 쪽이 bottleneck**. = roofline 식 latency 추정.

**시뮬레이터의 본질(라인 근거 종합)**: 실제 데이터를 흘리는 RTL 시뮬이 아니라, **워크로드 차원(layer/head/token/feature) × HW 파라미터(PE_width/height, 대역폭, 압축비)로 사이클을 산술 누적**하는 analytical performance model. denser/sparser 동적 분할(L159-L161), 두 엔진 병렬(max, L176/L227), encoder/decoder 압축(ratio 2/3 → 대역/전처리 절감)이 핵심 모델링 요소.

#### 3.1.5 `ViT_FFN.py` — 선형투영 + FFN(MLP) 사이클 시뮬

어텐션 외 나머지(QKV 임베딩 선형층, multi-head concat 후 proj, FFN 2-layer)를 별도 시뮬해 **end-to-end latency를 보완**(README L35-L51: 어텐션 사이클 + FFN 사이클 합산).

- 임베딩/QKV 선형층(L107-L132): Q/K/V마다 가중치 `preload_weight`(L111) + `SDDMM_PE_cycles += embedding*shape/(PE_width*PE_height/head)`(L113-L114). **PE는 64×8(width×height)** 로 한 사이클당 처리량 산정.
- encoder를 통한 Q/K **압축 후 저장**(L136-L153): `preload_encoder` + 압축 store(`bandwidth_ratio=1/(head*ratio)` → 압축으로 저장 대역 절감).
- multi-head concat(L161-L166), FFN(L180-L191): `embedding*embedding*4`(4× hidden 확장) 2개 GEMM을 `PE_height*PE_width`로 나눠 카운트.
- 총합(L209-L213): `linear = max(연산+전처리, preload)`, `ffn = max(연산, preload)`, `total = linear + ffn`.

### 3.2 Algorithm — Sparse Attention 프루닝 + Auto-encoder (PyTorch)

#### 3.2.1 `timm/vision_transformer.py` — 수정된 ViT 어텐션 (★★)

표준 timm ViT를 fork해 **(1) low-rank/auto-encoder 어텐션, (2) 마스크 적용, (3) 어텐션 맵 누적**을 추가. `Attention`(L209) 클래스가 핵심.

- 생성자(L209-L249): 표준 QKV(L219) 외에 `svd_type`에 따라 **헤드 차원 압축 auto-encoder** 구축:
  - `mix_head_fc_qk`(L235-L240): `hidden = num_heads//2`(+1). `encoder_q = Linear(num_heads, hidden)`, `decoder_q = Linear(hidden, num_heads)`(K도 동일). → **헤드 수를 절반으로 압축(encoder)했다가 복원(decoder)**. README의 "lightweight learnable auto-encoder"가 바로 이 FC 기반 인코더/디코더. (단, SVD라는 이름과 달리 실제 채택 경로 `mix_head_fc_qk`는 SVD가 아니라 학습형 FC auto-encoder)
  - `mix_head_fc_q`/`mix_head_fc_k`: Q만/K만 압축.
- forward — `mix_head_fc_qk` 경로(L359-L386):
  ```python
  q = q.view(b, h, -1).permute(0,2,1)          # b x (nd) x h  : 헤드축을 마지막으로
  pre_q = q
  q = self.decoder_q(self.encoder_q(q))        # h → hidden → h  압축-복원
  self.recon_loss = dist(q,pre_q)+dist(k,pre_k)# 재구성 손실(학습 시 정규화)
  ```
  → **헤드 간 상관을 hidden 차원으로 압축**해 데이터 이동을 줄임. `recon_loss`(L376)는 학습에 반영(auto-encoder가 정보를 잃지 않도록).
- 다른 svd_type: `single_head`(L260-)는 `torch.svd_lowrank(q, q=100)`로 토큰축 SVD 저랭크 근사(L277-L285), `mix_head`(L299-)는 `torch.pca_lowrank(q=6)`(L333-L334). → **연구 과정의 여러 압축 변형**이 남아있으나 실제 권장(README/스크립트)은 `mix_head_fc_qk`.
- **마스크 적용**(L434-L438): `attn_mask`가 있으면 `attn.masked_fill(attn_mask.bool(), -inf)` → softmax 전에 마스크 위치를 -inf로 → **고정 희소 패턴 강제(프루닝)**. 이 마스크가 곧 HW 시뮬레이터의 입력 npy.
- **어텐션 맵 누적**(L231-L233, L441-L443): `need_weight`면 `attention_sum += attn.sum(0)`, `num_attention += B` → 학습데이터 전체의 **평균 어텐션 맵 산출**(gen_mask 입력).
- 모델 레벨 `_generate_patterns`(L750-L777): `mask_utils.mask_read_files`로 npy 마스크를 레이어별 텐서로 로드해 각 블록에 주입(L705-L715).

#### 3.2.2 `gen_mask.py` — 어텐션 마스크 생성 (★)

평균 어텐션 맵 → 희소 마스크 npy 생성. 4가지 method(L132-L152), 논문 채택은 `info`(README deit L83).

- **`gen_info_based_mask`**(L51-L88) + **`info_cutoff`**(L91-L105): 각 토큰의 어텐션 분포를 내림차순 정렬 후 **누적합이 `info`(예: 0.58) 이상이 될 때까지의 상위 원소만 유지**, 나머지를 마스크. → **정보량(누적 어텐션 확률) 기준 컷오프**. cls 토큰 행/열은 보존(L67-L68: `temp[0,:]=0; temp[:,0]=0`).
- `gen_ratio_based_mask`(L22-L47): 단순 상위 비율(rank) 컷오프.
- `gen_random_mask`(L7-L19): `scipy.sparse.random`으로 랜덤(베이스라인).
- `gen_std_based_mask`(L108-L129): 평균+계수×표준편차 임계.
- sparsity 산정은 모델별 (layer×head×197×197)로 정규화(L77-L84, base=12, small=6, tiny=3 layers).

#### 3.2.3 `mask_utils.py` — 마스크 통계/FLOPs 절감 (보조)

- `cal_reduced_Gflops`(L8-L23): 마스크 비율로 절감 GFLOPs 계산 `2*ratio*N*N*C/1e9 * L`.
- `mask_read_files`(L84-L105): npy(num_layers,heads,N,N)를 레이어별 텐서로 로드, 크기 검증 후 반환 → vision_transformer가 사용.

#### 3.2.4 학습 워크플로 (`deit/README.md` 4-step)

1. **Low-rank finetune**: 사전학습 DeiT를 `--svd_type mix_head_fc_qk`로 finetune(auto-encoder 삽입). (deit base 평가 Acc@1 81.576)
2. **평균 어텐션 맵 생성**: `--need_weight`로 학습데이터 어텐션 누적 → npy.
3. **마스크 생성**: `gen_mask.py --method info --info_cut 0.58`.
4. **Sparse finetune**: `--mask_path`로 마스크 주입 후 재학습(`--restart_finetune`). (90% sparse 평가 Acc@1 80.720, 약 0.85%p 하락)

### 3.3 Profile — FLOPs/Latency 프로파일링 (SW 측정)

- `GPU_benchmark/benchmark.py`: `torch.profiler`로 batch size를 OOM 직전까지 키워(L100-L122) **throughput(FPS)/메모리/op별 시간** 측정(L42-L98). ViT/Linformer 구조를 `seq_len/dim/heads`로 파라미터화(`models/`).
- `TX2_benchmark/`: 동일 측정을 **NVIDIA Jetson TX2 Edge GPU**에서 수행(`parse_json.py`로 결과 파싱, `print_softmax_percentage.sh`로 softmax 비중 추출). → 논문의 **GPU/Edge GPU baseline 대비 ViTCoD 가속기 비교**용.

---

## 4. 데이터플로우 / 실행 흐름

**알고리즘(오프라인) → 마스크/모델 → HW 시뮬(사이클 추정)** 의 단방향 파이프라인.

```
[Algorithm/deit]
 (1) DeiT + auto-encoder(encoder/decoder_q,k) low-rank finetune  (vision_transformer.py)
 (2) need_weight로 평균 attention map 누적 → attn npy           (attention_sum/num_attention)
 (3) gen_mask.py (info_cutoff: 누적 어텐션 ≥ info 까지만 유지)   → info_0.95.npy (sparse mask)
 (4) mask로 sparse finetune (masked_fill -inf)                  → 정확도 검증
           │
           ▼ (info_*.npy 를 Hardware/Simulator/masks/ 로 전달)
[Hardware/Simulator]
 (5) reorder.py: DGL out-degree>threshold(50)로 global(denser) 토큰 식별
                 → 행렬 앞으로 재배열
                 → reodered_info_*.npy (sparser 패턴) + global_token_info_*.npy (denser 토큰 수)
 (6) ViTCoD.py: 마스크를 denser(global cols) / sparser(coo non-zeros)로 분할
                → dense_ratio로 PE 폭 동적 분배 (denser/sparser 엔진)
                → SDDMM(Q·K) / SpMM(S·V) 각각 두 엔진 max() 병렬 합산
                → encoder/decoder(ratio=2/3) 압축으로 preload/PRE 사이클 절감
                → Total = max(preload, PRE+SDDMM+SpMM)   [어텐션 latency]
 (7) ViT_FFN.py: 선형투영+FFN 사이클                          [나머지 latency]
 (8) 어텐션 latency + FFN latency = end-to-end 사이클
```

- **어텐션 맵 분할(sparser/denser)**: `reorder.py`가 out-degree로 polarize(고밀도 토큰=denser/global, 나머지=sparser). `ViTCoD.py`가 이를 컬럼 슬라이싱(`mask[:, :global]` vs `mask[:, global:]`)으로 실제 분할.
- **dedicated engine 매핑**: dense_ratio로 PE_width를 denser/sparser 엔진에 동적 배분 → 두 엔진 동시 실행(max).
- **양자화 스킴**: 시뮬레이터 전반 **8-bit**(`bits=8` 고정, SRAM.py 호출부; INT8 가정). 알고리즘 측 학습은 FP(양자화 학습 코드는 본 repo에 없음 — "확인 불가").
- **희소화 스킴**: 고정 패턴(fixed sparse) info-cutoff 프루닝(동적 예측 불필요, README L21). HW는 sparser 영역을 COO/row-wise(num_list)로 처리.
- **메모리 계층**: HBM(76.8GB/s) → on-chip SRAM(Q/K/V 각 53KB, output 108KB, index 20KB) → PE(64×8) 레지스터. encoder로 Q/K를 압축 저장(2/3)해 SRAM/대역 절감.

---

## 5. HW/SW 매핑

| 알고리즘(SW, PyTorch) | 하드웨어 시뮬레이터(Python) | 비고 |
|---|---|---|
| `Attention.encoder_q/decoder_q`(헤드 압축 auto-encoder) | `SRAM.preload_encoder/decoder` + `PRE_cycles`(ViTCoD.py L100-L115, ViT_FFN.py L136-L153) | `ratio=2/3`로 압축 효과 모델링 |
| `attn_mask` masked_fill(-inf) → info 프루닝 | `reorder.py` polarize → `ViTCoD.py` denser/sparser 분할 | npy 마스크가 SW↔HW 인터페이스 |
| Q·K (SDDMM) | `dense/sparse_SDDMM_PE_cycles`(L163-L176) | 동적 PE 분할 |
| softmax·V (SpMM) | `dense/sparse_SpMM_PE_cycles`(L189-L227) | row-wise 희소 |
| QKV/proj/FFN 선형층 | `ViT_FFN.py` linear/ffn 사이클 | end-to-end 보완 |
| GPU/TX2 실측 latency | (baseline 비교용) `Profile/` | HW 시뮬 결과와 대조 |

**RTL 부재의 한계(중요)**:
- 본 repo에는 **합성 가능한 RTL/HLS가 없다.** 따라서 면적/전력/타이밍(주파수 달성 여부)은 **코드로 검증 불가** — 500MHz/76.8GB/s/SRAM 용량은 시뮬레이터의 *가정값*일 뿐 실제 구현으로 입증되지 않음("확인 불가").
- 동적 PE 분할/이중 엔진/COO 디코더의 실제 컨트롤 로직(reconfigurable interconnect, 워크로드 밸런싱 FSM)은 시뮬레이터에 추상화되어 있고 RTL로는 미공개.
- 사이클 모델은 데이터 이동/연산 완전 오버랩(max), 무손실 압축, 재적재 단순화(reload_ratio 0/1) 등 **낙관적 가정**이 포함됨.

---

## 6. 빌드·실행 방법

**Algorithm (deit)** — `requirements.txt`: torch==1.7.0, torchvision==0.8.1, timm==0.3.2. timm의 `vision_transformer.py`를 repo 버전으로 교체하고 `mask_utils.py`/`utils.py`를 `timm/models/`에 추가해야 함(deit README L27-L28).
```
# low-rank finetune
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --model deit_base_patch16_224 --resume <deit.pth> --data-path <imagenet> \
  --svd_type 'mix_head_fc_qk' ...
# 어텐션 맵 → 마스크
python main.py --eval --need_weight --output_dir <attn> ...
python gen_mask.py --method info --attn <attn.npy> --info_cut 0.58 --output_dir <mask>
# sparse finetune
python -m torch.distributed.launch ... --mask_path <mask.npy> --restart_finetune
# 일괄: run_deit_tiny_svd_sparse.sh (info_0.5/0.8/0.9/0.95)
```

**Hardware/Simulator** (Simulator README L13-L51):
```
python reorder.py                          # 양극화 (단, L69 경로 하드코딩 수정 필요)
python ViTCoD.py --root masks/deit_tiny_lowrank --sparse 0.95 \
  --feature_dim 64 --ratio 0.667 --PE_width 64 --PE_height 8     # 어텐션 사이클
python ViT_FFN.py --root masks/deit_tiny_lowrank --sparse 0.95 \
  --feature_dim 64 --embedding 192 --ratio 0.667 --PE_width 64 --PE_height 8  # FFN 사이클
# 두 결과(총 사이클)를 합산 → end-to-end latency
```
결과는 `--root` 폴더의 `vitcod_atten_0.95_wo.txt`/`vitcod_atten_ffn.txt`에 로깅(ViTCoD.py L32, ViT_FFN.py L43).

**Profile**:
```
python GPU_benchmark/benchmark.py --model <m> --seq_len <> --dim <> --heads <> [--enable_op_profiling]
# TX2_benchmark/op_profile.sh 로 Jetson TX2 측정
```

---

## 7. 의존성

- **공통**: numpy.
- **Algorithm**: PyTorch 1.7.0, torchvision 0.8.1, **timm 0.3.2**(필수, 코드 교체 방식), scipy(gen_mask), matplotlib(plot류), `torch.distributed`(멀티 GPU). DeiT(facebookresearch/deit) fork, cait/resmlp 모델 포함.
- **Hardware/Simulator**: numpy, scipy(`coo_matrix`), **dgl**(reorder.py 그래프 분석), torch(reorder.py), matplotlib. `SRAM`/`PE`는 자체 모듈.
- **Profile**: torch, `torch.profiler`, `torch.utils.tensorboard`, json/csv. linformer 모델 포함.
- 외부 호스팅(제외): 사전학습 *.pth 체크포인트(Google Drive), ImageNet-2012, masks npy.

---

## 8. 강점 / 한계 / 리스크

**강점**
- ViT 고유 특성(고정 토큰 수)을 정조준한 **fixed sparse + polarize** 설계 — 동적 예측 불필요, HW 단순화.
- **denser/sparser 동적 PE 분할**(dense_ratio)로 워크로드 불균형/저활용 문제를 정량적으로 완화하는 명확한 모델.
- **학습형 auto-encoder(encoder/decoder)** 로 데이터 이동을 연산으로 치환(메모리 바운드 완화)하는 알고리즘-HW 공동 설계가 코드로 일관되게 구현됨.
- 알고리즘(정확도)·시뮬레이터(사이클)·프로파일(실측 baseline) 3축이 명확히 분리되어 재현/확장 용이.

**한계**
- **RTL/HLS 부재** → 면적/전력/주파수 미검증, 실제 구현 가능성은 시뮬레이션 가정에 의존(가장 큰 한계).
- 시뮬레이터의 낙관적 가정: 완전 오버랩(max 합산), SRAM 용량 초과 시 경고만 하고 무시(`exit()` 주석처리), `reload_ratio` 단순화, head 1개만 순회(head 완전 병렬 가정).
- `reorder.py` 하드코딩 경로(L69) → 그대로 실행 불가.
- `PE.py`는 명세용이고 실제 사이클은 `ViTCoD.py` 산술로 산정 → 모델 일관성 추적이 코드 분산.
- LeViT README 미완성("Will update ASAP"), 일부 svd_type(single_head/mix_head)은 실험 잔재.

**리스크**
- 8-bit 양자화는 시뮬레이터 가정(bits=8)일 뿐 알고리즘 측 양자화 학습 코드 부재 → 정확도-양자화 동시 검증 안 됨.
- 시뮬레이터 사이클 수치를 그대로 칩 성능으로 해석하면 과대평가 위험.

---

## 9. 우리 프로젝트 관점 시사점 (고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적)

> HG-PIPE 계열(고처리량 파이프라인 FPGA ViT 가속기) + XR eye-tracking ViT 맥락에서의 재사용 포인트.

1. **Sparse attention 분할(denser/sparser) → FPGA 이중 엔진 매핑**
   - ViTCoD의 `dense_ratio` 기반 PE 분할(ViTCoD.py L159-L161)을 FPGA에서는 **컴파일 타임 고정 분할**로 단순화 가능. XR eye-tracking은 입력 해상도/토큰 수가 고정(웨어러블 카메라) → ViT보다 더 정적 → **dense/sparse 엔진 폭을 합성 시 상수로 박아 reconfigurable interconnect 비용 제거**. HG-PIPE의 stage별 파이프라인에 dense 엔진(systolic)과 sparse 엔진(row-wise/COO)을 별도 stage로 배치.

2. **고정 희소 패턴(info-cutoff)의 FPGA 친화성**
   - 동적 sparsity 예측 회로가 불필요(README L21) → FPGA에서 **마스크를 BRAM 상수/비트맵으로 사전 저장**, masked GEMM을 스킵하는 단순 게이팅으로 구현. XR은 시선 영역(fovea) 중심으로 어텐션이 더 집중되므로 info-cutoff 프루닝이 ViT 일반보다 공격적으로 적용 가능(정확도 여유) → eye-tracking 전용 마스크를 오프라인 생성해 BRAM에 적재.

3. **reorder(polarize) 방법론을 전처리로 흡수**
   - `reorder.py`의 out-degree 기반 global token 추출은 **오프라인 1회**면 충분(고정 토큰). FPGA 비트스트림에 reorder 순열을 하드와이어 → 런타임 reorder 로직 불필요. HG-PIPE의 토큰 라우팅에 이 순열을 정적 인터커넥트로 구워넣기.

4. **auto-encoder(encoder/decoder) 압축 → on-chip 데이터 이동 절감**
   - `encoder_q/k`(헤드 절반 압축)는 작은 FC → **FPGA DSP로 저비용 구현 가능**, off-chip(HBM/DDR) Q·K 전송량을 ~1/3로 줄임. HG-PIPE처럼 weight-stationary 파이프라인에서 **HBM 대역이 병목일 때 직접 효과** → XR 저전력(웨어러블) 환경에서 메모리 전력 절감으로 직결.

5. **사이클 시뮬레이터 방법론 재사용 (가장 즉시 활용 가능)**
   - ViTCoD.py/ViT_FFN.py의 **analytical cycle model(워크로드 차원 × PE/대역/압축 → 사이클)** 구조를 그대로 차용해 **HG-PIPE 계열 FPGA의 빠른 DSE(설계공간탐색)** 도구로 활용. FPGA 파라미터(PE 배열, BRAM 대역, II)로 치환하고, `Total=max(preload, compute)` roofline 식을 FPGA II/dataflow로 보정. RTL 합성 전에 sparsity-vs-latency 트레이드오프를 초고속 탐색 → 우리 프로젝트의 HW/SW co-design 루프에 직접 삽입.

6. **8-bit 가정의 검증 필요**
   - 시뮬레이터의 INT8 가정을 우리 측에서 **실제 양자화(QAT) + FPGA INT8 DSP 패킹**으로 구체화해야 ViTCoD의 절감 주장이 성립. XR eye-tracking은 정확도 허용폭이 넓어(시선 좌표 회귀) INT8/INT4까지 공격적 양자화 + 희소화 동시 적용 여지가 큼.

7. **한계 보완 방향**: ViTCoD가 미제공한 RTL/면적/전력을 우리가 **실제 HLS/RTL로 구현·합성**하면 ViTCoD 방법론의 FPGA 실증이라는 명확한 기여 포인트가 됨(특히 dedicated dense/sparse 엔진의 LUT/DSP/BRAM 비용 보고).

---

## 10. 근거 / 한계 표기

- **확정 사실(코드/README 직접 근거)**:
  - HPCA 2023 논문 = ViTCoD (루트 README L1-L8, citation L59-L64). "추정"이 아니라 명시.
  - **RTL/HLS 소스 부재** → `.v/.sv/.cpp/.h/.cl` Glob 결과 전부 "No files found". 하드웨어는 **순수 Python 사이클 시뮬레이터**.
  - denser/sparser 동적 PE 분할: ViTCoD.py L159-L161, 병렬 max 합산 L176/L227.
  - auto-encoder(encoder/decoder, 헤드 압축): vision_transformer.py L235-L240, L359-L386.
  - info-cutoff 프루닝: gen_mask.py L51-L105.
  - polarize/reorder(DGL out-degree, threshold=50): reorder.py L8-L59, L93.
  - 메모리/주파수 가정(SRAM 53/108/20KB, HBM 76.8GB/s, 500MHz, INT8): SRAM.py L12-L19, 호출부 bits=8.
- **추정**:
  - 타깃이 ASIC급(SRAM/HBM 가정에서) → "ASIC 가속기(시뮬레이션상)"로 추정. 단, **시뮬레이터 vs 실제 RTL은 명확히 다르며 RTL은 본 repo에 없음**.
  - `mix_head_fc_qk`가 README 권장 경로라는 점은 run 스크립트/README 정합으로 추정(deit run_deit_tiny_svd_sparse.sh L13 등).
  - `PE.py` 연산 메서드 미사용(참조용)은 import 후 호출 부재로 추정.
- **확인 불가**:
  - 면적/전력/타이밍 closure(주파수 달성), 실제 reconfigurable interconnect/컨트롤 FSM의 비용 — RTL 부재로 검증 불가.
  - 알고리즘 측 정수 양자화(QAT) 코드 — 본 repo 미포함.
  - LeViT 경로 상세 — README 미완성("Will update ASAP").
- **시뮬레이터 vs 실제 RTL 구분(재강조)**: 본 repo의 "Hardware"는 **사이클 추정 analytical model**이며, 합성·배치·타이밍이 검증된 RTL/하드웨어 구현이 아니다. 모든 성능 수치는 모델 가정 하의 추정치로 해석해야 한다.
