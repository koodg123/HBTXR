# HiSpMM 코드베이스 정밀 분석

> 분석 대상 경로: `REF/Others/HiSpMM`
> 작성 기준: 실제 소스 Read 기반, 라인 근거 표기. 추정/확인불가는 명시.

---

## 1. 개요

- **프로젝트명**: HiSpMM — *High Performance High Bandwidth Sparse-Dense Matrix Multiplication on HBM-equipped FPGAs* (`README.md:1-3`)
- **목적**: **희소 행렬 A × 조밀 행렬 B = 조밀 행렬 C** (SpMM, Sparse-Dense MatMul)을 HBM 장착 FPGA에서 고대역폭으로 가속. SpMV(벡터)가 아니라 **벡터 N개를 동시에 처리하는 SpMM**(B의 N0=8 컬럼 묶음). HiSpMV의 후속/자매 작업으로, 같은 PEG/RDN/누산 골격을 N차원으로 확장.
- **한줄요약**: `C = alpha*(A·B) + beta*C_in` 을, A는 64b 패킹 nnz 스트림, B는 PEG 로컬 BRAM에 N0-벡터로 적재하여 PE당 N0개 MAC을 SIMD로 수행, 결과를 (Accumulator URAM → Arbiter → Compute_C → HBM)로 흘리는 TAPA dataflow. 워크로드 불균형은 **Row-Sharing(RS) 가능 커널 + Dense Row Distribution Network(DRDN)** 로, 균형 워크로드는 **NRS(no row sharing) 커널**로 처리. **automation_tool**이 DSE로 두 변형 중 선택하고 task 템플릿을 합성.
- **연산 형태**: `C_out = alpha*(A·B) + beta*C_in` (`Compute_C.cpp:13`).
- **타깃 디바이스**: Xilinx Alveo **U280** (`xilinx_u280_gen3x16_xdma_1_202211_1`) (`README.md:5,49-51`).
- **설계 변형**(`README.md:31-34`):
  - **HiSpMM-balanced**: 80 PE, A포트 10, HBM 22채널, 균형 워크로드, SLR 단위 floorplan.
  - **HiSpMM-imbalanced**: 64 PE, A포트 8, HBM 20채널, 불균형 워크로드, Half-SLR floorplan.
- **HLS 프레임워크**: TAPA + PASTA, Vitis 2023.2, XRT 2.14+, C++17 (`README.md:53-66, 78-80`).
- **원논문**: Sedigh Baroughi, Rajashekar, Baranwal, Fang, *HiSpMM*, **ACM TRETS 2025**, DOI `10.1145/3774327` (Just Accepted) (`README.md:264-283`). 키워드: SpMM, Imbalanced Workload, FPGA, HLS, DSE.
- **소속**: SFU-HiAccel (`README.md:91,297`), 연락처 asa582@sfu.ca.

---

## 2. 디렉토리 구조

### 자체 핵심 소스 트리 (분석 대상)
```
HiSpMM/
├── README.md
├── automation_tool/                       # ★ 코드 생성 + DSE 패키지 (python -m automation_tool)
│   ├── __main__.py / __init__.py
│   ├── generate.py                        # ★ 커널/호스트/빌드에셋 합성 오케스트레이터
│   ├── advisor.py                         # ★ cycle_analysis 실행→RECOMMEND 파싱(또는 --pick)
│   ├── spmm_header.py                     # hispmm.h 매크로 패치(NUM_A_CH 등)
│   ├── spmm_top.py                        # hispmm.cpp top 패치(RS/DRDN/arbiter invoke)
│   ├── templates.py                       # task 템플릿 로더/배너/concat
│   ├── misc_assets.py                     # Makefile/link_config/floorplan 생성
│   ├── drdn.py                            # DRDN 그래프→필요 task 파일 목록
│   ├── crossbar.py                        # ★ DRDN 토폴로지/스트림 깊이 생성
│   ├── dse/                               # ★ DSE (cycle/resource 추정)
│   │   ├── cycle_analysis.py              # ★ DSE 진입점(변형 sweep, RECOMMEND 출력)
│   │   ├── cycle_models.py                # term1/2/3 사이클 식
│   │   ├── load_models.py                 # ★ delta(불균형)/run-length/row-sharing 모델
│   │   ├── resource_models.py             # 리소스 추정/DRDN 카운트
│   │   ├── mm_parser.py                   # .mtx COO 파서
│   │   └── reporting.py                   # 표 출력
│   └── assets/
│       ├── tasks/                         # ★ HLS task 템플릿 (커널 빌딩블록)
│       │   ├── hispmm.h / hispmm.cpp      # 헤더 + top dataflow
│       │   ├── PEG_NRS.cpp / PEG_RS.cpp   # ★ PE 그룹(균형/행공유)
│       │   ├── Accumulator.cpp            # ★ 출력 누산(URAM)
│       │   ├── Compute_C.cpp / Arbiter_C_*.cpp / S2MM_C.cpp
│       │   ├── MM2S_A/B/C.cpp / DummyRead.cpp / inline_tasks.cpp
│       │   └── ADD_*/SWB*/SSW.cpp         # DRDN 스위칭/가산 블록
│       └── misc/
│           ├── host/{main/hispmm_host.cpp, common/*}  # ★ host (패킹/검증)
│           └── Makefile/common.mk
├── HiSpMM-balanced/   src/{spmm.cpp, spmm.h, spmm-host.cpp, helper_functions.*, mmio.h}, link_config.ini
└── HiSpMM-imbalanced/ src/{...동일 구조...}, link_config.ini
```

### 제외 목록 (vendor/생성물/데이터 — 이름만)
- `HiSpMM-balanced/`, `HiSpMM-imbalanced/` 의 `spmm.cpp`/`spmm.h`/`spmm-host.cpp`: **automation_tool가 생성/패치한 산출물 변형**. 원본 템플릿(`assets/tasks/*`)을 정밀 분석하고 변형은 구성차이만 참조.
- `src/mmio.h`: NIST MatrixMarket I/O 표준 코드(vendor) → 이름만.
- 비트스트림 `.xclbin`/`.xo`, `bitstream/`, `vitis_run_hw/`, floorplan.json: 빌드 산출물 → 제외.
- `matrices/*.mtx`(airfoil_2d, hangGlider_3 등) 행렬 데이터 → 제외.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 커널 헤더 `assets/tasks/hispmm.h`

설계 상수의 단일 출처 (NUM_PES=80 기본값 = balanced).

- 채널/PE 상수(`:9-24`): `NUM_A_CH 10, NUM_C_CH 4, NUM_B_CH 4`, `NUM_PES 80`, `GROUP_SIZE 4`(PEG당 PE 4개), `NUM_PEG = NUM_PES/GROUP_SIZE`(=20), `II_DIST 8`, `PES_PER_CH 8`.
- **N차원 타일 상수**(`:27-37`): `N0 8`(B의 컬럼 SIMD 폭), `B_CHUNK_SIZE = 2*N0 = 16`, `K0 = NUM_B_CH*8*1024/N0 = 4096`(K 타일 깊이), `B_READ_LEN = 512`, `U 2`(PE당 URAM 수), `MAX_ROWS_PER_PE = U*4096 = 8192`, `M0 = NUM_PES*MAX_ROWS_PER_PE`(M 타일 깊이). → **3D 타일링(M0×K0×N)**.
- 벡터 타입(`:40-45`): `uint64_v = vec_t<uint64_t, PES_PER_CH>`(A 채널), `uint64_v2 = vec_t<uint64_t, 2>`(PE 2개분 nnz), `float_vN = vec_t<float, N0>`(C 한 행의 N0개), `float_vB = vec_t<float, B_CHUNK_SIZE>`(B/C HBM 채널 폭).
- 패킷(`:47-69`): `flags_pkt{sharedRow,tileEnd,last}`, `Cnoc_pkt{...,float val[8]}`(RS/DRDN용, N0=8개 부분합 운반), `Cvec_pkt{dummy,tileEnd,row16,float val[8]}`(NRS용).
- top 선언 `hispmm(A,B,c_in,c_out, alpha,beta, M,N,K, numTilesM/N/K, len, last_tile_idx, rp_time)` (`:71-78`).
- **HiSpMV와의 핵심 차이**: 모든 결과 패킷이 `val[8]`(N0=8) 벡터 → SpMV는 스칼라 val. PE가 **N0개 출력을 동시 산출**.

### 3.2 Top dataflow `assets/tasks/hispmm.cpp`

`hispmm(...)` 본문(`:1-45`):
- 스트림(`:10-23`): `FIFO_A_IN`(uint64_v2×NUM_PES_HALF), `FIFO_B_IN`(float_vB×(NUM_PEG+1)*NUM_B_CH), `FIFO_C_ARB`(float_vN×NUM_PES, 깊이11), `FIFO_C_AB/IN/OUT`(float_vB×NUM_C_CH), `FIFO_C_SHF`(RS면 Cnoc_pkt 아니면 Cvec_pkt). NUM_PES==80이면 2단 arbiter용 `FIFO_C_AB_INTER`(`:21-23`).
- invoke(`:25-44`):
  `MM2S_A×10` → `MM2S_B×4` → `MM2S_C×4` → `PEG×NUM_PEG` → `DummyRead×4`(detach) → `Accumulator×NUM_PES` → **Arbiter**(NUM_PES별 분기: 80=`Arbiter_C_10_1`+`Arbiter_C_8_4` 2단, 64&C8=`Arbiter_C_8_1`, 64&C4=`Arbiter_C_16_1`, else 단일 `Arbiter_C`, `:32-41`) → `Compute_C×4`(detach) → `S2MM_C×4`.
- 컴파일 분기 `RS_DESIGN`(`:16-20`): 정의 시 DRDN(Cnoc_pkt) 경로, 아니면 NRS(Cvec_pkt) 직결.

### 3.3 PE 그룹: `PEG_NRS.cpp` / `PEG_RS.cpp` (가장 중요)

두 변형 모두 `PEG(a_in[GROUP_SIZE/2], b_in[NUM_B_CH], b_out, c_out[GROUP_SIZE], K, numTilesK, last_tile_idx)`.

**공통 구조 (`PEG_NRS.cpp:2-82`, `PEG_RS.cpp:2-107`)**:
- **로컬 B 버퍼** `buff_B[GROUP_SIZE/2][N0][B_CHUNK_SIZE/N0][NUM_B_CH*B_READ_LEN]` BRAM(`NRS:9-14`/`RS:9-14`). 차원1(N0)·2·3 완전분할, 차원4 cyclic factor 2 → **N0개 B 컬럼을 동시 읽기** 위한 뱅킹.
- **load_B 루프**(`NRS:29-40`): K 타일마다 B 청크를 HBM 스트림에서 받아 `buff_B`에 적재(`buff_B[g][p%N0][p/N0][...] = temp[p]`), b_out으로 다음 PEG에 전달(systolic forwarding). last K타일은 `read_len = K%K0/KX`로 부분 적재(`:28`).
- **mul_AB 루프**(`NRS:43-78`): A nnz 디코딩 — `val_bits=a&0xFFFFFFFF`(float), `col_id=(a>>32)&0x1FFF`(13b 열), `row=(a>>48)&0xFFFF`, `rowEnd=(a>>46)&1`, `tileEnd=(a>>47)&1`(`:50,59-62`). 곱: `temp.val[n] = val_in[p] * buff_B[p/2][n][col_id%2][col_id/2]` for n in 0..N0 (`:69-71`) → **PE당 N0=8개 MAC을 UNROLL**. `dummy = !rowEnd`(행 종료 마커가 아니면 dummy).

**NRS vs RS 차이 (불균형 처리의 핵심)**:
- **NRS** (`PEG_NRS.cpp`): `Cvec_pkt`만 출력, `row16`/`dummy`/`tileEnd`만 설정(`:73-76`). 행은 항상 `row%NUM_PES` PE에 고정 → **행 공유 없음**(균형 워크로드 가정).
- **RS** (`PEG_RS.cpp:55,79-99`): `sharedRow=(a>>45)&1` 플래그 디코딩(`:55`). sharedRow면 **인접 PE 쌍(p, p+1)** 의 부분합을 한 행으로 묶기 위해 `bank`(목적 누산뱅크=`row_out[p-1]&(NUM_PES-1)`)와 `row16`를 교차 설정(`:79-95`). → 무거운 행을 여러 PE에 분산 계산 후 DRDN으로 한 누산기에 모음. `Cnoc_pkt`(bank/sharedRow/last 포함) 출력.

### 3.4 출력 누산 `assets/tasks/Accumulator.cpp`

`Accumulator(c_out, c_in, M, numTilesM, last_tile_idx)` (`:2-51`):
- `float_vN buffer_C[MAX_ROWS_PER_PE]` URAM(`:7-8`) — 행별 **N0=8 벡터** 누산 버퍼.
- M 타일 루프(`:12-50`): init_c(0초기화) → acc_c(누산) → updt_C(출력). last M타일은 `read_len=(M%M0)/NUM_PES`로 부분(`:16`).
- **누산 핵심**(`:23-41`): `#pragma HLS dependence variable=buffer_C inter RAW distance=8 true`(`:25`, =II_DIST) — RAW 의존거리 8로 II=1 유지. `Cvec_pkt` 읽어 `!dummy`이면 `buffer_C[m][n] += temp_in.val[n]` for n in N0(`:35-40`). `m = temp_in.row16`.
- **HiSpMV 차이**: SpMV의 순환버퍼 forwarding(circbuff) 대신 단순 `+=` + DEPENDENCE distance=8. N0 벡터 누산이라 BRAM/URAM 폭이 8×float.

### 3.5 후처리: Arbiter / Compute_C / S2MM_C / MM2S_*

- `Arbiter_C_10_1.cpp` (`:2-14`): 10개 `float_vN` 입력을 순회하며 1개 `c_ab`로 직렬화(`tapa::detach` 무한루프). 80PE는 `Arbiter_C_10_1`(8개 인스턴스) → `Arbiter_C_8_4` 2단 트리로 NUM_C_CH=4 폭으로 모음(`hispmm.cpp:32-34`).
- `Compute_C.cpp` (`:2-17`): `temp2[p] = alpha*c_ab[p] + beta*c_in[p]` for p in B_CHUNK_SIZE(`:12-13`), `try_read`로 backpressure 처리. **axpy 후처리** = HiSpMV의 Compute_C와 동형(N차원 확장).
- `MM2S_B.cpp` (`:2-28`): B를 (N타일→M타일→K타일) 4중 루프로 HBM에서 스트림. `start_addr = k*numTilesN*B_READ_LEN + n*read_len`(`:13`) — **K 타일이 채널당 B_READ_LEN 주소를 차지**하는 레이아웃. `loop_flatten OFF`로 타일 경계 유지. (MM2S_A/C는 유사 async_mmap 패턴, inline_tasks.cpp의 `async_readB` 헬퍼 사용.)
- `S2MM_C.cpp`: C 결과를 numTilesM/N 순서로 HBM write(packed M-row 모델).

### 3.6 DRDN 스위칭 블록 `assets/tasks/{ADD_*, SWB*, SSW}.cpp`

HiSpMV와 동형의 butterfly add+switch 네트워크지만 **N0=8 벡터(`Cnoc_pkt.val[8]`)** 를 운반. `ADD_0/ADD_1/ADD_X`(가산), `SWB0_n/SWB1_n`(bank 비트 라우팅), `SSW`(shared swap). RS 커널에서만 사용. `drdn.py`의 `required_task_files_for_drdn`이 그래프 노드에서 필요한 파일을 추려 합성(`generate.py:146`).

### 3.7 코드 생성 오케스트레이터 `automation_tool/generate.py` (가장 중요)

- `KernelPlan` dataclass(`:22-44`): label/pick/a_ch/b_ch(고정 4)/c_ch/num_pes/rs_capable. `from_recommendation`이 advisor 결과를 변환.
- `_base_task_files`(`:47-61`): 공통 task + (`PEG_RS.cpp` if rs_capable else `PEG_NRS.cpp`). → **balanced=NRS, imbalanced=RS** (README.md automation_tool:20).
- `_arbiter_task_files`/`_arbiter_invoke_block`(`:63-92`): NUM_PES/C_CH별 arbiter 선택 — 80=2단(10_1+8_4), 64&C8=8_1, 64&C4=16_1, 그 외 monolithic(a4/a6). hispmm.cpp의 `#if`와 정확히 일치.
- `generate_kernel_sources`(`:95-197`): ① `hispmm.h` 매크로 패치(`patch_hispmm_h` + `SpmmDefines`) ② `hispmm.cpp` = 배너+include+task들+패치된 top(`patch_hispmm_cpp`, DRDN 그래프 주입) ③ host 소스 생성 ④ Makefile/link_config/floorplan 에셋 ⑤ 행렬 복사. RS면 `build_drdn_graph(num_pes)` + SWB0/SWB1 템플릿 + DRDN 필요 파일 추가(`:138-146`).
- `_generate_host_sources`(`:272-366`): host main 템플릿을 마커(`@CODEGEN:PREPARE_A_CONFIG`, `PREPARE_C_CONFIG`)로 패치 — A 패킹은 RS 정책(`kAuto`/`kForceDisabled`)·shared_row_limit·25% 임계값 주입(`:298-316`), C 레이아웃은 a_ch별로 결정(`_host_c_layout`, `:261-269`: a4/a6=LinearChunkInterleave, a8=TiledPackedRows+AdjacentPair, a10=TiledPackedRows+HalfGroupPair). → **A_CH별 C/Cin 메모리 레이아웃·행쌍(pairing) 전략이 자동 선택**.

### 3.8 DSE: `dse/cycle_analysis.py` + `cycle_models.py` + `load_models.py`

- `cycle_analysis.py`(진입점, `:92-321`):
  - 변형 집합 `_build_runs`(`:54-89`): balanced(a10), imbalanced(a8), both, all(a10/a8/a6/a4 × NRS/RS 조합), sweep(a4/6/8/10 × c4/c8 × NRS/RS). 각 run = (label, A_CH, C_CH, can_row_share, force).
  - 디바이스 주파수 테이블(`FREQ_MHZ_BY_LABEL`, `:30-41`): 측정값 — balanced_a10_c4=216MHz, balanced_a8_c8=204MHz, 나머지 225MHz. → **실측 주파수로 time_us 계산**.
  - 각 run: `compute_delta_with_row_sharing_auto`로 delta1(NRS)/delta2(RS)/개선% 산출, RS 가능+개선≥25%면 RS 채택(`:136-164`). NRS 강제 시 1.03 페널티(`:166`). term2는 runlen(기본) 또는 nnz_delta(`:182-197`). `cycle_terms`로 t1/t2/t3 합산 × 페널티(`:199-214`). 리소스 추정 시 fit 검사(`:218-235`).
  - 정렬·추천(`:269-321`): (fit, time_us, -a_ch) 순. `RECOMMEND: <label>` 출력.
- `cycle_models.py` `cycle_terms`(`:4-43`): **3-term 사이클 식**.
  - t1(B 이동, `:26-31`): `((K0*N)/(16*B_CH) + SB*ceil(N/N0)) * ceil(K/K0) * ceil(M/M0)`, SB=60(타일당 셋업 경험상수).
  - t2(연산, 외부 주입 = run-length 또는 `(M*K*rho)/(A_CH*PES_PER_CH)*(N/N0)*(1+delta)`, `:33-34`).
  - t3(C 이동, `:37`): `(M*N)/(16*C_CH)` — packed(실제 M행만 이동).
- `load_models.py`(불균형 핵심, `:1-414`):
  - `compute_delta_no_row_sharing`(`:46-95`): 타일별 행을 `row%NUM_PES` PE의 II_DIST lane 중 최소부하에 그리디 배치, PE 부하의 stddev/mean = **delta(불균형 지표)**.
  - `_balance_workload_shared_rows_for_tile`(`:98-149`): 가장 무거운 행들을 shared로 빼며 imbalance(`(scheduled-total)/total`) 개선을 추적, 임계 도달 시 중단 → **host의 imbalanced 패킹 로직 포팅**(주석 명시 `:106`).
  - `compute_delta_with_row_sharing_auto`(`:152-247`): delta1/delta2 비교, 개선≥25%면 row_sharing 채택(`:245-246`) — **generate.py의 25% 임계값과 일치**.
  - `compute_term2_from_runlen_tiled`(`:312-411`): NRS/RS 각각 lane 스케줄로 `run_len = (max_bucket+padding)*II_DIST` 합산, `t2_runlen = run_len*(N/N0)` (`:393-410`). → **legacy host run-length 모델의 추정 재현**.
- `mm_parser.py`: .mtx → `MatrixShape(m,k,nnz)` + entries. `resource_models.py`: area.log 기반 task 면적 추정 + `drdn_counts(num_pes)`.

### 3.9 advisor / spmm_top / spmm_header / crossbar / misc_assets

- `advisor.py`(`:46-218`): `automation_tool.dse.cycle_analysis`를 subprocess로 실행, 정규식으로 summary 표/`RECOMMEND:` 파싱(`_RE_SUMMARY_ROW`, `_RE_RECOMMEND`, `:26-43`). `--pick` 별칭 해석(legacy `hispmm-balanced→balanced_a10_c4` 등, `:120-151`). `row_sharing_capable`= mode가 `(k)`로 안 끝나면 RS 가능(`:21-23`). `unified_pick_name`으로 balanced/imbalanced 라벨 통일(`:221-238`).
- `spmm_top.py`/`spmm_header.py`: top/헤더 텍스트 패치(NUM_A_CH 등 define 치환, RS/DRDN/arbiter invoke 블록 주입). `crossbar.py`: HiSpMV와 거의 동일한 DRDN 2-페이즈 butterfly 토폴로지(가산 latency 보정 깊이 +6, `:26-29`). `misc_assets.py`: standalone Makefile/link_config/floorplan 로드·생성.

### 3.10 Host `assets/misc/host/main/hispmm_host.cpp` + `common/*`

- `hispmm_host.cpp`: CLI `./hispmm [--bitstream] matrix.mtx iterations [dense_cols]`(`README.md:182-191`). `@CODEGEN:` 마커로 A/C 패킹 config가 codegen에 의해 주입됨(`generate.py:311-362`).
- `common/prepare_amt_unified.{h,cpp}`: **A 행렬 통합 패킹** — RS 정책(kAuto/kForceDisabled), shared_row_limit, delta 개선 임계 25%로 shared row 선택·64b 인코딩. (host의 실제 밸런싱 = load_models가 추정한 것의 실구현.)
- `common/prepare_fpga_cin_unified.h` / `compare_fpga_c_unified.h` / `host_common.h`: Cin 레이아웃(Linear/Tiled, row pairing) 패킹, FPGA 결과 vs CPU 참조 비교. `mmio.h`(vendor).

---

## 4. 데이터플로우 / 실행 흐름

```
A(.mtx)→host 패킹(64b nnz, shared row 선택)→HBM
B(dense)→HBM,  C_in(dense)→HBM
                    │
HBM ─(MM2S_A×10)→ a_in ──┐
HBM ─(MM2S_B×4) → b_in ──┤
                    (PEG×NUM_PEG: load_B→buff_B(BRAM, N0뱅킹), mul_AB: val*B[n] for n<N0)
                    │   b_out→다음 PEG(systolic B forwarding)
                    │  Cvec_pkt(NRS) / Cnoc_pkt(RS, val[8])
        (RS만) (DRDN: ADD/SWB/SSW 다단 — shared row를 한 누산뱅크로)
                    │
            (Accumulator×NUM_PES: URAM buffer_C[행][N0] += val[N0], RAW dist=8)
                    │ float_vN(=N0)
            (Arbiter: NUM_PES → NUM_C_CH 폭, 80PE는 2단 트리)
                    │
HBM ─(MM2S_C×4)→ c_in →(Compute_C×4: alpha*Ab + beta*Cin)→(S2MM_C×4)→HBM(C_out)
```

- **3D 타일링**: M0(행)×K0(축약)×N(B 컬럼). N0=8씩 SIMD. K0=4096, M0=NUM_PES*8192.
- **메모리 계층**: HBM → async_mmap 스트림 → PEG 로컬 BRAM(B, N0 뱅킹, 재사용) → URAM(C 누산, N0 폭). B는 PEG 간 systolic forwarding(b_out).
- **병렬화**: NUM_PES(64/80) PE × N0(8) MAC = PE당 8 SIMD lane. PEG=4 PE 묶음. Arbiter/DRDN은 로그단.
- **파이프라인**: mul_AB/load_B/Accumulator 모두 II=1, 누산 RAW는 `dependence distance=8`.
- **데이터타입/양자화**: **FP32 전용**. A nnz=64b(float val 32b + col 13b + row 16b + rowEnd/tileEnd/sharedRow 제어비트). 양자화/저비트 **없음**.
- **불균형 적응**: delta(stddev/mean) ≥ 임계면 RS 커널(행 공유+DRDN), 아니면 NRS. DSE가 행렬별로 선택.

---

## 5. HW/SW 매핑

| 계층 | 구성요소 | 역할 |
|---|---|---|
| SW(py, 빌드타임) | `automation_tool/{generate,advisor}.py`, `dse/*` | 행렬 DSE(delta/run-length/리소스)→변형(balanced/imbalanced)·채널 선택→커널/host/빌드에셋 합성 |
| HW 템플릿(HLS) | `assets/tasks/*.cpp,*.h` | PEG(NRS/RS)/Accumulator/DRDN/Arbiter/Compute_C/IO task |
| HW(산출) | `HiSpMM-{balanced,imbalanced}/` xclbin | U280 비트스트림(80PE/64PE) |
| SW(런타임, host C++) | `assets/misc/host/*`, `*/src/spmm-host.cpp` | A/Cin 패킹(레이아웃·행쌍·shared row), CPU 참조, XRT 실행/검증 |
| 데이터 | `matrices/*.mtx`(외부) | 입력 희소행렬 |

핵심: **codegen이 DSE 추천을 받아 NRS/RS 커널을 골라 합성**하고, **host 패킹 config까지 같은 결정으로 주입**해 HW/SW 일관성 유지(`generate.py`가 양쪽을 함께 패치).

---

## 6. 빌드·실행

- 환경: Vitis 2023.2, XRT 2.14+, PASTA, glog/gflags/OpenCL, C++17 (`README.md:53-86`).
- codegen: `python -m automation_tool --matrix <mtx> --n 8 --out generated/kernel_auto` (DSE 추천), `--pick imbalanced_a8_c4`(강제), `--list-picks`, `python -m automation_tool.dse.cycle_analysis --matrix <mtx> --n 8 --variant sweep`(DSE만) (`README.md:122-156`).
- 빌드: `make host` → `make tapa`(.xo) → `make hw-build`(.xclbin) (`README.md:99-174`).
- 실행: `./hispmm matrices/airfoil_2d.mtx 1 32`(sw-test) / `--bitstream=...xclbin matrices/X.mtx 1000`(HW) (`README.md:193-234`).
- 사전 빌드 변형: HiSpMM-balanced(`make host` after `cd`) / HiSpMM-imbalanced.

---

## 7. 의존성

- HW: TAPA/PASTA, Vitis HLS, `ap_int.h`, `tapa.h`. host: XRT, glog, gflags, OpenCL, FRT.
- SW(py): 표준 lib 중심(argparse/re/subprocess/pathlib/dataclasses). `dse/`는 순수 파이썬(numpy 미사용 — load_models가 순수 리스트 연산). `mm_parser`가 .mtx 파싱.
- 데이터: SuiteSparse `.mtx`. 사전 빌드 비트스트림 제공.

---

## 8. 강점 / 한계 / 리스크

**강점**
- **SpMM(N차원)**: SpMV를 N0=8 SIMD로 확장 → dense B의 여러 컬럼을 한 번에, attention/GEMM류에 더 근접.
- **이중 변형 자동 선택**: delta(불균형 지표)로 NRS(균형, 80PE) vs RS(불균형, 64PE+DRDN)를 DSE가 자동 결정, 25% 개선 임계로 RS on/off.
- **codegen이 커널+host+빌드에셋+레이아웃을 일괄 합성**: A_CH별 C/Cin 레이아웃·row pairing까지 자동 — HW/SW 결정 일관성.
- **실측 주파수 테이블 기반 time 모델**: 216/204/225MHz 등 측정값으로 현실적 추정.
- systolic B forwarding(b_out) + 로컬 BRAM N0 뱅킹으로 B 재사용 극대화.
- 정식 논문(ACM TRETS 2025, DOI 보유) → 재현성/신뢰성 상대적으로 높음.

**한계 / 리스크**
- **FP32 전용** — 저비트/정수 양자화 없음(우리 INT8 ViT/Mamba에 직접 부적합).
- 변형이 사실상 **2종(balanced a10/imbalanced a8)** 에 최적화·실측됨 — sweep의 다른 점은 추정 위주(주파수 default 225MHz). 임의 행렬/디바이스 일반화는 추정.
- DSE 사이클 식의 **경험 상수**(SB=60, NRS 페널티 1.03, area.log 의존) → 디바이스/툴 변하면 부정확(추정).
- DRDN/host 밸런싱 로직이 **py(추정)와 C++(실구현) 이원화** — 불일치 가능.
- N0=8, K0=4096 등 상수 하드코딩 → 차원이 다른 워크로드는 재튜닝 필요.

**HiSpMV 대비 차이 요약**: SpMV(스칼라 b)→SpMM(N0=8 벡터 B), 결과 패킷 val[8], Accumulator는 순환버퍼 forwarding 대신 단순 += + dependence distance, codegen이 host 레이아웃까지 패치, 논문 정식 출판(DOI).

---

## 9. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적, 추정)

- **SpMM = sparse GEMM 골격**: ViT의 attention/MLP가 결국 (희소화된) GEMM이라면, HiSpMM의 **PE당 N0 SIMD + URAM N0 누산 + Arbiter 트리** 구조는 sparse/pruned ViT GEMM 데이터패스의 직접 참조 모델. 특히 토큰 가지치기(token pruning)된 attention을 SpMM으로 매핑하는 데 적합.
- **불균형 적응(delta→NRS/RS 자동 전환)**: 입력 sparsity 패턴 통계로 행 공유 on/off를 자동 결정하는 DSE 휴리스틱(`load_models.compute_delta_*`, 25% 임계)은, ViT의 동적/구조적 sparsity에 맞춰 가속기 구성을 고르는 데 이식 가능.
- **codegen이 HW+host 레이아웃을 동시 합성**(`generate.py` 마커 패치): HG-PIPE류 파이프라인 가속기를 레이어 config별로 자동 생성할 때, **커널과 데이터 패킹/레이아웃을 하나의 결정으로 동기화**하는 설계 패턴은 그대로 채택할 가치. 특히 A_CH별 C/Cin 레이아웃 선택(`_host_c_layout`)은 멀티-HBM 배치 자동화 템플릿.
- **DRDN(N0 벡터 운반 butterfly)**: 부분합을 다단 가산+스위칭으로 모으는 네트워크는 sparse attention score 누적, multi-head 결과 병합, Mamba의 segmented scan 누적 등에 응용 가능.
- **3D 타일링(M0×K0×N) + B systolic forwarding**: ViT GEMM의 타일링·온칩 재사용 전략(weight/activation 재사용)을 잡는 레퍼런스. K0=4096, N0=8 같은 타일 상수 튜닝 방식 참고.
- **3-term 사이클 모델(t1 B이동/t2 연산/t3 C이동)**: ViT 레이어의 roofline 추정(메모리 vs 연산 bound)에 바로 응용 가능한 간결한 분석 모델.
- **한계 인지**: FP32 전용 → 우리의 INT8/저비트 경로엔 PE 곱셈기·패킹 폭 재설계 필요. 양자화는 ViT-Quantization 계열에서, **dataflow/불균형 적응/codegen 골격은 HiSpMM에서** 차용하는 분업이 현실적(추정).

---

## 10. 근거 표기

- **확인(라인 근거)**: 모든 코드 동작은 `assets/tasks/{hispmm.h, hispmm.cpp, PEG_NRS.cpp, PEG_RS.cpp, Accumulator.cpp, Compute_C.cpp, MM2S_B.cpp, Arbiter_C_10_1.cpp}`, `automation_tool/{generate.py, advisor.py, crossbar.py}`, `dse/{cycle_analysis.py, cycle_models.py, load_models.py}`, `README.md`(+automation_tool/README.md) 의 명시 라인 기반.
- **추정**: TAPA=PASTA 확장 활용 정도, sweep의 비측정 점 정확도, 경험 상수(SB=60/1.03/area.log)의 디바이스 일반화, prepare_amt/prepare_cin 세부(역할은 host 마커·generate.py 주입으로 확인, 라인별 알고리즘은 미정독), HG-PIPE 이식 시사점.
- **확인 불가**: ACM TRETS 2025 본문 성능/정확도 수치(논문 미정독), 사전 빌드 비트스트림의 실측 주파수 외 상세, `HiSpMM-balanced/imbalanced/src/spmm.cpp`(생성물) vs 템플릿의 완전 동일성(템플릿 기준 분석).
```
