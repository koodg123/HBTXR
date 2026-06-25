# HiSpMM 모듈 통합 가이드

> 1차 요약: [`../HiSpMM.md`](../HiSpMM.md) — 본 문서는 그 요약을 모듈 단위로 심화한 통합 가이드다.
> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\Others\HiSpMM`
> 작성 원칙: 실제 소스 Read 후 `파일:라인` 근거 표기. 라인 근거 없는 추론은 "추정", 코드로 확인 불가는 "확인 불가"로 명시.
> 형제 가이드(동형 구조): [`../HiSparse/MODULE_GUIDE.md`](../HiSparse/MODULE_GUIDE.md). HiSpMM은 HiSpMV(SpMV)의 SpMM(N0=8 SIMD) 확장이므로 각 모듈에 **HiSpMV 대비 차이**를 명시.

---

## 0. 문서 머리말

### 0.1 대표 케이스 선정
- **대표 연산: SpMM `C = alpha*(A·B) + beta*C_in`** (A=희소, B/C=조밀, FP32). 근거: top 시그니처 `hispmm(...)`의 인자 alpha/beta/A/B/c_in/c_out(`hispmm.h:71-78`)와 axpy 후처리 `temp2[p]=alpha*temp0[p]+beta*temp1[p]`(`Compute_C.cpp:13`). SpMV(`y=A·x`)와 달리 B가 **N0=8 컬럼 묶음**이라 PE 1개가 사이클당 N0=8 MAC을 SIMD로 수행(`PEG_NRS.cpp:69-71`).
- **대표 변형: HiSpMM-balanced(NRS, 80 PE)** 와 **HiSpMM-imbalanced(RS, 64 PE)** 2종. 근거: `README.md:31-34`(balanced=80PE/A10/HBM22, imbalanced=64PE/A8/HBM20), 커널 헤더 기본값 `NUM_PES 80`(`hispmm.h:14`) vs imbalanced standalone `NUM_PES 64`(`HiSpMM-imbalanced/src/spmm.h:15`). **두 변형 중 선택은 codegen+DSE가 delta 지표로 자동 결정**(`generate.py:59`, `load_models.py:246`).
- **대표 PE 그룹: PEG(GROUP_SIZE=4 PE)** — `PEG_NRS.cpp`(균형)와 `PEG_RS.cpp`(행공유) 2 템플릿. 균형 워크로드는 NRS, 불균형은 RS+DRDN. balanced는 PEG×20개(NUM_PEG=NUM_PES/GROUP_SIZE=80/4), imbalanced는 PEG×16개(64/4).
- **대표 누산 lane: Accumulator(PE 1개당 1 인스턴스)** — `float_vN buffer_C[MAX_ROWS_PER_PE]` URAM에 행별 N0=8 벡터를 `+=` 누산(`Accumulator.cpp:7-8,38`). 8단 RAW 의존거리로 II=1(`:25`).
- **대표 코드생성 경로: automation_tool** — DSE(`dse/cycle_analysis.py`)가 변형을 추천하면 `generate.py`가 커널(`hispmm.cpp/.h`)+호스트+빌드에셋을 일괄 합성하고 host A/C 패킹 config까지 같은 결정으로 주입(`generate.py:95-197,272-366`).

### 0.2 수치 표기 규약
- **MAC lanes** = NUM_PES × N0(=8 SIMD). balanced = `80 × 8 = 640` MAC/cycle, imbalanced = `64 × 8 = 512` MAC/cycle. 근거: PE 곱셈 루프 `temp.val[n]=val_in[p]*buff_B[...]` for n in 0..N0(`PEG_NRS.cpp:69-71`), N0=8(`hispmm.h:27`), NUM_PES 80/64(`hispmm.h:14`/`spmm.h:15`). PEG 1개 = GROUP_SIZE(4) PE × N0(8) = 32 MAC/cycle.
- **scalar MAC-equivalent** = `nnz × N` (희소 A의 비영점 1개가 B의 N 컬럼 각각에 곱해짐). N0=8 SIMD가 N을 N/N0 묶음으로 처리(`PEG`의 곱 unroll + DSE term2 인자 `(n/n0)`, `cycle_models.py:34`). 더미/패딩 nnz는 MAC에서 제외(`dummy=!rowEnd`, `PEG_NRS.cpp:74`; 패딩은 host가 lane 길이 균등화용으로 삽입, `prepare_amt_unified.cpp:388-403`).
- **loop trips**:
  - load_B = `read_len`(기본 B_READ_LEN=512, 마지막 K타일은 `K%K0/KX`, `PEG_NRS.cpp:28-29`) × NUM_B_CH(4).
  - mul_AB = tileEnd까지(가변, `loop_tripcount min=1 max=100000`, `PEG_NRS.cpp:43-44`). 정적으론 tile당 `run_len = (max_lane_load+PADDING)*II_DIST`(host 패킹이 결정, `prepare_amt_unified.cpp:198`).
  - Accumulator = M 타일마다 init_c(read_len) + acc_c(tileEnd까지) + updt_C(read_len)(`Accumulator.cpp:18-48`), read_len=MAX_ROWS_PER_PE(8192) 또는 부분.
  - 외부 반복 = numTilesM × numTilesN × numTilesK (3D 타일링, MM2S_B 4중 루프 `MM2S_B.cpp:7-10`).
- **memory size**(payload/버퍼):
  - PEG 로컬 B버퍼 `buff_B[GROUP_SIZE/2][N0][B_CHUNK_SIZE/N0][NUM_B_CH*B_READ_LEN]` = `2×8×2×(4×512)` = 65,536 float = **2.0Mbit/PEG** (BRAM `RAM_T2P`, `PEG_NRS.cpp:9-10`). N0 차원 완전분할로 8 컬럼 동시읽기.
  - Accumulator URAM `buffer_C[MAX_ROWS_PER_PE]=8192` × `float_vN(8×32b=256b)` = **2.1Mbit/PE** (URAM `RAM_2P`, `Accumulator.cpp:7-8`). MAX_ROWS_PER_PE=U×4096=2×4096=8192(`hispmm.h:33`).
  - 스테이지 FIFO = FIFO_DEPTH(2)·FIFO_LARGE_DEPTH(11) × payload (`hispmm.h:22-23`, `hispmm.cpp:10-23`). DRDN 중간 스트림 깊이는 crossbar가 가산 latency(+6) 보정해 산출(`crossbar.py:26-29`).
- **A nnz 패킷 = 64b**: `[63:48]row(16b) | [47]tileEnd | [46]rowEnd | [45]sharedRow | [44:32]col(13b) | [31:0]val(float32)`. 근거: host `encode()`(`prepare_amt_unified.h:74-88`)와 PE 디코딩(`PEG_RS.cpp:53,55,61,64-66`)이 비트정확 정합. 채널 폭 `uint64_v=vec_t<uint64_t,8>`(PES_PER_CH=8, `hispmm.h:40`), PE 입력은 2개분 `uint64_v2`(`hispmm.h:41`).
- **타깃 데이터타입**: **FP32 전용**. 값=`float`(A의 val 32b, B/C=float), 인덱스 col=13b·row=16b 정수(`hispmm.h:40-45`, `prepare_amt_unified.h:74-88`). 양자화/저비트 **없음**(확인: PE 곱이 `float * float`, `PEG_NRS.cpp:70`).

### 0.3 운영 경로
```
[데이터: matrices/*.mtx (SuiteSparse COO)]
      │ mm_parser.load_mtx_coo_shape / host readMatrixCSC→CSR (prepare_amt_unified.cpp:779,836)
      ▼
[DSE: python -m automation_tool.dse.cycle_analysis --matrix X.mtx --n 8 --variant ...]
      │ delta1(NRS)/delta2(RS) 계산 → 개선≥25%면 RS 채택 (load_models.py:152-247)
      │ 3-term 사이클식 t1/t2/t3 + 실측 주파수표 → time_us → RECOMMEND (cycle_analysis.py:199-321)
      ▼
[codegen: python -m automation_tool --matrix X.mtx --n 8 --out gen/ (또는 --pick)]
      │ advisor가 RECOMMEND 파싱 (advisor.py:153-181)
      │ generate.py: hispmm.h 매크로 패치 + hispmm.cpp(task들+top, DRDN 주입) + host(A/C config 마커 패치) + Makefile/link_config (generate.py:95-197)
      ▼
[HW 빌드: make host → make tapa(.xo) → make hw-build(.xclbin), TAPA+PASTA, Vitis 2023.2]
      │ U280 (xilinx_u280_gen3x16_xdma_1_202211_1), HBM 22/20채널 (README.md:33-34,50)
      ▼
[런타임: ./hispmm [--bitstream] matrix.mtx iterations [dense_cols]]
      │ host: A패킹(64b nnz, shared row 선택)·B/Cin 패킹·XRT 실행·CPU 참조 비교 (hispmm_host.cpp, README.md:182-191)
```
- 타깃: **Alveo U280**, shell `xilinx_u280_gen3x16_xdma_1_202211_1`, Vitis 2023.2, XRT 2.14+, TAPA/PASTA, C++17 (`README.md:49-66,78-80`). 사전 빌드 비트스트림 제공(`README.md:42,220`).
- 측정 주파수(DSE 테이블, 실측값으로 명시): balanced_a10_c4=216MHz, balanced_a8_c8=204MHz, 그 외 225MHz 기본(`cycle_analysis.py:30-41`). **합성 PPA 절대치는 area.log 스냅샷(모델용)만 존재** — §13 참조.

### 0.4 HiSpMV(형제) 대비 핵심 차이 (요약, 각 모듈에서 재상술)
| 항목 | HiSpMV (SpMV) | HiSpMM (본 repo) | 근거 |
|---|---|---|---|
| 연산 | `y=A·x` 스칼라 b | `C=alpha·A·B+beta·C_in`, B=N0 컬럼 | `Compute_C.cpp:13`, `hispmm.h:71` |
| PE 출력 | 스칼라 val | **`val[8]`(N0=8 벡터)** SIMD | `Cnoc_pkt.val[8]`(`hispmm.h:60`), `PEG_NRS.cpp:69-71` |
| 누산 | (HiSpMV는 circular-buffer forwarding) | **단순 `+=` + dependence distance=8** | `Accumulator.cpp:25,38` |
| 불균형 처리 | (단일 dataflow) | **NRS/RS 2변형 자동 선택 + DRDN** | `generate.py:59`, `load_models.py:246` |
| 코드생성 | (수작업/단일) | **DSE→codegen이 커널+host 일괄 합성** | `generate.py:95-366` |
| 프레임워크 | Vitis HLS native | **TAPA + PASTA** | `README.md:44,58,80` |
| 데이터타입 | (정수형 ap_ufixed 등) | **FP32 전용** | `PEG_NRS.cpp:70` |
| 출판 | — | **ACM TRETS 2025, DOI 10.1145/3774327** | `README.md:269-282` |
> 주의: HiSparse 가이드의 대상은 HiSpMV가 아닌 HiSparse(별도 SpMV 가속기)다. 본 가이드의 "HiSpMV 대비"는 README/논문이 명시한 HiSpMM의 직계 전신(SpMV)과의 대비이며, 전신 소스가 본 repo에 없으므로 일부는 **추정**으로 표기.

---

## 1. Repo / 시스템 개요

HiSpMM = HBM 장착 U280에서 희소 A × 조밀 B = 조밀 C(SpMM)를 NUM_PES(64/80)×N0(8) SIMD로 가속하는 TAPA 기반 가속기 + **변형 자동 선택 코드생성/DSE 도구**(ACM TRETS 2025, SFU-HiAccel, `README.md:1-8,269-293`). 본 repo는 **HW task 템플릿(`automation_tool/assets/tasks/`)**, **호스트(`assets/misc/host/`)**, **codegen+DSE(`automation_tool/*.py`, `dse/`)**, **사전구성 standalone 변형 2종(`HiSpMM-balanced/`, `HiSpMM-imbalanced/`)** 으로 구성. 정식 커널은 **codegen이 task 템플릿을 조합·패치해 생성**하며, standalone 변형은 그 산출물의 고정 스냅샷.

### 1.1 자체 소스 vs vendor/생성물

| 구분 | 파일(자체 소스) | 역할 |
|---|---|---|
| **HW task 템플릿(HLS)** | `assets/tasks/hispmm.h` | 설계 상수 단일 출처(NUM_PES/N0/타일/패킷타입) |
| | `assets/tasks/hispmm.cpp` | top dataflow(MM2S→PEG→Accumulator→Arbiter→Compute_C→S2MM) |
| | `assets/tasks/PEG_NRS.cpp` / `PEG_RS.cpp` | PE 그룹(균형/행공유), N0=8 SIMD 곱 |
| | `assets/tasks/Accumulator.cpp` | 행별 N0 벡터 URAM 누산 |
| | `assets/tasks/Compute_C.cpp` | axpy 후처리(alpha·AB+beta·Cin) |
| | `assets/tasks/Arbiter_C_*.cpp` | NUM_PES→NUM_C_CH 직렬화(트리) |
| | `assets/tasks/MM2S_A/B/C.cpp`, `S2MM_C.cpp`, `inline_tasks.cpp`, `DummyRead.cpp` | HBM IO(async_mmap) |
| | `assets/tasks/{ADD_*,SWB*,SSW}.cpp` | DRDN butterfly 가산/스위치(RS 전용) |
| **codegen(SW, py)** | `automation_tool/generate.py` | 커널/호스트/빌드에셋 합성 오케스트레이터 |
| | `automation_tool/advisor.py` | DSE 실행→RECOMMEND 파싱(또는 --pick) |
| | `automation_tool/{spmm_top,spmm_header,templates,crossbar,drdn,misc_assets}.py` | top/헤더 패치·템플릿 로더·DRDN 토폴로지·빌드에셋 |
| **DSE(SW, py)** | `dse/cycle_analysis.py` | DSE 진입점(변형 sweep, RECOMMEND) |
| | `dse/{cycle_models,load_models,resource_models,mm_parser,reporting}.py` | 3-term 사이클식 / delta·run-length 모델 / 리소스 추정 / .mtx 파서 |
| **호스트(C++)** | `assets/misc/host/main/hispmm_host.cpp` | CLI·실행·검증(codegen 마커 패치) |
| | `assets/misc/host/common/prepare_amt_unified.{h,cpp}` | A 통합 패킹(NRS/RS, delta 선택, 64b 인코딩) |
| | `assets/misc/host/common/{prepare_fpga_cin_unified.h, compare_fpga_c_unified.h, host_common.h}` | Cin 패킹·결과 비교 |
| **standalone 변형(산출 스냅샷)** | `HiSpMM-{balanced,imbalanced}/src/*`, `link_config.ini` | 사전구성 80PE/64PE 빌드 |

### 1.2 호출 계층 (정식 커널, balanced=NRS 기준)
```
hispmm (top, hispmm.cpp:1)
 ├─ MM2S_A ×NUM_A_CH ────→ FIFO_A_IN ─┐               (HBM A nnz → uint64_v2, inline_tasks.async_readA)
 ├─ MM2S_B ×NUM_B_CH ────→ FIFO_B_IN ─┤
 ├─ MM2S_C ×NUM_C_CH ────→ FIFO_C_IN ─┤
 ├─ PEG ×NUM_PEG ─────────────────────┤  load_B(buff_B BRAM, N0뱅킹) + mul_AB(val*B[n] for n<N0)
 │     └─ b_out → 다음 PEG b_in       │  (systolic B forwarding, hispmm.cpp:29 FIFO_B_IN 양방향)
 ├─ DummyRead ×NUM_B_CH (detach) ─────┤  마지막 PEG의 b_out 흡수
 │   [RS만] DRDN(ADD_*/SWB*/SSW) ─────┤  Cnoc_pkt(val[8]) butterfly → Cvec_pkt
 ├─ Accumulator ×NUM_PES ─────────────┤  buffer_C[row][N0] += val[N0] (URAM, RAW dist=8)
 ├─ Arbiter (NUM_PES별 분기) ─────────┤  80=Arbiter_C_10_1×8 + Arbiter_C_8_4; 64&C4=Arbiter_C_16_1; ...
 ├─ Compute_C ×NUM_C_CH (detach) ─────┤  alpha*AB + beta*Cin
 └─ S2MM_C ×NUM_C_CH ─────────────────┘  → HBM C_out
```
근거: `hispmm.cpp:25-44`(invoke 순서), `spmm_top.py:64-98`(RS면 PEG→FIFO_C_SHF(Cnoc)→DRDN→FIFO_C_BUF(Cvec)→Accumulator), `generate.py:63-92`(arbiter 분기).

### 1.3 제외 목록(이름만)
- **vendor**: `assets/misc/host/common/mmio.h`(NIST MatrixMarket I/O), `HiSpMM-balanced/src/mmio.h`, `HiSpMM-imbalanced/src/mmio.h`. 분석 제외.
- **생성물/바이너리**: `bitstream/*.xclbin`, `*.xo`, `vitis_run_hw/`, `floorplan.json`, `HiSpMM.png`. 제외.
- **데이터**: `matrices/*.mtx`(airfoil_2d, hangGlider_3 등), `assets/common/matrices/`. 제외.
- **산출 스냅샷(구성차이만 참조)**: `HiSpMM-balanced/src/{spmm.cpp,spmm.h,spmm-host.cpp,helper_functions.*}`, `HiSpMM-imbalanced/src/{...}`. codegen 출력의 고정본이라 **원본 템플릿(`assets/tasks/*`)을 정밀 분석**하고 변형은 매크로 차이(NUM_PES 80↔64 등, `spmm.h:14-15`)만 참조.
- **모델 내장 스냅샷**: `resource_models.py`의 `builtin_area_model()`은 합성 area.log의 **내장 복제 스냅샷**(`:21-66`) — 실시간 합성 리포트가 아님(§13 PPA 주의).
- **부재(확인 불가)**: 논문 본문(PDF 미동봉)의 측정 성능/정확도, csynth/cosim 원본 리포트(repo 내 부재) → 합성 PPA 절대치는 area.log 스냅샷 외 **확인 불가**.

---

## 2. 모듈: 설계 상수 + 패킷 타입 — `hispmm.h`

### 2.1 역할 + 상위/하위
- **역할**: 전 모듈 기반 상수(채널/PE/타일/SIMD 폭)와 벡터/패킷 타입의 단일 출처. codegen이 `NUM_A_CH/NUM_B_CH/NUM_C_CH/NUM_PES/LOG_2_NUM_PES`를 패치(`spmm_header.py:26-32`).
- **상위**: 모든 task `.cpp`가 `#include "hispmm.h"`(생성 시 주입, `generate.py:120`). **하위**: `tapa.h`, `ap_int.h`(`:4-5`).

### 2.2 데이터플로우 (payload 변천)
```
HBM A (uint64_v: 8×uint64) ──MM2S_A──> uint64_v2(2×nnz) ──PEG디코딩──> {val(float),col13,row16,플래그}
HBM B (float_vB: 16 float) ──MM2S_B──> buff_B(BRAM) ──gather──> N0개 B값
PEG 곱 ─> Cvec_pkt{dummy,tileEnd,row16,val[8]} (NRS)  /  Cnoc_pkt{...,bank,sharedRow,val[8]} (RS)
Accumulator ─> float_vN(8) ──Arbiter──> float_vB(16) ──Compute_C(axpy)──> float_vB ──S2MM_C──> HBM C
```

### 2.3 대표 코드 위치
`assets/tasks/hispmm.h`(79줄). 채널/PE 상수 `:9-24`, N차원 타일 상수 `:27-38`, 벡터타입 `:40-45`, 패킷 `:47-69`, top 선언 `:71-78`.

### 2.4 대표 코드 블록
```c
#define NUM_PES 80           // hispmm.h:14 (balanced; imbalanced=64)
#define GROUP_SIZE 4         // :16 (PEG당 PE 4)
#define NUM_PEG (NUM_PES/GROUP_SIZE)  // :18 (=20)
#define II_DIST 8            // :20 (누산 RAW 의존거리)
#define N0 8                 // :27 (B 컬럼 SIMD 폭)
#define B_CHUNK_SIZE (2*N0)  // :28 (=16, B/C HBM 채널 폭)
#define K0 ((NUM_B_CH*8*1024)/N0)     // :29 (=4096, K 타일 깊이)
#define B_READ_LEN (((K0*N0)/NUM_B_CH)/B_CHUNK_SIZE)  // :31 (=512)
#define U 2                  // :32 (PE당 URAM 수)
#define MAX_ROWS_PER_PE (U*4096)      // :33 (=8192)
#define M0 (NUM_PES * MAX_ROWS_PER_PE)// :34 (M 타일 깊이)
```
→ **3D 타일링 상수**: M0(행)×K0(=4096, 축약)×N(B 컬럼). N0=8 SIMD. 모두 컴파일타임 상수 → 다른 차원 워크로드는 재튜닝.

```c
struct Cnoc_pkt { bool dummy,last,tileEnd,sharedRow; uint16_t row16; uint8_t bank; float val[8]; };  // :53-61 (RS/DRDN)
struct Cvec_pkt { bool dummy,tileEnd; uint16_t row16; float val[8]; };                                // :64-69 (NRS)
```
→ **N0=8 결과 벡터 `val[8]`** 가 SpMV(스칼라)와의 핵심 차이. RS는 `bank`(목적 누산뱅크)·`sharedRow`로 DRDN 라우팅 정보 추가.

### 2.5 마이크로아키텍처
- **메모리/폭**: `uint64_v=vec_t<uint64_t,8>`(A 채널 = 8 nnz/word, `:40`), `float_vB=vec_t<float,16>`(B/C 채널, `:45`), `float_vN=vec_t<float,8>`(PE 출력, `:43`).
- **정량**: N0=8과 NUM_PES가 전 모듈 폭의 단일 노브. NUM_PES 변경 시 PEG 수·Arbiter 토폴로지·DRDN 단수·host 패킹 레이아웃이 연동(§11~12).
- **HiSpMV 대비**: 패킷이 스칼라 val→`val[8]` 벡터. B_CHUNK_SIZE/K0/N0 같은 N차원 타일 상수가 신설.

---

## 3. 모듈: PE 그룹 (희소 핵심) — `PEG_NRS.cpp` / `PEG_RS.cpp`

### 3.1 역할 + 상위/하위
- **역할**: 희소 A nnz를 디코딩하고, 로컬 BRAM에 N0뱅킹으로 적재된 B와 곱해 **PE당 N0=8개 부분곱**을 산출. B는 PEG 간 systolic forwarding(`b_out`). NRS=행공유 없음(균형), RS=sharedRow 플래그로 무거운 행을 인접 PE 쌍에 분산(불균형).
- **상위**: `hispmm.cpp:29` `PEG ×NUM_PEG`. **하위**: 없음(BRAM·곱셈기 직접). 입력 `a_in[GROUP_SIZE/2]`(PE 2개분 묶음), `b_in[NUM_B_CH]`, 출력 `b_out[NUM_B_CH]`(다음 PEG), `c_out[GROUP_SIZE]`(`PEG_NRS.cpp:2-6`).

### 3.2 데이터플로우
```
        b_in[ch] ─load_B(II=1)─> buff_B[g][n][col%2][col/2]  (BRAM RAM_T2P, N0 완전분할)
                          └────> b_out[ch] (다음 PEG로 systolic)
a_in[g] ─mul_AB(II=1)─> a(64b) ─디코딩─> {val, col13, row16, rowEnd, tileEnd[, sharedRow]}
                                 └─ for n<N0: temp.val[n] = val * buff_B[p/2][n][col%2][col/2]
                                 └─ dummy = !rowEnd
                       (NRS) Cvec_pkt → c_out[p]
                       (RS)  sharedRow면 인접 PE쌍 row16/bank 교차 설정 → Cnoc_pkt → c_out[p]
```

### 3.3 Function call stack
`hispmm.cpp:29` → `PEG(...)` 외부 타일루프(`PIPELINE OFF`, `PEG_NRS.cpp:19-20`) → `load_B`(II=1, `:29-40`) → `mul_AB`(II=1, tileEnd까지, `:43-78`). RS는 동형 + sharedRow 분기(`PEG_RS.cpp:55,79-99`).

### 3.4 대표 코드 위치
`PEG_NRS.cpp`(82줄): buff_B `:9-14`, load_B `:29-40`, 디코딩 `:54-64`, 곱 `:66-77`. `PEG_RS.cpp`(107줄): sharedRow 디코딩 `:55`, RS 분기 `:79-99`.

### 3.5 대표 코드 블록
```c
float buff_B[GROUP_SIZE/2][N0][B_CHUNK_SIZE / N0][NUM_B_CH * B_READ_LEN]; // PEG_NRS.cpp:9
#pragma HLS array_partition variable=buff_B type=complete dim=1   // :11 (N0... 실제 dim1=GROUP/2)
#pragma HLS array_partition variable=buff_B type=complete dim=2   // :12 (N0 완전분할 → 8 동시읽기)
#pragma HLS array_partition variable=buff_B type=cyclic factor=2 dim=4 // :14
```
→ **N0=8 동시읽기 뱅킹**: dim2(N0)을 완전분할해 PE 1개가 사이클당 B의 8 컬럼을 동시에 읽어 8 MAC을 수행. SpMV엔 없는 SpMM 고유 구조.

```c
uint64_t a = temp_in[p/2][p%2];
uint32_t val_bits = a & 0xFFFFFFFF; val_in[p] = *(float*)(&val_bits);  // PEG_NRS.cpp:57-58
col_id[p] = (a >> 32) & 0x1FFF;            // :59 (13b col)
uint16_t row = (a >> 48) & 0xFFFF;         // :60 (16b row)
bool rowEnd = (a >> 46) & 1;               // :61 (행 종료 마커)
tileEnd = (temp_in[0][0] >> 47) & 1;       // :50 (타일 종료)
```
→ **64b nnz 디코딩**. host `encode()`(`prepare_amt_unified.h:74-88`)와 비트정확. rowEnd=1인 nnz만 실 데이터(`dummy=!rowEnd`, `:74`), 나머지는 lane 길이 균등화 패딩.

```c
for (int n = 0; n < N0; n++)
  temp.val[n] = val_in[p] * buff_B[p/2][n][col_id[p]%2][col_id[p]/2];  // PEG_NRS.cpp:69-71
```
→ **N0=8 SIMD MAC**(UNROLL, `:68`). PE 1개 = 8 곱/cycle. = MAC lanes 핵심.

```c
// PEG_RS.cpp: sharedRow면 인접 PE쌍(p,p+1)에 부분합 분산
temp_flag.sharedRow = (temp_in[0][0] >> 45) & 1;     // :55
if (temp_flag.sharedRow) {
  if (p % 2) { temp.row16=row_out[p]&0x7FFF; temp.bank=(uint8_t)(row_out[p-1]&((1<<LOG_2_NUM_PES)-1)); } // :80-83
  else       { temp.row16=row_out[p+1]&0x7FFF; temp.bank=(uint8_t)(row_out[p]&((1<<LOG_2_NUM_PES)-1)); } // :85-88
}
```
→ **불균형 처리 핵심**: 무거운 행을 인접 PE 쌍이 나눠 계산하고 `bank`(=목적 누산기 id)를 기록 → DRDN이 한 누산뱅크로 모음. NRS엔 sharedRow/bank 없음(`PEG_NRS.cpp:73-76`만).

### 3.6 마이크로아키텍처
- **Stage 분해**: 외부 타일루프(K타일 i 진행, `PIPELINE OFF`) → load_B(B청크를 buff_B 적재 + b_out forward, II=1) → mul_AB(nnz 디코딩 + N0 곱, II=1).
- **MAC lanes**: PEG=GROUP_SIZE(4)×N0(8)=**32 MAC/cycle**. balanced 20 PEG=640, imbalanced 16 PEG=512.
- **메모리/재사용**: buff_B 2.0Mbit/PEG, **B를 K타일 동안 재사용**(load 1회 후 mul_AB 다회). systolic forwarding(`b_out`)으로 1 채널 read를 모든 PEG가 공유(`:38`, `hispmm.cpp:29` FIFO_B_IN을 in/out 양쪽 전달).
- **정량/병목**: ① mul_AB trips = host 패킹 run_len(=`(max_lane_load+PADDING)*II_DIST`, `prepare_amt_unified.cpp:198`)에 의존 → **행 불균형이 클수록 max_lane_load↑ → 패딩 nnz↑ → 사이클 낭비**. RS가 이를 완화. ② N0 곱이 FP32 곱셈기 N0개/PE → DSP 부담(area.log PEG DSP=96, `resource_models.py:42`). ③ `printf`(`:42`)는 csim용, 합성 시 무시(추정).
- **HiSpMV 대비**: SpMV는 PE당 1 MAC·스칼라 누산. HiSpMM은 PE당 N0=8 MAC + buff_B N0뱅킹 + RS/NRS 2변형. RS의 sharedRow→bank→DRDN 경로가 신설.

---

## 4. 모듈: 출력 누산 — `Accumulator.cpp`

### 4.1 역할 + 상위/하위
- **역할**: PE 1개당 1 인스턴스. 행별 N0=8 벡터를 URAM `buffer_C`에 `+=` 누산하고 M 타일 끝에 dump. RAW 의존거리 8로 II=1 유지.
- **상위**: `hispmm.cpp:31` `Accumulator ×NUM_PES`. 입력은 NRS면 PEG 직결(`FIFO_C_SHF`, Cvec), RS면 DRDN 출력(`FIFO_C_BUF`, Cvec)(`spmm_top.py:85-86`). **하위**: 없음(URAM 직접).

### 4.2 데이터플로우
```
M타일 루프(i ≤ last_tile_idx, PIPELINE OFF):
  init_c: buffer_C[m][n]=0 for m<read_len, n<N0   (II=1)
  acc_c : Cvec_pkt 읽어 !dummy면 buffer_C[row16][n]+=val[n] for n<N0  (II=1, RAW dist=8)
  updt_C: c_out << buffer_C[l] for l<read_len      (II=1)
```

### 4.3 대표 코드 위치
`Accumulator.cpp`(51줄): buffer_C `:7-8`, M타일 루프 `:12-50`, init `:18-21`, acc `:23-41`, dump `:45-48`.

### 4.4 대표 코드 블록
```c
float_vN buffer_C[MAX_ROWS_PER_PE];                       // :7 (=8192행 × N0)
#pragma HLS bind_storage variable=buffer_C type=RAM_2P impl=URAM  // :8
...
acc_c:for(bool tileEnd=false; !(tileEnd); ) {
  #pragma HLS PIPELINE II=1                                // :24
  #pragma HLS dependence variable=buffer_C inter RAW distance=8 true  // :25 (=II_DIST)
  Cvec_pkt temp_in = c_in.read();
  if (!dummy) for(int n=0;n<N0;n++) buffer_C[m][n] += temp_in.val[n];  // :35-40 (m=row16)
}
```
→ **누산 핵심**: 같은 행(=같은 buffer_C 주소)이 8 사이클 안에 재방문되지 않도록 host가 II_DIST=8 lane으로 스케줄(`prepare_amt_unified.cpp:162-183`의 `Loads[NUM_PES][II_DIST]` 최소부하 배치) → FP32 가산 latency를 8단 의존거리로 흡수해 II=1. **HiSpMV의 circular-buffer write-forwarding을 단순 `+=` + dependence pragma로 대체**.

### 4.5 마이크로아키텍처
- **메모리**: URAM 2.1Mbit/PE (`buffer_C[8192]×256b`). NUM_PES개 = balanced 168Mbit / imbalanced 134Mbit (단순곱, 합성 매핑은 area.log URAM/PE=8, `resource_models.py:30`).
- **정량/병목**: ① init_c+updt_C가 M타일마다 read_len(최대 8192) 사이클 고정 오버헤드. ② acc_c는 패딩 nnz(dummy)도 읽지만 `+=` 안 함 → 패딩이 사이클은 소비(throughput에 불균형 페널티 반영, §12). ③ RAW distance=8이 II_DIST와 강결합 — host 스케줄이 이를 보장해야 정확.
- **HiSpMV 대비**: 누산이 스칼라→N0=8 벡터(URAM 폭 8×float). forwarding 큐 대신 dependence distance 방식.

---

## 5. 모듈: Dense Row Distribution Network (DRDN) — `{ADD_*, SWB*, SSW}.cpp` + `crossbar.py` (RS 전용)

### 5.1 역할 + 상위/하위
- **역할**: RS 커널에서만 활성. PEG가 인접 PE 쌍에 분산 계산한 **shared row의 N0=8 부분합**을 butterfly add+switch 네트워크로 한 누산뱅크(`bank`)에 모음. NRS는 DRDN 없이 PEG→Accumulator 직결.
- **상위**: `spmm_top.py:92-98`가 DummyRead 뒤·Accumulator 앞에 DRDN invoke 주입(`generate.py` RS 경로). 토폴로지는 `crossbar.py`/`drdn.py`가 NUM_PES로 생성. **하위**: 없음(가산기/스위치).
- 블록: `ADD_0/ADD_1/ADD_X`(가산), `SWB0_n/SWB1_n`(bank 비트 라우팅, Cnoc→Cvec 변환), `SSW`(shared swap).

### 5.2 데이터플로우 (butterfly 2-phase)
```
FIFO_C_SHF[*] (Cnoc_pkt, val[8]) ─forward phase(ADD: sharedRow면 두 입력 합)─> s_*_* (Cnoc)
                                  ─backward phase(SWB: bank 비트로 라우팅, 끝단 Cnoc→Cvec)─> FIFO_C_BUF[*] (Cvec)
SSW: 두 입력을 sharedRow 비트로 교차(swap)
```

### 5.3 Function call stack
`generate.py:123` `build_drdn_graph(num_pes)` → `crossbar.CrossBarGen(n).buildGraph()`(`drdn.py:42-45`) → 노드별 `.invoke(fn, incoming..., outgoing...)`(`spmm_top.py:40-46`). 필요 task 파일은 `required_task_files_for_drdn`(`drdn.py:71-74`).

### 5.4 대표 코드 위치
`crossbar.py`(143줄): forward phase `:73-104`, backward phase `:106-142`, 스트림 깊이 `:21-30`. `ADD_0.cpp`(64줄), `SWB0_0.cpp`(27줄), `SSW.cpp`(21줄). `drdn.py`(76줄).

### 5.5 대표 코드 블록
```c
// ADD_0: sharedRow & 둘 다 !dummy면 두 입력 N0 합을 한쪽에 모음
for (int p=0;p<8;p++) sum[p] = curr_in0.val[p] + curr_in1.val[p];        // ADD_0.cpp:14-15
if ((curr_in0.sharedRow) & !(curr_in0.dummy|curr_in1.dummy)) {
  temp[0]=sum; dummy[0]=false; temp[1]=0; dummy[1]=true;                  // :17-26 (한 쪽으로 병합)
}
```
→ **shared row 부분합 합산**(N0=8 벡터 단위). HiSpMV DRDN과 동형이나 운반 단위가 스칼라→`val[8]`.

```c
// SWB0_0: bank 최하위 비트로 출력 라우팅 + Cnoc→Cvec 변환(끝단)
bool i = (curr_in0.bank & 1) && curr_in0.sharedRow;                       // SWB0_0.cpp:10
curr_out[i] = {curr_in0.dummy, ..., curr_in0.val};                        // :11-20
```
→ butterfly backward 단에서 `bank` 비트로 목적 PE 누산기로 정렬, 출력 타입을 Accumulator 입력형(Cvec)으로 변환.

```python
# crossbar.py: 스트림 깊이 = 가산 latency(+6) 보정
depth = (end - start) * 2                                                 # crossbar.py:25
if (i%2==1) and (i < 2*floor(log2(n))): depth += 6                        # :27-28
```
→ DRDN 중간 FIFO 깊이를 단계별 가산 latency를 고려해 산출(deadlock 방지). `spmm_top.py:36`이 이를 `tapa::stream<Cnoc_pkt, depth>`로 선언.

### 5.6 마이크로아키텍처
- **정량/병목**: ① 블록 수는 NUM_PES의 함수(`drdn_counts(num_pes)`, `resource_models.py:96-118`) → 64 PE에서 ADD/SWB/SSW 다수 인스턴스. ② 각 블록 II=1 패스스루지만 butterfly 깊이만큼 latency 추가(파이프라인이라 throughput 영향은 적음, 추정). ③ DRDN은 LUT/FF 비용(area.log SWB0_1 LUT≈2138, `resource_models.py:47`) → imbalanced가 PE를 64로 줄인 이유(80PE+DRDN은 자원 초과 추정).
- **HiSpMV 대비**: 구조는 동형(butterfly add+switch), 운반 단위가 N0=8 벡터. RS 변형에서만 존재(NRS=없음).

---

## 6. 모듈: 후처리 — Arbiter / Compute_C / S2MM_C

### 6.1 역할 + 상위/하위
- **Arbiter**: NUM_PES개 `float_vN`(=N0) 스트림을 NUM_C_CH 폭 `float_vB`(=16=2×N0)로 직렬화/패킹. NUM_PES별 분기.
- **Compute_C**: `alpha*AB + beta*Cin` (axpy). **S2MM_C**: C 결과를 HBM에 packed write.
- **상위**: `hispmm.cpp:32-43`. **하위**: 없음. 모두 `tapa::detach`(Arbiter/Compute_C) 또는 join(S2MM).

### 6.2 대표 코드 위치/블록
```c
// Arbiter_C_10_1: 10개 float_vN을 순회 직렬화 (80PE 1단)
for (int i=0;i<10;i++){ temp_in=c_arb[i].read(); c_ab<<temp_in; }         // Arbiter_C_10_1.cpp:9-12
// Arbiter_C_8_4: 8개 float_vN을 NUM_C_CH(4) × float_vB(16)로 패킹 (80PE 2단)
temp_out[j/2][(j%2)*N0 + k] = temp_in[k];                                 // Arbiter_C_8_4.cpp:12
```
→ **80PE arbiter는 2단 트리**: `Arbiter_C_10_1`×8(80=8×10) → `Arbiter_C_8_4`(8→4채널). `FIFO_C_AB_INTER`(8 스트림) 경유(`hispmm.cpp:21-23,33-34`). 64PE는 단일단(`Arbiter_C_16_1` 또는 `_8_1`, `:36-38`).

```c
// Compute_C: backpressure 안전 axpy
if (!c_ab.empty() && !c_in.empty()) {
  c_ab.try_read(temp0); c_in.try_read(temp1);
  for(int p=0;p<B_CHUNK_SIZE;p++) temp2[p]=alpha*temp0[p]+beta*temp1[p];  // Compute_C.cpp:9-14
}
```
→ axpy 후처리. B_CHUNK_SIZE(16) 폭 SIMD. HiSpMV Compute_C와 동형의 N차원 확장.

```c
// MM2S_B / S2MM_C: K(또는 M) 타일이 채널당 B_READ_LEN(또는 C_READ_LEN) 주소 점유
int start_addr = (k*numTilesN*B_READ_LEN) + (n*read_len);                 // MM2S_B.cpp:13
#pragma HLS loop_flatten OFF                                              // :11 (타일 경계 유지)
```

### 6.3 마이크로아키텍처
- **정량/병목**: ① Arbiter 80PE 2단 트리가 NUM_C_CH=4로 모음 → C 출력 대역폭 = 16 float/채널×4=64 float/cycle 이상치. ② Compute_C는 `try_read` empty 체크로 deadlock/backpressure 회피(detach 무한루프). ③ S2MM_C는 packed model(실 M행만 write, `async_writeC`, `inline_tasks.cpp:73-102`). ④ Arbiter가 NUM_PES별로 4종(10_1+8_4 / 8_1 / 16_1 / monolithic a4·a6) → codegen이 정확히 선택(`generate.py:63-92`, hispmm.cpp `#if`와 일치).
- **HiSpMV 대비**: float_vN(N0=8) 운반, Arbiter가 N0→2N0 패킹, axpy가 16 폭. 구조 골격은 동형.

---

## 7. 모듈: HBM IO — `MM2S_A/B/C.cpp` · `S2MM_C.cpp` · `inline_tasks.cpp` · `DummyRead.cpp`

### 7.1 역할 + 상위/하위
- **역할**: HBM ↔ 온칩 스트림 브리지. async_mmap 비차단 req/resp. A=8 nnz/word를 PE 2개분(`uint64_v2`)으로 분배, B/C=16 float/word.
- **상위**: `hispmm.cpp:26-43`. **하위**: `inline_tasks.cpp`의 `async_readA/B/C`, `async_writeC`(inline 헬퍼).

### 7.2 대표 코드 블록
```c
// async_readA: A word(8 nnz) → PE 2개분 묶음 8개로 분배
A.read_data.try_read(tmp);
for(int a=0;a<PES_PER_CH/2;a++){ uint64_v2 t; t[0]=tmp[a*2]; t[1]=tmp[a*2+1]; fifo_A[a].try_write(t); } // inline_tasks.cpp:24-28
```
→ 채널 1 word(8 nnz)가 PE 8개(=PES_PER_CH)에 2개씩 공급. backpressure는 `full` 체크(`:15-20`).

```c
// DummyRead: 마지막 PEG의 b_out 흡수 (systolic 종단)
for(;;){ float_vB temp = b_in.read(); }                                    // DummyRead.cpp:5-8
```
→ B systolic forwarding 체인의 종단. detach 무한루프(`hispmm.cpp:30`).

### 7.3 마이크로아키텍처
- **정량/병목**: ① async_mmap의 req/resp 분리로 HBM latency 은닉(II=1, `MM2S_*.cpp`). ② A는 채널당 8 nnz/cycle 공급(=PES_PER_CH), B/C는 16 float/cycle. ③ `loop_flatten OFF`로 타일 경계 burst 유지(`MM2S_B.cpp:11`, `S2MM_C.cpp:9`). ④ DummyRead는 자원 거의 0(area.log LUT≈569, `resource_models.py:38`).
- **HiSpMV 대비**: A 워드 폭(8 nnz)·B/C 16 float 채널은 SpMM의 dense B/C 처리를 위한 확장. SpMV는 dense 벡터 1열.

---

## 8. 모듈: 코드생성 오케스트레이터 — `generate.py`

### 8.1 역할 + 상위/하위
- **역할**: DSE 추천(또는 --pick)을 받아 ① `hispmm.h` 매크로 패치 ② `hispmm.cpp`(task 템플릿 concat + top 패치, RS면 DRDN 주입) ③ host 소스(A/C config 마커 패치) ④ Makefile/link_config/floorplan 에셋 ⑤ 행렬 복사를 자체 완결 빌드 폴더로 합성.
- **상위**: `automation_tool/__main__.py`(CLI). **하위**: `advisor`, `drdn`, `crossbar`, `spmm_header/top`, `templates`, `misc_assets`.

### 8.2 대표 코드 위치/블록
```python
def _base_task_files(*, rs_capable):
  base = [inline_tasks, MM2S_A/B/C, DummyRead, Accumulator, Compute_C, S2MM_C]
  base.append("PEG_RS.cpp" if rs_capable else "PEG_NRS.cpp")   # generate.py:59
```
→ **balanced=NRS, imbalanced=RS**. RS면 추가로 SWB0/SWB1 + DRDN 그래프가 요구하는 task(`required_task_files_for_drdn`)를 합성(`:138-146`).

```python
def _arbiter_task_files(*, a_ch, c_ch, num_pes):   # generate.py:63-76
  if num_pes==80: return [Arbiter_C_10_1, Arbiter_C_8_4]
  if num_pes==64 and c_ch==8: return [Arbiter_C_8_1]
  if num_pes==64 and c_ch==4: return [Arbiter_C_16_1]
  if a_ch==4: return [Arbiter_C-a4]; if a_ch==6: return [Arbiter_C-a6]
```
→ hispmm.cpp의 `#if NUM_PES==80 ... #elif ...` 분기(`hispmm.cpp:32-41`)와 정확히 일치. codegen이 전처리 ladder를 단일 invoke로 치환(`spmm_top.py:100-104`).

```python
def _host_c_layout(*, a_ch):                        # generate.py:261-269
  if a_ch in (4,6): return (LinearChunkInterleave, None)
  if a_ch==8:  return (TiledPackedRows, AdjacentPair)
  if a_ch==10: return (TiledPackedRows, HalfGroupPair)
```
→ **A_CH별 C/Cin 메모리 레이아웃·행쌍(pairing)이 자동 선택**되어 host에 마커 주입(`@CODEGEN:PREPARE_C_CONFIG`, `:357-362`). HW 선택과 SW 패킹의 일관성 보장.

### 8.3 마이크로아키텍처
- **정량/병목**: ① task 파일 stable-unique 정렬 후 concat(`:148-160`) → 정의 순서 의존성(inline_tasks가 MM2S 앞, `:49`) 관리. ② RS면 DRDN 그래프가 동적으로 task 집합·top invoke를 결정 → 커널이 NUM_PES에 따라 구조 변동. ③ floorplan.json 부재 시 Makefile에서 pre-assignment 제거(`:177-194`) — graceful degrade.
- **HiSpMV 대비**: HiSpMV(전신)엔 없던 **DSE 연동 codegen 전체**가 신설. 커널과 host 패킹을 하나의 결정으로 동기화하는 것이 본 repo의 가장 큰 구조적 차별점.

---

## 9. 모듈: DSE 진입점 + 사이클 모델 — `dse/cycle_analysis.py` · `cycle_models.py`

### 9.1 역할 + 상위/하위
- **역할**: 행렬(.mtx)+N을 받아 변형 집합을 sweep, 각 변형의 delta·run-length·3-term 사이클·실측 주파수로 `time_us`를 추정하고 최소를 `RECOMMEND`. 리소스 추정 시 fit 필터.
- **상위**: `advisor._run_cycle_analysis`(subprocess, `advisor.py:56-83`). **하위**: `cycle_models`, `load_models`, `resource_models`, `mm_parser`.

### 9.2 대표 코드 위치/블록
```python
# 변형 집합 (cycle_analysis.py:54-89)
balanced → [(balanced_a10_cN, A=10, NRS)]
imbalanced → [(imbalanced_a8_cN, A=8, RS-capable)]
sweep → a_ch∈{4,6,8,10} × c_ch∈{4,8} × {NRS, RS}(비실용 점 제외)
```

```python
# 실측 주파수 테이블 (cycle_analysis.py:30-41)
balanced_a10_c4: 216.0   balanced_a8_c8: 204.0   (그 외 225.0 기본)
```
→ **실측 주파수로 time 계산**(`time_us = total / freq_mhz`, `:216`). 비측정 sweep 점은 225MHz 기본 → 그 점들은 추정.

```python
# 3-term 사이클식 (cycle_models.py:26-37)
SB = 60  # 타일당 셋업 경험상수
t1 = ((k0*n)/(16*b_ch) + SB*ceil(n/n0)) * ceil(k/k0) * ceil(m/m0)   # B 이동
t2 = (외부 주입: run-length 또는 (m*k*rho)/(a_ch*pes_per_ch)*(n/n0)*(1+delta))  # 연산
t3 = (m*n)/(16*c_ch)                                                 # C 이동(packed)
```
→ **t1(B이동)/t2(연산)/t3(C이동) roofline 모델**. t2가 delta(불균형)와 (n/n0)(N0 SIMD)에 비례 → SpMM의 핵심 비용 항.

```python
nrs_penalty = 1.03 if (can_row_share and chosen=="no_row_sharing") else 1.0  # cycle_analysis.py:166
total = (t1+t2+t3) * nrs_penalty                                              # :214
```

### 9.3 마이크로아키텍처
- **정량/병목**: ① 정렬 키 `(fit, time_us, -a_ch)`(`:271-279`) → 자원 맞으면서 빠르고 A채널 큰 것 선호. ② term2 모델 2종(runlen 기본 / nnz_delta, `:106,197`). ③ 경험 상수(SB=60, NRS 1.03 페널티) → 디바이스/툴 변하면 부정확(추정). ④ 순수 파이썬(numpy 미사용, 리스트 연산) → 대형 행렬은 느릴 수 있음(추정).

---

## 10. 모듈: 불균형(delta)·run-length 모델 — `dse/load_models.py`

### 10.1 역할 + 상위/하위
- **역할**: 타일별 행을 PE의 II_DIST lane에 그리디 배치해 PE 부하 분산(stddev/mean=**delta**)을 산출하고, NRS(delta1)/RS(delta2)를 비교해 개선≥25%면 RS 채택. host의 실제 밸런싱 로직을 파이썬으로 포팅한 추정 모델.
- **상위**: `cycle_analysis.py:136-194`. **하위**: `mm_parser.MatrixShape`.

### 10.2 대표 코드 블록
```python
# NRS delta: 행을 row%num_pes PE의 최소부하 lane에 배치, PE부하 stddev/mean
loads = [[0]*ii_dist for _ in range(num_pes)]
for row_id,row_size in sorted_rows: pe=row_id%num_pes; loads[pe][argmin]+=row_size  # load_models.py:69-80
delta1 = sqrt(var)/mean                                                              # :95

# RS: 무거운 행을 shared로 빼며 imbalance 개선 추적
imb = (scheduled - total)/total                                                      # :124
if (imp := (delta1-delta2)/delta1*100) >= 25.0: chosen="row_sharing"                # :245-246
```
→ **25% 임계가 generate.py(`_host_a_cfg`)·host(`prepare_amt_unified.cpp:593`)와 일치**. 셋이 동일 휴리스틱.

```python
# run-length: lane 최대버킷 + 패딩, t2 = run_len*(N/N0)
run_len += (max_bucket+padding)*ii_dist                                              # :363,406
t2_runlen = run_len * (n/n0)                                                         # :410
```
→ **legacy host run-length 모델 재현**: 가장 긴 lane이 tile 사이클을 결정(=`prepare_amt_unified.cpp:198`의 `(max_lane_load+PADDING)*II_DIST`와 동형).

### 10.3 마이크로아키텍처
- **정량/병목**: ① delta = PE 부하 변동계수. SpMM 불균형의 정량 지표. ② RS 스케줄: shared 행을 ceil(row_size/num_pes)씩 전 PE에 분산(`:204`), 나머지는 NRS 규칙 → DRDN 비용 vs 부하 균형 트레이드오프. ③ **py(추정)와 C++(`prepare_amt_unified.cpp`, 실구현)가 이원화** → 미세 불일치 가능(추정). ④ NRS의 term2 delta는 tile별 재계산이 비효율적 코드(`:296` 전체 delta 재호출) — 성능 이슈(추정).
- **HiSpMV 대비**: delta 기반 NRS/RS 자동 선택은 HiSpMM 신설 기능. 불균형 워크로드 적응이 본 repo의 핵심 기여(논문 키워드 "Imbalanced Workload").

---

## 11. 모듈: top 패치 + 헤더 패치 + DRDN 토폴로지 — `spmm_top.py` · `spmm_header.py` · `crossbar.py`/`drdn.py`

### 11.1 역할 + 대표 블록
- **spmm_header.py**: `#define NUM_A_CH/.../NUM_PES/LOG_2_NUM_PES`를 정규식 치환(`:24-49`). `LOG_2_NUM_PES=ceil(log2(num_pes))`(`:18`).
- **spmm_top.py**: RS면 FIFO_C_SHF를 Cnoc_pkt로, FIFO_C_BUF(Cvec) 추가, DRDN 중간 스트림 선언·invoke 주입, Accumulator 입력을 FIFO_C_BUF로 교체, arbiter ladder를 선택 invoke로 치환(`:64-104`).
```python
if patch.rs_capable:
  fifo_block = 'tapa::streams<Cnoc_pkt,...> FIFO_C_SHF; tapa::streams<Cvec_pkt,...> FIFO_C_BUF;'  # :65-67
  out.replace("Accumulator, FIFO_C_ARB, FIFO_C_SHF,", "Accumulator, FIFO_C_ARB, FIFO_C_BUF,")    # :85-86
```
→ **NRS: PEG→FIFO_C_SHF(Cvec)→Accumulator 직결. RS: PEG→FIFO_C_SHF(Cnoc)→DRDN→FIFO_C_BUF(Cvec)→Accumulator.** 동일 템플릿에서 2변형을 텍스트 패치로 분기.
- **crossbar.py/drdn.py**: NUM_PES butterfly 그래프(forward ADD phase + backward SWB phase + SSW), 스트림 깊이(가산 latency +6 보정, `crossbar.py:27-28`)를 생성. `drdn.build_drdn_graph`가 노드/깊이를 dataclass로 래핑(`drdn.py:42-68`).

### 11.2 마이크로아키텍처
- **정량/병목**: 텍스트 정규식 패치라 템플릿의 마커/구조 변경 시 취약(추정). DRDN 그래프가 NUM_PES에 따라 task 집합·스트림 수를 동적 결정 → 커널 구조가 변형별로 상이.

---

## 12. 모듈: 호스트 패킹·검증 — `hispmm_host.cpp` · `prepare_amt_unified.{h,cpp}` · `prepare_fpga_cin_unified.h` · `compare_fpga_c_unified.h`

### 12.1 역할 + 상위/하위
- **역할**: ① .mtx→CSR→tile→**64b nnz 패킹**(NRS/RS, delta로 RS on/off) ② B/Cin 패킹(A_CH별 레이아웃) ③ XRT 실행·CPU 참조 비교. codegen이 A/C config를 마커로 주입(`generate.py:298-362`).
- **상위**: CLI(`README.md:182-191`). **하위**: `mmio.h`(vendor), `tapa`.

### 12.2 대표 코드 블록
```c
// 64b nnz 인코딩 (host) — PE 디코딩과 비트정확
inline uint64_t encode(bool tileEnd,bool rowEnd,bool sharedRow,uint16_t row,int col,uint32_t val){
  res|=row; res<<=1; res|=tileEnd; res<<=1; res|=rowEnd; res<<=1; res|=sharedRow;
  res<<=13; res|=col&0x1FFF; res<<=32; res|=val; }                       // prepare_amt_unified.h:74-88
```
→ `[63:48]row | [47]tileEnd | [46]rowEnd | [45]sharedRow | [44:32]col | [31:0]val`. PE(`PEG_RS.cpp:53-66`)와 정합.

```c
// delta 기반 RS 자동 선택 (host 실구현)
result.delta_improvement_percent = (no_rs.delta - with_rs.delta)/no_rs.delta*100.0;  // prepare_amt_unified.cpp:592
use_rs = (improvement >= cfg.delta_improvement_threshold_percent);  // :593 (=25.0, generate.py:306)
```
→ **DSE(py)와 동일한 25% 임계**. py는 추천용, 여기는 실제 패킹 결정.

```c
// II_DIST lane 최소부하 배치 + 패딩으로 lane 균등화 + tileEnd 마커
const int addr = curr_tile_offset + (((Loads[pe][min_idx]*II_DIST)+min_idx)*PES_PER_CH)+inter_ch_pe;  // :381
while (Loads[p][ii] < (curr_tile_size/II_DIST)) { ...encode(tileEnd,...) }                              // :392-401
```
→ host가 PE×II_DIST lane을 균등화하고 패딩 nnz로 tile 길이를 맞춤 → HW(Accumulator RAW dist=8, PEG run_len)와 정합.

### 12.3 마이크로아키텍처
- **정량/병목**: ① RS 패킹은 shared 행을 전 PE에 ceil(row/NUM_PES) 분산하며 row16에 `(pe&1)?rowh16:rowl16` 교차 저장(`:474`) → PEG_RS의 인접쌍 디코딩(`:80-88`)과 짝. ② OpenMP 병렬(`#pragma omp parallel for`, `:67,155`) — 전처리 가속. ③ run_len = `fpgaAmtx[0].size()/PES_PER_CH`(`:623`)가 MM2S_A `len` 인자로 전달.
- **HiSpMV 대비**: SpMV는 단순 nnz 스트림. HiSpMM은 64b에 row16/tileEnd/rowEnd/sharedRow 제어비트 + A_CH별 C 레이아웃 + RS 패킹이 신설.

---

## 13. 모듈: 리소스 추정 — `dse/resource_models.py` (PPA 주의)

### 13.1 역할 + 대표 수치
- **역할**: 내장 area.log 스냅샷으로 task별 면적을 NUM_PES/채널로 스케일해 U280 용량 대비 utilization 추정. RS-only task는 `drdn_counts(num_pes)`로 인스턴스 수 결정.
- **U280 용량(내장)**: BRAM 3504, DSP 8496, FF 2,331,840, LUT 1,165,920, URAM 960 (`resource_models.py:12-18`).
- **task당 면적(내장 스냅샷, 예)**: PEG = BRAM 128 / DSP 96 / URAM 0 (`:42`); Accumulator = URAM 8 / scale NUM_PES (`:30`); Compute_C = DSP 128 (`:37`); Arbiter_C_10_1 = LUT≈5609 (`:35`); SWB0_1 = LUT≈2138 (`:47`).

### 13.2 PPA 주의
- 이 수치는 **area.log의 내장 복제 스냅샷**(`builtin_area_model`, `:21-66`)이며 **실시간 합성 리포트가 아님**. 본 repo에 csynth/cosim 원본 리포트·논문 측정 PPA는 **부재** → 합성 PPA 절대치는 **이 스냅샷 외 확인 불가**. utilization은 추정 도구의 입력값.
- `estimate_kernel_area`(`:163-202`): PEG는 `num_pes/group_size`배, RS-only task는 crossbar 카운트배, Arbiter_C_8_4는 80PE에서만 1 → fit 판정에 사용.

---

## 14. 모듈 한눈 요약 표

| 모듈 | 파일 | 핵심 함수/심볼(라인) | 역할 | 대표 정량 |
|---|---|---|---|---|
| 상수/타입 | hispmm.h | NUM_PES(:14), N0(:27), 패킷(:53-69) | 설계 상수·val[8] 패킷 | N0=8, K0=4096, M0=NUM_PES×8192 |
| PE 그룹 | PEG_NRS/RS.cpp | mul_AB(:43), 곱(:69-71), RS분기(RS:79) | N0=8 SIMD 곱 + 행공유 | PEG=32 MAC/cycle, buff_B 2.0Mbit |
| 누산 | Accumulator.cpp | acc_c(:23), +=(:38), RAW8(:25) | 행별 N0 URAM 누산 | URAM 2.1Mbit/PE, II=1 |
| DRDN | ADD_*/SWB*/SSW.cpp, crossbar.py | ADD_0(:14), SWB0_0(:10) | shared row butterfly 합산 | RS 전용, depth +6 보정 |
| 후처리 | Arbiter_C_*/Compute_C/S2MM_C.cpp | axpy(:13), Arbiter(:9) | 직렬화·axpy·HBM write | 80PE 2단 arbiter |
| HBM IO | MM2S_*/inline_tasks.cpp | async_readA(:2) | async_mmap 브리지 | A=8nnz/word, B/C=16float |
| codegen | generate.py | _base_task_files(:47), _host_c_layout(:261) | 커널+host 합성 | NRS=balanced, RS=imbalanced |
| DSE | cycle_analysis.py, cycle_models.py | _build_runs(:54), cycle_terms(:26) | 변형 sweep·3-term·time | SB=60, freq 216/204/225MHz |
| delta 모델 | load_models.py | delta_auto(:152), 25%(:246) | NRS/RS 불균형 선택 | delta=stddev/mean, 임계 25% |
| top/헤더 패치 | spmm_top.py, spmm_header.py | patch_hispmm_cpp(:49) | 2변형 텍스트 분기 | Cnoc↔Cvec, DRDN 주입 |
| host 패킹 | prepare_amt_unified.{h,cpp} | encode(:74), use_rs(:593) | 64b nnz·delta 패킹 | 64b layout, II_DIST 균등화 |
| 리소스 | resource_models.py | builtin_area_model(:21) | PPA 추정(스냅샷) | URAM 960, PEG DSP 96 |

---

## 15. 읽기 순서 / 코드 추적 순서

1. **상수/타입**: `hispmm.h` — NUM_PES/N0(:14,27)·타일(:27-38)·Cnoc/Cvec 패킷(:53-69). 모든 모듈 공통어.
2. **top dataflow**: `hispmm.cpp` invoke 순서(:25-44) → `spmm_top.py:64-98`(NRS/RS 분기) → 전체 파이프 골격.
3. **희소 핵심 PE**: `PEG_NRS.cpp` N0 곱(:69-71)·buff_B 뱅킹(:9-14)·64b 디코딩(:54-64) → `PEG_RS.cpp` sharedRow/bank(:55,79-99).
4. **누산**: `Accumulator.cpp` `+=` + RAW dist=8(:25,38) — II=1의 본질.
5. **불균형 경로**: `crossbar.py`/`drdn.py` butterfly → `ADD_0.cpp`/`SWB0_0.cpp`/`SSW.cpp` (RS 전용 DRDN).
6. **후처리/IO**: `Arbiter_C_10_1`/`Arbiter_C_8_4`(2단 트리) → `Compute_C.cpp`(axpy) → `MM2S_*`/`inline_tasks.cpp`(async_mmap).
7. **codegen**: `generate.py`(:47-92,261-269) — task 선택·arbiter 분기·host 레이아웃. `spmm_header.py`·`spmm_top.py`(패치).
8. **DSE**: `cycle_analysis.py`(:54-89,199-321) → `cycle_models.py`(3-term) → `load_models.py`(delta/run-length, 25% 임계).
9. **HW/SW 정합**: `prepare_amt_unified.h:74-88` `encode()` ↔ `PEG_RS.cpp:53-66` 디코딩 비트정확 비교. host delta 결정(`prepare_amt_unified.cpp:593`) ↔ DSE delta(`load_models.py:246`) 동일 임계.
10. **자원**: `resource_models.py`(:12-66) — 용량·task 면적 스냅샷(PPA는 추정·스냅샷 한정).

---

## 16. 병목 후보 & 병렬도/DSE 노브

### 16.1 병목 후보
1. **행 불균형 → 패딩 사이클 낭비**(`prepare_amt_unified.cpp:198,388-403`, `Accumulator.cpp:35`): lane을 `(max_lane_load+PADDING)*II_DIST`로 균등화하므로 가장 무거운 lane이 tile 사이클을 결정. dummy 패딩은 PEG/Accumulator 사이클을 소비하나 MAC은 안 함 → **불균형 워크로드의 근본 비용**. RS+DRDN이 완화하나 DRDN 자원 비용 발생.
2. **NRS vs RS 트레이드오프**(`load_models.py:152-247`): RS는 부하 균형(delta↓)을 얻지만 DRDN(ADD/SWB/SSW) LUT/FF 비용 + PE를 80→64로 축소. 25% 미만 개선이면 NRS가 유리. **변형 선택이 행렬 sparsity 패턴에 강하게 의존**.
3. **B systolic forwarding 체인 latency**(`hispmm.cpp:29`, `DummyRead.cpp`): NUM_PEG단 forwarding이라 첫 PEG와 마지막 PEG의 B 도착 시점 차이. 파이프라인이라 throughput 영향은 작으나 fill latency 존재(추정).
4. **Accumulator init_c+updt_C 고정 오버헤드**(`Accumulator.cpp:18-21,45-48`): M타일마다 최대 read_len(8192) 사이클씩 0초기화/dump. M0보다 작은 행렬은 부분(read_len)이나 타일 多면 누적.
5. **FP32 곱셈기 DSP 부담**(area.log PEG DSP=96, `resource_models.py:42`): PE당 N0=8 FP32 곱 → DSP가 자원 한계가 되기 쉬움. imbalanced가 64PE로 줄인 주 이유(추정).
6. **codegen 텍스트 패치 취약성**(`spmm_top.py` 정규식): 템플릿 마커/구조 변경 시 패치 실패(예외). 유지보수 부담(추정).
7. **DSE 경험 상수 일반화**(`cycle_models.py:26` SB=60, `cycle_analysis.py:166` 1.03, `resource_models.py` area 스냅샷): 측정된 2점(a10/a8) 외 sweep 점·다른 디바이스는 추정 정확도 저하.

### 16.2 병렬도/DSE 노브
- **NUM_PES(80/64)**(`hispmm.h:14`): PE 병렬도. 변경 시 PEG 수·Arbiter 토폴로지·DRDN 단수·host 레이아웃 연동(codegen이 일괄 처리).
- **N0(=8)**(`hispmm.h:27`): B 컬럼 SIMD 폭. MAC lanes=NUM_PES×N0의 직접 인자. buff_B 뱅킹·val[8] 폭·DSE (n/n0) 항 연동.
- **NUM_A_CH(10/8) / NUM_C_CH(4/8) / NUM_B_CH(4)**(`hispmm.h:9-11`): HBM 채널 분배. A_CH가 변형(balanced a10 / imbalanced a8)을 좌우하고 C 레이아웃 선택(`generate.py:261-269`)을 결정.
- **II_DIST(=8)**(`hispmm.h:20`): 누산 RAW 의존거리 = lane 수. FP32 가산 latency 흡수. host 스케줄과 강결합.
- **K0(4096)/MAX_ROWS_PER_PE(8192)/B_READ_LEN(512)**(`hispmm.h:29,31,33`): 3D 타일 입도. URAM/BRAM 용량 vs 타일 수 트레이드오프.
- **RowSharingPolicy(kAuto/kForceDisabled/kForceEnabled) + delta 임계(25%)**(`prepare_amt_unified.h:125-153`, `generate.py:306`): RS on/off 휴리스틱. 행렬별 자동 결정의 핵심 노브.
- **--variant(balanced/imbalanced/both/all/sweep) + --term2-model(runlen/nnz_delta) + --resource-limit**(`cycle_analysis.py:100,106,113`): DSE 탐색 범위·모델·자원 필터.

---

*근거 파일(절대경로)*:
`\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\Others\HiSpMM\automation_tool\assets\tasks\{hispmm.h,hispmm.cpp,PEG_NRS.cpp,PEG_RS.cpp,Accumulator.cpp,Compute_C.cpp,Arbiter_C_10_1.cpp,Arbiter_C_8_4.cpp,MM2S_A.cpp,MM2S_B.cpp,MM2S_C.cpp,S2MM_C.cpp,inline_tasks.cpp,DummyRead.cpp,ADD_0.cpp,SWB0_0.cpp,SSW.cpp}`,
`...\automation_tool\{generate.py,advisor.py,spmm_top.py,spmm_header.py,crossbar.py,drdn.py}`,
`...\automation_tool\dse\{cycle_analysis.py,cycle_models.py,load_models.py,resource_models.py}`,
`...\automation_tool\assets\misc\host\common\{prepare_amt_unified.h,prepare_amt_unified.cpp}`,
`...\automation_tool\assets\misc\host\main\hispmm_host.cpp`,
`...\HiSpMM-imbalanced\src\spmm.h`(변형 매크로 대비),
`...\README.md`.
