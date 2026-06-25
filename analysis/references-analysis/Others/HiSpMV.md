# HiSpMV (MAD-HiSpMV) 코드베이스 정밀 분석

> 분석 대상 경로: `REF/Others/HiSpMV`
> 작성 기준: 실제 소스 Read 기반, 라인 근거 표기. 추정/확인불가는 명시.

---

## 1. 개요

- **프로젝트명**: MAD-HiSpMV — *MAtrix Adaptive Design for Highly Imbalanced SpMV Accelerator (with GeMV Support) on HBM-based FPGAs* (`README.md:1-2`)
- **목적**: 고도로 불균형(highly imbalanced)한 희소 행렬에 대한 **희소 행렬-벡터 곱(SpMV)** 을 HBM 기반 FPGA에서 고처리량으로 가속. 옵션으로 **dense overlay**를 켜면 단일 커널이 SpMV와 **GeMV(dense MV)** 를 모두 지원 (`README.md:10-16`).
- **한줄요약**: TAPA dataflow + HBM 멀티채널 + (PE→Hybrid Row Distribution Network→y_Ax handler) 구조로, 행 단위 워크로드 불균형을 **행 분배 네트워크(Row Distribution Network)** 와 **Adder Chain/Pre-Accumulator** 로 흡수하고, **행렬 적응형(matrix-adaptive) 자동 설계 생성기**(automation_tool)로 채널 수·최적화 옵션을 행렬별로 튜닝해 코드를 생성한다.
- **수행 연산**: `c_out = alpha * (A · b) + beta * c_in` (SpMV + 스케일/바이어스, GEMV의 axpy 형태). `top_function.cpp:5`(alpha/beta), `base_functions.cpp:535`(`Compute_C`에서 `beta*tmp_in0 + alpha*tmp_in1`).
- **타깃 디바이스**: Xilinx Alveo **U280**, **U50** (Ultrascale+), 그리고 **V80**(Versal HBM, DSE 전용 — 코드 생성 미지원) (`fpgas.py:3-43`, `main.py:95-97`).
- **HLS 프레임워크**: **TAPA / PASTA + AutoBridge** (sb-dev 브랜치), Vitis HLS 2023.2+, XRT (`README.md:29-34`). `tapa::buffers`/`tapa::ibuffer`/`tapa::obuffers`(hybrid buffer) 사용 — PASTA 확장 기능으로 추정.
- **원논문**: "upcoming publication" — 출판 전(미확정). 이전 작업 HiSpMV를 확장 (`README.md:10, 270-271`). **확인 불가**(arXiv/DOI 미기재).
- **소속(추정)**: SFU-HiAccel (PASTA 저장소 소유자 동일, `README.md:31`).

---

## 2. 디렉토리 구조

### 자체 핵심 소스 트리 (분석 대상)
```
HiSpMV/
├── README.md                         # 문서 (MAD-HiSpMV)
├── get_tb_matrices.py                # 벤치 행렬 다운로더 (SuiteSparse 등)
├── automation_tool/                  # ★ 핵심: 행렬 적응형 코드 생성 + DSE
│   ├── assets/                       # HLS 커널 템플릿
│   │   ├── spmv.h                    # 커널 헤더 (TAPA 타입/구조체/top 선언)
│   │   ├── base_functions.cpp        # ★ 모든 HLS task 구현 (PE/네트워크/버퍼)
│   │   └── top_function.cpp          # ★ SpMV top dataflow (tapa::task invoke)
│   └── src/
│       ├── main.py                   # 자동(행렬적응) 진입점: DSE→codegen
│       ├── spmvcodegen.py            # ★ 코드 생성기 (수동 파라미터)
│       ├── dse.py                    # ★ Design Space Exploration
│       ├── cyclecount_est.py         # 사이클 수 추정 (타일/패딩/run-length)
│       ├── preprocessor.py           # ★ 워크로드 밸런싱(행 분배/OoO 스케줄) 모델
│       ├── resource_est.py           # 리소스(BRAM/URAM/DSP/LUT/REG) 추정
│       ├── crossbar.py               # ★ Row Distribution Network 토폴로지 생성
│       ├── fpgas.py                  # U280/U50/V80 디바이스 스펙
│       └── commons.py                # dataclass(FPGA/Resource/SpMVConfig) + 로깅
├── common/                           # 공통 host + 커널 빌드 인프라
│   ├── common.mk                     # TAPA/Vitis 빌드 Makefile
│   ├── include/{spmv-helper.h, fpga-power.h}
│   └── src/{spmv-host.cpp, spmv-helper.cpp, fpga-power.cpp}  # ★ host: 전처리/CPU참조/XRT실행
├── pyhispmv/                         # pybind11 래퍼 (Python에서 XRT 커널 호출)
│   ├── setup.py, include/fpga_handle.h, src/{pyhispmv_bindings.cpp, fpga_handle.cpp}
├── apps/                             # Python 앱 (general_test, model_test, DNN 모델)
├── cpu/                              # CPU 벤치 (Intel MKL) — 자체 host wrapper만 분석
├── gpu/                              # GPU 벤치 (cuSPARSE) — 자체 host wrapper만 분석
└── builds/                           # 생성물 예시 (config별 src + xclbin/리포트)
```

### 제외 목록 (vendor/생성물/데이터 — 이름만 언급)
- `builds/*/` 의 생성된 `spmv.cpp`/`hw_defs.h`/`link_config.ini`: **automation_tool가 자동 생성한 산출물**(코드 생성기 출력). 원본 템플릿(`assets/*`)을 분석하고 생성물은 대표 1개만 교차 확인.
- 비트스트림 `.xclbin`/`.xo`, floorplan, usage report: 빌드 산출물 → 제외.
- `matrices/` 의 `.mtx` 대용량 행렬 데이터 → 제외.
- `cpu/src/mmio.h`: NIST MatrixMarket I/O — 외부 표준 코드(vendor) → 이름만.
- `gpu/include/nvmlPower.h`, cuSPARSE 호출부: NVIDIA 라이브러리 의존 → 자체 wrapper 외 제외.
- `miniconda3/`, `.Xil`, `.ip_user_files` 류(존재 시) → 제외.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 커널 헤더 `automation_tool/assets/spmv.h`

데이터타입·상수·구조체·top 선언의 단일 출처.

- **컴파일 타임 매크로 분기** (`spmv.h:10-38`):
  - `FIFO_DEPTH`: row-dist-net 사용 시 2, 아니면 8 (`:10-14`).
  - `FP_ACC_LATENCY`: high-freq=10, pre-accum=4, 기본=5 (`:16-24`). FP 누산기의 파이프라인 지연이 II 의존거리 `II_DIST = FP_ACC_LATENCY + 1`(`:34`)을 결정 → AccumBuffer의 순환버퍼 깊이/RAW 처리 핵심.
  - `PES_PER_CH = CH_WIDTH/64` (`:26`): 512b 채널이면 PE 8개/채널. 64b = float val(32b) + (row/col 인덱스 32b) 패킹 1 nnz 엔트리.
  - `NUM_PES = NUM_A_CH * PES_PER_CH` (`:27`): A 채널 수 × 채널당 PE.
  - `MAX_ROWS_PER_PE = URAMS_PER_PE * 4096` (`:35`): URAM 1뱅크=4096 word(32b) 기준 PE당 출력 누산 버퍼 행수.
- **TAPA 하이브리드 버퍼 타입** (`spmv.h:40-58`): `tapa::buffers/ibuffer/obuffers<float[B_WINDOW], ...>` — 입력벡터 b를 PE 그룹 로컬 BRAM에 분할(`tapa::cyclic<B_PART>`, `memcore<bram>`)로 보관. PASTA 확장 타입으로 추정.
- **채널(vec_t) 타입** (`:60-62`): `channelA_t = vec_t<uint64_t, PES_PER_CH>`(A: 64b 패킹 nnz × PE수), `channelB_t/channelC_t = vec_t<float, FP32_PER_CH>`.
- **패킷 구조체**:
  - `flags_pkt{sharedRow, tileEnd, last}` (`:65-69`): PE→누산기 제어 플래그.
  - `Cnoc_pkt{dummy,last,tileEnd,sharedRow,uint16 row16,uint8 bank,float val}` (`:71-79`): **Row Distribution Network(NoC)** 를 흐르는 결과 패킷. `bank`=목적 PE/누산뱅크, `row16`=행 인덱스.
- **top 선언** `SpMV(...)` (`:82-91`): mmaps A/b/c_in/c_out + alpha,beta + 타일/길이/반복(rp_time) + DENSE_MODE.

### 3.2 HLS task 구현 `automation_tool/assets/base_functions.cpp` (가장 중요)

TAPA dataflow를 구성하는 모든 task. 단계별 정밀 분석:

**(1) `MM2S_A` (`:3-31`)** — HBM→스트림 (A 로딩)
- `tapa::async_mmap<channelA_t>` 에서 `read_addr.try_write/read_data.try_read` 로 비동기 버스트. 응답 1개(`channelA_t`, PES_PER_CH개 uint64)를 PES_PER_CH개 개별 스트림에 언팩(`:23-25`). II=1 파이프라인(`:9`). `rp_time` 만큼 반복(벤치마킹용 반복 실행).
- **핵심**: addr-issue / data-consume를 분리해 메모리 latency 은닉(전형적 TAPA async_mmap 패턴).

**(2) `MM2S_B` (`:33-54`)** — 입력벡터 b 로딩, `num_tiles_r * rp_time` 회 반복(타일마다 b 재로딩).

**(3) `MM2S_C`/`S2MM_C` (`:56-103`)** — 바이어스 c_in 읽기 / 결과 c_out 쓰기. `len = num_rows_per_pe * NUM_PES / FP32_PER_CH / NUM_C_CH` (`:60, 84`)로 출력 채널 분할. `S2MM_C`는 write_addr/write_data/write_resp 핸드셰이크로 ack 카운트(`:90-99`).

**(4) `LoadB` (`:105-150`)** — b를 PE 그룹 로컬 BRAM(`obuffersB_t local_B`)에 적재
- `tapa::section`의 `acquire/release_section`로 더블버퍼링(producer/consumer 동기화)(`:117-119, 144-147`). `W_WINDOW=512` 워드 윈도우 단위로 채널 입력을 받아 `LOAD_GROUP_SIZE`개 버퍼 복제(`:130-137`). 각 PE 그룹이 자기 b 윈도우를 로컬 보유 → 반복 재사용.

**(5) `ComputeAB` (`:158-254`)** — ★ PE 핵심 (곱셈)
- PE 2개를 한 묶음으로 처리(`a_in[2]`). 64b nnz 엔트리 디코딩:
  - 일반(sparse) 경로(`:228-242`): `val_bits = a & 0xFFFFFFFF`(하위32b=float 값), `col_id = (a>>32)&0x3FFF`(14b 열), `row = (a>>48)&0xFFFF`, `tileEnd=(a>>47)&1`, `sharedRow=(a>>46)&1`. 곱: `val_out = val * buf_ref[col_id]` (로컬 b에서 col_id 조회) (`:234`).
  - **Dense overlay 경로**(`BUILD_DENSE_OVERLAY`, `:174-226`): `DENSE_MODE` 런타임 플래그로 분기. dense면 행/열 카운터(`row_cnt,col_cnt`)로 인덱스 생성, 한 엔트리에 2개 a값(a_val0,a_val1)을 패킹해 b 2개와 곱-덧셈(`val_out = a_val0*b[0] + a_val1*b[1]`)(`:204-208`) → **dense일 때 PE당 2 MAC** 로 처리량 2배. row에 `(1<<15)` rowEnd 플래그 설정(`:198`).
- **시사점**: 단일 비트필드 패킹(val|col|row|tileEnd|sharedRow)으로 64b에 nnz 메타까지 담아 HBM 대역폭 절약. dense/sparse를 같은 PE에서 런타임 분기 → "dense overlay" = GeMV 지원의 본질.

**(6) `PreAccumulator` (`:257-353`)** — ★ RAW 의존성 사전 제거 (옵션)
- `BUILD_PRE_ACCUMULATOR` 시: PE별 `val_buff_part[2][II_DIST]`, `row_buff_part[2][II_DIST]` LUTRAM 시프트 레지스터(`:260-266`). 동일 행(`curr_row`)이 II_DIST 거리 내 재등장하면 그 값들을 미리 합산(`temp[0] += temp[l]`, `:319-326`)하여, 후단 AccumBuffer의 부동소수 누산 RAW 의존을 완화. shared row 처리: `shared_bank = row_in[0] & (NUM_PES-1)`, `shared_row16 = row_in[1]`(`:295-296`) — 두 PE의 출력을 한 행으로 묶어 다른 뱅크로 보낼 준비.
- 출력 `Cnoc_pkt{dummy=!rowEnd, bank, row16, val}` 생성(`:329-345`).
- **핵심 알고리즘**: 부동소수 가산의 긴 latency(=II_DIST) 때문에 같은 누산 주소에 연속 쓰기가 stall을 유발 → 시프트버퍼 안에서 같은 행을 미리 합쳐 의존거리를 깬다.

**(7) Row Distribution Network: `ADD<m,sw>`, `SWB<m,n>`, `SSW` (`:355-437`, `BUILD_ROW_DIST_NETWORK`)** — ★ 하이브리드 행 분배 NoC
- `ADD` (`:356-392`): 두 입력 패킷 값을 조건부 합산. `shared_cond`(둘 다 sharedRow & 비-dummy)면 `sum`을 한쪽(`i`)에 몰고 반대쪽은 0+dummy(`:366-374`). `sw`면 bank의 MSB로 라우팅 방향 결정, 아니면 템플릿 상수 `m`(`:370`). → 같은 행이 다른 PE에 분산된 경우(shared row) 합쳐서 한 누산기로 보냄.
- `SWB<m,n>` (`:394-415`): bank의 n번째 비트(`(bank>>n)&1`)로 스위치/라우팅 — butterfly/Benes 류 다단 스위칭. shared 여부로 방향 보정(`:404-407`).
- `SSW` (`:417-436`): sharedRow 비트로 두 입력 swap.
- 래퍼 함수 `ADD_0/ADD_1/ADD_SWB`, `SWB0_0..SWB1_6` (`:543-626`): 템플릿 인스턴스. **이들이 `crossbar.py`가 생성하는 그래프의 노드 명과 1:1 매칭**된다.

**(8) `AccumBuffer` (`:439-504`)** — ★ 출력 누산 (URAM)
- PE당 `BUFF_C[MAX_ROWS_PER_PE]` URAM(`:443-444`)에 행별 누산. **순환버퍼**(`circbuff_val[8]/circbuff_row[8]` LUTRAM, `:446-450`)로 in-flight 누산 중복 처리.
- 핵심 누산 루프(`:476-488`): `#pragma HLS DEPENDENCE true inter variable=circbuff_val distance=II_DIST`(`:478`) — RAW 의존거리를 II_DIST로 명시해 II=1 유지. `circbuff_val[w_idx] = curr_in.val + (circbuff_row[r_idx]==curr_in.row16 ? circbuff_val[r_idx] : BUFF_C[curr_in.row16])` (`:483`): 같은 행이 순환버퍼에 있으면 그 값을, 없으면 URAM 값을 더함 → forwarding. `bind_op fadd latency=FP_ACC_LATENCY impl=fabric`(`:482`).
- 타일별 init→acc→out 3단계, `num_tiles_c` 타일 누적 후 출력(`:466-501`).

**(9) `Arbiter_C` (`:506-519`)** — NUM_PES개 누산 출력을 NUM_C_CH개 HBM 채널 폭으로 재정렬/패킹.

**(10) `Compute_C` (`:521-540`)** — ★ axpy: `tmp_out[jj] = beta*c_in[jj] + alpha*(A·b)[jj]` (`:535`). SpMV 결과에 스케일·바이어스 적용 후 c_out으로.

### 3.3 Top dataflow `automation_tool/assets/top_function.cpp`

`SpMV(...)` 본문(`:1-48`). 모든 task를 `tapa::task().invoke<...>()` 로 연결:
- 스트림 선언(`:13-30`): A_in, C_row/val/flag, B_in, C_shf(NoC), C_buf, C_arb, C_ab/in/out.
- invoke 그래프(`:32-47`):
  - `MM2S_A × NUM_A_CH` → `MM2S_B × NUM_B_CH` → `LoadB × (NUM_PES_HALF/LOAD_GROUP_SIZE)` → `ComputeAB × NUM_PES_HALF`(PE 2개씩) → `DummyReadB`(detach) → `PreAccumulator × NUM_PES_HALF` → (RDN 시 `FIFO_C_SHF` 경유) → `AccumBuffer × NUM_PES` → `Arbiter_C` → `MM2S_C × NUM_C_CH` → `Compute_C × NUM_C_CH` → `S2MM_C × NUM_C_CH`.
- **RDN 분기**(`:38-42`): row_dist_net 정의 시 PreAccumulator는 `FIFO_C_SHF`로 출력 → spmvcodegen이 삽입한 crossbar invoke들이 `FIFO_C_SHF`→`FIFO_C_BUF`로 라우팅. 미정의 시 PreAccumulator가 직접 `FIFO_C_BUF`로.

### 3.4 코드 생성기 `automation_tool/src/spmvcodegen.py` (가장 중요)

클래스 `SpMVCodeGen` (`:25-192`):
- 파생 파라미터 계산(`__init__`, `:26-38`): `pes_per_ch=ch_width//64`, `num_pes`, `b_part`, `b_window=min(b_part*1024, 1<<14)`, `log2_num_pes`.
- `generateAll` (`:40-56`): Versal이면 리소스만 출력 후 종료(코드 생성 미지원, `:41-44`). 아니면 Makefile/hw_defs.h/spmv.h복사/spmv.cpp/link_config.ini 생성.
- `createLinkConfig` (`:58-77`): `link_config.ini`에 `sp=SpMV.b_i:HBM[i]`, A는 `HBM[i+num_ch_B]`, c_in/c_out 순차 배치 → **HBM 채널-포트 매핑** 자동 생성.
- `createHwDefsHeader` (`:110-129`): `#define NUM_A_CH/NUM_B_CH/NUM_C_CH/CH_WIDTH/URAMS_PER_PE` + 옵션 매크로(`BUILD_DENSE_OVERLAY/PRE_ACCUMULATOR/ROW_DIST_NETWORK`, `HIGH_FREQ_DESIGN`) + `LOAD_GROUP_SIZE`, `LOG_2_NUM_PES` 출력.
- `createKernelCode` (`:139-179`): base_functions.cpp + top_function.cpp 를 이어붙이되, `tapa::task()` 직전에 `#ifdef BUILD_ROW_DIST_NETWORK`로 crossbar 스트림 선언 삽입, `#else` 위치에 crossbar invoke 삽입(`:167-179`). → **CrossBarGen이 만든 그래프를 TAPA invoke 텍스트로 코드 합성**.
- `generateCBstreams/Invokes` (`:181-192`): depth_dict로 `tapa::stream<Cnoc_pkt,depth> s_i_j`, graph_dict로 `.invoke(name, in0, in1, out0, out1)` 라인 생성.
- `__main__` (`:195-240`): CLI 인자(`--num-ch-A/x/y, --ch-width, --urams-per-pe, --dense-overlay, --pre-accumulator, --row-dist-net, --high-freq, --device`). 제약 검증 `num_ch_A % (2*num_ch_C) == 0`(`:234`), `ch_width ∈ {256,512}`(`:235`). 디렉토리명은 `encodeSpMVConfig`로 인코딩(예 `PA-HI-SpMV-16-2-4`).

### 3.5 Row Distribution Network 생성기 `automation_tool/src/crossbar.py`

클래스 `CrossBarGen` (`:3-144`):
- `buildGraph` (`:74-143`): **2-페이즈 다단 네트워크**. 1페이즈(`d=1..depth`, `:75-105`): ADD 블록(누산) + SSW(shared swap), 마지막 단은 `ADD_SWB`. 2페이즈(`d=depth..1`, `:107-142`): SWB 블록(라우팅 복원) + SSW. butterfly 형태의 add+route 네트워크를 `NUM_PES` 크기로 생성. `(i>>d)&1`로 ADD_0/ADD_1, SWB0_*/SWB1_* 선택(`:101-130`).
- `computeDepth` (`:22-31`): 스트림 FIFO 깊이를 레벨 차×2 + 홀수 레벨 보정(high_freq=10, 기본=6)으로 산정 → 파이프라인 균형/floorplan용 깊이.
- **핵심**: 행이 어느 PE에 있든 올바른 누산뱅크로 모이도록 하는 **로그단 스위칭+가산 네트워크**를 PE 수에 맞춰 파라메트릭 생성.

### 3.6 DSE `automation_tool/src/dse.py`

클래스 `DSE` (`:11-218`):
- `getBestConfig` (`:22-95`): 입력 행렬(.mtx)을 읽어 **채널 배분 최적화**.
  - 모델(`:24-37`): 총 시간 ≈ `nnz/A채널/8 + cols/B채널/16 + rows/C채널/16`. 균등화 조건에서 `B채널 ∝ A채널·cols/nnz`, `C채널 ∝ A채널·rows/nnz`. `opt_ch_A = HBM채널 / (1 + norm_cols + 2·norm_rows)`(`:35`).
  - 탐색(`:48-89`): C/B 채널 2의 거듭제곱, row_dist_net∈{F,T}, pre_accumulator∈{F,T}(Versal은 T만)로 격자 탐색. A = HBM채널 − B − 2C 를 `2C` 배수로 맞춤(`:63`). 각 config에 대해 `ResourceEstimator`로 리소스 한도 검사(`:79`) 통과 시 `CycleCountEstimator`로 사이클 추정, 최소 사이클 config 선택.
- `getSingleBestConfig` (`:97-142`): dense overlay 전용 — 행렬 없이 "조밀 가정"으로 B=C=1, A 최대화. 이후 C 채널 증설 시도(`:127-139`).
- `mm_read` (`:144-218`): MatrixMarket 파서. symmetric이면 (c,r) 대칭 항 추가, pattern은 값 1.0, float32 0 스킵 → COO 반환.

### 3.7 사이클/리소스/전처리 모델

- `cyclecount_est.py` `getCC` (`:9-57`): NUM_PES/DEPTH/B_PART/WINDOW 계산, 행/열 패딩·타일 수 산정(`:20-24`), `np.add.at`으로 (타일r,타일c,PE,PE행)별 nnz 카운트(`:36-49`). `CC_TOTAL = run_length(스트림A) + num_tiles_r*load_b + update_c`(`:51-55`). run_length는 `preprocessor`가 산출.
- `preprocessor.py` `PreProcessor` (`:10-128`): ★ **워크로드 밸런싱 사이클 모델**.
  - `get_tile_size` (`:30-55`): row_dist_net 없을 때 — pre_accum이면 `tile_max`(최대 PE 부하), 아니면 OoO 스케줄 사이즈. 있을 때 — `get_intra_mode_rows`로 dense 행을 intra-row 모드(여러 PE에 분산)로 빼서 균형화.
  - `get_out_of_order_size` (`@njit`, `:58-83`): PE별 II_DIST개 lane에 행을 그리디 배치(`argmin`), 최대 lane 부하 × II_DIST = 사이클.
  - `get_intra_mode_rows` (`:86-124`): 가장 무거운 행들을 모든 PE에 분산(shared/intra mode)했을 때의 균형 부하 계산, 개선<10%면 미적용(`:122-123`). → **불균형 행을 어떻게 PE에 흩뿌릴지** 결정하는 핵심 휴리스틱.
- `resource_est.py` `ResourceEstimator` (`:4-205`): task별 경험적 리소스(LUT/REG/DSP/URAM/BRAM)를 PE 수/채널 수로 곱해 합산(`getDesignResource`, `:5-54`). Ultrascale+ vs Versal 분기(DSP58은 fp mul/add 1 DSP). 스트림 LUT/FF는 폭·깊이 공식(`Streams`, `:192-197`).
- `fpgas.py` (`:1-43`): U280/U50(28채널 가정, 512b@225MHz), V80(64채널 256b@400MHz). `limit`은 utilization 상한(LUT 0.62/0.70 등), `fixed`는 셸 점유 리소스.
- `commons.py` (`:1-79`): dataclass + `encodeSpMVConfig`(prefix `Dense/PA/HI` + `A-B-C`).
- `main.py` (`:43-136`): 행렬 없으면 `getSingleBestConfig`(dense overlay), 있으면 행렬마다 `getBestConfig`→codegen, 중복 config 스킵, 매핑 CSV 저장(`:52-69`).

### 3.8 Host `common/src/spmv-host.cpp` + `common/include/spmv-helper.h`

- `HiSpmvHandle` 클래스(`spmv-helper.h:62-202`): mtx 로드, **타일·패딩**(`tileAndPad`), **워크로드 밸런싱**(`balanceWorkload`/`oldBalanceWorkload`, `:133-136`), **FPGA 포맷 패킹**(`prepareTile`→`encode`).
  - `encode(tileEnd,rowEnd,sharedRow,row,col,val)` (`spmv-helper.h:45-60`): 64b 비트필드 패킹 — `[rowEnd][row:15][tileEnd:47bit][sharedRow:46bit][col:14][val:32]`. **커널 `ComputeAB`의 디코딩과 정확히 대칭**.
- `main` (`spmv-host.cpp:41-191`): alpha=0.55, beta=−2.05(`:43-44`). 인자 1개=sparse mtx, 2개=dense(rows,cols, dense overlay 시). CPU 참조(`cpuSequential`)와 비교, GFLOPS=`2*(nnz+rows)/time`(`:100,185`). 실행: xclbin+device면 XRT(`fpgaRun`), 아니면 `tapa::invoke`(csim/cosim)(`:131-178`). `rp_time`으로 벤치 반복 횟수 조절(`:120-125`).

### 3.9 pyhispmv (Python 바인딩)
- `pyhispmv/src/pyhispmv_bindings.cpp` + `fpga_handle.{h,cpp}`: pybind11로 `HiSpmvHandle`(또는 별도 FpgaHandle)을 노출해 `apps/`의 Python에서 XRT 커널 직접 호출. setup.py로 `build_ext --inplace`(`README.md:90-94`). 상세 구현은 미정독(분량) — 역할만 확인.

---

## 4. 데이터플로우 / 실행 흐름

```
HBM ─(MM2S_A)→ [a_in×NUM_PES] ─┐
HBM ─(MM2S_B)→ [b_in] →(LoadB)→ 로컬 BRAM(b 윈도우) ─┘
                                       │
                              (ComputeAB, PE 2개/묶음)  ← col_id로 로컬 b 조회, val*b
                                       │ (row,val,flags)
                              (PreAccumulator)  ← 동일행 사전합산(RAW 완화) [옵션]
                                       │ Cnoc_pkt
                       (Row Distribution Network: ADD/SWB/SSW 다단)  [옵션, crossbar.py 생성]
                                       │
                              (AccumBuffer×NUM_PES)  ← URAM 행별 누산 + 순환버퍼 forwarding
                                       │
                              (Arbiter_C) → NUM_C_CH 폭 재정렬
                                       │
HBM ─(MM2S_C: c_in)──────────→ (Compute_C: beta*c_in + alpha*Ab) ─(S2MM_C)→ HBM(c_out)
```

- **메모리 계층**: HBM(다채널) → async_mmap 버스트 → TAPA FIFO 스트림 → PE 그룹 로컬 BRAM(b 재사용) → URAM(출력 누산). A는 64b 패킹으로 대역폭 절감, b는 타일별 재로딩하되 로컬 재사용.
- **병렬화**: A 채널 수 × (채널폭/64) = NUM_PES 만큼 PE 병렬. PE는 2개씩 묶여 shared-row 처리에 협력. NoC는 로그단 병렬 스위칭.
- **파이프라인**: 거의 모든 task `#pragma HLS PIPELINE II=1`. 누산 RAW는 `II_DIST`(= FP_ACC_LATENCY+1) 의존거리 + 순환버퍼 forwarding + PreAccumulator로 II=1 유지.
- **데이터타입/양자화**: **FP32**(single precision)만 사용. 양자화/저비트는 없음. nnz 엔트리 64b = float32 값 + 14b 열 + 16b 행 + 제어비트. dense overlay 시 PE당 2 MAC(2개 float a값 패킹).
- **타일링**: 행 방향 `DEPTH = NUM_PES*URAMS_PER_PE*4096`, 열 방향 `WINDOW`로 2D 타일. 패딩으로 정렬(`cyclecount_est.py:20-24`).

---

## 5. HW/SW 매핑

| 계층 | 구성요소 | 역할 |
|---|---|---|
| SW(Python, 빌드타임) | `main.py`/`dse.py`/`preprocessor.py`/`resource_est.py`/`cyclecount_est.py`/`crossbar.py`/`spmvcodegen.py` | 행렬 적응형 DSE → 채널·옵션 결정 → HLS 코드(spmv.cpp/hw_defs.h/link_config.ini) 자동 생성 |
| HW 템플릿(HLS C++) | `assets/base_functions.cpp` + `top_function.cpp` + `spmv.h` | TAPA task 구현 (PE/누산/NoC/IO) — 매크로로 옵션 분기 |
| HW(빌드 산출) | `builds/<config>/` xclbin | U280/U50 비트스트림 |
| SW(런타임, host C++) | `common/src/spmv-host.cpp`, `spmv-helper.cpp`, `fpga_handle.cpp` | mtx 로드·타일·밸런싱·64b 패킹(encode), CPU 참조, XRT 실행/검증 |
| SW(런타임, Python) | `pyhispmv` + `apps/*.py` | pybind11로 커널 호출, DNN 레이어/일반 테스트 |
| 벤치(외부) | `cpu`(MKL), `gpu`(cuSPARSE) | 비교 베이스라인 |

핵심: **코드 생성은 빌드타임 SW, 데이터 전처리(밸런싱·패킹)는 런타임 SW(host), 연산은 HW**. 밸런싱 알고리즘이 host와 preprocessor(추정 모델) 양쪽에 중복 구현(host=실제, py=DSE 추정).

---

## 6. 빌드·실행

- 환경: Vitis HLS 2023.2+, XRT, PASTA+AutoBridge(sb-dev), conda (`README.md:29-58`).
- 코드 생성(자동): `python main.py <build_dir> --device U280 --matrices ../matrices/` (`README.md:173-176`).
- 코드 생성(수동): `python spmvcodegen.py <out> --device U280 --num-ch-A 4 ... --row-dist-net --dense-overlay` (`README.md:207-211`).
- 빌드: `make host` → `make tapa`(RTL/xo) → `make hw`(xclbin) (`common/common.mk`, `README.md:229-258`).
- 실행: `./spmv-host A.mtx`(csim) / `--bitstream=...xclbin --device=N`(HW) (`README.md:235-264`).
- Python: `pyhispmv` build_ext → `apps/general_test.py`, `model_test.py`(DNN, density 인자) (`README.md:90-115`).

---

## 7. 의존성

- HW: TAPA/PASTA+AutoBridge, Vitis HLS, `ap_int.h`, `tapa.h`. host: XRT(`xrt/*.h`), gflags, tapa runtime.
- SW(py): numpy, scipy(`coo_matrix`), numba(`@njit` for OoO 스케줄), argparse/logging.
- 외부 데이터: SuiteSparse `.mtx`(`get_tb_matrices.py`). 벤치: Intel MKL, NVIDIA cuSPARSE/NVML.

---

## 8. 강점 / 한계 / 리스크

**강점**
- 행렬 적응형 자동 설계: DSE가 행렬 통계(nnz/rows/cols)로 HBM 채널 배분·옵션을 정해 코드까지 생성 → 행렬별 최적 가속기.
- 불균형 흡수 3종 세트: Pre-Accumulator(RAW 완화) + Row Distribution Network(행 재분배) + intra-row 분산 스케줄.
- 64b 패킹으로 HBM 대역폭 절감, async_mmap으로 latency 은닉, II=1 유지 설계.
- dense overlay로 단일 커널이 SpMV+GeMV 지원 → 혼합 워크로드 대응.
- 멀티 디바이스(U280/U50/V80) 리소스 모델 분리.

**한계 / 리스크**
- **FP32 전용** — 저비트/정수 양자화 미지원(Transformer/ViT INT8 가속에는 직접 부적합). (확인: 코드 전반 float만 사용.)
- PASTA+AutoBridge sb-dev가 **비공개**(출판 전) → 재현성 제약(`README.md:34`).
- V80은 DSE만, 코드 생성 미지원(`main.py:96`, `spmvcodegen.py:41`).
- 리소스/사이클 모델이 **경험적 상수**(resource_est의 하드코딩 LUT/DSP 값) → 디바이스/툴 버전 변하면 부정확 가능(추정).
- 밸런싱 로직이 host(C++)와 preprocessor(py)에 이원화 → 불일치 리스크(py는 추정용).
- 원논문 미공개 → 성능 수치/정확도 주장 **확인 불가**.

---

## 9. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적, 추정)

- **불균형 워크로드 라우팅 NoC** (`crossbar.py` + `ADD/SWB/SSW`): 희소 attention(예: ViTCoD/sparse ViT)이나 가지치기된 GEMM에서 PE 부하 불균형을 흡수하는 **로그단 가산+스위칭 네트워크** 설계 패턴을 그대로 차용 가능. HG-PIPE의 systolic/pipeline에 sparse 경로를 붙일 때 참고.
- **출력 누산 RAW 회피 기법** (`AccumBuffer`의 순환버퍼 forwarding + `II_DIST` DEPENDENCE pragma + `PreAccumulator`): FP 누산 latency로 인한 stall을 II=1로 유지하는 정석. Transformer의 LayerNorm/softmax 누산, attention score 누적 등에 재사용 포인트.
- **행렬 적응형 DSE→코드젠 파이프라인** (`dse.py`+`spmvcodegen.py`+`crossbar.py`): "입력 통계 → 채널/옵션 결정 → HLS 텍스트 합성" 자동화 틀은 ViT 레이어별(토큰수/임베딩차원/sparsity) 가속기 자동 생성에 이식 가능. 특히 **HBM 채널 배분 최적화 공식**(`dse.py:24-37`)은 멀티-HBM ViT GEMM 배치에 응용.
- **dense overlay = 단일 커널 SpMV/GeMV** (`ComputeAB` DENSE_MODE): dense GEMM과 sparse를 한 PE에서 런타임 분기하는 구조는, ViT의 dense MLP와 sparse attention을 하나의 가속기로 통합하는 데 직접 적용 가능(자원 재사용).
- **64b 비트필드 패킹**(`encode`/디코딩 대칭): 인덱스+값+제어를 한 워드에 담는 패킹은 sparse Mamba/SSM 상태나 sparse token 스트림 전송에 응용.
- **한계 인지**: FP32 전용이라 우리의 INT8/저비트 ViT/Mamba 양자화 경로에는 데이터패스 재설계 필요. 양자화는 별도(ViT-Quantization 계열)에서 가져오고, **여기서는 dataflow/밸런싱/코드젠 골격만** 차용하는 것이 현실적(추정).

---

## 10. 근거 표기

- **확인(라인 근거)**: 본문의 모든 코드 동작은 `assets/spmv.h`, `assets/base_functions.cpp`, `assets/top_function.cpp`, `src/spmvcodegen.py`, `src/dse.py`, `src/preprocessor.py`, `src/resource_est.py`, `src/crossbar.py`, `src/cyclecount_est.py`, `src/fpgas.py`, `src/commons.py`, `src/main.py`, `common/src/spmv-host.cpp`, `common/include/spmv-helper.h` 의 명시 라인 기반.
- **추정**: TAPA `tapa::buffers/section` = PASTA 하이브리드버퍼 확장이라는 점, SFU-HiAccel 소속, 경험적 리소스 모델의 정확도, HG-PIPE 이식 시사점.
- **확인 불가**: 원논문(미출판)·성능/정확도 수치, PASTA sb-dev 내부(비공개), `pyhispmv`/`fpga_handle.cpp` 세부 구현(역할만 확인), `builds/*` 생성물의 실제 합성 결과 수치.
