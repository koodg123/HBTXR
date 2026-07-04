# acap-gemm-sa 정밀 분석

> 작성 기준: 실제 소스(`src/` 기본 변형)를 Read로 직접 읽고 라인 근거 기반으로 작성. 모든 라인 번호는 `src/` 변형의 파일 내용 기준.

---

## 1. 개요

- **목적**: AMD/Xilinx Versal ACAP의 AI Engine(AIE) 시스톨릭 어레이를 사용해 대형 밀집 GEMM(`C = A·B`)을 가속하는 연구용 구현체.
- **한줄요약**: `R×C`(기본 8×50) AIE 코어를 2D 시스톨릭 어레이로 배치하고, PL(Programmable Logic)의 단일 HLS `dma` 커널이 DRAM↔AIE 사이의 타일 스트리밍/멀티캐스트/누산 합산을 담당하는 **AIE + PL 협조형 GEMM 가속기**.
- **원논문**: 저장소 자체에는 논문 링크 명시 없음("확인 불가"). 디렉토리/스크립트 구성(`model.mod` AMPL/Gurobi 솔버, `heatmap.py`, `parse_profile.py`, `monitor_power.py`)으로 보아 **Versal AIE GEMM 매핑/설계공간탐색(DSE) 연구 코드**로 추정.
- **타깃 디바이스** (라인 근거):
  - `cmake/xilinx-setup.cmake:7` → 기본 플랫폼 `xilinx_vck5000_gen4x8_qdma_2_202220_1` (Versal **VCK5000** 데이터센터 카드).
  - `scripts/device.py:54-77` → `VC1902`(VCK5000의 디바이스), `AIE_ROWS=8, AIE_COLS=50`, `PL_BRAM=BRAM_18K(1934)`, `PL_URAM=URAM_288K(463)`. `VCK5000 = VC1902` 별칭.
  - `src/parameters.hh:14-19` → `DEF_AIE_ROWS 8`, `DEF_AIE_COLS 50` (8×50 = 400 AIE 코어 사용, VC1902는 400 AIE 타일).
  - Vitis 2023.1 / C++17 요구(`README.md:5-12`).

---

## 2. 디렉토리 구조

### 자체 핵심 소스 트리
```
acap-gemm-sa/
├── README.md                  # 빌드/실행/시뮬/솔버/스크립트 가이드
├── CMakeLists.txt             # 최상위 cmake
├── cmake/
│   ├── xilinx-setup.cmake     # vitis_flow() 함수: AIE/HLS/link/package/host 빌드 플로우 정의
│   ├── hls.cfg.in             # HLS 설정 템플릿
│   ├── pre_sim.tcl.in
│   ├── xrt.ini
├── src/                       # ★ 메인 변형 (본 분석 대상)
│   ├── parameters.hh          # 설계 파라미터(타일/어레이/데이터폭) + static_assert 검증
│   ├── gemm.hh / gemm.cc      # AIE 커널 (Gemm 클래스, compute MAC 루프)
│   ├── graph.hh / graph.cc    # ADF 그래프(시스톨릭 PE 배치/연결)
│   ├── dma.cc                 # PL HLS 커널(DRAM↔AIE 스트리밍/멀티캐스트/누산)
│   ├── host.cc                # XRT 호스트(데이터 변환/실행/검증)
│   ├── generate_gemm_data.py  # AIE 시뮬용 데이터 생성기
│   ├── xsa.cfg                # PL↔AIE 연결(connectivity) 설정
│   ├── place_design_pre.tcl
│   └── CMakeLists.txt
├── src-prof/  src-ideal/  src-trad/  src-bw/   # 동일 구조의 실험 변형들(프로파일/이상/전통/대역폭)
└── scripts/                   # 솔버·플롯·프로파일 파싱·전력 모니터·디바이스 모델
    ├── model.py / model.mod   # AMPL+Gurobi 설계공간 솔버
    ├── device.py              # VC1902 디바이스 자원/성능 모델
    ├── parse_profile.py, heatmap.py, plot_bar.py, plot_misc.py,
    ├── plot_multicast.py, pie.py, monitor_power.py, check_simulation.py
```

### 변형(variant) 디렉토리 관계 ("추정")
- `src/`: 기준(현행) 구현.
- `src-prof/`: 프로파일링용(추가 `common.hh` 보유, `xsa.cfg`에 profile 관련 설정 추정).
- `src-ideal/`: AIE 커널만 두고 PL/host 없이 이상적 상한 측정용(파일이 `gemm.cc/gemm.hh`만 존재).
- `src-trad/`: 전통적(traditional) 멀티캐스트 비대조군.
- `src-bw/`: 대역폭(bandwidth) 실험.
- 변형별 차이는 본 분석 범위 밖(요청에 따라 `src/` 중심). 차이의 정밀 비교는 "확인 불가"로 표기.

### 제외 목록 (분석에서 이름만 언급)
- `.git/` (objects/pack, hooks 샘플 등) — 버전관리 메타.
- 빌드 산출물(`build/`, `*.xclbin`, `*.xsa`, `libadf.*`) — 본 트리에 미포함이나 빌드 시 생성.
- 대용량 시뮬 데이터(`data/`) — 런타임 생성.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `src/parameters.hh` — 설계 파라미터 & 타일 계층 정의
컴파일 타임 상수와 `static_assert`로 전체 타일링 계층을 강제하는 단일 진실 원천(single source of truth).

- **데이터타입/폭** (`parameters.hh:9-12`):
  - `using DT = float;` → 기본 연산 자료형 **FP32**. `DATA_WIDTH = sizeof(DT)*8 = 32`.
  - `DRAM_WIDTH = 512`(NoC/AXI 비트폭), `PLIO_WIDTH = 128`(PL↔AIE PLIO 스트림 폭).
- **어레이/파티션** (`parameters.hh:14-19`): `AIE_ROWS=8`, `AIE_COLS=50`, `PARTS=2`(DRAM 포트/파티션 분할 수).
- **문제 크기** (`parameters.hh:20-22`): `M=K=N=8192`(8K 정방 GEMM).
- **3단 타일 계층** — 변수 명명 `B{M,K,N}{1,2,3}`:
  - 레벨1(PL 타일): `BM1=ceil(M/PL_M)`, `BK1=ceil(K/PL_K)`, `BN1=ceil(N/PL_N)` (`:37-39`). `PL_M=1024, PL_K=512, PL_N=800` (`:29-31`).
  - 레벨2(AIE 타일 over PL 타일): `BM2=PL_M/AIE_M/AIE_ROWS`, `BK2=PL_K/AIE_K`, `BN2=PL_N/AIE_N/AIE_COLS` (`:40-42`).
  - 레벨3(서브타일/mmul): `BM3=AIE_M/AIE_MM` 등 (`:43-45`). `AIE_M=16, AIE_K=64, AIE_N=16`(개별 AIE 코어가 처리하는 타일), `AIE_MM=AIE_KK=AIE_NN=1`.
- **데이터 패킹** (`:48-52`): `DRAM_PACK=512/32=16`, `PLIO_PACK=128/32=4`, `DRAM_PLIO_PACK=4`. `PACK_PER_ROW_STREAM/PACK_PER_COL_STREAM`은 행/열 스트림당 패킹 계수.
- **정합성 검증** (`:54-81`): 24개 `static_assert`로 PL 타일이 AIE×어레이 배수인지, 데이터폭이 정렬되는지, `AIE_ROWS/COLS % PARTS == 0` 등 합법성을 컴파일 타임에 강제. **이 파일이 잘못되면 빌드 자체가 막히는 안전장치**가 핵심 설계 의도.

### 3.2 `src/gemm.hh` + `src/gemm.cc` — AIE 시스톨릭 PE 커널 (가장 중요)
단일 AIE 코어 1개에 매핑되는 템플릿 클래스 `Gemm<DT,R,TM,TK,TN,MM,KK,NN,FWD>`.

- **클래스 시그니처** (`gemm.hh:14-15`): 템플릿 인자 `R`(어레이 행 수), `TM/TK/TN`(타일), `FWD`(상류 partial-sum 전달 여부).
- **인터페이스 선택** (`gemm.hh:19-38`):
  - `FWD==true` → `in3out1` 등록(`in0=A행`, `in1=B열`, `in2=상류 PE의 부분합`, `out0`).
  - `FWD==false` → `in2out1`(상류 부분합 없음 = 어레이 최하단 행).
  - 즉, **출력-고정(output-stationary) 부분합을 행 방향으로 누산하며 위로 전달하는 시스톨릭 구조**.
- **상태** (`gemm.hh:54-60`): `DT buf0[TM][TN]` = PE 내부 부분합 누산기(C 타일). `stop_iter`, `stop_lap`로 K 누산 반복/전체 랩 제어.
- **생성자** (`gemm.cc:47-55`): `stop_iter = BK2`(K 방향 AIE 타일 수), `stop_lap = BM1*BN1*BK1*BM2*BN2`(전체 처리해야 할 출력 타일 수).
- **`impl()` 실행 루프** (`gemm.cc:57-170`): PE의 메인 상태기계.
  - 각 `lap`마다 `buf0`를 0으로 벡터 초기화 (`:89-90`, `aie::zeros`).
  - 내부 `iter`(=`BK2`)마다 `rin.acquire()`(A행), `cin.acquire()`(B열) 비동기 버퍼 획득 → `compute()` 호출 → release (`:98-136`).
  - **`flush_step` 람다** (`:70-86`): 상류(위쪽 행)에서 내려온 부분합을 그대로 아래로 흘려보내는 패스스루. `flush = R - row - 1`(`:161`)로 자기 위쪽 PE 개수만큼 flush 단계를 수행 → **시스톨릭 파이프라인 배수(drain) 메커니즘**. 어레이가 깊을수록 더 많은 flush가 필요.
  - 랩 종료 시 `buf0`를 `oout`로 출력 (`:147-159`).
- **`compute()` — 핵심 MAC 커널** (`gemm.cc:172-248`): AIE 벡터 API(`aie::mac`)로 4-way 누산 언롤.
  - `M_STEP=2`, FP32일 때 `N_STEP=1`(`:183-184`) → 2(M)×1(N) 출력 타일 동시 처리, `aC00/aC10/aC01/aC11` 누산기 레지스터.
  - `ALEN=8`(FP32 AIE 누산 벡터 길이, `gemm.cc:44` / `aie_alen<32>=8` `:16`), `SIZE_A=ALEN/PACK_A`, `SIZE_B=ALEN`.
  - 3중 루프: `m(TM step 2)` × `n(TN step N_STEP·ALEN)` × `kp(K)` 내부에서 `aie::load_v`로 A/B 벡터 로드 후 `aie::mac(aCxx, vBx, vAx[kk])` 누산 (`:226-231`).
  - 결과를 `aie::store_v`로 `buf0`에 기록 (`:235-239`).
  - **설계 의도**: AIE의 SIMD MAC 처리량을 채우기 위해 M/N 방향 언롤 + K 내부 누산. FP32 기준 AIE는 사이클당 8 MAC(`device.py:66-70` `AIE_MAC_PER_CYCLE[32]=8`).
- **시뮬 보조** (`gemm.cc:6-39`): `aie_vlen`/`aie_alen` 트레이트로 데이터폭별 벡터 길이 매핑(8b=64/128, 16b=32/32, 32b=16/8). `vec2str` 디버그 출력.

### 3.3 `src/graph.hh` + `src/graph.cc` — ADF 그래프(시스톨릭 어레이 토폴로지)
`GemmGraph` 클래스가 `R×C` PE를 실제 AIE 타일에 배치하고 연결.

- **노드 배열** (`graph.hh:21-24`): `kernel[R][C]`, 입력 `in0[R]`(A행 PLIO), `in1[C]`(B열 PLIO), 출력 `out0[C]`(C열 PLIO).
- **PLIO 생성** (`graph.hh:36-46`): 각 행은 `in0_r`, 각 열은 `in1_c`/`out0_c` PLIO를 `data/*.txt`로 생성(시뮬용).
- **PE 배치/연결 루프** (`graph.hh:48-83`):
  - `r=R-1`(최하단)은 `FWD=false`(상류 부분합 없음), 그 외는 `FWD=true` (`:50-54`).
  - `adf::location<adf::kernel> = adf::tile(c, r)` → **물리 타일 좌표 (열=c, 행=r) 직접 배치** (`:59`).
  - 차원: `in[0]={TM*TK}`(A), `in[1]={TK*TN}`(B), `in[2]={TM*TN}`(상류 C), `out[0]={TM*TN}` (`:66-69`).
  - 연결: `in0[r]→kernel[r][c].in[0]`(A를 같은 행에 브로드캐스트), `in1[c]→kernel[r][c].in[1]`(B를 같은 열에 브로드캐스트) (`:71-72`).
  - **세로 부분합 체인**: `kernel[r+1][c].out[0] → kernel[r][c].in[2]` (`:75`) → 아래→위로 부분합 누산.
  - 최상단 `r==0`만 `out0[c]`로 출력 (`:78-80`).
  - **결론**: A는 행 멀티캐스트, B는 열 멀티캐스트, 부분합은 열 내부에서 세로 누산되는 **출력-고정 시스톨릭(weight/activation 멀티캐스트 + partial-sum systolic) 구조**.
- **그래프 인스턴스/시뮬 main** (`graph.cc:3-23`): `graph.init()/run(ADF_ITERS)/end()`. `ADF_ITERS=1`(`parameters.hh:46`).

### 3.4 `src/dma.cc` — PL HLS `dma` 커널 (DRAM↔AIE 데이터 무버, 1140줄)
파일 대부분(약 51~567행)은 **가변 스트림 개수(최대 50)에 대응하기 위한 코드 생성 매크로**(`ARGS_n`, `CALL_STREAMS_n`, `CALL_STREAMS_TASK_n`, `CALL_PARTS{1,2,3}`)이며, 실제 로직은 593행 이후.

핵심 함수(dataflow 단계별):
- **`load_A` / `load_B`** (`dma.cc:593-639`): DRAM(`dram_t* pl`)에서 PL 타일 단위로 `dram_stream_t`에 적재. `load_A`는 `(bm1*BK1+bk1)` 인덱싱(A는 N 재사용), `load_B`는 `(bn1*BK1+bk1)`(B는 M 재사용). `#pragma HLS pipeline II=1`.
- **`store_C`** (`dma.cc:641-661`): 최종 C 타일을 DRAM에 기록.
- **`split_stream`** (`dma.cc:663-692`): 512b DRAM 워드를 `DRAM_PLIO_PACK`(=4)개의 128b PLIO 워드로 쪼개 `ps[s][z]` 스트림에 라운드로빈 분배(`s=(j+p)%(S/P)`). DRAM 대역폭을 다수 AIE 행/열 스트림으로 디멀티플렉싱.
- **`merge_stream`** (`dma.cc:694-723`): 역방향(여러 PLIO 스트림 → 512b DRAM 워드 병합).
- **`load_buf`** (`dma.cc:725-742`): split 스트림 → 온칩 버퍼(`plio_t buf[...]`) 적재. `last`면 스킵(이중버퍼링 마지막 처리).
- **`send_buf_A`/`send_buf_B`** (`dma.cc:787-835`): 온칩 버퍼를 AIE 타일 순서(`bm2,bn2,bk2` 루프 + `AIE_M*AIE_K/PLIO_PACK`)로 AXIS 스트림에 전송 → AIE의 입력 PLIO로.
- **이중버퍼 파이프라인 `send_A`/`send_B`** (`dma.cc:856-934`): `buf[2][...]` 핑퐁. `send_inner_X`는 `#pragma HLS dataflow`로 "현재 버퍼 송신 + 다음 버퍼 적재"를 동시 수행(`send_buf_X(buf_0)` ∥ `load_buf(buf_1)`). `bm1,bn1,bk1` 3중 루프로 전체 PL 타일 순회.
- **`recv_buf`** (`dma.cc:936-967`): AIE 출력 AXIS를 수신. **K 방향 부분합 누산이 PL에서 일어남**: `if (bk1 > 0)` 분기에서 기존 `buf[idx]`와 수신 데이터를 `union_t`로 풀어 `u1.val += u2.val` 부동소수 가산 후 재저장 (`:950-958`). → AIE 어레이가 K를 다 못 담을 때 BK1 회차에 걸쳐 PL이 부분합을 합산.
- **`store_buf`/`recv_C`** (`dma.cc:969-1070`): 누산된 C 버퍼를 split_C 스트림으로 내보내고 이중버퍼로 다음 타일 수신. `first`(bm1==0&&bn1==0)면 skip.
- **탑레벨 `dma`** (`dma.cc:1072-1139`):
  - 인터페이스: `pl_in0_*`(A), `pl_in1_*`(B), `pl_out0_*`(C) DRAM 포트 ×`PARTS`, AIE AXIS 스트림 `aie_in0_*`(×8 행), `aie_in1_*`(×50 열), `aie_out0_*`(×50 열).
  - **온칩 버퍼 바인딩** (`:1082-1104`): `buf_A`→**URAM**(`impl=uram`), `buf_B`→**BRAM**, `buf_C`→**URAM**. 각각 `array_partition complete`(dim1~3) + `cyclic`(dim4)로 멀티포트화. **A/C는 URAM, B는 BRAM**이라는 메모리 계층 분담이 핵심 자원 전략.
  - `#pragma HLS dataflow`로 load→split→send→(AIE)→recv→merge→store 전 단계를 동시 파이프라인.
  - HW 빌드(`XILINX_TARGET_IS_HW`)에서는 함수형 dataflow, 그 외(시뮬)는 `hls::task`(자유 실행 스레드)로 송수신 구성 (`:1119-1133`).

### 3.5 `src/host.cc` — XRT 호스트 프로그램
- **메인 흐름** (`host.cc:40-350`): xclbin 로드 → `dma` 커널 객체 생성(`:71`) → 데이터 생성 → `transform_in`으로 A/B를 어레이 레이아웃으로 재배치 → XRT BO에 write/sync(`:204-233`, `PARTS`개 버퍼) → (sw_emu면 `xrt::graph`로 AIE 그래프 실행) → `dma()` 커널 `iters`(=100, 워밍업 10)회 실행 → C 읽기 → `transform_out`으로 복원 → CPU `gemm` 레퍼런스와 `tol=1e-3` 비교(`:332-349`).
- **`transform_in`** (`host.cc:371-492`): 호스트측 데이터 레이아웃 변환. 3단 타일(`PY/PX` PL타일, `AY/AX` AIE타일, `YY/XX` mmul) 계층을 그대로 펼쳐 행/열 방향(`Direction::Row/Col`)에 따라 스트림 분배. 패딩(`Y_pad/X_pad`) 처리 포함(`:376-377`). 즉 **AIE 그래프의 PLIO 입력 순서와 정확히 일치하도록 DRAM 레이아웃을 사전 변환**.
- **`transform_out`** (`host.cc:494-632`): 출력 C를 원본 `(M,N)` 좌표로 역변환. `SS`(=`AIE_ROWS`) 추가 차원으로 행 방향 부분합 분산을 복원.
- **레퍼런스 GEMM** (`host.cc:352-369`): 순수 3중 루프 CPU 검증용.
- **성능 측정** (`host.cc:283-295`): `ops=2·M·K·N·iters`로 GOP/s, 패딩 포함 GOP/s, 전송 포함 GOP/s 산출.
- 주목: 기본 테스트 데이터가 A=인덱스, B=단위행렬(`:91-97`)이라 정합성 검증 단순화.

### 3.6 `src/generate_gemm_data.py` — AIE 시뮬 데이터 생성기
- argparse로 `-d M,K,N`, `-t TM,TK,TN`, `-m MM,KK,NN`, `-a R,C`, dtype 등 입력 (`:9-28`).
- numpy로 A,B 생성(random/indices/ones/rows/cols/identity 모드, `:67-88`), `C=np.matmul(A,B)` (`:88`).
- **핵심**: `host.cc::transform_in`과 동일한 6중 타일 순회로 `in0_r.txt`(A행), `in1_c.txt`(B열), `out0_c.txt`(C열) 시뮬 입력 파일 생성 (`:156-216`). 즉 graph.hh의 PLIO 파일명 규약과 1:1 대응.
- `serialize_mmul`로 mmul 블록 직렬화(`:218-228`), 타일 메모리 32KB 초과 경고(`:254-255`, AIE 타일 메모리 한계).

### 3.7 `scripts/` (보조 — 요약)
- `model.py` + `model.mod`: AMPL+Gurobi 기반 **설계공간 솔버**(타일/어레이 매핑 최적화). `README.md:60-64`.
- `device.py`: VC1902 자원/성능 상수 모델(앞서 인용).
- `parse_profile.py`, `heatmap.py`, `plot_*`, `pie.py`, `monitor_power.py`, `check_simulation.py`: 프로파일 파싱/시각화/전력/시뮬 검증.

---

## 4. 데이터플로우 / 실행 흐름

### 전체 파이프라인
```
DRAM(A,B)
  └─load_A/load_B → dram_stream → split_stream(512b→4×128b, 행/열 라운드로빈)
       → send_A/send_B(이중버퍼 핑퐁, dataflow) → AXIS → AIE PLIO 입력
            → [AIE 8×50 시스톨릭 어레이]
                 A=행 멀티캐스트, B=열 멀티캐스트
                 각 PE: buf0[TM][TN] 출력-고정 누산, K(BK2)만큼 mac
                 부분합 세로 체인: kernel[r+1]→kernel[r].in[2] (아래→위)
                 최상단 행만 out0 출력
            → AXIS → recv_buf(PL): bk1>0이면 부분합 PL 누산(+=)
       → merge_stream(128b→512b) → store_C → DRAM(C)
  └ host: transform_in/out으로 레이아웃 변환 + CPU 레퍼런스 검증
```

### 메모리 계층
- **DRAM/NoC**: 512b 워드, `PARTS=2` 포트로 분할(`xsa.cfg:112-117` `noc.read_bw/write_bw=14000.16`).
- **PL 온칩**: `buf_A`/`buf_C` = **URAM**, `buf_B` = **BRAM** (`dma.cc:1083/1091/1099`). 모두 이중버퍼(dim3 size=2).
- **AIE 로컬**: 코어당 `buf0[TM][TN]` + 입출력 비동기 버퍼. 타일 메모리 32KB 제약(`device.py:72`, `generate_gemm_data.py:254`).

### 병렬화 / dataflow
- PL: 탑레벨 `#pragma HLS dataflow`(`dma.cc:1080`) + 단계별 `II=1` 파이프라인 + `send/recv_inner`의 중첩 dataflow로 송신∥적재 중첩.
- AIE: 8×50=400 PE 공간 병렬 + 각 PE 내부 SIMD MAC(`aie::mac`) 4-way 언롤.
- **이중버퍼링**: A/B 송신, C 수신 모두 핑퐁 버퍼로 통신-계산 중첩.

### 양자화 / 데이터타입
- 기본 **FP32**(`DT=float`, `parameters.hh:9`). `aie_vlen`/`aie_alen` 트레이트(`gemm.cc:6-17`)로 8b/16b/32b 모두 대응 가능하게 설계되어 있으나 현재 활성은 32b.
- INT8/INT16로 바꾸면 `compute()`의 `N_STEP`(`gemm.cc:184` `f?1:2`)이 2로 늘고 AIE MAC 처리량이 16배(`device.py:66-70`)까지 증가 가능 → **저정밀 전환 여지가 코드에 내장**.

---

## 5. HW/SW 매핑

| 구성요소 | 물리 위치 | 역할 | 근거 |
|---|---|---|---|
| `Gemm` 커널 | AIE 타일 `tile(c,r)` | PE별 출력-고정 MAC 누산 | `graph.hh:59` |
| `GemmGraph` | AIE 어레이(ADF) | 8×50 시스톨릭 토폴로지/멀티캐스트/세로 누산 | `graph.hh:48-83` |
| `dma` | PL(HLS) | DRAM↔AIE 스트리밍/디멀티플렉싱/부분합 합산 | `dma.cc:1072` |
| `buf_A/buf_C` | URAM | A/C 온칩 이중버퍼 | `dma.cc:1083,1099` |
| `buf_B` | BRAM | B 온칩 이중버퍼 | `dma.cc:1091` |
| `host.cc` | x86 호스트(XRT) | 레이아웃 변환·실행·검증 | `host.cc:40-350` |
| PL↔AIE 연결 | `xsa.cfg` connectivity | `dma.aie_in0_*↔ai_engine_0.in0_*` 등 | `xsa.cfg:3-110` |
| 빌드 플로우 | `vitis_flow()` | AIE/HLS/link/package/host 통합 | `xilinx-setup.cmake:149-416` |

- PL 주파수: `PL_FREQ_MHZ 1250/4 = 312.5 MHz` (`src/CMakeLists.txt:16`).
- 연결 스케일: A 입력 8행, B 입력 50열, C 출력 50열 (`xsa.cfg:3-110` = 8×50 어레이와 정합).

---

## 6. 빌드·실행 (README + cmake 근거)

빌드 (`README.md:14-34`):
```
mkdir build && cd build
cmake .. [-DVPP_JOBS=n] [-DVPP_OPTIMIZE=0..3] [-DXILINX_TARGET=hw|hw_emu|sw_emu]
make -j gemm          # 또는 gemm-adf gemm-xsa gemm-xclbin gemm-host 개별
```
실행 (`README.md:36-42`):
```
[XCL_EMULATION_MODE=sw_emu|hw_emu] ./bin/gemm ./<xclbin> [DEV_IDX]
```
AIE 시뮬 (`README.md:44-58`): `generate_gemm_data.py`로 데이터 생성 후 `make gemm-x86sim|gemm-aiesim`.
솔버/스크립트: `scripts/model.py`, `parse_profile.py` 등 (`README.md:60-75`).

빌드 플로우 내부(`xilinx-setup.cmake:266-401`): `v++ --mode aie`(x86sim/hw 각각) → `--mode hls`(dma.xo) → `--link`(xsa, 주파수/배치 옵션) → `--package`(xclbin) → host 실행파일. Vitis ≥2022.2 요구(`:44-47`).

---

## 7. 의존성

- **툴체인**: Vitis 2023.1, v++(aie/hls/link/package), AIE 컴파일러(aietools), XRT, x86simulator/aiesimulator (`xilinx-setup.cmake:49-54`).
- **라이브러리**: AIE `adf.h`/`aie_api/aie.hpp`(`gemm.cc:2`), HLS `hls_stream.h`/`hls_task.h`/`ap_axi_sdata.h`(`dma.cc:1-3`), XRT `xrt_*`(`host.cc:1-4`).
- **언어/빌드**: C++17, CMake 3.23+, GNU Make (`README.md:5-10`).
- **Python**: numpy(데이터 생성), amplpy+Gurobi(솔버), matplotlib/seaborn(플롯) (`README.md:11-12`).
- **플랫폼**: `xilinx_vck5000_gen4x8_qdma_2_202220_1`(VCK5000) (`xilinx-setup.cmake:7`).

---

## 8. 강점 / 한계 / 리스크

### 강점
- **3단 타일 계층 + 컴파일타임 검증**: `parameters.hh`의 24개 `static_assert`로 매핑 불일치를 사전 차단.
- **출력-고정 시스톨릭 + PL 부분합 합산 하이브리드**: AIE 어레이가 K를 다 못 담아도 PL `recv_buf`에서 BK1 누산(`dma.cc:950`)으로 임의 K 처리.
- **통신-계산 중첩**: 전 구간 이중버퍼 + dataflow.
- **메모리 계층 분담**: A/C URAM, B BRAM으로 포트 충돌 완화 및 자원 균형.
- **저정밀 확장성 내장**: 8/16/32b 트레이트가 이미 코드에 존재.
- **DSE 인프라**: AMPL/Gurobi 솔버 + 디바이스 모델 + 프로파일 파서까지 갖춘 연구급 완결성.

### 한계
- **VCK5000 특정**: 8×50 어레이는 VC1902 고정. KV260급 소형 디바이스로 이식하려면 어레이/타일 전면 재설정 필요.
- **FP32 기본**: 트랜스포머 추론용 INT8 경로는 미활성(전환 여지만 존재).
- **거대 매크로 코드 생성**: `dma.cc`의 50개 분기 매크로는 가독성/유지보수성이 낮음(스트림 개수 하드 상한 50).
- **테스트 데이터 단순화**: 호스트 기본값이 단위행렬(`host.cc:91-97`)이라 일반 정확도 검증력 제한("추정").

### 리스크
- AIE 타일 메모리 32KB 초과 시 빌드 실패(`generate_gemm_data.py:254`) — 타일 크기 변경 시 주의.
- 변형(`src-*`) 간 동기화 누락 가능성("확인 불가").

---

## 9. 우리 프로젝트(HG-PIPE 계열 ViT/Transformer FPGA 가속기 + XR 시선추적) 관점 시사점

> 전제: 본 레포는 **데이터센터급 Versal AIE GEMM**으로, HG-PIPE(소형 FPGA, 고처리량 ViT)와는 타깃이 다르나(추정), GEMM 매핑/데이터무빙 기법은 직접 재사용 가능.

1. **3단 타일 계층 설계 패턴**: `PL타일 → AIE타일 → mmul` 계층(`parameters.hh`)은 ViT의 `토큰×채널` GEMM, 어텐션 Q/K/V 프로젝션을 FPGA PE 어레이에 매핑할 때 그대로 차용 가능. 특히 `static_assert` 기반 컴파일타임 정합성 강제는 HG-PIPE 계열 파이프라인 파라미터 검증에 유용.
2. **출력-고정 + PL 부분합 합산 하이브리드**(`dma.cc:950`): 온칩 PE가 K(채널/시퀀스)를 한 번에 못 담는 대형 어텐션에서, 외부 누산으로 임의 길이 K를 처리하는 패턴은 긴 시퀀스 ViT/Transformer에 직접 적용.
3. **이중버퍼 dataflow 데이터무버**(`send_A/recv_C`): HG-PIPE의 레이어 간 무중단 스트리밍 파이프라인 설계에 참고할 통신-계산 중첩 레퍼런스.
4. **메모리 계층 분담**(URAM/BRAM 역할 분리): 가중치(BRAM) vs 활성/출력(URAM) 분담 전략은 한정된 온칩 메모리에서 ViT 가속기 자원 배분에 시사.
5. **저정밀 전환 경로**(`gemm.cc:183-184`, `device.py:66-70`): INT8/INT4 양자화로 MAC 처리량 16배 확보 가능 — Mamba/ViT 양자화 가속기에서 정밀도별 PE 처리량 모델링의 정량 근거.
6. **DSE 솔버**(`model.mod` AMPL/Gurobi): 타일/어레이/주파수 설계공간을 수리최적화로 탐색하는 프레임은 우리 가속기 파라미터 자동튜닝에 이식 가치 큼.
7. **XR 시선추적 직접 연결성은 낮음**(추정): 본 레포는 범용 GEMM 백엔드이므로, 시선추적 CNN/ViT의 GEMM 백본 가속 시 연산 커널 레퍼런스로만 활용.

---

## 10. 근거 표기

- **확인된 사실(라인 근거)**: 타깃 디바이스(VCK5000/VC1902), 8×50 어레이, FP32, 3단 타일, 출력-고정 시스톨릭, PL 부분합 누산, URAM/BRAM 분담, 이중버퍼 dataflow, 빌드 플로우 — 모두 위 인용 라인 직접 확인.
- **추정**: 원논문 정체(저장소 내 명시 없음), `src-*` 변형의 세부 차이, 호스트 기본 데이터의 검증 의도, HG-PIPE/XR 연계성.
- **확인 불가**: 실제 합성 후 PL 자원 점유율·달성 주파수·실측 GOP/s(빌드 산출물 미포함), 변형 간 성능 비교 수치.
