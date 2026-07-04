# hls-spmv 정밀 분석

## 1. 개요

- **목적**: 희소 행렬-벡터 곱(SpMV, Sparse Matrix-Vector multiplication)을 Vivado HLS로 구현하고, 여러 최적화 기법(파이프라인·부분 언롤링·dataflow 스트리밍)에 따른 면적/지연 트레이드오프를 학습/실험하는 교육용 코드베이스.
- **한줄요약**: CSR 포맷 256×256 희소 행렬에 대해 `y = A·x`를 계산하는 SpMV 커널을 5가지 버전(naive MV, 기본 SpMV, partial unrolling, naive stream, fast stream)으로 작성하고 Vivado HLS C-synthesis 리포트로 비교한 실습 과제.
- **원논문/출처**: 학술 논문 없음. National Taiwan University 강의 "Advanced Computer Architecture (CSIE5059), Spring 2021"의 Lab #B 과제물 (`README.md` L1-5). 상세는 `slides.pdf`(바이너리, 분석 제외)에 있다고 명시 — 확인 불가.
- **타깃 디바이스**: Xilinx Zynq-7000 `xc7z020-clg484-1` (PYNQ-Z2/Zedboard 급 소형 SoC FPGA). 근거: `rpt/fast_stream/spmv_csynth.rpt` L11-12 `Product family: zynq`, `Target device: xc7z020-clg484-1`. 툴체인: Vivado HLS 2019.2 (`rpt/.../spmv_csynth.rpt` L8). 목표 클럭 5.00 ns(=200 MHz) (L23).

> 본 repo는 소규모 단일 자체 소스이며 vendor/생성물이 거의 없다. 분석 대상은 전부 자체 작성 핵심 소스.

## 2. 디렉토리 구조

### 자체 소스 트리 (분석 대상)
```
hls-spmv/
├── README.md                          # 과제 설명(5줄)
├── src/
│   ├── spmv.h                         # 공통 헤더: 상수·typedef·함수 프로토타입
│   ├── spmv.cpp                       # [v2] 기본 SpMV(L2 inner-loop pipeline) + 밀집 mv()
│   ├── spmv_partial_unrolling.cpp     # [v3] S=9 부분 언롤링, pipeline II=S
│   ├── spmv_naive_stream.cpp          # [v4] hls::stream + DATAFLOW 기반 스트리밍 SpMV
│   └── spmv_fast_stream.cpp           # [v5] II=9 패딩 기반 고처리량 스트리밍 SpMV
├── tb/
│   ├── spmv_test.cpp                  # C 시뮬레이션 테스트벤치(밀집 MV와 결과 비교)
│   ├── testcase.py                    # scipy.sparse로 행렬/CSR 데이터 생성기
│   ├── matrix.dat / data.dat / rows.dat / cols.dat   # 입력 데이터(생성물, 분석 제외)
└── rpt/                               # Vivado HLS csynth 리포트(생성물, 수치만 인용)
    ├── pipeline/                      # spmv.cpp 합성 결과
    ├── pipeline&partial_unrolling/    # spmv_partial_unrolling.cpp 결과
    ├── pipeline&partial_unrolling_auto/
    ├── naive_stream/                  # spmv_naive_stream.cpp 결과
    └── fast_stream/                   # spmv_fast_stream.cpp 결과
```

### 제외 목록 (vendor/생성물/바이너리)
- `slides.pdf` (바이너리 강의자료)
- `tb/*.dat` (행렬/벡터 데이터 — 생성물, `testcase.py` 산출)
- `rpt/**/*.rpt` (Vivado HLS 생성 리포트 — 수치 근거로만 인용)
- `.git/`, `.gitignore`

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 `src/spmv.h` — 공통 헤더 (전 버전 공유)

```c
const static int SIZE = 256;       // 정사각 행렬 한 변 크기 (spmv.h L4)
const static int NNZ = 3277;       // 비제로 원소 개수 (L5)
const static int NUM_ROWS = 256;   // 행 수 (L6)
typedef float DTYPE;               // 데이터 타입 = float (L7)
void spmv(int rowPtr[NUM_ROWS+1], int columnIndex[NNZ],
          DTYPE values[NNZ], DTYPE y[SIZE], DTYPE x[SIZE]);  // L8-9
void mv(DTYPE A[SIZE][SIZE], DTYPE y[SIZE], DTYPE x[SIZE]);  // L10
```
- **CSR(Compressed Sparse Row) 인터페이스**: `rowPtr`(행 포인터, 길이 NUM_ROWS+1), `columnIndex`(각 nnz의 열 인덱스, 길이 NNZ), `values`(각 nnz의 값, 길이 NNZ), 출력 `y`, 입력 벡터 `x`.
- **데이터 타입은 single-precision float(`DTYPE=float`)** — 양자화/고정소수점 미적용. 모든 버전이 이 한 정의를 공유하므로, 버전 간 차이는 **루프 스케줄링/메모리 액세스 패턴**에 한정됨.
- 행렬 밀도 ≈ NNZ/SIZE² = 3277/65536 ≈ 5%. 단, `testcase.py`는 density=0.3으로 생성하므로 헤더 상수와 데이터 생성기 설정이 불일치할 수 있음("추정": NNZ=3277은 특정 random_state 산출값으로 보이나 testcase.py density=0.3이면 NNZ≈19660이어야 함 → 헤더와 데이터가 다른 시드로 생성됐을 가능성, **확인 불가**).

### 3.2 `src/spmv.cpp` — 기본 SpMV(v2) + 밀집 MV 참조

#### `spmv()` (L3-14)
```c
L1: for (int i = 0; i < NUM_ROWS; i++) {        // 각 행 (L6)
        DTYPE y0 = 0;
    L2: for (int k = rowPtr[i]; k < rowPtr[i+1]; k++) {   // 해당 행의 nnz 구간 (L8)
#pragma HLS pipeline                            // inner loop 파이프라인 (L9)
            y0 += values[k] * x[columnIndex[k]];          // MAC + gather (L10)
        }
        y[i] = y0;                              // 행 결과 저장 (L12)
    }
```
- **알고리즘**: 표준 CSR-row SpMV. 외부 루프 L1은 행, 내부 루프 L2는 `rowPtr[i]`~`rowPtr[i+1]` 구간의 nnz를 순회하며 `values[k] * x[columnIndex[k]]`를 누산.
- **입력**: rowPtr, columnIndex, values, x / **출력**: y.
- **HW 특성**: inner-loop에 `#pragma HLS pipeline`(L9). 그러나 `y0 += ...`는 float 누산기 의존(loop-carried dependency)이고, float 덧셈 지연 + `x[columnIndex[k]]`의 **간접(gather) 메모리 접근**(BRAM 단일 포트) 때문에 II=1 달성이 어렵다. 합성 결과 `rpt/pipeline/spmv_csynth.rpt` L46: inner loop L2가 **II achieved = 9, target=1**, iteration latency=18로 보고됨(목표 미달). 외부/내부 trip count가 데이터 의존(`rowPtr[i+1]` 가변)이라 전체 latency가 `?`로 보고(L32-33). 자원: DSP 5, LUT 958, FF 808 (L60-66) — 매우 작음.

#### `mv()` (L17-27)
- 밀집 행렬-벡터 곱 참조 구현. `A[i][j]*x[j]` 전체 순회, inner-loop pipeline. SpMV 대비 정확도/성능 비교용 baseline. 실제 합성보다 테스트벤치 검증의 의미가 큼.

### 3.3 `src/spmv_partial_unrolling.cpp` — 부분 언롤(v3)

```c
const static int S = 9;                          // 언롤 폭 (L3)
L2_1: for (int k = rowPtr[i]; k < rowPtr[i+1]; k+=S) {  // S개씩 묶어 처리 (L10)
#pragma HLS pipeline II=S                         // II=9로 명시 (L11)
        DTYPE yt = values[k] * x[columnIndex[k]];
    L2_2: for (int j = 1; j < S; j++) {          // 나머지 S-1개 (L13)
            if (k + j < rowPtr[i + 1]) {          // 경계 검사 (L14)
                yt += values[k+j] * x[columnIndex[k+j]];
            }
        }
        y0 += yt;                                 // 부분합 누산 (L18)
    }
```
- **알고리즘**: 내부 루프를 stride S=9로 분할. 한 외부 반복(L2_1)당 S개의 곱-누산을 inner(L2_2)에서 수행, 부분합 `yt`를 `y0`에 합산.
- **목적**: float 누산 의존성을 완화. inner 곱셈은 병렬화하되 누산 체인을 짧게 하여 II=9 파이프라인에서 처리량 향상.
- **한계**: `II=S=9`로 명시했으나, gather 접근(`x[columnIndex[k+j]]`)이 S개 동시 → 단일 포트 BRAM 충돌로 자원/타이밍 압박. `rpt/pipeline&partial_unrolling/spmv_csynth.rpt` 수치는 본 분석에서 직접 인용하지 않았으나(파일 존재 확인), v3는 v2 대비 inner 곱셈 병렬화를 시도한 중간 단계.
- **`_auto` 변형**: `rpt/pipeline&partial_unrolling_auto/`는 HLS auto-unroll/스케줄 옵션 차이 실험으로 추정(소스는 동일, **확인 불가**).

### 3.4 `src/spmv_naive_stream.cpp` — 스트리밍 SpMV(v4)

#### `spmv()` 래퍼 (L52-62)
- `rowPtr` → `rows_length`(행별 nnz 개수) 변환(L55-59): `rows_length[i-1] = rowPtr[i] - rowPtr[i-1]`. 이후 `spmv_kernel` 호출.

#### `spmv_kernel()` (L4-49) — `#pragma HLS DATAFLOW` (L6)
4개 FIFO 선언: `rows_fifo`, `values_fifo`, `cols_fifo`, `results_fifo` (L8-11).
- **Stage 1 (L14-17)**: `rows_length[i]`를 `rows_fifo`로 push (256회).
- **Stage 2 (L19-23)**: `values[i]`, `cols[i]`를 각각 FIFO로 push (NNZ회).
- **Stage 3 (L30-43) — 핵심 누산 루프**:
```c
for (int i = 0; i < NNZ; i++) {
#pragma HLS PIPELINE                  // L31
    if (col_left == 0) {              // 새 행 시작
        col_left = rows_fifo.read();  // 이 행의 nnz 개수
        sum = 0;
    }
    value = values_fifo.read();
    col   = cols_fifo.read();
    sum  += value * x[col];           // MAC + gather (L38)
    col_left--;
    if (col_left == 0) {              // 행 끝 → 결과 emit
        results_fifo << sum;          // L41
    }
}
```
  - **알고리즘 핵심**: 행 경계를 `col_left` 카운터로 추적해 **단일 평탄화 루프**로 모든 nnz를 순회(중첩 루프 제거). 행이 끝날 때만 `results_fifo`에 결과를 쓴다.
  - **장점**: 외부/내부 이중 루프의 데이터 의존 trip count 문제를 평탄화로 해결, dataflow로 stage 병렬 실행.
  - **한계**: `sum += value * x[col]`의 float 누산 loop-carried dependency가 여전히 존재 → II=1 미달성. `rpt/naive_stream/spmv_csynth.rpt` L49-51: memset_rows_length latency 255, Loop2 II=1 달성(255 trip), 전체 latency=30026 cycles(0.150 ms @200MHz, L32). 자원: DSP 5, LUT 1981, FF 1353 (L62-71). v2보다 약간 큼.
- **Stage 4 (L45-48)**: `results_fifo`에서 256개 결과를 읽어 `y[i]`로 write.

### 3.5 `src/spmv_fast_stream.cpp` — 고처리량 스트리밍 SpMV(v5, 최적화 정점)

`#define II 9` (L4). 핵심 아이디어: **행 길이를 II=9의 배수로 패딩**하여 누산을 9-wide 트리로 처리, loop-carried 의존 거리를 늘려 파이프라인 효율을 끌어올림.

#### `spmv()` 래퍼 (L72-100)
1. rowPtr → `rows_length`(L74-79).
2. **패딩 메타데이터 계산 (L81-97)**: 각 행 길이 `r`에 대해
   - `r==0` → `rows_length_pad=II`(빈 행도 1블록 처리), `new_nnz += II`
   - `r % II != 0` → `r + (II - r%II)`로 올림 패딩
   - 그 외 → 그대로
   - `new_nnz`는 패딩 후 총 nnz(9의 배수).
3. `spmv_kernel(rows_length, rows_length_pad, cols, values, y, x, new_nnz)` 호출.

#### `spmv_kernel()` (L6-69) — `#pragma HLS DATAFLOW` (L15)
- **Stage 1 (L27-31)**: values/cols를 FIFO로 push.
- **Stage 2 (L33-63) — 9-wide 병렬 MAC 루프**:
```c
for (int i = 0; i < new_nnz; i += II) {    // II개씩 전진 (L33)
#pragma HLS PIPELINE                        // L34
    if (row_length_pad == 0) {              // 새 행 로드
        row_length_pad = rows_length_pad[k];
        row_length = rows_length[k++];
        row_counter = 0;
        sum = 0;
    }
    for (int j = 0; j < II; j++) {          // 9개 곱 동시 (L42)
        row_counter++;
        if (row_counter > row_length) {
            term[j] = 0;                    // 패딩 영역은 0 (L45)
        } else {
            value = values_fifo.read();
            col   = cols_fifo.read();
            term[j] = value * x[col];       // MAC 항 (L49)
        }
    }
    DTYPE sum_tmp = 0;
    for (int j = 0; j < II; j++) sum_tmp += term[j];   // 9항 합산(트리) (L54-56)
    sum += sum_tmp;                          // 행 부분합 누산 (L57)
    row_length_pad -= II;
    if (row_length_pad == 0) results_fifo << sum;       // 행 완료시 emit (L60-61)
}
```
  - **알고리즘 핵심**: 행 길이를 II=9 배수로 패딩하여, 한 파이프라인 단계에서 정확히 9개의 곱(`term[0..8]`)을 계산하고 합산. `term[II]` 배열(L25)이 9-wide 부분합 레지스터. 패딩 영역(`row_counter > row_length`)은 0으로 채워 정확도 유지.
  - **장점**: 행 경계 분기를 매 nnz가 아닌 매 9-nnz 블록 단위로 줄여 파이프라인 stall 감소, 9개 MAC 병렬화로 처리량 향상.
- **Stage 3 (L65-68)**: 결과 256개 → `y[i]`.

#### 합성 결과 (근거: `rpt/fast_stream/spmv_csynth.rpt`)
- 클럭: 목표 5.00 ns, 추정 5.409 ns (**타이밍 미세 위반**, L23). Uncertainty 0.62.
- 루프(L49-52): memset 255, **Loop2 II=1 달성(259 cyc)**, **Loop3 II=1 달성(294 cyc, iteration latency 40)**. 9-wide 트리 합산으로 II=1 파이프라인 달성이 핵심 성과.
- 자원(L60-76): BRAM 2, DSP 5, FF 9070(8%), LUT 6267(11%). v2(LUT 958) 대비 약 6.5배 LUT 증가 — 9-wide 병렬화·패딩 메타데이터 처리(`spmv_srem_32ns` 나눗셈 모듈 FF 2283/LUT 1738, L84)의 비용.
- 인터페이스(L286-314): 모든 배열이 `ap_memory`(BRAM/외부 메모리 포트). `rowPtr`만 2-포트(q0/q1), 나머지 1-포트.

### 3.6 `tb/spmv_test.cpp` — C 시뮬레이션 테스트벤치

- `load_matrix/load_data/load_rows/load_cols` (L27-89): `.dat` 파일에서 밀집 행렬 M, values, rowPtr, columnIndex를 로드.
- `gen_input()` (L91-96): `x[i] = rand()%100`로 입력 벡터 생성.
- `main()` (L99-121): `spmv(...)`(피검증)와 `matrixvector(M, y_sw, x)`(밀집 참조, L17-25)를 모두 실행 → `y_sw[i] != y[i]` **정확한 비교**(L111). 정수 값(`rand()%100`, 정수 행렬)이라 float 오차 없이 등호 비교 가능.

### 3.7 `tb/testcase.py` — 데이터 생성기

- `scipy.sparse.random(256, 256, density=0.3, format="csr", random_state=2906, data_rvs=randint(1,100), dtype=int32)` (L10): 256×256, 밀도 0.3, 정수값 1~99 희소 행렬 생성.
- `matrix.dat`(밀집 형태, L11), CSR 변환 후 `rows.dat`(rowPtr), `cols.dat`(columnIndex), `data.dat`(values) 저장(L40-42).
- CSR 변환 로직(L24-34): 행 변화 감지하여 누적 nnz 카운트를 `rows_compressed`에 push (수동 CSR indptr 구성).

## 4. 데이터플로우 / 실행 흐름

### 메모리 계층
- 모든 배열(rowPtr/cols/values/x/y)이 HLS top 함수 인자 → `ap_memory` 인터페이스(BRAM 매핑, 리포트 L286-314). HBM/외부 DRAM 명시적 사용 없음(Zynq BRAM 스케일).
- gather 접근 `x[columnIndex[k]]`가 **불규칙 인덱싱**(SpMV의 본질적 병목). 단일 포트 BRAM이라 동시 다중 gather 불가 → 병렬화의 근본 제약.

### 파이프라인/병렬화 진화
| 버전 | 기법 | inner II | 비고 |
|------|------|----------|------|
| spmv.cpp (v2) | inner pipeline | 9 달성(target1 미달) | float 누산 의존 |
| partial_unrolling (v3) | S=9 부분언롤, II=9 | II=9 명시 | inner 곱 병렬 |
| naive_stream (v4) | DATAFLOW + 평탄화 루프 | Loop2 II=1 | 행경계 카운터 |
| fast_stream (v5) | DATAFLOW + II=9 패딩 + 9-wide 트리합 | **Loop2/3 II=1** | 처리량 정점, 타이밍 5.409ns 미세위반 |

### 데이터타입/양자화
- 전 버전 `DTYPE=float`(single precision). **양자화·고정소수점 미적용**. 이는 본 repo의 분명한 한계이자, HiSparse(ap_ufixed) 및 우리 프로젝트(양자화 가속기)와의 대비점.

### dataflow 구조 (v4/v5)
- `#pragma HLS DATAFLOW`로 "FIFO push → 누산 → 결과 drain" 3~4 스테이지를 task-level 병렬화. 스테이지 간 `hls::stream` FIFO로 연결. 단, 전체 dataflow latency가 데이터 의존이라 리포트에 `?`로 표기되는 경우가 많음.

## 5. HW/SW 매핑

| 계층 | 구성요소 | 역할 |
|------|----------|------|
| SW (host, C-sim) | `tb/spmv_test.cpp` | 데이터 로드·입력 생성·정확도 검증(밀집 MV 대조) |
| SW (data gen) | `tb/testcase.py` | scipy로 희소행렬·CSR 데이터 생성 |
| HW (HLS kernel) | `spmv()` / `spmv_kernel()` | CSR SpMV 연산. BRAM 인터페이스(ap_memory) |
| HW 합성 | Vivado HLS 2019.2 | `xc7z020` 타깃 C-synth, 리포트 산출 |

- **실제 SW/HW co-execution(XRT/OpenCL 호스트)은 없음** — 본 repo는 C-sim + C-synthesis 수준의 교육 과제. 비트스트림/board 실행 인프라 부재("추정", 근거: host/Makefile/xclbin 부재).

## 6. 빌드·실행

- **명시적 빌드 스크립트 없음**(Makefile/tcl 부재 확인). README는 `slides.pdf` 참조만 안내(L5).
- 일반적 흐름("추정", HLS 표준 절차): ① `testcase.py` 실행 → `.dat` 생성. ② Vivado HLS 프로젝트에 `src/<버전>.cpp` + `tb/spmv_test.cpp` 추가. ③ C-sim(정확도) → C-synth(리포트 `rpt/`에 대응). ④ 버전 교체로 최적화 비교.
- 데이터 파일은 C-sim 시 cwd 기준 `./matrix.dat` 등 상대경로 로드(`spmv_test.cpp` L29 등).

## 7. 의존성

- **HLS 라이브러리**: `<hls_stream.h>`(v4/v5의 `hls::stream`). float 연산은 표준 C.
- **SW 측**: Python `scipy.sparse`, `numpy`(`testcase.py` L7-8).
- **툴**: Vivado HLS 2019.2 (리포트 L8). 외부 BLAS/cnpy 등 의존 없음.

## 8. 강점 / 한계 / 리스크

### 강점
- SpMV 최적화 기법의 **점진적 진화(v2→v5)를 한 repo에서 직접 비교** 가능. 교육 가치 높음.
- v5의 "행 길이 패딩 + 고정 폭(II=9) 트리 합산"으로 가변 trip count 루프를 **II=1 파이프라인으로 변환**한 기법은 실전 응용 가치가 있음.
- dataflow + FIFO 평탄화로 중첩 루프 데이터 의존 문제를 해소.

### 한계
- **소규모 고정 크기**(256×256, NNZ=3277)에 하드코딩. 동적 크기/대형 행렬 미지원.
- **float 전용**, 양자화/고정소수점 없음 → 자원 효율 낮음(DSP/LUT가 float MAC에 소모).
- **단일 포트 BRAM gather**가 병렬화 상한. 멀티뱅크/셔플 네트워크 없음(HiSparse 대비 결정적 차이).
- v5 타이밍 5.409 ns > 목표 5.0 ns(미세 위반) — 200 MHz 미달 가능.
- 호스트/board 실행 인프라 부재(C-sim/C-synth 한정).

### 리스크
- 헤더 NNZ=3277과 `testcase.py` density=0.3 불일치 가능성 → 데이터-헤더 정합성 검증 필요("추정").
- 빌드 자동화 부재로 재현은 수동 Vivado HLS 절차 의존.

## 9. 우리 프로젝트 관점 시사점 (고처리량 ViT/Transformer FPGA 가속기 HG-PIPE 계열 + XR 시선추적)

> 우리 프로젝트가 ViT/Transformer 가속기 + XR 시선추적이라는 추정 하에:

1. **희소 GEMM/attention pruning 재사용**: Transformer attention/FFN을 pruning하면 SpMV/SpMM 구조가 된다. v5의 **행 길이 패딩 → 고정 폭 II=1 파이프라인** 기법은, 가변 nnz를 가진 sparse attention head를 systolic/벡터 파이프라인에 매핑할 때 직접 차용 가능. 단, 본 repo의 단일포트 BRAM gather 한계를 멀티뱅크로 극복해야 함.
2. **dataflow 평탄화 패턴**: 중첩 루프(행×nnz)를 카운터 기반 단일 루프로 평탄화하는 v4 기법은, 토큰×채널 같은 가변 길이 루프를 HLS dataflow로 매핑할 때 stall 감소에 유용.
3. **반면교사 — 양자화 부재**: 본 repo는 float 전용이라 자원 비효율. 우리 양자화(INT8/4, Mamba 양자화) 가속기에서는 HiSparse의 `ap_ufixed` 접근이 더 적합한 baseline.
4. **교육적 baseline**: SpMV 최적화 5단계 비교는 우리 DSE(설계공간탐색) 시 "naive → pipeline → unroll → dataflow → padding"의 단계별 면적/지연 곡선을 빠르게 감 잡는 참조로 활용.
5. **시선추적 직접 연관성은 낮음**: XR gaze estimation 자체와는 무관(SpMV 일반론). 단, gaze 모델이 sparse/pruned MLP를 쓴다면 간접 재사용 가능.

## 10. 근거 표기

- **확인(코드 라인 직접)**: §3 전체(spmv.h/cpp/partial/naive_stream/fast_stream 라인 인용), §3.6-3.7 tb, 합성 리포트 수치(rpt/*.rpt 직접 인용).
- **추정**: 헤더 NNZ와 testcase density 불일치, `_auto` 리포트 변형의 차이, 빌드 절차(Makefile 부재로 표준 HLS 흐름 가정), host/board 실행 인프라 부재.
- **확인 불가**: `slides.pdf` 내용(바이너리), v3 partial_unrolling 합성 수치(리포트 파일은 존재하나 본 분석에서 미인용), 데이터 생성 시드와 헤더 상수의 정확한 정합 경위.
