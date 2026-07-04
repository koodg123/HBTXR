# hls-fpga-accelerators 정밀 분석

## 1. 개요
- **한 줄 요약**: LLM(특히 LLaMA 계열) 실행에 필요한 핵심 연산자(GEMM, elementwise, unary 활성화, RMSNorm, Softmax)를 각각 독립적인 Vitis HLS 커널로 구현한 **재사용 가능 연산자 라이브러리**.
- **목적**: 데이터타입(FP4/FP8/FP16/FP32/FIXED8/FIXED16), 버스폭(64~2048bit), 행렬 크기를 환경변수로 파라미터화하여 동일 알고리즘을 다양한 정밀도/플랫폼에 합성할 수 있게 함. DSE(Design Space Exploration)를 전제로 한 구조.
- **출처/저자**: Luis G. Leon-Vega (luis.leon@ieee.org), Luis Prieto Sibaja. (논문 명시는 README에 없음 — 확인 불가, 학술 그룹의 커널 모음으로 추정)
- **타깃 디바이스**: Alveo U250 (`xcu250-figd2104-2L-e`) 및 Kria K26 (`xck26-sfvc784-2LV-c`). 기본값은 U250. (README:22~24)

## 2. 디렉토리 구조 (자체 소스)
```
hls-fpga-accelerators/
├── README.md                  # 커널별 빌드/시그니처/환경변수 문서
├── common/config.h            # 공통 데이터타입·버스폭·패킷화 정의 (핵심 헤더)
├── matmul/
│   ├── matmul.cpp             # GEMM 커널 (dataflow + 스트리밍)
│   ├── matmul.h               # 행렬 크기 상수(kARows=2, kBCols/kCCols 기본 32768)
│   ├── matmul.tcl            # Vitis HLS 합성 스크립트
│   └── matmul_tb.cc          # 테스트벤치
├── elementwise/
│   ├── elementwise.cpp       # 원소별 add/mult 커널
│   ├── elementwise.h, .tcl
├── unary/
│   ├── unary.cpp             # ReLU / SiLU 활성화 커널
│   ├── unary.h, .tcl
│   └── axc-math/
│       ├── exponential-lut.hpp        # LUT 기반 exp 근사 (선형보간)
│       └── interpolation-wrapper.hpp  # 선형보간 헬퍼
├── rmsnorm/
│   ├── rmsnorm.cpp           # RMS 정규화 커널
│   ├── rmsnorm.h, .tcl
└── softmax/
    ├── softmax.cpp           # softmax 커널
    ├── softmax.h, .tcl
```
- **제외한 third-party/vendor**: 없음(저장소 전체가 자체 소스). `*_tb.cc` 테스트벤치는 분석 대상에서 보조적으로만 다룸.

## 3. 핵심 모듈·파일별 정밀 분석

### 3.1 common/config.h — 데이터타입·패킷화 추상화 (가장 중요한 공통 인프라)
- 버스폭 `kBusWidth`(기본 512)와 데이터타입을 매크로(`USE_FLOAT32/16/8/4`, `USE_FIXED16`, 기본 FIXED8)로 선택 (config.h:14~39).
- **핵심 설계 패턴**: 메모리 인터페이스는 항상 `ap_uint<kBusWidth>`(=`RawDataT`)의 단일 워드로 전송하고, 그 안에 `kPackets = kBusWidth / kDataWidth`개의 실제 데이터 원소를 SIMD처럼 패킹 (config.h:41~44). 예: 512bit 버스 + FIXED8 → 64-way 병렬.
- `AccT`(누산기 타입)는 부동소수면 `union{uint; half/float}`로(비트 재해석 위해), 고정소수면 `ap_fixed`로 정의. 접근은 `GET_NUMBER`/`GET_RAW` 매크로로 통일하여 float/fixed 코드 경로를 단일 소스로 유지 (config.h:46~70). FIXED16은 Q6.10(`ap_fixed<16,6>`), FIXED8은 Q4.4(`ap_fixed<8,4>`).

### 3.2 matmul/matmul.cpp — 스트리밍 GEMM 커널
- 외부 함수 `matmul(a, b, c, a_rows, b_cols, c_cols)` (matmul.cpp:146). `a`=입력(samples×inputs), `b`=가중치(outputs×inputs, **전치 가정**), `c`=출력(samples×outputs). m_axi 3뱅크(gmem0/1/2) + s_axilite 제어 (matmul.cpp:148~154).
- **dataflow 4단 파이프라인** (matmul.cpp:163~167):
  1. `matmul_to_stream_a` — A를 스트림으로 공급. 각 A행을 `rep_rows`(=c_cols)번 반복 전송하여 B의 모든 출력열과 매칭 (matmul.cpp:68~96).
  2. `matmul_to_stream_b` — B를 스트림으로 공급. 전체 B를 `rep_mats`(=a_rows)번 반복 (matmul.cpp:98~126).
  3. `matmul_gemm` — 실제 곱셈누산. c_row→c_col(kPackets 단위)→c_p(패킷 내 출력)→b_col 루프. 패킷 내 `kPackets`개 곱셈을 `#pragma HLS UNROLL`로 완전 병렬화하여 1 사이클에 다중 MAC (matmul.cpp:13~66, 특히 34~52).
  4. `matmul_from_stream` — 결과 패킷을 C로 write-back (matmul.cpp:128~136).
- **알고리즘 특성**: B를 전치 저장 가정 → A행·B행이 동일 reduction축으로 정렬되어 내적이 연속 메모리 접근이 됨. 누산은 `AccT`(고정/부동)로 수행 후 패킷에 재패킹.
- **한계 근거**: `kARows=2`로 매우 작게 고정(matmul.h:11) → 사실상 행 2개(예: 토큰 2개)짜리 GEMM 벤치/PoC 성격. B/C 열은 기본 32768로 큼.

### 3.3 softmax/softmax.cpp — 2-pass 스트리밍 softmax
- `compute()`가 핵심: **1패스**에서 `hls::exp`로 각 원소 지수화 후 전역 합 `sum` 누적(cumsum_out/in, softmax.cpp:15~47), **2패스**에서 `scale=1/sum`을 곱해 정규화 (softmax.cpp:49~77).
- 입력 스트림을 두 번 소비하므로 `load_input`이 동일 데이터를 **2회 반복 전송**(mem_reps i<2, softmax.cpp:83~92)하는 구조 — 즉 재계산형 2-pass. depth=32 FIFO로 dataflow 연결.
- **주의**: max-subtraction(수치안정화)이 없음 → 큰 입력에서 overflow 위험. 그래서 README가 "softmax/rmsnorm은 FLOAT 타입 권장"이라고 명시(README:154). 우리 관점에서 이는 개선 포인트.

### 3.4 rmsnorm/rmsnorm.cpp — RMS 정규화
- softmax와 동일한 2-pass 골격. 1패스: 제곱합 `sum` → `mean=sum/n` → `scale=1/sqrt(mean+eps)` (eps=0.01 하드코딩, rmsnorm.cpp:15~55). 2패스: 각 원소에 `scale` 곱 (rmsnorm.cpp:57~82).
- weight(γ) 곱이 없음 — 순수 RMS 스케일만 수행(LLaMA RMSNorm의 learnable weight는 별도 elementwise 곱으로 처리해야 함). eps가 0.01로 일반적 1e-5/1e-6보다 큼(저정밀 안정화 의도로 추정).

### 3.5 unary/unary.cpp — 활성화(ReLU/SiLU) + LUT exp
- op=1 ReLU, op=2 SiLU (unary.cpp:10~17). SiLU = x·sigmoid(x) = x·exp(x)/(1+exp(x)).
- **두 가지 exp 구현 경로**(IMPLEXP 환경변수):
  - `USE_LUT`: `axc::...::Exponential<ap_fixed<24,12>, ratio<-6>, ratio<6>, 32>` LUT(32포인트, [-6,6] 구간)로 선형보간 근사 (unary.cpp:35~43, 78~83). 구간 밖(x≥6)은 포화 처리.
  - 비-LUT: `hls::exp` 표준 사용 (unary.cpp:85~88).
- **재사용 가치**: exponential-lut.hpp(3.6)는 도메인 한정 LUT+선형보간 패턴으로, GELU/softmax exp 근사에 그대로 응용 가능.

### 3.6 unary/axc-math/exponential-lut.hpp — 도메인 한정 LUT exp 생성기
- `generator::Exponential`: 컴파일타임에 `[BEGIN,END]` 구간을 `S`등분하여 `std::exp(x)` LUT 채움 (exponential-lut.hpp:65~71). `BEGIN/END`를 `std::ratio`로 받아 컴파일타임 상수화.
- `lut::Exponential::operator()`: `LinearInterpolation` 헬퍼로 LUT 보간 (exponential-lut.hpp:116~125). 정밀도(`T`), 구간, 포인트수가 전부 템플릿 파라미터 → 면적/정확도 트레이드오프 튜닝 용이.

### 3.7 elementwise/elementwise.cpp — 원소별 add/mult
- op=0 add, op=1 mult (elementwise.cpp:8). 두 입력 스트림을 동시에 읽어 패킷 단위 `kPackets`-way 병렬 연산 후 출력 (elementwise.cpp:22~80). residual add, RMSNorm weight 곱 등에 활용.

## 4. 데이터플로우 / 실행 흐름
- **공통 패턴**: 모든 커널이 `load_input → compute → store_result`의 3~4단 `#pragma HLS dataflow` 구조. 단계 간 `hls::stream`(depth 16~32) FIFO로 연결되어 메모리 read/연산/write가 시간적으로 중첩(오버랩) 실행됨.
- **병렬화**: 패킷 내부 `kPackets`개 원소를 `#pragma HLS UNROLL`로 동시 처리(SIMD형), 외부 루프는 `#pragma HLS PIPELINE`으로 II=1 목표. 즉 "벡터화(패킷)×파이프라이닝" 2축 병렬.
- **메모리계층**: 단일 레벨 — HBM/DDR(m_axi) ↔ on-chip stream FIFO ↔ 연산 레지스터. on-chip 가중치 캐싱·타일링은 없음(가중치를 매 호출 스트리밍).
- **양자화/데이터타입**: 빌드타임 매크로로 FP4~FP32, FIXED8/16 선택. 연산 정밀도(`AccT`)와 전송 정밀도(`DataT`)가 동일(별도 누산 승격 없음) — 저정밀에서 누산 오차 누적 가능.

## 5. HW/SW 매핑
- 본 저장소는 **HLS 단일 계층**(C++ → RTL via Vitis HLS). Python/RTL 수작업 대응물은 없음. 단, 동일 그룹의 `HLSTransformation/llama_xrt`(LLaMA2 XRT 호스트)와 결합해 쓰는 연산자 풀로 보임(추정).
- SW(host) ↔ kernel 인터페이스: m_axi(데이터) + s_axilite(스칼라 인자/제어). XRT 호스트가 버퍼를 GMEM에 두고 커널 enqueue하는 표준 Vitis 흐름.

## 6. 빌드·실행
- 커널별 `vitis_hls -f <kernel>.tcl` (README:9~12 등). 환경변수로 DATATYPE/BUS/행렬크기/PART/IMPLEXP 지정.
- 예: `DATATYPE=FIXED16 BUS=512 B_COLS=4096 C_COLS=4096 vitis_hls -f matmul/matmul.tcl`.

## 7. 의존성
- Xilinx Vitis HLS (`hls_math.h`, `hls_stream.h`, `ap_int.h`, `ap_fixed.h`). 외부 ML 프레임워크 의존 없음. C++ 표준 `<ratio>`(LUT 컴파일타임 구간).

## 8. 강점 / 한계 / 리스크
- **강점**:
  - 정밀도·버스폭·크기 전면 파라미터화 → DSE/플랫폼 이식성 우수. 단일 소스로 float/fixed 동시 지원(GET_NUMBER 매크로 패턴).
  - 패킷화(SIMD)+dataflow+pipeline의 깔끔한 HLS 관용구 → 학습/재사용 표본으로 매우 좋음.
  - 도메인 한정 LUT exp(템플릿) — 비선형 함수 근사 재사용 자산.
- **한계/리스크**:
  - softmax에 max-subtraction 부재 → 저정밀 수치 불안정(README도 float 권장).
  - matmul `kARows=2` 고정, 타일링/가중치 재사용 없음 → 대형 GEMM 처리량·대역폭 효율 미검증. 실제 LLM 통합 데이터플로우(레이어 융합)는 없음.
  - 커널이 개별 분리 → 레이어 간 on-chip 데이터 유지(fusion)가 없어 메모리 왕복 큼.

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기, HG-PIPE 계열) 관점 시사점
- **즉시 재사용 가능**:
  - `config.h`의 버스워드 패킷화 + `GET_NUMBER/GET_RAW` 매크로 패턴 → 우리 가속기의 다정밀(INT8/FP16) SIMD 데이터패스 추상화 템플릿으로 차용.
  - `exponential-lut.hpp`의 컴파일타임 도메인 한정 LUT + 선형보간 → ViT의 **Softmax exp**, **GELU** 근사 유닛에 직접 응용(포인트수·구간을 면적예산에 맞춰 튜닝).
  - dataflow `load→compute→store` 3단 골격 → ViT 각 서브연산(LayerNorm, MHSA score, MLP) 커널의 표준 골격으로 활용.
- **개선해 가져갈 점**:
  - softmax는 online-softmax(running max/sum, FlashAttention식)로 교체해 단일 패스 + 수치안정 확보(입력 재전송 2회 제거).
  - matmul은 출력 스테이셔너리 + 타일링 + 가중치 on-chip 재사용으로 확장해야 HG-PIPE류 고처리량 달성. 현재 형태는 연산자 PoC 수준.
- **시사**: 본 repo는 "연산자 단위 라이브러리" 접근의 좋은 레퍼런스이나, 고처리량 파이프라인(레이어 융합·상주 가중치)은 우리가 추가 설계해야 함.

## 10. 근거/한계 표기
- 코드 라인 근거: matmul.cpp / softmax.cpp / rmsnorm.cpp / unary.cpp / elementwise.cpp / config.h / exponential-lut.hpp 전문 직접 확인.
- "원 논문" 및 동일 그룹의 llama_xrt 연계 여부는 README에 명시 없음 → **추정**.
- 실제 합성 결과(LUT/DSP/지연/주파수)는 저장소에 리포트 부재 → **확인 불가**.
