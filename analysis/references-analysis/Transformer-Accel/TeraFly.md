# TeraFly 정밀 분석

> 대상 경로: `REF/Transformer-Accel/TeraFly`
> 원본 저장소: `https://github.com/zjnyly/TeraFly` (확인 — `.git/config:9`, `.git/logs/HEAD:1`)
> 분석 방식: bash 미사용, Glob/Grep/Read만 사용. 라인 근거는 `파일명:라인범위`로 표기.

---

## 0. 핵심 경고 — 이 체크아웃에는 가속기 "소스코드"가 없다

이 분석의 가장 중요한 발견을 먼저 명시한다. **현재 디렉토리에는 HLS 커널(`loopLynx.cpp`), 헤더(`loopLynx.h`, `params.h`), 호스트(`host.cpp`), 코드 생성기(Python), tcl 스크립트가 단 하나도 존재하지 않는다.**

- Glob `**/*.cpp`, `**/*.h`, `**/*.py`, `**/*.tcl`, `**/*.v`, `**/*.sv` → 전부 **0건** (확인 — Glob 결과 No files found).
- 존재하는 것은: **빌드 인프라**(Makefile/`.mk`/`.cfg`/`.ini`), **사전 빌드 바이너리**(`loopLynx` ELF 실행파일, `tokenizer` ELF 실행파일), **대량 입력 데이터**(`tokenizer/inputs_hardware_full/input_*.bin` 수천 개), 그리고 **빌드/크래시 로그**뿐이다.
- `.gitignore`(`gitignore:1-4`)가 `OPT-1.3b/`, `GPT-2-two-node.zip`, `DATE_experiments/`를 추적 제외 → 실제 소스/생성물이 의도적으로 커밋에서 빠졌을 가능성이 높다(추정).

따라서 본 문서의 "핵심 모듈 정밀 분석"은 **소스 라인 단위 분석이 불가능**하며, 대신 **빌드 시스템·연결 토폴로지·바이너리 메타데이터에서 역으로 추론한 아키텍처**를 라인 근거와 함께 제시한다. 소스 기반 알고리즘 분석(attention/FFN/matmul/양자화 커널 내부)은 **확인 불가**로 명확히 표기한다.

---

## 1. 개요

- **TeraFly**는 AMD/Xilinx **Alveo HBM FPGA(U50/U280 계열)** 상에서 **OPT-1.3B 규모 디코더-온리 LLM**을 추론하는 가속기다(추정 — 디렉토리명 `OPT-1.3b_optimize`, `gitignore:2-3`의 `GPT-2-two-node.zip`/`OPT-1.3b/`, 토크나이저 존재로 LLM 디코더 추론 파이프라인 확인).
- 가속기 커널의 이름은 **`loopLynx`**이며, 이것이 HLS 합성 대상 커널이자 최종 호스트 실행파일 이름이다(확인 — `makefile_us_alveo.mk:74,91-104`).
- 설계는 **2개 SLR(Super Logic Region)에 동일 커널을 복제 배치하고 스트림으로 링 연결**하는 멀티-다이/멀티-노드 구조다(확인 — `connectivity.cfg:3-8`). 이는 단일 FPGA 내 칩렛 간(SLR0↔SLR1) 파이프라인 또는 다중 FPGA 노드 확장을 위한 패턴이다.
- **HeteroCL/Allo 계열 코드 생성 흐름**으로 만들어졌다(확인 — Makefile 경로 `.../transformer/heterocl_file/...`가 `Makefile:19`, `utils.mk:20`에 하드코딩; `utils.mk:113`이 `$(XF_PROJ_ROOT)/allo/harness/readme_gen/readme_gen.py` 호출). HeteroCL/Allo는 Cornell의 Python→HLS DSL이다.
- `template/` 디렉토리는 **플랫폼/주파수/커널 개수에 따라 Makefile·connectivity.cfg를 자동 생성**하는 템플릿(placeholder 치환)이다(확인 — `template/Makefile:26`의 `{PLATFORM}`, `template/makefile_us_alveo.mk:45`의 `{FREQ}`, `:91`의 `{XO_REGION}`, `:93`의 `{XO_CALL}`).
- 빌드 환경은 **Vitis HLS 2023.2**, 합성 백엔드는 **LLVM-10**(확인 — `hs_err_pid14993.log:6,9,21`). 즉 LLM-from-Python→HLS 합성 중 HLS 프론트엔드(`libLLVM-10.so`)에서 SIGSEGV 크래시가 기록되어 있다.

> 한 줄 요약: TeraFly는 HeteroCL/Allo로 자동 생성되어 Alveo HBM FPGA 위에서 멀티-SLR 링으로 동작하는 OPT-1.3B LLM 가속기지만, **이 체크아웃은 소스 없이 빌드 인프라+바이너리+데이터만** 담고 있다.

---

## 2. 디렉토리 구조

### 2.1 자체 분석 대상 (실제 Read한 파일)

```
TeraFly/
├── LICENSE                         # Apache-2.0 (LICENSE:1-2)
├── .gitignore                      # OPT-1.3b/, GPT-2-two-node.zip, DATE_experiments/ 제외 (gitignore:1-4)
├── hs_err_pid14993.log             # Vitis HLS(LLVM-10) 크래시 덤프 (hs_err...:6,9,21)
├── OPT-1.3b_optimize/              # OPT-1.3B용 빌드 인스턴스
│   ├── Makefile                    # 플랫폼→.mk 분기 (Makefile:26-53)
│   ├── makefile_us_alveo.mk        # 핵심 빌드 규칙 (loopLynx_0/_1 .xo, link, package)
│   ├── utils.mk                    # 환경 체크/XSA/readme_gen (utils.mk:113)
│   ├── connectivity.cfg            # SLR 배치 + HBM 매핑 + 스트림 링 (connectivity.cfg:3-29)
│   ├── xrt.ini                     # XRT 프로파일링 설정 (xrt.ini:5-6)
│   ├── xrt.run_summary             # 실행 프로파일 요약(JSON, 31줄) (run_summary:8)
│   └── tokenizer/
│       ├── tokenizer               # ELF 바이너리 (소스 없음)
│       └── inputs_hardware_full/   # input_*.bin 수천 개 (데이터, 아래 제외)
└── template/                       # 빌드 파일 자동 생성용 템플릿
    ├── Makefile                    # {PLATFORM} placeholder (template/Makefile:26)
    ├── makefile_us_alveo.mk        # {FREQ}/{XO_REGION}/{XO_CALL} placeholder
    ├── utils.mk                    # OPT판과 동일
    └── connectivity.cfg            # 2-SLR 토폴로지 원형 (template/connectivity.cfg)
```

### 2.2 제외 (vendor / 생성물 / 데이터 / 바이너리 — 이름만 언급)

- `.git/` 전체 (버전관리 메타, pack 파일)
- `OPT-1.3b_optimize/loopLynx` — **사전 빌드된 호스트 ELF 실행파일** (소스 없음, 바이너리)
- `OPT-1.3b_optimize/tokenizer/tokenizer` — **ELF 바이너리 토크나이저** (확인 — `tokenizer:1` 매직넘버 `ELF`)
- `OPT-1.3b_optimize/tokenizer/inputs_hardware_full/input_*.bin` — **테스트 입력 토큰 시퀀스 수천 개** (데이터, 분석 비대상)
- `OPT-1.3b_optimize/xrt.run_summary` 내부가 참조하는 `device_trace_*.csv` 등 — 프로파일 산출물
- `.gitignore`로 제외된 미존재 항목: `OPT-1.3b/`, `GPT-2-two-node.zip`, `DATE_experiments/`

---

## 3. 핵심 모듈 정밀 분석 (라인 근거)

> 소스 부재로 "함수/HLS 커널 내부" 분석은 불가하므로, **빌드 그래프·연결 토폴로지·메타데이터**에서 아키텍처를 정밀 재구성한다. 각 항목 끝에 [확인]/[추정]/[확인 불가]를 명시.

### 3.1 빌드 진입점과 플랫폼 분기 로직 — `Makefile`

`Makefile`은 플랫폼 정보를 조회해 적절한 하위 `.mk`로 분기하는 디스패처다.

- `Makefile:26` `PLATFORM ?= xilinx_u50lv_gen3x4_xdma_2_202010_1` → **기본 타깃 보드가 Alveo U50LV** (확인). 이는 HBM 8GB·2-SLR 보드로, `connectivity.cfg`의 2-SLR 구조와 정합.
- `Makefile:28-29` `platforminfo`로 `FPGA Family`/`CPU Type` 추출 → `DEV_ARCH`/`CPU_TYPE` 결정 (확인).
- `Makefile:31-37` CPU 타입으로 `HOST_ARCH`(aarch32/aarch64/x86) 결정 (확인).
- `Makefile:39-53` `DEV_ARCH`별 분기: `zynquplus`→zynqmp, `versal`→versal, 그 외→`makefile_us_alveo.mk` (확인). 기본 경로는 **US+/Alveo**.
- `Makefile:19` `COMMON_REPO`가 경로 `/home/jz2292/project/transformer/heterocl_file/*`를 기준으로 잘라냄 → **원개발 환경이 HeteroCL 기반 트랜스포머 프로젝트**였음을 드러냄 (확인, 추론 근거).

### 3.2 합성·링크·패키징 규칙 — `makefile_us_alveo.mk` (가장 중요)

이 파일이 TeraFly 가속기 빌드의 실질적 핵심이다.

- `makefile_us_alveo.mk:44` `TARGET := hw` → 기본 합성 타깃이 **실 하드웨어**(sw_emu/hw_emu 아님) (확인).
- `makefile_us_alveo.mk:45` `VPP_LDFLAGS := --kernel_frequency 250 --optimize 3 --config ./connectivity.cfg` → **목표 커널 주파수 250 MHz**, v++ 최적화 레벨 3, 연결은 connectivity.cfg로 (확인). 250 MHz는 HBM Alveo 설계의 전형적 목표 주파수.
- `makefile_us_alveo.mk:56` 호스트 컴파일에 `-I$(XILINX_HLS)/include/ -O0 -g -std=c++1y` → HLS 헤더(ap_int 등) 포함, C++14 (확인).
- `makefile_us_alveo.mk:57` `LDFLAGS += -lOpenCL -lxilinxopencl -lxrt_core -lxrt_coreutil` → **호스트는 OpenCL/XRT API**로 FPGA 제어 (확인).
- `makefile_us_alveo.mk:64` `HOST_SRCS += host.cpp` → **호스트 소스는 단일 `host.cpp`** (확인). 단, 이 체크아웃에는 부재(확인 불가).
- **커널 컴파일 규칙 (가장 결정적):**
  - `makefile_us_alveo.mk:91-93`
    ```
    $(TEMP_DIR)/loopLynx_0.xo: loopLynx.cpp loopLynx.h params.h
        v++ -c ... -k loopLynx_0 ... -o'$@' $^
    ```
  - `makefile_us_alveo.mk:95-97` 동일 패턴으로 `loopLynx_1.xo`, `-k loopLynx_1`.
  - → **단일 HLS 소스 `loopLynx.cpp`에서 `loopLynx_0`/`loopLynx_1` 두 커널을 컴파일**한다. 의존 헤더는 `loopLynx.h`, `params.h`(=하이퍼파라미터/타일 크기 정의로 추정) (확인 — 의존 목록; 헤더 내용은 확인 불가).
- `makefile_us_alveo.mk:101-104` 링크 단계:
  - `:103` `v++ -l ... $(VPP_LDFLAGS) ... -o loopLynx.link.xclbin <두 .xo>` → 두 커널을 하나의 xclbin으로 링크 (확인).
  - `:104` `v++ -p ... -o loopLynx.xclbin` → 패키징 (확인).
- `makefile_us_alveo.mk:108-109` 호스트: `g++ -o loopLynx host.cpp ...` → 호스트 실행파일도 이름이 `loopLynx` (확인).
- `makefile_us_alveo.mk:116-122` `run` 타깃: hw일 때 `./loopLynx <xclbin>` 직접 실행, emu일 때 `XCL_EMULATION_MODE` 설정 후 실행 (확인).
- **결론:** TeraFly의 HW 부분은 **단일 HLS 커널 템플릿(loopLynx)을 N개로 복제**(여기선 0/1 두 개)하여 멀티-SLR/멀티-노드로 확장하는 구조. 커널 내부 연산(attention/FFN/matmul/quant)은 소스 부재로 **확인 불가**.

### 3.3 하드웨어 연결 토폴로지 — `connectivity.cfg` (HW 매핑의 직접 증거)

`connectivity.cfg`는 v++ 링커가 커널 인스턴스를 물리 자원에 배치하는 지시문으로, **TeraFly의 데이터플로우 토폴로지를 직접 노출**한다.

- `connectivity.cfg:3-4` SLR 배치:
  - `slr=loopLynx_0_1:SLR0`, `slr=loopLynx_1_1:SLR1` → **커널 인스턴스 0은 SLR0, 인스턴스 1은 SLR1에 고정 배치** (확인). 다이 간 균등 분할.
- `connectivity.cfg:7-8` 스트림 링 연결:
  - `stream_connect=loopLynx_0_1.stream_next:loopLynx_1_1.stream_previous:64`
  - `stream_connect=loopLynx_1_1.stream_next:loopLynx_0_1.stream_previous:64`
  - → **두 커널이 64-depth AXI-Stream FIFO로 양방향 링을 형성** (확인). `stream_next`/`stream_previous` 포트명은 **선형/링형 파이프라인 토폴로지**(노드 i → 노드 i+1)를 의미하며, OPT의 레이어들을 두 다이에 분산해 토큰/액티베이션을 스트리밍한다는 강한 증거(추정).
- `connectivity.cfg:10` `sp=loopLynx_0_1.host_addr:PLRAM[0]` → 호스트↔커널 제어/입력 버퍼는 **PLRAM** (확인).
- `connectivity.cfg:11-18` `w_addr_0..7 → HBM[0..7]` → **인스턴스0의 가중치 8개 포트가 HBM 채널 0~7에 매핑** (확인). 가중치 멀티-뱅킹으로 HBM 대역폭 병렬 활용.
- `connectivity.cfg:21-29` 인스턴스1은 `host_addr→PLRAM[2]`, `w_addr_0..7→HBM[16..23]` → **다른 HBM 그룹** 사용. 주석 `:20`이 "PLRAM[1]은 SLR0에 있어 주파수 저하 유발"이라 명시 → **물리 배치-타이밍 최적화가 수동 튜닝**되었음 (확인). 이는 설계자가 SLR-aware 자원 배치로 250 MHz를 맞추려 했다는 직접 증거.
- `template/connectivity.cfg`와의 차이:
  - 템플릿(`template/connectivity.cfg:9-23`)은 인스턴스당 `w_addr_0..5`(6 포트), HBM[0..5]/HBM[8..13] 매핑이고, 스트림 링도 `loopLynx_0_1`이 자기 자신과 연결되는 단일-노드 형태(`template/connectivity.cfg:5`).
  - OPT 인스턴스(`connectivity.cfg`)는 **w_addr 포트가 8개로 확장**되고 HBM[0..7]/HBM[16..23]으로 더 멀리 떨어진 채널을 사용 → **OPT-1.3B에 맞춰 가중치 대역폭/SLR 배치를 재튜닝**한 것 (확인, 두 파일 비교).
  - 또한 OPT판은 `[profile]`의 `exec=` 라인이 주석 처리(`connectivity.cfg:33-34`)되어 프로파일 오버헤드 제거; 템플릿은 활성(`template/connectivity.cfg:26-27`) (확인).

### 3.4 빌드 파일 자동 생성 시스템 — `template/` (소프트웨어 측 "템플릿 생성 로직")

작업 지시의 "템플릿 생성 로직"에 해당하는 부분이 바로 이 디렉토리다. 단, **생성기 스크립트(.py) 자체는 부재**하고 **생성 대상 템플릿만** 존재한다.

- `template/Makefile:26` `PLATFORM ?= {PLATFORM}` → 플레이스홀더 `{PLATFORM}`을 실제 보드명으로 치환 (확인). OPT판에서는 `xilinx_u50lv...`로 치환됨(`Makefile:26`).
- `template/makefile_us_alveo.mk:45` `--kernel_frequency {FREQ}` → 목표 주파수도 파라미터화. OPT판은 `250`으로 치환됨(`makefile_us_alveo.mk:45`).
- `template/makefile_us_alveo.mk:91` `{XO_REGION}` → **커널 .xo 빌드 규칙 블록을 통째로 생성** (확인). OPT판은 이 자리에 `loopLynx_0.xo`/`loopLynx_1.xo` 두 규칙(`makefile_us_alveo.mk:91-97`)이 들어감.
- `template/makefile_us_alveo.mk:93` `$(BUILD_DIR)/loopLynx.xclbin: {XO_CALL}` → **링크 의존성 목록(.xo 나열)도 생성** (확인).
- **해석:** 외부 코드 생성기가 (a) 목표 보드/주파수, (b) 노드(커널 복제) 개수를 받아 `{PLATFORM}`/`{FREQ}`/`{XO_REGION}`/`{XO_CALL}`를 채워 빌드 파일을 찍어낸다. 즉 **"노드 수에 따라 Makefile·connectivity를 스케일아웃"하는 자동화**가 TeraFly 설계의 핵심 SW 자동화 포인트(추정 — 생성기 소스는 확인 불가, 그러나 placeholder 세트가 명확한 증거).
- `utils.mk:112-113` `README.rst: description.json` → `$(XF_PROJ_ROOT)/allo/harness/readme_gen/readme_gen.py` 호출 → **Allo 프레임워크의 하네스에 의존**, 즉 TeraFly는 **Allo(=HeteroCL 후속) 생태계의 산출물**임을 재확인 (확인).

### 3.5 토크나이저 및 입력 데이터 파이프라인

- `tokenizer/tokenizer`는 ELF 실행파일(확인 — `tokenizer:1`의 `ELF` 매직). 소스 없음.
- `tokenizer/inputs_hardware_full/input_*.bin` 수천 개 → 사전 토큰화된 입력 시퀀스(이진), 하드웨어 추론 시 호스트가 PLRAM/HBM으로 적재할 데이터로 추정. 파일당 4자리 인덱스(`input_0075.bin`~`input_4811.bin`) → **약 5천 개 규모의 추론 입력 코퍼스** (확인 — Glob 결과 패턴).

### 3.6 실행 환경 메타데이터 — `xrt.ini` / `xrt.run_summary` / `hs_err`

- `xrt.ini:5-6` `opencl_trace=true`, `device_trace=fine` → 세밀한 디바이스 트레이싱 활성 (확인).
- `xrt.run_summary:8` 생성 경로 `/home/zjnyly/codegen-uram/OPT-1.3b-x2-512bit/OPT-1.3b_optimize/...` → **원개발 디렉토리명이 "codegen-uram", "OPT-1.3b-x2-512bit"** (확인). 시사점: (a) **codegen**=자동 코드 생성, (b) **uram**=URAM(UltraRAM) 활용 메모리 설계, (c) **x2**=2-노드/2-커널 복제, (d) **512bit**=512비트 데이터버스 폭. 이는 §3.3의 2-SLR/HBM 멀티뱅크 구조와 정합 (확인+추정).
- `xrt.run_summary:12` `"target": "TT_HW"` → 실 하드웨어 실행 프로파일 (확인).
- `hs_err_pid14993.log:6,21` → **Vitis HLS 2023.2**, JRE Temurin 11; `:9` `libLLVM-10.so.1`에서 SIGSEGV → **HLS C-합성(LLVM-10 프론트엔드) 도중 크래시** (확인). `:23` 빌드 호스트 i9-12900K/Ubuntu 18.04. `:11` 원경로 `/home/zjnyly/repos/TeraFly`.

---

## 4. 데이터 플로우 (재구성)

소스 부재로 커널 내부 연산 순서는 확인 불가. 토폴로지 기반 재구성:

1. **호스트(loopLynx ELF)** 가 OpenCL/XRT로 xclbin 적재, 토큰 입력(`input_*.bin`)을 PLRAM(`host_addr`)에 적재 (확인 — `makefile_us_alveo.mk:57`, `connectivity.cfg:10,21`).
2. **가중치**는 HBM 다중 채널(인스턴스0: HBM[0..7], 인스턴스1: HBM[16..23])에서 8포트 병렬 스트리밍 (확인 — `connectivity.cfg:11-18,22-29`).
3. **레이어 연산**은 `loopLynx_0`(SLR0)와 `loopLynx_1`(SLR1)에 분산. 두 커널은 64-폭 AXI-Stream 링으로 액티베이션을 주고받음 (확인 — `connectivity.cfg:7-8`). OPT 디코더 레이어를 두 다이에 파이프라인 분할하는 형태로 추정.
4. 데이터버스 폭은 512비트(run_summary 경로 `...-512bit`로 추정), 스트림은 64바이트=512비트 정합 (추정).
5. 출력 토큰을 호스트가 회수 → 토크나이저로 디코딩 (추정).

> attention(QKV·softmax)/FFN(GEMM)/양자화(INT8/INT4 등) 연산의 구체적 datapath, 타일링, PE 배열, accumulate 구조는 **소스 부재로 전부 확인 불가**. (`loopLynx.cpp`, `params.h` 필요)

---

## 5. HW/SW 매핑

| 계층 | 구성요소 | 근거 |
|---|---|---|
| SW (호스트) | `host.cpp` → `loopLynx` ELF, OpenCL/XRT로 FPGA 제어, 입력 적재/출력 회수 | `makefile_us_alveo.mk:64,108-109`, 부재(확인 불가) |
| SW (전처리) | `tokenizer` ELF + `input_*.bin` 코퍼스 | Glob 결과, `tokenizer:1` |
| SW (자동화) | template placeholder 치환 생성기(.py 부재) + Allo `readme_gen.py` | `template/*`, `utils.mk:113` |
| HW 커널 | `loopLynx_0`(SLR0), `loopLynx_1`(SLR1) — 동일 소스 복제 | `makefile_us_alveo.mk:91-97`, `connectivity.cfg:3-4` |
| HW 메모리 | 가중치 HBM 멀티뱅크(8포트×2), 제어/IO PLRAM, URAM 온칩(추정) | `connectivity.cfg:10-29`, run_summary 경로 |
| HW 인터커넥트 | 64B AXI-Stream 양방향 링(SLR0↔SLR1) | `connectivity.cfg:7-8` |
| 합성 흐름 | HeteroCL/Allo Python → Vitis HLS 2023.2(LLVM-10) → v++ link/package, 250 MHz | `Makefile:19`, `utils.mk:113`, `hs_err...:6,9`, `makefile_us_alveo.mk:45` |

---

## 6. 빌드·실행

- 빌드: `make all TARGET=hw PLATFORM=xilinx_u50lv_gen3x4_xdma_2_202010_1` (확인 — `Makefile:57-58`, `makefile_us_alveo.mk:79`). 흐름: `loopLynx_{0,1}.xo` 합성 → `loopLynx.link.xclbin` 링크(connectivity 적용) → `loopLynx.xclbin` 패키징 → 호스트 `loopLynx` g++ 컴파일 (`makefile_us_alveo.mk:91-109`).
- 실행: `make run` 또는 `./loopLynx build_dir.hw.<XSA>/loopLynx.xclbin` (확인 — `makefile_us_alveo.mk:55,116-122`).
- **단, 이 체크아웃에서는 `loopLynx.cpp`/`loopLynx.h`/`params.h`/`host.cpp` 부재로 `make`가 즉시 실패한다** (확인 — 소스 Glob 0건). 사전 빌드된 `loopLynx` ELF 바이너리로만 (해당 보드가 있으면) 실행 가능(추정).
- 환경 의존: `XILINX_VITIS`, `XILINX_XRT`, `XILINX_HLS` 필요 (확인 — `utils.mk:33-42`, `makefile_us_alveo.mk:56`).

---

## 7. 의존성

- **Xilinx Vitis 2023.2 / Vitis HLS 2023.2** (확인 — `hs_err...:21`).
- **XRT + OpenCL** 런타임 (확인 — `makefile_us_alveo.mk:57`).
- **Alveo HBM 보드** (기본 U50LV, U280 호환 추정) (확인 — `Makefile:26`).
- **HeteroCL / Allo** 코드생성·하네스 (확인 — `Makefile:19`, `utils.mk:113`).
- LLVM-10 (HLS 내부) (확인 — `hs_err...:9`).
- (부재이나 빌드가 요구) `loopLynx.cpp/h`, `params.h`, `host.cpp`, 코드 생성 스크립트.

---

## 8. 강점·한계

**강점 (근거 기반)**
- **스케일아웃 자동화**: template placeholder(`{PLATFORM}/{FREQ}/{XO_REGION}/{XO_CALL}`)로 보드·주파수·노드 수를 파라미터화 → 멀티-FPGA/멀티-다이로 손쉽게 확장 (확인 — `template/makefile_us_alveo.mk:45,91,93`).
- **HBM 대역폭 최대화**: 가중치를 8 HBM 채널에 멀티뱅킹 (확인 — `connectivity.cfg:11-18`).
- **SLR-aware 물리 최적화**: PLRAM 배치까지 다이 위치를 고려해 주파수 저하 회피, 250 MHz 목표 (확인 — `connectivity.cfg:20`, `makefile_us_alveo.mk:45`).
- **Python→HLS 생산성**: HeteroCL/Allo로 LLM을 HLS로 자동 합성 (확인 — `Makefile:19`).

**한계 (근거 기반)**
- **이 배포본은 재현 불가**: 핵심 소스 전무, 빌드 즉시 실패 (확인 — 소스 Glob 0건).
- **HLS 합성 불안정성 증거**: 동봉된 `hs_err` 크래시 로그 → LLVM-10 프론트엔드 SIGSEGV (확인 — `hs_err...:9`). 대규모 LLM의 HLS 합성이 도구 한계에 부딪힘을 시사.
- **하드코딩 경로 잔존**: `/home/jz2292/...`, `/home/zjnyly/...` 절대경로가 빌드/요약에 남아 이식성 저해 (확인 — `Makefile:19`, `run_summary:8`).
- **알고리즘 분석 불가**: attention/FFN/quant datapath를 검증할 소스 없음 (확인 불가).

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

전제(추정): 본 프로젝트는 "고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적".

1. **멀티-SLR 스트림 링 패턴 차용**: HG-PIPE류 파이프라인을 단일 FPGA의 SLR 경계로 분할할 때, TeraFly의 `stream_connect ...:64` 양방향 링(`connectivity.cfg:7-8`)과 SLR 고정 배치(`:3-4`)는 그대로 참고할 만한 레퍼런스 토폴로지다(고처리량 ViT를 다이 간 분할 시 유용).
2. **HBM 멀티뱅킹 정석**: ViT 가중치/패치 임베딩을 다중 HBM 채널에 분산하는 `w_addr_0..7→HBM[...]` 매핑(`connectivity.cfg:11-18`)은 대역폭 병목 회피의 검증된 패턴.
3. **PLRAM-SLR 배치 함정 회피**: `connectivity.cfg:20`의 "PLRAM이 잘못된 SLR에 있으면 주파수 저하" 교훈은 XR 저지연 설계에서 타이밍 클로저에 직접 도움.
4. **빌드 파일 자동 생성 도입**: template placeholder 방식(`template/`)을 모방하면, 우리 ViT 가속기를 여러 Alveo/Versal 보드·주파수로 빠르게 포팅하는 CI 자동화를 구축할 수 있음.
5. **HLS 합성 리스크 인지**: 대형 모델을 통째 HLS 합성하면 TeraFly처럼 도구 크래시(`hs_err`) 위험. XR 시선추적은 모델이 작으므로 오히려 안전 마진이 크다는 점이 우리 쪽 강점(상대적 시사점).
6. **소스 부재 → 직접 활용 제한**: TeraFly는 "아키텍처 패턴 레퍼런스"로는 가치가 크지만, 코드 재사용 자산으로는 부적합(소스 없음). LLM(OPT) 타깃이라 XR 실시간 추론과는 모델 규모도 다름.

---

## 10. 근거 표기 요약

- **[확인 — 라인]**: §2 디렉토리, §3.1~3.6 빌드/연결/메타데이터 전부 (Makefile/.mk/.cfg/.ini/log 실제 Read).
- **[추정]**: OPT-1.3B 디코더 레이어의 SLR 분할 방식, 512bit 버스, URAM 사용, 입력 코퍼스 용도, template 생성기 동작.
- **[확인 불가]**: HLS 커널 내부(attention/FFN/matmul/softmax/quant) datapath·타일링·PE 구조, `host.cpp` 로직, 코드 생성기(.py) 구현 — **소스 파일이 이 체크아웃에 존재하지 않음**.
