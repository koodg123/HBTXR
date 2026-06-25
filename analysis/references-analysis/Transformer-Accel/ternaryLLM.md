# ternaryLLM 코드베이스 정밀 분석

> 대상: `REF/Transformer-Accel/ternaryLLM`
> 분석일: 2026-06-20
> 근거 표기 규칙: 라인 근거가 있는 사실은 `파일명:라인` 으로 표기, 코드로 확인 불가한 부분은 "추정" / "확인 불가" 로 명시.

---

## 1. 개요

**ternaryLLM** 은 1.58-bit Ternary LLM(BitNet 계열, 가중치 ∈ {-1, 0, +1}) 의 핵심 연산인 **Ternary Sparse GEMM/GEMV** 를 CPU / GPU / FPGA 세 플랫폼에서 가속하는 연구용 코드베이스다. 세 서브프로젝트(`ternaryLLM_CPU`, `ternaryLLM_GPU`, `ternaryLLM_FPGA`)로 구성된다.

핵심 알고리즘 아이디어(전 플랫폼 공통, 라인 근거 `InputAndCheck.scala:36-90`, `SIMD_Generator.ipynb`):

- 삼진 가중치를 **값으로 저장하지 않고 인덱스로 저장**한다. 0 은 아예 생략(sparse), +1 가중치는 해당 활성값(activation)의 인덱스를 "positive index" 리스트에, -1 가중치는 "negative index" 리스트에 넣는다.
- 따라서 곱셈이 사라지고, GEMM 의 내적은 **활성값 gather → (양의 항 합) − (음의 항 합)** 의 가산/감산만으로 계산된다. 즉 `Y += x[pos] - x[neg]` (`InputAndCheck.scala:80-84`).
- 이 포맷을 코드는 **TCSC (Ternary Compressed Sparse Column)** 라 부른다(`ternaryLLM.vcxproj:32`, 노트북 함수명 `..._TCSC_...`).

이 "인덱스 기반 gather + 부호별 누산" 방식이 CPU(SIMD gather), FPGA(ActivationBuffer gather + TernaryGEMM 감산 트리) 양쪽에 동일하게 구현되어 있다는 점이 이 repo의 정체성이다.

**중요 주의 (완전성):** 이 체크아웃은 *부분 체크아웃* 이다. 빌드 메타데이터/시뮬레이션/글루 로직은 존재하지만, **실제 설계 소스의 상당수가 누락**되어 있다(아래 §2, §3 에서 라인 근거로 명시). 구체적으로:
- FPGA: `gemmacc.src.*` 패키지(설계 본체: `ConfigSys`, `TopLevel`, `TernaryGEMM`, `ActivationBuffer`, `DataFSM_Base` 등)가 `import` 로만 참조되고 실제 `.scala` 파일은 존재하지 않음(확인: `Glob gemmacc/src/**` → "No files found", `Grep "package gemmacc.src"` → 매치 0).
- CPU: `ternaryLLM_CPU/src/*.cpp/.hpp` 가 `.vcxproj` 에 등록만 되어 있고 실제 파일 없음(확인: `Glob ternaryLLM_CPU/src/**` → "No files found").

따라서 §3 의 설계 모듈 분석은 **시뮬레이션 코드 / 합성 산출물(svh) / 골든 모델로부터 역추적한 인터페이스·동작** 에 기반한다(근거 라인 명시). 모듈 *내부 RTL* 본문은 본 체크아웃에서 확인 불가.

---

## 2. 디렉토리 구조

### 2-1. 자체 소스 (분석 대상)

```
ternaryLLM/
├─ LICENSE, .gitignore
├─ ternaryLLM_CPU/                         ← C++ (libtorch + Eigen + AVX/OpenMP) CPU 레퍼런스/가속
│   ├─ SIMD_Generator.ipynb               ← AVX2/AVX-512 TCSC 커널 코드 생성기 (Python)
│   ├─ ternaryLLM.sln / .vcxproj / .filters / .vcxproj.user   ← MSVC 프로젝트
│   └─ (src/ : .vcxproj 가 참조하나 본 체크아웃에 없음 — 확인 불가)
│        GEMM_CPU_FP32.{cpp,hpp}, GEMM_CPU_INT8.{cpp,hpp},
│        initData.{cpp,hpp}, LlamaModel.hpp, TCSC.hpp, main.cpp
├─ ternaryLLM_GPU/
│   └─ Benchmarks.ipynb                    ← GPU 벤치마크 노트북 (CUDA/torch, 추정)
└─ ternaryLLM_FPGA/                        ← SpinalHDL(Scala) RTL, Coyote v2 shell 타겟
    ├─ build.sbt / build.sc / project/plugins.sbt   ← sbt/mill 빌드
    ├─ hw/spinal/                          ← Scala 소스 루트 (build.sbt:13)
    │   ├─ MatrixAdd/                      ← Coyote v2 동작 검증용 최소 예제(X+W=Y)
    │   │   ├─ logic.scala                 ← FSM 기반 read-add-write 로직 (실존)
    │   │   └─ Wrap_logic.scala            ← AXI-Lite 제어 레지스터 래퍼 (실존)
    │   └─ gemmacc/
    │       ├─ coyote/
    │       │   ├─ Types.scala             ← Coyote v2 디스크립터/AXI4SR 타입 (실존)
    │       │   └─ AxiCoyote.scala         ← Coyote SQ/CQ ↔ 표준 AXI4 변환기 (실존)
    │       ├─ util/
    │       │   ├─ InputAndCheck.scala     ← 입력 생성 + naive ternary GEMM 골든모델 (실존, 핵심)
    │       │   └─ AxiMemorySim.scala      ← 시뮬레이션용 AXI 메모리 모델 (실존)
    │       ├─ sim/                        ← 시뮬레이션 (설계 모듈 인터페이스 역추적 근거)
    │       │   ├─ ternaryGEMM/TopLevelSim.scala
    │       │   ├─ testLogic/logic_sim.scala
    │       │   └─ old/Base/ , old/Op/ , old/Op/coyote_v2/   ← 구버전 sim들
    │       └─ src/                        ← 설계 본체. import 로 참조되나 파일 부재 (확인 불가)
    ├─ hw/vhdl/version_build/vfpga_top.svh ← 합성된 top 인스턴스화 (DEBUG/ILA 포함, 실존)
    ├─ hw/vhdl/vfpga_top.svh               ← 합성된 top 인스턴스화 (clean, 실존)
    └─ coyote_files/hw/vfpga_top.svh       ← Coyote vFPGA top (clean, 실존)
```

### 2-2. 제외 대상 (생성물 / vendor — 이름만 언급)

- `ternaryLLM_FPGA/coyote_files/hw/cyt_top.bit` — **합성 비트스트림(생성물)**. 분석 제외.
- `ternaryLLM_FPGA/Results/Benchmarking_TernaryGEMM.odt`, `Results/Results.ods` — 결과 문서/스프레드시트(바이너리). 내용 분석 제외(텍스트 추출 불가).
- `ternaryLLM_FPGA/png/Thesis_Figures-TopLevel.drawio__2_.png`, `png/.gitkeep` — 도면 이미지. 제외.
- `ternaryLLM_FPGA/.gitignore`, `hw/vhdl/.gitignore`, `hw/verilog/.gitignore` (verilog 디렉토리는 `.gitignore` 만 존재 = 빈 산출물 폴더), `png/.gitkeep` — placeholder.
- `.vcxproj.user` — 로컬 사용자 IDE 설정(생성물 성격). 제외.
- `.gitignore` 가 참조하는 `PyTorch_Extension/`, `torch_gpu/`, `x64/` 등 — 본 체크아웃에 부재(`.gitignore:6-14`). 외부 의존 vendor: **libtorch**, **Eigen 3.4.0** (`ternaryLLM.vcxproj:129,149` 의 include 경로).

---

## 3. 핵심 모듈 정밀 분석

### 3-A. Ternary Sparse GEMM 알고리즘 — 골든 모델 (가장 중요)

**파일:** `hw/spinal/gemmacc/util/InputAndCheck.scala`

이 파일의 `naiveGEMM` 이 전 플랫폼 공통 알고리즘의 *명세(spec)* 다. 하드웨어/SIMD 구현이 모두 이것과 bit-exact 하게 검증된다(`TopLevelSim.scala:93,108`, `TopLevelBaseSim.scala:65,80`).

- **가중치 표현 (TCSC):** `W_matrix(base)` 한 행은 한 출력 컬럼 그룹의 비제로 가중치 엔트리들이며, 각 엔트리는 짝수 위치 = positive index(`pos`), 홀수 위치 = negative index(`neg`) 로 구성된다 — `InputAndCheck.scala:76-77`:
  ```scala
  val pos = W_matrix(base)(2 * col_offset).toInt
  val neg = W_matrix(base)(2 * col_offset + 1).toInt
  ```
- **연산 (곱셈 없음):** 활성 X 에서 pos/neg 인덱스로 값을 gather 한 뒤 차를 누산 — `InputAndCheck.scala:80-84`:
  ```scala
  val x_pos = X_matrix(m)(pos + slice_weights * 256)
  val x_neg = X_matrix(m)(neg + slice_weights * 256)
  Y(m)(n) += (x_pos - x_neg)
  ```
- **타일링/슬라이싱:** `S_2 = S/2` (`:38`), 출력 N 을 `S/2` 폭의 컬럼 그룹으로 나누고(`col_group = n/(S_2)`, `col_offset = n%(S_2)` — `:65-66`), K 차원은 `K_slice`(=128) 단위로 슬라이스(`entries_per_K_slice = entries/(K/128)` — `:40`, `slice_weights * 256` 오프셋 — `:80-81`). 이는 활성 버퍼가 한 번에 K_slice=128 폭 만큼만 들고 있는 온칩 타일링과 정합한다(아래 3-D).
- **입력 생성기:** `generateX` (`:14-21`) 는 M×K 활성을 byte 로(짝수열 5, 홀수열 1), `generateW` (`:24-32`) 는 `times = N/(S/2)` 그룹 × `entries` × `S` 형태의 인덱스 텐서를 생성. → 데이터 레이아웃 디버깅용 결정론적 패턴.

> 시사점: 출력 컬럼당 가산 횟수 = entries(비제로 수)에 비례 → **연산량이 sparsity 에 선형**. dense MAC 대비 곱셈기 완전 제거.

### 3-B. TernaryGEMM PE (감산-누산 코어)

**근거 파일:** `sim/old/Base/TernaryGEMMSim.scala`, `sim/old/Base/ActivationToTernaryGEMMSim.scala` (설계 본체 `gemmacc.src.Base.TernaryGEMM` 는 부재 — 인터페이스/동작만 역추적).

- **생성 파라미터:** `new TernaryGEMM(S, bitwidth)` — `TernaryGEMMSim.scala:20`. S 는 짝수(`:11` 주석 "Must be even"), bitwidth 는 누산 폭(테스트 32 또는 16).
- **IO:**
  - 입력 `io.buffer(0..S-1)` : 폭 S 의 Stream 벡터(valid/payload) — `TernaryGEMMSim.scala:35-37`.
  - 출력 `io.output(0..S/2-1)` : S/2 개 — `:94-99`.
  - 제어 `io.start` (1-cycle 펄스로 누산 리셋, `:29-31`), (주석 처리된 `accumulate`, `out_valid` 핀 존재 — 다중 라운드 누산 의도, `:45,57,84`).
- **연산 정의(가장 중요):** 출력 i = 짝수 입력 − 홀수 입력 — `TernaryGEMMSim.scala:90`:
  ```scala
  expectedValue(i) = testValues(i * 2) - testValues(2 * i + 1)
  ```
  즉 PE 는 S 개 gather 된 활성을 (pos, neg) 쌍으로 받아 **S/2 개의 차(差)** 를 만들고, 라운드를 거쳐 누산한다. 골든 모델의 `x_pos - x_neg` (3-A) 의 하드웨어 직접 사상.
- **다중 라운드 누산:** 주석된 second-round 코드(`:58-90`)는 `(in0+plus) - (in1+minus)` 형태의 누산 검증을 보여줌 → 내부에 라운드 간 accumulator 레지스터가 있음을 시사(확인: accumulator 디버그 핀 `dut.accDebug(i)` 참조 `:54`).

> 정리: TernaryGEMM = "삼진 MAC PE" 의 역할이지만 실제로는 곱셈기 0개, **S/2 개의 부호반전 가산기 + 누산 레지스터** 로 구성된 감산-트리. (RTL 본문 확인 불가, 동작은 sim 으로 확정.)

### 3-C. acc 핀 — 합성 산출물에서의 PE 폭 단서

**파일:** `hw/vhdl/version_build/vfpga_top.svh`

합성된 top 은 모듈명 `ternaryGEMMOP` 로 인스턴스화되며(`vfpga_top.svh:11` `inst_ternary_GEMM`), 디버그용 누산기 핀이 노출된다 — `:3-6, 96-99`:
```systemverilog
logic [15:0] acc_32;  logic [15:0] acc_33;
logic [15:0] acc_4;   logic [15:0] acc_5;
...
.io_acc_4(acc_4), .io_acc_5(acc_5), .io_acc_32(acc_32), .io_acc_33(acc_33)
```
- acc 폭 16-bit → 누산기/출력 데이터 폭 16-bit(`signedValue` 16-bit 해석과 일치, `TopLevelSim.scala:104-107`).
- acc 인덱스가 4,5,32,33 (연속 쌍) → 누산기 배열이 최소 수십 개(≥34) 존재 = PE 병렬도/출력 폭 단서(추정: S 또는 S/2 규모). ILA 프로브로도 동일 4개 노출(`:135-138`).

### 3-D. ActivationBuffer (sparse gather 유닛)

**근거 파일:** `sim/old/Base/ActivationBufferSim.scala`, `sim/old/Base/ActivationToTernaryGEMMSim.scala` (본체 `gemmacc.src.Base.ActivationBuffer` 부재).

- **생성 파라미터:** `new ActivationBuffer(S, K_slice, idx_bitwidth, bitwidth)` — `ActivationBufferSim.scala:17`. 예: S=64, K_slice=128, index_bitwidth=16 (`ActivationToTernaryGEMMSim.scala:28-31`).
- **IO:**
  - `io.x(0..K_slice-1)` : 온칩에 적재된 활성 타일(K_slice=128 폭) — `ActivationBufferSim.scala:23`.
  - `io.kernel(0..S-1)` : Stream(인덱스) 입력. valid/ready/payload — `:35-39`.
  - `io.output(0..S-1)` : Stream(gather 결과). valid/ready/payload — `:43,50`.
- **동작(핵심):** 인덱스 `kernel(i)` 를 받아 `x[kernel(i)]` 를 출력. golden: `expected = X(kernel_idx(i))` — `ActivationBufferSim.scala:47`. 즉 **인덱스 기반 활성 gather 엔진**(메모리 gather/crossbar 추정). FSM 기반 handshake(인덱스 accept 후 output valid 대기 — `:39-43`).
- **연결:** ActivationBuffer.output(i) → TernaryGEMM.buffer(i) 로 직결되어 "gather → 감산누산" 파이프 형성 — `ActivationToTernaryGEMMSim.scala:18-21`:
  ```scala
  for (i <- 0 until S) { ternaryGEMM.io.buffer(i) << activationBuffer.io.output(i) }
  ```

> 정리: ActivationBuffer = **TCSC 인덱스 → 활성값** 의 sparse gather (CPU 의 `_mm256_i32gather_ps` 와 대응, 3-G). 활성을 K_slice=128 폭으로 온칩에 미리 적재해 random-access gather 를 가능케 함.

### 3-E. DataFSM (DMA / 메모리 스케줄러)

**근거 파일:** `sim/old/Base/DataFSMSim.scala` (본체 `gemmacc.src.Base.DataFSM_Base` 부재).

- **생성:** `new DataFSM_Base(config)` — `DataFSMSim.scala:24`.
- **IO(역추적):** 스칼라 파라미터 `M, N, K, Non_zero_per_K_slice` (`:30-33`), `base_addr_X/W/Y` (`:34-36`), 결과 입력 `result_GEMM(0..S_2-1)` (`:65-66`), 표준 `io.AXI` master 포트(`:46`), `start/done` (`:69,82`).
- **역할:** off-chip(HBM/DDR) 에서 X(활성), W(TCSC 인덱스)를 burst read 하여 ActivationBuffer/PE 에 공급하고, 결과 Y 를 write-back 하는 **DMA 컨트롤+주소생성 FSM**. `result_GEMM(i)` 를 받아 Y 주소로 store(`:62-67`).
- 주의: `DataFSM_Base` 는 표준 AXI4 포트 직결 버전, 별도로 Coyote v2 디스크립터 버전(`TopLevelOP_coyote_v2`, 3-F)이 존재.

### 3-F. TopLevel 계층 (3가지 변종)

설계 진화가 sim import 경로로 드러난다(본체 부재, import 라인이 근거):

| 변종 | TopLevel 클래스 (import) | 메모리 인터페이스 | 근거 |
|---|---|---|---|
| Base | `gemmacc.src.Base.TopLevelBase` | 표준 AXI4 (`io.AXI`) | `TopLevelBaseSim.scala:3,52` |
| AXI version | `gemmacc.src.AXI_version.TopLevelOP_AXI` | 표준 AXI4 | `TopLevelOPSim.scala:3,55` |
| Coyote v2 (old) | `gemmacc.src.Op.coyote_v2.old.TopLevelOP_coyote_v2` | Coyote SQ/CQ + AXI4SR | `sim/old/Op/coyote_v2/TopLevelSim.scala:5,48` |
| Coyote v2 (current) | `gemmacc.src.design.TopLevel` | Coyote SQ/CQ + AXI4SR | `sim/ternaryGEMM/TopLevelSim.scala:5,53` |

- **현행 top 의 IO(역추적):** `io.M/N/K`, `io.Non_zero_per_K_slice`, `io.base_addr_X/W/Y`, `io.expected_beats_X` (`ternaryGEMM/TopLevelSim.scala:70-77`), Coyote 포트 `sq_rd/sq_wr/cq_rd/cq_wr/axis_card_recv/axis_card_send` (`:57-62`), `start/done` (`:88-90`).
- **합성된 최종 top:** `ternaryGEMMOP` (= TopLevel + AXI-Lite 제어). AXI-Lite ctrl(awvalid…rresp), Coyote 디스크립터, 512-bit AXI4SR card stream 노출 — `vfpga_top.svh:11-104`.

### 3-G. CPU SIMD 커널 생성기 (SW 가속 본체)

**파일:** `ternaryLLM_CPU/SIMD_Generator.ipynb` (Python 메타프로그래밍; 산출 C++ 는 누락된 `src/GEMM_CPU_*.cpp` 로 들어갈 의도)

FPGA 와 동일한 TCSC gather-subtract 를 x86 SIMD 로 구현하는 **커널 코드 생성기**다. 두 패밀리 확인:

1. **FP32 GEMV (AVX2):** `GEMV_CPU_FP32_rowMajor_TCSC_Merged_GroupMin_G16_AVX2_OpenMP` — `SIMD_Generator.ipynb:21`.
   - row_index 에서 pos/neg 인덱스를 `_mm256_load_si256` 로 로드(`:28-31`), 활성 X 를 `_mm256_i32gather_ps` 로 gather(`:32-35`), `_mm256_sub_ps(pos,neg)` 후 `_mm256_add_ps` 로 누산(`:36-37`) → **gather + (pos−neg) 누산**. 골든 모델 `x_pos - x_neg` 의 SIMD 직역.
   - 언롤 파라미터 `N_UNROLL`, `SIMD_SIZE=8`, `N_SIMD=N_UNROLL/8` 로 커널을 자동 전개(`:57-83`). "GroupMin" = 컬럼 그룹별 최소 비제로 정렬 기법(추정), "Merged" = pos/neg 인덱스 병합 저장.
2. **INT8 GEMM (AVX-512):** `GEMM_CPU_INT8_colMajor_TCSC_Uniform_64xG4_AVX512_OpenMP` — `:113`.
   - `_mm512_load_si512` 로 X 컬럼(`X + row_index[k]*M_ROW + i`)을 512-bit 단위 로드(`:122-123`), `NonZeroPerCol`(컬럼당 균일 비제로 수, "Uniform" 구조적 sparsity) 만큼 반복 누산(`:121,156-159`). 64×G4 = 64행×4그룹 타일.
- 공통: `metadata`, `row_index` 포인터(TCSC), `OpenMP` 멀티스레딩, 결과 store(`_mm256_store_ps` `:40-41`).

> CPU 측 핵심: 두 sparsity 모드 — **GroupMin(unstructured, FP32)** vs **Uniform/NonZeroPerCol(structured, INT8)** — 를 지원. FPGA 의 `Non_zero_per_K_slice` (균일 비제로) 가 Uniform 모드와 직접 대응(§3-A의 `entries_per_K_slice`).

### 3-H. Coyote v2 글루 로직 (실존 RTL)

**파일:** `coyote/Types.scala`, `coyote/AxiCoyote.scala`, `MatrixAdd/logic.scala`, `MatrixAdd/Wrap_logic.scala`

- **Types.scala:** Coyote v2 디스크립터 정의.
  - `ReqT` (128-bit 요청): `opcode(5), strm(2), vaddr(48), len(28)` 등 — `Types.scala:10-32`.
  - `ack_t` (완료 ack) — `:34-43`.
  - `AXI4SR` : 512-bit card stream (`tdata 512, tkeep 64, tid 6, tlast/tvalid/tready`) — `:48-78`. 출처는 dlm repo(`Types.scala:1` 주석).
- **AxiCoyote.scala:** Coyote SQ/CQ(요청/완료 큐) + AXI4SR ↔ **표준 Axi4 메모리 버스** 변환기. sq_rd → AR(`AxiCoyote.scala:28-34`), R → axis_card_recv(`:37-42`), axis_card_send → W(`:55-59`), len→burst 계산 `len/BYTES_PER_BEAT - 1` (`:30,48`). 시뮬레이션에서 `AxiMemorySim` 과 붙여 검증(`TopLevelSim.scala:79`).
- **MatrixAdd/logic.scala:** Coyote v2 동작을 익히기 위한 **최소 예제**(주석 `:12-13`). 길이 10 벡터 X,W 를 읽어 더해 Y 로 저장하는 7-state FSM(IDLE→READ_X→Wait_X→READ_W→WAIT_W→COMPUTE→WRITE_Y→Wait_Y) — `logic.scala:65-175`. SQ 디스크립터 작성(`:81-87`), 512-bit tdata 에서 16-bit×10 슬라이싱(`:101`), 결과 packing(`:163`). **삼진 가속과 무관한 인프라 검증 코드.**
- **Wrap_logic.scala:** AXI-Lite 제어 레지스터 래퍼. `start@0x00, done@0x08, base_addr_X@0x10, W@0x18, Y@0x20` 매핑(`Wrap_logic.scala:32-36`) → host 가 가속기를 제어하는 표준 패턴. (TopLevel 도 동일 패턴으로 `ternaryGEMMOP` 합성됨.)

---

## 4. 데이터 플로우 (FPGA, 현행 Coyote v2 기준)

```
[Host] --AXI-Lite ctrl(start, M/N/K, base_addr_X/W/Y)--> [Wrap/TopLevel ternaryGEMMOP]
                                                              |
        (1) DataFSM: SQ_rd 디스크립터 발행 → Coyote shell → HBM
            - X(활성) burst read  → axis_card_recv(512b) → ActivationBuffer.io.x  (K_slice=128 타일 적재)
            - W(TCSC 인덱스) burst read → kernel 인덱스 스트림
                                                              |
        (2) ActivationBuffer: kernel 인덱스 → x[idx] gather → output(S개)
                                                              |
        (3) TernaryGEMM PE: (pos,neg) 쌍 → S/2개의 (x_pos - x_neg) → 라운드 누산 → 16-bit acc
                                                              |
        (4) DataFSM: result_GEMM(S/2) → SQ_wr → axis_card_send(512b) → HBM(Y)
                                                              |
                                                       done=1 → Host
```
- 근거: 적재/gather/감산 연결은 `ActivationToTernaryGEMMSim.scala:9-21`, 메모리 측은 `DataFSMSim.scala` + `AxiCoyote.scala`, 핸드셰이크/주소는 `ternaryGEMM/TopLevelSim.scala:70-90`, 16-bit 결과는 `vfpga_top.svh:3-6` + `TopLevelSim.scala:104-107`.
- 타일 루프 순서(골든 모델 기준): m(행) → n(컬럼, S/2 그룹) → e(entries, 비제로) — `InputAndCheck.scala:60-85`. K_slice 단위 재적재는 `slice_weights*256` 오프셋으로 표현(`:80`).

---

## 5. HW / SW 매핑

| 단계 | 골든 모델 (Scala) | FPGA RTL (SpinalHDL) | CPU SIMD (C++ 생성) |
|---|---|---|---|
| 가중치 포맷 | pos/neg 인덱스 쌍 `InputAndCheck.scala:76-77` | kernel 인덱스 스트림 (ActivationBuffer.io.kernel) | TCSC `row_index` (`ipynb:28-31`) |
| 활성 gather | `X_matrix(m)(pos)` `:80` | ActivationBuffer `x[idx]` `ActivationBufferSim.scala:47` | `_mm256_i32gather_ps` / `_mm512_load_si512` `ipynb:32,122` |
| 삼진 연산 | `x_pos - x_neg` `:84` | TernaryGEMM `in[2i]-in[2i+1]` `TernaryGEMMSim.scala:90` | `_mm256_sub_ps`+`_mm256_add_ps` `ipynb:36-37` |
| 병렬도 | — | S/2 PE (acc 핀 폭 단서 `vfpga_top.svh:96-99`) | N_UNROLL/SIMD 언롤 `ipynb:57-83` |
| sparsity 모드 | uniform(`entries_per_K_slice`) | `Non_zero_per_K_slice` `TopLevelSim.scala:73` | Uniform(INT8) / GroupMin(FP32) `ipynb:113,21` |
| 메모리 | `AxiMemorySim` | Coyote SQ/CQ + AXI4SR 512b | malloc + OpenMP |
| 정확도 검증 | 기준(spec) | sim 에서 골든과 bit-exact assert `TopLevelSim.scala:108` | (생성 커널 본체 부재로 확인 불가) |

핵심: **세 플랫폼이 단일 알고리즘 명세(naiveGEMM)를 공유**하며, FPGA sim 은 매 테스트에서 골든과 정확히 비교(`assert signedValue == Y(i)` — `TopLevelSim.scala:108`).

---

## 6. 빌드 · 실행

### FPGA (SpinalHDL → SystemVerilog → Coyote/Vivado)
- 빌드 도구: **sbt** (`build.sbt`) 및 **mill** (`build.sc`) 병행. Scala 2.13.14, **SpinalHDL 1.12.0** (`build.sbt:2,5-8`). 소스 루트 `hw/spinal` (`build.sbt:13`), `fork := true` (`:17`).
- 시뮬레이션: 각 `*Sim.scala` 의 `main` 실행 → `SimConfig.withWave.compile(...).doSim{}` (예 `TopLevelSim.scala:52,117`). Verilator/내장 시뮬레이터로 파형 생성.
- 합성/배포: SpinalHDL → `ternaryGEMMOP.sv` 생성 → `vfpga_top.svh` 에서 인스턴스화(`vfpga_top.svh:11`) → **Coyote v2 shell** 에 통합 → `cyt_top.bit`. version_build 변종은 `ila_0` (Vivado ILA) 디버그 코어 포함(`hw/vhdl/version_build/vfpga_top.svh:106-140`).
- 실행: Host 가 AXI-Lite 로 `start@0x00`, base addr 들 write 후 `done@0x08` 폴링(`Wrap_logic.scala:32-36`).

### CPU (MSVC)
- **Visual Studio 2022 (v143)**, C++17 (`ternaryLLM.vcxproj:45,128`). Release x64 에서 **AVX-512** (`EnableEnhancedInstructionSet=AdvancedVectorExtensions512` `:154`), **OpenMP experimental** (`:150-152`), `/fp:fast`, LTCG (`:155-169`).
- 의존: **libtorch**, **Eigen 3.4.0** (include `:129,149`; lib `:136,168` — torch/c10/fbgemm/dnnl 등). 경로 하드코딩 `C:\Personal\Projects\...` (이식성 낮음).
- SIMD_Generator.ipynb → 커널 C++ 텍스트 생성 → `src/GEMM_CPU_*.cpp` 에 붙여 빌드(워크플로 추정; src 본체 부재).

### GPU
- `ternaryLLM_GPU/Benchmarks.ipynb` 단일 파일. 내용 미열람(노트북). CUDA/torch 벤치마크 추정. `.gitignore:8` `torch_gpu/` 참조로 PyTorch CUDA 환경 사용 추정. **확인 불가.**

---

## 7. 의존성

- **SpinalHDL 1.12.0** (`build.sbt:5`) — Scala HDL.
- **Coyote v2** — AMD/Xilinx FPGA shell (디스크립터 SQ/CQ, AXI4SR 512b). `Types.scala` 는 외부 dlm repo 차용(`Types.scala:1`).
- **Verilator** (추정; SpinalHDL sim 백엔드).
- **libtorch + Eigen 3.4.0 + OpenMP + AVX-512** (CPU, `ternaryLLM.vcxproj:129,149-154`).
- **fbgemm/dnnl/cpuinfo** (libtorch 동봉 lib, `:136`).
- 빌드 산출물 vendor 디렉토리(`PyTorch_Extension/`, `x64/`, `torch_gpu/`)는 gitignore 처리(`.gitignore`).

---

## 8. 강점 · 한계

### 강점
- **곱셈기 제거:** ternary 를 인덱스+감산으로 환원 → DSP/곱셈기 거의 0, FPGA LUT 효율 극대화(`InputAndCheck.scala:84`, `TernaryGEMMSim.scala:90`).
- **sparsity 선형 비례 연산:** 비제로 entries 만 처리 → 0 가중치 완전 skip(`InputAndCheck.scala:68-85`).
- **단일 명세, 3-플랫폼 검증:** golden(naiveGEMM)으로 HW/SW 모두 bit-exact 검증(`TopLevelSim.scala:108`). 크로스 플랫폼 공정 비교 가능.
- **Coyote v2 통합:** HBM 직결, 512-bit 고대역 card stream, descriptor 기반 DMA(`AxiCoyote.scala`, `Types.scala:48`).
- **메타프로그래밍 커널 생성:** 언롤/타일 파라미터화된 SIMD 자동 생성(`SIMD_Generator.ipynb:57-83`) → DSE 친화적.

### 한계
- **GEMM 만, 풀 Transformer 미포함:** attention(softmax), LayerNorm, GELU, KV-cache, 위치인코딩 RTL **없음**. 이름이 LLM 이지만 가속 범위는 *선형 계층 GEMM/GEMV* 에 국한(repo 전체 Grep 상 attention/softmax/layernorm 모듈 부재 — 확인 불가가 아니라 부재).
- **부분 체크아웃:** 설계 본체(`gemmacc.src.*`, CPU `src/*`)가 누락되어 **내부 RTL/커널 본문 검증 불가**(§1, §2). 본 분석은 sim/svh/golden 역추적 기반.
- **활성은 dense, 가중치만 sparse:** ActivationBuffer 가 K_slice=128 전체를 적재(`ActivationToTernaryGEMMSim.scala:30`) → 활성 sparsity 미활용.
- **이식성:** CPU 경로 하드코딩(`vcxproj:87,129`), Coyote/특정 FPGA 보드 종속.
- **결과 누산 폭 16-bit:** 큰 K 에서 overflow 위험(acc 16-bit, `vfpga_top.svh:3`; golden 은 32-bit Int `InputAndCheck.scala:39`이므로 sim 통과해도 실HW 포화 가능 — 추정).
- **GPU 경로 불투명:** 노트북 1개, 내용 미확인.

---

## 9. 우리 프로젝트(PRJXR-HBTXR: 고처리량 ViT/Transformer FPGA 가속 + XR 시선추적) 시사점

> 우리 타깃이 HG-PIPE 계열 고처리량 ViT 가속 + XR eye-tracking 이라는 가정 하의 시사점.

1. **삼진/극저비트 GEMM 의 곱셈기 제거 기법은 직접 채용 가치 큼.** XR 엣지에서 DSP 가 병목일 때, ViT 의 FFN/projection 을 ternary 로 양자화하면 본 repo 의 "인덱스 gather + 감산 트리"(`TernaryGEMM`, `InputAndCheck.scala:84`)로 곱셈기 없이 구현 가능. HG-PIPE 의 dense systolic 대비 DSP 절감 트레이드오프 후보.
2. **단, 우리는 attention/softmax/LayerNorm 이 필수** — 본 repo 에 부재. 즉 ternaryLLM 은 **선형층 가속 IP 블록** 으로만 차용하고, 비선형/attention 데이터패스는 별도 설계(HG-PIPE 또는 우리 ViT-Accelerator) 필요.
3. **Coyote v2 vs HG-PIPE 스트리밍:** 본 repo 는 descriptor/HBM 기반 batch 처리(`AxiCoyote.scala`)로 *지연(latency)* 보다 *처리량(throughput)* 지향. XR 시선추적은 frame-level 저지연이 중요 → HG-PIPE 식 fully-pipelined stationary 데이터플로우가 우리에 더 적합. 본 repo 의 gather 기반은 random-access 라 파이프라인 정합성이 낮을 수 있음(추정).
4. **TCSC 인덱스 포맷 + 균일 sparsity(`Non_zero_per_K_slice`)** 는 우리 양자화 파이프라인(ViT-Quantization)에 구조적 sparsity 옵션으로 통합 검토 가치. structured(Uniform)가 HW 친화적(`ipynb:113`)임을 본 repo 가 실증.
5. **단일 golden 명세로 HW/SW 동시 검증하는 방법론**(`naiveGEMM` ↔ sim assert)은 우리 HW/SW 코드베이스 정합성 검증 패턴으로 그대로 도입 권장(`TopLevelSim.scala:108`).
6. **SIMD 메타프로그래밍 생성기**(`SIMD_Generator.ipynb`)는 우리 알고리즘 피드백/DSE(algo2fpga) 루프의 CPU 레퍼런스 자동생성에 응용 가능.

---

## 10. 근거 표기 요약

- **라인 확인(실존 코드):** §3-A(InputAndCheck.scala), §3-H(Types/AxiCoyote/logic/Wrap_logic.scala), §3-C(vfpga_top.svh), §3-G(SIMD_Generator.ipynb grep), §6 빌드(build.sbt, vcxproj).
- **인터페이스/동작 역추적(sim·svh 기반, 본체 부재):** §3-B TernaryGEMM, §3-D ActivationBuffer, §3-E DataFSM, §3-F TopLevel — 모두 `gemmacc.src.*` 본체 파일 부재(확인: Glob/Grep 0건). 동작은 sim assert 로 확정, **내부 RTL 본문은 본 체크아웃에서 확인 불가**.
- **추정 명시:** PE 병렬도(acc 핀 수 단서), GPU 노트북 내용, GroupMin/Uniform 세부 의미, 16-bit 누산 overflow, latency 특성, libtorch/Verilator 사용.
- **부재(확인된 누락):** attention/softmax/LayerNorm/GELU RTL, CPU `src/*.cpp/.hpp`, FPGA `gemmacc/src/*.scala`, `PyTorch_Extension/`.
```
