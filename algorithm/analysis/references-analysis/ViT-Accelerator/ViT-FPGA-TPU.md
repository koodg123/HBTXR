# ViT-FPGA-TPU 정밀 분석 (v2.0)

> 분석 대상: `\\wsl.localhost\ubuntu-24.04\home\user\project\PRJXR-HBTXR\REF\ViT-Accelerator\ViT-FPGA-TPU`
> 분석 범위: 자체 작성 RTL(`fp_sys_array`) + C++ 호스트(`v2.0/sw`) + Python 예제. Vivado/Xilinx IP 생성물은 이름만 언급하고 내부 미분석.
> 작성일: 2026-06-20

---

## 1. 개요

- **목적**: Vision Transformer(ViT)의 핵심 연산인 행렬곱(GEMM)을 FPGA 상의 TPU 스타일 **시스토릭 어레이(systolic array)**로 가속하는 것. 어텐션/MLP의 선형 변환을 16×16 타일 GEMM으로 분해해 하드웨어로 오프로딩한다.
- **한 줄 요약**: PCIe(XDMA)로 호스트와 연결된 Virtex-7 FPGA에 16×16 **부동소수점(FP16) 시스토릭 어레이**를 올리고, Xilinx Floating-Point IP 2개(곱셈+누산)로 만든 파이프라인 MAC을 PE로 사용하는 행렬곱 가속기. C++/Python 호스트가 행렬을 skew(slant) 변환·타일링해 FPGA DRAM에 DMA로 보내고 결과를 회수한다.
- **원 프로젝트 성격**: Harvard **CS205**(고성능 컴퓨팅) 과목의 학생 팀 프로젝트로 보임(루트 `README.md` 2행 "Harvard CS205", `milestone/2~5` PDF 다수, 기여자 표 Eric/Hongyi/Wenyun/Sebastian). milestone 기반 단계적 개발 흔적이 명확하며 v2.0이 최신 버전.
  - 근거: `README.md:1-2`, `v2.0/README.md:1-7`, `milestone/{2,3,4,5}/*.pdf`(milestone 5에 `report.pdf` 존재).
- **타깃 디바이스 / 보드** (확정):
  - **FPGA Part: `xc7vx485t-ffg1761-2` (Xilinx Virtex-7)**, **Board: Xilinx VC707** — 근거: `const.xdc:4-9`("Family virtex7 / Part xc7vx485t / Package ffg1761 / Speed grade -2"), `pci_mig_16_v2.xpr:10`(`Part="xc7vx485tffg1761-2"`), `pci_mig_16_v2.xpr:50`(`BoardPart="xilinx.com:vc707:part0:1.4"`).
  - 호스트 인터페이스: **PCIe Gen2 x8** + Xilinx **XDMA** — 근거: `const.xdc:4-7`("Link Width x8 / Link Speed gen2"), 블록디자인 IP `design_1_xdma_0_0.xci`.
  - 동작 주파수: MAC IP는 100 MHz로 패키징(`pipeline_mac.v:26,30,33,36` `FREQ_HZ 100000000`), 단 `example.py:58`은 성능 환산에 `fpga_freq=200e6`(200 MHz)을 사용 → 실측 합성 주파수는 **확인 불가**(둘 중 어느 것이 실제 구현 클럭인지 명시 없음, "추정" 영역).

---

## 2. 디렉토리 구조

### 2.1 자체 작성 소스(분석 대상)

```
ViT-FPGA-TPU/
├─ README.md                         # 루트 개요(CS205, libTorch ViT 실행법, FPGA TB 안내)
├─ milestone/{2,3,4,5}/*.pdf         # 발표/리포트(학생 프로젝트 단계별 산출물)
├─ code/code_fpga/accel_driver/README.md  # 구버전 드라이버 빌드/실행 안내
└─ v2.0/                             # ★ 최신 버전
   ├─ README.md
   ├─ hw/tpu_16/ip_repo/fp_sys_array/src/
   │   ├─ systolic_array.sv          # ★ 16×16 PE 어레이(generate)
   │   ├─ PE.sv                      # ★ 단일 PE(파이프라인 MAC 래핑 + skew용 패스스루)
   │   ├─ pipeline_mac.v             # FP 곱셈+누산 IP 2개 결선(BD 넷리스트, 자체 결선)
   │   ├─ pipeline_mac_wrapper.v     #   └ 상기 모듈 래퍼
   │   ├─ systolic_array_wrapper.sv  # 1비트 IO로 좁힌 합성용 래퍼(IP 패키징용)
   │   ├─ top.sv                     # 시뮬레이션 top(clk gen + TB 인스턴스)
   │   └─ systolic_array_tb.sv       # ★ Output-Stationary 검증 TB(skew/eye/ones)
   └─ sw/                            # ★ C++/Python 호스트
      ├─ CMakeLists.txt              # libTorch 링크, libfpga_gemm.so + main 빌드
      ├─ README.md                   # 빌드/XDMA 드라이버 로드/실행 안내
      ├─ src/main.cpp                # ★ XDMA 기반 GEMM 드라이버(타일링/skew/레지스터)
      ├─ src/example.py             # ★ ctypes로 .so 호출, 처리량 벤치마크
      └─ include/
          ├─ accelerator.h          # ★ MMIO 레지스터 맵 / 명령·상태 enum / MATRIX_SIZE
          ├─ dma_utils.h            # Xilinx DMA read/write 헬퍼(BSD, Xilinx 원본 유래)
          └─ matrix_utils.h         # skew/transpose/eye/print 등 행렬 유틸 + FP16 typedef
```

### 2.2 제외한 Vivado/Xilinx IP 생성물 (이름만 언급, 미분석)

- 블록디자인/프로젝트: `pci_mig_16_v2.xpr`, `tmp_edit_project.*`, `design_1.bd`, `*.bda/*.bxml/*.hwh/*.hwdef`, `*.gen`, `*.cache`, `*.hw`, `*.srcs`(IP 생성물).
- 핵심 행렬곱 PE에 쓰인 부동소수점 IP: `pipeline_mac_floating_point_0_0.xci`(FP **곱셈**), `pipeline_mac_floating_point_1_0.xci`(FP **누산/덧셈**) → **Xilinx Floating-Point IP 사용**으로만 취급(내부 RTL은 생성물).
- 시스템 IP(블록디자인 `design_1` 구성요소, 전부 Xilinx IP): `xdma`(PCIe DMA), `smartconnect`(AXI 인터커넥트), `axi_bram_ctrl` + `blk_mem_gen`(BRAM 스크래치패드), `util_ds_buf`(PCIe 차동클럭 버퍼), MIG/메모리(`pci_mig_*`), 그리고 자체 컨트롤러 IP를 감싼 `pci_mig_accelerator`(컨트롤러 RTL 소스는 v2.0 트리에서 **확인 불가** — 후술).

---

## 3. 핵심 RTL/SW 모듈별 정밀 분석 (가장 중요)

### 3.1 `systolic_array.sv` — 16×16 PE 어레이

**파라미터/포트** (`systolic_array.sv:3-19`):
- 파라미터: `INPUT_WIDTH=32`, `WEIGHT_WIDTH=32`, `OUTPUT_WIDTH=32`, `NUM_ROWS=16`, `NUM_COLS=16`. → **물리 어레이는 16×16 = 256 PE**.
  - 주의: 데이터 폭이 32비트로 선언돼 있으나, 실제 산술 IP는 **FP16(Half)** 로 설정됨(§3.3·§4 참조). 32비트는 AXIS 버스 폭(바이트 정렬)으로 보이며, FP16 데이터가 하위 비트에 실린다("추정", IP의 `TDATA_NUM_BYTES 4`와 `A_Precision_Type=Half`의 조합 근거).
- 입력: `input_i[NUM_ROWS]`(좌측에서 행 단위 주입) + `input_valid_i[NUM_ROWS]`, `weight_i[NUM_COLS]`(상단에서 열 단위 주입) + `weight_valid_i[NUM_COLS]`.
- 출력: `output_o[NUM_ROWS][NUM_COLS]`(각 PE의 누산 결과) + `output_valid_o[NUM_ROWS][NUM_COLS]`. → **모든 PE가 자기 누산값을 보유 = Output-Stationary 성향**(주석 22-23 "input is on the left / weight is on the top").

**데이터플로우(generate 결선, `systolic_array.sv:39-82`)**:
- 2중 generate로 256개 `PE` 인스턴스 생성(40-60행).
- 경계 주입: `i==0`이면 상단에서 `weight_i[j]` 주입(61-64행), `j==0`이면 좌측에서 `input_i[i]` 주입(66-69행).
- 내부 전파: weight는 **아래로**(`_weight_i[i+1][j] = _weight_o[i][j]`, 71-74행), input은 **오른쪽으로**(`_input_i[i][j+1] = _input_o[i][j]`, 76-79행) 한 클럭씩 흐른다.
- 즉 **input은 좌→우, weight는 상→하로 1-PE/사이클 systolic 전파**하고, 각 PE는 통과하는 input·weight를 곱·누산해 `output_o`에 머무는 전형적 출력 고정형 어레이.

### 3.2 `PE.sv` — 단일 처리요소(곱셈누산 + skew 패스스루)

**파라미터/포트**(`PE.sv:22-43`): `INPUT/WEIGHT/OUTPUT_WIDTH=32`. 입력 `input_i/weight_i`(+valid), 출력으로 (a) 다음 PE로 흘려보내는 `input_o/weight_o`(+valid)와 (b) 누산 결과 `output_o`(+`output_valid_o`).

**구성**:
1. **MAC 인스턴스**(`PE.sv:54-68`): `pipeline_mac_wrapper`를 인스턴스화. AXI-Stream 결선:
   - `S_AXIS_A_0_tdata = input_i`, `S_AXIS_B_0_tdata = weight_i` (62, 60행),
   - 두 스트림 valid = `_mac_operands_ready`(59, 61행),
   - `M_AXIS_RESULT_0_tready = 1'b1`(57행, 항상 결과 수용),
   - 결과 `M_AXIS_RESULT_0_tdata → output_o`, `tvalid → output_valid_o`(63, 65행).
2. **연산 트리거**(`PE.sv:85-92`): `_mac_operands_ready = input_valid_i & weight_valid_i`(86행) — input과 weight가 동시에 유효할 때만 MAC을 구동.
3. **systolic 패스스루**(`PE.sv:95-104`): 매 클럭 `input_o<=input_i`, `weight_o<=weight_i`, valid도 한 클럭 지연 전달 — 이것이 어레이 전파의 1-cycle 레지스터 단.
4. **리셋**(`PE.sv:71-78`): 패스스루 레지스터와 `output_o`를 0으로.
   - 한계: `output_o`는 MAC IP 출력에 직접 결선(63행)되어 있어 `reset()`의 `output_o<='0`(77행)는 IP 출력과 충돌/무효일 수 있음(드라이버 충돌 가능성, "추정"). 또한 PE 자체에는 누산 클리어/결과 캡처 FSM이 없어 **누산 리셋은 MAC IP의 Accumulator 동작과 TLAST에 의존**.

### 3.3 `pipeline_mac.v` / `pipeline_mac_wrapper.v` — FP16 파이프라인 MAC

이 두 파일은 Vivado가 블록디자인(`pipeline_mac.bd`)에서 생성한 **넷리스트**지만, 두 Floating-Point IP를 직렬로 잇는 **결선 자체는 설계 의도를 드러내므로 분석 대상**으로 본다(IP 내부 RTL은 미분석).

- 인터페이스(`pipeline_mac.v:13-37`): AXIS slave `S_AXIS_A_0`(피연산자 A), `S_AXIS_B_0`(피연산자 B), AXIS master `M_AXIS_RESULT_0`(결과, `tlast` 보유), `clk`/`rst_n`. 버스 폭 전부 `[31:0]`(`TDATA_NUM_BYTES 4`).
- 내부 구조(`pipeline_mac.v:67-89`): **2단 직렬**.
  - `floating_point_0`(곱셈기): `s_axis_a = A`, `s_axis_b = B` → 곱 결과(`floating_point_0_M_AXIS_RESULT_TDATA`).
  - `floating_point_1`(누산기): `s_axis_a = floating_point_0의 곱 결과` → 누적 합(`M_AXIS_RESULT_0_tdata`로 출력, `tlast` 동반).
  - 즉 **C ← C + A·B** 의 FMA를 곱셈 IP + 누산 IP 직렬 파이프라인으로 구현.
- **IP 설정(.xci 근거, 핵심)**:
  - `pipeline_mac_floating_point_0_0.xci`: `Operation_Type=Multiply`, `A_Precision_Type=Half`, `Result_Precision_Type=Half`, `C_Latency=4`, `C_Mult_Usage=Full_Usage`, `Flow_Control=Blocking`(`xci:171,189,190,197,211,213`).
  - `pipeline_mac_floating_point_1_0.xci`: `Operation_Type=Accumulator`, `Add_Sub_Value=Add`, `A_Precision_Type=Half`, `C_Latency=4`(`xci:182,184,200,208,222`).
  - → **데이터 타입은 FP16(half-precision)**, 각 단 레이턴시 4 → **MAC 파이프라인 깊이 ≈ 8 사이클**(곱4 + 누산4). DSP **Full_Usage**로 곱셈에 DSP48 적극 사용.
  - 곱셈/누산 모두 **Blocking** flow control → 백프레셔 기반 AXIS 핸드셰이크.

### 3.4 `systolic_array_wrapper.sv` — IP 패키징용 축약 래퍼

- 외부 IO를 **1비트**로 좁혀(`_input_i`, `_weight_i`, `_output_o[NUM_ROWS]`) 합성/IP 패키징 시 포트 수를 줄인 래퍼(`systolic_array_wrapper.sv:3-18`).
- 내부에서 1비트 입력을 전 행/열에 브로드캐스트(`28-34행`)하고 `output_o[i][j]`를 행 단위로 AND 리덕션해 `_output_o[i]` 생성(36행).
- 평가: 실제 기능 어레이라기보다 **합성 가능성/리소스 추정·IP 외형 패키징용 스텁**에 가까움(데이터 폭이 1비트로 죽으므로 실연산 의미 없음, "추정").

### 3.5 `top.sv` + `systolic_array_tb.sv` — Output-Stationary 검증 환경

- `top.sv:3-43`: 시뮬레이션 top. 10ns 주기 클럭 생성(11행), `systolic_array_OS_tb` 인스턴스(16행, **OS=Output Stationary** 명명), VCD 덤프(30-31행), 타임아웃 트랩(19-25행).
- `systolic_array_tb.sv` — 실제 DUT 검증 로직(파라미터 `NUM_ROWS=NUM_COLS=16`, `*_WIDTH=32`, `tb:5-9`):
  - DUT = `systolic_array`(44-52행).
  - `$bitstoshortreal`/`$shortrealtobits`로 32비트 워드를 **IEEE 단정밀(shortreal)** 로 해석해 검증(29, 33, 35행, 84, 95행). → TB 레벨에서는 FP32로 다루는 점에 주목(IP는 Half이나 TB는 shortreal=FP32; 폭/정밀도 불일치 가능성, "확인 필요" 영역).
  - **skew(slant) 행렬 생성**: `get_slant_input_matrix`(100-109행), `get_slant_weight_matrix`(111-120행) — systolic 어레이 주입을 위해 입력은 행별로, weight는 열별로 대각 시프트시켜 시차 주입.
  - 테스트 케이스: weight = 대각 2.0 행렬(`eye_matrix`, 80-90행), input = 전부 2.5(`ones`, 92-98행) → 결과 검증.
  - 주입 순서: `send_input_weights`가 anti-diagonal 순서로 한 열씩 `set_inputs`/`set_weights` 호출(122-135행).

### 3.6 `accelerator.h` — MMIO 레지스터 맵 & 명령/상태 모델

가속기의 **소프트웨어 계약(ABI)**. 컨트롤러 RTL 소스는 트리에 없지만, 이 헤더가 컨트롤러의 레지스터/FSM 의미를 그대로 드러냄.

- `MATRIX_SIZE 16`(`accelerator.h:6`) → 호스트도 **16×16 타일**을 단위로 동작(어레이 크기와 일치).
- `FPGA_DATA_WIDTH = 128/8 = 16바이트`(MATRIX_SIZE==16, `:9`), `REG_SHIFT_AMT=6`(레지스터 간격 64바이트, `:10`). → AXI-Lite/메모리맵 레지스터는 64B 정렬.
- **MMIO 레지스터 맵**(`:19-33`, base `ACCEL_ADDR=0x100000000`):
  - `R_MATRIX_A/B/C_ADDR`(FPGA DRAM 내 행렬 A/B/C 주소), `R_MATRIX_A/B_SIZE`(바이트 길이), `R_MATRIX_RD_CNT`(읽을 행렬 개수), `R_MATRIX_M/N/K_SIZE`(GEMM 차원), `R_ACCEL_INSTR`(명령), `R_ACCEL_STATE`(상태), `R_ACCEL_DATA`, `R_SYS_ARR_OUTPUT`.
- **명령 enum**(`accel_instr_e`, `:35-42`): `I_IDLE/I_R_MAT_A/I_R_MAT_B/I_R_MAT_C/I_GEMM/I_RESET`.
- **상태 enum**(`accel_state_e`, `:44-51`): `S_IDLE → S_WAIT_ARW_READY → S_WAIT_RW → S_WAIT_COMPUTE → S_SEND_MAT_C → S_DONE` — 컨트롤러 FSM이 DMA read→compute→write back 순으로 진행함을 시사.

### 3.7 `main.cpp` — XDMA 기반 GEMM 드라이버 (호스트 핵심)

- **디바이스 노드**(`main.cpp:40-41`): `/dev/xdma0_h2c_0`(host→card 쓰기), `/dev/xdma0_c2h_0`(card→host 읽기) — XDMA 캐릭터 디바이스.
- **저수준 액세스**:
  - `fpga_write/read`(`:56-62`) → `dma_utils.h`의 `write_from_buffer/read_to_buffer`로 위임(seek+read/write).
  - `fpga_reg_write/read`(`:64-76`): `FPGA_DATA_WIDTH(16B)` 정렬 버퍼로 단일 레지스터 R/W.
  - `issue_accel_instr`(`:78-89`): `R_ACCEL_INSTR`에 명령 기록 후, blocking이면 같은 레지스터를 폴링해 idle 될 때까지 대기(소프트웨어 동기화).
- **행렬 적재**: `write_mat_ab_all`(`:123-154`)이 A·B를 FPGA DRAM에 연속 배치하고(`mat_b_addr=mat_a_addr+total`, `mat_c_addr=mat_b_addr+total`) 주소·크기·개수·M 차원을 레지스터에 기록.
- **GEMM 본체**: `gemm_padded`(`:330-411`) / `gemm_padded_ptr`(`:243-327`):
  - 차원을 `MATRIX_SIZE`로 나눠 **3중 타일 루프**(`m,n,k`, `:264-266`, `:350-352`).
  - 각 타일을 `at::transpose_copy`로 전치(systolic 주입 형식 맞춤, `:276-277`)하고 `skew_matrix`로 **slant 변환**(`:294-295`)한 뒤 다시 전치해 버퍼에 적재.
  - `fpga_gemm`(`:158-218`)을 타일마다 호출: A·B를 한 버퍼로 합쳐 DMA write → `issue_accel_instr(I_R_MAT_A, blocking)`로 연산 개시 → `mat_c_addr`에서 결과 read → 타일 결과를 `mat_c`에 누적(`+=`, `:313`, `:399`).
  - 시간 측정: `clock_gettime(CLOCK_MONOTONIC_RAW)`로 raw FPGA compute 시간 반환(`:191-200`).
- **데이터 타입**: 전 구간 **`float16_t`(=`_Float16`)** 사용(`matrix_utils.h:10`), libTorch 텐서는 `torch::kFloat16`(`:249-251`). 골든 비교 시에만 FP32로 승격 후 FP16으로 환원(`:437`).
- **C-ABI 노출**(`:29-32`): `extern "C" gemm_padded`, `gemm_padded_ptr`, `get_fpga_matrix_size` → Python ctypes 호출용.

### 3.8 `dma_utils.h` / `matrix_utils.h`

- `dma_utils.h`: **Xilinx dma_ip_drivers 유래(BSD)** 의 표준 `read_to_buffer`/`write_from_buffer`(lseek 후 `RW_MAX_SIZE=0x7ffff000` 단위 분할 전송, `:43-146`) + `timespec` 헬퍼. 자체 작성이 아닌 벤더 코드 재사용.
- `matrix_utils.h`: `float16_t` typedef(`:10`), 다양한 초기화(`set_row/col_consecutive`, `set_eye_matrix_f16`), 핵심 **`skew_matrix`**(`:90-118`, 각 행을 인덱스만큼 우측 시프트해 대각 적재 = systolic 주입용 slant), 디버그 출력(`print_imatrix/xmatrix`), 참조용 `outer_product_mmm`(CPU GEMM, `:121-130`).

### 3.9 `example.py` — Python 벤치마크 프런트엔드

- `ctypes.CDLL("../build/libfpga_gemm.so")`로 C++ 라이브러리 로드(`:7-8`), `get_fpga_matrix_size()`로 어레이 크기 확인(`:9`).
- `fpga_gemm_pad`(`:19-43`): numpy 행렬을 `MATRIX_SIZE` 배수로 zero-pad 후 `gemm_padded_ptr` 호출, 패딩 제거 후 반환.
- 벤치마크(`:45-95`): 150×150 FP16 랜덤 행렬로 FPGA vs CPU(`a@b`) 처리량 비교. 이론 최대 처리량 = `(sys_size**2)*2 = 16*16*2 = 512 FLOP/cycle`(`:68`), 200 MHz 가정 시 약 102.4 GFLOPS(FP16)로 환산.

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 시스토릭 16×16 데이터플로우 (Output-Stationary)

1. 호스트가 16×16 타일 A(input)·B(weight)를 **전치 + skew(slant)** 하여 FPGA DRAM에 배치(§3.7).
2. 어레이 좌측에서 input이 행별로, 상단에서 weight가 열별로 시차 주입(`systolic_array.sv:61-69`).
3. input은 매 사이클 좌→우(`:76-79`), weight는 상→하(`:71-74`)로 1-PE씩 전파.
4. 각 PE는 자기 위치를 지나는 (input,weight) 쌍을 FP16 MAC으로 곱·누산, **결과는 PE에 머무름**(`output_o[i][j]`). → 16×16 = 256개 부분합이 동시에 누적되는 출력 고정형.

### 4.2 FP MAC 파이프라인 깊이

- 1 PE = `pipeline_mac` = FP16 곱(latency 4) → FP16 누산(latency 4) 직렬 = **약 8 사이클**(.xci `C_Latency=4`×2). Blocking AXIS 핸드셰이크로 백프레셔(`pipeline_mac_floating_point_*.xci`).
- valid 게이팅: `_mac_operands_ready = input_valid_i & weight_valid_i`(`PE.sv:86`) — 두 피연산자 동시 유효 시에만 누산 1회.

### 4.3 메모리 계층 / 인터페이스

- 호스트 ↔ FPGA: **PCIe Gen2 x8 + XDMA**(`/dev/xdma0_{h2c,c2h}_0`). 제어는 BAR 매핑 MMIO 레지스터(`accelerator.h` 레지스터 맵, base `0x100000000`).
- 온칩: 블록디자인의 **AXI BRAM(`axi_bram_ctrl`+`blk_mem_gen`) 스크래치패드**, **SmartConnect** AXI 인터커넥트, 메모리 컨트롤러(`pci_mig_*`). 컨트롤러 IP(`pci_mig_accelerator`)가 DMA·BRAM·시스토릭 어레이를 조율(컨트롤러 RTL은 v2.0 트리에서 확인 불가).
- 데이터: 행렬은 FPGA 측 DRAM/BRAM 주소(`mat_a_addr=0x0`, `mat_b_addr=0x2000`, `mat_c_addr=0x4000` 기본값, `main.cpp:47-49`)에 배치.

### 4.4 ViT 행렬곱 매핑

- ViT의 모든 선형 연산(QKV projection, attention score `QK^T`, `softmax·V`, MLP FC1/FC2)은 결국 GEMM. 호스트는 임의 (M,K,N)을 16의 배수로 패딩(`example.py:26-28`)하고 16×16 타일 3중 루프로 분해(`main.cpp:264-266`)해 어레이에 흘린다. → **ViT = 타일 GEMM 시퀀스로 환원**.
- 단, 본 코드베이스의 v2.0 SW는 **순수 GEMM 가속/벤치마크**까지만 구현돼 있고, ViT 전체 그래프(어텐션/softmax/LayerNorm/GELU)는 루트 `code/code_cpu`의 libTorch ViT(CPU)에 있으며 v2.0에서 어텐션 일체를 FPGA로 매핑하는 통합 코드는 **확인 불가**(GEMM 오프로딩 경로만 확인됨).

### 4.5 데이터 타입 (FP16 vs FP32)

- **하드웨어 MAC: FP16(Half)** — 곱·누산 IP 모두 `A_Precision_Type=Half`(.xci, §3.3).
- **호스트: FP16** — `float16_t`/`torch::kFloat16` 전 구간(§3.7).
- **시뮬레이션 TB: FP32(shortreal)** — `$bitstoshortreal`로 32비트를 단정밀로 해석(`systolic_array_tb.sv:29`). RTL 포트 폭 32비트와 IP Half 설정 사이의 정합은 "추정"(버스 폭 32, 데이터 의미 16) 영역으로, **TB와 합성 IP의 정밀도가 불일치**할 가능성이 있어 검증 신뢰도에 유의.

---

## 5. HW/SW 매핑

```
example.py (numpy FP16, pad to ×16)
   │  ctypes CDLL → libfpga_gemm.so
   ▼
gemm_padded_ptr / gemm_padded (main.cpp)
   │  타일 루프(m,n,k) → transpose + skew_matrix(slant)
   │  write_from_buffer (XDMA H2C) → FPGA DRAM(A,B)
   │  fpga_reg_write (MMIO: A/B/C addr, size, cnt, M)
   │  issue_accel_instr(I_R_MAT_A) → R_ACCEL_INSTR 기록 + 폴링 동기화
   ▼
[FPGA] pci_mig_accelerator 컨트롤러 FSM (S_IDLE→…→S_DONE)
   │  BRAM/DRAM → systolic_array 16×16 주입
   ▼
systolic_array.sv → 256× PE.sv → pipeline_mac (FP16 곱4 + 누산4)
   │  output_o[16][16] 누산 결과
   ▼
컨트롤러가 mat_c_addr에 결과 적재
   ▲
read_to_buffer (XDMA C2H) → mat_c (main.cpp) → numpy 반환(example.py)
```

- 계약점: `MATRIX_SIZE=16`(SW)과 `NUM_ROWS/COLS=16`(RTL)이 일치해야 함. `get_fpga_matrix_size()`가 런타임에 이를 노출(`main.cpp:52-54`, `example.py:9`).
- 동기화: 인터럽트 없이 **MMIO 폴링**(`issue_accel_instr`/`matmul_sync`, `main.cpp:78-97`).

---

## 6. 빌드 · 실행 방법

### 6.1 호스트(C++/Python)

- 의존: libTorch(C++) — 루트 README대로 다운로드(`README.md:11-13`), `CMakeLists.txt:5` `CMAKE_PREFIX_PATH=/home/ericd/libtorch`(경로 하드코딩 → 사용자 수정 필요), g++-12, C++23(`CMakeLists.txt:4,28`).
- 빌드(`v2.0/sw/README.md:6-15`): `mkdir build && cd build && cmake .. && make` → `libfpga_gemm.so` + `main` 생성.
- XDMA 드라이버 로드: `cd PATH_TO_dma_ip_drivers/XDMA/linux-kernel/tests && sudo ./load_driver.sh`(`sw/README.md:17-21`, `load_xdma` 별칭 권장).
- 실행: `sudo ./main`(C++) 또는 `python3 example.py`(`sw/README.md:23-31`). FPGA가 장착된 서버(README상 Hyperion/Eric 계정)에서만 실연산 가능.

### 6.2 하드웨어(Vivado)

- 블록디자인 흐름: Vivado 2021.1(`pipeline_mac.v:3`)로 `pci_mig_16_v2` 프로젝트 열기 → `design_1.bd`(XDMA+SmartConnect+BRAM+MIG+`pci_mig_accelerator`) 합성/구현 → VC707(xc7vx485t) 비트스트림. `fp_sys_array`는 `ip_repo`의 사용자 IP로 패키징(`component.xml`).
- 시뮬레이션: `top.sv` + `systolic_array_tb.sv`로 어레이 단독 검증(VCD 출력).

---

## 7. 의존성

| 구분 | 의존성 | 근거 |
|---|---|---|
| 합성/구현 | Vivado **2021.1**, Virtex-7 xc7vx485t, VC707 보드 | `pipeline_mac.v:3`, `const.xdc:4-9`, `xpr:50` |
| FPGA IP | Xilinx **Floating-Point IP**(Half, Multiply/Accumulator), XDMA, SmartConnect, AXI BRAM Ctrl, blk_mem_gen, MIG | `*.xci`, `design_1` BD |
| 드라이버 | Xilinx **dma_ip_drivers**(XDMA 커널 모듈) | `sw/README.md:17`, `dma_utils.h:4-9` |
| C++ | **libTorch**, g++-12, **C++23**, CMake ≥3.0 | `CMakeLists.txt:1,4,9,28` |
| Python | numpy, psutil, ctypes(stdlib) | `example.py:1-5` |

---

## 8. 강점 / 한계 / 리스크

**강점**
- 부동소수점(FP16) systolic GEMM을 **Xilinx FP IP 조합(곱+누산)** 으로 깔끔히 구현 — 정수 양자화 없이 ViT를 가속하는 단순·이식성 좋은 접근.
- HW/SW 경계가 명확(MMIO 레지스터 ABI + XDMA + ctypes), Python까지 한 번에 닿는 end-to-end 데모.
- 타일링·패딩·skew를 호스트에서 처리해 임의 크기 GEMM을 16×16 어레이로 매핑.

**한계**
- **컨트롤러 RTL 미공개(v2.0)**: `accelerator.h`의 FSM/레지스터 계약은 있으나, 실제 `pci_mig_accelerator` HDL 소스가 v2.0 트리에 없어 DMA↔BRAM↔어레이 조율 로직 검증 불가(생성물/구버전 `code/code_fpga`에 일부 존재 가능, 확인 불가).
- **FP16 누산 정밀도**: 누산까지 FP16(Half)이라 큰 K(누적 길이)에서 정밀도 손실 위험(누산기를 FP32로 두는 일반적 best practice와 상이). TB는 FP32(shortreal)로 검증해 **정밀도 불일치** 가능.
- **저주파/저병렬**: IP 100 MHz 패키징(`pipeline_mac.v:36`), 단일 16×16 어레이 1개 → 이론 512 FLOP/cycle 수준. 고처리량 목표엔 부족.
- **동기화 = 폴링**: 인터럽트 없이 MMIO 폴링(`main.cpp:78-97`) → 호스트 CPU 점유·레이턴시.
- 코드 성숙도: 하드코딩 경로(`CMakeLists.txt:5`), 다수 주석 처리 코드, `PE.sv`의 `output_o` 리셋/IP 출력 충돌 가능성 등 학생 프로젝트 수준의 거칠음.

**리스크**
- v2.0의 ViT 전체(어텐션/softmax/LN/GELU) FPGA 통합 경로는 확인 불가 — 사실상 **GEMM 가속기 + 벤치마크**까지가 검증 가능 범위.
- 200 MHz(example.py) vs 100 MHz(IP) 불일치로 보고된 처리량 수치의 신뢰도 주의.

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer + XR 시선추적) 관점 시사점

우리 목표는 **고처리량·저지연 ViT/Transformer FPGA 가속**(HG-PIPE 계열의 파이프라인형) + **XR 시선추적**이므로, 본 학생 프로젝트는 "참고용 기준선"으로서 다음이 재사용 가능하다.

1. **FP systolic PE 설계 패턴(재사용 가능, 단 개량 필요)**
   - `pipeline_mac` = "FP 곱셈 IP → FP 누산 IP 직렬" 패턴은 그대로 차용 가능. 단 우리 쪽은 (a) **누산기를 FP32로 승격**(또는 bfloat16 누산)해 긴 K에서 정밀도 확보, (b) Xilinx FP IP 대신 **DSP-패킹/커스텀 FP MAC**으로 II=1 보장, (c) latency 8을 줄이거나 깊은 파이프라인을 채우는 데이터 스케줄링이 필요.
   - `PE.sv`의 valid 게이팅(`input_valid & weight_valid`)·systolic 패스스루(1-cycle 레지스터)는 우리 PE의 기본 골격으로 유용.

2. **Output-Stationary vs 우리 파이프라인형**
   - 본 프로젝트는 OS(출력 고정) + skew 주입. HG-PIPE 계열의 **레이어 파이프라인(전 레이어를 공간 전개)** 과는 데이터플로우가 다르므로, 본 어레이는 단일 GEMM 엔진 블록으로만 흡수하고, 우리는 **여러 GEMM/어텐션 스테이지를 스트리밍 연결**하는 상위 구조를 별도 설계해야 함.

3. **XDMA 호스트 인터페이스(직접 재사용 가치 높음)**
   - `dma_utils.h` + `/dev/xdma0_{h2c,c2h}_0` + MMIO 레지스터 맵(`accelerator.h`) + ctypes(`example.py`)는 **검증된 호스트 브리지 템플릿**. 우리 XR 파이프라인의 (호스트→FPGA 프레임/특징 전송, 결과 회수) 골격으로 거의 그대로 차용 가능. 단 폴링→**인터럽트/AXI4-Stream 연속 스트리밍**으로 교체하고, XR 실시간성을 위해 더블버퍼링/DMA 디스크립터 체이닝 도입 권장.
   - 레지스터-기반 명령/상태 FSM(`accel_instr_e`/`accel_state_e`)은 우리 컨트롤 평면 설계의 좋은 출발점.

4. **ViT→TPU 매핑(호스트 측 타일링/패딩/skew 로직)**
   - `gemm_padded`의 (M,K,N) 패딩 + 16-타일 3중 루프 + transpose/skew 전처리는 **임의 ViT 레이어를 고정 크기 어레이로 던지는 호스트 측 어댑터**로 재사용 가능. XR 시선추적용 경량 ViT(작은 패치/적은 토큰)일수록 패딩 오버헤드가 커지므로, **타일 크기·토큰 수를 시선추적 모델에 맞춰 공동 설계**(예: 어레이를 토큰 차원에 정렬)하는 것이 시사점.

5. **정량 기준선**
   - `example.py`의 처리량/사이클 계산식(`(sys_size**2)*2` 이론 FLOP/cycle, raw compute 시간 측정)은 우리 가속기 벤치마크 하니스의 출발 템플릿. 본 기준선(16×16, FP16, ~100–200 MHz, 단일 어레이)을 **개선 목표 대비 baseline**으로 삼아 HG-PIPE의 파이프라인 처리량 우위를 정량화할 수 있음.

요약: **PE의 FP MAC 결선 패턴과 XDMA 호스트 브리지/타일링 어댑터는 직접 재사용**, 어레이의 OS 데이터플로우·저병렬·FP16 누산·폴링 동기화는 **우리 고처리량 목표에 맞게 교체/개량**해야 함.

---

## 10. 근거 / 한계 표기

- **확정(라인 근거 있음)**: 어레이 16×16(`systolic_array.sv:7-8`), PE의 MAC=Xilinx FP IP 2단 직렬(`pipeline_mac.v:67-89`), **FP16(Half) 곱셈+누산, 각 latency 4**(`pipeline_mac_floating_point_{0,1}_0.xci`), Output-Stationary 전파(`systolic_array.sv:71-79`, TB명 `_OS_tb`), XDMA 디바이스/레지스터 맵/명령·상태 FSM(`main.cpp:40-41`, `accelerator.h:19-51`), 호스트 FP16 타일/패딩/skew(`main.cpp:243-411`, `matrix_utils.h:90-118`), 타깃 **Virtex-7 xc7vx485t / VC707 / PCIe Gen2 x8**(`const.xdc:4-9`, `xpr:10,50`), libTorch+C++23 빌드(`CMakeLists.txt`).
- **추정**: 32비트 포트 폭에 FP16 데이터가 하위 비트로 실린다는 해석(IP Half + AXIS 4바이트 근거의 합리적 추정), MAC 파이프라인 깊이 ≈8(latency 4+4 합산), `systolic_array_wrapper.sv`가 IP 패키징용 스텁이라는 판단, 실제 합성 클럭(100 vs 200 MHz).
- **확인 불가**: v2.0의 `pci_mig_accelerator` 컨트롤러 RTL 소스(레지스터 계약만 존재), v2.0에서 어텐션/softmax/LayerNorm/GELU까지 포함한 ViT 전체 그래프의 FPGA 통합(루트 `code/code_cpu`는 CPU libTorch ViT, FPGA 측은 GEMM 오프로딩만 확인), TB(FP32 shortreal) vs 합성 IP(FP16) 정밀도 정합의 의도, milestone PDF 내부 수치(미열람).
```