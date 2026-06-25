# Transformer-Accelerator-Based-on-FPGA 정밀 분석

> 분석 대상 repo: `REF/Transformer-Accel/Transformer-Accelerator-Based-on-FPGA`
> 분석 방식: 자체 소스(.v / .sv / .cpp / .h / .c / .py / .tcl) 전수 Read 후 라인 근거 기반 분석.
> 근거 표기 규칙: **확인** = 파일명:라인범위 명시 / **추정** = 코드 정황 기반 해석 / **확인 불가** = repo 내 근거 없음.

---

## 1. 개요

이 repo는 **정수(INT8) 양자화 행렬곱(GEMM)을 가속하는 파라미터화된 시스톨릭 어레이(systolic array) 기반 FPGA 가속기**다. 핵심은 Transformer/ViT의 가장 무거운 연산인 선형 변환(Q/K/V projection, FFN, attention score 등 모두 행렬곱)을 weight-stationary 시스톨릭 어레이로 처리하는 것이며, PYNQ-Z1(Zynq-7020) 보드를 타깃으로 한다 (README.md:1-2 / prj.tcl:53).

- **완성된(main) 부분**: 16×16(기본값) 시스톨릭 어레이 + 3단 버퍼(입력 더블버퍼 / 가중치·피처 스테이징 / 출력 누산 버퍼) + AXI4-Lite 제어 + AXI4-Stream 데이터 경로. SystemVerilog 테스트벤치로 소프트 GEMM 결과와 비트 정확 비교 검증 (sim/MM_Ultra_tb.sv:343-376).
- **진행 중(In Progress) 부분**: Transformer의 비선형 연산자 — **Softmax / GELU / LayerNorm(Ln) / Exp** 를 정수 고정소수점으로 근사하는 RTL. README.md:3에서 저자가 "정확도·성능 개선 작업 중"이라고 명시.
- **SW**: 동일 GEMM의 소프트웨어 레퍼런스 및 DMA 드라이버를 Vitis(C++), 베어메탈 SDK(C), PYNQ(Python) 세 가지로 제공.

README.md:3에서 저자가 명시적으로 "ViT(Transformer 기반 신경망) 가속"을 목표로 언급하므로, 이 repo는 **ViT/Transformer FPGA 가속기의 GEMM 코어 + 비선형 연산자 라이브러리 레퍼런스**로 분류된다. (HG-PIPE 계열의 "완전 파이프라인 ViT"와 비교 시, 본 repo는 weight-stationary 시분할 재사용형 GEMM 엔진에 가깝다 — 7장 시사점 참조.)

---

## 2. 디렉토리 구조 (자체 소스 + 제외 항목)

### 2.1 자체 소스 (분석 대상)

```
Transformer-Accelerator-Based-on-FPGA/
├── README.md                       # 프로젝트 설명, 재현 절차
├── prj.tcl                         # Vivado 2019.1 블록디자인 생성 스크립트(자동생성)
├── src/                            # [완성] GEMM 시스톨릭 어레이 RTL
│   ├── PE.v                        # 단일 MAC 처리요소 (weight-stationary)
│   ├── PE_line.v                   # PE 1행(가로 array_n개) — 누산 전파
│   ├── PE_array.v                  # array_m × array_n 2D 어레이 + skew 버퍼
│   ├── MM.v                        # 어레이 래퍼: weight 로드 + feature 스트리밍 제어
│   ├── MM_buffer.v                 # weight/feature 스테이징 버퍼 + FSM
│   ├── MM_in_buffer.v              # 입력 더블버퍼(블록 단위 재사용)
│   ├── MM_out_buffer.v            # 출력 누산 버퍼(부분합 가산) + 우시프트 양자화
│   ├── AdderS.v                    # 포화(saturating) 벡터 가산기
│   ├── right_shifter.v             # 반올림·포화 우시프트(재양자화)
│   ├── MM_ultra.v                  # 3버퍼+어레이 통합 top (datapath)
│   ├── MM_ultra_top.v              # AXI 래퍼 인스턴스화 top
│   └── MM_ultra_axi.v              # AXI4-Lite 슬레이브 + AXI-Stream 연결
├── sim/
│   ├── MM_Ultra_tb.sv              # GEMM 검증 테스트벤치(soft vs hard 비트비교)
│   └── MM_Ultra_tb_behav.wcfg      # 파형 설정(데이터, 분석 비대상)
├── In Progress/                    # [미완] 비선형 연산자 RTL
│   ├── src/
│   │   ├── Softmax.v               # 정수 Softmax 코어(max-shift, exp, ln-sum)
│   │   ├── Softmax_control.v       # Softmax 3-pass 제어 + 버퍼
│   │   ├── Softmax_top.v / _top_axi.v  # AXI 래퍼
│   │   ├── Exp_module.v            # 2^x 기반 지수 근사 (latency=3)
│   │   ├── Ln_module.v             # log2 기반 자연로그 근사 (latency=2)
│   │   ├── gelu.v                  # 정수 GELU 근사 (구간별 다항식)
│   │   ├── lin.v                   # GELU 보조 2차 다항식 (latency=4)
│   │   ├── EightGelus.v            # GELU SIMD lane (num_gelu 병렬)
│   │   └── Gelus_top.v / Gelus_axi.v   # AXI 래퍼
│   └── sim/
│       ├── gelu_tb.sv              # GELU 테스트벤치
│       └── Softmax_top_tb.sv       # Softmax 테스트벤치
├── vitis/                          # Vitis(C++) 호스트: soft/hard GEMM + DMA 드라이버
│   ├── main.cpp, Matrix.cpp, Matrix.h, Defines.h
│   ├── lscript.ld, Xilinx.spec, README.txt
├── sdk/                            # 베어메탈 SDK(C): 동일 GEMM 레퍼런스
│   ├── main.c, matrix.c, matrix.h, defines.h, README.txt
└── pynq/
    └── MM.py                       # PYNQ 호스트: numpy 레퍼런스 + MMIO DMA 제어
```

### 2.2 제외 항목 (third-party / 생성물 — 이름만 언급, 미분석)

- `.git/` (버전관리 메타데이터: HEAD, index, objects/pack, hooks/*.sample 등) — 생성물.
- `prj.tcl` 내 **PS7(processing_system7) MIO/PLL 자동생성 블록**(prj.tcl:336-707+ 대량의 `CONFIG.PCW_*`) — Vivado 보드프리셋 자동생성이라 라인 단위 분석에서 제외하고 토폴로지만 인용.
- `prj.tcl`이 의존하는 Xilinx IP(아래 vendor IP) — repo에 소스 없음:
  `axi_dma:7.1`, `smartconnect:1.0`, `axis_dwidth_converter:1.1`, `processing_system7:5.5`, `proc_sys_reset:5.0` (prj.tcl:134-138).
- `sim/MM_Ultra_tb_behav.wcfg` — 시뮬레이터 파형 GUI 설정(데이터).
- `In Progress/sim/gelu_tb.sv`, `Softmax_top_tb.sv` — 시간 제약상 본 분석에서 **미정독(스킵)**, 존재만 명기.

---

## 3. 핵심 모듈 정밀 분석 (라인 근거)

GEMM 데이터패스 계층: `MM_ultra_axi` → `MM_ultra` → { `MM_in_buffer`, `MM_buffer`(→`MM`→`PE_array`→`PE_line`→`PE`), `MM_out_buffer`(→`AdderS`,`right_shifter`) }.

### 3.1 PE.v — 단일 MAC 처리요소 (weight-stationary)

가장 안쪽의 곱셈-누산 단위. **가중치를 내부 레지스터에 고정(stationary)** 하고 입력 액티베이션을 흘려보내는 전형적인 weight-stationary PE.

- 파라미터: `data_width=8`(INT8), `array_m=16`, `array_n=16`, `log2_array_m=4` (PE.v:5-8).
- 포트: `x_in`(액티베이션, signed 8b), `w`(가중치), `psum_in`/`psum_out`(부분합, 폭 `2*data_width+log2_array_m-1:0` = 20b), `x_out`(다음 PE로 전달), `set_w`(가중치 래치 enable) (PE.v:11-19).
- 핵심 동작(PE.v:28-33):
  - `if(set_w) reg_w <= w;` → 가중치 프리로드.
  - `psum_out <= psum_in + x_in*reg_w;` → **곱셈-누산 1 클럭**.
  - `x_out <= x_in;` → 액티베이션을 가로(systolic) 방향으로 1 클럭 지연 후 전달.
- 누산 비트폭 근거: 8b×8b=16b, array_m=16개 누적 → +4b(log2 16) = 20b. psum 폭 정의(PE.v:16,18)가 이를 정확히 반영. 오버플로 없는 정밀 누산 설계. **확인** (PE.v:16-18,31).

### 3.2 PE_line.v — PE 1행 (가로 array_n개 체인)

`array_n`개의 PE를 가로로 연결. 액티베이션 `x`가 행을 따라 좌→우로 흐르고(`x_array[i] → x_array[i+1]`), 각 PE는 자기 열의 가중치/부분합을 담당.

- packed 버스를 PE별 배열로 분해: `w_array[i] = w_packed[data_width*i +: data_width]` (PE_line.v:30), psum in/out도 동일 슬라이싱(PE_line.v:31-32).
- `assign x_array[0]=x;` 입력 액티베이션을 행 첫 PE에 주입, 이후 `x_out`→`x_in` 체인(PE_line.v:25,48-52).
- generate로 `array_n`개 PE 인스턴스(PE_line.v:36-55). **확인**.

### 3.3 PE_array.v — 2D 시스톨릭 어레이 (skew 버퍼 포함)

`array_m × array_n` 2차원 어레이. 세로 방향으로 부분합이 누적 전파되고, **입력 skew(대각 정렬)와 출력 de-skew를 위한 시프트 버퍼**가 핵심.

- 부분합 세로 전파: `psum_in_packed_array[0]=0`(PE_array.v:61), 각 행이 `psum_in_packed_array[i] → [i+1]`로 누적(PE_array.v:88,124), 최종 출력 `psum_in_packed_array[array_m]`(PE_array.v:31-32).
- **입력 skew 버퍼**(PE_array.v:91-108): 행 i(≥1)마다 길이 i의 시프트 레지스터 `x_buf[i-1:0]`를 두어 액티베이션을 i 클럭 지연시킨 뒤 그 행에 주입. 시스톨릭 어레이의 대각 입력 타이밍을 맞추는 표준 기법.
- **출력 de-skew 버퍼**(PE_array.v:35-59): 열 i마다 길이 `array_n-2-i`의 출력 버퍼 `out_buf`로 열별 도착 시점 차이를 보정해 `PE_out_packed`로 동시 정렬. 마지막 열(i==array_n-1)은 버퍼 0(PE_array.v:36-38).
- x/w packed 분해(PE_array.v:64-69). **확인**. 이 모듈이 시스톨릭 타이밍의 핵심이며, 가장 정밀하게 봐야 할 부분.

### 3.4 MM.v — 어레이 래퍼 (가중치 로드 + 피처 스트리밍 제어)

`PE_array`를 감싸 **가중치 행렬을 시프트 체인으로 적재**하고, 그 후 피처를 흘려 결과를 뽑는 제어 래퍼.

- weight_buffer: `array_m`개 깊이의 시프트 레지스터(MM.v:50). 가중치 입력 시 `weight_buffer[0]<=weight_in; weight_buffer[j]<=weight_buffer[j-1]` (MM.v:128-147).
- weight/feature 다중화: `wdata_flag`로 같은 입력 포트(`MM_in_data`)를 가중치/피처로 분기(MM.v:75-76,95-104). `wdata_flag_up`/`MM_in_last`의 2비트 case로 토글(MM.v:99-103).
- `set_w` 생성: 가중치 카운터가 array_m 도달 시 1 클럭 펄스 → PE 어레이에 가중치 일괄 래치(MM.v:77,84-93).
- 출력 valid/last를 `2*array_m` 길이 시프트 레지스터로 지연(MM.v:51-52,155-182) → 어레이 파이프라인 latency 보상.
- 주석 처리된 `right_shifter` 블록(MM.v:201-216) — 양자화 시프트를 어레이 직후가 아니라 **out_buffer 단계로 이동**시킨 흔적. `assign data_out = PE_out_packed;`로 부분합을 그대로 출력(MM.v:218). **확인**.

### 3.5 MM_buffer.v — weight/feature 스테이징 버퍼 + 3-state FSM

`MM`에 공급할 가중치/피처를 BRAM에 저장하고, "둘 다 준비되면(both_full) 계산 시작" 핸드셰이크를 관리.

- FSM 3상태: `IDLE / SET_WEIGHT / SET_FEATURE` (MM_buffer.v:2-4,219-236).
- 듀얼 버퍼: `weight_buffer[weight_buffer_depth]`, `feature_buffer[feature_buffer_depth]` (MM_buffer.v:56,60). 깊이는 `2^(폭)`로 산정(MM_buffer.v:37-39).
- `both_full` 조건으로 start 생성(MM_buffer.v:92,104) → 가중치 먼저 MM에 push, 그 다음 피처(MM_buffer.v:107-130 case 디코드).
- 가중치 주소 계산: `input_weight_addr = input_weight_row * num_blobk_W + input_weight_col` (MM_buffer.v:103) → 블록열 단위로 가중치를 재정렬해 어레이에 공급. **이것이 블록 GEMM 타일링의 핵심 인덱싱.**
- `w_end`/`total_last`로 한 가중치-타일에 대한 전체 피처 패스 종료 판단(MM_buffer.v:96,176-185). **확인**.

### 3.6 MM_in_buffer.v — 입력 더블버퍼 (블록 단위 가중치 재사용)

호스트(DMA)에서 들어오는 피처/가중치 스트림을 BRAM 배열에 적재하고, **가중치를 고정한 채 피처를 여러 출력열 블록에 재사용**하도록 주소를 발생.

- 두 BRAM: `in_F_array[IN_Feature_Block_num]`, `in_W_array[Weight_Block_num]` (MM_in_buffer.v:67-68).
- `clogb2` 함수로 주소폭 자동 산정(MM_in_buffer.v:49-64).
- 출력 피처 주소: `out_F_addr = out_F_row_addr * F_width_block_num + out_F_col_addr` (MM_in_buffer.v:91). col_addr를 0..F_width_block_num로 증가시키며 **같은 가중치 블록에 대해 피처를 행 단위로 반복 공급**(MM_in_buffer.v:223-232) → 데이터 재사용으로 DMA 대역폭 절감.
- 상태머신 `IDLE→IN_DATA→CAL`(MM_in_buffer.v:130-142). `start` 조건은 "F·W 둘 다 적재 완료 또는 직전 버퍼 출력 종료 & 아직 처리할 col 남음"(MM_in_buffer.v:92-94). **확인**. 이중버퍼링으로 적재와 계산을 오버랩하려는 의도(추정: CAL 중 다음 블록 적재 차단 로직 `state!=CAL`, MM_in_buffer.v:89-90).

### 3.7 MM_out_buffer.v — 출력 누산 버퍼 + 재양자화

부분합(20b급)을 누적 BRAM에 더해가며(블록 GEMM의 K 방향 누산), 완료 시 우시프트로 INT8 재양자화하여 출력.

- 누산 BRAM: `(*ram_style="block"*) F_array[OUT_Feature_Block_num]`, 폭 `A_size*OUT_MEM_WIDTH` (MM_out_buffer.v:47). OUT_MEM_WIDTH=21~32로 확장 비트폭 누산.
- 누산기: 들어온 부분합 A를 기존 B(BRAM)에 더해 C로(MM_out_buffer.v:85,217-258) → `AdderS` 인스턴스(MM_out_buffer.v:300-309). read-modify-write 누산 구조.
- A 부호확장: `A_in_adder[i] = $signed(A[i*(log2_array_m+data_width*2)...])` (MM_out_buffer.v:311-315).
- 출력 시 재양자화: `right_shifter` per-lane 인스턴스(MM_out_buffer.v:317-330), `shift`만큼 우시프트 후 INT8 포화.
- 다단 valid 지연(delay1/delay2)으로 BRAM read latency 보상(MM_out_buffer.v:60-66,107-187). **확인**. 이 모듈은 블록-GEMM 누산의 정확성과 타이밍을 동시에 책임지는 가장 복잡한 제어 블록.

### 3.8 AdderS.v — 포화 벡터 가산기

`A_size`개 레인을 병렬 가산하되 **부호 오버플로를 포화 처리**.

- 더블 부호비트 확장: `temp[i] = {A[부호], A} + {B[부호], B}` (AdderS.v:23-26).
- 포화 판정: `temp[i][data_width:data_width-1]` 상위 2비트가 `01`이면 양 포화(최대), `10`이면 음 포화(최소), 그 외 정상(AdderS.v:30-37). **확인**. 누산 시 INT 오버플로를 막는 표준 클리핑 가산기.

### 3.9 right_shifter.v — 반올림·포화 우시프트 (재양자화 핵심)

부분합(before_data_width)을 shift만큼 우시프트하여 after_data_width(INT8)로 재양자화. **반올림 + 양/음 포화**.

- 산술 우시프트: `temp1_out = data_in >>> shift` (right_shifter.v:22).
- 반올림: `temp2_out = data_in[shift-1] ? temp1_out+1 : temp1_out` (잘려나가는 최상위 비트로 round-half-up, right_shifter.v:23).
- 포화 검출: `under_min`/`over_max`를 시프트 후/전(shift==0) 두 경로로 분기 계산(right_shifter.v:25-29), case로 최소/최대/정상 출력(right_shifter.v:31-45). **확인**. INT8 양자화 가속기에서 누산 후 스케일링의 정석 구현.

### 3.10 MM_ultra.v / MM_ultra_top.v / MM_ultra_axi.v — 통합 + AXI

- **MM_ultra.v**: `MM_in_buffer + MM_buffer + MM_out_buffer`를 연결한 데이터패스 top(MM_ultra.v:118-220). `clogb2(A_size)`로 log2_array_m 자동 산정(MM_ultra.v:55). 런타임 파라미터(shift, F_length, F/W_width_block_num)를 변화 감지 후 래치(MM_ultra.v:73-98). **확인**.
- **MM_ultra_axi.v**: 표준 AXI4-Lite 슬레이브(슬레이브 레지스터 4개) + 사용자 로직. **레지스터 매핑**(MM_ultra_axi.v:427-465):
  - `slv_reg0 → shift_in`, `slv_reg1 → F_length_in`, `slv_reg2 → F_width_block_num_in`, `slv_reg3 → W_width_block_num_in`.
  - AXI-Stream: `s0_axis(피처 in)`, `s1_axis(가중치 in)`, `m0_axis(결과 out)` → 각각 in_F/in_W/out_data로 연결(MM_ultra_axi.v:451-464). 폭 = `array_size*data_width` (예 16×8=128b, 또는 24×8=192b).
- **MM_ultra_top.v**: `MM_ultra_axi` 단일 인스턴스 래퍼(MM_ultra_top.v:69-123). 기본 `array_size=24`(MM_ultra_top.v:6) — 단, README/tb는 16 사용. **확인** (기본값 불일치는 5장 참조).

### 3.11 비선형 연산자 (In Progress)

#### Exp_module.v — 정수 지수 (latency=3)
2^x 분해법으로 e^x 근사. `e^x = 2^(x·log2(e))`.
- `x_log2e = x_S9Q10 * 23` → 23 ≈ log2(e)·2^4 (Q4 스케일, Exp_module.v:8-10).
- 절댓값 후 정수부/소수부 분리(Exp_module.v:16-21): 정수부는 `2^(-int)` = `>> x_int`로(Exp_module.v:28-30), 소수부는 1차 근사 `1 - 0.5·frac`(Exp_module.v:23).
- 두 항 곱으로 최종 지수, 포화(Exp_module.v:36-46). **확인**. 곱셈기 1~2개로 구현하는 경량 지수 근사.

#### Ln_module.v — 정수 자연로그 (latency=2)
log2 기반 자연로그 근사. 최상위 1의 위치(w)로 지수부 추출 후 가수부 선형 보정.
- priority encoder로 leading-1 위치 w와 정규화 가수 k 산출(Ln_module.v:8-47).
- `P = {w,k} * 11` (11 ≈ ln2 관련 스케일, Ln_module.v:50-61). **확인**. Softmax의 log-sum-exp 안정화에 사용(3.11 Softmax 참조).

#### lin.v — GELU 보조 2차 다항식 (latency=4)
GELU 중심 구간을 `a·(|x|-b)^2 + c` 형태 2차식으로 근사.
- `b=7/4`, `a=-37/128`(a_0Q7)로 `l = a·(|x|-b)^2 + 1.0` 계산(lin.v:19-33), 부호 복원(lin.v:39-49). **확인**.

#### gelu.v — 정수 GELU (latency=4+4)
구간 분할 근사: 작은 음수→0, 큰 양수→x(통과), 중심 구간→`0.5·x·(L+1)` 형태.
- 입력 스케일 정규화 `x_in_8Q7`(gelu.v:51-58).
- 3구간 분기(gelu.v:96-107): `x<0 & x≤-2.5`→0, `x>0 & x≥2.5`→x, 그 외→`lin` 출력으로 다항식.
- `lin` 인스턴스(gelu.v:83-87), 출력 `0.5·L·x`(gelu.v:88-94). **확인**.

#### EightGelus.v — GELU SIMD lane
`num_gelu`(기본4)개의 `gelu`를 병렬, AXI-Stream valid/last/keep을 9단(=gelu latency) 시프트로 정렬(EightGelus.v:32-105), per-lane `gelu` 인스턴스(EightGelus.v:107-116). **확인**.

#### Softmax.v / Softmax_control.v — 정수 Softmax (3-pass)
Softmax를 **max-subtract → exp-sum → normalize**의 3 패스로 스트리밍 처리.
- `stage` 1/2/3으로 패스 구분(Softmax.v:78). 1패스: running max(Softmax.v:172-179). 2패스: `e_sum` 누적(Softmax.v:202-209, Exp_module 1번 인스턴스). 3패스: `exp(x - max - ln(sum))`로 정규화(Softmax.v:75-76, Exp_module 2번 + Ln_module, Softmax.v:214-233).
- 출력 스케일(scale_out 7~14)로 Q포맷 절단·포화(Softmax.v:85-105).
- `Softmax_control.v`: 입력을 BRAM(`in_data_buffer[1023:0]`)에 저장 후 3회 재독출(Softmax_control.v:38,63-66,169-176), length×3 카운터로 패스 진행(Softmax_control.v:87-109). **확인**. length≤1023 제약(Softmax_top_axi.v:426).

#### AXI 래퍼 (Softmax_top/_axi, Gelus_top/_axi)
표준 AXI4-Lite 슬레이브(reg0~3)로 런타임 파라미터 전달. Softmax: reg0=length, reg1=scale_in, reg2=scale_out(Softmax_top_axi.v:426-428). GELU: reg0=scale(Gelus_axi.v:436). **확인**.

---

## 4. 데이터 플로우

### 4.1 GEMM 경로 (C = A × B, A=피처, B=가중치)
1. **호스트**가 A(피처)·B(가중치)를 `A_SIZE`의 배수로 zero-padding하여 DDR에 배치(Matrix.cpp:13-16 / MM.py:13-26).
2. 호스트가 AXI-Lite로 shift / F_length / F_width_block_num / W_width_block_num 레지스터 설정(Matrix.cpp:138-141 / MM.py:148-151).
3. **3개 DMA**: 가중치 MM2S → s1_axis, 피처 MM2S → s0_axis, 결과 S2MM ← m0_axis. 수신 채널을 먼저 열고 송신 시작(Matrix.cpp:144-157).
4. **MM_in_buffer**가 스트림을 BRAM에 적재 → **MM_buffer**가 both_full 시 가중치 시프트-로드 후 피처 스트리밍 → **PE_array**가 weight-stationary MAC 수행.
5. **MM_out_buffer**가 K-블록 부분합을 누산(AdderS) → 완료 시 `right_shifter`로 INT8 재양자화 → m0_axis로 송출.
6. 호스트가 S2MM 완료(IDLE 비트) 폴링 후 결과 회수(Matrix.cpp:160-162 / MM.py:104-105).

### 4.2 비선형 경로 (In Progress)
입력 AXI-Stream → (Softmax: BRAM 버퍼 후 3-pass / GELU: 직접 SIMD) → AXI-Stream 출력. 스칼라 파라미터는 AXI-Lite. **추정**: GEMM 코어와 별개 IP로 설계되어, 시스템 통합 시 데이터무브먼트(DMA chaining)는 별도 구성 필요(repo에 통합 BD 근거 없음 → **확인 불가**).

---

## 5. HW/SW 매핑

| 계층 | 구현 | 근거 |
|---|---|---|
| HW: GEMM 코어 | src/*.v (PE→…→MM_ultra) | src/PE.v:31, MM_ultra.v:118-220 |
| HW: AXI 제어 | MM_ultra_axi.v reg0-3 | MM_ultra_axi.v:427-465 |
| HW: 비선형 | In Progress/src/*.v | Softmax.v, gelu.v 등 |
| SW: 레퍼런스 GEMM (검증) | Matrix_mul_soft / mat_mul / mat_mul_soft | Matrix.cpp:67-102, matrix.c:47-83, MM.py:42-46 |
| SW: HW 드라이버 | Matrix_mul_hard (DMA+레지스터) | Matrix.cpp:104-163 |
| SW: PYNQ MMIO 드라이버 | ini_MM / mat_mul / *_transfer | MM.py:48-155 |
| 검증(시뮬) | MM_soft task vs z_hard 비트비교 | sim/MM_Ultra_tb.sv:28-58, 343-376 |

- 양자화 규약 일치: soft/hard 모두 누산 후 `(temp + (1<<(shift-1))) >> shift` 반올림 + [-128,127] 포화(Matrix.cpp:90-97, sim/MM_Ultra_tb.sv:46-53). RTL `right_shifter`의 round-half-up·포화와 정확히 대응. **확인**.
- 주소 정렬 책임은 SW에 위임: `real_cols = ceil(cols/A_SIZE)*A_SIZE` 패딩(Matrix.cpp:13-14, MM.py:16-24). 호스트가 패딩·블록화를 처리하고 HW는 블록 단위만 본다.
- PYNQ vs Vitis 레지스터 오프셋 동일: shift=0x0, FL=0x4, FWBN=0x8, WWBN=0xc (MM.py:148-151, Defines.h:11-14). **확인**.

---

## 6. 빌드·실행 / 의존성

### 6.1 빌드 (README.md:5-11, prj.tcl)
1. Vivado **2019.1**에서 PYNQ-Z1 보드파일로 신규 프로젝트(prj.tcl:23,53-54).
2. 모든 RTL 추가 후 `prj.tcl` 실행 → 블록디자인 `design_1` 생성(prj.tcl:60, 196-).
3. block design wrapper 생성 후 top 지정 → synth/impl → bitstream(README.md:9-11).

### 6.2 시스템 토폴로지 (prj.tcl)
- 타깃: `xc7z020clg400-1`(Zynq-7020) (prj.tcl:53).
- **3× axi_dma**(prj.tcl:253-296): dma_0/dma_1 = MM2S(피처/가중치 송신, 64b), dma_2 = S2MM(결과 수신). MM2S DRE 활성, SG 미사용.
- **3× axis_dwidth_converter**(prj.tcl:316-332): 64b DMA ↔ 어레이 폭(24B=192b, 결과 8B). array_size=24 기준(M_TDATA_NUM_BYTES 24/24/8).
- **3× smartconnect**, **processing_system7**(FCLK0=100MHz, prj.tcl:382,521), **proc_sys_reset**.
- MM_ultra_top 인스턴스 파라미터: array_size=24, Weight/in_feature/out_feature_block_num=2500 (prj.tcl:244-250). **확인**.

### 6.3 SW 빌드 (Vitis/SDK)
- Vitis C++: `xil_types.h, xil_io.h, xil_cache.h, xtime_l.h`, `xparameters.h`, `xaxidma_hw.h` 의존(Matrix.cpp:1-8, Defines.h:1-4). 베어메탈 Xilinx BSP 필요.
- SW측 A_SIZE=16, block_num=4096 (Defines.h:36-39) — **HW(tcl)의 array_size=24와 불일치**(5장).
- PYNQ: `pynq.allocate`, `pynq.MMIO`, numpy(MM.py:1-3). A_SIZE=25 (MM.py:5) — **또 다른 값**.

### 6.4 의존성 요약
- HW: Xilinx IP(axi_dma, smartconnect, axis_dwidth_converter, processing_system7, proc_sys_reset) — vendor, repo 외부.
- SW: Xilinx 베어메탈 BSP(Vitis/SDK) 또는 PYNQ 런타임 + numpy.

---

## 7. 강점 · 한계

### 7.1 강점
- **완전 파라미터화**: array 크기, data_width, 버퍼 깊이, 블록수가 모두 파라미터. `clogb2`로 주소폭 자동 산정(MM_in_buffer.v:49-64).
- **정확한 양자화 규약 일치**: SW soft GEMM ↔ RTL이 round-half-up·포화까지 비트 정확(검증 tb 존재, sim/MM_Ultra_tb.sv:351). 양자화 가속기로서 신뢰성 높음.
- **블록 GEMM + 데이터 재사용**: 가중치 고정 후 피처 다중 재사용(MM_in_buffer.v:223-232), 출력 누산 BRAM으로 K-타일링(MM_out_buffer.v) → 임의 크기 행렬 지원.
- **세 가지 호스트 인터페이스**(Vitis/SDK/PYNQ) 제공 → 재현·이식 용이.
- **비선형 연산자 전부 정수 근사** 시도(Softmax/GELU/LayerNorm용 Exp/Ln) → end-to-end 정수 Transformer 지향.

### 7.2 한계
- **비선형부 미완성**: README.md:3 및 "In Progress" 폴더 명시. 정확도/성능 미검증, GEMM 코어와의 시스템 통합 BD 부재(**확인 불가**).
- **파라미터 불일치(이식 함정)**: array_size가 RTL 기본 24(MM_ultra_top.v:6) / tcl 24(prj.tcl:246) / tb 16(sim/MM_Ultra_tb.sv:3) / SDK 16(Defines.h:36) / PYNQ 25(MM.py:5)로 제각각. 사용자가 일관되게 맞춰야 함. **확인**.
- **단정밀 INT8 고정**: data_width 파라미터지만 SW DATA_TYPE=s8 고정(Defines.h:6), 혼합정밀/FP 미지원.
- **LayerNorm 본체 부재**: Ln_module은 자연로그 근사일 뿐 평균/분산 정규화 LayerNorm RTL은 없음(Softmax 내부 log-sum-exp 용도). README의 "LayerNorm 가속" 목표는 **미구현으로 추정**.
- **클럭 100MHz**(prj.tcl:521)로 보수적. 고처리량(HG-PIPE급 GHz·완전파이프라인)과는 거리.
- **코드 주석 다수 깨짐**(중국어 인코딩 깨짐, 예 MM.v:69, right_shifter.v:2) → 가독성 저하.
- 비선형 testbench(gelu_tb.sv, Softmax_top_tb.sv) 미정독(본 분석 스킵).

---

## 8. 우리 프로젝트(고처리량 ViT/Transformer FPGA 가속기 + XR 시선추적) 시사점

> 전제: 우리 목표는 HG-PIPE 계열의 고처리량 ViT 가속기에 XR 시선추적을 얹는 것(분석자 제공 가정).

- **재사용 가능 컴포넌트**:
  - `right_shifter.v`(round-half-up + 포화 재양자화)와 `AdderS.v`(포화 누산)는 우리의 INT8/혼합정밀 데이터패스에 거의 그대로 차용 가능한 검증된 빌딩블록. **확인 근거 존재**.
  - `Exp_module`/`Ln_module`/`gelu`/`lin`의 **곱셈기 최소화 정수 근사 기법**(2^x 분해, leading-1 log2, 구간별 다항식)은 ViT의 Softmax/GELU 하드웨어화 레퍼런스로 유용. 단 정확도 검증은 자체 수행 필요.
- **아키텍처 대비**:
  - 본 repo는 **weight-stationary 시분할 재사용형 GEMM 엔진**(가중치 로드 후 피처 스트림, 출력 BRAM 누산)이다. HG-PIPE의 **완전 파이프라인(레이어별 전용 PE, 중간 BRAM 스트리밍)** 과 패러다임이 다르다. 고처리량(저지연 XR)에는 본 repo의 시분할 재사용보다 HG-PIPE식 공간 매핑이 유리. 본 repo는 **자원 적은 보드에서 임의 크기 GEMM을 돌리는 유연성** 측면 레퍼런스로 위치지음이 타당(추정).
- **XR 시선추적 관점**: 100MHz/PYNQ-Z1/시분할 구조는 XR의 ms급 저지연·고프레임 요구에 직접 부합하지 않음. 다만 **정수 비선형 연산자 + 양자화 규약 일치 검증 방법론**(soft vs hard 비트비교 tb, sim/MM_Ultra_tb.sv)은 우리 검증 파이프라인에 그대로 적용 가능한 좋은 패턴.
- **이식 시 주의**: array_size 등 파라미터를 전 계층(RTL/tcl/SW)에서 일치시키는 single-source-of-truth가 필요(본 repo의 불일치가 교훈).

---

### 부록: 근거 표기 인덱스 (핵심)
- weight-stationary MAC: PE.v:29-33 **확인**
- 시스톨릭 skew/de-skew: PE_array.v:35-59, 91-108 **확인**
- 블록 GEMM 타일 인덱싱: MM_buffer.v:103, MM_in_buffer.v:91, MM_out_buffer.v:88 **확인**
- 재양자화(round+saturate): right_shifter.v:22-45, Matrix.cpp:90-97 **확인**
- AXI 레지스터 맵: MM_ultra_axi.v:445-448, Defines.h:11-14, MM.py:148-151 **확인**
- DMA 토폴로지: prj.tcl:253-296, Matrix.cpp:144-157 **확인**
- 정수 비선형 근사: Exp_module.v:8-46, Ln_module.v:50-61, gelu.v:96-107 **확인**
- LayerNorm 본체 부재 / 시스템 통합 BD 부재: **확인 불가**
- 비선형 testbench(gelu_tb/Softmax_top_tb): 미정독 **(스킵)**
