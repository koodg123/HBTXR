# Transformer-Accelerator-Based-on-FPGA 정밀 분석

> 대상 repo: `REF/ViT-Accelerator/Transformer-Accelerator-Based-on-FPGA`
> 분석 방식: 실제 소스 Read 후 `파일:라인` 근거 기반. third-party(Xilinx IP, prj.tcl 자동생성 BD) 제외, "In Progress" 폴더는 미완성으로 명시.
> 보드: PYNQ-Z1 (Zynq-7000 `xc7z020clg400-1`), Vivado 2019.1 (`prj.tcl:23,53-54`).

---

## 1. 개요

이 프로젝트는 **양자화(INT8) 행렬곱을 가속하는 systolic PE array 기반 FPGA 가속기**다. README(`README.md:1-3`)에 따르면 PYNQ-Z1(또는 임의 Zynq) 위에서 동작하며, systolic array 크기를 파라미터화했고 현재 기본값은 16×16(`README.md:2`)이다. ViT(Vision Transformer)를 비롯한 Transformer 계열 추론의 핵심 연산인 **GEMM(General Matrix Multiply)** 을 하드웨어로 처리하는 것이 목적이며, Softmax/GELU/LayerNorm 같은 비선형 연산자는 `In Progress` 폴더에서 별도로 진행 중이다(`README.md:3`).

핵심 구성:
- **완성된 GEMM 가속기**: `src/*.v` 12개 모듈. AXI4-Lite(제어) + 3×AXI4-Stream(weight/feature_in/feature_out) 인터페이스. 출력 단에서 right_shifter로 INT8 재양자화 수행.
- **미완성 비선형 연산자(In Progress)**: Softmax(LUT 기반 exp/ln), GELU(2차 다항식 근사), LayerNorm용 `Ln_module`. RTL 자체는 동작 수준이나 README가 "정확도/성능 개선 중"이라 명시(`README.md:3`).
- **SW 3종**: Vitis HLS/베어메탈 참조 검증(`vitis/`), 베어메탈 SDK 드라이버(`sdk/`), PYNQ Python 호스트(`pynq/MM.py`).

데이터 흐름의 핵심 특징은 **weight-stationary systolic array**(가중치를 PE에 적재 후 feature 스트리밍) + **block 단위 타일링**(`A_size` 배수 패딩) + **누적은 외부 32비트, 출력은 8비트로 right-shift 재양자화**다.

---

## 2. 디렉토리 구조

```
Transformer-Accelerator-Based-on-FPGA/
├─ README.md                      # 목적·보드·재현 절차
├─ prj.tcl                        # Vivado 2019.1 BD 자동생성 스크립트(MM_ultra_top + 3×AXI DMA + PS7)
├─ src/                           # ★ 완성된 GEMM 가속기 RTL (핵심)
│   ├─ MM_ultra_top.v             # 최상위 래퍼 (AXI4-Lite + AXIS 포트)
│   ├─ MM_ultra_axi.v             # AXI4-Lite slave(slv_reg0~3) + MM_ultra 인스턴스
│   ├─ MM_ultra.v                 # in_buffer → MM_buffer → out_buffer 계층 결선
│   ├─ MM_in_buffer.v             # 입력 weight/feature BRAM 버퍼 + 재배열(타일 주소)
│   ├─ MM_buffer.v                # weight/feature 스테이징 + MM(PE) FSM 구동
│   ├─ MM.v                       # weight 적재(set_w) + feature 스트리밍 + PE_array 연결
│   ├─ PE_array.v                 # array_m×array_n PE 격자 + skew(대각) 버퍼
│   ├─ PE_line.v                  # 한 행(array_n개 PE)
│   ├─ PE.v                       # 단일 MAC 셀(psum_out = psum_in + x*w)
│   ├─ MM_out_buffer.v            # ★ 부분합 누적(BRAM) + AdderS + right_shifter 재양자화
│   ├─ AdderS.v                   # saturating 벡터 가산기
│   └─ right_shifter.v            # ★ round + saturate 우측시프트(재양자화)
├─ sim/
│   └─ MM_Ultra_tb.sv             # GEMM 전체 테스트벤치(soft 모델 vs RTL 비교)
├─ vitis/                         # HLS/베어메탈 참조 검증 (Vivado HLS/Vitis)
│   ├─ Matrix.cpp / Matrix.h      # Matrix 클래스, soft/hard 곱 + 비교
│   ├─ main.cpp                   # 랜덤 행렬 soft vs hard 검증
│   └─ Defines.h                  # 레지스터 맵, A_SIZE=16, DATA_TYPE=s8
├─ sdk/                           # Zynq 베어메탈 SDK 드라이버
│   ├─ matrix.c / matrix.h        # malloc 기반 matrix + soft 곱
│   ├─ main.c                     # 10회 반복, soft vs hard 타이밍 측정
│   └─ defines.h                  # 레지스터 맵, A_SIZE=24
├─ pynq/
│   └─ MM.py                      # PYNQ 호스트: allocate + MMIO + 3 DMA 구동
└─ In Progress/                   # ⚠ 미완성 비선형 연산자 (README.md:3)
    ├─ src/
    │   ├─ Softmax.v / Softmax_top.v / Softmax_top_axi.v / Softmax_control.v
    │   ├─ Exp_module.v           # 2^x 분해 기반 exp
    │   ├─ Ln_module.v            # 자연로그 근사(priority-encoder + 선형)
    │   ├─ gelu.v / lin.v         # GELU 2차근사 + |x| 다항식 보조
    │   ├─ EightGelus.v / Gelus_top.v / Gelus_axi.v  # GELU SIMD 래퍼 + AXIS
    └─ sim/
        ├─ Softmax_top_tb.sv / gelu_tb.sv
```

핵심(섹션 3에서 정밀 분석): `src/` 12개 모듈 전부 + `In Progress/src` 의 Softmax/Exp/Ln/gelu/lin/EightGelus.

---

## 3. 핵심 모듈 정밀 분석 ★

### 3.1 데이터폭 / 양자화 규약 (전 모듈 공통)

- 입출력 데이터: `data_width=8` (signed INT8). `MM_ultra.v:7`, `PE.v:5`.
- PE 내부 누적폭: `2*data_width + log2_array_m` = 16 + log2(array_size). 16×16이면 4비트 → 20비트(`PE.v:18`). 곱(16비트) + array_m개 누적 캐리(log2_array_m)를 담는 폭.
- out_buffer 누적폭: `OUT_MEM_WIDTH` (예 21 또는 32, `MM_ultra.v:12`, `sim/MM_Ultra_tb.sv:9`=21, `vitis/Defines.h`=32). 여러 K-블록(`W_width_block_num`)에 걸친 부분합을 BRAM에 누적하므로 PE 누적폭보다 넓게 잡는다.
- 재양자화: out_buffer 단의 `right_shifter`가 32(또는 21)비트 누적값을 `shift`만큼 **반올림 우측시프트 후 INT8로 saturate** (`right_shifter.v`). `shift` 값은 AXI4-Lite `slv_reg0`로 런타임 주입(`MM_ultra_axi.v:445`).

### 3.2 PE — 단일 MAC 셀 (`src/PE.v`)

입출력(`PE.v:10-19`):
- in: `x_in[7:0]`(feature, signed), `w[7:0]`(weight, signed), `psum_in[2*dw+log2_m-1:0]`(위 행 부분합), `set_w`(가중치 래치), `clk/rst_n`.
- out: `x_out`(아래로 전달할 feature, 1clk 지연), `psum_out`(부분합).

알고리즘(`PE.v:22-34`): 동기 리셋. `set_w`=1이면 `reg_w <= w`로 가중치 적재. 매 클럭 `psum_out <= psum_in + x_in*reg_w`, `x_out <= x_in`. **weight-stationary + output-flowing** 구조다. 즉 가중치는 PE에 고정되고, feature(x)는 행 방향으로 전달되며, 부분합(psum)은 열 방향으로 흘러 내려간다(아래 PE_line/PE_array에서 결선).

### 3.3 PE_line — 한 행 (`src/PE_line.v`)

`array_n`개의 PE를 가로로 연결(`PE_line.v:36-55`). `x`는 `x_array[0]`로 들어와 PE 체인을 따라 `x_array[i+1]`로 전파(`PE_line.v:25,51`) → **feature가 한 행 안에서 좌→우로 systolic 전파**. 각 PE는 자기 열의 `psum_in`을 받아 `psum_out` 생성(`PE_line.v:30-32`). w/psum은 packed 벡터 ↔ 배열로 슬라이싱.

### 3.4 PE_array — systolic 격자 + skew 버퍼 (`src/PE_array.v`)

`array_m`개의 PE_line을 세로로 쌓아 `array_m × array_n` 격자 구성(`PE_array.v:71-128`). systolic array의 핵심인 **대각 지연(skew)** 을 2곳에서 구현:

1. **입력 x skew** (`PE_array.v:91-108`): i번째 행(i≥1)의 feature 입력은 `x_buf[i-1:0]` 시프트 레지스터를 거쳐 **i 클럭 지연** 후 PE_line에 들어간다(`PE_array.v:121`). → 행이 아래로 갈수록 feature가 한 클럭씩 늦게 도착(데이터 대각 정렬).
2. **출력 psum skew** (`PE_array.v:34-60`): 마지막 행에서 나온 열별 부분합(`psum_in_packed_last`)을 열 인덱스별로 `out_buf` 시프트 체인으로 지연시켜(`PE_array.v:40-57`) 모든 열의 결과를 동일 시점에 정렬(de-skew). 마지막 열(`i==array_n-1`)은 지연 0(`PE_array.v:36-38`).

psum 입력 체인 시작은 0(`PE_array.v:61`), array_m번째 출력이 최종 PE_out_packed. 이 구조가 전형적인 **2D weight-stationary systolic array**다.

### 3.5 MM — 가중치 적재 + feature 스트리밍 컨트롤러 (`src/MM.v`)

PE_array를 감싸 시간 다중화로 weight/feature를 주입(`MM.v:184-199`).
- `MM_in_data` 한 포트로 weight와 feature를 시분할. `wdata_flag`가 1이면 weight, 0이면 feature(`MM.v:75-76`). `wdata_flag`는 `wdata_flag_up`/`MM_in_last`로 토글(`MM.v:95-104`).
- weight 적재: `weight_buffer[array_m-1:0]` 시프트 레지스터에 array_m 행을 차례로 밀어넣고(`MM.v:128-147`), array_m개 채워지면 `set_w=1`(`MM.v:77`) → PE_array 전체에 동시에 가중치 래치(`MM.v:195`). w_packed는 행 역순으로 매핑(`MM.v:67-71`, 주석 "주의: 첨자 역순").
- feature: `feature_in_reg1`을 `x_packed`로 PE_array에 공급(`MM.v:78`).
- valid/last 지연 정렬: `MM_out_data_valid_reg_array[2*array_m]`로 **2·array_m 클럭** 파이프라인 지연 보정(`MM.v:51,155-164`) — array 통과 latency 보상.
- 주의: 이 모듈 내 `right_shifter` 인스턴스는 **주석 처리**되어 있고(`MM.v:201-216`) `data_out = PE_out_packed`(`MM.v:218`). 즉 재양자화는 여기가 아니라 MM_out_buffer에서 수행.

### 3.6 MM_buffer — feature/weight 스테이징 + FSM (`src/MM_buffer.v`)

weight/feature를 각각 BRAM(`weight_buffer`, `feature_buffer`, `MM_buffer.v:56,60`)에 모은 뒤, IDLE→SET_WEIGHT→SET_FEATURE 3-state FSM(`MM_buffer.v:2-4,219-236`)으로 MM(PE)에 순서대로 흘려보낸다.
- 양쪽 버퍼가 다 차면(`both_full`, `MM_buffer.v:92`) `start` 펄스 발생 → 계산 시작.
- weight 주소 생성: `input_weight_addr = input_weight_row * num_blobk_W + input_weight_col`(`MM_buffer.v:103`). 즉 weight를 **블록 전치(transpose)** 순서로 읽어 systolic 적재 포맷에 맞춤.
- feature 주소는 `input_feature_addr`로 행 단위 순회(`MM_buffer.v:289-300`).
- `total_last = output_feature_last & w_end`(`MM_buffer.v:96`)로 한 weight 세트 처리 종료 표시.

### 3.7 MM_in_buffer — 외부 AXIS 수신 + 타일 재배열 (`src/MM_in_buffer.v`)

AXI-Stream으로 들어오는 feature(`in_F_*`)/weight(`in_W_*`)를 BRAM(`in_F_array`, `in_W_array`, `MM_in_buffer.v:67-68`)에 적재하고, MM_buffer가 요구하는 순서로 재출력.
- IDLE→IN_DATA→CAL FSM(`MM_in_buffer.v:3-4,130-142`).
- feature 출력 주소: `out_F_addr = out_F_row_addr * F_width_block_num + out_F_col_addr`(`MM_in_buffer.v:91`) — feature 행렬을 **K-블록 열 단위로 순회**하며 같은 행 블록을 여러 weight 열 블록에 재사용.
- 블록 카운트는 `F_block_size = F_length * F_width_block_num`, `W_block_size = W_width_block_num * F_width_block_num * A_size`(`MM_in_buffer.v:99-109`)로 계산. → **타일링 차원(M·K·N 블록 수)** 이 여기서 결정된다.

### 3.8 MM_out_buffer — 부분합 누적 + 재양자화 ★ (`src/MM_out_buffer.v`)

이 모듈이 **K차원 분할 누적과 INT8 재양자화의 핵심**이다.
- `F_array`는 `(*ram_style="block"*)` BRAM(`MM_out_buffer.v:47`)으로 `A_size*OUT_MEM_WIDTH` 폭. 부분합을 누적 저장.
- 누적: PE_array가 내보낸 한 타일 결과 `A`(`in_data`, `MM_out_buffer.v:85`)를 부호확장(`A_in_adder`, `MM_out_buffer.v:311-315`)하고, 같은 출력 위치의 기존 누적값 `B`(BRAM read, `MM_out_buffer.v:269`)와 **AdderS로 saturating 가산** → `C`를 다시 BRAM에 write(`MM_out_buffer.v:246-258`). 즉 `W_width_block_num`(K-블록 수)만큼 부분곱을 누적.
- 재양자화: 누적 완료된 `out_F_data_delay1`을 열별로 `right_shifter`에 통과(`MM_out_buffer.v:317-330`). `shift`(런타임)와 `OUT_MEM_WIDTH→data_width` 변환으로 **INT32(또는 21)→INT8** 재양자화. 출력은 `out_data`(`MM_out_buffer.v:176-181`).
- 출력 주소: `out_data_addr = out_data_col_addr * F_length + out_data_row_addr`(`MM_out_buffer.v:88`)로 결과 행렬을 열 블록·행 순으로 송출.

### 3.9 AdderS — saturating 벡터 가산기 (`src/AdderS.v`)

`A_size`개 lane을 병렬 가산(`AdderS.v:22-39`). 각 lane은 sign-extend 1비트 추가(`temp[i]`, `AdderS.v:24-25`) 후 overflow/underflow를 `temp[i][dw:dw-1]` 2비트로 판정해 양수/음수 포화값으로 클램프(`AdderS.v:31-35`). out_buffer의 부분합 누적에 사용.

### 3.10 right_shifter — 반올림 + 포화 우측시프트 ★ (`src/right_shifter.v`)

재양자화의 산술 코어(`right_shifter.v`).
- 산술 우측시프트: `temp1_out = data_in >>> shift`(`right_shifter.v:22`).
- **반올림(round half up)**: `temp2_out = data_in[shift-1] ? temp1_out+1 : temp1_out`(`right_shifter.v:23`) — 버려지는 최상위 비트로 올림.
- 포화: `before_data_width`→`after_data_width`로 줄일 때 상위 잉여비트로 over/under flow 검출(`right_shifter.v:25-29`), INT8 min(`1000_0000`)/max(`0111_1111`)로 클램프(`right_shifter.v:39-43`). `shift==0` 특수 경로 별도 처리(`right_shifter.v:31-37`).
- 이 round+saturate 규칙은 SW 참조 `Matrix_mul_soft`의 `(temp+(1<<(R_shift-1)))>>R_shift` + clip(`vitis/Matrix.cpp:90-97`)과 정확히 일치 → bit-exact 검증 가능.

### 3.11 MM_ultra / MM_ultra_axi / MM_ultra_top — 계층 + AXI (`src/MM_ultra*.v`)

- `MM_ultra`(`MM_ultra.v:4-221`): in_buffer→MM_buffer→out_buffer 3단 인스턴스 결선. shift/F_length/F_width_block_num/W_width_block_num 4개 런타임 파라미터를 입력 변화 감지 후 래치(`MM_ultra.v:73-98`). feature는 `s0`(in_F), weight는 `s1`(in_W), 결과는 out_data로 통일.
- `MM_ultra_axi`(`MM_ultra_axi.v`): 표준 AXI4-Lite slave 4 레지스터(`slv_reg0~3`, `MM_ultra_axi.v:133-136`). 매핑(`MM_ultra_axi.v:445-448`): **reg0=shift, reg1=F_length(=입력 행 수), reg2=F_width_block_num(K/A_size), reg3=W_width_block_num(N/A_size)**. 3개 AXIS(s0=feature, s1=weight, m0=결과)를 MM_ultra에 직결(`MM_ultra_axi.v:451-464`).
- `MM_ultra_top`(`MM_ultra_top.v`): 위 두 모듈의 얇은 포트 래퍼. prj.tcl이 BD에서 module reference로 인스턴스화(`prj.tcl:43,234-237`).

### 3.12 In Progress — 비선형 연산자 RTL (⚠ 미완성, README.md:3)

> 이하 모듈은 `In Progress/`에 있으며 README가 정확도/성능 개선 중이라 명시. RTL은 작성되어 있으나 메인 GEMM 데이터패스(prj.tcl BD)에는 통합되어 있지 않다(prj.tcl에 Softmax/Gelu 인스턴스 없음 — grep 확인).

**Softmax (`In Progress/src/Softmax.v`)** — 3-pass 스트리밍 알고리즘:
- 입력 INT8, scale_in/scale_out 런타임(`Softmax.v:11-12`). cnt로 stage 1/2/3 구분(`Softmax.v:78`): stage1=최댓값 탐색(`data_in_max`, `Softmax.v:172-179`), stage2=`exp(x-max)` 누적합(`e_sum`, `Softmax.v:202-209`), stage3=정규화 출력.
- exp(x-max): `Exp_module`로 계산(`Softmax.v:214-218`). 분모 정규화는 **나눗셈 대신 ln 후 빼고 다시 exp**: `x_max_ln = (x-max) - ln(sum)` (`Softmax.v:75-76`) → `exp(x_max_ln)` (`Softmax.v:229-233`). 즉 `softmax = exp(x-max-ln(Σexp))`로 제수기 제거.
- 고정소수점 포맷이 신호명에 명시(S9Q10, U8Q12, U0Q25 등). 출력 scale은 `scale_out`에 따라 비트 선택(`Softmax.v:85-104`).
- 10단 파이프라인(valid/stage delay 1~10, `Softmax.v:107-151`).

**Exp_module (`In Progress/src/Exp_module.v`, latency 3)**: `exp(x)=2^(x·log2e)`. `x·log2e`를 정수부/소수부로 분해(`Exp_module.v:10-21`), 정수부는 시프트(`12'b1000... >> x_int`, `Exp_module.v:28-30`), 소수부는 1차 근사 `1+0.5·frac`(`Exp_module.v:23`), 둘을 곱해 2^x 산출(`Exp_module.v:36-42`). LUT/곱셈 기반 저비용 exp.

**Ln_module (`In Progress/src/Ln_module.v`, latency 2)**: priority encoder로 최상위 1비트 위치 `w`(정수부 지수)와 정규화 가수 `k`를 추출(`Ln_module.v:14-47`), `ln ≈ (w,k)·상수(4'b1011)` 선형 근사(`Ln_module.v:50-61`). Softmax 분모 처리 및 LayerNorm용. **LayerNorm 전용 top 모듈은 없고 Ln_module(자연로그)만 제공** → 본격 LayerNorm 데이터패스는 미구현.

**gelu (`In Progress/src/gelu.v`, latency 8) + lin (`lin.v`, latency 4)**:
- GELU를 구간별 처리: x가 충분히 작으면(`<-2.5`) 0, 충분히 크면(`≥2.5`) x 통과, 중간 구간은 다항식 근사(`gelu.v:96-107`).
- 중간 구간: `lin`이 `L = a·(|0.75x|-7/4)² + 1` 형태의 2차 다항식(`lin.v:19-33`, a=-37/128, b=7/4)을 계산 → `y ≈ 0.5·L·x`(`gelu.v:88-94`). 즉 **piecewise 2차 다항식으로 GELU 근사**.
- in_scale로 입력 고정소수점 정렬(`gelu.v:51-58`), 출력 재시프트(`gelu.v:105`).

**EightGelus (`In Progress/src/EightGelus.v`)**: `num_gelu`개 gelu를 병렬 인스턴스화하는 SIMD 래퍼(`EightGelus.v:107-116`). AXIS valid/last/keep를 9단 지연으로 동기화(`EightGelus.v:32-105`). `Gelus_top.v`/`Gelus_axi.v`가 AXI 래핑, `Softmax_top*.v`/`Softmax_control.v`가 Softmax AXIS 래핑·버퍼링(`Softmax_control.v:38` 1024-deep 입력 버퍼).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 GEMM 전체 파이프라인 (C = A × B, A=feature, B=weight)

```
PS(DDR) ─AXI DMA0─▶ MM_in_buffer(in_F_array)  ┐
PS(DDR) ─AXI DMA1─▶ MM_in_buffer(in_W_array)  ┘
                          │ (타일 재배열, K-블록 순회)
                          ▼
                     MM_buffer (feature_buffer/weight_buffer BRAM)
                          │  FSM: SET_WEIGHT → SET_FEATURE
                          ▼
                     MM (weight 시프트 적재 → set_w → feature 스트리밍)
                          ▼
                  PE_array  (16×16 weight-stationary systolic)
                   - x: 행 방향 전파(좌→우), 입력 skew 대각 정렬
                   - psum: 열 방향 누적(위→아래)
                   - 출력 de-skew 버퍼
                          ▼ (열별 부분합, 20비트)
                  MM_out_buffer
                   - AdderS로 K-블록 부분합을 BRAM(F_array)에 누적(32/21비트)
                   - right_shifter로 round+saturate → INT8 재양자화
                          ▼
                  m0_axis ─AXI DMA2─▶ PS(DDR) feature_out
```

근거: `MM_ultra.v:118-220`(3단 결선), `prj.tcl:1117-1135`(DMA0=feature MM2S, DMA1=weight MM2S, DMA2=result S2MM, axis_dwidth_converter 3개로 폭 변환).

### 4.2 병렬화 차원

- **공간 병렬**: PE_array의 `array_m × array_n` = 16×16 = 256 MAC 동시 연산(`PE_array.v:72`, README 16×16). 한 클럭에 16-wide feature × 16-wide weight 부분곱.
- **K차원 시간 누적**: `W_width_block_num`(=N/A_size) 및 `F_width_block_num`(=K/A_size) 블록을 시간축으로 순회하며 MM_out_buffer에서 누적(`MM_out_buffer.v:246-258`).
- **출력 lane 병렬 재양자화**: A_size개 right_shifter 병렬(`MM_out_buffer.v:317-330`).
- **AXIS 폭**: `array_size*data_width` = 16×8 = 128비트 스트림(`MM_ultra_axi.v:31`), DMA(32/64비트)와 axis_dwidth_converter로 정합(`prj.tcl:317-329`).

### 4.3 데이터 타입 / 포맷

- 입력/가중치/출력: signed INT8(`vitis/Defines.h:6` DATA_TYPE=s8, MAX=127/MIN=-128 `:7-8`).
- 패딩 규약: 행·열을 `A_SIZE` 배수로 0-패딩(`vitis/Matrix.cpp:13-16`, `pynq/MM.py:16-24`). 그래서 임의 크기 행렬을 타일로 분할 가능.
- 제약: A의 width = B의 height, 둘 다 A_SIZE 배수여야 함(`pynq/MM.py:128-135`).

### 4.4 SW 실행 시퀀스 (PYNQ 기준, `pynq/MM.py:114-155`)

1. AXI4-Lite로 shift/A_h(F_length)/F_width_block_num/W_width_block_num 레지스터 write(`MM.py:148-151`).
2. 결과 수신 채널(out DMA, S2MM) 먼저 open(`MM.py:152` → `out_feature_transfer`).
3. feature DMA(`in_f_dma`) → weight DMA(`in_w_dma`) 순으로 MM2S 전송 시작(`MM.py:153-154`).
4. `out_feature_wait()`로 S2MM IDLE 폴링까지 블로킹(`MM.py:155, 104-105`).
베어메탈 SDK도 동일 순서(`sdk/main.c`, `vitis/Matrix.cpp:144-162`): 수신 채널 우선 open 후 weight/feature 송신, S2MM IDLE 폴링(`Matrix.cpp:160-162`).

---

## 5. HW/SW 매핑

| 계층 | 파일 | 역할 | 대응 관계 |
|---|---|---|---|
| **PYNQ Python 호스트** | `pynq/MM.py` | `allocate`로 물리연속 버퍼, `MMIO`로 DMA/제어 레지스터 접근, `mat_mul()` | A_SIZE=25, DMA addr 0xA000_0000/0xA001_0000/0xA002_0000, 제어 0xA003_0000(`MM.py:5,54-71`) |
| **베어메탈 드라이버** | `sdk/matrix.c`, `sdk/main.c`, `sdk/defines.h` | `Xil_Out32`로 레지스터, `Xil_DCacheFlushRange` 캐시 일관성, DMA 직접 구동 | A_SIZE=24(`sdk/defines.h:30`), 레지스터 맵 SHIFT/FL/FWBN/WWBN(`defines.h:6-10`) |
| **HLS/참조 검증** | `vitis/Matrix.cpp`, `vitis/main.cpp` | `Matrix_mul_soft`(SW 골든) vs `Matrix_mul_hard`(HW 호출) 비교 | round+clip 규칙이 RTL right_shifter와 동일(`Matrix.cpp:90-97` ↔ `right_shifter.v:22-43`) |
| **AXI4-Lite slave** | `src/MM_ultra_axi.v` | slv_reg0~3 = shift/F_length/FWBN/WWBN | SW의 4개 write(`MM.py:148-151`, `Matrix.cpp:138-141`)가 정확히 이 레지스터 |
| **AXIS 데이터무브** | `src/MM_ultra_axi.v` + `prj.tcl` 3×AXI DMA | s0=feature, s1=weight, m0=result | DMA0/1=MM2S, DMA2=S2MM(`prj.tcl:1117-1135`) |
| **systolic 코어** | `src/PE*.v`, `MM*.v` | INT8 GEMM | A_SIZE = array_m = array_n. SW의 A_SIZE 패딩 = 타일 = array 변 길이 |

**중요 일관성 주의**: `A_SIZE`가 SW마다 다름 — PYNQ=25(`MM.py:5`), SDK=24(`sdk/defines.h:30`), Vitis=16(`vitis/Defines.h:36`), RTL 기본 파라미터=24(`MM_ultra_axi.v:7`) / README는 16(`README.md:2`). **A_SIZE는 RTL 합성 시 파라미터와 SW 상수가 반드시 일치해야 함** — 현재 repo는 데모 설정이 혼재(섹션 8 리스크).

레지스터 오프셋: 0x0=shift, 0x4=F_length(행 수), 0x8=F_width_block_num, 0xC=W_width_block_num (`sdk/defines.h:6-10`, `MM.py:148-151`, `MM_ultra_axi.v:445-448`).

---

## 6. 빌드 / 실행

1. **Vivado 2019.1**(반드시) + PYNQ-Z1 board file 설치(`README.md:6`, `prj.tcl:23`).
2. 새 프로젝트 생성(part `xc7z020clg400-1`), `src/*.v` 전체 추가(`prj.tcl:53-54`).
3. `prj.tcl` source → BD 자동 생성: MM_ultra_top + axi_dma 0/1/2 + smartconnect 3 + axis_dwidth_converter 3 + PS7(`prj.tcl:234-335`).
4. BD wrapper 생성→top 지정, synth/impl→bitstream(`README.md:9-10`).
5. SW 실행 택1:
   - **PYNQ**: bitstream+hwh 업로드 후 `MM.py` import, `ini_MM()` → `mat_mul(A,B,C,shift)`.
   - **베어메탈**: Vitis/SDK에서 `sdk/*.c` 빌드, `main.c`가 160×160×160 GEMM을 10회 soft/hard 비교·타이밍(`sdk/main.c:18,23-29`).
   - **HLS 참조검증**: `vitis/main.cpp`로 121×311×72 랜덤 행렬 soft vs hard 일치 확인(`main.cpp:5-42`).

> In Progress(Softmax/Gelu)는 별도 top(`Softmax_top`, `Gelus_top`)과 tb(`Softmax_top_tb.sv`, `gelu_tb.sv`)만 존재. 메인 BD에 미통합(prj.tcl에 없음).

---

## 7. 의존성

- **HW 합성 툴체인**: Vivado 2019.1(엄격, `prj.tcl:23-31`에서 버전 불일치 시 에러). PYNQ-Z1 board file.
- **Xilinx IP**(vendor, 분석 제외): axi_dma 7.1, smartconnect 1.0, axis_dwidth_converter 1.1, processing_system7 5.5, axi_interconnect 2.1(`prj.tcl:134,253-335`).
- **베어메탈 BSP**: `xparameters.h`, `xil_io.h`, `xil_cache.h`, `xtime_l.h`, `xaxidma_hw.h`(`sdk/main.c:1-11`, `vitis/Matrix.cpp:1-8`). XAXIDMA 오프셋 매크로 사용(`defines.h:13-32`).
- **PYNQ**: `pynq.allocate`, `pynq.MMIO`, numpy(`MM.py:1-3`).
- RTL 내부 의존: 순수 Verilog-2001, third-party 없음. `$pow`(`MM_buffer.v:37`) 등 합성 가능 시스템함수만 사용.

---

## 8. 강점 / 한계 / 리스크

### 강점
- **명확한 weight-stationary systolic 설계**: PE→PE_line→PE_array 계층이 교과서적이고 skew/de-skew 버퍼가 정확히 구현됨(`PE_array.v:91-127`).
- **완전한 HW/SW 스택**: 동일 GEMM을 PYNQ/베어메탈/HLS 3경로로 검증, soft 골든 모델 bit-exact 비교 가능(round+clip 규칙 일치).
- **런타임 파라미터화**: A_SIZE만 합성 고정, 행렬 크기(M,K,N)는 AXI4-Lite로 런타임 조절(`MM_ultra_axi.v:445-448`). 임의 크기 행렬을 타일링으로 처리.
- **재양자화 내장**: shift 기반 round+saturate가 out_buffer에 통합되어 INT8 추론 파이프라인에 바로 적합(`MM_out_buffer.v:317-330`).
- **저비용 비선형 근사 RTL 존재**: division-free softmax, LUT-free exp/ln, piecewise 2차 GELU — Transformer full-stack 가속 기반(미완성이나 재사용성 높음).

### 한계
- **GEMM만 완성**, Softmax/GELU/LayerNorm은 미통합(`README.md:3`, prj.tcl에 인스턴스 없음). 본격 LayerNorm top 없음(Ln_module만 존재).
- **INT8 고정**: data_width 파라미터지만 SW(s8)·포화 로직이 8비트 가정. 다른 비트폭 미검증.
- **AXI4-Lite 폴링 동기화**: 인터럽트 없이 S2MM IDLE 폴링(`MM.py:104-105`, `Matrix.cpp:160-162`) → CPU 점유.
- **단일 PE array**: 멀티 array/멀티 헤드 병렬 없음. ViT 어텐션 전체 파이프라인(QKV→softmax→AV) 자동화 없음.
- **PE_array skew 버퍼**가 generate로 행마다 시프트 레지스터 생성 → array_size 커지면 레지스터/면적 증가.

### 리스크
- **A_SIZE 불일치(중대)**: README=16, RTL param=24, SDK=24, PYNQ=25, Vitis=16으로 제각각. 합성 파라미터와 SW 상수가 어긋나면 **잘못된 결과/행글 위험**. 사용 전 단일 값으로 통일 필수.
- **PYNQ A_SIZE=25는 비-2의 거듭제곱**: log2_array_m 계산(`clogb2`) 및 폭 산정에 영향 — 검증 필요.
- **In Progress 정확도 미보증**: 다항식/LUT 근사라 정량 오차 미문서화(`README.md:3`).
- **Vivado 2019.1 종속**: 최신 Vivado로 이식 시 BD 재생성 필요.
- 일부 소스에 깨진 주석(GBK 인코딩 한자, `right_shifter.v:2`, `MM.v:69` 등) — 가독성 저하이나 기능 무관.

---

## 9. 우리 프로젝트 관점 시사점 (HG-PIPE 계열 ViT/Transformer 가속기 + XR 시선추적)

우리 연구(고처리량 ViT/Transformer FPGA 가속기 + XR eye-tracking)에서 **직접 재사용/참고 가능한 요소**:

1. **systolic PE array 골격 (높은 재사용성)** — `PE.v`/`PE_line.v`/`PE_array.v`의 weight-stationary + skew/de-skew 구조는 그대로 백본 GEMM 엔진으로 차용 가능. 특히 `PE_array.v:91-127`의 입력 skew(`x_buf`)·출력 de-skew(`out_buf`) 패턴은 HG-PIPE류 파이프라인에서 타일 경계 정렬에 바로 쓸 수 있다. 단 HG-PIPE의 핵심인 **완전 파이프라인(레이어 간 stall-free)** 을 위해서는 본 repo의 "버퍼 다 채운 뒤 계산하는" both_full 기반 배치 방식(`MM_buffer.v:92`)을 **스트리밍/더블버퍼링**으로 개조해야 함.

2. **right_shifter 재양자화 모듈 (즉시 재사용)** — `right_shifter.v`의 round(half-up)+saturate 로직과 SW 골든(`Matrix.cpp:90-97`)의 bit-exact 일치는 우리 양자화 ViT의 requantization 단에 그대로 이식 가능. per-channel scale로 확장하려면 shift를 lane별 벡터로 일반화하면 됨(현재는 단일 shift broadcast, `MM_out_buffer.v:317-330`).

3. **out_buffer의 K-블록 누적 패턴** — `AdderS`+BRAM 누적(`MM_out_buffer.v:246-258`)은 K차원이 array보다 큰 ViT FFN/projection GEMM에 필수적인 구조. saturating accumulate(`AdderS.v:31-35`)는 INT 오버플로 안전성 측면에서 참고 가치.

4. **division-free Softmax (어텐션 가속 핵심)** — `Softmax.v:75-76`의 `exp(x-max-ln(Σexp))` 기법은 제수기를 제거해 어텐션 softmax를 저비용으로 파이프라인화한다. `Exp_module`(2^x 분해, `Exp_module.v:10-42`)·`Ln_module`(priority-encoder ln, `Ln_module.v:14-61`)은 우리 attention 블록의 softmax LUT 대안으로 직접 평가할 만함. 단 미완성·정확도 미보증이므로 우리 양자화 정책(scale/zero-point)과 정합 후 재검증 필요.

5. **piecewise 2차 GELU + SIMD 래퍼** — `gelu.v`(구간별 2차 근사)+`EightGelus.v`(num_gelu 병렬)는 ViT MLP의 GELU를 lane 병렬로 처리하는 템플릿. 우리 처리량 목표상 `num_gelu`를 array 폭에 맞춰 확장하는 형태로 재사용 가능(`EightGelus.v:107-116`).

6. **AXI 통합 패턴** — feature/weight 분리 입력 + 결과 단일 출력의 3-DMA 구조(`prj.tcl:1117-1135`)와 AXI4-Lite 4-레지스터 제어(`MM_ultra_axi.v:445-448`)는 우리 SoC 통합의 미니멀 레퍼런스. XR 시선추적 추론처럼 작은 입력·실시간 요구에는 폴링 대신 **인터럽트/데이터무버 자동화**로 개선 필요.

7. **XR 관점** — 시선추적 ViT는 저지연·소형 입력이 특징이므로, 본 repo의 배치형 GEMM보다 HG-PIPE식 **레이어 융합 + 상시 스트리밍**이 적합. 본 repo는 (a) 양자화 GEMM 코어, (b) 비선형 근사 RTL의 **부품 라이브러리**로 활용하고, 파이프라인 스케줄링·온칩 메모리 계층은 우리 쪽에서 재설계하는 것이 합리적.

**비재사용/주의**: both_full 배치 동기화, A_SIZE 혼재 설정, 폴링 기반 호스트 드라이버는 고처리량 목표에 부적합 → 참고만.

---

## 10. 근거 / 한계 표기

### 분석 근거 (Read 완료 파일)
- 완성 RTL 12개 전부: `MM_ultra_top.v`, `MM_ultra_axi.v`, `MM_ultra.v`, `MM_in_buffer.v`, `MM_buffer.v`, `MM.v`, `PE_array.v`, `PE_line.v`, `PE.v`, `MM_out_buffer.v`, `AdderS.v`, `right_shifter.v` — 전문 Read.
- In Progress 7개: `Softmax.v`(전문), `Exp_module.v`(전문), `Ln_module.v`(전문), `gelu.v`(전문), `lin.v`(전문), `EightGelus.v`(전문), `Softmax_top.v`(전문), `Softmax_control.v`(부분 1-60).
- SW: `vitis/Matrix.cpp`·`Matrix.h`·`main.cpp`·`Defines.h`(전문), `sdk/matrix.c`·`defines.h`(전문), `sdk/main.c`(1-50), `pynq/MM.py`(전문).
- 빌드/검증: `README.md`(전문), `prj.tcl`(1-60 + grep으로 BD 결선/IP 확인), `sim/MM_Ultra_tb.sv`(1-70).

### 분석 한계 (미완료/추정 부분과 이유)
- **prj.tcl 전체(1100+행) 미정독**: 자동생성 BD 스크립트라 vendor 영역. grep으로 핵심 결선(DMA↔converter↔MM_ultra_top, `prj.tcl:1117-1135`)·IP 목록만 확인. 클럭/주소맵 세부는 미검증.
- **sdk/main.c 후반부(50행~) 미정독**: 앞부분으로 흐름(soft/hard 비교·타이밍) 충분 확인. 세부 DMA 호출은 `vitis/Matrix.cpp:104-163`와 동일 패턴으로 추정.
- **Softmax_top_axi.v / Gelus_top.v / Gelus_axi.v / Softmax_control.v(61행~)**: AXIS 래퍼로 판단해 핵심 연산 모듈 우선 분석. 래퍼 세부 타이밍(ready 백프레셔 등)은 미정밀.
- **testbench(sim/*.sv) 일부만 Read**: GEMM tb는 soft 모델·파라미터(`MM_Ultra_tb.sv:1-70`)만 확인, 자극 시퀀스 전체는 미정독. In Progress tb 미Read.
- **실제 합성/시뮬레이션 미수행**(정적 코드 분석만). 면적/주파수/정확도 수치는 repo에 미제공이라 본 문서에도 없음.
- **A_SIZE 혼재**는 코드 사실 그대로 기재(섹션 5,8). 어느 값이 "정답"인지는 repo에 명시 없음 → 사용자 판단 필요.
