# MSD-FCCM23 코드베이스 정밀 분석

> 분석 대상 경로: `REF/Others/MSD-FCCM23`
> 분석 방식: 실제 소스 Read 기반(라인 근거). bash 미사용(UNC), Glob/Grep/Read만 사용.
> 대표 RTL: `hardware/vivado/zcu102/rtl` (pynqz2/ultrascale와 RTL 동일, 차이는 `def.sv` 파라미터/플랫폼 — 5절 요약).

---

## 1. 개요

- **목적/한줄요약**: MSD는 **혼합 부호자리(Mixing Signed Digit) 표현**으로 DNN을 양자화하고, FPGA의 **이종 자원(LUT + DSP)을 동시에** 활용해 추론을 가속하는 프레임워크. weight를 "유효 부호자리(essential/effective bit, EB)" 개수로 제한(restricted signed digit, RSD)하고, **LUT 기반 bit-serial 시스톨릭 어레이**가 부호자리 단위로 곱셈을 수행 → DSP-only 대비 더 높은 이론 peak를 달성.
- **원논문**: Jiajun Wu, Jiajun Zhou, Yizhao Gao, Yuhao Ding, Ngai Wong, Hayden So, *"MSD: Mixing Signed Digit Representations for Hardware-efficient DNN Acceleration on FPGA with Heterogeneous Resources"*, IEEE FCCM 2023, pp.94-104 (DOI 10.1109/FCCM57271.2023.00019). 근거: 루트 `README.md` 4-13행.
- **타깃 디바이스**: 3개 플랫폼 — **Pynq-Z2(XC7Z020)**, **Ultra96-V2(ZU3EG)**, **ZCU102(ZU9EG)**. 도구 Xilinx Vivado **2021.2**, 호스트 PYNQ 2.6. 근거: `hardware/README.md` 6-10행, 결과표 113-221행.
- **지원 모델**: VGG-16, ResNet-18/50, MobileNet-V2, **Vision Transformer(ViT-base)**. 근거: `software/README.md` 108-110행, `hardware/README.md` 215행. → ViT 지원이 명시되어 우리 프로젝트와 직접 연관.
- **핵심 성과(논문 자체보고)**: ZU9EG에서 ViT-base 22.30ms / 1481 GOPS, ResNet-50 15.94ms / 516.9 GOPS 등(hardware/README 결과표). VGG-16 mixed-EB로 최대 ~2.28x speedup(software/README 128-169행).

MSD는 **HW/SW 분리** + **스케줄러로 연결**되는 구조. SW = (1) MSD 양자화 QAT(`msd_quant`), (2) 이종자원 스케줄러(`msd_scheduler`). HW = (1) RTL 가속기(`vivado`, LUT+DSP 이종 어레이), (2) PYNQ 호스트(`host`).

---

## 2. 디렉토리 구조 (자체 소스 + 제외 목록)

### 자체 핵심 소스 트리
```
MSD-FCCM23/
├── README.md
├── software/
│   ├── README.md
│   ├── msd_quant/                          # MSD 양자화 QAT
│   │   └── msd_quantization/
│   │       ├── msdquant/
│   │       │   ├── binary_converter.py     # float↔bit 변환(IEEE-754)
│   │       │   ├── quant_modules.py        # ★ Quantizer (EB/CSD 그리드, MSE 스케일)
│   │       │   ├── quant_affine.py / quant_utils.py / quant_model.py
│   │       │   └── multihead_attention.py  # ViT용 MHA 양자화
│   │       ├── ImageNet/{main.py, dataloader.py}
│   │       └── quant/setup.py              # quant_cuda CUDA 커널 빌드
│   │   └── msd_analysis/                    # mixed-EB 탐색(adaptive_search_*.py)
│   └── msd_scheduler/                       # ★ 이종자원 스케줄러
│       ├── scheduler.py                     # 브루트포스 타일링 DSE
│       ├── dsp_backend.py / lut_backend.py  # DSP/LUT latency·buffer 모델
│       ├── hw_model.py / mem_backend.py     # 통합 HW 모델, 메모리 latency
│       ├── simulator.py / run_simulator.py  # 사이클 정확 시뮬
│       ├── generator.py / run_hw_gen.py     # 명령어/HW 산출
│       ├── dnn_model.py / lat_eb_comb.py / utils.py
│       └── archs/ aux/ device/ models/ results/  # 구성/결과(데이터)
└── hardware/
    ├── README.md
    ├── host/{pynqz2,ultrascale,zcu102}/     # PYNQ 호스트 드라이버
    │   └── acc_ctrl.py, run_eval.py, instr_gen_model.py, start_eval.sh
    └── vivado/{pynqz2,ultrascale,zcu102}/
        ├── rtl/                             # ★ SystemVerilog 가속기
        │   ├── def.sv                       # 전역 파라미터(어레이 크기/비트폭)
        │   ├── fthbs_top.sv                 # 최상위(PS ctrl + aux + core)
        │   ├── sys_top.sv                   # anga core (LUT+DSP 통합)
        │   ├── bit_serial_mul.sv mp_lsf.sv  # ★ 부호자리 bit-serial 곱
        │   ├── lut_pe.sv lut_sys.sv bs_lut_core.sv lut_ctrl_ex.sv  # LUT 어레이
        │   ├── dsp_pe.sv dsp_sys.sv bp_dsp_core.sv dsp_ctrl_ex.sv common_mul.sv  # DSP 어레이
        │   ├── buf_sdp.sv fifo_axis.sv mem_itf.sv glb_ctrl.sv  # 버퍼/메모리/제어
        │   ├── aux_instr.sv ps_ctrl.sv ctrl_ld.sv ctrl_wb.sv
        │   └── top_wrapper.v
        ├── tb/                              # SV 테스트벤치(tb_drv*, sys_top_tb)
        └── bd/ ...                          # 블록디자인 wrapper(생성물성)
```

### 제외 목록(이름만, 분석 제외)
- `hardware/vivado/*/bd/`, `*.ip_user_files`, `.Xil`, `*.bit`/`*.hwh`(비트스트림/handoff), Vivado 프로젝트 산출물.
- `software/msd_quant/msd_quantization/quant/`의 CUDA 빌드 산출물(`quant_cuda` 확장).
- 체크포인트/QAT 결과, `training_logs/`, `results/`·`aux/` CSV(데이터).
- DALI 등 외부 데이터로더 의존(vendor).

### 동일 RTL 3중 복제
`pynqz2`/`ultrascale`/`zcu102`의 `rtl/*.sv` 파일명·모듈은 동일. 차이는 `def.sv` 파라미터(어레이 크기·버퍼 깊이)와 보드 BD. zcu102를 대표 정독.

---

## 3. 핵심 모듈·파일별 정밀 분석

### 3.A 하드웨어 (RTL)

#### 3.A.1 전역 파라미터 (`def.sv`)

- **LUT bit-serial 어레이**: `HW_IDX_DW 4`(부호자리 인덱스 폭), `HW_WGT_DW 8`, `HW_ACT_DW 8`, `HW_PSU_DW 8`. `HW_LUT_PE_ROWS=80`, `HW_LUT_PE_COLS=80`(80×80). 근거: 20-26행.
- **`BIT_SERIAL` 매크로 정의**(28행): 활성화 시 LUT PE 가로버스 폭이 `IDX_DW(4)`, 비활성 시 `WGT_DW(8)`. 근거: 30-35행. → **LUT 어레이는 weight를 4비트 "부호자리 인덱스"로 전달**(bit-serial), DSP 어레이는 8비트 weight.
- **DSP bit-parallel 어레이**: `HW_DSP_PE_ROWS=48`, `HW_DSP_PE_COLS=48`(48×48). DSP48 포트폭 `A_DW=27, B_DW=18, P_DW=45`. 세로버스 16비트, 가로버스 8비트. 근거: 44-52행. → **두 종류 시스톨릭 어레이가 공존**(LUT 80×80 + DSP 48×48).
- 버퍼 깊이/크기: `HW_BS_BUF_SIZE=65536`, `HW_BP_BUF_SIZE=65536`, 글로벌 메모리 BW 32(60행). 주석 "4-12, 8-11, 16-10"(37행) → EB(4/8/16)별 act 버퍼 깊이.

#### 3.A.2 부호자리 곱셈 — bit-serial 곱 (`bit_serial_mul.sv`, `mp_lsf.sv`) ★핵심

- **`mp_lsf`**(`mp_lsf.sv` 6-25행): 동적 좌측 시프터 `c = a << b`. barrel shifter를 generate로 구현(`tmp[i] = b[i] ? tmp[i-1] << 2**i`, 16-22행). → weight의 "부호자리 위치(b)"만큼 활성화를 시프트 = **2의 거듭제곱 곱셈**.
- **`bit_serial_mul`**(`bit_serial_mul.sv` 36-62행): 한 부호자리 곱셈.
  - `wgt_in`(4비트)의 MSB가 부호: `act_in_neg = wgt_in[IDX_DW-1] ? ~act_in : act_in`(49행) → **부호자리의 +/- 처리**(2의 보수 근사).
  - `mp_lsf`로 `act_in`을 `wgt_in`만큼 시프트(51-58행) → partial product.
  - 출력은 시프트 결과의 중간 비트 슬라이스(60행).
  - → **weight를 "시프트량(부호자리 위치) + 부호"로 인코딩**해, 곱셈을 시프트+부호반전으로 대체(LUT만으로 곱셈 = DSP 절약). 한 weight가 EB개 부호자리를 가지면 EB 사이클에 걸쳐 누산(bit-serial).
- (대비) **`common_mul`**(`common_mul.sv` 6-33행): `BIT_SERIAL` 미정의 시 일반 곱(`act_s * wgt_s`, 28행), `use_dsp="no"`로 LUT 강제(5행). 즉 LUT 어레이를 bit-parallel로 쓸 때의 대안.

#### 3.A.3 LUT PE & 어레이 (`lut_pe.sv`, `bs_lut_core.sv`)

- **`lut_pe`**(`lut_pe.sv` 7-85행): **input-stationary** 시스톨릭 PE(주석 5행).
  - 가로 흐름: `left_in`(weight 부호자리)을 1단 레지스터 후 `right_out`으로 전달(30-38행).
  - 곱: `BIT_SERIAL`이면 `bit_serial_mul`(40-49행, act=top_in, wgt=left_in), 아니면 `common_mul`(51-63행).
  - 누산: `adder_out = psum_out + psum_acc`(65행), `psum_acc` 레지스터(67-73행) → **output-stationary 누산**.
  - 세로 흐름: `psum_sel`이면 누산값, 아니면 top_in 그대로(`bottom_out`, 75-83행) → 결과 배출 vs 활성화 패스스루 선택.
- **`bs_lut_core`**(`bs_lut_core.sv` 6-60행+): LUT 어레이 래퍼.
  - act 버퍼(`buf_sdp`)는 COLS개, **weight 버퍼는 ROWS개이며 `IDX_DW(4)`폭**(51-52행 `bs_lut_weights[ROWS][IDX_DW]`). → DSP core(아래)는 `WGT_DW(8)`폭. 이 폭 차이가 LUT/DSP 구분의 RTL 증거.
  - AXI-Stream으로 act/wgt 적재(`s_axis_bs_*_ld`, 21-27행), 출력 wb(29행).

#### 3.A.4 DSP PE & 어레이 (`dsp_pe.sv`, `dsp_sys.sv`, `bp_dsp_core.sv`) ★핵심

- **`dsp_pe`**(`dsp_pe.sv` 6-177행): **DSP48E2 직접 인스턴스화**로 1개 DSP에 2개 8비트 MAC 패킹.
  - 활성 패킹: `dsp_a_in = {1'b0, top_in[15:8], 10'h000, top_in[7:0]}`(32행) → 상위/하위 바이트를 27비트 A포트에 분리 배치.
  - 누산 패킹: `dsp_c_in = {19'h0, psum_acc[15:8], 10'h000, psum_acc[7:0]}`(34행).
  - DSP48E2 설정(47-156행): MULTIPLY, ONE48, INMODE=5'b10001, OPMODE=9'b00_011_01_01 → A×B + C. 파이프 레지스터 AREG/BREG/MREG/PREG=1.
  - psum 추출: `psum_acc <= dsp_p_out[31:16]`(173행). → **8비트 MACC 최적화**(주석 22, 31행): 2개 활성×weight를 한 DSP로.
- **`dsp_sys`**(`dsp_sys.sv` 7-91행): 48×48 시스톨릭 어레이.
  - weight(`wgt_idx`)는 **가로 좌→우 전파**(row_connections), 활성(`act_in`)은 **세로 위→아래 전파**(col_connections)(36-80행 generate, 4가지 경계 케이스 lt/fc/fr/nm).
  - psum 출력은 마지막 행(`psu_out[c]=col_connections[ROWS-1][c]`, 86-89행) → **weight-stationary + 세로 누산** 토폴로지.
- **`bp_dsp_core`**(`bp_dsp_core.sv` 6-177행): DSP core 래퍼.
  - COLS개 act 버퍼 + ROWS개 weight 버퍼(`WGT_DW=8`폭) + COLS개 out 버퍼, 모두 `buf_sdp`(simple dual-port)(60-121행).
  - AXI-Stream 128비트 → act는 16비트×8씩(138행), weight는 8비트×16씩(147행) 디인터리브.
  - 출력 128비트 패킹(153-173행), `dsp_sys` 인스턴스(124-131행).

#### 3.A.5 최상위 & 제어 (`fthbs_top.sv`, `sys_top.sv`)

- **`fthbs_top`**(`fthbs_top.sv` 5-274행): 3블록 구성.
  - `ps_ctrl`(143-180행): AXI4-Lite 슬레이브, ap_start/done/idle/ready + aux instr/debug 주소 레지스터 + `core_scalar[4]`/`core_status[2]`.
  - `aux_instr`(184-210행): m_axi로 **명령어(instruction)를 DRAM에서 FIFO로 prefetch**(INSTR_FIFO_DEPTH=256) → AXI-Stream으로 core에 공급. → **명령어 기반(programmable) 가속기**.
  - `sys_top`(214-271행): anga core. m_axi 3채널(iofm 입출력 feature, wgt weight) + instr stream. → LUT/DSP 두 어레이를 포함하는 통합 코어.
- → 전체는 **CPU(PS)가 명령어 시퀀스를 DRAM에 적재 → aux가 prefetch → core가 layer-by-layer 실행** 구조. 스케줄러(SW)가 이 명령어/타일링을 생성.

### 3.B 소프트웨어

#### 3.B.1 MSD 양자화 — Quantizer (`quant_modules.py`) ★핵심

MSD 양자화의 본질은 **양자화 그리드(quant_grid)를 "EB개 부호자리로 표현 가능한 값들"로 제한**하는 것. weight를 EB(effective bit, 부호자리 개수)로 제약 → HW의 IDX_DW=4 bit-serial 인코딩과 정합.

- **`quantize_csd`**(14-48행): 입력 정수를 **CSD(Canonical Signed Digit) 근사** — `eb`개의 2의 거듭제곱(twoslist `[1,2,...,128]`)으로 부호 ±를 섞어 표현(가장 가까운 거듭제곱을 반복 차감, 18-46행). → "혼합 부호자리" 표현의 핵심 함수.
- **`bit_essential`**(117-122행): 이진 문자열의 '1' 개수 = 유효 부호자리(EB) 측정.
- **`lsb_quant`/`lsb_quant_0`**(124-230행): 정수의 LSB 쪽 '1'들을 제거/추가해 **EB를 expect_eb로 강제**(부호자리 개수 제한). → RSD(restricted signed digit).
- **`Quantizer.int_value`**(380-445행): 8비트 정수 격자를 만들되 각 값을 `lsb_quant_0`/`lsb_quant`로 가공해 **모든 격자값이 expect_eb개 부호자리를 갖도록** 함(395-426행). signed면 ± 둘 다(437-438행).
- **`hamha_csd_value`**(448-486행): `quantize_csd`로 expect_eb개 자리의 CSD 격자 생성. `self.eb == "csd_eb2/3"`에서 사용(`_init_quant_para` 1156-1161행).
- **`apot_value`**(331-377행): APoT(Additive Powers-of-Two) 격자(대안 수치형).
- **`flint_value`**(561-616행): Flint(부동소수 유사) 격자(대안 수치형, ANT 계열).
- **스케일 탐색 `search_mse`**(625-664행): per-channel(weight) 또는 per-tensor(activation)로 `alpha`(클리핑 범위)를 0.01 step으로 sweep하며 MSE 최소화(637-663행). → **per-channel learnable scale + grid search**.
- **EB 자동탐색 `search_adaptive_effective_bit(_kernel)`**(757-1053행): EB 1~5 각각의 MSE를 비교해 최적 EB 선택(argsort, 856-874행). → 레이어/커널별 mixed-EB.
- **수치형 자동탐색 `search_adaptive_numeric_type`**(666-754행): int/flint/pot/float/apot 중 MSE 최소 선택(ANT, "ant-" 모드). → MSD는 ANT(Adaptive Numeric Type) 양자화를 기반으로 확장.
- **outlier 분리**(`outlier_set`/`outlier_quant`, 1055-1103행): percentile 기준으로 대부분은 int4, outlier는 int16으로 양자화(1093-1101행). → **혼합 정밀도(저비트+outlier 고비트)**.
- **forward**(`_forward`, 1389-1417행): `scale = alpha/max(grid)`로 나눈 뒤 `QuantBase.forward`(CUDA 커널 `quant_cuda.quant`, 68-83행)로 격자 양자화, STE(`(quant-data).detach()+data`, 1408행).
- **`MultiheadAttentionQuantizer`**(486행, grep 확인): ViT의 MHA(softmax는 159행 F.softmax) 양자화 지원 → **transformer 양자화 경로 존재**.

→ **요약**: weight를 "EB개 부호자리(±2^k 합)"로 제약하는 양자화 격자를 만들고, per-channel scale을 MSE로 학습. EB가 작을수록 HW bit-serial 사이클이 줄어 빠름 → **정확도-속도 트레이드오프를 EB로 제어**.

#### 3.B.2 float↔bit 변환 (`binary_converter.py`)

- `bit2float`/`float2bit`(34-116행): IEEE-754(8 exp, 23 mantissa) 변환. `integer2bit`(137-152행), `remainder2bit`(119-134행). → 양자화 격자/부호자리 분석의 보조 유틸. (MSD 핵심 알고리즘은 quant_modules 쪽)

#### 3.B.3 이종자원 스케줄러 (`scheduler.py`, `dsp_backend.py`) ★핵심

DNN을 6/7차원 for-loop `[K,H,W,C,I,J,STRD]`(출력채널/출력HW/입력채널/커널HW/stride)로 추상화하고, 타일 크기를 **브루트포스 탐색**해 최소 latency를 찾는다. 핵심은 **출력채널을 LUT 어레이와 DSP 어레이에 분할(och_lut / och_dsp)**.

- **`Scheduler.get_best_schedule_opt`**(`scheduler.py` 92-357행): 메인 DSE.
  - `tile_k`(출력채널 타일) sweep(127행) → 각 tile_k 내에서 **LUT 할당 비율** `och_lut = tile_k*0.2~0.7`(132행), `och_dsp = tile_k - och_lut`(133행). → **이종 자원 워크로드 파티셔닝**(MSD의 핵심 기여).
  - `tile_c`(입력채널)·`tile_o`(출력 feature) sweep(135-353행, C/H 크기별 분기).
  - 각 후보에 대해 `hw_model.get_opt_lat(schd_tile, och_lut, och_dsp, ess_bit)`(149행)로 buffer overflow 검사 + tile latency, `get_glb_ld_lat`/`get_glb_wr_lat`로 메모리 latency(171-174행).
  - 총 latency = `num_tiles * opt_lat_cycle + mem_ld + mem_wr`(176행), 최소값 추적(182-186행). 반환: `best_lat, best_schd, best_ratio(LUT비율), best_roof(roofline bound)`.
- 변형들: `get_best_schedule`(11-90행, EB 미고려 기본), `get_best_schedule_attention`(359-450행, **ViT attention 전용** — och_lut=1로 고정, num_o_tiles에 W도 곱함), `get_best_schedule_dpws`(452-611행, depthwise), `get_best_schedule_baseline(_dpws)`(613-1014행, DSP-only 비교군 och_lut=0).
- **`DSPBackend`**(`dsp_backend.py` 5-134행):
  - `get_compute_latency`(16-43행): GEMM 매핑(ifm_h=oh*ow, ifm_w=ic*kh*kw, wgt_w=oc). latency `= mat_wgt_h * ceil(ifm_h, rows*2) * ceil(wgt_w, cols)`(40-42행) — `rows*2`는 **DSP 2-MAC 패킹**(한 행이 2개 처리) 반영.
  - `get_compute_latency_dpws`(45-75행): depthwise(output-stationary, wgt_w=1).
  - `get_buf_util`(77-104행): ifm/wgt/ofm 버퍼 깊이로 overflow 판정.
  - (대응 `lut_backend.py`는 LUT 어레이용 — EB 사이클 반영 추정, 본 정독 범위 외이나 구조 동형.)
- `hw_model.py`가 DSP/LUT backend + mem_backend를 통합해 `get_opt_lat` 제공(scheduler가 호출).

→ **요약**: 한 conv/attention 레이어를 LUT(저EB·고병렬)와 DSP(정밀) 어레이에 출력채널 단위로 나눠 동시 실행, 타일/분할비를 DSE로 최적화 → **이종 자원 동시활용으로 이론 peak 극대화**(MSD 제목 그대로).

#### 3.B.4 PYNQ 호스트 (`host/zcu102/acc_ctrl.py`)

- **`AccCtrl`**(`acc_ctrl.py` 4-90행+): PYNQ로 가속기 구동.
  - AXI-Lite 레지스터 매핑(`top_wrapper_0/s_axi_control`, 6-10행), instr/debug 버퍼 물리주소를 레지스터에 기록(16-21행).
  - `report_latency`(61-72행): `core_status_0`(latency cycle), `core_status_1`(end flag=143)로 사이클 측정 → `lat_ms = lat_cycle/clk_freq`.
  - `run`(86-90행): 버퍼를 device로 sync 후 `ap_ctrl[0]=1`로 기동.
  - → 명령어 배열(`instr_array`)을 DRAM에 적재하고 ap_start로 실행하는 명령어 기반 구동(3.A.5 aux_instr와 정합).

---

## 4. 데이터플로우 / 실행 흐름

```
[SW QAT] quant_modules: weight를 EB개 부호자리로 양자화(per-channel MSE scale) + ViT MHA 양자화
   │  (EB 격자 = int_value/hamha_csd_value, EB 작을수록 빠름)
   ▼
[SW Scheduler] DNN→[K,H,W,C,I,J,STRD] for-loop → 타일링 DSE + 출력채널을 LUT/DSP로 분할(och_lut/och_dsp)
   │  → 명령어 시퀀스 + 스케줄 CSV
   ▼
[HW] fthbs_top: PS(AXI-Lite) → aux_instr(DRAM→FIFO prefetch) → sys_top(core)
   │
   ├─ LUT 어레이(80×80, bit-serial): weight=4b 부호자리(IDX_DW), act=8b
   │     PE: mp_lsf 시프트 + 부호반전 → EB 사이클 누산(input-stationary)
   └─ DSP 어레이(48×48, bit-parallel): weight=8b, DSP48E2로 2-MAC 패킹
         PE: weight 좌→우, act 위→아래, 세로 누산(weight-stationary)
   │
   ▼
[HW] 출력 feature → DRAM(m_axi iofm), latency를 core_status로 보고 → 호스트(acc_ctrl) 측정
```

- **메모리 계층**: DRAM(iofm/wgt/instr) ↔ AXI-Stream ↔ on-chip act/wgt/out 버퍼(`buf_sdp`, BUF_SIZE 65536) ↔ 시스톨릭 어레이 레지스터.
- **병렬화**: 두 시스톨릭 어레이 동시 가동(LUT 80×80 + DSP 48×48), DSP 2-MAC 패킹, EB 단위 bit-serial.
- **양자화/데이터타입**: act 8b, DSP weight 8b, LUT weight 4b 부호자리 인덱스, psum 16b(DSP)/8b(LUT). mixed-EB(레이어별 1~3) + outlier int4/int16.

---

## 5. HW/SW 매핑

| 소프트웨어 | 하드웨어 | 근거 |
|---|---|---|
| EB(부호자리 개수) 양자화 grid | LUT bit-serial 어레이(IDX_DW=4, EB 사이클) | quant_modules 380-486 / bit_serial_mul.sv, def.sv 20 |
| 출력채널 LUT/DSP 분할(och_lut/och_dsp) | LUT core(80×80) + DSP core(48×48) 동시 | scheduler.py 132-133 / bs_lut_core.sv, bp_dsp_core.sv |
| 타일링 [K,H,W,C,I,J,STRD] | aux_instr 명령어 시퀀스 + buf_sdp 버퍼 | scheduler.py 92-357 / fthbs_top.sv 184-210 |
| DSP 2-MAC latency 모델(rows*2) | dsp_pe DSP48E2 8b 패킹 | dsp_backend.py 40-42 / dsp_pe.sv 32-34 |
| ViT MHA 양자화 / attention 스케줄 | LUT/DSP GEMM 어레이 | multihead_attention.py 486 / scheduler.py 359-450 |
| 호스트 구동·latency 측정 | core_status 레지스터, ap_ctrl | acc_ctrl.py 61-90 / ps_ctrl |

## 6. 빌드·실행

- **SW**(software/README): `conda` env(python 3.8, PyTorch 1.12) + `pip install ./quant`(CUDA 양자화 커널) + DALI(26-29행). QAT 예: `... main.py --model=resnet18 --wbit=8 --abit=8 --eb=csd_eb2 --train`(52행). mixed-EB 탐색→QAT(60-64행). 스케줄러 `./scripts/run_scheduler.sh`(85행) → 결과 CSV가 `../hardware/host/schd_csv`로 복사(88행).
- **HW**(hardware/README): `source $vivado$/settings64.sh`(2021.2) → 보드파일 `BOARD_PART_REPO` export(31-37행) → `cd vivado/pynqz2/prj && make all`(49-53행) 또는 `source make_all.sh`(57-61행). 보드테스트 `source start_eval.sh`(69행). 호스트는 `run_eval.py`에서 `pynq.Overlay('...msd_hw_*.bit')` 경로 수정(80-85행).

## 7. 의존성

- HW: Ubuntu 18.04, Vivado 2021.2, PYNQ 2.6, DSP48E2 프리미티브(Xilinx UltraScale+). SystemVerilog.
- SW: Python 3.8, PyTorch 1.12, CUDA 11.3, 커스텀 `quant_cuda` 확장(setup.py 빌드), NVIDIA DALI(ImageNet 로더), numpy, torch.distributed(멀티GPU QAT, quant_modules 8행).

## 8. 강점 / 한계 / 리스크

**강점**
- **이종 자원 동시활용**: LUT(bit-serial 부호자리)와 DSP(bit-parallel)를 한 레이어에 분할 → DSP-only 대비 이론 peak↑(특히 DSP가 적은 소형 FPGA에서 GOPS/DSP, GOPS/kLUT 동시 개선, hardware/README 결과표).
- **EB로 정확도-속도 연속 조절**: layer-wise mixed-EB(1~3)로 부드러운 트레이드오프(software/README 128-212행 실측표). EB 작을수록 LUT 사이클↓.
- **명령어 기반 programmable 코어**: 한 비트스트림으로 여러 모델(VGG/ResNet/MobileNet/ViT) 실행 — 스케줄러가 명령어 생성.
- **ViT 지원**(MHA 양자화 + attention 스케줄러) — transformer 가속 경로 내장.
- **DSP 2-MAC 패킹**(dsp_pe.sv)으로 DSP 효율 2배.

**한계/리스크**
- **부호자리 곱 정확도**: `bit_serial_mul`의 부호 처리(`~act_in`, 49행)는 2의 보수 정확보정 없이 근사 — 비트정확 검증 필요(주석에 original/수정 버전 병기, 5-34행).
- **스케줄러 브루트포스**: tile_k×ratio×tile_c×tile_o 4중 루프 — 큰 모델에서 느림(README 스케줄러 5시간). 휴리스틱 가지치기 의존.
- 어레이 크기·버퍼 깊이가 `def.sv` 고정 → 플랫폼별 재합성 필요.
- LUT bit-serial은 EB가 크면(예: 일반 8비트) DSP보다 느릴 수 있음 — EB 제한이 정확도에 영향.
- `lut_backend.py`/`hw_model.py` 세부는 본 분석 범위 외(구조는 dsp_backend 동형 추정).

## 9. 우리 프로젝트(고처리량 ViT/Transformer FPGA + XR 시선추적) 관점 시사점

- **EB(부호자리) 양자화 ↔ Mamba/ViT 저비트 양자화**: weight를 소수의 ±2^k 합으로 제한하는 RSD/CSD 격자(quant_modules `int_value`/`hamha_csd_value`)는 **transformer weight의 곱셈을 시프트+덧셈으로 대체** — GEMM/attention의 DSP 부담을 줄이는 데 직접 재사용 가능(재사용 1순위, 우리 ViT 양자화 경로에 이식).
- **이종 자원(LUT+DSP) 동시 GEMM ↔ systolic GEMM 확장**: 출력채널/출력타일을 LUT·DSP 어레이로 분할하는 스케줄링은 HG-PIPE류 파이프라인의 **자원 활용률 극대화**에 응용. DSP 부족한 소형 FPGA(XR 엣지 디바이스)에서 특히 유효.
- **DSP48E2 2-MAC 패킹**(dsp_pe.sv 32-34행, INMODE/OPMODE): 우리 INT8 GEMM PE의 DSP 효율 2배 — 즉시 이식 가능(DPACS의 HLS 패킹과 동일 아이디어, 여기선 RTL 직접).
- **bit-serial 시스톨릭 PE**(lut_pe + mp_lsf): 가변 정밀도(EB) 연산을 사이클 단위로 trade-off — Mamba의 selective scan/저비트 연산이나 동적 정밀도 attention에 응용.
- **명령어 기반 programmable core + 스케줄러**: aux_instr prefetch + DSE 스케줄러 구조는, 우리 가속기를 "한 비트스트림 + 모델별 명령어"로 운용(ViT/시선추적 모델 교체)하는 설계 패턴으로 차용.
- **attention 전용 스케줄(get_best_schedule_attention)**: ViT MHA의 타일링 탐색 로직을 우리 attention 엔진 DSE에 참고.
- **roofline/buffer-overflow 기반 가지치기**: latency·자원 제약을 DSE 단계에서 거르는 방식 — 우리 algo2fpga DSE에 적용.

## 10. 근거 표기

- **확인**: 인용 라인은 실제 Read 기준 — RTL(def.sv, bit_serial_mul.sv, mp_lsf.sv, lut_pe.sv, common_mul.sv, dsp_pe.sv, dsp_sys.sv, bp_dsp_core.sv, bs_lut_core.sv 일부, fthbs_top.sv), SW(quant_modules.py 1-1447행, binary_converter.py, scheduler.py, dsp_backend.py), host(acc_ctrl.py 일부), 각 README.
- **추정**: (a) LUT 어레이가 EB 사이클에 걸쳐 누산(bit-serial)한다는 동작 — lut_pe/bit_serial_mul + def.sv IDX_DW=4 + EB 양자화의 정합에서 강하게 추정(타이밍 제어 lut_ctrl_ex.sv 미정독). (b) `lut_backend.py`/`hw_model.py` 세부 latency 식 — dsp_backend 동형 추정. (c) 부호자리 부호처리의 비트정확성 — RTL 주석상 보정 버전 병기로 추정.
- **확인 불가(미정독)**: `quant_modules.py` 1448-1716행(나머지 269행), `lut_backend.py`/`hw_model.py`/`mem_backend.py`/`simulator.py`/`generator.py` 본문, `sys_top.sv`/`glb_ctrl.sv`/제어 FSM(ctrl_ld/ctrl_wb/lut_ctrl_ex/dsp_ctrl_ex) 본문, pynqz2/ultrascale의 `def.sv` 파라미터 실값. 단 모듈 인터페이스/역할은 인스턴스화부(fthbs_top/bp_dsp_core/bs_lut_core)와 README로 파악.
- **플랫폼 차이**: zcu102 대표 정독. 3 플랫폼 RTL 파일명·모듈 동일, 차이는 def.sv 파라미터·BD(추정) — README 결과표의 kLUT/DSP/BRAM 차이로 뒷받침.
