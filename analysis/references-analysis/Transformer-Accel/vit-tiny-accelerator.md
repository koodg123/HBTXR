# vit-tiny-accelerator 코드베이스 정밀 분석

> 대상: `REF/Transformer-Accel/vit-tiny-accelerator`
> 분석 방법: Glob 인벤토리 + 핵심 소스 직접 Read. 라인 근거는 `파일명:라인범위`로 표기. 미확인 항목은 "추정"/"확인 불가"로 명시.
> 작성일: 2026-06-20

---

## 1. 개요

TinyViT-5M (Microsoft, 하이브리드 CNN+윈도우 어텐션 ViT)을 **Xilinx Zynq-7000 (Arty-Z7 / Z7-20급)** FPGA에 INT8로 가속하는 **HW/SW 혼합 프로젝트**다. 전 연산 코어는 **순수 Verilog RTL**로 구현되어 있고(README:339 "implemented entirely in Verilog RTL"), PyTorch 측은 **모델 정의 + 양자화(PTQ/QAT) 도구 + 추론 서버**를 담당한다.

핵심 설계 결정:
- **8×8 INT8 시스톨릭 어레이 GEMM 코어**(INT32 누산)를 모든 행렬곱(Q/K/V 투영, QKᵀ, Softmax×V, MLP FC1/FC2, PatchMerging, Classifier)에 공유.
- **Central Interconnect 아키텍처**: 단일 `scheduler_tiler` FSM이 모든 컴퓨트 모듈을 직접 조율(README:395). 이전의 계층적 attention_block/mlp_block 설계를 폐기하고 자원 공유·제어 단순화를 추구.
- 고정소수 비선형: Softmax(EXP LUT + reciprocal MSR), LayerNorm(역제곱근 LUT), GELU 대신 **ReLU 채택**(README:405, 하드웨어 단순화 목적).
- INT8 대칭 양자화 + per-channel 가중치 / per-tensor 활성, Q1.31 multiplier + shift requant(README:193, 577-605).
- AXI4-Stream(64-bit = 8×INT8) 데이터 평면, AXI4-Lite CSR 제어 평면, Xilinx AXI DMA IP를 `axi_dma_shim`으로 래핑.

성숙도는 **모듈별 편차가 크다**. GEMM/Softmax/LayerNorm/Depthwise/Requant/Residual/ReLU 등 개별 유닛은 RTL+TB+ILA까지 완성도가 높으나, 시스템 통합의 핵심인 `scheduler_tiler`는 **문법 오류 포함 미완성**(아래 §3.2 참조). README도 scheduler_tiler/central_interconnect/buffer_bank를 "Planned"로 표기(README:412-414).

---

## 2. 디렉토리 구조

### 2.1 자체 소스 (분석 대상)

```
vit-tiny-accelerator/
├── README.md                  # 설계 문서 (1087줄, 매우 상세: 아키텍처/레지스터맵/시퀀싱)
├── SCHED.md                   # 스케줄러 타일링 의사코드 (베트남어 혼용 작업노트)
├── SHAPE_TRACE.md             # 텐서 shape 추적 (미확인)
├── requirements.txt           # Python 의존성
├── setup_env.sh
│
├── fpga/
│   ├── rtl/                    # ★ 핵심 RTL (자체 구현)
│   │   ├── gemm/               # systolic_array, processing_element, gemm_core_top,
│   │   │                       #   input_buffer_controller, output_collector
│   │   ├── softmax/            # softmax_unit, msr_unit, exp_rom, softmax_fifo + lut/
│   │   ├── layer_norm/         # layer_norm, accumulator, avg_var_calc, recip_sqrt,
│   │   │                       #   final_norm_calc, beat_fifo, stats_fifo + lut/
│   │   ├── depthwise_conv/     # depthwise_conv_unit, mac_unit, line_buffer,
│   │   │                       #   kernel_buffer, srl_delay_tap
│   │   ├── requant/            # requant_unit
│   │   ├── residual/           # residual_add
│   │   ├── relu/               # relu
│   │   ├── scheduler_tiler/    # scheduler_tiler, buffer_bank, transpose
│   │   ├── axis_central_interconnect/  # axis_central_interconnect(+_system),
│   │   │                       #   axis_mux_static, axis_fifo
│   │   ├── axi_dma_shim/       # axi_dma_shim, axis_source
│   │   ├── axi_lite/           # axi_lite, axi_lite_reg
│   │   └── hdmi/               # tmds_encode, tmds_oserdes, top_DMA_VDMA (데모 출력용)
│   ├── tb/                     # 모듈별 SystemVerilog/Verilog 테스트벤치
│   ├── ila/                    # ILA top/stimulus + ila_waveform/*.png (디버깅)
│   ├── benchmarking/           # test_*_wrapper.v (fmax 측정 래퍼)
│   ├── sim/                    # Makefile, compile.f/rtl.f/tb.f (Questa/ModelSim)
│   ├── scripts/                # Vivado TCL (synth_fmax, run_timing_*, bd/, program_board)
│   ├── constraints/            # *.xdc (arty_z7 + 모듈별 OOC 타이밍 제약)
│   └── docs/                   # 모듈별 .md 사양서 + waveform/*.json
│
├── models/                    # PyTorch 측
│   ├── core/                   # tiny_vit.py(★ 모델), build.py, clip.py, remap_layer.py
│   ├── tools/                  # ptq_int8_export, qat_train, emit_requant_table,
│   │                           #   qdq_error_report, cpu_golden_infer
│   ├── inference/              # server.py(FastAPI), client_demo, video_processor
│   ├── common/                 # config.py (yacs)
│   ├── configs/                # tiny_vit_5m/11m/21m.yaml
│   ├── checkpoints/            # qat_int8/, int8_5m/ → scales.json, qdq_report.json
│   ├── Makefile
│   └── MODEL_QUANTIZATION.md
│
└── sw/                        # Zynq PS(ARM) 베어메탈/드라이버 C
    ├── sources/                # main.c, main_gemm_test.c, platform_init.c,
    │                           #   axi_dma_vdma.c, img_hex_gen.py, tests/test_*.c
    └── includes/               # axi_dma_vdma.h, font8x8_basic.h, images_raw.h
```

### 2.2 제외 항목 (third-party / vendor / 생성물 — 이름만 언급)

- **`models/core/tiny_vit.py`의 모델 정의 본체**: Microsoft TinyViT 상위 코드를 adapt한 것(tiny_vit.py:1-8 헤더 "Copyright (c) 2022 Microsoft / Adapted from LeViT and Swin Transformer"). 모델 골격은 third-party로 분류, 단 양자화 친화 fuse 로직은 자체 활용. clip.py도 OpenAI CLIP 계열 추정(확인 불가).
- **`fpga/rtl/hdmi/`의 TMDS/VDMA**: HDMI 출력 데모용 표준 IP-스타일 코드. 가속기 본체와 무관(데모 경로). 분석 비중 낮음.
- **Xilinx AXI DMA IP**: 외부 IP(README:720), 코드 미포함. `axi_dma_shim`만 자체 구현.
- **생성물/바이너리**: `*.bin`(weights.bin), `ila_waveform/*.png`, `*.mem`/`*.hex`(LUT 룩업테이블, py 스크립트로 생성), Questa `work/` 등 빌드 산출물.
- **`xilinx_lib/` (UNISIM 컴파일 라이브러리)**: sim/Makefile이 생성(sim/Makefile:99-104). 산출물.

---

## 3. 핵심 모듈 정밀 분석

### 3.1 GEMM 코어 (★ 가장 중요)

전체 네트워크의 모든 행렬곱을 담당하는 **8×8 출력-고정(output-stationary) 시스톨릭 어레이**. 4개 서브모듈로 구성.

#### 3.1.1 processing_element.v (PE, MAC 셀)

라인 근거: `fpga/rtl/gemm/processing_element.v:1-111`

- **2-스테이지 MAC 파이프라인**으로 ~196 MHz 목표(PE:26-34 주석). Stage1: 곱셈→`product_r` 레지스터(PE:43-59), Stage2: 누산 `accumulator + product_r`(PE:61-83).
- `(* use_dsp = "yes" *)`로 DSP48E1 추론 강제(PE:44-46). 곱 폭이 `ACC_WIDTH=32`로 선언되어 DSP 곱셈기 매핑.
- **시스톨릭 데이터 전파**: a는 수평(좌→우), b는 수직(상→하)으로 각 1사이클 지연 통과(PE:85-98) — 전형적 시스톨릭 토폴로지.
- **완료 검출**: `mac_count`가 `ARRAY_SIZE(=8)`에 도달하면 `acc_done` 세트(PE:75-80). 즉 각 PE는 K=8 누산 후 완료. → K 차원이 8 초과면 외부에서 타일 누적 필요(타일링은 scheduler 책임).
- `clear_acc`로 타일 시작 시 누산기 리셋(PE:71-74).

> 평가: 정석적인 INT8×INT8→INT32 MAC. DSP 파이프라인 + 셀당 완료 카운터 설계는 깔끔. 단 K=8 고정 완료 조건은 부분합 누적 흐름을 상위가 책임지게 함(아래 §3.1.4 한계).

#### 3.1.2 systolic_array.v (8×8 PE 그리드)

라인 근거: `fpga/rtl/gemm/systolic_array.v:1-438`

- `generate`로 64개 PE 인스턴스(systolic:386-421). 내부는 2D 배열 wire(`a_wire`,`b_wire`,`acc_wire`,`acc_done_wire`)로 연결하되, **포트는 전부 평탄화(flatten)**되어 있음(systolic:10-193, `a_in_0..7`, `acc_out_r_c` 64개, `acc_done_r_c` 64개). 주석 "using arrays internally is OK"(systolic:197) — 포트 배열 미지원 시뮬/합성 호환성 고려.
- `array_active`: 모든 PE의 `a_valid&&b_valid` OR 축약으로 "MAC in-flight" 표시(systolic:417-436).

> 평가: 기능상 정확. 다만 64×(32+1) 평탄화 포트는 코드량 폭증(438줄 중 대부분이 assign 나열). 파라미터화 의도는 있으나 `ARRAY_SIZE` 변경 시 포트 수동 수정 필요 — **실질적으로 8×8 하드코딩**.

#### 3.1.3 input_buffer_controller.v / output_collector.v

- **input_buffer_controller** (`fpga/rtl/gemm/input_buffer_controller.v:1-97`): AXIS 64-bit 1-beat를 8×INT8로 언팩하여 어레이 한 모서리에 공급하는 **skid 버퍼**. `present/ready_next/accept` 조합으로 동시 accept+present 안전 처리(IBC:36-43). `stream_reset`(=start_tile)으로 스트림 경계 리셋, `tlast`로 stream_done(IBC:89-93).
- **output_collector** (`fpga/rtl/gemm/output_collector.v:1-471`): 64개 acc를 `VALUES_PER_BEAT=2`(2×INT32=64bit)로 직렬화하여 AXIS 출력. 핵심 트릭은 **워터마크 트리거**: `acc_done_0_4`(중간 PE) 완료를 보고 출력 시작(OC:413-424). "절반 지점이 끝났으면 나머지가 곧 끝난다"는 안전 버퍼 논리. `beat_ready`로 비트당 2셀 완료 확인 후만 valid(OC:386-396, 429-454), `tlast`는 마지막 비트(OC:440-442).

#### 3.1.4 gemm_core_top.v (조립)

라인 근거: `fpga/rtl/gemm/gemm_core_top.v:1-503`

- buffer_a/b → systolic_array → output_collector 연결. A는 8행, B는 8열로 동일 `*_valid_beat`를 8레인에 브로드캐스트(top:162-187).
- `start_tile`이 곧 `clear_acc`이자 buffer `stream_reset`이자 output_collector start(top:95,105). `tile_done`은 output_collector의 done.

> **한계/주의**: 한 번의 `start_tile`은 8-deep K 누산 1타일만 처리(PE 완료 조건이 K=8). README 의사코드(SCHED.md:103 `k += 8`)와 일치하나, **K>8(예: common_depth=288)인 실제 레이어는 동일 누산기에 여러 타일을 누적해야** 한다. 그러나 `clear_acc=start_tile`이므로 매 start마다 누산기가 0으로 클리어 → **타일 간 부분합 누적이 top 레벨에서 자동 지원되지 않음**. 이 누적 책임이 scheduler/requant 흐름에 위임된 것으로 보이나(SCHED.md:124 `Acc += A*B`), 통합 RTL이 미완이라 **실제 K-누적 동작은 확인 불가(추정: 미완성)**.

### 3.2 scheduler_tiler.v (★ 시스템 마스터 FSM — 미완성)

라인 근거: `fpga/rtl/scheduler_tiler/scheduler_tiler.v:1-593`

- **목적**: AXI-Lite CSR(tile_cfg/layer_cfg/addr_*)을 읽어 전체 TinyViT 레이어 시퀀스를 자율 구동(DMA 로드 → compute → writeback). FSM: `S_IDLE→S_LOAD_WEIGHT→S_LOAD_INPUT→S_COMPUTE→S_STORE_OUTPUT→S_DONE`(sched:61-66, 465-538).
- **메모리 맵 하드코딩**: PING=0, PONG=25600, WEIGHT=51200, SCRATCH=57344/61440(sched:69-73). PatchEmbed/QKV/Score/Softmax/Context별 오프셋·ping-pong 더블버퍼링 주소 생성을 `get_read_base`/`get_write_base` 함수로 op_class·stage별 case 분기(sched:217-362). → **단일 온칩 BRAM에 모든 중간 텐서를 배치하는 정적 스케줄링** 설계.
- **op_class/stage_id 디코딩**: layer_cfg[27:24]=stage, [23:20]=block_role, tile_cfg[30:28]=op_class(sched:128-132). README 레지스터맵(README:509-630)과 정합.
- **타일 루프**: `tiling_idx`/`add_counter`/`max_tiles`/`max_add_loop`로 다중 타일·다중 K-누적 루프 카운팅(sched:430-448). depthwise면 conv_*, 아니면 gemm_* 핸드셰이크(sched:491-513).

> **결정적 문제 — 컴파일 불가 추정**:
> - **포트 선언 콤마 누락**: `input wire irq_enable`(sched:9) 뒤 콤마 없이 `output reg [2:0] status`(sched:10) → 문법 오류.
> - **포트 리스트 마지막 항목 뒤 콤마**: `input wire requant_valid,`(sched:51) 다음에 바로 `);`(sched:52) → 트레일링 콤마 문법 오류.
> 이 두 가지만으로도 **현 상태로는 합성/시뮬 불가**. 베트남어 주석(`// [FIX] Thay...`, sched:232)·README의 "Planned" 표기를 종합하면 scheduler_tiler는 **활발히 작업 중인 미완성 모듈**로 판단. (근거: sched:9-10, 51-52; README:412-414)

### 3.3 Softmax 유닛 (★)

라인 근거: `fpga/rtl/softmax/softmax_unit.v:1-591`, `msr_unit.v:1-100`

- **5-state FSM**: `S_IDLE→S_FIND_MAX→S_ACCUMULATE→S_CALC_RECIP→S_NORMALIZE`(softmax:37-41). **수치 안정성 위한 max-subtract** 채택(2-pass: 1패스 max 찾고 입력 FIFO 보관, 2패스 exp).
- **2-stage 파이프라인 max-tree**(8→4→1, softmax:114-148)와 **2-stage 가산기 트리**(8→4→1 exp_sum, softmax:197-216) — fmax 위해 모든 reduction을 레지스터로 분할. 곳곳에 "breaks critical path" 주석.
- **EXP**: `exp_rom`(256-entry, addr=shifted_lane≤0)로 e^(x-max) 룩업(softmax:167-185). LUT는 `lut/exp_table_q4_16.mem`(Q4.16).
- **나눗셈 회피**: `msr_unit`(Multiply-Shift-Round)이 1/sum 근사. priority encoder로 MSB 위치 찾아 shift 결정 → 64-entry recip LUT 룩업(msr:41-98). 2-cycle 파이프라인. 정규화는 `exp_pop × msr_mult >> shift`의 3-stage 파이프(softmax:282-316, 544-586).
- 카운터 12-bit로 축소(최대 4096 토큰, softmax:46-47) — fmax/면적 트레이드오프.

> 평가: **가장 정교하게 파이프라인된 모듈**. max-subtract + LUT-exp + MSR-reciprocal은 FPGA softmax 정석. backpressure(`m_axis_tready`) 완비, down-counter `tokens_remaining`로 tlast 사전계산(softmax:333-341).

### 3.4 LayerNorm 유닛 (★)

라인 근거: `fpga/rtl/layer_norm/layer_norm.v:1-206`

- **스트리밍 2-경로 분기**: 입력을 stats FIFO와 data FIFO로 동시 기록(LN:62-87). stats 경로에서 μ/σ² 계산 동안 data 경로는 FIFO에 대기 → 동기화.
- 통계 체인: `accumulator`(Σx, Σx²) → `stats_fifo`(버퍼) → `avg_var_calc`(mean/var) → `recip_sqrt`(1/√var, "Peano"/뉴턴식 근사 추정, LN:119-125) → `final_norm_calc`(γ·(x−μ)·invσ + β, DO_REQUANTIZE=1로 INT8 재양자화, LN:168-192).
- **파라미터 FIFO**(128-bit = {β,γ,invSqrt,μ})로 통계 산출 시점의 cfg_gamma/beta를 정확히 매칭(LN:142-164) — 파이프라인 in-flight 동안 cfg 변경 대비.
- μ는 2단 지연 레지스터로 latency 정합(LN:131-140).

> 평가: μ/σ²/1/√σ²를 분리 서브모듈로 깔끔히 분해. recip_sqrt 정확도/구현은 별도 파일 확인 필요(미Read, 추정: LUT+뉴턴).

### 3.5 Depthwise Conv 유닛 (★ 가장 복잡한 단일 모듈)

라인 근거: `fpga/rtl/depthwise_conv/depthwise_conv_unit.v:1-1095`, `mac_unit.v:1-155`

- 3×3 depthwise(LocalConv/MBConv용). 8채널 병렬(LANES=8), INT8→INT32.
- **line_buffer**(3행+1 순환 = mod-3/mod-4 인덱싱, DW:264-269) + **kernel_buffer**(9탭×채널) + **SRL32E 지연탭**(`srl_delay_tap`)으로 3×3 윈도우 슬라이딩 조립(DW:757-934).
- **입력 FIFO**(BRAM, IN_FIFO_DEPTH=MAX_WIDTH×MAX_CHAN_BEATS, DW:101-114)로 AXIS↔line_buffer 디커플.
- **제로패딩 처리**: is_top/bottom/left/right_col 경계 검출로 윈도우 9탭 중 패딩 위치를 0으로(DW:936-950).
- **이중 뱅크(a/b) SRL + prefetch**: `shift_bank_sel` 토글로 현재 행 처리 중 다음 행 프리페치(DW:553-558, 765-770) — 행 경계 stall 제거.
- **mac_unit**(`mac_unit.v`): 9탭을 9-stage 파이프라인으로 누산, II=1(mac:41-153). 각 stage가 한 커널 위치 처리, `(* use_dsp *)` 곱셈.
- 출력 직렬화 FIFO + `output_enable` watermark(DW:230-243).

> 평가: **방대한 파이프라인 부기(bookkeeping)**(1095줄, d/dd/ddd 지연 레지스터 다수). 정확하면 고성능이나 검증 난이도 최상. `cfg_channels >= LANES` 가정(DW:284), MAX_WIDTH=28/MAX_CHANNELS=128 고정 — Stage1(56×56) 처리는 타일 분할 필요(추정).

### 3.6 Requant 유닛 (★ 양자화 핵심)

라인 근거: `fpga/rtl/requant/requant_unit.v:1-369`

- **두 모드**: (A) 2×INT32→8×INT8 패킹(`cfg_mode_int32`, GEMM 출력용), (B) 8×INT8 패스스루 재스케일(softmax/conv 출력용).
- **per-channel scale/bias RAM**(`sb_mem`, 64-bit=[scale_q31, bias_int32], 최대 512채널, RU:44-84). AXIS로 테이블 로드.
- **requant_lane 함수**(RU:139-167): `prod=(acc+bias)*scale_q31` → `aligned=prod>>>31`(Q1.31 정렬) → `round_shift_rne64`(round-to-nearest-even) → saturate[−128,127]. README 사양(README:577-605)과 정합. RNE 라운딩 정확 구현(RU:113-136).
- INT32 모드는 짝수 채널 인덱싱(chan_ptr ±2), depth-2 출력 FIFO로 backpressure.

> 평가: **수학적으로 PTQ 익스포트 파이프와 정확히 매칭**(아래 §6). RNE + per-channel + 포화 모두 갖춤. 양자화 정합성 측면에서 신뢰도 높은 모듈.

### 3.7 Residual / ReLU / Central Interconnect / DMA Shim

- **residual_add** (`residual/residual_add.v:1-89`): 8레인 포화 INT8 가산. **lock-step join**(양 스트림 valid 시에만 ready, RA:39-41)으로 프레임 정렬. 부호확장 오버플로 검출 포화(RA:44-61).
- **relu** (`relu/relu.v:1-42`): 순조합. 각 INT8 레인 MSB(부호)가 1이면 0, 아니면 통과(relu:27-29). valid/ready/last 완전 패스스루. GELU 대체(README:405).
- **axis_central_interconnect** (`axis_central_interconnect/axis_central_interconnect.v:1-152`): 6소스×7목적지. 목적지마다 `axis_mux_static`+`axis_fifo` 쌍 생성(ACI:60-121). 7개 sel_*(ext/norm/relu/gemm_a/gemm_b/resid_a/resid_b)로 라우팅(ACI:48-55). ready 집계는 "선택된 모든 목적지의 FIFO ready AND"(ACI:127-150).
- **axi_dma_shim** (`axi_dma_shim/axi_dma_shim.v:1-392`): Xilinx AXI DMA를 Direct Register Mode로 구동하는 8-state FSM(DMACR→SA/DA→LENGTH→poll DMASR→ACK IRQ→DONE, shim:116-389). AXIS는 단순 패스스루(MM2S/S2MM, shim:81-94). **주의**: `s_axis_tready=1'b1` 하드와이어(shim:87) — DMA로부터 항상 수락. 백프레셔 미전파 가능성(추정 위험).

---

## 4. 데이터 플로우

### 4.1 시스템 레벨 (README §5.3, §7 + scheduler 근거)

```
PS(ARM) --AXI-Lite--> axi_lite_reg --> scheduler_tiler (마스터 FSM)
                                          │
   ┌──────── DMA 커맨드 (addr/len/dir) ───┘
   ▼
axi_dma_shim --> Xilinx AXI DMA IP <==DDR==>  (MM2S: 가중치/입력 / S2MM: 결과)
   │ MM2S AXIS(64b)
   ▼
axis_central_interconnect (op_class 기반 라우팅)
   ├─> gemm_core (A,B) → INT32 결과 → requant_unit → INT8
   ├─> softmax_unit / layer_norm / relu / residual_add / depthwise_conv
   └─> buffer_bank (온칩 중간 텐서, ping/pong)
```

### 4.2 어텐션 1블록 (op_class 시퀀스, README:428-436 / SCHED.md §4.2)

```
[Norm] LayerNorm
→ op 000: Q/K/V proj = GEMM(tokens × W_qkv) → requant → buffer
→ op 001: QKᵀ        = GEMM(Q × K)          → softmax
→ op 010: softmax    = exp/sum/normalize    → buffer(score)
→ op 011: ctx        = GEMM(score × V)      → requant → buffer
→ op 110: residual1  = attn_out + input     → buffer
[LocalConv] depthwise_conv (3×3)
→ MLP: op 100 FC1 → requant → ReLU → op 101 FC2 → requant → op 011 residual2
```

### 4.3 GEMM 타일 내부 (검증된 RTL 흐름)

`start_tile` → buffer_a/b skid가 매 beat 8×INT8 공급 → 8×8 PE가 시스톨릭 누산(K=8) → 각 PE `acc_done` → output_collector가 `acc_done_0_4` 워터마크 후 2×INT32/beat 직렬화 → `tile_done`.

---

## 5. HW/SW 매핑

| TinyViT 연산 | HW 모듈 | 근거 |
|---|---|---|
| PatchEmbed / PatchMerging (Conv) | scheduler_tiler가 GEMM으로 im2col 매핑 | SCHED.md:43-130, 319-351 |
| Q/K/V 투영, QKᵀ, Softmax×V, MLP FC1/2, Classifier | gemm_core (8×8 systolic) | README:760-783, gemm_core_top.v |
| Softmax | softmax_unit + msr_unit + exp_rom | softmax_unit.v |
| LayerNorm | layer_norm (+accumulator/avg_var/recip_sqrt/final_norm) | layer_norm.v |
| GELU → **ReLU 대체** | relu | relu.v, README:405 |
| LocalConv / MBConv depthwise 3×3 | depthwise_conv_unit + mac_unit | depthwise_conv_unit.v |
| Residual add (skip) | residual_add | residual_add.v |
| INT32→INT8 재양자화 | requant_unit | requant_unit.v |
| 데이터 이동 (DDR↔PL) | axi_dma_shim + Xilinx AXI DMA | axi_dma_shim.v, README:720 |
| 제어/상태 | axi_lite_reg + scheduler_tiler | README:439-651 |

**SW(PS) 역할**(README:352-360): DDR 버퍼 할당, CSR 프로그래밍(addr/tile_cfg/layer_cfg/requant), start 트리거, IRQ/timeout 처리, argmax 후처리. 베어메탈 C(`sw/sources/main.c`, `axi_dma_vdma.c`, 모듈별 `tests/test_*.c`).

**PyTorch(오프라인) 역할**: 모델 학습/양자화 → `weights.bin` + `scales.json` 생성(ptq_int8_export.py) → 레이어별 requant 테이블(emit_requant_table.py) → CSR 값으로 변환.

---

## 6. 빌드·실행

### 6.1 RTL 시뮬레이션 (Questa/ModelSim)

- `fpga/sim/Makefile`: `make all TESTNAME=tb_<unit>` (clean→build→run). `compile.f`/`rtl.f`/`tb.f` 파일리스트. **Questa 전용 명령**(vlib/vmap/vlog/vsim, sim/Makefile:48-71). 커버리지(`*_cov`)·UNISIM 라이브러리 빌드(`xilinx_libs`, sim/Makefile:99-104) 지원. UNISIM 필요(DMA/SRL 등 Xilinx 프리미티브 시뮬용).
- 모듈별 TB는 `fpga/tb/<unit>/tb_*.v`, Python 골든 비교는 `tb/softmax/softmax_distribution_tests.py`, `tb/depthwise_conv/depthwise_conv_tests.py`.

### 6.2 합성/타이밍/구현 (Vivado)

- `fpga/scripts/`: `synth_fmax.tcl`(OOC fmax 측정), `run_timing_<unit>.tcl`, `bd/AXI_DMA_system.tcl`(블록디자인), `program_board.tcl`. 보드 `arty_z7.xdc` + 모듈별 OOC `.xdc`.
- 목표 Fmax: 초기 150 MHz, 최적 180-200 MHz(README:376). PE/softmax/layer_norm 파이프라인 분할이 이 목표를 위함.

### 6.3 모델/양자화 (Python)

- `models/Makefile` + tools: `qat_train.py`, `ptq_int8_export.py`(PTQ 캘리브 + INT8 익스포트), `cpu_golden_infer.py`(골든), `qdq_error_report.py`, `emit_requant_table.py`.
- 추론 데모: `models/inference/server.py`(FastAPI + websockets), `client_demo.py`, `video_processor.py`.
- 의존성: `requirements.txt`(timm==0.4.12, yacs, fastapi/uvicorn/opencv 등).

---

## 7. 의존성

- **HW 외부 IP**: Xilinx AXI DMA IP(필수, 미포함), UNISIM 시뮬 라이브러리(SRL32E 등). 도구: Questa/ModelSim(sim), Vivado(synth/impl). 타깃 Zynq-7000 / Arty-Z7.
- **SW**: Xilinx 베어메탈 BSP(platform.h/platform_init.c), AXI DMA 드라이버(`axi_dma_vdma.c/.h`).
- **Python**: PyTorch + torchvision, timm==0.4.12(모델), yacs(config), numpy, tqdm, FastAPI/uvicorn/opencv(추론 서버).
- **내부 결합**: scheduler_tiler가 axi_lite_reg/axi_dma_shim/gemm/conv/requant와 강결합. central_interconnect가 모든 컴퓨트 모듈의 AXIS 허브. requant_unit↔ptq_int8_export/emit_requant_table는 수치 포맷(Q1.31, RNE, per-channel)으로 정합.

---

## 8. 강점·한계

### 강점
- **개별 컴퓨트 유닛의 RTL 완성도·파이프라인 품질이 높다**. Softmax(max-subtract+LUT+MSR), LayerNorm(서브모듈 분해), Requant(RNE+per-channel+포화), Depthwise(이중뱅크 SRL+prefetch)는 production-grade에 근접.
- **검증 인프라가 충실**: 모듈별 TB + Python 골든 + ILA 파형 + OOC 타이밍 스크립트 + 커버리지 흐름.
- **양자화 정합성**: RTL requant 수식이 PyTorch 익스포트(scale_q31, RNE, per-channel weight / per-tensor act)와 일치 → HW/SW 수치 신뢰성.
- **상세한 설계 문서**(README 1087줄: 레지스터맵, 비트필드, op_class 시퀀싱)로 의도 추적 가능.
- **자원 공유형 단일 GEMM 코어** 전략으로 작은 Zynq에서 면적 효율 추구.

### 한계
- **시스템 통합 미완성**: `scheduler_tiler`가 **문법 오류로 컴파일 불가 상태**(sched:9-10, 51-52). README도 scheduler/interconnect/buffer_bank "Planned"(README:412-414). → **현재는 모듈 단위 검증 단계, 풀-네트워크 end-to-end 동작 미확인(추정: 미완)**.
- **GEMM 코어는 8×8·K=8 단일 타일**만 자체 처리. `clear_acc=start_tile`이라 타일 간 부분합 누적이 top에서 자동 지원 안 됨 → K>8 누적 흐름이 scheduler에 의존하나 그 scheduler가 미완(§3.1.4).
- **하드코딩 다수**: 메모리 맵 상수(sched:69-88), 어레이 8×8 포트 평탄화, depthwise MAX_WIDTH=28/MAX_CHANNELS=128. 재파라미터화 부담.
- **axi_dma_shim `s_axis_tready=1`** 하드와이어(shim:87) → MM2S 백프레셔 미전파, 가속기 stall 시 데이터 유실 위험(추정).
- 작업 노트 수준의 베트남어/영어 혼용 주석·미해결 TODO("STUCK HERE", SCHED.md:78)로 일부 흐름은 의도만 존재.
- 처리량 목표가 ≥1 FPS(stretch 2-5 FPS, README:337)로 **고처리량이라기보다 임베디드 실증 수준**.

---

## 9. 우리 프로젝트(PRJXR-HBTXR) 시사점

우리 프로젝트는 **고처리량 ViT/Transformer FPGA 가속기(HG-PIPE 계열) + XR 시선추적**으로 추정된다. 이 repo는 HG-PIPE 같은 **완전 파이프라인·고처리량** 아키텍처와는 결이 다른 **단일 공유 GEMM + 정적 스케줄링(시간다중화)** 설계이므로, "하지 말아야 할 것"과 "재사용할 것"을 분리해 참고할 가치가 크다.

### 9.1 직접 재사용·차용 가능 (검증된 모듈)
- **Softmax 유닛**: max-subtract + exp LUT(Q4.16) + MSR reciprocal + 다단 파이프라인. XR 어텐션의 softmax 블록에 거의 그대로 차용 가능. 나눗셈 회피(MSR) 패턴은 면적/주파수에 유리.
- **Requant 유닛**: Q1.31 + RNE + per-channel scale/bias RAM + 포화. 우리 양자화 파이프(per-channel INT8)와 정합시키기 쉬움. PTQ 익스포트(ptq_int8_export.py)↔RTL requant 수식 일치 패턴은 그대로 채택 권장.
- **LayerNorm 서브모듈 분해**(accumulator/avg_var/recip_sqrt/final_norm + 파라미터 FIFO)와 **PE의 2-stage DSP 파이프라인** 패턴.

### 9.2 아키텍처 교훈 (반면교사)
- **공유 단일 GEMM + 정적 스케줄러는 통합 난이도가 매우 높다**. 이 repo의 최대 병목이 바로 scheduler_tiler 미완성. HG-PIPE식 **레이어별 전용 스테이지 완전 파이프라인**이 고처리량·통합 단순성 양면에서 우월함을 역으로 확인. → 우리 프로젝트는 시간다중화 GEMM보다 **공간 펼침(spatial unrolling) 파이프라인**을 우선 고려.
- **clear_acc=start_tile 같은 단순화가 K-누적을 막는다**: GEMM 코어 설계 시 **부분합 누적/타일 경계 처리를 datapath에 내장**해야 함(스케줄러에 떠넘기지 말 것).
- **포트 평탄화 64개 나열은 유지보수 재앙**: SystemVerilog `interface`/packed array/2D 포트로 추상화 권장.
- **백프레셔를 끝까지 전파**: axi_dma_shim의 `tready=1` 하드와이어 같은 단축은 통합 단계에서 데이터 유실로 이어짐. 전 경로 valid/ready 무결성 유지.

### 9.3 XR 시선추적 관점
- 이 repo에 **시선추적 특화 로직은 없음**(ImageNet 분류 TinyViT). 단, HDMI/VDMA 출력 경로(`fpga/rtl/hdmi/`)와 video_processor.py는 **저지연 영상 입출력 파이프라인** 참고용으로만 가치. XR의 실시간(저지연) 요구에는 이 repo의 ≥1 FPS 목표가 부족하므로 throughput 목표 재설정 필요.
- 양자화 워크플로(QAT/PTQ → scales.json → CSR)는 시선추적 백본을 INT8로 내릴 때 그대로 이식 가능한 검증된 흐름.

### 9.4 검증 인프라 차용
모듈별 TB + Python 골든 비교 + ILA 파형 + OOC fmax 스크립트(synth_fmax.tcl) + Questa 커버리지 흐름은 **그대로 우리 RTL 검증 표준으로 채택**할 만하다.

---

## 부록: 근거 표기 요약

- **확인(라인 근거)**: GEMM(PE/array/top/IBC/OC), Softmax/MSR, LayerNorm, Depthwise/MAC, Requant, Residual, ReLU, Central Interconnect, DMA Shim, scheduler FSM 골격·문법오류, PTQ 익스포트/requant 테이블 — 모두 직접 Read.
- **추정**: K>8 누적 실동작, depthwise 대형 해상도 타일링, recip_sqrt 내부 알고리즘, axi_dma_shim 백프레셔 위험, end-to-end 풀네트워크 동작.
- **확인 불가**: SHAPE_TRACE.md 내용, clip.py 출처, buffer_bank/transpose 상세(미Read), hdmi 경로 상세, sw/main.c 전체 시퀀스(헤더만 인지), QAT 학습 결과 정확도.
