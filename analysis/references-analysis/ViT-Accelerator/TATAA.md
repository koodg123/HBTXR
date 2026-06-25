# TATAA 정밀 분석

> 대상: `REF/ViT-Accelerator/TATAA`
> 분석 방식: 자체 소스(HLS/RTL + Python) 라인 단위 정독. 벤더/생성물(`.gen`, `.ip_user_files`, Xilinx IP `axis_register_*`, `axis_*_v1_1_vl_rfs.v` 등) 제외.
> 표기 규칙: 코드 라인으로 직접 확인한 사실은 단정, 추론은 "추정", 코드/문서에 없는 것은 "확인 불가"로 명시.

---

## 1. 개요

- **목적**: Transformer(특히 ViT/DeiT) 추론을 FPGA에서 **정수(INT8)–부동소수점(bfloat16) 혼합정밀도**로 가속. 핵심은 **하나의 연산 유닛(systolic array의 PE)을 모드 전환만으로 INT8 MAC ↔ bfloat16 곱/덧셈/역제곱근으로 "변형(transform)"** 시켜, 선형 연산(GEMM)은 INT8로, 비선형/어텐션 연산(Softmax·LayerNorm·GELU)은 bfloat16로 같은 하드웨어에서 수행하는 것.
- **한줄요약**: "DSP48E2 기반 4×16 systolic PE 어레이를 `mode_sel` 2비트로 INT8 행렬곱 / bfloat16 곱·덧셈·isqrt 중 하나로 재구성하는 Transformable Arithmetic 구조 + 이를 구동하는 커스텀 ISA·컴파일러·혼합정밀 양자화 도구."
- **원논문**(루트 README.md L1–L44에서 확인): *"TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture"*, **ACM Transactions on Reconfigurable Technology and Systems (TRETS)**, vol.18 no.1, articleno 14, **2025**, DOI `10.1145/3714416`. 저자 Jiajun Wu, Mo Song, Jingmin Zhao, Yizhao Gao, Jia Li, Hayden Kwok-Hay So (HKU). 키워드: *transformable arithmetic architecture, mixed integer-floating-point inference, systolic array, SIMD, FPGA*.
- **타깃 디바이스**: **Xilinx/AMD Alveo U280** (hardware/README.md L3에서 명시). Vitis **2023.2** 환경 강제(vitis_kernel/README.md L4). HBM 사용(host에서 instr/yizo/xi base address 설정). 호스트는 **XRT**(`xrt::ip`) 기반(host_flow.cpp).
- **3축 구성**(루트 README.md): `hardware/`(HLS/RTL 커널), `quantization/`(PyTorch 혼합정밀 양자화), `compilation/`(모델 파싱 → 명령어 생성). README 5개 위치 확인: 루트·hardware·hardware/vitis_kernel·quantization·compilation. 단, `quantization/README.md`와 `compilation/README.md`는 **제목 한 줄만 존재**(각각 "# Quantization of TATAA", "# Compiler of TATAA")로 사실상 내용 없음(확인됨).

> 주의: 작업 지시의 "HLS 커널(.cpp/.h)" 가정과 달리, **hardware 연산 커널은 HLS C++가 아니라 직접 작성한 Verilog/SystemVerilog RTL**이다. `hardware/host/`에만 `.cpp/.h`가 있고, 이는 호스트 소프트웨어(XRT)일 뿐 커널이 아니다. Vitis는 RTL을 `.xo` 커널로 패키징하는 용도로만 쓰인다(vitis_kernel/README.md L1–L18).

---

## 2. 디렉토리 구조 (자체 소스 3축)

```
TATAA/
├─ README.md                      # 논문/3축 개요 + ACM citation
├─ hardware/
│  ├─ README.md                   # Alveo U280, host/rtl/vitis_kernel 설명
│  ├─ host/                       # 호스트 SW (XRT, C++)
│  │  ├─ host.cpp, host_flow.cpp
│  │  ├─ libs.h, param_def.h, axi_control_regs.h, mem_tag.h
│  ├─ rtl/                        # ★ 핵심 RTL (Verilog/SystemVerilog) - sim 포함 최신본
│  │  ├─ tapu.sv                  # Transformable Arithmetic PU (1개 = 4×16 PE)
│  │  ├─ pe_sys.sv                # 4행×16열 PE 어레이
│  │  ├─ pe_stg_0.sv ~ pe_stg_3.sv# ★ 4단 PE = INT8 MAC / bfloat16 분해 파이프라인
│  │  ├─ proc_core.sv             # 8×TAPU + DMRFX/Y + dm_quant + transpose + S2MM
│  │  ├─ dm_quant.sv              # ★ Dual-mode 양자화 (int8↔fp16 4모드)
│  │  ├─ exec_ctrl_int8.sv        # systolic 실행 컨트롤러
│  │  ├─ core_instr_ctrl.sv       # ★ 64-bit ISA 디코더 → pccmd 생성
│  │  ├─ pccmd_ctrl.sv            # 32-bit micro-cmd 디코더 (proc 측)
│  │  ├─ dmrf_x.sv, dmrf_y.sv     # Dual-mode Register File (활성/가중치)
│  │  ├─ data_loader.sv, instr_loader.sv  # AXI datamover 제어
│  │  ├─ bs_rsf.sv, lzc_48b.sv    # 우측시프트(스케일), 48b leading-zero count
│  │  ├─ sm_twos_convert.sv, twos_sm_convert.sv  # 부호크기↔2의보수
│  │  ├─ transpose_int.sv, zout_ctrl.sv, dm_quant.sv ...
│  │  ├─ delay_chain.sv, fifo_common.sv, fifo_axis.sv, bram_sdp_wrapper.sv
│  │  ├─ proc_kernel.v, mem_kernel.v, ps_ctrl.v, tata_top_wrapper.v
│  │  └─ sim_tata_top.sv          # 시뮬 top
│  └─ vitis_kernel/
│     ├─ README.md
│     ├─ tata_int8os_proc/        # 연산 커널 (proc) - rtl/ + prj/(Vivado, *.gen 제외)
│     └─ tata_int8os_mem/         # 메모리 IO 커널 (mem) - rtl/ + prj/
│        (rtl/ 내용은 hardware/rtl/와 거의 동일; *.gen·ip_user_files는 생성물=제외)
├─ quantization/
│  ├─ README.md (제목만)
│  └─ hlbfp_quantization/
│     ├─ hlbfp_bfloat16_bs16/     # ★ 알고리즘 양자화 (BFP bs16 + bfloat16)
│     │  ├─ hlbfp_format.py       # HybridLowBlockFP 포맷 정의
│     │  ├─ quant_function.py     # BFPQuantFunction, bfp_TD1~TD4
│     │  ├─ quant_module.py       # HMQLinear/Conv2d/Softmax/GELU/LayerNorm/isqrt
│     │  ├─ format_config.py      # 레이어별 mixed-precision 포맷 config
│     │  ├─ para_config.py        # BLOCK_DIM=4, BLOCK_SIZE=16
│     │  ├─ hlbfp_vit.py, hlbfp_sub_layers.py, hlbfp_quant_run.py, utils.py
│     └─ hlibf_bfloat16_int8/     # ★ HW-faithful 양자화 (int8 + bfloat16)
│        ├─ hlibf_vit/  hlibf_bert/  hlibf_swin/  hlibf_gpt2/  (모델별)
│        │   각: quant_module.py, build_observer.py, build_quantizer.py,
│        │       config.py(BitType), *_run.py, sub_layer(s).py
└─ compilation/
   ├─ README.md (제목만)
   └─ parser/
      ├─ parser_module/           # ★ 범용 파서 프레임워크
      │  ├─ model_parser.py, layer_parser.py, model_spec.py
      │  ├─ node_param.py, operation_param.py, base_param.py
      ├─ vit/                     # DeiT/ViT 파서 (quantization과 동일 파일 다수 재사용)
      ├─ bert/, swin/, gpt2/      # 모델별 파서
      └─ */run_*.py, run_parser.py
```

**제외물(이름만 언급, 미분석)**: `tata_int8os_*/prj/*.gen/**`, `*.ip_user_files/**`(Vivado 생성물), Xilinx IP RTL(`axis_register_8/32/256.v`, `axis_register_slice_v1_1_vl_rfs.v`, `axis_infrastructure_v1_1_vl_rfs.v`, `glbl.v`), 합성 산출물(`.xo/.xclbin/.bit` — 리포에 미존재 확인). `.cl`(OpenCL) 파일: **없음**(확인됨).

---

## 3. 핵심 모듈·파일별 정밀 분석 (가장 중요)

### 3.1 TAPU = Transformable Arithmetic Processing Unit (`tapu.sv`)

`tapu`(L5–L106)는 TATAA 연산의 최소 묶음 단위다. 파라미터: `LEFT_WIDTH=8`(활성 X, INT8), `TOP_WIDTH=48`/`BOTTOM_WIDTH=48`(세로 데이터/누산 48b), `PRELD_WIDTH=16`(Y 프리로드), `COLS=16`(L6–L12). 내부는 **4행 × 16열 PE systolic 어레이**(`pe_sys`, L85–L103).

- **모드 신호** `mode_sel_in[1:0]`(L22 주석): `00=matrix multiplication(INT8)`, `10=fp mul`, `11=fp add`. (pe_stg_0 L17 주석에서 `01=fp mag/isqrt`도 등장 → 후술.)
- **systolic skew(지연사슬)**: X 입력은 `delay_chain`으로 `TAPU_IDX*4 + idx_row` 사이클 지연(L40–L51) → 8개 TAPU가 세로로 cascade될 때 systolic 타이밍을 맞춤. Y 입력은 **TAPU_IDX==0일 때만** matmul 모드에서 열별 지연(L55–L72)되며, **FP 모드에서는 지연 없이 그대로**(`mode_sel_in==2'b00 ? sys_y_in_matmul : y_in`, L69). 즉 systolic skew는 INT8 GEMM에서만 활성, FP 모드에서는 SIMD-broadcast 형태로 동작(추정: 논문의 "systolic + SIMD" 듀얼 데이터플로우).

### 3.2 PE 어레이 (`pe_sys.sv`)

`pe_sys`(L5–L147)는 16개 열을 generate로 펼친다(L55). 각 열은 **세로로 `pe_stg_0 → pe_stg_1 → pe_stg_2 → pe_stg_3` 4단**을 cascade(L71–L143). 데이터 흐름:
- 가로(systolic): `left_in → right_out` 1단 레지스터(각 stage L78–L80), 행 연결 `row_connection_*`(L40–L43, L85·L103·L121·L139).
- 세로: `top_in → bottom_out`, 즉 `col_connection_01/12/23`로 stage0→1→2→3 연결(L88·L105·L123·L141). **위 단의 출력(bottom)이 아래 단의 입력(top)이 됨.**
- 제어신호(`y_sel`, `sys_buf_en`, `mode_sel`, `psu_clr`)는 열별 1단 레지스터로 팬아웃 최적화(L46–L69, `(* keep = "true" *)`).

핵심: **4단은 INT8 모드에서는 4개의 독립 MAC(부분합 cascade 누산)이지만, FP 모드에서는 1개 bfloat16 연산을 4단계로 분해하는 파이프라인**으로 동작한다. 이 이중 의미가 "transformable arithmetic"의 본질이다.

### 3.3 PE 4단 데이터패스 — INT8 MAC ↔ bfloat16 분해 (`pe_stg_0~3.sv`) ★

각 stage는 **Xilinx DSP48E2 프리미티브 1개**를 직접 인스턴스화(`pe_stg_0.sv` L204–L313 등). 모든 stage가 동일 DSP 설정(`AMULTSEL="AD"`, `USE_MULT="MULTIPLY"`, `USE_SIMD="ONE48"`, `OPMODE=9'b00_011_01_01`)을 공유하되, 입력 포트(A/B/C/D)에 들어가는 값을 `mode_sel`로 멀티플렉싱하여 의미를 바꾼다.

**(A) INT8 행렬곱 모드(`mode_sel==00`)** — 모든 stage 공통 패턴:
- DSP는 **하나의 PE에서 2개의 INT8 MAC을 동시 수행**(SIMD pack). `dsp_a_in = {sext, y_reg[15:8], 19'd0}`(상위 가중치, pe_stg_0 L92), `dsp_d_in = {sext, y_reg[7:0]}`(하위 가중치, L121), `dsp_b_in = {sext, left_in[7:0]}`(활성, L102). 즉 pre-adder가 아닌 **DSP의 A·D 두 포트에 두 개의 INT8 가중치를 실어** B(활성)와 곱해 2-way INT8 곱을 한 DSP로 처리(추정: 논문의 "INT8 packing"). `dsp_c_in = psu_acc_reg`로 누산(L113, L120).
- `psu_acc_reg`(48b)는 부분합 레지스터: `psu_clr`이면 0, `sys_buf_en`이면 top_in(상위 TAPU 결과)을 load, 아니면 `dsp_p_out` 누산(L315–L325). `y_reg`는 `y_sel_in`으로 Y 타일 2개 중 선택(L74–L75) → 더블버퍼.

**(B) bfloat16 모드(`mode_sel==10/11/01`)** — 4단 분산 FPU:
- **stage 0 = 부호크기/지수 추출 + 2의보수 변환**: bfloat16 입력 2개 `fp_alpha=top_in[15:0]`, `fp_beta=top_in[31:16]`(L33–L34). 가수에 hidden bit 복원 `ext_man_0={sign,1'b1,man[6:0]}`(L144), `sm_twos_convert`로 부호크기→2의보수(L147–L155), 지수 `[14:7]` 추출(L189). isqrt(`mode 01`)에서는 **`dsp_d_in = 27'h0005f37`**(L123) — 이는 **fast inverse square root의 매직 넘버**(Quake식 0x5f37, bfloat16용)이며, 소프트웨어 `quant_module.py`의 `isqrt()` 함수(`0x5f37`, L221)와 정확히 일치. `dsp_c_in = -48'sd127`(L111)로 bfloat16 지수 바이어스(127) 보정. 출력 `bottom_out`은 `{exp1, exp0, dsp_p[9:0], twos_man1, twos_man0}`(L333–L335)로 재패킹.
- **stage 1 = 지수 연산/가수 정렬 선택**: `fp mul`이면 지수 합 처리 후 가수 통과(L246–L247). `fp add`이면 **지수차 `exp_diff`로 어느 가수를 시프트할지 결정**(L99–L110): 큰 지수 쪽을 `man_tobe_remained`, 작은 쪽을 `man_tobe_shifted`로 분기(L108–L109), `twos_sm_convert`로 지수차를 부호크기화(L113–L116). 결과 패킹(L248–L252).
- **stage 2 = 가수 곱(mul) / 배럴시프트(add)**: `fp add`는 `fpadd_man_shift_bits`(0~7)를 **LUT로 9비트 곱셈 상수(`9'b0100_0000 >> n`)로 변환**(L47–L67)하여 DSP 곱으로 우측시프트를 구현(`dsp_b_in = shift_mult`, L91). `fp mul`은 두 가수를 DSP로 곱함(`dsp_a_in=top_in[8:0]`, `dsp_b_in=top_in[17:9]`, L81·L89). 결과 패킹(L271–L276).
- **stage 3 = 정규화(normalization)**: 곱/덧셈 결과의 leading bit를 검사(`fp_normdet`, L97–L100)해 가수 시프트·지수+1 보정(L112–L115), `twos_sm_convert`로 부호크기 복원, 최종 bfloat16 `{sign, exp(8), man(7)}` 패킹(L254). `mode 00`이면 단순 top_in 통과(L251–L252)하여 INT8 누산값을 그대로 아래로.

> 요약: **DSP 4개(=한 PE 열)가 INT8에서는 4-deep MAC accumulator, bfloat16에서는 [추출→지수정렬→가수곱/시프트→정규화]의 1-FP-op 파이프라인**으로 재구성된다. DSP·라우팅·BRAM이 100% 공유되므로 모드 전환에 추가 연산 자원이 들지 않는다 — 이것이 TATAA의 핵심 효율 논거(추정).

### 3.4 보조 산술 모듈

- `sm_twos_convert.sv`(L4–L19): 9비트 부호크기→2의보수. `twos_sm_convert.sv`(L4–L20): 역변환, `DW` 파라미터화. FP 가수 처리 전반에 사용.
- `lzc_48b.sv`: 48비트 leading-zero count → `right_shift_bits`(int→fp 정규화용, dm_quant에서 사용).
- `bs_rsf.sv`: 48비트 입력을 `b`(지수 스케일)만큼 우측 산술 시프트(양자화 스케일 적용).
- `delay_chain.sv`: `LEN` 파라미터 사이클 지연(systolic skew·파이프라인 정렬 전반). `BFLOAT_LAT_CYCLE=16`(proc_core L43) → bfloat16 1회 연산 레이턴시 16사이클(추정).

### 3.5 Processing Core (`proc_core.sv`) — 전체 데이터패스 통합 ★

`proc_core`(L3–L832)는 **8개 TAPU**(`TAPU_NUM=8`, L17)를 묶는다. 8×(4×16) = **세로 32행 × 16열 = 512 PE**(=512 INT8 MAC 베이스, FP 모드는 다른 매핑). 주요 블록:

- **DMRFX**(`dmrf_x`, L145–L169): 활성 X용 Dual-mode Register File. `X_BRAM_SIZE=262144`, 256b 폭, 더블버퍼 타일(`load/exec_tile_sel`). 32포트(`X_SYS_PORT_NUM=32`)로 32행에 INT8 활성을 공급.
- **DMRFY**(`dmrf_y`, L261–L349): 가중치/FP벡터용. **TAPU0은 X와 같은 큰 크기**(INT8 GEMM용, L261–L292), **TAPU1~7은 작은 FP RF**(`Y_BRAM_SIZE=8192`, L305–L348). FP 모드에서 `dmrfy_fp_updt_*`로 **z_out 피드백을 RF에 재기록**(L501–L518, `dmrfy_fp_updt_data = z_out[..][15:0]`) → 비선형 연산의 중간결과를 어레이로 되먹임(LayerNorm/Softmax의 reduction 누적 추정).
- **Const Reg Tree**(L112–L127): `const_fp16_reg`를 8 TAPU × 16열로 파이프라인 브로드캐스트 → FP 모드에서 상수 피연산자(예: GELU 계수, eps) 공급.
- **TAPU cascade**(L444–L520): INT8 모드는 `y_in[tapu] = z_out[tapu-1]`로 **세로 누산 cascade**(L489–L492), FP 모드(`mode_sel[1]==1`)는 각 TAPU가 독립적으로 DMRFY+const를 입력(L489–L491) → **병렬 FP 처리**. 이 분기가 systolic(INT8) vs SIMD(FP) 데이터플로우 전환의 코어.
- **DMB (Dual Mode Buffer, small FIFO)**(L632–L687): TAPU별 z_out을 32b×16=512b로 FIFO 버퍼링(L656–L673), INT8/FP 공용 출력 경로.
- **Quantization**(`dm_quant`, L749–L770): 후술.
- **Transpose**(`transpose_int`, L777–L787): `store_mt_en`일 때 출력 전치(어텐션 K^T 등 추정).
- **S2MM FIFO**(L798–L817): 256b AXI-Stream으로 결과를 메모리 커널로 송출.

### 3.6 Dual-Mode 양자화 유닛 (`dm_quant.sv`) ★

`dm_quant`(L4–L519)는 **출력 단에서 4가지 정밀도 변환**을 수행(L19 주석): `00: int8→int8`, `01: int8→fp16`, `10: fp16→int8`, `11: fp16→fp16`. 이것이 레이어 경계에서 INT↔FP를 오가는 접착제다.

- **공유부**(L32–L77): 스케일 가수 `sf_man` 곱(`use_dsp`, L51–L52, L60–L63), 지수 `sf_exp`만큼 `bs_rsf` 우측시프트(L66–L75).
- **int8→int8**(L80–L154): 시프트 후 하위 8b 절단(truncate)으로 재양자화, 32개 INT8을 256b로 결합.
- **int8→fp16**(L157–L403): 48b 고정소수 곱 결과를 부호크기화(L171–L188) → `lzc_48b`로 leading-zero 측정(L197–L204) → 지수=`shift+sf_exp`, 가수=정규화 결과로 bfloat16 생성(L218–L227). FIFO로 512b→256b 2분할 송출(L351–L398). **선형(INT8) 출력 → 비선형(bfloat16) 입력 변환의 하드웨어 경로.**
- **fp16→int8**(L405–L465): bfloat16 결과를 스케일·절단으로 INT8화. **비선형(bfloat16) 출력 → 다음 선형(INT8) 입력 변환.**
- **fp16→fp16**(L468–L494): 무변환 통과.
- **출력 MUX**(L498–L516): `quant_mode`로 4경로 선택.

### 3.7 커스텀 ISA & 명령어 디코더 (`core_instr_ctrl.sv`, `pccmd_ctrl.sv`) ★

**2계층 명령 구조**: (1) 메모리 커널의 `core_instr_ctrl`이 **64비트 ISA**를 HBM에서 읽어 디코드, (2) 32비트 micro-command(`pccmd`)로 변환해 처리 커널의 `pccmd_ctrl`이 실제 제어신호 생성.

- **ISA 명령 타입**(`core_instr_ctrl.sv` L55–L60): `LAYER_CONFIG(000)`, `PCLOADBLK(001)`(INT8 블록 로드), `PCLOADFPV(010)`(FP 벡터 로드), `PCEXECBLK(011)`(블록 실행), `PCEXECFPV(100)`(FP 벡터 실행), `PCSTORE(110)`(저장). 명령어 비트 `[63]`=valid, `[62:60]`=type(L63–L65).
- **디코드**(L113–L176): 각 타입별로 mm2s/s2mm 주소·BTT(bytes-to-transfer)·depth·tile_sel·실행모드·스케일링 팩터(`sf_exp/sf_man`)·FP op_sel·const reg를 추출. 예: `PCEXECBLK`은 `pc_exec_mode_sel=tdata[1:0]`, `psu_acc/psu_clr`(L151–L157); `PCEXECFPV`은 `pc_fp0/1_exec_vec`, `pc_fp_updt_*`, `pc_fp_op_sel`(L158–L166).
- **ILP/해저드 제어**(L199–L264): load/exec/store busy 플래그 + 타일 충돌 검사(`ldx_ex_conflict`, `ex_st_conflict` 등, L254–L263)로 `s_axis_instr_tready` 게이팅 → load·exec·store **다단 파이프라이닝(타일 더블버퍼)**으로 데이터무브와 연산을 중첩.
- **데이터무버 cmd 생성**(L267–L373): AXI DataMover용 88비트 cmd(BTT/SADDR/EOF) 생성.
- **`pccmd_ctrl.sv`**(L3–L210): 32b micro-cmd를 받아 `proc_core`가 쓰는 모든 제어신호로 풀어냄. const fp16 reg 8뱅크(L61–L79), 스케일 팩터, FP exec/load vector 주소 등. feedback(`pcfbk`)로 load/exec/store done 신호를 메모리 커널에 반환(L201–L202).

### 3.8 호스트 SW (`host_flow.cpp`, host.cpp)

XRT 기반. `tata_node_op`(L3–L55): 각 코어 IP에 instr 길이·instr base addr·yizo base addr·xi base addr를 레지스터로 기록(L16–L29) 후 `IP_START`(L31), `ap_idle` 폴링으로 완료 대기(L35–L44). `instr_load`(L110–L129): 컴파일러가 생성한 instr **바이너리 파일을 uint64 배열로 읽어 BO(buffer object)에 매핑** → ISA가 64비트임과 일치. `tata_node_status`(L57–L108): latency_cycles 레지스터로 ms 변환, throughput = `bfp_mult_num*tapu_num*...`(L103 주석)으로 다중코어(`num_cores`) 처리량 산정. **즉 한 디바이스에 여러 TATAA 코어를 인스턴스화**하는 멀티코어 배치(확인됨).

### 3.9 양자화 알고리즘 (Python)

두 갈래가 존재한다.

**(a) `hlbfp_bfloat16_bs16/` — 알고리즘 레벨(BFP bs16 + bfloat16)**, 컴파일러(`compilation/parser/vit`)와 다수 파일을 공유:
- `hlbfp_format.py` `HybridLowBlockFP`(L5–): **Block Floating Point**(공유지수+가수, INT8 등가)와 **저정밀 FP(bfloat)**를 한 클래스로 표현. `block_size`로 batch/channel/vector/tensor/element-wise 블록 단위 선택(L9–L13). `exp_bits/man_bits=-1`이면 fp32 원본(L52–L53). 값범위·가능값 생성 유틸 제공(L76–L114).
- `para_config.py`: **`BLOCK_DIM=4`, `BLOCK_SIZE=16`** → "bs16"의 의미 = 4×4 블록 BFP(확인됨).
- `quant_function.py` `BFPQuantFunction`(L17–) + `bfp_TD1~TD4`(텐서 차원별): `torch.frexp`로 가수·지수 분리 → 블록 내 **max 지수를 공유지수**로 선택(L121–L122) → `interval=2^(exp-man_bits)`로 라운딩(L127). 4D(conv 임베딩)/3D(어텐션)/2D(linear)/1D(bias) 별 블록화·언폴드 처리(L78–L279). **이것이 선형 연산용 INT8/BFP 양자화의 본체.**
- `quant_module.py`: 레이어 래퍼. `HMQLinear`/`HMQConv2d`는 가중치·bias·출력을 BFP 양자화(L24–L194). 비선형은 **bfloat16 근사**: `HMQSoftmax`는 `2^floor(x/ln2)`로 exp 근사 후 `isqrt`로 정규화(L297–L318); `HMQLayerNorm`은 mean/var 계산 후 `isqrt`로 1/sqrt(var)(L371–L405); `HMQGeLU`는 `x*sigmoid(1.702x)` bfloat16(L320–L369); `isqrt(x)`는 **`0x5f37` 매직넘버 + 1회 뉴턴 반복**(L218–L233) — 하드웨어 `pe_stg_0` `27'h0005f37`와 일치. `pade_tanh_tensor`(L236–)는 GELU용 tanh Padé 근사 대안.
- `format_config.py` `FormatConfig`(L86–): DeiT 12블록을 레이어별 dict로 펼침. **레이어별 혼합정밀 지정**(L121–L194): `embed/qkv/proj/fc1/fc2/head`는 [act(lfp), w(bfp), bias] 3원소 → 선형=BFP; `mulqk/sftm/mulsv`는 act만 → 비선형=lfp/bfloat16. `init_lfp_exp=4, init_bfp_bit=7, shared_exp_bits=4`가 기본(L111–L127). mixed-precision search용 list↔dict 변환 지원(L199–L294).

**(b) `hlibf_bfloat16_int8/` — HW-faithful(int8 + bfloat16)**, 모델별(`hlibf_vit/bert/swin/gpt2`):
- `config.py`의 `BitType`, `build_observer.py`/`build_quantizer.py`로 **PTQ식 INT8 양자화 관측자·양자화기** 구성(파일 존재 확인; 세부 미정독). `quant_module.py`(vit 기준 L1–)에는 비선형의 **CORDIC exp**(주석처리, L10–L50), **Padé tanh**(L53–), `isqrt` 등 bfloat16 근사가 하드웨어 동작과 비트단위로 맞춰 구현됨(추정: 검증용 골든모델). 즉 (a)는 정확도 탐색·알고리즘, (b)는 하드웨어 비트정합 검증/배포용으로 역할 분리(추정).

### 3.10 컴파일러 (Python, `compilation/parser/`)

- `model_parser.py`(L1–L143): PyTorch 모델을 **노드(레이어)–오퍼레이션** 계층의 `ModelSpec`으로 파싱. **연산 분류**(L8–L11): `bfp_ops=['linear','matmul']`(INT8/BFP 경로), `fp_ops=['mul','sub','add','div']`, `nonlinear_layer_list=['norm','sftm','gelu']`(bfloat16 경로). `parse_layer`/`parse_ops`로 각 op의 입력/출력 shape를 전치·정규화해 기록(L32–L139), `padShape(n, dim=32)`로 32(=systolic 32행) 정렬 패딩(L13–L17).
- `layer_parser.py`, `node_param.py`, `operation_param.py`, `model_spec.py`, `base_param.py`: 파라미터 직렬화 프레임워크(저장 키 관리).
- `vit/hlbfp_vit.py`(L30–L213): `VisionTransformer`가 forward 중 `ModelParser mp`를 받아 각 연산을 파싱(예: `blk(x, mp, i, ...)`, L130). DeiT-tiny/small/base 팩토리(L150–L213, timm 가중치 URL). → 모델 실행 그래프에서 **레이어→op→(BFP/FP 분류)→TATAA 명령어** 매핑을 추출하는 구조. 최종 산출물은 호스트가 읽는 **64비트 ISA 바이너리**(host `instr_load`와 정합; 생성 코드 세부는 `run_parser.py`/`hlbfp_parser_run.py`에 위임, 본 분석에서 미정독 — 확인 가능하나 시간상 생략).

---

## 4. 데이터플로우 / 실행 흐름

### 4.1 전체 추론 파이프라인
```
[Python] 모델 → quantization(혼합정밀 BFP/bfloat16) + compilation(파싱→64b ISA 바이너리)
        ↓ (가중치·명령어·입력을 HBM에 배치)
[mem_kernel] instr_loader(HBM→ISA) → core_instr_ctrl(디코드/ILP) ─┬→ data_loader(AXI DataMover: xi/yi MM2S, zo S2MM)
                                                                 └→ pccmd(32b) ─AXIS→ proc_kernel
[proc_kernel] pccmd_ctrl → proc_core
        DMRFX/DMRFY(더블버퍼 RF) → 8×TAPU(systolic INT8 / SIMD bfloat16) → DMB → dm_quant(4모드) → transpose → S2MM → HBM
```

### 4.2 INT8 ↔ bfloat16 전환 메커니즘 (3중 전환)
1. **연산 유닛 내부 전환**: `mode_sel[1:0]`로 PE 4단을 INT8 MAC / bfloat16 [추출·지수·가수·정규화] 파이프라인으로 변형(§3.3). DSP·라우팅 공유.
2. **데이터플로우 전환**: INT8은 TAPU 세로 cascade(systolic 누산, proc_core L489–L492), bfloat16은 TAPU 독립 병렬(SIMD). `mode_sel[1]`이 분기(L477·L489).
3. **레이어 경계 정밀도 전환**: `dm_quant`의 4모드(int8↔fp16)로 선형(INT8)→비선형(bfloat16)→선형(INT8) 사이 변환(§3.6). isqrt 매직넘버가 HW/SW 양쪽에 동일(`0x5f37`).

### 4.3 양자화 방식
- **선형(GEMM)**: Block Floating Point. 블록(기본 4×4=16개 원소) 단위 공유지수 + 7비트 가수(`init_bfp_bit=7`) → 사실상 INT8 정수 곱 + 블록 지수 정렬. shared exp 4비트.
- **비선형/어텐션**: bfloat16(exp 8, man 7). Softmax exp는 `2^floor(x/ln2)`, 1/sqrt는 fast inverse sqrt(`0x5f37`+뉴턴 1회), GELU는 `x·sigmoid(1.702x)`, LayerNorm은 mean/var+isqrt — 모두 bfloat16로 systolic FP 모드에서 수행.

### 4.4 dataflow / pipeline / 메모리 계층
- **메모리 계층**: HBM(외부) → AXI4 MM2S/S2MM DataMover → AXIS register slice(256b) → **DMRFX/DMRFY(on-chip BRAM RF, 더블버퍼 타일)** → PE 내부 `psu_acc_reg`(48b 부분합 레지스터) → DMB FIFO → dm_quant → S2MM FIFO → HBM.
- **파이프라인**: `delay_chain` 기반 systolic skew(TAPU별·열별), DSP 내부 다단 레지스터(AREG/BREG/MREG 등), bfloat16 16사이클 레이턴시(`BFLOAT_LAT_CYCLE=16`). load/exec/store는 ILP 컨트롤러로 타일 더블버퍼 중첩.
- **폭**: 데이터 AXIS 256b, ISA 64b, micro-cmd 32b, feedback 8b.

---

## 5. HW/SW 매핑

| 소프트웨어(Python) | 중간표현 | 하드웨어(RTL) |
|---|---|---|
| `FormatConfig` 레이어별 BFP/bfloat16 지정 | model_format dict | `mode_sel`(INT8 vs FP), `sf_exp/sf_man`(dm_quant 스케일) |
| `HMQLinear/Conv2d` (BFP 양자화) | `bfp_ops=['linear','matmul']` | TAPU INT8 systolic(mode 00), DMRFX/Y |
| `HMQSoftmax/LayerNorm/GELU` (bfloat16) | `nonlinear_layer_list` | TAPU FP(mode 10/11/01), const reg, fp_updt 피드백 |
| `isqrt()` `0x5f37`+뉴턴 | — | `pe_stg_0` `dsp_d_in=27'h0005f37`, mode 01 |
| `ModelParser.parse_ops` shape/패딩(32정렬) | ModelSpec 노드/op | 64b ISA(PCLOADBLK/EXECBLK/EXECFPV/STORE) |
| 컴파일러 출력 instr 바이너리 | `.bin`(uint64) | `instr_loader`→`core_instr_ctrl` 디코드 |
| host `xrt::ip` 레지스터 설정 | instr/xi/yizo base addr | `ps_ctrl`(AXI-Lite), HBM, multi-core |

**물리 매핑**: Vitis 2023.2가 `tata_int8os_proc`(연산)·`tata_int8os_mem`(메모리 IO) 두 RTL을 각각 `.xo`로 패키징(timing 분리 목적, vitis_kernel/README L3) → linker로 `.xclbin` 생성(리포에 산출물 없음) → Alveo U280 HBM에 배치, 호스트 XRT가 멀티코어 구동.

---

## 6. 빌드·실행 방법 (README + 코드 근거)

- **양자화**(quantization): PyTorch 환경에서 `hlbfp_quant_run.py`(bs16) 또는 모델별 `*_run.py`(hlibf) 실행 → 혼합정밀 정확도 평가 + 포맷 확정. (구체 절차는 README가 제목만이라 **문서로는 확인 불가**; 코드 진입점은 `*_run.py`로 추정.)
- **컴파일**(compilation): `parser/*/run_parser.py` 또는 `*_run.py`로 모델 파싱 → 64비트 ISA **instr 바이너리** 생성. hardware/README L11: "host 실행 전 모델 컴파일로 instr 바이너리 필요".
- **하드웨어 커널**(vitis_kernel/README L7–L18): 폴더별(`tata_int8os_mem`, `tata_int8os_proc`)로 ① Vivado 프로젝트 생성 후 `add_files.tcl`·`add_ips.tcl` 실행 → ② RTL 커널 패키지(중간 프로젝트 창) → ③ `config_kernel.tcl`로 AXI 인터페이스 설정 → ④ `.xo` 패키징. **Vitis 2023.2 강제**(다수 Xilinx IP 사용).
- **호스트**(host): instr 바이너리 + `.xclbin`을 XRT로 로드, base address 레지스터 설정 후 `IP_START`. (정확한 빌드 커맨드라인은 **확인 불가** — Makefile/CMake가 리포에 보이지 않음; `libs.h` 등 헤더만 확인.)

---

## 7. 의존성

- **하드웨어**: Vitis/Vivado **2023.2**(필수), Xilinx XRT, DSP48E2 프리미티브(UltraScale+), Xilinx IP(`axis_register_slice`, `axis_infrastructure`, AXI DataMover), Alveo U280 플랫폼(HBM).
- **소프트웨어(Python)**: PyTorch(`torch`, `torch.nn`, `torch.autograd.Function`, bfloat16 지원 필수), NumPy, timm 계열 사전학습 가중치(DeiT URL). GPU(`cuda`, `torch.cuda.power_draw` 호출 — bfloat16 연산·전력측정). 컴파일러는 추가 외부 의존성 거의 없음(자체 파서 프레임워크).
- **호스트**: C++17(`std::vector`, `<iomanip>`), XRT 라이브러리(`xrt::ip`).
- (정확한 버전 pin은 quantization/README가 비어 있어 **확인 불가** — 루트 README L23 "required python environment in ./quantization"이라 했으나 실제 환경파일은 발견되지 않음.)

---

## 8. 강점 / 한계 / 리스크

**강점**
- DSP·라우팅·RF를 INT8/bfloat16가 **완전 공유** → 정수 GEMM 효율 + FP 비선형 정확도를 추가 자원 없이 동시 달성(자원 효율 핵심).
- 비선형(Softmax/LayerNorm/GELU/isqrt)을 별도 전용회로 없이 **같은 systolic 어레이의 FP 모드**로 처리 → 어텐션 전체를 온칩에서 종단(end-to-end) 처리.
- HW/SW 비트정합(isqrt `0x5f37` 일치, 골든모델 `hlibf`)로 검증 가능성 높음.
- 명확한 2계층 커스텀 ISA + ILP 더블버퍼링으로 프로그래머블·다중모델(ViT/BERT/Swin/GPT2 파서 존재).
- mem/proc 커널 분리로 timing closure 유리(README 명시).

**한계**
- **HLS가 아닌 hand-written RTL** + DSP48E2 직접 인스턴스 → 이식성 낮음(UltraScale+ 종속), 학습곡선 가파름. 작업 지시의 "HLS 커널" 기대와 불일치.
- `quantization/README`·`compilation/README`가 **사실상 비어 있음** → 빌드/실행 절차·환경 재현이 코드 리딩에 의존(문서화 부족).
- bfloat16 비선형은 **근사**(2^floor exp, fast-isqrt, Padé tanh) → 모델/정밀도에 따라 정확도 열화 가능(softmax exp 정밀도 특히).
- 컴파일러가 DeiT/ViT 중심(`format_config`은 `model_category=="deit"`만 구현, 그 외 `NotImplementedError`). 일반 Transformer 자동화 한계.
- `pe_stg_0.sv` L3 `@TODO: discuss the overflow issue in bfp8` 등 미해결 TODO 다수 → BFP8 오버플로 처리 미완(리스크).

**리스크**
- Vitis 2023.2 외 버전에서 IP 호환성 깨질 수 있음(README가 "strictly follow"라 경고).
- 산출물(`.xo/.xclbin`)·정확한 instr 생성 코드 세부·환경 파일 부재 → **그대로 재현 빌드는 추가 작업 필요**.

---

## 9. 우리 프로젝트(HG-PIPE 계열 고처리량 ViT/Transformer FPGA 가속 + XR 시선추적) 관점 시사점

1. **FP/INT 겸용 MAC(transformable arithmetic)의 직접 채용 가치 — 가장 핵심 재사용 포인트**
   HG-PIPE 류는 INT/저정밀 파이프라인에 강하지만, 어텐션의 **Softmax·LayerNorm·GELU를 INT로 근사하면 ViT 정확도가 흔들리는** 문제가 있다. TATAA의 "한 DSP 열을 mode_sel로 INT8 MAC ↔ bfloat16 FPU로 변형" 기법은, **HG-PIPE의 GEMM 파이프라인은 INT로 유지하면서 비선형 단만 동일 DSP를 FP로 재사용**하도록 이식할 수 있다. 특히 `pe_stg_0~3`의 [지수추출→정렬→가수곱/시프트→정규화] 4단 분해는 DSP48E2를 쓰는 한 거의 그대로 포팅 가능(우리도 동일 디바이스 계열이면).

2. **isqrt/exp 비선형의 systolic 내장**: `0x5f37` fast-isqrt + `2^floor(x/ln2)` exp를 **별도 LUT/SFU 없이 PE에서** 처리하는 방식은 XR용 경량 ViT(시선추적 백본)에서 자원 압박이 큰 LayerNorm/Softmax를 저비용으로 처리하는 데 직접 유효. 단, 시선추적은 정밀도 요구가 상대적으로 낮아 **bfloat16 대신 더 좁은 FP(예: fp16/bf12)로 가수폭을 줄여** 자원·전력을 더 아끼는 변형 검토 권장(추정 개선안).

3. **양자화–컴파일–HW 통합 플로우 재사용**: `FormatConfig`(레이어별 BFP/bfloat16 지정) → `ModelParser`(op 분류 `bfp_ops`/`nonlinear_layer_list`, 32정렬 패딩) → 64b ISA 바이너리 → `core_instr_ctrl` 디코드의 **end-to-end 자동화 골격**은 우리 컴파일 스택의 레퍼런스로 직접 차용 가능. 특히 "선형=정수경로 / 비선형=FP경로"를 컴파일 타임에 분류해 명령 스트림으로 떨구는 패턴은 XR 모델별 빠른 리타게팅에 유리.

4. **mem/proc 커널 분리 + ILP 더블버퍼 컨트롤러**: 고처리량 목표 시 timing closure와 load/exec/store 중첩이 관건인데, TATAA의 `core_instr_ctrl` 해저드/타일충돌 로직(L254–L263)은 우리 파이프라인의 데이터무브-연산 중첩 설계에 바로 참고 가능.

5. **멀티코어 배치 패턴**: host `num_cores` 루프(여러 TATAA 코어를 한 U280에 인스턴스화)는 **XR의 저지연 + 멀티스트림(좌/우안, 멀티프레임)** 요구에 맞춰 코어 복제로 처리량을 스케일하는 전략의 선례.

6. **주의점(우리 적용 시 리스크)**: TATAA는 hand-written RTL이라 HG-PIPE가 HLS 중심이면 **두 패러다임 접합 비용**이 큼. 또한 컴파일러가 DeiT 한정 → XR 백본(예: 경량 ViT/Hybrid CNN-ViT)에 맞춰 파서 확장 필요. 비선형 근사 정확도는 시선추적 회귀(좌표) 출력에 대해 별도 검증 필수.

---

## 10. 근거 / 한계 표기

- **직접 확인(라인 근거 있음)**: 논문/디바이스(루트 README, hardware README), PE 4단 INT8↔bfloat16 변형(`pe_stg_0~3.sv`, `tapu.sv`, `pe_sys.sv`), proc_core 통합/8 TAPU/FP 피드백(`proc_core.sv`), dual-mode 양자화 4모드(`dm_quant.sv`), 64b ISA·디코더·ILP(`core_instr_ctrl.sv`, `pccmd_ctrl.sv`), mem 커널 AXI 구조(`mem_kernel.v`, `proc_kernel.v`), 양자화 알고리즘/포맷(`hlbfp_format.py`, `quant_function.py`, `quant_module.py`, `format_config.py`, `para_config.py`), 컴파일 op 분류(`model_parser.py`), 호스트 XRT 흐름(`host_flow.cpp`), BLOCK_DIM=4/BLOCK_SIZE=16, isqrt `0x5f37` HW/SW 일치.
- **추정(코드 정황 기반, 단정 회피)**: systolic(INT8) vs SIMD(FP) 데이터플로우 해석, DSP 2-way INT8 packing, FP 피드백의 LayerNorm/Softmax reduction 용도, `hlibf`가 비트정합 골든모델이라는 역할 분리, 16사이클 FP 레이턴시 해석, bfloat16→더 좁은 FP 변형 제안.
- **확인 불가 / 미정독**: quantization·compilation README 본문(제목만 존재 → 빌드 절차 문서 부재), 정확한 Python 환경 버전 pin(환경파일 미발견), instr 바이너리 생성 코드 세부(`run_parser.py`/`hlbfp_parser_run.py` 미정독 — 존재는 확인), `.xo/.xclbin/.bit` 산출물(리포에 없음), Makefile/CMake 빌드 커맨드, `hlibf_*` 모델별 quant_module 전문(일부만 정독), vitis_kernel/tata_int8os_*/rtl이 hardware/rtl과 100% 동일한지(파일명 일치·내용 동일 추정, 전수 diff 미수행).
- **특이사항**: 작업 지시의 "HLS 커널(.cpp/.h) transformable MAC" 가정과 달리 **연산 커널은 RTL(SystemVerilog)**이며 `.cpp/.h`는 호스트 전용. `.cl` 파일 없음. README 5개 중 2개(quant/compile)는 빈 제목.
