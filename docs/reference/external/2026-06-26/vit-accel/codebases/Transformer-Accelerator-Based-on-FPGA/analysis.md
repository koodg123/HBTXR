---
source_type: codebase
source_name: Transformer-Accelerator-Based-on-FPGA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/Transformer-Accelerator-Based-on-FPGA/analysis.md -->

# Transformer-Accelerator-Based-on-FPGA Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA`
- repo_remote: `origin	https://github.com/Buck008/Transformer-Accelerator-Based-on-FPGA (fetch)`
- category: `PYNQ transformer accelerator project`
- HGTXR relevance: `medium`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `42` |
| 주요 언어 | `Verilog:23, C/C++ header:4, SystemVerilog:3, .txt:2, C:2, C++:2, Tcl:1, Markdown:1, .spec:1, .ld:1` |
| LOC 추정 | `Verilog:4205, Tcl:1515, SystemVerilog:719, C:255, C++:210, Python:155, C/C++ header:150, Markdown:10, .txt:2` |

### Directory Map
- `In Progress/` (2 entries)
- `In Progress/sim/` (2 entries)
- `In Progress/src/` (11 entries)
- `pynq/` (1 entries)
- `sdk/` (5 entries)
- `sim/` (2 entries)
- `src/` (12 entries)
- `vitis/` (7 entries)

### Metadata / Config Refs
- `prj.tcl`
- `README.md`
- `sdk/README.txt`
- `vitis/README.txt`

### Core Source Refs
- `prj.tcl`: 1515 lines
- `src/MM_ultra_axi.v`: 468 lines
- `In Progress/src/Gelus_axi.v`: 440 lines
- `In Progress/src/Softmax_top_axi.v`: 432 lines
- `sim/MM_Ultra_tb.sv`: 378 lines
- `src/MM_buffer.v`: 359 lines
- `src/MM_out_buffer.v`: 331 lines
- `src/MM_in_buffer.v`: 268 lines
- `In Progress/src/Softmax.v`: 234 lines
- `In Progress/sim/Softmax_top_tb.sv`: 234 lines

### Hardware-Oriented Source Refs
- `prj.tcl`
- `sdk/matrix.h`
- `sdk/main.c`
- `sdk/matrix.c`
- `sdk/defines.h`
- `vitis/Matrix.cpp`
- `vitis/main.cpp`
- `vitis/Defines.h`
- `vitis/Matrix.h`
- `sim/MM_Ultra_tb.sv`
- `src/PE_line.v`
- `src/MM.v`
- `src/MM_buffer.v`
- `src/PE.v`
- `src/PE_array.v`
- `src/right_shifter.v`
- `src/MM_ultra.v`
- `src/MM_in_buffer.v`
- `src/AdderS.v`
- `src/MM_ultra_axi.v`
- `src/MM_ultra_top.v`
- `src/MM_out_buffer.v`
- `In Progress/sim/Softmax_top_tb.sv`
- `In Progress/sim/gelu_tb.sv`
- `In Progress/src/Softmax_control.v`
- `In Progress/src/EightGelus.v`
- `In Progress/src/gelu.v`
- `In Progress/src/Exp_module.v`
- `In Progress/src/Gelus_axi.v`
- `In Progress/src/Softmax.v`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/README.md`
- # Transformer Accelerator Based on FPGA
- How to reproduce this project:
- 1. In vivado2019.1, create a new project (note that the boardfile is pynq z1, you can download the corresponding boardfile here: https://pynq.readthedocs.io/en/v3.0.0/overlay_design_methodology/board_settings.html ).
- 2. Add all the code to the project
- 3. Run prj.tcl
- 4. Create a wrapper for the block design and set it as the top module.
- 5. Run the generated synthesis and implementation strategies and generate the bitstream.

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/prj.tcl:944:   CONFIG.PCW_TRACE_PIPELINE_WIDTH {8} \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/README.md:3:In the future, I might add some nonlinear hardware acceleration operators (for accelerating ViT, it's a kind of neural network based on Transformer), such as those that compute Softmax, Gelu and LayerNorm functions. I am still working on to improve the accuracy and performance of this part.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top_axi.v:4:	module Softmax_top_axi #`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top_axi.v:413:    Softmax_control u_Softmax_control(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:4:	module Softmax_top #`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:58:	Softmax_top_axi # ( `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:61:	) u_Softmax_top_axi (`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:4:	module Gelus_top #`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:7:		parameter integer num_gelu = 4,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:21:    	input [num_gelu*8-1:0] 	                    s_axis_tdata,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:25:    	input [num_gelu-1:0]                        s_axis_tkeep,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:27:    	output [num_gelu*8-1:0] 	                m_axis_tdata,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:31:    	output [num_gelu-1:0]                         m_axis_tkeep,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:60:	Gelus_axi # ( `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:61:		.num_gelu(num_gelu),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_top.v:64:	) Gelus_axi (`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_axi.v:4:	module Gelus_axi #`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_axi.v:9:        parameter integer num_gelu = 4,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_axi.v:22:    	input [num_gelu*8-1:0] 	                    s_axis_tdata,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_axi.v:26:    	input [num_gelu-1:0]                        s_axis_tkeep,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Gelus_axi.v:28:    	output [num_gelu*8-1:0] 	                m_axis_tdata,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/pynq/MM.py:13:def mat_create(shape, data_type = np.int8):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/sim/MM_Ultra_tb.sv:91:    .data_width(`DATA_WIDTH),                                           //Determines the quantized data bit width`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/README.md:1:# Transformer Accelerator Based on FPGA`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/README.md:2:You can run it on pynq z1 (or any other Zynq device, since the systolic array is parameterized). The repository contains the relevant Verilog code, Vivado configuration and C/Python code for sdk/PYNQ testing. The size of the systolic array can be changed, now it is 16X16.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/README.md:6:1. In vivado2019.1, create a new project (note that the boardfile is pynq z1, you can download the corresponding boardfile here: https://pynq.readthedocs.io/en/v3.0.0/overlay_design_methodology/board_settings.html ).`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:12:		// Parameters of Axi Slave Bus Interface S00_AXI`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:13:		localparam integer C_S00_AXI_DATA_WIDTH	= 32,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:14:		localparam integer C_S00_AXI_ADDR_WIDTH	= 4`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:18:		input 										axis_aclk,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:21:    	input [7:0] 	                            s_axis_tdata,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:22:    	input       								s_axis_tvalid,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:23:    	output      								s_axis_tready,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:24:    	input       								s_axis_tlast,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Transformer-Accelerator-Based-on-FPGA/In Progress/src/Softmax_top.v:26:    	output [7:0] 	                            m_axis_tdata,`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

### 적용 가능 모듈
- Attention / Softmax / token sparsity ablation
- HLS/Vivado/PYNQ integration and resource-flow references

## 6. 실험 옵션으로 변환할 때의 규칙
- 먼저 HGTXR-SW 또는 C++ reference에서 accuracy/bit-exact behavior를 검증한다.
- HLS에 반영할 때는 C3b signoff path를 덮어쓰지 말고 새로운 suffix variant로 추가한다.
- 완료된 `csynth.xml`, routed timing/power, PYNQ smoke JSON 없이는 resource matrix의 완료 variant로 승격하지 않는다.
- HGTXR 논문 범위를 벗어나는 구조 변경은 `paper_scope_review_required`로 둔다.

## 7. 리스크와 비적용 조건
- 동적 MoE routing, softmax-free attention, ternary/LUT-heavy compute는 정확도 또는 resource 정책과 충돌할 수 있다.
- 현재 사용자의 resource 방향은 DSP/URAM 활용 증가와 LUT 과사용 억제이므로 LUT-LLM/LUT-GEMM류는 기본값이 아니라 negative-control이다.
- codebase가 LLM 중심이면 HGTXR eye-tracking 적용은 kernel/system-flow 수준으로 제한한다.

## 8. 다음 분석/구현 액션
- 핵심 source file을 1개 선택해 line-by-line 분석을 수행한다.
- 해당 방법을 HGTXR `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml`의 planned experiment와 연결한다.
- SW accuracy metric과 HW resource metric을 같은 manifest에 기록한다.
