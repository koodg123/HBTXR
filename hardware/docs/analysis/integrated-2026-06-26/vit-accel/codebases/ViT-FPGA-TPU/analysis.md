---
source_type: codebase
source_name: ViT-FPGA-TPU
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/ViT-FPGA-TPU/analysis.md -->

# ViT-FPGA-TPU Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU`
- repo_remote: `origin	https://github.com/gnodipac886/ViT-FPGA-TPU (fetch)`
- category: `ViT FPGA student/project accelerator`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `308` |
| 주요 언어 | `SystemVerilog:47, Verilog:26, .xci:24, C/C++ header:13, C++:12, Tcl:11, Shell:10, .txt:9, .bda:9, .protoinst:9` |
| LOC 추정 | `SystemVerilog:14924, Verilog:6139, C++:2597, .txt:2324, Tcl:1567, C/C++ header:1281, C:386, Python:275, Shell:266, JSON:231` |

### Directory Map
- `code/` (4 entries)
- `code/.vscode/` (1 entries)
- `code/code_cpu/` (15 entries)
- `code/code_fpga/` (18 entries)
- `code/fpga/` (2 entries)
- `images/` (3 entries)
- `milestone/` (4 entries)
- `milestone/2/` (1 entries)
- `milestone/3/` (1 entries)
- `milestone/4/` (1 entries)
- `milestone/5/` (2 entries)
- `v2.0/` (3 entries)
- `v2.0/hw/` (1 entries)
- `v2.0/sw/` (4 entries)

### Metadata / Config Refs
- `README.md`
- `code/.vscode/settings.json`
- `code/code_cpu/.vscode/settings.json`
- `code/fpga/test_accel/test_accel.sim/sim_1/behav/xsim/top.tcl`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/xgui/pci_mig_accelerator_v1_0.tcl`
- `code/code_fpga/.vscode/settings.json`
- `code/code_fpga/accel_driver/README.md`
- `code/code_fpga/fpga/pci_mig_accelerator_1.0_32/xgui/pci_mig_accelerator_v1_0.tcl`
- `code/code_fpga/fpga/pci_mig_accelerator_1.0_32/bd/bd.tcl`
- `code/code_fpga/fpga/pci_mig_top/pci_mig.gen/sources_1/bd/pci_mig/pci_mig_ooc.xdc`
- `code/code_fpga/fpga/pci_mig_top/pci_mig.srcs/constrs_1/new/const.xdc`
- `code/code_fpga/fpga/pci_mig_top/pci_mig.sim/sim_1/behav/xsim/accelerator_tb.tcl`
- `code/code_fpga/fpga/pci_mig_accelerator_1.0/xgui/pci_mig_accelerator_v1_0.tcl`
- `code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/bd.tcl`
- `code/code_fpga/fpga/pci_mig_32/pci_mig.gen/sources_1/bd/pci_mig/pci_mig_ooc.xdc`
- `code/code_fpga/fpga/pci_mig_32/pci_mig.srcs/constrs_1/new/const.xdc`
- `code/code_fpga/fpga/pci_mig_32/pci_mig.sim/sim_1/behav/xsim/accelerator_tb.tcl`
- `v2.0/README.md`
- `v2.0/hw/tpu_16/pci_mig_16_v2/pci_mig_16_v2.gen/sources_1/bd/design_1/design_1_ooc.xdc`
- `v2.0/hw/tpu_16/pci_mig_16_v2/pci_mig_16_v2.gen/sources_1/bd/design_1/hw_handoff/design_1_bd.tcl`
- `v2.0/hw/tpu_16/pci_mig_16_v2/pci_mig_16_v2.srcs/constrs_1/new/const.xdc`
- `v2.0/hw/tpu_16/ip_repo/fp_sys_array/xgui/systolic_array_wrapper_v1_0.tcl`
- `v2.0/hw/tpu_16/ip_repo/fp_sys_array/xgui/fp_systolic_array_v1_0.tcl`
- `v2.0/hw/tpu_16/ip_repo/fp_sys_array/src/pipeline_mac_ooc.xdc`
- `v2.0/sw/README.md`

### Core Source Refs
- `code/code_cpu/output.txt`: 2197 lines
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/pci_mig_accelerator_v1_0_M00_AXI.sv`: 922 lines
- `code/code_fpga/fpga/pci_mig_accelerator_1.0_32/src/pci_mig_accelerator_v1_0_M00_AXI.sv`: 922 lines
- `code/code_fpga/fpga/pci_mig_accelerator_1.0/src/pci_mig_accelerator_v1_0_M00_AXI.sv`: 922 lines
- `code/fpga/pci_mig_accelerator_1.0_16_auto/hdl/pci_mig_accelerator_v1_0_M00_AXI.sv`: 907 lines
- `code/code_fpga/fpga/pci_mig_accelerator_1.0_32/hdl/pci_mig_accelerator_v1_0_M00_AXI.sv`: 907 lines
- `code/code_fpga/msa.cpp`: 819 lines
- `code/code_fpga/fpga/pci_mig_top/pci_mig.gen/sources_1/bd/pci_mig/synth/pci_mig.v`: 775 lines
- `code/code_fpga/fpga/pci_mig_top/pci_mig.gen/sources_1/bd/pci_mig/sim/pci_mig.v`: 775 lines
- `code/fpga/pci_mig_accelerator_1.0_16_auto/hdl/pci_mig_accelerator_v1_0_S00_AXI.sv`: 644 lines

### Hardware-Oriented Source Refs
- `code/code_cpu/vit_block.cpp`
- `code/code_cpu/vit.h`
- `code/code_cpu/msa.cpp`
- `code/code_cpu/vit.cpp`
- `code/code_cpu/main.cpp`
- `code/code_cpu/vit_block.h`
- `code/code_cpu/msa.h`
- `code/code_cpu/parallel_mm.cpp`
- `code/fpga/test_accel/test_accel.srcs/sources_1/imports/src/accelerator_ctl.sv`
- `code/fpga/test_accel/test_accel.srcs/sources_1/imports/src/accelerator_types.sv`
- `code/fpga/test_accel/test_accel.srcs/sources_1/imports/src/spad_arbiter.sv`
- `code/fpga/test_accel/test_accel.srcs/sim_1/imports/src/top.sv`
- `code/fpga/test_accel/test_accel.srcs/sim_1/imports/src/accelerator_ctl_tb.sv`
- `code/fpga/test_accel/test_accel.sim/sim_1/behav/xsim/top.tcl`
- `code/fpga/test_accel/test_accel.sim/sim_1/behav/xsim/glbl.v`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/hdl/pci_mig_accelerator_v1_0_M00_AXI.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/hdl/pci_mig_accelerator_v1_0_S00_AXI.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/hdl/pci_mig_accelerator_v1_0.v`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/xgui/pci_mig_accelerator_v1_0.tcl`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/pci_mig_accelerator_v1_0_M00_AXI.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/PE.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/systolic_array.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/pci_mig_accelerator_v1_0_S00_AXI.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/accelerator_ctl.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/top.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/accelerator_types.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/pci_mig_accelerator_sv.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/accelerator_ctl_tb.sv`
- `code/fpga/pci_mig_accelerator_1.0_16_auto/src/spad_arbiter.sv`
- `code/code_fpga/vit_block.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/README.md`
- # ViT-FPGA-TPU
- FPGA based Vision Transformer accelerator (Harvard CS205)
- # To Run ViT in libTorch (PyTorch C++ API)
- #### Go into the `code/code_cpu` directory
- - `cd code/code_cpu`
- - Everything below is done in the `code/code_cpu` directory!!!
- #### First we download the LibTorch library if it's not in the folder yet:
- - `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`
- - `unzip libtorch-shared-with-deps-latest.zip`
- #### Then build the project
- - `mkdir build`
- - `cd build`
- - `cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch_folder/that/you/just/downloaded ..`
- - `cmake --build . --config Release`
- - `cd ..` (we exit the `build` directory)
- #### Then download the MNIST dataset if it's not in the folder yet:
- - `python download_mnist.py`
- - `mkdir mnist`
- - Put all the downloaded `.ubyte` files into `./mnist` folder
- #### Run the model :)

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:356:        <spirit:name>src/pipeline_mac_ooc.xdc</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:359:        <spirit:userFileType>SCOPED_TO_REF_pipeline_mac</spirit:userFileType>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:367:        <spirit:name>src/pipeline_mac_floating_point_1_0/pipeline_mac_floating_point_1_0.xci</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:370:        <spirit:userFileType>CELL_NAME_genblk1[0].genblk1[0].pe/_mac/pipeline_mac_i/floating_point_1</spirit:userFileType>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:373:        <spirit:name>src/pipeline_mac_floating_point_0_0/pipeline_mac_floating_point_0_0.xci</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:376:        <spirit:userFileType>CELL_NAME_genblk1[0].genblk1[0].pe/_mac/pipeline_mac_i/floating_point_0</spirit:userFileType>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:379:        <spirit:name>src/pipeline_mac.v</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:388:        <spirit:name>src/pipeline_mac_wrapper.v</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:417:        <spirit:name>src/pipeline_mac_floating_point_1_0/pipeline_mac_floating_point_1_0.xci</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:420:        <spirit:userFileType>CELL_NAME_genblk1[0].genblk1[0].pe/_mac/pipeline_mac_i/floating_point_1</spirit:userFileType>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:423:        <spirit:name>src/pipeline_mac_floating_point_0_0/pipeline_mac_floating_point_0_0.xci</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:426:        <spirit:userFileType>CELL_NAME_genblk1[0].genblk1[0].pe/_mac/pipeline_mac_i/floating_point_0</spirit:userFileType>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/hw/tpu_16/ip_repo/fp_sys_array/component.xml:429:        <spirit:name>src/pipeline_mac_wrapper.v</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/sim/pipeline_mac.protoinst:4:		"pipeline_mac": {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/synth/pipeline_mac.v:6://Command     : generate_target pipeline_mac.bd`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/synth/pipeline_mac.v:7://Design      : pipeline_mac`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/synth/pipeline_mac.v:12:(* CORE_GENERATION_INFO = "pipeline_mac,IP_Integrator,{x_ipVendor=xilinx.com,x_ipLibrary=BlockDiagram,x_ipName=pipeline_mac,x_ipVersion=1.00.a,x_ipLanguage=VERILOG,numBlks=2,numReposBlks=2,numNonXlnxBlks=0,numHierBlks=0,maxHierDepth=0,numSysgenBlks=0,numHlsBlks=0,numHdlrefBlks=0,numPkgbdBlks=0,bdsource=USER,synth_mode=OOC_per_IP}" *) (* HW_HANDOFF = "pipeline_mac.hwdef" *) `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/synth/pipeline_mac.v:13:module pipeline_mac`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/synth/pipeline_mac.v:26:  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS_RESULT_0 TDATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXIS_RESULT_0, CLK_DOMAIN pipeline_mac_aclk_0, FREQ_HZ 100000000, HAS_TKEEP 0, HAS_TLAST 1, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA xilinx.com:interface:datatypes:1.0 {TDATA {datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value data} bitwidth {attribs {resolve_type generated dependency width format long minimum {} maximum {}} value 16} bitoffset {attribs {resolve_type immediate dependency {} format long minimum {} maximum {}} value 0} real {float {sigwidth {attribs {resolve_type generated dependency fractwidth format long minimum {} maximum {}} value 11}}}}} TDATA_WIDTH 16 TUSER {datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type automatic dependency {} format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type immediate dependency {} format long minimum {} maximum {}} value 0} struct {field_underflow {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value underflow} enabled {attribs {resolve_type generated dependency underflow_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency underflow_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type immediate dependency {} format long minimum {} maximum {}} value 0}}} field_overflow {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value overflow} enabled {attribs {resolve_type generated dependency overflow_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency overflow_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency overflow_bitoffset format long minimum {} maximum {}} value 0}}} field_invalid_op {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value invalid_op} enabled {attribs {resolve_type generated dependency invalid_op_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency invalid_op_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency invalid_op_bitoffset format long minimum {} maximum {}} value 0}}} field_div_by_zero {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value div_by_zero} enabled {attribs {resolve_type generated dependency div_by_zero_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency div_by_zero_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency div_by_zero_bitoffset format long minimum {} maximum {}} value 0}}} field_accum_input_overflow {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value accum_input_overflow} enabled {attribs {resolve_type generated dependency accum_input_overflow_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency accum_input_overflow_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency accum_input_overflow_bitoffset format long minimum {} maximum {}} value 0}}} field_accum_overflow {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value accum_overflow} enabled {attribs {resolve_type generated dependency accum_overflow_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency accum_overflow_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency accum_overflow_bitoffset format long minimum {} maximum {}} value 0}}} field_a_tuser {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value a_tuser} enabled {attribs {resolve_type generated dependency a_tuser_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency a_tuser_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency a_tuser_bitoffset format long minimum {} maximum {}} value 0}}} field_b_tuser {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value b_tuser} enabled {attribs {resolve_type generated dependency b_tuser_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency b_tuser_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency b_tuser_bitoffset format long minimum {} maximum {}} value 0}}} field_c_tuser {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value c_tuser} enabled {attribs {resolve_type generated dependency c_tuser_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency c_tuser_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency c_tuser_bitoffset format long minimum {} maximum {}} value 0}}} field_operation_tuser {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value operation_tuser} enabled {attribs {resolve_type generated dependency operation_tuser_enabled format bool minimum {} maximum {}} value false} datatype {name {attribs {resolve_type immediate dependency {} format string minimum {} maximum {}} value {}} bitwidth {attribs {resolve_type generated dependency operation_tuser_bitwidth format long minimum {} maximum {}} value 0} bitoffset {attribs {resolve_type generated dependency operation_tuser_bitoffset format long minimum {} maximum {}} value 0}}}}}} TUSER_WIDTH 0}, PHASE 0.0, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) output [15:0]M_AXIS_RESULT_0_tdata;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga/pci_mig_accelerator_1.0/bd/pipeline_mac/synth/pipeline_mac.v:30:  (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 S_AXIS_A_0 TDATA" *) (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXIS_A_0, CLK_DOMAIN pipeline_mac_aclk_0, FREQ_HZ 100000000, HAS_TKEEP 0, HAS_TLAST 0, HAS_TREADY 1, HAS_TSTRB 0, INSERT_VIP 0, LAYERED_METADATA undef, PHASE 0.0, TDATA_NUM_BYTES 2, TDEST_WIDTH 0, TID_WIDTH 0, TUSER_WIDTH 0" *) input [15:0]S_AXIS_A_0_tdata;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/sw/src/example.py:56:# print(np.matmul(a, b)[:16, :16])`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/sw/src/example.py:99:# print(np.matmul(a, b)[-10:, -10:])`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/vit.cpp:134:    tokens = torch::nn::functional::softmax(tokens, torch::nn::functional::SoftmaxFuncOptions(0));`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/vit_block.h:12:        torch::nn::LayerNorm norm1;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/vit_block.h:13:        torch::nn::LayerNorm norm2;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/vit_block.h:16:        torch::nn::GELU g1 = nullptr;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/parallel_mm.cpp:6:void matmul(float* a, float* b, float* c, int m, int n) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/parallel_mm.cpp:59:// Calculate torch::functional::softmax(A, /*dim=*/1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/parallel_mm.cpp:60:// void softmax(float *A, int m) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/sw/src/main.cpp:91:void matmul_sync(){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/v2.0/sw/src/main.cpp:437:	torch::Tensor tensor_c_golden	= torch::matmul(tensor_a.to(torch::kFloat32), tensor_b.to(torch::kFloat32)).to(torch::kFloat16);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga.h:75:void matmul_sync(){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga.h:261:	torch::Tensor tensor_c_golden	= torch::matmul(tensor_a.to(torch::kFloat32), tensor_b.to(torch::kFloat32)).to(torch::kFloat16);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga.h:310:// 	torch::Tensor tensor_c_golden	= torch::matmul(tensor_a.to(torch::kFloat32), tensor_b.to(torch::kFloat32)).to(torch::kFloat16);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ViT-FPGA-TPU/code/code_fpga/fpga.h:419:// 	// 	tensor_c_golden	= torch::matmul(tensor_a.to(torch::kFloat32), tensor_b.to(torch::kFloat32)).to(torch::kFloat16);`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

### 적용 가능 모듈
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
