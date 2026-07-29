---
source_type: codebase
source_name: llama-fpga
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/llama-fpga/analysis.md -->

# llama-fpga Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga`
- repo_remote: `origin	https://github.com/adamgallas/llama-fpga (fetch)`
- category: `LLM FPGA system with AXI/DMA packaging`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `2205` |
| 주요 언어 | `.xml:875, .xci:865, Scala:148, .bd:57, .mem:36, .bxml:31, Verilog:30, .vhd:25, .vho:12, .veo:12` |
| LOC 추정 | `Verilog:69122, Scala:19229, C:2264, Python:588, Markdown:226, Notebook:207, Tcl:60, Shell:11, .txt:7` |

### Directory Map
- `au250/` (7 entries)
- `au250/au250_vivado/` (1 entries)
- `au250_half/` (7 entries)
- `au250_half/u250.cache/` (1 entries)
- `au250_half/u250.gen/` (1 entries)
- `au250_half/u250.hw/` (2 entries)
- `au250_half/u250.ip_user_files/` (3 entries)
- `au250_half/u250.runs/` (1 entries)
- `au250_half/u250.srcs/` (3 entries)
- `kv260/` (5 entries)
- `kv260/kv260_vivado/` (9 entries)
- `python/` (2 entries)
- `scala/` (3 entries)
- `scala/data/` (3 entries)
- `scala/src/` (1 entries)
- `zcu104_pl/` (3 entries)
- `zcu104_pl/zcu104_pl_vivado/` (8 entries)
- `zcu104_ps_pl/` (5 entries)
- `zcu104_ps_pl/zcu104_vivado/` (8 entries)

### Metadata / Config Refs
- `README.md`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.ip_user_files/README.txt`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/mref/EdgeLLMWrapper/xgui/EdgeLLMWrapper_v1_0.tcl`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.ip_user_files/README.txt`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.srcs/constrs_1/new/pblock_xdc.xdc`
- `kv260/kv260_vivado/kv260_vivado.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `kv260/kv260_vivado/kv260_vivado.srcs/constrs_1/new/debug.xdc`
- `kv260/kv260_vivado/kv260_vivado.ip_user_files/README.txt`
- `au250_half/u250.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `au250_half/u250.srcs/constrs_1/new/constraints.xdc`
- `au250_half/u250.ip_user_files/README.txt`
- `zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.ip_user_files/README.txt`

### Core Source Refs
- `zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v`: 42361 lines
- `kv260/kv260_vivado/kv260_vivado.ip_user_files/ipstatic/simulation/fifo_generator_vlog_beh.v`: 10519 lines
- `kv260/kv260_vivado/kv260_vivado.ip_user_files/ipstatic/hdl/fifo_generator_v13_2_rfs.v`: 7800 lines
- `scala/src/main/scala/util/XilinxFloatIPCollection.scala`: 1029 lines
- `scala/src/main/scala/cfgGen/GenMemCmdSmallBurst.scala`: 876 lines
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/top/ip/top_axi_smc_3/bd_0/hdl/bd_b57a_wrapper.v`: 857 lines
- `scala/src/main/scala/cfgGen/GenMemCmdLenAlign.scala`: 825 lines
- `scala/src/main/scala/top/DataPath.scala`: 733 lines
- `scala/src/main/scala/cfgGen/GenMemCmd.scala`: 722 lines
- `scala/src/main/scala/busdemux/AxiBusDistributor.scala`: 591 lines

### Hardware-Oriented Source Refs
- `zcu104_ps_pl/zcu104_sdk.c`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/zcu104bd/ip/zcu104bd_axi_smc_4_0/bd_0/hdl/bd_5a44_wrapper.v`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/zcu104bd/ip/zcu104bd_axi_smc_0/bd_0/hdl/bd_2bae_wrapper.v`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/zcu104bd/ip/zcu104bd_axi_smc_3_0/bd_0/hdl/bd_9bf5_wrapper.v`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/zcu104bd/ip/zcu104bd_axi_smc_2_0/bd_0/hdl/bd_5ba4_wrapper.v`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.gen/sources_1/bd/zcu104bd/ip/zcu104bd_axi_smc_1_0/bd_0/hdl/bd_5b54_wrapper.v`
- `zcu104_ps_pl/zcu104_vivado/zcu104_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v`
- `au250/host_sdk.c`
- `au250/host_program_x2.c`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/mref/EdgeLLMWrapper/xgui/EdgeLLMWrapper_v1_0.tcl`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/mref/DataPath_xN/xgui/DataPath_xN_v1_0.tcl`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/top/ip/top_smartconnect_0_2/bd_0/hdl/bd_6d47_wrapper.v`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/top/ip/top_smartconnect_3_2/bd_0/hdl/bd_6db7_wrapper.v`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/top/ip/top_smartconnect_1_2/bd_0/hdl/bd_ad16_wrapper.v`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/top/ip/top_smartconnect_2_2/bd_0/hdl/bd_ade6_wrapper.v`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.gen/sources_1/bd/top/ip/top_axi_smc_3/bd_0/hdl/bd_b57a_wrapper.v`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.ip_user_files/ipstatic/hdl/floating_point_v7_1_rfs.v`
- `au250/au250_vivado/au250EdgeLLM/au250EdgeLLM.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v`
- `scala/data/SimpleDualPortRam.v`
- `scala/src/main/scala/rope/SerialRoPENew.scala`
- `scala/src/main/scala/rope/CosSinIndexGen.scala`
- `scala/src/main/scala/rope/InvFreqRom.scala`
- `scala/src/main/scala/rope/CosSinGen.scala`
- `scala/src/main/scala/rope/SerialRoPE32.scala`
- `scala/src/main/scala/rope/SerialRoPE.scala`
- `scala/src/main/scala/rope/RoPERotate.scala`
- `scala/src/main/scala/rope/CosSinGen32.scala`
- `scala/src/main/scala/busdemux/KvCacheCase.scala`
- `scala/src/main/scala/busdemux/AxiBusDistributor.scala`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/README.md`
- ---
- #  llama-fpga: FPGA-based LLM Accelerator
- **llama-fpga** is *(most probably)* the **world’s first open-source project** for building an **FPGA-based Large Language Model (LLM) accelerator**, capable of running **LLaMA2-7B** in **AWQ 4-bit quantized format**.
- This project demonstrates how to deploy modern transformer-based LLMs on embedded and data center FPGAs, offering both **research** and **educational** value in hardware-accelerated AI inference.
- ##  Hardware Requirements
- | Hardware Platform     | Supported?  | Notes                                          |
- | --------------------- | ----------- | ---------------------------------------------- |
- | **Xilinx KV260**      | ✅  | Uses PS-side 4 GB RAM for model weights        |
- | **Xilinx ZCU104**     | ✅  | Two variants: PS/PL weight distribution        |
- | **Xilinx Alveo U250** | ✅  | Uses all 4 DDR4 channels for maximum bandwidth |
- **Additional Requirements:**
- * SD card: **≥ 8 GB**
- * For **ZCU104**, a **4 GB DDR4 (Rank = 1)** SODIMM memory module is required.
- ---
- ##  Repository Structure
- This repository includes **four hardware-specific subprojects**, each corresponding to a distinct FPGA setup:
- | Directory       | Platform         | Description                                                      |
- | --------------- | ---------------- | ---------------------------------------------------------------- |
- | `kv260/`        | KV260            | LLM inference with model weights loaded into PS-side RAM (4 GB). |
- | `zcu104_pl/`    | ZCU104 (PL only) | Model weights fully loaded into PL-side 4 GB DDR4 memory.        |

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/README.md:168:   Adapting the design to other models (e.g., Mistral, LLaMA3, or GPT-NeoX) would require **RTL-level modifications**, including changes to matrix dimensions, memory mapping, and dataflow scheduling.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/ip/AxiDatamover128/AxiDatamover128.xml:5165:      <spirit:displayName>Address Pipeline Depth</spirit:displayName>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/ip/AxiDatamover128/AxiDatamover128.xml:5304:      <spirit:displayName>Address Pipeline Depth</spirit:displayName>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/ip/AxiDatamover512/AxiDatamover512.xml:5127:      <spirit:displayName>Address Pipeline Depth</spirit:displayName>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/ip/AxiDatamover512/AxiDatamover512.xml:5266:      <spirit:displayName>Address Pipeline Depth</spirit:displayName>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xml:3109:        <spirit:name>C_MUX_PIPELINE_STAGES</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xml:3110:        <spirit:value spirit:format="long" spirit:resolve="generated" spirit:id="MODELPARAM_VALUE.C_MUX_PIPELINE_STAGES">0</spirit:value>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xml:3796:      <spirit:name>Pipeline_Stages</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xml:3797:      <spirit:value spirit:resolve="user" spirit:id="PARAM_VALUE.Pipeline_Stages" spirit:choiceRef="choice_list_6e3ded9c" spirit:order="37">0</spirit:value>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xml:3801:            <xilinx:isEnabled xilinx:resolve="dependent" xilinx:id="PARAM_ENABLEMENT.Pipeline_Stages">false</xilinx:isEnabled>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xci:52:        "Pipeline_Stages": [ { "value": "0", "resolve_type": "user", "enabled": false, "usage": "all" } ],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_1/xdma_v4_1_20_blk_mem_64_reg_be.xci:136:        "C_MUX_PIPELINE_STAGES": [ { "value": "0", "resolve_type": "generated", "format": "long", "usage": "all" } ],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xml:22492:        <spirit:name>DISABLE_BRAM_PIPELINE</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xml:22493:        <spirit:value spirit:resolve="generated" spirit:id="MODELPARAM_VALUE.DISABLE_BRAM_PIPELINE">FALSE</spirit:value>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xml:28093:      <spirit:name>disable_bram_pipeline</spirit:name>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xml:28094:      <spirit:value spirit:format="bool" spirit:resolve="user" spirit:id="PARAM_VALUE.disable_bram_pipeline" spirit:order="453">false</spirit:value>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xml:28695:        <xilinx:configElementInfo xilinx:referenceId="PARAM_VALUE.disable_bram_pipeline" xilinx:valueSource="propagated"/>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xci:546:        "disable_bram_pipeline": [ { "value": "false", "value_src": "propagated", "resolve_type": "user", "format": "bool", "usage": "all" } ],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_xdma_0_2/ip_0/design_1_xdma_0_2_pcie4_ip.xci:1066:        "DISABLE_BRAM_PIPELINE": [ { "value": "FALSE", "resolve_type": "generated", "usage": "all" } ],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/au250_half/u250.gen/sources_1/bd/design_1/ip/design_1_ddr4_0_0/bd_0/ip/ip_6/bd_45eb_lmb_bram_I_0.xci:53:        "Pipeline_Stages": [ { "value": "0", "resolve_type": "user", "enabled": false, "usage": "all" } ],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:420:  wire                attn_io_softmaxOut_valid;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:421:  wire                attn_io_softmaxOut_payload_last;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:422:  wire       [15:0]   attn_io_softmaxOut_payload_tdata;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:423:  wire       [5:0]    attn_io_softmaxOut_payload_tuser;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:750:    .io_softmaxOut_valid            (attn_io_softmaxOut_valid                ), //o`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:751:    .io_softmaxOut_payload_last     (attn_io_softmaxOut_payload_last         ), //o`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:752:    .io_softmaxOut_payload_tdata    (attn_io_softmaxOut_payload_tdata[15:0]  ), //o`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:753:    .io_softmaxOut_payload_tuser    (attn_io_softmaxOut_payload_tuser[5:0]   ), //o`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:797:    .io_softmaxOut_valid               (attn_io_softmaxOut_valid                ), //i`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:798:    .io_softmaxOut_payload_last        (attn_io_softmaxOut_payload_last         ), //i`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:799:    .io_softmaxOut_payload_tdata       (attn_io_softmaxOut_payload_tdata[15:0]  ), //i`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:800:    .io_softmaxOut_payload_tuser       (attn_io_softmaxOut_payload_tuser[5:0]   ), //i`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:6979:  input  wire          io_softmaxOut_valid,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:6980:  input  wire          io_softmaxOut_payload_last,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/llama-fpga/zcu104_pl/zcu104_pl_vivado/zcu104_pl_vivado.srcs/sources_1/imports/EdgeLLM/DataPath_xN.v:6981:  input  wire [15:0]   io_softmaxOut_payload_tdata,`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation
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
