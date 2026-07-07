# Integrated Codebase Analysis: References/Hardware

Source directory: `/home/kjm26/project/PRJXR/References/Hardware`

## 1. Corpus Summary

| Metric | Value |
|---|---:|
| Top-level repositories | 37 |
| Primary focus | FPGA/ASIC accelerators, ViT/Transformer RTL/HLS, sparse kernels, quantization, benchmark archives |
| Direct RTL/HLS content | High |
| Direct HGTXR value | Hardware block structure, AXI/HLS templates, attention/MLP datapaths, memory hierarchy |

## 2. Repository Coverage

| Repository | Category | Key hardware files or folders | HGTXR relevance | Priority |
|---|---|---|---|---|
| `A.U.R.A.---FlashAttention-ASIC-Accelerator` | FlashAttention ASIC-style RTL | `verilog/AURA.sv`, `PE.sv`, `*SRAM.sv`, `memory_controller.sv`, `tree_reduce.sv` | SRAM reuse, attention datapath comparison | P1 |
| `AGNA-FCCM2023` | HLS accelerator/scheduler | `hardware/hls/hls_src/*`, `software/agna.py` | HLS scheduling and platform/model spec separation | P0 |
| `DPACS` | Sparse/dataflow archive | mixed hardware/software | Secondary sparse reference | P3 |
| `Diff-DiT` | HLS diffusion/DiT blocks | `src/TOP.cpp`, `AMA_rtl.cpp`, `config.h` | DiT block and HLS RTL hybrid examples | P2 |
| `ESDA` | Event/accelerator design archive | `hardware/gen_prj.py`, `software/int_inference.py` | Event-related design context, less direct | P3 |
| `Edge-MoE` | HLS MoE/ViT accelerator | `include/*.hpp`, `src/attention.cpp`, `src/moe.cpp`, `vitis_hls.tcl` | Search/track mode routing and expert execution | P1 |
| `FPGA_Friendly_SpinQuant` | LLM quantization | scripts/models | Indirect scale/rotation quantization reference | P3 |
| `FlexCNN` | CNN HLS accelerator | HLS and SDx kernels | CNN/local feature front-end reference | P2 |
| `FlexLLM` | LLM FPGA flow | kernels/toolflow | Memory/scheduling reference | P2 |
| `HG-PIPE` | ViT FPGA pipeline | generated HLS/Verilog, case modules | Direct attention/MLP pipeline reference | P0 |
| `HLS-Acceleration-of-LLaMA2` | LLM HLS | HLS kernels | Indirect transformer kernel reference | P2 |
| `HiSpMM`, `HiSpMV`, `HiSparse` | Sparse matrix kernels | sparse RTL/HLS files | Memory/dataflow microbenchmarks | P2 |
| `Kria-YOLOv4-Tiny-FPGA-Accelerator` | CNN accelerator | Kria hardware flow | Peripheral reference only | P3 |
| `LLM_FPGA` | LLM archive | no root README noted | Indirect memory and quantization context | P3 |
| `Lightening-Transformer-AE` | Transformer accelerator | transformer-focused hardware | Attention/encoder comparison | P2 |
| `MSD-FCCM23` | RTL DSP system | `hardware/vivado/zcu102/rtl/dsp_pe.sv`, `dsp_sys.sv` | DSP PE decomposition | P0 |
| `MobileVit-AI-Hardware-Accelerator` | MobileViT RTL/HLS | hardware sources | MobileViT reference | P1 |
| `REMOT-FPGA-22` | HLS/device flow | HLS, driver, bitstream artifacts | Integration flow and board packaging | P1 |
| `TATAA` | Attention RTL/dataflow | `hardware/rtl/pe_stg_3.sv`, `data_loader.sv`, `bsr*.sv` | Attention block comparison | P1 |
| `TMMA` | Transformer HLS | HLS kernels | MHA/MLP comparison | P2 |
| `Transformer-Accelerator-Based-on-FPGA` | RTL attention blocks | `In Progress/src/Softmax_top.v` | Softmax/MHA reference | P1 |
| `Transformer_dataflow` | Dataflow archive | no root README noted | Needs filtering | P3 |
| `ViM-Q-FCCM-2026` | Quantized ViM hardware | `HW/SPINAL/src/main/verilog/*/all.v` | Quantized vision sequence accelerator | P1 |
| `ViT-Accelerator` | HLS ViT | `hls_source/kernel.cpp`, `scripts/run_hls.tcl` | Direct HLS kernel baseline | P0 |
| `ViT-Accelerator-on-FPGA-with-INT8-quantization` | INT8 HLS ViT | `vitis_hls_proj/run_hls.tcl` | INT8 baseline and HLS project pattern | P1 |
| `ViT-FPGA-TPU` | FPGA/TPU-style ViT | `code_fpga/accel_driver/include/accelerator.h` | Host/accelerator interface reference | P1 |
| `ViTALiTy` | ViT hardware-aware attention | source tree | Approximate attention comparison | P1/P2 |
| `ViTCoD` | sparse ViT co-design | codebase sources | Sparse attention co-design | P1 |
| `acap-gemm-sa` | ACAP GEMM systolic array | GEMM sources | Systolic MAC reference | P2 |
| `efficient-transformer-accelerator` | quantized systolic RTL | `hw/src/systolic_quant_32x16.sv`, `quant_top.sv`, `quant.sv` | DSP/systolic quant datapath | P0 |
| `hls-spmv` | HLS sparse matrix kernels | `src/spmv_fast_stream.cpp`, variants | Stream/memory microbenchmark | P2 |
| `submission` | CPU/GPU benchmark archive | `cpu_benchmarks`, `gpu_benchmarks` | Reference only; noisy | P3 |
| `trans-fat` | transformer/quant archive | software-focused | Indirect quantization reference | P3 |
| `transformer-hls-thesis` | transformer HLS | HLS thesis code | HLS block decomposition | P1 |
| `vit-tiny-accelerator` | tiny ViT accelerator | RTL/HLS sources | Small end-to-end baseline | P2 |

## 3. Hardware Architecture Lessons

| Lesson | Sources | HGTXR action |
|---|---|---|
| DSP PE structure should be explicit and visible in reports | `MSD-FCCM23`, `efficient-transformer-accelerator`, `acap-gemm-sa` | Add attention/QKV/MLP/block-level DSP/LUT/URAM breakdown for each HLS variant |
| Large SRAM-style attention buffers map better to URAM/BRAM than LUTRAM | `AURA`, `TATAA`, `HG-PIPE` | Bind deep FIFO, activation, and score/value buffers to URAM when depth is large |
| HLS project scripts are part of reproducibility | `ViT-Accelerator`, `AGNA-FCCM2023`, `ViT-Accelerator-on-FPGA-with-INT8-quantization` | Store run scripts and synthesis reports with each experiment variant |
| Softmax/nonlinear blocks are high-risk for LUT growth | `Transformer-Accelerator-Based-on-FPGA`, `TATAA`, quantization papers | Keep approximation optional and report LUT size/precision/calibration |
| Host/driver boundaries need explicit measurement | `REMOT-FPGA-22`, `ViT-FPGA-TPU`, `Edge-MoE` | Add DMA bandwidth, batch size, mode latency, p95/p99 reporting when board data exists |

## 4. C3b Successor Hardware Work Items

| Work item | Reference | Expected result | Validation |
|---|---|---|---|
| Increase QKV and MLP parallel lanes | `HG-PIPE`, `MSD-FCCM23`, `efficient-transformer-accelerator` | Higher DSP utilization, lower LUT-dominant arithmetic | csynth resource breakdown by RMU/SMU/MHA/MLP |
| Bind large score/value/activation buffers to URAM | `AURA`, `TATAA`, `HG-PIPE` | Higher URAM use and reduced BRAM/LUTRAM pressure | HLS bind-storage report and utilization table |
| Keep small routing/scale tables in LUTRAM | `P2-ViT`, `Edge-MoE` | Avoid wasting BRAM/URAM on scalar tables | object-level memory report |
| Add softmax/LUT approximation calibration harness | `Transformer-Accelerator-Based-on-FPGA`, `TATAA`, `AHCPTQ` | Quantified approximation error | bit-accurate sim and error distribution |
| Add host/DMA reporting template | `REMOT-FPGA-22`, `ViT-FPGA-TPU` | Complete board-level methodology | measured board JSON, not static analysis |

## 5. Risks and Non-Claims

| Item | Status |
|---|---|
| Board-measured latency, DMA bandwidth, p95/p99 | Not measured by this analysis |
| ZCU104 utilization for these references | Not normalized; target boards differ |
| Accuracy improvements | Not claimed without HGTXR rerun |
| Importing full reference RTL | Not recommended; use design patterns and bounded experiments |

