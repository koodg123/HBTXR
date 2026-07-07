---
source_type: codebase
source_name: UbiMoE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/UbiMoE/analysis.md -->

# UbiMoE Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE`
- repo_remote: `origin	https://github.com/DJ000011/UbiMoE (fetch)`
- category: `MoE-ViT FPGA accelerator and search`
- HGTXR relevance: `high`
- matched_paper: `UbiMoE`
- paper_title: UbiMoE: A Ubiquitous Mixture-of-Experts Vision Transformer Accelerator With Hybrid Computation Pattern on FPGA
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/UbiMoE.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `60` |
| 주요 언어 | `C++ header:30, C++:20, .png:4, Markdown:3, Python:3` |
| LOC 추정 | `C++:4517, C++ header:1088, Markdown:473, Python:334` |

### Directory Map
- `ViT_Fixed16/` (6 entries)
- `ViT_Fixed16/HW_src/` (7 entries)
- `ViT_Fixed16/SW_src/` (5 entries)
- `ViT_Fixed16/include/` (15 entries)
- `images/` (2 entries)
- `src/` (26 entries)
- `testbench/` (1 entries)

### Metadata / Config Refs
- `README.md`
- `ViT_Fixed16/README.md`
- `ViT_Fixed16/SW_src/README.md`

### Core Source Refs
- `testbench/moe_tb.cpp`: 654 lines
- `src/linear.cpp`: 641 lines
- `src/moe.cpp`: 441 lines
- `ViT_Fixed16/SW_src/testbench.cpp`: 441 lines
- `ViT_Fixed16/HW_src/linear.cpp`: 347 lines
- `ViT_Fixed16/SW_src/README.md`: 340 lines
- `src/attention.cpp`: 309 lines
- `ViT_Fixed16/HW_src/attention.cpp`: 305 lines
- `src/conv.cpp`: 205 lines
- `ViT_Fixed16/include/xcl2.cpp`: 180 lines

### Hardware-Oriented Source Refs
- `testbench/moe_tb.cpp`
- `src/moe.hpp`
- `src/gelu.hpp`
- `src/gelu.cpp`
- `src/linear.cpp`
- `src/ViT_compute.cpp`
- `src/datatypes.hpp`
- `src/Feed_Forward.hpp`
- `src/conv.hpp`
- `src/attention.hpp`
- `src/model.hpp`
- `src/conv.cpp`
- `src/hardware.hpp`
- `src/layernorm.cpp`
- `src/kernel.hpp`
- `src/util.hpp`
- `src/add.hpp`
- `src/patch_embed.cpp`
- `src/layernorm.hpp`
- `src/patch_embed.hpp`
- `src/Feed_Forward.cpp`
- `src/add.cpp`
- `src/tbutil.hpp`
- `src/dcl.hpp`
- `src/linear.hpp`
- `src/moe.cpp`
- `src/attention.cpp`
- `ViT_Fixed16/HW_src/gelu.cpp`
- `ViT_Fixed16/HW_src/linear.cpp`
- `ViT_Fixed16/HW_src/ViT_compute.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/README.md`
- # UbiMoE: a Ubiquitous Mixture-of-Experts Vision Transformer Accelerator with Hybrid Computation Pattern on FPGA
- ## Environment
- - **Ubuntu 20.04**
- - **Vitis**, **XRT** (Xilinx Runtime) and **XCU280 platform** 2022.1 [link](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/alveo/u280.html)
- - **Model and Dataset** : We use the [$M^3ViT$](https://github.com/VITA-Group/M3ViT) as our model and evaluate on [Cityscape Dataset](https://www.cityscapes-dataset.com/)
- ## Overview
- ## Compile and Run
- - **Open Vitis tools**：
- ```shell
- source /opt/Xilinx/xrt/setup.sh
- source /path/to/vitis/2022.1/settings64.sh
- vitis
- ```
- - **Create new projects**:
- Select the Application Project with the corresponding board files. After that, you can open the project as follows：
- - **Add sources**:
- Place the src files in the `project_name_kernels` directory (eg,`dct_project_kernels` in figure), and the testbench files in the `project_name` directory.
- - **Set top functions**:
- - **Set compile configs**:
- - **Build and Run**:

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: latency-optimized streaming attention과 resource-efficient reusable linear kernel, two-stage heuristic search를 결합한다.
- 알고리즘 축: resource constraint를 입력으로 하여 hardware parameter를 탐색하고 MoE attention/linear kernel을 hybrid computation pattern으로 배치한다.
- 하드웨어 축: streaming attention kernel과 reusable linear kernel을 중심으로 ZCU102/U280같은 서로 다른 resource budget에 맞춘다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/README.md:46:This design enables concurrent execution of MSA and FFN modules. Within each module, kernels are activated in parallel to further enhance throughput. The result is a pipeline architecture that overlaps computation across modules and kernels for improved performance.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/SW_src/README.md:131:#pragma HLS interface m_axi port=x offset=slave bundle=in`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/SW_src/README.md:132:#pragma HLS interface m_axi port=attn_weights offset=slave bundle=weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/SW_src/README.md:199:#pragma HLS interface m_axi port=attn_weights bundle=weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/SW_src/README.md:200:#pragma HLS interface m_axi port=proj_weights bundle=weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/SW_src/README.md:201:#pragma HLS interface m_axi port=vit_weights_l1 bundle=weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/SW_src/README.md:202:#pragma HLS interface m_axi port=vit_weights_l2 bundle=weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/add.cpp:4:#pragma HLS inline off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/add.cpp:11:#pragma HLS pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/add.cpp:12:#pragma HLS dependence variable=x_blocks inter false`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/add.cpp:13:#pragma HLS dependence variable=y_blocks inter false`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/add.cpp:14:#pragma HLS dependence variable=out_blocks inter false`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:26:#pragma HLS interface m_axi depth = 1 port = input offset = slave bundle = in max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:27:#pragma HLS interface m_axi depth = 1 port = output offset = slave bundle = out max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:28:#pragma HLS interface m_axi depth = 1 port = moe_w_gate offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:29:#pragma HLS interface m_axi depth = 1 port = moe_weights_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:30:#pragma HLS interface m_axi depth = 1 port = moe_bias_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:31:#pragma HLS interface m_axi depth = 1 port = moe_weights_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:32:#pragma HLS interface m_axi depth = 1 port = moe_bias_l2 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/Feed_Forward.cpp:33:#pragma HLS interface m_axi depth = 1 port = vit_weights_l1 offset = slave bundle = weights max_widen_bitwidth = AXI_XFER_BIT_WIDTH`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/README.md:13:Inspired by the Computation Reordering method in [$M^3ViT$](https://github.com/VITA-Group/M3ViT), UbiMoE focuses on the differing memory access requirements of the attention and FFN components in MoE-ViT to achieve a trade-off between resource utilization and performance. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/README.md:15:In practice, we implement a fully streaming attention kernel optimized for latency and a reusable linear kernel optimized for resource efficiency. Since all the cores are parameterized, they can be easily adapted to different FPGAs.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/README.md:24:The UbiMoE accelerator is designed to execute the Multi-Head Self-Attention (MSA) and Feed-Forward Network (FFN) modules in parallel. The directory structure and kernel layout are as follows:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/README.md:29:│   ├── layerNorm_kernel        # Layer normalization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/README.md:33:├── atten_compute/              # HLS kernels for attention computation`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/README.md:34:│   ├── attention_kernel        # Scaled dot-product attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/ViT_Fixed16/README.md:38:├── layerNorm_kernel            # Layer normalization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:1:#include "attention.hpp"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:30:	FOR_BLOCK(q_patch, NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:31:			  ATTN_MATMUL_PARALLEL)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:47:void compute_q_matmul_k(hls::stream<attn_parallel_t> &attn_stream, hls::stream<fm_block_t> &q_stream, hls::stream<fm_block_t> &k_stream, hls::stream<heads_t> &attnmax_stream)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:51:	fm_blocks_t q_blocks[ATTN_MATMUL_PARALLEL];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:54:	fm_t attn_blocks[ATTN_MATMUL_PARALLEL][NUM_HEADS];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:55:	fm_t attn_max[ATTN_MATMUL_PARALLEL][NUM_HEADS];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/UbiMoE/src/attention.cpp:60:	static_assert(ATTN_MATMUL_PARALLEL < NUM_PATCHES, "ATTN_MATMUL_PARALLEL must be less than NUM_PATCHES");`

## 5. HGTXR 적용 해석
- C3b 이후 PAR/MEM bank sweep의 search policy 참고로 유용하다. search/track expert 실험의 hardware scheduling 근거가 된다.

### 적용 가능 모듈
- Search/Track mode-specific expert or head routing
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
