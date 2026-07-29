---
source_type: codebase
source_name: ternaryLLM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/ternaryLLM/analysis.md -->

# ternaryLLM Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM`
- repo_remote: `not_available`
- category: `ternary LLM FPGA experiments`
- HGTXR relevance: `low`
- matched_paper: `TerEffic`
- paper_title: TerEffic: Highly Efficient Ternary LLM Inference on FPGA
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/TerEffic.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `93` |
| 주요 언어 | `Scala:35, Python:13, no_ext:6, C++:6, Markdown:5, C++ header:5, .svh:3, Notebook:2, .cuh:2, .sbt:2` |
| LOC 추정 | `C++:20133, Scala:5169, Notebook:4591, C++ header:1695, Python:1205, Markdown:308, .txt:45, JSON:42, Shell:4` |

### Directory Map
- `SSR/` (1 entries)
- `ternaryLLM_CPU/` (7 entries)
- `ternaryLLM_CPU/src/` (9 entries)
- `ternaryLLM_FPGA/` (10 entries)
- `ternaryLLM_FPGA/Results/` (2 entries)
- `ternaryLLM_FPGA/coyote_files/` (2 entries)
- `ternaryLLM_FPGA/hw/` (3 entries)
- `ternaryLLM_FPGA/png/` (2 entries)
- `ternaryLLM_FPGA/project/` (1 entries)
- `ternaryLLM_FPGA/sw/` (2 entries)
- `ternaryLLM_GPU/` (11 entries)
- `ternaryLLM_GPU/TernaryLLM/` (7 entries)
- `ternaryLLM_GPU/benchmark/` (2 entries)
- `ternaryLLM_GPU/config/` (1 entries)
- `ternaryLLM_GPU/csrc/` (3 entries)

### Metadata / Config Refs
- `README.md`
- `ternaryLLM_CPU/README.md`
- `SSR/README.md`
- `ternaryLLM_GPU/README.md`
- `ternaryLLM_GPU/config/config.json`
- `ternaryLLM_FPGA/README.md`

### Core Source Refs
- `ternaryLLM_CPU/src/GEMM_CPU_FP32.cpp`: 17041 lines
- `ternaryLLM_CPU/SIMD_Generator.ipynb`: 4124 lines
- `ternaryLLM_CPU/src/main.cpp`: 1946 lines
- `ternaryLLM_FPGA/sw/gemmacc/main.cpp`: 541 lines
- `ternaryLLM_CPU/src/LlamaModel.hpp`: 527 lines
- `ternaryLLM_CPU/src/initData.hpp`: 524 lines
- `ternaryLLM_FPGA/hw/spinal/gemmacc/util/AxiMemorySim.scala`: 515 lines
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/design/DataFSM.scala`: 480 lines
- `ternaryLLM_GPU/Benchmarks.ipynb`: 467 lines
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/Op/coyote_v2/DataFSM.scala`: 448 lines

### Hardware-Oriented Source Refs
- `ternaryLLM_CPU/src/initData.hpp`
- `ternaryLLM_CPU/src/initData.cpp`
- `ternaryLLM_CPU/src/LlamaModel.hpp`
- `ternaryLLM_CPU/src/GEMM_CPU_FP32.cpp`
- `ternaryLLM_CPU/src/GEMM_CPU_FP32.hpp`
- `ternaryLLM_CPU/src/TCSC.hpp`
- `ternaryLLM_CPU/src/main.cpp`
- `ternaryLLM_CPU/src/GEMM_CPU_INT8.cpp`
- `ternaryLLM_CPU/src/GEMM_CPU_INT8.hpp`
- `ternaryLLM_FPGA/hw/spinal/MatrixAdd/logic.scala`
- `ternaryLLM_FPGA/hw/spinal/MatrixAdd/Wrap_logic.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Op/DataFSMOPSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Op/TopLevelOPSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Op/coyote_v2/TopLevelSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Base/DataFSMSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Base/TernaryGEMMSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Base/ActivationBufferSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Base/TopLevelBaseSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/old/Base/ActivationToTernaryGEMMSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/ternaryGEMM/TopLevelSim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/sim/testLogic/logic_sim.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/coyote/AxiCoyote.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/coyote/Types.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/Config.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/Generator.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/Op/coyote_v2/TopLevel_new.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/Op/coyote_v2/WrapGEMM.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/Op/coyote_v2/DataFSM.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/Op/coyote_v2/PE.scala`
- `ternaryLLM_FPGA/hw/spinal/gemmacc/src/old/Op/coyote_v2/old/WrapSys.scala`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/README.md`
- # Ternary LLM
- - Existing CPU and GPU do not support native 2-bit operations, and existing libraries like PyTorch and CUDA do not have dedicated computing kernels for ternary weights.
- - Existing sparse formats like Compressed Sparse Column are not optimized for ternary values, causing extra storage and decompression overhead.
- - Methods optimized for ternary weights, such as BitNet, RSR, and RSR++, fail to capture sparsity structures.
- Therefore, we aim to solve these challenges by novel algorithms, code optimization, and hardware accelerators. This repository contains code for three projects:
- - **SSR: Sparse Segment Reduction for Ternary GEMM Acceleration** (target limitation 3)
- - **Fast Ternary Large Language Model Inference with Addition-Based Sparse GEMM on Edge Devices** (target limitations 1 and 2)
- - **An Accelerator for Ternary Language Models based on FPGA** (target limitation 1)
- File organization and main contributors:
- - SSR: Adeline Pittet, Valerie Verdan, and Shien Zhu
- - ternaryLLM_CPU: Mila Kjoseva, and Shien Zhu
- - ternaryLLM_GPU: Guanshujie Fu
- - ternaryLLM_FPGA: Gabriele Giacone
- Please refer to the README inside each folder for the detailed experiment setups. If you find this repository helpful, please cite the following paper(s):
- ```
- @inproceedings{SSR_DATE_2026,
- title={SSR: Sparse Segment Reduction for Ternary GEMM Acceleration},
- author={Adeline Pittet and Shien Zhu and Valerie Verdan and Gustavo Alonso},
- booktitle={Design, Automation and Test in Europe (DATE)},
- year={2026}

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: ternary quantization과 custom compute/memory hierarchy로 on-chip inference를 지향한다.
- 알고리즘 축: ternary weight compression과 ternary compute unit, HBM-assisted variant를 구성한다.
- 하드웨어 축: fully on-chip smaller model path와 HBM-assisted larger model path를 제안한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_llama.py:10:from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:281:    "from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_mlp.py:69:        attention_dropout=llama_3_1b_json["attention_dropout"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_mlp.py:70:        num_attention_heads=llama_3_1b_json["num_attention_heads"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:3:from TernaryLLM import TernaryConfig, LlamaTernaryAttention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:4:from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:9:        attn_layer = LlamaTernaryAttention(config, 0).to('cuda')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:11:        attn_layer = LlamaAttention(config, 0).to('cuda')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:12:        # attn_layer = LlamaSdpaAttention(llama_config, 0).to('cuda')`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:31:            output = attn_layer(hidden_states=input_hidden_states, position_embeddings=position_embeddings, attention_mask=None)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:41:            output = attn_layer(hidden_states=input_hidden_states.detach(), position_embeddings=position_embeddings,  attention_mask=None)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:70:        attention_dropout=llama_3_1b_json["attention_dropout"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/benchmark_ternary_attn.py:71:        num_attention_heads=llama_3_1b_json["num_attention_heads"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:24:    "from transformers.models.llama.modeling_llama import LlamaAttention, LlamaRotaryEmbedding\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:25:    "from TernaryLLM import TernaryConfig, LlamaTernaryAttention\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:30:    "        attn_layer = LlamaTernaryAttention(config, 0).to('cuda')\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:32:    "        attn_layer = LlamaAttention(config, 0).to('cuda')\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:33:    "        # attn_layer = LlamaSdpaAttention(llama_config, 0).to('cuda')\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:50:    "            output = attn_layer(hidden_states=input_hidden_states, position_embeddings=position_embeddings, attention_mask=None)\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:60:    "            output = attn_layer(hidden_states=input_hidden_states.detach(), position_embeddings=position_embeddings,  attention_mask=None)\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:96:    "    attention_dropout=llama_3_1b_json[\"attention_dropout\"],\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:97:    "    num_attention_heads=llama_3_1b_json[\"num_attention_heads\"],\n",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/README.md:3:Large Language Models (LLMs) require substantial computational resources, limiting their deployment on resource-constrained hardware. Ternary LLMs mitigate these demands through weight quantization via 2-bit ternary values {-1, 0, +1}, achieving significant compression often with 50 − 90% sparsity. However, existing approaches face limitations: `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:18:   "id": "97edcfbd-c474-4371-b965-4ada895687d6",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_GPU/Benchmarks.ipynb:211:   "id": "dfd52d78-09bf-4c87-bfe8-0ad784a4c3a8",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/README.md:3:Large Language Models contain billions of parameters, leading to high demands on computation, memory, and energy. These requirements pose significant challenges for edge deployment, where latency, power, and hardware resources are strictly constrained. Ternary quantization addresses these challenges by reducing weights to three discrete values {-1, 0, 1}, which decreases model size and enables efficient sparse matrix multiplication by replacing multiplications with simple additions and subtractions, thereby lowering memory usage and compute cost. Since CPUs and GPUs are not well optimized for arbitrary sparse matrix multiplication, this thesis/repository develops an FPGA-based accelerator for ternary sparse GEMM in LLM inference.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:54:void generateX(int8_t *X, int M, int K)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:75:void generateWUniformal(uint8_t *W, int S, int entries, int N)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:115:int16_t *naiveGEMM(int M, int N, int S, int K, int entries, const int8_t *X, const uint8_t *W, int Nz_K_Slice)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:153:                uint8_t pos = W[w_index] + slice_weights * K_SLICE;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:154:                uint8_t neg = W[w_index + 1] + slice_weights * K_SLICE;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:193:    u_int size_X = (uint)(M * K) * (uint)sizeof(int8_t);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:194:    u_int size_W = (uint)(total_row_W * S_SLICE) * (uint)sizeof(uint8_t);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:209:    int8_t *X = (int8_t *)coyote_thread->getMem({CoyoteAlloc::HPF, size_X});`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/ternaryLLM/ternaryLLM_FPGA/sw/gemmacc/main.cpp:210:    uint8_t *W = (uint8_t *)coyote_thread->getMem({CoyoteAlloc::HPF, size_W});`

## 5. HGTXR 적용 해석
- 정확도 리스크가 큰 ternary 압축은 P3 negative-control이다.

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
