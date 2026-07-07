---
source_type: codebase
source_name: TeraFly
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/TeraFly/analysis.md -->

# TeraFly Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly`
- repo_remote: `origin	https://github.com/zjnyly/TeraFly (fetch)`
- category: `multi-node LLM FPGA accelerator repo`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `5953` |
| 주요 언어 | `.bin:5871, C++:14, Python:11, C/C++ header:7, no_ext:6, JSON:4, .mk:4, .js:4, .png:4, Markdown:3` |
| LOC 추정 | `.txt:62000, C++ header:24766, C++:5256, Python:1744, C/C++ header:1668, Markdown:215, JSON:16, Shell:4` |

### Directory Map
- `LLM-demo-gui/` (2 entries)
- `LLM-demo-gui/alveo/` (4 entries)
- `LLM-demo-gui/llm-gui/` (4 entries)
- `OPT-1.3b_optimize/` (19 entries)
- `OPT-1.3b_optimize/build_dir.hw.xilinx_u50lv_gen3x4_xdma_2_202010_1/` (6 entries)
- `OPT-1.3b_optimize/tokenizer/` (13 entries)
- `assets/` (1 entries)
- `template/` (13 entries)

### Metadata / Config Refs
- `OPT-1.3b.json`
- `README.md`
- `LLM-demo-gui/alveo/tokenizer-1.3b/special_tokens_map.json`
- `LLM-demo-gui/alveo/tokenizer-1.3b/tokenizer_config.json`
- `LLM-demo-gui/alveo/tokenizer-1.3b/vocab.json`
- `LLM-demo-gui/llm-gui/README.md`

### Core Source Refs
- `LLM-demo-gui/alveo/tokenizer-1.3b/merges.txt`: 50001 lines
- `OPT-1.3b_optimize/tokenizer/json.hpp`: 24766 lines
- `OPT-1.3b_optimize/tokenizer/output.txt`: 11972 lines
- `OPT-1.3b_optimize/loopLynx.cpp`: 1705 lines
- `OPT-1.3b_optimize/data.h`: 1247 lines
- `template/attention.cpp`: 445 lines
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict_eigen.cpp`: 441 lines
- `codegen.py`: 418 lines
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict.cpp`: 413 lines
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict_original.cpp`: 412 lines

### Hardware-Oriented Source Refs
- `template/kernels.cpp`
- `template/router.cpp`
- `template/gemm_quant.cpp`
- `template/layerNorm.cpp`
- `template/adder_tree_64.cpp`
- `template/top.cpp`
- `template/top.h`
- `template/adder_tree_32.cpp`
- `template/attention.cpp`
- `OPT-1.3b_optimize/host.cpp`
- `OPT-1.3b_optimize/loopLynx.cpp`
- `OPT-1.3b_optimize/loopLynx.h`
- `OPT-1.3b_optimize/params.h`
- `OPT-1.3b_optimize/data.h`
- `OPT-1.3b_optimize/utils.h`
- `OPT-1.3b_optimize/tokenizer/json.hpp`
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict.cpp`
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict_original.cpp`
- `OPT-1.3b_optimize/tokenizer/loopLynx.h`
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict_generate.cpp`
- `OPT-1.3b_optimize/tokenizer/tokenizer_predict_eigen.cpp`
- `OPT-1.3b_optimize/tokenizer/utils.h`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/README.md`
- # 🚀 Terafly: A Multi-Node FPGA-Based Accelerator for Efficient Cooperative LLM Inference
- > **Terafly** enables high-throughput, low-latency inference of Large Language Models (LLMs) by leveraging a multi-node FPGA architecture optimized for cooperative execution.
- ---
- ## 💡 Highlight
- We provide **HLS kernels** that can be rapidly customized for research purposes, enabling efficient experimentation and algorithm validation on FPGAs.
- ---
- ## 🔍 Overview
- ---
- ## 📚 Related Work
- If you're exploring FPGA-based LLM acceleration, you might also be interested in:
- - [**llama-fpga**](https://github.com/adamgallas/llama-fpga)
- ---
- ## ⚙️ Prerequisites
- To ensure compatibility, we recommend replicating our experimental environment:
- | Component        | Version / Configuration                     |
- |------------------|---------------------------------------------|
- | **OS**           | Ubuntu 18.04                                |
- | **Shell**        | `xilinx-u50lv-gen3x4-xdma-base_2`           |
- | **XRT**          | 2023.2                                      |
- | **Vitis HLS & Vivado** | 2023.2                              |

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/README.md:113:  title          = {LoopLynx: {A} Scalable Dataflow Architecture for Efficient {LLM} Inference},`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:300:mem_pragma = "#pragma HLS interface m_axi port = w_addr_{processor_id} offset = slave bundle = gmem{processor_id}\n"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/OPT-1.3b_optimize/summary.csv:15:Device,Compute Unit,Kernel,Global Work Size,Local Work Size,Number Of Calls,Dataflow Execution,Max Overlapping Executions,Dataflow Acceleration,Total Time (ms),Minimum Time (ms),Average Time (ms),Maximum Time (ms),Clock Frequency (MHz),`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:15:#pragma HLS pipeline II = 1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:19:#pragma HLS UNROLL`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:35:#pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:53:// #pragma HLS UNROLL factor=INP_PARALLEL/2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:54:#pragma HLS UNROLL`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:79:#pragma HLS DATAFLOW`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:81:#pragma HLS STREAM variable = res_out depth = 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:82:#pragma HLS BIND_STORAGE variable = res_out type = fifo impl = srl`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:84:#pragma HLS STREAM variable = for_router depth = 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:85:#pragma HLS BIND_STORAGE variable = for_router type = fifo impl = srl`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:87:#pragma HLS STREAM variable = from_router depth = 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/layerNorm.cpp:88:#pragma HLS BIND_STORAGE variable = from_router type = fifo impl = srl`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/attention.cpp:18:#pragma HLS DATAFLOW`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/attention.cpp:20:#pragma HLS STREAM variable = head_loader depth = 2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/attention.cpp:21:#pragma HLS BIND_STORAGE variable = head_loader type = fifo impl = srl`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/attention.cpp:32:#pragma HLS DATAFLOW`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/template/attention.cpp:52:#pragma HLS UNROLL`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:28:    "ATTENTION_CHANNELS": config["HEAD_PARALLEL"] * config["HEAD_LEN"] // config["INP_NUM"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:150:# 3. generate layerNorm`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:153:with open('template/layerNorm.cpp', 'r') as file:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:223:# 5. generate attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:226:with open('template/attention.cpp', 'r') as file:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:231:    ring_connection_str = '\trouter_attention(ctx_requant_merged, stream_previous, stream_next, router_out);\n\twrite_buffer_int(router_out, output_buffer, device_id);\n'`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:238:for processor_id in range(calculated_values['ATTENTION_CHANNELS']):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:243:for processor_id in range(calculated_values['ATTENTION_CHANNELS']):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:244:    formatted_str = v_loader.format(processor_id=processor_id + calculated_values['ATTENTION_CHANNELS'], stream_id = processor_id)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:251:for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:268:    'WEATHER_USE_ROUTER_ATTENTION': ring_connection_str,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:274:attention_str = ""`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:279:attention_str = current_content`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:308:for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/TeraFly/codegen.py:314:for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):`

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
