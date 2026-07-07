---
source_type: codebase
source_name: AURA-FlashAttention-AISC-Accelerator
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/AURA-FlashAttention-AISC-Accelerator/analysis.md -->

# AURA-FlashAttention-AISC-Accelerator Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator`
- repo_remote: `not_available`
- category: `SystemVerilog FlashAttention ASIC accelerator`
- HGTXR relevance: `medium`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `107` |
| 주요 언어 | `SystemVerilog:40, .mem:23, C++:11, .dec:9, YAML:6, Python:5, no_ext:4, .txt:2, Shell:1, .rc:1` |
| LOC 추정 | `SystemVerilog:6041, C++:1482, .txt:561, Python:323, Tcl:303, YAML:80, JSON:43, Shell:26, Markdown:11` |

### Directory Map
- `.github/` (1 entries)
- `.github/ISSUE_TEMPLATE/` (6 entries)
- `.vscode/` (1 entries)
- `_deprecated/` (14 entries)
- `cpp/` (12 entries)
- `docs/` (1 entries)
- `include/` (1 entries)
- `models/` (3 entries)
- `models/bert-base-uncased/` (9 entries)
- `models/bert-large-cased/` (18 entries)
- `models/random_test1/` (5 entries)
- `python/` (5 entries)
- `synth/` (1 entries)
- `test/` (7 entries)
- `verilog/` (20 entries)

### Metadata / Config Refs
- `README.md`
- `.vscode/settings.json`
- `synth/AURAsynth.tcl`

### Core Source Refs
- `test/debug.txt`: 514 lines
- `verilog/memory_controller.sv`: 447 lines
- `test/aura_test.sv`: 367 lines
- `_deprecated/memctrlV2.sv`: 354 lines
- `_deprecated/memory_controller.sv`: 341 lines
- `test/int_division_test.sv`: 315 lines
- `test/expmul_test.sv`: 307 lines
- `synth/AURAsynth.tcl`: 303 lines
- `_deprecated/memory_controller_old.sv`: 301 lines
- `_deprecated/math_utils_pkg.sv`: 248 lines

### Hardware-Oriented Source Refs
- `verilog/reduction_step.sv`
- `verilog/dot_product.sv`
- `verilog/q_align_frac.sv`
- `verilog/tree_reduce.sv`
- `verilog/memory_controller.sv`
- `verilog/PE.sv`
- `verilog/OSRAM.sv`
- `verilog/expmul_stage.sv`
- `verilog/int_division.sv`
- `verilog/q_saturate.sv`
- `verilog/expmul.sv`
- `verilog/max.sv`
- `verilog/VSRAM.sv`
- `verilog/AURA.sv`
- `verilog/q_convert.sv`
- `verilog/QSRAM.sv`
- `verilog/KSRAM.sv`
- `verilog/q_sign_extend.sv`
- `verilog/vector_division.sv`
- `verilog/q_align_int.sv`
- `cpp/input_to_f8.cpp`
- `cpp/attention_f8.cpp`
- `cpp/generate_output_f8.cpp`
- `cpp/precision_measure.cpp`
- `cpp/generate_output_fp64.cpp`
- `cpp/fp32_to_f8.cpp`
- `cpp/precision_measuref16.cpp`
- `cpp/attention_fp32.cpp`
- `cpp/output_to_f8.cpp`
- `cpp/fp32_to_f16.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/README.md`
- # A.U.R.A.---FlashAttention-ASIC-Accelerator
- A.U.R.A. is a SystemVerilog based ASIC accelerator for the FlashAttention kernel used in modern transformers.
- ### Problem Definition and Motivation
- ### Related Work

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/math_utils_pkg.sv:217:    //We can throw away some fractional bits in the middle of the pipeline if they cannot`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/dot_product_test.sv:179:            drain_cycles = 1 + `NUM_REDUCE_STAGES; // worst-case latency of pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/dot_product_test.sv:199:                $display("[ERROR] Pipeline drained but %0d results still pending!",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/mem.sv:5:// Description : This is a clock-based latency, pipelined memory with  //`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/multiply.sv:23:    //Internal Pipeline Registers`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/multiply.sv:41:            end else if(rdy_in) begin //Only downstream is ready (clear internal pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/handshake_reg.sv:19:    //Internal Pipeline Registers`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/handshake_reg.sv:36:            end else if(rdy_in) begin //Only downstream is ready (clear internal pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/int2_division.sv:36:    //         end else if(rdy_in) begin //Only downstream is ready (clear internal pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/aura_test.sv:5://          like in test/pipeline_print.c or just do everything in verilog.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/aura_test.sv:9:// These link to the pipeline_print.c file in this directory, and are used below to print`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/aura_test.sv:10:// detailed output to the pipeline_output_file, initialized by open_pipeline_output_file()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/aura_test.sv:12://import "DPI-C" function void open_pipeline_output_file(string file_name);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/aura_test.sv:19://import "DPI-C" function void close_pipeline_output_file();`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/test/aura_test.sv:29:    // named OUTPUT.{out cpi, wb, ppln} for the memory, cpi, writeback, and pipeline outputs.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/vec_add.sv:23:    //Internal Pipeline Registers`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/vec_add.sv:42:            end else if(rdy_in) begin //Only downstream is ready (clear internal pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/dot_product.sv:28:    //Internal Pipeline Registers`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/dot_product.sv:57:            end else if(all_valid && rdy_in && (row_counter == 0)) begin //Only downstream is ready (clear internal pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/_deprecated/dot_product.sv:72:            end else if(all_valid && rdy_in) begin //Only downstream is ready (clear internal pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/README.md:1:# A.U.R.A.---FlashAttention-ASIC-Accelerator`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/README.md:2:A.U.R.A. is a SystemVerilog based ASIC accelerator for the FlashAttention kernel used in modern transformers.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/README.md:5:The use of Transformers for various machine learning applications is becoming increasingly common. Their ability to capture more context and use attention to focus on specific parts of the input leads to more accurate and desirable outputs. However, they are computationally expensive, which restricts their use to situations where large amounts of power are readily available. A custom hardware accelerator built to perform the specific calculations required by a Transformer will enable the model to be used in more low power settings.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/README.md:7:We will be focusing on machine learning applications on the edge.  These applications require low power and area costs while maintaining high accuracy and performance.  Our project aims to build on previous accelerator architectures and algorithmic advancements to create a new state-of-the-art, open-source, edge accelerator ASIC for the FlashAttention kernel used in modern Transformers. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/README.md:10:A comprehensive breakdown of prior works in this area is detailed in the Appendix.  We have found that most previous implementations for attention accelerators are not tuned for edge devices with tight power and area constraints.  Many designs are FPGA based which impose more overhead for performance, or they use large systolic arrays that do not translate well to smaller architectures.  One architecture, SwiftTron, was specifically developed to target tinyML applications.  However, it was implemented without considerations for FlashAttention and other modern hardware-algorithm co-optimizations such as ExpMul and FLASH-D.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/Makefile:42:# cpp/attention_fp32: cpp/attention_fp32.cpp | cpp`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/Makefile:43:# 	$(CXX) $(CXXFLAGS) -o cpp/attention_fp32.cpp cpp/attention_fp32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/Makefile:294:models/%/O_float_correct.mem: models/%/Q32.mem models/%/K32.mem models/%/V32.mem | cpp/attention_fp32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/Makefile:296:	./cpp/attention_fp32 $^ $@`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/python/QKV_obtain.py:14:HEAD_IDX = 0          # which attention head to extract`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/python/QKV_obtain.py:42:model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/python/QKV_obtain.py:49:# Extract Q/K/V from first attention layer`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/python/QKV_obtain.py:51:attn_layer = model.encoder.layer[0].attention.self`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/python/QKV_obtain.py:55:    Q = attn_layer.query(x).view(B, T, attn_layer.num_attention_heads, -1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/AURA-FlashAttention-AISC-Accelerator/python/QKV_obtain.py:56:    K = attn_layer.key(x).view(B, T, attn_layer.num_attention_heads, -1)`

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
