---
source_type: codebase
source_name: FlexLLM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/FlexLLM/analysis.md -->

# FlexLLM Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM`
- repo_remote: `not_available`
- category: `composable HLS library for transformer accelerators`
- HGTXR relevance: `medium`
- matched_paper: `FlexLLM`
- paper_title: FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/FlexLLM.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `196` |
| 주요 언어 | `C/C++ header:52, C++:42, Python:31, JSON:27, .ini:18, no_ext:9, .txt:9, .xclbin:6, Markdown:1, .csv:1` |
| LOC 추정 | `C/C++ header:32158, C++:22618, Python:2111, JSON:1958, Markdown:124, .txt:118` |

### Directory Map
- `HMT_SpinQuant_Llama_32_1B/` (39 entries)
- `HMT_SpinQuant_Llama_32_1B/Rapidstream_pref_u280/` (9 entries)
- `Modules/` (17 entries)
- `SpinQuant_Llama_32_1B/` (33 entries)
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/` (10 entries)
- `SpinQuant_Llama_32_1B/RapidStream_pref_u280/` (9 entries)
- `SpinQuant_Llama_32_1B/run/` (14 entries)
- `SpinQuant_Llama_32_1B_Ins/` (34 entries)
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/` (10 entries)
- `SpinQuant_Llama_32_1B_Ins/RapidStream_pref_u280/` (9 entries)
- `SpinQuant_Llama_32_1B_Ins/run/` (13 entries)

### Metadata / Config Refs
- `README.md`
- `HMT_SpinQuant_Llama_32_1B/Rapidstream_pref_u280/impl_config.json`
- `HMT_SpinQuant_Llama_32_1B/Rapidstream_pref_u280/floorplan_config.json`
- `HMT_SpinQuant_Llama_32_1B/Rapidstream_pref_u280/u280_device.json`
- `HMT_SpinQuant_Llama_32_1B/Rapidstream_pref_u280/pipeline_config.json`
- `HMT_SpinQuant_Llama_32_1B/Rapidstream_pref_u280/floorplan_config_mem_opt.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/impl_config.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/floorplan_config_mem_opt_new.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/floorplan_config.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/u280_device.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/pipeline_config.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_dec_u280/floorplan_config_mem_opt.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_pref_u280/impl_config.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_pref_u280/floorplan_config.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_pref_u280/u280_device.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_pref_u280/pipeline_config.json`
- `SpinQuant_Llama_32_1B_Ins/RapidStream_pref_u280/floorplan_config_mem_opt.json`
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/impl_config.json`
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/floorplan_config_mem_opt_new.json`
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/floorplan_config.json`
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/u280_device.json`
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/pipeline_config.json`
- `SpinQuant_Llama_32_1B/RapidStream_dec_u280/floorplan_config_mem_opt.json`
- `SpinQuant_Llama_32_1B/RapidStream_pref_u280/impl_config.json`
- `SpinQuant_Llama_32_1B/RapidStream_pref_u280/floorplan_config.json`
- `SpinQuant_Llama_32_1B/RapidStream_pref_u280/u280_device.json`
- `SpinQuant_Llama_32_1B/RapidStream_pref_u280/pipeline_config.json`
- `SpinQuant_Llama_32_1B/RapidStream_pref_u280/floorplan_config_mem_opt.json`

### Core Source Refs
- `SpinQuant_Llama_32_1B_Ins/SpinQuant_Decoding_mem_opt_new.h`: 1170 lines
- `SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt_new.h`: 1170 lines
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt_new.h`: 1170 lines
- `SpinQuant_Llama_32_1B_Ins/SpinQuant_Decoding.h`: 1136 lines
- `SpinQuant_Llama_32_1B/SpinQuant_Decoding.h`: 1136 lines
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding.h`: 1136 lines
- `SpinQuant_Llama_32_1B_Ins/SpinQuant_Decoding_mem_opt.h`: 1135 lines
- `SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt.h`: 1135 lines
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt.h`: 1135 lines
- `Modules/Linear_Layer.h`: 1126 lines

### Hardware-Oriented Source Refs
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Prefilling_mem_opt_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Prefilling_mem_opt.h`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Prefilling.h`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Prefilling_Decoding_mem_opt_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/MHA_i8xi8_decoding_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Prefilling.h`
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Prefilling_mem_opt.h`
- `HMT_SpinQuant_Llama_32_1B/Linear_Layer_test.h`
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Prefilling_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/MHA_i8xi8_prefilling_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt_new.h`
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Prefilling_test.h`
- `HMT_SpinQuant_Llama_32_1B/config_u280_mem_opt.h`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Prefilling_Decoding_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/MHA_i8xi8.h`
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Unit.h`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding.h`
- `HMT_SpinQuant_Llama_32_1B/Linear_Layer_Decoding_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/Linear_Layer_Prefilling_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/HMT_SpinQuant_Prefilling_test_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/MHA_i8xi8_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/config_u280.h`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt_new_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Decoding_mem_opt.h`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Prefilling_tb.cpp`
- `HMT_SpinQuant_Llama_32_1B/SpinQuant_Prefilling_mem_opt_tb.cpp`
- `SpinQuant_Llama_32_1B_Ins/SpinQuant_Prefilling_mem_opt.h`
- `SpinQuant_Llama_32_1B_Ins/SpinQuant_Prefilling.h`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/README.md`
- # 🔥 FlexLLM: A Composable HLS Library for Rapid LLM Accelerator Design
- [![DOI](https://zenodo.org/badge/1109534356.svg)](https://doi.org/10.5281/zenodo.18793354)
- FlexLLM is a **composable High-Level Synthesis (HLS) library** for rapidly building **hybrid temporal–spatial accelerators** for Large Language Models (LLMs).
- It provides parameterized module templates, optimized memory-access/dataflow components, and a complete quantization suite, enabling FPGA-based LLM systems to be built with **minimal manual engineering effort**.
- Using FlexLLM, we implemented a **full Llama-3.2-1B inference system**—including prefill, decode, tokenizer integration, and long-context memory—**in under two months with ~1K lines of code**.
- ---
- ## ✨ Key Features
- - **Composable HLS Library** for LLM accelerator development
- - **Hybrid Temporal–Spatial Architecture**
- - **Hardware-Efficient Quantization Suite**
- - **Hierarchical Memory Transformer (HMT) Plug-In**
- - **FPGA Deployment Ready**
- ---
- ## 📊 Performance Summary
- ### AMD U280 FPGA (16nm) vs. NVIDIA A100 GPU (7nm)
- - 1.29× end-to-end speedup
- - 1.64× higher decode throughput
- - 3.14× better energy efficiency
- ### Projected V80 FPGA (7nm)
- - 4.71× end-to-end speedup

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: composable HLS library와 quantization suite로 hybrid temporal-spatial accelerator를 빠르게 구성한다.
- 알고리즘 축: prefill/decode stage에 따라 dataflow와 reuse를 다르게 구성하는 library abstraction을 제공한다.
- 하드웨어 축: HLS component library, stage-customized accelerator, long-context HMT plug-in으로 구성된다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/README.md:5:It provides parameterized module templates, optimized memory-access/dataflow components, and a complete quantization suite, enabling FPGA-based LLM systems to be built with **minimal manual engineering effort**.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:22:    #pragma HLS ARRAY_PARTITION variable=token_buffer dim=1 complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:23:    #pragma HLS ARRAY_PARTITION variable=token_buffer dim=2 type=block factor=hmt_t_block_parallel`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:26:    #pragma HLS ARRAY_PARTITION variable=hmt_Sn_Mn_buffer type=block factor=hmt_t_block_parallel`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:31:    #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:59:                        #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:68:                        #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:77:                        #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:86:                #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:101:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:107:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:117:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:135:                    #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:144:                    #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:153:                    #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:162:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:177:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:183:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:217:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/HMT.h:222:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Logits.h:34:template <typename T, int block_parallel_samp, int max_logits_num, int top_k, int max_hidden_dim=HIDDEN_DIM, int block_logits_num=HIDDEN_DIM, int inner_block_parallel=1, bool enable_softmax=true, bool enable_sub_max=true>`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Logits.h:113:    // 5) apply softmax if enabled`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Logits.h:114:    if (enable_softmax) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Logits.h:230:    // 5) apply softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:1:#ifndef _SOFTMAX_H`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:2:#define _SOFTMAX_H`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:7:void pref_Softmax(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:49:                        if(M==0 && H == 0 && k == 0) cout << "softmax input data: ";`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:248:void dec_Softmax(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:433:void dec_MHA_Softmax(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax_backup.h:472:                    if(H == 0 && k == 0) cout << "softmax input data: ";`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax.h:1:#ifndef _SOFTMAX_H`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax.h:2:#define _SOFTMAX_H`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax.h:7:void pref_Softmax(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/FlexLLM/Modules/Softmax.h:49:                        if(M==0 && H == 0 && k == 0) cout << "softmax input data: ";`

## 5. HGTXR 적용 해석
- 모델보다는 실험 생성/manifest/tooling 참조로 유용하다.

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
