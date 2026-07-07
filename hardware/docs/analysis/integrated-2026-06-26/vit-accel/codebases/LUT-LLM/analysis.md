---
source_type: codebase
source_name: LUT-LLM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/LUT-LLM/analysis.md -->

# LUT-LLM Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM`
- repo_remote: `origin	https://github.com/LUT-FPGA/LUT-LLM (fetch)`
- category: `memory-based LLM FPGA accelerator`
- HGTXR relevance: `low`
- matched_paper: `LUT-LLM`
- paper_title: LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/LUT-LLM.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `82` |
| 주요 언어 | `JSON:23, C++:12, no_ext:11, C/C++ header:11, .png:7, Tcl:5, Python:4, .rpt:2, .xo:2, Shell:2` |
| LOC 추정 | `C++:8233, C/C++ header:5351, Tcl:3581, Python:1248, JSON:559, Markdown:111, Shell:47` |

### Directory Map
- `attention_block/` (3 entries)
- `ccu/` (3 entries)
- `config/` (1 entries)
- `custom_design/` (4 entries)
- `ffn/` (3 entries)
- `figs/` (3 entries)
- `gqa/` (3 entries)
- `imm/` (3 entries)
- `lut-dla/` (6 entries)
- `qwen_block/` (9 entries)
- `qwen_lut_model/` (13 entries)
- `qwen_lut_model/param_settings/` (13 entries)
- `rapidstream_script/` (7 entries)
- `rms_norm/` (3 entries)
- `rope/` (3 entries)
- `silu/` (3 entries)

### Metadata / Config Refs
- `create_bd_design_final.tcl`
- `README.md`
- `qwen_lut_model/qwen2.5_0.5b.json`
- `qwen_lut_model/v80.json`
- `qwen_lut_model/vhk158.json`
- `qwen_lut_model/fpga_resource_config.json`
- `qwen_lut_model/u280.json`
- `qwen_lut_model/vpk180.json`
- `qwen_lut_model/qwen3_1.7b.json`
- `qwen_lut_model/param_settings/setting_1.json`
- `qwen_lut_model/param_settings/setting_8.json`
- `qwen_lut_model/param_settings/setting_9.json`
- `qwen_lut_model/param_settings/setting_5.json`
- `qwen_lut_model/param_settings/setting_w_vq.json`
- `qwen_lut_model/param_settings/setting_3.json`
- `qwen_lut_model/param_settings/setting_6.json`
- `qwen_lut_model/param_settings/setting_7.json`
- `qwen_lut_model/param_settings/setting_4.json`
- `qwen_lut_model/param_settings/setting_2.json`
- `qwen_lut_model/param_settings/setting_11.json`
- `qwen_lut_model/param_settings/setting_12.json`
- `qwen_lut_model/param_settings/setting_10.json`
- `rapidstream_script/v80_device.json`
- `rapidstream_script/floorplan_config.json`
- `rapidstream_script/pipeline_config.json`
- `custom_design/run.tcl`
- `custom_design/arm_bd_wide.tcl`
- `custom_design/constraint.tcl`
- `custom_design/arm_bd.tcl`

### Core Source Refs
- `create_bd_design_final.tcl`: 2647 lines
- `qwen_block/qwen_block_decode_tb.cpp`: 1820 lines
- `imm/imm.h`: 1718 lines
- `qwen_block/qwen_block_tb.cpp`: 1672 lines
- `ffn/ffn_tb.cpp`: 1267 lines
- `qwen_lut_model/model.py`: 1207 lines
- `attention_block/attention_block_tb.cpp`: 900 lines
- `ccu/ccu_fp32.h`: 683 lines
- `qwen_block/qwen_block.h`: 635 lines
- `gqa/gqa.h`: 597 lines

### Hardware-Oriented Source Refs
- `create_bd_design_final.tcl`
- `qwen_block/qwen_block_decode_tb.cpp`
- `qwen_block/e2e_latency.cpp`
- `qwen_block/qwen_block_tb.cpp`
- `qwen_block/qwen_block.h`
- `rope/rope_tb.cpp`
- `rope/rope.h`
- `ffn/ffn_tb.cpp`
- `ffn/ffn.h`
- `config/config.h`
- `rms_norm/rms_norm.h`
- `rms_norm/rms_norm_tb.cpp`
- `attention_block/attention_block_tb.cpp`
- `attention_block/attention_block.h`
- `gqa/gqa_tb.cpp`
- `gqa/gqa.h`
- `imm/imm_tb.cpp`
- `imm/imm.h`
- `lut-dla/lut_dla.h`
- `lut-dla/lut_dla_tb.cpp`
- `ccu/ccu_fp32_tb.cpp`
- `ccu/ccu_fp32.h`
- `silu/silu_tb.cpp`
- `silu/silu.h`
- `custom_design/run.tcl`
- `custom_design/arm_bd_wide.tcl`
- `custom_design/constraint.tcl`
- `custom_design/arm_bd.tcl`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/README.md`
- # LUT-LLM: Efficient Language Model Inference with Memory-based Computations on FPGAs
- [![DOI](https://zenodo.org/badge/1168934775.svg)](https://doi.org/10.5281/zenodo.18809342)
- <p align="center">
- </p>
- <p align="center">
- </p>
- **LUT-LLM** is the first FPGA accelerator that deploy 1B+ language model with memory-based computation, leveraging vector quantization. LUT-LLM features:
- - **Activation-weight Co-quantization**: shrinked lookup tables with comparable accuracy compared with standard scalar quantization schemes.
- - **Bandwidth-aware Parallel Centroid Search**: tradeoffs between resource consumption for parallel search and latency of pipeline propagation during decoding.
- - **Efficient 2D table lookup**: extract rows and then copy to reduce fanout with low on-chip capacity required per operation at runtime.
- - **Temporal-Spatial Hybrid Execution**: LUT-LLM sequentially execute between LUTLinear and other engines, and keep dataflow inside each engine.
- ---
- ## Artifact Evaluation
- 0. Install [TAPA](https://drive.google.com/file/d/1-GJDFHiaIDOldNGgtdgDSRxlDvRoBzSt/view?usp=drive_link): Download and untar this folder into your home and add the `PATH` variable in your `~/.bashrc`
- ```bash
- tar -xf tapa.tar
- export PATH="$PATH:$HOME/.rapidstream-tapa/usr/bin"
- ```
- If you got `tapa.tar.gz`:
- ```bash

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: activation-weight co-quantization과 vector quantization으로 arithmetic을 table lookup으로 바꾼다.
- 알고리즘 축: parallel centroid search, 2D table lookup, spatial-temporal hybrid execution을 사용한다.
- 하드웨어 축: on-chip memory 기반 lookup engine과 data caching 최소화 구조가 핵심이다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/README.md:17:- **Bandwidth-aware Parallel Centroid Search**: tradeoffs between resource consumption for parallel search and latency of pipeline propagation during decoding.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/README.md:19:- **Temporal-Spatial Hybrid Execution**: LUT-LLM sequentially execute between LUTLinear and other engines, and keep dataflow inside each engine.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/silu/silu.h:24:        #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/silu/silu.h:45:        #pragma HLS pipeline II=1 style=stp`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/silu/silu.h:76:            #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/silu/silu.h:80:                #pragma HLS unroll`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:31:    #pragma HLS array_partition variable=diff complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:35:        #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:36:        #pragma HLS loop_tripcount min=1 max=128`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:43:            #pragma HLS unroll`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:45:            #pragma HLS bind_op variable=diff_real op=fsub impl=primitivedsp`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:55:                #pragma HLS unroll`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:84:        #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:85:        #pragma HLS loop_tripcount min=1 max=128`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:115:        #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:116:        #pragma HLS loop_tripcount min=1 max=128`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:138:        #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:141:            #pragma HLS unroll`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:144:                #pragma HLS unroll`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/ccu/ccu_fp32.h:161:            #pragma HLS pipeline II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/README.md:99:- `gqa`: the grouped-query attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:214:    tapa::ostream<tapa::vec_t<float, 16>>& pre_softmax_fifo`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:216:    // compute grouped query attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:331:                    tapa::vec_t<float, 16> qk_pre_softmax;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:336:                        qk_pre_softmax[k] = (qk_reg_row[k][0] + qk_reg_row[k][8] + qk_reg_row[k][16] + qk_reg_row[k][24]);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:339:                    pre_softmax_fifo.write(qk_pre_softmax);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:351:    tapa::istream<tapa::vec_t<float, 16>>& post_softmax_fifo,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:401:            // step 3: write batch of rows for softmax and compute AV`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:431:                            qk_vec = post_softmax_fifo.read();`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:482:void softmax(`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:485:    tapa::istream<tapa::vec_t<float, 16>>& pre_softmax_fifo,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:486:    tapa::ostream<tapa::vec_t<float, 16>>& post_softmax_fifo`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:501:                float softmax_buf[MAX_KV_LEN];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:502:                #pragma HLS array_partition variable=softmax_buf cyclic factor=16 dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/LUT-LLM/gqa/gqa.h:507:                    tapa::vec_t<float, 16> pre_softmax_vec = pre_softmax_fifo.read();`

## 5. HGTXR 적용 해석
- 현재 사용자가 LUT 과사용을 줄이길 원하므로 default가 아니라 negative-control이다. URAM lookup 기반 특수 MLP만 먼 후순위 검토.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation
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
