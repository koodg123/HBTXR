---
source_type: codebase
source_name: Edge-MoE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/Edge-MoE/analysis.md -->

# Edge-MoE Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE`
- repo_remote: `origin	https://github.com/sharc-lab/Edge-MoE (fetch)`
- category: `multi-task MoE-ViT FPGA accelerator`
- HGTXR relevance: `high`
- matched_paper: `Edge-MoE`
- paper_title: Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/Edge-MoE.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `196` |
| 주요 언어 | `.bin:161, C++ header:14, C++:9, no_ext:3, Tcl:2, .mp4:1, Markdown:1, .svg:1, Python:1, .hwh:1` |
| LOC 추정 | `C++:2692, C++ header:461, Python:152, Tcl:45, Markdown:16` |

### Directory Map
- `bitstream/` (3 entries)
- `images/` (1 entries)
- `include/` (14 entries)
- `src/` (8 entries)
- `testbench/` (1 entries)
- `weights/` (163 entries)
- `weights/onboard/` (1 entries)
- `weights/scripts/` (1 entries)

### Metadata / Config Refs
- `vitis_hls.tcl`
- `vivado.tcl`
- `README.md`

### Core Source Refs
- `src/linear.cpp`: 530 lines
- `src/attention.cpp`: 494 lines
- `testbench/e2e.cpp`: 439 lines
- `src/moe.cpp`: 436 lines
- `src/gelu.cpp`: 227 lines
- `src/ViT_compute.cpp`: 208 lines
- `src/layernorm.cpp`: 172 lines
- `src/conv.cpp`: 166 lines
- `weights/scripts/prepare_for_onboard.py`: 152 lines
- `include/util.hpp`: 99 lines

### Hardware-Oriented Source Refs
- `vitis_hls.tcl`
- `vivado.tcl`
- `testbench/e2e.cpp`
- `src/gelu.cpp`
- `src/linear.cpp`
- `src/ViT_compute.cpp`
- `src/conv.cpp`
- `src/layernorm.cpp`
- `src/add.cpp`
- `src/moe.cpp`
- `src/attention.cpp`
- `include/moe.hpp`
- `include/gelu.hpp`
- `include/datatypes.hpp`
- `include/conv.hpp`
- `include/attention.hpp`
- `include/model.hpp`
- `include/hardware.hpp`
- `include/kernel.hpp`
- `include/util.hpp`
- `include/add.hpp`
- `include/layernorm.hpp`
- `include/tbutil.hpp`
- `include/dcl.hpp`
- `include/linear.hpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/README.md`
- # Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts
- Rishov Sarkar<sup>1</sup>, Hanxue Liang<sup>2</sup>, Zhiwen Fan<sup>2</sup>, Zhangyang Wang<sup>2</sup>, Cong Hao<sup>1</sup>
- <sup>1</sup>School of Electrical and Computer Engineering, Georgia Institute of Technology
- <sup>2</sup>School of Electrical and Computer Engineering, University of Texas at Austin
- ICCAD 2023 [paper](https://arxiv.org/abs/2305.18691)
- ## Overview
- This is **Edge-MoE**, the *first end-to-end* FPGA accelerator for *multi-task ViT* with a rich collection of architectural innovations.

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: task-level sparsity와 multi-task MoE를 FPGA accelerator에 맞춰 reordering, single-pass softmax, low-cost GELU, unified compute unit으로 구현한다.
- 알고리즘 축: task에 따라 sparse expert pathway만 실행하고, attention reordering과 patch reordering으로 memory access를 줄인다.
- 하드웨어 축: unified flexible computing unit이 대부분 layer를 공유하며 softmax/GELU approximation이 control/datapath를 단순화한다.

### 핵심 모듈 근거
- `include/model.hpp:4-20`은 입력 크기, patch 크기, `FEATURE_DIM=192`, `VIT_HIDDEN_DIM=768`, `NUM_LAYERS=12`, `NUM_HEADS=3`, `NUM_EXPERTS=16`, `NUM_SELECTED_EXPERTS=2`를 고정한다. HGTXR 적용 시 동적 expert 수를 늘리는 방식보다 search/track별 정적 후보를 제한하는 방식이 안전하다.
- `src/ViT_compute.cpp:6-66`의 `load_one_time_weights()`는 patch embedding weight/bias를 on-chip static buffer에 캐시하고 `attn_scale=0.125`, `norm_eps=1e-6` 같은 상수를 고정한다. 이는 HGTXR에서 작은 고정 테이블/상수는 LUTRAM 또는 FF로 두고 큰 frame/QKV/hidden buffer만 URAM으로 보내는 분리 정책과 맞다.
- `src/ViT_compute.cpp:68-208`의 top 함수는 `m_axi` bundle을 `inout1..4`, `weights`로 분리하고 layer loop 안에서 norm, Q/K/V/proj, QK, softmax, attn*V, residual, MLP/MoE를 순서대로 호출한다. HGTXR에는 "mode-specific routing" 실험으로만 가져오고 C3b 기본 dense datapath를 즉시 교체하지 않는다.
- 같은 top 함수는 even layer에는 dense MLP, odd layer에는 `compute_moe()`를 배치한다. search/track 분기 실험은 이처럼 layer 단위로 정적 schedule화해야 HLS control과 검증 비용을 제한할 수 있다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:12:    #pragma HLS inline off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:18:            #pragma HLS pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:27:    #pragma HLS inline off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:35:                #pragma HLS pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:51:    #pragma HLS inline off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:54:    #pragma HLS array_partition variable=q_blocks complete dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:57:    #pragma HLS array_partition variable=attn_blocks complete dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:58:    #pragma HLS array_partition variable=attn_blocks complete dim=2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:68:                #pragma HLS pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:82:                    #pragma HLS occurrence cycle=dim_iters`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:110:                    #pragma HLS occurrence cycle=dim_iters`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:133:    #pragma HLS inline off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:136:    #pragma HLS array_partition variable=softmax_sums complete dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:137:    #pragma HLS array_partition variable=softmax_sums complete dim=2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:139:    #pragma HLS array_partition variable=softmax_biases complete dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:140:    #pragma HLS array_partition variable=softmax_biases complete dim=2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:143:    #pragma HLS array_partition variable=attn_blocks complete dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:151:                #pragma HLS pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:152:                #pragma HLS dependence variable=softmax_sums inter true distance=q_patch_unadjusted_step`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/src/attention.cpp:153:                #pragma HLS dependence variable=softmax_biases inter true distance=q_patch_unadjusted_step`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/linear.hpp:41:    bool use_gelu,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/layernorm.hpp:1:#ifndef __LAYERNORM_HPP__`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/layernorm.hpp:2:#define __LAYERNORM_HPP__`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/layernorm.hpp:6:enum LayerNorm {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/kernel.hpp:9:#include "attention.hpp"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/kernel.hpp:10:#include "layernorm.hpp"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/kernel.hpp:25:        softmax_info_t attn_softmax_info,`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/hardware.hpp:15:constexpr unsigned int ATTN_MATMUL_PARALLEL = 4;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/hardware.hpp:25:typedef hls::vector<heads_t, ATTN_MATMUL_PARALLEL> attn_parallel_t;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/hardware.hpp:26:typedef heads_t qxk_out_t[ceildiv(NUM_PATCHES + ATTN_MATMUL_PARALLEL - 1, ATTN_MATMUL_PARALLEL)][NUM_PATCHES][ATTN_MATMUL_PARALLEL];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/hardware.hpp:27:typedef hls::vector<fm_t, roundup_p2(NUM_HEADS * 2)> softmax_info_row_t;`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/hardware.hpp:28:typedef softmax_info_row_t softmax_info_t[NUM_PATCHES];`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/attention.hpp:1:#ifndef __ATTENTION_HPP__`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/attention.hpp:2:#define __ATTENTION_HPP__`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Edge-MoE/include/attention.hpp:7:enum AttentionLinear {`

## 5. HGTXR 적용 해석
- HGTXR search/track mode가 task split이므로 P1 실험으로 강하게 적합하다. 정확도 개선 가능성이 있으나 C3b 기본 signoff와 분리해야 한다.

### 적용 가능 모듈
- Search/Track mode-specific expert or head routing
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
