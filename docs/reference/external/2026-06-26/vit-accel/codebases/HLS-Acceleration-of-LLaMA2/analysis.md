---
source_type: codebase
source_name: HLS-Acceleration-of-LLaMA2
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/HLS-Acceleration-of-LLaMA2/analysis.md -->

# HLS-Acceleration-of-LLaMA2 Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2`
- repo_remote: `not_available`
- category: `Vitis HLS transformer/LLM demo`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `3` |
| 주요 언어 | `C++:2, Markdown:1` |
| LOC 추정 | `C++:1326, Markdown:82` |

### Directory Map

### Metadata / Config Refs
- `README.md`

### Core Source Refs
- `main.cpp`: 939 lines
- `kernel_forward.cpp`: 387 lines
- `README.md`: 82 lines

### Hardware-Oriented Source Refs
- `kernel_forward.cpp`
- `main.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/README.md`
- # HLS-Acceleration-of-LLaMA2
- This project implements a hardware-accelerated Transformer forward pass.
- It uses Vitis HLS to design computational kernels (kernel_forward.cpp) and a host program (main.cpp) to manage FPGA execution. The system is deployed on the AMD Zynq UltraScale+ MPSoC ZCU106 board.
- ## Requirements
- ### Development Environment
- - Ubuntu 22.04
- - Vitis 2024.2
- - Vivado 2024.2
- ### Execution Environment
- - ZCU106 development board
- - PetaLinux 2024.2
- ## Running on ZCU106
- 1. Copy the following files to the SD card:
- - Llama2_host (host executable)
- - binary_container_1.bin (FPGA kernel binary)
- - Model weights (e.g., stories15M.bin)
- - tokenizer.bin
- 2. Boot the ZCU106 in SD card mode.
- 3. Monitor boot via UART:
- - Use Vitis → New Feature Preview → Serial Monitor Install

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:17:	#pragma HLS ARRAY_PARTITION variable=W cyclic factor=32 dim=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:21:		#pragma HLS UNROLL factor=8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:26:    #pragma HLS ARRAY_PARTITION variable=partial complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:30:    	#pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:41:    	#pragma HLS UNROLL factor=8`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:51:    	#pragma HLS LOOP_TRIPCOUNT min=1 max=257`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:52:    	#pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:59:    #pragma HLS ARRAY_PARTITION variable=partial complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:63:    	#pragma HLS LOOP_TRIPCOUNT min=1 max=257`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:64:    	#pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:74:    	#pragma HLS LOOP_TRIPCOUNT min=1 max=257`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:75:    	#pragma HLS UNROLL factor=32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:83:        #pragma HLS loop_flatten off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:86:    	#pragma HLS ARRAY_PARTITION variable=partial complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:90:        	#pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:101:        #pragma HLS loop_flatten off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:104:    	#pragma HLS ARRAY_PARTITION variable=partial complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:108:        	#pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:119:        #pragma HLS loop_flatten off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:122:    	#pragma HLS ARRAY_PARTITION variable=partial complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:46:void softmax(float* x, int size) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:80:void matmul_dim_dim(float* o, float* x, float* w) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:98:void matmul_dim_kvdim(float* o, float* x, float* w) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:116:void matmul_dim_hiddendim(float* o, float* x, float* w) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:134:void matmul_hiddendim_dim(float* o, float* x, float* w) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:152:void matmul_dim_vocabsize(float* o, float* x, float* w) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:212:void attention(int loff, int pos, float* S_xb, float* S_q, float* S_att, float* S_key_cache, float* S_value_cache) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:213:    attention_heads:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:239:        softmax(att, pos + 1);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:253:        apply_attention:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:366:        matmul_dim_dim(S_q, S_xb, W_wq + l*P_DIM*P_DIM);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:367:        matmul_dim_kvdim(S_k, S_xb, W_wk + l*P_DIM*P_KV_DIM);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:368:        matmul_dim_kvdim(S_v, S_xb, W_wv + l*P_DIM*P_KV_DIM);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:371:        attention(loff, pos, S_xb, S_q, S_att, S_key_cache, S_value_cache);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLS-Acceleration-of-LLaMA2/kernel_forward.cpp:372:        matmul_dim_dim(S_xb2, S_xb, W_wo + l*P_DIM*P_DIM);`

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
