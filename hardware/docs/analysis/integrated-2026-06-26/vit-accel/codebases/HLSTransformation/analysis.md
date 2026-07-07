---
source_type: codebase
source_name: HLSTransformation
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/HLSTransformation/analysis.md -->

# HLSTransformation Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation`
- repo_remote: `not_available`
- category: `AWS/Vitis transformer system flow`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `12517` |
| 주요 언어 | `.txt:12437, no_ext:19, Python:15, Markdown:10, C:9, C/C++ header:7, .prj:3, .model:2, C++:2, .bash:2` |
| LOC 추정 | `.txt:66278, C:8307, Python:2224, C++:1454, Markdown:488, C/C++ header:419, Notebook:130, Shell:70` |

### Directory Map
- `cpu_benchmarks/` (35 entries)
- `cpu_benchmarks/assets/` (1 entries)
- `cpu_benchmarks/benchmark_stories/` (12432 entries)
- `cpu_benchmarks/doc/` (2 entries)
- `gpu_benchmarks/` (23 entries)
- `gpu_benchmarks/llama/` (1 entries)
- `llama_xrt/` (6 entries)
- `llama_xrt/.settings/` (1 entries)
- `llama_xrt/src/` (5 entries)
- `llama_xrt_kernels/` (4 entries)
- `llama_xrt_kernels/src/` (4 entries)
- `llama_xrt_system/` (4 entries)
- `llama_xrt_system_hw_link/` (3 entries)

### Metadata / Config Refs
- `README.md`
- `gpu_benchmarks/README.md`
- `cpu_benchmarks/README.md`

### Core Source Refs
- `cpu_benchmarks/runq_ppl.c`: 1397 lines
- `cpu_benchmarks/runq_latency_1024.c`: 1377 lines
- `cpu_benchmarks/runq_power_consumption.c`: 1102 lines
- `cpu_benchmarks/runq_latency_256.c`: 1096 lines
- `cpu_benchmarks/runq.c`: 1092 lines
- `cpu_benchmarks/run_ppl.c`: 1006 lines
- `cpu_benchmarks/run.c`: 973 lines
- `llama_xrt/src/llama2.cpp`: 967 lines
- `cpu_benchmarks/export.py`: 567 lines
- `llama_xrt_kernels/src/forward.cpp`: 487 lines

### Hardware-Oriented Source Refs
- `llama_xrt_kernels/src/typedefs.h`
- `llama_xrt_kernels/src/forward.h`
- `llama_xrt_kernels/src/forward.cpp`
- `llama_xrt_kernels/src/config.h`
- `cpu_benchmarks/runq.c`
- `cpu_benchmarks/runq_power_consumption.c`
- `cpu_benchmarks/runq_ppl.c`
- `cpu_benchmarks/win.h`
- `cpu_benchmarks/test.c`
- `cpu_benchmarks/run.c`
- `cpu_benchmarks/run_ppl.c`
- `cpu_benchmarks/runq_latency_1024.c`
- `cpu_benchmarks/runq_latency_256.c`
- `cpu_benchmarks/win.c`
- `llama_xrt/src/typedefs.h`
- `llama_xrt/src/forward.h`
- `llama_xrt/src/config.h`
- `llama_xrt/src/llama2.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/README.md`
- # HLS Implementation of Llama 2
- ## Prerequisites
- 1. EC2 instance (z1d.2xlarge recommended)
- 3. AWS FPGA Developer AMI [(install)](https://github.com/aws/aws-fpga/blob/master/Vitis/README.md)
- 4. S3 Bucket
- 5. Pretrained model parameters & tokenizer can be found [here](https://drive.google.com/drive/folders/13awUjl2nhhBiRMesXR6O9zhnMoYbeACW?usp=sharing)
- ## Build Instructions
- 1. Follow the setup for AWS FPGA Developer AMI
- 2. Open the project in Vitis IDE
- 3. Verify the install by building the Software Emulation
- 4. If no run configurations exist, add a new System Project Debug configuration with the following user provided arguments
- `${project_loc:llama_xrt}/src/weights.bin -z ${project_loc:llama_xrt}/src/tokenizer.bin -t 0.8 -n 256 -i "{prompt}" -k`
- ## Run Instructions
- 1. Run the Hardware build, should take around ~12 hours.
- 2. Extract the .xclbin file and export to an AWS AMI following the directions [here](https://github.com/aws/aws-fpga/blob/master/Vitis/README.md#2-create-an-amazon-fpga-image-afi)
- 3. Launch an EC2 F1 instance and copy the generated .awsxclbin file to the same directory as the host code
- 4. Start the FPGA runtime (more instructions [here](https://github.com/aws/aws-fpga/blob/master/Vitis/README.md#2-create-an-amazon-fpga-image-afi))
- `cd $AWS_FPGA_REPO_DIR`
- `source vitis_setup.sh`
- `source vitis_runtime_setup.sh`

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:21:#pragma HLS array_partition variable = x_buff type = cyclic factor = 128`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:22:#pragma HLS array_partition variable = weight_buff type = cyclic factor = 64`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:23:#pragma HLS array_partition variable = out_buff type = cyclic factor = 64`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:30:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:31:#pragma HLS UNROLL factor = 128 skip_exit_check`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:42:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:43:#pragma HLS UNROLL factor = 64`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:60:#pragma HLS loop_tripcount min = 0 max = 257 avg = 129`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:61:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:73:#pragma HLS loop_tripcount min = 0 max = 257 avg = 129`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:74:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:75:#pragma HLS UNROLL factor = 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:83:#pragma HLS loop_tripcount min = 0 max = 257 avg = 129`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:91:#pragma HLS loop_tripcount min = 0 max = 257 avg = 129`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:92:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:93:#pragma HLS UNROLL factor = 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:117:#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:118:#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:119:#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 128`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt_kernels/src/forward.cpp:120:#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt/src/llama2.cpp:54:void softmax(float *x, int size)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt/src/llama2.cpp:649:    // apply softmax to the logits to get the probabilities for next token`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/llama_xrt/src/llama2.cpp:650:    softmax(logits, sampler->vocab_size);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/sample.py:31:torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:47:    // weights for matmuls. note dim == n_heads * head_size`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:74:    float *att; // buffer for scores/attention values (n_heads, seq_len)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:297:void softmax(float* x, int size) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:317:void matmul(float* xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:330:        // do the matmul in groups of GS`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:363:        // attention rmsnorm`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:366:        // qkv matmuls for this position`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:368:        matmul(s->q, &s->xq, w->wq + l, dim, dim);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:369:        matmul(s->k, &s->xq, w->wk + l, dim, kv_dim);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:370:        matmul(s->v, &s->xq, w->wv + l, dim, kv_dim);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HLSTransformation/cpu_benchmarks/runq_latency_256.c:396:        // multihead attention. iterate over all heads`

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
