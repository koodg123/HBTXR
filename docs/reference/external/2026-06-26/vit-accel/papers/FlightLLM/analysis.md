---
source_type: paper
source_name: FlightLLM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/FlightLLM/analysis.md -->

# FlightLLM Detailed Paper Analysis

- title: FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs
- url: https://arxiv.org/abs/2401.03868
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/FlightLLM.pdf`
- related_codebases: `flightllm_test_demo`

## 1. 기존 방법의 문제점
- 요약: compressed LLM은 GPU/일반 accelerator에서 sparse compute, bandwidth, compile overhead 병목이 있다.

### Paper Evidence: Introduction / Motivation
```text
1   INTRODUCTION
                                        For all other uses, contact the owner/author(s).                                            Recently, we have witnessed the rapid development and significant
                                        FPGA ’24, March 3–5, 2024, Monterey, CA, USA                                                impact of Large Language Models (LLMs) [5, 48]. LLMs demonstrate
                                        © 2024 Copyright held by the owner/author(s).
                                        ACM ISBN 979-8-4007-0418-5/24/03.                                                           amazing power to understand all the users’ input requests (prefill
                                        https://doi.org/10.1145/3626202.3637562                                                     stage) and generate accurate responses token-by-token (decode

FPGA ’24, March 3–5, 2024, Monterey, CA, USA                                                                                                              Shulin Zeng and Jun Liu, et al.


                                        GPU           Cost          Performance                              Challenges for LLMs                 Solutions in FlightLLM
                                         V100       Price Power
                                        (12 nm)   (~$12K) (~220W)
                                                                          ~45                                                                          d c     b      a
                     Deploy with opt.                                    token/s                  1. Heavy Computation
                                                                                                     Compress model with
                     (vLLM, INT8, …)                                 1x throughput/price             various sparse pattern         MAC      MAC                      MAC        MAC
      Large                                                          1x throughput/power
```

## 2. 제안하는 방법
- 요약: configurable sparse DSP chain, always-on-chip decode, length-adaptive compilation을 제공한다.

### Paper Evidence: Method Section
```text
algorithm accuracy loss of LLMs [19]. In contrast, the unstructured                          underutilized memory bandwidth, FlightLLM proposes an always-
sparsity ensuring algorithm accuracy cannot bring end-to-end ac-                             on-chip decode scheme. Activations reside in the on-chip memory
celeration for LLMs. For example, the 75% unstructured sparsity                              during the decode stage with the support of mixed-precision quan-
only leads to negligible end-to-end speedup [18]. From the mem-                              tization. To reduce the compilation overhead, FlightLLM proposes
ory perspective, quantization and large on-chip memory can re-                               a length adaptive compilation method. Instructions for consecutive
duce data access. Recent algorithm studies [14, 28] are pushing                              input token length are grouped, and the total storage overhead for
the limit of bit-width with mixed-precision quantization. How-                               instructions can be reduced.
ever, the alignment feature of GPU’s cache and SIMD architecture                                The main contributions of this paper are as follows.
requires homogeneous bit-widths of LLM parameters for weight
access reduction [49]. Compounding the issue, GPU’s KB-scaled                                     • We propose a configurable sparse DSP chain to support differ-
share memory of SMs cannot hold all the activations for LLM text                                    ent sparsity patterns. FlightLLM improves the computation
generation.                                                                                         efficiency by 1.6× with block-wise and N:M sparsity.
   FPGAs are potential solutions to accelerate LLM inference and                                  • We propose an always-on-chip decode scheme with mixed-
explore the benefits brought by model compression, which has                                        precision support. FlightLLM boosts the memory bandwidth
been proven in previous deep learning models [21, 22, 39, 43, 55].                                  from 35.6% to 65.9%.
However, efficient LLM inference on FPGAs needs to solve the                                      • We propose a length adaptive compilation method to reduce
following challenges (Fig. 2):                                                                      the instruction storage overhead by 500× (∼GB), enabling
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: sparsity pattern별 DSP chain mapping과 sequence length별 compilation strategy를 사용한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `bit-width`:
```text
tage with the support of mixed-precision quan- only leads to negligible end-to-end speedup [18]. From the mem- tization. To reduce the compilation overhead, FlightLLM proposes ory perspective, quantization and large on-chip memory can re- a length adaptive compilation method. Instructions for consecutive duce data access. Recent algorithm studies [14, 28] are pushing input token length are grouped, and the total storage overhead for the limit of bit-width with mixed-precision quantization. How- instructions can be reduced. ever, the alignment feature of GPU’s cache and SIMD architecture The main contributions of this paper are as follows. requires homogeneous bit-widths of LLM parameters for weight access reduction [49]. Compounding the issue, GPU’s KB-scaled • We propose a configurable sparse DSP chain to support differ- share memory of SMs cannot hold all the activations for LLM text e
```
- keyword `sparsity`:
```text
address the above issues. (e.g., DSP48 and heterogeneous memory hierarchy). To address However, the unique computation schemes of these methods are the challenges of low computation efficiency, FlightLLM exploits not efficiently supported by current hardware platforms, like GPUs, a configurable sparse DSP chain. We introduce a flexible cascaded for LLMs. From the computation perspective, current GPUs only DSP48 architecture to support different sparsity patterns with high support structured sparsity (e.g., 2:4 sparsity), leading to significant computation efficiency (i.e., runtime DSP utilization). To tackle the algorithm accuracy loss of LLMs [19]. In contrast, the unstructured underutilized memory bandwidth, FlightLLM proposes an always- sparsity ensuring algorithm accuracy cannot bring end-to-end ac- on-chip decode scheme. Activations reside in the on-chip memory celeration for LLMs.
```
- keyword `parallelism`:
```text
pt and generates the duce the inference overhead. As shown in Fig. 4, the overall hard- first response token (e.g. the "Alex" in Figure.3 (a)). All the 𝑁 input ware architecture of FlightLLM mainly includes a task scheduler, tokens are processed simultaneously with high throughput. In the memory controller, and multiple computing cores (short as cores). decode stage, LLM treats the newly generated token as length 𝑁 = 1 The accelerator uses model parallelism on multiple cores to com- input and generates the next token (e.g. the "won" in Figure.3 (b)). plete the LLM inference task. The task scheduler assigns tasks to Since LLM only processes one new token at a time in the decode different cores and controls data synchronization. stage, the matrix-matrix multiplications in equation 1 and 2 become The components of each core include the unified Matrix Pro- matrix-vector multiplications. The
```

## 4. 하드웨어 아키텍처
- 요약: U280/VHK158 FPGA 대상 DSP48와 heterogeneous memory hierarchy를 적극 활용한다.

### Paper Evidence: Architecture / Implementation
```text
architectures. To maximize the benefits of sparsification, we design                               in DSP48. DSP cascading makes the most use of the accumulator,
the unified Matrix Processing Engine (MPE) to handle all operations                                the result carry-out port, and the result carry-in port, improving the
related to matrix computation, including General Matrix Multiplica-                                hard-core utilization. However, the fully cascaded DSP architecture
tion (GEMM), Sparse Matrix-Matrix multiplication (SpMM), General                                   are not friendly to sparse computation since the cascaded chain is a
Matrix-Vector multiplication (GEMV), Sparse Matrix-Vector multi-                                   fixed path. In FlightLLM, we propose a configurable sparse DSP
plication (SpMV), and Sampled-Dense-Dense Matrix Multiplication                                    chain (CSD-Chain) to supplement the fixed DSP chain. In the
(SDDMM). As shown in Fig. 5(a), the MPE includes multiple Matrix                                   CSD-Chain, a long DSP chain is divided into several DSP groups.
Processing Units (MPUs), which transfer weights from the weight                                    A DSP group (DG) has several DSP48 cores, that are cascaded in
buffer using the streaming approach. The activation buffer and the                                 a fixed manner. We pack two INT8 MACs on DSP48 [1]. Differ-
global buffer store the input and output activations of the MPE,                                   ent DGs are cascaded with a configurable path. A VPU is made
respectively. By configuring the MPU, the MPE can support both                                     up of a CSD-Chain and a MPU consists of several VPUs. Fig. 5(d)
matrix-matrix multiplication (MM) (Fig. 5(b)) and matrix-vector                                    shows the architecture of the CSD-Chain based VPU. Each DG has
multiplication (MV) mode (Fig. 5(c)). The MPU is composed of multi-                                two DSP48 cores. We use configurable cascading to support sparse
ple vector processing units (VPUs). The VPU is the basic component                                 matrix operations by adding three units to the fixed DSP chain.
in the MPE, which performs the dot product of two vectors.                                            Sparse Mux. As shown in Fig. 5(d), two activations (A and B) are
   We build the unified MPE to support all the five operator on the                                delivered to one DSP48 core simultaneously for wei
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs Shulin Zeng∗ Jun Liu∗ Guohao Dai† Tsinghua University Shanghai Jiao Tong University Shanghai Jiao Tong University Infinigence-AI Infinigence-AI Infinigence-AI Xinhao Yang Tianyu Fu Hongyi Wang Tsinghua University Tsinghua University Tsinghua University Infinigence-AI Infinigence-AI Infinigence-AI arXiv:2401.03868v2 [cs.AR] 9 Jan 2024 Wenheng Ma Hanbo Sun Shiyao Li Tsinghua University Tsinghua University Tsinghua University Infinigence-AI Zi
```
- keyword `U280`:
```text
hua University Infinigence-AI Infinigence-AI Zehao Wang Ruoyu Zhang Kairui Wen Infinigence-AI Infinigence-AI Infinigence-AI Xuefei Ning Yu Wang† Tsinghua University Tsinghua University ABSTRACT for real-world LLMs, we propose a length adaptive compilation Transformer-based Large Language Models (LLMs) have made method to reduce the compilation overhead. Implemented on the a significant impact on various domains. However, LLMs’ effi- Xilinx Alveo U280 FPGA, FlightLLM achieves 6.0× higher energy ciency suffers from both heavy computation and memory over- efficiency and 1.8× better cost efficiency against commercial GPUs heads. Compression techniques like sparsification and quantization (e.g., NVIDIA V100S) on modern LLMs (e.g., LLaMA2-7B) using are commonly used to mitigate the gap between LLM’s computa- vLLM and SmoothQuant under the batch size of one. FlightLLM tion/memory overheads and
```
- keyword `Vitis`:
```text
the data improved compared to U280, reaching 32GB and 819GB/s. We im- stored in the HBM will be partitioned into appropriate channels plement FlightLLM on the real system with U280 FPGAs (Fig. 10(a)). to prevent inefficient access across different channels, thus lever- For VHK158 evaluation, we implement a cycle-accurate simulator, aging the FPGA’s high bandwidth effectively. Lastly, the compiler which has been verified with RTL emulation using Vitis 2023.1. will generate instructions using the optimized IR and schedule the FPGA Implementation. Fig. 10 shows the layout of our imple- on-chip buffer according to manually defined templates. mentation on U280 FPGA. Since cross-die connections are more In addition, we support generating corresponding RTL for differ- likely to become the critical paths for timing closure, we instantiate ent FPGA platforms. Specifically, the RTL Generator take
```

## 5. 실험 방법
```text
6 EVALUATION                                                              duct evaluations on the selected model with huggingface PyTorch
                                                                          as the GPU-naive design, vLLM [31], and SmoothQuant [49] as the
6.1 Evaluation Setup                                                      GPU-opt design. vLLM is the commonly used LLM framework with
Models and Datasets. We evaluate the effectiveness of FlightLLM           KV cache memory optimization, and SmoothQuant is the SOTA
with state-of-the-art large language models LLaMA2-7B [41] and            LLM quantization framework with INT8 CUDA kernel for both ac-
OPT-6.7B [54]. We finetune the compressed model with a small              tivations and weights. We use nvprof [2] to profile the GPU power
sampled subset of RedPajama dataset [11] consisting 8192 rows             consumption at runtime.
with 56M tokens. The accuracy evaluation is performed on the              SOTA Accelerator Baselines. We also selected three domain-
commonly used WikiText-103 and WikiText-2 [35] datasets.                  specific accelerators targeting at accelerating attention mechanism:

FlightLLM: Efficient Large Language Model Inference with a Complete Mapping Flow on FPGAs                                              FPGA ’24, March 3–5, 2024, Monterey, CA, USA


                             V100S-naive              V100S-opt              Ours (U280)           A100-naive            A100-opt                Ours (VHK158-est.)
                      4                            OPT-6.7B                                                                    LLaMA2-7B
         Normalized

                                                                                           2.5×                                                                    2.8×
          Latency
                      3
                      2
                      1
        (a)           0
        (b)           4                            OPT-6.7B                                                                    LLaMA2-7B
                                                                                                                                                                   2.8×
                                                                                   Accelerator
        Throughput
        Normalized




                      3                                                               2.5×
                      2
                      1
                      0
                          [128,512] [128,1024] [128,1536] [512,512] [512,1024] [512,1536] Geomean [128,512] [128,1024] [128,1536] [512,512] [512,1024] [512,1536] Geomean
Figure 11: Latency and throughput of FlightLLM and V100S/A100 GPU. The horizontal axis represents [prefill size, decode size].
Table 4: Perplexity of LLMs under different optimization                                          Table 5: The bandwid
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `LLaMA`:
```text
ge Models (LLMs) have made method to reduce the compilation overhead. Implemented on the a significant impact on various domains. However, LLMs’ effi- Xilinx Alveo U280 FPGA, FlightLLM achieves 6.0× higher energy ciency suffers from both heavy computation and memory over- efficiency and 1.8× better cost efficiency against commercial GPUs heads. Compression techniques like sparsification and quantization (e.g., NVIDIA V100S) on modern LLMs (e.g., LLaMA2-7B) using are commonly used to mitigate the gap between LLM’s computa- vLLM and SmoothQuant under the batch size of one. FlightLLM tion/memory overheads and hardware capacity. However, existing beats NVIDIA A100 GPU with 1.2× higher throughput using the GPU and transformer-based accelerators cannot efficiently process latest Versal VHK158 FPGA. compressed LLMs, due to the following unresolved challenges: low computational efficiency, under
```
- keyword `dataset`:
```text
the Xilinx Board Utility tool xbutil [3]. optimal performance on different platforms. GPU Baselines. We choose NVIDIA A100 and V100S as our GPU baselines, and their specifications are also listed in Table 2. We con- 6 EVALUATION duct evaluations on the selected model with huggingface PyTorch as the GPU-naive design, vLLM [31], and SmoothQuant [49] as the 6.1 Evaluation Setup GPU-opt design. vLLM is the commonly used LLM framework with Models and Datasets. We evaluate the effectiveness of FlightLLM KV cache memory optimization, and SmoothQuant is the SOTA with state-of-the-art large language models LLaMA2-7B [41] and LLM quantization framework with INT8 CUDA kernel for both ac- OPT-6.7B [54]. We finetune the compressed model with a small tivations and weights. We use nvprof [2] to profile the GPU power sampled subset of RedPajama dataset [11] consisting 8192 rows consumption at runtime. w
```

## 7. 실험 결과
- keyword `speedup`:
```text
lization). To tackle the algorithm accuracy loss of LLMs [19]. In contrast, the unstructured underutilized memory bandwidth, FlightLLM proposes an always- sparsity ensuring algorithm accuracy cannot bring end-to-end ac- on-chip decode scheme. Activations reside in the on-chip memory celeration for LLMs. For example, the 75% unstructured sparsity during the decode stage with the support of mixed-precision quan- only leads to negligible end-to-end speedup [18]. From the mem- tization. To reduce the compilation overhead, FlightLLM proposes ory perspective, quantization and large on-chip memory can re- a length adaptive compilation method. Instructions for consecutive duce data access. Recent algorithm studies [14, 28] are pushing input token length are grouped, and the total storage overhead for the limit of bit-width with mixed-precision quantization. How- instructions can be reduced. ever
```
- keyword `energy`:
```text
AI Zehao Wang Ruoyu Zhang Kairui Wen Infinigence-AI Infinigence-AI Infinigence-AI Xuefei Ning Yu Wang† Tsinghua University Tsinghua University ABSTRACT for real-world LLMs, we propose a length adaptive compilation Transformer-based Large Language Models (LLMs) have made method to reduce the compilation overhead. Implemented on the a significant impact on various domains. However, LLMs’ effi- Xilinx Alveo U280 FPGA, FlightLLM achieves 6.0× higher energy ciency suffers from both heavy computation and memory over- efficiency and 1.8× better cost efficiency against commercial GPUs heads. Compression techniques like sparsification and quantization (e.g., NVIDIA V100S) on modern LLMs (e.g., LLaMA2-7B) using are commonly used to mitigate the gap between LLM’s computa- vLLM and SmoothQuant under the batch size of one. FlightLLM tion/memory overheads and hardware capacity. However, existing beats
```
- keyword `accuracy`:
```text
n efficiency, FlightLLM exploits not efficiently supported by current hardware platforms, like GPUs, a configurable sparse DSP chain. We introduce a flexible cascaded for LLMs. From the computation perspective, current GPUs only DSP48 architecture to support different sparsity patterns with high support structured sparsity (e.g., 2:4 sparsity), leading to significant computation efficiency (i.e., runtime DSP utilization). To tackle the algorithm accuracy loss of LLMs [19]. In contrast, the unstructured underutilized memory bandwidth, FlightLLM proposes an always- sparsity ensuring algorithm accuracy cannot bring end-to-end ac- on-chip decode scheme. Activations reside in the on-chip memory celeration for LLMs. For example, the 75% unstructured sparsity during the decode stage with the support of mixed-precision quan- only leads to negligible end-to-end speedup [18]. From the mem- tizatio
```
- keyword `throughput`:
```text
avy computation and memory over- efficiency and 1.8× better cost efficiency against commercial GPUs heads. Compression techniques like sparsification and quantization (e.g., NVIDIA V100S) on modern LLMs (e.g., LLaMA2-7B) using are commonly used to mitigate the gap between LLM’s computa- vLLM and SmoothQuant under the batch size of one. FlightLLM tion/memory overheads and hardware capacity. However, existing beats NVIDIA A100 GPU with 1.2× higher throughput using the GPU and transformer-based accelerators cannot efficiently process latest Versal VHK158 FPGA. compressed LLMs, due to the following unresolved challenges: low computational efficiency, underutilized memory bandwidth, and CCS CONCEPTS large compilation overheads. • Hardware → Hardware accelerators; • Computer systems This paper proposes FlightLLM, enabling efficient LLMs infer- organization → Reconfigurable computing. ence with
```
- keyword `latency`:
```text
1.8x throughput/price 6.0x throughput/power 2. Heavy Memory Access On-chip buffer Off-chip memory Repetitive activation write to off-chip memory Figure 1: FlightLLM on Alveo U280 FPGA outperforms Bandwidth NVIDIA V100S GPU (using vLLM [31] and SmoothQuant [49]) Always-on-chip decode 1.9x↑ & mixed-precision support 8b 8b 8b 8b 8b 8b 4b 4b 4b 4b 8b 8b with better performance and cost efficiency. Inference case stage). LLMs are being widely used in latency-sensitive scenar- 3. Dynamic Input Length HW. info ios [36], such as code completion [42], real-time chatbots [7, 40], Each instruction file for Perf. each input length customer support [26], online legal advice [12], and beyond. The latency is critical for a good user experience, and the batch size ~TB → GB Length adaptive is usually set as 1 to meet the real-time requirement. However, compilation Large Small space! space! current LLMs s
```

## 8. 실험 옵션 / Ablation 축
- URAM/BRAM/LUTRAM binding
- memory banks 8/16/32
- parallelism factor 8/16/32
- FIFO depth and double buffering

## 9. HGTXR 적용 판단
- DSP chain scheduling과 compile partitioning 참고용. ViT/eye tracking 직접성은 낮다.

### 우선순위 판정
- recommended_priority: `P3`
- C3b board-smoke gate는 유지한다.
- HLS/Vivado 증거가 생기기 전에는 resource matrix 완료 variant로 승격하지 않는다.

## 10. HGTXR 실험 설계로 변환
| 단계 | 작업 | 산출물 | 승격 조건 |
|---|---|---|---|
| SW | 알고리즘을 Python/C++ reference에 반영 | accuracy/bit-exact report | 정확도 개선 또는 무손실 |
| HLS | parameterized macro/defines로 구현 | csim/csynth report | csynth.xml 생성 및 resource gate 통과 |
| Vivado | C3b successor overlay로 구현 | routed timing/power/util | WNS >= 0, resource 정책 유지 |
| PYNQ | smoke run | board JSON | validator pass |
