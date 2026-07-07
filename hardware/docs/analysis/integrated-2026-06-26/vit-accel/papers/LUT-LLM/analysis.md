---
source_type: paper
source_name: LUT-LLM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/LUT-LLM/analysis.md -->

# LUT-LLM Detailed Paper Analysis

- title: LUT-LLM: Efficient Large Language Model Inference with Memory-based Computations on FPGAs
- url: https://arxiv.org/abs/2511.06174
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/LUT-LLM.pdf`
- related_codebases: `LUT-LLM`

## 1. 기존 방법의 문제점
- 요약: FPGA의 arithmetic advantage가 GPU 최적화로 약해졌고 single-batch LLM은 memory wall이 크다.

### Paper Evidence: Abstract
```text
Abstract—The rapid development of large language models                           Motivation 1: Growing performance
                                                                                                                              of GPU w/ only algorithm updates
                                                                                                                                                                                              Motivation 2: FPGA vs. GPU - Low
                                                                                                                                                                                                 Compute but High Memory
                                         (LLM) has greatly enhanced everyday applications. While many
                                         FPGA-based accelerators, with flexibility for fine-grained data
arXiv:2511.06174v2 [cs.AR] 22 Mar 2026




                                         control, exhibit superior speed and energy efficiency compared to
                                         GPUs, recent GPU-specific optimizations have diminished this ad-
                                         vantage. When limited to arithmetic-based computation, FPGAs
                                         often underperform GPUs due to their comparatively fewer com-
                                         putational resources. To address this challenge, we exploit a key            Promising Goal: Improve LLM inference efficiency on FPGA with memory
                                         advantage of FPGAs over GPUs: abundant distributed on-chip                   based operations to exploit the abundant and distributed on-chip mem
```

## 2. 제안하는 방법
- 요약: activation-weight co-quantization과 vector quantization으로 arithmetic을 table lookup으로 바꾼다.

### Paper Evidence: Method Section
```text
approaches, or be infeasible with limited on-chip resources.       A. Language Model Acceleration
• Additional Complexity of LLMs (C2): prior memory-
                                                                      Most LLM accelerators adopt the transformer architecture
based neural accelerators overlook the additional complex-
                                                                   [29], which contains a series of attention blocks implemented
ity of LLM accelerations, where centroid search is hard to
                                                                   in grouped-query attention (GQA) [30] and feedforward net-
pipeline in decoding, lookup table access is limited by on-
                                                                   works (FFN) with residual connections [6]. Previous works
chip memory ports, and interactions between linear layers and
                                                                   [7], [8], [27], [31] develop customized transformer accelerators
other components further hinder data movement efficiency.
                                                                   with arithmetic-based operations. FlightLLM and DFX [8],
                                                                   [31] sequentially execute operations with various sparsification
  In this paper, we propose LUT-LLM, the first FPGA ac-
                                                                   and quantization techniques. Allo [7] and StreamTensor [32]
celerator targeting 1B+ parameter LLMs using memory-based
                                                                   are dataflow accelerator design frameworks that can be used
computation. LUT-LLM replaces conventional linear layers
                                                                   for LLMs, and InTAR [27] proposes a spatial-temporal hybrid
with table lookups over pre-computed dot-product results. To
                                                                   design for efficient on-chip management of LLMs.
enable efficient table construction, we adopt vector quanti-
zation [26], which maps multiple values to low-bit indices.        B. Weight Vector Quantization
This design reduces the number of operations and short-
ens per-operation latency, enabling competitive performance
compared to GPUs that often underutilize available memory
bandwidth [8]. To address C1, we develop a performance
model for LLM inference with memory-based computation
and vector quantization to understand the scaling (Section III).
We innovatively show t
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: parallel centroid search, 2D table lookup, spatial-temporal hybrid execution을 사용한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `bit-width`:
```text
e mainly show the pre-computed in full precision. The pre-computed lookup table key results for different quantization strategies, and leave the can be further quantized with low loss [20], used in our design. detailed derivation in Appendix VII. TABLE I S YMBOLS USED IN PERFORMANCE MODELING OF LINEAR LAYER Symbol Meaning Symbol Meaning Used Hardware Resource Quantization Configurations Np On-chip memory ports G Vectors per quantization group bp Bit-width per access v Vector Length Nc Compute units cw Centroids per weight codebook Opf p32 FP32 MACs/cycle/compute unit ca Centroids per activation codebook Opint8 INT8 MACs/cycle/compute unit D, M Input/Output weight dimensions C Off-chip bandwidth (bytes/cycle) L Activation (Sequence) length For weight vector quantization, the latency of loading codebook and weight centroid indices is Tmem = (4M Dcw /Gv + M D log(cw )/8v)/C (1) and the late
```
- keyword `mixed precision`:
```text
owever, this significantly increases fanout and introduces Based on the performance analysis, we design the LUT- ca −1 comparators, which escalate routing difficulty, negatively LLM accelerator with activation-weight co-quantization ap- impact timing, and consume more power on driving resources. plied. Since not all operations can be executed through table In LUT-LLM, we propose the bandwidth-aware parallel lookup and the language model involves mixed precisions and centroid search unit (BPCSU), where we design the structure computation orders, we carefully orchestrate the execution of data communication among dPEs to match the bandwidth of operations and allocate on-chip resources to ensure its of data loading in the 2D-PSum. We found that a complete feasibility on the target FPGA. LUT-LLM integrates a LUT- reduction tree is often an overkill: we can support longer Linear engine with a
```
- keyword `parallelism`:
```text
h INT8 lookup tables, and FP32 codebooks. Compared with demand; (2) co-quantization has higher data reuse of lookup conventional roofline analyzes of transformer acceleration [6], tables than weight-only or activation-only quantization due to [46] with arithmetic-based computations, we observe two repeating access to the same centroid, lowering memory port properties of the naive memory-based computation for the vec- requirements to sustain high parallelism; and (3) INT8 table tor quantized model. First, memory-based compute can have lookup and accumulation reduce on-chip memory pressure accumulator along with the output buffer. Before distributing outputs to other modules, the dequantizer converts each value in the output buffer to FP32 with the specified per-tensor scaling and shifting factors. B. Bandwidth-aware Parallel Centroid Search As shown in Figure 4, during inference, the firs
```

## 4. 하드웨어 아키텍처
- 요약: on-chip memory 기반 lookup engine과 data caching 최소화 구조가 핵심이다.

### Paper Evidence: Architecture / Implementation
```text
implementation of FPGA HLS designs,” in Proceedings of the 2022
     ACM/SIGDA International Symposium on Field-Programmable Gate
     Arrays, 2022, pp. 1–12.
[59] L. Guo, Y. Chi, J. Wang, J. Lau, W. Qiao, E. Ustun, Z. Zhang,
     and J. Cong, “AutoBridge: Coupling coarse-grained floorplanning and
     pipelining for high-frequency HLS design on multi-die FPGAs,” in The
     2021 ACM/SIGDA International Symposium on Field-Programmable
     Gate Arrays, 2021, pp. 81–92.
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
LUT-LLM: Efficient Language Model Inference with Memory-based Computations on FPGAs Zifan He∗ , Shengyu Ye† , Rui Ma† , Yang Wang† and Jason Cong∗ ∗ University of California, Los Angeles † Microsoft Research Email: zifanhe1202@g.ucla.edu, {v-yeshengyu, mrui, Yang.Wang92}@microsoft.com, cong@cs.ucla.edu Abstract—The rapid development of large language models Motivation 1: Growing performance of GPU w/ only algorithm updates Motivation 2: FPGA vs. GPU - Low Compute but High Memory (LLM) has greatly enhanced everyday applicat
```
- keyword `U280`:
```text
W.-M. Chen, W.-C. Wang, G. Xiao, [15] E. Frantar, S. Ashkboos, T. Hoefler, and D. Alistarh, “GPTQ: Accurate X. Dang, C. Gan, and S. Han, “AWQ: Activation-aware weight quanti- post-training quantization for generative pre-trained transformers,” arXiv zation for on-device llm compression and acceleration,” Proceedings of preprint arXiv:2210.17323, 2022. machine learning and systems, vol. 6, pp. 87–100, 2024. [16] AMD Adaptive Computing, “AMD Alveo U280 Data Center Accel- [35] G. Xiao, J. Lin, M. Seznec, H. Wu, J. Demouth, and S. Han, erator Card Data Sheet (DS963, v1.7),” https://docs.amd.com/r/en-US/ “Smoothquant: Accurate and efficient post-training quantization for large ds963-u280, Jun. 2023, accessed September 2, 2025. language models,” in International conference on machine learning. [17] NVIDIA Corporation, “NVIDIA V100 Tesla Tensor Core GPU PMLR, 2023, pp. 38 087–38 099. Data Sheet
```
- keyword `Vivado`:
```text
circuit, we use Xilinx decode stage only has a single token Q vector involved. In Vitis HLS 2024.2 with TAPA framework [57] and utilize LUT-LLM, we parallelize computations only along the hidden RapidStream [58], [59] for coarse-grain floorplanning to re- dimension and sequence dimension for K and V . This ensures duce routing congestion and improve timing. The design is full utilization of every compute element in both stages. implemented with Vivado 2024.2 with our customized block KV Cache Prefetch and Write-out Orchestrations. During design to integrate the HLS-based RTL block and generate the the prefill stage, the KV cache is streamed to the HBM bitstream. Resource utilizations is in Table ??. controller while delivering to the dataflow attention engine. GPU Benchmarking. We select AMD Instinct MI210 [60] During the decode stage, previously dumped KV cache is and NVIDIA A100 [19]
```

## 5. 실험 방법
- 실험 section heading을 자동 추출하지 못했다. PDF 원문 확인 필요.

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `LLaMA`:
```text
- on computation than on memory access. However, FPGAs sonal AI assistants [9], [10], smart home devices [11], and typically provide larger amounts of distributed on-chip mem- interactive robotics [12]. Nevertheless, the advancements in ory units: the AMD V80 (7nm) [18] integrates 14.9× more GPU kernel and algorithm optimization have narrowed this on-chip memory units with a 2.5× larger capacity than the performance gap. In particular, serving a Llama 2 7B model on NVIDIA A100 (7nm) [19]. Therefore, shifting the computa- a single NVIDIA A100 GPU obtains a 2.38× throughput im- tional paradigm to memory-based computations is promis- provement and a 3.27× energy efficiency gain when FlashAt- ing. Prior memory-based accelerators [20]–[23] show the tention [13], FlashDecoding [14], and quantization with GPTQ efficiency on neural networks within 10M parameters, but face [15] are enabled. Under
```
- keyword `dataset`:
```text
-operation communication. Conven- expanded by looking up the LUT row register using the weight tional language model accelerators typically adopt either pure dataflow [7], [32] or sequential [8], [31] execution strategies. V. E XPERIMENT While both exhibit strong performance for arithmetic-intensive A. Experiment Setup operations, they are suboptimal when the LUTLinear engine is introduced. For example, in a linear projection followed by Models, Datasets, and Algorithm. Our overall scheme (fol- attention computations, pure dataflow execution (Figure 9(a)) lowing the notation in Table I) is: G = 512, v = 2, cw = 16, instantiates separate hardware modules for each operation and ca = 64, and INT8 quantized lookup tables. Based on Qwen 3 streams partial outputs through FIFOs. However, in the case of 1.7B model [51], we first calibrate for activation quantization. LUTLinear, generating partia
```

## 7. 실험 결과
- keyword `speedup`:
```text
recovered matrix with activations. Compared to scalar quantization, VQ We develop a training recipe for vector-quantized model for provides non-uniform compression [25], yielding higher rep- our scheme to convert and deploy Qwen 3 1.7B on LUT-LLM resentational power for superior accuracy at the same bitwidth. prototyped with AMD V80 FPGA. Compared with GPUs at the same technology node, our design achieves a 1.10−3.29× C. Memory-based Computation speedup and 3.05−6.60× higher energy efficiency than AMD Memory-based computation replaces arithmetic by loading MI210 and NVIDIA A100. pre-computed outputs from memory, making it well-suited for on-chip memory-rich platforms such as FPGAs. It reduces re- A. Performance of Vector Quantized Linear Layer source usage and latency since a multiply-accumulate (MAC) We start with modeling the performance of the linear operation (two reads, one multiply
```
- keyword `energy`:
```text
edu, {v-yeshengyu, mrui, Yang.Wang92}@microsoft.com, cong@cs.ucla.edu Abstract—The rapid development of large language models Motivation 1: Growing performance of GPU w/ only algorithm updates Motivation 2: FPGA vs. GPU - Low Compute but High Memory (LLM) has greatly enhanced everyday applications. While many FPGA-based accelerators, with flexibility for fine-grained data arXiv:2511.06174v2 [cs.AR] 22 Mar 2026 control, exhibit superior speed and energy efficiency compared to GPUs, recent GPU-specific optimizations have diminished this ad- vantage. When limited to arithmetic-based computation, FPGAs often underperform GPUs due to their comparatively fewer com- putational resources. To address this challenge, we exploit a key Promising Goal: Improve LLM inference efficiency on FPGA with memory advantage of FPGAs over GPUs: abundant distributed on-chip based operations to exploit the abunda
```
- keyword `accuracy`:
```text
perations lookup hybrid design Challenge 2: Additional Complexity of Contri. 2: Specialized Arch. for LLM lookups, and (3) a spatial-temporal hybrid design to reduce data LLMs w/ memory-based operations w/ memory-based operations (Sec. 4) caching for a higher throughput table lookup. We develop a training recipe that converts existing models to support table Fig. 1. Motivations and challenges of memory-based computation for LLM lookups with high accuracy and prototype LUT-LLM for Qwen inference on FPGA, with the corresponding solutions as the technical contri- 3 1.7B model on the AMD V80 FPGA, reducing arithmetic butions in LUT-LLM. operations by 4× and achieving a 1.10 ∼ 3.29× faster generation speed and a 3.05 ∼ 6.60× higher energy efficiency than GPUs. I. I NTRODUCTION highlight the growing competitiveness of GPUs in efficient inference and suggest that sustaining FPGA’s advantages wi
```
- keyword `throughput`:
```text
, we exploit a key Promising Goal: Improve LLM inference efficiency on FPGA with memory advantage of FPGAs over GPUs: abundant distributed on-chip based operations to exploit the abundant and distributed on-chip memory memory embedded among computational units. We believe that resources. High Mem. Requirement High Comp. Cost shifting LLM inference from arithmetic-based to memory-based (e.g., activation vector (e.g., weight VQ.) Activation-weight throughput quantization (VQ).) Co-VQ computations through table lookups can improve the efficiency memory compute on FPGAs to compete with GPUs. However, existing methods cores are inefficient or unable to scale and deploy language models Cannot fit on FPGA (e.g., LUT operational intensity due to algorithm and architecture design limitations. This paper based Neural Accelerator w/ weights hardwired on chip) Contri. 1: Performance Model & New Effi
```
- keyword `GOPS`:
```text
d September 2, 2025. pp. 1–15. [61] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonza- [41] J. Wei, S. Cao, T. Cao, L. Ma, L. Wang, Y. Zhang, and M. Yang, “T- lez, H. Zhang, and I. Stoica, “Efficient Memory Management for Large MAC: CPU renaissance via table lookup for low-bit LLM deployment Language Model Serving with PagedAttention,” in Proceedings of the on edge,” in Proceedings of the Twentieth European Conference on ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023, Computer Systems, 2025, pp. 278–292. introduces the vLLM framework and the PagedAttention mechanism. [42] A. Raha and V. Raghunathan, “qLUT: Input-aware quantized table [62] E. Frantar, R. L. Castro, J. Chen, T. Hoefler, and D. Alistarh, “Marlin: lookup for energy-efficient approximate accelerators,” ACM Transac- Mixed-precision auto-regressive parallel inference on large language tions o
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization
- URAM/BRAM/LUTRAM binding
- memory banks 8/16/32
- parallelism factor 8/16/32
- FIFO depth and double buffering

## 9. HGTXR 적용 판단
- 현재 사용자가 LUT 과사용을 줄이길 원하므로 default가 아니라 negative-control이다. URAM lookup 기반 특수 MLP만 먼 후순위 검토.

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
