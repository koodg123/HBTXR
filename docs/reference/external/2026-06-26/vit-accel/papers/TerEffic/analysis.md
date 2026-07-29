---
source_type: paper
source_name: TerEffic
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/TerEffic/analysis.md -->

# TerEffic Detailed Paper Analysis

- title: TerEffic: Highly Efficient Ternary LLM Inference on FPGA
- url: https://arxiv.org/abs/2502.16473
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/TerEffic.pdf`
- related_codebases: `ternaryLLM`

## 1. 기존 방법의 문제점
- 요약: edge LLM은 off-chip memory와 weight footprint가 병목이다.

### Paper Evidence: Abstract
```text
Abstract—Deploying Large Language Models (LLMs) effi-             GPT-4o mini [11] demonstrates superior capabilities to GPT-4
                                        ciently on edge devices is often constrained by limited memory       at significantly reduced inference costs.
                                        capacity and high power consumption. Low-bit quantization               Nevertheless, existing AI inference accelerators—such as
                                        methods, particularly ternary quantization, have demonstrated
                                        significant potential in preserving model accuracy while sub-        GPUs, NPUs, and TPUs—do not fully align with this emerg-
                                        stantially decreasing memory footprint and computational costs.      ing trend. Firstly, these accelerators lack specialized hardware
                                        However, existing general-purpose architectures and accelerators     tailored to efficiently handle the arithmetic operations and
                                        have not fully exploited the advantages of low-bit quantization      data movement patterns specific to low-bit quantized models,
                                        due to insufficient specialized hardware support.                    particularly binary and ternary LLMs. Secondly, current ac-
                                           We introduce TerEffic, an FPGA-based architecture tailored
                                        for ternary-quantized LLM inference. The proposed system             celerators are predominantly DRAM-centric, relying heavily
                                        offers flexibility through reconfigurable hardwar
```

## 2. 제안하는 방법
- 요약: ternary quantization과 custom compute/memory hierarchy로 on-chip inference를 지향한다.

### Paper Evidence: Method Section
```text
methods, particularly ternary quantization, have demonstrated
                                        significant potential in preserving model accuracy while sub-        GPUs, NPUs, and TPUs—do not fully align with this emerg-
                                        stantially decreasing memory footprint and computational costs.      ing trend. Firstly, these accelerators lack specialized hardware
                                        However, existing general-purpose architectures and accelerators     tailored to efficiently handle the arithmetic operations and
                                        have not fully exploited the advantages of low-bit quantization      data movement patterns specific to low-bit quantized models,
                                        due to insufficient specialized hardware support.                    particularly binary and ternary LLMs. Secondly, current ac-
                                           We introduce TerEffic, an FPGA-based architecture tailored
                                        for ternary-quantized LLM inference. The proposed system             celerators are predominantly DRAM-centric, relying heavily
                                        offers flexibility through reconfigurable hardware to meet various   on off-chip High Bandwidth Memory (HBM). Such DRAM-
                                        system requirements. We evaluated two representative config-         based designs incur considerable energy overhead and suffer
                                        urations: a fully on-chip design that stores all weights within      from bandwidth bottlenecks due to the memory wall [12].
                                        on-chip memories, scaling out using multiple FPGAs, and an           However, aggressive model distillation combined with low-
                                        HBM-assisted design capable of accommodating larger models
                                        on a single FPGA board.                                              bit quantization (e.g., 1-bit LLMs) significantly reduces mem-
                                           Experimental results demonstrate significant performance and      ory requirements, practically enabling fully SRAM-based on-
                                        energy efficiency improvements. For single-batch inference on a      chip inference. As illustrated in Figure 1, low-bit quanti-
                                        370 M-parameter model, our fully on-chip architecture achieves
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: ternary weight compression과 ternary compute unit, HBM-assisted variant를 구성한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `sparsity`:
```text
s/second with 46W Section IV-C, the HBM-assisted architecture is memory-bound power consumption, which is comparable to the typical power when the batch size and the arithmetic intensity are small. usage of a personal laptop. Our architecture shows remarkable The single-batch throughput is thereby constrained by the performance advantages over the SOTA FPGA LLM accelera- HBM bandwidth, which is 1489 tokens/second for the 1.3B tors. By leveraging sparsity, FlightLLM[30] delivers 55 token- model and 727 tokens/second for the 2.7B model. When the s/second for a 7B model under 45W on the same AMD U280 batch size increases, we utilize the batch-parallelism, and FPGA, while EdgeLLM[31] generates 75 tokens/second for the system will transform from memory-bound to compute- a 6B model under 51W on a bigger FPGA(AMD VCU128). bound. Referencing Figure 9, the multi-batch throughput will Therefore, o
```
- keyword `parallelism`:
```text
SRAM bandwidth. However, as the model size grows, the available on-chip memory capacity inevitably becomes insufficient. In such cases, utilizing an off-chip HBM with large capacity becomes necessary. Nonetheless, due to its com- paratively lower bandwidth relative to on-chip SRAM, using HBM may introduce performance bottlenecks, as illustrated by the purple line in Figure 6. To address these varying constraints, we propose two ar- Fig. 7: Layer-Parallelism for Fully On-chip Inference with chitecture variants: the fully on-chip and the HBM-assisted Multiple Cards architectures. The fully on-chip architecture prioritizes short latency but is constrained to smaller models, requiring either HBM on U280 is able to store up to 8GB of data off-chip, FPGAs with larger on-chip memory or scaling across multiple approximately the size of a 40B ternary-quantized model. As cards to provide sufficien
```
- keyword `buffer`:
```text
, RMSNorm(X) = (1) d i=1 i r Fig. 2: Architecture Overview where X ∈ R1×d denotes the input activation, Wn ∈ R1×d denotes the normalization weight, r denotes the Root-Mean- Square(RMS) result , ϵ is a small constant and ⊙ represents dot product. As shown in the architecture in Figure 4, the RMS computation and X ⊙ Wn can be executed in parallel. As the RMS process has longer latency, the X ⊙Wn results are temporarily stored in SRAM-based on-chip buffers. Moreover, as divisions incur high hardware cost and long cycle latency, we replace the expensive divisions (÷r) with DSP-based Fig. 3: 1.6-Bit Weight Compression multiplications (× 1r ), using r as an index to retrieve 1/r from buffer between them. The module details will be provided in an on-chip look-up table consisting of a small amount of Section III-C and III-D. In addition, a reconfigurable module SRAMs. This hardware optimization
```

## 4. 하드웨어 아키텍처
- 요약: fully on-chip smaller model path와 HBM-assisted larger model path를 제안한다.

### Paper Evidence: Architecture / Implementation
```text
architecture processes 727 tokens/second for a larger 2.7B-          chip memory bandwidth, FPGAs can substantially improve
                                        parameter model—3× the throughput of NVIDIA A100—while               throughput and energy efficiency compared to conventional
                                        consuming only 46W, resulting in a power efficiency of 16            GPU-based architectures.
                                        tokens/second/W, an 8× improvement over the A100.
                                           Index Terms—Hardware Acceleration, FPGA, LLM Inference,                   FPGA inference with on-chip    FPGA inference with DRAM
                                                                                                                                                    and on-chip SRAM (after-quantization)
                                                                                                                     SRAM (before-quantization)
                                        Ternary Quantization                                                         FPGA inference with on-chip     FPGA potential with more        GPU inference mainly with
                                                                                                                     SRAM (after-quantization)       on-chip SRAM capacity           DRAM and on-chip SRAM


                                                              I. I NTRODUCTION                                   Throughput                                     Energy Consumption


                                           Large Language Models (LLMs) have revolutionized AI by
                                                                                                                                                                                      Quantization
                                        delivering unprecedented performance across diverse applica-                            Quantization                                          enables larger
                                                                                                                                enables larger                                        on-chip inference
                                        tions. State-of-the-art LLMs such as ChatGPT [1], Claude [2],                           on-chip inference
                                        and Google Gemini [3] continue scaling upwards in parameter
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
TerEffic: Highly Efficient Ternary LLM Inference on FPGA Chenyang Yin∗ , Zhenyu Bai† , Pranav Venkatram† , Shivam Aggarval† , Zhaoying Li† , and Tulika Mitra† ∗ School of Electronic Engineering and Computer Science, Peking University Email: ycy@stu.pku.edu.cn † School of Computing, National University of Singapore Email: zhenyu.bai@nus.edu.sg Email: e0552200@u.nus.edu Email: shivam@comp.nus.edu.sg arXiv:2502.16473v2 [cs.AR] 1 May 2025 Email: zhaoying@comp.nus.edu.sg Email: tulika@comp.nus.edu.sg A
```
- keyword `U280`:
```text
rnel library. FlightLLM[30] utilized the FPGA-specific performance. For a 2.7B-parameter model, it provides a single- DSPs and hierarchical memory. It introducd a sparse DSP batch throughput of 727 tokens/second and a power efficiency chain and an always-on-chip decode scheme to reduce memory of 16 tokens/second/W, which is 3× and 8× better compared overhead, providing 55 tokens/second for a 7B LLM on to NVIDIA’s A100, respectively. Xilinx Alveo U280. Most recently, EdgeLLM[31] proposed a CPU-FPGA heterogeneous acceleration system. Highlighting a II. BACKGROUND & R ELATED WORK unified data format and customized FP16*INT4 computational A. Binary and Ternary Quantizations units, it outperformed the performance of [30] by 10%-20%. Quantization has been a key strategy to accelerate inference These works have demonstrated the potential for efficient as neural networks grow rapidly in paramete
```
- keyword `Vivado`:
```text
005 14 nm 77GB/s 30.5 MB include addition, subtraction and dot product (Dot) that can ([17]) NVIDIA Jetson Orin 8 nm 68.29GB/s (L1+L2) 1.25 MB be directly implemented using LUTs and DSPs, while the NVIDIA A100 7 nm 1935GB/s (L1+L2) 100.25 MB sigmoid function is implemented by a BRAM-based look-up table. B. TerEffic Implementation Details Table II shows different attributes of the models for eval- We performed synthesis, placement, and routing on Vivado uation. The model parameters range from 370M to 2.7B: we v2023.2, achieving 150MHz frequency. The layouts of the two evaluate the fully on-chip architecture using the 370M model architectures are presented in Figure 11. and the HBM-assisted architecture using the larger models. 1) Resource Utilization: Table IV shows the resource uti- lization in the two architectures, where ’QSFP’ and ’HBM’ TABLE II: Attributes of the Demonstration Models
```

## 5. 실험 방법
```text
evaluations. In addition, the HBM loading is asynchronous
                                                                     TensorRT [41] for maximum inference performance. Since
from the compute core, with HBM running at 450 MHz and
                                                                     larger models (with 1.3B and 2.7B parameters) exceed the
the core at 150 MHz. A BRAM-composed FIFO buffer is
                                                                     memory capacity of the Jetson Orin Nano, these models are
thereby inserted between the HBM and the core computing
                                                                     deployed on NVIDIA A100 GPU, again utilizing TensorRT
logic. Concretely, on the U280 board, all 32 HBM channels
                                                                     optimization. GPU baselines are quantized to the lowest sup-
are used at 450 MHz, providing a peak bandwidth of 460
                                                                     ported precision: INT8 for the Jetson Orin Nano and INT4
GB/s.
                                                                     for the A100. The hardware specification of the baselines and
                                                                     ours is listed in Table III.
                        V. E VALUATION                                  It is worth noting that TerEffic specifically accelerates
                                                                     the decoder layers, whereas our GPU baseline experiments
A. Evaluation Methodology
                                                                     include the other layers in the entire model (embedding,
   1) Demonstration Ternary Models: We use the MatMul-               position encoding, and output layers). However, additional
free LM models [17] as the demonstration ternary models.             GPU experiments indicate that these extra layers account
Figure 10 shows the layer computation graph. The models              for less than 0.1% of the total computational workload and
have been proven to match the performance of the similar-size        inference runtime on GPUs. Thus, the throughput comparisons
SOTA Transformers, while eliminating high-precision matrix           presented remain fair and representative of actual performance
multiplications [17]. Analogy to traditional Transformers [34],      differences.
the HGRN [38] is an RNN-based alternative to the self-
attention mechanism, while the GLU [39], widely used in              TABLE III: Hardware Specification of Baselines and TerEffic
models like Llama[40], is recognized by many as a robust                                     Tech Node           Off-chip bw.              On-chip SRAM
                                                                        AMD Alveo U280
enhancement for feed-forward networks (FFNs). The model-                 (Ours TerEffic)
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
1–5, 05 2024. [38] Z. Qin, S. Yang, and Y. Zhong, “Hierarchically gated recurrent neu- [13] I. Hubara, M. Courbariaux, D. Soudry, R. El-Yaniv, and Y. Bengio, ral network for sequence modeling,” Advances in Neural Information “Binarized neural networks,” NIPS, vol. 29, 2016. Processing Systems, vol. 36, 2024. [14] M. Rastegari, V. Ordonez, J. Redmon, and A. Farhadi, “Xnor-net: [39] Y. Dauphin, A. Fan, M. Auli, and D. Grangier, “Language modeling Imagenet classification using binary convolutional neural networks,” in with gated convolutional networks,” 12 2016. European conference on computer vision, pp. 525–542, Springer, 2016. [40] T. Hugo et al., “Llama: Open and efficient foundation language models,” [15] F. Li, B. Liu, X. Wang, B. Zhang, and J. Yan, “Ternary weight networks,” 2023. 2022. [41] NVIDIA, “Tensorrt,” 2024. [16] T. Chen, Z. Li, W. Xu, Z. Zhu, D. Li, L. Tian, E. Barsoum, P.
```
- keyword `LLaMA`:
```text
to enhance their capabilities. However, a concurrent and significant trend has emerged toward more efficient inference through the deployment of smaller yet highly effective mod- els. This efficiency is primarily achieved through advanced Tiny model Quantized large model Model Parameters Tiny model Quantized large model Model Parameters techniques like model distillation [4], [5] and low-bit quanti- zation [6], [7], [8], [9]. For example, Meta’s Llama3.2 [10] Fig. 1: On-chip vs. Off-chip Memory for Inference: Through- provides lightweight models with 1B and 3B parameters put and Energy Trends with Increasing Model Parameters specifically optimized for high-performance edge tasks, while Therefore, we propose TerEffic, a specialized FPGA- based accelerator architecture optimized explicitly for efficient performance. Furthermore, this 1-bit quantization trend was ternary LLM inference. TerE
```

## 7. 실험 결과
- keyword `energy`:
```text
. particularly binary and ternary LLMs. Secondly, current ac- We introduce TerEffic, an FPGA-based architecture tailored for ternary-quantized LLM inference. The proposed system celerators are predominantly DRAM-centric, relying heavily offers flexibility through reconfigurable hardware to meet various on off-chip High Bandwidth Memory (HBM). Such DRAM- system requirements. We evaluated two representative config- based designs incur considerable energy overhead and suffer urations: a fully on-chip design that stores all weights within from bandwidth bottlenecks due to the memory wall [12]. on-chip memories, scaling out using multiple FPGAs, and an However, aggressive model distillation combined with low- HBM-assisted design capable of accommodating larger models on a single FPGA board. bit quantization (e.g., 1-bit LLMs) significantly reduces mem- Experimental results demonstrate signifi
```
- keyword `accuracy`:
```text
nus.edu.sg Abstract—Deploying Large Language Models (LLMs) effi- GPT-4o mini [11] demonstrates superior capabilities to GPT-4 ciently on edge devices is often constrained by limited memory at significantly reduced inference costs. capacity and high power consumption. Low-bit quantization Nevertheless, existing AI inference accelerators—such as methods, particularly ternary quantization, have demonstrated significant potential in preserving model accuracy while sub- GPUs, NPUs, and TPUs—do not fully align with this emerg- stantially decreasing memory footprint and computational costs. ing trend. Firstly, these accelerators lack specialized hardware However, existing general-purpose architectures and accelerators tailored to efficiently handle the arithmetic operations and have not fully exploited the advantages of low-bit quantization data movement patterns specific to low-bit quantized m
```
- keyword `throughput`:
```text
n (e.g., 1-bit LLMs) significantly reduces mem- Experimental results demonstrate significant performance and ory requirements, practically enabling fully SRAM-based on- energy efficiency improvements. For single-batch inference on a chip inference. As illustrated in Figure 1, low-bit quanti- 370 M-parameter model, our fully on-chip architecture achieves zation allows larger models to fit entirely within on-chip 16,300 tokens/second, delivering a throughput 192× higher than SRAM, making architectures with distributed SRAM—such NVIDIA Jetson Orin Nano with a power efficiency of 455 to- kens/second/W, marking a 19× improvement. The HBM-assisted as FPGAs—particularly advantageous. By leveraging high on- architecture processes 727 tokens/second for a larger 2.7B- chip memory bandwidth, FPGAs can substantially improve parameter model—3× the throughput of NVIDIA A100—while throughput and energy
```
- keyword `latency`:
```text
is the most important module model, [17] used an RNN variant to replace the traditional in the architecture, consisting of the RMSNorm Module, the self-attention in their ternary model and achieved excellent TMat (Ternary Matrix Multiplication) Core, and the activation decoded into 2-bit format where 01 corresponds to 1, 11 to -1, and 00 to 0. As the decoding involves only bitwise operations, such as + and &, it incurs minimal hardware cost and latency. C. RMSNorm Module The RMSNorm[33] is more computationally efficient than the traditional LayerNorm[34] while maintaining high ac- curacy, making it well-suited for FPGA. The algorithm for RMSNorm is presented below: v u d u1 X X ⊙ Wn r=t x2 + ϵ , RMSNorm(X) = (1) d i=1 i r Fig. 2: Architecture Overview where X ∈ R1×d denotes the input activation, Wn ∈ R1×d denotes the normalization weight, r denotes the Root-Mean- Square(RMS) result , ϵ
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
- 정확도 리스크가 큰 ternary 압축은 P3 negative-control이다.

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
