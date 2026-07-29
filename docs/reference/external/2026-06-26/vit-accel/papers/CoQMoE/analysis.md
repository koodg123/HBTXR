---
source_type: paper
source_name: CoQMoE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/CoQMoE/analysis.md -->

# CoQMoE Detailed Paper Analysis

- title: CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA
- url: https://arxiv.org/abs/2506.08496
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/CoQMoE.pdf`
- related_codebases: `CoQMoE`

## 1. 기존 방법의 문제점
- 요약: MoE-ViT는 계산량을 줄이지만 FPGA에서는 expert routing, quantization, reusable compute scheduling이 resource/latency 병목을 만든다.

### Paper Evidence: Introduction / Motivation
```text
1        Introduction

                                             The Vision Transformer (ViT) has garnered significant attention for its
                                         outstanding performance in various computer vision tasks [1,7,16,22,26]. Building
                                         on the Mixture-of-Experts (MoE) architecture, MoE-ViT extends the original ViT
                                         by scaling model size without proportional increases in computational complexity,
                                         thereby enhancing multitasking capabilities and emerging as a focal point of
                                         recent research [2, 4, 6, 10, 14, 24]. However, as the models grow in size, the rapid
                                         increase in the number of parameters introduces new challenges, particularly the
                                         need for more efficient computation strategies.
                                                                     †
                                             *
                                                 Equal contribution. Corresponding authors.

2         Dong et al.

    Current research on ViT and MoE-ViT optimizations predominantly falls
into two categories: algorithmic improvements and hardware implementations.
On the algorithmic front, a substantial body of work [11, 12, 19, 20] focuses on
optimizing aspects such as quantization and sparsity, with a significant emphasis
on quantization techniques aimed at reducing the computational footprint without
compromising accuracy. From a hardware design perspective, FPGA-based ViT
accelerators have become a topic of great interest. The architectures for these
accelerators can generally be classified into two types: pipeline-based [9, 22]
and reuse-based [5, 15, 18]. The reuse-based architecture typically instantiates a
single processing element (PE) and relies on the host CPU for data exchange. In
contrast, the pipeline-based architecture divides tasks into multiple sequential
stages, with each stage executed by dedicated hardware, significantly improving
frames per
```

## 2. 제안하는 방법
- 요약: precision-preserving complex quantizer와 hardware-friendly simplified quantizer를 결합하고, resource-aware accelerator에서 streaming attention과 reusable linear operator를 조합한다.

## 3. 구체적인 알고리즘
- 핵심 알고리즘: scale reparameterization 기반 dual-stage quantization과 expert computation orchestration으로 MoE token/expert workload를 정렬한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `bit-width`:
```text
s Our contribution primarily targets MoE-ViT, but the lack of existing quantized MoE accelerator implementations makes our validation relatively limited. Thus, we also conducted experiments and comparisons within ViT-related work to validate the feasibility of our results. CoQMoE 11 Table 1: Quantization results of image classification on ImageNet dataset, where each data presents the Top-1 accuracy (%).“W/A/Attn” indicates that the quantization bit-width of weights/activations/attention maps, respectively. W/A/Attn M3 ViT ViT-T ViT-S ViT-B DeiT-T DeiT-S DeiT-B Full Precision 32/32/32 85.17 75.46 81.39 84.53 72.21 79.85 81.85 MinMax [12] 8/8/8 82.54 19.87 30.28 23.64 70.94 75.05 78.02 RePQ-ViT [11]∗ 8/8/8 - 72.85 81.31 84.36 71.94 79.76 81.79 TSPTQ-ViT [20] 8/8/8 - - 81.20 84.11 71.87 79.56 81.72 FQ-ViT [12] 8/8/4 - - - 82.68 71.07 78.40 80.85 P2-ViT [19] 8/8/4 - - - 82.80 70.92 78.24 80
```
- keyword `sparsity`:
```text
the number of parameters introduces new challenges, particularly the need for more efficient computation strategies. † * Equal contribution. Corresponding authors. 2 Dong et al. Current research on ViT and MoE-ViT optimizations predominantly falls into two categories: algorithmic improvements and hardware implementations. On the algorithmic front, a substantial body of work [11, 12, 19, 20] focuses on optimizing aspects such as quantization and sparsity, with a significant emphasis on quantization techniques aimed at reducing the computational footprint without compromising accuracy. From a hardware design perspective, FPGA-based ViT accelerators have become a topic of great interest. The architectures for these accelerators can generally be classified into two types: pipeline-based [9, 22] and reuse-based [5, 15, 18]. The reuse-based architecture typically instantiates a single process
```
- keyword `parallelism`:
```text
aded into on-chip buffers to maximize data reuse, or streamed directly from off-chip memory during execution to accommodate larger model sizes. This hybrid strategy ensures adaptability to varying model scales and aligns with the memory hierarchy and bandwidth characteristics of different FPGA platforms. 4.2 Bandwidth Optimization in Parallel Computation Quantization can effectively reduce computational resource usage. However, simply increasing parallelism to improve resource utilization does not always yield linear performance gains, as memory bandwidth often becomes a critical bottleneck, limiting system efficiency and scalability. This issue is particularly pronounced in MoE architectures, where frequent off-chip loading of expert weights exacerbates memory access demands. To address this challenge, we redesigned the memory access patterns of both the attention and linear kernels. Co
```

## 4. 하드웨어 아키텍처
- 요약: streaming attention kernel, reusable linear operator, expert scheduling buffer가 핵심이다.

### Paper Evidence: Architecture / Implementation
```text
architecture, which excels in sequential processing, conflicts with the random ex-
pert selection required in MoE models. This is evident in systems like HGpipe [9],
where even relatively small models like ViT-Tiny cannot fully deploy all layers
efficiently. Similarly, reuse architectures struggle with inefficiencies due to the
need for separate instantiation of PEs for sparse and dense linear operations,
further exacerbating performance bottlenecks.
    In response to these challenges, our work aims to explore both the quantization
opportunities and the hardware acceleration potential specific to MoE-ViTs. The
key contributions of this paper are as follows:

    – We propose a novel quantization algorithm that applies specialized quantizers
      to activations following LayerNorm and Softmax. Through reparameterization,
      the quantizers are transformed into a hardware-efficient version, enabling
      accelerator-friendly implementation while preserving accuracy.
    – We introduce a hybrid computation mode for hardware accelerators, includ-
      ing a streaming attention kernel designed to reduce latency and a unified
      sparse/dense linear kernel to enable efficient cross-layer computation in MoE
      and MLP modules. These components leverage computation reordering and
      broadcasting techniques to achieve O(1) off-chip memory access, regardless
      of the parallelization scale, thereby enhancing performance.
    – Our experimental evaluations demonstrate that the proposed approach out-
      performs the SOTA M3 ViT in key performance metrics, including speed
      and accuracy. Additionally, the proposed quantization algorithm and hard-

                                                                                                                               CoQMoE         3


                          Block × L                                                                       Gate
                                                                                                                   Expert 0




                           QKV Linear
                                         Q




              LayerNorm




                                                                                              LayerNorm
                                                                                Linear proj
                                             MatMul1

                                                        Softmax
                                                                  P
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA Jiale Dong∗ , Hao Wu∗ , Zihao Wang, Wenqi Lou† , Zhendong Zheng, Lei Gong, Chao Wang† , Xuehai Zhou arXiv:2506.08496v1 [cs.AR] 10 Jun 2025 University of Science and Technology of China, Hefei, China djl190011@mail.ustc.edu.cn {louwenqi, cswang}@ustc.edu.cn Abstract. Vision Transformers (ViTs) exhibit superior performance in computer vision tasks but face deployment challenges on resource- constrained devices due to high computational/memory
```
- keyword `ZCU102`:
```text
6 81.39 84.53 72.21 79.85 81.85 MinMax [12] 8/8/8 82.54 19.87 30.28 23.64 70.94 75.05 78.02 RePQ-ViT [11]∗ 8/8/8 - 72.85 81.31 84.36 71.94 79.76 81.79 TSPTQ-ViT [20] 8/8/8 - - 81.20 84.11 71.87 79.56 81.72 FQ-ViT [12] 8/8/4 - - - 82.68 71.07 78.40 80.85 P2-ViT [19] 8/8/4 - - - 82.80 70.92 78.24 80.96 Our Method 8/8/4 84.89 71.79 80.12 83.99 71.64 79.38 81.60 ∗ Reproduced from open-source code. Table 2: Resource consumption of deploying CoQMoE on ZCU102 and U280. Platform Model DSPs BRAMs LUTs Flip-Flop (FFs) Power ZCU102 ViT-tiny 1754 383 156.1K 198.2K 9.83W U280 ViT-small 2635 696.5 311.6K 454.1K 33.7W 5.1 Experimental Setup Quantization Evaluation We evaluate ViTs and DeiTs quantization perfor- mance on ImageNet [3]. As for quantization details, we randomly select 32 images from ImageNet’s training set as the calibration data, which are used to analyze activation distributions for offl
```
- keyword `U280`:
```text
53 72.21 79.85 81.85 MinMax [12] 8/8/8 82.54 19.87 30.28 23.64 70.94 75.05 78.02 RePQ-ViT [11]∗ 8/8/8 - 72.85 81.31 84.36 71.94 79.76 81.79 TSPTQ-ViT [20] 8/8/8 - - 81.20 84.11 71.87 79.56 81.72 FQ-ViT [12] 8/8/4 - - - 82.68 71.07 78.40 80.85 P2-ViT [19] 8/8/4 - - - 82.80 70.92 78.24 80.96 Our Method 8/8/4 84.89 71.79 80.12 83.99 71.64 79.38 81.60 ∗ Reproduced from open-source code. Table 2: Resource consumption of deploying CoQMoE on ZCU102 and U280. Platform Model DSPs BRAMs LUTs Flip-Flop (FFs) Power ZCU102 ViT-tiny 1754 383 156.1K 198.2K 9.83W U280 ViT-small 2635 696.5 311.6K 454.1K 33.7W 5.1 Experimental Setup Quantization Evaluation We evaluate ViTs and DeiTs quantization perfor- mance on ImageNet [3]. As for quantization details, we randomly select 32 images from ImageNet’s training set as the calibration data, which are used to analyze activation distributions for offline calcula
```

## 5. 실험 방법
```text
5     Experiments
    Our contribution primarily targets MoE-ViT, but the lack of existing quantized
MoE accelerator implementations makes our validation relatively limited. Thus,
we also conducted experiments and comparisons within ViT-related work to
validate the feasibility of our results.

                                                                                  CoQMoE           11


Table 1: Quantization results of image classification on ImageNet dataset, where
each data presents the Top-1 accuracy (%).“W/A/Attn” indicates that the
quantization bit-width of weights/activations/attention maps, respectively.
                       W/A/Attn M3 ViT ViT-T ViT-S ViT-B DeiT-T DeiT-S DeiT-B
     Full Precision     32/32/32        85.17     75.46   81.39   84.53   72.21    79.85     81.85
     MinMax [12]          8/8/8         82.54     19.87   30.28   23.64   70.94    75.05     78.02
    RePQ-ViT [11]∗        8/8/8           -       72.85   81.31   84.36   71.94    79.76     81.79
    TSPTQ-ViT [20]        8/8/8           -         -     81.20   84.11   71.87    79.56     81.72
     FQ-ViT [12]          8/8/4           -         -       -     82.68   71.07    78.40     80.85
      P2-ViT [19]         8/8/4           -         -       -     82.80   70.92    78.24     80.96
     Our Method           8/8/4         84.89     71.79   80.12   83.99   71.64    79.38     81.60
∗
    Reproduced from open-source code.

 Table 2: Resource consumption of deploying CoQMoE on ZCU102 and U280.
    Platform          Model        DSPs         BRAMs      LUTs      Flip-Flop (FFs)       Power
     ZCU102         ViT-tiny       1754          383       156.1K         198.2K           9.83W
      U280          ViT-small      2635          696.5     311.6K         454.1K           33.7W

5.1     Experimental Setup

Quantization Evaluation We evaluate ViTs and DeiTs quantization perfor-
mance on ImageNet [3]. As for quantization details, we randomly select 32 images
from ImageNet’s training set as the calibration data, which are used to analyze
activation distributions for offline calculating scaling factors of activations and
weights [12, 13, 23], and then evaluate accuracy on its validation set.


Hardware Deployment and Platform Selection We deploy MoE-ViT on the
Xilinx ZCU102 and Alveo U280 platforms using Vitis HLS and Vivado (v2022.2).
Since VMoE [17] lacks a PyTorch implementation, we adopt M3 ViT [8], which
shares the same computation but differs in execution order. We use PyTorch
(v2.0.1) with a batch size of 4.
    The ZCU102 features a single-SLR architecture with limited resources (2520
DSPs, 1824 BRAMs, 21 GB/s DDR bandwidth). In contrast, the U280 provides
a multi-SLR architecture with significantly more resources (9024 DSPs, 2160
BRAMs distributed across 3 SLRs) and high memory bandwidth (460 GB/s) via
8 GB of HBM2. In our evaluation, latency specifically denotes the actual kernel
execution time, defined as the interval between enqueueing the kerne
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
ip(l(x)) · sv . This step requires only Ts multipliers, ensuring efficient resource utilization. 5 Experiments Our contribution primarily targets MoE-ViT, but the lack of existing quantized MoE accelerator implementations makes our validation relatively limited. Thus, we also conducted experiments and comparisons within ViT-related work to validate the feasibility of our results. CoQMoE 11 Table 1: Quantization results of image classification on ImageNet dataset, where each data presents the Top-1 accuracy (%).“W/A/Attn” indicates that the quantization bit-width of weights/activations/attention maps, respectively. W/A/Attn M3 ViT ViT-T ViT-S ViT-B DeiT-T DeiT-S DeiT-B Full Precision 32/32/32 85.17 75.46 81.39 84.53 72.21 79.85 81.85 MinMax [12] 8/8/8 82.54 19.87 30.28 23.64 70.94 75.05 78.02 RePQ-ViT [11]∗ 8/8/8 - 72.85 81.31 84.36 71.94 79.76 81.79 TSPTQ-ViT [20] 8/8/8 - - 81.20 84.11 7
```
- keyword `DeiT`:
```text
implementations makes our validation relatively limited. Thus, we also conducted experiments and comparisons within ViT-related work to validate the feasibility of our results. CoQMoE 11 Table 1: Quantization results of image classification on ImageNet dataset, where each data presents the Top-1 accuracy (%).“W/A/Attn” indicates that the quantization bit-width of weights/activations/attention maps, respectively. W/A/Attn M3 ViT ViT-T ViT-S ViT-B DeiT-T DeiT-S DeiT-B Full Precision 32/32/32 85.17 75.46 81.39 84.53 72.21 79.85 81.85 MinMax [12] 8/8/8 82.54 19.87 30.28 23.64 70.94 75.05 78.02 RePQ-ViT [11]∗ 8/8/8 - 72.85 81.31 84.36 71.94 79.76 81.79 TSPTQ-ViT [20] 8/8/8 - - 81.20 84.11 71.87 79.56 81.72 FQ-ViT [12] 8/8/4 - - - 82.68 71.07 78.40 80.85 P2-ViT [19] 8/8/4 - - - 82.80 70.92 78.24 80.96 Our Method 8/8/4 84.89 71.79 80.12 83.99 71.64 79.38 81.60 ∗ Reproduced from open-source code
```
- keyword `dataset`:
```text
· sv . This step requires only Ts multipliers, ensuring efficient resource utilization. 5 Experiments Our contribution primarily targets MoE-ViT, but the lack of existing quantized MoE accelerator implementations makes our validation relatively limited. Thus, we also conducted experiments and comparisons within ViT-related work to validate the feasibility of our results. CoQMoE 11 Table 1: Quantization results of image classification on ImageNet dataset, where each data presents the Top-1 accuracy (%).“W/A/Attn” indicates that the quantization bit-width of weights/activations/attention maps, respectively. W/A/Attn M3 ViT ViT-T ViT-S ViT-B DeiT-T DeiT-S DeiT-B Full Precision 32/32/32 85.17 75.46 81.39 84.53 72.21 79.85 81.85 MinMax [12] 8/8/8 82.54 19.87 30.28 23.64 70.94 75.05 78.02 RePQ-ViT [11]∗ 8/8/8 - 72.85 81.31 84.36 71.94 79.76 81.79 TSPTQ-ViT [20] 8/8/8 - - 81.20 84.11 71.87 79.5
```

## 7. 실험 결과
- keyword `energy`:
```text
dly simplified quantizers via scale repa- rameterization, with only 0.28% accuracy loss compared to full precision; (2) A resource-aware accelerator architecture featuring latency-optimized streaming attention kernels and reusable linear operators, effectively balancing performance and resource consumption. Experimental results demonstrate that our accelerator achieves nearly 155 frames per second, a 5.35× improvement in throughput, and over 80% energy reduction compared to state-of-the-art (SOTA) FPGA MoE accelerators, while maintaining <1% accuracy loss across vision benchmarks. Our implemen- tation is available at https://github.com/DJ000011/CoQMoE. 1 Introduction The Vision Transformer (ViT) has garnered significant attention for its outstanding performance in various computer vision tasks [1,7,16,22,26]. Building on the Mixture-of-Experts (MoE) architecture, MoE-ViT extends the orig
```
- keyword `accuracy`:
```text
le architecture with sub-linear computational growth, their hard- ware implementation on FPGAs remains constrained by resource limita- tions. This paper proposes a novel accelerator for efficiently implementing quantized MoE models on FPGAs through two key innovations: (1) A dual-stage quantization scheme combining precision-preserving complex quantizers with hardware-friendly simplified quantizers via scale repa- rameterization, with only 0.28% accuracy loss compared to full precision; (2) A resource-aware accelerator architecture featuring latency-optimized streaming attention kernels and reusable linear operators, effectively balancing performance and resource consumption. Experimental results demonstrate that our accelerator achieves nearly 155 frames per second, a 5.35× improvement in throughput, and over 80% energy reduction compared to state-of-the-art (SOTA) FPGA MoE accelerators
```
- keyword `throughput`:
```text
izers with hardware-friendly simplified quantizers via scale repa- rameterization, with only 0.28% accuracy loss compared to full precision; (2) A resource-aware accelerator architecture featuring latency-optimized streaming attention kernels and reusable linear operators, effectively balancing performance and resource consumption. Experimental results demonstrate that our accelerator achieves nearly 155 frames per second, a 5.35× improvement in throughput, and over 80% energy reduction compared to state-of-the-art (SOTA) FPGA MoE accelerators, while maintaining <1% accuracy loss across vision benchmarks. Our implemen- tation is available at https://github.com/DJ000011/CoQMoE. 1 Introduction The Vision Transformer (ViT) has garnered significant attention for its outstanding performance in various computer vision tasks [1,7,16,22,26]. Building on the Mixture-of-Experts (MoE) architecture,
```
- keyword `GOPS`:
```text
12, 19, 20]. As summarized in Table 1, our approach achieves hardware-compatible quantization 12 Dong et al. Table 3: Comparison with GPU and Edge-MoE [18] on M3 ViT Attribute M3 ViT Edge-MoE Ours Platform Tesla V100S ZCU102 ZCU102 U280 Bit-width FP32 W 16 A32 INT8 INT8 Model M3 ViT-T M3 ViT-S M3 ViT-T M3 ViT-T M3 ViT-S Frequency (Mhz) 1245 1245 300 300 250 Power (W) 42.98 47.12 14.54 9.83 33.7 Latency (ms) 3.65 4.45 34.64 6.47 9.16 Throughput (GOPS) 561.79 1936.84 72.15 386.3 1004.3 Efficiency (GOPS/W) 11.88 19.25 4.83 38.639 29.8 while maintaining minimal accuracy degradation. Notably, the method delivers 83.99% top-1 accuracy under an 8/8/4-bit configuration on ViT-B. In addition, the accuracy loss of our quantization scheme on M3 ViT is also presented in Table 1, showing only a 0.28% reduction compared to full precision. 5.3 Comparison With Prior Transformer Accelerators Since we em
```
- keyword `FPS`:
```text
he architectures for these accelerators can generally be classified into two types: pipeline-based [9, 22] and reuse-based [5, 15, 18]. The reuse-based architecture typically instantiates a single processing element (PE) and relies on the host CPU for data exchange. In contrast, the pipeline-based architecture divides tasks into multiple sequential stages, with each stage executed by dedicated hardware, significantly improving frames per second (FPS) performance. Despite the notable efficiency gains achieved by these accelerators for standard ViTs, they are often inadequate when applied to MoE-ViTs, due to the inher- ent differences in computational demands. Algorithmic Limitations: Achieving a balance between hardware-friendly quantization algorithms and maintaining accuracy remains a significant challenge. Complex quantizers, e.g., FQViT [12], are proposed to minimize precision loss, w
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization
- expert count and top-k/static task gate
- search-only vs track-only expert specialization
- shared backbone vs mode-specific head
- exact softmax vs approximated softmax
- Taylor/linear attention vs exact attention
- fixed sparse token pattern vs dense token pattern
- URAM/BRAM/LUTRAM binding
- memory banks 8/16/32
- parallelism factor 8/16/32
- FIFO depth and double buffering

## 9. HGTXR 적용 판단
- search/track mode별 expert/head 분기 실험에 적합하다. 단, routing은 정적이고 bounded여야 HGTXR paper scope를 지킨다.

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
