---
source_type: paper
source_name: Edge-MoE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/Edge-MoE/analysis.md -->

# Edge-MoE Detailed Paper Analysis

- title: Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts
- url: https://arxiv.org/abs/2305.18691
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/Edge-MoE.pdf`
- related_codebases: `Edge-MoE`

## 1. 기존 방법의 문제점
- 요약: MTL ViT는 단일 task에도 전체 모델을 활성화해 연산/메모리 낭비가 크고, MoE-ViT는 FPGA 배치와 bandwidth가 까다롭다.

### Paper Evidence: Abstract
```text
Abstract—The computer vision community is embracing two promising
                                         learning paradigms: the Vision Transformer (ViT) and Multi-task Learn-
                                         ing (MTL). ViT models show extraordinary performance over traditional
                                                                                                                                             Raw
                                         convolution networks but are commonly recognized as computation-
arXiv:2305.18691v2 [cs.AR] 13 Sep 2023




                                                                                                                                            image
                                         intensive, especially the self-attention with quadratic complexity. MTL
                                         uses one model to infer multiple tasks with better performance by                                  Output
                                         enforcing shared representation among tasks, but a huge drawback is                                Image
                                         that, most MTL regimes require activation of the entire model even
                                         when only one or a few tasks are needed, causing significant computing
                                         waste. M3 ViT is the latest multi-task ViT model that introduces mixture-
                                         of-experts (MoE), where only a small portion of subnetworks (“experts”)
                                         are sparsely and dynamically activated based on the current task. M3 ViT            Task switch
```

## 2. 제안하는 방법
- 요약: task-level sparsity와 multi-task MoE를 FPGA accelerator에 맞춰 reordering, single-pass softmax, low-cost GELU, unified compute unit으로 구현한다.

### Paper Evidence: Method Section
```text
Approach           Data Load      Latency      Bandwidth Memory                                         b
                                                                                          −20                                        Rounds to zero
                                         N2
  w/o reorder         N2 + N                         ∼p         p+1
                 2                   2
                                          p                                                      −4          −2          0             2            4
                N                  N
  w/ reorder     p
                     +N +p−1        p
                                         +p−1        ∼1         p+1
                                                                                                                       Bias b

                                                                               Fig. 6. The usable range (marked in green) of the function exp(x − b) for
arbitrary parallelism with a constant bandwidth for the input                  a 32-bit signed fixed-point datatype with 10 integer bits for different values
matrices. Specifically, for a desired parallelism factor p (4 in the           of the bias b. Too large b will result in rounding to zeros, while too small b
                                                                               will result in overflow; optimal b value depends on the input x.
figure), we first cache one batch of p tokens of Qi in a local buffer
(e.g., Q1 to Q4 in the first batch); then, during each iteration, we
load a new Kj token and multiply it with the p tokens of Q in the              affects the model accuracy since softmax is extensively used in
local buffer. The memory required is p buffers for Q token and 1               Transformer and ViT models.
buffer for K token, also p + 1 buffers in total.                                  To address the overflow problem to compensate accuracy loss,
   Notice that some certain tokens of Q are not aligned with the start         we propose a compensation bias, denoted by b, to re-adjust the
of the K matrix, such as Q2 in the figure. Thus certain outputs are            output range for exp(xi ) and exp(xj ) while computing PNexp(x i)
                                                                                                                                                    .
                                                                                                                                       j=1 exp(xj )
initially “missing,” lik
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: task에 따라 sparse expert pathway만 실행하고, attention reordering과 patch reordering으로 memory access를 줄인다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
unified linear layer stands FPGA design without our proposed techniques. We find it difficult to out: it takes the largest portion of LUT resources, but as a result, it make a fair comparison with prior works for two main reasons. First, greatly accelerates the attention linear layers, ViT blocks, and MoE we target ViT acceleration from a different angle than prior works and blocks, which take only 35% of the overall latency combined. 9 TABLE V Ablation study of our proposed techniques. All the latency, resource, and accuracy values are measured on-board. The baseline is a fully functional M3 ViT accelerator design without our proposed techniques. Applying all six techniques can result in more than 18× speedup and no accuracy drop. Hardware resources Sem. seg. Depth est. MTL acc. Architecture Latency (Speedup) BRAM DSP LUT FF (mIoU ↑) (RMSE ↓) gain ∆m (↑) Baseline w/o our proposed techn
```
- keyword `sparsity`:
```text
Edge-MoE: Memory-Efficient Multi-Task Vision Transformer Architecture with Task-level Sparsity via Mixture-of-Experts Rishov Sarkar1 , Hanxue Liang2 , Zhiwen Fan2 , Zhangyang Wang2 , Cong Hao1 1 School of Electrical and Computer Engineering, Georgia Institute of Technology 2 School of Electrical and Computer Engineering, University of Texas at Austin rishov.sarkar@gatech.edu, lhx92505991@gmail.com, {zhiwenfan, atlaswang}@utexas.edu, callie.hao@ece.gatech.edu Abstract—The computer vision community is embracing two promising learnin
```
- keyword `parallelism`:
```text
with a rich collection of architectural innovations. First, for general Transformer/ViT where a single compact algorithm can simultaneously learn many models, we propose (1) a novel reordering mechanism for self-attention, different tasks with a much smaller model size than single-task which reduces the bandwidth requirement from proportional to constant learning (STL) [31]. In addition, MTL can learn an improved feature regardless of the target parallelism; (2) a fast single-pass softmax representation by sharing representations and utilizing regularizations approximation; (3) an accurate and low-cost GELU approximation, which between related tasks [20], [34]. MTL is important and yet challeng- can significantly reduce the computation latency and resource usage; and (4) a unified and flexible computing unit that can be shared by almost ing for real-world applications, especially when th
```

## 4. 하드웨어 아키텍처
- 요약: unified flexible computing unit이 대부분 layer를 공유하며 softmax/GELU approximation이 control/datapath를 단순화한다.

### Paper Evidence: Architecture / Implementation
```text
Architecture with Task-level Sparsity via Mixture-of-Experts
                                                                     Rishov Sarkar1 , Hanxue Liang2 , Zhiwen Fan2 , Zhangyang Wang2 , Cong Hao1
                                                                        1
                                                                       School of Electrical and Computer Engineering, Georgia Institute of Technology
                                                                       2
                                                                         School of Electrical and Computer Engineering, University of Texas at Austin
                                                     rishov.sarkar@gatech.edu, lhx92505991@gmail.com, {zhiwenfan, atlaswang}@utexas.edu, callie.hao@ece.gatech.edu


                                            Abstract—The computer vision community is embracing two promising
                                         learning paradigms: the Vision Transformer (ViT) and Multi-task Learn-
                                         ing (MTL). ViT models show extraordinary performance over traditional
                                                                                                                                             Raw
                                         convolution networks but are commonly recognized as computation-
arXiv:2305.18691v2 [cs.AR] 13 Sep 2023




                                                                                                                                            image
                                         intensive, especially the self-attention with quadratic complexity. MTL
                                         uses one model to infer multiple tasks with better performance by                                  Output
                                         enforcing shared representation among tasks, but a huge drawback is                                Image
                                         that, most MTL regimes require activation of the entire model even
                                         when only one or a few tasks are needed, causing significant computing
                                         waste. M3 ViT is the latest multi-task ViT model that introduces mixture-
                                         of-experts (MoE), where only a small portion of subnetworks (“experts”)
                                         are sparsely and dynamically activated based on the current task. M3 ViT
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
mixture- of-experts (MoE), where only a small portion of subnetworks (“experts”) are sparsely and dynamically activated based on the current task. M3 ViT Task switch achieves better accuracy and over 80% computation reduction and paves between depth the way for efficient real-time MTL using ViT. estimation and Despite the algorithmic advantages of MTL, ViT, and even M3 ViT, segmentation there are still many challenges for efficient deployment on FPGA. For instance, in general Transformer/ViT models, the self-attention is known as computational intensive and requires high bandwidth. In addition, softmax operations and the activation function GELU are extensively used, which unfortunately can consume more than half of the entire Fig. 1. On-board implementation demo for our MTL ViT accelerator with FPGA resource (LUTs). In the M3 ViT model, the promising MoE guaranteed functionality and per
```
- keyword `ZCU102`:
```text
ive and requires high bandwidth. In addition, softmax operations and the activation function GELU are extensively used, which unfortunately can consume more than half of the entire Fig. 1. On-board implementation demo for our MTL ViT accelerator with FPGA resource (LUTs). In the M3 ViT model, the promising MoE guaranteed functionality and performance. The entire M3 ViT runs on the mechanism for multi-task exposes new challenges for memory access ZCU102 FPGA board; outputs are streamed to laptop only for visualization. overhead and also increases resource usage because of more layer types. To address these challenges in both general Transformer/ViT models and the state-of-the-art multi-task M3 ViT with MoE, we propose Edge- Meanwhile, Multi-task Learning (MTL) is a promising scenario MoE, the first end-to-end FPGA accelerator for multi-task ViT with a rich collection of architectural inno
```
- keyword `Vitis`:
```text
erally specific gating network. applicable and effective when applied to any ViT model. V. E XPERIMENTS C. Latency and Resource Breakdown We conduct several on-board, verified experiments to demonstrate To understand which parts of our model are the most expensive the effectiveness of our proposed methods. All on-board code is and which take the most time to compute, we measure the on-board implemented through High-Level Synthesis through Xilinx Vitis HLS latency and resource usage of the different components of our M3 ViT 2021.1. We deploy bitstreams to Xilinx ZCU102 FPGA and use the implementation. Figure 12 displays our findings. PYNQ library for host code. The clock frequency of our design is 300 Even at 4× parallelism, the attention multiplications Q × K and MHz. All experiments use the Cityscapes dataset [5], with images of M ′ × V take half of the total computation time, demonstra
```

## 5. 실험 방법
```text
evaluation.                                                                 rely on model compression or re-training and, in fact, are orthogonal
                                                                              to many of the compression techniques proposed in prior works,
             II. P RELIMINARY AND R ELATED W ORK
                                                                              which can be applied together with our proposed techniques.
A. Vision Transformers and M3 ViT                                                In addition, instead of focusing on the entire Transformer/ViT
   The Vision Transformer (ViT) is first proposed by Dosovitskiy              model, Zhang et al. [35] propose an FPGA-based self-attention ac-
et. al [7] by adapting Transformers in Natural Language Processing            celerator by weight pruning, while Lu et. al [19] propose an systolic-
(NLP) to processing images for computer vision tasks. Similar to              array based design for attention layer implementation. By contrast,
the “tokens” in Transformers, each image is first split into “patches”,       we propose an end-to-end fully functional ViT model implementation,
where each patch is of size P × P and has P 2 pixels. Then patches            which exposes additional challenges that are concealed by looking at
will be flattened into vectors, projected to linear embeddings, and fed       only part of the model.
into to a standard transformer encoder in sequential order, usually              For multi-task learning, the state-of-the-art M3 ViT [16] excels
with positional embeddings. Each block of the encoder is usually              in both high algorithm accuracy and low computation complexity,
composed of a self-attention layer, normalization layers, a multi-layer       thanks to the MoE mechanism. It suggests a hardware-friendly
perceptron (MLP) layer, and activation layers. In the self-attention          method for MoE computation, but unfortunately, it only provides
layer (blue block in Fig. 3 left), we denote the input patch embeddings       an optimistic estimation using FPGA without any actual hardware
by X ∈ RL×d where d is the patch embedding dimension and L is                 implementation. Therefore, our work is, to our best knowledge, the
the patch count. Three matrices, Q, K, and V , can be computed as:            first FPGA accelerator for a multi-task ViT model with MoE.
Q = W Q X, K = W K X, V = W V X, where W Q , W K , and W V
                                                                                                        III. C HALLENGES
are weights. Finally, output Y = softmax(QK T )V ∈ RL×d . This
self-attention has a quadratic complexity in the softmax, where each             Despite the algorithmic advantages of ViT models and the com-
patch needs to compute its attention score with all n patches.                putational sparsity of M3 ViT for multi-task learning, there are still
   On top of ViT,
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
of the IEEE Conference on Computer using FPGA,” in Proceedings of the ACM/IEEE International Sympo- Vision and Pattern Recognition, 2018, pp. 675–684. sium on Low Power Electronics and Design, ser. ISLPED ’20. New [33] L. Yuan, Y. Chen, T. Wang, W. Yu, Y. Shi, Z.-H. Jiang, F. E. Tay, J. Feng, York, NY, USA: Association for Computing Machinery, Aug. 2020, pp. and S. Yan, “Tokens-to-token vit: Training vision transformers from 175–180. scratch on imagenet,” in Proceedings of the IEEE/CVF International [15] Z. Li, M. Sun, A. Lu, H. Ma, G. Yuan, Y. Xie, H. Tang, Y. Li, M. Leeser, Conference on Computer Vision, 2021, pp. 558–567. Z. Wang et al., “Auto-vit-acc: An fpga-aware automatic acceleration [34] A. R. Zamir, A. Sax, W. Shen, L. J. Guibas, J. Malik, and S. Savarese, framework for vision transformer with mixed-scheme quantization,” “Taskonomy: Disentangling task transfer learning,” in Pr
```
- keyword `Cityscapes`:
```text
time to compute, we measure the on-board implemented through High-Level Synthesis through Xilinx Vitis HLS latency and resource usage of the different components of our M3 ViT 2021.1. We deploy bitstreams to Xilinx ZCU102 FPGA and use the implementation. Figure 12 displays our findings. PYNQ library for host code. The clock frequency of our design is 300 Even at 4× parallelism, the attention multiplications Q × K and MHz. All experiments use the Cityscapes dataset [5], with images of M ′ × V take half of the total computation time, demonstrating the size 128×256 split into patches of size 16×16. necessity of accelerating this computation. Our main comparison is with CPU and GPU baselines and the Additionally, the effectiveness of our unified linear layer stands FPGA design without our proposed techniques. We find it difficult to out: it takes the largest portion of LUT resources, but as
```
- keyword `DeiT`:
```text
n layer computation [4], [7], [35]: sourced hardware design using High-Level Synthesis (HLS). Fig. 1 with N tokens in total, each token needs to compute attention factors depicts our on-board implementation with real-time performance with all N tokens including itself. The self-attention computation on an autonomous driving dataset; a full video clip is available on has became a major bottleneck even for models with smaller size GitHub.2 such as DeiT [29] and T2T-ViT [33]. Second, although M3 ViT • For general Transformer/ViT models with common challenges proposes sparsely activated MoE with largely reduced computation, (e.g., heavy self-attention, softmax, and GELU), we propose a it introduces new challenges to memory access: since experts are collection of innovative techniques, including: ➊ a novel attention dynamically determined on-the-fly, one has to either store all the reordering
```
- keyword `dataset`:
```text
an detection, segmentation, etc., where MTL is hardware design, which achieves 2.24× and 4.90× better energy efficiency expected to deliver real-time performance for each task as well as comparing with GPU (A6000) and CPU (Xeon 6226R), respectively. A swift task switch. Therefore, real-time MTL with swift task switch is real-time video demonstration of our accelerated multi-task ViT on an in great demand for future AI systems. autonomous driving dataset is available on GitHub,1 together with our FPGA design using High-Level Synthesis, host code, FPGA bitstream, While there is a rich amount of work exploring ViT and MTL sepa- and on-board performance results. rately, applying MTL to ViT also has emerged with attractive results. One prevailing type of MTL architectures [8], [17], [21], [25] adopt a I. I NTRODUCTION shared backbone with independent task-specific head, while another type of
```

## 7. 실험 결과
- keyword `speedup`:
```text
y accurate and low-cost GELU approximation with extremely low used after each attention layer (as opposed to convolution networks hardware resource; and ➍ a unified and flexible computing unit that where softmax is only used in the final output), which requires a huge can be shared by almost all linear layers, which drastically reduces amount of non-linear FPGA-unfriendly computations that consume the resource usage and thus leads to significant speedup. The pro- a large amount of resource and execution time (more details in posed techniques can be directly applied to any Transformer/ViT Sec. III-A2). Fourth, in many state-of-the-art Transformers and ViT model for resource and latency reduction. models, a new type of activation function, GELU (Gaussian Error • For the advanced multi-task ViT model with mixture-of-expert, Linear Unit) [11], has been widely used to improve training efficie
```
- keyword `energy`:
```text
h real-time requirement. For instance, autonomous driving [13] patch reordering method to completely eliminate any memory access requires many tasks executed on the same platform, such as lane overhead. Third, we deliver on-board implementation and measurement on Xilinx ZCU102 FPGA, with verified functionality and open-sourced detection, pedestrian detection, segmentation, etc., where MTL is hardware design, which achieves 2.24× and 4.90× better energy efficiency expected to deliver real-time performance for each task as well as comparing with GPU (A6000) and CPU (Xeon 6226R), respectively. A swift task switch. Therefore, real-time MTL with swift task switch is real-time video demonstration of our accelerated multi-task ViT on an in great demand for future AI systems. autonomous driving dataset is available on GitHub,1 together with our FPGA design using High-Level Synthesis, host code,
```
- keyword `accuracy`:
```text
shared representation among tasks, but a huge drawback is Image that, most MTL regimes require activation of the entire model even when only one or a few tasks are needed, causing significant computing waste. M3 ViT is the latest multi-task ViT model that introduces mixture- of-experts (MoE), where only a small portion of subnetworks (“experts”) are sparsely and dynamically activated based on the current task. M3 ViT Task switch achieves better accuracy and over 80% computation reduction and paves between depth the way for efficient real-time MTL using ViT. estimation and Despite the algorithmic advantages of MTL, ViT, and even M3 ViT, segmentation there are still many challenges for efficient deployment on FPGA. For instance, in general Transformer/ViT models, the self-attention is known as computational intensive and requires high bandwidth. In addition, softmax operations and the act
```
- keyword `latency`:
```text
uces the bandwidth requirement from proportional to constant learning (STL) [31]. In addition, MTL can learn an improved feature regardless of the target parallelism; (2) a fast single-pass softmax representation by sharing representations and utilizing regularizations approximation; (3) an accurate and low-cost GELU approximation, which between related tasks [20], [34]. MTL is important and yet challeng- can significantly reduce the computation latency and resource usage; and (4) a unified and flexible computing unit that can be shared by almost ing for real-world applications, especially when the model will be all computational layers to maximally reduce resource usage. Second, deployed in an environment with limited computational capability but for the advanced multi-task M3 ViT with MoE, we propose a novel with real-time requirement. For instance, autonomous driving [13] patch reorde
```

## 8. 실험 옵션 / Ablation 축
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
- HGTXR search/track mode가 task split이므로 P1 실험으로 강하게 적합하다. 정확도 개선 가능성이 있으나 C3b 기본 signoff와 분리해야 한다.

### 우선순위 판정
- recommended_priority: `P1`
- C3b board-smoke gate는 유지한다.
- HLS/Vivado 증거가 생기기 전에는 resource matrix 완료 variant로 승격하지 않는다.

## 10. HGTXR 실험 설계로 변환
| 단계 | 작업 | 산출물 | 승격 조건 |
|---|---|---|---|
| SW | 알고리즘을 Python/C++ reference에 반영 | accuracy/bit-exact report | 정확도 개선 또는 무손실 |
| HLS | parameterized macro/defines로 구현 | csim/csynth report | csynth.xml 생성 및 resource gate 통과 |
| Vivado | C3b successor overlay로 구현 | routed timing/power/util | WNS >= 0, resource 정책 유지 |
| PYNQ | smoke run | board JSON | validator pass |
