---
source_type: paper
source_name: TATAA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/TATAA/analysis.md -->

# TATAA Detailed Paper Analysis

- title: TATAA: Programmable Mixed-Precision Transformer Acceleration with a Transformable Arithmetic Architecture
- url: https://arxiv.org/abs/2411.03697
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/TATAA.pdf`
- related_codebases: `TATAA`

## 1. 기존 방법의 문제점
- 요약: linear layer는 low-bit가 가능하지만 nonlinear transformer op는 precision 민감도가 높아 end-to-end low-bit accelerator가 어렵다.

### Paper Evidence: Introduction / Motivation
```text
1       Introduction
Since its introduction in 2017, the Transformer model [1] and its variations have rapidly risen to the forefront of
modern deep learning architectures. Unlike previous-generation convolutional neural networks (CNNs) that were based
predominantly on linear operations, modern transformer models are increasingly reliance on high-precision non-linear
operations in their designs. For instance, the self-attention mechanism of a transformer model is typically based on the
SoftMax function, which has been demonstrated to require high precision computation in order to achieve a model’s
accuracy [2]. Normalization layers such as LayerNorm [3] or root mean square normalization (RMSNorm) [4], require
complex nonlinear operations on data that cannot easily be fused into preceding linear layers. Finally, sophisticated
activation functions such as GELU [5], SiLU [6] and SwiGLU [7] are often used in transformer models which require
precise computation, unlike in CNNs.
To address the need to approximate these non-linear functions with high precision and high performance, specialized
hardware modules have previously been extensively explored [8, 9, 10, 11]. Yet, customized hardware must be designed
for every non-linear function that is being employed in a model, which is impractical when new non-linear operations
for transformers are still actively being developed in this rapidly-evolving field [12, 13]. Other researchers have focused
on quantizing such non-linear functions into low-bitwidth fixed point formats in order to reduce the computation
complexity [14, 15, 16, 17]. Due to the outliers in transformers [18, 19], retraining is generally required for such
quantization to maintain good accuracy. However, the large size of modern transformer models, data availability and
privacy concerns, have all made such retraining method impractical in most real-world scenarios. Besides, existing
accelerators either rely on individual and specific non-linear function units [20, 21, 22], or attempt to handle non-linear
functions with general arithmetic logic units [17]. Both strategies often lead to increased hardware overhead, reduced
hardware efficiency, and it compli
```

## 2. 제안하는 방법
- 요약: int8 systolic mode와 bfloat16 SIMD mode를 runtime 전환하는 transformable arithmetic architecture를 제안한다.

### Paper Evidence: Method Section
```text
Algorithm 1 Fast Inverse Square Root
Input: Input bfloat16 number y
Output: The inverse square root result √1y
 1: yint = y.view(int16)                                   ▷ Does not change data bits, only changes the data format it refers to
 2: tint = 0x5f37 − (yint >> 1)                                                  ▷ 0x5f37 is the magic number in int16 [58]
 3: t = tint .view(bf loat16)
 4: √1y = y · (1.5 − (y · 0.5 · t2 )             ▷ Define t2 as fpapp operation. In TATAA, fpapp is one of the basic operations



fpadd operations. Hence, we designate it as an approximated calculation for the square root and division, abbreviated as
fpapp. This term will be utilized in the subsequent discussion in this paper.
                                                       (
                                             x    1        x · √1y · √1y , y > 0
                                               =x· =                                                                         (5)
                                             y    y        x · √−1
                                                                 −y
                                                                     · √−1
                                                                         −y
                                                                            ,y < 0

Based on transformable arithmetic, all basic floating-point operations can be transformed to a series of integer atom
operations. As int8 has been the most commonly used format for linear layers quantization, we choose bfloat16 as
the high-precision format for non-linear functions, featuring an 8-bit exponent and an 8-bit mantissa. The bfloat16
format has been extensively employed in the deep learning field for many years and has developed into a well-established
standard for both training and inference [59]. The bitwidth of this unique floating-point format is perfectly aligned with
the widely used int8, for both the exponent and the mantissa. Consequently, based on the analysis aforementioned, it
is feasible to repurpose standard int8 processing units for fundamental bfloat16 operations, such as fpmul, fpadd,
and fpdiv discussed in this section. We also find that the most commonly used architecture for MatMul, systolic array,
can actually match the vectorized floating-point execution in terms of computation and data layout. We will further
explain the details of hardware design, ISA support, and workload mapping in the following sections.

4     Hardware Design
4.1   System Architecture

Figure 2 illustrates the prop
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: linear op는 int8 PTQ, nonlinear op는 bf16 approximation으로 mapping하고 compiler가 instruction/dataflow를 생성한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `parallelism`:
```text
sed in Section 4. 5.1 Instruction Set Architecture (ISA) To better decouple hardware and software, we have developed a customized Instruction Set Architecture (ISA). Our software system can map linear layers in int8 and non-linear operations with a high-precision approximation in bfloat16. Table 2 presents the simple ISA design in TATAA. In an ISA-level perspective, the controller is able to detect data dependencies and exploit instruction-level parallelism (ILP) to improve throughput performance. As an example, the double buffer optimization allows the parallel execution of the LOAD.M and MATMUL instructions. Furthermore, the previously mentioned data layout conversion with specific write-back addresses is incorporated into the STORE.M and STORE.V instructions, offering sufficient flexibility and comprehensive support for inference runtime. After compilation, all the runtime instruction
```
- keyword `buffer`:
```text
& fp vectors) B A C B A C … … S0 S0 S0 S0 … S0 MUL ADD R L MUL ADD R … L S1 S1 S1 S1 … S1 4 M P M P RFX (int8 matrix only) … … … … … Const Const S3 S3 S3 S3 … S3 FPU Bottom-Logic Bottom-Logic RFY 1 (fp vectors) DMPU 0 … W columns in total … … (b) Dual Mode Processing Unit (DMPU) Design Mode MUX … Matrix Input from RFY Vector from DMRFY0 Run-time Transformable Dual DMPU 1 DMPU0 DMPU0 … … Mode Reconfiguration … Matrix RFX … … Controller Dual-mode Buffers (DMBs) Vector Output DMPU1 … Vector from RFY1 Quantization & Layout Conversion … DMPU1 … … … … … … … Memory Write Channel Matrix Output External Memory Linear Layers (int8 MatMul) Non-linear Layers (bfloat16) (a) TATAA Core Architecture (c) Run-time Transformable Architecture Figure 2: TATAA hardware architecture and dual-mode processing unit (DMPU) design. The TATAA core can be configured for two kinds of workload during runtime, to supp
```
- keyword `batch size`:
```text
esults [21, 51, 55, 52, 68]. Our TATAA architecture exhibits comparable overhead across three resource types, with only 10.5% FFs, and no DSPs overhead especially. As exceptions, EFA-Trans [51] reports linear operation units including SoftMax, and in the work by Lu et al. [68], where LUTs are employed for linear computations and the DSP overhead for non-linear operations reaches 100%. 6.4 TATAA Runtime Analysis We choose DeiT-Small (DeiT-S) with batch size 16 and BERT with sequence length 128 & batch size 32 to measure layer latency with its workload size Figure 14. The linear MatMul layers (such as QKV-GEN, QK-MUL, SV-MUL, etc.) are the primary contributors to the transformer workload when measured in terms of GOP (giga operations) or GFLOP (giga floating-point operations), and therefore heavily influence total latency. Among these linear layers, QK-MUL and SV-MUL are slightly less effi
```

## 4. 하드웨어 아키텍처
- 요약: systolic array mode와 SIMD vector mode를 같은 arithmetic fabric에서 전환한다.

### Paper Evidence: Architecture / Implementation
```text
architecture can be transformed between a systolic array for int8 matrix multiplications and a SIMD-like vectorized
bfloat16 computing unit. In particular, the proposed TATAA architecture employs a single type of processing unit,
which is reused for all run-time operations, leveraging the bit field patterns of bfloat16. This design choice minimizes
hardware overhead and maximizes flexibility compared to previous studies. By minimizing the overhead for run-time
reconfiguration, the proposed transformable architecture ensures the high hardware processing density necessary to
deliver the highest performance on FPGAs with limited resources. Finally, a compilation framework is developed that
maps the user-provided transformer models to the custom instruction set architecture (ISA) of the TATAA processor
cores to facilitate all operations in both linear and non-linear layers.


                                                                           2

                                                                                     A PREPRINT - N OVEMBER 7, 2024


To the best of our knowledge, TATAA is the first FPGA-based acceleration framework for transformer inference that
integrates floating-point non-linear functions into integer-based linear processing units. It is programmable and is ready
to support emerging transformer models with potentially new non-linear functions. Our experimental results indicate
that when simulating model performance with a hybrid data format for transformer inference, TATAA achieves only a
minimal accuracy reduction, with 0.14 % to 1.16 % decrease across all evaluated models compared to the original pre-
trained model in single-precision floating point (fp32). Additionally, the FPGA accelerator reaches a peak throughput
of 2935.2 giga-operations-per-second (GOPS) for int8 linear operations and a peak throughput of 169.8 giga-floating-
point-operations-per-second (GFLOPS) when the processor is configured for bfloat16 non-linear operations at a clock
frequency of 225 MHz. Compared to related studies, TATAA achieves up to 1.45× higher throughput and 2.29× higher
throughput efficiency on DSP blocks. With the transformable architecture for non-linear functions, our implementation
achieves 4.25× lower latency for these complex bfloat16 operations compared with other works, while supporting
flexible and general compilation for emerging functions. Our end-to-end compilation framework also presents optimal
mapping from non-linear functions to hardware ISA by appropriate approximation schemes and efficient dataflow
control. Moreove
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
or vectorized bfloat16 operations. An end-to-end compiler is presented to enable flexible mapping from emerging transformer models to the proposed hardware. Experimental results indicate that our mixed-precision design incurs only 0.14 % to 1.16 % accuracy drop when compared with the pre-trained single-precision transformer models across a range of vision, language, and generative text applications. Our prototype implementation on the Alveo U280 FPGA currently achieves 2935.2 GOPS throughput on linear layers and a maximum of 189.5 GFLOPS for non-linear operations, outperforming related works by up to 1.45× in end-to-end throughput and 2.29× in DSP efficiency, while achieving 2.19× higher power efficiency than modern NVIDIA RTX4090 GPU. * These authors contributed equally to this work A PREPRINT - N OVEMBER 7, 2024 Pretrained Models: Computation Inside one Transformer Block Transformer Bl
```
- keyword `ZCU102`:
```text
f GOPS/DSP and GOPS/kLUT, comparing TATAA with several FPGA-based acceleration frameworks 19 A PREPRINT - N OVEMBER 7, 2024 Table 7: Hardware Performance Comparison with Relative FPGA-based Accelerators for Transformer Models Data End2end FPGA FPGA Utilization Freq. Power Eval. Throughput Throughput DSP Work Formats‡ Support Platform LUT(k) FF(k) BRAM DSP (MHz) (W) Models Inf./sec§ (GOPS) Efficiency Auto-ViT- DeiT-S 99.7 907.8 0.585 fxp, fp32 No ZCU102 185.0 - - 1552 150 9.6 Acc [42] DeiT-B 34.0 1181.5 0.761 Huang ViT-S 89.7 762.7 0.601 int8, int8 Yes ZCU102 144.5 168.0 648 1268 300 29.6 et al. [21] ViT-T 245.3 616.1 0.486 BERT 81.9 - 0.035† HPTA[71] int8, int8 Yes ZCU102 209.9 368.4 345 2307 200 20.0 Swin-T 148.8 - 0.065† NPE [17] int16, fxp Yes VCU118 192.4 351.1 369 2020 200 20.0 BERT 36.8 - 0.018† FTRANS [46] fp16, fp32 Yes VCU118 451.1 506.6 - 6531 - - RoBERTa 94.25 - 0.014 ViA [20]
```
- keyword `U280`:
```text
ode for vectorized bfloat16 operations. An end-to-end compiler is presented to enable flexible mapping from emerging transformer models to the proposed hardware. Experimental results indicate that our mixed-precision design incurs only 0.14 % to 1.16 % accuracy drop when compared with the pre-trained single-precision transformer models across a range of vision, language, and generative text applications. Our prototype implementation on the Alveo U280 FPGA currently achieves 2935.2 GOPS throughput on linear layers and a maximum of 189.5 GFLOPS for non-linear operations, outperforming related works by up to 1.45× in end-to-end throughput and 2.29× in DSP efficiency, while achieving 2.19× higher power efficiency than modern NVIDIA RTX4090 GPU. * These authors contributed equally to this work A PREPRINT - N OVEMBER 7, 2024 Pretrained Models: Computation Inside one Transformer Block Transform
```

## 5. 실험 방법
```text
results accumulate across DMPUs. We choose to deploy the output stationary dataflow for matrix multiplication. In
such an execution flow, the X and Y matrices go through the systolic array in the X (horizontal) and Y (vertical)
directions, respectively. Registers L and R are responsible for horizontal and vertical data passing, while the bottom
register directly accepts the data from the top and sends them to the next PE. After MatMul finishes, the results stored
in register P will be sent to the corresponding dual-mode buffer (DMB). The intermediate sums are accumulated in
the int16 format and subsequently quantized to either int8 or bfloat16 before being saved to external memory,
depending on the format required by the subsequent layer. The static scaling factors are pre-loaded to the quantization
unit in the TATAA core before MatMul starts.
When the TATAA architecture is set in bfloat16 mode, it can execute three basic operations: multiplication (fpmul),
addition (fpadd) and the approximation step for the inverse square root in Equation (5) and Algorithm 1 to support
(0x5f 37 − (yint >> 1))2 (fpapp) operation. These operations can be assigned directly to the 4 pipeline stages in the 4
rows of integer-based PE, as illustrated in Figure 5. Thanks to the arithmetic analysis from Equation (3) to Equation (5),
we can convert bfloat16 operations into a sequence of integer operations. The integer multiplier and adder (MUL


                                                                                9

                                                                                                                            A PREPRINT - N OVEMBER 7, 2024


                        𝐷𝑚𝑎𝑡 𝐷𝑚𝑎𝑡                     RFYa
                                                             16 ∙ 𝑊-bit        RFYb
                                                                                      16 ∙ 𝑊-bit                                 DMPU
                                                                                                     Addr                               …
                                          0                                                              …
                                                                                                                                        RFY




                               … …
                   … …




                                                   𝐷𝑚𝑎𝑡
            8𝑏
                                                          RVY a 2                  RVY b 2               2
                                                                    RMY0                      RMY1




                                                                                                             RFX
                                                          RVY a 1                  RVY b 1               1                                    …
                                          1
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
ls for the 8 TATAA cores and each AXI channel has 256-bit memory bitwidth. We have chosen a range of transformer models to evaluate the accuracy of quantization and their runtime performance in tasks such as image classification, text classification, and text generation. The selected models are listed below and their feature dimensions are shown in Table 3. • For Vision Transformers (ViT), we select DeiT[24] and Swin Transformer (Swin) [25] with ImageNet-1k [63] dataset for the image classification. • BERT [26], as a widely used language model, is also selected for evaluation based on the GLUE benchmark including different tasks [64]. • We also evaluate other popular language models, GPT-2 [65] and OPT [27], for the text generation task with LAMBADA [66] and WikiText-2 datasets. • To evaluate the general support of TATAA for non-linear functions in transformer models, we also select stat
```
- keyword `LLaMA`:
```text
INT - N OVEMBER 7, 2024 Pretrained Models: Computation Inside one Transformer Block Transformer Block QKV- V 𝑄, 𝐾, 𝑉 = 𝑁𝑜𝑟𝑚 𝑋𝑛 ∙ [𝑊𝑄 , 𝑊𝐾 , 𝑊𝑉 ] QK- SV- ATT-PROJ … K Normalization GEN Q MUL SoftMax MUL 𝐴𝑛 = 𝑋𝑛 + 𝑆𝑜𝑓𝑡𝑀𝑎𝑥(𝑄𝐾 𝑇 )𝑉 ∙ 𝑊𝑃𝑅𝑂𝐽 X … Add Normalization FFN1 Activation FFN2 Add 𝑋𝑛+1 = 𝐴𝑛 + 𝐴𝑐𝑡(𝑁𝑜𝑟𝑚 𝐴𝑛 ∙ 𝑊𝐹𝐹𝑁1 ) ∙ 𝑊𝐹𝐹𝑁2 Various Non-linear Functions in Transformer Linear matrix multiplication Non-linear functions and residual adder ViT OPT GPT Llama# … Approximation Norm. L/Norm L/Norm L/Norm RMSNorm … Integer operations Transformable Act. GELU RELU GELU* SwiGLU … (Multiply-accumulation) Architecture Bfloat16 operations *Can be other variants, e.g., SiLU TATAA Deployment on FPGA (MUL, ADD, etc.) #Concludes Rotary Positional Embeddings as non-lienar functions Figure 1: Illustration of a typical Transformer block, and how TATAA maps different operations in Transformers including linear M
```
- keyword `DeiT`:
```text
computation to be performed on-chip, thereby maximizing performance. Furthermore, to mitigate potential data hazards during this consistent computation, a two-by-two accumulation method is employed, which adds every two lines of input vectors and stores the results 14 A PREPRINT - N OVEMBER 7, 2024 Table 3: Selected Transformer Models or Non-linear Functions in the Experiments Model Type # Blocks # Heads Hidden Size MLP Size Non-linear Functions Deit-S Encoder 12 6 384 1536 Deit-B Encoder 12 12 768 3072 SoftMax † † † † Swin-T Encoder {2,2,6,2} {3,6,12,24} {96,192,384,768} {562 , 282 ,142 ,72 } LayerNorm BERT Encoder 12 12 768 3072 GELU GPT2 Decoder 24 16 1024 4096 OPT-1.3B# Decoder 24 16 2048 8192 SoftMax, LayerNorm, ReLU # Llama-7B Decoder 32 32 4096 11008 SoftMax, RMSNorm, SwiGLU ChatGLM2# Decoder 28 32 4096 13696 SoftMax, RMSNorm, SiLU † {. . . } shows dimension variance of each stage
```
- keyword `dataset`:
```text
TQ approach is more practical in transformer applications and is applied in our quantization framework [33, 30]. In TATAA, we develop the quantization emulator based on a hardware matching style instead of fake quantization to get more convincing results, following the HAWQ setups [34]. Besides, TATAA can integrate existing PTQ schemes like FQ-ViT [30] and SmoothQuant [33]. The static PTQ scheme only requires to access a relatively small part of dataset for calibration and getting all the scaling factors (i.e., Sx , Sy , Sz ) before deploying inference. Once we have the scaling factors, our mixed-precision quantization can be done through Equation (1), switching between floating-point and integer numbers for different kinds of layers. 3 A PREPRINT - N OVEMBER 7, 2024 2.2 Non-linear Functions in Transformer Beyond integer-based MatMul layers, transformers require non-linear functions to a
```

## 7. 실험 결과
- keyword `energy`:
```text
ng. Hardware acceleration of fully quantized bert for efficient natural language processing. In 2021 Design, Automation & Test in Europe Conference & Exhibition (DATE), pages 513–516. IEEE, 2021. [45] NVIDIA. Transformer engine documentation. https://docs.nvidia.com/deeplearning/ transformer-engine/user-guide/. [46] Bingbing Li, Santosh Pandey, Haowen Fang, Yanjun Lyv, Ji Li, Jieyang Chen, Mimi Xie, Lipeng Wan, Hang Liu, and Caiwen Ding. Ftrans: Energy-efficient acceleration of transformers using fpga. In Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design, ISLPED ’20, page 175–180, New York, NY, USA, 2020. Association for Computing Machinery. [47] Suyeon Hur, Seongmin Na, Dongup Kwon, Joonsung Kim, Andrew Boutros, Eriko Nurvitadhi, and Jangwoo Kim. A fast and flexible fpga-based accelerator for natural language processing neural networks. ACM Trans. A
```
- keyword `accuracy`:
```text
ansformable arithmetic architecture that supports both formats during runtime with minimal overhead, enabling it to switch between a systolic array mode for int8 matrix multiplications and a SIMD mode for vectorized bfloat16 operations. An end-to-end compiler is presented to enable flexible mapping from emerging transformer models to the proposed hardware. Experimental results indicate that our mixed-precision design incurs only 0.14 % to 1.16 % accuracy drop when compared with the pre-trained single-precision transformer models across a range of vision, language, and generative text applications. Our prototype implementation on the Alveo U280 FPGA currently achieves 2935.2 GOPS throughput on linear layers and a maximum of 189.5 GFLOPS for non-linear operations, outperforming related works by up to 1.45× in end-to-end throughput and 2.29× in DSP efficiency, while achieving 2.19× higher p
```
- keyword `throughput`:
```text
n end-to-end compiler is presented to enable flexible mapping from emerging transformer models to the proposed hardware. Experimental results indicate that our mixed-precision design incurs only 0.14 % to 1.16 % accuracy drop when compared with the pre-trained single-precision transformer models across a range of vision, language, and generative text applications. Our prototype implementation on the Alveo U280 FPGA currently achieves 2935.2 GOPS throughput on linear layers and a maximum of 189.5 GFLOPS for non-linear operations, outperforming related works by up to 1.45× in end-to-end throughput and 2.29× in DSP efficiency, while achieving 2.19× higher power efficiency than modern NVIDIA RTX4090 GPU. * These authors contributed equally to this work A PREPRINT - N OVEMBER 7, 2024 Pretrained Models: Computation Inside one Transformer Block Transformer Block QKV- V 𝑄, 𝐾, 𝑉 = 𝑁𝑜𝑟𝑚 𝑋𝑛 ∙ [𝑊𝑄 ,
```
- keyword `GOPS`:
```text
ns. An end-to-end compiler is presented to enable flexible mapping from emerging transformer models to the proposed hardware. Experimental results indicate that our mixed-precision design incurs only 0.14 % to 1.16 % accuracy drop when compared with the pre-trained single-precision transformer models across a range of vision, language, and generative text applications. Our prototype implementation on the Alveo U280 FPGA currently achieves 2935.2 GOPS throughput on linear layers and a maximum of 189.5 GFLOPS for non-linear operations, outperforming related works by up to 1.45× in end-to-end throughput and 2.29× in DSP efficiency, while achieving 2.19× higher power efficiency than modern NVIDIA RTX4090 GPU. * These authors contributed equally to this work A PREPRINT - N OVEMBER 7, 2024 Pretrained Models: Computation Inside one Transformer Block Transformer Block QKV- V 𝑄, 𝐾, 𝑉 = 𝑁𝑜𝑟𝑚 𝑋𝑛 ∙
```
- keyword `latency`:
```text
d (GOPS) for int8 linear operations and a peak throughput of 169.8 giga-floating- point-operations-per-second (GFLOPS) when the processor is configured for bfloat16 non-linear operations at a clock frequency of 225 MHz. Compared to related studies, TATAA achieves up to 1.45× higher throughput and 2.29× higher throughput efficiency on DSP blocks. With the transformable architecture for non-linear functions, our implementation achieves 4.25× lower latency for these complex bfloat16 operations compared with other works, while supporting flexible and general compilation for emerging functions. Our end-to-end compilation framework also presents optimal mapping from non-linear functions to hardware ISA by appropriate approximation schemes and efficient dataflow control. Moreover, compared to state-of-the-art GPUs, our TATAA architecture outperforms a maximum 2.19× higher power efficiency over
```

## 8. 실험 옵션 / Ablation 축
- software-only reference comparison
- microkernel latency/resource comparison

## 9. HGTXR 적용 판단
- Q4/Q8에서 정확도 병목이 LayerNorm/Softmax/GELU라면 해당 경로만 higher precision으로 올리는 P1 실험에 적합하다.

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
