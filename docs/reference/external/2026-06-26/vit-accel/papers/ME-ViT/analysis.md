---
source_type: paper
source_name: ME-ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/ME-ViT/analysis.md -->

# ME-ViT Detailed Paper Analysis

- title: ME-ViT: A Single-Load Memory-Efficient FPGA Accelerator for Vision Transformers
- url: https://arxiv.org/abs/2402.09709
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/ME-ViT.pdf`
- related_codebases: `none`

## 1. 기존 방법의 문제점
- 요약: ViT FPGA accelerator는 off-chip memory access가 frequent하고 bandwidth bottleneck으로 throughput이 제한된다.

### Paper Evidence: Abstract
```text
Abstract—Vision Transformers (ViTs) have emerged as a state-
arXiv:2402.09709v1 [eess.IV] 15 Feb 2024




                                           of-the-art solution for object classification tasks. However, their
                                           computational demands and high parameter count make them
                                           unsuitable for real-time inference, prompting the need for effi-
                                           cient hardware implementations. Existing hardware accelerators
                                           for ViTs suffer from frequent off-chip memory access, restricting
                                           the achievable throughput by memory bandwidth. In devices with
                                           a high compute-to-communication ratio (e.g., edge FPGAs with
                                           limited bandwidth), off-chip memory access imposes a severe
                                           bottleneck on overall throughput. This work proposes ME-ViT,
                                           a novel Memory Efficient FPGA accelerator for ViT inference
                                           that minimizes memory traffic. We propose a single-load policy
                                           in designing ME-ViT: model parameters are only loaded once,
                                           intermediate results are stored on-chip, and all operations are
                                           implemented in a single processing element. To achieve this goal,
                                                                                                                 Fig. 1: Roofline model of state-of-the-art (SOTA) archit
```

## 2. 제안하는 방법
- 요약: single-load policy로 model parameters를 한 번만 읽고 intermediate를 on-chip에 유지한다.

### Paper Evidence: Method Section
```text
approach.                                                          for similar FPGA architectures, a non optimized approach is
                                                                   calculated with the following characteristics that are common
D. Overall Throughput and Latency                                  in various designs [21], [23], [30]: Each BMM loads two
   Overall throughput measured in frames per second (FPS) for      input matrices. If an input block matrix was used for the
the 4 models is shown in Table IV. FPS improves as model           previous multiply, it remains loaded. 2) All calculated matrix
sizes get smaller, and PSYS = 16 experiences roughly 0.25×         blocks are written back to DRAM. 3) Softmax and LayerNorm
the throughput of corresponding PSYS = 32 designs. Latencies       [33] are calculated on the CPU and are implicitly included in
per ME-ViT mode are presented in Figure 15. MLP Mode               intermediate write-backs.
exhibits the longest duration out of the three modes, taking          Figure 16 shows memory bandwidth figures for ME-ViT
approximately 60 percent of execution time across all models.      on all four models for both PSYS = 32 and PSYS = 16. Total

                                              TABLE III: Platform Performance Comparison

                 CPU i7-     GPU            GPU                                                           ME-ViT
                                                       Auto Vit       ViTA            ME-ViT                                  Multi-PE
  Platform        9800X      Titan         Jetson                                                        Theoretical
                                                       Acc [22]       [30]
                   [23]     RTX [23]      TX2 [22]                              150 MHz   300 MHz    150 MHz 300 MHz     150 MHz    300 MHz
Latency (ms)       65.35       5.45         127         38.61         363.64      83.38     41.69     75.73      37.86     75.73      37.86
    FPS            15.3       183.4         7.87         25.9          2.75       11.99     23.98     13.20      26.40     66.02     132.04
 Power (W)          100        260         12.28         9.4           0.88        6.5       9.3       6.5        9.3       17.8      31.8
 FPS/Watt          0.15        0.71         0.64         2.76          3.13        1.84      2.57      2.03       2.83      3.71      4.15
  FPS/DSP            –          –            –          0.012           –         0.012     0.023     0.013      0.026     0.013     0.026
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: multi-purpose buffers를 재사용하고 Softmax/LayerNorm을 ME-PE 안에 통합해 matmul 사이 stall을 줄인다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `parallelism`:
```text
h, but are still constrained from the overall The self-attention based model of the Transformer [1] has architectural limitations imposed by the GPU. led to significant advancements in machine learning, impacting Field Programmable Gate Arrays (FPGAs) provide a good a diverse range of applications [2]–[6]. Originally gaining platform for ViT acceleration due to the high computa- prominence due to its remarkable success in natural language tional parallelism and custom architectures that can be de- processing [7], the Transformer has been adapted to the signed [21]–[23]. However, like GPUs, the high memory domain of computer vision via Vision Transformers (ViTs) [8], bandwidth needs of Transformers significantly limits their achieving superior performance over convolutional networks implementation on FPGAs. As shown in Figure 1, ViT and [9], [10]. Despite the substantial achievements of V
```
- keyword `buffer`:
```text
e only loaded once, intermediate results are stored on-chip, and all operations are implemented in a single processing element. To achieve this goal, Fig. 1: Roofline model of state-of-the-art (SOTA) architectures we design a memory-efficient processing element (ME-PE), which processes multiple key operations of ViT inference on the same and ME-ViT for various models. Vertical axis is in log scale. architecture through the reuse of multi-purpose buffers. We also ME-ViT optimizes memory bandwidth, enabling nearly peak integrate the Softmax and LayerNorm functions into the ME-PE, performance in GOPS (Giga Operations per Second). SOTA minimizing stalls between matrix multiplications. We evaluate implementations are bottlenecked by memory bandwidth. Four ME-ViT on systolic array sizes of 32 and 16, achieving up to a ViT variants are shown: ViT-Base model from [8] (ViT-B), and 9.22× and 17.89
```

## 4. 하드웨어 아키텍처
- 요약: ME-PE가 ViT key operations를 같은 architecture에서 처리하며 여러 PE instantiation을 지원한다.

### Paper Evidence: Architecture / Implementation
```text
architecture through the reuse of multi-purpose buffers. We also      ME-ViT optimizes memory bandwidth, enabling nearly peak
                                           integrate the Softmax and LayerNorm functions into the ME-PE,         performance in GOPS (Giga Operations per Second). SOTA
                                           minimizing stalls between matrix multiplications. We evaluate         implementations are bottlenecked by memory bandwidth. Four
                                           ME-ViT on systolic array sizes of 32 and 16, achieving up to a
                                                                                                                 ViT variants are shown: ViT-Base model from [8] (ViT-B), and
                                           9.22× and 17.89× overall improvement in memory bandwidth,
                                           and a 2.16× improvement in throughput per DSP for both                three models from [10] (DeiT-B, DeiT-S, and DeiT-T).
                                           designs over state-of-the-art ViT accelerators on FPGA. ME-
                                           ViT achieves a power efficiency improvement of up to 4.00×
                                           (1.03×) over a GPU (FPGA) baseline. ME-ViT enables up to              these methods do not directly address the main performance
                                           5 ME-PE instantiations on a Xilinx Alveo U200, achieving a            bottleneck of ViT inference on modern hardware: memory
                                           5.10× improvement in throughput over the state-of-the art FPGA
                                           baseline, and a 5.85× (1.51×) improvement in power efficiency         bandwidth. Computing capabilities of hardware such as GPUs
                                           over the GPU (FPGA) baseline.                                         and TPUs have outpaced memory bandwidth improvements,
                                              Index Terms—Vision Transformer, FPGA Accelerator, Mem-             resulting in a poor Compute-to-Communication (C2C) ratio
                                           ory Bandwidth                                                         and limited model performance [16], [17]. Various works
                                                                                                                 [18]–[20] have focused on algorithmic approaches to reducing
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
ME-ViT: A Single-Load Memory-Efficient FPGA Accelerator for Vision Transformers Kyle Marino Pengmiao Zhang Viktor K. Prasanna University of Southern California University of Southern California University of Southern California Los Angeles, USA Los Angeles, USA Los Angeles, USA kmarino@usc.edu pengmiao@usc.edu prasanna@usc.edu Abstract—Vision Transformers (ViTs) have emerged as a state- arXiv:2402.09709v1 [eess.IV] 15 Feb 2024 of-the-art solution for object classification tasks. Howev
```
- keyword `Vitis`:
```text
Ps, 1766 36k BRAMs, 892K LUTs, for computation which is parallelized with matrix multiplica- and 1831K FFs. It has 4 channels of DDR memory, and a tion. The Pseudo-Softmax uses base 2 instead of e to leverage total bandwidth of 77 GB/s. Experimental results are evaluated floating point number properties to evaluate exponentiation. independently for each ME-ViT mode, and theoretical values 2xi are presented which remove extra latencies added from Vitis pei = PN (9) k=1 2 xk synthesis and Place and Route (P&R). All implementations are designed for 300 MHz, and 150 MHz figures are provided Let ai be a floating-point number with exponent xi . This to compare with other designs. The ME-PE is designed removes the need to calculate exponentiation in hardware since and evaluated using Vitis HLS 2023.1. Power estimates are it is implicitly handled by the float representation. calculated using AMD
```
- keyword `DSP`:
```text
e Softmax and LayerNorm functions into the ME-PE, performance in GOPS (Giga Operations per Second). SOTA minimizing stalls between matrix multiplications. We evaluate implementations are bottlenecked by memory bandwidth. Four ME-ViT on systolic array sizes of 32 and 16, achieving up to a ViT variants are shown: ViT-Base model from [8] (ViT-B), and 9.22× and 17.89× overall improvement in memory bandwidth, and a 2.16× improvement in throughput per DSP for both three models from [10] (DeiT-B, DeiT-S, and DeiT-T). designs over state-of-the-art ViT accelerators on FPGA. ME- ViT achieves a power efficiency improvement of up to 4.00× (1.03×) over a GPU (FPGA) baseline. ME-ViT enables up to these methods do not directly address the main performance 5 ME-PE instantiations on a Xilinx Alveo U200, achieving a bottleneck of ViT inference on modern hardware: memory 5.10× improvement in throughput ove
```

## 5. 실험 방법
```text
results computed in timestamp 4 are added to the staged values                                                                  1X 2           1X
in the S1 buffer, and the next two output layer blocks are                                                                σi2 =       Xi,j − (       Xi,j )2           (8)
                                                                                                                                n j=1          n j=1
calculated. This process repeats until all sub-blocks involving
M1,1 are calculated. At timestamp 6, the S1 buffer containing                                                  This reduces the full LayerNorm calculation to two passes;
the staged values is stored back into the Feature Buffer, and                                                The first pass accumulates a sum and squared sum of each
multiplication repeats with M2,1 . At timestamp 7, the next pair                                             row i. √After the row sums are calculated, the RowMeani
of Layer Buffer row blocks is processed. This continues until                                                and 1/ RowVari are calculated using fixed-point arithmetic

                                   Alveo U200                              U200 contains 3 Super Logic Regions (SLRs), with SLRs 0
    FPGA DDR          SLR 0             SLR 1           SLR 2              and 2 having 2275 DSPs each and SLR 1 only having 1317.
     Memory           ME-PE             ME-PE           ME-PE              With a PSYS = 32, up to 5 PEs can fit as shown in Figure 14.
                                                                           A smaller PSYS can fit more SAs in the FPGA, however the
                      ME-PE             Scheduler       ME-PE
                                                                           BRAM needed for such a design is the same as PSYS = 32
                    FPGA Shell         FPGA Shell      FPGA Shell          since buffering requirements are unchanged. Therefore, only
                                                                           the PSYS = 32 design is implemented to maximally utilize
Fig. 14: Multi-PE ME-ViT architecture on the Alveo U200.
                                                                           the available DSPs. Resource utilization is discussed in more
                                                                           detail in Section IV-B. The Multi-PE architecture can achieve
functions. These constants are fed into the LayerNorm Module               remarkably higher throughput, but an increase in data traffic
shown in Figure 13a. By passing each row element j through                 causes a bottleneck due to the limited 77 GB/s bandwidth to
the LayerNorm Module in a pipelined manner, the final layer-               the FPGA DRAM. These results are discussed in Section IV-G.
normalized value is efficiently computed.
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `DeiT`:
```text
o the ME-PE, performance in GOPS (Giga Operations per Second). SOTA minimizing stalls between matrix multiplications. We evaluate implementations are bottlenecked by memory bandwidth. Four ME-ViT on systolic array sizes of 32 and 16, achieving up to a ViT variants are shown: ViT-Base model from [8] (ViT-B), and 9.22× and 17.89× overall improvement in memory bandwidth, and a 2.16× improvement in throughput per DSP for both three models from [10] (DeiT-B, DeiT-S, and DeiT-T). designs over state-of-the-art ViT accelerators on FPGA. ME- ViT achieves a power efficiency improvement of up to 4.00× (1.03×) over a GPU (FPGA) baseline. ME-ViT enables up to these methods do not directly address the main performance 5 ME-PE instantiations on a Xilinx Alveo U200, achieving a bottleneck of ViT inference on modern hardware: memory 5.10× improvement in throughput over the state-of-the art FPGA baseline,
```

## 7. 실험 결과
- keyword `accuracy`:
```text
2 384 6 12 22M as per Equation 12. 127 is added to the score row to convert DeiT-T 2242 192 3 12 6M to a floating point exponent which is stored unsigned. A 1 is prepended to the mantissa since it is implicitly present in the ViT variants shown in Table I are evaluated on the ME-ViT floating point format. The result is stored in fixed-point format architecture. ViT-B refers to the base ViT model presented in with only fractional bits to maximize accuracy. The upper 8 [8] but with 256 input image resolution. DeiT-B, DeiT-S, and bits after the bit-shift contain the fixed-point representation of DeiT-T refer to model sizes presented in [10], all on 224 input the Pseudo-Softmax function. image resolution. The key difference between DeiT models F. Multiple ME-PE Architecture for ME-ViT lies in their respective model dimensions: 768, 384, and 192. A Multiple ME-PE architecture (Multi-PE) is pr
```
- keyword `throughput`:
```text
sanna@usc.edu Abstract—Vision Transformers (ViTs) have emerged as a state- arXiv:2402.09709v1 [eess.IV] 15 Feb 2024 of-the-art solution for object classification tasks. However, their computational demands and high parameter count make them unsuitable for real-time inference, prompting the need for effi- cient hardware implementations. Existing hardware accelerators for ViTs suffer from frequent off-chip memory access, restricting the achievable throughput by memory bandwidth. In devices with a high compute-to-communication ratio (e.g., edge FPGAs with limited bandwidth), off-chip memory access imposes a severe bottleneck on overall throughput. This work proposes ME-ViT, a novel Memory Efficient FPGA accelerator for ViT inference that minimizes memory traffic. We propose a single-load policy in designing ME-ViT: model parameters are only loaded once, intermediate results are stored on-ch
```
- keyword `GOPS`:
```text
Fig. 1: Roofline model of state-of-the-art (SOTA) architectures we design a memory-efficient processing element (ME-PE), which processes multiple key operations of ViT inference on the same and ME-ViT for various models. Vertical axis is in log scale. architecture through the reuse of multi-purpose buffers. We also ME-ViT optimizes memory bandwidth, enabling nearly peak integrate the Softmax and LayerNorm functions into the ME-PE, performance in GOPS (Giga Operations per Second). SOTA minimizing stalls between matrix multiplications. We evaluate implementations are bottlenecked by memory bandwidth. Four ME-ViT on systolic array sizes of 32 and 16, achieving up to a ViT variants are shown: ViT-Base model from [8] (ViT-B), and 9.22× and 17.89× overall improvement in memory bandwidth, and a 2.16× improvement in throughput per DSP for both three models from [10] (DeiT-B, DeiT-S, and DeiT-T).
```
- keyword `FPS`:
```text
1 ak sizes shown in Table I. ME-PEs with systolic array size The summation term can be expressed as a single floating point PSYS = 32 and PSYS = 16 are analyzed to provide insight number. In this representation, expsum and mantsum denote the into the performance scalability across FPGAs of varying exponent and mantissa of the resulting number. DSP resources. As the size of the systolic array decreases, there is a corresponding reduction in total FPS (frames per N X second). Memory bandwidth also reduces despite smaller ak = 2expsum · mantsum (11) systolic arrays requiring more frequent data transfers. This k=1 relationship between scale and memory bandwidth is explored The Pseudo-Softmax pei for element xi , is calculated as: in Section IV-E. Finally, Multi-PE results are calculated based 1 on single PSYS = 32 ME-PE performance to maximally utilize pei = 2xi −expsum · . (12) 1 · mantsum
```
- keyword `latency`:
```text
chitecture for ME-ViT lies in their respective model dimensions: 768, 384, and 192. A Multiple ME-PE architecture (Multi-PE) is proposed that B. Results on Hardware contains parallel instantiations of the ME-PE along with a Results for hardware utilization are shown in Table II. The scheduler to coordinate data traffic between them. The Alveo three ME-ViT modes are implemented separately, and through- put values in Table III are derived from the latency per mode. Since each mode largely utilizes the same resources but with different control logic, the unified design’s resources would marginally exceed that of the largest mode (MLP Mode). Resources are shown for the PSYS = 32 ME-PE for the ViT-B model. Resource consumption is unchanged for DeiT-B, and BRAM usage drops to 176 and 144 for DeiT-S and DeiT-T respectively. For PSYS = 16, 256 DSPs are used, with other values remaining the same
```

## 8. 실험 옵션 / Ablation 축
- exact softmax vs approximated softmax
- Taylor/linear attention vs exact attention
- fixed sparse token pattern vs dense token pattern
- URAM/BRAM/LUTRAM binding
- memory banks 8/16/32
- parallelism factor 8/16/32
- FIFO depth and double buffering

## 9. HGTXR 적용 판단
- C3b URAM/global buffer 정책과 직접 맞는 P0 후보. 정확도보다 memory traffic/resource 안정화에 초점.

### 우선순위 판정
- recommended_priority: `P0`
- C3b board-smoke gate는 유지한다.
- HLS/Vivado 증거가 생기기 전에는 resource matrix 완료 variant로 승격하지 않는다.

## 10. HGTXR 실험 설계로 변환
| 단계 | 작업 | 산출물 | 승격 조건 |
|---|---|---|---|
| SW | 알고리즘을 Python/C++ reference에 반영 | accuracy/bit-exact report | 정확도 개선 또는 무손실 |
| HLS | parameterized macro/defines로 구현 | csim/csynth report | csynth.xml 생성 및 resource gate 통과 |
| Vivado | C3b successor overlay로 구현 | routed timing/power/util | WNS >= 0, resource 정책 유지 |
| PYNQ | smoke run | board JSON | validator pass |
