---
source_type: paper
source_name: HG-PIPE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/HG-PIPE/analysis.md -->

# HG-PIPE Detailed Paper Analysis

- title: HG-PIPE: Vision Transformer Acceleration with Hybrid-Grained Pipeline
- url: https://arxiv.org/abs/2407.17879
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/HG-PIPE.pdf`
- related_codebases: `HG-PIPE`

## 1. 기존 방법의 문제점
- 요약: temporal ViT accelerator는 memory access overhead가 크고, coarse/fine pipeline은 resource constraint와 bubble 문제가 있다.

### Paper Evidence: Introduction / Motivation
```text
1    INTRODUCTION                                                            architectures compute multiple layers simultaneously and natively
                                                                                                                     require more hardware resources, e.g., digital signal processing
                                        Recent years have witnessed the wide adoption of Transformer
                                                                                                                     (DSP) blocks. The complex non-linear functions in ViT, including
                                        models in the field of computer vision (CV)[6, 20, 31, 2, 32, 36].
                                                                                                                     GeLU, Softmax, LayerNorm, etc, also require high-precision com-
                                        While Vision Transformers (ViTs) achieve state-of-the-art (SOTA)
                                                                                                                     putation and DSP usage[33, 25, 47]. As shown in Figure 1, only 3.2
                                        performance compared to convolutional neural networks (CNNs),
                                                                                                                     TOP/s can be achieved for a coarse-grained pipeline design due
                                        they suffer from a drastic increase in parameters and computation,
                                                                                                                     to the DSP limitation. If lookup tables (LUTs) are also utilized to
                                        which calls for more efficient acceleration.
                                                                                                                     construct the PEs, the roofline can be improved, but the design will
                                           ViT acceleration based on field-programmable gate array (FPGA)
```

## 2. 제안하는 방법
- 요약: hybrid-grained pipeline으로 buffer cost를 줄이고 dataflow/parallelism을 결합해 bubble을 제거한다.

### Paper Evidence: Method Section
```text
approach is visualized in Figure 10d. The orange line is the abs
                                                                                                                                                                                                                                    
error of the original LUT implementation of Recip, and the blue                                                                                                   $FFXUDF\


one is the segmented implementation with the segmentation pivot
                                                                             (a) DSP usage and accuracy with all the optimization applied step by step
annotated to it. With more entries between 0 and 1, the sampling
is more accurate, reducing Mean Squared Error (MSE) from 0.032
                                                                                                  Ablation                                                             Deit-tiny 3bit                                           Deit-tiny 4bit
to 0.0034.                                                                                        w/o Inverted Exp                                                     28.80% (-42.25%)                                         48.87% (-25.50%)
                                                                                                  w/o ReQuant Calib.                                                   70.56% (-0.49%)                                          74.08% (-0.29%)
4.4.7 Inversed Exponential Table. In our experiment, we observed                                  w/o GeLU Calib.                                                      69.49% (-1.56%)                                          72.44% (-1.93%)
that the PoT approximation on Exp will cause huge accuracy degra-                                 w/o Segmented Recip                                                  70.57% (-0.48%)                                          74.04% (-0.33%)
dation. The possible explanation is that in the calculation of Softmax,
                                                                                                                                  (b) Ablation study experiment results.
each element is subtracted by the maximum value in its group to
maintain numerical stability. Therefore, the max value of the input
                                                                                                     Non-linear                                   Table                 Table
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: operator별 pipeline granularity와 parallelism을 조정하고 linear/nonlinear operator approximation을 사용한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
 approach is visualized in Figure 10d. The orange line is the abs      error of the original LUT implementation of Recip, and the blue $FFXUDF\  one is the segmented implementation with the segmentation pivot (a) DSP usage and accuracy with all the optimization applied step by step annotated to it. With more entries between 0 and 1, the sampling is more accurate, reducing Mean Squared Error (MSE) from 0.032 Ablation Deit-tiny 3bit Deit-tiny 4bit to 0.0034. w/o Inverted Exp 28.80% (-42.25%) 48.87% (-25.50%) w/o ReQuant Calib. 70.56% (-0.49%) 74.08% (-0.29%) 4.4.7 Inversed Exponential Table. In our experiment, we observed w/o GeLU Calib. 69.49% (-1.56%) 72.44% (-1.93%) that the PoT approximation on Exp will cause huge accuracy degra- w/o Segmented Recip 70.57% (-0.48%) 74.04% (-0.33%) dation. The possible explanation is that in the calculation of Soft
```
- keyword `bit-width`:
```text
tec- HG-PIPE can be summarized as follows: tures, in contrast, instantiate distinct PEs specialized for different operators and directly stream activations among PEs to reduce • HG-PIPE features a hybrid-grained pipelined architecture Data1 Bubble Data2 to simultaneously achieve low off-chip memory access, low FPGA （1） buffer requirements, and negligible pipeline bubbles. PE (GeMM) Aux PE L1 L2 L3 L1 L2 L3 Aux A A A A A A • HG-PIPE leverages low bit-width quantization and intro- duces careful approximations for activation functions. The FPGA PE1 L1 L1 latency （2） PE2 L2 L2 PIPO PIPO PE1 PE2 PE3 reduced abundant LUT resources are utilized to process both linear PE3 L3 L3 and non-linear operators, achieving a higher roofline. FIFO FPGA PE1 L1 L1 （3） • HG-PIPE demonstrates 2.72× better throughput and 2.46× PE2 L2 L2 latency reduced PE1 PE2 PE3 better resource efficiency over prior-art accel
```
- keyword `sparsity`:
```text
fferent layers. Such PE is typically dedicated to perform- ing General Matrix Multiplication, commonly referred to as GeMM. • Floating point implementation employs 32-bit or 16-bit float- Most existing FPGA-based ViT accelerators belong to the category ing point computation. Despite its simplicity and precision, [4, 5, 10, 12, 11, 15, 19, 22, 23, 29, 33, 35, 37, 40, 44, 45]. Efficient it demands substantial DSPs and LUTs. systolic arrays [16] or sparsity-aware PEs [5, 45] enable temporal • Fixed-point polynomial approximation uses low-order poly- architectures to accelerate linear layers effectively. However, these nomials to implement non-linear functions within specified architectures often necessitate frequent off-chip memory access ranges[14]. This method is a compromise between computa- for intermediate results and tend to underutilize resources due to a tional complexity and accura
```

## 4. 하드웨어 아키텍처
- 요약: ZCU102/VCK190 대상 pipelined FPGA accelerator이며 layer/operator instance를 spatial/temporal 혼합 배치한다.

### Paper Evidence: Architecture / Implementation
```text
architectures to accelerate linear layers effectively. However, these
      nomials to implement non-linear functions within specified
                                                                              architectures often necessitate frequent off-chip memory access
      ranges[14]. This method is a compromise between computa-
                                                                              for intermediate results and tend to underutilize resources due to a
      tional complexity and accuracy.
                                                                              lack of concurrent multi-operator execution.
    • Lookup table method involves discretizing the function in-
                                                                                 Pipelined architectures, as illustrated in Figure 2(a), optimize
      put range and recording the output. While it reduces DSP
                                                                              layer processing by customizing PEs for varied operators across lay-
      usage, it usually requires Block RAMs (BRAMs) for accurate
                                                                              ers [42, 7], enhancing resource utilization and minimizing off-chip
      sampling[34].
                                                                              memory costs through inter-PE data transfer [42, 39, 8]. Coarse-
                                                                              grained pipeline processes entire tensors for different operators [21,
2.2     FPGA-based ViT Acceleration                                           18, 43, 8, 17, 28, 42]. On the contrary, fine-grained pipeline tiles acti-
The FPGA-based ViT accelerators can be categorized into two ar-               vation tensors for sub-tensor processing[26, 48]. A coarse-grained
chitectures: temporal architecture and pipelined architecture. As
shown in Figure 2(a), temporal architectures leverage unified PEs to

pipeline uses Ping Pong (PIPO) buffers, requiring double the ten-                                           HLS synthesis experiments, naive floating-point implementations
sor’s memory size, while a fine-grained pipeline employs First In                                           of functions like Exp, Rsqrt, and Recip are DSP-intensive, con-
First Out (FIFO) buffers with tile-level granularity. The behavior of                                       suming 7, 8, and 9 DSPs respectively. The Ge
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
its, Peking University, China 2 Institute for Artificial Intelligence, Peking University, China 3 School of Software and Microelectronics, Peking University, China 4 Beijing Advanced Innovation Center for Integrated Circuits, China *meng.li@pku.edu.cn, *wangyuan@pku.edu.cn ABSTRACT Throughput (TOP/s) 17.8 TOP/s arXiv:2407.17879v2 [cs.AR] 1 Aug 2024 Vision Transformer (ViT) acceleration with field programmable (measured) LUT Roofline: gate array (FPGA) is promising but challenging. Existing FPGA- estim. @425MHz on VCK190 s ③ 900k LUTs based ViT accelerators mainly rely on temporal architectures, which GB/ ≈ 20k MAC units 3 process different operators by reusing the same hardware blocks 8 .5 7.8 TOP/s Hybrid- h: Grained dt and suffer from extensive memory access overhead. Pipelined ar- wi +LUT Pipeline chitectures, either coarse-grained or fine-grained, unroll the ViT an d PEs ② computatio
```
- keyword `ZCU102`:
```text
ecture to reduce 131 OPs/byte 909 OPs/byte 12715 OPs/byte on-chip buffer cost and couples the computation dataflow and par- Computational Intensity allelism design to eliminate the pipeline bubbles. HG-PIPE further (OPs/byte) introduces careful approximations to implement both linear and non-linear operators with abundant Lookup Tables (LUTs), thus alle- Figure 1: Roofline model for FPGA-based ViT acceleration. viating resource constraints. On a ZCU102 FPGA, HG-PIPE achieves 2.78× better throughput and 2.52× better resource efficiency than off-chip memory access [42, 26, 48, 49]. Since they allow concur- the prior-art accelerators, e.g., AutoViTAcc. With a VCK190 FPGA, rent processing of multiple layers, pipeline architectures hold the HG-PIPE realizes end-to-end ViT acceleration on a single device promise to enable efficient and low-latency ViT acceleration. and achieves 7118 images/s,
```
- keyword `VCK190`:
```text
, Peking University, China 3 School of Software and Microelectronics, Peking University, China 4 Beijing Advanced Innovation Center for Integrated Circuits, China *meng.li@pku.edu.cn, *wangyuan@pku.edu.cn ABSTRACT Throughput (TOP/s) 17.8 TOP/s arXiv:2407.17879v2 [cs.AR] 1 Aug 2024 Vision Transformer (ViT) acceleration with field programmable (measured) LUT Roofline: gate array (FPGA) is promising but challenging. Existing FPGA- estim. @425MHz on VCK190 s ③ 900k LUTs based ViT accelerators mainly rely on temporal architectures, which GB/ ≈ 20k MAC units 3 process different operators by reusing the same hardware blocks 8 .5 7.8 TOP/s Hybrid- h: Grained dt and suffer from extensive memory access overhead. Pipelined ar- wi +LUT Pipeline chitectures, either coarse-grained or fine-grained, unroll the ViT an d PEs ② computation spatially for memory access efficiency. However, they B usually suf
```

## 5. 실험 방법
```text
results on the sheer range between 0 to 1, the table needs to be             3R7,QGH[                                                                 
                                                                             $SSUR[
very large. Initially, storing the reciprocal function would have                                                 
                                                                             ,QYHUVHG 
needed an entire BRAM bank (depth=1024, width=36) to maintain                ([S                                                                                                                                                             
accuracy. To minimize BRAM usage, we exploited the function’s                5H4XDQW 
                                                                             &DOLE                                                                                                                                    
                             
inherent properties and segmented it into two parts, each owning             *H/8
                                                                             &DOLE                                                                                                                                                            
an independent scaling factor. We empirically divide the input range
                                                                             6HJPHQW                                                                           ELW$FFXUDF\
at the first 1/8 for the steep part and the remainder for the flat. This     7DEOH                                                                                   ELW$FFXUDF\                                                      
approach is visualized in Figure 10d. The orange line is the abs
                                                                                                                                                                                                                                    
error of the original LUT implementation of Recip, and the blue                                                                                                   $FFXUDF\


one is the segmented implementation with the segmentation pivot
                                                                             (a) DSP usage and accuracy with all the optimization applied step by step
annotated to it. With more entries between 0 and 1, the sampling
is more accurate, reducing Mean Squared Error (MSE) from 0.032
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `DeiT`:
```text
el granularity. The behavior of suming 7, 8, and 9 DSPs respectively. The GeLU function is even PIPO and FIFO are depicted in Figure 2(b). Fine-grained pipelined more DSP-intensive, requiring 26 DSPs. The ReQuant function architectures are predominantly utilized in CNN acceleration, pro- additionally uses 1 DSP. In the estimation, implementing these non- viding high hardware utilization and low buffer costs [9, 48, 26, 30], linear functions in a Deit-tiny model requires 3024 DSPs, exceeding whereas coarse-grained architectures are preferred for ViT acceler- the DSP capacity of a VCK190 FPGA. Therefore, reducing DSP ation [21, 18, 43, 8, 17, 28, 42] for data access ability. A comparative usage in ViT accelerators is crucial for FPGA implementation. analysis of these architectures is presented in Table 2(c). 4 MAIN METHODS 3 CHALLENGES 4.1 Overview of HG-PIPE Although ViT has outstanding p
```

## 7. 실험 결과
- keyword `accuracy`:
```text
6] or sparsity-aware PEs [5, 45] enable temporal • Fixed-point polynomial approximation uses low-order poly- architectures to accelerate linear layers effectively. However, these nomials to implement non-linear functions within specified architectures often necessitate frequent off-chip memory access ranges[14]. This method is a compromise between computa- for intermediate results and tend to underutilize resources due to a tional complexity and accuracy. lack of concurrent multi-operator execution. • Lookup table method involves discretizing the function in- Pipelined architectures, as illustrated in Figure 2(a), optimize put range and recording the output. While it reduces DSP layer processing by customizing PEs for varied operators across lay- usage, it usually requires Block RAMs (BRAMs) for accurate ers [42, 7], enhancing resource utilization and minimizing off-chip sampling[34]. me
```
- keyword `throughput`:
```text
E: Vision Transformer Acceleration with Hybrid-Grained Pipeline Qingyu Guo1 , Jiayong Wan3 , Songqiang Xu3 , Meng Li2,1,4* , Yuan Wang1,4* 1 School of Integrated Circuits, Peking University, China 2 Institute for Artificial Intelligence, Peking University, China 3 School of Software and Microelectronics, Peking University, China 4 Beijing Advanced Innovation Center for Integrated Circuits, China *meng.li@pku.edu.cn, *wangyuan@pku.edu.cn ABSTRACT Throughput (TOP/s) 17.8 TOP/s arXiv:2407.17879v2 [cs.AR] 1 Aug 2024 Vision Transformer (ViT) acceleration with field programmable (measured) LUT Roofline: gate array (FPGA) is promising but challenging. Existing FPGA- estim. @425MHz on VCK190 s ③ 900k LUTs based ViT accelerators mainly rely on temporal architectures, which GB/ ≈ 20k MAC units 3 process different operators by reusing the same hardware blocks 8 .5 7.8 TOP/s Hybrid- h: Grained dt an
```
- keyword `GOPS`:
```text
input ten- Table 2. Our hybrid-grained pipeline design demonstrated signifi- sor tiles. As Image1’s loading completes, Image2’s begins, indicat- cant improvements in throughput, resource efficiency, and power ing overlapped execution. The MHA block exhibits coarse-grained efficiency. On the ZCU102 platform, compared to AutoViTAcc, HG- buffering, causing a slight delay in outputting the first tile. For the PIPE achieves a LUT efficiency of 18.55 GOPs/kLUT, which is 2.52 third image, the stable II measured was 57,624 cycles as expected, times higher under the same 4-bit quantization and platform condi- validating the hybrid-grained pipeline’s effectiveness. The trace tions. On the VCK190 platform, HG-PIPE achieved a throughput of also reveals that the total processing time for Image1 is 824,843 7118 images/s and 17.8 TOPs/s, which is 96.8% of the ideal 7353 im- cycles or 1.94ms. Using the
```
- keyword `FPS`:
```text
times higher under the same 4-bit quantization and platform condi- validating the hybrid-grained pipeline’s effectiveness. The trace tions. On the VCK190 platform, HG-PIPE achieved a throughput of also reveals that the total processing time for Image1 is 824,843 7118 images/s and 17.8 TOPs/s, which is 96.8% of the ideal 7353 im- cycles or 1.94ms. Using the stable II, the computed latency is 0.136 ages/s throughput. Compared to the V100 GPU (2529 FPS), HG-PIPE ms, equating to an ideal frame rate of 7353 images/s. outperforms it by 2.81 times. For DSP usage, HG-PIPE drastically Deit GPU TCAS-I AutoViTAcc HeatViT SSR HG-PIPE Baseline [38] 2023 [12] FPL 2022 [19] HPCA 2023 [5] FPGA 2024 [49] This work 2024 Coarse-Grained Paradigm GPU GeMM GeMM GeMM Hybrid-Grained Pipeline Pipeline FPGA/GPU V100 GPU ZCU102 ZCU102 ZCU102 VCK190 ZCU1023 VCK190 VCK190 VCK190 PL:250MHz, Frequency 1455MHz 300MHz 1
```
- keyword `latency`:
```text
ther coarse-grained or fine-grained, unroll the ViT an d PEs ② computation spatially for memory access efficiency. However, they B usually suffer from significant hardware resource constraints and DSP Roofline: 3.2 TOP/s 2k DSPs Coarse- pipeline bubbles induced by the global computation dependency ① Grained = 4k MAC units of ViT. In this paper, we introduce HG-PIPE, a pipelined FPGA GeMM 1.1 TOP/s Pipeline accelerator for high-throughput and low-latency ViT processing. weight & act off-chip: act on-chip: weight & act on-chip: HG-PIPE features a hybrid-grained pipeline architecture to reduce 131 OPs/byte 909 OPs/byte 12715 OPs/byte on-chip buffer cost and couples the computation dataflow and par- Computational Intensity allelism design to eliminate the pipeline bubbles. HG-PIPE further (OPs/byte) introduces careful approximations to implement both linear and non-linear operators with abun
```

## 8. 실험 옵션 / Ablation 축
- software-only reference comparison
- microkernel latency/resource comparison

## 9. HGTXR 적용 판단
- 이미 legacy 근거로 쓰였고, 현 HGTXR은 HG-PIPE의 pipeline 교훈을 유지하되 LUT-heavy approximation은 DSP/URAM 정책으로 재해석한다.

### 우선순위 판정
- recommended_priority: `P2`
- C3b board-smoke gate는 유지한다.
- HLS/Vivado 증거가 생기기 전에는 resource matrix 완료 variant로 승격하지 않는다.

## 10. HGTXR 실험 설계로 변환
| 단계 | 작업 | 산출물 | 승격 조건 |
|---|---|---|---|
| SW | 알고리즘을 Python/C++ reference에 반영 | accuracy/bit-exact report | 정확도 개선 또는 무손실 |
| HLS | parameterized macro/defines로 구현 | csim/csynth report | csynth.xml 생성 및 resource gate 통과 |
| Vivado | C3b successor overlay로 구현 | routed timing/power/util | WNS >= 0, resource 정책 유지 |
| PYNQ | smoke run | board JSON | validator pass |
