---
source_type: paper
source_name: Trio-ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/Trio-ViT/analysis.md -->

# Trio-ViT Detailed Paper Analysis

- title: Trio-ViT: Post-Training Quantization and Acceleration for Softmax-Free Efficient Vision Transformer
- url: https://arxiv.org/abs/2405.03882
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/Trio-ViT.pdf`
- related_codebases: `Trio-ViT`

## 1. 기존 방법의 문제점
- 요약: standard ViT의 softmax는 quantization-sensitive하고 hardware-unfriendly하다.

### Paper Evidence: Abstract
```text
Abstract—Motivated by the huge success of Transformers in                 shown superior performance compared to their convolution-
arXiv:2405.03882v3 [cs.CV] 30 Sep 2024




                                         the field of natural language processing (NLP), Vision Transform-            based counterparts. However, their enormous model sizes and
                                         ers (ViTs) have been rapidly developed and achieved remark-                  intensive computations challenge the deployment of ViTs on
                                         able performance in various computer vision tasks. However,
                                         their huge model sizes and intensive computations hinder ViTs’               embedded/mobile devices, where both memory and computing
                                         deployment on embedded devices, calling for effective model                  resources are limited. For example, ViT-Large [4] contains
                                         compression methods, such as quantization. Unfortunately, due to             307M parameters and yields 190.7G FLOPs during inference.
                                         the existence of hardware-unfriendly and quantization-sensitive              Thus, effective model compression techniques are highly de-
                                         non-linear operations, particularly Softmax, it is non-trivial               sired to facilitate ViTs’ real-world applications.
                                         to completely quantize all operations in ViTs, yielding either
                                         significant accuracy drops or non-negligible hardware costs. In                 Among them, model quantiz
```

## 2. 제안하는 방법
- 요약: softmax-free efficient ViT를 대상으로 tailored PTQ engine과 dedicated accelerator를 구성한다.

### Paper Evidence: Method Section
```text
approach greatly facilitates activation quantization without in-                                                                                                           channel needs to be updated to bˆj following Eq. (10), where
creasing learnable scaling factors of activations and impeding                                                                                                             N donate the input channel number. This bias update can be
weight quantization. Note that weights can be pre-transformed                                                                                                              pre-computed to eliminate the on-chip processing.
before deployment to eliminate the on-chip computation. As                                                                                                                                                 max(Ai ) − min(Ai )
for activations that depend on the input images during in-                                                                                                                                        ci =                         , Âi = Ai − ci .                                                     (9)
                                                                                                                                                                                                                    2

                                                                                                                                         6


                              N
 Oj = A·W j +bj = Â·W j +(
                              X
                                ci w(i,j) +bj ) = Â·W j +bˆj . (10)   efficiently support PWConvs with channel parallelism [35],
                              i=1                                      [36]. Specifically, as illustrated in Fig. 7 (a), each process-
                                                                       ing element (PE) lane in the MAT engine is responsible
D. Log2 Quantization for Divisors in MSAs                              for the multiplication (via multipliers), summation (via the
                                                                       adder tree), and accumulation (via the accumulator) along the
   As illustrated in Figs. 4 (a) and (b), log2 quantization will
                                                                       input channel dimension to generate each output pixel (for
allocate more quantization bins to smaller values and vice
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: efficient ViT activation distribution을 반영한 PTQ와 conv-transformer hybrid operator mapping을 사용한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
ViTs. Besides, we propose SOTA one, replaces the vanilla Softmax-based self-attention a pipeline architecture to facilitate both inter- and intra- of quadratic complexity with a novel Softmax-free lightweight layer fusions, thus enhancing hardware utilization and multi-scale attention, achieving a global receptive field while easing the bandwidth requirement. enhancing hardware efficiency. Besides, Flatten Transformer • Extensive experiments and ablation studies consistently [14] opts for an innovative Softmax-based focused linear atten- validate the effectiveness of our Trio-ViT framework. tion, preserving expressiveness with low computational com- For example, we can offer up to ↑3.6×, ↑5.0×, and plexity. Despite the inherent algorithmic benefits of Softmax- ↑7.3× FPS with comparable accuracy over state-of- free linear attentions in efficient ViTs [13], [14], including (i) the-art (SOT
```
- keyword `parallelism`:
```text
N donate the input channel number. This bias update can be weight quantization. Note that weights can be pre-transformed pre-computed to eliminate the on-chip processing. before deployment to eliminate the on-chip computation. As max(Ai ) − min(Ai ) for activations that depend on the input images during in- ci = , Âi = Ai − ci . (9) 2 6 N Oj = A·W j +bj = Â·W j +( X ci w(i,j) +bj ) = Â·W j +bˆj . (10) efficiently support PWConvs with channel parallelism [35], i=1 [36]. Specifically, as illustrated in Fig. 7 (a), each process- ing element (PE) lane in the MAT engine is responsible D. Log2 Quantization for Divisors in MSAs for the multiplication (via multipliers), summation (via the adder tree), and accumulation (via the accumulator) along the As illustrated in Figs. 4 (a) and (b), log2 quantization will input channel dimension to generate each output pixel (for allocate more quantizat
```
- keyword `buffer`:
```text
alue of EfficientViT feature various kernel sizes (3×3 and 5×5) and the (i-1)th bit to obtain the result. For instance, if XQ is strides (1 and 2), resulting in different sizes of sliding windows (0110 0011)2 , then the index i of the first non-zero bit is 6 and distinct overlap patterns between adjacent sliding win- and the value of the 5th bit is 1, thus the log2-quantized XQ dows when conducting convolutions. This necessitates extra log2 line buffers and substantial memory management overheads (XQ ) is 7. By adopting log2 quantization for divisors, we can fur- to support the multiplication and summation functionality ther replace hardware-unfriendly divisions in Eq. (4) with within PE lanes for generating consecutive output pixels [35]. hardware-efficient bit-wise shifts, further enhancing hardware Thirdly, the lack of input reuse opportunities within DWConvs efficiency while boosting
```

## 4. 하드웨어 아키텍처
- 요약: convolution-transformer hybrid architecture에 맞춘 dedicated accelerator로 operation type을 분리/융합한다.

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
ngineering, the hardware perspective [16]–[18]. For instance, Auto-ViT- Nanjing University, and the School of Integrated Circuits, Sun Yat-sen Acc [17] adopts mixed quantization schemes, i.e., fixed-point University (email: zfwang@nju.edu.cn). Correspondence should be addressed to Wendong Mao and Zhongfeng and power-of-two, to quantize ViTs, and develops a dedicated Wang. accelerator to fully leverage the computational resources avail- 2 able on FPGAs. Moreover, ViTCoD [18] proposes pruning and II. R ELATED W ORKS polarization techniques to transform ViTs’ attention maps into A. Model Quantization for Vision Transformers (ViTs) denser and sparser variants, and then develops a dedicated accelerator incorporating both dense and sparse engines to Model quantization, which represents floating-point weights simultaneously execute the above two workloads. Despite the and activations with integ
```
- keyword `ZCU102`:
```text
DSP packing strategy [48] to of MSA. Note that once the computation for S of all heads is accommodate two 8-bit multiplications within each DSP, sim- finished, the R-MAC engine can be reused to compute divisors ilar to Auto-ViT-Acc [17] for fair comparisons. Evaluation: and dividends, together with the MAT engine. We implement our accelerator with Verilog, synthesize through VI. E XPERIMENTAL R ESULTS Vivado Design Suite, and evaluate on Xilinx ZCU102 FPGA at 200-MHz frequency. Table III lists our resource consumptions. A. Experimental Setup Furthermore, we follow [9], [28] to develop a cycle-accurate Dataset, Baselines, and Metrics. We validate our Trio- simulator for our accelerator to obtain fast and reliable esti- ViT’s post-training quantization algorithm on the ImageNet mations and verify them against the RTL implementation to dataset [38] and implement it on the NVIDIA GeForce en
```
- keyword `Vivado`:
```text
final outputs utilization, we adopt the SOTA DSP packing strategy [48] to of MSA. Note that once the computation for S of all heads is accommodate two 8-bit multiplications within each DSP, sim- finished, the R-MAC engine can be reused to compute divisors ilar to Auto-ViT-Acc [17] for fair comparisons. Evaluation: and dividends, together with the MAT engine. We implement our accelerator with Verilog, synthesize through VI. E XPERIMENTAL R ESULTS Vivado Design Suite, and evaluate on Xilinx ZCU102 FPGA at 200-MHz frequency. Table III lists our resource consumptions. A. Experimental Setup Furthermore, we follow [9], [28] to develop a cycle-accurate Dataset, Baselines, and Metrics. We validate our Trio- simulator for our accelerator to obtain fast and reliable esti- ViT’s post-training quantization algorithm on the ImageNet mations and verify them against the RTL implementation to dataset [3
```

## 5. 실험 방법
```text
results in hybrid architectures for efficient ViTs that comprise    functionality of the self-attention mechanism during quantiza-
both convolutions and Transformer blocks, thus calling for          tion, successfully quantizing linear operations (matrix multipli-
dedicated accelerators to unleash their potential benefits.         cations) in ViTs. Additionally, FQ-ViT [10] further introduces
                                                                    Power-of-Two Factor (PTF) and Log-Int-Softmax (LIS) to
   To grasp the inherent quantization and acceleration opportu-     quantize the hardware- and quantization-unfriendly non-linear
nities in efficient ViTs, we make the following contributions:      operations (i.e., LayerNorm and Softmax) in ViTs, achieving
                                                                    full quantization. However, these works are developed for
  • We propose Trio-ViT, a post-training quantization and           standard ViTs and cannot capture quantization opportunities
    acceleration framework for efficient Vision Transformers        offered by efficient ViTs [13], [14], which feature Softmax-
    (ViTs) via algorithm and hardware co-design. To the best        free attention with linear computational complexity to win
    of our knowledge, this is the first work dedicated to the       both quantization accuracy and hardware efficiency.
    quantization and acceleration of efficient ViTs.
  • At the algorithm level, we conduct a comprehensive anal-        B. Efficient ViTs
    ysis of distinct activations of Softmax-free efficient ViTs        ViTs [4], [5], [13], [14], [23]–[25] have gained growing
    and unveil specific quantization challenges. Then, we           attention recently and have been developed rapidly in the
    develop a tailored post-training quantization engine that       computer vision field. Among them, ViT [4] firstly applies
    incorporates several novel strategies, including channel-       a pure Transformer to process sequences of image patches,
    wise migration, filter-wise shifting, and log2 quanti-          achieving remarkable performance. Furthermore, DeiT [5]
    zation, to address the involved challenges with boosted         offers a better training recipe for ViT, significantly reducing
    quantization accuracy.                                          training costs. However, standard ViTs still incur intensive
  • At the hardware level, we advocate a hybrid design incor-       computational costs and huge memory footprints during in-
    porating multiple computing cores to effectively support        ference to achieve superior performance, calling for efficient
    various operation types in the Convolution-Transformer          ViTs [13], [14], [23]–[25]. Particularly, EfficientViT [13], the
    hybrid architecture of efficient ViTs. Besides, we propose      SOTA one, replaces the vanilla Softmax-based self-attention
    a pipeline architecture to facilitate bo
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
respectively, we thus explore quantization ARE QUANTIZED TO DIFFERENT BITS and acceleration on top of EfficientViT to win both quantiza- EfficientViT-B1 [13] W8 W8A16 W8A12 W8A10 W8A8 tion accuracy and hardware efficiency. Top-1 Accuracy* (%) 79.39 79.32 78.86 75.08 3.23 We adopt the most widely applied hardware-friendly quanti- Drop (%) ↑0.01 ↓0.06 ↓0.52 ↓4.30 ↓76.15 zation setting [6] by default, i.e., the symmetric layer-wise and * Tested on ImageNet with the input size of 224 × 224 by default. filter-wise uniform quantization for activations X and weights W , respectively. Formally, as expressed in Eq. (5), XQ /WQ are TABLE II quantized X/W , Sa /Sw are the corresponding scaling factors, ACCURACY OF E FFICIENT V I T-B1 [13] WHEN WEIGHTS AND ACTIVATIONS ( EXCEPT M AT M ULS IN MSA S ) ARE BOTH QUANTIZED TO 8- BIT ⌊·⌉ means rounding to the nearest, and b is the quantization bit- Effici
```
- keyword `Cityscapes`:
```text
c Segmentation. To assess the DW’s inputs and inter-channel asymmetries in PW2’s inputs, generalization capability of our proposed quantization method, as introduced in Sec. IV-A1, vanilla uniform quantization fails we apply it to EfficientViT-B0-R1024 [13] and evaluate its to quantize MBConvs in EfficientViT. (ii) By incorporating performance on the semantic segmentation task, using the our channel-wise migration and filter-wise shifting, which Cityscapes [49] as the dataset and mean Intersection over are proposed to solve the above two issues, respectively, Union (mIoU) as the evaluation metric. As listed in Table we can effectively quantize MBConvs with only an average VII, we can see that: For the quantization within MBConvs, ↓0.54% accuracy. On top of this, regarding the quantization vanilla uniform quantization fails due to the inter-channel of lightweight MSA, (iii) owing to the e
```
- keyword `DeiT`:
```text
ng and unveil specific quantization challenges. Then, we attention recently and have been developed rapidly in the develop a tailored post-training quantization engine that computer vision field. Among them, ViT [4] firstly applies incorporates several novel strategies, including channel- a pure Transformer to process sequences of image patches, wise migration, filter-wise shifting, and log2 quanti- achieving remarkable performance. Furthermore, DeiT [5] zation, to address the involved challenges with boosted offers a better training recipe for ViT, significantly reducing quantization accuracy. training costs. However, standard ViTs still incur intensive • At the hardware level, we advocate a hybrid design incor- computational costs and huge memory footprints during in- porating multiple computing cores to effectively support ference to achieve superior performance, calling for efficient
```
- keyword `EfficientViT`:
```text
or ViT, significantly reducing quantization accuracy. training costs. However, standard ViTs still incur intensive • At the hardware level, we advocate a hybrid design incor- computational costs and huge memory footprints during in- porating multiple computing cores to effectively support ference to achieve superior performance, calling for efficient various operation types in the Convolution-Transformer ViTs [13], [14], [23]–[25]. Particularly, EfficientViT [13], the hybrid architecture of efficient ViTs. Besides, we propose SOTA one, replaces the vanilla Softmax-based self-attention a pipeline architecture to facilitate both inter- and intra- of quadratic complexity with a novel Softmax-free lightweight layer fusions, thus enhancing hardware utilization and multi-scale attention, achieving a global receptive field while easing the bandwidth requirement. enhancing hardware efficiency. B
```

## 7. 실험 결과
- keyword `speedup`:
```text
– – 28.75 0.94 Ablation Study ✓ ✓ ✓ 1.12 ✓ ✓ – – 79.05 81.36 in MSA ✓ ✓ ✓ 74.83 ✓ ✓ ✓ NaN NaN ✓ ✓ ✓ 78.64 80.97 * denotes the 8-bit uniform quantization and 4-bit log2 quantization. 10000 EdgeGPU T egra X2 TABLE VI FasterTransformer 1000 ACCURACY COMPARISONS BETWEEN VANILLA CHANNEL - WISE ( CW ) I-BERT Throughput (GOPS)(x) QUANTIZATION AND OUR PROPOSED CHANNEL - WISE ( CW ) MIGRATION 100 4,438.8 6242 4,136.3 4,297.9 I-ViT 3,097.6 2,822.1 3,016.4 Speedup 1,614.9 1,494.3 1,566.3 6328 703 EdgeCPU 382.3 1217 281.3 Quantization for EfficientViT EfficientViT EfficientViT EfficientViT 136.6 10 63.2 54.7 Jetson Nano 41.9 DW’s Inputs -B1-R224 -B1-R256 -B1-R288 -B2-R224 Jetson Orin 1 CW Quantization 69.25 78.83 79.36 79.98 DeiT-Tiny DeiT-Small DeiT-Tiny DeiT-Smal DeiT-Base EfficientViT-B1 ientViT-B2 ﬃcientViT-B1 Deit-Base Eﬃ EfficientViT-B2 Ours CW Migration 78.64 78.93 79.58 80.97 Improve (%) ↑9.
```
- keyword `energy`:
```text
eLU(Qi−1 ) and the already obtained ReLU(Ki−1 )Tsum as including (viii) Auto-ViT-Acc [17] and (ix) Huang et al. [46] well as Si−1 to generate divisors and dividends in Eq. (4) for tailored for standard ViTs and (x) ViA [47] dedicated to Swin the (i − 1)th head, respectively. This means that steps iv/iii in Transformer [23] (one of efficient ViTs). We compare them in the Sec. V-A2 are computed consecutively on the MAT engine. terms of throughput, energy efficiency, frame rate (FPS), and During this process, the firstly generated divisors are cached DSP efficiency. in the divisor buffer and then routed to the log2 quantization Accelerator Setup. Characteristics: The parallelism of module for log2 quantization. (iii) Once the dividends are computing engines in our accelerator (N × M + T × S) × L obtained, they can be re-quantized via the re-quantization (as depicted in Fig. 8) is configured
```
- keyword `accuracy`:
```text
[4] contains compression methods, such as quantization. Unfortunately, due to 307M parameters and yields 190.7G FLOPs during inference. the existence of hardware-unfriendly and quantization-sensitive Thus, effective model compression techniques are highly de- non-linear operations, particularly Softmax, it is non-trivial sired to facilitate ViTs’ real-world applications. to completely quantize all operations in ViTs, yielding either significant accuracy drops or non-negligible hardware costs. In Among them, model quantization [6]–[9] stands out as response to challenges associated with standard ViTs, we focus our one of the most effective and widely adopted compression attention towards the quantization and acceleration for efficient methods. It converts floating-point weights/activations into ViTs, which not only eliminate the troublesome Softmax but also integers, leading to a reducti
```
- keyword `throughput`:
```text
n-based operations in EfficientViT, ex- computations in MSAs). This architecture inherently offers cluding DWConvs, as explained in Design Choice # 1 in Sec. opportunities for pipeline processing, where various operations V-A1. Besides, the log2 quantization module is developed to can be simultaneously executed on distinct computing units, quantize divisors in Eq. (4) following steps outlined at the end thereby enhancing hardware utilization and throughput. As for of the first paragraph in Sec. IV-D, thus boosting quantization the inter-layer pipeline, as shown in Figs. 9 (a) and (b), when accuracy as well as enabling the conversion of costly divi- the R-MAC engine handles DWConv, the resulting outputs sions into hardware-efficient bit-wise shifts. Our accelerator are first subtracted by the channel-wise mean obtained on the is also equipped with several low-cost auxiliary processors, ca
```
- keyword `GOPS`:
```text
79.39 82.10 ✓ – – 36.89 ✓ ✓ NaN NaN Ablation Study ✓ – – 72.51 ✓ – – 3.23 0.68 in MBConv ✓ – – 29.02 ✓ – – 7.51 78.52 ✓ ✓ – – 75.22 ✓ – – 28.75 0.94 Ablation Study ✓ ✓ ✓ 1.12 ✓ ✓ – – 79.05 81.36 in MSA ✓ ✓ ✓ 74.83 ✓ ✓ ✓ NaN NaN ✓ ✓ ✓ 78.64 80.97 * denotes the 8-bit uniform quantization and 4-bit log2 quantization. 10000 EdgeGPU T egra X2 TABLE VI FasterTransformer 1000 ACCURACY COMPARISONS BETWEEN VANILLA CHANNEL - WISE ( CW ) I-BERT Throughput (GOPS)(x) QUANTIZATION AND OUR PROPOSED CHANNEL - WISE ( CW ) MIGRATION 100 4,438.8 6242 4,136.3 4,297.9 I-ViT 3,097.6 2,822.1 3,016.4 Speedup 1,614.9 1,494.3 1,566.3 6328 703 EdgeCPU 382.3 1217 281.3 Quantization for EfficientViT EfficientViT EfficientViT EfficientViT 136.6 10 63.2 54.7 Jetson Nano 41.9 DW’s Inputs -B1-R224 -B1-R256 -B1-R288 -B2-R224 Jetson Orin 1 CW Quantization 69.25 78.83 79.36 79.98 DeiT-Tiny DeiT-Small DeiT-Tiny DeiT-Smal De
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization
- exact softmax vs approximated softmax
- Taylor/linear attention vs exact attention
- fixed sparse token pattern vs dense token pattern

## 9. HGTXR 적용 판단
- softmax-free 변경은 HGTXR 논문 범위를 벗어날 수 있으므로 P2 paper-scope review가 필요하다.

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
