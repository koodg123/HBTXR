---
source_type: paper
source_name: AHCPTQ
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/AHCPTQ/analysis.md -->

# AHCPTQ Detailed Paper Analysis

- title: AHCPTQ: Accurate and Hardware-Compatible Post-Training Quantization for Segment Anything Model
- url: https://arxiv.org/abs/2503.03088
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/AHCPTQ.pdf`
- related_codebases: `AHCPTQ`

## 1. 기존 방법의 문제점
- 요약: SAM 계열 모델은 post-GELU activation의 heavy-tail/skew와 projection activation의 channel variation 때문에 일반 PTQ가 정확도를 크게 잃기 쉽다.

### Paper Evidence: Abstract
```text
Abstract— The Segment Anything Model (SAM) has revolutionized image and video segmentation with its powerful zero-shot capabili-
                                             ties. However, its massive parameter scale and high computational demands hinder efficient deployment on resource-constrained edge
                                             devices. While Post-Training Quantization (PTQ) offers a practical solution, existing methods still fail to handle four critical quantization
                                             challenges: (1) ill-conditioned weights; (2) skewed and long-tailed post-GELU activations; (3) pronounced inter-channel variance in
                                             linear projections; and (4) exponentially scaled and heterogeneous attention scores. To mitigate these bottlenecks, we propose AHCQ-
arXiv:2503.03088v4 [cs.CV] 8 Apr 2026




                                             SAM, an accurate and hardware-compatible PTQ framework featuring four synergistic components: (1) Activation-aware Condition
                                             Number Reduction (ACNR), which regularizes weight matrices via a proximal point algorithm to suppress ill-conditioning; (2) Hybrid
                                             Log-Uniform Quantization (HLUQ), which combines power-of-two and uniform quantizers to capture skewed post-GELU activations; (3)
                                             Channel-Aware Grouping (CAG), which clusters channels with homogeneous statistics to achieve high accuracy with minimal hardware
                                             overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust qua
```

## 2. 제안하는 방법
- 요약: Hybrid Log-Uniform Quantization(HLUQ)과 Channel-Aware Grouping(CAG)을 결합해 hardware-compatible PTQ를 구성한다.

### Paper Evidence: Method Section
```text
methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN
                                             on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement
                                             in J &F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ-
                                             SAM, delivering a 7.12× speedup and a 6.62× power efficiency improvement over the floating-point baseline. Code is available at
                                             https://github.com/Wenlun-Zhang/AHCQ-SAM.

                                             Index Terms—Segment Anything Model, Network quantization, Post-training quantization, Vision transformers.

                                                                                                                      ✦



                                        1    I NTRODUCTION                                                                of SAM relies on a co-developed data engine and conse-
                                                                                                                          quently utilizes the SA-1B dataset comprising 1.1B masks

                                        T     He Segment Anything Model (SAM) [1], [2] is a power-
                                              ful tool for image/video segmentation, demonstrating
                                        strong zero-shot performance across diverse visual domains
                                                                                                                          and 11M images [1]. As a more practical alternative, Post-
                                                                                                                          Training Quantization (PTQ) has gained increasing atten-
                                                                                                                          tion. PTQ requires only a small calibration dataset, signif-
                                        and broad applicability in real-world scenarios [3], [4], [5],                    icantly reducing data and computational demands while
                                        [6], [7]. However, its large-scale parameters, substantial stor-                  maintaining competitive accuracy [18], [19], [20], [21], [22],
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: HLUQ는 작은 값 dense 영역은 log2 형태로, sparse large 값은 uniform 형태로 다루며 CAG는 유사 activation channel을 묶어 공유 quant parameter를 사용한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
ECQ 66.97 85.77 32.48 45.36 31.33 47.16 QDrop 74.07 85.94 45.35 63.82 46.47 66.50 SAM2-Small 88.28 76.10 PTQ4SAM 76.67 86.04 47.84 70.45 77.43 50.07 73.41 TODO 78.02 86.32 48.72 70.85 52.11 74.46 BRECQ 25.37 80.59 20.71 47.43 18.48 47.24 QDrop 30.59 83.48 13.21 55.12 14.08 57.95 SAM2-Base+ 88.61 76.35 78.74 PTQ4SAM 29.19 83.29 25.88 64.03 25.29 65.50 TODO 31.44 83.69 26.92 64.69 26.11 65.95 QDrop and PTQ4SAM by 6.20% and 8.85%, respectively. 4.3 Ablation Studies Notably, on the more challenging SA-V Val and SA-V Test 4.3.1 Ablation of Components sets, AHCQ-SAM respectively elevates the performance of 4-bit SAM2-Tiny to 47.34% and 49.71%, representing a sub- Tab. 3 presents the ablation study to evaluate the con- stantial improvement of 12.55% and 11.22% compared to tributions of ACNR, HLUQ, CAG, and LNQ within the the best-performing baseline. The advantages of AHCQ- AHCQ-SAM framework.
```
- keyword `bit-width`:
```text
struction j x m  We employ block-wise reconstruction [45], as adopted in xq = clamp + z, 0, 2k − 1 , (1) PTQ4SAM [29], to mitigate the quantization-induced error s in weight and activation quantization by minimizing the mean squared error: x ≈ x̂ = s · (xq − z). (2) L = ∥OB − ÔB ∥22 , (6) Here, x is the original floating-point input, xq is the quan- where OB and ÔB represent the floating-point and quan- tized integer representation, k is the bit-width, s is the tized outputs of the B -th block, respectively. scale factor, and z is the zero point. The rounding function ⌊·⌉ ensures proper discretization. Uniform quantization is widely adopted due to its straightforward hardware imple- 3.2 AHCQ-SAM mentation, allowing integer arithmetic to replace floating- point operations, leading to higher efficiency and lower To advance SAM quantization, we propose AHCQ-SAM, a computational cost. Fo
```
- keyword `mixed precision`:
```text
“Learn- bit quantization for image super-resolution,” arXiv preprint able lookup table for neural network quantization,” Conference on arXiv:2502.15478, 2025. Computer Vision and Pattern Recognition, pp. 12 413–12 423, 2022. [34] X. Sun, J. Liu, H. Shen, X. Zhu, and P. Hu, “On efficient variants [56] V. Chikin and M. Antiukh, “Data-free network compression via of segment anything model: A survey,” International Journal of parametric non-uniform mixed precision quantization,” in Confer- Computer Vision, vol. 133, no. 10, pp. 7406–7436, 2025. ence on Computer Vision and Pattern Recognition, 2022, pp. 450–459. [35] C. Zhou, X. Li, C. C. Loy, and B. Dai, “Edgesam: Prompt-in-the- [57] C. Hong, H. Kim, J. Oh, and K. M. Lee, “Daq: distribution-aware loop distillation for sam,” International Journal of Computer Vision, quantization for deep image super-resolution networks,” arXiv vol. 133, no.
```

## 4. 하드웨어 아키텍처
- 요약: FPGA 구현에서 HLUQ/CAG가 복잡한 full-precision 보정 없이 quantized execution에 맞게 설계된다.

### Paper Evidence: Architecture / Implementation
```text
implementation by reducing on-chip register overhead by                                                      metic computations.
99.7%. Finally, to accommodate the exponentially scaled                                                          By integrating these innovations, AHCQ-SAM signifi-
and heterogeneous attention scores, LNQ utilizes a loga-                                                     cantly reduces accuracy degradation at 5-bit precision and

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE UNDER REVIEW                                                    3

improves 4-bit performance by a large margin. Furthermore,         sparse window routing and memory retrieval, effectively
we extend AHCQ-SAM to establish a PTQ benchmark for                filtering background redundancy. However, these methods
SAM2 [2], which targets Video Object Segmentation (VOS)            still rely on expensive full-precision computation.
tasks. Experimental results demonstrate that AHCQ-SAM
consistently outperforms existing state-of-the-art methods         2.2   Post-training Quantization
across all metrics. For instance, AHCQ-SAM improves the            Post-training Quantization (PTQ) utilizes a small calibration
mAP by 15.2% over PTQ4SAM for 4-bit SAM-B with Faster              dataset to determine quantization parameters. Compared
R-CNN on the COCO dataset, and increases the J &F score            with quantization-aware training (QAT), it enables rapid
by 14.01% for 4-bit SAM2-Tiny on the SA-V Test dataset.            deployment on edge devices without requiring extensive re-
Finally, we validate the practical effectiveness of AHCQ-          training. AdaRound [43] identifies the sensitivity of weight
SAM by implementing it on an FPGA-based accelerator,               rounding and introduces an optimization technique to re-
demonstrating significant gains in both processing speed           duce overall model loss. BRECQ [44] employs block recon-
and power efficiency. Our primary contributions are sum-           struction to strike a balance between cross-layer dependency
marized as follows:                                                and generalization error. QDrop [45] integrates dropout
• We systematically identify four critical challenges in SAM       into the reconstruction process to improve the flatness of
  quantization: (1) ill-conditioned weights causing sensitiv-      the optimized models. Despite their success, these methods
  ity to quantization errors; (2) skewed and long-tailed post-     are primarily designed for CNN-based mode
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in J &F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ- SAM, delivering a 7.12× speedup and a 6.62× power efficiency improvement over the floating-point baseline. Code is available at https://github.com/Wenlun-Zhang/AHCQ-SAM. Index Terms—Segment Anything Model, Network quantization, Post-training quantization, Vision transformers. ✦ 1 I NTRODUCTION of SAM relies on a co-developed data engine and conse- quently utilizes the SA-1B datase
```
- keyword `ZCU102`:
```text
itative comparison of segmentation masks generated by different quantization methods on W4A4 SAM-H with YOLOX. AHCQ-SAM closely matches the floating-point reference, significantly outperforming other baselines. 40 for SAM-B, where CAG, HLUQ, and LNQ are applied to the 35 corresponding layers, as illustrated in Fig. 2. The accelerator 32.6 is implemented in Verilog, synthesized using Vivado De- 30 29.5 29.2 27.9 sign Suite, and deployed on an AMD ZCU102 evaluation 25 mAP (%) board operating at 300 MHz. The overall system archi- 20 tecture comprises three components: an FPGA accelerator 17.9 15 responsible for large-scale computation, a DDR4 DRAM for 10 CondiQuant data buffering, and a host PC that transfers activations via 5 ACNR Ethernet, as depicted in Fig. 13. For benchmarking purposes, 3.8 0 we implemented two baseline accelerators: a standard FP32 SAM-B SAM-L SAM-H accelerator and a
```
- keyword `Vivado`:
```text
UNDER REVIEW 12 FP32 BRECQ QDrop PTQ4SAM AHCQ-SAM Fig. 11: Qualitative comparison of segmentation masks generated by different quantization methods on W4A4 SAM-H with YOLOX. AHCQ-SAM closely matches the floating-point reference, significantly outperforming other baselines. 40 for SAM-B, where CAG, HLUQ, and LNQ are applied to the 35 corresponding layers, as illustrated in Fig. 2. The accelerator 32.6 is implemented in Verilog, synthesized using Vivado De- 30 29.5 29.2 27.9 sign Suite, and deployed on an AMD ZCU102 evaluation 25 mAP (%) board operating at 300 MHz. The overall system archi- 20 tecture comprises three components: an FPGA accelerator 17.9 15 responsible for large-scale computation, a DDR4 DRAM for 10 CondiQuant data buffering, and a host PC that transfers activations via 5 ACNR Ethernet, as depicted in Fig. 13. For benchmarking purposes, 3.8 0 we implemented two baseline ac
```

## 5. 실험 방법
```text
results are evaluated on 4-bit SAM-B/L/H models with                ger accelerator, are synthesized using Vitis HLS. In contrast,
Faster R-CNN.                                                       the PEs of the FP32 accelerator are implemented using the
                                                                    Floating-Point Operator IP generator and utilize on-chip
                                                                    DSP resources.
4.4         Comparison of Visualization Results
Fig. 11 illustrates the visualization results for W4A4 quan-            As shown in Tab. 4, we report the frame rates and
tization of SAM-H using YOLOX. In comparison with                   power efficiency of SAM-B when the outputs of Faster R-
existing methods such as BRECQ [44], QDrop [45], and                CNN are used as box prompts. We also summarize the
PTQ4SAM [29], AHCQ-SAM consistently generates seg-                  FPGA resource utilization and supported parallelism for
mentation masks that are resemble the original floating-            each configuration. The FP32 accelerator is limited by the
point model, preserving fine structural details and sharp           number of available on-chip DSP resources and supports
object boundaries. These qualitative results demonstrate            only 16 parallel PE lanes. In contrast, both the default INT8
that AHCQ-SAM effectively mitigates the quantization chal-          accelerator and the proposed AHCQ-SAM INT4 accelera-
lenges inherent in SAM, achieving segmentation quality on           tor replace DSP-based PEs with LUT-based PEs, enabling
par with the floating-point baseline. This further confirms         a substantial increase in parallelism to 64 and 128 lanes,
its efficacy and robustness for practical low-bit deployment.       respectively. Meanwhile, the DSP resources in the integer
                                                                    accelerators are allocated to complex arithmetic operations
                                                                    and the quantization/dequantization processes, improving
4.5         Hardware Validation                                     overall resource utilization and system-level performance.
To evaluate the resource efficiency and practical perfor-           In addition, the BRAM resources remain largely under-
mance of AHCQ-SAM in real-world applications, we devel-             utilized in the accelerators. We therefore leverage on-chip
oped an FPGA-based accelerator. The accelerator is tailored         BRAM as LUTs to simplify the quantization operations in

IEEE TRANSACTIONS ON PATTERN ANALYSIS AND MACHINE INTELLIGENCE UNDER REVIEW                                                              13

LNQ. The additional LUTs require only 3.2 Mb of BRAM, re-       for SAM2, where AHCQ-SAM also outperforms existing
sulting in negligible overhead. As a result, the 4-bit AHCQ-    methods, setting a strong baseline for future
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `COCO`:
```text
atistics to achieve high accuracy with minimal hardware overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in J &F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ- SAM, delivering a 7.12× speedup and a 6.62× power efficiency improvement over the floating-point baseline. Code is available at https://github.com/Wenlun-Zhang/AHCQ-SAM. Index Terms—Segment Anything Model, Netwo
```
- keyword `EfficientViT`:
```text
25. ence on Computer Vision and Pattern Recognition, 2022, pp. 450–459. [35] C. Zhou, X. Li, C. C. Loy, and B. Dai, “Edgesam: Prompt-in-the- [57] C. Hong, H. Kim, J. Oh, and K. M. Lee, “Daq: distribution-aware loop distillation for sam,” International Journal of Computer Vision, quantization for deep image super-resolution networks,” arXiv vol. 133, no. 12, pp. 8452–8468, 2025. preprint arXiv:2012.11230, 2020. [36] Z. Zhang, H. Cai, and S. Han, “Efficientvit-sam: Accelerated seg- [58] Y. Li, X. Dong, and W. Wang, “Additive powers-of-two quantiza- ment anything model without performance loss,” in Proceedings of tion: An efficient non-uniform discretization for neural networks,” the IEEE/CVF Conference on Computer Vision and Pattern Recognition, arXiv preprint arXiv:1909.13144, 2019. 2024, pp. 7859–7863. [59] T. Xia, B. Zhao, J. Ma, G. Fu, W. Zhao, N. Zheng, and P. Ren, “An [37] C. Zhang,
```
- keyword `dataset`:
```text
ics to achieve high accuracy with minimal hardware overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in J &F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ- SAM, delivering a 7.12× speedup and a 6.62× power efficiency improvement over the floating-point baseline. Code is available at https://github.com/Wenlun-Zhang/AHCQ-SAM. Index Terms—Segment Anything Model, Network qu
```

## 7. 실험 결과
- keyword `speedup`:
```text
ental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore, we establish a PTQ benchmark for SAM2, where AHCQ-SAM yields a 14.01% improvement in J &F for 4-bit SAM2-Tiny on the SA-V Test dataset. Finally, FPGA-based implementation validates the practical utility of AHCQ- SAM, delivering a 7.12× speedup and a 6.62× power efficiency improvement over the floating-point baseline. Code is available at https://github.com/Wenlun-Zhang/AHCQ-SAM. Index Terms—Segment Anything Model, Network quantization, Post-training quantization, Vision transformers. ✦ 1 I NTRODUCTION of SAM relies on a co-developed data engine and conse- quently utilizes the SA-1B dataset comprising 1.1B masks T He Segment Anything Model (SAM) [1], [2] is a power- ful tool for
```
- keyword `energy`:
```text
s against the actual like CondiQuant’s PGD, which interleaves gradient steps error distribution of quantized activations. (2) The proximal with proximal operations, our approach eliminates gradient point algorithm is more stable for low-bit quantization, as it steps to avoid optimization drift caused by noisy gradients. avoids the noisy gradient updates present in PGD. (3) The Starting from W0 = Worig , we iteratively refine the weights spectral energy preservation strategy selectively protects by solving a sequence of proximal subproblems. At iteration dominant singular directions while adaptively adjusting tail k , with the help of an auxiliary variable Z , we compute the singular values, better balancing condition number reduc- next iterate as: tion with weights’ representational capacity preservation. As shown in the green curve of Fig. 1a, the condition 1 Wk+1 = arg min ∥Z − Wk ∥2F
```
- keyword `accuracy`:
```text
patible PTQ framework featuring four synergistic components: (1) Activation-aware Condition Number Reduction (ACNR), which regularizes weight matrices via a proximal point algorithm to suppress ill-conditioning; (2) Hybrid Log-Uniform Quantization (HLUQ), which combines power-of-two and uniform quantizers to capture skewed post-GELU activations; (3) Channel-Aware Grouping (CAG), which clusters channels with homogeneous statistics to achieve high accuracy with minimal hardware overhead; and (4) Logarithmic Nonlinear Quantization (LNQ), which utilizes logarithmic transformations to adaptively adjust quantization resolution for exponential and heterogeneous attention scores. Experimental results demonstrate that AHCQ-SAM outperforms current methods on SAM. Compared with the SOTA method, it achieves a 15.2% improvement in mAP for 4-bit SAM-B with Faster R-CNN on the COCO dataset. Furthermore
```
- keyword `GOPS`:
```text
R. Rädle, C. Rolland, L. Gustafson et al., “Sam 2: Segment anything LUT Usage 150,540 183,505 177,001 in images and videos,” in International Conference on Learning DSP Usage 1,886 453 474 Representations, 2025. BRAM Usage 253 146 243 [3] J. Cheng, J. Ye, Z. Deng, J. Chen, T. Li, H. Wang, Y. Su, Parallelism 16 64 128 Z. Huang, J. Chen, L. Jiang et al., “Sam-med2d,” arXiv preprint Frame Rate (FPS) 4.75 16.34 33.82 arXiv:2308.16184, 2023. Power (GOPS/W) 7.21 25.11 47.73 [4] R. Zhang, Z. Jiang, Z. Guo, S. Yan, J. Pan, X. Ma, H. Dong, P. Gao, and H. Li, “Personalize segment anything model with one shot,” arXiv preprint arXiv:2305.03048, 2023. [5] Y. Wang, W. Zhou, Y. Mao, and H. Li, “Detect any shadow: Segment anything for video shadow detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 5, pp. 3782– 3794, 2024. [6] A. Maalouf, N. Jadhav, K. M. Jatavalla
```
- keyword `FPS`:
```text
. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson et al., “Sam 2: Segment anything LUT Usage 150,540 183,505 177,001 in images and videos,” in International Conference on Learning DSP Usage 1,886 453 474 Representations, 2025. BRAM Usage 253 146 243 [3] J. Cheng, J. Ye, Z. Deng, J. Chen, T. Li, H. Wang, Y. Su, Parallelism 16 64 128 Z. Huang, J. Chen, L. Jiang et al., “Sam-med2d,” arXiv preprint Frame Rate (FPS) 4.75 16.34 33.82 arXiv:2308.16184, 2023. Power (GOPS/W) 7.21 25.11 47.73 [4] R. Zhang, Z. Jiang, Z. Guo, S. Yan, J. Pan, X. Ma, H. Dong, P. Gao, and H. Li, “Personalize segment anything model with one shot,” arXiv preprint arXiv:2305.03048, 2023. [5] Y. Wang, W. Zhou, Y. Mao, and H. Li, “Detect any shadow: Segment anything for video shadow detection,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 5, pp. 3782– 3
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization

## 9. HGTXR 적용 판단
- 눈 영역 segmentation/ROI preprocessing을 SAM류 보조 네트워크로 붙일 경우 유용하다. HGTXR backbone 자체에는 P0가 아니라 보조 SW accuracy 실험으로 제한한다.

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
