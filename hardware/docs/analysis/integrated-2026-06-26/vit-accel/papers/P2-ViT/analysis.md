---
source_type: paper
source_name: P2-ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/P2-ViT/analysis.md -->

# P2-ViT Detailed Paper Analysis

- title: P2-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer
- url: https://arxiv.org/abs/2405.19915
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/P2-ViT.pdf`
- related_codebases: `P2-ViT`

## 1. 기존 방법의 문제점
- 요약: 기존 ViT quantization은 floating scale factor를 유지해 requant overhead가 크고 hardware 효율을 제한한다.

### Paper Evidence: Abstract
```text
Abstract—Vision Transformers (ViTs) have excelled in com-                achieving 87.76% top-1 accuracy on ImageNet [4]. This chal-
arXiv:2405.19915v1 [cs.AI] 30 May 2024




                                         puter vision tasks but are memory-consuming and computation-                lenges the deployment of ViTs on resource-constrained edge
                                         intensive, challenging their deployment on resource-constrained             devices, calling for effective model compression solutions.
                                         devices. To tackle this limitation, prior works have explored
                                         ViT-tailored quantization algorithms but retained floating-point               Model quantization, which converts weights and/or acti-
                                         scaling factors, which yield non-negligible re-quantization over-           vations from floating-point ones to low-precision integers
                                         head, limiting ViTs’ hardware efficiency and motivating more                without modifying network architectures, stands out as a
                                         hardware-friendly solutions. To this end, we propose P2 -ViT, the           generic and effective model compression technology. Thus, to
                                         first Power-of-Two (PoT) post-training quantization and acceler-            facilitate ViTs’ deployment, it is natural to adopt quantization
                                         ation framework to accelerate fully quantized ViTs. Specifically,
                                         as for quantization, we explore a dedicated quantization scheme             to reduce both
```

## 2. 제안하는 방법
- 요약: PoT scaling factor 기반 PTQ와 coarse-to-fine automatic mixed precision을 결합한다.

### Paper Evidence: Method Section
```text
Algorithm 1: P2 -ViT’s Post-Training Quantization                                                           Controllor
                                                                                                                                                            PS-MAC

   Input: Full precision ViT and model size constraint




                                                                             Q,K,V
                                                                                      GB
                                                                                             LN                           PE Array
                                                                                                                                                              X        X
   Output: Quantized ViT with power-of-two scaling                                         Shifters           PS-MAC Array               Shifter
                                                                                                                                         Array               <<m
             factors and mixed-precision                                                    MAS            PS-MAC ... PS-MAC                                       +




                                                                      DRAM
                                                                                                                                         S ... S




                                                                                 Act. GB
                                                                                            Array
   // Dedicated Quantization Scheme                                                                        PS-MAC
                                                                                                                    ...
                                                                                                                           PS-MAC        S
                                                                                                                                               ...
                                                                                                                                                     S
                                                                                           Adders
 1 Leverage channel-wise quantization for weights and
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: activation/weight scale을 power-of-two로 제한해 rescale multiply를 shift로 바꾸고, layer별 bit/scale 후보를 coarse-to-fine 탐색한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
-sen University (email: zfwang@nju.edu.cn). • We propose P -ViT (see Fig. 1), a Power-of-Two (PoT) Correspondence are addressed to Wendong Mao and Zhongfeng Wang. post-training quantization and acceleration framework to 2 1 Dedicated Quantization Scheme 1 Chunk-Based Design III and Sec. IV, respectively; Then, Sec. V demonstrates the Adaptive PoT Rounding PoT-Aware Smoothing superiority of our P2 -ViT framework via extensive experiments DRAM and ablation studies; Finally, Sec. VI summarizes this paper. Global Buffer LayerNorm LayerNorm Projection Projection Self-Attn. GELU ... ... PE Array FC1 FC2 + + PS-MAC Array II. R ELATED W ORK AND M OTIVATIONS Co- LN Design A. Vision Transformers (ViTs) Shifter Array Hessian-Based Coarse-Grained Softmax Re-Quant Motivated by the powerful capability of the self-attention Evolutionary Algorithm Based Fine-Grained 2 Mixed-Precision Quantization 2 Tail
```
- keyword `bit-width`:
```text
, 2b − 1). (4) Ceil ✓ 8/8/4 69.82 76.80 79.36 2 · Sg Floor 68.29 73.26 80.09 µ(X) = µ(2α Sg · XQ ) = Sg · µ(XQ << α), Baseline ✗ 4/8/4 65.63 76.07 79.65 (5) Nearest 63.39 71.94 78.01 σ 2 (X) = σ 2 (2α Sg · XQ ) = Sg · σ 2 (XQ << α). Ceil ✓ 4/8/4 62.46 71.49 76.66 ii) LIS. Besides LN’s inputs, outliers also exist in ViTs’ Floor 61.49 69.04 78.17 attention maps M (i.e., the outputs of Softmax(Q, K, d) in † denotes PoT scaling factors; ∗ represents bit-widths for weights, activa- Eq. (2)). It motivates FQ-ViT [7] to adopt log2 quantization tions, and attention maps, respectively; following Eq. (6), which is on par with the 8-bit uniform quantization with MQ (i.e., the quantized M ) being repre- where αx , αw , and αy indicate exponents of PoT scaling sented with only 4-bit (b = 4 here). By doing this, the factors for input, weight, and output, respectively. subsequent matrix multiplications
```
- keyword `sparsity`:
```text
[15], [28], [29] have constructed dedicated accelerators to Perceptron (MLP). ’MatMul.’ is the abbreviation of matrix multiplications. Activation Weight boost Transformers’ hardware efficiency. For example, for Channel dim. Feature dim. NLP tasks, Sanger [28] and DOTA [30] dynamically prune Token dim. Channel dim. the computation-intensive attention maps in Transformers and further develop a reconfigurable architecture to support the X resulting sparsity patterns. For CV tasks, ViTCoD [12] and HeatViT [16] apply static attention map pruning and adaptive (a) Layer-Wise (b) Group-Wise (c) Channel-Wise (d) Feature-Wise token pruning for ViTs, respectively, then build dedicated Fig. 4. Illustrating the (a) layer-wise, (b) group-wise, and (c) channel-wise accelerators to accelerate resultant sparsity workloads. Addi- quantization for activations, and (d) feature-wise quantization for weights.
```

## 4. 하드웨어 아키텍처
- 요약: chunk-based accelerator와 row-stationary dataflow가 PoT requant pipeline을 이용해 throughput을 높인다.

### Paper Evidence: Architecture / Implementation
```text
architecture and specific algorithmic characteristics.                                                                  LayerNorm
                                                                                                             Blocks
                                                                                                                                              Softmax
   Thus, to boost ViTs’ re-quantization efficiency and facilitate                                                            +

their real-world applications, we conduct a comprehensive                                                                  MSA                   MatMul.
analysis of ViTs’ properties and develop a dedicated quanti-                                                            LayerNorm               Q     K    V
zation scheme to fully quantize ViTs with PoT scaling factors                                                                              Linear Projection
                                                                                                                      Token & Position
without any fine-tuning.                                                                                                Embedding



C. Transformer Accelerators                                          Fig. 3. The illustration of standard Vision Transformers’ (ViTs’) architecture
                                                                     (e.g., ViT [1] and DeiT [2]) that consists of multiple Transformer blocks. Each
   Apart from the algorithmic optimization, many works [12]–         block includes a Multi-head Self-Attention module (MSA) and a Multi-Layer
[15], [28], [29] have constructed dedicated accelerators to          Perceptron (MLP). ’MatMul.’ is the abbreviation of matrix multiplications.
                                                                                                       Activation                                                  Weight
boost Transformers’ hardware efficiency. For example, for
                                                                                  Channel dim.                                                                  Feature dim.
NLP tasks, Sanger [28] and DOTA [30] dynamically prune




                                                                     Token dim.




                                                                                                                                               Channel dim.
the computation-intensive attention maps in Transformers
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
ization, Vision leverages pruning to reduce ViTs’ redundancy, then develops Transformer, ViT accelerator, fully quantized ViT. a dedicated accelerator to gain hardware speedup. Addi- I. I NTRODUCTION tionally, VAQF [14] and Auto-ViT-Acc [15] implement the T HANKS to the powerful capability of Transformers’ self- attention mechanism in extracting global information, Vision Transformers (ViTs) have shown great potential and acceleration of ViTs on FPGAs with FPGA-aware automatic quantization. While dedicated accelerators can significantly enhance hardware efficiency, existing ViT accelerators [14]– achieved remarkable success in various computer vision (CV) [16] typically focus on the acceleration of matrix multiplica- tasks [1]–[3]. Despite their promising performance, ViTs typ- tions while ignoring the remaining non-linear operations and ically have more parameters and intensive computat
```
- keyword `DSP`:
```text
uding the (RTN) method, which yields 0.90%∼4.72% accuracy degra- GPU equipped with quantization acceleration strategies (such dation, further verifying our effectiveness. as FastTransformer [49], I-BERT [33], and I-ViT [11]), the 12 TABLE VII TABLE IX C OMPARISONS WITH THE SOTA V I T ACCELERATOR AUTO -V I T-ACC [15] C OMPARISONS TO SOTA V I T ACCELERATORS [9], [16] Formats DeiT-Small DeiT-Base Average Models DeiT-Tiny DeiT-Small Methods W8A8 FPS/DSP GOPS/DSP FPS/DSP GOPS/DSP Imprv. (×) Methods HeatViT [16] TCAS-1’23 [9] Ours HeatViT [16] TCAS-1’23 [9] Ours Auto-ViT- Fixed 0.040 0.367 0.013 0.435 1.00 Formats W8 W8A8 W8A8 W8 W8A8 W8A8 GOPS/DSP 0.160 0.486 0.890 0.174 0.601 0.923 Acc [15] Fixed+PoT 0.064 0.585 0.022 0.759 1.74 Ours Fixed 0.068 0.623 0.023 0.799 1.84 TABLE X A BLATION STUDIES OF OUR LOW- BIT AND MIXED - PRECISION TABLE VIII QUANTIZATION WHEN EXECUTED ON OUR DEDICATED ACCELE
```
- keyword `systolic`:
```text
Layer Intra-layer Attention MLP Overall quantization for fully quantized vision transformer,” in International Joint Conference on Artificial Intelligence, 2021. ✗ ✗ 1.79 1.15 1.57 [8] Z. Yuan, C. Xue, Y. Chen, Q. Wu, and G. Sun, “Ptq4vit: Post- ✓ ✗ 1.45 1.00 1.32 training quantization framework for vision transformers,” ArXiv, vol. ✗ ✓ 1.34 1.15 1.37 abs/2111.12293, 2021. ✓ ✓ 1.00 1.00 1.00 [9] M. Huang et al., “An integer-only and group-vector systolic accelerator for efficiently mapping vision transformer on edge,” IEEE Transactions on Circuits and Systems I: Regular Papers, 2023. the reconfigurable overhead, and further design a tailored [10] H. Yao, P. Li, J. Cao, X. Liu, C. Xie, and B. Wang, “Rapq: Rescuing row-stationary dataflow to embrace the pipeline opportunity accuracy for power-of-two low-bit post-training quantization,” in Inter- national Joint Conference on Artificial Inte
```

## 5. 실험 방법
- 실험 section heading을 자동 추출하지 못했다. PDF 원문 확인 필요.

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
1 P2-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer Huihong Shi, Xin Cheng, Wendong Mao, and Zhongfeng Wang, Fellow, IEEE Abstract—Vision Transformers (ViTs) have excelled in com- achieving 87.76% top-1 accuracy on ImageNet [4]. This chal- arXiv:2405.19915v1 [cs.AI] 30 May 2024 puter vision tasks but are memory-consuming and computation- lenges the deployment of ViTs on resource-constrained edge intensive, challenging their deployment on resource-constrained devices, calling for effective model compression solutions. devices. To tackle this limitation, prior works have explored ViT-tailored quantization algorithms but retained floating-point Model quantizat
```
- keyword `DeiT`:
```text
-ViT’s algorithm integrates a dedicated quanti- a pure Transformer model to process sequences of image zation scheme to obtain fully quantized ViTs with Power-of-Two (PoT) scal- patches, achieving competitive results over SOTA CNNs when ing factors, and further comprises coarse-to-fine automatic mixed-precision quantization to achieve better accuracy-efficiency trade-offs. (b) Furthermore, pretrained on extremely large datasets. On top of that, DeiT [2] P2 -ViT’s dedicated accelerator advocates a chunk-based design incorporating further refines training recipes for ViT and offers comparable a tailored row-stationary dataflow to boost hardware efficiency. results when only pretrained on ImageNet [4], thus saving (a) FP Scaling Factors (b) PoT Scaling Factors training costs. Furthermore, many efforts [3], [18] have been Input (INT8) Weight (INT8) Input (INT8) Weight (INT8) made to boost V
```
- keyword `EfficientViT`:
```text
aining/fine-tuning costs, we 455. [17] M. Alwani, H. Chen, M. Ferdman, and P. Milder, “Fused-layer cnn can expect that our post-training method is potentially well- accelerators,” 2016 49th Annual IEEE/ACM International Symposium positioned to compress and accelerate them without necessitat- on Microarchitecture (MICRO), pp. 1–12, 2016. ing fine-tuning, thus enhancing the immersive user experiences [18] H. Cai, J. Li, M. Hu, C. Gan, and S. Han, “Efficientvit: Lightweight multi-scale attention for high-resolution dense prediction,” in Proceed- in our daily life. ings of the IEEE/CVF International Conference on Computer Vision, Future Works. Despite P2 -ViT is effective for standard 2023, pp. 17 302–17 313. ViTs [1], [2], it overlooks the inherent advantages of ef- [19] B. Jacob et al., “Quantization and training of neural networks for efficient integer-arithmetic-only inference,” 2018 IEE
```
- keyword `dataset`:
```text
work. Specifically, (a) P2 -ViT’s algorithm integrates a dedicated quanti- a pure Transformer model to process sequences of image zation scheme to obtain fully quantized ViTs with Power-of-Two (PoT) scal- patches, achieving competitive results over SOTA CNNs when ing factors, and further comprises coarse-to-fine automatic mixed-precision quantization to achieve better accuracy-efficiency trade-offs. (b) Furthermore, pretrained on extremely large datasets. On top of that, DeiT [2] P2 -ViT’s dedicated accelerator advocates a chunk-based design incorporating further refines training recipes for ViT and offers comparable a tailored row-stationary dataflow to boost hardware efficiency. results when only pretrained on ImageNet [4], thus saving (a) FP Scaling Factors (b) PoT Scaling Factors training costs. Furthermore, many efforts [3], [18] have been Input (INT8) Weight (INT8) Input (INT8) Wei
```

## 7. 실험 결과
- keyword `speedup`:
```text
hile overlooking factors, thereby enhancing throughput. Extensive experiments and retraining scaling factors in floating-point. This yields non- consistently validate P2 -ViT’s effectiveness. Particularly, we offer comparable or even superior quantization performance with PoT negligible re-quantization overheads and hinders ViTs’ integer- scaling factors when compared to the counterpart with floating- only inference (see Fig. 2a), limiting their speedup on existing point scaling factors. Besides, we achieve up to 10.1× speedup hardware platforms [10], [11] and motivating the exploration and 36.8× energy saving over GPU’s Turing Tensor Cores, and for more hardware-efficient solutions. up to 1.84× higher computation utilization efficiency against In parallel, prior works [9], [12]–[16] have developed ded- SOTA quantization-based ViT accelerators. Codes are available at https://github.com/s
```
- keyword `energy`:
```text
tly validate P2 -ViT’s effectiveness. Particularly, we offer comparable or even superior quantization performance with PoT negligible re-quantization overheads and hinders ViTs’ integer- scaling factors when compared to the counterpart with floating- only inference (see Fig. 2a), limiting their speedup on existing point scaling factors. Besides, we achieve up to 10.1× speedup hardware platforms [10], [11] and motivating the exploration and 36.8× energy saving over GPU’s Turing Tensor Cores, and for more hardware-efficient solutions. up to 1.84× higher computation utilization efficiency against In parallel, prior works [9], [12]–[16] have developed ded- SOTA quantization-based ViT accelerators. Codes are available at https://github.com/shihuihong214/P2-ViT. icated accelerators to boost ViTs’ hardware efficiency from the hardware perspective. For example, ViTCoD [12] first Index Terms—Powe
```
- keyword `accuracy`:
```text
1 P2-ViT: Power-of-Two Post-Training Quantization and Acceleration for Fully Quantized Vision Transformer Huihong Shi, Xin Cheng, Wendong Mao, and Zhongfeng Wang, Fellow, IEEE Abstract—Vision Transformers (ViTs) have excelled in com- achieving 87.76% top-1 accuracy on ImageNet [4]. This chal- arXiv:2405.19915v1 [cs.AI] 30 May 2024 puter vision tasks but are memory-consuming and computation- lenges the deployment of ViTs on resource-constrained edge intensive, challenging their deployment on resource-constrained devices, calling for effective model compression solutions. devices. To tackle this limitation, prior works have explored ViT-tailored quantization algorithms but retained floating-point Mod
```
- keyword `throughput`:
```text
oftmax), thus offering fully quantized types of operations, alleviating reconfigurable overhead. Addi- ViTs. However, despite the effectiveness of existing ViT quan- tionally, we design a tailored row-stationary dataflow to seize the tization methods, they generally target quantization of linear pipeline processing opportunity introduced by our PoT scaling [5], [6], [8]/non-linear [7], [9]) operations while overlooking factors, thereby enhancing throughput. Extensive experiments and retraining scaling factors in floating-point. This yields non- consistently validate P2 -ViT’s effectiveness. Particularly, we offer comparable or even superior quantization performance with PoT negligible re-quantization overheads and hinders ViTs’ integer- scaling factors when compared to the counterpart with floating- only inference (see Fig. 2a), limiting their speedup on existing point scaling factors. B
```
- keyword `GOPS`:
```text
g the (RTN) method, which yields 0.90%∼4.72% accuracy degra- GPU equipped with quantization acceleration strategies (such dation, further verifying our effectiveness. as FastTransformer [49], I-BERT [33], and I-ViT [11]), the 12 TABLE VII TABLE IX C OMPARISONS WITH THE SOTA V I T ACCELERATOR AUTO -V I T-ACC [15] C OMPARISONS TO SOTA V I T ACCELERATORS [9], [16] Formats DeiT-Small DeiT-Base Average Models DeiT-Tiny DeiT-Small Methods W8A8 FPS/DSP GOPS/DSP FPS/DSP GOPS/DSP Imprv. (×) Methods HeatViT [16] TCAS-1’23 [9] Ours HeatViT [16] TCAS-1’23 [9] Ours Auto-ViT- Fixed 0.040 0.367 0.013 0.435 1.00 Formats W8 W8A8 W8A8 W8 W8A8 W8A8 GOPS/DSP 0.160 0.486 0.890 0.174 0.601 0.923 Acc [15] Fixed+PoT 0.064 0.585 0.022 0.759 1.74 Ours Fixed 0.068 0.623 0.023 0.799 1.84 TABLE X A BLATION STUDIES OF OUR LOW- BIT AND MIXED - PRECISION TABLE VIII QUANTIZATION WHEN EXECUTED ON OUR DEDICATED ACCELERATO
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization

## 9. HGTXR 적용 판단
- 가장 직접적인 P0 후보. HGTXR Q4 weight/Q8 activation에서 scale을 shift-friendly하게 재보정하면 정확도와 resource를 동시에 개선할 수 있다.

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
