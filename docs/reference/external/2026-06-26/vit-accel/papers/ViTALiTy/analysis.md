---
source_type: paper
source_name: ViTALiTy
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/ViTALiTy/analysis.md -->

# ViTALiTy Detailed Paper Analysis

- title: ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention
- url: https://arxiv.org/abs/2211.05109
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/ViTALiTy.pdf`
- related_codebases: `ViTALiTy`

## 1. 기존 방법의 문제점
- 요약: ViT attention matrix는 token 수에 대해 quadratic cost를 가지며 sparse-only NLP accelerator를 그대로 적용하기 어렵다.

### Paper Evidence: Abstract
```text
Abstract—Vision Transformer (ViT) has emerged as a com-                         long-range interactions, showing superior accuracy against
                                        petitive alternative to convolutional neural networks for various                  CNNs [16]. Nevertheless, computing and storing such atten-
                                        computer vision applications. Specifically, ViTs’ multi-head at-                   tion matrices incurs a quadratic computational and memory
                                        tention layers make it possible to embed information globally
                                        across the overall image. Nevertheless, computing and storing                      cost dependency on the number of patches (input resolution).
                                        such attention matrices incurs a quadratic cost dependency                            To better understand the runtime breakdown for ViTs’
                                        on the number of patches, limiting its achievable efficiency                       MHA module, we profile DeiT-Tiny [38], a popular ViT
                                        and scalability and prohibiting more extensive real-world ViT                      model, on various commercial devices, such as NVIDIA RTX
                                        applications on resource-constrained devices. Sparse attention                     2080Ti [32], NVIDIA Edge GPU TX2 [31], and Google Pixel3
                                        has been shown to be a promising direction for improving
                                        hardware acceleration efficiency for NLP models. However, a                        phone [19]. In Fig. 1, we observe
```

## 2. 제안하는 방법
- 요약: first-order Taylor attention과 row-mean centering으로 low-rank component를 만들고 sparsity regularization을 결합한다.

### Paper Evidence: Method Section
```text
Algorithm 1, the Steps 1 and 3 for pre-processing the keys
                                                                                             and a systolic array for supporting the diverse operators in our
and values via column-wise accumulations and element-wise
                                                                                             Taylor attention. In particular, the pre/post-processors consist
additions, respectively, the post-processing in Steps 4 and 5 via
                                                                                             of an accumulator array, a divider array, and an adder array
element-wise additions, and the post-processing in Step 6 via
                                                                                             for performing column(token)-wise summation, element-wise
row-wise divisions. Note that although row-wise divisions are
                                                                                             divisions, and element-wise additions, respectively. Specifi-
also used in the softmax operation of the vanilla attention, our
                                                                                             cally, the accumulator array is to pre-process the keys and
Taylor attention reduces the number of divisions by (n/d)×
                                                                                             values for generating corresponding column summations via
(see Eq. (3)) as discussed above. Therefore, there exists an
                                                                                             accumulating all elements along the column/token dimension,
opportunity for the dedicated accelerator design to fully un-
                                                                                             i.e., computing the column summation of the keys 1Tn K (see
leash the hardware efficiency benefits of our proposed Taylor
                                                                                             Step 1 in Algorithm 1) and the column summation of both
attention’s property of “trades higher-cost multiplications and
                                                                                             the mean-centering keys k̂sum and values vsum (see Step
softmax operation with lower-cost pre/post-processing steps”.
                                                                                             3); The divider array is to process element-wise divisions in
  Opportunity 2: Da
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: softmax dot-product attention을 Taylor 근사로 linearize하고 low-rank/sparse 성분을 통합한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
arize the cost of attention blocks and further boost the accuracy by incorporat- Pixel3 13 58 29 ing a sparsity-based regularization. At the hardware level, we develop a dedicated accelerator to better leverage the resulting TX2 21 55 24 workload and pipeline from V I TAL I T Y’s linear Taylor attention which requires the execution of only the low-rank component, to further boost the hardware efficiency. Extensive experiments 2080Ti 25 52 23 and ablation studies validate that V I TAL I T Y offers boosted end- 0% 25% 50% 75% 100% to-end efficiency (e.g., 3× faster and 3× energy-efficient) under comparable accuracy, with respect to the state-of-the-art solution. Fig. 1. Runtime breakdown of DeiT-Tiny MHA on various devices. To alleviate the above quadratic complexity, a simple ap- I. I NTRODUCTION proach is to reduce the number of patches or input resolu- Vision Transformers (ViT) are gain
```
- keyword `sparsity`:
```text
ssing for accelerat- attention (Step 2) consistently dominates (52% − 58%) the ing ViT models. To close the above gap, we propose a first- MHA runtime, especially when devices become less powerful of-its-kind algorithm-hardware codesigned framework, dubbed and more resource-constrained. Hence, the major bottleneck V I TAL I T Y, for boosting the inference efficiency of ViTs. Unlike for ViTs is the softmax attention, which limits their achievable sparsity-based Transformer accelerators for NLP, V I TAL I T Y unifies both low-rank and sparse components of the attention efficiency and scalability, and prohibits extensive real-world in ViTs. At the algorithm level, we approximate the dot-product ViT applications on resource-constrained devices. softmax operation via first-order Taylor attention with row-mean Step 1: Query,Key, Value Step 2: Softmax Attention Map Step 3: Attention Score cente
```

## 4. 하드웨어 아키텍처
- 요약: linear Taylor attention workload에 맞춘 pipeline accelerator가 attention matrix materialization을 줄인다.

### Paper Evidence: Architecture / Implementation
```text
2019.                                                                                architecture,” in MICRO-54: 54th Annual IEEE/ACM International
[11] K. M. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane,                     Symposium on Microarchitecture, 2021, pp. 977–991.
     T. Sarlos, P. Hawkins, J. Q. Davis, A. Mohiuddin, L. Kaiser,                    [30] S. Mehta and M. Rastegari, “Mobilevit: light-weight, general-
     D. B. Belanger, L. J. Colwell, and A. Weller, “Rethinking                            purpose, and mobile-friendly vision transformer,” arXiv preprint
     attention with performers,” in International Conference on Learning                  arXiv:2110.02178, 2021.
     Representations, 2021. [Online]. Available: https://openreview.net/             [31] NVIDIA Inc., “NVIDIA Jetson TX2,” https://www.nvidia.com/en-us/
     forum?id=Ua6zuk0WRH                                                                  autonomous-machines/embedded-systems/jetson-tx2/, accessed 2020-
[12] G. M. Correia, V. Niculae, and A. F. Martins, “Adaptively sparse                     09-01.
     transformers,” arXiv preprint arXiv:1909.00015, 2019.                           [32] NVIDIA LLC., “GeForce RTX 2080 TI Graphics Card — NVIDIA,”
[13] B. Cui, Y. Li, M. Chen, and Z. Zhang, “Fine-tune bert with sparse self-              2021, https://www.nvidia.com/en-me/geforce/graphics-cards/rtx-2080-
     attention mechanism,” in Proceedings of the 2019 conference on empir-                ti/, accessed 2020-09-01.
     ical methods in natural language processing and the 9th international           [33] Z. Qu, L. Liu, F. Tu, Z. Chen, Y. Ding, and Y. Xie, “Dota: detect and omit
     joint conference on natural language processing (EMNLP-IJCNLP),                      weak attentions for scalable transformer acceleration,” in Proceedings
     2019, pp. 3548–3553.                                                                 of the 27th ACM International Conference on Architectural Support for
[14] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, “Imagenet:             Programming Languages and Operating Systems, 2022, pp. 14–26.
     A large-scale hierarchical image database,” in 2009 IEEE conference on          [34] A. Radford, K. Narasimhan, T. Salimans, and I. Sutskever, “Improving
     computer vision and pattern recognition. Ieee, 2009, pp. 248–255.                    language understanding by generative pre-training,” 2018.
[15] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training           [35] G. Shen, J. Zhao, Q.
```

### Hardware Keyword Evidence
- keyword `systolic`:
```text
cated ac- next step. WQ , WK , WV ∈ Rd×d are learned weights. celerator to better leverage the algorithmic properties Step 2: Compute the softmax attention map of V I TAL I T Y’s linear attention, where only a low-  QKT  rank component is executed during inference favoring S = softmax √ hardware efficiency. Specifically, V I TAL I T Y’s acceler- d ator features a chunk-based design integrating both a Step 3: Compute the attention score, Z = SV systolic array tailored for matrix multiplications and Before moving to the next layer, the attention scores are sent pre/post-processors customized for V I TAL I T Y atten- to the MLP module, O = ZWO , where, WO ∈ Rd×d . tions’ pre/post-processing steps. Furthermore, we adopt an intra-layer pipeline design to leverage the intra-layer B. Related Work data dependency for enhancing the overall throughput, Efficient Vision Transformers. Motivated by
```

## 5. 실험 방법
- 실험 section heading을 자동 추출하지 못했다. PDF 원문 확인 필요.

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
key challenge here is to accurately locate the weak (query, Fig. 3. Distribution of attentions (inputs to softmax) across various layers key) connections within the attention matrix for a given input. 0-11: (a) the vanilla attention distribution shifts left, and (b) row-wise mean- centering of attentions centers the distribution towards [-1,1). Here we use the In concurrent work for NLP-based Transformer accelerators DeiT-Tiny [38] model on the ImageNet dataset as an example. [22], [29], [33], [40], these weak connections are dynami- cally computed or predicted during runtime which are pruned resulting in irregular sparse attention patterns and complex efficiently compute the mean-centering rows of the attention designs for the corresponding accelerator. In contrast, for ViTs in linear time by directly modifying the key matrix, K. our novelty lies in decoupling the softmax attentions as
```
- keyword `DeiT`:
```text
i-head at- tion matrices incurs a quadratic computational and memory tention layers make it possible to embed information globally across the overall image. Nevertheless, computing and storing cost dependency on the number of patches (input resolution). such attention matrices incurs a quadratic cost dependency To better understand the runtime breakdown for ViTs’ on the number of patches, limiting its achievable efficiency MHA module, we profile DeiT-Tiny [38], a popular ViT and scalability and prohibiting more extensive real-world ViT model, on various commercial devices, such as NVIDIA RTX applications on resource-constrained devices. Sparse attention 2080Ti [32], NVIDIA Edge GPU TX2 [31], and Google Pixel3 has been shown to be a promising direction for improving hardware acceleration efficiency for NLP models. However, a phone [19]. In Fig. 1, we observe that computing the softmax sys
```
- keyword `EfficientViT`:
```text
H. Choi, S. J. Jung, and J. W. document transformer,” arXiv preprint arXiv:2004.05150, 2020. Lee, “Elsa: Hardware-software co-design for efficient, lightweight self- [3] I. Beltagy, M. E. Peters, and A. Cohan, “Longformer: The long- attention mechanism in neural networks,” in 2021 ACM/IEEE 48th document transformer,” ArXiv, vol. abs/2004.05150, 2020. Annual International Symposium on Computer Architecture (ISCA). [4] H. Cai, C. Gan, and S. Han, “Efficientvit: Enhanced linear attention for IEEE, 2021, pp. 692–705. high-resolution low-computation visual recognition,” 2022. [Online]. [23] B. Heo, S. Yun, D. Han, S. Chun, J. Choe, and S. J. Oh, “Rethinking spa- Available: https://arxiv.org/abs/2205.14756 tial dimensions of vision transformers,” in Proceedings of the IEEE/CVF [5] N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and International Conference on Computer Vision, 2021,
```
- keyword `dataset`:
```text
lenge here is to accurately locate the weak (query, Fig. 3. Distribution of attentions (inputs to softmax) across various layers key) connections within the attention matrix for a given input. 0-11: (a) the vanilla attention distribution shifts left, and (b) row-wise mean- centering of attentions centers the distribution towards [-1,1). Here we use the In concurrent work for NLP-based Transformer accelerators DeiT-Tiny [38] model on the ImageNet dataset as an example. [22], [29], [33], [40], these weak connections are dynami- cally computed or predicted during runtime which are pruned resulting in irregular sparse attention patterns and complex efficiently compute the mean-centering rows of the attention designs for the corresponding accelerator. In contrast, for ViTs in linear time by directly modifying the key matrix, K. our novelty lies in decoupling the softmax attentions as a combin
```

## 7. 실험 결과
- keyword `speedup`:
```text
ward accumulation dataflow for throughs of Transformers [15], [26], [34], [39] in NLP, there the systolic array to improve hardware efficiency. has been a growing interest in developing Transformers for 4) We perform extensive experiments and ablation studies vision tasks. ViT [16] was the first to show that Transform- to demonstrate the effectiveness of V I TAL I T Y in terms ers can completely replace convolutions by treating images of latency speedup (3×), and energy efficiency (3×) as a sequence of patches of fixed length. Since then, ViT 2 models and their variants have been successfully used for suffer from a degraded accuracy; and DOTA [33] adopts both image recognition [16], [38], object detection [5], [51], and low precision and low-rank linear transformation to predict segmentation [46]. To capture fine-grained spatial details of the sparse attention masks by jointly optimizing
```
- keyword `energy`:
```text
ation. At the hardware level, we develop a dedicated accelerator to better leverage the resulting TX2 21 55 24 workload and pipeline from V I TAL I T Y’s linear Taylor attention which requires the execution of only the low-rank component, to further boost the hardware efficiency. Extensive experiments 2080Ti 25 52 23 and ablation studies validate that V I TAL I T Y offers boosted end- 0% 25% 50% 75% 100% to-end efficiency (e.g., 3× faster and 3× energy-efficient) under comparable accuracy, with respect to the state-of-the-art solution. Fig. 1. Runtime breakdown of DeiT-Tiny MHA on various devices. To alleviate the above quadratic complexity, a simple ap- I. I NTRODUCTION proach is to reduce the number of patches or input resolu- Vision Transformers (ViT) are gaining increasing popularity tion. However, this would result in larger patch sizes with with their state-of-the-art performance i
```
- keyword `accuracy`:
```text
ng Shi*‡ , Chaojian Li‡ , Zhifan Ye† , Zhongfeng Wang§ and Yingyan Lin‡ † Rice University, Houston, TX Email: {jdass, sw99, zy50}@rice.edu ‡ Georgia Institute of Technology, Atlanta, GA arXiv:2211.05109v1 [cs.CV] 9 Nov 2022 Email: eiclab.gatech@gmail.com, {cli851, celine.lin}@gatech.edu § Nanjing University, Nanjing, Jiangsu Email: zfwang@nju.edu.cn Abstract—Vision Transformer (ViT) has emerged as a com- long-range interactions, showing superior accuracy against petitive alternative to convolutional neural networks for various CNNs [16]. Nevertheless, computing and storing such atten- computer vision applications. Specifically, ViTs’ multi-head at- tion matrices incurs a quadratic computational and memory tention layers make it possible to embed information globally across the overall image. Nevertheless, computing and storing cost dependency on the number of patches (input resolution).
```
- keyword `throughput`:
```text
ing both a Step 3: Compute the attention score, Z = SV systolic array tailored for matrix multiplications and Before moving to the next layer, the attention scores are sent pre/post-processors customized for V I TAL I T Y atten- to the MLP module, O = ZWO , where, WO ∈ Rd×d . tions’ pre/post-processing steps. Furthermore, we adopt an intra-layer pipeline design to leverage the intra-layer B. Related Work data dependency for enhancing the overall throughput, Efficient Vision Transformers. Motivated by the break- together with a down-forward accumulation dataflow for throughs of Transformers [15], [26], [34], [39] in NLP, there the systolic array to improve hardware efficiency. has been a growing interest in developing Transformers for 4) We perform extensive experiments and ablation studies vision tasks. ViT [16] was the first to show that Transform- to demonstrate the effectiveness of V
```
- keyword `latency`:
```text
module and a Multi-Layer Perceptron (MLP) module. proposed for NLP-based Transformer models [11], [24]. How- ever, there is a missed opportunity in applying linear attentions under comparable model accuracy with respect to the to ViTs. Unlike sparsity-based accelerators, we seek to exploit state-of-the-art solution. the low-rank property of the proposed linear attention to II. BACKGROUND AND M OTIVATION design dedicated accelerator for improved latency and energy efficiency. Our main contributions are summarized below: A. Preliminaries of Vision Transformers ViT Model Architecture. Fig. 2 illustrates the model 1) We propose an algorithm-accelerator codesign frame- architecture for ViTs. Here, each input image is divided and work, dubbed V I TAL I T Y, that unifies low-rank and arranged into a sequence of patches (or tokens), which are sparse approximation to boost the achievable accurac
```

## 8. 실험 옵션 / Ablation 축
- exact softmax vs approximated softmax
- Taylor/linear attention vs exact attention
- fixed sparse token pattern vs dense token pattern

## 9. HGTXR 적용 판단
- eye-region token 수가 고정/작은 편이면 approximation ablation으로 가치가 있다. exact attention fallback 필수.

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
