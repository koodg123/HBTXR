---
source_type: paper
source_name: ViTCoD
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/ViTCoD/analysis.md -->

# ViTCoD Detailed Paper Analysis

- title: ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design
- url: https://arxiv.org/abs/2210.09573
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/ViTCoD.pdf`
- related_codebases: `ViTCoD`

## 1. 기존 방법의 문제점
- 요약: ViT attention은 fixed token count 특성이 있는데 기존 NLP sparse accelerator는 dynamic sparsity에 맞춰져 비효율적이다.

### Paper Evidence: Abstract
```text
Abstract—Vision Transformers (ViTs) have achieved state-of-                     NLP Trans. (BigBird)              NLP Trans. (Routing)
                                                                                                                           NLP Trans. (Sf. k-means)          NLP Trans. (Longformer)
arXiv:2210.09573v3 [cs.LG] 3 Mar 2025




                                        the-art performance on various vision tasks. However, ViTs’ self-                  NLP Trans. (Reformer)             DeiT-Base (InfoPruning)
                                        attention module is still arguably a major bottleneck, limiting                    NLP Trans. (Sf. quant)            DeiT-Small (InfoPruning)
                                        their achievable hardware efficiency and more extensive appli-             38
                                                                                                                                                                               82
                                        cations to resource constrained platforms. Meanwhile, existing             36
                                                                                                                                                                               80
                                        accelerators dedicated to NLP Transformers are not optimal                 34




                                                                                                                                                                                        Accuracy (%)
                                        for ViTs. This is because there is a large difference between
```

## 2. 제안하는 방법
- 요약: attention map을 dense/sparse fixed pattern으로 polarize하고 autoencoder module로 data movement를 줄인다.

### Paper Evidence: Method Section
```text
algorithm- and accelerator-level innovations. To the best of           ViTCoD framework, leading to 235.3×, 142.9×, 86.0×,
   our knowledge, ViTCoD is the first co-design framework                 10.1×, and 6.8× speedups over both general computing
   dedicated to accelerating sparse ViTs’ inference, offering a           platforms CPUs, EdgeGPUs, GPUs, and prior-art Trans-
   new perspective on efficient ViT solutions.                            former accelerators SpAtten and Sanger, respectively, while
 • On the algorithm level, ViTCoD prunes and polarizes the                maintaining the model accuracy.



                                                                    2

                                                               TABLE I
                             A TAXONOMY FOR CLASSIFYING AND COMPARING REPRESENTATIVE SPARSE ACCELERATORS .
                           OuterSpace [27]          ExTensor [14]           SpArch [52]          Gamma [49]         SpAtten [42]         Sanger [24]       ViTCoD (Ours)
 Application Field          Tensor Algebra          Tensor Algebra         Tensor Algebra       Tensor Algebra     NLP Transformer     NLP Transformer           ViT
                                                                                                                   Sparse Attention:   Sparse Attention:   Sparse Attention:
 Workloads                    SpGEMM                  SpGEMM                  SpGEMM               SpGEMM
                                                                                                                   SDDMM; SpMM         SDDMM; SpMM         SDDMM; SpMM
                                                 Hybrid Outer-product        Condensed
                             Outer-product                                                     Gustavson(Row)-                                              K-stationary;
 Dataflow                                       & Inner-product (Input-     Outer-product                          Top-k Selection       S-stationary
                           (Input-stationary)                                                     stationary                                               Output-stationary
                                                 & Output-stationary)     (Input-stationary)
                                                                                                                      Dynamic &          Dynamic &
 Sparsity Pattern                Static                 Static                  Static               Static
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: attention map pruning/polarization으로 두 workload class를 만들고 encoder/decoder가 data movement와 compute를 trade한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
multaneously coordinate the aforementioned enforced denser alleviate the bottleneck complexity of self-attention modules, and sparser workloads for boosted hardware utilization, while integrating on-chip encoder and decoder engines to leverage sparse attention techniques have emerged as a promising ViTCoD’s algorithm pipeline for much reduced data movements. solution and been considered by both algorithm [4], [39], [48] Extensive experiments and ablation studies validate that ViTCoD and hardware acceleration [13], [24], [30], [42] works. largely reduces the dominant data movement costs, achieving Despite their great promise, existing sparse attention accel- speedups of up to 235.3×, 142.9×, 86.0×, 10.1×, and 6.8× erators or algorithm-accelerator co-design works (e.g., Sanger over general computing platforms CPUs, EdgeGPUs, GPUs, and prior-art Transformer accelerators SpAtten and Sanger u
```
- keyword `sparsity`:
```text
ViTs have a relatively fixed number of input tokens, 74 28 whose attention maps can be pruned by up to 90% even NLP Transformer Vision Transformer 72 with fixed sparse patterns, without severely hurting the model 26 (Dynamic) (Fixed) 70 accuracy (e.g., <=1.5% under 90% pruning ratio); while NLP 24 68 Transformers need to handle input sequences of varying numbers 22 10 30 50 70 90 100 of tokens and rely on on-the-fly predictions of dynamic sparse Sparsity Ratio of Attention Maps (%) attention patterns for each input to achieve a decent sparsity (e.g., >=50%). To this end, we propose a dedicated algorithm Fig. 1. Comparison between NLP Transformers and ViTs in terms of BLEU- and accelerator co-design framework dubbed ViTCoD for accel- sparsity or accuracy-sparsity trade-offs. Note that for NLP Transformer, we collect the results on machine translation task, IWSLT EN → DE, following erating
```
- keyword `parallelism`:
```text
ifically, we leverage (1) a then packed and split to be more regular and friendly supported split and conquer algorithm to prune the attention maps by up by a reconfigurable architecture. DOTA [30] considers both to 90% sparsity and to simultaneously polarize the attention low precision and low rank linear transformation to predict the maps to be either denser or sparser for enhancing more regular sparse attention masks, and explores token-level parallelism workloads; and (2) an auto-encoder module to compress the and out-of-order execution for locality-aware computing. All corresponding vectors for calculating attentions to a much above works focus on NLP Transformers, and thus require more compact representation, without hurting the model accu- dynamic and input-dependent sparse masks prediction. All ac- racy. On top of that, we further design a dedicated two-pronged celerators above t
```

## 4. 하드웨어 아키텍처
- 요약: dense/sparse workload와 encoder/decoder engine을 동시에 coordinate하는 dedicated accelerator를 사용한다.

### Paper Evidence: Architecture / Implementation
```text
architectures for both natural language processing (NLP) and            challenges for efficient acceleration of ViTs: First, ViTs have
                                        computer vision (CV) tasks. The powerful performance of                 a relatively fixed number of input tokens during both training
                                        Transformers largely benefits from their self-attention module          and inference (e.g., a commonly adopted token size of 16×16
                                        that is capable of extracting global context information [10],          for an image resolution of 224 × 224, which leads to a total of
                                        [40], [46]. However, the self-attention module comes at a               196 tokens), while NLP Transformers adopt input-dependant
                                        cost of inefficiency during both training and inference due             varying numbers of tokens across different NLP datasets/tasks.
                                        to its quadratic complexity dependency on the number of                 ViTs’ relatively fixed number of tokens offers an opportunity
                                        input tokens, and has been recognized as a major efficiency             to design ViT accelerators, which can potentially avoid on-the-



                                                                                                            1

                                                                              Performance            Dense ViTs         Sparse ViTs
                                                                                 (GOPS)
                                                                                                     ViTCoD (Denser/Sparser + Auto-encoder)

                                                                                                                    ViTCoD Comp. Roof
                                                                               256

                                                                               100                    I/O oof
                                                                                                 C oD th R
                                                                                               T     d
                                                                                            Vi Wi
                                                                                              a
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
, without hurting the model accu- dynamic and input-dependent sparse masks prediction. All ac- racy. On top of that, we further design a dedicated two-pronged celerators above targeting NLP Transformers require dynamic accelerator integrating the encoder and decoder engines to and input-dependent sparse masks prediction. For accelerating (1) cooperatively handle the denser or sparse workloads and ViTs, VAQF [36] designs inference accelerators on FPGAs for (2) leverage the auto-encoder module of ViTCoD’s algorithm quantized ViTs with binary weights and low precision activa- pipeline for much reduced data movements. Next, we will tions; In contrast, ViTCoD is the first algorithm and accelerator introduce the ViTCoD algorithm and accelerator in detail. co-design framework dedicated to sparse ViTs, aiming to fully IV. P ROPOSED V I TC O D A LGORITHM exploit ViTs’ fixed sparse patterns and in
```

## 5. 실험 방법
```text
experiments on DeiT-Base/Small/Tiny models. Compared with                                                      the loaded Q/K vectors, reducing data movements at the cost
reordering only, pruning makes sparse parts sparser, and thus                                                  of large computation workloads; (2) ViTCoD’s data move-
enhances the polarization effect (i.e., more regular), offer-                                                  ments are largely reduced from 50% to 28% after adopting AE
ing on-average 5.14× speedups across 60%/70%/80%/90%                                                           modules, indicating the effectiveness of ViTCoD in alleviating
pruning ratio (e.g., 8.14× speedups under 90% sparsity).                                                       the performance bottleneck.
Compared with pruning only, reordering makes the sparse
pattern polarized and more regular, offering on-average 2.59×                                                                          VII. C ONCLUSIONS
speedups across 60%/70%/80%/90% pruning ratios (e.g.,
2.03× speedups under 90% sparsity).                                                                              We present ViTCoD, the first algorithm and accelerator co-
                                                                                                               design framework for sparse ViTs. On the algorithm level,
D. Evaluation of the ViTCoD Accelerator                                                                        ViTCoD integrates (1) a split and conquer algorithm to prune
                                                                                                               and polarize the attention maps to be either denser or sparser
   We first benchmark ViTCoD against five baselines. In Fig.                                                   with fixed masks and (2) an auto-encoder module to trade
19, ViTCoD consistently achieves both improved normalized                                                      costly data movements for cheaper computations, without
efficiency and latency speedups over all baselines. Specifically,                                              compromising the model accuracy. On the hardware level,
ViTCoD offers on-average (among six DeiT & LeViT models                                                        ViTCoD incorporates (1) a dedicated two-pronged accelerator
mentioned in our experiment setting) 235.3×, 142.9×, 86.0×,                                                    to process each of the aforementioned denser or sparser
10.1×, and 6.8× attention speedups compared with CPU,                                                          workloads and (2) encoder and decoder engines to cooperate
EdgeGPU, GPU, SpAtten, and Sanger baselines when eval-                                                         with auto-encoder modules, boosting the overall utilization
uating on attention layers of 90% sparsity. For
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
elerator co-design framework dubbed ViTCoD for accel- sparsity or accuracy-sparsity trade-offs. Note that for NLP Transformer, we collect the results on machine translation task, IWSLT EN → DE, following erating ViTs. Specifically, on the algorithm level, ViTCoD prunes [39]; For ViTs, we apply an info-based pruning technique on DeiT-Base/Small and polarizes the attention maps to have either denser or sparser models and classification task (e.g., ImageNet), following [19]. fixed patterns for regularizing two levels of workloads without hurting the accuracy, largely reducing the attention computations bottleneck for the inference acceleration of Transformers. For while leaving room for alleviating the remaining dominant data movements; on top of that, we further integrate a lightweight example, the self-attention module of the GPT-2 model [31] and learnable auto-encoder module to enable tr
```
- keyword `DeiT`:
```text
e University, Houston, TX ‡ Oracle Health and AI, Redwood, CA {hyou37, eiclab, zyu401, yzhang919, cli851, celine.lin}@gatech.edu, {zs19, zy34}@rice.edu, baopu.li@oracle.com Abstract—Vision Transformers (ViTs) have achieved state-of- NLP Trans. (BigBird) NLP Trans. (Routing) NLP Trans. (Sf. k-means) NLP Trans. (Longformer) arXiv:2210.09573v3 [cs.LG] 3 Mar 2025 the-art performance on various vision tasks. However, ViTs’ self- NLP Trans. (Reformer) DeiT-Base (InfoPruning) attention module is still arguably a major bottleneck, limiting NLP Trans. (Sf. quant) DeiT-Small (InfoPruning) their achievable hardware efficiency and more extensive appli- 38 82 cations to resource constrained platforms. Meanwhile, existing 36 80 accelerators dedicated to NLP Transformers are not optimal 34 Accuracy (%) for ViTs. This is because there is a large difference between 78 32 76 BLEU ViTs and Transformers for
```
- keyword `dataset`:
```text
nefits from their self-attention module and inference (e.g., a commonly adopted token size of 16×16 that is capable of extracting global context information [10], for an image resolution of 224 × 224, which leads to a total of [40], [46]. However, the self-attention module comes at a 196 tokens), while NLP Transformers adopt input-dependant cost of inefficiency during both training and inference due varying numbers of tokens across different NLP datasets/tasks. to its quadratic complexity dependency on the number of ViTs’ relatively fixed number of tokens offers an opportunity input tokens, and has been recognized as a major efficiency to design ViT accelerators, which can potentially avoid on-the- 1 Performance Dense ViTs Sparse ViTs (GOPS) ViTCoD (Denser/Sparser + Auto-encoder) ViTCoD Comp. Roof 256 100 I/O oof C oD th R T d Vi Wi a nd 10 B Fig. 2. Illustrating the fixed sparse attenti
```

## 7. 실험 결과
- keyword `speedup`:
```text
decoder engines to leverage sparse attention techniques have emerged as a promising ViTCoD’s algorithm pipeline for much reduced data movements. solution and been considered by both algorithm [4], [39], [48] Extensive experiments and ablation studies validate that ViTCoD and hardware acceleration [13], [24], [30], [42] works. largely reduces the dominant data movement costs, achieving Despite their great promise, existing sparse attention accel- speedups of up to 235.3×, 142.9×, 86.0×, 10.1×, and 6.8× erators or algorithm-accelerator co-design works (e.g., Sanger over general computing platforms CPUs, EdgeGPUs, GPUs, and prior-art Transformer accelerators SpAtten and Sanger under an [24]) focus on accelerating NLP Transformers, and adopt hard- attention sparsity of 90%, respectively. Our code implementation ware designs with on-the-fly sparse attention prediction and is available at http
```
- keyword `energy`:
```text
for generating Q/K/V so as to compress benchmarking with GPUs w/ larger batch size, we scale up Q/K before transferring them back to the off-chip memory. the accelerators’ hardware resource to have a comparable peak Also, their computation can be fully pipelined to hide the throughput for a fair comparison following [30]. Metrics: We processing time of the encoder engine; The decoder engine evaluate all platforms in terms of latency speedups and energy is then needed before loading Q/K into the PE arrays, which efficiency. In addition, we compared the achieved attention will be pipelined with the data movements instead. sparsity and model accuracy for all ViT models. 3) Reconfigurability: To support the potential need of task Hardware Platform Setup. Characteristics: ViTCoD is change after deployment, e.g., ViT models with different mask designed with a total area of 3 mm2 , a DDR4-2400
```
- keyword `accuracy`:
```text
9573v3 [cs.LG] 3 Mar 2025 the-art performance on various vision tasks. However, ViTs’ self- NLP Trans. (Reformer) DeiT-Base (InfoPruning) attention module is still arguably a major bottleneck, limiting NLP Trans. (Sf. quant) DeiT-Small (InfoPruning) their achievable hardware efficiency and more extensive appli- 38 82 cations to resource constrained platforms. Meanwhile, existing 36 80 accelerators dedicated to NLP Transformers are not optimal 34 Accuracy (%) for ViTs. This is because there is a large difference between 78 32 76 BLEU ViTs and Transformers for natural language processing (NLP) 30 tasks: ViTs have a relatively fixed number of input tokens, 74 28 whose attention maps can be pruned by up to 90% even NLP Transformer Vision Transformer 72 with fixed sparse patterns, without severely hurting the model 26 (Dynamic) (Fixed) 70 accuracy (e.g., <=1.5% under 90% pruning ratio); while
```
- keyword `throughput`:
```text
U (Nvidia 2080Ti), and two attention In particular, the encoder engine is enabled right after the accelerators: SpAtten [42] and Sanger [24]. Note that when linear projection for generating Q/K/V so as to compress benchmarking with GPUs w/ larger batch size, we scale up Q/K before transferring them back to the off-chip memory. the accelerators’ hardware resource to have a comparable peak Also, their computation can be fully pipelined to hide the throughput for a fair comparison following [30]. Metrics: We processing time of the encoder engine; The decoder engine evaluate all platforms in terms of latency speedups and energy is then needed before loading Q/K into the PE arrays, which efficiency. In addition, we compared the achieved attention will be pipelined with the data movements instead. sparsity and model accuracy for all ViT models. 3) Reconfigurability: To support the potential ne
```
- keyword `GOPS`:
```text
, while NLP Transformers adopt input-dependant cost of inefficiency during both training and inference due varying numbers of tokens across different NLP datasets/tasks. to its quadratic complexity dependency on the number of ViTs’ relatively fixed number of tokens offers an opportunity input tokens, and has been recognized as a major efficiency to design ViT accelerators, which can potentially avoid on-the- 1 Performance Dense ViTs Sparse ViTs (GOPS) ViTCoD (Denser/Sparser + Auto-encoder) ViTCoD Comp. Roof 256 100 I/O oof C oD th R T d Vi Wi a nd 10 B Fig. 2. Illustrating the fixed sparse attention mask. fly sparse attention pattern prediction adapting to each input, 0.6 3.9 0 0.1 1.0 10 Comp. Intensity via co-designing with sparse ViT algorithms. Second, as shown Computation to Communication Ratio (Ops/Byte) in Fig. 1, ViTs allow their attention maps to be pruned by up to 90%∼95% with
```

## 8. 실험 옵션 / Ablation 축
- exact softmax vs approximated softmax
- Taylor/linear attention vs exact attention
- fixed sparse token pattern vs dense token pattern

## 9. HGTXR 적용 판단
- 고정 eye token pattern pruning 실험에 유용하지만 정확도 리스크가 높다. P1/P2 ablation으로 제한.

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
