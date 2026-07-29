---
source_type: paper
source_name: LUT-GEMM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/LUT-GEMM/analysis.md -->

# LUT-GEMM Detailed Paper Analysis

- title: LUT-GEMM: Quantized Matrix Multiplication based on LUTs for Efficient Inference in Large-Scale Generative Language Models
- url: https://arxiv.org/abs/2206.09557
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/LUT-GEMM.pdf`
- related_codebases: `lut-gemm`

## 1. 기존 방법의 문제점
- 요약: weight-only quantization은 memory movement를 줄이지만 dequantization cost가 남는다.

## 2. 제안하는 방법
- 요약: LUT 기반 quantized GEMM으로 dequantization을 제거하고 계산량을 줄인다.

### Paper Evidence: Method Section
```text
approach enhances generation latency by reducing memory movement despite the dequantization
overheads.

2.3   B INARY-C ODING Q UANTIZATION

Binary-coding quantization (BCQ) initially introduced by Xu et al. (2018), presents a compelling
                                                                    Pq when a weight vector w (of
alternative to conventional uniform quantization methods. For instance,
size n) is quantized into q-bit by BCQ, w is approximated to be i=1 αi bi where αi ∈ R+ is a
scaling factor and bi ∈ {−1, +1}n is a binary vector. In this paper, we have chosen to employ BCQ
as our primary quantization technique for weights while retaining activations in full precision to
address the challenges mentioned earlier. Moreover, we introduce an extension to BCQ, enhancing
its capabilities to encompass both uniform quantization and group-wise quantization. This extension,
outlined in the subsequent section, not only broadens BCQ’s utility but also enables the applicability
of LUT-GEMM to various quantization methods.

3     D ESIGN M ETHODOLOGY OF LUT-GEMM
LUT-GEMM is devised to develop high-performance, energy-efficient inference systems for LLMs.
To achieve this objective, LUT-GEMM incorporates several innovative approaches. Firstly, we em-
ploy a lookup table (LUT) based computation technique to mitigate redundant calculations caused
by digitized binary weights after BCQ. Furthermore, since most non-uniform quantization meth-
ods involve complex operations with limited parallelism and often lack hardware support, we de-
sign LUT-GEMM to efficiently support BCQ formats. Our proposed kernel, LUT-GEMM, directly
utilizes BCQ formats without additional overhead, such as dequantization. Secondly, we expand
conventional BCQ methods by introducing a bias term, significantly enhancing representational ca-
pability. This simple yet profound enhancement enables the representation of both non-uniform and
uniform quantization methods within the extended BCQ format, providing us with the flexibility to
leverage various quantization techniques based on the specific requirements of LLMs. Finally, We
further refine the implementation details of the binary-coding quantization scheme, enabling a trade-
off between compression ratio and quantization error to better exploit the characteristics of LLMs.
As a result, LUT-GEMM demonstrates reduced latency and/or a decreased number of GPUs required
for LLM inference while inherently accommodating various weight-only quantization methods.

3.1   LUT BASED Q UANTIZED M ATRIX M ULTIPLICATION

Our quantization scheme, w
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: group-wise quantization과 lookup table product accumulation으로 low-bit GEMM을 처리한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `sparsity`:
```text
rs), pp. 4171–4186, 2019. Ali Edalati, Marzieh Tahaei, Ahmad Rashid, Vahid Partovi Nia, James J. Clark, and Mehdi Reza- gholizadeh. Kronecker decomposition for gpt compression, 2021. Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Optq: Accurate quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations, 2022. Trevor Gale, Erich Elsen, and Sara Hooker. The state of sparsity in deep neural networks. arXiv preprint arXiv:1902.09574, 2019. Song Han, Huizi Mao, and William J. Dally. Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding. In International Conference on Learning Representations (ICLR), 2016. Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. In NIPS Deep Learning and Representation Learning Workshop, 2015.
```
- keyword `parallelism`:
```text
INT8) O (FP16) O (FP16) (a) W8/A8 Quant. (b) W4/A16 Quant. (c) W4/A16 Quant. (On-the-fly Dequant.) (LUT-GEMM, Ours) Figure 1: Three matrix multiplication schemes for W8/A8 or W4/A16 (i.e., weight-only) quanti- zation format. Our proposed LUT-GEMM can adopt W4 and A16 without requiring an additional dequantization step. to enhance memory bandwidth (Migacz, 2017; Yu et al., 2017). To address such a concern, re- searchers have proposed to use model parallelism, which distributes computations over multiple GPUs through GPU-to-GPU communication (Shoeybi et al., 2019; Narayanan et al., 2021). Never- theless, it is worth noting that model parallelism introduces additional overheads, stemming from the inter-GPU communication. Consequently, the performance gains achieved through model paral- lelism exhibit a sub-linear relationship with the number of GPUs employed. To mitigate the challenges rela
```
- keyword `batch size`:
```text
Note that due to the limited memory capacity of a single GPU, large LMs may need multiple GPUs, resulting in increased communication latency. GPUs are commonly adopted to accelerate inference as GPUs embed lots of arithmetic units and support multiple threads, critical for speeding up matrix multipli- cations (Narayanan et al., 2021; Migacz, 2017). However, extracting high performance from GPUs depends on arithmetic intensity, and therefore, the batch size should be large enough to ensure a high reuse ratio from main memory (Markidis et al., 2018). 2.2 Q UANTIZATION M ETHODS AND L IMITATIONS Various research efforts have been made to enhance the serviceability of large and heavy deep neural networks by improving their latency and throughput. These efforts include quantiza- tion (Rastegari et al., 2016; Jacob et al., 2018; Nagel et al., 2017; Xu et al., 2018; Chung et al., 2020), pruning
```

## 4. 하드웨어 아키텍처
- 요약: GPU kernel 중심이나 LUT computation idea가 FPGA에도 이전 가능하다.

### Paper Evidence: Architecture / Implementation
```text
implementations with AWQ quantization method and model parallelism on multiple GPUs.
                                          Quant.       Perplexity           Latency (ms)
           Model         Kernel-q-g
                                          method         Wiki2      1-GPU      2-GPU 4-GPU
                      cuBLAS-16-N/A        FP16           4.10       43.6       25.1     16.9
          LLaMA
                     LUT-GEMM-4-128       AWQ             4.23       20.7       15.5     12.2
           30B
                     LUT-GEMM-3-128       AWQ             4.88       18.1       13.7     11.2
                      cuBLAS-16-N/A        FP16           3.53      OOM         46.9     29.1
          LLaMA
                     LUT-GEMM-4-128       AWQ             3.67       35.7       25.4     19.9
           65B
                     LUT-GEMM-3-128       AWQ             4.24       31.3       23.0     18.3



5   ACCELERATING Q UANTIZED OPT-175B

Table 4 provides a comparison of the end-to-end latency for generating a token in OPT-175B, a rep-
resentative large-scale LM, using the FasterTransformer framework. LUT-GEMM demonstrates its
ability to decrease the number of GPUs needed for running inference, while concurrently reducing
latency as q diminishes or as the number of GPUs rises. For OPT-175B with FP16, a minimum of
8 GPUs is necessary for executing inference. However, upon utilizing the BCQ format for quan-
tization, LUT-GEMM is able to perform inference using just a single GPU, while maintaining a
comparable overall latency. It should be noted that, when comparing identical 3-bit (weight-only
and row-wise) quantization scenarios, the latency for token generation using LUT-GEMM is 2.1×
lower than that of the OPTQ library. This significant reduction in latency can be primarily attributed
to LUT-GEMM’s ability to directly accept quantized weights, thereby eliminating the need for de-
quantization.


Table 4: End-to-end latency per token for OPT-175B model. The latency is measured on A100
80GB.
                                            Latency per token (ms)
                      GPUs     Baseline   OPTQ             LUT-GEMM
                                FP16       3-bit   1-bit 2-bit 3-bit         4-bit
                        1       OOM       106.5    30.4 40.1 51.6            OOM
                        2       OOM        N/A     25.2 30.1 35.8            41.2
                        4       OOM        N/A     20.3 23.8 27.2            30.1
                        8       42.4       N/A     20.1 22.4 24.2            25.8

Let us demonst
```

## 5. 실험 방법
```text
experiments on three pre-trained OPT models Zhang et al. (2022), which are publicly available2 .
Specifically, we apply post-training quantization (with an iterative solver introduced in Xu et al.
    2
        https://huggingface.co/facebook/opt-30b


                                                           16

Published as a conference paper at ICLR 2024




(2018)) to pre-trained OPT models while g and q vary. Then, each quantized model is evaluated
on the LAMBADA Paperno et al. (2016) dataset to find the relationship between compression ratio
and accuracy. Figure 7 shows accuracy and compression ratio3 when we try various q values and
g values. From Figure 7, we observe that compared to the conventional row-wise quantization,
group-wise BCQ offers new optimal configurations. Thus, to achieve the best compression ratio (or
minimum accuracy degradation), it is necessary to explore different q and g values simultaneously
for a given target. Note that for OPT-13B and OPT-30B, as we discussed the limits of row-wise
quantization for large-scale LMs, a small g value is critical to achieving low accuracy degradation
(while latency is not heavily affected by a small g). All in all, the effects of q and g on accuracy
differ with each model such that q and g are hyper-parameters to be optimized.

                                          g = 32    g = 64      g = 128        g = 256   g = 512   g = row-wise
                                   75
                                          30B, FP16
                                   70                 13B, FP16
                    Accuracy (%)




                                   65                                             6.7B, FP16

                                   60

                                   55         6.7B, q = 4
                                              13B, q = 3
                                              30B, q = 3
                                   50
                                        2.0     2.5       3.0        3.5          4.0      4.5     5.0      5.5
                                                                 Compression ratio
Figure 7: Accuracy and compression ratio with the various combinations of quantization bits (q)
and group size (g). Three pre-trained OPT models are quantized (by post-training BCQ method)
and then evaluated on the LAMBADA dataset.


F        A DDITIONAL E XPERIMENTAL R ESULTS
This section contains additional experimental results.


     Table 6: Comparison of perplexity and end-to-end latency per token for OPT family models.
                                                   Quant.                   Perplexity                     Latency (ms)
    Model           Kernel-q-g
                                                   method    Wiki2         PTB       LAMBADA       1-GPU      2-GPU 4-GPU
                cuBLAS-16-N/A                       FP16      9.56         11.84        15.84       40.5       23.5     14.7
                LU
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
s, James Bradbury, Matthew Johnson, Blake Hechtman, Laura Weidinger, Iason Gabriel, William Isaac, Ed Lockhart, Simon Osindero, Laura Rimell, Chris Dyer, Oriol Vinyals, Kareem Ayoub, Jeff Stanway, Lorrayne Bennett, Demis Hassabis, Koray Kavukcuoglu, and Geoffrey Irving. Scaling language models: Methods, analysis & insights from training gopher. arXiv:2112.11446, 2021. Mohammad Rastegari, Vicente Ordonez, Joseph Redmon, and Ali Farhadi. XNOR-Net: Imagenet classification using binary convolutional neural networks. In ECCV, 2016. Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model par- allelism. arXiv preprint arXiv:1909.08053, 2019. Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Ze
```
- keyword `LLaMA`:
```text
et al., 2022) have reported that LLM performance follows a predictable power-law scaling as a function of model size. Accord- ingly, in recent years, several large-scale generative language models, including GPT-3 (175B) (Brown et al., 2020), HyperCLOVA (204B) (Kim et al., 2021a), Gopher (280B) (Rae et al., 2021), Chinchilla (70B) (Hoffmann et al., 2022), Megatron Turing NLG (530B) (Smith et al., 2022), PaLM (540B) (Chowdhery et al., 2022), and LLaMA (65B) (Touvron et al., 2023), have been proposed to further advance state-of-the-art performance. However, models with billions of parameters cannot be accommodated on a single GPU due to the limited memory size of GPUs, which is sacrificed ∗ Equal contribution 1 Published as a conference paper at ICLR 2024 W (INT8) A (INT8) W ( 4bit) A (FP16) W ( 4bit) A (FP16) W ( 4bit) A (FP16) Matmul (INT8) Dequant Pre- computation O (INT32) W (FP16)
```
- keyword `dataset`:
```text
at https://github.com/naver-aics/lut-gemm 1 I NTRODUCTION Recent years have observed large-scale language models (LLMs) presenting state-of-the-art perfor- mance on various natural language process (NLP) tasks. Such rapid progress in NLP performance has been highly facilitated by the self-supervised learning methods, avoiding expensive manual la- beling (Devlin et al., 2019; Baevski et al., 2020; Chen et al., 2020). Leveraging extensive training datasets, these models benefit from efficient sequence-to-sequence architectures like the Transformer model (Vaswani et al., 2017), leading to notable increases in model parameters. Previous studies (Brown et al., 2020; Kaplan et al., 2020; Hoffmann et al., 2022) have reported that LLM performance follows a predictable power-law scaling as a function of model size. Accord- ingly, in recent years, several large-scale generative language models, in
```

## 7. 실험 결과
- keyword `speedup`:
```text
ment in performance cannot be achieved when dequantization is included. 6 S UMMARY AND L IMITATIONS In this paper, we introduce LUT-GEMM, a highly efficient matrix multiplication kernel designed to operate directly on quantized weights, thereby eliminating the need for an additional dequantization step. Leveraging an extended BCQ format, LUT-GEMM exhibits the capability to process both uniformly and non-uniformly quantized models, achieving 2.1× speedup over GPTQ. However, 9 Published as a conference paper at ICLR 2024 Table 5: Perplexity of quantized OPT-175B using OPTQ and end-to-end latency per token by LUT- GEMM for various q and g configurations. g of ‘-’ indicates row-wise quantization. Model size(GB) Kernel q g PPL* Latency(ms) (Comp. Ratio) cuBLAS 16 N/A 8.34 347.9 (×1.00) 42.4 (8-GPU) 4 - 8.37 87.0 (×4.00) OOM (1-GPU) 3 - 8.68 65.3 (×5.33) 51.6 (1-GPU) LUT-GEMM 2 32 8.94 54.4 (×
```
- keyword `energy`:
```text
in full precision to address the challenges mentioned earlier. Moreover, we introduce an extension to BCQ, enhancing its capabilities to encompass both uniform quantization and group-wise quantization. This extension, outlined in the subsequent section, not only broadens BCQ’s utility but also enables the applicability of LUT-GEMM to various quantization methods. 3 D ESIGN M ETHODOLOGY OF LUT-GEMM LUT-GEMM is devised to develop high-performance, energy-efficient inference systems for LLMs. To achieve this objective, LUT-GEMM incorporates several innovative approaches. Firstly, we em- ploy a lookup table (LUT) based computation technique to mitigate redundant calculations caused by digitized binary weights after BCQ. Furthermore, since most non-uniform quantization meth- ods involve complex operations with limited parallelism and often lack hardware support, we de- sign LUT-GEMM to effici
```
- keyword `accuracy`:
```text
ntensive dequantization process rather than actual computational reduction. In this paper, we introduce LUT-GEMM, an efficient kernel for quantized matrix multiplication, which not only eliminates the resource-intensive dequantization process but also reduces computational costs compared to previous kernels for weight-only quantization. Furthermore, we pro- posed group-wise quantization to offer a flexible trade-off between compression ratio and accuracy. The impact of LUT-GEMM is facilitated by implementing high compression ratios through low-bit quantization and efficient LUT-based op- erations. We show experimentally that when applied to the OPT-175B model with 3-bit quantization, LUT-GEMM substantially accelerates token generation latency, achieving a remarkable 2.1× improvement on a single GPU when com- pared to OPTQ, which relies on the costly dequantization process. The code is av
```
- keyword `throughput`:
```text
ix multipli- cations (Narayanan et al., 2021; Migacz, 2017). However, extracting high performance from GPUs depends on arithmetic intensity, and therefore, the batch size should be large enough to ensure a high reuse ratio from main memory (Markidis et al., 2018). 2.2 Q UANTIZATION M ETHODS AND L IMITATIONS Various research efforts have been made to enhance the serviceability of large and heavy deep neural networks by improving their latency and throughput. These efforts include quantiza- tion (Rastegari et al., 2016; Jacob et al., 2018; Nagel et al., 2017; Xu et al., 2018; Chung et al., 2020), pruning (Han et al., 2016; Zhu & Gupta, 2017; Gale et al., 2019), knowledge distillation (Hinton et al., 2015; Polino et al., 2018), and low-rank approximation (N. Sainath et al., 2013; Chen et al., 2018; Edalati et al., 2021). Among these, quantization is the most extensively re- searched field,
```
- keyword `latency`:
```text
vious kernels for weight-only quantization. Furthermore, we pro- posed group-wise quantization to offer a flexible trade-off between compression ratio and accuracy. The impact of LUT-GEMM is facilitated by implementing high compression ratios through low-bit quantization and efficient LUT-based op- erations. We show experimentally that when applied to the OPT-175B model with 3-bit quantization, LUT-GEMM substantially accelerates token generation latency, achieving a remarkable 2.1× improvement on a single GPU when com- pared to OPTQ, which relies on the costly dequantization process. The code is available at https://github.com/naver-aics/lut-gemm 1 I NTRODUCTION Recent years have observed large-scale language models (LLMs) presenting state-of-the-art perfor- mance on various natural language process (NLP) tasks. Such rapid progress in NLP performance has been highly facilitated by the se
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization

## 9. HGTXR 적용 판단
- 현재 resource 정책과 충돌하므로 P3 negative-control로만 둔다.

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
