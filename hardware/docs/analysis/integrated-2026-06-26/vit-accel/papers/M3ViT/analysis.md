---
source_type: paper
source_name: M3ViT
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/M3ViT/analysis.md -->

# M3ViT Detailed Paper Analysis

- title: M3ViT: Mixture-of-Experts Vision Transformer for Efficient Multi-task Learning with Model-Accelerator Co-design
- url: https://arxiv.org/abs/2210.14793
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/M3ViT.pdf`
- related_codebases: `M3ViT`

## 1. 기존 방법의 문제점
- 요약: multi-task learning은 training gradient conflict와 inference 시 전체 모델 활성화 비용을 동시에 가진다.

### Paper Evidence: Introduction / Motivation
```text
1       Introduction
                                         Vision Transformers (ViTs) [3, 4, 5, 6], as the latest performant deep models, have achieved impressive
                                         performance on various computer vision tasks [7, 8, 9]. These models are specially trained or tested
                                             ∗
                                                 Equal contribution


                                         36th Conference on Neural Information Processing Systems (NeurIPS 2022).

for only one or a few tasks; however, many real-world applications require one compact system
that can handle many different tasks efficiently, and often need to swiftly switch between tasks per
demand. For example, an autonomous driving system [10] needs to perform and switch between many
tasks such as drivable area estimation, lane detection, pedestrian detection, and scene classification:
apparently both single task inference and cross-task switching need to happen at ultra-low latency.
As another example, smart-home indoor robots [11] are expected to address semantic segmentation,
navigation, tracking, or other tasks in varying contexts, with very limited on-board resources. Multi-
task learning (MTL) [12, 13, 14] solves multiple tasks simultaneously within a single model and learns
improved feature representations [15] shared by related tasks [16, 17]. Therefore, accomplishing
realistic efficient MTL is becoming a key knob for building real-time sophisticated AI systems.
Despite the promise, challenges persist to build an efficient MTL model suitable for real-world
applications: Ê during training, prior works [18, 19, 20] indicate the competition of different tasks
in training may degrade MTL, since the same weights might receive and be confused by conflicting
update directions. Specifically, [19] reveals that negative cosine similarities between different tasks’
gradients are detrimental. [21, 22] confirm that conflicting gradients not only slow down convergence
but also bias the learned representations against some tasks. That is only getting worse on compact
models owing to their limi
```

## 2. 제안하는 방법
- 요약: ViT backbone에 task-specific MoE layer를 넣어 task별 sparse expert pathway만 활성화한다.

### Paper Evidence: Method Section
```text
methods are lossy and require compression-aware training to regain accuracy. To our best knowledge,
there is no existing FPGA accelerator for MoE in a Transformer-based model. The MoE mechanism
exposes great challenges to FPGA since it requires swift expert switching between tokens and frames,
which may introduce significant overhead of memory and parameter loading. In this work, however,
we propose a novel expert-by-expert computation-reordering approach that can reduce the overhead
to negligible despite the number of experts, and does not require model compression or re-training.

3     Method
Overview We first describe the standard Vision Transformer and MoEs, and then show the proposed
MoE ViT design for MTL. To enable dynamically adapting between different tasks with minimum
overhead on FPGA, we detail the hardware implementation. Figure 1 shows the whole framework.

                                 (a) MoE ViT Design                                     (b) Hardware Design
                            Task A    Task B     Both                  NotActivated               Decoder A              Decoder B
                                                                                    Intermediate input   Initial input

                                                                                                 Self-Attention
                                                                                                 Gating Function

                                            …                Experts
                                                                                 Load
                                                                              Parameters

                             token                                             Compute
                                     …                …
                           embedding
                   Layer




                                                                                Expert
                                                                           Intermediate
                                                                           output                        Time
                                         Linear Projection                        Final output




Figure 1: The overall structure of the proposed M3 ViT pipeline. The input image is split into
fixed-size patches, embedded, and combined with position embeddings. In training, the MTL MoE
ViT adaptively activates the model by sparsely selecting relevant experts using its task-dependent
rout
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: task gate가 expert path를 선택하고 hardware reordering이 task switching overhead를 줄인다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `sparsity`:
```text
xture of experts. arXiv preprint arXiv:1312.4314, 2013. [60] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. Gshard: Scaling giant models with conditional computation and automatic sharding. arXiv preprint arXiv:2006.16668, 2020. [61] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. arXiv preprint arXiv:2101.03961, 2021. [62] Jakob Vogdrup Hansen. Combining predictors: comparison of five meta machine learning methods. Information Sciences, 119(1-2):91–105, 1999. [63] Mike Lewis, Shruti Bhosale, Tim Dettmers, Naman Goyal, and Luke Zettlemoyer. Base layers: Simplifying training of large, sparse models. In International Conference on Machine Learning, pages 6265–6274. PMLR, 2021. [64] Aidan Clark, Diego de las Casas,
```
- keyword `parallelism`:
```text
predictions or intermediate features of all the tasks, both in training and inference, to improve the predictions. However, activating all tasks in inference violates our motivation: sparsely activating the network to achieve efficient MTL inference. Moreover, those models consume a large number of FLOPs [14], which makes them difficult to deploy onto real-world edge devices with resource and latency constraints. This is because they need higher parallelism factors, more resources, or clever tricks to hit the desired latency requirement, which is out of scope of the discussion of this paper. Ignoring the previously mentioned efficiency and memory bottleneck, we conduct comparisons between our M3 ViT-base model and decoder-focused work PAD-Net [42], which have similar FLOPs (PAD-Net: 212 GFLOPs vs. Ours: 191 GFLOPs). Our MoE ViT-base model achieves higher performance than PAD-Net on both
```
- keyword `buffer`:
```text
ving highly sparse and efficient inference for the specific task. In the hardware level, we propose a novel computation reordering mechanism tailored for memory-constrained MTL and MoE, which allows scaling up to any number of experts and also achieves zero-overhead switching between tasks. Specifically, based on ViT, we push tokens to per-expert queues to enable expert-by-expert computation rather than token-by-token. We then implement a double-buffered computation strategy that hides the memory access latency required to load each expert’s weights from off-chip memory, regardless of task-specific expert selection. This design naturally incurs no overhead for switching between frames or tasks in FPGA. To validate the effectiveness, we evaluate our performance gain using the ViT-small backbone on the NYUD-v2 and PASCAL-Context datasets. On the NYUD-v2 dataset with two tasks, our model ac
```

## 4. 하드웨어 아키텍처
- 요약: ZCU104 FPGA에서 sparse expert execution과 memory-efficient task switching을 co-design한다.

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
e- tween tasks and can scale to any number of experts. Extensive experiments on PASCAL-Context [1] and NYUD-v2 [2] datasets at both software and hardware levels are conducted to demonstrate the effectiveness of the proposed design. When executing single-task inference, M3 ViT achieves higher accuracies than encoder- focused MTL methods, while significantly reducing 88% inference FLOPs. When implemented on a hardware platform of one Xilinx ZCU104 FPGA, our co-design framework reduces the memory requirement by 2.40×, while achieving energy efficiency up to 9.23× higher than a comparable FPGA baseline. Code is available at: https://github.com/VITA-Group/M3ViT. 1 Introduction Vision Transformers (ViTs) [3, 4, 5, 6], as the latest performant deep models, have achieved impressive performance on various computer vision tasks [7, 8, 9]. These models are specially trained or tested ∗ Equal contri
```
- keyword `ZCU104`:
```text
ching be- tween tasks and can scale to any number of experts. Extensive experiments on PASCAL-Context [1] and NYUD-v2 [2] datasets at both software and hardware levels are conducted to demonstrate the effectiveness of the proposed design. When executing single-task inference, M3 ViT achieves higher accuracies than encoder- focused MTL methods, while significantly reducing 88% inference FLOPs. When implemented on a hardware platform of one Xilinx ZCU104 FPGA, our co-design framework reduces the memory requirement by 2.40×, while achieving energy efficiency up to 9.23× higher than a comparable FPGA baseline. Code is available at: https://github.com/VITA-Group/M3ViT. 1 Introduction Vision Transformers (ViTs) [3, 4, 5, 6], as the latest performant deep models, have achieved impressive performance on various computer vision tasks [7, 8, 9]. These models are specially trained or tested ∗ Equal
```
- keyword `DSP`:
```text
er on ImageNet following the same strategy as its counterpart DeiT ViT encoder in [4]. MTL Training For both NYUD-v2 and PASCAL-Context datasets, we adopt a polynomial learning rate decay schedule and employ SGD as the optimizer with initial learning rate 0.002. Momentum and weight decay are set to 0.9 and 0.0001, respectively. The batch size is 16. A.3 Hardware Details Platform Specifications Our targeted FPGA, the Xilinx ZCU104 FPGA, has 1,728 DSPs, 504K LUTs, 461K registers, 11 Mbit block RAM, and 27 Mbit UltraRAM. Our GPU used for baseline measurements, the NVIDIA Quadro RTX 8000, has 4,608 CUDA cores and 48 GB of GDDR6 memory. It runs at a clock frequency of 1,395 MHz and consumes 295 W of power. B More Experiment Results B.1 Additional Experiments on ViT-tiny and ViT-base We further evaluate M3 ViT on different variants of ViT including ViT-tiny and ViT-base; results are shown in T
```

## 5. 실험 방법
```text
results on its entire token queue while loading another expert’s parameters, swapping buffers between
iterations.


                                                                     6

 Table 1: Comparisons with encoder-focused MTL architectures on the PASCAL-Context dataset.
                                           Seg. Norm. H. Parts Sal.        Edge ∆m FLOPS Energy
Model                       Backbone
                                         (mIoU↑) (mErr)↓ (mIoU)↑ (mIoU)↑ (odsF) ↑ (%) ↑ (G) ↓ (W·s)↓
STL-B                       ResNet-18      66.2    13.9    59.9    66.3    68.8 0.00 167       1.029
MTL-B                       ResNet-18      63.8    14.9     58.6     65.1    69.2   −2.86   167    1.029
Uncertainty [25] (MTL-B)    ResNet-18      65.4    16.5     59.2     65.6    68.6   −4.60   167    1.029
DWA [52] (MTL-B)            ResNet-18      63.4    14.9     58.9     65.1    69.1   −2.94   167    1.029
GradNorm [20] (MTL-B)       ResNet-18      64.7    15.4     59.0     64.5    67.0   −3.97   167    1.029
MGDA [27] (MTL-B)           ResNet-18      64.9    15.6     57.9     62.5    61.4   −6.81   167    1.029
MTAN [27]                   ResNet-18      63.7    14.8     58.9     65.4    69.6   −2.39   212    5.306
Cross-Stitch [23]           ResNet-18      66.1    13.9     60.6     66.8    69.9   +0.60   647    6.001
NDDR-CNN [26]               ResNet-18      65.4    13.9     60.5     66.8    69.8   +0.39   747    5.034
M-ViT (MTL-B)           ViT-small          70.7    15.5     58.7     64.9    68.8   −1.77    83    3.062
M2 ViT (+MoE)          MoE ViT-small       72.8    14.5     62.1     66.3    71.7   +2.71    84    7.446
M3 ViT (+MoE+Codesign) MoE ViT-small       72.8    14.5     62.1     66.3    71.7   +2.71    84    0.690


Specifically, we propose to add each token to a queue for its selected top-K experts, instead of
computing the token output immediately.
Our hardware then makes use of the per-expert queues via a double-buffered computation flow, also
known as ping-pong buffering: one buffer is filled with an expert’s weights from off-chip memory
accesses, while another already-loaded buffer is used to compute another expert’s results for its entire
token queue. After both operations finish, the buffers are swapped, and the process repeats.
Scalability and Efficiency Our approach hides nearly all latency from off-chip memory accesses
to load expert weights, and it uses O(1) on-chip memory with respect to K and N , making it scalable
to any number of experts. Additionally, our method’s efficiency does not rely on any specific usage
pattern of experts for a given frame or a given task, so we naturally achieve zero-overhead switches
between frames and between tasks. Task switches and frame switches in our hardware design do not
change our computation flow at all, and there is no specific step taken to execute the switch.

4     Experiments
4.1   Experiment Setup

To evaluate the propose method, we conduct experiments on two popu
```

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `ImageNet`:
```text
dimension corresponding to task prediction) and four upsampling layers. This decoder is of lighter weight and consumes fewer FLOPs than Deeplab. The output feature of last and second last conv layers will also be used in a multi-tasks feature distillation module. The distillation module will only be used during train stage and deactivated during inference stage, thus adding no extra FLOPs to the whole network. A.2 Training Setup Pre-training on ImageNet During the MTL pre-train stage, all the encoder backbones will be pre-trained on ImageNet and the decoder will be randomly initialized. In the M-ViT models, we use the pre-trained weights provided by DeiT [4] to initialize all the transformer layers and the input linear projection layer in the encoder. In the MoE ViT models, we pre-train our encoder on ImageNet following the same strategy as its counterpart DeiT ViT encoder in [4]. MTL T
```
- keyword `DeiT`:
```text
s (a.k.a. tokens) are then passed through several consecutive transformer layers. Each layer contains a self-attention module and a feed-forward network (MLPs). The self-attention is computed using the scaled-dot product: ! QK T Attention(Q, K, V ) = softmax √ V (1) C where Q, K, V ∈ RN ×C are the query, key and value matrices computed from input tokens; N and C indicate the token number and the hidden dimension. In our experiments, we adopt the DeiT [4] as the backbone encoder, which is a data-efficient ViT variant that distills tokens to ensure the student learns from the teacher through attention. Mixture of Experts Layer A Mixture of Experts (MoE) layer typically consists a group of N experts f1 , f2 , · · · , fN along with a router R (or gating network) to select the corresponding experts. The experts network stands for multi-layer perceptrons [61, 100] in ViTs. The router R plays a
```
- keyword `dataset`:
```text
ask of interest, the same design allows for activating only the task-corresponding sparse “expert” pathway, instead of the full model. Our new model design is further enhanced by hardware-level innovations, in particular, a novel computation reordering scheme tailored for memory-constrained MTL that achieves zero-overhead switching be- tween tasks and can scale to any number of experts. Extensive experiments on PASCAL-Context [1] and NYUD-v2 [2] datasets at both software and hardware levels are conducted to demonstrate the effectiveness of the proposed design. When executing single-task inference, M3 ViT achieves higher accuracies than encoder- focused MTL methods, while significantly reducing 88% inference FLOPs. When implemented on a hardware platform of one Xilinx ZCU104 FPGA, our co-design framework reduces the memory requirement by 2.40×, while achieving energy efficiency up to 9.23
```

## 7. 실험 결과
- keyword `energy`:
```text
Context [1] and NYUD-v2 [2] datasets at both software and hardware levels are conducted to demonstrate the effectiveness of the proposed design. When executing single-task inference, M3 ViT achieves higher accuracies than encoder- focused MTL methods, while significantly reducing 88% inference FLOPs. When implemented on a hardware platform of one Xilinx ZCU104 FPGA, our co-design framework reduces the memory requirement by 2.40×, while achieving energy efficiency up to 9.23× higher than a comparable FPGA baseline. Code is available at: https://github.com/VITA-Group/M3ViT. 1 Introduction Vision Transformers (ViTs) [3, 4, 5, 6], as the latest performant deep models, have achieved impressive performance on various computer vision tasks [7, 8, 9]. These models are specially trained or tested ∗ Equal contribution 36th Conference on Neural Information Processing Systems (NeurIPS 2022). for onl
```
- keyword `accuracy`:
```text
ardware platform of one Xilinx ZCU104 FPGA, which enables us to exploit a memory-efficient computation reordering scheme that consolidates per-expert Multiply-and-ACcumulate (MAC) operations such that only one expert’s weights are needed on-chip at a time. Our design is scalable to any number of experts while requiring no frame-switching or task-switching overhead. • We conduct extensive experiments to justify its inference effectiveness in both accuracy and on-edge efficiency metrics. Our framework, dubbed M3 ViT, achieves higher accuracies than encoder-focused MTL methods, while significantly reducing 88% inference FLOPs; on hardware, it reduces the memory requirement by 2.40× and costs up to 9.23× and 10.79× less energy compared to the FPGA and GPU baselines, respectively. 2 Related Works Multi-task Learning The generic multi-task learning problem has been studied for a long history.
```
- keyword `latency`:
```text
as A&M University, 4 Protagolabs Inc, 5 Microsoft Research {haliang,zhiwenfan,tianlong.chen,atlaswang}@utexas.edu {rishov.sarkar,callie.hao}@gatech.edu, jiangziyu@tamu.edu kz@protagolabs.com, yu.cheng@microsoft.com Abstract Multi-task learning (MTL) encapsulates multiple learned tasks in a single model and often lets those tasks learn better jointly. However, when deploying MTL onto those real-world systems that are often resource-constrained or latency-sensitive, two prominent challenges arise: (i) during training, simultaneously optimizing all tasks is often difficult due to gradient conflicts across tasks, and the challenge is amplified when a growing number of tasks have to be squeezed into one compact model; (ii) at inference, current MTL regimes have to activate nearly the entire model even to just execute a single task. Yet most real systems demand only one or two tasks at each mo
```

## 8. 실험 옵션 / Ablation 축
- expert count and top-k/static task gate
- search-only vs track-only expert specialization
- shared backbone vs mode-specific head

## 9. HGTXR 적용 판단
- eye tracking의 search/track/head 분기와 직접 맞는다. SW 학습/정확도 실험 우선, HW는 static expert only.

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
