---
source_type: paper
source_name: UbiMoE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/UbiMoE/analysis.md -->

# UbiMoE Detailed Paper Analysis

- title: UbiMoE: A Ubiquitous Mixture-of-Experts Vision Transformer Accelerator With Hybrid Computation Pattern on FPGA
- url: https://arxiv.org/abs/2502.05602
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/UbiMoE.pdf`
- related_codebases: `UbiMoE`

## 1. 기존 방법의 문제점
- 요약: 기존 FPGA MoE accelerator는 resource constraint별 design-space exploration이 부족해 throughput/resource tradeoff가 비최적이다.

### Paper Evidence: Abstract
```text
Abstract—Compared to traditional Vision Transformers (ViT),                                                                    Gate
                                         Mixture-of-Experts Vision Transformers (MoE-ViT) are intro-                                 Q                                              Expert 0




                                                                                                                                                      Linear proj
                                                                                                                                         Attention
                                         duced to scale model size without a proportional increase in




                                                                                                                            Linear




                                                                                                                                                                    Norm




                                                                                                                                                                                               Norm
                                                                                                                             QKV
                                                                                                                    Input            K                                              Expert 1
```

## 2. 제안하는 방법
- 요약: latency-optimized streaming attention과 resource-efficient reusable linear kernel, two-stage heuristic search를 결합한다.

### Paper Evidence: Method Section
```text
Algorithm 1: 2-stage Hardware Accelerator Search
       query q1 q2 q3 q4 q5 q6                      query q11 q22 q33 q44 qq55 qq66                     (HAS) Process
                                                                                                   1  𭟋c = [num, Ta , Na , Tin , Tout , NL ]c
               PE1        PE2     PE3                       PE11     PE22       PE33               2  Initialize the hardware constraint (Dtotal , Btotal , BW total )
                                                                                                      // MoE stage part 1
                                                                                                    3 Calculate the best latency LMoE of MoE block depending on the
        key     k1   k2     k3   k4    k5      k6    key     k11   k22   k33   k44   kk55   kk66
                                                                                                        constraint (Dtotal )
                                                                                                      // MSA stage
       (a) Traditonal Single-q computation.          (b) Computation after reordering.
                                                                                                    4 for c in num do

      Fig. 4: Running process before and after optimization. Blue q                                 5       Randomly initialize each individual
                                                                                                    6       Set Fit Scores to LMoE /LMSA
      blocks are fixed to specific PEs, while the color of k blocks                                 7       while i <Iteration do
      changes during kernel running.                                                                8            Use traditional GA algorithm to calculate the best LMSA
                                                                                                    9            if Fit Scores ≥ 1 then
      sponding max registers m(x) to store the respective maximum                                  10                 Return latency (LMoE ) and hardware parameters
      values. As both computations run simultaneously, the runtime
                                                                                                      // MoE stage part 2
      latency remains unchanged compared to the former QK dot.                                     11 Use binary search f
```

## 3. 구체적인 알고리즘
- 핵심 알고리즘: resource constraint를 입력으로 하여 hardware parameter를 탐색하고 MoE attention/linear kernel을 hybrid computation pattern으로 배치한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `bit-width`:
```text
e same for calculations within a single the hardware generation process. Due to space limitations, we head, after obtaining the sum of the results, only one division only present the modeling process of the attention kernel. operation is needed to produce the final result, reducing the DSP resources are typically used for multiplication and number of computations required. accumulation. Hence, the DSP usage in the attention kernel depends on the bit-width of the input data, the degree of C. Reusable Linear Kernel parallelism in the attention PEs, and the dimensions Ta after It feels intuitive to implement parallel execution using mul- tiling. Overall, the total DSP utilization can be expressed as: tiple kernels for linear computations. However, as mentioned Dattn = (2Ψ(q) × Ta + Dexp × h) × Na (2) in Sec. II, the patch indices corresponding to the chosen expert are dynamic. In this case,
```
- keyword `sparsity`:
```text
at Lakes Symposium on VLSI 2024, 2024, pp. 599–603. [14] X. Liu, H. Peng, N. Zheng, Y. Yang, H. Hu, and Y. Yuan, “Efficientvit: Memory efficient vision transformer with cascaded group attention,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 420–14 430. [15] R. Sarkar, H. Liang, Z. Fan, Z. Wang, and C. Hao, “Edge-moe: Memory- efficient multi-task vision transformer architecture with task-level sparsity via mixture-of-experts,” in 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD). IEEE, 2023, pp. 01–09. [16] W. Lou, L. Gong, C. Wang, Z. Du, and X. Zhou, “Octcnn: A high throughput fpga accelerator for cnns using octave convolution algorithm,” IEEE Transactions on Computers, vol. 71, no. 8, pp. 1847–1859, 2021. [17] S.-C. Kao, S. Subramanian, G. Agrawal, A. Yazdanbakhsh, and T. Kr- ishna, “Flat: An optimized dataflow
```
- keyword `parallelism`:
```text
ss. Due to space limitations, we head, after obtaining the sum of the results, only one division only present the modeling process of the attention kernel. operation is needed to produce the final result, reducing the DSP resources are typically used for multiplication and number of computations required. accumulation. Hence, the DSP usage in the attention kernel depends on the bit-width of the input data, the degree of C. Reusable Linear Kernel parallelism in the attention PEs, and the dimensions Ta after It feels intuitive to implement parallel execution using mul- tiling. Overall, the total DSP utilization can be expressed as: tiple kernels for linear computations. However, as mentioned Dattn = (2Ψ(q) × Ta + Dexp × h) × Na (2) in Sec. II, the patch indices corresponding to the chosen expert are dynamic. In this case, using pre-configured allocation may Specifically, Ψ(q) represents th
```

## 4. 하드웨어 아키텍처
- 요약: streaming attention kernel과 reusable linear kernel을 중심으로 ZCU102/U280같은 서로 다른 resource budget에 맞춘다.

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
Gating Expert2 Expert2 Layer Layer Norm ... Norm ... Expertn-1 Expertn-1 Task2 Gating Expertn Expertn UbiMoE: A Ubiquitous Mixture-of-Experts Vision Transformer Accelerator With Hybrid Computation V Pattern on FPGA Q K Jiale Dong, Wenqi Lou† , Zhendong Zheng, Yunji Qin, Lei Gong, Chao Wang† , Xuehai Zhou University of Science and Technology of China, Hefei, China Suzhou Institute for Advanced Research, University of Science and Technology of China, Suzhou, China {louwenqi, cswang}@ustc.edu.cn Abstract—Compared to traditional Vision Transformers (ViT), Gate Mixture-of-Experts Vision Transformers (MoE-ViT) are intro- Q Expert 0 Linear proj Attention duce
```
- keyword `ZCU102`:
```text
cy, we propose a two-stage heuristic iting the ability to find optimal solutions across different FPGA search algorithm that optimally tunes hardware parameters for platforms [18], [19]. As FPGA platforms continue to evolve various FPGA resource constraints. Compared to state-of-the- art (SOTA) FPGA designs, UbiMoE achieves 1.34× and 3.35× and offer richer resources (e.g., Alveo U250/U280 with multi- throughput improvements for MoE-ViT on Xilinx ZCU102 and chip architectures), efficiently utilizing computational resources Alveo U280 platforms, respectively, while enhancing energy effi- becomes crucial. To this end, we propose UbiMoE, an efficient ciency by 1.75× and 1.54×. Our implementation is available at FPGA-based accelerator for MoE-ViTs. The design integrates https://github.com/DJ000011/UbiMoE. highly optimized kernels with distinct computation patterns while employing heuristic al
```
- keyword `U280`:
```text
design space exploration, lim- further enhance design efficiency, we propose a two-stage heuristic iting the ability to find optimal solutions across different FPGA search algorithm that optimally tunes hardware parameters for platforms [18], [19]. As FPGA platforms continue to evolve various FPGA resource constraints. Compared to state-of-the- art (SOTA) FPGA designs, UbiMoE achieves 1.34× and 3.35× and offer richer resources (e.g., Alveo U250/U280 with multi- throughput improvements for MoE-ViT on Xilinx ZCU102 and chip architectures), efficiently utilizing computational resources Alveo U280 platforms, respectively, while enhancing energy effi- becomes crucial. To this end, we propose UbiMoE, an efficient ciency by 1.75× and 1.54×. Our implementation is available at FPGA-based accelerator for MoE-ViTs. The design integrates https://github.com/DJ000011/UbiMoE. highly optimized kernels
```

## 5. 실험 방법
- 실험 section heading을 자동 추출하지 못했다. PDF 원문 확인 필요.

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `DeiT`:
```text
Throughput (GOPS) 54.86 72.15 97.04 242.01 they are not repeatedly executed. Thus, the target latency for the Efficiency (GOPS/W) 1.075 4.83 8.438 7.451 MSA block can be dynamically adjusted to the lower bound, TABLE III: Comparison with Previous FPGA Implementations LMoE , set by the MoE block. This allows for early termination Attribute HeatViT [11] UbiMoE-E TECS’23 [12] UbiMoE-C of computations, reducing unnecessary processing overhead. Model DeiT-S ViT-T BERT-B ViT-S While the MSA block remains the primary bottleneck after Platform ZCU102 ZCU102 U250 U280 HAS, the previously optimized MoE module becomes idle, Bit-width INT8 INT16 INT8 INT16 Freq. (Mhz) 300 300 300 250 resulting in reduced utilization efficiency. To address this, we Power (W) 10.697 9.94 77.168 31.36 can set the dynamically adjusted latency upper bound, LMSA , Latency (ms) 9.15 8.20 - 11.66 of the MSA block as the tar
```
- keyword `EfficientViT`:
```text
mechanism on fpgas based on efficient reconfigurable systolic array,” ACM Transactions on Embedded Computing Systems, vol. 22, no. 6, pp. 1–22, 2023. [13] Y. Qin, W. Lou, C. Wang, L. Gong, and X. Zhou, “Enhancing long sequence input processing in fpga-based transformer accelerators through attention fusion,” in Proceedings of the Great Lakes Symposium on VLSI 2024, 2024, pp. 599–603. [14] X. Liu, H. Peng, N. Zheng, Y. Yang, H. Hu, and Y. Yuan, “Efficientvit: Memory efficient vision transformer with cascaded group attention,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 420–14 430. [15] R. Sarkar, H. Liang, Z. Fan, Z. Wang, and C. Hao, “Edge-moe: Memory- efficient multi-task vision transformer architecture with task-level sparsity via mixture-of-experts,” in 2023 IEEE/ACM International Conference on Computer Aided Design (ICCAD). IEEE
```

## 7. 실험 결과
- keyword `speedup`:
```text
ciency, surpassing previous work. We benchmarked UbiMoE against leading works, as detailed in Table II. To ensure a fair comparison of accelerator perfor- VII. ACKNOWLEDGMENT mance across different platforms, we use efficiency (GOPS/W) This work was supported in part by the National Key as the evaluation metric. On the ZCU102 platform, compared R&D Program of China under Grants 2022YFB4501600 and to GPU and Edge-MoE [10], we achieve 1.77×, 1.34× speedup, 2022YFB4501603, in part by the National Natural Science and 7.85×, 1.75× energy efficiency improvement. On the U280 Foundation of China under Grants 62102383, 61976200, and platform, due to the DSP consumption in the 32-bit mul- 62172380, in part by Jiangsu Provincial Natural Science Foun- tiplication process and the extra use of resources for data dation under Grant BK20241818, in part by Youth Innovation transfer between the host CPU a
```
- keyword `energy`:
```text
ally tunes hardware parameters for platforms [18], [19]. As FPGA platforms continue to evolve various FPGA resource constraints. Compared to state-of-the- art (SOTA) FPGA designs, UbiMoE achieves 1.34× and 3.35× and offer richer resources (e.g., Alveo U250/U280 with multi- throughput improvements for MoE-ViT on Xilinx ZCU102 and chip architectures), efficiently utilizing computational resources Alveo U280 platforms, respectively, while enhancing energy effi- becomes crucial. To this end, we propose UbiMoE, an efficient ciency by 1.75× and 1.54×. Our implementation is available at FPGA-based accelerator for MoE-ViTs. The design integrates https://github.com/DJ000011/UbiMoE. highly optimized kernels with distinct computation patterns while employing heuristic algorithms to explore deployment I. I NTRODUCTION strategies, achieving optimized solutions across different FPGA The Vision Transfo
```
- keyword `throughput`:
```text
loration, lim- further enhance design efficiency, we propose a two-stage heuristic iting the ability to find optimal solutions across different FPGA search algorithm that optimally tunes hardware parameters for platforms [18], [19]. As FPGA platforms continue to evolve various FPGA resource constraints. Compared to state-of-the- art (SOTA) FPGA designs, UbiMoE achieves 1.34× and 3.35× and offer richer resources (e.g., Alveo U250/U280 with multi- throughput improvements for MoE-ViT on Xilinx ZCU102 and chip architectures), efficiently utilizing computational resources Alveo U280 platforms, respectively, while enhancing energy effi- becomes crucial. To this end, we propose UbiMoE, an efficient ciency by 1.75× and 1.54×. Our implementation is available at FPGA-based accelerator for MoE-ViTs. The design integrates https://github.com/DJ000011/UbiMoE. highly optimized kernels with distinct com
```
- keyword `GOPS`:
```text
ip-Flop (FFs) Platform Tesla V100S ZCU102 ZCU102 U280 ZCU102 (Edge) 1850 458 123.4K 142.6K Bit-width FP32 W 16 A32 W 16 A32 W 16 A32 Alveo U280 (Cloud) 3413 974 316.1K 385.9K Frequency (Mhz) 1245 300 300 200 be deployed independently. As a result, we track the number Power (W) 51 14.54 11.50 32.49 of streaming modules, denoted as num. Latency (ms) 40.1 34.64 25.76 10.33 In practice, modules outside the encoder are less critical since Throughput (GOPS) 54.86 72.15 97.04 242.01 they are not repeatedly executed. Thus, the target latency for the Efficiency (GOPS/W) 1.075 4.83 8.438 7.451 MSA block can be dynamically adjusted to the lower bound, TABLE III: Comparison with Previous FPGA Implementations LMoE , set by the MoE block. This allows for early termination Attribute HeatViT [11] UbiMoE-E TECS’23 [12] UbiMoE-C of computations, reducing unnecessary processing overhead. Model DeiT-S ViT-T
```
- keyword `latency`:
```text
ading to suboptimal trade-offs between resource utilization and per- Fig. 1: The structure of the MoE Vision Transformer. formance. To overcome this problem, we introduce UbiMoE, gling to balance performance and resource utilization effec- a novel end-to-end FPGA accelerator tailored for MoE-ViT. tively. Specifically: 1) the hardware design only emphasizes Leveraging the unique computational and memory access pat- terns of MoE-ViTs, we develop a latency-optimized streaming reusable computational kernels, overlooking latency optimiza- attention kernel and a resource-efficient reusable linear kernel, tion for critical bottlenecks [16], [17]; and 2) the deployment effectively balancing performance and resource consumption. To strategy lacks efficient hardware design space exploration, lim- further enhance design efficiency, we propose a two-stage heuristic iting the ability to find optimal
```

## 8. 실험 옵션 / Ablation 축
- expert count and top-k/static task gate
- search-only vs track-only expert specialization
- shared backbone vs mode-specific head
- exact softmax vs approximated softmax
- Taylor/linear attention vs exact attention
- fixed sparse token pattern vs dense token pattern
- URAM/BRAM/LUTRAM binding
- memory banks 8/16/32
- parallelism factor 8/16/32
- FIFO depth and double buffering

## 9. HGTXR 적용 판단
- C3b 이후 PAR/MEM bank sweep의 search policy 참고로 유용하다. search/track expert 실험의 hardware scheduling 근거가 된다.

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
