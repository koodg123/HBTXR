---
source_type: paper
source_name: EfficientViT-FPGA
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/EfficientViT-FPGA/analysis.md -->

# EfficientViT-FPGA Detailed Paper Analysis

- title: An FPGA-Based Reconfigurable Accelerator for Convolution-Transformer Hybrid EfficientViT
- url: https://arxiv.org/abs/2403.20230
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/EfficientViT-FPGA.pdf`
- related_codebases: `efficient-transformer-accelerator`

## 1. 기존 방법의 문제점
- 요약: EfficientViT의 conv-transformer hybrid 구조를 기존 accelerator가 충분히 활용하지 못한다.

### Paper Evidence: Abstract
```text
Abstract—Vision Transformers (ViTs) have achieved significant           MBConv Head                                     Linear Projection
                                         success in computer vision. However, their intensive computa-                                       MBConv
arXiv:2403.20230v1 [cs.AR] 29 Mar 2024




                                         tions and massive memory footprint challenge ViTs’ deployment                                                                Concatenate
                                         on embedded devices, calling for efficient ViTs. Among them,                EfficientViT          Lightweight
                                                                                                                       Module     ×L          MSA                              ReLU
                                         EfficientViT, the state-of-the-art one, features a Convolution-
                                                                                                                                                                              Global
                                         Transformer hybrid architecture, enhancing both accuracy and                  MBConv                                      ReLU      Attention
                                         hardware efficiency. Unfortunately, existing accelerators cannot                                                         Global
                                         fully exploit the hardware benefits of EfficientViT due to its                MBConv ×N             PWConv              Attention    GConv
                                         unique architecture. In this paper, we propose an FPGA-base
```

## 2. 제안하는 방법
- 요약: operation type별 reconfigurable architecture와 time-multiplexed pipelined dataflow를 제안한다.

## 3. 구체적인 알고리즘
- 핵심 알고리즘: lightweight convolution과 attention을 layer fusion 관점에서 배치한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `parallelism`:
```text
in a reduction in local feature extraction MSA), which exhibits distinct computational patterns com- capabilities. This limitation necessitates the incorporation of pared to the vanilla self-attention in standard ViTs. Moreover, the aforementioned lightweight components in EfficientViT This work was supported in part by the National Key R&D Program of China under Grant 2022YFB4400604. (Corresponding Author: Wendong Mao exhibit reduced computing parallelism and fewer data reuse and Zhongfeng Wang) opportunities than their standard counterparts, yielding either √ 𝐶𝐶𝑖𝑖 𝐶𝐶𝑖𝑖 Exp(QK T / d)) into ReLU(Q)ReLU(K)T , thus not only ⊛ ⊛ eliminating the need for Softmax but also achieving linear 𝐾𝐾ℎ 𝐶𝐶𝑜𝑜 computational complexity by utilizing the associative property 𝐾𝐾𝑤𝑤 of matrix multiplication. 1 DWConv 𝐶𝐶𝑖𝑖 PWConv 𝐶𝐶𝑜𝑜 (a) III. P ROPOSED H ARDWARE D ESIGN ReLU(𝑄𝑄𝑖𝑖 )(∑𝑁𝑁 𝑇𝑇 𝑗𝑗=1 ReLU(𝐾𝐾𝑗𝑗 ) 𝑉𝑉𝑗𝑗
```
- keyword `buffer`:
```text
lity to a large scale. which is a combination of a depth-wise convolution (DWConv Moreover, DWConvs with different kernel sizes and strides in Fig. 2(a) left) and a point-wise convolution (PWConv in Fig. yield distinct overlap patterns between adjacent sliding win- 2(a) right). After that, two key types of blocks are involved in dows when conducting convolutions, resulting in significant EfficientViT: the MBConv [13] and the EfficientViT Module. buffer overheads or complex memory management to support The MBConv features two PWConvs separated by a DWConv. the generation of consecutive output pixels within MATs [19]. Each layer is followed by BatchNorm (BN) and Hardswish activation [14] (except the final PWConv). Notably, BN can B. Reconfigurable Architecture Design be implemented via 1 × 1 convolutions, which can be inte- To boost flexibility, we develop a reconfigurable processing grate
```
- keyword `batch size`:
```text
f matrix multiplication. 1 DWConv 𝐶𝐶𝑖𝑖 PWConv 𝐶𝐶𝑜𝑜 (a) III. P ROPOSED H ARDWARE D ESIGN ReLU(𝑄𝑄𝑖𝑖 )(∑𝑁𝑁 𝑇𝑇 𝑗𝑗=1 ReLU(𝐾𝐾𝑗𝑗 ) 𝑉𝑉𝑗𝑗 ) 𝑂𝑂𝑖𝑖 = As discussed above, there are four main types of operations V ReLU(𝑄𝑄𝑖𝑖 )(∑𝑁𝑁 𝑇𝑇 𝑗𝑗=1 ReLU(𝐾𝐾𝑗𝑗 ) ) Transpose in the backbone of EfficientViT: generic Convs, PWConvs, K ReLU ReLU(𝐾𝐾)T Z d×1 DWConvs, and matrix multiplications (MatMuls). As MatMuls Row-wise N Q Summation N×d can be treated as PWConvs with large batch sizes, an efficient ReLU ÷ O hardware architecture that can effectively handle MSA and d d×1 various types of convolutions is highly desired. Additionally, (b) lightweight operations (i.e., PWConvs/DWConvs/MSA) in Ef- Fig. 2. (a) DSConv: Depthwise Convolution followed by Pointwise Convolu- tion. (b) The computation flow of ReLU-based global attention in EfficientViT. ficientViT features reduced computing parallelism and fewer data reuse opp
```

## 4. 하드웨어 아키텍처
- 요약: ZCU102에서 intra/inter-layer fusion, shared datapath, quantized execution으로 구성된다.

### Paper Evidence: Architecture / Implementation
```text
hardware architecture that can effectively handle MSA and
        d                                                                    d×1
                                                                                                             various types of convolutions is highly desired. Additionally,
                                              (b)
                                                                                                             lightweight operations (i.e., PWConvs/DWConvs/MSA) in Ef-
Fig. 2. (a) DSConv: Depthwise Convolution followed by Pointwise Convolu-
tion. (b) The computation flow of ReLU-based global attention in EfficientViT.                               ficientViT features reduced computing parallelism and fewer
                                                                                                             data reuse opportunities, calling for an effective dataflow to en-
high memory bandwidth requirement or low computation                                                         hance hardware utilization and ease bandwidth requirements.
resource utilization. Thus, in this paper, we present an FPGA-
based accelerator for EfficientViT to tackle these challenges.                                               A. Multipliers and Adder-Trees Design Paradigm
The main contributions are summarized as follows.                                                               Convolutions in neural networks can be fundamentally de-
   • A reconfigurable architecture is designed to efficiently                                                composed into a series of multiplication and addition computa-
     support various operation types in the Convolution-                                                     tions. Hence, parallelized hardware architectures incorporating
     Transformer hybrid architecture of EfficientViT, includ-                                                multipliers and adder-trees (MAT) offer a straightforward and
     ing lightweight convolutions and lightweight attention.                                                 efficient solution for generic Convs and PWConvs (which are
   • A novel time-multiplexed and pipelined dataflow is pro-                                                 essentially generic Convs with 1×1 kernels). Specifically, in-
     posed to fuse computations among adjacent lightweight                                                   puts or weights can be
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
An FPGA-Based Reconfigurable Accelerator for Convolution-Transformer Hybrid EfficientViT Haikuo Shao1 , Huihong Shi1 , Wendong Mao2 , and Zhongfeng Wang1,2 1 School of Electronic Science and Engineering, Nanjing University, Nanjing, China 2 School of Integrated Circuits, Sun Yat-sen University, Shenzhen, China Email: {hkshao, shihh}@smail.nju.edu.cn, maowd@mail.sysu.edu.cn, zfwang@nju.edu.cn Abstract—Vision Transformers (ViTs) have achieved signific
```
- keyword `ZCU102`:
```text
BConv consists of dataflow to facilitate both intra- and inter-layer fusions, reducing two pointwise convolutions (PWConvs) separated by a depthwise convolution off-chip data access costs. Experimental results show that our (DWConv). Besides, the key component of the EfficientViT module is the accelerator achieves up to 780.2 GOPS in throughput and 105.1 lightweight Multi-Scale Attention (MSA). GOPS/W in energy efficiency at 200MHz on the Xilinx ZCU102 FPGA, which significantly outperforms prior works. supplementary components such as convolutions [8], [11], Index Terms—Vision Transformer, convolution, hybrid archi- yielding hybrid architectures for efficient ViTs that integrate tecture, hardware accelerator, FPGA both convolutions and Transformer blocks. Particularly, the state-of-the-art (SOTA) efficient ViT, dubbed EfficientViT [8], I. I NTRODUCTION can achieve higher accuracy than Sw
```
- keyword `Vivado`:
```text
sh the final divisions of MSA. efficiency, and ↑5.9× DSP efficiency; (3) Although Auto-ViT- Acc consumes 1.9× more DSP resources than us, we can offer IV. E XPERIMENTAL R ESULTS ↑1.1× throughput, ↑1.25× energy efficiency, and ↑2.1× DSP efficiency, further validating our effectiveness. A. Experimental Setup V. C ONCLUSION Our accelerator is coded with Verilog, synthesized and In this paper, we proposed an FPGA-based accelerator for implemented by Vivado Design Suite, and evaluated on Xilinx Convolution-Transformer hybrid networks like EfficientViT. ZCU102 FPGA at 200-MHz frequency. The hardware resource Specifically, we design a reconfigurable design to effectively of (M × N + S × T ) × L is configured as (8 × 8 + 8 × 8) × 16. support various types of convolutions and the Multi-Scale Each multiplier in both RPE and MAT engines can execute the Attention (MSA). Furthermore, we propose a tim
```

## 5. 실험 방법
- 실험 section heading을 자동 추출하지 못했다. PDF 원문 확인 필요.

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `DeiT`:
```text
s such as convolutions [8], [11], Index Terms—Vision Transformer, convolution, hybrid archi- yielding hybrid architectures for efficient ViTs that integrate tecture, hardware accelerator, FPGA both convolutions and Transformer blocks. Particularly, the state-of-the-art (SOTA) efficient ViT, dubbed EfficientViT [8], I. I NTRODUCTION can achieve higher accuracy than Swin-T [12] (by +1.4%) Recently, Vision Transformers (ViTs) have been proposed and DeiT [2] (by +2.9%) with a comparable number of and attracted increasing attention in the computer vision field parameters. As illustrated in Fig. 1, EfficientViT features a [1], [2]. Despite ViTs’ remarkable performance against their Convolution-Transformer hybrid architecture, primarily com- convolution-based counterparts, the intensive computations prising MBConvs [13] and EfficientViT Modules. The latter and huge memory footprint during infer
```
- keyword `EfficientViT`:
```text
An FPGA-Based Reconfigurable Accelerator for Convolution-Transformer Hybrid EfficientViT Haikuo Shao1 , Huihong Shi1 , Wendong Mao2 , and Zhongfeng Wang1,2 1 School of Electronic Science and Engineering, Nanjing University, Nanjing, China 2 School of Integrated Circuits, Sun Yat-sen University, Shenzhen, China Email: {hkshao, shihh}@smail.nju.edu.cn, maowd@mail.sysu.edu.cn, zfwang@nju.edu.cn Abstract—Vision Transformers (ViTs) have achieved significant MBConv Head Linear Projection success in computer vision. However, th
```

## 7. 실험 결과
- keyword `speedup`:
```text
icient ViT) accelerator ViA [16] with FP16 divisors and dividends of MSA, respectively. During this format, and a standard ViT (DeiT) accelerator Auto-Vit-Acc process, the pre-generated divisors are temporarily saved in [17] with FIX8 precision. From Table II, we can see that: a small divisor buffer. Once dividends are computed by the (1) Compared with EfficientViT on CPU, we can gain 14.3× MAT engine, they can be divided by the previously saved speedup and ↑21.1× energy efficiency; (2) Compared to divisors via dividers in the post-processing module (Fig. 4) to ViA, our design achieves ↑2.0× throughput, ↑13.3× energy accomplish the final divisions of MSA. efficiency, and ↑5.9× DSP efficiency; (3) Although Auto-ViT- Acc consumes 1.9× more DSP resources than us, we can offer IV. E XPERIMENTAL R ESULTS ↑1.1× throughput, ↑1.25× energy efficiency, and ↑2.1× DSP efficiency, further validating
```
- keyword `energy`:
```text
o architecture of EfficientViT [8]. Each MBConv consists of dataflow to facilitate both intra- and inter-layer fusions, reducing two pointwise convolutions (PWConvs) separated by a depthwise convolution off-chip data access costs. Experimental results show that our (DWConv). Besides, the key component of the EfficientViT module is the accelerator achieves up to 780.2 GOPS in throughput and 105.1 lightweight Multi-Scale Attention (MSA). GOPS/W in energy efficiency at 200MHz on the Xilinx ZCU102 FPGA, which significantly outperforms prior works. supplementary components such as convolutions [8], [11], Index Terms—Vision Transformer, convolution, hybrid archi- yielding hybrid architectures for efficient ViTs that integrate tecture, hardware accelerator, FPGA both convolutions and Transformer blocks. Particularly, the state-of-the-art (SOTA) efficient ViT, dubbed EfficientViT [8], I. I NTROD
```
- keyword `accuracy`:
```text
d significant MBConv Head Linear Projection success in computer vision. However, their intensive computa- MBConv arXiv:2403.20230v1 [cs.AR] 29 Mar 2024 tions and massive memory footprint challenge ViTs’ deployment Concatenate on embedded devices, calling for efficient ViTs. Among them, EfficientViT Lightweight Module ×L MSA ReLU EfficientViT, the state-of-the-art one, features a Convolution- Global Transformer hybrid architecture, enhancing both accuracy and MBConv ReLU Attention hardware efficiency. Unfortunately, existing accelerators cannot Global fully exploit the hardware benefits of EfficientViT due to its MBConv ×N PWConv Attention GConv unique architecture. In this paper, we propose an FPGA-based DSConv DWConv DWConv accelerator for EfficientViT to advance the hardware efficiency frontier of ViTs. Specifically, we design a reconfigurable archi- Conv PWConv Linear Projection tectu
```
- keyword `throughput`:
```text
ditionally, we present a time-multiplexed and pipelined Fig. 1. The macro architecture of EfficientViT [8]. Each MBConv consists of dataflow to facilitate both intra- and inter-layer fusions, reducing two pointwise convolutions (PWConvs) separated by a depthwise convolution off-chip data access costs. Experimental results show that our (DWConv). Besides, the key component of the EfficientViT module is the accelerator achieves up to 780.2 GOPS in throughput and 105.1 lightweight Multi-Scale Attention (MSA). GOPS/W in energy efficiency at 200MHz on the Xilinx ZCU102 FPGA, which significantly outperforms prior works. supplementary components such as convolutions [8], [11], Index Terms—Vision Transformer, convolution, hybrid archi- yielding hybrid architectures for efficient ViTs that integrate tecture, hardware accelerator, FPGA both convolutions and Transformer blocks. Particularly, the st
```
- keyword `GOPS`:
```text
tion. Additionally, we present a time-multiplexed and pipelined Fig. 1. The macro architecture of EfficientViT [8]. Each MBConv consists of dataflow to facilitate both intra- and inter-layer fusions, reducing two pointwise convolutions (PWConvs) separated by a depthwise convolution off-chip data access costs. Experimental results show that our (DWConv). Besides, the key component of the EfficientViT module is the accelerator achieves up to 780.2 GOPS in throughput and 105.1 lightweight Multi-Scale Attention (MSA). GOPS/W in energy efficiency at 200MHz on the Xilinx ZCU102 FPGA, which significantly outperforms prior works. supplementary components such as convolutions [8], [11], Index Terms—Vision Transformer, convolution, hybrid archi- yielding hybrid architectures for efficient ViTs that integrate tecture, hardware accelerator, FPGA both convolutions and Transformer blocks. Particularly
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
- hybrid patch stem이나 lightweight local feature extractor를 붙이는 accuracy 실험에 유용하지만 paper-scope 검토 필요.

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
