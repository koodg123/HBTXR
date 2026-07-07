---
source_type: paper
source_name: FlexLLM
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-paper
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/papers/FlexLLM/analysis.md -->

# FlexLLM Detailed Paper Analysis

- title: FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design
- url: https://arxiv.org/abs/2601.15710
- local_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/FlexLLM.pdf`
- related_codebases: `FlexLLM`

## 1. 기존 방법의 문제점
- 요약: LLM accelerator 개발은 stage별 dataflow/quantization을 재사용 가능하게 구성하기 어렵다.

### Paper Evidence: Abstract
```text
Abstract—We present FlexLLM, a composable High-Level             offering high arithmetic intensity and abundant parallelism.
                                         Synthesis (HLS) library for rapid development of domain-            During decode, tokens are generated autoregressively; compu-
                                         specific LLM accelerators. FlexLLM exposes key architectural        tation is dominated by data dependencies and frequent memory
                                         degrees of freedom for stage-customized inference, enabling
                                         hybrid designs that tailor temporal reuse and spatial dataflow      accesses, making it strongly memory-bandwidth-bound. As
                                         differently for prefill and decode, and provides a comprehensive    a result, the same model is often compute-bound in prefill
                                         quantization suite to support accurate low-bit deployment. Using    but memory-/dependency-bound in decode, creating conflicting
arXiv:2601.15710v1 [cs.AR] 22 Jan 2026




                                         FlexLLM, we build a complete inference system for the Llama-        optimization goals within one serving pipeline.
                                         3.2 1B model in under two months with only 1K lines of code.           Existing FPGA accelerators primarily follow either temporal
                                         The system includes: (1) a stage-customized accelerator with
                                         hardware-efficient quantization (12.68 WikiText-2 PPL) surpassing   architectures [8], [20] or spatial architectures [21], [22],
                                         S
```

## 2. 제안하는 방법
- 요약: composable HLS library와 quantization suite로 hybrid temporal-spatial accelerator를 빠르게 구성한다.

## 3. 구체적인 알고리즘
- 핵심 알고리즘: prefill/decode stage에 따라 dataflow와 reuse를 다르게 구성하는 library abstraction을 제공한다.
- HGTXR 변환 관점:
  - SW reference에서 먼저 algorithmic delta를 검증한다.
  - quantization/attention/expert routing 변경은 bit-exact HW/SW contract를 새로 만든다.
  - 기존 C3b evidence와 비교 가능한 metric 이름을 유지한다.

### 알고리즘/옵션 관련 추출 근거
- keyword `ablation`:
```text
id quantization to retain accuracy. accelerator design. We illustrate a detailed methodology for Integer vocabulary projection. We further quantize stage-customized hybrid design through a case study in Sec. IV. lm_head to INT4, matching the other linear layers, reducing After composing an accelerator with FlexLLM, users can resource cost and improving decode throughput. leverage its seamless compatibility with AutoBridge [39] Table V reports an ablation on WikiText-2. Starting from to optimize placement and routing (P&R), enabling parallel BF16 (PPL = 8.94), the original SpinQuant setup (INT4 linear exploration of design candidates to achieve higher frequency layers with BF16-INT4 attention) yields PPL = 13.30. Raising and performance. This end-to-end toolchain accelerates iteration attention precision to INT8 recovers most of the lost accuracy; and supports efficient on-board deploymen
```
- keyword `parallelism`:
```text
FlexLLM: Composable HLS Library for Flexible Hybrid LLM Accelerator Design Jiahao Zhang∗ , Zifan He∗ , Nicholas Fraser† , Michaela Blott† , Yizhou Sun∗ , Jason Cong∗ ∗ Computer Science, University of California, Los Angeles, California † AMD, Dublin, Ireland Abstract—We present FlexLLM, a composable High-Level offering high arithmetic intensity and abundant parallelism. Synthesis (HLS) library for rapid development of domain- During decode, tokens are generated autoregressively; compu- specific LLM accelerators. FlexLLM exposes key architectural tation is dominated by data dependencies and frequent memory degrees of freedom for stage-customized inference, enabling hybrid designs that tailor temporal reuse and spatial dataflow accesses, making it strongly memory-bandwidth-bound. As differently for pr
```
- keyword `buffer`:
```text
ith hardware-efficient quantization (12.68 WikiText-2 PPL) surpassing architectures [8], [20] or spatial architectures [21], [22], SpinQuant baseline, and (2) a Hierarchical Memory Transformer [23], [24]. Temporal designs reuse shared compute engines (HMT) plug-in for efficient long-context processing. On the AMD across layers, but incur frequent off-chip traffic in prefill due U280 FPGA at 16nm, the accelerator achieves 1.29× end-to- to limited buffering and struggle to support heterogeneous end speedup, 1.64× higher decode throughput, and 3.14× better kernels/precisions within a single engine (Fig. 1(b)(c)). Spatial energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm designs map kernels to dedicated modules and stream inter- reach 4.71×, 6.55×, and 4.13×, respectively. In long-context mediate data through on-chip FIFOs, but s
```

## 4. 하드웨어 아키텍처
- 요약: HLS component library, stage-customized accelerator, long-context HMT plug-in으로 구성된다.

### Paper Evidence: Architecture / Implementation
```text
architectures for the Llama-3.2 1B model. Specifically, we first
                                                                               refine the SpinQuant framework to obtain a more hardware-
                                                                               efficient representation while minimizing accuracy degradation.
                                                                               We then build customized architectures for prefill and decode
                                                                               with composable modules from FlexLLM. Table IV summarizes
                                                                               the major module components, along with their corresponding
                                                                               lines of code and estimated development workload.

                                                                               A. Hardware-Efficient Model Quantization
                                                                                  SpinQuant improves low-bit accuracy by mitigating activa-
                                                                               tion outliers via learned rotations [27]. While many rotations
                                                                               can be folded into weights or implemented efficiently with
Fig. 4. Example code illustrating hybrid architecture construction combining   FHT, the remaining boundary rotations still require FP compute
temporal reuse and spatial dataflow with FlexLLM.                              comparable to a full linear layer, which is costly on FPGAs.
   In the temporal-reuse part, the same templated module is                    Moreover, the baseline SpinQuant setup keeps the MHA query
instantiated and invoked multiple times within a single function               path and the final vocabulary projection (lm_head) in FP,
to sequentially process similar operations. In this example, the               leaving substantial hardware efficiency unrealized.
prefill module of the linear layer and RoPE is reused for                         To obtain a practical model with minimal accuracy loss, we
both Key and Query computations. This approach maximizes                       make the following hardware-oriented refinements:
hardware utilization by reusing computation resources across                      Remove boundary rotations. We eliminate the first
```

### Hardware Keyword Evidence
- keyword `FPGA`:
```text
efill and decode, and provides a comprehensive a result, the same model is often compute-bound in prefill quantization suite to support accurate low-bit deployment. Using but memory-/dependency-bound in decode, creating conflicting arXiv:2601.15710v1 [cs.AR] 22 Jan 2026 FlexLLM, we build a complete inference system for the Llama- optimization goals within one serving pipeline. 3.2 1B model in under two months with only 1K lines of code. Existing FPGA accelerators primarily follow either temporal The system includes: (1) a stage-customized accelerator with hardware-efficient quantization (12.68 WikiText-2 PPL) surpassing architectures [8], [20] or spatial architectures [21], [22], SpinQuant baseline, and (2) a Hierarchical Memory Transformer [23], [24]. Temporal designs reuse shared compute engines (HMT) plug-in for efficient long-context processing. On the AMD across layers, but incur fr
```
- keyword `U280`:
```text
er temporal The system includes: (1) a stage-customized accelerator with hardware-efficient quantization (12.68 WikiText-2 PPL) surpassing architectures [8], [20] or spatial architectures [21], [22], SpinQuant baseline, and (2) a Hierarchical Memory Transformer [23], [24]. Temporal designs reuse shared compute engines (HMT) plug-in for efficient long-context processing. On the AMD across layers, but incur frequent off-chip traffic in prefill due U280 FPGA at 16nm, the accelerator achieves 1.29× end-to- to limited buffering and struggle to support heterogeneous end speedup, 1.64× higher decode throughput, and 3.14× better kernels/precisions within a single engine (Fig. 1(b)(c)). Spatial energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm designs map kernels to dedicated modules and stream inter- reach 4.71×, 6.55×, and 4.13×, re
```
- keyword `Vivado`:
```text
mplementation layout of (a) prefill and (b) decode architectures 10/26/2025 15 Fig. 5(c) shows the end-to-end workflow. A long prompt for quantized Llama-3.2 1B on U280. is split into into multiple short segments. For each segment, the segment processor first forms a summary prompt by measured directly on hardware. For V80 implementation, we concatenating the first half of current segment (SegnT ) with a perform RTL synthesis and P&R using Vitis/Vivado 2024.2 topic token embedding Tn , and sends it to the LLM accelerator and AVED toolchain, and project on-board performance by to produce a topic summary vector Sn . The HMT plug-in then scaling from the U280 implementation. We use Allo [23] with performs cross-attention between Sn and the most recent N W4A8KV8 SmoothQuant as the SOTA accelerator baseline. memory embeddings {M emn−N , . . . , M emn−1 } in a memory GPU Baselines: We use the
```

## 5. 실험 방법
- 실험 section heading을 자동 추출하지 못했다. PDF 원문 확인 필요.

## 6. 데이터셋 / 모델 / 벤치마크
- keyword `LLaMA`:
```text
ns that tailor temporal reuse and spatial dataflow accesses, making it strongly memory-bandwidth-bound. As differently for prefill and decode, and provides a comprehensive a result, the same model is often compute-bound in prefill quantization suite to support accurate low-bit deployment. Using but memory-/dependency-bound in decode, creating conflicting arXiv:2601.15710v1 [cs.AR] 22 Jan 2026 FlexLLM, we build a complete inference system for the Llama- optimization goals within one serving pipeline. 3.2 1B model in under two months with only 1K lines of code. Existing FPGA accelerators primarily follow either temporal The system includes: (1) a stage-customized accelerator with hardware-efficient quantization (12.68 WikiText-2 PPL) surpassing architectures [8], [20] or spatial architectures [21], [22], SpinQuant baseline, and (2) a Hierarchical Memory Transformer [23], [24]. Temporal des
```

## 7. 실험 결과
- keyword `speedup`:
```text
2 PPL) surpassing architectures [8], [20] or spatial architectures [21], [22], SpinQuant baseline, and (2) a Hierarchical Memory Transformer [23], [24]. Temporal designs reuse shared compute engines (HMT) plug-in for efficient long-context processing. On the AMD across layers, but incur frequent off-chip traffic in prefill due U280 FPGA at 16nm, the accelerator achieves 1.29× end-to- to limited buffering and struggle to support heterogeneous end speedup, 1.64× higher decode throughput, and 3.14× better kernels/precisions within a single engine (Fig. 1(b)(c)). Spatial energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm designs map kernels to dedicated modules and stream inter- reach 4.71×, 6.55×, and 4.13×, respectively. In long-context mediate data through on-chip FIFOs, but suffer from pipeline scenarios, integrating the HMT p
```
- keyword `energy`:
```text
mory Transformer [23], [24]. Temporal designs reuse shared compute engines (HMT) plug-in for efficient long-context processing. On the AMD across layers, but incur frequent off-chip traffic in prefill due U280 FPGA at 16nm, the accelerator achieves 1.29× end-to- to limited buffering and struggle to support heterogeneous end speedup, 1.64× higher decode throughput, and 3.14× better kernels/precisions within a single engine (Fig. 1(b)(c)). Spatial energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm designs map kernels to dedicated modules and stream inter- reach 4.71×, 6.55×, and 4.13×, respectively. In long-context mediate data through on-chip FIFOs, but suffer from pipeline scenarios, integrating the HMT plug-in reduces prefill latency stalls when kernel latencies are unbalanced or when intrinsic by 23.23× and extends the conte
```
- keyword `accuracy`:
```text
ill and breakthroughs across natural language processing [1], [2], code decode must be stage-customized, with each stage adopting generation [3], [4], and multimodal understanding [5], [6]. a different mix of spatial parallelism and temporal reuse to While cloud providers continue investing heavily in GPU- match its distinct compute/memory constraints. based infrastructures to support large-scale inference [7], there Challenge 2: Balancing model accuracy and low-bit is a growing demand for domain-specific accelerators that compute/memory efficiency. LLMs have enormous parameter provide customized, efficient, and scalable alternatives across counts, making compression indispensable for accelerator deployment environments—from data centers [8], [9] to edge deployment [9], [20]. Quantization is an effective way to platforms [10], [11]. reduce compute and memory cost; however, prior FPGA LLM
```
- keyword `throughput`:
```text
es [8], [20] or spatial architectures [21], [22], SpinQuant baseline, and (2) a Hierarchical Memory Transformer [23], [24]. Temporal designs reuse shared compute engines (HMT) plug-in for efficient long-context processing. On the AMD across layers, but incur frequent off-chip traffic in prefill due U280 FPGA at 16nm, the accelerator achieves 1.29× end-to- to limited buffering and struggle to support heterogeneous end speedup, 1.64× higher decode throughput, and 3.14× better kernels/precisions within a single engine (Fig. 1(b)(c)). Spatial energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm designs map kernels to dedicated modules and stream inter- reach 4.71×, 6.55×, and 4.13×, respectively. In long-context mediate data through on-chip FIFOs, but suffer from pipeline scenarios, integrating the HMT plug-in reduces prefill latenc
```
- keyword `latency`:
```text
ecode throughput, and 3.14× better kernels/precisions within a single engine (Fig. 1(b)(c)). Spatial energy efficiency than an NVIDIA A100 GPU (7nm) running BF16 inference; projected results on the V80 FPGA at 7nm designs map kernels to dedicated modules and stream inter- reach 4.71×, 6.55×, and 4.13×, respectively. In long-context mediate data through on-chip FIFOs, but suffer from pipeline scenarios, integrating the HMT plug-in reduces prefill latency stalls when kernel latencies are unbalanced or when intrinsic by 23.23× and extends the context window by 64×, delivering dependencies dominate (Fig. 1(d)(e)). Despite their differences, 1.10×/4.86× lower end-to-end latency and 5.21×/6.27× higher both paradigms typically share the same implicit choice: using energy efficiency on the U280/V80 compared to the A100 baseline. FlexLLM thus bridges algorithmic innovation in LLM inference a sing
```

## 8. 실험 옵션 / Ablation 축
- bit width: W4/A8, W6/A8, W8/A8
- scale policy: power-of-two vs learned/floating scale
- per-tensor vs per-channel/group quantization

## 9. HGTXR 적용 판단
- 모델보다는 실험 생성/manifest/tooling 참조로 유용하다.

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
