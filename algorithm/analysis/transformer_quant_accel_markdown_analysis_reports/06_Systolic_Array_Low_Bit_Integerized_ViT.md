# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | Systolic Array-based Architecture for Low-Bit Integerized Vision Transformers |
| Authors | Ching-Yi Lin, Sahil Shah |
| Venue | arXiv:2508.20334v1 |
| Year | 2025 |
| Research Field | Low-bit integerized ViT, FPGA accelerator, systolic array, MSA acceleration |
| Paper Link / DOI | 논문 PDF 기준 DOI 명시 없음 |
| Code / Project Link | 논문에 명시되지 않음 |
| Analysis Coverage | Main paper 전체 14 pages. FPGA synthesis 결과는 논문 수치 분석만 수행하고 Vivado 재실행은 수행하지 않음. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** Modern inference service에서는 throughput과 power efficiency가 중요하지만, GPU는 flexibility 때문에 low operational intensity workload에서 비효율적이다. Low-bit quantized ViT도 dequantized operands를 사용하면 실제 compute-intensive modules가 high precision으로 남는 문제가 있다.
- **Research Objective:** 3-bit integerized DeiT/ViT MSA workload를 대상으로, low communication overhead와 high OP reuse를 갖는 systolic array-based accelerator를 설계하여 low-bandwidth 환경에서도 높은 throughput/power efficiency를 달성하는 것.
- **Why This Problem Matters:** 서비스형 inference는 같은 model에 대해 많은 query가 들어오므로 model-specialized accelerator가 비용 효율적일 수 있다. Low-bit arithmetic은 compute density를 높이지만, systolic 구조가 아니면 bandwidth와 communication이 병목이 된다.

## 1.2 Core Idea

- **Core Idea:** Q-ViT-style low-bit quantized model을 operand reordering으로 integerized하여 compute-intensive Linear/MatMul을 3-bit integer로 실행한다. Hardware는 2D MAC array, post-MAC array, systolic aggregation, triangular delay, post-aggregation array를 갖는 systolic template으로 Softmax/LayerNorm/quantization까지 처리한다.
- **Proposed Approach:** Q-ViT-style low-bit quantized model을 operand reordering으로 integerized하여 compute-intensive Linear/MatMul을 3-bit integer로 실행한다. Hardware는 2D MAC array, post-MAC array, systolic aggregation, triangular delay, post-aggregation array를 갖는 systolic template으로 Softmax/LayerNorm/quantization까지 처리한다.
- **Key Differentiator:** 단순 quantized model compression이 아니라 dequantized operand를 제거하고, MSA의 Q/K/V, attention score, weighted sum을 low-bit systolic pipeline으로 mapping한다. Communication cut point를 선택하여 host-FPGA traffic을 28kB 수준으로 줄이는 operational intensity 최적화가 중심이다.

## 1.3 Main Contributions

1. Low-bit MSA를 위한 systolic array-based architecture를 제안하고 low-bandwidth constraint에서 high throughput을 목표로 한다.
2. Softmax와 LayerNorm 같은 aggregate operations를 systolic array-compatible units로 설계하였다.
3. Low-bit arithmetic과 accelerator pipelining을 통해 power efficiency를 높였다.
4. Power/area breakdown을 통해 MSA accelerator에서 MAC operations가 지배적임을 분석하였다.
5. Alveo U250 FPGA에서 3-bit integerized model의 throughput과 power efficiency를 검증하였다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| CIFAR-10 | Image classification | Accuracy | 96.83% | 3-bit integerized model | High accuracy | Abstract |
| ImageNet / DeiT-S | Image classification | Top-1 Acc | 77.81% | Full/quantized baselines | Low-bit viable | Abstract/Conclusion |
| Alveo U250 16nm | MSA acceleration | Throughput | 13,568 GOP/s | Same-technology GPU GTX 1080 / FPGA works | 1.50× GTX1080 throughput | Abstract/Table III |
| Alveo U250 16nm | Power efficiency | GOP/s/W | 219.4 GOP/s/W | GTX1080 and RTX5090 comparisons | 4.47× GTX1080 efficiency, 20% better than RTX5090 efficiency | Abstract |
| Full DeiT latency estimate | End-to-end estimate | Latency | DeiT-T 324 µs, DeiT-S 661 µs, DeiT-B 1741 µs | FP8/FP5 works | Lower latency estimates | Table IV/V |
| U250 | Roofline metric | OP/Byte | 4342 OP/Byte | SOTA FPGA works | 28× higher than existing SOTA claimed | Table III |

## 1.5 Brief Overview

> 이 논문은 3-bit integerized Vision Transformer의 MSA를 고효율 systolic accelerator로 실행하는 연구다. 핵심은 dequantized operand를 제거하여 Linear/MatMul을 low-bit integer로 수행하고, Softmax/LayerNorm 같은 aggregate operation도 systolic-compatible structure로 넣는 것이다. Alveo U250에서 13,568 GOP/s와 219.4 GOP/s/W를 보고하며, operational intensity를 크게 높였다는 점이 특징이다.

---

# 2. Research Background

## 2.1 Research Context

- LLM/Transformer inference 서비스는 token 처리량과 전력 비용이 직접 비용으로 연결된다.
- GPU는 다양한 model을 지원하지만, model-specific reuse와 low-bit specialized datapath 측면에서 효율이 낮을 수 있다.
- Quantized ViT는 storage를 줄여도 compute-intensive Linear/MatMul이 dequantized FP operand를 쓰면 hardware benefit이 제한된다.
- Systolic array는 local interconnect와 regular dataflow로 높은 operational intensity를 달성할 수 있지만, Softmax/LayerNorm 같은 aggregation operation을 어떻게 포함할지가 문제다.

## 2.2 Conventional Approaches

- **일반적인 quantization/acceleration 접근:** MatMul/GEMM을 low precision으로 낮추거나, sparse attention/pruning을 적용하거나, FPGA/ASIC/GPU kernel을 특화하는 방식이 주류이다.
- **Non-linear operation 처리:** 많은 연구가 Softmax, LayerNorm, GELU 등을 approximation, LUT, polynomial, shift, or partial FP fallback으로 처리한다.
- **Hardware mapping:** Systolic array, tensor core, CUDA/Triton kernel, FPGA RTL/VHDL/Verilog, cycle-accurate simulator 등 다양한 구현 계층이 사용된다.

## 2.3 Limitations of Existing Approaches

- Compute precision을 낮춰도 data flow precision, scale management, off-chip memory movement가 남으면 end-to-end acceleration이 제한된다.
- Transformer 계열은 activation outlier, dynamic normalization, Softmax, attention reshape/transpose 때문에 CNN보다 integer-only deployment가 어렵다.
- GPU 중심 방법은 tensor core와 kernel library에 의존하고, FPGA/ASIC 중심 방법은 task/model-specific datapath와 memory layout 최적화가 필요하다.

## 2.4 Research Motivation

- 저자들은 Transformer의 비선형 연산, attention dataflow, quantization scale, memory access 중 하나 또는 복수의 병목을 직접 해결하기 위해 본 연구를 수행하였다.
- 분석 관점에서 이 논문은 “Transformer acceleration에서 어디를 low-bit로 만들 것인가”뿐 아니라 “어떤 data가 어느 precision으로 이동하고 저장되는가”를 핵심 문제로 다룬다.

## 2.5 Significance of the Problem

- Edge/XR/AIoT/FPGA/NPU 환경에서는 latency, throughput, energy, area, memory footprint가 동시에 중요하다.
- Transformer는 다양한 application backbone으로 확산되고 있으므로, integer-only 및 hardware-aware acceleration은 모델 배포 가능성을 직접 좌우한다.

---

# 3. Related Works

## 3.1 Related Work Categories

| Category | Representative Work | Core Idea | Limitation | Difference from This Paper |
|---|---|---|---|---|
| Low-bit quantization | DoReFa, PACT, Q-ViT, FQ-ViT | Weight/activation bitwidth 감소 | Dequantized operand 때문에 compute가 high precision으로 남을 수 있음 | 본 논문은 integerized compute graph로 low-bit Linear/MatMul 수행 |
| Integerized Transformer | I-BERT, I-ViT, Huang et al. | Integer Softmax/GELU/LayerNorm | 주로 8-bit 또는 operator-specific | 본 논문은 3-bit MSA systolic architecture에 초점 |
| Systolic accelerators | TPU, Calabash, ME-ViT, HG-PIPE | Regular matrix computation과 data reuse | Aggregation/communication bottleneck | Systolic aggregate units와 accelerator pipelining 제공 |
| FPGA Transformer accelerators | FTRANS, HeatViT, Auto-ViT-acc, HG-PIPE | FPGA-based Transformer/ViT acceleration | Bandwidth 또는 precision trade-off | 3-bit integerized MSA와 높은 OP/Byte를 강조 |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 Low-bit Integerization

- **Definition:** Quantized values와 step-size multiplication을 재배열하여 compute-intensive module이 low-bit integer input을 직접 받도록 하는 방식.
- **Role in This Paper:** Linear/MatMul을 3-bit integer arithmetic으로 실행하게 한다.
- **Why It Is Required:** Linear/MatMul을 3-bit integer arithmetic으로 실행하게 한다.
- **Related Components / Source:** Figure 3, Equations (7)-(9)
### 4.2 SA Offloading Cut Point

- **Definition:** Host와 FPGA 사이에 전달할 intermediate를 선택하여 input/output size를 최소화하는 전략.
- **Role in This Paper:** Quantized z와 SA3b(z)를 주고받아 communication을 줄인다.
- **Why It Is Required:** Quantized z와 SA3b(z)를 주고받아 communication을 줄인다.
- **Related Components / Source:** Figure 4
### 4.3 Systolic Array Template

- **Definition:** 2D MAC, post-MAC, aggregation, triangular delay, post-aggregation으로 구성된 template.
- **Role in This Paper:** Matrix multiplication과 aggregation operation을 local interconnect로 처리한다.
- **Why It Is Required:** Matrix multiplication과 aggregation operation을 local interconnect로 처리한다.
- **Related Components / Source:** Figure 6
### 4.4 Triangular Delay

- **Definition:** Aggregation result와 post-aggregation input timing을 맞추기 위한 delay structure.
- **Role in This Paper:** Broadcast 없이 systolic-compatible timing alignment를 제공한다.
- **Why It Is Required:** Broadcast 없이 systolic-compatible timing alignment를 제공한다.
- **Related Components / Source:** Figure 7
### 4.5 NormQ

- **Definition:** Division-free, square-root-free LayerNorm + quantization module.
- **Role in This Paper:** LayerNorm을 comparator와 precomputed parameters 중심으로 변환한다.
- **Why It Is Required:** LayerNorm을 comparator와 precomputed parameters 중심으로 변환한다.
- **Related Components / Source:** Figure 10


## 4.6 Concept Relationships

```text
Transformer deployment bottleneck
   ↓
Quantization / sparsity / operator approximation / dataflow optimization
   ↓
Paper-specific core method
   ↓
Accuracy-efficiency trade-off evaluation
   ↓
Hardware feasibility or system-level speed/memory/energy result
```

---

# 5. Methodology

## 5.1 Overall Structure

1. Target model은 DeiT이며 accelerator focus는 MSA module의 QKV projection과 scaled dot-product attention이다.
2. Quantized model의 dequantization step-size를 channel-wise에서 global/shared scale로 단순화하여 `z3b · U3b` 형태의 integerized MatMul을 가능하게 한다.
3. SA offloading에서는 host가 full-precision z를 보관하고 quantized z3b만 FPGA로 전달한다. FPGA는 Q/K/V, attention A, AV를 계산한 뒤 low-bit SA result를 host로 반환한다.
4. Hardware dataflow는 Q, K, V를 각각 systolic arrays로 생성하고, attention score와 weighted value product를 추가 arrays에서 계산한다.
5. Systolic template은 MAC output을 post-MAC units로 처리한 뒤, row-wise aggregation을 systolic manner로 수행하고, triangular delay를 통해 post-aggregation units에 timing-align된 결과를 전달한다.
6. Softmax는 exponential approximation, systolic sum, dynamic-scaled quantizer로 구성된다. Exponential은 `e^x = 2^{x log2 e}` 분해와 first-order approximation을 사용한다.
7. LayerNorm은 variance의 sqrt/division을 직접 계산하지 않고 comparator form으로 재작성한다. Mean/variance는 Welford’s algorithm으로 online systolic computation한다.
8. Accelerator pipelining은 여러 SA heads를 dedicated hardware로 완전 unroll하지 않고, 단일 hardware가 head parameter를 순차 교체하여 처리한다. 이는 throughput을 H배 낮추지만 resource와 bandwidth를 줄인다.

## 5.2 Input and Output

- **Input:** 논문별로 pretrained Transformer/ViT model, quantized tensors, activation/weight/gradient, time-series sequence, attention map, 또는 hardware configuration이 입력으로 사용된다.
- **Output:** 논문별로 approximated non-linear function, INT8 training/inference tensor flow, sparse attention execution plan, FPGA accelerator output, 또는 forecasting/classification/translation output이 생성된다.

## 5.3 Core Modules

- Quantization module 또는 scale management.
- Non-linear function approximation module.
- Attention or MatMul acceleration module.
- Hardware dataflow, memory hierarchy, or kernel implementation.
- Training/fine-tuning or QAT procedure.

## 5.4 Forward / Inference Process

- 입력 tensor는 low-bit integer 또는 quantized representation으로 변환된다.
- 논문별 핵심 module이 MatMul, attention, normalization, Softmax, activation, or forecasting block을 처리한다.
- 필요 시 scale, zero point, LUT, approximation parameter, mask, index, or hardware schedule이 함께 전달된다.
- 출력은 다시 다음 module의 low-bit representation 또는 task-level prediction으로 전달된다.

## 5.5 Loss and Optimization

- 논문이 별도 loss를 제시한 경우 해당 loss를 사용한다.
- QAT 또는 fine-tuning을 수행하는 논문은 task loss와 quantization-aware parameter update를 결합한다.
- Hardware-only evaluation 중심 논문은 training objective보다 dataflow/resource/timing optimization이 중심이다.
- 제시되지 않은 세부 loss term은 `논문에 명시되지 않음`.

## 5.6 Training Procedure

- Training/fine-tuning이 필요한 경우 pretrained model initialization, QAT, or architecture modification 후 fine-tuning을 수행한다.
- Hardware implementation 논문은 training recipe가 제한적으로 제시되거나 기존 quantized checkpoint를 사용한다.
- 코드 실행 및 재현 학습은 본 분석에서 수행하지 않았다.

## 5.7 Inference Procedure

- 논문별 method는 FP fallback 최소화, INT8/low-bit data movement, sparse attention mapping, or FPGA accelerator execution으로 inference path를 구성한다.
- FPGA/ASIC 논문은 DMA, buffer, systolic array, LUT, divider, Softmax/LayerNorm units 등 hardware stage를 거쳐 output을 만든다.

## 5.8 Computational Characteristics

- 주요 지표는 accuracy/BLEU/PPL/RMSE/mIoU, latency, throughput, speedup, memory reduction, area, power, energy, resource utilization이다.
- 결과 수치는 Section 6의 tables에 정리하였다.
- Operation 범위가 full-network인지 module-level인지 반드시 구분해야 한다.

---

# 6. Experiments Results

## 6.1 Experimental Setup

- Model: 3-bit integerized DeiT-S 중심, CIFAR-10/ImageNet evaluation.
- Training: pretrained quantized model checkpoint를 initialization으로 사용, 20 epochs fine-tuning, LR 1e-5, Adam.
- Hardware: AMD Alveo U250 Data Center Accelerator FPGA, 16nm FinFET, Vivado 2024.2.
- Reported metrics: latency, throughput, power, area/resource, OP/Byte, accuracy.
- Communication: PCIe-like low-bandwidth setting, reported read/write bandwidth 3.13/3.13 GB/s.
- Comparisons: FPGA works and GPUs including GTX1080/RTX5090/H100/A100, normalized by technology scaling in additional analysis.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| CIFAR-10 | Image classification | Accuracy | 96.83% | 3-bit integerized model | High accuracy | Abstract |
| ImageNet / DeiT-S | Image classification | Top-1 Acc | 77.81% | Full/quantized baselines | Low-bit viable | Abstract/Conclusion |
| Alveo U250 16nm | MSA acceleration | Throughput | 13,568 GOP/s | Same-technology GPU GTX 1080 / FPGA works | 1.50× GTX1080 throughput | Abstract/Table III |
| Alveo U250 16nm | Power efficiency | GOP/s/W | 219.4 GOP/s/W | GTX1080 and RTX5090 comparisons | 4.47× GTX1080 efficiency, 20% better than RTX5090 efficiency | Abstract |
| Full DeiT latency estimate | End-to-end estimate | Latency | DeiT-T 324 µs, DeiT-S 661 µs, DeiT-B 1741 µs | FP8/FP5 works | Lower latency estimates | Table IV/V |
| U250 | Roofline metric | OP/Byte | 4342 OP/Byte | SOTA FPGA works | 28× higher than existing SOTA claimed | Table III |

## 6.3 Result Interpretation

- Figure 1 roofline은 많은 FPGA designs가 low operational intensity로 bandwidth에 제한되는 반면, 본 논문은 OP/Byte를 크게 높여 bandwidth demand를 줄이는 방향임을 보여준다.
- Area/power breakdown에서 MAC array가 지배적인 것은 low-bit MAC density가 전체 accelerator efficiency를 좌우함을 의미한다. Softmax/NormQ는 PE당 cost가 크지만 PE 수가 적어 총비중은 제한적이다.
- Table III의 13,568 GOP/s와 219.4 GOP/s/W는 MSA accelerator 기준이며 full model end-to-end silicon measurement가 아니라는 점에 주의해야 한다.
- Full DeiT latency는 추정이며 MLP accelerator를 별도 가정한다. 따라서 real full-system latency는 host-FPGA transfer, MLP implementation, scheduling에 따라 달라질 수 있다.
- DSP-free design은 frequency를 높일 수 있으나 LUT arithmetic power가 증가하여 power efficiency가 낮아진다. 이는 high-throughput vs energy-efficient design point를 구분한다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

이 논문은 3-bit integerized Vision Transformer의 MSA를 고효율 systolic accelerator로 실행하는 연구다. 핵심은 dequantized operand를 제거하여 Linear/MatMul을 low-bit integer로 수행하고, Softmax/LayerNorm 같은 aggregate operation도 systolic-compatible structure로 넣는 것이다. Alveo U250에서 13,568 GOP/s와 219.4 GOP/s/W를 보고하며, operational intensity를 크게 높였다는 점이 특징이다.

## 7.2 Key Takeaways

1. 이 논문은 Transformer 계열 모델의 특정 배포 병목을 명확히 겨냥한다.
2. 제안 방법은 algorithm-level modification과 hardware/runtime-level optimization 중 하나 또는 둘을 결합한다.
3. 결과는 accuracy 유지와 efficiency 개선을 보여주지만, target model/hardware/application 조건에 따라 직접 비교에는 주의가 필요하다.
4. 연구 재사용 관점에서는 quantization-aware approximation, INT8 dataflow, systolic mapping, attention sparsity, FPGA template generation 등으로 확장 가능하다.

## 7.3 Relevance to Hardware-aware AI Research

- FPGA/NPU/ASIC 가속기 설계에서는 operator-level quantization만이 아니라 memory layout, scale handling, data movement, and end-to-end scheduling이 중요하다.
- 본 논문은 accelerator DSE, integer-only datapath, non-linear unit approximation, and Transformer-specific compiler/runtime 설계에 참고할 수 있다.

---

# 8. Pros.

- Quantization을 storage reduction이 아니라 실제 low-bit compute graph로 연결했다.
- Softmax/LayerNorm을 systolic-compatible unit으로 통합하려는 설계가 구체적이다.
- Operational intensity, bandwidth, power/area breakdown까지 분석하여 hardware 논문으로 완성도가 높다.
- 3-bit integerized model의 accuracy-power trade-off를 Pareto frontier 관점에서 제시한다.
- Accelerator pipelining으로 resource/bandwidth/throughput trade-off를 명확히 설명한다.

---

# 9. Cons.

- 가속 대상이 MSA 중심이며 full DeiT end-to-end 결과는 추정에 가깝다.
- U250은 datacenter FPGA이므로 edge FPGA deployment와는 resource/power scale이 다르다.
- 3-bit integerization은 specific DeiT-S checkpoint/fine-tuning recipe에 의존할 가능성이 있다.
- Softmax/LayerNorm approximation의 downstream task 일반성은 CIFAR/ImageNet classification 외 추가 검증이 필요하다.
- Power와 performance는 synthesis report 기반이며 post-route/real board power measurement 상세는 제한적이다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
