# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | An Integer-Only and Group-Vector Systolic Accelerator for Efficiently Mapping Vision Transformer on Edge |
| Authors | Mingqiang Huang, Junyi Luo, Chenchen Ding, Zikun Wei, Sixiao Huang, Hao Yu |
| Venue | IEEE Transactions on Circuits and Systems I: Regular Papers |
| Year | 2023 |
| Research Field | Integer-only ViT inference, FPGA accelerator, systolic array |
| Paper Link / DOI | 10.1109/TCSI.2023.3312775 |
| Code / Project Link | 논문에 명시되지 않음 |
| Analysis Coverage | Main paper 전체 13 pages. FPGA bitstream/RTL 재합성 및 보드 실행은 수행하지 않음. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** ViT를 edge FPGA에 배치할 때 LayerNorm/Softmax/GELU 같은 nonlinear floating-point operation과 Multi-Head Self-Attention (MSA)의 irregular memory access가 전체 성능과 자원 효율을 제한하는 문제.
- **Research Objective:** ViT 전체 inference를 INT8 integer-only arithmetic과 bit-shift 중심으로 실행하고, regular MatMul/Conv와 irregular MSA를 하나의 group-vector systolic accelerator에서 효율적으로 처리하는 것.
- **Why This Problem Matters:** ViT는 CNN보다 attention과 normalization 등 non-linear/data-dependent operation이 많고, Q/K/V가 online generated tensor라 matrix transpose/reshape/data access가 복잡하다. Edge FPGA에서는 division, exp, sqrt, DRAM access가 핵심 병목이다.

## 1.2 Core Idea

- **Core Idea:** Integer-only LayerNorm/Softmax/GELU hardware를 설계하고, channel-group 중심 unified data package와 group-vector systolic array를 사용하여 matrix computation과 self-attention을 통합 가속한다. On-chip BRAM feature storage와 flexible memory management로 operation 간 data movement를 줄인다.
- **Proposed Approach:** Integer-only LayerNorm/Softmax/GELU hardware를 설계하고, channel-group 중심 unified data package와 group-vector systolic array를 사용하여 matrix computation과 self-attention을 통합 가속한다. On-chip BRAM feature storage와 flexible memory management로 operation 간 data movement를 줄인다.
- **Key Differentiator:** I-ViT-style integer-only quantization을 FPGA hardware 관점에서 더 효율적인 nonlinear datapath로 재구성하고, ViT의 high-dimensional tensor를 unified package로 표현하여 reshape overhead를 줄인다.

## 1.3 Main Contributions

1. ViT 전체 network를 floating-point 없이 INT8 integer-only arithmetic으로 수행하는 quantization/hardware scheme을 제안하였다.
2. Regular matrix multiplication/convolution과 irregular MSA를 모두 지원하는 unified data package scheme과 group-vector systolic array를 제안하였다.
3. LayerNorm, Softmax, GELU에 대한 integer-only hardware approximation을 설계하고 오류 분석을 제시하였다.
4. On-chip/off-chip storage management를 설계하여 feature memory access를 줄였다.
5. Xilinx ZCU102에서 ViT-tiny와 ViT-small을 end-to-end 실행하고 latency/throughput/resource를 보고하였다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| ZCU102 / ViT-small | ImageNet classification | Latency / Throughput / Accuracy | 11.14~11.15 ms, 762.7 GOP/s, Top-1 81.27% | Prior FPGA Transformer accelerators | Higher throughput | Table II/III |
| ZCU102 / ViT-tiny | ImageNet classification | Latency / Throughput / Accuracy | 4.077 ms, 616.14 GOP/s, Top-1 about 73% | Prior FPGA accelerators | 245.3 fps | Table II |
| Integer-only LayerNorm | Operator approximation | Error | 95.1% no error in Monte-Carlo test | FP reference | Low absolute error | Figure 4 |
| Integer-only Softmax | Operator approximation | Error | weighted avg abs error 1.9, relative error 1.48% | FP reference | Division-reduced approximation | Figure 6 |
| Integer-only GELU | Operator approximation | Error | 86.7% no error | FP reference | Low error | Figure 7 |
| ZCU102 | Power | Board power | Idle ~27W, acceleration 29.6W for ViT-s | Board-level measurement | About +2.6W active | Section V |

## 1.5 Brief Overview

> 이 논문은 ViT inference를 edge FPGA에서 INT8 integer-only로 수행하기 위한 algorithm-hardware co-design이다. Nonlinear functions는 integer-only approximation으로 구현하고, feature/weight tensor는 unified data package로 표현하여 reshape/transposition overhead를 줄인다. ZCU102 실험에서 ViT-small 11.14ms, 762.7 GOP/s와 ViT-tiny 4.077ms를 보고한다.

---

# 2. Research Background

## 2.1 Research Context

- ViT는 patch embedding, MSA, MLP, LayerNorm, Softmax, GELU, reshape/transpose 등 다양한 operation을 포함한다.
- 기존 CNN accelerator의 loop/dataflow 최적화는 fixed weight convolution에는 잘 맞지만, MSA에서는 K/V가 online generated activation이므로 weight처럼 offline reorder하기 어렵다.
- LayerNorm은 inference 중 mean/variance를 동적으로 계산해야 하며, Softmax는 exp/division overflow와 division cost를 유발한다.
- FPGA에서는 division/sqrt/exp의 latency와 resource가 높으므로 integer approximation 및 operation grouping이 필수적이다.

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
| ViT quantization | I-ViT, PoT/mixed quantization | Nonlinear ops를 integer-friendly하게 근사 | GPU 중심 또는 accuracy loss/mixed training complexity | 본 논문은 FPGA hardware datapath까지 구현 |
| FPGA Transformer accelerators | FTRANS, ViA, Auto-ViT-Acc, HeatViT | Transformer layer 또는 ViT를 FPGA에 mapping | 일부 operation만 최적화하거나 pruning/PoT에 의존 | 본 논문은 full INT8 and unified data package를 강조 |
| Systolic accelerators | TPU, vector systolic array | Matrix multiplication을 high throughput으로 처리 | Register/power overhead 또는 fanout 문제 | Group-vector systolic array로 register cost와 parallelism trade-off를 조정 |
| Integer non-linear approximation | I-BERT/I-ViT-style methods | LayerNorm/Softmax/GELU를 integer arithmetic으로 근사 | Division 또는 operator-specific overhead | Softmax division을 logarithm/exponential approximation으로 줄임 |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 Integer-only LayerNorm

- **Definition:** Mean과 variance를 integer accumulation으로 계산하고 std inverse를 integer divider/sqrt로 처리하는 LayerNorm hardware.
- **Role in This Paper:** Dynamic normalization을 FP 없이 수행한다.
- **Why It Is Required:** Dynamic normalization을 FP 없이 수행한다.
- **Related Components / Source:** Figure 3, Equations (1)(2)
### 4.2 Integer-only Softmax

- **Definition:** Max subtraction 후 exp/log approximation을 bit-shift/multiply/add 중심 linear functions로 처리하는 Softmax.
- **Role in This Paper:** Division과 overflow를 줄인다.
- **Why It Is Required:** Division과 overflow를 줄인다.
- **Related Components / Source:** Figure 5, Equations (3)-(7)
### 4.3 Integer-only GELU

- **Definition:** Second-order polynomial approximation L(x)를 이용한 GELU integer implementation.
- **Role in This Paper:** GELU를 matrix computation datapath에 삽입해 별도 data dependency 없이 처리한다.
- **Why It Is Required:** GELU를 matrix computation datapath에 삽입해 별도 data dependency 없이 처리한다.
- **Related Components / Source:** Figure 7, Equations (8)(9)
### 4.4 Unified Data Package

- **Definition:** Feature를 `(CH/Tout, H, W, Tout)` 또는 MSA에서는 `(head, CH/Tout, H, W, Tout)`로 통일하는 packaging.
- **Role in This Paper:** Operation 간 reshape/rearrange를 줄이고 input/output structure를 동일하게 유지한다.
- **Why It Is Required:** Operation 간 reshape/rearrange를 줄이고 input/output structure를 동일하게 유지한다.
- **Related Components / Source:** Figure 8, 13
### 4.5 Group-vector Systolic Array

- **Definition:** Feature row를 여러 PE rows로 전달하는 vector systolic variant.
- **Role in This Paper:** FF/register cost와 parallelism/PPA trade-off를 개선한다.
- **Why It Is Required:** FF/register cost와 parallelism/PPA trade-off를 개선한다.
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

1. 입력 모델은 INT8 quantized ViT-tiny/ViT-small이다. Hardware input/output은 unified packaged feature/weight tensors이며, accelerator는 task-by-task로 suboperation을 수행한다.
2. LayerNorm은 `Var(x)=E(x^2)-E(x)^2`를 사용하여 feature를 두 번 읽지 않고 mean/variance를 동시에 계산한다. Channel parallelism Tout=32 기준으로 192 channels는 6 cycles에 stat computation이 가능하다.
3. Softmax는 `Softmax(x_i)=exp(x_i-Xmax-ln(sum_j exp(x_j-Xmax)))`로 변환하고 exp/log를 linear approximation으로 구현한다. Temporary BUF와 DMA를 사용하여 channel-group feature를 pixel-group으로 재배열한다.
4. GELU는 sigmoid 기반 대신 polynomial L(x) approximation을 사용하고, MatMul with GELU datapath에 삽입한다.
5. Unified data package는 모든 operation의 input/output data layout을 동일하게 유지한다. Weight는 `(CHout/Tout, CHin/Tin, Ky, Kx, Tout, Tin)`으로 pretreat/reorder된다.
6. MSA에서는 Q/K/V generation 후 multi-head reshape를 package layout과 맞추어 사실상 무시할 수 있게 하고, K transpose는 package address reorder 중심으로 구현한다.
7. Memory optimization은 on-chip BRAM에 feature를 저장하여 LayerNorm/Softmax/MatMul operation latency를 줄인다. 논문은 on-chip BRAM 사용 시 약 60% latency 감소를 보고한다.

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

- Platform: Xilinx ZCU102 FPGA, Vivado 2018.1.
- Resources: ZCU102 has 274K LUT, 548K FF, 2520 DSP, 912 BRAM.
- Parallelism: Tin=64, Tout=32 in reported configuration.
- Frequency: up to 300 MHz.
- Models: ViT-small, ViT-tiny.
- Metrics: Top-1 accuracy, latency, fps, throughput, power, resource utilization.
- Comparison: Prior FPGA Transformer/ViT accelerators including pruning, PoT quantization, ViA, Auto-ViT-Acc, NMT accelerators.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| ZCU102 / ViT-small | ImageNet classification | Latency / Throughput / Accuracy | 11.14~11.15 ms, 762.7 GOP/s, Top-1 81.27% | Prior FPGA Transformer accelerators | Higher throughput | Table II/III |
| ZCU102 / ViT-tiny | ImageNet classification | Latency / Throughput / Accuracy | 4.077 ms, 616.14 GOP/s, Top-1 about 73% | Prior FPGA accelerators | 245.3 fps | Table II |
| Integer-only LayerNorm | Operator approximation | Error | 95.1% no error in Monte-Carlo test | FP reference | Low absolute error | Figure 4 |
| Integer-only Softmax | Operator approximation | Error | weighted avg abs error 1.9, relative error 1.48% | FP reference | Division-reduced approximation | Figure 6 |
| Integer-only GELU | Operator approximation | Error | 86.7% no error | FP reference | Low error | Figure 7 |
| ZCU102 | Power | Board power | Idle ~27W, acceleration 29.6W for ViT-s | Board-level measurement | About +2.6W active | Section V |

## 6.3 Result Interpretation

- LayerNorm과 Softmax의 approximation error analysis는 nonlinear integer hardware가 accuracy loss 없이 사용 가능함을 보조한다. Softmax relative error가 LayerNorm보다 크지만 network-level accuracy는 유지된다.
- ViT-small에서 21.4M parameters, 81.27% Top-1 accuracy, 11.14ms latency는 full-network FPGA result로 의미가 있다.
- ViT-tiny는 6.1M model size와 4.077ms latency로 edge deployment 가능성을 보여준다.
- Table III comparison에서 Auto-ViT-Acc는 더 높은 throughput을 보이나 mixed PoT quantization accuracy/training complexity 문제가 언급된다. 직접 비교에는 quantization scheme과 model accuracy 조건 차이를 주의해야 한다.
- Data access가 matrix computation latency를 지배한다는 논문 분석은 향후 weight prefetch, compression, double buffering, layer fusion 연구 포인트를 제공한다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

이 논문은 ViT inference를 edge FPGA에서 INT8 integer-only로 수행하기 위한 algorithm-hardware co-design이다. Nonlinear functions는 integer-only approximation으로 구현하고, feature/weight tensor는 unified data package로 표현하여 reshape/transposition overhead를 줄인다. ZCU102 실험에서 ViT-small 11.14ms, 762.7 GOP/s와 ViT-tiny 4.077ms를 보고한다.

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

- Nonlinear operator, tensor layout, systolic array, memory management를 모두 포함한 end-to-end FPGA 설계다.
- LayerNorm/Softmax/GELU에 대한 hardware-level bitwidth/error 분석이 구체적이다.
- ViT-tiny와 ViT-small 두 모델의 full-network latency/throughput 결과를 제시한다.
- Unified data package가 MSA reshape/transpose 문제를 실용적으로 다룬다.
- ZCU102 보드 기반 구현 결과가 있어 실험 신뢰도가 높다.

---

# 9. Cons.

- Board-level power가 idle power를 포함하므로 accelerator core energy efficiency만 분리하기 어렵다.
- INT8 quantization training 절차와 accuracy recovery recipe가 hardware 설명에 비해 제한적으로 제공된다.
- Softmax가 여전히 세 번의 data fetching을 요구하며, attention sequence length가 커질 경우 scalability가 문제될 수 있다.
- ViT-small/tiny classification 중심으로 검증되어 detection/segmentation/higher-resolution ViT에는 추가 검증 필요.
- AXI/SDRAM bandwidth, DMA overlap, task scheduling에 대한 cycle-level 공개 자료가 제한적이다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
