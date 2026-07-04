# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers |
| Authors | Pingcheng Dong, Yonghao Tan, Dong Zhang, Tianwei Ni, Xuejiao Liu, Yu Liu, Peng Luo, Luhong Liang, Shih-Yang Liu, Xijie Huang, Huaiyu Zhu, Yun Pan, Fengwei An, Kwang-Ting Cheng |
| Venue | DAC 2024 / arXiv:2403.19591v2 |
| Year | 2024 |
| Research Field | Transformer quantization, non-linear operator approximation, LUT-based hardware co-design |
| Paper Link / DOI | 논문 PDF 내 ACM DOI placeholder로 표시됨; arXiv:2403.19591v2 |
| Code / Project Link | https://github.com/PingchengDong/GQA-LUT |
| Analysis Coverage | Main paper 전체 6 pages. Appendix 없음. 코드 실행 및 RTL 재합성은 수행하지 않음. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** Transformer 계열 모델에서 Softmax, GELU, LayerNorm, HSWISH, DIV, RSQRT 같은 non-linear operation이 FP32/INT32 고정밀 산술 또는 operator-specific dataflow에 의존하여, INT8 integer-only accelerator에서 면적·전력·일반성 병목이 되는 문제.
- **Research Objective:** Non-linear functions를 unified piece-wise linear (PWL) LUT approximation으로 근사하되, INT8 quantization scale과 fixed-point conversion에 의해 발생하는 approximation error를 설계 단계에서 반영하는 것.
- **Why This Problem Matters:** Matrix multiplication은 INT8로 줄이기 쉽지만, Softmax/LayerNorm/GELU 등은 자주 FP32/INT32 fallback이 발생한다. 따라서 end-to-end integer-only inference의 병목은 비선형 연산이다.

## 1.2 Core Idea

- **Core Idea:** Quantization-aware LUT approximation을 위해 GQA-LUT라는 genetic algorithm 기반 breakpoint search를 제안하고, scaling factor가 큰 영역에서 발생하는 breakpoint deviation을 Rounding Mutation (RM)으로 완화한다.
- **Proposed Approach:** Quantization-aware LUT approximation을 위해 GQA-LUT라는 genetic algorithm 기반 breakpoint search를 제안하고, scaling factor가 큰 영역에서 발생하는 breakpoint deviation을 Rounding Mutation (RM)으로 완화한다.
- **Key Differentiator:** NN-LUT처럼 학습 데이터 100K로 neural approximation을 학습하지 않고, 소량의 sampling data와 genetic search로 breakpoints를 직접 탐색한다. 또한 PWL parameter를 INT8/INT16 LUT에 맞게 quantization-aware하게 결정한다.

## 1.3 Main Contributions

1. Scaling factor와 LUT parameter precision 사이의 상호작용을 분석하고, INT8/INT16 LUT storage pattern을 포함한 quantization-aware PWL computing flow를 정식화하였다.
2. Non-linear function별 optimal breakpoints를 자동 결정하는 GQA-LUT genetic algorithm을 제안하였다.
3. 큰 scaling factor에서 breakpoint quantization이 크게 이동하는 breakpoint deviation 현상을 분석하고, fixed-point conversion을 mutation으로 보는 Rounding Mutation (RM)을 제안하였다.
4. Verilog HDL 기반 LUT PWL unit을 TSMC 28-nm에서 합성하여 INT8 PWL의 면적/전력 이득을 제시하였다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| Cityscapes / Segformer-B0 | Semantic segmentation | mIoU | 74.53% with GQA-LUT w/RM, all non-linear replaced | NN-LUT 73.46%, baseline 74.60% | -0.07% vs no replacement, +1.07% vs NN-LUT | Table 4 |
| Cityscapes / EfficientViT-B0 | Semantic segmentation | mIoU | 74.15% with GQA-LUT w/RM, all non-linear replaced | NN-LUT 73.27%, baseline 74.17% | -0.02% vs no replacement, +0.88% vs NN-LUT | Table 5 |
| TSMC 28-nm synthesis | PWL hardware | Area / Power | 8-entry INT8: 961 µm² / 0.40 mW | 8-entry FP32: 5135 µm² / 2.02 mW, INT32: 5243 µm² / 1.93 mW | 81.3~81.7% area saving, 79.3~80.2% power saving | Table 6 |

## 1.5 Brief Overview

> 이 논문은 Transformer의 non-linear operations를 INT8-friendly LUT PWL approximation으로 변환하기 위한 GQA-LUT를 제안한다. 핵심은 quantization scale이 LUT breakpoints/intercepts에 미치는 영향을 직접 최적화에 반영하고, large scale에서 발생하는 breakpoint deviation을 RM으로 완화하는 것이다. Segformer-B0와 EfficientViT-B0에서 거의 손실 없는 mIoU를 유지하면서 INT8 PWL unit의 면적과 전력을 크게 줄인 점이 핵심 결과다.

---

# 2. Research Background

## 2.1 Research Context

- Transformer acceleration은 대부분 MatMul/GEMM 최적화에 집중해 왔지만, 실제 integer-only deployment에서는 Softmax, LayerNorm, GELU 같은 non-linear operation이 FP32/INT32 연산을 유발한다.
- 기존 PWL/LUT 방식은 범용성을 갖지만 FP32/INT32 LUT storage 또는 input decomposition에 의존하여 INT8 accelerator 관점에서 resource wastage가 발생한다.
- Integer-only quantization에서는 activation q와 scale S가 분리되어야 하며, PWL이 `pwl(Sq)=S·pwl(q)` 형태의 separability를 제공할 수 있다는 점이 이 논문의 출발점이다.
- 다만 scale S가 커질수록 breakpoints를 quantized integer domain으로 변환할 때 오차가 커지며, 이는 function approximation MSE를 지배한다.

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
| Integer-only Transformer quantization | I-BERT, integer-only BERT-style methods | GELU/Softmax/LayerNorm을 integer approximation으로 대체 | Operator별 dataflow가 달라 범용 LUT engine으로 통합하기 어려움 | GQA-LUT는 여러 non-linear function을 PWL LUT로 통합 처리 |
| Hardware-friendly Softmax | Softermax 등 | Softmax base replacement 및 low-precision approximation | Softmax-specific design이며 GELU/RSQRT/HSWISH까지 일반화 제한 | GQA-LUT는 GELU, HSWISH, EXP, DIV, RSQRT를 같은 PWL framework로 다룸 |
| Neural LUT approximation | NN-LUT | Neural network로 LUT parameter를 학습 | 100K 수준 training data 필요, bitwidth/breakpoint quantization 반영이 복잡 | GQA-LUT는 breakpoint 기반 genetic search와 RM으로 low-bit parameter를 직접 고려 |
| Range-invariant LUT | RI-LUT | Floating-point value를 mantissa/power-of-two로 분해 | Floating-point decomposition 기반이라 INT8 q-domain에 직접 적용 어려움 | GQA-LUT는 quantized integer input과 power-of-two scale을 전제로 설계 |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 Piece-wise Linear (PWL) LUT Approximation

- **Definition:** 비선형 함수 f(x)를 여러 구간에서 `k_i x + b_i`로 근사하고, 구간 경계 p_i와 slope/intercept를 LUT에 저장하는 방식.
- **Role in This Paper:** 다양한 non-linear function을 하나의 비교기+LUT+multiply/add datapath로 통합하기 위한 핵심 연산 모델.
- **Why It Is Required:** 다양한 non-linear function을 하나의 비교기+LUT+multiply/add datapath로 통합하기 위한 핵심 연산 모델.
- **Related Components / Source:** Equation (1), Figure 1
### 4.2 Quantization-Aware Approximation

- **Definition:** 입력이 FP32가 아니라 INT8 q와 scale S로 표현된다는 점을 approximation parameter 결정에 반영하는 방식.
- **Role in This Paper:** breakpoint와 intercept가 scale S에 의해 quantized domain에서 변형되는 문제를 제어한다.
- **Why It Is Required:** breakpoint와 intercept가 scale S에 의해 quantized domain에서 변형되는 문제를 제어한다.
- **Related Components / Source:** Section 3.1, Equation (3)
### 4.3 Power-of-two Scaling

- **Definition:** learnable parameter α의 log2 값을 rounding하여 scale S를 2의 거듭제곱으로 제한하는 기법.
- **Role in This Paper:** divider를 shift로 대체하고 hardware-friendly한 scale handling을 가능하게 한다.
- **Why It Is Required:** divider를 shift로 대체하고 hardware-friendly한 scale handling을 가능하게 한다.
- **Related Components / Source:** Section 3.1
### 4.4 GQA-LUT

- **Definition:** breakpoint set을 genetic population으로 두고 crossover/mutation/tournament selection으로 MSE가 낮은 PWL을 탐색하는 알고리즘.
- **Role in This Paper:** NN-LUT보다 적은 data size로 INT8-aware approximation을 생성한다.
- **Why It Is Required:** NN-LUT보다 적은 data size로 INT8-aware approximation을 생성한다.
- **Related Components / Source:** Algorithm 1
### 4.5 Rounding Mutation (RM)

- **Definition:** breakpoint fixed-point conversion을 stochastic mutation으로 모델링하여 large-scale breakpoint deviation을 탐색 과정에 포함하는 기법.
- **Role in This Paper:** 큰 S에서 MSE가 지배되는 현상을 줄인다.
- **Why It Is Required:** 큰 S에서 MSE가 지배되는 현상을 줄인다.
- **Related Components / Source:** Algorithm 2, Figure 2


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

1. 입력은 approximation 대상 non-linear function f, breakpoint 수, population size, mutation/crossover 확률, search range, decimal bitwidth λ이다. 출력은 fixed-point 또는 quantized LUT에 저장할 slopes K, intercepts B, breakpoints P이다.
2. 먼저 integer-only quantization flow를 분석하고 scale S를 power-of-two로 제한한다. 이때 intercept는 shift로 scaling되고 breakpoint는 clipping/rounding되어 quantized breakpoint가 된다.
3. GQA-LUT는 breakpoint set을 individual로 정의한다. 각 individual에 대해 PWL function을 만들고 지정 range에서 MSE를 평가한다. 이후 crossover와 mutation을 통해 population을 진화시키고, tournament selection으로 낮은 MSE individual을 선택한다.
4. Rounding Mutation은 normal noise mutation 대신 여러 fixed-point scale로 breakpoint를 무작위 rounding한다. 이는 실제 hardware LUT 저장 시 발생하는 breakpoint quantization error를 optimization loop 안으로 끌어온다.
5. DIV와 RSQRT처럼 wide-range FXP input을 받는 연산에는 Multi-Range Input Scaling을 적용한다. 입력 범위를 여러 sub-range로 나누고 각 구간별 power-of-two scale을 사용한다.
6. 모델-level 적용에서는 Segformer-B0와 EfficientViT-B0를 INT8 integer-only quantization baseline으로 두고, EXP/GELU/DIV/RSQRT 또는 HSWISH/DIV를 8-entry PWL로 대체한다.

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

- Functions: GELU, HSWISH, EXP, DIV, RSQRT.
- Models: Segformer-B0, EfficientViT-B0.
- Dataset: Cityscapes semantic segmentation.
- Metric: operator-level MSE, model-level mIoU, hardware area/power.
- Hardware synthesis: Verilog HDL implementation, Synopsys Design Compiler, TSMC 28-nm, 500 MHz.
- Baselines: NN-LUT, GQA-LUT without RM, GQA-LUT with RM, high-precision FP32/INT32 LUT units.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| Cityscapes / Segformer-B0 | Semantic segmentation | mIoU | 74.53% with GQA-LUT w/RM, all non-linear replaced | NN-LUT 73.46%, baseline 74.60% | -0.07% vs no replacement, +1.07% vs NN-LUT | Table 4 |
| Cityscapes / EfficientViT-B0 | Semantic segmentation | mIoU | 74.15% with GQA-LUT w/RM, all non-linear replaced | NN-LUT 73.27%, baseline 74.17% | -0.02% vs no replacement, +0.88% vs NN-LUT | Table 5 |
| TSMC 28-nm synthesis | PWL hardware | Area / Power | 8-entry INT8: 961 µm² / 0.40 mW | 8-entry FP32: 5135 µm² / 2.02 mW, INT32: 5243 µm² / 1.93 mW | 81.3~81.7% area saving, 79.3~80.2% power saving | Table 6 |

## 6.3 Result Interpretation

- Operator-level MSE에서 GQA-LUT w/RM은 GELU/HSWISH/EXP처럼 input scale 영향을 크게 받는 함수에서 NN-LUT보다 안정적인 결과를 보인다.
- DIV/RSQRT는 RM보다 w/o RM이 더 적합한 경우가 있는데, 논문은 이를 이들 연산이 merely quantized input을 받기 때문이라고 해석한다.
- Model-level mIoU에서 all replacement 조건의 손실은 Segformer-B0 -0.07%, EfficientViT-B0 -0.02%에 그친다. 이는 non-linear PWL approximation이 task accuracy를 거의 유지한다는 주장을 뒷받침한다.
- Hardware 결과는 low-bit LUT storage의 직접적 이득을 보여준다. 8-entry INT8 PWL은 FP32/INT32 대비 면적과 전력에서 약 80% 수준 절감된다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

이 논문은 Transformer의 non-linear operations를 INT8-friendly LUT PWL approximation으로 변환하기 위한 GQA-LUT를 제안한다. 핵심은 quantization scale이 LUT breakpoints/intercepts에 미치는 영향을 직접 최적화에 반영하고, large scale에서 발생하는 breakpoint deviation을 RM으로 완화하는 것이다. Segformer-B0와 EfficientViT-B0에서 거의 손실 없는 mIoU를 유지하면서 INT8 PWL unit의 면적과 전력을 크게 줄인 점이 핵심 결과다.

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

- Transformer non-linear function acceleration을 operator-specific hard-coded design이 아니라 generic PWL LUT design으로 정식화했다.
- Quantization scale과 breakpoint precision 문제를 명확히 분석하여 hardware-aware approximation으로 연결했다.
- Operator-level MSE, model-level mIoU, hardware synthesis를 모두 제시하여 algorithm-hardware 연결성이 좋다.
- Segformer와 EfficientViT처럼 vanilla/linear Transformer variant 모두에서 검증했다.
- INT8 8-entry LUT의 면적/전력 이득이 명확하여 accelerator datapath 설계에 바로 참고 가능하다.

---

# 9. Cons.

- 평가는 semantic segmentation 두 모델에 한정되어 LLM/decoder-only transformer, speech transformer, time-series transformer에 대한 일반성은 추가 검증이 필요하다.
- GQA-LUT parameter search 자체의 runtime, design-time cost, function별 search stability는 제한적으로만 설명된다.
- DIV/RSQRT에서 RM이 항상 유리하지 않은 점은 method selection rule을 복잡하게 만든다.
- TSMC 28-nm 단일 synthesis 조건이므로 FPGA LUT/DSP/BRAM 기반 구현 비용은 별도 분석이 필요하다.
- End-to-end accelerator integration에서 memory interface, pipeline stall, multi-operator scheduling cost는 논문 범위 밖이다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
