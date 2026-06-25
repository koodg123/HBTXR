# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | Integer-only Quantized Transformers for Embedded FPGA-based Time-series Forecasting in AIoT |
| Authors | Tianheng Ling, Chao Qian, Gregor Schiele |
| Venue | arXiv:2407.11041v5 |
| Year | 2025 |
| Research Field | AIoT, embedded FPGA, time-series forecasting, integer-only Transformer |
| Paper Link / DOI | 논문 PDF 기준 DOI 명시 없음 |
| Code / Project Link | https://github.com/tianheng-ling/TinyTransformer4TS |
| Analysis Coverage | Main paper 전체 7 pages. VHDL generation, GHDL simulation, Vivado synthesis 및 hardware validation은 논문 결과만 분석하고 재실행하지 않음. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** Transformer 기반 time-series forecasting을 IoT-class embedded FPGA에 배치할 때 resource, timing, power, energy 제약이 매우 강하며, 단순 bitwidth 감소가 항상 latency/energy 감소로 이어지지 않는 문제.
- **Research Objective:** FPGA-friendly Transformer architecture, integer-only QAT, VHDL template 기반 accelerator generation을 결합하여 8/6/4-bit time-series Transformer를 Spartan-7 XC7S15에서 검증하는 것.
- **Why This Problem Matters:** AIoT 환경에서는 센서 데이터를 로컬에서 처리해야 통신 비용과 network dependency를 줄일 수 있다. 하지만 Transformer는 sequence dependency modeling에 강점이 있어도 MCU/소형 FPGA에는 무겁다.

## 1.2 Core Idea

- **Core Idea:** LayerNorm을 BatchNorm으로 대체하고 MHA head count를 1로 단순화한 FPGA-friendly Transformer를 설계한다. PyTorch QAT로 quantized model과 quantization parameters를 얻고, Python scripts가 이를 VHDL templates에 주입하여 accelerator를 생성한다.
- **Proposed Approach:** LayerNorm을 BatchNorm으로 대체하고 MHA head count를 1로 단순화한 FPGA-friendly Transformer를 설계한다. PyTorch QAT로 quantized model과 quantization parameters를 얻고, Python scripts가 이를 VHDL templates에 주입하여 accelerator를 생성한다.
- **Key Differentiator:** 서버급 FPGA나 GPU가 아니라 매우 작은 Spartan-7 XC7S15를 대상으로 하며, time-series forecasting regression task에서 8/6/4-bit integer-only deployment의 precision-resource-energy trade-off를 정량 분석한다.

## 1.3 Main Contributions

1. Traffic flow와 air quality dataset에서 FPGA-friendly 8-bit Transformer가 기존 8-bit benchmark보다 각각 8.47%, 33.47% 나은 precision을 보였다고 보고한다.
2. 8/6/4-bit trained quantized model을 FPGA-ready hardware accelerator로 변환하는 reusable, pipeline-enabled VHDL template을 제공한다.
3. Xilinx Spartan-7 XC7S15에서 model precision, resource utilization, timing, power, energy를 통합 분석한다.
4. 4-bit quantized Transformer가 관련 8-bit model 대비 test loss +0.63%만 증가하면서 132.33× faster, 48.19× less energy를 달성한다고 보고한다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| PeMS | Traffic flow forecasting | RMSE | 8-bit model improves related 8-bit by 8.47%; 6-bit comparable | Related 8-bit model | Better or comparable | Section V-C |
| AirU | Air quality forecasting | RMSE | 8-bit improves related 8-bit by 33.47%; 4-bit outperforms related 8-bit by 2.83% | Becnel et al. 8-bit | Better even at 4-bit | Section V-C |
| Spartan-7 / AirU (6,64,8) | Inference | Time / Energy | 2.82 ms, 0.212 mJ, RMSE 3.506 | Candidate 8-bit | Best precision candidate | Table IV |
| Spartan-7 / AirU (12,64,6) | Inference | Time / Energy | 4.61 ms, 0.364 mJ, RMSE 3.763 | 8-bit candidate | Slower and more energy | Table IV |
| Spartan-7 / AirU (12,32,4) | Inference | Time / Energy | 1.33 ms, 0.084 mJ, RMSE 5.474 | 8-bit candidate | 2.12× faster, 2.52× more energy-efficient vs 8-bit candidate | Table IV |

## 1.5 Brief Overview

> 이 논문은 AIoT time-series forecasting용 Transformer를 작은 Spartan-7 FPGA에 배치하기 위해 architecture simplification, integer-only QAT, VHDL template generation을 결합한다. 결과는 bitwidth 감소가 precision/resource/timing/energy에 미치는 영향이 단순하지 않음을 보여준다. 특히 4-bit 모델은 정확도 손실을 감수하는 대신 매우 낮은 latency와 energy를 달성한다.

---

# 2. Research Background

## 2.1 Research Context

- AIoT에서는 sensor data를 edge에서 추론하여 network dependency와 transmission cost를 줄이는 것이 중요하다.
- Transformer는 long sequence와 global dependency modeling에 강하지만, embedded IoT device에는 parameter와 operation 수가 부담된다.
- LayerNorm은 inference 시 mean/std computation과 division/sqrt를 요구하여 small FPGA에 부적합하다.
- Time-series forecasting은 classification과 달리 regression precision이 중요하여 Softmax/quantization approximation error가 RMSE에 민감하게 반영된다.

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
| Tiny time-series Transformer | Becnel et al. | MCU에서 8-bit Transformer 기반 sensor inference | 176 ms inference time, edge FPGA acceleration 미흡 | 본 논문은 embedded FPGA accelerator로 speed/energy 개선 |
| Transformer FPGA acceleration for TS/ASR | Yamini et al. 등 | FPGA에서 Transformer inference accelerator 구현 | Server/edge-grade FPGA 중심 | Spartan-7 같은 IoT-class FPGA 대상 |
| Integer-only quantization | Jacob et al., I-BERT, I-ViT | Integer arithmetic-only inference | 주로 CNN/NLP/CV 중심 | Time-series forecasting regression task에 적용 |
| Efficient Transformers | Efficient Transformer survey works | Architecture simplification | Hardware template generation과 직접 연결되지 않음 | FPGA-friendly architecture와 VHDL generation 통합 |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 FPGA-friendly Transformer

- **Definition:** Input module, one encoder layer, output module로 구성된 compact single-step forecasting Transformer.
- **Role in This Paper:** Small FPGA에 들어가는 parameter/resource budget을 맞춘다.
- **Why It Is Required:** Small FPGA에 들어가는 parameter/resource budget을 맞춘다.
- **Related Components / Source:** Figure 1, Table I
### 4.2 Parameter Simplification

- **Definition:** Q/K/V/output vector dimension을 d_model로 통일하고 MHA head h=1, FFN dimension=4d_model로 단순화.
- **Role in This Paper:** Model parameter formula를 단순화하고 hardware generation을 쉽게 한다.
- **Why It Is Required:** Model parameter formula를 단순화하고 hardware generation을 쉽게 한다.
- **Related Components / Source:** Equation (1)
### 4.3 BatchNorm Replacement

- **Definition:** LayerNorm을 BatchNorm으로 대체하여 training에서 pre-computed statistics를 사용.
- **Role in This Paper:** Inference-time division/sqrt 부담을 줄인다.
- **Why It Is Required:** Inference-time division/sqrt 부담을 줄인다.
- **Related Components / Source:** Section II-B
### 4.4 Integer-only Quantization

- **Definition:** Scale S와 zero point Z로 real tensor를 b-bit signed integer로 mapping.
- **Role in This Paper:** All weights, inputs, outputs, inter-layer activations를 integer-only operation으로 실행한다.
- **Why It Is Required:** All weights, inputs, outputs, inter-layer activations를 integer-only operation으로 실행한다.
- **Related Components / Source:** Equations (3)-(6)
### 4.5 VHDL Template Generation

- **Definition:** Trained quantized model과 quantization parameters를 Python script가 VHDL templates에 주입.
- **Role in This Paper:** FPGA expertise가 낮아도 accelerator generation 가능하게 한다.
- **Why It Is Required:** FPGA expertise가 낮아도 accelerator generation 가능하게 한다.
- **Related Components / Source:** Section IV


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

1. Input은 time-series sequence `X`이며, output은 single-step ahead forecast이다. Model은 input linear + positional encoding, MHA, BN, FFN, GAP, output linear로 구성된다.
2. MHA head count h를 1로 제한하고 Q/K/V/output dimensions를 d_model로 통일하여 parameter count를 `12d_model^2 + (15+m)d_model + 1`로 단순화한다.
3. Scaled Dot-Product Attention의 `sqrt(d_model/h)` scaling은 MatMulScore의 quantization scaling factor에 통합하여 RTL-level 추가 연산을 제거한다.
4. LayerNorm 대신 BatchNorm을 사용한다. BN은 mean/variance를 training 중 추정하고 inference에서 fixed parameters로 적용하므로 embedded FPGA에서 더 단순하다.
5. Quantization은 asymmetric scheme을 기본으로 하되, BN bias/offset은 symmetric scheme을 사용한다. Mixed precision은 framework가 지원하지만 본 논문 실험은 uniform bitwidth를 사용한다.
6. Addition, MatMul, Softmax, BatchNorm, GAP component는 모두 integer-only 형태로 변환된다. Floating scale ratio는 precomputed ApproxMul 형태의 integer multiply + right shift로 근사한다.
7. Softmax는 NLUT/DLUT를 사용하며 exp output range를 quantize한다. FPGA divider는 resource/timing 병목이므로 Radix-2 non-restoring divider를 사용한다.
8. 최종 hardware generation은 PyTorch-trained quantized model과 quantization parameters를 VHDL templates로 변환하고, GHDL cycles, Vivado resource/timing/power, actual hardware validation으로 평가한다.

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

- Datasets: PeMS traffic flow, AirU air quality.
- PeMS: 11,160 sensor measurements over four weeks, selected sensor index 4192.
- AirU: 19,380 observations, 7 variables, Ozone target; after cleaning 15,258 pairs, 14,427 train / 831 test.
- Training: 50 sessions, 100 epochs, early stopping, batch size 256, Adam β1=0.9 β2=0.98 ε=1e-9, LR 0.001, step scheduler size 3 gamma 0.5.
- Software: CUDA 11.0, PyTorch 3.11, Ubuntu 20.04.6 LTS, NVIDIA RTX 2080 SUPER for training.
- Hardware: Xilinx Spartan-7 XC7S15 FPGA, GHDL, Vivado.
- Metrics: RMSE, LUT/BRAM/DSP utilization, clock frequency, cycles, inference time, static/dynamic/total power, energy.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| PeMS | Traffic flow forecasting | RMSE | 8-bit model improves related 8-bit by 8.47%; 6-bit comparable | Related 8-bit model | Better or comparable | Section V-C |
| AirU | Air quality forecasting | RMSE | 8-bit improves related 8-bit by 33.47%; 4-bit outperforms related 8-bit by 2.83% | Becnel et al. 8-bit | Better even at 4-bit | Section V-C |
| Spartan-7 / AirU (6,64,8) | Inference | Time / Energy | 2.82 ms, 0.212 mJ, RMSE 3.506 | Candidate 8-bit | Best precision candidate | Table IV |
| Spartan-7 / AirU (12,64,6) | Inference | Time / Energy | 4.61 ms, 0.364 mJ, RMSE 3.763 | 8-bit candidate | Slower and more energy | Table IV |
| Spartan-7 / AirU (12,32,4) | Inference | Time / Energy | 1.33 ms, 0.084 mJ, RMSE 5.474 | 8-bit candidate | 2.12× faster, 2.52× more energy-efficient vs 8-bit candidate | Table IV |

## 6.3 Result Interpretation

- Table II는 d_model 증가가 항상 FP32 precision 개선으로 이어지지 않음을 보여준다. 이는 small dataset/time-series에서는 model capacity와 overfitting/quantization sensitivity가 상호작용함을 의미한다.
- 4-bit quantization은 PeMS에서 RMSE 증가가 더 크며, 논문은 univariate dataset이 quantization에 더 취약할 가능성을 제시한다.
- Resource table에서 d_model=64, n=24 같은 큰 configuration은 4-bit에서도 작은 FPGA에 들어가지 않는다. 따라서 bitwidth뿐 아니라 sequence length와 embedding dimension의 joint DSE가 필요하다.
- Table IV에서 6-bit 모델은 8-bit보다 frequency가 높지만 cycles가 많아 latency/energy가 더 나쁘다. 이는 bitwidth reduction이 무조건 energy 감소를 보장하지 않음을 보여주는 중요한 결과다.
- 4-bit 모델은 precision을 희생하지만 energy-latency 측면에서 IoT-class deployment에 가장 유리한 candidate로 제시된다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

이 논문은 AIoT time-series forecasting용 Transformer를 작은 Spartan-7 FPGA에 배치하기 위해 architecture simplification, integer-only QAT, VHDL template generation을 결합한다. 결과는 bitwidth 감소가 precision/resource/timing/energy에 미치는 영향이 단순하지 않음을 보여준다. 특히 4-bit 모델은 정확도 손실을 감수하는 대신 매우 낮은 latency와 energy를 달성한다.

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

- 소형 embedded FPGA를 대상으로 하여 실제 AIoT deployment constraint를 잘 반영한다.
- Precision, resource, timing, power, energy를 함께 분석하여 단순 accuracy 논문보다 hardware relevance가 높다.
- VHDL template과 generation script를 제시하여 reproducible hardware flow에 가깝다.
- Bitwidth뿐 아니라 n, d_model configuration을 포함한 DSE 관점이 명확하다.
- LayerNorm→BatchNorm, scaling integration 등 hardware-friendly architecture modification이 실용적이다.

---

# 9. Cons.

- MHA head count h=1과 single encoder layer 중심이므로 일반적인 larger Transformer와 구조 차이가 크다.
- Regression task에서 Softmax LUT/divider design이 dataset/model별로 얼마나 일반화되는지는 추가 검증 필요.
- 4-bit 모델은 AirU best 8-bit 대비 RMSE가 56.13% 높아 precision-critical application에는 부적합할 수 있다.
- Spartan-7 단일 FPGA 기준이라 larger FPGA/ASIC에서의 optimal design point는 달라질 수 있다.
- Mixed precision을 지원한다고 하지만 본 논문 실험은 uniform bitwidth 중심이므로 mixed-precision DSE는 future work로 남는다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
