# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | Towards Fully 8-bit Integer Inference for the Transformer Model |
| Authors | Ye Lin, Yanyang Li, Tengbo Liu, Tong Xiao, Tongran Liu, Jingbo Zhu |
| Venue | IJCAI 2020 |
| Year | 2020 |
| Research Field | Integer-only Transformer inference, NLP quantization |
| Paper Link / DOI | 논문 PDF 기준 DOI 명시 없음 |
| Code / Project Link | 논문에 명시되지 않음 |
| Analysis Coverage | Main paper 전체 7 pages. 실제 CPU INT8 runtime 재현은 수행하지 않음. Speed-up은 논문에서 estimated value로 제시됨. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** Transformer INT8 inference에서 scale incompatibility 때문에 operation마다 quantization/dequantization이 반복되고, Softmax exponential과 LayerNorm square root가 INT8-incompatible하여 완전 INT8 data flow가 어려운 문제.
- **Research Objective:** INT8 tensor와 scale을 함께 전파하는 Scale Propagation을 정의하고, Softmax/LN을 INT8-compatible 함수로 바꾼 Integer Transformer를 제안하여 거의 fully 8-bit integer inference를 가능하게 하는 것.
- **Why This Problem Matters:** INT8는 FP32보다 storage, latency, energy, chip area 측면에서 유리하지만, 기존 practical INT8 inference는 operation마다 FP32 fallback과 dequantization이 있어 효율을 충분히 얻지 못한다.

## 1.2 Core Idea

- **Core Idea:** Scale Propagation은 `{x, s}` pair를 network 전반에 전파하며, scale matching과 re-scaling으로 INT8 operation을 유지한다. Integer Transformer는 exponential Softmax를 Polynomial Attention으로, sqrt 기반 LayerNorm을 L1 LayerNorm으로 대체한다.
- **Proposed Approach:** Scale Propagation은 `{x, s}` pair를 network 전반에 전파하며, scale matching과 re-scaling으로 INT8 operation을 유지한다. Integer Transformer는 exponential Softmax를 Polynomial Attention으로, sqrt 기반 LayerNorm을 L1 LayerNorm으로 대체한다.
- **Key Differentiator:** 기존 Transformer architecture를 그대로 quantize하는 대신, INT8 compatibility를 위해 attention과 normalization을 architecture-level로 수정한다. 이 때문에 INT8 forward propagation이 대부분 dequantization 없이 흐를 수 있다.

## 1.3 Main Contributions

1. INT8 tensor와 scale을 함께 다루는 Scale Propagation protocol을 제안하였다.
2. Scale incompatibility 문제를 scale matching과 re-scaling으로 해결하였다.
3. Softmax exponential을 ReLU + polynomial 기반 Polynomial Attention으로 대체하였다.
4. LayerNorm의 square root를 L1-norm 기반 L1 Layer Normalization으로 대체하였다.
5. MT와 LM tasks에서 FP32 baseline과 유사한 성능, 약 4× storage reduction, 평균 3.47× estimated speed-up을 보고하였다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| WMT16 En→Ro / base | Machine translation | BLEU / Storage / Speed | INT8 32.54, 80M, 3.53× | FP32 baseline 32.55, 318M, 1× | Comparable BLEU, ~4× less storage | Table 2 |
| WMT14 En→De / base | Machine translation | BLEU / Storage / Speed | INT8 26.91, 76M, 3.24× | FP32 baseline 26.95, 302M | -0.04 BLEU | Table 2 |
| WMT14 En→Fr / base | Machine translation | BLEU / Storage / Speed | INT8 40.00, 107M, 3.03× | FP32 baseline 40.88, 425M | -0.88 BLEU vs baseline | Table 2 |
| WMT14 De→En / big | Machine translation | BLEU / Storage / Speed | INT8 33.46, 236M, 3.68× | FP32 baseline 33.07, 939M | +0.39 BLEU | Table 2 |
| WikiText-103 / lm-big | Language modeling | Test PPL / Storage / Speed | INT8 18.23, 280M, 3.78× | Baseline FP32 18.86, 944M | Better PPL, ~3.37× less storage | Table 3 |

## 1.5 Brief Overview

> 이 논문은 Transformer를 거의 완전한 INT8 inference로 실행하기 위해 Scale Propagation과 Integer Transformer를 제안한다. 핵심은 tensor뿐 아니라 scale을 함께 전파하여 반복적인 dequantization을 줄이고, Softmax와 LayerNorm을 INT8-compatible 함수로 대체하는 것이다. MT/LM 실험에서 저장공간을 약 4배 줄이고 평균 3.47배 estimated speed-up을 제시하지만, architecture modification에 따른 task별 성능 차이가 있다.

---

# 2. Research Background

## 2.1 Research Context

- INT8 inference는 저장공간, 연산속도, 에너지, chip area 측면에서 FP32보다 유리하다.
- 그러나 practical INT8 inference는 각 operation 전후에 Q/D를 삽입해 FP32 interface를 유지하는 경우가 많아 overhead가 크다.
- Scale incompatibility는 서로 다른 scale로 quantize된 INT8 tensors를 직접 add/matmul할 수 없게 만든다.
- INT8 incompatibility는 exp, sqrt처럼 integer input에 integer output을 보장하지 않거나 INT8 arithmetic으로 표현하기 어려운 함수에서 발생한다.

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
| Low precision neural networks | Binary/Ternary networks, FP16 training, INT8 inference | Bitwidth를 낮춰 storage/compute 절감 | 복잡한 Transformer에는 직접 적용 어려움 | Scale Propagation은 Transformer 연산 전반을 INT8 pair로 표현 |
| INT8 Transformer inference | Bhandare et al., Prato et al., Wu | Transformer INT8 quantization | 일부 FP32 operation 유지 또는 BLEU 평가 제한 | 본 논문은 architecture modification으로 INT8 compatibility를 높임 |
| Efficient Transformer design | Attention sharing, efficient architectures | Model 자체를 경량화 | 정수 inference problem을 직접 해결하지 않음 | Integer Transformer는 quantization-friendly architecture change |
| Normalization alternatives | L1 BatchNorm | sqrt 대신 L1 norm 사용 | Transformer LayerNorm에 직접 적용 필요 | L1 LayerNorm으로 Transformer에 확장 |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 Scale Propagation

- **Definition:** INT8 tensor x와 scale s를 pair로 묶어 operation마다 함께 갱신·전파하는 protocol.
- **Role in This Paper:** Dequantization 없이 INT8 forward flow를 유지한다.
- **Why It Is Required:** Dequantization 없이 INT8 forward flow를 유지한다.
- **Related Components / Source:** Algorithm 1, Table 1
### 4.2 Scale Matching

- **Definition:** 여러 input tensors의 scale을 공통 scale로 맞추기 위해 더 작은 scale 기준으로 tensor를 조정하는 과정.
- **Role in This Paper:** Addition/MatMul에서 scale incompatibility를 해결한다.
- **Why It Is Required:** Addition/MatMul에서 scale incompatibility를 해결한다.
- **Related Components / Source:** Equation (8)
### 4.3 Re-scaling

- **Definition:** Operation 결과가 INT8 range를 넘을 때 scale과 tensor를 재조정하여 INT8로 되돌리는 과정.
- **Role in This Paper:** INT32 intermediate를 다음 INT8 operation에 연결한다.
- **Why It Is Required:** INT32 intermediate를 다음 INT8 operation에 연결한다.
- **Related Components / Source:** Equation (10)
### 4.4 Polynomial Attention

- **Definition:** Softmax의 exp를 ReLU+bias 후 polynomial function으로 대체한 attention.
- **Role in This Paper:** INT8-compatible positive attention scores를 만든다.
- **Why It Is Required:** INT8-compatible positive attention scores를 만든다.
- **Related Components / Source:** Equations (11)(12), Figure 4
### 4.5 L1 Layer Normalization

- **Definition:** Standard deviation의 sqrt를 L1 norm approximation으로 바꾼 LayerNorm.
- **Role in This Paper:** Square root incompatibility를 제거한다.
- **Why It Is Required:** Square root incompatibility를 제거한다.
- **Related Components / Source:** Equation (13)


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

1. Scale Propagation의 기본 input/output은 `{x, s}` pair이다. x는 INT8 tensor, s는 대응하는 scale이다.
2. Concatenation/transpose는 tensor와 scale에 동일 shape transform을 적용한다. Element-wise multiplication은 tensor끼리, scale끼리 독립적으로 곱한다.
3. Addition은 scale matching을 먼저 수행하여 공통 scale을 만든 뒤 INT8 tensors를 더한다.
4. MatMul은 accumulation dimension의 scale을 matching한 뒤 tensor matmul과 scale matmul을 각각 수행한다.
5. Operation result는 overflow 가능성이 있으므로 INT32로 저장한 뒤 re-scaling을 통해 INT8로 투영한다.
6. Polynomial Attention은 `Poly(x) = [ReLU(x+b)]^n + |δ|`를 사용한다. ReLU는 negative scores를 제거하고 polynomial은 exp처럼 큰 score를 더 강조하는 역할을 한다.
7. LayerNorm은 standard deviation 대신 L1 norm equivalent를 사용한다. 이는 absolute value operation을 사용하므로 INT8-compatible하다.
8. Training은 modified Integer Transformer architecture를 FP32로 학습한 뒤 INT8 inference를 수행한다. 학습 자체를 INT8로 하는 논문은 아니다.

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

- Tasks: WMT16 En↔Ro, WMT14 En↔De, WMT14 En→Fr machine translation, WikiText-103 language modeling.
- MT model: Transformer-base and Transformer-big, 6-layer encoder/decoder.
- Embedding size: 512 for base, 1024 for big. Heads: 8/16 for base/big.
- Optimizer: Adam for MT, inverse square root LR schedule, warmup 8K.
- LM: lm-base/lm-big following Baevski and Auli, Nesterov for lm-big, cosine LR schedule.
- Hardware: 8 NVIDIA TITAN V GPUs for experiments.
- Metrics: BLEU for MT, perplexity for LM, storage MB, estimated speed-up.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| WMT16 En→Ro / base | Machine translation | BLEU / Storage / Speed | INT8 32.54, 80M, 3.53× | FP32 baseline 32.55, 318M, 1× | Comparable BLEU, ~4× less storage | Table 2 |
| WMT14 En→De / base | Machine translation | BLEU / Storage / Speed | INT8 26.91, 76M, 3.24× | FP32 baseline 26.95, 302M | -0.04 BLEU | Table 2 |
| WMT14 En→Fr / base | Machine translation | BLEU / Storage / Speed | INT8 40.00, 107M, 3.03× | FP32 baseline 40.88, 425M | -0.88 BLEU vs baseline | Table 2 |
| WMT14 De→En / big | Machine translation | BLEU / Storage / Speed | INT8 33.46, 236M, 3.68× | FP32 baseline 33.07, 939M | +0.39 BLEU | Table 2 |
| WikiText-103 / lm-big | Language modeling | Test PPL / Storage / Speed | INT8 18.23, 280M, 3.78× | Baseline FP32 18.86, 944M | Better PPL, ~3.37× less storage | Table 3 |

## 6.3 Result Interpretation

- Table 2/3 결과는 storage reduction이 거의 4×에 가깝지만 scale 저장 때문에 정확히 4×는 아님을 보여준다.
- Average speed-up은 실제 CPU/GPU runtime이 아니라 operation별 time consumption과 INT8 speed-up 가정에 기반한 estimated value다. 따라서 직접 hardware 측정으로 해석하면 안 된다.
- Ablation에서 Polynomial Attention은 FP32 성능을 약간 개선하는 경우가 있고, L1 LayerNorm은 standard LayerNorm에 근접한다. 이는 architecture modification이 FP32에서도 유효할 수 있음을 시사한다.
- En→Fr에서는 INT8 성능 하락이 더 크며, 논문은 마지막 residual connection과 LayerNorm의 precision loss가 중요한 원인이라고 분석한다.
- Scale granularity가 더 세밀할수록 성능이 좋아지지만 scale storage/compute overhead가 늘어난다. 이는 후속 integer-only design의 핵심 trade-off다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

이 논문은 Transformer를 거의 완전한 INT8 inference로 실행하기 위해 Scale Propagation과 Integer Transformer를 제안한다. 핵심은 tensor뿐 아니라 scale을 함께 전파하여 반복적인 dequantization을 줄이고, Softmax와 LayerNorm을 INT8-compatible 함수로 대체하는 것이다. MT/LM 실험에서 저장공간을 약 4배 줄이고 평균 3.47배 estimated speed-up을 제시하지만, architecture modification에 따른 task별 성능 차이가 있다.

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

- Scale incompatibility와 INT8 incompatibility를 명확히 분리하여 문제 정의가 좋다.
- 연산별 INT8 equivalent를 Table 1로 정리하여 integer inference flow를 재현하기 쉽다.
- Softmax/LN을 architecture-level로 수정하여 dequantization fallback을 줄이는 접근이 선구적이다.
- MT와 LM 모두에서 평가하여 NLP Transformer 관점의 일반성을 확보하려 했다.
- 정확도 하락 module을 분석하여 precision loss 위치를 진단했다.

---

# 9. Cons.

- Speed-up은 estimated이며 실제 CPU/GPU/ASIC/FPGA kernel 측정이 아니다.
- Architecture를 바꾸므로 pretrained standard Transformer에 post-training으로 바로 적용하기 어렵다.
- Polynomial Attention과 L1 LayerNorm이 모든 task/model에서 standard attention/LN과 동등하다고 보기 어렵다.
- En→Fr처럼 residual/LN precision loss에 민감한 task에서 성능 하락이 존재한다.
- INT8 inference는 다루지만 training-side quantization이나 optimizer state reduction은 다루지 않는다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
