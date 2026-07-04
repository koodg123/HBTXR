# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization |
| Authors | Haocheng Xi, Yuxiang Chen, Kang Zhao, KAI JUN TEH, Jianfei Chen, Jun Zhu |
| Venue | ICML 2024 / PMLR 235 / arXiv:2403.12422v2 |
| Year | 2024 |
| Research Field | Fully Quantized Training (FQT), Transformer pretraining, INT8 GPU kernels |
| Paper Link / DOI | 논문 PDF 기준 DOI 명시 없음 |
| Code / Project Link | https://github.com/thu-ml/Jetfire-INT8Training |
| Analysis Coverage | Main paper + Appendix 포함 PDF 15 pages. 코드 실행 및 GPU benchmark 재현은 수행하지 않음. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** Transformer pretraining을 fully quantized training으로 가속하려 할 때 기존 Quantize-Compute-Dequantize (QCD) 방식은 FP16 data flow를 유지하여 memory access overhead가 크고, transformer activation/gradient outlier 때문에 정확도 손실이 발생하는 문제.
- **Research Objective:** INT8 data flow와 per-block quantization을 결합하여 Transformer pretraining에서 compute, memory access, activation memory, communication을 동시에 줄이면서 FP16 baseline 수준의 정확도를 유지하는 것.
- **Why This Problem Matters:** Transformer pretraining은 대규모 token 처리, gradient 저장, weight update로 compute와 memory bandwidth를 동시에 요구한다. Linear layer는 compute-bound이나 LayerNorm/GELU/Dropout/Add는 memory-bound이므로 단순 INT8 matmul만으로는 end-to-end speedup이 제한된다.

## 1.2 Core Idea

- **Core Idea:** Activation, weight, gradient를 INT8 format으로 저장·전달하고, operator 내부 shared memory/register에서 dequantize-compute-quantize를 수행한다. Per-block quantization은 token/channel outlier를 B×B block 단위로 제한하여 tensor core 친화성과 정확도를 동시에 확보한다.
- **Proposed Approach:** Activation, weight, gradient를 INT8 format으로 저장·전달하고, operator 내부 shared memory/register에서 dequantize-compute-quantize를 수행한다. Per-block quantization은 token/channel outlier를 B×B block 단위로 제한하여 tensor core 친화성과 정확도를 동시에 확보한다.
- **Key Differentiator:** 기존 SwitchBack은 weight gradient를 FP로 남기고 FP data flow를 유지하지만, Jetfire는 8-bit gradient와 INT8 data flow를 동시에 지원하며 non-linear operators까지 INT8 input/output으로 구현한다.

## 1.3 Main Contributions

1. Transformer pretraining을 위한 INT8 data flow를 제안하여 operator 간 activation/gradient movement를 INT8로 유지한다.
2. Per-block quantization을 도입하여 activation channel-wise outlier와 gradient token-wise outlier를 동시에 완화한다.
3. CUDA 기반 INT8 linear operator와 Triton 기반 INT8 non-linear operator를 구현한다.
4. Machine translation, DeiT/Swin/ViT image classification, GPT2 pretraining/GLUE fine-tuning에서 FP16 baseline에 근접하거나 더 좋은 성능을 보인다.
5. RTX 4090에서 transformer block 기준 1.42× end-to-end training speedup과 1.49× activation memory reduction을 보고한다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| WMT14 En-De / Transformer-base | Machine translation | BLEU | 26.49 | FP 26.49, SwitchBack 26.46, Per-tensor 26.04 | FP와 동일 | Table 4 |
| ImageNet1K / DeiT-Base | Image classification pretraining | Top-1 Acc | 76.03 | FP 75.67, SwitchBack 75.62 | +0.36 vs FP | Table 4 |
| OpenWebText / GPT2-Large | Generative pretraining | Validation loss | 2.4696 | FP 2.5993, SwitchBack 3.0512 | FP보다 낮은 valid loss | Table 4 |
| GPT2-Large / GLUE | Fine-tuning | GLUE score | 82.94±0.70 | FP 83.01±0.24, SwitchBack 78.74±0.24 | -0.07 vs FP | Table 4 |
| RTX 4090 / Transformer block | Training speed | Overall speedup | 1.42× at hidden size 4096 | FP16 baseline 1× | +42% | Table 6 |
| GPT2 activation memory | Memory | Reduction | Up to 1.49× | SwitchBack lower | Better than SwitchBack | Table 7 |

## 1.5 Brief Overview

> Jetfire는 Transformer pretraining에서 INT8 quantization을 단순 matmul acceleration이 아니라 data movement format 자체로 확장한다. Per-block quantization은 per-channel과 per-token의 장점을 절충하여 tensor core에서 구현 가능한 정확한 low-precision format을 제공한다. 실험적으로 FP16 수준 accuracy를 유지하면서 custom CUDA/Triton kernels로 linear/non-linear operator와 end-to-end transformer block을 가속한다.

---

# 2. Research Background

## 2.1 Research Context

- FQT는 forward/backward 모두를 low precision으로 수행해야 training speedup을 얻을 수 있다.
- 기존 QCD 방식은 operator interface를 FP16으로 유지하기 때문에 quantize/dequantize overhead와 FP16 memory traffic이 남는다.
- Transformer activation에는 channel-wise outlier가 존재하고 gradient에는 token-axis outlier가 존재하여 per-tensor/per-token quantization만으로는 오류가 커진다.
- Non-linear operator는 arithmetic보다 global memory load/store가 병목이므로 INT8 data flow가 직접적 speedup 요인이 된다.

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
| Fully Quantized Training for CNNs | Banner, Zhu, Zhao 등 | Forward/backward를 INT8 또는 low precision으로 수행 | Transformer activation/gradient outlier에 취약 | Jetfire는 Transformer-specific per-block quantization을 설계 |
| SwitchBack | Wortsman et al. | Per-token/per-channel INT8 training for CLIP-style models | Weight gradient 계산을 FP로 남김, FP data flow 유지 | Jetfire는 8-bit gradient와 INT8 data flow를 지원 |
| FP8 training | TransformerEngine, FP8-LM | Hopper GPU의 FP8 tensor core 사용 | 특정 GPU architecture 의존 | Jetfire는 INT8 MM을 지원하는 더 넓은 GPU 범위 지향 |
| PTQ/QAT | SmoothQuant, AWQ, Q-BERT 등 | Inference 또는 fine-tuning quantization | Pretraining acceleration과 backward graph까지 다루지 않음 | Jetfire는 pretraining FQT가 목표 |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 INT8 Data Flow

- **Definition:** Operator 사이의 activation/gradient storage 및 movement를 INT8로 유지하는 방식.
- **Role in This Paper:** Memory-bound non-linear operators까지 가속하고 activation memory와 communication을 줄인다.
- **Why It Is Required:** Memory-bound non-linear operators까지 가속하고 activation memory와 communication을 줄인다.
- **Related Components / Source:** Figure 1, Section 3
### 4.2 Quantize-Compute-Dequantize (QCD)

- **Definition:** FP16 input을 임시 INT8로 바꿔 compute하고 output을 다시 FP16으로 되돌리는 방식.
- **Role in This Paper:** 기존 FQT의 한계를 설명하는 비교 기준.
- **Why It Is Required:** 기존 FQT의 한계를 설명하는 비교 기준.
- **Related Components / Source:** Section 3.1
### 4.3 Per-block Quantization

- **Definition:** N×C matrix를 B×B block으로 나누고 block별 FP16 scale factor를 부여하는 scheme.
- **Role in This Paper:** Channel-wise/token-wise outlier 영향을 block 내부로 제한하면서 WMMA outer-axis 제약을 우회한다.
- **Why It Is Required:** Channel-wise/token-wise outlier 영향을 block 내부로 제한하면서 WMMA outer-axis 제약을 우회한다.
- **Related Components / Source:** Figure 3, Equation (1)
### 4.4 3-Level Tiling

- **Definition:** CUDA block level, quantization block level, WMMA operation level로 matmul을 계층적으로 tile하는 방식.
- **Role in This Paper:** Per-block scale 처리와 INT8 tensor core 사용을 결합한다.
- **Why It Is Required:** Per-block scale 처리와 INT8 tensor core 사용을 결합한다.
- **Related Components / Source:** Section 5.2
### 4.5 INT8 Non-linear Operator

- **Definition:** INT8 input을 shared memory에서 FP32로 dequantize하고 non-linear compute 후 INT8 output으로 quantize하여 저장하는 fused Triton kernel.
- **Role in This Paper:** GELU/LayerNorm/Dropout/Add의 global memory traffic을 줄인다.
- **Why It Is Required:** GELU/LayerNorm/Dropout/Add의 global memory traffic을 줄인다.
- **Related Components / Source:** Section 6, Algorithm 2


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

1. 입력 tensor는 INT8 matrix와 FP16 scale matrix의 쌍으로 표현된다. Activation, weight, gradient 모두 이 format을 사용한다.
2. Linear operator는 CUDA로 구현된다. Global memory에서 INT8 block과 scale factor를 읽고, tensor core WMMA로 INT8 matmul을 수행한 뒤 INT32 output을 FP32로 dequantize/accumulate하고, 최종 결과를 다시 INT8 block과 scale로 quantize한다.
3. Per-block quantization은 B=32를 기본 quantization block size로 사용한다. 이는 channel-wise outlier와 token-wise outlier를 모두 완화하면서 tensor core에서 scaling 가능한 구조를 만든다.
4. Non-linear operator는 Triton으로 구현된다. Input/output read/write는 INT8이며, computation은 shared memory/register 내부에서 FP32로 수행된다. 이는 arithmetic speedup보다는 bandwidth reduction에 초점이 있다.
5. LayerNorm은 Add operator에서 row-wise mean/sum-of-squares를 미리 계산·저장하여 LayerNorm의 추가 load/store를 줄인다.
6. 실험에서는 MLP와 attention module 내 linear layers 및 GELU/LayerNorm/Dropout을 INT8로 quantize하되, multi-head attention 자체는 FlashAttention을 사용하여 FP16으로 둔다. 따라서 완전한 모든 submodule INT8은 아니며, 논문은 이 점을 설정에 명시한다.

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

- Tasks: WMT14 En-De machine translation, DeiT ImageNet1K pretraining, Swin/ViT image classification, GPT2 OpenWebText pretraining, GLUE fine-tuning.
- Models: Transformer-base, DeiT-Tiny/Small/Base, Swin-Tiny/Small/Base, ViT-Base/Large, GPT2-Base/Medium/Large.
- Implementation: CUDA linear operators, Triton non-linear operators.
- Hardware: NVIDIA RTX 4090 for main speed experiments, RTX 3090 appendix.
- Kernel settings: CUDA block size 128×32×128, Triton block size 64×64, quantization block B=32.
- Baselines: FP training, SwitchBack, per-tensor quantization, selected CNN-targeted INT8 training methods in appendix.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| WMT14 En-De / Transformer-base | Machine translation | BLEU | 26.49 | FP 26.49, SwitchBack 26.46, Per-tensor 26.04 | FP와 동일 | Table 4 |
| ImageNet1K / DeiT-Base | Image classification pretraining | Top-1 Acc | 76.03 | FP 75.67, SwitchBack 75.62 | +0.36 vs FP | Table 4 |
| OpenWebText / GPT2-Large | Generative pretraining | Validation loss | 2.4696 | FP 2.5993, SwitchBack 3.0512 | FP보다 낮은 valid loss | Table 4 |
| GPT2-Large / GLUE | Fine-tuning | GLUE score | 82.94±0.70 | FP 83.01±0.24, SwitchBack 78.74±0.24 | -0.07 vs FP | Table 4 |
| RTX 4090 / Transformer block | Training speed | Overall speedup | 1.42× at hidden size 4096 | FP16 baseline 1× | +42% | Table 6 |
| GPT2 activation memory | Memory | Reduction | Up to 1.49× | SwitchBack lower | Better than SwitchBack | Table 7 |

## 6.3 Result Interpretation

- Accuracy 결과는 Transformer-base BLEU와 DeiT/GPT2/GLUE에서 Jetfire가 FP baseline에 거의 근접하거나 일부 지표에서 더 좋음을 보인다. 논문은 일부 loss 개선을 quantization regularization effect로 해석한다.
- Per-tensor quantization은 DeiT에서 수렴하지 않는 경우가 있어 Transformer activation outlier 문제를 보여준다.
- Speed 결과에서 linear layer는 약 60% component-level improvement와 40% forward+backward overall speedup을 보이며, non-linear operators는 GELU 80%, LayerNorm backward 최대 90% speedup을 보인다.
- End-to-end transformer block speedup은 hidden size가 클수록 overhead 비율이 작아져 개선이 커진다. hidden=4096에서 overall 1.42×가 대표 결과다.
- Memory reduction은 activation storage를 INT8로 유지하는 직접 효과이며, SwitchBack보다 non-linear operator footprint까지 줄였다는 점이 차별점이다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

Jetfire는 Transformer pretraining에서 INT8 quantization을 단순 matmul acceleration이 아니라 data movement format 자체로 확장한다. Per-block quantization은 per-channel과 per-token의 장점을 절충하여 tensor core에서 구현 가능한 정확한 low-precision format을 제공한다. 실험적으로 FP16 수준 accuracy를 유지하면서 custom CUDA/Triton kernels로 linear/non-linear operator와 end-to-end transformer block을 가속한다.

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

- Training-side quantization에서 compute precision만이 아니라 data flow precision을 문제화한 점이 명확하다.
- Per-block quantization은 transformer outlier 구조와 tensor core 제약을 동시에 고려한 실용적 설계다.
- CUDA/Triton kernel 구현과 end-to-end transformer block speedup을 함께 제시하여 실현 가능성이 높다.
- NLP, CV, generative pretraining을 모두 포함하여 실험 범위가 넓다.
- Appendix에 GLUE 상세 결과와 RTX 3090 결과를 제공하여 검증 자료가 풍부하다.

---

# 9. Cons.

- Multi-head attention은 FlashAttention FP16으로 남겨, 엄밀한 의미의 모든 Transformer operation 완전 INT8 training은 아니다.
- Scale matrix가 FP16으로 유지되므로 hardware memory layout 관점에서는 INT8-only accelerator와 다르다.
- GPU tensor core와 Triton/CUDA에 강하게 의존하여 FPGA/ASIC으로 직접 이전하려면 datapath와 scale handling 재설계가 필요하다.
- Optimizer state, master weight copy는 FP32로 유지되므로 전체 training memory saving은 activation 중심이다.
- Long-context LLM, MoE, decoder-only large-scale distributed setting에서 communication 효과는 추가 검증이 필요하다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
