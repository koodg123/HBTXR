# Paper Summary

## Paper Information

| Item | Description |
|---|---|
| Title | ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design |
| Authors | Haoran You, Zhanyi Sun, Huihong Shi, Zhongzhi Yu, Yang Zhao, Yongan Zhang, Chaojian Li, Baopu Li, Yingyan Lin |
| Venue | arXiv:2210.09573v3 / HPCA lineage |
| Year | 2025 arXiv v3 |
| Research Field | Vision Transformer sparse attention acceleration, algorithm-hardware co-design |
| Paper Link / DOI | 논문 PDF 기준 DOI 명시 없음 |
| Code / Project Link | https://github.com/GATECH-EIC/ViTCoD |
| Analysis Coverage | Main paper 전체 14 pages. 코드 실행 및 RTL/cycle simulator 재현은 수행하지 않음. |

---

# 1. Overview

## 1.1 Problem Statement

- **Target Problem:** ViT inference에서 self-attention은 latency 병목이며, ViT attention sparsity는 fixed sparse pattern으로 높게 만들 수 있지만 high sparsity는 irregular access, workload imbalance, data movement bottleneck을 유발한다.
- **Research Objective:** ViT의 fixed token count와 fixed sparse attention 특성을 활용하여 attention computation과 Q/K data movement를 동시에 줄이는 algorithm-accelerator co-design을 제안하는 것.
- **Why This Problem Matters:** NLP Transformer accelerator는 variable-length sequence와 dynamic sparse mask를 전제로 하므로 ViT의 fixed sparse attention에 최적이 아니다. ViT는 90% 수준 sparse attention에서도 accuracy drop이 작지만, diagonal sparse pattern 때문에 PE utilization과 memory bandwidth 문제가 생긴다.

## 1.2 Core Idea

- **Core Idea:** Split-and-conquer algorithm으로 attention map을 fixed mask 기반으로 prune하고 denser/sparser 두 workload로 polarize한다. Q/K를 head dimension 방향 lightweight auto-encoder로 압축하여 off-chip movement를 줄이고, hardware는 denser/sparser engine이 병렬 처리하는 two-pronged accelerator를 사용한다.
- **Proposed Approach:** Split-and-conquer algorithm으로 attention map을 fixed mask 기반으로 prune하고 denser/sparser 두 workload로 polarize한다. Q/K를 head dimension 방향 lightweight auto-encoder로 압축하여 off-chip movement를 줄이고, hardware는 denser/sparser engine이 병렬 처리하는 two-pronged accelerator를 사용한다.
- **Key Differentiator:** Sparse attention acceleration을 NLP의 dynamic sparsity가 아니라 ViT의 static/fixed sparsity에 맞추고, AE module로 data movement를 computation으로 교환하는 구조를 accelerator에 직접 통합한다.

## 1.3 Main Contributions

1. Sparse ViT 전용 algorithm-accelerator co-design framework인 ViTCoD를 제안하였다.
2. Attention map을 fixed masks로 prune하고 denser/sparser pattern으로 reorder/polarize하는 split-and-conquer algorithm을 제안하였다.
3. Q/K vectors를 압축·복원하는 learnable auto-encoder module을 삽입하여 off-chip data movement를 줄였다.
4. Denser Engine과 Sparser Engine, encoder/decoder engine을 갖는 two-pronged accelerator를 설계하였다.
5. CPU, EdgeGPU, GPU, SpAtten, Sanger 대비 large speedup과 energy efficiency 향상을 보고하였다.

## 1.4 Representative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| Attention layers / 90% sparsity | Core attention acceleration | Speedup | 235.3× vs CPU, 142.9×/160.6× vs EdgeGPU, 86.0× vs GPU | General platforms | Large speedup | Figure 15, 19 |
| Attention layers / 90% sparsity | Accelerator comparison | Speedup | 10.1× vs SpAtten, 6.8× vs Sanger | Similar hardware configuration/area simulation | Better sparse ViT-specific acceleration | Section VI-B |
| 60/70/80/90% sparsity average | Accelerator comparison | Speedup | 127.2×, 77.0×, 46.5×, 6.8×, 4.3× over CPU/EdgeGPU/GPU/SpAtten/Sanger | Multiple sparsity settings | Consistent benefit | Section VI-D |
| ImageNet / DeiT, LeViT | Accuracy-latency trade-off | Accuracy drop | <1% for 90% DeiT and 80% LeViT sparsity | Unpruned baseline | Negligible accuracy drop | Section VI-C |
| ViTCoD vs Sanger | Energy efficiency | Energy efficiency improvement | 9.8× | Most competitive baseline Sanger | +9.8× | Figure 19 |

## 1.5 Brief Overview

> ViTCoD는 ViT attention의 static sparsity 가능성을 이용해 attention map을 denser/sparser 두 workload로 구조화하고, Q/K auto-encoder로 data movement를 줄이는 co-design이다. Hardware는 두 workload를 전용 engine으로 분리하고 K-stationary/output-stationary dataflow를 조합한다. 논문은 90% sparsity 조건에서 기존 attention accelerators와 general platforms 대비 큰 speedup과 energy efficiency 향상을 보인다.

---

# 2. Research Background

## 2.1 Research Context

- ViT는 image를 고정 크기 patch/token sequence로 처리하므로 입력 token 수가 상대적으로 고정된다.
- NLP Transformer는 sequence length가 입력마다 달라 dynamic sparse attention prediction과 reconfigurable hardware가 필요하지만, ViT는 fixed sparse attention mask가 가능하다.
- Self-attention의 Q·K^T와 S·V는 SDDMM/SpMM으로 변하며 sparsity가 높을수록 computation은 줄지만 Q/K/V data access irregularity가 커진다.
- Roofline 관점에서 sparse ViT attention은 computation intensity가 낮아져 memory/bandwidth bound가 되기 쉽다.

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
| ViT model design | ViT, DeiT, Swin, LeViT, MobileViT | Vision task를 patch token과 attention으로 처리 | Self-attention latency와 hardware cost | ViTCoD는 model architecture가 아니라 sparse attention execution을 co-design |
| Sparse attention algorithms | BigBird, Reformer, Longformer, BlockBERT | Attention map sparsification | NLP 중심, dynamic/input-dependent pattern | ViTCoD는 ViT fixed mask와 high sparsity를 사용 |
| Sparse tensor accelerators | OuterSpace, ExTensor, SpArch, Gamma | SpGEMM sparse dataflow 최적화 | Unstructured sparse algebra 중심 | ViTCoD는 SDDMM/SpMM attention workload에 특화 |
| Transformer accelerators | A3, ELSA, SpAtten, Sanger, DOTA | Attention approximation/sparsity accelerator | NLP 중심, dynamic mask overhead | ViTCoD는 denser/sparser two-pronged static workload accelerator |

## 3.2 Relationship to Prior Work

- 이 논문은 기존 연구의 단순 연장이라기보다, 특정 bottleneck을 명확히 분리하고 그 bottleneck에 맞는 algorithm-hardware interface를 설계한다.
- 직접 비교 시 dataset, model, precision, training setting, target hardware가 다를 수 있으므로 `직접 비교에 주의가 필요함`.
- 특히 reported throughput, GOP/s, speedup, energy efficiency는 대상 operation 범위가 full-network인지 submodule인지 확인해야 한다.

---

# 4. Key Concepts

### 4.1 Fixed Sparse Attention Mask

- **Definition:** ViT attention map에서 input-independent mask를 생성하여 inference 동안 고정 사용하는 sparsity pattern.
- **Role in This Paper:** Mask generation overhead를 줄이고 hardware scheduling을 단순화한다.
- **Why It Is Required:** Mask generation overhead를 줄이고 hardware scheduling을 단순화한다.
- **Related Components / Source:** Figure 2, Algorithm 1
### 4.2 Split and Conquer Algorithm

- **Definition:** Attention map을 pruning하고 token reorder를 통해 denser block과 sparser region으로 분리하는 알고리즘.
- **Role in This Paper:** Workload imbalance를 줄이고 two-pronged accelerator로 mapping 가능하게 한다.
- **Why It Is Required:** Workload imbalance를 줄이고 two-pronged accelerator로 mapping 가능하게 한다.
- **Related Components / Source:** Algorithm 1, Figure 8
### 4.3 Auto-encoder Module

- **Definition:** Q/K를 attention head dimension에서 압축 후 복원하는 lightweight learnable module.
- **Role in This Paper:** High-cost off-chip Q/K movement를 lower-cost on-chip computation으로 대체한다.
- **Why It Is Required:** High-cost off-chip Q/K movement를 lower-cost on-chip computation으로 대체한다.
- **Related Components / Source:** Figure 9, Equation (2)
### 4.4 Two-pronged Architecture

- **Definition:** Denser Engine과 Sparser Engine으로 sparse attention의 두 workload를 병렬 처리하는 accelerator microarchitecture.
- **Role in This Paper:** 각 pattern에 맞는 PE/MAC allocation과 buffer/dataflow를 제공한다.
- **Why It Is Required:** 각 pattern에 맞는 PE/MAC allocation과 buffer/dataflow를 제공한다.
- **Related Components / Source:** Figure 12
### 4.5 K-stationary Dataflow

- **Definition:** K vector를 유지하고 관련 Q vector를 순차적으로 곱해 attention scores를 column-wise 생성하는 dataflow.
- **Role in This Paper:** Sparse attention에서 필요한 Q/K pair만 계산하기 좋다.
- **Why It Is Required:** Sparse attention에서 필요한 Q/K pair만 계산하기 좋다.
- **Related Components / Source:** Figure 11, 13


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

1. Input은 pretrained ViT model과 training set에서 추출한 average normalized attention map이다. Output은 fixed sparse mask, reordered attention map, global token 수, 그리고 이를 실행할 accelerator instruction/configuration이다.
2. Split-and-conquer는 먼저 attention scores를 정렬하고 threshold θp 기준으로 cumulative information quantity가 만족될 때까지 중요한 attention만 보존한다.
3. 이후 non-zero element 수가 threshold θd보다 큰 token을 global token으로 보고 앞쪽으로 reorder하여 denser block을 만든다. 나머지는 sparser workload로 처리한다.
4. Auto-encoder는 Q/K의 attention head dimension redundancy를 활용해 12 heads → 6 heads 같은 compression을 수행하고, reconstruction loss `||Q-Q'|| + ||K-K'||`를 cross-entropy와 함께 optimize한다.
5. Unified pipeline은 pretrained ViT에 AE module 삽입 → fine-tuning → split-and-conquer → fine-tuning 순서로 구성된다.
6. Accelerator는 Denser/Sparser engines, encoder/decoder engines, dedicated buffers, SoftMax/activation units, CSC index format, query-based Q forwarding을 포함한다.
7. Algorithm-hardware interface는 PyTorch model을 parser/compiler로 전달하여 SDDMM/SpMM/FC partition, global tokens, Q/K/V/S/H/F 등을 추출하고 hardware parameters와 runtime instructions를 생성한다.

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

- Models: DeiT-Base/Small/Tiny, LeViT-128/192/256, Strided Transformer.
- Datasets: ImageNet for DeiT/LeViT, Human3.6M for Strided Transformer.
- Baselines: CPU Intel Xeon Gold 6230R, EdgeGPU Nvidia Jetson Xavier NX, GPU Nvidia 2080Ti, SpAtten, Sanger.
- Metrics: latency speedup, energy efficiency, attention sparsity, model accuracy.
- Hardware: 3 mm², DDR4-2400 76.8GB/s, 323.9mW, 500MHz, 320KB SRAM, 512 MACs, commercial 28nm CMOS.
- Evaluation: cycle-accurate simulator with MAC/memory cost from post-layout simulation, verified against RTL.

## 6.2 Quantitative Results

| Dataset / Platform | Task | Metric | Proposed Method | Baseline / Previous SOTA | Improvement / Meaning | Source |
|---|---|---|---|---|---|---|
| Attention layers / 90% sparsity | Core attention acceleration | Speedup | 235.3× vs CPU, 142.9×/160.6× vs EdgeGPU, 86.0× vs GPU | General platforms | Large speedup | Figure 15, 19 |
| Attention layers / 90% sparsity | Accelerator comparison | Speedup | 10.1× vs SpAtten, 6.8× vs Sanger | Similar hardware configuration/area simulation | Better sparse ViT-specific acceleration | Section VI-B |
| 60/70/80/90% sparsity average | Accelerator comparison | Speedup | 127.2×, 77.0×, 46.5×, 6.8×, 4.3× over CPU/EdgeGPU/GPU/SpAtten/Sanger | Multiple sparsity settings | Consistent benefit | Section VI-D |
| ImageNet / DeiT, LeViT | Accuracy-latency trade-off | Accuracy drop | <1% for 90% DeiT and 80% LeViT sparsity | Unpruned baseline | Negligible accuracy drop | Section VI-C |
| ViTCoD vs Sanger | Energy efficiency | Energy efficiency improvement | 9.8× | Most competitive baseline Sanger | +9.8× | Figure 19 |

## 6.3 Result Interpretation

- Figure 4 profiling은 SA module이 FLOPs보다 latency에서 더 큰 병목임을 보여주며, SA-MatMul과 reshape/split이 EdgeGPU latency의 큰 부분을 차지한다.
- Split-and-conquer의 pruning은 sparse part를 더 sparse하게 만들어 polarization을 강화하고, reordering은 pattern regularity를 높인다. 논문은 pruning only/reordering only breakdown으로 각각의 효과를 분리했다.
- AE module은 Q/K movement를 줄이며, latency breakdown에서 data movement 비중을 50%에서 28%로 낮춘다.
- ViTCoD는 NLP model에 static sparse pattern을 강제할 경우 accuracy degradation이 발생한다고 명시한다. 따라서 적용 범위는 ViT-like fixed-token vision workload에 더 적합하다.

## 6.4 Trade-off Analysis

- **Accuracy vs Efficiency:** Low-bit quantization, sparse pruning, approximation, hardware specialization은 대부분 accuracy 또는 generality와 efficiency 사이 trade-off를 만든다.
- **Compute vs Memory:** Transformer acceleration에서 compute reduction만으로 충분하지 않으며, activation/scale/index movement와 off-chip traffic이 자주 지배적이다.
- **Generality vs Specialization:** GPU kernel 방식은 비교적 범용적이고 FPGA/ASIC co-design 방식은 특정 model/workload에 더 높은 효율을 낸다.
- **Design-space implication:** 실제 활용에는 bitwidth, scale granularity, tile size, head count, sequence length, buffer size, bandwidth, target device를 함께 탐색해야 한다.

---

# 7. Summary

## 7.1 One-paragraph Summary

ViTCoD는 ViT attention의 static sparsity 가능성을 이용해 attention map을 denser/sparser 두 workload로 구조화하고, Q/K auto-encoder로 data movement를 줄이는 co-design이다. Hardware는 두 workload를 전용 engine으로 분리하고 K-stationary/output-stationary dataflow를 조합한다. 논문은 90% sparsity 조건에서 기존 attention accelerators와 general platforms 대비 큰 speedup과 energy efficiency 향상을 보인다.

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

- ViT와 NLP Transformer의 sparsity 구조 차이를 명확히 구분하고 hardware design에 반영했다.
- Algorithm-level sparsification, representation compression, accelerator dataflow를 통합한 강한 co-design이다.
- Cycle simulator와 RTL verification, post-layout cost를 포함하여 hardware evaluation 근거가 비교적 탄탄하다.
- Pruning, reordering, AE module의 ablation을 통해 speedup 원인을 분리했다.
- Compiler/parser interface까지 제시하여 deployment path가 구체적이다.

---

# 9. Cons.

- Sparse masks가 fixed라는 가정은 classification-style ViT에는 적합하지만 dynamic-resolution/detection/segmentation transformer에는 별도 검증이 필요하다.
- AE module insertion과 fine-tuning이 필요하므로 pretrained model을 바로 가속하는 PTQ형 접근은 아니다.
- 비교 대상 accelerator를 논문에서 재구현/시뮬레이션한 것이므로 실제 chip-to-chip 측정 비교는 아니다.
- Softmax/non-linear function approximation 정확도와 numerical robustness는 상대적으로 덜 강조된다.
- NLP model에서 static sparsity가 성능 하락을 만든다는 점은 범용 Transformer accelerator로는 한계다.

---

## Self-check

- 9개 필수 섹션 포함 여부: 완료.
- 논문 근거 기반 작성 여부: 제공 PDF 내용 기반.
- 실험/코드/합성 재현 여부: 수행하지 않음.
- 직접 비교 주의사항 표기 여부: 포함.
