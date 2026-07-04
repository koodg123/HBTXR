# Transformer Quantization / ViT Accelerator Markdown Analysis Index

## Package Summary

| Item | Description |
|---|---|
| Generated at | 2026-06-23T02:44:06 |
| Number of papers | 7 |
| Output format | Markdown reports + ZIP |
| Template basis | Paper-Review-Summary-Prompt.md / Paper-Review-Summary-Template.md |
| Reproduction scope | No code execution, no model training, no FPGA/GPU synthesis or board run |

## Included Reports

| No. | Paper | Research Field | Markdown File | Coverage |
|---|---|---|---|---|
| 01 | Genetic Quantization-Aware Approximation for Non-Linear Operations in Transformers | Transformer quantization, non-linear operator approximation, LUT-based hardware co-design | 01_GQA_LUT_Genetic_Quantization_Aware_Approximation.md | Main paper 전체 6 pages. Appendix 없음. 코드 실행 및 RTL 재합성은 수행하지 않음. |
| 02 | Jetfire: Efficient and Accurate Transformer Pretraining with INT8 Data Flow and Per-Block Quantization | Fully Quantized Training (FQT), Transformer pretraining, INT8 GPU kernels | 02_Jetfire_INT8_Data_Flow_Per_Block_Quantization.md | Main paper + Appendix 포함 PDF 15 pages. 코드 실행 및 GPU benchmark 재현은 수행하지 않음. |
| 03 | ViTCoD: Vision Transformer Acceleration via Dedicated Algorithm and Accelerator Co-Design | Vision Transformer sparse attention acceleration, algorithm-hardware co-design | 03_ViTCoD_Vision_Transformer_Codesign.md | Main paper 전체 14 pages. 코드 실행 및 RTL/cycle simulator 재현은 수행하지 않음. |
| 04 | An Integer-Only and Group-Vector Systolic Accelerator for Efficiently Mapping Vision Transformer on Edge | Integer-only ViT inference, FPGA accelerator, systolic array | 04_Integer_Only_Group_Vector_Systolic_Accelerator.md | Main paper 전체 13 pages. FPGA bitstream/RTL 재합성 및 보드 실행은 수행하지 않음. |
| 05 | Integer-only Quantized Transformers for Embedded FPGA-based Time-series Forecasting in AIoT | AIoT, embedded FPGA, time-series forecasting, integer-only Transformer | 05_Integer_Only_Quantized_Transformers_AIoT_Time_Series.md | Main paper 전체 7 pages. VHDL generation, GHDL simulation, Vivado synthesis 및 hardware validation은 논문 결과만 분석하고 재실행하지 않음. |
| 06 | Systolic Array-based Architecture for Low-Bit Integerized Vision Transformers | Low-bit integerized ViT, FPGA accelerator, systolic array, MSA acceleration | 06_Systolic_Array_Low_Bit_Integerized_ViT.md | Main paper 전체 14 pages. FPGA synthesis 결과는 논문 수치 분석만 수행하고 Vivado 재실행은 수행하지 않음. |
| 07 | Towards Fully 8-bit Integer Inference for the Transformer Model | Integer-only Transformer inference, NLP quantization | 07_Towards_Fully_8bit_Integer_Inference_Transformer.md | Main paper 전체 7 pages. 실제 CPU INT8 runtime 재현은 수행하지 않음. Speed-up은 논문에서 estimated value로 제시됨. |

## Cross-paper Technical Map

| Axis | Relevant Papers | Notes |
|---|---|---|
| Non-linear operation approximation | GQA-LUT, Integer Transformer, Group-vector systolic, AIoT Transformer, Low-bit systolic ViT | Softmax, GELU, LayerNorm, RSQRT, DIV 등의 integer/LUT/polynomial approximation |
| INT8 / low-bit data flow | Jetfire, Integer Transformer, AIoT Transformer, Low-bit systolic ViT | Scale propagation, per-block quantization, QAT, 3-bit integerization |
| ViT acceleration | ViTCoD, Group-vector systolic, Low-bit systolic ViT | Sparse attention, systolic array, FPGA/ASIC-style accelerator |
| FPGA implementation | Group-vector systolic, AIoT Transformer, Low-bit systolic ViT | ZCU102, Spartan-7, Alveo U250 |
| GPU kernel implementation | Jetfire | CUDA linear operator, Triton non-linear operator |
| Algorithm-hardware co-design | GQA-LUT, ViTCoD, Group-vector systolic, AIoT Transformer, Low-bit systolic ViT | Algorithmic approximation or sparsity directly matched to hardware structure |

## Recommended Reading Order

1. `07_Towards_Fully_8bit_Integer_Inference_Transformer.md` — early conceptual basis for scale propagation and INT8-compatible Transformer architecture.
2. `01_GQA_LUT_Genetic_Quantization_Aware_Approximation.md` — non-linear approximation and LUT hardware cost.
3. `02_Jetfire_INT8_Data_Flow_Per_Block_Quantization.md` — training-side INT8 data flow and GPU kernels.
4. `04_Integer_Only_Group_Vector_Systolic_Accelerator.md` — FPGA full-network INT8 ViT mapping.
5. `03_ViTCoD_Vision_Transformer_Codesign.md` — sparse ViT-specific co-design.
6. `06_Systolic_Array_Low_Bit_Integerized_ViT.md` — 3-bit systolic MSA accelerator and operational intensity.
7. `05_Integer_Only_Quantized_Transformers_AIoT_Time_Series.md` — embedded AIoT time-series deployment case.

## Notes

- 이 패키지는 논문 분석용 문서이며, 실험 재현 결과가 아니다.
- 모든 논문별 Markdown 문서는 9개 필수 섹션을 포함한다.
- 수치 비교는 논문별 실험 조건이 다르므로, cross-paper ranking에는 주의가 필요하다.
