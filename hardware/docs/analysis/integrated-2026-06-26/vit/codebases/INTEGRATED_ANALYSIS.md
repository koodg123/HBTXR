# Integrated Codebase Analysis: References/ViT

Source directory: `/home/kjm26/project/PRJXR/References/ViT`

## 1. Corpus Summary

| Metric | Value |
|---|---:|
| Top-level repositories | 10 |
| Primary focus | ViT variants, quantization, mobile deployment, MoE, operator replacement |
| Direct RTL/HLS content | Low |
| Direct HGTXR value | Algorithm and quantization experiment design; deployment metrics; operator replacement references |

## 2. Top-Level Repository Coverage

| Repository | Category | Key files inspected or inventoried | HGTXR relevance | Priority |
|---|---|---|---|---|
| `AMD_QTViT` | EfficientViT/ImageNet training and evaluation | `README.md`, `efficientvit/*`, `train_cls_model.py`, `eval_cls_model.py` | EfficientViT-style lightweight backbone reference | P2 |
| `Castling-ViT` | Attention replacement experiment | `README.md`, `attention.py` | Minimal attention variant reference; low implementation depth | P3 |
| `FQ-ViT` | PTQ/quantization | `config.py`, `models/vit_quant.py`, `models/swin_quant.py`, `models/ptq/*` | Calibration and fixed-point accuracy study | P1 |
| `M3ViT` | MoE and hardware-aware ViT | `models/vision_transformer_moe.py`, `models/custom_moe_layer.py`, `models/vits_gate.py` | Search/track mode-aware conditional compute candidate | P1 |
| `Next-ViT` | Mobile/edge deployment | `README.md`, `deployment/*`, `classification/`, `detection/`, `segmentation/` | Edge latency/deployment baseline, model variant candidates | P1 |
| `P2-ViT` | PoT quantization | `models/vit_fquant.py`, `models/ptq/*`, `test_quant.py` | Most direct quantization path for HGTXR fixed-point scale policy | P0 |
| `PTQ4ViT` | PTQ calibration and layer replacement | `configs/*`, `quant_layers/*`, `example/*` | Calibration procedure and bit-width ablation | P1 |
| `RepViT` | Mobile CNN/ViT hybrid deployment | `README.md`, `export_coreml.py`, `main.py`, `sam/`, `detection/`, `segmentation/` | Lightweight front-end or local feature extractor reference | P1 |
| `ShiftAddViT` | Shift/add operator replacement and TVM/CUDA kernels | `pvt/hw_utils.py`, `OPs_Speedups/*`, custom kernels | Operator-level cost study; use cautiously with DSP policy | P2 |
| `ViTALiTy` | Linear/Taylor attention and hardware-aware ViT | `src/vision_transformer.py`, `src/models.py`, `src/quant_utils.py` | Attention approximation ablation with explicit error reporting | P1/P2 |

## 3. Detailed Technical Themes

### 3.1 Quantization and Calibration

| Source | Proposed method class | Concrete HGTXR conversion |
|---|---|---|
| `P2-ViT` | Power-of-two and fixed-point friendly quantization | Add PoT scale export and bit-accurate scale propagation report |
| `FQ-ViT` | PTQ with ViT/Swin quant modules | Reuse calibration structure to report pre/post quantization accuracy |
| `PTQ4ViT` | Post-training calibration and quant layer wrappers | Create calibration artifact contract: weights, scales, activation ranges, seed, dataset subset |

Expected result: better scale handling and lower requantization cost without adding LUT-heavy arithmetic. Required gate: accuracy before/after quantization plus HLS bit-accurate simulation.

### 3.2 Mobile/Deployment Backbones

| Source | Proposed method class | Concrete HGTXR conversion |
|---|---|---|
| `RepViT` | Reparameterized efficient vision backbone | Evaluate lightweight local feature extractor before transformer blocks |
| `Next-ViT` | Hybrid mobile ViT deployment | Use latency/deployment structure for model-size comparisons |
| `AMD_QTViT` | EfficientViT-style training/evaluation | Candidate for conv-transformer hybrid accuracy ablation |

Expected result: potential accuracy improvement or smaller token workload. Hardware risk: conv/local blocks may add memory layout complexity; keep software-first until accuracy justifies HLS work.

### 3.3 Conditional Compute and Routing

| Source | Proposed method class | Concrete HGTXR conversion |
|---|---|---|
| `M3ViT` | MoE and hardware-aware routing | Map search/track mode selection to fixed routing policy first |
| `Castling-ViT` | Attention replacement | Use only as a small reference for attention API shape |

Expected result: lower average work if search/track distribution is skewed. Gate: invocation distribution and worst-case path latency.

### 3.4 Operator Replacement and Approximation

| Source | Proposed method class | Concrete HGTXR conversion |
|---|---|---|
| `ShiftAddViT` | Shift/add operator substitution | Benchmark against DSP-bound MAC baseline; do not default to LUT arithmetic |
| `ViTALiTy` | Linear/Taylor attention | Add exact-attention fallback and approximation-error table |

Expected result: possible memory/latency reduction. Risk: approximation can reduce accuracy or move arithmetic into LUTs, which conflicts with current resource objective.

## 4. Recommended Experiment List

| ID | Experiment | Source | Expected result | Required evidence |
|---|---|---|---|---|
| VIT-Q0 | PoT scale export and fixed-point calibration | `P2-ViT` | Lower requantization cost, stable accuracy | SW accuracy, scale manifest, bit-accurate sim |
| VIT-Q1 | PTQ calibration comparison | `FQ-ViT`, `PTQ4ViT` | Identify best calibration method for HGTXR | mean/median/p95 error, subject/motion split if dataset available |
| VIT-A0 | Linear/Taylor attention ablation | `ViTALiTy` | Attention memory reduction candidate | approximation error and exact fallback |
| VIT-M0 | Search/track fixed routing | `M3ViT` | Lower average compute under mode skew | invocation distribution and p95/p99 latency |
| VIT-B0 | Lightweight local backbone ablation | `RepViT`, `Next-ViT`, `AMD_QTViT` | Accuracy/resource tradeoff candidate | accuracy, token count, latency estimate |

## 5. Rejection and Caution Cases

| Case | Reason |
|---|---|
| Directly replacing MACs with shift/add without evidence | Current objective is higher DSP utilization and lower LUT pressure |
| Treating mobile CPU/GPU latency as FPGA throughput | Different hardware targets and memory hierarchy |
| Importing dynamic MoE routing directly into HLS | Control complexity and worst-case latency risk |
| Claiming accuracy improvements without HGTXR dataset rerun | Reference corpora are not HGTXR-specific results |

