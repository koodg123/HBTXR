---
source_type: codebase
source_name: CoQMoE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/CoQMoE/analysis.md -->

# CoQMoE Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE`
- repo_remote: `origin	https://github.com/DJ000011/CoQMoE (fetch)`
- category: `MoE-ViT quantization and FPGA orchestration`
- HGTXR relevance: `high`
- matched_paper: `CoQMoE`
- paper_title: CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/CoQMoE.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `6` |
| 주요 언어 | `Python:4, Markdown:2` |
| LOC 추정 | `Python:710, Markdown:21` |

### Directory Map
- `Quant_algorithm/` (5 entries)

### Metadata / Config Refs
- `README.md`
- `Quant_algorithm/README.md`

### Core Source Refs
- `Quant_algorithm/quantizer.py`: 260 lines
- `Quant_algorithm/quant_modules.py`: 256 lines
- `Quant_algorithm/quant_model.py`: 191 lines
- `README.md`: 16 lines
- `Quant_algorithm/README.md`: 5 lines
- `Quant_algorithm/__init__.py`: 3 lines

### Hardware-Oriented Source Refs
- no obvious HLS/RTL source found in first scan

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/README.md`
- ## CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA
- ## overview
- Compared to UbiMoE, CoQMoE introduces a co-designed quantization and computation orchestration strategy that achieves a better trade-off between performance and resource utilization.
- ## Environment
- - **Ubuntu 20.04**
- - **Vitis**, **XRT** (Xilinx Runtime) and **XCU280 platform** 2022.2 [link](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/alveo/u280.html)
- - **Model and Dataset** : We use the [$M^3ViT$](https://github.com/VITA-Group/M3ViT) as our model and evaluate on [Cityscape Dataset](https://www.cityscapes-dataset.com/)
- ## Compile and Run

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: precision-preserving complex quantizer와 hardware-friendly simplified quantizer를 결합하고, resource-aware accelerator에서 streaming attention과 reusable linear operator를 조합한다.
- 알고리즘 축: scale reparameterization 기반 dual-stage quantization과 expert computation orchestration으로 MoE token/expert workload를 정렬한다.
- 하드웨어 축: streaming attention kernel, reusable linear operator, expert scheduling buffer가 핵심이다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/README.md:6:This work is based on our previous work *[UbiMoE: a Ubiquitous Mixture-of-Experts Vision Transformer Accelerator with Hybrid Computation Pattern on FPGA](https://arxiv.org/abs/2502.05602)*, which implements a fully streaming attention kernel optimized for latency and a reusable linear kernel optimized for resource efficiency.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:4:from utils.build_model import MatMul, GetEx, SoftMax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:5:from .quant_modules import QuantConv2d, QuantLinear, QuantMatMul, QuantGetex, QuantLayerNorm, QuantSoftmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:10:    # post-softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:11:    input_quant_params_matmul2 = deepcopy(input_quant_params)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:12:    input_quant_params_matmul2['log_quant'] = True`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:13:    input_quant_params_matmul2['exp_quant'] = True`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:15:    # softmax-ex`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:65:        elif isinstance(m, MatMul):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:66:            # Matmul Layer`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:68:            if 'matmul2' in name:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:69:                new_m = QuantMatMul(input_quant_params_matmul2, sym)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:71:                new_m = QuantMatMul(input_quant_params, sym)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:84:    # post-softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:85:    input_quant_params_matmul2 = deepcopy(input_quant_params)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:86:    input_quant_params_matmul2['log_quant'] = False`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:87:    #input_quant_params_matmul2['exp_quant'] = False`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:89:    # softmax-ex`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:140:        elif isinstance(m, MatMul):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quant_model.py:141:            # Matmul Layer`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/README.md:1:## CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/README.md:4:This repository contains the implementation of Python codes for our paper *[CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA](https://arxiv.org/abs/2506.08496)*, which focuses on the quantization and computation orchestration of Mixture-of-Experts (MoE) Vision Transformers (ViTs) on FPGA. `
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/README.md:8:Compared to UbiMoE, CoQMoE introduces a co-designed quantization and computation orchestration strategy that achieves a better trade-off between performance and resource utilization.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/README.md:1:## CoQMoE: Co-Designed Quantization and Computation Orchestration for Mixture-of-Experts Vision Transformer on FPGA`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/README.md:3:Below are the implementation codes for our paper, which focuses on the quantization and computation orchestration of Mixture-of-Experts (MoE) Vision Transformers (ViTs) on FPGA.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:18:class UniformQuantizer(nn.Module):`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:20:    PyTorch Function that can be used for asymmetric quantization (also called uniform affine`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:21:    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:22:    through' on the backward pass, ignoring the quantization that occurred.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:24:    :param n_bits: number of bit for quantization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:28:        super(UniformQuantizer, self).__init__()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:39:        s = super(UniformQuantizer, self).__repr__()`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:50:                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:53:        # start quantization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/CoQMoE/Quant_algorithm/quantizer.py:55:        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)`

## 5. HGTXR 적용 해석
- search/track mode별 expert/head 분기 실험에 적합하다. 단, routing은 정적이고 bounded여야 HGTXR paper scope를 지킨다.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation
- Search/Track mode-specific expert or head routing
- Attention / Softmax / token sparsity ablation
- HLS/Vivado/PYNQ integration and resource-flow references

## 6. 실험 옵션으로 변환할 때의 규칙
- 먼저 HGTXR-SW 또는 C++ reference에서 accuracy/bit-exact behavior를 검증한다.
- HLS에 반영할 때는 C3b signoff path를 덮어쓰지 말고 새로운 suffix variant로 추가한다.
- 완료된 `csynth.xml`, routed timing/power, PYNQ smoke JSON 없이는 resource matrix의 완료 variant로 승격하지 않는다.
- HGTXR 논문 범위를 벗어나는 구조 변경은 `paper_scope_review_required`로 둔다.

## 7. 리스크와 비적용 조건
- 동적 MoE routing, softmax-free attention, ternary/LUT-heavy compute는 정확도 또는 resource 정책과 충돌할 수 있다.
- 현재 사용자의 resource 방향은 DSP/URAM 활용 증가와 LUT 과사용 억제이므로 LUT-LLM/LUT-GEMM류는 기본값이 아니라 negative-control이다.
- codebase가 LLM 중심이면 HGTXR eye-tracking 적용은 kernel/system-flow 수준으로 제한한다.

## 8. 다음 분석/구현 액션
- 핵심 source file을 1개 선택해 line-by-line 분석을 수행한다.
- 해당 방법을 HGTXR `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml`의 planned experiment와 연결한다.
- SW accuracy metric과 HW resource metric을 같은 manifest에 기록한다.
