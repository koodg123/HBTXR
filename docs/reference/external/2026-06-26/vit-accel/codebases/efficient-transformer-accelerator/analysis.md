---
source_type: codebase
source_name: efficient-transformer-accelerator
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/efficient-transformer-accelerator/analysis.md -->

# efficient-transformer-accelerator Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator`
- repo_remote: `not_available`
- category: `EfficientViT/linear-attention accelerator prototype`
- HGTXR relevance: `medium`
- matched_paper: `EfficientViT-FPGA`
- paper_title: An FPGA-Based Reconfigurable Accelerator for Convolution-Transformer Hybrid EfficientViT
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/EfficientViT-FPGA.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `66` |
| 주요 언어 | `Python:19, SystemVerilog:13, Markdown:7, .png:7, Tcl:5, .pth:4, no_ext:3, .txt:3, .log:2, Shell:1` |
| LOC 추정 | `Python:4219, SystemVerilog:1947, Markdown:1274, .txt:625, Tcl:604, Shell:262` |

### Directory Map
- `ViTALiTy/` (8 entries)
- `ViTALiTy/.github/` (6 entries)
- `ViTALiTy/figures/` (3 entries)
- `ViTALiTy/logs/` (2 entries)
- `ViTALiTy/src/` (18 entries)
- `hw/` (7 entries)
- `hw/constraints/` (2 entries)
- `hw/scripts/` (4 entries)
- `hw/src/` (10 entries)
- `hw/tb/` (4 entries)
- `models/` (3 entries)
- `models/degree_1_train/` (3 entries)
- `models/degree_2_quant/` (1 entries)
- `models/degree_2_train/` (3 entries)

### Metadata / Config Refs
- `readme.md`
- `ViTALiTy/README.md`
- `hw/README_VIVADO.md`
- `hw/tb/debug_signals.tcl`
- `hw/scripts/vivado_synth.tcl`
- `hw/scripts/vivado_sim.tcl`
- `hw/scripts/vivado_flow.tcl`
- `hw/scripts/vivado_impl.tcl`
- `hw/constraints/timing.xdc`

### Core Source Refs
- `ViTALiTy/src/quantize.py`: 589 lines
- `ViTALiTy/src/quant_utils.py`: 482 lines
- `ViTALiTy/src/main.py`: 462 lines
- `hw/README_VIVADO.md`: 460 lines
- `ViTALiTy/src/vision_transformer.py`: 439 lines
- `hw/IMPLEMENTATION_GUIDE.md`: 405 lines
- `hw/tb/systolic_quant_tb.sv`: 376 lines
- `ViTALiTy/src/quantize_model.py`: 371 lines
- `ViTALiTy/src/patchconvnet_models.py`: 350 lines
- `hw/tb/systolic_32x16_tb.sv`: 304 lines

### Hardware-Oriented Source Refs
- `hw/tb/systolic_quant_tb.sv`
- `hw/tb/debug_signals.tcl`
- `hw/tb/systolic_32x16_tb.sv`
- `hw/tb/systolic_tb.sv`
- `hw/src/systolic_quant_32x16.sv`
- `hw/src/quant_shared.sv`
- `hw/src/quant.sv`
- `hw/src/quant_top.sv`
- `hw/src/accumulator_bank.sv`
- `hw/src/pe.sv`
- `hw/src/systolic_top.sv`
- `hw/src/systolic_mac.sv`
- `hw/src/systolic_mac_rect.sv`
- `hw/src/systolic_quant_integrated.sv`
- `hw/scripts/vivado_synth.tcl`
- `hw/scripts/vivado_sim.tcl`
- `hw/scripts/vivado_flow.tcl`
- `hw/scripts/vivado_impl.tcl`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/readme.md`
- # Glide Accelerator
- Low-power hardware accelerator for efficient Vision Transformer inference using linear attention approximation and systolic array architecture.
- ## Architecture
- - **Systolic Array**: 32×16 processing elements (512 MACs)
- - **Precision**: INT8 quantization with 32-bit accumulation
- - **Attention Mechanism**: Taylor-series approximated softmax (degree-1 & degree-2)
- - **Throughput**: 512 MACs/cycle @ 200 MHz
- - **Resource Efficiency**: Shared quantization units (64 units time-multiplexed)
- ## Performance
- | Metric | Value |
- |--------|-------|
- | Target Frequency | 200 MHz |
- | Peak Throughput | 102.4 GOPS |
- | LUT Utilization | ~36% |
- | DSP Utilization | ~15% |
- | Latency (End-to-End) | ~61 cycles |
- ## Quick Start
- ### Simulation
- ```bash
- cd hw

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: operation type별 reconfigurable architecture와 time-multiplexed pipelined dataflow를 제안한다.
- 알고리즘 축: lightweight convolution과 attention을 layer fusion 관점에서 배치한다.
- 하드웨어 축: ZCU102에서 intra/inter-layer fusion, shared datapath, quantized execution으로 구성된다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/PRODUCTION_RELEASE_NOTES.md:53:- **Systolic Latency**: 48 cycles (ROWS + COLS pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/PRODUCTION_RELEASE_NOTES.md:54:- **Quantization Latency**: 13 cycles (8 batches + 4 pipeline + 1 done)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/PRODUCTION_RELEASE_NOTES.md:129:✅ pe.sv                      - Processing element (3-stage pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/PRODUCTION_RELEASE_NOTES.md:132:✅ quant.sv                   - Single quantization unit (4-stage pipeline)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/constraints/synthesis_asic.sdc:126:# The systolic array has pipelined datapath`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/constraints/timing.xdc:108:#   Aggressive:   3.0 ns  (333 MHz)  - May require pipeline optimization`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/constraints/timing.xdc:115:#   4. If WNS < 0, timing failed - reduce frequency or add pipeline stages`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/scripts/vivado_impl.tcl:142:        puts "  - Add pipeline stages to critical paths"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/README_VIVADO.md:287:1. **Add pipeline stages** - Break long combinational paths`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/README_VIVADO.md:358:If **WNS < 0**: ❌ Setup timing failed - Reduce clock frequency or add pipeline stages`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/IMPLEMENTATION_GUIDE.md:203:#### Strategy 3: Add Pipeline Stages (Best long-term)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/IMPLEMENTATION_GUIDE.md:207:3. Example: Add pipeline stage in accumulator or requantization module`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/IMPLEMENTATION_GUIDE.md:358:3. Add pipeline stages in critical paths`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/quant.sv:13:    // Pipeline stage 1: Input registration`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/quant.sv:18:    // Pipeline stage 2: Multiplication`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/quant.sv:23:    // Pipeline stage 3: Shift operation`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/quant.sv:27:    // Pipeline stage 4: Clamping`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/pe.sv:22:// - Pipeline MAC operation for better timing`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/pe.sv:42:    // Internal registers for pipelined multiply`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/systolic_mac.sv:25:    // Input pipeline registers for proper timing`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/readme.md:3:Low-power hardware accelerator for efficient Vision Transformer inference using linear attention approximation and systolic array architecture.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/readme.md:9:- **Attention Mechanism**: Taylor-series approximated softmax (degree-1 & degree-2)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/src/systolic_quant_32x16.sv:14://   Throughput: 32×16×K MatMul in K+~40 cycles`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/README_VIVADO.md:79:Test 2: Multi-Pass Accumulation (Simulating Tiled MatMul)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/tb/systolic_quant_tb.sv:187:        // Test 2: Multi-Pass Accumulation (Tiled MatMul)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/hw/tb/systolic_quant_tb.sv:189:        $display("\n\nTest 2: Multi-Pass Accumulation (Simulating Tiled MatMul)");`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:2:# ViTALiTy: Unifying Low-rank and Sparse Approximation for Vision Transformer Acceleration with a Linear Taylor Attention`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:19:* ***On the algorithm level***, we propose a linear attention for reducing the computational and memory cost by decoupling the vanilla softmax attention into its corresponding “weak” and “strong” Taylor attention maps. Unlike the vanilla attentions, the linear attention in ViTALiTy generates a global context matrix G by multiplying the keys with the values. Then, we unify the low-rank property of the linear attention with a sparse approximation of “strong” attention for training the ViT model. Here, the low-rank component of our ViTALiTy attention captures global information with a linear complexity, while the sparse component boosts the accuracy of linear attention model by enhancing its local feature extraction capacity.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:25:Fig.1 - ViTALiTy workflow comprising the proposed (Low-Rank) Linear Taylor attention (order, m = 1): (i) Higher-order Taylor terms (m > 1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:26:when added results in vanilla softmax attention score, (ii) Training phase (unifying low-rank and sparse approximation) where higher-order Taylor terms are approximated as Sparse attention (computed using SANGER [28]), and (iii) Inference phase that uses only the (Low-Rank) Linear Taylor attention.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:31:<img src="./figures/TaylorAttentionFlow2.png" width="400">`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:34:Fig.2 - Computational steps (a) vanilla Softmax Attention and (b) our Taylor attention (see Algorithm 1), where the global context matrix G provides linear computation and memory benefits over the vanilla quadratic QK^T.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:37:* ***On the hardware level***, we develop a dedicated accelerator to better leverage the algorithmic properties of ViTALiTy’s linear attention, where only a low-rank component is executed during inference favoring hardware efficiency. Specifically, ViTALiTy's accelerator features a chunk-based design integrating both a systolic array tailored for matrix multiplications and pre/post-processors customized for ViTALiTy attentions’ pre/post-processing steps. Furthermore, we adopt an intra-layer pipeline design to leverage the intra-layer data dependency for enhancing the overall throughput together with a down-forward accumulation dataflow for the systolic array to improve hardware efficiency.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:50:### Training (DeiT-Tiny with vanilla softmax)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/efficient-transformer-accelerator/ViTALiTy/README.md:56:### Inference (DeiT-Tiny with vanilla softmax)`

## 5. HGTXR 적용 해석
- hybrid patch stem이나 lightweight local feature extractor를 붙이는 accuracy 실험에 유용하지만 paper-scope 검토 필요.

### 적용 가능 모듈
- Quantization / scale calibration / weight generation
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
