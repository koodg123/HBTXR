---
source_type: codebase
source_name: HG-PIPE
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/HG-PIPE/analysis.md -->

# HG-PIPE Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE`
- repo_remote: `origin	https://github.com/hguq/HG-PIPE (fetch)`
- category: `ViT hybrid-grained pipeline FPGA accelerator`
- HGTXR relevance: `high`
- matched_paper: `HG-PIPE`
- paper_title: HG-PIPE: Vision Transformer Acceleration with Hybrid-Grained Pipeline
- paper_pdf: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Papers/HG-PIPE.pdf`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `1207` |
| 주요 언어 | `.txt:1037, C++:36, .xml:31, Verilog:30, C/C++ header:15, Scala:15, Python:13, .png:12, no_ext:3, Tcl:2` |
| LOC 추정 | `Verilog:707998, C++:12965, .txt:2979, C/C++ header:2888, Tcl:1738, Python:1692, Scala:1290, Markdown:486, Notebook:261, JSON:1` |

### Directory Map
- `SPINAL/` (11 entries)
- `SPINAL/.bsp/` (1 entries)
- `SPINAL/.idea/` (11 entries)
- `SPINAL/latency/` (27 entries)
- `SPINAL/project/` (1 entries)
- `SPINAL/src/` (2 entries)
- `SPINAL/vivado/` (2 entries)
- `assets/` (12 entries)
- `case/` (40 entries)
- `case/refs/` (1009 entries)
- `notebooks/` (3 entries)
- `src/` (15 entries)
- `statistics/` (4 entries)

### Metadata / Config Refs
- `template.tcl`
- `VCK190-bd-base.tcl`
- `README.md`
- `README.zh-CN.md`
- `SPINAL/.bsp/sbt.json`

### Core Source Refs
- `SPINAL/src/main/verilog/MLP6/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP5/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP4/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP3/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP2/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP11/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP1/all.v`: 55494 lines
- `SPINAL/src/main/verilog/MLP0/all.v`: 55487 lines
- `SPINAL/src/main/verilog/MLP9/all.v`: 55461 lines
- `SPINAL/src/main/verilog/MLP7/all.v`: 55461 lines

### Hardware-Oriented Source Refs
- `template.tcl`
- `VCK190-bd-base.tcl`
- `case/ATTN8.cpp`
- `case/ATTN9.cpp`
- `case/HEAD_SPLIT.cpp`
- `case/ATTN0.cpp`
- `case/MLP8.cpp`
- `case/MLP6.cpp`
- `case/LAYERNORM_1X2.cpp`
- `case/ATTN7.cpp`
- `case/RESHAPER.cpp`
- `case/MLP3.cpp`
- `case/ATTN6.cpp`
- `case/ATTN11.cpp`
- `case/ATTN4.cpp`
- `case/SOFTMAX_2X2.cpp`
- `case/QUANT.cpp`
- `case/MLP2.cpp`
- `case/LAYERNORM_2X1.cpp`
- `case/LAYERNORM_2X2.cpp`
- `case/HEAD.cpp`
- `case/MLP11.cpp`
- `case/ATTN3.cpp`
- `case/SOFTMAX_1X2.cpp`
- `case/ATTN5.cpp`
- `case/ATTN2.cpp`
- `case/MLP7.cpp`
- `case/MLP5.cpp`
- `case/GELU.cpp`
- `case/MLP10.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.md`
- # HG-PIPE
- <!-- ![Build Status](https://img.shields.io/badge/build-passing-brightgreen) -->
- [English](README.md) | [中文](README.zh-CN.md)
- ---
- ## Accelerator Features
- <!-- Add a table -->
- | LUTs | DSPs | BRAMs | Frequency | FPS (ImageNet@224x224) | TOPs | GOPs/W | Accuracy |
- |:----:|:----:|:-----:|:---------:|:----------------------:|:----:|:------:|:--------:|
- |  669k|  312 | 1006.5| 425MHz    |7118                   | 17.8 |  381.0 |  71.05%  |
- ---
- ## Requirements
- - Vivado HLS 2020.1 or later (recommended: 2023.2 for faster compilation)
- - Python 3
- - IDEA + Scala (2.11.12) + Spinal (1.7.1) + Verilator (4.228)
- ## File Structure
- ```text
- HG-PIPE/
- ├── src/                    # HLS design files
- ├── statistics/             # Neural network data type statistics as template parameters
- ├── case/                   # Modules generated via case generation and component unit tests

## 4. Function / Dataflow 관점
- 논문 기준 핵심 방법: hybrid-grained pipeline으로 buffer cost를 줄이고 dataflow/parallelism을 결합해 bubble을 제거한다.
- 알고리즘 축: operator별 pipeline granularity와 parallelism을 조정하고 linear/nonlinear operator approximation을 사용한다.
- 하드웨어 축: ZCU102/VCK190 대상 pipelined FPGA accelerator이며 layer/operator instance를 spatial/temporal 혼합 배치한다.

### 핵심 모듈 근거
- `src/attn.h:340-365`는 residual stream merge를 `#pragma HLS pipeline II=1`과 unroll된 TP/CAP 루프로 처리한다. HGTXR의 DSP 활용률을 올리는 방향에서는 이 residual 경로를 LUT 조합 로직으로 키우지 말고, MAC-heavy stage의 parallelism 증가와 stream 폭 균형을 먼저 맞춰야 한다.
- `src/attn.h:368-460`의 `do_attn()`은 `#pragma HLS dataflow` 아래 residual, layernorm, qkv, qkv quant, head split, qk, softmax, reshape, rv, head merge, output projection stream을 연결한다. 이는 HGTXR attention pipeline 문서화 기준으로 가장 직접적인 참고 구조다.
- 같은 `do_attn()` 블록은 여러 `#pragma HLS stream variable=... depth=...`를 사용한다. Deep FIFO나 large intermediate는 URAM 후보, 짧은 control/token FIFO는 BRAM/LUTRAM 후보로 분리하는 현재 자원 정책과 연결된다.
- `case/ATTN*.cpp`, `case/MLP*.cpp`, `case/SOFTMAX_*.cpp`는 operator별 HLS project를 분리한다. HGTXR 실험도 C3b mainline을 덮지 말고 `PAR`, `MEM`, `FIFO_IMPL`, `SMALL_MEM_IMPL` suffix variant로 분리해야 한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP1.cpp:196:    #pragma HLS dataflow`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP1.cpp:197:    #pragma HLS interface ap_ctrl_chain port=return`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP1.cpp:198:    #pragma HLS interface axis          port=i_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP1.cpp:199:    #pragma HLS interface axis          port=o_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/SOFTMAX_2X1.cpp:87:    #pragma HLS interface axis          port=i_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/SOFTMAX_2X1.cpp:88:    #pragma HLS interface axis          port=o_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN1.cpp:550:    #pragma HLS dataflow`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN1.cpp:551:    #pragma HLS interface ap_ctrl_chain port=return`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN1.cpp:552:    #pragma HLS interface axis          port=i_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN1.cpp:553:    #pragma HLS interface axis          port=o_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP0.cpp:196:    #pragma HLS dataflow`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP0.cpp:197:    #pragma HLS interface ap_ctrl_chain port=return`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP0.cpp:198:    #pragma HLS interface axis          port=i_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP0.cpp:199:    #pragma HLS interface axis          port=o_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN10.cpp:550:    #pragma HLS dataflow`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN10.cpp:551:    #pragma HLS interface ap_ctrl_chain port=return`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN10.cpp:552:    #pragma HLS interface axis          port=i_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/ATTN10.cpp:553:    #pragma HLS interface axis          port=o_stream`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP9.cpp:196:    #pragma HLS dataflow`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/case/MLP9.cpp:197:    #pragma HLS interface ap_ctrl_chain port=return`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.zh-CN.md:35:│   ├── ATTN.cpp.template   # Attention模块的模板文件`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.zh-CN.md:37:│   ├── SOFTMAX_1X2.cpp     # Softmax组件的单元测试文件`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.zh-CN.md:38:│   ├── GELU.cpp            # GELU组件的单元测试文件`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.zh-CN.md:42:│   ├── proj_ATTN0          # Attention层项目（第0层）`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.zh-CN.md:43:│   ├── proj_ATTN1          # Attention层项目（第1层）`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.md:35:│   ├── ATTN.cpp.template   # Template file for the Attention module`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.md:37:│   ├── SOFTMAX_1X2.cpp     # Unit test file for the Softmax component`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.md:38:│   ├── GELU.cpp            # Unit test file for the GELU component`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.md:42:│   ├── proj_ATTN0          # Attention layer project (layer 0)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/README.md:43:│   ├── proj_ATTN1          # Attention layer project (layer 1)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/constants.py:12:NUM_BLOCKS = 24  # 24 blocks in DeiT-Tiny, 12 Attention blocks and 12 MLP blocks`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/constants.py:51:    "QK_MATMUL_ADPT",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/constants.py:52:    "QK_MATMUL_WIND",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/constants.py:53:    "QK_MATMUL_WGHT",`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/HG-PIPE/constants.py:54:    "QK_MATMUL_MACS",`

## 5. HGTXR 적용 해석
- 이미 legacy 근거로 쓰였고, 현 HGTXR은 HG-PIPE의 pipeline 교훈을 유지하되 LUT-heavy approximation은 DSP/URAM 정책으로 재해석한다.

### 적용 가능 모듈
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
