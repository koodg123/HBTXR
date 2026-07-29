---
source_type: codebase
source_name: transformer-hls-thesis
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/transformer-hls-thesis/analysis.md -->

# transformer-hls-thesis Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis`
- repo_remote: `not_available`
- category: `Transformer encoder HLS thesis implementation`
- HGTXR relevance: `medium`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `29` |
| 주요 언어 | `C/C++ header:11, C++:10, Python:5, no_ext:1, Markdown:1, .cfg:1` |
| LOC 추정 | `C++:1524, Python:1078, C/C++ header:780, Markdown:106` |

### Directory Map
- `config/` (1 entries)
- `hls/` (1 entries)
- `hls/src/` (21 entries)
- `python/` (5 entries)

### Metadata / Config Refs
- `README.md`

### Core Source Refs
- `hls/src/newDebugTb.cpp`: 456 lines
- `python/plotCometResults.py`: 379 lines
- `python/comet2.py`: 344 lines
- `hls/src/exp_lut_values.h`: 260 lines
- `hls/src/multi_head_attention.cpp`: 195 lines
- `python/embedding.py`: 138 lines
- `hls/src/transformer_layer.cpp`: 136 lines
- `hls/src/ffn_linear1.cpp`: 136 lines
- `hls/src/ffn_linear2.cpp`: 128 lines
- `hls/src/attention_types.h`: 128 lines

### Hardware-Oriented Source Refs
- `hls/src/newDebugTb.cpp`
- `hls/src/layer_norm.h`
- `hls/src/transformer_layer.h`
- `hls/src/multi_head_attention.cpp`
- `hls/src/ffn_linear2.h`
- `hls/src/layer_norm_cl.cpp`
- `hls/src/multi_head_attention.h`
- `hls/src/opus_mt_weights.h`
- `hls/src/ffn_linear1.h`
- `hls/src/transformer_layer.cpp`
- `hls/src/opus_mt_embeddings.h`
- `hls/src/attention_scores2.cpp`
- `hls/src/lut_values.h`
- `hls/src/attention_types.h`
- `hls/src/ffn_linear2.cpp`
- `hls/src/exp_lut_values.h`
- `hls/src/attn_output.cpp`
- `hls/src/softmax2.cpp`
- `hls/src/softmax2.h`
- `hls/src/matmul_qkv.cpp`
- `hls/src/ffn_linear1.cpp`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md`
- # Transformer Encoder HLS Implementation
- Master's thesis project: Hardware implementation of a Transformer encoder layer using Vivado HLS for machine translation (EN-ES).
- ## Repository Structure
- ```
- repo/
- ├── hls/
- │   └── src/                    # Core HLS C++ kernels (21 files)
- │       ├── transformer_layer.cpp/.h    # Top-level transformer layer
- │       ├── multi_head_attention.cpp/.h # Multi-head attention module
- │       ├── attention_scores2.cpp       # Attention score computation
- │       ├── attn_output.cpp             # Attention output projection
- │       ├── matmul_qkv.cpp              # Q/K/V matrix multiplications
- │       ├── softmax2.cpp/.h             # Softmax with fixed-point LUT
- │       ├── ffn_linear1.cpp/.h          # FFN first linear layer
- │       ├── ffn_linear2.cpp/.h          # FFN second linear layer
- │       ├── layer_norm_cl.cpp           # Layer normalization
- │       ├── layer_norm.h                # Layer norm header
- │       ├── attention_types.h           # Data types and fixed-point definitions
- │       ├── lut_values.h                # Lookup table values
- │       ├── exp_lut_values.h            # Exponential LUT for softmax

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:13:        #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:30:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:46:    #pragma HLS ALLOCATION operation instances=mul limit=64`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:50:        #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:79:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:94:    #pragma HLS INTERFACE mode=m_axi port=input_global bundle=gmem0`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:95:    #pragma HLS INTERFACE mode=m_axi port=weights_global bundle=gmem1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:96:    #pragma HLS INTERFACE mode=m_axi port=bias_global bundle=gmem2`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:97:    #pragma HLS INTERFACE mode=m_axi port=output_global bundle=gmem3`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:101:    #pragma HLS ARRAY_PARTITION variable=input_local dim=2 cyclic factor=D_UNROLL_FACTOR`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:104:    #pragma HLS ARRAY_PARTITION variable=weights_local dim=1 cyclic factor=D_UNROLL_FACTOR`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/ffn_linear1.cpp:114:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:12:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:24:        #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:38:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:52:    #pragma HLS ALLOCATION operation instances=mul limit=32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:58:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:80:            #pragma HLS PIPELINE II=1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:93:    #pragma HLS INTERFACE mode=m_axi port=weights_global bundle=gmem0`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/hls/src/matmul_qkv.cpp:94:    #pragma HLS INTERFACE mode=m_axi port=bias_global bundle=gmem1 // <--- NEW BUNDLE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:12:│       ├── multi_head_attention.cpp/.h # Multi-head attention module`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:13:│       ├── attention_scores2.cpp       # Attention score computation`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:14:│       ├── attn_output.cpp             # Attention output projection`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:15:│       ├── matmul_qkv.cpp              # Q/K/V matrix multiplications`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:16:│       ├── softmax2.cpp/.h             # Softmax with fixed-point LUT`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:21:│       ├── attention_types.h           # Data types and fixed-point definitions`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:23:│       ├── exp_lut_values.h            # Exponential LUT for softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:44:- **Multi-Head Self-Attention**: 8 parallel attention heads with Q/K/V projections`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:45:- **Feed-Forward Network**: Two-layer MLP with GELU activation (d_model=512, d_ff=2048)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/README.md:53:typedef ap_fixed<10, 2> w_attn_t;   // Attention weights`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/config/hls_config.cfg:12:syn.file=../hls/src/attention_scores2.cpp`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/config/hls_config.cfg:13:syn.file=../hls/src/attention_types.h`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/config/hls_config.cfg:23:syn.file=../hls/src/multi_head_attention.cpp`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/config/hls_config.cfg:24:syn.file=../hls/src/multi_head_attention.h`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/transformer-hls-thesis/config/hls_config.cfg:27:syn.file=../hls/src/softmax2.cpp`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

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
