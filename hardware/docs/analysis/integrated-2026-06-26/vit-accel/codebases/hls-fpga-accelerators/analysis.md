---
source_type: codebase
source_name: hls-fpga-accelerators
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/hls-fpga-accelerators/analysis.md -->

# hls-fpga-accelerators Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators`
- repo_remote: `not_available`
- category: `HLS matmul/softmax/rmsnorm kernel collection`
- HGTXR relevance: `medium`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `24` |
| 주요 언어 | `C++:9, C/C++ header:6, Tcl:5, C++ header:2, no_ext:1, Markdown:1` |
| LOC 추정 | `C++:920, Tcl:340, C++ header:228, C/C++ header:216, Markdown:170` |

### Directory Map
- `common/` (1 entries)
- `elementwise/` (3 entries)
- `matmul/` (4 entries)
- `rmsnorm/` (4 entries)
- `softmax/` (4 entries)
- `unary/` (5 entries)
- `unary/axc-math/` (2 entries)

### Metadata / Config Refs
- `README.md`
- `unary/unary.tcl`
- `rmsnorm/rmsnorm.tcl`
- `matmul/matmul.tcl`
- `elementwise/elementwise.tcl`
- `softmax/softmax.tcl`

### Core Source Refs
- `README.md`: 170 lines
- `matmul/matmul.cpp`: 169 lines
- `unary/unary.cpp`: 160 lines
- `rmsnorm/rmsnorm.cpp`: 135 lines
- `unary/axc-math/exponential-lut.hpp`: 131 lines
- `softmax/softmax.cpp`: 130 lines
- `elementwise/elementwise.cpp`: 127 lines
- `unary/axc-math/interpolation-wrapper.hpp`: 97 lines
- `unary/unary.tcl`: 73 lines
- `common/config.h`: 72 lines

### Hardware-Oriented Source Refs
- `common/config.h`
- `unary/unary.tcl`
- `unary/unary_tb.cc`
- `unary/unary.cpp`
- `unary/unary.h`
- `unary/axc-math/interpolation-wrapper.hpp`
- `unary/axc-math/exponential-lut.hpp`
- `rmsnorm/rmsnorm.tcl`
- `rmsnorm/rmsnorm_tb.cc`
- `rmsnorm/rmsnorm.h`
- `rmsnorm/rmsnorm.cpp`
- `matmul/matmul_tb.cc`
- `matmul/matmul.h`
- `matmul/matmul.tcl`
- `matmul/matmul.cpp`
- `elementwise/elementwise.h`
- `elementwise/elementwise.cpp`
- `elementwise/elementwise.tcl`
- `softmax/softmax.h`
- `softmax/softmax.cpp`
- `softmax/softmax.tcl`
- `softmax/softmax_tb.cc`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md`
- # hls-fpga-accelerators
- Collection of kernel accelerators optimised for LLM execution
- ## Compilation
- ### Matrix Multiplication
- ```bash
- cd matmul
- vitis_hls -f matmul.tcl
- ```
- Possible adjustments through environment variables:
- | Environment Variable | Possible Values | Default |
- |----------------------|-----------------|---------|
- | DATATYPE             | FLOAT4, FLOAT8, FLOAT16, FLOAT32, FIXED8, FIXED16 | FIXED16 |
- | BUS             | 64, 128, 256, 512, 1024, 2048 | 512 |
- | B_COLS             | Power of two from 64 on | 4096 |
- | C_COLS             | Power of two from 64 on | 4096 |
- | PART               | xcu250-figd2104-2L-e, xck26-sfvc784-2LV-c | xcu250-figd2104-2L-e |
- The `xcu250-figd2104-2L-e` is an Alveo U250, whereas `xck26-sfvc784-2LV-c` is a Kria K26
- Function signature:
- ```c
- void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols)

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:52:# v++ --advanced.param compiler.hlsDataflowStrictMode`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:53:config_dataflow -strict_mode warning`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:11:#pragma HLS INLINE off`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:17:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:18:#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg =       \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:27:#pragma HLS UNROLL`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:42:#pragma HLS UNROLL`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:54:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:55:#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg =       \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:61:#pragma HLS UNROLL`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:87:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:88:#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg =       \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:100:#pragma HLS PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:101:#pragma HLS LOOP_TRIPCOUNT min = kTotalMaxSize max = kTotalMaxSize avg =       \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:115:#pragma HLS INTERFACE m_axi offset = slave port = in1 bundle = gmem0`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:116:#pragma HLS INTERFACE m_axi offset = slave port = out bundle = gmem1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:117:#pragma HLS INTERFACE s_axilite register port = size`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:118:#pragma HLS INTERFACE s_axilite register port = return`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:122:#pragma HLS stream variable = stream_a depth = 32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.cpp:123:#pragma HLS stream variable = stream_c depth = 32`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:10:cd matmul`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:11:vitis_hls -f matmul.tcl`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:29:void matmul(RawDataT *a, RawDataT *b, RawDataT *c, int a_rows, int b_cols, int c_cols)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:135:### Softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:138:cd softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:139:vitis_hls -f softmax.tcl`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/README.md:159:void softmax(RawDataT *in, RawDataT *out, uint64_t size);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax_tb.cc:6:#include "softmax.h"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax_tb.cc:34:  softmax(a, c, rows * cols);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:44:open_project softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:45:set_top softmax`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:46:# v++ -g, -D, -I, --advanced.prop kernel.softmax.kernel_flags`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:47:add_files "./softmax.cpp" -cflags " -DUSE_$datatype -DBUS=$bus -DM_COLS=$cols -DM_ROWS=$rows "`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:48:add_files -tb "./softmax_tb.cc" -cflags " -I . -DUSE_$datatype -DBUS=$bus -DM_COLS=$cols -DM_ROWS=$rows "`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/hls-fpga-accelerators/softmax/softmax.tcl:61:config_export -format xo -ipname softmax`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

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
