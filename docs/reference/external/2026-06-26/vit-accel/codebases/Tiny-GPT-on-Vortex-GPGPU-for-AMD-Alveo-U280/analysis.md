---
source_type: codebase
source_name: Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280
analysis_scope: static
integrated_package: analysis/integrated-2026-06-26
source_group: vit-accel-codebase
---

<!-- Integrated per-source copy. Original source: /home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel/codebases/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/analysis.md -->

# Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280 Detailed Codebase Analysis

- local_path: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280`
- repo_remote: `not_available`
- category: `Vortex GPGPU FPGA demo`
- HGTXR relevance: `low`
- matched_paper: `paper_not_confirmed`

## 1. 분석 범위와 판정
- 이 문서는 README, git remote, 파일 구조, 주요 소스 확장자, HLS/RTL 키워드, quantization/attention 키워드를 근거로 작성한 정적 분석이다.
- 실제 학습/합성/보드 실행 결과가 아니라 코드베이스 구조와 HGTXR 적용 가능성 평가다.
- eye/gaze/pupil 직접 근거가 없으면 ViT/Transformer 가속 기반의 간접 적용으로 분류한다.

## 2. Repo / Layer 구조

| 항목 | 값 |
|---|---:|
| 전체 파일 수 | `2553` |
| 주요 언어 | `.bin:634, C:486, SystemVerilog:273, C++:207, C/C++ header:191, no_ext:170, Verilog:43, .o:35, Markdown:33, .db:29` |
| LOC 추정 | `C++:56205, C:52112, SystemVerilog:50209, Verilog:46304, .txt:29678, C/C++ header:26818, C++ header:25184, Tcl:3403, Markdown:2816, Python:2667` |

### Directory Map
- `.github/` (1 entries)
- `.github/workflows/` (1 entries)
- `ci/` (13 entries)
- `docs/` (16 entries)
- `docs/assets/` (1 entries)
- `hw/` (8 entries)
- `hw/dpi/` (4 entries)
- `hw/rtl/` (18 entries)
- `hw/scripts/` (7 entries)
- `hw/syn/` (5 entries)
- `hw/unittest/` (9 entries)
- `kernel/` (14 entries)
- `kernel/include/` (3 entries)
- `kernel/scripts/` (3 entries)
- `kernel/src/` (10 entries)
- `miscs/` (3 entries)
- `miscs/docker/` (2 entries)
- `miscs/patches/` (3 entries)
- `perf/` (1 entries)
- `perf/cache/` (2 entries)
- `runtime/` (13 entries)
- `runtime/common/` (7 entries)
- `runtime/include/` (1 entries)
- `runtime/opae/` (4 entries)
- `runtime/rtlsim/` (2 entries)
- `runtime/simx/` (2 entries)
- `runtime/stub/` (3 entries)
- `runtime/xrt/` (2 entries)
- `sim/` (7 entries)
- `sim/common/` (13 entries)
- `sim/opaesim/` (7 entries)
- `sim/rtlsim/` (5 entries)
- `sim/simx/` (40 entries)
- `sim/xrtsim/` (7 entries)
- `tests/` (6 entries)
- `tests/kernel/` (5 entries)
- `tests/opencl/` (24 entries)
- `tests/regression/` (16 entries)
- `tests/riscv/` (5 entries)
- `tests/unittest/` (3 entries)

### Metadata / Config Refs
- `README.md`
- `hw/syn/altera/README`
- `hw/syn/altera/analyze_timing.tcl`
- `hw/syn/altera/report_area.tcl`
- `hw/syn/altera/quartus/timing-html.tcl`
- `hw/syn/altera/quartus/project.tcl`
- `hw/syn/altera/opae/vortex_afu.json`
- `hw/syn/xilinx/README`
- `hw/syn/xilinx/xrt/scripts/xsim.tcl`
- `hw/syn/xilinx/xrt/scripts/gen_xo.tcl`
- `hw/syn/xilinx/xrt/scripts/package_kernel.tcl`
- `hw/syn/xilinx/xrt/scripts/gen_ip.tcl`
- `hw/syn/xilinx/test/project.tcl`
- `hw/syn/synopsys/esyn.tcl`
- `hw/syn/synopsys/fsyn.tcl`
- `hw/syn/synopsys/syn.tcl`
- `hw/syn/synopsys/models/memory/cln28hpm/convert_lib_to_db.tcl`
- `hw/scripts/parse_vcs_list.tcl`
- `third_party/fpnew/CITATION.cff`
- `third_party/fpnew/README.license.md`
- `third_party/fpnew/README.md`
- `third_party/fpnew/vendor/openc910/README.md`
- `third_party/fpnew/vendor/opene906/README.md`
- `third_party/fpnew/tb/flexfloat/README.md`
- `third_party/fpnew/docs/README.md`
- `third_party/fpnew/src/common_cells/README.md`
- `third_party/fpnew/src/common_cells/formal/README.md`
- `third_party/fpnew/src/common_cells/test/cdc_2phase_synth.tcl`
- `third_party/fpnew/src/common_cells/test/waves/cdc_fifo_2phase.tcl`
- `third_party/fpnew/src/common_cells/test/waves/cdc_fifo_gray.tcl`

### Core Source Refs
- `tests/opencl/bfs/graph4k.txt`: 28677 lines
- `runtime/common/nlohmann_json.hpp`: 24674 lines
- `hw/syn/synopsys/models/memory/cln28hpc/rf2_32x128_wm1/rf2_32x128_wm1.v`: 15361 lines
- `hw/syn/synopsys/models/memory/cln28hpm/rf2_256x19_wm0/rf2_256x19_wm0.v`: 8697 lines
- `hw/syn/synopsys/models/memory/cln28hpm/rf2_32x19_wm0/rf2_32x19_wm0.v`: 8309 lines
- `third_party/fpnew/src/fpu_div_sqrt_mvp/hdl/control_mvp.sv`: 3413 lines
- `hw/syn/xilinx/test/project.tcl`: 2228 lines
- `tests/opencl/dotproduct/shrUtils.cpp`: 1954 lines
- `tests/opencl/blackscholes/shrUtils.cpp`: 1954 lines
- `tests/opencl/transpose/shrUtils.cpp`: 1950 lines

### Hardware-Oriented Source Refs
- `runtime/common/scope.h`
- `runtime/common/scope.cpp`
- `runtime/common/callbacks.h`
- `runtime/common/common.h`
- `runtime/common/malloc.h`
- `runtime/common/nlohmann_json.hpp`
- `runtime/xrt/vortex.cpp`
- `runtime/simx/vortex.cpp`
- `runtime/stub/vortex.cpp`
- `runtime/stub/utils.cpp`
- `runtime/rtlsim/vortex.cpp`
- `runtime/opae/vortex.cpp`
- `runtime/opae/driver.cpp`
- `runtime/opae/driver.h`
- `runtime/include/vortex.h`
- `hw/VX_config.h`
- `hw/VX_types.h`
- `hw/dpi/float_dpi.cpp`
- `hw/dpi/util_dpi.cpp`
- `hw/unittest/issue_top/main.cpp`
- `hw/unittest/common/vl_simulator.h`
- `hw/unittest/generic_queue/main.cpp`
- `hw/unittest/core_top/main.cpp`
- `hw/unittest/cache_top/main.cpp`
- `hw/unittest/cache/ram.h`
- `hw/unittest/cache/cachesim.h`
- `hw/unittest/cache/cachesim.cpp`
- `hw/unittest/cache/testbench.cpp`
- `hw/unittest/mem_streamer/ram.h`
- `hw/unittest/mem_streamer/memsim.h`

## 3. README 기반 기능 요약
- README: `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/README.md`
- # Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280
- # AMD Open Hardware Competition (Winner 2025)
- This project was also part of AMD Open Hardware Competition (Adaptive Computing- Student Level) for the year 2025.
- https://www.openhw.eu/2025_results_1
- Youtube Video Link:
- https://youtu.be/SDaqhbEOV1Q
- ## Team Information
- - Team number: AOHW25_616
- - Project name: Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280
- - University name: University of Essex
- - Participant(s):
- -- Muhammad Ahmed Khan
- -- Ya-lun Lee
- -- Qizal Arsalan
- - Supervisor: Dr. Xiajun Zhai
- # Vortex GPGPU
- ## Credits & Upstream
- This work builds on **Vortex GPGPU** (Apache-2.0): https://github.com/vortexgpgpu/vortex
- Major changes here: TinyGPT kernels, host flow, U280 configs, and build scripts.
- If you use this repo, please also cite the Vortex MICRO’21 paper (see upstream README).

## 4. Function / Dataflow 관점
- paper-backed algorithm mapping은 확인되지 않았다.
- local source에서 attention/matmul/softmax/quantization/HLS 키워드를 기준으로만 mapping한다.

### Keyword Evidence
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/README.md:38:    - configurable pipeline issue width.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/codebase.md:7:    - `core`: core pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/continuous_integration.md:2:- Each time you push to the repo, the Continuous Integration pipeline will run`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/continuous_integration.md:3:- This pipeline consists of creating the correct development environment, building your code, and running all tests`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/continuous_integration.md:4:- This is an extensive pipeline so it might take some time to complete`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/testing.md:51:## Adding Your Tests to the CI Pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/contributing.md:12:- When you make a PR it will be tested against the continuous integration (ci) pipeline (see `continuous_integration.md`)`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/contributing.md:13:- It is not sufficient to just write some tests, they need to be incorporated into the ci pipeline to make sure they are run`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/debugging.md:19:A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (decoded instruction, register states, pipeline states, etc..). You can increase the verbosity of the trace by changing the debug level.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/debugging.md:36:A debug trace `run.log` is generated in the current directory during the program execution. The trace includes important states of the simulated processor (memory, caches, pipeline, stalls, etc..). A waveform trace `trace.vcd` is also generated in the current directory during the program execution. You can visualize the waveform trace using any tool that can open VCD files (Modelsim, Quartus, Vivado, etc..). [GTKwave] (http://gtkwave.sourceforge.net) is a great open-source scope analyzer that also works with VCD files.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/ci/blackbox.sh:28:    echo "--class: 0=disable, 1=pipeline, 2=memsys"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/third_party/ramulator/perf_comparison/configs/usimm.cfg:9:PIPELINEDEPTH	1`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/cache_subsystem.md:6:- Non-blocking pipelined write-through cache architecture with per-bank MSHR`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/cache_subsystem.md:20:Incoming requests entering the cache are sent to a dispatch crossbar that select the corresponding bank for each request, resolving bank collisions with stalls. The result output of each bank is merge back into outgoing response port via merger crossbar. Each bank intergates a non-blocking pipeline with a local Miss Status Holding Register (MSHR) to reduce the miss rate. The bank pipeline consists of the following stages:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/cache_subsystem.md:22:- **Schedule**: Selects the next request into the pipeline from the incoming core request, memory fill, or the MSHR entry, with priority given to the latter.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/cache_subsystem.md:27:Deadlocks inside the cache can occur when the MSHR is full and a new request is already in the pipeline. It can also occur when the memory request queue is full, and there is an incoming memory response. The cache mitigates MSHR deadlocks by using an early full signal before a new request is issued and similarly mitigates memory deadlocks by ensuring that its request queue never fills up.`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/microarchitecture.md:34:### Vortex Pipeline/Datapath`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/microarchitecture.md:38:Vortex has a 6-stage pipeline:`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/docs/microarchitecture.md:42:    - Schedule the next PC into the pipeline`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/sim/xrtsim/Makefile:21:DBG_TRACE_FLAGS += -DDBG_TRACE_PIPELINE`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/miscs/patches/riscv-tests.patch:15: 	mt-matmul \`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/guassian/OriginalParallel.c:136:** Pay attention to the index.  Index i give the range`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/main.cc:67:  "softmax","output"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/main.cc:72:void softmax(std::vector<float>& vec, float temp=1.0f){`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/main.cc:225:    softmax(h_out, temperature);`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:10:    "softmax", "output"`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:23:    ["gpt", "runs", "inference", "on", "vortex", "core", "using", "int8", "matvec", "activation", "softmax", "output"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:28:    ["prompt", "token", "next", "token", "predicted", "by", "softmax", "output", "layer"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:35:    ["activation", "softmax", "output", "used", "for", "classification", "in", "model"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:40:    ["gpt", "uses", "softmax", "to", "predict", "token", "from", "quantized", "activation", "vector"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:42:    ["tinygpt", "runs", "loop", "over", "tokens", "predicting", "each", "with", "softmax", "and", "matvec"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:45:    ["core", "executes", "thread", "matvec", "activation", "softmax", "output", "token"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:49:    ["softmax", "output", "activation", "used", "in", "next", "token", "prediction"],`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv1/scripts/train_model.py:81:        # Softmax + cross-entropy`
- `/home/kjm26/project/PRJXR/References/ViT-Accelerator/Codebase/Tiny-GPT-on-Vortex-GPGPU-for-AMD-Alveo-U280/tests/opencl/tinygptv2/kernel.cl:33:inline int softmax_sample_top1(__local float* logits, int vocab) {`

## 5. HGTXR 적용 해석
- 직접 논문 근거가 약하므로 HGTXR 기본 경로에는 넣지 않는다.
- 단, HLS kernel, PYNQ packaging, AXI/DMA, testbench, golden model 등 구현 보조 자료로 활용 가능하다.

### 적용 가능 모듈
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
