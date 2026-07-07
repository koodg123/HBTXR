# Integrated Framework Analysis: References/HW_Framework

Source directory: `/home/kjm26/project/PRJXR/References/HW_Framework`

## 1. Corpus Summary

| Metric | Value |
|---|---:|
| Top-level repositories | 11 |
| Primary focus | HLS frameworks, compiler passes, DSLs, RTL integration, accelerator generation |
| Direct RTL/HLS content | Medium to high |
| Direct HGTXR value | Toolflow templates, report parsing, HLS generation, host/kernel structure |

## 2. Framework Coverage

| Framework | Category | Key files inventoried | HGTXR relevance | Priority |
|---|---|---|---|---|
| `FlexCNN` | Direct HLS/CNN accelerator | `HLS_Codes/kernel.cpp`, `hls_script.tcl`, `SDx_project/src/hw_kernel.cpp`, `host.cpp` | Local-feature/front-end kernel and HLS automation reference | P1 |
| `HLS4ML` | Model-to-HLS conversion | `hls4ml/converters/*`, `writer/*`, `backends/*`, `Transformer_Neural_Network_HLS/hls/*.tcl` | Conversion and writer patterns for quantized model export | P1/P2 |
| `NVDLA` | RTL accelerator integration | `nvdla-attn-mechanism/src/rtl/NV_NVDLA_attn.v`, `NV_NVDLA_attn_partition.v`, `src/sw/nvdla_attn.c` | Attention block integration boundary reference | P2 |
| `Prometheus` | HLS code generation and evaluation | `main.py`, `launch.py`, `code_gen/*.py`, `write_tcl.py`, `parse_vitis_report.py`, `script/*.tcl` | Experiment automation and Vitis report parsing | P0 |
| `Sextans` | TAPA/HLS sparse accelerator | `src/sextans.cpp`, `sextans-host.cpp`, `bitstream/run_tapa_hls.sh` | Host/kernel/bitstream flow and sparse dataflow | P1 |
| `Stream-HLS` | Streaming HLS compiler | `include/streamhls/*`, `lib/Transforms/*`, `lib/Translation/*`, `examples/scripts/hls.tcl` | Stream/dataflow pass ideas; higher integration cost | P2 |
| `allo` | Python DSL to HLS/TAPA/Vitis | `allo/dsl.py`, `dataflow.py`, `backend/hls.py`, `backend/vitis.py`, `backend/tapa.py` | Future DSL exploration; not default C3b path | P2 |
| `cgra4ml` | SoC/CGRA flow | `deepsocflow/rtl/*`, `ibex-soc/*`, `run/*.py` | Too broad for immediate HGTXR; useful for SoC boundary ideas | P3 |
| `mase` | Model compression/HW pipeline | `src/chop/*`, `configs/*`, scripts | Quantization/search pipeline reference | P2 |
| `scalehls` | MLIR/ScaleHLS compiler | `lib/Translation/EmitHLSCpp.cpp`, `tools/pyscalehls/pyscalehls.py`, `build-scalehls.sh` | Compiler-level HLS generation reference | P2 |
| `soda` | Stencil/dataflow DSL | `src/soda/sodac.py`, `core.py`, `grammar.py`, `dataflow.py` | Dataflow DSL ideas; indirect | P3 |

## 3. Toolflow Lessons

| Lesson | Sources | HGTXR action |
|---|---|---|
| Report parsing should be automated | `Prometheus/parse_vitis_report.py`, HLS4ML writers | Add normalized csynth/vivado report collectors for every experiment variant |
| HLS scripts are first-class artifacts | `FlexCNN`, `Sextans`, `HLS4ML`, `Prometheus` | Store Tcl/build scripts with generated reports |
| DSL/compiler frameworks are useful after the manual baseline is stable | `allo`, `scalehls`, `Stream-HLS`, `soda` | Do not migrate current C3b path to a new compiler yet |
| Host/kernel boundary templates improve measurement discipline | `Sextans`, `FlexCNN`, `NVDLA` | Standardize DMA/batch/latency JSON schema |
| Frameworks can increase reproducibility but also introduce version risk | HLS4ML, scalehls, Stream-HLS | Pin tool versions before any generated-code experiment |

## 4. Recommended Framework-Derived Artifacts for HGTXR

| Artifact | Source inspiration | Purpose | Priority |
|---|---|---|---|
| `csynth_resource_matrix.py` or equivalent collector | `Prometheus/parse_vitis_report.py` | Normalize DSP/LUT/FF/BRAM/URAM by block and variant | P0 |
| HLS run-script template per variant | `FlexCNN`, `HLS4ML`, `Sextans` | Reproducible Vitis HLS runs | P0 |
| Board measurement JSON schema | `Sextans`, `FlexCNN` host flows | Capture DMA bandwidth, batch size, p95/p99, mode latency | P1 |
| Quantized model export manifest | HLS4ML, mase | Track weights, fixed-point scale, calibration method | P1 |
| Dataflow DSL feasibility note | allo, Stream-HLS, scalehls | Evaluate future automation only after C3b successor is stable | P2 |

## 5. Integration Judgment

For the current HGTXR hardware goal, `Prometheus`-style report parsing and `FlexCNN`/`Sextans`-style reproducible HLS/host flow are the most immediately useful. Compiler/DSL frameworks are valuable references, but switching the main implementation path to them now would add avoidable risk.

