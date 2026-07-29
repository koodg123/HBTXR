# E2E AXI-Stream ViT Baseline - 2026-06-09

This artifact records the first HGTXR E2E HLS baseline shell requested for
ZCU104 Q4W/Q8A execution.

Implemented modules:

- AXI-Stream DMA-facing top: `hgtxr_e2e_axis_top`
- Conv-style patch embedding over the 256x256 frame stream
- Explicit global token buffer
- Two named MLP compute units: `hgtxr_e2e_mlp_unit<0>` and
  `hgtxr_e2e_mlp_unit<1>`
- Two named attention compute units: `hgtxr_e2e_attn_unit<0>` and
  `hgtxr_e2e_attn_unit<1>`
- MLP-based state/head projection
- Controller schedule: `ATTN0 -> MLP0 -> ATTN1 -> MLP1 -> ...` for six blocks

HG-PIPE mapping used for this shell:

- DeiT-Tiny dimensions: `C=192`, `H=3`, `CH=768` are retained as the target
  model dimensions.
- The MLP split mirrors HG-PIPE's `M1 -> GELU -> M2` ownership, but this first
  shell uses deterministic synthetic weight seeds rather than full packed dense
  Q4 weights.
- The attention split mirrors HG-PIPE's separate attention block ownership, but
  currently uses pooled-context attention to keep the first E2E AXI-Stream top
  synthesizable before exact softmax vector equivalence is closed.
- FIFO depth, bus width, bit width, block count, and unit parallelism are
  compile-time macros in `hardware/configs/zcu104_e2e_q4w8a_defines.h`.

Important limitation: this is an E2E HLS shell and DMA interface baseline, not
yet the final HG-PIPE numerical implementation. The next refinement must replace
pooled attention and synthetic MLP weights with packed Q4 dense QKV/WO/W1/W2
weights, add exact bias/head-scale handling, and compare against PyTorch/HLS
vectors.

## Validation Results

- g++ E2E AXI-Stream smoke passed: runtime_state=1 count=6 last=1.
- Vitis HLS csim is blocked by the existing Ubuntu/WSL header issue.
- Vitis HLS csynth passed for solution_e2e_q4w8a: 273.97 MHz, 1,610,485..1,610,491 cycles, 8.052 ms, 52 BRAM, 18 DSP, 3798 FF, 6411 LUT, 0 URAM.
- Weight m_axi read path is live: hgtxr_e2e_axis_top_csynth.rpt includes gmem_e2e_weights_m_axi_U and AR/R interface ports.

## Full Dense ViT E2E Update - 2026-06-09

The E2E AXI-Stream top now contains a full dense ViT execution path for the
DeiT-Tiny/HG-PIPE dimensions used in the HGTXR target:

- `C=192`, `heads=3`, `head_dim=64`, `MLP hidden=768`.
- Default full target config uses `active_tokens=196`, `blocks=6`,
  `patch_grid=14x14`, `Q4W/Q8A`, `bus_width=128`, `FIFO_DEPTH=64`.
- The controller alternates two physical ATTN units and two physical MLP units:
  `ATTN0 -> MLP0 -> ATTN1 -> MLP1 -> ATTN0 -> MLP0 ...`.
- The top has AXI-Stream input/output, AXI master packed-weight reads, AXI
  master runtime-state write, a controller, and BRAM-backed global buffers.
- The full path performs conv-style patch embedding, pre-LN attention
  (`Q/K/V -> score -> softmax approximation -> value -> output projection`),
  pre-LN MLP (`W1 -> GELU approximation -> W2`), residual updates, and an
  MLP-based output head.

Validation:

- `g++` smoke passed with reduced verification config:
  `runtime_state=1 count=6 last=1`.
- Vitis HLS `csim_design` passed after fixing Ubuntu 24.04/WSL integration:
  multiarch include paths were added, stale testbench files were removed by
  using a reset E2E project, and PATH/LIBRARY_PATH were set so system linker
  and libraries are used instead of the incompatible Vitis 2023.2 bundled
  binutils path.
- Vitis HLS `csynth_design` passed for the full target config in
  `hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a`.
- Full target report:
  - estimated clock: `3.650 ns` (`273.97 MHz`)
  - latency: `619,214,991 cycles`
  - absolute latency at 5 ns target: `3.096 sec`
  - utilization: `269 BRAM_18K`, `48 DSP`, `41,885 FF`, `61,369 LUT`, `0 URAM`
  - utilization percent: `43% BRAM`, `2% DSP`, `9% FF`, `26% LUT`, `0% URAM`

Current limitation: this is the first full dense E2E hardware path. It is
functionally complete at the block-operation level, but it still uses
hardware-friendly approximate LayerNorm denominator, GELU, and softmax math
instead of the final HG-PIPE table math. The next performance step is to
increase datapath parallelism and prefetch/cache packed weights so the design
uses more ZCU104 DSP bandwidth and reduces the current sequential latency.

## ZCU104 Fit Tuning Update - 2026-06-09

The full dense E2E path was retuned to use the available ZCU104 fabric more
aggressively while staying inside HLS-estimated device resources.

Final selected fit configuration:

- HGTXR_BUS_WIDTH=256
- HGTXR_BUFFER_SIZE=256
- HGTXR_FIFO_DEPTH=128
- HGTXR_PARALLELISM_FACTOR=5
- HGTXR_E2E_ATTN_PAR=5
- HGTXR_E2E_DENSE_PAR=5
- gb.hidden is bound to URAM.
- token, norm, Q, K, V, and attention buffers are cyclically partitioned by HGTXR_E2E_DENSE_PAR.

Explored candidates:

| Candidate | Result | Latency | BRAM | DSP | FF | LUT | URAM | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline dense | fit | 3.096 sec | 43% | 2% | 9% | 26% | 0% | Sequential dense path, 128-bit bus |
| PAR16 | not fit | 2.647 sec | 41% | 2% | 27% | 238% | 50% | LUT explosion |
| PAR8 | not fit | 2.677 sec | 39% | 2% | 18% | 130% | 50% | Good latency, LUT over device |
| PAR6 | not fit | 2.706 sec | 30% | 3% | 16% | 106% | 50% | Close, still over LUT |
| PAR5 | fit | 2.777..2.778 sec | 45% | 3% | 15% | 92% | 52% | Selected ZCU104-fit point |

Final PAR5 report:

- report: hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt
- target clock: 5.00 ns
- estimated clock: 4.035 ns / 247.80 MHz
- latency: 555,319,367..555,681,575 cycles
- absolute latency at 5 ns target: 2.777..2.778 sec
- utilization: 284 BRAM_18K, 63 DSP, 70,410 FF, 212,865 LUT, 50 URAM
- utilization percent: 45% BRAM, 3% DSP, 15% FF, 92% LUT, 52% URAM

Validation:

- Vitis HLS csim_design passed for the reduced E2E smoke config: runtime_state=1 count=6 last=1.
- Vitis HLS csynth_design passed for the full six-block target config.
- python3 hardware/tools/static_validate_hgtxr.py passed.

Remaining performance bottleneck: several dense projection loops still report
II violations on gmem_e2e_weights because parallel lanes request multiple packed
weight words from one AXI master port in the same cycle. The next resource-fit
performance step should add local tiled weight caches in BRAM/URAM and/or split
the packed weight interface into multiple AXI bundles before raising dense
parallelism above five again.

## HG-PIPE Compact LUT Math Update - 2026-06-09

The selected ZCU104-fit E2E path now routes the main nonlinear math wrappers
through compact HG-PIPE-style LUT helpers when HGTXR_USE_HGPIPE_LUT_MATH=1:

- hgtxr_cyclic_math.hpp defines compact LUT paths for GELU, shifted-score exp,
  reciprocal square-root, and quantization clamp helpers.
- hgtxr_e2e_layernorm now uses hgtxr_rsqrt_approx(var_mean) instead of the
  earlier 1 + variance denominator surrogate.
- hgtxr_e2e_gelu and hgtxr_e2e_exp_approx now route into cyclic math helpers,
  so E2E MLP GELU and E2E attention softmax exp use the same LUT-gated path.
- hardware/configs/zcu104_e2e_q4w8a_defines.h explicitly enables
  HGTXR_USE_HGPIPE_LUT_MATH=1 for this target.

Host-side LUT validation artifact:
docs/resources/hgpipe_lut_math_validation_2026_06_09.json.

Validation summary for compact tables:

- GELU range [-4, 4]: max_abs_error 0.8000702459, mean_abs_error 0.1694300897.
- Exp shifted-score range [-8, 0]: max_abs_error 0.4121303269, mean_abs_error 0.0365899631.
- Rsqrt variance range [0.0078125, 64]: max_abs_error 10.6079353431, mean_abs_error 0.3264125885.

Latest full-target Vitis HLS result after connecting E2E attention exp to the
HG-PIPE-style LUT path:

- report: hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt
- target clock: 5.00 ns
- estimated clock: 4.035 ns / 247.80 MHz
- latency: 555,505,175..556,229,591 cycles
- absolute latency at 5 ns target: 2.778..2.781 sec
- utilization: 284 BRAM_18K, 67 DSP, 52,318 FF, 199,975 LUT, 50 URAM
- utilization percent: 45% BRAM, 3% DSP, 11% FF, 86% LUT, 52% URAM

The Spark sub-agent audit identified the next highest-priority HG-PIPE gap as
head-aware S2/cyclic attention: split channels into heads=3 and head_dim=64,
and apply the DeiT/HG-PIPE score scale of 1/sqrt(64)=1/8 at the S2 attention
first-step and fused-block call sites. This remains open for the next code pass.

Residual numerical risk: these compact LUTs are synthesis-friendly baseline
approximations, not final per-layer HG-PIPE-equivalent tables. The low-variance
rsqrt max error is intentionally recorded as a remaining accuracy gap.


## 2026-06-10 E2E Vector CSim Update

The existing reduced E2E csim target now performs deterministic vector comparison instead of only structural smoke checks.

- Testbench: hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp.
- Regression: software/tests/test_e2e_axis_vector_csim.py.
- Vitis script: hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl.
- Reduced csim configuration: blocks=1, active_tokens=4, patch_grid=1x4, ff_dim=32.
- Latest Vitis evidence: CSim done with 0 errors; raw state vector {16, 0, 0, 0, 0, 0}; runtime_state=2.

This is the current public-ABI E2E vector gate. The full PAR5 196-token/6-block csynth evidence remains the resource-fit reference, while full-size E2E numerical equivalence remains a later gate.


## 2026-06-10 Independent Reduced E2E Reference Update

The reduced E2E vector csim gate now has an independent Python reference:

- Tool: hardware/tools/validate_e2e_axis_vector.py.
- Expected raw state: {16, 0, 0, 0, 0, 0}.
- Reference details: Q4 ap_fixed<4,2> lane values, deterministic frame pattern, patch embedding channel 0, zero-delta controller traversal, MLP head scaling, live weight bit, and AXI int(value * 16) output packing.
- Validation: targeted pytest reports 2 passed, and Vitis HLS csim still reports CSim done with 0 errors.

This improves the public E2E gate from a C++-only golden check to an SW/HW-aligned reduced reference. The PAR5 full-size csynth result remains the resource-fit point; full-size numerical equivalence remains future work.


## 2026-06-10 Nonzero Reduced E2E Vector Update

The reduced E2E vector gate now uses nonzero Transformer block weights instead of a zero-delta block traversal.

- Enabled paths in the reduced gate: LN beta, Q/K/V projection, softmax attention, WO residual, MLP GELU/W2 residual, MLP head, live weight bit, and AXI state writer.
- Expected raw state changed from {16, 0, 0, 0, 0, 0} to {23, -9, 0, 0, 0, 0}.
- Independent reference: hardware/tools/validate_e2e_axis_vector.py.
- Latest Vitis evidence: run_e2e_q4w8a_csim.tcl reports CSim done with 0 errors.

This is still a reduced public-ABI E2E gate. The full PAR5 196-token/6-block csynth result remains the resource-fit reference, and full-size numerical equivalence remains future work.


## 2026-06-10 Multi-Channel Reduced E2E Vector Update

The reduced public E2E vector gate now covers four deterministic channel groups instead of only channel 0.

- Patch raw weights: channels 0..3 use {1, 2, -1, 3}.
- Head raw weights: outputs 0..3 use {7, -8, 4, -4} on channels 0..3.
- Block paths exercised: LN beta, diagonal Q/K/V, grouped WO, grouped MLP W1/W2, GELU, head, live weight bit, and AXI writer.
- Expected raw state: {18, -4, 0, -2, 0, 0}.
- Latest Vitis evidence: run_e2e_q4w8a_csim.tcl reports CSim done with 0 errors.

This is still reduced-size verification; full PAR5 196-token/6-block csynth evidence remains the resource-fit point until a full-size numerical equivalence gate is practical.

## 2026-06-10 Six-Output Reduced E2E Vector Update

The reduced public E2E vector gate now drives all six public head output lanes with nonzero deterministic grouped Q4 weights.

- Patch raw weights: channels 0..5 use {1, 2, -1, 3, -2, 4}.
- Head raw weights: outputs 0..5 use {7, -8, 4, -4, 6, -6} on channels 0..5.
- Block paths exercised: LN beta, diagonal Q/K/V, grouped WO, grouped MLP W1/W2, GELU, head, live weight bit, and AXI writer.
- Expected raw state: {18, -3, 0, -2, 0, -4}.
- Latest Vitis evidence: run_e2e_q4w8a_csim.tcl reports CSim done with 0 errors.

This remains reduced-size verification; full PAR5 196-token/6-block csynth evidence remains the resource-fit reference until full-size numerical equivalence and resource recheck are practical.

## 2026-06-10 active16_ff128 E2E Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active16/FF128 E2E Q4W/Q8A csynth point. This is not the final full 196-token/6-block equivalence gate; it is an intermediate shape check before raising FF dim, token count, and block count together.

- Scale mode: `HGTXR_E2E_SCALE=active16_ff128`.
- Compile-time shape: `blocks=2`, `active_tokens=16`, `patch_grid_h=1`, `patch_grid_w=16`, `ff_dim=128`.
- Golden output: `[24, -9, 8, -5, 2, -14]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.695 ns, about 270.64 MHz.
- Latency: 6,743,956..6,747,284 cycles, 33.720..33.736 ms.
- Utilization: 284 BRAM_18K, 66 DSP, 49,246 FF, 196,407 LUT, 10 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 10% FF, 85% LUT, 10% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 6,627,781..6,631,109 cycles, 204 BRAM, 64 DSP, 43,711 FF, 185,730 LUT, 10 URAM.

## 2026-06-10 active16_ff256 E2E High-Hidden-Tap Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active16/FF256 E2E Q4W/Q8A csynth point with high hidden-index MLP taps. This is still not the final full 196-token/6-block equivalence gate.

- Scale mode: `HGTXR_E2E_SCALE=active16_ff256`.
- Compile-time shape: `blocks=2`, `active_tokens=16`, `patch_grid_h=1`, `patch_grid_w=16`, `ff_dim=256`.
- High hidden taps: 128, 143, 159, 191, 224, and 255.
- Golden output: `[25, -6, 13, -5, 7, -8]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.695 ns, about 270.64 MHz.
- Latency: 8,360,538..8,373,850 cycles, 41.803..41.869 ms.
- Utilization: 284 BRAM_18K, 66 DSP, 49,499 FF, 196,675 LUT, 20 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 10% FF, 85% LUT, 20% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 8,244,357..8,257,669 cycles, 204 BRAM, 64 DSP, 43,961 FF, 185,980 LUT, 20 URAM.

## 2026-06-10 active16_ff768 E2E Final-Dim Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active16/FF768 E2E Q4W/Q8A csynth point. This reaches the final DeiT-Tiny FF dimension but still uses reduced active-token and block counts.

- Scale mode: `HGTXR_E2E_SCALE=active16_ff768`.
- Compile-time shape: `blocks=2`, `active_tokens=16`, `patch_grid_h=1`, `patch_grid_w=16`, `ff_dim=768`.
- High hidden taps: 256, 383, 511, 512, 640, and 767, plus channels 12..17.
- Golden output: `[26, -5, 19, -2, 8, -6]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.985 ns, about 250.97 MHz.
- Latency: 14,760,660..14,780,372 cycles, 73.803..73.902 ms.
- Utilization: 284 BRAM_18K, 66 DSP, 49,913 FF, 197,249 LUT, 50 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 10% FF, 85% LUT, 52% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 14,644,485..14,664,197 cycles, 204 BRAM, 64 DSP, 44,378 FF, 186,572 LUT, 50 URAM.

## 2026-06-10 active32_ff768 E2E Token Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active32/FF768 E2E Q4W/Q8A csynth point. This reaches two rows of patch-grid coverage but still uses reduced block count and token count versus full active196/block6.

- Scale mode: `HGTXR_E2E_SCALE=active32_ff768`.
- Compile-time shape: `blocks=2`, `active_tokens=32`, `patch_grid_h=2`, `patch_grid_w=16`, `ff_dim=768`.
- Golden output: `[26, -5, 19, -2, 8, -6]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.985 ns, about 250.97 MHz.
- Latency: 29,488,052..29,527,476 cycles, 0.147..0.148 sec.
- Utilization: 284 BRAM_18K, 66 DSP, 50,256 FF, 197,595 LUT, 50 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 10% FF, 85% LUT, 52% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 29,371,877..29,411,301 cycles, 204 BRAM, 64 DSP, 44,724 FF, 186,868 LUT, 50 URAM.

## 2026-06-10 active64_ff768 E2E Token Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active64/FF768 E2E Q4W/Q8A csynth point. This reaches four rows of patch-grid coverage but still uses reduced block count and token count versus full active196/block6.

- Scale mode: `HGTXR_E2E_SCALE=active64_ff768`.
- Compile-time shape: `blocks=2`, `active_tokens=64`, `patch_grid_h=4`, `patch_grid_w=16`, `ff_dim=768`.
- Golden output: `[26, -5, 19, -2, 8, -6]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.985 ns, about 250.97 MHz.
- Latency: 59,191,668..59,270,516 cycles, 0.296 sec.
- Utilization: 284 BRAM_18K, 66 DSP, 50,606 FF, 197,896 LUT, 50 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 10% FF, 85% LUT, 52% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 59,075,493..59,154,341 cycles, 204 BRAM, 64 DSP, 45,070 FF, 187,164 LUT, 50 URAM.

## 2026-06-10 active128_ff768 E2E Token Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active128/FF768 E2E Q4W/Q8A csynth point. This reaches eight rows of patch-grid coverage but still uses reduced block count and token count versus full active196/block6.

- Scale mode: `HGTXR_E2E_SCALE=active128_ff768`.
- Compile-time shape: `blocks=2`, `active_tokens=128`, `patch_grid_h=8`, `patch_grid_w=16`, `ff_dim=768`.
- Golden output: `[26, -5, 19, -2, 8, -6]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.985 ns, about 250.97 MHz.
- Latency: 119,602,705..119,760,401 cycles, 0.598..0.599 sec.
- Utilization: 284 BRAM_18K, 66 DSP, 50,968 FF, 198,256 LUT, 50 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 11% FF, 86% LUT, 52% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 119,478,053..119,635,749 cycles, 204 BRAM, 64 DSP, 45,428 FF, 187,520 LUT, 50 URAM.

## 2026-06-10 active196_ff768 E2E Full-Token Ramp C-Synthesis Update

The staged full-scale ramp now has a strict active196/FF768 E2E Q4W/Q8A csynth point. This reaches full 14x14 token-grid coverage but still uses reduced `blocks=2` versus final block6 equivalence.

- Scale mode: `HGTXR_E2E_SCALE=active196_ff768`.
- Compile-time shape: `blocks=2`, `active_tokens=196`, `patch_grid_h=14`, `patch_grid_w=14`, `ff_dim=768`.
- Golden output: `[26, -5, 19, -2, 8, -6]`.
- Target clock: 5.00 ns.
- Estimated clock: 3.985 ns, about 250.97 MHz.
- Latency: 185,285,394..185,526,866 cycles, 0.926..0.928 sec.
- Utilization: 284 BRAM_18K, 67 DSP, 51,406 FF, 199,218 LUT, 50 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 11% FF, 86% LUT, 52% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 185,109,885..185,351,357 cycles, 204 BRAM, 64 DSP, 45,436 FF, 187,514 LUT, 50 URAM.


## 2026-06-10 active196_b6_ff768 E2E Full-Block C-Synthesis Update

The staged full-scale ramp now has a strict active196/block6/FF768 E2E Q4W/Q8A csim and csynth point. This is the full 14x14 token-grid and six-block controller gate for the deterministic generated-golden equivalence flow.

- Scale mode: `HGTXR_E2E_SCALE=active196_b6_ff768`.
- Compile-time shape: `blocks=6`, `active_tokens=196`, `patch_grid_h=14`, `patch_grid_w=14`, `ff_dim=768`.
- Golden output: `[32, -13, 26, -6, 14, -11]`.
- Target clock: 5.00 ns.
- Estimated clock: 4.035 ns, about 247.80 MHz.
- Latency: 555,505,175..556,229,591 cycles, 2.778..2.781 sec.
- Utilization: 284 BRAM_18K, 67 DSP, 52,318 FF, 199,975 LUT, 50 URAM.
- Percentage utilization: 45% BRAM, 3% DSP, 11% FF, 86% LUT, 52% URAM.
- Dominant instance: `hgtxr_e2e_controller_run`, 555,329,660..556,054,076 cycles, 204 BRAM, 64 DSP, 46,343 FF, 188,242 LUT, 50 URAM.
