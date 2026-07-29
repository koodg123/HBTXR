# Eight-Question Full-Learned Transformer Experiment - 2026-06-28

## Scope

This note records the current Question / Plan / Progress / Results state for the
eight Search/Track hybrid reporting questions after restoring the non-fastpath
learned Transformer path for the E2E AXIS top.

Target requested by the user:

- Single E2E AXIS cyclic accelerator top.
- Shared learned ATTN/MLP datapath for Search and Track runtime modes.
- Frame/Event Conv, Transformer, nonlinear LUTs, and Head parameters on a learned
  parameter path. For this experiment, deterministic placeholder learned
  parameters are acceptable when software and hardware observe the same values.
- HLS clock target: 300 MHz.
- ZCU104 latency goals:
  - Search <= 4 ms.
  - Track <= 1 ms.
- Hybrid invocation distribution:
  - Search 10%.
  - Track 90%.

## Prompt Brief

Goal:

- Implement the full learned Transformer datapath in HLS/Vivado flow far enough
  to answer the eight measurement questions with evidence.

Inputs and evidence sources:

- HLS source and runtime scheduler.
- HLS C simulation output.
- HLS C synthesis reports.
- Existing Vivado routed reports only when they match the current candidate.

Assumptions and unknowns:

- HLS min latency is treated as the Track-mode envelope.
- HLS max latency is treated as the Search-mode envelope.
- The current experiment uses deterministic on-chip parameter ROM values for the
  learned parameter path, not a loaded training checkpoint.
- Mode-specific p95/p99 latency and mode-specific power are unavailable until
  RTL or board traces are collected.

Constraints and safety limits:

- Do not claim a routed Vivado implementation for a candidate that fails HLS
  ZCU104 fit.
- Do not reuse power numbers from an older overlay as if they apply to this
  full learned 300 MHz candidate.
- Separate HLS estimates from routed implementation evidence.

Acceptance criteria:

- Search and Track CSim both pass with the expected runtime-state markers.
- Learned dense paths for Q/K/V, output projection, and MLP use the shared
  weight-loading mechanism rather than bypassing the Transformer.
- HLS csynth meets the 300 MHz timing estimate.
- ZCU104 fit must be checked with routed Vivado utilization when implementation
  is available; HLS utilization is treated as an estimate and can overstate LUT
  cost.
- Routed full learned evidence is accepted only when DSP/BRAM/URAM order of
  magnitude matches the learned datapath, not a collapsed AXI shell.

## Historical Answer Snapshot: Event-Conv ROM-Only DSP3 Candidate

This section is retained as experiment history. The current answer baseline is
the later full-block dispatcher build in the next section.

Current implementation profile:

- `par8_runtime_rom_only_dispatch_dsp3_300_mem8`
- HLS project:
  `hardware/generated/hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_dsp3_300_mem8_no_board/solution_e2e_q4w8a`
- Vivado project:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par8_runtime_rom_only_dispatch_dsp3_300_mem8_overlay`
- Runtime Search/Track scheduler is enabled.
- Weight AXI is omitted; valid model parameters resolve through deterministic
  on-chip parameter ROM.
- Frame Conv and Event Conv now have distinct ROM regions.
- Search mode uses the Frame Conv path.
- Track mode uses the Event Conv path.
- Q/K/V, output projection, MLP, LayerNorm, nonlinear LUTs, and Head are on the
  shared learned ROM-backed path.

Validation commands completed:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8
```

Functional result:

- CSim passed Search and Track runtime mode checks.
- Search output words after the Event Conv integration run:
  `[-712, -712, -712, -712, -712, -696]`.
- Track output words after the Event Conv integration run:
  `[-235, -235, -235, -235, -235, -299]`.
- Search `runtime_state=0`, Track `runtime_state=1`, output count `6`, TLAST
  correct, and `CSim done with 0 errors`.

HLS package result:

- Package/export succeeded.
- HLS timing estimate after Event Conv is worse than target:
  target `3.33 ns`, estimated `3.495 ns`.
- The physical Vivado route still closes 300 MHz after placement/routing
  optimization, so final timing evidence is the routed timing report, not the
  HLS estimate.

Routed timing:

| Metric | Result |
|---|---:|
| Clock | `clk_pl_0` |
| Period | `3.333 ns` |
| Frequency | `300.030 MHz` |
| WNS | `0.007 ns` |
| TNS | `0.000 ns` |
| WHS | `0.010 ns` |
| THS | `0.000 ns` |
| Failing setup endpoints | `0` |
| Failing hold endpoints | `0` |
| Route errors | `0` |
| Fully routed nets | `107,641 / 107,641` |
| Bitstream | generated successfully |

Latency at routed clock:

| Mode/envelope | Cycles | Latency @ 300.030 MHz | Rate |
|---|---:|---:|---:|
| Track-bound min | 1,605,053 | 5.350 ms | 186.93 Hz |
| Search-bound max | 12,989,355 | 43.294 ms | 23.10 Hz |
| Hybrid 10% Search / 90% Track | 2,743,483.2 | 9.144 ms | 109.36 Hz |
| Worst case | 12,989,355 | 43.294 ms | 23.10 Hz |

HLS report absolute latency uses the HLS estimated `3.495 ns` clock, so the
same report also shows `5.610 ms` min and `45.398 ms` max. For 300 MHz target
comparison, use the routed-clock conversion above.

Routed placed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUTs | 24,159 | 230,400 | 10.49% |
| CLB Registers | 26,753 | 460,800 | 5.81% |
| Block RAM Tile | 238 | 312 | 76.28% |
| RAMB36/FIFO | 224 | 312 | 71.79% |
| RAMB18 | 28 | 624 | 4.49% |
| DSP48E2 | 1,627 | 1,728 | 94.16% |
| URAM | 32 | 96 | 33.33% |

HLS OOC utilization estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 386 | 624 | 61% |
| DSP | 1,624 | 1,728 | 93% |
| FF | 167,547 | 460,800 | 36% |
| LUT | 224,356 | 230,400 | 97% |
| URAM | 32 | 96 | 33% |

Resource interpretation:

- The routed wrapper-level CLB LUT number is much lower than the HLS OOC
  estimate because Vivado performs OOC and top-level logic optimization,
  constant propagation, LUT packing, and AXI/control simplification.
- DSP, BRAM, and URAM scale confirms that the full learned compute datapath is
  physically present: routed `1,627` DSP and `32` URAM match the learned
  Transformer scale.
- For paper-style major-block breakdown, report both:
  routed top-level utilization as final implementation fit, and HLS OOC instance
  estimates for block attribution.

Routed vectorless power:

| Bucket | Power |
|---|---:|
| Total on-chip | 6.037 W |
| Dynamic | 5.322 W |
| Device static | 0.715 W |
| PS static | 0.102 W |
| PL static | 0.614 W |
| Clocks | 0.272 W |
| CLB logic | 0.291 W |
| Signals | 0.443 W |
| Block RAM | 0.267 W |
| URAM | 0.083 W |
| DSPs | 1.295 W |
| PS8 | 2.671 W |

Routed dynamic power by hierarchy:

| Hierarchy | Dynamic power |
|---|---:|
| `hgtxr_e2e_axis_top_0` | 2.408 W |
| `psu` | 2.673 W |
| `axi_mem` | 0.187 W |
| `axi_ctrl` | 0.020 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.026 W |

DDR-related PS rails in the vectorless report:

| Rail | Total current | Dynamic current | Static current |
|---|---:|---:|---:|
| `VCC_PSINTFP_DDR` | 0.719 A | 0.714 A | 0.005 A |
| `VCCO_PSDDR_504` | 0.620 A | 0.586 A | 0.034 A |
| `VCC_PSDDR_PLL` | 0.001 A | 0.000 A | 0.001 A |

Status against user goals:

- Full learned Transformer path: implemented and routed.
- Frame Conv ROM path: implemented.
- Event Conv ROM path: implemented and used by Track mode.
- Head ROM path: implemented.
- Track Transformer block ROM path: implemented.
- Nonlinear LUT ROM path: implemented.
- Runtime scheduler: implemented.
- ZCU104 route/bitgen: pass.
- 300 MHz routed timing: pass.
- Search latency target `<= 4 ms`: fail, current routed-clock estimate
  `43.294 ms`.
- Track latency target `<= 1 ms`: fail, current routed-clock estimate
  `5.350 ms`.
- Mode-specific p95/p99 latency and measured rail power: not available until
  board or RTL trace experiments are run.

## Current Answer Snapshot: Full-Block Dispatcher Force-Mode Candidate

Current implementation profile:

- `par8_runtime_rom_only_dispatch_dsp3_300_mem8`
- Search-only HLS profile:
  `par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only`
- Track-only HLS profile:
  `par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only`
- Vivado routed profile:
  `hgtxr_e2e_axis_dma_par8_runtime_rom_only_dispatch_dsp3_300_mem8_overlay`
- Weight AXI is omitted; deterministic learned parameters are served from
  on-chip ROM/cache paths.
- Full-block dispatcher prefetch bank is enabled and mapped to URAM.

Validation commands completed:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8
```

Functional result:

- Search-only CSim passed with output words
  `[-712, -712, -712, -712, -712, -676]`.
- Track-only CSim passed with output words
  `[-235, -235, -235, -235, -235, -226]`.
- Search `runtime_state=0`, Track `runtime_state=1`, output count `6`, TLAST
  correct, and `CSim done with 0 errors`.
- Search-only CSim prefetch trace proved ordered prefetch-before-use for the
  extra Search blocks observed by the trace:
  `prefetch_trace_done block=2 seq=1`, `block=3 seq=2`;
  first load hits occurred later at `seq=3/4`; summary `violations=0`.
- This trace proves sequential full-block preload before use. It does not prove
  overlapped dataflow/double-buffer execution because the current HLS top still
  executes at transaction level with `Pipeline Type = no`.

Mode-specific csynth result:

| Mode | HLS target | HLS estimated | Max cycles | Latency | Interval max | Target status |
|---|---:|---:|---:|---:|---:|---|
| Search-only | 3.33 ns | 2.924 ns | 13,001,422 | 43.334 ms | 13,001,423 | Fail, `>4 ms` |
| Track-only | 3.33 ns | 3.473 ns | 1,606,660 | 5.580 ms in report, 5.355 ms at 300.030 MHz | 1,606,661 | Fail, `>1 ms` |

Hybrid 10% Search / 90% Track:

| Metric | Value |
|---|---:|
| Expected cycles | 2,746,136.2 |
| Expected latency at 300.030 MHz | 9.153 ms |
| Expected throughput | 109.25 events/s |
| Worst-case cycles | 13,001,422 |
| Worst-case latency | 43.334 ms |

Latest routed timing for the matching full-block build:

| Metric | Result |
|---|---:|
| Clock | `clk_pl_0` |
| Frequency | `300.030 MHz` |
| WNS | `0.010 ns` |
| TNS | `0.000 ns` |
| WHS | `0.009 ns` |
| THS | `0.000 ns` |
| Failing setup endpoints | `0 / 223,037` |
| Route errors | `0` |
| Fully routed nets | `141,580 / 141,580` |
| Bitstream | generated successfully |

Latest routed placed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUTs | 44,894 | 230,400 | 19.49% |
| CLB Registers | 44,799 | 460,800 | 9.72% |
| Block RAM Tile | 238 | 312 | 76.28% |
| DSP48E2 | 1,630 | 1,728 | 94.33% |
| URAM | 48 | 96 | 50.00% |

Latest routed vectorless power:

| Bucket | Power |
|---|---:|
| Total on-chip | 7.109 W |
| Dynamic | 6.383 W |
| Device static | 0.725 W |
| PS static | 0.103 W |
| PL static | 0.622 W |

Latest routed hierarchy power:

| Hierarchy | Power |
|---|---:|
| `hgtxr_e2e_axis_top_0` | 3.467 W |
| `psu` | 2.673 W |
| `axi_mem` | 0.188 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.027 W |

Status against user goals:

- Full learned Transformer path: implemented and routed.
- Frame Conv ROM path: implemented.
- Event Conv ROM path: implemented and used by Track mode.
- Head ROM path: implemented.
- Track Transformer block ROM path: implemented.
- Nonlinear LUT ROM path: implemented.
- Runtime scheduler: implemented.
- ZCU104 route/bitgen: pass.
- 300 MHz routed timing: pass.
- Search latency target `<= 4 ms`: fail, current Search-only csynth max
  `43.334 ms`.
- Track latency target `<= 1 ms`: fail, current Track-only csynth max
  `5.580 ms` in HLS report and `5.355 ms` at routed 300.030 MHz.
- Mode-specific p95/p99 latency and measured rail power: not available until
  board or RTL activity experiments are run.

## Current Answers To The Eight Questions

### 1. HW Top Total Resource, Hybrid Inference Latency, Hybrid Power Breakdown

Result:

- Routed top resources: CLB LUT `44,894/230,400 = 19.49%`, CLB registers
  `44,799/460,800 = 9.72%`, Block RAM Tile `238/312 = 76.28%`, DSP48E2
  `1,630/1,728 = 94.33%`, URAM `48/96 = 50.00%`.
- Search-only HLS OOC estimate: BRAM_18K `386/624 = 61%`, DSP
  `1,497/1,728 = 86%`, FF `161,370/460,800 = 35%`, LUT
  `236,034/230,400 = 102%`, URAM `48/96 = 50%`.
- Track-only HLS OOC estimate: BRAM_18K `378/624 = 60%`, DSP
  `1,108/1,728 = 64%`, FF `157,213/460,800 = 34%`, LUT
  `197,528/230,400 = 85%`, URAM `32/96 = 33%`.
- Hybrid latency at Search 10% / Track 90%: `2,746,136.2` cycles,
  `9.153 ms` at `300.030 MHz`.
- Hybrid throughput: `109.25 events/s`.
- Power: Vivado post-route vectorless total on-chip `7.109 W`, dynamic
  `6.383 W`, static `0.725 W`; no Search/Track SAIF or board rail activity
  trace is available yet, so hybrid mode-specific power is not separately
  measured.

### 2. Search Mode Inference Latency And Power Breakdown

Result:

- Search-only max latency: `13,001,422` cycles, `43.334 ms`.
- Search-only interval max: `13,001,423` cycles.
- Search-only throughput: `23.08 events/s`.
- Search target `<=4 ms`: not met.
- Current Search power: not separately measured. Use routed vectorless envelope:
  `hgtxr_e2e_axis_top_0 = 3.467 W`, total on-chip `7.109 W`.

### 3. Search Mode Active Block And Path

Result:

- Active path: AXIS frame input -> Frame Conv ROM embedding -> shared cyclic
  ATTN/MLP block pair scheduler -> learned Head ROM -> AXIS state output.
- Active Transformer depth: Search schedule requests `HGTXR_SEARCH_DEPTH`,
  bounded by `HGTXR_E2E_BLOCKS`; current profile uses active token cap `64` and
  block-pair execution through the shared controller.
- Shared resources: the same two ATTN/MLP datapath instances are temporally
  reused across Search and Track runtime modes.
- Prefetch trace result: full-block prefetch-before-use is proven in CSim for
  traced Search blocks `2` and `3` with `violations=0`.
- Limitation: Search dispatcher is not yet a proven fully overlapped
  double-buffer prefetch engine.

### 4. Track Mode Inference Latency And Power Breakdown

Result:

- Track-only max latency: `1,606,660` cycles.
- HLS report latency: `5.580 ms`; routed-clock conversion at `300.030 MHz`:
  `5.355 ms`.
- Track-only interval max: `1,606,661` cycles.
- Track-only throughput: `186.74 events/s`.
- Track target `<=1 ms`: not met.
- Current Track power: not separately measured. Use routed vectorless envelope:
  `hgtxr_e2e_axis_top_0 = 3.467 W`, total on-chip `7.109 W`.

### 5. Track Mode Active Block And Path

Result:

- Active path: AXIS frame input -> derived event-delta buffer -> Event Conv ROM
  embedding -> shared cyclic ATTN/MLP block pair scheduler -> learned Head ROM
  -> AXIS state output.
- Active Transformer depth: Track schedule requests `HGTXR_TRACK_CUT_DEPTH`,
  bounded by `HGTXR_E2E_BLOCKS`; current profile uses active token cap `16` for
  the Track envelope.
- Track uses the same physical ATTN/MLP datapath as Search; mode selection only
  changes token count, depth, and embedding path.
- Limitation: there is no separate external event AXIS input stream yet.

### 6. Search/Track Only And Hybrid Throughput, II, Max Rate, Worst-Case Latency

Result:

| Scenario | Interval/cycles | Latency | Max rate |
|---|---:|---:|---:|
| Track only | 1,606,661 | 5.355 ms at routed clock | 186.74 Hz |
| Search only | 13,001,423 | 43.334 ms | 23.08 Hz |
| Hybrid 10/90 | 2,746,137.2 | 9.153 ms | 109.25 Hz |
| Worst case | 13,001,423 | 43.334 ms | 23.08 Hz |

- Transaction-level pipeline: `no` in HLS report.
- Effective II/interval is approximately latency plus one cycle.

### 7. Power Methodology, Inclusion Scope, DMA Bandwidth, Batch, Throughput, Latency Distribution

Result:

- Power methodology: Vivado post-route vectorless `report_power`, no
  mode-specific SAIF/VCD activity file yet.
- Included: PL fabric dynamic/static, PS8 model, PS static/PL static, DMA/AXI
  interconnect hierarchy, PS DDR-related rails in the power supply summary.
- Not included as measured evidence: sensor I/O, external board rail
  instrumentation, real DDR traffic trace, thermally stabilized board power.
- DMA bandwidth: not measured in this experiment. Current top has AXIS DMA
  input/output and `m_axi_gmem_e2e_runtime`; weight AXI is intentionally omitted.
- Batch size: single frame/event transaction per invocation.
- Throughput: Track `186.74 Hz`, Search `23.08 Hz`, 10/90 hybrid
  `109.25 Hz` at routed clock.
- Mode-specific latency: HLS deterministic envelope only; board p95/p99 are not
  available.
- Invocation distribution used for hybrid math: Search `10%`, Track `90%`.

### 8. Source Utilization Absolute/Percent And Major Block Breakdown

Result:

- Final routed source utilization uses the Vivado placed report:
  CLB LUT `44,894/230,400 = 19.49%`, FF `44,799/460,800 = 9.72%`,
  Block RAM Tile `238/312 = 76.28%`, DSP `1,630/1,728 = 94.33%`, URAM
  `48/96 = 50.00%`.
- Search-only major HLS block attribution:
  - `hgtxr_e2e_controller_run`: BRAM_18K `342`, DSP `1,489`, FF `159,077`,
    LUT `229,948`, URAM `48`.
  - `hgtxr_e2e_mlp_head`: BRAM_18K `0`, DSP `8`, FF `735`, LUT `2,544`,
    URAM `0`.
  - `hgtxr_conv_patch_embedding`: BRAM_18K `0`, DSP `0`, FF `90`,
    LUT `845`, URAM `0`.
  - `hgtxr_global_buffer_load`: BRAM_18K `0`, DSP `0`, FF `142`,
    LUT `284`, URAM `0`.
- Track-only major HLS block attribution:
  - `hgtxr_e2e_controller_run`: BRAM_18K `334`, DSP `1,100`, FF `154,826`,
    LUT `191,446`, URAM `32`.
  - `hgtxr_e2e_mlp_head`: BRAM_18K `0`, DSP `8`, FF `731`, LUT `2,540`,
    URAM `0`.
  - `hgtxr_event_conv_patch_embedding`: BRAM_18K `0`, DSP `0`, FF `91`,
    LUT `854`, URAM `0`.
  - `hgtxr_global_buffer_load`: BRAM_18K `0`, DSP `0`, FF `241`,
    LUT `279`, URAM `0`.
- Use routed utilization for final ZCU104 fit; use HLS instance estimates for
  major-block attribution.

## Historical Implementation Summary: Earlier ROM Dispatch Candidates

The current answer baseline is the full-block dispatcher force-mode candidate
above. This lower section is retained to preserve earlier candidate evidence and
should not be used as the latest answer snapshot.

Profile:

- Primary script profile: `par8_runtime_rom_dispatch_300_mem8`
- HLS scale: `runtime_rom_dispatch_active64_b8_ff768`
- HLS clock target: `3.333 ns`
- Search configuration: 64 active tokens, depth 8, reported runtime state `0`
- Track configuration: 16 active tokens, cut depth 4, reported runtime state `1`
- With `HGTXR_E2E_MODE_DEPTH_COUNTS_LAYERS=1`, the controller executes
  `(depth_limit + 1) / 2` ATTN/MLP block pairs.

Code changes made for this experiment:

- Added mixed ROM/AXI packed weight loading:
  - `hgtxr_e2e_pack_weight_word_from_rom`
  - `hgtxr_e2e_load_weight_word`
- Routed packed Q/K/V, output projection, MLP W1, MLP W2, and LayerNorm
  parameter loads through the shared loader.
- Added local packed caches for Q/K/V, output projection, and MLP weights to
  reduce inner-loop AXI read pressure.
- Enabled the current profile flags:
  - `HGTXR_E2E_USE_ONCHIP_PARAM_ROM=1`
  - `HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM=1`
  - `HGTXR_E2E_USE_SEARCH_WEIGHT_DISPATCHER=1`
  - `HGTXR_E2E_QKV_WEIGHT_CACHE=1`
  - `HGTXR_E2E_LN_PARAM_CACHE=1`
  - `HGTXR_E2E_WEIGHT_VEC_CACHE=1`

Important boundary:

- The current Search dispatcher has marker/preload behavior and the dense weights
  are loaded into local caches at each operator entry. It is not yet a fully
  overlapped double-buffer weight prefetch engine across Search layers.
- Event Conv has now been wired into the E2E AXIS top for Track mode, but this is
  still a derived event-delta path from the input frame buffer, not a separate
  external AXIS event input stream.

## Validation Evidence

CSim command:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_dispatch_300_mem16
```

CSim result:

- Search output words: `-221, -237, -237, -237, -237, -237`
- Search `runtime_state=0`, expected `0`, count `6`, TLAST `1`, failures `0`
- Track output words: `-219, -235, -235, -235, -235, -235`
- Track `runtime_state=1`, expected `1`, count `6`, TLAST `1`
- `E2E AXIS vector comparison passed`
- `CSim done with 0 errors`

Par32 C synthesis command:

```sh
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_dispatch_300_mem16
```

Par16 fit probe command:

```sh
env HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_par16_runtime_rom_dispatch_300_mem16_no_board \
  HGTXR_E2E_PAR=16 \
  HGTXR_E2E_MEM_BANK_PAR=16 \
  timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_dispatch_300_mem16
```

## Candidate Results

### Par32 Runtime ROM Dispatch 300 MHz

Report:

- `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_dispatch_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`

Timing:

- Target clock: `3.333 ns`
- Estimated clock: `2.777 ns`
- Estimated Fmax: `360.10 MHz`

Latency:

| Envelope | Cycles | Latency @ 300 MHz |
|---|---:|---:|
| Track-bound min | 605,592 | 2.018 ms |
| Search-bound max | 4,733,900 | 15.778 ms |
| Hybrid expected, Search 10% / Track 90% | 1,018,423 | 3.394 ms |
| Worst case | 4,733,900 | 15.778 ms |

Throughput:

| Scenario | Interval cycles | Max rate @ 300 MHz |
|---|---:|---:|
| Track-only | 605,593 | 495.38 Hz |
| Search-only | 4,733,901 | 63.37 Hz |
| Hybrid, Search 10% / Track 90% | 1,018,424 | 294.57 Hz |

HLS resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 440 | 624 | 70% |
| DSP | 1,634 | 1,728 | 94% |
| FF | 213,114 | 460,800 | 46% |
| LUT | 287,796 | 230,400 | 124% |
| URAM | 32 | 96 | 33% |

Major HLS instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 358 | 1,600 | 206,365 | 267,998 | 32 |
| `hgtxr_e2e_mlp_head` | 0 | 34 | n/a | 13,722 | 0 |

Status:

- 300 MHz HLS timing estimate passes.
- ZCU104 fit fails because LUT usage is 124%.
- Vivado place/route and routed power are not valid next steps for this
  candidate until LUT usage is reduced.

### Par16 Runtime ROM Dispatch 300 MHz Fit Probe

Report:

- `hardware/generated/hgtxr_e2e_axis_par16_runtime_rom_dispatch_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`

Timing:

- Target clock: `3.333 ns`
- Estimated clock: `2.924 ns`
- Estimated Fmax: `342.00 MHz`

Latency:

| Envelope | Cycles | Latency @ 300 MHz |
|---|---:|---:|
| Track-bound min | 836,738 | 2.789 ms |
| Search-bound max | 6,656,520 | 22.186 ms |
| Hybrid expected, Search 10% / Track 90% | 1,418,716 | 4.729 ms |
| Worst case | 6,656,520 | 22.186 ms |

Throughput:

| Scenario | Interval cycles | Max rate @ 300 MHz |
|---|---:|---:|
| Track-only | 836,739 | 358.53 Hz |
| Search-only | 6,656,521 | 45.07 Hz |
| Hybrid, Search 10% / Track 90% | 1,418,717 | 211.46 Hz |

HLS resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 440 | 624 | 70% |
| DSP | 1,456 | 1,728 | 84% |
| FF | 197,657 | 460,800 | 42% |
| LUT | 249,421 | 230,400 | 108% |
| URAM | 32 | 96 | 33% |

Major HLS instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 358 | 1,440 | 192,615 | 236,322 | 32 |
| `hgtxr_e2e_mlp_head` | 0 | 16 | n/a | 7,023 | 0 |

Status:

- 300 MHz HLS timing estimate passes.
- ZCU104 fit still fails because LUT usage is 108%.
- Latency is worse than Par32.

### Par8 Runtime ROM Dispatch 300 MHz Fit And Vivado Probe

Profile:

- Script profile: `par8_runtime_rom_dispatch_300_mem8`
- HLS scale: `runtime_rom_dispatch_active64_b8_ff768`
- Parallelism: `E2E_PAR=8`, `MEM_BANK_PAR=8`
- HLS clock target: `3.333 ns`
- Vivado PL clock target: `300.0 MHz`
- Current preservation mode: `HGTXR_E2E_OOC_DONT_TOUCH=datapath`
- Current DSP timing hint: `HGTXR_E2E_DSP_MUL_LATENCY=2`
- Current AXI timing hint: `HGTXR_E2E_LOW_FANOUT_WEIGHT_AXI=1`, using
  `max_read_burst_length=16` and `num_read_outstanding=2` on the weight AXI
  master.

CSim command:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_dispatch_300_mem8
```

CSim result:

- Search output words: `-221, -237, -237, -237, -237, -274`
- Search `runtime_state=0`, expected `0`, count `6`, TLAST `1`, failures `0`
- Track output words: `-219, -235, -235, -235, -235, -176`
- Track `runtime_state=1`, expected `1`, count `6`, TLAST `1`
- `E2E AXIS vector comparison passed`
- `CSim done with 0 errors`

CSynth command:

```sh
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par8_runtime_rom_dispatch_300_mem8
```

Report:

- `hardware/generated/hgtxr_e2e_axis_par8_runtime_rom_dispatch_300_mem8_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`

Timing:

- Target clock: `3.333 ns`
- Estimated clock: `2.924 ns`
- Estimated Fmax: `342.00 MHz`

Latency:

| Envelope | Cycles | Latency @ 300 MHz target |
|---|---:|---:|
| Track-bound min | 1,588,174 | 5.293 ms |
| Search-bound max | 12,929,562 | 43.094 ms |
| Hybrid expected, Search 10% / Track 90% | 2,722,313 | 9.074 ms |
| Worst case | 12,929,562 | 43.094 ms |

Throughput:

| Scenario | Interval cycles | Max rate @ 300 MHz |
|---|---:|---:|
| Track-only | 1,588,174 | 188.90 Hz |
| Search-only | 12,929,562 | 23.20 Hz |
| Hybrid, Search 10% / Track 90% | 2,722,313 | 110.20 Hz |

HLS resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 408 | 624 | 65% |
| DSP | 1,368 | 1,728 | 79% |
| FF | 186,572 | 460,800 | 40% |
| LUT | 222,223 | 230,400 | 96% |
| URAM | 32 | 96 | 33% |

Major HLS instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 334 | 1,360 | 182,303 | 214,396 | 32 |
| `hgtxr_e2e_mlp_head` | 0 | 8 | 1,603 | 3,118 | 0 |

Package command:

```sh
timeout 3600 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_dispatch_300_mem8
```

Package result:

- `impl/ip/component.xml` exists.
- `impl/export.zip` exists.
- `component.xml` contains `m_axi_gmem_e2e_weights` and
  `m_axi_gmem_e2e_runtime`.
- The profile exports `HGTXR_E2E_OOC_DONT_TOUCH=datapath`.
- The package script appended `DONT_TOUCH` and `KEEP_HIERARCHY` constraints to
  `impl/ip/constraints/hgtxr_e2e_axis_top_ooc.xdc`.
- HLS RTL generation logs include the learned QKV, attention, MLP, nonlinear
  ROM, BRAM, URAM, and multiplier modules; this confirms RTL generation, but
  not final physical preservation.
- The first OOC XDC-only preservation attempts were ineffective because the
  generated Vivado OOC run marked the HLS IP constraint file
  `used_in_implementation false`.
- The package script now injects RTL module attributes directly into exported
  HLS Verilog when `HGTXR_E2E_OOC_DONT_TOUCH=all`:
  `(* keep_hierarchy = "yes", dont_touch = "yes" *)`.
- The first strict physical-preservation run patched `61` generated Verilog
  module files with `HGTXR_E2E_OOC_DONT_TOUCH=all`.
- The current timing-closure run patches `38` generated datapath Verilog module
  files with `HGTXR_E2E_OOC_DONT_TOUCH=datapath`, leaving AXI and control
  wrapper modules available for Vivado optimization.

Vivado command:

```sh
sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_dispatch_300_mem8
```

Routed artifacts:

- `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par8_runtime_rom_dispatch_300_mem8_overlay/hgtxr_e2e_axis_dma_par8_runtime_rom_dispatch_300_mem8.bit`
- `.hwh` was copied from the BD handoff when no implementation-local `.hwh`
  was emitted.

Routed timing:

- Clock: `clk_pl_0`, period `3.333 ns`, frequency `300.030 MHz`
- WNS `-0.265 ns`, TNS `-113.997 ns`
- WHS `0.010 ns`, THS `0.000 ns`
- Route status: fully routed `122,577/122,577`, route errors `0`
- Bitgen completed successfully, but timing constraints are not met.
- Worst setup path:
  - Source: `gmem_e2e_weights_m_axi_U/load_unit/fifo_rreq/full_n_reg/C`
  - Destination:
    `gmem_e2e_weights_m_axi_U/load_unit/buff_rdata/U_fifo_mem/mem_reg_2/ENARDEN`
  - Data path delay `3.030 ns`, with route `2.421 ns = 79.903%`
  - Logic levels `7`, through `m_axi_gmem_e2e_weights_ARREADY/RREADY`

Routed placed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUT | 37,300 | 230,400 | 16.19% |
| CLB registers | 40,008 | 460,800 | 8.68% |
| Block RAM Tile | 242 | 312 | 77.56% |
| RAMB36/FIFO | 228 | 312 | 73.08% |
| RAMB18 | 28 | 624 | 4.49% |
| DSP | 1,371 | 1,728 | 79.34% |
| URAM | 32 | 96 | 33.33% |

Routed power report:

| Bucket | Power |
|---|---:|
| Total on-chip | 6.817 W |
| Dynamic | 6.096 W |
| Device static | 0.721 W |
| PS static | 0.103 W |
| PL static | 0.619 W |
| PS8 component | 2.671 W |
| Clocks | 0.380 W |
| CLB logic | 0.473 W |
| Signals | 0.741 W |
| Block RAM | 0.285 W |
| URAM | 0.084 W |
| DSPs | 1.462 W |

Routed hierarchy and DDR-related rails:

| Bucket | Power |
|---|---:|
| `axi_ctrl` | 0.020 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.025 W |
| `axi_mem` | 0.288 W |
| `hgtxr_e2e_axis_top_0` | 3.079 W |
| `psu` | 2.676 W |
| `VCC_PSINTFP_DDR` | 0.719 W total, 0.714 W dynamic |
| `VCCO_PSDDR_504` | 0.620 W total, 0.586 W dynamic |
| `VCC_PSDDR_PLL` | 0.001 W total |

OOC preservation and implementation result:

- Attempt 1: append selective `DONT_TOUCH` / `KEEP_HIERARCHY` constraints for
  HLS top and data-mover related cells. Vivado completed, but the OOC HLS IP
  remained `3,048 LUT`, `3 DSP`, `1 Block RAM Tile`, `0 URAM`.
- Attempt 2: append `DONT_TOUCH` / `KEEP_HIERARCHY` to all cells selected by
  `get_cells -hier -quiet *` in the HLS IP OOC XDC. Vivado completed and timing
  closed, but the OOC HLS IP utilization stayed at the same collapsed scale.
- The build script recreates the Vivado build directory, so this is not simply
  an old top-level implementation cache. The remaining issue is that the full
  learned computations are not physically preserved through OOC synthesis, or
  the OOC constraint is not being applied early enough to protect the relevant
  generated RTL structure.
- Final fix in this run: inject RTL-level attributes into the exported HLS
  Verilog modules before Vivado imports the IP. This preserved the full learned
  datapath through OOC/top implementation.
- Evidence: linked top-level synthesis transformed `1,371` `DSP48E2`
  instances; placed utilization reports `1,371 DSP`, `242` Block RAM tiles,
  and `32 URAM`.
- The current lower CLB LUT count is expected: datapath-only preservation still
  keeps the learned compute memories/DSP datapath at full scale, but lets Vivado
  simplify AXI/control glue, unused write-side weight paths, and observable-only
  surrounding logic. The HLS LUT estimate is also pessimistic compared with
  routed LUT packing and logic sharing.
- Low-fanout AXI iteration result: reducing the `gmem_e2e_weights` read master
  burst/outstanding depth improved routed WNS from `-0.300 ns` to `-0.265 ns`,
  but the same read-side FIFO/RREADY path remains the worst setup path.

Critical interpretation:

- The latest HLS csynth estimate says the full learned Par8 datapath consumes
  `1,368 DSP`, `222,223 LUT`, `408 BRAM_18K`, and `32 URAM`.
- The current routed Vivado design reports `1,371 DSP`, `37,300 CLB LUT`,
  `242 Block RAM Tile`, and `32 URAM`.
- Therefore the full learned compute scale is physically present in the routed
  design through DSP, BRAM, and URAM usage; the much lower routed LUT count is
  caused by Vivado optimization, not by losing the dense learned compute arrays.
- The HLS LUT estimate was overly pessimistic after Vivado logic optimization;
  DSP/URAM scale matches closely.
- The design still fails 300 MHz timing by `0.265 ns`, so the bitstream is not a
  timing-clean 300 MHz implementation.
- The routed power report is a Vivado vectorless estimate with medium
  confidence. It is valid as a post-route estimate for this full learned netlist,
  but not a measured mode-specific or board-level power result.

Status:

- Functional CSim: pass.
- 300 MHz HLS timing estimate: pass.
- Routed ZCU104 fit: pass for resources.
- Search target `<= 4 ms`: fail, `43.094 ms`.
- Track target `<= 1 ms`: fail, `5.293 ms`.
- Vivado route/bitgen: pass.
- 300 MHz timing closure: fail, WNS `-0.265 ns`.

## Eight Questions

### 1. HW Top Total Resource, Hybrid Inference Latency, Hybrid Power Breakdown

Question:

- Report the full HW top resource usage and the hybrid latency/power for Search
  10% and Track 90%.

Plan:

- Use routed Vivado utilization for the implemented Par8 top.
- Compute hybrid latency as `0.1 * Search + 0.9 * Track`.
- Use routed Vivado power as vectorless post-route estimate, and keep it
  separate from mode-specific measured power.

Progress:

- Par32, Par16, and Par8 300 MHz csynth completed.
- Par32 and Par16 fail ZCU104 LUT fit.
- Par8 is the first HLS-level fit candidate, but it fails the latency goals.
- Par8 Vivado route/bitgen completed and preserves the full learned datapath,
  but final timing at 300 MHz fails with WNS `-0.265 ns`.

Results:

- Best current latency candidate: Par32.
  - Hybrid expected latency: `3.394 ms`.
  - Worst-case latency: `15.778 ms`.
- Best current area candidate: Par16.
  - Hybrid expected latency: `4.729 ms`.
  - Worst-case latency: `22.186 ms`.
- Best current HLS fit candidate: Par8.
  - Hybrid expected latency: `9.074 ms`.
  - Worst-case latency: `43.094 ms`.
  - Routed resources: CLB LUT `37,300/230,400 = 16.19%`, registers
    `40,008/460,800 = 8.68%`, Block RAM Tile `242/312 = 77.56%`, DSP
    `1,371/1,728 = 79.34%`, URAM `32/96 = 33.33%`.
  - Routed power estimate: total on-chip `6.817 W`, dynamic `6.096 W`,
    static `0.721 W`.
  - Hybrid power is not separately measured; with vectorless activity the same
    post-route power report is the current envelope for the implemented top.

### 2. Search Mode Inference Latency And Power Breakdown

Question:

- Report Search-only inference latency and power.

Plan:

- Use HLS max latency as the current Search envelope.
- Use mode-specific SAIF/VCD or board rails for Search power after fit.

Progress:

- Search CSim passed with `runtime_state=0`.
- Search path uses the runtime scheduler and shared learned ATTN/MLP datapath.

Results:

- Par32 Search latency: `4,733,900 cycles`, `15.778 ms @ 300 MHz`.
- Par16 Search latency: `6,656,520 cycles`, `22.186 ms @ 300 MHz`.
- Par8 Search latency: `12,929,562 cycles`, `43.094 ms @ 300 MHz target`.
- Search target `<= 4 ms`: not met.
- Search-specific full learned power: not measured. Current full-top routed
  vectorless power estimate is total `6.817 W`, dynamic `6.096 W`.

### 3. Search Mode Active Block And Path

Question:

- Report the active blocks and path for Search mode.

Plan:

- Trace runtime scheduler, active token count, depth, and shared learned
  ATTN/MLP calls.

Progress:

- Runtime scheduler and CSim mode marker are confirmed.

Results:

- Search active tokens: `64`.
- Search requested depth: `8`.
- With depth-counts-layers mode enabled, controller block pairs:
  `(8 + 1) / 2 = 4`.
- Active path:
  - AXIS input.
  - Patch/frame embedding path.
  - Global buffer.
  - Runtime scheduler selects Search.
  - Shared Q/K/V learned projection.
  - Shared attention path.
  - Shared output projection.
  - Shared MLP W1/W2 path.
  - Learned head.
  - AXIS state output and runtime-state report.

### 4. Track Mode Inference Latency And Power Breakdown

Question:

- Report Track-only inference latency and power.

Plan:

- Use HLS min latency as the current Track envelope.
- Use mode-specific SAIF/VCD or board rails for Track power after fit.

Progress:

- Track CSim passed with `runtime_state=1`.
- Track path uses the same shared learned ATTN/MLP datapath with smaller active
  token and depth limits.

Results:

- Par32 Track latency: `605,592 cycles`, `2.018 ms @ 300 MHz`.
- Par16 Track latency: `836,738 cycles`, `2.789 ms @ 300 MHz`.
- Par8 Track latency: `1,588,174 cycles`, `5.293 ms @ 300 MHz target`.
- Track target `<= 1 ms`: not met.
- Track-specific full learned power: not measured. Current full-top routed
  vectorless power estimate is total `6.817 W`, dynamic `6.096 W`.

### 5. Track Mode Active Block And Path

Question:

- Report the active blocks and path for Track mode.

Plan:

- Trace runtime scheduler, active token count, cut depth, and shared learned
  ATTN/MLP calls.

Progress:

- Runtime scheduler and CSim mode marker are confirmed.

Results:

- Track active tokens: `16`.
- Track requested cut depth: `4`.
- With depth-counts-layers mode enabled, controller block pairs:
  `(4 + 1) / 2 = 2`.
- Active path:
  - AXIS input.
  - Patch/frame embedding path for the Track-sized active token region.
  - Global buffer with inactive Search tokens outside the Track bound ignored by
    active-token limits.
  - Runtime scheduler selects Track.
  - Same shared Q/K/V, attention, output projection, MLP, and head families.
  - AXIS state output and runtime-state report.

### 6. Search/Track Only Versus Hybrid Throughput, II, Max Rate, Worst-Case Latency

Question:

- Report throughput, initiation interval, max frame/event rate, and worst-case
  latency for Search-only, Track-only, and hybrid.

Plan:

- Use HLS top interval as the current end-to-end initiation interval.
- Convert interval to rate at 300 MHz.
- Use Search latency as the worst case.

Progress:

- HLS intervals are available for Par32, Par16, and Par8.

Results:

- Par32:
  - Track-only rate: `495.38 Hz`.
  - Search-only rate: `63.37 Hz`.
  - Hybrid expected rate: `294.57 Hz`.
  - Worst-case latency: `15.778 ms`.
- Par16:
  - Track-only rate: `358.53 Hz`.
  - Search-only rate: `45.07 Hz`.
  - Hybrid expected rate: `211.46 Hz`.
  - Worst-case latency: `22.186 ms`.
- Par8:
  - Track-only rate: `188.90 Hz`.
  - Search-only rate: `23.20 Hz`.
  - Hybrid expected rate: `110.20 Hz`.
  - Worst-case latency: `43.094 ms`.
- Top-level II is effectively one transaction per top interval, not a new
  frame/event every cycle.
- Most dense inner loops are II=1 after caching, but the report still says all
  loop constraints are not satisfied, mainly due remaining non-critical
  loop/memory-port constraints such as head pooling.

### 7. Measurement Methodology And Coverage

Question:

- Report power methodology, inclusion of PS/PL/DRAM/sensor I/O, DMA bandwidth,
  batch size, throughput, mode-specific latency, worst case, p95/p99, and
  Search/Track invocation distribution.

Plan:

- Report only what is measured or estimated by current HLS evidence.
- Defer power, DMA, and percentile claims until a fit-clean Vivado/board flow.

Progress:

- HLS latency and throughput estimates are available.
- Routed vectorless power is available for the full learned Par8 netlist.
- Board counters and mode-specific traces are not available yet.

Results:

- Power methodology:
  - Current full learned Par8 candidate: Vivado post-route vectorless estimate,
    confidence `Medium`, no SAIF/VCD imported.
  - Required next: Search-only and Track-only SAIF/VCD, then board rail
    measurement.
- Inclusion scope:
  - Routed report includes on-chip PS and PL model power.
  - It does not include sensor I/O, board regulators, or measured external DDR
    rail power.
  - DDR-related PS rails are present in the rail table, but this is still a
    model estimate, not board rail measurement.
  - HLS results exclude PS, external DDR power, sensor I/O, board regulators,
    DMA stalls, and software overhead.
- DMA bandwidth:
  - Not measured.
  - AXIS stream width is 256 bits = 32 bytes.
  - Full-rate theoretical raw stream bandwidth at 300 MHz is 9.6 GB/s per
    stream, before DMA/protocol/backpressure effects.
- Batch size:
  - One frame/event invocation per top transaction in the current testbench.
- Throughput:
  - See Question 6.
- Mode-specific latency:
  - Available as HLS Search/Track envelopes.
- Worst-case latency:
  - Search envelope.
- p95/p99:
  - Not measured. HLS report is deterministic min/max, not a distribution.
- Invocation distribution:
  - Assumed, not measured: Search 10%, Track 90%.

### 8. Resource Utilization Absolute/Percentage And Major Block Breakdown

Question:

- Report resource utilization as absolute values and ZCU104 percentages, with
  major block breakdown.

Plan:

- Use routed Vivado resource utilization for the implemented Par8 top.
- Keep HLS Par32/Par16/Par8 estimates as design-space evidence.
- Break down the dominant HLS instances.

Progress:

- Par32 and Par16 HLS resources are available.
- Par8 full learned routed resources are available.

Results:

- Par32:
  - BRAM_18K `440/624 = 70%`
  - DSP `1,634/1,728 = 94%`
  - FF `213,114/460,800 = 46%`
  - LUT `287,796/230,400 = 124%`
  - URAM `32/96 = 33%`
  - Major block: `hgtxr_e2e_controller_run`, `267,998 LUT`, `1,600 DSP`,
    `358 BRAM_18K`, `32 URAM`.
- Par16:
  - BRAM_18K `440/624 = 70%`
  - DSP `1,456/1,728 = 84%`
  - FF `197,657/460,800 = 42%`
  - LUT `249,421/230,400 = 108%`
  - URAM `32/96 = 33%`
  - Major block: `hgtxr_e2e_controller_run`, `236,322 LUT`, `1,440 DSP`,
    `358 BRAM_18K`, `32 URAM`.
- Par8 HLS estimate:
  - BRAM_18K `408/624 = 65%`
  - DSP `1,368/1,728 = 79%`
  - FF `186,580/460,800 = 40%`
  - LUT `222,223/230,400 = 96%`
  - URAM `32/96 = 33%`
  - Major block: `hgtxr_e2e_controller_run`, `214,396 LUT`, `1,360 DSP`,
    `334 BRAM_18K`, `32 URAM`.
- Par8 routed full learned implementation:
  - CLB LUT `37,300/230,400 = 16.19%`
  - CLB registers `40,008/460,800 = 8.68%`
  - Block RAM Tile `242/312 = 77.56%`
  - DSP `1,371/1,728 = 79.34%`
  - URAM `32/96 = 33.33%`
  - Major routed hierarchy power: `hgtxr_e2e_axis_top_0 = 3.079 W`,
    `axi_mem = 0.288 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.025 W`,
    `psu = 2.676 W`.

## Current Answer

The full learned non-fastpath Transformer path has been restored and validated
through HLS CSim, C synthesis, HLS IP package, Vivado route, and bitgen. The
current implemented ZCU104 candidate is Par8. It physically preserves the full
learned datapath, but it misses the Search/Track latency targets and does not
meet 300 MHz timing closure.

Current status:

- Functional Search/Track CSim: pass for Par32 and Par8.
- 300 MHz HLS timing estimate: pass for Par32, Par16, and Par8.
- HLS-level ZCU104 fit:
  - Par32 fails at `124% LUT`.
  - Par16 fails at `108% LUT`.
  - Par8 fits at `96% LUT`, with `1,368/1,728 DSP = 79%`,
    `408/624 BRAM_18K = 65%`, and `32/96 URAM = 33%`.
- Vivado implementation:
  - Par8 route/bitgen completes.
  - Par8 routed resources are `1,371 DSP`, `37,300 CLB LUT`, `242 Block RAM
    Tile`, and `32 URAM`, so the full learned datapath is physically preserved.
  - Timing fails at 300 MHz: WNS `-0.265 ns`, TNS `-113.997 ns`.
- Search latency target: fail.
  - Best latency candidate is Par32 Search at `15.778 ms`, target is `4 ms`.
  - HLS fit candidate Par8 Search is `43.094 ms`.
- Track latency target: fail.
  - Best latency candidate is Par32 Track at `2.018 ms`, target is `1 ms`.
  - HLS fit candidate Par8 Track is `5.293 ms`.
- Hybrid average latency under 10/90 Search/Track:
  - Par32 is `3.394 ms`.
  - Par16 is `4.729 ms`.
  - Par8 is `9.074 ms`.
- Current routed power:
  - Par8 full learned post-route vectorless estimate: total on-chip `6.817 W`,
    dynamic `6.096 W`, device static `0.721 W`, PS static `0.103 W`, PL static
    `0.619 W`.
  - Main dynamic buckets: `hgtxr_e2e_axis_top_0 = 3.079 W`, `psu = 2.676 W`,
    `axi_mem = 0.288 W`, `DSPs = 1.462 W`, `Signals = 0.741 W`,
    `CLB logic = 0.473 W`, `Block RAM = 0.285 W`, `URAM = 0.084 W`.
- Still unavailable for the full learned candidate:
  - Mode-specific Search/Track power.
  - p95/p99 latency.
  - Measured DMA bandwidth.
  - Board throughput and board rail power.

## Required Next Experiments

1. Close timing at 300 MHz:
   - Add missing DSP pipeline stages; routed DRC reports many DSP `MREG=0`
     warnings.
   - Relax blanket `DONT_TOUCH` where possible, especially AXI/FIFO and wrapper
     modules, while retaining datapath observability and preservation.
   - Retain enough hierarchy/observability to prevent datapath collapse.

2. Recover Search/Track latency while preserving ZCU104 fit:
   - Par8 fits at HLS level but is too slow; Par32 is faster but over LUT.
   - Replace wide control-heavy unrolled dense reductions with narrower staged
     reductions.
   - Revisit DSP binding and sharing so LUT glue does not dominate.
   - Move additional parameter storage and lookup structures from LUT-heavy
     logic into BRAM/URAM where viable.
   - Explore mixed Par settings by operator instead of a single global `PAR`.

3. Complete the dispatcher:
   - Replace marker prefetch with real layer/block weight double buffering.
   - Overlap next Search block weight load with current block compute.
   - Preserve the same parameter source for SW/HW CSim comparison.

4. Wire the event path explicitly:
   - Add a distinct Event Conv path into the E2E AXIS top or document that this
     top is frame/patch-only.
   - Add CSim cases that exercise frame-only, event-only, and combined modes.

5. Run timing-clean implementation:
   - Re-run CSim.
   - Re-run csynth.
   - Package HLS IP.
   - Run Vivado OOC and full overlay implementation.
   - Use only timing-clean routed reports for final 300 MHz claims.

6. Measure mode-specific power:
   - Generate Search and Track RTL traces.
   - Import SAIF/VCD into Vivado power.
   - Then run board rail measurement for PS/PL/DDR if final paper-quality power
     claims are needed.

7. Measure runtime distribution:
   - Add PS-side timestamping or hardware counters.
   - Collect repeated Search/Track traces.
   - Report p50/p95/p99, worst case, DMA bandwidth, and real invocation
     distribution.

## Latest Addendum: Par8 ROM-Only Full Learned Vivado Probe

This addendum supersedes the `par8_runtime_rom_dispatch_300_mem8` routed numbers
for the current full learned on-chip-parameter experiment. The earlier Par8
candidate kept a `gmem_e2e_weights` AXI master and failed timing on the weight
AXI read FIFO path. The latest candidate removes that master from the full
learned build.

Profile:

- Script profile: `par8_runtime_rom_only_dispatch_300_mem8`
- HLS scale: `runtime_rom_only_dispatch_active64_b8_ff768`
- Compile-time flags include:
  - `HGTXR_E2E_USE_MODE_PROFILE_FASTPATH=0`
  - `HGTXR_E2E_OMIT_WEIGHT_AXI=1`
  - `HGTXR_E2E_USE_ONCHIP_PARAM_ROM=1`
  - `HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM=1`
  - `HGTXR_E2E_USE_SEARCH_WEIGHT_DISPATCHER=1`
  - `HGTXR_E2E_QKV_WEIGHT_CACHE=1`
  - `HGTXR_E2E_LN_PARAM_CACHE=1`
  - `HGTXR_E2E_WEIGHT_VEC_CACHE=1`
  - `HGTXR_E2E_OBSERVE_DATAPATH=1`

Implementation changes:

- Under `HGTXR_E2E_OMIT_WEIGHT_AXI=1`, every valid model-weight element in
  `[0, kRequiredWeightElems)` is served from deterministic on-chip parameter
  ROM.
- Weight-word loads now return packed ROM words when the word maps to required
  model parameters, and return zero instead of touching AXI when the weight AXI
  interface is omitted.
- Search dispatcher prefetch no longer directly reads `weights[...]`; it uses
  the shared packed-weight loader.
- HLS/Vivado wrappers now expose a dedicated
  `par8_runtime_rom_only_dispatch_300_mem8` profile.

CSim command:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_300_mem8
```

CSim result:

- Search output words: `-712, -712, -712, -712, -712, -696`
- Search `runtime_state=0`, expected `0`, count `6`, TLAST `1`, failures `0`
- Track output words: `-235, -235, -235, -235, -235, -176`
- Track `runtime_state=1`, expected `1`, count `6`, TLAST `1`
- `E2E AXIS vector comparison passed`
- `CSim done with 0 errors`

Package command:

```sh
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_300_mem8
```

HLS package result:

- HLS package pass.
- `impl/ip/component.xml` and `impl/export.zip` exist.
- HLS log sets interface mode for `gmem_e2e_runtime` only as `m_axi`.
- `weights` remains only as `s_axilite & ap_none` control metadata.
- `component.xml` contains `m_axi_gmem_e2e_runtime`.
- `component.xml` does not contain `m_axi_gmem_e2e_weights`.

CSynth report:

- `hardware/generated/hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_300_mem8_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`

HLS timing and latency:

| Metric | Value |
|---|---:|
| Target clock | 3.333 ns |
| Estimated clock | 2.983 ns |
| Estimated Fmax | 335.24 MHz |
| Track-bound min latency | 1,600,948 cycles / 5.336 ms |
| Search-bound max latency | 12,972,958 cycles / 43.239 ms |
| Hybrid expected latency, Search 10% / Track 90% | 2,738,149 cycles / 9.126 ms |
| Worst-case latency | 12,972,958 cycles / 43.239 ms |

HLS throughput at nominal 300 MHz:

| Scenario | Interval cycles | Max rate |
|---|---:|---:|
| Track-only | 1,600,949 | 187.39 Hz |
| Search-only | 12,972,959 | 23.13 Hz |
| Hybrid, Search 10% / Track 90% | 2,738,150 | 109.56 Hz |

HLS resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 386 | 624 | 61% |
| DSP | 1,624 | 1,728 | 93% |
| FF | 168,413 | 460,800 | 36% |
| LUT | 222,677 | 230,400 | 96% |
| URAM | 32 | 96 | 33% |

Major HLS instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 342 | 1,616 | 165,209 | 215,638 | 32 |
| `hgtxr_e2e_mlp_head` | 0 | 8 | 1,603 | 3,118 | 0 |
| `gmem_e2e_runtime_m_axi` | 4 | 0 | 818 | 678 | 0 |

Vivado command:

```sh
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_300_mem8
```

Vivado implementation result:

- Block design logs skip absent `hgtxr_e2e_axis_top_0/m_axi_gmem_e2e_weights`.
- Route completes with route errors `0`.
- Bitgen completes successfully.
- Timing is not clean at 300 MHz.

Routed timing:

| Metric | Value |
|---|---:|
| `clk_pl_0` period/frequency | 3.333 ns / 300.030 MHz |
| WNS | -0.065 ns |
| TNS | -2.886 ns |
| TNS failing endpoints | 99 |
| WHS | 0.003 ns |
| THS | 0.000 ns |
| Route status | 107,269 / 107,269 routable nets fully routed, 0 route errors |

Worst setup path:

- Source:
  `.../hgtxr_e2e_axis_top_stream_stream_ap_uint_256_const_volatile_int_int_gb_27_U/ram_reg_uram_2/CLK`
- Destination:
  `.../hgtxr_e2e_attn_unit_1_s_fu_208/.../mul_ln667_43_reg_1740_reg/DSP_A_B_DATA_INST/A[25]`
- Data path delay: `2.923 ns`
- Logic / route split: `1.309 ns = 44.783%`, `1.614 ns = 55.217%`

Routed placed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUT | 24,112 | 230,400 | 10.47% |
| LUT as Logic | 22,705 | 230,400 | 9.85% |
| LUT as Distributed RAM | 1,040 | 101,760 | 1.02% |
| CLB registers | 26,348 | 460,800 | 5.72% |
| Block RAM Tile | 238 | 312 | 76.28% |
| RAMB36/FIFO | 224 | 312 | 71.79% |
| RAMB18 | 28 | 624 | 4.49% |
| DSP | 1,627 | 1,728 | 94.16% |
| URAM | 32 | 96 | 33.33% |
| PS8 | 1 | 1 | 100.00% |

Routed power report:

| Bucket | Power |
|---|---:|
| Total on-chip | 5.936 W |
| Dynamic | 5.221 W |
| Device static | 0.715 W |
| PS static | 0.101 W |
| PL static | 0.613 W |
| PS8 component | 2.671 W |
| Clocks | 0.268 W |
| CLB logic | 0.286 W |
| Signals | 0.415 W |
| Block RAM | 0.270 W |
| URAM | 0.077 W |
| DSPs | 1.235 W |

Routed hierarchy and DDR-related rails:

| Bucket | Power |
|---|---:|
| `axi_ctrl` | 0.020 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.026 W |
| `axi_mem` | 0.189 W |
| `hgtxr_e2e_axis_top_0` | 2.305 W |
| `psu` | 2.673 W |
| `VCC_PSINTFP_DDR` | 0.719 W total, 0.714 W dynamic |
| `VCCO_PSDDR_504` | 0.620 W total, 0.586 W dynamic |
| `VCC_PSDDR_PLL` | 0.001 W total |

Current eight-question answers for this ROM-only candidate:

1. HW top resource / hybrid latency / hybrid power:
   - Resources: CLB LUT `24,112/230,400 = 10.47%`, registers
     `26,348/460,800 = 5.72%`, Block RAM Tile `238/312 = 76.28%`,
     DSP `1,627/1,728 = 94.16%`, URAM `32/96 = 33.33%`.
   - Hybrid expected latency: `9.126 ms`.
   - Hybrid throughput: `109.56 Hz`.
   - Current power is post-route vectorless top power, not mode-weighted measured
     power: total `5.936 W`, dynamic `5.221 W`, static `0.715 W`.

2. Search-mode inference latency / power:
   - Search latency: `12,972,958 cycles / 43.239 ms`.
   - Search-only throughput: `23.13 Hz`.
   - Search target `<= 4 ms`: fail.
   - Search-specific power: not measured. Use full-top vectorless estimate only
     as the current envelope.

3. Search-mode active block and path:
   - Active tokens: `64`.
   - Requested depth: `8`.
   - Executed shared ATTN/MLP block pairs: `(8 + 1) / 2 = 4`.
   - Active path: AXIS input, patch/frame embedding, global buffer, Search
     schedule, shared learned Q/K/V projection, attention, output projection,
     MLP W1/GELU/W2, learned head, AXIS output.
   - Search remaining block weights are now ROM-served through the shared loader.
     The dispatcher still behaves as prefetch/marker scaffolding, not a proven
     overlapped double-buffer engine.

4. Track-mode inference latency / power:
   - Track latency: `1,600,948 cycles / 5.336 ms`.
   - Track-only throughput: `187.39 Hz`.
   - Track target `<= 1 ms`: fail.
   - Track-specific power: not measured. Use full-top vectorless estimate only
     as the current envelope.

5. Track-mode active block and path:
   - Active tokens: `16`.
   - Requested cut depth: `4`.
   - Executed shared ATTN/MLP block pairs: `(4 + 1) / 2 = 2`.
   - Active path uses the same learned Q/K/V, attention, output projection, MLP,
     nonlinear ROM, and head families with smaller token/depth bounds.

6. Search/Track-only versus hybrid throughput, II, max rate, worst case:
   - Search-only interval: `12,972,959 cycles`, max rate `23.13 Hz`.
   - Track-only interval: `1,600,949 cycles`, max rate `187.39 Hz`.
   - Hybrid expected interval: `2,738,150 cycles`, max rate `109.56 Hz`.
   - Worst-case latency: Search, `43.239 ms`.
   - Top-level II is one invocation per top interval, not one frame per cycle.

7. Measurement methodology and coverage:
   - Power methodology: Vivado post-route vectorless estimate, medium
     confidence, no SAIF/VCD, no board rails.
   - Includes modeled on-chip PS and PL.
   - DDR-related PS rails appear in the rail table, but this is still modeled
     power and should not be treated as external board DDR rail measurement.
   - Does not include sensor I/O, board regulators, external measurement error,
     or PS software overhead.
   - DMA bandwidth was not measured. The raw 256-bit AXIS stream at 300 MHz is
     9.6 GB/s per stream before DMA/protocol/backpressure effects.
   - Batch size is one invocation per transaction in the current testbench.
   - p95/p99 are not available; current HLS latency is deterministic min/max.
   - Search/Track invocation distribution is assumed `10% / 90%`, not measured.

8. Resource absolute/percentage and major block breakdown:
   - Routed full top: see routed placed utilization table above.
   - Major HLS block estimate: `hgtxr_e2e_controller_run` dominates with
     `342 BRAM_18K`, `1,616 DSP`, `165,209 FF`, `215,638 LUT`, `32 URAM`.
   - Major routed power hierarchy: `hgtxr_e2e_axis_top_0 = 2.305 W`,
     `psu = 2.673 W`, `axi_mem = 0.189 W`, `axi_dma_in = 0.007 W`,
     `axi_dma_out = 0.026 W`, `axi_ctrl = 0.020 W`.

Current status:

- Full learned ROM-only HLS/Vivado path is implemented far enough to generate a
  routed bitstream and a post-route power report.
- The full learned compute scale is physically present through DSP/BRAM/URAM
  usage.
- Weight AXI critical path is removed.
- 300 MHz timing still fails by `0.065 ns`.
- Search and Track latency goals are still not met.
- Event Conv is still not a distinct exercised event input path in this AXIS
  top.
- Mode-specific power, board-measured power, measured DMA bandwidth, and
  p95/p99 latency still require additional instrumentation and board/RTL trace
  experiments.

## Addendum: DSP3 ROM-Only 300MHz Closure

This addendum supersedes the timing status of the previous ROM-only candidate.
It does not supersede the latency and measurement-coverage gaps.

### Question

Can the full learned ROM-only shared Search/Track Transformer top be implemented
in Vivado at the requested 300 MHz while keeping learned Conv/ATTN/MLP/Head
parameters on chip and omitting the external weight AXI master?

### Plan

1. Make the `HGTXR_E2E_DSP_MUL_LATENCY` macro actually control the HLS DSP
   multiply binding latency.
2. Add an isolated `dsp3` profile so the previous ROM-only result remains
   comparable.
3. Re-run CSim, HLS package, and Vivado route/bitgen.
4. Re-answer the 8 questions from routed reports and HLS latency estimates.

### Progress

- Implemented `par8_runtime_rom_only_dispatch_dsp3_300_mem8`.
- Implemented `runtime_rom_only_dispatch_dsp3_active64_b8_ff768`.
- Preserved the ROM-only contract: `m_axi_gmem_e2e_weights` is absent from the
  packaged IP and BD construction skips that absent memory master.
- CSim passed for both runtime modes.
- Package passed.
- Vivado route and bitgen passed.

### Results

Commands:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8
```

CSim:

- Search output: `[-712, -712, -712, -712, -712, -696]`.
- Track output: `[-235, -235, -235, -235, -235, -176]`.
- Search runtime state: `0`.
- Track runtime state: `1`.
- Result: `CSim done with 0 errors`.

HLS:

| Metric | Value |
|---|---:|
| Target clock | 3.333 ns |
| Estimated clock | 2.983 ns |
| Track latency | 1,600,952 cycles / 5.336 ms |
| Search latency | 12,972,966 cycles / 43.239 ms |
| 10/90 hybrid expected latency | 2,738,153 cycles / 9.126 ms |
| Track throughput | 187.41 Hz |
| Search throughput | 23.13 Hz |
| 10/90 hybrid throughput | 109.57 Hz |

HLS resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 386 | 624 | 61% |
| DSP | 1,624 | 1,728 | 93% |
| FF | 166,668 | 460,800 | 36% |
| LUT | 222,429 | 230,400 | 96% |
| URAM | 32 | 96 | 33% |

Major HLS instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 342 | 1,616 | 163,663 | 215,390 | 32 |
| `hgtxr_e2e_mlp_head` | 0 | 8 | 1,404 | 3,118 | 0 |
| `gmem_e2e_runtime_m_axi` | 4 | 0 | 818 | 678 | 0 |

Vivado routed timing:

| Metric | Value |
|---|---:|
| `clk_pl_0` | 3.333 ns / 300.030 MHz |
| WNS | 0.009 ns |
| TNS | 0.000 ns |
| TNS failing endpoints | 0 |
| WHS | 0.010 ns |
| THS | 0.000 ns |
| Route status | 107,584 / 107,584 routable nets fully routed, 0 route errors |

Vivado routed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUT | 23,887 | 230,400 | 10.37% |
| LUT as Logic | 22,486 | 230,400 | 9.76% |
| LUT as Memory | 1,401 | 101,760 | 1.38% |
| CLB registers | 26,595 | 460,800 | 5.77% |
| Block RAM Tile | 238 | 312 | 76.28% |
| DSP | 1,627 | 1,728 | 94.16% |
| URAM | 32 | 96 | 33.33% |

Vivado routed vectorless power:

| Bucket | Power |
|---|---:|
| Total on-chip | 5.973 W |
| Dynamic | 5.258 W |
| Device static | 0.715 W |
| PS static | 0.102 W |
| PL static | 0.613 W |
| Clocks | 0.274 W |
| CLB logic | 0.288 W |
| Signals | 0.451 W |
| Block RAM | 0.270 W |
| URAM | 0.077 W |
| DSPs | 1.227 W |
| PS8 | 2.671 W |

Routed hierarchy and DDR-related rails:

| Bucket | Power |
|---|---:|
| `hgtxr_e2e_axis_top_0` | 2.341 W |
| `psu` | 2.673 W |
| `axi_mem` | 0.190 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.027 W |
| `axi_ctrl` | 0.020 W |
| `VCC_PSINTFP_DDR` | 0.719 W total, 0.714 W dynamic |
| `VCCO_PSDDR_504` | 0.620 W total, 0.586 W dynamic |
| `VCC_PSDDR_PLL` | 0.001 W total |

DRC:

| Check | Count |
|---|---:|
| DPIP-2 | 9 |
| DPOP-4 | 3 |
| REQP-1934 | 2 |
| REQP-1935 | 4 |
| RTSTAT-10 | 1 |

### Updated Answers To The 8 Questions

1. HW top resource / hybrid latency / hybrid power:
   - Routed resources are CLB LUT `23,887/230,400 = 10.37%`, registers
     `26,595/460,800 = 5.77%`, Block RAM Tile `238/312 = 76.28%`, DSP
     `1,627/1,728 = 94.16%`, URAM `32/96 = 33.33%`.
   - 10/90 hybrid expected latency is `9.126 ms`.
   - 10/90 hybrid expected throughput is `109.57 Hz`.
   - Hybrid power is not measured as a real mode-weighted trace. Current
     envelope is routed vectorless full-top power: total `5.973 W`, dynamic
     `5.258 W`, static `0.715 W`.

2. Search-mode inference latency / power:
   - Search latency is `12,972,966 cycles / 43.239 ms`.
   - Search-only throughput is `23.13 Hz`.
   - Search target `<= 4 ms`: fail.
   - Search-specific power is not measured. It needs Search-only SAIF/VCD or
     board rail measurement.

3. Search-mode active block and path:
   - Active tokens: `64`.
   - Requested depth: `8`.
   - Shared ATTN/MLP block pairs executed: `(8 + 1) / 2 = 4`.
   - Active path: AXIS input, frame/patch embedding, global buffer, runtime
     Search schedule, shared learned Q/K/V, attention, output projection, MLP
     W1/GELU/W2, learned head, AXIS output.
   - Weight access is ROM-only for model parameters; the Search dispatcher
     calls the shared ROM word loader and no external weight AXI master exists.

4. Track-mode inference latency / power:
   - Track latency is `1,600,952 cycles / 5.336 ms`.
   - Track-only throughput is `187.41 Hz`.
   - Track target `<= 1 ms`: fail.
   - Track-specific power is not measured. It needs Track-only SAIF/VCD or
     board rail measurement.

5. Track-mode active block and path:
   - Active tokens: `16`.
   - Requested cut depth: `4`.
   - Shared ATTN/MLP block pairs executed: `(4 + 1) / 2 = 2`.
   - Active path uses the same shared learned Q/K/V, attention, output
     projection, MLP, nonlinear ROM, and head families with smaller token/depth
     bounds.

6. Search/Track-only versus Hybrid throughput, II, max rate, worst case:
   - Search-only interval is `12,972,967 cycles`, max rate `23.13 Hz`.
   - Track-only interval is `1,600,953 cycles`, max rate `187.41 Hz`.
   - 10/90 hybrid expected interval is about `2,738,154 cycles`, max rate
     `109.57 Hz`.
   - Worst-case latency is Search, `43.239 ms`.
   - The top-level initiation interval is one invocation per reported interval,
     not one frame/event per cycle.

7. Measurement methodology and coverage:
   - Current power is Vivado post-route vectorless power, confidence `Medium`.
   - It includes modeled on-chip PS and PL. DDR-related PS rails are listed by
     Vivado, but this is not an external board DDR rail measurement.
   - It does not include sensor I/O, board regulators, external measurement
     uncertainty, PS software overhead, or measured mode distribution.
   - DMA bandwidth is not measured. Raw 256-bit AXIS at 300 MHz is `9.6 GB/s`
     per stream before DMA/protocol/backpressure effects.
   - Batch size is one invocation per transaction in the current testbench.
   - p95/p99 are not available; the current evidence is deterministic HLS
     min/max latency plus one CSim functional pass.
   - Search/Track distribution is assumed `10% / 90%`, not measured.

8. Source utilization absolute/percentage and major block breakdown:
   - Routed absolute and percentage resource table is listed above.
   - HLS major block estimate shows `hgtxr_e2e_controller_run` dominates:
     `342 BRAM_18K`, `1,616 DSP`, `163,663 FF`, `215,390 LUT`, `32 URAM`.
   - Routed power hierarchy is dominated by `psu = 2.673 W` and
     `hgtxr_e2e_axis_top_0 = 2.341 W`; AXI/DMA/SmartConnect are much smaller.

### Updated Status

- Full learned ROM-only Transformer now implements and routes cleanly at
  300 MHz on ZCU104.
- The design fits the device, but DSP headroom is tight at `94.16%`.
- Search/Track latency goals remain unmet by large margins.
- Mode-specific power, DMA bandwidth, board power, repeated latency
  distribution, p95/p99, and sensor I/O inclusion still require additional
  experiments.

## 2026-06-28 Dispatcher Ping-Pong Prefix Prefetch Vivado Rerun

This rerun adds a concrete dispatcher prefetch bank to the same full learned
ROM-only DSP3 profile.

Implementation delta:

- `HGTXR_E2E_DISPATCH_PREFETCH_WORDS` defaults to `16`.
- `HgtxrGlobalBuffer` now carries two dispatcher prefetch banks:
  `dispatch_prefetch[2][16]` plus base, valid-word, block, and ready metadata.
- Search-extra block dispatch fills the next block's ping-pong bank before the
  current block executes.
- LayerNorm, Q/K/V, output projection, and MLP W1/W2 cache fills now call
  `hgtxr_e2e_load_weight_word_prefetched(...)`; the prefetch is connected to
  actual weight-cache loads, not only to an observable marker.
- The dispatcher remains a prefix prefetch of the first `16` packed words, not
  a full-block double-buffer engine. Therefore this proves a real cache-load
  connection, but it does not yet prove that every next-layer weight is ready
  immediately when the previous layer finishes.

Validation commands completed:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8
```

Functional result:

- CSim passed Search and Track runtime checks.
- Search output: `[-712, -712, -712, -712, -712, -676]`.
- Track output: `[-235, -235, -235, -235, -235, -291]`.
- Search `runtime_state=0`, Track `runtime_state=1`, output count `6`,
  TLAST correct, and `CSim done with 0 errors`.

HLS package result:

| Metric | Value |
|---|---:|
| Target clock | 3.333 ns |
| Estimated clock | 2.924 ns |
| Top min latency | 26,190 cycles / 87.291 us |
| Top max latency | 13,008,010 cycles / 43.356 ms |
| Top min interval | 26,191 cycles |
| Top max interval | 13,008,011 cycles |

Important latency caveat:

- The current HLS top min is not used as Track latency. It is too small for the
  full Track path and appears to be a variable-bound/control lower envelope.
- Current Search bound is taken from the HLS max latency:
  `13,008,010 cycles / 43.356 ms`.
- Current exact Track-mode latency requires a mode-specific trace, RTL trace,
  or board measurement. Until that is collected, the best comparable Track
  envelope remains the prior full-depth Track estimate:
  `1,605,053 cycles / 5.350 ms`.
- A planning-only hybrid estimate using current Search max plus that prior Track
  envelope is `2,745,348.7 cycles / 9.150 ms`. This is not a direct current
  mode-specific measurement.

HLS OOC resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 386 | 624 | 61% |
| DSP | 1,753 | 1,728 | 101% |
| FF | 220,360 | 460,800 | 47% |
| LUT | 298,616 | 230,400 | 129% |
| URAM | 32 | 96 | 33% |

HLS major instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 342 | 1,745 | 217,256 | 290,406 | 32 |
| `hgtxr_e2e_mlp_head` | 0 | 8 | n/a | n/a | 0 |

The HLS estimate overstates final LUT/DSP fit for this candidate. Final device
fit must be read from the routed Vivado utilization below.

Vivado routed timing:

| Metric | Value |
|---|---:|
| `clk_pl_0` | 3.333 ns / 300.030 MHz |
| WNS | 0.000 ns |
| TNS | 0.000 ns |
| Failing setup endpoints | 0 / 231,222 |
| WHS | 0.007 ns |
| THS | 0.000 ns |
| Failing hold endpoints | 0 / 231,222 |
| WPWS | 0.166 ns |
| Route status | 138,804 / 138,804 routable nets fully routed |
| Route errors | 0 |
| Bitstream | generated successfully |

Vivado routed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUT | 43,929 | 230,400 | 19.07% |
| LUT as Logic | 42,292 | 230,400 | 18.36% |
| LUT as Memory | 1,637 | 101,760 | 1.61% |
| CLB registers | 43,016 | 460,800 | 9.34% |
| CARRY8 | 3,591 | 28,800 | 12.47% |
| Block RAM Tile | 238 | 312 | 76.28% |
| RAMB36/FIFO | 224 | 312 | 71.79% |
| RAMB18 | 28 | 624 | 4.49% |
| DSP | 1,630 | 1,728 | 94.33% |
| URAM | 32 | 96 | 33.33% |
| PS8 | 1 | 1 | 100% |

Vivado routed vectorless power:

| Bucket | Power |
|---|---:|
| Total on-chip | 6.833 W |
| Dynamic | 6.111 W |
| Device static | 0.721 W |
| Clocks | 0.406 W |
| CLB logic | 0.705 W |
| Signals | 0.879 W |
| Block RAM | 0.273 W |
| DSPs | 1.094 W |
| PS8 | 2.671 W |

Routed hierarchy dynamic power:

| Hierarchy | Dynamic power |
|---|---:|
| `hgtxr_e2e_axis_top_0` | 3.197 W |
| `psu` | 2.674 W |
| `axi_mem` | 0.186 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.026 W |

DDR-related PS rails are reported as current in the Vivado power report, not as
direct watts:

| Rail | Total current | Dynamic current | Static current | Approx. rail power |
|---|---:|---:|---:|---:|
| `VCC_PSINTFP_DDR` | 0.719 A | 0.714 A | 0.005 A | ~0.611 W at 0.85 V |
| `VCCO_PSDDR_504` | 0.620 A | 0.586 A | 0.034 A | ~0.744 W at 1.20 V |
| `VCC_PSDDR_PLL` | 0.001 A | 0.000 A | 0.001 A | n/a |

### Current Answers After Dispatcher Prefetch Rerun

1. HW top total resources, hybrid latency, hybrid power:
   - Final routed resources are CLB LUT `43,929/230,400 = 19.07%`,
     registers `43,016/460,800 = 9.34%`, Block RAM Tile
     `238/312 = 76.28%`, DSP `1,630/1,728 = 94.33%`, and URAM
     `32/96 = 33.33%`.
   - Direct current hybrid latency is not fully measured because current
     Track-mode cycle count is not isolated.
   - Planning estimate with current Search max and prior Track-bound envelope:
     `~2.745M cycles / ~9.150 ms`, `~109.3 Hz`.
   - Conservative worst-case hybrid latency is Search max: `43.356 ms`.
   - Power envelope is routed vectorless full-top power: total `6.833 W`,
     dynamic `6.111 W`, static `0.721 W`.

2. Search-mode inference latency and power:
   - Search bound: `13,008,010 cycles / 43.356 ms`.
   - Search-only rate: `~23.06 Hz`.
   - Search target `<= 4 ms`: fail.
   - Search-specific power is not measured. Current evidence is the routed
     vectorless full-top envelope and hierarchy power.

3. Search-mode active block and path:
   - Active tokens: `64`.
   - Requested depth: `8`, interpreted as `4` shared ATTN/MLP block-pairs for
     the layer-counting profile.
   - Active path: AXIS input, Frame Conv ROM path, global buffer, shared
     learned LayerNorm/QKV/Attention/output-projection/MLP units, nonlinear ROM
     operators, Head ROM path, AXIS output.
   - Search-extra blocks use dispatcher ping-pong prefix prefetch into
     `dispatch_prefetch[2][16]`; cache fills check the prefetch bank before
     falling back to normal ROM/weight word loading.

4. Track-mode inference latency and power:
   - Track CSim functional path passes.
   - Current HLS top min is not a valid Track-mode latency. The prior
     comparable full-depth Track envelope remains `1,605,053 cycles / 5.350 ms`
     until a current mode-specific trace is captured.
   - Track target `<= 1 ms`: still fail by the best available full-depth
     envelope.
   - Track-specific power is not measured.

5. Track-mode active block and path:
   - Active tokens: `16`.
   - Requested cut depth: `4`, interpreted as `2` shared ATTN/MLP block-pairs.
   - Active path: derived Event Conv ROM path, global buffer, the same shared
     learned LayerNorm/QKV/Attention/output-projection/MLP units, nonlinear ROM
     operators, Head ROM path, AXIS output.
   - Track weights are on-chip ROM in this ROM-only profile.

6. Search/Track-only versus Hybrid throughput, II, max rate, worst case:
   - Search-only max rate: `~23.06 Hz`.
   - Track-only rate from prior full-depth envelope: `~186.9 Hz`; current exact
     Track rate is pending mode-specific measurement.
   - Planning hybrid rate: `~109.3 Hz`.
   - Worst-case latency: Search max `43.356 ms`.
   - Top-level transaction interval is essentially one invocation per reported
     latency/interval. There is no proven overlapped multi-invocation pipeline.

7. Power methodology and coverage:
   - Power is Vivado post-route vectorless power with `Medium` confidence.
   - It includes modeled on-chip PS, PL, DSP, BRAM/URAM, AXI interconnect, and
     DMA blocks in the Vivado design.
   - Vivado reports DDR-related PS rail currents, but this is not an external
     DDR board-rail measurement.
   - Sensor I/O, board regulators, external DDR module power, PS software
     overhead, and measurement uncertainty are excluded.
   - DMA bandwidth is not measured. Raw 256-bit at 300 MHz is `9.6 GB/s` per
     stream before protocol overhead and backpressure.
   - Batch size is `1` invocation per current testbench transaction.
   - Mode-specific latency distributions, p95/p99, real Search/Track invocation
     distribution, and board throughput require repeated board or RTL traces.
   - The only distribution used here is the requested assumption: Search `10%`,
     Track `90%`.

8. Source utilization absolute/percentage and major block breakdown:
   - Final routed absolute and percentage utilization is listed above.
   - HLS instance estimates identify `hgtxr_e2e_controller_run` as the dominant
     compute block, but HLS overestimates final routed LUT/DSP fit for this
     candidate.
   - Paper/report tables should use routed utilization for final device fit and
     HLS OOC instance data only for major-block attribution.
   - Routed dynamic hierarchy is dominated by `hgtxr_e2e_axis_top_0 = 3.197 W`
     and `psu = 2.674 W`; AXI memory/interconnect and DMA are much smaller.

### Current Conclusion

- The full learned Transformer path is implemented and routed through Vivado at
  the requested `300 MHz` clock on ZCU104.
- Frame Conv, derived Event Conv, Head, Track Transformer weights, Search
  Transformer weights, and nonlinear LUTs are all represented by deterministic
  on-chip learned-parameter/LUT ROM paths in this profile.
- The runtime scheduler and shared Search/Track ATTN/MLP datapath are active.
- The dispatcher is now connected to real weight-cache fills through a
  ping-pong prefix prefetch, but the requested full next-layer immediate
  prefetch behavior still needs a full-block double-buffer design or trace proof.
- Latency goals are not met: Search is `43.356 ms > 4 ms`, and the best
  comparable Track envelope is `5.350 ms > 1 ms`.
- Mode-specific power, DMA bandwidth, p95/p99 latency, and real board power
  remain additional experiments.

## 2026-06-28 Dispatcher Full-Block Prefetch Vivado Rerun

This rerun replaces the previous `16`-word dispatcher prefix bank with a
full-block dispatcher prefetch bank for the same
`par8_runtime_rom_only_dispatch_dsp3_300_mem8` full learned ROM-only DSP3
profile.

Question:

- Can the Search-extra Transformer layer weights be prefetched as a complete
  next block, rather than only a prefix, while keeping Frame Conv, derived Event
  Conv, Head, Track path, Transformer parameters, and nonlinear LUTs on chip?
- Does the resulting design still fit ZCU104 and meet the requested 300 MHz
  Vivado timing target?

Plan:

1. Size the dispatcher bank from `kBlockWeightWords` instead of the old fixed
   `HGTXR_E2E_DISPATCH_PREFETCH_WORDS=16` prefix.
2. Bind the full-block dispatcher prefetch bank to URAM so it is a real
   implementable storage structure, not LUTRAM fanout.
3. Enable the full-block dispatcher only in the
   `runtime_rom_only_dispatch_dsp3_active64_b8_ff768` profile.
4. Re-run CSim, HLS package, and Vivado route/bitgen.
5. Update the eight-question answer with the exact evidence boundary.

Progress:

- Implemented `HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK` and
  `HGTXR_E2E_URAM_DISPATCH_PREFETCH`.
- `kDispatchPrefetchWords` is now `kBlockWeightWords` for the full-block
  dispatcher build.
- The generated HLS report shows `gb_dispatch_prefetch` implemented as
  `hgtxr_e2e_controller_run_gb_dispatch_prefetch_RAM_2P_URAM_1R1W`.
- CSim passed both Search and Track runtime checks.
- HLS package completed.
- Vivado route and bitgen completed with `impl_strategy=Performance_Explore`,
  and final post-route timing now meets the 300 MHz constraint.

Baseline policy:

- `par32_dsp_mixed_stream_mem16` should remain the physical/timing baseline:
  it is the earlier best routed PAR32 stream candidate with strong WNS margin
  and bitgen pass, but it is not the full learned runtime/on-chip ROM evidence.
- `par32_runtime_full_axi_mem16` should remain the full learned functional
  baseline: it restores learned Conv/ATTN/MLP/Head through AXI weights and
  provides a Search/Track latency reference, but it is not on-chip ROM and was
  not the final routed on-chip dispatcher candidate.
- `par8_runtime_rom_only_dispatch_dsp3_300_mem8` is the current objective
  baseline: it is the full learned ROM-only runtime scheduler design with
  on-chip params/LUTs and the full-block dispatcher prefetch.

Implementation evidence:

- Source enablement:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp`
  - `hardware/hls/src/hgtxr_e2e_axis_top.cpp`
  - `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl`
  - `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`
- HLS dispatcher storage:
  - `gb_dispatch_prefetch_U`
  - `13,848` words
  - `255` bits per word
  - `16` URAM
  - `STORAGESIZE="255 13848 1"`
  - `IMPL="uram"`

Validation commands completed:

```sh
timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_dsp3_300_mem8
timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8
```

Functional result:

- Search output: `[-712, -712, -712, -712, -712, -676]`.
- Track output: `[-235, -235, -235, -235, -235, -291]`.
- Search `runtime_state=0`, Track `runtime_state=1`, output count `6`,
  TLAST correct.
- CSim ended with `CSim done with 0 errors`.

HLS package result:

| Metric | Value |
|---|---:|
| Target clock | 3.333 ns |
| Estimated clock | 2.924 ns |
| Top min latency | 26,190 cycles / 87.291 us |
| Top max latency | 13,042,545 cycles / 43.471 ms |
| Top min interval | 26,191 cycles |
| Top max interval | 13,042,546 cycles |
| `hgtxr_e2e_controller_run` max | 12,964,132 cycles / 43.209 ms |

HLS OOC resource estimate:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| BRAM_18K | 386 | 624 | 61% |
| DSP | 1,753 | 1,728 | 101% |
| FF | 227,004 | 460,800 | 49% |
| LUT | 308,055 | 230,400 | 133% |
| URAM | 48 | 96 | 50% |

HLS major instance estimate:

| Instance | BRAM_18K | DSP | FF | LUT | URAM |
|---|---:|---:|---:|---:|---:|
| `hgtxr_e2e_controller_run` | 342 | 1,745 | 223,900 | 299,845 | 48 |

The HLS OOC estimate is not the final device-fit answer. The final Vivado
placed utilization below is the fit evidence.

Vivado routed timing:

| Metric | Value |
|---|---:|
| `clk_pl_0` | 3.333 ns / 300.030 MHz |
| Implementation strategy | Performance_Explore |
| WNS | 0.010 ns |
| TNS | 0.000 ns |
| Failing setup endpoints | 0 / 223,037 |
| WHS | 0.009 ns |
| THS | 0.000 ns |
| Failing hold endpoints | 0 / 223,037 |
| WPWS | 0.166 ns |
| Timing constraints | met |
| Route status | 141,580 / 141,580 routable nets fully routed |
| Route errors | 0 |
| Bitstream | generated successfully |

Vivado placed utilization:

| Resource | Used | ZCU104 total | Percent |
|---|---:|---:|---:|
| CLB LUT | 44,894 | 230,400 | 19.49% |
| LUT as Logic | 43,455 | 230,400 | 18.86% |
| LUT as Memory | 1,439 | 101,760 | 1.41% |
| CLB registers | 44,799 | 460,800 | 9.72% |
| Block RAM Tile | 238 | 312 | 76.28% |
| RAMB36/FIFO | 224 | 312 | 71.79% |
| RAMB18 | 28 | 624 | 4.49% |
| DSP | 1,630 | 1,728 | 94.33% |
| URAM | 48 | 96 | 50.00% |

Vivado routed vectorless power:

| Bucket | Power |
|---|---:|
| Total on-chip | 7.109 W |
| Dynamic | 6.383 W |
| Device static | 0.725 W |
| Clocks | 0.432 W |
| CLB logic | 0.715 W |
| Signals | 1.036 W |
| Block RAM | 0.289 W |
| URAM | 0.115 W |
| DSPs | 1.127 W |
| PS8 | 2.671 W |
| PS static | 0.103 W |
| PL static | 0.622 W |

Routed hierarchy dynamic power:

| Hierarchy | Dynamic power |
|---|---:|
| `hgtxr_e2e_axis_top_0` | 3.467 W |
| `psu` | 2.673 W |
| `axi_mem` | 0.188 W |
| `axi_dma_in` | 0.007 W |
| `axi_dma_out` | 0.027 W |

DDR-related PS rails are reported as current in the Vivado power report, not as
direct external board DDR watts:

| Rail | Total current | Dynamic current | Static current | Approx. rail power |
|---|---:|---:|---:|---:|
| `VCC_PSINTFP_DDR` | 0.719 A | 0.714 A | 0.005 A | ~0.611 W at 0.85 V |
| `VCCO_PSDDR_504` | 0.620 A | 0.586 A | 0.034 A | ~0.744 W at 1.20 V |
| `VCC_PSDDR_PLL` | 0.001 A | 0.000 A | 0.001 A | n/a |

### Current Answers After Full-Block Dispatcher Rerun

1. HW top total resources, hybrid latency, hybrid power:
   - Final placed resources are CLB LUT `44,894/230,400 = 19.49%`,
     registers `44,799/460,800 = 9.72%`, Block RAM Tile
     `238/312 = 76.28%`, DSP `1,630/1,728 = 94.33%`, and URAM
     `48/96 = 50.00%`.
   - Direct current hybrid latency is not fully measured because current
     Track-mode cycle count is not isolated.
   - Planning estimate with current Search max and prior Track-bound envelope:
     `~2.749M cycles / ~9.162 ms`, `~109.15 Hz`.
   - Conservative worst-case hybrid latency is Search max: `43.471 ms`.
   - Power envelope is routed vectorless full-top power: total `7.109 W`,
     dynamic `6.383 W`, static `0.725 W`.
   - This build meets 300 MHz routed timing with WNS `0.010 ns`, but the
     margin is small.

2. Search-mode inference latency and power:
   - Search bound: `13,042,545 cycles / 43.471 ms`.
   - Search-only rate at 300.030 MHz: `~23.00 Hz`.
   - Search target `<= 4 ms`: fail.
   - Search-specific power is not measured. Current evidence is the routed
     vectorless full-top envelope and hierarchy power.

3. Search-mode active block and path:
   - Active tokens: `64`.
   - Requested depth: `8`, interpreted as `4` shared ATTN/MLP block-pairs for
     the layer-counting profile.
   - Active path: AXIS input, Frame Conv ROM path, global buffer, shared
     learned LayerNorm/QKV/Attention/output-projection/MLP units, nonlinear ROM
     operators, Head ROM path, AXIS output.
   - Search-extra blocks use dispatcher ping-pong full-block prefetch into
     `dispatch_prefetch[2][kBlockWeightWords]`, implemented as a
     `13,848 x 255-bit` URAM-backed bank pair.

4. Track-mode inference latency and power:
   - Track CSim functional path passes.
   - Current HLS top min is not accepted as Track-mode latency. The prior
     comparable full-depth Track envelope remains `1,605,053 cycles / 5.350 ms`
     until a current mode-specific trace is captured.
   - Track target `<= 1 ms`: fail by the best available full-depth envelope.
   - Track-specific power is not measured.

5. Track-mode active block and path:
   - Active tokens: `16`.
   - Requested cut depth: `4`, interpreted as `2` shared ATTN/MLP block-pairs.
   - Active path: derived Event Conv ROM path, global buffer, the same shared
     learned LayerNorm/QKV/Attention/output-projection/MLP units, nonlinear ROM
     operators, Head ROM path, AXIS output.
   - Track weights are on-chip ROM in this ROM-only profile.

6. Search/Track-only versus Hybrid throughput, II, max rate, worst case:
   - Search-only max rate: `~23.00 Hz`.
   - Track-only rate from prior full-depth envelope: `~186.93 Hz`; current
     exact Track rate is pending mode-specific measurement.
   - Planning hybrid rate: `~109.15 Hz`.
   - Worst-case latency: Search max `43.471 ms`.
   - Top-level transaction interval is essentially one invocation per reported
     latency/interval. There is no proven overlapped multi-invocation pipeline.

7. Power methodology and coverage:
   - Power is Vivado post-route vectorless power with `Medium` confidence.
   - It includes modeled on-chip PS, PL, DSP, BRAM/URAM, AXI interconnect, and
     DMA blocks in the Vivado design.
   - Vivado reports DDR-related PS rail currents, but this is not an external
     DDR board-rail measurement.
   - Sensor I/O, board regulators, external DDR module power, PS software
     overhead, and measurement uncertainty are excluded.
   - DMA bandwidth is not measured. Raw 256-bit at 300 MHz is `9.6 GB/s` per
     stream before protocol overhead and backpressure.
   - Batch size is `1` invocation per current testbench transaction.
   - Mode-specific latency distributions, p95/p99, real Search/Track invocation
     distribution, and board throughput require repeated board or RTL traces.
   - The only distribution used here is the requested assumption: Search `10%`,
     Track `90%`.

8. Source utilization absolute/percentage and major block breakdown:
   - Final placed absolute and percentage utilization is listed above.
   - HLS instance estimates identify `hgtxr_e2e_controller_run` as the dominant
     compute block, but HLS overestimates final routed LUT/DSP fit for this
     candidate.
   - Paper/report tables should use placed Vivado utilization for final device
     fit and HLS OOC instance data only for major-block attribution.
   - Routed dynamic hierarchy is dominated by `hgtxr_e2e_axis_top_0 = 3.467 W`
     and `psu = 2.673 W`; AXI memory/interconnect and DMA are much smaller.

### Current Conclusion After Full-Block Dispatcher Rerun

- The full learned Transformer path is implemented through CSim, HLS package,
  Vivado route, and bitgen with a real full-block dispatcher prefetch bank.
- ZCU104 resource fit is still feasible in placed utilization terms.
- The design is timing-clean at 300 MHz after applying
  `Performance_Explore`: WNS `0.010 ns`, TNS `0.000 ns`, WHS `0.009 ns`.
- Frame Conv, derived Event Conv, Head, Track Transformer weights, Search
  Transformer weights, and nonlinear LUTs are represented by deterministic
  on-chip learned-parameter/LUT ROM paths in this profile.
- Search latency still fails badly: `43.471 ms > 4 ms`.
- Track latency remains pending for this exact rerun; the best comparable
  full-depth envelope remains `5.350 ms > 1 ms`.
- Mode-specific power, DMA bandwidth, p95/p99 latency, and board rail power
  remain additional experiments.

## Addendum: PAR32 Baseline Recheck

The earlier PAR32 profiles are still useful, but they serve different baseline
roles:

- `par32_dsp_mixed_stream_mem16`: physical/timing baseline. It is the earlier
  routed PAR32 stream candidate with strong timing margin and bitgen pass.
- `par32_runtime_full_axi_mem16`: full learned AXI functional baseline. It
  restores learned Conv/ATTN/MLP/Head through the external AXI weight path.
- Current objective baseline: ROM-only learned weights/LUTs, runtime scheduler,
  and dispatcher prefetch in one design. The prior PAR32 baselines do not
  satisfy this complete objective envelope.

PAR32 objective-path probe:

| Profile | Result |
|---|---|
| `par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only` | csynth pass, `4,764,228` cycles / `15.879 ms`, estimated clock `2.777 ns`, DSP `1,763/1,728`, LUT `302,026/230,400` |
| `par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only` | csynth pass, `604,652` cycles / `2.100 ms`, estimated clock `3.473 ns`, DSP `1,374/1,728`, LUT `263,377/230,400` |
| `par32_runtime_rom_only_dispatch_dsp3_300_mem16` | package pass, Vivado implementation fails before placement with DRC `UTLZ-1`: `1729` DSP cells required, `1728` available |

Conclusion: PAR32 improves latency versus the current PAR8 objective build, but
the ROM-only objective variant is not yet a ZCU104-fit implementation point and
still misses Search `4 ms` and Track `1 ms`.

## Addendum: PAR32 Core-Fabric DSP-Fit Objective Probe

The first PAR32 ROM-only objective implementation failed at `1729/1728` DSP.
The follow-up `corefabric` profile moves one core dense lane to fabric so the
same objective envelope can reach placement, routing, and bitstream generation.

| Profile | CSim | CSynth | Vivado |
|---|---|---|---|
| `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16` | Pass; Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]` | Pass; target `3.333 ns`, estimate `2.777 ns`, max `4,805,312` cycles / `16.016 ms`, resources BRAM_18K `418`, DSP `1,976`, LUT `399,002`, URAM `48` | Route/bitgen pass; route errors `0`; DSP48E2 `1,728/1,728`; timing fail WNS `-0.631 ns`, TNS `-3338.375 ns` |

Routed implementation evidence:

- Placed utilization: CLB LUT `89,102/230,400 = 38.67%`, registers
  `60,400/460,800 = 13.11%`, Block RAM Tile `232/312 = 74.36%`, DSP
  `1,728/1,728 = 100.00%`, URAM `48/96 = 50.00%`.
- Routed vectorless power: total on-chip `8.593 W`, dynamic `7.856 W`,
  static `0.737 W`, Medium confidence.
- Dynamic hierarchy: `hgtxr_e2e_axis_top_0 = 4.933 W`, `psu = 2.674 W`,
  `axi_mem = 0.194 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.025 W`,
  `axi_ctrl = 0.023 W`.

Updated baseline judgment:

- `par32_dsp_mixed_stream_mem16` remains the physical/timing comparison
  baseline.
- `par32_runtime_full_axi_mem16` remains the full learned AXI functional
  comparison baseline.
- `par8_runtime_rom_only_dispatch_dsp3_300_mem8` remains the current
  timing-clean objective baseline for ROM-only learned weights/LUTs plus
  runtime scheduler plus dispatcher prefetch.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16`
  is now the latency-improved PAR32 objective probe. It is not the signoff
  baseline because 300 MHz setup timing fails and mode-specific latency targets
  are still not met.

Updated eight-question status:

1. Hybrid resource/power can now be reported for the PAR32 objective probe
   using placed utilization and routed vectorless power above. Hybrid latency
   should not be claimed as measured beyond the HLS max envelope `16.016 ms`
   until exact corefabric Search/Track force-mode runs are captured.
2. Search-mode latency for the comparable PAR32 objective path remains
   `15.879 ms` from the earlier Search-only force-mode profile; exact
   corefabric Search-only is pending. Search-specific power remains pending.
3. Search active path is the ROM-only Frame Conv, shared learned
   ATTN/MLP/nonlinear path, full-block dispatcher prefetch, and Head path.
4. Track-mode latency for the comparable PAR32 objective path remains
   `2.100 ms` from the earlier Track-only force-mode profile; exact corefabric
   Track-only is pending. Track-specific power remains pending.
5. Track active path is the ROM-only Event Conv, same shared learned
   ATTN/MLP/nonlinear path, and Head path.
6. Search/Track-only throughput, II, max rate, and worst-case latency require
   exact corefabric force-mode csynth before final reporting. Current
   conservative worst case is the combined HLS max `16.016 ms`.
7. Power methodology is still Vivado post-route vectorless, Medium confidence.
   It includes modeled PS/PL/on-chip DMA/AXI activity in the design but not
   external board rail, sensor I/O, or measured DDR module power.
8. Source utilization should use Vivado placed absolute values and percentages
   for device fit; HLS instance resources remain attribution evidence only.

## Addendum: PAR32 Baseline Taxonomy and Norm-URAM Timing Probe

The earlier best PAR32 profiles should not be dropped. They should be used as
reference baselines with explicit roles:

| Profile | Baseline role | Why it is useful | Why it is not the objective-signoff baseline |
|---|---|---|---|
| `par32_dsp_mixed_stream_mem16` | Physical/timing baseline | Earlier best routed PAR32 stream candidate: WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen pass; HLS resources BRAM_18K `332`, DSP `1,148`, LUT `193,546`, URAM `64` | It is not the full ROM-only learned-weight runtime scheduler/dispatcher objective envelope |
| `par32_runtime_full_axi_mem16` | Full learned AXI functional baseline | Restores learned Conv/ATTN/MLP/Head through external AXI weights; HLS Search `25.510 ms`, Track `3.136 ms` | It uses the external AXI weight path, while the current objective requires on-chip ROM/LUT plus scheduler/prefetch evidence |
| `par8_runtime_rom_only_dispatch_dsp3_300_mem8` | Timing-clean objective baseline | Full learned ROM-only scheduler/dispatcher profile that routes cleanly at 300 MHz | It is timing-clean but too slow: Search `43.334 ms`, Track `5.355 ms` |
| `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16` | PAR32 objective timing probe | Same ROM-only objective envelope with better latency envelope and improved WNS versus corefabric | It still fails 300 MHz setup timing and therefore is not signoff |

Norm-URAM probe result:

| Stage | Result |
|---|---|
| CSim | Pass; Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1` |
| CSynth | Pass; target `3.333 ns`, estimate `2.777 ns`, min `26,088` cycles / `86.951 us`, max `4,805,312` cycles / `16.016 ms`, interval max `4,805,313` |
| CSynth resources | BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64` |
| Vivado | Route/bitgen pass, route errors `0`, DSP48E2 `1,728/1,728`, but setup timing fail |
| Routed timing | WNS `-0.496 ns`, TNS `-2860.064 ns`, WHS `0.000 ns`, THS `0.000 ns`, failing setup endpoints `15,033/303,394` |
| Routed utilization | CLB LUT `89,329/230,400 = 38.77%`, registers `60,699/460,800 = 13.17%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%` |
| Routed power | Total `8.568 W`, dynamic `7.829 W`, static `0.738 W` |

Interpretation:

- `normuram` proves that moving norm/global-buffer pressure from BRAM to URAM is
  directionally useful: WNS improves from the prior core-fabric `-0.631 ns` to
  `-0.496 ns`, and Block RAM Tile drops from `232/312` to `216/312`.
- The design remains just outside 300 MHz timing closure and still misses the
  Search/Track latency targets, so it should be reported as a PAR32 objective
  probe, not as the signoff baseline.
- All future eight-question tables should include at least four rows:
  physical/timing baseline, learned-AXI functional baseline, timing-clean
  ROM-only objective baseline, and latest PAR32 objective probe.

## Addendum: PAR32 Norm-Stage LayerNorm Write Probe

This probe tested whether staging LayerNorm output through a local LUTRAM row
buffer before writing `gb.norm` would improve the PAR32 objective timing path.
It did not improve timing versus the prior `normuram` probe.

| Stage | Result |
|---|---|
| Profile | `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16` |
| CSim | Pass; Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1` |
| CSynth | Pass; target `3.333 ns`, estimate `2.777 ns`, min `26,088` cycles / `86.951 us`, max `4,908,208` cycles / `16.359 ms`, interval max `4,908,209` |
| Controller latency | `4,829,897` cycles / `16.098 ms` |
| CSynth resources | BRAM_18K `386`, DSP `1,976`, FF `249,856`, LUT `399,142`, URAM `64` |
| Vivado | Route/bitgen pass, route errors `0`, fully routed nets `201,593/201,593`, DSP48E2 `1,728/1,728`, but setup timing fail |
| Routed timing | WNS `-0.644 ns`, TNS `-3987.069 ns`, WHS `0.002 ns`, THS `0.000 ns`, failing setup endpoints `16,238/303,146` |
| Routed utilization | CLB LUT `89,035/230,400 = 38.64%`, registers `59,380/460,800 = 12.89%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%` |
| Routed power | Total `8.615 W`, dynamic `7.876 W`, static `0.739 W`, PS static `0.105 W`, PL static `0.634 W` |

Interpretation:

- The earlier best PAR32 profiles are still baselines, not discarded:
  `par32_dsp_mixed_stream_mem16` is the physical/timing baseline and
  `par32_runtime_full_axi_mem16` is the learned-AXI functional baseline.
- `normstage` remains a PAR32 objective probe because it contains the objective
  ROM-only learned-weight path, runtime scheduler, and dispatcher/prefetch path.
- It is not the signoff baseline: route/bitgen passes, but 300 MHz timing fails
  with worse WNS/TNS than the `normuram` probe.
- Worst reported paths point to output-projection/MLP DSP input paths and
  `gb_dispatch_prefetch` URAM read fanout. The next implementation step should
  add explicit DSP input/output pipeline stages and reduce broad datapath
  `DONT_TOUCH` where it blocks useful physopt.

Updated eight-question status:

1. Hybrid resource and vectorless power can be reported for `normstage`, but
   hybrid latency remains an HLS estimate, not a board-measured distribution.
2. Search-mode latency for this exact `normstage` force-mode is not isolated;
   the combined max envelope is `16.359 ms`, still above `4 ms`.
3. Search active path remains Frame Conv -> shared ATTN/MLP/nonlinear path ->
   Head with ROM-only learned parameters and dispatcher prefetch.
4. Track-mode latency for this exact `normstage` force-mode is not isolated;
   the combined/controller envelope is still above the `1 ms` target.
5. Track active path remains Event Conv -> shared ATTN/MLP/nonlinear path ->
   Head with the same shared blocks.
6. Throughput/II/worst-case can be derived from HLS latency for a single
   deterministic transaction, but frame/event-rate p95/p99 requires repeated
   RTL or board traces.
7. Power remains Vivado post-route vectorless, Medium confidence. It includes
   modeled PS/PL/on-chip AXI/DMA in the Vivado design and excludes external
   board rail, sensor I/O, and measured DDR module power.
8. Final device-fit utilization should use the Vivado placed values above;
   HLS resources remain major-block attribution evidence only.

## Addendum: PAR32 Norm-URAM No-DONT-TOUCH Negative Control

This probe tested whether removing broad exported-HLS datapath preservation
could be used as the next timing closure step for the PAR32 ROM-only objective
path. It cannot be used as a valid accelerator result, because Vivado optimized
the learned compute fabric away.

| Stage | Result |
|---|---|
| Profile | `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16` |
| CSim | Pass; Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1` |
| CSynth | Pass; target `3.333 ns`, estimate `2.777 ns`, max `4,805,312` cycles / `16.016 ms` |
| CSynth resources | BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64` |
| Vivado route/bitgen | Pass; route errors `0`, bitgen completed |
| Routed timing | WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, THS `0.000 ns` |
| Placed utilization | CLB LUT `7,817/230,400 = 3.39%`, registers `11,522/460,800 = 2.50%`, Block RAM Tile `5/312 = 1.60%`, DSP `0/1,728 = 0.00%`, URAM `0/96 = 0.00%` |
| Routed power | Total `3.624 W`, dynamic `2.931 W`, static `0.693 W`, PS static `0.099 W`, PL static `0.595 W` |

Interpretation:

- The timing pass is not representative. The resource collapse from the
  preserved `normuram` implementation (`DSP 1,728`, `URAM 64`, `BRAM tile 216`)
  to `DSP 0`, `URAM 0`, `BRAM tile 5` means the learned compute datapath was
  pruned/optimized out during Vivado implementation.
- This negative control explains why global `DONT_TOUCH` removal cannot be the
  next baseline. Future work should selectively relax preservation around
  non-critical wrappers or known fanout bottlenecks, while preserving the
  arithmetic/cache datapath that proves full learned Transformer execution.
- For the eight requested reporting questions, this `nodt` result must not be
  used for resource, power, latency, throughput, or active-path reporting except
  as a cautionary note. Use `par32_dsp_mixed_stream_mem16` for physical/timing
  comparison, `par32_runtime_full_axi_mem16` for learned-AXI functional
  comparison, and preserved ROM-only PAR32 probes for objective-path evidence.

Updated eight-question status after `nodt`:

1. Hybrid resource/power: do not use `nodt`; use preserved `normuram` or
   `normstage` for objective-path resource/power, and keep earlier PAR32
   baselines as comparison rows.
2. Search-mode latency: unchanged; `nodt` did not produce valid placed compute.
3. Search active path: unchanged; valid evidence remains preserved ROM-only
   Frame Conv -> shared ATTN/MLP/nonlinear -> Head.
4. Track-mode latency: unchanged; `nodt` did not produce valid placed compute.
5. Track active path: unchanged; valid evidence remains preserved ROM-only
   Event Conv -> shared ATTN/MLP/nonlinear -> Head.
6. Throughput/II/worst-case: unchanged; derive only from valid HLS/placed
   compute profiles, not from `nodt`.
7. Power methodology: `nodt` vectorless power is a shell/pruned-design power
   number and should not be included in mode-specific power breakdown.
8. Source utilization: `nodt` utilization is useful only to detect pruning;
   it is not the accelerator source-utilization answer.

## Addendum: PAR32 Norm-URAM Top-DONT-TOUCH Negative Control

This probe tested whether preserving only the exported HLS top module could keep
the learned ROM-only objective datapath while allowing Vivado to optimize the
internal hierarchy. It also cannot be used as a valid accelerator result.

| Stage | Result |
|---|---|
| Profile | `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16` |
| Preservation mode | `HGTXR_E2E_OOC_DONT_TOUCH=top`; package patched `1` exported Verilog module |
| CSim | Pass; Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1` |
| CSynth | Pass; target `3.333 ns`, estimate `2.777 ns`, max `4,805,312` cycles / `16.016 ms` |
| CSynth resources | BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64` |
| Vivado route/bitgen | Pass; route errors `0`, bitgen completed |
| Routed timing | WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, THS `0.000 ns` |
| Placed utilization | CLB LUT `7,817/230,400 = 3.39%`, registers `11,522/460,800 = 2.50%`, Block RAM Tile `5/312 = 1.60%`, DSP `0/1,728 = 0.00%`, URAM `0/96 = 0.00%` |
| Routed power | Total `3.624 W`, dynamic `2.931 W`, static `0.693 W`, PS static `0.099 W`, PL static `0.595 W` |

Interpretation:

- Top-only preservation is not enough. The result is timing-clean only because
  the internal learned arithmetic/cache datapath was optimized/pruned away,
  producing the same shell-sized utilization as the `nodt` negative control.
- The earlier `par32_dsp_mixed_stream_mem16` and
  `par32_runtime_full_axi_mem16` profiles remain required comparison baselines:
  physical/timing and learned-AXI functional respectively.
- Valid objective-path reporting must continue to use preserved PAR32 ROM-only
  probes such as `normuram`/`corefabric`, despite their current timing failure,
  until selective preservation and explicit pipeline changes produce a
  non-pruned timing-clean implementation.

Updated eight-question status after `topdt`:

1. Hybrid resource/power: do not use `topdt`; use preserved objective-path
   resources or the established comparison baselines.
2. Search-mode latency: unchanged; `topdt` HLS latency is objective-scale, but
   its placed implementation is pruned and invalid.
3. Search active path: unchanged; valid evidence requires preserved Frame Conv
   -> shared ATTN/MLP/nonlinear -> Head datapath.
4. Track-mode latency: unchanged; `topdt` placed implementation is not a valid
   Track datapath.
5. Track active path: unchanged; valid evidence requires preserved Event Conv
   -> shared ATTN/MLP/nonlinear -> Head datapath.
6. Throughput/II/worst-case: do not derive from `topdt` placed timing; use valid
   HLS/placed compute profiles.
7. Power methodology: `topdt` vectorless power is pruned-shell power, not
   accelerator power.
8. Source utilization: `topdt` utilization is a pruning detector only, not the
   source-utilization answer.

## Addendum: PAR32 PipePref Fabric-Lane Closure Probes

The earlier best PAR32 profiles are still the required comparison baselines,
not discarded candidates:

| Profile | Role in the eight-question report | Reason |
|---|---|---|
| `par32_dsp_mixed_stream_mem16` | Physical/timing comparison baseline | Best prior routed PAR32 stream result: WNS `3.885 ns`, route errors `0`, bitgen pass, HLS BRAM_18K `332`, DSP `1,148`, LUT `193,546`, URAM `64` |
| `par32_runtime_full_axi_mem16` | Full learned AXI functional baseline | Restores learned Conv/ATTN/MLP/Head through external AXI weights; HLS Search `25.510 ms`, Track `3.136 ms` |
| ROM-only scheduler/prefetch profiles | Objective-path candidates | Required for the current claim: on-chip ROM weights/LUTs, runtime Search/Track scheduler, and full-block dispatcher prefetch in one design |

The latest pipe-prefetch/fabric-lane probes were run to see whether the PAR32
objective path can close ZCU104 DSP and timing without falling back to AXI
weights or to the older stream-only physical baseline.

| Profile | Stage reached | Main result | Signoff decision |
|---|---|---|---|
| `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16` | CSim/package/Vivado DRC | CSim passed; package max `4,817,564` cycles / `16.057 ms`; HLS resources BRAM_18K `386`, DSP `1,976`, LUT `453,254`, URAM `64`; Vivado DRC required DSP `1,740/1,728` | Invalid: DSP over-utilization |
| `..._pipepref_tail2_300_mem16` | Package | Max latency unchanged; HLS resources BRAM_18K `386`, DSP `1,966`, LUT `455,718`, URAM `64` | HLS-only probe; not promoted because DSP trend remained over budget |
| `..._pipepref_tail4_300_mem16` | Package/Vivado DRC | HLS DSP `1,946`, LUT `460,646`; Vivado DRC still required DSP `1,740/1,728` | Invalid: physical DSP unchanged enough to fail |
| `..._pipepref_cttail4_300_mem16` | Package | Template/switch lane split added via `HGTXR_E2E_CORE_LANE_CT_SWITCH=1`; HLS result matched `tail4`: max `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,946`, LUT `460,646`, URAM `64` | HLS-only negative result; no improvement over already failed `tail4` |
| `..._pipepref_tail8_300_mem16` | Package/Vivado DRC | HLS DSP `1,906`, LUT `470,502`; Vivado still materialized DSP48E2 `1,740` and failed DRC | Invalid: runtime tail-lane condition did not reduce final DSP |
| `..._pipepref_coreallfabric_300_mem16` | Package | HLS DSP `1,666`, LUT `529,792/230,400 = 229.9%` | Invalid: LUT impossible for ZCU104 |
| `..._pipepref_coreallfabric_nowide_300_mem16` | Package | HLS target `3.333 ns`, estimate `2.979 ns`, estimated Fmax `335.63 MHz`, max `4,817,820` cycles / `16.058 ms`, BRAM_18K `386`, DSP `1,666`, FF `286,960`, LUT `647,424/230,400 = 281%`, URAM `64` | Invalid: LUT impossible for ZCU104 |

Interpretation:

- The tail-lane experiment is a useful negative result. A runtime condition on
  `lane` can reduce HLS estimates, but it did not reduce the final Vivado
  DSP48E2 count enough; `tail4` and `tail8` both still failed at `1,740/1,728`.
- The template/switch lane split (`cttail4`) did not improve over `tail4`, so
  the next step needs a more explicit operator/data-path split rather than a
  wrapper around the same unrolled lane loop.
- All-core fabric proves a DSP-fit direction exists, but the dense
  Transformer multiply path becomes LUT-bound and exceeds the ZCU104 CLB LUT
  budget by more than 2x.
- The next valid experiment should not use global all-fabric mapping. It should
  either split operators at compile time so only a small known subset maps away
  from DSP in the final netlist, or keep DSP mapping and add explicit pipeline
  stages/MREG/PREG-friendly binding to close 300 MHz timing.

Updated eight-question status after pipe-prefetch/fabric-lane probes:

1. Hybrid resource/power: still use valid preserved ROM-only routed probes
   (`corefabric`/`normuram`) and the two fixed PAR32 comparison baselines.
   Do not use `coreallfabric*` for physical resource/power because LUT exceeds
   device capacity and no valid route exists.
2. Search-mode latency: the latest pipe-prefetch combined envelope remains
   about `16.057-16.058 ms`, above the `4 ms` target. Mode-specific Search
   rerun for these exact candidates is not useful until a physical candidate
   exists.
3. Search active path: unchanged for valid candidates: Frame Conv -> shared
   ATTN/MLP/nonlinear path -> Head, with ROM-only learned parameters and
   dispatcher prefetch.
4. Track-mode latency: unchanged for signoff purposes; prior PAR32 Track-only
   objective estimate is `2.100 ms`, still above the `1 ms` target, and the
   latest pipe-prefetch candidates are not valid physical points.
5. Track active path: unchanged for valid candidates: Event Conv -> shared
   ATTN/MLP/nonlinear path -> Head through the same shared blocks.
6. Throughput/II/worst-case: pipe-prefetch did not improve worst-case enough;
   combined interval remains about `4,817,565-4,817,821` cycles.
7. Power methodology: no new valid routed power exists for pipe-prefetch
   candidates because DRC or HLS resource overrun prevents representative
   implementation.
8. Source utilization: report `par32_dsp_mixed_stream_mem16` and
   `par32_runtime_full_axi_mem16` as comparison rows, but use preserved
   ROM-only objective probes for the objective-path utilization. Mark
   `coreallfabric*` as HLS infeasible on LUT.
