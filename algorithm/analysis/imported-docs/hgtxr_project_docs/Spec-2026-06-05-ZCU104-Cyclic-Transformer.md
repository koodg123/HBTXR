# HGTXR ZCU104 Cyclic Transformer Accelerator Spec

Date: 2026-06-05
Target: Xilinx ZCU104, Vitis/Vivado 2023.2, PYNQ overlay

## Goal

Move the current HGTXR HLS-to-PYNQ flow from a PYNQ-compatible scaffold toward a paper-faithful Transformer Block accelerator with a cyclic execution structure and ZCU104 resource fit.

## Inputs And Evidence Sources

- HLS source: `hardware/hls/src`, `hardware/hls/include`
- HLS top: `hardware/hls/src/hgtxr_top.cpp`
- Vivado scripts: `hardware/vivado/scripts`
- Reference resource image: `../PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png`
- Experimental references: `../impl_repos`
- Generated HLS IP: `hardware/generated/hgtxr_hls/solution/impl/ip`
- PYNQ package: `hardware/pynq/hgtxr`

## Design Requirements

1. Keep the top function PYNQ-friendly with AXI4 master ports for tensor buffers and AXI4-Lite control registers.
2. Preserve the existing `csynth -> export IP -> Vivado block design -> bit/hwh -> PYNQ driver` flow.
3. Implement the Transformer Block as reusable HLS modules:
   - LayerNorm or normalization substitute with explicit fixed-point policy.
   - QKV projection.
   - Attention score and weighted-value datapath.
   - MLP or feed-forward stage.
   - Residual state update.
4. Implement a cyclic accelerator control structure so reusable compute engines are scheduled across tiles, heads, tokens, and channel blocks rather than fully unrolling the whole model.
5. Parameterize tunable hardware dimensions:
   - `HGTXR_TILE_TOKENS`
   - `HGTXR_TILE_CHANNELS`
   - `HGTXR_HEAD_PAR`
   - `HGTXR_PE_PAR`
   - `HGTXR_BUS_WIDTH_BITS`
   - `HGTXR_DATA_W`
   - `HGTXR_DATA_I`
   - `HGTXR_ACC_W`
   - `HGTXR_ACC_I`
   - `HGTXR_LOCAL_BUFFER_DEPTH`
   - `HGTXR_STREAM_FIFO_DEPTH`
6. Fit on ZCU104 with explicit resource gates for LUT, FF, DSP, BRAM, URAM, timing, and AXI bandwidth.
7. Generate reports that compare resource use against the DeiT-Tiny C-synthesis reference image and the `impl_repos` experiment results.

## Initial Parameter Policy

The first ZCU104-fit implementation should prefer reuse over aggressive unroll:

- Data type: start from fixed-point `ap_fixed<16,6>` unless the reference experiment proves another width.
- Accumulator: use at least 32-bit fixed-point for MAC reductions.
- Parallelism: begin with small PE/head parallelism, then sweep upward.
- Tiling: tile tokens and channels independently.
- Memory: keep external buffers AXI-backed and stage active tiles into BRAM/URAM-capable local buffers.
- FIFOs: expose stream FIFO depth as compile-time constants and sweep for II/timing closure.

## Experiment Plan

1. Baseline preservation:
   - Compile host C++ smoke test.
   - Run Vitis HLS csynth.
   - Export HLS IP.
   - Build Vivado BD and generate bit/hwh.
   - Run local PYNQ driver unit smoke.
2. Resource reference extraction:
   - Inspect `DeiT-Tiny C-Syn Results.png` and transcribe resource/latency targets into a markdown table.
   - Inspect `impl_repos` for prior synthesis and implementation experiments.
3. Transformer module implementation:
   - Add parameter header.
   - Add tiled linear/MAC primitive.
   - Add cyclic scheduler.
   - Add Transformer Block wrapper.
   - Integrate into `hgtxr_top.cpp` without breaking AXI register compatibility.
4. Sweep:
   - Sweep tile sizes, PE parallelism, bus width, bit width, buffer depth, and FIFO depth.
   - Record csynth resource/latency/timing.
   - Promote only configurations that can plausibly route on ZCU104.
5. Implementation:
   - Rebuild bit/hwh for selected candidate.
   - Copy overlay artifacts to `hardware/generated/build/vivado/overlay/hgtxr_overlay` and `hardware/pynq/hgtxr`.
6. PYNQ validation:
   - Verify `.hwh` register offsets match the driver.
   - Run board-side DMA/buffer smoke when hardware is available.

## Acceptance Criteria

- `hgtxr_top.cpp` has PYNQ-friendly AXI pragmas and HLS report confirms AXI4 master plus AXI4-Lite control interfaces.
- HLS csynth completes for the selected parameter set.
- HLS IP export produces `component.xml`.
- Vivado BD validates and bitstream generation completes for ZCU104.
- `.bit` and `.hwh` are copied to the overlay package.
- PYNQ driver can allocate buffers, program 64-bit addresses, start the IP, and read output state.
- The selected implementation has a documented ZCU104 fit judgment based on actual utilization/timing reports.
- Paper-reproduction gaps are explicitly documented instead of implied complete.

## Known Risks

- The current HLS implementation may be a structural surrogate rather than the full paper architecture.
- Full DeiT-Tiny attention can exceed ZCU104 resources without cyclic reuse and tiling.
- Fixed-point accuracy must be validated against golden vectors before claiming paper-equivalent reproduction.
- Vivado routing can fail even when HLS resource estimates look acceptable.
- AXI bandwidth and PS-PL port selection may bottleneck multi-buffer attention traffic.

