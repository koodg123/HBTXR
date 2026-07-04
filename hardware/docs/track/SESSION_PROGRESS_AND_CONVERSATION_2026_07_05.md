# Session Progress And Conversation Summary - 2026-07-05

## Scope

This document captures the current HGTXR work state and the conversation path
that led to the present commit request. It is written for continuation from the
repository root:

```text
/home/kjm26/project/PRJXR/XR-VIT/HGTXR
```

Current git topology was rechecked on 2026-07-05:

- Root `git rev-parse --show-toplevel`: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`
- `software/` resolves to the same root repository.
- `hardware/` resolves to the same root repository.
- Current branch: `kjm26/feat/capture-hgtxr-current-state`

Therefore this is a single root-repository commit, not separate `software/` and
`hardware/` nested commits.

## Conversation Timeline

### Repository And Codex Environment

- The user first asked how to commit when the root, `software/`, and `hardware/`
  appeared to have separate git directories. The resulting policy was to verify
  the actual top-level repository first, then commit from the root when all
  paths resolve to the same repo.
- The user then asked to track all current changes and commit. A root branch
  and broad snapshot commit workflow was established.
- The user later asked to ignore generated `run` outputs and recommit. The
  accepted policy was to keep generated run directories local-only, especially
  `software/runs/`.
- Separate Codex environment work followed: tools under `/home/kjm26/agent/tools`
  and `useful-tools.txt` were analyzed, multiple plugins/MCP/skills were
  installed or registered, and a Codex handover package was generated for
  transfer to another machine. Those environment actions are not part of the
  current HGTXR hardware commit except as historical context.

### Hardware Power, Clock, And Report Analysis

- The user requested power report interpretation from the HGTXR hardware Vivado
  implementation directory.
- The power discussion separated PL, PS, AXI, DMA, and DDR/DRAM boundaries:
  AXI/DMA are PL-side infrastructure, while Vivado PS8/DDR treatment is not the
  same as an external board rail measurement.
- The user then asked for clock information, including PS ARM and DRAM
  frequencies. The answer distinguished Vivado timing clocks from PS/DDR
  platform configuration evidence.
- Subsequent power requests asked for static and dynamic breakdown by PS/PL and
  DRAM/ARM/AXI/DMA/IP categories. The current methodology remains: Vivado
  vectorless reports can separate top BD hierarchy, but not internal
  Conv/ATTN/MLP/Head dynamic power without deeper hierarchy or activity data.

### Search/Track Semantics And Runtime Mode

- The user challenged why Search and Track profiles were separate when the
  cyclic accelerator should share two ATTN and MLP blocks temporally across the
  model.
- The target was clarified as a single E2E AXIS cyclic accelerator with shared
  ATTN/MLP blocks and runtime Search/Track branching.
- The user requested HW emulation and asked whether `v++ hw_emu` or HLS RTL
  co-simulation is more accurate. The resulting direction was to use HLS/RTL
  co-simulation for kernel-cycle fidelity and board/PYNQ execution for system
  measurements, while keeping Vitis `hw_emu` distinct from Vivado block-design
  and PYNQ evidence.

### Fifteen-Step Goal Plan And Eight Questions

- The user asked for the plan toward:

```text
Configure TOP so one E2E AXIS cyclic accelerator shares identical ATTN/MLP
blocks while performing Search/Track runtime branching, and reach Search 4 ms
and Track 1 ms on ZCU104.
```

- A 15-step checklist was used to structure progress: learned parameter paths,
  runtime scheduler, Search/Track paths, HLS validation, Vivado fit, power,
  latency, board measurement, and reporting gates.
- The user then supplied eight reporting questions covering:
  hybrid resource/latency/power, Search/Track-only metrics, active block/path,
  throughput/II/rate/worst-case, measurement methodology, DMA, batch size,
  percentile latency, mode distribution, and major-block utilization.
- Those questions led to `AQ2` Search/Track metric documentation and to the
  rule that deterministic HLS estimates must not be mislabeled as measured
  board percentiles.

### Learned Path Correction

- The user asked whether nonlinear LUTs and parameters for Frame Conv, Event
  Conv, Track Transformer, and Head were actually stored as ROM.
- The implementation discussion distinguished placeholder/fastpath behavior
  from full learned parameter paths.
- The user explicitly requested:
  - restore Conv/Head to learned parameter paths;
  - choose QKV/MLP weight path via AXI or ROM/BRAM/URAM clearly;
  - remeasure Search/Track latency;
  - separate fastpath latency from full-model latency.
- Full AXI learned and ROM-only learned profiles were then documented, along
  with the latency gap between fastpath/runtime-mode estimates and full learned
  Transformer implementations.

### Full Learned Physical And 300 MHz Policy

- The user asked why the stronger historical profiles,
  `par32_runtime_full_axi_mem16` and `par32_dsp_mixed_stream_mem16`, were not
  treated as baselines. The resulting policy was to keep them as baselines:
  one as full learned AXI functional baseline and one as a physical/timing
  baseline.
- The user fixed the experiment clock target at `300 MHz` and allowed routed
  WNS down to `-0.500 ns` for experiment continuation. Official clean timing
  remains `WNS >= 0.000 ns`.
- The user asked to relieve DSP pressure by moving some DSP work to LUT and
  moving small high-bank memories from BRAM to LUTRAM. Several pressure-relief
  profiles were tested, including nonlinear LUTROM, fabric tail lanes,
  DSP pipeline latency, targeted token LUTRAM, tail4/tail16 variants, and
  `compute_keep`.
- A critical rule emerged: timing-clean is not sufficient. The `compute_keep`
  candidate achieved positive WNS but collapsed to a pruned shell, so future
  candidates require a resource-preservation gate.

### Best Search/Track Latency And AQ2 Reporting

- The fastest full learned HLS Search/Track pair observed so far is based on
  `AQ8 + hpar8 + w1c2 + qkvbram_tail2 + exppart`:
  - Search: about `1,288,836 cycles / 4.296 ms`
  - Track: about `229,783 cycles / 0.766 ms`
  - 10/90 hybrid mean II: about `1.119 ms`
  - Not promotable because direct hidden-lane parallelism exceeds ZCU104 DSP
    and LUT capacity.
- The fastest Search-only point with `ACC24` reached:
  - Search: `1,284,676 cycles / 4.282 ms`
  - Still above the 4 ms target and still not resource-fit.
- AQ2 remains the clean requested metric package:
  - Search latency: `5.699210-5.699277 ms`
  - Track latency: `0.895700 ms`
  - Search implemented power: `7.032 W`
  - Track implemented power: `6.777 W`
  - Search implemented resources: LUT `77,494`, BRAM tile `191.5`, URAM `92`,
    DSP `1,257`
  - Track implemented resources: LUT `80,565`, BRAM tile `185.5`, URAM `64`,
    DSP `1,270`
- AQ2 still lacks board histograms, measured DMA counters, SAIF/VCD activity
  power, external DDR/sensor I/O rail power, and internal Conv/ATTN/MLP/Head
  dynamic power split.

### Reference Analysis And ZCU104 Plan

- The user requested a detailed experiment plan for ZCU104 fit plus full
  learned physical implementation and asked to directly read all files under:
  - `hardware/analysis`
  - `/home/kjm26/project/PRJXR/References/reports`
- A direct read pass covered `548` files, `213,905` lines, and `15,649,226`
  bytes. No persistent index artifact is used in the final plan.
- The resulting plan was saved in:
  - `hardware/docs/track/ZCU104_FULL_LEARNED_PHYSICAL_EXPERIMENT_PLAN_2026_07_01.md`
- Priority from that plan:
  1. Add resource-preservation gates before accepting any timing result.
  2. Close Search latency first; Track already meets 1 ms in AQ2.
  3. Avoid repeating direct HPar16 or PatchPar64; fix GELU ROM banking and
     shared/time-multiplexed W2 instead.
  4. Run resource fit cleanup together with latency DSE.
  5. Use exact tiled attention before sparse/linear approximations.
  6. Gate PoT/PTF/LIS and nonlinear replacements with SW/CSim evidence.
  7. Promote through CSim, CSynth, IP, Vivado route, resource-preservation,
     timing, and board measurements.

## Current Working State

### Modified Tracked Files

The current tracked modifications cover:

- `hardware/docs/Validation.md`
- `hardware/docs/track/PROGRESS.md`
- `hardware/docs/track/log.md`
- HLS data types and E2E Transformer implementation:
  - `hardware/hls/include/fixed_types.h`
  - `hardware/hls/include/hgtxr_e2e_vit.hpp`
  - `hardware/hls/src/hgtxr_e2e_axis_top.cpp`
  - `hardware/hls/src/hgtxr_e2e_m_axi_top.cpp`
  - `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`
- PYNQ overlay/runtime helpers:
  - `hardware/pynq/hgtxr/README.md`
  - `hardware/pynq/hgtxr/e2e_axis_dma_overlay.py`
  - `hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py`
  - `hardware/pynq/hgtxr/test_hgtxr_overlay.py`
- HLS/Vivado run scripts:
  - `hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh`
  - `hardware/scripts/run/run_e2e_q4w8a_no_board.sh`
  - `hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl`
  - `hardware/vivado/scripts/package_e2e_axis_ip.tcl`
  - `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl`
  - `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`
- Tooling/tests for PYNQ bundles, smoke results, remote smoke plans, and
  runtime-mode latency gates.

### New Documentation Artifacts

New untracked documentation prepared for commit includes:

- `hardware/docs/track/AQ2_REQUESTED_SEARCH_TRACK_METRICS_2026_07_01.md`
- `hardware/docs/track/AQ2_SEARCH_TRACK_MEASUREMENT_2026_06_30.md`
- `hardware/docs/track/AQ2_SEARCH_TRACK_METRIC_CHECKLIST_2026_07_01.md`
- `hardware/docs/track/EIGHT_QUESTION_FULL_LEARNED_VIVADO_EXPERIMENT_2026_06_28.md`
- `hardware/docs/track/EIGHT_QUESTION_HYBRID_RUNTIME_REPORT_2026_06_28.md`
- `hardware/docs/track/RUNTIME_MODE_E2E_SHARED_TOP_2026_06_27.md`
- `hardware/docs/track/RUNTIME_MODE_FULL_AXI_WEIGHT_PATH_2026_06_28.md`
- `hardware/docs/track/ZCU104_FULL_LEARNED_PHYSICAL_EXPERIMENT_PLAN_2026_07_01.md`
- `hardware/docs/track/SESSION_PROGRESS_AND_CONVERSATION_2026_07_05.md`

### New Hardware/PYNQ Artifacts

New untracked `.bit` and `.hwh` files under `hardware/pynq/hgtxr/` capture
multiple runtime, full learned, AQ2, pressure-relief, and Search/Track force-mode
overlays. Individual `.bit` files are about `19.3 MB`, below the 100 MB
single-file threshold.

### Removed Root Scratch Files

Two untracked root-level zero-byte files were present during commit
preparation:

- `1838`
- `384`

They appear to be accidental shell-redirection artifacts from numeric resource
values. They were removed before staging because they are not meaningful
hardware evidence and would otherwise pollute the root repository.

## Current Technical Status

### What Is Achieved

- Runtime Search/Track scheduler exists and is validated in CSim for multiple
  profiles.
- Full learned AXI and ROM-only learned profiles exist and are documented.
- Frame Conv, Event Conv, Head, Track ROM, Transformer ROM, nonlinear LUT ROM,
  and prefetch-all4 structural intent are documented.
- Active 300 MHz combined runtime candidate has routed evidence and bitgen pass
  under the experiment WNS floor.
- AQ2 Search/Track metric package is documented with latency, II, DMA model,
  throughput, major-block resources, implemented resources, timing, and
  vectorless power.
- Hybrid 10/90 PYNQ runner and bundle plumbing exist.
- Multiple negative DSE cases are documented: PatchPar64, HPar16, AQ6, W1 bank,
  token bank, direct hpar8/hpar16 resource overrun, and pruned `compute_keep`.

### What Remains Open

- Search latency target `<= 4 ms` is not met by any resource-fit full learned
  candidate.
- Track target `<= 1 ms` is met in AQ2 and fastest HLS pairs, but final
  promoted combined full learned candidate still needs complete board evidence.
- Full RTL co-simulation remains blocked by XSIM launch/runtime issues.
- Board Search/Track/hybrid JSONs are missing.
- Measured DMA bandwidth and p95/p99 distributions are missing.
- SAIF/VCD or board power is needed for activity-aware power.
- External DDR/sensor I/O rails are not covered by current Vivado vectorless
  reports.
- Internal Conv/ATTN/MLP/Head dynamic power split requires deeper hierarchy or
  explicit power-report hooks.

## Next Recommended Work

1. Do not accept timing-clean candidates unless full learned resource
   preservation is proven.
2. Implement the Search-focused P1 plan:
   - GELU ROM banking/replication for hpar16-style read parallelism.
   - Shared/time-multiplexed W2 path instead of direct hidden-lane duplication.
3. Keep AQ2 as the clean reporting baseline and fastest `hpar8/ACC24` as the
   latency lower-bound reference.
4. Run Vivado only for candidates that either meet Search/Track latency or
   explain a specific resource/timing tradeoff.
5. Capture ZCU104 board JSONs for Search, Track, and 10/90 hybrid once a
   promotable bit/hwh pair is selected.

## Commit Preparation Status

Before this document was added, `git status --porcelain` reported `119` changed
or untracked entries. No currently visible status-entry file exceeds 100 MB,
though ignored generated directories and `software/.venv` contain very large
files and remain outside the commit candidate set.

The commit should capture:

- Documentation and conversation/progress summaries.
- HLS full learned/runtime-mode implementation changes.
- PYNQ runtime and validation tooling.
- Vivado/HLS run-script extensions.
- PYNQ bit/hwh handoff artifacts that are intentionally present under
  `hardware/pynq/hgtxr/`.
