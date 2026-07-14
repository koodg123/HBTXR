# DeiT-Tiny Full Multi-Board Report

Date: 2026-04-10
Design preset: `full_deit_tiny_fit`
Model preset: `deit_tiny_baseline`

Targets covered:
- `vck190`
- `zu15eg`
- `zcu102`

Detailed per-target reports:
- `workspace/artifacts/reports/pnr/vck190_deit_tiny_full_status_20260410.md`
- `workspace/artifacts/reports/pnr/zu15eg_deit_tiny_full_status_20260410.md`
- `workspace/artifacts/reports/pnr/zcu102_deit_tiny_full_status_20260410.md`

## Executive Summary

| Target | Current status | Key outcome | Board image outputs |
| --- | --- | --- | --- |
| `vck190` | `step4-ooc` completed | OOC implementation finished, but timing is not clean at `400 MHz` | `step4-ooc` alone does not emit `xsa/hwh/bit`; `step5` now prepares full-board export |
| `zu15eg` | `step4-bitstream` synthesis completed, implementation failed | Post-synth timing met for setup, but placement stopped on resource over-utilization | No `xsa/hwh/bit/pdi` generated |
| `zcu102` | `step4-bitstream` synthesis completed, implementation failed | Post-synth timing met for setup, but placement stopped on resource over-utilization | No `xsa/hwh/bit` generated |

## VCK190

Experiment:
- `configs/experiments/vck190_full_deit_tiny_fit.json`

Build root:
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit`

Flow state:
- `step4-ooc` completed through route.
- This is an accelerator-only OOC flow, not a board-level image export flow.
- `step5` summary/handoff bundle was generated.
- `step5` full-board export Tcl was generated, and validate-only integration has been confirmed separately.

Final OOC timing:
- WNS: `-0.973 ns`
- TNS: `-38507.668 ns`
- Hold WNS: `0.005 ns`
- Hold TNS: `0.000 ns`
- Failing setup endpoints: `200,884 / 1,728,154`
- Clock target: `2.500 ns` (`400 MHz`)

Final OOC utilization:
- LUT: `644,533`
- Logic LUT: `567,791`
- LUTRAM: `70,973`
- FF: `502,742`
- RAMB36: `595`
- RAMB18: `78`
- URAM: `287`
- DSP: `743`

Generated outputs:
- Present:
  - `impl/project/post_route.dcp`
  - `impl/project/timing_summary.rpt`
  - `impl/project/utilization.rpt`
  - `impl/project/route_status.rpt`
  - `results/full_board_export/run_full_board_export.tcl`
- Not yet emitted by the completed run:
  - `xsa`
  - `hwh`
  - `bit`
  - `pdi`

Primary report files:
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/project/timing_summary.rpt`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/project/utilization.rpt`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/project/route_status.rpt`
- `workspace/artifacts/reports/vck190/deit_tiny_baseline/full_deit_tiny_fit/step5_vck190_pnr_summary.md`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/results/full_board_export/full_board_export_plan.json`

Interpretation:
- `vck190` is the only target that completed routed implementation in the current experiment set.
- However, the routed OOC design still misses setup timing at `400 MHz`.
- Board-image export is now wired into `step5`, but end-to-end `xsa/hwh/pdi(bit)` generation has not yet been run to completion in this report set.

## ZU15EG

Experiment:
- `configs/experiments/zu15eg_full_deit_tiny_fit.json`

Build root:
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit`

Flow state:
- Top-level synthesis completed.
- Post-synth reports were generated.
- `opt_design` completed.
- `place_design` did not start because pre-place DRC stopped the run on resource over-utilization.

Post-synth timing snapshot:
- Device: `xczu15eg-ffvb1156`
- Setup WNS: `1.405 ns`
- Setup TNS: `0.000 ns`
- Hold WNS: `-0.097 ns`
- Hold TNS: `-51460.258 ns`

Post-synth top utilization:
- Total LUTs: `797,066`
- Logic LUTs: `595,123`
- LUTRAMs: `185,969`
- FFs: `512,383`
- RAMB36: `753`
- RAMB18: `16`
- URAM: `0`
- DSP: `891`

Implementation failure point:
- `opt_design completed successfully`
- `place_design failed`
- Failure reason: DRC `UTLZ-1` resource over-utilization

Pre-place DRC over-utilization:
- LUT as Logic: `590,531 / 341,280` (`173.0%`, excess `249,251`)
- LUT as Distributed RAM: `185,861 / 184,320` (`100.8%`, excess `1,541`)
- LUT as Memory: `201,103 / 184,320` (`109.1%`, excess `16,783`)
- Slice LUTs: `791,634 / 341,280` (`231.9%`, excess `450,354`)
- RAMB18/RAMB36/FIFO: `1,522 / 1,488` (`102.3%`, excess `34`)
- RAMB36/FIFO: `753 / 744` (`101.2%`, excess `9`)

Generated outputs:
- Present:
  - `project/bitstream_post_synth_utilization.rpt`
  - `project/bitstream_post_synth_timing_summary.rpt`
  - `project/...runs/impl_1/hgpipe_bd_wrapper_opt.dcp`
  - `project/...runs/impl_1/hgpipe_bd_wrapper_drc_opted.rpt`
- Not present:
  - `xsa`
  - `hwh`
  - `bit`
  - `pdi`

Primary report files:
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_utilization.rpt`
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_timing_summary.rpt`
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/project/zu15eg_full_deit_tiny_fit_bitstream.runs/impl_1/runme.log`
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/bitstream_post_impl.log`

Interpretation:
- `zu15eg` can synthesize the full design and produce post-synth reports.
- The design is still too large for final placement on this device.
- The main blocker is LUT pressure, with BRAM also slightly over limit.

## ZCU102

Experiment:
- `configs/experiments/zcu102_full_deit_tiny_fit.json`

Build root:
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit`

Flow state:
- Top-level synthesis completed.
- Post-synth reports were generated.
- `opt_design` completed.
- `place_design` did not start because pre-place DRC stopped the run on resource over-utilization.

Post-synth timing snapshot:
- Device: `xczu9eg-ffvb1156`
- Setup WNS: `2.044 ns`
- Setup TNS: `0.000 ns`
- Hold WNS: `-0.096 ns`
- Hold TNS: `-44173.332 ns`

Post-synth top utilization:
- Total LUTs: `726,706`
- Logic LUTs: `571,522`
- LUTRAMs: `145,313`
- FFs: `525,640`
- RAMB36: `890`
- RAMB18: `76`
- URAM: `0`
- DSP: `1,254`

Implementation failure point:
- `opt_design completed successfully`
- `place_design failed`
- Failure reason: DRC `UTLZ-1` resource over-utilization

Pre-place DRC over-utilization:
- LUT as Logic: `567,104 / 274,080` (`206.9%`, excess `293,024`)
- LUT as Distributed RAM: `145,205 / 144,000` (`100.8%`, excess `1,205`)
- LUT as Memory: `154,872 / 144,000` (`107.5%`, excess `10,872`)
- Slice LUTs: `721,976 / 274,080` (`263.4%`, excess `447,896`)
- RAMB18/RAMB36/FIFO: `1,856 / 1,824` (`101.8%`, excess `32`)

Generated outputs:
- Present:
  - `project/bitstream_post_synth_utilization.rpt`
  - `project/bitstream_post_synth_timing_summary.rpt`
  - `project/bitstream_utilization.rpt`
  - `project/bitstream_timing_summary.rpt`
  - `project/...runs/impl_1/hgpipe_bd_wrapper_opt.dcp`
- Not present:
  - `xsa`
  - `hwh`
  - `bit`

Primary report files:
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_utilization.rpt`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_timing_summary.rpt`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/zcu102_full_deit_tiny_fit_bitstream.runs/impl_1/runme.log`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_utilization.rpt`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_timing_summary.rpt`

Interpretation:
- `zcu102` also synthesizes successfully but fails before placement.
- Compared with `zu15eg`, the LUT shortfall is even more severe.
- This target is further from routable closure than `zu15eg`.

## Cross-Target Takeaways

- `vck190`:
  - only target that completed routed implementation
  - still fails setup timing at the current `400 MHz` OOC target
- `zu15eg`:
  - synthesis okay, implementation blocked mainly by LUT overuse
- `zcu102`:
  - synthesis okay, implementation blocked mainly by an even larger LUT overuse

Most important comparison:
- Board-image generation is currently a `flow availability` issue only for `vck190`.
- For `zu15eg` and `zcu102`, board-image generation is blocked earlier by real device capacity limits.
