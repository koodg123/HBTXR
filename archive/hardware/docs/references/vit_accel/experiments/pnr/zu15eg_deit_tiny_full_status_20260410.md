# ZU15EG DeiT-Tiny Full Detailed Report

Date: 2026-04-10
Design preset: `full_deit_tiny_fit`
Model preset: `deit_tiny_baseline`
Experiment: `zu15eg_full_deit_tiny_fit`
Target: `zu15eg`

## Summary

- Current flow status: top-level synthesis completed, implementation failed before placement
- Outcome: post-synth timing and utilization reports were generated
- Main limitation: pre-place DRC stopped the run on resource over-utilization
- Board-image export status: no `xsa`, `hwh`, `bit`, or `pdi` was generated

## Build Root

- Build root: `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit`

## Flow Interpretation

- `step4-bitstream` did complete synthesis and `opt_design`.
- The run did not proceed to placement because Vivado DRC `UTLZ-1` stopped the flow before `place_design`.
- This is not a wrapper hang at this stage; it is a real capacity limit on the device.

## Post-Synth Timing Snapshot

- Device: `xczu15eg-ffvb1156`
- Setup WNS: `1.405 ns`
- Setup TNS: `0.000 ns`
- Hold WNS: `-0.097 ns`
- Hold TNS: `-51460.258 ns`

## Post-Synth Top Utilization

- Total LUTs: `797,066`
- Logic LUTs: `595,123`
- LUTRAMs: `185,969`
- FFs: `512,383`
- RAMB36: `753`
- RAMB18: `16`
- URAM: `0`
- DSP: `891`

## Implementation Failure Point

- `opt_design completed successfully`
- `place_design failed`
- Failure reason: DRC `UTLZ-1` resource over-utilization

## Pre-Place DRC Over-Utilization

- LUT as Logic: `590,531 / 341,280`
  - Usage: `173.0%`
  - Excess: `249,251`
- LUT as Distributed RAM: `185,861 / 184,320`
  - Usage: `100.8%`
  - Excess: `1,541`
- LUT as Memory: `201,103 / 184,320`
  - Usage: `109.1%`
  - Excess: `16,783`
- Slice LUTs: `791,634 / 341,280`
  - Usage: `231.9%`
  - Excess: `450,354`
- RAMB18/RAMB36/FIFO: `1,522 / 1,488`
  - Usage: `102.3%`
  - Excess: `34`
- RAMB36/FIFO: `753 / 744`
  - Usage: `101.2%`
  - Excess: `9`

## Generated Outputs

- Present:
  - `project/bitstream_post_synth_utilization.rpt`
  - `project/bitstream_post_synth_timing_summary.rpt`
  - `project/zu15eg_full_deit_tiny_fit_bitstream.runs/impl_1/hgpipe_bd_wrapper_opt.dcp`
  - `project/zu15eg_full_deit_tiny_fit_bitstream.runs/impl_1/hgpipe_bd_wrapper_drc_opted.rpt`
- Not present:
  - `xsa`
  - `hwh`
  - `bit`
  - `pdi`

## Primary Report Files

- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_utilization.rpt`
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_timing_summary.rpt`
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/project/zu15eg_full_deit_tiny_fit_bitstream.runs/impl_1/runme.log`
- `workspace/artifacts/build/zu15eg/full_deit_tiny_fit/deit_tiny_baseline/zu15eg_full_deit_tiny_fit/bitstream/bitstream_post_impl.log`

## Conclusion

- `zu15eg` can synthesize the full design and emit post-synth reports.
- The design is too large to enter placement on this device.
- The dominant blocker is LUT pressure, with BRAM also slightly above capacity.
