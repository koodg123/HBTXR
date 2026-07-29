# ZCU102 DeiT-Tiny Full Detailed Report

Date: 2026-04-10
Design preset: `full_deit_tiny_fit`
Model preset: `deit_tiny_baseline`
Experiment: `zcu102_full_deit_tiny_fit`
Target: `zcu102`

## Summary

- Current flow status: top-level synthesis completed, implementation failed before placement
- Outcome: post-synth timing and utilization reports were generated
- Main limitation: pre-place DRC stopped the run on resource over-utilization
- Board-image export status: no `xsa`, `hwh`, or `bit` was generated

## Build Root

- Build root: `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit`

## Flow Interpretation

- `step4-bitstream` did complete synthesis and `opt_design`.
- The run did not proceed to placement because Vivado DRC `UTLZ-1` stopped the flow before `place_design`.
- The failure mode is the same class as `zu15eg`, but the LUT deficit is even larger.

## Post-Synth Timing Snapshot

- Device: `xczu9eg-ffvb1156`
- Setup WNS: `2.044 ns`
- Setup TNS: `0.000 ns`
- Hold WNS: `-0.096 ns`
- Hold TNS: `-44173.332 ns`

## Post-Synth Top Utilization

- Total LUTs: `726,706`
- Logic LUTs: `571,522`
- LUTRAMs: `145,313`
- FFs: `525,640`
- RAMB36: `890`
- RAMB18: `76`
- URAM: `0`
- DSP: `1,254`

## Implementation Failure Point

- `opt_design completed successfully`
- `place_design failed`
- Failure reason: DRC `UTLZ-1` resource over-utilization

## Pre-Place DRC Over-Utilization

- LUT as Logic: `567,104 / 274,080`
  - Usage: `206.9%`
  - Excess: `293,024`
- LUT as Distributed RAM: `145,205 / 144,000`
  - Usage: `100.8%`
  - Excess: `1,205`
- LUT as Memory: `154,872 / 144,000`
  - Usage: `107.5%`
  - Excess: `10,872`
- Slice LUTs: `721,976 / 274,080`
  - Usage: `263.4%`
  - Excess: `447,896`
- RAMB18/RAMB36/FIFO: `1,856 / 1,824`
  - Usage: `101.8%`
  - Excess: `32`

## Generated Outputs

- Present:
  - `project/bitstream_post_synth_utilization.rpt`
  - `project/bitstream_post_synth_timing_summary.rpt`
  - `project/bitstream_utilization.rpt`
  - `project/bitstream_timing_summary.rpt`
  - `project/zcu102_full_deit_tiny_fit_bitstream.runs/impl_1/hgpipe_bd_wrapper_opt.dcp`
- Not present:
  - `xsa`
  - `hwh`
  - `bit`

## Primary Report Files

- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_utilization.rpt`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_post_synth_timing_summary.rpt`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/zcu102_full_deit_tiny_fit_bitstream.runs/impl_1/runme.log`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_utilization.rpt`
- `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/bitstream_timing_summary.rpt`

## Conclusion

- `zcu102` also synthesizes successfully but fails before placement.
- Compared with `zu15eg`, the LUT shortfall is more severe.
- In the current experiment set, `zcu102` is further from routable closure than `zu15eg`.
