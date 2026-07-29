# VCK190 DeiT-Tiny Full Detailed Report

Date: 2026-04-10
Design preset: `full_deit_tiny_fit`
Model preset: `deit_tiny_baseline`
Experiment: `vck190_full_deit_tiny_fit`
Target: `vck190`

## Summary

- Current flow status: `step4-ooc` completed through routed OOC implementation
- Outcome: routed accelerator checkpoint and reports were generated
- Main limitation: setup timing is not clean at the current `400 MHz` target
- Board-image export status: not produced by `step4-ooc`; `step5` prepared the full-board export scaffold

## Build Roots

- Build root: `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit`
- Step5 summary: `workspace/artifacts/reports/vck190/deit_tiny_baseline/full_deit_tiny_fit/step5_vck190_pnr_summary.md`
- Results root: `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/results`

## Flow Interpretation

- `step4-ooc` is an accelerator-only OOC implementation flow.
- It generates routed accelerator artifacts such as `post_route.dcp`, timing reports, utilization reports, and routed netlists.
- It does not by itself emit `xsa`, `hwh`, `bit`, or `pdi`.
- `step5` now gathers the OOC outputs and prepares a full-board export scaffold under `results/full_board_export/`.

## Final OOC Timing

- Clock target: `2.500 ns` (`400.000 MHz`)
- Setup WNS: `-0.973 ns`
- Setup TNS: `-38507.668 ns`
- Hold WNS: `0.005 ns`
- Hold TNS: `0.000 ns`
- Failing setup endpoints: `200,884 / 1,728,154`

## Final OOC Utilization

- LUT: `644,533`
- Logic LUT: `567,791`
- LUTRAM: `70,973`
- FF: `502,742`
- RAMB36: `595`
- RAMB18: `78`
- URAM: `287`
- DSP: `743`

## Worst Reported Timing Path

- Slack: `-0.973 ns`
- Source:
  - `block_0/do_patch_embed_U0/step2_mac_replace_shift_U0/mac_muladd_8s_8s_21s_21_4_1_U380/PATCH_EMBED_mac_muladd_8s_8s_21s_21_4_1_DSP48_1_U/p_reg_reg/DSP58C<0>_INST/DSP_OUTPUT58_INST/CLK`
- Destination:
  - `block_0/do_patch_embed_U0/step2_mac_replace_shift_U0/mac_muladd_8s_8s_21s_21_4_1_U380/PATCH_EMBED_mac_muladd_8s_8s_21s_21_4_1_DSP48_1_U/p_reg_reg/DSP58C<0>_INST/DSP_OUTPUT58_INST/ALU_OUT[15]`
- Interpretation:
  - The worst path is inside `PATCH_EMBED`, specifically around the DSP-heavy MAC datapath.
  - This matches the overall observation that the design can route on `vck190`, but still misses setup timing at the current target clock.

## Generated Outputs

- Present:
  - `impl/project/post_route.dcp`
  - `impl/project/timing_summary.rpt`
  - `impl/project/utilization.rpt`
  - `impl/project/route_status.rpt`
  - `impl/project/post_route_netlist.v`
  - `results/full_board_export/run_full_board_export.tcl`
  - `results/full_board_export/full_board_export_plan.json`
- Not yet emitted by the completed report set:
  - `xsa`
  - `hwh`
  - `bit`
  - `pdi`

## Primary Report Files

- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/project/timing_summary.rpt`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/project/utilization.rpt`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/project/route_status.rpt`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/impl.log`
- `workspace/artifacts/reports/vck190/deit_tiny_baseline/full_deit_tiny_fit/step5_vck190_pnr_summary.md`
- `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/results/full_board_export/full_board_export_plan.json`

## Conclusion

- `vck190` is the only target in this report set that completed routed implementation.
- The current experiment is still not timing-clean at `400 MHz`.
- The next engineering question is not basic routability, but timing closure and, separately, full-board export completion for `xsa/hwh/pdi(bit)`.
