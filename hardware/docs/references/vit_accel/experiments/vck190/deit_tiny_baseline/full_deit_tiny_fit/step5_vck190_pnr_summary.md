# VCK190 Step5 PnR Summary

- Date: 2026-04-10 09:49:58 
- Experiment: `vck190_full_deit_tiny_fit`
- Target: `vck190`
- Build dir: `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit`
- Bundle dir: `workspace/artifacts/deploy/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/pynq_step5`
- Results dir: `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/results`

## Timing

- WNS: `-0.973 ns`
- TNS: `-38507.668 ns`
- Hold WNS: `0.005 ns`
- Hold TNS: `0.000 ns`
- Failing setup endpoints: `200884` / `1728154`
- Clock: `2.500 ns` / `400.000 MHz`

## Utilization

- LUT: `644533`
- Logic LUT: `567791`
- LUTRAM: `70973`
- FF: `502742`
- RAMB36: `595`
- RAMB18: `78`
- URAM: `287`
- DSP: `743`

## Worst Path

- Slack: `-0.973 ns`
- Source: `block_0/do_patch_embed_U0/step2_mac_replace_shift_U0/mac_muladd_8s_8s_21s_21_4_1_U380/PATCH_EMBED_mac_muladd_8s_8s_21s_21_4_1_DSP48_1_U/p_reg_reg/DSP58C<0>_INST/DSP_OUTPUT58_INST/CLK`
- Destination: `block_0/do_patch_embed_U0/step2_mac_replace_shift_U0/mac_muladd_8s_8s_21s_21_4_1_U380/PATCH_EMBED_mac_muladd_8s_8s_21s_21_4_1_DSP48_1_U/p_reg_reg/DSP58C<0>_INST/DSP_OUTPUT58_INST/ALU_OUT[15]`

## Handoff Bundle

- OOC impl reports copied: `15` files
- Spinal export copied: `343` `.dat` files + `BlockSequence` RTL
- Package status: `packaged IP emitted under bundle/package/ip`
- Direct PYNQ overlay ready: `No`
- Full-board export status: `prepared`
- Full-board export Tcl: `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/results/full_board_export/run_full_board_export.tcl`
- XSA present: `False`
- HWH present: `False`
- Device image: `none`

## Notes

- `step4-ooc` is an accelerator-only OOC implementation flow, not a full board bitstream flow.
- This bundle organizes the routed accelerator checkpoint/netlist/reports plus the Spinal export payload for board-side integration work.
- `results/full_board_export/` now contains the packaged-IP handoff, interface manifest, integration Tcl, and board-export run logs/outputs.
- Versal board exports may emit `.pdi` instead of `.bit`; step5 records whichever device image was actually produced.
