# ZCU102 Step4 Bitstream Status

Date: 2026-04-08
Experiment root: `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream`
Flow: `workspace/flow/scripts/run_zynqmp_readme_flow.sh step4-bitstream zcu102`

## Summary

- The host-identity-mitigated Docker environment no longer fails immediately at `launch_runs synth_1`.
- The ZynqMP block design and wrapper generation completed successfully.
- OOC synthesis for the board IP blocks completed and produced `.dcp` and `*_utilization_synth.rpt` files.
- The accelerator OOC synthesis `hgpipe_bd_hgpipe_accel_0_0_synth_1` completed with:
  - `0 errors`
  - `0 critical warnings`
  - `2480 warnings`
- At report time the parent Vivado process was still running and one worker thread was consuming `~99.9%` CPU, so the run is treated as in progress rather than stalled.

## Completed Outputs

- Generated block-level outputs:
  - `hgpipe_bd_proc_sys_reset_0_0.dcp`
  - `hgpipe_bd_proc_sys_reset_0_0_utilization_synth.rpt`
  - `hgpipe_bd_smartconnect_ctrl_0.dcp`
  - `hgpipe_bd_smartconnect_ctrl_0_utilization_synth.rpt`
  - `hgpipe_bd_smartconnect_mem_0.dcp`
  - `hgpipe_bd_smartconnect_mem_0_utilization_synth.rpt`
  - `hgpipe_bd_axi_dma_0_0.dcp`
  - `hgpipe_bd_axi_dma_0_0_utilization_synth.rpt`
  - `hgpipe_bd_zynq_ultra_ps_e_0_0.dcp`
  - `hgpipe_bd_zynq_ultra_ps_e_0_0_utilization_synth.rpt`

## Current Phase

- `hgpipe_bd_hgpipe_accel_0_0_synth_1/runme.log` reached:
  - `Finished Writing Synthesis Report`
  - `Synthesis finished with 0 errors, 0 critical warnings and 2480 warnings.`
- `impl_1` has not been created yet.
- Top-level `synth_1` exists as a run directory but does not yet expose a `runme.log`.

## Timing Snapshot

- Last accelerator synth log update:
  - `2026-04-08 21:15:12 +0900`
- Observation time:
  - `2026-04-08 21:30:20 +0900`
- Runtime state at observation:
  - Vivado worker thread `2456` active at `99.9%` CPU

## Key Files

- Wrapper log:
  - `workspace/artifacts/logs/docker-zcu102-step4-bitstream-hostid-retry.log`
- Plan:
  - `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/bitstream_plan.json`
- Accelerator synth log:
  - `workspace/artifacts/build/zcu102/full_deit_tiny_fit/deit_tiny_baseline/zcu102_full_deit_tiny_fit/bitstream/project/zcu102_full_deit_tiny_fit_bitstream.runs/hgpipe_bd_hgpipe_accel_0_0_synth_1/runme.log`

## Current Interpretation

- This run has progressed significantly beyond the previous immediate crash profile.
- The remaining work is now inside later Vivado project execution rather than early host-info or WebTalk failure.
- Because the worker thread remains active, the correct interpretation is "still running" rather than "hung".
