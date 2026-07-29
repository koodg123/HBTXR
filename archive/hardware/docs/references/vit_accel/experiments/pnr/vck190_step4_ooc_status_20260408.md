# VCK190 Step4 OOC PNR Status

Date: 2026-04-08
Experiment: `configs/experiments/vck190_full_deit_tiny_fit.json`
Flow: `workspace/flow/scripts/run_vck190_readme_flow.sh step4-ooc`
Build root: `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl`

## Summary

- Host-identity mitigation removed the previous immediate Vivado crash after license checkout.
- The run advanced through `synth_design` and completed synthesis.
- The run did not advance to `opt_design`, `place_design`, or `route_design`.
- At observation time the Vivado process tree was still present, but all threads were sleeping and instantaneous CPU usage was `0.0%`.
- No new output was written after `2026-04-08 19:30:07 +0900`, so the run is treated as stalled after synthesis.

## Evidence

- Last log update: `2026-04-08 19:30:07 +0900`
- Observation time: `2026-04-08 20:58:08 +0900`
- Idle window at report time: about `1h 28m`
- `impl.log` last successful milestone:
  - `Finished Writing Synthesis Report`
  - `Synthesis finished with 0 errors, 0 critical warnings and 1735 warnings.`

## Generated Results

- Synthesis completed:
  - `impl.log`
  - `vivado.log`
- Not generated:
  - `post_synth_utilization.rpt`
  - `post_synth_timing_summary.rpt`
  - `post_place.dcp`
  - `post_place_utilization.rpt`
  - `post_place_timing_summary.rpt`
  - `post_route.dcp`
  - `utilization.rpt`
  - `timing_summary.rpt`

## Current Interpretation

- This is no longer the earlier `license checkout -> realloc()` failure.
- The host-identity and WebTalk mitigations improved the run far enough to finish synthesis.
- The remaining issue appears to be a Vivado stall or deadlock between synthesis completion and the next implementation Tcl step.

## Key Files

- Wrapper log:
  - `workspace/artifacts/logs/docker-vck190-step4-ooc-hostid-retry.log`
- Vivado log:
  - `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/vivado.log`
- Implementation log:
  - `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/impl.log`
- Plan:
  - `workspace/artifacts/build/vck190/full_deit_tiny_fit/deit_tiny_baseline/vck190_full_deit_tiny_fit/impl/impl_plan.json`

## Next Action

- Stop the stalled `vck190 step4-ooc` process tree.
- Reuse the same host-identity-mitigated Docker environment for `zcu102 step4-bitstream`.
- Save an equivalent PNR report for `zcu102`.
