> **작성** 2026-06-16 · **갱신** 2026-07-29
> **상태** frozen — 완료된 계획. 결과는 experiments/2026-06-16-third-goal-audit/
> **소유** hardware

# HGTXR Third-Goal Validation

Date: 2026-06-16

Selected Path Execution Audit: current third-goal plan preserves Path 1 = A2 then A1, Path 2 = C, and E pending while final signoff remains blocked only on external inputs.

## Validation Policy
- Documentation updates require Markdown/JSON syntax checks and no trailing whitespace.
- Sweep/config updates require parseable YAML or at least static grep inspection when PyYAML is unavailable.
- HLS changes require C simulation before csynth.
- Resource claims require HLS synthesis reports.
- Final signoff requires routed timing/power and board-smoke JSON.

## 2026-07-01 Patch32 MLPToken3/DenseToken4 Combined CSynth Recheck
- Re-ran combined CSim for `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`; Search and Track both passed vector comparison, runtime-state, TLAST, and strict prefetch-immediate checks.
- Combined CSynth completed for project `hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok3_dtok4_300_mem16_no_board`.
- CSynth evidence: top latency `99,928-15,280,489 cycles`, top interval `99,929-15,280,490 cycles`, estimated clock `2.777 ns`, and exact-300MHz max latency `50.935 ms`.
- Resource evidence: BRAM_18K `347/624 = 55%`, DSP `1,354/1,728 = 78%`, FF `175,489/460,800 = 38%`, LUT `320,682/230,400 = 139%`, URAM `76/96 = 79%`.
- Validation judgment: functionally valid but not promotable for Vivado route as-is. The current physical baseline remains `patch32_dtok4`; AQ2 remains the current fastest documented HLS Search/Track point, with board histogram and measured DMA/power still pending.

## 2026-07-01 PatchPar64 Search-only DSE Validation
- Added and validated profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
- CSim evidence: Search runtime state `0`, output count `6`, TLAST true, vector comparison pass, strict prefetch trace `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- CSynth evidence: target clock `3.333 ns`, estimated clock `2.789 ns`, top latency `1,296,943-1,296,963 cycles`, and top interval `1,296,944-1,296,964 cycles`.
- Exact-300MHz latency: Search max latency `4.323210 ms`; Search max initial interval `4.323213 ms`.
- Major path evidence: `hgtxr_axis_read_frame` `16,386 cycles`, `hgtxr_conv_patch_embedding` `270,340 cycles`, `hgtxr_global_buffer_load` `12,292 cycles`, `hgtxr_e2e_controller_run` `972,938-972,958 cycles`, `hgtxr_e2e_mlp_head` `24,970 cycles`.
- Resource evidence: BRAM_18K `513/624 = 82%`, DSP `2,408/1,728 = 139%`, FF `179,024/460,800 = 38%`, LUT `358,606/230,400 = 155%`, URAM `28/96 = 29%`.
- Validation judgment: rejected. PatchPar64 does not reduce the Search gap; it regresses versus `hpar8_acc24` because the patch embedding loop still has memory-port II pressure and Conv/Patch latency rises by `12,288 cycles`.

## 2026-07-01 HPar16 Search-only DSE Validation
- Added and validated profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
- CSim evidence: Search runtime state `0`, output count `6`, TLAST true, vector comparison pass, strict prefetch trace `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- CSynth evidence: target clock `3.333 ns`, estimated clock `2.795 ns`, top latency `2,229,295-2,229,315 cycles`, and top interval `2,229,296-2,229,316 cycles`.
- Exact-300MHz latency: Search max latency `7.431050 ms`; Search max initial interval `7.431053 ms`.
- Major path evidence: `hgtxr_axis_read_frame` `16,386 cycles`, `hgtxr_conv_patch_embedding` `258,052 cycles`, `hgtxr_global_buffer_load` `12,292 cycles`, `hgtxr_e2e_controller_run` `1,917,578-1,917,598 cycles`, `hgtxr_e2e_mlp_head` `24,970 cycles`.
- Controller evidence: dispatcher prefetch `6,928-6,933 cycles`, shared attention `91,652 cycles`, shared MLP `380,806 cycles`, dispatch loop `27,724-27,744 cycles`, body loop `1,889,852 cycles`.
- Scheduler/root-cause evidence: Vitis HLS reports `kGeluRom` as `ROM_1P_LUTRAM`; the MLP activation loop `VITIS_LOOP_3279_18_VITIS_LOOP_3281_19` has final II `64` due to limited memory ports on `kGeluRom`.
- Resource evidence: BRAM_18K `513/624 = 82%`, DSP `3,370/1,728 = 195%`, FF `209,907/460,800 = 45%`, LUT `401,685/230,400 = 174%`, URAM `28/96 = 29%`.
- Validation judgment: rejected. Direct hpar16 is functionally valid but slower than `hpar8_acc24` and much farther outside ZCU104 DSP/LUT capacity. The evidence points to GELU ROM replication/partitioning and shared/time-multiplexed W2 scheduling as the next Search optimization target.

## 2026-07-01 AQ2 Requested Search/Track Metrics Refresh
- Created `hardware/docs/track/AQ2_REQUESTED_SEARCH_TRACK_METRICS_2026_07_01.md` as the current request-specific AQ2 metric report.
- Re-read AQ2 Search/Track HLS CSynth reports for DMA ideal/effective bandwidth, latency min/max, HLS-envelope mean/median/P95/P99, initiation interval, GOPS, AXIS width, HLS major-block latency, and HLS major-block resource attribution.
- Re-read AQ2 Vivado implemented/routed reports for physical ZCU104 utilization, timing, route status, and vector-less power hierarchy.
- Current AQ2 Search: latency `1,709,763-1,709,783 cycles` / `5.699210-5.699277 ms` at 300 MHz, initiation interval `1,709,784 cycles`, throughput `43.05 GOPS`, implemented power `7.032 W`.
- Current AQ2 Track: latency `268,710 cycles` / `0.895700 ms` at 300 MHz, initiation interval `268,711 cycles`, throughput `33.81 GOPS`, implemented power `6.777 W`.
- Evidence boundary: AQ2 still lacks board-runtime sample histograms, board DMA counter data, external DDR/sensor I/O rail power, SAIF/VCD activity power, and internal Conv/ATTN/MLP/Head dynamic-power split inside the E2E IP.

## Current 300 MHz Experiment Acceptance Rule
- Active PL clock target: `300 MHz`, implemented as `clk_pl_0` period `3.333 ns`.
- User-approved experiment continuation threshold: routed WNS must be at least `-0.500 ns`.
- Official Vivado timing signoff remains stricter: routed WNS must be at least `0.000 ns`.
- Current active combined runtime candidate `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16` passes the experiment threshold with post-route physopt WNS `-0.236 ns`, route errors `0`, hold clean, and bitgen pass.
- Earlier `compute_300` PostRoutePhys run also passed the experiment threshold with WNS `-0.402 ns`, but neither negative-WNS run is official timing-clean signoff.

## 2026-06-29 300 MHz Prefetch-All4 Goal Status
- Generated current status artifacts: `hardware/generated/signoff/prefetchall4_300_goal_status_2026_06_29.json` and `.md`.
- Current status is `experiment-pass-missing-board-rtlcosim`: the active 300 MHz routed run fits ZCU104 and passes the user-approved WNS floor, but official timing, RTL cosim, and board measurements are not complete.
- Generated structural objective-contract artifacts: `hardware/generated/signoff/prefetchall4_300_contract_audit_2026_06_29.json` and `.md`.
- Structural contract audit status is `pass`, `16/16` checks: runtime Search/Track scheduler, Frame/Event Conv modules, on-chip ROM parameter classifier, Head ROM path, Track Transformer ROM path, omitted weight AXI, three nonlinear ROM/LUT paths, Search dispatcher URAM prefetch, prefetch-all4 trace, strict immediate-start gate, full Transformer RTL modules, CSim Search/Track expected outputs, and 300 MHz csynth evidence.
- Routed clock/timing: `clk_pl_0 = 300.030 MHz`, period `3.333 ns`, WNS `-0.236 ns`, TNS `-718.875 ns`, WHS `0.001 ns`, setup failing endpoints `6814`.
- Routed utilization: CLB LUT `73133/230400 = 31.74%`, CLB registers `60949/460800 = 13.23%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1723/1728 = 99.71%`.
- Vivado vectorless power: total on-chip `8.364 W`, dynamic `7.626 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`; hierarchy dynamic buckets include IP `4.706 W`, PS `2.674 W`, AXI mem `0.188 W`, DMA in/out `0.007/0.025 W`.
- HLS top latency envelope remains `86.951 us` to `15.993 ms`; this is not board p95/p99 and not a mode-specific measured distribution.
- Board gate status remains `missing` because canonical Search/Track board JSONs are absent.
- The `axis-par32-rom-compute-300-*` and `axis-par32-prefetchall4-300-*` PYNQ validator presets now enforce accelerator latency targets: Search `<= 4.0 ms`, Track `<= 1.0 ms`.
- Added a measured interleaved Search `10%` / Track `90%` PYNQ runner: `hardware/pynq/hgtxr/run_e2e_axis_dma_hybrid_smoke.py`. It executes one Search invocation followed by nine Track invocations per cycle, validates per-mode runtime/output vectors, enforces Search `<=4.0 ms` and Track `<=1.0 ms` accelerator latency, and reports measured hybrid p95/p99 latency, throughput, DMA bandwidth, and invocation distribution.
- Generated and validated the hybrid transfer bundle: `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle.tar.gz`, sha256 `a902fb80e4e453d625e6f245fdc6824da82fda9abca1452fb3ef83724714024e`, validation status `pass`.
- Generated hybrid ZCU104 dry-run plan: `hardware/generated/signoff/zcu104_par32_prefetchall4_300_hybrid_10_90_smoke_remote_run_2026_06_10.json` and `.md`, status `dry-run`, errors `0`.
- `hardware/tools/write_prefetchall4_300_goal_status.py` now includes the contract audit result, reads the canonical Search/Track board JSON paths plus `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json`, and prefers measured interleaved hybrid data when that JSON is present. Without the hybrid JSON, the report keeps hybrid values as missing or synthetic from per-mode samples.

## 2026-06-29 DSP/BRAM Pressure Relief Successor
- Added successor profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16`; this does not replace the active `prefetchall4_300` routed baseline.
- The profile keeps the 300 MHz target and the active prefetch-all4 scheduler, enables `HGTXR_E2E_NONLINEAR_ROM_LUTRAM=1`, and maps two tail core lanes through fabric multiplication with `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=2`, `HGTXR_E2E_CORE_LANE_CT_SWITCH=1`, and `HGTXR_E2E_FABRIC_MUL_LATENCY=3`.
- Nonlinear ROM placement is now selectable. Default builds keep `kGeluRom`, `kExpRom`, and `kRsqrtRom` on BRAM; the successor binds them to LUTRAM for small-ROM pressure relief.
- CSim pass: Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track output `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, strict prefetch trace `violations=0`, `not_ready=0`, `immediate_gaps=0`, and `E2E AXIS vector comparison passed`.
- CSynth pass: target clock `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`. Top latency is min `26,088` cycles / `86.951 us`, max `4,810,678` cycles / `16.034 ms`, compared with active `prefetchall4_300` max `4,798,390` cycles / `15.993 ms`.
- HLS resource estimate changed from active `prefetchall4_300` to successor as follows: BRAM_18K `386 -> 384`, DSP `1848 -> 1838`, FF `242,912 -> 243,144`, LUT `400,236 -> 402,640`, URAM `76 -> 76`.
- RTL evidence confirms at least the GeLU nonlinear table moved to distributed ROM: `hgtxr_e2e_axis_top_hgtxr_e2e_mlp_unit_0_s_kGeluRom_ROM_1P_LUTRAM_1R.v`. The attention probability scratch remains LUTRAM, while large QKV/MLP weight caches remain BRAM/URAM because they are not small replicated tables.
- Vivado package/route/bitgen completed for the successor at requested PL `300 MHz`. Final implemented clock is `clk_pl_0 = 300.030 MHz`; route errors are `0`; fully routed nets are `176,835/176,835`; `.bit` and `.hwh` were exported under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16_overlay/`.
- Implemented resource delta versus active `prefetchall4_300`: CLB LUT `73,133 -> 72,960`, CLB registers `60,949 -> 64,095`, Block RAM Tile `216 -> 215`, URAM `76 -> 76`, DSP `1,723 -> 1,669`. This confirms the physical DSP pressure relief: DSP utilization drops from `99.71%` to `96.59%`.
- Implemented LUTRAM evidence: top `LUT as Memory = 1,577/101,760 = 1.55%`, and hierarchical utilization contains `kGeluRom_U` modules mapped as `kGeluRom_ROM_1P_LUTRAM_1R` with zero BRAM/DSP.
- Implemented vectorless power estimate: total on-chip `8.194 W`, dynamic `7.457 W`, device static `0.737 W`, PL static `0.632 W`; DSP dynamic bucket is `1.279 W`.
- Timing result: route and bitgen pass, but the successor does not meet the user-approved continuation floor `WNS >= -0.500 ns`. Final post-route physopt timing is WNS `-0.588 ns`, TNS `-2779.720 ns`, WHS `0.001 ns`, THS `0.000 ns`, with `13,250` setup-failing endpoints. The worst path is `mlp_unit_0` pipeline logic into a controller DSP output path.
- Current judgment: the requested DSP/BRAM pressure relief is physically confirmed, but this successor is not promotable over active `prefetchall4_300` because timing regresses from active WNS `-0.236 ns` to `-0.588 ns`. The next closure step is explicit DSP input/output pipeline staging for the remaining controller/MLP multiply paths and selective relaxation of broad OOC `DONT_TOUCH`, not further small-ROM LUTRAM movement.

## 2026-06-29 Aggressive DSP/LUTRAM Pressure Relief Probe
- Added selectable LUTRAM bindings for the small/high-bank-count runtime buffers: frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`. Defaults remain unchanged; the new macros are enabled only by explicit profiles.
- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
- Profile intent: keep 300 MHz, prefetch-all4, nonlinear LUT ROM, and DSP multiply latency `4`; move `16/32` dense core tail lanes from DSP to fabric; move small banked token/norm/Q/K/V/attention buffers from BRAM/URAM toward LUTRAM. The deeper `hidden` buffer remains BRAM/URAM because it is not a small high-bank-count memory.
- CSim pass: Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track output `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, strict prefetch trace `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- CSynth pass: target clock `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`; latency min `26,087` cycles / `86.948 us`, max `4,807,861` cycles / `16.025 ms`; resources BRAM_18K `336`, DSP `1,698`, FF `247,663`, LUT `461,054`, URAM `28`.
- HLS resource delta versus `lutrom_tail2_dsppipe4`: BRAM_18K `384 -> 336`, DSP `1,838 -> 1,698`, FF `244,988 -> 247,663`, LUT `402,794 -> 461,054`, URAM `76 -> 28`. This confirms the intended DSP and memory-pressure movement at HLS level.
- Vivado route/post-route physopt/bitgen completed. Route errors are `0`; fully routed nets are `201,673/201,673`; `.bit` and `.hwh` were exported under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_overlay/`.
- Implemented utilization: CLB LUT `113,354/230,400 = 49.20%`, LUT as Distributed RAM `35,024`, CLB registers `64,368/460,800 = 13.97%`, Block RAM Tile `184/312 = 58.97%`, URAM `28/96 = 29.17%`, DSP `1,573/1,728 = 91.03%`.
- Compared with `lutrom_tail2_dsppipe4`, implemented resources change as follows: DSP `1,713 -> 1,573`, Block RAM Tile `201 -> 184`, URAM `76 -> 28`, CLB LUT `71,232 -> 113,354`, LUT as Distributed RAM `1,232 -> 35,024`. This is the strongest physical DSP/BRAM/URAM relief so far, paid for with much higher LUTRAM/LUT pressure.
- Final post-route physopt timing is WNS `-0.500 ns`, TNS `-6623.548 ns`, WHS `0.006 ns`, THS `0.000 ns`. This exactly meets the user-approved continuation floor `WNS >= -0.500 ns` but remains negative-WNS and therefore is not official clean Vivado timing signoff.
- Implemented vectorless power: total on-chip `8.811 W`, dynamic `8.074 W`, device static `0.736 W`; major dynamic buckets include clocks `0.695 W`, CLB logic `1.239 W`, signals `2.062 W`, Block RAM `0.211 W`, URAM `0.099 W`, DSPs `1.098 W`, and PS8 `2.671 W`.
- Current judgment: `tail16_lutbuf_dsppipe4` is valid as an aggressive resource-pressure relief point, not as the default promotion candidate. It leaves far more DSP headroom than `tail2_dsppipe4` (`155` spare DSPs instead of `15`) and reduces BRAM/URAM substantially, but increases distributed RAM/fanout and TNS. Keep `tail2_dsppipe4` as the balanced pressure-relief candidate; use `tail16_lutbuf_dsppipe4` when DSP/BRAM/URAM pressure is the limiting constraint.

### DSP Pipeline Latency Bound Check
- Tried a narrower timing-cleanup candidate that kept the `tail2` fabric-lane split but raised `HGTXR_E2E_DSP_MUL_LATENCY` from `4` to `6`.
- CSim passed for Search/Track expected vectors and strict prefetch trace, but CSynth failed during source synthesis.
- HLS error: `Latency value 6 is out of range, valid value is [0, 4]` for the bound DSP multiply in `hgtxr_e2e_dsp_mul`.
- Decision: reject this candidate and do not keep it as a selectable profile. Further timing cleanup must use explicit register/pipeline restructuring around DSP inputs/outputs or narrower preservation constraints, not a bind-op latency above `4`.

## 2026-06-29 Profile-Parametric Direct RTL Probe Tooling
- Updated `hardware/tools/check_hls_xsimk_direct_probe.py` and `hardware/tools/check_hls_xsimk_progress_probe.py` so direct XSIMK probes are no longer hard-coded to the active `prefetchall4_300` profile.
- New arguments: `--profile`, `--search-tail`, and `--track-tail`; non-default profiles get profile-specific signoff output names under `hardware/generated/signoff/`.
- This makes the RTL same-input/same-output evidence path reusable for `lutrom_tail2_dsppipe4`, `tail16_lutbuf_dsppipe4`, or future pressure-relief profiles once their HLS `sim/verilog` snapshots exist.
- Smoke validation on the existing active snapshot passed as a bounded direct-kernel timeout:
  - Command: `python3 hardware/tools/check_hls_xsimk_direct_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16 --timeout-s 5 --json-out /tmp/hgtxr_direct_probe_param_smoke.json --markdown-out /tmp/hgtxr_direct_probe_param_smoke.md`.
  - Result: status `direct-kernel-timeout`, `xsimk_exists=true`, `entered_kernel=true`, `exec_run_complete=true`, `exec_continue_started=true`, progress banner seen, first RTL progress sample `0/2 [0.00%] @ 109000`.
- Missing-snapshot validation on `tail16_lutbuf_dsppipe4` completed with status `missing-sim-root` and wrote `/tmp/hgtxr_progress_probe_tail16_missing.{json,md}`. This confirms the tool records a bounded evidence state instead of crashing when a pressure-relief profile has routed artifacts but no HLS `sim/verilog` snapshot yet.
- Evidence boundary: this validates the generalized probe path, not RTL output signoff. Full Search/Track RTL output equality still requires a long enough direct XSIMK run or a repaired standard XSIM wrapper path.

## 2026-06-29 Tail2 DSP-Pipeline RTL Snapshot/Direct Probe
- Ran HLS cosim for `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16`.
- C TB passed before RTL launch: Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track output `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, and `E2E AXIS vector comparison passed`.
- Strict Search prefetch trace remained clean: `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`; prefetched Search blocks started immediately after the previous block ended.
- XELAB built the Verilog snapshot at `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`.
- Standard HLS Verilog cosim still failed at the same XSIM wrapper command, `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}`. `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` with latency `NA`.
- Direct XSIMK smoke artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_direct_probe_2026_06_29.{json,md}`. Status is `direct-kernel-timeout`, with kernel entry, `-exec-run`, `-exec-continue`, progress banner, and first RTL progress `0/2 [0.00%] @ 109000`.
- Instrumented progress artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_progress_probe_2026_06_29.{json,md}`. Status is `progress-timeout`; rebuilt snapshot passed, direct XSIMK advanced from `0/2 [0.00%] @ 109000` to `0/2 [8.31%] @ 1336125000` in the bounded `120 s` run.
- Evidence boundary: this proves the balanced pressure-relief profile now has an RTL snapshot and the direct simulator advances inside transaction 0. It still does not prove full RTL Search/Track output equality because neither expected RTL output vector was reached before timeout.

## 2026-06-29 300 MHz Combined Runtime Cosim Diagnosis
- Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`.
- Timing rule used for this experiment remains PL `300 MHz`, routed WNS continuation floor `-0.500 ns`, official clean signoff floor `0.000 ns`.
- `config_cosim -disable_deadlock_detection` was applied and recorded in `sim/report/cosim_options.xml`, but Vitis HLS still generated an XSIM launch path containing `-autoloadwcfg`.
- C testbench and scheduler evidence are clean: Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track output `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, `E2E AXIS vector comparison passed`, Search scheduler `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`, Track scheduler `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- XELAB built the Verilog snapshot `hgtxr_e2e_axis_top`, but Verilog cosim still reports `Fail` and latency `NA` because XSIM fails during launch at `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}`.
- Generated diagnosis artifacts: `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`; current status is `blocked-xsim-launch`.
- Manual no-autoload / explicit-view probes did not produce a valid RTL simulation pass: hiding the generated `.wcfg` kept the same launch exception, explicit `-view hgtxr_e2e_axis_top_dataflow_ana.wcfg` also failed, and direct `xsimk` invocation elaborated then exited without producing RTL TV output.
- A minimal standalone Verilog XSIM snapshot smoke was added as `hardware/tools/check_xsim_snapshot_smoke.py`. It builds a trivial `tb_snapshot` successfully, but `xsim tb_snapshot -R` fails with the same `unexpected exception when evaluating tcl command` during snapshot launch and never prints `XSIM_SMOKE_PASS`.
- Generated XSIM environment evidence: `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.json` and `.md`; status is `blocked-xsim-runtime`. Current judgment is that RTL functional signoff is blocked by this host XSIM runtime issue, not by a proven HGTXR RTL output mismatch.

## Checks Run For Reference Integration
- `python3 -m json.tool analysis/vit-accel/manifest.json`
- `git diff --check -- analysis/vit-accel`
- `rg -l "## 1\\. 기존 방법의 문제점" analysis/vit-accel/papers | wc -l`
- `rg -l "## 8\\. 실험 옵션 / Ablation 축" analysis/vit-accel/papers | wc -l`

## Remaining Validation
- C3b physical PYNQ smoke is not captured.
- XR-VITs exact source or approved replacement policy remains unresolved in broader signoff.
- VREF P0 experiments need SW/HW exact-match evidence before hardware promotion.
- HG-PIPE operator audit is local sampled/reference-vector equivalence, not formal exhaustive proof over all possible inputs.

## Runtime Full AXI Weight-Path Validation Added
- `par32_runtime_full_axi_mem16` restores the learned non-fastpath E2E path for runtime Search/Track: learned conv patch embedding, weighted controller ATTN/MLP, learned MLP head, and `gmem_e2e_weights` AXI master.
- CSim pass: Search `runtime_state=0`, Track `runtime_state=1`, 6 output state words, TLAST correct, `CSim done with 0 errors`.
- CSynth pass/package pass: `generated/hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` and `impl/ip/component.xml`.
- Full AXI HLS latency does not meet the Search/Track targets: Track-bound min `627,117` cycles / `3.136 ms`; Search-bound max `5,101,975` cycles / `25.510 ms`.
- Full AXI HLS resource estimate: BRAM_18K `332/624`, DSP `1,148/1,728`, FF `86,123/460,800`, LUT `193,627/230,400`, URAM `64/96`.
- Interface check: packaged `component.xml` includes both `m_axi_gmem_e2e_weights` and `m_axi_gmem_e2e_runtime`.
- Evidence boundary: fastpath runtime-mode latency remains separate from full AXI learned-model latency. The fastpath meets 4 ms / 1 ms, but the restored learned AXI path currently fails both targets and needs weight-path optimization before routed power/board signoff.

## 2026-06-28 Full Learned Runtime ROM Dispatch Validation Added
- `par32_runtime_rom_dispatch_300_mem16` restores the non-fastpath learned Transformer path with shared packed ROM/AXI weight loading, Q/K/V cache, output-projection cache, MLP W1/W2 caches, LayerNorm parameter cache, on-chip nonlinear ROM, and runtime Search/Track scheduling.
- CSim pass: Search outputs `[-221, -237, -237, -237, -237, -237]`, `runtime_state=0`, failures `0`; Track outputs `[-219, -235, -235, -235, -235, -235]`, `runtime_state=1`; `E2E AXIS vector comparison passed`; `CSim done with 0 errors`.
- Par32 300 MHz CSynth pass: target clock `3.333 ns`, estimated clock `2.777 ns`; Track-bound min `605,592` cycles / `2.018 ms`; Search-bound max `4,733,900` cycles / `15.778 ms`; hybrid 10% Search / 90% Track expected latency `3.394 ms`.
- Par32 ZCU104 fit status: fail due LUT estimate `287,796/230,400 = 124%`; other estimates are BRAM_18K `440/624`, DSP `1,634/1,728`, FF `213,114/460,800`, URAM `32/96`.
- Par16 fit probe CSynth pass: target clock `3.333 ns`, estimated clock `2.924 ns`; Track-bound min `836,738` cycles / `2.789 ms`; Search-bound max `6,656,520` cycles / `22.186 ms`; LUT estimate `249,421/230,400 = 108%`, so it also fails ZCU104 fit.
- Par8 HLS fit probe added as `par8_runtime_rom_dispatch_300_mem8`: CSim pass with Search `runtime_state=0`, Track `runtime_state=1`, TLAST correct, and `CSim done with 0 errors`.
- Par8 300 MHz CSynth pass after DSP latency timing hint and low-fanout weight AXI hint: target clock `3.333 ns`, estimated clock `2.924 ns`; Track-bound min `1,588,174` cycles / `5.293 ms`; Search-bound max `12,929,562` cycles / `43.094 ms`; hybrid 10% Search / 90% Track expected latency `9.074 ms`.
- Par8 HLS resource estimate fits ZCU104 at HLS level: BRAM_18K `408/624 = 65%`, DSP `1,368/1,728 = 79%`, FF `186,572/460,800 = 40%`, LUT `222,223/230,400 = 96%`, URAM `32/96 = 33%`.
- Par8 IP package pass: `generated/hgtxr_e2e_axis_par8_runtime_rom_dispatch_300_mem8_no_board/solution_e2e_q4w8a/impl/ip/component.xml` and `impl/export.zip` exist; packaged IP exposes `m_axi_gmem_e2e_weights` and `m_axi_gmem_e2e_runtime`.
- Vivado profile `par8_runtime_rom_dispatch_300_mem8` now passes the requested `300 MHz` PL clock through the overlay builder and injects RTL-level `keep_hierarchy` / `dont_touch` preservation attributes into `38` datapath HLS Verilog module files when `HGTXR_E2E_OOC_DONT_TOUCH=datapath`.
- Par8 Vivado route/bitgen completes and preserves the full learned compute scale: routed placed utilization is CLB LUT `37,300/230,400 = 16.19%`, registers `40,008/460,800 = 8.68%`, Block RAM Tile `242/312 = 77.56%`, DSP `1,371/1,728 = 79.34%`, URAM `32/96 = 33.33%`.
- Par8 routed timing does not meet the 300 MHz constraint: `clk_pl_0 = 300.030 MHz`, WNS `-0.265 ns`, TNS `-113.997 ns`, WHS `0.010 ns`, THS `0.000 ns`; fully routed `122,577/122,577`, route errors `0`; bitgen completes.
- Par8 routed critical path remains the `gmem_e2e_weights` read FIFO/RREADY path from `fifo_rreq/full_n_reg/C` to `buff_rdata/U_fifo_mem/mem_reg_2/ENARDEN`; data path delay is `3.030 ns`, route is `2.421 ns = 79.903%`.
- Par8 routed power estimate for the full learned netlist is total on-chip `6.817 W`, dynamic `6.096 W`, device static `0.721 W`, PS static `0.103 W`, PL static `0.619 W`; confidence is Vivado vectorless `Medium`, not mode-specific or board-measured power.
- Evidence boundary: full learned physical preservation is now proven at routed resource scale, but the implementation is not timing-clean at 300 MHz and still fails Search `<=4 ms` / Track `<=1 ms` latency targets.
- Latency boundary: none of the full learned candidates meets the Search `<= 4 ms` and Track `<= 1 ms` goals yet. The earlier fastpath runtime-mode result remains separate from the full learned Transformer evidence.

## 2026-06-28 Full Learned Runtime ROM-Only Dispatch Validation Added
- `par8_runtime_rom_only_dispatch_300_mem8` adds a full learned ROM-only profile for the E2E AXIS top: `HGTXR_E2E_USE_MODE_PROFILE_FASTPATH=0`, `HGTXR_E2E_OMIT_WEIGHT_AXI=1`, `HGTXR_E2E_USE_ONCHIP_PARAM_ROM=1`, `HGTXR_E2E_USE_ONCHIP_NONLINEAR_ROM=1`, and Search/Track runtime scheduling enabled.
- Weight loads for valid model parameters now resolve to deterministic on-chip parameter ROM when weight AXI is omitted; Search dispatcher prefetch uses the shared packed-weight loader instead of direct `weights[...]` access.
- CSim pass: Search outputs `[-712, -712, -712, -712, -712, -696]`, `runtime_state=0`, failures `0`; Track outputs `[-235, -235, -235, -235, -235, -176]`, `runtime_state=1`; `E2E AXIS vector comparison passed`; `CSim done with 0 errors`.
- HLS package pass: `generated/hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_300_mem8_no_board/solution_e2e_q4w8a/impl/ip/component.xml` and `impl/export.zip` exist; packaged IP exposes `m_axi_gmem_e2e_runtime` but does not expose `m_axi_gmem_e2e_weights`.
- ROM-only 300 MHz CSynth pass: target clock `3.333 ns`, estimated clock `2.983 ns`; Track-bound min `1,600,948` cycles / `5.336 ms`; Search-bound max `12,972,958` cycles / `43.239 ms`; hybrid 10% Search / 90% Track expected latency `9.126 ms`.
- ROM-only HLS resource estimate: BRAM_18K `386/624 = 61%`, DSP `1,624/1,728 = 93%`, FF `168,413/460,800 = 36%`, LUT `222,677/230,400 = 96%`, URAM `32/96 = 33%`.
- ROM-only Vivado route/bitgen completes at requested `300.0 MHz` PL clock; block design logs skip absent `hgtxr_e2e_axis_top_0/m_axi_gmem_e2e_weights`; fully routed nets `107,269/107,269`; route errors `0`; bitgen completes.
- ROM-only routed timing still fails the 300 MHz constraint but improves over the weight-AXI version: `clk_pl_0 = 300.030 MHz`, WNS `-0.065 ns`, TNS `-2.886 ns`, WHS `0.003 ns`, THS `0.000 ns`.
- ROM-only routed critical path is now internal ROM/URAM/BRAM/DSP datapath, not the removed weight AXI read FIFO: URAM source to attention DSP input, data path delay `2.923 ns`, route `1.614 ns = 55.217%`.
- ROM-only routed placed utilization: CLB LUT `24,112/230,400 = 10.47%`, registers `26,348/460,800 = 5.72%`, Block RAM Tile `238/312 = 76.28%`, DSP `1,627/1,728 = 94.16%`, URAM `32/96 = 33.33%`.
- ROM-only routed power estimate: total on-chip `5.936 W`, dynamic `5.221 W`, device static `0.715 W`, PS static `0.101 W`, PL static `0.613 W`; confidence is Vivado vectorless `Medium`, not mode-specific or board-measured power.
- Evidence boundary: full learned ROM-only physical implementation and bitgen are proven, but the design is not timing-clean at 300 MHz and still fails Search `<= 4 ms` / Track `<= 1 ms` latency targets. Event Conv remains not separately exercised as a distinct AXIS event input path.

## 2026-06-28 Event-Conv ROM-Only DSP3 300 MHz Validation Added
- `par8_runtime_rom_only_dispatch_dsp3_300_mem8` now has distinct Frame Conv and Event Conv on-chip parameter ROM regions; Search uses Frame Conv, Track uses Event Conv.
- Weight AXI remains omitted; the packaged IP exposes `m_axi_gmem_e2e_runtime` and intentionally omits `m_axi_gmem_e2e_weights`.
- CSim pass after Event Conv integration: Search outputs `[-712, -712, -712, -712, -712, -696]`, `runtime_state=0`; Track outputs `[-235, -235, -235, -235, -235, -299]`, `runtime_state=1`; output count `6`, TLAST correct, and `CSim done with 0 errors`.
- HLS package pass: `generated/hgtxr_e2e_axis_par8_runtime_rom_only_dispatch_dsp3_300_mem8_no_board/solution_e2e_q4w8a/impl/ip/component.xml` and `impl/export.zip` exist.
- HLS estimate after Event Conv: target clock `3.333 ns`, estimated clock `3.495 ns`; Track-bound min `1,605,053` cycles, Search-bound max `12,989,355` cycles. At routed `300.030 MHz`, this is Track `5.350 ms`, Search `43.294 ms`, and hybrid 10% Search / 90% Track `9.144 ms`.
- Vivado route/bitgen pass at requested 300 MHz: `clk_pl_0 = 300.030 MHz`, WNS `0.007 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`, fully routed nets `107,641/107,641`, route errors `0`, bitgen completed.
- Routed resources: CLB LUT `24,159/230,400 = 10.49%`, registers `26,753/460,800 = 5.81%`, Block RAM Tile `238/312 = 76.28%`, DSP `1,627/1,728 = 94.16%`, URAM `32/96 = 33.33%`.
- HLS OOC resource estimate for block attribution: BRAM_18K `386/624 = 61%`, DSP `1,624/1,728 = 93%`, FF `167,547/460,800 = 36%`, LUT `224,356/230,400 = 97%`, URAM `32/96 = 33%`.
- Routed vectorless power estimate: total on-chip `6.037 W`, dynamic `5.322 W`, device static `0.715 W`, PS static `0.102 W`, PL static `0.614 W`; hierarchy dynamic includes `hgtxr_e2e_axis_top_0 = 2.408 W`, `psu = 2.673 W`, `axi_mem = 0.187 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.026 W`.
- Current goal status: full learned Transformer is physically implemented and timing-clean at 300 MHz, with Frame/Event Conv, Head, Track ROM path, Transformer ROM path, nonlinear LUT ROM, and runtime scheduler active. Latency targets still fail: Search `43.294 ms > 4 ms`, Track `5.350 ms > 1 ms`.
- Remaining evidence gaps: Search dispatcher is not yet a proven overlapped double-buffer prefetch engine; Event Conv is a derived event-delta path, not a separate external event AXIS stream; mode-specific power, DMA bandwidth, board p95/p99 latency, and sensor I/O power require additional RTL/board experiments.

## 2026-06-28 PAR32 Norm-Stage Objective Timing Probe Added
- Baseline policy remains unchanged: `par32_dsp_mixed_stream_mem16` is the physical/timing baseline, `par32_runtime_full_axi_mem16` is the full learned AXI functional baseline, and ROM-only scheduler/dispatcher builds are objective-path probes.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16` stages LayerNorm writes through a local LUTRAM row buffer before writing `gb.norm`, aiming to break the LayerNorm-to-URAM write path.
- CSim pass: Search outputs `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track outputs `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, TLAST correct, and `CSim done with 0 errors`.
- CSynth pass: target clock `3.333 ns`, estimated clock `2.777 ns`; top min `26,088` cycles / `86.951 us`, max `4,908,208` cycles / `16.359 ms`; `hgtxr_e2e_controller_run` max `4,829,897` cycles / `16.098 ms`.
- CSynth resource estimate: BRAM_18K `386/624`, DSP `1,976/1,728`, FF `249,856/460,800`, LUT `399,142/230,400`, URAM `64/96`. This remains an OOC estimate; final fit must use Vivado placed resources.
- Vivado route and bitgen completed with route errors `0` and fully routed nets `201,593/201,593`, but 300 MHz timing fails: `clk_pl_0 = 300.030 MHz`, WNS `-0.644 ns`, TNS `-3987.069 ns`, WHS `0.002 ns`, THS `0.000 ns`, failing setup endpoints `16,238/303,146`.
- Placed utilization: CLB LUT `89,035/230,400 = 38.64%`, CLB registers `59,380/460,800 = 12.89%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%`.
- Routed vectorless power estimate: total on-chip `8.615 W`, dynamic `7.876 W`, device static `0.739 W`, PS static `0.105 W`, PL static `0.634 W`; hierarchy/component buckets include clocks `0.556 W`, CLB logic `1.245 W`, signals `1.519 W`, Block RAM `0.247 W`, URAM `0.156 W`, DSPs `1.484 W`, PS8 `2.671 W`.
- Critical setup paths remain dominated by output-projection/MLP DSP input paths and `gb_dispatch_prefetch` URAM read paths, with route delay up to `82.775%` on the worst reported path. Vivado DRC also reports many DSP MREG/PREG pipeline warnings.
- Current judgment: `normstage` is not an improvement over the prior `normuram` timing probe (`-0.644 ns` vs `-0.496 ns` WNS). The next closure step is not to abandon the PAR32 baselines; it is to keep them as comparison rows while adding explicit DSP pipeline stages and reviewing datapath-wide `DONT_TOUCH`.

## 2026-06-28 Dispatcher Prefix-Prefetch Full Learned Vivado Rerun

- `par8_runtime_rom_only_dispatch_dsp3_300_mem8` now connects the Search dispatcher to actual weight-cache fills through two `16`-word ping-pong prefetch banks.
- Prefetch consumers include LayerNorm parameter caches, Q/K/V caches, output-projection cache, and MLP W1/W2 caches through `hgtxr_e2e_load_weight_word_prefetched`.
- This proves a real prefix-prefetch/cache-load connection. It is not yet a full-block double-buffer prefetch proof.
- CSim pass: Search outputs `[-712, -712, -712, -712, -712, -676]`, `runtime_state=0`; Track outputs `[-235, -235, -235, -235, -235, -291]`, `runtime_state=1`; output count `6`, TLAST correct, and `CSim done with 0 errors`.
- HLS package pass: target clock `3.333 ns`, estimated clock `2.924 ns`, top max latency `13,008,010` cycles / `43.356 ms`.
- The current HLS top min latency `26,190` cycles is not accepted as Track latency because it is a variable-bound/control lower envelope, not a full Track-mode measurement.
- Vivado route/bitgen pass at requested 300 MHz: `clk_pl_0 = 300.030 MHz`, WNS `0.000 ns`, TNS `0.000 ns`, WHS `0.007 ns`, THS `0.000 ns`, fully routed nets `138,804/138,804`, route errors `0`.
- Routed resources: CLB LUT `43,929/230,400 = 19.07%`, registers `43,016/460,800 = 9.34%`, Block RAM Tile `238/312 = 76.28%`, DSP `1,630/1,728 = 94.33%`, URAM `32/96 = 33.33%`.
- Routed vectorless power estimate: total on-chip `6.833 W`, dynamic `6.111 W`, device static `0.721 W`; hierarchy includes `hgtxr_e2e_axis_top_0 = 3.197 W`, `psu = 2.674 W`, `axi_mem = 0.186 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.026 W`.
- Current goal status: full learned Vivado implementation remains physically proven and timing-clean at 300 MHz, but Search latency still fails `4 ms` and the best comparable Track envelope still fails `1 ms`.

## VREF-P0 Validation Added
- `tools/write_vref_p0_pot_scale_audit.py`: checks Q4/Q8 contract, power-of-two scale macros, DSP helper/bind evidence, and URAM/LUTRAM policy flags.
- `tests/test_write_vref_p0_pot_scale_audit.py`: unittest coverage for direct audit and CLI output.
- `tools/write_vref_p0_pot_scale_sweep.py`: evaluates PoT scale candidates against reduced E2E HG-PIPE vector references without mutating HLS sources.
- `tests/test_write_vref_p0_pot_scale_sweep.py`: unittest coverage for current-scale recommendation and CLI output.
- `tools/write_vref_p0_buffer_lifetime_audit.py`: checks ME-ViT-style large-buffer URAM, small-buffer LUTRAM/BRAM, QKV cache, C3b PAR16/MEM16, DSP/LUT/URAM resource policy.
- `tests/test_write_vref_p0_buffer_lifetime_audit.py`: unittest coverage for direct audit and CLI output.
- `tools/write_c3b_protection_checklist.py`: freezes C3b baseline gates for VREF successor promotion without running HLS/Vivado.
- `tests/test_write_c3b_protection_checklist.py`: unittest coverage for direct checklist generation and CLI output.

## VREF-P0-01 Validation Added
- `generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.md`
- Result: `3` specs, `45` candidates, `0` failed candidates.
- Current PoT scale profile remains the C3b recommendation.
- `generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md`
- `softmax_input_x2` successor validation: regenerated spec/header, header sync pass, custom CSim flags pass, E2E AXIS CSim pass.
- CSim evidence markers: `E2E AXIS vector comparison passed`, `runtime_state=2 count=6 last=1 failures=0`, `CSim done with 0 errors`.
- CSynth evidence: `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth/solution_e2e_q4w8a/syn/report/csynth.xml`.
- CSynth result: estimated clock `4.069 ns`, latency `498485` cycles, `64 BRAM_18K`, `128 DSP`, `19664 FF`, `43236 LUT`, `88 URAM`.
- Resource-policy evidence: `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_csynth/solution_e2e_q4w8a/syn/report/csynth.xml`.
- Recommended variant `dsp_mixed_stream`: estimated clock `4.069 ns`, latency `498485` cycles, `144 BRAM_18K`, `128 DSP`, `19664 FF`, `43236 LUT`, `32 URAM`.
- IP package evidence: `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip/solution_e2e_q4w8a/impl/ip/component.xml` and `export.zip`.
- Vivado overlay evidence: `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit` and matching `.hwh`.
- Routed timing evidence: WNS `4.497 ns`, TNS `0.000 ns`, WHS `0.010 ns`, route errors `0`, fully routed nets `19829`.
- PYNQ successor smoke bundle evidence: `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle/BUNDLE_MANIFEST.json`.
- PYNQ successor smoke session evidence: `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.json` and `.md`.
- Successor smoke bundle validation: tar contents and manifest hashes pass; expected raw `[58, -51, 42, -28, 36, -41]`, expected `runtime_state=2`.
- Successor remote runner evidence: `generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.json` and `.md`.
- Recommended C3b projection: no HLS resource failures; routed timing passes; physical smoke remains pending.
- Resource-report caveat: wrapper-level Vivado utilization can show generated/OOC IP resources as black-box effects. HLS `csynth.xml` remains the evidence for DSP/URAM/LUT resource-policy claims.
- Additional non-current candidates are SW-valid PoT successor rows only; promotion requires their own regenerated golden headers and CSim.

## C3b Protection Validation Added
- `generated/signoff/c3b_protection_checklist_2026_06_16.md`
- Result: `30/32 pass`, `0 fail`, `2 pending`.
- Pending items are external unblockers, not static checklist failures:
  - physical C3b smoke JSON.
  - XR-VITs exact sibling or approved replacement policy.
- Successor thresholds: latency `<= 37508072`, WNS `>= 4.415 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`.

## HG-PIPE Operator Audit Added
- `tools/write_hgpipe_operator_audit.py`: validates LayerNorm, GeLU, Softmax, and Quantization implementation markers, HG-PIPE guide markers, VREF successor compile flags, CSim status, and captured LUT/ref-vector contracts.
- `generated/signoff/hgpipe_operator_audit_2026_06_16.md`: status `pass`; LayerNorm, GeLU, Softmax, and Quantization all `pass`.
- Contract evidence: `97/97` reference checks passed with `5899008` checked samples.
- This upgrades third-goal Req8 to `reflected` in current audit while retaining the non-exhaustive equivalence caveat.

## XR-VITs Gate Audit Added
- `tools/write_xr_vits_gate_audit.py`: checks the exact requested `/home/kjm26/project/PRJXR/XR-VITs` path, active replacement policy, `XR_Accel` candidate audit, and existing reference/unblock artifacts without writing policy files.
- `generated/signoff/xr_vits_gate_audit_2026_06_16.md`: status `blocked`, resolution mode `candidate-ready-needs-approval`.
- Current evidence: exact path is missing; `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel` exists and matches the candidate audit recommendation, but active replacement policy is missing.
- Req11 remains blocked externally until the exact checkout is restored or an approved replacement policy is created with explicit approval metadata.

## C3b Physical-Smoke Gate Audit Added
- `tools/write_c3b_physical_smoke_gate_audit.py`: checks canonical C3b board result JSON, canonical validation JSON, PYNQ smoke bundle, tarball, session runbook, expected output, and validator preset without running board commands.
- `generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.md`: status `blocked_missing_canonical_physical_smoke_result`.
- Current evidence: `ready_for_board=True`, `physical_smoke_pass=False`; bundle and session are `pass`, but `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` is missing.
- Expected C3b board output remains `[32, -13, 26, -6, 14, -11]` with runtime state `2`.

## Req2 Spec/Sub-Agent Gate Audit Added
- `tools/write_req2_spec_subagent_gate_audit.py`: checks plan/spec documents, `spec-kit`/`specify` PATH availability, manual Spec fallback records, Spark-first routing records, GPT5.5 fallback records, and existing conformance artifacts.
- `generated/signoff/req2_spec_subagent_gate_audit_2026_06_16.md`: status `pass-manual-spec-and-model-fallback`.
- Current evidence: `spec-kit` and `specify` are unavailable; manual Spec fallback is recorded; Spark-first and GPT5.5 fallback routing are recorded.
- This does not claim spec-kit CLI execution or unlimited Spark availability.

## 2026-06-16 Successor Validation Run
- `python3 -m py_compile tools/write_vref_p0_pot_scale_successor.py tests/test_write_vref_p0_pot_scale_successor.py`: pass.
- `python3 -m json.tool generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json`: pass.
- `python3 tools/validate_e2e_axis_vector.py --spec refs/vref_p0/e2e_axis_vector_hgpipe_math_lnq_active16_softmax_input_x2_spec.json --check-c-header hls/tb/e2e_axis_vector_vref_p0_hgpipe_lnq_active16_softmax_input_x2_golden.hpp`: pass.
- `python3 -m unittest tests.test_check_third_goal_preflight tests.test_write_vref_p0_pot_scale_successor tests.test_write_vref_p0_pot_scale_sweep tests.test_write_vref_p0_pot_scale_audit tests.test_write_vref_p0_buffer_lifetime_audit tests.test_write_c3b_protection_checklist tests.test_write_e2e_resource_policy_audit`: `34` tests passed.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode neutral --json-out /tmp/hgtxr_third_goal_preflight_2026_06_16_after_vref_route.json`: `ok=63 warn=5 fail=0`.
- `git diff --check -- docs hls refs tools tests generated/signoff vivado/scripts configs/sweeps/zcu104_cyclic_transformer_sweep.yaml README.md analysis/vit-accel`: pass.
- `LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth HGTXR_E2E_SCALE=custom HGTXR_E2E_CUSTOM_SCALE_FLAGS=<softmax_input_x2 flags> /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csynth.tcl`: pass.
- `LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_csynth HGTXR_E2E_SCALE=custom HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream HGTXR_E2E_CUSTOM_SCALE_FLAGS=<softmax_input_x2 flags> /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/run_e2e_q4w8a_csynth.tcl`: pass.
- `LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip HGTXR_E2E_SCALE=custom HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream HGTXR_E2E_CUSTOM_SCALE_FLAGS=<softmax_input_x2 flags> /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/package_e2e_axis_ip.tcl`: pass.
- `LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH /tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch -source vivado/scripts/build_e2e_axis_dma_bitstream.tcl -tclargs -project_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_overlay -bd_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_system -artifact_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream -hls_ip_repo generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip/solution_e2e_q4w8a/impl/ip`: pass.

## 2026-06-16 Successor Smoke Bundle Validation Run
- `python3 -m unittest tests.test_check_third_goal_preflight tests.test_write_vref_p0_pot_scale_successor tests.test_write_vref_p0_pot_scale_sweep tests.test_write_vref_p0_pot_scale_audit tests.test_write_vref_p0_buffer_lifetime_audit tests.test_write_c3b_protection_checklist tests.test_write_e2e_resource_policy_audit tests.test_package_e2e_axis_dma_pynq_bundle tests.test_validate_pynq_smoke_result tests.test_validate_pynq_bundle_package tests.test_import_pynq_smoke_result tests.test_prepare_zcu104_smoke_session tests.test_run_zcu104_c3b_smoke_remote`: `63` tests passed.
- `python3 tools/validate_pynq_bundle_package.py --bundle-dir generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle --tar generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz --variant vref-p0-softmax-input-x2-dsp-mixed-stream --json-out generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle_validation.json`: pass.
- `python3 tools/prepare_zcu104_smoke_session.py --bundle-dir generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle --tar generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz --variant vref-p0-softmax-input-x2-dsp-mixed-stream --preset axis-vref-p0-softmax-input-x2-dsp-mixed-stream --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.json --markdown-out generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md`: pass.
- `python3 tools/run_zcu104_c3b_smoke_remote.py --profile vref-p0-softmax-input-x2-dsp-mixed-stream --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host zcu104.local --user xilinx --json-out generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.json --markdown-out generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.md`: dry-run pass.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode neutral --json-out /tmp/hgtxr_third_goal_preflight_2026_06_16_after_vref_remote_runner.json`: `ok=76 warn=5 fail=0`.

## 2026-06-16 Successor Physical-Smoke Receiver Validation Run
- `python3 -m unittest tests.test_check_third_goal_preflight tests.test_write_vref_p0_pot_scale_successor tests.test_validate_pynq_smoke_result tests.test_import_pynq_smoke_result tests.test_run_zcu104_c3b_smoke_remote`: `49` tests passed.
- `python3 tools/write_vref_p0_pot_scale_successor.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json --markdown-out generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md`: pass.
- `python3 -m json.tool generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json`: pass.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode neutral --json-out /tmp/hgtxr_third_goal_preflight_2026_06_16_after_vref_physical_receiver.json`: `ok=76 warn=6 fail=0`.
- `git diff --check -- docs hls refs tools tests generated/signoff generated/pynq vivado/scripts configs/sweeps/zcu104_cyclic_transformer_sweep.yaml README.md analysis/vit-accel pynq/hgtxr`: pass.
- Successor package now records `physical_smoke_result.status=not_captured` until board JSON is copied back.
- `physical_smoke_available` in the recommended projection now changes to `pass` only after a valid
  `axis-vref-p0-softmax-input-x2-dsp-mixed-stream` ZCU104 smoke JSON is present.

## 2026-06-16 Final Unblock Closeout Refresh
- `python3 -m unittest tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet`: `5` tests passed.
- `python3 tools/write_final_unblock_closeout_packet.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/final_unblock_closeout_packet_2026_06_16.json --markdown-out generated/signoff/final_unblock_closeout_packet_2026_06_16.md`: ready-for-operator-unblock, blockers `2`, missing `0`.
- `python3 tools/validate_final_unblock_closeout_packet.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --packet generated/signoff/final_unblock_closeout_packet_2026_06_16.json --json-out generated/signoff/final_unblock_closeout_packet_validation_2026_06_16.json --markdown-out generated/signoff/final_unblock_closeout_packet_validation_2026_06_16.md`: pass, `33` checks, `0` fail.
- Closeout packet now records `vref_successor_gate.status=ready-for-physical-smoke` and `required_for_final_signoff=false`.

## 2026-06-16 VREF-Required Closeout Policy
- `python3 -m unittest tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet`: `9` tests passed.
- Default closeout packet regenerated: blockers `2` (`C3b AXIS/DMA physical smoke result`, `requested XR-VITs sibling`), validation pass, `34` checks.
- VREF-required closeout packet generated with `--require-vref-successor-physical-smoke`: blockers `3`, adding `VREF-P0 successor physical smoke result`, validation pass, `34` checks.
- Artifacts:
  - `generated/signoff/final_unblock_closeout_packet_2026_06_16.md`
  - `generated/signoff/final_unblock_closeout_packet_vref_required_2026_06_16.md`
  - `generated/signoff/final_unblock_closeout_packet_validation_2026_06_16.md`
  - `generated/signoff/final_unblock_closeout_packet_vref_required_validation_2026_06_16.md`

## 2026-06-16 Final Runner VREF Successor Integration
- `python3 -m unittest tests.test_run_third_goal_final_signoff`: `11` tests passed.
- Runner now supports VREF successor import, dry-run import, remote dry-run/execute, and hard-gate closeout policy forwarding.
- VREF remote smoke no longer inherits the C3b `--zcu104-remote-dir` override by default; use `--vref-successor-remote-dir` for explicit VREF remote path override.

## 2026-06-16 Current Third-Goal Audit
- `python3 -m unittest tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit`: `5` tests passed.
- `python3 -m unittest tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet tests.test_check_third_goal_preflight`: `44` tests passed.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `46`, sources `159`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `8`, partial `3`, blocked `1`.
- `python3 -m json.tool generated/signoff/third_goal_current_audit_2026_06_16.json`: pass.
- Evidence classes are recorded as `doc-reported`, `external-blocker`, `file-exists`, and `validated-by-tool`.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode neutral --json-out /tmp/hgtxr_third_goal_preflight_2026_06_16_after_current_audit.json`: `ok=76 warn=6 fail=0`.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff --json-out /tmp/hgtxr_third_goal_final_preflight_2026_06_16_after_current_audit.json`: expected blocked result, `ok=76 warn=4 fail=2`.
  Failing gates: `C3b AXIS/DMA physical smoke result` and `requested XR-VITs sibling`.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff --json-out /tmp/hgtxr_third_goal_preflight_2026_06_16_after_choice_final.json`: expected blocked result, `ok=76 warn=4 fail=2`.
  XR-VITs failure detail now reports the `XR_Accel` candidate and explicitly keeps approval required.
- New artifacts:
  - `generated/signoff/third_goal_source_audit_2026_06_16.json`
  - `generated/signoff/third_goal_source_audit_2026_06_16.md`
  - `generated/signoff/third_goal_current_audit_2026_06_16.json`
  - `generated/signoff/third_goal_current_audit_2026_06_16.md`

## 2026-06-16 C3b Import Provenance
- `python3 -m unittest tests.test_import_pynq_smoke_result tests.test_run_third_goal_final_signoff tests.test_check_c3b_board_smoke_readiness`: `21` tests passed.
- `tools/import_pynq_smoke_result.py` import reports now include `source_sha256`, `dest_is_default`, `dest_exists_before`, `dest_exists_after`, `require_paths`, and `payload_summary`.
- This does not create board smoke results. It improves provenance for a real board-produced JSON after copy-back.

## 2026-06-16 Smoke Import Destination Gate Revalidation
- `tools/import_pynq_smoke_result.py` now revalidates the destination JSON after active copy and records:
  `dest_validation.status`, `dest_validation.dest_sha256`, `dest_validation.dest_matches_source`, destination payload summary, and `would_clear_current_gate`.
- `would_clear_current_gate` is true only when import is active, destination is the default canonical path, destination validation passes, and destination SHA256 matches the source.
  Dry-runs and custom destinations remain validation-only and do not claim gate closure.
- `python3 -m unittest tests.test_import_pynq_smoke_result tests.test_run_third_goal_final_signoff tests.test_check_final_blocker_closure_readiness tests.test_validate_pynq_smoke_result`: `35` tests passed.
- `python3 -m py_compile tools/import_pynq_smoke_result.py tests/test_import_pynq_smoke_result.py tools/run_third_goal_final_signoff.py tools/check_final_blocker_closure_readiness.py tools/validate_pynq_smoke_result.py`: pass.

## 2026-06-16 C3b Candidate Discovery Dry-run Path
- `python3 -m unittest tests.test_discover_c3b_smoke_candidates tests.test_import_pynq_smoke_result tests.test_run_third_goal_final_signoff`: `23` tests passed.
- `tools/discover_c3b_smoke_candidates.py` now emits `dry_run_import_command` before the active import command.
- Metadata JSON under `docs/resources` is skipped so discovery reports only real candidate smoke-result JSON files.
- Current generated discovery state remains `missing`, pass `0`, candidates `0`; no board result is fabricated.

## 2026-06-16 Unblock Checklist Dry-run Import Sequence
- `python3 -m unittest tests.test_write_third_goal_unblock_checklist tests.test_import_pynq_smoke_result tests.test_run_third_goal_final_signoff`: `20` tests passed.
- `generated/signoff/third_goal_unblock_checklist_2026_06_10.md` now uses the final-signoff runner for C3b dry-run import before active import.
- The runner path preserves import status in `generated/signoff/c3b_smoke_import_2026_06_10.json`,
  `generated/signoff/c3b_smoke_import_validation_2026_06_10.json`, and
  `generated/signoff/third_goal_final_signoff_run_2026_06_10.json`.
- Current checklist remains `pending-unblock` with blockers `2`; no board result or XR-VITs policy is created.

## 2026-06-16 ZCU104 Smoke Session Runner Import Flow
- `python3 -m unittest tests.test_prepare_zcu104_smoke_session tests.test_check_third_goal_preflight tests.test_check_c3b_board_smoke_readiness tests.test_write_c3b_physical_smoke_gate_audit`: `31` tests passed.
- C3b session regenerated:
  `generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.{json,md}`.
- VREF successor session regenerated:
  `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.{json,md}`.
- Host copy-back steps now use `tools/run_third_goal_final_signoff.py` dry-run import before active import.
- Neutral preflight after regeneration: `ok=76 warn=6 fail=0`.
- C3b physical-smoke gate audit remains `blocked_missing_canonical_physical_smoke_result`; bundle/session pass, canonical board JSON missing.

## 2026-06-16 Current Gate Recheck
- `python3 -m unittest tests.test_prepare_zcu104_smoke_session tests.test_check_third_goal_preflight tests.test_check_c3b_board_smoke_readiness tests.test_write_c3b_physical_smoke_gate_audit tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit`: `36` tests passed.
- `python3 -m json.tool generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`: pass.
- `python3 -m json.tool generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.json`: pass.
- `python3 -m json.tool generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.json`: pass.
- `python3 -m json.tool generated/signoff/third_goal_source_audit_2026_06_16.json`: pass.
- `python3 -m json.tool generated/signoff/third_goal_current_audit_2026_06_16.json`: pass.
- `git diff --check -- docs tools tests generated/pynq generated/signoff`: pass.
- Neutral preflight: `ok=76 warn=6 fail=0`.
- Final-signoff preflight remains expected-blocked: `ok=76 warn=4 fail=2`.
  Failing gates: `C3b AXIS/DMA physical smoke result` and `requested XR-VITs sibling`.
- Spark verifier could not run due quota. GPT5.5 fallback evaluator checked the runbook/import-flow concern; current runner-import flow is supported by `tools/run_third_goal_final_signoff.py` and intentionally documented in this validation record.

## 2026-06-16 PYNQ Smoke Candidate Discovery Generalization
- Spark verifier could not run due quota; GPT5.5 fallback confirmed the automation gap: VREF successor had validate/import/session/runner support, but candidate discovery was C3b-only.
- Added `tools/discover_pynq_smoke_candidates.py` with preset-driven patterns and runner import command templates.
- Added `tests/test_discover_pynq_smoke_candidates.py` for C3b and VREF successor discovery, invalid-candidate rejection, metadata/session noise skipping, and no-copy CLI behavior.
- `python3 -m py_compile tools/discover_pynq_smoke_candidates.py tools/run_third_goal_final_signoff.py tests/test_discover_pynq_smoke_candidates.py tests/test_run_third_goal_final_signoff.py`: pass.
- `python3 -m unittest tests.test_discover_pynq_smoke_candidates tests.test_run_third_goal_final_signoff`: `16` tests passed.
- `python3 -m unittest tests.test_discover_pynq_smoke_candidates tests.test_discover_c3b_smoke_candidates tests.test_run_third_goal_final_signoff tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit`: `27` tests passed.
- Final-signoff runner now emits `vref-successor-smoke-candidate-discovery` and records `vref_smoke_discovery_status`.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked result, `fail=2`, blockers `2`; runner evidence manifest pass with `50/50` required artifacts.
- New artifacts:
  - `generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.{json,md}`
  - `generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.{json,md}`
  - `generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.{json,md}`
- Current VREF discovery status is `missing`, pass `0`, candidates `0`; no board result is fabricated.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `48`, sources `161`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `8`, partial `3`, blocked `1`.

## 2026-06-16 Generic Discovery Gate Tracking
- Added required source-audit coverage for:
  - `generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.{json,md}`
  - `generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.{json,md}`
- Preflight now validates legacy VREF discovery plus generic C3b/VREF discovery JSON safety flags without creating board results or canonical inputs.
- Current audit now reports `gate_modes.smoke_candidate_discovery.{c3b_generic,vref_generic,vref_runner}` and keeps legacy `vref_smoke_discovery` for compatibility.
- `python3 -m py_compile tools/check_third_goal_preflight.py tools/write_third_goal_source_audit.py tools/write_third_goal_current_audit.py tests/test_check_third_goal_preflight.py tests/test_write_third_goal_source_audit.py tests/test_write_third_goal_current_audit.py`: pass.
- `python3 -m unittest tests.test_check_third_goal_preflight tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_discover_pynq_smoke_candidates`: `42` tests passed.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `52`, sources `171`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `8`, partial `3`, blocked `1`.
- Neutral preflight: `ok=79 warn=6 fail=0`.
- Final-signoff preflight remains expected-blocked: `ok=79 warn=4 fail=2`.
  Failing gates remain `C3b AXIS/DMA physical smoke result` and `requested XR-VITs sibling`.

## 2026-06-16 Final Unblock Route Surfacing
- Spark sidecar hit quota; GPT5.5 fallback identified that operator unblock command and XR-VITs choice packets were only optional-discovered.
- Promoted these artifacts to required source-audit evidence:
  - `generated/signoff/final_blocker_closure_readiness_2026_06_10.{json,md}`
  - `generated/signoff/final_unblock_intake_2026_06_10.{json,md}`
  - `generated/signoff/final_unblock_commands_2026_06_10.{json,md}`
  - `generated/signoff/xr_vits_unblock_packet_2026_06_10.{json,md}`
- Current audit now reports `final_blocker_closure`, `final_unblock_intake`, `final_unblock_commands`, and `xr_vits_unblock_packet`.
- Preflight blocker messages now point to `final_unblock_commands_2026_06_10.md`, `xr_vits_unblock_packet_2026_06_10.md`, C3b dry-run import, and XR-VITs policy dry-run.
- `python3 -m py_compile tools/check_third_goal_preflight.py tools/write_third_goal_source_audit.py tools/write_third_goal_current_audit.py tests/test_check_third_goal_preflight.py tests/test_write_third_goal_source_audit.py tests/test_write_third_goal_current_audit.py`: pass.
- `python3 -m unittest tests.test_check_third_goal_preflight tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit tests.test_check_final_blocker_closure_readiness tests.test_write_final_unblock_intake tests.test_write_final_unblock_commands tests.test_write_xr_vits_unblock_packet`: `40` tests passed.
- `python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/final_blocker_closure_readiness_2026_06_10.json --markdown-out generated/signoff/final_blocker_closure_readiness_2026_06_10.md`: expected blocked, `current_ready=False`, `candidate_ready=False`.
- `python3 tools/write_final_unblock_intake.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/final_unblock_intake_2026_06_10.json --markdown-out generated/signoff/final_unblock_intake_2026_06_10.md`: expected blocked, `candidate_ready=False`.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `60`, sources `173`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `8`, partial `3`, blocked `1`.
- Neutral preflight remains `ok=79 warn=6 fail=0`; final-signoff remains expected-blocked with `ok=79 warn=4 fail=2`.

## 2026-06-16 Final Runner Evidence Refresh
- Re-ran final signoff runner after final unblock route surfacing:
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`.
- Result remains expected-blocked: `status=blocked`, `fail=2`, blockers `2`.
- Final preflight summary: `ok=79 warn=4 fail=2`.
- Final evidence manifest: `pass`, required `50/50`; required evidence includes final unblock commands, XR-VITs unblock packet, blocker closure, and unblock intake artifacts.
- Final runner mirrored refreshed evidence to `HGTXR/docs/resources`.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `60`, sources `173`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `8`, partial `3`, blocked `1`.
- `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_check_third_goal_preflight tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit tests.test_write_final_unblock_commands tests.test_write_xr_vits_unblock_packet`: `48` tests passed.
- JSON parse checks passed for:
  - `generated/signoff/third_goal_final_signoff_run_2026_06_10.json`
  - `generated/signoff/final_evidence_manifest_2026_06_10.json`
  - `generated/signoff/third_goal_source_audit_2026_06_16.json`
  - `generated/signoff/third_goal_current_audit_2026_06_16.json`
- `git diff --check -- docs tools tests generated/signoff ../docs/resources`: pass.

## 2026-06-16 Final Operator Handoff Tracking
- Promoted these artifacts into required source-audit coverage:
  - `generated/signoff/final_unblock_candidate_audit_2026_06_10.{json,md}`
  - `generated/signoff/final_operator_handoff_2026_06_10.{json,md}`
  - `generated/signoff/final_operator_handoff_validation_2026_06_10.{json,md}`
- Current audit now reports `gate_modes.final_unblock_candidate_audit`, `gate_modes.final_operator_handoff`, and `gate_modes.final_operator_handoff_validation`.
- No board smoke result, XR-VITs policy, network command, or canonical input was created.
- `python3 -m py_compile tools/write_third_goal_source_audit.py tools/write_third_goal_current_audit.py tests/test_write_third_goal_source_audit.py tests/test_write_third_goal_current_audit.py`: pass.
- `python3 -m unittest tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit`: `5` tests passed.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `66`, sources `175`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `8`, partial `3`, blocked `1`.
- `python3 -m json.tool generated/signoff/third_goal_source_audit_2026_06_16.json`: pass.
- `python3 -m json.tool generated/signoff/third_goal_current_audit_2026_06_16.json`: pass.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff --json-out /tmp/hgtxr_third_goal_final_preflight_after_operator_handoff_tracking.json`: expected blocked, `ok=79 warn=4 fail=2`.
- `git diff --check -- docs tools tests generated/signoff`: pass.

## 2026-06-16 Final Evidence Audit Linkage Hardening
- Centralized `CURRENT_AUDIT_DATE_TAG` in `tools/write_final_evidence_manifest.py` for source/current audit artifacts.
- Added explicit failure coverage for:
  - `third_goal_source_audit_missing_zero`
  - `third_goal_current_audit_requirements_count`
- Documented the final runner sequence:
  `third-goal-source-audit` -> `third-goal-current-audit` -> `third-goal-source-audit-refresh`.
- `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tests/test_run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py`: pass.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff`: `17` tests passed.
- `python3 tools/write_third_goal_source_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, required `74`, sources `193`, missing `0`, markerless required `0`.
- `python3 tools/write_third_goal_current_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `blocked-external`, reflected `9`, partial `2`, blocked `1`.
- `python3 tools/write_req5_q4q8_swhw_match_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, fail `0`.
- `python3 tools/write_req6_parameterization_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, checks `39/39`.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked result, `fail=2`, blockers `2`; final evidence manifest `pass`, required `62/62`; source audit `pass`, current audit `blocked-external`.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit tests.test_check_third_goal_preflight`: `43` tests passed.
- JSON status checks passed for final evidence manifest, final runner summary, source audit, and current audit.
- `git diff --check -- docs tools tests generated/signoff ../docs/resources`: pass.

## 2026-06-16 VREF-P0-02 QKV Weight Cache URAM Successor
- Ran isolated CSim project:
  `hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csim`.
- Ran isolated csynth project:
  `hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth`.
- Compile flags added `-DHGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1` on top of `softmax_input_x2` and `dsp_mixed_stream`.
- CSim result: expected raw `[58, -51, 42, -28, 36, -41]`, `runtime_state=2`, `CSim done with 0 errors`.
- CSynth result from `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth/solution_e2e_q4w8a/syn/report/csynth.xml`:
  estimated clock `4.069 ns`, latency `498485`, `114 BRAM_18K`, `128 DSP`, `19664 FF`, `43236 LUT`, `40 URAM`.
- Delta versus `dsp_mixed_stream`: `-30 BRAM_18K`, `+8 URAM`, `0 DSP`, `0 LUT`, `0 latency`.
- C3b HLS thresholds pass: latency `<= 37508072`, estimated clock `<= 5.0 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`.
- HLS IP package generated:
  `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/ip/component.xml`
  and `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/export.zip`.
- Routed overlay generated:
  `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit`
  and matching `.hwh`, also copied under `pynq/hgtxr/`.
- Routed timing met: setup slack `4.517 ns`, hold slack `0.010 ns`, pulse-width slack `3.500 ns`.
- Route status: `19846/19846` routable nets fully routed, routing errors `0`.
- Remaining promotion gate: physical smoke.
- PYNQ plumbing added for QKV URAM successor:
  variant `vref-p0-softmax-input-x2-qkv-uram`, preset `axis-vref-p0-softmax-input-x2-qkv-uram`, canonical smoke JSON `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json`.
- `python3 tools/write_vref_p0_qkv_uram_cache_successor.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, checks `10/10`.
- `LD_LIBRARY_PATH=/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip ... /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f vivado/scripts/package_e2e_axis_ip.tcl`: pass, `export.zip` generated.
- `LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH /tools/Xilinx/Vivado/2023.2/bin/vivado -mode batch -source vivado/scripts/build_e2e_axis_dma_bitstream.tcl -tclargs -project_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay -bd_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_system -artifact_name hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram -hls_ip_repo generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/ip`: pass, bitstream generated.
- `python3 tools/package_e2e_axis_dma_pynq_bundle.py --variant vref-p0-softmax-input-x2-qkv-uram --out-dir generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle --tar generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz`: pass, tar sha256 `0fcf20560549b222519e5972ed827db8c0014a809faf352dbe5097b8d5aa70a4`.
- `python3 tools/prepare_zcu104_smoke_session.py --variant vref-p0-softmax-input-x2-qkv-uram --preset axis-vref-p0-softmax-input-x2-qkv-uram --bundle-dir generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle --tar generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz --json-out generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.json --markdown-out generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.md`: pass.
- QKV URAM preflight integration:
  `tools/check_third_goal_preflight.py` validates QKV bit/hwh, bundle manifest/tar, run/validate scripts, ZCU104 session JSON/Markdown contract, and QKV physical-smoke JSON with preset `axis-vref-p0-softmax-input-x2-qkv-uram`.
- `python3 -m unittest tests.test_check_third_goal_preflight`: `24` tests passed.
- `python3 tools/check_third_goal_preflight.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --mode final-signoff --json-out generated/signoff/third_goal_final_signoff_2026_06_10.json`: expected blocked, `ok=96`, `warn=5`, `fail=2`; QKV artifact/session checks are `ok`, QKV physical smoke is `warn`.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked, final evidence manifest `62/62`, fail `2`, blockers `2`.
- `python3 -m py_compile tools/write_vref_p0_qkv_uram_cache_successor.py tools/write_third_goal_source_audit.py tools/write_third_goal_current_audit.py tests/test_write_vref_p0_qkv_uram_cache_successor.py tests/test_write_third_goal_source_audit.py tests/test_write_third_goal_current_audit.py`: pass.
- `python3 -m unittest tests.test_write_vref_p0_qkv_uram_cache_successor tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit`: `7` tests passed.
- `python3 -m unittest tests.test_write_vref_p0_qkv_uram_cache_successor tests.test_package_e2e_axis_dma_pynq_bundle tests.test_validate_pynq_bundle_package tests.test_validate_pynq_smoke_result tests.test_import_pynq_smoke_result tests.test_prepare_zcu104_smoke_session tests.test_run_zcu104_c3b_smoke_remote`: `39` tests passed.
- Final-signoff runner QKV integration:
  `tools/run_third_goal_final_signoff.py` now records `qkv_uram_import_status`, `qkv_uram_remote_status`,
  `qkv_uram_remote_execute`, `qkv_uram_required_for_final_signoff`, and `qkv_uram_smoke_discovery_status`.
- QKV smoke candidate discovery artifact generated:
  `generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.{json,md}` and mirrored to `../docs/resources/`.
- `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/discover_pynq_smoke_candidates.py tests/test_run_third_goal_final_signoff.py tests/test_discover_pynq_smoke_candidates.py`: pass.
- `python3 -m unittest tests.test_discover_pynq_smoke_candidates tests.test_run_third_goal_final_signoff`: `20` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked result,
  `status=blocked`, `qkv_uram_smoke_discovery_status=missing`, `qkv_uram_import_status=skipped`,
  `qkv_uram_remote_status=skipped`, blockers remain only `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.
- Current/source audit QKV tracking hardened:
  source audit now requires `generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.{json,md}`.
  current audit exposes `gate_modes.qkv_uram_runner` with `qkv_uram_import_status`, `qkv_uram_remote_status`,
  `qkv_uram_remote_execute`, `qkv_uram_required_for_final_signoff`, and `qkv_uram_smoke_discovery_status`.
- `python3 -m unittest tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit tests.test_discover_pynq_smoke_candidates tests.test_run_third_goal_final_signoff tests.test_check_third_goal_preflight`: `49` tests passed.
- Latest source audit: `required_count=74`, `source_count=193`, `missing_required=0`.
- Latest current audit: `blocked-external`; reflected `9`, partial `2`, blocked `1`; `qkv_uram_runner` import `skipped`, remote `skipped`, discovery `missing`, required `False`.
- Latest Req5 Q4/Q8 SW-HW match audit: `pass`, fail `0`; physical board smoke remains a Req4/final external gate, not a Req5 software-local blocker.
- Latest Req6 parameterization audit: `pass`, checks `62/62`; no HLS/Vivado/board execution.
- `git diff --check -- docs tools tests generated/signoff ../docs/resources generated/pynq pynq`: pass.

## 2026-06-16 QKV URAM Final Manifest Consistency Refresh
- Closed completed Spark sidecar `019eccf0-38e4-75a2-9a9c-d2df33171528`; new Spark spawn was attempted for read-only doc/evidence review but failed with `agent thread limit reached`, so integration was completed by the main agent.
- Final evidence manifest QKV consistency now directly checks successor pass state, URAM ceiling, latency, DSP, LUT, BRAM reduction, BRAM delta, routed WNS, route errors, and fully-routed net count.
- Current QKV URAM consistency values:
  latency `498485 <= 37508072`, DSP `128 <= 604`, LUT `43236 <= 126506`, URAM `40 <= 64`, BRAM delta `-30`, WNS `4.517 >= 4.415`, route errors `0`, fully routed `19846/19846`.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked result, final evidence manifest `pass`, required `62/62`; source audit `pass`, required `74`, sources `193`, missing `0`; current audit `blocked-external`.
- `generated/signoff/final_signoff_bundle_validation_2026_06_10.json`: `pass`, fail `0`, `evidence_contract_consistency_count=42`, trace/handoff contracts match live final evidence manifest.
- Remaining blockers unchanged: `C3b AXIS/DMA physical smoke result` and `requested XR-VITs sibling` or approved replacement policy.

## 2026-06-16 Final Unblock Command Card Hardening
- Spark sidecar `019eccf6-f127-7c91-9302-f6d3d5d57f32` completed read-only review and identified that the command card lacked a local blocker snapshot command with required `--json-out/--markdown-out` outputs.
- `tools/write_final_unblock_commands.py` U0 commands now include output paths and a no-side-effect current-state snapshot command:
  `check_final_blocker_closure_readiness.py --c3b-no-require-paths --json-out ... --markdown-out ...`.
- `tools/write_final_unblock_commands.py` now emits optional U5 QKV URAM successor smoke/promotion commands without making QKV a default final-signoff blocker.
- `docs/CHOICE.md` now records QKV options Q1/Q2/Q3 and the local blocker snapshot command.
- `python3 -m unittest tests.test_write_final_unblock_commands`: `3` tests passed.
- `python3 -m py_compile tools/write_final_unblock_commands.py tests/test_write_final_unblock_commands.py`: pass.
- `python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-no-require-paths --json-out /tmp/hgtxr_final_blocker_readiness_current.json --markdown-out /tmp/hgtxr_final_blocker_readiness_current.md`: expected `blocked`, `current_ready=False`, `candidate_ready=False`.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked, command card sections `6`, final evidence manifest `62/62`, final bundle validation `pass`, fail `0`.

## 2026-06-16 QKV URAM Closeout Packet Integration
- Spark sidecar `019eccfa-dfa3-7fc2-ae05-c977739d0a6c` completed read-only review and confirmed that U5/QKV URAM successor commands were absent from the closeout packet.
- `tools/write_final_unblock_closeout_packet.py` now emits `qkv_uram_successor_gate`, adds U5 commands under `operator_commands.qkv_uram_successor`, and includes QKV smoke/import/remote outputs in successor outputs.
- `tools/validate_final_unblock_closeout_packet.py` now checks that QKV URAM remains optional, has known gate status, has pass csynth evidence, has resource keys, has canonical physical-smoke path, and exposes execute/dry-run import commands.
- Current QKV closeout gate:
  status `ready-for-physical-smoke`, required_for_final_signoff `False`, csynth `pass`, latency `498485`, resources `114 BRAM_18K`, `128 DSP`, `43236 LUT`, `40 URAM`, routed WNS `4.517`, route errors `0`.
- `python3 -m unittest tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet`: `10` tests passed.
- `python3 -m py_compile tools/write_final_unblock_closeout_packet.py tools/validate_final_unblock_closeout_packet.py tests/test_write_final_unblock_closeout_packet.py tests/test_validate_final_unblock_closeout_packet.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked, closeout validation `pass`, checks `40`, fail `0`; default blockers remain `2`.

## 2026-06-16 QKV URAM Closeout Audit/Manifest Hardening
- Spark sidecar `019eccfe-dcb8-79d2-a42e-05cd75c058d0` completed read-only review and confirmed QKV U5 was represented as optional, but no manifest-level conditional check existed for the case where QKV physical smoke is promoted to a hard gate.
- `tools/write_third_goal_current_audit.py` now reports `final_unblock_commands.section_ids`, `has_qkv_uram_u5`, and closeout `qkv_uram_successor_gate` / `qkv_uram_command_count`.
- `tools/write_final_evidence_manifest.py` now checks closeout QKV gate optional status, known gate status, U5 execute/import commands, and conditional required-smoke consistency:
  if QKV is not required, missing smoke is allowed; if QKV is required, physical smoke status and JSON must be present.
- Regenerated 2026-06-16 closeout packets:
  default blockers `2`, VREF-required blockers `3`, both validation runs `40/40`, fail `0`.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_write_third_goal_current_audit`: `9` tests passed.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_write_third_goal_current_audit tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff`: `29` tests passed.
- Latest final evidence manifest: `pass`, required `62/62`, consistency checks `47`, failed consistency `0`.
- Latest current audit command-card summary: section count `6`, section ids `U0,U1,U2,U4,U5,U3`, `has_qkv_uram_u5=True`.

## 2026-06-16 Final Signoff Safety Schema Normalization
- Spark sidecar `019ecd04-90f4-78e0-b98b-3de1b9061716` completed read-only review and found no failing signoff inconsistency, but identified split safety keys: command card used `executes_network`, closeout packet used `executes_commands`.
- `tools/write_final_unblock_commands.py` and `tools/write_final_unblock_closeout_packet.py` now emit both `executes_commands=False` and `executes_network=False`; command card also emits `writes_canonical_inputs=False`.
- `tools/validate_final_signoff_bundle.py` now validates both command and network execution safety, accepting either key as an alias for backward compatibility.
- `tools/validate_final_unblock_closeout_packet.py` now normalizes both safety keys and reports both in validation output.
- `python3 -m unittest tests.test_write_final_unblock_commands tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff`: `42` tests passed.
- `python3 -m py_compile tools/write_final_unblock_commands.py tools/write_final_unblock_closeout_packet.py tools/validate_final_unblock_closeout_packet.py tools/validate_final_signoff_bundle.py tests/test_write_final_unblock_commands.py tests/test_write_final_unblock_closeout_packet.py tests/test_validate_final_unblock_closeout_packet.py tests/test_validate_final_signoff_bundle.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked, final signoff bundle validation `pass`, fail `0`, checks `64`; safety checks all `pass`.
- Regenerated `generated/signoff/final_unblock_closeout_packet_2026_06_16.json` and VREF-required variant with normalized safety keys; both validation snapshots remain `pass`, checks `40`, fail `0`.

## 2026-06-16 Final Signoff Blocker/Date Provenance Clarification
- Spark sidecar `019ecd09-7b8f-7593-b526-6650c8083785` completed read-only review and recommended explicit mapping between default external blocker names and missing input paths, plus explicit date-tag provenance for mixed `2026_06_10`/`2026_06_16` manifest artifacts.
- `tools/write_third_goal_current_audit.py` now emits:
  `external_blocker_names`, `blocked_external_inputs`, `blocked_external_input_paths_by_blocker`, `remaining_external_input_details`, and `blocker_schema`.
- Current default blocker-path mapping:
  `C3b AXIS/DMA physical smoke result` -> `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`;
  `requested XR-VITs sibling` -> `/home/kjm26/project/PRJXR/XR-VITs`.
- `tools/write_final_evidence_manifest.py` now emits `source_date_tag` per artifact, `artifact_date_tags`, `date_tag_policy`, and `date_tag_profile`.
- Latest manifest date tags: canonical signoff `2026_06_10`, current audit `2026_06_16`, plus `undated` for SHA-only artifacts.
- `python3 -m unittest tests.test_write_third_goal_current_audit tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff`: `29` tests passed.
- `python3 -m py_compile tools/write_third_goal_current_audit.py tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tools/run_third_goal_final_signoff.py tests/test_write_third_goal_current_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_run_third_goal_final_signoff.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked, source audit `pass`, required `74`, sources `195`, missing `0`; final evidence manifest `pass`, required `62/62`; current audit `blocked-external`.

## 2026-06-16 VREF-P0-02 Buffer Lifetime Final Manifest Promotion
- Promoted `generated/signoff/vref_p0_buffer_lifetime_audit_2026_06_16.{json,md}` into required final evidence.
- Final evidence manifest now checks VREF-P0-02 large-buffer URAM, small-memory LUTRAM, Q/K/V URAM, pooled/score/probability LUTRAM, and C3b DSP/LUT resource-policy deltas.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff tests.test_write_vref_p0_buffer_lifetime_audit`: `30` tests passed.
- `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tools/run_third_goal_final_signoff.py tools/write_vref_p0_buffer_lifetime_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_run_third_goal_final_signoff.py tests/test_write_vref_p0_buffer_lifetime_audit.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final evidence manifest `pass`, required `64/64`, consistency checks `61`, failed consistency `0`; final bundle validation `pass`, fail `0`; source audit `pass`, required `74`, sources `197`, missing `0`; final preflight `ok=96`, `warn=5`, `fail=2`.
- Remaining blockers are unchanged: `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 C3b Canonical Physical-Smoke Gate Hardening
- `tools/check_third_goal_preflight.py` now accepts only `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` as C3b physical-smoke evidence.
- Generated bundle or remote result JSON is reported as noncanonical and import-required; it no longer clears final signoff before import.
- `tools/write_c3b_physical_smoke_gate_audit.py` now parses canonical validation JSON when present and reports `canonical_validation.status`, `preset`, `result_json`, `source_sha256`, and validation errors.
- `python3 -m unittest tests.test_check_third_goal_preflight tests.test_write_c3b_physical_smoke_gate_audit`: `29` tests passed.
- `python3 -m py_compile tools/check_third_goal_preflight.py tools/write_c3b_physical_smoke_gate_audit.py tests/test_check_third_goal_preflight.py tests/test_write_c3b_physical_smoke_gate_audit.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; final manifest `64/64`; source audit `74/197`, missing `0`.

## 2026-06-16 XR-VITs Replacement Policy Integrity Hardening
- `tools/create_xr_vits_replacement_policy.py` now writes `candidate_audit_fingerprint`, `candidate_audit_recommendation_snapshot`, `candidate_audit_meta`, and `policy_fingerprint`.
- `tools/check_final_blocker_closure_readiness.py` and `tools/check_third_goal_preflight.py` require a matching candidate-audit fingerprint/snapshot before an approved replacement policy can clear Req11.
- Legacy policy JSON remains parseable, but it does not clear final signoff; regenerate with `tools/create_xr_vits_replacement_policy.py` to bind approval to current candidate audit content.
- `tools/write_xr_vits_gate_audit.py` now reports `active_policy_integrity.status`, `policy_fingerprint`, and `candidate_audit_fingerprint`.
- `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_xr_vits_gate_audit tests.test_create_xr_vits_replacement_policy tests.test_check_final_blocker_closure_readiness tests.test_check_third_goal_preflight tests.test_validate_final_signoff_bundle`: `64` tests passed.
- `python3 -m py_compile tools/create_xr_vits_replacement_policy.py tools/check_final_blocker_closure_readiness.py tools/check_third_goal_preflight.py tools/write_xr_vits_gate_audit.py tools/run_third_goal_final_signoff.py tests/test_run_third_goal_final_signoff.py tests/test_write_xr_vits_gate_audit.py tests/test_create_xr_vits_replacement_policy.py tests/test_check_final_blocker_closure_readiness.py tests/test_check_third_goal_preflight.py tests/test_validate_final_signoff_bundle.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; final manifest `64/64`; source audit `74/197`, missing `0`.

## 2026-06-16 XR-VITs Policy Integrity Operator Packet Propagation
- Spark sidecar `019ecd1f-eff9-7c53-9642-0d4d1f5acc3e` found that operator artifacts still treated fingerprint-bound policy mostly as metadata.
- `tools/write_final_unblock_commands.py` now exposes `xr_vits_policy_integrity` and embeds it into U2b/U4b replacement paths.
- `tools/write_final_unblock_closeout_packet.py` and `tools/write_final_operator_handoff.py` now carry policy path, candidate audit path, required integrity fields, policy existence, validation status, and fingerprints when a policy exists.
- `tools/validate_final_unblock_closeout_packet.py`, `tools/validate_final_operator_handoff.py`, and `tools/validate_final_signoff_bundle.py` now reject legacy/drifted policies and compare embedded validation status with live policy validation. Current pending state is accepted only while Req11 remains blocked and no policy exists.
- `python3 -m unittest tests.test_write_final_unblock_commands tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet tests.test_write_final_operator_handoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `31` tests passed.
- `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit tests.test_write_third_goal_requirements_trace tests.test_write_final_unblock_commands tests.test_write_final_unblock_closeout_packet tests.test_validate_final_unblock_closeout_packet tests.test_write_final_operator_handoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `60` tests passed.
- `python3 -m py_compile ...`: pass for touched tools/tests.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; closeout validation `pass`, checks `49`; final bundle validation `pass`, fail `0`; final manifest `64/64`; source audit `74/197`, missing `0`.

## 2026-06-16 Current Audit XR-VITs Policy Integrity Surfacing
- Spark spawn for read-only gap review failed with `agent thread limit reached`; main agent completed local fallback under the same Task Card.
- `tools/write_third_goal_current_audit.py` now summarizes XR-VITs policy integrity at current-audit level:
  command-card required fields, operator-handoff validation status, policy existence, policy check names/count, and cross-artifact consistency.
- Current state in `generated/signoff/third_goal_current_audit_2026_06_16.json`:
  `xr_vits_policy_integrity.consistent=True`, `operator_handoff_policy_check_count=9`,
  `operator_handoff.validation_status=pending-policy-creation`, `operator_handoff.policy_exists=False`.
- `python3 -m unittest tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `30` tests passed.
- `python3 -m py_compile ...`: pass for touched current-audit/signoff tools and tests.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; closeout validation `49/49`, final manifest `64/64`, source audit `74/197`, final preflight `ok=96`, `warn=5`, `fail=2`.

## 2026-06-16 Completion Audit XR-VITs Policy Integrity Surfacing
- Spark spawn for completion-audit read-only review failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/write_third_goal_completion_audit.py` now reads `third_goal_current_audit_2026_06_16.json` and exposes:
  `xr_vits_policy_integrity`, `external_blocker_paths`, `current_audit_status`, policy validation status, and policy check count in Req11 evidence and Markdown sections.
- Current completion audit state:
  status `blocked`, pass `8`, partial `4`, blocked `2`, current audit `blocked-external`,
  policy integrity `consistent=True`, policy check count `9`, validation `pending-policy-creation`.
- `python3 -m unittest tests.test_write_third_goal_completion_audit tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_write_third_goal_requirements_trace`: `37` tests passed.
- `python3 -m py_compile ...`: pass for touched completion/current/signoff tools and tests.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; closeout validation `49/49`; final manifest `64/64`.

## 2026-06-16 Completion Audit Req0 Current Handover Evidence Refresh
- Spark spawn for Req0 handover evidence review failed with `agent thread limit reached`; main agent completed local fallback under Task Card `T-3G-REQ0-HANDOVER-001`.
- `tools/write_third_goal_completion_audit.py` Req0 now checks `docs/track/HANDOVER.md` and `docs/track/log.md` in addition to `PROGRESS.md`, legacy E2E handover, `CHOICE.md`, and `Validation.md`.
- Current completion audit Req0 evidence includes:
  `docs/track/HANDOVER.md: exists` and `docs/track/log.md: exists`.
- `python3 -m unittest tests.test_write_third_goal_completion_audit`: `4` tests passed.
- `python3 -m py_compile tools/write_third_goal_completion_audit.py tests/test_write_third_goal_completion_audit.py`: pass.
- `python3 -m unittest tests.test_write_third_goal_completion_audit tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `32` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; completion audit `pass=8`, `partial=4`, `blocked=2`; final preflight `ok=96`, `warn=5`, `fail=2`; final manifest `64/64`.

## 2026-06-16 Requirements Trace XR-VITs Policy Integrity Closure
- Spark spawn for next-gap review failed with `agent thread limit reached`; main agent completed local fallback under Task Card `T-3G-NEXT-GAP-001`.
- `tools/write_third_goal_requirements_trace.py` now exposes `xr_vits_policy_integrity` and makes the `requested XR-VITs sibling` blocker acceptance require:
  replacement policy generator `tools/create_xr_vits_replacement_policy.py`,
  `candidate_audit_fingerprint`, `candidate_audit_recommendation_snapshot`, `candidate_audit_meta`, and `policy_fingerprint`,
  and rejection of legacy or drifted policies for final signoff closure.
- Current trace state:
  status `blocked`, blocked requirement ids `['11']`, partial requirement ids `['2', '3', '4', '10']`,
  required policy fields complete `True`, legacy policy clears final signoff `False`.
- `python3 -m unittest tests.test_write_third_goal_requirements_trace`: `3` tests passed.
- `python3 -m py_compile tools/write_third_goal_requirements_trace.py tests/test_write_third_goal_requirements_trace.py`: pass.
- `python3 -m unittest tests.test_write_third_goal_requirements_trace tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `31` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; final manifest `64/64`; requirements trace remains `blocked` only because external blockers remain.
- `git diff --check -- docs tools tests generated/signoff ../docs/resources`: pass.

## 2026-06-16 Final Manifest Requirements Trace Policy Gate
- Spark spawn for Task Card `T-3G-MANIFEST-TRACE-POLICY-001` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/write_final_evidence_manifest.py` now validates `third_goal_requirements_trace_2026_06_10.json` policy integrity with consistency checks:
  `third_goal_requirements_trace_status_known`,
  `third_goal_requirements_trace_req11_blocked`,
  `third_goal_requirements_trace_xr_vits_policy_fields_complete`,
  and `third_goal_requirements_trace_xr_vits_legacy_policy_rejected`.
- Current final manifest state:
  status `pass`, required `64/64`, consistency checks `65`, failed consistency `[]`.
- New trace-policy checks are all `pass`: status `blocked`, Req11 blocked `['11']`, policy fields complete `True`, legacy policy clears final signoff `False`.
- `python3 -m unittest tests.test_write_final_evidence_manifest`: `9` tests passed.
- `python3 -m py_compile tools/write_final_evidence_manifest.py tests/test_write_final_evidence_manifest.py`: pass.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff tests.test_write_third_goal_requirements_trace`: `32` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; final manifest `64/64`, consistency `65`, failed `0`.

## 2026-06-16 Final Bundle Trace-Policy Manifest Gate
- Spark spawn for Task Card `T-3G-BUNDLE-TRACE-POLICY-001` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/validate_final_signoff_bundle.py` now requires the final evidence manifest to include and pass these requirements-trace policy checks:
  `third_goal_requirements_trace_status_known`,
  `third_goal_requirements_trace_req11_blocked`,
  `third_goal_requirements_trace_xr_vits_policy_fields_complete`,
  and `third_goal_requirements_trace_xr_vits_legacy_policy_rejected`.
- Current final bundle validation state:
  status `pass`, pass `74`, fail `0`; `evidence_contract_consistency_count=65`,
  `final_evidence_manifest_trace_policy_checks_present=pass`, and
  `final_evidence_manifest_trace_policy_checks_pass=pass`.
- `python3 -m unittest tests.test_validate_final_signoff_bundle`: `8` tests passed.
- `python3 -m py_compile tools/validate_final_signoff_bundle.py tests/test_validate_final_signoff_bundle.py`: pass.
- `python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest`: `31` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; bundle validation `pass`, fail `0`.

## 2026-06-16 Final Operator Handoff Evidence Contract Threshold
- Spark spawn for Task Card `T-3G-HANDOFF-CONTRACT-001` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/validate_final_operator_handoff.py` now requires `final_evidence_manifest_contract.consistency_count >= 72`.
- Current operator handoff validation state:
  status `pass`, checks `44`, fail `0`; `evidence_manifest_consistency_count=72`,
  `evidence_manifest_failed_consistency_zero=[]`,
  XR-VITs required policy fields present, and legacy policy accepted `False`.
- `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_write_final_operator_handoff`: `10` tests passed.
- `python3 -m py_compile tools/validate_final_operator_handoff.py tests/test_validate_final_operator_handoff.py tests/test_write_final_operator_handoff.py`: pass.
- `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_write_final_operator_handoff tests.test_run_third_goal_final_signoff tests.test_validate_final_signoff_bundle`: `32` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; operator handoff validation `pass`, fail `0`.

## 2026-06-16 Final Validator Count Freshness Gates
- Spark spawn for Task Card `T-3G-VALIDATION-DOCS` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/validate_final_signoff_bundle.py` now emits `check_count=len(checks)` in JSON and Markdown output; current bundle validation is `pass`, `check_count=74`, `fail_count=0`.
- `tools/write_final_evidence_manifest.py` now checks validator freshness counts:
  `final_operator_handoff_validation_check_count >= 44` and `final_signoff_bundle_validation_check_count >= 74`.
  It intentionally avoids checking validator pass/fail status inside the manifest because those validators consume the manifest contract.
- Current final evidence manifest state:
  status `pass`, required `66/66`, consistency checks `72`, failed consistency `[]`.
- `python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `19` tests passed.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_write_final_operator_handoff`: `43` tests passed.
- `python3 -m py_compile tools/validate_final_signoff_bundle.py tests/test_validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_write_final_evidence_manifest.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; final manifest `pass`; operator handoff validation `pass`, `check_count=44`, `fail_count=0`; bundle validation `pass`, `check_count=74`, `fail_count=0`.

## 2026-06-16 XR-VITs Replacement Policy Preview Artifact
- Spark spawn for Task Card `T-3G-XRVITS-PREVIEW-001` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/create_xr_vits_replacement_policy.py` now supports dry-run-only preview outputs:
  `--preview-json-out` and `--preview-markdown-out`.
  Preview output is rejected unless `--dry-run` is set, so it cannot be mistaken for an active policy write.
- `tools/run_third_goal_final_signoff.py` now always emits non-active preview evidence:
  `generated/signoff/xr_vits_replacement_policy_preview_2026_06_10.{json,md}`
  and mirrors it to `../docs/resources/`.
- Preview state:
  status `pass`, `preview_only=True`, `active_policy_written=False`, integrity `pass`,
  policy fingerprint `4882e521d6700319905085d3d1641036aa410d3be8a5210793091308aa2537fd`.
- `tools/write_third_goal_source_audit.py` now requires the preview JSON/Markdown; latest source audit is `pass`, required `76`, sources `199`, missing `0`.
- `tools/write_final_evidence_manifest.py` now requires the mirrored preview JSON/Markdown and validates preview-only/no-side-effect integrity.
  Latest final evidence manifest is `pass`, required `66/66`, consistency checks `72`, failed consistency `[]`.
- `python3 -m unittest tests.test_create_xr_vits_replacement_policy tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_write_third_goal_source_audit tests.test_validate_final_operator_handoff tests.test_write_final_operator_handoff tests.test_validate_final_signoff_bundle`: `55` tests passed.
- `python3 -m py_compile tools/create_xr_vits_replacement_policy.py tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tools/write_third_goal_source_audit.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_create_xr_vits_replacement_policy.py tests/test_run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py tests/test_write_third_goal_source_audit.py tests/test_validate_final_operator_handoff.py tests/test_write_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final preflight `ok=96`, `warn=5`, `fail=2`; completion audit `pass=11`, `partial=1`, `blocked=2`; operator handoff validation `pass`, `44/44`; bundle validation `pass`, `74/74`; source audit `76/199`; final manifest `66/66`.

## 2026-06-16 Completion/Requirements Trace Evidence Alignment
- Spark spawn for Task Card `T-3G-GAP-AUDIT-002` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/write_third_goal_completion_audit.py` now separates implemented baseline/reference evidence from final external signoff evidence:
  Req2 passes when manual Spec fallback and model-routing fallback are reflected in current audit;
  Req3 passes when C3b bit/hwh/smoke-bundle baseline artifacts exist;
  Req10 passes when XR-VIT reference evidence and candidate audit exist.
- Req4 remains partial until physical ZCU104 E2E smoke is captured; Req11 remains blocked until exact XR-VITs or approved replacement policy exists.
- Latest completion audit: status `blocked`, pass `11`, partial `1`, blocked `2`.
- Latest requirements trace: status `blocked`, blocked requirement ids `['11']`, partial requirement ids `['4']`.
- `python3 -m unittest tests.test_write_third_goal_completion_audit tests.test_write_third_goal_requirements_trace tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff`: `33` tests passed.
- `python3 -m py_compile tools/write_third_goal_completion_audit.py tests/test_write_third_goal_completion_audit.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `66/66`, consistency checks `72`, source audit `76/199`; remaining blockers stay `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 HG-PIPE Operator Property Coverage
- Spark sidecar `019ecd5b-edd9-77a1-9698-7d43691cc23d` identified HG-PIPE operator coverage as the next internally actionable gap after excluding board smoke and active XR-VITs policy approval.
- `tools/write_hgpipe_operator_audit.py` now adds deterministic contract property checks:
  table length vs cursor entries, cursor bound vs entries, integer output table range vs `ap_int`/`ap_uint` output type,
  Softmax exp nonnegative/finite/nonincreasing checks, LayerNorm rsqrt positive/finite checks, and quantize-clamp bit-range checks.
- Latest HG-PIPE operator audit:
  status `pass`, reference checks `97/97`, sampled values `5899008`, property checks `211/211`, fail `0`.
- Current audit now separates Req10 reference usage from Req11 exact-source/replacement approval:
  current audit status `blocked-external`, reflected `10`, partial `1`, blocked `1`.
- `python3 -m unittest tests.test_write_hgpipe_operator_audit tests.test_write_third_goal_current_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_requirements_trace tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff`: `37` tests passed.
- `python3 -m py_compile tools/write_hgpipe_operator_audit.py tests/test_write_hgpipe_operator_audit.py tools/write_third_goal_current_audit.py tests/test_write_third_goal_current_audit.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `66/66`, consistency checks `72`, source audit `76/199`; remaining blockers stay `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 HG-PIPE Final Evidence Contract Integration
- Spark sidecar `019ecd62-a1f7-7912-87b4-8088eee4f200` completed read-only review and recommended binding HG-PIPE operator `property_summary` into final evidence manifest and validator contracts.
- `tools/write_final_evidence_manifest.py` now requires `hgpipe_operator_audit_2026_06_16.{json,md}` under `docs/resources` and adds HG-PIPE consistency checks for:
  file presence, audit pass status, reference checks `97/97`, sampled values `5899008`, property summary pass/no-fail/count consistency, property checks `211`, all operator statuses, and per-operator property failure counts.
- `tools/validate_final_operator_handoff.py` now requires `final_evidence_manifest_contract.consistency_count >= 83`.
- `tools/validate_final_signoff_bundle.py` now requires the HG-PIPE manifest artifact ids and key HG-PIPE consistency checks to be present and passing.
- `tools/run_third_goal_final_signoff.py` now regenerates and mirrors HG-PIPE operator audit evidence before final manifest generation.
- `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_operator_handoff tests.test_run_third_goal_final_signoff`: `45` tests passed.
- `python3 -m unittest tests.test_write_hgpipe_operator_audit tests.test_write_third_goal_current_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_requirements_trace tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `54` tests passed.
- `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_operator_handoff.py tests/test_run_third_goal_final_signoff.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `68/68`, consistency checks `83`, failed `0`; source audit `76/201`; operator handoff validation `44/44`; bundle validation `74/74`; remaining blockers stay `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 Spec/Plan Current-Doc Freshness Gate
- Spark spawn for Task Card `T-3G-NEXT-GAP-001` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/write_spec_plan_conformance_audit.py` now validates hardware-local current docs against live final evidence counts:
  `docs/Spec.md`, `docs/track/PROGRESS.md`, and `docs/track/HANDOVER.md`.
- Freshness checks require final manifest `68/68`, consistency checks `83`, source audit `76/201`, and current audit reflected `10`, partial `1`, blocked `1`.
- The audit also rejects stale final-evidence tokens in current docs, including `66/66`, consistency `72`, and sources `199`.
- Current spec/plan conformance audit: status `pass`, checks `46/46`, fail `0`.
- `python3 -m unittest tests.test_write_spec_plan_conformance_audit`: `4` tests passed.
- `python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `47` tests passed.
- `python3 -m py_compile tools/write_spec_plan_conformance_audit.py tests/test_write_spec_plan_conformance_audit.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec/plan conformance `46/46`; final manifest `68/68`, consistency checks `83`, failed `0`; source audit `76/201`; current audit reflected `10`, partial `1`, blocked `1`; remaining blockers stay `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 VREF-P0-01 PoT Scale Final Evidence Promotion
- Spark spawn for Task Card `T-3G-INTERNAL-GAP-002` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/run_third_goal_final_signoff.py` now regenerates and mirrors:
  `vref_p0_pot_scale_audit_2026_06_16.{json,md}` and
  `vref_p0_pot_scale_sweep_2026_06_16.{json,md}`.
- `tools/write_third_goal_source_audit.py` now requires those four PoT scale artifacts.
- `tools/write_final_evidence_manifest.py` now requires those artifacts and checks PoT audit pass/count, sweep pass, sweep candidate coverage, zero failures, and current-scale recommendation.
- `tools/validate_final_operator_handoff.py` now requires evidence contract consistency count `>=88`.
- `tools/validate_final_signoff_bundle.py` now requires the PoT artifact ids and manifest consistency checks to be present/pass.
- Latest PoT evidence: audit `pass`, checks `14/14`; sweep `pass`, specs `3`, candidates `45`, fail `0`, recommendation `keep current PoT scales for C3b`.
- `python3 -m unittest tests.test_write_vref_p0_pot_scale_audit tests.test_write_vref_p0_pot_scale_sweep tests.test_write_third_goal_source_audit tests.test_write_spec_plan_conformance_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff`: `54` tests passed.
- `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_third_goal_source_audit.py tools/write_spec_plan_conformance_audit.py tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_vref_p0_pot_scale_audit.py tools/write_vref_p0_pot_scale_sweep.py tests/test_run_third_goal_final_signoff.py tests/test_write_third_goal_source_audit.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_vref_p0_pot_scale_audit.py tests/test_write_vref_p0_pot_scale_sweep.py`: pass.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec/plan conformance `46/46`; final manifest `72/72`, consistency checks `88`, failed `0`; source audit `80/205`; current audit reflected `10`, partial `1`, blocked `1`; operator handoff validation `44/44`; bundle validation `74/74`; remaining blockers stay `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 Final Signoff Runner Self-Consistency Refresh
- Spark sub-agent for Task Card `T-3G-RUNNER-SELF-CONSISTENCY-001` hit usage limit; GPT5.5 fallback evaluator confirmed the stale-contract risk.
- `tools/run_third_goal_final_signoff.py` now refreshes final manifest dependents after the first final evidence manifest pass:
  requirements trace, final operator handoff, handoff validation, bundle validation, and final evidence manifest are regenerated in the same run.
- This removes the previous two-run convergence requirement when final evidence required counts or consistency checks change.
- Regression coverage added in `tests/test_run_third_goal_final_signoff.py`:
  `test_final_refresh_uses_fresh_evidence_manifest_contract` seeds a stale manifest and asserts the final refresh reads the fresh contract.
- `python3 -m unittest tests.test_run_third_goal_final_signoff`: `15` tests passed.
- `python3 -m unittest tests.test_write_vref_p0_pot_scale_audit tests.test_write_vref_p0_pot_scale_sweep tests.test_write_third_goal_source_audit tests.test_write_spec_plan_conformance_audit tests.test_write_hgpipe_operator_audit tests.test_write_third_goal_current_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_requirements_trace tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `66` tests passed.
- `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final runner now emits `final-evidence-manifest`, `requirements-trace-final-refresh`, `final-operator-handoff-final-refresh`, `final-operator-handoff-validation-final-refresh`, `final-signoff-bundle-validation-final-refresh`, and `final-evidence-manifest-final-refresh` in one pass.
- Post-run validators against current `docs/resources` passed:
  `tools/validate_final_operator_handoff.py` status `pass`, fail `0`;
  `tools/validate_final_signoff_bundle.py` status `pass`, fail `0`.

## 2026-06-16 RMU/SMU Small-Memory LUTRAM Final Evidence
- `hls/src/rmu_smu.cpp` now binds the SMU relation-stage `score` and `prob` token scratch arrays to LUTRAM, matching the third-goal small-memory policy while preserving large-buffer URAM policy.
- `tools/write_vref_p0_buffer_lifetime_audit.py` now tracks RMU/SMU small-memory placement with `rmu_smu_small_score_lutram` and `rmu_smu_small_prob_lutram`.
- `tools/write_vref_p0_buffer_lifetime_audit.py` also links QKV successor URAM branch/resource evidence: Q/K/V URAM branch refs, successor CSim/CSynth/routed overlay pass state, BRAM reduction, URAM increase, positive URAM use, and physical-smoke pending-only promotion state.
- `tools/write_final_evidence_manifest.py` now promotes both RMU/SMU LUTRAM checks and QKV successor linkage checks into final evidence consistency checks.
- Latest generated evidence:
  VREF-P0 buffer lifetime audit `pass`, checks `51/51`;
  final evidence manifest `pass`, required `76/76`, consistency checks `170`, failed consistency `0`;
  source audit required `84`, sources `213`, missing `0`.
- Validation:
  `python3 -m unittest tests.test_write_vref_p0_buffer_lifetime_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit`: `35` tests passed.
  `g++ -std=c++17 -Ihls/include -Ihls/src -fsyntax-only hls/src/rmu_smu.cpp`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked only on external blockers; spec/plan conformance `46/46`, final manifest `76/76`, consistency checks `170`, source audit `84/213`.

## 2026-06-16 Req1 Environment Final Evidence
- `tools/write_req1_environment_audit.py` now records Ubuntu Linux and relocated host-path evidence without executing Xilinx tools.
- Latest generated evidence:
  Req1 environment audit `pass`, checks `13/13`;
  `/etc/os-release` reports Ubuntu `22.04`;
  kernel check rejects WSL `microsoft` markers;
  hardware root is under `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware`;
  `/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls` and `/tools/Xilinx/Vivado/2023.2/bin/vivado` exist and are executable.

## 2026-06-16 Req6 Parameterization Legality Guard Evidence
- `tools/write_req6_parameterization_audit.py` now checks HLS-legal macro relationships in addition to knob coverage:
  bus byte alignment, bus/data divisibility, bus/weight divisibility, weight width <= data width, data width < accumulator width,
  model dimension/head count divisibility, head dimension match, FF dimension match, dense parallelism match/divisibility,
  packed weight lanes divisible by dense parallelism, positive buffer size, and positive FIFO depth.
- Negative coverage added for illegal bus width and illegal dense parallelism.
- Validation:
  `python3 -m unittest tests.test_write_req6_parameterization_audit`: `5` tests passed.
  `python3 tools/write_req6_parameterization_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/req6_parameterization_audit_2026_06_16.json --markdown-out generated/signoff/req6_parameterization_audit_2026_06_16.md`: `pass`, checks `62/62`.
  `g++ -std=c++17 -I/tools/Xilinx/Vitis_HLS/2023.2/include -Ihls/include -Ihls/src -fsyntax-only hls/src/hgtxr_e2e_axis_top.cpp`: pass.

## 2026-06-16 Req9 DeiT Image Reference Evidence
- `tools/write_req9_deit_image_reference_audit.py` now promotes the requested `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` reference into JSON/Markdown signoff evidence.
- The audit verifies requested path normalization, PNG existence, non-empty size, PNG magic, SHA256, HGPIPE substitute match, and hardware docs-copy match.
- `tools/run_third_goal_final_signoff.py` regenerates and mirrors the Req9 audit before source/current/final manifest checks.
- Latest generated evidence:
  Req9 audit `pass`, checks `11/11`;
  source audit required `84`, sources `213`, missing `0`;
  final evidence manifest `pass`, required `76/76`, consistency checks `170`, failed consistency `0`.

## 2026-06-16 Direct Hard-Blocker Gate Evidence Promotion
- `tools/write_final_evidence_manifest.py` now requires and checks:
  `xr_vits_gate_audit_2026_06_16.{json,md}` and
  `c3b_physical_smoke_gate_audit_2026_06_16.{json,md}`.
- `tools/run_third_goal_final_signoff.py` now regenerates and mirrors both gate audits before buffer/source/current/final manifest refresh.
- `tools/validate_final_signoff_bundle.py` now requires the two gate-audit artifact pairs and key manifest checks:
  XR-VITs gate status, replacement candidate, candidate recommendation match, C3b gate status, ready-for-board, bundle pass, and session pass.
- Spark sidecar for `T-XRVITS-VERIFY` hit the GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback completed read-only verification and confirmed Req11 cannot be closed without exact `/home/kjm26/project/PRJXR/XR-VITs` restoration or explicit replacement-policy approval.
- Validation:
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_run_third_goal_final_signoff`: `50` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `80/80`, consistency checks `184`, failed consistency `0`; source audit `84/217`; final bundle validation `pass`, fail `0`; operator handoff validation `pass`, fail `0`.
  Remaining blockers are unchanged and external: `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 2026-06-16 C3b Candidate Discovery Metadata-Noise Filter
- `tools/discover_c3b_smoke_candidates.py` now excludes `docs/resources/c3b_physical_smoke_gate_audit_` artifacts from C3b board-smoke candidate discovery.
- Reason: `c3b_physical_smoke_gate_audit_2026_06_16.json` is a gate/status artifact, not a board-produced smoke result JSON, and must not appear as a failed candidate.
- Regression coverage: `tests/test_discover_c3b_smoke_candidates.py` now includes the exact `c3b_physical_smoke_gate_audit_2026_06_16.json` metadata filename in the docs-resource noise skip test.
- Validation:
  `python3 -m unittest tests.test_discover_c3b_smoke_candidates`: `6` tests passed.
  `python3 tools/discover_c3b_smoke_candidates.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/c3b_smoke_candidate_discovery_2026_06_10.json --markdown-out generated/signoff/c3b_smoke_candidate_discovery_2026_06_10.md`: expected `status=missing`, `pass=0`, `candidates=0`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `80/80`, final bundle validation `pass`, C3b smoke candidate discovery `status=missing`, `pass=0`, `candidates=0`.

## 2026-06-16 Generic PYNQ Discovery Gate-Audit Noise Filter
- `tools/discover_pynq_smoke_candidates.py` now excludes `_gate_audit_` filenames across PYNQ smoke presets.
- Reason: generic PYNQ discovery scans `docs/resources` as well as `generated/pynq`; without a name-level guard, C3b physical-smoke gate audit metadata can match `*c3b*smoke*.json` and appear as a failed candidate even though it is not a board result.
- Regression coverage:
  `tests/test_discover_pynq_smoke_candidates.py` covers docs-resource gate-audit noise and generated-signoff metadata noise (`c3b_physical_smoke_gate_audit_*.json`, `zcu104_*_smoke_remote_run_*.json`).
- Validation:
  `python3 -m unittest tests.test_discover_pynq_smoke_candidates tests.test_discover_c3b_smoke_candidates`: `14` tests passed.
  `python3 tools/discover_pynq_smoke_candidates.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --preset axis-c3b-mem16 --json-out /tmp/pynq_c3b_discovery_check.json --markdown-out /tmp/pynq_c3b_discovery_check.md`: expected `status=missing`, `pass=0`, `candidates=0`.

## 2026-06-16 PYNQ Smoke Overlay Provenance Gate
- `tools/validate_pynq_smoke_result.py` now binds each preset to its expected overlay artifact prefix and checks `Path(bitfile).name` / `Path(hwhfile).name` when `require_paths=True`.
- Protected basenames:
  C3b `hgtxr_e2e_axis_dma_c3b_mem16.{bit,hwh}`;
  VREF DSP mixed stream `hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.{bit,hwh}`;
  QKV URAM `hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.{bit,hwh}`;
  A1/C1/m_axi keep their own prefixes.
- Reason: a board JSON with correct numeric output but wrong non-empty overlay paths must not clear C3b/VREF/QKV import or discovery gates.
- Validation:
  `python3 -m unittest tests.test_validate_pynq_smoke_result tests.test_import_pynq_smoke_result tests.test_discover_pynq_smoke_candidates tests.test_discover_c3b_smoke_candidates tests.test_check_final_blocker_closure_readiness`: `39` tests passed.
  `python3 -m unittest tests.test_check_third_goal_preflight tests.test_run_third_goal_final_signoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_source_audit tests.test_write_third_goal_current_audit`: `74` tests passed.
  `python3 -m py_compile tools/validate_pynq_smoke_result.py tools/import_pynq_smoke_result.py tools/discover_pynq_smoke_candidates.py tools/discover_c3b_smoke_candidates.py tools/check_final_blocker_closure_readiness.py tests/test_validate_pynq_smoke_result.py tests/test_import_pynq_smoke_result.py tests/test_discover_pynq_smoke_candidates.py tests/test_discover_c3b_smoke_candidates.py tests/test_check_final_blocker_closure_readiness.py`: pass.
  C3b generic and legacy discovery remain expected-missing with `pass=0`, `candidates=0`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; preflight `ok=96`, `warn=5`, `fail=2`; final manifest `80/80`, consistency checks `184`, failed consistency `0`; source audit `84/217`, missing `0`.

## 2026-06-16 Overlay Provenance Final Manifest Promotion
- `tools/write_final_evidence_manifest.py` now directly checks the PYNQ smoke validator source for:
  `pynq_smoke_validator_overlay_prefix_contracts`,
  `pynq_smoke_validator_basename_check`, and
  `pynq_smoke_validator_require_paths_gate`.
- `tools/validate_final_signoff_bundle.py` now requires those three manifest consistency checks to be present and passing, so a stale final evidence manifest that drops the overlay-provenance contract fails bundle validation.
- `tools/run_third_goal_final_signoff.py` now refreshes spec-plan conformance after the final manifest refresh, then refreshes the manifest again so spec-plan status and hashes are one-run self-consistent.
- Validation:
  `python3 -m unittest tests.test_validate_final_signoff_bundle`: `9` tests passed, including missing PYNQ basename-gate manifest regression.
  `python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_write_spec_plan_conformance_audit tests.test_validate_final_operator_handoff`: `57` tests passed.
  `python3 -m py_compile tools/validate_final_signoff_bundle.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_write_spec_plan_conformance_audit tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `56` tests passed.
  `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tests/test_run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; preflight `ok=96`, `warn=5`, `fail=2`; final manifest `80/80`, consistency checks `187`, failed consistency `0`; spec-plan conformance `46/46`; source audit `84/217`, missing `0`.

## 2026-06-16 Generic PYNQ Discovery Evidence-Chain Promotion
- `tools/run_third_goal_final_signoff.py` now generates and mirrors generic C3b/VREF discovery artifacts:
  `pynq_smoke_candidate_discovery_c3b_2026_06_16.{json,md}` and
  `pynq_smoke_candidate_discovery_vref_p0_2026_06_16.{json,md}`.
- Legacy discovery artifacts remain preserved:
  `c3b_smoke_candidate_discovery_2026_06_10.{json,md}` and
  `vref_successor_smoke_candidate_discovery_2026_06_10.{json,md}`.
- `tools/write_final_evidence_manifest.py` now requires the generic C3b/VREF discovery artifacts and checks their `status` and numeric `pass_count`.
- `tools/validate_final_signoff_bundle.py` now requires those generic discovery artifact ids and final-manifest consistency checks.
- Validation:
  `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tests/test_run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `45` tests passed.

## 2026-06-16 Generic PYNQ Discovery Semantic Gates
- GPT5.5 evaluator found that required generic PYNQ discovery artifacts were hashed but not fully semantically checked, especially QKV URAM discovery.
- `tools/write_final_evidence_manifest.py` now checks generic C3b/VREF/QKV PYNQ discovery payloads for:
  expected preset, status, numeric candidate/pass counts, `pass_count <= candidate_count`, no recommended candidate when missing, and no side effects.
- `tools/validate_final_signoff_bundle.py` now requires those semantic/safety consistency checks.
- `tools/write_third_goal_current_audit.py` now reflects final-runner generic C3b/VREF discovery statuses in the runner summary, falling back to direct discovery artifacts when the runner says `not-run`.
- Validation:
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_write_third_goal_current_audit`: `34` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tools/write_third_goal_current_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_write_third_goal_current_audit.py`: pass.
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_write_spec_plan_conformance_audit`: `53` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tools/write_third_goal_current_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_write_third_goal_current_audit.py tests/test_run_third_goal_final_signoff.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  `git diff --check`: pass.
  Final evidence manifest is `pass`, required artifacts `84/84`, consistency checks `239`, failed consistency `0`.
  Source audit is `pass`, required artifacts `84`, source count `221`, missing required `0`.
  Final bundle validation is `pass`, check count `74`, fail count `0`.
  Final signoff runner remains expected-blocked only on external closure inputs: exact XR-VITs source or approved replacement policy, and board-produced C3b AXIS/DMA physical-smoke JSON.

## 2026-06-16 Blocker Readiness Discovery and Runner Path Evidence
- Spark sidecar for `T-301` hit the GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback completed a read-only audit and recommended aligning blocker-readiness output with generic PYNQ discovery gates.
- `tools/check_final_blocker_closure_readiness.py` now includes side-effect-free `pynq_discovery` summaries for C3b, VREF-P0 DSP mixed stream, and QKV URAM smoke presets.
- `tools/run_third_goal_final_signoff.py` now records `remaining_blocker_input_paths` and `remaining_blocker_details` in the final runner summary, including the C3b canonical smoke JSON, exact XR-VITs path, and approved-policy path.
- Validation:
  `python3 -m unittest tests.test_check_final_blocker_closure_readiness tests.test_run_third_goal_final_signoff tests.test_discover_pynq_smoke_candidates tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `62` tests passed.
  `python3 -m py_compile tools/check_final_blocker_closure_readiness.py tools/run_third_goal_final_signoff.py tests/test_check_final_blocker_closure_readiness.py tests/test_run_third_goal_final_signoff.py`: pass.
  `git diff --check -- tools/check_final_blocker_closure_readiness.py tools/run_third_goal_final_signoff.py tests/test_check_final_blocker_closure_readiness.py tests/test_run_third_goal_final_signoff.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; preflight `ok=96`, `warn=5`, `fail=2`; final manifest `84/84`, consistency checks `239`, failed consistency `0`; source audit `84/221`; final bundle validation `pass`, fail `0`.
  Runner blocker paths now map `C3b AXIS/DMA physical smoke result` to `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` and `requested XR-VITs sibling` to `/home/kjm26/project/PRJXR/XR-VITs` plus `docs/resources/xr_vits_replacement_policy.json`.

## 2026-06-16 Blocker Readiness and Runner Contract Manifest Gates
- `tools/write_final_evidence_manifest.py` now promotes final-runner blocker path support and generated runner blocker path payloads into consistency checks.
- `tools/write_final_evidence_manifest.py` also gates final blocker-readiness C3b/VREF-P0/QKV URAM `pynq_discovery` summaries, including preset, numeric counts, missing-state recommended path behavior, and no-side-effect safety.
- `tools/validate_final_signoff_bundle.py` requires these new manifest checks to be present and passing.
- Validation:
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `36` tests passed.
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff tests.test_check_final_blocker_closure_readiness tests.test_write_third_goal_current_audit tests.test_write_spec_plan_conformance_audit`: `64` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `84/84`, consistency checks `239`, failed consistency `0`; final bundle validation `pass`, fail `0`; source audit `84/221`.

## 2026-06-16 Operator Handoff Live Evidence Contract Gate
- Spark read-only stale-contract audit found that `tools/validate_final_operator_handoff.py` enforced a low consistency threshold but did not compare the embedded handoff evidence contract against live `docs/resources/final_evidence_manifest_2026_06_10.json`.
- `tools/validate_final_operator_handoff.py` now requires:
  `final_evidence_manifest_contract.consistency_count >= 239`,
  `failed_consistency_checks == []`,
  live manifest contract present,
  and exact embedded/live contract equality.
- `tools/validate_final_signoff_bundle.py` also keeps the final evidence consistency threshold at `>=239`.
- Regression coverage rejects stale embedded consistency count `238` and missing live evidence manifest.
- Validation:
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit`: `25` tests passed.
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `84/84`, consistency checks `239`, failed consistency `0`; operator handoff validation `pass`, checks `46`, fail `0`; final bundle validation `pass`, checks `74`, fail `0`; spec-plan conformance `46/46`; remaining blockers are still external C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Req6 PAR16/PAR32 Sweep Extension
- `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml` now includes `parallelism_factor: [1, 2, 4, 8, 16, 32]`.
- C3b PAR16 is recorded as the validated board-smoke candidate; PAR32 is recorded as exploratory and not promotable without fresh csynth, routed timing, resource-fit audit, and no C3b artifact overwrite.
- `tools/write_req6_parameterization_audit.py` now verifies that both CSim and CSynth Tcl scripts support PAR16/PAR32 overrides.
- `tools/write_final_evidence_manifest.py` promotes those PAR16/PAR32 Tcl override checks into final evidence consistency checks.
- Validation:
  `python3 -m unittest tests.test_write_req6_parameterization_audit tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit`: `57` tests passed.
  `python3 -m py_compile tools/write_req6_parameterization_audit.py tools/write_final_evidence_manifest.py tests/test_write_req6_parameterization_audit.py tests/test_write_final_evidence_manifest.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 tools/write_req6_parameterization_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: `pass`, checks `64/64`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `84/84`, consistency checks `241`, failed consistency `0`; Req6 audit `64/64`; final bundle validation `pass`, checks `74`, fail `0`; remaining blockers are still external C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Blocker Readiness No-Require-Paths Alignment
- `tools/check_final_blocker_closure_readiness.py` now applies `--c3b-no-require-paths` semantics to current canonical C3b smoke validation, not only candidate validation.
- Strict mode remains unchanged: current canonical C3b validation still requires preset-matched bit/hwh path fields unless the relaxed flag is explicitly used.
- Regression coverage verifies that a path-less canonical C3b JSON clears current readiness only when `c3b_require_paths=False`, and remains blocked in strict mode.
- Validation:
  `python3 -m unittest tests.test_check_final_blocker_closure_readiness`: `8` tests passed.
  `python3 -m unittest tests.test_check_final_blocker_closure_readiness tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `44` tests passed.
  `python3 -m py_compile tools/check_final_blocker_closure_readiness.py tests/test_check_final_blocker_closure_readiness.py`: pass.
  `git diff --check -- tools/check_final_blocker_closure_readiness.py tests/test_check_final_blocker_closure_readiness.py docs/CHOICE.md docs/Validation.md docs/track/log.md`: pass.
  `python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-no-require-paths --json-out /tmp/hgtxr_final_blocker_readiness_current.json --markdown-out /tmp/hgtxr_final_blocker_readiness_current.md`: expected `blocked`; current canonical C3b JSON and XR-VITs/policy are still missing.

## 2026-06-16 C3b Transfer Manifest Runner Integration
- `tools/run_third_goal_final_signoff.py` now invokes `tools/write_c3b_smoke_transfer_manifest.py` as `c3b-smoke-transfer-manifest`.
- The final runner now regenerates and mirrors:
  `c3b_smoke_transfer_manifest_2026_06_10.{json,md}` and
  `e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`.
- The final runner summary now exposes `c3b_transfer_manifest_status`; current value is `pass`.
- Validation:
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_c3b_smoke_transfer_manifest`: `18` tests passed.
  `python3 -m unittest tests.test_check_final_blocker_closure_readiness tests.test_run_third_goal_final_signoff tests.test_write_c3b_smoke_transfer_manifest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle`: `62` tests passed.
  `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_c3b_smoke_transfer_manifest.py tests/test_run_third_goal_final_signoff.py tests/test_write_c3b_smoke_transfer_manifest.py`: pass.
  `python3 tools/write_c3b_smoke_transfer_manifest.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`: pass, tar sha256 `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`, errors `0`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; `c3b_transfer_manifest_status=pass`, final manifest `84/84`, consistency checks `241`, failed consistency `0`, operator handoff validation `46/46`, bundle validation `74/74`.

## 2026-06-16 C3b Transfer Manifest Semantic Gates
- `tools/write_final_evidence_manifest.py` now checks C3b transfer-manifest semantics instead of only requiring artifact presence/hash.
- Added consistency gates for loaded JSON type, pass status, preset/variant/target, tar shape, transfer files, clean bundle validation errors, sha256 line match, expected runtime/output, board verify/run commands, host copyback command contract, sha256-file format, readiness sha match, and bundle/readiness tar alignment.
- `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency count `>=256`.
- Validation:
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_spec_plan_conformance_audit`: `52` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tools/validate_final_operator_handoff.py tools/write_spec_plan_conformance_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_validate_final_operator_handoff.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; `c3b_transfer_manifest_status=pass`, final manifest `84/84`, consistency checks `256`, failed consistency `0`, spec-plan conformance `46/46`, operator handoff validation `46/46`, bundle validation `74/74`.

## 2026-06-16 Req6 PAR32 Promotion-Policy Gates
- `tools/write_req6_parameterization_audit.py` now parses the sweep YAML and checks the `parallelism_extensions` contract.
- Added Req6 checks for C3b PAR16 `validated_resource_matrix` status, C3b resource-matrix evidence/result tuple, PAR32 `exploratory_not_default` status, PAR32 fresh-report promotion rule, and PAR32 resource/timing risk record.
- `tools/write_final_evidence_manifest.py` now promotes those Req6 parallelism-extension checks into final evidence consistency gates.
- `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency count `>=263`.
- Validation:
  `python3 -m unittest tests.test_write_req6_parameterization_audit tests.test_write_final_evidence_manifest`: `32` tests passed.
  `python3 -m py_compile tools/write_req6_parameterization_audit.py tools/write_final_evidence_manifest.py tests/test_write_req6_parameterization_audit.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; Req6 audit `71/71`, final manifest `84/84`, consistency checks `263`, failed consistency `0`, spec-plan conformance `46/46`, operator handoff validation `46/46`, bundle validation `74/74`; remaining blockers are still external C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 XR-VITs Policy Preview Path Contract
- `tools/write_final_evidence_manifest.py` now checks the XR-VITs replacement-policy preview active policy path, tracked candidate-audit relative path, and integrity candidate-audit absolute path.
- `tools/validate_final_signoff_bundle.py` now requires those preview path-contract consistency checks to be present/pass.
- `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency count `>=266`.
- Validation:
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `48` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `84/84`, consistency checks `266`, failed consistency `0`, operator handoff validation `46/46`, bundle validation `74/74`; spec-plan conformance pending current-doc refresh at this step.

## 2026-06-16 XR-VITs Policy Approval Event Contract
- `tools/create_xr_vits_replacement_policy.py` now emits `approval_event` with fixed protocol, event type, approver, approved-at, reason code, reason, replacement role, requested path, replacement path, candidate-audit path, candidate-audit fingerprint, generator, and event-id hash.
- `tools/write_final_evidence_manifest.py` now checks preview approval-event schema, policy-field match, and event-id hash.
- `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency count `>=269`.
- Validation:
  `python3 -m unittest tests.test_create_xr_vits_replacement_policy tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `57` tests passed.
  `python3 -m py_compile tools/create_xr_vits_replacement_policy.py tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tools/validate_final_operator_handoff.py tools/write_third_goal_requirements_trace.py tools/write_final_unblock_commands.py tools/validate_final_unblock_closeout_packet.py tools/write_third_goal_current_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `84/84`, consistency checks `269`, failed consistency `0`; remaining blockers are still external C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 XR-VITs Active Approval Placeholder Guard
- `tools/create_xr_vits_replacement_policy.py` now rejects placeholder approver values such as `<approved-by>` for active policy writes.
- Dry-run preview remains placeholder-tolerant so review artifacts can still be generated without writing an active replacement policy.
- `tools/run_third_goal_final_signoff.py` now treats active XR-VITs replacement-policy creation failure as an overall blocked final-runner state.
- `tools/write_final_evidence_manifest.py` now checks the source contract for the active placeholder guard, dry-run-only allowance, and final-runner active-policy-failure blocking.
- `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency count `>=272`.
- Regression coverage verifies active placeholder rejection, no active policy file creation on failure, dry-run placeholder allowance, and final-runner blocked summary when active policy creation fails.
- Validation:
  `python3 -m unittest tests.test_create_xr_vits_replacement_policy tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit`: `80` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py`: pass.
  `git diff --check -- docs/CHOICE.md docs/Spec.md docs/Master-Plan.md docs/Sub-Plan.md docs/Validation.md docs/track/PROGRESS.md docs/track/HANDOVER.md docs/track/log.md tools/create_xr_vits_replacement_policy.py tools/run_third_goal_final_signoff.py tests/test_create_xr_vits_replacement_policy.py tests/test_run_third_goal_final_signoff.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; preflight `ok=96`, `warn=5`, `fail=2`; final manifest `84/84`, consistency checks `272`, failed consistency `0`; spec-plan conformance `46/46`; final operator handoff validation `pass`; final bundle validation `pass`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Spec Fallback Final Evidence Gate
- `tools/write_final_evidence_manifest.py` now consumes `spec_plan_conformance_audit_2026_06_10.json` and checks that it proves manual Spec fallback while `spec-kit`/`specify` are unavailable.
- Final evidence now requires spec-plan proof for ZCU104, Q4W/Q8A, parameter knobs, A2/A1/C/PAR16 continuity, selected path records, and `/tools/Xilinx`.
- `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency count `>=278`.
- Validation:
  `python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit`: `54` tests passed.
  `python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; preflight `ok=96`, `warn=5`, `fail=2`; final manifest `84/84`, consistency checks `278`, failed consistency `0`; spec-plan conformance `46/46`; final operator handoff validation `pass`; final bundle validation `pass`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Runner Mirrored Artifact Dedupe
- `tools/run_third_goal_final_signoff.py` now writes `mirrored_artifacts` as an order-preserving unique list.
- The final summary is also deduped again after adding the summary JSON/Markdown mirrors.
- Regression coverage checks that the list length equals the unique set length and that `final_evidence_manifest_2026_06_10.json` appears once.
- Validation:
  `python3 -m unittest tests.test_run_third_goal_final_signoff`: `17` tests passed.
  `python3 -m py_compile tools/run_third_goal_final_signoff.py tests/test_run_third_goal_final_signoff.py`: pass.
  `git diff --check -- tools/run_third_goal_final_signoff.py tests/test_run_third_goal_final_signoff.py docs/CHOICE.md docs/track/log.md`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; preflight `ok=96`, `warn=5`, `fail=2`; final manifest `84/84`, consistency checks `278`, failed consistency `0`; `mirrored_artifacts` `90/90` unique, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy. Current summary has advanced to `92/92` unique after later evidence artifacts were added.

## 2026-06-16 P2-ViT Scale Calibration Final Evidence
- `tools/write_p2_vit_scale_calibration_report.py` combines Req5 Q4/Q8 SW-HW match evidence with VREF-P0-01 PoT scale readiness/sweep evidence.
- The report records the current decision: keep current PoT scales for C3b; non-current candidates remain successor-only until header regeneration, CSim, HLS, and board evidence.
- `tools/run_third_goal_final_signoff.py` now regenerates and mirrors the report; source audit and final evidence manifest require the JSON/Markdown artifacts.
- `tools/write_final_evidence_manifest.py` checks report pass status, Q4W/Q8A precision, current-scale decision, candidate coverage, and no-side-effect safety.
- Validation:
  `python3 -m unittest tests.test_write_p2_vit_scale_calibration_report tests.test_write_third_goal_source_audit tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff`: `51` tests passed.
  `python3 -m py_compile tools/write_p2_vit_scale_calibration_report.py tools/run_third_goal_final_signoff.py tools/write_third_goal_source_audit.py tools/write_final_evidence_manifest.py tests/test_write_p2_vit_scale_calibration_report.py tests/test_run_third_goal_final_signoff.py tests/test_write_third_goal_source_audit.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; P2 report `pass`, checks `10/10`; final manifest `86/86`, consistency checks `283`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Unblock Intake/Root Command Hardening
- `tools/write_final_unblock_intake.py` now emits `blocker_status` and `next_inputs` so the two final external inputs are explicit in machine-readable JSON:
  C3b canonical board-smoke JSON and exact XR-VITs source or approved replacement policy.
- `tools/check_xr_vits_reference_resolution.py` now emits `--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` in exact-restore, dry-run replacement, and active replacement approval commands.
- `tools/audit_final_unblock_candidates.py` now calls the XR-VITs policy validator with the current dry-run approval metadata contract.
- Validation:
  `python3 -m unittest tests.test_check_xr_vits_reference_resolution tests.test_write_final_operator_handoff tests.test_write_final_unblock_intake tests.test_check_final_blocker_closure_readiness tests.test_audit_final_unblock_candidates tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_write_third_goal_source_audit`: `70` tests passed.
  `python3 -m py_compile tools/check_xr_vits_reference_resolution.py tools/write_final_operator_handoff.py tools/write_final_unblock_intake.py tools/audit_final_unblock_candidates.py tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tools/write_third_goal_source_audit.py tests/test_check_xr_vits_reference_resolution.py tests/test_write_final_operator_handoff.py tests/test_write_final_unblock_intake.py tests/test_check_final_blocker_closure_readiness.py tests/test_audit_final_unblock_candidates.py tests/test_run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py tests/test_write_third_goal_source_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec-plan conformance `46/46`; final manifest `86/86`, consistency checks `283`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Unblock Intake/Operator Handoff Live Gates
- `tools/write_final_evidence_manifest.py` now gates final unblock intake semantics:
  `final_unblock_intake_blocker_status_keys`,
  `final_unblock_intake_c3b_next_input_path`,
  `final_unblock_intake_xr_vits_next_input_path`,
  and `final_unblock_intake_next_inputs_match_blockers`.
- `tools/validate_final_operator_handoff.py` now loads live `xr_vits_reference_resolution_2026_06_10.json` and `final_unblock_intake_2026_06_10.json`, then compares them against embedded operator handoff state through:
  `live_xr_vits_reference_resolution_present`,
  `xr_resolution_embedded_matches_live`,
  `xr_resolution_live_status_matches_blocker`,
  `live_final_unblock_intake_present`,
  `handoff_remaining_blockers_match_intake`,
  and `handoff_candidate_states_match_intake`.
- `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require operator handoff validation `check_count >= 52`.
- Verification:
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_write_third_goal_current_audit tests.test_write_third_goal_completion_audit`: `88` tests passed.
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/write_final_evidence_manifest.py tools/validate_final_signoff_bundle.py tests/test_validate_final_operator_handoff.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_signoff_bundle.py tests/test_write_third_goal_current_audit.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final operator handoff validation `pass`, checks `52`, fail `0`; final bundle validation `pass`, checks `74`, fail `0`; final manifest `86/86`, consistency checks `287`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Current-Doc Freshness Gate
- `tools/write_spec_plan_conformance_audit.py` now includes `docs/Master-Plan.md`, `docs/Sub-Plan.md`, `docs/Spec.md`, `docs/Validation.md`, `docs/track/PROGRESS.md`, `docs/track/HANDOVER.md`, `docs/CHOICE.md`, and `docs/track/log.md` in live current-doc freshness checks.
- `tools/write_final_evidence_manifest.py` now requires spec-plan conformance `check_count >= 66` and keeps `spec_plan_current_doc_freshness_contract`.
- Current evidence target:
  final manifest `86/86`, consistency checks `302`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.
- Verification:
  `python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_write_final_evidence_manifest`: `40` tests passed.
  `python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_third_goal_completion_audit`: `118` tests passed.
  `python3 -m py_compile tools/write_spec_plan_conformance_audit.py tools/write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec-plan conformance `66/66`; final manifest `86/86`, consistency checks `302`, failed consistency `0`; operator handoff validation `52/52`; bundle validation `74/74`; source audit required `86`, sources `224`, missing `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 RMU/SMU DSP Helper Resource Gate
- `hls/src/rmu_smu.cpp` now routes RMU projection and SMU relation-stage multiply-heavy paths through DSP-bound helper functions with `#pragma HLS bind_op ... impl=dsp`.
- `tools/write_e2e_resource_policy_audit.py` now checks RMU/SMU DSP helper definitions, bind-op count, RMU projection helper use, and SMU relation helper use.
- `tools/write_final_evidence_manifest.py` now requires resource-policy `check_count >= 27` and gates the RMU/SMU DSP helper checks.
- Verification:
  `python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest`: `35` tests passed.
  `python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_third_goal_completion_audit tests.test_write_vref_p0_buffer_lifetime_audit`: `115` tests passed.
  `g++ -std=c++17 -Ihls/include -Ihls/src -fsyntax-only hls/src/rmu_smu.cpp`: pass.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; resource-policy audit `27/27`; final manifest `86/86`, consistency checks `302`, failed consistency `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Req5 Q4 Packed Weight Integrity Gate
- `tools/write_req5_q4q8_swhw_match_audit.py` now validates the packed Q4 binary referenced by the manifest:
  binary exists, actual SHA256 matches manifest SHA256, byte count matches manifest bytes, expected runtime state is `2`, and expected C3b raw output is `[32, -13, 26, -6, 14, -11]`.
- `tools/write_final_evidence_manifest.py` now gates the Req5 packed-weight contract directly:
  packed manifest, binary existence, SHA256 match, byte-count match, expected runtime state, expected C3b output, strict testbench golden compare, C3b AXIS CSim, VREF softmax-input-x2 CSim, and QKV URAM CSim.
- Validation:
  `python3 -m unittest tests.test_write_req5_q4q8_swhw_match_audit tests.test_write_final_evidence_manifest`: `37` tests passed.
  `python3 -m unittest tests.test_write_req5_q4q8_swhw_match_audit tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_p2_vit_scale_calibration_report`: `121` tests passed.
  `python3 -m py_compile tools/write_req5_q4q8_swhw_match_audit.py tools/write_final_evidence_manifest.py tests/test_write_req5_q4q8_swhw_match_audit.py tests/test_write_final_evidence_manifest.py`: pass.
  Direct Req5 audit status: `pass`, checks `11/11`, fail `0`.
  Current target: final manifest `86/86`, consistency checks `312`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.

## 2026-06-16 Final Evidence Consistency Threshold Refresh
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require final evidence consistency count `>=312`, matching the live final evidence manifest after Req5 packed-weight integrity gates.
- Regression fixtures now build live/embedded evidence contracts with `312` consistency checks and reject stale `311` embedded contracts.
- Current target: final manifest `86/86`, consistency checks `312`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.

## 2026-06-16 Req5 Final Bundle Required Checks
- `tools/validate_final_signoff_bundle.py` now requires all Req5 Q4/Q8 packed-weight final-manifest consistency checks to be present and passing:
  SW/HW match, Q4W/Q8A precision, packed-weight manifest, binary existence, SHA256 match, byte-count match, expected runtime state, expected C3b output, strict testbench golden compare, C3b AXIS CSim, VREF CSim, and QKV URAM CSim.
- Regression coverage removes `req5_q4q8_packed_weight_sha256_matches_manifest` from a live manifest and expects bundle validation to fail.
- Current target: final manifest `86/86`, consistency checks `312`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.

## 2026-06-16 Req5 Operator Handoff Required Checks
- `tools/validate_final_operator_handoff.py` now loads the live final evidence manifest and requires all Req5 Q4/Q8 packed-weight consistency checks to be present and passing.
- `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=54`, so a stale `52`-check validation cannot satisfy final evidence.
- Regression coverage removes `req5_q4q8_packed_weight_sha256_matches_manifest` while preserving total consistency count `312`; operator handoff validation still fails on the missing Req5 check.
- Validation:
  `python3 -m unittest tests.test_validate_final_operator_handoff`: `12` tests passed.
  `python3 -m py_compile tools/validate_final_operator_handoff.py tests/test_validate_final_operator_handoff.py`: pass.
  Expected current target after final runner refresh: operator handoff validation `54/54`; final manifest `86/86`, consistency checks `312`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.

## 2026-06-16 Bundle Operator Handoff Count Freshness
- `tools/validate_final_signoff_bundle.py` now requires final operator handoff validation `check_count >=54`, matching the latest Req5 live-manifest checks.
- Regression coverage writes a stale operator handoff validation artifact with `status=pass`, `fail_count=0`, and `check_count=52`; final bundle validation fails on `handoff_validation_check_count`.
- Validation:
  `python3 -m py_compile tools/validate_final_signoff_bundle.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_final_evidence_manifest`: `60` tests passed.
  `python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_final_evidence_manifest tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff tests.test_write_req5_q4q8_swhw_match_audit`: `89` tests passed.
  `python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/hgtxr_spec_plan_n18.json --markdown-out /tmp/hgtxr_spec_plan_n18.md`: `pass`, checks `66/66`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; bundle validation `74/74`; operator handoff validation `54/54`; final manifest `86/86`, consistency checks `312`, failed consistency `[]`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Cyclic Baseline URAM/LUTRAM Placement Gate
- `hls/include/hgtxr_cyclic_transformer_params.hpp` now exposes parameterized storage-policy macros:
  `HGTXR_CYCLIC_FORCE_URAM_WEIGHT_TILES`, `HGTXR_CYCLIC_FORCE_URAM_LARGE_TEMPS`, and `HGTXR_CYCLIC_SMALL_TILE_LUTRAM`.
- `hls/src/hgtxr_top.cpp` now applies those macros to legacy cyclic baseline storage:
  packed block weight tiles `wq/wk/wv/wo/w1/w2` bind to URAM, large temporaries `attn_tiles/hidden_tiles/residual0/residual1` bind to URAM, and small tile scratch buffers bind to LUTRAM when enabled.
- `tools/write_e2e_resource_policy_audit.py` now gates those cyclic source contracts; target resource-policy audit is `36/36` after the C3b threshold gates.
- `tools/write_final_evidence_manifest.py` now requires resource-policy `check_count >=36` and gates the cyclic URAM/LUTRAM checks directly.
- Validation:
  `g++ -std=c++17 -Ihls/include -Ihls/src -fsyntax-only hls/src/hgtxr_top.cpp`: pass.
  `python3 -m py_compile tools/write_e2e_resource_policy_audit.py tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_e2e_resource_policy_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `63` tests passed.
  `python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff`: `88` tests passed.
  `python3 tools/write_e2e_resource_policy_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/hgtxr_resource_policy_c3b_thresholds.json --markdown-out /tmp/hgtxr_resource_policy_c3b_thresholds.md`: `pass`, checks `36/36`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; resource-policy audit `36/36`; final manifest `86/86`, consistency checks `321`, failed consistency `[]`; operator handoff validation target now `58/58`; bundle validation `74/74`; spec-plan conformance `66/66`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Cyclic Resource Named-Check Validator Gate
- `tools/validate_final_operator_handoff.py` now requires all six cyclic resource-policy final-manifest consistency checks to be present and passing in the live final evidence manifest.
- `tools/validate_final_signoff_bundle.py` now includes those same cyclic resource-policy consistency checks in the final-manifest required named-check set.
- `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=58`, and final bundle validation also requires handoff validation `check_count >=58`.
- Regression coverage removes `resource_policy_audit_cyclic_weight_tiles_uram_pragmas` while preserving total consistency count `321`; operator handoff and final bundle validation fail on the missing named cyclic resource check.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `62` tests passed.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff`: `90` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; operator handoff validation `58/58`; final bundle validation `74/74`; final manifest `86/86`, consistency checks `321`, failed consistency `[]`; spec-plan conformance `66/66`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 C3b Threshold Resource-Policy Gate
- `tools/write_e2e_resource_policy_audit.py` now requires the selected C3b baseline to preserve:
  csynth LUT `<=126506`, csynth latency `<=37508072`, and routed WNS `>=4.415 ns`.
- `tools/write_final_evidence_manifest.py` now promotes those three C3b threshold checks into final evidence consistency gates and requires resource-policy audit `check_count >=36`.
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require final evidence consistency count `>=321`.
- Regression coverage patches threshold constants to prove C3b LUT, latency, and WNS threshold drift fails the resource-policy audit.
- Validation:
  `python3 tools/write_e2e_resource_policy_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/hgtxr_resource_policy_c3b_thresholds.json --markdown-out /tmp/hgtxr_resource_policy_c3b_thresholds.md`: `pass`, checks `36/36`.
  `python3 -m py_compile tools/write_e2e_resource_policy_audit.py tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_e2e_resource_policy_audit.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_spec_plan_conformance_audit.py`: pass.
  Focused validation before document refresh: `python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `66` tests passed.
  Post-refresh validation: `python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff`: `91` tests passed.
  `python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/hgtxr_spec_plan_c3b_threshold_pre.json --markdown-out /tmp/hgtxr_spec_plan_c3b_threshold_pre.md`: `pass`, checks `66/66`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; resource-policy audit `36/36`; final manifest `86/86`, consistency checks `321`, failed consistency `[]`; operator handoff validation `58/58`; bundle validation `74/74`; spec-plan conformance `66/66`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 C3b Threshold Named-Check Validator Gate
- `tools/validate_final_operator_handoff.py` now requires the three C3b threshold final-manifest consistency checks to be present and passing in the live final evidence manifest:
  `resource_policy_audit_c3b_csynth_lut_lte_threshold`,
  `resource_policy_audit_c3b_csynth_latency_lte_threshold`,
  and `resource_policy_audit_c3b_routed_wns_gte_threshold`.
- `tools/validate_final_signoff_bundle.py` now includes the same C3b threshold checks in the final-manifest required named-check set.
- `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=58`.
- Regression coverage removes `resource_policy_audit_c3b_routed_wns_gte_threshold` while preserving total consistency count `321`; operator handoff and final bundle validation fail on the missing named C3b threshold check.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `64` tests passed.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff`: `93` tests passed.
  `python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/hgtxr_spec_plan_c3b_threshold_named_after_runner.json --markdown-out /tmp/hgtxr_spec_plan_c3b_threshold_named_after_runner.md`: `pass`, checks `66/66`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; resource-policy audit `36/36`; final manifest `86/86`, consistency checks `321`, failed consistency `[]`; operator handoff validation `58/58`; bundle validation `74/74`; spec-plan conformance `66/66`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Runner Summary Count Schema Gate
- `tools/run_third_goal_final_signoff.py` now writes explicit summary count fields:
  `blocker_count`, `remaining_blocker_detail_count`, `mirrored_artifact_count`,
  `mirrored_artifact_unique_count`, and `mirrored_artifact_duplicate_count`.
- `tools/write_final_evidence_manifest.py` now checks those fields against the actual runner summary lists/maps, so count-preserving drift in blocker or mirror evidence fails final evidence.
- Regression coverage mutates blocker/detail/mirror counts and expects final evidence failure on:
  `final_runner_blocker_count_matches_list`,
  `final_runner_remaining_blocker_detail_count_matches`,
  and `final_runner_mirrored_artifact_counts_match`.
- Validation:
  `python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tests/test_run_third_goal_final_signoff.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest`: `51` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff`: `111` tests passed.
  `python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out /tmp/hgtxr_spec_plan_runner_summary_schema_after.json --markdown-out /tmp/hgtxr_spec_plan_runner_summary_schema_after.md`: `pass`, checks `66/66`.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `325`, failed consistency `[]`; operator handoff validation `60/60`; bundle validation `74/74`; spec-plan conformance `66/66`; runner blocker count `2`, blocker detail count `2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Runner Summary Named-Check Validator Gate
- Spark sidecar spawn for `T-3G-RUNNER-SUMMARY-NAMED-CHECK-SIDECAR-013` failed with `agent thread limit reached`; main agent completed the bounded validator implementation.
- `tools/validate_final_operator_handoff.py` now requires the final-runner summary count checks to be present and passing in the live final evidence manifest:
  `final_runner_remaining_blockers_present`,
  `final_runner_blocker_count_matches_list`,
  `final_runner_remaining_blocker_detail_count_matches`,
  and `final_runner_mirrored_artifact_counts_match`.
- `tools/validate_final_signoff_bundle.py` now includes the same runner-summary checks in its required final-manifest named-check set.
- `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=60`.
- Regression coverage preserves final evidence consistency count `325` while removing `final_runner_mirrored_artifact_counts_match`; operator handoff and bundle validation fail on the missing named runner-summary check.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `67` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit`: `96` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `325`, failed consistency `[]`; operator handoff validation `60/60`; bundle validation `74/74`; spec-plan conformance `66/66`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Unblock Closeout Validation Freshness Gate
- Spark sidecar spawn for `T-3G-NEXT-INTERNAL-GAP-AUDIT-014` failed with `agent thread limit reached`; main agent selected the closeout-validation freshness gap from current evidence.
- `generated/signoff/final_unblock_closeout_packet_validation_2026_06_10.json` is current at `49/49`, fail `0`.
- `tools/write_final_evidence_manifest.py` now requires `final_unblock_closeout_validation_check_count >=49` instead of the stale `>=40` threshold.
- Regression coverage mutates closeout validation to `check_count=48`; final evidence fails on `final_unblock_closeout_validation_check_count`.
- Validation:
  `python3 -m py_compile tools/write_final_evidence_manifest.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_write_final_evidence_manifest`: `34` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit`: `135` tests passed.

## 2026-06-16 Third-Goal Source Audit Freshness Gate
- Spark sidecar spawn for `T-3G-NEXT-EVIDENCE-GAP-AUDIT-015` failed with `agent thread limit reached`; main agent selected the source-audit freshness gap from current evidence.
- `generated/signoff/third_goal_source_audit_2026_06_16.json` is current at required `86`, sources `224`, missing `0`.
- `tools/write_final_evidence_manifest.py` now requires `third_goal_source_audit_required_count >=86` and `third_goal_source_audit_source_count >=224`.
- Regression coverage mutates the source audit to required `85` and sources `224`; final evidence fails on `third_goal_source_audit_required_count` and `third_goal_source_audit_source_count`.
- Validation:
  `python3 -m py_compile tools/write_final_evidence_manifest.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_write_final_evidence_manifest`: `35` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `140` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `60/60`; bundle validation `74/74`; spec-plan conformance `66/66`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Third-Goal Source Audit Named-Check Validator Gate
- Spark sidecar spawn for `T-3G-NEXT-GAP-AUDIT-016` failed with `agent thread limit reached`; main agent selected the source-audit named-check validator gap.
- `tools/validate_final_operator_handoff.py` now requires `third_goal_source_audit_required_count` and `third_goal_source_audit_source_count` to be present and passing in the live final evidence manifest.
- `tools/validate_final_signoff_bundle.py` now includes the same source-audit freshness checks in the required final-manifest named-check set.
- `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=62`.
- Regression coverage preserves final evidence consistency count `332` while removing `third_goal_source_audit_source_count`; operator handoff and bundle validation fail on the missing named source-audit check.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `70` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `142` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `62/62`; bundle validation `74/74`; spec-plan conformance `66/66`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Evidence Consistency Threshold Refresh
- Spark sidecar spawn for `T-3G-NEXT-GAP-AUDIT-017` failed with `agent thread limit reached`; main agent selected the stale minimum-consistency threshold gap.
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require final evidence consistency count `>=332`, matching the live final evidence manifest.
- Regression coverage mutates the operator handoff evidence contract to consistency `325`; operator handoff validation fails on `evidence_manifest_consistency_count`.
- Bundle regression coverage writes matching trace/handoff/live final evidence contracts at consistency `325`; bundle validation fails on `evidence_contract_consistency_count` while contract equality checks remain pass.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `36` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `143` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; operator handoff validation `62/62`; bundle validation `74/74`; spec-plan conformance `66/66`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final Evidence Required-Count Threshold Refresh
- Spark sidecar spawn for `T-3G-REQUIRED-COUNT-GAP-AUDIT-018` failed with `agent thread limit reached`; main agent selected the stale minimum-required-count threshold gap.
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require final evidence required count `>=86`, matching the live final evidence manifest.
- Regression coverage mutates the operator handoff evidence contract to required/present `85/85`; operator handoff validation fails on `evidence_manifest_required_complete`.
- Bundle regression coverage writes matching trace/handoff/live final evidence contracts at required/present `85/85`; bundle validation fails on `evidence_contract_required_complete` while contract equality checks remain pass.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `38` tests passed.

## 2026-06-16 Third-Goal Source Audit Direct Validator Gate
- Spark sidecar spawn for `T-3G-N30-GAP-AUDIT` failed with `agent thread limit reached`; main agent selected the direct source-audit validator gap.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/third_goal_source_audit_2026_06_16.json` and requires status `pass`, required count `>=86`, and source count `>=223`.
- `tools/validate_final_signoff_bundle.py` now includes `source_audit` in the bundle source set and directly requires status `pass`, required count `>=86`, source count `>=223`, and `missing_required == []`.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=66` and final signoff bundle validation `check_count >=80`.
- Regression coverage writes stale live source-audit counts `85/223` while leaving final-manifest source-audit named checks intact; both operator handoff and bundle validation fail on direct live source-audit checks.
- Validation:
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit`: `115` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `66/66`; bundle validation `80/80`; spec-plan conformance `66/66`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Resource Policy Direct Validator Gate
- Spark sidecar spawn for `T-3G-N31-RESOURCE-POLICY-DIRECT` failed with `agent thread limit reached`; main agent selected the direct resource-policy validator gap.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/e2e_resource_policy_audit_2026_06_10.json` and requires status `pass`, fail count `0`, check count `>=36`, and core DSP/URAM/LUTRAM/C3b checks present/pass.
- `tools/validate_final_signoff_bundle.py` now includes `resource_policy` in the bundle source set and directly requires the same live resource-policy status/count/core-check contract.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=72` and final signoff bundle validation `check_count >=87`.
- Regression coverage writes stale live resource-policy count `35` and a failed `c3b_csynth_lut_lte_threshold` core check while leaving final-manifest resource-policy named checks intact; both operator handoff and bundle validation fail on direct live resource-policy checks.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit`: `119` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `72/72`; bundle validation `87/87`; spec-plan conformance `66/66`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Spec-Plan Direct Validator Gate
- Spark sidecar spawn for `T-3G-N32-SPEC-PLAN-DIRECT` failed with `agent thread limit reached`; main agent selected the direct spec-plan validator gap.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/spec_plan_conformance_audit_2026_06_10.json` and requires status `pass`, check count `>=84`, current observed counts, no side effects, and core plan/spec/current-doc checks present/pass.
- `tools/validate_final_signoff_bundle.py` now includes `spec_plan` in the bundle source set and directly requires the same live spec-plan status/count/core-check contract.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=79` and final signoff bundle validation `check_count >=95`.
- Regression coverage writes stale live spec-plan count `65` and failed core spec-plan checks while leaving final-manifest spec-plan named checks intact; both operator handoff and bundle validation fail on direct live spec-plan checks.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit`: `123` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `155` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; spec-plan conformance `66/66`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `79/79`; bundle validation `95/95`; runner blocker/detail counts `2/2`; mirrored artifacts `92/92`, duplicates `0`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Spec-Plan Validator-Count Current-Doc Gate
- Spark sidecar spawn for `T-3G-N33-INTERNAL-GAP-AUDIT` failed with `agent thread limit reached`; main agent selected the current-doc validator-count freshness gap.
- `tools/write_spec_plan_conformance_audit.py` now reads live final operator handoff validation and final signoff bundle validation JSON and checks that all current docs record operator handoff validation `79/79` and bundle validation `95/95`.
- `tools/write_final_evidence_manifest.py`, `tools/validate_final_operator_handoff.py`, and `tools/validate_final_signoff_bundle.py` now require spec-plan conformance `check_count >=84`.
- Validation:
  `python3 -m py_compile tools/write_spec_plan_conformance_audit.py tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py`: pass.
  `python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest`: `92` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `156` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec-plan conformance `86/86`; final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `79/79`; bundle validation `95/95`; runner blockers `2`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Final-Runner Direct Validator Gate
- Spark sidecar spawn for `T-3G-N34-FINAL-RUNNER-DIRECT` failed with `agent thread limit reached`; main agent selected the direct final-runner JSON validator gap.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/third_goal_final_signoff_run_2026_06_10.json` and requires blocked status, two expected blockers, two blocker-detail records, and nonduplicated mirrored-artifact counts.
- `tools/validate_final_signoff_bundle.py` now includes `final_runner` in the bundle source set and directly requires the same live final-runner status/count/mirror contract.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=85` and final signoff bundle validation `check_count >=102`.
- Regression coverage writes stale live final-runner blocker/detail/mirror counts while preserving final-manifest runner checks; both operator handoff and bundle validation fail on direct live final-runner checks.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle`: `52` tests passed.
  Expected current target after final-runner refresh: final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `85/85`; bundle validation `102/102`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Req6 Parameterization Direct Validator Gate
- Spark sidecar spawn for `T-3G-N35-REQ6-DIRECT` failed with `agent thread limit reached`; main agent selected the direct Req6 parameterization JSON validator gap.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/req6_parameterization_audit_2026_06_16.json` and requires pass status, fail count `0`, check count `>=71`, expected knob defaults, and core PAR16/PAR32 policy checks present/pass.
- `tools/validate_final_signoff_bundle.py` now includes `req6_parameterization` in the bundle source set and directly requires the same live Req6 status/count/knob/core-check contract.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=92` and final signoff bundle validation `check_count >=110`.
- Regression coverage writes stale live Req6 count `70` and failed `parallelism_extension_par32_requires_fresh_reports` while preserving final-manifest Req6 checks; both operator handoff and bundle validation fail on direct live Req6 checks.
- Expected current target after final-runner refresh: final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `92/92`; bundle validation `110/110`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Current-Audit Direct Validator Gate
- Spark sidecar spawn for `T-3G-N36-CURRENT-AUDIT-DIRECT` failed with `agent thread limit reached`; main agent selected the direct third-goal current-audit JSON validator gap.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/third_goal_current_audit_2026_06_16.json` and requires blocked-external status, expected summary counts, exact external blocker names, C3b/XR-VITs blocker paths, XR policy integrity consistency, QKV optional-state, and no-side-effect safety.
- `tools/validate_final_signoff_bundle.py` now includes `current_audit` in the bundle source set and directly requires the same live current-audit contract.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=116` and final signoff bundle validation `check_count >=137`.
- Regression coverage writes stale current-audit summary, XR policy consistency drift, and QKV required-state drift while preserving final-manifest checks; both operator handoff and bundle validation fail on direct live current-audit checks.
- Expected current target after final-runner refresh: final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`; operator handoff validation `116/116`; bundle validation `137/137`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.

## 2026-06-16 Current-Doc XR Policy Count Freshness Gate
- Spark sidecar spawn for `T-3G-N37-DOC-POLICY-COUNT-FRESHNESS` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/write_spec_plan_conformance_audit.py` now rejects current docs that retain stale XR-VITs policy check count `7` after live current-audit and completion-audit policy check count advanced to `9`.
- The same current-doc freshness gate now rejects older validator anchors including `79/79`, `85/85`, `92/92`, `95/95`, `102/102`, and `110/110`; current anchors remain operator handoff validation `116/116` and bundle validation `137/137`.
- Current docs were refreshed to record policy check count `9` and the live validator targets.
- Validation:
  `python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit`: `147` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `171` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec-plan conformance `86/86`; final manifest `86/86`, consistency checks `332`, failed consistency `0`; operator handoff validation `116/116`; bundle validation `137/137`; remaining blockers unchanged.
  `git diff --check`: pass.

## 2026-06-16 Closeout-Validation Direct Validator Gate
- Spark sidecar spawn for `T-3G-N38-CLOSEOUT-VALIDATION-DIRECT` failed with `agent thread limit reached`; main agent completed the bounded fallback.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/final_unblock_closeout_packet_validation_2026_06_10.json` and requires pass status, fail count `0`, check count `>=49`, core closeout/board/XR-policy/QKV checks present/pass, and no-side-effect safety.
- `tools/validate_final_signoff_bundle.py` now includes `closeout_validation` in the bundle source set and applies the same live closeout-validation contract.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=116` and final signoff bundle validation `check_count >=137`.
- Regression coverage writes stale live closeout validation count `48` and failed `board_package_ready` while preserving final-manifest checks; both operator handoff and bundle validation fail on direct live closeout-validation checks.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tools/write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit`: `151` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `175` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec-plan conformance `86/86`; final manifest `86/86`, consistency checks `332`, failed consistency `0`; closeout validation `49/49`; operator handoff validation `116/116`; bundle validation `137/137`; remaining blockers unchanged.

## 2026-06-16 Closeout-Packet Direct Validator Gate
- Spark sidecar spawn for `T-3G-N39-CLOSEOUT-PACKET-DIRECT` failed with `agent thread limit reached`; main agent completed the bounded fallback.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/final_unblock_closeout_packet_2026_06_10.json` and requires ready status, blocker count `2`, board package readiness, C3b smoke contract, XR-VITs policy integrity, optional QKV URAM commands, and no-side-effect safety.
- `tools/validate_final_signoff_bundle.py` now includes `closeout_packet` in the bundle source set and applies the same live closeout-packet contract.
- `tools/write_final_evidence_manifest.py` keeps the final validator freshness thresholds at operator handoff validation `check_count >=116` and final signoff bundle validation `check_count >=137`.
- `tools/write_spec_plan_conformance_audit.py` rejects current docs that carry stale operator handoff validation `108/108`, bundle validation `128/128`, or XR policy check count `8` anchors after the live validator targets advanced.
- Regression coverage writes stale live closeout-packet status and QKV command drift while preserving final-manifest and closeout-validation checks; both operator handoff and bundle validation fail on direct live closeout-packet checks.
- Validation:
  `python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tools/write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_third_goal_completion_audit.py`: pass.
  `python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit`: `155` tests passed.
  `python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit`: `179` tests passed.
  `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked`: expected blocked; spec-plan conformance `86/86`; final manifest `86/86`, consistency checks `332`, failed consistency `0`; closeout validation `49/49`; operator handoff validation `116/116`; bundle validation `137/137`; remaining blockers unchanged.

## 2026-06-16 C3b/XR Gate Direct Validator Gate
- Spark sidecar spawn for `T-3G-N40-BLOCKER-GATE-DIRECT` failed with `agent thread limit reached`; main agent completed the bounded fallback.
- `tools/validate_final_operator_handoff.py` now directly loads `docs/resources/c3b_physical_smoke_gate_audit_2026_06_16.json` and `docs/resources/xr_vits_gate_audit_2026_06_16.json`.
- `tools/validate_final_signoff_bundle.py` now includes `c3b_physical_gate` and `xr_vits_gate` in the bundle source set and applies the same live blocker-gate contracts.
- `tools/write_final_evidence_manifest.py` now requires final operator handoff validation `check_count >=131` and final signoff bundle validation `check_count >=152`.
- `tools/write_spec_plan_conformance_audit.py` now rejects current docs that carry stale operator handoff validation `116/116` or bundle validation `137/137` anchors after the live validator targets advanced.
- Regression coverage writes C3b ready-for-board drift and XR candidate drift while preserving final-manifest gate checks; both operator handoff and bundle validation fail on direct live blocker-gate checks.
- Validation after final-runner refresh: final manifest `86/86`, consistency checks `332`, failed consistency `[]`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; closeout validation `49/49`; operator handoff validation `131/131`; bundle validation `152/152`; remaining blockers unchanged: C3b physical smoke and XR-VITs source/policy.
- Regression validation: py_compile passed; focused final-validator tests passed (`159` tests); broader final-signoff regression tests passed (`183` tests); `git diff --check` passed.

## 2026-06-16 Final Blocker Operator Plan Gate
- Spark sidecar spawn for `T-3G-N41-UNBLOCK-READINESS-SNAPSHOT` failed with `agent thread limit reached`; main agent completed local fallback.
- `tools/check_final_blocker_closure_readiness.py` now emits `operator_unblock_plan` with required C3b/XR inputs, dry-run command templates, active command state, and no-side-effect safety.
- `tools/write_final_evidence_manifest.py` now gates the operator plan through `final_blocker_closure_operator_plan_*` consistency checks.
- Validation after final-runner refresh: final manifest `86/86`, consistency checks `332`, failed consistency `[]`; spec-plan conformance `86/86`; operator handoff validation `131/131`; bundle validation `152/152`; remaining blockers unchanged.
- Regression validation: py_compile passed; focused N41 tests passed (`128` tests); broader final-signoff regression passed (`193` tests); `git diff --check` passed.

## 2026-06-16 Final Unblock Intake Operator-Plan Cross-Check
- Spark sidecar `T-3G-N42-INTAKE-PLAN-CROSSCHECK` completed read-only review and identified a false-green risk if final unblock intake `next_inputs` or `operator_sequence` drifted from final blocker closure `operator_unblock_plan`.
- `tools/write_final_evidence_manifest.py` now gates required-input agreement, next-input path agreement, expected operator-sequence steps, dry-run/active command agreement, and side-effect profile agreement.
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require those five manifest checks by name and require final evidence consistency count `>=347`; stale `340` contracts fail validation.
- Expected current target after final-runner refresh: final manifest `86/86`, consistency checks `347`, failed consistency `[]`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `131/131`; bundle validation `152/152`; remaining blockers unchanged.

## 2026-06-16 Final Runner Mirror Integrity Gate
- Spark sidecar `019ecfbc-76cc-7622-b6fa-e0d062fa2bb9` completed a read-only gap audit and identified a final-signoff evidence-path integrity risk between `hardware/generated/signoff` and canonical `docs/resources` mirrors.
- `tools/run_third_goal_final_signoff.py` now records `canonical_evidence_root`, `generated_signoff_root`, `mirrored_artifact_integrity_*`, and per-artifact source/mirror SHA256/size contracts. The final-runner summary JSON/MD are excluded from SHA256 self-checks because they are self-referential outputs.
- `tools/write_final_evidence_manifest.py` now requires mirror-integrity presence/pass/count checks and the runner source contract.
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now directly reject stale or missing live final-runner mirror-integrity fields.
- Expected current target after final-runner refresh: final manifest `86/86`, consistency checks `347`, operator handoff validation `131/131`, bundle validation `152/152`, and only the two external blockers remain.

## 2026-06-16 HANDOVER Validator Summary Freshness Gate
- `tools/write_spec_plan_conformance_audit.py` now requires `docs/track/HANDOVER.md` to record raw final operator handoff and final signoff bundle artifact summary counts matching live JSON: operator handoff validation checks `131`, bundle validation checks `152`.
- `tools/write_final_evidence_manifest.py` now includes these two HANDOVER raw-summary checks under `spec_plan_current_doc_freshness_contract`.
- Regression coverage mutates HANDOVER to keep live `131/131` and `152/152` anchors while reverting raw artifact summaries to checks `129` and `150`; spec-plan conformance fails both new checks.
- Expected current target after refresh: spec-plan conformance `86/86`, final manifest `86/86`, consistency checks `347`, operator handoff validation `131/131`, bundle validation `152/152`.

## 2026-06-16 Resource-Policy Artifact Singularity Gate
- GPT5.3-Codex-Spark sidecar hit usage limit; GPT5.5 sidecar completed read-only artifact inventory and found stale generated `e2e_resource_policy_audit_2026_06_15.{json,md}` with only `22/22` checks and mismatched filename/payload date semantics.
- Removed stale generated `2026_06_15` resource-policy pair; canonical pair remains `e2e_resource_policy_audit_2026_06_10.{json,md}` in both `docs/resources` and `hardware/generated/signoff`.
- `tools/write_final_evidence_manifest.py` now gates resource-policy canonical pair presence, docs/resources singularity, generated/signoff singularity, filename/payload date-tag agreement, canonical generated/docs SHA256 match, and combined canonical artifact-set status.
- `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require those six resource-policy singularity checks by name and require final evidence consistency count `>=347`.
- Regression coverage adds stale generated sibling, stale docs sibling, and canonical mirror hash mismatch cases.

## 2026-06-27 Runtime-Mode E2E Shared Top HLS Gate
- Script syntax: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh` passed.
- Script syntax: `bash -n hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh` passed.
- CSim: `timeout 900 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_mode_mem16` passed with Search `runtime_state=0`, Track `runtime_state=1`, 6 output state words per mode, correct TLAST, and `CSim done with 0 errors`.
- CSynth: `timeout 3600 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_mode_mem16` passed for ZCU104 part `xczu7ev-ffvc1156-2-e`.
- HLS latency gate: Track-bound min `76,150` cycles / `0.381 ms` passes the `1.000 ms` target; Search-bound max `594,840` cycles / `2.974 ms` passes the `4.000 ms` target.
- HLS resource estimate: BRAM_18K `68/624`, DSP `400/1728`, FF `39,696/460,800`, LUT `88,246/230,400`, URAM `0/96`.
- Package: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par32_runtime_mode_mem16` generated `generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` and `impl/export.zip`.
- Vivado route/bitgen: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_mode_mem16` completed with routed WNS `3.038 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`, fully routed nets `50,076/50,076`, route errors `0`, and bitgen pass.
- Routed artifacts: `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16.{bit,hwh}` and matching copies under `pynq/hgtxr/`.
- Runtime-mode unused weight-memory master cleanup verified: `m_axi_gmem_e2e_weights` / `gmem_e2e_weights` is absent from the packaged IP, generated RTL, overlay `.hwh`, and copied PYNQ `.hwh`; the remaining AXI-Lite `weights` fields are only pointer-control ABI.
- HW Emulation / HLS RTL cosim runner added: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_mode_mem16`.
- HLS RTL cosim reached C testbench pass for both runtime modes and built the XSIM snapshot:
  Search `runtime_state=0`, 6 output words, TLAST correct; Track `runtime_state=1`, 6 output words, TLAST correct; snapshot `hgtxr_e2e_axis_top` built.
- HLS RTL cosim is not a PASS yet: XSIM fails on generated command `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}` with `unexpected exception when evaluating tcl command`.
- XSIM negative checks completed: explicit `LD_LIBRARY_PATH`, `TERMINFO=/lib/terminfo`, multiple `TERM` values, generated `.wcfg` disable test, direct `xsim -R`, and direct `xsimk` load. The C/RTL report remains `Verilog Fail` with `NA` latency.
- Routed placed utilization: CLB LUT `22,412/230,400`, CLB registers `21,105/460,800`, block RAM tile `37/312`, DSP `401/1,728`, URAM `0/96`.
- Routed power estimate: total on-chip `3.736 W`, dynamic `3.042 W`, device static `0.694 W`.
- PYNQ runtime support: `runtime-mode-par32` Search uses a 128x128 frame and `num_pixels` control `16384`; Track uses a 64x64 frame and `num_pixels` control `1`, preventing AXIS input DMA beat-count mismatch for the runtime-mode top.
- PYNQ board-result validation: `axis-runtime-mode-par32-search` enforces `accelerator_latency_ms_summary.max <= 4.000 ms` and all Search accelerator-latency samples `<= 4.000 ms`; `axis-runtime-mode-par32-track` enforces `accelerator_latency_ms_summary.max <= 1.000 ms` and all Track accelerator-latency samples `<= 1.000 ms`.
- PYNQ bundle validation: `runtime-mode-par32-search` and `runtime-mode-par32-track` bundles both passed `tools/validate_pynq_bundle_package.py` with `errors=[]`.
- ZCU104 smoke-session runbooks: `generated/pynq/e2e_axis_dma_par32_runtime_mode_search_smoke_session.{json,md}` and `generated/pynq/e2e_axis_dma_par32_runtime_mode_track_smoke_session.{json,md}` generated with `status=pass`.
- ZCU104 remote-run dry-run: `tools/run_zcu104_c3b_smoke_remote.py --profile runtime-mode-par32-search ...` and `--profile runtime-mode-par32-track ...` both produced `status=dry-run`, `errors=0`.
- Combined board-latency gate: `tools/check_runtime_mode_board_latency_gate.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` generated `runtime_mode_board_latency_gate_2026_06_28.{json,md}` with `status=missing`, `cases_missing=2`, proving final board evidence is still absent rather than silently assumed.
- Combined Search/Track board runner: `tools/run_runtime_mode_board_latency.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host zcu104.local --user xilinx` generated `runtime_mode_board_latency_run_2026_06_28.{json,md}` with `status=dry-run`, `execute=False`, and board-latency gate `missing`.
- Combined Search/Track board runner execute attempt: `tools/run_runtime_mode_board_latency.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --host zcu104.local --user xilinx --execute` generated `runtime_mode_board_latency_run_2026_06_28_execute.{json,md}` with `status=host-unresolved`; host preflight failed with `[Errno -2] Name or service not known`, so SSH/SCP was not attempted and no board JSON was imported.
- Regression validation after host-preflight addition: `python3 -m py_compile ...` passed, focused runtime-mode unittest set passed `59` tests, `git diff --check` passed, and both runtime-mode board-latency JSON artifacts parse with `python3 -m json.tool`.
- Evidence file: `docs/track/RUNTIME_MODE_E2E_SHARED_TOP_2026_06_27.md`.
- Remaining validation gaps: no board smoke/physical runtime latency for this exact runtime-mode E2E top yet; the fastpath remains mode-profile compute evidence rather than a full exact weight-consuming ViT datapath proof.

## 2026-06-28 Full-Learned Transformer Vivado Probe
- Full learned non-fastpath path instrumentation was added for the runtime ROM dispatch profile: QKV, attention, MLP, head, dispatch prefetch, and intermediate buffers are mixed into an externally visible output word when `HGTXR_E2E_OBSERVE_DATAPATH=1`.
- Par8 CSim: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_dispatch_300_mem8` passed. Search output was `-221, -237, -237, -237, -237, -274`; Track output was `-219, -235, -235, -235, -235, -176`; both modes reported correct runtime state, count `6`, TLAST `1`, and `CSim done with 0 errors`.
- Par8 CSynth/package after datapath preservation, DSP latency hint, and low-fanout weight AXI hint: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_dispatch_300_mem8` passed with target clock `3.333 ns`, estimated clock `2.924 ns`, Track-bound latency `1,588,174` cycles / `5.293 ms`, Search-bound latency `12,929,562` cycles / `43.094 ms`, and hybrid 10/90 latency `9.074 ms`.
- Par8 HLS estimate: BRAM_18K `408/624 = 65%`, DSP `1,368/1,728 = 79%`, FF `186,572/460,800 = 40%`, LUT `222,223/230,400 = 96%`, URAM `32/96 = 33%`. Dominant instance is `hgtxr_e2e_controller_run` with `334` BRAM_18K, `1,360` DSP, `182,303` FF, `214,396` LUT, and `32` URAM.
- Package/OOC preservation: OOC XDC-only preservation was ineffective because the generated OOC flow disabled the HLS IP constraint file for implementation. The package script now injects RTL-level preservation attributes into exported HLS Verilog; current package run patched `38` datapath Verilog module files and leaves AXI/control wrappers optimizable.
- Par8 Vivado route/bitgen completed and produced a full learned compute-scale netlist: top placed utilization CLB LUT `37,300/230,400 = 16.19%`, CLB registers `40,008/460,800 = 8.68%`, Block RAM Tile `242/312 = 77.56%`, DSP `1,371/1,728 = 79.34%`, URAM `32/96 = 33.33%`.
- Par8 routed timing at `clk_pl_0 = 300.030 MHz` fails setup: WNS `-0.265 ns`, TNS `-113.997 ns`, WHS `0.010 ns`, THS `0.000 ns`; all `122,577/122,577` routable nets are fully routed, route errors `0`, and bitgen completes.
- Par8 routed vectorless power for the full learned netlist: total on-chip `6.817 W`, dynamic `6.096 W`, device static `0.721 W`, PS static `0.103 W`, PL static `0.619 W`. Hierarchy highlights: `hgtxr_e2e_axis_top_0` `3.079 W`, `axi_mem` `0.288 W`, `psu` `2.676 W`.
- Acceptance decision: full learned physical preservation is validated through Vivado route/bitgen, but timing closure and latency targets are not met. Mode-specific Search/Track power, DMA bandwidth, p95/p99 latency, and board rail power remain unmeasured.
- Evidence file: `docs/track/EIGHT_QUESTION_FULL_LEARNED_VIVADO_EXPERIMENT_2026_06_28.md`.
- Remaining validation gaps: close 300 MHz timing, reduce full learned Search/Track latency, then collect mode-specific Search/Track SAIF or board power, DMA bandwidth, repeated latency distribution, and p95/p99.

## 2026-06-28 Full-Learned ROM-Only DSP3 300MHz Closure
- Change: `HGTXR_E2E_DSP_MUL_LATENCY` now drives the HLS `bind_op ... latency=` pragma directly, instead of the previous hard-coded latency `2`.
- New profile: `par8_runtime_rom_only_dispatch_dsp3_300_mem8`, scale `runtime_rom_only_dispatch_dsp3_active64_b8_ff768`.
- CSim: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8` passed. Search output stayed `[-712, -712, -712, -712, -712, -696]`; Track output stayed `[-235, -235, -235, -235, -235, -176]`; Search `runtime_state=0`, Track `runtime_state=1`, and `CSim done with 0 errors`.
- Package: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_dsp3_300_mem8` passed. HLS estimated clock is `2.983 ns` against the `3.333 ns` target; HLS estimated Fmax is `335.24 MHz`.
- Packaged IP check: `m_axi_gmem_e2e_weights` is absent from `component.xml`; only `m_axi_gmem_e2e_runtime`, AXIS input/output, and AXI-Lite control remain. AXI-Lite `weights` registers remain pointer ABI fields, not an external weight memory master.
- Vivado route/bitgen: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8` completed and copied `.bit/.hwh` to `hardware/pynq/hgtxr/`.
- Post-route timing at `clk_pl_0 = 300.030 MHz`: WNS `0.009 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`; all user timing constraints are met.
- Route status: `107,584 / 107,584` routable nets fully routed; route errors `0`.
- Routed utilization: CLB LUT `23,887/230,400 = 10.37%`, CLB registers `26,595/460,800 = 5.77%`, Block RAM Tile `238/312 = 76.28%`, DSP `1,627/1,728 = 94.16%`, URAM `32/96 = 33.33%`.
- Routed vectorless power: total on-chip `5.973 W`, dynamic `5.258 W`, device static `0.715 W`, PS static `0.102 W`, PL static `0.613 W`.
- Routed hierarchy power: `hgtxr_e2e_axis_top_0 = 2.341 W`, `psu = 2.673 W`, `axi_mem = 0.190 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.027 W`, `axi_ctrl = 0.020 W`.
- HLS latency remains above the target: Track `1,600,952 cycles / 5.336 ms`, Search `12,972,966 cycles / 43.239 ms`, 10/90 Hybrid expected `2,738,153 cycles / 9.126 ms`.
- Throughput from HLS interval at 300.03 MHz: Track `187.41 Hz`, Search `23.13 Hz`, 10/90 Hybrid `109.57 Hz`.
- DRC warnings remain but no errors: DPIP-2 `9`, DPOP-4 `3`, REQP-1934 `2`, REQP-1935 `4`, RTSTAT-10 `1`.
- Acceptance decision: full learned ROM-only Transformer is now physically implemented through Vivado with clean 300 MHz post-route timing on ZCU104. The 4 ms Search and 1 ms Track latency goals are still not met, and mode-specific/board-measured power and p95/p99 latency still require additional experiments.

## 2026-06-28 Full-Block Dispatcher Prefetch Rerun

- Change: the Search dispatcher prefetch bank now uses `kBlockWeightWords`
  instead of the old fixed `16`-word prefix. The full-block bank is enabled for
  `par8_runtime_rom_only_dispatch_dsp3_300_mem8` with
  `HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK=1` and
  `HGTXR_E2E_URAM_DISPATCH_PREFETCH=1`.
- Baseline policy: `par32_dsp_mixed_stream_mem16` is retained as the
  physical/timing baseline because it routes with large timing margin;
  `par32_runtime_full_axi_mem16` is retained as the learned-weight functional
  baseline because it restores the AXI weight path. The current
  `par8_runtime_rom_only_dispatch_dsp3_300_mem8` candidate is the objective
  baseline for on-chip ROM + runtime scheduler + full-block dispatcher.
- CSim: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8` passed. Search output was `[-712, -712, -712, -712, -712, -676]`; Track output was `[-235, -235, -235, -235, -235, -291]`; runtime states were `0/1`, output count `6`, TLAST correct, and `CSim done with 0 errors`.
- Package: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par8_runtime_rom_only_dispatch_dsp3_300_mem8` passed. HLS target clock `3.333 ns`, estimated clock `2.924 ns`; top max latency `13,042,545` cycles / `43.471 ms`.
- Dispatcher storage proof: HLS generated `gb_dispatch_prefetch_U` as `RAM_2P_URAM_1R1W`, `13,848` words x `255` bits, using `16` URAM.
- HLS OOC resource estimate: BRAM_18K `386/624`, DSP `1,753/1,728`, FF `227,004/460,800`, LUT `308,055/230,400`, URAM `48/96`.
- Vivado route/bitgen: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par8_runtime_rom_only_dispatch_dsp3_300_mem8` completed with `impl_strategy=Performance_Explore` and generated the bitstream, with route errors `0`.
- Post-route timing at `clk_pl_0 = 300.030 MHz` now passes setup and hold: WNS `0.010 ns`, TNS `0.000 ns`, WHS `0.009 ns`, THS `0.000 ns`; failing setup endpoints `0/223,037`.
- Route status: `141,580 / 141,580` routable nets fully routed; route errors `0`.
- Placed utilization: CLB LUT `44,894/230,400 = 19.49%`, CLB registers `44,799/460,800 = 9.72%`, Block RAM Tile `238/312 = 76.28%`, DSP `1,630/1,728 = 94.33%`, URAM `48/96 = 50.00%`.
- Routed vectorless power: total on-chip `7.109 W`, dynamic `6.383 W`, device static `0.725 W`, PS static `0.103 W`, PL static `0.622 W`.
- Routed hierarchy power: `hgtxr_e2e_axis_top_0 = 3.467 W`, `psu = 2.673 W`, `axi_mem = 0.188 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.027 W`.
- Implementation note: during placement, Vivado reported MLP/ATTN hierarchy and `w1_weight_cache_U` `DONT_TOUCH` constraints that limited some fanout optimization, and bitgen reported many DSP `MREG=0` pipelining warnings. These are the next timing/power cleanup targets, even though timing now meets the current 300 MHz constraint.
- Acceptance decision: full learned Transformer plus full-block dispatcher prefetch is functionally and physically implemented through bitgen, ZCU104 resource fit is feasible, and this exact full-block build is timing-clean at 300 MHz with small margin. Search latency still fails `4 ms`; current exact Track-mode latency and all mode-specific/board-measured power and p95/p99 metrics remain pending.

## 2026-06-28 Full-Block Dispatcher Mode-Forced CSim/CSynth

- Added force-mode HLS profiles for the current full-block dispatcher build:
  `par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only` and
  `par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only`.
- Search-only CSim: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only` passed with output `[-712, -712, -712, -712, -712, -676]`, `runtime_state=0`, output count `6`, TLAST correct, and `CSim done with 0 errors`.
- Track-only CSim: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only` passed with output `[-235, -235, -235, -235, -235, -226]`, `runtime_state=1`, output count `6`, TLAST correct, and `CSim done with 0 errors`.
- Search-only CSim prefetch trace proved ordered prefetch-before-use for the traced extra Search blocks: block `2` and `3` prefetch completed at sequence `1/2`; first load hits occurred later at sequence `3/4`; `prefetch_trace_summary block_pairs=4 violations=0`.
- Search-only csynth: target `3.33 ns`, estimated `2.924 ns`, max latency `13,001,422` cycles / `43.334 ms`, interval max `13,001,423`, HLS resources BRAM_18K `386/624`, DSP `1,497/1,728`, FF `161,370/460,800`, LUT `236,034/230,400`, URAM `48/96`.
- Track-only csynth: target `3.33 ns`, estimated `3.473 ns`, max latency `1,606,660` cycles / `5.580 ms` in the HLS report, interval max `1,606,661`, HLS resources BRAM_18K `378/624`, DSP `1,108/1,728`, FF `157,213/460,800`, LUT `197,528/230,400`, URAM `32/96`.
- At the routed `300.030 MHz` clock, Track max latency converts to `5.355 ms`; Search remains `43.334 ms`; the 10% Search / 90% Track hybrid expectation is `2,746,136.2` cycles / `9.153 ms`.
- Acceptance decision: mode-specific latency is now isolated for the current full-block build, but both latency targets still fail. The CSim trace proves prefetch-before-use, not overlapped double-buffer execution. Mode-specific SAIF/board power, DMA bandwidth, and p95/p99 latency remain unmeasured.

## 2026-06-28 PAR32 Baseline Recheck for Objective Path

- Baseline role clarification: `par32_dsp_mixed_stream_mem16` is still the physical/timing baseline, and `par32_runtime_full_axi_mem16` is still the full learned AXI functional baseline.
- They are not the current objective baseline because the current objective requires the combined condition of on-chip ROM weights/LUTs, runtime Search/Track scheduler, and dispatcher prefetch. The older PAR32 profiles do not satisfy that full objective envelope.
- A PAR32 objective-path probe was added as `par32_runtime_rom_only_dispatch_dsp3_300_mem16` plus force-mode Search/Track variants.
- Search-only csynth for the PAR32 objective-path probe passed with target `3.33 ns`, estimated `2.777 ns`, max latency `4,764,228` cycles / `15.879 ms`, interval max `4,764,229`, and HLS resources BRAM_18K `418/624`, DSP `1,763/1,728`, FF `180,201/460,800`, LUT `302,026/230,400`, URAM `48/96`.
- Track-only csynth passed with target `3.33 ns`, estimated `3.473 ns`, max latency `604,652` cycles / `2.100 ms`, interval max `604,653`, and HLS resources BRAM_18K `410/624`, DSP `1,374/1,728`, FF `176,105/460,800`, LUT `263,377/230,400`, URAM `32/96`.
- PAR32 package completed for the combined objective profile, but Vivado implementation failed before placement: DRC `UTLZ-1` reported DSP over-utilization, requiring `1729` DSP cells while ZCU104 provides `1728`.
- Acceptance decision: the previous PAR32 profiles remain valid baselines for comparison, but the PAR32 ROM-only objective path is not a valid ZCU104 implementation point yet. It improves latency versus PAR8 but still misses Search `4 ms` and Track `1 ms`, and it currently exceeds the DSP budget by one DSP at implementation.

## 2026-06-28 PAR32 Objective DSP-Fit Core-Fabric Rerun

- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16` for the PAR32 objective path. It keeps the ROM-only learned weights/LUTs, runtime Search/Track scheduler, full-block dispatcher prefetch, raw head multiply, and prefetch address switching, then moves the last dense-lane core multiplier to fabric through `HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=1`.
- CSim passed. Search output was `[-1169, -1169, -1169, -1169, -1169, -1133]`; Track output was `[-235, -235, -235, -235, -235, -291]`; runtime states were `0/1`; the E2E AXIS vector comparison passed.
- CSynth passed with target clock `3.333 ns`, estimated clock `2.777 ns`, estimated Fmax `360.10 MHz`, top latency min `26,088` cycles / `86.951 us`, max `4,805,312` cycles / `16.016 ms`, and interval max `4,805,313`.
- CSynth resources were BRAM_18K `418`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `48`. The controller remained dominant with BRAM_18K `366`, DSP `1,974`, FF `241,522`, LUT `355,655`, URAM `48`; the raw/fabric head used DSP `2`.
- HLS package passed and patched `43` exported Verilog datapath module files for `HGTXR_E2E_OOC_DONT_TOUCH=datapath`.
- Vivado implementation reached route and bitgen. Route errors were `0`, `.bit` and `.hwh` artifacts were generated and copied to `hardware/pynq/hgtxr/`, and the implementation exactly fit the ZCU104 DSP budget at DSP48E2 `1,728/1,728`.
- Post-route 300 MHz setup timing did not close: `clk_pl_0` period `3.333 ns`, frequency `300.030 MHz`, WNS `-0.631 ns`, TNS `-3338.375 ns`, failing setup endpoints `15,703/303,013`. Hold timing was clean with WHS `0.008 ns`, THS `0.000 ns`.
- Placed utilization was CLB LUT `89,102/230,400 = 38.67%`, CLB registers `60,400/460,800 = 13.11%`, Block RAM Tile `232/312 = 74.36%`, DSP `1,728/1,728 = 100.00%`, and URAM `48/96 = 50.00%`.
- Routed vectorless power was total on-chip `8.593 W`, dynamic `7.856 W`, device static `0.737 W`, with Medium confidence. Component dynamic power included Clocks `0.558 W`, CLB Logic `1.247 W`, Signals `1.574 W`, Block RAM `0.264 W`, URAM `0.118 W`, DSPs `1.425 W`, and PS8 `2.671 W`. Hierarchy dynamic power included `hgtxr_e2e_axis_top_0 = 4.933 W`, `psu = 2.674 W`, `axi_mem = 0.194 W`, `axi_dma_in = 0.007 W`, `axi_dma_out = 0.025 W`, and `axi_ctrl = 0.023 W`.
- Acceptance decision: this profile promotes the PAR32 ROM-only objective path from "blocked by DSP overuse" to "DSP-fit routed/bitgen evidence." It is still not the signoff objective baseline because setup timing fails at 300 MHz and latency still misses the Search `4 ms` / Track `1 ms` targets.
- Next timing cleanup targets: add DSP MREG/PREG or equivalent pipeline stages for the DRC-reported DSP paths, reduce LayerNorm-to-BRAM and BRAM-to-QKV critical paths, revisit datapath `DONT_TOUCH` scope, and then rerun mode-specific Search/Track csynth plus board or SAIF power measurement.

## 2026-06-28 PAR32 Baseline Taxonomy and Norm-URAM Probe

- Baseline taxonomy was clarified after rechecking the earlier best PAR32 profiles:
  - `par32_dsp_mixed_stream_mem16` is the physical/timing comparison baseline. It has the best prior routed PAR32 margin: WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen pass, and HLS resources BRAM_18K `332`, DSP `1,148`, LUT `193,546`, URAM `64`.
  - `par32_runtime_full_axi_mem16` is the full learned AXI functional baseline. It restores learned Conv/ATTN/MLP/Head through the external AXI weight path, with HLS latency Search `25.510 ms` and Track `3.136 ms`.
  - The active objective baseline is separate: ROM-only learned weights/LUTs, runtime Search/Track scheduler, and dispatcher prefetch in one design. The earlier two PAR32 profiles remain required comparison baselines, but they are not objective-signoff candidates for the ROM-only scheduler/prefetch claim.
- Added resource policy `dsp_mixed_normstream` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16`. This keeps the PAR32 ROM-only objective path and moves the norm/global-buffer storage pressure from BRAM toward URAM.
- CSim passed with Search output `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track output `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and `CSim done with 0 errors`.
- CSynth passed with target `3.333 ns`, estimate `2.777 ns`, min latency `26,088` cycles / `86.951 us`, max latency `4,805,312` cycles / `16.016 ms`, interval max `4,805,313`, and resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`.
- Compared with the previous PAR32 core-fabric probe, CSynth latency is unchanged while BRAM_18K drops from `418` to `386` and URAM rises from `48` to `64`.
- Package passed and patched `43` exported Verilog datapath modules for `HGTXR_E2E_OOC_DONT_TOUCH=datapath`.
- Vivado route and bitgen completed. Route errors were `0`, `.bit/.hwh` artifacts were generated and copied to `hardware/pynq/hgtxr/`, and the design exactly fit DSP at `1,728/1,728`.
- Post-route timing improved but still failed at `clk_pl_0 = 300.030 MHz`: WNS `-0.496 ns`, TNS `-2860.064 ns`, WHS `0.000 ns`, THS `0.000 ns`, failing setup endpoints `15,033/303,394`.
- Placed utilization was CLB LUT `89,329/230,400 = 38.77%`, CLB registers `60,699/460,800 = 13.17%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, and URAM `64/96 = 66.67%`.
- Routed vectorless power was total on-chip `8.568 W`, dynamic `7.829 W`, and device static `0.738 W`.
- Acceptance decision: `normuram` is a useful timing/resource probe because WNS improved from `-0.631 ns` to `-0.496 ns` and BRAM pressure dropped, but it is still not a 300 MHz signoff baseline. The next closure work should target DSP MREG/PREG or explicit multiply/cache-read pipeline stages and a narrower `DONT_TOUCH` scope.

## 2026-06-29 PAR32 Compute-Selective DONT_TOUCH Probe

- Baseline decision: `par32_dsp_mixed_stream_mem16` remains the physical/timing baseline, and `par32_runtime_full_axi_mem16` remains the full learned AXI functional baseline. They are comparison baselines, not replacements for the ROM-only scheduler/prefetch objective path.
- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16`, which uses `HGTXR_E2E_OOC_DONT_TOUCH=compute` to preserve only the controller and key ATTN/MLP/QKV/attention/output/head modules.
- HLS package passed with target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, and resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`.
- Package preservation scope was reduced from broad datapath preservation to exactly 9 compute module Verilog files.
- Vivado route and bitgen completed. Route status was fully routed with `199,142/199,142` routed nets and `0` routing errors. Bitgen completed successfully at `2026-06-29 00:27:09 KST`.
- Routed timing still failed at the 300 MHz PL clock: WNS `-0.530 ns`, TNS `-3005.251 ns`, setup failing endpoints `15,103/302,281`, WHS `0.000 ns`, THS `0.000 ns`.
- Placed utilization was CLB LUT `86,983/230,400 = 37.75%`, CLB registers `59,792/460,800 = 12.98%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, and URAM `64/96 = 66.67%`.
- Routed vectorless power was total on-chip `8.537 W`, dynamic `7.799 W`, device static `0.738 W`, PS static `0.105 W`, and PL static `0.633 W`.
- Acceptance decision: this is a valid non-pruned objective-path routed/bitgen probe, but it is not a 300 MHz signoff baseline. The best current baseline taxonomy remains split: physical/timing baseline = `par32_dsp_mixed_stream_mem16`, learned AXI functional baseline = `par32_runtime_full_axi_mem16`, active objective candidate family = preserved ROM-only dispatcher profiles.
- Next validation target: add explicit DSP output/register pipeline stages and narrow compute preservation further so fanout replication is not blocked inside ATTN/MLP units.

## 2026-06-28 PAR32 Norm-URAM No-DONT-TOUCH Sanity Probe

- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16` to test whether removing broad exported-HLS datapath preservation lets Vivado recover timing for the same PAR32 ROM-only objective envelope.
- CSim passed with the same deterministic mode outputs as the preserved `normuram` profile: Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and `CSim done with 0 errors`.
- CSynth/package remained the same objective-scale HLS design: target `3.333 ns`, estimate `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`, and `impl/export.zip` generated.
- Vivado route and bitgen completed with clean timing at `clk_pl_0 = 300.030 MHz`: WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, THS `0.000 ns`, and route errors `0`.
- However, the placed utilization proves this timing result is invalid for accelerator signoff: CLB LUT `7,817/230,400 = 3.39%`, CLB registers `11,522/460,800 = 2.50%`, Block RAM Tile `5/312 = 1.60%`, DSP `0/1,728 = 0.00%`, URAM `0/96 = 0.00%`.
- Routed vectorless power also collapsed to a small-shell design: total on-chip `3.624 W`, dynamic `2.931 W`, device static `0.693 W`, PS static `0.099 W`, PL static `0.595 W`.
- Acceptance decision: `nodt` is a useful negative control, not a valid baseline. Removing datapath preservation allowed Vivado to optimize/prune the learned compute fabric, producing a timing-clean but non-representative design. Keep `par32_dsp_mixed_stream_mem16` as the physical/timing baseline, keep `par32_runtime_full_axi_mem16` as the full learned AXI functional baseline, and continue objective-path closure from preserved `normuram`/`corefabric` with selective rather than global `DONT_TOUCH` relaxation.

## 2026-06-28 PAR32 Norm-URAM Top-DONT-TOUCH Sanity Probe

- Added `HGTXR_E2E_OOC_DONT_TOUCH=top` support and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16` to test whether preserving only the exported HLS top module is enough to keep the PAR32 learned ROM-only datapath while allowing internal Vivado optimization.
- CSim passed with the same deterministic objective-path outputs as the preserved `normuram` profile: Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and `CSim done with 0 errors`.
- CSynth/package remained an objective-scale HLS design: target `3.333 ns`, estimate `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`; package patched exactly `1` exported Verilog module file.
- Vivado route and bitgen completed with clean timing at `clk_pl_0 = 300.030 MHz`: WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, THS `0.000 ns`, and route errors `0`.
- However, the placed utilization again proves this result is invalid for accelerator signoff: CLB LUT `7,817/230,400 = 3.39%`, CLB registers `11,522/460,800 = 2.50%`, Block RAM Tile `5/312 = 1.60%`, DSP `0/1,728 = 0.00%`, URAM `0/96 = 0.00%`.
- Routed vectorless power also matched the pruned-shell result: total on-chip `3.624 W`, dynamic `2.931 W`, device static `0.693 W`, PS static `0.099 W`, PL static `0.595 W`.
- Acceptance decision: `topdt` is also a negative control, not a valid baseline. Top-only preservation does not protect the internal learned arithmetic/cache datapath. Valid PAR32 objective closure must keep enough internal datapath preservation to prevent pruning, then add explicit pipeline stages or targeted preservation relaxation around known timing bottlenecks.

## 2026-06-28 PAR32 PipePref Fabric-Lane Closure Probes

- Baseline policy remains unchanged after these probes:
  - `par32_dsp_mixed_stream_mem16` remains the physical/timing comparison baseline because it is the best prior routed PAR32 stream candidate: WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen pass, HLS BRAM_18K `332`, DSP `1,148`, LUT `193,546`, URAM `64`.
  - `par32_runtime_full_axi_mem16` remains the full learned AXI functional baseline because it restores learned Conv/ATTN/MLP/Head through the AXI weight path, with HLS Search `25.510 ms` and Track `3.136 ms`.
  - The active objective path is still separate: ROM-only learned weights/LUTs, runtime Search/Track scheduler, and full-block dispatcher prefetch in one implementation.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16` passed CSim and package. CSim produced Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and `0` errors. CSynth reported target `3.333 ns`, estimate `2.777 ns`, max latency `4,817,564` cycles / `16.057 ms`, resources BRAM_18K `386`, DSP `1,976`, FF `277,802`, LUT `453,254`, URAM `64`. Vivado failed before placement with DRC `UTLZ-1`: required DSP `1,740`, available `1,728`.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail2_300_mem16` passed package. CSynth kept the same latency envelope and reduced estimated DSP only slightly to `1,966`, with BRAM_18K `386`, FF `278,120`, LUT `455,718`, URAM `64`. Vivado was not run because the estimate was still above the known physical DSP ceiling trend.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail4_300_mem16` passed package with estimated DSP `1,946`, LUT `460,646`, and the same latency envelope, but Vivado still failed DRC at required DSP `1,740/1,728`.
- Added `HGTXR_E2E_CORE_LANE_CT_SWITCH` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_cttail4_300_mem16` to test whether a template/switch-based compile-time lane split changes the tail-lane binding. Package passed, but the result matched `tail4`: target `3.333 ns`, estimate `2.777 ns`, max latency `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,946`, FF `278,756`, LUT `460,646`, URAM `64`. Vivado was not run because it did not improve over the already failed `tail4` implementation point.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_tail8_300_mem16` passed package with estimated DSP `1,906`, LUT `470,502`, and the same latency envelope, but Vivado still saw `1,740` DSP48E2 instances and failed DRC at `1,740/1,728`.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_300_mem16` passed package and reduced estimated DSP to `1,666`, but it is not a physical candidate because LUT rose to `529,792/230,400 = 229.9%`; Vivado was not run.
- `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_coreallfabric_nowide_300_mem16` passed package with target `3.333 ns`, estimate `2.979 ns`, estimated Fmax `335.63 MHz`, max latency `4,817,820` cycles / `16.058 ms`, resources BRAM_18K `386`, DSP `1,666`, FF `286,960`, LUT `647,424`, URAM `64`.

## 2026-06-29 300MHz Conditional WNS Threshold and PostRoutePhys Probe

- User-directed experimental timing rule: keep the PL clock at `300 MHz` (`clk_pl_0` period `3.333 ns`) and accept this experiment when final routed WNS is no worse than `-0.500 ns`. This is an experiment acceptance rule only; official Vivado timing signoff still requires WNS `>= 0.000 ns`.
- Baseline taxonomy remains unchanged:
  - `par32_dsp_mixed_stream_mem16` remains the physical/timing comparison baseline: routed WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen pass, HLS BRAM_18K `332`, DSP `1,148`, LUT `193,546`, URAM `64`.
  - `par32_runtime_full_axi_mem16` remains the full learned AXI functional baseline with learned Conv/ATTN/MLP/Head weights supplied through external AXI and HLS latency Search `25.510 ms`, Track `3.136 ms`.
  - The active objective candidates remain the preserved ROM-only dispatcher profiles because the target claim requires on-chip learned weights/LUTs, runtime Search/Track scheduler, and dispatcher prefetch in one implementation.
- Added runner profiles for 300 MHz pipe-prefetch follow-up:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16`.
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16`.
- `compute_pipepref_300` package passed with target `3.333 ns`, estimate `2.777 ns`, max latency `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,976`, FF `277,802`, LUT `453,254`, URAM `64`, but Vivado failed before placement with DRC `UTLZ-1`: required DSP `1,740`, available `1,728`.
- `compute_pipepref_tail4_nowide_300` package passed with target `3.333 ns`, estimate `2.979 ns`, max latency `4,817,820` cycles / `16.058 ms`, BRAM_18K `386`, DSP `2,226`, FF `289,648`, LUT `481,104`, URAM `64`. It was rejected without Vivado because DSP pressure worsened.
- Reran the existing valid `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16` IP through a new Vivado implementation project with `Performance_ExplorePostRoutePhysOpt`:
  - Project: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_postroutephys_300_mem16_overlay`.
  - Route status: `199,370/199,370` routable nets fully routed, routing errors `0`.
  - Bitgen: completed successfully at `2026-06-29 02:15:36 KST`.
  - Final post-route physopt timing: WNS `-0.402 ns`, TNS `-985.598 ns`, setup failing endpoints `7,376/302,473`, WHS `0.005 ns`, THS `0.000 ns`.
  - Experimental 300 MHz decision: pass under the user-approved WNS tolerance because `-0.402 ns >= -0.500 ns`.
  - Official Vivado decision: timing constraints are still not met because WNS is negative.
- Placed utilization for the postroutephys run: CLB LUT `87,181/230,400 = 37.84%`, CLB registers `59,652/460,800 = 12.95%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%`.
- Routed vectorless power for the postroutephys run: total on-chip `8.539 W`, dynamic `7.801 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`. Component power includes Clocks `0.576 W`, CLB Logic `1.219 W`, Signals `1.486 W`, Block RAM `0.265 W`, URAM `0.155 W`, DSPs `1.429 W`, and PS8 `2.671 W`.
- Critical-path evidence after route/physopt still points to `wo_weight_cache`, `gb_dispatch_prefetch`, and MLP DSP multiply/carry paths. Vivado DRC also reports repeated DSP input/output pipelining warnings (`DPIP-2`, `DPOP-4`). Next official-clean timing work should add explicit cache/URAM read output staging and DSP MREG/PREG-equivalent pipeline stages, and further narrow `DONT_TOUCH` where it blocks fanout replication.
- Acceptance decision: runtime tail-lane fabric selection is not enough to reduce final physical DSP use; Vivado still materialized `1,740` DSP instances for `tail4` and `tail8`. The first template/switch compile-time lane split (`cttail4`) also did not change the HLS result versus `tail4`. All-core fabric mapping proves DSP can be reduced, but the LUT cost is impossible for ZCU104 (`229.9%` to `281%`). The next valid closure step is not another lane-selection wrapper; it is an explicit separate operator/data-path split or pipelined DSP binding that reduces physical DSP/timing pressure without converting the dense core into LUT fabric.

## 2026-06-29 PAR32 ROM Compute 300MHz PYNQ SW/HW Equality Bundle

- Added board-smoke plumbing for the current postroutephys objective artifact under runner variant `par32-rom-compute-300`. This selects `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16.{bit,hwh}`.
- Added Search/Track bundle variants:
  - `par32-rom-compute-300-search`: mode profile `search`, expected runtime state `0`, expected raw output `[-1169, -1169, -1169, -1169, -1169, -1133]`.
  - `par32-rom-compute-300-track`: mode profile `track`, expected runtime state `1`, expected raw output `[-235, -235, -235, -235, -235, -226]`.
- These expected outputs match the current force-mode CSim evidence for the preserved PAR32 ROM-only objective path. They turn the next ZCU104 board smoke from a runtime-state-only check into a SW/HW equality check for the current arbitrary learned-parameter/LUT configuration.
- Generated and validated transfer bundles:
  - Search bundle: `hardware/generated/pynq/e2e_axis_dma_par32_rom_compute_300_search_smoke_bundle.tar.gz`, sha256 `d21042211c8aec38252618c339818a4a5e86786283813026ecf9218086544fa0`, validator status `pass`, errors `[]`.
  - Track bundle: `hardware/generated/pynq/e2e_axis_dma_par32_rom_compute_300_track_smoke_bundle.tar.gz`, sha256 `15e570fa6950cc0e501d54af7d2c9e5920073df10126fc649597f9efb03cf47b`, validator status `pass`, errors `[]`.
- Verification performed:
  - `python3 -m py_compile hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py hardware/tools/package_e2e_axis_dma_pynq_bundle.py hardware/tools/validate_pynq_smoke_result.py hardware/tools/validate_pynq_bundle_package.py hardware/pynq/hgtxr/test_hgtxr_overlay.py hardware/tests/test_package_e2e_axis_dma_pynq_bundle.py hardware/tests/test_validate_pynq_smoke_result.py hardware/tests/test_validate_pynq_bundle_package.py`
  - `PYTHONPATH=hardware/pynq python3 -m unittest hardware.pynq.hgtxr.test_hgtxr_overlay.E2EAxisDmaSmokeCliTests`: `6` tests passed.
  - `python3 -m unittest hardware.tests.test_validate_pynq_bundle_package hardware.tests.test_package_e2e_axis_dma_pynq_bundle hardware.tests.test_validate_pynq_smoke_result`: `30` tests passed.
- Remaining gap: the bundles are ready and locally validated, but no physical ZCU104 run JSON has been captured yet. Board p95/p99 latency, measured DMA bandwidth, and board power still require executing these bundles on the board.

## 2026-06-29 PAR32 ROM Compute 300MHz Board Runner/Gate Contract

- User-directed experiment rule is fixed for this lane: PL clock remains `300 MHz`, and routed WNS down to `-0.500 ns` is acceptable for experiment continuation. Official Vivado timing signoff still requires WNS `>= 0.000 ns`.
- Added a separate board latency profile set `par32-rom-compute-300` so the current objective artifact is not mixed with the older `runtime-mode-par32` profile set.
- Added remote runner profiles:
  - `par32-rom-compute-300-search`, preset `axis-par32-rom-compute-300-search`.
  - `par32-rom-compute-300-track`, preset `axis-par32-rom-compute-300-track`.
- Added canonical import/gate destinations:
  - Search: `hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json`.
  - Track: `hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json`.
- Generated dry-run board plan:
  - JSON: `hardware/generated/signoff/par32_rom_compute_300_board_latency_run_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/par32_rom_compute_300_board_latency_run_2026_06_29.md`.
  - Status: `dry-run`, host preflight `fail` for `zcu104.local` name resolution in the current environment, no SSH/SCP commands executed.
- Generated standalone gate evidence:
  - JSON: `hardware/generated/signoff/par32_rom_compute_300_board_latency_gate_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/par32_rom_compute_300_board_latency_gate_2026_06_29.md`.
  - Status: `missing`, because both canonical board result JSON files are not captured yet.
- Verification performed:
  - `python3 -m py_compile hardware/tools/import_pynq_smoke_result.py hardware/tools/run_zcu104_c3b_smoke_remote.py hardware/tools/check_runtime_mode_board_latency_gate.py hardware/tools/run_runtime_mode_board_latency.py hardware/tests/test_run_runtime_mode_board_latency.py hardware/tests/test_check_runtime_mode_board_latency_gate.py`
  - `python3 -m unittest hardware.tests.test_run_runtime_mode_board_latency hardware.tests.test_check_runtime_mode_board_latency_gate hardware.tests.test_run_zcu104_c3b_smoke_remote hardware.tests.test_import_pynq_smoke_result`: `31` tests passed.
  - `python3 hardware/tools/run_runtime_mode_board_latency.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --profile-set par32-rom-compute-300 --host zcu104.local --user xilinx`: dry-run completed with gate `missing`.
  - `python3 hardware/tools/check_runtime_mode_board_latency_gate.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --profile-set par32-rom-compute-300`: returned status `missing` as expected because board JSON is absent.
- Remaining gap: execute the prepared Search/Track bundles on ZCU104, import the two board JSON files into the canonical paths above, then rerun the gate. Until then, Search/Track board latency, p95/p99 latency, measured DMA bandwidth, and measured board power remain unavailable.

## 2026-06-29 PAR32 ROM Compute 300MHz Mode-Specific HLS Evidence

- User-directed experiment rule remains: PL clock target is `300 MHz` (`3.333 ns`) and routed WNS down to `-0.500 ns` is acceptable for experiment continuation. This is not official Vivado signoff; official timing still requires WNS `>= 0.000 ns`.
- Added force-mode HLS profiles for the current objective artifact:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_search_only`.
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16_track_only`.
- Search-only CSim passed with output `[-1169, -1169, -1169, -1169, -1169, -1133]`, `runtime_state=0`, six output words, and vector comparison pass. Prefetch trace was clean: block pair violations `0`; first loads for blocks `2` and `3` were ordered after prefetch completion.
- Track-only CSim passed with output `[-235, -235, -235, -235, -235, -226]`, `runtime_state=1`, six output words, and vector comparison pass. This corrected the stale board-bundle Track expectation from `-291` to `-226` for the current `compute_300` contract.
- Search-only CSynth at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; latency min `4,736,464` cycles / `15.787 ms`, max `4,764,201` cycles / `15.879 ms`; interval min `4,736,465`, max `4,764,202`; HLS resources BRAM_18K `386`, DSP `1,720`, FF `185,414`, LUT `322,452`, URAM `64`.
- Track-only CSynth at target `3.333 ns`: estimated clock `3.473 ns` / estimated Fmax `287.94 MHz`; latency min `592,360` cycles / `2.057 ms`, max `604,652` cycles / `2.100 ms`; interval min `592,361`, max `604,653`; HLS resources BRAM_18K `378`, DSP `1,332`, FF `181,667`, LUT `288,332`, URAM `48`.
- Hybrid 10% Search / 90% Track HLS average latency estimate from force-mode max latency is `3.478 ms` (`0.1 * 15.879 + 0.9 * 2.100`). Worst-case mode latency remains Search `15.879 ms`; this does not meet the target Search `4 ms` or Track `1 ms`.
- Resource reporting rule: use Vivado post-route utilization for the implemented shared Top, not force-mode HLS utilization, because HLS utilization overestimates LUT versus final placed implementation. Current implemented Top remains the postroutephys `compute_300` build: CLB LUT `87,181/230,400 = 37.84%`, CLB registers `59,652/460,800 = 12.95%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%`.
- Power reporting rule: current available power is Vivado vectorless routed power for the shared Top, not mode-specific board power. Current postroutephys power remains total on-chip `8.539 W`, dynamic `7.801 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`.
- Regenerated and validated PYNQ SW/HW equality bundles after the Track expectation correction:
  - Search tar sha256 `d21042211c8aec38252618c339818a4a5e86786283813026ecf9218086544fa0`, validation status `pass`, errors `[]`.
  - Track tar sha256 `15e570fa6950cc0e501d54af7d2c9e5920073df10126fc649597f9efb03cf47b`, validation status `pass`, errors `[]`.
- Remaining gap: no physical ZCU104 Search/Track JSON has been captured yet. Therefore measured board latency, p95/p99 latency, measured DMA bandwidth, and measured board power remain unavailable.

## 2026-06-29 Structured-ROM 300MHz Latency DSE Probe

- User-directed experiment rule remains fixed: PL clock target is `300 MHz` (`3.333 ns`) and routed WNS down to `-0.500 ns` is acceptable only for experiment continuation. Official Vivado timing signoff still requires WNS `>= 0.000 ns`.
- Added separate structured-ROM DSE profiles:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16`.
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_search_only`.
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_structrom_300_mem16_track_only`.
- Parameter contract for this probe: block-local Transformer ROM uses a selected structured learned-parameter set, with LayerNorm gamma/beta as identity/zero, Q/K as zero, V/O as diagonal identity under the existing fixed-point scale, and MLP W1/W2 as diagonal identity-style projections. The nonlinear tables remain on-chip and the structured path still calls LayerNorm, softmax-exp approximation, and GeLU logic.
- This is a design-space probe for the user-allowed arbitrary learned-parameter/LUT setting. It is not final dense arbitrary-weight Transformer signoff, because HLS can optimize the selected structured parameter algebra much more aggressively than the dense ROM compute path.
- Search-only CSim passed from terminal output before the csynth project reset: output `[1, 1, 1, 1, 1, -10]`, `runtime_state=0`, six output words, vector comparison pass, and prefetch trace violations `0`. The prefetch trace showed blocks `2` and `3` completed before first loads.
- Track-only CSim passed from terminal output before the csynth project reset: output `[0, 0, 0, 0, 0, 8]`, `runtime_state=1`, six output words, and vector comparison pass.
- Search-only CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; latency min/max `24,886` cycles / `82.945 us`; interval `24,887`; resources BRAM_18K `20`, DSP `2`, FF `8,372`, LUT `38,521`, URAM `0`.
- Track-only CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; latency min `12,443` cycles / `41.473 us`, max `24,731` cycles / `82.428 us`; interval min `12,444`, max `24,732`; resources BRAM_18K `20`, DSP `2`, FF `8,367`, LUT `38,522`, URAM `0`.
- Hybrid 10% Search / 90% Track HLS max-latency weighted average for this selected structured-ROM probe is `82.480 us` (`0.08248 ms`), and worst-case mode latency is Search `82.945 us`. Under this selected-parameter probe, the Search `4 ms` and Track `1 ms` latency targets are met at HLS level with large margin.
- Evidence boundary: generated CSynth XML contains the structured ATTN/MLP/controller functions, but the final top report collapses the hierarchy to a very small resource footprint. Therefore this proves the selected structured parameter contract can meet the latency target, not that the dense full learned Transformer datapath has been physically implemented with arbitrary parameters at that latency.
- Remaining gap: no Vivado implementation has been run for the structured-ROM DSE profile, no board JSON has been captured, and this result must not replace the current dense objective evidence where Search `15.879 ms` and Track `2.100 ms` still miss the targets.

## 2026-06-29 Dense Search Dispatcher Prefetch-All4 Probe

- Purpose: strengthen the Search dispatcher evidence for the actual dense ROM-only objective path, without replacing it with the structured-ROM shortcut. The new profile keeps full learned dense ATTN/MLP execution, on-chip parameter ROM, on-chip nonlinear LUT ROM, Search force-mode, and `compute`-scope datapath preservation.
- Added macros:
  - `HGTXR_E2E_DISPATCH_PREFETCH_BANKS`, default `2`.
  - `HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE`, default `0`.
  - `HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START`, defaulting to `HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE`.
- Added runner profile:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only`.
  - Extra flags: `HGTXR_E2E_DISPATCH_PREFETCH_BANKS=4`, `HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1`, `HGTXR_E2E_FORCE_RUNTIME_MODE=0`, `HGTXR_E2E_CSIM_PREFETCH_TRACE=1`, `HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1`.
- Search-only CSim passed from terminal output. The trace proves the remaining Search dispatcher blocks are prefetched before compute and that block-to-block compute start has no intervening prefetch event:
  - `prefetch_trace_done block=2 seq=1 words=6924`.
  - `prefetch_trace_done block=3 seq=2 words=6924`.
  - `prefetch_trace_interblock prev=0 next=1 ... immediate=1`.
  - `prefetch_trace_interblock prev=1 next=2 ... immediate=1`.
  - `prefetch_trace_interblock prev=2 next=3 ... immediate=1`.
  - `prefetch_trace_summary block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Output vector: `[-1169, -1169, -1169, -1169, -1169, -1125]`, `runtime_state=0`, `failures=0`, `CSim done with 0 errors`.
- CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; latency min `4,757,253` cycles / `15.856 ms`, max `4,757,273` cycles / `15.856 ms`; interval min `4,757,254`, max `4,757,274`.
- CSynth resource estimate: BRAM_18K `386/624 = 61%`, DSP `1,592/1,728 = 92%`, FF `174,661/460,800 = 37%`, LUT `320,413/230,400 = 139%`, URAM `76/96 = 79%`.
- Storage proof: `gb_dispatch_prefetch_U` is implemented as `ram_2p` with `impl=uram`, shape `255 x 27696 x 1`, and estimated URAM `28` for the four-bank prefetch buffer. This is the expected cost increase versus the existing two-bank `compute_300` path.
- Vivado physical implementation was run for the same profile with PL clock `300 MHz` and strategy `Performance_ExplorePostRoutePhysOpt`.
  - Final post-route physopt timing: WNS `-0.142 ns`, TNS `-69.079 ns`, setup failing endpoints `1,224/304,969`, WHS `0.006 ns`, THS `0.000 ns`.
  - Routed status: routing errors `0`; route_design reported failed nets `0`, unrouted nets `0`, partially routed nets `0`, node overlaps `0`, and route verification success.
  - Bitgen completed successfully with `0` errors. The overlay directory contains the `.bit` and `.hwh` artifacts for `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only`.
  - Placed utilization: CLB LUTs `53,926/230,400 = 23.41%`, CLB registers `46,812/460,800 = 10.16%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,592/1,728 = 92.13%`.
  - Vector-less routed power estimate: total on-chip `7.189 W`, dynamic `6.460 W`, device static `0.729 W`. Component dynamic includes clocks `0.405 W`, CLB logic `0.518 W`, signals `1.029 W`, Block RAM `0.259 W`, URAM `0.213 W`, DSP `1.365 W`, PS8 `2.671 W`. Confidence is `Medium`; no simulation activity file was used.
- Acceptance decision: this profile now has ZCU104 physical route/bitgen evidence and passes the user-approved experimental 300 MHz rule because WNS `-0.142 ns >= -0.500 ns`. It is not official clean Vivado timing signoff because WNS is still negative and the timing report says constraints are not met. Search latency remains `15.856 ms`, still above the `4 ms` target.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 300MHz Probe

- Purpose: validate the same dense ROM-only objective path with runtime Search/Track mode branching in a single E2E AXIS/DMA Top, not a force-mode Search-only artifact. The user-approved experiment rule remains PL clock `300 MHz` with final routed WNS accepted down to `-0.500 ns`; official Vivado signoff still requires non-negative WNS.
- Added combined HLS/Vivado profile:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`.
  - HLS flags include `HGTXR_E2E_DISPATCH_PREFETCH_BANKS=4`, `HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE=1`, `HGTXR_E2E_CSIM_PREFETCH_TRACE=1`, and `HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START=1`.
  - Vivado artifact: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_overlay`.
- Combined CSim passed from terminal output for both runtime modes:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, `search_runtime_state=0`, six output words, vector comparison pass.
  - Track output `[-235, -235, -235, -235, -235, -239]`, `track_runtime_state=1`, six output words, vector comparison pass.
  - Search dispatcher trace: blocks `2` and `3` prefetched with `6924` words each; interblock starts `0->1`, `1->2`, and `2->3` all reported `immediate=1`; summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track trace summary: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Combined CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,798,390` cycles / `15.993 ms`, interval min `26,089`, max `4,798,391`.
- Combined CSynth instance latency highlights:
  - Frame read: `4,100-16,388` cycles / `13.665-54.621 us`.
  - Frame Conv and Event Conv: each `12,436-24,724` cycles / `41.449-82.405 us`.
  - Global buffer load: `3,077-12,293` cycles / `10.256-40.973 us`.
  - Shared controller/Transformer runtime: `1-4,720,079` cycles / `3.333 ns-15.732 ms`.
  - Head: `6,465-24,897` cycles / `21.548-82.982 us`.
- Combined CSynth resource estimate remains pessimistic versus Vivado fit: BRAM_18K `386/624 = 61%`, DSP `1,848/1,728 = 106%`, FF `242,912/460,800 = 52%`, LUT `400,236/230,400 = 173%`, URAM `76/96 = 79%`.
- Storage evidence: generated RTL includes `gb_dispatch_prefetch_RAM_2P_URAM_1R1W` for the four-bank dispatcher prefetch buffer and `kGeluRom_ROM_1P_BRAM_1R` for nonlinear LUT storage. QKV/MLP dense compute and cache modules remain present in the generated RTL.
- Vivado implementation completed route, post-route physopt, reports, and bitgen at `clk_pl_0 = 300.030 MHz`.
  - Final post-route physopt timing: WNS `-0.236 ns`, TNS `-718.875 ns`, setup failing endpoints `6,814/300,626`, WHS `0.001 ns`, THS `0.000 ns`.
  - Route status: routing errors `0`; bitgen completed successfully with `0` errors.
  - Overlay artifacts generated: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.bit` and `.hwh`.
- Physical placed utilization for the combined runtime Top: CLB LUTs `73,133/230,400 = 31.74%`, CLB registers `60,949/460,800 = 13.23%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,723/1,728 = 99.71%`.
- Routed vector-less power estimate: total on-chip `8.364 W`, dynamic `7.626 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`. Component dynamic includes clocks `0.566 W`, CLB logic `0.883 W`, signals `1.606 W`, Block RAM `0.265 W`, URAM `0.222 W`, DSPs `1.414 W`, and PS8 `2.671 W`.
- Acceptance decision: the combined runtime Top passes the user-approved 300 MHz experiment threshold because WNS `-0.236 ns >= -0.500 ns`, route errors are `0`, hold is clean, and bitgen completed. It is not official clean Vivado timing signoff because timing constraints are not met. The combined HLS max latency `15.993 ms` still misses the Search `4 ms` target, while the min/runtime fast path `86.951 us` is not a substitute for measured mode-specific board p95/p99. ZCU104 board Search/Track JSON, measured DMA bandwidth, and measured board power remain pending.

## 2026-06-29 Dense Runtime Search/Track DSP/BRAM Pressure Relief Follow-up

- Purpose: respond to DSP and BRAM pressure by moving selected small-memory structures to LUTRAM and moving a small number of multiply lanes away from DSP, while keeping the same dense runtime Search/Track Top and the user-approved PL `300 MHz` / WNS `>= -0.500 ns` experiment rule.
- Added follow-up profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16`.
  - Inherits `HGTXR_E2E_NONLINEAR_ROM_LUTRAM=1`, `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=2`, and `HGTXR_E2E_CORE_LANE_CT_SWITCH=1`.
  - Adds `HGTXR_E2E_DSP_MUL_LATENCY=4` to expose more DSP pipeline movement to Vivado.
- CSim passed for both runtime modes:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, runtime state `0`, vector comparison pass.
  - Track output `[-235, -235, -235, -235, -235, -239]`, runtime state `1`, vector comparison pass.
  - Strict prefetch trace passed for Search and Track with `violations=0`, `not_ready=0`, and `immediate_gaps=0`.
- CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,810,934` cycles / `16.035 ms`, interval max `4,810,935`.
- CSynth resources: BRAM_18K `384/624 = 61%`, DSP `1,838/1,728 = 106%`, FF `244,988/460,800 = 53%`, LUT `402,794/230,400 = 174%`, URAM `76/96 = 79%`.
- Physical implementation completed route, post-route physopt, reports, and bitgen:
  - Final implemented timing at `clk_pl_0 = 300.030 MHz`: WNS `-0.372 ns`, TNS `-1326.235 ns`, setup failing endpoints `8,485/303,851`, WHS `0.005 ns`, THS `0.000 ns`.
  - Route status: routing errors `0`; bitgen completed successfully.
  - Overlay artifacts generated under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_overlay/`.
- Implemented utilization: CLB LUTs `71,232/230,400 = 30.92%`, CLB registers `63,821/460,800 = 13.85%`, Block RAM Tile `201/312 = 64.42%`, RAMB36E2 `180`, RAMB18E2 `42`, URAM `76/96 = 79.17%`, DSP `1,713/1,728 = 99.13%`, LUT as Memory `1,589/101,760 = 1.56%`, LUT as Distributed RAM `1,232/101,760 = 1.21%`.
- Resource deltas versus active `prefetchall4_300` implemented baseline:
  - Block RAM Tile `216 -> 201`, DSP `1,723 -> 1,713`, CLB LUT `73,133 -> 71,232`, registers `60,949 -> 63,821`, URAM unchanged `76`.
  - This is the best routed pressure-relief point so far because the earlier `lutrom_tail2` reduced DSP further to `1,669` but missed the experiment WNS floor with WNS `-0.588 ns`.
- Vectorless power estimate: total on-chip `8.396 W`, dynamic `7.658 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`. Component dynamic includes clocks `0.543 W`, CLB logic `0.910 W`, signals `1.650 W`, Block RAM `0.220 W`, URAM `0.222 W`, DSPs `1.442 W`, and PS8 `2.671 W`.
- Acceptance decision: `lutrom_tail2_dsppipe4` passes the user-approved 300 MHz experiment continuation threshold because WNS `-0.372 ns >= -0.500 ns`, hold is clean, route errors are `0`, and bitgen completed. It is not official clean Vivado timing signoff because WNS is still negative and the timing report says constraints are not met.
- Remaining pressure: DSP is still nearly saturated at `1,713/1,728 = 99.13%`. Vivado still reports many DSP `MREG=0` pipelining warnings and prior phys_opt messages showed broad compute `DONT_TOUCH` limiting ATTN/MLP replication; the next timing/resource cleanup should target narrower `DONT_TOUCH` scope plus remaining MREG/PREG-equivalent pipeline stages.

### Tail2 DSP-Pipeline RTL Output Check

- Long direct `xsimk` execution of the same `lutrom_tail2_dsppipe4` HLS snapshot completed both HLS RTL transactions in `1538.798 s`: progress reached `RTL Simulation : 2 / 2 [100.00%] @ 18092130000`.
- The direct stdout did not print the C testbench Search/Track vectors, so TV output files were compared directly.
- Added `hardware/tools/check_hls_xsimk_direct_probe.py --compare-tv-only` and per-port C/RTL TV comparison. The generated artifact is `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.json` and `.md`.
- TV comparison status is `tv-output-mismatch`:
  - `axis_out_V_data_V` mismatches at first payload line: C reference `0x000000000000000000000000000000000000000000000000000000000000fb6f`, RTL `0x0000000000000000000000000000000000000000000000000000000000000000`.
  - `axis_out_V_keep_V`, `axis_out_V_strb_V`, `axis_out_V_last_V`, and `gmem_e2e_runtime` match after normalizing transaction-line whitespace.
- Validation decision: the pressure-relief implementation remains useful for area/timing/power exploration, but it is not RTL functional signoff. Do not use the `lutrom_tail2_dsppipe4` RTL result as SW/HW same-input same-output evidence until the zeroed RTL AXIS data output is fixed and the TV comparison passes.

### Targeted Token LUTRAM Follow-up

- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16`.
  - Keeps the balanced `tail2_dsppipe4` compute policy: nonlinear LUTROM, two fabric tail multiply lanes, `HGTXR_E2E_DSP_MUL_LATENCY=4`, strict prefetch-all4 tracing, and PL target `300 MHz`.
  - Adds only `HGTXR_E2E_LUTRAM_FRAME_TOKENS=1` and `HGTXR_E2E_LUTRAM_GB_TOKENS=1`. This intentionally targets the small/high-bank-count token buffers first instead of moving norm/Q/K/V/attention buffers wholesale.
- AXIS output data path fix: `hgtxr_data_to_axis()` now emits raw `ap_fixed` payload bits under synthesis, and `hgtxr_axis_write_state_values()` is inlined so the generated top connects `axis_out_TDATA` to the captured head output states instead of a constant zero writer path.
- CSim passed after the change:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, runtime state `0`, vector comparison pass.
  - Track output `[-235, -235, -235, -235, -235, -239]`, runtime state `1`, vector comparison pass.
  - Strict prefetch trace passed for Search and Track with `violations=0`, `not_ready=0`, and `immediate_gaps=0`.
- CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; top latency min `26,094` cycles / `86.971 us`, max `4,809,404` cycles / `16.030 ms`.
- CSynth resources: BRAM_18K `336/624 = 53%`, DSP `1,838/1,728 = 106%`, FF `243,190/460,800 = 52%`, LUT `409,665/230,400 = 177%`, URAM `76/96 = 79%`.
- HLS resource delta versus `lutrom_tail2_dsppipe4`: BRAM_18K `384 -> 336`, DSP unchanged `1,838`, FF `244,988 -> 243,190`, LUT `402,794 -> 409,665`, URAM unchanged `76`.
- Memory evidence: the CSynth Memory table now reports `BRAM_18K 0`, `LUT 6912`, `Words 98304`, `Bits 144`, `Banks 32` for the token/gb-token memories. This is the requested targeted BRAM-to-LUTRAM move for small data with many banks.
- RTL structural evidence after the AXIS payload fix: generated `hgtxr_e2e_axis_top.v` drives `axis_out_TDATA_int_regslice` from `zext_ln437*` wires derived from `out_state_*` registers, and the output regslice uses `.data_in(axis_out_TDATA_int_regslice)`. It is no longer structurally tied to `256'd0`.
- Vivado route/post-route physopt/bitgen completed. Route errors are `0`, fully routed nets are `183,782/183,782`, and the bitstream was generated under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16_overlay/`. The impl directory had no direct `.hwh`, so the overlay script copied the BD handoff `.hwh` via fallback.
- Implemented utilization: CLB LUT `82,306/230,400 = 35.72%`, LUT as Distributed RAM `8,912`, CLB registers `63,181/460,800 = 13.71%`, Block RAM Tile `184/312 = 58.97%`, RAMB36E2 `171`, RAMB18E2 `26`, URAM `76/96 = 79.17%`, DSP `1,713/1,728 = 99.13%`.
- Implemented resource delta versus `lutrom_tail2_dsppipe4`: Block RAM Tile `201 -> 184`, LUT as Distributed RAM `1,232 -> 8,912`, CLB LUT `71,232 -> 82,306`, registers `63,821 -> 63,181`, DSP unchanged `1,713`, URAM unchanged `76`. This confirms the targeted token-buffer BRAM-to-LUTRAM move physically saves BRAM without the broader URAM reduction of `tail16_lutbuf`.
- Final post-route physopt timing is WNS `-0.560 ns`, TNS `-6463.610 ns`, WHS `0.005 ns`, THS `0.000 ns`, with `29,627` setup-failing endpoints at `clk_pl_0 = 300.030 MHz`. This misses the user-approved continuation floor `WNS >= -0.500 ns` by `0.060 ns`.
- Implemented vectorless power: total on-chip `8.769 W`, dynamic `8.028 W`, device static `0.741 W`, PS static `0.105 W`, PL static `0.636 W`; component buckets include clocks `0.626 W`, CLB logic `0.944 W`, signals `1.917 W`, Block RAM `0.210 W`, URAM `0.217 W`, DSPs `1.443 W`, and PS8 `2.671 W`. Confidence remains Vivado vectorless `Medium`, not board/SAIF power.
- Current judgment: the targeted small-data/many-bank memory move is physically proven and saves the same BRAM Tile count as the aggressive `tail16_lutbuf` point (`184`), while keeping URAM at `76` and DSP at `1,713`. It is not promoted over `lutrom_tail2_dsppipe4` because timing regresses from WNS `-0.372 ns` to `-0.560 ns` and misses the current experiment floor. Full RTL TV equality for this profile remains pending.

## 2026-06-29 Dense Runtime Search/Track Tail4 DSP Relief Probe

- Purpose: test whether moving four dense core tail lanes from DSP to fabric can further reduce DSP pressure beyond `lutrom_tail2_dsppipe4` while preserving the same 300 MHz experiment rule.
- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16`.
  - Keeps nonlinear ROM LUTRAM, strict prefetch-all4 tracing, `HGTXR_E2E_DSP_MUL_LATENCY=4`, and compile-time lane switching.
  - Changes `HGTXR_E2E_CORE_FABRIC_TAIL_LANES` from `2` to `4`.
- CSim passed for both runtime modes:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, runtime state `0`, vector comparison pass.
  - Track output `[-235, -235, -235, -235, -235, -239]`, runtime state `1`, vector comparison pass.
  - Strict prefetch trace passed with `violations=0`, `not_ready=0`, and `immediate_gaps=0`.
- CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,810,934` cycles / `16.035 ms`, interval max `4,810,935`.
- CSynth resources: BRAM_18K `384/624 = 61%`, DSP `1,818/1,728 = 105%`, FF `245,528/460,800 = 53%`, LUT `407,722/230,400 = 176%`, URAM `76/96 = 79%`.
- Physical implementation completed route, post-route physopt, reports, and bitgen:
  - Final implemented timing at `clk_pl_0 = 300.030 MHz`: WNS `-0.573 ns`, TNS `-1629.740 ns`, setup failing endpoints `7,453/299,787`, WHS `0.009 ns`, THS `0.000 ns`.
  - Route status: routing errors `0`; bitgen completed successfully, but the timing report still fails setup.
- Implemented utilization: CLB LUTs `70,766/230,400 = 30.71%`, CLB registers `63,836/460,800 = 13.85%`, Block RAM Tile `201/312 = 64.42%`, RAMB36E2 `180`, RAMB18E2 `42`, URAM `76/96 = 79.17%`, DSP `1,693/1,728 = 97.97%`, LUT as Memory `1,601/101,760 = 1.57%`, LUT as Distributed RAM `1,232/101,760 = 1.21%`.
- Resource deltas versus the accepted `lutrom_tail2_dsppipe4` pressure-relief point:
  - DSP `1,713 -> 1,693`, CLB LUT `71,232 -> 70,766`, registers `63,821 -> 63,836`, Block RAM Tile unchanged `201`, URAM unchanged `76`.
  - Vectorless power shifts from total `8.396 W` to `8.329 W`, dynamic `7.658 W` to `7.592 W`, and DSP bucket `1.442 W` to `1.415 W`.
- Acceptance decision: rejected for promotion. The DSP reduction is real, but WNS `-0.573 ns` misses the user-approved continuation floor `-0.500 ns`; the current pressure-relief recommendation remains `lutrom_tail2_dsppipe4` with WNS `-0.372 ns`.
- Next closure target: add explicit MREG/PREG-equivalent staging around MLP/attention DSP paths and narrow compute `DONT_TOUCH` where it blocks fanout replication, rather than moving more lanes to LUT fabric.

## 2026-06-29 Dense Runtime Search/Track Compute-Keep Probe

- Purpose: test whether relaxing broad compute `DONT_TOUCH` to `KEEP_HIERARCHY` only can let Vivado replication/retiming improve timing while keeping the same `lutrom_tail2_dsppipe4` pressure-relief datapath.
- Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16`.
  - Keeps nonlinear ROM LUTRAM, two core tail lanes in fabric, DSP multiply latency `4`, strict prefetch-all4 tracing, and PL target `300 MHz`.
  - Uses `HGTXR_E2E_OOC_DONT_TOUCH=compute_keep`, which patches compute modules with `keep_hierarchy` but deliberately omits `dont_touch`.
- CSim passed for both runtime modes:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, runtime state `0`, vector comparison pass.
  - Track output `[-235, -235, -235, -235, -235, -239]`, runtime state `1`, vector comparison pass.
  - Strict prefetch trace passed with `violations=0`, `not_ready=0`, and `immediate_gaps=0`.
- CSynth passed at target `3.333 ns`: estimated clock `2.777 ns` / estimated Fmax `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,810,934` cycles / `16.035 ms`, interval max `4,810,935`.
- CSynth resources remained objective-scale: BRAM_18K `384/624 = 61%`, DSP `1,838/1,728 = 106%`, FF `244,988/460,800 = 53%`, LUT `402,794/230,400 = 174%`, URAM `76/96 = 79%`.
- Package evidence confirmed the intended guard style:
  - The generated OOC XDC applies `KEEP_HIERARCHY true` to the compute cells and does not apply `DONT_TOUCH`.
  - Patched Verilog module guards are `(* keep_hierarchy = "yes" *)`.
- Vivado implementation completed route, post-route physopt, reports, and bitgen, and timing was clean: WNS `0.064 ns`, TNS `0.000 ns`, WHS `0.009 ns`, THS `0.000 ns`; route errors `0`.
- Rejection evidence: final implemented utilization collapsed to a pruned shell, not the full learned accelerator:
  - CLB LUTs `7,901/230,400 = 3.43%`, CLB registers `11,546/460,800 = 2.51%`, Block RAM Tile `5/312 = 1.60%`, URAM `0/96 = 0.00%`, DSP `1/1,728 = 0.06%`.
  - Vectorless power likewise dropped to total on-chip `3.615 W`, dynamic `2.922 W`, static `0.693 W`, with DSP bucket `<0.001 W`.
- Acceptance decision: rejected as invalid despite positive WNS. Removing `DONT_TOUCH` allowed Vivado to optimize away the full learned compute fabric, so this cannot be used for Search/Track latency, power, or resource signoff. The current valid pressure-relief recommendation remains `lutrom_tail2_dsppipe4`: WNS `-0.372 ns`, route errors `0`, bitgen pass, BRAM Tile `201`, URAM `76`, DSP `1,713`.
- Constraint for next work: timing experiments must preserve full learned compute resources. Any future narrower-preservation profile must pass both timing and a resource-preservation gate near the known full-design scale (`DSP` roughly `1.7k`, `URAM 76`, `BRAM Tile ~201`) before promotion.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 Board Bundle/Gate

- Added PYNQ smoke variant `par32-prefetchall4-300` for the combined runtime artifact:
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.bit`.
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16.hwh`.
- Added validator presets:
  - `axis-par32-prefetchall4-300-search`: variant `par32-prefetchall4-300`, mode profile `search`, expected runtime state `0`, expected raw output `[-1169, -1169, -1169, -1169, -1169, -1125]`.
  - `axis-par32-prefetchall4-300-track`: variant `par32-prefetchall4-300`, mode profile `track`, expected runtime state `1`, expected raw output `[-235, -235, -235, -235, -235, -239]`.
- Generated and validated transfer bundles:
  - Search bundle: `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_search_smoke_bundle.tar.gz`, sha256 `34f76304a0ca41bed3b60bfad610898bb41bb7cfc6e698c7b71f072f4de98491`, validator status `pass`, errors `[]`.
  - Track bundle: `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_track_smoke_bundle.tar.gz`, sha256 `9b24856588b0ae691104aa79476bc1da80b8a3884a8d568d7e0660279a08d4e0`, validator status `pass`, errors `[]`.
- Added board runner/gate profile set `par32-prefetchall4-300`:
  - Remote profiles: `par32-prefetchall4-300-search`, `par32-prefetchall4-300-track`.
  - Canonical Search JSON: `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json`.
  - Canonical Track JSON: `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json`.
  - Dry-run plan: `hardware/generated/signoff/par32_prefetchall4_300_board_latency_run_2026_06_29.json` and `.md`, status `dry-run`, gate `missing`.
  - Standalone gate: `hardware/generated/signoff/par32_prefetchall4_300_board_latency_gate_2026_06_29.json` and `.md`, status `missing` because both canonical board result JSON files are absent.
- Verification performed:
  - `python3 -m py_compile hardware/pynq/hgtxr/run_e2e_axis_dma_smoke.py hardware/tools/package_e2e_axis_dma_pynq_bundle.py hardware/tools/validate_pynq_bundle_package.py hardware/tools/validate_pynq_smoke_result.py hardware/tools/import_pynq_smoke_result.py hardware/tools/run_zcu104_c3b_smoke_remote.py hardware/tools/check_runtime_mode_board_latency_gate.py hardware/tools/run_runtime_mode_board_latency.py`
  - `python3 hardware/tools/validate_pynq_bundle_package.py --variant par32-prefetchall4-300-search ...`: `pass`.
  - `python3 hardware/tools/validate_pynq_bundle_package.py --variant par32-prefetchall4-300-track ...`: `pass`.
  - `python3 -m unittest hardware.tests.test_validate_pynq_smoke_result hardware.tests.test_validate_pynq_bundle_package hardware.tests.test_package_e2e_axis_dma_pynq_bundle hardware.tests.test_run_runtime_mode_board_latency hardware.tests.test_check_runtime_mode_board_latency_gate hardware.tests.test_run_zcu104_c3b_smoke_remote hardware.tests.test_import_pynq_smoke_result`: `63` tests passed.
- Remaining gap: no physical ZCU104 Search/Track result JSON has been captured for this combined runtime artifact yet. SW/HW equality, p95/p99 latency, measured DMA bandwidth, and measured board power remain pending until the two prepared bundles are executed on the board and imported into the canonical paths above.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 RTL Cosim Attempt

- User-directed physical timing rule for this lane remains fixed: PL clock target `300 MHz`, routed WNS accepted down to `-0.500 ns` for experiment continuation, and official clean Vivado timing signoff still requires WNS `>= 0.000 ns`.
- Ran HLS RTL co-simulation for the combined runtime profile:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`.
  - Profile flags included four dispatcher prefetch banks, prefetch-all-before-compute, CSim prefetch trace, and strict immediate-start checks.
- The flow reran CSynth before cosim and kept the target clock at `3.333 ns`:
  - Estimated clock `2.777 ns`.
  - Estimated Fmax `360.10 MHz`.
- C testbench/post-check evidence before Verilog simulation was clean:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`.
  - Search runtime state `0`, expected `0`, output count `6`, `last=1`, failures `0`.
  - Track output `[-235, -235, -235, -235, -235, -239]`.
  - Track runtime state `1`, expected `1`, output count `6`, `last=1`.
  - `E2E AXIS vector comparison passed`.
  - Search trace summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track trace summary `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Verilog cosim did not complete:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_no_board/solution_e2e_q4w8a/sim/report/hgtxr_e2e_axis_top_cosim.rpt`.
  - Verilog status: `Fail`; latency/interval fields are `NA`.
  - XSIM log: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_no_board/solution_e2e_q4w8a/sim/verilog/xsim.log`.
  - XSIM failure: `ERROR: unexpected exception when evaluating tcl command` while executing `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}`.
- Added a reproducible diagnosis helper:
  - Tool: `hardware/tools/diagnose_hls_cosim_xsim.py`.
  - JSON: `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.md`.
  - Diagnosis status: `blocked-xsim-launch`.
  - Checks: C testbench output match `true`, scheduler trace clean `true`, Search expected output `true`, Track expected output `true`, RTL cosim pass `false`, XSIM launch exception `true`, `-autoloadwcfg` in failure `true`.
- Interpretation: this attempt strengthens C-level Search/Track output and scheduler evidence inside the cosim flow, but it is not RTL functional signoff because Verilog XSIM failed before producing valid RTL latency/comparison results. The next RTL verification step is to isolate the XSIM launch exception, rerun Verilog cosim, and only then promote SW/HW same-input same-output beyond CSim/PYNQ bundle preparation.

## 2026-06-29 XSIM Launch Alternate-Probe Update

- Updated `hardware/tools/check_xsim_snapshot_smoke.py` to record additional launch probes for the minimal Verilog snapshot:
  - Standard `xsim tb_snapshot -R`.
  - Alternate `xsim tb_snapshot --runall`.
  - Alternate `xsim tb_snapshot --tclbatch run_all_no_wave.tcl`.
  - Direct snapshot-kernel execution through `xsim.dir/tb_snapshot/xsimk`.
- Regenerated `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.json` and `.md`.
- Result remains `blocked-xsim-runtime`:
  - `xvlog` passed.
  - `xelab` built snapshot `tb_snapshot`.
  - Standard and alternate `xsim` launches still source `xsim_script.tcl`, which injects `-autoloadwcfg` and fails with `ERROR: unexpected exception when evaluating tcl command`.
  - Direct `xsimk` execution enters the simulator kernel and prints `elaboration-done`, but does not run the testbench or emit `XSIM_SMOKE_PASS`; this is not a functional simulation pass.
- Regenerated `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`.
- Updated diagnosis result:
  - HGTXR C testbench output match: `true`.
  - Search/Track scheduler trace clean: `true`.
  - Search expected output: `true`.
  - Track expected output: `true`.
  - XELAB snapshot built: `true`.
  - RTL cosim pass: `false`.
  - XSIM environment smoke pass: `false`.
  - XSIM environment smoke blocked: `true`.
- Current conclusion: the RTL cosim blocker is further narrowed to the host XSIM wrapper/Tcl snapshot-launch path. The HGTXR RTL is not proven correct by this, but the failure remains independent of HGTXR RTL because even a trivial Verilog snapshot cannot launch through `xsim`.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 Resource Breakdown

- Added resource breakdown generator `hardware/tools/write_prefetchall4_300_resource_breakdown.py`.
- Generated artifacts:
  - JSON: `hardware/generated/signoff/prefetchall4_300_resource_breakdown_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/prefetchall4_300_resource_breakdown_2026_06_29.md`.
- Generated implemented Vivado reports from the existing post-route physopt DCP using `hardware/vivado/scripts/report_e2e_axis_dma_impl_utilization.tcl`:
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_utilization_implemented.rpt`.
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_utilization_hierarchical_implemented.rpt`.
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_timing_summary_implemented.rpt`.
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_route_status_implemented.rpt`.
  - `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_power_implemented.rpt`.
- Methodology:
  - Top resources are parsed from Vivado placed utilization for the active `prefetchall4_300` implementation.
  - Major block resources now prefer the implemented hierarchical utilization report; OOC synth utilization reports remain the fallback when the hierarchical report is absent.
- Top placed utilization remains:
  - CLB LUTs `73,133/230,400 = 31.74%`.
  - CLB registers `60,949/460,800 = 13.23%`.
  - Block RAM Tile `216/312 = 69.23%`.
  - URAM `76/96 = 79.17%`.
  - DSP `1,723/1,728 = 99.71%`.
- Major block implemented hierarchical highlights:
  - `pl_hgtxr_e2e_axis_top`: LUT `67,148`, FF `53,157`, Block RAM Tile `211`, URAM `76`, DSP `1,723`.
  - `axi_dma_in`: LUT `377`, FF `609`, Block RAM Tile `0.5`, URAM `0`, DSP `0`.
  - `axi_dma_out`: LUT `1,575`, FF `2,346`, Block RAM Tile `4.5`, URAM `0`, DSP `0`.
  - `axi_mem_interconnect`: LUT `3,310`, FF `5,211`, Block RAM Tile `0`, URAM `0`, DSP `0`.
  - `axi_control_interconnect`: LUT `785`, FF `891`, Block RAM Tile `0`, URAM `0`, DSP `0`.
  - `psu`: LUT `0`, FF `0`, Block RAM Tile `0`, URAM `0`, DSP `0`.
- Current conclusion for question 8:
  - ZCU104 absolute/percentage top utilization is available.
  - Major block implemented hierarchical breakdown is available and shows DSP/URAM pressure is entirely in `pl_hgtxr_e2e_axis_top`.
  - Q8 is now available for current evidence; remaining gaps in the broader eight-question set are board latency/DMA/power and RTL cosim, not resource attribution.
- Verification performed:
  - `python3 hardware/tools/write_prefetchall4_300_resource_breakdown.py`: status `pass`, missing block reports `[]`.
  - `python3 -m unittest hardware.tests.test_write_prefetchall4_300_resource_breakdown`: `3` tests passed.
  - `python3 hardware/tools/write_prefetchall4_300_goal_status.py`: goal-status now includes `resource_breakdown.status=pass`, all six major blocks as `implemented_hierarchical_utilization`, and Q8 status `available`.

## 2026-06-29 XSIM Direct Kernel Probe Update

- Updated `hardware/tools/check_xsim_snapshot_smoke.py` to drive direct `xsimk` with MI commands after snapshot elaboration:
  - `-exec-run`
  - `-exec-continue`
  - `-gdb-exit`
- Minimal Verilog smoke result:
  - JSON: `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.md`.
  - Status: `wrapper-blocked-kernel-pass`.
  - Standard `xsim tb_snapshot -R`, `xsim --runall`, and `xsim --tclbatch` still fail because the wrapper/Tcl launch injects `-autoloadwcfg` and reports `unexpected exception when evaluating tcl command`.
  - Direct `xsimk` enters the simulator kernel, completes `-exec-run`, completes `-exec-continue`, emits `XSIM_SMOKE_PASS`, and exits without timeout.
- Added HGTXR-specific direct-kernel probe:
  - Tool: `hardware/tools/check_hls_xsimk_direct_probe.py`.
  - JSON: `hardware/generated/signoff/prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.md`.
  - Status with 120 s timeout: `direct-kernel-timeout`.
  - Evidence: HGTXR snapshot exists, enters kernel, completes `-exec-run`, starts `-exec-continue`, prints the HLS RTL simulation progress banner, and reaches `RTL Simulation : 0 / 2 [0.00%] @ "109000"`.
  - Missing evidence: no Search/Track RTL output comparison yet, because the full two-transaction HLS cosim did not finish inside the bounded timeout.
- Regenerated `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`.
  - Diagnosis remains `blocked-xsim-launch` for standard HLS cosim.
  - New classification: normal XSIM wrapper/Tcl launch is broken, direct kernel execution is viable, and HGTXR RTL has started but not completed within the bounded probe.
- Current conclusion:
  - This is progress toward RTL verification because the simulator kernel can run snapshots directly.
  - It is still not RTL functional signoff; either the standard wrapper must be repaired, or the HGTXR direct `xsimk` probe must be allowed to run long enough to finish both transactions and capture the expected Search/Track outputs.

## 2026-06-29 Instrumented XSIMK Progress Probe

- Problem addressed:
  - The direct HGTXR `xsimk` probe with the generated HLS testbench printed only the initial `RTL Simulation : 0 / 2 [0.00%] @ "109000"` line.
  - The generated testbench uses `PROGRESS_TIMEOUT = 10000000`, so a long run can appear silent while the first transaction is still advancing.
- Added reproducible instrumented probe:
  - Tool: `hardware/tools/check_hls_xsimk_progress_probe.py`.
  - The tool copies the generated `sim/verilog` and sibling `sim/tv` directory to `/tmp/hgtxr_xsim_progress_probe_full_300`, patches only the copied testbench `PROGRESS_TIMEOUT` to `100000`, reruns `xelab`, then runs the copied snapshot through direct `xsimk`.
  - The hardware RTL design files are not modified by this instrumentation; only the copied HLS testbench progress-print interval is changed.
- Generated artifacts:
  - JSON: `hardware/generated/signoff/prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.json`.
  - Markdown: `hardware/generated/signoff/prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.md`.
- Result:
  - Status: `progress-timeout`.
  - `xelab` rebuilt the snapshot successfully.
  - direct `xsimk` entered the kernel, completed `-exec-run`, and started `-exec-continue`.
  - 120 s bounded run emitted 5 progress samples.
  - First progress: `0 / 2`, `0.00%`, sim time `109000`.
  - Last progress: `0 / 2`, `8.34%`, sim time `1336125000`.
- Interpretation:
  - The direct `xsimk` path is not frozen; RTL simulation time advances inside transaction 0.
  - It still did not complete Search/Track RTL output comparison inside the bounded run, so RTL functional signoff remains open.

## 2026-06-29 Tail16 Aggressive DSP/LUTRAM RTL Snapshot/Direct Probe

- Profile under test: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
- Purpose: take the strongest physical DSP/BRAM/URAM pressure-relief point and generate a reusable HLS RTL snapshot for the same direct `xsimk` verification path used on the balanced `tail2_dsppipe4` profile.
- Physical evidence recap:
  - Routed timing report: `clk_pl_0 = 300.030 MHz`, WNS `-0.500 ns`, TNS `-6623.548 ns`, WHS `0.006 ns`, THS `0.000 ns`; this exactly meets the user-approved experiment floor `WNS >= -0.500 ns` but is not official clean timing.
  - Post-route physopt/implemented timing report: WNS `-0.487 ns`, TNS `-6574.602 ns`, WHS `0.000 ns`, THS `0.000 ns`.
  - Route status reports are clean with routing errors `0`; the normal route report has `201,671/201,671` fully routed routable nets and the implemented checkpoint report has `201,673/201,673`.
  - Implemented utilization remains the strongest headroom point: DSP `1,573/1,728 = 91.03%`, Block RAM Tile `184/312 = 58.97%`, URAM `28/96 = 29.17%`, CLB LUT `113,354/230,400 = 49.20%`, LUT as Distributed RAM `35,024`.
- HLS cosim command:
  - `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`
- C testbench evidence before Verilog launch:
  - Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, runtime state `0`.
  - Track output `[-235, -235, -235, -235, -235, -239]`, runtime state `1`.
  - `E2E AXIS vector comparison passed`.
  - Search strict prefetch summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track strict prefetch summary `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- HLS memory mapping evidence:
  - Nonlinear `kGeluRom` is bound as LUTRAM ROM.
  - Frame `tokens` and selected `gb.*` buffers are bound as LUTRAM `ram_2p`, matching the intended small/high-bank-count BRAM-to-LUTRAM move.
  - Dense MLP/cache storage is intentionally not fully forced into LUTRAM because that would explode LUT/fanout risk.
- Verilog snapshot evidence:
  - XELAB built `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`.
  - Standard HLS Verilog cosim still fails at the host wrapper launch, not with a completed RTL data comparison: `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}` reports `ERROR: unexpected exception when evaluating tcl command`; `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` and latency `NA`.
- Direct `xsimk` short probe:
  - Artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_direct_probe_2026_06_29.json` and `.md`.
  - Status: `direct-kernel-timeout`.
  - Evidence: `xsimk_exists=true`, kernel entered, `-exec-run` complete, `-exec-continue` started, RTL progress banner observed, first progress `0 / 2 [0.00%] @ 109000`.
- Instrumented direct `xsimk` progress probe:
  - Artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_progress_probe_2026_06_29.json` and `.md`.
  - Status: `progress-timeout`.
  - XELAB rebuild passed in `/tmp/hgtxr_xsim_progress_probe_full_300`.
  - A 120 s run emitted 5 progress samples: first `0 / 2 [0.00%] @ 109000`, last `0 / 2 [8.32%] @ 1336125000`.
  - Estimated by the probe from observed progress: one transaction about `1442.62 s`; both transactions about `2885.24 s`.
- TV compare-only artifact:
  - Artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.json` and `.md`.
  - This early compare-only artifact reported `tv-output-mismatch`, but at that time the RTL transaction had not completed and RTL output files were missing/incomplete or placeholder-only.
- Long direct `xsimk` completion run:
  - Artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_direct_probe_3300s_2026_06_29.json` and `.md`.
  - Status: `completed-tv-output-mismatch`.
  - Runtime: `1546.076 s`, no timeout; RTL progress reached `2 / 2 [100.00%] @ 18080694000`.
  - TV files are now complete and present for all checked ports.
  - Mismatch: `axis_out_V_data_V` first payload line differs: C reference `0x...fb6f`, RTL `0x...0003`.
  - Matching ports after normalization: `axis_out_V_keep_V`, `axis_out_V_strb_V`, `axis_out_V_last_V`, and `gmem_e2e_runtime`.
- Current decision:
  - `tail16_lutbuf_dsppipe4` is the best physical DSP/BRAM/URAM relief candidate when headroom matters, because DSP drops to `91.03%`, BRAM Tile to `58.97%`, and URAM to `29.17%`.
  - It is still not official timing-clean signoff because WNS is negative.
  - It is not SW/HW same-input same-output signoff because the completed RTL run produces a real `axis_out_V_data_V` mismatch. The next functional debug target is the RTL AXIS TDATA/state-payload path for the pressure-relief profiles.

## 2026-06-29 Fixed-CSim Scale Path and Divider Pressure Follow-up

- Profile under test: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
- Purpose:
  - Make CSim use the same fixed-point type family as synthesis via `HGTXR_HLS_FIXED_CSIM`.
  - Emit raw fixed-point AXIS payload bits in CSim/synthesis paths so C TV and RTL TV use the same payload representation.
  - Keep the requested pressure-relief policy: fabric tail lanes for core multiplies and LUTRAM binding for small/high-bank-count buffers.
- Code changes validated:
  - `hardware/hls/include/fixed_types.h` now selects `ap_fixed` when `HGTXR_HLS_FIXED_CSIM` is set.
  - `hardware/hls/include/hgtxr_e2e_vit.hpp` adds compile-time scale helpers for constant scale conversions and uses token-count helpers for pressure profiles to reduce accidental divider inference.
  - The pressure-relief profiles in `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` now pass `-DHGTXR_HLS_FIXED_CSIM=1`; the fixed follow-up additionally passes `-DHGTXR_E2E_AVOID_RUNTIME_DIVIDERS=1`.
- CSim result after the fixed-point payload path:
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Strict prefetch trace remains clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth result after constant-scale/divider cleanup:
  - Target `3.333 ns`; estimated clock `2.846 ns` (`351.37 MHz`).
  - Top latency max `5,178,119` cycles / `17.259 ms` for controller, full top max remains dominated by frame/event input phases.
  - Resources: BRAM_18K `416`, DSP `2369`, FF `331918`, LUT `640454`, URAM `28`.
  - Major instance resources: `hgtxr_e2e_controller_run` BRAM_18K `280`, DSP `2364`, FF `320990`, LUT `550818`; conv/event/head remain small by comparison.
  - Memory evidence: nonlinear ROMs are LUTRAM/distributed ROM; `tokens` and many `gb.*` 3072x8 high-bank memories are LUTRAM; larger 12288x7/8 banks remain BRAM; `gb_dispatch_prefetch` remains URAM.
- Current judgment:
  - The fixed-point C reference no longer uses the old signed integer-scaled payloads (`0x...fb6f`); it now emits raw fixed payloads such as Search `0x...0003`.
  - This fixes the C reference representation issue, but this HLS estimate is not ZCU104-fit: DSP `2369/1728 = 137.1%` and LUT `640454/230400 = 277.97%`.
  - The remaining major DSP pressure is inside the shared Transformer controller/core multiply paths. Moving only tail lanes and small memories is insufficient for this fixed-Csim variant; a next candidate must either enable `HGTXR_E2E_CORE_ALL_FABRIC_MUL` for more lanes or reduce parallelism/cache fanout.
  - One dynamic softmax divider remains in the generated RTL (`sdiv_28ns_28ns_8_32_1`), so full divider removal is not yet complete.

## 2026-06-29 Core-All-Fabric / No-Forced-DSP / LUTRAM Pressure Probe

- Profile under test: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16`.
- Purpose:
  - Move the shared Transformer core multiply path away from DSP by enabling `HGTXR_E2E_CORE_ALL_FABRIC_MUL=1`.
  - Disable explicit DSP binding with `HGTXR_E2E_FORCE_DSP_MUL=0` and `HGTXR_E2E_FORCE_WIDE_DSP_MUL=0`.
  - Move small/high-bank-count buffers to LUTRAM: frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
- CSim result:
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Strict prefetch trace remains clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth result:
  - Target `3.333 ns`; estimated clock `2.979 ns` (`335.63 MHz`).
  - Top max latency `8,647,111` cycles / `28.821 ms`.
  - Controller max latency `5,173,511` cycles / `17.243 ms`.
  - Resources: BRAM_18K `416/624 = 66.7%`, DSP `2115/1728 = 122.4%`, FF `341844/460800 = 74.2%`, LUT `907718/230400 = 394.0%`, URAM `28/96 = 29.2%`.
  - Major pressure remains in `hgtxr_e2e_controller_run`: BRAM_18K `280`, DSP `2108`, FF `330428`, LUT `817790`.
- Memory mapping evidence:
  - `gb_dispatch_prefetch` remains URAM (`28` URAM).
  - Nonlinear ROMs and many `tokens` / `gb.*` 3072x8 memories are distributed/LUTRAM.
  - Larger `12288`-word banks remain BRAM, which is appropriate; pushing these to LUTRAM would further inflate the already failing LUT estimate.
- Current judgment:
  - LUTRAM migration worked, but the all-fabric/no-forced-DSP policy is not a viable ZCU104-fit point.
  - Disabling explicit DSP bind does not materially reduce the HLS DSP estimate because the remaining 21-24 bit multiply structures are still inferred as DSP by HLS/Vivado.
  - This probe reduces neither the DSP blocker nor the LUT blocker sufficiently; it should not be promoted to Vivado implementation.
  - Next DSE should not push more arithmetic blindly to fabric. It should reduce multiply width, remove/approximate the remaining softmax `sdiv_28ns_28ns_8_32_1`, or reduce parallelism/fanout while keeping the known physically routed `tail16_lutbuf_dsppipe4` and `tail2_toklutbuf_dsppipe4` points as references.

## 2026-06-29 Acc24 / Integer Nonlinear / LUTRAM Pressure Follow-up

- Purpose:
  - Respond to the DSP-pressure request by narrowing the shared accumulator type from the default `ap_fixed<32,12>` to `ap_fixed<24,10>` through `HGTXR_ACC_W=24` and `HGTXR_ACC_I=10`.
  - Push additional nonlinear paths to integer/LUT-oriented implementations with `HGTXR_E2E_USE_HGPIPE_INT_GELUQ`, `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ`, and `HGTXR_E2E_USE_HGPIPE_INT_LAYERNORMQ`.
  - Keep the targeted BRAM-to-LUTRAM move only for small or high-bank-count buffers: frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
  - Avoid moving large 12288-word dense banks into LUTRAM because the HLS estimates are already LUT-bound.
- Code changes:
  - `hardware/hls/include/fixed_types.h` now exposes macro-configurable accumulator width/integer width through `HGTXR_ACC_W` and `HGTXR_ACC_I`.
  - `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` now has two selectable 300 MHz profiles:
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_intnl_lutbuf_acc24_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_intnl_lutbuf_acc24_300_mem16`
- CSim result, both profiles:
  - Search output `[244, 244, 244, 244, 244, 242]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[200, 200, 200, 200, 200, 196]`, runtime state `1`, `count=6`, `last=1`.
  - Strict prefetch trace remains clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth results:

| Profile | Est. clock | Top max cycles | Top latency @300 MHz | BRAM_18K | DSP | FF | LUT | URAM | Judgment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `tail16_intnl_lutbuf_acc24_dsppipe4` | `2.846 ns` | `8,611,527` | `28.702 ms` | `416/624 = 66%` | `2323/1728 = 134%` | `316855/460800 = 68%` | `621865/230400 = 269%` | `28/96 = 29%` | Reject |
| `coreallfabric_intnl_lutbuf_acc24` | `2.846 ns` | `8,586,695` | `28.619 ms` | `416/624 = 66%` | `2067/1728 = 119%` | `322437/460800 = 69%` | `721301/230400 = 313%` | `28/96 = 29%` | Reject |

- Major pressure evidence:
  - `tail16_intnl_lutbuf_acc24_dsppipe4`: `hgtxr_e2e_controller_run` estimates BRAM_18K `280`, DSP `2318`, FF `306972`, LUT `538084`.
  - `coreallfabric_intnl_lutbuf_acc24`: `hgtxr_e2e_controller_run` estimates BRAM_18K `280`, DSP `2062`, FF `312462`, LUT `637452`.
  - Conv/event/head remain small relative to the shared Transformer controller: conv DSP `1`, event DSP `1`, head DSP `2`.
- Memory mapping evidence:
  - Nonlinear ROMs and selected `tokens`/`gb.*` buffers are generated as LUTRAM/distributed RAM.
  - Large QKV/MLP weight caches still use BRAM, and dispatcher prefetch remains URAM. This is intentional because moving those dense banks to LUTRAM would worsen an already failing LUT estimate.
- Comparison against prior fixed pressure probes:
  - Versus fixed `tail16_lutbuf_dsppipe4`, `tail16_intnl_lutbuf_acc24_dsppipe4` reduces DSP from `2369` to `2323` and LUT from `640454` to `621865`, but it still exceeds both DSP and LUT device capacity.
  - Versus prior `coreallfabric_nowide_lutbuf`, `coreallfabric_intnl_lutbuf_acc24` reduces DSP from `2115` to `2067` and LUT from `907718` to `721301`, but it still exceeds DSP capacity by `339` DSPs and LUT capacity by about `490901` LUTs.
- Current decision:
  - The requested techniques work locally: accumulator narrowing and integer nonlinear paths reduce DSP/LUT estimates, and small/high-bank memories can be moved out of BRAM into LUTRAM.
  - These two new HLS points are not ZCU104 implementation candidates. The best reduced-DSP point still estimates DSP `2067/1728 = 119%` and LUT `721301/230400 = 313%`.
  - Do not launch Vivado implementation for either new profile. The next viable DSE must reduce parallelism/fanout or restructure QKV/MLP weight-cache reuse instead of blindly moving more multiply work from DSP into LUT fabric.

## 2026-06-29 Par16/Mem8 Runtime-ROM Fit Probe

- Purpose:
  - Test whether reducing compute parallelism and memory-bank fanout is a better ZCU104-fit direction than moving more arithmetic into LUT fabric.
  - Keep the actual runtime-ROM full learned Transformer conditions active: learned Frame Conv/Event Conv/Head/Track resident weight path through on-chip parameter ROM, Search dispatcher path for later Search layers, on-chip nonlinear ROM, runtime Search/Track scheduler, and Search/Track CSim coverage.
  - New profile: `par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8`.
- Profile changes versus the prior par32 token-LUTRAM pressure point:
  - `E2E_PAR=16`, `MEM_BANK_PAR=8`.
  - Accumulator remains narrowed through `HGTXR_ACC_W=24`, `HGTXR_ACC_I=10`.
  - Integer nonlinear paths remain enabled for GELU, softmax, and layernorm.
  - Small token buffers remain LUTRAM; dense weight/cache buffers and dispatcher prefetch are not forced into LUT fabric.
- CSim result:
  - Search output `[244, 244, 244, 244, 244, 242]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Search dispatcher prefetch was ordered before use for blocks 2 and 3: `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track output `[200, 200, 200, 200, 200, 196]`, runtime state `1`, `count=6`, `last=1`.
  - Track path remained immediate between its two resident blocks: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth result:

| Metric | Result |
|---|---:|
| Target clock | `3.333 ns` |
| Estimated clock | `2.846 ns` / `351.37 MHz` |
| Top max latency | `13,077,605 cycles` / `43.588 ms` |
| Controller max latency | `9,616,135 cycles` / `32.051 ms` |
| BRAM_18K | `400/624 = 64%` |
| DSP | `2289/1728 = 132%` |
| FF | `294526/460800 = 63%` |
| LUT | `485714/230400 = 210%` |
| URAM | `108/96 = 112%` |

- Major pressure evidence:
  - `hgtxr_e2e_controller_run` remains dominant: BRAM_18K `280`, DSP `2286`, FF `288292`, LUT `448016`.
  - Frame Conv/Event Conv/Head are not the fit blockers: conv DSP `1`, event DSP `1`, head DSP `0`.
  - Memory total shows `69` banks instead of the broader par32 banking, and token buffers are distributed RAM, but URAM rises to `108` because norm/q/k/v/attn and dispatcher prefetch remain URAM-bound under this policy.
- Decision:
  - Reject this par16/mem8 candidate. It keeps the correct runtime-ROM/prefetch/full-controller behavior and passes CSim, but it is worse than the best par32 physical candidates for the final objective.
  - Lowering `E2E_PAR` without restructuring QKV/MLP reuse doubles controller latency and does not solve DSP/LUT/URAM fit.
  - The next useful path is not a simple par16 shrink. Continue from the physically routed par32 pressure-relief candidates and focus on timing cleanup plus RTL AXIS TDATA equality, or redesign the QKV/MLP weight-cache reuse to reduce controller replication.

## 2026-06-29 Aux-Fabric / Norm-LUTRAM Pressure Probe

- Purpose:
  - Add selective, profile-controlled DSP-to-fabric switches for non-core blocks that still used direct `hgtxr_e2e_dsp_mul*` calls.
  - Add an explicit `HGTXR_E2E_LUTRAM_HIDDEN` storage switch for future experiments while keeping the large hidden buffer off by default.
  - Test a conservative memory move that keeps `tokens`/`gb.tokens` in LUTRAM and additionally moves `gb.norm` to LUTRAM, without forcing large Q/K/V/ATTN/hidden buffers into LUT fabric.
- Code changes:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp` now defines:
    - `HGTXR_E2E_PATCH_FABRIC_MUL`
    - `HGTXR_E2E_LAYERNORM_FABRIC_MUL`
    - `HGTXR_E2E_FASTPATH_FABRIC_MUL`
    - `HGTXR_E2E_LUTRAM_HIDDEN`
  - Frame/Event Conv multiply calls now route through `hgtxr_e2e_patch_mul()`.
  - LayerNorm variance/affine multiply calls now route through `hgtxr_e2e_layernorm_mul_acc()`.
  - Fastpath/datapath mix multiply calls now route through `hgtxr_e2e_fastpath_mul()`.
  - `hardware/hls/src/hgtxr_e2e_axis_top.cpp` and `hardware/hls/src/hgtxr_e2e_m_axi_top.cpp` can bind `gb.hidden` to LUTRAM when `HGTXR_E2E_LUTRAM_HIDDEN=1`.
  - `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` now exposes `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
- CSim result:
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Search dispatcher prefetch remained ordered: `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Track resident path remained immediate: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth result:

| Metric | Prior `tail2_toklutbuf_dsppipe4` | New `auxfabric_normlut` | Delta |
|---|---:|---:|---:|
| Target clock | `3.333 ns` | `3.333 ns` | `0` |
| Estimated clock | `2.846 ns` / `351.37 MHz` | `3.167 ns` / `315.77 MHz` | slower, still >300MHz |
| Top max latency | n/a in this comparison table | `8,651,783 cycles` / `28.836 ms` | n/a |
| BRAM_18K | `416/624 = 66%` | `416/624 = 66%` | `0` |
| DSP | `2593/1728 = 150%` | `2567/1728 = 148%` | `-26` |
| FF | `324481/460800 = 70%` | `324721/460800 = 70%` | `+240` |
| LUT | `549344/230400 = 238%` | `564547/230400 = 245%` | `+15203` |
| URAM | `108/96 = 112%` | `92/96 = 95%` | `-16` |

- Major pressure evidence:
  - Frame Conv and Event Conv now estimate `DSP=0`, confirming the selective conv DSP-to-fabric switch worked.
  - `hgtxr_e2e_controller_run` remains dominant: BRAM_18K `280`, DSP `2564`, FF `314692`, LUT `499010`.
  - `gb.norm` LUTRAM movement reduces URAM pressure from `108` to `92`, but does not reduce total BRAM because the remaining BRAM is dominated by large weight/cache memories.
  - LUT rises by `15203`, so this direction improves DSP/URAM but worsens an already failing LUT estimate.
- Decision:
  - Keep the new switches because they provide controlled DSE levers and do not change default behavior.
  - Reject `auxfabric_normlut` as a ZCU104 implementation candidate. It passes CSim and HLS 300MHz timing estimate, but DSP and LUT remain far above device capacity.
  - Do not move `gb.hidden`, Q/K/V/ATTN, or large weight caches into LUTRAM by default. Those buffers are too large for the current LUT-bound design.

## 2026-06-29 Weight-Cache / Tail8 Aux-Fabric Follow-up

- Purpose:
  - Test whether removing local dense weight-vector caches can cut BRAM pressure without changing the runtime-ROM scheduler semantics.
  - Test whether a wider `tail8` fabric-multiply policy plus small-buffer LUTRAM migration is a useful follow-up to the prior `tail2` aux-fabric probe.
  - Keep the 300 MHz HLS target, runtime Search/Track mode scheduler, learned parameter path, on-chip nonlinear ROM path, and strict dispatcher prefetch checks active.
- New profiles:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16`
    - Disables QKV and WO/W1/W2 weight-vector caches through `HGTXR_E2E_QKV_WEIGHT_CACHE=0` and `HGTXR_E2E_WEIGHT_VEC_CACHE=0`.
    - Keeps token LUTRAM and nonlinear LUTRAM, but does not force Q/K/V/ATTN or large hidden buffers into LUTRAM.
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16`
    - Enables `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=8` plus patch, LayerNorm, and fastpath fabric-multiply switches.
    - Enables LUTRAM for tokens, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
- CSim result:
  - Both profiles passed Search and Track CSim.
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Strict prefetch traces stayed clean for both profiles: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth comparison:

| Profile | Estimated clock | Top max latency | BRAM_18K | DSP | FF | LUT | URAM | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tail2_toklutbuf_nocache` | `2.846 ns` / `351.37 MHz` | `8,645,943 cycles` / `28.817 ms` | `136/624 = 21%` | `2209/1728 = 127%` | `252925/460800 = 54%` | `523286/230400 = 227%` | `108/96 = 112%` | Reject: BRAM improves, but DSP/LUT/URAM still fail |
| Existing `tail8_lutbuf_dsppipe4` reference | `2.777 ns` / `360.10 MHz` | `4,807,861 cycles` / `16.025 ms` | `336/624 = 53%` | `1778/1728 = 102%` | `245421/460800 = 53%` | `441342/230400 = 191%` | `28/96 = 29%` | Reference only |
| `tail8_lutbuf_auxfabric_normlut` | `3.167 ns` / `315.77 MHz` | `8,650,183 cycles` / `28.831 ms` | `416/624 = 66%` | `2471/1728 = 142%` | `328436/460800 = 71%` | `615001/230400 = 266%` | `28/96 = 29%` | Reject: worse than `tail8` reference |

- Major pressure evidence:
  - `tail2_toklutbuf_nocache` confirms that disabling dense weight caches removes a large BRAM source, dropping total BRAM_18K to `136`, but it exposes unresolved controller pressure: `hgtxr_e2e_controller_run` still estimates DSP `2204` and LUT `464364`, and total URAM remains over capacity at `108`.
  - `tail8_lutbuf_auxfabric_normlut` is worse than the existing `tail8_lutbuf_dsppipe4` reference: DSP increases from `1778` to `2471`, LUT from `441342` to `615001`, BRAM from `336` to `416`, and latency from `16.025 ms` to `28.831 ms`.
  - The useful small-buffer LUTRAM move is limited to token-like high-bank memories. Large 12288-word banks, dense QKV/MLP caches, and hidden buffers should not be blindly moved to LUTRAM in the current LUT-bound design.
- Decision:
  - Keep both profile entries as DSE evidence and reproducible pressure probes.
  - Do not promote either new profile to Vivado implementation.
  - The next useful implementation work is RTL AXIS `TDATA` mismatch debug for the physical par32 candidates and a real QKV/MLP reuse/fanout restructuring, not broader DSP-to-LUT or BRAM-to-LUTRAM migration.

## 2026-06-29 Tail16 AXIS Payload Contract Recheck

- Purpose:
  - Separate the previous `axis_out_V_data_V` mismatch into stale RTL structure, active payload width, and missing RTL TV evidence.
  - Avoid using old `tail2`/`tail16` TV files as signoff evidence after source-level fixes changed the AXIS payload writer.
  - Keep the same physical pressure-relief candidate: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
- Added diagnostic tool:
  - `hardware/tools/check_hls_axis_payload_contract.py`
  - It audits generated `syn/verilog/hgtxr_e2e_axis_top.v`, detects constant-zero `axis_out_TDATA` writer paths, records out_state payload width, and compares existing C/RTL TV data by active low payload bits.
- Stale snapshot finding:
  - The old `tail2_dsppipe4` synthesis snapshot still contains `hgtxr_axis_write_state_values` with `assign axis_out_TDATA = 256'd0`.
  - Diagnostic status: `fail-constant-axis-tdata`.
  - Decision: old `tail2_dsppipe4` RTL TV mismatch is stale and must not be used as evidence against the current source.
- Current `tail16_lutbuf_dsppipe4` CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Search dispatcher trace is clean: `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track resident path trace is clean: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Current `tail16_lutbuf_dsppipe4` CSynth:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`
  - Target clock `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
  - Controller max latency `5,178,119 cycles` / `17.259 ms`.
  - Resource estimate: BRAM_18K `416/624 = 66%`, DSP `2369/1728 = 137%`, FF `331918/460800 = 72%`, LUT `640454/230400 = 278%`, URAM `28/96 = 29%`.
  - Dominant instance remains `hgtxr_e2e_controller_run`: BRAM_18K `280`, DSP `2364`, FF `320990`, LUT `550818`.
  - Generated RTL still contains `sdiv_28ns_28ns_8_32_1` in the attention softmax path.
- Current AXIS payload contract audit:
  - Artifact: `hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_axis_payload_contract_2026_06_29.{json,md}`.
  - Status: `structural-pass-missing-rtl-tv`.
  - Generated top exists and has no constant-zero `axis_out_TDATA` assignment.
  - `axis_out_TDATA` is driven from out_state-derived `zext_ln545*` sources.
  - Out_state width is `[8]`, consistent with the active `runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_active64_b8_ff768` scale.
  - C/RTL TV files are absent after the CSim/CSynth-only regeneration; this is not RTL functional signoff.
- Decision:
  - The stale constant-zero writer bug is proven absent from the current `tail16_lutbuf_dsppipe4` synthesis RTL.
  - The current blocker is now missing completed RTL TV equality for the current regenerated snapshot, plus unresolved resource fit.
  - Next verification gate: rerun bounded direct `xsimk` on the current `tail16` snapshot and compare active 8-bit payload TV if RTL TV is emitted.
  - Next resource gate: remove or approximate the remaining softmax divider and restructure QKV/MLP reuse/fanout; additional broad DSP-to-LUT migration is not a promising path.

## 2026-06-29 Tail16 Direct XSIMK Timeout / Partial TV Result

- Purpose:
  - Reuse the regenerated `tail16_lutbuf_dsppipe4` RTL snapshot after HLS cosim reached XELAB snapshot generation but failed in the XSIM Tcl wrapper.
  - Bypass the wrapper through direct `xsimk` MI commands and then compare the active low 8-bit AXIS payload TV.
- HLS cosim result:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`
  - C TB passed and generated the same Search `[3, 3, 3, 3, 3, 11]` and Track `[217, 217, 217, 217, 217, 213]` vectors.
  - XELAB built `sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`.
  - Standard XSIM wrapper still failed with `unexpected exception when evaluating tcl command`.
- Direct `xsimk` probe:
  - Artifact: `hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_current_hls_xsimk_direct_probe_2026_06_29.{json,md}`.
  - Status: `direct-kernel-timeout`.
  - The probe entered the RTL kernel, completed `-exec-run`, started `-exec-continue`, and reached RTL progress `1 / 2` transactions at simulation time `28632442000`.
  - The run timed out at `3600.071 s` before completing the second transaction.
- Active 8-bit payload contract after the partial RTL run:
  - Artifact: `hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_axis_payload_contract_2026_06_29.{json,md}`.
  - Status: `fail-active-payload-tv`, but the mismatch is caused by missing second-transaction RTL TV, not by a Search payload mismatch.
  - C active payloads: `[3, 3, 3, 3, 3, 11, -39, -39, -39, -39, -39, -43]`.
  - RTL active payloads emitted before timeout: `[3, 3, 3, 3, 3, 11]`.
  - Interpretation: Search transaction active payload matches C; Track transaction did not finish within the timeout, so the RTL TV is incomplete.
- Decision:
  - The current `tail16_lutbuf_dsppipe4` snapshot no longer has the stale constant-zero AXIS writer issue, and direct RTL execution proves the first transaction can emit the expected active 8-bit Search payload.
  - This is still not full RTL functional signoff because Track transaction completion is missing.
  - Do not add broader DSP-to-LUT or BRAM-to-LUTRAM migration now. Existing DSE shows that broad migration worsens LUT pressure; the next useful work is to reduce controller QKV/MLP fanout or remove/approximate the remaining softmax divider.

## 2026-06-29 Tail16 SoftmaxQ Divider-Removal Probe

- Purpose:
  - Test a focused pressure-relief candidate after the `tail16_lutbuf_dsppipe4` report showed a remaining generated attention softmax divider.
  - Isolate the softmax quantized-reciprocal path without mixing in accumulator narrowing, integer GELU, or integer LayerNorm.
  - Keep the already requested small/high-bank memory policy: nonlinear ROMs and selected token/Q/K/V/ATTN-like scratch buffers stay in LUTRAM; large dense banks and dispatcher prefetch are not forced into LUT fabric.
- Profile:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`
  - Delta versus `tail16_lutbuf_dsppipe4`: adds `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1`.
  - It preserves `E2E_PAR=32`, `MEM_BANK_PAR=16`, target `3.333 ns`, runtime Search/Track scheduling, Search prefetch-all4, nonlinear LUTRAM, selected small-buffer LUTRAM, tail16 fabric lanes, and DSP pipe latency `4`.
- CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`
  - Search output `[0, 0, 0, 0, 0, 238]`, runtime state `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[237, 237, 237, 237, 237, 233]`, runtime state `1`, `count=6`, `last=1`.
  - Search dispatcher trace stayed clean: `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track resident path trace stayed clean: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  - The output changes relative to `tail16_lutbuf_dsppipe4` are expected for this quantized softmax approximation; C reference and HLS execution remain self-consistent.
- CSynth:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`
  - Target clock `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
  - Top max latency `8,620,231 cycles` / `28.731 ms`.
  - Controller max latency `5,158,919 cycles` / `17.195 ms`.
  - Resource estimate: BRAM_18K `416/624 = 66%`, DSP `2371/1728 = 137%`, FF `328898/460800 = 71%`, LUT `640308/230400 = 277%`, URAM `28/96 = 29%`.
  - Dominant instance remains `hgtxr_e2e_controller_run`: BRAM_18K `280`, DSP `2366`, FF `317970`, LUT `550672`.
- Divider check:
  - `rg -n "sdiv_28ns_28ns_8_32_1" .../solution_e2e_q4w8a -S` found no match in the new project.
  - Remaining `udiv`-named signals in RTL are narrow index/constant-division artifacts, not the prior softmax `sdiv_28ns_28ns_8_32_1` module.
- Decision:
  - Reject this candidate for promotion. It removes the specific softmax divider artifact, but does not relieve the actual fit pressure: DSP changes `2369 -> 2371`, LUT is effectively unchanged, and top latency regresses `17.259 ms -> 28.731 ms`.
  - Keep `tail16_lutbuf_dsppipe4` as the best headroom-oriented HLS/physical pressure-relief reference and `lutrom_tail2_dsppipe4` as the better routed timing reference.
  - Do not broaden DSP-to-LUT or BRAM-to-LUTRAM migration now. The measured blockers are controller QKV/MLP multiply/fanout and reuse structure, not Conv/Event/Head or token-like memory binding.

## 2026-06-29 Tail16 Shared Runtime Unit DSE

- Purpose:
  - Test the more structural resource-relief path implied by the cyclic accelerator target: the runtime ATTN/MLP units should be temporally shared, not duplicated as independent `<0>` and `<1>` template instances.
  - Keep the same full learned runtime-ROM candidate conditions as `tail16_lutbuf_dsppipe4`: 300 MHz HLS target, `E2E_PAR=32`, `MEM_BANK_PAR=16`, runtime Search/Track scheduling, Search prefetch-all4, on-chip parameter/nonlinear ROMs, selected small/high-bank LUTRAM buffers, and DSP pipe latency `4`.
- Code/profile change:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp` adds `HGTXR_E2E_SHARE_RUNTIME_UNITS`.
  - When enabled, fastpath, structured, and full learned runtime paths always call ATTN/MLP unit `<0>` rather than selecting `<0>` or `<1>` by block parity.
  - This is behavior-preserving for the current implementation because `UNIT_ID` is not used inside the ATTN/MLP unit logic; it only changed HLS template instantiation and resource duplication.
  - `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` adds profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`.
- CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, expected `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, expected `1`, `count=6`, `last=1`.
  - Search dispatcher trace stayed clean: `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - Track resident path trace stayed clean: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  - Result: `E2E AXIS vector comparison passed`.
- CSynth:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Target clock `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
  - Top max latency `8,639,430 cycles` / `28.795 ms`.
  - Controller max latency `5,178,118 cycles` / `17.259 ms`.
  - Resource estimate: BRAM_18K `276/624 = 44%`, DSP `1057/1728 = 61%`, FF `165590/460800 = 35%`, LUT `372037/230400 = 161%`, URAM `28/96 = 29%`.
  - Dominant instance remains `hgtxr_e2e_controller_run`: BRAM_18K `140`, DSP `1052`, FF `154662`, LUT `282401`.
- Delta versus current `tail16_lutbuf_dsppipe4` baseline:

| Metric | `tail16_lutbuf_dsppipe4` | `tail16_lutbuf_shareunit_dsppipe4` | Delta |
|---|---:|---:|---:|
| Top max latency | `8,639,431 cycles` | `8,639,430 cycles` | `-1 cycle` |
| Controller max latency | `5,178,119 cycles` | `5,178,118 cycles` | `-1 cycle` |
| BRAM_18K | `416` | `276` | `-140` |
| DSP | `2369` | `1057` | `-1312` |
| FF | `331918` | `165590` | `-166328` |
| LUT | `640454` | `372037` | `-268417` |
| URAM | `28` | `28` | `0` |

- Structural evidence:
  - Negative check for `hgtxr_e2e_mlp_unit<1>`, `hgtxr_e2e_mlp_unit_1`, `hgtxr_e2e_attn_unit<1>`, and `hgtxr_e2e_attn_unit_1` over the shareunit `syn/report` and `syn/verilog` directories found no matches.
  - Positive check in `csynth_design_size.rpt` shows only `hgtxr_e2e_attn_unit<0>` and `hgtxr_e2e_mlp_unit<0>` under the controller.
- Decision:
  - Promote `tail16_lutbuf_shareunit_dsppipe4` as the new primary HLS resource-relief candidate. It directly implements the intended temporal sharing of the runtime ATTN/MLP compute units and removes the largest duplicated resource source without changing CSim outputs or latency.
  - The design is still not ZCU104-fit by HLS LUT estimate: LUT remains `372037/230400 = 161%`.
  - The next gate is controller LUT/fanout reduction and/or a Vivado implementation attempt on the shareunit candidate to determine how much of the remaining HLS LUT estimate survives physical synthesis.

## 2026-06-29 Shareunit LUT Relief Follow-up

- Purpose:
  - Continue the resource-fit path after `tail16_lutbuf_shareunit_dsppipe4` brought DSP under capacity but left LUT at `161%`.
  - Use the new DSP headroom to move selected fabric multiply paths back to DSP and reduce the separate Head parallelism without changing the full learned Transformer runtime path.
- Code/profile changes:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp` adds `HGTXR_E2E_HEAD_PAR` and uses it only inside `hgtxr_e2e_mlp_head`.
  - The Transformer body keeps `HGTXR_E2E_DENSE_PAR=32`; Head can now be synthesized with a smaller local parallelism such as `8`.
  - `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` adds:
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_headpar8_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16`
- CSim:
  - All five follow-up profiles passed CSim with unchanged Search and Track outputs.
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Strict prefetch traces remained clean for both modes: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- CSynth comparison:

| Profile | Main delta | Estimated clock | Top max latency | BRAM_18K | DSP | FF | LUT | URAM | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `tail16_lutbuf_shareunit_dsppipe4` | Shared ATTN/MLP, tail16 fabric lanes | `2.846 ns` / `351.37 MHz` | `8,639,430 cycles` / `28.795 ms` | `276/624 = 44%` | `1057/1728 = 61%` | `165590/460800 = 35%` | `372037/230400 = 161%` | `28/96 = 29%` | Previous primary resource-relief candidate |
| `lutbuf_shareunit_dsppipe4` | Removes tail16 fabric lanes; keeps selected LUTRAM | `2.846 ns` / `351.37 MHz` | `8,639,430 cycles` / `28.795 ms` | `276/624 = 44%` | `1177/1728 = 68%` | `162265/460800 = 35%` | `339682/230400 = 147%` | `28/96 = 29%` | Better LUT/DSP tradeoff, still not fit |
| `lutrom_shareunit_dsppipe4` | Removes selected LUTRAM; uses BRAM/URAM instead | `2.846 ns` / `351.37 MHz` | `8,640,519 cycles` / `28.799 ms` | `340/624 = 54%` | `1185/1728 = 68%` | `160464/460800 = 34%` | `294575/230400 = 127%` | `108/96 = 112%` | Reject: LUT improves but URAM exceeds device |
| `lutbuf_shareunit_headpar8_dsppipe4` | Removes tail16 fabric lanes and reduces Head local parallelism to 8 | `2.846 ns` / `351.37 MHz` | `8,637,990 cycles` / `28.790 ms` | `276/624 = 44%` | `1183/1728 = 68%` | `157807/460800 = 34%` | `313925/230400 = 136%` | `28/96 = 29%` | New best balanced HLS candidate, still not fit |
| `normlut_shareunit_headpar8_dsppipe4` | Keeps only `gb.norm` in LUTRAM; returns other scratch buffers to BRAM/URAM | `2.846 ns` / `351.37 MHz` | `8,640,615 cycles` / `28.799 ms` | `340/624 = 54%` | `1183/1728 = 68%` | `156547/460800 = 33%` | `277119/230400 = 120%` | `92/96 = 95%` | New best resource candidate; still not fit by LUT |
| `normlut_shareunit_headpar8_nocache_dsppipe4` | Disables QKV/LN caches; uses dispatcher/on-chip ROM reads directly | `2.846 ns` / `351.37 MHz` | `8,630,387 cycles` / `28.765 ms` | `295/624 = 47%` | `685/1728 = 39%` | `104198/460800 = 22%` | `225520/230400 = 97%` | `92/96 = 95%` | New best HLS fit candidate |

- Key observations:
  - Removing tail16 fabric lanes uses DSP headroom effectively: DSP rises from `1057` to about `1177`, still below ZCU104 capacity, while LUT drops by about `32k`.
  - Moving selected scratch buffers out of LUTRAM removes memory LUT pressure, but HLS maps them to URAM and exceeds capacity: `108/96 = 112%`.
  - `HGTXR_E2E_HEAD_PAR=8` reduces the Head instance from LUT `31259` to `7659` and DSP `2` to `0`, while preserving CSim outputs. Head latency increases only from `82.905 us` to `83.225 us`, which is negligible at the current top latency scale.
  - A selective LUTRAM policy is better than all-or-nothing binding. `normlut_shareunit_headpar8_dsppipe4` keeps only `gb.norm` in LUTRAM, reducing memory LUT from `43008` to `6144` while keeping URAM within capacity at `92/96`.
  - Disabling QKV/LN local caches removes a large controller fanout source while preserving CSim outputs. Controller resource drops from BRAM `140`, DSP `1180`, FF `151226`, LUT `247909` to BRAM `95`, DSP `682`, FF `99109`, LUT `196336`.
  - `normlut_shareunit_headpar8_nocache_dsppipe4` is the first candidate that fits the ZCU104 resource envelope by HLS estimate: LUT `97%`, DSP `39%`, BRAM `47%`, URAM `95%`.
- Decision:
  - Promote `normlut_shareunit_headpar8_nocache_dsppipe4` as the new primary HLS fit candidate.
  - Next gate is physical Vivado implementation at 300 MHz. URAM is tight at `92/96 = 95%`, so implementation must check route feasibility, timing, and power rather than relying on HLS fit alone.

### Vivado implementation result

- Command:
  - `sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16`
- Implemented artifact:
  - Overlay directory: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_overlay`
  - Bitstream: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16.bit`
  - Handoff: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16.hwh`
- Route status:
  - Fully routed nets: `115081/115081`.
  - Routing errors: `0`.
- Timing at 300 MHz:

| Clock | Period | Frequency | WNS | TNS | WHS | THS | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `clk_pl_0` | `3.333 ns` | `300.030 MHz` | `-0.171 ns` | `-286.331 ns` | `0.005 ns` | `0.000 ns` | Vivado reports timing not met, but within the user-allowed `-0.5 ns` WNS tolerance |

- Implemented utilization:

| Resource | Used | Available | Utilization |
|---|---:|---:|---:|
| CLB LUTs | `61679` | `230400` | `26.77%` |
| LUT as Logic | `52460` | `230400` | `22.77%` |
| LUT as Distributed RAM | `8820` | `101760` | `8.67%` |
| CLB Registers | `42737` | `460800` | `9.27%` |
| Block RAM Tile | `154.5` | `312` | `49.52%` |
| RAMB36 | `142` | `312` | `45.51%` |
| RAMB18 | `25` | `624` | `4.01%` |
| URAM | `92` | `96` | `95.83%` |
| DSP48E2 | `623` | `1728` | `36.05%` |

- Vector-less implemented power estimate:

| Metric | Power |
|---|---:|
| Total on-chip | `6.642 W` |
| Dynamic | `5.917 W` |
| Device static | `0.725 W` |
| PS static | `0.102 W` |
| PL static | `0.623 W` |
| Clocks dynamic | `0.419 W` |
| CLB logic dynamic | `0.753 W` |
| Signals dynamic | `1.013 W` |
| Block RAM dynamic | `0.159 W` |
| URAM dynamic | `0.245 W` |
| DSP dynamic | `0.657 W` |
| PS8 dynamic | `2.671 W` |

- Caveats:
  - Power confidence is `Medium` and no SAIF/VCD activity file was used. Vivado warned that high-fanout reset activity may make vector-less power inaccurate.
  - The implemented profile proves ZCU104 fit, route completion, bitgen, and 300 MHz implementation within the allowed negative-WNS tolerance.

### Mode-specific force-mode CSynth for implemented candidate

- Commands:
  - `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only`
  - `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only`
- Evidence reports:
  - `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`

| Mode/profile | HLS target | HLS estimated clock | Latency cycles | Interval cycles | Latency @ 300 MHz | Interval @ 300 MHz | Target status |
|---|---:|---:|---:|---:|---:|---:|---|
| Search-only forced | `3.333 ns` | `2.846 ns` / `351.37 MHz` | `8,604,864` | `8,604,865` | `28.683 ms` | `28.683 ms` | Fail vs `4 ms` |
| Track-only forced, min-bound | `3.333 ns` | `3.473 ns` / `287.94 MHz` | `698,166` | `698,167` | `2.327 ms` | `2.327 ms` | Fail vs `1 ms` |
| Track-only forced, max-bound | `3.333 ns` | `3.473 ns` / `287.94 MHz` | `4,003,638` | `4,003,639` | `13.345 ms` | `13.345 ms` | Fail vs `1 ms` |

| Hybrid assumption | Expected interval cycles | Expected interval @ 300 MHz | Expected rate | Worst-case interval |
|---|---:|---:|---:|---:|
| Search `10%`, Track `90%`, using Track min-bound | `1,488,837` | `4.963 ms` | `201.5 invocations/s` | Search `28.683 ms` |
| Search `10%`, Track `90%`, using Track max-bound | `4,463,762` | `14.879 ms` | `67.2 invocations/s` | Search `28.683 ms` |

- Path latency breakdown from the force-mode top reports:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `3,293,188`, `global_buffer_load` `12,292`, `controller_run` `5,168,162`, `mlp_head` `24,970`, observable mix `89,849`.
  - Track-only min-bound: `axis_read_frame` `4,098`, `event_conv_patch_embedding` `12,484`, `global_buffer_load` `3,076`, `controller_run` `628,712`, `mlp_head` `6,538`, observable mix `43,241`.
  - Track-only max-bound: the range is dominated by `event_conv_patch_embedding` max `3,317,956 cycles`; keep the min-bound for the intended Track smoke/TB case and the max-bound for worst-case latency reporting.
- Decision:
  - The force-mode measurements now replace the earlier combined min/max envelope as mode-specific HLS evidence for this exact implemented candidate.
  - The design remains physically useful because Vivado route/bitgen and ZCU104 fit are proven, but it still fails the Search `4 ms` and Track `1 ms` latency targets.
  - Track-only force-mode HLS estimated clock is `3.473 ns`, so HLS timing alone does not meet 300 MHz for that force-mode project. The combined implemented Vivado build remains the stronger physical timing evidence: routed `clk_pl_0 = 300.030 MHz`, WNS `-0.171 ns`, inside the user-allowed `-0.5 ns` tolerance.

### Patch-parallel Conv/EventConv latency probe

- Purpose:
  - Reduce the front-end Conv/EventConv latency that dominated the previous force-mode Search and Track worst-case paths.
  - Keep the existing full learned runtime-ROM candidate conditions: on-chip parameter ROM, on-chip nonlinear ROM, Search dispatcher prefetch, shared ATTN/MLP unit, Head parallelism `8`, selective `gb.norm` LUTRAM, QKV/LN cache disabled, and 300 MHz HLS target.
- Code/profile delta:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp` adds `HGTXR_E2E_PATCH_PAR`.
  - `hgtxr_conv_patch_embedding` and `hgtxr_event_conv_patch_embedding` now use `kPatchPar` accumulator lanes for the patch MAC loop.
  - Default `HGTXR_E2E_PATCH_PAR=1` preserves old behavior for existing profiles.
  - New probe profiles set `HGTXR_E2E_PATCH_PAR=8`:
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_search_only`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_track_only`
- CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16`
  - Search output `[3, 3, 3, 3, 3, 11]`, runtime state `0`, expected `0`, `count=6`, `last=1`, failures `0`.
  - Track output `[217, 217, 217, 217, 217, 213]`, runtime state `1`, expected `1`, `count=6`, `last=1`.
  - Strict prefetch traces remained clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  - Result: `E2E AXIS vector comparison passed`.
- Combined CSynth:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - HLS target `3.333 ns`; estimated clock `2.797 ns` / `357.46 MHz`.
  - Top min/max latency `69,458 / 7,094,387 cycles`; interval `69,459 / 7,094,388 cycles`.
  - Top max interval at 300 MHz: `23.648 ms`, versus previous `nocache_dsppipe4` Search force interval `28.683 ms`.
  - Resource estimate: BRAM_18K `295/624 = 47%`, DSP `706/1728 = 40%`, FF `105819/460800 = 22%`, LUT `228799/230400 = 99%`, URAM `92/96 = 95%`.
  - Resource caveat: HLS LUT margin is only about `1601 LUTs`, so this is a tight HLS-fit probe and needs Vivado implementation before being treated as physically fit.
- Mode-specific force-mode CSynth:
  - Search command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_search_only`
  - Search report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search result: HLS target `3.333 ns`, estimated clock `2.797 ns` / `357.46 MHz`, latency `7,105,728 cycles`, interval `7,105,729 cycles`, `23.686 ms` at 300 MHz.
  - Track command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_track_only`
  - Track report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track result: HLS target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, min-bound latency/interval `698,166/698,167 cycles`, max-bound latency/interval `2,467,638/2,467,639 cycles`.
  - At 300 MHz, Track min-bound interval is `2.327 ms`; Track max-bound interval is `8.225 ms`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Max invocation rate @ 300 MHz | Target status |
|---|---:|---:|---:|---|
| Previous `nocache_dsppipe4` Search-only | `8,604,865` | `28.683 ms` | `34.9/s` | Fail vs `4 ms` |
| Patchpar8 Search-only | `7,105,729` | `23.686 ms` | `42.2/s` | Fail vs `4 ms` |
| Previous `nocache_dsppipe4` Track min-bound | `698,167` | `2.327 ms` | `429.7/s` | Fail vs `1 ms` |
| Patchpar8 Track min-bound | `698,167` | `2.327 ms` | `429.7/s` | Fail vs `1 ms` |
| Previous `nocache_dsppipe4` Track max-bound | `4,003,639` | `13.345 ms` | `74.9/s` | Fail vs `1 ms` |
| Patchpar8 Track max-bound | `2,467,639` | `8.225 ms` | `121.6/s` | Fail vs `1 ms` |

| Hybrid assumption | Expected interval cycles | Expected interval @ 300 MHz | Expected rate | Worst-case interval |
|---|---:|---:|---:|---:|
| Search `10%`, Track `90%`, Track min-bound | `1,338,923` | `4.463 ms` | `224.1/s` | Search `23.686 ms` |
| Search `10%`, Track `90%`, Track max-bound | `2,931,448` | `9.771 ms` | `102.3/s` | Search `23.686 ms` |

- Path latency breakdown:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `1,794,052`, `global_buffer_load` `12,292`, `controller_run` `5,168,162`, `mlp_head` `24,970`, observable mix `89,849`.
  - Track-only min-bound: `axis_read_frame` `4,098`, `event_conv_patch_embedding` `12,484`, `global_buffer_load` `3,076`, `controller_run` `628,712`, `mlp_head` `6,538`, observable mix `43,241`.
  - Track-only max-bound: dominated by `event_conv_patch_embedding` max `1,781,956 cycles`.
- Decision:
  - Patch parallelism is aligned with the objective because it accelerates the learned Conv/EventConv path while preserving CSim outputs and runtime prefetch behavior.
  - It materially reduces Search latency by about `4.997 ms` and Track max-bound latency by about `5.120 ms` at 300 MHz.
  - It still does not meet the Search `4 ms` or Track `1 ms` latency targets.
  - The next latency work should target `controller_run` and EventConv memory/loop structure. Search remains dominated by `controller_run` (`5.168M cycles`) plus Conv (`1.794M cycles`); Track min-bound remains dominated by `controller_run` (`628,712 cycles`).
  - Track-only force-mode HLS estimated clock is still `3.473 ns` / `287.94 MHz`, so a physical Vivado implementation is needed before accepting the patchpar8 candidate as a 300 MHz routed candidate.

### Token-loop Patch Embedding Probe

- Purpose:
  - Reduce Track worst-case/EventConv pessimism from the previous grid loop structure.
  - The previous Conv/EventConv code iterated the full patch grid and guarded work with `if (token < active_tokens)`. For Track, `active_tokens=16` while the configured patch grid has `64` tokens, so HLS reported a large max-bound even though the intended Track path only uses the first 16 tokens.
- Code/profile delta:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp` adds `HGTXR_E2E_PATCH_TOKEN_LOOP`, default `0`.
  - When enabled, `hgtxr_conv_patch_embedding` and `hgtxr_event_conv_patch_embedding` loop directly over `token < active_tokens` and derive `gy/gx` from the token id.
  - Existing profiles keep the old grid-loop behavior by default.
  - New probe profiles set both `HGTXR_E2E_PATCH_PAR=8` and `HGTXR_E2E_PATCH_TOKEN_LOOP=1`:
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_search_only`
    - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_track_only`
- CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16`
  - Search output stayed `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`.
  - Track output stayed `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
  - Strict prefetch traces remained clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  - Result: `E2E AXIS vector comparison passed`.
- Combined CSynth:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - HLS target `3.333 ns`; estimated clock `2.797 ns` / `357.46 MHz`.
  - Combined top interval min/max `514,707 / 6,295,476 cycles`, `1.716 ms / 20.983 ms` at 300 MHz.
  - Resource estimate: BRAM_18K `295/624 = 47%`, DSP `706/1728 = 40%`, FF `105817/460800 = 22%`, LUT `228745/230400 = 99%`, URAM `92/96 = 95%`.
  - Combined-report caveat: this dynamic branch envelope is useful as a sanity check, but mode-specific latency reporting should use the force-mode Search-only and Track-only reports below.
- Mode-specific force-mode CSynth:
  - Search command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_search_only`
  - Search report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search result: target `3.333 ns`, estimated clock `2.797 ns` / `357.46 MHz`, interval `7,105,729 cycles`, `23.686 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `1,794,052`, `global_buffer_load` `12,292`, `controller_run` `5,168,162`, `mlp_head` `24,970`, observable mix `89,849`.
  - Track command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_track_only`
  - Track report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track result: target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, latency/interval `1,143,414 / 1,143,415 cycles`, `3.811 ms` at 300 MHz.
  - Track path: `axis_read_frame` `4,098`, `event_conv_patch_embedding` `457,732`, `global_buffer_load` `3,076`, `controller_run` `628,712`, `mlp_head` `6,538`, observable mix `43,241`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Max invocation rate @ 300 MHz | Target status |
|---|---:|---:|---:|---|
| Patchpar8 Search-only | `7,105,729` | `23.686 ms` | `42.2/s` | Fail vs `4 ms` |
| Token-loop patchpar8 Search-only | `7,105,729` | `23.686 ms` | `42.2/s` | Fail vs `4 ms` |
| Patchpar8 Track max-bound | `2,467,639` | `8.225 ms` | `121.6/s` | Fail vs `1 ms` |
| Token-loop patchpar8 Track | `1,143,415` | `3.811 ms` | `262.4/s` | Fail vs `1 ms` |

| Hybrid assumption | Expected interval cycles | Expected interval @ 300 MHz | Expected rate | Worst-case interval |
|---|---:|---:|---:|---:|
| Patchpar8 Search `10%`, Track `90%`, Track max-bound | `2,931,448` | `9.771 ms` | `102.3/s` | Search `23.686 ms` |
| Token-loop patchpar8 Search `10%`, Track `90%` | `1,739,646` | `5.799 ms` | `172.4/s` | Search `23.686 ms` |

- Decision:
  - Token-loop is aligned with the objective because it preserves full learned path outputs and runtime prefetch ordering while removing unnecessary Track-mode patch-grid traversal.
  - It reduces EventConv force-mode latency from `1,781,956` cycles to `457,732` cycles and Track interval from `8.225 ms` to `3.811 ms` at 300 MHz, about `2.16x` better than the previous patchpar8 Track max-bound.
  - Search latency is unchanged because Search already uses all `64` active tokens.
  - The design still misses the Search `4 ms` and Track `1 ms` targets. Remaining measured bottlenecks are Search `controller_run` `5.168M cycles`, Search Conv `1.794M cycles`, and Track `controller_run` `628,712 cycles`.
  - Token-loop still needs physical Vivado implementation before it can replace the previously routed `nocache_dsppipe4` candidate as the accepted 300 MHz physical baseline.

### Patchpar16 Token-loop DSE

- Purpose:
  - Test whether increasing learned Conv/EventConv patch parallelism from `8` to `16` can reduce the remaining Search Conv and Track EventConv latency while preserving the same full learned runtime path.
  - This is a latency DSE on top of token-loop patch embedding, not a functional shortcut.
- Profiles added:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16`
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_search_only`
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_track_only`
- CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16`
  - Passed with unchanged Search `[3, 3, 3, 3, 3, 11]` and Track `[217, 217, 217, 217, 217, 213]` outputs.
  - Strict prefetch traces stayed clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Mode-specific CSynth:
  - Search report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search result: estimated clock `2.797 ns` / `357.46 MHz`, interval `5,692,609 cycles`, `18.975 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `5,168,162`, `mlp_head` `24,970`, observable mix `89,849`.
  - Track report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track result: estimated clock `3.473 ns` / `287.94 MHz`, interval `796,290 cycles`, `2.654 ms` at 300 MHz.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `628,712`, `mlp_head` `6,538`, observable mix `43,241`.
- Combined CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Combined interval min/max `167,583 / 5,705,664 cycles`, `0.559 / 19.019 ms` at 300 MHz.
  - Resource estimate: BRAM_18K `309/624 = 49%`, DSP `720/1728 = 41%`, FF `106881/460800 = 23%`, LUT `238345/230400 = 103%`, URAM `92/96 = 95%`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Change vs patchpar8 token-loop | ZCU104-fit status |
|---|---:|---:|---:|---|
| Patchpar8 token-loop Search-only | `7,105,729` | `23.686 ms` | baseline | Force-mode fit estimate |
| Patchpar16 token-loop Search-only | `5,692,609` | `18.975 ms` | `19.9%` faster | Force-mode fit estimate |
| Patchpar8 token-loop Track-only | `1,143,415` | `3.811 ms` | baseline | Force-mode fit estimate |
| Patchpar16 token-loop Track-only | `796,290` | `2.654 ms` | `30.4%` faster | Force-mode fit estimate |
| Patchpar8 token-loop combined | max `6,295,476` | `20.985 ms` | baseline | LUT `99%`, tight fit estimate |
| Patchpar16 token-loop combined | max `5,705,664` | `19.019 ms` | `9.4%` faster top max | LUT `103%`, reject |

| Hybrid assumption | Expected interval cycles | Expected interval @ 300 MHz | Expected rate | Worst-case interval |
|---|---:|---:|---:|---:|
| Patchpar8 token-loop Search `10%`, Track `90%` | `1,739,646` | `5.799 ms` | `172.4/s` | Search `23.686 ms` |
| Patchpar16 token-loop Search `10%`, Track `90%` | `1,285,922` | `4.286 ms` | `233.3/s` | Search `18.975 ms` |

- Legal parallelism check:
  - A temporary `PATCH_PAR=12` compile probe was tried as a possible middle point, but CSim compilation failed at `static_assert(kPatchElems % kPatchPar == 0)`.
  - Current legal probe points between the tested values are therefore not available under the existing `kPatchElems` divisibility rule.
- Decision:
  - `PATCH_PAR=16` is a useful negative DSE: it proves more patch parallelism reduces both Search Conv and Track EventConv, but the combined runtime top exceeds ZCU104 LUT capacity at HLS estimate level.
  - Do not promote patchpar16 to Vivado implementation in this form.
  - Current fit-capable latency candidate remains `patchpar8_tokenloop`. Next work should target `controller_run` or reduce combined controller/Conv LUT before reintroducing higher patch parallelism.

### Patchpar16 No-observe Norm-BRAM Fit DSE

- Purpose:
  - Recover enough LUT from the rejected patchpar16 token-loop profile to keep the lower Search/Track latency while fitting the ZCU104 HLS resource envelope.
  - Changes versus rejected patchpar16 token-loop: disable the debug observable datapath, move norm storage back to BRAM instead of LUTRAM/URAM, and reduce local Head parallelism from `8` to `4`.
- Profile:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode variants use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Passed with Search runtime state `0`, Track runtime state `1`, and clean strict prefetch traces.
  - Because `HGTXR_E2E_OBSERVE_DATAPATH=0`, the debug mixed output word is intentionally removed from the observable result stream. This profile is a resource/latency DSE and still needs strict golden refresh before final functional acceptance.
- AXIS payload structural audit:
  - Command: `python3 hardware/tools/check_hls_axis_payload_contract.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16 --json-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_2026_06_30.json --markdown-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_2026_06_30.md`
  - Result status: `structural-pass-missing-rtl-tv`.
  - The generated synthesis RTL structurally drives AXIS `TDATA` from `out_state` through 8-bit zero-extension sources and has no constant-zero `TDATA` assignment. This separates the current no-observe candidate from the rejected `tail2_dsppipe4` structure, where `axis_out_TDATA` was constant zero.
  - C/RTL TV payload files are still missing for this profile, so this is not a SW/HW same-input same-output proof.
- Structural objective-contract audit:
  - Tool update: `hardware/tools/write_prefetchall4_300_contract_audit.py` now accepts `--profile`, while preserving the previous default active-profile behavior.
  - Command: `python3 hardware/tools/write_prefetchall4_300_contract_audit.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16 --json-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_contract_audit_2026_06_30.json --markdown-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_contract_audit_2026_06_30.md`
  - Result status: `pass`, checks `16/16`.
  - Covered requirements: runtime scheduler, Search/Track mode constants, frame/event Conv generated modules, Conv/Head/Track Transformer on-chip parameter ROM range, omitted external weight AXI master, nonlinear ROM source paths, Search dispatcher prefetch-all4 and immediate-start trace gate, generated LayerNorm/QKV/Attention/OutputProjection/shared MLP/Head modules, Search/Track CSim vector checks, and 300 MHz ZCU104 csynth report availability.
- Combined CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Target `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`.
  - Combined interval min/max `124,486 / 5,617,495 cycles`, `0.415 / 18.725 ms` at 300 MHz.
  - Resource estimate: BRAM_18K `341/624 = 54%`, DSP `719/1728 = 41%`, FF `105285/460800 = 22%`, LUT `222252/230400 = 96%`, URAM `92/96 = 95%`.
  - Main resource delta versus rejected patchpar16: LUT drops from `238345` (`103%`) to `222252` (`96%`), while BRAM rises from `309` to `341`.
- Mode-specific force-mode CSynth:
  - Search report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search result: estimated clock `2.777 ns` / `360.10 MHz`, interval max `5,604,440 cycles`, `18.681 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `5,169,698`, `mlp_head` `25,114`, plus top-level overhead.
  - Track report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track result: estimated clock `3.473 ns` / `287.94 MHz`, interval `753,383 cycles`, `2.511 ms` at 300 MHz or `2.616 ms` at estimated Fmax.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `628,902`, `mlp_head` `6,682`, plus top-level overhead.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Max invocation rate @ 300 MHz | Timing status |
|---|---:|---:|---:|---|
| No-observe norm-BRAM Search-only | `5,604,440` | `18.681 ms` | `53.5/s` | HLS estimated clock passes 300 MHz |
| No-observe norm-BRAM Track-only | `753,383` | `2.511 ms` | `398.2/s` | HLS estimated clock misses 300 MHz (`287.94 MHz`) |
| No-observe norm-BRAM combined min | `124,486` | `0.415 ms` | `2409.9/s` | Combined HLS estimated clock passes 300 MHz |
| No-observe norm-BRAM combined max | `5,617,495` | `18.725 ms` | `53.4/s` | Combined HLS estimated clock passes 300 MHz |

| Hybrid assumption | Expected interval cycles | Expected interval @ 300 MHz | Expected rate | Worst-case interval |
|---|---:|---:|---:|---:|
| Force-mode Search `10%`, Track `90%` | `1,238,489` | `4.128 ms` | `242.2/s` | Search `18.681 ms` |
| Combined runtime envelope Search `10%`, Track `90%` | `673,787` | `2.246 ms` | `445.2/s` | Combined max `18.725 ms` |

- Decision:
  - This is now the best HLS-fit latency/resource candidate among the full learned runtime-ROM profiles tested so far: it preserves patchpar16 token-loop latency direction while fitting LUT at `96%`.
  - It still misses the final latency targets: Search `18.681 ms > 4 ms`; force-mode Track `2.511 ms > 1 ms`.
  - It is not yet a final functional acceptance result. Required next steps are strict golden refresh for the no-observe output contract and mode-specific board or SAIF-backed power/latency measurement.

### Patchpar16 No-observe Norm-BRAM Vivado Implementation

- Purpose:
  - Close the physical implementation gap for the current best HLS-fit latency/resource candidate.
  - Keep the requested `300 MHz` PL clock and the full learned runtime-ROM path with shared ATTN/MLP, Search prefetch-all4, on-chip parameter/nonlinear ROMs, `PATCH_PAR=16`, token-loop Conv/EventConv, no debug observable datapath, Head parallelism `4`, and norm storage back on BRAM.
- Command:
  - `sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
- Package/IP evidence:
  - HLS package succeeded for `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip`.
  - `component.xml` and `export.zip` were generated.
- Vivado implementation:
  - Project/profile: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay`.
  - PL clock target: `300.0 MHz`.
  - Strategy: `Performance_ExplorePostRoutePhysOpt`.
  - `synth_1`, `impl_1`, post-route `phys_opt_design`, and `write_bitstream` completed successfully.
  - Route status: routable nets `106,191`, fully routed nets `106,191`, routing errors `0`.
  - Final implemented timing: `WNS 0.000 ns`, `TNS 0.000 ns`, `WHS 0.006 ns`, `THS 0.000 ns`; Vivado reports `All user specified timing constraints are met.`
  - Exported overlay: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/`.
  - Exported bitstream: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.bit`, size about `19 MB`.
  - Exported handoff: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.hwh`, size about `482 KB`.
- Implemented utilization:
  - CLB LUTs `48,909/230,400 = 21.23%`.
  - LUT as Logic `47,375/230,400 = 20.56%`.
  - LUT as Memory `1,534/101,760 = 1.51%`.
  - CLB Registers `44,104/460,800 = 9.57%`.
  - Block RAM Tile `169.5/312 = 54.33%`.
  - URAM `92/96 = 95.83%`.
  - DSP48E2 `690/1,728 = 39.93%`.
- Implemented vectorless power:
  - Total on-chip `6.412 W`.
  - Dynamic `5.688 W`.
  - Device static `0.724 W`.
  - PS static `0.102 W`; PL static `0.622 W`.
  - URAM dynamic bucket `0.232 W`; DSP dynamic bucket `0.671 W`.
  - Caveat: no SAIF/VCD mode-specific activity was supplied, and Vivado emitted the high-fanout reset activity warning. Treat this as vectorless implementation power, not board-measured Search/Track power.
- Current judgment:
  - This candidate now proves physical ZCU104 fit, route completion, bitgen, and official clean timing at the requested `300 MHz`.
  - It is the strongest physical fit/timing result so far for the full learned runtime-ROM objective path.
  - The objective remains open because Search and Track latency targets still fail at HLS force-mode estimates: Search `18.680 ms > 4 ms`, Track `2.511 ms @ 300 MHz` / `2.616 ms @ estimated Fmax > 1 ms`.
  - Functional evidence is still incomplete for final signoff: CSim validates runtime state and strict prefetch trace, but disabling `HGTXR_E2E_OBSERVE_DATAPATH` intentionally changed the observable output contract, so strict golden refresh and RTL/board same-input/same-output evidence remain pending.
  - The structural contract audit now directly covers the active no-observe profile and passes `16/16`. The AXIS payload structural audit also reduces one functional risk: the active no-observe synthesis RTL is not a constant-zero output shell. The remaining functional gap is producing and comparing C/RTL or board output payloads for the refreshed no-observe contract.
  - C/RTL cosim was rerun for the same active no-observe profile on 2026-06-30. The C testbench passed and generated C reference TV with Search output payloads `[3, 3, 3, 3, 3, 3]`, Track output payloads `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, and strict prefetch trace violations `0`.
  - The standard HLS Verilog cosim still failed before RTL transaction completion: `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` with latency/interval `NA`, and `xsim` exited with `ERROR: unexpected exception when evaluating tcl command`.
  - Direct `xsimk` probing of the generated snapshot entered elaboration and started RTL simulation progress (`0 / 2` transactions at sim time `109000`), but timed out at `180 s` before completing any transaction. Generated RTL TV files are present but only contain the runtime marker, so payload comparison currently fails.
  - Latest signoff artifacts:
    - `hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_after_xsimk_2026_06_30.{json,md}`: status `fail-active-payload-tv`, because C payloads exist but RTL payloads are empty/incomplete.
    - `hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_xsimk_direct_probe_after_cosim_2026_06_30.{json,md}`: status `direct-kernel-timeout`, proving snapshot entry and progress banner but not output equivalence.

### Patchpar16 No-observe Norm-BRAM Dense64 DSE

- Purpose:
  - Test whether increasing the full Transformer dense lane parallelism from `32` to `64` can reduce the remaining `controller_run` bottleneck without changing the full learned runtime-ROM objective path.
  - Keep the current best physical candidate's other constraints: Search dispatcher prefetch-all4, on-chip parameter/nonlinear ROMs, `PATCH_PAR=16`, token-loop Conv/EventConv, observable datapath disabled, Head parallelism `4`, norm storage on BRAM, shared runtime units, and `300 MHz` HLS target.
- Profiles added:
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`
  - `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: `CSim done with 0 errors`.
  - Search output payloads stayed `[3, 3, 3, 3, 3, 3]`, runtime state `0`, count `6`, TLAST correct.
  - Track output payloads stayed `[217, 217, 217, 217, 217, 217]`, runtime state `1`, count `6`, TLAST correct.
  - Strict prefetch traces stayed clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Search-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`, interval max `5,833,024 cycles`, `19.443 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `5,398,282`, `mlp_head` `25,114`, plus top-level overhead.
  - Controller subpath: attention unit `208,086 cycles`, MLP unit `1,134,543 cycles`, block-loop iteration latency `1,342,634 cycles`, trip count `4`.
  - Resources: BRAM_18K `339/624 = 54%`, DSP `955/1728 = 55%`, FF `95020/460800 = 20%`, LUT `225097/230400 = 97%`, URAM `92/96 = 95%`.
- Track-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, interval `781,851 cycles`, `2.606 ms` at 300 MHz or `2.715 ms` at estimated Fmax.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `657,370`, `mlp_head` `6,682`, plus top-level overhead.
  - Controller subpath: attention unit `38,691 cycles`, MLP unit `283,060 cycles`, block-loop iteration latency `321,755 cycles`, trip count `2`.
  - Resources: BRAM_18K `337/624 = 54%`, DSP `950/1728 = 54%`, FF `98703/460800 = 21%`, LUT `221339/230400 = 96%`, URAM `71/96 = 73%`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Change vs active no-observe | Decision |
|---|---:|---:|---:|---|
| Active no-observe Search-only | `5,604,440` | `18.681 ms` | baseline | Keep |
| Dense64 no-observe Search-only | `5,833,024` | `19.443 ms` | `+4.1%` slower | Reject |
| Active no-observe Track-only | `753,383` | `2.511 ms` | baseline | Keep |
| Dense64 no-observe Track-only | `781,851` | `2.606 ms` | `+3.8%` slower | Reject |

- Decision:
  - Dense64 is a useful negative DSE, not a forward candidate.
  - Although the Search attention unit improves from the active profile's `288,732 cycles` to `208,086 cycles`, the MLP unit worsens from `996,751 cycles` to `1,134,543 cycles`, and the controller remains the dominant bottleneck.
  - Track also worsens: controller latency rises from `628,902` to `657,370 cycles`.
  - The next latency work should not increase global `kDensePar` blindly. It should target the MLP dataflow/weight-cache structure specifically, or introduce a mode-aware MLP scheduling change that reduces the `W1/W2` loops without increasing mux/BRAM pressure.

### Patchpar16 No-observe Norm-BRAM MLP Token-parallel DSE

- Purpose:
  - Test whether the MLP W1/W2 weight-vector reuse can be improved by processing two tokens per MLP group while preserving the full learned runtime-ROM path.
  - Keep the active physical candidate constraints unchanged otherwise: on-chip parameter/nonlinear ROMs, Search dispatcher prefetch-all4, shared runtime units, `PATCH_PAR=16`, token-loop Conv/EventConv, observable datapath disabled, Head parallelism `4`, norm storage on BRAM, and `300 MHz` HLS target.
- Implementation:
  - Added `HGTXR_E2E_MLP_TOKEN_PAR`, default `1`.
  - Added a guarded `HGTXR_E2E_MLP_TOKEN_PAR > 1` branch in `hgtxr_e2e_mlp_unit`.
  - The branch reuses one W1/W2 weight vector across `kMlpTokenPar` token lanes and leaves the default MLP implementation unchanged for all existing profiles.
  - Added `mlptok2` combined, Search-only, and Track-only profiles with `HGTXR_E2E_MLP_TOKEN_PAR=2`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: `CSim done with 0 errors`.
  - Search output payloads stayed `[3, 3, 3, 3, 3, 3]`, runtime state `0`, count `6`, TLAST correct.
  - Track output payloads stayed `[217, 217, 217, 217, 217, 217]`, runtime state `1`, count `6`, TLAST correct.
  - Strict prefetch traces stayed clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Search-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`, interval max `6,153,432 cycles`, `20.511 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `5,718,690`, `mlp_head` `25,114`, plus top-level overhead.
  - MLP unit latency worsened from the active profile's `996,751 cycles` to `1,133,999 cycles`.
  - Resources: BRAM_18K `355/624 = 56%`, DSP `571/1728 = 33%`, FF `63073/460800 = 13%`, LUT `167589/230400 = 72%`, URAM `92/96 = 95%`.
- Track-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, interval `822,007 cycles`, `2.740 ms` at 300 MHz or `2.855 ms` at estimated Fmax.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `697,526`, `mlp_head` `6,682`, plus top-level overhead.
  - MLP unit latency worsened from the active profile's `248,612 cycles` to `282,924 cycles`.
  - Resources: BRAM_18K `353/624 = 56%`, DSP `694/1728 = 40%`, FF `74814/460800 = 16%`, LUT `163318/230400 = 70%`, URAM `71/96 = 73%`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Change vs active no-observe | Decision |
|---|---:|---:|---:|---|
| Active no-observe Search-only | `5,604,440` | `18.681 ms` | baseline | Keep |
| MLP token-parallel 2 Search-only | `6,153,432` | `20.511 ms` | `+9.8%` slower | Reject for latency |
| Active no-observe Track-only | `753,383` | `2.511 ms` | baseline | Keep |
| MLP token-parallel 2 Track-only | `822,007` | `2.740 ms` | `+9.1%` slower | Reject for latency |

- Hybrid 10% Search / 90% Track force-mode estimate:
  - Active no-observe baseline: `1,238,489 cycles`, `4.128 ms`.
  - MLP token-parallel 2: about `1,355,150 cycles`, `4.517 ms`.
- Decision:
  - `mlptok2` is an area-saving but latency-negative DSE.
  - HLS automatically inferred token-dimension cyclic factor `2` partitioning on `gb.norm` and `gb.hidden`, and top-level DSP/LUT dropped substantially.
  - The latency still worsened because the grouped MLP iteration latency rose from `14,916` to `34,121 cycles`; the extra muxing/partition pressure outweighed weight-vector reuse.
  - Do not adopt `mlptok2` as the latency baseline. The active physical candidate remains `patchpar16_tokenloop_noobs_normbram_headpar4`.

### Patchpar16 No-observe Norm-BRAM MLP Fused-W2 DSE

- Purpose:
  - Test whether the dominant MLP W2 read/reduction bottleneck can be reduced without token-dimension buffer replication.
  - Fuse W1 hidden generation and W2 output accumulation inside one token pass, so the W2 path consumes the freshly computed hidden lane values rather than rereading the full hidden buffer in a separate pass.
  - Preserve the current active physical candidate constraints otherwise: on-chip parameter/nonlinear ROMs, Search dispatcher prefetch-all4, shared runtime units, `PATCH_PAR=16`, token-loop Conv/EventConv, observable datapath disabled, Head parallelism `4`, norm storage on BRAM, and `300 MHz` HLS target.
- Implementation:
  - Added `HGTXR_E2E_MLP_FUSED_W2`, default `0`.
  - Added a guarded fused MLP branch in `hgtxr_e2e_mlp_unit`.
  - The fused branch keeps a complete-partitioned `out_acc[kEmbed]`, computes hidden values by `kDensePar` chunks, updates the W2 output accumulator immediately, and still writes `gb.hidden[t][h]` for compatibility/observability.
  - Added `mlpfuse` combined, Search-only, and Track-only profiles with `HGTXR_E2E_MLP_FUSED_W2=1`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: `CSim done with 0 errors`.
  - Search output payloads stayed `[3, 3, 3, 3, 3, 3]`, runtime state `0`, count `6`, TLAST correct.
  - Track output payloads stayed `[217, 217, 217, 217, 217, 217]`, runtime state `1`, count `6`, TLAST correct.
  - Strict prefetch traces stayed clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Search-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated Fmax `232.10 MHz`, interval max `4,488,024 cycles`, `14.960 ms` at 300 MHz or `19.337 ms` at estimated Fmax.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `4,053,282`, `mlp_head` `25,114`, plus top-level overhead.
  - Controller subpath: attention unit `288,732 cycles`, MLP unit `717,647 cycles`, block-loop iteration latency about `1,006,384 cycles`, trip count `4`.
  - Resources: BRAM_18K `227/624 = 36%`, DSP `507/1728 = 29%`, FF `75299/460800 = 16%`, LUT `168027/230400 = 72%`, URAM `92/96 = 95%`.
- Track-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated Fmax `232.10 MHz`, interval `613,831 cycles`, `2.046 ms` at 300 MHz or `2.645 ms` at estimated Fmax.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `489,350`, `mlp_head` `6,682`, plus top-level overhead.
  - Controller subpath: attention unit `58,905 cycles`, MLP unit `178,836 cycles`, block-loop iteration latency `237,745 cycles`, trip count `2`.
  - Resources: BRAM_18K `225/624 = 36%`, DSP `630/1728 = 36%`, FF `87201/460800 = 18%`, LUT `163787/230400 = 71%`, URAM `71/96 = 73%`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Interval @ estimated Fmax | Change vs active no-observe | Decision |
|---|---:|---:|---:|---:|---|
| Active no-observe Search-only | `5,604,440` | `18.681 ms` | n/a | baseline | Keep as physical baseline |
| MLP fused-W2 Search-only | `4,488,024` | `14.960 ms` | `19.337 ms` | `19.9%` fewer cycles | Useful DSE, not promotable at current Fmax |
| Active no-observe Track-only | `753,383` | `2.511 ms` | n/a | baseline | Keep as physical baseline |
| MLP fused-W2 Track-only | `613,831` | `2.046 ms` | `2.645 ms` | `18.5%` fewer cycles | Useful DSE, not promotable at current Fmax |

- Hybrid 10% Search / 90% Track force-mode estimate:
  - Active no-observe baseline: `1,238,489 cycles`, `4.128 ms`.
  - MLP fused-W2: about `1,001,250 cycles`, `3.338 ms` at 300 MHz, or `4.314 ms` at estimated Fmax.
- Decision:
  - `mlpfuse` is the first MLP-focused DSE here that improves both Search and Track cycle latency.
  - It reduces Search MLP unit latency from `996,751` to `717,647 cycles` and Track MLP unit latency from `248,612` to `178,836 cycles`.
  - It is not a promotion candidate yet because HLS estimated Fmax drops to `232.10 MHz`, far below the requested `300 MHz`.
  - The likely timing cost is the complete-partitioned `out_acc[kEmbed]` accumulator and associated W2 accumulation mux/fanout.
  - Next latency work should preserve this fused W2 scheduling direction but reduce timing pressure, for example by tiling `out_acc` by output-channel group instead of complete `kEmbed` partitioning, or by staging the W2 accumulation tree across smaller output chunks.

### Patchpar16 No-observe Norm-BRAM MLP Fused-W2 Banked-Accumulator DSE

- Purpose:
  - Preserve the useful fused-W2 cycle reduction while reducing the timing and fanout pressure from complete partitioning of `out_acc[kEmbed]`.
  - Keep the same full learned runtime-ROM objective path as the active physical baseline: on-chip parameter/nonlinear ROMs, Search dispatcher prefetch-all4, shared runtime units, `PATCH_PAR=16`, token-loop Conv/EventConv, observable datapath disabled, Head parallelism `4`, norm storage on BRAM, and `300 MHz` HLS target.
- Implementation:
  - Added `HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC`, default `0`.
  - When enabled with `HGTXR_E2E_MLP_FUSED_W2=1`, `out_acc` uses cyclic partitioning by `kDensePar` instead of complete partitioning.
  - Added `mlpfusebank` combined, Search-only, and Track-only profiles with `HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC=1`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: `CSim done with 0 errors`.
  - Search output payloads stayed `[3, 3, 3, 3, 3, 3]`, runtime state `0`, count `6`, TLAST correct.
  - Track output payloads stayed `[217, 217, 217, 217, 217, 217]`, runtime state `1`, count `6`, TLAST correct.
  - Strict prefetch traces stayed clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Search-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`, interval max `4,490,328 cycles`, `14.968 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `4,055,586`, `mlp_head` `25,114`, plus top-level overhead.
  - Controller subpath: attention unit `288,732 cycles`, MLP unit `718,223 cycles`, block-loop iteration latency `1,006,960 cycles`, trip count `4`.
  - Resources: BRAM_18K `227/624 = 36%`, DSP `507/1728 = 29%`, FF `60218/460800 = 13%`, LUT `156534/230400 = 67%`, URAM `92/96 = 95%`.
- Track-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, interval `614,119 cycles`, `2.047 ms` at 300 MHz or `2.133 ms` at estimated Fmax.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `489,638`, `mlp_head` `6,682`, plus top-level overhead.
  - Controller subpath: attention unit `58,905 cycles`, MLP unit `178,980 cycles`, block-loop iteration latency `237,889 cycles`, trip count `2`.
  - Resources: BRAM_18K `225/624 = 36%`, DSP `630/1728 = 36%`, FF `72120/460800 = 15%`, LUT `152295/230400 = 66%`, URAM `71/96 = 73%`.

| Profile/mode | Interval cycles | Interval @ 300 MHz | Interval @ estimated Fmax | Change vs active no-observe | Decision |
|---|---:|---:|---:|---:|---|
| Active no-observe Search-only | `5,604,440` | `18.681 ms` | n/a | baseline | Keep as physical baseline |
| MLP fused-W2 banked Search-only | `4,490,328` | `14.968 ms` | `12.470 ms` at HLS estimated Search Fmax | `19.9%` fewer cycles | Best HLS cycle/Fmax Search candidate so far |
| Active no-observe Track-only | `753,383` | `2.511 ms` | n/a | baseline | Keep as physical baseline |
| MLP fused-W2 banked Track-only | `614,119` | `2.047 ms` | `2.133 ms` | `18.5%` fewer cycles | Useful DSE, Track Fmax still short of 300 MHz |

- Hybrid 10% Search / 90% Track force-mode estimate:
  - Active no-observe baseline: `1,238,489 cycles`, `4.128 ms`.
  - MLP fused-W2 banked: about `1,001,740 cycles`, `3.339 ms` at 300 MHz, or `3.479 ms` if limited by the Track estimated Fmax of `287.94 MHz`.
- Decision:
  - `mlpfusebank` fixes most of the timing damage from complete-partition `mlpfuse` for Search: estimated Fmax improves from `232.10 MHz` to `360.10 MHz`, while Search interval changes only from `4,488,024` to `4,490,328 cycles`.
  - It also reduces Search LUT/FF versus `mlpfuse` (`168,027 -> 156,534 LUT`, `75,299 -> 60,218 FF`) and preserves the BRAM reduction.
  - Track cycle latency remains improved versus the active physical baseline, but Track-only HLS estimated clock remains `3.473 ns` / `287.94 MHz`, so this is not yet a 300 MHz signoff candidate.
  - Current physical baseline remains the routed no-observe norm-BRAM profile. Current best HLS latency direction is `mlpfusebank`, with the next concrete bottleneck being Track force-mode timing closure around the controller/MLP path.

### Patchpar16 No-observe Norm-BRAM MLP Fused-W2 Banked-Accumulator Vivado Implementation

- Purpose:
  - Physically implement the best current HLS latency direction, `mlpfusebank`, in the AXIS/DMA overlay at the requested `300 MHz` PL clock.
  - Check whether the HLS Track-only estimated clock miss prevents ZCU104 physical fit.
- Command:
  - `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
- HLS IP package:
  - `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`
- Implemented reports:
  - Timing: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_timing_summary_implemented.rpt`
  - Route: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_route_status_implemented.rpt`
  - Utilization: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_utilization_implemented.rpt`
  - Power: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_power_implemented.rpt`
- Bitstream and handoff artifacts:
  - `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.bit`
  - `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.hwh`
  - Copied to `hardware/pynq/hgtxr/` with the same artifact stem.
- Timing:
  - Clock: `clk_pl_0`, period `3.333 ns`, frequency `300.030 MHz`.
  - Final implemented setup: `WNS = -0.064 ns`, `TNS = -39.952 ns`, failing setup endpoints `1309 / 186815`.
  - Final implemented hold: `WHS = 0.004 ns`, `THS = 0.000 ns`, hold failing endpoints `0`.
  - Vivado nominal timing status: not met.
  - User-relaxed WNS status: pass under the allowed `-0.5 ns` setup slack tolerance.
- Route:
  - Logical nets `697,204`; routable nets `108,842`; fully routed nets `108,842`; routing errors `0`.
- Implemented resources:
  - CLB LUT `49,022 / 230,400 = 21.28%`.
  - CLB registers `45,244 / 460,800 = 9.82%`.
  - Block RAM Tile `121.5 / 312 = 38.94%`.
  - URAM `92 / 96 = 95.83%`.
  - DSP `690 / 1,728 = 39.93%`.
- Implemented vectorless power:
  - Total on-chip `6.231 W`; dynamic `5.509 W`; device static `0.722 W`.
  - Static split: PS static `0.102 W`, PL static `0.620 W`.
  - Main dynamic hierarchy: `hgtxr_e2e_axis_top_0 = 2.581 W`, `psu = 2.674 W`, `axi_mem = 0.202 W`, `axi_dma_out = 0.024 W`, `axi_dma_in = 0.007 W`, `axi_ctrl = 0.021 W`.
  - Confidence: medium, vectorless internal activity; mode-specific SAIF or board power is still required for final Search/Track power.
- Decision:
  - `mlpfusebank` is now physically fit on ZCU104 and produces bitstream/handoff artifacts.
  - It is better than the active no-observe physical baseline for HLS latency: Search `18.681 ms -> 14.968 ms`, Track `2.511 ms -> 2.047 ms`, and hybrid 10/90 `4.128 ms -> 3.339 ms` at ideal 300 MHz.
  - It is not final-goal complete: Search still misses `4 ms`, Track still misses `1 ms`, Vivado nominal timing is slightly negative, and same-input/same-output RTL or board validation remains open.
  - Next DSE should keep fused-W2 banked accumulation, add explicit DSP/MLP pipeline staging around the reported MREG/PREG warnings, and reduce layer/operator work per Search/Track invocation rather than only pushing placement.

### Patchpar16 No-observe Norm-BRAM MLP Fused-W2 No-hidden-store DSE

- Purpose:
  - Check whether the fused-W2 branch still pays scheduling or memory cost for writing `gb.hidden[t][h]` after the fused W1/GELU/W2 accumulation already has the local `hidden_vec`.
  - Keep the same `mlpfusebank` objective envelope: full learned runtime-ROM path, Search dispatcher prefetch-all4, `PATCH_PAR=16`, token-loop Conv/EventConv, Head parallelism `4`, no observe datapath, norm storage on BRAM, and `300 MHz` HLS target.
- Implementation:
  - Added `HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE`, default `0`.
  - Added combined, Search-only, and Track-only `mlpfusebank_nohidden` profiles with `HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: `CSim done with 0 errors`.
  - Search output payloads stayed `[3, 3, 3, 3, 3, 3]`, runtime state `0`, count `6`, TLAST correct.
  - Track output payloads stayed `[217, 217, 217, 217, 217, 217]`, runtime state `1`, count `6`, TLAST correct.
  - Strict prefetch traces stayed clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- Search-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`, interval max `4,490,328 cycles`, `14.968 ms` at 300 MHz.
  - Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `4,055,586`, `mlp_head` `25,114`, plus top-level overhead.
  - Resources: BRAM_18K `227/624 = 36%`, DSP `507/1728 = 29%`, FF `60218/460800 = 13%`, LUT `156534/230400 = 67%`, URAM `92/96 = 95%`.
- Track-only force-mode CSynth:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Result: target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, interval `614,119 cycles`, `2.047 ms` at 300 MHz or `2.133 ms` at estimated Fmax.
  - Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `489,638`, `mlp_head` `6,682`, plus top-level overhead.
  - Resources: BRAM_18K `225/624 = 36%`, DSP `630/1728 = 36%`, FF `72120/460800 = 15%`, LUT `152295/230400 = 66%`, URAM `71/96 = 73%`.
- Decision:
  - `nohidden` is latency/resource neutral versus `mlpfusebank`.
  - Search and Track intervals are unchanged at `4,490,328` and `614,119` cycles, and top-level resources are unchanged.
  - Do not route this candidate. Keep `mlpfusebank` as the current physical baseline and focus the next DSE on reducing controller/MLP work, where Search spends `4,055,586` cycles and Track spends `489,638` cycles.

### HPar2 MLP Fused-W2 Banked-Accumulator Physical Implementation

- Purpose:
  - Reduce the remaining `mlpfusebank` Search/Track latency by increasing the useful fused-W2/attention parallelism while preserving the full learned runtime-ROM objective path.
  - Keep the same objective envelope: full learned runtime-ROM path, Search dispatcher prefetch-all4, on-chip parameter/nonlinear ROMs, shared runtime ATTN/MLP units, `PATCH_PAR=16`, token-loop Conv/EventConv, Head parallelism `4`, no observe datapath, norm storage on BRAM, and `300 MHz` PL clock.
- HLS force-mode latency:
  - Search-only interval `3,900,504 cycles`, `13.002 ms` at 300 MHz.
  - Track-only interval `540,391 cycles`, `1.801 ms` at 300 MHz.
  - Hybrid 10% Search / 90% Track estimate: `876,402 cycles`, `2.921 ms` at 300 MHz.
  - Compared with `mlpfusebank`, Search improves from `14.968 ms` to `13.002 ms`, Track from `2.047 ms` to `1.801 ms`, and hybrid from `3.339 ms` to `2.921 ms`.
- Combined HLS package:
  - IP: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`
  - Top interval max `3,913,559 cycles`, `13.045 ms` at 300 MHz.
  - HLS top resources: BRAM_18K `235/624 = 37%`, DSP `751/1728 = 43%`, FF `107,770/460,800 = 23%`, LUT `225,293/230,400 = 97%`, URAM `92/96 = 95%`.
- Vivado implementation:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Bitstream generation completed successfully, but the runner exited after bitgen because the generated hierarchical utilization report filename exceeded the filesystem filename limit. The bit/hwh artifacts were then copied manually to the overlay and PYNQ output directories.
  - Runner fix: `hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl` now uses a short report prefix when the full artifact-derived report filename would be too long.
- Implemented artifacts:
  - `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.bit`
  - `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.hwh`
  - Same `.bit/.hwh` stems are copied under `hardware/pynq/hgtxr/`.
- Implemented timing:
  - Clock: `clk_pl_0`, period `3.333 ns`, frequency `300.030 MHz`.
  - Post-route physopt timing: `WNS = -0.094 ns`, `TNS = -107.248 ns`, setup failing endpoints `2197 / 192241`.
  - Hold timing: `WHS = 0.007 ns`, `THS = 0.000 ns`, hold failing endpoints `0`.
  - Vivado nominal timing status: not met.
  - User-relaxed WNS status: pass under the allowed `-0.5 ns` setup slack tolerance.
- Route:
  - Logical nets `725,542`; routable nets `110,620`; fully routed nets `110,620`; routing errors `0`.
- Implemented resources:
  - CLB LUT `49,867 / 230,400 = 21.64%`.
  - CLB registers `45,300 / 460,800 = 9.83%`.
  - Block RAM Tile `121.5 / 312 = 38.94%`.
  - URAM `92 / 96 = 95.83%`.
  - DSP `722 / 1,728 = 41.78%`.
  - Resource preservation check: implemented DSP/URAM usage remains high (`722 DSP`, `92 URAM`), so this is not a pruned-away compute-fabric result.
- Implemented vectorless power:
  - Total on-chip `6.231 W`; dynamic `5.509 W`; device static `0.722 W`.
  - Static split: PS static `0.102 W`, PL static `0.620 W`.
  - Main dynamic components include clocks `0.385 W`, CLB logic `0.635 W`, signals `0.798 W`, Block RAM `0.119 W`, URAM `0.232 W`, DSPs `0.670 W`, and PS8 `2.671 W`.
  - Confidence: medium, vectorless internal activity; mode-specific SAIF or board power is still required for final Search/Track power.
- Decision:
  - `hpar2` is now the best physically implemented full learned runtime-ROM latency candidate.
  - It improves over `mlpfusebank` and produces bitstream/handoff artifacts, with routed timing inside the user-relaxed `-0.5 ns` WNS tolerance.
  - The goal is still not complete: Search `13.002 ms > 4 ms`, Track `1.801 ms > 1 ms`, Vivado nominal timing remains negative, and RTL/board same-input/same-output evidence plus mode-specific board latency/power remain open.
  - Next DSE should keep the hpar2/fused-W2 banked direction, add explicit MLP/DSP pipeline staging for the DPIP/DPOP MREG/PREG warnings, and reduce full learned operator work per invocation rather than relying on placement alone.

### Attention-Token2 / Attention-BRAM Physical Implementation

- Purpose:
  - Reduce the HPar2 physical baseline latency by increasing dense-token attention parallelism and moving the attention intermediate storage away from the high-URAM pressure point.
  - Preserve the same objective envelope: full learned runtime-ROM path, Search dispatcher prefetch-all4, on-chip parameter/nonlinear ROMs, shared runtime ATTN/MLP units, `PATCH_PAR=16`, token-loop Conv/EventConv, no observe datapath, norm storage on BRAM, fused-W2 banked MLP, `HPAR=2`, and `300 MHz` PL clock.
- Profile:
  - Combined/Vivado profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- HLS force-mode latency:
  - Search-only: estimated clock `2.777 ns` / `360.10 MHz`; latency `2,623,811..2,623,831 cycles`; interval `2,623,812..2,623,832 cycles`; max interval `8.745 ms` at 300 MHz.
  - Track-only: estimated clock `3.473 ns` / `287.94 MHz`; latency `380,710 cycles`; interval `380,711 cycles`; max interval `1.322 ms` at 300 MHz.
  - Hybrid 10% Search / 90% Track estimate from force-mode intervals: `604,023 cycles`, `2.013 ms`, `496.7 invocations/s`; worst-case remains Search `8.745 ms`.
- Force-mode path breakdown:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090` max, `mlp_head` `25,114`.
  - Track-only: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
  - The remaining bottleneck is the controller/Transformer path, not AXIS input, global buffer load, or head.
- HLS resources:
  - Search-only: BRAM_18K `265/624 = 42%`, DSP `763/1728 = 44%`, FF `75,794/460,800 = 16%`, LUT `188,424/230,400 = 81%`, URAM `76/96 = 79%`.
  - Track-only: BRAM_18K `263/624 = 42%`, DSP `886/1728 = 51%`, FF `87,415/460,800 = 18%`, LUT `184,150/230,400 = 79%`, URAM `55/96 = 57%`.
  - Combined CSynth: target `3.333 ns`, estimated `2.777 ns`; latency envelope `124,485..12,989,782 cycles`; interval envelope `124,486..12,989,783 cycles`; HLS resources BRAM_18K `267/624 = 42%`, DSP `975/1728 = 56%`, FF `124,502/460,800 = 27%`, LUT `255,013/230,400 = 110%`, URAM `76/96 = 79%`.
  - The combined HLS LUT estimate is over device capacity, but implementation packing is much lower; use the Vivado implemented report for physical fit.
- Vivado implementation:
  - Command: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: implementation and bitgen completed successfully; `.bit` and `.hwh` were generated/copied into both the overlay output and `hardware/pynq/hgtxr/`.
- Implemented artifacts:
  - `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16.bit`
  - `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16.hwh`
  - Same `.bit/.hwh` stems are copied under `hardware/pynq/hgtxr/`.
- Implemented timing:
  - Clock: `clk_pl_0`, period `3.333 ns`, frequency `300.030 MHz`.
  - Implemented timing: `WNS = -0.357 ns`, `TNS = -479.165 ns`, setup failing endpoints `4,384 / 265,551`.
  - Hold timing: `WHS = 0.010 ns`, `THS = 0.000 ns`, hold failing endpoints `0`.
  - Vivado nominal timing status: not met.
  - User-relaxed WNS status: pass under the allowed `-0.5 ns` setup slack tolerance.
- Route:
  - Logical nets `936,375`; routable nets `133,219`; fully routed nets `133,219`; routing errors `0`.
- Implemented resources:
  - CLB LUT `60,878 / 230,400 = 26.42%`.
  - LUT as Memory `3,692 / 101,760 = 3.63%`.
  - CLB registers `55,536 / 460,800 = 12.05%`.
  - Block RAM Tile `137.5 / 312 = 44.07%` (`RAMB36 = 70`, `RAMB18 = 135`).
  - URAM `76 / 96 = 79.17%`.
  - DSP `946 / 1,728 = 54.75%`.
  - Main accelerator hierarchy `hgtxr_e2e_axis_top_0`: LUT `54,588`, FF `45,862`, RAMB36 `65`, RAMB18 `133`, URAM `76`, DSP `946`.
- Implemented vectorless power:
  - Total on-chip `6.099 W`; dynamic `5.381 W`; device static `0.719 W`.
  - Static split: PS static `0.102 W`, PL static `0.617 W`.
  - Main dynamic components include clocks `0.425 W`, CLB logic `0.567 W`, signals `0.697 W`, Block RAM `0.138 W`, URAM `0.183 W`, DSPs `0.700 W`, and PS8 `2.671 W`.
  - Confidence: medium, vectorless internal activity; mode-specific SAIF or board power is still required for final Search/Track power.
- Comparison with the previous HPar2 physical baseline:
  - Search interval improves from `3,900,504 cycles / 13.002 ms` to `2,623,832 cycles / 8.745 ms`, a `32.7%` cycle reduction.
  - Track interval improves from `540,391 cycles / 1.801 ms` to `380,711 cycles / 1.322 ms`, a `29.5%` cycle reduction.
  - Hybrid 10/90 estimate improves from `876,402 cycles / 2.921 ms` to `604,023 cycles / 2.013 ms`, a `31.1%` cycle reduction.
  - URAM drops from `92` to `76`; DSP increases from `722` to `946`; CLB LUT increases from `49,867` to `60,878`; total vectorless power drops slightly from `6.231 W` to `6.099 W`.
- Decision:
  - `attntok2_attnbram` is now the best physically implemented full learned runtime-ROM latency candidate.
  - It proves ZCU104 physical fit, clean routing, bitstream generation, and 300 MHz implementation inside the user-relaxed `-0.5 ns` WNS tolerance.
  - The goal is still not complete: Search `8.745 ms > 4 ms`, Track `1.322 ms > 1 ms`, Vivado nominal timing remains negative, and RTL/board same-input/same-output evidence plus mode-specific board latency/power remain open.
  - Search must remove about `1,423,832 cycles` from the current force-mode II to reach 4 ms at 300 MHz; Track must remove about `80,711 cycles` to reach 1 ms.
  - Next DSE should keep the Attention-Token2/Attention-BRAM direction, reduce the remaining `controller_run` cycles, add explicit DSP/MLP pipeline staging for the MREG/PREG warnings, and only then rerun physical timing and board/SAIF-backed measurement.

### HPar4 Fused-W2 Negative DSE

- Purpose:
  - Test whether increasing the fused W2 hidden-lane parallelism from `HGTXR_E2E_MLP_FUSED_W2_HP_PAR=2` to `4` reduces the remaining controller-dominated Search/Track interval in the current `attntok2_attnbram` best candidate.
  - Keep all other objective-envelope decisions unchanged: full learned runtime-ROM path, Search dispatcher prefetch-all4, on-chip parameter/nonlinear ROMs, shared runtime ATTN/MLP units, `PATCH_PAR=16`, token-loop Conv/EventConv, no observe datapath, norm storage on BRAM, Attention-Token2, Attention-BRAM, and `300 MHz` HLS clock target.
- Profiles:
  - Combined/CSim profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: pass. Search output remained `[3, 3, 3, 3, 3, 3]`; Track output remained `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`, TLAST, and strict prefetch-immediate traces passed.
- HLS force-mode latency:
  - Search-only: estimated clock `2.777 ns` / `360.10 MHz`; latency `2,623,811..2,623,831 cycles`; interval `2,623,812..2,623,832 cycles`; max interval `8.745 ms` at 300 MHz.
  - Track-only: estimated clock `3.473 ns` / `287.94 MHz`; latency `380,710 cycles`; interval `380,711 cycles`; max interval `1.322 ms` at 300 MHz.
  - Hybrid 10% Search / 90% Track estimate is unchanged from `attntok2_attnbram`: `604,023 cycles`, `2.013 ms`, about `496.7 invocations/s`.
- Path breakdown:
  - Search-only is unchanged: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090` max, `mlp_head` `25,114`.
  - Track-only is unchanged: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
- HLS resources versus the promoted `hpar2` `attntok2_attnbram` candidate:
  - Search-only resource regression: DSP `763 -> 891`, FF `75,794 -> 88,531`, LUT `188,424 -> 199,119`; BRAM_18K and URAM remain `265` and `76`.
  - Track-only resource regression: DSP `886 -> 1,014`, FF `87,415 -> 100,152`, LUT `184,150 -> 194,845`; BRAM_18K and URAM remain `263` and `55`.
- Bottleneck evidence:
  - HLS still reports memory-port II violations in the MLP/head path, including `w2_weight_cache` and global-buffer/token memories implemented as 2-port RAMs.
  - Increasing `HP_PAR` alone adds arithmetic hardware but does not increase the effective memory port/bank bandwidth feeding the fused W2 path, so it cannot reduce the current top-level interval.
- Decision:
  - `hpar4` is a negative DSE. It is functionally valid in CSim and HLS-synthesizable, but it gives no Search/Track latency improvement and increases DSP/FF/LUT pressure.
  - Do not promote or route this candidate.
  - Keep `attntok2_attnbram` with `HGTXR_E2E_MLP_FUSED_W2_HP_PAR=2` as the best physically implemented full learned runtime-ROM candidate.
  - Next DSE should target the memory-port bottleneck directly: split/bank `w2_weight_cache`, bank or reshape the global buffer hot arrays used by attention/MLP/head, or reduce per-layer controller work. More hidden-lane arithmetic parallelism is not useful until the corresponding weight/global-buffer reads are banked.

### W2Bank8 Fused-W2 Negative DSE

- Purpose:
  - Test whether explicit cyclic partitioning of the fused MLP W2 cache removes the remaining `w2_weight_cache` memory-port II violation that survived the `hpar4` attempt.
  - Keep the same full learned runtime-ROM design envelope as `hpar4_tok2_attntok2_attnbram`, but add `HGTXR_E2E_W2_WEIGHT_CACHE_BANKS=8`.
- Implementation:
  - Added a compile-time default `HGTXR_E2E_W2_WEIGHT_CACHE_BANKS=1` so promoted profiles are unchanged unless a profile opts in.
  - Added conditional `#pragma HLS ARRAY_PARTITION variable=w2_weight_cache cyclic factor=HGTXR_E2E_W2_WEIGHT_CACHE_BANKS dim=1`.
  - Added combined/Search-only/Track-only `hpar4_w2bank8` profiles in `hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2bank8_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: pass. Search output remained `[3, 3, 3, 3, 3, 3]`; Track output remained `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`, TLAST, and strict prefetch-immediate traces passed.
- Search-only CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_hpar4_w2b8_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Estimated clock `2.777 ns`; latency `2,623,815..2,623,835 cycles`; interval `2,623,816..2,623,836 cycles`; max interval remains `8.745 ms` at 300 MHz.
  - Path: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,094` max, `mlp_head` `25,114`.
  - Resource estimate: BRAM_18K `339`, DSP `891`, FF `90,532`, LUT `199,958`, URAM `76`.
- Bottleneck evidence:
  - The HLS report confirms the pragma was applied: `array_partition | variable=w2_weight_cache cyclic factor=8 dim=1`.
  - The generated MLP W2 pipeline still reports a memory-port II violation on `w2_weight_cache_load_2`, so the key loop remains at achieved II `2`.
  - The partition creates multiple BRAM-backed W2 cache banks, but the current load-address pattern still maps conflicting unrolled reads onto the same effective bank/port group.
- Decision:
  - `w2bank8` is also a negative DSE. It does not improve Search latency, slightly worsens the controller/top interval by a few cycles, and increases BRAM_18K from `265` to `339` versus the `hpar4` Search-only estimate.
  - Do not run Track/Vivado for this candidate unless a later code change alters the W2 access pattern.
  - Keep the promoted `hpar2` `attntok2_attnbram` physical candidate as the current best.
  - The next credible latency DSE should change the schedule or data layout, not only add cyclic partitioning: split W2 storage by hidden lane with explicit bank-select-free access, restructure the MLP W2 loop to read one bank per lane deterministically, or reduce per-layer controller work outside the W2 load loop.

### SkipHidden Fused-W2 Neutral DSE

- Purpose:
  - Test whether enabling `HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1` reduces the remaining controller/global-buffer pressure by removing the `gb.hidden[t][h]` write in the fused W2 path.
  - Keep the promoted `hpar2` `attntok2_attnbram` profile otherwise unchanged.
- Profiles:
  - Combined/CSim profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_skiphidden_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_skiphidden_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: pass. Search output remained `[3, 3, 3, 3, 3, 3]`; Track output remained `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`, TLAST, and strict prefetch-immediate traces passed.
- Search-only CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_skiphid_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Estimated clock `2.777 ns`; latency `2,623,811..2,623,831 cycles`; interval `2,623,812..2,623,832 cycles`; max interval remains `8.745 ms` at 300 MHz.
  - Path remains unchanged: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090` max, `mlp_head` `25,114`.
  - Resource estimate matches the promoted Search-only estimate: BRAM_18K `265`, DSP `763`, FF `75,794`, LUT `188,424`, URAM `76`.
- Track-only CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_skiphid_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Estimated clock `3.473 ns`; latency `380,710 cycles`; interval `380,711 cycles`; max interval remains `1.322 ms` in the HLS report.
  - Resource estimate matches the promoted Track-only estimate: BRAM_18K `263`, DSP `886`, FF `87,415`, LUT `184,150`, URAM `55`.
- Bottleneck evidence:
  - The fused W2 loop report now shows `VITIS_LOOP_2837_15_VITIS_LOOP_2839_16` at achieved II `1`, but the top-level and controller intervals are unchanged versus the promoted `hpar2` `attntok2_attnbram` Search-only build.
  - This indicates the hidden-store write was not the top-level limiting factor in the promoted profile, or was already effectively removed from the critical transaction schedule by HLS.
- Decision:
  - `skiphidden` is functionally valid and harmless as an opt-in profile, but it is not a latency improvement.
  - Do not promote or route it as-is; keep the promoted `hpar2` `attntok2_attnbram` physical candidate as the current best.
  - The remaining latency work must reduce controller work at a higher level, not just remove the fused hidden write.

### DenseToken4 Attention-Projection DSE

- Purpose:
  - Test whether increasing dense projection token parallelism from `HGTXR_E2E_DENSE_TOKEN_PAR=2` to `4` reduces the attention QKV/output-projection bottleneck while keeping the MLP token path at `HGTXR_E2E_MLP_TOKEN_PAR=2`.
  - Keep the current full learned runtime-ROM design envelope unchanged: Search/Track runtime mode, shared ATTN/MLP units, on-chip learned parameter ROM, on-chip nonlinear LUT ROM, dispatcher prefetch-all4, Attention-Token2, Attention-BRAM, `PATCH_PAR=16`, and `300 MHz` HLS clock target.
- Profiles:
  - Combined/CSim profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Command: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Result: pass. Search output remained `[3, 3, 3, 3, 3, 3]`; Track output remained `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`, TLAST, and strict prefetch-immediate traces passed.
- HLS force-mode latency:
  - Search-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search-only estimated clock `2.777 ns`; latency `2,465,731..2,465,751 cycles`; interval `2,465,732..2,465,752 cycles`; max interval `8.218 ms` in the HLS report, or about `8.219 ms` at exact `300 MHz`.
  - Track-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track-only estimated clock `3.473 ns` / `287.94 MHz`; latency `360,902 cycles`; interval `360,903 cycles`; max interval `1.253 ms` in the HLS report, or about `1.203 ms` at exact `300 MHz`.
  - Hybrid 10% Search / 90% Track estimate from force-mode II: `571,388 cycles`, about `1.905 ms` at `300 MHz`, about `524.9 invocations/s`; worst case remains Search `8.218 ms`.
- Path breakdown:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,030,990..2,031,010`, `mlp_head` `25,114`.
  - Track-only: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `236,422`, `mlp_head` `6,682`.
  - Controller details: Search compute loop is `4 x 500,816 cycles`; Track compute loop is `2 x 111,281 cycles`. The shared ATTN/MLP structure is still temporal, with mode-specific trip counts.
- HLS resources versus the promoted `hpar2` `attntok2_attnbram` candidate:
  - Search-only resources increase from BRAM_18K `265 -> 329`, DSP `763 -> 1,019`, FF `75,794 -> 92,588`, LUT `188,424 -> 212,438`, URAM `76 -> 76`.
  - Track-only resources increase from BRAM_18K `263 -> 327`, DSP `886 -> 1,142`, FF `87,415 -> 104,192`, LUT `184,150 -> 208,129`, URAM `55 -> 55`.
  - Search LUT utilization reaches `92%` of ZCU104 in HLS estimates, and Track LUT utilization reaches `90%`; this is a significant physical-implementation risk even though the HLS resource table remains under the device total.
- Latency comparison:
  - Search improves from `2,623,832 cycles` / `8.745 ms` to `2,465,752 cycles` / `8.218 ms`, saving `158,080 cycles` or about `6.0%`.
  - Track improves from `380,711 cycles` / `1.322 ms` to `360,903 cycles` / `1.253 ms` in HLS report time, saving `19,808 cycles` or about `5.2%`.
  - Hybrid 10/90 improves from `604,023 cycles` / `2.013 ms` to `571,388 cycles` / about `1.905 ms`.
- Decision:
  - `dtok4` is a positive HLS latency DSE, unlike `hpar4`, `w2bank8`, and `skiphidden`.
  - Combined full-runtime packaging later failed the ZCU104 fit gate at HLS-estimate level: report `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` shows BRAM_18K `331/624 = 53%`, DSP `1,231/1,728 = 71%`, FF `145,770/460,800 = 31%`, LUT `278,828/230,400 = 121%`, URAM `76/96 = 79%`.
  - Therefore `dtok4` is not a physical baseline and should not be routed in Vivado as-is. The force-mode latency improvement is real, but the combined runtime implementation exceeds the LUT budget.
  - The final latency target remains open: Search still exceeds `4 ms` by about `1,265,752 II cycles` at `300 MHz`, and Track still exceeds `1 ms` by about `60,903 II cycles` at `300 MHz`.

### DenseToken3 Attention-Projection DSE

- Purpose:
  - Recover some of the `dtok4` latency improvement with lower resource pressure by setting `HGTXR_E2E_DENSE_TOKEN_PAR=3` and keeping `HGTXR_E2E_MLP_TOKEN_PAR=2`.
- Profiles:
  - Combined/package profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok3_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles were registered with `_search_only` and `_track_only`, but were not run because the combined fit gate failed first.
- CSim:
  - Combined CSim passed. Search output remained `[3, 3, 3, 3, 3, 3]`; Track output remained `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`, TLAST, and strict prefetch-immediate traces passed.
- Combined/package CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok3_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Estimated clock `2.777 ns` / `360.10 MHz`.
  - Top max latency `14,180,694 cycles`; max interval `14,180,695 cycles`; max absolute latency `47.264 ms`. This combined runtime max is not a mode-specific latency and is used here only as synthesis/package evidence.
  - HLS resource estimate: BRAM_18K `299/624 = 47%`, DSP `1,103/1,728 = 63%`, FF `137,005/460,800 = 29%`, LUT `274,923/230,400 = 119%`, URAM `76/96 = 79%`.
- Decision:
  - `dtok3` also fails the ZCU104 fit gate at HLS-estimate level because LUT utilization is `119%`.
  - Do not run force-mode Search/Track or Vivado for `dtok3` unless the implementation is first refactored to reduce LUT pressure.
  - Current physically implemented best remains `attntok2_attnbram`: Search II `2,623,832 cycles` / `8.745 ms`, Track II `380,711 cycles` / `1.322 ms`, Hybrid 10/90 `604,023 cycles` / `2.013 ms`, with Vivado bitgen complete and WNS `-0.357 ns` under the user-relaxed `-0.5 ns` tolerance.

### PatchPar32 Conv/EventConv DSE

- Purpose:
  - Test whether raising `HGTXR_E2E_PATCH_PAR` from `16` to `32` reduces the Search frame-conv and Track event-conv front-end latency while keeping the promoted `hpar2` `attntok2_attnbram` full learned runtime-ROM design envelope unchanged.
- Profiles:
  - Combined profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Combined CSim passed with the same functional outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
- HLS force-mode latency:
  - Search-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search-only estimated clock `2.777 ns` / `360.10 MHz`; latency `2,500,931..2,500,951 cycles`; interval `2,500,932..2,500,952 cycles`; max interval `8.336 ms`.
  - Track-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track-only estimated clock `3.473 ns` / `287.94 MHz`; latency `356,134 cycles`; interval `356,135 cycles`; HLS report absolute latency `1.237 ms`, or about `1.187 ms` at exact `300 MHz`.
  - Hybrid 10% Search / 90% Track estimate from force-mode II: `570,617 cycles`, about `1.902 ms` at exact `300 MHz`, about `525.7 invocations/s`; worst case remains Search `8.336 ms`.
- Path breakdown:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `2,189,070..2,189,090`, `mlp_head` `25,114`.
  - Track-only: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
- Latency comparison versus the current physically implemented `patchpar16` `attntok2_attnbram` candidate:
  - Search improves from `2,623,832` to `2,500,952` II cycles, saving `122,880 cycles` or about `4.7%`.
  - Track improves from `380,711` to `356,135` II cycles, saving `24,576 cycles` or about `6.5%`.
  - Hybrid 10/90 improves from `604,023` to `570,617` II cycles, saving `33,406 cycles` or about `5.5%`.
- HLS resources:
  - Search-only: BRAM_18K `265/624 = 42%`, DSP `776/1,728 = 44%`, FF `76,468/460,800 = 16%`, LUT `190,496/230,400 = 82%`, URAM `76/96 = 79%`.
  - Track-only: BRAM_18K `263/624 = 42%`, DSP `900/1,728 = 52%`, FF `88,561/460,800 = 19%`, LUT `186,306/230,400 = 80%`, URAM `55/96 = 57%`.
  - Combined full-runtime report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Combined full-runtime HLS estimate: BRAM_18K `267/624 = 42%`, DSP `1,002/1,728 = 57%`, FF `126,407/460,800 = 27%`, LUT `258,869/230,400 = 112%`, URAM `76/96 = 79%`.
- Decision:
  - `patchpar32` is now the best physically implemented full learned runtime-ROM candidate within the `attntok2_attnbram` design family.
  - The combined HLS LUT estimate was over-conservative: HLS estimated LUT `258,869/230,400 = 112%`, but Vivado post-route implemented CLB LUTs are `63,327/230,400 = 27.49%`.
  - Vivado route and bitgen completed for `hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay`.
  - Final implemented timing is nominally failing but within the user-relaxed gate: WNS `-0.082 ns`, TNS `-60.586 ns`, WHS `0.002 ns`, THS `0.000 ns`, with user allowance WNS down to `-0.5 ns`.
  - Route status is clean: logical nets `962,898`, routable nets `136,548`, fully routed nets `136,548`, routing errors `0`.
  - Implemented resources are LUT `63,327/230,400 = 27.49%`, FF `57,218/460,800 = 12.42%`, BRAM Tile `137.5/312 = 44.07%`, URAM `76/96 = 79.17%`, DSP `973/1,728 = 56.31%`.
  - Vectorless implemented power is total `6.217 W`, dynamic `5.498 W`, static `0.720 W`; PS static `0.102 W`, PL static `0.618 W`. This is not mode-specific power.
  - Artifacts:
    - Bitstream: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16.bit`
    - HWH: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16.hwh`
  - Remaining latency gap versus the final targets:
    - Search: `2,500,952 cycles`, about `8.336 ms` at `300 MHz`; still above `4 ms` by about `1,300,952 cycles`.
    - Track: `356,135 cycles`, about `1.187 ms` at exact `300 MHz`; still above `1 ms` by about `56,135 cycles`.

### PatchPar32 DenseToken4 Combined DSE

- Purpose:
  - Combine the two independently positive latency directions: `PATCH_PAR=32` for Conv/EventConv and `DENSE_TOKEN_PAR=4` with `MLP_TOKEN_PAR=2` for QKV/output projection.
  - Preserve the requested full learned runtime-ROM envelope: shared ATTN/MLP units, runtime Search/Track scheduler, Search dispatcher prefetch-all4, on-chip learned parameter ROM, on-chip nonlinear ROM, and `300 MHz` HLS target.
- Profiles:
  - Combined/CSim profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Result: pass. Search output `[3, 3, 3, 3, 3, 3]`; Track output `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`; TLAST and strict prefetch-immediate traces passed.
- HLS force-mode latency:
  - Search-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search-only estimated clock `2.777 ns`; interval max `2,342,872 cycles`; about `7.810 ms` at exact `300 MHz`.
  - Track-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track-only estimated clock `3.473 ns`; interval `336,327 cycles`; about `1.121 ms` at exact `300 MHz`.
  - Hybrid 10% Search / 90% Track estimate from force-mode II: `536,981.5 cycles`, about `1.790 ms` at exact `300 MHz`, about `558.7 invocations/s`.
- Path breakdown:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `2,031,010`, `mlp_head` `25,114`.
  - Track-only: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `236,422`, `mlp_head` `6,682`.
- HLS resources:
  - Search-only: BRAM_18K `329/624 = 52%`, DSP `1,032/1,728 = 59%`, FF `93,262/460,800 = 20%`, LUT `214,510/230,400 = 93%`, URAM `76/96 = 79%`.
  - Track-only: BRAM_18K `327/624 = 52%`, DSP `1,156/1,728 = 66%`, FF `105,338/460,800 = 22%`, LUT `210,285/230,400 = 91%`, URAM `55/96 = 57%`.
  - Combined full-runtime report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Combined HLS estimate: BRAM_18K `331/624 = 53%`, DSP `1,258/1,728 = 72%`, FF `147,675/460,800 = 32%`, LUT `282,684/230,400 = 122%`, URAM `76/96 = 79%`.
- Decision:
  - `patch32_dtok4` is now the best HLS force-mode latency candidate observed so far: Search improves from `8.336 ms` to about `7.810 ms`, Track improves from about `1.187 ms` to about `1.121 ms`, and Hybrid improves from about `1.902 ms` to about `1.790 ms`.
  - Package/IP export completed: `generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` and `impl/export.zip`.
  - Vivado route/bitgen completed for `hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_overlay`: route errors `0`, fully routed nets `158,993/158,993`, bitgen pass.
  - Final post-route physopt timing: WNS `-0.238 ns`, TNS `-549.019 ns`, WHS `0.000 ns`, THS `0.000 ns`, setup failing endpoints `4,620`. This meets the user-approved `WNS >= -0.500 ns` experiment floor but remains official negative-WNS timing fail.
  - Implemented utilization: CLB LUT `73,832/230,400 = 32.05%`, CLB registers `69,126/460,800 = 15.00%`, Block RAM Tile `169.5/312 = 54.33%`, URAM `76/96 = 79.17%`, DSP `1,229/1,728 = 71.12%`.
  - Implemented vectorless power: total on-chip `7.150 W`, dynamic `6.423 W`, device static `0.727 W`.
  - Structural contract audit for this exact candidate: `hardware/generated/signoff/patch32_dtok4_300_contract_audit_2026_06_30.{json,md}`, status `pass`, `16/16` checks.
  - Board-smoke plumbing added for this exact candidate under PYNQ variant `par32-patch32-dtok4-300`, including Search, Track, and Hybrid 10:90 bundle packaging and result-validation presets.
  - Bundle validation artifacts all pass:
    - `hardware/generated/signoff/par32_patch32_dtok4_300_search_bundle_validation_2026_06_30.json`
    - `hardware/generated/signoff/par32_patch32_dtok4_300_track_bundle_validation_2026_06_30.json`
    - `hardware/generated/signoff/par32_patch32_dtok4_300_hybrid_10_90_bundle_validation_2026_06_30.json`
  - ZCU104 remote dry-run plan generated: `hardware/generated/signoff/par32_patch32_dtok4_300_board_latency_run_2026_06_30.{json,md}`, status `dry-run`.
  - Board latency gate generated: `hardware/generated/signoff/par32_patch32_dtok4_300_board_latency_gate_2026_06_30.{json,md}`, status `missing` because physical board result JSONs are not available yet.
  - Goal-status artifact for this exact candidate: `hardware/generated/signoff/patch32_dtok4_300_goal_status_2026_06_30.{json,md}`, status `physical-structural-pass-latency-target-fail`, board plumbing `pass`.
  - `patch32_dtok4` is now the best physical latency candidate under the user-approved negative-WNS experiment floor, but it is not official timing-clean signoff.
  - Remaining latency gap versus final targets: Search still exceeds `4 ms` by `1,142,872 cycles`; Track still exceeds `1 ms` by `36,327 cycles` at exact `300 MHz`.
  - Physical ZCU104 execution remains pending, so board p95/p99, measured DMA bandwidth, and measured Search/Track/Hybrid latency are still unavailable.

### PatchPar32 DenseToken4 Board Contract Correction

- Scope:
  - Corrected the PYNQ board-smoke expected output contract for the exact `patch32_dtok4` physical candidate.
  - The board bitstream uses the combined runtime profile, so the board-smoke expected vectors must follow the combined runtime CSim contract rather than the single-mode force-mode CSim output.
- Correct board-smoke contract:
  - Search expected raw output: `[3, 3, 3, 3, 3, 3]`
  - Track expected raw output: `[217, 217, 217, 217, 217, 217]`
- Reason:
  - Combined runtime CSim evidence for this candidate reports Search `[3, 3, 3, 3, 3, 3]` and Track `[217, 217, 217, 217, 217, 217]`.
  - A Track-only force-mode CSim run observed `[215, 215, 215, 215, 215, 215]`, but that is not the board contract because it compiles a single-mode force profile rather than the combined runtime bitstream.
- Updated files:
  - `hardware/tools/package_e2e_axis_dma_pynq_bundle.py`
  - `hardware/tools/validate_pynq_smoke_result.py`
  - `hardware/tools/validate_pynq_bundle_package.py`
  - `hardware/pynq/hgtxr/run_e2e_axis_dma_hybrid_smoke.py`
- Regenerated bundles:
  - Search tar SHA256: `8ffded397fb82d2996e74623328a66aca4291225c29484ad8143144e1cdb36d8`
  - Track tar SHA256: `095c86b477254dcb3a2f9912aae85bbedf1196d3348b948f558705ec34e448cc`
  - Hybrid 10:90 tar SHA256: `c1d81829eb13a8276b518571f9325a8cc1d285e98a0b768a51a389200febcf4c`
  - Validation:
  - `python3 -m py_compile hardware/tools/package_e2e_axis_dma_pynq_bundle.py hardware/tools/validate_pynq_smoke_result.py hardware/tools/validate_pynq_bundle_package.py hardware/pynq/hgtxr/run_e2e_axis_dma_hybrid_smoke.py` passed.
  - Search, Track, and Hybrid 10:90 bundle validations regenerated under `hardware/generated/signoff/par32_patch32_dtok4_300_{search,track,hybrid_10_90}_bundle_validation_2026_06_30.json`; all `pass`.
  - Bundle script audit confirms the Track smoke command now uses `--expect-out-raw 217 217 217 217 217 217`.
  - Goal-status artifact regenerated: `hardware/generated/signoff/patch32_dtok4_300_goal_status_2026_06_30.{json,md}`, status `physical-structural-pass-latency-target-fail`.

### PatchPar32 MLPToken3/DenseToken4 DSE

- Purpose:
  - Test the midpoint between the physically promoted `patch32_dtok4` profile (`MLP_TOKEN_PAR=2`, `DENSE_TOKEN_PAR=4`) and the faster but more resource-heavy `mtok4_dtok4` profile.
  - Keep the requested full learned runtime-ROM envelope unchanged: shared ATTN/MLP units, runtime Search/Track scheduler, Search dispatcher prefetch-all4, on-chip learned parameter ROM, and on-chip nonlinear ROM.
- Profiles:
  - Combined/CSim profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`
  - Search-only and Track-only force-mode profiles use the same suffix with `_search_only` and `_track_only`.
- CSim:
  - Result: pass. Search output `[3, 3, 3, 3, 3, 3]`; Track output `[217, 217, 217, 217, 217, 217]`; runtime states `0/1`; TLAST and strict prefetch-immediate traces passed.
- HLS force-mode latency:
  - Search-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok3_dtok4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Search-only estimated clock `2.777 ns`; interval max `2,147,147 cycles`; about `7.157 ms` at exact `300 MHz`.
  - Track-only report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok3_dtok4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Track-only estimated clock `3.473 ns`; interval `321,651 cycles`; about `1.072 ms` at exact `300 MHz`.
  - Hybrid 10% Search / 90% Track estimate from force-mode II: `504,200.6 cycles`, about `1.681 ms` at exact `300 MHz`, about `595.0 invocations/s`.
- Path breakdown:
  - Search-only: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,301`, `controller_run` `1,835,266`, `mlp_head` `25,124`.
  - Track-only: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,082`, `controller_run` `221,734`, `mlp_head` `6,688`.
- Comparison against `patch32_dtok4`:
  - Search II improves from `2,342,872` to `2,147,147 cycles`, an `8.35%` reduction.
  - Track II improves from `336,327` to `321,651 cycles`, a `4.36%` reduction.
  - Hybrid 10:90 II improves from `536,981.5` to `504,200.6 cycles`.
- HLS resources:
  - Search-only: BRAM_18K `345/624 = 55%`, DSP `1,128/1,728 = 65%`, FF `122,359/460,800 = 26%`, LUT `252,281/230,400 = 109%`, URAM `76/96 = 79%`.
  - Track-only: BRAM_18K `343/624 = 54%`, DSP `1,252/1,728 = 72%`, FF `134,505/460,800 = 29%`, LUT `247,750/230,400 = 107%`, URAM `55/96 = 57%`.
- Decision:
  - Do not promote `mtok3_dtok4` as the physical candidate. It improves latency, but Search and Track force-mode HLS LUT estimates both exceed the ZCU104 device budget.
  - `mtok4_dtok4` remains the faster HLS point but is also over budget: Search II `1,862,616 cycles`, Track II `276,295 cycles`, with Search LUT `111%` and Track LUT `109%`.
  - The current physical baseline remains `patch32_dtok4`: it is routed/bitgen complete under the user-approved WNS floor and has passing structural and PYNQ bundle plumbing, but final latency targets and board measurements remain open.

### Initial Interval Calculation: Full AXI Learned vs Runtime-Mode mem16

- Method:
  - Use the HLS top-level `Interval max` as the initiation interval for one non-pipelined E2E invocation.
  - Convert cycles to latency with `time_ms = II_cycles / clock_hz * 1000`.
  - The older `full_axi_mem16` and `runtime_mode_mem16` baselines were built at `200 MHz`; the newer ROM/dispatcher DSE points are reported at the current `300 MHz` target.
- Previous full AXI learned profile, `par32_runtime_full_axi_mem16`, at `200 MHz`:
  - Track Path: II `627,118 cycles`, about `3.135590 ms`, max rate about `318.9 invocations/s`.
  - Search Path: II `5,101,976 cycles`, about `25.509880 ms`, max rate about `39.2 invocations/s`.
- Previous runtime-mode profile, `par32_runtime_mode_mem16`, at `200 MHz`:
  - Track Path: II `76,151 cycles`, about `0.380755 ms`, max rate about `2,626.4 invocations/s`.
  - Search Path: II `594,841 cycles`, about `2.974205 ms`, max rate about `336.2 invocations/s`.
- Interpretation:
  - `runtime_mode_mem16` was much faster than the full AXI learned path because it was a simplified runtime-mode/fast-path profile, not the current full learned runtime-ROM Transformer implementation.
  - It is therefore useful as an architectural speed reference, but it is not an apples-to-apples replacement for the current full learned Search/Track shared ATTN/MLP implementation.

### PatchPar32 MLPToken4/DenseToken4 AttentionQuery2 DSE

- Purpose:
  - Add a real attention-query parallelism knob after confirming that the earlier `attntok2` profile name did not actually change query-token parallelism.
  - Test whether `HGTXR_E2E_ATTN_QUERY_PAR=2` can reduce the remaining Search bottleneck in the `mtok4_dtok4` high-speed DSE point.
- Implementation:
  - Added `HGTXR_E2E_ATTN_QUERY_PAR` with default `1` in `hardware/hls/include/hgtxr_e2e_vit.hpp`.
  - Added an attention-core branch that processes two query tokens per key/value sweep when `HGTXR_E2E_ATTN_QUERY_PAR=2`.
  - Added combined/Search-only/Track-only no-board runner profiles under the `patch32_mtok4_dtok4_aq2` suffix.
- Search-only CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Estimated clock `2.777 ns`; top latency max `1,709,783 cycles`; II max `1,709,784 cycles`.
  - Exact `300 MHz` latency/II: about `5.699 ms`; max rate about `175.5 invocations/s`.
  - Path breakdown: `axis_read_frame` `16,386 cycles`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,397,922`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `131,388 cycles`, MLP unit `211,151 cycles`, 4 runtime blocks.
  - Attention breakdown: layernorm `39,809 cycles`, QKV projection `20,769`, attention core `50,503`, output projection `20,298`.
  - HLS resources: BRAM_18K `361/624 = 57%`, DSP `1,288/1,728 = 74%`, FF `133,107/460,800 = 28%`, LUT `264,616/230,400 = 114%`, URAM `92/96 = 95%`.
- Track-only CSynth:
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`
  - Estimated clock `3.473 ns`; top latency `268,710 cycles`; II `268,711 cycles`.
  - Exact `300 MHz` latency/II: about `0.896 ms`; max rate about `1,116.4 invocations/s`.
  - Path breakdown: `axis_read_frame` `4,109 cycles`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `168,806`, `mlp_head` `6,682`.
  - Controller breakdown: dispatcher prefetch `6,926 cycles`, attention unit `25,257 cycles`, MLP unit `52,212 cycles`, 2 runtime blocks.
  - Attention breakdown: layernorm `9,953 cycles`, QKV projection `5,217`, attention core `4,567`, output projection `5,513`.
  - HLS resources: BRAM_18K `359/624 = 57%`, DSP `1,412/1,728 = 81%`, FF `145,179/460,800 = 31%`, LUT `260,389/230,400 = 113%`, URAM `71/96 = 73%`.
- Hybrid 10% Search / 90% Track estimate:
  - II `0.1 * 1,709,784 + 0.9 * 268,711 = 412,818.3 cycles`.
  - Exact `300 MHz` average interval: about `1.376 ms`.
  - Average max rate: about `726.7 invocations/s`.
  - Worst-case mode latency remains Search: about `5.699 ms`.
- Comparison:
  - Versus `patch32_dtok4`: Search II `2,342,872 -> 1,709,784 cycles`; Track II `336,327 -> 268,711 cycles`; Hybrid II `536,981.5 -> 412,818.3 cycles`.
  - Versus `mtok4_dtok4`: Search II `1,862,616 -> 1,709,784 cycles`; Track II `276,295 -> 268,711 cycles`; Hybrid II `434,927.1 -> 412,818.3 cycles`.
- Decision:
  - `AQ2` is the best HLS latency result observed so far for the current full learned runtime-ROM family: Search about `5.699 ms`, Track about `0.896 ms`, Hybrid 10:90 about `1.376 ms`.
  - Do not promote it as a ZCU104 physical candidate yet. It still misses the `4 ms` Search target and the force-mode HLS estimates exceed the LUT budget: Search `114%`, Track `113%`.
  - The current physical baseline remains `patch32_dtok4` because it has completed route/bitgen under the user-approved WNS floor, even though its latency is slower.

### PatchPar32 AQ2 MLP Follow-up DSE

- Purpose:
  - Determine whether the remaining Search latency after `AQ2` is reducible through MLP-side tweaks.
  - Treat these as HLS probes only unless they improve both latency and ZCU104 fit risk.
- `AQ2 + skip hidden store`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok4_dtok4_aq2_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_skiph_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Result: identical to `AQ2` Search-only. Top II remains `1,709,784 cycles`, about `5.699 ms`; MLP unit remains `211,151 cycles`; resources remain BRAM_18K `361`, DSP `1,288`, LUT `264,616`, URAM `92`.
  - Decision: reject as a latency/resource improvement.
- `AQ2 + HGTXR_E2E_MLP_FUSED_W2_HP_PAR=4`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_hp4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Top II remains `1,709,784 cycles`, about `5.699 ms`; controller remains `1,397,922 cycles`; MLP unit remains `211,151 cycles`.
  - Resource impact is negative: DSP increases to `1,544/1,728 = 89%`; LUT increases to `275,267/230,400 = 119%`; BRAM_18K remains `361`; URAM remains `92`.
  - Decision: reject. It spends DSP/LUT without reducing the controlling MLP loop latency.
- `AQ2 + HGTXR_E2E_MLP_TOKEN_PAR=8`:
  - Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok8_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - CSynth was intentionally interrupted after the HLS IR expansion showed the probe was too large for fast DSE. `csynth_design_size.rpt` reported `354,635` instructions after `Array/Struct`; `hgtxr_e2e_mlp_unit<0>` alone expanded to `276,469` instructions at that phase.
  - No latency or utilization signoff should be inferred because C-synthesis did not complete.
  - Decision: do not continue this exact `mtok8` shape. The next useful direction is not more brute-force MLP token unroll; it is a lower-resource MLP refactor or memory/banking change that reduces the `VITIS_LOOP_2974_3` MLP body without exploding IR size.
- `AQ2 + full hidden-lane W2 bank cache`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hbank_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_hp4_w2hbank_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Result: top II `1,636,068 cycles`, about `5.454 ms` at exact `300 MHz`; controller `1,324,206 cycles`; MLP unit `192,722 cycles`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,324,206`, `mlp_head` `25,114`.
  - Resource impact is not viable: BRAM_18K `827/624 = 132%`, DSP `1,544/1,728 = 89%`, FF `146,090/460,800 = 31%`, LUT `281,384/230,400 = 122%`, URAM `92/96 = 95%`.
  - Decision: latency improves versus AQ2, but the full hidden-lane W2 cache explodes BRAM and cannot be promoted.
- `AQ2 + hgroup-local W2 bank cache`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq2_hp4_w2hgroup_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Result: top II `1,625,268 cycles`, about `5.418 ms` at exact `300 MHz`; controller `1,313,406 cycles`; MLP unit `190,022 cycles`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,313,406`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `131,388 cycles`, MLP unit `190,022 cycles`, 4 runtime blocks.
  - HLS resources: BRAM_18K `315/624 = 50%`, DSP `1,544/1,728 = 89%`, FF `152,242/460,800 = 33%`, LUT `273,839/230,400 = 118%`, URAM `92/96 = 95%`.
  - Decision: this is the fastest Search-only HLS latency point observed so far in the current full learned runtime-ROM family and fixes the full hidden-bank BRAM explosion, but it is still not promotable because Search is `5.418 ms > 4 ms` and LUT is `118%` of ZCU104.
- `AQ4 + hgroup-local W2 bank cache`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Result: top II `1,549,044 cycles`, about `5.163 ms` at exact `300 MHz`; controller `1,237,182 cycles`; MLP unit remains `190,022 cycles`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,237,182`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `112,332 cycles`, MLP unit `190,022 cycles`, 4 runtime blocks.
  - Attention-core local latency improves to `31,447 cycles`, but the ATTN unit still includes LayerNorm/QKV/output projection and only improves from `131,388` to `112,332 cycles`.
  - HLS resources become non-viable: BRAM_18K `315/624 = 50%`, DSP `1,672/1,728 = 96%`, FF `163,522/460,800 = 35%`, LUT `288,995/230,400 = 125%`, URAM `124/96 = 129%`.
  - Decision: reject for promotion. `ATTN_QUERY_PAR=4` is cycle-positive but violates URAM, LUT, and near-saturates DSP; it does not close the `4 ms` Search target.
- `AQ4 + hgroup-local W2 bank cache + Q/K/V/ATTN BRAM`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Result: top II `1,549,044 cycles`, about `5.163 ms` at exact `300 MHz`; controller `1,237,182 cycles`; attention unit `112,332 cycles`; MLP unit `190,022 cycles`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,237,182`, `mlp_head` `25,114`.
  - Resource shift versus AQ4+w2hgroup: BRAM_18K `315 -> 443`, URAM `124 -> 28`, with DSP/FF/LUT effectively unchanged at DSP `1,672`, FF `163,522`, LUT `288,995`.
  - HLS resources: BRAM_18K `443/624 = 70%`, DSP `1,672/1,728 = 96%`, FF `163,522/460,800 = 35%`, LUT `288,995/230,400 = 125%`, URAM `28/96 = 29%`.
  - Decision: this fixes the AQ4 URAM overflow without changing cycles, but it is still not promotable because LUT remains `125%`, DSP is `96%`, and Search remains `5.163 ms > 4 ms`.
- `AQ4 + hgroup-local W2 bank cache + Q/K/V/ATTN BRAM` Vivado no-board physical probe:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Project: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_overlay`.
  - Output overlay: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_overlay`.
  - Bitstream was generated and copied to the overlay and PYNQ package paths. The `.hwh` came from the BD handoff because no implemented-run `.hwh` was present.
  - Route status: fully routed nets `207,407/207,407`; route errors `0`.
  - Final post-route physopt timing: WNS `-0.527 ns`, TNS `-2247.483 ns`, setup failing endpoints `17,645`, WHS `0.006 ns`, THS `0.000 ns`, clock `300.030 MHz`.
  - User-approved experimental floor was `WNS >= -0.500 ns`; this run misses it by `0.027 ns`. It is therefore a bitgen-complete but timing-failed physical probe.
  - Worst setup path is inside the MLP unit from `mul_14s_15s_29_5_1_U4940/buff2_reg/DSP_OUTPUT_INST/CLK` to `add_ln3247_312_reg_33461_reg[25]/D`. Data path delay is `3.649 ns`, with route `3.077 ns` (`84.323%`) and logic `0.572 ns` (`15.677%`).
  - Implemented utilization: CLB LUT `95,920/230,400 = 41.63%`, LUT-as-logic `83,297/230,400 = 36.15%`, CLB registers `98,554/460,800 = 21.39%`, Block RAM Tile `227/312 = 72.76%`, URAM `28/96 = 29.17%`, DSP `1,641/1,728 = 94.97%`.
  - Implemented vectorless power: total on-chip `7.723 W`, dynamic `6.996 W`, device static `0.727 W`; static split is PS `0.104 W`, PL `0.623 W`.
  - Major dynamic buckets: `hgtxr_e2e_axis_top_0` `4.089 W`, `psu` `2.681 W`, `axi_mem` `0.183 W`, `axi_ctrl` `0.020 W`, `axi_dma_out` `0.018 W`, `axi_dma_in` `0.005 W`. Component view: clocks `0.664 W`, CLB logic `0.851 W`, signals `1.312 W`, Block RAM `0.226 W`, URAM `0.081 W`, DSPs `1.191 W`, PS8 `2.671 W`.
  - Decision: this is the best physical evidence for the fast Search-only `AQ4+w2hgroup+qkvbram` point so far. It is routable and bitgen-complete, but it is not promoted as a passing 300 MHz candidate because timing misses the relaxed floor by `27 ps`, Search is still `5.163 ms > 4 ms`, and DSP pressure is very high.
- `AQ4 + hgroup-local W2 bank cache + Q/K/V/ATTN BRAM + tail2 DSP relief` Search-only probe:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - HLS report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS result: Search-only top II `1,549,428 cycles`, about `5.165 ms` at exact `300 MHz`; this is `384 cycles` slower than the non-tail2 qkv-BRAM point and still above the `4 ms` target.
  - HLS path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,237,566`, `mlp_head` `25,114`.
  - HLS resources: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,220/460,800 = 36%`, LUT `312,155/230,400 = 135%`, URAM `28/96 = 29%`.
  - Vivado project: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_overlay`.
  - Output overlay: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_overlay`.
  - Bitstream and handoff copied to PYNQ as `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only.{bit,hwh}`.
  - Route status: all `208,988` routable nets fully routed; route errors `0`; bitgen completed successfully.
  - Final postroute-physopt timing: WNS `-0.029 ns`, TNS `-0.365 ns`, setup failing endpoints `31`, WHS `0.002 ns`, THS `0.000 ns`. This passes the user-approved `WNS >= -0.500 ns` experimental floor, but remains negative-WNS and is not official clean Vivado signoff.
  - Implemented utilization: CLB LUT `97,759/230,400 = 42.43%`, CLB registers `99,826/460,800 = 21.66%`, Block RAM Tile `227/312 = 72.76%`, RAMB18 `260/624 = 41.67%`, URAM `28/96 = 29.17%`, DSP `1,553/1,728 = 89.87%`.
  - Implemented vectorless power: total on-chip `8.031 W`, dynamic `7.302 W`, device static `0.729 W`; static split is PS `0.104 W`, PL `0.625 W`.
  - Major dynamic buckets: `hgtxr_e2e_axis_top_0` `4.398 W`, `psu` `2.678 W`, `axi_mem` `0.185 W`, `axi_ctrl` `0.018 W`, `axi_dma_out` `0.018 W`, `axi_dma_in` `0.005 W`. Component view: clocks `0.666 W`, CLB logic `1.112 W`, signals `1.416 W`, Block RAM `0.237 W`, URAM `0.081 W`, DSPs `1.120 W`, PS8 `2.671 W`.
  - Comparison versus non-tail2 qkv-BRAM physical probe: Search II regresses by `384 cycles`, but implemented DSP drops `1,641 -> 1,553`, WNS improves `-0.527 ns -> -0.029 ns`, and the design moves from just below the relaxed floor to comfortably inside it.
  - Decision: promote `qkvbram_tail2` as the best Search-only physical closure evidence in this branch, not as the final latency solution. It proves ZCU104 fit, route, bitgen, and the relaxed 300 MHz timing floor for the current fast Search-only body, but Search remains about `5.165 ms > 4 ms`; next work must reduce Search body cycles rather than only relieve DSP pressure.
- Current conclusion:
  - Fastest measured Search-only HLS latency point remains tied between `AQ4 + hgroup-local W2 bank cache` and the Q/K/V/ATTN BRAM variant: Search `5.163 ms`; however, the best Search-only physical closure evidence is now `qkvbram_tail2`, with Search about `5.165 ms`, route errors `0`, bitgen pass, and postroute-physopt WNS `-0.029 ns`.
  - Fastest measured complete Search/Track pair remains `AQ2`: Search `5.699 ms`, Track `0.896 ms`, Hybrid 10:90 `1.376 ms`, because `w2hgroup` has only been run as Search-only so far.
  - Best physical/routed baseline remains `patch32_dtok4`: Search `7.810 ms`, Track `1.121 ms`, routed/bitgen complete under the user-approved WNS floor.
  - Search `4 ms` still requires a larger structural body-cycle reduction or a lower-LUT MLP/attention refactor. `tail2` fixes physical closure and DSP pressure for the fast Search-only point, but it does not solve latency.

### qkvbram_tail2 Follow-up DSE: PatchPar64 and Softmax Exp Partition

- `qkvbram_tail2 + HGTXR_E2E_PATCH_PAR=64`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_patch64_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - CSim result: pass. Search runtime state and strict prefetch-immediate trace passed; vector comparison passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch64_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II `1,561,716 cycles`, about `5.206 ms` at exact `300 MHz`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `270,340`, `global_buffer_load` `12,292`, `controller_run` `1,237,566`, `mlp_head` `25,114`.
  - HLS resources: BRAM_18K `443/624 = 70%`, DSP `1,582/1,728 = 91%`, FF `167,393/460,800 = 36%`, LUT `312,422/230,400 = 135%`, URAM `28/96 = 29%`.
  - Decision: reject. Compared with `qkvbram_tail2`, Search II regresses by `12,288 cycles`; the root cause is `conv_patch_embedding` worsening from `258,052` to `270,340 cycles` due to frame-memory port pressure.
- `qkvbram_tail2 + HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Implementation: added `HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION`; when enabled, `kExpRom` is completely partitioned instead of bound to a 1-port ROM.
  - CSim result: pass. Search runtime state and strict prefetch-immediate trace passed; vector comparison passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II `1,512,564 cycles`, about `5.042 ms` at exact `300 MHz`; this improves `qkvbram_tail2` by `36,864 cycles`, about `0.123 ms`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,200,702`, `mlp_head` `25,114`.
  - Attention core latency improves from `31,543` to `22,327 cycles`; the softmax exp loop instance drops from `260` to `68 cycles`.
  - HLS resources: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,307/460,800 = 36%`, LUT `312,706/230,400 = 135%`, URAM `28/96 = 29%`.
  - Decision: keep as the best Search-only HLS latency point in the `qkvbram_tail2` physical-relief branch, but do not route yet. It still misses Search `4 ms`, and HLS LUT remains far above the ZCU104 budget.
- `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether removing the fused MLP hidden-store path reduces the controlling MLP loop or LUT pressure after `exppart`.
  - CSim result: pass. Search runtime state and strict prefetch-immediate trace passed; vector comparison passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_skiph_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II `1,512,564 cycles`, about `5.042 ms` at exact `300 MHz`, identical to `exppart`.
  - Path breakdown remains unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,200,702`, `mlp_head` `25,114`.
  - MLP unit remains unchanged at `190,022 cycles`; the dominant MLP loop remains `VITIS_LOOP_3091_2` at `150,208 cycles`.
  - HLS resources remain unchanged versus `exppart`: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,307/460,800 = 36%`, LUT `312,706/230,400 = 135%`, URAM `28/96 = 29%`.
  - Decision: reject as a no-op in this branch. The next Search-latency DSE must reduce MLP body cycles, controller loop work, or patch/conv work; hidden-store removal does not move the HLS schedule.
- `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether reading and accumulating two W1 input channels per fused MLP inner step reduces the dominant MLP body loop after `exppart`.
  - Implementation: added `HGTXR_E2E_MLP_W1_C_PAR` with default `1`; the `>1` branch is currently targeted at the fused token-parallel MLP path used by this profile.
  - CSim result: pass. Search runtime state and strict prefetch-immediate trace passed; vector comparison passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II `1,368,180 cycles`, about `4.561 ms` at exact `300 MHz`; this improves `exppart` by `144,384 cycles`, about `0.481 ms`.
  - Remaining gap to Search `4 ms` at exact `300 MHz`: `168,180 cycles`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,056,318`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `103,212 cycles`, MLP unit `153,926 cycles`, 4 runtime blocks.
  - MLP unit improves from `190,022` to `153,926 cycles`; the dominant fused W1 loop in this build is `VITIS_LOOP_3097_2` at `114,112 cycles`.
  - HLS resources: BRAM_18K `449/624 = 71%`, DSP `1,704/1,728 = 98%`, FF `170,564/460,800 = 37%`, LUT `320,789/230,400 = 139%`, URAM `28/96 = 29%`.
  - Decision: keep as the fastest Search-only HLS latency point observed in the current full learned runtime-ROM family, but do not promote or route as-is. It still misses Search `4 ms`, nearly exhausts DSP, and worsens the already-over-budget LUT estimate. The next candidate should preserve the W1 cycle reduction while moving some multipliers or small memories out of the critical DSP/LUT pressure path.
- `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=3`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c3_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether increasing fused MLP W1 input-channel grouping from `2` to `3` can close the remaining Search `4 ms` gap without changing the broader scheduler/dispatcher structure.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_exppart_w1c3_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II regressed to `1,417,332 cycles`, about `4.724 ms` at exact `300 MHz`; this is `49,152 cycles` slower than `w1c2`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,105,470`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `103,212 cycles`, MLP unit `166,214 cycles`, 4 runtime blocks.
  - MLP unit regresses from `153,926` to `166,214 cycles`; the dominant fused W1 loop increases from `114,112` to `126,400 cycles` because the larger grouping worsens the scheduled iteration body.
  - HLS resources: BRAM_18K `449/624 = 71%`, DSP `1,824/1,728 = 105%`, FF `181,636/460,800 = 39%`, LUT `330,657/230,400 = 143%`, URAM `28/96 = 29%`.
  - Decision: reject. `W1_C_PAR=3` is a negative DSE: it fails DSP capacity, remains far over LUT budget, and is slower than `W1_C_PAR=2`. Do not continue to larger W1 channel grouping without a different accumulation schedule or resource-sharing strategy.
- `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2 + HGTXR_E2E_CORE_FABRIC_TAIL_LANES=4`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail4_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether preserving the `w1c2` MLP cycle gain while moving more core tail-lane multiplies to fabric reduces DSP pressure enough for a later physical route attempt.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail4_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II `1,368,180 cycles`, about `4.561 ms` at exact `300 MHz`, unchanged versus `tail2+w1c2`.
  - Path breakdown remains unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,056,318`, `mlp_head` `25,114`.
  - MLP unit remains `153,926 cycles`; the dominant fused W1 loop remains `VITIS_LOOP_3097_2` at `114,112 cycles`.
  - HLS resources shift versus `tail2+w1c2`: BRAM_18K `449 -> 449`, DSP `1,704 -> 1,608`, FF `170,564 -> 172,598`, LUT `320,789 -> 346,133`, URAM `28 -> 28`.
  - Device percentages: BRAM_18K `449/624 = 71%`, DSP `1,608/1,728 = 93%`, FF `172,598/460,800 = 37%`, LUT `346,133/230,400 = 150%`, URAM `28/96 = 29%`.
  - Decision: reject as a promotion candidate. It does reduce DSP by `96`, but it does not improve Search latency and makes the LUT overrun materially worse. The viable next step needs a lower-LUT multiplier/memory restructuring or a body-cycle reduction, not only more tail-lane fabric mapping.
- `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2 + HGTXR_E2E_ATTN_QUERY_PAR=8`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether larger attention query parallelism can reduce the remaining Search latency after the `w1c2` MLP body-cycle improvement.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - Search-only top II `1,324,308 cycles`, about `4.414 ms` at exact `300 MHz`; this improves `w1c2` by `43,872 cycles`, about `0.146 ms`.
  - Remaining gap to Search `4 ms` at exact `300 MHz`: `124,308 cycles`, about `0.414 ms`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,446`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`, 4 runtime blocks.
  - Attention improvement versus `w1c2`: attention unit `103,212 -> 92,244 cycles`; attention core `22,327 -> 11,263 cycles`. MLP is unchanged at `153,926 cycles`, so the remaining bottleneck has shifted back toward MLP/body and fixed front-end work.
  - HLS resources: BRAM_18K `513/624 = 82%`, DSP `1,944/1,728 = 112%`, FF `198,021/460,800 = 42%`, LUT `359,425/230,400 = 156%`, URAM `28/96 = 29%`.
  - Resource delta versus `w1c2`: BRAM_18K `+64`, DSP `+240`, FF `+27,457`, LUT `+38,636`, URAM unchanged.
  - Decision: keep as the fastest Search-only HLS latency point observed so far in this full learned runtime-ROM family, but reject as a promotion/route candidate. It still misses Search `4 ms` and exceeds both DSP and LUT capacity. The next candidate needs to preserve the AQ8 attention gain only if the extra parallelism can be time-multiplexed or resource-shared; otherwise, effort should move to MLP/body-cycle reduction and DSP/LUT pressure relief.
- `AQ8 + qkvbram_tail8 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail8_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether the AQ8 latency gain can be preserved while moving more tail-lane multiply pressure from DSP into fabric.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail8_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.789 ns`.
  - Search-only top II remains `1,324,308 cycles`, about `4.414 ms` at exact `300 MHz`, unchanged versus `AQ8+tail2+w1c2`.
  - Path breakdown remains unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,446`, `mlp_head` `25,114`.
  - Controller breakdown remains unchanged: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`.
  - HLS resources: BRAM_18K `513/624 = 82%`, DSP `1,608/1,728 = 93%`, FF `205,197/460,800 = 44%`, LUT `448,561/230,400 = 194%`, URAM `28/96 = 29%`.
  - Resource delta versus `AQ8+tail2+w1c2`: BRAM_18K unchanged, DSP `-336`, FF `+7,176`, LUT `+89,136`, URAM unchanged.
  - Decision: reject as a promotion/route candidate. This recovers DSP headroom while preserving the fastest Search II, but it makes the LUT overrun much worse (`156% -> 194%`). Tail-lane fabric mapping is therefore not the right standalone pressure-relief mechanism for the current AQ8 design.
- `AQ6 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq6_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether `HGTXR_E2E_ATTN_QUERY_PAR=6` gives a useful latency/resource midpoint between `AQ4+w1c2` and the faster but over-capacity `AQ8+w1c2` point.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq6_hp4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.777 ns`.
  - Search-only top II regressed to `1,484,260 cycles`, about `4.948 ms` at exact `300 MHz`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,172,398`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `130,054-132,232 cycles`, MLP unit `153,926 cycles`.
  - Attention path regression: attention unit worsens versus `AQ4+w1c2` (`103,212 -> 132,232 cycles`) and `AQ8+w1c2` (`92,244 -> 132,232 cycles`); attention core in the AQ6 build is `48,913-51,091 cycles`.
  - HLS resources: BRAM_18K `481/624 = 77%`, DSP `1,824/1,728 = 105%`, FF `194,951/460,800 = 42%`, LUT `348,116/230,400 = 151%`, URAM `28/96 = 29%`.
  - Decision: reject. AQ6 is not a useful interpolation point; it is slower than AQ4/AQ8 in this scheduled design and still exceeds both DSP and LUT capacity. Future attention-query DSE should avoid this non-power-of-two point unless the Q/K/V banking and attention-core schedule are restructured first.
- `AQ8 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2 + HGTXR_E2E_MLP_FUSED_W2_HP_PAR=8`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether increasing W2 hidden-lane parallelism from `4` to `8` can reduce the remaining MLP/body latency after `AQ8+w1c2`.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.789 ns`; estimated Fmax `358.55 MHz`.
  - Search-only top II improves to `1,288,836 cycles`, about `4.296 ms` at exact `300 MHz`.
  - Remaining Search `4 ms` gap: `88,836 cycles`, about `0.296 ms`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `977,118`, `mlp_head` `24,970`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `145,094 cycles`.
  - MLP improvement versus `AQ8+hpar4+w1c2`: MLP unit `153,926 -> 145,094 cycles`, saving `8,832 cycles` per runtime block and `35,328 cycles` over the four Search blocks. MLP head improves only `25,114 -> 24,970 cycles`, so almost all top-level gain comes from the MLP unit.
  - HLS resources: BRAM_18K `513/624 = 82%`, DSP `2,424/1,728 = 140%`, FF `211,693/460,800 = 45%`, LUT `385,749/230,400 = 167%`, URAM `28/96 = 29%`.
  - Resource delta versus `AQ8+hpar4+w1c2`: BRAM_18K unchanged, DSP `+480`, FF `+13,672`, LUT `+26,324`, URAM unchanged.
  - Decision: keep as the fastest Search-only HLS latency point observed so far, but reject as a promotion/route candidate. `hpar8` proves W2 hidden parallelism has real cycle headroom, but the direct implementation is far beyond ZCU104 DSP capacity and still misses Search `4 ms`. A useful successor must time-multiplex or resource-share the `hpar8` W2 gain rather than simply increasing parallel multipliers.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + ACC24`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: keep the fastest direct `hpar8` schedule while reducing accumulator width from the default to `HGTXR_ACC_W=24`, `HGTXR_ACC_I=10` to test DSP/LUT/FF pressure relief.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors; the trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.789 ns`; estimated Fmax `358.55 MHz`.
  - Search-only top II improves slightly from direct `hpar8` `1,288,836` to `1,284,676 cycles`, about `4.282 ms` at exact `300 MHz`.
  - Remaining Search `4 ms` gap: `84,676 cycles`, about `0.282 ms`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `972,938-972,958`, `mlp_head` `24,970`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `91,652 cycles`, MLP unit `144,646 cycles`.
  - Major sub-block resources: attention unit BRAM_18K `271`, DSP `1,038`, FF `82,737`, LUT `181,181`; MLP unit BRAM_18K `46`, DSP `1,214`, FF `78,321`, LUT `130,748`; controller total BRAM_18K `381`, DSP `2,380`, FF `173,510`, LUT `338,058`, URAM `28`.
  - HLS top resources: BRAM_18K `513/624 = 82%`, DSP `2,410/1,728 = 139%`, FF `178,153/460,800 = 38%`, LUT `358,589/230,400 = 155%`, URAM `28/96 = 29%`.
  - Resource delta versus direct `hpar8`: BRAM_18K unchanged, DSP `-14`, FF `-33,540`, LUT `-27,160`, URAM unchanged.
  - Decision: reject as a promotion/route candidate, but keep as a useful pressure-relief data point. `ACC24` improves Search II by `4,160 cycles` and reduces FF/LUT materially, yet DSP and LUT remain far above ZCU104 capacity and Search still misses `4 ms`. The next useful step needs structural resource sharing or reduced learned-operator body work, not only narrower accumulators.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + ACC24 + HGTXR_E2E_FORCE_WIDE_DSP_MUL=0`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_nowide_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether disabling the forced wide-DSP multiply path reduces DSP pressure after the `hpar8_acc24` point.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors; the trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_nowide_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.789 ns`; estimated Fmax `358.55 MHz`.
  - Search-only top II is unchanged versus `hpar8_acc24`: `1,284,676 cycles`, about `4.282 ms` at exact `300 MHz`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `972,938-972,958`, `mlp_head` `24,970`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `91,652 cycles`, MLP unit `144,646 cycles`.
  - HLS top resources: BRAM_18K `513/624 = 82%`, DSP `2,410/1,728 = 139%`, FF `178,245/460,800 = 38%`, LUT `379,140/230,400 = 164%`, URAM `28/96 = 29%`.
  - Resource delta versus `hpar8_acc24`: BRAM_18K unchanged, DSP unchanged, FF `+92`, LUT `+20,551`, URAM unchanged.
  - Decision: reject. Disabling the forced wide-DSP multiply path does not reduce DSP pressure and worsens LUT pressure, while leaving the controlling Search schedule unchanged. The next useful DSE should target controller/global-buffer memory-port pressure or real time-multiplexed sharing, not this macro.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + ACC24 + token-bank4`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_tokbank4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether cyclic partitioning of the token dimension in `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, `gb.attn`, `gb.hidden`, and the top-level `tokens` buffer removes the remaining memory-port bottleneck after `hpar8_acc24`.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors; the trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_acc24_tokbank4_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.800 ns`; estimated Fmax `357.14 MHz`.
  - Search-only top II regressed versus `hpar8_acc24`: `1,284,676 -> 1,294,276 cycles`, about `4.282 -> 4.314 ms` at exact `300 MHz`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `982,538-982,558`, `mlp_head` `24,970`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `94,052 cycles`, MLP unit `144,646 cycles`.
  - Major sub-block resources: attention unit BRAM_18K `271`, DSP `1,038`, FF `81,304`, LUT `183,891`; MLP unit BRAM_18K `46`, DSP `1,214`, FF `78,321`, LUT `130,433`; controller total BRAM_18K `381`, DSP `2,380`, FF `172,077`, LUT `340,453`, URAM `28`.
  - HLS top resources: BRAM_18K `545/624 = 87%`, DSP `2,410/1,728 = 139%`, FF `176,840/460,800 = 38%`, LUT `365,042/230,400 = 158%`, URAM `28/96 = 29%`.
  - Resource delta versus `hpar8_acc24`: BRAM_18K `+32`, DSP unchanged, FF `-1,313`, LUT `+6,453`, URAM unchanged.
  - Decision: reject. Token-dimension cyclic banking does not address the controlling port conflicts; CSynth still reports II=2 memory-port violations inside the attention score path and MLP output path. It worsens Search latency and BRAM/LUT pressure while remaining far beyond DSP/LUT capacity.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8` Track-only pair:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  - Purpose: pair the fastest Search-only `hpar8` branch with a matching Track-only force-mode measurement for 10% Search / 90% Track analysis.
  - CSim result: pass. Track runtime state, TLAST, vector comparison, and strict prefetch trace passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `3.473 ns`; at the requested exact `300 MHz`, Track latency is `229,782 cycles` / `0.765940 ms` and Track II is `229,783 cycles` / `0.765943 ms`. The HLS report's printed `0.798 ms` uses the slower estimated clock, not the target clock.
  - Track path breakdown: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `130,022`, `mlp_head` `6,538`.
  - Track controller breakdown: dispatcher prefetch `6,926 cycles`, attention unit `21,801 cycles`, MLP unit `36,276 cycles`, dispatch loop `13,858 cycles` with trip `2`, body loop `116,162 cycles` with trip `2`.
  - HLS resources: BRAM_18K `513/624 = 82%`, DSP `2,420/1,728 = 140%`, FF `210,928/460,800 = 45%`, LUT `379,171/230,400 = 164%`, URAM `7/96 = 7%`.
  - 10% Search / 90% Track pair estimate using exact `300 MHz`: mean latency `1.118958 ms`, mean II `1.118961 ms`, throughput `893.69 inv/s`, median `0.765940 ms`, P95/P99 and worst-case `4.296117 ms` Search.
  - Decision: this pair is useful for latency-envelope analysis and confirms Track is below `1 ms` in this fastest branch, but it is still not a ZCU104-fit candidate. Search remains above `4 ms`, and both force-mode reports are far beyond DSP/LUT capacity.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_skiph_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether skipping the intermediate hidden-store path can preserve the direct `hpar8` cycle gain while reducing local memory or schedule pressure.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors; the strict trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_skiph_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.789 ns`; estimated Fmax `358.55 MHz`.
  - Search-only top II is unchanged from direct `hpar8`: `1,288,836 cycles`, about `4.296 ms` at exact `300 MHz`.
  - Path breakdown is unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `977,118`, `mlp_head` `24,970`.
  - Controller breakdown is unchanged: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `145,094 cycles`.
  - HLS resources are unchanged from direct `hpar8`: BRAM_18K `513/624 = 82%`, DSP `2,424/1,728 = 140%`, FF `211,693/460,800 = 45%`, LUT `385,749/230,400 = 167%`, URAM `28/96 = 29%`.
  - Decision: reject as an improvement. The skip-hidden-store flag is functionally safe in CSim but is a no-op for the controlling Search schedule and resource pressure in this `hpar8` configuration. The next useful work must target W2 memory-port/banking pressure or a real time-multiplexed/shared W2 datapath.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + HGTXR_E2E_MLP_W2_FABRIC_HP_LANES=2`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2fl2_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: preserve the `hpar8` W2-cycle reduction while moving two W2 hidden-lane multiply lanes from DSP to fabric.
  - CSim result: pass. Search runtime state, strict prefetch-immediate trace, TLAST, and vector comparison all passed with `0` errors.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_w2fl2_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.777 ns`; estimated Fmax `360.10 MHz`.
  - Search-only top II regressed to `16,615,339 cycles`, about `55.384 ms` at exact `300 MHz`.
  - Path breakdown: `axis_read_frame` `16,400`, `conv_patch_embedding` `294,916`, `global_buffer_load` `12,293`, `controller_run` `16,266,742`, `mlp_head` `24,970`.
  - Controller breakdown: dispatcher prefetch `6,928-6,931 cycles`, attention unit max `2,341,596 cycles`, MLP unit max `1,718,150 cycles`.
  - MLP unit loop evidence: `VITIS_LOOP_3114_2` ranges up to `1,678,336 cycles`; the `VITIS_LOOP_3365_27` pipeline also reports a memory-port II violation and final II `2` in the CSynth log.
  - HLS resources: BRAM_18K `515/624 = 82%`, DSP `2,410/1,728 = 139%`, FF `283,889/460,800 = 61%`, LUT `526,262/230,400 = 228%`, URAM `28/96 = 29%`.
  - Resource delta versus direct `hpar8`: BRAM_18K `+2`, DSP `-14`, FF `+72,196`, LUT `+140,513`, URAM unchanged.
  - Decision: reject. `w2fl2` provides almost no DSP relief, destroys LUT/FF feasibility, and collapses the Search latency schedule. This confirms that moving only two W2 lanes to fabric is not a viable resource-sharing strategy for the `hpar8` branch; the next useful successor needs a real time-multiplexed/shared W2 datapath or a smaller operator body, not partial fabric remapping of the already-expanded hpar8 lanes.
- `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + LN parameter cache`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_lncache_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether explicitly enabling LayerNorm parameter caching can remove enough controller/body overhead after the fastest direct `hpar8` Search-only point.
  - CSim result: pass. Runtime state was Search, TLAST/output count passed, vector comparison passed, and the strict prefetch trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp8_lncache_w2hgroup_qkvbram_tail2_exppart_w1c2_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.33 ns`, estimated `2.789 ns`; estimated Fmax `358.55 MHz`.
  - Search-only top II regressed to `1,298,192 cycles`, about `4.327 ms` at exact `300 MHz`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `986,454-986,474`, `mlp_head` `24,970`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,257 cycles`, MLP unit `147,420 cycles`.
  - Resource impact is strongly negative versus direct `hpar8`: BRAM_18K unchanged at `513/624 = 82%`, but DSP increases `2,424 -> 2,924` (`169%`), FF increases `211,693 -> 265,062`, and LUT increases `385,749 -> 450,440` (`195%`).
  - Decision: reject. LN parameter caching does not improve the controlling schedule; it is slower than direct `hpar8` by `9,356 cycles` and substantially worsens DSP/LUT pressure.
- `AQ8 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=4`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether a wider W1 channel grouping can reduce the unchanged MLP/body schedule after the `AQ8+w1c2` point without using the over-capacity direct `hpar8` W2 datapath.
  - CSim result: pass. Search runtime state, TLAST/output count, vector comparison, and strict prefetch-immediate trace passed with `0` errors; the trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail2_exppart_w1c4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.333 ns`, estimated Fmax `358.55 MHz`.
  - Search-only top II is `1,324,308 cycles`, about `4.414 ms` at exact `300 MHz`; top latency is `1,324,287-1,324,307 cycles`.
  - Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,426-1,012,446`, `mlp_head` `25,114`.
  - Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`.
  - HLS resources: BRAM_18K `513/624 = 82%`, DSP `2,184/1,728 = 126%`, FF `212,462/460,800 = 46%`, LUT `375,182/230,400 = 162%`, URAM `28/96 = 29%`.
  - Decision: reject. `w1c4` is functionally safe but does not improve latency over `AQ8+w1c2` (`4.414 ms`) and worsens DSP/LUT pressure. The controlling MLP unit remains `153,926 cycles`, so larger W1 channel grouping is not the right next lever without restructuring the MLP schedule.
- `AQ8 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=4 + HGTXR_E2E_W1_WEIGHT_CACHE_BANKS=16`:
  - Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_w1bank16_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  - Purpose: test whether explicitly banking the W1 dense weight cache can remove the W1 loop memory-port bottleneck seen in the `w1c4` probe.
  - Implementation: added default-off `HGTXR_E2E_W1_WEIGHT_CACHE_BANKS`; when greater than `1`, `w1_weight_cache` receives a cyclic `ARRAY_PARTITION` with the requested factor.
  - CSim result: pass. Search runtime state, TLAST/output count, vector comparison, and strict prefetch-immediate trace passed with `0` errors; the trace reported `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  - CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq8_hp4_w2hgroup_qkvbram_tail2_exppart_w1c4_w1b16_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  - HLS timing estimate: target `3.333 ns`, estimated Fmax `358.55 MHz`.
  - Search-only top II is still `1,324,308 cycles`, about `4.414 ms` at exact `300 MHz`; top latency is `1,324,287-1,324,307 cycles`.
  - Path breakdown remains unchanged versus `w1c4`: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,426-1,012,446`, `mlp_head` `25,114`.
  - Controller breakdown remains unchanged: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`.
  - W1 loop evidence: `VITIS_LOOP_3168_9` remains final II `2` with latency `106 cycles`; the scheduler still reports a limited-port load on `w1_weight_cache`.
  - HLS resources worsen materially: BRAM_18K `707/624 = 113%`, DSP `2,184/1,728 = 126%`, FF `212,833/460,800 = 46%`, LUT `376,481/230,400 = 163%`, URAM `28/96 = 29%`.
  - Decision: reject. The banking pragma as applied here does not remove the W1 memory-port bottleneck, does not improve Search latency, and increases BRAM by `194` BRAM_18K versus `w1c4`. The next W1 attempt needs a different storage layout or access schedule, not only a larger cyclic partition factor.
