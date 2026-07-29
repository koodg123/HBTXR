> **작성** 2026-07-05 · **갱신** 2026-07-15
> **상태** active — 내용은 2026-07-15 구 트리에서 이어받음, 재구성 진행분을 여기에 이어 씁니다
> **소유** hardware

# HGTXR Hardware Progress

Date: 2026-06-16

Latest Continuation: Selected Path Execution Audit.

## 2026-07-05 Documentation And Commit Preparation
- [x] Rechecked repository topology: root, `software/`, and `hardware` all
  resolve to `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`; commit target is the
  root repository.
- [x] Confirmed active branch `kjm26/feat/capture-hgtxr-current-state`, not
  `main` or `master`.
- [x] Added comprehensive continuation document:
  `hardware/docs/track/SESSION_PROGRESS_AND_CONVERSATION_2026_07_05.md`.
- [x] Updated `hardware/docs/track/CONVERSATION.md` and
  `hardware/docs/track/log.md` with the 2026-07-05 commit-preparation summary.
- [x] Confirmed ignored large generated paths are outside the current status
  candidate set: `hardware/generated/` and `software/.venv/`.
- [x] Commit-preparation documentation is complete for this snapshot; the actual
  commit hash is reported in the final response after `git commit` completes.

## 2026-06-30 AQ2 Search/Track Measurement
- [x] AQ2 Search/Track HLS latency, initial interval, DMA bandwidth, throughput, and deterministic percentile estimates documented in `hardware/docs/track/AQ2_SEARCH_TRACK_MEASUREMENT_2026_06_30.md`.
- [x] AQ2 requested metric checklist saved in `hardware/docs/track/AQ2_SEARCH_TRACK_METRIC_CHECKLIST_2026_07_01.md`, covering DMA ideal/effective bandwidth, mean/median/min/max/P95/P99 latency, initial interval, GOPS, AXIS width, major-block resources, and major-block power limits.
- [x] AQ2 Search-only Vivado overlay implemented at 300 MHz with bitgen pass, route errors `0`, WNS `0.000 ns`, and strict timing pass.
- [x] AQ2 Track-only Vivado overlay implemented at 300 MHz with bitgen pass, route errors `0`, WNS `-0.017 ns`, strict timing fail, and user-tolerance pass under the approved `WNS >= -0.500 ns` continuation threshold.
- [x] AQ2 implemented resource utilization captured by mode: Search CLB LUT `77,494/230,400`, BRAM tile `191.5/312`, URAM `92/96`, DSP `1,257/1,728`; Track CLB LUT `80,565/230,400`, BRAM tile `185.5/312`, URAM `64/96`, DSP `1,270/1,728`.
- [x] AQ2 vector-less Vivado power captured by mode: Search total `7.032 W`, dynamic `6.304 W`, static `0.728 W`; Track total `6.777 W`, dynamic `6.054 W`, static `0.723 W`.
- [ ] AQ2 board-runtime histogram remains pending, so mean/median/min/max/P95/P99 are currently HLS deterministic estimates rather than repeated board measurements.
- [ ] AQ2 external DDR/sensor I/O rail power and internal conv/attention/MLP/head dynamic power split remain pending because the current Vivado report only separates top-level BD hierarchy.

## 2026-07-01 PatchPar64 Search-only DSE
- [x] Added Search-only profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
- [x] Kept the previous fastest Search-only settings `AQ8+hpar8+ACC24+w1c2+mtok4/dtok4+qkvbram_tail2+exppart`; changed only `HGTXR_E2E_PATCH_PAR=32 -> 64`.
- [x] CSim passed for Search with runtime state `0`, TLAST, vector comparison, and strict prefetch-immediate trace `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- [x] CSynth completed at the 300 MHz HLS target with estimated clock `2.789 ns` / `358.55 MHz`.
- [x] Search top latency is `1,296,943-1,296,963 cycles` and Search initial interval is `1,296,944-1,296,964 cycles`; at exact `300 MHz`, max latency is `4.323210 ms` and max II is `4.323213 ms`.
- [x] Major path latency: AXIS read `16,386 cycles`, Conv/Patch embedding `270,340 cycles`, global buffer load `12,292 cycles`, controller `972,938-972,958 cycles`, MLP head `24,970 cycles`.
- [x] Resource estimate remains non-promotable: BRAM_18K `513/624 = 82%`, DSP `2,408/1,728 = 139%`, FF `179,024/460,800 = 38%`, LUT `358,606/230,400 = 155%`, URAM `28/96 = 29%`.
- [ ] Promotion rejected: PatchPar64 worsens Search II versus the previous fastest `hpar8_acc24` point by `12,288 cycles` (`0.040960 ms`) because the patch embedding loop still hits a memory-port II=2 limit and Conv/Patch latency rises from `258,052` to `270,340 cycles`.

## 2026-07-01 HPar16 Search-only DSE
- [x] Added Search-only profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
- [x] Kept the previous fastest `hpar8_acc24` Search-only settings and changed only the fused MLP hidden-lane parallelism `HGTXR_E2E_MLP_FUSED_W2_HP_PAR=8 -> 16`.
- [x] CSim passed for Search with runtime state `0`, output count `6`, TLAST, vector comparison, and strict prefetch-immediate trace `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- [x] CSynth completed at the 300 MHz HLS target with estimated clock `2.795 ns` / `357.78 MHz`.
- [x] Search top latency is `2,229,295-2,229,315 cycles` and Search initial interval is `2,229,296-2,229,316 cycles`; at exact `300 MHz`, max latency is `7.431050 ms` and max II is `7.431053 ms`.
- [x] Major path latency: AXIS read `16,386 cycles`, Conv/Patch embedding `258,052 cycles`, global buffer load `12,292 cycles`, controller `1,917,578-1,917,598 cycles`, MLP head `24,970 cycles`.
- [x] Controller breakdown: dispatcher prefetch `6,928-6,933 cycles`, shared attention `91,652 cycles`, shared MLP `380,806 cycles`, dispatch loop `27,724-27,744 cycles`, body loop `1,889,852 cycles`.
- [x] Root cause captured: `kGeluRom` is bound as `ROM_1P_LUTRAM`, causing the MLP activation loop `VITIS_LOOP_3279_18_VITIS_LOOP_3281_19` to miss the target and settle at final II `64`.
- [x] Resource estimate is non-promotable and worse than `hpar8_acc24`: BRAM_18K `513/624 = 82%`, DSP `3,370/1,728 = 195%`, FF `209,907/460,800 = 45%`, LUT `401,685/230,400 = 174%`, URAM `28/96 = 29%`.
- [ ] Promotion rejected: direct hpar16 worsens Search II by `944,640 cycles` (`3.148800 ms`) versus `hpar8_acc24`. The next useful Search DSE is GELU ROM replication/partitioning or a true shared/time-multiplexed W2 datapath, not more direct hidden-lane replication.

## Active 300 MHz Runtime Experiment Status
- [x] PL clock target fixed at `300 MHz` for the active dense combined runtime profile.
- [x] Experiment continuation threshold set to routed WNS `>= -0.500 ns`; official clean signoff remains WNS `>= 0.000 ns`.
- [x] Active combined `prefetchall4_300` Vivado run passes the experiment threshold with post-route physopt WNS `-0.236 ns`, route errors `0`, hold clean, and bitgen pass.
- [x] HLS/C TB evidence for combined Search/Track is clean: expected outputs, runtime states, TLAST/vector comparison, and strict scheduler immediate-start trace all pass.
- [x] Verilog snapshot build reached `Built simulation snapshot hgtxr_e2e_axis_top`.
- [x] Minimal standalone XSIM snapshot smoke added; it builds `tb_snapshot` but fails before printing `XSIM_SMOKE_PASS`, so the current blocker is classified as host XSIM runtime `blocked-xsim-runtime`.
- [x] Current 300 MHz goal-status report generated: `hardware/generated/signoff/prefetchall4_300_goal_status_2026_06_29.{json,md}`.
- [x] Current 300 MHz structural objective-contract audit generated: `hardware/generated/signoff/prefetchall4_300_contract_audit_2026_06_29.{json,md}`, status `pass`, `16/16` checks.
- [x] Goal-status report now includes the contract audit gate, so the active 300 MHz status covers timing/resource/power plus runtime scheduler, ROM parameter/nonlinear paths, prefetch-all4, full Transformer module generation, and CSim Search/Track expected-output evidence.
- [x] Prefetch-all4 and ROM-compute board validators now hard-enforce Search `<=4.0 ms` and Track `<=1.0 ms` accelerator latency targets.
- [x] Goal-status report now computes per-mode board latency/DMA plus synthetic Search `10%` / Track `90%` hybrid latency, throughput, worst-case latency, and DMA means once canonical board JSONs are imported.
- [x] Measured interleaved Search `10%` / Track `90%` PYNQ runner added for the active prefetch-all4 bitstream.
- [x] Hybrid 10/90 transfer bundle generated and validated: `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle.tar.gz`, sha256 `a902fb80e4e453d625e6f245fdc6824da82fda9abca1452fb3ef83724714024e`.
- [x] Hybrid 10/90 ZCU104 remote dry-run plan generated with errors `0`.
- [x] Goal-status report now reads canonical hybrid JSON `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json` and prefers measured interleaved data over synthetic per-mode aggregation when present.
- [ ] Verilog RTL cosim pass is still blocked by XSIM snapshot launch failure on generated waveform-load options (`-autoloadwcfg` / explicit `-view *.wcfg`); current HGTXR diagnosis status is `blocked-xsim-launch`.
- [ ] Board Search/Track JSON, hybrid 10/90 JSON, p95/p99 latency, measured DMA bandwidth, and measured board power are still pending.

## 2026-06-29 DSP/BRAM Pressure Relief Successor
- [x] Added non-default successor profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16`.
- [x] Kept the active routed `prefetchall4_300` baseline unchanged; the new profile is a pressure-relief probe, not the promoted baseline.
- [x] Added `HGTXR_E2E_NONLINEAR_ROM_LUTRAM` so tiny nonlinear ROMs can move from BRAM to LUTRAM when the profile requests it.
- [x] Moved two tail core lanes to fabric multiplication in the successor profile using `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=2` and `HGTXR_E2E_CORE_LANE_CT_SWITCH=1`.
- [x] CSim passed for Search/Track outputs, runtime states, TLAST/vector comparison, and strict prefetch trace.
- [x] CSynth passed at target `3.333 ns`; estimated clock is `2.777 ns` / `360.10 MHz`.
- [x] HLS estimate reduced BRAM_18K `386 -> 384` and DSP `1848 -> 1838` versus active `prefetchall4_300`.
- [x] Generated RTL shows `kGeluRom` as `ROM_1P_LUTRAM_1R`; large QKV/MLP caches remain BRAM/URAM by design.
- [x] Vivado package/route/bitgen completed; route errors `0`, fully routed nets `176,835/176,835`, `.bit/.hwh` exported to the overlay directory.
- [x] Implemented pressure relief confirmed: CLB LUT `72,960/230,400 = 31.67%`, registers `64,095/460,800 = 13.91%`, Block RAM Tile `215/312 = 68.91%`, URAM `76/96 = 79.17%`, DSP `1,669/1,728 = 96.59%`.
- [x] Implemented vectorless power captured: total `8.194 W`, dynamic `7.457 W`, device static `0.737 W`, PL static `0.632 W`.
- [ ] Timing promotion blocked: final post-route physopt WNS `-0.588 ns`, TNS `-2779.720 ns`, WHS `0.001 ns`, THS `0.000 ns`; this misses the user-approved continuation floor `-0.500 ns` and is worse than active `prefetchall4_300` WNS `-0.236 ns`.

## 2026-06-29 DSP/BRAM Pressure Relief DSP-Pipeline Successor
- [x] Added follow-up profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16`.
- [x] Kept the LUTRAM nonlinear ROM path and two-lane fabric tail multiply from `lutrom_tail2`; added `HGTXR_E2E_DSP_MUL_LATENCY=4` to give Vivado more DSP register movement room.
- [x] CSim passed for Search/Track expected outputs, runtime states, vector comparison, and strict prefetch trace.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.777 ns` / `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,810,934` cycles / `16.035 ms`.
- [x] CSynth resources: BRAM_18K `384/624 = 61%`, DSP `1,838/1,728 = 106%`, FF `244,988/460,800 = 53%`, LUT `402,794/230,400 = 174%`, URAM `76/96 = 79%`.
- [x] Vivado route/post-route physopt/bitgen completed at `clk_pl_0 = 300.030 MHz`; final WNS `-0.372 ns`, TNS `-1326.235 ns`, WHS `0.005 ns`, THS `0.000 ns`, route errors `0`, bitgen pass.
- [x] Implemented utilization confirms BRAM pressure relief versus active `prefetchall4_300`: CLB LUT `71,232/230,400 = 30.92%`, registers `63,821/460,800 = 13.85%`, Block RAM Tile `201/312 = 64.42%`, URAM `76/96 = 79.17%`, DSP `1,713/1,728 = 99.13%`, LUT as distributed RAM `1,232/101,760 = 1.21%`.
- [x] Implemented vectorless power captured: total `8.396 W`, dynamic `7.658 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`; component buckets include Block RAM `0.220 W`, URAM `0.222 W`, DSPs `1.442 W`.
- [x] Experiment acceptance: passes the user-approved 300 MHz continuation rule because WNS `-0.372 ns >= -0.500 ns`, hold is clean, routing errors are `0`, and bitgen completed.
- [ ] Official clean timing signoff remains open because WNS is still negative and Vivado reports `Timing constraints are not met`.

## 2026-06-29 Targeted Token LUTRAM Pressure Relief Follow-up
- [x] Added follow-up profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16`.
- [x] Kept the balanced `tail2_dsppipe4` compute policy: nonlinear LUTROM, two fabric tail multiply lanes, DSP multiply latency `4`, strict prefetch-all4 tracing, and 300 MHz HLS target.
- [x] Moved only the small/high-bank-count token memories to LUTRAM with `HGTXR_E2E_LUTRAM_FRAME_TOKENS=1` and `HGTXR_E2E_LUTRAM_GB_TOKENS=1`; norm/Q/K/V/attention buffers remain out of this targeted profile.
- [x] Fixed the generated AXIS payload path by inlining the state writer and emitting raw `ap_fixed` bits under synthesis in `hgtxr_data_to_axis()`.
- [x] CSim passed for Search/Track expected outputs, runtime states, vector comparison, and strict prefetch trace.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.777 ns` / `360.10 MHz`; top latency min `26,094` cycles / `86.971 us`, max `4,809,404` cycles / `16.030 ms`.
- [x] CSynth resources show the targeted memory move: BRAM_18K `336/624 = 53%`, DSP `1,838/1,728 = 106%`, FF `243,190/460,800 = 52%`, LUT `409,665/230,400 = 177%`, URAM `76/96 = 79%`.
- [x] Compared with `lutrom_tail2_dsppipe4`, HLS BRAM_18K drops `384 -> 336` while DSP stays `1,838`; the cost is LUT `402,794 -> 409,665`.
- [x] Generated RTL no longer ties output `TDATA` to `256'd0`; it drives `axis_out_TDATA_int_regslice` from `zext_ln437*` wires derived from `out_state_*` registers.
- [x] Vivado route/bitgen completed: route errors `0`, fully routed nets `183,782/183,782`, bitgen pass, implemented CLB LUT `82,306/230,400 = 35.72%`, LUT as Distributed RAM `8,912`, Block RAM Tile `184/312 = 58.97%`, URAM `76/96 = 79.17%`, DSP `1,713/1,728 = 99.13%`.
- [x] Promotion rejected for the current 300 MHz experiment floor: final post-route physopt WNS `-0.560 ns`, TNS `-6463.610 ns`, WHS `0.005 ns`, THS `0.000 ns`; this misses the approved `WNS >= -0.500 ns` floor by `0.060 ns`.
- [ ] Full RTL TV equality is still pending for this profile; current same-output evidence is CSim plus structural RTL inspection only.

## 2026-06-29 DSP/BRAM Pressure Relief Tail4 Probe
- [x] Added more aggressive follow-up profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16`.
- [x] Kept nonlinear ROM LUTRAM and DSP latency `4`; moved four core tail lanes to fabric multiply with `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=4`.
- [x] CSim passed for Search/Track expected outputs, runtime states, vector comparison, and strict prefetch trace.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.777 ns` / `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,810,934` cycles / `16.035 ms`.
- [x] CSynth resources: BRAM_18K `384/624 = 61%`, DSP `1,818/1,728 = 105%`, FF `245,528/460,800 = 53%`, LUT `407,722/230,400 = 176%`, URAM `76/96 = 79%`.
- [x] Vivado route/post-route physopt/bitgen completed at `clk_pl_0 = 300.030 MHz`; route errors `0`, WHS `0.009 ns`, THS `0.000 ns`.
- [x] Implemented utilization confirms additional DSP relief: CLB LUT `70,766/230,400 = 30.71%`, registers `63,836/460,800 = 13.85%`, Block RAM Tile `201/312 = 64.42%`, URAM `76/96 = 79.17%`, DSP `1,693/1,728 = 97.97%`, LUT as distributed RAM `1,232/101,760 = 1.21%`.
- [x] Implemented vectorless power captured: total `8.329 W`, dynamic `7.592 W`, device static `0.738 W`, PS static `0.104 W`, PL static `0.633 W`; component buckets include Block RAM `0.220 W`, URAM `0.227 W`, DSPs `1.415 W`.
- [ ] Promotion rejected: WNS `-0.573 ns` misses the user-approved `-0.500 ns` continuation floor, so `lutrom_tail2_dsppipe4` remains the recommended pressure-relief point.

## 2026-06-29 DSP/BRAM Pressure Relief Compute-Keep Probe
- [x] Added `compute_keep` guard mode in `package_e2e_axis_ip.tcl`: compute modules receive `keep_hierarchy` but no `dont_touch`.
- [x] Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16`.
- [x] CSim passed for Search/Track expected outputs, runtime states, vector comparison, and strict prefetch trace.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.777 ns` / `360.10 MHz`; top latency min `26,088` cycles / `86.951 us`, max `4,810,934` cycles / `16.035 ms`.
- [x] CSynth resources stayed full-design scale: BRAM_18K `384/624 = 61%`, DSP `1,838/1,728 = 106%`, FF `244,988/460,800 = 53%`, LUT `402,794/230,400 = 174%`, URAM `76/96 = 79%`.
- [x] Vivado route/post-route physopt/bitgen completed and timing was clean: WNS `0.064 ns`, TNS `0.000 ns`, WHS `0.009 ns`, THS `0.000 ns`, route errors `0`.
- [ ] Promotion rejected: final implemented utilization collapsed to a pruned shell, with CLB LUT `7,901/230,400 = 3.43%`, registers `11,546/460,800 = 2.51%`, Block RAM Tile `5/312 = 1.60%`, URAM `0/96`, and DSP `1/1,728 = 0.06%`.
- [ ] Resource-preservation gate remains required for future timing probes: positive WNS is not acceptable unless full learned compute resources remain physically present.

## Plan 대비 진행상황
- [x] Hardware directory cleanup phase 1 completed: target taxonomy created,
  root HLS/Vivado logs moved under `generated/logs/`, empty accidental root
  files moved under `archive/migration-logs/`, and compatibility-path migration
  policy documented in `docs/architecture/DIRECTORY_LAYOUT.md` and
  `docs/track/CLEANUP_2026_06_17.md`.
- [x] Third-goal progress/results/future-experiment/checklist documentation package added:
  `docs/THIRD_GOAL_PROGRESS_EXPERIMENT_REPORT_2026_06_16.md`,
  `docs/THIRD_GOAL_FUTURE_EXPERIMENTS_2026_06_16.md`,
  `docs/track/THIRD_GOAL_ACHIEVEMENT_CHECKLIST_2026_06_16.md`.
- [x] ViT accelerator papers/codebases collected and analyzed.
- [x] Codebase analysis expanded to `SRC_CASE_MODULE_GUIDE.md`-style detail.
- [x] Paper analysis expanded with problems, method, algorithm, architecture, experiments, datasets, results, and options.
- [x] VREF experiments added to third-goal plan.
- [x] C3b remains selected board-smoke candidate.
- [x] E2E DSP/URAM/LUTRAM resource policy audit rerun for 2026-06-15.
- [x] VREF P0 execution plan added.
- [x] Updated third-goal requirement map added for 2026-06-16.
- [x] VREF-P0-01 PoT scale readiness audit tool added.
- [x] Spark explorer VREF-P0-01 touchpoint review reflected in execution plan.
- [x] VREF-P0-01 PoT scale candidate sweep added.
- [x] VREF-P0-01 SW candidate sweep generated: `3` specs, `45` candidates, `0` fail.
- [x] VREF-P0-02 buffer lifetime/resource placement audit tool added.
- [x] VREF-P0-02 static audit generated: `36/36 pass`.
- [x] C3b baseline protection checklist added.
- [x] C3b successor gate generated: `30/32 pass`, `0 fail`, `2 pending`.
- [ ] C3b AXIS/DMA physical smoke JSON captured.
- [ ] XR-VITs exact source or approved replacement policy resolved.
- [x] `VREF-P0-01` P2-ViT PoT readiness and SW candidate sweep completed.
- [x] `VREF-P0-01` `softmax_input_x2` non-current scale successor golden header regeneration and E2E AXIS CSim completed.
- [x] `VREF-P0-01` `softmax_input_x2` E2E AXIS csynth completed as an isolated `_csynth` successor project.
- [x] `VREF-P0-01` `softmax_input_x2` `dsp_mixed_stream` HLS resource projection cleared C3b thresholds.
- [x] `VREF-P0-01` `softmax_input_x2` `dsp_mixed_stream` HLS IP package generated and validated.
- [x] `VREF-P0-01` `softmax_input_x2` `dsp_mixed_stream` Vivado overlay routed and bit/hwh generated.
- [x] `VREF-P0-01` `softmax_input_x2` routed timing cleared C3b protection threshold: WNS `4.497 ns` >= `4.415 ns`.
- [x] `VREF-P0-01` `softmax_input_x2` ZCU104 PYNQ smoke bundle and session runbook generated.
- [x] `VREF-P0-01` `softmax_input_x2` PYNQ smoke bundle manifest/tar validation passed.
- [x] `VREF-P0-01` `softmax_input_x2` SSH/SCP remote smoke dry-run plan generated.
- [x] `VREF-P0-01` `softmax_input_x2` physical-smoke result receiver added to preflight and successor projection.
- [x] Final unblock closeout packet refreshed with VREF successor gate as optional promotion evidence.
- [x] Final unblock closeout packet supports selectable VREF-required hard-gate policy.
- [x] Final signoff runner supports VREF successor smoke import/remote execution and VREF-required policy forwarding.
- [x] Third-goal source audit generated for plan/progress/HANDOVER/signoff/reference sources: required `46`, sources `159`, missing `0`.
- [x] HG-PIPE operator audit generated for LayerNorm/GeLU/Softmax/Quantization: `97/97` ref checks pass, `5899008` samples checked.
- [x] Req2 spec/sub-agent gate audit generated: spec-kit unavailable, manual Spec fallback recorded, Spark-first/GPT5.5 fallback recorded.
- [x] Current third-goal audit generated for requirements 0-11: `blocked-external`, reflected `10`, partial `1`, blocked `1`.
- [x] XR-VITs gate audit generated: exact `/home/kjm26/project/PRJXR/XR-VITs` missing, `XR_Accel` candidate ready but approval required.
- [x] XR-VITs replacement policy integrity hardened: generated policies bind approval to candidate audit fingerprint, recommendation snapshot, and policy fingerprint; legacy/drifted policies no longer clear final signoff.
- [x] Choice record added: `docs/CHOICE.md` captures Req11 X1/X2/X3 and C3b C1/C2 options.
- [x] C3b physical-smoke gate audit generated: bundle/session ready, canonical board JSON missing.
- [x] C3b physical-smoke final gate hardened to canonical-only result evidence; noncanonical bundle/remote JSON no longer clears final signoff before import.
- [x] C3b smoke import provenance added: source sha256, canonical dest check, copy state, payload summary.
- [x] C3b smoke candidate discovery dry-run import command added; metadata noise skipped.
- [x] Third-goal unblock checklist updated to require C3b dry-run import before active import.
- [x] C3b and VREF ZCU104 smoke session runbooks updated to runner-based dry-run/active import flow.
- [x] Current gate recheck completed: `36` tests passed, smoke-session/signoff JSON parsed, neutral preflight `ok=76 warn=6 fail=0`, final-signoff remains blocked only by external C3b/XR-VITs gates.
- [x] Generic PYNQ smoke candidate discovery added for C3b and VREF successor; final-signoff runner now records `vref_smoke_discovery_status`, current audit tracks legacy and generic discovery gates.
- [x] Third-goal source/current audit refreshed after generic discovery linkage: source required `52`, sources `171`, missing `0`; neutral preflight `ok=79 warn=6 fail=0`, final-signoff expected-blocked `ok=79 warn=4 fail=2`.
- [x] Final unblock route surfaced as required evidence: closure readiness, unblock intake, final unblock command card, and XR-VITs unblock packet; source required `60`, sources `173`, missing `0`.
- [x] Final signoff runner re-run after unblock-route surfacing: evidence manifest `50/50` pass, source audit `60/173` pass, current audit remains `blocked-external`.
- [x] Final operator handoff/candidate audit promoted to source/current audit evidence: source required `66`, sources `175`, missing `0`; current audit remains `blocked-external`.
- [x] Final evidence manifest audit linkage hardened and runner refreshed: manifest `68/68` pass, source audit required `76`, sources `201`, missing `0`; current audit remains `blocked-external`.
- [x] Req5 Q4/Q8 SW-HW match audit added and reflected: audit `pass`, fail `0`; manifest `68/68` pass, source audit required `76`, sources `201`, missing `0`; current audit `blocked-external`, reflected `10`, partial `1`, blocked `1`.
- [x] Req6 parameterization audit added and reflected: tiling/parallelism/bus/bit/buffer/FIFO knobs traced through config, derived params, HLS pragmas, Tcl overrides, sweep matrix, HLS-legal macro relationships, E2E compile-time `static_assert` guards, PAR16/PAR32 Tcl override support, and PAR16/PAR32 promotion-policy checks; checks `71/71`.
- [ ] `VREF-P0-01` `softmax_input_x2` full C3b-replacement projection cleared with physical smoke.
- [x] `VREF-P0-02` ME-ViT static buffer audit completed.
- [x] `VREF-P0-02` QKV weight-cache URAM successor CSim and HLS csynth completed: BRAM_18K `114`, DSP `128`, LUT `43236`, URAM `40`, latency `498485`.
- [x] `VREF-P0-02` QKV URAM PYNQ/remote/import plumbing added for variant `vref-p0-softmax-input-x2-qkv-uram`; bundle/session artifacts generated.
- [x] `VREF-P0-02` QKV weight-cache URAM HLS IP package generated and recorded.
- [x] `VREF-P0-02` QKV weight-cache URAM routed overlay timing captured: setup slack `4.517 ns`, hold slack `0.010 ns`, route errors `0`.
- [x] `VREF-P0-02` QKV URAM bundle/session/physical-smoke checks integrated into third-goal preflight.
- [x] `VREF-P0-02` QKV URAM import/remote/discovery fields integrated into final signoff runner; current `qkv_uram_smoke_discovery_status` is `missing` until board JSON is captured.
- [x] `VREF-P0-02` QKV URAM discovery/runner fields integrated into source/current audit tracking: source required `76`, sources `201`, missing `0`.
- [x] `VREF-P0-02`, Req5, Req6, buffer-lifetime, and HG-PIPE operator evidence integrated into final source/current/manifest tracking: source required `76`, sources `201`, missing `0`; final manifest `68/68`.
- [x] `VREF-P0-02` QKV URAM final-manifest consistency hardened: QKV resource/timing/route checks are all `pass`; manifest consistency count `47`; bundle validation `pass`, fail `0`.
- [x] Final unblock command card hardened: U0 local blocker snapshot commands now include required JSON/Markdown outputs; optional U5 QKV URAM successor smoke/promotion path added without changing default final blockers.
- [x] Final unblock closeout packet hardened: QKV URAM successor gate and U5 operator commands now appear in closeout packet and validation checks; validation `40/40`, default blockers unchanged.
- [x] Final evidence/current audit hardened for QKV U5: manifest consistency checks now `47`, including conditional required-smoke checks; current audit records `has_qkv_uram_u5=True`.
- [x] Final signoff safety schema normalized: command card and closeout packet now expose both `executes_commands=False` and `executes_network=False`; bundle safety checks pass, fail `0`.
- [x] Final signoff schema/date provenance clarified: current audit maps blocker names to exact external input paths; final manifest records canonical `2026_06_10` plus current-audit `2026_06_16` artifact tags; source audit now required `76`, sources `201`, missing `0`.
- [x] VREF-P0-02 buffer lifetime/resource-placement audit promoted to required final evidence: final manifest `68/68`, consistency checks `83`, bundle validation `pass`, source audit sources `201`.
- [x] C3b canonical physical-smoke gate tightened: preflight rejects noncanonical generated result JSON for final signoff, and C3b gate audit records canonical validation JSON status.
- [x] C3b/VREF/QKV smoke import reports now revalidate destination JSON after copy, record destination SHA256/source match, and expose `would_clear_current_gate` so dry-runs or custom destinations cannot be mistaken for current gate closure.
- [x] XR-VITs gate audit/readiness/preflight now expose and enforce active policy integrity; final runner remains expected-blocked only because no approved fingerprint-bound policy exists.
- [x] XR-VITs fingerprint-bound policy requirement propagated to final unblock command card, closeout packet, operator handoff, closeout validator, handoff validator, and final bundle validation; closeout validation now `49/49`, fail `0`.
- [x] Final signoff bundle now explicitly requires PYNQ smoke overlay-provenance manifest checks: prefix contract, bit/hwh basename check, and `require_paths` gate.
- [x] Current third-goal audit now exposes XR-VITs policy integrity propagation: command-card contract, operator-handoff validation status, policy existence, and policy-check count `9`.
- [x] Completion audit now exposes XR-VITs policy integrity and external blocker path mapping: `consistent=True`, policy check count `9`, validation `pending-policy-creation`.
- [x] Completion audit Req0 now verifies current planning/handover continuity evidence: `docs/track/HANDOVER.md` and `docs/track/log.md` are checked alongside progress, choice, validation, and legacy E2E handover docs.
- [x] Requirements trace now makes XR-VITs blocker closure depend on fingerprint-bound replacement-policy integrity: required fields complete `True`, legacy/drifted policy clears final signoff `False`.
- [x] Final evidence manifest now validates requirements-trace XR-VITs policy integrity as consistency gates: Req11 blocked, required fields complete, legacy/drifted policy rejected; consistency checks `65`, failed `0`.
- [x] Final signoff bundle validator now requires the manifest trace-policy gates to be present/pass: bundle validation `pass`, checks `74`, fail `0`.
- [x] XR-VITs replacement policy preview generated as non-active review evidence: preview status `pass`, `preview_only=True`, `active_policy_written=False`, integrity `pass`; no active replacement policy was written.
- [x] Final evidence manifest now validates XR-VITs policy preview evidence as required no-side-effect gates: final manifest `68/68` pass, consistency checks `83`, failed `0`.
- [x] Final operator handoff validator requires the latest final evidence contract required-count threshold `>=86`, consistency count threshold `>=347`, failed consistency `[]`, live manifest equality, live XR-VITs reference-resolution equality, live final-unblock-intake blocker/candidate-state equality, Req5 live-manifest checks, cyclic resource-policy live-manifest checks, C3b LUT/latency/WNS threshold gates, final-runner summary count checks, mirror-integrity checks, source-audit freshness checks by name, final-unblock-intake operator-plan checks by name, direct live source-audit JSON status/count validation, direct live resource-policy JSON core-check validation, direct live spec-plan JSON core-check validation, direct live final-runner JSON validation, direct live Req6 validation, direct live current-audit validation, direct live closeout-validation status/count/core-check/safety validation, direct live closeout-packet status/blocker/board/C3b/XR/QKV/safety validation, and direct live C3b physical-smoke/XR-VITs gate audit status/path/safety validation: validation target `131/131`.
- [x] Final signoff bundle validator also directly validates the live closeout packet, so stale `final_unblock_closeout_packet_2026_06_10.json` status drift or QKV command drift fails even when final-manifest and closeout-validation freshness checks remain present.
- [x] Updated third-goal final evidence freshness gates are reflected: final manifest `86/86` pass, consistency checks `347`, failed `0`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; operator handoff validation `131/131`; bundle validation `152/152`; runner remains expected-blocked only on C3b physical smoke and XR-VITs approval/source.
- [x] Final blocker closure readiness now emits an operator unblock plan for the two external blockers, with required inputs, dry-run templates, active-run state, and no-side-effect safety; final evidence consistency checks now gate that plan and cross-check final unblock intake `next_inputs`/`operator_sequence` against it.
- [x] Completion/requirements trace status aligned with current evidence split: completion audit remains externally blocked; requirements trace keeps only Req4 partial and Req11 blocked.
- [x] HG-PIPE operator audit strengthened with deterministic contract property checks: reference checks `97/97`, samples `5899008`, property checks `211/211`, fail `0`.
- [x] HG-PIPE operator audit promoted into final evidence manifest/signoff contract: final manifest `68/68` pass, consistency checks `83`, failed `0`; source audit required `76`, sources `201`, missing `0`; handoff validator `44/44`, bundle validator `74/74`; runner remains expected-blocked only on C3b physical smoke and XR-VITs approval/source.
- [x] VREF-P0-01 PoT scale audit/sweep promoted into final evidence manifest/signoff contract: audit `14/14`, sweep specs `3`, candidates `45`, fail `0`; final manifest `72/72` pass, consistency checks `88`, failed `0`; source audit required `80`, sources `205`, missing `0`.
- [x] Spec/plan conformance audit now checks hardware-local current docs against live final manifest/source/current-audit counts; current target is final manifest `86/86`, consistency checks `347`, source audit required `86`, sources `224`, missing `0`.
- [x] Final signoff runner now refreshes manifest-dependent artifacts in one pass after manifest generation: requirements trace, operator handoff, handoff validation, bundle validation, and final evidence manifest are regenerated to avoid stale evidence-contract validation after required-count changes.
- [x] RMU/SMU small token `score`/`prob` buffers now use LUTRAM and are tracked by the VREF-P0-02 buffer lifetime/resource-placement audit; QKV successor URAM branch/resource linkage is also tracked by the same audit; Req6 legality/static-assert guards, C3b `csynth.xml` resource-policy proof, Req1 Ubuntu `/tools/Xilinx` environment proof, Req9 PAPER_PRJXR DeiT image proof, XR-VITs gate audit, and C3b physical-smoke gate audit are promoted to final evidence; final evidence consistency checks rise to `184`.
- [x] Req1 environment audit added and promoted: Ubuntu `22.04`, non-WSL kernel, hardware root under `/home/kjm26/project`, and `/tools/Xilinx` Vitis/Vivado executables are checked without running Xilinx tools.
- [x] C3b smoke candidate discovery metadata-noise filter tightened: `c3b_physical_smoke_gate_audit_2026_06_16.json` is excluded as gate metadata, not a board-produced smoke result; discovery is clean with `candidate_count=0`, `pass_count=0`.
- [x] Generic PYNQ smoke discovery metadata-noise filter tightened: `_gate_audit_` filenames are excluded across presets and generated-signoff metadata regression covers C3b gate/remote-run artifacts.
- [x] PYNQ smoke validator overlay provenance hardened: C3b/VREF/QKV/m_axi presets now require `bitfile` and `hwhfile` basenames to match the expected overlay artifact prefix when path validation is enabled; importer and discovery inherit the stricter gate.
- [x] Overlay-provenance hardening promoted to final evidence manifest: validator source contract checks are now manifest consistency gates; live manifest `84/84`, consistency `266`, failed `0`.
- [x] Generic C3b/VREF PYNQ smoke discovery is now final-runner generated, mirrored to docs/resources, and required by final evidence manifest.
- [x] Generic C3b/VREF/QKV PYNQ smoke discovery semantic/safety checks are now final evidence and bundle gates.
- [x] Final blocker-readiness and runner summaries now expose direct unblock inputs: C3b canonical smoke JSON path, exact XR-VITs path, active replacement-policy path, and side-effect-free C3b/VREF/QKV PYNQ discovery summaries.
- [x] Operator handoff live-evidence contract gate is now hard: `final_evidence_manifest_contract` must exactly match live `docs/resources/final_evidence_manifest_2026_06_10.json`; stale embedded `255`/missing-live manifest regressions fail.
- [x] Final blocker-readiness discovery summaries and final-runner blocker input path fields are now final evidence manifest consistency gates and bundle-required checks.
- [x] Sweep/Req6 parallelism space now includes PAR16 and PAR32: C3b PAR16 remains the validated board-smoke candidate; PAR32 is exploratory and requires fresh csynth/routed evidence before promotion.
- [x] Final blocker readiness `--c3b-no-require-paths` semantics are aligned for current and candidate C3b checks; strict default remains path-enforcing.
- [x] C3b transfer manifest regeneration is integrated into the final runner: transfer JSON/Markdown/SHA256 are regenerated and mirrored, and runner summary reports `c3b_transfer_manifest_status=pass`.
- [x] C3b transfer manifest is now semantically gated in final evidence: status/preset/variant/target, tar shape, transfer files, clean bundle validation, expected outputs, board verify/run commands, host copyback, sha256-file format, and readiness cross-alignment are checked.
- [x] Req6 PAR32 promotion-policy is now semantically gated in final evidence: C3b PAR16 must remain validated resource-matrix evidence, and PAR32 must remain exploratory until fresh csynth, routed timing, resource-fit, and no-C3b-overwrite evidence exist.
- [x] XR-VITs replacement-policy preview path contract is now semantically gated in final evidence: active policy path, tracked candidate-audit relative path, and integrity candidate-audit absolute path must match default signoff locations.
- [x] XR-VITs replacement-policy preview approval-event contract is now semantically gated in final evidence: schema, policy-field match, and event-id hash must pass; live manifest `84/84`, consistency `269`, failed `0`.
- [x] XR-VITs active replacement-policy creation now rejects placeholder approver metadata: `--approved-by <approved-by>` is allowed only for dry-run preview, active writes fail and do not create the policy file; final runner treats active policy creation failure as blocked.
- [x] XR-VITs active placeholder guard is now semantically gated in final evidence through source-contract checks; live manifest consistency is `272`, failed `0`.
- [x] Manual Spec fallback is now semantically gated in final evidence through spec-plan conformance checks; live manifest consistency is `297`, failed `0`.
- [x] Completion audit freshness is now semantically gated in final evidence without a self-referential deadlock: completion Req12 reads the stored manifest contract, ignores only completion self-gates during convergence, and keeps non-completion manifest failures blocked.
- [x] Current-doc freshness is now semantically gated through spec-plan conformance: current-doc live count checks cover Master-Plan, Sub-Plan, Spec, Validation, PROGRESS, HANDOVER, CHOICE, and log; spec-plan audit target is `86/86` and includes live operator handoff validation `131/131` plus bundle validation `152/152`.
- [x] HANDOVER raw validator summary freshness is now gated: stale `checks 129` or `checks 150` artifact summaries fail spec-plan conformance even if live `131/131` and `152/152` anchors are present.
- [x] RMU/SMU projection and relation-stage multiply-heavy paths now route through DSP-bound helper functions and are final-evidence gated through resource-policy audit; cyclic packed-weight tiles/large temporaries now use URAM and small tile scratch uses LUTRAM through parameterized macros. C3b LUT/latency/WNS thresholds, final-runner blocker/mirror count contracts, and source-audit source-count freshness are now final-evidence gated. Resource-policy audit target is `36/36`; final evidence consistency target is `326`.
- [x] Req5 packed Q4 binary/manifest integrity is now final-evidence gated: packed binary exists, SHA256 matches manifest, byte count matches manifest, expected runtime state is `2`, expected C3b raw output is `[32, -13, 26, -6, 14, -11]`, and strict TB/CSim checks pass.
- [x] Final signoff bundle validation now requires the Req5 packed-weight final-manifest checks to be present/pass, so a stale bundle cannot pass by carrying only the Req5 artifact ids.
- [x] Final operator handoff validation now also requires the Req5 packed-weight live final-manifest checks to be present/pass; stale live manifests with unchanged consistency count are rejected.
- [x] Final signoff bundle validation now also requires latest operator handoff validation freshness: `check_count >=131`; stale handoff validation fails bundle validation.
- [x] Final evidence manifest now also requires latest final-unblock closeout validation freshness: `check_count >=49`; stale closeout validation fails final evidence.
- [x] Final evidence manifest now also requires third-goal source-audit freshness: `required_count >=86` and `source_count >=224`; stale source audit artifacts fail final evidence.
- [x] Final operator handoff and final signoff bundle validation now require source-audit freshness manifest checks by name; count-preserving removal of either source-audit freshness check fails validation.
- [x] Final operator handoff and final signoff bundle validation now directly validate live `third_goal_source_audit_2026_06_16.json`; stale `85/223` live source-audit payloads fail even when final-manifest source-audit check names remain present.
- [x] Final operator handoff and final signoff bundle validation now directly validate live `e2e_resource_policy_audit_2026_06_10.json`; stale `35`-check resource-policy payloads and failed core DSP/URAM/LUTRAM/C3b checks fail even when final-manifest resource-policy check names remain present.
- [x] Final operator handoff and final signoff bundle validation now directly validate live `spec_plan_conformance_audit_2026_06_10.json`; stale spec-plan payloads below `86` checks or failed core plan/spec/current-doc checks fail even when final-manifest spec-plan check names remain present.
- [x] Final operator handoff and final signoff bundle validation now require final evidence consistency count `>=347`; stale `340` evidence contracts fail validation.
- [x] Final operator handoff and final signoff bundle validation now require final evidence required count `>=86`; stale `85` evidence contracts fail validation.
- [x] Final operator handoff and final signoff bundle validation now directly validate live `third_goal_final_signoff_run_2026_06_10.json`; stale final-runner blocker/detail/mirror counts fail even when final-manifest runner checks remain present.
- [x] Final operator handoff and final signoff bundle validation now directly validate live `req6_parameterization_audit_2026_06_16.json`; stale Req6 knob/count/PAR16/PAR32 policy drift fails even when final-manifest Req6 checks remain present.
- [x] Final operator handoff and final signoff bundle validation now directly validate live `third_goal_current_audit_2026_06_16.json`; stale current-audit summary, blocker path, XR policy integrity, or QKV optional-state drift fails final validators.
- [x] Final operator handoff and final signoff bundle validation now require cyclic URAM/LUTRAM resource-policy manifest checks by name; count-preserving removal of a cyclic check fails validation.
- [x] Final unblock intake blocker-status and next-input paths are now semantically gated in final evidence; operator handoff validation also cross-checks live XR-VITs reference resolution and live final-unblock-intake state.
- [x] Final-runner `mirrored_artifacts` summary is order-preserving deduped and manifest-gated: latest summary has `92` mirrored paths, `92` unique paths, and `0` duplicates.
- [x] Final-runner summary count checks are now required by name in operator handoff and bundle validators; count-preserving removal of `final_runner_blocker_count_matches_list`, `final_runner_remaining_blocker_detail_count_matches`, or `final_runner_mirrored_artifact_counts_match` fails validation.
- [x] Final-runner docs/resources mirror integrity is now hard-gated: summary records canonical docs root, generated signoff root, checked/excluded/fail counts, and per-artifact sha256/size match; final evidence and both final validators reject stale or missing mirror-integrity fields.
- [x] Resource-policy artifact singularity is now hard-gated: only canonical `e2e_resource_policy_audit_2026_06_10.{json,md}` may exist in `docs/resources` and `hardware/generated/signoff`; filename/payload date drift and generated/docs SHA256 drift fail final evidence and final validators.
- [x] Final-unblock closeout validation freshness is current-gated in final evidence: stale `48`-check validation artifacts fail `final_unblock_closeout_validation_check_count`.
- [x] P2-ViT Q4/Q8 software-first scale calibration is now summarized as a dedicated report and final evidence gate: report `pass`, checks `10/10`, current PoT scales kept for C3b, non-current candidates remain successor-only.
- [x] Final unblock intake now exposes blocker-by-blocker machine-readable next inputs: C3b canonical smoke JSON path and XR-VITs exact-source/approved-policy path are listed in `blocker_status` and `next_inputs`.
- [x] XR-VITs reference-resolution/operator-handoff commands now include explicit `--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR` to avoid accidental cwd-dependent approval runs.
- [x] Overlay-provenance hardening validated: focused tests `39` passed, final-signoff-related tests `74` passed, manifest/signoff tests `56` passed, py_compile passed, final runner remains expected-blocked only on C3b physical smoke and XR-VITs source/policy.
- [ ] `VREF-P0-02` QKV weight-cache URAM physical-smoke evidence captured.
- [x] VREF hardware successor synthesized, packaged, routed, and exported as bit/hwh.

## Active Priorities
1. C3b board-smoke unblock.
2. XR-VITs source/replacement policy unblock.
3. P2-ViT Q4/Q8 scale calibration, software-first.
4. VREF successor promotion only if it passes C3b protection thresholds.
5. Attention/MoE ablations only after scope review.

## 2026-06-27 No-Board P0 / PAR32 Vivado Continuation

- [x] PAR32 csynth sweep completed for `par32_dsp_uram_mem16`, `par32_dsp_mixed_mem16`, and `par32_dsp_mixed_stream_mem16`.
- [x] Best csynth-fit candidate selected for route check: `par32_dsp_mixed_stream_mem16`, latency `26,139,644` cycles, DSP `1,148`, LUT `193,546`, BRAM_18K `332`, URAM `64`.
- [x] HLS IP package completed for `par32_dsp_mixed_stream_mem16`: package gate `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` exists.
- [x] Vivado route completed for `par32_dsp_mixed_stream_mem16`: WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen completed, and `.bit/.hwh` artifacts generated.
- [x] Search 4 ms / Track 1 ms mode-specific HLS profile latency proven: Search `772,268` cycles / `3.861 ms`, Track `99,449` cycles / `0.497 ms`.
- [x] Search/Track mode-profile HLS IP packages generated: `generated/hgtxr_mode_search_par32_no_board/solution_mode_q4w8a/impl/ip/component.xml`, `generated/hgtxr_mode_track_par32_no_board/solution_mode_q4w8a/impl/ip/component.xml`.
- [x] Mode-profile Vivado route completed: Search WNS `2.638 ns`, Track WNS `1.481 ns`, route errors `0`, bitgen completed for both.
- [x] Full learned ROM-only E2E AXIS top physically implemented at 300 MHz after Event Conv integration: profile `par8_runtime_rom_only_dispatch_dsp3_300_mem8`, routed `clk_pl_0 = 300.030 MHz`, WNS `0.007 ns`, WHS `0.010 ns`, route errors `0`, bitgen pass.
- [x] Frame Conv and Event Conv now use distinct on-chip ROM regions; Search mode uses Frame Conv and Track mode uses Event Conv in the single shared E2E AXIS top.
- [x] Full learned routed resource fit captured: CLB LUT `24,159/230,400`, FF `26,753/460,800`, Block RAM Tile `238/312`, DSP `1,627/1,728`, URAM `32/96`.
- [x] Full learned routed vectorless power captured: total `6.037 W`, dynamic `5.322 W`, static `0.715 W`, PS static `0.102 W`, PL static `0.614 W`.
- [ ] Full learned Search latency target met: current routed-clock estimate is `43.294 ms`, target `<=4 ms`.
- [ ] Full learned Track latency target met: current routed-clock estimate is `5.350 ms`, target `<=1 ms`.
- [ ] Search dispatcher proven as overlapped double-buffer prefetch rather than observable preload/marker logic.
- [ ] Mode-specific board latency distribution captured: p95/p99, DMA bandwidth, throughput, and real Search/Track invocation distribution are still missing.
- [ ] Mode-specific board power captured: PS/PL/DDR/sensor I/O rail measurement is still missing.
- [x] No-board P0 collector gate added and passed: `python3 scripts/report/collect_no_board_reports.py --output-dir /tmp/hgtxr_p0_collect_check --enforce-p0`.
- [x] Current work/progress/experiment conversation consolidated into `docs/track/WORK_PROGRESS_EXPERIMENT_SUMMARY_2026_06_27.md`.
- [x] Latest continuation handover saved as `docs/track/HANDOVER_2026_06_27.md`.
- [x] Full E2E vs mode-profile evidence boundary clarified: mode-profile Search/Track meets 4 ms/1 ms, but full E2E Search 8-block / Track 4-block latency is not proven yet.
- [ ] Search mode URAM route margin improved: pending; current Search HLS estimate is `96/96` URAM.

## 2026-06-27 Runtime-Mode E2E Shared Top

- [x] Single `hgtxr_e2e_axis_top` runtime mode branch added for Search/Track workload selection.
- [x] Search/Track CSim regression passed under `par32_runtime_mode_mem16`: Search `runtime_state=0`, Track `runtime_state=1`, state output count `6`, TLAST correct, `CSim done with 0 errors`.
- [x] ZCU104-part HLS csynth latency target passed for the runtime-mode shared top: Track-bound min `76,150` cycles / `0.381 ms`; Search-bound max `594,840` cycles / `2.974 ms`.
- [x] Runtime-mode HLS resource estimate captured: BRAM_18K `68/624`, DSP `400/1728`, FF `39,696/460,800`, LUT `88,246/230,400`, URAM `0/96`.
- [x] Runtime-mode HLS IP package generated: `generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`.
- [x] Runtime-mode AXIS/DMA Vivado route and bitgen completed: WNS `3.038 ns`, TNS `0.000 ns`, fully routed nets `50,076/50,076`, route errors `0`, `.bit/.hwh` copied to overlay and `pynq/hgtxr`.
- [x] Runtime-mode unused `gmem_e2e_weights` AXI memory master removed for the fast profile; packaged IP, generated RTL, overlay `.hwh`, and PYNQ `.hwh` do not contain `m_axi_gmem_e2e_weights`.
- [x] Routed power/resource snapshot captured: total on-chip power `3.736 W`, dynamic `3.042 W`, static `0.694 W`; placed CLB LUT `22,412`, register `21,105`, BRAM tile `37`, DSP `401`, URAM `0`.
- [x] Runtime-mode PYNQ Search/Track smoke bundles generated and validated: `e2e_axis_dma_par32_runtime_mode_search_smoke_bundle.tar.gz`, `e2e_axis_dma_par32_runtime_mode_track_smoke_bundle.tar.gz`.
- [x] Runtime-mode ZCU104 Search/Track smoke-session runbooks generated: `e2e_axis_dma_par32_runtime_mode_search_smoke_session.{json,md}`, `e2e_axis_dma_par32_runtime_mode_track_smoke_session.{json,md}`.
- [x] Runtime-mode PYNQ result validation now enforces board accelerator latency targets: Search `<=4.000 ms`, Track `<=1.000 ms`.
- [x] Runtime-mode ZCU104 SSH/SCP remote-run dry-run plans generated: `zcu104_runtime_mode_par32_search_smoke_remote_run_2026_06_10.{json,md}`, `zcu104_runtime_mode_par32_track_smoke_remote_run_2026_06_10.{json,md}`.
- [x] Runtime-mode HLS RTL cosim/HW Emulation runner added: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_mode_mem16`.
- [x] Runtime-mode HLS RTL cosim C testbench coverage reached Search and Track: Search `runtime_state=0`, Track `runtime_state=1`, both with 6 output state words and correct TLAST.
- [ ] Runtime-mode HLS RTL cosim final PASS remains pending: XSIM builds snapshot `hgtxr_e2e_axis_top` but fails on generated `-autoloadwcfg` snapshot-load command with `unexpected exception when evaluating tcl command`.
- [x] Runtime-mode combined board-latency gate added: `runtime_mode_board_latency_gate_2026_06_28.{json,md}` currently reports `missing` until both canonical board result JSON files are imported.
- [x] Runtime-mode combined Search/Track board runner added: `tools/run_runtime_mode_board_latency.py`; dry-run output `runtime_mode_board_latency_run_2026_06_28.{json,md}` reports `status=dry-run`, gate `missing`.
- [ ] Runtime-mode combined Search/Track board runner execute remains pending: latest execute attempt `runtime_mode_board_latency_run_2026_06_28_execute.{json,md}` stopped at host preflight with `status=host-unresolved` because `zcu104.local` could not be resolved.
- [x] Evidence report saved as `docs/track/RUNTIME_MODE_E2E_SHARED_TOP_2026_06_27.md`.
- [ ] Runtime-mode top board latency smoke remains pending.

## 2026-06-28 Runtime-Mode Full AXI Learned Path

- [x] Added explicit full learned-parameter runtime profile: `par32_runtime_full_axi_mem16`.
- [x] Selected QKV/MLP/Conv/Head parameter source as external AXI `gmem_e2e_weights`, not persistent ROM/BRAM/URAM constants.
- [x] Restored non-fastpath learned path for the runtime top: `hgtxr_conv_patch_embedding`, weighted `hgtxr_e2e_controller_run`, and `hgtxr_e2e_mlp_head`.
- [x] Full AXI CSim passed Search/Track runtime checks: Search `runtime_state=0`, Track `runtime_state=1`, output count `6`, TLAST correct.
- [x] Full AXI csynth completed and remeasured latency: Track-bound min `627,117` cycles / `3.136 ms`; Search-bound max `5,101,975` cycles / `25.510 ms`.
- [x] Full AXI HLS resource estimate captured: BRAM_18K `332/624`, DSP `1,148/1,728`, FF `86,123/460,800`, LUT `193,627/230,400`, URAM `64/96`.
- [x] Full AXI HLS IP package generated: `generated/hgtxr_e2e_axis_par32_runtime_full_axi_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`.
- [x] Fastpath vs full-model evidence boundary documented in `docs/track/RUNTIME_MODE_FULL_AXI_WEIGHT_PATH_2026_06_28.md`.
- [ ] Full AXI learned path latency target recovery remains pending: current HLS estimate fails Search `<=4 ms` and Track `<=1 ms`.
- [ ] Full AXI Vivado route/power and board p95/p99/DMA measurement remain pending.

## 2026-06-28 Full-Learned ROM-Only DSP3 Vivado Closure

- [x] Added `par8_runtime_rom_only_dispatch_dsp3_300_mem8` and `runtime_rom_only_dispatch_dsp3_active64_b8_ff768`.
- [x] Wired `HGTXR_E2E_DSP_MUL_LATENCY` into HLS DSP `bind_op` latency so the dsp3 profile changes generated DSP cores instead of only defining an unused macro.
- [x] CSim passed Search/Track with deterministic outputs, correct runtime states, output count `6`, and TLAST.
- [x] HLS package passed; `component.xml` contains `m_axi_gmem_e2e_runtime` and no `m_axi_gmem_e2e_weights`.
- [x] Vivado route/bitgen passed at `clk_pl_0 = 300.030 MHz`: WNS `0.009 ns`, TNS `0.000 ns`, WHS `0.010 ns`, route errors `0`.
- [x] Routed ZCU104 resources captured: CLB LUT `23,887/230,400`, register `26,595/460,800`, BRAM tile `238/312`, DSP `1,627/1,728`, URAM `32/96`.
- [x] Routed vectorless power captured: total `5.973 W`, dynamic `5.258 W`, static `0.715 W`.
- [ ] Search/Track latency targets remain unmet: Search `43.239 ms > 4 ms`, Track `5.336 ms > 1 ms`.
- [ ] Mode-specific power, measured DMA bandwidth, board rail power, repeated latency distribution, and p95/p99 remain pending.

## 2026-06-28 Dispatcher Prefix-Prefetch Full-Learned Vivado Rerun

- [x] Replaced dispatcher marker/preload behavior with two concrete `16`-word ping-pong prefix-prefetch banks in `HgtxrGlobalBuffer`.
- [x] Connected dispatcher prefetch to actual learned-parameter cache fills for LayerNorm, Q/K/V, output projection, and MLP W1/W2.
- [x] CSim passed Search/Track runtime checks with correct runtime states, six output words, TLAST, and `CSim done with 0 errors`.
- [x] HLS package passed at target `3.333 ns`, estimated `2.924 ns`; Search/max envelope is `13,008,010` cycles / `43.356 ms`.
- [x] Vivado route/bitgen passed at `clk_pl_0 = 300.030 MHz`: WNS `0.000 ns`, TNS `0.000 ns`, WHS `0.007 ns`, route errors `0`.
- [x] Current routed resources captured: CLB LUT `43,929/230,400`, registers `43,016/460,800`, Block RAM Tile `238/312`, DSP `1,630/1,728`, URAM `32/96`.
- [x] Current routed vectorless power captured: total `6.833 W`, dynamic `6.111 W`, static `0.721 W`.
- [ ] Full-block double-buffer dispatcher proof remains pending; current design only prefetches a `16`-word block prefix.
- [ ] Current Track-mode cycle isolation remains pending; HLS top min `26,190` cycles is not accepted as full Track latency.
- [ ] Search/Track latency targets remain unmet: current Search bound is `43.356 ms > 4 ms`, and prior comparable Track envelope is `5.350 ms > 1 ms`.
- [ ] Mode-specific board p95/p99 latency, measured DMA bandwidth, real Search/Track distribution, board rail power, and sensor I/O power remain pending.

## 2026-06-28 Dispatcher Full-Block Prefetch Full-Learned Vivado Rerun

- [x] Replaced dispatcher prefix sizing with full-block sizing through `kBlockWeightWords` for the dsp3 ROM-only profile.
- [x] Bound the full-block dispatcher prefetch bank to URAM for the enabled profile.
- [x] CSim passed Search/Track runtime checks with correct runtime states, six output words, TLAST, and `CSim done with 0 errors`.
- [x] HLS package passed at target `3.333 ns`, estimated `2.924 ns`; Search/max envelope is `13,042,545` cycles / `43.471 ms`.
- [x] HLS storage proof captured: `gb_dispatch_prefetch_U` is `13,848 x 255-bit`, implemented as `RAM_2P_URAM_1R1W`, using `16` URAM.
- [x] Vivado route/bitgen completed with `impl_strategy=Performance_Explore`, route errors `0`, and fully routed nets `141,580/141,580`.
- [x] 300 MHz timing signoff now passes for this exact full-block build: `clk_pl_0 = 300.030 MHz`, WNS `0.010 ns`, TNS `0.000 ns`, WHS `0.009 ns`, THS `0.000 ns`.
- [x] Current placed resources captured: CLB LUT `44,894/230,400`, registers `44,799/460,800`, Block RAM Tile `238/312`, DSP `1,630/1,728`, URAM `48/96`.
- [x] Current routed vectorless power captured: total `7.109 W`, dynamic `6.383 W`, static `0.725 W`.
- [x] Current Search-mode cycle isolation captured with force-mode csynth: max `13,001,422` cycles / `43.334 ms`, interval max `13,001,423`.
- [x] Current Track-mode cycle isolation captured with force-mode csynth: max `1,606,660` cycles / `5.580 ms` in the HLS report, or `5.355 ms` at the routed `300.030 MHz` clock.
- [x] Search dispatcher full-block prefetch-before-use proof captured in CSim trace for blocks `2` and `3`: prefetch done before first load, `violations=0`.
- [ ] Search dispatcher overlapped double-buffer execution proof remains pending; current trace proves ordering, not concurrent dataflow.
- [ ] Search/Track latency targets remain unmet: current Search is `43.334 ms > 4 ms`, and current Track is `5.355 ms > 1 ms` at routed clock.
- [ ] Mode-specific board p95/p99 latency, measured DMA bandwidth, real Search/Track distribution, board rail power, and sensor I/O power remain pending.

## 2026-06-28 PAR32 Baseline Recheck for Objective Path

- [x] Baseline policy confirmed: keep `par32_dsp_mixed_stream_mem16` as the physical/timing baseline and `par32_runtime_full_axi_mem16` as the full learned AXI functional baseline.
- [x] Added PAR32 ROM-only objective-path profiles: `par32_runtime_rom_only_dispatch_dsp3_300_mem16`, `par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only`, and `par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only`.
- [x] Search-only PAR32 objective csynth completed: `4,764,228` cycles / `15.879 ms`, estimated clock `2.777 ns`.
- [x] Track-only PAR32 objective csynth completed: `604,652` cycles / `2.100 ms`, estimated clock `3.473 ns`.
- [x] Combined PAR32 objective package completed and generated HLS IP.
- [x] Vivado implementation result captured: place did not run because DRC `UTLZ-1` found `1729/1728` DSP over-utilization.
- [ ] PAR32 ROM-only objective path is not a valid ZCU104 implementation point until DSP usage is reduced by at least one DSP and LUT pressure is rechecked.
- [ ] Even with PAR32, latency targets remain unmet: Search `15.879 ms > 4 ms`, Track `2.100 ms > 1 ms`.

## 2026-06-28 PAR32 Objective DSP-Fit Core-Fabric Rerun

- [x] Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16` with `HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=1` to move the last core dense lane off DSP.
- [x] CSim passed Search/Track runtime checks: Search output `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track output `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, vector comparison passed.
- [x] CSynth passed: target `3.333 ns`, estimate `2.777 ns`, top max latency `4,805,312` cycles / `16.016 ms`, interval max `4,805,313`.
- [x] HLS package passed and patched `43` exported Verilog datapath modules for `HGTXR_E2E_OOC_DONT_TOUCH=datapath`.
- [x] Vivado route and bitgen completed; route errors `0`; `.bit/.hwh` generated and copied to `hardware/pynq/hgtxr/`.
- [x] PAR32 objective implementation now exactly fits ZCU104 DSP at DSP48E2 `1,728/1,728`.
- [x] Placed resources captured: CLB LUT `89,102/230,400`, registers `60,400/460,800`, Block RAM Tile `232/312`, DSP `1,728/1,728`, URAM `48/96`.
- [x] Routed vectorless power captured: total `8.593 W`, dynamic `7.856 W`, static `0.737 W`; hierarchy dynamic includes `hgtxr_e2e_axis_top_0 = 4.933 W`, `psu = 2.674 W`, and `axi_mem = 0.194 W`.
- [ ] 300 MHz timing signoff fails for this PAR32 objective probe: WNS `-0.631 ns`, TNS `-3338.375 ns`, failing setup endpoints `15,703/303,013`.
- [ ] Latency targets still fail: combined PAR32 max envelope is `16.016 ms`; prior PAR32 force-mode Search/Track are `15.879 ms` and `2.100 ms`; exact corefabric force-mode Search/Track remains pending.
- [ ] Next cleanup: DSP MREG/PREG or added pipeline stages, LayerNorm/BRAM/QKV critical-path reduction, `DONT_TOUCH` scope review, then rerun mode-specific latency and power experiments.

## 2026-06-28 PAR32 Baseline Taxonomy and Norm-URAM Probe

- [x] Baseline taxonomy corrected: `par32_dsp_mixed_stream_mem16` is the physical/timing baseline, `par32_runtime_full_axi_mem16` is the full learned AXI functional baseline, and the ROM-only scheduler/dispatcher design is the separate objective baseline.
- [x] Added `dsp_mixed_normstream` resource policy and `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16`.
- [x] CSim passed for the `normuram` objective probe: Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`.
- [x] CSynth passed at target `3.333 ns`, estimate `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, interval max `4,805,313`.
- [x] CSynth resource shift confirmed: BRAM_18K `418 -> 386`, URAM `48 -> 64`, latency unchanged versus the previous PAR32 core-fabric probe.
- [x] Vivado route and bitgen completed; route errors `0`; DSP fits exactly at `1,728/1,728`; `.bit/.hwh` copied to `hardware/pynq/hgtxr/`.
- [x] Routed resource/power snapshot captured: CLB LUT `89,329/230,400`, registers `60,699/460,800`, Block RAM Tile `216/312`, DSP `1,728/1,728`, URAM `64/96`; total power `8.568 W`, dynamic `7.829 W`, static `0.738 W`.
- [ ] 300 MHz timing signoff still fails for `normuram`: WNS `-0.496 ns`, TNS `-2860.064 ns`, failing setup endpoints `15,033/303,394`.
- [ ] The earlier two PAR32 profiles must stay in all comparison tables, but neither should be used as the objective-signoff baseline for the ROM-only scheduler/prefetch claim.

## 2026-06-28 PAR32 Norm-Stage LayerNorm Write Probe

- [x] Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16` to test staged LayerNorm writes before `gb.norm`.
- [x] CSim passed Search/Track runtime checks: Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, vector comparison passed.
- [x] CSynth passed at target `3.333 ns`, estimate `2.777 ns`, top max `4,908,208` cycles / `16.359 ms`, controller max `4,829,897` cycles / `16.098 ms`.
- [x] Vivado route and bitgen completed; route errors `0`, fully routed nets `201,593/201,593`, DSP exactly fits at `1,728/1,728`.
- [x] Placed resource/power snapshot captured: CLB LUT `89,035/230,400`, registers `59,380/460,800`, Block RAM Tile `216/312`, DSP `1,728/1,728`, URAM `64/96`; total power `8.615 W`, dynamic `7.876 W`, static `0.739 W`.
- [ ] 300 MHz timing signoff fails for `normstage`: WNS `-0.644 ns`, TNS `-3987.069 ns`, failing setup endpoints `16,238/303,146`.
- [ ] `normstage` did not improve over `normuram`; next PAR32 closure should target DSP MREG/PREG pipeline insertion, `gb_dispatch_prefetch` URAM read staging, and selective `DONT_TOUCH` relaxation.

## 2026-06-28 PAR32 Norm-URAM No-DONT-TOUCH Sanity Probe

- [x] Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16` as a negative-control timing experiment for global datapath-preservation removal.
- [x] CSim passed with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and vector comparison pass.
- [x] CSynth/package passed with the same HLS objective-scale resources as `normuram`: BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`, max latency `4,805,312` cycles / `16.016 ms`.
- [x] Vivado route/bitgen passed timing at 300 MHz: WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, route errors `0`.
- [ ] This is not valid accelerator signoff because placed utilization collapsed to DSP `0/1,728`, URAM `0/96`, BRAM tile `5/312`, CLB LUT `7,817/230,400`; the learned compute fabric was optimized/pruned away.
- [ ] Baseline policy remains: use `par32_dsp_mixed_stream_mem16` as physical/timing baseline, `par32_runtime_full_axi_mem16` as learned AXI functional baseline, and preserved ROM-only PAR32 probes as objective-path candidates.

## 2026-06-28 PAR32 Norm-URAM Top-DONT-TOUCH Sanity Probe

- [x] Added `HGTXR_E2E_OOC_DONT_TOUCH=top` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16`.
- [x] Package patched exactly one exported Verilog module, the HLS top wrapper, instead of the broad datapath module set.
- [x] CSim passed with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and vector comparison pass.
- [x] CSynth/package preserved the HLS objective-scale estimate: BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`, max latency `4,805,312` cycles / `16.016 ms`.
- [x] Vivado route/bitgen passed timing at 300 MHz: WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, route errors `0`.
- [ ] This is also not valid accelerator signoff because placed utilization collapsed to the same pruned shell as `nodt`: DSP `0/1,728`, URAM `0/96`, BRAM tile `5/312`, CLB LUT `7,817/230,400`.
- [ ] Next valid PAR32 closure path remains selective internal datapath preservation plus explicit DSP/URAM/cache-read pipeline insertion, not top-only or global DONT_TOUCH removal.

## 2026-06-28 PAR32 PipePref Fabric-Lane Closure Probes

- [x] Baseline roles are fixed for all follow-up tables: `par32_dsp_mixed_stream_mem16` is the physical/timing baseline, and `par32_runtime_full_axi_mem16` is the learned-AXI functional baseline.
- [x] Added pipe-prefetch and fabric-lane profiles through the no-board and Vivado runners: `pipepref`, `pipepref_tail2`, `pipepref_tail4`, `pipepref_tail8`, `pipepref_coreallfabric`, and `pipepref_coreallfabric_nowide`.
- [x] `pipepref` CSim passed Search/Track runtime checks with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and `0` errors.
- [x] `pipepref` package passed: max latency `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,976`, FF `277,802`, LUT `453,254`, URAM `64`.
- [x] `pipepref` Vivado DRC result captured: failed before placement because required DSP was `1,740/1,728`.
- [x] `pipepref_tail4` and `pipepref_tail8` showed that runtime tail-lane fabric selection does not reduce final physical DSP enough; both still failed Vivado DRC at required DSP `1,740/1,728`.
- [x] Added optional `HGTXR_E2E_CORE_LANE_CT_SWITCH` plus `pipepref_cttail4` to test a template/switch lane split without changing default profiles.
- [x] `pipepref_cttail4` package passed, but it matched `tail4`: max latency `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,946`, FF `278,756`, LUT `460,646`, URAM `64`.
- [x] `pipepref_coreallfabric` reduced HLS DSP to `1,666` but exceeded LUT capacity at `529,792/230,400 = 229.9%`; Vivado was not run.
- [x] `pipepref_coreallfabric_nowide` reduced HLS DSP to `1,666`, but LUT grew further to `647,424/230,400 = 281%`; package passed with max latency `4,817,820` cycles / `16.058 ms`.
- [ ] None of the pipe-prefetch fabric-lane probes is a valid ZCU104 signoff point.
- [ ] Next implementation step should replace lane-selection wrappers with an explicit separate operator/data-path split or explicit pipelined DSP binding, then rerun HLS/Vivado on the preserved ROM-only objective path.

## 2026-06-29 PAR32 Compute-Selective DONT_TOUCH Probe

- [x] Reconfirmed baseline policy: `par32_dsp_mixed_stream_mem16` is the physical/timing baseline and `par32_runtime_full_axi_mem16` is the full learned AXI functional baseline. They are required comparison baselines, not discarded candidates.
- [x] Added `HGTXR_E2E_OOC_DONT_TOUCH=compute` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16` to preserve only the key learned compute modules instead of the broad datapath set.
- [x] HLS package passed at target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, interval max `4,805,313`.
- [x] HLS package patched exactly 9 compute module Verilog files: controller, two ATTN units, two MLP units, QKV projection, attention core, output projection, and MLP head.
- [x] Vivado route and bitgen completed; routed nets were fully routed `199,142/199,142`, route errors `0`, and bitgen completed successfully.
- [x] Placed resources captured: CLB LUT `86,983/230,400 = 37.75%`, CLB registers `59,792/460,800 = 12.98%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%`.
- [x] Routed vectorless power captured: total on-chip `8.537 W`, dynamic `7.799 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.633 W`.
- [ ] 300 MHz timing signoff still fails: WNS `-0.530 ns`, TNS `-3005.251 ns`, setup failing endpoints `15,103/302,281`, hold clean at WHS `0.000 ns`, THS `0.000 ns`.
- [ ] The compute-selective probe is a valid routed/bitgen objective-path experiment, but it is not a signoff baseline because timing and latency targets still fail.
- [ ] Next closure target: remove or narrow DONT_TOUCH on ATTN/MLP child cells that block fanout replication, add explicit DSP MREG/PREG or equivalent HLS pipeline stages, and stage URAM/cache read paths reported in worst setup paths.

## 2026-06-29 300MHz Conditional WNS Threshold and PostRoutePhys Probe

- [x] Applied the user-directed experiment rule: PL clock remains `300 MHz`, and this experiment accepts routed WNS down to `-0.500 ns`. Official Vivado signoff remains WNS `>= 0.000 ns`.
- [x] Added 300 MHz pipe-prefetch follow-up profiles in the HLS and Vivado runners: `compute_pipepref_300` and `compute_pipepref_tail4_nowide_300`.
- [x] `compute_pipepref_300` package passed with max latency `4,817,564` cycles / `16.057 ms`, but Vivado failed before placement because required DSP was `1,740/1,728`.
- [x] `compute_pipepref_tail4_nowide_300` package passed with max latency `4,817,820` cycles / `16.058 ms`, but it was rejected without Vivado because HLS DSP worsened to `2,226`.
- [x] Reran the existing valid `compute_300` IP with Vivado strategy `Performance_ExplorePostRoutePhysOpt` under project `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_postroutephys_300_mem16_overlay`.
- [x] PostRoutePhys implementation completed route and bitgen: route errors `0`, fully routed nets `199,370/199,370`, bitgen pass.
- [x] PostRoutePhys run passes the user-approved 300 MHz experiment threshold: WNS `-0.402 ns >= -0.500 ns`, TNS `-985.598 ns`, WHS `0.005 ns`, THS `0.000 ns`.
- [ ] PostRoutePhys run is not official clean timing signoff because WNS is still negative and Vivado reports `Timing constraints are not met`.
- [x] PostRoutePhys resources captured: CLB LUT `87,181/230,400 = 37.84%`, CLB registers `59,652/460,800 = 12.95%`, Block RAM Tile `216/312 = 69.23%`, DSP `1,728/1,728 = 100.00%`, URAM `64/96 = 66.67%`.
- [x] PostRoutePhys vectorless power captured: total `8.539 W`, dynamic `7.801 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`.
- [ ] Next official-clean timing target: add explicit cache/URAM read output staging and DSP MREG/PREG-equivalent pipeline stages for `wo_weight_cache`, `gb_dispatch_prefetch`, and MLP multiply paths.

## 2026-06-29 PAR32 ROM Compute 300MHz PYNQ SW/HW Equality Bundle

- [x] Added PYNQ runner variant `par32-rom-compute-300` for the current postroutephys objective artifact.
- [x] Added board-smoke bundle variants `par32-rom-compute-300-search` and `par32-rom-compute-300-track`.
- [x] Search board-smoke contract now checks mode profile `search`, runtime state `0`, and expected raw output `[-1169, -1169, -1169, -1169, -1169, -1133]`.
- [x] Track board-smoke contract now checks mode profile `track`, runtime state `1`, and expected raw output `[-235, -235, -235, -235, -235, -226]`.
- [x] Generated Search transfer bundle: `hardware/generated/pynq/e2e_axis_dma_par32_rom_compute_300_search_smoke_bundle.tar.gz`, sha256 `d21042211c8aec38252618c339818a4a5e86786283813026ecf9218086544fa0`.
- [x] Generated Track transfer bundle: `hardware/generated/pynq/e2e_axis_dma_par32_rom_compute_300_track_smoke_bundle.tar.gz`, sha256 `15e570fa6950cc0e501d54af7d2c9e5920073df10126fc649597f9efb03cf47b`.
- [x] Bundle validation passed for both Search and Track with `errors=[]`.
- [x] Regression tests passed: PYNQ smoke CLI `6` tests; bundle/validator focused set `30` tests.
- [ ] Physical ZCU104 run JSON is still missing; these bundles prepare the SW/HW equality check but do not replace board execution.

## 2026-06-29 PAR32 ROM Compute 300MHz Board Runner/Gate Contract

- [x] Locked the current experiment rule: PL clock `300 MHz`, with WNS accepted down to `-0.500 ns` for experiment continuation only.
- [x] Added separate runner/gate profile set `par32-rom-compute-300` so objective Search/Track board evidence is not mixed with the older `runtime-mode-par32` profile set.
- [x] Added remote runner profiles `par32-rom-compute-300-search` and `par32-rom-compute-300-track`.
- [x] Added import presets `axis-par32-rom-compute-300-search` and `axis-par32-rom-compute-300-track`.
- [x] Added board latency gate cases for the objective Search target `4.0 ms` and Track target `1.0 ms`.
- [x] Generated dry-run run artifact `hardware/generated/signoff/par32_rom_compute_300_board_latency_run_2026_06_29.json`.
- [x] Generated standalone gate artifact `hardware/generated/signoff/par32_rom_compute_300_board_latency_gate_2026_06_29.json`.
- [x] Tooling verification passed: `py_compile` on modified runner/gate/import files and `31` focused unit tests.
- [ ] Gate status is still `missing` because ZCU104 Search/Track result JSON files are not captured in `hardware/pynq/hgtxr/`.
- [ ] Need actual board execution with `--execute`, result import, then gate rerun before reporting measured Search/Track latency, p95/p99 latency, measured DMA bandwidth, or board power.

## 2026-06-29 PAR32 ROM Compute 300MHz Mode-Specific HLS Evidence

- [x] User rule applied for this lane: PL clock target `300 MHz`; experiment accepts routed WNS down to `-0.500 ns`; official Vivado signoff still requires WNS `>= 0.000 ns`.
- [x] Added current `compute_300` force-mode HLS profiles for Search-only and Track-only.
- [x] Search-only CSim passed with output `[-1169, -1169, -1169, -1169, -1169, -1133]`, `runtime_state=0`, vector comparison pass, and prefetch trace violations `0`.
- [x] Track-only CSim passed with output `[-235, -235, -235, -235, -235, -226]`, `runtime_state=1`, and vector comparison pass.
- [x] Corrected the current `par32-rom-compute-300-track` PYNQ expected raw value from stale `-291` to `-226`.
- [x] Search-only CSynth passed: target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,764,201` cycles / `15.879 ms`, interval max `4,764,202`, HLS resources BRAM_18K `386`, DSP `1,720`, FF `185,414`, LUT `322,452`, URAM `64`.
- [x] Track-only CSynth passed: target `3.333 ns`, estimated clock `3.473 ns`, max latency `604,652` cycles / `2.100 ms`, interval max `604,653`, HLS resources BRAM_18K `378`, DSP `1,332`, FF `181,667`, LUT `288,332`, URAM `48`.
- [x] Hybrid 10% Search / 90% Track HLS max-latency weighted average computed as `3.478 ms`.
- [ ] Search latency target `4 ms` is not met by HLS force-mode evidence: Search max `15.879 ms`.
- [ ] Track latency target `1 ms` is not met by HLS force-mode evidence: Track max `2.100 ms`.
- [ ] Physical ZCU104 Search/Track JSON is still missing, so measured p95/p99 latency, measured DMA bandwidth, and measured board power remain pending.

## 2026-06-29 Structured-ROM 300MHz Latency DSE Probe

- [x] Kept the user-approved experiment rule: PL clock `300 MHz`, routed WNS accepted down to `-0.500 ns` for experiment continuation only.
- [x] Added structured-ROM DSE profiles for combined, Search-only, and Track-only force-mode runs.
- [x] Search-only CSim passed with selected structured parameters: output `[1, 1, 1, 1, 1, -10]`, `runtime_state=0`, vector comparison pass, and prefetch trace violations `0`.
- [x] Track-only CSim passed with selected structured parameters: output `[0, 0, 0, 0, 0, 8]`, `runtime_state=1`, and vector comparison pass.
- [x] Search-only CSynth passed at target `3.333 ns`: estimated clock `2.777 ns`, latency `24,886` cycles / `82.945 us`, interval `24,887`, resources BRAM_18K `20`, DSP `2`, FF `8,372`, LUT `38,521`, URAM `0`.
- [x] Track-only CSynth passed at target `3.333 ns`: estimated clock `2.777 ns`, max latency `24,731` cycles / `82.428 us`, interval max `24,732`, resources BRAM_18K `20`, DSP `2`, FF `8,367`, LUT `38,522`, URAM `0`.
- [x] Structured-ROM selected-parameter HLS latency targets pass: Search `0.083 ms < 4 ms`, Track `0.083 ms < 1 ms`, hybrid 10/90 max-latency weighted average `0.08248 ms`.
- [ ] This is not dense arbitrary-weight full Transformer signoff; HLS optimized the selected structured parameter contract into a very small datapath.
- [ ] Structured-ROM Vivado implementation, board Search/Track JSON, p95/p99 latency, measured DMA bandwidth, and measured board power remain pending.

## 2026-06-29 Dense Search Dispatcher Prefetch-All4 Probe

- [x] Added configurable dispatcher bank count: `HGTXR_E2E_DISPATCH_PREFETCH_BANKS`, default `2`.
- [x] Added prefetch-all-before-compute mode: `HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE`, default `0`.
- [x] Added strict CSim trace guard: `HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START`.
- [x] Added Search-only dense objective profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only`.
- [x] Search-only CSim passed with full learned dense path and strict scheduler trace: dispatcher blocks `2` and `3` prefetched before compute, block interconnect trace `immediate=1` for `0->1`, `1->2`, and `2->3`, summary `violations=0 not_ready=0 immediate_gaps=0`.
- [x] Search-only CSim output was `[-1169, -1169, -1169, -1169, -1169, -1125]`, `runtime_state=0`, vector comparison pass, and `CSim done with 0 errors`.
- [x] CSynth passed at target `3.333 ns`, estimated clock `2.777 ns`, Search max latency `4,757,273` cycles / `15.856 ms`, interval max `4,757,274`.
- [x] CSynth storage proof captured: `gb_dispatch_prefetch_U` implemented as `ram_2p` / `impl=uram`, shape `255 x 27696 x 1`, estimated URAM `28`.
- [x] Vivado route/bitgen completed for the same profile at PL `300 MHz`: final post-route physopt WNS `-0.142 ns`, TNS `-69.079 ns`, WHS `0.006 ns`, THS `0.000 ns`, route errors `0`, bitgen pass.
- [x] Physical placed utilization fits ZCU104 despite the pessimistic HLS LUT estimate: CLB LUTs `53,926/230,400 = 23.41%`, CLB registers `46,812/460,800 = 10.16%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,592/1,728 = 92.13%`.
- [x] The profile passes the user-approved 300 MHz experiment threshold: WNS `-0.142 ns >= -0.500 ns`.
- [ ] The profile is not official clean Vivado timing signoff: timing constraints are not met because WNS is still negative.
- [ ] Search latency remains above the broader `4 ms` target: `15.856 ms`.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 300MHz Probe

- [x] Added combined runtime HLS profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16` with four Search dispatcher prefetch banks and strict immediate-start CSim tracing.
- [x] Added matching Vivado profile/artifact `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_overlay` at PL `300 MHz` with `Performance_ExplorePostRoutePhysOpt`.
- [x] Combined CSim passed for both runtime modes: Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track output `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, vector comparison pass.
- [x] Scheduler trace passed: Search dispatcher blocks `2` and `3` prefetched before compute, interblock starts `0->1`, `1->2`, and `2->3` all `immediate=1`; Search summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track summary `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Combined CSynth passed at target `3.333 ns`, estimated clock `2.777 ns`, top latency min `26,088` cycles / `86.951 us`, max `4,798,390` cycles / `15.993 ms`, interval max `4,798,391`.
- [x] Combined CSynth resource estimate captured: BRAM_18K `386/624 = 61%`, DSP `1,848/1,728 = 106%`, FF `242,912/460,800 = 52%`, LUT `400,236/230,400 = 173%`, URAM `76/96 = 79%`.
- [x] Vivado route/post-route physopt/bitgen completed for the combined runtime Top: `clk_pl_0 = 300.030 MHz`, WNS `-0.236 ns`, TNS `-718.875 ns`, WHS `0.001 ns`, THS `0.000 ns`, route errors `0`, bitgen pass.
- [x] Physical utilization fits ZCU104: CLB LUTs `73,133/230,400 = 31.74%`, CLB registers `60,949/460,800 = 13.23%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,723/1,728 = 99.71%`.
- [x] Routed vector-less power captured: total `8.364 W`, dynamic `7.626 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`.
- [x] The combined runtime Top passes the user-approved 300 MHz experiment threshold: WNS `-0.236 ns >= -0.500 ns`, hold clean, route errors `0`, bitgen pass.
- [ ] The combined runtime Top is not official clean Vivado timing signoff because WNS is negative and Vivado reports `Timing constraints are not met`.
- [ ] Latency targets remain open: combined HLS max latency `15.993 ms` still misses Search `4 ms`; board Search/Track p95/p99, measured DMA bandwidth, and measured power are still pending.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 Board Bundle/Gate

- [x] Added PYNQ runner variant `par32-prefetchall4-300` for the combined runtime `.bit/.hwh` artifact.
- [x] Added validator presets `axis-par32-prefetchall4-300-search` and `axis-par32-prefetchall4-300-track`.
- [x] Search board-smoke contract checks mode profile `search`, runtime state `0`, and expected raw output `[-1169, -1169, -1169, -1169, -1169, -1125]`.
- [x] Track board-smoke contract checks mode profile `track`, runtime state `1`, and expected raw output `[-235, -235, -235, -235, -235, -239]`.
- [x] Generated Search transfer bundle `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_search_smoke_bundle.tar.gz`, sha256 `34f76304a0ca41bed3b60bfad610898bb41bb7cfc6e698c7b71f072f4de98491`.
- [x] Generated Track transfer bundle `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_track_smoke_bundle.tar.gz`, sha256 `9b24856588b0ae691104aa79476bc1da80b8a3884a8d568d7e0660279a08d4e0`.
- [x] Bundle validation passed for both Search and Track with `errors=[]`.
- [x] Added board runner/gate profile set `par32-prefetchall4-300`; dry-run status is `dry-run`, gate status is `missing`.
- [x] Canonical board result destinations are fixed: `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json` and `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json`.
- [x] Focused tooling tests passed: `py_compile` and `63` unit tests across PYNQ validator, bundle packager, remote runner, import, and latency gate tools.
- [ ] Physical ZCU104 run JSON is still missing; SW/HW equality on board, p95/p99 latency, measured DMA bandwidth, and measured board power remain pending.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 RTL Cosim Attempt

- [x] Kept the experiment timing rule explicit: PL clock `300 MHz`, routed WNS acceptable down to `-0.500 ns` for continuation, official clean timing still requires WNS `>= 0.000 ns`.
- [x] Launched HLS RTL cosim for `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`.
- [x] Cosim reran CSynth at target `3.333 ns`; estimated clock remained `2.777 ns` / estimated Fmax `360.10 MHz`.
- [x] C testbench/post-check vectors passed before Verilog simulation: Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, and `E2E AXIS vector comparison passed`.
- [x] Scheduler trace remained clean inside the cosim flow: Search summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track summary `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [ ] Verilog RTL cosim failed: `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` with latency `NA`.
- [ ] Failure reason to isolate: XSIM reports `ERROR: unexpected exception when evaluating tcl command` while executing `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}`.
- [x] Added diagnostic tool `hardware/tools/diagnose_hls_cosim_xsim.py`.
- [x] Generated diagnosis artifacts: `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`.
- [x] Diagnosis status is `blocked-xsim-launch`; checks confirm C output/scheduler evidence is clean while RTL cosim pass remains `false`.
- [x] Extended `hardware/tools/check_xsim_snapshot_smoke.py` with alternate `xsim --runall`, `xsim --tclbatch`, and direct `xsimk` probes.
- [x] Minimal XSIM smoke still fails through the wrapper/Tcl launch path: standard and alternate `xsim` commands inject `-autoloadwcfg` and hit `unexpected exception when evaluating tcl command`.
- [x] Direct `xsimk` enters the simulator kernel and reports `elaboration-done`, but does not run the testbench or emit `XSIM_SMOKE_PASS`.
- [x] Regenerated XSIM smoke and HLS cosim diagnosis artifacts with the alternate-probe evidence.
- [ ] This is not RTL functional signoff yet; rerun or debug XSIM before claiming C/RTL same-input same-output.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 Resource Breakdown

- [x] Added `hardware/tools/write_prefetchall4_300_resource_breakdown.py`.
- [x] Added `hardware/vivado/scripts/report_e2e_axis_dma_impl_utilization.tcl` to regenerate implemented utilization, hierarchical utilization, timing, route, and power reports from an existing checkpoint.
- [x] Ran Vivado on the existing `hgtxr_e2e_axis_dma_system_wrapper_postroute_physopt.dcp` with Vivado local `LD_LIBRARY_PATH` workaround for `libtinfo.so.5`.
- [x] Generated `hardware/generated/signoff/prefetchall4_300_resource_breakdown_2026_06_29.json` and `.md`.
- [x] Resource breakdown status is `pass`; missing primary top resources `[]`, missing block reports `[]`.
- [x] Top placed utilization is captured as CLB LUTs `73,133/230,400 = 31.74%`, CLB registers `60,949/460,800 = 13.23%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,723/1,728 = 99.71%`.
- [x] Implemented hierarchical report is captured for `pl_hgtxr_e2e_axis_top`, `axi_dma_in`, `axi_dma_out`, `axi_mem_interconnect`, `axi_control_interconnect`, and `psu`.
- [x] `pl_hgtxr_e2e_axis_top` implemented hierarchy accounts for DSP `1,723` and URAM `76`, i.e. the DSP/URAM pressure is in the PL accelerator IP.
- [x] Goal-status now reads `prefetchall4_300_resource_breakdown_2026_06_29.json` and updates Q8 to `available`.
- [ ] Physical ZCU104 latency/DMA/power measurement remains pending; this resource breakdown closes only the resource-attribution part of the eight-question report.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 XSIM Direct Kernel Probe

- [x] Updated `hardware/tools/check_xsim_snapshot_smoke.py` to drive direct `xsimk` with MI commands after `elaboration-done`.
- [x] Minimal Verilog smoke now proves direct kernel execution: status `wrapper-blocked-kernel-pass`, `XSIM_SMOKE_PASS` observed, while standard `xsim` wrapper/Tcl launch still fails through `-autoloadwcfg`.
- [x] Added `hardware/tools/check_hls_xsimk_direct_probe.py` for the active `prefetchall4_300` HLS RTL snapshot.
- [x] HGTXR direct `xsimk` probe starts RTL simulation: kernel entered, `-exec-run` complete, `-exec-continue` started, HLS RTL progress banner observed, and `RTL Simulation : 0 / 2 [0.00%] @ "109000"` emitted.
- [x] Regenerated `prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.{json,md}` and `prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.{json,md}`.
- [ ] HGTXR direct `xsimk` probe did not finish both transactions in the bounded 120 s run, so Search/Track RTL output comparison is still missing.
- [ ] Standard HLS RTL cosim remains blocked by XSIM wrapper/Tcl launch; direct kernel execution is now the practical bypass candidate.

## 2026-06-29 Dense Runtime Search/Track Combined Prefetch-All4 Instrumented XSIMK Progress Probe

- [x] Added `hardware/tools/check_hls_xsimk_progress_probe.py`.
- [x] The tool copies `sim/verilog` plus sibling `sim/tv` to `/tmp/hgtxr_xsim_progress_probe_full_300`, patches only the copied HLS testbench `PROGRESS_TIMEOUT` from `10000000` to `100000`, reruns `xelab`, and runs direct `xsimk`.
- [x] Generated `hardware/generated/signoff/prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.json` and `.md`.
- [x] Instrumented 120 s run status is `progress-timeout`.
- [x] Snapshot rebuild passed and direct `xsimk` advanced RTL simulation inside transaction 0: first progress `0/2 0.00% @ 109000`, last progress `0/2 8.34% @ 1336125000`.
- [x] Goal-status now reports `progress_xsimk_status=progress-timeout` and includes the last progress sample.
- [ ] RTL output signoff is still pending because Search/Track expected RTL outputs were not observed before timeout.

## 2026-06-29 Aggressive DSP/LUTRAM Pressure Relief Probe

- [x] Added explicit LUTRAM binding switches for small/high-bank-count buffers: frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
- [x] Added HLS/Vivado profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
- [x] CSim passed for Search/Track expected vectors and strict prefetch trace: Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- [x] CSynth passed: target `3.333 ns`, estimated `2.777 ns`, max latency `4,807,861` cycles / `16.025 ms`, resources BRAM_18K `336`, DSP `1,698`, FF `247,663`, LUT `461,054`, URAM `28`.
- [x] Vivado route/post-route physopt/bitgen completed: route errors `0`, fully routed nets `201,673/201,673`, bitgen pass.
- [x] Implemented resource pressure relief confirmed: DSP `1,573/1,728 = 91.03%`, Block RAM Tile `184/312 = 58.97%`, URAM `28/96 = 29.17%`.
- [x] Implemented LUTRAM increase confirmed: CLB LUT `113,354/230,400 = 49.20%`, LUT as Distributed RAM `35,024`.
- [x] User-approved 300 MHz continuation floor passed exactly: post-route physopt WNS `-0.500 ns >= -0.500 ns`, WHS `0.006 ns`, route errors `0`.
- [x] Rejected a `tail2`/`DSP_MUL_LATENCY=6` timing-cleanup probe: CSim passed, but CSynth failed because HLS only accepts bound DSP multiply latency in `[0, 4]`.
- [ ] Official clean Vivado timing is still not met because WNS remains negative and TNS is `-6623.548 ns`.
- [ ] Promotion decision remains conditional: use this profile only when DSP/BRAM/URAM headroom matters more than LUT/TNS; keep `lutrom_tail2_dsppipe4` as the balanced pressure-relief candidate.

## 2026-06-29 Profile-Parametric Direct RTL Probe Tooling

- [x] Generalized `hardware/tools/check_hls_xsimk_direct_probe.py` with `--profile`, `--search-tail`, and `--track-tail`.
- [x] Generalized `hardware/tools/check_hls_xsimk_progress_probe.py` with matching profile and expected-tail arguments.
- [x] Preserved existing active-profile default outputs for `prefetchall4_300`; non-default profiles now write profile-specific signoff artifact names unless explicit output paths are passed.
- [x] `py_compile` passed for both probe tools.
- [x] `--help` passed for both probe tools and shows the new arguments.
- [x] Active-profile smoke passed as expected bounded progress: direct XSIMK entered kernel, completed `-exec-run`, started `-exec-continue`, and emitted first RTL progress sample `0/2 [0.00%] @ 109000` under a 5 s timeout.
- [x] Missing-snapshot behavior validated for `tail16_lutbuf_dsppipe4`: status `missing-sim-root`, with `/tmp/hgtxr_progress_probe_tail16_missing.{json,md}` produced.
- [ ] This is not RTL output signoff; it only makes the direct XSIMK evidence path reusable for pressure-relief profiles once their `sim/verilog` snapshots exist.

## 2026-06-29 Tail2 DSP-Pipeline RTL Snapshot/Direct Probe

- [x] Ran `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16`.
- [x] C TB passed before RTL launch: Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, and `E2E AXIS vector comparison passed`.
- [x] Search dispatcher prefetch trace remained clean: `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
- [x] Verilog snapshot was built and `sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk` now exists for the `tail2_dsppipe4` profile.
- [x] Standard HLS Verilog cosim failure reproduced at XSIM wrapper launch; `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` with latency `NA`.
- [x] Direct XSIMK smoke generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_direct_probe_2026_06_29.{json,md}` with status `direct-kernel-timeout`.
- [x] Instrumented progress probe generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_progress_probe_2026_06_29.{json,md}` with status `progress-timeout`; RTL progress advanced to `8.31%` in `120 s`.
- [x] Long direct XSIMK run completed both HLS RTL transactions in `1538.798 s`, reaching `RTL Simulation : 2 / 2 [100.00%] @ 18092130000`.
- [x] Added TV-output comparison to `hardware/tools/check_hls_xsimk_direct_probe.py`, including `--compare-tv-only`, per-port C/RTL TV hashes, and first mismatch reporting.
- [x] Generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.{json,md}`.
- [ ] RTL output equality fails for the balanced pressure-relief profile: `axis_out_V_data_V` first payload mismatch is C `0x...fb6f` versus RTL `0x...0000`; keep/strb/last and `gmem_e2e_runtime` match after transaction-line normalization.
- [ ] This profile remains a valid routed resource/timing pressure-relief candidate, but not a functional RTL signoff candidate until the zeroed RTL AXIS data path is fixed and revalidated.

## 2026-06-29 Tail16 Aggressive DSP/LUTRAM RTL Snapshot/Direct Probe

- [x] Generated the HLS RTL snapshot for `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16` by running the profile through `cosim`.
- [x] C TB passed before the Verilog launch: Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, and `E2E AXIS vector comparison passed`.
- [x] Strict prefetch trace remained clean in the cosim flow: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] HLS memory binding evidence confirms nonlinear ROM LUTRAM plus targeted `tokens`/`gb.*` LUTRAM buffers for the small/high-bank-count memory move.
- [x] XELAB built the profile snapshot at `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`.
- [x] Standard HLS Verilog cosim still reproduces the host wrapper failure: `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}` reports `unexpected exception when evaluating tcl command`, and Verilog latency is `NA`.
- [x] Direct XSIMK short probe generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_direct_probe_2026_06_29.{json,md}` with status `direct-kernel-timeout`.
- [x] Direct short probe entered the kernel, completed `-exec-run`, started `-exec-continue`, and emitted first RTL progress `0/2 [0.00%] @ 109000`.
- [x] Instrumented 120 s progress probe generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_progress_probe_2026_06_29.{json,md}` with status `progress-timeout`.
- [x] Instrumented progress advanced inside transaction 0 from `0/2 [0.00%] @ 109000` to `0/2 [8.32%] @ 1336125000`; estimated full two-transaction runtime is about `2885 s`.
- [x] TV compare-only artifact generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.{json,md}`.
- [x] Long direct XSIMK run generated `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_direct_probe_3300s_2026_06_29.{json,md}`.
- [x] Long direct XSIMK completed both RTL transactions in `1546.076 s`, reaching `2/2 [100.00%] @ 18080694000`.
- [ ] Full RTL output equality fails for `tail16_lutbuf_dsppipe4`: `axis_out_V_data_V` first payload mismatch is C `0x...fb6f` versus RTL `0x...0003`.
- [x] Non-payload TV ports match after normalization: `axis_out_V_keep_V`, `axis_out_V_strb_V`, `axis_out_V_last_V`, and `gmem_e2e_runtime`.
- [ ] This profile remains a valid physical DSP/BRAM/URAM pressure-relief candidate, but not a functional RTL signoff candidate until the RTL AXIS TDATA/state-payload path is fixed and revalidated.
- [ ] Official clean Vivado timing remains pending even though the profile passes the user-approved experimental floor: routed WNS `-0.500 ns`, post-route physopt/implemented WNS `-0.487 ns`, official signoff still needs WNS `>= 0.000 ns`.

## 2026-06-29 Fixed-CSim DSP/LUTRAM Pressure Follow-up

- [x] Enabled fixed-point CSim type selection through `HGTXR_HLS_FIXED_CSIM`.
- [x] Changed the fixed/synthesis AXIS writer to emit raw fixed-point payload bits, aligning C TV payload representation with RTL payload representation.
- [x] Added compile-time constant scale helpers and replaced the repeated `/8`, `/16`, `/32`, `/64`, `/128`, and `HGTXR_E2E_ACC_SCALE` conversion paths that were creating avoidable divider/multiplier pressure.
- [x] Added `HGTXR_E2E_AVOID_RUNTIME_DIVIDERS` for pressure profiles and enabled it on the fixed `tail*_dsppipe4` / LUTRAM profiles.
- [x] Re-ran CSim for `tail16_lutbuf_dsppipe4`; Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, strict prefetch traces clean.
- [x] Re-ran CSynth for `tail16_lutbuf_dsppipe4`; target `3.333 ns`, estimated clock `2.846 ns`, controller max latency `5,178,119` cycles / `17.259 ms`.
- [x] Confirmed LUTRAM memory move in HLS evidence: nonlinear ROMs distributed, `tokens` and many `gb.*` 3072x8 high-bank memories mapped to LUTRAM; large 12288-word banks remain BRAM; dispatcher prefetch remains URAM.
- [ ] Fixed-Csim resource pressure is still not fit: BRAM_18K `416`, DSP `2369/1728 = 137.1%`, FF `331918`, LUT `640454/230400 = 278.0%`, URAM `28`.
- [ ] Remaining DSP pressure is dominated by `hgtxr_e2e_controller_run` (`DSP=2364`, `LUT=550818`), not by Conv/Event Conv/Head.
- [ ] One dynamic softmax divider remains in generated RTL (`sdiv_28ns_28ns_8_32_1`); remove or approximate it before a full no-divider RTL probe.
- [ ] Next resource DSE candidate: enable `HGTXR_E2E_CORE_ALL_FABRIC_MUL` or reduce parallelism/cache fanout, then rerun CSim/CSynth before spending time on long XSIMK.

## 2026-06-29 Core-All-Fabric / No-Forced-DSP / LUTRAM Pressure Probe

- [x] Added/updated profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16`.
- [x] Enabled core-wide fabric multiply policy: `HGTXR_E2E_CORE_ALL_FABRIC_MUL=1`.
- [x] Disabled explicit DSP binding for the candidate: `HGTXR_E2E_FORCE_DSP_MUL=0` and `HGTXR_E2E_FORCE_WIDE_DSP_MUL=0`.
- [x] Enabled LUTRAM migration for small/high-bank-count buffers: frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
- [x] Static checks passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh` and `git diff --check` on touched HLS/script files.
- [x] CSim passed: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`.
- [x] Strict prefetch traces remain clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] CSynth passed at target `3.333 ns`, estimated clock `2.979 ns` (`335.63 MHz`).
- [x] LUTRAM movement is visible in the HLS memory report: many 3072x8 `tokens`/`gb.*` banks map to distributed RAM, while larger banks remain BRAM and dispatcher prefetch remains URAM.
- [ ] Candidate is not ZCU104-fit: BRAM_18K `416/624 = 66.7%`, DSP `2115/1728 = 122.4%`, FF `341844/460800 = 74.2%`, LUT `907718/230400 = 394.0%`, URAM `28/96 = 29.2%`.
- [ ] Major pressure remains in `hgtxr_e2e_controller_run`: DSP `2108`, LUT `817790`.
- [ ] `FORCE_DSP_MUL=0` did not lower HLS DSP estimate because wide multiply structures are still inferred as DSP by HLS/Vivado.
- [ ] Remaining generated softmax divider `sdiv_28ns_28ns_8_32_1` is still present.
- [ ] Next DSE should reduce multiply width / remove softmax divide / reduce parallelism or fanout, not keep expanding fabric mapping.

## 2026-06-29 Acc24 / Integer Nonlinear / LUTRAM Pressure Follow-up

- [x] Made `hgtxr_acc_t` macro-configurable with `HGTXR_ACC_W` and `HGTXR_ACC_I`.
- [x] Added `tail16_intnl_lutbuf_acc24_dsppipe4` and `coreallfabric_intnl_lutbuf_acc24` 300 MHz profiles.
- [x] Enabled accumulator narrowing to `ap_fixed<24,10>` for both profiles.
- [x] Enabled integer/LUT-oriented nonlinear paths for GELU, softmax, and layernorm quantization.
- [x] Kept the intended BRAM-to-LUTRAM move for small/high-bank-count frame and `gb.*` buffers.
- [x] CSim passed for both profiles with Search `[244, 244, 244, 244, 244, 242]`, Track `[200, 200, 200, 200, 200, 196]`, runtime states `0/1`, and clean strict prefetch summaries.
- [x] CSynth passed for `tail16_intnl_lutbuf_acc24_dsppipe4`: target `3.333 ns`, estimated `2.846 ns`, top max latency `8,611,527` cycles / `28.702 ms`.
- [x] CSynth passed for `coreallfabric_intnl_lutbuf_acc24`: target `3.333 ns`, estimated `2.846 ns`, top max latency `8,586,695` cycles / `28.619 ms`.
- [x] Confirmed LUTRAM/distributed RAM evidence for nonlinear ROMs and selected small/high-bank buffers.
- [ ] `tail16_intnl_lutbuf_acc24_dsppipe4` is not ZCU104-fit: DSP `2323/1728 = 134%`, LUT `621865/230400 = 269%`.
- [ ] `coreallfabric_intnl_lutbuf_acc24` is not ZCU104-fit: DSP `2067/1728 = 119%`, LUT `721301/230400 = 313%`.
- [ ] Do not promote either new HLS point to Vivado implementation; next DSE must reduce parallelism/fanout or restructure QKV/MLP weight-cache reuse.

## 2026-06-29 Par16/Mem8 Runtime-ROM Fit Probe

- [x] Added profile `par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8`.
- [x] Kept full runtime-ROM conditions active: on-chip parameter ROM, on-chip nonlinear ROM, Search weight dispatcher, prefetch-all-before-compute, runtime Search/Track scheduling, and learned Conv/Event Conv/Head path.
- [x] Reduced compute parallelism to `E2E_PAR=16` and memory-bank partitioning to `MEM_BANK_PAR=8`.
- [x] CSim passed for both runtime modes: Search `[244, 244, 244, 244, 244, 242]`, Track `[200, 200, 200, 200, 200, 196]`, runtime states `0/1`.
- [x] Search dispatcher prefetch remained ordered and immediate: `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Track resident path remained immediate: `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
- [ ] Candidate is not ZCU104-fit: BRAM_18K `400/624 = 64%`, DSP `2289/1728 = 132%`, LUT `485714/230400 = 210%`, URAM `108/96 = 112%`.
- [ ] Controller latency regressed to `9,616,135 cycles` / `32.051 ms`; top max latency is `13,077,605 cycles` / `43.588 ms`.
- [ ] Simple par16 shrink is rejected. Continue with par32 physical pressure-relief candidates for timing/RTL-equality cleanup, or redesign QKV/MLP weight-cache reuse before another parallelism shrink.

## 2026-06-29 Aux-Fabric / Norm-LUTRAM Pressure Probe

- [x] Added selective DSP-to-fabric switches for patch Conv, LayerNorm, and fastpath/data-observable multiply paths.
- [x] Added `HGTXR_E2E_LUTRAM_HIDDEN` binding support to AXIS and M_AXI E2E tops, but kept it disabled in the tested profile.
- [x] Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
- [x] Enabled token LUTRAM plus `gb.norm` LUTRAM for the profile; did not force Q/K/V/ATTN/hidden or weight caches into LUTRAM.
- [x] Static checks passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh` and Python compile of the xsim probe tools.
- [x] CSim passed with Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict prefetch summaries.
- [x] CSynth passed at target `3.333 ns`; estimated clock `3.167 ns` / `315.77 MHz`.
- [x] Selective conv DSP relief worked: Frame Conv and Event Conv estimate `DSP=0`.
- [x] URAM pressure improved versus prior `tail2_toklutbuf_dsppipe4`: `108 -> 92`.
- [ ] Candidate is not ZCU104-fit: BRAM_18K `416/624 = 66%`, DSP `2567/1728 = 148%`, LUT `564547/230400 = 245%`, URAM `92/96 = 95%`.
- [ ] LUT increased by `15203`, so the profile is useful evidence but not an implementation candidate.
- [ ] Next resource work should focus on controller QKV/MLP reuse/fanout, not moving more large buffers into LUTRAM.

## 2026-06-29 Weight-Cache / Tail8 Aux-Fabric Follow-up

- [x] Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16`.
- [x] Disabled QKV and WO/W1/W2 dense weight-vector caches for the `nocache` pressure probe.
- [x] Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
- [x] Enabled tail8 fabric-multiply policy plus patch/LayerNorm/fastpath fabric switches for the `tail8_lutbuf_auxfabric_normlut` probe.
- [x] Enabled LUTRAM for selected token-like high-bank buffers in the tail8 probe: frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
- [x] Static script check passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
- [x] `tail2_toklutbuf_nocache` CSim passed: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`.
- [x] `tail2_toklutbuf_nocache` strict prefetch traces passed: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] `tail2_toklutbuf_nocache` CSynth passed at target `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
- [x] `tail2_toklutbuf_nocache` reduced total BRAM_18K to `136/624 = 21%`.
- [ ] `tail2_toklutbuf_nocache` is not ZCU104-fit: DSP `2209/1728 = 127%`, LUT `523286/230400 = 227%`, URAM `108/96 = 112%`.
- [x] `tail8_lutbuf_auxfabric_normlut` CSim passed with the same Search/Track outputs and clean strict prefetch traces.
- [x] `tail8_lutbuf_auxfabric_normlut` CSynth passed at target `3.333 ns`; estimated clock `3.167 ns` / `315.77 MHz`.
- [ ] `tail8_lutbuf_auxfabric_normlut` is worse than existing `tail8_lutbuf_dsppipe4`: BRAM `336 -> 416`, DSP `1778 -> 2471`, LUT `441342 -> 615001`, top latency `16.025 ms -> 28.831 ms`.
- [ ] Do not promote either new follow-up profile to Vivado implementation.
- [ ] Next useful work is RTL AXIS `TDATA` mismatch debug for the physical par32 candidates plus QKV/MLP reuse/fanout restructuring.

## 2026-06-29 Tail16 AXIS Payload Contract Recheck

- [x] Added `hardware/tools/check_hls_axis_payload_contract.py` to audit generated AXIS payload structure and active low-bit C/RTL TV data.
- [x] Confirmed the old `tail2_dsppipe4` snapshot is stale for RTL data evidence: diagnostic status `fail-constant-axis-tdata`, with `axis_out_TDATA = 256'd0`.
- [x] Re-ran CSim for current `tail16_lutbuf_dsppipe4`.
- [x] Current `tail16_lutbuf_dsppipe4` CSim Search passed: `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`.
- [x] Current `tail16_lutbuf_dsppipe4` CSim Track passed: `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`.
- [x] Current CSim strict prefetch traces are clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Re-ran CSynth for current `tail16_lutbuf_dsppipe4`.
- [x] Current CSynth meets the 300MHz HLS estimate: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`.
- [x] Current generated `tail16` RTL structurally drives `axis_out_TDATA` from out_state-derived `zext_ln545*` sources.
- [x] Current generated `tail16` RTL has no constant-zero `axis_out_TDATA` assignment.
- [x] Current generated `tail16` out_state width is `[8]`, matching the `active64_b8_ff768` scale.
- [x] Wrote `hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_axis_payload_contract_2026_06_29.{json,md}`.
- [ ] Current `tail16` is still not resource-fit at HLS estimate: BRAM_18K `416/624 = 66%`, DSP `2369/1728 = 137%`, LUT `640454/230400 = 278%`, URAM `28/96 = 29%`.
- [ ] Dominant pressure remains `hgtxr_e2e_controller_run`: DSP `2364`, LUT `550818`.
- [ ] RTL functional signoff remains pending: current CSim/CSynth regeneration did not emit current RTL TV files.
- [ ] Remaining softmax divider `sdiv_28ns_28ns_8_32_1` is still generated; remove or approximate it before the next resource DSE.

## 2026-06-29 Tail16 Direct XSIMK Timeout / Partial TV Result

- [x] Re-ran HLS cosim for current `tail16_lutbuf_dsppipe4`.
- [x] HLS cosim C TB passed with Search `[3, 3, 3, 3, 3, 11]` and Track `[217, 217, 217, 217, 217, 213]`.
- [x] XELAB generated `sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`.
- [x] Standard XSIM wrapper still failed with `unexpected exception when evaluating tcl command`.
- [x] Ran direct `xsimk` MI probe for `3600 s`.
- [x] Direct probe entered the RTL kernel and reached RTL progress `1 / 2` transactions.
- [x] Direct probe artifact written: `hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_current_hls_xsimk_direct_probe_2026_06_29.{json,md}`.
- [x] Re-ran active 8-bit AXIS payload contract after the partial RTL run.
- [x] Search transaction active payload matches C: RTL `[3, 3, 3, 3, 3, 11]`.
- [ ] Full RTL functional signoff is still pending because Track transaction TV is missing after timeout.
- [ ] Current `tail16` remains not resource-fit by HLS estimate: DSP `2369/1728 = 137%`, LUT `640454/230400 = 278%`.
- [ ] Next resource work remains architectural: reduce QKV/MLP controller fanout/reuse or remove/approximate the softmax divider, not broader DSP-to-LUT/LUTRAM migration.

## 2026-06-29 Tail16 SoftmaxQ Divider-Removal Probe

- [x] Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`.
- [x] Isolated only `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1` on top of the existing `tail16_lutbuf_dsppipe4` pressure-relief profile.
- [x] Preserved runtime Search/Track scheduling, prefetch-all4, nonlinear LUTRAM, selected small/high-bank LUTRAM buffers, tail16 fabric lanes, and DSP pipe latency `4`.
- [x] CSim passed for both modes: Search `[0, 0, 0, 0, 0, 238]`, Track `[237, 237, 237, 237, 237, 233]`, runtime states `0/1`.
- [x] Strict prefetch traces remained clean for both modes: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
- [x] The prior `sdiv_28ns_28ns_8_32_1` artifact is absent from the generated project.
- [ ] Candidate is not a pressure-relief improvement: DSP `2369 -> 2371`, LUT `640454 -> 640308`, and top max latency `17.259 ms -> 28.731 ms`.
- [ ] Reject `tail16_softmaxq_lutbuf_dsppipe4` for Vivado promotion.
- [ ] Next useful resource work is controller QKV/MLP reuse/fanout restructuring. Broad DSP-to-LUT or large-memory-to-LUTRAM migration remains rejected by the measured DSE.

## 2026-06-29 Tail16 Shared Runtime Unit DSE

- [x] Added `HGTXR_E2E_SHARE_RUNTIME_UNITS` to force runtime ATTN/MLP calls through shared template unit `<0>`.
- [x] Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`.
- [x] Preserved the current full learned runtime-ROM candidate conditions: 300 MHz target, runtime Search/Track scheduling, Search prefetch-all4, nonlinear ROM, selected small/high-bank LUTRAM buffers, tail16 fabric lanes, and DSP pipe latency `4`.
- [x] CSim passed with unchanged baseline outputs: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`.
- [x] Strict prefetch traces remained clean for both modes: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] CSynth passed at target `3.333 ns`; estimated clock `2.846 ns` / `351.37 MHz`.
- [x] Confirmed duplicated `<1>` runtime ATTN/MLP template instances are absent from the shareunit generated reports/Verilog.
- [x] Confirmed `<0>` ATTN/MLP instances remain in `csynth_design_size.rpt`.
- [x] Major pressure improved without latency penalty: BRAM_18K `416 -> 276`, DSP `2369 -> 1057`, FF `331918 -> 165590`, LUT `640454 -> 372037`, URAM `28 -> 28`.
- [x] DSP is now under ZCU104 capacity at HLS estimate: `1057/1728 = 61%`.
- [ ] Candidate is still not ZCU104-fit by HLS LUT estimate: `372037/230400 = 161%`.
- [ ] Promote this as the primary HLS resource-relief candidate, but do not mark final hardware complete until controller LUT/fanout is reduced or Vivado implementation proves physical fit.

## 2026-06-29 Shareunit LUT Relief Follow-up

- [x] Added `HGTXR_E2E_HEAD_PAR` to decouple Head parallelism from Transformer core dense parallelism.
- [x] Updated `hgtxr_e2e_mlp_head` to use `kHeadPar` for local lane arrays, loop stride, and reduction.
- [x] Added `lutbuf_shareunit_dsppipe4`, `lutrom_shareunit_dsppipe4`, `lutbuf_shareunit_headpar8_dsppipe4`, `normlut_shareunit_headpar8_dsppipe4`, and `normlut_shareunit_headpar8_nocache_dsppipe4` profiles.
- [x] `lutbuf_shareunit_dsppipe4` CSim passed with Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, and clean strict prefetch traces.
- [x] `lutbuf_shareunit_dsppipe4` CSynth passed: BRAM_18K `276/624 = 44%`, DSP `1177/1728 = 68%`, LUT `339682/230400 = 147%`, URAM `28/96 = 29%`.
- [x] `lutrom_shareunit_dsppipe4` CSim passed with the same Search/Track outputs and clean strict prefetch traces.
- [x] `lutrom_shareunit_dsppipe4` CSynth passed: BRAM_18K `340/624 = 54%`, DSP `1185/1728 = 68%`, LUT `294575/230400 = 127%`, URAM `108/96 = 112%`.
- [x] Reject `lutrom_shareunit_dsppipe4` as a fit candidate because URAM exceeds capacity.
- [x] `lutbuf_shareunit_headpar8_dsppipe4` CSim passed with the same Search/Track outputs and clean strict prefetch traces.
- [x] `lutbuf_shareunit_headpar8_dsppipe4` CSynth passed: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`.
- [x] Head instance improved from LUT `31259` to `7659`; Head DSP dropped from `2` to `0`.
- [x] `lutbuf_shareunit_headpar8_dsppipe4` improved Head but still overuses LUT: BRAM_18K `276/624 = 44%`, DSP `1183/1728 = 68%`, FF `157807/460800 = 34%`, LUT `313925/230400 = 136%`, URAM `28/96 = 29%`.
- [x] `normlut_shareunit_headpar8_dsppipe4` CSim passed with the same Search/Track outputs and clean strict prefetch traces.
- [x] `normlut_shareunit_headpar8_dsppipe4` CSynth passed: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`.
- [x] Selective LUTRAM binding improved memory pressure: memory LUT `43008 -> 6144`, URAM `28 -> 92`, still within `96`.
- [x] `normlut_shareunit_headpar8_dsppipe4` improved memory pressure but still exceeded LUT: BRAM_18K `340/624 = 54%`, DSP `1183/1728 = 68%`, FF `156547/460800 = 33%`, LUT `277119/230400 = 120%`, URAM `92/96 = 95%`.
- [x] `normlut_shareunit_headpar8_nocache_dsppipe4` CSim passed with the same Search/Track outputs and clean strict prefetch traces.
- [x] `normlut_shareunit_headpar8_nocache_dsppipe4` CSynth passed: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`.
- [x] Cache-removal reduced controller pressure: controller LUT `247909 -> 196336`, DSP `1180 -> 682`, BRAM `140 -> 95`.
- [x] `normlut_shareunit_headpar8_nocache_dsppipe4` is the first HLS ZCU104-fit candidate: BRAM_18K `295/624 = 47%`, DSP `685/1728 = 39%`, FF `104198/460800 = 22%`, LUT `225520/230400 = 97%`, URAM `92/96 = 95%`.
- [x] Vivado implementation at 300 MHz completed for `normlut_shareunit_headpar8_nocache_dsppipe4`.
- [x] Bitgen completed and overlay artifacts were exported: `.bit` `19311230` bytes and `.hwh` `492672` bytes under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_overlay`.
- [x] Route completed with `115081/115081` routable nets fully routed and `0` routing errors.
- [x] Implemented timing at `300.030 MHz`: WNS `-0.171 ns`, TNS `-286.331 ns`, WHS `0.005 ns`, THS `0.000 ns`. Vivado reports timing not met, but this is within the user-allowed `-0.5 ns` WNS tolerance.
- [x] Implemented resources fit ZCU104: CLB LUTs `61679/230400 = 26.77%`, FF `42737/460800 = 9.27%`, Block RAM Tile `154.5/312 = 49.52%`, URAM `92/96 = 95.83%`, DSP `623/1728 = 36.05%`.
- [x] Implemented vector-less power was generated: total on-chip `6.642 W`, dynamic `5.917 W`, device static `0.725 W`, PS static `0.102 W`, PL static `0.623 W`.
- [x] Mode-specific Search-only force-mode CSynth completed for this exact implemented `nocache_dsppipe4` candidate: latency `8,604,864` cycles, interval `8,604,865` cycles, `28.683 ms` at 300 MHz, HLS estimated clock `2.846 ns` / `351.37 MHz`.
- [x] Mode-specific Track-only force-mode CSynth completed for this exact implemented `nocache_dsppipe4` candidate: min-bound latency/interval `698,166/698,167` cycles (`2.327 ms` at 300 MHz), max-bound latency/interval `4,003,638/4,003,639` cycles (`13.345 ms` at 300 MHz), HLS estimated clock `3.473 ns` / `287.94 MHz`.
- [x] Hybrid 10% Search / 90% Track estimate documented: using Track min-bound, expected interval `1,488,837` cycles / `4.963 ms` / `201.5 invocations/s`; using Track max-bound, expected interval `4,463,762` cycles / `14.879 ms` / `67.2 invocations/s`.
- [ ] Latency targets remain unmet for the exact implemented candidate: Search `28.683 ms > 4 ms`; Track min-bound `2.327 ms > 1 ms`; Track max-bound `13.345 ms > 1 ms`.
- [ ] Mode-specific SAIF/board power and repeated board latency distribution are still pending.

## 2026-06-29 Patch-parallel Conv/EventConv Probe

- [x] Added `HGTXR_E2E_PATCH_PAR` with default `1` so existing profiles preserve old serial patch accumulation.
- [x] Updated `hgtxr_conv_patch_embedding` and `hgtxr_event_conv_patch_embedding` to use partitioned patch accumulator lanes when `HGTXR_E2E_PATCH_PAR > 1`.
- [x] Added patch-parallel profiles with `HGTXR_E2E_PATCH_PAR=8` for combined, Search-only, and Track-only runs.
- [x] Combined patchpar8 CSim passed with unchanged outputs: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`.
- [x] Combined patchpar8 strict prefetch traces remained clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Combined patchpar8 CSynth passed: target `3.333 ns`, estimated `2.797 ns` / `357.46 MHz`, top max interval `7,094,388 cycles` / `23.648 ms` at 300 MHz.
- [x] Combined patchpar8 HLS resources remain within device estimate but are tight: BRAM_18K `295/624 = 47%`, DSP `706/1728 = 40%`, FF `105819/460800 = 22%`, LUT `228799/230400 = 99%`, URAM `92/96 = 95%`.
- [x] Search-only patchpar8 CSynth passed: interval `7,105,729 cycles`, `23.686 ms` at 300 MHz, estimated clock `2.797 ns` / `357.46 MHz`.
- [x] Track-only patchpar8 CSynth passed: min-bound interval `698,167 cycles` / `2.327 ms`, max-bound interval `2,467,639 cycles` / `8.225 ms` at 300 MHz.
- [x] Patchpar8 improves versus previous `nocache_dsppipe4`: Search interval `28.683 ms -> 23.686 ms`; Track max-bound interval `13.345 ms -> 8.225 ms`; Track min-bound unchanged at `2.327 ms`.
- [x] Hybrid 10% Search / 90% Track patchpar8 estimate documented: Track min-bound expected interval `1,338,923 cycles` / `4.463 ms`; Track max-bound expected interval `2,931,448 cycles` / `9.771 ms`.
- [ ] Patchpar8 still misses latency targets: Search `23.686 ms > 4 ms`; Track min-bound `2.327 ms > 1 ms`; Track max-bound `8.225 ms > 1 ms`.
- [ ] Track-only patchpar8 HLS estimated clock remains below 300 MHz: `3.473 ns` / `287.94 MHz`; physical Vivado timing is still unverified for this new probe.
- [ ] Next latency work should target `controller_run` and EventConv memory/loop structure, not only patch MAC parallelism.

## 2026-06-29 Token-loop Patch Embedding Probe

- [x] Added `HGTXR_E2E_PATCH_TOKEN_LOOP`, default `0`, so existing profiles preserve the old patch-grid loop.
- [x] Updated Conv/EventConv token-loop mode to iterate directly over `token < active_tokens` and derive `gy/gx` from the token id.
- [x] Added token-loop patchpar8 profiles for combined, Search-only, and Track-only runs.
- [x] Combined token-loop CSim passed with unchanged outputs: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`.
- [x] Combined token-loop strict prefetch traces remained clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Combined token-loop CSynth passed: target `3.333 ns`, estimated `2.797 ns` / `357.46 MHz`, top max interval `6,295,476 cycles` / `20.983 ms` at 300 MHz.
- [x] Combined token-loop HLS resources remain within device estimate but are tight: BRAM_18K `295/624 = 47%`, DSP `706/1728 = 40%`, FF `105817/460800 = 22%`, LUT `228745/230400 = 99%`, URAM `92/96 = 95%`.
- [x] Search-only token-loop CSynth passed: interval `7,105,729 cycles`, `23.686 ms` at 300 MHz, estimated clock `2.797 ns` / `357.46 MHz`.
- [x] Track-only token-loop CSynth passed: interval `1,143,415 cycles`, `3.811 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`.
- [x] Token-loop improves Track worst-case versus patchpar8: `2,467,639 cycles` / `8.225 ms` -> `1,143,415 cycles` / `3.811 ms`.
- [x] EventConv force-mode latency improved from `1,781,956 cycles` to `457,732 cycles`.
- [x] Hybrid 10% Search / 90% Track token-loop estimate documented: expected interval `1,739,646 cycles` / `5.799 ms` / `172.4 invocations/s`; worst-case remains Search `23.686 ms`.
- [ ] Token-loop still misses latency targets: Search `23.686 ms > 4 ms`; Track `3.811 ms > 1 ms`.
- [ ] Track-only token-loop HLS estimated clock remains below 300 MHz: `3.473 ns` / `287.94 MHz`; physical Vivado timing is still unverified for this probe.
- [ ] Next latency work should target the shared `controller_run` path and Search Conv, because Search remains dominated by `controller_run` `5.168M cycles` plus Conv `1.794M cycles`, and Track remains dominated by `controller_run` `628,712 cycles`.

## 2026-06-29 Patchpar16 Token-loop DSE

- [x] Added patchpar16 token-loop profiles for combined, Search-only, and Track-only runs.
- [x] Combined patchpar16 CSim passed with unchanged outputs: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`.
- [x] Combined patchpar16 strict prefetch traces remained clean: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Search-only patchpar16 CSynth passed: interval `5,692,609 cycles`, `18.975 ms` at 300 MHz, estimated clock `2.797 ns` / `357.46 MHz`.
- [x] Track-only patchpar16 CSynth passed: interval `796,290 cycles`, `2.654 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`.
- [x] Patchpar16 improved force-mode latency versus patchpar8 token-loop: Search `23.686 ms -> 18.975 ms` (`19.9%` faster), Track `3.811 ms -> 2.654 ms` (`30.4%` faster).
- [x] Combined patchpar16 CSynth passed but is not ZCU104-fit at HLS estimate: BRAM_18K `309/624 = 49%`, DSP `720/1728 = 41%`, FF `106881/460800 = 23%`, LUT `238345/230400 = 103%`, URAM `92/96 = 95%`.
- [x] Hybrid 10% Search / 90% Track patchpar16 estimate: expected interval `1,285,922 cycles` / `4.286 ms` / `233.3 invocations/s`; worst-case Search `18.975 ms`.
- [x] Tested `PATCH_PAR=12` as a temporary compile probe, but CSim compilation failed because `kPatchElems % kPatchPar != 0`; it is not legal under the current patch loop invariant and is not retained as a runnable profile.
- [ ] Patchpar16 is rejected as a direct Vivado candidate because combined LUT exceeds ZCU104 capacity.
- [ ] Current fit-capable candidate remains `patchpar8_tokenloop`; next work should reduce `controller_run` LUT/latency or reduce combined LUT enough to re-enable higher patch parallelism.

## 2026-06-29 Patchpar16 No-observe Norm-BRAM Fit DSE

- [x] Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` combined, Search-only, and Track-only profiles.
- [x] Disabled debug observable datapath for this DSE with `HGTXR_E2E_OBSERVE_DATAPATH=0`; this intentionally changes the observable debug mixed output word and requires a strict golden refresh before final functional acceptance.
- [x] Moved norm storage back to BRAM and reduced Head local parallelism to `HGTXR_E2E_HEAD_PAR=4` to recover LUT from rejected patchpar16.
- [x] Combined CSim passed with Search runtime state `0`, Track runtime state `1`, and clean strict prefetch traces: Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
- [x] Combined CSynth passed: target `3.333 ns`, estimated `2.777 ns` / `360.10 MHz`, interval min/max `124,486 / 5,617,495 cycles`, `0.415 / 18.725 ms` at 300 MHz.
- [x] Combined resources now fit at HLS estimate: BRAM_18K `341/624 = 54%`, DSP `719/1728 = 41%`, FF `105285/460800 = 22%`, LUT `222252/230400 = 96%`, URAM `92/96 = 95%`.
- [x] Search-only force-mode CSynth passed: interval max `5,604,440 cycles`, `18.681 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`.
- [x] Search path is still dominated by `controller_run` `5,169,698 cycles`; Conv is reduced to `380,932 cycles`.
- [x] Track-only force-mode CSynth passed: interval `753,383 cycles`, `2.511 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`.
- [x] Track path is still dominated by `controller_run` `628,902 cycles`; EventConv is reduced to `110,596 cycles`.
- [x] Hybrid 10% Search / 90% Track force-mode estimate: expected interval `1,238,489 cycles` / `4.128 ms` / `242.2 invocations/s`; worst-case Search `18.681 ms`.
- [x] Hybrid 10% Search / 90% Track combined-envelope estimate: expected interval `673,787 cycles` / `2.246 ms` / `445.2 invocations/s`; worst-case combined max `18.725 ms`.
- [x] Current best HLS-fit latency/resource candidate is now `patchpar16_tokenloop_noobs_normbram_headpar4`, not `patchpar8_tokenloop`.
- [x] Vivado package/route/post-route physopt/bitgen completed for the same candidate at requested `300 MHz`.
- [x] Final implemented timing is official-clean: `WNS 0.000 ns`, `TNS 0.000 ns`, `WHS 0.006 ns`, `THS 0.000 ns`; Vivado reports all user timing constraints are met.
- [x] Route is clean: routable nets `106,191`, fully routed nets `106,191`, routing errors `0`.
- [x] Overlay artifacts were exported under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/`.
- [x] Implemented resources fit ZCU104 with headroom except tight URAM: CLB LUT `48,909/230,400 = 21.23%`, CLB registers `44,104/460,800 = 9.57%`, Block RAM Tile `169.5/312 = 54.33%`, URAM `92/96 = 95.83%`, DSP `690/1,728 = 39.93%`.
- [x] Implemented vectorless power captured: total on-chip `6.412 W`, dynamic `5.688 W`, device static `0.724 W`, PS static `0.102 W`, PL static `0.622 W`.
- [x] Generalized `hardware/tools/write_prefetchall4_300_contract_audit.py` with `--profile` so structural objective-contract checks can target the current no-observe profile directly.
- [x] Active no-observe structural objective-contract audit generated: `hardware/generated/signoff/par32_patchpar16_noobs_normbram_contract_audit_2026_06_30.{json,md}`, status `pass`, checks `16/16`.
- [x] AXIS payload structural audit generated for the active no-observe profile: `hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_2026_06_30.{json,md}`.
- [x] AXIS payload audit status is `structural-pass-missing-rtl-tv`: synthesis RTL drives `TDATA` from 8-bit `out_state` zero-extension sources and has no constant-zero `TDATA` assignment.
- [x] Active no-observe HLS cosim rerun reached C TB pass: Search C TV payloads `[3, 3, 3, 3, 3, 3]`, Track C TV payloads `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, strict prefetch trace violations `0`.
- [x] Verilog snapshot was generated for the same profile, but standard HLS cosim failed before RTL transaction completion: `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` and latency/interval `NA`.
- [x] Direct `xsimk` probe entered the generated RTL snapshot and started RTL simulation progress (`0 / 2` transactions at sim time `109000`) before the `180 s` probe timeout.
- [x] Latest payload audit after direct probe generated: `hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_after_xsimk_2026_06_30.{json,md}`, status `fail-active-payload-tv` because RTL TV files are present but contain only the runtime marker and no payloads.
- [ ] Search still misses the `4 ms` target: `18.681 ms > 4 ms`.
- [ ] Force-mode Track still misses the `1 ms` target: `2.511 ms > 1 ms`.
- [ ] Track-only HLS estimated clock still misses 300 MHz: `3.473 ns` / `287.94 MHz`; combined profile and final Vivado overlay meet 300 MHz.
- [ ] Next acceptance work: complete RTL cosim or board same-input/same-output evidence for the no-observe output contract, plus board or SAIF-backed mode-specific power/latency measurement.

## 2026-06-30 Dense64 Controller DSE

- [x] Added dense64 variants of the current best patchpar16 no-observe norm-BRAM profile: combined, Search-only, and Track-only.
- [x] The dense64 variants preserve the full learned runtime-ROM objective path: on-chip parameter/nonlinear ROMs, Search dispatcher prefetch-all4, shared runtime units, `PATCH_PAR=16`, token-loop Conv/EventConv, observable datapath disabled, Head parallelism `4`, norm storage on BRAM, and `300 MHz` HLS target.
- [x] Combined dense64 CSim passed with Search output `[3, 3, 3, 3, 3, 3]`, Track output `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
- [x] Search-only dense64 CSynth passed: interval `5,833,024 cycles`, `19.443 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`.
- [x] Track-only dense64 CSynth passed: interval `781,851 cycles`, `2.606 ms` at 300 MHz or `2.715 ms` at estimated Fmax, estimated clock `3.473 ns` / `287.94 MHz`.
- [x] Dense64 increased resource pressure: Search-only resources BRAM_18K `339/624 = 54%`, DSP `955/1728 = 55%`, LUT `225097/230400 = 97%`, URAM `92/96 = 95%`; Track-only resources BRAM_18K `337/624 = 54%`, DSP `950/1728 = 54%`, LUT `221339/230400 = 96%`, URAM `71/96 = 73%`.
- [x] Dense64 is rejected as a latency candidate: Search worsened from `18.681 ms` to `19.443 ms`, and Track worsened from `2.511 ms` to `2.606 ms` at 300 MHz.
- [x] Root cause from controller reports: attention improves in Search (`288,732 -> 208,086 cycles`), but MLP worsens (`996,751 -> 1,134,543 cycles`), so global `kDensePar=64` increases mux/BRAM pressure instead of reducing the dominant MLP path.
- [ ] Next latency DSE should target MLP dataflow/weight-cache structure directly instead of raising global dense parallelism.

## 2026-06-30 MLP Token-parallel DSE

- [x] Added `HGTXR_E2E_MLP_TOKEN_PAR`, default `1`, so existing profiles preserve the original MLP dataflow.
- [x] Added a guarded MLP token-group branch that reuses one W1/W2 weight vector across two token lanes when `HGTXR_E2E_MLP_TOKEN_PAR=2`.
- [x] Added `mlptok2` variants of the current best patchpar16 no-observe norm-BRAM profile: combined, Search-only, and Track-only.
- [x] Combined `mlptok2` CSim passed with Search output `[3, 3, 3, 3, 3, 3]`, Track output `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
- [x] Search-only `mlptok2` CSynth passed: interval `6,153,432 cycles`, `20.511 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`.
- [x] Track-only `mlptok2` CSynth passed: interval `822,007 cycles`, `2.740 ms` at 300 MHz or `2.855 ms` at estimated Fmax, estimated clock `3.473 ns` / `287.94 MHz`.
- [x] `mlptok2` reduced top-level HLS area estimates versus dense64 and the active profile, especially LUT and DSP: Search-only `DSP 571`, `LUT 167,589`; Track-only `DSP 694`, `LUT 163,318`.
- [x] HLS inferred token-dimension cyclic factor `2` partitioning on `gb.norm` and `gb.hidden`, confirming the token-group branch materially changed memory structure.
- [x] `mlptok2` is rejected as a latency candidate: Search worsened from `18.681 ms` to `20.511 ms`, and Track worsened from `2.511 ms` to `2.740 ms` at 300 MHz.
- [x] Root cause from MLP reports: the grouped MLP loop lowers token trip count but raises iteration latency from `14,916` to `34,121 cycles`, so mux/memory pressure outweighs weight-vector reuse.
- [ ] Keep the active physical candidate as `patchpar16_tokenloop_noobs_normbram_headpar4`.
- [ ] Next latency DSE should avoid token-dim memory replication and instead target MLP loop scheduling at the W1/W2 inner-loop level or reduce full learned operator count per invocation.

## 2026-06-30 MLP Fused-W2 DSE

- [x] Added `HGTXR_E2E_MLP_FUSED_W2`, default `0`, so existing profiles preserve the original MLP dataflow.
- [x] Added a guarded fused MLP branch that computes hidden chunks and immediately accumulates W2 output into a local `out_acc` vector while still writing `gb.hidden` for compatibility.
- [x] Added `mlpfuse` variants of the current best patchpar16 no-observe norm-BRAM profile: combined, Search-only, and Track-only.
- [x] Combined `mlpfuse` CSim passed with Search output `[3, 3, 3, 3, 3, 3]`, Track output `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
- [x] Search-only `mlpfuse` CSynth passed: interval `4,488,024 cycles`, `14.960 ms` at 300 MHz or `19.337 ms` at estimated Fmax, estimated Fmax `232.10 MHz`.
- [x] Track-only `mlpfuse` CSynth passed: interval `613,831 cycles`, `2.046 ms` at 300 MHz or `2.645 ms` at estimated Fmax, estimated Fmax `232.10 MHz`.
- [x] `mlpfuse` improves cycle latency versus the active no-observe baseline: Search `5,604,440 -> 4,488,024 cycles` (`19.9%` fewer), Track `753,383 -> 613,831 cycles` (`18.5%` fewer).
- [x] `mlpfuse` reduces MLP unit latency versus the active profile: Search `996,751 -> 717,647 cycles`, Track `248,612 -> 178,836 cycles`.
- [x] Hybrid 10% Search / 90% Track force-mode estimate improves at ideal 300 MHz: active `1,238,489 cycles` / `4.128 ms`; `mlpfuse` about `1,001,250 cycles` / `3.338 ms`.
- [ ] `mlpfuse` is not promotable to the active physical baseline because HLS estimated Fmax drops to `232.10 MHz`, far below the `300 MHz` target.
- [ ] Next DSE should keep the fused W2 scheduling direction but tile or stage `out_acc` instead of complete `kEmbed` partitioning, to recover timing while preserving the cycle reduction.

## 2026-06-30 MLP Fused-W2 Banked-Accumulator DSE

- [x] Added `HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC`, default `0`, so existing profiles preserve the original `mlpfuse` accumulator partitioning unless the new macro is enabled.
- [x] Updated the fused-W2 branch so `out_acc` can use cyclic partitioning by `kDensePar` instead of complete `kEmbed` partitioning.
- [x] Added `mlpfusebank` variants of the current best patchpar16 no-observe norm-BRAM profile: combined, Search-only, and Track-only.
- [x] Combined `mlpfusebank` CSim passed with Search output `[3, 3, 3, 3, 3, 3]`, Track output `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
- [x] Search-only `mlpfusebank` CSynth passed: interval `4,490,328 cycles`, `14.968 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`.
- [x] Track-only `mlpfusebank` CSynth passed: interval `614,119 cycles`, `2.047 ms` at 300 MHz or `2.133 ms` at estimated Fmax, estimated clock `3.473 ns` / `287.94 MHz`.
- [x] `mlpfusebank` preserves the fused-W2 cycle gain while recovering Search timing versus complete-partition `mlpfuse`: Search estimated Fmax `232.10 MHz -> 360.10 MHz` with only `+2,304` interval cycles.
- [x] `mlpfusebank` improves top-level HLS area versus complete-partition `mlpfuse`: Search-only LUT `168,027 -> 156,534`, FF `75,299 -> 60,218`; Track-only LUT `163,787 -> 152,295`, FF `87,201 -> 72,120`.
- [x] Hybrid 10% Search / 90% Track force-mode estimate improves at ideal 300 MHz: active `1,238,489 cycles` / `4.128 ms`; `mlpfusebank` about `1,001,740 cycles` / `3.339 ms`.
- [x] Current best HLS latency direction is now `mlpfusebank`, not complete-partition `mlpfuse`.
- [x] `mlpfusebank` AXIS/DMA Vivado implementation completed at the requested `300 MHz` PL clock and generated `.bit/.hwh` artifacts.
- [x] `mlpfusebank` route is clean: routable nets `108,842`, fully routed nets `108,842`, routing errors `0`.
- [x] `mlpfusebank` implemented timing is acceptable under the user-relaxed WNS tolerance: clock `300.030 MHz`, WNS `-0.064 ns`, TNS `-39.952 ns`, WHS `0.004 ns`, THS `0.000 ns`. Vivado nominal timing is still reported as not met.
- [x] `mlpfusebank` implemented resources fit ZCU104: CLB LUT `49,022/230,400 = 21.28%`, CLB registers `45,244/460,800 = 9.82%`, Block RAM Tile `121.5/312 = 38.94%`, URAM `92/96 = 95.83%`, DSP `690/1,728 = 39.93%`.
- [x] `mlpfusebank` implemented vectorless power captured: total on-chip `6.231 W`, dynamic `5.509 W`, device static `0.722 W`, PS static `0.102 W`, PL static `0.620 W`, kernel hierarchy `hgtxr_e2e_axis_top_0 = 2.581 W`.
- [x] Added and measured `mlpfusebank_nohidden`, which disables the fused-W2 branch writeback to `gb.hidden` after local `hidden_vec` is available.
- [x] `mlpfusebank_nohidden` CSim passed with the same Search/Track outputs, runtime states, TLAST, and strict prefetch trace evidence as `mlpfusebank`.
- [x] `mlpfusebank_nohidden` Search-only CSynth is neutral: interval `4,490,328 cycles`, `14.968 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`, unchanged resources.
- [x] `mlpfusebank_nohidden` Track-only CSynth is neutral: interval `614,119 cycles`, `2.047 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`, unchanged resources.
- [ ] `mlpfusebank_nohidden` is not promoted or routed because it does not improve latency, resources, or timing over `mlpfusebank`.
- [ ] Search still misses the `4 ms` target: `14.968 ms > 4 ms`; Track still misses the `1 ms` target: `2.047 ms > 1 ms`.
- [ ] Same-input/same-output RTL or board validation remains open for `mlpfusebank`; current functional evidence is CSim plus structural/full-transformer path evidence.
- [ ] Next DSE should keep fused-W2 banked accumulation, add explicit MLP/DSP pipeline staging around the Vivado DRC MREG/PREG warnings, and reduce full learned operator work per invocation to close the Search `4 ms` and Track `1 ms` targets.

## 2026-06-30 HPar2 Physical Implementation

- [x] Added and packaged the `mlpfusebank_hpar2` full learned runtime-ROM profile for combined, Search-only, and Track-only force-mode evidence.
- [x] Search-only `mlpfusebank_hpar2` CSynth passed: interval `3,900,504 cycles`, `13.002 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`.
- [x] Track-only `mlpfusebank_hpar2` CSynth passed: interval `540,391 cycles`, `1.801 ms` at 300 MHz; Track estimated clock remains below nominal 300 MHz in HLS-only force-mode evidence.
- [x] Combined `mlpfusebank_hpar2` HLS package generated: top interval max `3,913,559 cycles`, `13.045 ms` at 300 MHz; HLS resources BRAM_18K `235`, DSP `751`, LUT `225,293`, URAM `92`.
- [x] `mlpfusebank_hpar2` improves force-mode latency versus `mlpfusebank`: Search `14.968 ms -> 13.002 ms`, Track `2.047 ms -> 1.801 ms`, hybrid 10/90 `3.339 ms -> 2.921 ms`.
- [x] `mlpfusebank_hpar2` Vivado route and bitgen completed at `clk_pl_0 = 300.030 MHz`.
- [x] `mlpfusebank_hpar2` route is clean: routable nets `110,620`, fully routed nets `110,620`, routing errors `0`.
- [x] `mlpfusebank_hpar2` implemented timing is acceptable under the user-relaxed WNS tolerance: post-route physopt WNS `-0.094 ns`, TNS `-107.248 ns`, WHS `0.007 ns`, THS `0.000 ns`. Vivado nominal timing is still reported as not met.
- [x] `mlpfusebank_hpar2` implemented resources fit ZCU104 and preserve compute fabric: CLB LUT `49,867/230,400 = 21.64%`, CLB registers `45,300/460,800 = 9.83%`, Block RAM Tile `121.5/312 = 38.94%`, URAM `92/96 = 95.83%`, DSP `722/1,728 = 41.78%`.
- [x] `mlpfusebank_hpar2` implemented vectorless power captured: total on-chip `6.231 W`, dynamic `5.509 W`, device static `0.722 W`, PS static `0.102 W`, PL static `0.620 W`.
- [x] `mlpfusebank_hpar2` `.bit/.hwh` artifacts are present in both `hardware/generated/build/vivado/overlay/...` and `hardware/pynq/hgtxr/`.
- [x] Fixed the AXIS/DMA Vivado TCL report-prefix path-length issue so long artifact names use a short report prefix for post-implementation report files.
- [ ] `mlpfusebank_hpar2` does not meet final latency targets: Search `13.002 ms > 4 ms`, Track `1.801 ms > 1 ms`.
- [ ] Same-input/same-output RTL or board validation remains open for `mlpfusebank_hpar2`; current evidence is HLS/CSim plus physical implementation.
- [ ] Mode-specific board latency distribution, DMA bandwidth, SAIF/board power, and p95/p99 evidence remain pending.

## 2026-06-30 Attention-Token2 / Attention-BRAM Physical DSE

- [x] Added and built the full learned runtime-ROM `attntok2_attnbram` profile:
  `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
- [x] Search-only force-mode HLS completed: estimated clock `2.777 ns`, II `2,623,832 cycles`, `8.745 ms` at 300 MHz.
- [x] Track-only force-mode HLS completed: estimated clock `3.473 ns`, II `380,711 cycles`, `1.322 ms` at 300 MHz.
- [x] Hybrid 10% Search / 90% Track estimate computed from force-mode II: `604,023 cycles`, `2.013 ms`, `496.7 invocations/s`.
- [x] Path breakdown captured:
  Search `axis_read_frame=16,386`, `conv_patch_embedding=380,932`, `global_buffer_load=12,292`, `controller_run=2,189,090`, `mlp_head=25,114`.
  Track `axis_read_frame=4,109`, `event_conv_patch_embedding=110,596`, `global_buffer_load=3,076`, `controller_run=256,230`, `mlp_head=6,682`.
- [x] Vivado implementation and bitgen completed; artifacts copied to overlay and `hardware/pynq/hgtxr/`.
- [x] Implemented timing captured: `clk_pl_0 = 300.030 MHz`, `WNS = -0.357 ns`, `TNS = -479.165 ns`, `4,384` setup failing endpoints, hold clean.
- [x] User-relaxed timing status is pass under the allowed `-0.5 ns` WNS tolerance; Vivado nominal timing remains not met.
- [x] Route clean: logical nets `936,375`, routable nets `133,219`, fully routed nets `133,219`, routing errors `0`.
- [x] Implemented resources captured: LUT `60,878 / 230,400 = 26.42%`, FF `55,536 / 460,800 = 12.05%`, BRAM Tile `137.5 / 312 = 44.07%`, URAM `76 / 96 = 79.17%`, DSP `946 / 1,728 = 54.75%`.
- [x] Implemented vectorless power captured: total `6.099 W`, dynamic `5.381 W`, static `0.719 W`, PS static `0.102 W`, PL static `0.617 W`; confidence `Medium`.
- [x] New best physically implemented full learned runtime-ROM latency candidate selected:
  Search improves from HPar2 `13.002 ms` to `8.745 ms`, Track from `1.801 ms` to `1.322 ms`, Hybrid from `2.921 ms` to `2.013 ms`.
- [ ] Search latency target still open: `8.745 ms > 4 ms`; needs about `1,423,832` fewer II cycles at 300 MHz.
- [ ] Track latency target still open: `1.322 ms > 1 ms`; needs about `80,711` fewer II cycles at 300 MHz.
- [ ] RTL/board same-input same-output validation and SAIF/board mode-specific power are still pending.

## 2026-06-30 HPar4 Fused-W2 Negative DSE

- [x] Added `hpar4_tok2_attntok2_attnbram` profiles to test `HGTXR_E2E_MLP_FUSED_W2_HP_PAR=4` on top of the current best Attention-Token2 / Attention-BRAM candidate.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Search-only CSynth completed: estimated clock `2.777 ns`, II `2,623,832 cycles`, `8.745 ms` at 300 MHz.
- [x] Track-only CSynth completed: estimated clock `3.473 ns`, II `380,711 cycles`, `1.322 ms` at 300 MHz.
- [x] Latency comparison is neutral versus promoted `hpar2` `attntok2_attnbram`: Search unchanged, Track unchanged, Hybrid unchanged at `604,023 cycles` / `2.013 ms`.
- [x] Resource comparison is worse: Search DSP `763 -> 891`, FF `75,794 -> 88,531`, LUT `188,424 -> 199,119`; Track DSP `886 -> 1,014`, FF `87,415 -> 100,152`, LUT `184,150 -> 194,845`.
- [x] Bottleneck evidence captured: HLS reports memory-port II violations around `w2_weight_cache` and global-buffer/token memories, so hidden-lane arithmetic expansion is not useful without matching memory banking.
- [x] Do not promote or route `hpar4`; keep the promoted `hpar2` `attntok2_attnbram` physical candidate as the current best.
- [x] Next DSE should target weight/global-buffer banking or controller work reduction, not further `HP_PAR` expansion.

## 2026-06-30 W2Bank8 Fused-W2 Negative DSE

- [x] Added `HGTXR_E2E_W2_WEIGHT_CACHE_BANKS` with default `1`, so existing promoted profiles are unchanged unless a profile opts into explicit W2 cache banking.
- [x] Added conditional cyclic partitioning for `w2_weight_cache` and `hpar4_w2bank8` combined/Search-only/Track-only profiles with `HGTXR_E2E_W2_WEIGHT_CACHE_BANKS=8`.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Search-only CSynth completed: estimated clock `2.777 ns`, II `2,623,836 cycles`, `8.745 ms` at 300 MHz.
- [x] Search path captured: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,094`, `mlp_head` `25,114`.
- [x] Resource regression captured: BRAM_18K `339`, DSP `891`, FF `90,532`, LUT `199,958`, URAM `76`; BRAM_18K is worse than `hpar4` Search-only `265`.
- [x] Bottleneck evidence captured: HLS applied `array_partition variable=w2_weight_cache cyclic factor=8 dim=1`, but the MLP W2 loop still reports `w2_weight_cache` memory-port II pressure and remains at achieved II `2`.
- [x] Do not promote or route `w2bank8`; the next candidate needs an explicit W2 data-layout/schedule change or controller work reduction.

## 2026-06-30 SkipHidden Fused-W2 Neutral DSE

- [x] Added `hpar2_skiphidden_tok2_attntok2_attnbram` combined/Search-only/Track-only profiles with `HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1`.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Search-only CSynth completed: estimated clock `2.777 ns`, II `2,623,832 cycles`, `8.745 ms` at 300 MHz.
- [x] Search path is unchanged versus promoted `attntok2_attnbram`: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090`, `mlp_head` `25,114`.
- [x] Resource estimate is unchanged versus promoted Search-only: BRAM_18K `265`, DSP `763`, FF `75,794`, LUT `188,424`, URAM `76`.
- [x] Track-only CSynth completed: estimated clock `3.473 ns`, II `380,711 cycles`, `1.322 ms` in the HLS report.
- [x] Track resource estimate is unchanged versus promoted Track-only: BRAM_18K `263`, DSP `886`, FF `87,415`, LUT `184,150`, URAM `55`.
- [x] Internal fused-W2 loop evidence captured: `VITIS_LOOP_2837_15_VITIS_LOOP_2839_16` shows achieved II `1`, but top/controller II is unchanged.
- [x] Do not promote or route `skiphidden`; remaining latency work must reduce higher-level controller work or reshape the actual limiting memory schedule.

## 2026-06-30 DenseToken4 Attention-Projection DSE

- [x] Added `mtok2_dtok4_attntok2_attnbram` combined/Search-only/Track-only profiles to raise dense projection token parallelism to `4` while keeping MLP token parallelism at `2`.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Search-only CSynth completed: estimated clock `2.777 ns`, II `2,465,752 cycles`, `8.218 ms` in the HLS report.
- [x] Search path captured: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,031,010`, `mlp_head` `25,114`.
- [x] Track-only CSynth completed: estimated clock `3.473 ns` / `287.94 MHz`, II `360,903 cycles`, `1.253 ms` in the HLS report.
- [x] Track path captured: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `236,422`, `mlp_head` `6,682`.
- [x] Hybrid 10% Search / 90% Track estimate improved to `571,388 cycles`, about `1.905 ms` at exact `300 MHz`, about `524.9 invocations/s`.
- [x] Latency improved versus promoted `attntok2_attnbram`: Search `2,623,832 -> 2,465,752 cycles` and Track `380,711 -> 360,903 cycles`.
- [x] Resource risk captured: Search-only LUT `212,438/230,400 = 92%`, DSP `1,019/1,728 = 58%`; Track-only LUT `208,129/230,400 = 90%`, DSP `1,142/1,728 = 66%`.
- [x] Combined package/CSynth completed and failed the ZCU104 fit gate: LUT `278,828/230,400 = 121%`, BRAM_18K `331/624 = 53%`, DSP `1,231/1,728 = 71%`, URAM `76/96 = 79%`.
- [x] Do not promote or route `dtok4` as-is; the force-mode latency improvement is outweighed by combined full-runtime LUT overflow.
- [x] Current physical baseline remains `attntok2_attnbram`.
- [ ] Final latency targets remain open against the best physical baseline: Search `8.745 ms > 4 ms`; Track `1.322 ms > 1 ms`.

## 2026-06-30 DenseToken3 Attention-Projection DSE

- [x] Added `mtok2_dtok3_attntok2_attnbram` combined/Search-only/Track-only profiles to test a smaller dense-token parallelism step after `dtok4` exceeded LUT budget.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Combined package/IP export completed.
- [x] Combined CSynth failed the ZCU104 fit gate: LUT `274,923/230,400 = 119%`, BRAM_18K `299/624 = 47%`, DSP `1,103/1,728 = 63%`, URAM `76/96 = 79%`.
- [x] Search-only/Track-only force-mode runs were intentionally skipped because the combined runtime profile already exceeds the device LUT budget.
- [x] Do not promote or route `dtok3` as-is; current physical baseline remains `attntok2_attnbram`.

## 2026-06-30 PatchPar32 Conv/EventConv DSE

- [x] Added `patchpar32_tok2_attntok2_attnbram` combined/Search-only/Track-only profiles with `HGTXR_E2E_PATCH_PAR=32`.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Search-only CSynth completed: estimated clock `2.777 ns` / `360.10 MHz`, II `2,500,952 cycles`, `8.336 ms`.
- [x] Search path captured: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `2,189,090`, `mlp_head` `25,114`.
- [x] Track-only CSynth completed: estimated clock `3.473 ns` / `287.94 MHz`, II `356,135 cycles`, HLS report latency `1.237 ms`, exact `300 MHz` latency about `1.187 ms`.
- [x] Track path captured: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
- [x] Hybrid 10% Search / 90% Track estimate improved to `570,617 cycles`, about `1.902 ms` at exact `300 MHz`, about `525.7 invocations/s`.
- [x] Latency improved versus promoted `patchpar16` `attntok2_attnbram`: Search `2,623,832 -> 2,500,952 cycles`, Track `380,711 -> 356,135 cycles`, Hybrid `604,023 -> 570,617 cycles`.
- [x] Combined CSynth completed: BRAM_18K `267/624 = 42%`, DSP `1,002/1,728 = 57%`, FF `126,407/460,800 = 27%`, LUT `258,869/230,400 = 112%`, URAM `76/96 = 79%`.
- [x] Package/IP export completed for the combined `patchpar32` runtime profile.
- [x] Added the combined `patchpar32` runtime profile to `run_e2e_axis_dma_vivado_no_board.sh`.
- [x] Vivado no-board route and bitgen completed for `hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay`.
- [x] Final implemented timing captured: WNS `-0.082 ns`, TNS `-60.586 ns`, WHS `0.002 ns`, THS `0.000 ns`. Nominal timing is not met, but this passes the user-relaxed WNS allowance of `-0.5 ns`.
- [x] Route status captured: logical nets `962,898`, routable nets `136,548`, fully routed nets `136,548`, routing errors `0`.
- [x] Implemented resource utilization captured: LUT `63,327/230,400 = 27.49%`, FF `57,218/460,800 = 12.42%`, BRAM Tile `137.5/312 = 44.07%`, URAM `76/96 = 79.17%`, DSP `973/1,728 = 56.31%`.
- [x] Implemented vectorless power captured: total `6.217 W`, dynamic `5.498 W`, static `0.720 W`, PS static `0.102 W`, PL static `0.618 W`.
- [x] Decision updated: `patchpar32` replaces `patchpar16` as the current best physically implemented full learned runtime-ROM baseline.
- [ ] Final latency targets remain open against the new physical baseline: Search `8.336 ms > 4 ms`; Track about `1.187 ms > 1 ms`.
- [ ] Mode-specific SAIF/board power and RTL/board same-input same-output validation remain pending.

## 2026-06-30 PatchPar32 DenseToken4 Combined DSE

- [x] Added `patch32_dtok4` combined/Search-only/Track-only no-board profiles by combining `HGTXR_E2E_PATCH_PAR=32`, `HGTXR_E2E_DENSE_TOKEN_PAR=4`, and `HGTXR_E2E_MLP_TOKEN_PAR=2`.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Search-only CSynth completed: estimated clock `2.777 ns` / `360.10 MHz`, II `2,342,872 cycles`, about `7.810 ms` at exact `300 MHz`.
- [x] Search path captured: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `2,031,010`, `mlp_head` `25,114`.
- [x] Track-only CSynth completed: estimated clock `3.473 ns` / `287.94 MHz`, II `336,327 cycles`, about `1.121 ms` at exact `300 MHz`.
- [x] Track path captured: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `236,422`, `mlp_head` `6,682`.
- [x] Hybrid 10% Search / 90% Track estimate improved to `536,981.5 cycles`, about `1.790 ms` at exact `300 MHz`, about `558.7 invocations/s`.
- [x] Combined CSynth completed: BRAM_18K `331/624 = 53%`, DSP `1,258/1,728 = 72%`, FF `147,675/460,800 = 32%`, LUT `282,684/230,400 = 122%`, URAM `76/96 = 79%`.
- [x] Added the combined `patch32_dtok4` runtime profile to `run_e2e_axis_dma_vivado_no_board.sh`.
- [x] Package/IP export completed: `generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` and `impl/export.zip`.
- [x] Vivado route/bitgen completed for `hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_overlay`: route errors `0`, fully routed nets `158,993/158,993`, bitgen pass.
- [x] Final post-route physopt timing meets the user-approved experiment floor but not official timing clean signoff: WNS `-0.238 ns`, TNS `-549.019 ns`, WHS `0.000 ns`, THS `0.000 ns`, setup failing endpoints `4,620`.
- [x] Implemented utilization: CLB LUT `73,832/230,400 = 32.05%`, CLB registers `69,126/460,800 = 15.00%`, Block RAM Tile `169.5/312 = 54.33%`, URAM `76/96 = 79.17%`, DSP `1,229/1,728 = 71.12%`.
- [x] Implemented vectorless power: total on-chip `7.150 W`, dynamic `6.423 W`, device static `0.727 W`; major dynamic buckets include PS8 `2.671 W`, DSPs `1.133 W`, signals `0.991 W`, CLB logic `0.734 W`, clocks `0.521 W`, Block RAM `0.176 W`, URAM `0.197 W`.
- [x] Latest structural contract audit generated for this exact candidate: `hardware/generated/signoff/patch32_dtok4_300_contract_audit_2026_06_30.{json,md}`, status `pass`, `16/16` checks.
- [x] Added PYNQ/Search/Track/Hybrid 10:90 board plumbing for this exact candidate under variant `par32-patch32-dtok4-300`.
- [x] Generated and validated Search, Track, and Hybrid 10:90 board-smoke bundles: `hardware/generated/signoff/par32_patch32_dtok4_300_{search,track,hybrid_10_90}_bundle_validation_2026_06_30.json`, all `pass`.
- [x] Generated ZCU104 dry-run remote plan: `hardware/generated/signoff/par32_patch32_dtok4_300_board_latency_run_2026_06_30.{json,md}`, status `dry-run`, profiles Search/Track/Hybrid 10:90.
- [x] Generated board latency gate artifact: `hardware/generated/signoff/par32_patch32_dtok4_300_board_latency_gate_2026_06_30.{json,md}`, status `missing` because physical board result JSONs are not present yet.
- [x] Latest goal-status artifact regenerated for this exact candidate: `hardware/generated/signoff/patch32_dtok4_300_goal_status_2026_06_30.{json,md}`, status `physical-structural-pass-latency-target-fail`, board plumbing `pass`.
- [ ] Final latency targets remain open: Search about `7.810 ms > 4 ms`; Track about `1.121 ms > 1 ms`.
- [ ] Physical ZCU104 Search/Track/Hybrid run is still pending; therefore board p95/p99, measured DMA bandwidth, and mode-specific measured latency remain unavailable.

## 2026-06-30 PatchPar32 DenseToken4 Board Contract Correction

- [x] Corrected the PYNQ/Search/Track/Hybrid expected-output contract for `par32-patch32-dtok4-300` to follow the combined runtime CSim board contract.
- [x] Search board expected raw output remains `[3, 3, 3, 3, 3, 3]`.
- [x] Track board expected raw output is `[217, 217, 217, 217, 217, 217]`.
- [x] Explicitly rejected the Track-only force-mode CSim output `[215, 215, 215, 215, 215, 215]` as a board-smoke expected vector because it is not the combined runtime bitstream contract.
- [x] Updated `package_e2e_axis_dma_pynq_bundle.py`, `validate_pynq_smoke_result.py`, `validate_pynq_bundle_package.py`, and `run_e2e_axis_dma_hybrid_smoke.py`.
- [x] Rebuilt Search, Track, and Hybrid 10:90 PYNQ bundles; tar SHA256 values are Search `8ffded397fb82d2996e74623328a66aca4291225c29484ad8143144e1cdb36d8`, Track `095c86b477254dcb3a2f9912aae85bbedf1196d3348b948f558705ec34e448cc`, Hybrid `c1d81829eb13a8276b518571f9325a8cc1d285e98a0b768a51a389200febcf4c`.
- [x] Re-ran bundle validation for Search, Track, and Hybrid 10:90; all three signoff JSONs are `pass`.
- [x] Re-ran `write_patch32_dtok4_300_goal_status.py`; status remains `physical-structural-pass-latency-target-fail`, with board plumbing `pass`.

## 2026-06-30 PatchPar32 MLPToken3/DenseToken4 DSE

- [x] Added `patch32_mtok3_dtok4` combined/Search-only/Track-only no-board profiles by setting `HGTXR_E2E_MLP_TOKEN_PAR=3` and keeping `HGTXR_E2E_DENSE_TOKEN_PAR=4`.
- [x] Combined CSim passed: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch-immediate traces clean.
- [x] Combined CSynth completed on 2026-07-01: estimated clock `2.777 ns` / `360.10 MHz`, top latency envelope `99,928-15,280,489 cycles`, top II envelope `99,929-15,280,490 cycles`, and top max latency `50.935 ms` at exact `300 MHz`.
- [x] Combined CSynth resource risk captured: BRAM_18K `347/624 = 55%`, DSP `1,354/1,728 = 78%`, FF `175,489/460,800 = 38%`, LUT `320,682/230,400 = 139%`, URAM `76/96 = 79%`.
- [x] Search-only CSynth completed: estimated clock `2.777 ns` / `360.10 MHz`, II `2,147,147 cycles`, about `7.157 ms` at exact `300 MHz`.
- [x] Search path captured: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,301`, `controller_run` `1,835,266`, `mlp_head` `25,124`.
- [x] Track-only CSynth completed: estimated clock `3.473 ns` / `287.94 MHz`, II `321,651 cycles`, about `1.072 ms` at exact `300 MHz`.
- [x] Track path captured: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,082`, `controller_run` `221,734`, `mlp_head` `6,688`.
- [x] Hybrid 10% Search / 90% Track estimate improved to `504,200.6 cycles`, about `1.681 ms` at exact `300 MHz`, about `595.0 invocations/s`.
- [x] Latency improves versus `patch32_dtok4`: Search `2,342,872 -> 2,147,147 cycles`, Track `336,327 -> 321,651 cycles`, Hybrid `536,981.5 -> 504,200.6 cycles`.
- [x] Resource risk captured: Search-only LUT `252,281/230,400 = 109%`, DSP `1,128/1,728 = 65%`; Track-only LUT `247,750/230,400 = 107%`, DSP `1,252/1,728 = 72%`.
- [x] Do not promote or route `mtok3_dtok4` as-is; its force-mode and combined CSynth LUT estimates exceed the ZCU104 budget, the combined worst-case envelope regresses badly, and Search still misses the `4 ms` target.
- [ ] Current physical baseline remains `patch32_dtok4`; final latency targets remain open: Search about `7.810 ms > 4 ms`, Track about `1.121 ms > 1 ms`.

## 2026-06-30 Initial Interval and AttentionQuery2 DSE

- [x] Calculated previous `par32_runtime_full_axi_mem16` II at `200 MHz`: Track `627,118 cycles` / `3.135590 ms`, Search `5,101,976 cycles` / `25.509880 ms`.
- [x] Calculated previous `par32_runtime_mode_mem16` II at `200 MHz`: Track `76,151 cycles` / `0.380755 ms`, Search `594,841 cycles` / `2.974205 ms`.
- [x] Documented that `runtime_mode_mem16` is a useful speed reference but is not the current full learned runtime-ROM Transformer path.
- [x] Added a real `HGTXR_E2E_ATTN_QUERY_PAR` knob with default `1` and a query-parallel attention-core branch for `ATTN_QUERY_PAR=2`.
- [x] Added `patch32_mtok4_dtok4_aq2` combined/Search-only/Track-only runner profiles.
- [x] Search-only `AQ2` CSynth completed: II `1,709,784 cycles`, about `5.699 ms` at exact `300 MHz`.
- [x] Search path captured: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,397,922`, `mlp_head` `25,114`.
- [x] Search resource risk captured: BRAM_18K `361/624 = 57%`, DSP `1,288/1,728 = 74%`, FF `133,107/460,800 = 28%`, LUT `264,616/230,400 = 114%`, URAM `92/96 = 95%`.
- [x] Track-only `AQ2` CSynth completed: II `268,711 cycles`, about `0.896 ms` at exact `300 MHz`.
- [x] Track path captured: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `168,806`, `mlp_head` `6,682`.
- [x] Track resource risk captured: BRAM_18K `359/624 = 57%`, DSP `1,412/1,728 = 81%`, FF `145,179/460,800 = 31%`, LUT `260,389/230,400 = 113%`, URAM `71/96 = 73%`.
- [x] Hybrid 10% Search / 90% Track estimate: II `412,818.3 cycles`, about `1.376 ms` at exact `300 MHz`, about `726.7 invocations/s`.
- [x] Current fastest HLS latency point is `AQ2`, but it is not promoted because Search still misses `4 ms` and HLS LUT exceeds ZCU104 capacity.

## 2026-06-30 AQ2 MLP Follow-up DSE

- [x] Added and tested `AQ2 + skip hidden store` Search-only profile.
- [x] `skiph` result showed no measurable improvement: Search II stayed `1,709,784 cycles`, MLP unit stayed `211,151 cycles`, resources stayed BRAM_18K `361`, DSP `1,288`, LUT `264,616`, URAM `92`.
- [x] Added and tested `AQ2 + HGTXR_E2E_MLP_FUSED_W2_HP_PAR=4` Search-only profile.
- [x] `hp4` result showed no latency improvement: Search II stayed `1,709,784 cycles`, controller stayed `1,397,922 cycles`, MLP unit stayed `211,151 cycles`.
- [x] `hp4` resource impact is negative: DSP increased to `1,544/1,728 = 89%`, LUT increased to `275,267/230,400 = 119%`.
- [x] Added `AQ2 + HGTXR_E2E_MLP_TOKEN_PAR=8` Search-only profile as a latency lower-bound probe.
- [x] Interrupted `mtok8` CSynth after HLS IR expansion made it unsuitable for fast DSE: `354,635` instructions after `Array/Struct`, with `hgtxr_e2e_mlp_unit<0>` at `276,469` instructions.
- [x] Do not infer latency/resource signoff from `mtok8`; C-synthesis did not complete.
- [x] Added and tested `AQ2 + full hidden-lane W2 bank cache`; Search II improved to `1,636,068 cycles` / `5.454 ms`, MLP unit `192,722 cycles`, but BRAM_18K exploded to `827/624 = 132%`.
- [x] Added and tested `AQ2 + hgroup-local W2 bank cache`; Search II improved to `1,625,268 cycles` / `5.418 ms`, controller `1,313,406 cycles`, MLP unit `190,022 cycles`.
- [x] Captured `w2hgroup` resource status: BRAM_18K `315/624 = 50%`, DSP `1,544/1,728 = 89%`, FF `152,242/460,800 = 33%`, LUT `273,839/230,400 = 118%`, URAM `92/96 = 95%`.
- [x] Added and tested `AQ4 + hgroup-local W2 bank cache`; Search II improved to `1,549,044 cycles` / `5.163 ms`, controller `1,237,182 cycles`, attention unit `112,332 cycles`, MLP unit `190,022 cycles`.
- [x] Captured raw `AQ4+w2hgroup` resource failure: BRAM_18K `315/624 = 50%`, DSP `1,672/1,728 = 96%`, FF `163,522/460,800 = 35%`, LUT `288,995/230,400 = 125%`, URAM `124/96 = 129%`.
- [x] Added and tested `AQ4 + hgroup-local W2 bank cache + Q/K/V/ATTN BRAM`; Search II stayed `1,549,044 cycles` / `5.163 ms`, while URAM improved from `124/96 = 129%` to `28/96 = 29%`.
- [x] Captured qkv-BRAM resource tradeoff: BRAM_18K increases to `443/624 = 70%`, DSP remains `1,672/1,728 = 96%`, LUT remains `288,995/230,400 = 125%`.
- [x] Decision: qkv-BRAM fixes AQ4 URAM overflow without hurting cycles, but it is still not promotable because LUT and DSP remain too high and Search still misses `4 ms`.
- [x] Ran Vivado no-board physical implementation for the qkv-BRAM Search-only profile.
- [x] Route and bitgen completed: fully routed nets `207,407/207,407`, route errors `0`, bitstream copied to overlay and PYNQ paths.
- [x] Final implemented timing captured: WNS `-0.527 ns`, TNS `-2247.483 ns`, WHS `0.006 ns`, THS `0.000 ns`, setup failing endpoints `17,645`.
- [x] The qkv-BRAM physical probe misses the user-approved `WNS >= -0.500 ns` experimental floor by `27 ps`; classify it as bitgen-complete but timing-failed.
- [x] Implemented resources captured: CLB LUT `95,920/230,400 = 41.63%`, CLB registers `98,554/460,800 = 21.39%`, Block RAM Tile `227/312 = 72.76%`, URAM `28/96 = 29.17%`, DSP `1,641/1,728 = 94.97%`.
- [x] Implemented vectorless power captured: total `7.723 W`, dynamic `6.996 W`, static `0.727 W`; dynamic hierarchy includes top IP `4.089 W`, PS `2.681 W`, AXI memory `0.183 W`, AXI control `0.020 W`, DMA out `0.018 W`, DMA in `0.005 W`.
- [x] Added and packaged `AQ4 + hgroup-local W2 bank cache + Q/K/V/ATTN BRAM + tail2 DSP relief` Search-only profile.
- [x] `qkvbram_tail2` HLS completed: Search II `1,549,428 cycles`, about `5.165 ms` at exact `300 MHz`; this is `384 cycles` slower than non-tail2 qkv-BRAM and still misses Search `4 ms`.
- [x] `qkvbram_tail2` HLS resources captured: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,220/460,800 = 36%`, LUT `312,155/230,400 = 135%`, URAM `28/96 = 29%`.
- [x] `qkvbram_tail2` Vivado route and bitgen completed: fully routed nets `208,988/208,988`, route errors `0`, `.bit/.hwh` copied to overlay and PYNQ paths.
- [x] `qkvbram_tail2` final postroute-physopt timing captured: WNS `-0.029 ns`, TNS `-0.365 ns`, setup failing endpoints `31`, WHS `0.002 ns`, THS `0.000 ns`.
- [x] `qkvbram_tail2` passes the user-approved `WNS >= -0.500 ns` experimental floor, while remaining negative-WNS and therefore not official clean Vivado timing signoff.
- [x] `qkvbram_tail2` implemented resources captured: CLB LUT `97,759/230,400 = 42.43%`, CLB registers `99,826/460,800 = 21.66%`, Block RAM Tile `227/312 = 72.76%`, URAM `28/96 = 29.17%`, DSP `1,553/1,728 = 89.87%`.
- [x] `qkvbram_tail2` implemented vectorless power captured: total `8.031 W`, dynamic `7.302 W`, static `0.729 W`; dynamic hierarchy includes top IP `4.398 W`, PS `2.678 W`, AXI memory `0.185 W`, AXI control `0.018 W`, DMA out `0.018 W`, DMA in `0.005 W`.
- [x] Decision: `qkvbram_tail2` becomes the best Search-only physical closure evidence in this branch because it reduces implemented DSP `1,641 -> 1,553` and improves WNS `-0.527 ns -> -0.029 ns`; it is not the final latency solution because Search remains about `5.165 ms > 4 ms`.
- [x] Added and tested `qkvbram_tail2 + PATCH_PAR=64`; CSim passed, but Search II worsened to `1,561,716 cycles` / `5.206 ms` because `conv_patch_embedding` increased to `270,340 cycles` under frame-memory port pressure. Decision: reject.
- [x] Added and tested `qkvbram_tail2 + HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1`; CSim passed and Search II improved to `1,512,564 cycles` / `5.042 ms`.
- [x] Captured `exppart` attention improvement: attention core latency `31,543 -> 22,327 cycles`, softmax exp loop instance `260 -> 68 cycles`, controller `1,237,566 -> 1,200,702 cycles`.
- [x] Captured `exppart` resources: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,307/460,800 = 36%`, LUT `312,706/230,400 = 135%`, URAM `28/96 = 29%`.
- [x] Decision: `exppart` is the best Search-only HLS latency point in the `qkvbram_tail2` physical-relief branch, but it is not routed/promoted yet because Search remains `5.042 ms > 4 ms` and HLS LUT remains over budget.
- [x] Added and tested `qkvbram_tail2 + exppart + skip hidden store`; CSim passed, but CSynth was identical to `exppart`: Search II `1,512,564 cycles` / `5.042 ms`, controller `1,200,702 cycles`, MLP unit `190,022 cycles`, resources BRAM_18K `443`, DSP `1,584`, FF `166,307`, LUT `312,706`, URAM `28`.
- [x] Decision: `exppart_skiph` is rejected as a no-op; the next useful DSE must change the MLP/controller body schedule or patch/conv path rather than only remove the hidden temporary store.
- [x] Added and tested `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`; CSim passed and CSynth improved Search II to `1,368,180 cycles` / `4.561 ms`.
- [x] Captured `w1c2` path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,056,318`, `mlp_head` `25,114`.
- [x] Captured `w1c2` MLP improvement: MLP unit `190,022 -> 153,926 cycles`; dominant fused W1 loop `VITIS_LOOP_3097_2` is `114,112 cycles`.
- [x] Captured `w1c2` resource status: BRAM_18K `449/624 = 71%`, DSP `1,704/1,728 = 98%`, FF `170,564/460,800 = 37%`, LUT `320,789/230,400 = 139%`, URAM `28/96 = 29%`.
- [x] Decision: `w1c2` is the fastest Search-only HLS point in this family, but it is not promotable as-is because Search still misses `4 ms`, DSP is nearly exhausted, and LUT remains far over budget.
- [x] Added and tested `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=3`; CSim passed, but CSynth regressed Search II to `1,417,332 cycles` / `4.724 ms`.
- [x] Captured `w1c3` failure mode: MLP unit worsened `153,926 -> 166,214 cycles`, dominant W1 loop worsened `114,112 -> 126,400 cycles`, DSP exceeded device capacity at `1,824/1,728 = 105%`, and LUT stayed over budget at `330,657/230,400 = 143%`.
- [x] Decision: reject `w1c3`; larger W1 channel grouping is not the next path unless the accumulation schedule/resource sharing is changed first.
- [x] Added and tested `qkvbram_tail4 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`; CSim passed and CSynth preserved Search II at `1,368,180 cycles` / `4.561 ms`.
- [x] Captured `tail4+w1c2` pressure tradeoff: DSP improves `1,704 -> 1,608`, but LUT worsens `320,789 -> 346,133` and Search II is unchanged. Decision: reject as a promotion candidate.
- [x] Added and tested `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2 + HGTXR_E2E_ATTN_QUERY_PAR=8`; CSim passed and CSynth improved Search II to `1,324,308 cycles` / `4.414 ms`.
- [x] Captured `AQ8+w1c2` attention gain: controller improves `1,056,318 -> 1,012,446 cycles`, attention unit `103,212 -> 92,244 cycles`, and attention core `22,327 -> 11,263 cycles`; MLP remains `153,926 cycles`.
- [x] Captured `AQ8+w1c2` resource failure: BRAM_18K `513/624 = 82%`, DSP `1,944/1,728 = 112%`, FF `198,021/460,800 = 42%`, LUT `359,425/230,400 = 156%`, URAM `28/96 = 29%`.
- [x] Decision: `AQ8+w1c2` is the fastest Search-only HLS latency point so far, but it is not promotable or routable as-is because Search remains `4.414 ms > 4 ms` and both DSP/LUT exceed ZCU104 capacity.
- [x] Added and tested `AQ8 + qkvbram_tail8 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`; CSim passed and CSynth preserved Search II at `1,324,308 cycles` / `4.414 ms`.
- [x] Captured `AQ8+tail8+w1c2` pressure tradeoff: DSP improves `1,944 -> 1,608` and returns under ZCU104 capacity at `93%`, but LUT worsens `359,425 -> 448,561` (`156% -> 194%`) with no latency gain. Decision: reject as a promotion candidate.
- [x] Added and tested `AQ6 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2`; CSim passed, but CSynth regressed Search II to `1,484,260 cycles` / `4.948 ms`.
- [x] Captured `AQ6` failure mode: controller `1,172,398 cycles`, attention unit `130,054-132,232 cycles`, MLP unit unchanged at `153,926 cycles`; resources BRAM_18K `481/624 = 77%`, DSP `1,824/1,728 = 105%`, LUT `348,116/230,400 = 151%`, URAM `28/96 = 29%`.
- [x] Decision: reject `AQ6`; it is slower than both `AQ4+w1c2` and `AQ8+w1c2` while still exceeding DSP/LUT capacity. Avoid non-power-of-two attention query parallelism unless the attention banking/schedule is redesigned.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + HGTXR_E2E_MLP_FUSED_W2_HP_PAR=8`; CSim passed and CSynth improved Search II to `1,288,836 cycles` / `4.296 ms`.
- [x] Captured `hpar8` MLP gain: MLP unit improves `153,926 -> 145,094 cycles`, saving `8,832 cycles` per runtime block and `35,328 cycles` over four Search blocks; MLP head improves only `25,114 -> 24,970 cycles`.
- [x] Captured `hpar8` resource failure: BRAM_18K `513/624 = 82%`, DSP `2,424/1,728 = 140%`, FF `211,693/460,800 = 45%`, LUT `385,749/230,400 = 167%`, URAM `28/96 = 29%`.
- [x] Decision: `hpar8` is now the fastest Search-only HLS latency point, but it is not promotable or routable as-is because it still misses `4 ms` and is far beyond DSP/LUT capacity.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + ACC24`; CSim passed and CSynth improved Search II to `1,284,676 cycles` / `4.282 ms`.
- [x] Captured `hpar8_acc24` pressure relief: versus direct `hpar8`, DSP `2,424 -> 2,410`, FF `211,693 -> 178,153`, LUT `385,749 -> 358,589`; BRAM_18K remains `513/624 = 82%`, URAM remains `28/96 = 29%`.
- [x] Decision: reject `hpar8_acc24` as a promotion candidate; it is the fastest Search-only HLS point so far, but Search still misses `4 ms` and DSP/LUT remain over capacity at `139%`/`155%`.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + ACC24 + HGTXR_E2E_FORCE_WIDE_DSP_MUL=0`; CSim passed and CSynth preserved Search II at `1,284,676 cycles` / `4.282 ms`.
- [x] Captured `hpar8_acc24_nowide` regression: DSP remains `2,410/1,728 = 139%`, FF changes slightly to `178,245`, and LUT worsens to `379,140/230,400 = 164%`.
- [x] Decision: reject `hpar8_acc24_nowide`; disabling the forced wide-DSP multiply path does not reduce DSP usage and increases LUT pressure. Next DSE should target controller/global-buffer memory-port pressure or real time-multiplexed sharing.
- [x] Added default-off `HGTXR_E2E_TOKEN_BANK_PAR` and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + ACC24 + token-bank4`; CSim passed with strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
- [x] `tokbank4` CSynth regressed Search II versus `hpar8_acc24`: `1,284,676 -> 1,294,276 cycles`, or about `4.282 -> 4.314 ms` at exact 300 MHz.
- [x] Captured `tokbank4` resource regression: BRAM_18K `545/624 = 87%`, DSP `2,410/1,728 = 139%`, FF `176,840/460,800 = 38%`, LUT `365,042/230,400 = 158%`, URAM `28/96 = 29%`.
- [x] Decision: reject `tokbank4`; token-dimension partitioning does not remove the remaining attention/MLP memory-port II=2 conflicts and worsens BRAM/LUT pressure.
- [x] Added and tested matching `hpar8` Track-only force-mode profile; CSim passed and CSynth reported Track II `229,783 cycles` / `0.765943 ms` at exact 300 MHz.
- [x] Captured matching `hpar8` Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `130,022`, `mlp_head` `6,538`.
- [x] Captured matching `hpar8` 10/90 pair estimate: Search `4.296117 ms`, Track `0.765940 ms`, mean latency `1.118958 ms`, mean II `1.118961 ms`, throughput `893.69 inv/s`, P95/P99 `4.296117 ms`.
- [x] Matching `hpar8` Track confirms Track is below `1 ms`, but the pair is still rejected for goal promotion because Search misses `4 ms` and resources exceed ZCU104 capacity: Track HLS DSP `2,420/1,728 = 140%`, LUT `379,171/230,400 = 164%`.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1`; CSim passed with prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
- [x] `hpar8_skiph` CSynth was identical to direct `hpar8`: Search II `1,288,836 cycles` / `4.296 ms`, controller `977,118 cycles`, attention unit `92,244 cycles`, MLP unit `145,094 cycles`.
- [x] Captured `hpar8_skiph` resource no-op: BRAM_18K `513/624 = 82%`, DSP `2,424/1,728 = 140%`, FF `211,693/460,800 = 45%`, LUT `385,749/230,400 = 167%`, URAM `28/96 = 29%`.
- [x] Decision: reject `hpar8_skiph`; it is functionally safe but does not reduce the controlling Search schedule or DSP/LUT pressure versus direct `hpar8`.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + HGTXR_E2E_MLP_W2_FABRIC_HP_LANES=2`; CSim passed, but CSynth regressed Search II to `16,615,339 cycles` / `55.384 ms`.
- [x] Captured `w2fl2` failure mode: controller max `16,266,742 cycles`, attention unit max `2,341,596 cycles`, MLP unit max `1,718,150 cycles`, and MLP loop `VITIS_LOOP_3114_2` up to `1,678,336 cycles`.
- [x] Captured `w2fl2` resource failure: BRAM_18K `515/624 = 82%`, DSP `2,410/1,728 = 139%`, FF `283,889/460,800 = 61%`, LUT `526,262/230,400 = 228%`, URAM `28/96 = 29%`.
- [x] Decision: reject `w2fl2`; it saves only `14` DSP versus direct `hpar8`, while adding `140,513` LUT and destroying the Search schedule.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + w1c2 + hpar8 + LN parameter cache`; CSim passed with prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
- [x] `lncache` CSynth regressed Search II to `1,298,192 cycles` / `4.327 ms`; controller `986,474 cycles`, attention unit `92,257 cycles`, MLP unit `147,420 cycles`.
- [x] Captured `lncache` resource failure: BRAM_18K `513/624 = 82%`, DSP `2,924/1,728 = 169%`, FF `265,062/460,800 = 57%`, LUT `450,440/230,400 = 195%`, URAM `28/96 = 29%`.
- [x] Decision: reject `lncache`; it is slower than direct `hpar8` and significantly worsens DSP/LUT pressure.
- [x] Added and tested `AQ8 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=4`; CSim passed with strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
- [x] `w1c4` CSynth preserved the `AQ8+w1c2` Search II at `1,324,308 cycles` / `4.414 ms`; controller `1,012,446 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`.
- [x] Captured `w1c4` resource failure: BRAM_18K `513/624 = 82%`, DSP `2,184/1,728 = 126%`, FF `212,462/460,800 = 46%`, LUT `375,182/230,400 = 162%`, URAM `28/96 = 29%`.
- [x] Decision: reject `w1c4`; it does not reduce MLP/body latency and worsens DSP/LUT pressure versus `AQ8+w1c2`.
- [x] Added default-off `HGTXR_E2E_W1_WEIGHT_CACHE_BANKS` and tested `w1c4_w1bank16`; CSim passed with strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
- [x] `w1c4_w1bank16` CSynth preserved the same Search II at `1,324,308 cycles` / `4.414 ms`; controller `1,012,446 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`.
- [x] Captured W1 banking failure evidence: the W1 loop remains final II `2` (`VITIS_LOOP_3168_9` latency `106 cycles`) and the scheduler still reports limited `w1_weight_cache` memory ports.
- [x] Captured `w1c4_w1bank16` resource regression: BRAM_18K `707/624 = 113%`, DSP `2,184/1,728 = 126%`, FF `212,833/460,800 = 46%`, LUT `376,481/230,400 = 163%`, URAM `28/96 = 29%`.
- [x] Decision: reject `w1c4_w1bank16`; it does not improve latency and increases BRAM by `194` BRAM_18K versus `w1c4`.
- [ ] Next DSE should preserve the `hpar8`/AQ8/w1c2 cycle reductions only if the added parallelism can be time-multiplexed or resource-shared; otherwise reduce MLP/body cycles and DSP/LUT pressure directly. Tail-lane, skip-hidden-store, partial W2 fabric mapping, and Track-only validation are not enough because the shared Search/Track candidate still fails Search latency and DSP/LUT fit.
- [x] Added and tested `AQ4 + hgroup-local W2 bank cache`; Search II improved further to `1,549,044 cycles` / `5.163 ms`, controller `1,237,182 cycles`, ATTN unit `112,332 cycles`, MLP unit `190,022 cycles`.
- [x] Captured `AQ4+w2hgroup` resource failure: BRAM_18K `315/624 = 50%`, DSP `1,672/1,728 = 96%`, FF `163,522/460,800 = 35%`, LUT `288,995/230,400 = 125%`, URAM `124/96 = 129%`.
- [x] Current fastest measured Search/Track HLS pair is `qkvbram_tail2 + exppart + w1c2 + AQ8 + hpar8`: Search `1,288,836 cycles` / `4.296 ms`, Track `229,783 cycles` / `0.766 ms`, 10/90 mean II `1.119 ms`; it is not promotable because Search remains above `4 ms` and DSP/LUT exceed ZCU104 capacity.
- [ ] Next implementation direction: reduce full learned operator/body work or lower DSP/LUT pressure in the MLP/attention datapath; raising attention query parallelism alone improves cycles but violates capacity.

## 2026-07-01 AQ2 Requested Metrics Refresh
- [x] Added request-specific report `hardware/docs/track/AQ2_REQUESTED_SEARCH_TRACK_METRICS_2026_07_01.md`.
- [x] Rechecked AQ2 Search/Track HLS reports for DMA ideal/effective bandwidth, HLS-envelope latency statistics, initiation interval, GOPS, AXIS width, and HLS major-block resource attribution.
- [x] Rechecked AQ2 implemented Vivado reports for physical resource utilization, hierarchy split, timing, route status, and vector-less power breakdown.
- [x] Current AQ2 Search: latency `5.699277 ms` max, initiation interval `1,709,784 cycles`, effective wire DMA bandwidth `0.0920 GB/s`, throughput `43.05 GOPS`, implemented total power `7.032 W`.
- [x] Current AQ2 Track: latency `0.895700 ms`, initiation interval `268,711 cycles`, effective wire DMA bandwidth `0.1465 GB/s`, throughput `33.81 GOPS`, implemented total power `6.777 W`.
- [x] Track final implemented timing remains inside the user-approved `WNS >= -0.500 ns` tolerance with WNS `-0.017 ns`; strict Vivado timing is still not clean for Track.
- [ ] Board-sampled mean/median/P95/P99 latency, measured DMA counters, SAIF/VCD power, external DDR/sensor I/O rail power, and Conv/ATTN/MLP/Head internal dynamic-power split remain additional-experiment items.

## 2026-07-08 IMPL_REPOS/HANDOVER Import
- [x] Imported XR_Accel HBTXR-specific architecture, ZCU104 status, validation,
      and JSON config materials under `hardware/docs/*/xr_accel` and
      `hardware/configs/xr_accel`.
- [x] Imported compact HGTXR handover evidence under
      `hardware/docs/resources/hgtxr_handover_additional`.
- [x] Imported ViT_Accel documentation as reference-only material under
      `hardware/docs/references/vit_accel`.
- [x] Imported ICCAD24 HG-PIPE README/header contracts as legacy hardware
      reference material under `references/legacy-codebase/hardware`.
- [x] Documented selection rules and exclusions in
      `hardware/docs/track/IMPL_REPOS_HANDOVER_IMPORT_2026_07_08.md`.
