> **작성** 2026-06-16 · **갱신** 2026-07-29
> **상태** frozen — 완료된 계획. 결과는 experiments/2026-06-16-third-goal-audit/
> **소유** hardware

# HGTXR Third-Goal Execution Notes

Date: 2026-06-15

## Executed
- Imported detailed ViT accelerator paper/codebase analysis into third-goal planning.
- Added priority mapping:
  - `HG-PIPE`: C-path dataflow/FIFO/pipeline hygiene.
  - `P2-ViT`: Q4/Q8 PoT scale calibration.
  - `ME-ViT`: single-load/on-chip buffer audit.
  - `ViTCoD` and `ViTALiTy`: software-first attention ablation.
  - `Edge-MoE`: bounded search/track routing ablation.
- Preserved C3b board-smoke gate and non-regression policy.
- Reran E2E resource policy audit: `generated/signoff/e2e_resource_policy_audit_2026_06_15.md`.
- Added VREF P0 execution plan: `generated/signoff/vref_p0_execution_plan_2026_06_15.md`.
- Spawned Spark explorer for VREF-P0-01 touchpoint review and folded the read-only findings into the execution plan.
- Added VREF-P0-01 PoT scale candidate sweep: `generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.md`.
- Spawned Spark explorer for VREF-P0-01 reduced-reference scale support and used it to confirm spec math fields, HLS macro mapping, and promotion invariants.
- Added VREF-P0-02 buffer lifetime/placement audit: `generated/signoff/vref_p0_buffer_lifetime_audit_2026_06_16.md`.
- Spawned Spark explorer for VREF-P0-02 buffer/resource review and used it to confirm large-buffer URAM and small-buffer LUTRAM/BRAM policy.
- Added C3b baseline protection checklist: `generated/signoff/c3b_protection_checklist_2026_06_16.md`.
- Folded Spark C3b signoff review into successor gates: no C3b overwrite, WNS/latency/resource ceilings, bit/hwh marker, physical smoke, and XR-VITs reference policy.
- Added VREF-P0-01 `softmax_input_x2` successor package with regenerated spec/header and passing E2E AXIS CSim: `generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md`.
- Added custom-scale csynth support and ran VREF-P0-01 `softmax_input_x2` E2E AXIS csynth in a separate `_csynth` project.
- Packaged the recommended `softmax_input_x2` `dsp_mixed_stream` successor as HLS IP:
  `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip/solution_e2e_q4w8a/impl/ip/component.xml`.
- Built and routed the matching Vivado AXI DMA overlay after applying the Ubuntu/Xilinx 2023.2 `libtinfo.so.5` runtime path:
  `LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:$LD_LIBRARY_PATH`.
- Produced routed overlay collateral:
  `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit`
  and matching `.hwh`, also copied under `pynq/hgtxr/`.
- Captured routed timing evidence: WNS `4.497 ns`, TNS `0.000 ns`, WHS `0.010 ns`, route errors `0`, fully routed nets `19829`.
- Added ZCU104 PYNQ transfer bundle and runbook for the successor:
  `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz`
  and `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md`.
- Generalized the ZCU104 SSH/SCP smoke runner profile path and emitted a successor dry-run remote plan:
  `generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.md`.
- Wired the successor physical-smoke result receiver into third-goal preflight and VREF-P0-01 successor projection.
  Missing successor board JSON now remains a promotion `pending` gate; a captured JSON is validated with preset
  `axis-vref-p0-softmax-input-x2-dsp-mixed-stream`.
- Refreshed the final unblock closeout packet with the VREF successor gate included as an optional promotion gate:
  `generated/signoff/final_unblock_closeout_packet_2026_06_16.md`.
- Added a policy switch for VREF successor physical smoke:
  `tools/write_final_unblock_closeout_packet.py --require-vref-successor-physical-smoke`.
  The default packet keeps VREF as optional successor promotion evidence; the VREF-required packet treats it as a hard blocker.
- Extended `tools/run_third_goal_final_signoff.py` with VREF successor smoke controls:
  `--require-vref-successor-physical-smoke`, `--execute-vref-successor-smoke`,
  `--import-vref-successor-smoke-json`, and `--vref-successor-remote-dir`.
  VREF remote smoke uses the profile default remote directory unless explicitly overridden.

## Current Execution Decision
Proceed in parallel where safe:
- Continue C3b board-smoke unblock when board access/result is available.
- Keep `VREF-P0-01` and `VREF-P0-02` non-invasive software/audit results as successor gates.
- Defer hardware promotion until C3b smoke evidence exists.
- Treat any VREF successor as blocked from replacing C3b unless it passes the C3b protection checklist and final-signoff external gates.

## Next Concrete Work
1. Capture/validate C3b physical board smoke JSON.
2. Resolve requested XR-VITs sibling or approved replacement policy.
3. Keep additional non-current `VREF-P0-01` scale candidates as successor experiments until golden headers are regenerated and CSim passes.
4. Only after exact-match passes, consider HLS constants or successor variant changes.

## VREF-P0-01 Result
- `VREF-P0-01` readiness audit status: pass.
- PoT scale candidate sweep status: pass.
- Specs evaluated: `3`.
- Candidates evaluated: `45`.
- Failed candidates: `0`.
- Current scale profile remains recommended for C3b.
- `softmax_input_x2` successor generated a dedicated spec/header and passed E2E AXIS CSim with expected raw output `[58, -51, 42, -28, 36, -41]`, `runtime_state=2`, and `CSim done with 0 errors`.
- `softmax_input_x2` E2E AXIS csynth passed: estimated clock `4.069 ns`, latency `498485` cycles, resources `64 BRAM_18K`, `128 DSP`, `19664 FF`, `43236 LUT`, `88 URAM`.
- `softmax_input_x2` `dsp_mixed_stream` resource-policy csynth passed with the same clock/latency/DSP/LUT and reduced URAM from `88` to `32` at BRAM cost `64 -> 144`.
- Recommended resource variant is now `dsp_mixed_stream`: HLS resource projection and routed timing clear C3b thresholds; physical smoke remains pending.
- Successor projection now consumes physical-smoke JSON automatically when copied to
  `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json`.
- Routed overlay result: WNS `4.497 ns` versus C3b protection threshold `>= 4.415 ns`.
- Successor board-smoke bundle is ready: expected raw `[58, -51, 42, -28, 36, -41]`, expected `runtime_state=2`, validator preset `axis-vref-p0-softmax-input-x2-dsp-mixed-stream`.
- Successor remote runner dry-run is ready; execute with `tools/run_zcu104_c3b_smoke_remote.py --profile vref-p0-softmax-input-x2-dsp-mixed-stream --host <zcu104-ip-or-host> --user xilinx --execute`.
- Resource-report caveat: top-level Vivado wrapper utilization can under-report HLS IP DSP/URAM for generated/OOC IP integration. Use HLS `csynth.xml` as the resource-policy evidence for this successor.
- Additional non-current scale rows remain successor candidates; each still requires its own regenerated golden header and CSim before HLS macro promotion.

## VREF-P0-02 Result
- `VREF-P0-02` static audit status: pass.
- Large buffers checked: frame/tokens/global/norm/Q/K/V/attention/hidden URAM candidates.
- Small buffers checked: pooled and attention-row scratch stay LUTRAM/BRAM candidates.
- QKV weight cache remains BRAM by default for the protected C3b path.
- Successor-only QKV weight-cache URAM experiment now has CSim and HLS csynth evidence:
  `generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.md`.
- QKV URAM result uses `dsp_mixed_stream + HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1`; CSim passes with expected raw `[58, -51, 42, -28, 36, -41]`.
- QKV URAM HLS csynth result: estimated clock `4.069 ns`, latency `498485` cycles, resources `114 BRAM_18K`, `128 DSP`, `19664 FF`, `43236 LUT`, `40 URAM`.
- Compared with `dsp_mixed_stream`, QKV URAM trades `-30 BRAM_18K` for `+8 URAM`, with unchanged DSP/LUT/latency and still below C3b HLS thresholds.
- QKV URAM promotion remains successor-only until routed overlay timing and physical smoke are captured.

## C3b Protection Result
- C3b protection checklist status: `ready-for-board-smoke`.
- Current checks: `30/32 pass`, `0 fail`, `2 pending`.
- Pending external gates: physical C3b smoke JSON and XR-VITs reference/replacement policy.
- Successor thresholds now captured: latency `<= 37508072`, WNS `>= 4.415 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`.
