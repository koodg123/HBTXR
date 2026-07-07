# Third Goal Requirement Mapping

Date: 2026-06-16

## Status Summary
- Updated third-goal objective is reflected in local hardware planning docs.
- C3b board-smoke remains the immediate hardware gate.
- VREF-P0 work proceeds without overwriting C3b.
- Default final closeout keeps VREF-P0 as optional successor promotion evidence.
- `--require-vref-successor-physical-smoke` switches VREF-P0 successor smoke into a hard final-signoff gate.

## Requirement Map
| ID | Requirement | Current Evidence | Status |
|---|---|---|---|
| 0 | Analyze plan/progress/HANDOVER/docs and continue work | `docs/Master-Plan.md`, `docs/Execution.md`, `docs/Validation.md`, `docs/track/PROGRESS.md`, `generated/signoff/third_goal_source_audit_2026_06_16.md`, final closeout packets, generated signoff traces | reflected |
| 1 | Ubuntu Linux and `/tools/Xilinx` path changes | `generated/signoff/req1_environment_audit_2026_06_16.md`; third-goal preflight checks Vitis/Vivado under `/tools/Xilinx` | reflected |
| 2 | Proper experiment plan/spec first, spec-kit where available, use Spark where possible | `generated/signoff/req2_spec_subagent_gate_audit_2026_06_16.md`; spec-kit unavailable, manual Spec fallback recorded, Spark-first/GPT5.5 fallback recorded | reflected |
| 3 | Prioritize ZCU104 cyclic hardware accelerator baseline | C3b remains selected board-smoke candidate; `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml` | reflected |
| 4 | E2E ViT model operation | E2E m_axi and AXIS/DMA tops, `generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.md`; bundle/session ready but canonical board JSON missing | partial; physical C3b smoke missing |
| 5 | Q4 weights and Q8 activations, arbitrary weights allowed if SW/HW match | Q4/Q8 config and packed weight manifest; Req5 Q4/Q8 SW-HW match audit `pass`, fail `0`; VREF-P0 PoT audit and scale sweep; `softmax_input_x2` SW/HW path expects raw `[58, -51, 42, -28, 36, -41]` | reflected |
| 6 | Parameterize tiling, parallelism, bus width, bit width, buffer size, FIFO depth | `configs/zcu104_e2e_q4w8a_defines.h`; Req6 parameterization audit `pass`, checks `64/64`; sweep covers PAR16/PAR32; Tcl overrides support PAR16/PAR32; preflight macro checks; closeout runner VREF smoke/import/remote-dir parameters | reflected |
| 7 | Analyze HG-PIPE and reflect for ZCU104 cyclic accelerator | `docs/SRC_CASE_MODULE_GUIDE.md`, `analysis/vit-accel/codebases/HG-PIPE/analysis.md` | reflected |
| 8 | Implement LayerNorm, GeLU, Softmax, Quantization from HG-PIPE paper/code | `hls/include/hgtxr_e2e_vit.hpp`, `hls/include/hgtxr_cyclic_math.hpp`, `generated/signoff/hgpipe_operator_audit_2026_06_16.md` | reflected |
| 9 | Refer to DeiT-Tiny C-Syn image | `generated/signoff/req9_deit_image_reference_audit_2026_06_16.md`; requested PAPER_PRJXR PNG, HGPIPE substitute, hardware docs copy, SHA256 match, and PNG magic checks pass | reflected |
| 10 | Refer to XR-VIT experiment results/code | `docs/legacy/legacy_experiment_analysis_2026_06_12.md`, `analysis/vit-accel`, `generated/signoff/xr_vits_gate_audit_2026_06_16.md` | partial; exact XR-VITs policy unresolved |
| 11 | Use `/home/kjm26/project/PRJXR/XR-VITs` HLS code for ZCU104 fit/low latency | `generated/signoff/xr_vits_gate_audit_2026_06_16.md`; exact path missing, `XR_Accel` candidate ready but approval required | blocked externally |

## Active Next Work
1. Keep C3b physical smoke as board-dependent blocker.
2. Resolve requested XR-VITs sibling or approved replacement policy.
   The active choice record is `docs/CHOICE.md`.
3. Promote only non-current `VREF-P0-01` scale successors that regenerate golden headers, pass CSim, and capture matching ZCU104 physical-smoke JSON.
4. Promote only `VREF-P0-02` memory successor variants that preserve C3b thresholds and add fresh csynth/routed evidence.

## Current Gate Modes
- Default closeout mode: `ready-for-operator-unblock` with 2 external blockers:
  `C3b AXIS/DMA physical smoke result` and `requested XR-VITs sibling`.
- VREF-required closeout mode: `ready-for-operator-unblock` with 3 external blockers:
  the default 2 blockers plus `VREF-P0 successor physical smoke result`.
- Final signoff runner supports both modes through
  `tools/run_third_goal_final_signoff.py --require-vref-successor-physical-smoke`.

## Latest Evidence
- Default closeout packet: `generated/signoff/final_unblock_closeout_packet_2026_06_16.md`.
- VREF-required closeout packet: `generated/signoff/final_unblock_closeout_packet_vref_required_2026_06_16.md`.
- VREF successor projection: `generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md`.
- VREF-P0-02 QKV URAM successor evidence:
  `generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.md`.
- VREF successor board runbook: `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md`.
- VREF successor remote dry-run: `generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.md`.
- C3b physical-smoke gate audit: `generated/signoff/c3b_physical_smoke_gate_audit_2026_06_16.md`.
- Req2 spec/sub-agent gate audit: `generated/signoff/req2_spec_subagent_gate_audit_2026_06_16.md`.
- Req5 Q4/Q8 SW-HW match audit: `generated/signoff/req5_q4q8_swhw_match_audit_2026_06_16.md`.
- Req6 parameterization audit: `generated/signoff/req6_parameterization_audit_2026_06_16.md`.
- Req9 DeiT image reference audit: `generated/signoff/req9_deit_image_reference_audit_2026_06_16.md`.
- HG-PIPE operator audit: `generated/signoff/hgpipe_operator_audit_2026_06_16.md`.
- XR-VITs gate audit: `generated/signoff/xr_vits_gate_audit_2026_06_16.md`.
- Third-goal source audit: `generated/signoff/third_goal_source_audit_2026_06_16.md`.
- Third-goal current audit: `generated/signoff/third_goal_current_audit_2026_06_16.md`.

## Remaining External Inputs
- C3b physical smoke JSON:
  `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Requested XR-VITs exact source path or approved replacement policy:
  `/home/kjm26/project/PRJXR/XR-VITs`.
- VREF successor physical smoke JSON, optional by default and required only in VREF-required gate mode:
  `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json`.

## Sub-agent Note
- The latest Spark follow-up agent hit the GPT-5.3-Codex-Spark quota limit before returning additional evidence.
- GPT5.5 fallback completed read-only XR-VITs verification and confirmed Req11 must remain blocked unless the exact `/home/kjm26/project/PRJXR/XR-VITs` checkout is restored or an explicit fingerprint-bound XR_Accel replacement policy is approved.
  Existing Spark outputs already reflected in `docs/Execution.md` remain valid; this document was updated from local evidence instead.
