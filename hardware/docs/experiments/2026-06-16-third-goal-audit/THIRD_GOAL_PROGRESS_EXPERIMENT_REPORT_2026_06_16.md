# HGTXR Third-Goal Progress And Experiment Report

Date: 2026-06-16

## Scope

This report consolidates work completed for the third goal: Ubuntu `/tools/Xilinx` migration, ZCU104 cyclic ViT accelerator baseline, Q4W/Q8A E2E execution path, HG-PIPE operator integration, resource-policy hardening, VREF successor experiments, final-signoff evidence, and remaining blockers.

Primary evidence:

- `docs/track/PROGRESS.md`
- `docs/track/HANDOVER.md`
- `docs/CHOICE.md`
- `../docs/resources/final_evidence_manifest_2026_06_10.json`
- `../docs/resources/e2e_resource_matrix_2026_06_10.json`
- `../docs/resources/third_goal_completion_audit_2026_06_10.json`
- `../docs/resources/third_goal_current_audit_2026_06_16.json`
- `../docs/resources/third_goal_source_audit_2026_06_16.json`

## Current Status

| Area | Status | Evidence |
|---|---:|---|
| Final evidence manifest | `pass`, `86/86`, failed consistency `[]` | `final_evidence_manifest_2026_06_10.json` |
| Spec/plan conformance | `pass`, `86/86` | `spec_plan_conformance_audit_2026_06_10.json` |
| Operator handoff validation | `pass`, `131/131` | `final_operator_handoff_validation_2026_06_10.json` |
| Final bundle validation | `pass`, `152/152` | `final_signoff_bundle_validation_2026_06_10.json` |
| Source audit | `pass`, required `86`, sources `224` | `third_goal_source_audit_2026_06_16.json` |
| Completion audit | `blocked`, pass `11`, partial `1`, blocked `2` | `third_goal_completion_audit_2026_06_10.json` |
| Current audit | `blocked-external`, reflected `10`, partial `1`, blocked `1` | `third_goal_current_audit_2026_06_16.json` |
| Final runner | `blocked`, blocker count `2` | `third_goal_final_signoff_run_2026_06_10.json` |

Final signoff is blocked only by external inputs:

- C3b AXIS/DMA canonical board smoke JSON:
  `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- Exact `/home/kjm26/project/PRJXR/XR-VITs` source or approved replacement policy:
  `docs/resources/xr_vits_replacement_policy.json`.

## Environment And Tooling

| Requirement | Result | Evidence |
|---|---|---|
| Ubuntu Linux environment | Checked and promoted to final evidence | `req1_environment_audit_2026_06_16.json`, `13/13 pass` |
| Xilinx path migration | `/tools/Xilinx` Vitis/Vivado path checked | `req1_environment_audit_2026_06_16.json` |
| Working directory migration | Hardware root under `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware` | `req1_environment_audit_2026_06_16.json` |
| spec-kit usage | `spec-kit`/`specify` not on PATH; manual Spec fallback recorded and gated | `spec_plan_conformance_audit_2026_06_10.json` |
| Spark-first subagent policy | Used where available; fallback to GPT5.5 when Spark limit hit | `docs/CHOICE.md`, `docs/track/HANDOVER.md` |

## Baseline E2E Resource Matrix

Current default board-smoke candidate: `C3b PAR16 MEM16 AXIS/DMA`.

| Variant | Latency cycles | Runtime | Parallelism | Banks | DSP | LUT | BRAM_18K | URAM | Routed WNS | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A2 E2E m_axi | `81,514,836` | `0.408 sec` | `8` | `8` | `334` | `84,220` | `326` | `64` | `1.926 ns` | bit/hwh ready |
| A1 E2E AXIS/DMA | `71,316,968` | `0.357 sec` | `8` | `8` | `332` | `81,168` | `298` | `64` | `4.120 ns` | bit/hwh ready |
| C1 PAR16 AXIS/DMA | `37,508,072` | `0.188 sec` | `16` | `8` | `604` | `127,916` | `338` | `64` | `4.723 ns` | bit/hwh ready |
| C3b PAR16 MEM16 AXIS/DMA | `37,508,072` | `0.188 sec` | `16` | `16` | `604` | `126,506` | `332` | `64` | `4.415 ns` | ready for board smoke |

Result:

- Best default latency: `C1` and `C3b`.
- Highest DSP use: `C1` and `C3b`, `604 DSP`.
- Best resource-balanced C-path: `C3b`, because it keeps PAR16 latency/DSP while reducing LUT vs `C1`.
- C3b vs A1: latency improves by about `47.41%`, DSP increases from `332` to `604`.

## E2E Q4W/Q8A Implementation Result

| Item | Result |
|---|---|
| Weight precision | Q4 packed weights |
| Activation precision | Q8 activations |
| Packed weight binary | `refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin` |
| Packed weight manifest | `refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json` |
| Expected C3b raw output | `[32, -13, 26, -6, 14, -11]` |
| Runtime state | `2` |
| SW/HW match audit | `req5_q4q8_swhw_match_audit_2026_06_16.json`, `11/11 pass` |
| Strict CSim/TB evidence | promoted to final evidence |

## Parameterization Result

Req6 parameterization audit is `pass`, `71/71`.

Covered knobs:

- Tiling factor and cyclic tile relationships.
- Parallelism factor, including validated PAR16 and exploratory PAR32.
- Bus width and packed lanes.
- Bit width and weight bit width.
- Buffer size.
- FIFO depth.
- Dense parallelism.
- AXIS width.
- Static-assert legality guards.
- Tcl override support for PAR16/PAR32.
- Promotion policy: PAR32 remains exploratory until fresh csynth, routed timing, resource-fit, and no-C3b-overwrite evidence exist.

## HG-PIPE Operator Integration

HG-PIPE analysis and operator implementation evidence is promoted to final evidence.

| Operator | Status |
|---|---|
| LayerNorm | `pass` |
| GeLU | `pass` |
| Softmax | `pass` |
| Quantization | `pass` |

Audit evidence:

- `hgpipe_operator_audit_2026_06_16.json`
- Reference checks: `97/97`
- Sampled values: `5,899,008`
- Deterministic property checks: `211/211`
- Failures: `0`

## Resource Policy Results

Resource policy audit is `pass`, `36/36`.

Implemented and gated:

- RMU/SMU multiply-heavy paths use DSP-bound helper functions.
- RMU/SMU small `score` and `prob` buffers use LUTRAM.
- Cyclic packed weight tiles and large temporaries use URAM through macros.
- Cyclic small tile scratch buffers use LUTRAM through macros.
- C3b LUT threshold: `<=126,506`.
- C3b latency threshold: `<=37,508,072 cycles`.
- C3b routed WNS threshold: `>=4.415 ns`.
- Resource-policy artifact singularity: only canonical `e2e_resource_policy_audit_2026_06_10.{json,md}` may exist in docs/resources and generated/signoff.

## VREF And Successor Experiment Results

### VREF-P0-01 PoT Scale Work

| Item | Result |
|---|---|
| PoT readiness audit | `pass`, `14/14` |
| Candidate sweep | `3` specs, `45` candidates, `0` fail |
| Decision | Keep current PoT scales for C3b |
| Non-current scale rows | Successor-only until regenerated header, CSim, HLS, and board evidence |

### VREF-P0-01 `softmax_input_x2`

Completed:

- Golden header regeneration.
- E2E AXIS CSim.
- Isolated csynth.
- `dsp_mixed_stream` HLS projection.
- IP package.
- Vivado overlay route.
- PYNQ smoke bundle and session runbook.

Key result:

- Routed WNS `4.497 ns`, above C3b protection threshold `4.415 ns`.
- Physical smoke still missing, so it is not promoted as C3b replacement.

### VREF-P0-02 QKV Weight-Cache URAM Successor

Status: `ready-for-physical-smoke`, successor-only.

| Metric | Result |
|---|---:|
| Latency | `498,485 cycles` |
| DSP | `128` |
| LUT | `43,236` |
| FF | `19,664` |
| BRAM_18K | `114` |
| URAM | `40` |
| Routed WNS | `4.517 ns` |
| Route errors | `0` |

Notes:

- HLS CSim/csynth passed.
- HLS IP package generated.
- Routed overlay exported to `pynq/hgtxr/`.
- PYNQ bundle/session/import/remote plumbing exists.
- Physical-smoke JSON is not captured, so this remains optional successor evidence.

## Board-Run Readiness

Default runnable board target:

- Variant: `C3b PAR16 MEM16 AXIS/DMA`
- bit: `pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit`
- hwh: `pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh`
- bundle: `generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle/`
- expected output: `[32, -13, 26, -6, 14, -11]`

Board execution can run when ZCU104 is available. Final signoff requires importing a board-produced JSON to:

```text
pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json
```

## Final Blockers

| Blocker | Current State | Required Input |
|---|---|---|
| C3b AXIS/DMA physical smoke result | `blocked_missing_canonical_physical_smoke_result` | Board-produced `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` |
| requested XR-VITs sibling | `blocked` | Restore `/home/kjm26/project/PRJXR/XR-VITs` or approve fingerprint-bound XR_Accel replacement policy |

## Validation Summary

Latest live anchors:

- Final evidence: `86/86 pass`, consistency `347`, failed consistency `0`.
- Spec-plan: `86/86 pass`.
- Operator handoff validation: `131/131 pass`.
- Bundle validation: `152/152 pass`.
- Source audit: required `86`, sources `224`, missing `0`.
- Final runner: `blocked`, blockers `2`.
