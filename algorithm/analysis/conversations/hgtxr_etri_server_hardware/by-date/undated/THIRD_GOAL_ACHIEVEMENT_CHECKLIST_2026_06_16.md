# HGTXR Third-Goal Achievement Checklist

Date: 2026-06-16

## Summary

Overall status: `blocked-external`.

Internal evidence is complete for the default signoff package. Final completion is blocked by C3b board-smoke JSON and XR-VITs source/policy.

Achievement estimate:

- Original objective items: `10/12 complete`, `1/12 partial`, `1/12 blocked`.
- Final signoff package: internal evidence `86/86 pass`.
- Completion audit: pass `11`, partial `1`, blocked `2`.
- Current audit: reflected `10`, partial `1`, blocked `1`.

## Objective Checklist

| ID | Objective | Status | Evidence | Remaining Work |
|---|---|---|---|---|
| 0 | Analyze plan, progress, HANDOVER, and continue work | Complete | `docs/track/HANDOVER.md`, `docs/track/PROGRESS.md`, `docs/CHOICE.md`, `third_goal_completion_audit_2026_06_10.json` | None |
| 1 | Adapt to Ubuntu Linux, `/tools/Xilinx`, changed workdir | Complete | `req1_environment_audit_2026_06_16.json`, `13/13 pass` | None |
| 2 | Build proper experiment plan/spec first; use spec-kit/Spark when possible | Complete with fallback | `spec_plan_conformance_audit_2026_06_10.json`, `86/86 pass`; manual Spec fallback because `spec-kit` unavailable; Spark fallback recorded | Install spec-kit only if future policy requires it |
| 3 | Prioritize ZCU104 cyclic hardware accelerator baseline | Complete | C3b selected in `e2e_resource_matrix_2026_06_10.json`; bit/hwh ready | Board smoke still required for final signoff |
| 4 | Make ViT model run E2E | Partial | E2E HLS/CSim/csynth/bit/hwh/bundle ready; final runner blocked by missing board JSON | Run C3b on ZCU104 and import canonical JSON |
| 5 | Use 4-bit weights and 8-bit activations; arbitrary weights OK if SW/HW match | Complete | `req5_q4q8_swhw_match_audit_2026_06_16.json`, `11/11 pass`; packed binary/manifest gated | None |
| 6 | Parameterize tiling, parallelism, bus width, bit width, buffer size, FIFO depth | Complete | `req6_parameterization_audit_2026_06_16.json`, `71/71 pass` | Fresh reports needed only for new promoted configs |
| 7 | Analyze HG-PIPE code and reflect needed parts for ZCU104 cyclic accelerator | Complete | HG-PIPE operator audit and final evidence gates | None |
| 8 | Implement LayerNorm, GeLU, Softmax, Quantization from HG-PIPE paper/code | Complete | `hgpipe_operator_audit_2026_06_16.json`, `97/97` refs, `211/211` properties | None |
| 9 | Reference PAPER_PRJXR DeiT image | Complete | `req9_deit_image_reference_audit_2026_06_16.json`, `11/11 pass` | None |
| 10 | Reference `/home/kjm26/project/PRJXR/XR-VIT` experiments/code | Complete | VREF/P2/QKV/resource-policy artifacts integrated into final evidence | None |
| 11 | Use `/home/kjm26/project/PRJXR/XR-VITs` HLS code to fit ZCU104 low-latency target | Blocked | `xr_vits_gate_audit_2026_06_16.json`, exact source missing; `XR_Accel` candidate ready | Restore exact `/XR-VITs` or approve XR_Accel replacement policy |

## Experiment Result Checklist

| Experiment | Status | Key Result | Next Gate |
|---|---|---|---|
| A2 E2E m_axi | Complete | latency `81,514,836`, DSP `334`, LUT `84,220`, URAM `64`, WNS `1.926 ns` | Not selected as best default |
| A1 E2E AXIS/DMA | Complete | latency `71,316,968`, DSP `332`, LUT `81,168`, URAM `64`, WNS `4.120 ns` | Superseded by C3b |
| C1 PAR16 AXIS/DMA | Complete | latency `37,508,072`, DSP `604`, LUT `127,916`, URAM `64`, WNS `4.723 ns` | Superseded by C3b for lower LUT |
| C3b PAR16 MEM16 AXIS/DMA | Ready for board smoke | latency `37,508,072`, DSP `604`, LUT `126,506`, URAM `64`, WNS `4.415 ns` | Capture canonical C3b smoke JSON |
| VREF-P0-01 PoT scale audit | Complete | `14/14 pass` | Keep current scales for C3b |
| VREF-P0-01 PoT sweep | Complete | `45` candidates, `0` fail | Non-current rows successor-only |
| VREF-P0-01 `softmax_input_x2` DSP mixed stream | Ready for board smoke | routed WNS `4.497 ns` | Capture optional physical smoke |
| VREF-P0-02 QKV URAM | Ready for board smoke | latency `498,485`, DSP `128`, LUT `43,236`, URAM `40`, WNS `4.517 ns` | Capture optional QKV smoke JSON |
| Req5 Q4/Q8 SW-HW match | Complete | `11/11 pass` | Add more vectors only if desired |
| Req6 parameterization | Complete | `71/71 pass` | New configs require fresh evidence |
| HG-PIPE operators | Complete | refs `97/97`, properties `211/211` | Stress tests optional |
| Resource policy | Complete | `36/36 pass` | Re-run after resource-affecting edits |

## Signoff Checklist

| Gate | Status | Evidence |
|---|---|---|
| Final evidence manifest | Complete | `86/86 pass`, consistency `347`, failed `0` |
| Spec-plan conformance | Complete | `86/86 pass` |
| Operator handoff validation | Complete | `131/131 pass` |
| Final bundle validation | Complete | `152/152 pass` |
| Source audit | Complete | required `86`, sources `224`, missing `0` |
| C3b physical smoke | Blocked | canonical JSON missing |
| XR-VITs source/policy | Blocked | exact source missing; active policy missing |
| Final runner | Blocked | blocker count `2` |

## Blocker Checklist

- [ ] Run C3b bundle on ZCU104.
- [ ] Validate board JSON with preset `axis-c3b-mem16`.
- [ ] Import C3b JSON to `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- [ ] Restore `/home/kjm26/project/PRJXR/XR-VITs` or approve replacement policy.
- [ ] Rerun `python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR`.

## Optional Successor Checklist

- [ ] Run VREF-P0-01 DSP mixed-stream physical smoke.
- [ ] Run VREF-P0-02 QKV URAM physical smoke.
- [ ] Decide whether QKV URAM becomes a hard final-signoff gate.
- [ ] Run PAR32 csynth and route if higher parallelism is selected.
- [ ] Extend Q4/Q8 SW-HW vector coverage.
- [ ] Extend HG-PIPE operator stress tests.
