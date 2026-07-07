# HGTXR Hardware Work Progress and Experiment Summary

Date: 2026-06-27
Scope: `hardware/` directory. This document consolidates the current conversation, completed work, experiment plan, results, evidence boundaries, and next actions.

## 1. Prompt Brief

| Field | Summary |
|---|---|
| Goal | Summarize all current HGTXR hardware work, planned experiments, results, and handoff context. |
| Inputs | Existing docs, HLS/Vivado reports, no-board result artifacts, integrated reference analysis, current conversation decisions. |
| Constraints | Keep C3b protected; separate full E2E evidence from mode-profile proxy evidence; do not claim board measurements without board JSON. |
| Expected outputs | Durable progress summary and updated HANDOVER document. |
| Acceptance criteria | Reports current status, experiment queue, results, blockers, commands, evidence paths, and explicit uncertainty. |

## 2. Role and Agent Workflow

| Layer | Current handling |
|---|---|
| Master | Scope normalized to hardware-only documentation and evidence consolidation. |
| Expert Council | Hardware architecture, HLS/Vivado evidence, and artifact/provenance perspectives used for synthesis. |
| Runtime Manager | Current document generation ran in main agent because no callable sub-agent runtime was exposed in this turn. |
| Prior sub-agents | Earlier no-board P0 work used Spark read-only sidecars for mode-top structure, script/Tcl integration, Vivado route audit, and documentation-prep. |
| Artifact Manager | Outputs are saved under `docs/track/` and linked to existing `analysis/no-board-results/` artifacts. |

## 3. Current High-Level State

| Area | Status | Evidence |
|---|---|---|
| C3b protected baseline | Active board-smoke candidate | `analysis/no-board-results/P0_PROGRESS_2026_06_26.md` |
| C3b physical smoke | Missing | Canonical expected path: `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` |
| XR-VITs exact source or approved policy | Still external blocker | `docs/track/HANDOVER.md`, generated signoff audits |
| VREF-P0-01 PoT scale successor | Routed successor evidence exists, physical smoke pending | `docs/Execution.md`, generated VREF signoff artifacts |
| VREF-P0-02 QKV URAM successor | CSim/csynth/IP/routed evidence exists, physical smoke pending | `docs/track/HANDOVER.md` |
| No-board PAR32 full E2E candidate | CSim/csynth/IP/Vivado route completed | `analysis/no-board-results/P0_PROGRESS_2026_06_26.md` |
| No-board Search/Track mode-profile | HLS/IP/Vivado route completed | `analysis/no-board-results/MODE_PROFILE_RESULTS_2026_06_27.md` |
| Board runtime DMA/p95/p99/search-track distribution | Not measured | Board inference was excluded for this phase |

## 4. Key User Decisions Captured

| Decision | Current interpretation |
|---|---|
| Use hardware directory as main scope | All current file changes and reports remain inside `hardware/`. |
| Use agent hierarchy and sub-agent instructions | Planning docs and task cards reflect this; current turn used fallback because no callable sub-agent runtime was available. |
| Increase DSP/URAM use and reduce LUT-only arithmetic | DSP-bound helper and memory-placement audits were added; PAR32 increased DSP utilization in full E2E no-board candidate. |
| Use LUTRAM for small memories | RMU/SMU score/prob and cyclic small scratch policies are tracked. |
| Choose path A2 -> A1 and C, keep E pending | Reflected in Master/Sub/Spec and CHOICE docs. |
| Exclude board inference for current no-board plan | All current P0/PAR32/mode-profile claims are HLS/Vivado no-board claims only. |
| Ask for option analysis before multi-direction work | Experiment plans now include expected result, tradeoff, and promotion gate. |
| Ask whether full E2E Search/Track meets 4ms/1ms | Answer: not proven. Current 4ms/1ms evidence is mode-profile proxy, not deployed full E2E. |

## 5. Full E2E vs Mode-Profile Boundary

This distinction is critical.

| Claim | Status | Correct wording |
|---|---|---|
| Full E2E Search mode runs 8 transformer blocks | Partially true by configuration intent | `HGTXR_SEARCH_DEPTH = 8`; deployed full E2E timing is not proven at 4 ms. |
| Full E2E Track mode runs 4 transformer blocks | Partially true by configuration intent | `HGTXR_TRACK_CUT_DEPTH = 4`; deployed full E2E timing is not proven at 1 ms. |
| Full E2E Search 8-block path takes 4 ms | Not proven | Do not claim. |
| Full E2E Track 4-block path takes 1 ms | Not proven | Do not claim. |
| Search mode-profile proxy meets 4 ms | Proven at no-board HLS/IP/Vivado route level | `hgtxr_search_profile_top`: `772,268` cycles, `3.861 ms`. |
| Track mode-profile proxy meets 1 ms | Proven at no-board HLS/IP/Vivado route level | `hgtxr_track_profile_top`: `99,449` cycles, `0.497 ms`. |

The mode-profile tops in `hls/src/hgtxr_mode_profile_top.cpp` use synthetic profile kernels such as `mode_attention_active`, `mode_mlp_active`, and `mode_projection_active`. They are useful for latency-envelope exploration, but they are not a replacement for routed full E2E mode-dispatch evidence.

## 6. Main Experiment Results

### 6.1 C3b Protected Baseline

| Metric | Value |
|---|---:|
| Latency cycles | 37,508,072 |
| Latency @ 5 ns | 187.540 ms |
| LUT | 126,506 |
| DSP | 604 |
| BRAM_18K | 332 |
| URAM | 64 |
| Role | Protected board-smoke candidate |

### 6.2 Full E2E PAR32 No-Board Candidate

| Metric | `par32_dsp_mixed_stream_mem16` |
|---|---:|
| HLS latency cycles | 26,139,644 |
| HLS latency @ 5 ns | 130.698 ms |
| LUT | 193,546 |
| DSP | 1,148 |
| BRAM_18K | 332 |
| URAM | 64 |
| Vivado WNS | 3.885 ns |
| Route errors | 0 |
| Bitgen | pass |
| Interpretation | Full E2E no-board routed candidate; improves HLS latency and DSP use, but LUT is high at about 84%. |

### 6.3 Mode-Profile Search/Track Results

| Mode | Top | Worst cycles | Latency @ 5 ns | Target | HLS resource | Routed timing |
|---|---|---:|---:|---:|---|---|
| Search | `hgtxr_search_profile_top` | 772,268 | 3.861 ms | 4.000 ms | LUT 65,557, DSP 294, BRAM18 8, URAM 96 | WNS 2.638 ns, route errors 0 |
| Track | `hgtxr_track_profile_top` | 99,449 | 0.497 ms | 1.000 ms | LUT 69,728, DSP 302, BRAM18 51, URAM 64 | WNS 1.481 ns, route errors 0 |

Post-route mode-profile utilization is much lower than HLS LUT estimates:

| Mode | Post-route LUT | FF | DSP | BRAM18 | URAM | Note |
|---|---:|---:|---:|---:|---:|---|
| Search | 18,825 | 15,171 | 294 | 2 | 96 | HLS IP included; Search has zero URAM headroom. |
| Track | 22,245 | 20,826 | 302 | 42 | 64 | HLS IP included; resource margin is cleaner. |

## 7. Current Experiment Plan

| Priority | Experiment | Status | Next decision |
|---|---|---|---|
| P0 | C3b canonical board smoke | Pending external board JSON | Required for final signoff. |
| P0 | Full E2E PAR32 no-board route | Completed | Keep as routed no-board candidate; do not replace C3b without board evidence. |
| P0 | Full E2E Search/Track mode-dispatch timing | Not implemented | Add real full E2E mode-dispatch top/counters before claiming 4 ms/1 ms in full E2E. |
| P0 | Search mode URAM margin | Pending | Reduce `96/96` URAM or document zero-margin risk. |
| P1 | P2-ViT PoT scale calibration | Planned/partially reported | Extend accuracy/error proxy before precision-sensitive HLS promotion. |
| P1 | Q4/Q8 deterministic coverage | Planned | Add more SW/HW vectors for exact-match confidence. |
| P1 | Nonlinear operator stress | Planned | Stress LayerNorm/GELU/Softmax/Quantization LUT and precision variants. |
| P1 | Exact attention tiled dataflow | Planned | Preferred over approximation if resource/latency improves without paper-scope change. |
| P2 | Sparse/linear/Taylor attention ablations | Planned, software-first | Requires explicit accuracy/error report and paper-scope decision. |
| P3 | LUT-heavy negative controls | Planned | Use as reviewer-facing rationale for rejecting LUT-heavy compute defaults. |

## 8. What Can Be Claimed Now

| Claim type | Allowed statement |
|---|---|
| C3b baseline | C3b remains the protected board-smoke candidate with HLS latency/resource and routed timing evidence. |
| PAR32 full E2E | `par32_dsp_mixed_stream_mem16` is a routed no-board full E2E candidate with lower HLS latency and higher DSP use than C3b, but higher LUT pressure. |
| Search/Track mode-profile | Separate no-board mode-profile tops meet 4 ms and 1 ms targets through HLS/IP/Vivado route. |
| Board runtime | Not claimable yet. No board latency, DMA bandwidth, p95/p99, or search/track invocation distribution has been measured in this phase. |
| Full E2E Search/Track 4 ms/1 ms | Not claimable yet. Needs deployed full E2E mode-dispatch implementation or counters with csynth/route evidence. |

## 9. Immediate Commands

No-board report gate:

```sh
python3 scripts/report/collect_no_board_reports.py --output-dir /tmp/hgtxr_p0_collect_check --enforce-p0
```

Mode-profile HLS/Vivado reruns:

```sh
scripts/run/run_mode_profile_no_board.sh csim search_par32
scripts/run/run_mode_profile_no_board.sh csynth search_par32
scripts/run/run_mode_profile_no_board.sh package search_par32
scripts/run/run_mode_profile_vivado_no_board.sh search_par32

scripts/run/run_mode_profile_no_board.sh csim track_par32
scripts/run/run_mode_profile_no_board.sh csynth track_par32
scripts/run/run_mode_profile_no_board.sh package track_par32
scripts/run/run_mode_profile_vivado_no_board.sh track_par32
```

Full E2E PAR32 no-board rerun:

```sh
scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_mixed_stream_mem16
scripts/run/run_e2e_q4w8a_no_board.sh package par32_dsp_mixed_stream_mem16
scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_dsp_mixed_stream_mem16
```

Final signoff status refresh after external blockers change:

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked
```

## 10. Remaining Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Search mode-profile uses `96/96` URAM | Routes now, but zero headroom for integration changes | Reduce Search URAM or document as no-margin profile evidence. |
| Mode-profile is synthetic | Can be misread as full E2E Search/Track proof | Keep boundary explicit in all docs and claims. |
| PAR32 full E2E LUT pressure is high | Route passed, but design may be less robust under added logic | Keep C3b protected; use PAR32 as no-board candidate only. |
| C3b board smoke missing | Final signoff cannot close | Capture/import canonical C3b board JSON. |
| XR-VITs exact source/policy unresolved | Final signoff remains externally blocked | Restore exact path or create approved replacement policy with fingerprint-bound tool. |

## 11. Best Next Engineering Move

The next technical step should be one of these two, depending on priority:

| Option | Work | Expected result |
|---|---|---|
| F1 | Build a real full E2E mode-dispatch top or counters for Search/Track paths | Enables defensible full E2E Search 8-block / Track 4-block latency claims. |
| F2 | Reduce Search mode-profile URAM from `96/96` | Gives route/resource headroom while preserving no-board mode-profile target evidence. |

F1 is the correct path if the paper/report needs the 4 ms / 1 ms statement to apply to full E2E behavior. F2 is the correct path if the immediate concern is ZCU104 resource robustness.
