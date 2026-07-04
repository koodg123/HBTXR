# HGTXR Third-Goal Reference Integration Spec

Date: 2026-06-16

## Scope
Add the ViT accelerator reference analysis into the third-goal execution plan and keep work moving through safe P0 experiments.

Spec-kit status: `spec-kit`/`specify` are unavailable on PATH, so the current spec is maintained manually until the toolchain is installed.
Hardware target: ZCU104 E2E Q4W/Q8A cyclic accelerator baseline with parameter knobs including `HGTXR_PARALLELISM_FACTOR`, `HGTXR_BUS_WIDTH`, and `HGTXR_FIFO_DEPTH`; selected path remains A2 -> A1 plus C/PAR=16 evidence.

## In Scope
- Preserve C3b as board-smoke candidate.
- Add `VREF-*` experiments as successors or software-first ablations.
- Freeze C3b as a protected baseline while final external blockers remain open.
- Apply P2-ViT only to quantization/scale calibration first.
- Apply ME-ViT/HG-PIPE only to memory/dataflow/resource-policy audits first.
- Keep ViTALiTy/ViTCoD/Edge-MoE as software-first or paper-scope-review items.

## Out of Scope
- Replacing exact attention in C3b without explicit scope approval.
- Replacing HGTXR paper-defined search/track behavior.
- Promoting LUT-heavy LLM/GEMM methods as default compute mapping.
- Claiming final resource/accuracy wins without new HLS/Vivado/PYNQ evidence.

## Current Evidence Snapshot
- C3b remains default protected board-smoke path.
- C3b physical-smoke signoff is canonical-path strict: generated bundle or remote result files do not clear Req4/final signoff until imported to `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`.
- PYNQ smoke signoff is overlay-provenance strict when path validation is enabled: C3b/VREF/QKV/m_axi JSON must report preset-matched `bitfile` and `hwhfile` basenames, so correct numeric output with the wrong overlay cannot clear import or discovery gates.
- XR-VITs replacement-policy signoff is candidate-audit bound: approved replacement policies must include a matching candidate-audit fingerprint, recommendation snapshot, approval-event envelope, and policy fingerprint.
- VREF-P0-01 `softmax_input_x2` successor has routed overlay evidence but still needs physical smoke before promotion.
- VREF-P0-02 QKV weight-cache URAM successor has CSim, csynth, HLS IP package, and routed overlay evidence: latency `498485`, `114 BRAM_18K`, `128 DSP`, `19664 FF`, `43236 LUT`, `40 URAM`; routed timing setup slack `4.517 ns`, hold slack `0.010 ns`; PYNQ bundle/session artifacts exist for `vref-p0-softmax-input-x2-qkv-uram`; physical smoke remains pending.
- No-board Search/Track mode-profile evidence now exists outside the full E2E AXIS top: Search profile `hgtxr_search_profile_top` reports `772,268` cycles / `3.861 ms @ 5 ns` and routed WNS `2.638 ns`; Track profile `hgtxr_track_profile_top` reports `99,449` cycles / `0.497 ms @ 5 ns` and routed WNS `1.481 ns`. Both pass CSim, csynth, HLS IP package, ZCU104 Vivado route, and bitgen. This evidence does not yet prove deployed board-runtime latency or runtime search/track invocation distribution.
- Req5 Q4/Q8 SW-HW match audit is `pass` with `fail=0`; arbitrary Q4 weights are reflected through local CSim/golden evidence, and packed Q4 binary/manifest integrity is now checked by binary existence, SHA256 match, byte-count match, expected runtime state, expected C3b raw output, strict testbench golden compare, and C3b/VREF/QKV CSim pass checks. Physical smoke remains a Req4/final board gate.
- Req6 parameterization audit is `pass` with `71/71` checks; tiling, parallelism, bus width, bit width, buffer size, FIFO depth, E2E compile-time `static_assert` guards, PAR16/PAR32 Tcl override support, and the sweep `parallelism_extensions` contract are traced from config macros through derived HLS params, pragmas, Tcl overrides, YAML sweep matrix, and HLS-legal macro relationships. C3b PAR16 remains the validated resource-matrix path; PAR32 remains exploratory until fresh csynth, routed timing, resource-fit audit, and no-C3b-overwrite evidence exist.
- Final evidence manifest is `pass` with required `86/86`; Req1 Ubuntu `/tools/Xilinx` environment proof, Req2 manual Spec fallback/spec-plan proof, Req5 Q4/Q8 SW-HW match proof plus packed-weight binary/manifest integrity gates, P2-ViT scale calibration report, Req9 PAPER_PRJXR DeiT image proof, direct XR-VITs gate audit, direct C3b physical-smoke gate audit, C3b transfer manifest semantic/cross-readiness gates, generic C3b/VREF/QKV PYNQ smoke discovery semantic/safety gates, final-runner blocker path/count/detail contract checks, final-runner mirrored-artifact uniqueness/count checks, final blocker-readiness PYNQ discovery summaries and operator unblock plan, final unblock intake blocker-status/next-input gates plus operator-plan cross-checks, QKV URAM resource/timing/route, VREF-P0-02 buffer lifetime/resource-placement, QKV successor URAM branch/resource linkage, RMU/SMU small-memory LUTRAM placement, RMU/SMU DSP-bound multiply helper coverage, cyclic packed-weight tile URAM placement, cyclic large-temporary URAM placement, cyclic small-tile LUTRAM placement, C3b csynth LUT/latency threshold checks, C3b routed WNS threshold check, Req6 legality/static-assert/PAR16-PAR32 override and promotion-policy guards, VREF-P0-01 PoT scale audit/sweep, HG-PIPE operator property coverage, C3b `csynth.xml` resource-policy proof, resource-policy artifact singularity/hash/date gates, XR-VITs policy preview path/candidate-audit/approval-event contract, active-policy placeholder-approver rejection source contract, validator freshness checks, spec-plan current-doc freshness checks, completion-audit freshness checks, PYNQ smoke overlay-provenance contract checks, and third-goal source-audit source-count freshness are included in the `347` manifest consistency checks; spec-plan conformance is `86/86`; operator handoff validation `131/131`; bundle validation `152/152`; source audit is `pass` with required `86`, sources `224`, missing `0`; current audit remains `blocked-external` with reflected `10`, partial `1`, blocked `1`.
- Completion audit is `blocked` with pass `11`, partial `1`, blocked `2`; requirements trace is `blocked` with blocked requirement ids `['11']` and partial requirement ids `['4']`.
- HG-PIPE operator audit covers `97/97` reference checks, `5899008` sampled values, and `211/211` deterministic contract property checks.

## Acceptance Criteria
- Third-goal plan lists all VREF experiments with priority and promotion gate.
- C3b smoke remains the immediate unblock path.
- C3b protection checklist exists and reports no static failures.
- Any VREF successor must meet or beat C3b protection thresholds before replacement: latency `<= 37508072`, WNS `>= 4.415 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`.
- P0 tasks can proceed without board access.
- Search/Track 4 ms / 1 ms claims are accepted at no-board Vivado mode-profile level; deployed board-runtime claims still require runtime/board instrumentation.
- Documentation records unresolved external blockers.
- Validation commands pass for syntax/JSON/diff hygiene.
- Successor-only experiments are recorded in source/current/final evidence manifests before any promotion claim.
- Board-smoke JSON references the expected overlay artifact basename for the selected preset.
