# HGTXR Hardware Progress

Date: 2026-06-16

Latest Continuation: Selected Path Execution Audit.

## Plan 대비 진행상황
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
