# HGTXR Hardware Continuation Log

## 2026-06-16 - Third-Goal Documentation Package

- Added consolidated progress and experiment report:
  `docs/THIRD_GOAL_PROGRESS_EXPERIMENT_REPORT_2026_06_16.md`.
- Added future experiment list:
  `docs/THIRD_GOAL_FUTURE_EXPERIMENTS_2026_06_16.md`.
- Added objective achievement-rate checklist:
  `docs/track/THIRD_GOAL_ACHIEVEMENT_CHECKLIST_2026_06_16.md`.
- Baseline result recorded: C3b is default best board-smoke candidate with latency `37,508,072`, DSP `604`, LUT `126,506`, URAM `64`, routed WNS `4.415 ns`.
- Successor result recorded: VREF-P0-02 QKV URAM has latency `498,485`, DSP `128`, LUT `43,236`, URAM `40`, routed WNS `4.517 ns`, and remains optional until physical smoke.
- Current signoff state remains blocked only by C3b canonical board JSON and XR-VITs source/policy.

Selected path execution audit remains active for A2/A1/C/PAR16 continuity.

## 2026-06-15
- Added detailed ViT accelerator analysis under `analysis/vit-accel`.
- Integrated reference-derived experiments into third-goal planning.
- Added local hardware tracking docs under `docs/` and `docs/track/`.
- Current blocked items remain external: C3b physical smoke and XR-VITs replacement policy.

## 2026-06-16
- Added VREF-P0-01 PoT scale readiness audit and VREF-P0-02 buffer lifetime/resource placement audit.
- Added VREF-P0-01 PoT scale candidate sweep across reduced HG-PIPE vector specs.
- Added C3b protection checklist and successor thresholds.
- Current VREF-P0-01 sweep result: `3` specs, `45` candidates, `0` fail; keep current C3b scales.
- Current C3b protection result: `30/32 pass`, `0 fail`, `2 pending`.
- Added VREF-P0-01 `softmax_input_x2` successor package with regenerated spec/header and passing E2E AXIS CSim.
- Ran VREF-P0-01 `softmax_input_x2` E2E AXIS csynth in isolated project: `498485` cycles, `4.069 ns`, `128 DSP`, `43236 LUT`, `88 URAM`.
- Ran VREF-P0-01 `softmax_input_x2` `dsp_mixed_stream` csynth: URAM reduced from `88` to `32`, with `144 BRAM_18K`, `128 DSP`, `43236 LUT`, and unchanged `498485` cycles / `4.069 ns`.
- Current recommended `softmax_input_x2` resource variant is `dsp_mixed_stream`; HLS resource thresholds clear.
- Packaged `softmax_input_x2` `dsp_mixed_stream` as HLS IP and generated `component.xml` / `export.zip`.
- Built Vivado AXI DMA overlay for the successor with Ubuntu/Xilinx 2023.2 `libtinfo.so.5` LD_LIBRARY_PATH workaround.
- Routed timing passed: WNS `4.497 ns`, TNS `0.000 ns`, WHS `0.010 ns`, route errors `0`, fully routed nets `19829`.
- Exported successor bit/hwh under `generated/build/vivado/overlay/...` and `pynq/hgtxr/`.
- Successor promotion now has HLS resource and routed timing evidence; physical board smoke remains pending.
- Added successor ZCU104 PYNQ smoke bundle and session runbook:
  `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz`
  and `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md`.
- Successor smoke bundle validation passed; expected raw `[58, -51, 42, -28, 36, -41]`, expected `runtime_state=2`.
- Generalized the ZCU104 remote smoke runner with a successor profile and emitted dry-run SSH/SCP plan:
  `generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.md`.
- Validation rerun: `63` focused HLS successor/smoke/bundle/remote-runner unittest cases passed; third-goal neutral preflight reported `ok=76 warn=5 fail=0`.
- Added VREF-P0-01 successor physical-smoke receiver to preflight and successor projection.
  Missing board JSON remains pending; valid copied JSON will promote `physical_smoke_available` to pass.
- Validation rerun: `49` focused preflight/successor/smoke/import/remote-runner unittest cases passed.
- Refreshed final unblock closeout packet with VREF successor gate included as an optional promotion gate.
  New packet: `generated/signoff/final_unblock_closeout_packet_2026_06_16.md`.
  Validation: all thirty-three closeout checks passed.
- Added selectable VREF-required closeout policy via `--require-vref-successor-physical-smoke`.
  Default packet keeps blockers at `2`; VREF-required packet adds `VREF-P0 successor physical smoke result` as blocker `3`.
  Both closeout validations pass with `34/34` checks.
- Extended final signoff runner with VREF successor smoke controls:
  `--import-vref-successor-smoke-json`, `--dry-run-import-vref-successor-smoke`,
  `--execute-vref-successor-smoke`, `--vref-successor-remote-dir`, and
  `--require-vref-successor-physical-smoke`.
  Validation: `tests.test_run_third_goal_final_signoff` passed with `11` tests.
- Spark follow-up hit quota, so a GPT5.5 read-only evaluator was spawned for current third-goal evidence review.
- Added current third-goal requirement audit:
  `generated/signoff/third_goal_current_audit_2026_06_16.md`.
- Added direct source/HANDOVER audit for req0:
  `generated/signoff/third_goal_source_audit_2026_06_16.md`.
  Source status: `pass`, required `35`, sources `148`, missing `0`, markerless required `0`.
- Refreshed current third-goal requirement audit after source-audit linkage:
  `blocked-external`, requirements reflected `6`, partial `5`, blocked `1`;
  evidence classes are recorded as `doc-reported`, `external-blocker`, `file-exists`, and `validated-by-tool`.
- Added HG-PIPE operator audit for LayerNorm, GeLU, Softmax, and Quantization:
  `97/97` reference checks pass and `5899008` samples checked.
- Refreshed source/current audits after operator-audit linkage:
  source audit required `37`, sources `150`, missing `0`;
  current audit remains `blocked-external`, with requirements reflected `7`, partial `4`, blocked `1`.
- Added XR-VITs gate audit:
  exact `/home/kjm26/project/PRJXR/XR-VITs` is missing; `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`
  is candidate-ready but still needs explicit approval as a replacement.
- Refreshed source/current audits after XR-VITs gate linkage:
  source audit required `41`, sources `154`, missing `0`;
  current audit remains `blocked-external`, with requirements reflected `7`, partial `4`, blocked `1`.
- Added C3b physical-smoke gate audit:
  `ready_for_board=True`, bundle/session `pass`, canonical board JSON missing, `physical_smoke_pass=False`.
- Added Req2 spec/sub-agent gate audit:
  spec-kit/specify unavailable, manual Spec fallback recorded, Spark-first/GPT5.5 fallback routing recorded.
- Refreshed source/current audits after Req2 gate linkage:
  source audit required `45`, sources `158`, missing `0`;
  current audit remains `blocked-external`, with requirements reflected `8`, partial `3`, blocked `1`.
- Pending items remain external: C3b physical smoke JSON and XR-VITs exact source or approved replacement policy.
- Added `docs/CHOICE.md` so Req11 and C3b unblock choices are durable:
  X1 exact XR-VITs restore, X2 approved XR_Accel replacement, X3 keep blocked;
  C1 board-smoke import, C2 continue without final signoff.
- Refreshed source/current audits after adding `docs/CHOICE.md` as required evidence:
  source audit required `46`, sources `159`, missing `0`;
  current audit remains `blocked-external`, with requirements reflected `8`, partial `3`, blocked `1`.
- Enhanced final-signoff preflight Req11 failure detail:
  `XR_Accel` candidate is reported when present, but the gate still fails until exact source exists or approval policy validates.
- Enhanced `tools/import_pynq_smoke_result.py` provenance fields:
  source sha256, canonical destination status, copy state, path-check mode, and payload summary.
  Validation: import/final-signoff/readiness tests passed with `21` cases.
- Enhanced C3b candidate discovery:
  emits `dry_run_import_command` before active import command, skips docs/resources metadata JSON noise.
  Current discovery remains `missing`, pass `0`, candidates `0`; no canonical result written.
- Refreshed third-goal unblock checklist:
  C3b host import sequence now uses final-signoff runner dry-run import before active import and records runner/import report paths.
  Checklist remains `pending-unblock` with blockers `2`.
- Refreshed ZCU104 smoke session runbooks:
  C3b and VREF successor host copy-back steps now use final-signoff runner dry-run import before active import.
  Preflight accepts runner import flow; C3b gate remains blocked only by missing canonical board JSON.
- Rechecked current third-goal gates:
  `36` focused tests passed; C3b/VREF smoke-session JSON, C3b gate audit JSON, source audit JSON,
  and current audit JSON parse cleanly; `git diff --check -- docs tools tests generated/pynq generated/signoff` passed.
  Neutral preflight remains `ok=76 warn=6 fail=0`; final-signoff remains expected-blocked with `ok=76 warn=4 fail=2`
  on C3b physical smoke and requested XR-VITs source/policy.
- Added generic PYNQ smoke candidate discovery:
  C3b and VREF successor presets now share a side-effect-free candidate scanner with runner import command templates.
  Final-signoff runner emits `vref-successor-smoke-candidate-discovery` and records `vref_smoke_discovery_status`.
  Current VREF discovery remains `missing`, pass `0`, candidates `0`; no board result is fabricated.
  Validation: `27` focused discovery/runner/source/current audit tests passed; source audit is now required `48`, sources `161`, missing `0`.
- Closed generic discovery tracking gap:
  source audit now requires generic C3b/VREF discovery `{json,md}` artifacts, preflight validates legacy and generic discovery JSON safety fields,
  and current audit reports `smoke_candidate_discovery.{c3b_generic,vref_generic,vref_runner}` while preserving legacy `vref_smoke_discovery`.
  Spark sidecar hit quota; GPT5.5 fallback completed read-only review. Validation: `42` focused tests passed;
  source audit required `52`, sources `171`, missing `0`; neutral preflight `ok=79 warn=6 fail=0`;
  final-signoff remains expected-blocked with `ok=79 warn=4 fail=2` on C3b physical smoke and requested XR-VITs source/policy.
- Surfaced final unblock route as hard evidence:
  required source audit now includes closure readiness, unblock intake, final unblock commands, and XR-VITs unblock packet.
  Current audit reports these four artifacts directly; preflight blocker messages now point to the command card, C3b dry-run import,
  XR-VITs unblock packet, and policy dry-run command. Spark hit quota; GPT5.5 fallback identified the missing command-card/policy-packet surfacing.
  Validation: `40` focused tests passed; closure/intake regenerated as expected-blocked; source audit required `60`, sources `173`, missing `0`;
  neutral preflight `ok=79 warn=6 fail=0`; final-signoff expected-blocked `ok=79 warn=4 fail=2`.
- Re-ran final signoff runner after unblock-route surfacing:
  runner refreshed hardware signoff outputs and mirrored parent `docs/resources` evidence; final evidence manifest passed with `50/50` required artifacts.
  Final runner summary remains expected-blocked with `fail=2`, blockers `2`, final preflight `ok=79 warn=4 fail=2`.
  Source audit after runner refresh remains `pass`, required `60`, sources `173`, missing `0`; current audit remains `blocked-external`.
  Validation: `48` runner/evidence/source/current/preflight tests passed; JSON parse and `git diff --check` passed.
- Promoted final operator handoff evidence into current tracking:
  source audit now requires final unblock candidate audit, final operator handoff, and final operator handoff validation `{json,md}` artifacts.
  Current audit reports `final_unblock_candidate_audit`, `final_operator_handoff`, and `final_operator_handoff_validation`.
  Validation: focused source/current audit tests passed; regenerated source audit is `pass`, required `66`, sources `175`, missing `0`;
  current audit remains `blocked-external` with reflected `8`, partial `3`, blocked `1`.
- Hardened final evidence manifest audit linkage:
  centralized the current-audit date tag in `tools/write_final_evidence_manifest.py`, added direct failure tests for source-audit missing-required
  and current-audit requirement-count consistency checks, and documented the runner's source-audit refresh step.
  Validation: focused manifest/runner py_compile passed; `17` manifest/runner unit tests passed.
  Final runner refresh remains expected-blocked with `fail=2`, blockers `2`; evidence manifest is `pass`, required `58/58`.
  Source audit is `pass`, required `70`, sources `185`, missing `0`; current audit remains `blocked-external`.
- Ran VREF-P0-02 QKV weight-cache URAM successor experiment:
  CSim passed in isolated project `hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csim`.
  HLS csynth passed in isolated project `hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_csynth`.
  Result: latency `498485`, estimated clock `4.069 ns`, `114 BRAM_18K`, `128 DSP`, `43236 LUT`, `40 URAM`.
  Delta versus `dsp_mixed_stream`: `-30 BRAM_18K`, `+8 URAM`, unchanged DSP/LUT/latency.
  Artifact: `generated/signoff/vref_p0_qkv_uram_cache_successor_2026_06_16.md`.
- Refreshed handover/plan/spec after QKV URAM successor:
  `docs/track/HANDOVER.md`, `docs/Master-Plan.md`, and `docs/Spec.md` now carry the current successor and final evidence state.
  Current durable state remains final evidence manifest `58/58` pass, source audit required `70`, current audit `blocked-external`.
  No external blocker was closed; C3b physical smoke JSON and XR-VITs exact/replacement policy remain required.
- Extended QKV URAM successor writer with optional IP package, routed overlay, and physical-smoke evidence fields.
  Current optional evidence remains missing, so promotion remaining is `ip_package`, `routed_overlay_timing`, and `physical_smoke_json`.
- Added QKV URAM PYNQ/remote/import plumbing:
  variant `vref-p0-softmax-input-x2-qkv-uram`, preset `axis-vref-p0-softmax-input-x2-qkv-uram`, canonical result `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json`.
  Validation: focused QKV/PYNQ/remote/import/session tests passed with `39` cases.
- Generated QKV URAM HLS IP package:
  `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/ip/component.xml`
  and `generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_qkv_uram_ip/solution_e2e_q4w8a/impl/export.zip`.
  QKV successor promotion remaining is now `routed_overlay_timing` and `physical_smoke_json`.
- Generated QKV URAM routed overlay:
  `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram.bit`
  and matching `.hwh`, also copied under `pynq/hgtxr/`.
  Routed timing met: setup slack `4.517 ns`, hold slack `0.010 ns`; route status `19846/19846` routable nets fully routed, routing errors `0`.
  Generated QKV PYNQ smoke bundle/session:
  `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz`
  and `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.md`.
  QKV successor promotion remaining is now `physical_smoke_json`.
- Integrated QKV URAM successor into third-goal preflight:
  `tools/check_third_goal_preflight.py` now checks QKV bit/hwh, bundle manifest/tar, run/validate scripts,
  ZCU104 session JSON/Markdown contract, and QKV physical-smoke JSON with preset `axis-vref-p0-softmax-input-x2-qkv-uram`.
  Validation: `python3 -m unittest tests.test_check_third_goal_preflight` passed with `24` tests.
  Final signoff runner refreshed: preflight summary `ok=96 warn=5 fail=2`; remaining failures are C3b physical smoke and XR-VITs approval.
- Integrated QKV URAM successor into final-signoff runner discovery/import/remote state:
  `tools/discover_pynq_smoke_candidates.py` supports preset `axis-vref-p0-softmax-input-x2-qkv-uram` and runner import flags
  `--import-qkv-uram-smoke-json` / `--dry-run-import-qkv-uram-smoke`.
  `tools/run_third_goal_final_signoff.py` now records QKV import, remote, required-gate, and smoke-discovery status fields.
  Validation: py_compile passed for runner/discovery/tests; `20` focused discovery/runner tests passed.
  Final runner refresh is expected-blocked with `fail=2`; QKV discovery is `missing` only because physical board JSON has not been captured.
- Hardened QKV URAM source/current audit tracking:
  Spark evaluator hit quota; GPT5.5 fallback identified missing explicit current-audit QKV runner fields.
  `tools/write_third_goal_source_audit.py` now requires QKV smoke discovery `{json,md}` artifacts.
  `tools/write_third_goal_current_audit.py` now exposes `gate_modes.qkv_uram_runner`,
  `qkv_uram_physical_smoke`, `pynq_plumbing`, and `smoke_candidate_discovery.qkv_runner`.
  Validation: `49` focused tests passed; source audit `required_count=70`, `source_count=185`, `missing_required=0`;
  current audit remains `blocked-external`; final runner remains expected-blocked with only C3b physical smoke and XR-VITs approval unresolved.
- Refreshed final evidence manifest contract tracking after QKV discovery evidence:
  live manifest is `58/58` pass and includes QKV successor plus QKV smoke-discovery artifacts.
  `tools/validate_final_signoff_bundle.py` now compares requirements trace/operator handoff manifest contracts against the live manifest and fails stale `56/56` contracts.
- Reflected updated Req5 Q4/Q8 SW-HW match evidence into third-goal tracking:
  Spark sidecar hit quota; GPT5.5 fallback completed read-only gap review. Added `tools/write_req5_q4q8_swhw_match_audit.py`
  and integrated its `{json,md}` outputs into source/current/final evidence tracking.
  Req5 audit is `pass`, fail `0`; final evidence manifest is `60/60` pass; source audit is `required_count=72`, `source_count=189`, `missing_required=0`;
  current audit remains `blocked-external` with reflected `9`, partial `2`, blocked `1`.
  True default external blockers remain C3b physical smoke JSON and requested XR-VITs source/replacement policy.
- Strengthened Req6 parameterization proof:
  Spark sidecar hit quota; GPT5.5 fallback identified the same Req6 evidence gap. Added `tools/write_req6_parameterization_audit.py`
  with checks from config macros to derived HLS params, E2E pragmas, CSim/CSynth Tcl parallelism overrides, and sweep-matrix values.
  Req6 audit is `pass`, checks `39/39`; final evidence manifest is `62/62` pass; source audit is `required_count=74`, `source_count=193`, `missing_required=0`;
  final-signoff bundle validation is `pass`; current audit remains `blocked-external` only because C3b physical smoke and XR-VITs source/policy remain external.
- Refreshed QKV URAM final-manifest consistency:
  closed completed Spark sidecar `019eccf0-38e4-75a2-9a9c-d2df33171528`; a new Spark read-only review spawn failed with `agent thread limit reached`, so the main agent completed the integration.
  Final evidence manifest remains `62/62` pass with `42` consistency checks and no failures.
  QKV checks cover successor status, URAM ceiling, latency, DSP, LUT, BRAM reduction, BRAM delta, routed WNS, route errors, and fully-routed net count.
  Rerun final signoff remains expected-blocked with the same two external blockers: C3b AXIS/DMA physical smoke JSON and XR-VITs exact source/replacement policy.
- Hardened final unblock command card:
  Spark sidecar `019eccf6-f127-7c91-9302-f6d3d5d57f32` found that U0 local blocker checks lacked required output paths.
  Updated `tools/write_final_unblock_commands.py` so U0 current/exact/replacement readiness commands write JSON/Markdown outputs.
  Added optional U5 QKV URAM successor smoke/promotion commands while keeping default final blockers unchanged.
  `docs/CHOICE.md` now records QKV Q1/Q2/Q3 choices and local blocker snapshot command.
  Final runner regenerated command card with `sections=6`; final state remains expected-blocked with C3b physical smoke and XR-VITs gate.
- Integrated U5/QKV into final unblock closeout packet:
  Spark sidecar `019eccfa-dfa3-7fc2-ae05-c977739d0a6c` found the closeout packet ignored command-card U5 and had no QKV successor gate.
  Added `qkv_uram_successor_gate`, `operator_commands.qkv_uram_successor`, and QKV expected successor outputs to `tools/write_final_unblock_closeout_packet.py`.
  Added validator checks for QKV optional status, csynth pass, resource keys, physical-smoke path, and U5 execute/import commands.
  Final runner regenerated closeout validation with `40/40` checks, fail `0`; default blockers remain unchanged.
- Hardened QKV U5 audit and final manifest checks:
  Spark sidecar `019eccfe-dcb8-79d2-a42e-05cd75c058d0` confirmed QKV is optional and recommended conditional hard-gate checks if QKV is later selected as required.
  `tools/write_third_goal_current_audit.py` now reports command-card section ids, `has_qkv_uram_u5`, closeout QKV gate, and QKV command count.
  `tools/write_final_evidence_manifest.py` now includes conditional QKV required-smoke status/JSON checks.
  Latest final evidence manifest is `62/62` pass with `47` consistency checks; current audit sees `has_qkv_uram_u5=True`.
- Normalized final signoff safety schema:
  Spark sidecar `019ecd04-90f4-78e0-b98b-3de1b9061716` found command-card and closeout safety keys were split across `executes_network` and `executes_commands`.
  Command card and closeout packet now emit both keys as `False`; bundle validator and closeout validator accept/normalize both.
  Rerun final signoff remains expected-blocked with two external blockers, but bundle validation is `pass`, fail `0`, checks `64`; closeout validation remains `40/40`.
- Clarified final signoff blocker/date provenance:
  Spark sidecar `019ecd09-7b8f-7593-b526-6650c8083785` recommended default-blocker-to-path mapping and explicit manifest date-tag provenance.
  `tools/write_third_goal_current_audit.py` now emits `blocked_external_input_paths_by_blocker` and `remaining_external_input_details`.
  `tools/write_final_evidence_manifest.py` now emits per-artifact `source_date_tag`, `artifact_date_tags`, `date_tag_policy`, and `date_tag_profile`.
  Rerun final signoff remains expected-blocked with two external blockers; source audit is `required_count=74`, `source_count=195`, `missing_required=0`; final manifest remains `62/62` pass.
- Promoted VREF-P0-02 buffer lifetime/resource-placement audit into required final evidence:
  `tools/write_final_evidence_manifest.py` now requires `vref_p0_buffer_lifetime_audit_2026_06_16.{json,md}` and checks large-buffer URAM, small-memory LUTRAM, Q/K/V URAM, pooled/score/probability LUTRAM, and C3b DSP/LUT policy deltas.
  `tools/run_third_goal_final_signoff.py` now regenerates and mirrors the buffer lifetime audit before final manifest generation.
  Validation: `30` focused tests passed; py_compile passed; final manifest is `64/64` pass with `61` consistency checks and `0` failures; source audit is `required_count=74`, `source_count=197`, `missing_required=0`.
- Hardened C3b canonical physical-smoke gate:
  Spark sidecar `019ecd14-cbbf-7d42-af3c-cd934fbbf86c` reviewed XR-VITs policy integrity and recommended fingerprint/snapshot strengthening for the next policy step.
  Main agent tightened `tools/check_third_goal_preflight.py` so only `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` clears C3b physical-smoke evidence; generated bundle/remote JSON is now import-required.
  `tools/write_c3b_physical_smoke_gate_audit.py` now reports canonical validation JSON status/preset/result/source hash when present.
  Validation: `29` C3b/preflight tests passed; py_compile passed; final runner remains expected-blocked with `ok=96`, `warn=5`, `fail=2`, final manifest `64/64`, source audit `source_count=197`.
- Hardened XR-VITs replacement policy integrity:
  Spark sidecar `019ecd1a-484d-7e23-b0bf-fd769e59e878` reviewed compatibility rules and recommended fingerprint mismatch as a hard signoff blocker.
  `tools/create_xr_vits_replacement_policy.py` now emits candidate audit fingerprint, recommendation snapshot, candidate meta, and policy fingerprint.
  `tools/check_final_blocker_closure_readiness.py` and `tools/check_third_goal_preflight.py` now reject legacy/drifted policies for Req11 closure while still parsing them and surfacing regeneration guidance.
  `tools/write_xr_vits_gate_audit.py` exposes active policy integrity status and fingerprints.
  Validation: `64` policy/preflight/runner tests passed; py_compile passed; final runner remains expected-blocked with the same two external blockers and final manifest `64/64`.
- Propagated XR-VITs fingerprint-bound policy into operator artifacts:
  Spark sidecar `019ecd1f-eff9-7c53-9642-0d4d1f5acc3e` found gaps in final unblock commands, closeout packet, operator handoff, and validators.
  Command card, closeout packet, operator handoff, closeout validator, handoff validator, and bundle validator now require P2 policy integrity fields and live policy validation when a policy file exists.
  Current no-policy state is explicitly `pending-policy-creation`; legacy/drifted policy files fail instead of clearing Req11.
  Validation: `31` focused tests passed; `60` broader final-signoff tests passed; py_compile passed; final runner remains expected-blocked with final preflight `ok=96`, `warn=5`, `fail=2`, closeout validation `49/49`, final manifest `64/64`, source audit `74/197`.
- Surfaced XR-VITs policy integrity in current third-goal audit:
  Spark spawn for the next read-only gap review failed with `agent thread limit reached`, so the main agent completed local fallback.
  `tools/write_third_goal_current_audit.py` now emits `xr_vits_policy_integrity` at top level and under `gate_modes`, with command-card/handoff summaries, validation status, policy existence, and `operator_handoff_policy_check_count=9`.
  Regenerated current audit and final runner artifacts. Validation: `30` focused current-audit/final-signoff tests passed; py_compile passed; final runner remains expected-blocked with only C3b physical smoke and XR-VITs source/policy unresolved.
- Surfaced XR-VITs policy integrity in third-goal completion audit:
  Spark spawn for completion-audit read-only review failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/write_third_goal_completion_audit.py` now consumes current audit and records `xr_vits_policy_integrity`, external blocker path mapping, current audit status, policy validation status, and policy check count in JSON/Markdown.
  Completion audit remains correctly `blocked` because C3b physical smoke and XR-VITs source/policy are unresolved, but Req11 now carries `policy integrity consistent=True`, `policy validation status=pending-policy-creation`, and `policy check count=7`.
  Validation: `37` focused tests passed; py_compile passed; final runner regenerated completion/current/mirrored artifacts.
- Refreshed completion audit Req0 current handover evidence:
  Spark spawn for Task Card `T-3G-REQ0-HANDOVER-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/write_third_goal_completion_audit.py` now checks `docs/track/HANDOVER.md` and `docs/track/log.md` as current continuity evidence.
  Validation: `4` completion-audit tests passed; `32` broader final-signoff tests passed; final runner remains expected-blocked with completion audit `pass=8`, `partial=4`, `blocked=2`, final manifest `64/64`, preflight `ok=96`, `warn=5`, `fail=2`.
- Strengthened requirements trace XR-VITs closure conditions:
  Spark spawn for Task Card `T-3G-NEXT-GAP-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/write_third_goal_requirements_trace.py` now exposes `xr_vits_policy_integrity` and requires fingerprint-bound replacement-policy fields before the `requested XR-VITs sibling` blocker can clear via replacement.
  Current trace records required fields complete `True`, legacy/drifted policy clears final signoff `False`, status `blocked`, blocked requirement ids `['11']`, partial ids `['2', '3', '4', '10']`.
  Validation: `3` trace tests passed; `31` broader final-signoff tests passed; py_compile passed; final runner remains expected-blocked with final manifest `64/64`, preflight `ok=96`, `warn=5`, `fail=2`; diff check passed.
- Added final-manifest gate for requirements trace policy integrity:
  Spark spawn for Task Card `T-3G-MANIFEST-TRACE-POLICY-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/write_final_evidence_manifest.py` now checks requirements-trace status, Req11 blocked state, XR-VITs required policy fields completeness, and legacy/drifted policy rejection.
  Validation: `9` manifest tests passed; `32` broader final-signoff tests passed; py_compile passed; final runner remains expected-blocked with final manifest `64/64`, consistency checks `65`, failed consistency `0`, preflight `ok=96`, `warn=5`, `fail=2`.
- Added final-bundle gate for manifest trace-policy checks:
  Spark spawn for Task Card `T-3G-BUNDLE-TRACE-POLICY-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/validate_final_signoff_bundle.py` now checks that the final evidence manifest contains and passes all requirements-trace XR-VITs policy consistency checks.
  Validation: `8` bundle-validator tests passed; `31` broader final-signoff tests passed; py_compile passed; final runner remains expected-blocked with bundle validation `pass`, checks `74`, fail `0`, preflight `ok=96`, `warn=5`, `fail=2`.
- Raised final-operator handoff evidence-contract threshold:
  Spark spawn for Task Card `T-3G-HANDOFF-CONTRACT-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/validate_final_operator_handoff.py` now requires `final_evidence_manifest_contract.consistency_count >= 67`, matching the current manifest trace-policy gates.
  Validation: `10` operator-handoff tests passed; `32` broader final-signoff tests passed; py_compile passed; final runner remains expected-blocked with operator handoff validation `pass`, checks `44`, fail `0`, preflight `ok=96`, `warn=5`, `fail=2`.
- Added final validator count freshness gates:
  Spark spawn for Task Card `T-3G-VALIDATION-DOCS` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/validate_final_signoff_bundle.py` now emits `check_count=74`; `tools/write_final_evidence_manifest.py` now checks operator handoff validator `check_count>=44` and bundle validator `check_count>=74` without creating a cyclic validator pass/fail dependency.
  Validation: `19` focused tests passed; `43` broader final-signoff tests passed; py_compile passed; final runner remains expected-blocked with final manifest `64/64`, consistency checks `67`, failed consistency `0`, operator handoff validation `pass` checks `44`, bundle validation `pass` checks `74`, preflight `ok=96`, `warn=5`, `fail=2`.
- Hardened smoke import destination gate revalidation:
  Spark spawn for Task Card `T-3G-UNBLOCK-GAP-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/import_pynq_smoke_result.py` now revalidates destination JSON after active copy, records destination SHA256/source match, and exposes `would_clear_current_gate`.
  Dry-run and custom destination imports remain validation-only and cannot be mistaken for current gate closure.
  Validation: `35` import/runner/readiness/PYNQ validation tests passed; py_compile passed.
- Added non-active XR-VITs replacement policy preview evidence:
  Spark spawn for Task Card `T-3G-XRVITS-PREVIEW-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/create_xr_vits_replacement_policy.py` now writes preview JSON/Markdown only in `--dry-run` mode, and `tools/run_third_goal_final_signoff.py` emits the preview every run without activating a policy.
  Preview state is `pass`, `preview_only=True`, `active_policy_written=False`, integrity `pass`, fingerprint `4882e521d6700319905085d3d1641036aa410d3be8a5210793091308aa2537fd`.
  Source audit now requires `76` artifacts and sees `199` sources with `0` missing; final evidence manifest is `66/66` pass with `72` consistency checks and `0` failures.
  Final operator handoff validation remains `pass` with `44` checks; final signoff bundle validation remains `pass` with `74` checks.
  Final runner remains expected-blocked with only the two external blockers: C3b physical smoke JSON and XR-VITs exact source or active approved replacement policy.
- Aligned completion audit and requirements trace with current evidence split:
  Spark spawn for Task Card `T-3G-GAP-AUDIT-002` failed with `agent thread limit reached`; main agent completed local fallback.
  Req2, Req3, and Req10 now pass in completion audit when current fallback/spec evidence, C3b baseline artifacts, and XR-VIT reference evidence are present.
  Req4 remains partial for missing physical ZCU104 E2E smoke; Req11 remains blocked for missing exact XR-VITs or active approved replacement policy.
  Validation: `33` focused completion/trace/manifest/runner tests passed; py_compile passed.
  Final runner remains expected-blocked with completion audit `pass=11`, `partial=1`, `blocked=2`; requirements trace blocked ids `['11']`, partial ids `['4']`; final manifest `66/66`, consistency `72`; source audit `76/199`.
- Strengthened HG-PIPE operator property coverage and current Req10 split:
  Spark sidecar `019ecd5b-edd9-77a1-9698-7d43691cc23d` identified operator coverage as the next internal gap after excluding board smoke and XR-VITs active policy approval.
  `tools/write_hgpipe_operator_audit.py` now adds deterministic contract property checks for table lengths, cursor bounds, output ranges, Softmax exp monotonicity/nonnegativity, LayerNorm rsqrt positivity, and quantize-clamp bit ranges.
  HG-PIPE operator audit remains `pass`: reference checks `97/97`, sampled values `5899008`, property checks `211/211`, fail `0`.
  `tools/write_third_goal_current_audit.py` now treats Req10 as reference-usage evidence and leaves exact XR-VITs source/replacement approval solely under Req11.
  Validation: `37` focused tests passed; py_compile passed; final runner remains expected-blocked with current audit reflected `10`, partial `1`, blocked `1`, final manifest `66/66`, source audit `76/199`.
- Integrated HG-PIPE operator audit into final evidence manifest and validator contracts:
  Spark sidecar `019ecd62-a1f7-7912-87b4-8088eee4f200` completed read-only review and recommended binding `property_summary` into final signoff evidence.
  `tools/write_final_evidence_manifest.py` now requires `docs/resources/hgpipe_operator_audit_2026_06_16.{json,md}` and checks HG-PIPE audit status, reference checks `97/97`, sampled values `5899008`, property summary `211/211`, all operator status, and per-operator property fail counts.
  `tools/validate_final_operator_handoff.py` now requires evidence contract consistency count `>=83`; `tools/validate_final_signoff_bundle.py` requires HG-PIPE artifact ids and key HG-PIPE consistency checks to be present/pass.
  `tools/run_third_goal_final_signoff.py` now regenerates and mirrors HG-PIPE operator audit evidence before final manifest generation.
  Validation: `45` focused manifest/validator/runner tests passed; `54` broader final-signoff tests passed; py_compile passed.
  Final runner remains expected-blocked with final manifest `68/68`, consistency checks `83`, failed consistency `0`; source audit `76/201`; operator handoff validation `44/44`; bundle validation `74/74`; blockers remain C3b physical smoke and XR-VITs source/approval.
- Added spec/plan current-doc freshness gate:
  Spark spawn for Task Card `T-3G-NEXT-GAP-001` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/write_spec_plan_conformance_audit.py` now checks hardware-local `docs/Spec.md`, `docs/track/PROGRESS.md`, and `docs/track/HANDOVER.md` against live final manifest/source/current-audit counts.
  Required current counts are final manifest `68/68`, consistency checks `83`, source audit `76/201`, and current audit reflected `10`, partial `1`, blocked `1`; stale tokens `66/66`, `72`, and `199` are rejected in the current-doc gate.
  Validation: `4` focused spec-plan tests passed; `47` broader final-signoff/spec-plan tests passed; py_compile passed.
  Current spec/plan conformance audit is `pass` with `46/46` checks and `0` failures.
  Final runner remains expected-blocked with spec-plan conformance `46/46`, final manifest `68/68`, consistency checks `83`, source audit `76/201`, and only the external C3b physical-smoke/XR-VITs source-or-approval blockers remaining.
- Promoted VREF-P0-01 PoT scale audit/sweep into final evidence:
  Spark spawn for Task Card `T-3G-INTERNAL-GAP-002` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/run_third_goal_final_signoff.py` now regenerates and mirrors PoT scale audit/sweep JSON/Markdown artifacts, `tools/write_third_goal_source_audit.py` requires them, and `tools/write_final_evidence_manifest.py` checks audit pass/count plus sweep coverage/current-scale recommendation.
  `tools/validate_final_operator_handoff.py` now requires evidence contract consistency count `>=88`; `tools/validate_final_signoff_bundle.py` requires the PoT artifact ids and consistency checks.
  Validation: `54` focused PoT/source/spec/manifest/validator/runner tests passed; py_compile passed.
  Final runner remains expected-blocked with spec-plan conformance `46/46`, final manifest `72/72`, consistency checks `88`, source audit `80/205`, operator handoff validation `44/44`, bundle validation `74/74`, and only the external C3b physical-smoke/XR-VITs source-or-approval blockers remaining.
- Hardened final signoff runner one-pass self-consistency:
  Spark Task Card `T-3G-RUNNER-SELF-CONSISTENCY-001` hit usage limit, so GPT5.5 fallback performed read-only review and confirmed stale `final_evidence_manifest` contract risk.
  `tools/run_third_goal_final_signoff.py` now refreshes requirements trace, final operator handoff, handoff validation, bundle validation, and final evidence manifest after the first final evidence manifest pass.
  Added regression test `test_final_refresh_uses_fresh_evidence_manifest_contract`.
  Validation: runner unit tests `15` passed; broader signoff tests `66` passed; py_compile passed; final runner expected-blocked but one-pass refreshed final manifest `72/72`, source audit `80/205`, handoff validation `pass`, bundle validation `pass`.
  Post-run direct validators against current `docs/resources` also passed with fail `0`.
- Reflected RMU/SMU small-memory LUTRAM placement in third-goal final evidence:
  `hls/src/rmu_smu.cpp` now binds SMU `score` and `prob` token scratch arrays to LUTRAM.
  `tools/write_vref_p0_buffer_lifetime_audit.py` records both RMU/SMU checks, raising buffer-lifetime audit coverage to `38/38`.
  `tools/write_final_evidence_manifest.py`, final handoff validation, bundle validation, and spec-plan freshness tests now use final evidence consistency count `90`.
  Validation: focused final-evidence/spec-plan tests `35` passed; C++ syntax check for `hls/src/rmu_smu.cpp` passed; final signoff runner remains expected-blocked only on C3b physical-smoke JSON and exact XR-VITs source or approved replacement policy.
- Linked QKV successor URAM evidence into VREF-P0-02 buffer lifetime/resource-placement audit:
  the audit now checks Q/K/V URAM branch refs in `hls/include/hgtxr_e2e_vit.hpp`, successor CSim/CSynth/routed overlay pass state,
  `+8` URAM / `-30` BRAM delta versus `dsp_mixed_stream`, positive URAM use, and physical-smoke pending-only promotion state.
  Buffer-lifetime audit coverage is now `51/51`.
  Final evidence manifest is `72/72` pass with consistency count `103` and failed consistency `[]`.
- Added Req6 HLS legality guard evidence:
  `tools/write_req6_parameterization_audit.py` now checks legal bus/data/weight divisibility, accumulator-width relation,
  head/FF dimension consistency, dense-parallelism divisibility, packed weight-lane compatibility, and positive buffer/FIFO sizing.
  Negative tests cover illegal bus width and illegal dense parallelism.
  Req6 audit is now `pass`, checks `53/53`.
  Final evidence manifest is now `72/72` pass with consistency count `119` and failed consistency `[]`.
- Added E2E compile-time static assertions for Req6 parameter legality:
  `hls/include/hgtxr_e2e_vit.hpp` now checks active token bounds, head dimension coverage, dense-parallelism positivity/divisibility,
  feed-forward dimension divisibility, packed weight lanes, and AXIS width consistency at compile time.
  Req6 audit now verifies these header guards and reports `62/62`.
  Final evidence manifest consistency count is now `128` with failed consistency `[]`.
- Promoted C3b resource-policy proof from matrix-only checks to direct `csynth.xml` consistency checks:
  resource-policy audit now checks C3b report existence, resource match, latency match, DSP threshold, and URAM threshold.
  Final evidence manifest consistency count is now `150` with failed consistency `[]`.
- Added Req1 environment audit and final evidence gates:
  Ubuntu 22.04, non-WSL kernel, `/home/kjm26/project` workspace root, and `/tools/Xilinx` Vitis/Vivado executable paths are checked without running Xilinx tools.
  Final evidence manifest is now `74/74` pass with consistency count `161` and failed consistency `[]`.
- Added Req9 DeiT image reference audit and final evidence gates:
  requested `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png`, HGPIPE substitute, and hardware docs copy are checked for PNG magic and SHA256 match.
  Final evidence manifest is now `76/76` pass with consistency count `170` and failed consistency `[]`; source audit is `84/213`.
- Promoted direct hard-blocker gate audits into final evidence:
  `tools/write_final_evidence_manifest.py` now requires `xr_vits_gate_audit_2026_06_16.{json,md}` and
  `c3b_physical_smoke_gate_audit_2026_06_16.{json,md}` and checks XR-VITs gate status/replacement-candidate evidence plus C3b ready-for-board/bundle/session evidence.
  `tools/run_third_goal_final_signoff.py` regenerates and mirrors both gate audits before final manifest refresh.
  `tools/validate_final_signoff_bundle.py` now requires these artifact ids and key manifest checks.
  Spark `T-XRVITS-VERIFY` hit GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback read-only review confirmed Req11 must stay blocked without exact XR-VITs or explicit replacement-policy approval.
  Validation: `50` focused manifest/validator/runner tests passed; final runner remains expected-blocked with final manifest `80/80`, consistency `184`, source audit `84/217`, bundle validation `pass`, and remaining blockers `requested XR-VITs sibling` plus `C3b AXIS/DMA physical smoke result`.
- Cleaned C3b smoke candidate discovery metadata noise:
  `tools/discover_c3b_smoke_candidates.py` now skips `docs/resources/c3b_physical_smoke_gate_audit_` artifacts so gate/status JSON is not treated as a failed physical-smoke candidate.
  Regression coverage added for the exact metadata filename.
  Spark `T-C3B-UNBLOCK-GAP-VERIFY` hit GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback confirmed the skip marker and regenerated discovery artifacts are clean.
  Validation: C3b discovery tests `6` passed; regenerated discovery reports `status=missing`, `pass=0`, `candidates=0`; final runner remains expected-blocked with final manifest `80/80` and final bundle validation `pass`.
- Cleaned generic PYNQ smoke candidate discovery metadata noise:
  local audit reproduced that docs-resource `c3b_physical_smoke_gate_audit_2026_06_16.json` can match the generic C3b pattern unless name-level `_gate_audit_` is skipped.
  `tools/discover_pynq_smoke_candidates.py` now excludes `_gate_audit_` names across presets.
  GPT5.5 sidecar verified generated-signoff discovery artifacts were already clean and recommended metadata regression coverage; tests now cover docs-resource and generated-signoff metadata noise.
  Validation: generic/C3b discovery tests `14` passed; manual generic C3b discovery reports `status=missing`, `pass=0`, `candidates=0`.
- Hardened PYNQ smoke overlay provenance:
  GPT5.5 read-only evaluator found that C3b smoke JSON could pass with correct numeric output but wrong non-empty bit/hwh paths.
  `tools/validate_pynq_smoke_result.py` now checks preset-specific bit/hwh basenames for C3b, VREF DSP mixed stream, QKV URAM, A1/C1, and m_axi when `require_paths=True`; importer and discovery inherit this gate.
  Regression coverage rejects wrong C3b bit/hwh basename and wrong-overlay import without copying.
  Validation: focused validator/import/discovery/readiness tests `39` passed; final-signoff-related tests `74` passed; py_compile passed; generic and legacy C3b discovery remain `status=missing`, `pass=0`, `candidates=0`.
  Final runner remains expected-blocked with preflight `ok=96`, `warn=5`, `fail=2`, final manifest `80/80`, consistency `184`, source audit `84/217`, and only the external C3b physical-smoke/XR-VITs source-or-policy blockers remaining.
- Promoted overlay-provenance hardening into final evidence manifest:
  `tools/write_final_evidence_manifest.py` now checks validator source for required overlay prefixes, basename comparison logic, and `require_paths` gating.
  `tools/run_third_goal_final_signoff.py` now includes a spec-plan final refresh followed by a post-spec final-manifest refresh, removing the hidden two-run convergence issue after manifest consistency-count changes.
  Validation: manifest/runner/spec-plan/bundle/handoff tests `56` passed; py_compile passed; final runner remains expected-blocked with spec-plan `46/46`, final manifest `80/80`, consistency `187`, failed consistency `0`, source audit `84/217`, and only the external C3b physical-smoke/XR-VITs source-or-policy blockers remaining.
- Promoted overlay-provenance hardening into final signoff bundle validation:
  `tools/validate_final_signoff_bundle.py` now requires `pynq_smoke_validator_overlay_prefix_contracts`,
  `pynq_smoke_validator_basename_check`, and `pynq_smoke_validator_require_paths_gate` to be present/pass in the live final evidence manifest.
  Regression coverage added for a manifest missing the PYNQ basename gate.
  Validation: `python3 -m unittest tests.test_validate_final_signoff_bundle` passed (`9` tests);
  focused manifest/runner/spec-plan/handoff/bundle tests passed (`57` tests);
  py_compile passed; final runner remains expected-blocked with preflight `ok=96`, `warn=5`, `fail=2`,
  final manifest `80/80`, consistency `187`, failed consistency `0`, bundle validation `pass`, and only the external C3b physical-smoke/XR-VITs source-or-policy blockers remaining.
- Promoted generic PYNQ discovery artifacts into the final evidence chain:
  after closing one completed stale subagent, a Spark evaluator spawn still failed with GPT-5.3-Codex-Spark usage limit.
  Main-agent fallback added final-runner generation/mirroring for
  `pynq_smoke_candidate_discovery_c3b_2026_06_16.{json,md}` and
  `pynq_smoke_candidate_discovery_vref_p0_2026_06_16.{json,md}` while preserving legacy discovery artifacts.
  `tools/write_final_evidence_manifest.py` now requires those artifacts and checks their status/pass-count fields;
  `tools/validate_final_signoff_bundle.py` now requires the corresponding artifact ids and consistency checks.
  Validation so far: py_compile passed; focused runner/manifest/bundle tests passed (`45` tests).
- Added semantic/safety final-evidence gates for generic PYNQ discovery:
  GPT5.5 sidecar completed a read-only evaluator pass and identified that QKV URAM discovery was required as an artifact but not semantically checked.
  `tools/write_final_evidence_manifest.py` now checks generic C3b/VREF/QKV discovery preset, status, candidate/pass counts,
  count bounds, missing-state recommended-candidate behavior, and no-side-effect safety fields.
  `tools/validate_final_signoff_bundle.py` requires those checks, and `tools/write_third_goal_current_audit.py`
  now reflects final-runner C3b/VREF generic discovery statuses in the runner summary.
  Validation: semantic gate/current-audit tests passed (`34` tests); expanded final-signoff/spec-plan regression tests passed (`53` tests);
  py_compile passed; `git diff --check` passed.
  Final evidence manifest is `pass`, required artifacts `84/84`, consistency checks `239`, failed consistency `0`.
  Source audit is `pass`, required artifacts `84`, source count `221`, missing required `0`.
  Final bundle validation is `pass`, check count `74`, fail count `0`.
  Final runner remains expected-blocked only on external closure inputs: exact XR-VITs source or approved replacement policy, and board-produced C3b AXIS/DMA physical-smoke JSON.
- Added blocker-readiness discovery summaries and runner blocker-path evidence:
  Spark `T-301` again hit the GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback completed read-only audit and recommended aligning
  `final_blocker_closure_readiness` with the generic PYNQ discovery gates.
  `tools/check_final_blocker_closure_readiness.py` now includes side-effect-free C3b/VREF-P0/QKV URAM `pynq_discovery` summaries.
  `tools/run_third_goal_final_signoff.py` now records `remaining_blocker_input_paths` and `remaining_blocker_details`.
  Validation: focused readiness/runner/discovery/bundle/manifest tests passed (`62` tests); py_compile passed; scoped `git diff --check` passed.
  Final runner remains expected-blocked with final manifest `84/84`, consistency `239`, source audit `84/221`, bundle validation `pass`.
  Runner blocker paths now point directly to the C3b canonical smoke JSON, exact `/home/kjm26/project/PRJXR/XR-VITs`, and active replacement-policy path.
- Promoted blocker-readiness and final-runner unblock path evidence into final manifest/bundle gates:
  Spark `T-3G-READONLY-EVAL` completed read-only review and identified that the new runner blocker path fields and blocker-readiness discovery summaries were not yet hard bundle gates.
  `tools/write_final_evidence_manifest.py` now checks final-runner blocker path source contract, generated runner summary payload keys/non-empty paths,
  final blocker-readiness C3b/XR-VITs exact input paths, and side-effect-free C3b/VREF-P0/QKV URAM `pynq_discovery` summaries.
  `tools/validate_final_signoff_bundle.py` now requires these checks to be present/pass.
  Validation: focused manifest/bundle tests passed (`36` tests); broader signoff regression passed (`64` tests); py_compile passed.
  Final manifest is `84/84`, consistency `239`, failed consistency `0`; final bundle validation remains `pass`, fail `0`; final runner remains expected-blocked only on external C3b physical smoke and XR-VITs source/policy.
- Hardened operator handoff live evidence contract:
  Spark `T-3G-OPERATOR-HANDOFF-LIVE-EVIDENCE-GATE` completed a read-only stale-contract audit and identified that operator handoff validation could accept an embedded final-evidence contract without comparing it to the live manifest.
  `tools/validate_final_operator_handoff.py` now requires live `docs/resources/final_evidence_manifest_2026_06_10.json` contract presence, exact embedded/live equality, consistency count `>=239`, and failed consistency `[]`.
  `tools/validate_final_signoff_bundle.py` keeps the same `>=239` evidence consistency threshold.
  Regression coverage rejects stale embedded `238` and missing live manifest cases.
  Validation: focused operator/bundle/spec-plan tests passed (`25` tests); py_compile passed; final runner remains expected-blocked with final manifest `84/84`, consistency `239`, operator handoff validation `46/46`, bundle validation `74/74`, spec-plan `46/46`.
- Extended Req6/sweep parallelism space for PAR16/PAR32:
  Spark sidecar for next-action audit errored from output-token overflow, so the main agent used current HANDOVER/resource evidence directly.
  `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml` now covers `parallelism_factor: [1, 2, 4, 8, 16, 32]`.
  C3b PAR16 is recorded as validated resource-matrix evidence; PAR32 is explicit exploratory and requires fresh csynth/routed evidence before promotion.
  `tools/write_req6_parameterization_audit.py` now checks CSim/CSynth PAR16/PAR32 override support; Req6 audit reports `64/64`.
  `tools/write_final_evidence_manifest.py` promotes those override checks into final consistency checks.
  Validation: focused Req6/manifest/completion tests passed (`57` tests); py_compile passed; Req6 audit `64/64`; final runner remains expected-blocked with final manifest `84/84`, consistency `241`, failed consistency `0`, bundle validation `74/74`.
- Aligned final blocker readiness no-require-paths semantics:
  Spark sidecar `019ece20-161b-7650-aefc-9b5031fbb542` recommended one safe unblocked fix: use the existing `c3b_require_paths` flag for current canonical C3b validation in `tools/check_final_blocker_closure_readiness.py`.
  Implemented behavior change: `--c3b-no-require-paths` now controls both current and candidate C3b validation.
  Strict mode remains unchanged; relaxed mode only applies when explicitly requested.
  Validation: readiness tests `8` passed; manifest/bundle/readiness focused tests `44` passed; py_compile and scoped `git diff --check` passed; CLI readiness snapshot remains expected-blocked because canonical C3b JSON and XR-VITs/policy are still missing.
- Integrated C3b transfer manifest regeneration into final runner:
  `tools/run_third_goal_final_signoff.py` now calls `write_c3b_smoke_transfer_manifest.py` and mirrors transfer JSON/Markdown plus bundle SHA256 into `docs/resources`.
  Summary now exposes `c3b_transfer_manifest_status=pass`.
  Validation: transfer/runner tests `18` passed; broader focused signoff tests `62` passed; standalone transfer manifest pass with sha256 `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`; final runner remains expected-blocked with final manifest `84/84`, consistency `241`, operator handoff `46/46`, bundle `74/74`.
- Promoted C3b transfer manifest semantics into final evidence gates:
  Spark sidecar `019ece2b-b512-7ed0-bcf0-f8bc0649bd62` completed read-only review and recommended semantic checks for transfer-manifest status, preset/variant/target, tar shape, transfer files, clean bundle validation, expected outputs, board verify/run commands, host copyback, SHA256 format, and readiness alignment.
  `tools/write_final_evidence_manifest.py` now enforces those transfer/readiness checks as manifest consistency gates.
  `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require the live final evidence contract consistency count `>=256`.
  Validation: focused manifest/bundle/operator/spec-plan tests passed (`52` tests); py_compile passed; final runner remains expected-blocked with `c3b_transfer_manifest_status=pass`, final manifest `84/84`, consistency `256`, failed consistency `0`, spec-plan `46/46`, operator handoff `46/46`, bundle `74/74`.
- Promoted Req6 PAR32 exploratory policy into final evidence gates:
  Spark sidecar `019ece38-3276-7553-956e-10adbdf6c3dd` recommended local hardening around external unblock readiness and VREF/PAR resource-fit checks; main agent selected the parallelism/resource-fit path to match the current user objective.
  `tools/write_req6_parameterization_audit.py` now parses `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml` and verifies that C3b PAR16 is the validated resource-matrix path while PAR32 remains exploratory until fresh csynth, routed timing, resource-fit audit, and no-C3b-overwrite evidence exist.
  `tools/write_final_evidence_manifest.py` now consumes those Req6 extension checks as final consistency gates.
  `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require the live final evidence contract consistency count `>=263`.
  Validation: focused Req6/manifest tests passed (`32` tests); py_compile passed; final runner remains expected-blocked with Req6 `71/71`, final manifest `84/84`, consistency `263`, failed consistency `0`.
- Promoted XR-VITs replacement-policy preview path contract into final evidence gates:
  Spark sidecar `019ece3f-c950-7561-9b96-45de6868aec8` completed read-only review and identified XR-VITs pre-approval hardening gaps around candidate-audit path policy and approval metadata.
  The main agent implemented the lower-risk path-contract part: `tools/write_final_evidence_manifest.py` now checks the preview active policy path, tracked candidate-audit relative path, and integrity candidate-audit absolute path.
  `tools/validate_final_signoff_bundle.py` now requires those checks, and final evidence consistency threshold is `>=266`.
  Validation: focused preview/validator tests passed (`48` tests); py_compile passed; final runner remains expected-blocked with final manifest `84/84`, consistency `266`, failed consistency `0`.
- Promoted XR-VITs replacement-policy approval-event contract into final evidence gates:
  Spark sidecar `019ece4b-58f0-7693-9565-b1db2269fe04` completed read-only review and recommended treating approval metadata as a contract, not descriptive text.
  `tools/create_xr_vits_replacement_policy.py` now emits `approval_event` with protocol, event type, approver, approved-at, reason code, reason, replacement role, requested path, replacement path, candidate-audit path, candidate-audit fingerprint, generator, and event-id hash.
  `tools/write_final_evidence_manifest.py` now checks approval-event schema, policy-field match, and event-id hash.
  `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency threshold `>=269`.
  Validation: focused policy/manifest/validator tests passed (`57` tests); py_compile passed; final runner remains expected-blocked with final manifest `84/84`, consistency `269`, failed consistency `0`.
- Hardened active XR-VITs replacement-policy approval metadata:
  active policy writes now reject placeholder approver values such as `<approved-by>`, while dry-run preview remains allowed for review artifacts.
  The final signoff runner now reports active policy creation failure as blocked instead of ignoring that failure.
  Validation: focused policy/runner/manifest/validator/spec-plan tests passed (`79` tests); py_compile passed; scoped diff check passed.
  Final runner remains expected-blocked with preflight `ok=96`, `warn=5`, `fail=2`, final manifest `84/84`, spec-plan conformance `46/46`, source audit `84/221`, and the same two external blockers.
- Promoted X6 active-placeholder guard into final manifest source-contract gates:
  `tools/write_final_evidence_manifest.py` now checks the policy tool placeholder guard, dry-run-only allowance, and final-runner active policy failure blocking.
  `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency threshold `>=272`.
  Validation: focused policy/runner/manifest/operator/bundle/spec-plan tests passed (`80` tests); py_compile passed; scoped diff check passed.
  Final runner remains expected-blocked with preflight `ok=96`, `warn=5`, `fail=2`, final manifest `84/84`, consistency `272`, failed consistency `0`, spec-plan conformance `46/46`, operator handoff validation `pass`, bundle validation `pass`, and the same two external blockers.
- Promoted manual Spec fallback into final manifest gates:
  `tools/write_final_evidence_manifest.py` now checks spec-plan conformance for spec-kit/manual fallback, ZCU104, Q4W/Q8A, parameter knobs, A2/A1/C/PAR16, selected paths, and `/tools/Xilinx`.
  `tools/validate_final_signoff_bundle.py` and `tools/validate_final_operator_handoff.py` now require final evidence consistency threshold `>=278`.
  Validation: focused manifest/operator/bundle/spec-plan tests passed (`54` tests); py_compile passed.
  Final runner remains expected-blocked with preflight `ok=96`, `warn=5`, `fail=2`, final manifest `84/84`, consistency `278`, failed consistency `0`, spec-plan conformance `46/46`, operator handoff validation `pass`, bundle validation `pass`, and the same two external blockers.
- Selected final-runner mirrored-artifact summary dedupe:
  Spark sidecar `019ece6f-cc9c-7101-9eef-7fa89c5144d6` found duplicate entries in `generated/signoff/third_goal_final_signoff_run_2026_06_10.json` `mirrored_artifacts`, including repeated final evidence manifest paths.
  `tools/run_third_goal_final_signoff.py` now preserves order while deduplicating mirrored artifact paths before writing summary JSON/Markdown and after adding the summary artifacts themselves.
  Regression coverage checks that `mirrored_artifacts` has no duplicate paths and that `final_evidence_manifest_2026_06_10.json` appears once.
  Validation: `tests.test_run_third_goal_final_signoff` passed (`17` tests); py_compile passed; scoped diff check passed.
  Final runner remains expected-blocked with preflight `ok=96`, `warn=5`, `fail=2`, final manifest `84/84`, consistency `278`, failed consistency `0`, and `mirrored_artifacts` `90/90` unique with `0` duplicates.
- Promoted P2-ViT SW-first scale calibration into final evidence:
  `tools/write_p2_vit_scale_calibration_report.py` now combines Req5 Q4/Q8 SW-HW match, VREF-P0-01 PoT readiness, and PoT sweep evidence into one calibration decision report.
  `tools/run_third_goal_final_signoff.py` regenerates and mirrors the report; `tools/write_third_goal_source_audit.py` requires the report JSON/Markdown; `tools/write_final_evidence_manifest.py` gates report pass status, Q4W/Q8A precision, current-scale decision, candidate coverage, and no-side-effect safety.
  Validation: focused P2/source/manifest/runner tests passed (`51` tests); py_compile passed.
  Final runner remains expected-blocked with P2 report `pass`, checks `10/10`; final manifest `86/86`, consistency `283`, failed consistency `0`; source audit required `86`, sources `224`, missing `0`; blockers remain C3b physical smoke and XR-VITs source/policy.
- Hardened final unblock intake and XR-VITs root-pinned operator commands:
  Spark audit hit quota, so GPT5.5 read-only evaluator `019ece80-37ca-7e13-8f7d-7e08a7eacd09` audited the remaining blockers and confirmed the only live blockers are C3b physical smoke and XR-VITs source/policy.
  `tools/write_final_unblock_intake.py` now emits `blocker_status` and `next_inputs` for the two exact external inputs.
  `tools/check_xr_vits_reference_resolution.py` now emits root-pinned dry-run/active replacement approval commands, and `tools/audit_final_unblock_candidates.py` uses the current dry-run approval metadata validator contract.
  Validation: focused unblock/operator/runner/source/manifest tests passed (`70` tests); py_compile passed; scoped diff check passed.
  Final runner remains expected-blocked with spec-plan `46/46`, final manifest `86/86`, consistency `283`, source audit `86/224`, and the same two external blockers.
- Promoted final unblock intake and live operator-handoff cross-checks:
  GPT5.5 read-only evaluator `019ece87-2331-7ac2-bf36-63c3272dd5e8` identified that operator handoff validation did not compare embedded state against live XR-VITs resolution or final-unblock-intake JSON.
  `tools/write_final_evidence_manifest.py` now gates final unblock intake blocker-status keys, C3b next-input path, XR-VITs next-input path, and `next_inputs`/blocker-status agreement.
  `tools/validate_final_operator_handoff.py` now compares embedded handoff state to live `xr_vits_reference_resolution_2026_06_10.json` and `final_unblock_intake_2026_06_10.json`.
  `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require operator handoff validation `check_count >=52`.
  Validation: focused operator/manifest/bundle/current/completion tests passed (`88` tests); py_compile passed.
  Final runner remains expected-blocked with final manifest `86/86`, consistency `287`, failed consistency `0`, operator handoff validation `52/52`, bundle validation `74/74`, source audit `86/224`, and the same two external blockers.
- Promoted Master/Sub plan freshness into spec-plan and final evidence gates:
  GPT-5.3-Codex-Spark follow-up hit quota, so GPT5.5 read-only evaluator `019ece93-459f-74c3-904b-13172f62eec2` audited current plan freshness.
  `tools/write_spec_plan_conformance_audit.py` now checks live count freshness in Master-Plan and Sub-Plan in addition to Spec, PROGRESS, and HANDOVER.
  `tools/write_final_evidence_manifest.py` now requires spec-plan `check_count >=54` and `spec_plan_current_doc_freshness_contract`.
  Focused validation: spec-plan/manifest/bundle tests passed (`46` tests); py_compile passed.
  Final runner remains expected-blocked with spec-plan `54/54`, final manifest `86/86`, consistency `288`, failed consistency `0`, operator handoff `52/52`, bundle `74/74`, source audit `86/224`, and the same two external blockers.
- Hardened completion-audit freshness without self-referential deadlock:
  GPT-5.3-Codex-Spark sidecar hit quota and GPT5.5 fallback spawn hit the live agent-thread limit, so the main agent completed the integration fallback.
  `tools/write_third_goal_completion_audit.py` no longer recomputes final evidence consistency from inside Req12; it reads the stored final manifest contract and ignores only completion self-gates during convergence.
  `tools/write_spec_plan_conformance_audit.py` now treats final evidence as effectively passing when required artifacts are complete and remaining manifest failures are only spec-plan/completion bootstrap self-gates; unrelated manifest failures still block the audit.
  Current-doc anchors were updated to final manifest `86/86`, consistency checks `294`, source audit required `86`, sources `224`, and current audit reflected `10`, partial `1`, blocked `1`.
  Focused validation: completion/manifest/runner/spec-plan/operator/bundle tests passed (`112` tests); py_compile passed; final runner remains expected-blocked with final manifest `86/86`, consistency `294`, failed consistency `0`, spec-plan `54/54`, completion audit pass `11`, partial `1`, blocked `2`, operator handoff validation `52/52`, bundle validation `74/74`, source audit `86/224`, and the same two external blockers.
- Promoted final-runner mirrored-artifact uniqueness into final evidence:
  Spark next-gap sidecar spawn failed with `agent thread limit reached`, so the main agent audited the current final summary.
  `tools/write_final_evidence_manifest.py` now gates `mirrored_artifacts` presence, uniqueness, single final-manifest mirror entry, and single summary mirror entry.
  Current-doc anchors were updated to final manifest `86/86`, consistency checks `297`, source audit required `86`, sources `224`, and current audit reflected `10`, partial `1`, blocked `1`; live summary records `92` mirrored paths, `92` unique paths, and `0` duplicates.
- Promoted Validation/CHOICE/log freshness into spec-plan and final evidence gates:
  Spark sidecar spawn for `T-3G-FRESHNESS-SIDECAR-001` failed with `agent thread limit reached`; main agent completed the bounded integration.
  `tools/write_spec_plan_conformance_audit.py` now checks live final-manifest/source/current-audit anchors in Validation, CHOICE, and log in addition to Master/Sub/Spec/PROGRESS/HANDOVER; direct `docs/CHOICE.md` is preferred over `docs/track/CHOICE.md` when present.
  `tools/write_final_evidence_manifest.py` now requires spec-plan `check_count >=66` and the expanded `spec_plan_current_doc_freshness_contract`.
  Validation: focused spec-plan/manifest tests passed (`40` tests); broader final-signoff freshness tests passed (`118` tests); py_compile passed.
  Final runner remains expected-blocked with spec-plan `66/66`, final manifest `86/86`, consistency `297`, failed consistency `0`, operator handoff `52/52`, bundle `74/74`, source audit `86/224`, and the same two external blockers.
- Promoted RMU/SMU DSP helper coverage into resource-policy and final evidence gates:
  Spark sidecar spawn for `T-3G-RESOURCE-PRAGMA-SIDECAR-001` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `hls/src/rmu_smu.cpp` now routes RMU projection and SMU relation-stage multiply-heavy paths through DSP-bound helpers with `#pragma HLS bind_op ... impl=dsp`.
  `tools/write_e2e_resource_policy_audit.py` now checks RMU/SMU DSP bind-op presence, helper definitions, RMU projection helper use, and SMU relation helper use; audit status is `pass`, checks `27/27`.
  `tools/write_final_evidence_manifest.py` now requires resource-policy `check_count >=27` and gates those helper checks.
  Validation: focused resource-policy/manifest tests passed (`35` tests); broader final-signoff resource tests passed (`115` tests); C++ syntax check for `hls/src/rmu_smu.cpp` passed.
  Final runner remains expected-blocked with resource-policy `27/27`, final manifest `86/86`, consistency `302`, failed consistency `0`, source audit `86/224`, and the same two external blockers.
- Promoted Req5 packed Q4 binary/manifest integrity into final evidence:
  Spark sidecar spawn for `T-3G-Q4Q8-CONTRACT-SIDECAR-001` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/write_req5_q4q8_swhw_match_audit.py` now validates packed binary existence, SHA256 match, byte-count match, expected runtime state `2`, expected C3b raw output `[32, -13, 26, -6, 14, -11]`, and strict TB/CSim evidence.
  `tools/write_final_evidence_manifest.py` now gates those Req5 checks directly.
  Validation: Req5/final manifest tests passed (`37` tests); broader final-signoff related tests passed (`121` tests); py_compile passed; direct Req5 audit is `pass`, checks `11/11`, fail `0`.
  Current target is final manifest `86/86`, consistency checks `312`, failed consistency `0`, source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.
- Refreshed final evidence consistency threshold to the live count:
  `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require consistency count `>=312`, replacing the older `>=278` threshold.
  Test fixtures now generate `312` consistency checks, and stale `311` embedded contracts fail validation.
  Current target is final manifest `86/86`, consistency checks `312`, failed consistency `0`, source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.
- Promoted Req5 packed-weight final-manifest checks into final bundle validation:
  `tools/validate_final_signoff_bundle.py` now requires all Req5 Q4/Q8 packed-weight consistency checks to be present/pass, not only the Req5 JSON/Markdown artifact ids.
  Regression coverage removes `req5_q4q8_packed_weight_sha256_matches_manifest` and confirms bundle validation fails.
  Current target is final manifest `86/86`, consistency checks `312`, failed consistency `0`, source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.
- Promoted Req5 packed-weight live-manifest checks into final operator handoff validation:
  Spark sidecar spawn for `T-3G-HANDOFF-REQ5-SIDECAR-004` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` now requires all Req5 Q4/Q8 packed-weight consistency checks to be present/pass in the live final evidence manifest.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=54`, preventing stale `52`-check validation from satisfying final evidence.
  Regression coverage preserves consistency count `312` while removing `req5_q4q8_packed_weight_sha256_matches_manifest`; operator handoff validation fails on the missing Req5 check.
  Expected current target after final runner refresh is operator handoff validation `54/54`, final manifest `86/86`, consistency checks `312`, failed consistency `0`, source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.
- Promoted latest operator-handoff validator freshness into final bundle validation:
  Spark sidecar spawn for `T-3G-BUNDLE-OP-HANDOFF-COUNT-SIDECAR-005` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_signoff_bundle.py` now requires final operator handoff validation `check_count >=54`, matching the current Req5 live-manifest handoff validator contract.
  Regression coverage writes a stale `52`-check handoff validation artifact with passing status/fail count and confirms bundle validation fails on `handoff_validation_check_count`.
  Validation: py_compile passed; focused bundle/operator/final-evidence tests passed (`60` tests); broader final-signoff tests passed (`89` tests); spec-plan conformance passed `66/66`.
  Final runner remains expected-blocked with final bundle validation `74/74`, operator handoff validation `54/54`, final manifest `86/86`, consistency checks `312`, failed consistency `[]`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Applied cyclic baseline URAM/LUTRAM storage policy:
  Spark sidecar spawn for `T-3G-CYCLIC-RESOURCE-SIDECAR-006` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `hls/include/hgtxr_cyclic_transformer_params.hpp` now defines macro-controlled cyclic storage policy for weight-tile URAM, large-temporary URAM, and small-tile LUTRAM.
  `hls/src/hgtxr_top.cpp` now binds legacy cyclic packed weight tiles and large temporaries to URAM when enabled, and binds small tile scratch buffers to LUTRAM when enabled.
  `tools/write_e2e_resource_policy_audit.py` now reports resource-policy audit `36/36`; `tools/write_final_evidence_manifest.py` now requires resource-policy `check_count >=36` and gates the cyclic URAM/LUTRAM checks plus C3b threshold checks.
  Validation: hgtxr_top syntax check passed; py_compile passed; focused resource/manifest/operator/bundle tests passed (`63` tests); broader final-signoff tests passed (`88` tests); resource-policy audit is now `36/36` after C3b threshold gates.
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `321`, failed consistency `[]`, operator handoff validation target now `58/58`, bundle validation `74/74`, spec-plan `66/66`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Promoted cyclic resource-policy checks into validator named-check contracts:
  Spark sidecar spawn for `T-3G-CYCLIC-RESOURCE-NAMED-CHECK-SIDECAR-007` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` now requires the six cyclic URAM/LUTRAM resource-policy checks to be present/pass in the live final evidence manifest.
  `tools/validate_final_signoff_bundle.py` now includes the same cyclic checks in the final-manifest required named-check set.
  `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=58`.
  Regression coverage preserves final evidence consistency count `321` while removing `resource_policy_audit_cyclic_weight_tiles_uram_pragmas`; operator handoff and bundle validation fail on the missing named cyclic check.
  Validation: py_compile passed; focused operator/bundle/final-evidence tests passed (`62` tests); broader final-signoff tests passed (`90` tests).
  Final runner remains expected-blocked with operator handoff validation `58/58`, bundle validation `74/74`, final manifest `86/86`, consistency checks `321`, failed consistency `[]`, spec-plan `66/66`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Promoted C3b LUT/latency/WNS protection thresholds into resource-policy/final-evidence gates:
  Spark sidecar spawn for `T-3G-NEXT-INTERNAL-GAP-008` failed with `agent thread limit reached`; main agent selected and completed the bounded threshold-gate implementation.
  `tools/write_e2e_resource_policy_audit.py` now requires C3b csynth LUT `<=126506`, csynth latency `<=37508072`, and routed WNS `>=4.415 ns`.
  `tools/write_final_evidence_manifest.py` now requires those named checks and resource-policy `check_count >=36`; operator handoff and bundle validators now require final evidence consistency count `>=321`.
  Validation: resource-policy audit passed `36/36`; py_compile passed; focused resource/manifest/operator/bundle tests passed (`66` tests).
  Post-refresh validation: focused final-signoff tests passed (`91` tests); spec-plan conformance passed `66/66`; final runner remains expected-blocked with resource-policy `36/36`, final manifest `86/86`, consistency checks `321`, failed consistency `[]`, operator handoff validation `58/58`, bundle validation `74/74`, and the same two external blockers.
- Promoted C3b threshold checks into validator named-check contracts:
  Spark sidecar spawn for `T-3G-C3B-THRESHOLD-NAMED-CHECK-SIDECAR-009` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` now requires the C3b LUT/latency/WNS threshold final-manifest checks to be present/pass in the live final evidence manifest.
  `tools/validate_final_signoff_bundle.py` now includes the same C3b threshold checks in the final-manifest required named-check set.
  `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=58`.
  Regression coverage preserves final evidence consistency count `321` while removing `resource_policy_audit_c3b_routed_wns_gte_threshold`; operator handoff and bundle validation fail on the missing named C3b threshold check.
  Validation: py_compile passed; focused operator/bundle/final-evidence tests passed (`64` tests); broader final-signoff tests passed (`93` tests); spec-plan conformance passed `66/66`.
  Final runner remains expected-blocked with resource-policy `36/36`, final manifest `86/86`, consistency checks `321`, failed consistency `[]`, operator handoff validation `58/58`, bundle validation `74/74`, spec-plan `66/66`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Hardened final-runner summary count schema:
  Spark sidecar spawn for `T-3G-RUNNER-SUMMARY-SCHEMA-SIDECAR-011` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/run_third_goal_final_signoff.py` now writes blocker count, blocker-detail count, mirrored-artifact count, unique mirrored-artifact count, and duplicate mirrored-artifact count fields.
  `tools/write_final_evidence_manifest.py` checks those count fields against actual runner summary lists/maps.
  Regression coverage mutates blocker/detail/mirror counts and confirms final evidence fails on the new count-match gates.
  Validation: py_compile passed; focused runner/final-evidence tests passed (`51` tests); broader final-signoff tests passed (`111` tests).
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `325`, failed consistency `[]`, operator handoff validation `58/58`, bundle validation `74/74`, spec-plan `66/66`, runner blocker/detail counts `2/2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Promoted final-runner summary count checks into validator named-check contracts:
  Spark sidecar spawn for `T-3G-RUNNER-SUMMARY-NAMED-CHECK-SIDECAR-013` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` now requires `final_runner_remaining_blockers_present`, `final_runner_blocker_count_matches_list`, `final_runner_remaining_blocker_detail_count_matches`, and `final_runner_mirrored_artifact_counts_match` to be present/pass in the live final evidence manifest.
  `tools/validate_final_signoff_bundle.py` now includes those same runner-summary checks in the final-manifest required named-check set.
  `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=60`.
  Regression coverage preserves consistency count `325` while removing `final_runner_mirrored_artifact_counts_match`; operator handoff and bundle validation fail on the missing named check.
  Validation: py_compile passed; focused operator/bundle/final-evidence tests passed (`67` tests); broader final-signoff tests passed (`96` tests).
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `325`, failed consistency `[]`, operator handoff validation `60/60`, bundle validation `74/74`, spec-plan `66/66`, runner blocker/detail counts `2/2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Refreshed final-unblock closeout validation freshness gate:
  Spark sidecar spawn for `T-3G-NEXT-INTERNAL-GAP-AUDIT-014` failed with `agent thread limit reached`; main agent selected and completed the bounded freshness-gate update.
  `generated/signoff/final_unblock_closeout_packet_validation_2026_06_10.json` is current at `49/49`, fail `0`.
  `tools/write_final_evidence_manifest.py` now requires `final_unblock_closeout_validation_check_count >=49` instead of the old `>=40` threshold.
  Regression coverage mutates closeout validation to `check_count=48` and confirms final evidence fails on `final_unblock_closeout_validation_check_count`.
  Validation: py_compile passed; focused final-evidence tests passed (`34` tests); broader final-signoff/completion tests passed (`135` tests).
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `325`, failed consistency `[]`, closeout validation `49/49`, operator handoff validation `60/60`, bundle validation `74/74`, spec-plan `66/66`, runner blocker/detail counts `2/2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Refreshed third-goal source-audit freshness gate:
  Spark sidecar spawn for `T-3G-NEXT-EVIDENCE-GAP-AUDIT-015` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/write_final_evidence_manifest.py` now requires third-goal source audit `required_count >=86` and `source_count >=224`.
  Regression coverage mutates source audit counts to `85/223` and confirms final evidence fails on both source-audit freshness checks.
  Validation: py_compile passed; focused final-evidence tests passed (`35` tests); broader final-signoff/source-audit tests passed (`140` tests); spec-plan conformance passed `66/66`.
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, source audit required `86`, sources `224`, missing `0`, operator handoff validation `60/60`, bundle validation `74/74`, spec-plan `66/66`, runner blocker/detail counts `2/2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Promoted source-audit freshness checks into validator named-check contracts:
  Spark sidecar spawn for `T-3G-NEXT-GAP-AUDIT-016` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` now requires `third_goal_source_audit_required_count` and `third_goal_source_audit_source_count` to be present/pass in the live final evidence manifest.
  `tools/validate_final_signoff_bundle.py` now includes those checks in the final-manifest required named-check set.
  `tools/write_final_evidence_manifest.py` and `tools/validate_final_signoff_bundle.py` now require final operator handoff validation `check_count >=62`.
  Regression coverage preserves consistency count `332` while removing `third_goal_source_audit_source_count`; operator handoff and bundle validation fail on the missing named source-audit check.
  Validation: py_compile passed; focused operator/bundle/final-evidence tests passed (`70` tests); broader final-signoff/source-audit tests passed (`142` tests); spec-plan conformance passed `66/66`.
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, source audit required `86`, sources `224`, missing `0`, operator handoff validation `62/62`, bundle validation `74/74`, spec-plan `66/66`, runner blocker/detail counts `2/2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Refreshed final evidence consistency minimum threshold:
  Spark sidecar spawn for `T-3G-NEXT-GAP-AUDIT-017` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require final evidence consistency count `>=332`.
  Regression coverage rejects stale `325` consistency contracts in operator handoff and bundle validation.
  Validation: py_compile passed; focused operator/bundle tests passed (`36` tests); broader final-signoff/source-audit tests passed (`143` tests); spec-plan conformance passed `66/66`.
  Final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, operator handoff validation `62/62`, bundle validation `74/74`, spec-plan `66/66`, runner blocker/detail counts `2/2`, mirrored artifacts `92/92`, duplicates `0`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Refreshed final evidence required-count minimum threshold:
  Spark sidecar spawn for `T-3G-REQUIRED-COUNT-GAP-AUDIT-018` failed with `agent thread limit reached`; main agent completed the bounded implementation.
  `tools/validate_final_operator_handoff.py` and `tools/validate_final_signoff_bundle.py` now require final evidence required count `>=86`.
  Regression coverage rejects stale `85/85` required-count contracts in operator handoff and bundle validation.
  Validation: py_compile passed; focused operator/bundle tests passed (`38` tests).
  Expected current target after final-runner refresh is final manifest `86/86`, consistency checks `332`, failed consistency `[]`, operator handoff validation `62/62`, bundle validation `74/74`; remaining blockers stay C3b physical smoke and XR-VITs source/policy.
- Added direct live source-audit validation to final validators:
  Spark sidecar spawn for `T-3G-N30-GAP-AUDIT` failed with `agent thread limit reached`; main agent selected and completed direct source-audit validator hardening.
  `tools/validate_final_operator_handoff.py` now loads `third_goal_source_audit_2026_06_16.json` directly and requires pass status, `required_count >=86`, and `source_count >=224`.
  `tools/validate_final_signoff_bundle.py` now includes `source_audit` in bundle inputs and requires pass status, `required_count >=86`, `source_count >=224`, and empty `missing_required`.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=66` and final bundle validation `check_count >=80`.
  Regression coverage mutates live source-audit counts to `85/223` while preserving final-manifest named checks; operator handoff and bundle validation fail on direct source-audit checks.
  Validation: focused operator/bundle/final-evidence/completion tests passed (`115` tests); final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, source audit `86/224`, operator handoff validation `66/66`, bundle validation `80/80`, spec-plan `66/66`, and the same two external blockers.
- Added direct live resource-policy validation to final validators:
  Spark sidecar spawn for `T-3G-N31-RESOURCE-POLICY-DIRECT` failed with `agent thread limit reached`; main agent selected and completed direct resource-policy validator hardening.
  `tools/validate_final_operator_handoff.py` now loads `e2e_resource_policy_audit_2026_06_10.json` directly and requires pass status, fail count `0`, check count `>=36`, and core DSP/URAM/LUTRAM/C3b checks present/pass.
  `tools/validate_final_signoff_bundle.py` now includes `resource_policy` in bundle inputs and requires the same live resource-policy contract.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=72` and final bundle validation `check_count >=87`.
  Regression coverage mutates live resource-policy count to `35` and fails `c3b_csynth_lut_lte_threshold` while preserving final-manifest named checks; operator handoff and bundle validation fail on direct resource-policy checks.
  Validation: focused operator/bundle/final-evidence/completion tests passed (`119` tests); final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, resource-policy `36/36`, source audit `86/224`, operator handoff validation `72/72`, bundle validation `87/87`, spec-plan `66/66`, and the same two external blockers.
- Added direct live spec-plan validation to final validators:
  Spark sidecar spawn for `T-3G-N32-SPEC-PLAN-DIRECT` failed with `agent thread limit reached`; main agent selected and completed direct spec-plan validator hardening.
  `tools/validate_final_operator_handoff.py` now loads `spec_plan_conformance_audit_2026_06_10.json` directly and requires pass status, check count `>=66`, current observed counts, no side effects, and core plan/spec/current-doc checks present/pass.
  `tools/validate_final_signoff_bundle.py` now includes `spec_plan` in bundle inputs and requires the same live spec-plan contract.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=79` and final bundle validation `check_count >=95`.
  Regression coverage mutates live spec-plan count to `65` and fails core spec-plan checks while preserving final-manifest named checks; operator handoff and bundle validation fail on direct spec-plan checks.
  Validation: focused operator/bundle/final-evidence/completion tests passed (`123` tests); broader final-signoff/source-audit regression tests passed (`155` tests); final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, resource-policy `36/36`, spec-plan `66/66`, source audit `86/224`, operator handoff validation `79/79`, bundle validation `95/95`, and the same two external blockers.
- Added validator-count current-doc freshness to spec-plan conformance:
  Spark sidecar spawn for `T-3G-N33-INTERNAL-GAP-AUDIT` failed with `agent thread limit reached`; main agent selected and completed validator-count current-doc freshness hardening.
  `tools/write_spec_plan_conformance_audit.py` now reads live final operator handoff validation and final signoff bundle validation JSON and requires current docs to record operator handoff validation `79/79` plus bundle validation `95/95`.
  `tools/write_final_evidence_manifest.py`, `tools/validate_final_operator_handoff.py`, and `tools/validate_final_signoff_bundle.py` now require spec-plan conformance `check_count >=84`.
  Validation: py_compile passed; focused spec-plan/operator/bundle/final-evidence tests passed (`92` tests); broader final-signoff/source-audit regression tests passed (`156` tests); final runner remains expected-blocked with final manifest `86/86`, consistency checks `332`, failed consistency `[]`, resource-policy `36/36`, spec-plan `83/83`, source audit `86/224`, operator handoff validation `79/79`, bundle validation `95/95`, and the same two external blockers.
- Added direct live final-runner validation to final validators:
  Spark sidecar spawn for `T-3G-N34-FINAL-RUNNER-DIRECT` failed with `agent thread limit reached`; main agent selected and completed direct final-runner validator hardening.
  `tools/validate_final_operator_handoff.py` now loads `third_goal_final_signoff_run_2026_06_10.json` directly and requires blocked status, expected two blockers, two blocker details, and nonduplicated mirrored-artifact counts.
  `tools/validate_final_signoff_bundle.py` now includes `final_runner` in bundle inputs and requires the same live final-runner status/count/mirror contract.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=85` and final bundle validation `check_count >=102`.
  Regression coverage mutates live final-runner blocker/detail/mirror counts while preserving final-manifest runner checks; operator handoff and bundle validation fail on direct final-runner checks.
  Expected current target after final-runner refresh is final manifest `86/86`, consistency checks `332`, failed consistency `[]`, resource-policy `36/36`, spec-plan `83/83`, source audit `86/224`, operator handoff validation `85/85`, bundle validation `102/102`, and the same two external blockers.
- Added direct live Req6 parameterization validation to final validators:
  Spark sidecar spawn for `T-3G-N35-REQ6-DIRECT` failed with `agent thread limit reached`; main agent selected and completed direct Req6 validator hardening.
  `tools/validate_final_operator_handoff.py` now loads `req6_parameterization_audit_2026_06_16.json` directly and requires pass status, fail count `0`, check count `>=71`, expected knob defaults, and core PAR16/PAR32 checks present/pass.
  `tools/validate_final_signoff_bundle.py` now includes `req6_parameterization` in bundle inputs and requires the same live Req6 status/count/knob/core-check contract.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=92` and final bundle validation `check_count >=110`.
  Regression coverage mutates live Req6 count to `70` and fails `parallelism_extension_par32_requires_fresh_reports` while preserving final-manifest Req6 checks; operator handoff and bundle validation fail on direct Req6 checks.
  Expected current target after final-runner refresh is final manifest `86/86`, consistency checks `332`, failed consistency `[]`, resource-policy `36/36`, spec-plan `83/83`, source audit `86/224`, operator handoff validation `92/92`, bundle validation `110/110`, and the same two external blockers.
- Added direct live current-audit validation to final validators:
  Spark sidecar spawn for `T-3G-N36-CURRENT-AUDIT-DIRECT` failed with `agent thread limit reached`; main agent selected and completed direct current-audit validator hardening.
  `tools/validate_final_operator_handoff.py` now loads `third_goal_current_audit_2026_06_16.json` directly and requires blocked-external status, reflected/partial/blocked summary counts, external blocker names, blocker paths, XR policy integrity, QKV optional state, and no-side-effect safety.
  `tools/validate_final_signoff_bundle.py` now includes `current_audit` in bundle inputs and requires the same live current-audit contract.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=116` and final bundle validation `check_count >=137`.
  Regression coverage mutates live current-audit summary, XR policy consistency, and QKV required state while preserving final-manifest checks; operator handoff and bundle validation fail on direct current-audit checks.
  Expected current target after final-runner refresh is final manifest `86/86`, consistency checks `332`, failed consistency `[]`, resource-policy `36/36`, spec-plan `83/83`, source audit `86/224`, current audit reflected `10`, partial `1`, blocked `1`, operator handoff validation `116/116`, bundle validation `137/137`, and the same two external blockers.

- Added current-doc XR policy-count freshness gate:
  Spark sidecar spawn for `T-3G-N37-DOC-POLICY-COUNT-FRESHNESS` failed with `agent thread limit reached`; main agent selected and completed the current-doc stale-count hardening.
  `tools/write_spec_plan_conformance_audit.py` now rejects current docs that retain stale XR policy check count `7` or older validator-count anchors after live current audit advanced to policy count `9`, operator handoff validation `116/116`, and bundle validation `137/137`.
  Updated current HANDOVER/PROGRESS/CHOICE/Sub-Plan/Validation/log text to the live anchors.
  Validation: focused spec-plan/operator/bundle/final-evidence/completion tests passed (`147` tests); broader final-signoff/source-audit/resource-policy regression tests passed (`171` tests); final runner remains expected-blocked with spec-plan `86/86`, final manifest `86/86`, consistency checks `332`, failed consistency `0`, operator handoff validation `116/116`, bundle validation `137/137`, and the same two external blockers.

- Added direct live closeout-validation validation to final validators:
  Spark sidecar spawn for `T-3G-N38-CLOSEOUT-VALIDATION-DIRECT` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/validate_final_operator_handoff.py` now directly validates `final_unblock_closeout_packet_validation_2026_06_10.json` status, fail count, check count, core closeout/board/XR/QKV checks, and no-side-effect safety.
  `tools/validate_final_signoff_bundle.py` now includes `closeout_validation` in the bundle source set and applies the same live contract.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=116` and final bundle validation `check_count >=137`.
  Validation: focused validator/final-evidence/spec-plan/completion tests passed (`151` tests); broader final-signoff regression tests passed (`175` tests); final runner remains expected-blocked with spec-plan `86/86`, final manifest `86/86`, consistency checks `332`, failed consistency `0`, closeout validation `49/49`, operator handoff validation `116/116`, bundle validation `137/137`; blockers remain C3b physical smoke and XR-VITs source/policy.

- Added direct live closeout-packet validation to final validators:
  Spark sidecar spawn for `T-3G-N39-CLOSEOUT-PACKET-DIRECT` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/validate_final_operator_handoff.py` now directly validates `final_unblock_closeout_packet_2026_06_10.json` ready status, two expected blockers, board package readiness, C3b smoke contract, XR-VITs policy integrity, optional QKV URAM commands, and no-side-effect safety.
  `tools/validate_final_signoff_bundle.py` now includes `closeout_packet` in the bundle source set and applies the same live contract.
  `tools/write_spec_plan_conformance_audit.py` now rejects stale current-doc anchors for operator handoff validation `108/108`, bundle validation `128/128`, and XR policy check count `8`.
  Validation: focused validator/final-evidence/spec-plan/completion tests passed (`155` tests); broader final-signoff regression tests passed (`179` tests); final runner remains expected-blocked with spec-plan `86/86`, final manifest `86/86`, consistency checks `332`, failed consistency `0`, closeout validation `49/49`, operator handoff validation `116/116`, bundle validation `137/137`; blockers remain C3b physical smoke and XR-VITs source/policy.

- Added direct live C3b/XR blocker-gate validation to final validators:
  Spark sidecar spawn for `T-3G-N40-BLOCKER-GATE-DIRECT` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/validate_final_operator_handoff.py` now directly validates `c3b_physical_smoke_gate_audit_2026_06_16.json` and `xr_vits_gate_audit_2026_06_16.json` status, blocker path, candidate, and no-side-effect safety.
  `tools/validate_final_signoff_bundle.py` now includes `c3b_physical_gate` and `xr_vits_gate` in the bundle source set and applies the same live contracts.
  `tools/write_final_evidence_manifest.py` now requires operator handoff validation `check_count >=131` and final bundle validation `check_count >=152`.
  Validation after final-runner refresh is final manifest `86/86`, consistency checks `332`, failed consistency `[]`, spec-plan `86/86`, operator handoff validation `131/131`, bundle validation `152/152`, and the same two external blockers.
  Regression validation passed: py_compile, focused final-validator tests (`159` tests), broader final-signoff regression tests (`183` tests), and `git diff --check`.

- Added final blocker operator unblock plan gate:
  Spark sidecar spawn for `T-3G-N41-UNBLOCK-READINESS-SNAPSHOT` failed with `agent thread limit reached`; main agent completed local fallback.
  `tools/check_final_blocker_closure_readiness.py` now emits `operator_unblock_plan` for C3b canonical board JSON and XR-VITs exact-source/policy unblock inputs.
  The plan includes dry-run templates for C3b import and XR replacement-policy preview, active final-runner state, required-input acceptance criteria, and no-side-effect safety.
  `tools/write_final_evidence_manifest.py` now gates the operator plan through final-blocker closure consistency checks; validator minimum consistency threshold advanced to `332`.
  Validation after final-runner refresh is final manifest `86/86`, consistency checks `332`, failed consistency `[]`, spec-plan `86/86`, operator handoff validation `131/131`, bundle validation `152/152`, and the same two external blockers.
  Regression validation passed: py_compile, focused N41 tests (`128` tests), broader final-signoff regression (`193` tests), and `git diff --check`.

- Added final unblock intake/operator plan cross-check:
  Spark sidecar `T-3G-N42-INTAKE-PLAN-CROSSCHECK` completed read-only review and recommended binding final unblock intake `next_inputs` and `operator_sequence` to final blocker closure `operator_unblock_plan`.
  `tools/write_final_evidence_manifest.py` now adds five intake/operator-plan consistency checks, and final operator handoff plus final signoff bundle validators require those checks by name.
  Current target is final manifest `86/86`, consistency checks `347`, failed consistency `[]`, source audit required `86`, sources `224`, missing `0`, operator handoff validation `131/131`, bundle validation `152/152`, and the same two external blockers.

- Added final-runner mirror-integrity gate:
  Spark sidecar `019ecfbc-76cc-7622-b6fa-e0d062fa2bb9` completed a read-only gap audit and recommended hardening final signoff evidence-path integrity between `hardware/generated/signoff` and canonical `docs/resources`.
  `tools/run_third_goal_final_signoff.py` now records generated/source and docs/mirror SHA256/size contracts, with self-referential summary JSON/MD explicitly excluded.
  `tools/write_final_evidence_manifest.py`, `tools/validate_final_operator_handoff.py`, and `tools/validate_final_signoff_bundle.py` now require mirror-integrity fields to be present and passing.
  Current target remains final manifest `86/86`, consistency checks `347`, operator handoff validation `131/131`, bundle validation `152/152`, and the same two external blockers.

- Added HANDOVER raw validator summary freshness gate:
  Main agent selected a local current-doc freshness gap while Spark sidecar audited resource-policy singularity.
  `tools/write_spec_plan_conformance_audit.py` now checks raw HANDOVER artifact summaries for final operator handoff validation checks `131` and bundle validation checks `152`.
  `tools/write_final_evidence_manifest.py` includes those checks in `spec_plan_current_doc_freshness_contract`.
  Spec-plan conformance target advanced to `86/86`; external blockers unchanged.
- Added resource-policy artifact singularity gate:
  GPT5.3-Codex-Spark sidecar hit usage limit; GPT5.5 sidecar completed read-only inventory and found stale generated `e2e_resource_policy_audit_2026_06_15.{json,md}` with a 22-check payload and noncanonical filename semantics.
  Removed stale generated `2026_06_15` resource-policy pair.
  `tools/write_final_evidence_manifest.py` now gates canonical pair presence, docs/resources singularity, generated/signoff singularity, filename/payload date-tag agreement, generated/docs SHA256 equality, and combined resource-policy artifact-set status.
  Final operator handoff and final signoff bundle validators require those six manifest checks by name; final evidence consistency target is now `347`.

- Continued no-board P0 / PAR32 Vivado work:
  `par32_dsp_mixed_stream_mem16` remains the best csynth-fit candidate with latency `26,139,644` cycles, DSP `1,148`, LUT `193,546`, BRAM_18K `332`, and URAM `64`.
  HLS IP package completed and produced `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`.
  First route launch exposed a missing executable bit on `scripts/run/run_e2e_axis_dma_vivado_no_board.sh`; `chmod +x` fixed it.
  Vivado route completed for `par32_dsp_mixed_stream_mem16`: WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen completed, and routed `.bit/.hwh` artifacts were generated under `generated/build/vivado/overlay/...` and copied to `pynq/hgtxr/`.
  Added `analysis/no-board-results/VIVADO_PROGRESS_2026_06_27.md` and updated the no-board experiment plan/progress docs.
  Current Search/Track 4 ms/1 ms proof is not available because `hls/src/hgtxr_e2e_axis_top.cpp` executes a single full E2E path and has no mode-specific HLS counters.

- Added no-board Search/Track mode-profile HLS evidence:
  Spark subagents completed read-only audits for mode-top structure and script/Tcl integration points.
  Added `hls/src/hgtxr_mode_profile_top.cpp`, `hls/tb/tb_hgtxr_mode_profile_top.cpp`, mode-profile CSim/csynth/package Tcl scripts, and `scripts/run/run_mode_profile_no_board.sh`.
  Search profile `hgtxr_search_profile_top` passed CSim, csynth, and IP package: `772,268` cycles, `3.861 ms @ 5 ns`, LUT `65,557`, DSP `294`, BRAM_18K `8`, URAM `96`, target `4.000 ms` met.
  Track profile `hgtxr_track_profile_top` passed CSim, csynth, and IP package: `99,449` cycles, `0.497 ms @ 5 ns`, LUT `69,728`, DSP `302`, BRAM_18K `51`, URAM `64`, target `1.000 ms` met.
  Refreshed collector output under `analysis/no-board-results/p0_mode_profile_2026_06_27/` and added `analysis/no-board-results/MODE_PROFILE_RESULTS_2026_06_27.md`.
  This HLS/IP step left mode-profile Vivado route and Search `96/96` URAM margin as the next checks.

- Completed no-board Search/Track mode-profile Vivado route:
  Added `vivado/scripts/build_mode_profile_bitstream.tcl` and `scripts/run/run_mode_profile_vivado_no_board.sh` for pure m_axi + AXI-Lite mode-profile overlays.
  Search profile routed on ZCU104 with WNS `2.638 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`, route errors `0`, fully routed nets `39,455/39,455`, bitgen pass, post-route `18,825` LUT, `15,171` FF, `294` DSP, `2` RAMB18, `96` URAM, and routed power estimate `3.917 W`.
  Track profile routed on ZCU104 with WNS `1.481 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`, route errors `0`, fully routed nets `47,137/47,137`, bitgen pass, post-route `22,245` LUT, `20,826` FF, `302` DSP, `42` RAMB18, `64` URAM, and routed power estimate `3.778 W`.
  Routed artifacts were copied to `generated/build/vivado/overlay/hgtxr_mode_{search,track}_par32_overlay/` and `pynq/hgtxr/`.
  Added and verified `collect_no_board_reports.py --enforce-p0`; it passed on the current report set and gates Search/Track mode target and ZCU104 HLS capacity violations without failing historical exploratory PAR32 resource-fail rows.
  Remaining P0 resource risk: Search fits and routes, but consumes `96/96` URAM and therefore has no URAM headroom.

- Consolidated current work/progress/experiment conversation and created latest handover:
  Added `docs/track/WORK_PROGRESS_EXPERIMENT_SUMMARY_2026_06_27.md` to summarize decisions, experiment results, evidence boundaries, next plans, commands, and remaining risks.
  Added `docs/track/HANDOVER_2026_06_27.md` as the latest continuation handover.
  Updated `docs/track/HANDOVER.md` and `docs/track/PROGRESS.md` to point to the new documents and to clarify that Search/Track `4 ms` / `1 ms` evidence is mode-profile proxy evidence, not full E2E Search 8-block / Track 4-block deployed runtime evidence.

- Added runtime-mode E2E shared-top Search/Track HLS evidence:
  Updated `hls/src/hgtxr_e2e_axis_top.cpp`, `hls/include/hgtxr_e2e_vit.hpp`, the E2E AXIS testbench, and the no-board csim/csynth runner scripts so one E2E AXIS top selects Search/Track workload shape at runtime while using the same controller-level fast ATTN/MLP path.
  CSim passed for Search and Track runtime states with correct 6-word state output and TLAST.
  ZCU104-part csynth for `par32_runtime_mode_mem16` passed the HLS latency targets: Track-bound min `76,150` cycles / `0.381 ms`; Search-bound max `594,840` cycles / `2.974 ms`; clock target `5.00 ns`, estimated `3.691 ns`.
  Resource estimate is BRAM_18K `68`, DSP `400`, FF `39,696`, LUT `88,246`, URAM `0`.
  HLS IP package completed for `generated/hgtxr_e2e_axis_par32_runtime_mode_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`.
  Added `par32_runtime_mode_mem16` to `scripts/run/run_e2e_axis_dma_vivado_no_board.sh`, guarded the unused fastpath `weights` memory master with `HGTXR_E2E_OMIT_WEIGHT_AXI=1`, then rebuilt the matching AXIS/DMA overlay.
  Packaged IP, generated RTL, overlay `.hwh`, and copied PYNQ `.hwh` no longer contain `m_axi_gmem_e2e_weights`; only `m_axi_gmem_e2e_runtime` remains as the top-level memory master.
  Vivado route and bitgen completed for `hgtxr_e2e_axis_dma_par32_runtime_mode_mem16_overlay`: WNS `3.038 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`, fully routed nets `50,076/50,076`, route errors `0`, bitgen pass.
  Routed artifacts were copied to `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_mode_mem16_overlay/` and `pynq/hgtxr/`.
  Routed placed resources are CLB LUT `22,412`, CLB registers `21,105`, block RAM tile `37`, DSP `401`, URAM `0`; routed power estimate is total `3.736 W`, dynamic `3.042 W`, static `0.694 W`.
  Updated PYNQ runtime support so `runtime-mode-par32` can run Search with a 128x128 frame and control `16384`, and Track with a 64x64 frame and control `1`, avoiding a DMA beat-count mismatch against the runtime-mode top.
  Generated and validated Search/Track board-smoke bundles and ZCU104 session runbooks under `generated/pynq/e2e_axis_dma_par32_runtime_mode_{search,track}_smoke_*`.
  Added runtime-mode validation presets that fail board JSON when Search `accelerator_latency_ms_summary.max` or any Search sample exceeds `4.000 ms`, or when Track max/sample exceeds `1.000 ms`.
  Added runtime-mode Search/Track profiles to the SSH/SCP remote smoke runner and generated dry-run plans under `generated/signoff/zcu104_runtime_mode_par32_{search,track}_smoke_remote_run_2026_06_10.{json,md}`.
  Added `tools/check_runtime_mode_board_latency_gate.py` so the completion gate requires both canonical board result JSON files and validates them with the Search/Track latency-target presets; current generated gate status is `missing`.
  Added `tools/run_runtime_mode_board_latency.py` to run both runtime-mode Search/Track remote profiles, import both board JSON files when `--execute` is used, and evaluate the combined latency gate; current generated run summary is `dry-run` with gate `missing`.
  Added host preflight to `tools/run_runtime_mode_board_latency.py` and retried `--execute` against `xilinx@zcu104.local`; the run now stops with `status=host-unresolved` before SSH/SCP because DNS reports `[Errno -2] Name or service not known`, so no board JSON was imported.
  Checked local resolver, SSH known_hosts, and the local neighbor table for an obvious ZCU104 host/IP; no usable board target was identified, so a real ZCU104 IP/hostname is still required for physical latency signoff.
  Saved detailed evidence and residual-risk notes in `docs/track/RUNTIME_MODE_E2E_SHARED_TOP_2026_06_27.md`.
  Remaining signoff gap: board smoke/physical runtime latency is still pending; the fastpath is still a mode-profile compute path, not a full exact weight-consuming ViT datapath proof.

- Added runtime-mode HW Emulation / HLS RTL co-simulation attempt:
  Added `cosim` mode to `scripts/run/run_e2e_q4w8a_no_board.sh` and `vivado/scripts/run_e2e_q4w8a_cosim.tcl`, reusing the same runtime-mode csynth setup with `cosim_design -rtl verilog -trace_level none`.
  The runtime-mode cosim build enables `HGTXR_E2E_TEST_RUNTIME_MODES=1`, so the C testbench exercises both Search and Track in the same run.
  Command run: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_mode_mem16`.
  C testbench passed for Search and Track: Search output state words are all `15`, `search_runtime_state=0 expected=0 count=6 last=1 failures=0`; Track output state words are all `9`, `track_runtime_state=1 expected=1 count=6 last=1`; final message `E2E AXIS vector comparison passed`.
  XSIM compile/elaboration built snapshot `hgtxr_e2e_axis_top`, but HLS C/RTL co-simulation report remains `Verilog Fail` because XSIM exits on `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}` with `unexpected exception when evaluating tcl command`.
  Tried explicit Vivado/Vitis library path, `TERMINFO=/lib/terminfo`, multiple `TERM` values, disabling the generated `.wcfg`, direct `xsim -R`, and direct `xsimk`; none produced a valid HLS C/RTL PASS report on this host.

- Restored and measured runtime-mode full AXI learned-parameter path:
  Added `par32_runtime_full_axi_mem16` to the E2E no-board runner and added `runtime_full_axi_active64_b8_ff768` to the CSim/CSynth Tcl flows.
  This profile sets `HGTXR_E2E_USE_MODE_PROFILE_FASTPATH=0` and `HGTXR_E2E_OMIT_WEIGHT_AXI=0`, so the top uses `hgtxr_conv_patch_embedding`, weighted `hgtxr_e2e_controller_run`, `hgtxr_e2e_mlp_head`, and the `gmem_e2e_weights` AXI master instead of the fast embedding/head path.
  CSim passed for both Search and Track runtime states with 6-word state output and correct TLAST.
  CSynth completed on `xczu7ev-ffvc1156-2-e`: Track-bound min `627,117` cycles / `3.136 ms`, Search-bound max `5,101,975` cycles / `25.510 ms`, interval equal to latency plus one cycle, no transaction-level pipeline.
  HLS resources are BRAM_18K `332/624`, DSP `1,148/1,728`, FF `86,123/460,800`, LUT `193,627/230,400`, and URAM `64/96`.
  HLS IP package completed and `component.xml` contains `m_axi_gmem_e2e_weights` plus `m_axi_gmem_e2e_runtime`.
  Documented the fastpath/full-model split and the 8 reporting-question plan in `docs/track/RUNTIME_MODE_FULL_AXI_WEIGHT_PATH_2026_06_28.md`.
  Current judgment: the learned AXI path is restored and measurable, but latency now fails the 4 ms Search and 1 ms Track targets; the next work is weight-port/caching optimization before full AXI Vivado route/power and board p95/p99 measurement.

- Closed the full-learned ROM-only dsp3 Vivado timing experiment:
  Added `par8_runtime_rom_only_dispatch_dsp3_300_mem8` and wired `HGTXR_E2E_DSP_MUL_LATENCY` into the HLS DSP `bind_op` pragmas so the dsp3 latency macro is physically expressed.
  CSim passed Search/Track with deterministic outputs, runtime states `0/1`, six output state words, and correct TLAST.
  HLS package completed with estimated clock `2.983 ns` against the `3.333 ns` target; packaged IP contains `m_axi_gmem_e2e_runtime` and intentionally omits `m_axi_gmem_e2e_weights`.
  Vivado route and bitgen completed at `clk_pl_0 = 300.030 MHz` with WNS `0.009 ns`, TNS `0.000 ns`, WHS `0.010 ns`, and route errors `0`.
  Routed resources are CLB LUT `23,887/230,400`, register `26,595/460,800`, BRAM tile `238/312`, DSP `1,627/1,728`, and URAM `32/96`; routed vectorless power is total `5.973 W`, dynamic `5.258 W`, static `0.715 W`.
  Current judgment: the full learned ROM-only Transformer is physically implemented and timing-clean at 300 MHz, but Search latency `43.239 ms` and Track latency `5.336 ms` still miss the `4 ms` and `1 ms` goals; mode-specific power, DMA bandwidth, board rail power, and p95/p99 still require additional measurement.

- Integrated Event Conv into the full-learned ROM-only dsp3 E2E AXIS path:
  Added a distinct Event Conv on-chip ROM region after the Frame Conv region, shifted the Transformer block weight base, and guarded total required ROM words with a compile-time check against `HGTXR_E2E_WEIGHT_DEPTH`.
  Search mode now calls the Frame Conv embedding path; Track mode now calls the Event Conv embedding path using a derived event-delta buffer from the input frame.
  CSim passed after the change: Search output `[-712, -712, -712, -712, -712, -696]`, Track output `[-235, -235, -235, -235, -235, -299]`, runtime states `0/1`, TLAST correct, and `CSim done with 0 errors`.
  The first package attempt found an HLS C++ ambiguity in the event-delta reference expression; replacing the nested conditional with explicit `if/else` fixed it.
  HLS package then completed, but HLS estimated clock regressed to `3.495 ns` against the `3.333 ns` target; min/max latency became `1,605,053` and `12,989,355` cycles.
  Vivado route and bitgen still completed timing-clean at `clk_pl_0 = 300.030 MHz`: WNS `0.007 ns`, TNS `0.000 ns`, WHS `0.010 ns`, THS `0.000 ns`, route errors `0`, fully routed nets `107,641/107,641`.
  Routed resources after Event Conv are CLB LUT `24,159/230,400`, registers `26,753/460,800`, Block RAM Tile `238/312`, DSP `1,627/1,728`, and URAM `32/96`.
  Routed vectorless power after Event Conv is total `6.037 W`, dynamic `5.322 W`, static `0.715 W`, PS static `0.102 W`, PL static `0.614 W`.
  At routed clock, the current full learned latency is Track `5.350 ms`, Search `43.294 ms`, and 10%/90% hybrid `9.144 ms`, so the latency goals remain unmet even though the physical 300 MHz implementation is now proven.
  Remaining technical gaps: Search dispatcher is observable prefetch/marker logic rather than proven overlapped double-buffer prefetch; Event Conv is derived from the frame buffer rather than a separate external event AXIS stream; mode-specific power, DMA bandwidth, p95/p99, and board rail measurements are still pending.

- Connected Search dispatcher prefix prefetch to full-learned cache fills and reran Vivado:
  Replaced the dispatcher marker/preload state with two concrete `16`-word ping-pong prefetch banks and routed LayerNorm, Q/K/V, output projection, and MLP W1/W2 cache fills through `hgtxr_e2e_load_weight_word_prefetched`.
  CSim passed Search/Track: Search output `[-712, -712, -712, -712, -712, -676]`, Track output `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, six output words, TLAST correct, and `CSim done with 0 errors`.
  HLS package completed for `par8_runtime_rom_only_dispatch_dsp3_300_mem8`: target clock `3.333 ns`, estimated clock `2.924 ns`, max latency `13,008,010` cycles / `43.356 ms`.
  The current HLS min latency `26,190` cycles is not accepted as Track latency because it is a lower-bound/control envelope rather than a full Track-mode trace.
  Vivado route and bitgen completed at `clk_pl_0 = 300.030 MHz` with WNS `0.000 ns`, TNS `0.000 ns`, WHS `0.007 ns`, fully routed nets `138,804/138,804`, and route errors `0`.
  Routed resources are CLB LUT `43,929/230,400`, registers `43,016/460,800`, Block RAM Tile `238/312`, DSP `1,630/1,728`, and URAM `32/96`.
  Routed vectorless power is total `6.833 W`, dynamic `6.111 W`, static `0.721 W`; hierarchy includes `hgtxr_e2e_axis_top_0 = 3.197 W`, `psu = 2.674 W`, `axi_mem = 0.186 W`, `axi_dma_in = 0.007 W`, and `axi_dma_out = 0.026 W`.
  Current judgment: full learned Vivado implementation is still physically proven and timing-clean at 300 MHz, but Search remains `43.356 ms > 4 ms`; Track remains unresolved for the current rerun and the prior comparable full-depth envelope `5.350 ms` still misses `1 ms`.

- Replaced the dispatcher prefix prefetch with a full-block URAM prefetch bank and reran the full learned ROM-only dsp3 flow:
  `HGTXR_E2E_DISPATCH_PREFETCH_FULL_BLOCK=1` and `HGTXR_E2E_URAM_DISPATCH_PREFETCH=1` are enabled for `runtime_rom_only_dispatch_dsp3_active64_b8_ff768`.
  The dispatcher bank now uses `kBlockWeightWords`, and HLS generated `gb_dispatch_prefetch_U` as a `13,848 x 255-bit` `RAM_2P_URAM_1R1W` using `16` URAM.
  CSim passed Search/Track with deterministic outputs, runtime states `0/1`, six output state words, TLAST correct, and `CSim done with 0 errors`.
  HLS package completed with target clock `3.333 ns`, estimated clock `2.924 ns`, top max latency `13,042,545` cycles / `43.471 ms`, and HLS OOC resources BRAM_18K `386/624`, DSP `1,753/1,728`, FF `227,004/460,800`, LUT `308,055/230,400`, URAM `48/96`.
  The first Vivado implementation attempt completed bitgen but missed setup by `-0.107 ns`; rerunning the same profile with `impl_strategy=Performance_Explore` closed timing.
  Vivado route and bitgen completed with route errors `0`, fully routed nets `141,580/141,580`, and timing met at `clk_pl_0 = 300.030 MHz`: WNS `0.010 ns`, TNS `0.000 ns`, WHS `0.009 ns`, THS `0.000 ns`.
  Placed resources are CLB LUT `44,894/230,400`, registers `44,799/460,800`, BRAM tile `238/312`, DSP `1,630/1,728`, and URAM `48/96`.
  Routed vectorless power is total `7.109 W`, dynamic `6.383 W`, static `0.725 W`; hierarchy includes `hgtxr_e2e_axis_top_0 = 3.467 W`, `psu = 2.673 W`, `axi_mem = 0.188 W`, `axi_dma_in = 0.007 W`, and `axi_dma_out = 0.027 W`.
  Baseline policy clarified: use `par32_dsp_mixed_stream_mem16` as the physical/timing baseline, `par32_runtime_full_axi_mem16` as the full learned AXI functional baseline, and this `par8_runtime_rom_only_dispatch_dsp3_300_mem8` build as the current objective baseline for on-chip ROM + scheduler + full-block dispatcher.
  Current judgment: the full-block dispatcher is physically implemented, resource-fit, and timing-clean at 300 MHz with small margin. Search remains `43.471 ms > 4 ms`; Track remains unresolved for this exact rerun and the prior comparable full-depth envelope `5.350 ms` still misses `1 ms`.

- Isolated current full-block Search/Track latency with force-mode profiles:
  Added `HGTXR_E2E_FORCE_RUNTIME_MODE` handling and profile-specific CSim/csynth entries for `par8_runtime_rom_only_dispatch_dsp3_300_mem8_search_only` and `par8_runtime_rom_only_dispatch_dsp3_300_mem8_track_only`.
  Search-only CSim passed with output `[-712, -712, -712, -712, -712, -676]`, `runtime_state=0`, TLAST correct, and `CSim done with 0 errors`.
  Track-only CSim passed with output `[-235, -235, -235, -235, -235, -226]`, `runtime_state=1`, TLAST correct, and `CSim done with 0 errors`.
  Search-only CSim prefetch trace showed `prefetch_trace_done block=2 seq=1`, `block=3 seq=2`, first load hits at `seq=3/4`, and `prefetch_trace_summary block_pairs=4 violations=0`; this proves prefetch-before-use but not overlapped dataflow.
  Search-only csynth completed with target `3.33 ns`, estimated `2.924 ns`, max latency `13,001,422` cycles / `43.334 ms`, interval max `13,001,423`, and resources BRAM_18K `386`, DSP `1,497`, FF `161,370`, LUT `236,034`, URAM `48`.
  Track-only csynth completed with target `3.33 ns`, estimated `3.473 ns`, max latency `1,606,660` cycles / `5.580 ms` in the HLS report, interval max `1,606,661`, and resources BRAM_18K `378`, DSP `1,108`, FF `157,213`, LUT `197,528`, URAM `32`.
  At the routed `300.030 MHz` clock, Track converts to `5.355 ms`; the 10% Search / 90% Track hybrid expectation is `2,746,136.2` cycles / `9.153 ms`.
  Updated `docs/Validation.md`, `docs/track/PROGRESS.md`, and `docs/track/EIGHT_QUESTION_FULL_LEARNED_VIVADO_EXPERIMENT_2026_06_28.md` with the force-mode results.

- Rechecked whether the earlier best PAR32 profiles should be the active baseline:
  Confirmed the intended split: `par32_dsp_mixed_stream_mem16` remains the physical/timing baseline and `par32_runtime_full_axi_mem16` remains the full learned AXI functional baseline.
  Added PAR32 objective-path profiles for ROM-only weights/LUTs plus runtime scheduler plus dispatcher prefetch: `par32_runtime_rom_only_dispatch_dsp3_300_mem16`, `par32_runtime_rom_only_dispatch_dsp3_300_mem16_search_only`, and `par32_runtime_rom_only_dispatch_dsp3_300_mem16_track_only`.
  Search-only PAR32 objective csynth completed at `4,764,228` cycles / `15.879 ms`, estimated clock `2.777 ns`, with resources BRAM_18K `418`, DSP `1,763`, FF `180,201`, LUT `302,026`, URAM `48`.
  Track-only PAR32 objective csynth completed at `604,652` cycles / `2.100 ms`, estimated clock `3.473 ns`, with resources BRAM_18K `410`, DSP `1,374`, FF `176,105`, LUT `263,377`, URAM `32`.
  Combined PAR32 objective package completed, but Vivado implementation failed before placement because DRC `UTLZ-1` reported DSP over-utilization: `1729` DSP cells required, `1728` available on ZCU104.
  Current judgment: use the two earlier PAR32 profiles as comparison baselines, but not as the active objective baseline. The active objective path still needs DSP/LUT reduction and latency reduction before it can replace the PAR8 routed ROM-only baseline.

- Reran the PAR32 objective path with a core-fabric DSP-fit profile:
  Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_300_mem16`, which keeps the ROM-only learned weights/LUTs, runtime scheduler, full-block dispatcher prefetch, raw head multiply, and prefetch address switching, then enables `HGTXR_E2E_CORE_LAST_LANE_FABRIC_MUL=1`.
  CSim passed Search/Track with runtime states `0/1` and vector comparison pass.
  CSynth passed with target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, and resources BRAM_18K `418`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `48`.
  HLS package passed and patched `43` exported Verilog datapath modules for datapath preservation.
  Vivado route and bitgen completed; route errors were `0`, bit/hwh were generated, and the design exactly fit ZCU104 DSP at DSP48E2 `1,728/1,728`.
  Post-route setup timing failed at `clk_pl_0 = 300.030 MHz`: WNS `-0.631 ns`, TNS `-3338.375 ns`, failing setup endpoints `15,703/303,013`; hold timing was clean.
  Placed resources are CLB LUT `89,102/230,400`, registers `60,400/460,800`, Block RAM Tile `232/312`, DSP `1,728/1,728`, URAM `48/96`; routed vectorless power is total `8.593 W`, dynamic `7.856 W`, static `0.737 W`.
  Current judgment: the PAR32 objective path is no longer blocked by DSP overuse and is valid routed/bitgen evidence, but it is still not the active signoff baseline because 300 MHz timing and Search/Track latency targets remain unmet.

- Clarified the PAR32 baseline taxonomy and ran a norm-URAM timing probe:
  The earlier `par32_dsp_mixed_stream_mem16` profile is explicitly kept as the physical/timing baseline, and `par32_runtime_full_axi_mem16` is explicitly kept as the full learned AXI functional baseline. They were not discarded; they are not the objective-signoff baseline only because the current claim requires ROM-only learned weights/LUTs plus runtime scheduler plus dispatcher prefetch in one design.
  Added resource policy `dsp_mixed_normstream` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_300_mem16`.
  CSim passed with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and `CSim done with 0 errors`.
  CSynth passed at target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, and resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`; compared with the prior core-fabric probe, latency is unchanged while BRAM drops `418 -> 386` and URAM rises `48 -> 64`.
  Vivado route and bitgen completed with route errors `0`, `.bit/.hwh` copied to `hardware/pynq/hgtxr/`, and DSP exactly fit at `1,728/1,728`.
  Post-route timing improved but still fails 300 MHz: WNS `-0.496 ns`, TNS `-2860.064 ns`, WHS `0.000 ns`, THS `0.000 ns`, failing setup endpoints `15,033/303,394`.
  Placed resources are CLB LUT `89,329/230,400`, registers `60,699/460,800`, Block RAM Tile `216/312`, DSP `1,728/1,728`, URAM `64/96`; routed vectorless power is total `8.568 W`, dynamic `7.829 W`, static `0.738 W`.
  Current judgment: `normuram` is a useful PAR32 objective timing probe and should be included in comparison tables, but it is still not the signoff baseline. Next timing work is DSP MREG/PREG or explicit pipeline insertion plus selective relaxation of datapath-wide `DONT_TOUCH`.

- Ran a PAR32 norm-stage LayerNorm write timing probe:
  Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normstage_300_mem16`, which stages LayerNorm writes through a local LUTRAM row buffer before committing to `gb.norm`.
  CSim passed Search/Track with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and vector comparison pass.
  CSynth passed with target `3.333 ns`, estimate `2.777 ns`, max top latency `4,908,208` cycles / `16.359 ms`, controller max `4,829,897` cycles / `16.098 ms`, and resources BRAM_18K `386`, DSP `1,976`, FF `249,856`, LUT `399,142`, URAM `64`.
  Vivado route and bitgen completed with route errors `0`, fully routed nets `201,593/201,593`, and DSP exactly fit at `1,728/1,728`.
  Post-route timing failed at 300 MHz: WNS `-0.644 ns`, TNS `-3987.069 ns`, WHS `0.002 ns`, THS `0.000 ns`, failing setup endpoints `16,238/303,146`.
  Placed resources are CLB LUT `89,035/230,400`, registers `59,380/460,800`, Block RAM Tile `216/312`, DSP `1,728/1,728`, URAM `64/96`; routed vectorless power is total `8.615 W`, dynamic `7.876 W`, static `0.739 W`.
  Current judgment: norm-stage did not improve over norm-URAM, so the immediate next closure path is explicit DSP MREG/PREG pipeline staging, `gb_dispatch_prefetch` URAM read staging, and a selective review of broad datapath `DONT_TOUCH`.

- Ran a PAR32 norm-URAM no-DONT-TOUCH sanity probe:
  Added `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_nodt_300_mem16`, which keeps the same HLS objective flags as `normuram` but sets `E2E_OOC_DONT_TOUCH=0`.
  CSim passed with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, and vector comparison pass.
  CSynth/package passed with target `3.333 ns`, estimate `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, and resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`.
  Vivado route and bitgen completed with clean timing: WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, THS `0.000 ns`, route errors `0`.
  The result is invalid for accelerator signoff because placed resources collapsed to CLB LUT `7,817/230,400`, registers `11,522/460,800`, Block RAM Tile `5/312`, DSP `0/1,728`, URAM `0/96`; vectorless power also collapsed to total `3.624 W`, dynamic `2.931 W`, static `0.693 W`.
  Current judgment: global `DONT_TOUCH` removal lets Vivado optimize/prune away the learned compute datapath. The earlier `par32_dsp_mixed_stream_mem16` and `par32_runtime_full_axi_mem16` must remain comparison baselines, while valid objective closure must continue from preserved PAR32 profiles with selective preservation relaxation and explicit pipeline insertion.

- Ran a PAR32 norm-URAM top-DONT-TOUCH sanity probe:
  Added `HGTXR_E2E_OOC_DONT_TOUCH=top` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_topdt_300_mem16`.
  CSim passed with the same deterministic objective-path outputs as `normuram`, and HLS package patched exactly one exported Verilog module.
  CSynth/package still showed the objective-scale HLS design: BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`, max latency `4,805,312` cycles / `16.016 ms`.
  Vivado route and bitgen passed timing at `clk_pl_0 = 300.030 MHz`: WNS `0.516 ns`, TNS `0.000 ns`, WHS `0.011 ns`, route errors `0`.
  The result is also invalid for accelerator signoff because placed resources collapsed to the same pruned shell as `nodt`: CLB LUT `7,817/230,400`, registers `11,522/460,800`, Block RAM Tile `5/312`, DSP `0/1,728`, URAM `0/96`; vectorless power was total `3.624 W`, dynamic `2.931 W`, static `0.693 W`.
  Current judgment: top-only preservation is not enough to protect the learned arithmetic/cache datapath. Valid PAR32 objective closure must keep selective internal datapath preservation and add explicit DSP/URAM/cache-read pipeline stages rather than relying on top-only or global DONT_TOUCH removal.

- Ran PAR32 pipe-prefetch and fabric-lane closure probes:
  Confirmed again that `par32_dsp_mixed_stream_mem16` remains the physical/timing baseline and `par32_runtime_full_axi_mem16` remains the full learned AXI functional baseline; neither is discarded, but neither satisfies the ROM-only scheduler/prefetch objective envelope by itself.
  `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_pipepref_300_mem16` passed CSim and package, with Search `[-1169, -1169, -1169, -1169, -1169, -1133]`, Track `[-235, -235, -235, -235, -235, -291]`, runtime states `0/1`, max latency `4,817,564` cycles / `16.057 ms`, and HLS resources BRAM_18K `386`, DSP `1,976`, FF `277,802`, LUT `453,254`, URAM `64`; Vivado DRC failed with required DSP `1,740/1,728`.
  `pipepref_tail2` passed package with HLS DSP `1,966`, LUT `455,718`; `pipepref_tail4` passed package with HLS DSP `1,946`, LUT `460,646`, but Vivado still failed at required DSP `1,740/1,728`; `pipepref_tail8` passed package with HLS DSP `1,906`, LUT `470,502`, but Vivado again materialized DSP48E2 `1,740` and failed DRC.
  `pipepref_coreallfabric` passed package with HLS DSP `1,666` but LUT `529,792/230,400 = 229.9%`; `pipepref_coreallfabric_nowide` passed package with target `3.333 ns`, estimate `2.979 ns`, max latency `4,817,820` cycles / `16.058 ms`, DSP `1,666`, and LUT `647,424/230,400 = 281%`.
  Added optional `HGTXR_E2E_CORE_LANE_CT_SWITCH` and profile `pipepref_cttail4` to test whether a template/switch lane split changes binding. Package passed, but the HLS result matched `tail4`: max latency `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,946`, FF `278,756`, LUT `460,646`, URAM `64`; Vivado was not run because it did not improve over the already failed `tail4`.
  Current judgment: runtime tail-lane fabric selection and the first template/switch lane split do not reduce final physical DSP enough, while all-core fabric mapping is impossible on LUT. The next valid closure path is an explicit separate operator/data-path split or explicit pipelined DSP binding, followed by preserved ROM-only objective HLS/Vivado reruns.

- Ran PAR32 compute-selective DONT_TOUCH probe:
  Added `HGTXR_E2E_OOC_DONT_TOUCH=compute` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_300_mem16`.
  Package preserved exactly 9 key compute modules instead of the full datapath module set.
  HLS package passed with target `3.333 ns`, estimate `2.777 ns`, max latency `4,805,312` cycles / `16.016 ms`, and resources BRAM_18K `386`, DSP `1,976`, FF `250,856`, LUT `399,002`, URAM `64`.
  Vivado route and bitgen completed with route errors `0` and fully routed nets `199,142/199,142`; bitgen completed at `2026-06-29 00:27:09 KST`.
  Routed timing still failed: WNS `-0.530 ns`, TNS `-3005.251 ns`, setup failing endpoints `15,103/302,281`; hold was clean.
  Placed resources were CLB LUT `86,983/230,400`, registers `59,792/460,800`, Block RAM Tile `216/312`, DSP `1,728/1,728`, URAM `64/96`.
  Routed vectorless power was total `8.537 W`, dynamic `7.799 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.633 W`.
  Current judgment: the earlier best profiles should remain baselines, but with distinct roles. `par32_dsp_mixed_stream_mem16` is the physical/timing baseline; `par32_runtime_full_axi_mem16` is the learned AXI functional baseline; the preserved ROM-only dispatcher profiles are objective-path candidates and still need timing/latency closure.

- Applied the user-approved 300 MHz conditional timing rule and ran a PostRoutePhys follow-up:
  The experiment rule is now fixed as PL clock `300 MHz` with acceptable routed WNS down to `-0.500 ns`; this is not official Vivado timing signoff, which still requires non-negative WNS.
  Added HLS/Vivado runner profiles `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_300_mem16` and `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_pipepref_tail4_nowide_300_mem16`.
  `compute_pipepref_300` passed package with max latency `4,817,564` cycles / `16.057 ms`, BRAM_18K `386`, DSP `1,976`, FF `277,802`, LUT `453,254`, URAM `64`, but Vivado failed before placement at DSP `1,740/1,728`.
  `compute_pipepref_tail4_nowide_300` passed package with max latency `4,817,820` cycles / `16.058 ms`, BRAM_18K `386`, DSP `2,226`, FF `289,648`, LUT `481,104`, URAM `64`; it was rejected without Vivado because DSP pressure worsened.
  Reran the existing valid `compute_300` IP through `Performance_ExplorePostRoutePhysOpt` as project `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_postroutephys_300_mem16_overlay`.
  Route and bitgen completed successfully: route errors `0`, fully routed nets `199,370/199,370`, bitgen completed at `2026-06-29 02:15:36 KST`.
  Final postroute physopt timing was WNS `-0.402 ns`, TNS `-985.598 ns`, setup failing endpoints `7,376/302,473`, WHS `0.005 ns`, THS `0.000 ns`.
  Current judgment: this passes the user-approved 300 MHz experiment threshold (`-0.402 >= -0.500`) but remains below official clean timing signoff. Placed resources were CLB LUT `87,181/230,400`, registers `59,652/460,800`, Block RAM Tile `216/312`, DSP `1,728/1,728`, URAM `64/96`; vectorless power was total `8.539 W`, dynamic `7.801 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`.
  Next official-clean timing target remains explicit cache/URAM read output staging plus DSP MREG/PREG-equivalent pipeline insertion for the reported `wo_weight_cache`, `gb_dispatch_prefetch`, and MLP DSP paths.

- Prepared current PAR32 ROM compute 300 MHz board SW/HW equality bundles:
  Added PYNQ runner variant `par32-rom-compute-300`, plus transfer/validation support for `par32-rom-compute-300-search` and `par32-rom-compute-300-track`.
  The Search contract selects mode profile `search`, expects runtime state `0`, and checks raw output `[-1169, -1169, -1169, -1169, -1169, -1133]`.
  The Track contract selects mode profile `track`, expects runtime state `1`, and checks raw output `[-235, -235, -235, -235, -235, -226]`.
  Generated Search bundle `hardware/generated/pynq/e2e_axis_dma_par32_rom_compute_300_search_smoke_bundle.tar.gz` with sha256 `d21042211c8aec38252618c339818a4a5e86786283813026ecf9218086544fa0`; generated Track bundle `hardware/generated/pynq/e2e_axis_dma_par32_rom_compute_300_track_smoke_bundle.tar.gz` with sha256 `15e570fa6950cc0e501d54af7d2c9e5920073df10126fc649597f9efb03cf47b`.
  Both generated bundles passed `tools/validate_pynq_bundle_package.py` with `errors=[]`.
  Focused regressions passed: `py_compile` on the changed PYNQ/tool/test files, `E2EAxisDmaSmokeCliTests` (`6` tests), and bundle/smoke validator tests (`30` tests).
  Remaining gap: no physical ZCU104 result JSON has been captured for these bundles yet, so board p95/p99 latency, measured DMA bandwidth, and board power are still pending.

- Added board runner/gate support for the PAR32 ROM compute 300 MHz objective profile set:
  Kept the experiment clock rule as PL `300 MHz` with user-approved WNS tolerance down to `-0.500 ns`; this remains separate from official Vivado timing signoff, which requires non-negative WNS.
  Added a separate `par32-rom-compute-300` profile set to `run_runtime_mode_board_latency.py` and `check_runtime_mode_board_latency_gate.py`, with Search preset `axis-par32-rom-compute-300-search` and Track preset `axis-par32-rom-compute-300-track`.
  Added remote runner configs in `run_zcu104_c3b_smoke_remote.py` and default import destinations in `import_pynq_smoke_result.py` for the two canonical board result JSON files.
  Dry-run board plan generation passed and wrote `hardware/generated/signoff/par32_rom_compute_300_board_latency_run_2026_06_29.json`; standalone gate generation wrote `hardware/generated/signoff/par32_rom_compute_300_board_latency_gate_2026_06_29.json`.
  Current gate status is `missing`: `hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json` and `hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json` are not present yet.
  Verification passed: `py_compile` on the modified board tooling/tests and `python3 -m unittest hardware.tests.test_run_runtime_mode_board_latency hardware.tests.test_check_runtime_mode_board_latency_gate hardware.tests.test_run_zcu104_c3b_smoke_remote hardware.tests.test_import_pynq_smoke_result` (`31` tests).

- Captured current PAR32 ROM compute 300 MHz force-mode HLS evidence under the user-approved timing rule:
  The experimental rule is PL `300 MHz` with routed WNS accepted down to `-0.500 ns`; official Vivado timing signoff still requires non-negative WNS.
  Added current `compute_300` Search-only and Track-only force-mode profiles to `hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
  Search-only CSim passed with output `[-1169, -1169, -1169, -1169, -1169, -1133]`, `runtime_state=0`, vector comparison pass, and dispatcher prefetch trace violations `0`.
  Track-only CSim passed with output `[-235, -235, -235, -235, -235, -226]`, `runtime_state=1`, and vector comparison pass. This corrected the stale `par32-rom-compute-300-track` PYNQ expected raw value from `-291` to `-226`.
  Search-only CSynth passed at target `3.333 ns` with estimated clock `2.777 ns`, max latency `4,764,201` cycles / `15.879 ms`, interval max `4,764,202`, and HLS resources BRAM_18K `386`, DSP `1,720`, FF `185,414`, LUT `322,452`, URAM `64`.
  Track-only CSynth passed at target `3.333 ns` with estimated clock `3.473 ns`, max latency `604,652` cycles / `2.100 ms`, interval max `604,653`, and HLS resources BRAM_18K `378`, DSP `1,332`, FF `181,667`, LUT `288,332`, URAM `48`.
  Hybrid 10% Search / 90% Track weighted max-latency estimate is `3.478 ms`, but mode targets still fail: Search target `4 ms` vs `15.879 ms`, Track target `1 ms` vs `2.100 ms`.
  Regenerated PYNQ equality bundles with validation status `pass`: Search tar sha256 `d21042211c8aec38252618c339818a4a5e86786283813026ecf9218086544fa0`, Track tar sha256 `15e570fa6950cc0e501d54af7d2c9e5920073df10126fc649597f9efb03cf47b`.
  Remaining gap: no physical ZCU104 Search/Track result JSON has been captured, so measured p95/p99 latency, measured DMA bandwidth, and measured board power are still pending.

- Ran a structured-ROM 300 MHz latency DSE probe under the same user-approved clock rule:
  Added `HGTXR_E2E_STRUCTURED_PARAM_ROM` and `HGTXR_E2E_STRUCTURED_FAST_MATH` support, plus combined/Search-only/Track-only structured-ROM profiles in the no-board runner.
  The selected Transformer parameter contract uses identity-like LayerNorm/diagonal V/O and MLP projections with zero Q/K, while still touching the nonlinear on-chip ROM path through softmax-exp and GeLU logic.
  Search-only CSim passed from terminal output with `[1, 1, 1, 1, 1, -10]`, runtime state `0`, vector comparison pass, and prefetch trace violations `0`; Track-only CSim passed with `[0, 0, 0, 0, 0, 8]`, runtime state `1`, and vector comparison pass.
  Search-only CSynth passed at target `3.333 ns` with estimate `2.777 ns`, latency `24,886` cycles / `82.945 us`, and resources BRAM_18K `20`, DSP `2`, FF `8,372`, LUT `38,521`, URAM `0`.
  Track-only CSynth passed at target `3.333 ns` with estimate `2.777 ns`, max latency `24,731` cycles / `82.428 us`, and resources BRAM_18K `20`, DSP `2`, FF `8,367`, LUT `38,522`, URAM `0`.
  Hybrid 10% Search / 90% Track selected-parameter max-latency average is `82.480 us`; worst-case is Search `82.945 us`.
  Current judgment: the structured-ROM probe proves a selected arbitrary parameter setting can meet the Search/Track latency targets at HLS level, but it is not dense arbitrary-weight full learned Transformer signoff because the resulting datapath is heavily optimized/collapsed and has not been implemented in Vivado or run on the board.

- Ran a dense Search dispatcher prefetch-all4 probe:
  Added `HGTXR_E2E_DISPATCH_PREFETCH_BANKS`, `HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE`, and `HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START`.
  Added no-board profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only`, using four dispatcher prefetch banks and force Search mode on the dense ROM-only objective path.
  First CSim attempt intentionally failed strict trace because the check incorrectly required prefetch readiness for block `0/1`, which are Track-ROM shared-path blocks and not Search dispatcher blocks. The check was corrected to require readiness only for blocks where `block_pairs > kTrackRomBlockPairs && block_idx >= kTrackRomBlockPairs`.
  Corrected Search-only CSim passed. Trace evidence: `prefetch_trace_done block=2 seq=1 words=6924`, `prefetch_trace_done block=3 seq=2 words=6924`, interblock immediate checks all `1` for `0->1`, `1->2`, and `2->3`, and summary `violations=0 not_ready=0 immediate_gaps=0`.
  CSim output was `[-1169, -1169, -1169, -1169, -1169, -1125]`, runtime state `0`, vector comparison pass, and `CSim done with 0 errors`.
  CSynth passed at target `3.333 ns`, estimated clock `2.777 ns`, Search max latency `4,757,273` cycles / `15.856 ms`, interval max `4,757,274`.
  CSynth resources were BRAM_18K `386/624`, DSP `1,592/1,728`, FF `174,661/460,800`, LUT `320,413/230,400`, URAM `76/96`.
  Storage proof: `gb_dispatch_prefetch_U` is `ram_2p` with `impl=uram`, shape `255 x 27696 x 1`, and estimated URAM `28`.
  Current judgment: this improves evidence for the requested Search dispatcher prefetch and immediate next-block start contract, but it does not solve final resource/latency closure. The HLS LUT estimate is still above ZCU104 and Search latency remains `15.856 ms`.

- Implemented the dense Search dispatcher prefetch-all4 profile through Vivado at the user-approved 300 MHz experiment threshold:
  Added the matching Vivado runner profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_search_only` with PL clock `300 MHz` and `Performance_ExplorePostRoutePhysOpt`.
  Packaging passed and produced the exported IP for the profile; the implementation command completed route, post-route physopt, and bitgen.
  Final post-route physopt timing is WNS `-0.142 ns`, TNS `-69.079 ns`, WHS `0.006 ns`, THS `0.000 ns`; Vivado reports timing constraints are not met, but the run passes the user-approved experiment rule because WNS is above `-0.500 ns`.
  Route status is clean with routing errors `0`; route_design reported failed nets `0`, unrouted nets `0`, partially routed nets `0`, and node overlaps `0`.
  Physical utilization now replaces the pessimistic HLS LUT estimate for fit judgment: CLB LUTs `53,926/230,400 = 23.41%`, CLB registers `46,812/460,800 = 10.16%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,592/1,728 = 92.13%`.
  Routed vector-less power estimate is total on-chip `7.189 W`, dynamic `6.460 W`, device static `0.729 W`, with component dynamic clocks `0.405 W`, CLB logic `0.518 W`, signals `1.029 W`, Block RAM `0.259 W`, URAM `0.213 W`, DSP `1.365 W`, PS8 `2.671 W`.
  Bitgen completed and the overlay directory contains `.bit` and `.hwh` artifacts. Remaining gaps: official clean timing still needs WNS `>=0`, Search latency is still `15.856 ms`, and board p95/p99 latency, measured DMA bandwidth, and measured power are still pending.

- Implemented the dense combined runtime Search/Track prefetch-all4 profile through HLS and Vivado at the user-approved 300 MHz experiment threshold:
  Added no-board and Vivado runner profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`, keeping the dense ROM-only objective path, runtime Search/Track mode branch, four dispatcher prefetch banks, prefetch-all-before-compute, and strict immediate-start CSim tracing.
  Combined CSim passed both modes. Search output was `[-1169, -1169, -1169, -1169, -1169, -1125]` with runtime state `0`; Track output was `[-235, -235, -235, -235, -235, -239]` with runtime state `1`; vector comparison passed. Search trace prefetched blocks `2` and `3` before compute and reported `immediate=1` for `0->1`, `1->2`, and `2->3`; Search summary was `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`, Track summary was `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  Combined CSynth passed at target `3.333 ns` with estimated clock `2.777 ns`; top latency min/max was `26,088/4,798,390` cycles, `86.951 us/15.993 ms`, with interval max `4,798,391`. HLS resource estimate remained pessimistic: BRAM_18K `386`, DSP `1,848`, FF `242,912`, LUT `400,236`, URAM `76`.
  Vivado route, post-route physopt, reports, and bitgen completed for `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_overlay`.
  Final post-route physopt timing is `clk_pl_0 = 300.030 MHz`, WNS `-0.236 ns`, TNS `-718.875 ns`, setup failing endpoints `6,814/300,626`, WHS `0.001 ns`, THS `0.000 ns`. This passes the user-approved experiment rule because WNS is above `-0.500 ns`, but it is not official clean timing signoff.
  Route status is clean with routing errors `0`, and bitgen completed successfully. Overlay `.bit` and `.hwh` were generated under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16_overlay/`.
  Physical utilization is CLB LUTs `73,133/230,400 = 31.74%`, CLB registers `60,949/460,800 = 13.23%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,723/1,728 = 99.71%`.
  Routed vector-less power estimate is total on-chip `8.364 W`, dynamic `7.626 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`; component dynamic includes clocks `0.566 W`, CLB logic `0.883 W`, signals `1.606 W`, Block RAM `0.265 W`, URAM `0.222 W`, DSPs `1.414 W`, and PS8 `2.671 W`.
  Remaining gaps: official timing clean still needs WNS `>=0`; combined HLS max latency `15.993 ms` still misses the Search `4 ms` target; board Search/Track p95/p99 latency, measured DMA bandwidth, and measured board power remain pending.

- Prepared ZCU104 board SW/HW equality bundles and gate profile for the dense combined runtime Search/Track prefetch-all4 artifact:
  Added PYNQ runner variant `par32-prefetchall4-300`, validator presets `axis-par32-prefetchall4-300-search` and `axis-par32-prefetchall4-300-track`, bundle-packager support, remote-runner profiles, import destinations, and board latency gate profile set `par32-prefetchall4-300`.
  Search contract uses mode profile `search`, runtime state `0`, and expected raw output `[-1169, -1169, -1169, -1169, -1169, -1125]`; Track contract uses mode profile `track`, runtime state `1`, and expected raw output `[-235, -235, -235, -235, -235, -239]`.
  Generated Search bundle `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_search_smoke_bundle.tar.gz` with sha256 `34f76304a0ca41bed3b60bfad610898bb41bb7cfc6e698c7b71f072f4de98491`; generated Track bundle `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_track_smoke_bundle.tar.gz` with sha256 `9b24856588b0ae691104aa79476bc1da80b8a3884a8d568d7e0660279a08d4e0`.
  Both generated bundles passed `tools/validate_pynq_bundle_package.py` with `errors=[]`.
  Generated dry-run board plan `hardware/generated/signoff/par32_prefetchall4_300_board_latency_run_2026_06_29.json` and `.md`; status is `dry-run`, gate is `missing`.
  Generated standalone gate artifact `hardware/generated/signoff/par32_prefetchall4_300_board_latency_gate_2026_06_29.json` and `.md`; status is `missing` because canonical board result files are absent.
  Verification passed: `py_compile` on modified PYNQ/tool files and `python3 -m unittest hardware.tests.test_validate_pynq_smoke_result hardware.tests.test_validate_pynq_bundle_package hardware.tests.test_package_e2e_axis_dma_pynq_bundle hardware.tests.test_run_runtime_mode_board_latency hardware.tests.test_check_runtime_mode_board_latency_gate hardware.tests.test_run_zcu104_c3b_smoke_remote hardware.tests.test_import_pynq_smoke_result` (`63` tests).
  Remaining gap: execute the prepared Search/Track bundles on ZCU104 and import results into `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json` and `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json`. Until then, physical SW/HW equality, measured p95/p99 latency, measured DMA bandwidth, and measured board power remain pending.

- Attempted HLS RTL cosim for the dense combined runtime Search/Track prefetch-all4 artifact under the same user-approved timing rule:
  The rule remains PL `300 MHz`, experiment WNS floor `-0.500 ns`, official clean timing `>= 0.000 ns`.
  Command was `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`.
  The flow reran CSynth at target `3.333 ns` and reported estimated Fmax `360.10 MHz`.
  C testbench/post-check evidence was clean: Search output `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track output `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, `E2E AXIS vector comparison passed`, Search scheduler summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`, and Track summary `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  Verilog cosim failed before usable RTL latency/comparison results: `sim/report/hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` and latency `NA`; `sim/verilog/xsim.log` reports `ERROR: unexpected exception when evaluating tcl command` while running `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}`.
  Added `hardware/tools/diagnose_hls_cosim_xsim.py` to make this distinction reproducible. The generated diagnosis artifacts are `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`, with status `blocked-xsim-launch`.
  The diagnosis checks are: C testbench output match `true`, scheduler trace clean `true`, Search expected output `true`, Track expected output `true`, RTL cosim pass `false`, XSIM launch exception `true`, and `-autoloadwcfg` in failure `true`.
  Current judgment: C-level output/scheduler evidence is strengthened, but RTL functional signoff remains open until the XSIM launch exception is isolated and Verilog cosim completes.

- Reconfirmed the active experiment timing rule after the user set the working point:
  Use PL clock `300 MHz` for the active dense combined runtime candidate, and allow routed WNS down to `-0.500 ns` only for experiment continuation.
  The current active combined `prefetchall4_300` runtime Top remains experiment-accepted at WNS `-0.236 ns`, route errors `0`, hold clean, and bitgen pass.
  Official timing-clean signoff remains separate and still requires routed WNS `>= 0.000 ns`.

- Updated the dense combined runtime cosim diagnosis under the same 300 MHz rule:
  `hardware/vivado/scripts/run_e2e_q4w8a_cosim.tcl` now applies `config_cosim -disable_deadlock_detection` before Verilog cosim.
  Vitis HLS recorded that option in `sim/report/cosim_options.xml`, and XELAB built the snapshot `hgtxr_e2e_axis_top`.
  The C testbench/Search-Track scheduler evidence remains clean: Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, vector comparison pass, and scheduler summaries with `violations=0 not_ready=0 immediate_gaps=0`.
  Verilog cosim remains blocked because XSIM still launches with generated `-autoloadwcfg` and exits with `unexpected exception when evaluating tcl command`; RTL latency/comparison output is still unavailable.
  Manual no-autoload probes did not clear the block: hiding `.wcfg` kept the same launch exception, and direct `xsimk` elaborated then exited without RTL TV output.
  Regenerated diagnosis artifacts at `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`; status is `blocked-xsim-launch`.

- Added host XSIM snapshot smoke to separate design RTL issues from tool/runtime launch issues:
  Tool: `hardware/tools/check_xsim_snapshot_smoke.py`.
  The smoke writes a trivial Verilog testbench, runs `xvlog`, `xelab tb -s tb_snapshot`, then `xsim tb_snapshot -R` using the same `/tools/Xilinx/Vivado/2023.2` install.
  Result: `xvlog` passes and `xelab` builds `tb_snapshot`, but XSIM exits during `xsim {tb_snapshot} -autoloadwcfg -runall` with the same `unexpected exception when evaluating tcl command`; the pass token `XSIM_SMOKE_PASS` is not observed.
  Generated artifacts: `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.json` and `.md`; status is `blocked-xsim-runtime`.
  Updated `hardware/tools/diagnose_hls_cosim_xsim.py` to import the smoke result and classify HGTXR cosim as a host XSIM runtime blocker rather than a proven HGTXR RTL mismatch. The diagnosis also treats explicit `-view *.wcfg` and `-autoloadwcfg` as the same waveform snapshot-launch failure class.

- Extended the XSIM blocker diagnosis with alternate snapshot-launch probes:
  Updated `hardware/tools/check_xsim_snapshot_smoke.py` to run `xsim tb_snapshot --runall`, `xsim tb_snapshot --tclbatch run_all_no_wave.tcl`, and direct `xsim.dir/tb_snapshot/xsimk` after the standard `xsim tb_snapshot -R` attempt.
  Regenerated `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.json` and `.md`.
  Standard `-R`, alternate `--runall`, and alternate `--tclbatch` all still source `xsim_script.tcl`, inject `-autoloadwcfg`, and fail with `unexpected exception when evaluating tcl command`.
  Direct `xsimk` starts the simulator kernel and reports `elaboration-done`, but it does not run the testbench and does not emit `XSIM_SMOKE_PASS`.
  Regenerated `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`; status remains `blocked-xsim-launch`.
  Updated conclusion: C testbench output/scheduler evidence remains clean, XELAB snapshot build is confirmed, and the blocker is narrowed to host XSIM wrapper/Tcl snapshot-launch rather than a proven HGTXR RTL mismatch.

- Refreshed the active 300 MHz goal status and board gate contract after the user-approved WNS rule:
  Rule is PL `300 MHz` with experiment continuation allowed down to routed WNS `-0.500 ns`; official Vivado timing signoff still requires WNS `>= 0.000 ns`.
  Added `hardware/tools/write_prefetchall4_300_goal_status.py`.
  Generated `hardware/generated/signoff/prefetchall4_300_goal_status_2026_06_29.json` and `.md`; status is `experiment-pass-missing-board-rtlcosim`.
  Current active routed evidence remains `clk_pl_0 = 300.030 MHz`, WNS `-0.236 ns`, WHS `0.001 ns`, route errors `0`, and bitgen pass; ZCU104 top placed utilization fits, with DSP usage high at `1723/1728`.
  Vivado vectorless power evidence remains estimate-only: total `8.364 W`, dynamic `7.626 W`, static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`.
  Tightened PYNQ validator presets for `axis-par32-rom-compute-300-*` and `axis-par32-prefetchall4-300-*` so canonical board JSONs must meet Search `<=4.0 ms` and Track `<=1.0 ms` accelerator latency targets.
  Focused validation passed: `python3 -m py_compile hardware/tools/write_prefetchall4_300_goal_status.py hardware/tools/validate_pynq_smoke_result.py hardware/tools/check_runtime_mode_board_latency_gate.py`; `python3 -m unittest hardware.tests.test_validate_pynq_smoke_result hardware.tests.test_check_runtime_mode_board_latency_gate`; `python3 hardware/tools/write_prefetchall4_300_goal_status.py`.

- Extended the 300 MHz goal-status report with board measurement aggregation for the eight-question report:
  The report now reads canonical Search and Track board JSON paths under `hardware/pynq/hgtxr/` when present.
  It extracts per-mode accelerator latency, board latency, input DMA bandwidth, aggregate DMA bandwidth, runtime state, and repeat count.
  It computes synthetic Search `10%` / Track `90%` hybrid accelerator latency, board latency, throughput, worst-case latency, and DMA means from per-mode board samples and summaries.
  The report clearly marks the hybrid values as `synthetic-from-per-mode-samples`; a physically interleaved 10/90 board run is still required before calling them measured hybrid distribution.
  Focused validation passed: `python3 -m unittest hardware.tests.test_write_prefetchall4_300_goal_status hardware.tests.test_validate_pynq_smoke_result hardware.tests.test_check_runtime_mode_board_latency_gate`; `python3 hardware/tools/write_prefetchall4_300_goal_status.py`; `git diff --check` over the changed docs/tools/tests.

- Added the active 300 MHz structural objective-contract audit:
  Tool: `hardware/tools/write_prefetchall4_300_contract_audit.py`.
  Generated artifacts: `hardware/generated/signoff/prefetchall4_300_contract_audit_2026_06_29.json` and `.md`.
  Result: status `pass`, `16/16` checks.
  The audit checks runtime Search/Track scheduler symbols, Frame/Event Conv RTL modules, on-chip ROM parameter classifier, Head ROM path, Track Transformer ROM path, omitted external weight AXI in csynth/top RTL, GeLU/softmax-exp/layernorm-rsqrt ROM/LUT paths, dispatcher URAM prefetch, prefetch-all4 ordering, strict immediate-start CSim gate, full Transformer RTL modules, CSim expected Search/Track outputs, and 300 MHz csynth evidence.
  Integrated this audit into `hardware/tools/write_prefetchall4_300_goal_status.py`, so the current goal-status report now records `contract_audit.status=pass` and `16/16` checks alongside the user-approved PL `300 MHz` / routed WNS `>= -0.500 ns` experiment rule.

- Added a measured interleaved Search 10% / Track 90% board-run path for the active 300 MHz prefetch-all4 artifact:
  New PYNQ runner: `hardware/pynq/hgtxr/run_e2e_axis_dma_hybrid_smoke.py`.
  The runner executes one Search invocation followed by nine Track invocations per cycle on the same loaded overlay, checks Search runtime state `0` and output `[-1169, -1169, -1169, -1169, -1169, -1125]`, checks Track runtime state `1` and output `[-235, -235, -235, -235, -235, -239]`, and emits per-mode plus aggregate latency, throughput, and DMA summaries.
  Added validator preset `axis-par32-prefetchall4-300-hybrid-10-90`; it enforces the 1:9 invocation distribution, Search latency `<=4.0 ms`, Track latency `<=1.0 ms`, pass status, expected outputs, expected runtime states, and active bit/hwh artifact prefix.
  Generated and validated bundle `hardware/generated/pynq/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_smoke_bundle.tar.gz` with sha256 `a902fb80e4e453d625e6f245fdc6824da82fda9abca1452fb3ef83724714024e`; bundle validation status is `pass`.
  Generated dry-run remote execution plan `hardware/generated/signoff/zcu104_par32_prefetchall4_300_hybrid_10_90_smoke_remote_run_2026_06_10.json` and `.md`; status is `dry-run`, execute is `false`, and errors are `0`.
  Updated `hardware/tools/write_prefetchall4_300_goal_status.py` to read canonical hybrid board JSON `hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json` and prefer measured interleaved hybrid data over synthetic per-mode aggregation when present.
  Focused validation passed: `py_compile` for changed Python files, `python3 -m unittest` across the PYNQ validator/package/import/remote-runner/goal-status tests (`65` tests), JSON validation for generated signoff and bundle artifacts, and bundle-package validation.
  Remaining gap: the canonical hybrid board JSON is not present yet, so Search/Track/Hybrid physical board p95/p99 latency, DMA bandwidth, and measured board power are still pending.

- Added Q8 resource-breakdown evidence for the active 300 MHz prefetch-all4 artifact:
  Tool: `hardware/tools/write_prefetchall4_300_resource_breakdown.py`.
  Vivado helper: `hardware/vivado/scripts/report_e2e_axis_dma_impl_utilization.tcl`.
  Generated artifacts: `hardware/generated/signoff/prefetchall4_300_resource_breakdown_2026_06_29.json` and `.md`.
  Also generated implemented Vivado reports from the existing post-route physopt checkpoint under the active `impl_1` directory: `*_utilization_implemented.rpt`, `*_utilization_hierarchical_implemented.rpt`, `*_timing_summary_implemented.rpt`, `*_route_status_implemented.rpt`, and `*_power_implemented.rpt`.
  The first direct Vivado invocation failed because the wrapper could not load `libtinfo.so.5`; rerunning with `LD_LIBRARY_PATH=/tools/Xilinx/Vivado/2023.2/lib/lnx64.o/Rhel/9:/tools/Xilinx/Vivado/2023.2/lib/lnx64.o` succeeded.
  Status is `pass`; missing primary top resources `[]`; missing block reports `[]`; hierarchical rows parsed `3,166`.
  Top placed utilization remains CLB LUTs `73,133/230,400 = 31.74%`, CLB registers `60,949/460,800 = 13.23%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,723/1,728 = 99.71%`.
  Major block implemented hierarchical breakdown now covers `pl_hgtxr_e2e_axis_top`, `axi_dma_in`, `axi_dma_out`, `axi_mem_interconnect`, `axi_control_interconnect`, and `psu`.
  The PL accelerator hierarchy row shows LUT `67,148`, FF `53,157`, Block RAM Tile `211`, URAM `76`, DSP `1,723`; this captures that the DSP/URAM pressure is in the full learned accelerator IP rather than DMA/AXI/PS blocks.
  Integrated the breakdown into `hardware/tools/write_prefetchall4_300_goal_status.py`, so Q8 now reports top placed absolute/percent utilization plus implemented hierarchical major-block breakdown as available evidence with Q8 status `available`.
  Remaining gap: physical ZCU104 latency/DMA/power measurements and RTL cosim remain open; resource attribution itself now has implemented hierarchical evidence.

- Advanced the XSIM/RTL cosim blocker diagnosis with a direct simulator-kernel probe:
  Updated `hardware/tools/check_xsim_snapshot_smoke.py` so direct `xsimk` is driven through MI commands after `elaboration-done`.
  Regenerated `hardware/generated/signoff/xsim_snapshot_smoke_2026_06_29.json` and `.md`.
  Result: status `wrapper-blocked-kernel-pass`.
  The normal `xsim` wrapper/Tcl launch still fails with injected `-autoloadwcfg` and `unexpected exception when evaluating tcl command`, but direct `xsimk` completes `-exec-run`, completes `-exec-continue`, emits `XSIM_SMOKE_PASS`, and exits cleanly.
  Added `hardware/tools/check_hls_xsimk_direct_probe.py` for the active HGTXR HLS RTL snapshot.
  Ran `python3 hardware/tools/check_hls_xsimk_direct_probe.py --timeout-s 120`.
  Generated `hardware/generated/signoff/prefetchall4_300_hls_xsimk_direct_probe_2026_06_29.json` and `.md`.
  Result: status `direct-kernel-timeout`; HGTXR snapshot entered the kernel, completed `-exec-run`, started `-exec-continue`, printed the HLS RTL simulation progress banner, and reached `RTL Simulation : 0 / 2 [0.00%] @ "109000"`.
  Regenerated `hardware/generated/signoff/prefetchall4_300_cosim_xsim_diagnosis_2026_06_29.json` and `.md`.
  Current judgment: the standard wrapper path is still blocked, direct kernel execution is viable, but HGTXR RTL output signoff remains open until the direct run completes both Search/Track transactions or the wrapper path is repaired.

- Added an instrumented direct XSIMK progress probe for the active 300 MHz prefetch-all4 HLS RTL snapshot:
  Tool: `hardware/tools/check_hls_xsimk_progress_probe.py`.
  The tool copies generated `sim/verilog` and sibling `sim/tv` to `/tmp/hgtxr_xsim_progress_probe_full_300`, patches only the copied HLS testbench `PROGRESS_TIMEOUT` from `10000000` to `100000`, rebuilds the snapshot with `xelab`, and drives direct `xsimk` through MI commands.
  Generated artifacts: `hardware/generated/signoff/prefetchall4_300_hls_xsimk_progress_probe_2026_06_29.json` and `.md`.
  Result: status `progress-timeout`; snapshot rebuild passed, direct `xsimk` entered the kernel, and RTL simulation advanced inside transaction 0 from `0/2 0.00% @ 109000` to `0/2 8.34% @ 1336125000` during the bounded 120 s run.
  Updated `hardware/tools/diagnose_hls_cosim_xsim.py` and `hardware/tools/write_prefetchall4_300_goal_status.py` to include direct/progress XSIMK evidence.
  Regenerated diagnosis and goal-status artifacts under `hardware/generated/signoff/`.
  Current judgment: the RTL simulator is not frozen under direct `xsimk`, but RTL output signoff remains pending because the Search/Track expected RTL outputs were not reached before timeout.

- Added a DSP/BRAM pressure-relief successor for the active 300 MHz prefetch-all4 profile:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16`.
  Source change: added `HGTXR_E2E_NONLINEAR_ROM_LUTRAM` to select LUTRAM for tiny nonlinear ROM tables while preserving BRAM as the default.
  Profile change: enabled LUTRAM nonlinear ROMs and moved two tail core lanes to fabric multiply with `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=2`, `HGTXR_E2E_CORE_LANE_CT_SWITCH=1`, and `HGTXR_E2E_FABRIC_MUL_LATENCY=3`.
  CSim passed for Search/Track expected vectors and strict prefetch immediate-start trace.
  CSynth passed: estimated clock `2.777 ns` / `360.10 MHz`, top max latency `4,810,678` cycles / `16.034 ms`.
  HLS resource delta versus active `prefetchall4_300`: BRAM_18K `386 -> 384`, DSP `1848 -> 1838`, FF `242,912 -> 243,144`, LUT `400,236 -> 402,640`, URAM unchanged `76`.
  Generated RTL includes `hgtxr_e2e_axis_top_hgtxr_e2e_mlp_unit_0_s_kGeluRom_ROM_1P_LUTRAM_1R.v`, confirming the small nonlinear ROM LUTRAM path. Large QKV/MLP caches remain BRAM/URAM because they are not the small high-bank-count tables targeted by this change.
  Vivado package/route/post-route physopt/bitgen completed. The routed design has route errors `0`, fully routed nets `176,835/176,835`, and exported `.bit/.hwh` under `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_300_mem16_overlay/`.
  Implemented utilization confirms physical pressure relief: CLB LUT `72,960/230,400 = 31.67%`, CLB registers `64,095/460,800 = 13.91%`, Block RAM Tile `215/312 = 68.91%`, URAM `76/96 = 79.17%`, DSP `1,669/1,728 = 96.59%`.
  Compared with the active `prefetchall4_300` routed baseline, the successor changes CLB LUT `73,133 -> 72,960`, CLB registers `60,949 -> 64,095`, Block RAM Tile `216 -> 215`, URAM `76 -> 76`, and DSP `1,723 -> 1,669`.
  Implemented vectorless power is total on-chip `8.194 W`, dynamic `7.457 W`, device static `0.737 W`, PL static `0.632 W`, and DSP bucket `1.279 W`.
  Final post-route physopt timing is not acceptable for promotion: `clk_pl_0 = 300.030 MHz`, WNS `-0.588 ns`, TNS `-2779.720 ns`, WHS `0.001 ns`, THS `0.000 ns`, setup-failing endpoints `13,250/293,026`. This misses the user-approved continuation floor `-0.500 ns` and is worse than active `prefetchall4_300` WNS `-0.236 ns`.
  Current judgment: the requested DSP/BRAM relief is achieved, but the successor should remain a non-promoted probe. Next timing work should target explicit DSP input/output pipeline staging in controller/MLP multiply paths and selective relaxation of broad OOC `DONT_TOUCH`, because Vivado repeatedly reported remaining MLP/ATTN optimization limits and unpipelined DSP DRC warnings.

- Added and validated a DSP-pipeline pressure-relief successor for the active 300 MHz prefetch-all4 profile:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16`.
  Source/profile intent: keep nonlinear ROM LUTRAM, keep two tail core lanes in fabric, and add `HGTXR_E2E_DSP_MUL_LATENCY=4` so Vivado can move more DSP registers during phys_opt.
  CSim passed for Search/Track expected vectors, runtime states, vector comparison, and strict prefetch trace.
  CSynth passed: target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,810,934` cycles / `16.035 ms`, resources BRAM_18K `384`, DSP `1,838`, FF `244,988`, LUT `402,794`, URAM `76`.
  Vivado route/post-route physopt/bitgen completed at `clk_pl_0 = 300.030 MHz`: final implemented WNS `-0.372 ns`, TNS `-1326.235 ns`, WHS `0.005 ns`, THS `0.000 ns`, route errors `0`, bitgen pass.
  Implemented utilization: CLB LUT `71,232/230,400 = 30.92%`, CLB registers `63,821/460,800 = 13.85%`, Block RAM Tile `201/312 = 64.42%`, RAMB36E2 `180`, RAMB18E2 `42`, URAM `76/96 = 79.17%`, DSP `1,713/1,728 = 99.13%`, LUT as Distributed RAM `1,232/101,760 = 1.21%`.
  Compared with active `prefetchall4_300`, this reduces Block RAM Tile `216 -> 201`, DSP `1,723 -> 1,713`, and CLB LUT `73,133 -> 71,232`; URAM remains `76`.
  Implemented vectorless power: total `8.396 W`, dynamic `7.658 W`, device static `0.738 W`, PS static `0.105 W`, PL static `0.634 W`, Block RAM `0.220 W`, URAM `0.222 W`, DSPs `1.442 W`.
  Current judgment: this is the best routed pressure-relief successor so far. It passes the user-approved 300 MHz experiment threshold because WNS `-0.372 ns >= -0.500 ns`, hold is clean, route errors are `0`, and bitgen completed. It is still not official clean Vivado signoff because WNS remains negative.

- Added and validated a more aggressive tail4 DSP-relief probe for the same 300 MHz prefetch-all4 pressure path:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail4_dsppipe4_300_mem16`.
  Source/profile intent: keep nonlinear ROM LUTRAM and DSP latency `4`, but move four dense core tail lanes to fabric through `HGTXR_E2E_CORE_FABRIC_TAIL_LANES=4`.
  CSim passed for Search/Track expected vectors, runtime states, vector comparison, and strict prefetch trace.
  CSynth passed: target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,810,934` cycles / `16.035 ms`, resources BRAM_18K `384`, DSP `1,818`, FF `245,528`, LUT `407,722`, URAM `76`.
  Vivado route/post-route physopt/bitgen completed at `clk_pl_0 = 300.030 MHz`: final implemented WNS `-0.573 ns`, TNS `-1629.740 ns`, WHS `0.009 ns`, THS `0.000 ns`, route errors `0`, bitgen pass.
  Implemented utilization: CLB LUT `70,766/230,400 = 30.71%`, CLB registers `63,836/460,800 = 13.85%`, Block RAM Tile `201/312 = 64.42%`, RAMB36E2 `180`, RAMB18E2 `42`, URAM `76/96 = 79.17%`, DSP `1,693/1,728 = 97.97%`, LUT as Distributed RAM `1,232/101,760 = 1.21%`.
  Compared with `lutrom_tail2_dsppipe4`, this reduces DSP `1,713 -> 1,693` and total vectorless power `8.396 W -> 8.329 W`, but timing regresses from WNS `-0.372 ns` to `-0.573 ns`.
  Current judgment: reject tail4 for promotion because it misses the user-approved WNS floor `-0.500 ns`. Keep `lutrom_tail2_dsppipe4` as the recommended pressure-relief point, and next target explicit DSP MREG/PREG-equivalent staging plus narrower compute `DONT_TOUCH`.

- Added and rejected a `compute_keep` preservation-scope probe for the 300 MHz `lutrom_tail2_dsppipe4` pressure path:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_keep_300_mem16`.
  Packaging change: `HGTXR_E2E_OOC_DONT_TOUCH=compute_keep` patches compute modules with `keep_hierarchy` only and emits OOC XDC `KEEP_HIERARCHY true` without `DONT_TOUCH true`.
  CSim passed for Search/Track expected vectors, runtime states, vector comparison, and strict prefetch trace.
  CSynth passed: target `3.333 ns`, estimated clock `2.777 ns`, max latency `4,810,934` cycles / `16.035 ms`, resources BRAM_18K `384`, DSP `1,838`, FF `244,988`, LUT `402,794`, URAM `76`.
  Vivado implementation, route, post-route physopt, reports, and bitgen completed. Timing was clean with WNS `0.064 ns`, TNS `0.000 ns`, WHS `0.009 ns`, THS `0.000 ns`, and route errors `0`.
  Rejection evidence: final implemented resources collapsed to a pruned shell, CLB LUT `7,901`, registers `11,546`, Block RAM Tile `5`, URAM `0`, DSP `1`, with vectorless total power `3.615 W`.
  Current judgment: reject `compute_keep` for signoff despite positive WNS because the full learned Transformer compute fabric is not physically preserved. Keep `lutrom_tail2_dsppipe4` as the valid pressure-relief point until a narrower preservation strategy can pass both timing and full-resource gates.

- Added and validated an aggressive DSP/LUTRAM pressure-relief probe:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  Source change: added explicit LUTRAM binding controls for `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`; the profile enables those controls and moves `16/32` dense core tail lanes to fabric. The deeper `hidden` buffer intentionally stays BRAM/URAM.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  CSim evidence: Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, prefetch trace `violations=0`, `not_ready=0`, `immediate_gaps=0`.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated `2.777 ns`, max latency `4,807,861` cycles / `16.025 ms`, BRAM_18K `336`, DSP `1,698`, FF `247,663`, LUT `461,054`, URAM `28`.
  Package command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh package par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  Vivado command passed: `sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  Vivado result: route errors `0`, fully routed nets `201,673/201,673`, bitgen completed successfully, exported overlay under `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_overlay/`.
  Final implemented utilization: CLB LUT `113,354/230,400 = 49.20%`, LUT as Distributed RAM `35,024`, registers `64,368/460,800 = 13.97%`, Block RAM Tile `184/312 = 58.97%`, URAM `28/96 = 29.17%`, DSP `1,573/1,728 = 91.03%`.
  Timing result: post-route physopt WNS `-0.500 ns`, TNS `-6623.548 ns`, WHS `0.006 ns`, THS `0.000 ns`; this exactly passes the user-approved continuation floor `WNS >= -0.500 ns` but is not official clean Vivado timing signoff.
  Vectorless power: total on-chip `8.811 W`, dynamic `8.074 W`, device static `0.736 W`; component buckets include clocks `0.695 W`, CLB logic `1.239 W`, signals `2.062 W`, Block RAM `0.211 W`, URAM `0.099 W`, DSPs `1.098 W`, PS8 `2.671 W`.
  Current judgment: this is the strongest physical DSP/BRAM/URAM relief point, but it trades those savings for high LUTRAM/fanout and large negative TNS. Keep it as a conditional resource-pressure profile; keep `lutrom_tail2_dsppipe4` as the balanced pressure-relief candidate.

- Checked a narrower DSP-pipeline timing-cleanup idea:
  Candidate intent: keep the `tail2` resource balance and raise `HGTXR_E2E_DSP_MUL_LATENCY` from `4` to `6`.
  CSim passed with the same Search/Track expected vectors and strict prefetch trace.
  CSynth rejected the candidate before scheduling: HLS reports `Latency value 6 is out of range, valid value is [0, 4]` for the bound DSP multiply.
  Decision: do not keep this as a selectable profile. Future DSP timing cleanup must add explicit register/pipeline structure or alter preservation constraints instead of relying on bind-op latency above `4`.

- Generalized direct RTL probe tooling for profile-specific evidence:
  Updated `hardware/tools/check_hls_xsimk_direct_probe.py` to accept `--profile`, `--search-tail`, and `--track-tail`. The default remains the active `prefetchall4_300` profile and keeps the existing default output paths.
  Updated `hardware/tools/check_hls_xsimk_progress_probe.py` with the same profile and expected-tail arguments, and made it copy `sim/verilog`/`sim/tv` from the requested profile instead of a hard-coded active profile.
  Non-default profiles now write profile-specific signoff artifact names under `hardware/generated/signoff/` when output paths are not explicitly supplied.
  Validation passed: `python3 -m py_compile hardware/tools/check_hls_xsimk_direct_probe.py hardware/tools/check_hls_xsimk_progress_probe.py`; `python3 hardware/tools/check_hls_xsimk_direct_probe.py --help`; `python3 hardware/tools/check_hls_xsimk_progress_probe.py --help`.
  Smoke command: `python3 hardware/tools/check_hls_xsimk_direct_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16 --timeout-s 5 --json-out /tmp/hgtxr_direct_probe_param_smoke.json --markdown-out /tmp/hgtxr_direct_probe_param_smoke.md`.
  Smoke result: status `direct-kernel-timeout`, `xsimk_exists=true`, `entered_kernel=true`, `exec_run_complete=true`, `exec_continue_started=true`, and RTL progress sample `0/2 [0.00%] @ 109000`.
  Missing-snapshot validation: `python3 hardware/tools/check_hls_xsimk_progress_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16 --timeout-s 1 --json-out /tmp/hgtxr_progress_probe_tail16_missing.json --markdown-out /tmp/hgtxr_progress_probe_tail16_missing.md` returned status `missing-sim-root` and recorded that the requested profile has no HLS `sim/verilog` snapshot yet.
  Current judgment: this does not close RTL output signoff, but it removes a tooling blocker for applying the same direct XSIMK same-output verification path to `lutrom_tail2_dsppipe4`, `tail16_lutbuf_dsppipe4`, or later pressure-relief profiles after their HLS cosim snapshots exist.

- Generated and probed the `lutrom_tail2_dsppipe4` HLS RTL snapshot:
  Command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16`.
  C TB passed before RTL launch with Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, and `E2E AXIS vector comparison passed`.
  Search dispatcher prefetch trace stayed clean: `block_pairs=4`, `violations=0`, `not_ready=0`, `immediate_gaps=0`.
  XELAB built `sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`, but standard HLS Verilog cosim failed at the same wrapper command `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}` and `hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` with latency `NA`.
  Direct smoke command: `python3 hardware/tools/check_hls_xsimk_direct_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16 --timeout-s 5`.
  Direct smoke result: status `direct-kernel-timeout`, kernel entry true, `-exec-run` complete, `-exec-continue` started, and first RTL progress `0/2 [0.00%] @ 109000`.
  Progress command: `python3 hardware/tools/check_hls_xsimk_progress_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16 --timeout-s 120`.
  Progress result: status `progress-timeout`, snapshot rebuild passed, and direct XSIMK advanced from `0/2 [0.00%] @ 109000` to `0/2 [8.31%] @ 1336125000`.
  Generated artifacts: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_direct_probe_2026_06_29.{json,md}` and `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_progress_probe_2026_06_29.{json,md}`.
  Current judgment: the balanced pressure-relief profile now has reusable RTL snapshot/probe evidence, but full RTL output equality remains open because expected Search/Track RTL output vectors were not reached before timeout.

- Closed the `lutrom_tail2_dsppipe4` direct XSIMK output-comparison loop and found a functional mismatch:
  Long direct XSIMK artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_direct_probe_1800s_2026_06_29.json`.
  Result: the direct kernel run completed both transactions in `1538.798 s`, with progress reaching `2 / 2 [100.00%] @ 18092130000`.
  Tool update: `hardware/tools/check_hls_xsimk_direct_probe.py` now compares existing HLS C/RTL TV output files, supports `--compare-tv-only`, records per-port hashes, and reports first mismatch lines.
  TV compare command: `python3 hardware/tools/check_hls_xsimk_direct_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16 --compare-tv-only --json-out hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.json --markdown-out hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.md`.
  Expected result: command exits `1` with status `tv-output-mismatch`.
  Mismatch: `axis_out_V_data_V` first payload is C `0x...fb6f` versus RTL `0x...0000`; `axis_out_V_keep_V`, `axis_out_V_strb_V`, `axis_out_V_last_V`, and `gmem_e2e_runtime` match after transaction-line whitespace normalization.
  Current judgment: DSP/BRAM/LUTRAM pressure relief is physically implemented and documented, but this RTL snapshot is not SW/HW same-output signoff. The next functional debug target is the RTL AXIS data payload path, not further resource DSE.

- Added targeted token-LUTRAM pressure-relief follow-up:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_dsppipe4_300_mem16`.
  Scope: keep the balanced `tail2_dsppipe4` compute policy, move only frame `tokens` and `gb.tokens` to LUTRAM, and avoid the broader norm/Q/K/V/attention LUTRAM move used by the aggressive `tail16_lutbuf` probe.
  AXIS data fix: `hgtxr_axis_write_state_values()` is now inlined and `hgtxr_data_to_axis()` emits raw `ap_fixed` bits under synthesis. Generated RTL now drives `axis_out_TDATA_int_regslice` from `zext_ln437*` wires derived from `out_state_*`; the output data regslice is not tied to `256'd0`.
  CSim passed with Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, and strict prefetch trace `violations=0`, `not_ready=0`, `immediate_gaps=0`.
  CSynth passed at target `3.333 ns`: estimated clock `2.777 ns`, max latency `4,809,404` cycles / `16.030 ms`, BRAM_18K `336`, DSP `1,838`, FF `243,190`, LUT `409,665`, URAM `76`.
  Resource delta versus `lutrom_tail2_dsppipe4`: BRAM_18K `384 -> 336`, DSP unchanged `1,838`, FF `244,988 -> 243,190`, LUT `402,794 -> 409,665`, URAM unchanged `76`.
  Vivado route/post-route physopt/bitgen completed. Route errors `0`, fully routed nets `183,782/183,782`, and bitgen pass. The impl directory did not contain a direct `.hwh`, so the overlay script copied the BD handoff `.hwh` through its fallback path.
  Implemented utilization: CLB LUT `82,306/230,400 = 35.72%`, LUT as Distributed RAM `8,912`, registers `63,181/460,800 = 13.71%`, Block RAM Tile `184/312 = 58.97%`, URAM `76/96 = 79.17%`, DSP `1,713/1,728 = 99.13%`.
  Implemented delta versus `lutrom_tail2_dsppipe4`: Block RAM Tile `201 -> 184`, LUT as Distributed RAM `1,232 -> 8,912`, CLB LUT `71,232 -> 82,306`, registers `63,821 -> 63,181`, DSP unchanged `1,713`, URAM unchanged `76`.
  Final post-route physopt timing: WNS `-0.560 ns`, TNS `-6463.610 ns`, WHS `0.005 ns`, THS `0.000 ns` at `clk_pl_0 = 300.030 MHz`. This misses the user-approved continuation floor `WNS >= -0.500 ns` by `0.060 ns`.
  Vectorless power: total `8.769 W`, dynamic `8.028 W`, device static `0.741 W`, PS static `0.105 W`, PL static `0.636 W`; component buckets include clocks `0.626 W`, CLB logic `0.944 W`, signals `1.917 W`, Block RAM `0.210 W`, URAM `0.217 W`, DSPs `1.443 W`, PS8 `2.671 W`.
  Current judgment: this precisely addresses small-data/many-bank BRAM waste and physically saves BRAM, but it is not promoted over `lutrom_tail2_dsppipe4` because timing regresses from WNS `-0.372 ns` to `-0.560 ns`. Full RTL TV equality for this profile is still pending.

- Generated and probed the aggressive `tail16_lutbuf_dsppipe4` HLS RTL snapshot:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  Cosim command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  C TB passed before Verilog launch with Search `[-1169, -1169, -1169, -1169, -1169, -1125]`, Track `[-235, -235, -235, -235, -235, -239]`, runtime states `0/1`, vector comparison pass, and strict prefetch summaries Search `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`, Track `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  HLS storage evidence confirms nonlinear `kGeluRom` LUTRAM ROM and targeted `tokens`/`gb.*` LUTRAM buffers for the small/high-bank-count memory move.
  XELAB built `sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`; standard HLS Verilog cosim still failed only at the XSIM wrapper command `xsim {hgtxr_e2e_axis_top} -autoloadwcfg -tclbatch {hgtxr_e2e_axis_top.tcl}` with `unexpected exception when evaluating tcl command`.
  Direct probe artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_direct_probe_2026_06_29.{json,md}`.
  Direct probe result: status `direct-kernel-timeout`, kernel entered, `-exec-run` complete, `-exec-continue` started, first RTL progress `0/2 [0.00%] @ 109000`.
  Instrumented progress artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_progress_probe_2026_06_29.{json,md}`.
  Instrumented progress result: status `progress-timeout`, 120 s run advanced from `0/2 [0.00%] @ 109000` to `0/2 [8.32%] @ 1336125000`, with estimated two-transaction runtime about `2885 s`.
  TV compare-only artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_tv_compare_2026_06_29.{json,md}`.
  Early current judgment: do not treat the initial tail16 TV compare-only status as a real data mismatch, because at that point the RTL transactions had not finished and RTL TV output files were missing/incomplete or placeholder-only.

- Completed the long direct XSIMK run for `tail16_lutbuf_dsppipe4` and confirmed a real RTL TV mismatch:
  Artifact: `hardware/generated/signoff/par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16_hls_xsimk_direct_probe_3300s_2026_06_29.{json,md}`.
  Result: status `completed-tv-output-mismatch`, elapsed `1546.076 s`, return code `1` from the probe because TV comparison failed, while the underlying `xsimk` return code was `0`.
  RTL completed both transactions: progress reached `2 / 2 [100.00%] @ 18080694000`.
  TV comparison: all checked RTL output files are present; `axis_out_V_data_V` mismatches at first payload line, C `0x...fb6f` versus RTL `0x...0003`.
  Matching ports: `axis_out_V_keep_V`, `axis_out_V_strb_V`, `axis_out_V_last_V`, and `gmem_e2e_runtime` match after normalization.
  Current judgment: tail16 remains the strongest physical DSP/BRAM/URAM pressure-relief point, but it is not SW/HW same-output signoff. The functional debug target is the RTL AXIS TDATA/state-payload path shared by pressure-relief profiles.

- Applied fixed-point CSim and divider-pressure follow-up for the aggressive `tail16_lutbuf_dsppipe4` profile:
  `hardware/hls/include/fixed_types.h` now selects `ap_fixed` when `HGTXR_HLS_FIXED_CSIM` is set.
  `hardware/hls/include/hgtxr_e2e_vit.hpp` now emits raw fixed-point AXIS payload bits on fixed/synthesis paths and uses compile-time scale helpers for repeated constant scale conversions.
  Pressure profiles in `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` now pass `HGTXR_HLS_FIXED_CSIM`; fixed pressure profiles also pass `HGTXR_E2E_AVOID_RUNTIME_DIVIDERS`.
  Static validation passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh` and `git diff --check` on the touched HLS/script files.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  CSim result is now raw fixed payload output: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and strict prefetch summaries clean for Search and Track.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated clock `2.846 ns`, controller max latency `5,178,119` cycles / `17.259 ms`, BRAM_18K `416`, DSP `2369`, FF `331,918`, LUT `640,454`, URAM `28`.
  Memory mapping result: nonlinear ROMs are distributed/LUTRAM, `tokens` and many 3072x8 `gb.*` banks are LUTRAM, large 12288-word banks remain BRAM, dispatcher prefetch remains URAM.
  Current judgment: CSim representation is now aligned with raw fixed payloads, but this fixed-Csim HLS estimate is worse than the previously routed physical tail16 candidate for resource fit. The blocker is now DSP/LUT pressure in `hgtxr_e2e_controller_run`, plus one remaining generated softmax `sdiv`. Do not launch another long XSIMK run until a lower-DSP CSynth point is found.

- Ran the core-all-fabric / no-forced-DSP / broad-LUTRAM pressure probe:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16`.
  Script update: the profile now passes `-UHGTXR_E2E_FORCE_DSP_MUL -DHGTXR_E2E_FORCE_DSP_MUL=0`, `-UHGTXR_E2E_FORCE_WIDE_DSP_MUL -DHGTXR_E2E_FORCE_WIDE_DSP_MUL=0`, `HGTXR_E2E_CORE_ALL_FABRIC_MUL=1`, and LUTRAM switches for frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`.
  Static checks passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh`; `git diff --check -- hardware/scripts/run/run_e2e_q4w8a_no_board.sh hardware/hls/include/hgtxr_e2e_vit.hpp hardware/hls/include/fixed_types.h`.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16`.
  CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and strict prefetch summaries clean.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_nowide_lutbuf_300_mem16`.
  CSynth result: target `3.333 ns`, estimated clock `2.979 ns`, top max latency `8,647,111` cycles / `28.821 ms`, controller max latency `5,173,511` cycles / `17.243 ms`.
  Resource result: BRAM_18K `416/624 = 66.7%`, DSP `2115/1728 = 122.4%`, FF `341844/460800 = 74.2%`, LUT `907718/230400 = 394.0%`, URAM `28/96 = 29.2%`.
  Memory result: broad LUTRAM movement is effective for 3072x8 `tokens`/`gb.*` banks, but large 12288-word banks remain BRAM and should not be pushed to LUTRAM given the failed LUT estimate.
  Current judgment: this profile is rejected as an implementation candidate. Disabling explicit DSP binding does not lower the HLS DSP estimate because the wide multiply structures are still inferred as DSP by the tool. The next DSE should reduce multiply width, remove/approximate the remaining softmax `sdiv_28ns_28ns_8_32_1`, or reduce parallelism/fanout rather than pushing more logic into LUT fabric.

- Added and tested accumulator-narrowed / integer-nonlinear pressure-relief profiles:
  Code update: `hardware/hls/include/fixed_types.h` now allows `hgtxr_acc_t` to be configured with `HGTXR_ACC_W` and `HGTXR_ACC_I`; the new profiles set `HGTXR_ACC_W=24` and `HGTXR_ACC_I=10`.
  Script update: `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` now exposes `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_intnl_lutbuf_acc24_dsppipe4_300_mem16` and `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_coreallfabric_intnl_lutbuf_acc24_300_mem16`.
  Both profiles keep the small/high-bank BRAM-to-LUTRAM policy for frame `tokens`, `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, and `gb.attn`, while avoiding LUTRAM binding for large dense 12288-word banks.
  CSim passed for both profiles. Search output was `[244, 244, 244, 244, 244, 242]`; Track output was `[200, 200, 200, 200, 200, 196]`; runtime states were Search `0` and Track `1`; strict prefetch traces were clean for both modes.
  CSynth passed for `tail16_intnl_lutbuf_acc24_dsppipe4`: target `3.333 ns`, estimated `2.846 ns`, top max latency `8,611,527` cycles / `28.702 ms`, BRAM_18K `416/624 = 66%`, DSP `2323/1728 = 134%`, FF `316855/460800 = 68%`, LUT `621865/230400 = 269%`, URAM `28/96 = 29%`.
  CSynth passed for `coreallfabric_intnl_lutbuf_acc24`: target `3.333 ns`, estimated `2.846 ns`, top max latency `8,586,695` cycles / `28.619 ms`, BRAM_18K `416/624 = 66%`, DSP `2067/1728 = 119%`, FF `322437/460800 = 69%`, LUT `721301/230400 = 313%`, URAM `28/96 = 29%`.
  Memory evidence confirms that nonlinear ROMs and selected small/high-bank buffers are LUTRAM/distributed RAM. Remaining BRAM/URAM pressure comes from large weight caches and dispatcher prefetch storage that should not be blindly moved into LUT fabric.
  Current judgment: the requested DSP-to-LUT and BRAM-to-LUTRAM mechanisms are implemented and measured, but neither new profile is a ZCU104-fit implementation candidate. The best new DSP number is still `2067/1728 = 119%`, and the LUT estimate is much worse at `721301/230400 = 313%`. Stop promoting these HLS points to Vivado; the next useful DSE is reducing parallelism/fanout or restructuring QKV/MLP weight-cache reuse.

- Added and tested a par16/mem8 runtime-ROM fit probe:
  Profile: `par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8`.
  Intent: keep the full learned runtime-ROM path and prefetch scheduler intact, but reduce `E2E_PAR` from `32` to `16` and reduce `MEM_BANK_PAR` from `16` to `8` to test whether simple parallelism/fanout shrink moves the design toward ZCU104 fit.
  Static validation passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh` and `git diff --check -- hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8`.
  CSim result: Search `[244, 244, 244, 244, 244, 242]`, Track `[200, 200, 200, 200, 200, 196]`, runtime states `0/1`, and `E2E AXIS vector comparison passed`.
  Search dispatcher proof: blocks 2 and 3 were prefetched before first load; summary `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  Track resident path proof: summary `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par16_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_intnl_acc24_dsppipe4_300_mem8`.
  CSynth result: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`, top max latency `13,077,605 cycles` / `43.588 ms`, controller max latency `9,616,135 cycles` / `32.051 ms`.
  Resource result: BRAM_18K `400/624 = 64%`, DSP `2289/1728 = 132%`, FF `294526/460800 = 63%`, LUT `485714/230400 = 210%`, URAM `108/96 = 112%`.
  Major pressure: `hgtxr_e2e_controller_run` still dominates with BRAM_18K `280`, DSP `2286`, FF `288292`, LUT `448016`.
  Current judgment: reject the simple par16/mem8 shrink. It passes behavior and 300MHz HLS clock estimate, but it is not a fit candidate and its controller latency is much worse than the par32 pressure-relief points. The next aligned work should focus on timing cleanup and RTL equality for the physically routed par32 candidates, or a real QKV/MLP reuse/cache restructuring rather than just lowering `E2E_PAR`.

- Added and tested selective aux-fabric / norm-LUTRAM pressure-relief controls:
  Code update: `hardware/hls/include/hgtxr_e2e_vit.hpp` now provides `HGTXR_E2E_PATCH_FABRIC_MUL`, `HGTXR_E2E_LAYERNORM_FABRIC_MUL`, `HGTXR_E2E_FASTPATH_FABRIC_MUL`, and `HGTXR_E2E_LUTRAM_HIDDEN`.
  Code update: `hardware/hls/src/hgtxr_e2e_axis_top.cpp` and `hardware/hls/src/hgtxr_e2e_m_axi_top.cpp` can bind `gb.hidden` to LUTRAM when explicitly enabled.
  Script update: `hardware/scripts/run/run_e2e_q4w8a_no_board.sh` now exposes `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
  Profile policy: patch Conv, LayerNorm, and fastpath/data-observable multiply paths are allowed to use fabric; `tokens`, `gb.tokens`, and `gb.norm` are LUTRAM; large Q/K/V/ATTN/hidden and weight-cache buffers are not moved to LUTRAM by default.
  Static validation passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh` and `python3 -m py_compile hardware/tools/check_hls_xsimk_direct_probe.py hardware/tools/check_hls_xsimk_progress_probe.py`.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
  CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and strict prefetch summaries clean.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated clock `3.167 ns` / `315.77 MHz`, top max latency `8,651,783 cycles` / `28.836 ms`.
  Resource delta versus `tail2_toklutbuf_dsppipe4`: BRAM_18K `416 -> 416`, DSP `2593 -> 2567`, FF `324481 -> 324721`, LUT `549344 -> 564547`, URAM `108 -> 92`.
  Current judgment: the requested technique works locally for specific blocks. Conv DSP drops to zero and URAM pressure drops below capacity, but LUT and DSP still fail fit badly. Keep the controls as DSE levers; do not promote this profile to Vivado implementation.

- Added and tested two follow-up DSP/BRAM pressure probes:
  `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16` disables QKV and WO/W1/W2 dense weight-vector caches while keeping token LUTRAM, nonlinear LUTRAM, runtime Search/Track scheduling, and strict dispatcher checks.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16`.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail2_toklutbuf_nocache_dsppipe4_300_mem16`.
  `nocache` result: target `3.333 ns`, estimated clock `2.846 ns` / `351.37 MHz`, top max latency `8,645,943 cycles` / `28.817 ms`, BRAM_18K `136/624 = 21%`, DSP `2209/1728 = 127%`, FF `252925/460800 = 54%`, LUT `523286/230400 = 227%`, URAM `108/96 = 112%`.
  Current judgment for `nocache`: BRAM pressure is greatly improved, but controller DSP/LUT and URAM remain over capacity. Keep as DSE evidence; do not promote to Vivado.
  `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16` enables tail8 fabric multiply, patch/LayerNorm/fastpath fabric switches, and selected token-like LUTRAM migration.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail8_lutbuf_auxfabric_normlut_dsppipe4_300_mem16`.
  `tail8_lutbuf_auxfabric_normlut` result: target `3.333 ns`, estimated clock `3.167 ns` / `315.77 MHz`, top max latency `8,650,183 cycles` / `28.831 ms`, BRAM_18K `416/624 = 66%`, DSP `2471/1728 = 142%`, FF `328436/460800 = 71%`, LUT `615001/230400 = 266%`, URAM `28/96 = 29%`.
  Current judgment for `tail8_lutbuf_auxfabric_normlut`: worse than the existing `tail8_lutbuf_dsppipe4` reference in BRAM, DSP, LUT, and latency. Reject. The next useful path is RTL AXIS `TDATA` mismatch debug plus QKV/MLP reuse/fanout restructuring rather than more broad DSP-to-LUT or LUTRAM migration.

- Added and ran an AXIS payload contract recheck for the physical `tail16_lutbuf_dsppipe4` candidate:
  Tool added: `hardware/tools/check_hls_axis_payload_contract.py`.
  The tool audits generated synthesis Verilog for constant-zero `axis_out_TDATA`, records out_state payload width, and compares existing C/RTL TV `axis_out_V_data_V` files by active low payload bits.
  Stale evidence found: the old `tail2_dsppipe4` generated snapshot still contains `hgtxr_axis_write_state_values` with `assign axis_out_TDATA = 256'd0`; diagnostic status is `fail-constant-axis-tdata`. Do not use that old TV mismatch as evidence against current source.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  Current CSim result: Search `[3, 3, 3, 3, 3, 11]`, runtime state `0`, `count=6`, `last=1`; Track `[217, 217, 217, 217, 217, 213]`, runtime state `1`, `count=6`, `last=1`; strict Search/Track prefetch summaries are clean.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  Current CSynth result: target `3.333 ns`, estimated clock `2.846 ns` / `351.37 MHz`, controller max latency `5,178,119 cycles` / `17.259 ms`.
  Current resource result: BRAM_18K `416/624 = 66%`, DSP `2369/1728 = 137%`, FF `331918/460800 = 72%`, LUT `640454/230400 = 278%`, URAM `28/96 = 29%`; `hgtxr_e2e_controller_run` dominates with DSP `2364`, LUT `550818`.
  Current structural AXIS result: `hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_axis_payload_contract_2026_06_29.{json,md}` reports `structural-pass-missing-rtl-tv`; generated `axis_out_TDATA` is driven from out_state-derived `zext_ln545*` sources, out_state width is `[8]`, and there is no constant-zero TDATA assignment.
  Remaining blocker: current regenerated snapshot has no RTL TV output yet, so this is not RTL functional signoff. Next gate is a direct `xsimk` run on the current regenerated snapshot, then active 8-bit TV comparison if RTL TV is emitted.
  Resource next gate remains architectural: remove or approximate the generated `sdiv_28ns_28ns_8_32_1` softmax divider and restructure QKV/MLP reuse/fanout; more broad DSP-to-LUT migration is not supported by the measured DSE.

- Completed the bounded direct `xsimk` follow-up for current `tail16_lutbuf_dsppipe4`:
  HLS cosim command: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16`.
  HLS cosim C TB passed with Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict prefetch summaries.
  XELAB built `sim/verilog/xsim.dir/hgtxr_e2e_axis_top/xsimk`, but the normal XSIM wrapper still failed with `unexpected exception when evaluating tcl command`.
  Direct probe command: `python3 hardware/tools/check_hls_xsimk_direct_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_dsppipe4_300_mem16 --timeout-s 3600 --json-out hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_current_hls_xsimk_direct_probe_2026_06_29.json --markdown-out hardware/generated/signoff/par32_tail16_lutbuf_dsppipe4_current_hls_xsimk_direct_probe_2026_06_29.md`.
  Direct probe result: `direct-kernel-timeout`; it entered the RTL kernel, completed `-exec-run`, started `-exec-continue`, and reached RTL progress `1 / 2` transactions at simulation time `28632442000` before the 3600s timeout.
  Active payload contract rerun result: `fail-active-payload-tv` because only the first transaction RTL TV exists. C active low 8-bit payloads were `[3, 3, 3, 3, 3, 11, -39, -39, -39, -39, -39, -43]`; RTL emitted `[3, 3, 3, 3, 3, 11]`.
  Current judgment: the Search transaction active payload matches in RTL, and the stale zero-TDATA bug is absent, but Track RTL completion remains pending. Resource pressure remains unresolved, so the next implementation direction should reduce QKV/MLP controller fanout or remove/approximate the softmax divider rather than pushing additional large memories or multipliers into LUT fabric.

- Added and tested a focused `tail16` softmaxQ divider-removal probe:
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`.
  Isolated profile delta: `HGTXR_E2E_USE_HGPIPE_INT_SOFTMAXQ=1` on top of the existing `tail16_lutbuf_dsppipe4` conditions.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`.
  CSim result: Search `[0, 0, 0, 0, 0, 238]`, Track `[237, 237, 237, 237, 237, 233]`, runtime states `0/1`, and clean strict Search/Track prefetch traces.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_softmaxq_lutbuf_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated clock `2.846 ns` / `351.37 MHz`, top max latency `8,620,231 cycles` / `28.731 ms`, controller max `5,158,919 cycles` / `17.195 ms`.
  Resource result: BRAM_18K `416/624 = 66%`, DSP `2371/1728 = 137%`, FF `328898/460800 = 71%`, LUT `640308/230400 = 277%`, URAM `28/96 = 29%`.
  Divider check: generated RTL no longer contains `sdiv_28ns_28ns_8_32_1`, but remaining narrow `udiv` names are index/constant-division artifacts.
  Current judgment: reject this candidate. It removes the specific softmax divider artifact but worsens DSP by `+2` and top latency from `17.259 ms` to `28.731 ms`. The useful next resource work is QKV/MLP controller reuse/fanout restructuring, not broader DSP-to-LUT or large-memory-to-LUTRAM migration.

- Added and tested the shared runtime ATTN/MLP unit candidate:
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`.
  Code update: `hardware/hls/include/hgtxr_e2e_vit.hpp` now supports `HGTXR_E2E_SHARE_RUNTIME_UNITS`; when enabled, fastpath, structured, and full learned runtime paths call ATTN/MLP unit `<0>` instead of instantiating separate `<0>` and `<1>` parity units.
  Rationale: `UNIT_ID` is unused in the current ATTN/MLP unit bodies, so the previous parity split duplicated hardware without changing behavior. Forcing `<0>` aligns the HLS structure with the target cyclic accelerator behavior where the same ATTN/MLP block is temporally shared across Search/Track runtime execution.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`.
  CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, `E2E AXIS vector comparison passed`, and strict Search/Track prefetch traces stayed clean.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_tail16_lutbuf_shareunit_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated clock `2.846 ns` / `351.37 MHz`, top max latency `8,639,430 cycles` / `28.795 ms`, controller max `5,178,118 cycles` / `17.259 ms`.
  Resource result: BRAM_18K `276/624 = 44%`, DSP `1057/1728 = 61%`, FF `165590/460800 = 35%`, LUT `372037/230400 = 161%`, URAM `28/96 = 29%`.
  Baseline delta versus current `tail16_lutbuf_dsppipe4`: BRAM `416 -> 276`, DSP `2369 -> 1057`, FF `331918 -> 165590`, LUT `640454 -> 372037`, URAM unchanged, latency unchanged within one cycle.
  Structural check: no `<1>` ATTN/MLP unit names were found in the shareunit synthesis report/Verilog tree; `csynth_design_size.rpt` shows `hgtxr_e2e_attn_unit<0>` and `hgtxr_e2e_mlp_unit<0>`.
  Current judgment: promote this as the new primary HLS resource-relief candidate. It fixes the largest known duplication and brings DSP under capacity, but LUT is still over capacity at `161%`, so the hardware goal remains open. Next gate is controller LUT/fanout reduction and/or a Vivado implementation attempt on the shareunit candidate.

- Added and tested the shareunit LUT-relief follow-up:
  Code update: `hardware/hls/include/hgtxr_e2e_vit.hpp` now defines `HGTXR_E2E_HEAD_PAR` and uses `kHeadPar` in `hgtxr_e2e_mlp_head`, decoupling Head local parallelism from the Transformer core dense parallelism.
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_dsppipe4_300_mem16`.
  This profile keeps selected LUTRAM scratch buffers and shared runtime ATTN/MLP, but removes the tail16 fabric-lane override. CSim passed with unchanged Search/Track outputs and clean strict prefetch traces. CSynth passed with target `3.333 ns`, estimated `2.846 ns`, top max latency `8,639,430 cycles` / `28.795 ms`, BRAM_18K `276/624 = 44%`, DSP `1177/1728 = 68%`, LUT `339682/230400 = 147%`, URAM `28/96 = 29%`.
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_dsppipe4_300_mem16`.
  This profile removes selected LUTRAM scratch binding to spend BRAM/URAM instead of LUT. CSim passed with unchanged outputs. CSynth passed with top max latency `8,640,519 cycles` / `28.799 ms`, BRAM_18K `340/624 = 54%`, DSP `1185/1728 = 68%`, LUT `294575/230400 = 127%`, URAM `108/96 = 112%`. Reject as a fit candidate because URAM exceeds ZCU104 capacity.
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_lutbuf_shareunit_headpar8_dsppipe4_300_mem16`.
  This profile keeps selected LUTRAM, shared runtime ATTN/MLP, removes tail16 fabric-lane override, and sets `HGTXR_E2E_HEAD_PAR=8`. CSim passed with unchanged Search `[3, 3, 3, 3, 3, 11]` and Track `[217, 217, 217, 217, 217, 213]` outputs and clean prefetch traces.
  `headpar8` CSynth result: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`, top max latency `8,637,990 cycles` / `28.790 ms`, BRAM_18K `276/624 = 44%`, DSP `1183/1728 = 68%`, FF `157807/460800 = 34%`, LUT `313925/230400 = 136%`, URAM `28/96 = 29%`.
  Head-specific result: `hgtxr_e2e_mlp_head` drops from LUT `31259` to `7659` and DSP `2` to `0`; Head latency changes only from `82.905 us` to `83.225 us`.
  Current judgment: `lutbuf_shareunit_headpar8_dsppipe4` is the best balanced HLS candidate so far, but it is not ZCU104-fit because total LUT remains `136%` and the controller alone still estimates LUT `247889`. Continue with controller internal LUT/fanout reduction before promotion to a normal Vivado fit run.

- Added and tested the selective LUTRAM resource candidate requested for many-bank small memories:
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16`.
  Isolated profile delta from `headpar8`: keep shared runtime ATTN/MLP, keep Head local parallelism at `8`, keep 300 MHz target, but keep only `HGTXR_E2E_LUTRAM_NORM=1` among the large scratch buffers. This avoids the over-broad all-LUTRAM policy while still moving enough partitioned memory away from URAM to stay within ZCU104 capacity.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16`.
  CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict prefetch traces. Search stayed at `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track stayed at `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`, top max latency `8,640,615 cycles` / `28.799 ms`, controller max `5,179,142 cycles` / `17.262 ms`.
  Resource result: BRAM_18K `340/624 = 54%`, DSP `1183/1728 = 68%`, FF `156547/460800 = 33%`, LUT `277119/230400 = 120%`, URAM `92/96 = 95%`.
  Memory-policy result: memory LUT drops from `43008` in `lutbuf_shareunit_headpar8` to `6144`, while URAM rises from `28` to `92`, still within device capacity. This is the best resource candidate so far, but it is still not ZCU104-fit because controller LUT remains `247909`.
  Current judgment: do not push DSP into LUT fabric now; DSP is only `68%` and LUT is the blocker. Continue with controller logic reduction, especially weight-cache/unpack/address fanout in QKV/MLP and the 32-lane learned Transformer datapath.

- Added and tested the controller cache-removal fit candidate:
  Profile added: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16`.
  Isolated profile delta from `normlut_shareunit_headpar8`: keep selective `gb.norm` LUTRAM, shared runtime ATTN/MLP, Head local parallelism `8`, and 300 MHz target; disable `HGTXR_E2E_QKV_WEIGHT_CACHE` and `HGTXR_E2E_LN_PARAM_CACHE` so the full learned runtime path reads through the dispatcher/on-chip ROM path directly.
  CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16`.
  CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict prefetch traces. Search stayed at `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`; Track stayed at `block_pairs=2 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16`.
  CSynth result: target `3.333 ns`, estimated `2.846 ns` / `351.37 MHz`, top max latency `8,630,387 cycles` / `28.765 ms`, controller max `5,168,914 cycles` / `17.228 ms`.
  Resource result: BRAM_18K `295/624 = 47%`, DSP `685/1728 = 39%`, FF `104198/460800 = 22%`, LUT `225520/230400 = 97%`, URAM `92/96 = 95%`.
  Controller-specific result: `hgtxr_e2e_controller_run` drops from BRAM `140`, DSP `1180`, FF `151226`, LUT `247909` in `normlut_shareunit_headpar8` to BRAM `95`, DSP `682`, FF `99109`, LUT `196336`.
  Current judgment: promote `normlut_shareunit_headpar8_nocache_dsppipe4` as the first HLS ZCU104-fit candidate. The goal remains open because physical Vivado implementation, routed timing, and power are not yet verified; URAM is tight at `92/96 = 95%`.

- Implemented the `normlut_shareunit_headpar8_nocache_dsppipe4` candidate in Vivado:
  Command passed: `sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16`.
  HLS package already existed for `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip`.
  Vivado used `PL_CLK_MHZ=300.0` and `Performance_ExplorePostRoutePhysOpt`.
  `synth_1` completed with no errors. The opened implemented design reports `DSP48E2 => DSP48E2 ... 623 instances`.
  `impl_1` completed, `write_bitstream` completed successfully, and the script exited with code `0`.
  Overlay output directory: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_overlay`.
  Exported bitstream: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16.bit`, size `19311230` bytes.
  Exported handoff: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16.hwh`, size `492672` bytes.
  Implemented route status: routable nets `115081`, fully routed nets `115081`, routing errors `0`.
  Implemented timing at `clk_pl_0 = 300.030 MHz`: WNS `-0.171 ns`, TNS `-286.331 ns`, WHS `0.005 ns`, THS `0.000 ns`. Vivado reports timing not met, but this is inside the user-allowed `-0.5 ns` WNS tolerance.
  Implemented utilization: CLB LUTs `61679/230400 = 26.77%`, LUT as logic `52460/230400 = 22.77%`, LUT as distributed RAM `8820/101760 = 8.67%`, CLB registers `42737/460800 = 9.27%`, Block RAM Tile `154.5/312 = 49.52%`, URAM `92/96 = 95.83%`, DSP48E2 `623/1728 = 36.05%`.
  Implemented vector-less power estimate: total on-chip `6.642 W`, dynamic `5.917 W`, device static `0.725 W`, PS static `0.102 W`, PL static `0.623 W`; component dynamic includes clocks `0.419 W`, CLB logic `0.753 W`, signals `1.013 W`, Block RAM `0.159 W`, URAM `0.245 W`, DSP `0.657 W`, and PS8 `2.671 W`.
  Power caveat: no SAIF/VCD activity file was supplied and Vivado warned that high-fanout reset activity may make vector-less power inaccurate.
  Current judgment: this candidate now proves physical ZCU104 fit, route completion, bitgen, and 300 MHz implementation within the agreed negative-WNS tolerance. The goal remains open because Search-only and Track-only force-mode latency for this exact candidate has not yet been measured, and board/SAIF mode-specific power is still pending.

- Measured mode-specific force-mode CSynth latency for the exact implemented `normlut_shareunit_headpar8_nocache_dsppipe4` candidate:
  Search command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only`.
  Search report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_search_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  Search result: HLS target `3.333 ns`, estimated clock `2.846 ns` / `351.37 MHz`, latency `8,604,864` cycles, interval `8,604,865` cycles, `28.683 ms` at 300 MHz. Path breakdown: `axis_read_frame` `16,386`, `conv_patch_embedding` `3,293,188`, `global_buffer_load` `12,292`, `controller_run` `5,168,162`, `mlp_head` `24,970`, observable mix `89,849`.
  Track command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only`.
  Track report: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_dsppipe4_300_mem16_track_only_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  Track result: HLS target `3.333 ns`, estimated clock `3.473 ns` / `287.94 MHz`, min-bound latency/interval `698,166/698,167` cycles (`2.327 ms` at 300 MHz), max-bound latency/interval `4,003,638/4,003,639` cycles (`13.345 ms` at 300 MHz). The max-bound is dominated by `event_conv_patch_embedding` max `3,317,956` cycles; use min-bound for the intended Track smoke/TB case and max-bound for worst-case reporting.
  Hybrid 10% Search / 90% Track estimate: using Track min-bound, expected interval `1,488,837` cycles / `4.963 ms` / `201.5 invocations/s`; using Track max-bound, expected interval `4,463,762` cycles / `14.879 ms` / `67.2 invocations/s`; worst-case remains Search `28.683 ms`.
  Current judgment: mode-specific HLS evidence is now available for the exact physically implemented candidate. The candidate proves ZCU104 fit and bitgen but still fails the latency targets: Search `28.683 ms > 4 ms`, Track min-bound `2.327 ms > 1 ms`, Track max-bound `13.345 ms > 1 ms`. Mode-specific SAIF/board power and repeated board latency distribution remain pending.

- Added and measured a patch-parallel Conv/EventConv probe on top of the current full learned runtime-ROM candidate:
  Code update: `hardware/hls/include/hgtxr_e2e_vit.hpp` now defines `HGTXR_E2E_PATCH_PAR`, defaulting to `1`, and both `hgtxr_conv_patch_embedding` and `hgtxr_event_conv_patch_embedding` use partitioned patch accumulator lanes when the macro is greater than one.
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16` variants, all with `HGTXR_E2E_PATCH_PAR=8`.
  Combined CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16`.
  Combined CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict Search/Track prefetch traces.
  Combined CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16`.
  Combined CSynth result: target `3.333 ns`, estimated clock `2.797 ns` / `357.46 MHz`, top interval max `7,094,388 cycles` / `23.648 ms` at 300 MHz, BRAM_18K `295/624 = 47%`, DSP `706/1728 = 40%`, FF `105819/460800 = 22%`, LUT `228799/230400 = 99%`, URAM `92/96 = 95%`.
  Search-only patchpar8 command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_search_only`.
  Search-only patchpar8 result: interval `7,105,729 cycles`, `23.686 ms` at 300 MHz, estimated clock `2.797 ns` / `357.46 MHz`; path remains dominated by `controller_run` `5,168,162 cycles` plus `conv_patch_embedding` `1,794,052 cycles`.
  Track-only patchpar8 command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_dsppipe4_300_mem16_track_only`.
  Track-only patchpar8 result: min-bound interval `698,167 cycles` / `2.327 ms`, max-bound interval `2,467,639 cycles` / `8.225 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`; max-bound remains dominated by `event_conv_patch_embedding` `1,781,956 cycles`.
  Hybrid 10% Search / 90% Track patchpar8 estimate: using Track min-bound, expected interval `1,338,923 cycles` / `4.463 ms` / `224.1 invocations/s`; using Track max-bound, expected interval `2,931,448 cycles` / `9.771 ms` / `102.3 invocations/s`; worst-case remains Search `23.686 ms`.
  Current judgment: patchpar8 is useful but insufficient. It reduces Search from `28.683 ms` to `23.686 ms` and Track max-bound from `13.345 ms` to `8.225 ms`, but the design still fails Search `4 ms` and Track `1 ms`. The next latency work should target `controller_run` and EventConv memory/loop structure; this new probe also needs Vivado implementation before physical 300 MHz acceptance because Track-only HLS estimate is still `287.94 MHz`.

- Added and measured a token-loop Patch Embedding probe on top of patchpar8:
  Code update: `hardware/hls/include/hgtxr_e2e_vit.hpp` now defines `HGTXR_E2E_PATCH_TOKEN_LOOP`, defaulting to `0`, and both `hgtxr_conv_patch_embedding` and `hgtxr_event_conv_patch_embedding` can loop directly over `token < active_tokens` when enabled.
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16` variants, all with `HGTXR_E2E_PATCH_PAR=8` and `HGTXR_E2E_PATCH_TOKEN_LOOP=1`.
  Combined CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16`.
  Combined CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict Search/Track prefetch traces.
  Track-only token-loop command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_track_only`.
  Track-only token-loop result: interval `1,143,415 cycles`, `3.811 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`; path breakdown is `axis_read_frame` `4,098`, `event_conv_patch_embedding` `457,732`, `global_buffer_load` `3,076`, `controller_run` `628,712`, `mlp_head` `6,538`, observable mix `43,241`.
  Search-only token-loop command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16_search_only`.
  Search-only token-loop result: interval `7,105,729 cycles`, `23.686 ms` at 300 MHz, estimated clock `2.797 ns` / `357.46 MHz`; Search path remains dominated by `controller_run` `5,168,162 cycles` plus `conv_patch_embedding` `1,794,052 cycles`.
  Combined token-loop command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar8_tokenloop_dsppipe4_300_mem16`.
  Combined token-loop result: target `3.333 ns`, estimated clock `2.797 ns` / `357.46 MHz`, top interval max `6,295,476 cycles` / `20.983 ms` at 300 MHz, BRAM_18K `295/624 = 47%`, DSP `706/1728 = 40%`, FF `105817/460800 = 22%`, LUT `228745/230400 = 99%`, URAM `92/96 = 95%`.
  Hybrid 10% Search / 90% Track token-loop estimate: expected interval `1,739,646 cycles` / `5.799 ms` / `172.4 invocations/s`; worst-case remains Search `23.686 ms`.
  Current judgment: token-loop is useful and preserves outputs. It reduces EventConv force-mode latency from `1,781,956` to `457,732 cycles` and Track interval from `8.225 ms` to `3.811 ms`, but Search stays `23.686 ms` and Track still misses `1 ms`. The next latency work should target `controller_run` and Search Conv. This probe also needs Vivado implementation before physical 300 MHz acceptance because Track-only HLS estimate is still `287.94 MHz`.

- Added and measured a patchpar16 token-loop DSE:
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16` variants, all with `HGTXR_E2E_PATCH_PAR=16` and `HGTXR_E2E_PATCH_TOKEN_LOOP=1`.
  Combined CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16`.
  Combined CSim result: Search `[3, 3, 3, 3, 3, 11]`, Track `[217, 217, 217, 217, 217, 213]`, runtime states `0/1`, and clean strict Search/Track prefetch traces.
  Track-only patchpar16 command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_track_only`.
  Track-only patchpar16 result: interval `796,290 cycles`, `2.654 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`; path breakdown is `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `628,712`, `mlp_head` `6,538`, observable mix `43,241`.
  Search-only patchpar16 command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16_search_only`.
  Search-only patchpar16 result: interval `5,692,609 cycles`, `18.975 ms` at 300 MHz, estimated clock `2.797 ns` / `357.46 MHz`; path remains dominated by `controller_run` `5,168,162 cycles`, while `conv_patch_embedding` drops to `380,932 cycles`.
  Combined patchpar16 command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_normlut_shareunit_headpar8_nocache_patchpar16_tokenloop_dsppipe4_300_mem16`.
  Combined patchpar16 result: target `3.333 ns`, estimated clock `2.797 ns` / `357.46 MHz`, top interval max `5,705,664 cycles` / `19.019 ms` at 300 MHz, BRAM_18K `309/624 = 49%`, DSP `720/1728 = 41%`, FF `106881/460800 = 23%`, LUT `238345/230400 = 103%`, URAM `92/96 = 95%`.
  Hybrid 10% Search / 90% Track patchpar16 estimate: expected interval `1,285,922 cycles` / `4.286 ms` / `233.3 invocations/s`; worst-case remains Search `18.975 ms`.
  Legal middle-point check: `PATCH_PAR=12` was tried as a temporary compile probe, but CSim compilation failed at `static_assert(kPatchElems % kPatchPar == 0)`, so it is not legal under the current patch loop invariant and is not retained as a runnable profile.
  Current judgment: patchpar16 is a useful negative DSE. It improves Search by `19.9%` and Track by `30.4%` versus patchpar8 token-loop, but combined runtime top exceeds ZCU104 LUT capacity (`103%`). Do not push this profile to Vivado in its current form. Keep `patchpar8_tokenloop` as the current fit-capable latency candidate and next reduce `controller_run` latency/LUT or recover enough LUT to reintroduce higher patch parallelism.

- Added and measured a patchpar16 no-observe norm-BRAM fit DSE:
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` variants.
  Profile intent: keep `PATCH_PAR=16` and token-loop Conv/EventConv latency benefits, but recover LUT by disabling the debug observable datapath, moving norm storage back to BRAM, and reducing local Head parallelism to `HGTXR_E2E_HEAD_PAR=4`.
  Combined CSim command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined CSim result: Search runtime state `0`, Track runtime state `1`, and clean strict prefetch traces for both modes. Because `HGTXR_E2E_OBSERVE_DATAPATH=0`, the debug mixed output word is intentionally removed; strict golden refresh is required before final functional acceptance.
  Combined CSynth command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined CSynth result: target `3.333 ns`, estimated clock `2.777 ns` / `360.10 MHz`, interval min/max `124,486 / 5,617,495 cycles`, `0.415 / 18.725 ms` at 300 MHz.
  Combined resources: BRAM_18K `341/624 = 54%`, DSP `719/1728 = 41%`, FF `105285/460800 = 22%`, LUT `222252/230400 = 96%`, URAM `92/96 = 95%`; this recovers enough LUT versus rejected patchpar16 (`238345`, `103%`).
  Search-only force-mode command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Search-only result: interval max `5,604,440 cycles`, `18.681 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`; path breakdown is `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `5,169,698`, `mlp_head` `25,114`, plus top-level overhead.
  Track-only force-mode command passed: `sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  Track-only result: interval `753,383 cycles`, `2.511 ms` at 300 MHz or `2.616 ms` at estimated Fmax, estimated clock `3.473 ns` / `287.94 MHz`; path breakdown is `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `628,902`, `mlp_head` `6,682`, plus top-level overhead.
  Hybrid 10% Search / 90% Track estimate: force-mode expected interval `1,238,489 cycles` / `4.128 ms` / `242.2 invocations/s`; combined-envelope expected interval `673,787 cycles` / `2.246 ms` / `445.2 invocations/s`; worst-case remains Search or combined max around `18.7 ms`.
  Current judgment: this becomes the best HLS-fit latency/resource candidate so far, but it is not a final acceptance point. Search still misses `4 ms`, force-mode Track still misses `1 ms`, Track-only HLS estimated clock misses 300 MHz, and Vivado implementation/board or SAIF-backed power are still pending.

- Implemented the patchpar16 no-observe norm-BRAM fit candidate in Vivado:
  Command passed: `sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  HLS package succeeded and generated `component.xml` plus `export.zip` under `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip`.
  Vivado used `PL_CLK_MHZ=300.0` and `Performance_ExplorePostRoutePhysOpt`.
  `synth_1`, `impl_1`, post-route `phys_opt_design`, and `write_bitstream` completed successfully; the script exited with code `0`.
  Final implemented timing is clean at the requested 300 MHz: WNS `0.000 ns`, TNS `0.000 ns`, WHS `0.006 ns`, THS `0.000 ns`, and the timing summary says all user-specified constraints are met.
  Implemented route status: routable nets `106,191`, fully routed nets `106,191`, routing errors `0`.
  Overlay output directory: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay`.
  Exported bitstream: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.bit`, about `19 MB`.
  Exported handoff: `hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16.hwh`, about `482 KB`.
  Implemented utilization: CLB LUT `48,909/230,400 = 21.23%`, LUT as Memory `1,534/101,760 = 1.51%`, CLB registers `44,104/460,800 = 9.57%`, Block RAM Tile `169.5/312 = 54.33%`, URAM `92/96 = 95.83%`, DSP48E2 `690/1,728 = 39.93%`.
  Implemented vectorless power: total on-chip `6.412 W`, dynamic `5.688 W`, device static `0.724 W`, PS static `0.102 W`, PL static `0.622 W`.
  Current judgment: this is now the strongest physical 300 MHz full learned runtime-ROM fit candidate because it is official timing-clean, routed, and bitgen-complete. The goal remains open because Search `18.680 ms` and Track `2.511 ms @300MHz` still miss the user latency targets, and no-observe strict golden/RTL or board same-input/same-output evidence is still pending.

- Audited the active patchpar16 no-observe norm-BRAM AXIS payload contract:
  Command passed: `python3 hardware/tools/check_hls_axis_payload_contract.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16 --json-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_2026_06_30.json --markdown-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_2026_06_30.md`.
  Result status: `structural-pass-missing-rtl-tv`.
  The generated synthesis RTL has `out_state_widths=[8]`, `axis_out_tdata_zero_assignment_count=0`, and `axis_out_tdata_has_out_state_source=true`.
  C/RTL TV output files are still absent for this profile, so this is structural evidence only, not same-input/same-output signoff.

- Audited the active patchpar16 no-observe norm-BRAM structural objective contract:
  Updated `hardware/tools/write_prefetchall4_300_contract_audit.py` to accept `--profile` and to recognize profile-specific LUTRAM nonlinear ROMs plus shared runtime ATTN/MLP module naming.
  Regression passed: `python3 -m unittest hardware.tests.test_write_prefetchall4_300_contract_audit`.
  Command passed: `python3 hardware/tools/write_prefetchall4_300_contract_audit.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16 --json-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_contract_audit_2026_06_30.json --markdown-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_contract_audit_2026_06_30.md`.
  Result status: `pass`, checks `16/16`; this directly covers runtime scheduler, on-chip ROM classification, omitted weight AXI, nonlinear ROM source paths, Search dispatcher prefetch/immediate-start trace gate, and generated full Transformer module structure for the active no-observe profile.

- Re-ran HLS C/RTL cosim and direct XSIMK probing for the active patchpar16 no-observe norm-BRAM profile:
  Command run: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh cosim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  C TB passed and generated C reference TV. Search payloads were `[3, 3, 3, 3, 3, 3]`; Track payloads were `[217, 217, 217, 217, 217, 217]`; runtime states were `0/1`; strict prefetch traces had `violations=0`.
  Standard Verilog cosim still failed before RTL transaction completion. `sim/report/hgtxr_e2e_axis_top_cosim.rpt` reports Verilog `Fail` with all latency/interval fields `NA`, and XSIM reported `ERROR: unexpected exception when evaluating tcl command`.
  Command run: `python3 hardware/tools/check_hls_xsimk_direct_probe.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16 --timeout-s 180 --search-tail 3 --track-tail 217 --json-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_xsimk_direct_probe_after_cosim_2026_06_30.json --markdown-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_xsimk_direct_probe_after_cosim_2026_06_30.md`.
  Direct XSIMK status was `direct-kernel-timeout`: the snapshot exists, elaboration completed, `-exec-continue` started, and the RTL progress banner reached `0 / 2` transactions at sim time `109000`, but no transaction completed within `180 s`.
  Command run: `python3 hardware/tools/check_hls_axis_payload_contract.py --profile par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16 --json-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_after_xsimk_2026_06_30.json --markdown-out hardware/generated/signoff/par32_patchpar16_noobs_normbram_hls_axis_payload_contract_after_xsimk_2026_06_30.md`.
  Payload audit status after direct probing was `fail-active-payload-tv`: C output payloads exist, but RTL output TV files contain only the runtime marker and no payload values. This means same-input/same-output functional signoff is still open even though C TB and structural RTL checks progressed.

- Added and measured a dense64 DSE on top of the current best patchpar16 no-observe norm-BRAM candidate:
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` variants.
  Profile intent: keep the same full learned runtime-ROM objective path while testing `HGTXR_E2E_DENSE_PAR=64` and `HGTXR_E2E_ATTN_PAR=64` as a direct controller bottleneck DSE.
  Script syntax check passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
  Combined dense64 CSim command passed: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined dense64 CSim result: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
  Search-only dense64 command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Search-only dense64 result: interval `5,833,024 cycles`, `19.443 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`, resources BRAM_18K `339`, DSP `955`, LUT `225,097`, URAM `92`.
  Track-only dense64 command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_dense64_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  Track-only dense64 result: interval `781,851 cycles`, `2.606 ms` at 300 MHz, estimated clock `3.473 ns` / `287.94 MHz`, resources BRAM_18K `337`, DSP `950`, LUT `221,339`, URAM `71`.
  Current judgment: dense64 is rejected. It preserves functional CSim output but worsens the active no-observe latency envelope: Search `18.681 ms -> 19.443 ms`, Track `2.511 ms -> 2.606 ms`. Controller reports show the reason: Search attention improves (`288,732 -> 208,086 cycles`) but MLP worsens (`996,751 -> 1,134,543 cycles`), so the next latency DSE should target MLP dataflow/weight-cache structure rather than raising global dense parallelism.

- Added and measured an MLP token-parallel DSE on top of the current best patchpar16 no-observe norm-BRAM candidate:
  Code changes: added `HGTXR_E2E_MLP_TOKEN_PAR` with default `1`, and added a guarded token-group branch in `hgtxr_e2e_mlp_unit` for `HGTXR_E2E_MLP_TOKEN_PAR > 1`. Existing profiles keep the original MLP path.
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` variants.
  Profile intent: reuse one W1/W2 weight vector across two token lanes while preserving the full learned runtime-ROM objective path.
  Script and whitespace checks passed: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh`; `git diff --check -- hardware/hls/include/hgtxr_e2e_vit.hpp hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
  Combined `mlptok2` CSim command passed: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined `mlptok2` CSim result: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
  Search-only `mlptok2` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Search-only `mlptok2` result: interval `6,153,432 cycles`, `20.511 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`; resources BRAM_18K `355`, DSP `571`, LUT `167,589`, URAM `92`.
  Track-only `mlptok2` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlptok2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  Track-only `mlptok2` result: interval `822,007 cycles`, `2.740 ms` at 300 MHz or `2.855 ms` at estimated Fmax; resources BRAM_18K `353`, DSP `694`, LUT `163,318`, URAM `71`.
  HLS inferred token-dimension cyclic factor `2` partitioning on `gb.norm` and `gb.hidden`. This reduced LUT/DSP pressure, but the MLP grouped loop iteration latency rose from `14,916` to `34,121 cycles`.
  Current judgment: `mlptok2` is rejected for latency. It worsens the active no-observe envelope: Search `18.681 ms -> 20.511 ms`, Track `2.511 ms -> 2.740 ms` at 300 MHz, and hybrid 10/90 expected interval worsens from about `4.128 ms` to about `4.517 ms`.

- Added and measured an MLP fused-W2 DSE on top of the current best patchpar16 no-observe norm-BRAM candidate:
  Code changes: added `HGTXR_E2E_MLP_FUSED_W2` with default `0`, and added a guarded fused branch in `hgtxr_e2e_mlp_unit`. Existing profiles keep the original MLP path.
  Fused branch intent: compute W1 hidden chunks and immediately accumulate W2 output into a complete-partitioned local `out_acc` vector, avoiding the separate full hidden-buffer W2 read pass while still writing `gb.hidden` for compatibility.
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` variants.
  Script and whitespace checks passed before synthesis: `bash -n hardware/scripts/run/run_e2e_q4w8a_no_board.sh`; `git diff --check -- hardware/hls/include/hgtxr_e2e_vit.hpp hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
  Combined `mlpfuse` CSim command passed: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined `mlpfuse` CSim result: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
  Search-only `mlpfuse` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Search-only `mlpfuse` result: interval `4,488,024 cycles`, `14.960 ms` at 300 MHz or `19.337 ms` at estimated Fmax; estimated Fmax `232.10 MHz`; resources BRAM_18K `227`, DSP `507`, LUT `168,027`, URAM `92`.
  Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `4,053,282`, `mlp_head` `25,114`, plus top-level overhead.
  Search controller subpath: attention unit `288,732 cycles`, MLP unit `717,647 cycles`; the active profile MLP was `996,751 cycles`.
  Track-only `mlpfuse` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfuse_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  Track-only `mlpfuse` result: interval `613,831 cycles`, `2.046 ms` at 300 MHz or `2.645 ms` at estimated Fmax; estimated Fmax `232.10 MHz`; resources BRAM_18K `225`, DSP `630`, LUT `163,787`, URAM `71`.
  Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `489,350`, `mlp_head` `6,682`, plus top-level overhead.
  Track controller subpath: attention unit `58,905 cycles`, MLP unit `178,836 cycles`; the active profile MLP was `248,612 cycles`.
  Hybrid 10% Search / 90% Track force-mode estimate: about `1,001,250 cycles`, `3.338 ms` at 300 MHz or `4.314 ms` at estimated Fmax.
  Current judgment: `mlpfuse` is cycle-positive but timing-negative. It improves active no-observe cycle latency by `19.9%` for Search and `18.5%` for Track, but the complete-partitioned `out_acc[kEmbed]` accumulation structure drops HLS estimated Fmax to `232.10 MHz`, so it cannot replace the active physical baseline. Next DSE should preserve fused W2 scheduling but tile or stage the output accumulator to recover timing.

- Added and measured an MLP fused-W2 banked-accumulator DSE on top of the current best patchpar16 no-observe norm-BRAM candidate:
  Code changes: added `HGTXR_E2E_MLP_FUSED_W2_BANKED_ACC` with default `0`; when enabled inside the fused-W2 branch, `out_acc` is cyclic-partitioned by `kDensePar` rather than complete-partitioned across `kEmbed`.
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` variants.
  Profile intent: keep the fused-W2 schedule that reduced MLP cycles while reducing accumulator fanout/timing pressure.
  Combined `mlpfusebank` CSim had already passed before Track synthesis: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
  Search-only `mlpfusebank` result: interval `4,490,328 cycles`, `14.968 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`; resources BRAM_18K `227`, DSP `507`, FF `60,218`, LUT `156,534`, URAM `92`.
  Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `4,055,586`, `mlp_head` `25,114`, plus top-level overhead.
  Search controller subpath: dispatcher prefetch `6,933 cycles`, attention unit `288,732 cycles`, MLP unit `718,223 cycles`, block-loop iteration latency `1,006,960 cycles`, trip count `4`.
  Track-only `mlpfusebank` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  Track-only `mlpfusebank` result: interval `614,119 cycles`, `2.047 ms` at 300 MHz or `2.133 ms` at estimated Fmax; estimated clock `3.473 ns` / `287.94 MHz`; resources BRAM_18K `225`, DSP `630`, FF `72,120`, LUT `152,295`, URAM `71`.
  Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `489,638`, `mlp_head` `6,682`, plus top-level overhead.
  Track controller subpath: dispatcher prefetch `6,926 cycles`, attention unit `58,905 cycles`, MLP unit `178,980 cycles`, block-loop iteration latency `237,889 cycles`, trip count `2`.
  Hybrid 10% Search / 90% Track force-mode estimate: about `1,001,740 cycles`, `3.339 ms` at 300 MHz, or `3.479 ms` if limited by Track estimated Fmax `287.94 MHz`.
  Current judgment: `mlpfusebank` is the best HLS latency direction so far because it keeps the `mlpfuse` cycle gain and recovers Search HLS Fmax from `232.10 MHz` to `360.10 MHz`. It is still not the active physical baseline because Track-only HLS estimated clock remains `3.473 ns` / `287.94 MHz`, Search remains above `4 ms`, Track remains above `1 ms`, and Vivado route/board or RTL same-output signoff has not been run for this new DSE.

- Ran Vivado AXIS/DMA implementation for the `mlpfusebank` DSE:
  Command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  HLS IP package used: `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`.
  Bitstream generation passed and copied artifacts to:
  `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/`
  and `hardware/pynq/hgtxr/`.
  Implemented timing report:
  `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay.runs/impl_1/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_timing_summary_implemented.rpt`.
  Timing result: `clk_pl_0` period `3.333 ns` / `300.030 MHz`; WNS `-0.064 ns`; TNS `-39.952 ns`; setup failing endpoints `1309`; WHS `0.004 ns`; THS `0.000 ns`.
  Interpretation: Vivado nominal timing is not met, but this passes the user-relaxed setup WNS tolerance of `-0.5 ns`.
  Route status: routable nets `108,842`, fully routed nets `108,842`, routing errors `0`.
  Implemented resources: CLB LUT `49,022/230,400 = 21.28%`, CLB registers `45,244/460,800 = 9.82%`, Block RAM Tile `121.5/312 = 38.94%`, URAM `92/96 = 95.83%`, DSP `690/1,728 = 39.93%`.
  Implemented vectorless power: total on-chip `6.231 W`, dynamic `5.509 W`, device static `0.722 W`, PS static `0.102 W`, PL static `0.620 W`. Main dynamic hierarchy: `hgtxr_e2e_axis_top_0 = 2.581 W`, `psu = 2.674 W`, `axi_mem = 0.202 W`, `axi_dma_out = 0.024 W`, `axi_dma_in = 0.007 W`, `axi_ctrl = 0.021 W`.
  Current judgment update: `mlpfusebank` is physically fit and has better HLS latency than the active no-observe baseline, but the goal is still incomplete because Search is `14.968 ms > 4 ms`, Track is `2.047 ms > 1 ms`, nominal timing is slightly negative, and RTL/board same-input/same-output evidence is still missing.
  Next concrete DSE: keep fused-W2 banked accumulation; add explicit DSP/MLP pipeline staging for the Vivado DPIP/DPOP MREG/PREG warnings; then reduce the full learned Search/Track operator count or per-layer work because placement alone cannot close a `3.7x` Search latency gap.

- Added and measured an MLP fused-W2 no-hidden-store DSE on top of `mlpfusebank`:
  Code changes: added `HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE` with default `0`; when enabled inside the fused-W2 branch, the local `hidden_vec` is still used for W2 accumulation but the compatibility write to `gb.hidden[t][h]` is skipped.
  Profiles added: combined, Search-only, and Track-only `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16` variants.
  Combined `mlpfusebank_nohidden` CSim command passed: `timeout 1800 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csim par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined `mlpfusebank_nohidden` CSim result: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict prefetch traces clean.
  Search-only `mlpfusebank_nohidden` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Search-only `mlpfusebank_nohidden` result: interval `4,490,328 cycles`, `14.968 ms` at 300 MHz, estimated clock `2.777 ns` / `360.10 MHz`; resources BRAM_18K `227`, DSP `507`, FF `60,218`, LUT `156,534`, URAM `92`.
  Track-only `mlpfusebank_nohidden` command passed: `timeout 7200 sh hardware/scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_nohidden_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  Track-only `mlpfusebank_nohidden` result: interval `614,119 cycles`, `2.047 ms` at 300 MHz or `2.133 ms` at estimated Fmax; estimated clock `3.473 ns` / `287.94 MHz`; resources BRAM_18K `225`, DSP `630`, FF `72,120`, LUT `152,295`, URAM `71`.
  Current judgment: `mlpfusebank_nohidden` is neutral. HLS already schedules or optimizes away the hidden write cost in this fused path, so skipping it does not reduce cycles, resources, or timing pressure. Do not route this candidate; keep `mlpfusebank` as the current physical baseline.

- Implemented the `mlpfusebank_hpar2` full learned runtime-ROM candidate in Vivado:
  HLS package completed for `hardware/generated/hgtxr_e2e_axis_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml`.
  Force-mode latency improved versus `mlpfusebank`: Search interval `3,900,504 cycles` / `13.002 ms`, Track interval `540,391 cycles` / `1.801 ms`, hybrid 10% Search / 90% Track `876,402 cycles` / `2.921 ms` at 300 MHz.
  Vivado command reached bitgen success: `timeout 7200 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  The runner exited after bitgen with `ERROR: [Common 17-36] Cannot write file ... utilization_hierarchical_implemented.rpt: File name too long`; the implemented design and bitstream were already generated.
  Copied the generated `.bit/.hwh` manually into `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16_overlay/` and `hardware/pynq/hgtxr/`.
  Fixed `hardware/vivado/scripts/build_e2e_axis_dma_bitstream.tcl` so long artifact names use `hgtxr_e2e_axis_dma_impl` as the report prefix while preserving full artifact names for `.bit/.hwh` copies.
  Post-route physopt timing: clock `300.030 MHz`, WNS `-0.094 ns`, TNS `-107.248 ns`, WHS `0.007 ns`, THS `0.000 ns`; Vivado nominal timing is not met, but the user-relaxed `-0.5 ns` WNS tolerance is met.
  Route status is clean: routable nets `110,620`, fully routed nets `110,620`, routing errors `0`.
  Implemented resources fit and preserve compute fabric: CLB LUT `49,867/230,400 = 21.64%`, CLB registers `45,300/460,800 = 9.83%`, Block RAM Tile `121.5/312 = 38.94%`, URAM `92/96 = 95.83%`, DSP `722/1,728 = 41.78%`.
  Vectorless power: total on-chip `6.231 W`, dynamic `5.509 W`, device static `0.722 W`, PS static `0.102 W`, PL static `0.620 W`.
  Current judgment: `mlpfusebank_hpar2` is the best physically implemented full learned runtime-ROM candidate so far, but the overall goal remains incomplete because Search `13.002 ms > 4 ms`, Track `1.801 ms > 1 ms`, same-input/same-output RTL or board validation is missing, and mode-specific measured latency/power are still pending.

- Implemented the `attntok2_attnbram` full learned runtime-ROM candidate in Vivado:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Force-mode HLS Search result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,623,832 cycles`, `8.745 ms` at 300 MHz. Search path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090`, `mlp_head` `25,114`.
  Force-mode HLS Track result: estimated clock `3.473 ns` / `287.94 MHz`, interval `380,711 cycles`, `1.322 ms` at 300 MHz. Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
  Hybrid 10% Search / 90% Track force-mode estimate: `604,023 cycles`, `2.013 ms`, about `496.7 invocations/s`; worst-case remains Search `8.745 ms`.
  Vivado implementation and bitgen completed; `.bit/.hwh` artifacts were copied to `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_300_mem16_overlay/` and `hardware/pynq/hgtxr/`.
  Implemented timing: clock `300.030 MHz`, WNS `-0.357 ns`, TNS `-479.165 ns`, setup failing endpoints `4,384`, WHS `0.010 ns`, THS `0.000 ns`; Vivado nominal timing is not met, but the user-relaxed `-0.5 ns` WNS tolerance is met.
  Route status is clean: logical nets `936,375`, routable nets `133,219`, fully routed nets `133,219`, routing errors `0`.
  Implemented resources: CLB LUT `60,878/230,400 = 26.42%`, CLB registers `55,536/460,800 = 12.05%`, Block RAM Tile `137.5/312 = 44.07%`, URAM `76/96 = 79.17%`, DSP `946/1,728 = 54.75%`.
  Vectorless power: total on-chip `6.099 W`, dynamic `5.381 W`, device static `0.719 W`, PS static `0.102 W`, PL static `0.617 W`; confidence `Medium`.
  Current judgment: `attntok2_attnbram` replaces `mlpfusebank_hpar2` as the best physically implemented full learned runtime-ROM latency candidate. Search improves `13.002 ms -> 8.745 ms`, Track improves `1.801 ms -> 1.322 ms`, and hybrid 10/90 improves `2.921 ms -> 2.013 ms`. The goal remains incomplete because Search still needs about `1,423,832` fewer II cycles to reach `4 ms`, Track still needs about `80,711` fewer II cycles to reach `1 ms`, Vivado nominal timing is negative, and RTL/board same-input same-output plus mode-specific measured latency/power are still pending.

- Added and measured the `hpar4_tok2_attntok2_attnbram` negative DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: test whether raising `HGTXR_E2E_MLP_FUSED_W2_HP_PAR` from `2` to `4` reduces the remaining controller-dominated latency in the promoted Attention-Token2 / Attention-BRAM candidate.
  Combined CSim passed with the same functional outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,623,832 cycles`, `8.745 ms` at 300 MHz. Path remains `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090`, `mlp_head` `25,114`.
  Track-only HLS result: estimated clock `3.473 ns` / `287.94 MHz`, interval `380,711 cycles`, `1.322 ms` at 300 MHz. Path remains `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
  Resource comparison versus promoted `hpar2` `attntok2_attnbram`: Search-only DSP `763 -> 891`, FF `75,794 -> 88,531`, LUT `188,424 -> 199,119`; Track-only DSP `886 -> 1,014`, FF `87,415 -> 100,152`, LUT `184,150 -> 194,845`.
  Bottleneck evidence: HLS continues to report memory-port II violations around `w2_weight_cache` and global-buffer/token memories. Increasing hidden-lane arithmetic parallelism without matching memory banking leaves Search/Track II unchanged.
  Current judgment: do not promote or route `hpar4`; keep `hpar2` `attntok2_attnbram` as the current best physical candidate. Next DSE should directly bank/reshape the hot weight and global-buffer memories or reduce the controller work per layer.

- Added and measured the `hpar4_w2bank8_tok2_attntok2_attnbram` negative DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2bank8_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: test whether `HGTXR_E2E_W2_WEIGHT_CACHE_BANKS=8` plus cyclic partitioning of `w2_weight_cache` removes the MLP W2 memory-port II violation.
  Code change: added default `HGTXR_E2E_W2_WEIGHT_CACHE_BANKS=1`, conditional `#pragma HLS ARRAY_PARTITION variable=w2_weight_cache cyclic factor=HGTXR_E2E_W2_WEIGHT_CACHE_BANKS dim=1`, and combined/Search-only/Track-only `hpar4_w2bank8` runner profiles.
  Combined CSim passed with the same outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,623,836 cycles`, `8.745 ms` at 300 MHz. Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,094`, `mlp_head` `25,114`.
  Resource result: BRAM_18K `339`, DSP `891`, FF `90,532`, LUT `199,958`, URAM `76`; this is worse than `hpar4` Search-only BRAM_18K `265` with no latency gain.
  Bottleneck evidence: HLS confirms `array_partition variable=w2_weight_cache cyclic factor=8 dim=1`, but the MLP W2 loop still reports memory-port II pressure on `w2_weight_cache` and keeps achieved II `2`.
  Current judgment: do not promote, route, or spend Track/Vivado time on `w2bank8` as-is. The next candidate needs explicit W2 data-layout/schedule changes so each unrolled lane reads a distinct bank, or a controller-work reduction that bypasses this bottleneck.

- Added and measured the `hpar2_skiphidden_tok2_attntok2_attnbram` neutral DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_skiphidden_tok2_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: test whether `HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1` reduces controller/global-buffer pressure by removing the fused-path `gb.hidden` write.
  Combined CSim passed with the same outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,623,832 cycles`, `8.745 ms` at 300 MHz. Path is unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,189,090`, `mlp_head` `25,114`.
  Resource result matches the promoted Search-only estimate: BRAM_18K `265`, DSP `763`, FF `75,794`, LUT `188,424`, URAM `76`.
  Track-only HLS result: estimated clock `3.473 ns` / `287.94 MHz`, interval `380,711 cycles`, `1.322 ms` in the HLS report. Resources match the promoted Track-only estimate: BRAM_18K `263`, DSP `886`, FF `87,415`, LUT `184,150`, URAM `55`.
  Internal loop evidence: `VITIS_LOOP_2837_15_VITIS_LOOP_2839_16` reports achieved II `1`, but the top-level and controller intervals remain unchanged.
  Current judgment: `skiphidden` is functionally valid but not a latency improvement; do not promote or route it as-is. Keep `hpar2` `attntok2_attnbram` as the current best physical candidate.

- Added and measured the `mtok2_dtok4_attntok2_attnbram` positive HLS DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: raise `HGTXR_E2E_DENSE_TOKEN_PAR` from `2` to `4` while keeping `HGTXR_E2E_MLP_TOKEN_PAR=2`, targeting QKV/output-projection latency without further expanding the MLP W2 path.
  Combined CSim passed with the same outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,465,752 cycles`, `8.218 ms` in the HLS report. Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `380,932`, `global_buffer_load` `12,292`, `controller_run` `2,031,010`, `mlp_head` `25,114`.
  Track-only HLS result: estimated clock `3.473 ns` / `287.94 MHz`, interval `360,903 cycles`, `1.253 ms` in the HLS report. Path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `110,596`, `global_buffer_load` `3,076`, `controller_run` `236,422`, `mlp_head` `6,682`.
  Hybrid 10% Search / 90% Track force-mode estimate: `571,388 cycles`, about `1.905 ms` at exact `300 MHz`, about `524.9 invocations/s`.
  Latency improves versus the current physical candidate's HLS numbers: Search `2,623,832 -> 2,465,752 cycles`, Track `380,711 -> 360,903 cycles`.
  Resource risk is high: Search-only LUT `212,438/230,400 = 92%`, DSP `1,019/1,728 = 58%`; Track-only LUT `208,129/230,400 = 90%`, DSP `1,142/1,728 = 66%`.
  Combined package/CSynth later failed the fit gate: LUT `278,828/230,400 = 121%`, BRAM_18K `331/624 = 53%`, DSP `1,231/1,728 = 71%`, URAM `76/96 = 79%`.
  Current judgment: `dtok4` is a useful latency-direction signal, but not a physical baseline. Do not route it as-is; keep `attntok2_attnbram` as the current physical baseline.

- Added and measured the `mtok2_dtok3_attntok2_attnbram` fit-recovery DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok3_attntok2_attnbram_nocache_patchpar16_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: reduce dense-token parallelism from `dtok4` to `dtok3` while keeping `HGTXR_E2E_MLP_TOKEN_PAR=2`, trying to preserve part of the latency benefit without exceeding ZCU104 LUT capacity.
  Combined CSim passed with the same outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Combined package/IP export completed.
  Combined CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_dtok3_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  Fit result: LUT `274,923/230,400 = 119%`, BRAM_18K `299/624 = 47%`, DSP `1,103/1,728 = 63%`, FF `137,005/460,800 = 29%`, URAM `76/96 = 79%`.
  Current judgment: `dtok3` also fails the combined full-runtime fit gate. Search-only and Track-only force-mode runs were skipped because a profile that already exceeds the device LUT budget should not replace the current physically implemented baseline.

- Added and measured the `patchpar32_tok2_attntok2_attnbram` Conv/EventConv DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: raise `HGTXR_E2E_PATCH_PAR` from `16` to `32` while keeping the promoted `hpar2` `attntok2_attnbram` runtime-ROM design otherwise unchanged.
  Combined CSim passed with the same outputs as the promoted candidate: Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,500,952 cycles`, `8.336 ms`. Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `2,189,090`, `mlp_head` `25,114`.
  Track-only HLS result: estimated clock `3.473 ns` / `287.94 MHz`, interval `356,135 cycles`, HLS report latency `1.237 ms`, exact `300 MHz` latency about `1.187 ms`. Path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `256,230`, `mlp_head` `6,682`.
  Hybrid 10% Search / 90% Track force-mode estimate: `570,617 cycles`, about `1.902 ms` at exact `300 MHz`, about `525.7 invocations/s`.
  Latency improves versus the current physical candidate's HLS numbers: Search `2,623,832 -> 2,500,952 cycles`, Track `380,711 -> 356,135 cycles`, Hybrid `604,023 -> 570,617 cycles`.
  Combined CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  Combined HLS resource estimate: LUT `258,869/230,400 = 112%`, BRAM_18K `267/624 = 42%`, DSP `1,002/1,728 = 57%`, FF `126,407/460,800 = 27%`, URAM `76/96 = 79%`.
  Current judgment before physical route: `patchpar32` is the best force-mode HLS latency candidate in the `attntok2_attnbram` family, but not yet a physical baseline. Because the already-routed `patchpar16` candidate also had an over-conservative combined HLS LUT estimate (`110%`), the `112%` HLS LUT estimate is a route-risk signal rather than a proof of infeasibility.

- Promoted `patchpar32_tok2_attntok2_attnbram` through the Vivado physical gate:
  Added the combined `patchpar32` profile to `hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh` and verified `bash -n`, `git diff --check`, and packaged HLS IP `component.xml`.
  Command run: `timeout 21600 sh hardware/scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_tok2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Vivado result: route and bitgen completed for `hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay`.
  Final implemented timing: WNS `-0.082 ns`, TNS `-60.586 ns`, WHS `0.002 ns`, THS `0.000 ns`. This is nominally timing-failing, but passes the user-relaxed WNS allowance of `-0.5 ns`.
  Route status: logical nets `962,898`, routable nets `136,548`, fully routed nets `136,548`, routing errors `0`.
  Implemented resources: LUT `63,327/230,400 = 27.49%`, FF `57,218/460,800 = 12.42%`, BRAM Tile `137.5/312 = 44.07%`, URAM `76/96 = 79.17%`, DSP `973/1,728 = 56.31%`.
  Implemented vectorless power: total `6.217 W`, dynamic `5.498 W`, static `0.720 W`, PS static `0.102 W`, PL static `0.618 W`.
  Artifacts copied to `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_300_mem16_overlay/` and `hardware/pynq/hgtxr/`.
  Current judgment: `patchpar32` is now the best physically implemented full learned runtime-ROM baseline. Final latency targets remain open: Search `8.336 ms > 4 ms`; Track about `1.187 ms > 1 ms`.

- Added and measured the combined `patch32_dtok4` DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok2_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: combine the positive `PATCH_PAR=32` Conv/EventConv change with the positive `DENSE_TOKEN_PAR=4` controller/projection change while preserving the same full learned runtime-ROM, dispatcher prefetch-all4, shared ATTN/MLP, on-chip parameter ROM, and on-chip nonlinear ROM envelope.
  Combined CSim passed with Search output `[3, 3, 3, 3, 3, 3]`, Track output `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval max `2,342,872 cycles`, about `7.810 ms` at exact `300 MHz`. Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `2,031,010`, `mlp_head` `25,114`.
  Track-only HLS result: estimated clock `3.473 ns` / `287.94 MHz`, interval `336,327 cycles`, about `1.121 ms` at exact `300 MHz`. Path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `236,422`, `mlp_head` `6,682`.
  Hybrid 10% Search / 90% Track force-mode estimate: `536,981.5 cycles`, about `1.790 ms` at exact `300 MHz`, about `558.7 invocations/s`.
  Combined CSynth report: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
  Combined HLS resource estimate: BRAM_18K `331/624 = 53%`, DSP `1,258/1,728 = 72%`, FF `147,675/460,800 = 32%`, LUT `282,684/230,400 = 122%`, URAM `76/96 = 79%`.
  Latency improves versus the current `patchpar32` physical baseline HLS force-mode numbers: Search `2,500,952 -> 2,342,872 cycles`, Track `356,135 -> 336,327 cycles`, Hybrid `570,617 -> 536,981.5 cycles`.
  Package/IP export completed with `component.xml` and `export.zip`.
  Vivado route/bitgen completed for `hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_dtok4_300_mem16_overlay`: route errors `0`, fully routed nets `158,993/158,993`, bitgen pass.
  Final post-route physopt timing: WNS `-0.238 ns`, TNS `-549.019 ns`, WHS `0.000 ns`, THS `0.000 ns`, setup failing endpoints `4,620`. This meets the user-approved `WNS >= -0.500 ns` experiment floor but remains official negative-WNS timing fail.
  Implemented utilization: CLB LUT `73,832/230,400 = 32.05%`, CLB registers `69,126/460,800 = 15.00%`, Block RAM Tile `169.5/312 = 54.33%`, URAM `76/96 = 79.17%`, DSP `1,229/1,728 = 71.12%`.
  Implemented vectorless power: total on-chip `7.150 W`, dynamic `6.423 W`, device static `0.727 W`; mode-specific Search/Track power still requires switching activity or board measurement.
  Structural contract audit for the exact candidate now passes `16/16`: `hardware/generated/signoff/patch32_dtok4_300_contract_audit_2026_06_30.{json,md}`.
  Goal-status artifact now records the exact current status as `physical-structural-pass-latency-target-fail`: `hardware/generated/signoff/patch32_dtok4_300_goal_status_2026_06_30.{json,md}`.
  Current judgment: `patch32_dtok4` is now the best physical latency candidate under the user-approved negative-WNS experiment floor, but it is not official timing-clean signoff and still misses the final Search/Track latency targets.

- Wired the current `patch32_dtok4` physical candidate into the board-smoke validation path:
  Added PYNQ runner variant `par32-patch32-dtok4-300`, Search/Track/Hybrid 10:90 bundle packaging, result-validation presets, bundle-validation rules, remote-runner profiles, and board-latency gate profile-set.
  Generated Search, Track, and Hybrid 10:90 bundles under `hardware/generated/pynq/e2e_axis_dma_par32_patch32_dtok4_300_*_smoke_bundle*`.
  Bundle validations all passed with no errors:
  `hardware/generated/signoff/par32_patch32_dtok4_300_search_bundle_validation_2026_06_30.json`,
  `hardware/generated/signoff/par32_patch32_dtok4_300_track_bundle_validation_2026_06_30.json`, and
  `hardware/generated/signoff/par32_patch32_dtok4_300_hybrid_10_90_bundle_validation_2026_06_30.json`.
  Generated the dry-run remote execution plan `hardware/generated/signoff/par32_patch32_dtok4_300_board_latency_run_2026_06_30.{json,md}`; status is `dry-run` with Search, Track, and Hybrid 10:90 profiles.
  Generated the board latency gate `hardware/generated/signoff/par32_patch32_dtok4_300_board_latency_gate_2026_06_30.{json,md}`; status is `missing` because the physical board result JSONs are not present yet.
  Regenerated `hardware/generated/signoff/patch32_dtok4_300_goal_status_2026_06_30.{json,md}`; it now records board plumbing `pass` separately from the still-missing board measurements.

- Corrected the `patch32_dtok4` PYNQ board-smoke expected-output contract:
  The combined runtime bitstream contract is Search `[3, 3, 3, 3, 3, 3]` and Track `[217, 217, 217, 217, 217, 217]`.
  A Track-only force-mode CSim run observed `[215, 215, 215, 215, 215, 215]`, but that is not the board-smoke contract because it is not the combined runtime bitstream profile.
  Updated `hardware/tools/package_e2e_axis_dma_pynq_bundle.py`, `hardware/tools/validate_pynq_smoke_result.py`, `hardware/tools/validate_pynq_bundle_package.py`, and `hardware/pynq/hgtxr/run_e2e_axis_dma_hybrid_smoke.py`.
  Rebuilt the Search, Track, and Hybrid 10:90 bundles. Tar SHA256 values are Search `8ffded397fb82d2996e74623328a66aca4291225c29484ad8143144e1cdb36d8`, Track `095c86b477254dcb3a2f9912aae85bbedf1196d3348b948f558705ec34e448cc`, and Hybrid `c1d81829eb13a8276b518571f9325a8cc1d285e98a0b768a51a389200febcf4c`.
  Validation: py_compile passed for the four touched Python files; Search, Track, and Hybrid 10:90 bundle validations all passed; the regenerated Track bundle command now contains `--expect-out-raw 217 217 217 217 217 217`.
  Regenerated `hardware/generated/signoff/patch32_dtok4_300_goal_status_2026_06_30.{json,md}`; status remains `physical-structural-pass-latency-target-fail`.

- Added and measured the `patch32_mtok3_dtok4` midpoint DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Purpose: test whether `MLP_TOKEN_PAR=3` preserves most of the `mtok4` latency gain while reducing the force-mode HLS resource overflow.
  Combined CSim passed with Search `[3, 3, 3, 3, 3, 3]`, Track `[217, 217, 217, 217, 217, 217]`, runtime states `0/1`, TLAST correct, and strict immediate-start prefetch traces clean.
  Search-only HLS result: estimated clock `2.777 ns` / `360.10 MHz`, interval `2,147,147 cycles`, about `7.157 ms` at exact `300 MHz`. Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,301`, `controller_run` `1,835,266`, `mlp_head` `25,124`.
  Track-only HLS result: estimated clock `3.473 ns` / `287.94 MHz`, interval `321,651 cycles`, about `1.072 ms` at exact `300 MHz`. Path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,082`, `controller_run` `221,734`, `mlp_head` `6,688`.
  Hybrid 10% Search / 90% Track force-mode estimate: `504,200.6 cycles`, about `1.681 ms` at exact `300 MHz`, about `595.0 invocations/s`.
  Comparison versus current `patch32_dtok4`: Search II improves by `195,725 cycles` (`8.35%`), Track II improves by `14,676 cycles` (`4.36%`), and Hybrid II improves by `32,780.9 cycles`.
  Resource result: Search-only LUT `252,281/230,400 = 109%`, DSP `1,128/1,728 = 65%`; Track-only LUT `247,750/230,400 = 107%`, DSP `1,252/1,728 = 72%`.
  Decision: do not promote or route `mtok3_dtok4` as-is. It remains over the HLS LUT budget and Search still misses the `4 ms` target. Current physical baseline remains `patch32_dtok4`.

- Calculated the requested initial intervals for older baselines:
  Previous `par32_runtime_full_axi_mem16` at `200 MHz`: Track II `627,118 cycles`, about `3.135590 ms`; Search II `5,101,976 cycles`, about `25.509880 ms`.
  Previous `par32_runtime_mode_mem16` at `200 MHz`: Track II `76,151 cycles`, about `0.380755 ms`; Search II `594,841 cycles`, about `2.974205 ms`.
  Interpretation: `runtime_mode_mem16` remains a fast architectural reference, but it is not the current full learned runtime-ROM Transformer implementation.

- Added and measured the `patch32_mtok4_dtok4_aq2` attention-query-parallel DSE:
  Implementation: added `HGTXR_E2E_ATTN_QUERY_PAR` with default `1` and a two-query attention-core branch when set to `2`; added combined/Search-only/Track-only no-board profiles in `hardware/scripts/run/run_e2e_q4w8a_no_board.sh`.
  Search-only HLS result: interval max `1,709,784 cycles`, about `5.699 ms` at exact `300 MHz`; path `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,397,922`, `mlp_head` `25,114`; resources BRAM_18K `361/624 = 57%`, DSP `1,288/1,728 = 74%`, FF `133,107/460,800 = 28%`, LUT `264,616/230,400 = 114%`, URAM `92/96 = 95%`.
  Track-only HLS result: interval `268,711 cycles`, about `0.896 ms` at exact `300 MHz`; path `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `168,806`, `mlp_head` `6,682`; resources BRAM_18K `359/624 = 57%`, DSP `1,412/1,728 = 81%`, FF `145,179/460,800 = 31%`, LUT `260,389/230,400 = 113%`, URAM `71/96 = 73%`.
  Hybrid 10% Search / 90% Track estimate: II `412,818.3 cycles`, about `1.376 ms` at exact `300 MHz`, about `726.7 invocations/s`.
  Decision: `AQ2` is the fastest HLS latency point observed for the current full learned runtime-ROM family, but it is not a physical promotion candidate yet because Search still misses `4 ms` and LUT is over the ZCU104 budget in both force-mode reports.

- Followed up the `AQ2` point with MLP-side probes:
  Added `AQ2 + skip hidden store`; Search-only CSynth produced the same result as `AQ2`: top II `1,709,784 cycles`, MLP unit `211,151 cycles`, resources BRAM_18K `361`, DSP `1,288`, LUT `264,616`, URAM `92`. Decision: reject as no-op for latency/resource.
  Added `AQ2 + HGTXR_E2E_MLP_FUSED_W2_HP_PAR=4`; Search-only CSynth again kept top II at `1,709,784 cycles`, controller `1,397,922 cycles`, and MLP `211,151 cycles`, but increased DSP to `1,544/1,728 = 89%` and LUT to `275,267/230,400 = 119%`. Decision: reject because it spends resources without moving the controlling loop.
  Added `AQ2 + HGTXR_E2E_MLP_TOKEN_PAR=8` as a Search-only lower-bound probe. CSynth was interrupted after HLS IR expansion made it too large for fast DSE: `csynth_design_size.rpt` showed `354,635` instructions after `Array/Struct`, with `hgtxr_e2e_mlp_unit<0>` at `276,469` instructions. No latency/resource signoff is inferred from this incomplete run.
  Current judgment: Search `4 ms` needs a structural MLP/body or memory-banking refactor, not hidden-store removal, W2 hidden-lane parallelism, or brute-force `MLP_TOKEN_PAR=8`.

- Added and measured two AQ2 W2-banking Search-only probes:
  Full hidden-lane W2 bank cache profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hbank_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only` completed CSynth. Search II improved to `1,636,068 cycles`, about `5.454 ms` at 300 MHz; controller `1,324,206 cycles`; MLP unit `192,722 cycles`. Reject for promotion because BRAM_18K is `827/624 = 132%`, LUT `281,384/230,400 = 122%`, DSP `1,544/1,728 = 89%`, URAM `92/96 = 95%`.
  Hgroup-local W2 bank cache profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only` completed CSynth. Search II improved to `1,625,268 cycles`, about `5.418 ms` at 300 MHz; controller `1,313,406 cycles`; MLP unit `190,022 cycles`; path `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,313,406`, `mlp_head` `25,114`.
  Hgroup-local W2 resources: BRAM_18K `315/624 = 50%`, DSP `1,544/1,728 = 89%`, FF `152,242/460,800 = 33%`, LUT `273,839/230,400 = 118%`, URAM `92/96 = 95%`.
  Current judgment: `w2hgroup` is the fastest Search-only HLS point observed so far for the current full learned runtime-ROM family, but it is not a promotion candidate yet because Search remains `5.418 ms > 4 ms` and LUT remains over ZCU104. W2 banking helps, but the next step must reduce full learned operator/body work or lower LUT pressure rather than only bank W2.

- Added and measured an `AQ4 + hgroup-local W2 bank cache` Search-only probe:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Purpose: test whether raising `HGTXR_E2E_ATTN_QUERY_PAR` from `2` to `4` on top of the fastest `w2hgroup` Search-only point can reduce the remaining controller/body latency.
  CSynth completed. Search II improved from `1,625,268` to `1,549,044 cycles`, about `5.163 ms` at 300 MHz. Controller improved from `1,313,406` to `1,237,182 cycles`. ATTN unit improved from `131,388` to `112,332 cycles`; MLP unit stayed at `190,022 cycles`.
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,237,182`, `mlp_head` `25,114`.
  Resource result is not viable: BRAM_18K `315/624 = 50%`, DSP `1,672/1,728 = 96%`, FF `163,522/460,800 = 35%`, LUT `288,995/230,400 = 125%`, URAM `124/96 = 129%`.
  Current judgment: `AQ4+w2hgroup` is the fastest Search-only HLS point measured so far, but it is a negative resource DSE and must not be promoted. Increasing attention query parallelism alone improves cycles but violates URAM/LUT and still misses Search `4 ms`.

- Added and measured an `AQ4 + hgroup-local W2 bank cache + Q/K/V/ATTN BRAM` Search-only probe:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Purpose: test whether the `AQ4+w2hgroup` URAM overflow was caused primarily by Q/K/V/ATTN buffers staying in URAM under the base compile flags.
  CSynth completed at the 300 MHz HLS target. Search II stayed at `1,549,044 cycles`, about `5.163 ms` at exact 300 MHz. Controller stayed at `1,237,182 cycles`; attention unit stayed at `112,332 cycles`; MLP unit stayed at `190,022 cycles`.
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,237,182`, `mlp_head` `25,114`.
  Resource shift versus raw `AQ4+w2hgroup`: BRAM_18K `315 -> 443`, URAM `124 -> 28`, while DSP `1,672`, FF `163,522`, and LUT `288,995` are effectively unchanged.
  Resource result: BRAM_18K `443/624 = 70%`, DSP `1,672/1,728 = 96%`, FF `163,522/460,800 = 35%`, LUT `288,995/230,400 = 125%`, URAM `28/96 = 29%`.
  Current judgment: qkv-BRAM is a useful storage-mapping correction because it fixes AQ4 URAM overflow without losing cycles, but it is still not a ZCU104 promotion candidate. The remaining blockers are LUT `125%`, DSP `96%`, and Search `5.163 ms > 4 ms`.

- Ran Vivado no-board implementation for the same qkv-BRAM Search-only probe:
  Command profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  Project/output root: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_overlay`.
  Overlay output: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only_overlay`.
  Bitstream and handoff copied to PYNQ as `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_300_mem16_search_only.{bit,hwh}`.
  Route status: all `207,407` routable nets fully routed; route errors `0`.
  Final postroute-physopt timing: WNS `-0.527 ns`, TNS `-2247.483 ns`, WHS `0.006 ns`, THS `0.000 ns`, failing setup endpoints `17,645` at `300.030 MHz`.
  The run misses the user-approved `WNS >= -0.500 ns` experimental floor by `27 ps`, so it is bitgen-complete but not a passing physical candidate.
  Implemented utilization: CLB LUT `95,920/230,400 = 41.63%`, CLB registers `98,554/460,800 = 21.39%`, Block RAM Tile `227/312 = 72.76%`, URAM `28/96 = 29.17%`, DSP `1,641/1,728 = 94.97%`.
  Implemented vectorless power: total `7.723 W`, dynamic `6.996 W`, device static `0.727 W`. Dynamic hierarchy: `hgtxr_e2e_axis_top_0` `4.089 W`, `psu` `2.681 W`, `axi_mem` `0.183 W`, `axi_ctrl` `0.020 W`, `axi_dma_out` `0.018 W`, `axi_dma_in` `0.005 W`.
  Worst setup path is MLP-side, from `mul_14s_15s_29_5_1_U4940/buff2_reg/DSP_OUTPUT_INST/CLK` to `add_ln3247_312_reg_33461_reg[25]/D`; data delay `3.649 ns`, route delay `3.077 ns` (`84.323%`).
  Next DSE should target the MLP DSP/output-to-adder critical path, DSP input/output pipelining, and fanout/placement pressure. Storage remapping alone has already fixed URAM but not timing or Search latency.

- Added, implemented, and documented the `qkvbram_tail2` Search-only physical closure probe:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  HLS project: `hardware/generated/hgtxr_e2e_axis_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_no_board`.
  HLS Search-only result: top II `1,549,428 cycles`, about `5.165 ms` at exact `300 MHz`; path `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,237,566`, `mlp_head` `25,114`.
  HLS resources: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,220/460,800 = 36%`, LUT `312,155/230,400 = 135%`, URAM `28/96 = 29%`.
  Vivado project/output root: `hardware/generated/build/vivado/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_overlay`.
  Overlay output: `hardware/generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only_overlay`.
  Bitstream and handoff copied to PYNQ as `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_par32_rt_attntok2_attnbram_patch32_mtok4_dtok4_aq4_hp4_w2hgroup_qkvbram_tail2_300_mem16_search_only.{bit,hwh}`.
  Route status: all `208,988` routable nets fully routed; route errors `0`; bitgen completed successfully.
  Final postroute-physopt timing: WNS `-0.029 ns`, TNS `-0.365 ns`, WHS `0.002 ns`, THS `0.000 ns`, setup failing endpoints `31`. This passes the user-approved `WNS >= -0.500 ns` floor but remains negative-WNS.
  Implemented utilization: CLB LUT `97,759/230,400 = 42.43%`, CLB registers `99,826/460,800 = 21.66%`, Block RAM Tile `227/312 = 72.76%`, URAM `28/96 = 29.17%`, DSP `1,553/1,728 = 89.87%`.
  Implemented vectorless power: total `8.031 W`, dynamic `7.302 W`, device static `0.729 W`; major dynamic buckets are `hgtxr_e2e_axis_top_0` `4.398 W`, `psu` `2.678 W`, `axi_mem` `0.185 W`, `axi_ctrl` `0.018 W`, `axi_dma_out` `0.018 W`, `axi_dma_in` `0.005 W`.
  Decision: `qkvbram_tail2` is now the best Search-only physical closure evidence in the `AQ4+w2hgroup+qkvbram` branch. It trades a negligible HLS Search II regression (`+384 cycles`) for much better physical DSP/timing (`1,641 -> 1,553` implemented DSP and WNS `-0.527 ns -> -0.029 ns`). It is not a final goal solution because Search remains about `5.165 ms > 4 ms` and no corresponding Track/hybrid physical pair has been generated for this exact Search-only branch.

- Followed up `qkvbram_tail2` with two Search-only DSE probes:
  `PATCH_PAR=64` profile passed CSim, but CSynth regressed Search II to `1,561,716 cycles` (`5.206 ms` at 300 MHz). The regression comes from `conv_patch_embedding` increasing from `258,052` to `270,340 cycles` under frame-memory port pressure. Decision: reject.
  `HGTXR_E2E_SOFTMAX_EXP_ROM_PARTITION=1` profile passed CSim and improved Search II to `1,512,564 cycles` (`5.042 ms` at 300 MHz). The attention core improved from `31,543` to `22,327 cycles`, and the softmax exp loop instance dropped from `260` to `68 cycles`.
  `exppart` HLS resources remain non-promotable as a physical candidate without more resource work: BRAM_18K `443/624 = 70%`, DSP `1,584/1,728 = 91%`, FF `166,307/460,800 = 36%`, LUT `312,706/230,400 = 135%`, URAM `28/96 = 29%`.
  Current judgment: `exppart` is the fastest Search-only HLS point in the `qkvbram_tail2` physical-relief branch, but it still misses Search `4 ms` and should not be routed until the LUT/body-cycle issue is addressed.

- Followed up `exppart` with an MLP hidden-store removal probe:
  Added `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_FUSED_W2_SKIP_HIDDEN_STORE=1` as profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_skiph_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with strict prefetch-immediate trace and vector comparison.
  CSynth produced the same top result as `exppart`: Search II `1,512,564 cycles` (`5.042 ms` at 300 MHz), controller `1,200,702 cycles`, MLP unit `190,022 cycles`, and unchanged HLS resources BRAM_18K `443`, DSP `1,584`, FF `166,307`, LUT `312,706`, URAM `28`.
  Decision: reject `exppart_skiph` as a no-op. The controlling work is still MLP/body and controller-loop latency, especially MLP unit `190,022 cycles` per layer with dominant loop `VITIS_LOOP_3091_2` at `150,208 cycles`; hidden-store removal does not move the HLS schedule in this branch.

- Followed up `exppart` with a fused MLP W1 input-channel parallelism probe:
  Added `HGTXR_E2E_MLP_W1_C_PAR` with default `1` and a targeted fused token-parallel MLP branch for `HGTXR_E2E_MLP_W1_C_PAR=2`.
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with strict prefetch-immediate trace and vector comparison.
  CSynth improved Search II from `1,512,564` to `1,368,180 cycles` (`4.561 ms` at 300 MHz). Controller improved to `1,056,318 cycles`; MLP unit improved from `190,022` to `153,926 cycles`; the dominant fused W1 loop in this build is `VITIS_LOOP_3097_2` at `114,112 cycles`.
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,056,318`, `mlp_head` `25,114`.
  Resources are non-promotable as-is: BRAM_18K `449/624 = 71%`, DSP `1,704/1,728 = 98%`, FF `170,564/460,800 = 37%`, LUT `320,789/230,400 = 139%`, URAM `28/96 = 29%`.
  Decision: `w1c2` is the fastest Search-only HLS latency point in the current full learned runtime-ROM family, but it still misses Search `4 ms` by `168,180 cycles` and nearly exhausts DSP. The next useful DSE should preserve the W1 cycle gain while shifting selected multiply or small-memory pressure away from DSP/LUT-critical paths before any physical route attempt.

- Followed up `w1c2` with a tail-lane fabric pressure-relief probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail4_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  This keeps `HGTXR_E2E_MLP_W1_C_PAR=2` and raises `HGTXR_E2E_CORE_FABRIC_TAIL_LANES` from `2` to `4`.
  CSim passed for Search with strict prefetch-immediate trace, TLAST, runtime state, and vector comparison.
  CSynth completed at the 300 MHz HLS target with estimated clock `2.800 ns`.
  Search II stayed at `1,368,180 cycles` (`4.561 ms` at exact 300 MHz), so the extra fabric tail-lane mapping gives no latency gain.
  Path remains unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,056,318`, `mlp_head` `25,114`.
  MLP unit remains `153,926 cycles`, and the dominant fused W1 loop remains `VITIS_LOOP_3097_2` at `114,112 cycles`.
  Resource tradeoff versus `tail2+w1c2`: BRAM_18K `449 -> 449`, DSP `1,704 -> 1,608`, FF `170,564 -> 172,598`, LUT `320,789 -> 346,133`, URAM `28 -> 28`.
  Decision: reject as a promotion candidate. It relieves DSP by `96` but makes LUT pressure materially worse (`139% -> 150%`) while leaving Search latency unchanged. The next DSE must reduce the learned body work or reduce LUT-heavy multiply/control pressure, not only move more tail lanes to fabric.

- Followed up `w1c2` with a larger W1 channel-grouping latency probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq4_qkvbram_tail2_exppart_w1c3_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  This keeps the same `qkvbram_tail2 + exppart` base and raises `HGTXR_E2E_MLP_W1_C_PAR` from `2` to `3`.
  CSim passed for Search with strict prefetch-immediate trace, TLAST, runtime state, and vector comparison.
  CSynth completed at the 300 MHz HLS target with estimated clock `2.800 ns`.
  Search II regressed from `1,368,180` to `1,417,332 cycles` (`4.724 ms` at exact 300 MHz), so the larger W1 grouping does not close the `4 ms` gap.
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,105,470`, `mlp_head` `25,114`.
  MLP unit regressed from `153,926` to `166,214 cycles`, and the dominant fused W1 loop increased from `114,112` to `126,400 cycles`.
  Resources are also worse than `w1c2`: BRAM_18K `449`, DSP `1,824/1,728 = 105%`, FF `181,636`, LUT `330,657/230,400 = 143%`, URAM `28`.
  Decision: reject. `W1_C_PAR=3` is a negative latency/resource DSE. Larger W1 channel grouping should not be pursued without restructuring the accumulation schedule or reducing multiplier pressure first.

- Followed up `w1c2` with an `AQ8` attention-query parallelism latency probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  This keeps `qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=2` and raises `HGTXR_E2E_ATTN_QUERY_PAR` from `4` to `8`.
  CSim passed for Search with strict prefetch-immediate trace, TLAST, runtime state, and vector comparison.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.789 ns`.
  Search II improved from `1,368,180` to `1,324,308 cycles` (`4.414 ms` at exact 300 MHz), so this is the fastest Search-only HLS latency point currently measured in the full learned runtime-ROM family.
  Remaining Search `4 ms` gap: `124,308 cycles` (`0.414 ms`).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,446`, `mlp_head` `25,114`.
  Controller details: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`. Attention improved versus `w1c2` (`103,212 -> 92,244`), and attention core improved (`22,327 -> 11,263`), but MLP stayed unchanged.
  Resources are non-promotable: BRAM_18K `513/624 = 82%`, DSP `1,944/1,728 = 112%`, FF `198,021/460,800 = 42%`, LUT `359,425/230,400 = 156%`, URAM `28/96 = 29%`.
  Decision: reject as a route/promotion candidate despite the HLS latency improvement. AQ8 proves attention-query parallelism still has cycle headroom, but the straightforward parallel implementation violates DSP and LUT capacity. The next useful DSE must either time-multiplex/resource-share the AQ8 gain or attack the unchanged MLP/body and fixed front-end latency directly.

- Followed up `AQ8+w1c2` with a tail8 fabric pressure-relief probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail8_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  This keeps the fastest observed AQ8/w1c2 schedule and raises `HGTXR_E2E_CORE_FABRIC_TAIL_LANES` from `2` to `8`.
  CSim passed for Search with strict prefetch-immediate trace, TLAST, runtime state, and vector comparison.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.789 ns`.
  Search II remained `1,324,308 cycles` (`4.414 ms` at exact 300 MHz), so tail8 preserves the current fastest HLS Search latency but does not reduce the remaining `0.414 ms` gap to the `4 ms` target.
  Path and controller cycle breakdown remained unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,446`, `mlp_head` `25,114`; dispatcher `6,928-6,933`, attention unit `92,244`, MLP unit `153,926`.
  Resource tradeoff versus `AQ8+tail2+w1c2`: BRAM_18K `513 -> 513`, DSP `1,944 -> 1,608`, FF `198,021 -> 205,197`, LUT `359,425 -> 448,561`, URAM `28 -> 28`.
  Decision: reject as a promotion candidate. Tail8 solves the immediate DSP over-capacity problem (`112% -> 93%`) but makes LUT capacity much worse (`156% -> 194%`) with no latency improvement. The next step should not keep pushing tail-lane fabric mapping; it should reduce LUT-heavy arithmetic/control or restructure AQ8 sharing while also reducing MLP/body cycles.

- Tested an `AQ6` attention-query parallelism midpoint:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq6_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with strict prefetch-immediate trace, TLAST, runtime state, and vector comparison.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.777 ns`.
  Search II regressed to `1,484,260 cycles` (`4.948 ms` at exact 300 MHz), so AQ6 is slower than both `AQ4+w1c2` (`1,368,180 cycles`) and `AQ8+w1c2` (`1,324,308 cycles`).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,172,398`, `mlp_head` `25,114`.
  Controller details: dispatcher prefetch `6,928-6,933 cycles`, attention unit `130,054-132,232 cycles`, MLP unit `153,926 cycles`. The attention unit worsens versus AQ4 and AQ8, while MLP remains unchanged.
  Resources are non-promotable: BRAM_18K `481/624 = 77%`, DSP `1,824/1,728 = 105%`, FF `194,951/460,800 = 42%`, LUT `348,116/230,400 = 151%`, URAM `28/96 = 29%`.
  Decision: reject. AQ6 is a negative scheduling/resource DSE, not a useful Pareto midpoint. Future attention-query parallelism should use power-of-two schedules or redesign the attention banking before revisiting non-power-of-two query lanes.

- Tested an `AQ8+hpar8+w1c2` MLP W2 hidden-lane parallelism probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  This keeps `AQ8`, `qkvbram_tail2`, `exppart`, `w1c2`, and `mtok4/dtok4`, while raising `HGTXR_E2E_MLP_FUSED_W2_HP_PAR` from `4` to `8` and `HGTXR_E2E_HEAD_PAR` from `4` to `8`.
  CSim passed for Search with strict prefetch-immediate trace, TLAST, runtime state, and vector comparison.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.789 ns`.
  Search II improved from the previous fastest `1,324,308 cycles` to `1,288,836 cycles` (`4.296 ms` at exact 300 MHz). Remaining Search `4 ms` gap is `88,836 cycles` (`0.296 ms`).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `977,118`, `mlp_head` `24,970`.
  Controller details: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `145,094 cycles`.
  Main gain: MLP unit improves `153,926 -> 145,094 cycles`, saving `8,832 cycles` per runtime block and `35,328 cycles` across the four Search blocks. Head parallelism saves only `144` top-level cycles.
  Resources are far beyond promotion range: BRAM_18K `513/624 = 82%`, DSP `2,424/1,728 = 140%`, FF `211,693/460,800 = 45%`, LUT `385,749/230,400 = 167%`, URAM `28/96 = 29%`.
  Decision: keep as the fastest Search-only HLS latency point observed so far, but reject as a route/promotion candidate. Direct `hpar8` proves real MLP W2 cycle headroom, but the next useful design must time-multiplex or resource-share that W2 parallelism because brute-force hidden-lane parallelism is not ZCU104-fit.

- Added the matching `AQ8+hpar8+w1c2` Track-only force-mode measurement:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_track_only`.
  CSim passed with Track runtime state, TLAST, vector comparison, and strict prefetch trace.
  CSynth completed at the 300 MHz HLS target. The report estimates clock `3.473 ns`, so its printed absolute time is `0.798 ms`; normalized to the requested exact `300 MHz`, Track latency is `229,782 cycles` / `0.765940 ms` and Track II is `229,783 cycles` / `0.765943 ms`.
  Track path: `axis_read_frame` `4,109`, `event_conv_patch_embedding` `86,020`, `global_buffer_load` `3,076`, `controller_run` `130,022`, `mlp_head` `6,538`.
  Track controller details: dispatcher prefetch `6,926 cycles`, attention unit `21,801 cycles`, MLP unit `36,276 cycles`, dispatch loop `13,858 cycles`, body loop `116,162 cycles`.
  Track resources are also non-promotable: BRAM_18K `513/624 = 82%`, DSP `2,420/1,728 = 140%`, FF `210,928/460,800 = 45%`, LUT `379,171/230,400 = 164%`, URAM `7/96 = 7%`.
  Matching pair result at exact `300 MHz`: Search max latency `1,288,835 cycles` / `4.296117 ms`, Search max II `1,288,836 cycles` / `4.296120 ms`, Track latency `0.765940 ms`, Track II `0.765943 ms`, 10% Search / 90% Track mean latency `1.118958 ms`, mean II `1.118961 ms`, and throughput `893.69 invocations/s`.
  Decision: the `hpar8` pair is the fastest measured HLS Search/Track pair so far and Track meets the `1 ms` target, but it is not a goal candidate because Search still misses `4 ms` and the direct hidden-lane parallelism is far beyond ZCU104 DSP/LUT capacity.

- Rechecked the AQ2 Search/Track requested metric package:
  Confirmed that `hardware/docs/track/AQ2_SEARCH_TRACK_METRIC_CHECKLIST_2026_07_01.md` covers DMA ideal/effective bandwidth, mean/median/min/max/P95/P99 latency, initial interval, GOPS, AXIS width, HLS major-block resources, implemented Vivado top/hierarchy resources, and implemented Vivado power breakdown.
  Current AQ2 headline values remain Search latency `5.699210-5.699277 ms`, Search II `1,709,784 cycles`, Track latency `0.895700 ms`, Track II `268,711 cycles`, Search throughput `43.05 GOPS`, Track throughput `33.81 GOPS`, Search implemented power `7.032 W`, and Track implemented power `6.777 W`.
  Remaining gaps are unchanged: AQ2 has no board-runtime histogram, no board DMA counter data, no external DDR/sensor I/O rail power, and no internal Conv/ATTN/MLP/Head dynamic-power split.

- Tested `AQ8 + qkvbram_tail2 + exppart + HGTXR_E2E_MLP_W1_C_PAR=4` as a wider W1 grouping probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target. Search II is `1,324,308 cycles` (`4.414 ms` at exact 300 MHz), unchanged versus `AQ8+w1c2`.
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,012,426-1,012,446`, `mlp_head` `25,114`.
  Controller details: dispatcher prefetch `6,928-6,933 cycles`, attention unit `92,244 cycles`, MLP unit `153,926 cycles`.
  Resources are non-promotable: BRAM_18K `513/624 = 82%`, DSP `2,184/1,728 = 126%`, FF `212,462/460,800 = 46%`, LUT `375,182/230,400 = 162%`, URAM `28/96 = 29%`.
  Decision: reject. `w1c4` is functionally safe but does not reduce the controlling MLP/body schedule and worsens DSP/LUT pressure.

- Tested explicit W1 cache banking on top of `w1c4`:
  Added default-off macro `HGTXR_E2E_W1_WEIGHT_CACHE_BANKS` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c4_w1bank16_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target with estimated Fmax `358.55 MHz`.
  Search II remains `1,324,308 cycles` (`4.414 ms` at exact 300 MHz), unchanged versus `w1c4`; path and controller breakdown are unchanged.
  The targeted W1 loop remains final II `2` with latency `106 cycles`, and the scheduler still reports a limited-port load on `w1_weight_cache`.
  Resources regress: BRAM_18K `707/624 = 113%`, DSP `2,184/1,728 = 126%`, FF `212,833/460,800 = 46%`, LUT `376,481/230,400 = 163%`, URAM `28/96 = 29%`.
  Decision: reject. The current cyclic W1 cache partition does not change the HLS memory-port bottleneck and consumes far more BRAM, so the next W1 direction needs a different storage layout or schedule rather than simply increasing the bank factor.

- Tested `AQ8+hpar8+w1c2` with narrower accumulators (`HGTXR_ACC_W=24`, `HGTXR_ACC_I=10`) as a pressure-relief probe:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.789 ns`.
  Search II improves slightly from direct `hpar8` `1,288,836` to `1,284,676 cycles` (`4.282 ms` at exact 300 MHz). Remaining Search `4 ms` gap is `84,676 cycles` (`0.282 ms`).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `972,938-972,958`, `mlp_head` `24,970`.
  Controller details: dispatcher prefetch `6,928-6,933 cycles`, attention unit `91,652 cycles`, MLP unit `144,646 cycles`.
  Resources improve versus direct `hpar8` but remain non-promotable: BRAM_18K `513/624 = 82%`, DSP `2,410/1,728 = 139%`, FF `178,153/460,800 = 38%`, LUT `358,589/230,400 = 155%`, URAM `28/96 = 29%`.
  Decision: reject as a route/promotion candidate. `ACC24` is now the fastest Search-only HLS point and does reduce LUT/FF pressure, but it still misses Search `4 ms` and remains far above ZCU104 DSP/LUT capacity.

- Tested `AQ8+hpar8+w1c2+ACC24` with forced wide-DSP multiply disabled:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_nowide_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.789 ns`.
  Search II is unchanged versus `hpar8_acc24`: `1,284,676 cycles` (`4.282 ms` at exact 300 MHz).
  Path and controller breakdown are unchanged: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `972,938-972,958`, `mlp_head` `24,970`; dispatcher `6,928-6,933`, attention `91,652`, MLP `144,646`.
  Resources regress versus `hpar8_acc24`: BRAM_18K remains `513/624 = 82%`, DSP remains `2,410/1,728 = 139%`, FF changes to `178,245/460,800 = 38%`, LUT worsens to `379,140/230,400 = 164%`, URAM remains `28/96 = 29%`.
  Decision: reject. Disabling the forced wide-DSP multiply path does not reduce DSP usage and increases LUT pressure, so the next useful DSE should target controller/global-buffer memory-port pressure or a real time-multiplexed/shared datapath.

- Tested `AQ8+hpar8+w1c2+ACC24` with token-dimension cyclic banking:
  Added default-off macro `HGTXR_E2E_TOKEN_BANK_PAR` and profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_tokbank4_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  The macro cyclically partitions `gb.tokens`, `gb.norm`, `gb.q`, `gb.k`, `gb.v`, `gb.attn`, `gb.hidden`, and the top-level `tokens` buffer along the token dimension.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.800 ns` and estimated Fmax `357.14 MHz`.
  Search II regressed versus `hpar8_acc24`: `1,284,676 -> 1,294,276 cycles` (`4.282 -> 4.314 ms` at exact 300 MHz).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `982,538-982,558`, `mlp_head` `24,970`.
  Controller details: dispatcher prefetch `6,928-6,933 cycles`, attention unit `94,052 cycles`, MLP unit `144,646 cycles`.
  Resources regress versus `hpar8_acc24`: BRAM_18K `513 -> 545`, DSP unchanged at `2,410`, FF `178,153 -> 176,840`, LUT `358,589 -> 365,042`, URAM unchanged at `28`.
  Scheduler evidence still shows II=2 memory-port violations in the attention score loop and the MLP output loop, so the token-dimension banks do not remove the current controlling bottleneck.
  Decision: reject. `tokbank4` is functionally safe but slower, less memory-efficient, and still far beyond ZCU104 DSP/LUT capacity.

- Rechecked the `patch32_mtok3_dtok4` combined runtime profile after the force-mode DSE:
  Profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar4_mlpfusebank_hpar2_mtok3_dtok4_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16`.
  Combined CSim passed for Search `[3, 3, 3, 3, 3, 3]` and Track `[217, 217, 217, 217, 217, 217]`, with runtime states `0/1`, TLAST correct, vector comparison pass, and strict prefetch traces clean.
  Combined CSynth completed with estimated clock `2.777 ns`, top latency `99,928-15,280,489 cycles`, top interval `99,929-15,280,490 cycles`, and exact-300MHz max latency `50.935 ms`.
  Combined resource estimate: BRAM_18K `347/624 = 55%`, DSP `1,354/1,728 = 78%`, FF `175,489/460,800 = 38%`, LUT `320,682/230,400 = 139%`, URAM `76/96 = 79%`.
  Decision: keep rejected. The force-mode Search/Track latency improved versus `patch32_dtok4`, but the combined top envelope and LUT estimate are not promotable for route as-is.

- Tested `AQ8+hpar8+w1c2+ACC24` with wider patch embedding parallelism:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar8_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar64_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  The only intended latency change versus the previous fastest Search-only point was `HGTXR_E2E_PATCH_PAR=32 -> 64`.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.789 ns`.
  Search II regressed versus `hpar8_acc24`: `1,284,676 -> 1,296,964 cycles` (`4.282 -> 4.323 ms` at exact 300 MHz).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `270,340`, `global_buffer_load` `12,292`, `controller_run` `972,938-972,958`, `mlp_head` `24,970`.
  Resource estimate remains non-promotable: BRAM_18K `513/624 = 82%`, DSP `2,408/1,728 = 139%`, FF `179,024/460,800 = 38%`, LUT `358,606/230,400 = 155%`, URAM `28/96 = 29%`.
  Decision: reject. The wider patch parallelism hit a memory-port II=2 bottleneck in the patch embedding loop, so Conv/Patch latency increased from `258,052` to `270,340 cycles` instead of shrinking.

- Tested `AQ8+hpar8+w1c2+ACC24` with direct MLP hidden-lane parallelism raised to hpar16:
  Added profile `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_lutrom_shareunit_headpar8_mlpfusebank_hpar16_acc24_w2hgroup_mtok4_dtok4_aq8_qkvbram_tail2_exppart_w1c2_attntok2_attnbram_nocache_patchpar32_tokenloop_noobs_normbram_dsppipe4_300_mem16_search_only`.
  CSim passed for Search with runtime state, TLAST, vector comparison, and strict prefetch trace `block_pairs=4 violations=0 not_ready=0 immediate_gaps=0`.
  CSynth completed at the 300 MHz HLS target with estimated top clock `2.795 ns`.
  Search II regressed versus `hpar8_acc24`: `1,284,676 -> 2,229,316 cycles` (`4.282 -> 7.431 ms` at exact 300 MHz).
  Path: `axis_read_frame` `16,386`, `conv_patch_embedding` `258,052`, `global_buffer_load` `12,292`, `controller_run` `1,917,578-1,917,598`, `mlp_head` `24,970`.
  Controller details: dispatcher prefetch `6,928-6,933`, attention unit `91,652`, MLP unit `380,806`, dispatch loop `27,724-27,744`, body loop `1,889,852`.
  Root cause: HLS kept `kGeluRom` as `ROM_1P_LUTRAM`; the MLP activation loop `VITIS_LOOP_3279_18_VITIS_LOOP_3281_19` ended with final II `64` due to limited memory ports.
  Resource estimate is non-promotable and worse than hpar8: BRAM_18K `513/624 = 82%`, DSP `3,370/1,728 = 195%`, FF `209,907/460,800 = 45%`, LUT `401,685/230,400 = 174%`, URAM `28/96 = 29%`.
  Decision: reject. Direct hidden-lane replication is not the next viable path; the next Search optimization should replicate/partition the GELU ROM for parallel activation reads or redesign the W2 path as a shared/time-multiplexed datapath.

- Refreshed the AQ2 Search/Track requested metric report:
  Added `hardware/docs/track/AQ2_REQUESTED_SEARCH_TRACK_METRICS_2026_07_01.md`.
  Re-read the AQ2 Search-only and Track-only HLS `hgtxr_e2e_axis_top_csynth.rpt` and `hgtxr_e2e_controller_run_csynth.rpt` reports, plus the AQ2 implemented Vivado utilization, timing, route-status, and power reports.
  Current AQ2 Search is latency `1,709,763-1,709,783 cycles` / `5.699210-5.699277 ms` at 300 MHz, initiation interval `1,709,784 cycles`, effective wire DMA bandwidth `0.0920 GB/s`, throughput `43.05 GOPS`, and implemented total power `7.032 W`.
  Current AQ2 Track is latency `268,710 cycles` / `0.895700 ms` at 300 MHz, initiation interval `268,711 cycles`, effective wire DMA bandwidth `0.1465 GB/s`, throughput `33.81 GOPS`, and implemented total power `6.777 W`.
  Final implemented timing is Search WNS `0.000 ns` and Track WNS `-0.017 ns`; Track is inside the user-approved `-0.500 ns` experiment tolerance but is not strict timing-clean.
  Remaining AQ2 measurement gaps are board-runtime latency histograms, measured DMA counters, SAIF/VCD activity power, external DDR/sensor I/O rails, and internal Conv/ATTN/MLP/Head dynamic-power split.

- 2026-07-05 commit-preparation documentation pass:
  Added `hardware/docs/track/SESSION_PROGRESS_AND_CONVERSATION_2026_07_05.md`
  to consolidate the current work state and conversation path before committing.
  Rechecked git topology: repository root, `software/`, and `hardware/` all
  resolve to `/home/kjm26/project/PRJXR/XR-VIT/HGTXR`; current branch is
  `kjm26/feat/capture-hgtxr-current-state`. The commit candidate includes HLS
  learned/runtime-mode implementation changes, PYNQ runtime/validation tooling,
  Vivado/HLS run-script extensions, AQ2/eight-question/full-learned planning
  docs, and many PYNQ `.bit/.hwh` handoff artifacts. Two zero-byte root scratch
  files, `1838` and `384`, were identified as accidental artifacts and removed
  before staging because they were not meaningful hardware evidence.
