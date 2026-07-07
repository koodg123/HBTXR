# Changelog

## 0.1.117 - 2026-06-11

- Added `hardware/tools/write_final_unblock_closeout_packet.py`.
- Added `hardware/tests/test_write_final_unblock_closeout_packet.py`.
- Integrated `final-unblock-closeout-packet` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the closeout tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Added closeout packet JSON/Markdown to the final evidence manifest required artifact set.
- Final evidence manifest now requires `38/38` artifacts and checks `14` consistency invariants, including closeout packet readiness and no-side-effect safety.
- Regenerated final signoff evidence twice to settle the new closeout/manifest cycle.
- Current closeout packet is `ready-for-operator-unblock`; current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.116 - 2026-06-11

- Strengthened `hardware/tools/write_final_unblock_commands.py`.
- Final unblock command card now includes U0 `Dry-run final blocker closure readiness`.
- U0 provides no-side-effect `check_final_blocker_closure_readiness.py` commands for current evidence, C3b candidate plus exact XR-VITs, and C3b candidate plus XR_Accel replacement approval.
- Updated `hardware/tests/test_write_final_unblock_commands.py`.
- Regenerated final signoff evidence; current command card sections are `U0`, `U1`, `U2`, `U4`, `U3`.
- Current bundle validation remains `pass` with `57/57` checks, final evidence manifest remains `pass` with `36/36` required artifacts and `12` consistency checks.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.115 - 2026-06-11

- Strengthened `hardware/tools/validate_final_signoff_bundle.py`.
- Final signoff bundle validation now checks the final evidence manifest contract carried by both requirements trace and operator handoff.
- Added bundle checks for contract presence, pass status, trace/handoff match, required artifact completeness, consistency-count minimum, failed consistency checks, and no-write safety flags.
- Raised `hardware/tools/write_final_evidence_manifest.py` final-bundle pass-count invariant from `>=45` to `>=57`.
- Updated final bundle, evidence manifest, and completion audit tests.
- Regenerated final signoff evidence twice to settle the bundle/manifest/trace cycle; current bundle validation is `pass` with `57/57` checks and `fail_count=0`.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.114 - 2026-06-11

- Strengthened `hardware/tools/write_final_operator_handoff.py`.
- Operator handoff now includes `final_evidence_manifest_contract` and a Markdown `Final Evidence Manifest Contract` section.
- Strengthened `hardware/tools/validate_final_operator_handoff.py` with 8 manifest-contract checks covering presence, pass status, required artifact completeness, consistency count, failed consistency checks, and no-write safety.
- Raised `hardware/tools/write_final_evidence_manifest.py` handoff-validation count invariant from `>=29` to `>=37`.
- Updated operator handoff, handoff validation, evidence manifest, and final runner tests.
- Regenerated final signoff evidence twice to settle the manifest/validation/trace cycle; current handoff validation is `pass` with `37/37` checks and final evidence manifest remains `pass` with `36/36` required artifacts and `12` consistency checks.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.113 - 2026-06-11

- Strengthened `hardware/tools/write_third_goal_requirements_trace.py`.
- Requirements trace now includes `final_evidence_manifest_contract` with manifest status, required artifact count, consistency count, failed consistency checks, source, path, and no-write safety flags.
- Updated `hardware/tools/run_third_goal_final_signoff.py` so the trace receives the stable `docs/resources/final_evidence_manifest_2026_06_10.json` path without introducing manifest hash recursion.
- Updated `hardware/tests/test_write_third_goal_requirements_trace.py` and `hardware/tests/test_run_third_goal_final_signoff.py`.
- Regenerated final signoff evidence; current requirements trace reports final evidence manifest contract `pass`, required `36/36`, consistency `12`, failed consistency checks `[]`.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.112 - 2026-06-11

- Strengthened `hardware/tools/write_third_goal_completion_audit.py`.
- Completion audit now adds requirement `(12)` for the final evidence manifest consistency contract.
- Requirement `(12)` uses the same `write_final_evidence_manifest.build_consistency_checks()` invariant set and fails if the manifest is missing or any consistency check fails.
- Updated `hardware/tests/test_write_third_goal_completion_audit.py` with pass/fail coverage for the manifest consistency contract.
- Regenerated final signoff evidence; current completion audit is `blocked` overall but now reports `pass=8`, `partial=4`, `blocked=2`.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.111 - 2026-06-11

- Strengthened `hardware/tools/write_final_evidence_manifest.py`.
- Final evidence manifest now includes `consistency_checks` for handoff validation, bundle validation, XR-VITs reference resolution, final blocker closure, and C3b smoke discovery evidence.
- Manifest status now fails when key evidence status/count invariants fail, even if files exist and parse.
- Updated `hardware/tests/test_write_final_evidence_manifest.py` with realistic invariant fixtures and failure coverage.
- Regenerated final evidence manifest; current status is `pass` with `36/36` required artifacts and `12/12` consistency checks passing.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.110 - 2026-06-11

- Strengthened `hardware/tools/validate_final_operator_handoff.py`.
- Operator handoff validation now checks `xr_vits.reference_resolution` presence, known status, blocker consistency, candidate score, dry-run command, approval command, no policy write, and no canonical input write.
- Updated `hardware/tests/test_validate_final_operator_handoff.py` with XR-VITs resolution pass/fail coverage.
- Regenerated final handoff validation evidence; current status is `pass` with `29/29` checks and `fail_count=0`.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.109 - 2026-06-11

- Added XR-VITs reference-resolution evidence to `hardware/tools/write_third_goal_requirements_trace.py`.
- Added XR-VITs reference-resolution evidence to `hardware/tools/write_final_operator_handoff.py`.
- Updated `hardware/tools/run_third_goal_final_signoff.py` so trace and handoff receive the generated `xr_vits_reference_resolution_2026_06_10.json`.
- Updated `hardware/tests/test_write_third_goal_requirements_trace.py`, `hardware/tests/test_write_final_operator_handoff.py`, and `hardware/tests/test_run_third_goal_final_signoff.py`.
- Regenerated trace and handoff evidence; both now report `candidate-ready-needs-approval`, `resolution_ready=False`, and candidate score `99`.
- Current final status remains `blocked` pending C3b physical smoke import and exact/approved XR-VITs resolution.

## 0.1.108 - 2026-06-11

- Strengthened `hardware/tools/validate_final_signoff_bundle.py`.
- Final bundle validation now requires `docs/resources/xr_vits_reference_resolution_2026_06_10.json`.
- Added checks for XR-VITs resolution status, blocker consistency, no policy write, and no canonical input write.
- Updated `hardware/tests/test_validate_final_signoff_bundle.py` with blocked-pass and mismatch-fail coverage.
- Regenerated final signoff evidence; bundle validation now passes `45` checks with `fail_count=0`.
- Current final summary remains `blocked` with `xr_vits_reference_resolution_status=candidate-ready-needs-approval`.

## 0.1.107 - 2026-06-11

- Added `hardware/tools/check_xr_vits_reference_resolution.py`.
- Added `hardware/tests/test_check_xr_vits_reference_resolution.py`.
- Integrated `xr-vits-reference-resolution` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the reference-resolution tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Updated `hardware/tools/write_final_evidence_manifest.py` so the manifest includes XR-VITs reference-resolution evidence.
- Generated and mirrored `docs/resources/xr_vits_reference_resolution_2026_06_10.json/.md`.
- Final signoff summary now reports `xr_vits_reference_resolution_status=candidate-ready-needs-approval`.
- Current status remains blocked until exact `XR-VITs` is restored or the `XR_Accel` replacement is explicitly approved.

## 0.1.106 - 2026-06-11

- Added `hardware/tools/validate_final_operator_handoff.py`.
- Added `hardware/tests/test_validate_final_operator_handoff.py`.
- Integrated `final-operator-handoff-validation` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the validator tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_operator_handoff_validation_2026_06_10.json/.md`.
- Final signoff summary now reports `operator_handoff_validation_status=pass`.
- Current handoff validation passes 21 checks and confirms no command execution, board-result creation, active XR-VITs policy creation, or canonical input write.

## 0.1.105 - 2026-06-11

- Added `hardware/tools/write_final_operator_handoff.py`.
- Added `hardware/tests/test_write_final_operator_handoff.py`.
- Integrated `final-operator-handoff` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the handoff tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_operator_handoff_2026_06_10.json/.md`.
- Final signoff summary now reports `operator_handoff_status=pending-operator-actions`.
- The handoff consolidates board-smoke bundle SHA, expected C3b output, XR-VITs exact/replacement commands, combined unblock commands, final signoff command, and safety flags.

## 0.1.104 - 2026-06-11

- Added optional `--candidate-audit` input to `hardware/tools/write_third_goal_requirements_trace.py`.
- Requirements trace JSON now includes `candidate_unblock_audit` with candidate status, C3b/XR-VITs clearance flags, remaining blockers, and side-effect safety flags.
- Requirements trace Markdown now includes a `Candidate Unblock Audit` section.
- Updated `hardware/tools/run_third_goal_final_signoff.py` so requirements trace receives the generated `final_unblock_candidate_audit_2026_06_10.json`.
- Updated `hardware/tests/test_write_third_goal_requirements_trace.py` and `hardware/tests/test_run_third_goal_final_signoff.py`.
- Regenerated and mirrored final signoff evidence; current trace still reports `candidate_unblock_audit.status=blocked`.

## 0.1.103 - 2026-06-11

- Integrated `hardware/tools/audit_final_unblock_candidates.py` into `hardware/tools/run_third_goal_final_signoff.py`.
- The final signoff runner now emits `candidate_audit_status` and includes a `final-unblock-candidate-audit` step in `third_goal_final_signoff_run_2026_06_10.json/.md`.
- The runner mirrors `docs/resources/final_unblock_candidate_audit_2026_06_10.json/.md` as part of the integrated signoff artifact set.
- Default candidate audit mode is exact `XR-VITs`; replacement mode is used only when explicit XR-VITs replacement approval arguments are supplied.
- Current integrated candidate audit status remains `blocked`: no C3b physical smoke JSON was supplied and exact `/home/kjm26/project/PRJXR/XR-VITs` is missing.

## 0.1.102 - 2026-06-11

- Added `hardware/tools/audit_final_unblock_candidates.py`.
- Added `hardware/tests/test_audit_final_unblock_candidates.py`.
- Registered the audit tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_unblock_candidate_audit_2026_06_10.json/.md`.
- The audit previews whether supplied C3b smoke JSON plus XR-VITs exact/replacement evidence would clear final blockers without creating canonical smoke results, active XR-VITs replacement policy, or network side effects.
- Current audit status remains `blocked`: no C3b physical smoke JSON was supplied and `XR-VITs` is still unresolved.

## 0.1.101 - 2026-06-11

- Extended `hardware/tools/write_third_goal_requirements_trace.py` with `blocker_resolution_conditions`.
- The requirements trace now maps `requested XR-VITs sibling` to requirements `10` and `11`, required evidence paths, acceptance checks, and command-card options.
- The requirements trace now maps `C3b AXIS/DMA physical smoke result` to requirements `3` and `4`, required evidence paths, acceptance checks, and command-card options.
- Updated `hardware/tests/test_write_third_goal_requirements_trace.py` to validate blocker resolution conditions and command option extraction.
- Regenerated and mirrored `docs/resources/third_goal_requirements_trace_2026_06_10.json/.md`.

## 0.1.100 - 2026-06-11

- Added `--dry-run` to `hardware/tools/import_pynq_smoke_result.py` so a board smoke JSON can be validated without copying it into the canonical result path.
- Added `--dry-run-import-c3b-smoke` to `hardware/tools/run_third_goal_final_signoff.py`; successful dry-run import reports `c3b_import_status=dry-run-pass`.
- Updated U4b in `hardware/tools/write_final_unblock_commands.py` so the first combined command is now fully side-effect safe for both C3b import and XR-VITs replacement approval.
- Updated `hardware/tests/test_import_pynq_smoke_result.py`, `hardware/tests/test_run_third_goal_final_signoff.py`, and `hardware/tests/test_write_final_unblock_commands.py`.
- Regenerated and mirrored final runner and command-card evidence; current default status remains blocked because no physical C3b smoke JSON was supplied and `XR-VITs` remains unresolved.

## 0.1.99 - 2026-06-11

- Added U4 `Combined one-shot unblock` to `hardware/tools/write_final_unblock_commands.py`.
- U4a covers the exact `XR-VITs` restore path plus C3b board-result import through `run_third_goal_final_signoff.py`.
- U4b covers the explicit `XR_Accel` approval path plus C3b board-result import through `run_third_goal_final_signoff.py`.
- U4b includes a dry-run approval/import command before the active approval/import command.
- Updated `hardware/tests/test_write_final_unblock_commands.py` to validate the new U4 section and combined final-runner command flags.
- Regenerated and mirrored final runner and command-card evidence; current default status remains blocked because no physical C3b smoke JSON was supplied and `XR-VITs` remains unresolved.

## 0.1.98 - 2026-06-11

- Added `--import-c3b-smoke-json` to `hardware/tools/run_third_goal_final_signoff.py`.
- The final runner can now validate/import an already copied C3b board smoke JSON before readiness and final-preflight checks.
- Added `--import-c3b-no-require-paths` for controlled validation of smoke JSONs that omit board-local bit/HWH path fields.
- Final runner summaries now report `c3b_import_status`.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to verify import step ordering, import status, and docs-resource mirror outputs.
- Updated `hardware/tools/write_final_unblock_commands.py` so U1 includes both one-shot final-runner import and direct `import_pynq_smoke_result.py` fallback commands.
- Updated `hardware/tests/test_write_final_unblock_commands.py` to cover the new U1 import commands.
- Regenerated and mirrored final runner and command-card evidence; current default status remains blocked because no physical C3b smoke JSON was supplied and `XR-VITs` remains unresolved.

## 0.1.97 - 2026-06-11

- Added `--dry-run-xr-vits-replacement` to `hardware/tools/run_third_goal_final_signoff.py`.
- Dry-run approval mode passes `--dry-run` to `hardware/tools/create_xr_vits_replacement_policy.py` and reports `xr_vits_policy_status=dry-run-pass` when validation succeeds.
- Default final-runner behavior remains unchanged: no approval flag means `xr_vits_policy_status=skipped`.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to validate dry-run approval behavior and step ordering before XR-VITs packet generation.
- Updated `hardware/tools/write_final_unblock_commands.py` so U2b now starts with a one-shot final-runner dry-run approval command.
- Updated `hardware/tests/test_write_final_unblock_commands.py` to cover the dry-run command.
- Regenerated and mirrored final runner and command-card evidence; current default status remains blocked because no active policy or physical C3b smoke evidence was created.

## 0.1.96 - 2026-06-11

- Added gated XR-VITs replacement-policy controls to `hardware/tools/run_third_goal_final_signoff.py`.
- New final-runner options: `--approve-xr-vits-replacement`, `--xr-vits-approved-by`, `--xr-vits-replacement-reason`, and `--xr-vits-replacement-path`.
- Preserved default final-runner behavior as policy-safe: no active replacement policy is created unless the explicit approval flag and metadata are supplied.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to validate default `xr_vits_policy_status=skipped` and approval-step argument forwarding before XR-VITs packet generation.
- Updated `hardware/tools/write_final_unblock_commands.py` so U2b includes a one-shot final-runner approval command after the existing policy-helper commands.
- Updated `hardware/tests/test_write_final_unblock_commands.py` to cover the new U2b command.
- Regenerated and mirrored final runner and command-card evidence; current default status remains blocked because policy creation was skipped and physical C3b smoke evidence is still missing.

## 0.1.95 - 2026-06-11

- Added optional ZCU104 execution controls to `hardware/tools/run_third_goal_final_signoff.py`: `--zcu104-host`, `--zcu104-user`, `--zcu104-port`, `--zcu104-identity-file`, `--zcu104-remote-dir`, and `--execute-zcu104-smoke`.
- Preserved the default final runner behavior as dry-run only; no SSH/SCP command runs unless `--execute-zcu104-smoke` is passed.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to validate default dry-run behavior and execute-mode argv forwarding.
- Updated `hardware/tools/write_final_unblock_commands.py` so U1 now offers a one-shot final-runner board command before the lower-level remote runner fallback.
- Updated `hardware/tests/test_write_final_unblock_commands.py` to cover the new command-card path.
- Regenerated and mirrored final runner, command-card, matrix, and trace evidence; current final status remains blocked only by missing physical C3b smoke evidence and unresolved `XR-VITs`.

## 0.1.94 - 2026-06-11

- Integrated `hardware/tools/write_e2e_resource_matrix.py` into `hardware/tools/run_third_goal_final_signoff.py`.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to validate the `resource-matrix` step, generated-matrix trace input, summary status, and mirrored matrix Markdown.
- Regenerated and mirrored `docs/resources/third_goal_final_signoff_run_2026_06_10.json/.md` with `resource_matrix_status=pass`.
- The final runner now regenerates the resource matrix before requirements trace generation, avoiding stale docs-resource matrix input.
- Current integrated status remains `blocked` only because physical C3b smoke evidence and exact `XR-VITs`/approved replacement evidence are still missing.

## 0.1.93 - 2026-06-11

- Integrated `hardware/tools/write_third_goal_requirements_trace.py` into `hardware/tools/run_third_goal_final_signoff.py`.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to validate the `requirements-trace` step, summary status, and mirrored trace Markdown.
- Regenerated and mirrored `docs/resources/third_goal_final_signoff_run_2026_06_10.json/.md` with `requirements_trace_status=blocked`.
- The integrated runner now regenerates remote dry-run, XR-VITs packet, readiness, final preflight/audit, completion audit, unblock checklist, command card, and requirements trace in one host-side pass.
- Current integrated status remains `blocked` only because physical C3b smoke evidence and exact `XR-VITs`/approved replacement evidence are still missing.

## 0.1.92 - 2026-06-11

- Added `hardware/tools/write_third_goal_requirements_trace.py`.
- Added `hardware/tests/test_write_third_goal_requirements_trace.py`.
- Registered the requirements-trace tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/third_goal_requirements_trace_2026_06_10.json/.md`.
- Trace links the explicit [3차목표] requirements `(0)..(11)` to completion-audit evidence, C3b resource snapshot, unblock actions, and final signoff blockers.
- Trace preserves safety: no board smoke result, XR-VITs policy, or network command is created/executed.

## 0.1.91 - 2026-06-11

- Added `hardware/tools/write_e2e_resource_matrix.py`.
- Added `hardware/tests/test_write_e2e_resource_matrix.py`.
- Registered the resource-matrix tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/e2e_resource_matrix_2026_06_10.json/.md`.
- Matrix parses authoritative HLS resources from `csynth.xml` and Vivado WNS/power from routed reports for A2, A1, C1, and C3b.
- Current matrix confirms C1/C3b tie for best latency and highest DSP; C3b has lower LUT than C1 and remains the recommended board-smoke variant.

## 0.1.90 - 2026-06-11

- Added `hardware/tools/write_final_unblock_commands.py`.
- Added `hardware/tests/test_write_final_unblock_commands.py`.
- Integrated final unblock command-card generation into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the command-card tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_unblock_commands_2026_06_10.json/.md`.
- Command card status is `pending-unblock`; it lists exact operator commands for C3b ZCU104 smoke execution, XR-VITs resolution, and final signoff rerun.
- Safety preserved: command card does not create board smoke results, does not create an XR-VITs policy, and does not execute network commands.

## 0.1.89 - 2026-06-11

- Integrated `hardware/tools/run_zcu104_c3b_smoke_remote.py` and `hardware/tools/write_xr_vits_unblock_packet.py` into `hardware/tools/run_third_goal_final_signoff.py`.
- The final signoff runner now regenerates remote C3b smoke dry-run evidence and XR-VITs user-choice packet before final preflight/audit generation.
- Updated `hardware/tests/test_run_third_goal_final_signoff.py` to validate the new step order and summary fields.
- Regenerated and mirrored `docs/resources/third_goal_final_signoff_run_2026_06_10.json/.md`.
- Current integrated runner status remains `blocked`; added summary fields show `zcu104_remote_status=dry-run` and `xr_vits_packet_status=pending-user-choice`.

## 0.1.88 - 2026-06-11

- Added `hardware/tools/write_xr_vits_unblock_packet.py`.
- Added `hardware/tests/test_write_xr_vits_unblock_packet.py`.
- Registered the XR-VITs unblock packet tool/test and tracked mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/xr_vits_unblock_packet_2026_06_10.json/.md`.
- Packet status is `pending-user-choice`: exact `/home/kjm26/project/PRJXR/XR-VITs` is missing and active `docs/resources/xr_vits_replacement_policy.json` is absent.
- Packet preserves safety boundary: it does not create the active replacement policy; it only records exact-restore and explicit-approval command options.

## 0.1.87 - 2026-06-11

- Added `hardware/tools/run_zcu104_c3b_smoke_remote.py`.
- Added `hardware/tests/test_run_zcu104_c3b_smoke_remote.py`.
- Registered the ZCU104 remote smoke runner tool/test and tracked dry-run mirrors in `hardware/tools/static_validate_hgtxr.py`.
- The runner validates the local C3b tarball/SHA file, emits SSH/SCP commands in dry-run mode, and can execute transfer/run/fetch/import when `--execute` and board connection options are provided.
- Generated and mirrored `docs/resources/zcu104_c3b_smoke_remote_run_2026_06_10.json/.md`.
- Current remote runner dry-run status is `dry-run` with `errors=0`; it does not fabricate physical board evidence.

## 0.1.86 - 2026-06-11

- Added `hardware/tools/run_third_goal_final_signoff.py`.
- Added `hardware/tests/test_run_third_goal_final_signoff.py`.
- Registered the final-signoff runner tool/test and tracked runner mirrors in `hardware/tools/static_validate_hgtxr.py`.
- The runner regenerates C3b readiness, final preflight, final signoff audit, completion audit, and unblock checklist in one host-side pass.
- Generated and mirrored `docs/resources/third_goal_final_signoff_2026_06_10.json`.
- Generated and mirrored `docs/resources/third_goal_final_signoff_run_2026_06_10.json/.md`.
- Current runner summary is `status=blocked`, `fail=2`, remaining blockers: `requested XR-VITs sibling` and `C3b AXIS/DMA physical smoke result`.

## 0.1.85 - 2026-06-10

- Added `hardware/tools/check_c3b_board_smoke_readiness.py`.
- Added `hardware/tests/test_check_c3b_board_smoke_readiness.py`.
- Registered the readiness checker tool/test and tracked readiness mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/c3b_board_smoke_readiness_2026_06_10.json/.md`.
- Mirrored readiness evidence to `docs/resources/c3b_board_smoke_readiness_2026_06_10.json/.md`.
- Readiness status is `ready-for-board`: transfer manifest, session, bundle, tar SHA256, board commands, host import command, and expected output contract pass.
- Physical C3b board smoke result is still missing by design; final signoff remains blocked until ZCU104 smoke is run/imported and the `XR-VITs` gate is resolved.

## 0.1.84 - 2026-06-10

- Added `hardware/tools/write_c3b_smoke_transfer_manifest.py`.
- Added `hardware/tests/test_write_c3b_smoke_transfer_manifest.py`.
- Registered the transfer manifest tool/test and tracked transfer mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/c3b_smoke_transfer_manifest_2026_06_10.json/.md`.
- Generated `hardware/generated/signoff/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`.
- Mirrored transfer evidence to `docs/resources/c3b_smoke_transfer_manifest_2026_06_10.json/.md` and `docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256`.
- Transfer manifest status is `pass`; bundle SHA256 is `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.

## 0.1.83 - 2026-06-10

- Added `hardware/tools/write_third_goal_unblock_checklist.py`.
- Added `hardware/tests/test_write_third_goal_unblock_checklist.py`.
- Registered the unblock checklist tool/test and tracked checklist mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/third_goal_unblock_checklist_2026_06_10.json/.md`.
- Mirrored the checklist to `docs/resources/third_goal_unblock_checklist_2026_06_10.json/.md`.
- Checklist status is `pending-unblock` with 2 final blockers and 3 steps: resolve `XR-VITs`, run/import C3b physical smoke, rerun final host signoff.

## 0.1.82 - 2026-06-10

- Added `hardware/tools/write_third_goal_completion_audit.py`.
- Added `hardware/tests/test_write_third_goal_completion_audit.py`.
- Registered the completion audit tool/test and tracked audit mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/third_goal_completion_audit_2026_06_10.json/.md`.
- Mirrored the completion audit to `docs/resources/third_goal_completion_audit_2026_06_10.json/.md`.
- Requirement-by-requirement status: `pass=7`, `partial=4`, `blocked=2`.
- Blocked items: `(11) XR-VITs HLS source requirement` and `final signoff gate`.

## 0.1.81 - 2026-06-10

- Added `hardware/tools/create_xr_vits_replacement_policy.py`.
- Added `hardware/tests/test_create_xr_vits_replacement_policy.py`.
- Registered the policy helper tool and test in `hardware/tools/static_validate_hgtxr.py`.
- The helper refuses to write active `docs/resources/xr_vits_replacement_policy.json` unless `--approve`, `--approved-by`, and `--reason` are provided.
- The helper validates that `replacement_path` exists, the requested `XR-VITs` path is still missing, and the replacement matches `docs/resources/xr_vits_candidate_audit_2026_06_10.json`.
- Verified real-root dry-run for `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`; no active policy was written.

## 0.1.80 - 2026-06-10

- Added an explicit `XR-VITs` replacement policy gate to `hardware/tools/check_third_goal_preflight.py`.
- Final signoff still fails when `/home/kjm26/project/PRJXR/XR-VITs` is missing unless `docs/resources/xr_vits_replacement_policy.json` exists and records an approved replacement.
- Added inactive template `docs/resources/xr_vits_replacement_policy.template.json`.
- Extended `hardware/tests/test_check_third_goal_preflight.py` to 16 tests covering approved and rejected replacement policies.
- Registered the policy template in `hardware/tools/static_validate_hgtxr.py`.
- Regenerated final-signoff evidence; current status remains `ok=62`, `warn=4`, `fail=2`.
- Remaining blockers: C3b physical smoke result and missing `XR-VITs` with no approved replacement policy.

## 0.1.79 - 2026-06-10

- Added `hardware/tools/audit_xr_vits_candidates.py`.
- Added `hardware/tests/test_audit_xr_vits_candidates.py`.
- Registered the XR-VITs candidate audit tool and test in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/xr_vits_candidate_audit_2026_06_10.json/.md`.
- Mirrored the audit to `docs/resources/xr_vits_candidate_audit_2026_06_10.json/.md`.
- Candidate ranking: `XR_Accel` score `99`, `ViT_Accel` score `90`, `analysis/XR_Accel` score `0`.
- Recommended candidate is `XR_Accel`, but `approved_replacement=false`; final-signoff still requires explicit approval or exact `XR-VITs` restoration.

## 0.1.78 - 2026-06-10

- Restored the requested PAPER_PRJXR DeiT image path from the existing HGPIPE image.
- Created `/home/kjm26/project/PRJXR/XR-VIT/PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png`.
- Verified the restored image SHA256 matches HGPIPE: `90495791e6103be30268320d3b17dce2eaa52d003dd6f341393262745c1c3b79`.
- Regenerated final-signoff and reference-input audits.
- Current final-signoff status improved to `ok=62`, `warn=4`, `fail=2`.
- Remaining blockers: C3b physical smoke result and requested `XR-VITs` sibling.

## 0.1.77 - 2026-06-10

- Added `hardware/tools/audit_reference_inputs.py`.
- Added `hardware/tests/test_audit_reference_inputs.py`.
- Registered the reference-input audit tool and test in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/reference_input_audit_2026_06_10.json/.md`.
- Mirrored the audit to `docs/resources/reference_input_audit_2026_06_10.json/.md` because `hardware/generated/` is ignored.
- Current reference audit status is `needs-reference-input`: the requested `PAPER_PRJXR` image and `XR-VITs` sibling are missing.
- Candidate inputs found but not approved as replacements: `HGPIPE/DeiT-Tiny C-Syn Results.png`, `XR_Accel`, `analysis/XR_Accel`, and `ViT_Accel`.

## 0.1.76 - 2026-06-10

- Added `hardware/tools/write_final_signoff_audit.py`.
- Added `hardware/tests/test_write_final_signoff_audit.py`.
- Registered the final-signoff audit tool and test in `hardware/tools/static_validate_hgtxr.py`.
- Generated `hardware/generated/signoff/third_goal_final_signoff_2026_06_10.json`.
- Generated `hardware/generated/signoff/third_goal_board_ready_2026_06_10.json`.
- Generated `hardware/generated/signoff/final_signoff_audit_2026_06_10.json`.
- Generated `hardware/generated/signoff/final_signoff_audit_2026_06_10.md`.
- Mirrored the audit to `docs/resources/final_signoff_audit_2026_06_10.json/.md` because `hardware/generated/` is ignored.
- Current audit status is `blocked` with 3 blockers: C3b physical smoke result, requested `PAPER_PRJXR` image, and requested `XR-VITs` sibling.

## 0.1.75 - 2026-06-10

- Added `final-signoff` mode to `hardware/tools/check_third_goal_preflight.py`.
- Kept `board-ready` behavior unchanged: physical C3b smoke result remains a warning until board execution is copied back.
- Made `final-signoff` fail when the C3b physical smoke JSON is missing or invalid.
- Made `final-signoff` fail missing requested reference inputs: `PAPER_PRJXR` DeiT image and `XR-VITs` sibling.
- Extended `hardware/tests/test_check_third_goal_preflight.py` to cover board-ready warning behavior and final-signoff failure behavior.
- Current board-ready preflight: `ok=61`, `warn=7`, `fail=0`.
- Current final-signoff preflight: `ok=61`, `warn=4`, `fail=3`; physical ZCU104 C3b smoke still open.

## 0.1.74 - 2026-06-10

- Extended `hardware/tools/check_third_goal_preflight.py` with C3b ZCU104 smoke-session JSON/Markdown validation.
- Board-ready preflight now validates session status, variant, preset, expected output, tarball size/SHA256, canonical result path, board commands, host import command, and Markdown runbook command coverage.
- Extended `hardware/tests/test_check_third_goal_preflight.py` with complete-session acceptance and missing-session failure coverage.
- Current board-ready preflight reports `ok=61`, `warn=7`, `fail=0`.

## 0.1.73 - 2026-06-10

- Added `hardware/tools/prepare_zcu104_smoke_session.py` to generate a reproducible C3b ZCU104 PYNQ smoke-session JSON/Markdown runbook.
- Added `hardware/tests/test_prepare_zcu104_smoke_session.py`.
- Generated `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json`.
- Generated `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md`.
- Registered the session tool and tests in `hardware/tools/static_validate_hgtxr.py`.

## 0.1.72 - 2026-06-10

- Added `hardware/tools/import_pynq_smoke_result.py` to validate a board-produced PYNQ smoke result JSON before copying it into the canonical host-side result path.
- Added `hardware/tests/test_import_pynq_smoke_result.py`.
- The importer defaults C3b results to `hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json`, which is the path already consumed by board-ready preflight.
- Invalid smoke results are rejected without overwriting the current canonical result JSON.
- Registered the importer and tests in `hardware/tools/static_validate_hgtxr.py`.

## 0.1.71 - 2026-06-10

- Added `hardware/tools/validate_pynq_bundle_package.py` to validate C3b PYNQ bundle directory, manifest fields, required files, per-file SHA256 values, tarball SHA256, tar contents, and tar member hashes.
- Added `hardware/tests/test_validate_pynq_bundle_package.py` with good-bundle, broken-tar, and CLI JSON-output coverage.
- Integrated the package validator into `hardware/tools/check_third_goal_preflight.py` as `C3b AXIS/DMA PYNQ smoke bundle package`.
- Registered the validator and tests in `hardware/tools/static_validate_hgtxr.py`.
- Refreshed `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`; current board-ready preflight is `ok=57`, `warn=7`, `fail=0`.

## 0.1.70 - 2026-06-10

- Repacked the C3b AXIS/DMA PYNQ smoke bundle with `tools/validate_pynq_smoke_result.py` included under the bundle `tools/` directory.
- Added bundle-local result validation script `validate_e2e_axis_dma_c3b_mem16_file_smoke.sh`.
- Added `validation_command` to the C3b bundle manifest.
- Regenerated C3b bundle tarball with SHA256 `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712` and size `593,198` bytes.
- Updated PYNQ README with the bundle-local validation command.

## 0.1.69 - 2026-06-10

- Added `hardware/tools/validate_pynq_smoke_result.py` to validate physical PYNQ smoke result JSON for A2 m_axi and AXIS/DMA variants, including C3b.
- Added `hardware/tests/test_validate_pynq_smoke_result.py`.
- Extended board-ready preflight to validate copied-back C3b physical smoke JSON when present and warn when it has not been captured yet.
- Recorded the new physical-result gate evidence in `docs/resources/third_goal_preflight_board_c3b_result_gate_2026_06_10.json`; current result is `ok=54`, `warn=7`, `fail=0`.

## 0.1.68 - 2026-06-10

- Extended `hardware/tools/check_third_goal_preflight.py` so board-ready mode validates the C3b AXIS/DMA PYNQ smoke bundle when C3b artifacts are present.
- Added preflight checks for C3b bundle directory, manifest, tarball, run script, variant, expected output, command, required contents, and tar SHA256.
- Extended `hardware/tests/test_check_third_goal_preflight.py` with C3b bundle-pass and C3b artifact-without-bundle failure coverage.
- Recorded the new board-ready preflight evidence in `docs/resources/third_goal_preflight_board_c3b_bundle_2026_06_10.json`; current result is `ok=54`, `warn=6`, `fail=0`.

## 0.1.67 - 2026-06-10

- Added variant selection to the AXIS/DMA PYNQ smoke CLI, including `c3b-mem16`.
- Added `hardware/tools/package_e2e_axis_dma_pynq_bundle.py` for self-contained AXIS/DMA PYNQ smoke bundles.
- Generated C3b bundle directory `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle`.
- Generated C3b bundle tarball `hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz` with SHA256 `3de505a6627409b4717ea95e5e76d74f0667c61b601fd96db1e495319adda712`.
- Added bundle unittest coverage and documented C3b file-mode board-smoke commands in the PYNQ README.

## 0.1.66 - 2026-06-10

- Exported C3b HLS IP under isolated project `hgtxr_e2e_axis_par16_c3b_mem16_hls`.
- Built isolated C3b AXIS/DMA Vivado overlay `hgtxr_e2e_axis_dma_c3b_mem16_overlay`.
- Produced board-ready artifacts `hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit/.hwh`.
- Routed C3b timing passed with `WNS=4.415 ns`; routed power is `3.476 W`.
- Recorded C3b hashes: bit `989e77836341467da05d0ccb86f0b4764d38f4711af8fbcb567290b78fe3076d`, hwh `8ca659da3bbcc061f7299ac18314c00b8ee2112a274990d969635b3f6acf3c90`, export zip `a471fed1d43d2323efd22cc3581cfd7ee1c1cb5a0cfbc69ba57b2877da527e38`.
- Extended board-ready preflight and tests to recognize optional C3b PAR16/MEM16 artifacts.

## 0.1.65 - 2026-06-10

- Ran C3b PAR16/MEM16 fastpath-only CSim and CSynth.
- C3b restored C1-like latency: `37,508,072 cycles`.
- C3b preserved high resource mapping: `604 DSP`, `64 URAM`.
- C3b resources are `332 BRAM_18K`, `59,505 FF`, `126,506 LUT`.
- Recorded conclusion that the large C3/MEM8 LUT reduction came mostly from lower memory banking, while fastpath-only gives a small `1,410` LUT improvement versus C1.

## 0.1.64 - 2026-06-10

- Added `HGTXR_E2E_MEM_BANK_PAR` to decouple compute parallelism from memory-bank partitioning.
- Added aligned packed-weight vector fast path through `HGTXR_E2E_WEIGHT_VEC_ALIGNED_FASTPATH`.
- Extended E2E AXIS CSim/CSynth Tcl flows with `HGTXR_E2E_MEM_BANK_PAR` support.
- Ran C3 PAR16/MEM8 CSynth: `292 BRAM_18K`, `604 DSP`, `56,966 FF`, `113,124 LUT`, `64 URAM`, `48,598,922 cycles`.
- Recorded the C3 tradeoff: `11.6%` LUT reduction versus C1 while preserving DSP/URAM, but with higher latency due MLP W2 `Final II=2`.

## 0.1.63 - 2026-06-10

- Completed the C1 PAR16 E2E AXIS/DMA board-candidate path.
- Exported isolated C1 HLS/IP under `hardware/generated/hgtxr_e2e_axis_par16_hls`.
- Built isolated C1 Vivado overlay and copied `hgtxr_e2e_axis_dma_par16.bit/.hwh` to `hardware/pynq/hgtxr/`.
- Recorded C1 as the current routed high-DSP candidate: `604 DSP`, `64 URAM`, routed `WNS=4.723 ns`, with high LUT pressure at `127,916`.
- Extended board-ready preflight to record optional C1 PAR16 PYNQ artifacts when present.

## 0.1.25 - 2026-06-10

- Added A2 PYNQ file-smoke transfer bundle generation.
- Packaged selected E2E m_axi bit/HWH, Python runtime helpers, and exported packed Q4 weight artifacts under `hardware/generated/pynq/e2e_m_axi_smoke_bundle`.
- Generated `hardware/generated/pynq/e2e_m_axi_smoke_bundle.tar.gz` with SHA256 `56ade675e76d1abadab5a9bdbf00b2baf1f8d7b4f5488b9ee313405398803d3a`.
- Added bundle unittest coverage and updated PYNQ README so extracted-bundle file-mode smoke is self-contained.

## 0.1.26 - 2026-06-10

- Added A1 E2E AXIS/DMA board-flow scaffold with separate IP tree and artifact names.
- Added `package_e2e_axis_ip.tcl` and `build_e2e_axis_dma_bitstream.tcl`.
- Added PYNQ AXIS/DMA runtime helper and smoke CLI.
- Extended PYNQ off-board tests to cover fake DMA transfer, raw AXIS frame packing, register writes, and smoke modes.

## 0.1.24 - 2026-06-10

- Added per-buffer URAM steering for full E2E activation buffers.
- Added `dsp_mixed_hidden`, `dsp_mixed_stream`, and `dsp_mixed` E2E resource policies.
- Verified full `active196_b6_ff768` PAR8 CSynth resource probes.
- Selected `dsp_mixed_stream` as the current full-scale fit point: `210 BRAM_18K`, `268 DSP`, `39,370 FF`, `90,836 LUT`, `64 URAM`, `82,612,245 cycles`.
- Documented all-URAM full PAR8 overuse (`160/96 URAM`) and remaining AXI weight-port II limits.

## 0.1.23 - 2026-06-10

- Rebalanced staged E2E HLS resource mapping toward DSP and URAM.
- Added DSP-bound MAC helpers and Q4W8A constant-lane weight extraction.
- Moved large E2E activation buffers to URAM.
- Verified `hgpipe_math_lnq_active16` native, Vitis CSim, and Vitis CSynth; LUT dropped to `44,670`, DSP rose to `55`, and URAM rose to `80`.
- Added `HGTXR_E2E_RESOURCE_POLICY` and recorded same-scale active16 policy matrix in `docs/resources/e2e_active16_resource_policy_matrix_2026_06_10.md`.

## 0.1.0

- Created HGTXR software/hardware co-design repository skeleton.
- Added Python software reference implementation.
- Added HLS module tree and Tcl flow skeleton.
- Added tools, scripts, tests, and durable documentation.

## 0.1.1 - 2026-06-06

- Patched Vivado HWH copy fallback and populated overlay/PYNQ `.bit/.hwh` artifacts.
- Made cyclic Transformer HLS parameters compile-time overrideable from sweep-generated defines.
- Generated implementation-reference resource metrics from `impl_repos`.
- Verified static artifact checks, PYNQ helper smoke, and cyclic primitive smoke.

## 0.1.2 - 2026-06-06

- Integrated a gated cyclic Transformer top path while preserving the PYNQ-facing `hgtxr_top` ABI.
- Added conservative ZCU104 cyclic S0 HLS defines and cyclic csynth Tcl.
- Made HLS project creation consume cyclic gate, define file, and solution parameters.
- Fixed cyclic MLP HLS dataflow synthesis failure.
- Added report-backed default and cyclic S0 csynth summary CSV.

## 0.1.3 - 2026-06-06

- Specialized cyclic Transformer top scheduling by active token count for search and track branches.
- Reduced cyclic S0 HLS latency estimate from `78,024,797 cycles` to `15,023,573 cycles`.
- Updated validation, progress, execution, spec, handoff, and csynth summary artifacts with the latest report-backed resource-fit baseline.

## 0.1.4 - 2026-06-06

- Added optional packed cyclic weight AXI port behind `HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS` while preserving the default PYNQ ABI.
- Added tile-local packed norm/QKV/O/MLP weight layout constants and loader logic.
- Added S1 weight-port define/Tcl artifacts and report-backed HLS csynth summary row.
- Documented the weight packing contract and S1 resource/latency evidence.

## 0.1.5 - 2026-06-06

- Added host-side cyclic weight packer for the optional S1 packed-weight ABI.
- Added packer manifest, raw binary, and `.npz` output support.
- Added deterministic packer tests and documented validation evidence.
- Verified the DeiT-Tiny C-Syn image artifact exists while leaving OCR extraction as pending.

## 0.1.6 - 2026-06-06

- Added S2 channel-pair packed weight layout mode for future full dense projection accumulation.
- Added S2 manifest block records with layer/input/output channel tile metadata.
- Added S2 packer test coverage and validation evidence.


## 0.1.7 - 2026-06-06

- Added S2 HLS channel-pair layout helpers and packed projection accumulation wrapper.
- Added S2 HLS smoke test for index math, packed lane loads, direct accumulation, and packed-buffer wrapper accumulation.
- Extended static validation to require S2 HLS primitive artifacts.
- Updated progress, validation, spec, execution, layout, packing, and handoff docs with S2 primitive evidence and remaining top-integration gap.


## 0.1.8 - 2026-06-06

- Added gated `HGTXR_ENABLE_CYCLIC_S2_FIRST_STEP` top path for S2 `WO` channel-pair projection accumulation.
- Added S2 first-step ZCU104 define and Vitis HLS csynth Tcl artifacts.
- Made cyclic packed weight total depth S2-aware when the first-step gate is enabled.
- Ran default/S1/S2 g++ smokes and `solution_cyclic_s2_first_step` csynth; recorded report-backed metrics in the HLS summary CSV and tracking docs.


## 0.1.9 - 2026-06-06

- Added gated `HGTXR_ENABLE_CYCLIC_S2_QKV_FIRST_STEP` top path for S2 WQ/WK/WV channel-pair projection accumulation.
- Added S2 QKV first-step ZCU104 define and Vitis HLS csynth Tcl artifacts.
- Extended S2 projection smoke tests with packed Q/K/V accumulation checks.
- Ran S2 QKV top smoke and `solution_cyclic_s2_qkv_first_step` csynth; recorded report-backed metrics in CSV and tracking docs.


## 0.1.10 - 2026-06-06

- Added gated `HGTXR_ENABLE_CYCLIC_S2_ATTN_FIRST_STEP` top path for S2 QKV to `attention_tile` to WO first-step execution.
- Added S2 attention first-step ZCU104 define and Vitis HLS csynth Tcl artifacts.
- Preserved existing S2 WO and S2 QKV first-step modes while adding the new mutually exclusive attention gate.
- Ran S2 attention top smoke and `solution_cyclic_s2_attn_first_step` csynth; recorded report-backed metrics in CSV and tracking docs.


## 0.1.11 - 2026-06-06

- Added gated `HGTXR_ENABLE_CYCLIC_S2_MLP_FIRST_STEP` top path for explicit `mlp_ratio=4` MLP hidden expansion.
- Added S2 MLP W1/W2 hidden-pair layout helpers, packed accumulation wrappers, and top-stage scheduling.
- Added `s2_mlp_hidden_pairs` host packing mode and direct packer test coverage.
- Added S2 MLP first-step ZCU104 define and Vitis HLS csynth Tcl artifacts.
- Ran S2 MLP top smoke and `solution_cyclic_s2_mlp_first_step` csynth; recorded report-backed metrics in CSV and tracking docs.


## 0.1.12 - 2026-06-06

- Added gated `HGTXR_ENABLE_CYCLIC_S2_BLOCK_FIRST_STEP` top path for residual-aware fused attention plus MLP first-step execution.
- Added fused per-layer packed layout support through `s2_block_first_step` host packing mode.
- Added residual copy/add helpers around S2 attention and MLP first-step stages.
- Added S2 block first-step ZCU104 define and Vitis HLS csynth Tcl artifacts.
- Ran S2 block top smoke and `solution_cyclic_s2_block_first_step` csynth; recorded report-backed metrics in CSV and tracking docs.

## 0.1.13 - 2026-06-06

- Added `HGTXR_ENABLE_CYCLIC_S2_BLOCK_PRE_LN` for approximate pre-LayerNorm in the fused S2 block path.
- Enabled pre-LN in the ZCU104 S2 block first-step config and reused existing packed norm vectors from attention-phase blocks.
- Replaced the timing-heavy Newton `rsqrt` prototype with a bounded piecewise approximation for first-step synthesis.
- Updated `cyclic_s2_block_first_step` CSV metrics after csynth: `267.02 MHz`, `319 BRAM`, `88 DSP`, `32699 FF`, `58901 LUT`.

## 0.1.14 - 2026-06-06

- Added `hardware/tools/validate_s2_block_preln.py` for focused S2 fused-block pre-LayerNorm golden-vector validation.
- Added `docs/resources/s2_block_preln_validation_2026_06_06.json` as durable validation evidence.
- Refined low-variance piecewise `rsqrt` bins in the HLS pre-LN path, reducing synthetic max pre-LN error to below `0.004`.
- Updated static validation to require the new validator and validation artifact.
- Re-ran S2 block csynth: `267.02 MHz`, `319 BRAM`, `88 DSP`, `32701 FF`, `59141 LUT`.

## 0.1.15 - 2026-06-06

- Added `hardware/tools/validate_s2_block_full.py` for full fused S2 block host golden-vector validation.
- Added `docs/resources/s2_block_full_validation_2026_06_06.json` as durable validation evidence.
- Added pytest hook coverage for the full-block validator.
- Extended static validation to require the full-block validator and artifact.
- Confirmed synthetic packed-vs-direct full-block reference: `max_matrix_error=0.0`, `max_output_error=0.0`.



## 0.1.16 - 2026-06-06

- Extended hardware/tools/validate_s2_block_full.py with external weight, token, golden-output, and generated-output CLI controls.
- Reused packer tensor key resolution for external direct-matrix comparison instead of canonical synthetic keys only.
- Added test_validate_s2_block_full_external_input for external .npz weights, token input, output emission, and expected-output comparison.
- Re-ran synthetic full-block artifact generation, direct external-input validation, and static validation.
- Documented that existing .pt candidates require torch or .npz export before trained checkpoint equivalence can be claimed.


## 0.1.17 - 2026-06-06

- Ran the S2 full-block validator against hardware/refs/weights/software_initial_weights.pt under the torch-enabled HGTXR .venv.
- Added durable software-initial validation artifacts and generated per-layer output artifact under docs/resources.
- Packed software_initial_weights.pt into s2_block_first_step ABI artifacts under hardware/refs/weights with fallback_count=0.
- Updated static validation to require the software-initial validation and packed-weight artifacts.
- Documented remaining gaps: exact PyTorch golden, fixed-point quantized comparison, HLS csim, and board validation.


## 0.1.18 - 2026-06-08

- Added hardware/tools/export_s2_pytorch_golden.py for exact PyTorch block golden generation and current S2 approximation comparison.
- Added separate HGTXR_WEIGHT_BIT_WIDTH/HGTXR_WEIGHT_INT_WIDTH policy and updated packed weight unpacking to support Q4 weights with Q8 activations.
- Updated ZCU104 S2 block first-step config to HGTXR_BIT_WIDTH=8 and HGTXR_WEIGHT_BIT_WIDTH=4.
- Generated Q4 software-initial packed weight artifacts and q4w8a quantization report.
- Added HG-PIPE reference analysis for LayerNorm, GeLU, Quant, ATTN, and MLP template integration.

## 0.1.19 - 2026-06-09

- Added dedicated Q4W/Q8A Vitis HLS solution provenance for the S2 fused pre-LN block path.
- Ran Q4W/Q8A csynth and recorded report-backed ZCU104 HLS fit metrics in the csynth summary CSV.
- Documented HLS csim as blocked by the current Ubuntu/WSL header environment, while standalone g++ smoke still passes.
- Updated handoff, progress, spec, validation, and packing notes with the Q4W/Q8A resource evidence.

## 0.1.20 - 2026-06-09

- Added a separate E2E AXI-Stream Q4W/Q8A ViT shell top for DMA-facing execution.
- Added conv patch embedding, global buffer, two ATTN units, two MLP units, ATTN-first controller schedule, and MLP head shell.
- Added ZCU104 E2E define and Vitis HLS csim/csynth scripts.
- Recorded E2E shell csynth resource and latency evidence in the HLS summary CSV.

## 0.1.21 - 2026-06-09

- Added full dense ViT E2E execution path for the HGTXR/DeiT-Tiny target.
- Fixed Vitis HLS `csim_design` environment integration on Ubuntu 24.04/WSL.
- Added full-target HLS synthesis result for the E2E AXI-Stream top.

## 0.1.22 - 2026-06-09

- Tuned the full dense E2E AXI-Stream ViT path for ZCU104 HLS-estimated resource
  fit.
- Added dense-lane parallel projections and cyclic local-buffer partitioning for
  the E2E top.
- Bound the full MLP hidden buffer to URAM and raised E2E bus/buffer/FIFO
  parameters.
- Selected PAR5 after PAR16/PAR8/PAR6 exceeded LUT budget.
- Recorded PAR5 csim/csynth/static-validation evidence and updated the HLS
  resource summary CSV.

## 0.1.23 - 2026-06-09

- Reorganized HGTXR around hardware/ and software/ as the two primary source roots.
- Moved hardware configs/scripts/tools/PYNQ/generated HLS/Vivado outputs under hardware/.
- Moved software configs/scripts/tools/tests/data under software/.
- Kept .venv at the HGTXR root as the project-local virtual environment.
- Updated README, pytest configuration, Tcl/scripts, static validation, and cleanup planning paths for the new layout.

## 0.1.24 - 2026-06-09

- Added compact HG-PIPE-style LUT math helpers for GELU, shifted-score exp,
  reciprocal square-root, and quantization clamp support.
- Routed E2E LayerNorm, MLP GELU, and attention softmax exp wrappers through the
  LUT-gated cyclic math path under HGTXR_USE_HGPIPE_LUT_MATH=1.
- Added hardware/tools/validate_hgpipe_lut_math.py, pytest coverage, and the
  durable LUT validation JSON artifact.
- Re-ran C++ smoke, Vitis HLS csim, Vitis HLS csynth, pytest, and static
  validation for the selected ZCU104-fit PAR5 E2E target.
- Recorded the Spark sub-agent audit finding that head-aware attention scaling
  is the next highest-priority HG-PIPE alignment gap.

## 0.1.25 - 2026-06-09

- Added parameterized cyclic attention head controls: HGTXR_CYCLIC_HEADS, HGTXR_CYCLIC_HEAD_DIM, HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT, and HGTXR_CYCLIC_REQUIRE_HEAD_ALIGNED_ATTENTION.
- Converted S2 attention/block Q4W/Q8A configs to head-aligned 64-channel tiles for DeiT-Tiny heads=3 and head_dim=64.
- Routed S2 attention score scaling through HGTXR_CYCLIC_ATTN_SCORE_SCALE_SHIFT instead of a hard-coded zero.
- Added pytest coverage for the head-aligned S2 attention config contract.
- Fixed cyclic Vitis HLS csim setup by adding multiarch include flags, reset-project control, and csim-only system linker environment.
- Re-ran C++ smoke, Vitis HLS csim, Vitis HLS csynth, pytest, and static validation for the head-aligned S2 Q4W/Q8A block path.

## 0.1.26 - 2026-06-09

- Added hardware/hls/tb/tb_cyclic_head_attention.cpp for direct fixed-point HW/SW comparison of head-aligned attention tiles.
- Extended software/tests/test_cyclic_head_attention_config.py to compile and run the C++ comparator with the Q4W/Q8A S2 block config.
- Registered the new testbench in hardware/tools/static_validate_hgtxr.py.
- Verified pytest, standalone C++ comparator, and static validation for the new gate.

## 0.1.27 - 2026-06-09

- Added the canonical v3 hbtxr software package under software/src/hbtxr while preserving the legacy flat software modules.
- Ported v3 script and config surfaces under software/scripts/v3 and software/configs/v3.
- Added submitted-paper ZCU104 Option A reproduction config and fixed experiment config inheritance paths.
- Added reduced-depth event/track execution support, runtime selected-state outputs, reproduction manifest generation, and focused paper-reproduction tests.

## 0.1.28 - 2026-06-10

- Extended hardware/tools/validate_s2_block_full.py with multi-head attention, score-scale shift, HG-PIPE compact LUT math, and packed AXI round-trip validation.
- Updated hardware/tools/export_s2_pytorch_golden.py to use the same head-aware/scaled/LUT/packed-roundtrip approximation controls.
- Added software/tests/test_s2_block_q4_head_validator.py for fast Q4 head-aligned packed-roundtrip regression coverage.
- Generated head64 Q4 packed software-initial weight artifacts and full-block validation/PyTorch-comparison resources.
- Registered the new artifacts in static validation and re-ran full pytest, static validation, and the standalone C++ head-attention comparator.

## 0.1.29 - 2026-06-10

- Added hardware/hls/tb/tb_cyclic_s2_block_vector.cpp for direct S2 block C++ vector comparison against the Q4/head64 host golden.
- Added software/tests/test_s2_block_vector_csim.py to compile and run the comparator with the Q4W/Q8A S2 block config.
- Added f32 binary golden artifacts for comparator file I/O and registered them in static validation.
- Verified the internal S2 block wrapper across all six independent layers within Q8 activation tolerance.
- Re-ran full pytest and static validation.



## 0.1.30 - 2026-06-10

- Refactored the S2 block vector comparator so Vitis HLS can invoke it through a dedicated testbench main while native g++/pytest behavior remains intact.
- Added hardware/vivado/scripts/run_cyclic_s2_block_vector_csim.tcl for the Q4W/Q8A head64 S2 block vector csim gate.
- Fixed comparator reference-file loading from the Vitis csim/build working directory.
- Registered the Vitis wrapper/testbench artifacts in static validation.
- Verified standalone g++ comparator, Vitis HLS csim_design, targeted pytest, and static validation.


## 0.1.31 - 2026-06-10

- Converted hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp from an E2E AXIS smoke test into a deterministic vector comparator.
- Added software/tests/test_e2e_axis_vector_csim.py for the reduced Q4W/Q8A E2E comparator compile/run gate.
- Registered the new E2E vector pytest artifact in static validation.
- Verified native g++ comparator, Vitis HLS run_e2e_q4w8a_csim.tcl, targeted pytest, and static validation.


## 0.1.32 - 2026-06-10

- Added hardware/tools/validate_e2e_axis_vector.py as an independent Python software reference for the reduced E2E AXI vector gate.
- Extended software/tests/test_e2e_axis_vector_csim.py to verify the SW reference and the native C++ HLS testbench.
- Registered the new reference tool in static validation.
- Verified the reference tool, targeted pytest, Vitis HLS E2E csim, and static validation.


## 0.1.33 - 2026-06-10

- Upgraded the reduced E2E AXI vector gate to use nonzero deterministic block weights for LN, ATTN, WO, MLP, and head paths.
- Updated hardware/tools/validate_e2e_axis_vector.py to compute the nonzero reduced expected vector independently.
- Updated software/tests/test_e2e_axis_vector_csim.py expectations from {16, 0, 0, 0, 0, 0} to {23, -9, 0, 0, 0, 0}.
- Verified reference CLI, native C++ comparator, Vitis HLS E2E csim, targeted pytest, and static validation.


## 0.1.34 - 2026-06-10

- Broadened the reduced E2E AXI vector gate to deterministic four-channel grouped Q4 coverage.
- Updated the HLS testbench to exercise patch channels 0..3, grouped WO, grouped MLP W1/W2, and four head outputs.
- Updated hardware/tools/validate_e2e_axis_vector.py to compute grouped multi-channel reference values.
- Verified reference CLI, native C++ comparator, targeted pytest, Vitis HLS E2E csim, and static validation.

## 0.1.35 - 2026-06-10

- Extended the reduced E2E AXI vector gate to deterministic six-output grouped Q4 coverage.
- Updated the HLS testbench to exercise patch channels 0..5, grouped WO, grouped MLP W1/W2, and all six public head outputs.
- Updated hardware/tools/validate_e2e_axis_vector.py and software/tests/test_e2e_axis_vector_csim.py for expected_raw={18, -3, 0, -2, 0, -4}.
- Verified reference CLI, native C++ comparator, targeted pytest, Vitis HLS E2E csim, and static validation.

## 0.1.36 - 2026-06-10

- Added an isolated HG-PIPE `mlp_1_geluq` cursor-table primitive and contract entry for the 64-entry quantized GeLU table.
- Extended `validate_hgpipe_lut_math.py` to verify all 150,528 HG-PIPE GeLUQ64 input/output reference samples.
- Updated progress, validation, handover, TODO, and HG-PIPE reference notes with the GeLUQ64 integration boundary.
- Verified py_compile, HG-PIPE math contract validation, and static validation.

## 0.1.37 - 2026-06-10

- Added an isolated HG-PIPE `attn_0_q_q` quant cursor-table primitive and contract entry for the 64-entry signed quant table.
- Strengthened `validate_hgpipe_lut_math.py` with generic cursor-table ref validation, integer type range checks, raw cursor ranges, and clamp-hit metadata.
- Verified all 37,632 HG-PIPE `attn_0_qq` input/output reference samples with zero mismatches.
- Updated progress, validation, handover, TODO, and HG-PIPE reference notes with the Quant attn0_q integration boundary.

## 0.1.38 - 2026-06-10

- Added isolated HG-PIPE `attn_0_softmaxq` softmax building-block helpers for exp lookup, reciprocal lookup, and uint3 requant.
- Extended the HG-PIPE math contract with `SOFTMAX_2X1.cpp` provenance, 14 scalar fields, 32-entry exp table, and two 64-entry reciprocal tables.
- Added rowwise 3-pass softmax replay validation for all 115,248 HG-PIPE `attn_0_softmaxq` input/output samples.
- Verified py_compile, HG-PIPE math contract validation, static validation, and native E2E C++ header smoke.

## 0.1.39 - 2026-06-10

- Added isolated HG-PIPE `attn_1_lnq` LayerNorm helper primitives for fixed-point mean, 128-entry rsqrt lookup, and signed 3-bit affine requant.
- Extended the HG-PIPE math contract with `LAYERNORM_2X2.cpp` provenance, scalar/type metadata, 128-entry rsqrt table, and `attn_1_lnq_*` ref files.
- Added rowwise 3-pass LayerNorm replay validation for all 37,632 HG-PIPE `attn_1_lnq` input/output samples.
- Verified py_compile, HG-PIPE math contract validation, static validation, and native E2E C++ header smoke.

## 0.1.40 - 2026-06-10

- Extended the HG-PIPE attention-0 quant contract from Q-only to Q/K/V/A cursor-table refs.
- Added isolated HLS helpers for `attn_0_k_q`, `attn_0_v_q`, and `attn_0_a_q` 64-entry signed quant tables.
- Generalized `validate_hgpipe_lut_math.py` attention-0 quant registration across Q/K/V/A.
- Verified all four attention-0 quant slices over 150,528 total samples with zero mismatches, then re-ran static validation and native E2E header smoke.

## 0.1.41 - 2026-06-10

- Extended the HG-PIPE GeLUQ contract from MLP1-only to MLP0/MLP1.
- Added isolated HLS helper `hgtxr_hgpipe_mlp0_geluq64_int()` and preserved the existing MLP1-compatible helper alias.
- Generalized `validate_hgpipe_lut_math.py` GeLUQ registration across MLP0/MLP1.
- Verified both GeLUQ slices over 301,056 total samples with zero mismatches, then re-ran static validation and native E2E header smoke.

## 0.1.42 - 2026-06-10

- Extended the HG-PIPE GeLUQ contract to MLP10/MLP11 in addition to MLP0/MLP1.
- Added isolated HLS helpers for `mlp_10_geluq` and `mlp_11_geluq` 64-entry unsigned tables.
- Generalized `validate_hgpipe_lut_math.py` GeLUQ registration across MLP0/MLP1/MLP10/MLP11.
- Verified four GeLUQ slices over 602,112 total samples with zero mismatches, then re-ran static validation and native E2E header smoke.

## 0.1.43 - 2026-06-10

- Extended the HG-PIPE attention quant contract to `attn_1` Q/K/V/A in addition to `attn_0` Q/K/V/A.
- Generalized `validate_hgpipe_lut_math.py` attention quant registration across attention layers 0 and 1.
- Added isolated HLS helpers for `attn_1_q_q`, `attn_1_k_q`, `attn_1_v_q`, and `attn_1_a_q`.
- Verified attention layer 0/1 Q/K/V/A over 301,056 total samples with zero mismatches, then re-ran static validation and native E2E header smoke.

## 0.1.44 - 2026-06-10

- Added durable `HGTXR_E2E_SCALE=hgpipe_math_lnq` reduced gate for combined GeLUQ, SoftmaxQ, and LayerNormQ opt-in E2E math.
- Added `hardware/refs/e2e_axis_vector_hgpipe_math_lnq_spec.json` and generated `hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_golden.hpp` with expected raw `[22, -7, 6, -4, 4, -7]`.
- Extended the E2E Python mirror, C++ testbench, Vitis scale scripts, software test matrix, and static validator for the new named gate.
- Verified py_compile, reference/header sync, native C++ comparator, Vitis HLS csim, Vitis HLS csynth, and static validation for the named reduced gate.

## 0.1.45 - 2026-06-10

- Added staged `HGTXR_E2E_SCALE=hgpipe_math_lnq_active8` gate for two-block active8 GeLUQ, SoftmaxQ, and LayerNormQ E2E math.
- Added `hardware/refs/e2e_axis_vector_hgpipe_math_lnq_active8_spec.json` and generated `hardware/hls/tb/e2e_axis_vector_hgpipe_math_lnq_active8_golden.hpp` with expected raw `[37, -26, 22, -14, 18, -21]`.
- Extended the C++ testbench header selection, Vitis scale scripts, software test matrix, and static validator for the active8 gate.
- Verified py_compile, reference/header sync, native C++ comparator, Vitis HLS csim, Vitis HLS csynth, and static validation.

## 0.1.46 - 2026-06-10

- Raised the E2E active16 resource path parallelism from PAR5 to PAR8 through `HGTXR_PARALLELISM_FACTOR`, `HGTXR_E2E_ATTN_PAR`, and `HGTXR_E2E_DENSE_PAR`.
- Added packed-weight vector cache and aligned fast path for QKV, WO, MLP W1/W2, and head dense loops.
- Added local LayerNorm gamma/beta packed-word caching and mapped small attention scratch arrays `score`, `prob`, and `exp_raw` to LUTRAM.
- Verified `HGTXR_E2E_SCALE=hgpipe_math_lnq_active16`, `HGTXR_E2E_RESOURCE_POLICY=dsp_uram` CSim and CSynth.
- Captured final active16 PAR8 resource point: `42 BRAM_18K`, `128 DSP`, `19,576 FF`, `45,431 LUT`, `88 URAM`, `494,968 cycles`.

## 0.1.47 - 2026-06-10

- Added full-scale QKV packed-weight BRAM caches under `HGTXR_E2E_QKV_WEIGHT_CACHE`.
- Removed the full `active196_b6_ff768` QKV AXI weight-port `II=3` bottleneck; QKV now achieves `II=1`.
- Captured final full PAR8 `dsp_mixed_stream` resource point: `306 BRAM_18K`, `332 DSP`, `43,740 FF`, `81,144 LUT`, `64 URAM`, `71,316,968 cycles`.
- Kept small attention scratch memories on LUTRAM and large streaming Q/K/V/attn buffers on URAM.
- Verified py_compile, static validation, diff check, and Vitis HLS CSynth for the full selected policy.

## 0.1.48 - 2026-06-10

- Added `docs/track/NEXT-DECISION-2026-06-10-E2E.md` so the next E2E hardware direction is user-selected before major execution.
- Recorded option analysis for board implementation/timing, HLS timing pre-tune, further parallelism, ZCU104/PYNQ hardware validation, paper-trained weight/LUT alignment, and software paper-reproduction fixes.
- Updated HANDOVER, PROGRESS, TODO, CONVERSATION, and log tracking files with the choice rule.

## 0.1.49 - 2026-06-10

- Updated `docs/Master-Plan.md` and `docs/Sub-Plan.md` with the current E2E decision DAG, prompt brief, Task Cards, file ownership, and validation gates.
- Captured Option A sub-agent audit results: Spark quota exhausted, GPT5.5 A1/A2 read-only audits completed.
- Refined A1 with AXI DMA BD block requirements, one-256-bit-beat-per-pixel packing, `2,097,152` byte DMA input length risk, and E2E IP regeneration requirement.
- Refined A2 with the recommended `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)` wrapper contract and fresh CSim/CSynth gates.
- Updated HANDOVER, PROGRESS, Execution, Validation, and TODO tracking for the A1/A2/A3 sub-choice.

## 0.1.50 - 2026-06-10

- Added `docs/track/GOAL-AUDIT-2026-06-10-THIRD-GOAL.md` with a requirement-by-requirement audit of active `[3차목표]` items (0)..(11).
- Recorded that the current HLS E2E point is strong but the full goal remains incomplete until board implementation/PYNQ runtime and final paper artifact gates pass.
- Confirmed `spec-kit` and `specify` are unavailable in the current Ubuntu shell.
- Confirmed the requested `PAPER_PRJXR/05_RESOURCES/DeiT-Tiny C-Syn Results.png` path is absent while an HGPIPE image with the same filename exists.
- Confirmed `/home/kjm26/project/PRJXR/XR-VITs` is absent in this workspace.
- Updated HANDOVER, PROGRESS, and Validation with the audit result and next-choice dependency.

## 0.1.51 - 2026-06-10

- Added branch-neutral `hardware/tools/check_third_goal_preflight.py` for repeatable third-goal environment/artifact readiness checks.
- Added durable preflight result `docs/resources/third_goal_preflight_2026_06_10.json` with neutral `ok=22`, `warn=8`, `fail=0`.
- Registered the new checker and JSON/audit artifacts in `hardware/tools/static_validate_hgtxr.py`.
- Confirmed the checker warns, rather than fails, for choice-gated items such as E2E `component.xml`, generated-tree csynth report, old `hgtxr_top` board-flow mismatch, missing spec-kit, missing PAPER_PRJXR image, and missing `XR-VITs`.
- Added board-ready mode that fails stale/missing E2E board artifacts; current board-ready result is `ok=22`, `warn=5`, `fail=3` because the package/build/PYNQ artifacts still belong to old `hgtxr_top`.

## 0.1.52 - 2026-06-10

- Added `hardware/tests/test_check_third_goal_preflight.py` with fake-tree unit coverage for neutral versus board-ready preflight behavior.
- Covered old-flow warning behavior, board-ready stale artifact failure behavior, and selected E2E artifact pass behavior.
- Registered the new unit test file in `hardware/tools/static_validate_hgtxr.py`.
- Verified `python3 -m unittest tests/test_check_third_goal_preflight.py` and py_compile for the checker, static validator, and test module.

## 0.1.53 - 2026-06-10

- Extended `hardware/tests/test_check_third_goal_preflight.py` to cover `--json-out` artifact generation.
- Verified JSON shape for `mode`, `summary`, `checks`, selected E2E HWH status, E2E `component.xml` status, and board-ready failure status.
- Added stdout/JSON summary matching, check schema validation, and `shutil.which` isolation after GPT5.5 review.
- Confirmed the preflight unit suite now passes 5 tests and still avoids HLS/Vivado execution.

## 0.1.54 - 2026-06-10

- Implemented A2 memory-mapped E2E wrapper `hgtxr_e2e_m_axi_top`.
- Added m_axi E2E CSim/CSynth Tcl wrappers and registered them in static validation.
- Verified A2 reduced/full CSim and full CSynth for `active196_b6_ff768`, `dsp_mixed_stream`.
- Added A1 small-memory LUTRAM binding for `gb.pooled`.
- Added opt-in QKV weight-cache URAM switch, default off after confirming all-Q/K/V cache URAM use overflows ZCU104.
- Added C-path `HGTXR_E2E_PAR=16|32` AXIS sweep knob with invalid `12/24` rejection.
- Verified C PAR=16 full AXIS CSynth and reduced CSim; PAR=16 raises DSP to `604` and lowers latency to `37,508,072` cycles.

## 0.1.55 - 2026-06-10

- Hardened Ubuntu toolchain configuration by adding `HGTXR_XILINX_ROOT`, `HGTXR_GCC_INCLUDE`, `HGTXR_GCC_LIB`, `HGTXR_SYS_INCLUDE`, and `HGTXR_SYS_LIB` overrides to core HLS Tcl flows while keeping `/tools/Xilinx` as the default.
- Fixed `run_cyclic_s2_block_vector_csim.tcl` root detection so it can run from either repo root or `hardware/`.
- Extended the third-goal preflight checker to recognize both AXIS and A2 `m_axi` generated E2E trees and to require the selected A2 wrapper interface markers in board-ready mode.
- Extended static validation with E2E AXIS/m_axi marker checks and a warning-only spec-kit/specify availability check.
- Updated Master Plan, Sub Plan, Spec, HANDOVER, and Goal Audit to reflect A2 completion, C PAR16 validation, Spark quota fallback, and current Ubuntu root/toolchain assumptions.

## 0.1.56 - 2026-06-10

- Added A2 board-flow scaffold `hardware/vivado/scripts/package_e2e_m_axi_ip.tcl` for packaging `hgtxr_e2e_m_axi_top` as IP.
- Added separate `hardware/vivado/scripts/build_e2e_m_axi_bitstream.tcl` so the E2E m_axi board path does not overwrite the legacy `hgtxr_top` Vivado flow.
- Added PYNQ runtime helper `hardware/pynq/hgtxr/e2e_m_axi_overlay.py` plus package exports/docs for the memory-mapped E2E wrapper.
- Registered the new A2 package/build/runtime files in static validation.
- Verified Python compile, static validation, and whitespace checks; long Vivado package/implementation was not run in this step.

## 0.1.57 - 2026-06-10

- Completed the selected A2 E2E memory-mapped board artifact path.
- Exported `hgtxr_e2e_m_axi_top` as HLS IP and generated a separate Vivado E2E m_axi bitstream.
- Packaged `hgtxr_e2e_m_axi.bit/.hwh` under `hardware/pynq/hgtxr/` without replacing legacy `hgtxr.bit/.hwh`.
- Updated board-ready preflight to validate E2E m_axi artifacts and scripts; board-ready now reports `ok=25`, `warn=6`, `fail=0`.
- Updated PYNQ HWH parsing/default-weight behavior and related tests.

## 0.1.58 - 2026-06-10

- Added `hardware/pynq/hgtxr/run_e2e_m_axi_smoke.py` for board-side A2 PYNQ runtime smoke.
- Added live-weight and zero-weight smoke modes with runtime-state checks and JSON output.
- Extended PYNQ unit tests to cover the smoke CLI, repo-root import path, JSON output, and runtime-state pass/fail behavior.
- Registered the smoke script in static validation and documented the board commands in the PYNQ README.

## 0.1.59 - 2026-06-10

- Added `hardware/pynq/hgtxr/e2e_m_axi_weights.py` to build CSim-mirrored active196_b6_ff768 packed Q4 weights for A2 board smoke.
- Added `--weights-mode golden` to the A2 PYNQ smoke CLI; it checks `runtime_state=2` and output raw `[32, -13, 26, -6, 14, -11]`.
- Extended PYNQ tests to 14 tests covering signed Q4 packing, lane boundaries, known layout offsets, full buffer capacity, tail-zero behavior, and golden smoke output matching.
- Registered the weight builder in static validation and documented the golden board command in the PYNQ README.

## 0.1.60 - 2026-06-10

- Added `hardware/tools/export_e2e_m_axi_weights.py`.
- Exported `hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin` and its manifest for A2 file-based board smoke.
- Added `--weights-mode file --weights-bin` to the A2 PYNQ smoke CLI.
- Extended PYNQ tests to 16 tests covering export manifest validation and file-mode binary loading.

## 0.1.61 - 2026-06-10

- Extended `hardware/tools/check_third_goal_preflight.py` with A2 packed Q4 weight artifact integrity checks.
- Board-ready preflight now validates weight manifest fields, byte count, SHA256, expected raw/runtime contract, A2 layout constants, size math, tail-zero region, and known Q4 nibble probes.
- Extended preflight unit tests to 6 tests with missing-weight-artifact board-ready failure coverage.
- Board-ready preflight now reports `ok=36`, `warn=6`, `fail=0`.

## 0.1.62 - 2026-06-10

- Completed A1 E2E AXIS/DMA IP export and Vivado bitstream build.
- Fixed the A1 AXI DMA integration script so DMA MM2S/S2MM stream widths match the HLS 256-bit AXIS ports.
- Copied `hgtxr_e2e_axis_dma.bit/.hwh` to `hardware/pynq/hgtxr/`.
- Updated board-ready preflight to check both A2 m_axi and A1 AXIS/DMA PYNQ artifacts.
- Board-ready preflight now reports `ok=39`, `warn=6`, `fail=0`.
- Documented A1 timing/resource evidence and remaining physical ZCU104 DMA smoke gap.

## 0.1.107 - 2026-06-11

- Added `hardware/tools/write_final_evidence_manifest.py`.
- Added `hardware/tests/test_write_final_evidence_manifest.py`.
- Integrated `final-evidence-manifest` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the manifest tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_evidence_manifest_2026_06_10.json/.md`.
- Final runner summary now reports `final_evidence_manifest_status=pass`.
- Manifest hashes 28 stable final evidence artifacts and excludes final summary JSON/Markdown as volatile to avoid self-referential hash churn.

## 0.1.108 - 2026-06-11

- Added `hardware/tools/validate_final_signoff_bundle.py`.
- Added `hardware/tests/test_validate_final_signoff_bundle.py`.
- Integrated `final-signoff-bundle-validation` into `hardware/tools/run_third_goal_final_signoff.py` before evidence manifest generation.
- Registered the bundle validator tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_signoff_bundle_validation_2026_06_10.json/.md`.
- Final runner summary now reports `final_bundle_validation_status=pass`.
- Evidence manifest now hashes 30 required stable artifacts, including final bundle validation.

## 0.1.109 - 2026-06-11

- Added `hardware/tools/check_final_blocker_closure_readiness.py`.
- Added `hardware/tests/test_check_final_blocker_closure_readiness.py`.
- Integrated `final-blocker-closure-readiness` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the closure-readiness tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_blocker_closure_readiness_2026_06_10.json/.md`.
- Final runner summary now reports `final_blocker_closure_status=blocked`.
- Evidence manifest now hashes 32 required stable artifacts, including final blocker closure readiness.

## 0.1.110 - 2026-06-11

- Added `hardware/tools/discover_c3b_smoke_candidates.py`.
- Added `hardware/tests/test_discover_c3b_smoke_candidates.py`.
- Integrated `c3b-smoke-candidate-discovery` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the C3b smoke discovery tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/c3b_smoke_candidate_discovery_2026_06_10.json/.md`.
- Final runner summary now reports `c3b_smoke_discovery_status=missing`.
- Evidence manifest now hashes 34 required stable artifacts, including C3b smoke candidate discovery.

## 0.1.118 - 2026-06-11

- Added `hardware/tools/validate_final_unblock_closeout_packet.py`.
- Added `hardware/tests/test_validate_final_unblock_closeout_packet.py`.
- Integrated `final-unblock-closeout-validation` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered the closeout validator tool/test and docs mirrors in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/final_unblock_closeout_packet_validation_2026_06_10.json/.md`.
- Final runner summary now reports `final_unblock_closeout_validation_status=pass`.
- Evidence manifest now hashes 40 required stable artifacts and 17 consistency checks, including closeout validation.

## 0.1.119 - 2026-06-11

- Extended `hardware/tools/check_final_blocker_closure_readiness.py`.
- Added `dry_run_final_runner_command` so supplied C3b/XR-VITs candidates can be rehearsed without importing the smoke JSON or writing an active replacement policy.
- Fixed custom replacement-path forwarding in generated final runner commands.
- Updated `hardware/tests/test_check_final_blocker_closure_readiness.py`.
- Regenerated `docs/resources/final_blocker_closure_readiness_2026_06_10.json/.md` and downstream final signoff evidence.
- Final evidence manifest remains `pass`, required `40/40`, consistency checks `17`.

## 0.1.120 - 2026-06-11

- Added `hardware/tools/write_final_unblock_intake.py`.
- Added `hardware/tests/test_write_final_unblock_intake.py`.
- Integrated `final-unblock-intake` into `hardware/tools/run_third_goal_final_signoff.py`.
- Added `docs/resources/final_unblock_intake_2026_06_10.json/.md`.
- Updated the closeout packet to hash the final unblock intake report.
- Updated the final evidence manifest to require 42 stable artifacts and 19 consistency checks.
- Final runner now reports `final_unblock_intake_status=blocked` until real C3b smoke JSON and XR-VITs exact/replacement inputs are supplied.

## 0.1.121 - 2026-06-11

- Added `hardware/tools/write_c3b_smoke_result_contract.py`.
- Added `hardware/tests/test_write_c3b_smoke_result_contract.py`.
- Integrated `c3b-smoke-result-contract` into `hardware/tools/run_third_goal_final_signoff.py`.
- Updated `hardware/tools/write_final_unblock_closeout_packet.py` so the closeout packet hashes the C3b smoke result contract JSON/Markdown.
- Updated `hardware/tools/write_final_evidence_manifest.py` to require 44 stable artifacts and 22 consistency checks.
- Registered the contract tool/test and docs resources in `hardware/tools/static_validate_hgtxr.py`.
- Generated and mirrored `docs/resources/c3b_smoke_result_contract_2026_06_10.json/.md`.
- Final runner now reports `c3b_smoke_contract_status=pass` while final status remains blocked by the real board-smoke and XR-VITs inputs.

## 0.1.122 - 2026-06-11

- Updated `hardware/tools/write_final_unblock_commands.py` so the command card reads the C3b smoke result contract.
- Updated `hardware/tools/run_third_goal_final_signoff.py` to pass the generated contract into the command-card step.
- Updated `hardware/tools/write_final_unblock_closeout_packet.py` so the closeout packet includes a `c3b_smoke_contract` summary.
- Updated `hardware/tools/validate_final_unblock_closeout_packet.py` to validate contract status, preset, variant, canonical path, expected runtime, expected output, and validate/import commands.
- Raised final evidence manifest closeout-validation consistency threshold from `22` to `29` checks.
- Regenerated final signoff evidence; closeout validation now reports `pass`, checks `29/29`, and evidence manifest remains `pass`, required `44/44`.

## 0.1.123 - 2026-06-11

- Added `hardware/tools/write_e2e_resource_policy_audit.py`.
- Added `hardware/tests/test_write_e2e_resource_policy_audit.py`.
- Integrated `resource-policy-audit` into `hardware/tools/run_third_goal_final_signoff.py` after the resource matrix step.
- Registered resource policy audit tool/test/docs resources in `hardware/tools/static_validate_hgtxr.py`.
- Added `docs/resources/e2e_resource_policy_audit_2026_06_10.json/.md`.
- Updated final evidence manifest to require 46 stable artifacts and validate the resource policy audit with 17 checks.
- Resource policy audit now verifies DSP bind-op, URAM bind-storage, LUTRAM bind-storage, ZCU104 PAR/FIFO defaults, C3b PAR16/MEM16 selection, C3b DSP increase vs A1, C3b LUT reduction vs C1, positive URAM, and unchanged C3b latency vs C1.

## 0.1.124 - 2026-06-11

- Added `hardware/tools/write_selected_path_execution_audit.py`.
- Added `hardware/tests/test_write_selected_path_execution_audit.py`.
- Integrated `selected-path-execution-audit` into `hardware/tools/run_third_goal_final_signoff.py`.
- Registered selected-path audit tool/test/docs resources in `hardware/tools/static_validate_hgtxr.py`.
- Added `docs/resources/selected_path_execution_audit_2026_06_10.json/.md`.
- Updated final evidence manifest to require 48 stable artifacts and validate the selected path audit with 22 checks.
- Selected-path audit now verifies user-selected Path 1 as `A2 then A1`, Path 2 as `C` with C3b candidate, and `E=pending`.

## 0.1.125 - 2026-06-11

- Added `hardware/tools/write_spec_plan_conformance_audit.py`.
- Added `hardware/tests/test_write_spec_plan_conformance_audit.py`.
- Integrated `spec-plan-conformance-audit` into `hardware/tools/run_third_goal_final_signoff.py` after requirements trace generation.
- Registered spec/plan conformance audit tool/test/docs resources in `hardware/tools/static_validate_hgtxr.py`.
- Added `docs/resources/spec_plan_conformance_audit_2026_06_10.json/.md`.
- Updated final evidence manifest to require 50 stable artifacts and validate the spec/plan conformance audit with 32 checks.
- Spec/plan conformance audit now verifies manual spec-kit fallback, ZCU104/Q4W/Q8A/parameter coverage, selected A2/A1/C/E plan coverage, requirements trace IDs 0..11, completion audit shape, evidence manifest status, selected-path audit, and resource-policy audit.
