# HGTXR Third-Goal Choices

Date: 2026-06-16

## Current Decision State
- Selected execution path remains `(1) A2 after A1 preparation`, `(2) C path`, `(3) E pending`.
- Spec-plan anchor: E: pending; Selected Path Execution Audit remains the active continuity gate.
- Current final signoff is blocked by two external inputs:
  - C3b AXIS/DMA physical smoke JSON.
  - Exact `/home/kjm26/project/PRJXR/XR-VITs` source or approved replacement policy.
- Req2 is reflected through manual Spec fallback plus Spark-first/GPT5.5 fallback evidence because `spec-kit`/`specify` are not on PATH.
- QKV URAM successor is optional promotion evidence, not a default final-signoff blocker.

Local blocker snapshot command, no board run and no policy write:

```sh
python3 tools/check_final_blocker_closure_readiness.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --c3b-no-require-paths --json-out /tmp/hgtxr_final_blocker_readiness_current.json --markdown-out /tmp/hgtxr_final_blocker_readiness_current.md
```

## Req11 XR-VITs Decision Options
| Option | Action | Expected Result | Risk | Status |
|---|---|---|---|---|
| X1 | Restore exact `/home/kjm26/project/PRJXR/XR-VITs` checkout | Req11 clears as exact-source evidence; no replacement policy needed | Requires locating/restoring missing repository | available if source can be restored |
| X2 | Approve `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel` as replacement source | Req11 clears through explicit replacement policy while preserving provenance | Weaker than exact XR-VITs source; needs explicit user approval metadata | candidate-ready-needs-approval |
| X3 | Keep Req11 blocked | No policy written; implementation can continue on non-Req11 work | Final third-goal signoff cannot be claimed | current state |
| X4 | Require replacement-policy preview path contract | Preview must target default active policy path and tracked candidate-audit path before approval | Does not approve policy; only hardens pre-approval evidence | implemented-active |
| X5 | Require replacement-policy approval-event contract | Preview/approved policy must bind actor/time/reason/path/candidate audit in a hashed approval event | Does not approve policy; increases schema/check count to prevent stale or forged approval metadata | implemented-active |
| X6 | Reject placeholder approver on active policy write | Dry-run review keeps `<approved-by>` placeholders, but active replacement policy cannot be written with placeholder actor metadata | Requires real approver identifier when user chooses active replacement-policy path | implemented-active |
| X7 | Promote X6 source-contract proof into final evidence | Final manifest checks policy-tool placeholder guard, dry-run-only allowance, and final-runner active failure blocking | Source-contract check does not execute an active policy write; must stay aligned with tests | selected-active |

Dry-run command for X2:

```sh
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by USER --reason "User approved XR_Accel as XR-VITs replacement for HGTXR third-goal continuation" --dry-run
```

Active command for X2 after explicit approval:

```sh
python3 tools/create_xr_vits_replacement_policy.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by USER --reason "User approved XR_Accel as XR-VITs replacement for HGTXR third-goal continuation"
```

## C3b Physical Smoke Decision Options
| Option | Action | Expected Result | Risk | Status |
|---|---|---|---|---|
| C1 | Run prepared C3b bundle on ZCU104 and import JSON | Final C3b board gate clears if result validates | Requires board access and copied result JSON | preferred |
| C2 | Continue without board JSON | Keeps local docs/tools moving | Final signoff remains blocked | current fallback |
| C3 | Keep canonical-only signoff gate | Generated bundle/remote result JSON cannot clear C3b until imported to canonical path | Requires explicit import step after board run | active |
| C4 | Require preset-matched bit/hwh basenames inside smoke JSON | Correct numeric output cannot clear C3b if JSON points to wrong overlay files | Board script overrides must use matching artifact names | implemented-active |

Canonical C3b result path:

```text
pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json
```

Import command after board run:

```sh
python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out /tmp/hgtxr_c3b_smoke_import.json --validation-out /tmp/hgtxr_c3b_smoke_validation.json
```

## QKV URAM Successor Promotion Options
| Option | Action | Expected Result | Risk | Status |
|---|---|---|---|---|
| Q1 | Keep QKV URAM as successor-only evidence | Default final signoff remains tied to C3b + XR-VITs unblock | QKV resource improvement is not promoted to replacement | current default |
| Q2 | Run QKV URAM board smoke on ZCU104 and import JSON | QKV successor promotion gate clears if result validates | Requires board access; does not replace C3b unless explicitly selected later | ready-for-board |
| Q3 | Treat QKV URAM physical smoke as hard gate during signoff | Final runner reports QKV physical smoke as blocker until valid JSON exists | Adds an optional blocker to default closeout | available |
| Q4 | Link QKV URAM successor branch/resource evidence into VREF-P0-02 buffer-placement audit | Final evidence proves that the URAM branch exists and that successor HLS/resource/route evidence supports it without claiming physical smoke | Does not clear board-smoke blocker | implemented-active |

Canonical QKV URAM result path:

```text
pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json
```

QKV URAM dry-run and import commands:

```sh
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --require-qkv-uram-physical-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --dry-run-import-qkv-uram-smoke --allow-blocked
python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --import-qkv-uram-smoke-json /path/to/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json --require-qkv-uram-physical-smoke --allow-blocked
```

## XR-VITs Policy Integrity Options
| Option | Action | Expected Result | Risk | Status |
|---|---|---|---|---|
| P1 | Keep current approval policy schema | Backward-compatible with existing policy files | Candidate audit content can drift after approval | current |
| P2 | Add candidate-audit fingerprint and recommendation snapshot to replacement policy | Approval binds to exact candidate audit content; tampering/drift is detected | Legacy policy files must be regenerated before they can clear final signoff | implemented-active |
| P3 | Require fresh exact `/XR-VITs` checkout only | Strongest provenance, no replacement-policy ambiguity | Blocks progress until exact source exists | available |

P2 operational propagation:
- Final unblock command card includes `xr_vits_policy_integrity` and embeds it into U2b/U4b.
- Final unblock closeout packet and final operator handoff carry policy path, candidate-audit path, required fingerprint fields, and validation status.
- Closeout, handoff, and final bundle validators accept current `pending-policy-creation` only while Req11 is still blocked and reject legacy/drifted policy files once a policy exists.

## Next Internal Work Options
| Option | Action | Expected Result | Risk | Status |
|---|---|---|---|---|
| N1 | Add Req6 legality checks for parameter combinations | Req6 proves not only knob coverage but HLS-legal relationships among bus width, bit widths, parallelism, model dimension, FF dimension, buffer size, and FIFO depth | Audit-only guard first; still not exhaustive HLS sweep | implemented-active |
| N2 | Extend Req6/sweep parallelism space to PAR16/PAR32 | C3b PAR16 resource evidence is reflected in machine-readable sweep space; PAR32 becomes explicit exploratory candidate | PAR32 needs fresh csynth/routed evidence before any promotion | implemented-active |
| N3 | Align blocker-readiness no-require-paths semantics | `--c3b-no-require-paths` controls both candidate and current C3b validation, reducing false-negative dry-run snapshots | Relaxed mode can pass path-less current JSON only when explicitly requested; strict mode unchanged | implemented-active |
| N4 | Integrate C3b transfer manifest generation into final runner | Transfer manifest is regenerated and mirrored on every final signoff run | More runner churn; still cannot close physical board blocker without result JSON | implemented-active |
| N7 | Promote C3b transfer manifest semantic gates into final evidence | Final evidence fails if the transfer manifest is stale, wrong-preset, malformed, hash-mismatched, or inconsistent with board readiness | More strict evidence contract; still cannot close physical board blocker without result JSON | implemented-active |
| N8 | Promote PAR32 exploratory policy into Req6/final evidence gates | C3b PAR16 remains validated; PAR32 cannot be treated as default without fresh csynth, routed timing, resource-fit audit, and no-C3b-overwrite evidence | Does not create PAR32 hardware evidence by itself | implemented-active |
| N9 | Promote manual Spec fallback contract into final evidence | Final evidence fails if spec-plan no longer proves spec-kit/manual fallback, ZCU104, Q4W/Q8A, parameter knobs, A2/A1/C/PAR16, or `/tools/Xilinx` | Does not install spec-kit; hardens manual fallback until tool exists | implemented-active |
| N10 | Dedupe final-runner `mirrored_artifacts` summary | Final signoff summary has one canonical entry per mirrored output path, improving audit readability and incremental diff checks | Does not close C3b/XR-VITs blockers; metadata hygiene only | implemented-active |
| N11 | Promote P2-ViT SW-first scale calibration report into final evidence | Req5 Q4/Q8 SW-HW match and PoT scale sweep become one explicit calibration decision: keep current C3b scales; non-current rows stay successor-only | Does not improve board accuracy claim without physical smoke; report-level gate only | implemented-active |
| N14 | Promote Req5 packed Q4 binary/manifest integrity into final evidence | Final evidence fails if the packed Q4 binary is missing, SHA256/byte-count drift from the manifest, expected runtime/C3b output changes, or strict TB/CSim evidence regresses | Does not close physical board-smoke blocker; hardens software/manifest evidence only | implemented-active |
| N15 | Refresh final evidence consistency threshold to live count | Operator handoff and bundle validators reject contracts below current `312` consistency checks instead of old `278` threshold | Requires fixture/docs refresh whenever final evidence gates expand again | implemented-active |
| N16 | Require Req5 packed-weight checks in final bundle validation | Final signoff bundle fails if the final evidence manifest drops any Req5 Q4/Q8 packed-weight consistency check | Does not add new HLS/board evidence; validator-level hardening only | implemented-active |
| N17 | Require Req5 packed-weight checks in operator handoff validation | Operator handoff validation fails if live final evidence drops any Req5 Q4/Q8 packed-weight consistency check while preserving count | Does not close C3b/XR-VITs blockers; hardens operator handoff evidence only | implemented-active |
| N18 | Require latest operator-handoff check count in final bundle validation | Final signoff bundle rejects stale operator-handoff validation artifacts below the current `check_count=66` | Does not add new HLS/board evidence; closes validator freshness gap only | implemented-active |
| N19 | Apply cyclic baseline URAM/LUTRAM storage policy | Legacy cyclic weight tiles and large temporaries use URAM; small tile scratch uses LUTRAM through parameterized macros and resource-policy gates | Requires fresh HLS csynth before claiming post-change physical resource totals | implemented-active |
| N20 | Require cyclic resource checks by name in final validators | Operator handoff and bundle validation reject manifests that preserve count `321` while dropping cyclic URAM/LUTRAM checks | Validator-level hardening only; no new csynth resource totals | implemented-active |
| N21 | Promote C3b LUT/latency/WNS thresholds into final gates | Resource-policy and final evidence fail if C3b LUT, latency, or routed WNS drift beyond protected baseline thresholds | Gate-level hardening only; does not capture physical board smoke | implemented-active |
| N22 | Require C3b threshold checks by name in final validators | Operator handoff and bundle validation reject manifests that preserve count `321` while dropping C3b LUT/latency/WNS threshold checks | Validator-level hardening only; no new physical evidence | implemented-active |
| N23 | Require final-runner blocker/count/mirror schema consistency | Final evidence fails if runner blocker count/detail count or mirrored-artifact count fields diverge from actual lists | Host-side evidence hardening only; external blockers unchanged | implemented-active |
| N24 | Require final-runner summary count checks by name in final validators | Operator handoff and bundle validation reject manifests that preserve count `325` while dropping runner blocker/count/mirror consistency checks | Validator-level hardening only; external blockers unchanged | implemented-active |
| N25 | Refresh closeout validation freshness target to current count | Final evidence rejects stale final-unblock closeout validation artifacts below current `check_count=49` | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N26 | Refresh source-audit freshness target to current counts | Final evidence rejects stale third-goal source audit artifacts below required `86` or sources `224` | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N27 | Require source-audit freshness checks by name in final validators | Operator handoff and bundle validation reject manifests that preserve count `326` while dropping source-audit freshness checks | Validator-level hardening only; external blockers unchanged | implemented-active |
| N28 | Refresh validator minimum consistency threshold to live count | Operator handoff and bundle validation reject final evidence contracts below current `332` consistency checks | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N29 | Refresh validator minimum required-count threshold to live count | Operator handoff and bundle validation reject final evidence contracts below current required `86` artifacts | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N30 | Directly validate live source-audit JSON in final validators | Operator handoff and bundle validation reject stale `third_goal_source_audit_2026_06_16.json` even if final-manifest check names/counts are preserved | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N31 | Directly validate live resource-policy JSON in final validators | Operator handoff and bundle validation reject stale `e2e_resource_policy_audit_2026_06_10.json` even if final-manifest resource check names/counts are preserved | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N32 | Directly validate live spec-plan conformance JSON in final validators | Operator handoff and bundle validation reject stale `spec_plan_conformance_audit_2026_06_10.json` even if final-manifest spec-plan check names/counts are preserved | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N33 | Gate validator-count freshness in spec-plan current docs | Spec-plan conformance rejects current docs that omit live operator handoff `116/116` or bundle `137/137`, or carry stale validator count anchors | Documentation/validator freshness hardening only; external blockers unchanged | implemented-active |
| N34 | Directly validate live final-runner JSON in final validators | Operator handoff and bundle validation reject stale `third_goal_final_signoff_run_2026_06_10.json` blocker/mirror counts even if final-manifest runner checks remain present | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N35 | Directly validate live Req6 parameterization JSON in final validators | Operator handoff and bundle validation reject stale `req6_parameterization_audit_2026_06_16.json` knob/count/PAR16/PAR32 policy drift even if final-manifest Req6 checks remain present | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N36 | Directly validate live current-audit JSON in final validators | Operator handoff and bundle validation reject stale `third_goal_current_audit_2026_06_16.json` summary/blocker path/XR policy/QKV optional-state drift | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N37 | Gate stale XR policy-count/current-doc anchors in spec-plan conformance | Spec-plan conformance rejects current docs that still report XR policy check count `7` or old validator targets after live current audit advanced to policy count `9`, operator `116/116`, bundle `137/137` | Documentation freshness hardening only; external blockers unchanged | implemented-active |
| N38 | Directly validate live closeout-validation JSON in final validators | Operator handoff and bundle validation reject stale `final_unblock_closeout_packet_validation_2026_06_10.json` count/core-check/safety drift even if final-manifest freshness checks remain present | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N39 | Directly validate live closeout-packet JSON in final validators | Operator handoff and bundle validation reject stale `final_unblock_closeout_packet_2026_06_10.json` status/blocker/board/C3b/XR/QKV/safety drift even if final-manifest and closeout-validation checks remain present | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N40 | Directly validate live C3b/XR blocker gate audits in final validators | Operator handoff and bundle validation reject stale `c3b_physical_smoke_gate_audit_2026_06_16.json` or `xr_vits_gate_audit_2026_06_16.json` blocker/path/candidate/safety drift even if manifest gate checks remain present | Validator freshness hardening only; external blockers unchanged | implemented-active |
| N42 | Cross-check final-unblock intake against operator unblock plan | Final evidence and validators reject `next_inputs` or `operator_sequence` drift between final unblock intake and final blocker closure readiness | Validator/evidence hardening only; external blockers unchanged | implemented-active |
| N43 | Gate final-runner docs/resources mirror integrity | Final evidence and validators reject stale/missing docs evidence mirrors by requiring generated/signoff to docs/resources sha256/size match and live mirror-integrity fields | Evidence-path freshness hardening only; external blockers unchanged | implemented-active |
| N44 | Gate HANDOVER raw validator summary freshness | Spec-plan conformance rejects HANDOVER text that keeps stale raw `checks` counts for final operator handoff or bundle validation even when `131/131` and `152/152` anchors are present | Documentation/evidence freshness hardening only; external blockers unchanged | implemented-active |
| N45 | Gate resource-policy artifact singularity | Final evidence and validators reject noncanonical `e2e_resource_policy_audit_*` siblings, filename/payload date drift, and generated/docs hash mismatch | Evidence freshness hardening only; stale generated `2026_06_15` pair removed; external blockers unchanged | implemented-active |
| N41 | Add final-blocker operator unblock plan and final-evidence gates | `final_blocker_closure_readiness` now emits required inputs plus dry-run/active command plan; final evidence rejects missing or unsafe operator plan drift | Readiness hardening only; external C3b/XR inputs still required | implemented-active |
| N5 | Extend optional QKV board-smoke runner evidence | Stronger operator handoff for QKV board run | Requires board to close; can only improve runbooks locally | available |
| N6 | Continue C3b/XR-VITs external unblock preparation | Better operator instructions for the two remaining hard blockers | Cannot close without external JSON/source/policy approval | available |

Current internal choice:
- Selected `Q4` for QKV successor audit linkage.
- Selected and completed `N1` for this continuation turn.
- Selected and completed `N2` for this continuation turn: sweep parallelism values now include `[1, 2, 4, 8, 16, 32]`; C3b PAR16 remains validated, PAR32 remains exploratory.
- Selected and completed `N3` for this continuation turn: final-blocker readiness no-require-paths mode now applies to current canonical C3b validation as well as candidate validation.
- Selected and completed `N4` for this continuation turn: final runner now regenerates and mirrors C3b transfer manifest JSON/Markdown/SHA256 evidence before final manifest refresh.
- Selected and completed `N7` for this continuation turn: final evidence now checks C3b transfer manifest semantic fields and cross-readiness alignment.
- Selected and completed `N8` for this continuation turn: Req6 now parses sweep YAML `parallelism_extensions`; final evidence checks C3b PAR16 validated-resource-matrix status and PAR32 exploratory promotion rules.
- Selected and completed `X4` for this continuation turn: final evidence now checks XR-VITs replacement-policy preview active policy path, candidate-audit relative path, and integrity candidate-audit absolute path; live manifest is `84/84`, consistency `266`, failed consistency `0`.
- Selected and completed `X5` for this continuation turn: final evidence now checks XR-VITs replacement-policy preview approval-event schema, policy-field match, and event-id hash; live manifest is `84/84`, consistency `269`, failed consistency `0`.
- Selected and completed `X6` for this continuation turn: active XR-VITs replacement-policy creation rejects placeholder approver metadata; dry-run preview remains side-effect-free and placeholder-tolerant. Focused policy/runner tests pass, and the final runner remains expected-blocked only on external C3b physical smoke and XR-VITs source/policy.
- Selected and completed `X7` for this continuation turn: final evidence manifest now hard-gates the X6 source contract; live manifest is `84/84`, consistency `272`, failed consistency `0`, and final runner remains expected-blocked only on external C3b physical smoke and XR-VITs source/policy.
- Selected and completed `N9` for this continuation turn: final evidence manifest now hard-gates the manual Spec fallback/spec-plan contract; live manifest is `84/84`, consistency `278`, failed consistency `0`, and final runner remains expected-blocked only on external C3b physical smoke and XR-VITs source/policy.
- Selected and completed `N10` for this continuation turn after Spark sidecar audit found duplicate `mirrored_artifacts` entries in the final-runner summary. Live summary now has `92` mirrored paths, `92` unique paths, and `0` duplicates after later evidence artifacts were added; final signoff state remains expected-blocked only on external C3b physical smoke and XR-VITs source/policy.
- Selected and completed `N11` for this continuation turn: P2-ViT SW-first scale calibration now has a dedicated report and final evidence gates. Live manifest is `86/86`, consistency `283`, failed consistency `0`; report status is `pass`, checks `10/10`.
- Selected and completed `N12` for this continuation turn: Validation, CHOICE, and log are now part of current-doc freshness conformance. Spec-plan conformance is `66/66`; final manifest remains `86/86`, consistency `297`, failed consistency `0`.
- Selected and completed `N13` for this continuation turn: RMU/SMU projection and relation-stage multiply-heavy paths now use DSP-bound helpers; resource-policy audit is `27/27`; final manifest remains `86/86`, consistency `302`, failed consistency `0`.
- Selected and completed `N14` for this continuation turn: Req5 packed Q4 binary/manifest integrity is now final-evidence gated through binary existence, SHA256 match, byte-count match, expected runtime state, expected C3b raw output, strict testbench golden compare, and C3b/VREF/QKV CSim checks. Req5 audit is `11/11`; final manifest target is `86/86`, consistency `312`, failed consistency `0`.
- Selected and completed `N15` for this continuation turn: final operator handoff and final signoff bundle validators now require evidence contract consistency count `>=312`; stale `311` contracts fail.
- Selected and completed `N16` for this continuation turn: final signoff bundle validation now requires all Req5 Q4/Q8 packed-weight consistency checks to be present and passing in the final evidence manifest.
- Selected and completed `N17` for this continuation turn: final operator handoff validation now also requires all Req5 Q4/Q8 packed-weight consistency checks to be present and passing in the live final evidence manifest.
- Selected and completed `N18` for this continuation turn: final signoff bundle validation now requires current operator handoff validation freshness; stale `52`-check handoff validation fails bundle validation.
- Selected and completed `N19` for this continuation turn: legacy cyclic packed-weight tiles and large temporaries now use macro-controlled URAM binding, small tile scratch uses macro-controlled LUTRAM binding, and resource-policy/final-evidence gates require these checks. Resource-policy audit target is now `36/36` after N21; final manifest target is `86/86`, consistency `321`, failed consistency `0`.
- Selected and completed `N20` for this continuation turn: final operator handoff and final signoff bundle validators now require the cyclic resource-policy final-manifest checks by name. Operator handoff validation target later advanced to `66/66`; final bundle validation now requires operator handoff validation `check_count >=66`.
- Selected and completed `N21` for this continuation turn: C3b csynth LUT threshold `<=126506`, csynth latency threshold `<=37508072`, and routed WNS threshold `>=4.415 ns` are now resource-policy/final-evidence gates. Resource-policy target is `36/36`; final evidence consistency target is `321`.
- Selected and completed `N22` for this continuation turn: final operator handoff and final signoff bundle validators now require the C3b LUT/latency/WNS threshold final-manifest checks by name. Operator handoff validation target later advanced to `66/66`; final bundle validation now requires operator handoff validation `check_count >=66`.
- Selected and completed `N23` for this continuation turn: final-runner summary now exposes `blocker_count`, `remaining_blocker_detail_count`, `mirrored_artifact_count`, `mirrored_artifact_unique_count`, and `mirrored_artifact_duplicate_count`; final evidence rejects count/list mismatch.
- Selected and completed `N24` for this continuation turn: final operator handoff and final signoff bundle validators now require the final-runner summary count checks by name. Operator handoff validation target later advanced to `66/66`; final bundle validation requires operator handoff validation `check_count >=66`.
- Selected and completed `N25` for this continuation turn: final evidence manifest now requires final-unblock closeout validation `check_count >=49`; stale `48`-check closeout validation fails final evidence.
- Selected and completed `N26` for this continuation turn: final evidence manifest now requires third-goal source audit `required_count >=86` and `source_count >=224`; stale `85/223` source-audit artifacts fail final evidence.
- Selected and completed `N27` for this continuation turn: final operator handoff and final signoff bundle validators now require third-goal source-audit freshness checks by name; operator handoff validation target later advanced to `66/66`, and final bundle validation requires operator handoff validation `check_count >=66`.
- Selected and completed `N28` for this continuation turn: final operator handoff and final signoff bundle validators now require final evidence consistency count `>=332`; stale `331` evidence contracts fail validation.
- Selected and completed `N29` for this continuation turn: final operator handoff and final signoff bundle validators now require final evidence required count `>=86`; stale `85` evidence contracts fail validation.
- Selected and completed `N30` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live source-audit JSON counts/status, not only final-manifest consistency check names. Target later advanced by N31 to operator handoff validation `72/72`, bundle validation `87/87`.
- Selected and completed `N31` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live resource-policy JSON status/counts/core DSP/URAM/LUTRAM/C3b checks, not only final-manifest consistency check names. Target later advanced by N32 to operator handoff validation `79/79`, bundle validation `95/95`.
- Selected and completed `N32` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live spec-plan conformance JSON status/counts/core plan/spec/current-doc checks, not only final-manifest consistency check names. Current target later advanced by N36 to operator handoff validation `116/116`, bundle validation `137/137`.
- Selected and completed `N33` for this continuation turn: spec-plan conformance now checks that current docs carry live validator-count anchors for operator handoff validation `116/116` and bundle validation `137/137`; current target after refresh is spec-plan `83/83`.
- Selected and completed `N34` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live final-runner JSON status/blocker/detail/mirror counts, not only final-manifest runner summary check names. Current target after refresh is operator handoff validation `85/85`, bundle validation `102/102`.
- Selected and completed `N35` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live Req6 parameterization JSON status/counts/knobs/core PAR16-PAR32 checks, not only final-manifest Req6 check names. Current target later advanced by N36 to operator handoff validation `116/116`, bundle validation `137/137`.
- Selected and completed `N36` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live current-audit JSON status/summary/blocker paths/XR policy/QKV optional state, not only docs/HANDOVER text. Current target after refresh is operator handoff validation `116/116`, bundle validation `137/137`.
- Selected and completed `N37` for this continuation turn: spec-plan conformance now rejects current docs that retain stale XR policy check count `7` or older validator anchors after live current-audit policy count advanced to `9`.
- Selected and completed `N38` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live closeout-validation JSON status/count/core checks/safety, not only final-manifest closeout validation freshness checks. Current target after refresh is operator handoff validation `116/116`, bundle validation `137/137`.
- Selected and completed `N39` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live closeout-packet JSON status, blocker count, board package readiness, C3b smoke contract, XR policy integrity, QKV optional commands, and no-side-effect safety. Current target remains operator handoff validation `116/116`, bundle validation `137/137`.
- Selected and completed `N40` for this continuation turn: final operator handoff and final signoff bundle validators now directly load and validate live C3b physical-smoke gate and XR-VITs gate audit JSON status, blocker path, candidate, and no-side-effect safety. Current target after refresh is operator handoff validation `131/131`, bundle validation `152/152`.
- Selected and completed `N41` for this continuation turn after Spark spawn failed with `agent thread limit reached`: `final_blocker_closure_readiness` now emits an operator unblock plan with required C3b/XR inputs, dry-run templates, active-run command state, and no-side-effect safety; final evidence consistency gates validate the plan. Current final manifest consistency target is `332`.
- Selected and completed `N42` for this continuation turn with Spark read-only cross-check: final evidence now validates final unblock intake `next_inputs` paths and `operator_sequence` commands/side-effect profile against `final_blocker_closure_readiness.operator_unblock_plan`; final operator handoff and final signoff bundle validators require those five manifest checks by name. Current final manifest consistency target is `347`.
- Selected and completed `N43` for this continuation turn with Spark read-only gap audit: final-runner summary now records canonical `docs/resources` mirror integrity against `hardware/generated/signoff` sources, final evidence requires the mirror-integrity source/runtime checks, and final operator handoff plus bundle validators directly reject stale/missing mirror integrity. Current targets are final manifest consistency `347`, operator handoff validation `131/131`, and bundle validation `152/152`.
- Selected and completed `N44` for this continuation turn: spec-plan conformance now gates raw HANDOVER validator artifact summaries, so stale `checks 129` or `checks 150` cannot survive beside live `131/131` and `152/152` anchors. Spec-plan conformance target is now `86/86`.
- Selected and completed `N45` for this continuation turn after Spark usage limit and GPT5.5 read-only sidecar audit: final evidence now requires a single canonical resource-policy artifact pair in both `docs/resources` and `hardware/generated/signoff`, filename/payload date-tag agreement, and generated/docs SHA256 equality. The stale generated `e2e_resource_policy_audit_2026_06_15.{json,md}` pair was removed.
- Completed next internal candidate: E2E C++ `static_assert` coverage added and validated with Vitis HLS include-path syntax check.
- Live evidence anchor: final manifest `86/86`, consistency checks `347`, failed consistency `0`; closeout validation `49/49`; operator handoff validation `131/131`; bundle validation `152/152`; resource-policy audit `36/36`; spec-plan conformance `86/86`; source audit required `86`, sources `224`, missing `0`; current audit reflected `10`, partial `1`, blocked `1`.
- Next internal candidate: continue non-board evidence hardening, or wait for C3b/XR-VITs external unblock inputs.

## Decision Rule
- Do not mark Req11 complete unless X1 exact source exists or X2 policy is explicitly approved and validates.
- If P2 is selected, do not accept replacement policy unless candidate-audit fingerprint and recommendation snapshot match current audit content.
- Do not mark C3b complete unless canonical board-produced JSON exists and passes validation.
- Do not accept C3b/VREF/QKV smoke JSON when `bitfile` or `hwhfile` basename does not match the selected preset.
- Do not promote VREF successor as C3b replacement until physical smoke is captured and protection thresholds remain satisfied.
- Do not treat QKV URAM as default final-signoff requirement unless Q3 is explicitly selected.
