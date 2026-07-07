# HGTXR Hardware Sub-Plan

Date: 2026-06-16

## Sub-Agent Availability
Callable Spark-style sub-agents were available for focused read-only exploration in this session. The main agent owns integration, edits, validation, and final C3b/VREF consistency checks.
Latest Spark follow-up hit the GPT-5.3-Codex-Spark quota limit, so the current requirement-gap evaluation was routed to GPT5.5 fallback.
Latest XR-VITs blocker verification also hit the GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback completed read-only verification and kept Req11 blocked pending exact source restoration or explicit replacement-policy approval.
Latest blocker-readiness/discovery audit also hit the GPT-5.3-Codex-Spark usage limit; GPT5.5 fallback completed read-only audit and recommended adding PYNQ discovery summaries to blocker-readiness output.
Latest final evidence gate audit used GPT-5.3-Codex-Spark successfully for read-only review; it identified that runner blocker path payloads and blocker-readiness discovery summaries needed to become manifest/bundle-required checks.
Latest operator handoff contract audit used GPT-5.3-Codex-Spark successfully for read-only review; it identified that the operator handoff validator still allowed stale embedded final evidence contracts unless compared directly against the live manifest.
Latest plan-freshness hardening attempted GPT-5.3-Codex-Spark but hit quota, so GPT5.5 fallback was spawned for read-only audit while the main agent promoted Master/Sub plan freshness into spec-plan conformance.
Latest current-doc freshness extension attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`, so the main agent promoted Validation/CHOICE/log freshness into spec-plan conformance.
Live evidence anchor: final manifest `86/86`, consistency checks `347`, spec-plan conformance `86/86`, operator handoff validation `131/131`, bundle validation `152/152`, source audit required `86`, sources `224`, current audit reflected `10`, partial `1`, blocked `1`.
Latest resource-mapping hardening attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent added RMU/SMU DSP-bound helper coverage and final-evidence gates.
Live evidence anchor update: final manifest `86/86`, consistency checks `347`, closeout validation `49/49`, operator handoff validation `131/131`, bundle validation `152/152`, resource-policy audit `36/36`, spec-plan conformance `86/86`, source audit required `86`, sources `224`, current audit reflected `10`, partial `1`, blocked `1`.
Latest source-audit direct-validation sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation load and validate live source-audit JSON counts directly.
Latest resource-policy direct-validation sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation load and validate live resource-policy JSON status/count/core DSP/URAM/LUTRAM/C3b checks directly.
Latest spec-plan direct-validation sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation load and validate live spec-plan conformance JSON status/count/core plan/spec/current-doc checks directly.
Latest cyclic resource-placement sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent added macro-controlled URAM/LUTRAM placement for legacy cyclic weight tiles, large temporaries, and small tile scratch.
Latest next-unblocked-work Spark audit errored from output-token overflow, so the main agent proceeded with a bounded Req6/PAR16-PAR32 sweep extension using current resource evidence.
Latest Spark blocker-closure audit (`019ece20-161b-7650-aefc-9b5031fbb542`) completed read-only review and recommended aligning current C3b readiness with `--c3b-no-require-paths`; main agent implemented that fix and continued with C3b transfer-manifest runner integration.
Latest Req5 Q4/Q8 packed-weight integrity sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent completed binary/manifest SHA256, byte-count, expected runtime-state, expected C3b-output, and strict CSim/TB final-evidence gates.
Latest Req5 final-bundle sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final signoff bundle validation require the Req5 packed-weight final-manifest checks.
Latest Req5 operator-handoff sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff validation require the Req5 packed-weight live final-manifest checks.
Latest bundle operator-handoff count sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final signoff bundle validation require current operator handoff validation freshness.
Latest cyclic resource named-check sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation require cyclic URAM/LUTRAM final-manifest checks by name.
Latest C3b threshold sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made resource-policy/final-evidence gates require C3b csynth LUT, C3b csynth latency, and routed WNS threshold checks.
Latest C3b threshold named-check sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation require those C3b threshold checks by name.
Latest final-runner summary schema sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final evidence require runner blocker-count/detail-count/mirror-count consistency.
Latest final-runner summary named-check sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation require those runner summary count checks by name.
Latest next internal gap sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent refreshed final-unblock closeout validation freshness to the current `49/49` target.
Latest source-audit freshness sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final evidence require third-goal source audit required `86` and sources `224`.
Latest source-audit named-check sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation require source-audit freshness checks by name.
Latest consistency-threshold sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent refreshed final operator handoff and final signoff bundle minimum evidence consistency to the live `332` count.
Latest required-count threshold sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent refreshed final operator handoff and final signoff bundle minimum evidence required count to the live `86` artifact count.
Latest final-runner direct-validation sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation load and validate live final-runner JSON blocker/detail/mirror counts directly.
Latest Req6 direct-validation sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation load and validate live Req6 parameterization JSON knobs and core checks directly.
Latest current-audit direct-validation sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final operator handoff and final signoff bundle validation load and validate live current-audit JSON summary, blocker paths, XR-VITs policy integrity, and QKV optional state directly.
Latest final-blocker operator-plan sidecar attempted GPT-5.3-Codex-Spark but the runtime returned `agent thread limit reached`; main agent made final blocker closure readiness emit an operator unblock plan and promoted it into final evidence consistency checks.
Latest final-unblock intake/operator-plan cross-check used GPT-5.3-Codex-Spark read-only review successfully; main agent promoted the recommended intake/operator-sequence drift checks into final evidence and final validators.

## Sub-Agent Execution Used
- `Banach`: VREF-P0-01 PoT scale touchpoint exploration.
- `Ohm`: VREF-P0-02 buffer/resource placement exploration.
- `Copernicus`: C3b protection threshold and final-signoff blocker exploration.
- `Godel`: VREF-P0-01 reduced-reference scale support exploration.
- `Faraday`: VREF-P0-01 CSim/header wiring exploration.
- `Singer`: VREF-P0-01 successor documentation consistency check.
- `Boole`: GPT5.5 read-only third-goal requirement-gap evaluation.
- `Schrodinger the 2nd`: GPT5.5 read-only XR-VITs blocker verification after Spark quota exhaustion.
- `Volta the 2nd`: GPT5.5 read-only blocker-readiness/discovery audit after Spark quota exhaustion.
- `Dewey the 2nd`: GPT-5.3-Codex-Spark read-only final evidence gate audit for blocker path and blocker-readiness discovery checks.
- `Herschel the 2nd`: GPT-5.3-Codex-Spark read-only stale-contract audit for final operator handoff validation.
- `Goodall the 2nd`: GPT-5.3-Codex-Spark next-action read-only audit attempted; errored from output-token overflow before usable result.
- `Kuhn the 2nd`: GPT-5.3-Codex-Spark read-only blocker-closure audit; recommended current C3b `--c3b-no-require-paths` semantic alignment.
- `Raman the 2nd`: GPT-5.3-Codex-Spark read-only X6 documentation audit; identified missing direct placeholder-approver guard language in master/sub-plan/spec/handover/progress/validation/log docs.
- `T-3G-NEXT-GAP-AUDIT`: GPT-5.3-Codex-Spark spawn attempted for next-gap audit; runtime returned `agent thread limit reached`, so the main agent handled mirrored-artifact gate hardening.
- `T-3G-FRESHNESS-SIDECAR-001`: GPT-5.3-Codex-Spark spawn attempted for Validation/CHOICE/log freshness review; runtime returned `agent thread limit reached`, so the main agent handled the bounded integration.
- `T-3G-RMU-SMU-DSP-001`: GPT-5.3-Codex-Spark spawn attempted for resource-pragma review; runtime returned `agent thread limit reached`, so the main agent implemented RMU/SMU DSP helper coverage and final evidence gating.
- `T-3G-Q4Q8-CONTRACT-SIDECAR-001`: GPT-5.3-Codex-Spark spawn attempted for Req5 packed-weight contract review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded final-evidence hardening.
- `T-3G-BUNDLE-REQ5-SIDECAR-003`: GPT-5.3-Codex-Spark spawn attempted for Req5 bundle-check review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded final-bundle hardening.
- `T-3G-HANDOFF-REQ5-SIDECAR-004`: GPT-5.3-Codex-Spark spawn attempted for Req5 operator-handoff review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded handoff-validator hardening.
- `T-3G-BUNDLE-OP-HANDOFF-COUNT-SIDECAR-005`: GPT-5.3-Codex-Spark spawn attempted for bundle freshness review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded bundle-validator check-count hardening.
- `T-3G-CYCLIC-RESOURCE-SIDECAR-006`: GPT-5.3-Codex-Spark spawn attempted for cyclic resource-placement review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded cyclic URAM/LUTRAM policy and audit gates.
- `T-3G-CYCLIC-RESOURCE-NAMED-CHECK-SIDECAR-007`: GPT-5.3-Codex-Spark spawn attempted for cyclic named-check validator review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded validator hardening.
- `T-3G-C3B-THRESHOLD-NAMED-CHECK-SIDECAR-009`: GPT-5.3-Codex-Spark spawn attempted for C3b threshold named-check validator review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded validator hardening.
- `T-3G-RUNNER-SUMMARY-SCHEMA-SIDECAR-011`: GPT-5.3-Codex-Spark spawn attempted for final-runner summary schema review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded runner/final-evidence schema hardening.
- `T-3G-N34-FINAL-RUNNER-DIRECT`: GPT-5.3-Codex-Spark spawn attempted for final-runner direct validation; runtime returned `agent thread limit reached`, so the main agent implemented the bounded validator hardening.
- `T-3G-N35-REQ6-DIRECT`: GPT-5.3-Codex-Spark spawn attempted for Req6 direct validation; runtime returned `agent thread limit reached`, so the main agent implemented the bounded validator hardening.
- `T-3G-N36-CURRENT-AUDIT-DIRECT`: GPT-5.3-Codex-Spark spawn attempted for current-audit direct validation; runtime returned `agent thread limit reached`, so the main agent implemented the bounded validator hardening.
- `T-3G-N41-UNBLOCK-READINESS-SNAPSHOT`: GPT-5.3-Codex-Spark spawn attempted for final-blocker operator-plan review; runtime returned `agent thread limit reached`, so the main agent implemented the bounded readiness/final-evidence hardening.

## Task Cards

```yaml
task_card:
  task_id: T-3G-N42-INTAKE-PLAN-CROSSCHECK
  sub_agent: "gpt5.3-codex-spark"
  role: "evaluator + implementer"
  objective: "Make final evidence and final validators reject drift between final-unblock intake next_inputs/operator_sequence and final blocker closure operator_unblock_plan."
  file_ownership:
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_spec_plan_conformance_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman"
  inputs:
    - "docs/resources/final_blocker_closure_readiness_2026_06_10.json"
    - "docs/resources/final_unblock_intake_2026_06_10.json"
    - "Spark read-only recommendation from T-3G-N42-INTAKE-PLAN-CROSSCHECK"
  outputs:
    - "final_unblock_intake_*operator_plan* consistency checks"
    - "validator named-check requirements for intake/operator-plan gates"
  validation:
    - "python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "External C3b board JSON remains required for final closure"
    - "Exact XR-VITs source or approved replacement policy remains required for final closure"

task_card:
  task_id: T-3G-N41-UNBLOCK-READINESS-SNAPSHOT
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final blocker closure readiness emit a machine-readable operator unblock plan and make final evidence reject missing or unsafe plan drift."
  file_ownership:
    - "tools/check_final_blocker_closure_readiness.py"
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_spec_plan_conformance_audit.py"
    - "tests/test_check_final_blocker_closure_readiness.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/CHOICE.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman"
  inputs:
    - "docs/resources/final_blocker_closure_readiness_2026_06_10.json"
    - "C3b canonical path pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
    - "XR-VITs exact/policy paths"
  outputs:
    - "operator_unblock_plan"
    - "final_blocker_closure_operator_plan_* consistency checks"
  validation:
    - "python3 -m unittest tests.test_check_final_blocker_closure_readiness tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "External C3b board JSON remains required for final closure"
    - "Exact XR-VITs source or approved replacement policy remains required for final closure"

task_card:
  task_id: T-3G-N36-CURRENT-AUDIT-DIRECT
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation directly validate the live third-goal current-audit JSON summary, blocker paths, policy integrity, QKV optional state, and no-side-effect safety."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_write_third_goal_completion_audit.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live third_goal_current_audit_2026_06_16.json"
    - "expected summary: requirements 12, reflected 10, partial 1, blocked 1, default blockers 2"
    - "external blocker path map and QKV optional state"
  outputs:
    - "operator handoff validation target 116/116"
    - "final signoff bundle validation target 137/137"
    - "stale current-audit summary/policy/QKV state fails direct validation"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N35-REQ6-DIRECT"
```

```yaml
task_card:
  task_id: T-3G-N35-REQ6-DIRECT
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation directly validate the live Req6 parameterization audit JSON knobs and core PAR16/PAR32 checks."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_write_third_goal_completion_audit.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live req6_parameterization_audit_2026_06_16.json"
    - "expected knobs: tiling 1, parallelism 8, bus 256, activation 8, weight 4, buffer 256, FIFO 128"
    - "core checks: config defines, PAR16 validated, PAR32 exploratory/fresh-report policy"
  outputs:
    - "operator handoff validation target 116/116"
    - "final signoff bundle validation target 137/137"
    - "stale Req6 count or failed core Req6 checks fail direct validation"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N34-FINAL-RUNNER-DIRECT"
```

```yaml
task_card:
  task_id: T-3G-N34-FINAL-RUNNER-DIRECT
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation directly validate the live final-runner JSON blocker, blocker-detail, and mirrored-artifact counts."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_write_third_goal_completion_audit.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live third_goal_final_signoff_run_2026_06_10.json"
    - "expected remaining blockers: C3b AXIS/DMA physical smoke result; requested XR-VITs sibling"
    - "live mirror summary count contract"
  outputs:
    - "operator handoff validation target 85/85"
    - "final signoff bundle validation target 102/102"
    - "stale final-runner blocker/detail/mirror counts fail direct validation"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit"
    - "python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N33-SPEC-PLAN-VALIDATOR-COUNT-FRESHNESS"
```

```yaml
task_card:
  task_id: T-3G-N33-SPEC-PLAN-VALIDATOR-COUNT-FRESHNESS
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make spec-plan conformance gate current-doc freshness for live final operator handoff validation and final signoff bundle validation counts."
  file_ownership:
    - "tools/write_spec_plan_conformance_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live final_operator_handoff_validation check_count 79"
    - "live final_signoff_bundle_validation check_count 95"
    - "current docs freshness contract"
  outputs:
    - "spec-plan conformance target 84/84"
    - "current docs must record operator handoff validation 116/116 and bundle validation 137/137"
    - "final validators reject spec-plan conformance below 84 checks"
  validation:
    - "python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N32-SPEC-PLAN-DIRECT"
```

```yaml
task_card:
  task_id: T-3G-REQUIRED-COUNT-THRESHOLD-018
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Refresh final operator handoff and final signoff bundle minimum final-evidence required-count thresholds to the current 86-artifact contract."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live final evidence manifest required_count 86"
    - "final operator handoff validator"
    - "final signoff bundle validator"
  outputs:
    - "operator handoff validator rejects evidence contracts below required_count 86"
    - "bundle validator rejects matching stale 85-artifact evidence contracts"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-CONSISTENCY-THRESHOLD-017"
```

```yaml
task_card:
  task_id: T-3G-N30-SOURCE-AUDIT-DIRECT-VALIDATION
  sub_agent: "gpt5.3-codex-spark"
  role: "analyst + implementer"
  objective: "Make final validators directly validate live third-goal source-audit JSON counts/status instead of relying only on final-manifest named checks."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman"
  inputs:
    - "docs/resources/third_goal_source_audit_2026_06_16.json"
    - "docs/resources/final_evidence_manifest_2026_06_10.json"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "operator handoff validation checks live source audit directly"
    - "bundle validation checks live source audit directly"
    - "freshness thresholds advanced to operator 66 and bundle 80"
  validation:
    - "stale source audit 85/223 fails operator handoff validation"
    - "stale source audit 85/223 fails bundle validation"
    - "final runner remains expected-blocked only on external C3b/XR-VITs inputs"
  dependencies:
    - "T-3G-N26"
    - "T-3G-N27"
```

```yaml
task_card:
  task_id: T-3G-N31-RESOURCE-POLICY-DIRECT-VALIDATION
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "analyst + implementer"
  objective: "Make final validators directly validate live resource-policy JSON counts/status/core DSP/URAM/LUTRAM/C3b checks instead of relying only on final-manifest named checks."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman"
  inputs:
    - "docs/resources/e2e_resource_policy_audit_2026_06_10.json"
    - "docs/resources/final_evidence_manifest_2026_06_10.json"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "operator handoff validation checks live resource policy directly"
    - "bundle validation checks live resource policy directly"
    - "freshness thresholds advanced to operator 72 and bundle 87"
  validation:
    - "stale resource-policy count 35 fails operator handoff validation"
    - "stale resource-policy count 35 fails bundle validation"
    - "resource-policy core check failure fails both validators"
    - "final runner remains expected-blocked only on external C3b/XR-VITs inputs"
  dependencies:
    - "T-3G-N19"
    - "T-3G-N20"
    - "T-3G-N21"
    - "T-3G-N22"
```

```yaml
task_card:
  task_id: T-3G-CONSISTENCY-THRESHOLD-017
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Refresh final operator handoff and final signoff bundle minimum evidence consistency thresholds to the current 332-check contract."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live final evidence manifest consistency count 332"
    - "final operator handoff validator"
    - "final signoff bundle validator"
  outputs:
    - "operator handoff validator rejects evidence contracts below consistency count 332"
    - "bundle validator rejects matching stale 325-count evidence contracts"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-SOURCE-AUDIT-NAMED-CHECK-016"
```

```yaml
task_card:
  task_id: T-3G-SOURCE-AUDIT-NAMED-CHECK-016
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation require third-goal source-audit freshness checks by name."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live final evidence manifest source-audit checks"
    - "final operator handoff validator"
    - "final signoff bundle validator"
  outputs:
    - "operator handoff validator rejects missing source-audit freshness checks with count preserved"
    - "bundle validator rejects missing source-audit freshness checks"
    - "operator handoff validation freshness target 62"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-SOURCE-AUDIT-FRESHNESS-015"
```

```yaml
task_card:
  task_id: T-3G-SOURCE-AUDIT-FRESHNESS-015
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Refresh final evidence freshness gates for the third-goal source audit required/source-count contract."
  file_ownership:
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "generated/signoff/third_goal_source_audit_2026_06_16.json"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_final_evidence_manifest.py"
  outputs:
    - "final evidence rejects source audits below required_count 86"
    - "final evidence rejects source audits below source_count 224"
    - "regression coverage for stale 85/223 source-audit counts"
  validation:
    - "python3 -m unittest tests.test_write_final_evidence_manifest"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-CLOSEOUT-VALIDATION-FRESHNESS-014"
```

```yaml
task_card:
  task_id: T-3G-RUNNER-SUMMARY-SCHEMA-SIDECAR-011
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final-runner summary expose and final-evidence-check blocker count, blocker-detail count, and mirrored-artifact count consistency."
  file_ownership:
    - "tools/run_third_goal_final_signoff.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_run_third_goal_final_signoff.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "third_goal_final_signoff_run summary JSON"
    - "final evidence manifest consistency gates"
    - "external blocker closeout state"
  outputs:
    - "runner summary exposes blocker_count and remaining_blocker_detail_count"
    - "runner summary exposes mirrored_artifact_count, mirrored_artifact_unique_count, and mirrored_artifact_duplicate_count"
    - "final evidence rejects blocker/detail/mirror count mismatch"
  validation:
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_run_third_goal_final_signoff"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-C3B-THRESHOLD-NAMED-CHECK-SIDECAR-009"
```

```yaml
task_card:
  task_id: T-3G-CYCLIC-RESOURCE-NAMED-CHECK-SIDECAR-007
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation require cyclic URAM/LUTRAM resource-policy consistency checks by name."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "final evidence manifest cyclic resource-policy checks"
    - "final operator handoff validator"
    - "final signoff bundle validator"
  outputs:
    - "operator handoff validator rejects missing cyclic resource checks with count preserved"
    - "bundle validator rejects missing cyclic resource checks with count preserved"
    - "operator handoff validation freshness target later advanced to 60 after T-3G-RUNNER-SUMMARY-NAMED-CHECK-SIDECAR-013"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-CYCLIC-RESOURCE-SIDECAR-006"
```

```yaml
task_card:
  task_id: T-3G-C3B-THRESHOLD-NAMED-CHECK-SIDECAR-009
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation require C3b LUT, latency, and routed WNS threshold final-manifest checks by name."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "final evidence manifest C3b threshold checks"
    - "final operator handoff validator"
    - "final signoff bundle validator"
  outputs:
    - "operator handoff validator rejects missing C3b threshold checks with count preserved"
    - "bundle validator rejects missing C3b threshold checks with count preserved"
    - "operator handoff validation freshness target later advanced to 60"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-C3B-THRESHOLD-GATE-008"
```

```yaml
task_card:
  task_id: T-3G-CLOSEOUT-VALIDATION-FRESHNESS-014
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Refresh final evidence freshness gate for final-unblock closeout validation to the current 49-check contract."
  file_ownership:
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "generated/signoff/final_unblock_closeout_packet_validation_2026_06_10.json"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_final_evidence_manifest.py"
  outputs:
    - "final evidence rejects closeout validation check_count below 49"
    - "regression coverage for stale 48-check closeout validation"
  validation:
    - "python3 -m unittest tests.test_write_final_evidence_manifest"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-RUNNER-SUMMARY-NAMED-CHECK-SIDECAR-013"
```

```yaml
task_card:
  task_id: T-3G-RUNNER-SUMMARY-NAMED-CHECK-SIDECAR-013
  sub_agent: "gpt5.3-codex-spark attempted; spawn failed with agent thread limit reached; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff and final signoff bundle validation require final-runner summary count checks by name."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "final evidence manifest final-runner summary count checks"
    - "final operator handoff validator"
    - "final signoff bundle validator"
  outputs:
    - "operator handoff validator rejects missing runner summary count checks with count preserved"
    - "bundle validator rejects missing runner summary count checks with count preserved"
    - "operator handoff validation freshness target 60"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-RUNNER-SUMMARY-SCHEMA-SIDECAR-011"
```

```yaml
task_card:
  task_id: T-3G-CYCLIC-RESOURCE-SIDECAR-006
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Apply parameterized URAM/LUTRAM resource-placement policy to legacy cyclic HLS baseline without changing selected C3b board-smoke path."
  file_ownership:
    - "hls/include/hgtxr_cyclic_transformer_params.hpp"
    - "hls/src/hgtxr_top.cpp"
    - "tools/write_e2e_resource_policy_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_e2e_resource_policy_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "legacy cyclic HLS top"
    - "resource-policy audit"
    - "third-goal resource direction"
  outputs:
    - "cyclic weight tile URAM macro and pragmas"
    - "cyclic large temporary URAM macro and pragmas"
    - "cyclic small tile LUTRAM macro and pragmas"
    - "resource-policy audit 36/36 after T-3G-C3B-THRESHOLD-GATE-008"
    - "final evidence consistency 321 after T-3G-C3B-THRESHOLD-GATE-008"
  validation:
    - "g++ -std=c++17 -Ihls/include -Ihls/src -fsyntax-only hls/src/hgtxr_top.cpp"
    - "python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-C3B-THRESHOLD-GATE-008
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Promote C3b LUT, latency, and routed WNS protection thresholds into resource-policy and final-evidence gates."
  file_ownership:
    - "tools/write_e2e_resource_policy_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_write_e2e_resource_policy_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/Master-Plan.md"
    - "docs/Spec.md"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "C3b resource matrix row"
    - "C3b csynth.xml"
    - "C3b routed timing summary"
    - "resource-policy audit"
    - "final evidence manifest"
  outputs:
    - "resource-policy audit 36/36"
    - "final evidence consistency 321"
    - "validator minimum evidence consistency 321"
  validation:
    - "python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-CYCLIC-RESOURCE-NAMED-CHECK-SIDECAR-007"
```

```yaml
task_card:
  task_id: T-3G-BUNDLE-OP-HANDOFF-COUNT-SIDECAR-005
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final signoff bundle validation require latest final operator handoff validation check_count >=54."
  file_ownership:
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "final operator handoff validation artifact"
    - "final signoff bundle validator"
  outputs:
    - "bundle validator rejects stale 52-check handoff validation"
    - "regression for stale operator handoff validation check count"
  validation:
    - "python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-HANDOFF-REQ5-SIDECAR-004
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final operator handoff validation require the Req5 Q4/Q8 packed-weight live final-manifest consistency checks."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live final evidence manifest Req5 checks"
    - "final operator handoff validator"
  outputs:
    - "operator handoff validator requires Req5 packed-weight checks"
    - "regression for missing Req5 packed-weight check with unchanged consistency count"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-BUNDLE-REQ5-SIDECAR-003
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Make final signoff bundle validation require the Req5 Q4/Q8 packed-weight final-manifest consistency checks."
  file_ownership:
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "final evidence manifest Req5 checks"
    - "final signoff bundle validator"
  outputs:
    - "bundle validator requires Req5 packed-weight checks"
    - "regression for missing Req5 packed-weight check"
  validation:
    - "python3 -m unittest tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-Q4Q8-CONTRACT-SIDECAR-001
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Harden Req5 Q4/Q8 packed-weight evidence so final signoff gates the binary/manifest SHA256, byte count, expected runtime state, expected C3b output, and strict TB/CSim checks."
  file_ownership:
    - "tools/write_req5_q4q8_swhw_match_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_req5_q4q8_swhw_match_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "Req5 Q4/Q8 SW-HW match audit"
    - "packed Q4 weight binary manifest"
    - "final evidence manifest"
  outputs:
    - "Req5 audit 11/11"
    - "final evidence consistency 312"
  validation:
    - "python3 -m unittest tests.test_write_req5_q4q8_swhw_match_audit tests.test_write_final_evidence_manifest"
    - "python3 -m unittest tests.test_write_req5_q4q8_swhw_match_audit tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit tests.test_write_third_goal_current_audit tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_p2_vit_scale_calibration_report"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-033
  sub_agent: "gpt5.3-codex-spark + codex-native"
  role: "evaluator"
  objective: "Selected Path Execution Audit compatibility anchor for A2/A1/C/PAR16 third-goal continuity."
  file_ownership:
    - "docs/Sub-Plan.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
  inputs:
    - "selected-path execution audit"
  outputs:
    - "plan continuity marker"
  validation:
    - "python3 -m unittest tests.test_write_spec_plan_conformance_audit"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-RMU-SMU-DSP-001
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "implementer"
  objective: "Map RMU/SMU relation/projection multiply-heavy paths through DSP-bound helpers and gate the source contract in final evidence."
  file_ownership:
    - "hls/src/rmu_smu.cpp"
    - "tools/write_e2e_resource_policy_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_e2e_resource_policy_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/Validation.md"
    - "docs/CHOICE.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "RMU/SMU HLS source"
    - "E2E resource-policy audit"
    - "final evidence manifest"
  outputs:
    - "RMU/SMU DSP helper source changes"
    - "resource-policy audit 27/27"
    - "final evidence consistency 302"
  validation:
    - "g++ -std=c++17 -Ihls/include -Ihls/src -fsyntax-only hls/src/rmu_smu.cpp"
    - "python3 -m unittest tests.test_write_e2e_resource_policy_audit tests.test_write_final_evidence_manifest"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-FRESHNESS-SIDECAR-001
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator"
  objective: "Extend current-doc freshness gates from Master/Sub/Spec/PROGRESS/HANDOVER to include Validation, CHOICE, and log."
  file_ownership:
    - "tools/write_spec_plan_conformance_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "current final evidence manifest"
    - "spec-plan conformance audit"
    - "third-goal source/current audits"
  outputs:
    - "spec-plan conformance 66/66"
    - "final evidence freshness contract covering Validation/CHOICE/log"
  validation:
    - "python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_write_final_evidence_manifest tests.test_run_third_goal_final_signoff tests.test_validate_final_signoff_bundle tests.test_validate_final_operator_handoff tests.test_write_third_goal_completion_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-XRVITS-ACTIVE-APPROVER-GUARD
  sub_agent: "gpt5.3-codex-spark"
  role: "evaluator"
  objective: "Reject placeholder approver metadata for active XR-VITs replacement-policy writes while preserving dry-run preview review artifacts."
  file_ownership:
    - "tools/create_xr_vits_replacement_policy.py"
    - "tools/run_third_goal_final_signoff.py"
    - "tests/test_create_xr_vits_replacement_policy.py"
    - "tests/test_run_third_goal_final_signoff.py"
    - "docs/CHOICE.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "XR-VITs replacement-policy preview"
    - "candidate-audit fingerprint contract"
    - "final-runner expected-blocked contract"
  outputs:
    - "active write rejects placeholder approved-by"
    - "dry-run preview allows placeholder approved-by"
    - "final runner blocks on active policy creation failure"
    - "X6 decision recorded in CHOICE and tracking docs"
  validation:
    - "python3 -m unittest tests.test_create_xr_vits_replacement_policy tests.test_run_third_goal_final_signoff"
    - "python3 -m py_compile tools/create_xr_vits_replacement_policy.py tools/run_third_goal_final_signoff.py tests/test_create_xr_vits_replacement_policy.py tests/test_run_third_goal_final_signoff.py"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "Req11 exact source or approved replacement-policy external unblock remains unresolved"
```

```yaml
task_card:
  task_id: T-3G-C3B-READINESS-FLAG-TRANSFER-RUNNER
  sub_agent: "gpt5.3-codex-spark + codex-native"
  role: "evaluator + implementer"
  objective: "Align final-blocker C3b no-require-paths semantics and regenerate C3b transfer manifest evidence inside the final signoff runner."
  file_ownership:
    - "tools/check_final_blocker_closure_readiness.py"
    - "tools/run_third_goal_final_signoff.py"
    - "tests/test_check_final_blocker_closure_readiness.py"
    - "tests/test_run_third_goal_final_signoff.py"
    - "docs/CHOICE.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman"
  inputs:
    - "tools/write_c3b_smoke_transfer_manifest.py"
    - "docs/resources/final_evidence_manifest_2026_06_10.json"
    - "docs/resources/third_goal_final_signoff_run_2026_06_10.json"
    - "Spark sidecar 019ece20-161b-7650-aefc-9b5031fbb542"
  outputs:
    - "current C3b readiness honors c3b_require_paths"
    - "c3b_transfer_manifest_status in final runner summary"
    - "mirrored c3b_smoke_transfer_manifest_2026_06_10.{json,md}"
    - "mirrored e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
  validation:
    - "python3 -m unittest tests.test_check_final_blocker_closure_readiness tests.test_run_third_goal_final_signoff tests.test_write_c3b_smoke_transfer_manifest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle"
    - "python3 -m py_compile tools/check_final_blocker_closure_readiness.py tools/run_third_goal_final_signoff.py tools/write_c3b_smoke_transfer_manifest.py tests/test_check_final_blocker_closure_readiness.py tests/test_run_third_goal_final_signoff.py tests/test_write_c3b_smoke_transfer_manifest.py"
    - "python3 tools/write_c3b_smoke_transfer_manifest.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "C3b PYNQ bundle/session/tar must exist"
    - "C3b physical board JSON remains external"
```

```yaml
task_card:
  task_id: T-3G-REQ6-PAR16-PAR32-SWEEP-GATE
  sub_agent: "gpt5.3-codex-spark attempted + codex-native fallback"
  role: "evaluator + implementer"
  objective: "Reflect validated C3b PAR16 evidence and exploratory PAR32 in the machine-readable sweep and Req6/final-evidence gates."
  file_ownership:
    - "configs/sweeps/zcu104_cyclic_transformer_sweep.yaml"
    - "tools/write_req6_parameterization_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_req6_parameterization_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_write_third_goal_completion_audit.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
  inputs:
    - "docs/resources/e2e_resource_matrix_2026_06_10.json"
    - "vivado/scripts/run_e2e_q4w8a_csim.tcl"
    - "vivado/scripts/run_e2e_q4w8a_csynth.tcl"
    - "docs/resources/req6_parameterization_audit_2026_06_16.json"
  outputs:
    - "parallelism_factor values [1, 2, 4, 8, 16, 32]"
    - "csim_tcl_supports_par16_par32"
    - "csynth_tcl_supports_par16_par32"
  validation:
    - "python3 -m unittest tests.test_write_req6_parameterization_audit tests.test_write_final_evidence_manifest tests.test_write_third_goal_completion_audit"
    - "python3 tools/write_req6_parameterization_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "C3b PAR16 resource matrix evidence"
```

```yaml
task_card:
  task_id: T-3G-OPERATOR-HANDOFF-LIVE-EVIDENCE-GATE
  sub_agent: "gpt5.3-codex-spark + codex-native"
  role: "evaluator + implementer"
  objective: "Require final operator handoff embedded evidence contract to match the live final evidence manifest and current 332 consistency-count threshold."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
  inputs:
    - "docs/resources/final_evidence_manifest_2026_06_10.json"
    - "docs/resources/final_operator_handoff_2026_06_10.json"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
  outputs:
    - "live_evidence_manifest_contract_present"
    - "evidence_manifest_contract_matches_live"
    - "MIN_EVIDENCE_CONSISTENCY_COUNT=278"
  validation:
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit"
    - "python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_spec_plan_conformance_audit.py"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-MANIFEST-BLOCKER-GATES"
```

```yaml
task_card:
  task_id: T-3G-MANIFEST-BLOCKER-GATES
  sub_agent: "gpt5.3-codex-spark + codex-native"
  role: "evaluator + implementer"
  objective: "Promote final-runner blocker path payloads and blocker-readiness PYNQ discovery summaries into final evidence manifest and bundle-required gates."
  file_ownership:
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_validate_final_signoff_bundle.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
  inputs:
    - "generated/signoff/third_goal_final_signoff_run_2026_06_10.json"
    - "generated/signoff/final_blocker_closure_readiness_2026_06_10.json"
    - "docs/resources/final_evidence_manifest_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "final_runner_remaining_blocker_* consistency checks"
    - "final_blocker_closure_*_discovery consistency checks"
    - "bundle required manifest-check list update"
  validation:
    - "python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle"
    - "python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_signoff_bundle tests.test_run_third_goal_final_signoff tests.test_check_final_blocker_closure_readiness tests.test_write_third_goal_current_audit tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-BLOCKER-PATHS"
    - "T-3G-BLOCKER-DISCOVERY"
```

```yaml
task_card:
  task_id: T-3G-BLOCKER-PATHS
  sub_agent: "codex-native"
  role: "implementer"
  objective: "Expose exact remaining blocker input paths in the final signoff runner summary."
  file_ownership:
    - "tools/run_third_goal_final_signoff.py"
    - "tests/test_run_third_goal_final_signoff.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
  inputs:
    - "generated/signoff/final_blocker_closure_readiness_2026_06_10.json"
    - "generated/signoff/third_goal_final_signoff_run_2026_06_10.json"
  outputs:
    - "remaining_blocker_input_paths"
    - "remaining_blocker_details"
  validation:
    - "python3 -m unittest tests.test_run_third_goal_final_signoff"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-BLOCKER-DISCOVERY
  sub_agent: "gpt5.5 fallback + codex-native"
  role: "evaluator + implementer"
  objective: "Expose side-effect-free C3b/VREF/QKV PYNQ discovery summaries in final blocker-readiness output."
  file_ownership:
    - "tools/check_final_blocker_closure_readiness.py"
    - "tests/test_check_final_blocker_closure_readiness.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
  inputs:
    - "tools/discover_pynq_smoke_candidates.py"
    - "generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json"
    - "generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json"
    - "generated/signoff/qkv_uram_smoke_candidate_discovery_2026_06_16.json"
  outputs:
    - "final_blocker_closure_readiness_2026_06_10.json:pynq_discovery"
    - "final_blocker_closure_readiness_2026_06_10.md:PYNQ Discovery"
  validation:
    - "python3 -m unittest tests.test_check_final_blocker_closure_readiness tests.test_run_third_goal_final_signoff tests.test_discover_pynq_smoke_candidates tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest"
    - "python3 -m py_compile tools/check_final_blocker_closure_readiness.py tools/run_third_goal_final_signoff.py tests/test_check_final_blocker_closure_readiness.py tests/test_run_third_goal_final_signoff.py"
    - "git diff --check -- tools/check_final_blocker_closure_readiness.py tools/run_third_goal_final_signoff.py tests/test_check_final_blocker_closure_readiness.py tests/test_run_third_goal_final_signoff.py"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-C3B
  sub_agent: "codex-native"
  role: "evaluator"
  objective: "Preserve and unblock C3b AXIS/DMA board-smoke path."
  file_ownership:
    - "generated/signoff/*"
    - "pynq/hgtxr/*"
    - "tools/*c3b*"
  assigned_skill:
    - "fpga-asic-design-expert"
  inputs:
    - "generated/signoff/third_goal_requirements_trace_2026_06_10.md"
    - "generated/signoff/third_goal_unblock_checklist_2026_06_10.md"
  outputs:
    - "pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
    - "generated/signoff/c3b_protection_checklist_2026_06_16.md"
  validation:
    - "python3 tools/validate_pynq_smoke_result.py pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16"
    - "python3 tools/write_c3b_protection_checklist.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/c3b_protection_checklist_2026_06_16.json --markdown-out generated/signoff/c3b_protection_checklist_2026_06_16.md"
  dependencies: []
```

```yaml
task_card:
  task_id: T-3G-EVAL-001
  sub_agent: "gpt5.5"
  role: "evaluator"
  objective: "Audit third-goal requirements 0-11 against current local signoff evidence."
  file_ownership: []
  assigned_skill:
    - "hls-design-vitis"
    - "fpga-asic-mapping"
  inputs:
    - "docs/track/THIRD_GOAL_REQUIREMENTS_2026_06_16.md"
    - "docs/track/PROGRESS.md"
    - "docs/Execution.md"
    - "docs/Validation.md"
    - "tools/check_third_goal_preflight.py"
    - "generated/signoff/*"
  outputs:
    - "read-only requirement status table"
    - "recommendation for next local signoff artifact"
  validation:
    - "use only current filesystem evidence"
    - "mark indirect or external-gated evidence as partial/blocked"
  dependencies: []
```

```yaml
task_card:
  task_id: T-VREF-P0-01
  sub_agent: "gpt5.3-codex-spark"
  role: "research + implementer"
  objective: "Create SW-first P2-ViT PoT scale calibration plan for HGTXR Q4/Q8."
  file_ownership:
    - "analysis/vit-accel/papers/P2-ViT/analysis.md"
    - "analysis/vit-accel/codebases/P2-ViT/analysis.md"
    - "configs/sweeps/zcu104_cyclic_transformer_sweep.yaml"
  assigned_skill:
    - "code-analyzer"
    - "fpga-asic-design-expert"
  inputs:
    - "P2-ViT paper/codebase analysis"
    - "HGTXR Q4/Q8 weight contract"
  outputs:
    - "SW-first scale calibration checklist"
    - "generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.md"
    - "generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md"
    - "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth/solution_e2e_q4w8a/syn/report/csynth.xml"
    - "generated/hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_ip/solution_e2e_q4w8a/impl/ip/component.xml"
    - "generated/build/vivado/overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_overlay/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit"
    - "pynq/hgtxr/hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream.bit"
    - "generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz"
    - "generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md"
    - "generated/signoff/zcu104_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_remote_run_2026_06_16.md"
    - "successor experiment entry"
  validation:
    - "SW/HW exact-match remains pass"
    - "python3 tools/write_vref_p0_pot_scale_sweep.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.json --markdown-out generated/signoff/vref_p0_pot_scale_sweep_2026_06_16.md"
    - "python3 tools/write_vref_p0_pot_scale_successor.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.json --markdown-out generated/signoff/vref_p0_pot_scale_successor_softmax_input_x2_2026_06_16.md"
    - "python3 tools/validate_e2e_axis_vector.py --spec refs/vref_p0/e2e_axis_vector_hgpipe_math_lnq_active16_softmax_input_x2_spec.json --check-c-header hls/tb/e2e_axis_vector_vref_p0_hgpipe_lnq_active16_softmax_input_x2_golden.hpp"
    - "E2E AXIS CSim log contains comparison passed, runtime_state=2, and CSim done with 0 errors"
    - "E2E AXIS csynth report exists and is parsed into the successor manifest"
    - "HLS IP package component.xml/export.zip exist and are parsed into the successor manifest"
    - "Vivado routed timing WNS >= 4.415 ns and route error count is 0"
    - "successor bit/hwh exist under generated overlay and pynq/hgtxr"
    - "successor PYNQ smoke bundle validates with variant vref-p0-softmax-input-x2-dsp-mixed-stream"
    - "successor remote smoke dry-run emits SSH/SCP commands and import preset"
    - "no C3b artifact overwritten"
    - "successor promotion passes C3b protection checklist"
  dependencies:
    - "T-3G-C3B for hardware promotion only"
```

```yaml
task_card:
  task_id: T-3G-N37-DOC-POLICY-COUNT-FRESHNESS
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Reject stale XR-VITs policy-count and validator-count anchors in current third-goal docs."
  file_ownership:
    - "tools/write_spec_plan_conformance_audit.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/CHOICE.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/PROGRESS.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "live third_goal_current_audit_2026_06_16 policy check count 9"
    - "live final_operator_handoff_validation 116/116"
    - "live final_signoff_bundle_validation 137/137"
  outputs:
    - "current_docs_have_no_stale_xr_policy_check_count"
    - "expanded stale validator-count token gate"
    - "updated current docs and choice record"
  validation:
    - "python3 -m unittest tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N36-CURRENT-AUDIT-DIRECT"
```

```yaml
task_card:
  task_id: T-3G-N38-CLOSEOUT-VALIDATION-DIRECT
  sub_agent: "gpt5.3-codex-spark"
  role: "analyst/evaluator"
  objective: "Promote final-unblock closeout validation from final-manifest freshness-only evidence to direct live JSON validation in final validators."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman"
  inputs:
    - "docs/resources/final_unblock_closeout_packet_validation_2026_06_10.json"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "live closeout validation direct checks in operator handoff validator"
    - "live closeout validation direct checks in final bundle validator"
    - "updated validator freshness thresholds 116/116 and 137/137"
  validation:
    - "focused validator/final-evidence/spec-plan/completion tests"
    - "broader final-signoff regression tests"
    - "final runner expected-blocked with only external blockers"
  dependencies:
    - "T-3G-N37-DOC-POLICY-COUNT-FRESHNESS"
  runtime_note: "Spark spawn failed with agent thread limit reached; main agent completed bounded fallback."
```

```yaml
task_card:
  task_id: T-3G-N39-CLOSEOUT-PACKET-DIRECT
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Promote final-unblock closeout packet from validation-result-only evidence to direct live JSON validation in final validators."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tools/write_spec_plan_conformance_audit.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "tests/test_write_third_goal_completion_audit.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "docs/resources/final_unblock_closeout_packet_2026_06_10.json"
    - "docs/resources/final_unblock_closeout_packet_validation_2026_06_10.json"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "live closeout packet direct checks in operator handoff validator"
    - "live closeout packet direct checks in final bundle validator"
    - "regression tests for stale packet status and QKV command drift"
    - "updated choice/progress/handover/validation/log records"
  validation:
    - "python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tools/write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_third_goal_completion_audit.py"
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N38-CLOSEOUT-VALIDATION-DIRECT"
  runtime_note: "Spark spawn failed with agent thread limit reached; main agent completed bounded fallback."
```

```yaml
task_card:
  task_id: T-3G-N40-BLOCKER-GATE-DIRECT
  sub_agent: "gpt5.3-codex-spark attempted; codex-native fallback"
  role: "evaluator + implementer"
  objective: "Directly validate live C3b physical-smoke gate and XR-VITs gate audit JSON in final validators."
  file_ownership:
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_final_evidence_manifest.py"
    - "tools/write_spec_plan_conformance_audit.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "tests/test_write_third_goal_completion_audit.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "docs/resources/c3b_physical_smoke_gate_audit_2026_06_16.json"
    - "docs/resources/xr_vits_gate_audit_2026_06_16.json"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "live C3b physical-smoke gate direct checks in operator handoff validator"
    - "live XR-VITs gate direct checks in operator handoff validator"
    - "live C3b/XR gate direct checks in final bundle validator"
    - "validator freshness targets 131/131 and 152/152"
  validation:
    - "python3 -m py_compile tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_final_evidence_manifest.py tools/write_spec_plan_conformance_audit.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_third_goal_completion_audit.py"
    - "python3 -m unittest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_final_evidence_manifest tests.test_write_e2e_resource_policy_audit tests.test_write_spec_plan_conformance_audit tests.test_write_third_goal_completion_audit tests.test_write_third_goal_source_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N39-CLOSEOUT-PACKET-DIRECT"
  runtime_note: "Spark spawn failed with agent thread limit reached; main agent completed bounded fallback."
```

```yaml
task_card:
  task_id: T-3G-N43-MIRROR-INTEGRITY
  sub_agent: "gpt5.3-codex-spark"
  role: "analyst + implementer"
  objective: "Make final signoff evidence pass only when generated/signoff artifacts are mirrored into canonical docs/resources with matching size/SHA256 integrity."
  file_ownership:
    - "tools/run_third_goal_final_signoff.py"
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tools/write_spec_plan_conformance_audit.py"
    - "tests/test_run_third_goal_final_signoff.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "Spark subagent 019ecfbc-76cc-7622-b6fa-e0d062fa2bb9 read-only gap audit"
    - "hardware/generated/signoff/*"
    - "docs/resources/*"
  outputs:
    - "final-runner mirror-integrity summary fields"
    - "final evidence mirror-integrity consistency checks"
    - "operator handoff and bundle direct mirror-integrity gates"
    - "current-doc anchors for 347/131/152 targets"
  validation:
    - "python3 -m py_compile tools/run_third_goal_final_signoff.py tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tools/write_spec_plan_conformance_audit.py"
    - "python3 -m unittest tests.test_run_third_goal_final_signoff tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle tests.test_write_spec_plan_conformance_audit"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N42-INTAKE-PLAN-CROSSCHECK"
  runtime_note: "Spark completed read-only audit and identified the docs/resources mirror-integrity gap; main agent integrated and validated the fix."
```

```yaml
task_card:
  task_id: T-3G-N44-HANDOVER-RAW-VALIDATOR-FRESHNESS
  sub_agent: "codex-native with gpt5.3-codex-spark sidecar running T-3G-N44 resource-policy audit"
  role: "implementer + evaluator"
  objective: "Reject stale raw validator artifact-summary counts in HANDOVER even when live 131/131 and 152/152 anchors are present."
  file_ownership:
    - "tools/write_spec_plan_conformance_audit.py"
    - "tools/write_final_evidence_manifest.py"
    - "tests/test_write_spec_plan_conformance_audit.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/HANDOVER.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "docs/track/HANDOVER.md"
    - "docs/resources/final_operator_handoff_validation_2026_06_10.json"
    - "docs/resources/final_signoff_bundle_validation_2026_06_10.json"
  outputs:
    - "spec-plan conformance raw HANDOVER validator summary checks"
    - "final-evidence spec_plan_current_doc_freshness_contract expansion"
    - "spec-plan conformance target 86/86"
  validation:
    - "python3 -m py_compile tools/write_spec_plan_conformance_audit.py tools/write_final_evidence_manifest.py tests/test_write_spec_plan_conformance_audit.py tests/test_write_final_evidence_manifest.py"
    - "python3 -m unittest tests.test_write_spec_plan_conformance_audit tests.test_write_final_evidence_manifest"
    - "python3 tools/write_spec_plan_conformance_audit.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --json-out generated/signoff/spec_plan_conformance_audit_2026_06_10.json --markdown-out generated/signoff/spec_plan_conformance_audit_2026_06_10.md"
  dependencies:
    - "T-3G-N43-MIRROR-INTEGRITY"
  runtime_note: "Main agent completed bounded current-doc freshness hardening while Spark sidecar returned an independent resource-policy singularity recommendation."
```

```yaml
task_card:
  task_id: T-3G-N45-RESOURCE-POLICY-SINGULARITY
  sub_agent: "gpt5.3-codex-spark attempted, gpt5.5 read-only sidecar completed"
  role: "evaluator + implementer"
  objective: "Prevent stale or noncanonical e2e_resource_policy_audit sibling artifacts from being mistaken for current final-signoff evidence."
  file_ownership:
    - "tools/write_final_evidence_manifest.py"
    - "tools/validate_final_operator_handoff.py"
    - "tools/validate_final_signoff_bundle.py"
    - "tests/test_write_final_evidence_manifest.py"
    - "tests/test_validate_final_operator_handoff.py"
    - "tests/test_validate_final_signoff_bundle.py"
    - "generated/signoff/e2e_resource_policy_audit_2026_06_15.{json,md}"
    - "docs/CHOICE.md"
    - "docs/Sub-Plan.md"
    - "docs/Validation.md"
    - "docs/track/PROGRESS.md"
    - "docs/track/log.md"
  assigned_skill:
    - "agent-hierarchy-runtime-manager"
    - "hls-design-vitis"
    - "caveman:caveman"
  inputs:
    - "GPT5.5 sidecar resource-policy singularity audit"
    - "generated/signoff/e2e_resource_policy_audit_2026_06_10.{json,md}"
    - "generated/signoff/e2e_resource_policy_audit_2026_06_15.{json,md}"
    - "docs/resources/e2e_resource_policy_audit_2026_06_10.{json,md}"
  outputs:
    - "resource-policy canonical pair presence gate"
    - "docs/resources and generated/signoff singularity gates"
    - "filename/payload date-tag match gate"
    - "generated/docs canonical SHA256 mirror gate"
    - "stale generated 2026_06_15 pair removed"
  validation:
    - "python3 -m py_compile tools/write_final_evidence_manifest.py tools/validate_final_operator_handoff.py tools/validate_final_signoff_bundle.py tests/test_write_final_evidence_manifest.py tests/test_validate_final_operator_handoff.py tests/test_validate_final_signoff_bundle.py"
    - "python3 -m unittest tests.test_write_final_evidence_manifest tests.test_validate_final_operator_handoff tests.test_validate_final_signoff_bundle"
    - "python3 tools/run_third_goal_final_signoff.py --root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --allow-blocked"
  dependencies:
    - "T-3G-N43-MIRROR-INTEGRITY"
    - "T-3G-N44-HANDOVER-RAW-VALIDATOR-FRESHNESS"
  runtime_note: "GPT5.3-Codex-Spark returned usage-limit error; GPT5.5 sidecar completed read-only artifact inventory and gate recommendation; main agent integrated and validated."
```

```yaml
task_card:
  task_id: T-VREF-P0-02
  sub_agent: "gpt5.3-codex-spark"
  role: "analyst"
  objective: "Audit ME-ViT single-load memory policy against C3b URAM/global-buffer usage."
  file_ownership:
    - "analysis/vit-accel/papers/ME-ViT/analysis.md"
    - "hls/src/global_buffer.cpp"
    - "hls/src/weight_prefetcher.cpp"
    - "hls/src/hgtxr_e2e_axis_top.cpp"
  assigned_skill:
    - "fpga-asic-design-expert"
  inputs:
    - "ME-ViT paper analysis"
    - "C3b resource matrix"
  outputs:
    - "buffer lifetime audit"
    - "URAM/LUTRAM placement recommendation"
  validation:
    - "large buffers remain URAM candidates"
    - "small tables/FIFOs are not forced into URAM"
    - "successor promotion passes C3b protection checklist"
  dependencies:
    - "T-3G-C3B for final signoff"
```
