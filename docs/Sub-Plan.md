# HBTXR Commit And HANDOVER Analysis Sub-Plan

## Task Cards

```yaml
task_card:
  task_id: T-001
  sub_agent: codex-native
  role: evaluator
  objective: verify and checkpoint the plan files only
  file_ownership: [.agents/plans, docs]
  assigned_skill: [agentops:plan, git-progress-release-checkpoint]
  inputs: [live Git baseline]
  outputs: [plan commit]
  validation: [staged allowlist, git diff --cached --check]
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: codex-native
  role: implementer
  objective: checkpoint inactive FECET and SWIFT archive sources
  file_ownership: [algorithm/archive/imports/fecet_hbtxr, algorithm/archive/imports/swift_hbtxr]
  assigned_skill: [git-progress-release-checkpoint]
  inputs: [validated archive tree]
  outputs: [archive commit]
  validation: [Python compile, shell syntax, staged allowlist]
  dependencies: [T-001]
```

```yaml
task_card:
  task_id: T-003
  sub_agent: codex-native
  role: implementer
  objective: checkpoint EV-Eye and EX-Gaze analysis imports
  file_ownership: [algorithm/analysis, algorithm/docs/track/PROGRESS.md, algorithm/docs/track/IMPL_REPOS_HANDOVER_IMPORT_2026_07_08.md]
  assigned_skill: [git-progress-release-checkpoint]
  inputs: [analysis scripts, configs, compact results]
  outputs: [analysis import commit]
  validation: [JSON parse, Python compile/import, artifact exclusion, staged allowlist]
  dependencies: [T-002]
```

```yaml
task_card:
  task_id: T-004
  sub_agent: codex-native
  role: implementer
  objective: checkpoint hardware and legacy reference evidence
  file_ownership: [hardware/configs/xr_accel, hardware/docs, references/legacy-codebase/hardware/hgpipe_iccad24_docs]
  assigned_skill: [git-progress-release-checkpoint]
  inputs: [selected docs, configs, reports, headers]
  outputs: [hardware evidence commit]
  validation: [JSON parse, shell syntax, artifact exclusion, staged allowlist]
  dependencies: [T-003]
```

```yaml
task_card:
  task_id: T-005
  sub_agent: codex-native
  role: research
  objective: pin and inventory all HANDOVER repositories and collisions
  file_ownership: [docs/analysis/HANDOVER-INVENTORY.md]
  assigned_skill: [agentops:plan]
  inputs: [HANDOVER repositories, HBTXR]
  outputs: [provenance and collision inventory]
  validation: [repo SHA, counts, blob/path comparison, exclusions]
  dependencies: [T-004]
```

```yaml
task_card:
  task_id: T-006
  sub_agent: codex-native
  role: analyst
  objective: analyze HGTXR software gaps against current algorithm/hybrid
  file_ownership: [docs/analysis/HANDOVER-SOFTWARE-AUDIT.md]
  assigned_skill: [agentops:plan]
  inputs: [HGTXR software, HBTXR hybrid, tests/configs]
  outputs: [behavior/API/test candidate map]
  validation: [symbol paths, contract tests, no bulk-copy recommendations]
  dependencies: [T-005]
```

```yaml
task_card:
  task_id: T-007
  sub_agent: codex-native
  role: analyst
  objective: analyze HGTXR and XR_Accel hardware/automation gaps
  file_ownership: [docs/analysis/HANDOVER-HARDWARE-AUDIT.md]
  assigned_skill: [agentops:plan]
  inputs: [HGTXR hardware, XR_Accel, ViT_Accel, HBTXR hardware]
  outputs: [interface/config/test/toolchain candidate map]
  validation: [module and symbol evidence, board/tool dependencies]
  dependencies: [T-005]
```

```yaml
task_card:
  task_id: T-008
  sub_agent: codex-native
  role: evaluator
  objective: classify license, data, artifact, and regression risks
  file_ownership: [docs/analysis/HANDOVER-LICENSE-ARTIFACT-AUDIT.md]
  assigned_skill: [agentops:plan]
  inputs: [all candidates and LICENSE/NOTICE files]
  outputs: [risk register and promotion gates]
  validation: [every candidate has license and artifact disposition]
  dependencies: [T-005]
```

```yaml
task_card:
  task_id: T-009
  sub_agent: codex-native
  role: expert
  objective: synthesize the integration matrix and prioritized recommendations
  file_ownership: [docs/analysis/HANDOVER-INTEGRATION-MATRIX.md, docs/analysis/HANDOVER-RECOMMENDATIONS.md]
  assigned_skill: [agentops:plan]
  inputs: [T-005, T-006, T-007, T-008 outputs]
  outputs: [decision-ready recommendation package]
  validation: [complete matrix rows, council review, no implementation]
  dependencies: [T-006, T-007, T-008]
```

## Sequential Waves

- Wave 0: T-001.
- Wave 1: T-002, then T-003, then T-004; serialized to keep commits reviewable.
- Wave 2: T-005.
- Wave 3: T-006, then T-007, then T-008 sequentially, despite disjoint files.
- Wave 4: T-009 and Master tracking updates.

## File Ownership Rule

Only the named owner writes each analysis file. The Master alone updates
`docs/Execution.md`, `docs/Validation.md`, and `docs/track/*` after a wave.

## Follow-up Task Cards

Task Cards T-000 through T-630, ownership, commands, dependencies, approval
gates, and rollback rules are maintained in
`docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md`. They supersede
the completed analysis Task Cards only for the future implementation workstream.
