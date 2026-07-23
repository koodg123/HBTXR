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

## Standalone HBTXR Planning Task Cards

```yaml
task_card:
  task_id: T-301
  sub_agent: codex-native
  role: research analyst
  objective: classify standalone annotation-analysis content at file level
  file_ownership: []
  assigned_skill: [evidence-based codebase analysis]
  inputs: [standalone annotation revision 2ff5262, current target owners]
  outputs: [candidate, exclusion, dependency, and coordinate-risk matrix]
  validation: [exact paths, counts, API evidence, no writes]
  dependencies: []
```

```yaml
task_card:
  task_id: T-302
  sub_agent: codex-native
  role: license and dependency analyst
  objective: classify FECET, HG-PIPE Quantization, and SWIFT implementation roots
  file_ownership: []
  assigned_skill: [repository provenance and dependency audit]
  inputs: [standalone references/impl, current archives and quantization owners]
  outputs: [license, equivalence, dependency, and promotion matrix]
  validation: [license paths, normalized comparison, exact targets, no writes]
  dependencies: []
```

T-303 target architecture analysis could not be dispatched because the native
agent concurrency limit was reached; the Master performed that role directly.
Future execution tasks SI-000 through SI-200, exclusive file ownership, gates,
commands, commits, and rollback are canonical only in
`docs/aegis/plans/2026-07-15-standalone-hbtxr-selective-integration.md`.

## Semantic Census Task Cards

```yaml
task_card:
  task_id: T-701
  sub_agent: codex-native
  role: architecture researcher
  objective: trace maintained entry-to-domain-to-integration-to-test flows
  file_ownership: []
  assigned_skill: [agentops:codebase-recon]
  inputs: [committed blobs at ebe862b, package/config manifests, tests]
  outputs: [typed architecture findings with exact path-line evidence]
  validation: [read-only, no project execution or edits]
  dependencies: []
```

```yaml
task_card:
  task_id: T-702
  sub_agent: codex-native
  role: Python static-analysis evaluator
  objective: audit functions, classes, complexity, duplicates, cycles, side effects, and dead-code candidates
  file_ownership: []
  assigned_skill: [agentops:codebase-recon evidence discipline]
  inputs: [all tracked Python separated by authority zone]
  outputs: [symbol counts and prioritized line findings]
  validation: [stdlib AST only, no imports, installs, tests, or edits]
  dependencies: []
```

```yaml
task_card:
  task_id: T-703
  sub_agent: codex-native
  role: repository hygiene and non-Python analyst
  objective: audit HDL, C++, Tcl, shell, configs, generated artifacts, archives, and layout drift
  file_ownership: []
  assigned_skill: [agentops:codebase-recon evidence discipline]
  inputs: [tracked non-Python tree and provenance/layout docs]
  outputs: [language census, permanent reference-retention catalog candidates, and separately gated non-reference cleanup candidates]
  validation: [read-only, no build/test/experiment or edits]
  dependencies: []
```

The Master generated and validated the machine inventory, integrated all three
reports, and wrote the canonical RC/CL execution DAG. Future implementation
Task Cards are defined only in the semantic refactor/cleanup plan.

```yaml
task_card:
  task_id: T-704
  sub_agent: codex-native
  role: evidence evaluator
  objective: challenge census completeness, source isolation, counts, and claim evidence
  file_ownership: []
  assigned_skill: [agentops:codebase-recon]
  inputs: [recon pack, inventory CSV/JSON, semantic report]
  outputs: [severity-ranked evidence review]
  validation: [committed-blob isolation, exact counts, resolvable evidence]
  dependencies: [T-701, T-702, T-703]
```

```yaml
task_card:
  task_id: T-705
  sub_agent: codex-native
  role: plan evaluator
  objective: challenge refactor and cleanup plan executability, compatibility, dependencies, and destructive gates
  file_ownership: []
  assigned_skill: [aegis:writing-plans]
  inputs: [semantic plan, decision matrix, repository manifests]
  outputs: [severity-ranked plan review]
  validation: [exact paths, deterministic commands, package and cleanup gates]
  dependencies: [T-704]
```

```yaml
task_card:
  task_id: T-706
  sub_agent: codex-native
  role: repository and documentation evaluator
  objective: verify that only analysis artifacts changed and all validation claims match command evidence
  file_ownership: []
  assigned_skill: [agentops:codebase-recon evidence discipline]
  inputs: [git diff/status, tracking docs, validators]
  outputs: [scope and documentation consistency review]
  validation: [no production change, no deletion, no commit or push, explicit validator limitation]
  dependencies: [T-704, T-705]
```

T-704 through T-706 are read-only adversarial reviews. Their findings are
integrated by the Master; only the Master edits the analysis and planning artifacts.

```yaml
task_card:
  task_id: T-707
  sub_agent: codex-native
  role: architecture evaluator
  objective: verify algorithm modality ownership and determine whether Hybrid namespace migration is justified
  file_ownership: []
  assigned_skill: [aegis:writing-plans]
  inputs: [algorithm package manifests, provenance, bootstrap, consumers, semantic plan]
  outputs: [RC-100 decomposition and downstream dependency corrections]
  validation: [read-only evidence, no source changes, explicit NO-OP trigger]
  dependencies: [T-705]
```

```yaml
task_card:
  task_id: T-708
  sub_agent: codex-native
  role: governance evaluator
  objective: reconcile reference retention, licensing, and cleanup work items
  file_ownership: []
  assigned_skill: [aegis:writing-plans]
  inputs: [references policy, decision matrix, semantic plan, tracking docs]
  outputs: [permanent-retention policy and cancelled destructive tasks]
  validation: [zero reference delete/rename/hash drift, no external-pointer replacement]
  dependencies: [T-705]
```

```yaml
task_card:
  task_id: T-709
  sub_agent: codex-native
  role: plan integrator
  objective: synchronize the accepted algorithm and reference decisions across durable plans
  file_ownership: [docs planning and tracking artifacts only]
  assigned_skill: [aegis:writing-plans]
  inputs: [T-707 and T-708 findings, user decisions]
  outputs: [updated canonical plan, matrix, requirements, ADR, backlog, validation]
  validation: [no production edits, branch unchanged, Markdown/diff consistency]
  dependencies: [T-707, T-708]
```

## Session Handover Task Cards — 2026-07-16

```yaml
task_card:
  task_id: T-711
  sub_agent: codex-native
  role: repository state auditor
  objective: collect live branch, HEAD, dirty, commit, tracker, and worktree evidence
  file_ownership: []
  assigned_skill: [agentops:handoff]
  inputs: [Git state, plans, tracking docs]
  outputs: [evidence summary and first action]
  validation: [read-only, no mutation]
  dependencies: []
```

```yaml
task_card:
  task_id: T-712
  sub_agent: codex-native
  role: conversation and decision auditor
  objective: reconstruct decisions, completed/open tasks, blockers, and read order
  file_ownership: []
  assigned_skill: [agentops:handoff, aegis:writing-plans]
  inputs: [CONVERSATION, PROGRESS, TODO, ADR, canonical plans]
  outputs: [decision timeline and continuation facts]
  validation: [path/task evidence, no edits]
  dependencies: []
```

```yaml
task_card:
  task_id: T-713
  sub_agent: codex-native
  role: handover integrator
  objective: write the authoritative HANDOVER and pointer prompt
  file_ownership: [.agents/handoff/2026-07-16-hbtxr-semantic-refactor-planning.md, .agents/handoff/2026-07-16-hbtxr-semantic-refactor-planning-prompt.md, docs planning/tracking records]
  assigned_skill: [agentops:handoff]
  inputs: [live Git evidence, user decisions, T-711/T-712 scope]
  outputs: [two validated Markdown artifacts and synchronized tracking]
  validation: [required headings, metadata regex, pointer markers, scope diff]
  dependencies: [T-711, T-712]
```

```yaml
task_card:
  task_id: T-714
  sub_agent: codex-native
  role: handover evaluator
  objective: independently verify evidence, format, restart quality, and no source mutation
  file_ownership: []
  assigned_skill: [agentops:handoff]
  inputs: [handover, prompt, live Git state]
  outputs: [PASS or blocking findings]
  validation: [read-only, exact HEAD/status/markers]
  dependencies: [T-713]
```

T-711 and T-712 were dispatched as real read-only agents but were interrupted after WSL shell latency prevented a timely result. The Master collected the same evidence directly. T-714 remains the independent completion gate.

## Algorithm Naming And Pools Planning Task Cards — 2026-07-21

```yaml
task_card:
  task_id: T-715
  sub_agent: codex-native
  role: research analyst
  objective: census active FACET names, reference/history exclusions, pool consumers, and path collisions
  file_ownership: []
  assigned_skill: [architecture]
  inputs: [live algorithm tree, semantic plan, user scope]
  outputs: [exact match counts, consumer map, collision and validation evidence]
  validation: [read-only, branch and HEAD recorded, no algorithm diff]
  dependencies: []
```

```yaml
task_card:
  task_id: T-716
  sub_agent: codex-native
  role: architecture evaluator
  objective: design the minimum-entropy active-name and src.pools retirement boundary
  file_ownership: []
  assigned_skill: [architecture, aegis:writing-plans]
  inputs: [T-715 evidence, current domain owners and consumers]
  outputs: [canonical replacements, owner map, execution DAG, retirement gates]
  validation: [EvEye and modality preservation, no duplicate registry owner]
  dependencies: [T-715]
```

```yaml
task_card:
  task_id: T-717
  sub_agent: codex-native
  role: plan consistency evaluator
  objective: identify every durable planning artifact and dependency that must reflect RC-120 and RC-130
  file_ownership: []
  assigned_skill: [aegis:writing-plans]
  inputs: [canonical plan, Master/Sub/Spec/Execution/Validation, tracking docs]
  outputs: [document update map and stale-plan conflicts]
  validation: [read-only, unique task and ADR IDs, dependency consistency]
  dependencies: [T-715]
```

```yaml
task_card:
  task_id: T-718
  sub_agent: codex-native
  role: plan evaluator
  objective: verify the revised semantic plan is complete, aligned with the specification, and implementation-ready after its gates
  file_ownership: []
  assigned_skill: [aegis:writing-plans]
  inputs: [revised canonical plan, revised specification]
  outputs: [approved or blocking plan review]
  validation: [read-only, no source edits, serious gaps only]
  dependencies: [T-716, T-717]
```

T-715 through T-717 are read-only planning agents. The Master alone integrates
their findings into documentation. T-718 is the independent post-edit review.
RC-120 and RC-130 execution Task Cards remain canonical only in the semantic
refactor/cleanup plan.

T-718 initially found that frozen-zone verification could miss changes already
captured in intermediate commits. The Master changed RC-120/RC-900 to compare
the approved baseline with current path/blob state and untracked additions for
every frozen zone. T-718 then returned `Approved` with no blocking issue.

```yaml
task_card:
  task_id: T-719
  sub_agent: codex-native
  role: contract analyst
  objective: replace the proposed HBTXR-prefixed environment names with existing neutral config or semantic CLI controls
  file_ownership: []
  assigned_skill: [aegis:writing-plans]
  inputs: [RC-120, common launchers and utilities, current config keys]
  outputs: [exact neutral mapping, affected documents, compatibility and validation risks]
  validation: [read-only, no new product-prefixed control, no source edits]
  dependencies: [T-718]
```

T-719 confirmed that `runtime.disable_cudnn`, `trainer.devices`, and
`train.ckpt_path` already own the common controls. Only the standalone dataset
builder needs `--disable-cudnn`; no `HBTXR_*` replacement environment variable
is planned. Existing HBTXR project/model/config identity is outside this narrow
replacement-policy revision.

The independent T-718 reviewer rechecked this follow-up revision and returned
`Approved`: Spec, ADR, Validation, compatibility scope, and neutral control
mapping are consistent, with no blocking issue.

## Algorithm Modular Ownership Planning Task Cards — 2026-07-21

```yaml
task_card:
  task_id: T-724
  sub_agent: codex-native
  role: analyst
  objective: synthesize the preserved and active target algorithm tree with exact source-to-owner mapping
  file_ownership: []
  assigned_skill: [architecture, aegis:writing-plans]
  inputs: [live algorithm tree, user decisions, semantic plan]
  outputs: [target tree, mapping table, collision evidence]
  validation: [read-only, preserved paths asserted, no source edits]
  dependencies: []
```

```yaml
task_card:
  task_id: T-725
  sub_agent: codex-native
  role: architecture expert
  objective: define named package, compatibility, wheel, editable and direct-script migration boundaries
  file_ownership: []
  assigned_skill: [architecture, aegis:writing-plans]
  inputs: [pyproject, EvEye consumers, Hybrid src consumers, target tree]
  outputs: [package strategy, migration phases, exact verification]
  validation: [read-only, no generic bare imports, no edits]
  dependencies: []
```

```yaml
task_card:
  task_id: T-726
  sub_agent: codex-native
  role: evaluator
  objective: identify every durable document and prior decision superseded by the new ownership plan
  file_ownership: []
  assigned_skill: [aegis:writing-plans, architecture]
  inputs: [canonical plans, Master/Sub/Spec/Execution/Validation, tracking docs]
  outputs: [document change matrix and consistency gates]
  validation: [read-only, exact path evidence, no edits]
  dependencies: []
```

T-724 through T-726 were executed as real read-only agents. The Master alone
integrates their results. A fresh post-edit reviewer must evaluate this new
owner policy; the prior T-718 approval applies only to the superseded policy.

```yaml
task_card:
  task_id: T-727
  sub_agent: codex-native
  role: evaluator
  objective: independently verify the final modular ownership plan for ordering, dependency direction, packaging, preservation, and runnable gates
  file_ownership: []
  assigned_skill: [aegis:writing-plans, architecture]
  inputs: [AM-000 through AM-900, live import evidence, preserved-zone baseline]
  outputs: [approved or blocking plan review]
  validation: [read-only, git diff check, no production edits]
  dependencies: [T-724, T-725, T-726]
```

T-727 reviewed three revisions. The first two reviews exposed dependency-bearing
test/wheel mixing, lower-owner demo imports, and a move-before-extract ordering
error. After separating behavioral and isolated-wheel environments, adding
reverse-import gates, and moving both demo extractions ahead of their source
moves, T-727 returned `Approved` with no blocking issue.
