# Task-Card Operating Protocol (HBTXR)

> **Status:** adopted 2026-07-16 · **Scope:** standing operating protocol for the `HBTXR`
> workspace · **Standalone:** this document is self-contained and does not modify any other
> file. Wiring it into the tracking spine (CONTRIBUTING/ADR/INDEX/PROGRESS/log/CHANGELOG/memory)
> is a separate, future step — see [§9 Deferred integrations](#9-deferred-integrations).

---

## 1. Purpose & scope

All work in `HBTXR` is packaged and executed as **Task-Cards**: each unit of work is captured
as a small card, and cards are executed either **sequentially** or by **multiple agents in
parallel**, after which the main agent **integrates and verifies** the results.

This protocol is **additive and subordinate** to the existing HBTXR governance
(`docs/Master-Plan.md`, `docs/aegis/`, `docs/track/CONTRIBUTING.md`,
`docs/aegis/BASELINE-GOVERNANCE.md`). It changes **how work is packaged**, not **what is
allowed**. Adopting it authorizes **no** backlog execution — the standing state is
**"protocol confirmed, awaiting the first concrete task."**

It formalizes two things the repo did not yet have in one place:

1. an explicit **`model` / `effort`** dimension on the Task-Card (the existing `T-000…T-630`
   cards omit it), and
2. a written **operating loop** that encodes the Claude Code **runtime reality** (main agent +
   exactly one subagent level).

The **canonical existing card** (10 fields) lives at the end of
`docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md` and is used by `T-000…T-630`.
This protocol standardizes a **superset** of that card; it does not replace or re-issue it.

---

## 2. Augmented Task-Card schema

The card is the existing 10 fields **plus `model` and `effort`**. This fenced block is the
canonical blank template — copy it to author a new card.

```yaml
task_card:
  task_id: T-XXX            # (a) card number — unique, sortable (e.g. T-100, SI-010, RC-200)
  sub_agent: worker         # dispatch type: worker | evaluator | leader | expert-council | artifact-manager | skill-generator
  role: <human-readable>    # (b) Agent Role, e.g. "data-contract implementer"
  model: sonnet             # (d) opus | sonnet   (grandfather: if absent, resolve via the §3 policy table)
  effort: medium            # (d) high | medium | low
  assigned_skill: [ ... ]   # (c) role-based skill(s); bare ("worker-dispatch-manager") or namespaced ("aegis:writing-plans"); [] allowed
  objective: <one line>     # (e) imperative goal of the card
  file_ownership: [ ... ]   # (e) exact writable paths; DISJOINT sets across cards => those cards may run in parallel
  inputs: [ ... ]           # (e) read-only refs / prior card outputs / preconditions
  outputs: [ ... ]          # (e) artifacts the card produces
  validation: [ ... ]       # (e) how success is proven (tests, JSON/YAML parse, spec+quality review)
  dependencies: [ ... ]     # DAG edges: prior task_ids and/or named approval gates (e.g. "Gate B")
```

### 2.1 User's 5 requested fields → schema mapping

| Requested field | Schema field(s) |
|---|---|
| (a) Task-Card number | `task_id` |
| (b) Agent Role | `role` (+ `sub_agent` = the Agent-tool dispatch type) |
| (c) Skill(s) used, based on the role | `assigned_skill` |
| (d) Model / Effort used | `model` + `effort` **(new in this protocol)** |
| (e) Work content | `objective`, `file_ownership`, `inputs`, `outputs`, `validation`, `dependencies` |

> **Two role identifiers are kept on purpose.** `sub_agent` is the *machine dispatch type*
> (which Agent-tool subagent to launch); `role` is the *human-readable* role. Do not collapse
> them. `assigned_skill` accepts both bare and namespaced skill IDs.

---

## 3. Role → Model / Effort policy

New cards **must** set `model` and `effort` explicitly. Cards that omit them — including every
existing `T-000…T-630` card — are **grandfathered** and resolve their model/effort via this
table, so no existing card needs to be edited:

| Role class | model | effort |
|---|---|---|
| Master / Expert-Council / Leader / Runtime-Manager (main-agent orchestration phases) | `opus` | `high` |
| Worker (implementer) | `sonnet` | `medium` |
| Evaluator / mechanical / provenance | `sonnet` | `low`–`medium` |

Rationale: orchestration and design need the strongest model at high effort; per-card
implementation and verification are optimized for cost/speed on Sonnet. This mirrors the global
model/effort policy and is why the card carries the two fields at all.

---

## 4. Operating loop (runtime reality encoded)

**Runtime reality:** Claude Code is **one main agent + exactly one level of subagents**.
Subagents **cannot** spawn their own subagents. Therefore:

- **Master, Expert-Council, Leader, and Runtime-Manager are phases the *main agent* performs
  sequentially, in its own context** — they are *not* separate subagents by default.
- **Only Worker and Evaluator** (and **Artifact-Manager** / **Skill-Generator** when needed)
  are launched as **real `Agent`-tool subagents**.
- **The main agent always integrates, verifies, and reports.**

| # | Phase | Runtime | Model/Effort | Skill(s) |
|---|---|---|---|---|
| 1 | **Master — normalize** | main, in-context | opus / high | `master-input-normalizer` |
| 2 | **Expert Council — plan / blueprint** | main, in-context (optional `expert-council` subagent) | opus / high | `expert-council-planner`, `expert-council-blueprint-generator` |
| 3 | **Leader — DAG + cards** | main, in-context | opus / high | `execution-dag-manager`, `task-card-generator`, `task-card-skill-binder` |
| 3b | *(missing skill only)* generate skill | real `skill-generator` subagent | sonnet | `skill-generator-protocol`, `on-demand-skill-generator` |
| 4 | **Worker — execute** | **real `worker` subagent(s)** — parallel if `file_ownership` disjoint & deps met, else sequential | sonnet / medium | card-bound `assigned_skill` (dispatched via `worker-dispatch-manager`) |
| 5 | **Evaluator / Leader — verify** | **real `evaluator` subagent(s)** | sonnet | `evaluator-verification-manager` |
| 6 | **Main — integrate + report** | main, in-context | opus / high | `artifact-provenance-manager` (when artifacts produced), `memory-plan-generator` |

### Phase detail

1. **Master — normalize.** Turn the user request into a normalized objective, scope fence, and
   gate list. **Run the aegis drift check first** (`BASELINE-GOVERNANCE.md`:
   branch / HEAD / worktree / SHA / license / approval) before any card is issued.
2. **Expert Council — plan / blueprint.** Produce the team blueprint and success criteria.
   Reference `Master-Plan.md`'s Expert-Council / Team-Blueprint sections; do not re-derive them.
3. **Leader — DAG + cards.** Build the execution DAG from `dependencies`, emit cards from the
   §2 template, bind `assigned_skill` per role, and set `model`/`effort` per §3. Invoke a
   `skill-generator` subagent **only** if a required skill is missing from the pool.
4. **Worker — execute.** Dispatch real `worker` subagents. **Parallelize cards whose
   `file_ownership` sets are disjoint and whose `dependencies` are satisfied**; otherwise run
   sequentially. Each worker writes **only** its `file_ownership` paths (one-writer rule) and
   honors staged-allowlist and approval gates.
5. **Evaluator / Leader — verify.** Fresh implementer → fresh spec reviewer → fresh quality
   reviewer, matching the existing commit-and-review policy.
6. **Main — integrate + report.** Merge worker outputs, run integration verification, record
   evidence (see §6), report **progress vs plan**, and stage via **explicit allowlist**
   (never `git add -A`). Attach provenance when artifacts are produced.

---

## 5. Skill / agent binding (quick reference)

- **Card generation / binding:** `task-card-generator`, `task-card-skill-binder`,
  `skill-generator-protocol`, `on-demand-skill-generator`.
- **Orchestration (main-agent phases):** `master-input-normalizer`, `expert-council-planner`,
  `expert-council-blueprint-generator`, `execution-dag-manager`, `runtime-dag-scheduler`.
- **Execution / verification (real subagents):** `worker-dispatch-manager` (Worker),
  `evaluator-verification-manager` (Evaluator), `artifact-provenance-manager` (Artifact-Manager).
- **Agent-tool types available:** `expert-council`, `leader`, `worker`, `evaluator`,
  `artifact-manager`, `skill-generator`, plus `Explore` / `Plan` for read-only research and
  design.

---

## 6. Governance preservation

This protocol does not relax any existing rule. In every loop:

- **Git root = `HBTXR`.** Never stage sibling repositories. Use **explicit staged path
  allowlists**, never `git add -A`.
- **One writer per artifact** ≙ enforced by **disjoint `file_ownership`** across parallel cards.
- **Drift / authority check** (`BASELINE-GOVERNANCE.md`) runs at the Normalize phase of every
  loop; stop and revise on owner / source / license / tool / split / board drift.
- **Honest tool marking:** unavailable tools are reported, never inferred as a pass.
- **All existing backlog stays gated and unexecuted:** follow-up `T-000…T-630`, selective
  integration `SI-000…SI-200`, semantic refactor `RC-010…RC-900` / `CL-000…CL-131`. Adopting
  this protocol authorizes **no** backlog execution.
- **Separate approval still required** for: push, artifact-policy changes, licensed tools,
  boards, active promotion of license-unclear code, dependency installation, training, and
  HLS / synthesis / board actions.
- **Provenance preserved:** card executions still produce aegis work-slice evidence at
  `docs/aegis/work/<date-slice>/{10-intent,20-checkpoint,90-evidence,99-reflection}.md`, and
  produced artifacts are still registered in `docs/aegis/INDEX.md`.

---

## 7. Working-directory decision

- **Operating / git root = `HBTXR`**
  (`D:\dataset\EV_Eye\paper_works\HBTXR-Pool\HBTXR`), which is the git root
  (`git rev-parse --show-toplevel`).
- The **harness primary cwd cannot be changed** — it is the parent `HBTXR-Pool`. This protocol
  therefore **documents** `HBTXR` as the operating root and requires **all** card
  `file_ownership` / `inputs` / `outputs` paths to be expressed **relative to `HBTXR`** (or
  absolute under it). No system or configuration change is made by adopting this document.

---

## 8. Sample dry-run card (illustrative only)

The following shows the loop applied to a **hypothetical** task. It is **`NOT-FOR-EXECUTION`**
— it demonstrates the schema and does not authorize any work.

```yaml
# NOT-FOR-EXECUTION — illustration of the augmented schema only.
task_card:
  task_id: T-DRYRUN-001
  sub_agent: worker
  role: contract-test author
  model: sonnet
  effort: medium
  assigned_skill: [python-testing-patterns]
  objective: add a failing-then-passing unit test for the T-100 data contract
  file_ownership: [algorithm/hybrid/tests/handover/test_sample_tensor_contract.py]
  inputs: [docs/aegis/plans/2026-07-15-hbtxr-follow-up-implementation.md, algorithm/hybrid/src/data/contracts.py]
  outputs: [one new test module]
  validation: [pytest -q on the new module; spec review; quality review]
  dependencies: [T-100]
```

---

## 9. Deferred integrations

Intentionally **NOT** done now (per "document the protocol separately; do not modify existing
files"). Each is available on a later, separately approved request:

- Add a pointer bullet to `docs/track/CONTRIBUTING.md`.
- Add an index row to `docs/aegis/INDEX.md`
  (`2026-07-16 | protocol | Task-Card operating protocol | docs/track/TASK-CARD-PROTOCOL.md | adopted`).
- Add **ADR-008** to `docs/track/ADR.md` (adopt protocol; augment schema; fix operating root;
  encode single-subagent-level runtime).
- Add dated entries to `docs/track/PROGRESS.md`, `docs/track/log.md`, and
  `docs/track/CHANGELOG.md`.
- Split the §2 template into a reusable `docs/track/templates/task-card.template.yaml`.
- Persist a `feedback`-type memory so future sessions auto-apply this protocol.
