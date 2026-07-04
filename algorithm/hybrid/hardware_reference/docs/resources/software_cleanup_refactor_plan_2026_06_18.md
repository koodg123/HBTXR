# HGTXR Software Cleanup And Refactor Plan

Date: 2026-06-18

## Scope

Working directory:

- `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software`

Experiments remain paused. This cleanup pass did not delete, move, train, or
evaluate anything. It produced inventory and refactor-risk evidence first.

## Prompt Brief

Goal:

- Reduce software directory clutter without breaking experiment reproducibility.
- Prepare safe refactor phases for `runs/`, `scripts/external`, `configs/external`, root shim files, and docs.

Evidence sources:

- `docs/resources/software_cleanup_inventory_2026_06_18.md`
- `docs/resources/software_cleanup_inventory_2026_06_18.json`
- `docs/resources/software_path_reference_audit_2026_06_18.md`
- `docs/resources/software_path_reference_audit_2026_06_18.json`
- current pause/status docs under `docs/resources/`
- tracking docs under `docs/track/`

Constraints:

- No train/eval launch while pause directive is active.
- No destructive deletion without a frozen preserve list.
- Do not rename `anlaysis` yet; current links depend on that spelling.
- Do not move `scripts/external/run_xr*.sh` until compatibility wrappers exist.

Acceptance criteria:

- Cleanup candidates are evidence-backed.
- Current XR-64 resume checker/status reporter keep working.
- Leader and teacher checkpoint chains remain discoverable.
- Historical docs remain readable, even if archived later.

## Current Findings

Top-level size drivers:

| Area | Size | Decision |
|---|---:|---|
| `runs/` | `44.5 GiB` | classify first, archive/delete later |
| `.venv/` | `4.8 GiB` | keep local-only; already ignored by root gitignore |
| `data/` | `43.7 MiB` | preserve manifests; inspect generated subsets separately |
| `src/` | `2.6 MiB` | code refactor only after tests are green |
| `scripts/` | `1.6 MiB` | group logically before physical moves |
| `anlaysis/` | `1.4 MiB` | preserve misspelled path for compatibility |
| `docs/` | `1.3 MiB` | archive historical docs only after link index update |

Dirty state under software scope:

- `236` changed/untracked entries.
- `27` modified entries.
- `209` untracked entries.

Root file findings:

- legacy shim candidates: `config.py`, `dataset.py`, `evaluator.py`, `event_repr.py`, `geometry.py`, `heads.py`, `inference.py`, `io.py`, `losses.py`, `metrics.py`, `model.py`, `runtime.py`, `scheduler.py`, `trainer.py`, `transforms.py`, `visualization.py`
- stray nonstandard file: `32.56462665285383`, currently empty.

`runs/` classification:

| Label | Count | Action |
|---|---:|---|
| `preserve_or_review` | `390` | freeze before any archive/delete |
| `archive_candidate_eval` | `315` | archive candidate after reference check |
| `archive_candidate_closed_experiment` | `130` | archive candidate after leader check |
| `cleanup_candidate_smoke` | `8` | first deletion/archive candidate class |
| `archive_candidate_duplicate` | `6` | compare metrics before archive |

Path coupling:

| Finding kind | Count |
|---|---:|
| `project_root` | `411` |
| `.venv/bin/python` | `318` |
| `canonical_root` | `254` |
| `/home/...` absolute path | `130` |
| `PATHS_CONFIG`/paths config mentions | `51` |
| `/mnt/...` absolute path | `1` |
| Windows drive path | `1` |

Highest-coupled files:

- `scripts/external/run_prepare_and_train.sh`
- `configs/external/paths/reference_path.json`
- `scripts/external/run_followup_queue_20260611.sh`
- `scripts/external/_config.py`
- `scripts/external/run_mode2_end_to_end.sh`

## Must-Preserve Run Families

Do not delete or move these until XR-64 resume completes or a new authority
document supersedes them:

- XR-62A center leader and related eval rows.
- XR-39 P10 mixed-leader soup.
- XR-56B P5/direct P10-P5 soft branch.
- XR-58A P10 teacher branch.
- XR-63 oracle evidence.
- XR-64 smoke/prep traces and future generated target workspace.
- `runs/interpolated_checkpoints/`
- `runs/_logs/`
- `runs/diagnostics/`

## Agent Task Cards

```yaml
task_card:
  task_id: CLEAN-001
  sub_agent: gpt5.3-codex-spark
  role: analyst
  objective: Read-only classify runs cleanup risk and must-preserve patterns
  file_ownership: []
  assigned_skill: [codebase-cleanup-refactor-clean]
  inputs: [runs, docs/resources, docs/track]
  outputs: [classification rules, must-preserve patterns, delete-risk warnings]
  validation: [no deletion without reference check]
  dependencies: []
```

Result:

- Completed by real sub-agent.
- Integrated preserve list and metadata requirements.

```yaml
task_card:
  task_id: CLEAN-002
  sub_agent: gpt5.3-codex-spark
  role: analyst
  objective: Read-only analyze scripts/config/docs refactor risks
  file_ownership: []
  assigned_skill: [bash-defensive-patterns, documentation]
  inputs: [scripts/external, configs/external, docs/resources, docs/track]
  outputs: [move-map proposal, wrapper policy, validation commands]
  validation: [path reference checks]
  dependencies: []
```

Result:

- Completed by real sub-agent.
- Integrated wrapper-first policy and path-coupling risks.

## Refactor Strategy

### Phase 1: Inventory Freeze

Status: completed for initial pass.

Artifacts:

- `scripts/external/maintenance/inventory_software_tree.py`
- `scripts/external/maintenance/audit_path_references.py`
- `docs/resources/software_cleanup_inventory_2026_06_18.md`
- `docs/resources/software_path_reference_audit_2026_06_18.md`

### Phase 2: Safe No-Move Cleanup

Recommended next step, after explicit approval:

- delete `__pycache__/` and `tests/__pycache__/` only.
- remove empty stray file `32.56462665285383` only if user confirms it is accidental.
- keep `.venv/` untouched.

### Phase 3: Runs Archive Plan

Status: catalog view created; no deletion.

Created:

- `scripts/external/maintenance/build_runs_catalog.py`
- `docs/resources/runs_catalog_report_2026_06_18.md`
- `docs/resources/runs_catalog_report_2026_06_18.json`
- `runs/_catalog/<category>/<run_id>` symlinks

Catalog policy:

- original `runs/<run_id>` directories remain in place.
- category entries are symlinks only.
- existing scripts/docs can keep using original paths.
- archive/delete remains blocked until preserve manifest and dependency report exist.

Required before archive/delete:

1. extract `eval_summary.json` metrics from all run dirs.
2. build leader checkpoint table.
3. calculate doc reference count per run.
4. mark dependencies where one run checkpoint feeds another run.
5. archive only `cleanup_candidate_smoke` and confirmed duplicates first.

Target output:

- `docs/resources/software_runs_archive_candidates_2026_06_18.md`
- `docs/resources/software_runs_preserve_manifest_2026_06_18.json`

### Phase 4: Wrapper-First Script Refactor

Do not physically move scripts yet.

First normalize active scripts:

- centralize `PYTHON_BIN` resolution.
- make `CANONICAL_ROOT` env-overridable everywhere.
- add optional `PATHS_CONFIG` bridge.
- keep existing script basenames.

Only after that, move scripts into logical folders with compatibility wrappers.

### Phase 5: Config Path Policy

Use overlays:

- `configs/external/paths/ev_eye_raw_paths.json` remains local machine override.
- add or refresh `configs/external/paths.example.json` as portable template.
- do not put historical absolute paths in active config defaults.

### Phase 6: Root Shim Refactor

Before deleting any root `*.py` shim:

1. run `rg -n "import (config|dataset|model|trainer|...)"`.
2. check tests and old scripts.
3. replace with `src/hbtxr/...` imports or keep shim with explicit comment.

### Phase 7: Docs Archive

Do not rewrite historical commands in old docs.

Instead:

- preserve old docs as evidence.
- add supersession notes to current authority docs.
- move only after link index is updated.

## Validation Commands

```bash
python3 -m py_compile scripts/external/maintenance/inventory_software_tree.py
python3 -m py_compile scripts/external/maintenance/audit_path_references.py
.venv/bin/python scripts/external/maintenance/inventory_software_tree.py
.venv/bin/python scripts/external/maintenance/audit_path_references.py
.venv/bin/python scripts/external/maintenance/build_runs_catalog.py
.venv/bin/python scripts/external/report_second_goal_status.py --format summary
git diff --check -- scripts/external/maintenance docs/resources docs/track docs/Validation.md
pgrep -af "train_hbtxr|eval_hbtxr|run_xr64|run_xr|scripts/external/run_.*train"
```

## Current Decision

Proceed only with evidence-building and low-risk cleanup candidates. Defer
physical script/config/docs moves until path wrappers and preserve manifests are
complete.
