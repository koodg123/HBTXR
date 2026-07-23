# Validation Plan

## Checkpoint Gates

| Gate | Command or evidence | Pass condition |
|---|---|---|
| Repo boundary | `git rev-parse --show-toplevel` | ends in `/HBTXR` |
| Staged scope | `git diff --cached --name-status` | only slice allowlist |
| Diff integrity | `git diff --cached --check` | exit 0, or a reviewer-approved provenance-only waiver with exact file/class/count |
| JSON | parse imported JSON files | 0 parse errors |
| Python | compile/import targeted Python files | 0 unexpected errors |
| Shell | `bash -n` on imported shell files | 0 errors |
| Artifact | size/type/secret/nested-repo scan | prohibited files 0 |
| Commit | `git log -1 --oneline` | expected message and new SHA |

Current pre-plan evidence: 105 JSON, 81 Python, and 26 shell files passed; no
large file over 10 MiB, nested Git repo, submodule, symlink, or credential-like
filename was found in the current checkpoint scope.

## Executed Checkpoint Evidence

| Commit | Scope | Key validation | Result |
|---|---|---|---|
| `9535d24` | plan/tracking | 17-file allowlist, diff check, spec and quality review | PASS |
| `918d0fd` | inactive FECET/SWIFT archives | Python 71/71, shell 26/26, artifact scan | PASS with 25-file-format findings waived only for inherited EOF blanks |
| `0901a96` | EV-Eye/EX-Gaze evidence | Python 10/10, JSON 17/17, CSV 187/187, artifact scan | PASS with 16,743 inherited CSV CRLF findings waived |
| `6b880d5` | hardware/reference evidence | JSON 88/88, Markdown UTF-8 98/98, artifact and active-source isolation | PASS with exactly 60 enumerated provenance-format findings waived |

The waivers do not apply to active source, future edits, or unlisted formatting
classes. Runtime, training, synthesis, and board reproduction were not claimed.

## HANDOVER Analysis Gates

| Gate | Pass condition |
|---|---|
| Provenance | every source has branch, SHA, path, and mapping |
| Collision | identical content classified NO-OP |
| License | every active/adapt candidate has confirmed license scope |
| API | current target symbols and required adaptation identified |
| Data | shape, dtype, polarity, sequence, coordinate, and metric contract recorded |
| Tests | first regression or contract test named |
| Hardware | interface, pragma, config, golden vector, and tool/board dependency recorded |
| Artifact | generated/binary/cache disposition explicit |
| Recommendation | every matrix field complete and priority justified |

The C4 package contains all six required artifacts under `docs/analysis/` and
uses the common classification vocabulary `NO-OP`, `REFERENCE_ONLY`, `ADAPT`,
`ACTIVE_CANDIDATE`, `EXCLUDED`, and `LICENSE_BLOCKED`.

## Remaining Risk

Full training, FPGA tool execution, board measurements, and external artifact
availability cannot be verified in the analysis phase. They must remain
explicitly `UNAVAILABLE` or `PENDING`, not inferred as passing.

## Follow-up Plan Validation — 2026-07-15

The follow-up plan is documentation-only. Its acceptance checks are:

- exactly one sequential section and one Task Card for each of T-000 through
  T-630;
- explicit E1-E6 coverage and separate dependency, experiment, C-sim,
  synthesis, board, line-ending/LFS, Spec-Kit, and push gates;
- exact file ownership, commands, expected result, commit boundary, and
  rollback/stop condition for every task;
- current HBTXR remains the runtime owner and HANDOVER remains read-only;
- `git diff --check` passes and no source, generated artifact, install, run,
  board, remote, or commit action occurs during planning.

The plan must receive a fresh specification/executability review followed by a
fresh documentation-quality review before it is presented as execution-ready.

Final evidence: both independent reviews passed; 22 task headings and 22 Task
Cards match in order and are unique; all required Task Card fields are present;
`git diff --check` passes; only documentation changed during planning.

## Code/File-Only Execution Validation — 2026-07-15

| Task | Focused evidence | Result |
|---|---|---|
| T-010 | 19 source-registry tests plus live local revision/hash validation | PASS |
| T-020 | 11 artifact-manifest tests, schema/example validation, payload scan | PASS |
| T-100 | 14 data-contract tests and Python compile | PASS |
| T-110 | 9 evaluation-bridge tests and Python compile | PASS |
| T-210 | 14 XR schema tests, JSON parse, Python compile, 8/18 graph check | PASS |

The final consolidated run repeated all selected focused suites and validators,
Python compilation, JSON parsing, repository-boundary, CR-at-EOL-aware
diff-integrity, and fsck checks.
All commands passed: 67 tests, 11 registry candidates, and one artifact manifest.
The post-commit clean-worktree check remains the final closeout action.
Installation, full runtime/training reproduction, experiments, HLS/synthesis,
board validation, merge, release, and push are not covered or inferred as passing.

## Semantic Census Validation — 2026-07-15

Validation must prove:

- `semantic-summary.json` commit equals `ebe862b` and tracked file count equals 12,856;
- `file-inventory.csv` has one header plus 12,856 rows;
- `python-symbols.csv` and `nonpython-symbols.csv` line evidence resolves to tracked files;
- maintained runtime syntax errors are zero; invalid type comments are classified separately;
- all fact/inference evidence in `codebase-recon.json` resolves;
- the stored Python equivalent validator passes; the installed shell validator is separately recorded as blocked by CRLF and missing `jq`;
- no production source, remote, commit, or deletion action is included in the census.

The plan additionally requires exact CLI/schema/contract snapshots and a delta
recon before any old owner, Quantization shim, or non-reference binary is retired.
Validation fails on any `references/**` deletion, rename, deduplication,
external-pointer replacement, or content-hash drift. Hybrid namespace migration
remains NO-OP unless supported-workflow evidence and explicit approval are present.

## Session Handover Validation — 2026-07-16

Validation requires both handover files to be non-empty, all required headings and metadata fields to match the `agentops:handoff` format, the prompt to contain `Read first:` and `First action:`, live HEAD to equal the recorded 40-hex SHA, no production/reference diff, and an independent read-only PASS. The inherited dirty worktree is expected and must be described rather than cleaned. AgentOps closeout is recorded unavailable when `ao` is not exposed.

## Algorithm Naming And Pools Plan Validation — 2026-07-21

Planning validation requires:

- RC-120 and RC-130 appear in batch readiness, the authoritative dependency table, task bodies, RC-900 gates, risks, retirement and progress;
- RC-120 distinguishes active owned names from an exact line-level reference/history allowlist and requires zero baseline-to-current frozen-zone path/blob drift or untracked addition;
- the historical RC-130 review covered an only-LR policy that is now superseded
  by the AM domain-API retention policy;
- RC-210 follows RC-130 for shared `trainer.py` ownership and reuses existing data/training modules;
- Master/Sub/Spec/Execution/Validation/tracking documents agree that implementation has not started;
- `git diff --check` passes and the documentation diff does not include algorithm source/test, reference, deletion, commit, push, or cleanup actions.

Future RC-120 execution must prove active legacy contract search is empty and every remaining `facet` match is allowlisted. Pools execution follows AM-070: it must prove `src/pools` and maintained `src.pools` consumers are absent, all four domain APIs remain behavior-compatible, all Hybrid tests pass, and `src/optim/pool.py` remains present and consumed.

RC-120 must additionally prove that `(FACET|HBTXR)_(DISABLE_CUDNN|DEVICES|CKPT_PATH)` and `/tmp/(facet|hbtxr)_unet_dataset_smoke` are absent from active algorithm code/config. Characterization must cover `runtime.disable_cudnn`, `trainer.devices`, `train.ckpt_path`, evaluator config loading, and the standalone builder `--disable-cudnn` plus `disable_cudnn` metadata.

Independent T-718 result: PASS after the frozen-zone gate was strengthened to compare the approved baseline against current path/blob state and untracked additions for all reference/history zones. No blocking plan issue remains.

T-718 follow-up result for the T-719 naming revision: PASS. The reviewer confirmed that no `HBTXR_*` replacement environment control remains in the plan and that config/CLI ownership, scope, ADR, and verification are consistent.

## Algorithm Modular Ownership Plan Validation — 2026-07-21

The previous T-718 PASS is not completion evidence for the new owner policy.
Fresh validation must confirm:

- `algorithm/{analysis,archive,artifacts,requirements}` are path/blob frozen;
- `algorithm/{README.md,pyproject.toml}` remain at the root paths;
- the active target roots and current-to-target mapping appear in the plan;
- shared code uses `eveye.*` and the plan creates no bare generic package;
- frame remains without an empty source package until a real owner exists;
- event-only and shared models have explicit owners;
- `src.pools` is removed while `loss/losses.py`, `optim/optimizer.py`,
  `optim/lr_schedulers.py`, `runtime/runtime_schedulers.py`, and `optim/pool.py`
  remain with behavior-parity gates;
- Master/Sub/Spec/Execution/Validation/tracking docs all say implementation has
  not started;
- `git diff --check` and placeholder/trailing-whitespace scans pass.

T-727 result: PASS after three review rounds. The final plan orders the
DavisEyeCenter and HBTXR demo extractions before their implementation moves,
separates dependency-bearing editable tests from dependency-free wheel payload
checks, verifies direct-script help from an arbitrary CWD, enforces lower-owner
import direction, and keeps all preserved zones unchanged. No production
algorithm diff was present.
