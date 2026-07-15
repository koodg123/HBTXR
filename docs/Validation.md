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
