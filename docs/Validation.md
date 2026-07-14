# Validation Plan

## Checkpoint Gates

| Gate | Command or evidence | Pass condition |
|---|---|---|
| Repo boundary | `git rev-parse --show-toplevel` | ends in `/HBTXR` |
| Staged scope | `git diff --cached --name-status` | only slice allowlist |
| Diff integrity | `git diff --cached --check` | exit 0 |
| JSON | parse imported JSON files | 0 parse errors |
| Python | compile/import targeted Python files | 0 unexpected errors |
| Shell | `bash -n` on imported shell files | 0 errors |
| Artifact | size/type/secret/nested-repo scan | prohibited files 0 |
| Commit | `git log -1 --oneline` | expected message and new SHA |

Current pre-plan evidence: 105 JSON, 81 Python, and 26 shell files passed; no
large file over 10 MiB, nested Git repo, submodule, symlink, or credential-like
filename was found in the current checkpoint scope.

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

## Remaining Risk

Full training, FPGA tool execution, board measurements, and external artifact
availability cannot be verified in the analysis phase. They must remain
explicitly `UNAVAILABLE` or `PENDING`, not inferred as passing.
