# Code/File Integration Evidence

## Initial evidence

- Repository root: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR`.
- Branch at start: `refactor/hbtxr-structure`.
- Planning baseline HEAD: `97cea4ddc33b9af8c58e7d5c75345f74a7775836`.
- `git diff --check`: pass before execution-scope update.
- Available modules: PyYAML 6.0.1.
- Unavailable modules: NumPy, PyTorch, pytest.
- No install, experiment, synthesis, board, Spec-Kit, push, or remote mutation.

Per-task command output, reviews, commit SHAs, and uncovered scope will be added
after each selected slice.

## T-010 evidence

- Commits: `c24030f` and follow-up hardening `1764caf`.
- `python3 -m unittest tests/provenance/test_source_registry.py`: 19 tests pass.
- Live `--require-local` registry validation: 11 HANDOVER candidates pass.
- Spec and quality review: pass after credential-key and declared-revision blob fixes.

## T-020 evidence

- Commit: `28addce`.
- JSON schema parse and example manifest validator: pass.
- `python3 -m unittest discover -s tests/provenance -p 'test_artifact_manifest.py'`:
  11 tests pass.
- `git diff --cached --check`: pass before commit.
- Prohibited staged payload scan: no matches (`rg` exit 1).
- Spec and quality review: pass.
- No `.gitignore`, LFS, upload, move, deletion, install, reproduction, experiment,
  or push action occurred.

## T-100 evidence

- Commit: `7b3068e`.
- Dependency-light data contracts and HANDOVER tensor-shape bindings: 14 tests pass.
- Python compile and CR-at-EOL-aware diff check: pass.
- Existing runtime payloads are not transformed or reordered by the contract layer.
- Specification and quality reviews: pass after manifest digest type hardening.

## T-110 evidence

- Commit: `a3530aa`.
- Evaluation domain/task/policy bridge: 9 tests pass; Python compile and diff check pass.
- ROI, sensor, and post-transform coordinates use explicit per-axis scaling.
- Runtime type, overflow, bool/float tolerance, and policy errors fail as contract errors.
- Specification and quality reviews: pass.

## T-210 evidence

- Commit: `fe4df26`.
- XR schema/normalizer: 14 tests pass; JSON parse, Python compile, and diff check pass.
- Eight experiments resolve exactly 18 unique reference JSON files.
- Existing design/experiment/model/target JSON files have no diff.
- Source repository/revision, ZCU104 tuple, clocks, AXI width, runtime MMIO, links, and hashes are explicit.
- Specification and quality reviews: pass after provenance, schema-policy binding, and numeric-error hardening.

## Consolidated verification evidence

- All 67 selected unit tests passed in one fresh run.
- The source registry validated 11 local candidates and the artifact example validated.
- Python compilation, JSON parsing, repository boundary, CR-at-EOL-aware diff integrity, and Git fsck passed.
- The range diff uses `core.whitespace=cr-at-eol` because `data/contracts.py` preserves source CRLF.
- The existing 18 XR reference JSON files remained unchanged; excluded work was not run.
