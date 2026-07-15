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
