# Follow-up Implementation Baseline — 2026-07-15

- Repository: `/mnt/d/dataset/EV_Eye/paper_works/HBTXR-Pool/HBTXR`
- Branch: `refactor/hbtxr-structure`
- HEAD: `97cea4ddc33b9af8c58e7d5c75345f74a7775836`
- Worktree at planning start: clean
- Remote: no upstream is configured; the local `origin/refactor/hbtxr-structure`
  tracking ref matches HEAD (`0` behind, `0` ahead) at final plan verification
- Python: `3.12.3`; default interpreter lacks pytest, NumPy, and PyTorch;
  PyYAML `6.0.1` and `uv` are available
- Hardware: `g++` available; `vitis_hls` and `vivado` unavailable
- Active HLS shell/Tcl launchers use CRLF; Bash execution fails before the local
  fallback, and the current Tcl project root is `hardware/generated/hgtxr_hls`.
  T-490 must repair both defects before any XR C-sim/synthesis task.
- Planning: Spec-Kit CLI available but uninitialized; Git LFS `3.4.1` available
  but no repository policy is approved

## Authority references

- `docs/Spec.md`
- `docs/track/ADR.md` ADR-001 through ADR-004
- `docs/analysis/HANDOVER-INTEGRATION-MATRIX.md`
- `docs/analysis/HANDOVER-RECOMMENDATIONS.md`
- the software, hardware, and license/artifact audit documents

## Non-goals

No bulk HANDOVER copy, dataset/checkpoint import, generated FPGA workspace,
training, synthesis, board access, external upload, Git push, merge, or release
is authorized by this baseline or its linked plan.
