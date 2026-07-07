# HGTXR-SW Collaboration Rules

- Keep software work scoped to `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/software` unless user explicitly expands scope.
- Do not delete or revert unrelated user changes.
- Record long-running training commands, checkpoints, and metrics in `docs/track`.
- Treat GPU/CUDA checks inside Codex sandbox as potentially incomplete; confirm outside sandbox when required.
- Prefer Stage2-only ablations before full pipeline reruns unless evidence says Stage1 is broken.
