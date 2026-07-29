# Existing ViT-Accel Analysis Integration

Source directory: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware/analysis/vit-accel`

## Included Source Reports

| Group | Count | Source pattern |
|---|---:|---|
| Codebases | 30 | `analysis/vit-accel/codebases/*/analysis.md` |
| Papers | 18 | `analysis/vit-accel/papers/*/analysis.md` |
| Experiment extensions | 1 | `analysis/vit-accel/experiment_extensions_2026_06_15.md` |
| Third-goal integration | 1 | `analysis/vit-accel/third_goal_integration_2026_06_15.md` |

## Integration Decision

The existing reports are already detailed and should remain the source of truth for per-codebase and per-paper evidence. This package adds two integrated roll-up documents:

- `codebases/INTEGRATED_CODEBASE_ANALYSIS.md`
- `papers/INTEGRATED_PAPER_ANALYSIS.md`

The roll-ups are designed for current HGTXR hardware planning: DSP mapping, URAM/LUTRAM memory policy, parallelism factor selection, and experiment prioritization.

