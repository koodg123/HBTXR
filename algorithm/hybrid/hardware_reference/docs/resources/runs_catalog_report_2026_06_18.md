# Runs Catalog Report

Generated: `2026-06-18T10:19:32`

## Safety

- Destructive actions: `false`
- Original `runs/<run_id>` directories were preserved.
- Catalog entries are symlinks under `runs/_catalog/<category>/`.
- Existing scripts/docs can keep using original run paths.

## Summary

- Source run directories: `849`
- Catalog root: `_catalog`
- Action counts: `{"created": 2078}`

## Categories

| Category | Entries |
|---|---:|
| `runs/_catalog/00_current_authority` | 5 |
| `runs/_catalog/01_xr64_resume` | 2 |
| `runs/_catalog/02_leaders_teachers_review` | 130 |
| `runs/_catalog/03_shared_support` | 3 |
| `runs/_catalog/30_eval_runs` | 563 |
| `runs/_catalog/31_raw_stage_runs` | 253 |
| `runs/_catalog/32_raw_stage2_runs` | 244 |
| `runs/_catalog/33_raw_stage1_runs` | 9 |
| `runs/_catalog/80_smoke_cleanup_candidates` | 13 |
| `runs/_catalog/81_duplicate_review_candidates` | 10 |
| `runs/_catalog/90_archive_candidates` | 451 |
| `runs/_catalog/91_smoke_candidates` | 8 |
| `runs/_catalog/99_preserve_or_review` | 387 |

## Usage

Use original run paths for scripts and reproducibility.
Use `runs/_catalog/` for human navigation and cleanup review.

Examples:

```bash
find runs/_catalog/00_current_authority -maxdepth 1 -type l | sort
find runs/_catalog/90_archive_candidates -maxdepth 1 -type l | sort
```

## Next Gate

Do not delete archive candidates until a preserve manifest and metric/dependency report exist.
