# Large Local-Only Result Files

Date: 2026-07-03

These files are intentionally excluded from git tracking because they exceed GitHub's file size limits or are large generated per-sample tables. They remain reproducible from the committed scripts, checkpoints/configs, and compact summary files.

## Excluded Files

| Size | Path | Reason |
|---:|---|---|
| 421.22 MB | `analysis/results/Retina/logs/retina_train_20260630_190348.log` | Training log, too large for GitHub |
| 156.74 MB | `analysis/results/HBTXR/HBTXR_subject37_48_test_predictions_with_metadata.csv` | Per-sample table over 100 MB |
| 149.77 MB | `analysis/results/HBTXR/HBTXR_subject37_48_test_joined_motion_error.csv` | Per-sample joined table over 100 MB |
| 109.45 MB | `analysis/results/Retina/Retina_subject37_48_test_predictions_with_metadata.csv` | Per-sample table over 100 MB |
| 108.71 MB | `analysis/results/ERVT/ERVT_subject37_48_test_predictions_with_metadata.csv` | Per-sample table over 100 MB |
| 103.21 MB | `analysis/results/Retina/Retina_subject37_48_test_joined_motion_error.csv` | Per-sample joined table over 100 MB |
| 102.82 MB | `analysis/results/ERVT/ERVT_subject37_48_test_joined_motion_error.csv` | Per-sample joined table over 100 MB |
| 91.06 MB | `analysis/results/HBTXR/HBTXR_subject_independent_img64_patch4_test_sample_predictions.csv` | Large per-sample prediction table |
| 68.88 MB | `analysis/results/HBTXR/HBTXR_subject_independent_img64_patch4_test_sample_metadata.csv` | Large per-sample metadata table |
| 50.14 MB | `analysis/results/HBTXR/checkpoints/epoch=66-val_mean_distance=0.5401.ckpt` | Checkpoint above GitHub recommended file size |
| 43.14 MB | `analysis/results/Retina/Retina_subject37_48_test_sample_predictions.csv` | Large per-sample prediction table |
| 42.62 MB | `analysis/results/ERVT/ERVT_subject37_48_test_sample_predictions.csv` | Large per-sample prediction table |

## Tracked Alternatives

The commit keeps compact files needed for reporting:

- `analysis/results/*/*_error_distribution_by_subject_motion.csv`
- `analysis/results/*/*_joined_motion_counts.csv`
- `analysis/results/*/STATUS.md`
- `analysis/results/*/JETCAS_REPLY_TABLES*.xlsx`
- `analysis/results/available_metrics.csv`
- `analysis/scripts/*.py`
- `docs/*.md`

Use `analysis/scripts/generate_retina_ervt_error_distribution.py` and `analysis/scripts/rebuild_retina_ervt_with_hbtxr_motion_labels.py` to regenerate the excluded per-sample tables locally.
