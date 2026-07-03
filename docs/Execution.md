# Execution Log: Retina/ERVT Subject37-48 Package

Date: 2026-07-03
Branch: etri-desktop

## Summary

Retina and ERVT were evaluated on the HBTXR subject-independent test metadata, then repackaged into HBTXR-style result folders under `analysis/results`.

## Key Commands

Syntax checks:

```bash
.venv/bin/python -m py_compile analysis/scripts/generate_retina_ervt_error_distribution.py analysis/scripts/rebuild_retina_ervt_with_hbtxr_motion_labels.py analysis/scripts/create_retina_ervt_jetcas_tables.py
```

Initial full prediction generation:

```bash
.venv/bin/python analysis/scripts/generate_retina_ervt_error_distribution.py --models Retina ERVT --batch-size 256 --num-workers 8 --device cuda:0 --overwrite
```

Rebuild with HBTXR joined motion labels:

```bash
.venv/bin/python analysis/scripts/rebuild_retina_ervt_with_hbtxr_motion_labels.py --models Retina ERVT
```

Create Excel tables:

```bash
.venv/bin/python analysis/scripts/create_retina_ervt_jetcas_tables.py --models Retina ERVT --combined
```

## Implementation Notes

- `num_workers=8` DataLoader required execution outside the sandbox because the sandbox blocked multiprocessing socket/resource sharing.
- Retina prediction generation completed 1,431 batches.
- ERVT prediction generation completed 48 sequence batches.
- ERVT produces fewer prediction rows than Retina because it uses complete sequence segments.
- The final motion distribution was rebuilt from existing predictions without rerunning inference.
