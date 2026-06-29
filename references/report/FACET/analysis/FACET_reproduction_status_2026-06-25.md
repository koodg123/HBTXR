# FACET Reproduction Status

Overall status: `incomplete`

## Counts

- `missing`: 3
- `passed`: 8

## Items

### Gate 0 preflight

- state: `passed`
- evidence:
  - `{'python_import_check': {'returncode': 0, 'stdout': '2.12.1+cu130\nTrue', 'stderr': ''}}`
  - `{'nvidia_smi': {'returncode': 0, 'stdout': '0, NVIDIA GeForce RTX 5080, 16303 MiB, 2939 MiB\n1, NVIDIA GeForce RTX 5080, 16303 MiB, 18 MiB', 'stderr': ''}}`

### Phase 1 subset DeanDataset

- state: `passed`
- evidence:
  - `{'manifest': '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/manifest.json', 'num_samples': 8911}`

### Phase 2 U-Net labelled PNG dataset

- state: `passed`
- evidence:
  - `{'manifest': '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset/manifest.json', 'num_samples': 9011}`

### Phase 3 full DeanDataset_full_unet

- state: `missing`
- note: Required before full EPNet reproduction.
- missing:
  - `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/manifest.json`

### Report artifacts

- state: `passed`
- evidence:
  - `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_plan_2026-06-25.md`
  - `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_phase1_subset_smoke_2026-06-25.md`
  - `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_phase2_unet_dataset_prep_2026-06-25.md`
  - `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_phase3_full_expansion_prep_2026-06-25.md`
  - `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_phase4_evaluation_prep_2026-06-25.md`
  - `/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/FACET_reproduction_execution_runbook_2026-06-25.md`

### U-Net labelled subset visual samples

- state: `passed`
- evidence:
  - `{'manifest': '/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/unet_dataset_samples/manifest.json', 'num_records': 10}`

### Phase 1 EPNet smoke checkpoint

- state: `passed`
- evidence:
  - `{'checkpoint_count': 2, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/EPNet_local_train_smoke'}`

### Phase 2 U-Net smoke checkpoint

- state: `passed`
- evidence:
  - `{'checkpoint_count': 2, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/RGBUNet_local_train_smoke'}`

### Phase 2 full U-Net checkpoint

- state: `passed`
- evidence:
  - `{'checkpoint_count': 4, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/RGBUNet_local_subset'}`

### Phase 4 full EPNet checkpoint

- state: `missing`
- evidence:
  - `{'checkpoint_count': 0, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/EPNet_full_unet'}`
- missing:
  - `full EPNet training output`

### Phase 4 final evaluation artifacts

- state: `missing`
- missing:
  - `FACET_reproduction_results_<date>.json`
  - `FACET_table2_comparison_<date>.md`

## Completion Requirement

Do not mark the FACET reproduction complete until all items are `passed` and the final evaluation artifacts are produced from the full checkpoint and full validation split.
