# FACET Reproduction Status

Overall status: `incomplete`

## Counts

- `missing`: 8
- `passed`: 10

## Items

### Gate 0 preflight

- state: `passed`
- evidence:
  - `{'python_import_check': {'returncode': 0, 'stdout': '2.12.1+cu130\nTrue', 'stderr': ''}}`
  - `{'nvidia_smi': {'returncode': 0, 'stdout': '0, NVIDIA GeForce RTX 5080, 16303 MiB, 4240 MiB\n1, NVIDIA GeForce RTX 5080, 16303 MiB, 11358 MiB', 'stderr': ''}}`

### Phase 1 subset DeanDataset

- state: `passed`
- evidence:
  - `{'manifest': '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset/manifest.json', 'num_samples': 8911}`

### Phase 2 U-Net labelled PNG dataset

- state: `passed`
- evidence:
  - `{'manifest': '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DavisWithMaskDataset_labelled_subset/manifest.json', 'num_samples': 9011}`

### Phase 3 full DeanDataset_full_unet

- state: `passed`
- evidence:
  - `{'manifest': '/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet/manifest.json', 'valid_ellipse_count': 1457820}`

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

- state: `passed`
- evidence:
  - `{'checkpoint_count': 16, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/EPNet_full_unet'}`

### Phase 4 full EPNet training completion

- state: `missing`
- note: A checkpoint alone is not sufficient; this gate prevents intermediate epoch checkpoints from being treated as full reproduction completion.
- evidence:
  - `{'log': '/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/EPNet_full_unet_gpu0_train_2026-06-26.log', 'completion_marker_found': False}`
- missing:
  - `EPNet max_epochs=70 completion log`

### Phase 4B full HBTXR checkpoint

- state: `missing`
- evidence:
  - `{'checkpoint_count': 0, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/HBTXR_full_unet'}`
- missing:
  - `full HBTXR training output`

### Phase 4B full HBTXR training completion

- state: `missing`
- note: A checkpoint alone is not sufficient; this gate prevents intermediate epoch checkpoints from being treated as full reproduction completion.
- evidence:
  - `{'log': '/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/HBTXR_full_unet_gpu1_train_2026-06-26.log', 'completion_marker_found': False}`
- missing:
  - `HBTXR max_epochs=70 completion log`

### Phase 4 EPNet fpn_dw ablation checkpoint

- state: `missing`
- note: The reproduction plan calls for fpn_2d baseline plus fpn_dw ablation for paper correspondence.
- evidence:
  - `{'checkpoint_count': 0, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/EPNet_fpn_dw_full_unet'}`
- missing:
  - `EPNet fpn_dw ablation training output`

### Phase 4 EPNet fpn_dw ablation completion

- state: `missing`
- note: A checkpoint alone is not sufficient for the planned fpn_dw ablation.
- evidence:
  - `{'log': '/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/EPNet_fpn_dw_full_unet_gpu0_train_2026-06-26.log', 'completion_marker_found': False}`
- missing:
  - `EPNet fpn_dw max_epochs=70 completion log`

### Phase 4B HBTXR effective-batch-32 checkpoint

- state: `missing`
- note: This stricter comparison run matches EPNet's effective batch size of 32.
- evidence:
  - `{'checkpoint_count': 0, 'root': '/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/HBTXR_full_unet_effbs32'}`
- missing:
  - `HBTXR effective-batch-32 training output`

### Phase 4B HBTXR effective-batch-32 completion

- state: `missing`
- note: A checkpoint alone is not sufficient for the planned fair effective-batch comparison.
- evidence:
  - `{'log': '/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/HBTXR_full_unet_effbs32_gpu1_train_2026-06-26.log', 'completion_marker_found': False}`
- missing:
  - `HBTXR effective-batch-32 max_epochs=70 completion log`

### Phase 4 final evaluation artifacts

- state: `missing`
- note: Final reproduction requires both EPNet/FACET paper comparison and HBTXR-vs-EPNet comparison artifacts.
- missing:
  - `FACET_reproduction_results_*.json`
  - `FACET_reproduction_results_*.md`
  - `FACET_reproduction_summary_*.json`
  - `FACET_table2_comparison_*.md`
  - `FACET_hbtxr_reproduction_results_*.json`
  - `FACET_hbtxr_reproduction_results_*.md`
  - `FACET_epnet_vs_hbtxr_comparison_*.json`
  - `FACET_epnet_vs_hbtxr_comparison_*.md`
  - `FACET_epnet_fpn_dw_reproduction_results_*.json`
  - `FACET_epnet_fpn_dw_table2_comparison_*.md`
  - `FACET_hbtxr_effbs32_reproduction_results_*.json`
  - `FACET_hbtxr_effbs32_reproduction_results_*.md`
  - `FACET_epnet_vs_hbtxr_effbs32_comparison_*.json`
  - `FACET_epnet_vs_hbtxr_effbs32_comparison_*.md`

## Completion Requirement

Do not mark the FACET reproduction complete until all items are `passed` and the final evaluation artifacts are produced from the full checkpoint and full validation split.
