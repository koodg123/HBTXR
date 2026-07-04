# FACET Quantitative Comparison Workflow

This document describes the repository-level workflow for comparing the original FACET model against `FECET-HBTXR` using Grounded-SAM based annotations.

## Goal

- Use the same Grounded-SAM derived annotation source
- Train and evaluate:
  - original FACET EPNet
  - `FECET-HBTXR`
- Export prediction artifacts for offline metric comparison

## Prerequisites

- Grounded-SAM based canonical/manifests export already prepared
- `FECET-HBTXR` Python environment installed
- FACET runtime dependencies installed when running FACET wrappers

Important:

- The FACET reference wrappers depend on the external repo at `references/FACET-main/FACET-main`.
- The FACET reference dataset bridge writes new files only under `FECET-HBTXR/data/facet_reference`.
- It does not modify the raw EV-Eye root or the FACET reference repository.

## Step 1. Build the FACET reference-compatible dataset

```sh
sh scripts/facet_prepare_reference_dataset.sh \
  --canonical-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/_internal/canonical \
  --manifests-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/_internal/manifests \
  --output-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/facet_reference \
  --overwrite
```

## Step 2. Train the original FACET model

```sh
sh scripts/facet_train.sh \
  --dataset-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/facet_reference \
  --config DavisEyeEllipse_EPNet.yaml \
  --experiment-name facet_epnet_gsam_compare \
  --output-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/runs/comparison/facet \
  --device cuda:0
```

## Step 3. Evaluate the original FACET model

```sh
sh scripts/facet_eval.sh \
  --dataset-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/facet_reference \
  --checkpoint /abs/path/to/facet_checkpoint.ckpt \
  --output /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/runs/comparison/facet/facet_eval_summary.json \
  --device cuda:0
```

## Step 4. Run original FACET inference

```sh
sh scripts/facet_infer.sh \
  --dataset-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/facet_reference \
  --checkpoint /abs/path/to/facet_checkpoint.ckpt \
  --split test \
  --output-jsonl /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/runs/comparison/facet/facet_infer_predictions.jsonl \
  --output-summary /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/runs/comparison/facet/facet_infer_summary.json \
  --device cuda:0
```

## Step 5. Train `FECET-HBTXR`

Stage 1:

```sh
sh scripts/fecet_compare_train_stage1.sh --device cuda
```

Stage 2:

```sh
STAGE1_CHECKPOINT=/abs/path/to/best_search_p10.pt \
sh scripts/fecet_compare_train_stage2.sh --device cuda
```

## Step 6. Evaluate `FECET-HBTXR`

```sh
sh scripts/fecet_compare_eval.sh \
  --checkpoint /abs/path/to/best_track_p10.pt
```

## Step 7. Run `FECET-HBTXR` inference

```sh
sh scripts/fecet_compare_infer.sh \
  --checkpoint /abs/path/to/best_track_p10.pt
```

## Comparison Artifacts

- FACET validation summary JSON
- FACET inference JSONL
- `FECET-HBTXR` validation summary JSON
- `FECET-HBTXR` runtime trace JSONL

## Interpretation Guidance

- `FACET` metrics are produced by the original FACET validation path.
- `FECET-HBTXR` metrics are produced by the simplified trainer/eval path.
- For a strict benchmark table, use the same split definition and record:
  - validation metrics
  - test-set inference exports
  - model/checkpoint identifiers
  - annotation mode and Grounded-SAM checkpoint versions

