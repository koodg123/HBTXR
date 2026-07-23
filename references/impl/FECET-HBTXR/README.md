# FECET-HBTXR

`FECET-HBTXR` is a simplified FACET + `HBTXR_v3_0` baseline for event-based eye tracking. The repository keeps FACET-style event representation and ellipse supervision, but flattens the codebase so the end-to-end path is easier to understand and run.

## Design Summary

- FACET-style event accumulation: `fixed_count`, `causal_linear`, 2-channel polarity frame
- FACET-style geometry contract: ellipse supervision and `facet_square_direct` resize policy
- `HBTXR_v3_0` core runtime: hybrid frame/event tracker, 2-stage training, host-side Search/Track FSM
- Grounded-SAM based annotation flow for FACET-style dataset generation from raw EV-Eye data
- Non-destructive data policy: raw EV-Eye and Grounded-SAM input roots are treated as read-only

## Project Structure

```text
FECET-HBTXR/
  configs/
    base.yaml
    stage1_search.yaml
    stage2_hybrid.yaml
  data/
    _internal/
      canonical/
      manifests/
    splits/
      train/
      val/
      test/
    facet_style/
  doc/
    README.md
    UPDATE_HISTORY.md
    CONVERSATION_HISTORY.md
    PROGRESS_CHECKLIST.md
    FACET_QUANT_COMPARISON.md
  fecet_hbtxr/
    dataset.py
    event_repr.py
    transforms.py
    model.py
    scheduler.py
    losses.py
    metrics.py
    trainer.py
    runtime.py
    io.py
  scripts/
    _common.sh
    prepare_dataset.sh
    prepare_facet_gsam_dataset.sh
    facet_prepare_reference_dataset.sh
    facet_train.sh
    facet_eval.sh
    facet_infer.sh
    train_stage1.sh
    train_stage2.sh
    fecet_compare_train_stage1.sh
    fecet_compare_train_stage2.sh
    fecet_compare_eval.sh
    fecet_compare_infer.sh
    eval.sh
    infer.sh
    visualize.sh
    overlay_preview.sh
  tests/
  tools/
    prepare_dataset.py
    prepare_facet_gsam_dataset.py
    prepare_facet_reference_dataset.py
    facet_train.py
    facet_eval.py
    facet_infer.py
    train.py
    eval.py
    infer.py
    visualize.py
    overlay_preview.py
  .gitignore
  requirements.txt
  README.md
```

## Environment Setup

Run from Linux or WSL:

```sh
cd /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `tools/prepare_facet_gsam_dataset.py` imports the external Grounded-SAM repository from `--grounded-sam-root`.
- When `--annotation-mode` is `csv_then_gsam` or `grounded_sam`, install the extra dependencies required by that external repository in the same environment.
- Grounded-SAM checkpoints are not bundled here. By default the script looks for:
  - `groundingdino_swint_ogc.pth`
  - `sam_vit_h_4b8939.pth`

## Linux `sh` Execution

All wrappers under `scripts/` are POSIX `sh` compatible. You can run them with `sh scripts/<name>.sh ...` without changing the input raw dataset or the Grounded-SAM repository.

### 1. Prepare manifests from an existing canonical tree

```sh
sh scripts/prepare_dataset.sh \
  --canonical-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/_internal/canonical \
  --split-scheme exgaze_with_val
```

### 2. Build a FACET-style dataset from raw EV-Eye with Grounded-SAM

Default Linux paths:

- `RAW_ROOT=/mnt/e/WSL/Shared/dataset/Eye/EV_Eye/raw_data/Data_davis`
- `GROUNDED_SAM_ROOT=/mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/annotation_tools/Grounded-Segment-Anything-main`

Basic run:

```sh
sh scripts/prepare_facet_gsam_dataset.sh \
  --annotation-mode csv_then_gsam \
  --annotation-stride 100
```

Explicit path override:

```sh
RAW_ROOT=/mnt/e/WSL/Shared/dataset/Eye/EV_Eye/raw_data/Data_davis \
GROUNDED_SAM_ROOT=/mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/annotation_tools/Grounded-Segment-Anything-main \
sh scripts/prepare_facet_gsam_dataset.sh \
  --annotation-mode grounded_sam \
  --annotation-stride 1 \
  --grounded-checkpoint /path/to/groundingdino_swint_ogc.pth \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --device cuda
```

Important behavior:

- The script only reads from the raw EV-Eye root and the Grounded-SAM root.
- Outputs are written inside this project under `data/_internal/canonical`, `data/_internal/manifests`, `data/splits`, and `data/facet_style`.
- Output paths nested inside either input root are rejected by design.

### 3. Train stage 1

```sh
sh scripts/train_stage1.sh \
  --experiment-name fecet_stage1 \
  --device cuda
```

### 4. Train stage 2

You can provide the stage-1 checkpoint either as a CLI option or as `STAGE1_CHECKPOINT`.

```sh
STAGE1_CHECKPOINT=/abs/path/to/best_search_p10.pt \
sh scripts/train_stage2.sh \
  --experiment-name fecet_stage2 \
  --device cuda
```

Equivalent explicit CLI form:

```sh
sh scripts/train_stage2.sh \
  --stage1-checkpoint /abs/path/to/best_search_p10.pt \
  --experiment-name fecet_stage2 \
  --device cuda
```

### 5. Evaluate a checkpoint

```sh
sh scripts/eval.sh \
  --checkpoint /abs/path/to/best_track_p10.pt \
  --output runs/eval/stage2_metrics.json
```

### 6. Run inference with runtime trace

```sh
sh scripts/infer.sh \
  --checkpoint /abs/path/to/best_track_p10.pt \
  --output-jsonl runs/inference/runtime_trace.jsonl \
  --output-summary runs/inference/runtime_summary.json
```

### 7. Visualize a sample

```sh
sh scripts/visualize.sh \
  --index 0 \
  --output runs/visualize/sample.png
```

### 8. Build random annotation overlay previews per session

This tool selects random annotated frames from each user/session and overlays the stored annotation on the original frame image. The output is one contact-sheet image per session.

```sh
sh scripts/overlay_preview.sh \
  --canonical-root /mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/code/FECET-HBTXR/data/_internal/canonical \
  --samples-per-session 2 \
  --seed 42 \
  --output-dir runs/overlay_preview
```

Useful filters:

```sh
sh scripts/overlay_preview.sh \
  --user 1 \
  --eye left \
  --samples-per-session 3 \
  --max-sessions 5
```

Overlay content:

- eye ROI bounding box
- pupil bounding box
- ellipse contour
- optional pupil mask overlay
- annotation metadata text such as `session_key`, `frame`, `timestamp`, `annotation_source`, and `annotation_quality`

## Python Tool Entry Points

If you do not want the shell wrappers, the corresponding Python tools are:

- `tools/prepare_dataset.py`
- `tools/prepare_facet_gsam_dataset.py`
- `tools/train.py`
- `tools/eval.py`
- `tools/infer.py`
- `tools/visualize.py`
- `tools/overlay_preview.py`

## Data and Output Contract

- Trainer input ABI:
  - `frame [B,1,256,256]`
  - `event [B,2,256,256]`
  - `prev_state [B,6]`
- Target ABI includes:
  - `mask_target`
  - `eye_target`
  - `pupil_search_target`
  - `pupil_track_target`
  - `constraint_center`
  - `annotation_quality`
  - `similarity_target`
  - `event_density`
  - `closed_eye_flag`
  - `mask_valid`
  - `valid_track`
- FACET-style export is written as `train|val|test/data`, `label`, and `ellipse` text files.

## Documentation and History

The `doc/` directory tracks repository history and accepted decisions:

- `doc/UPDATE_HISTORY.md`: implementation updates
- `doc/CONVERSATION_HISTORY.md`: concise request/decision history
- `doc/README.md`: maintenance policy for project history
- `doc/PROGRESS_CHECKLIST.md`: plan-vs-progress status
- `doc/FACET_QUANT_COMPARISON.md`: FACET vs `FECET-HBTXR` comparison workflow

## Verification

Run the test suite with:

```sh
python -m pytest
```
