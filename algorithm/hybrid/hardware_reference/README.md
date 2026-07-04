# HGTXR Software

Software workspace for HBTXR hybrid event-frame eye tracking experiments.

## Current Focus

The active goal is accuracy recovery for EV-Eye training:

1. Analyze reference papers in `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/01_REFERENCES/PAPER_REF`.
2. Compare against the submission-level target in `/home/kjm26/project/PRJXR/XR-VIT/PAPER_WORKS/10_submission_initial`.
3. Run controlled Stage2 ablations before larger pipeline changes.

## Baseline

- Stage1 run: `runs/raw_mode1_stage1_event_count_20260610_192838`
- Stage2 run: `runs/raw_mode1_stage2_event_count_20260610_193719`
- Stage1 checkpoint: `runs/raw_mode1_stage1_event_count_20260610_192838/train/best_search_p10.pt`
- Stage2 checkpoint: `runs/raw_mode1_stage2_event_count_20260610_193719/train/best_track_p10.pt`

## Quick Commands

Summarize existing histories:

```bash
.venv/bin/python scripts/external/summarize_training_history.py \
  runs/raw_mode1_stage1_event_count_20260610_192838 \
  runs/raw_mode1_stage2_event_count_20260610_193719
```

Preview planned Stage2 ablations:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --dry-run
```

Run first accuracy ablation:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --case lr2e-4_nodistill_fullwidth --device cuda:0
```

## Tracking Docs

- Master plan: `docs/Master-Plan.md`
- Sub-agent plan: `docs/Sub-Plan.md`
- Experiment spec: `docs/Spec.md`
- Execution notes: `docs/Execution.md`
- Validation plan: `docs/Validation.md`
- Progress checklist: `docs/track/PROGRESS.md`
