# FACET Full Training Resume Support - 2026-06-26

## Summary

The full EPNet/FACET and HBTXR runs are long-running jobs. To reduce the cost
of future crashes or manual restarts, the training entrypoint and full-training
launchers now support checkpoint resume.

This does not interrupt the currently running EPNet/HBTXR processes. The change
applies to future launches or recovery restarts after a checkpoint exists.

## Code Changes

Training entrypoint:

```text
references/codebase/software/FACET/tools/train.py
```

Added:

- Optional `trainer.precision` passthrough from YAML.
- `FACET_CKPT_PATH` environment variable support.
- `config["train"]["ckpt_path"]` fallback support.
- `trainer.fit(..., ckpt_path=ckpt_path)`.

Full launchers:

```text
references/report/FACET/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

Added:

- `RUN_ROOT` per model.
- `latest_checkpoint()` helper.
- Default `FACET_RESUME_LATEST=1` behavior.
- `FACET_CKPT_PATH=<latest checkpoint>` export when a checkpoint exists.
- Log line showing whether a resume checkpoint was found.

## Runtime Behavior

Default behavior:

```text
FACET_RESUME_LATEST=1
```

If any checkpoint exists under the model run root, the launcher exports:

```text
FACET_CKPT_PATH=/path/to/latest.ckpt
```

Then `tools/train.py` passes that path to Lightning:

```text
trainer.fit(..., ckpt_path=ckpt_path)
```

To force a fresh restart:

```bash
FACET_RESUME_LATEST=0 bash references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

## Validation

Syntax checks:

```text
python3 -m py_compile references/codebase/software/FACET/tools/train.py
bash -n references/report/FACET/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
bash -n references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

All checks passed.

## Current Live Training State

At the time of this update:

```text
tmux: facet_epnet_full_gpu0
tmux: facet_hbtxr_full_gpu1
tmux: facet_full_eval_watcher
```

GPU compute apps:

```text
GPU0: EPNet/FACET, PID 1428589, 4214 MiB
GPU1: HBTXR, PID 1483023, 11334 MiB
```

No checkpoint exists yet, so the resume path has not been exercised in a live
restart.

## Remaining Gates

The overall FACET reproduction goal remains incomplete:

- EPNet full checkpoint and 70-epoch completion marker.
- HBTXR full checkpoint and 70-epoch completion marker.
- EPNet/FACET Table II comparison artifacts.
- HBTXR-vs-EPNet comparison artifacts.
