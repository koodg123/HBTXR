# FACET Full Evaluation Gate Hardening - 2026-06-26

## Summary

The full checkpoint evaluation script is the gate that produces the final
Phase 4 / Phase 4B artifacts after both EPNet/FACET and HBTXR checkpoints
exist. The script was hardened so that it runs from the FACET codebase root and
therefore resolves `configs/*.yaml` consistently.

This does not run final evaluation yet because no full checkpoint exists.

## Updated Script

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
```

Changes:

- Export `NO_ALBUMENTATIONS_UPDATE=1`.
- Export `PYTHONPYCACHEPREFIX=/tmp/facet_full_checkpoint_eval_pycache`.
- `cd "$FACET_ROOT"` before running the Python evaluation commands.
- Exclude `*/step_checkpoints/*` from final checkpoint discovery.

The `cd "$FACET_ROOT"` line is important because
`EvEye.utils.scripts.load_config.load_config()` resolves config paths from the
current working directory by default:

```text
configs/DavisEyeEllipse_EPNet_full_unet.yaml
configs/DavisEyeEllipse_HBTXR_full_unet.yaml
```

## Validation

Syntax checks:

```text
bash -n references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
bash -n references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

Both checks passed.

Dry run with missing checkpoint:

```text
missing EPNet full checkpoint
exit code: 2
```

This is the expected fail-closed behavior while no full checkpoint exists.

## Current Gate State

Current checkpoint counts:

```text
EPNet_full_unet checkpoints: 0
HBTXR_full_unet checkpoints: 0
```

The watcher remains responsible for waiting until both model checkpoints and
the required completion markers exist:

```text
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

Current watcher policy:

```text
FACET_WATCH_REQUIRE_COMPLETED=1
```

That means the watcher will not treat an intermediate epoch checkpoint as the
final reproduction result.

## Step Checkpoint Exclusion

After adding future-run step checkpoints for long-training recovery, the final
evaluation path was hardened to avoid selecting recovery checkpoints as final
model checkpoints.

Updated selection rule:

```text
find "$run_root" -path '*/checkpoints/*.ckpt' -type f \
  ! -path '*/step_checkpoints/*' \
  ! -name 'last.ckpt'
```

The same exclusion is applied in:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

This preserves the distinction between:

```text
resume candidates: may include step checkpoints
final evaluation checkpoints: exclude step checkpoints
```

Final evaluation now chooses the best validation checkpoint when checkpoint
filenames contain `val_mean_distance`; otherwise it falls back to the newest
eligible full checkpoint:

```text
preferred: lowest val_mean_distance in checkpoint filename
fallback: newest non-last full checkpoint
excluded: last.ckpt and */step_checkpoints/*
```

Validation after this update:

```text
bash -n run_full_checkpoint_evaluation_2026-06-26.sh: passed
bash -n watch_full_checkpoints_and_evaluate_2026-06-26.sh: passed
evaluation dry run without full checkpoints: exits 2 with "missing EPNet full checkpoint"
watcher one-loop dry run: ep_ckpt_count=0 hb_ckpt_count=0, exits 3 before evaluation
synthetic checkpoint selection: selected epoch=01-val_mean_distance=3.1000.ckpt over newer worse checkpoints and step checkpoint
```

Runtime watcher refresh:

```text
2026-06-26 11:09 KST
tmux session facet_full_eval_watcher was restarted so the live watcher uses
the updated step-checkpoint exclusion logic.
```

Post-restart watcher evidence:

```text
[2026-06-26T11:09:28+0900] loop=1 ep_ckpt_count=0 hb_ckpt_count=0 ep_done=0 hb_done=0 require_completed=1
  ep_latest=missing
  hb_latest=missing
```

## Remaining Gates

The overall FACET reproduction goal remains incomplete:

- EPNet full checkpoint and 70-epoch completion marker.
- HBTXR full checkpoint and 70-epoch completion marker.
- EPNet/FACET Table II comparison artifacts.
- HBTXR-vs-EPNet comparison artifacts.
