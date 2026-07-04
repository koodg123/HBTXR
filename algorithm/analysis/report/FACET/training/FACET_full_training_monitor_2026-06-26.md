# FACET Full Training Monitor

Date: 2026-06-26

## Summary

The active FACET reproduction goal is still in Phase 4 / Phase 4B. The full
`DeanDataset_full_unet` dataset is complete, EPNet/FACET and HBTXR-DeiT full
training jobs are running in parallel, and no full checkpoint has been produced
yet.

This is not a completion state. The remaining gates are full EPNet checkpoint,
full HBTXR checkpoint, and final evaluation/comparison artifacts.

## Current Runtime Snapshot

Snapshot time:

```text
2026-06-26 08:04:07 +0900
```

Active tmux sessions:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
```

Host GPU compute apps:

```text
00000000:02:00.0, 125368, /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
00000000:03:00.0, 125360, /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 5920 MiB
```

GPU utilization sampled during monitoring:

```text
GPU0: 94%
GPU1: 94%
```

This indicates both training processes are actively computing, not merely
holding GPU memory.

## Log Progress Evidence

The latest version-2 event and training logs were still growing at the snapshot:

```text
2026-06-26 08:04:06.977165017 +0900 794220 references/codebase/software/FACET/runs/logs/EPNet_full_unet/version_2/events.out.tfevents.1782428243.etrib.125368.0
2026-06-26 08:04:07.015165819 +0900 724156 references/codebase/software/FACET/runs/logs/HBTXR_full_unet/version_2/events.out.tfevents.1782428244.etrib.125360.0
2026-06-26 08:04:07.025166030 +0900 763033 references/report/FACET/EPNet_full_unet_gpu0_train_2026-06-26.log
2026-06-26 08:04:07.025166030 +0900 1152882 references/report/FACET/HBTXR_full_unet_gpu1_train_2026-06-26.log
```

Both jobs had already passed:

- dataset manifest gate
- train/val `DavisEyeEllipseDataset` smoke check
- Lightning sanity validation
- entry into epoch 0 training

Current tail progress:

```text
EPNet_full_unet version_2: epoch 0, about 8184 / 36415 steps
HBTXR_full_unet version_2: epoch 0, about 7548 / 582630 steps
```

The structured progress snapshot is stored at:

```text
references/report/FACET/FACET_full_training_progress_snapshot_2026-06-26.json
references/report/FACET/FACET_full_training_progress_snapshot_2026-06-26.md
```

The logs contain older tracebacks from failed pre-fix attempts in the same log
files. The current `version_2` tail shows continuing progress after the loader
and augmentation fixes.

## Reproduction Status Checker

The host-access status checker was rerun after monitoring. The checker now
separates checkpoint existence from full training completion, so an intermediate
epoch checkpoint cannot satisfy the final reproduction gate by itself:

```text
overall_status: incomplete
passed: 9
missing: 5
```
Remaining missing items:

```text
Phase 4 full EPNet checkpoint
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```

Output artifacts:

```text
references/report/FACET/FACET_reproduction_status_2026-06-26.json
references/report/FACET/FACET_reproduction_status_2026-06-26.md
```

## Important Note

Running the status checker inside the restricted sandbox can produce a false
negative GPU preflight because Python/NVML access may be blocked there. The
current status files were refreshed with host access and show the actual RTX
5080 jobs.

## Next Gate

Wait for checkpoint creation under:

```text
references/codebase/software/FACET/runs/logs/EPNet_full_unet/version_*/checkpoints/*.ckpt
references/codebase/software/FACET/runs/logs/HBTXR_full_unet/version_*/checkpoints/*.ckpt
```

Evaluation preparation added:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

`run_full_checkpoint_evaluation_2026-06-26.sh` evaluates the latest available
full EPNet and HBTXR checkpoints on GPU0 and GPU1 respectively. It currently
cannot run successfully because both full checkpoint gates are still missing.

`watch_full_checkpoints_and_evaluate_2026-06-26.sh` checks the checkpoint gates
periodically and can run the evaluation script automatically. By default it
requires both full training logs to show completion (`FACET_WATCH_REQUIRE_COMPLETED=1`)
before launching evaluation, so the first intermediate epoch checkpoint is not
mistaken for the final reproduction result. A one-loop dry run passed and
correctly exited with code 3 while both checkpoint counts were zero.

The watcher is now running in tmux:

```text
tmux: facet_full_eval_watcher
PID: 187215
first active loop: 2026-06-26T08:06:12+0900
loop state: ep_ckpt_count=0 hb_ckpt_count=0 ep_done=0 hb_done=0 require_completed=1
log: references/report/FACET/FACET_full_checkpoint_watch_2026-06-26.log
```

The watcher script has also been updated to refresh
`FACET_full_training_progress_snapshot_2026-06-26.{json,md}` on each loop.

After checkpoints exist, run full validation/evaluation and produce:

```text
references/report/FACET/FACET_reproduction_results_<date>.md
references/report/FACET/FACET_table2_comparison_<date>.md
references/report/FACET/FACET_hbtxr_reproduction_results_<date>.md
references/report/FACET/FACET_epnet_vs_hbtxr_comparison_<date>.md
```

## Update: 2026-06-26 10:21 KST

Live sessions remain active:

```text
tmux: facet_epnet_full_gpu0
tmux: facet_hbtxr_full_gpu1
tmux: facet_full_eval_watcher
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 90% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 5943 MiB used
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 15041 / 36415 | 41.30% | 11.44 it/s | 31:08 | 0 |
| HBTXR_full_unet | 0 | 91870 / 582630 | 15.77% | 10.61 it/s | 12:50:59 | 0 |

Recent log tails for both EPNet and HBTXR were checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
```

No matches were found in the current tails.

HBTXR bottleneck note:

- `DavisEyeEllipse_HBTXR_full_unet.yaml` currently uses `batch_size: 2`, which
  gives `582630` train steps per epoch.
- The HBTXR `patch_size: 4` is tied to the current 64x64 output heatmap
  resolution (`256 / 4 = 64`). Increasing patch size directly would change the
  output resolution and can break the loss/target shape contract.
- The safe optimization candidate is a larger HBTXR batch size, not a direct
  patch-size change.

Prepared but not run because both GPUs are actively training:

```text
references/codebase/software/FACET/EvEye/utils/scripts/probe_hbtxr_batch_size.py
```

## Update: 2026-06-26 22:49 KST

Monitoring cadence policy:

```text
routine training-result refresh interval: 3600 seconds
latest status/progress/audit refresh: 2026-06-26 22:33 KST
next one-shot refresh reservation: facet_next_hourly_refresh_once
```

The 22:49 KST check did not force a new log scan because the latest
status/progress/audit artifacts were only about 16 minutes old. The active
one-shot refresh reservation is waiting for the remaining interval before
calling `run_hourly_status_refresh_guard_2026-06-26.sh`.

Current tmux sessions relevant to the reproduction goal:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_training_watchdog
facet_full_eval_watcher
facet_epnet_fpn_dw_gpu0_waiter
facet_epnet_fpn_dw_eval_watcher
facet_hbtxr_effbs32_gpu1_waiter
facet_hbtxr_effbs32_eval_watcher
facet_followup_training_watchdog
facet_next_hourly_refresh_once
```

Current training processes detected without reading log tails:

```text
tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
```

Latest structured progress snapshot, generated at 2026-06-26 22:33 KST:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 10 | 9908 / 36415 | 27.21% | 11.48 it/s | 38:29 | 8 |
| HBTXR_full_unet | 0 | 232098 / 291315 | 79.67% | 5.49 it/s | 2:59:55 | 0 |

Current completion audit, generated from the same 22:33 KST status refresh:

```text
overall_status: incomplete
passed: 10
missing: 8
can_mark_goal_complete: false
```

Remaining gates:

```text
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 EPNet fpn_dw ablation checkpoint
Phase 4 EPNet fpn_dw ablation completion
Phase 4B HBTXR effective-batch-32 checkpoint
Phase 4B HBTXR effective-batch-32 completion
Phase 4 final evaluation artifacts
```

Conclusion: the reproduction plan is still active and incomplete. The next
meaningful transition should come from the hourly guard refreshing status after
the one-hour interval, a full HBTXR checkpoint appearing, or the full EPNet/HBTXR
completion markers triggering the evaluation watchers.

## Update: 2026-06-26 10:38 KST

EPNet `version_3` stopped during epoch 0 because one transformed ellipse label
contained `NaN` values:

```text
ValueError: cannot convert float NaN to integer
DavisEyeEllipseDataset.py: radius = gaussian_radius((math.ceil(b), math.ceil(a)))
```

`DavisEyeEllipseDataset` was patched so non-finite or non-positive-axis
ellipses are treated as closed/invalid samples. Target tensors remain
zero-filled and gaussian/mask drawing is skipped for those samples.

Validation passed:

```text
py_compile: DavisEyeEllipseDataset.py
bash -n: run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
dataset smoke ok: 28 checked train indices, all finite
```

HBTXR was also restarted so both live jobs load the patched dataset code.

Current tmux sessions:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 86% utilization, 4237 MiB used, PID 1428589
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 5943 MiB used, PID 1428417
```

Latest restart-sliced logs contain no matches for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 396 / 36415 | 1.09% | 10.54 it/s | 56:58 | 0 |
| HBTXR_full_unet | 0 | 417 / 582630 | 0.07% | 9.85 it/s | 16:25:09 | 0 |

Recovery report:

```text
references/report/FACET/FACET_epnet_invalid_ellipse_recovery_2026-06-26.md
```

## Update: 2026-06-26 10:44 KST

HBTXR was briefly stopped to run a GPU1 batch-size probe while EPNet/FACET kept
training on GPU0.

Probe summary:

| Batch size | Result | Samples/s | Peak allocated MiB |
|---:|---|---:|---:|
| 2 | ok | 7.08 | 5063 |
| 4 | ok | 22.28 | 9999 |
| 5 | ok | 12.16 | 12473 |
| 6 | OOM | n/a | 13745 |
| 8 | OOM | n/a | 13322 |
| 12 | OOM | n/a | 14180 |
| 16 | OOM | n/a | 11189 |

Selected HBTXR batch size:

```text
batch_size: 4
```

Modified config:

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet.yaml
```

HBTXR restarted successfully:

```text
tmux: facet_hbtxr_full_gpu1
PID: 1460336
GPU1 memory: 11334 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 4620 / 36415 | 12.69% | 11.41 it/s | 46:25 | 0 |
| HBTXR_full_unet | 0 | 273 / 291315 | 0.09% | 5.24 it/s | 15:25:40 | 0 |

The HBTXR step count confirms that the batch-size change is active. The
current wall-clock ETA is still long because batch throughput dropped roughly
in proportion to the larger batch size.

Batch probe report:

```text
references/report/FACET/FACET_hbtxr_batch_probe_2026-06-26.md
```

## Update: 2026-06-26 10:49 KST

Additional HBTXR mixed precision probes were run on GPU1:

| Precision | Batch size | Result | Samples/s | Peak reserved MiB |
|---|---:|---|---:|---:|
| bf16-mixed | 4 | ok | 10.29 | 13288 |
| bf16-mixed | 8 | OOM | n/a | 14670 |
| 16-mixed | 4 | ok | 10.49 | 12528 |
| 16-mixed | 8 | OOM | n/a | 15064 |

The mixed precision probes were slower than fp32 `batch_size: 4` and did not
allow `batch_size: 8`, so HBTXR was restarted with the selected fp32
`batch_size: 4` configuration.

Current GPU evidence:

```text
GPU0: EPNet, PID 1428589, 4214 MiB
GPU1: HBTXR, PID 1483023, 11334 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 7619 / 36415 | 20.92% | 11.45 it/s | 41:55 | 0 |
| HBTXR_full_unet | 0 | 196 / 291315 | 0.07% | 5.15 it/s | 15:42:16 | 0 |

Latest restart-sliced logs contain no crash signatures.

## Update: 2026-06-26 10:51 KST

Checkpoint resume support was added for future crash/manual restart recovery.
The current live training processes were not interrupted.

Updated files:

```text
references/codebase/software/FACET/tools/train.py
references/report/FACET/run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

Behavior:

```text
FACET_RESUME_LATEST=1 by default in the launchers
FACET_CKPT_PATH is exported when a latest checkpoint exists
tools/train.py passes ckpt_path to Lightning Trainer.fit
```

Validation passed:

```text
py_compile: tools/train.py
bash -n: run_epnet_full_unet_gpu0_when_ready_2026-06-26.sh
bash -n: run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

No checkpoint exists yet, so resume support is prepared but not yet live-tested.

Resume support report:

```text
references/report/FACET/FACET_full_training_resume_support_2026-06-26.md
```

## Update: 2026-06-26 10:54 KST

The full checkpoint evaluation gate was hardened. The evaluation script now
changes into the FACET codebase root before loading config files, so final
evaluation will resolve:

```text
configs/DavisEyeEllipse_EPNet_full_unet.yaml
configs/DavisEyeEllipse_HBTXR_full_unet.yaml
```

Updated script:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
```

Validation:

```text
bash -n: run_full_checkpoint_evaluation_2026-06-26.sh
bash -n: watch_full_checkpoints_and_evaluate_2026-06-26.sh
dry run without checkpoints: exits 2 with "missing EPNet full checkpoint"
```

Evaluation hardening report:

```text
references/report/FACET/FACET_full_evaluation_gate_hardening_2026-06-26.md
```

## Update: 2026-06-26 10:56 KST

A long-running training watchdog was added and started:

```text
tmux: facet_full_training_watchdog
script: references/report/FACET/watch_full_training_jobs_2026-06-26.sh
log: references/report/FACET/FACET_full_training_watchdog_2026-06-26.log
```

The watchdog checks:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
```

If a training session is missing and its log does not show completion, it is
restarted with the corresponding resume-capable launcher. The first live loop
reported all three watched sessions alive and did not restart any training.

Current progress snapshot from the first live watchdog loop:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 12622 / 36415 | 34.66% | 11.48 it/s | 34:32 | 0 |
| HBTXR_full_unet | 0 | 2577 / 291315 | 0.88% | 5.46 it/s | 14:41:50 | 0 |

Watchdog report:

```text
references/report/FACET/FACET_full_training_watchdog_2026-06-26.md
```

## Update: 2026-06-26 10:58 KST

The full training jobs and watchdog remain active.

Active tmux sessions:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
facet_full_training_watchdog
```

Current GPU compute apps:

```text
GPU0: EPNet/FACET, PID 1428589, 4214 MiB
GPU1: HBTXR, PID 1483023, 11334 MiB
```

Latest restart-sliced log checks for EPNet and HBTXR found no matches for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 13941 / 36415 | 38.28% | 11.48 it/s | 32:37 | 0 |
| HBTXR_full_unet | 0 | 3205 / 291315 | 1.10% | 5.46 it/s | 14:39:06 | 0 |

The status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

## Latest: 2026-06-26 12:03 KST Progress Refresh

Both full-training jobs remain alive and continue to use the intended GPUs.

Active sessions:

```text
facet_epnet_full_gpu0: alive
facet_hbtxr_full_gpu1: alive
facet_full_eval_watcher: alive
facet_full_training_watchdog: alive
```

Active training processes:

```text
PID 1428589: tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
PID 1483023: tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
PID 1573595: watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 19478 / 36415 | 53.49% | 11.52 it/s | 24:30 | 2 |
| HBTXR_full_unet | 0 | 24660 / 291315 | 8.47% | 5.48 it/s | 13:30:54 | 0 |

Current-run error scan:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

Final evaluation gate:

```text
bash references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh --dry-run
exit code: 2
output: missing HBTXR full checkpoint
```

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

Still missing:

```text
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```

## 2026-06-26 12:02 KST Progress Refresh

Both long-running training sessions are alive and still using the assigned GPUs.

Active sessions:

```text
facet_epnet_full_gpu0: alive
facet_hbtxr_full_gpu1: alive
facet_full_eval_watcher: alive
facet_full_training_watchdog: alive
```

Active training processes:

```text
PID 1428589: tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
PID 1483023: tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
PID 1573595: watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 11357 MiB used / 16303 MiB
```

Current compute apps:

```text
PID 1428589: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
PID 1483023: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 11334 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 18644 / 36415 | 51.20% | 11.52 it/s | 25:42 | 2 |
| HBTXR_full_unet | 0 | 24262 / 291315 | 8.33% | 5.48 it/s | 13:32:06 | 0 |

Checkpoint evidence:

```text
EPNet_full_unet/version_4/checkpoints/epoch=00-val_mean_distance=1.8744.ckpt
EPNet_full_unet/version_4/checkpoints/last.ckpt
HBTXR_full_unet: no checkpoint yet
```

The current-run log slices after the latest training markers were checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

No matches were found after the latest run markers.

The status checker now reports:

```text
overall_status: incomplete
passed: 10
missing: 4
```

Still missing:

```text
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```

## 2026-06-26 12:03 KST Progress Refresh

Both full-training jobs remain alive and continue to use the intended GPUs.

Active sessions:

```text
facet_epnet_full_gpu0: alive
facet_hbtxr_full_gpu1: alive
facet_full_eval_watcher: alive
facet_full_training_watchdog: alive
```

Active training processes:

```text
PID 1428589: tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
PID 1483023: tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
PID 1573595: watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current compute apps:

```text
PID 1428589: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
PID 1483023: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 11334 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 19478 / 36415 | 53.49% | 11.52 it/s | 24:30 | 2 |
| HBTXR_full_unet | 0 | 24660 / 291315 | 8.47% | 5.48 it/s | 13:30:54 | 0 |

Checkpoint evidence:

```text
EPNet_full_unet/version_4/checkpoints/epoch=00-val_mean_distance=1.8744.ckpt
EPNet_full_unet/version_4/checkpoints/last.ckpt
HBTXR_full_unet: no checkpoint yet
```

The current-run log slices after the latest training markers were checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

No matches were found after the latest run markers.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

Still missing:

```text
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```

## 2026-06-26 11:40 KST Progress Refresh

Both full-training jobs are still alive and actively using their assigned GPUs.

Active sessions:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
facet_full_training_watchdog
```

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 92% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 3628 / 36415 | 9.96% | 11.53 it/s | 47:24 | 2 |
| HBTXR_full_unet | 0 | 17111 / 291315 | 5.87% | 5.48 it/s | 13:54:06 | 0 |

Full validation checkpoints:

```text
EPNet_full_unet/version_4/checkpoints/epoch=00-val_mean_distance=1.8744.ckpt
EPNet_full_unet/version_4/checkpoints/last.ckpt
HBTXR_full_unet: none yet
```

Current-run error scan after the latest run markers checked for `Traceback`,
`AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM, kill signals,
interrupts, `ValueError`, and generic `Exception`. No matches were found for
either EPNet or HBTXR.

The status checker now reports:

```text
overall_status: incomplete
passed: 10
missing: 4
```

Remaining gates:

```text
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```

## 2026-06-26 11:42 KST Progress Refresh

Both full-training jobs remain active on their assigned GPUs. No new full
validation checkpoint has appeared since the 11:40 KST refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 4763 / 36415 | 13.08% | 11.53 it/s | 45:45 | 2 |
| HBTXR_full_unet | 0 | 17651 / 291315 | 6.06% | 5.48 it/s | 13:52:27 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:59 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 86% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 16114 / 36415 | 44.25% | 11.52 it/s | 29:22 | 2 |
| HBTXR_full_unet | 0 | 23056 / 291315 | 7.91% | 5.48 it/s | 13:35:49 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:57 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 15081 / 36415 | 41.41% | 11.52 it/s | 30:51 | 2 |
| HBTXR_full_unet | 0 | 22564 / 291315 | 7.75% | 5.48 it/s | 13:37:21 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:56 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 83% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 14048 / 36415 | 38.58% | 11.52 it/s | 32:21 | 2 |
| HBTXR_full_unet | 0 | 22072 / 291315 | 7.58% | 5.48 it/s | 13:38:52 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:54 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 13072 / 36415 | 35.90% | 11.52 it/s | 33:46 | 2 |
| HBTXR_full_unet | 0 | 21608 / 291315 | 7.42% | 5.48 it/s | 13:40:17 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:53 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 93% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 12103 / 36415 | 33.24% | 11.52 it/s | 35:10 | 2 |
| HBTXR_full_unet | 0 | 21147 / 291315 | 7.26% | 5.48 it/s | 13:41:42 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:51 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 88% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 11132 / 36415 | 30.57% | 11.52 it/s | 36:34 | 2 |
| HBTXR_full_unet | 0 | 20684 / 291315 | 7.10% | 5.48 it/s | 13:43:08 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:50 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 83% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 92% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 10251 / 36415 | 28.15% | 11.52 it/s | 37:51 | 2 |
| HBTXR_full_unet | 0 | 20265 / 291315 | 6.96% | 5.48 it/s | 13:44:25 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:48 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 82% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 9142 / 36415 | 25.11% | 11.52 it/s | 39:27 | 2 |
| HBTXR_full_unet | 0 | 19736 / 291315 | 6.77% | 5.48 it/s | 13:46:01 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:47 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 93% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 8236 / 36415 | 22.62% | 11.52 it/s | 40:45 | 2 |
| HBTXR_full_unet | 0 | 19305 / 291315 | 6.63% | 5.48 it/s | 13:47:21 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:45 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 88% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 7218 / 36415 | 19.82% | 11.52 it/s | 42:13 | 2 |
| HBTXR_full_unet | 0 | 18820 / 291315 | 6.46% | 5.48 it/s | 13:48:51 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:44 KST Progress Refresh

Both full-training jobs remain active and continue to update their logs. No new
full validation checkpoint has appeared since the previous refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 88% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 6396 / 36415 | 17.56% | 11.53 it/s | 43:24 | 2 |
| HBTXR_full_unet | 0 | 18429 / 291315 | 6.33% | 5.48 it/s | 13:50:03 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:43 KST Progress Refresh

Both full-training jobs remain active. The training logs are still being
updated, and no new full validation checkpoint has appeared since the previous
refresh.

Active training processes:

```text
EPNet/FACET: PID 1428589, GPU0, 4214 MiB
HBTXR: PID 1483023, GPU1, 11334 MiB
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 93% utilization, 11357 MiB used / 16303 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 5492 / 36415 | 15.08% | 11.53 it/s | 44:42 | 2 |
| HBTXR_full_unet | 0 | 17998 / 291315 | 6.18% | 5.48 it/s | 13:51:23 | 0 |

Current-run error scan after the latest run markers still reports no matches
for `Traceback`, `AssertionError`, `Data is invalid`, `RuntimeError`, CUDA OOM,
kill signals, interrupts, `ValueError`, or generic `Exception`.

The status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## 2026-06-26 11:05 KST Recovery Guard

The live EPNet/FACET and HBTXR training jobs were not interrupted.

Because HBTXR currently needs more than 14 hours for one epoch, the full configs were updated with a second `ModelCheckpoint` callback that will save future-run recovery checkpoints every 5000 train steps. This applies only after a restart through the launcher/watchdog path; it does not change the currently running trainer instances.

Updated configs:

```text
references/codebase/software/FACET/configs/DavisEyeEllipse_EPNet_full_unet.yaml
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_full_unet.yaml
```

Step checkpoint locations:

```text
references/codebase/software/FACET/runs/logs/EPNet_full_unet/step_checkpoints/checkpoints
references/codebase/software/FACET/runs/logs/HBTXR_full_unet/step_checkpoints/checkpoints
```

Validation:

```text
YAML parse: passed
callback instantiation: passed
bash -n EPNet launcher: passed
bash -n HBTXR launcher: passed
```

Detailed report:

```text
references/report/FACET/FACET_full_training_step_checkpoint_recovery_2026-06-26.md
```

## 2026-06-26 11:06 KST Status

Refreshed progress snapshot after the recovery-guard config update:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 19837 / 36415 | 54.47% | 11.49 it/s | 24:02 | 0 |
| HBTXR_full_unet | 0 | 6011 / 291315 | 2.06% | 5.47 it/s | 14:29:01 | 0 |

Status checker output remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

Evaluation is still gated on full checkpoints and completion markers.

## 2026-06-26 11:08 KST Evaluation Gate Check

The step checkpoint recovery guard introduced a possible checkpoint-selection risk for final evaluation. The final evaluation script and checkpoint watcher were updated to exclude recovery checkpoints under `*/step_checkpoints/*`.

Updated files:

```text
references/report/FACET/run_full_checkpoint_evaluation_2026-06-26.sh
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

Validation:

```text
bash -n run_full_checkpoint_evaluation_2026-06-26.sh: passed
bash -n watch_full_checkpoints_and_evaluate_2026-06-26.sh: passed
evaluation dry run: missing EPNet full checkpoint, exit code 2
watcher one-loop dry run: ep_ckpt_count=0 hb_ckpt_count=0, exit code 3 before evaluation
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 20812 / 36415 | 57.15% | 11.50 it/s | 22:37 | 0 |
| HBTXR_full_unet | 0 | 6475 / 291315 | 2.22% | 5.47 it/s | 14:27:27 | 0 |

Current-run error scan:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

Status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

## 2026-06-26 11:12 KST Status

Training sessions and watchdogs remain active:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
facet_full_training_watchdog
```

GPU/process evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 93% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 95% utilization, 11357 MiB used
PID 1428589: EPNet/FACET full training, 4214 MiB
PID 1483023: HBTXR full training, 11334 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 23969 / 36415 | 65.82% | 11.50 it/s | 18:02 | 0 |
| HBTXR_full_unet | 0 | 7976 / 291315 | 2.74% | 5.47 it/s | 14:22:32 | 0 |

Current-run error scan:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

Status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

No full checkpoint exists yet, so final evaluation remains gated.

## 2026-06-26 11:36 KST EPNet First Checkpoint

EPNet/FACET completed epoch 0 training and validation, then produced the first full checkpoint.

Checkpoint files:

```text
references/codebase/software/FACET/runs/logs/EPNet_full_unet/version_4/checkpoints/epoch=00-val_mean_distance=1.8744.ckpt
references/codebase/software/FACET/runs/logs/EPNet_full_unet/version_4/checkpoints/last.ckpt
```

Checkpoint evidence:

```text
epoch=00-val_mean_distance=1.8744.ckpt: 46M, 47192106 bytes
last.ckpt: 46M
```

Current progress snapshot after the checkpoint:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 569 / 36415 | 1.56% | 11.52 it/s | 51:50 | 2 |
| HBTXR_full_unet | 0 | 15655 / 291315 | 5.37% | 5.48 it/s | 13:58:37 | 0 |

Status checker advanced:

```text
overall_status: incomplete
passed: 10
missing: 4
```

The newly passed gate is:

```text
Phase 4 full EPNet checkpoint
```

Final evaluation dry run now fails at the expected next missing gate:

```text
missing HBTXR full checkpoint
exit code: 2
```

The EPNet checkpoint alone is not enough to complete Phase 4 because the run still requires the full 70-epoch completion marker and final comparison artifacts. Phase 4B also still requires HBTXR checkpoint and completion.

## 2026-06-26 11:37 KST Post-Checkpoint Stability

EPNet/FACET continued into epoch 1 after saving the epoch 0 validation checkpoint.

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 1571 / 36415 | 4.31% | 11.53 it/s | 50:22 | 2 |
| HBTXR_full_unet | 0 | 16133 / 291315 | 5.54% | 5.48 it/s | 13:57:08 | 0 |

Current-run error scan remains clean:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

Status checker remains:

```text
overall_status: incomplete
passed: 10
missing: 4
```

Final evaluation remains gated by:

```text
EPNet 70-epoch completion marker
HBTXR first/full checkpoint
HBTXR 70-epoch completion marker
final comparison artifacts
```

## 2026-06-26 11:09 KST Watcher Refresh

The evaluation watcher session was restarted so the live tmux process uses the updated final-checkpoint selection rule that excludes `*/step_checkpoints/*`.

Action:

```text
tmux kill-session -t facet_full_eval_watcher
tmux new-session -d -s facet_full_eval_watcher "bash '/home/kjm26/project/PRJXR/HBTXR/references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh'"
```

The EPNet/FACET and HBTXR training sessions were not interrupted:

```text
facet_epnet_full_gpu0: alive
facet_hbtxr_full_gpu1: alive
facet_full_training_watchdog: alive
facet_full_eval_watcher: recreated at 2026-06-26 11:09 KST
```

Post-restart watcher evidence:

```text
[2026-06-26T11:09:28+0900] loop=1 ep_ckpt_count=0 hb_ckpt_count=0 ep_done=0 hb_done=0 require_completed=1
  ep_latest=missing
  hb_latest=missing
```

Current progress snapshot after watcher restart:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 21784 / 36415 | 59.82% | 11.50 it/s | 21:12 | 0 |
| HBTXR_full_unet | 0 | 6937 / 291315 | 2.38% | 5.47 it/s | 14:25:55 | 0 |

Current-run error scan remains clean:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

## 2026-06-26 11:10 KST Evaluation Selection Policy

The final evaluation script was refined to choose the lowest `val_mean_distance` checkpoint when metric-bearing checkpoint filenames exist. If no metric can be parsed, it falls back to the newest eligible full checkpoint.

Selection policy:

```text
preferred: lowest val_mean_distance in checkpoint filename
fallback: newest non-last full checkpoint
excluded: last.ckpt and */step_checkpoints/*
```

Validation:

```text
bash -n run_full_checkpoint_evaluation_2026-06-26.sh: passed
evaluation dry run: missing EPNet full checkpoint, exit code 2
synthetic checkpoint selection: selected epoch=01-val_mean_distance=3.1000.ckpt over newer worse checkpoints and a step checkpoint
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 22696 / 36415 | 62.33% | 11.50 it/s | 19:52 | 0 |
| HBTXR_full_unet | 0 | 7369 / 291315 | 2.53% | 5.47 it/s | 14:24:30 | 0 |

Status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

## 2026-06-26 11:02 KST Status

Current full-training sessions remain active:

```text
tmux: facet_epnet_full_gpu0
tmux: facet_hbtxr_full_gpu1
tmux: facet_full_eval_watcher
tmux: facet_full_training_watchdog
```

GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 90% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 11357 MiB used
```

Compute processes:

```text
PID 1428589: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
PID 1483023: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 11334 MiB
```

Refreshed progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 16538 / 36415 | 45.42% | 11.49 it/s | 28:50 | 0 |
| HBTXR_full_unet | 0 | 4441 / 291315 | 1.52% | 5.47 it/s | 14:34:25 | 0 |

The current-run log slices after the latest training start markers were checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

No matches were found after the current start markers.

Status checker output was refreshed:

```text
overall_status: incomplete
passed: 9
missing: 5
```

The missing gates remain full EPNet checkpoint/completion, full HBTXR checkpoint/completion, and final evaluation/comparison artifacts.

## 2026-06-26 11:03 KST Status

Training and watcher sessions remain active:

```text
facet_epnet_full_gpu0
facet_hbtxr_full_gpu1
facet_full_eval_watcher
facet_full_training_watchdog
```

GPU and process evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 11357 MiB used
PID 1428589: EPNet/FACET full training, 4214 MiB
PID 1483023: HBTXR full training, 11334 MiB
```

Refreshed progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 17488 / 36415 | 48.02% | 11.49 it/s | 27:27 | 0 |
| HBTXR_full_unet | 0 | 4894 / 291315 | 1.68% | 5.47 it/s | 14:32:48 | 0 |

Current-run error scan after the latest training start markers:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

Checkpoint and completion gates remain open:

```text
full EPNet checkpoints: 0
full HBTXR checkpoints: 0
completion markers: none
final evaluation artifacts: not generated
```

The status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

The probe script checks candidate HBTXR batch sizes with real forward/backward
passes and records CUDA peak memory plus samples/sec. It should be run only
when a GPU can be used without disrupting the active full training jobs, or if
we intentionally decide to pause/restart HBTXR based on the throughput risk.

Prepared wait script:

```text
references/report/FACET/run_hbtxr_batch_probe_gpu0_when_free_2026-06-26.sh
```

This waits until GPU0 has no compute app, then runs the HBTXR batch-size probe
on GPU0. It is intentionally not started now because EPNet is actively training
on GPU0.

Dry-run note:

- A first `MAX_LOOPS=1` dry run exposed that `nvidia-smi` can fail inside the
  restricted sandbox and make GPU0 look free.
- The wait script was changed to fail closed: if GPU state cannot be read, it
  treats GPU0 as busy and does not start the probe.
- A second dry run confirmed the safe behavior:
  `GPU0 busy; waiting before HBTXR batch probe`, exit code `3`.
- `HBTXR_batch_probe_gpu0_2026-06-26.log` contains the failed dry-run trace;
  that trace is not from the active EPNet/HBTXR training jobs.

ETA and HBTXR batch-size risk note:

```text
references/report/FACET/FACET_full_training_eta_and_hbtxr_batch_risk_2026-06-26.md
```

At the current observed throughput, EPNet's train-only 70-epoch estimate is
about 62 hours, while HBTXR's train-only 70-epoch estimate is about 44.5 days.
This makes the HBTXR `batch_size: 2` setting the main practical risk for the
full parallel comparison goal.

## Update: 2026-06-26 10:26 KST

Live sessions remain active:

```text
tmux: facet_epnet_full_gpu0
tmux: facet_hbtxr_full_gpu1
tmux: facet_full_eval_watcher
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 92% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 5943 MiB used
```

Current compute apps:

```text
PID 1191693: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
PID 125360: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 5920 MiB
```

Current progress snapshot:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 18283 / 36415 | 50.21% | 11.44 it/s | 26:25 | 0 |
| HBTXR_full_unet | 0 | 94870 / 582630 | 16.28% | 10.61 it/s | 12:46:20 | 0 |

The active training logs are still growing:

```text
2026-06-26 10:26:52 +0900 5410186 references/report/FACET/EPNet_full_unet_gpu0_train_2026-06-26.log
2026-06-26 10:26:52 +0900 15655552 references/report/FACET/HBTXR_full_unet_gpu1_train_2026-06-26.log
```

Recent log tails for both EPNet and HBTXR were checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
```

No matches were found in the current tails.

Follow-up check at `2026-06-26 10:32 KST` confirmed both runs are still active:

```text
GPU0: 88% utilization, 4237 MiB used
GPU1: 94% utilization, 5943 MiB used
```

Updated progress:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 22345 / 36415 | 61.36% | 11.44 it/s | 20:29 | 0 |
| HBTXR_full_unet | 0 | 1648 / 582630 | 0.28% | 10.41 it/s | 15:30:24 | 0 |

The HBTXR log slice after the latest restart marker still has no error-string
matches.

The status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

No full checkpoint has been produced yet, so the next gates remain unchanged:

```text
Phase 4 full EPNet checkpoint
Phase 4 full EPNet training completion
Phase 4B full HBTXR checkpoint
Phase 4B full HBTXR training completion
Phase 4 final evaluation artifacts
```

## Update: 2026-06-26 10:30 KST

HBTXR `version_2` stopped during epoch 0 with a DataLoader-side
`ToFrameStack.normalize()` assertion:

```text
AssertionError: Data is invalid.
```

EPNet/FACET on GPU0 was not interrupted and continues to train.

Recovery changes:

```text
references/codebase/software/FACET/EvEye/utils/tonic/functional/ToFrameStack.py
references/report/FACET/run_hbtxr_full_unet_gpu1_when_ready_2026-06-26.sh
```

Validation completed:

```text
py_compile: passed
bash -n HBTXR launcher: passed
normalize() degenerate timestamp smoke: passed
DavisEyeEllipseDataset train sample smoke, including index 95784: passed
```

HBTXR was restarted on GPU1:

```text
tmux: facet_hbtxr_full_gpu1
PID: 1376090
Lightning run: HBTXR_full_unet/version_3
```

Current GPU evidence:

```text
GPU0: NVIDIA GeForce RTX 5080, 90% utilization, 4237 MiB used
GPU1: NVIDIA GeForce RTX 5080, 94% utilization, 5943 MiB used
```

Current compute apps:

```text
PID 1191693: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 4214 MiB
PID 1376090: /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python, 5920 MiB
```

Current progress snapshot after restart:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 0 | 20984 / 36415 | 57.62% | 11.44 it/s | 22:28 | 0 |
| HBTXR_full_unet | 0 | 393 / 582630 | 0.07% | 9.91 it/s | 16:19:20 | 0 |

The HBTXR log slice after the latest restart marker was checked for:

```text
Traceback
AssertionError
Data is invalid
RuntimeError
CUDA out of memory
Killed
KeyboardInterrupt
ValueError
Exception
```

No matches were found after the restart marker.

Recovery report:

```text
references/report/FACET/FACET_hbtxr_toframestack_recovery_2026-06-26.md
```

The status checker remains:

```text
overall_status: incomplete
passed: 9
missing: 5
```

## Latest EOF Snapshot: 2026-06-26 12:21 KST

This is the canonical latest snapshot for the current monitoring pass.

Runtime state:

```text
facet_epnet_full_gpu0: alive
facet_hbtxr_full_gpu1: alive
facet_full_eval_watcher: alive
facet_full_training_watchdog: alive
```

Active training processes:

```text
PID 1428589: tools/train.py -c DavisEyeEllipse_EPNet_full_unet.yaml
PID 1483023: tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml
PID 1573595: watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

GPU state:

```text
GPU0: NVIDIA GeForce RTX 5080, 94% utilization, 4237 MiB used / 16303 MiB
GPU1: NVIDIA GeForce RTX 5080, 100% utilization, 11357 MiB used / 16303 MiB
```

Progress:

| Model | Epoch | Step | Epoch progress | Rate | Epoch remaining | Checkpoints |
|---|---:|---:|---:|---:|---:|---:|
| EPNet_full_unet | 1 | 31734 / 36415 | 87.15% | 11.49 it/s | 06:47 | 2 |
| HBTXR_full_unet | 0 | 30535 / 291315 | 10.48% | 5.48 it/s | 13:12:57 | 0 |

Current-run error scan:

```text
EPNet: marker_found=True hits=[]
HBTXR: marker_found=True hits=[]
```

Final evaluation dry-run:

```text
exit code: 2
output: missing HBTXR full checkpoint
```

Status:

```text
overall_status: incomplete
passed: 10
missing: 4
```

## Monitoring Cadence Update: 2026-06-26 12:25 KST

User requested less frequent polling because the previous 300-second monitoring cadence was too aggressive.

Updated scripts:

```text
references/report/FACET/watch_full_training_jobs_2026-06-26.sh
references/report/FACET/watch_full_checkpoints_and_evaluate_2026-06-26.sh
```

New default intervals:

```text
FACET_TRAINING_WATCHDOG_INTERVAL_SECONDS default: 3600
FACET_WATCH_INTERVAL_SECONDS default: 3600
```

Runtime action:

```text
Restarted facet_full_training_watchdog.
The watchdog restarted facet_full_eval_watcher.
Training sessions facet_epnet_full_gpu0 and facet_hbtxr_full_gpu1 were left running.
```

## Monitoring Cadence Update: 2026-06-26 13:00 KST

User requested training-result monitoring to run hourly instead of checking too
frequently.

Confirmed or updated default intervals:

```text
FACET_TRAINING_WATCHDOG_INTERVAL_SECONDS default: 3600
FACET_WATCH_INTERVAL_SECONDS default: 3600
FACET_FOLLOWUP_WATCHDOG_INTERVAL_SECONDS default: 3600
FACET_FPN_DW_WATCH_INTERVAL_SECONDS default: 3600
FACET_EFFBS32_WATCH_INTERVAL_SECONDS default: 3600
FACET_FPN_DW_WAIT_INTERVAL_SECONDS default: 3600
FACET_EFFBS32_WAIT_INTERVAL_SECONDS default: 3600
FACET_HBTXR_PROBE_INTERVAL_SECONDS default: 3600
```

Current policy:

```text
Do not run routine training-result checks more frequently than once per hour.
Manual one-off checks are reserved for explicit user requests, error recovery,
or verifying script/session configuration changes.

## Monitoring Cadence Verification: 2026-06-26 14:56 KST

The active watcher/waiter sessions were checked after the hourly-monitoring
request. No training tmux session was stopped or restarted.

Observed live loop cadence:

```text
facet_full_training_watchdog: 12:25 -> 13:25 -> 14:25
facet_full_eval_watcher: 12:25 -> 13:25 -> 14:25
facet_followup_training_watchdog: 12:40 -> 13:40 -> 14:40
facet_epnet_fpn_dw_eval_watcher: 12:36 -> 13:36 -> 14:36
facet_hbtxr_effbs32_eval_watcher: 12:38 -> 13:38 -> 14:38
facet_epnet_fpn_dw_gpu0_waiter: 12:35 -> 13:35 -> 14:35
facet_hbtxr_effbs32_gpu1_waiter: 12:38 -> 13:38 -> 14:38
```

Conclusion:

```text
The active FACET training-result monitoring cadence is already hourly.
No watcher restart was needed.
```
```
