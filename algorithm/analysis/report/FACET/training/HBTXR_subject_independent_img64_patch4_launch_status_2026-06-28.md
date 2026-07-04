# HBTXR Subject-Independent Img64 Patch4 Launch Status

Date: 2026-06-28

## Scope

This status document records the execution state for the subject-independent HBTXR experiment planned in:

```text
references/report/FACET/HBTXR_subject_independent_img64_patch4_plan_2026-06-28.md
```

Repeated-run confidence intervals remain excluded. The target outputs after training are the same report categories as `HBTXR_val_motion_eval`, but for both validation and test subjects.

## Subject Split

The leak-free subject-independent split was generated at:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent
```

The split uses:

| split | subjects | samples |
|---|---:|---:|
| train | 1-32 | 968,873 |
| val | 33-36 | 122,776 |
| test | 37-48 | 366,171 |

The literal user wording `1-36 train / 33-36 val / 37-48 test` overlaps subjects 33-36 between train and val, so the implemented split corrects it to train 1-32 / val 33-36 / test 37-48.

## Generated Files

```text
references/codebase/software/FACET/EvEye/utils/scripts/resplit_dean_dataset_by_subject.py
references/codebase/software/FACET/configs/DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
references/report/FACET/run_hbtxr_subject_independent_img64_patch4_gpu1_2026-06-28.sh
references/report/FACET/run_hbtxr_subject_independent_img64_patch4_eval_after_training_2026-06-28.sh
```

The motion-evaluation script was generalized for arbitrary split names:

```text
references/codebase/software/FACET/EvEye/utils/scripts/evaluate_hbtxr_val_motion.py
```

## Verification Performed

- Re-split script dry-run and full run completed.
- New dataset root contains `manifest.json` and `progress_state.json`.
- Dataset loader smoke passed for all splits:
  - train: `(2, 64, 64)` input, `(1, 16, 16)` heatmap.
  - val: `(2, 64, 64)` input, `(1, 16, 16)` heatmap.
  - test: `(2, 64, 64)` input, `(1, 16, 16)` heatmap.
- Training config smoke passed:
  - HBTXR params: 4.4M trainable.
  - `img_size=64`, `patch_size=4`, output heatmap `16x16`.
  - Loss forward/backward contract smoke produced finite loss.
- Evaluation script smoke passed with `--split val --max-samples 8`.
- `bash -n` passed for both launcher scripts.

## Running Jobs

Training session:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_gpu1
command: tools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
log: references/report/FACET/HBTXR_subject_independent_img64_patch4_gpu1_train_2026-06-28.log
GPU: GPU1
```

Evaluation watcher:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_eval_after_training
log: references/report/FACET/HBTXR_subject_independent_img64_patch4_eval_after_training_2026-06-28.log
state: waiting for training to finish
```

As of launch verification, the training process had entered Epoch 0 and was running at roughly 32 it/s after warmup. Each epoch has 30,278 train steps.

## Expected Final Outputs

After training exits, the watcher will select the best checkpoint by lowest `val_mean_distance` filename and run:

```text
val:  references/report/FACET/HBTXR_subject_independent_img64_patch4_val_motion_eval/
test: references/report/FACET/HBTXR_subject_independent_img64_patch4_test_motion_eval/
```

Expected report files:

```text
references/report/FACET/HBTXR_subject_independent_img64_patch4_val_motion_eval_2026-06-28.md
references/report/FACET/HBTXR_subject_independent_img64_patch4_test_motion_eval_2026-06-28.md
```

Each report will cover:

1. Subject-wise pixel error / IoU distribution.
2. Subject-wise motion distribution.
3. Subject-wise mean / median / P95 / P99 pixel error.
4. Motion-wise mean / median / P95 / P99 pixel error.
5. Annotation precision and pseudo-label noise.

## Remaining Risk

The final reports are pending because the 70-epoch training run is still active. The evaluation pipeline has been smoke-tested, but full val/test evaluation must wait for a trained checkpoint.

## Status Refresh: 2026-06-28 23:52 KST

Current training status:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_gpu1
process: tools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
latest parsed progress: Epoch 0, 9,990 / 30,278 steps, 33%
throughput: about 33 it/s after warmup
```

Checkpoint state:

```text
step checkpoint: step-00005000.ckpt
epoch checkpoint: not yet available
val/test motion reports: not yet available
```

Watcher hardening:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_eval_after_training
script: references/report/FACET/run_hbtxr_subject_independent_img64_patch4_eval_after_training_2026-06-28.sh
```

The watcher was updated and restarted so it does not evaluate a crashed or partial run. It now loads the final `last.ckpt`, checks that the checkpoint epoch is at least `max_epochs - 1`, and only then selects the best `val_mean_distance` checkpoint for val/test motion evaluation. If the training process exits early, the watcher refuses final evaluation instead of producing misleading reports from an incomplete checkpoint.

## Status Refresh: 2026-06-28 23:53 KST

Current training status:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_gpu1
process: tools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
latest parsed progress: Epoch 0, 11,865 / 30,278 steps, 39%
throughput: about 33 it/s after warmup
```

Checkpoint state:

```text
step checkpoint: step-00010000.ckpt
epoch checkpoint: not yet available
val/test motion reports: not yet available
```

The evaluator watcher remains active and waiting:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_eval_after_training
state: waiting for the training process to exit, then checking final checkpoint epoch before evaluation
```

## Status Refresh: 2026-06-28 23:54 KST

Current training status:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_gpu1
process: tools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
latest parsed progress: Epoch 0, 13,705 / 30,278 steps, 45%
throughput: about 33 it/s after warmup
```

Checkpoint/report state:

```text
latest step checkpoint: step-00010000.ckpt
epoch validation checkpoint: not yet available
val/test motion reports: not yet available
```

## Status Refresh: 2026-06-28 23:55 KST

Current training status:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_gpu1
process: tools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
latest parsed progress: Epoch 0, 15,238 / 30,278 steps, 50%
throughput: about 32 it/s after warmup
```

Checkpoint/report state:

```text
latest step checkpoint: step-00015000.ckpt
epoch validation checkpoint: not yet available
val/test motion reports: not yet available
```

## Status Refresh: 2026-06-28 23:56 KST

Artifact validation was added for the final val/test motion reports:

```text
script: references/codebase/software/FACET/EvEye/utils/scripts/validate_hbtxr_motion_eval_artifacts.py
watcher: references/report/FACET/run_hbtxr_subject_independent_img64_patch4_eval_after_training_2026-06-28.sh
validation output: references/report/FACET/HBTXR_subject_independent_img64_patch4_motion_eval_validation_2026-06-28.json
```

The validator checks:

```text
required Markdown report sections
required CSV tables
required figures
prediction row count against expected split sample count
subject rows
motion states: Fixation, Saccade, Smooth
motion stats aggregate row: All
```

The validator was smoke-tested before final reports existed and correctly failed with missing val/test artifacts. The evaluation watcher was restarted at 23:56 KST so the new validation step is definitely active. The training session was not restarted.

## Status Refresh: 2026-06-28 23:58 KST

The final combined results-report builder was added and connected after artifact validation:

```text
script: references/codebase/software/FACET/EvEye/utils/scripts/build_hbtxr_subject_independent_results_report.py
final report: references/report/FACET/HBTXR_subject_independent_img64_patch4_results_2026-06-28.md
```

The builder reads the validated val/test motion-eval CSVs and validator JSON, then writes a single subject-independent results report with split-level summaries, motion-wise tables, highest-median subjects, artifact coverage, and the checkpoint/config provenance. It was syntax-checked and smoke-tested to fail cleanly while val/test artifacts are still missing. The evaluation watcher was restarted at 23:58 KST so this final report step is active. The training session was not restarted.

## Status Refresh: 2026-06-29 00:00 KST

Current training status:

```text
tmux: facet_hbtxr_subject_independent_img64_patch4_gpu1
process: tools/train.py -c DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml
latest parsed progress: Epoch 0, 25,319 / 30,278 steps, 84%
throughput: about 32 it/s after warmup
```

Checkpoint/report state:

```text
latest step checkpoint: step-00025000.ckpt
epoch validation checkpoint: not yet available
val/test motion reports: not yet available
final combined results report: not yet available
```

The evaluation watcher is still active and waiting for the training process to exit. The report filenames remain tied to the experiment launch date (`2026-06-28`) for consistency with the plan and launcher scripts.

## Status Refresh: 2026-06-29 00:03 KST

Epoch 0 completed and produced the first validation checkpoint:

```text
checkpoint: references/codebase/software/FACET/runs/logs/HBTXR_subject_independent_img64_patch4/version_0/checkpoints/epoch=00-val_mean_distance=1.3989.ckpt
val_loss: 8.660
val_mean_distance: 1.3989
val_IoU: 0.303
val_AP: 0.0133
val_p10_acc: 1.000
val_p5_acc: 0.981
val_p3_acc: 0.891
val_p1_acc: 0.487
```

Current training status after validation:

```text
latest parsed progress: Epoch 1, about 2,285 / 30,278 steps, 8%
training process: still running
evaluation watcher: still waiting for final 70-epoch completion
```

This is only the first epoch checkpoint. The final val/test subject-independent motion reports are still pending because the configured training target is `max_epochs: 70`, and the watcher intentionally refuses final evaluation until the final `last.ckpt` epoch is at least `max_epochs - 1`.

## Status Refresh: 2026-06-29 00:05 KST

Current training status:

```text
latest parsed progress: Epoch 1, 4,180 / 30,278 steps, 14%
latest validation carried in progress bar: val_mean_distance=1.400, val_IoU=0.303, val_AP=0.0133
training process: still running
evaluation watcher: still waiting for final completion
```

Checkpoint/report state:

```text
best epoch checkpoint so far: epoch=00-val_mean_distance=1.3989.ckpt
final val/test motion reports: not yet available
final combined results report: not yet available
```

## Status Refresh: 2026-06-29 01:16 KST

Current training status:

```text
latest parsed progress: Epoch 5, 19,402 / 30,278 steps, 64%
latest validation carried in progress bar: val_mean_distance=1.060, val_IoU=0.386, val_AP=0.130
training process: still running
evaluation watcher: still waiting for final 70-epoch completion
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=04-val_mean_distance=1.0617.ckpt
latest epoch checkpoint: epoch=04-val_mean_distance=1.0617.ckpt
latest step checkpoint: step-00170000.ckpt
final val/test motion reports: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 97% utilization
GPU1: pid 1212571, about 1.7 GiB, 76% utilization
```

## Status Refresh: 2026-06-29 02:16 KST

Current training status:

```text
latest parsed progress: Epoch 9, 8,551 / 30,278 steps, 28%
latest validation carried in progress bar: val_mean_distance=0.800, val_IoU=0.464, val_AP=0.276
latest carried val_loss: 6.370
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.982 / 0.936 / 0.798
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=08-val_mean_distance=0.7998.ckpt
latest epoch checkpoint: epoch=08-val_mean_distance=0.7998.ckpt
latest step checkpoint: step-00280000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 94% utilization
GPU1: pid 1212571, about 1.7 GiB, 69% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run is healthy and still training.
Validation has improved from the 01:16 KST snapshot: val_mean_distance moved from about 1.060 to about 0.800, IoU from 0.386 to 0.464, and AP from 0.130 to 0.276.
The final val/test motion evaluation reports remain pending because the configured training target is max_epochs=70 and the evaluation watcher intentionally waits for final training completion.
```

## Status Refresh: 2026-06-29 03:16 KST

Current training status:

```text
latest parsed progress: Epoch 13, 1,475 / 30,278 steps, 5%
latest validation carried in progress bar: val_mean_distance=0.777, val_IoU=0.474, val_AP=0.312
latest carried val_loss: 6.470
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.984 / 0.940 / 0.798
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=12-val_mean_distance=0.7773.ckpt
latest epoch checkpoint: epoch=12-val_mean_distance=0.7773.ckpt
latest step checkpoint: step-00395000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 94% utilization
GPU1: pid 1212571, about 1.7 GiB, 73% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run is still healthy and advancing.
Validation continued to improve from the 02:16 KST snapshot: val_mean_distance moved from about 0.800 to about 0.777, IoU from 0.464 to 0.474, and AP from 0.276 to 0.312.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```

## Status Refresh: 2026-06-29 04:16 KST

Current training status:

```text
latest parsed progress: Epoch 16, 23,942 / 30,278 steps, 79%
latest validation carried in progress bar: val_mean_distance=0.734, val_IoU=0.481, val_AP=0.286
latest carried val_loss: 6.590
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.985 / 0.945 / 0.814
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=15-val_mean_distance=0.7335.ckpt
latest epoch checkpoint: epoch=15-val_mean_distance=0.7335.ckpt
latest step checkpoint: step-00505000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 98% utilization
GPU1: pid 1212571, about 1.7 GiB, 66% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run remains healthy and is still training.
Validation continued to improve in mean-distance from the 03:16 KST snapshot: val_mean_distance moved from about 0.777 to about 0.734, and IoU moved from 0.474 to 0.481. AP moved from about 0.312 to 0.286, so AP is not monotonically improving at this snapshot.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```

## Status Refresh: 2026-06-29 05:16 KST

Current training status:

```text
latest parsed progress: Epoch 20, 17,963 / 30,278 steps, 59%
latest validation carried in progress bar: val_mean_distance=0.674, val_IoU=0.498, val_AP=0.310
latest carried val_loss: 6.410
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.989 / 0.953 / 0.831
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=19-val_mean_distance=0.6744.ckpt
latest epoch checkpoint: epoch=19-val_mean_distance=0.6744.ckpt
latest step checkpoint: step-00620000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 95% utilization
GPU1: pid 1212571, about 1.7 GiB, 67% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run remains healthy and continues to improve in the primary validation distance.
Validation improved from the 04:16 KST snapshot: val_mean_distance moved from about 0.734 to about 0.674, IoU from 0.481 to 0.498, and AP from 0.286 to 0.310.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```

## Status Refresh: 2026-06-29 13:22 KST

Current training status:

```text
latest parsed progress: Epoch 51, 5,829 / 30,278 steps, 19%
latest validation carried in progress bar: val_mean_distance=0.572, val_IoU=0.518, val_AP=0.391
latest carried val_loss: 6.070
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.992 / 0.966 / 0.861
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=50-val_mean_distance=0.5722.ckpt
latest epoch checkpoint: epoch=50-val_mean_distance=0.5722.ckpt
latest step checkpoint: step-01545000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 94% utilization
GPU1: pid 1212571, about 1.7 GiB, 71% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run remains healthy and is approaching the final third of the configured 70-epoch training schedule.
Validation improved substantially from the 05:16 KST snapshot: val_mean_distance moved from about 0.674 to about 0.572, IoU from 0.498 to 0.518, and AP from 0.310 to 0.391.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```

## Status Refresh: 2026-06-29 14:22 KST

Current training status:

```text
latest parsed progress: Epoch 54, 29,312 / 30,278 steps, 97%
latest validation carried in progress bar: val_mean_distance=0.602, val_IoU=0.516, val_AP=0.389
latest carried val_loss: 5.960
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.989 / 0.959 / 0.855
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=50-val_mean_distance=0.5722.ckpt
latest saved epoch checkpoint: epoch=52-val_mean_distance=0.5816.ckpt
latest step checkpoint: step-01660000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 100% utilization
GPU1: pid 1212571, about 1.7 GiB, 76% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run remains healthy and has reached the late part of epoch 54.
The latest carried val_mean_distance is worse than the best saved checkpoint at epoch 50, but the best checkpoint remains available as epoch=50-val_mean_distance=0.5722.ckpt.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```

## Status Refresh: 2026-06-29 15:22 KST

Current training status:

```text
latest parsed progress: Epoch 58, 21,490 / 30,278 steps, 71%
latest validation carried in progress bar: val_mean_distance=0.562, val_IoU=0.522, val_AP=0.410
latest carried val_loss: 5.830
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.992 / 0.963 / 0.867
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=57-val_mean_distance=0.5619.ckpt
latest saved epoch checkpoint: epoch=57-val_mean_distance=0.5619.ckpt
latest step checkpoint: step-01775000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 86% utilization
GPU1: pid 1212571, about 1.7 GiB, 75% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run remains healthy and is now in epoch 58 of the configured 70-epoch schedule.
The best saved checkpoint has improved to epoch=57-val_mean_distance=0.5619.ckpt, with the latest carried metrics also showing improved IoU and AP compared with the 14:22 KST snapshot.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```

## Status Refresh: 2026-06-29 17:09 KST

Current training status:

```text
latest parsed progress: Epoch 65, 11,489 / 30,278 steps, 38%
latest validation carried in progress bar: val_mean_distance=0.576, val_IoU=0.524, val_AP=0.404
latest carried val_loss: 5.920
latest carried val_p10_acc / val_p5_acc / val_p3_acc / val_p1_acc: 1.000 / 0.990 / 0.961 / 0.863
training process: still running, pid 1212571
evaluation watcher: still waiting for final 70-epoch completion, pid 1248503
```

Checkpoint/report state:

```text
best epoch checkpoint so far by filename metric: epoch=58-val_mean_distance=0.5552.ckpt
latest saved epoch checkpoint: epoch=62-val_mean_distance=0.5564.ckpt
latest step checkpoint: step-01975000.ckpt
final val/test motion reports: not yet available
final validation JSON: not yet available
final combined results report: not yet available
```

GPU snapshot:

```text
GPU0: pid 2397606, about 7.8 GiB, 94% utilization
GPU1: pid 1212571, about 1.7 GiB, 81% utilization
```

Interpretation:

```text
The subject-independent HBTXR img64 patch4 run remains healthy and has reached epoch 65 of the configured 70-epoch schedule.
The best saved checkpoint improved after the previous snapshot to epoch=58-val_mean_distance=0.5552.ckpt. The latest carried val_mean_distance at epoch 65 is 0.576, so the current carried metric is not better than the best saved checkpoint.
The final val/test motion evaluation reports remain pending because training has not reached the configured max_epochs=70 completion gate.
```
