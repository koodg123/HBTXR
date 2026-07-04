# HGTXR-SW Execution Notes

Date: 2026-06-10

## 2026-06-18 XR-63 Execution

Implemented and ran no-train teacher-target oracle diagnostic:

```bash
python3 -m py_compile scripts/external/analyze_p10_teacher_target_oracle.py
.venv/bin/python scripts/external/analyze_p10_teacher_target_oracle.py --limit 64 --output-json /tmp/xr63_smoke_evalstyle.json --output-md /tmp/xr63_smoke_evalstyle.md
.venv/bin/python scripts/external/analyze_p10_teacher_target_oracle.py
```

Outputs:

- `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.json`
- `docs/resources/xr63_p10_teacher_target_oracle_2026_06_18.md`

Result:

- XR-63 oracle upper bound: `16.04023192701366/36.48596938775512/13.41751700680271`.
- All active gates exceeded as an upper bound.
- Next executable branch should implement leakage-safe P10 teacher-target construction.

## 2026-06-18 XR-64 Implementation Validation

Implemented target override construction path:

```bash
python3 -m py_compile scripts/external/build_xr64_teacher_target_overrides.py scripts/external/analyze_p10_teacher_target_oracle.py src/hbtxr/data/dataset.py src/hbtxr/config/runtime_config.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/stage2.py
bash -n scripts/external/run_xr64_teacher_target_construction.sh
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_target_override.py tests/test_track_center_l2_loss.py
DRY_RUN=1 bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1
```

Next execution:

1. Generate train eval rows for XR-62A, XR-39, XR-56B, and XR-58A.
2. Build train-only override files with `scripts/external/build_xr64_teacher_target_overrides.py`.
3. Launch XR-64A/B.

## Immediate Commands

Summarize existing runs:

```bash
PYTHONPATH=src .venv/bin/python scripts/external/summarize_training_history.py \
  runs/raw_mode1_stage1_event_count_20260610_192838 \
  runs/raw_mode1_stage2_event_count_20260610_193719
```

Dry-run Stage2 ablation commands:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --dry-run
```

Run first Stage2 ablation on CUDA:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --case lr2e-4_nodistill_fullwidth --device cuda:0
```

Run parallel weak-distillation ablation on GPU1:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --case lr1e-4_weakdistill_fullwidth --device cuda:1
```

Sample DA-ROI on GPU0:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --case da_roi --device cuda:0
```

Run event-count sweep after first ablation:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --case weak_count1000 --device cuda:1
sh scripts/external/run_accuracy_experiment_matrix.sh --case weak_count2500 --device cuda:0
sh scripts/external/run_accuracy_experiment_matrix.sh --case weak_count10000 --device cuda:1
```

The event-count sweep intentionally overrides `data.mode1.event_builder.event_count_target`
at train time. The base manifest was built with `5000`, so each sweep run name must
carry the count value for provenance.

Corrected event-count sweep:

```text
GPU0 -> runs/raw_mode1_stage2_count2500_lr1e-4_weakdistill_fullwidth_20260610_210855 -> complete, checker ok=true
GPU1 -> runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_fullwidth_20260610_210856 -> complete, checker ok=true
GPU0 -> runs/raw_mode1_stage2_lr1e-4_weakdistill_fullwidth_20260610_212149 -> complete, checker ok=true
GPU1 -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_fullwidth_20260610_212150 -> complete, checker ok=true
```

Run center-aware count10000 experiments:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh --case center_ckpt_count10000 --device cuda:0
sh scripts/external/run_accuracy_experiment_matrix.sh --case centerloss_count10000 --device cuda:1
```

Center-aware count10000 results:

```text
GPU0 -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707 -> complete, checker ok=true, best center 39.5347
GPU1 -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709 -> complete, checker ok=true, best center 39.8471, best P10 14.3430
```

Run test split validation on GPU0:

```bash
HBTXR_DISABLE_CUDNN=1 PYTHON_BIN=.venv/bin/python sh scripts/external/run_eval.sh \
  --config configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_centerckpt_fullwidth.yaml \
  --mode mode1 \
  --stage stage2 \
  --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --checkpoint runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707/train/best_metric_track_center_px.pt \
  --experiment-name eval_centerckpt_bestcenter_test_gpu0_retry \
  --device cuda:0 \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.event_count_target=10000

HBTXR_DISABLE_CUDNN=1 PYTHON_BIN=.venv/bin/python sh scripts/external/run_eval.sh \
  --config configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_centerloss_fullwidth.yaml \
  --mode mode1 \
  --stage stage2 \
  --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --checkpoint runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_track_p10.pt \
  --experiment-name eval_centerloss_bestp10_test_gpu0_retry \
  --device cuda:0 \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.event_count_target=10000
```

Test split results:

```text
GPU0 -> runs/eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803 -> complete, test center 36.9280, test P10 9.9209
GPU0 -> runs/eval_centerloss_bestp10_test_gpu0_retry_20260610_222812 -> complete, test center 42.3723, test P10 10.9864
```

Run recency-sharpening experiment on GPU1:

```bash
HBTXR_DISABLE_CUDNN=1 sh scripts/external/run_prepare_and_train.sh \
  --mode mode1 \
  --paths-config configs/external/paths/ev_eye_raw_paths.json \
  --canonical-name canonical1 \
  --manifest-name manifest1 \
  --skip-prepare \
  --skip-stage1 \
  --raw-event-count-check \
  --stage1-checkpoint runs/raw_mode1_stage1_event_count_20260610_192838/train/best_search_p10.pt \
  --stage2-config configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_centerckpt_fullwidth.yaml \
  --stage2-experiment-name raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_recency2_fullwidth \
  --stage2-device cuda:1 \
  --stage2-arg --override=data.mode1.event_builder.event_count_target=10000 \
  --stage2-arg --override=data.mode1.event_builder.causal_weight_power=2.0
```

Active session:

```text
tmux session: hgtxr_recency2_gpu1 (stopped after epoch 10 validation)
log: runs/_logs/recency2_gpu1_tmux_20260610_222435.log
run: runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_recency2_fullwidth_20260610_222436
best val center before stop: 44.6194
```

Prepared next GPU1 candidate:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh \
  --case centerloss_lite_count10000 \
  --device cuda:1
```

This case uses `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_centerloss_lite_fullwidth.yaml`.
It is intended to test whether lighter center-loss weights can preserve the center-checkpoint
center error while recovering some of the P10 gain from the stronger centerloss run.

Active session:

```text
tmux session: hgtxr_centerloss_lite_gpu1 (stopped after epoch 10 validation)
log: runs/_logs/centerloss_lite_gpu1_20260610_223242.log
run: runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_lite_fullwidth_20260610_223243
best val center before stop: 44.2607
```

Active GPU1 candidate after the stopped LR `1e-4` run:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh \
  --case centerloss_lite_lr5e-5_count10000 \
  --device cuda:1
```

This uses the same centerloss-lite objective but halves LR to `5.0e-5`.
Rationale: completed evidence supports AdamW and center-aware checkpointing/loss;
there is not yet evidence to justify switching optimizer family.

Active session:

```text
tmux session: hgtxr_centerloss_lite_lr5e5_gpu1
log: runs/_logs/centerloss_lite_lr5e5_gpu1_20260610_224106.log
run: runs/raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerloss_lite_fullwidth_20260610_224120
```

Parallel center-focused GPU0 candidate:

```bash
sh scripts/external/run_accuracy_experiment_matrix.sh \
  --case center_ckpt_lr5e-5_count10000 \
  --device cuda:0
```

This keeps the current center-error leader structure and halves LR to `5.0e-5`.
It is independent from the GPU1 centerloss-lite LR probe.

Active session:

```text
tmux session: hgtxr_centerckpt_lr5e5_gpu0
log: runs/_logs/centerckpt_lr5e5_gpu0_20260610_225000.log
run: runs/raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerckpt_fullwidth_20260610_224355
```

Current GPU1 active/queued work:

```text
stopped: raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ema996_fullwidth_20260610_225302
stopped log: runs/_logs/centerckpt_ema996_gpu1_20260610_230000.log
stopped: raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230041
stopped log: runs/_logs/centerckpt_count1000_gpu1_queue_20260610_231500.log
stopped: raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230218
stopped log: runs/_logs/centerckpt_count5000_gpu0_20260610_231000.log
completed eval: runs/eval_centerloss_bestcenter_test_gpu1_20260610_230859
completed eval log: runs/_logs/eval_centerloss_bestcenter_gpu1_20260610_232000.log
```

EMA996, count1000 center-checkpoint, and count5000 center-checkpoint were
stopped because their epoch-8 center error stayed above the stop threshold.
The centerloss count10000 best-center checkpoint was evaluated on test and is
now the best observed test-center checkpoint.

Current active parallel runs:

```text
GPU0 -> stopped hgtxr_centerloss_strong_gpu0
log  -> runs/_logs/centerloss_strong_gpu0_20260610_232500.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_strong_fullwidth_20260610_231233
gate -> stopped after epoch 8; best val center 44.1654 > 42.0

GPU1 -> stopped hgtxr_centerloss_count5000_gpu1
log  -> runs/_logs/centerloss_count5000_gpu1_20260610_234000.log
run  -> runs/raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_231556
gate -> stopped after epoch-9 observation; best val center 44.3874 > 42.0

GPU0 -> stopped hgtxr_centerckpt_schedulefree_gpu0
log  -> runs/_logs/centerckpt_schedulefree_gpu0_20260611_000000.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_schedulefree_fullwidth_20260610_232114
gate -> stopped after epoch 8; best val center 45.3146 > 42.0

GPU1 -> stopped hgtxr_centerckpt_lion_gpu1
log  -> runs/_logs/centerckpt_lion_gpu1_20260611_000500.log
run  -> runs/raw_mode1_stage2_count10000_lr3e-5_weakdistill_centerckpt_lion_fullwidth_20260610_232431
gate -> stopped after epoch 8; best val center 43.4448 > 42.0

GPU1 -> stopped hgtxr_centerckpt_teachercenter_gpu1
log  -> runs/_logs/centerckpt_teachercenter_gpu1_20260611_001500.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_teachercenter_fullwidth_20260610_233212
gate -> stopped after epoch 8; best val center 44.0106 > 42.0

GPU0 -> aborted hgtxr_centerckpt_ellipseaux_gpu0
log  -> runs/_logs/centerckpt_ellipseaux_gpu0_20260611_002500.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233558
gate -> stopped during epoch 1 because initial OBB aux weights produced loss around 19k

GPU0 -> stopped hgtxr_centerckpt_ellipseaux_light_gpu0
log  -> runs/_logs/centerckpt_ellipseaux_light_gpu0_20260611_003000.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ellipseaux_fullwidth_20260610_233657
gate -> stopped after epoch 8; best val center 44.4728 > 42.0

GPU1 -> stopped hgtxr_centerckpt_selfreg_gpu1
log  -> runs/_logs/centerckpt_selfreg_gpu1_20260611_004000.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_selfreg_fullwidth_20260610_234009
gate -> stopped after epoch 9; best val center 43.5076 > 42.0

GPU0 -> completed hgtxr_centerloss_finetune_gpu0
log  -> runs/_logs/centerloss_finetune_gpu0_20260611_005000.log
run  -> runs/raw_mode1_stage2_count10000_lr2e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234649
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt
gate -> early-stop epoch 9; best val center 40.0042, not promoted versus 39.8471

GPU1 -> completed hgtxr_centerloss_finetune_lr1e5_gpu1
log  -> runs/_logs/centerloss_finetune_lr1e5_gpu1_20260611_010000.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-5_weakdistill_centerloss_finetune_fullwidth_20260610_234952
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt
gate -> early-stop epoch 9; best val center 39.9555, not promoted versus 39.8471

GPU0 -> completed hgtxr_trackonly_finetune_gpu0
log  -> runs/_logs/trackonly_finetune_gpu0_20260611_011000.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerloss_finetune_fullwidth_20260610_235630
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt
gate -> early-stop epoch 9; best val center 39.9155, not promoted versus 39.8471

GPU1 -> completed hgtxr_trackonly_finetune_lr5e6_gpu1
log  -> runs/_logs/trackonly_finetune_lr5e6_gpu1_20260611_012000.log
run  -> runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerloss_finetune_fullwidth_20260611_000139
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt
gate -> early-stop epoch 9; best val center 39.9662, not promoted

GPU0 -> completed hgtxr_trackonly_centerckptinit_gpu0
log  -> runs/_logs/trackonly_centerckptinit_gpu0_20260611_013000.log
run  -> runs/raw_mode1_stage2_count10000_lr1e-5_nodistill_trackonly_centerckptinit_fullwidth_20260611_000343
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707/train/best_metric_track_center_px.pt
gate -> early-stop epoch 8; best val center 39.7478, test center 37.2887, not final-promoted

GPU1 -> completed hgtxr_trackonly_centerckptinit_lr5e6_gpu1
log  -> runs/_logs/trackonly_centerckptinit_lr5e6_gpu1_20260611_014000.log
run  -> runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerckptinit_fullwidth_20260611_001034
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707/train/best_metric_track_center_px.pt
gate -> early-stop epoch 8; best val center 39.5315, best-center test center 37.0766, not final-promoted

interpolation -> completed
source A -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709/train/best_metric_track_center_px.pt
source B -> runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_trackonly_centerckptinit_fullwidth_20260611_001034/train/best_metric_track_center_px.pt
alpha 0.25 -> runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.25.pt; test center 36.7546
alpha 0.50 -> runs/interpolated_checkpoints/centerloss_bestcenter__trackonly_centerckptinit_lr5e6_bestcenter_alpha0.50.pt; test center 38.6547

GPU0 -> tmux hgtxr_centerloss_centerckptinit_nodistill_gpu0
log  -> runs/_logs/centerloss_centerckptinit_nodistill_gpu0_20260611_015000.log
run  -> runs/raw_mode1_stage2_count10000_lr5e-6_nodistill_centerloss_centerckptinit_fullwidth_20260611_004917
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707/train/best_metric_track_center_px.pt
gate -> stop after epoch 8 if best val center remains > 42.0

GPU1 -> tmux hgtxr_centerloss_centerckptinit_weakdistill_gpu1
log  -> runs/_logs/centerloss_centerckptinit_weakdistill_gpu1_20260611_015000.log
run  -> runs/raw_mode1_stage2_count10000_lr5e-6_weakdistill_centerloss_centerckptinit_fullwidth_20260611_004943
init -> runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707/train/best_metric_track_center_px.pt
gate -> stop after epoch 8 if best val center remains > 42.0
```

Do not promote pre-fix event-count runs. Before the event-builder fix,
event tensors were all zero after accumulation, so count sweeps changed
metadata/counts but not usable model input.

## Completed Runs

- `runs/raw_mode1_stage2_lr2e-4_nodistill_fullwidth_20260610_203608`: completed, early-stop epoch 13.
- `runs/raw_mode1_stage2_lr1e-4_weakdistill_fullwidth_20260610_204106`: completed, early-stop epoch 13; pre-fix reference only.
- `runs/raw_mode1_stage2_da_roi_lr2e-4_nodistill_fullwidth_20260610_204730`: sampled to epoch 7 and stopped manually because P10 remained low.
- `runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_fullwidth_20260610_210856`: completed, early-stop epoch 13; best val P10 `14.1543`.
- `runs/raw_mode1_stage2_count2500_lr1e-4_weakdistill_fullwidth_20260610_210855`: completed, early-stop epoch 13; best val P10 `13.6961`.
- `runs/raw_mode1_stage2_lr1e-4_weakdistill_fullwidth_20260610_212149`: completed, early-stop epoch 13; expected count `5000`; best val P10 `13.7298`, best center `44.5356`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_fullwidth_20260610_212150`: completed, early-stop epoch 13; best val P10 `13.8702`, best center `43.3363`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_213707`: completed epoch 30; best val P10 `13.8702`, best center `39.5347`.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_fullwidth_20260610_213709`: completed epoch 30; best val P10 `14.3430`, best center `39.8471`.
- `runs/eval_centerckpt_bestcenter_test_gpu0_retry_20260610_222803`: completed; test P10 `9.9209`, test center `36.9280`.
- `runs/eval_centerloss_bestp10_test_gpu0_retry_20260610_222812`: completed; test P10 `10.9864`, test center `42.3723`.
- `runs/eval_centerloss_bestcenter_test_gpu1_20260610_230859`: completed; test P10 `10.2759`, test center `36.5813`; current best test-center result.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerloss_lite_fullwidth_20260610_223243`: stopped after epoch 10 validation; best val center `44.2607`.
- `runs/raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerloss_lite_fullwidth_20260610_224120`: no active process now; best observed val center `43.3086`, not promoted.
- `runs/raw_mode1_stage2_count10000_lr5e-5_weakdistill_centerckpt_fullwidth_20260610_224355`: no active process now; best observed val center `44.5171`, not promoted.
- `runs/raw_mode1_stage2_count10000_lr1e-4_weakdistill_centerckpt_ema996_fullwidth_20260610_225302`: stopped as noncompetitive; epoch 8 best val center `44.6362`.
- `runs/raw_mode1_stage2_count1000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230041`: stopped as noncompetitive; best val center `44.9089`.
- `runs/raw_mode1_stage2_count5000_lr1e-4_weakdistill_centerckpt_fullwidth_20260610_230218`: stopped as noncompetitive; best val center `44.8672`.

## Event Builder Fix

Files:

```text
src/hbtxr/data/event_builder.py
tests/test_event_builder.py
scripts/external/check_event_count_override_sanity.py
```

Validation:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/test_event_builder.py \
  tests/test_raw_event_count_training_result.py \
  tests/test_raw_event_count_train_wrapper.py
```

## Stage1 Checkpoint

Default Stage1 checkpoint:

```text
runs/raw_mode1_stage1_event_count_20260610_192838/train/best_search_p10.pt
```

## Runtime Constraint

Set `HBTXR_DISABLE_CUDNN=1` on this host unless CUDA/cuDNN libraries are repaired. Current known failure without it: `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.

## 2026-06-16 PAPER_REF Second-Goal Continuation

Added PAPER_REF-specific experiment artifact:

```text
anlaysis/paper-ref/PAPER_REF_DETAILED_EXPERIMENT_MAP.md
```

Completed XR-01 count-branch training record:

```text
GPU0/tmux hgtxr_xr01_adamw250k_gpu0_20260616
log       runs/_logs/xr01_adamw_count250k_lr8e-6_gpu0_20260616.log
config    configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml
count     250000
optimizer AdamW
lr        8e-6

GPU1/tmux hgtxr_xr01_adamw260k_gpu1_20260616
log       runs/_logs/xr01_adamw_count260k_lr8e-6_gpu1_20260616.log
config    configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml
count     260000
optimizer AdamW
lr        8e-6
```

Prepared XR-03 geometry runner:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh 255000 0.025 0.01 6e-6 cuda:0
```

If XR-01 promotes, pass the new checkpoint explicitly:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh <count> 0.025 0.01 <lr> cuda:0 <new_best_center_checkpoint.pt>
```

XR-01 result scanner:

```bash
scripts/external/compare_xr01_adamw_bracket.py
```

Run XR-03 only after XR-01 finishes or a GPU is free. If XR-01 promotes, replace `count`, LR, and init checkpoint with the new best contract.

### 2026-06-16 XR-01 Count-Bracket Result And LR Bracket

XR-01 fixed-count AdamW bracket finished for `250000` and `260000` events at LR `8e-6`.

Leader gate:

```text
center < 27.089664377485004
or P10 > 15.710884816305978 with center <= 27.339664377485004
P5 guard reference = 4.630102171216692
```

Completed test results:

| case | checkpoint | center px | P10 % | P5 % | decision |
|---|---:|---:|---:|---:|---|
| fixed250k LR 8e-6 | best-center | 27.14032256262643 | 15.30442224230085 | 4.363520533697946 | no promotion |
| fixed250k LR 8e-6 | best-P10 | 28.13574755532401 | 14.668792915344238 | 4.3401361874171664 | no promotion |
| fixed260k LR 8e-6 | best-center | 27.208868653433665 | 15.642857592446463 | 4.374149778911046 | no promotion |
| fixed260k LR 8e-6 | best-P10 | 32.203995956693376 | 11.9897962978908 | 3.186224603652954 | no promotion |

Because neither count promoted, the next planned XR-01 branch is fixed255k LR bracket:

```text
tmux hgtxr_xr01_lr6e6_255k_gpu0_20260616
log  runs/_logs/xr01_adamw_count255k_lr6e-6_gpu0_20260616.log
cmd  bash scripts/external/run_optimizer_probe_queue_20260611.sh 255000 adamw 6e-6 cuda:0

tmux hgtxr_xr01_lr1e5_255k_gpu1_20260616
log  runs/_logs/xr01_adamw_count255k_lr1e-5_gpu1_20260616.log
cmd  bash scripts/external/run_optimizer_probe_queue_20260611.sh 255000 adamw 1e-5 cuda:1
```

The comparison helper now scans both the completed count bracket and fixed255k LR bracket:

```bash
scripts/external/compare_xr01_adamw_bracket.py
```

Fixed255k LR branch closeout:

| case | checkpoint | center px | P10 % | P5 % | decision |
|---|---:|---:|---:|---:|---|
| fixed255k LR 6e-6 | best-center | 28.610539082118443 | 14.511054842812674 | 4.037415075302124 | no promotion |
| fixed255k LR 6e-6 | best-P10 | 28.610539082118443 | 14.511054842812674 | 4.037415075302124 | no promotion |
| fixed255k LR 1e-5 | best-center | 26.174878706250873 | 16.502126346315656 | 5.099915143421718 | promoted; new primary leader |
| fixed255k LR 1e-5 | best-P10 | 26.50779542582376 | 16.803571953092302 | 4.259779044560024 | promoted; P10-specialized secondary |

New leader checkpoint:

```text
runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt
```

Next P0 run, using the promoted checkpoint:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.025 0.01 1e-5 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_centerloss_fixedcount_alphainit_fullwidth_20260616_005312/train/best_metric_track_center_px.pt
```

Launched in tmux on GPU0:

```text
tmux hgtxr_xr03_ellipsestate_lr1e5_gpu0_20260616
log  runs/_logs/xr03_ellipsestate_lr1e-5_gpu0_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016
```

Parallel stability branch launched in tmux on GPU1:

```text
tmux hgtxr_xr03_ellipsestate_lr6e6_gpu1_20260616
log  runs/_logs/xr03_ellipsestate_lr6e-6_gpu1_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012145
```

Startup verification: both branches passed the raw event-count contract, loaded the promoted checkpoint, and entered epoch 1/12 on their assigned GPU.

### 2026-06-16 XR-04 Failure-Bucket Diagnostic

Added CPU diagnostic helper:

```bash
scripts/external/summarize_eval_failure_buckets.py
```

Full leader diagnostic artifact:

```text
runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json
```

Command contract:

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_centerloss_adaptivecount_finetune_fullwidth.yaml \
  --mode mode1 \
  --stage stage2 \
  --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows runs/eval_fixed255k_adamw_lr8e_6_bestcenter_test_gpu0_w0_20260611_113913/eval/test/eval_rows.json \
  --top-k 12 \
  --output runs/diagnostics/xr04_failure_buckets_adamw255k_leader_test_20260616.json \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=false \
  --override training.num_workers=0
```

Initial finding:

- Low `similarity_target` dominates the hard cases: `similarity <=0.1` has weighted center `42.8642` and P10 `7.1259`.
- Worst subject buckets by weighted center are `subject_id:42` (`31.4714`) and `subject_id:41` (`30.9859`).
- Worst session bucket is `user41/right/session_201`, weighted center `42.4245`.
- Closed-eye and invalid-track rows are mostly zero-weighted by current metric, so they are diagnostic for failure context but not promotion metric drivers.

### 2026-06-16 XR-03 Closeout And XR-03A/B Queue

XR-03 decoded ellipse-state loss promoted the software leader.

| branch | checkpoint | center px | P10 % | P5 % | decision |
|---|---:|---:|---:|---:|---|
| LR 1e-5 axis 0.025 angle 0.01 | best-center | 21.682174137660436 | 21.820153658730643 | 7.044643061501639 | promoted; new primary leader |
| LR 1e-5 axis 0.025 angle 0.01 | best-P10 | 21.95296255179814 | 21.74532369886126 | 6.735119233812605 | promoted; secondary |
| LR 6e-6 axis 0.025 angle 0.01 | best-center | 22.879396969931467 | 20.383504002434865 | 5.965986571993146 | promoted over prior leader; weaker than LR 1e-5 |
| LR 6e-6 axis 0.025 angle 0.01 | best-P10 | 23.06402723789215 | 20.23724546432495 | 6.090986551557268 | promoted over prior leader; weaker than LR 1e-5 |

New primary leader checkpoint:

```text
runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016/train/best_metric_track_center_px.pt
```

Next queued refinement commands:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.05 0.02 1e-5 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016/train/best_metric_track_center_px.pt

bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.05 0.02 6e-6 cuda:1 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p025_angle0p01_adamwleaderinit_fullwidth_20260616_012016/train/best_metric_track_center_px.pt
```

Launched:

```text
GPU0 tmux hgtxr_xr03a_axis005_angle002_lr1e5_gpu0_20260616
log  runs/_logs/xr03a_axis0p05_angle0p02_lr1e-5_gpu0_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837

GPU1 tmux hgtxr_xr03b_axis005_angle002_lr6e6_gpu1_20260616
log  runs/_logs/xr03b_axis0p05_angle0p02_lr6e-6_gpu1_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014848
```

Startup verification: both branches passed the raw event-count contract, loaded the promoted XR-03 best-center checkpoint, resolved to the intended CUDA device, and entered epoch 1/12.

### 2026-06-16 XR-03A/B Closeout And Next Geometry Queue

XR-03A/B completed and strengthened the second-goal geometry branch.

| branch | checkpoint | center px | P10 % | P5 % | decision |
|---|---:|---:|---:|---:|---|
| XR-03A LR 1e-5 axis 0.05 angle 0.02 | best-center / best-P10 | 20.453865163666862 | 25.86862314088004 | 8.310374430247716 | promoted; new primary leader |
| XR-03B LR 6e-6 axis 0.05 angle 0.02 | best-center | 21.3059159551348 | 23.510204751150948 | 7.570153277260917 | promoted; secondary |
| XR-03B LR 6e-6 axis 0.05 angle 0.02 | best-P10 | 21.5007520709719 | 23.427296597617012 | 8.20748325756618 | promoted; secondary, weaker center/P10 than XR-03A |

New primary leader checkpoint:

```text
runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt
```

Next queued refinement commands:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.075 0.03 1e-5 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt

bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.075 0.03 6e-6 cuda:1 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt
```

Rationale: the axis/angle increase from `0.025/0.01` to `0.05/0.02` improved center, P10, and P5. The next bounded sweep tests whether the gain continues before moving to a direct ellipse auxiliary head or low-similarity-aware gating.

### 2026-06-16 XR-03C/D Launch

Launched bounded stronger-geometry sweep from XR-03A best-center.

```text
GPU0 tmux hgtxr_xr03c_axis0075_angle003_lr1e5_gpu0_20260616
log  runs/_logs/xr03c_axis0p075_angle0p03_lr1e-5_gpu0_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641

GPU1 tmux hgtxr_xr03d_axis0075_angle003_lr6e6_gpu1_20260616
log  runs/_logs/xr03d_axis0p075_angle0p03_lr6e-6_gpu1_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641
```

Launch commands:

```bash
bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.075 0.03 1e-5 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt

bash scripts/external/run_xr03_ellipsestate_adamw_leader_probe.sh \
  255000 0.075 0.03 6e-6 cuda:1 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt
```

Startup result:

- Raw event-count contract passed for both branches.
- Init checkpoint resolved to XR-03A best-center.
- XR-03C resolved `cuda:0`; XR-03D resolved `cuda:1`.
- Both entered epoch `1/12`.
- Promotion gate remains XR-03A: center `20.453865163666862`, P10 `25.86862314088004`, P5 `8.310374430247716`.

### 2026-06-16 XR-04 Low-Similarity Loss Candidate

Implemented next queued P0/P1 candidate from the XR-04 failure-bucket finding.

Changed files:

```text
src/hbtxr/loss/bundles/track.py
src/hbtxr/loss/stage2.py
tests/test_track_center_l2_loss.py
configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_ellipsestate_lowsim_finetune_fullwidth.yaml
scripts/external/run_xr04_lowsim_ellipsestate_probe.sh
```

Loss behavior:

```text
multiplier = 1 + scale * clamp((threshold - similarity_target) / threshold, 0, 1)
```

Default behavior is unchanged because `track_low_similarity_weight_scale` defaults to `0.0`. The candidate config uses threshold `0.3`, scale `1.0`, and max multiplier `2.0`.

Queued command template after XR-03C/D frees a GPU:

```bash
bash scripts/external/run_xr04_lowsim_ellipsestate_probe.sh \
  255000 0.05 0.02 0.3 1.0 6e-6 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_adamwleaderinit_fullwidth_20260616_014837/train/best_metric_track_center_px.pt
```

### 2026-06-16 XR-03C/D Closeout And XR-04 Launch

XR-03C/D completed train/eval.

```text
XR-03C LR 1e-5 best-center/best-P10:
  center 20.474989349501474
  P10    26.320153747286117
  P5     7.537415218353272

XR-03D LR 6e-6 best-center:
  center 20.49403738464628
  P10    25.965136766433716
  P5     8.044643136433193

XR-03D LR 6e-6 best-P10:
  center 20.433587510245186
  P10    25.78656539235796
  P5     7.749149915150234
```

Decision: XR-03D LR `6e-6`, axis/angle `0.075/0.03`, best-P10 checkpoint becomes the center-first leader. XR-03A remains the P10/P5-balanced secondary.

New center-first init checkpoint:

```text
runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641/train/best_track_p10.pt
```

Launched two XR-04 low-similarity-aware branches from that checkpoint:

```text
GPU0 tmux hgtxr_xr04_lowsim_t03_s10_lr6e6_gpu0_20260616
log  runs/_logs/xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_lr6e-6_gpu0_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p3_s1p0_fullwidth_20260616_023454

GPU1 tmux hgtxr_xr04_lowsim_t03_s10_lr1e5_gpu1_20260616
log  runs/_logs/xr04_lowsim_axis0p05_angle0p02_t0p3_s1p0_lr1e-5_gpu1_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr1e_5_nodistill_trackonly_ellipsestate_axis0p05_angle0p02_lowsim_t0p3_s1p0_fullwidth_20260616_023505
```

Both branches passed raw event-count contract and entered epoch `1/12`.

### 2026-06-16 XR-04 Closeout And XR-05A Launch

XR-04 low-similarity-aware loss completed train/eval on both GPUs.

Promotion gates:

```text
center-first: center < 20.433587510245186
balanced: P10 > 25.86862314088004 and P5 >= 8.310374430247716
```

Validated XR-04 test results:

```text
LR 1e-5 best-center: center 20.648689174652098, P10 25.145834023611886, P5 7.927296195711408
LR 1e-5 best-P10:    center 20.739680664879934, P10 25.440902028765,    P5 8.661990063531059
LR 6e-6 best-center: center 20.43466943332127,  P10 25.784439495631627, P5 8.428571728297642
LR 6e-6 best-P10:    center 20.639369245937893, P10 25.642432710102625, P5 7.786139719826835
```

Decision: no promotion. LR `6e-6` best-center is close on center but still worse by `0.001081923076083 px`, and its P10 does not beat the XR-03A balanced gate. The P5 gain is useful evidence, but not enough to replace either retained leader.

Implemented XR-05A direct track-state auxiliary head:

```text
src/hbtxr/models/heads.py
src/hbtxr/models/tracker/head_factory.py
src/hbtxr/models/tracker/track_branch.py
src/hbtxr/models/hybrid_tracker.py
src/hbtxr/training/model_factory.py
src/hbtxr/loss/bundles/__init__.py
src/hbtxr/loss/bundles/track.py
src/hbtxr/loss/stage2.py
tests/test_track_center_l2_loss.py
configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_finetune_fullwidth.yaml
scripts/external/run_xr05a_trackstateaux_probe.sh
```

Design: `track/fused -> track/state_aux` predicts absolute `(x, y, a, b, u, v)` as a training-only auxiliary target. The main inference path remains `track/pupil -> track/state`, so runtime contract is unchanged unless the new config opt-in is enabled.

Validation before launch:

```text
bash -n scripts/external/run_xr05a_trackstateaux_probe.sh
python3 -m py_compile src/hbtxr/models/heads.py src/hbtxr/models/tracker/head_factory.py src/hbtxr/models/tracker/track_branch.py src/hbtxr/models/hybrid_tracker.py src/hbtxr/training/model_factory.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/stage2.py
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py
# 8 passed; known parent .pytest_cache read-only warning only
.venv/bin/python scripts/external/check_raw_event_count_training_readiness.py --stage2-config configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_finetune_fullwidth.yaml
# ready=true, raw event-count contract ok
PYTHONPATH=src:scripts/external .venv/bin/python -c "... build_model(...) ..."
# TrackStateAuxHead, 150918 parameters
```

Launched XR-05A branches from the XR-03D center leader checkpoint:

```text
GPU0 tmux hgtxr_xr05a_trackstateaux_lr6e6_gpu0_20260616
log  runs/_logs/xr05a_trackstateaux_c0p001_axis0p025_angle0p01_lr6e-6_gpu0_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025651

GPU1 tmux hgtxr_xr05a_trackstateaux_lr3e6_gpu1_20260616
log  runs/_logs/xr05a_trackstateaux_c0p001_axis0p025_angle0p01_lr3e-6_gpu1_20260616.log
run  runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025701
```

Both XR-05A branches passed raw event-count contract, loaded the XR-03D checkpoint, resolved to the intended CUDA device, and entered epoch `1/12`.

### 2026-06-16 XR-05A Closeout And XR-05B/C Launch

XR-05A produced all four test summaries:

| Branch | Checkpoint | Test center | P10 | P5 | Decision |
|---|---|---:|---:|---:|---|
| GPU0 LR `6e-6` | best-center | 20.46567486694881 | 24.83716058731079 | 8.1322281564985 | no promotion |
| GPU0 LR `6e-6` | best-P10 | 20.316973662376405 | 25.863946315220424 | 8.239796202523367 | new center-first leader |
| GPU1 LR `3e-6` | best-center | 20.38325309753418 | 26.210459920338224 | 8.69557854788644 | P5/balanced secondary |
| GPU1 LR `3e-6` | best-P10 | 20.321967782293047 | 26.57610618046352 | 8.033588709150042 | P10 leader |

Promotion gates before XR-05A:

```text
center-first: center < 20.433587510245186
balanced: P10 > 25.86862314088004 and P5 >= 8.310374430247716
```

Decision: XR-05A promotes. Direct `track/state_aux` is the strongest accuracy axis so far under the 2차 목표 geometry branch. New active gates:

```text
center-first: center < 20.316973662376405
P10: P10 > 26.57610618046352
P5/balanced: P5 > 8.69557854788644 with non-degraded center/P10
```

Launched follow-up refinements:

```text
XR-05B GPU1
init runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025701/train/best_track_p10.pt
aux center/axis/angle 0.0005/0.0125/0.005
LR 2e-6
log runs/_logs/xr05b_trackstateaux_c0p0005_axis0p0125_angle0p005_lr2e-6_gpu1_20260616.log
run runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335

XR-05C GPU0
init runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641/train/best_track_p10.pt
aux center/axis/angle 0.001/0.025/0.01
LR 4.5e-6
log runs/_logs/xr05c_trackstateaux_c0p001_axis0p025_angle0p01_lr4p5e-6_gpu0_20260616.log
run runs/raw_mode1_stage2_count255000_adamw_lr4_5e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_031550
```

Both XR-05B and XR-05C passed the raw event-count contract, loaded the intended init checkpoints, resolved to the intended CUDA device, and entered epoch `1/12`.

### 2026-06-16 XR-05B/C Closeout And XR-05D/E Launch

XR-05B required manual test eval recovery because the runner stopped after train before appending eval output. First manual eval attempt failed with cuDNN sublibrary mismatch because `HBTXR_DISABLE_CUDNN=1` was missing. Re-running with `HBTXR_DISABLE_CUDNN=1` matched runner behavior and completed.

XR-05B and XR-05C test results:

| Branch | Checkpoint | Test center | P10 | P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-05B LR `2e-6`, aux `0.0005/0.0125/0.005` | best-center | 20.34794154507773 | 26.231718444824217 | 8.644558129991804 | no promotion |
| XR-05B LR `2e-6`, aux `0.0005/0.0125/0.005` | best-P10 | 20.29270099571773 | 26.04464359964643 | 8.07823155948094 | new center-first leader |
| XR-05C LR `4.5e-6`, aux `0.001/0.025/0.01` | best-center | 20.596219372749328 | 25.269133343015397 | 7.835034254619053 | no promotion |
| XR-05C LR `4.5e-6`, aux `0.001/0.025/0.01` | best-P10 | 20.32359070096697 | 26.16709260940552 | 7.850765575681414 | no promotion |

Updated active gates:

```text
center-first: center < 20.29270099571773
P10: P10 > 26.57610618046352
P5/balanced: P5 > 8.69557854788644 with non-degraded center/P10
```

Launched follow-up refinements:

```text
XR-05D GPU1
init runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335/train/best_track_p10.pt
aux center/axis/angle 0.00025/0.00625/0.0025
LR 1e-6
log runs/_logs/xr05d_trackstateaux_c0p00025_axis0p00625_angle0p0025_lr1e-6_gpu1_20260616.log
run runs/raw_mode1_stage2_count255000_adamw_lr1e_6_nodistill_trackonly_trackstateaux_c0p00025_axis0p00625_angle0p0025_fullwidth_20260616_033720

XR-05E GPU0
init runs/raw_mode1_stage2_count255000_adamw_lr3e_6_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_025701/train/best_metric_track_center_px.pt
aux center/axis/angle 0.00075/0.01875/0.0075
LR 1.5e-6
log runs/_logs/xr05e_trackstateaux_c0p00075_axis0p01875_angle0p0075_lr1p5e-6_gpu0_20260616.log
run runs/raw_mode1_stage2_count255000_adamw_lr1_5e_6_nodistill_trackonly_trackstateaux_c0p00075_axis0p01875_angle0p0075_fullwidth_20260616_033725
```

Both XR-05D and XR-05E passed raw event-count contract, loaded intended init checkpoints, resolved to intended CUDA devices, and entered epoch `1/12`.

### 2026-06-16 XR-05D/E Closeout And XR-07 Launch

XR-05D and XR-05E both stopped early at epoch 5/12. Final test evals:

| Experiment | Checkpoint | Center px | P10 % | P5 % | Decision |
|---|---:|---:|---:|---:|---|
| XR-05D LR `1e-6`, aux `0.00025/0.00625/0.0025` | best-center | 20.329813245364598 | 26.293368080684118 | 8.298044504438128 | no promotion |
| XR-05D LR `1e-6`, aux `0.00025/0.00625/0.0025` | best-P10 | 20.29814441204071 | 25.887330681937083 | 7.81760230745588 | no promotion |
| XR-05E LR `1.5e-6`, aux `0.00075/0.01875/0.0075` | best-center | 20.36729155949184 | 26.653912333079745 | 8.482993486949375 | new P10 leader |
| XR-05E LR `1.5e-6`, aux `0.00075/0.01875/0.0075` | best-P10 | 20.307639326368058 | 25.878827265330724 | 8.103741775240216 | no promotion |

Updated gates:

- Center-first: center `< 20.29270099571773` from XR-05B best-P10.
- P10: P10 `> 26.653912333079745` from XR-05E best-center.
- P5/balanced: P5 `> 8.69557854788644` with non-degraded center/P10 from XR-05A best-center.

Prepared fallback:

```text
configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_finetune_fullwidth.yaml
scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh
```

XR-07 tiny center-hinge recovery launched after XR-02 dense trajectories remained unavailable:

```text
XR-07A GPU1:
bash scripts/external/run_xr07_trackstateaux_centerhinge_probe.sh 255000 linear 10.0 0.01 1e-6 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335/train/best_track_p10.pt

XR-07B GPU0:
bash scripts/external/run_xr07_trackstateaux_centerhinge_probe.sh 255000 squared 10.0 0.001 1e-6 cuda:0 runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335/train/best_track_p10.pt
```

Startup validation: both logs show raw event-count contract pass, XR-05B best-P10 checkpoint load, intended CUDA device resolution, and epoch `1/12` train entry.

### 2026-06-16 XR-07 Closeout And XR-02 Dense-Gap Check

XR-07A/B both early-stopped at epoch 5/12. Validation results were clearly below the active gates:

| Experiment | Best val center | Best val P10 | Best val P5 | Decision |
|---|---:|---:|---:|---|
| XR-07A linear hinge `m10/w0.01`, LR `1e-6` | 23.626078749602694 | 21.898023947229927 | 7.794250002447164 | no promotion |
| XR-07B squared hinge `m10/w0.001`, LR `1e-6` | 23.628568100479413 | 21.898023947229927 | 7.794250002447164 | no promotion |

Test eval note:

- Runner-created eval directories:
  - `runs/eval_fixed255k_xr07_trackstateaux_linearhinge_m10p0_w0p01_adamw_lr1e_6_bestcenter_test_gpu1_w0_20260616_040106`
  - `runs/eval_fixed255k_xr07_trackstateaux_squaredhinge_m10p0_w0p001_adamw_lr1e_6_bestcenter_test_gpu0_w0_20260616_040108`
- Those directories contain run-contract/hypers files but no `eval/test/eval_summary.json`.
- A manual full eval of XR-07A best-center was interrupted after extended runtime with no output. Because validation is already far worse than active gates, XR-07 is closed as no-promotion rather than spending more GPU time on full test scoring.

XR-02 dense trajectory check:

```json
{
  "rows": 2238,
  "groups": 72,
  "group_len_median": 27.0,
  "gap_min_us": 360000,
  "gap_median_us": 4000003.0,
  "gap_max_us": 24000019
}
```

Decision: current test manifest is sparse. EyeLoRiN-style M2F/OFE should stay gated until dense/continuous trajectories are generated.

XR-06 weak-distill fallback launched after XR-07 closeout:

```text
tmux session: hgtxr_xr06_weakdistill_centerleader_gpu1_20260616
log: runs/_logs/xr06_weakdistill_centerleader_lr1e-6_gpu1_20260616.log
command:
bash scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh 255000 0.0005 0.0125 0.005 1e-6 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335/train/best_track_p10.pt runs/raw_mode1_stage2_count255000_adamw_lr2e_6_nodistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_031335/train/best_track_p10.pt
```

XR-06B complementary P10-leader weak-distill branch launched on GPU0 after Nash GPT5.5 sidecar recommendation:

```text
tmux session: hgtxr_xr06b_weakdistill_p10leader_gpu0_20260616
log: runs/_logs/xr06b_weakdistill_p10leader_lr1e-6_gpu0_20260616.log
command:
bash scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh 255000 0.00075 0.01875 0.0075 1e-6 cuda:0 runs/raw_mode1_stage2_count255000_adamw_lr1_5e_6_nodistill_trackonly_trackstateaux_c0p00075_axis0p01875_angle0p0075_fullwidth_20260616_033725/train/best_metric_track_center_px.pt runs/raw_mode1_stage2_count255000_adamw_lr1_5e_6_nodistill_trackonly_trackstateaux_c0p00075_axis0p01875_angle0p0075_fullwidth_20260616_033725/train/best_metric_track_center_px.pt
```

Gate: promote only if center `< 20.29270099571773`, P10 `> 26.653912333079745`, or P5 `> 8.69557854788644` with acceptable center/P10/P5 tradeoff.

### 2026-06-16 XR-06A Closeout, XR-06B Eval, And XR-05F Control Launch

XR-06A completed training and both test evals:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-06A weak-distill from XR-05B | best-center | 20.283847980839866 | 26.277211591175625 | 8.472364248548235 | promotes center |
| XR-06A weak-distill from XR-05B | best-P10 | 20.368657435689652 | 26.00425246102469 | 8.164115946633475 | no promotion |

Updated gates:

- Center-first: center `< 20.283847980839866` from XR-06A best-center.
- P10: P10 `> 26.653912333079745` from XR-05E best-center.
- P5/balanced: P5 `> 8.69557854788644` from XR-05A best-center.

XR-06B closeout:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-06B weak-distill from XR-05E | best-center | 20.295043339048114 | 26.217687790734427 | 8.519132954733712 | no promotion |
| XR-06B weak-distill from XR-05E | best-P10 | 20.46945102555411 | 25.931973491396224 | 8.630952685219901 | no promotion |

XR-05F no-distill control launched on GPU1:

```text
tmux session: hgtxr_xr05f_nodistill_xr05e_lightaux_gpu1_20260616
log: runs/_logs/xr05f_nodistill_xr05e_lightaux_lr1e-6_gpu1_20260616.log
command:
bash scripts/external/run_xr05a_trackstateaux_probe.sh 255000 0.0005 0.0125 0.005 1e-6 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr1_5e_6_nodistill_trackonly_trackstateaux_c0p00075_axis0p01875_angle0p0075_fullwidth_20260616_033725/train/best_metric_track_center_px.pt
```

Purpose: isolate weak-distillation effect. If XR-05F matches or beats XR-06A, the gain is likely low-LR lighter-aux polish rather than teacher-student transfer.

XR-05F closeout:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-05F no-distill from XR-05E | best-center | 20.356191604478017 | 26.432823869160245 | 8.298044504438128 | no promotion |
| XR-05F no-distill from XR-05E | best-P10 | 20.31945925780705 | 26.150936106273107 | 8.142857428959438 | no promotion |

XR-06C/XR-05G launched from the XR-06A center leader:

```text
XR-06C GPU0 weak-distill:
bash scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh 255000 0.0005 0.0125 0.005 5e-7 cuda:0 runs/raw_mode1_stage2_count255000_adamw_lr1e_6_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_041024/train/best_metric_track_center_px.pt runs/raw_mode1_stage2_count255000_adamw_lr1e_6_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_041024/train/best_metric_track_center_px.pt

XR-05G GPU1 no-distill:
bash scripts/external/run_xr05a_trackstateaux_probe.sh 255000 0.0005 0.0125 0.005 5e-7 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr1e_6_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_041024/train/best_metric_track_center_px.pt
```

Logs:

- `runs/_logs/xr06c_weakdistill_xr06a_lr5e-7_gpu0_20260616.log`
- `runs/_logs/xr05g_nodistill_xr06a_lr5e-7_gpu1_20260616.log`

### 2026-06-16 XR-06C/XR-05G Closeout And XR-06D/XR-06E Launch

XR-05G completed with no promotion:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-05G no-distill from XR-06A | best-center | 20.291977088791985 | 26.442177615846905 | 8.456632941109794 | no promotion |
| XR-05G no-distill from XR-06A | best-P10 | 20.375956610270908 | 26.27423542567662 | 8.281462873731341 | no promotion |

XR-06C completed and promoted center and P10:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-06C weak-distill from XR-06A | best-center | 20.283125744547164 | 26.363521112714494 | 8.43664994921003 | promotes center |
| XR-06C weak-distill from XR-06A | best-P10 | 20.382882516724724 | 26.74489871433803 | 8.526786014011927 | promotes P10 |

Updated gates:

- Center-first: center `< 20.283125744547164` from XR-06C best-center.
- P10: P10 `> 26.74489871433803` from XR-06C best-P10.
- P5/balanced: P5 `> 8.69557854788644` from XR-05A best-center.

XR-06D/XR-06E follow-up launched from the promoted XR-06C leaders:

```text
XR-06D GPU0 no-distill center-preserve:
bash scripts/external/run_xr05a_trackstateaux_probe.sh 255000 0.0005 0.0125 0.005 2.5e-7 cuda:0 runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt

XR-06E GPU1 weak-distill P10-preserve:
bash scripts/external/run_xr06_weakdistill_trackstateaux_probe.sh 255000 0.0005 0.0125 0.005 2.5e-7 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt
```

Logs:

- `runs/_logs/xr06d_nodistill_xr06c_center_lr2p5e-7_gpu0_20260616.log`
- `runs/_logs/xr06e_weakdistill_xr06c_p10_lr2p5e-7_gpu1_20260616.log`

### 2026-06-16 XR-06D/XR-06E Closeout

Neither XR-06D nor XR-06E promoted a gate:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-06D no-distill center-preserve | best-center | 20.295614736420767 | 26.158589158739364 | 8.504677152633667 | no promotion |
| XR-06D no-distill center-preserve | best-P10 | 20.295614736420767 | 26.158589158739364 | 8.504677152633667 | no promotion |
| XR-06E weak-distill P10-preserve | best-center | 20.29422003201076 | 26.280612965992518 | 8.538690771375384 | no promotion |
| XR-06E weak-distill P10-preserve | best-P10 | 20.31491228171757 | 26.13392930030823 | 8.681122759410313 | no promotion |

Current gates remain:

- Center-first: center `< 20.283125744547164`.
- P10: P10 `> 26.74489871433803`.
- P5/balanced: P5 `> 8.69557854788644`.

### 2026-06-16 XR-08 Checkpoint Interpolation Smoke

Added:

- `scripts/external/interpolate_hbtxr_checkpoints.py`
- `scripts/external/run_xr08_checkpoint_interp_eval.sh`

Smoke result:

| Experiment | Alpha | Test center | Test P10 | Test P5 | Decision |
|---|---:|---:|---:|---:|---|
| XR-08 XR-06C best-center to best-P10 interpolation | 0.50 | 20.32587662594659 | 26.49830005509513 | 8.386479888643537 | no promotion |

Evidence:

- Checkpoint: `runs/interpolated_checkpoints/xr08_xr06c_center_p10_alpha0p50r.pt`
- Eval summary: `runs/eval_fixed255k_xr08_xr06c_center_p10_interp_alpha0p50r_gpu0_w0_20260616_053156/eval/test/eval_summary.json`
- Log: `runs/_logs/xr08_interp_alpha0p50r_gpu0_20260616_053155.log`

Decision: keep XR-08 tooling, but primary queue moves to XR-05I P5-preserve direct-aux fallback because interpolation alpha `0.50` degraded all active gates.

### 2026-06-16 XR-05I/XR-06F P5-Preserve Launch

Active runs:

| Experiment | Device | Init / Teacher | Aux center/axis/angle | LR | Status |
|---|---|---|---:|---:|---|
| XR-05I no-distill P5-preserve | GPU0 | XR-05A P5/balanced best-center | `0.001/0.025/0.01` | `5e-7` | training active |
| XR-06F weak-distill P5-preserve | GPU1 | XR-05A P5/balanced best-center / same teacher | `0.001/0.025/0.01` | `5e-7` | training active |

Evidence:

- XR-05I run: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_053814`
- XR-05I log: `runs/_logs/xr05i_p5preserve_nodistill_xr05a_lr5e-7_gpu0_20260616.log`
- XR-06F run: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p001_axis0p025_angle0p01_fullwidth_20260616_053815`
- XR-06F log: `runs/_logs/xr06f_p5preserve_weakdistill_xr05a_lr5e-7_gpu1_20260616.log`

Both logs show raw event-count contract pass, intended checkpoint load, intended CUDA device resolution, and epoch `1/12` entry. Do not relaunch identical commands while these runs are active.

### 2026-06-16 XR-05I/XR-06F Closeout

Both P5-preserve fallback branches completed and failed the active gates:

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-05I no-distill P5-preserve | best-center | 20.351013261931282 | 26.15008576256888 | 8.428996889931815 | no promotion |
| XR-05I no-distill P5-preserve | best-P10 | 20.393892083849227 | 26.170068747656686 | 8.450255387169975 | no promotion |
| XR-06F weak-distill P5-preserve | best-center | 20.322171998023986 | 26.200681025641305 | 8.625850643430438 | no promotion |
| XR-06F weak-distill P5-preserve | best-P10 | 20.421796573911394 | 26.394558545521328 | 8.630102334703718 | no promotion |

Evidence:

- XR-05I best-center summary: `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_054521/eval/test/eval_summary.json`
- XR-05I best-P10 summary: `runs/eval_fixed255k_xr05a_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_054833/eval/test/eval_summary.json`
- XR-06F best-center summary: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_055145/eval/test/eval_summary.json`
- XR-06F best-P10 summary: `runs/eval_fixed255k_xr06_weakdistill_trackstateaux_c0p001_axis0p025_angle0p01_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_055409/eval/test/eval_summary.json`

Decision: do not relaunch the same P5-preserve low-LR surface. Move next execution priority to a targeted XR-04 failure-bucket branch using the existing low-similarity diagnostic, or run XR-02 only when dense trajectories are available.

### 2026-06-16 XR-04B/XR-04C Targeted Failure-Bucket Launch

Launched two focused low-similarity branches from the XR-03D center-first checkpoint:

| Experiment | Device | Init checkpoint | Threshold | Scale | LR | Status |
|---|---|---|---:|---:|---:|---|
| XR-04B focused low-sim | GPU0 | XR-03D best-P10 | `0.1` | `2.0` | `6e-6` | completed |
| XR-04C focused low-sim conservative LR | GPU1 | XR-03D best-P10 | `0.1` | `2.0` | `3e-6` | completed |

Commands:

```text
bash scripts/external/run_xr04_lowsim_ellipsestate_probe.sh 255000 0.05 0.02 0.1 2.0 6e-6 cuda:0 runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641/train/best_track_p10.pt

bash scripts/external/run_xr04_lowsim_ellipsestate_probe.sh 255000 0.05 0.02 0.1 2.0 3e-6 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr6e_6_nodistill_trackonly_ellipsestate_axis0p075_angle0p03_adamwleaderinit_fullwidth_20260616_021641/train/best_track_p10.pt
```

Logs:

- `runs/_logs/xr04b_lowsim_t0p1_s2_lr6e-6_gpu0_20260616.log`
- `runs/_logs/xr04c_lowsim_t0p1_s2_lr3e-6_gpu1_20260616.log`

Both startup logs show raw event-count contract pass, intended XR-03D checkpoint load, intended CUDA device resolution, and epoch `1/12` entry.

### 2026-06-16 XR-04B/XR-04C Targeted Failure-Bucket Closeout

Both focused low-similarity branches early-stopped at epoch `8/12` and completed best-center/best-P10 test eval.

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-04B threshold `0.1`, scale `2.0`, LR `6e-6` | best-center | 20.484137114456722 | 25.600340850012643 | 8.277636350904192 | no promotion |
| XR-04B threshold `0.1`, scale `2.0`, LR `6e-6` | best-P10 | 20.67380678653717 | 25.401361295155116 | 7.875425440924508 | no promotion |
| XR-04C threshold `0.1`, scale `2.0`, LR `3e-6` | best-center | 20.588273804528374 | 25.658163949421475 | 7.849915211541312 | no promotion |
| XR-04C threshold `0.1`, scale `2.0`, LR `3e-6` | best-P10 | 20.588273804528374 | 25.658163949421475 | 7.849915211541312 | no promotion |

Eval summaries:

- `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_061317/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_061619/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_061333/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr04_lowsim_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_061637/eval/test/eval_summary.json`

Gate comparison:

- Center gate remains `<20.283125744547164`.
- P10 gate remains `>26.74489871433803`.
- P5 gate remains `>8.69557854788644`.

Decision: no promotion. Reweighted low-similarity loss did not improve aggregate test metrics when focused to `similarity_target <= 0.1`. Next failure-bucket experiment should use manifest/sampler-level targeting around the diagnostic bucket (`similarity_target <= 0.1`, `session_201`) or move to XR-02 after dense trajectories are generated.

### 2026-06-16 XR-04D/XR-04E Failure-Bucket Manifest-Subset Launch

Added reproducible subset tooling:

- `scripts/external/build_failure_bucket_manifest.py`
- `scripts/external/run_xr04d_failbucket_subset_probe.sh`

Subset:

- Source: `data/_internal/manifests/manifest1`
- Output: `data/_internal/manifests/manifest1/focus_sim0p1_session_201`
- Filter: `similarity_target <= 0.1` and `session_key` contains `session_201`
- Counts: train `1186/5929`, val `182/844`

Launched two subset fine-tunes from XR-03D best-P10:

| Experiment | Device | Train/val subset | Threshold | Scale | LR | Status |
|---|---|---|---:|---:|---:|---|
| XR-04D fail-bucket subset | GPU0 | sim<=0.1 + session_201 | `0.1` | `2.0` | `6e-6` | training active |
| XR-04E fail-bucket subset conservative LR | GPU1 | sim<=0.1 + session_201 | `0.1` | `2.0` | `3e-6` | training active |

Logs:

- `runs/_logs/xr04d_failbucket_sim01_s201_lr6e-6_gpu0_20260616.log`
- `runs/_logs/xr04e_failbucket_sim01_s201_lr3e-6_gpu1_20260616.log`

Startup validation: both logs show subset manifest creation, raw event-count contract pass, intended subset manifest paths, XR-03D best-P10 checkpoint load, intended CUDA device resolution, and epoch `1/12` entry.

### 2026-06-16 XR-04D/XR-04E Failure-Bucket Manifest-Subset Closeout

Both subset branches early-stopped at epoch `10/12` and completed best-center/best-P10 full-test evaluation.

| Experiment | Checkpoint | Test center | Test P10 | Test P5 | Decision |
|---|---|---:|---:|---:|---|
| XR-04D subset, LR `6e-6` | best-center | 24.50812735216958 | 16.752551589693343 | 4.517857306344169 | no promotion |
| XR-04D subset, LR `6e-6` | best-P10 | 25.703870964050292 | 15.307398448671613 | 4.238095378875732 | no promotion |
| XR-04E subset, LR `3e-6` | best-center | 23.55767515386854 | 18.159439352580478 | 5.106292690549578 | no promotion |
| XR-04E subset, LR `3e-6` | best-P10 | 23.42795329945428 | 18.25552776881627 | 5.031037589481898 | no promotion |

Eval summaries:

- `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestcenter_test_gpu0_w0_20260616_063110/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr6e_6_bestp10_test_gpu0_w0_20260616_063417/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestcenter_test_gpu1_w0_20260616_063111/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr04d_failbucket_sim0p1_session_201_axis0p05_angle0p02_t0p1_s2p0_adamw_lr3e_6_bestp10_test_gpu1_w0_20260616_063417/eval/test/eval_summary.json`

Gate comparison:

- Center gate remains `<20.283125744547164`.
- P10 gate remains `>26.74489871433803`.
- P5 gate remains `>8.69557854788644`.

Decision: no promotion. Hard-subset fine-tuning on `similarity_target <= 0.1` + `session_201` is too narrow for full-test accuracy. Do not repeat hard-subset training; next robustness path should be full-manifest weighted sampling/loss balancing, or XR-02 after dense trajectories are available.

### 2026-06-16 XR-09 Full-Manifest Weighted-Sampler Setup

Added opt-in train-only weighted sampling:

- `src/hbtxr/data/loader.py`
- `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_weightedsampler_fullwidth.yaml`
- `scripts/external/run_xr09_weightedsampler_trackstateaux_probe.sh`
- `tests/test_weighted_sampler.py`

Weight sanity on `manifest1/train_manifest.jsonl` for low-sim `3x`, session_201 `2x`, cap `6x`:

- rows `5929`
- weight `6x`: `1186`
- weight `3x`: `438`
- weight `2x`: `1686`
- weight `1x`: `2619`
- mean weight `2.4322820037105752`

Planned parallel launch:

| Experiment | Device | Init checkpoint | Low-sim multiplier | Session multiplier | Cap | LR |
|---|---|---|---:|---:|---:|---:|
| XR-09A weighted sampler center-preserve | GPU0 | XR-06C best-center | 3.0 | 2.0 | 6.0 | 5e-7 |
| XR-09B weighted sampler P10-preserve | GPU1 | XR-06C best-P10 | 2.0 | 2.0 | 4.0 | 5e-7 |

Launched:

- XR-09A tmux `hgtxr_xr09a_weightedsampler_center_gpu0_20260616`, log `runs/_logs/xr09a_weightedsampler_center_lr5e-7_gpu0_20260616.log`.
- XR-09B tmux `hgtxr_xr09b_weightedsampler_p10_gpu1_20260616`, log `runs/_logs/xr09b_weightedsampler_p10_lr5e-7_gpu1_20260616.log`.

Startup validation: both logs show raw event-count contract pass, intended checkpoint load, intended CUDA device resolution, full train/val manifest counts `5929/844`, and epoch `1/12` entry.

Closeout:

| Experiment | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| XR-09A weighted sampler center-preserve | best-center | 20.579283670016697 | 25.632653781345912 | 8.010204356057303 | no |
| XR-09A weighted sampler center-preserve | best-P10 | 20.641361141204833 | 24.742772783551896 | 8.041241747992379 | no |
| XR-09B weighted sampler P10-preserve | best-center | 20.415359340395245 | 26.196429313932146 | 8.305697563716343 | no |
| XR-09B weighted sampler P10-preserve | best-P10 | 20.55930529321943 | 25.68239870071411 | 7.932823378699166 | no |

Decision: XR-09 is closed without promotion. Keeping the full manifest was much safer than XR-04D/E hard-subset training, but sampler-only oversampling still degraded relative to XR-06C leaders. Next experiment should add loss-side sample weighting or move to dense-trajectory/XR-02 preparation rather than repeating sampler-only variants.

### 2026-06-16 XR-10 Full-Manifest Loss-Weight Setup

Added opt-in Stage2 track-side sample weighting:

- `src/hbtxr/data/components.py`: preserves `meta.session_key` in batches.
- `src/hbtxr/loss/stage_common.py`: adds `build_loss_sample_weights()`.
- `src/hbtxr/loss/stage2.py`: applies `loss.track_sample_weight` to track, track-state-aux, consistency, and constraint track weights.
- `tests/test_loss_sample_weighting.py`
- `configs/external/mode1_stage2_raw_event_count_lr5e-6_nodistill_trackonly_trackstateaux_lossweight_fullwidth.yaml`
- `scripts/external/run_xr10_lossweight_trackstateaux_probe.sh`

Weight sanity on `manifest1/train_manifest.jsonl` for low-sim `2x`, session_201 `1.5x`, cap `3x`:

- rows `5929`
- weight `3x`: `1186`
- weight `2x`: `438`
- weight `1.5x`: `1686`
- weight `1x`: `2619`
- mean weight `1.6161241356046552`

Planned parallel launch:

| Experiment | Device | Init checkpoint | Low-sim multiplier | Session multiplier | Cap | LR |
|---|---|---|---:|---:|---:|---:|
| XR-10A loss-weight center-preserve | GPU0 | XR-06C best-center | 2.0 | 1.5 | 3.0 | 5e-7 |
| XR-10B loss-weight P10-preserve | GPU1 | XR-06C best-P10 | 1.5 | 1.5 | 2.5 | 5e-7 |

Launched:

- XR-10A tmux `hgtxr_xr10a_lossweight_center_gpu0_20260616`, log `runs/_logs/xr10a_lossweight_center_lr5e-7_gpu0_20260616.log`.
- XR-10B tmux `hgtxr_xr10b_lossweight_p10_gpu1_20260616`, log `runs/_logs/xr10b_lossweight_p10_lr5e-7_gpu1_20260616.log`.

Startup validation: both logs show raw event-count contract pass, intended XR-06C checkpoint load, intended CUDA device resolution, full train/val manifest counts `5929/844`, and epoch `1/12` entry.

Closeout:

| Experiment | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| XR-10A loss-weight center-preserve | best-center | 20.34876068319593 | 26.20620824268886 | 8.127976478849138 | no |
| XR-10A loss-weight center-preserve | best-P10 | 20.429714499201094 | 26.048044974463327 | 7.999575104032244 | no |
| XR-10B loss-weight P10-preserve | best-center | 20.3265901020595 | 25.97023880141122 | 8.133503682272774 | no |
| XR-10B loss-weight P10-preserve | best-P10 | 20.38969965662275 | 26.1870755808694 | 7.99447306905474 | no |

Decision: XR-10 is closed without promotion. Loss-side weighting is less harmful than XR-09A but still below XR-06C gates. Failure-bucket rebalancing is now exhausted as an accuracy path; next work should change model/head representation or prepare dense trajectories for XR-02.
## 2026-06-16 XR-11 SimDR Execution Note

- Selected next P0 after XR-10 no-promotion: track-branch SimDR-style coordinate auxiliary.
- Added `TrackStateSimDRHead` on `track/fused`; model output is `track/state_simdr`.
- Added Stage2 loss `loss_track_state_simdr`; target is `cur_state[:2]` mapped from 0..255 image coordinates into configurable SimDR bins.
- Default setup: bins `64`, sigma `1.5`, weight `0.0005`, AdamW LR `5e-7`, fixed event count `255000`, existing track-state aux weights `0.0005/0.0125/0.005`.
- Runtime/inference path remains unchanged: `track/pupil -> track/state`.
- Runner prepared: `scripts/external/run_xr11_trackstate_simdr_probe.sh`.
- Planned lanes:
  - GPU0 center lane from XR-06C best-center.
  - GPU1 P10 lane from XR-06C best-P10 via `XR11_INIT_CKPT`.
- Launch status:
  - GPU0 center lane launched in tmux `hgtxr_xr11_simdr_center_gpu0_20260616`; log `runs/_logs/xr11_simdr_center_lr5e-7_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_simdr0p0005_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_073321`.
  - GPU1 P10 lane launched in tmux `hgtxr_xr11_simdr_p10_gpu1_20260616`; log `runs/_logs/xr11_simdr_p10_lr5e-7_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_nodistill_trackonly_trackstateaux_simdr0p0005_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_073333`.

Closeout:

| Lane | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| Center-init | best-center | 20.338836230550495 | 26.16071500778198 | 8.301445865631104 | no |
| Center-init | best-P10 | 20.366218400001525 | 26.545493936538698 | 8.573979895455496 | no |
| P10-init | best-center | 20.34303183896201 | 25.954507521220616 | 8.235544497626169 | no |
| P10-init | best-P10 | 20.289304559571402 | 26.325255816323416 | 8.309949268613543 | no |

Decision: XR-11 is closed without promotion. The nearest center result is P10-init best-P10 at `20.2893`, but it remains worse than the active center gate `20.283125744547164` and does not improve P10/P5. Next execution path is XR-12: weak-distill plus lighter SimDR weight, preserving the XR-06C teacher/init contract.

## 2026-06-16 XR-12 Weak-Distill Light-SimDR

Created execution assets:

- `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_fullwidth.yaml`
- `scripts/external/run_xr12_weakdistill_simdr_probe.sh`

Default hyperparameters:

| Parameter | Value |
|---|---:|
| Fixed event count | 255000 |
| Optimizer | AdamW |
| LR | 5e-7 |
| SimDR weight | 0.0001 |
| SimDR bins | 64 |
| SimDR sigma | 1.5 |
| Aux center/axis/angle | 0.0005 / 0.0125 / 0.005 |
| Distillation | weak teacher-student, teacher = init checkpoint per lane |

Launch:

- GPU0 center lane: tmux `hgtxr_xr12_wdsimdr_center_gpu0_20260616`, log `runs/_logs/xr12_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log`, init/teacher `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt`.
- GPU1 P10 lane: tmux `hgtxr_xr12_wdsimdr_p10_gpu1_20260616`, log `runs/_logs/xr12_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log`, init/teacher `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`.

Startup validation: both logs show raw event-count contract pass, intended init/teacher checkpoint load, intended CUDA device, full train/val counts `5929/844`, and epoch `1/12` entry.

Closeout:

| Lane | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| Center-init weak-distill | best-center | 20.30488075869424 | 26.15136126109532 | 8.22066354070391 | no |
| Center-init weak-distill | best-P10 | 20.359868260792325 | 26.082058572769164 | 8.791666991370064 | P5-only |
| P10-init weak-distill | best-center | 20.306958419936045 | 26.15136125428336 | 8.169643129621234 | no |
| P10-init weak-distill | best-P10 | 20.306958419936045 | 26.15136125428336 | 8.169643129621234 | no |

Evaluation artifacts:

- `runs/eval_fixed255k_xr12_weakdistill_simdr0p0001_b64_s1p5_center_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_080246/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr12_weakdistill_simdr0p0001_b64_s1p5_center_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_080552/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr12_weakdistill_simdr0p0001_b64_s1p5_p10_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_080258/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr12_weakdistill_simdr0p0001_b64_s1p5_p10_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_080603/eval/test/eval_summary.json`

Decision: XR-12 is closed without center/P10 promotion. The only promoted scalar is P5 on the center-init best-P10 checkpoint, but that checkpoint degrades center to `20.3599` and P10 to `26.0821`, so it is not selected as a new leader. Next execution path is XR-13: repeat weak-distill light-SimDR with a trainable filter limited to prediction heads to reduce representation drift.

## 2026-06-16 XR-13 Head-Only Weak-Distill SimDR

Created execution assets:

- `configs/external/mode1_stage2_raw_event_count_lr5e-6_weakdistill_trackonly_trackstateaux_simdr_headonly_fullwidth.yaml`
- `scripts/external/run_xr13_headonly_weakdistill_simdr_probe.sh`

Default hyperparameters:

| Parameter | Value |
|---|---:|
| Fixed event count | 255000 |
| Optimizer | AdamW |
| LR | 5e-7 |
| SimDR weight | 0.0001 |
| SimDR bins | 64 |
| SimDR sigma | 1.5 |
| Aux center/axis/angle | 0.0005 / 0.0125 / 0.005 |
| Distillation | weak teacher-student, teacher = init checkpoint per lane |
| Trainable scope | `track_head.*`, `track_state_aux_head.*`, `track_state_simdr_head.*` |

Prepared launch plan:

- GPU0 center lane: init/teacher `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt`.
- GPU1 P10 lane: init/teacher `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt`.

Validation before launch: runner bash syntax passed; XR-06C checkpoints exist; config/model smoke matched `18` trainable head tensors and `500494/3505306` trainable params.

Launch:

- GPU0 center lane: tmux `hgtxr_xr13_headonly_center_gpu0_20260616`, log `runs/_logs/xr13_headonly_wdsimdr_center_w0p0001_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr_headonly0p0001_b64_s1p5_center_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_081531`.
- GPU1 P10 lane: tmux `hgtxr_xr13_headonly_p10_gpu1_20260616`, log `runs/_logs/xr13_headonly_wdsimdr_p10_w0p0001_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_simdr_headonly0p0001_b64_s1p5_p10_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_081528`.

Startup validation: both logs show raw event-count contract pass, intended XR-06C init/teacher checkpoint per lane, trainable-filter `18` tensors and `500494/3505306` trainable params, intended CUDA device, full train/val counts `5929/844`, and epoch `1/12` entry.

Closeout:

| Lane | Checkpoint | Test center px | Test P10 % | Test P5 % | Promotion |
|---|---|---:|---:|---:|---|
| Center-init head-only weak-distill | best-center | 20.29158056122916 | 26.29761974470956 | 8.469388048989432 | no |
| Center-init head-only weak-distill | best-P10 | 20.28970977578844 | 26.26785784448896 | 8.520408460072108 | no |
| P10-init head-only weak-distill | best-center | 20.35549293586186 | 26.46216060093471 | 8.49064655985151 | no |
| P10-init head-only weak-distill | best-P10 | 20.36963668550764 | 26.551446315220424 | 8.533163567951748 | no |

Evaluation artifacts:

- `runs/eval_fixed255k_xr13_headonly_weakdistill_simdr0p0001_b64_s1p5_center_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_082239/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr13_headonly_weakdistill_simdr0p0001_b64_s1p5_center_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_082548/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr13_headonly_weakdistill_simdr0p0001_b64_s1p5_p10_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_083118/eval/test/eval_summary.json`
- `runs/eval_fixed255k_xr13_headonly_weakdistill_simdr0p0001_b64_s1p5_p10_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_083338/eval/test/eval_summary.json`

Decision: XR-13 is closed without promotion. Head-only weak-distill SimDR reduced neither center nor P10 enough to beat XR-06C. Do not continue SimDR by only changing LR or weight unless a new diagnostic supports it. Next P0 is XR-02 dense/continuous prediction trajectory preparation, because EyeLoRiN-style post-processing is implemented but current sparse manifests invalidate smoothing across multi-second gaps.

## 2026-06-16 XR-02 Dense Trajectory Availability

Added diagnostic:

- `scripts/external/check_xr02_dense_trajectory_availability.py`

Run:

```bash
.venv/bin/python scripts/external/check_xr02_dense_trajectory_availability.py \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --canonical-root /home/kjm26/project/dataset/EV_Eye/canonical \
  --max-gap-us 50000 \
  --min-window 3
```

Result artifact:

- `runs/diagnostics/xr02_dense_trajectory_availability_20260616_084506.json`

Dense-trajectory check:

| Source | Groups | Rows | Min gap us | Median gap us | Dense pairs <=50k | Longest dense segment | Usable for M2F metric eval |
|---|---:|---:|---:|---:|---:|---:|---|
| test manifest | 72 | 2238 | 360000 | 4000003.0 | 0 | 1 | no |
| canonical annotations | 72 | 2238 | 360000 | 4000003.0 | 0 | 1 | no |

Decision: XR-02 EyeLoRiN-style post-process remains implemented, but current canonical1/test labels contain no time-contiguous labeled trajectory for M2F windows `{3,5,7,9}` under the `50000us` gap gate. Do not claim XR-02 metric promotion from this split. Next practical second-goal action is an XR-04 diagnostics refresh over blink/open-eye/low-event/fixation/saccade buckets, then a bounded temporal/head experiment only if the diagnostic identifies a targetable failure mode.

## 2026-06-16 XR-04 Diagnostics Refresh

Generated refreshed failure-bucket diagnostics for the active XR-06C leaders:

- Best-center artifact: `runs/diagnostics/xr04_refresh_xr06c_bestcenter_failure_buckets_20260616.json`
- Best-P10 artifact: `runs/diagnostics/xr04_refresh_xr06c_bestp10_failure_buckets_20260616.json`

Best-center diagnostic highlights:

| Bucket | Count | Weighted center px | Weighted P10 % | Weighted P5 % |
|---|---:|---:|---:|---:|
| Overall valid-track | 2238 | 20.1101 | 26.5202 | 8.4824 |
| `similarity_target <= 0.1` | 493 | 27.1247 | 19.2399 | 4.7506 |
| `similarity_target 0.1..0.3` | 311 | 21.7450 | 20.2206 | 4.7794 |
| `similarity_target 0.6..0.9` | 717 | 16.3534 | 33.2322 | 12.5948 |
| `session_key:user42/left/session_201` | 46 | 30.9384 | 14.6341 | 2.4390 |
| `subject_id:42` | 195 | 26.2739 | 19.2982 | 4.6784 |

Best-P10 diagnostic highlights:

| Bucket | Count | Weighted center px | Weighted P10 % | Weighted P5 % |
|---|---:|---:|---:|---:|
| Overall valid-track | 2238 | 20.2065 | 26.9801 | 8.5846 |
| `similarity_target <= 0.1` | 493 | 26.9471 | 19.0024 | 4.9881 |
| `similarity_target 0.1..0.3` | 311 | 21.7632 | 21.3235 | 5.1471 |
| `similarity_target 0.6..0.9` | 717 | 16.5990 | 34.4461 | 12.7466 |
| `session_key:user42/left/session_201` | 46 | 30.4950 | 14.6341 | 2.4390 |
| `subject_id:42` | 195 | 26.2106 | 19.2982 | 4.6784 |

Additional branch-state attempt:

- `runs/diagnostics/xr04_refresh_xr06c_bestcenter_searchstate_failure_buckets_20260616.json`
- `runs/diagnostics/xr04_refresh_xr06c_bestcenter_eventstate_failure_buckets_20260616.json`

Both joined `0/2238` rows because the active `trackonly` config normalizes `model.heads.active=track` and disables search/event heads at model build time. These artifacts are negative evidence only; they cannot compare fallback branch accuracy.

Decision: XR-04 refresh is complete. The dominant weighted failure remains low `similarity_target`, especially `session_201` and subjects `42/45/39`. Blink/closed-eye rows are not the immediate weighted-metric driver because closed-eye and invalid-track samples have zero weight. Since low-sim loss, hard subset, weighted sampler, and loss-side sample weighting all failed to promote, do not launch another reweighting replay. Next useful P0 is a no-train confidence/relocalization diagnostic: collect `track_pred` confidence/quality from the active leader, test whether confidence separates low-sim failures, then design a gated fallback or auxiliary confidence loss only if that diagnostic is positive.

Prepared confidence diagnostic support:

- `scripts/external/infer_hbtxr.py --limit` now supports bounded inference-row collection.
- `scripts/external/summarize_track_confidence_buckets.py` summarizes `track_pred` confidence/quality against target errors and manifest buckets.
- Limit-2 smoke artifacts:
  - `runs/diagnostics/xr04_confidence_xr06c_bestcenter_infer_limit2_20260616`
  - `runs/diagnostics/xr04_confidence_xr06c_bestcenter_limit2_buckets_20260616.json`

## 2026-06-16 XR-04 Confidence/Quality Probe

Added balanced confidence probe manifest:

- `scripts/external/build_confidence_probe_manifest.py`
- `data/_internal/manifests/manifest1/confidence_probe_low0p1_high0p6_128/test_manifest.jsonl`
- `data/_internal/manifests/manifest1/confidence_probe_low0p1_high0p6_128/confidence_probe_manifest_summary.json`

Manifest composition:

| Bucket | Rows available | Rows selected |
|---|---:|---:|
| `similarity_target <= 0.1` | 493 | 128 |
| `similarity_target >= 0.6` | 891 | 128 |

Inference and summary artifacts:

- Center leader inference: `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestcenter_infer_20260616`
- Center leader buckets: `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestcenter_buckets_20260616.json`
- P10 leader inference: `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestp10_infer_20260616`
- P10 leader buckets: `runs/diagnostics/xr04_confidence_probe_low0p1_high0p6_128_xr06c_bestp10_buckets_20260616.json`

Center leader result:

| Bucket | Count | Weighted center px | Weighted P10 % | Weighted P5 % | Mean confidence | Mean quality |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 256 | 19.5063 | 30.6604 | 8.4906 | 0.9999991 | 0.9999993 |
| `similarity_target <= 0.1` | 128 | 24.5408 | 21.5686 | 2.9412 | 0.9999997 | 0.9999998 |
| `similarity_target 0.6..0.9` | 98 | 15.5990 | 36.2637 | 14.2857 | 0.9999987 | 0.9999990 |
| `similarity_target >= 0.9` | 30 | 11.1933 | 52.6316 | 10.5263 | 0.9999980 | 0.9999985 |

P10 leader result:

| Bucket | Count | Weighted center px | Weighted P10 % | Weighted P5 % | Mean confidence | Mean quality |
|---|---:|---:|---:|---:|---:|---:|
| Overall | 256 | 19.6038 | 31.1321 | 9.4340 | 0.9999989 | 0.9999991 |
| `similarity_target <= 0.1` | 128 | 24.5044 | 20.5882 | 3.9216 | 0.9999997 | 0.9999997 |
| `similarity_target 0.6..0.9` | 98 | 15.8637 | 38.4615 | 15.3846 | 0.9999981 | 0.9999986 |
| `similarity_target >= 0.9` | 30 | 11.2084 | 52.6316 | 10.5263 | 0.9999978 | 0.9999984 |

Decision: existing `track_pred` confidence/quality logits are saturated and do not separate low-similarity failures. In this probe, the worst low-sim bucket is more confident than the easier high-sim buckets. Therefore an EX-Gaze-style no-train gate based on current confidence/quality is rejected. Next useful training branch must either calibrate confidence/quality with explicit error/support supervision and a usable fallback, or change representation/branch availability rather than using the current logits directly.

Root-cause check:

- `src/hbtxr/data/components.py` builds `pupil_track_target[..., 6]` from `valid_track`.
- `src/hbtxr/data/components.py` builds `pupil_track_target[..., 7]` from `annotation_quality`.
- `src/hbtxr/loss/bundles/track.py` trains `track_conf` and `track_quality` against those targets with BCE.

Therefore current confidence/quality heads are label-validity predictors, not error or relocalization confidence predictors. This explains saturation on valid rows and blocks direct EX-Gaze-style confidence gating.

## 2026-06-16 XR-14 Fallback Closeout And Next Branch

Result artifact:

- `runs/diagnostics/xr14_similarity_prevstate_fallback_xr06c_bestcenter_20260616.json`

Result: raw remains best. Previous-state and blend fallback both degrade the official-like batchmean metric, so no-train fallback is closed.

Aux-state probe:

- Inference: `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_infer_20260616`
- Track summary: `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_trackstate_buckets_20260616.json`
- Aux summary: `runs/diagnostics/xr14_auxstate_probe_xr06c_bestcenter_auxstate_buckets_20260616.json`

Result: `track_state_aux` is not usable as fallback. On the 256-row probe it has center `158.7825`, P10 `0.0`, P5 `0.0`.

Prepared next command:

```bash
bash scripts/external/run_xr14_allhead_relocalize_probe.sh 255000 5e-7 cuda:0
```

Use GPU1 for the complementary P10-lane variant if free:

```bash
XR14_INIT_CKPT=runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
XR14_TEACHER_CKPT=runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
bash scripts/external/run_xr14_allhead_relocalize_probe.sh 255000 5e-7 cuda:1
```

Launch note:

- Initial parallel launch was stopped after startup because both lanes shared the same experiment prefix and could select the wrong latest run root during post-train eval.
- Runner was patched to append lane tag to the experiment name.

Active v2 launches:

```bash
tmux new-session -d -s hgtxr_xr14a_allhead_center_gpu0_20260616_v2 \
  'bash scripts/external/run_xr14_allhead_relocalize_probe.sh 255000 5e-7 cuda:0 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt \
    center > runs/_logs/xr14a_allhead_relocalize_center_lr5e-7_gpu0_20260616_v2.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr14a_allhead_p10_gpu1_20260616_v2 \
  'bash scripts/external/run_xr14_allhead_relocalize_probe.sh 255000 5e-7 cuda:1 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    p10 > runs/_logs/xr14a_allhead_relocalize_p10_lr5e-7_gpu1_20260616_v2.log 2>&1'
```

Startup evidence:

- Center run root: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_center_trackpreserve_fullwidth_20260616_094218`
- P10 run root: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_allhead_relocalize_p10_trackpreserve_fullwidth_20260616_094220`
- Logs:
  - `runs/_logs/xr14a_allhead_relocalize_center_lr5e-7_gpu0_20260616_v2.log`
  - `runs/_logs/xr14a_allhead_relocalize_p10_lr5e-7_gpu1_20260616_v2.log`
- Both logs show raw event-count contract pass, intended checkpoint, `resolved_device=cuda:0/1`, `train_samples=5929`, `val_samples=844`, and epoch `1/12` training steps.

Post-train branch-state diagnostics:

Closeout:

- Both lanes early-stopped at epoch `8/12`.
- Center lane train exit: `XR14_ALLHEAD_RELOCALIZE_TRAIN_EXIT:0:2026-06-16T09:54:00+09:00`.
- P10 lane train exit: `XR14_ALLHEAD_RELOCALIZE_TRAIN_EXIT:0:2026-06-16T09:53:46+09:00`.
- Eval artifacts:
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestcenter_test_gpu0_w0_20260616_095401/eval/test/eval_summary.json`
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestp10_test_gpu0_w0_20260616_095710/eval/test/eval_summary.json`
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestcenter_test_gpu1_w0_20260616_095347/eval/test/eval_summary.json`
  - `runs/eval_fixed255k_xr14_allhead_relocalize_adamw_lr5e_7_bestp10_test_gpu1_w0_20260616_095652/eval/test/eval_summary.json`

Official test metric comparison:

| lane/checkpoint | center px | P10 pct | P5 pct | gate result |
|---|---:|---:|---:|---|
| center/best-center | `20.342467624800545` | `25.644133370263237` | `8.11947306905474` | fail center/P10/P5 |
| center/best-P10 | `20.342467624800545` | `25.644133370263237` | `8.11947306905474` | fail center/P10/P5 |
| P10/best-center | `20.348247524670192` | `25.703657184328353` | `7.97278938974653` | fail center/P10/P5 |
| P10/best-P10 | `20.348247524670192` | `25.703657184328353` | `7.97278938974653` | fail center/P10/P5 |

Active gates remain XR-06C: center `<20.283125744547164`, P10 `>26.74489871433803`, P5 `>8.69557854788644`.

Branch-state diagnostics were generated for each eval rows file and each state key:

- `runs/diagnostics/xr14a_center_bestcenter_track_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_center_bestcenter_search_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_center_bestcenter_event_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_center_bestp10_track_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_center_bestp10_search_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_center_bestp10_event_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_p10_bestcenter_track_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_p10_bestcenter_search_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_p10_bestcenter_event_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_p10_bestp10_track_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_p10_bestp10_search_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr14a_p10_bestp10_event_state_failure_buckets_20260616.json`

Bucket summary:

- Center-lane `track_state`: joined `2238`, weighted center `20.170973689926353`, P10 `25.753704649974452`, P5 `8.175779253960144`.
- P10-lane `track_state`: joined `2238`, weighted center `20.17721054100933`, P10 `25.8048032703117`, P5 `8.02248339294839`.
- Center-lane `search_state`: weighted center `177.7550336720757`, P10/P5 `0.0`.
- Center-lane `event_state`: weighted center `178.54195960106662`, P10/P5 `0.0`.
- The dominant track failure remains low similarity: `similarity_target < 0.1` has weighted center about `26.95px` and P10 about `18-19%`.

Decision: XR-14A all-head relocalization does not promote and does not produce a usable search/event fallback branch. Do not start calibrated fallback training from these search/event states.

Repro command pattern used for diagnostics:

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml \
  --mode mode1 --stage stage2 --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows <eval_rows.json> \
  --state-key track_state \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=false \
  --output runs/diagnostics/xr14a_<lane>_track_state_failure_buckets.json
```

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml \
  --mode mode1 --stage stage2 --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows <eval_rows.json> \
  --state-key search_state \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=false \
  --output runs/diagnostics/xr14a_<lane>_search_state_failure_buckets.json
```

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_allhead_relocalize_trackpreserve_fullwidth.yaml \
  --mode mode1 --stage stage2 --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows <eval_rows.json> \
  --state-key event_state \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=false \
  --output runs/diagnostics/xr14a_<lane>_event_state_failure_buckets.json
```

## 2026-06-16 XR-15 Support-Adaptive Event-Window Setup

Decision: next P0 is support-adaptive fixed-count event-window training. This follows EyeTrAES/EV-Eye event evidence guidance and avoids repeating pure low-sim weighting, hard-subset training, SimDR-only replay, and XR-14A fallback.

Artifacts:

- Plan: `docs/XR15_SUPPORT_ADAPTIVE_PLAN.md`
- Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml`
- Runner: `scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh`
- P5 eval helper: `scripts/external/eval_xr15_p5_checkpoint.sh`

Default XR-15A settings:

- policy: `fixed_count`
- base count: `255000`
- adaptive min/max: `192000/320000`
- reference gap: `4000003us`
- scale power: `0.5`
- LR: `5e-7`
- init/teacher: same lane-specific XR-06C checkpoint

Launch commands:

```bash
tmux new-session -d -s hgtxr_xr15_supportadaptive_center_gpu0_20260616 \
  'bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh 255000 192000 320000 4000003 0.5 5e-7 cuda:0 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_metric_track_center_px.pt \
    center > runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr15_supportadaptive_p10_gpu1_20260616 \
  'bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh 255000 192000 320000 4000003 0.5 5e-7 cuda:1 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    p10 > runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log 2>&1'
```

Post-eval diagnostic:

```bash
.venv/bin/python scripts/external/summarize_eval_failure_buckets.py \
  --config configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_fullwidth.yaml \
  --mode mode1 --stage stage2 --split test \
  --manifest data/_internal/manifests/manifest1/test_manifest.jsonl \
  --eval-rows <eval_dir>/eval/test/eval_rows.json \
  --state-key track_state \
  --override data.canonical_root=/home/kjm26/project/dataset/EV_Eye/canonical \
  --override data.mode1.event_builder.policy=fixed_count \
  --override data.mode1.event_builder.event_count_target=255000 \
  --override data.mode1.event_builder.adaptive_count.enabled=true \
  --override data.mode1.event_builder.adaptive_count.reference_us=4000003 \
  --override data.mode1.event_builder.adaptive_count.min_event_count=192000 \
  --override data.mode1.event_builder.adaptive_count.max_event_count=320000 \
  --override data.mode1.event_builder.adaptive_count.scale_power=0.5 \
  --output runs/diagnostics/xr15_<lane>_<ckpt>_track_state_failure_buckets_20260616.json
```

Launch status:

- Center lane started in tmux session `hgtxr_xr15_supportadaptive_center_gpu0_20260616`; log `runs/_logs/xr15_supportadaptive_center_lr5e-7_gpu0_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148`.
- P10 lane started in tmux session `hgtxr_xr15_supportadaptive_p10_gpu1_20260616`; log `runs/_logs/xr15_supportadaptive_p10_lr5e-7_gpu1_20260616.log`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205`.
- Both lanes passed startup validation: raw event-count contract, intended XR-06C checkpoint load, resolved CUDA device, train/val counts `5929/844`, and epoch `1/12` step logs.

Best-P5 follow-up after training:

```bash
bash scripts/external/eval_xr15_p5_checkpoint.sh \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103148 \
  center cuda:0
```

```bash
bash scripts/external/eval_xr15_p5_checkpoint.sh \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205 \
  p10 cuda:1
```

XR-15A closeout:

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center | best-center | 20.230558776855467 | 26.303146975381033 | 8.573554713385446 | center promotion vs XR-06C, not leader |
| center | best-P10 | 20.259641446386066 | 25.812925890513828 | 8.451105751310076 | center promotion vs XR-06C, not leader |
| p10 | best-center | 20.225680075372967 | 26.50085105895996 | 8.304422058377947 | new center leader |
| p10 | best-P10 | 20.259886418070113 | 26.12457557405744 | 8.357993486949375 | center promotion vs XR-06C, not leader |

New active gates:

- center `<20.225680075372967`
- P10 `>26.74489871433803`
- P5 `>8.69557854788644`

Diagnostics:

- `runs/diagnostics/xr15_p10_bestcenter_track_state_failure_buckets_20260616.json`
- `runs/diagnostics/xr15_center_bestcenter_track_state_failure_buckets_20260616.json`

Decision:

- XR-15A proves adaptive event support is useful for center error.
- P10/P5 did not promote; low-similarity/session_201/subject failure concentration remains.
- Best-P5 helper attempts did not produce summary artifacts in the sandbox/logged timeout path.
- Next execution should run XR-15B: min/base/max `224k/255k/288k`, same LR/init/teacher/weak-distill/direct-aux contract.

XR-15B launched:

```bash
tmux new-session -d -s hgtxr_xr15b_supportadaptive_center_gpu0_20260616 \
  'bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh 255000 224000 288000 4000003 0.5 5e-7 cuda:0 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205/train/best_metric_track_center_px.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_min192k_max320k_ref4000kus_pow0p5_fullwidth_20260616_103205/train/best_metric_track_center_px.pt \
    center_xr15b > runs/_logs/xr15b_supportadaptive_center_lr5e-7_gpu0_20260616.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr15b_supportadaptive_p10_gpu1_20260616 \
  'bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh 255000 224000 288000 4000003 0.5 5e-7 cuda:1 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    p10_xr15b > runs/_logs/xr15b_supportadaptive_p10_lr5e-7_gpu1_20260616.log 2>&1'
```

Startup evidence:

- Center lane log reached `resolved_device=cuda:0`, train/val `5929/844`, epoch `1/12`.
- P10 lane log reached `resolved_device=cuda:1`, train/val `5929/844`, epoch `1/12`.

## 2026-06-17 XR-60 Candidate-Head Preparation

Commands to validate:

```bash
bash -n scripts/external/run_xr60_p10_candidate_head.sh
```

```bash
python3 -m py_compile src/hbtxr/models/heads.py src/hbtxr/models/tracker/head_factory.py src/hbtxr/models/tracker/track_branch.py src/hbtxr/models/hybrid_tracker.py src/hbtxr/training/model_factory.py src/hbtxr/loss/bundles/track.py src/hbtxr/loss/bundles/__init__.py src/hbtxr/loss/stage2.py
```

```bash
.venv/bin/python -m pytest -q tests/test_track_center_l2_loss.py tests/test_trainable_filter.py
```

Dry-run commands:

```bash
DRY_RUN=1 bash scripts/external/run_xr60_p10_candidate_head.sh a cuda:0
DRY_RUN=1 bash scripts/external/run_xr60_p10_candidate_head.sh b cuda:1
```

Full-run commands after validation:

```bash
bash scripts/external/run_xr60_p10_candidate_head.sh a cuda:0
bash scripts/external/run_xr60_p10_candidate_head.sh b cuda:1
```

## 2026-06-16 XR-27 Track-Heatmap Coordinate Representation

Goal: close the scalar P10-boundary/LR/optimizer branch and test a real coordinate-representation change under the second-goal accuracy track.

Added artifacts:

- `configs/external/mode1_stage2_raw_event_count_lr1e-4_weakdistill_trackonly_heatmapstate_supportadaptive_headonly_fullwidth.yaml`
- `scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh`
- `docs/resources/xr27_trackheatmap_results_2026_06_16.md`

Implementation summary:

- Added `TrackCenterHeatmapHead`.
- Decoded heatmap peak plus sub-cell offset into `track/center_heatmap_xy`.
- Enabled optional heatmap-as-track-state replacement for `track/state[:, :2]`.
- Added grid CE and SmoothL1 offset losses.
- Limited trainable scope to `track_center_heatmap_head.*`.
- Disabled `distillation.state_similarity` after audit, because the teacher has random weights for the newly added heatmap head.

Clean commands:

```bash
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 1e-4 cuda:0 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr1e4
```

```bash
bash scripts/external/run_xr27_trackheatmap_p10teacher_probe.sh 255000 160000 384000 4000003 0.5 3e-5 cuda:1 0.005 0.001 0.001 32 xr06cp10_heatmapstate_nostatedistill_lr3e5
```

Results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Gate decision |
|---|---|---:|---:|---:|---|
| XR-27A LR `1e-4` | best-P10 | 17.274589475563594 | 31.915391901561193 | 10.289966331209456 | promotes all gates |
| XR-27A LR `1e-4` | best-P5 | 17.6397743497576 | 29.80569808142526 | 10.210459525244577 | promotes all gates, weaker than best-P10 |
| XR-27B LR `3e-5` | best-P10 | 18.388037020819528 | 27.238521099090576 | 8.488520683561052 | promotes center/P10 only |
| XR-27B LR `3e-5` | best-P5 | 19.03528357914516 | 24.192602627617973 | 6.854166896002633 | promotes center only |

Decision: XR-27A best-P10 becomes the new active software leader. Active gates are now center `<17.274589475563594`, P10 `>31.915391901561193`, and P5 `>10.289966331209456`.

Next execution: XR-28 should consolidate XR-27 with failure-bucket diagnostics, best-center checkpoint selection support, and a narrow LR/heatmap-weight refinement around LR `1e-4`.

## 2026-06-16 XR-15E Closeout And XR-16A Launch

XR-15E closeout:

- Center best-center `20.281941563742503 / 26.2521265574864 / 8.633928898402623`.
- Center best-P10 `20.297553059032985 / 26.096939522879463 / 8.618197590964181`.
- P10 best-center `20.3268527337483 / 26.333759232929776 / 8.409864234924317`.
- P10 best-P10 `20.3268527337483 / 26.333759232929776 / 8.409864234924317`.
- Decision: no promotion. Continue to XR-16A event-path adapter isolation instead of another both-adapter branch.

XR-16A runner added:

- Config: `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackeventadapter_centerloss_finetune_fullwidth.yaml`.
- Runner: `scripts/external/run_xr16_weakdistill_trackeventadapter_probe.sh`.
- Isolated trainable scope: `track_head.*`, `event_adapter.*`, and `patch_frontend.event_embed.proj.*`.

XR-16A launched:

- Center lane: tmux `hgtxr_xr16a_eventadapter_center_gpu0_20260616`, log `runs/_logs/xr16a_weakdistill_trackeventadapter_center_lr5e-7_gpu0_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_center_xr16a_fullwidth_20260616_125515`.
- P10 lane: tmux `hgtxr_xr16a_eventadapter_p10_gpu1_20260616`, log `runs/_logs/xr16a_weakdistill_trackeventadapter_p10_lr5e-7_gpu1_20260616.log`, run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackeventadapter_centerloss_p10_xr16a_fullwidth_20260616_125515`.
- Both lanes passed startup validation and entered epoch `1/12`.

XR-15C closeout:

- Center lane and P10 lane both completed training with exit `0`.
- Best-center eval summaries:
  - center lane: `20.19088832650866 / 26.009779623576573 / 8.436224787575858`.
  - P10 lane: `20.205999997683932 / 26.303146975381033 / 8.397959463936942`.
- Decision: center lane best-center is the new center leader. P10/P5 gates remain XR-06C/XR-05A, so active gates are center `<20.19088832650866`, P10 `>26.74489871433803`, and P5 `>8.69557854788644`.
- Next execution should switch to XR-15D bounded track-adapter/coordinate-head training rather than another support-adaptive range sweep.

XR-15D runner added:

- `scripts/external/run_xr15d_trackadapters_probe.sh`
- It wraps `scripts/external/run_prepare_and_train.sh`, forces fixed255k, disables adaptive count, runs Stage2-only training, then evaluates `best_metric_track_center_px.pt` and `best_track_p10.pt`.

XR-15D launched:

```bash
tmux new-session -d -s hgtxr_xr15d_trackadapters_center_gpu0_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr15d_trackadapters_probe.sh 255000 3e-6 cuda:0 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819/train/best_metric_track_center_px.pt \
    center_xr15d > runs/_logs/xr15d_trackadapters_center_lr3e-6_gpu0_20260616.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr15d_trackadapters_p10_gpu1_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr15d_trackadapters_probe.sh 255000 3e-6 cuda:1 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    p10_xr15d > runs/_logs/xr15d_trackadapters_p10_lr3e-6_gpu1_20260616.log 2>&1'
```

Startup evidence:

- Center lane log reached raw contract pass, intended XR-15C center checkpoint load, trainable filter `22` tensors / `448520` params, `resolved_device=cuda:0`, and epoch `1/16`.
- P10 lane log reached raw contract pass, intended XR-06C best-P10 checkpoint load, trainable filter `22` tensors / `448520` params, `resolved_device=cuda:1`, and epoch `1/16`.

XR-15D closeout:

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center_xr15d | best-center | 20.333278461865017 | 25.986395263671874 | 8.095238372257777 | no promotion |
| center_xr15d | best-P10 | 20.36335334096636 | 25.821854482378278 | 8.246173749651227 | no promotion |
| p10_xr15d | best-center | 20.362164442879813 | 26.141582359586444 | 8.562500286102296 | no promotion |
| p10_xr15d | best-P10 | 20.410626077651976 | 26.1883510862078 | 8.385629544939313 | no promotion |

Decision:

- XR-15D is closed. No-distill LR `3e-6` adapter updates drift away from the XR-15C center gate.
- Next execution is XR-15E: same adapter scope but weak distillation, teacher=init, LR `5e-7`.

XR-15E launched:

```bash
tmux new-session -d -s hgtxr_xr15e_weakdistill_trackadapters_center_gpu0_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr15e_weakdistill_trackadapters_probe.sh 255000 5e-7 cuda:0 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819/train/best_metric_track_center_px.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819/train/best_metric_track_center_px.pt \
    center_xr15e > runs/_logs/xr15e_weakdistill_trackadapters_center_lr5e-7_gpu0_20260616.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr15e_weakdistill_trackadapters_p10_gpu1_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr15e_weakdistill_trackadapters_probe.sh 255000 5e-7 cuda:1 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    p10_xr15e > runs/_logs/xr15e_weakdistill_trackadapters_p10_lr5e-7_gpu1_20260616.log 2>&1'
```

Startup evidence:

- Center lane log reached raw contract pass, intended XR-15C checkpoint load as init/teacher, trainable filter `22` tensors / `448520` params, `resolved_device=cuda:0`, and epoch `1/12`.
- P10 lane log reached raw contract pass, intended XR-06C P10 checkpoint load as init/teacher, trainable filter `22` tensors / `448520` params, `resolved_device=cuda:1`, and epoch `1/12`.

## 2026-06-16 XR-16A/XR-16B Closeout And XR-15C P5 Gate Update

XR-16A closeout:

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center_xr16a | best-center | 20.281941199302672 | 26.2521265574864 | 8.633928898402623 | no promotion |
| center_xr16a | best-P10 | 20.29755315440042 | 26.096939522879463 | 8.618197590964181 | no promotion |
| p10_xr16a | best-center | 20.3268525327955 | 26.333759232929776 | 8.409864234924317 | no promotion |
| p10_xr16a | best-P10 | 20.3268525327955 | 26.333759232929776 | 8.409864234924317 | no promotion |

Decision:

- XR-16A is closed. Event-path-only adapter training reproduced XR-15E-level results and did not promote over active gates.

XR-16B interpolation diagnostic:

- Runner: `scripts/external/run_xr16b_checkpoint_interp_eval.sh`.
- Input A: XR-15C center-lane best-center checkpoint.
- Input B: XR-06C best-P10 checkpoint.
- Alpha `0.125` output checkpoint: `runs/interpolated_checkpoints/xr16b_xr15c_center_xr06c_p10_alpha0p125.pt`.
- Test summary: `20.28155174595969 / 26.309524529320854 / 8.619047934668405`.
- Decision: no promotion. Cross-branch interpolation did not preserve the XR-15C center gain or XR-06C P10 gate.

XR-15C best-P5 artifact closeout:

| Lane | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---:|---|
| center_xr15c | best-P5 | 20.243119949953897 | 26.12159937449864 | 8.601190778187343 | no promotion |
| p10_xr15c | best-P5 | 20.245368467058455 | 26.487245675495693 | 8.703231593540737 | P5 promotion |

Updated active gates:

- Center: XR-15C center-lane best-center, `<20.19088832650866`.
- P10: XR-06C best-P10, `>26.74489871433803`.
- P5: XR-15C P10-lane best-P5, `>8.703231593540737`.

Next execution:

- Use the XR-15C P5/balanced checkpoint as the updated secondary anchor.
- Prefer a same-branch XR-15C center-to-P5 interpolation or failure-bucket comparison before launching another trainable adapter branch.

## 2026-06-16 XR-17A Same-Branch Center-To-P5 Interpolation

Runner:

- `scripts/external/run_xr17a_xr15c_center_p5_interp_eval.sh`

Endpoints:

- Alpha `0.0`: XR-15C center-lane best-center checkpoint, `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113819/train/best_metric_track_center_px.pt`.
- Alpha `1.0`: XR-15C P10-lane best-P5 checkpoint, `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15c_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_113816/train/best_track_p5.pt`.

Sweep:

| Alpha | Center px | P10 pct | P5 pct | Decision |
|---:|---:|---:|---:|---|
| 0.25 | 20.199484479427337 | 26.20960956301008 | 8.625425474984306 | no promotion |
| 0.50 | 20.211353632381986 | 26.35204153742109 | 8.679847247259957 | no promotion |
| 0.625 | 20.218607200895036 | 26.449830661501203 | 8.703231607164655 | tie-level P5 improvement only |
| 0.75 | 20.22669484274728 | 26.34566399029323 | 8.732993507385254 | P5 promotion |
| 0.875 | 20.235614109039307 | 26.527636807305473 | 8.673469693320138 | no promotion |

Decision:

- XR-17A alpha `0.75` becomes the P5/balanced secondary.
- Updated active gates: center `<20.19088832650866`, P10 `>26.74489871433803`, P5 `>8.732993507385254`.
- Same-branch interpolation is useful for P5 recovery, unlike XR-16B cross-branch interpolation.

## 2026-06-16 XR-17A Failure-Bucket Comparison

Artifacts:

- XR-15C center leader buckets: `runs/diagnostics/xr17a_compare_xr15c_center_failure_buckets_20260616.json`.
- XR-17A alpha `0.75` buckets: `runs/diagnostics/xr17a_compare_xr17a_alpha0p75_failure_buckets_20260616.json`.

Row-weighted aggregate:

| Checkpoint | Center px | P10 pct | P5 pct |
|---|---:|---:|---:|
| XR-15C center leader | 20.009468631680964 | 26.162493612672456 | 8.482370975983649 |
| XR-17A alpha 0.75 | 20.04153194769563 | 26.520183955033215 | 8.788962698007154 |

Key bucket deltas:

- Similarity buckets: P10/P5 improve in most bins, including `<=0.1`, `0.3..0.6`, `0.6..0.9`, and `>0.9`; center slightly worsens across most bins.
- Subject buckets: subjects `42`, `45`, and `39` improve on P10/P5; subjects `43` and `41` regress.
- Eye buckets: left-eye P10/P5 improve; right-eye P10 improves but P5 slightly regresses.

Decision:

- XR-17A should be treated as a P5-direction anchor, not as the next center seed.
- Next training branch should preserve the XR-15C center checkpoint as primary seed/teacher while using XR-17A alpha `0.75` only as a bounded P5 regularizer, dual-teacher target, or checkpoint-soup constraint.

## 2026-06-16 XR-17B P5-Anchor Support-Adaptive Branch

Added artifacts:

- `configs/external/mode1_stage2_raw_event_count_lr2p5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_fullwidth.yaml`.
- `scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh`.

Command:

```bash
bash scripts/external/run_xr17b_p5anchor_supportadaptive_probe.sh \
  255000 160000 384000 4000003 0.5 2.5e-7 cuda:0
```

Runtime notes:

- First sandboxed launch failed at CUDA resolution with `cuda_is_available=False` and `cuda_device_count=0`.
- Escalated GPU0 launch succeeded; train log resolved `cuda:0`.
- Run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_centerinit_p5teacher_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_143310`.
- Training completed epoch `8/8`, train exit `0`.

Test results:

| Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---|
| best-center | 20.182338142395018 | 26.113946315220424 | 8.474490090778895 | center promotion |
| best-P10 | 20.22993471963065 | 26.166242245265416 | 8.603316634041922 | no promotion |
| best-P5 | 20.234592584201266 | 26.09183748109 | 8.844813244683403 | P5 promotion |

Decision:

- XR-17B promotes two active gates: center and P5.
- Active gates are now center `<20.182338142395018`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- P10 remains the main unresolved axis.

## 2026-06-16 XR-18A P10 Interpolation Diagnostic

Runner:

- `scripts/external/run_xr18a_xr17b_center_xr06c_p10_interp_eval.sh`

Endpoints:

- Alpha `0.0`: XR-17B best-center checkpoint.
- Alpha `1.0`: XR-06C best-P10 checkpoint.

Sweep:

| Alpha | Center px | P10 pct | P5 pct | Decision |
|---:|---:|---:|---:|---|
| 0.025 | 20.184962025710515 | 26.113946315220424 | 8.474490090778895 | no promotion |
| 0.05 | 20.18765983411244 | 26.054422501155308 | 8.525510501861572 | no promotion |
| 0.075 | 20.190412517956325 | 26.156463316508702 | 8.525510501861572 | no promotion |
| 0.10 | 20.193220179421562 | 26.20748372077942 | 8.629677173069545 | no promotion |

Decision:

- No-train interpolation does not recover P10.
- Alpha `0.10` is the best P10/P5 point in this sweep, but it misses center `<20.182338142395018`, P10 `>26.74489871433803`, and P5 `>8.844813244683403`.

## 2026-06-16 XR-19A Trainable P10-Recovery Micro Polish

Config and runner:

- `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_fullwidth.yaml`
- `scripts/external/run_xr19a_p10recovery_micro_polish.sh`

Command:

```bash
bash scripts/external/run_xr19a_p10recovery_micro_polish.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0
```

Runtime notes:

- Static validation passed: `chmod +x`, `bash -n`, `git diff --check`, config-load smoke, default checkpoint existence, and raw event-count contract.
- Escalated GPU0 launch succeeded and resolved `cuda:0`.
- Run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_micro_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_152610`.
- Training completed epoch `8/8`, train exit `0`.

Test results:

| Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---|
| best-center | 20.182335831437793 | 25.973640203475952 | 8.588435670307705 | center tie-break improvement over XR-17B, but P10/P5 miss |
| best-P10 | 20.20784169435501 | 26.232143613270352 | 8.609694181169782 | no P10 promotion |
| best-P5 | 20.181213889803207 | 25.928997346333094 | 8.508503695896694 | center promotion |

Diagnostic:

- `runs/diagnostics/xr19a_p10recovery_bestp10_failure_buckets_20260616.json`
- Weighted aggregate: center `20.0236`, P10 `26.3669`, P5 `8.6357`.
- Low-similarity rows remain the dominant P10 bottleneck: `similarity_target < 0.1` weighted center `26.5449`, P10 `17.5772`, P5 `5.2257`.

Decision:

- XR-19A promotes the center gate only, to `<20.181213889803207`.
- P10 remains XR-06C `>26.74489871433803`; P5 remains XR-17B `>8.844813244683403`.
- The branch is center-safe but too weak for P10 recovery. Next P0 should use stronger constrained P10 recovery or a P10-specialized lane with explicit acceptance as a separate P10 leader.

## 2026-06-16 XR-20A/XR-20B P10-Recovery LR Ladder

Commands:

```bash
bash scripts/external/run_xr19a_p10recovery_micro_polish.sh \
  255000 160000 384000 4000003 0.5 2.5e-7 cuda:0 \
  runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_centerinit_p5teacher_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_143310/train/best_track_p5.pt \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
  p5init_xr06cp10teacher_lr2p5e7
```

```bash
bash scripts/external/run_xr19a_p10recovery_micro_polish.sh \
  255000 160000 384000 4000003 0.5 5e-7 cuda:1 \
  runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p5anchor_centerinit_p5teacher_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_143310/train/best_track_p5.pt \
  runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
  p5init_xr06cp10teacher_lr5e7
```

Runtime notes:

- XR-20A run root: `runs/raw_mode1_stage2_count255000_adamw_lr2_5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr2p5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155340`.
- XR-20B run root: `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10recovery_p5init_xr06cp10teacher_lr5e7_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_155656`.
- XR-20A completed epoch `8/8`; XR-20B early-stopped at epoch `6`.
- After eval closeout, no active train/eval processes remained and both GPUs were idle.

Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| XR-20A LR `2.5e-7` | best-center | 20.175542894431523 | 25.884354482378278 | 8.486394848142352 | center promotion |
| XR-20A LR `2.5e-7` | best-P10 | 20.202199470996856 | 26.031038168498448 | 8.74914996964591 | no P10/P5 promotion |
| XR-20A LR `2.5e-7` | best-P5 | 20.211376798152923 | 26.00552796636309 | 8.71088467325483 | no promotion |
| XR-20B LR `5e-7` | best-center | 20.257083107743945 | 25.838861281531198 | 8.380102341515677 | no promotion |
| XR-20B LR `5e-7` | best-P10 | 20.183283712182725 | 25.8312082358769 | 8.423894848142352 | no promotion |
| XR-20B LR `5e-7` | best-P5 | 20.218086302280426 | 25.805698026929583 | 8.588435690743582 | no promotion |

Decision:

- XR-20A updates the center gate to `<20.175542894431523`.
- P10 remains XR-06C `>26.74489871433803`; P5 remains XR-17B `>8.844813244683403`.
- XR-20B shows that simply increasing this P10-recovery LR hurts the branch. Stop the P5-init/XR-06C-P10-teacher LR ladder and move next to a mechanism change.

XR-15B closeout:

- Center lane completed epoch `12/12` with train exit `0`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512`.
- P10 lane completed epoch `12/12` with train exit `0`; run root `runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111515`.
- Best-center eval summaries:
  - center lane: `20.215672533852715 / 26.575255823135375 / 8.627551317214966`.
  - P10 lane: `20.21979672227587 / 26.324405458995273 / 8.28741525241307`.
- Best-P10 eval artifacts did not complete; only `hypers/*` exists for those runs.

## 2026-06-16 XR-21 P10-Margin Mechanism Change

Commands:

```bash
bash scripts/external/run_xr21_p10margin_supportadaptive_probe.sh \
  255000 160000 384000 4000003 0.5 1.25e-7 cuda:0 0.0015 10.0
```

```bash
bash scripts/external/run_xr21_p10margin_supportadaptive_probe.sh \
  255000 160000 384000 4000003 0.5 1.25e-7 cuda:1 0.003 10.0
```

```bash
bash scripts/external/run_xr21_p10leader_margin_probe.sh \
  255000 160000 384000 4000003 0.5 1.25e-7 cuda:1 0.0005 10.0
```

Runtime notes:

- Secondary `w=0.0015` run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_w0_0015_m10_0_xr20acenter_xr06cp10teacher_p10margin_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_162707`.
- Secondary `w=0.003` run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10margin_w0_003_m10_0_xr20acenter_xr06cp10teacher_p10margin_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_162709`.
- Primary P10-leader run root: `runs/raw_mode1_stage2_count255000_adamw_lr1_25e_7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_margin_w0_0005_m10_0_xr06cp10_init_teacher_p10margin_min160k_max384k_ref4000kus_pow0p5_fullwidth_20260616_164618`.
- Secondary runs completed with train exit `0`; primary completed epoch `8/8` with train exit `0`.

Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| XR-21 secondary `w=0.0015` | best-center | 20.18683280604226 | 25.886480338232857 | 8.49914995602199 | no promotion |
| XR-21 secondary `w=0.0015` | best-P10 | 20.19624582529068 | 25.90688850539071 | 8.558673770087106 | no promotion |
| XR-21 secondary `w=0.0015` | best-P5 | 20.19960424559457 | 25.96938850539071 | 8.77976222038269 | no promotion |
| XR-21 secondary `w=0.003` | best-center | 20.197126933506556 | 25.900510951450894 | 8.618197584152222 | no promotion |
| XR-21 secondary `w=0.003` | best-P10 | 20.187303059441703 | 26.126701450347902 | 8.791666977746146 | no promotion |
| XR-21 secondary `w=0.003` | best-P5 | 20.200117662974765 | 25.96938850539071 | 8.728741809300013 | no promotion |
| XR-21 primary `w=0.0005` | best-P10 | 20.26367484842028 | 26.20110617365156 | 8.696003689084733 | no promotion |
| XR-21 primary `w=0.0005` | best-P5 | 20.26367484842028 | 26.20110617365156 | 8.696003689084733 | no promotion |

Decision:

- No active gate promoted. Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.844813244683403`.
- The next P0 should not be another LR-only or tiny hinge-weight replay.

## 2026-06-16 XR-22 Tri-Leader Checkpoint Soup

Commands:

```bash
bash scripts/external/run_xr22_trileader_soup_eval.sh cuda:0 \
  0.50,0.25,0.25:c50_p25_f25 \
  0.34,0.33,0.33:c34_p33_f33 \
  0.25,0.50,0.25:c25_p50_f25
```

```bash
bash scripts/external/run_xr22_trileader_soup_eval.sh cuda:1 \
  0.25,0.25,0.50:c25_p25_f50 \
  0.20,0.60,0.20:c20_p60_f20 \
  0.20,0.40,0.40:c20_p40_f40
```

Runtime notes:

- Added `scripts/external/average_hbtxr_checkpoints.py` for weighted N-way checkpoint soup.
- Added `scripts/external/run_xr22_trileader_soup_eval.sh` for no-train soup generation and test eval.
- Both GPU sessions exited `0`; no active train/eval process remained after closeout.

Test results:

| Soup | Center px | P10 pct | P5 pct | Decision |
|---|---:|---:|---:|---|
| `c50_p25_f25` | 20.21947033064706 | 26.349915708814347 | 8.785289430618286 | no promotion |
| `c34_p33_f33` | 20.236038860252926 | 26.4604599407741 | 8.869047941480364 | P5 promotion |
| `c25_p50_f25` | 20.257381524358475 | 26.36267081669399 | 8.601190784999302 | no promotion |
| `c25_p25_f50` | 20.235262938908168 | 26.269133404323032 | 8.829932287761144 | no promotion |
| `c20_p60_f20` | 20.270640075206757 | 26.405187838418144 | 8.735119356427873 | no promotion |
| `c20_p40_f40` | 20.251396659442356 | 26.349915736062187 | 8.603316634041922 | no promotion |

Decision:

- `c34_p33_f33` promotes the P5/balanced gate to `8.869047941480364`.
- Center remains XR-20A `<20.175542894431523`; P10 remains XR-06C `>26.74489871433803`.

## 2026-06-16 XR-23 P10-Preserving Optimizer Probe

Commands:

```bash
bash scripts/external/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 adopt 2.5e-7 cuda:0
```

```bash
bash scripts/external/run_xr23_p10_optimizer_probe.sh 255000 160000 384000 4000003 0.5 lion 1e-7 cuda:1
```

Runtime notes:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10leader_optprobe_fullwidth.yaml`.
- Added runner `scripts/external/run_xr23_p10_optimizer_probe.sh`.
- XR-23A ADOPT loaded XR-06C best-P10 as init and teacher, completed epoch `8/8`, and wrote best-P10/best-P5 eval summaries.
- XR-23B Lion loaded XR-06C best-P10 as init and teacher, early-stopped at epoch `6/8`, and wrote best-P10/best-P5 eval summaries.

Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| ADOPT `2.5e-7` | best-P10 | 20.261837770257678 | 26.318027945927213 | 8.681122745786395 | no promotion |
| ADOPT `2.5e-7` | best-P5 | 20.219128920350755 | 25.956633363451278 | 8.469388042177473 | no promotion |
| Lion `1e-7` | best-P10 | 20.22320341382708 | 26.311650391987392 | 8.513180569240026 | no promotion |
| Lion `1e-7` | best-P5 | 20.271730688640048 | 26.129677595411028 | 8.532313203811645 | no promotion |

Decision:

- No active gate promoted.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- Optimizer substitution from XR-06C is weaker than the XR-22 soup anchor for the next P10-recovery branch.

## 2026-06-16 XR-24 XR-22-Anchor P10-Recovery

Commands:

```bash
bash scripts/external/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0
```

```bash
bash scripts/external/run_xr24_xr22_anchor_p10recovery.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1
```

Runtime notes:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_fullwidth.yaml`.
- Added runner `scripts/external/run_xr24_xr22_anchor_p10recovery.sh`.
- Both branches loaded `runs/interpolated_checkpoints/xr22_trileader_soup_c34_p33_f33.pt` as init and XR-06C best-P10 as teacher.
- LR `1.25e-7` and LR `2.5e-7` both completed epoch `8/8` with train exit `0`.

Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| AdamW `1.25e-7` | best-P10 | 20.217043702942984 | 26.254252440588814 | 8.764881270272392 | no promotion |
| AdamW `1.25e-7` | best-P5 | 20.223851100036075 | 26.415817070007325 | 8.722364262172155 | no promotion |
| AdamW `2.5e-7` | best-P10 | 20.23367166178567 | 26.121599388122558 | 8.588435677119664 | no promotion |
| AdamW `2.5e-7` | best-P5 | 20.238911376680647 | 26.07695653779166 | 8.791666984558105 | no promotion |

Decision:

- No active gate promoted.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- XR-22 soup should remain a no-train anchor/reference. The next P0 should change optimizer defaults materially with ADOPT beta/eps validation or add P10-specific head/loss supervision instead of another AdamW LR replay.

## 2026-06-16 XR-25 XR-22-Anchor ADOPT-Defaults Validation

Commands:

```bash
bash scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 1.25e-7 cuda:0
```

```bash
bash scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh 255000 160000 384000 4000003 0.5 2.5e-7 cuda:1
```

Runtime notes:

- Added config `configs/external/mode1_stage2_raw_event_count_lr1p25e-7_weakdistill_trackonly_trackstateaux_supportadaptive_xr22p10recovery_adoptdefaults_fullwidth.yaml`.
- Added runner `scripts/external/run_xr25_xr22_anchor_adopt_defaults.sh`.
- Both branches loaded `runs/interpolated_checkpoints/xr22_trileader_soup_c34_p33_f33.pt` as init and XR-06C best-P10 as teacher.
- ADOPT was forced to `betas=[0.9,0.9999]` and `eps=1e-6`.
- LR `1.25e-7` early-stopped at epoch `7/8`; LR `2.5e-7` completed epoch `8/8`; both train exits were `0`.

Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| ADOPT-defaults `1.25e-7` | best-P10 | 20.21177260194506 | 26.46045993396214 | 8.728741809300013 | no promotion |
| ADOPT-defaults `1.25e-7` | best-P5 | 20.22005627495902 | 26.181123181751797 | 8.707483305249895 | no promotion |
| ADOPT-defaults `2.5e-7` | best-P10 | 20.23533037390028 | 26.108844266619002 | 8.64583364214216 | no promotion |
| ADOPT-defaults `2.5e-7` | best-P5 | 20.23760941709791 | 26.187500749315536 | 8.86607174192156 | no promotion |

Decision:

- No active gate promoted.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- XR-22 trainable polish is closed for AdamW and ADOPT-defaults. Next P0 is XR-26 P10-boundary loss from XR-06C best-P10 init/teacher.

## 2026-06-16 XR-26 P10-Boundary Aux Head-Only Probe

Commands:

```bash
bash scripts/external/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:0
```

```bash
bash scripts/external/run_xr26_p10boundary_aux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:1 runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt xr06cp10_init_teacher_light 0.01 0.005
```

Runtime notes:

- Added P10-boundary surrogate in `src/hbtxr/loss/bundles/track.py` and tests in `tests/test_track_center_l2_loss.py`.
- Added config `configs/external/mode1_stage2_raw_event_count_lr5e-7_weakdistill_trackonly_trackstateaux_supportadaptive_p10boundary_headonly_fullwidth.yaml`.
- Added runner `scripts/external/run_xr26_p10boundary_aux_probe.sh`; it now accepts optional positional boundary weights for reproducible parallel lanes.
- Initial sandbox launch failed because PyTorch saw `cuda_device_count=0`; escalated relaunch resolved CUDA correctly.
- Both lanes used XR-06C best-P10 as init and teacher, trained only `track_head.*` and `track_state_aux_head.*`, and early-stopped at epoch `6/8` with train exit `0`.

Test results:

| Branch | Checkpoint | Center px | P10 pct | P5 pct | Decision |
|---|---|---:|---:|---:|---|
| boundary `0.02/0.01` | best-P10 | 20.32213627440589 | 26.45790890966143 | 8.535289403370449 | no promotion |
| boundary `0.02/0.01` | best-P5 | 20.318065077917918 | 26.42814700944083 | 8.541666957310268 | no promotion |
| boundary `0.01/0.005` | best-P10 | 20.32214218207768 | 26.45790890966143 | 8.535289403370449 | no promotion |
| boundary `0.01/0.005` | best-P5 | 20.318073788711004 | 26.42814700944083 | 8.541666957310268 | no promotion |

Decision:

- No active gate promoted.
- Active gates remain center `<20.175542894431523`, P10 `>26.74489871433803`, P5 `>8.869047941480364`.
- Scalar P10-boundary polish is closed. Next P0 should change coordinate representation or teacher training, not repeat hinge/boundary/LR-only variants.

XR-15C launched:

```bash
tmux new-session -d -s hgtxr_xr15c_supportadaptive_center_gpu0_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:0 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512/train/best_metric_track_center_px.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_supportadaptive_center_xr15b_min224k_max288k_ref4000kus_pow0p5_fullwidth_20260616_111512/train/best_metric_track_center_px.pt \
    center_xr15c > runs/_logs/xr15c_supportadaptive_center_lr5e-7_gpu0_20260616.log 2>&1'
```

```bash
tmux new-session -d -s hgtxr_xr15c_supportadaptive_p10_gpu1_20260616 \
  'cd /home/kjm26/project/PRJXR/XR-VIT/HGTXR/software && bash scripts/external/run_xr15_support_adaptive_trackstateaux_probe.sh 255000 160000 384000 4000003 0.5 5e-7 cuda:1 \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    runs/raw_mode1_stage2_count255000_adamw_lr5e_7_weakdistill_trackonly_trackstateaux_c0p0005_axis0p0125_angle0p005_fullwidth_20260616_044434/train/best_track_p10.pt \
    p10_xr15c > runs/_logs/xr15c_supportadaptive_p10_lr5e-7_gpu1_20260616.log 2>&1'
```

Startup evidence:

- Center lane log reached `resolved_device=cuda:0`, train/val `5929/844`, epoch `1/12`.
- P10 lane log reached `resolved_device=cuda:1`, train/val `5929/844`, epoch `1/12`.
