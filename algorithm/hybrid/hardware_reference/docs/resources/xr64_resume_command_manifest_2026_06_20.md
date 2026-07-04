# XR-64 Resume Command Manifest

Date: 2026-06-20

This is a future execution contract only. Experiments are paused by user directive, so the commands below must not be run until the user explicitly resumes experiments.

## Purpose

Make XR-64-prep reproducible and auditable:

- generate 8 train/val teacher eval-row files,
- generate 6 train/val override JSON files,
- run strict readiness checking,
- launch XR-64A/B only after readiness is proven,
- collect post-run candidates and print promotion-decision commands only after future XR-64A/B test eval summaries exist.

## Guardrails

- Do not run while `execution_state=paused_by_user_directive`.
- Do not include `test` in `XR64_SPLITS`.
- Do not generate test target overrides.
- Do not launch XR-64A/B before strict readiness reports `ready_to_train`.
- Do not run post-run decision review before XR-64A/B test eval summaries have ablation axis certification.
- This manifest is not model-result evidence.

## Precheck Commands

```bash
bash -n scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh scripts/external/run_xr64_teacher_target_construction.sh
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --allow-missing-generated --format summary
```

## Prep Runner Contract

`scripts/external/check_xr64_resume_command_manifest.py` also validates static safety markers in `scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh`.

Required properties:

- strict bash mode,
- explicit test split refusal,
- default `train,val` split scope,
- default teacher set `xr62a,xr39,xr56b,xr58a`,
- default `eval,build` action scope,
- teacher eval runs with `data.track_target_override_path=null`,
- teacher eval runs with `data.allow_test_target_override=false`,
- eval and override start/done log markers,
- JSON row-count validation,
- foreground `tee` logging,
- existing eval-row validation on skip,
- existing override validation on skip,
- build input eval-row validation before override generation.

Forbidden markers:

- `XR64_SPLITS:-train,val,test`,
- `XR64_SPLITS:-test`,
- `allow_test_target_override=true`,
- `data.track_target_override_path=test`.

## XR-64-Prep Commands

| ID | Split | Teacher | Command |
|---|---|---|---|
| XR64-EVAL-TRAIN-XR62A | train | xr62a | `XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr62a XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-TRAIN-XR39 | train | xr39 | `XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr39 XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-TRAIN-XR56B | train | xr56b | `XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr56b XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-TRAIN-XR58A | train | xr58a | `XR64_ACTIONS=eval XR64_SPLITS=train XR64_TEACHERS=xr58a XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-VAL-XR62A | val | xr62a | `XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr62a XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-VAL-XR39 | val | xr39 | `XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr39 XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-VAL-XR56B | val | xr56b | `XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr56b XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-EVAL-VAL-XR58A | val | xr58a | `XR64_ACTIONS=eval XR64_SPLITS=val XR64_TEACHERS=xr58a XR64_EVAL_DEVICE=cuda:0 bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-BUILD-TRAIN | train | all | `XR64_ACTIONS=build XR64_SPLITS=train bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |
| XR64-BUILD-VAL | val | all | `XR64_ACTIONS=build XR64_SPLITS=val bash scripts/external/run_xr64_teacher_eval_rows_and_overrides.sh` |

## Input Readiness

The eval commands require the manifest split files and teacher checkpoints. The two build commands require the eight train/val teacher eval-row files produced by the eval commands. The manifest checker reports both levels separately from static manifest validity.

Current paused-state readiness:

| Item | Value |
|---|---:|
| eval required input count | `6` |
| existing eval required input count | `6` |
| missing eval required input count | `0` |
| eval inputs ready | `true` |
| build required input count | `8` |
| existing build required input count | `0` |
| missing build required input count | `8` |
| build inputs ready | `false` |

This does not make the manifest invalid, because the missing build files are expected while experiments are paused. It does mean XR-64 eval commands have their static inputs available, but XR-64 build execution remains invalid until XR-64-prep eval rows exist.

## Expected Generated Counts

The manifest also pins the expected row cardinality for future generated artifacts. These counts are read from the current `manifest1` train/val split manifests and checked by `scripts/external/check_xr64_resume_command_manifest.py`.

| Item | Value |
|---|---:|
| train split rows | `5929` |
| val split rows | `844` |
| teacher count | `4` |
| override rule count | `3` |
| total teacher eval-row records | `27092` |
| total override records | `20319` |

Interpretation:

- each teacher `eval_rows.json` must cover all rows in its split,
- each override JSON must cover all rows in its split,
- total teacher eval-row records are `(5929 + 844) * 4`,
- total override records are `(5929 + 844) * 3`,
- these are contract counts only; they do not mean artifacts already exist.

## Next Prep Selector

```bash
.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format summary
.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --format commands
.venv/bin/python scripts/external/emit_xr64_next_prep_command.py --selection ready --format json
```

The selector reads this manifest and current generated-artifact state to identify the next XR-64-prep command that is ready after explicit resume. Current state selects `XR64-EVAL-TRAIN-XR62A`, with `ready_after_resume_count=8`, `blocked_count=2`, `completed_count=0`, and `allowed_to_run_now=false`.

## Strict Readiness Gate

```bash
.venv/bin/python scripts/external/check_xr64_resume_artifacts.py --format summary
```

Required summary state:

```text
resume_status: ready_to_train
ready_to_train: true
can_run_lane: true
missing_eval_rows: 0
missing_overrides: 0
leakage_risk: none
```

The strict checker also requires full manifest coverage. Each generated teacher `eval_rows.json` must contain every sample ID from its train/val split manifest exactly once, and each override JSON must contain every sample ID from its train/val split manifest exactly once. Subset artifacts can be useful for smoke debugging, but they do not satisfy `ready_to_train`.

## Launch Commands After Strict Ready

```bash
bash scripts/external/run_xr64_teacher_target_construction.sh a cuda:0
bash scripts/external/run_xr64_teacher_target_construction.sh b cuda:1
```

XR-64C remains conditional after A/B evidence.

## Post-Run Decision Commands After Launch

These commands are still review commands only. They are included so future resume work has a complete path from prep to launch to promotion decision. They must not be used as training/evaluation evidence.

```bash
.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format summary
.venv/bin/python scripts/external/collect_xr64_postrun_candidates.py --format commands
```

Required state before using these commands for any promotion decision:

```text
XR-64A and XR-64B launch completed
override-cleared test eval_summary.json files exist
eval summaries include ablation_axis and axis_certified=true
collector reports certified candidates before decision-helper command is trusted
```

Machine-readable source: `docs/resources/xr64_resume_command_manifest_2026_06_20.json`.
