# Event Experiment Import Candidates

This note curates event-side experiment results and workflows worth carrying
into the event-only algorithm track.

## Source Scope

- `external_hybrid_package/docs/prj/*event*`
- `external_hybrid_package/exps/configs/mode2*`
- `external_hybrid_package/exps/scripts/*event*`
- `hgtxr_software/docs/track/RAW_EVENT_COUNT_TRAINING.md`
- `hgtxr_software/docs/resources/xr50_*`
- `hgtxr_software/docs/resources/xr66_*`

## P0 Candidates

### Raw EV-Eye event-count training contract

Source:

- `hgtxr_software/docs/track/RAW_EVENT_COUNT_TRAINING.md`

Useful result:

- Raw pipeline trains without frame/event interpolation.
- Manifest contract:
  `frame_source = original`,
  `event_policy = fixed_count`,
  `event_count_target = 5000`,
  `resize_policy = facet_square_direct`.
- Split counts:
  train `5929`, val `844`, test `2238`.
- Full Stage1 frame-based run:
  best validation `metric_search_p10_pct = 3.0828841947159678` at epoch `6`,
  final validation loss `72.48529473790583`.
- Full Stage2 hybrid raw event-count run:
  best validation `metric_track_p10_pct = 7.2652742097962575` at epoch `1`,
  final validation loss `170.1914188456985`.
- Result checker returned `ok: true`.

Recommended action:

- Import the raw event-count manifest/checker idea into the event track.
- Keep fixed-count `5000` as the first reproducible event-only baseline before
  adaptive count or time-bin variants.
- Preserve CUDA/cuDNN runtime notes separately from algorithm logic.

Risk:

- Full-run accuracy was weak, so this is a contract baseline rather than a
  performance baseline.

### Low-similarity targeted recovery diagnostics

Source:

- `hgtxr_software/docs/resources/xr66_lowsim_targeted_recovery_results_2026_06_22.md`

Useful result:

- Hard low-similarity subset training improved targeted low-sim P10 in some
  checkpoints, e.g. `36.36136802550285`, but failed full-test promotion.
- Full-test best P10 stayed below the active P10 gate by at least
  `0.43664969716753`, and center worsened relative to the center leader.

Recommended action:

- Import the bucket-diagnostic method, not the hard-subset training recipe.
- Future event experiments should use loss weighting or curriculum over the
  full manifest instead of training only on a hard failure subset.

Risk:

- Bucket-only gains can overfit and hide full-test regression.

## P1 Candidates

### v2e event generation workflow

Source:

- `external_hybrid_package/docs/prj/20260410_140000_v2e_event_generation_experiment_workflow.md`
- `external_hybrid_package/exps/scripts/*v2e*`

Recommended action:

- Keep as an offline data-generation branch for synthetic event supervision.
- Do not make it the default event baseline until raw-event and generated-event
  distributions are compared.

### Mode2 target-FPS and interpolation configs

Source:

- `external_hybrid_package/exps/configs/mode2_stage1_optimizer_pool.yaml`
- `external_hybrid_package/exps/configs/mode2_stage2_lazy_2000fps.yaml`
- `external_hybrid_package/exps/scripts/*target_fps*`

Recommended action:

- Preserve configs as experiment templates for dense event/frame supervision.
- Require explicit manifest fields for source FPS, generated FPS, and interval
  construction before comparing with raw event-count results.

## Do Not Import As Active Event Baselines

- Event runs that do not state event-window policy.
- Hard-subset recovery as a main training recipe.
- Synthetic v2e results without raw-event distribution checks.

## Smoke commands (AM-080)

Both launchers accept an explicit config path, so they can be invoked from any
working directory. A bare file name still resolves under `configs/`.

```bash
export TMPDIR=/tmp TEMP=/tmp TMP=/tmp

# train smoke
python algorithm/common/scripts/train.py   --config algorithm/event/configs/DavisEyeEllipse_EPNet_local_smoke.yaml

# validate smoke
python algorithm/common/scripts/validate.py   --config algorithm/event/configs/DavisEyeEllipse_EPNet_local_smoke.yaml
```

Contract tests for every config in this modality live in
`algorithm/event/tests/test_event_config_contracts.py`. They load each
YAML and assert that the referenced model and dataset names are registered in
`eveye.engine.model_factory` and `eveye.dataset.dataset_factory`.
