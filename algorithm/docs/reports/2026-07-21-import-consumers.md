# Algorithm Import Consumers

## Contract and measurement boundary

AM-010 establishes the packaging contract before implementation ownership moves.
The measurement is locked to commit
`ebe862b11506e819c5bd5abc925299fc5fbb6f1a` and tracked Python files
under `algorithm/common/**`, `algorithm/frame/**`, `algorithm/event/**`,
and `algorithm/hybrid/**`. An import counts only when a line begins, after
optional whitespace, with `from` or `import` and the exact top-level name.

| Import surface | Lines | Files | AM decision |
|---|---:|---:|---|
| maintained `EvEye.*` | 176 | 57 | migrate through AM-020 to AM-050; retire only at AM-060 gates |
| maintained Hybrid `src.*` | 302 | 96 | retain Hybrid namespace; only the proven `src.pools` split is later in scope |

Repository-external consumers are unknown, not zero. This checkout cannot by
itself authorize removal of `EvEye` or retirement of the Hybrid `src` namespace.

The maintained census excludes frozen or non-runtime evidence, including
`references/**`, `algorithm/analysis/**`, `algorithm/archive/**`,
`algorithm/hybrid/hardware_reference/**`, modality analysis documents, Hybrid
handover tests, provenance, and historical tracking records.
`algorithm/analysis/**` contains a separately retained 15 `EvEye` import
lines in 7 files; those lines are evidence, not migration consumers.

## Packaging contract and case safety

The five `where` roots define the scan boundary, but they are not sufficient
to associate repeated `eveye` namespace portions with their physical owners.
AM-010 therefore maps all six package families explicitly:

| Package family | Physical source |
|---|---|
| `EvEye` | `common/src/EvEye` |
| `eveye.common` | `common/src/eveye/common` |
| `eveye.dataset` | `dataset/src/eveye/dataset` |
| `eveye.utils` | `utils/src/eveye/utils` |
| `eveye.engine` | `engine/src/eveye/engine` |
| `eveye.event` | `event/src/eveye/event` |


Discovery uses paired exact/descendant boundaries for each approved family:
`EvEye` plus `EvEye.*` and the exact name plus `.*` for each
`eveye.common`, `eveye.dataset`, `eveye.utils`, `eveye.engine`, and
`eveye.event`. Prefix lookalikes such as `EvEye2`,
`eveye.common_extra`, and `eveye.engineering` are outside the contract.

AM-010 synthetic packaging validation materializes a nested representative
package under every mapping plus those three lookalikes. The built wheel and
clean install contain/import all six approved families, while the lookalikes
and bare `dataset`, `utils`, and `engine` names remain absent.

`EvEye` and `eveye` are a case-only top-level-name transition and cannot be
validated safely when Python or pip stages files on a case-insensitive
filesystem. On this host, the default Python/pip temporary directory resolved
under `/mnt/c/...` and the dual-case synthetic validation failed as expected
because `EvEye` and `eveye` folded together. The same validation passed
when `TMPDIR`, `TEMP`, and `TMP` all pointed to one explicit
Linux-filesystem directory under `/tmp`.

No intermediate source-migration commit may contain both case-only top-level
wheel payloads. Migration slices and packaging validation must use the forced
Linux temporary environment below. AM-060 removes `EvEye` before the
migration checkpoint is committed or checked out on a case-insensitive host.

### Reproduce the Linux-temp packaging gate

Run from the repository root. This creates only disposable files under
`/tmp`.

```bash
am010_linux_tmp=$(mktemp -d /tmp/hbtxr-am010.XXXXXX)
export TMPDIR="$am010_linux_tmp/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
am010_project="$am010_linux_tmp/project"
mkdir -p "$TMPDIR" "$am010_project" "$am010_linux_tmp/dist"
cp algorithm/pyproject.toml "$am010_project/pyproject.toml"

python3 -c '
from pathlib import Path
import sys

root = Path(sys.argv[1])
approved = {
    "EvEye": root / "common/src/EvEye",
    "common": root / "common/src/eveye/common",
    "dataset": root / "dataset/src/eveye/dataset",
    "utils": root / "utils/src/eveye/utils",
    "engine": root / "engine/src/eveye/engine",
    "event": root / "event/src/eveye/event",
}
lookalikes = (
    root / "common/src/EvEye2",
    root / "common/src/eveye/common_extra",
    root / "engine/src/eveye/engineering",
)
for name, package in approved.items():
    nested = package / "nested"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "OWNER = " + repr(name) + "\n"
    )
    (nested / "__init__.py").write_text(
        "NESTED_OWNER = " + repr(name) + "\n"
    )
for package in lookalikes:
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("LOOKALIKE = True\n")
' "$am010_project"

python3 -m pip wheel --no-deps --no-build-isolation \
  -w "$am010_linux_tmp/dist" "$am010_project"
python3 -m venv "$am010_linux_tmp/venv"
"$am010_linux_tmp/venv/bin/python" -m pip install --no-deps \
  "$am010_linux_tmp"/dist/hbtxr_algorithm-*.whl

(
  cd "$am010_linux_tmp"
  "$am010_linux_tmp/venv/bin/python" -I -c '
from importlib.util import find_spec
import EvEye.nested
import eveye.common.nested
import eveye.dataset.nested
import eveye.engine.nested
import eveye.event.nested
import eveye.utils.nested

assert all(
    find_spec(name) is None
    for name in (
        "EvEye2",
        "eveye.common_extra",
        "eveye.engineering",
        "dataset",
        "utils",
        "engine",
    )
)
'
)
```

## Reproduction commands

Run from the repository root. These commands read the approved baseline commit
rather than the mutable worktree.

```bash
baseline_sha=ebe862b11506e819c5bd5abc925299fc5fbb6f1a
ev_pattern='^[[:space:]]*(from[[:space:]]+EvEye([.]|[[:space:]]|$)|import[[:space:]]+EvEye([.]|[[:space:]]|$))'
src_pattern='^[[:space:]]*(from[[:space:]]+src([.]|[[:space:]]|$)|import[[:space:]]+src([.]|[[:space:]]|$))'
active_globs=(
  'algorithm/common/**/*.py'
  'algorithm/frame/**/*.py'
  'algorithm/event/**/*.py'
  'algorithm/hybrid/**/*.py'
)

git grep -n -E "$ev_pattern" "$baseline_sha" -- "${active_globs[@]}"
git grep -l -E "$ev_pattern" "$baseline_sha" -- "${active_globs[@]}"
git grep -n -E "$src_pattern" "$baseline_sha" -- "${active_globs[@]}"
git grep -l -E "$src_pattern" "$baseline_sha" -- "${active_globs[@]}"
```

Expected assertions are `176/57` for `EvEye` and `302/96` for Hybrid
`src`. The maintained-current-tree review command for AM-010 is:

```bash
maintained_pattern='^[[:space:]]*(from|import)[[:space:]]+(EvEye|dataset|utils|engine)([.[:space:]]|$)'
rg -n "$maintained_pattern" \
  algorithm \
  --glob '!algorithm/analysis/**' \
  --glob '!algorithm/archive/**' \
  --glob '!algorithm/hybrid/hardware_reference/**'
```

## Maintained consumer categories

### EvEye

| Category | Lines | Files |
|---|---:|---:|
| `algorithm/common/scripts` | 20 | 4 |
| `algorithm/common/src/EvEye/callback` | 1 | 1 |
| `algorithm/common/src/EvEye/dataset` | 43 | 11 |
| `algorithm/common/src/EvEye/model` | 39 | 14 |
| `algorithm/common/src/EvEye/utils` | 46 | 23 |
| `algorithm/common/tests` | 27 | 4 |
| **Total** | **176** | **57** |

<details>
<summary>Exact EvEye consumer files (57)</summary>

- `algorithm/common/scripts/facet/inference.py`
- `algorithm/common/scripts/facet/train.py`
- `algorithm/common/scripts/facet/validate.py`
- `algorithm/common/scripts/facet/validate10times.py`
- `algorithm/common/src/EvEye/callback/callback_factory.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeCenter/DatDavisEyeCenterDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeCenter/DavisEyeCenterDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeCenter/MemmapDavisEyeCenterDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeCenter/NpyDavisEyeCenterDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeCenter/TestTextDavisEyeDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseCenterSequenceDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseFrameDataset.py`
- `algorithm/common/src/EvEye/dataset/DavisWithMask/DavisWithMaskDataset.py`
- `algorithm/common/src/EvEye/dataset/Test/TestDataset.py`
- `algorithm/common/src/EvEye/dataset/dataset_factory.py`
- `algorithm/common/src/EvEye/model/DavisEyeCenter/TennSt.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/EPNet/Loss.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/EPNet/Predict.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/ElNet/ElNet.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/EllipseMobileNet.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/HBTXR/HBTXR.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/HBTXR/Loss.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/HBTXR/Predict.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/UNet/Predict.py`
- `algorithm/common/src/EvEye/model/DavisEyeEllipse/UNet/UNet.py`
- `algorithm/common/src/EvEye/model/DavisWithMask/DeepLabV3.py`
- `algorithm/common/src/EvEye/model/DavisWithMask/UNet.py`
- `algorithm/common/src/EvEye/model/model_factory.py`
- `algorithm/common/src/EvEye/utils/cache/MemmapCacheFrameStack.py`
- `algorithm/common/src/EvEye/utils/cache/MemmapCacheStructedEvents.py`
- `algorithm/common/src/EvEye/utils/cache/NpyCacheFrameStack.py`
- `algorithm/common/src/EvEye/utils/dvs_common_utils/base/EventsIterator.py`
- `algorithm/common/src/EvEye/utils/dvs_common_utils/representation/FrameStack.py`
- `algorithm/common/src/EvEye/utils/dvs_common_utils/representation/Histgram.py`
- `algorithm/common/src/EvEye/utils/dvs_common_utils/representation/TimeSurface.py`
- `algorithm/common/src/EvEye/utils/processor/TxtProcessor.py`
- `algorithm/common/src/EvEye/utils/scripts/benchmark_hbtxr_frame_dataloader.py`
- `algorithm/common/src/EvEye/utils/scripts/build_cached_frames_for_dean.py`
- `algorithm/common/src/EvEye/utils/scripts/build_full_dean_dataset_with_unet.py`
- `algorithm/common/src/EvEye/utils/scripts/build_reproduction_summary.py`
- `algorithm/common/src/EvEye/utils/scripts/compare_model_evaluation_results.py`
- `algorithm/common/src/EvEye/utils/scripts/evaluate_epnet_checkpoint.py`
- `algorithm/common/src/EvEye/utils/scripts/evaluate_hbtxr_frame_motion_package.py`
- `algorithm/common/src/EvEye/utils/scripts/evaluate_hbtxr_val_motion.py`
- `algorithm/common/src/EvEye/utils/scripts/export_hbtxr_subject_independent_for_targets.py`
- `algorithm/common/src/EvEye/utils/scripts/find_center.py`
- `algorithm/common/src/EvEye/utils/scripts/h5_to_png.py`
- `algorithm/common/src/EvEye/utils/scripts/measure_hbtxr_target_complexity.py`
- `algorithm/common/src/EvEye/utils/scripts/output_groundtruth.py`
- `algorithm/common/src/EvEye/utils/scripts/probe_hbtxr_batch_size.py`
- `algorithm/common/src/EvEye/utils/scripts/validate_reproduction_artifact.py`
- `algorithm/common/tests/facet/visualize_TennSt_model_test.py`
- `algorithm/common/tests/facet/visualize_events_frame.py`
- `algorithm/common/tests/facet/visualize_events_test.py`
- `algorithm/common/tests/facet/visualize_model_test.py`

</details>

### Hybrid src

| Category | Lines | Files |
|---|---:|---:|
| `algorithm/hybrid/scripts` | 30 | 15 |
| `algorithm/hybrid/src` root facades | 8 | 4 |
| `algorithm/hybrid/src/config` | 1 | 1 |
| `algorithm/hybrid/src/data` | 11 | 6 |
| `algorithm/hybrid/src/loss` | 48 | 12 |
| `algorithm/hybrid/src/models` | 2 | 2 |
| `algorithm/hybrid/src/optim` | 18 | 4 |
| `algorithm/hybrid/src/pools` | 12 | 5 |
| `algorithm/hybrid/src/preprocess` | 65 | 19 |
| `algorithm/hybrid/src/runtime` | 2 | 1 |
| `algorithm/hybrid/src/training` | 16 | 4 |
| `algorithm/hybrid/src/utils` | 1 | 1 |
| `algorithm/hybrid/tests` | 88 | 22 |
| **Total** | **302** | **96** |

<details>
<summary>Exact Hybrid src consumer files (96)</summary>

- `algorithm/hybrid/scripts/external_pipeline/_config.py`
- `algorithm/hybrid/scripts/external_pipeline/annotate_groundedsam_ev_eye.py`
- `algorithm/hybrid/scripts/external_pipeline/build_groundedsam_dataset.py`
- `algorithm/hybrid/scripts/external_pipeline/build_mode_manifests.py`
- `algorithm/hybrid/scripts/external_pipeline/canonicalize_hbtxr.py`
- `algorithm/hybrid/scripts/external_pipeline/check_dataloader.py`
- `algorithm/hybrid/scripts/external_pipeline/eval_hbtxr.py`
- `algorithm/hybrid/scripts/external_pipeline/export_hbtxr.py`
- `algorithm/hybrid/scripts/external_pipeline/infer_hbtxr.py`
- `algorithm/hybrid/scripts/external_pipeline/prepare_ev_eye.py`
- `algorithm/hybrid/scripts/external_pipeline/relocate_ev_eye_dataset.py`
- `algorithm/hybrid/scripts/external_pipeline/sync_packages.py`
- `algorithm/hybrid/scripts/external_pipeline/train_hbtxr.py`
- `algorithm/hybrid/scripts/external_pipeline/visualize_dataset.py`
- `algorithm/hybrid/scripts/external_pipeline/visualize_inference_results.py`
- `algorithm/hybrid/src/config/run_contract.py`
- `algorithm/hybrid/src/data/components.py`
- `algorithm/hybrid/src/data/dataset.py`
- `algorithm/hybrid/src/data/event_builder.py`
- `algorithm/hybrid/src/data/loader.py`
- `algorithm/hybrid/src/data/session_reader.py`
- `algorithm/hybrid/src/data/utils.py`
- `algorithm/hybrid/src/loss/bundles/eye.py`
- `algorithm/hybrid/src/loss/bundles/search_event.py`
- `algorithm/hybrid/src/loss/bundles/shared.py`
- `algorithm/hybrid/src/loss/bundles/track.py`
- `algorithm/hybrid/src/loss/common.py`
- `algorithm/hybrid/src/loss/distillation.py`
- `algorithm/hybrid/src/loss/metrics.py`
- `algorithm/hybrid/src/loss/primitives.py`
- `algorithm/hybrid/src/loss/stage.py`
- `algorithm/hybrid/src/loss/stage1.py`
- `algorithm/hybrid/src/loss/stage2.py`
- `algorithm/hybrid/src/loss/stage_common.py`
- `algorithm/hybrid/src/loss_common.py`
- `algorithm/hybrid/src/loss_distillation.py`
- `algorithm/hybrid/src/loss_primitives.py`
- `algorithm/hybrid/src/loss_stage.py`
- `algorithm/hybrid/src/models/tracker/encoder.py`
- `algorithm/hybrid/src/models/tracker/registry.py`
- `algorithm/hybrid/src/optim/__init__.py`
- `algorithm/hybrid/src/optim/musgd.py`
- `algorithm/hybrid/src/optim/pool.py`
- `algorithm/hybrid/src/optim/registry.py`
- `algorithm/hybrid/src/pools/__init__.py`
- `algorithm/hybrid/src/pools/heads.py`
- `algorithm/hybrid/src/pools/losses.py`
- `algorithm/hybrid/src/pools/optimizers.py`
- `algorithm/hybrid/src/pools/runtime_schedulers.py`
- `algorithm/hybrid/src/preprocess/annotation_backends.py`
- `algorithm/hybrid/src/preprocess/annotation_groundedsam.py`
- `algorithm/hybrid/src/preprocess/build_manifests.py`
- `algorithm/hybrid/src/preprocess/canonicalize.py`
- `algorithm/hybrid/src/preprocess/canonicalize_pipeline.py`
- `algorithm/hybrid/src/preprocess/event_generation.py`
- `algorithm/hybrid/src/preprocess/groundedsam_build.py`
- `algorithm/hybrid/src/preprocess/groundedsam_eye_region_bbox.py`
- `algorithm/hybrid/src/preprocess/groundedsam_pipeline.py`
- `algorithm/hybrid/src/preprocess/interpolation.py`
- `algorithm/hybrid/src/preprocess/io_utils.py`
- `algorithm/hybrid/src/preprocess/path_utils.py`
- `algorithm/hybrid/src/preprocess/raw_ellipse_blink.py`
- `algorithm/hybrid/src/preprocess/relocate_dataset.py`
- `algorithm/hybrid/src/preprocess/target_fps_build.py`
- `algorithm/hybrid/src/preprocess/target_fps_canonical.py`
- `algorithm/hybrid/src/preprocess/timelens_xl_finetune.py`
- `algorithm/hybrid/src/preprocess/timelens_xl_native_entry.py`
- `algorithm/hybrid/src/preprocess/v2e_experiment.py`
- `algorithm/hybrid/src/runtime/tracker.py`
- `algorithm/hybrid/src/training/losses.py`
- `algorithm/hybrid/src/training/model_factory.py`
- `algorithm/hybrid/src/training/step_runner.py`
- `algorithm/hybrid/src/training/trainer.py`
- `algorithm/hybrid/src/utils/io.py`
- `algorithm/hybrid/tests/external_pipeline/conftest.py`
- `algorithm/hybrid/tests/external_pipeline/test_config.py`
- `algorithm/hybrid/tests/external_pipeline/test_dataset.py`
- `algorithm/hybrid/tests/external_pipeline/test_external_packages.py`
- `algorithm/hybrid/tests/external_pipeline/test_groundedsam_eye_region_bbox.py`
- `algorithm/hybrid/tests/external_pipeline/test_interpolation.py`
- `algorithm/hybrid/tests/external_pipeline/test_loss_catalog.py`
- `algorithm/hybrid/tests/external_pipeline/test_loss_stage_module_split.py`
- `algorithm/hybrid/tests/external_pipeline/test_model.py`
- `algorithm/hybrid/tests/external_pipeline/test_output_contract_snapshot.py`
- `algorithm/hybrid/tests/external_pipeline/test_preprocess.py`
- `algorithm/hybrid/tests/external_pipeline/test_pretrained_loader.py`
- `algorithm/hybrid/tests/external_pipeline/test_raw_ellipse_blink.py`
- `algorithm/hybrid/tests/external_pipeline/test_runtime_e2e.py`
- `algorithm/hybrid/tests/external_pipeline/test_script_workflows.py`
- `algorithm/hybrid/tests/external_pipeline/test_target_fps_build.py`
- `algorithm/hybrid/tests/external_pipeline/test_target_fps_canonical.py`
- `algorithm/hybrid/tests/external_pipeline/test_target_fps_dataset.py`
- `algorithm/hybrid/tests/external_pipeline/test_target_fps_manifest.py`
- `algorithm/hybrid/tests/external_pipeline/test_timelens_xl_finetune.py`
- `algorithm/hybrid/tests/external_pipeline/test_train_pipeline.py`
- `algorithm/hybrid/tests/external_pipeline/test_v2e_experiment.py`

</details>

## Future canonical mappings

The migration replaces implementation ownership, not behavior:

| Current import family | Future canonical family |
|---|---|
| `EvEye.dataset.*` | `eveye.dataset.*` |
| `EvEye.utils.scripts.*` | `eveye.engine.tools.*` |
| remaining `EvEye.utils.*` | `eveye.utils.*` |
| `EvEye.callback.*` | `eveye.engine.callback.*` |
| `EvEye.logger.*` | `eveye.engine.logger.*` |
| `EvEye.model.model_factory` | `eveye.engine.model_factory` |
| EPNet, ElNet, and TennSt under `EvEye.model.*` | `eveye.event.models.*` |
| remaining shared `EvEye.model.*` | `eveye.common.models.*` |

Hybrid keeps its `src.*` namespace. Later pool-owner mappings are
`src.pools.losses` to `src.loss.losses`, `src.pools.optimizers` to
`src.optim.optimizer`, `src.pools.lr_schedulers` to
`src.optim.lr_schedulers`, and `src.pools.runtime_schedulers` to
`src.runtime.runtime_schedulers`. `src.pools.heads` and the pool facade are
removed only after their own consumer/parity gates; `src.optim.pool` remains
preserved.

## AM-060 EvEye retirement gates

`EvEye` remains the implementation package at AM-010. It may be removed only
when all four gates pass:

1. Maintained `EvEye` import count is zero.
2. Required repository-external consumer count is zero or those consumers are
   separately migrated.
3. The wheel contains `eveye/*` and no `EvEye/*` payload.
4. Arbitrary-CWD import, editable-install import, and direct-script `--help`
   checks all pass.

At AM-010 the real source tree contains only the current `EvEye` implementation;
the explicit mappings predeclare the future owners without creating placeholder
packages. The current wheel therefore contains `EvEye/*` only, while the
representative synthetic wheel proves that all future mapped portions can be
built and imported together on the Linux validation filesystem. Bare
`dataset`, `utils`, and `engine` payloads remain forbidden.

The paired exact/descendant include list is a transient build contract, not
permission to commit a case-unsafe dual-payload migration state. AM-060 must remove `EvEye` and narrow
the include list before the migration checkpoint crosses to a case-insensitive
host.
