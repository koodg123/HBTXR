# Frame Experiment Import Candidates

This note curates experiment results that are worth carrying into the frame-only
algorithm track. Raw source documents are not bulk-copied here; this file records
what should be imported, why it matters, and what should stay as provenance only.

## Source Scope

- `external_hybrid_package/docs/exps/*`
- `external_hybrid_package/exps/configs/*`
- `hgtxr_software/docs/resources/current_stage1_frame_search_baseline_manifest.json`
- `hgtxr_software/docs/resources/stage1_frame_search_*`

## P0 Candidates

### Stage1 full-sensor letterbox and dense eye validation

Source:

- `external_hybrid_package/docs/exps/20260329_022334_12_mode0_stage1_experiment_results_analysis.md`

Useful result:

- `sensor_full_letterbox` fixed the aspect-ratio issue that made eye-region
  supervision unreliable.
- The eye-only full-sensor run produced meaningful ROI regression:
  best validation `loss_eye = 10.081159` at epoch `10`, and overlay mean IoU
  `0.663755`.

Recommended action:

- Preserve this as the frame track's default preprocessing constraint:
  sensor-space accumulation or frame loading first, then letterbox/crop.
- Do not mix older distorted transform metrics with current frame baselines.

Risk:

- The result validates ROI geometry, not final pupil-center accuracy by itself.

### Stage1 mask-guided center localization

Source:

- `external_hybrid_package/docs/exps/20260402_010646_15_mode0_stage1_mask_guidance_ablation_analysis.md`

Useful result:

| Variant | Best search P10 | Best search P5 | Best center px | Best eye IoU | Judgment |
|---|---:|---:|---:|---:|---|
| Baseline | `22.29` | `6.42` | `17.28` | `0.7446` | reference |
| Hard mask centroid | `99.63` | `98.89` | `1.82` | `0.7594` | strongest center solver |
| Soft center consistency | `27.25` | `8.86` | `16.58` | `0.7348` | clean conservative improvement |
| Soft center + axis | `20.26` | `5.76` | `19.85` | `0.7517` | not worth carrying forward |
| Soft mask cascade | `98.75` | `88.98` | `3.02` | `0.7042` | strong but less clean mask quality |

Recommended action:

- Keep hard mask-centroid routing as an upper-bound or teacher target.
- Keep soft center consistency as the safer trainable frame baseline.
- Treat mask cascade as a controlled follow-up only after its early instability
  is protected by tests.
- Drop the axis-consistency variant unless a new ellipse-axis loss is introduced.

Risk:

- Hard mask-centroid performance depends on high-quality mask predictions, so it
  should not be treated as a deployable pupil head without fallback logic.

### Stage1 bbox cascade experiments

Source:

- `external_hybrid_package/docs/exps/20260408_105839_17_mode0_stage1_expE_bbox_head_experiments.md`

Useful result:

| Variant | Best eye IoU | Best eye center px | Best pupil bbox IoU | Best pupil bbox center px | Judgment |
|---|---:|---:|---:|---:|---|
| ROI Align cascade | `0.0` | `183.00` | `0.05995` | `32.07` | failed eye stage |
| No-ROIAlign cascade | `0.0` | `132.90` | `0.05906` | `32.03` | still failed eye stage |
| Dense-eye guided cascade | `0.7311` | `14.36` | `0.10013` | `27.18` | usable eye provider, weak pupil refinement |

Recommended action:

- Keep dense-eye guided cascade as a frame experiment branch.
- Do not import the weak MLP eye-stage variants except as negative ablation
  evidence.
- If revisited, focus on the second-stage pupil refinement head, because the
  dense-eye provider already solves the upstream collapse.

Risk:

- Pupil bbox IoU remains weak, so this branch should not replace mask-guided
  center localization.

### Stage1 frame-search baseline from HGTXR software

Source:

- `hgtxr_software/docs/resources/current_stage1_frame_search_baseline_manifest.json`

Useful result:

- Active baseline is a center-polish self-distillation frame search checkpoint.
- Promoted follow-up metrics:
  `best_search_p10 = 28.27717937613433`,
  `best_search_p5 = 10.153863744915656`,
  `best_search_center_px = 17.257583906065744`.
- A no-distillation candidate had slightly higher P10
  (`28.303010526693093`) but failed center preservation.

Recommended action:

- Import the promotion rule, not just the checkpoint name:
  frame P10 improvements must preserve or improve center error.
- Use this as a model for future frame-only experiment manifests.

Risk:

- This baseline belongs to the HGTXR software line and may use different
  manifests or metric protocol. Keep metric bridge checks before direct
  comparison.

## P1 Candidates

- Grounded-SAM ROI-crop prompted label generation:
  useful for frame label preparation, not runtime inference.
- Target-FPS and TimeLens preprocessing configs:
  useful only if dense temporal supervision is explicitly reopened.

## Do Not Import As Active Frame Baselines

- Older distorted-transform Stage1 runs.
- Bbox cascade variants whose eye stage reports `eye_iou = 0.0`.
- Frame metrics that do not record the resize/crop policy.

## Smoke commands (AM-080)

Both launchers accept an explicit config path, so they can be invoked from any
working directory. A bare file name still resolves under `configs/`.

```bash
export TMPDIR=/tmp TEMP=/tmp TMP=/tmp

# train smoke
python algorithm/common/scripts/train.py   --config algorithm/frame/configs/DavisEyeEllipse_RGBUNet_local_train_smoke.yaml

# validate smoke
python algorithm/common/scripts/validate.py   --config algorithm/frame/configs/DavisEyeEllipse_RGBUNet_local_train_smoke.yaml
```

Contract tests for every config in this modality live in
`algorithm/frame/tests/test_frame_config_contracts.py`. They load each
YAML and assert that the referenced model and dataset names are registered in
`engine.model_factory` and `dataset.dataset_factory`.
