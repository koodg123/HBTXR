# FACET Codebase and Paper Analysis

Date: 2026-06-25

## 1. Executive Summary

This report documents the FACET reference codebase under:

- Code: `references/codebase/software/FACET`
- Paper: `references/papers/software/FACET_Fast_and_Accurate_Event-Based_Eye_Tracking_Using_Ellipse_Modeling_for_Extended_Reality.pdf`

FACET is an event-based eye tracking system that predicts pupil ellipse parameters directly from event data. The paper describes an end-to-end detector using fixed-count event binning, fast causal event volume, a MobileNetV3 + FPN/DSC network, ellipse-specific detection heads, and trigonometric loss for rotation discontinuity.

The codebase mostly contains the paper-described system. The central implementation is `EPNet`, configured by `configs/DavisEyeEllipse_EPNet.yaml`. It consumes a two-channel event frame representation and predicts ellipse-related outputs through `hm`, `ab`, `trig`, `reg`, and `mask` heads.

Important mismatch: the paper describes four output heads, but the inspected code uses an additional `mask` head and `mask_loss`. Also, the exact paper claim of fast causal event volume with limit `l=25` is only partially traceable. The code includes causal accumulation and clipping, but the active EPNet config uses `causal_linear_ori`, and the clipping branch uses `maxcount=255`, not `25`.

## 2. Repository Scale

Observed codebase size, excluding `.git`:

- Python files: 108
- Python LOC: about 12,572
- YAML config files: 9
- YAML LOC: about 621
- Jupyter notebooks: 32

Main directories:

- `EvEye/model`: model families and Lightning modules
- `EvEye/dataset`: dataset loaders and dataset factory
- `EvEye/utils`: event representation, caching, visualization, processing utilities
- `tools`: training, validation, repeated validation, inference scripts
- `configs`: experiment and runtime YAML configs
- `tests`: mostly visualization and notebook-style exploration scripts

## 3. Paper-To-Code Mapping

| Paper claim | Paper cue | Code evidence | Assessment |
|---|---|---|---|
| FACET directly predicts pupil ellipses from event data | Abstract, Fig. 2 | `configs/DavisEyeEllipse_EPNet.yaml`, `EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`, `EvEye/model/DavisEyeEllipse/EPNet/Predict.py` | Mostly implemented. EPNet consumes two-channel event input and post-processes `xs`, `ys`, `ab`, `ang` into an ellipse. |
| MobileNetV3 backbone + FPN/DSC lightweight architecture | Method IV-B | `EPNet.py`, `Backbone/MobileNetV3Backbone.py` | Implemented. MobileNetV3 returns `out2..out5`; EPNet performs FPN-style top-down fusion. `fpn_dw` uses depthwise-convolution helpers. |
| Four heads: center heatmap, offset, size, rotation | Fig. 2 / Heads | `Head/EPHead.py`, `DavisEyeEllipse_EPNet.yaml` | Partially matched. Code uses `hm`, `ab`, `trig`, `reg`, plus an extra `mask` head. |
| Rotation is represented by `(sin(2theta), cos(2theta))` | Loss section / Table I | `DavisEyeEllipseDataset.py`, `EPNet/Loss.py`, `EPNet/Predict.py` | Implemented. Dataset creates trig targets; loss uses MSE-style trig loss; prediction restores angle through `atan2`. |
| Total loss includes heatmap, offset, size, Gaussian IoU, trigonometric loss | Eq. 6 / Method IV-C | `EPNet/Loss.py`, `DavisEyeEllipse_EPNet.yaml` | Mostly implemented, with extra `mask_loss`. `ang_loss` remains in code but has weight 0 in the EPNet config. |
| EV-Eye labels are expanded from masks to ellipses using U-Net | Dataset section / Fig. 1 | `README.md`, `configs/DavisEyeEllipse_RGBUNet.yaml`, utility notebooks/scripts | Workflow is present. Exact paper split sizes are not encoded in the inspected YAML configs. |
| Fixed-count binning with 5000 events | Fig. 2 / Table III | `DavisEyeEllipseDataset.py`, `DavisEyeEllipse_EPNet.yaml` | Implemented. `load_event_segment(..., 5000)` is hardcoded in the dataset path. |
| Fast causal event volume with limit `l=25` | Algorithm 1 / Training details | `ToFrameStack.py`, `CutMaxCount.py`, `DavisEyeEllipseDataset.py` | Partial mismatch. Code has causal accumulation and clipping, but the active config/code path does not clearly implement paper's `l=25`. |
| Metrics include P10/P5/P1 and pixel error | Table II | `EPNet/Metric.py`, `EPNet.py` validation step | Implemented. Code logs `val_p10_acc`, `val_p5_acc`, `val_p1_acc`, and `val_mean_distance`; it also computes P3, IoU, and AP. |
| Reported paper numbers: P1 99.59%, PE 0.2030, 3.92M params, 3.44 GFLOPs, 0.5302ms | Table II | Timing/profile hooks in `model_factory.py` and `Predict.py` | Only measurement hooks were found. No clean committed reproduction artifact for Table II was found. TensorRT timing is not implemented in the inspected Python path. |

## 4. Execution Entry Points

### `main.py`

`main.py` is not a normal local training CLI. It is a SageMaker launcher. It constructs a SageMaker PyTorch estimator and points it to:

- `entry_point="tools/train.py"`
- `source_dir="./"`
- input channel `root`

The script also contains hardcoded cloud storage and execution role configuration. Treat it as an environment-specific submission script, not a portable training entry point.

### `tools/train.py`

`tools/train.py` is the primary local training script:

```bash
python tools/train.py --config DavisEyeEllipse_EPNet.yaml
```

Flow:

1. Seed Lightning with `42`.
2. Set multiprocessing start method to `spawn`.
3. Load config from `configs/<name>`.
4. If `SM_CHANNEL_ROOT` exists, override train/val dataset `root_path`.
5. Build train and validation dataloaders through `make_dataloader`.
6. Build model through `make_model`.
7. Apply optimizer config when present.
8. Run `lightning.Trainer(...).fit(...)`.

Important portability issue: `Trainer(devices=[2])` is hardcoded.

### `tools/validate.py`

Validation uses:

```bash
python tools/validate.py --config MemmapDavisEyeCenter_TennSt.yaml
```

It loads only the validation dataloader and runs `trainer.validate()` with `config["val"]["ckpt_path"]`.

Important portability issue: it also hardcodes `devices=[2]`.

### `tools/validate10times.py`

This script repeats validation and averages metrics. It can write a result file under a hardcoded local output directory when run.

### `tools/inference.py`

This script is not config-CLI driven. It has a hardcoded absolute config path and writes `submission.csv`. It is currently closer to a one-off experiment script than a reusable inference CLI.

## 5. Config Structure

Training and validation configs generally use this top-level structure:

```yaml
dataloader:
  train:
    dataset:
      type: ...
  val:
    dataset:
      type: ...
model:
  type: ...
train:
val:
logger:
callback:
```

Inference config `TestTextDavisEyeDataset_TennSt.yaml` instead uses:

```yaml
dataset:
model:
test:
```

Factory behavior:

- `EvEye/dataset/dataset_factory.py` maps dataset `type` to a class.
- `EvEye/model/model_factory.py` maps model `type` to a class.
- Both factories remove `type` from the supplied dictionary with `pop`, which can mutate caller-owned config dictionaries.

Potential config compatibility issue:

- `configs/TestMemmapDavisEyeCenter_TennSt.yaml` passes keys such as `max_count` and `spatial_factor` to `MemmapDavisEyeCenterDataset`.
- The inspected constructor expects `fixed_count`, `spatial_downsaple`, `saptial_transform`, and related names.
- Because the factory forwards keyword arguments directly, this config can raise `TypeError` in the current code.

## 6. Dataset And Data Flow

Registered datasets:

- `DavisWithMaskDataset`
- `TestDataset`
- `DavisEyeCenterDataset`
- `NpyDavisEyeCenterDataset`
- `DatDavisEyeCenterDataset`
- `MemmapDavisEyeCenterDataset`
- `TestTextDavisEyeDataset`
- `DavisEyeEllipseDataset`

`CitiBikeDataset` exists in the tree but is not registered in `DATASET_CLASSES`, so YAML factory creation will not instantiate it directly.

### `DavisEyeEllipseDataset`

This is the main FACET ellipse dataset. It expects:

- `root/split/cached_data`
- `root/split/cached_ellipse`

Per sample, it:

1. Loads one ellipse label.
2. Loads an event segment of 5000 events.
3. Applies event augmentation for training.
4. Converts events into a two-channel frame representation.
5. Applies image/ellipse transform replay through Albumentations.
6. Downsamples labels by `down_ratio = 4`.
7. Builds CenterNet-style targets:
   - `hm`
   - `ab`
   - `ang`
   - `trig`
   - `reg`
   - `ind`
   - `reg_mask`
   - `mask`
8. Returns a dictionary with input and targets.

Returned keys:

```text
input, hm, reg_mask, ind, ab, ang, trig, mask, reg, center, close, ellipse
```

### `DavisWithMaskDataset`

This dataset loads grayscale/RGB-style image-mask pairs for segmentation training. It supports the README-described U-Net step used to expand labels from the original mask-labeled subset.

### `DavisEyeCenterDataset` and `MemmapDavisEyeCenterDataset`

These support center-based event eye tracking, including TennSt-style temporal models. They return tuple-style tensors rather than the EPNet dictionary format.

### `TestTextDavisEyeDataset`

This is a single-file inference dataset for a text event stream and label file. `__len__` returns `1`, and `__getitem__` converts the entire event range to frame stacks.

## 7. Model Architecture

Registered model types:

- `DeepLabV3`
- `ConvLSTM`
- `TennSt`
- `EPNet`
- `ElNet`
- `UNet`

### EPNet

`EPNet` is the closest implementation of the FACET paper.

Main components:

- `MobileNetV3Backbone(input_channels=2)`
- FPN-style top-down feature fusion
- `EPHead(in_channels=64, head_dict=...)`
- `CtdetLoss`
- Lightning training and validation steps

Supported EPNet modes include:

- `standard`
- `light`
- `fpn_2d`
- `fpn_dw`

The paper emphasizes depthwise separable convolution in FPN. The code has a dedicated `fpn_dw` mode for depthwise convolution. The active `DavisEyeEllipse_EPNet.yaml` inspected during analysis uses `mode: fpn_2d`, which is not the depthwise variant.

### EPHead

The default head dictionary is:

```python
{"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1}
```

This means:

- `hm`: center heatmap
- `ab`: ellipse major/minor axes
- `trig`: `(sin(2theta), cos(2theta))`
- `reg`: center offset
- `mask`: auxiliary ellipse mask prediction

### Loss

`CtdetLoss` combines:

- focal heatmap loss
- `ab` regression loss
- offset regression loss
- optional angle loss
- trigonometric loss
- Gaussian/Wasserstein distance loss
- mask loss

The paper equation does not include mask loss, so this is an implementation extension or experiment variant.

## 8. Utility Layer

Important utility modules:

- `EvEye/utils/processor/TxtProcessor.py`: event/label/ellipse text parsing
- `EvEye/utils/cache/MemmapCacheStructedEvents.py`: event/label/ellipse memmap caching and loading
- `EvEye/utils/tonic/functional/ToFrameStack.py`: event-to-frame-stack conversion
- `EvEye/utils/tonic/functional/CutMaxCount.py`: clipping/normalization helper
- `EvEye/utils/visualization/visualization.py`: event and ellipse visualization helpers
- `EvEye/utils/PupilTracker.py`: ellipse-based tracking helper separate from EPNet detection

Security note: `MemmapCacheStructedEvents.load_memmap()` parses dtype metadata with `eval(dtype_str)`. This should be replaced with a safe dtype parser or structured metadata format before using untrusted cache metadata.

## 9. Quality And Reproducibility Risks

### High: broken import in Gaussian/Wasserstein loss path

`EPNet/Loss.py` uses:

```python
from predict import _topk
```

The inspected tree contains `Predict.py`, not lowercase `predict.py`, and the function exposed there is `topk`, not `_topk`. Since `DavisEyeEllipse_EPNet.yaml` sets `iou_weight: 15`, this path can fail during training.

### High: hardcoded CUDA calls

Several loss paths create tensors using `.cuda()` directly. This breaks CPU execution, non-CUDA devices, and some distributed or mixed-device contexts. Device should be inherited from existing tensors, for example `device=pred.device` or `.to(pred.device)`.

### Medium: hardcoded GPU index

`tools/train.py` and `tools/validate.py` use `devices=[2]`. This should be moved to config/CLI and default to auto or a user-provided device.

### Medium: hardcoded local paths

The code and configs contain many machine-specific paths such as `/mnt/data2T/...`. These should be parameterized.

### Medium: cloud-specific values in launcher

`main.py` includes cloud-specific SageMaker/S3/IAM configuration. This should be moved to environment variables or a private deployment config, not kept as reusable code defaults.

### Medium: config/schema drift

At least one config appears to pass keyword names that do not match the target dataset constructor. This suggests the configs and dataset APIs drifted during experimentation.

### Low: factory mutation

`make_model()` and `make_dataset()` mutate config dictionaries with `pop("type")`. This is fragile if the same config dictionary is reused.

## 10. Recommended Fix Order

1. Fix definite runtime breakages:
   - Replace the incorrect `from predict import _topk` path.
   - Fix any nonexistent imports such as stale cache import paths.
2. Remove hard `.cuda()` calls in loss and representation code.
3. Make `devices`, dataset roots, checkpoint paths, and output paths config/CLI-driven.
4. Normalize config schemas and validate YAML against constructors.
5. Make factories non-mutating by copying config dictionaries before `pop`.
6. Replace `eval(dtype_str)` in memmap loading.
7. Add import smoke tests for every registered dataset/model factory key.
8. Add a small synthetic EPNet forward/loss test.
9. Add a reproduction note for paper Table II, including exact checkpoint, config, TensorRT export path, and timing procedure.

## 11. Verification Performed

Static syntax verification was run for all Python files:

```bash
PYTHONPYCACHEPREFIX=/tmp/facet_pycache python3 -m py_compile $(find references/codebase/software/FACET -path '*/.git/*' -prune -o -type f -name '*.py' -print)
```

Result: syntax compilation passed.

Full training, validation, or inference was not run because the inspected code depends on external datasets, checkpoints, hardcoded local paths, GPU index assumptions, and environment-specific dependencies.

## 12. Sub-Agent And Evidence Notes

Three read-only explorer subtasks were used during the analysis phase:

- Execution/config/dataset flow
- Model hierarchy and quality risks
- Paper-to-code claim mapping

The final report integrates those outputs with direct file inspection and static syntax verification.

## 13. Remaining Uncertainties

- The exact implementation of the paper's fast causal event volume with `l=25` was not found in the active EPNet config path.
- The paper's four-head description differs from the inspected five-output code path with an auxiliary mask head.
- The paper's published benchmark numbers are not independently reproduced by committed result artifacts in the inspected code tree.
- Notebook-only workflows may contain additional evidence, but the core Python/YAML path already shows enough drift to require explicit reproduction documentation.
