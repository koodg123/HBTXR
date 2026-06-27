# FACET Data, Training, Metrics, and Tensor Contract

Date: 2026-06-27

## Scope

This document summarizes the FACET detector path used by `DavisEyeEllipseDataset` with EPNet/HBTXR-style heads. It focuses on:

1. Dataset generation, resolution, and labels.
2. Train/evaluation inputs and outputs.
3. Metric calculation.
4. Model input/output dimensions.

Primary evidence:

- `references/codebase/software/FACET/README.md`
- `references/codebase/software/FACET/EvEye/dataset/DavisEyeEllipse/DavisEyeEllipseDataset.py`
- `references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`
- `references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/EPNet/Predict.py`
- `references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/EPNet/Metric.py`
- `references/report/FACET/FACET_resolution_contract_correction_2026-06-27.md`

## 1. Dataset Generation, Resolution, and Labels

FACET uses EV-Eye style event data and ellipse labels.

The README describes the reference preparation flow:

1. Start from `Data_davis_labelled_with_mask`, which contains labelled `.h5` mask data.
2. Convert `.h5` data into image/mask files.
3. Train a U-Net segmentation model on the labelled subset.
4. Apply the trained U-Net to the remaining `Data_davis` frames to generate masks.
5. Convert masks into ellipse labels.
6. Organize event data and ellipse labels by split.

The detector dataset used for EPNet/HBTXR training expects cached split folders:

```text
<root>/
  train/
    cached_data/
    cached_ellipse/
  val/
    cached_data/
    cached_ellipse/
```

Each event sample loads a fixed 5000-event segment:

```text
event_segment = load_event_segment(index, self.data_path, 5000)
```

The raw event sensor contract is:

```text
sensor_size = (346, 260, 2)  # width, height, polarity/channel count
```

After event accumulation and image transform, the model input resolution is:

```text
input tensor: (B, 2, 256, 256)
```

Important resolution rule:

- `64x64` in the current FACET reproduction context is the final feature map, heatmap, mask, detection-head, and metric resolution.
- It is not the raw model input resolution.
- The dataset uses `default_resolution=[256, 256]` and `down_ratio=4`, therefore `256 / 4 = 64`.

Ellipse labels use a five-parameter shape at the detector output scale:

```text
ellipse = [x, y, a, b, angle]
```

The broader cached ellipse dtype also stores timestamp:

```text
t, x, y, a, b, ang
```

Invalid/closed-eye samples are marked by:

```text
close = 1
```

Valid samples use:

```text
close = 0
```

Only valid samples are used for losses/metrics that require an ellipse target.

## 2. Train and Evaluation Inputs/Outputs

`DavisEyeEllipseDataset.__getitem__()` returns a dictionary with these keys:

```text
input, hm, reg_mask, ind, ab, ang, trig, mask, reg, center, close, ellipse
```

The training step uses:

```python
input_tensor = batch["input"]
pred = model(input_tensor)
loss, loss_show = criterion(pred, batch)
```

The evaluation step uses:

```python
input_tensor = batch["input"]
pred = model(input_tensor)
dets = post_process(pred)
metrics = {
    "val_p10_acc": p_acc(dets, center, close, 10),
    "val_p5_acc": p_acc(dets, center, close, 5),
    "val_p3_acc": p_acc(dets, center, close, 3),
    "val_p1_acc": p_acc(dets, center, close, 1),
    "val_mean_distance": cal_mean_distance(dets, center, close),
    "val_IoU": cal_batch_iou(dets, ellipse, close),
    "val_AP": cal_batch_ap(dets, ellipse, close, iou_thres=0.5, score_threshold=0.5),
}
```

The detector model output is a multi-head dictionary:

```text
hm, ab, trig, reg, mask
```

Some legacy paths may also support an `ang` head, but the current EPNet/HBTXR configs use `trig` and set `ang_weight=0`.

## 3. Metric Calculation

`post_process()` converts raw detector heads into a detection dictionary:

1. Apply sigmoid to `hm`.
2. Apply NMS to the heatmap.
3. Select top-K candidate locations.
4. Gather `reg`, `ab`, and `trig` or `ang` at selected indices.
5. Restore angle from `trig` when present.
6. Concatenate:

```text
ellipse = [xs, ys, ab, ang]
```

The metric functions only use samples with:

```text
close == 0
```

Metrics:

- `p_acc`: center-point accuracy. It computes Euclidean distance between predicted center `(xs, ys)` and target `center`, then checks whether distance is below tolerance. The standard tolerances logged by the model are 10, 5, 3, and 1 pixels at `64x64` output resolution.
- `cal_mean_distance`: mean Euclidean center distance.
- `cal_batch_iou`: rasterizes predicted and target ellipses on a `64x64` canvas, then computes intersection-over-union.
- `cal_batch_ap`: sorts predictions by score, computes TP/FP/FN using `score_threshold=0.5` and `iou_thres=0.5`, then integrates precision/recall into AP.

## 4. Model Input/Output Dimensions

Detector input:

```text
input: (B, 2, 256, 256)
```

Detector output heads:

```text
hm:   (B, 1, 64, 64)
ab:   (B, 2, 64, 64)
trig: (B, 2, 64, 64)
reg:  (B, 2, 64, 64)
mask: (B, 1, 64, 64)
```

Target tensor shapes after batching:

```text
hm:       (B, 1, 64, 64)
mask:     (B, 1, 64, 64)
ab:       (B, 100, 2)
ang:      (B, 100, 1)
trig:     (B, 100, 2)
reg:      (B, 100, 2)
ind:      (B, 100)
reg_mask: (B, 100)
center:   (B, 2)
close:    (B,)
ellipse:  (B, 5)
```

EPNet uses a MobileNetV3/FPN path and applies `EPHead(in_channels=64)`.

HBTXR uses `img_size=256`, `patch_size=4`, so the DeiT patch map is `64x64`; it then applies projection and the same detector-head contract.

## Current Training Decision

For local training against:

```text
/mnt/e/DATASET/DeanDataset_full_unet
```

the selected default run is the FACET baseline:

```text
model: EPNet
mode: fpn_2d
```

Reason:

- EPNet is the primary FACET detector path.
- HBTXR was not requested for this run.
- The current GPU is a 12GB RTX 4070 Ti, so the local config uses a conservative batch size instead of the original full-run batch size.

