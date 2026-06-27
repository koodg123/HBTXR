# FACET HBTXR Independent Implementation

Date: 2026-06-25

## Summary

요청에 따라 EPNet 모듈을 직접 재사용하지 않고, EPNet-style 학습에 필요한 head/loss/predict/metric 구성요소를 HBTXR 디렉터리로 복사한 뒤 HBTXR 이름공간으로 독립시켰다.

현재 HBTXR는 FACET `tools/train.py`에서 `model.type: HBTXR`로 생성 및 학습 smoke가 가능하다.

## Implemented Files

New or modified HBTXR files:

- `EvEye/model/DavisEyeEllipse/HBTXR/HBTXR.py`
  - HBTXR LightningModule.
  - DeiT backbone -> dense patch map -> projection conv -> HBTXRHead.
  - HBTXR-local loss/predict/metric을 사용.
- `EvEye/model/DavisEyeEllipse/HBTXR/__init__.py`
  - `HBTXR` export.
- `EvEye/model/DavisEyeEllipse/HBTXR/Backbone/DeiT.py`
  - `forward_features()` return 누락 수정.
  - `forward_patch_map()` 추가.
  - classifier forward는 CLS token만 사용하도록 수정.
- `EvEye/model/DavisEyeEllipse/HBTXR/Head/HBTXRHead.py`
  - EPNet `EPHead.py`를 복사한 뒤 class명을 `HBTXRHead`로 변경.
- `EvEye/model/DavisEyeEllipse/HBTXR/Loss.py`
  - EPNet `Loss.py`를 복사.
  - `EPNet.Predict.topk` import를 `HBTXR.Predict.topk`로 변경.
  - `CtdetLoss` class명을 `HBTXRCtdetLoss`로 변경.
- `EvEye/model/DavisEyeEllipse/HBTXR/Predict.py`
  - EPNet `Predict.py`를 복사.
  - 예제 import를 HBTXR로 변경.
- `EvEye/model/DavisEyeEllipse/HBTXR/Metric.py`
  - EPNet `Metric.py`를 복사.
- `EvEye/model/DavisEyeEllipse/HBTXR/utils.py`
  - EPNet `utils.py`를 복사.

Factory/config changes:

- `EvEye/model/model_factory.py`
  - `HBTXR=("EvEye.model.DavisEyeEllipse.HBTXR.HBTXR", "HBTXR")` 등록.
- `configs/DavisEyeEllipse_HBTXR_local_train_smoke.yaml`
  - CPU smoke training config.
- `configs/DavisEyeEllipse_HBTXR_full_unet.yaml`
  - `DeanDataset_full_unet` 기반 full training config.

## Architecture

Input:

- `(B, 2, 256, 256)` event frame.

Backbone:

- DeiT with `patch_size=4`.
- Patch grid: `64x64`.
- Patch tokens are reshaped to dense feature map `(B, embed_dim, 64, 64)`.

Head:

- projection conv: `embed_dim -> 64`
- `HBTXRHead(in_channels=64)`

Output dict:

- `hm`: `(B, 1, 64, 64)`
- `ab`: `(B, 2, 64, 64)`
- `trig`: `(B, 2, 64, 64)`
- `reg`: `(B, 2, 64, 64)`
- `mask`: `(B, 1, 64, 64)`

This matches the DavisEyeEllipseDataset target contract.

## Other Files Investigated

Sub-agent `Locke` inspected:

- `DavisEyeEllipse/EllipseMobileNet.py`
- `DavisEyeEllipse/EPNet/*`
- `DavisEyeEllipse/ElNet/ElNet.py`
- `DavisEyeEllipse/UNet/*`
- `DavisEyeEllipse/HBTXR/*`
- `EvEye/model/model_factory.py`
- `EvEye/dataset/DavisEyeEllipse/*`
- FACET configs

Findings:

- `EllipseMobileNet.py` is a separate MobileNetV3 + FC regression style LightningModule. It uses `EvEye.dataset.DavisEyeEllipse.losses.cal_loss` and does not match EPNet heatmap/ellipse output ABI.
- `ElNet` also reuses EPNet loss/predict/metric and depends on `DCNv2`, so it is a higher build-risk source for HBTXR.
- `UNet` is mask segmentation only and does not provide the ellipse heatmap head/loss contract needed for HBTXR.
- Therefore, HBTXR should follow EPNet's output ABI but keep copied HBTXR-local components instead of importing EPNet modules.

## Validation

### HBTXR namespace does not import EPNet modules

Checked with:

```bash
rg -n "EvEye\\.model\\.DavisEyeEllipse\\.EPNet|from .*EPNet|import .*EPNet|EPHead|CtdetLoss\\(" \
  EvEye/model/DavisEyeEllipse/HBTXR
```

Only HBTXR-local names remain:

- `HBTXRCtdetLoss`
- `HBTXRHead`

### Python syntax

Checked with:

```bash
python3 -m py_compile \
  EvEye/model/DavisEyeEllipse/HBTXR/HBTXR.py \
  EvEye/model/DavisEyeEllipse/HBTXR/Backbone/DeiT.py \
  EvEye/model/DavisEyeEllipse/HBTXR/Head/HBTXRHead.py \
  EvEye/model/DavisEyeEllipse/HBTXR/Loss.py \
  EvEye/model/DavisEyeEllipse/HBTXR/Metric.py \
  EvEye/model/DavisEyeEllipse/HBTXR/Predict.py \
  EvEye/model/model_factory.py
```

Result: passed.

### Forward shape smoke

Command:

```bash
PYTHONPATH=. /home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
import torch
from EvEye.model.model_factory import make_model
cfg = {
    'type': 'HBTXR',
    'input_channels': 2,
    'img_size': 256,
    'patch_size': 4,
    'embed_dim': 192,
    'depth': 1,
    'num_heads': 3,
    'projection_channels': 64,
    'projection_kernel_size': 1,
    'head_conv': 256,
    'pretrained': False,
    'head_dict': {'hm': 1, 'ab': 2, 'trig': 2, 'reg': 2, 'mask': 1},
}
model = make_model(cfg)
model.eval()
with torch.no_grad():
    y = model(torch.randn(1, 2, 256, 256))
print({k: tuple(v.shape) for k, v in y.items()})
PY
```

Result:

```text
{'hm': (1, 1, 64, 64), 'ab': (1, 2, 64, 64), 'trig': (1, 2, 64, 64), 'reg': (1, 2, 64, 64), 'mask': (1, 1, 64, 64)}
```

### Train smoke

Command:

```bash
PYTHONPATH=. FACET_DISABLE_CUDNN=1 MPLCONFIGDIR=/tmp/matplotlib-facet NO_ALBUMENTATIONS_UPDATE=1 \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_HBTXR_local_train_smoke.yaml
```

Result: passed.

Observed:

- dataloader OK
- forward OK
- loss/backward OK
- validation OK
- checkpoint saved:
  - `runs/logs/HBTXR_local_train_smoke/version_0/checkpoints/epoch=00-val_mean_distance=19.4391.ckpt`

## Remaining Notes

- HBTXR no longer directly imports EPNet modules.
- HBTXR still intentionally uses the FACET training framework and DavisEyeEllipseDataset contract.
- Full HBTXR training should wait until `DeanDataset_full_unet/manifest.json` exists and train/val loader smoke passes.
- `patch_size=4` creates 4096 tokens and can be memory-heavy for full DeiT depth. Full config starts with batch size 2.
