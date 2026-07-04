# FACET HBTXR DeiT Training Plan

Date: 2026-06-25

## Summary

목표는 `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/EvEye/model/DavisEyeEllipse/HBTXR`의 DeiT backbone을 FACET `tools/train.py` 흐름에서 EPNet처럼 학습 가능하게 만드는 것이다.

요구 조건:

- EPNet의 dataset, loss, metric, train/validation loop 계약을 최대한 유지한다.
- EPNet의 `Neck` 모듈은 사용하지 않는다.
- HBTXR는 DeiT backbone + dense feature projection + EPHead 구조로 구성한다.
- smoke training 후 full `DeanDataset_full_unet` 생성이 끝나면 EPNet 재현 실험과 별도로 HBTXR 학습 실험을 진행할 수 있게 한다.

## Sub-agent Result

GPT5.5-high sub-agent `Wegener`를 호출하여 읽기 전용 분석을 수행했다.

결론:

- 현재 HBTXR에는 `Backbone/DeiT.py`만 있고 LightningModule, config, `model_factory` 등록이 없다.
- EPNet은 backbone, feature fusion, `EPHead`, `CtdetLoss`, validation metric, optimizer hook을 하나의 LightningModule에 포함한다.
- HBTXR를 학습 가능하게 만들려면 EPNet의 입출력 계약을 맞추는 HBTXR LightningModule이 필요하다.
- 현재 `DeiT.forward_features()`는 return이 없어 그대로는 forward가 깨진다.

## Architecture Target

입력:

- `batch["input"]`
- shape: `(B, 2, 256, 256)`

HBTXR forward:

1. DeiT patch embedding
2. Transformer encoder
3. CLS token 제거
4. patch tokens reshape: `(B, N, C)` -> `(B, C, 64, 64)`
5. feature projection: `Conv2d(embed_dim, 64, kernel_size=1 or 3)`
6. `EPHead(in_channels=64)`

출력 dict:

- `hm`: `(B, 1, 64, 64)`
- `ab`: `(B, 2, 64, 64)`
- `trig`: `(B, 2, 64, 64)`
- `reg`: `(B, 2, 64, 64)`
- `mask`: `(B, 1, 64, 64)`

이 출력 구조는 EPNet의 `CtdetLoss`, `post_process`, metric과 동일한 계약이다.

## File-by-file Plan

### 1. `HBTXR/Backbone/DeiT.py`

수정:

- `forward_features()`가 encoder token을 반환하도록 수정한다.
- `img_size=256`, `patch_size=4`, `in_chans=2`를 config에서 받을 수 있게 유지한다.
- classifier `head`는 HBTXR dense prediction wrapper에서는 사용하지 않는다.
- `forward_patch_tokens()` 또는 wrapper-side extraction으로 CLS 제외 patch tokens를 얻는다.

1차 구현 기본값:

- `patch_size=4`
- `embed_dim=192`
- `depth=8`
- `num_heads=3`
- `pretrained=False`

이유:

- `patch_size=4`이면 `256/4=64`, token grid가 EPNet loss target 해상도 `64x64`와 직접 일치한다.
- Neck 없이 feature map 해상도를 맞추는 가장 단순한 구성이다.

주의:

- `patch_size=4`는 4096 tokens라 실제 학습 메모리 비용이 크다.
- 메모리 문제가 있으면 2차로 `patch_size=8`과 lightweight upsample projection을 검토한다. 단, 이 경우 Neck은 아니지만 decoder 성격의 upsampling block이 추가된다.

### 2. `HBTXR/HBTXR.py`

새 파일 생성:

- `lightning.LightningModule`
- EPNet의 다음 메서드 구조를 재사용한다.
  - `_log`
  - `set_optimizer_config`
  - `lr_scheduler_step`
  - `configure_optimizers`
  - `training_step`
  - `validation_step`
  - `on_validation_epoch_end`

재사용 import:

- `EvEye.model.DavisEyeEllipse.EPNet.Head.EPHead`
- `EvEye.model.DavisEyeEllipse.EPNet.Loss.CtdetLoss`
- `EvEye.model.DavisEyeEllipse.EPNet.Predict.post_process`
- `EvEye.model.DavisEyeEllipse.EPNet.Metric.*`

명시적으로 사용하지 않을 것:

- `EvEye.model.DavisEyeEllipse.EPNet.Neck.FPN`
- `EvEye.model.DavisEyeEllipse.EPNet.Neck.SSD`

### 3. `HBTXR/__init__.py`

추가:

```python
from .HBTXR import HBTXR
```

### 4. `model_factory.py`

`MODEL_CLASSES`에 추가:

```python
HBTXR=("EvEye.model.DavisEyeEllipse.HBTXR.HBTXR", "HBTXR")
```

기존 `make_model()` 흐름은 유지한다.

### 5. `configs/DavisEyeEllipse_HBTXR_local_train_smoke.yaml`

EPNet local train smoke config를 복제해서 수정한다.

변경:

- `model.type: HBTXR`
- `model.input_channels: 2`
- `model.img_size: 256`
- `model.patch_size: 4`
- `model.embed_dim: 192`
- `model.depth: 8`
- `model.num_heads: 3`
- `head_dict`, `loss_weight`는 EPNet과 동일
- `limit_train_batches: 2`
- `limit_val_batches: 2`
- logger name: `HBTXR_local_train_smoke`

### 6. `configs/DavisEyeEllipse_HBTXR.yaml`

실제 학습용 config.

초기 기준:

- dataset은 `DavisEyeEllipse_EPNet_full_unet.yaml`과 동일한 `DeanDataset_full_unet` 사용
- trainer/logger/callback 구조는 EPNet config와 동일
- batch size는 메모리 확인 후 보수적으로 시작

## Validation Plan

### Gate 1. Import and factory smoke

```bash
cd /home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET
PYTHONPATH=. python - <<'PY'
from EvEye.model.model_factory import make_model
cfg = {
    "type": "HBTXR",
    "input_channels": 2,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 1,
    "num_heads": 3,
    "head_dict": {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1},
}
m = make_model(cfg)
print(type(m).__name__)
PY
```

### Gate 2. Forward shape smoke

```bash
PYTHONPATH=. python - <<'PY'
import torch
from EvEye.model.model_factory import make_model
cfg = {
    "type": "HBTXR",
    "input_channels": 2,
    "img_size": 256,
    "patch_size": 4,
    "embed_dim": 192,
    "depth": 1,
    "num_heads": 3,
    "head_dict": {"hm": 1, "ab": 2, "trig": 2, "reg": 2, "mask": 1},
}
m = make_model(cfg)
y = m(torch.randn(2, 2, 256, 256))
print({k: tuple(v.shape) for k, v in y.items()})
PY
```

Expected:

```text
hm: (2, 1, 64, 64)
ab: (2, 2, 64, 64)
trig: (2, 2, 64, 64)
reg: (2, 2, 64, 64)
mask: (2, 1, 64, 64)
```

### Gate 3. Train smoke

```bash
PYTHONPATH=. FACET_DEVICES=0 FACET_DISABLE_CUDNN=1 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_HBTXR_local_train_smoke.yaml
```

### Gate 4. Full training readiness

조건:

- `DeanDataset_full_unet/manifest.json` 존재
- `DavisEyeEllipseDataset` train/val loader smoke 통과
- HBTXR local train smoke 통과
- GPU memory가 batch size 1 또는 2 기준으로 안정적

## Risks

- `patch_size=4`는 4096 tokens라 DeiT self-attention 비용이 크다.
- pretrained DeiT URL checkpoint는 3채널/224/patch16 기준이므로 2채널/256/patch4와 직접 호환되지 않는다.
- HBTXR는 EPNet보다 훨씬 무거울 수 있어 full training batch size를 줄여야 할 가능성이 높다.
- `mask` head가 EPNet loss에서 사용되므로 HBTXR도 동일하게 출력해야 한다.

## Decision

1차 구현은 Neck 없는 dense DeiT 형태로 진행한다.

- `patch_size=4`
- token grid 직접 `64x64`
- projection conv + `EPHead`
- EPNet loss/metric/train loop 재사용

이 구성이 smoke를 통과하면 full `DeanDataset_full_unet` 생성 완료 후 EPNet full reproduction과 별도 HBTXR training config로 비교 실험을 진행한다.
