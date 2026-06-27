# FACET Phase 3 Full Data_davis Expansion Prep Log

Date: 2026-06-25

## Scope

이 문서는 `FACET_reproduction_plan_2026-06-25.md`의 Phase 3, 즉 U-Net checkpoint로 전체 `Data_davis` frame에 mask/ellipse label을 생성하고 `DeanDataset_full_unet`을 만드는 단계의 준비 및 smoke 검증 기록이다.

현재 full U-Net training checkpoint는 아직 확보되지 않았다. 따라서 이 문서는 실제 full label expansion 결과가 아니라, Phase 2 checkpoint가 생겼을 때 바로 실행 가능한 expansion script와 제한 smoke 검증을 기록한다.

## Implemented Changes

### 1. Full Data_davis to DeanDataset expansion script

추가 파일:

- `EvEye/utils/scripts/build_full_dean_dataset_with_unet.py`

역할:

- `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user*/left|right/session_*/frames/*.png`를 순회한다.
- 같은 session의 `events/events.txt`를 읽는다.
- U-Net checkpoint로 frame mask를 예측한다.
- 예측 mask에서 ellipse `(t, x, y, a, b, ang)`를 추출한다.
- 각 frame timestamp 직전 최대 5000개 event를 잘라낸다.
- `DavisEyeEllipseDataset`이 읽는 `cached_data/cached_ellipse` memmap 구조로 저장한다.

기본 출력 경로:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
```

주요 옵션:

```text
--checkpoint
--output-root
--train-ratio
--batch-size
--inference-batch-size
--input-height
--input-width
--mask-threshold
--min-mask-pixels
--events-per-sample
--max-sessions
--max-frames-per-session
--device
--dry-run
--overwrite
```

split rule:

- session-order split
- 기본 `train_ratio=0.8`

### 2. Full expanded EPNet config

추가 파일:

- `configs/DavisEyeEllipse_EPNet_full_unet.yaml`

역할:

- `DeanDataset_full_unet` 생성 후 Phase 4 EPNet 학습에 사용한다.
- root path:

```text
/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet
```

- model: EPNet
- mode: `fpn_2d`
- logger: FACET repo-local `runs/logs/EPNet_full_unet`

## Data_davis Inventory

Command summary:

```bash
python - <<'PY'
from pathlib import Path
root = Path('/home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis')
sessions = [p for p in root.glob('user*/*/session_*') if (p/'frames').exists() and (p/'events/events.txt').exists()]
frames = sum(len(list((p/'frames').glob('*.png'))) for p in sessions)
print('sessions', len(sessions))
print('frames', frames)
PY
```

Observed:

```text
sessions: 384
frames:   1490548
```

Implication:

- full expansion은 약 149만 frame에 U-Net inference를 수행해야 한다.
- CPU로는 비현실적으로 오래 걸릴 수 있으므로 GPU 환경에서 실행하는 것이 맞다.

## Dry-run Verification

Command:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/build_full_dean_dataset_with_unet.py \
  --dry-run \
  --max-sessions 3 \
  --max-frames-per-session 5 \
  --output-root /tmp/facet_full_unet_smoke
```

Result:

```text
num_sessions: 3
events_per_sample: 5000
preview_sessions:
  /home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user1/left/session_1_0_1
  /home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user1/left/session_1_0_2
  /home/kjm26/project/dataset/XR/EV_Eye/raw_data/Data_davis/user1/left/session_2_0_1
```

판정:

- script가 full `Data_davis` session 구조를 정상 발견한다.

## Limited Expansion Smoke

주의:

- 아래 smoke는 Phase 2 full U-Net checkpoint가 아니라, 제한 batch U-Net smoke checkpoint를 사용했다.
- 따라서 생성된 ellipse 품질은 재현 결과로 해석하면 안 된다.
- 목적은 script path, checkpoint loading, mask inference, ellipse extraction, event slicing, memmap writing, dataset loading 검증이다.

Command:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/build_full_dean_dataset_with_unet.py \
  --checkpoint runs/logs/RGBUNet_local_train_smoke/version_0/checkpoints/epoch=00-val_mean_distance=175.7243.ckpt \
  --output-root /tmp/facet_full_unet_smoke \
  --max-sessions 2 \
  --max-frames-per-session 5 \
  --train-ratio 0.5 \
  --mask-threshold 0.1 \
  --inference-batch-size 2 \
  --overwrite
```

Result:

```text
device: cpu
num_sessions: 2
total_frames_scanned: 10
valid_ellipse_count: 10
skipped_frame_count: 0
num_train: 5
num_val: 5
```

Generated files:

```text
/tmp/facet_full_unet_smoke/manifest.json
/tmp/facet_full_unet_smoke/train/cached_data/events_batch_0.memmap
/tmp/facet_full_unet_smoke/train/cached_data/events_batch_info_0.txt
/tmp/facet_full_unet_smoke/train/cached_data/events_indices_0.npy
/tmp/facet_full_unet_smoke/train/cached_ellipse/ellipses_batch_0.memmap
/tmp/facet_full_unet_smoke/train/cached_ellipse/ellipses_batch_info_0.txt
/tmp/facet_full_unet_smoke/train/cached_ellipse/ellipses_indices_0.npy
/tmp/facet_full_unet_smoke/val/cached_data/events_batch_0.memmap
/tmp/facet_full_unet_smoke/val/cached_data/events_batch_info_0.txt
/tmp/facet_full_unet_smoke/val/cached_data/events_indices_0.npy
/tmp/facet_full_unet_smoke/val/cached_ellipse/ellipses_batch_0.memmap
/tmp/facet_full_unet_smoke/val/cached_ellipse/ellipses_batch_info_0.txt
/tmp/facet_full_unet_smoke/val/cached_ellipse/ellipses_indices_0.npy
```

## DavisEyeEllipseDataset Load Check

Command summary:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
...
PY
```

Result:

```text
train len: 5
train input: (2, 256, 256)
train hm: (1, 64, 64)
train ellipse: tensor([21.9800, 37.1700, 63.0000, 63.0000, 86.8500])

val len: 5
val input: (2, 256, 256)
val hm: (1, 64, 64)
val ellipse: tensor([31.8700, 31.8800, 63.0000, 63.0000, 90.0000])
```

판정:

- Phase 3 output format은 `DavisEyeEllipseDataset`으로 로드 가능하다.
- full expansion 실행 전 pipeline-level smoke는 통과했다.

## Full Expansion Command Template

Phase 2 full U-Net training이 완료된 뒤, 실제 full expansion은 다음 형태로 실행한다.

```bash
PYTHONPATH=. FACET_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
EvEye/utils/scripts/build_full_dean_dataset_with_unet.py \
  --checkpoint /path/to/full_unet_checkpoint.ckpt \
  --output-root /home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet \
  --device cuda:0 \
  --inference-batch-size 32 \
  --train-ratio 0.8 \
  --mask-threshold 0.5 \
  --events-per-sample 5000 \
  --overwrite
```

완료 후 확인:

```bash
PYTHONPATH=. \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import DavisEyeEllipseDataset
for split in ['train', 'val']:
    ds = DavisEyeEllipseDataset('/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet', split=split)
    print(split, len(ds))
PY
```

## Current Phase 3 Status

Completed:

- full `Data_davis` session discovery
- U-Net checkpoint 기반 mask inference path
- mask-to-ellipse extraction path
- event slicing path
- `DeanDataset_full_unet` memmap writer
- limited expansion smoke
- `DavisEyeEllipseDataset` load check
- full expanded dataset용 EPNet config

Not completed:

- Phase 2 full U-Net checkpoint 확보
- 전체 1,490,548 frame expansion
- full `DeanDataset_full_unet` 생성
- full EPNet training/metric comparison

Current blocker:

- 현재 세션에서는 GPU driver 접근이 불가능하다.
- full expansion 자체는 CPU에서도 가능하지만 149만 frame 규모라 실용적이지 않다.
- 품질 있는 full expansion은 Phase 2 full U-Net checkpoint가 먼저 필요하다.

