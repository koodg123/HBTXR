# FACET Phase 1 Subset Smoke Log

Date: 2026-06-25

## Scope

이 문서는 `FACET_reproduction_plan_2026-06-25.md`의 Phase 1, 즉 현재 생성된 8911개 `DeanDataset` subset으로 EPNet pipeline을 먼저 재현 가능한 상태로 만드는 작업의 실행 로그이다.

대상 코드:

- `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET`

대상 데이터:

- `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset`
- train: 7128 samples
- val: 1783 samples
- total: 8911 samples

## Implemented Changes

### 1. Trainer device 설정 포터블화

수정 파일:

- `tools/train.py`
- `tools/validate.py`
- `tools/validate10times.py`

기존 코드는 Lightning Trainer에서 `devices=[2]`를 하드코딩하고 있어 단일 GPU, CPU, 다른 GPU index 환경에서 바로 실패할 수 있었다. 이를 다음 기준으로 변경했다.

- 기본값은 config의 `trainer.devices`, 없으면 `auto`
- 환경변수 `FACET_DEVICES`가 있으면 우선 사용
- `"0,1"` 형태는 `[0, 1]`로 변환
- `"0"` 형태는 `[0]`로 변환
- `trainer.accelerator`도 config에서 읽음

### 2. EPNet loss의 GPU 강제 의존 제거

수정 파일:

- `EvEye/model/DavisEyeEllipse/EPNet/Loss.py`

주요 수정:

- `from predict import _topk` 상대 import 오류 제거
- 실제 존재하는 `EvEye.model.DavisEyeEllipse.EPNet.Predict.topk` 사용
- loss 내부 `.cuda()` 강제 호출 제거
- `torch.eye(...)`와 zero tensor를 입력 tensor의 `device`와 `dtype`에 맞춰 생성
- `WeightLoss`의 target tensor도 `pred.new_tensor(...)`로 생성

이 수정으로 CPU smoke와 non-default GPU 환경에서 loss 계산이 가능해졌다.

### 3. Factory import 지연 처리

수정 파일:

- `EvEye/model/model_factory.py`
- `EvEye/callback/callback_factory.py`

문제:

- `model_factory.py`가 EPNet만 사용할 때도 ElNet을 즉시 import했고, ElNet은 `DCNv2`가 없으면 import에서 실패했다.
- `callback_factory.py`가 S3 callback을 즉시 import했고, 로컬 실험에서 `boto3`가 없으면 import에서 실패했다.

수정:

- 요청된 모델 type만 import하도록 `model_factory.py`를 lazy import 구조로 변경
- `S3Checkpoint`는 config에서 실제 요청될 때만 import하도록 변경

### 4. EPNet optimizer warning 제거

수정 파일:

- `EvEye/model/DavisEyeEllipse/EPNet/EPNet.py`

`configure_optimizers()` 내부의 불필요한 `super().configure_optimizers()` 호출을 제거했다. 이 호출은 Lightning에서 "optimizer가 구현되어야 한다"는 오해성 warning을 발생시켰다.

### 5. Local config 추가

추가 파일:

- `configs/DavisEyeEllipse_EPNet_local_subset.yaml`
- `configs/DavisEyeEllipse_EPNet_local_smoke.yaml`
- `configs/DavisEyeEllipse_EPNet_local_train_smoke.yaml`

`DavisEyeEllipse_EPNet_local_subset.yaml`:

- subset DeanDataset 기준 1 epoch baseline 실행용
- root path: `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset`
- batch size: 2
- workers: 0
- trainer devices/accelerator: config 기반 auto

`DavisEyeEllipse_EPNet_local_smoke.yaml`:

- `tools/train.py` 진입점 검증용
- CPU fast-dev-run
- train 1 batch + val 1 batch만 실행

`DavisEyeEllipse_EPNet_local_train_smoke.yaml`:

- checkpoint와 TensorBoard log 생성 검증용
- CPU 실행
- train 2 batches + val 2 batches
- `ModelCheckpoint` monitor: `val_mean_distance`

### 6. Validation AP metric NaN 방지

수정 파일:

- `EvEye/model/DavisEyeEllipse/EPNet/Metric.py`

초기 무작위 모델에서는 모든 detection score가 threshold 미만일 수 있다. 이때 기존 `cal_batch_ap()`는 `precision = 0 / 0` 또는 `recall = 0 / 0`을 만들 수 있었고, validation output 전체가 skip되었다.

수정:

- valid ground truth가 없는 batch는 AP `0.0` 반환
- precision/recall 분모가 0인 위치는 `np.divide(..., out=0, where=denom!=0)`로 처리

이 수정 후 초기 smoke validation에서도 `val_AP=0.0`으로 기록된다.

## Verification

### 1. Python syntax check

Command:

```bash
python3 -m py_compile \
  tools/train.py \
  tools/validate.py \
  tools/validate10times.py \
  EvEye/model/model_factory.py \
  EvEye/callback/callback_factory.py \
  EvEye/model/DavisEyeEllipse/EPNet/EPNet.py \
  EvEye/model/DavisEyeEllipse/EPNet/Loss.py
```

Result:

- Passed

### 2. Hardcoded GPU and stale loss import check

Command:

```bash
rg "devices=\[|\.cuda\(|from predict import _topk|_topk" tools EvEye/model/DavisEyeEllipse/EPNet/Loss.py
```

Result:

- No remaining matches in the checked scope.

### 3. Runtime environment preflight

Repo-local venv:

- `/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv`

Observed package state:

- `torch 2.12.1+cu130`
- `torch.cuda.is_available() == False`
- venv size: about 5.5G

Important note:

- FACET `requirements.txt` requests `torch==2.2`.
- 이 smoke 환경은 resolver가 최신 `torch/torchvision` 계열을 설치한 상태이므로, 논문 metric 재현용 최종 환경으로 확정하면 안 된다.
- Phase 1 smoke 목적은 코드 경로와 데이터 경로가 실행 가능한지 확인하는 것이다.

### 4. GPU availability check

Command:

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
```

Result:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

판정:

- 현재 세션에서는 GPU 학습 검증 불가.
- full EPNet training과 논문 metric 비교는 GPU driver 접근이 가능한 환경에서 별도로 실행해야 한다.

### 5. DataLoader smoke

Command summary:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
from EvEye.utils.scripts.load_config import load_config
from EvEye.dataset.dataset_factory import make_dataloader
cfg = load_config('DavisEyeEllipse_EPNet_local_subset.yaml')
...
PY
```

Result:

- `dataset_len`: 7128
- batch type: `dict`
- keys:
  - `ab`
  - `ang`
  - `center`
  - `close`
  - `ellipse`
  - `hm`
  - `ind`
  - `input`
  - `mask`
  - `reg`
  - `reg_mask`
  - `trig`

Representative shapes:

- `input`: `[1, 2, 256, 256]`
- `hm`: `[1, 1, 64, 64]`
- `ab`: `[1, 100, 2]`
- `trig`: `[1, 100, 2]`
- `mask`: `[1, 1, 64, 64]`
- `ellipse`: `[1, 5]`

판정:

- subset DeanDataset은 `DavisEyeEllipseDataset` + `EPNet` target format으로 로딩된다.

### 6. EPNet forward and loss smoke

Command summary:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
from EvEye.dataset.dataset_factory import make_dataloader
from EvEye.model.model_factory import make_model
...
pred = model(batch['input'])
loss, loss_show = model.criterion(pred, batch)
PY
```

Result:

- output heads:
  - `hm`: `[2, 1, 64, 64]`
  - `ab`: `[2, 2, 64, 64]`
  - `trig`: `[2, 2, 64, 64]`
  - `reg`: `[2, 2, 64, 64]`
  - `mask`: `[2, 1, 64, 64]`
- loss computed successfully
- representative loss: `685.7674`

판정:

- EPNet forward path와 `CtdetLoss` 계산은 CPU에서 통과했다.
- 기존 `.cuda()` 강제 의존과 `_topk` import 문제는 Phase 1 기준으로 해소됐다.

### 7. EPNet backward and validation_step smoke

Command summary:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python - <<'PY'
...
loss.backward()
metrics = model.validation_step(val_batch, 0)
PY
```

Result:

- `backward_loss`: `480.5820`
- `grad_norm_sum`: `86138.0553`
- validation metrics:
  - `val_loss`: `704.9833`
  - `val_p10_acc`: `0.0`
  - `val_p5_acc`: `0.0`
  - `val_p3_acc`: `0.0`
  - `val_p1_acc`: `0.0`
  - `val_mean_distance`: `51.5335`
  - `val_IoU`: `0.0`
  - `val_AP`: `0.0`

판정:

- `loss.backward()`가 1 batch 이상 통과했다.
- `validation_step()`이 NaN 없이 metric dict를 반환한다.

### 8. `tools/train.py` fast-dev-run smoke

Command:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_EPNet_local_smoke.yaml
```

Result:

- exit code: 0
- Lightning fast-dev-run completed:
  - train: 1 batch
  - val: 1 batch
- model size reported by Lightning:
  - trainable params: 3.9M
  - estimated model params size: 15.593 MB
- representative training metrics:
  - `train_loss`: about `1.23e+3`
  - `train_hm_loss`: about `1.22e+3`
  - `train_iou_loss`: about `13.10`
  - `train_mask_loss`: about `0.606`

판정:

- smoke 목적에서는 통과.
- metric 재현 목적에서는 불충분.
- 다음 단계에서 실제 subset training을 몇 epoch 이상 수행한 뒤 validation metric 경로를 다시 확인해야 한다.

### 9. `tools/train.py` checkpoint/log smoke

Command:

```bash
PYTHONPATH=. MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_EPNet_local_train_smoke.yaml
```

Result:

- exit code: 0
- train: 2 batches
- val: 2 batches
- checkpoint and TensorBoard files generated:
  - `runs/logs/EPNet_local_train_smoke/version_0/checkpoints/epoch=00-val_mean_distance=38.6962.ckpt`
  - `runs/logs/EPNet_local_train_smoke/version_0/checkpoints/last.ckpt`
  - `runs/logs/EPNet_local_train_smoke/version_0/events.out.tfevents.1782373061.etrib.2.0`
  - `runs/logs/EPNet_local_train_smoke/version_0/hparams.yaml`
- checkpoint sizes:
  - best checkpoint: `47180139 bytes`
  - last checkpoint: `47180267 bytes`

Representative metrics:

- `val_loss`: about `744.0`
- `val_p10_acc`: `0.000`
- `val_p5_acc`: `0.000`
- `val_p3_acc`: `0.000`
- `val_p1_acc`: `0.000`
- `val_mean_distance`: `38.70`
- `val_IoU`: `0.000`
- `val_AP`: `0.000`
- `train_loss`: about `1.23e+3`

판정:

- Phase 1 smoke 기준의 checkpoint 생성 요구와 metric log 생성 요구는 충족했다.
- 이는 full subset training 결과가 아니라 제한 batch smoke 결과이다.

## Current Phase 1 Status

통과:

- subset DeanDataset 생성 및 train/val split 확인
- DataLoader batch 생성
- EPNet forward
- EPNet loss 계산
- EPNet backward
- EPNet validation_step metric 반환
- `tools/train.py` 진입점 fast-dev-run
- 제한 batch train smoke
- checkpoint 생성
- TensorBoard event log 생성
- CPU smoke

아직 미완료:

- GPU training
- 논문 설정과 동일한 `torch==2.2` 고정 환경 재구성
- subset baseline의 실제 epoch training
- validation metric 안정화 확인
- 논문 metric과의 비교

## Next Actions

1. GPU driver가 정상 동작하는 환경에서 `.facet-train-venv` 대신 재현용 고정 환경을 새로 만든다.
2. `requirements.txt` 기준으로 `torch==2.2`와 호환되는 `torchvision`/Lightning 버전을 고정한다.
3. `DavisEyeEllipse_EPNet_local_subset.yaml`로 1 epoch subset baseline을 실행한다.
4. validation metric이 NaN 없이 집계되는지 확인한다.
5. subset 결과를 `FACET_reproduction_results_<date>.md`로 별도 저장한다.
6. 이후 Phase 2인 `Data_davis_labelled_with_mask` 기반 U-Net 재학습으로 넘어간다.
