# FACET GPU/cuDNN 진단 및 U-Net 임시 학습 로그

Date: 2026-06-25

## 결론

GPU 인식 자체는 정상이다.

- Python: `/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python`
- PyTorch: `2.12.1+cu130`
- `torch.cuda.is_available()`: `True`
- CUDA visible device count: `2`
- GPU: `NVIDIA GeForce RTX 5080` x 2
- Driver: `595.71.05`
- `nvidia-smi` reported CUDA version: `13.2`
- `/dev/nvidia0`, `/dev/nvidia1`, `/dev/nvidiactl`, `/dev/nvidia-uvm` device files exist.

실제 문제는 CUDA 장치 인식이 아니라 cuDNN backend 초기화 실패였다. FACET U-Net training sanity check에서 다음 계열의 오류가 발생했다.

```text
CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

따라서 현재 환경에서는 `FACET_DISABLE_CUDNN=1`로 cuDNN을 끄고 PyTorch CUDA kernel 경로로 실행해야 한다.

## 원인 판단

확인된 사실:

- 동일 venv에서 `torch.cuda.is_available()`는 `True`.
- GPU 2개와 device file은 host shell 기준 정상.
- 간단한 CUDA tensor/model 실행은 가능.
- cuDNN Conv2d 경로에서는 sublibrary version mismatch가 발생.
- `torch.backends.cudnn.enabled = False` 설정 후 CUDA Conv2d smoke가 통과.

판단:

- NVIDIA driver나 `/dev/nvidia*` 권한 문제가 아니다.
- Codex 기본 sandbox에서는 GPU device가 보이지 않을 수 있으므로, GPU 진단과 학습은 host/escalated 실행 기준으로 확인해야 한다.
- PyTorch wheel의 cuDNN bundle 또는 런타임에 로드되는 cuDNN component 간 버전 불일치가 핵심으로 보인다.

## 적용한 코드 우회

다음 실행 파일에 `FACET_DISABLE_CUDNN=1` 또는 `runtime.disable_cudnn: true` 처리 경로를 추가했다.

- `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/tools/train.py`
- `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/tools/validate.py`
- `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/tools/validate10times.py`
- `/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/EvEye/utils/scripts/evaluate_epnet_checkpoint.py`

권장 실행 형태:

```bash
PYTHONPATH=. FACET_DEVICES=0 FACET_DISABLE_CUDNN=1 MPLCONFIGDIR=/tmp/matplotlib-facet \
/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python \
tools/train.py -c DavisEyeEllipse_RGBUNet_local_subset.yaml
```

## U-Net 임시 학습 결과

실행 위치:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET
```

로그 디렉토리:

```text
runs/logs/RGBUNet_local_subset/version_1
```

학습은 cuDNN disabled 상태에서 정상 진행되었다. 장시간 full 70 epoch 실행 대신, GPU/cuDNN 문제 확인과 checkpoint 확보 목적에서 epoch 4 초반에 graceful shutdown했다.

생성 checkpoint:

```text
runs/logs/RGBUNet_local_subset/version_1/checkpoints/epoch=00-val_mean_distance=1.1978.ckpt
runs/logs/RGBUNet_local_subset/version_1/checkpoints/epoch=02-val_mean_distance=0.4997.ckpt
runs/logs/RGBUNet_local_subset/version_1/checkpoints/epoch=03-val_mean_distance=30.3504.ckpt
runs/logs/RGBUNet_local_subset/version_1/checkpoints/last.ckpt
```

현재 best 후보:

```text
/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET/runs/logs/RGBUNet_local_subset/version_1/checkpoints/epoch=02-val_mean_distance=0.4997.ckpt
```

관찰된 validation 지표:

| epoch | val_mean_distance | val_IoU | val_p1_acc | 판단 |
|---:|---:|---:|---:|---|
| 0 | 1.1978 | 0.764 | 0.578 | 유효 checkpoint |
| 1 | 43.10 | 0.556 | 0.378 | 일시 악화 |
| 2 | 0.4997 | 0.921 | 0.957 | 현재 best |
| 3 | 30.3504 | 0.378 | 0.00385 | 악화 |

`last.ckpt`는 epoch 4 초반 중단 시점이므로 Phase 3 full `Data_davis` 라벨 확장에는 사용하지 않는다. Phase 3에는 위의 epoch 2 best checkpoint를 사용하는 것이 합리적이다.

## 다음 권장 단계

1. `epoch=02-val_mean_distance=0.4997.ckpt`로 full `Data_davis` label expansion을 실행한다.
2. 생성 대상은 `/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet`로 둔다.
3. full expansion 후 `DavisEyeEllipseDataset` load smoke를 다시 확인한다.
4. 이후 `DavisEyeEllipse_EPNet_full_unet.yaml`로 EPNet training을 진행한다.
5. EPNet training도 현재 환경에서는 `FACET_DISABLE_CUDNN=1` 조건을 기본으로 둔다.

