#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent"
MANIFEST="${DATASET_ROOT}/manifest.json"
PROGRESS="${DATASET_ROOT}/progress_state.json"
FACET_ROOT="/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET"
PYTHON="/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python"
LOG_DIR="/home/kjm26/project/PRJXR/HBTXR/references/report/FACET"
LOG_FILE="${LOG_DIR}/HBTXR_subject_independent_img64_patch4_gpu1_train_2026-06-28.log"
CONFIG="DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml"

mkdir -p "${LOG_DIR}"

echo "[$(date --iso-8601=seconds)] validating ${DATASET_ROOT}" | tee -a "${LOG_FILE}"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
from pathlib import Path

root = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet_subject_independent")
manifest = json.loads((root / "manifest.json").read_text())
progress = json.loads((root / "progress_state.json").read_text())
expected_subjects = {
    "train": list(range(1, 33)),
    "val": list(range(33, 37)),
    "test": list(range(37, 49)),
}
if manifest["split_subjects"] != expected_subjects:
    raise SystemExit(f"unexpected split subjects: {manifest['split_subjects']}")
counts = {k: int(v) for k, v in progress["writer_counts"].items()}
expected_counts = {"train": 968873, "val": 122776, "test": 366171}
if counts != expected_counts:
    raise SystemExit(f"unexpected writer counts: {counts}")
print("dataset gate ok:", counts)
PY

echo "[$(date --iso-8601=seconds)] starting HBTXR subject-independent img64 patch4 training on GPU1" | tee -a "${LOG_FILE}"
cd "${FACET_ROOT}"

export PYTHONPATH=.
export FACET_DEVICES=1
export FACET_DISABLE_CUDNN=1
export MPLCONFIGDIR=/tmp/matplotlib-facet
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_hbtxr_subject_independent_img64_patch4_pycache

echo "[$(date --iso-8601=seconds)] running subject-independent img64 patch4 smoke check" | tee -a "${LOG_FILE}"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import copy
import torch
from torch.utils.data import DataLoader

from EvEye.dataset.dataset_factory import make_dataset
from EvEye.model.model_factory import make_model
from EvEye.utils.scripts.load_config import load_config

cfg = load_config("DavisEyeEllipse_HBTXR_subject_independent_img64_patch4.yaml")
ds = make_dataset(copy.deepcopy(cfg["dataloader"]["train"]["dataset"]))
sample = ds[0]
print("sample input", sample["input"].shape)
print("sample hm", sample["hm"].shape)
print("sample mask", sample["mask"].shape)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0)
batch = next(iter(loader))
model = make_model(copy.deepcopy(cfg["model"]))
model.eval()
with torch.no_grad():
    pred = model(batch["input"].float())
print("pred", {k: tuple(v.shape) for k, v in pred.items()})
loss, stats = model.criterion(pred, batch)
print("loss", float(loss.detach().cpu()))
print(
    "loss_stats",
    {
        k: float(v.detach().cpu()) if hasattr(v, "detach") else float(v)
        for k, v in stats.items()
    },
)
PY

"${PYTHON}" tools/train.py -c "${CONFIG}" 2>&1 | tee -a "${LOG_FILE}"
