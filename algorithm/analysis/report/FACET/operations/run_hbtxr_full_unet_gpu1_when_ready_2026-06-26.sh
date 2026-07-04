#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet"
MANIFEST="${DATASET_ROOT}/manifest.json"
PROGRESS="${DATASET_ROOT}/progress_state.json"
FACET_ROOT="/home/kjm26/project/PRJXR/HBTXR/references/codebase/software/FACET"
PYTHON="/home/kjm26/project/PRJXR/HBTXR/.facet-train-venv/bin/python"
LOG_DIR="/home/kjm26/project/PRJXR/HBTXR/references/report/FACET"
LOG_FILE="${LOG_DIR}/HBTXR_full_unet_gpu1_train_2026-06-26.log"
RUN_ROOT="${FACET_ROOT}/runs/logs/HBTXR_full_unet"

mkdir -p "${LOG_DIR}"

echo "[$(date --iso-8601=seconds)] waiting for ${MANIFEST}" | tee -a "${LOG_FILE}"
while true; do
  if [ -f "${MANIFEST}" ] && [ -f "${PROGRESS}" ]; then
    if "${PYTHON}" - <<'PY' >>"${LOG_FILE}" 2>&1
import json
from pathlib import Path

root = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet")
manifest = json.loads((root / "manifest.json").read_text())
progress = json.loads((root / "progress_state.json").read_text())
expected_sessions = int(manifest["num_sessions"])
completed_sessions = int(progress["completed_session_count"])
manifest_train = int(manifest["num_train"])
manifest_val = int(manifest["num_val"])
progress_train = int(progress["writer_counts"]["train"])
progress_val = int(progress["writer_counts"]["val"])
if completed_sessions != expected_sessions:
    raise SystemExit(
        f"not complete: sessions {completed_sessions}/{expected_sessions}"
    )
if (progress_train, progress_val) != (manifest_train, manifest_val):
    raise SystemExit(
        "manifest/progress count mismatch: "
        f"progress=({progress_train}, {progress_val}) "
        f"manifest=({manifest_train}, {manifest_val})"
    )
print(
    "dataset gate ok:",
    f"sessions={completed_sessions}",
    f"train={manifest_train}",
    f"val={manifest_val}",
)
PY
    then
      break
    fi
  fi
  sleep 120
done

echo "[$(date --iso-8601=seconds)] manifest found; starting HBTXR full training on GPU1" | tee -a "${LOG_FILE}"
cd "${FACET_ROOT}"

export PYTHONPATH=.
export FACET_DEVICES=1
export FACET_DISABLE_CUDNN=1
export MPLCONFIGDIR=/tmp/matplotlib-facet
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONPYCACHEPREFIX=/tmp/facet_hbtxr_full_unet_pycache

latest_checkpoint() {
  find "${RUN_ROOT}" -path '*/checkpoints/*.ckpt' -type f \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | awk 'NR == 1 {print $2}'
}

if [ "${FACET_RESUME_LATEST:-1}" = "1" ]; then
  LATEST_CKPT="$(latest_checkpoint || true)"
  if [ -n "${LATEST_CKPT}" ]; then
    export FACET_CKPT_PATH="${LATEST_CKPT}"
    echo "[$(date --iso-8601=seconds)] resume checkpoint: ${FACET_CKPT_PATH}" | tee -a "${LOG_FILE}"
  else
    echo "[$(date --iso-8601=seconds)] no resume checkpoint found; starting fresh" | tee -a "${LOG_FILE}"
  fi
fi

echo "[$(date --iso-8601=seconds)] running DeanDataset_full_unet smoke check" | tee -a "${LOG_FILE}"
"${PYTHON}" - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import json
from pathlib import Path

from EvEye.dataset.DavisEyeEllipse.DavisEyeEllipseDataset import (
    DavisEyeEllipseDataset,
)

root = Path("/home/kjm26/project/dataset/XR/EV_Eye/raw_data/DeanDataset_full_unet")
manifest = json.loads((root / "manifest.json").read_text())
print("manifest keys:", sorted(manifest.keys()))
for split in ("train", "val"):
    ds = DavisEyeEllipseDataset(
        root_path=str(root),
        split=split,
        accumulate_mode="fixed_count",
        sensor_size=[346, 260, 2],
        events_interpolation="causal_linear_ori",
        pupil_area=200,
        num_classes=1,
        default_resolution=[256, 256],
    )
    print(split, "len", len(ds))
    sample = ds[0]
    print(split, "sample keys", sorted(sample.keys()))
PY

"${PYTHON}" tools/train.py -c DavisEyeEllipse_HBTXR_full_unet.yaml 2>&1 | tee -a "${LOG_FILE}"
