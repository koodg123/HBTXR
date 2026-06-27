#!/usr/bin/env sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "${SCRIPT_DIR}/_common.sh"

RAW_ROOT=${RAW_ROOT:-/mnt/e/WSL/Shared/dataset/Eye/EV_Eye/raw_data/Data_davis}
GROUNDED_SAM_ROOT=${GROUNDED_SAM_ROOT:-/mnt/e/WSL/Shared/ETRI_SYNC/HBTXR/annotation_tools/Grounded-Segment-Anything-main}

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/tools/prepare_facet_gsam_dataset.py" \
    --raw-root "${RAW_ROOT}" \
    --grounded-sam-root "${GROUNDED_SAM_ROOT}" \
    "$@"

