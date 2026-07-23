#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

uv venv --python "${PYTHON_BIN}" .venv
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python --index-url "${PYTORCH_INDEX_URL}" torch torchvision torchaudio
uv pip install --python .venv/bin/python -r requirements-eval.txt

.venv/bin/python scripts/check_torch_gpu.py
