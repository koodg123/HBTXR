#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="python3"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo "[ERROR] No usable Python interpreter found." >&2
        exit 1
    fi
fi

if [ -n "${PYTHONPATH:-}" ]; then
    export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
else
    export PYTHONPATH="$PROJECT_ROOT/src"
fi

sanitize_component() {
    local value
    value=$(printf '%s' "${1:-}" | tr '[:space:]/:\\' '_' | tr -cd '[:alnum:]_.-')
    value=${value#[_\.-]}
    value=${value%[_\.-]}
    if [ -n "$value" ]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' "hbtxr"
    fi
}

is_dotted_override_option() {
    case "${1:-}" in
        --*.*|--*.*=*) return 0 ;;
        *) return 1 ;;
    esac
}

usage() {
    cat <<'USAGE'
Usage:
  sh scripts/hbtxr_mode_pipeline.sh <train|eval|infer|vis|dataloader|export> [options]

Common options:
  --config PATH
  --mode mode0|mode1|mode2
  --stage stage1|stage2
  --resume [RUN_ROOT|auto]
  --experiment-name NAME
  --device DEVICE
  --checkpoint PATH
  --split train|val|test
  --kind dataset|inference|runtime    (vis only)
  --num-workers N
  --override KEY=VALUE
  --training.epochs 30
  --training.epochs=30

Examples:
  sh scripts/hbtxr_mode_pipeline.sh train \
    --config configs/mode1_stage1.yaml \
    --mode mode1 \
    --stage stage1 \
    --experiment-name stage1_baseline

  sh scripts/hbtxr_mode_pipeline.sh infer \
    --config configs/mode1_stage2.yaml \
    --mode mode1 \
    --stage stage2 \
    --resume stage1_baseline_20260325_120000 \
    --split test

  sh scripts/hbtxr_mode_pipeline.sh vis \
    --config configs/mode1_stage2.yaml \
    --mode mode1 \
    --stage stage2 \
    --resume stage1_baseline_20260325_120000 \
    --kind inference \
    --split test

  sh scripts/hbtxr_mode_pipeline.sh dataloader \
    --config configs/mode1_stage1.yaml \
    --mode mode1 \
    --stage stage1 \
    --split train

  sh scripts/hbtxr_mode_pipeline.sh export \
    --config configs/mode1_stage2.yaml \
    --mode mode1 \
    --stage stage2 \
    --resume stage1_baseline_20260325_120000
USAGE
}

ACTION="${1:-help}"
case "$ACTION" in
    visualize) ACTION="vis" ;;
esac
case "$ACTION" in
    train|eval|infer|vis|dataloader|export) ;;
    help|-h|--help)
        usage
        exit 0
        ;;
    *)
        echo "[ERROR] Unknown action: $ACTION" >&2
        usage >&2
        exit 1
        ;;
esac
shift || true

CONFIG_PATH=""
MODE=""
STAGE=""
RESUME_VALUE=""
RESUME_SET=0
EXPERIMENT_NAME=""
DEVICE_OVERRIDE=""
CHECKPOINT_PATH=""
SPLIT="${SPLIT:-test}"
VIS_KIND="${VISUALIZE_KIND:-inference}"
OVERRIDES=()
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --config=*)
            CONFIG_PATH="${1#*=}"
            shift
            ;;
        --config)
            CONFIG_PATH="${2:-}"
            shift 2
            ;;
        --mode=*)
            MODE="${1#*=}"
            shift
            ;;
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --stage=*)
            STAGE="${1#*=}"
            shift
            ;;
        --stage)
            STAGE="${2:-}"
            shift 2
            ;;
        --resume=*)
            RESUME_VALUE="${1#*=}"
            RESUME_SET=1
            shift
            ;;
        --resume)
            RESUME_SET=1
            if [ $# -gt 1 ] && [[ "${2:-}" != --* ]]; then
                RESUME_VALUE="${2:-}"
                shift 2
            else
                RESUME_VALUE="auto"
                shift
            fi
            ;;
        --experiment-name=*)
            EXPERIMENT_NAME="$(sanitize_component "${1#*=}")"
            shift
            ;;
        --experiment-name)
            EXPERIMENT_NAME="$(sanitize_component "${2:-}")"
            shift 2
            ;;
        --device=*)
            DEVICE_OVERRIDE="${1#*=}"
            shift
            ;;
        --device)
            DEVICE_OVERRIDE="${2:-}"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_PATH="${1#*=}"
            shift
            ;;
        --checkpoint)
            CHECKPOINT_PATH="${2:-}"
            shift 2
            ;;
        --split=*)
            SPLIT="${1#*=}"
            shift
            ;;
        --split)
            SPLIT="${2:-}"
            shift 2
            ;;
        --kind=*)
            VIS_KIND="${1#*=}"
            shift
            ;;
        --kind)
            VIS_KIND="${2:-}"
            shift 2
            ;;
        --num-workers=*)
            OVERRIDES+=("training.num_workers=${1#*=}")
            shift
            ;;
        --num-workers)
            OVERRIDES+=("training.num_workers=${2:-}")
            shift 2
            ;;
        --override=*)
            OVERRIDES+=("${1#*=}")
            shift
            ;;
        --override)
            OVERRIDES+=("${2:-}")
            shift 2
            ;;
        --*=*)
            if is_dotted_override_option "$1"; then
                OVERRIDES+=("${1#--}")
                shift
            else
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        -*)
            if is_dotted_override_option "$1"; then
                if [ $# -lt 2 ]; then
                    echo "[ERROR] Missing value for override option: $1" >&2
                    exit 1
                fi
                OVERRIDES+=("${1#--}=${2:-}")
                shift 2
            else
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$CONFIG_PATH" ]; then
    echo "[ERROR] --config is required." >&2
    exit 1
fi

common_args=(--config "$CONFIG_PATH")
if [ -n "$MODE" ]; then
    common_args+=(--mode "$MODE")
fi
if [ -n "$STAGE" ]; then
    common_args+=(--stage "$STAGE")
fi
if [ "$RESUME_SET" -eq 1 ]; then
    common_args+=(--resume "$RESUME_VALUE")
fi
if [ -n "$EXPERIMENT_NAME" ]; then
    common_args+=(--experiment-name "$EXPERIMENT_NAME")
fi
if [ -n "$DEVICE_OVERRIDE" ]; then
    common_args+=(--device "$DEVICE_OVERRIDE")
fi
for item in "${OVERRIDES[@]}"; do
    common_args+=(--override "$item")
done

echo "[INFO] action=$ACTION"
echo "[INFO] config=$CONFIG_PATH"
echo "[INFO] mode=${MODE:-<config>}"
echo "[INFO] stage=${STAGE:-<config>}"
echo "[INFO] resume=${RESUME_VALUE:-<none>}"
echo "[INFO] experiment_name=${EXPERIMENT_NAME:-<config>}"
echo "[INFO] split=${SPLIT}"

case "$ACTION" in
    train)
        cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/train_hbtxr.py" "${common_args[@]}")
        if [ -n "$CHECKPOINT_PATH" ]; then
            cmd+=(--init-checkpoint "$CHECKPOINT_PATH")
        fi
        ;;
    eval)
        cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval_hbtxr.py" "${common_args[@]}" --split "$SPLIT")
        if [ -n "$CHECKPOINT_PATH" ]; then
            cmd+=(--checkpoint "$CHECKPOINT_PATH")
        fi
        ;;
    infer)
        cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/infer_hbtxr.py" "${common_args[@]}" --split "$SPLIT")
        if [ -n "$CHECKPOINT_PATH" ]; then
            cmd+=(--checkpoint "$CHECKPOINT_PATH")
        fi
        ;;
    vis)
        case "$VIS_KIND" in
            dataset)
                cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/visualize_dataset.py" "${common_args[@]}" --split "$SPLIT")
                ;;
            inference)
                cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/visualize_inference_results.py" "${common_args[@]}" --split "$SPLIT")
                ;;
            runtime)
                cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/visualize_runtime.py" "${common_args[@]}" --split "$SPLIT")
                ;;
            *)
                echo "[ERROR] Unknown vis kind: $VIS_KIND" >&2
                exit 1
                ;;
        esac
        ;;
    dataloader)
        cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/check_dataloader.py" "${common_args[@]}" --split "$SPLIT")
        ;;
    export)
        cmd=("$PYTHON_BIN" "$PROJECT_ROOT/scripts/export_hbtxr.py" "${common_args[@]}")
        if [ -n "$CHECKPOINT_PATH" ]; then
            cmd+=(--checkpoint "$CHECKPOINT_PATH")
        fi
        ;;
esac

cmd+=("${EXTRA_ARGS[@]}")
"${cmd[@]}"
