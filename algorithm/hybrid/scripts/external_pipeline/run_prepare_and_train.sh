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

sanitize_component() {
    local value
    value=$(printf '%s' "${1:-}" | tr '[:space:]/:\\' '_' | tr -cd '[:alnum:]_.-')
    value=${value#[_\.-]}
    value=${value%[_\.-]}
    if [ -n "$value" ]; then
        printf '%s\n' "$value"
    else
        printf '%s\n' "src"
    fi
}

is_timestamped_experiment_name() {
    [[ "${1:-}" =~ _[0-9]{8}_[0-9]{6}$ ]]
}

latest_run_root() {
    local experiment_name="$1"
    local runs_root="$PROJECT_ROOT/runs"
    local candidate=""
    local latest=""

    if [ ! -d "$runs_root" ]; then
        return 1
    fi

    if is_timestamped_experiment_name "$experiment_name"; then
        candidate="$runs_root/$experiment_name"
        if [ -d "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
        return 1
    fi

    while IFS= read -r candidate; do
        latest="$candidate"
    done < <(find "$runs_root" -mindepth 1 -maxdepth 1 -type d -name "${experiment_name}_*" | LC_ALL=C sort)

    if [ -n "$latest" ]; then
        printf '%s\n' "$latest"
        return 0
    fi
    return 1
}

print_cmd() {
    printf '[RUN]'
    for token in "$@"; do
        printf ' %q' "$token"
    done
    printf '\n'
}

run_cmd() {
    print_cmd "$@"
    if [ "$DRY_RUN" -eq 0 ]; then
        "$@"
    fi
}

usage() {
    cat <<'USAGE'
Usage:
  sh scripts/run_prepare_and_train.sh --mode mode0|mode1|mode2 [options]

This wrapper performs, in order:
  1. canonical + manifest generation via scripts/run_prepare.sh
  2. Stage1 training via scripts/run_train.sh
  3. Stage2 training via scripts/run_train.sh, initialized from Stage1 best_search_p10.pt

Core options:
  --mode mode0|mode1|mode2           Required.
  --annotation-mode auto|manual_csv|groundedsam
  --stage1-config PATH               Defaults to configs/<mode>_stage1.yaml
  --stage2-config PATH               Defaults to configs/<mode>_stage2.yaml
  --stage1-experiment-name NAME      Defaults to <mode>_stage1
  --stage2-experiment-name NAME      Defaults to <mode>_stage2
  --stage1-checkpoint PATH           Optional explicit Stage1 checkpoint for Stage2 warm start

Path options forwarded to prepare:
  --paths-config PATH
  --project-root PATH
  --raw-root PATH
  --canonical-root PATH
  --indexes-root PATH
  --manifests-root PATH
  --annotation-root PATH
  --groundedsam-root PATH
  --timelens-root PATH

Execution control:
  --device DEVICE                    Shared default device for both stages
  --stage1-device DEVICE             Overrides --device for Stage1 only
  --stage2-device DEVICE             Overrides --device for Stage2 only
  --skip-prepare                     Reuse existing canonical/manifests
  --skip-stage1                      Reuse an existing Stage1 run/checkpoint
  --dry-run                          Print commands without executing them

Pass-through options:
  --prepare-arg ARG                  Extra arg appended to run_prepare.sh (repeatable)
  --stage1-arg ARG                   Extra arg appended to Stage1 run_train.sh (repeatable)
  --stage2-arg ARG                   Extra arg appended to Stage2 run_train.sh (repeatable)

Examples:
  sh scripts/run_prepare_and_train.sh \
    --mode mode0 \
    --annotation-mode manual_csv \
    --paths-config configs/paths/ev_eye_groundedsam_paths.json

  sh scripts/run_prepare_and_train.sh \
    --mode mode1 \
    --paths-config configs/paths/ev_eye_groundedsam_paths.json

  sh scripts/run_prepare_and_train.sh \
    --mode mode2 \
    --annotation-mode groundedsam \
    --prepare-arg=--interp-backend=timelens \
    --prepare-arg=--interp-target-fps=60 \
    --stage1-arg=--training.epochs=20 \
    --stage2-arg=--training.epochs=30
USAGE
}

MODE=""
ANNOTATION_MODE="auto"
STAGE1_CONFIG=""
STAGE2_CONFIG=""
STAGE1_EXPERIMENT_NAME=""
STAGE2_EXPERIMENT_NAME=""
STAGE1_CHECKPOINT=""
PATHS_CONFIG=""
EXPLICIT_PROJECT_ROOT=""
RAW_ROOT=""
CANONICAL_ROOT=""
INDEXES_ROOT=""
MANIFESTS_ROOT=""
ANNOTATION_ROOT=""
GROUNDEDSAM_ROOT=""
TIMELENS_ROOT=""
CANONICAL_NAME=""
MANIFEST_NAME=""
TRAIN_MANIFEST=""
VAL_MANIFEST=""
DEVICE=""
STAGE1_DEVICE=""
STAGE2_DEVICE=""
SKIP_PREPARE=0
SKIP_STAGE1=0
DRY_RUN=0
PREPARE_ARGS=()
STAGE1_ARGS=()
STAGE2_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --mode=*) MODE="${1#*=}"; shift ;;
        --mode) MODE="${2:-}"; shift 2 ;;
        --annotation-mode=*) ANNOTATION_MODE="${1#*=}"; shift ;;
        --annotation-mode) ANNOTATION_MODE="${2:-}"; shift 2 ;;
        --stage1-config=*) STAGE1_CONFIG="${1#*=}"; shift ;;
        --stage1-config) STAGE1_CONFIG="${2:-}"; shift 2 ;;
        --stage2-config=*) STAGE2_CONFIG="${1#*=}"; shift ;;
        --stage2-config) STAGE2_CONFIG="${2:-}"; shift 2 ;;
        --stage1-experiment-name=*) STAGE1_EXPERIMENT_NAME="$(sanitize_component "${1#*=}")"; shift ;;
        --stage1-experiment-name) STAGE1_EXPERIMENT_NAME="$(sanitize_component "${2:-}")"; shift 2 ;;
        --stage2-experiment-name=*) STAGE2_EXPERIMENT_NAME="$(sanitize_component "${1#*=}")"; shift ;;
        --stage2-experiment-name) STAGE2_EXPERIMENT_NAME="$(sanitize_component "${2:-}")"; shift 2 ;;
        --stage1-checkpoint=*) STAGE1_CHECKPOINT="${1#*=}"; shift ;;
        --stage1-checkpoint) STAGE1_CHECKPOINT="${2:-}"; shift 2 ;;
        --paths-config=*) PATHS_CONFIG="${1#*=}"; shift ;;
        --paths-config) PATHS_CONFIG="${2:-}"; shift 2 ;;
        --project-root=*) EXPLICIT_PROJECT_ROOT="${1#*=}"; shift ;;
        --project-root) EXPLICIT_PROJECT_ROOT="${2:-}"; shift 2 ;;
        --raw-root=*) RAW_ROOT="${1#*=}"; shift ;;
        --raw-root) RAW_ROOT="${2:-}"; shift 2 ;;
        --canonical-root=*) CANONICAL_ROOT="${1#*=}"; shift ;;
        --canonical-root) CANONICAL_ROOT="${2:-}"; shift 2 ;;
        --indexes-root=*) INDEXES_ROOT="${1#*=}"; shift ;;
        --indexes-root) INDEXES_ROOT="${2:-}"; shift 2 ;;
        --manifests-root=*) MANIFESTS_ROOT="${1#*=}"; shift ;;
        --manifests-root) MANIFESTS_ROOT="${2:-}"; shift 2 ;;
        --annotation-root=*) ANNOTATION_ROOT="${1#*=}"; shift ;;
        --annotation-root) ANNOTATION_ROOT="${2:-}"; shift 2 ;;
        --groundedsam-root=*) GROUNDEDSAM_ROOT="${1#*=}"; shift ;;
        --groundedsam-root) GROUNDEDSAM_ROOT="${2:-}"; shift 2 ;;
        --timelens-root=*) TIMELENS_ROOT="${1#*=}"; shift ;;
        --timelens-root) TIMELENS_ROOT="${2:-}"; shift 2 ;;
        --canonical-name=*) CANONICAL_NAME="${1#*=}"; shift ;;
        --canonical-name) CANONICAL_NAME="${2:-}"; shift 2 ;;
        --manifest-name=*) MANIFEST_NAME="${1#*=}"; shift ;;
        --manifest-name) MANIFEST_NAME="${2:-}"; shift 2 ;;
        --train-manifest=*) TRAIN_MANIFEST="${1#*=}"; shift ;;
        --train-manifest) TRAIN_MANIFEST="${2:-}"; shift 2 ;;
        --val-manifest=*) VAL_MANIFEST="${1#*=}"; shift ;;
        --val-manifest) VAL_MANIFEST="${2:-}"; shift 2 ;;
        --device=*) DEVICE="${1#*=}"; shift ;;
        --device) DEVICE="${2:-}"; shift 2 ;;
        --stage1-device=*) STAGE1_DEVICE="${1#*=}"; shift ;;
        --stage1-device) STAGE1_DEVICE="${2:-}"; shift 2 ;;
        --stage2-device=*) STAGE2_DEVICE="${1#*=}"; shift ;;
        --stage2-device) STAGE2_DEVICE="${2:-}"; shift 2 ;;
        --skip-prepare) SKIP_PREPARE=1; shift ;;
        --skip-stage1) SKIP_STAGE1=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --prepare-arg=*) PREPARE_ARGS+=("${1#*=}"); shift ;;
        --prepare-arg) PREPARE_ARGS+=("${2:-}"); shift 2 ;;
        --stage1-arg=*) STAGE1_ARGS+=("${1#*=}"); shift ;;
        --stage1-arg) STAGE1_ARGS+=("${2:-}"); shift 2 ;;
        --stage2-arg=*) STAGE2_ARGS+=("${1#*=}"); shift ;;
        --stage2-arg) STAGE2_ARGS+=("${2:-}"); shift 2 ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "[ERROR] --mode is required." >&2
    usage >&2
    exit 1
fi

case "$MODE" in
    mode0)
        DEFAULT_CANONICAL_NAME="canonical0"
        DEFAULT_MANIFEST_NAME="manifest0"
        ;;
    mode1)
        DEFAULT_CANONICAL_NAME="canonical1"
        DEFAULT_MANIFEST_NAME="manifest1"
        ;;
    mode2)
        DEFAULT_CANONICAL_NAME="canonical2"
        DEFAULT_MANIFEST_NAME="manifest2"
        ;;
    *)
        echo "[ERROR] Unsupported mode: $MODE" >&2
        exit 1
        ;;
esac

CANONICAL_NAME="${CANONICAL_NAME:-$DEFAULT_CANONICAL_NAME}"
MANIFEST_NAME="${MANIFEST_NAME:-$DEFAULT_MANIFEST_NAME}"
STAGE1_CONFIG="${STAGE1_CONFIG:-configs/${MODE}_stage1.yaml}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/${MODE}_stage2.yaml}"
if [ -z "$STAGE1_EXPERIMENT_NAME" ]; then
    STAGE1_EXPERIMENT_NAME="$(sanitize_component "${MODE}_stage1")"
fi
if [ -z "$STAGE2_EXPERIMENT_NAME" ]; then
    STAGE2_EXPERIMENT_NAME="$(sanitize_component "${MODE}_stage2")"
fi
if [ -z "$MANIFESTS_ROOT" ] && [ -n "$PATHS_CONFIG" ] && [ -f "$PATHS_CONFIG" ]; then
    MANIFESTS_ROOT=$("$PYTHON_BIN" -c 'import json, sys; data = json.load(open(sys.argv[1], encoding="utf-8")); print(data.get("manifests_root", ""))' "$PATHS_CONFIG")
fi
MANIFESTS_ROOT="${MANIFESTS_ROOT:-$PROJECT_ROOT/manifests}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFESTS_ROOT/$MANIFEST_NAME/train_manifest.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-$MANIFESTS_ROOT/$MANIFEST_NAME/val_manifest.jsonl}"
STAGE1_DEVICE="${STAGE1_DEVICE:-$DEVICE}"
STAGE2_DEVICE="${STAGE2_DEVICE:-$DEVICE}"

if [ ! -f "$STAGE1_CONFIG" ]; then
    echo "[ERROR] Stage1 config not found: $STAGE1_CONFIG" >&2
    exit 1
fi
if [ ! -f "$STAGE2_CONFIG" ]; then
    echo "[ERROR] Stage2 config not found: $STAGE2_CONFIG" >&2
    exit 1
fi

echo "[INFO] mode=$MODE"
echo "[INFO] canonical_name=$CANONICAL_NAME manifest_name=$MANIFEST_NAME"
echo "[INFO] stage1_config=$STAGE1_CONFIG"
echo "[INFO] stage2_config=$STAGE2_CONFIG"
echo "[INFO] stage1_experiment_name=$STAGE1_EXPERIMENT_NAME"
echo "[INFO] stage2_experiment_name=$STAGE2_EXPERIMENT_NAME"
echo "[INFO] train_manifest=$TRAIN_MANIFEST"
echo "[INFO] val_manifest=$VAL_MANIFEST"

if [ "$SKIP_PREPARE" -eq 0 ]; then
    prepare_cmd=(sh "$SCRIPT_DIR/run_prepare.sh" --data-mode "$MODE" --annotation-mode "$ANNOTATION_MODE" --canonical-name "$CANONICAL_NAME" --manifest-name "$MANIFEST_NAME")
    if [ -n "$PATHS_CONFIG" ]; then
        prepare_cmd+=(--paths-config "$PATHS_CONFIG")
    fi
    if [ -n "$EXPLICIT_PROJECT_ROOT" ]; then
        prepare_cmd+=(--project-root "$EXPLICIT_PROJECT_ROOT")
    fi
    if [ -n "$RAW_ROOT" ]; then
        prepare_cmd+=(--raw-root "$RAW_ROOT")
    fi
    if [ -n "$CANONICAL_ROOT" ]; then
        prepare_cmd+=(--canonical-root "$CANONICAL_ROOT")
    fi
    if [ -n "$INDEXES_ROOT" ]; then
        prepare_cmd+=(--indexes-root "$INDEXES_ROOT")
    fi
    if [ -n "$MANIFESTS_ROOT" ]; then
        prepare_cmd+=(--manifests-root "$MANIFESTS_ROOT")
    fi
    if [ -n "$ANNOTATION_ROOT" ]; then
        prepare_cmd+=(--annotation-root "$ANNOTATION_ROOT")
    fi
    if [ -n "$GROUNDEDSAM_ROOT" ]; then
        prepare_cmd+=(--groundedsam-root "$GROUNDEDSAM_ROOT")
    fi
    if [ -n "$TIMELENS_ROOT" ]; then
        prepare_cmd+=(--timelens-root "$TIMELENS_ROOT")
    fi
    for item in "${PREPARE_ARGS[@]}"; do
        prepare_cmd+=("$item")
    done
    run_cmd "${prepare_cmd[@]}"
else
    echo "[INFO] skipping prepare step"
fi

if [ "$SKIP_STAGE1" -eq 0 ]; then
    stage1_cmd=(sh "$SCRIPT_DIR/run_train.sh" --config "$STAGE1_CONFIG" --mode "$MODE" --stage stage1 --experiment-name "$STAGE1_EXPERIMENT_NAME" --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST")
    if [ -n "$STAGE1_DEVICE" ]; then
        stage1_cmd+=(--device "$STAGE1_DEVICE")
    fi
    for item in "${STAGE1_ARGS[@]}"; do
        stage1_cmd+=("$item")
    done
    run_cmd "${stage1_cmd[@]}"
else
    echo "[INFO] skipping stage1 training"
fi

if [ -z "$STAGE1_CHECKPOINT" ]; then
    if [ "$DRY_RUN" -eq 1 ] && [ "$SKIP_STAGE1" -eq 0 ]; then
        STAGE1_CHECKPOINT="AUTO_RESOLVE_AFTER_STAGE1::runs/<stage1_exp>/train/best_search_p10.pt"
        echo "[INFO] dry-run stage1_checkpoint=$STAGE1_CHECKPOINT"
    else
        stage1_run_root=$(latest_run_root "$STAGE1_EXPERIMENT_NAME" || true)
        if [ -z "$stage1_run_root" ]; then
            echo "[ERROR] Could not resolve Stage1 run root for experiment: $STAGE1_EXPERIMENT_NAME" >&2
            exit 1
        fi
        STAGE1_CHECKPOINT="$stage1_run_root/train/best_search_p10.pt"
    fi
fi

if [ "$DRY_RUN" -eq 0 ] && [ ! -f "$STAGE1_CHECKPOINT" ]; then
    echo "[ERROR] Stage1 checkpoint not found: $STAGE1_CHECKPOINT" >&2
    exit 1
fi

echo "[INFO] stage2_init_checkpoint=$STAGE1_CHECKPOINT"

stage2_cmd=(sh "$SCRIPT_DIR/run_train.sh" --config "$STAGE2_CONFIG" --mode "$MODE" --stage stage2 --experiment-name "$STAGE2_EXPERIMENT_NAME" --train-manifest "$TRAIN_MANIFEST" --val-manifest "$VAL_MANIFEST" --checkpoint "$STAGE1_CHECKPOINT")
if [ -n "$STAGE2_DEVICE" ]; then
    stage2_cmd+=(--device "$STAGE2_DEVICE")
fi
for item in "${STAGE2_ARGS[@]}"; do
    stage2_cmd+=("$item")
done
run_cmd "${stage2_cmd[@]}"

if [ "$DRY_RUN" -eq 0 ]; then
    stage2_run_root=$(latest_run_root "$STAGE2_EXPERIMENT_NAME" || true)
    if [ -n "$stage2_run_root" ]; then
        echo "[DONE] stage2_run_root=$stage2_run_root"
        echo "[DONE] stage2_best_checkpoint=$stage2_run_root/train/best_track_p10.pt"
    else
        echo "[DONE] stage2 complete (run root could not be auto-resolved)"
    fi
else
    echo "[DONE] dry-run complete"
fi
