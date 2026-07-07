#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$SCRIPT_DIR/_bootstrap.sh"

ACTION="${1:-help}"
case "$ACTION" in
    prepare|stage1|stage2|help|-h|--help) ;;
    *)
        echo "[ERROR] Unknown action: $ACTION" >&2
        exit 1
        ;;
esac
shift || true

RAW_ROOT="${RAW_ROOT:-/mnt/d/dataset/EV_Eye/raw_data/Data_davis}"
CANONICAL_WORKSPACE="${CANONICAL_WORKSPACE:-$PROJECT_ROOT/dataset}"
MANIFESTS_ROOT="${MANIFESTS_ROOT:-$PROJECT_ROOT/manifests}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ANNOTATION_MODE="${ANNOTATION_MODE:-manual_csv}"
PRESET="${PRESET:-all}"
CHECKPOINT_OVERRIDE="${CHECKPOINT_OVERRIDE:-}"

usage() {
    cat <<'USAGE'
Usage:
  sh exps/scripts/run_mode1_event_sweep.sh <prepare|stage1|stage2> [options]

Options:
  --preset fc5000|fc8000|tb5000|all
  --raw-root PATH
  --canonical-workspace PATH
  --manifests-root PATH
  --device DEVICE
  --num-workers N
  --annotation-mode manual_csv|auto
  --checkpoint PATH

Examples:
  sh exps/scripts/run_mode1_event_sweep.sh prepare --preset all
  sh exps/scripts/run_mode1_event_sweep.sh stage1 --preset fc5000 --device cuda:0
  sh exps/scripts/run_mode1_event_sweep.sh stage2 --preset fc5000 --checkpoint runs/mode1_fc5000_stage1_YYYYMMDD_HHMMSS/train/best_search_p10.pt
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --preset)
            PRESET="${2:-}"
            shift 2
            ;;
        --preset=*)
            PRESET="${1#*=}"
            shift
            ;;
        --raw-root)
            RAW_ROOT="${2:-}"
            shift 2
            ;;
        --raw-root=*)
            RAW_ROOT="${1#*=}"
            shift
            ;;
        --canonical-workspace)
            CANONICAL_WORKSPACE="${2:-}"
            shift 2
            ;;
        --canonical-workspace=*)
            CANONICAL_WORKSPACE="${1#*=}"
            shift
            ;;
        --manifests-root)
            MANIFESTS_ROOT="${2:-}"
            shift 2
            ;;
        --manifests-root=*)
            MANIFESTS_ROOT="${1#*=}"
            shift
            ;;
        --device)
            DEVICE="${2:-}"
            shift 2
            ;;
        --device=*)
            DEVICE="${1#*=}"
            shift
            ;;
        --num-workers)
            NUM_WORKERS="${2:-}"
            shift 2
            ;;
        --num-workers=*)
            NUM_WORKERS="${1#*=}"
            shift
            ;;
        --annotation-mode)
            ANNOTATION_MODE="${2:-}"
            shift 2
            ;;
        --annotation-mode=*)
            ANNOTATION_MODE="${1#*=}"
            shift
            ;;
        --checkpoint)
            CHECKPOINT_OVERRIDE="${2:-}"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_OVERRIDE="${1#*=}"
            shift
            ;;
        help|-h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

resolve_presets() {
    case "$PRESET" in
        all) printf '%s\n' fc5000 fc8000 tb5000 ;;
        fc5000|fc8000|tb5000) printf '%s\n' "$PRESET" ;;
        *)
            echo "[ERROR] Unsupported preset: $PRESET" >&2
            exit 1
            ;;
    esac
}

preset_vars() {
    local preset="$1"
    case "$preset" in
        fc5000)
            CANONICAL_NAME="canonical1_fc5000"
            MANIFEST_NAME="manifest1_fc5000"
            EXPERIMENT_PREFIX="mode1_fc5000"
            EVENT_POLICY="fixed_count"
            EVENT_COUNT_TARGET="5000"
            TIME_BIN_US="5000"
            ACCUMULATION="fast_causal_linear"
            ;;
        fc8000)
            CANONICAL_NAME="canonical1_fc8000"
            MANIFEST_NAME="manifest1_fc8000"
            EXPERIMENT_PREFIX="mode1_fc8000"
            EVENT_POLICY="fixed_count"
            EVENT_COUNT_TARGET="8000"
            TIME_BIN_US="5000"
            ACCUMULATION="fast_causal_linear"
            ;;
        tb5000)
            CANONICAL_NAME="canonical1_tb5000"
            MANIFEST_NAME="manifest1_tb5000"
            EXPERIMENT_PREFIX="mode1_tb5000"
            EVENT_POLICY="time_bin"
            EVENT_COUNT_TARGET="5000"
            TIME_BIN_US="5000"
            ACCUMULATION="fast_causal_linear"
            ;;
        *)
            echo "[ERROR] Unsupported preset: $preset" >&2
            exit 1
            ;;
    esac
}

latest_stage1_checkpoint() {
    local prefix="$1"
    local candidate=""
    local latest=""
    for candidate in "$PROJECT_ROOT"/runs/"${prefix}_stage1"_*/train/best_search_p10.pt; do
        if [ -f "$candidate" ]; then
            latest="$candidate"
        fi
    done
    printf '%s\n' "$latest"
}

run_prepare_preset() {
    local preset="$1"
    preset_vars "$preset"
    echo "[PREPARE] preset=$preset canonical=$CANONICAL_NAME manifest=$MANIFEST_NAME"
    sh "$PROJECT_ROOT/scripts/run_prepare.sh" \
        --raw-root "$RAW_ROOT" \
        --canonical-root "$CANONICAL_WORKSPACE" \
        --manifests-root "$MANIFESTS_ROOT" \
        --data-mode mode1 \
        --annotation-mode "$ANNOTATION_MODE" \
        --canonical-name "$CANONICAL_NAME" \
        --manifest-name "$MANIFEST_NAME" \
        --frame-source original \
        --event-policy "$EVENT_POLICY" \
        --time-bin-us "$TIME_BIN_US" \
        --event-count-target "$EVENT_COUNT_TARGET" \
        --accumulation "$ACCUMULATION" \
        --num-workers "$NUM_WORKERS"
}

run_stage1_preset() {
    local preset="$1"
    preset_vars "$preset"
    echo "[STAGE1] preset=$preset experiment=${EXPERIMENT_PREFIX}_stage1"
    sh "$PROJECT_ROOT/scripts/run_train.sh" \
        --config configs/mode1_stage1.yaml \
        --mode mode1 \
        --stage stage1 \
        --device "$DEVICE" \
        --train-manifest "$MANIFESTS_ROOT/$MANIFEST_NAME/train_manifest.jsonl" \
        --val-manifest "$MANIFESTS_ROOT/$MANIFEST_NAME/val_manifest.jsonl" \
        --experiment-name "${EXPERIMENT_PREFIX}_stage1" \
        --override "data.canonical_root=$CANONICAL_WORKSPACE" \
        --override "data.mode1.canonical_name=$CANONICAL_NAME" \
        --override "data.mode1.manifest_name=$MANIFEST_NAME" \
        --override "data.mode1.event_builder.policy=$EVENT_POLICY" \
        --override "data.mode1.event_builder.time_bin_us=$TIME_BIN_US" \
        --override "data.mode1.event_builder.event_count_target=$EVENT_COUNT_TARGET" \
        --override "data.mode1.event_builder.accumulation=$ACCUMULATION"
}

run_stage2_preset() {
    local preset="$1"
    local checkpoint=""
    preset_vars "$preset"
    checkpoint="$CHECKPOINT_OVERRIDE"
    if [ -z "$checkpoint" ]; then
        checkpoint="$(latest_stage1_checkpoint "$EXPERIMENT_PREFIX")"
    fi
    if [ -z "$checkpoint" ] || [ ! -f "$checkpoint" ]; then
        echo "[ERROR] Stage2 requires a valid Stage1 checkpoint for preset '$preset'." >&2
        echo "[ERROR] Pass --checkpoint or run stage1 first." >&2
        exit 1
    fi
    echo "[STAGE2] preset=$preset experiment=${EXPERIMENT_PREFIX}_stage2 checkpoint=$checkpoint"
    sh "$PROJECT_ROOT/scripts/run_train.sh" \
        --config configs/mode1_stage2.yaml \
        --mode mode1 \
        --stage stage2 \
        --device "$DEVICE" \
        --train-manifest "$MANIFESTS_ROOT/$MANIFEST_NAME/train_manifest.jsonl" \
        --val-manifest "$MANIFESTS_ROOT/$MANIFEST_NAME/val_manifest.jsonl" \
        --checkpoint "$checkpoint" \
        --experiment-name "${EXPERIMENT_PREFIX}_stage2" \
        --override "data.canonical_root=$CANONICAL_WORKSPACE" \
        --override "data.mode1.canonical_name=$CANONICAL_NAME" \
        --override "data.mode1.manifest_name=$MANIFEST_NAME" \
        --override "data.mode1.event_builder.policy=$EVENT_POLICY" \
        --override "data.mode1.event_builder.time_bin_us=$TIME_BIN_US" \
        --override "data.mode1.event_builder.event_count_target=$EVENT_COUNT_TARGET" \
        --override "data.mode1.event_builder.accumulation=$ACCUMULATION"
}

case "$ACTION" in
    help|-h|--help)
        usage
        ;;
    prepare)
        while IFS= read -r preset; do
            run_prepare_preset "$preset"
        done < <(resolve_presets)
        ;;
    stage1)
        while IFS= read -r preset; do
            run_stage1_preset "$preset"
        done < <(resolve_presets)
        ;;
    stage2)
        while IFS= read -r preset; do
            run_stage2_preset "$preset"
        done < <(resolve_presets)
        ;;
esac
