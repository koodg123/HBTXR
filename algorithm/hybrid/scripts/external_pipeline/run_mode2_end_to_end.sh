#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT"

DEFAULT_PATHS_CONFIG="$PROJECT_ROOT/configs/paths/ev_eye_groundedsam_paths.json"

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
  sh scripts/run_mode2_end_to_end.sh [options]

This wrapper performs, in order:
  1. Grounded-SAM annotation on the raw dataset
  2. mode2 canonical + manifest generation
  3. mode2 Stage1 training
  4. mode2 Stage2 training initialized from Stage1 best_search_p10.pt

Defaults:
  - always uses `mode2`
  - automatically adds `--paths-config configs/paths/ev_eye_groundedsam_paths.json` when that file exists

Annotation options:
  --skip-annotation
  --overwrite-annotation
  --annotation-device DEVICE
  --groundingdino-config PATH
  --groundingdino-checkpoint PATH
  --sam-checkpoint PATH
  --sam-encoder-version NAME
  --classes TEXT
  --box-threshold FLOAT
  --text-threshold FLOAT
  --nms-threshold FLOAT
  --min-mask-area INT
  --frame-step INT
  --max-frames-per-session INT
  --max-sessions INT
  --annotation-arg ARG              Extra arg appended to run_groundedsam_annotation.sh (repeatable)

Shared options:
  --paths-config PATH
  --dry-run

All other options are forwarded to:
  sh scripts/run_prepare_and_train_mode2.sh ...

Examples:
  sh scripts/run_mode2_end_to_end.sh \
    --groundingdino-checkpoint /path/to/groundingdino_swint_ogc.pth \
    --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
    --annotation-device cuda:0,cuda:1 \
    --stage1-device cuda:0,cuda:1 \
    --stage2-device cuda:0,cuda:1

  sh scripts/run_mode2_end_to_end.sh \
    --skip-annotation \
    --prepare-arg=--interp-backend=timelens \
    --prepare-arg=--interp-target-fps=60
USAGE
}

PATHS_CONFIG=""
SKIP_ANNOTATION=0
OVERWRITE_ANNOTATION=0
DRY_RUN=0
ANNOTATION_ARGS=()
FORWARD_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --paths-config=*)
            PATHS_CONFIG="${1#*=}"
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --paths-config)
            PATHS_CONFIG="${2:-}"
            FORWARD_ARGS+=("$1" "${2:-}")
            shift 2
            ;;
        --skip-annotation)
            SKIP_ANNOTATION=1
            shift
            ;;
        --overwrite-annotation)
            OVERWRITE_ANNOTATION=1
            shift
            ;;
        --annotation-device=*)
            ANNOTATION_ARGS+=(--device "${1#*=}")
            shift
            ;;
        --annotation-device)
            ANNOTATION_ARGS+=(--device "${2:-}")
            shift 2
            ;;
        --groundingdino-config=*|--groundingdino-checkpoint=*|--sam-checkpoint=*|--sam-encoder-version=*|--classes=*|--box-threshold=*|--text-threshold=*|--nms-threshold=*|--min-mask-area=*|--frame-step=*|--max-frames-per-session=*|--max-sessions=*)
            ANNOTATION_ARGS+=("$1")
            shift
            ;;
        --groundingdino-config|--groundingdino-checkpoint|--sam-checkpoint|--sam-encoder-version|--classes|--box-threshold|--text-threshold|--nms-threshold|--min-mask-area|--frame-step|--max-frames-per-session|--max-sessions)
            ANNOTATION_ARGS+=("$1" "${2:-}")
            shift 2
            ;;
        --annotation-arg=*)
            ANNOTATION_ARGS+=("${1#*=}")
            shift
            ;;
        --annotation-arg)
            ANNOTATION_ARGS+=("${2:-}")
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --*=*)
            FORWARD_ARGS+=("$1")
            shift
            ;;
        --*)
            if [ $# -gt 1 ] && [[ "${2:-}" != --* ]]; then
                FORWARD_ARGS+=("$1" "${2:-}")
                shift 2
            else
                FORWARD_ARGS+=("$1")
                shift
            fi
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [ -z "$PATHS_CONFIG" ] && [ -f "$DEFAULT_PATHS_CONFIG" ]; then
    PATHS_CONFIG="$DEFAULT_PATHS_CONFIG"
    FORWARD_ARGS=(--paths-config "$PATHS_CONFIG" "${FORWARD_ARGS[@]}")
fi

if [ "$SKIP_ANNOTATION" -eq 0 ]; then
    annotation_cmd=(sh "$SCRIPT_DIR/run_groundedsam_annotation.sh")
    if [ -n "$PATHS_CONFIG" ]; then
        annotation_cmd+=(--paths-config "$PATHS_CONFIG")
    fi
    if [ "$OVERWRITE_ANNOTATION" -eq 1 ]; then
        annotation_cmd+=(--overwrite)
    fi
    for item in "${ANNOTATION_ARGS[@]}"; do
        annotation_cmd+=("$item")
    done
    run_cmd "${annotation_cmd[@]}"
else
    echo "[INFO] skipping Grounded-SAM annotation"
fi

train_cmd=(sh "$SCRIPT_DIR/run_prepare_and_train_mode2.sh" "${FORWARD_ARGS[@]}")
run_cmd "${train_cmd[@]}"
