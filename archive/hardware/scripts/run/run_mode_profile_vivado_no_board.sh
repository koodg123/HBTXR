#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT_DIR"

PROFILE=${1:-track_par32}

case "$PROFILE" in
  search_par32)
    PROJECT_NAME="hgtxr_mode_search_par32_overlay"
    ARTIFACT_NAME="hgtxr_mode_search_par32"
    HLS_TOP="hgtxr_search_profile_top"
    HLS_IP_REPO="generated/hgtxr_mode_search_par32_no_board/solution_mode_q4w8a/impl/ip"
    ;;
  track_par32)
    PROJECT_NAME="hgtxr_mode_track_par32_overlay"
    ARTIFACT_NAME="hgtxr_mode_track_par32"
    HLS_TOP="hgtxr_track_profile_top"
    HLS_IP_REPO="generated/hgtxr_mode_track_par32_no_board/solution_mode_q4w8a/impl/ip"
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    echo "profiles: search_par32 track_par32" >&2
    exit 2
    ;;
esac

if [ -n "${HGTXR_VIVADO_PROJECT_NAME:-}" ]; then
  PROJECT_NAME="$HGTXR_VIVADO_PROJECT_NAME"
fi
if [ -n "${HGTXR_VIVADO_ARTIFACT_NAME:-}" ]; then
  ARTIFACT_NAME="$HGTXR_VIVADO_ARTIFACT_NAME"
fi
if [ -n "${HGTXR_MODE_HLS_TOP:-}" ]; then
  HLS_TOP="$HGTXR_MODE_HLS_TOP"
fi
if [ -n "${HGTXR_HLS_IP_REPO:-}" ]; then
  HLS_IP_REPO="$HGTXR_HLS_IP_REPO"
fi

VIVADO_BIN=${VIVADO_BIN:-}
if [ -z "$VIVADO_BIN" ]; then
  if command -v vivado >/dev/null 2>&1; then
    VIVADO_BIN=$(command -v vivado)
  elif [ -x /tools/Xilinx/Vivado/2023.2/bin/vivado ]; then
    VIVADO_BIN=/tools/Xilinx/Vivado/2023.2/bin/vivado
  else
    echo "vivado not found. Set VIVADO_BIN or source Xilinx environment." >&2
    exit 127
  fi
fi

XILINX_ROOT=${HGTXR_XILINX_ROOT:-/tools/Xilinx}
VIVADO_LIB_DIR="$XILINX_ROOT/Vivado/2023.2/lib/lnx64.o/Rhel/9"
VITIS_HLS_LIB_DIR="$XILINX_ROOT/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9"
if [ -d "$VIVADO_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$VIVADO_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi
if [ -d "$VITIS_HLS_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$VITIS_HLS_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

OUT_DIR=${HGTXR_VIVADO_OUT_DIR:-generated/build/vivado}
JOBS=${HGTXR_VIVADO_JOBS:-4}

echo "profile=$PROFILE"
echo "project_name=$PROJECT_NAME"
echo "artifact_name=$ARTIFACT_NAME"
echo "hls_top=$HLS_TOP"
echo "hls_ip_repo=$HLS_IP_REPO"
echo "out_dir=$OUT_DIR"
echo "jobs=$JOBS"
echo "vivado_bin=$VIVADO_BIN"
echo "expected_bit=$OUT_DIR/overlay/$PROJECT_NAME/$ARTIFACT_NAME.bit"
echo "expected_hwh=$OUT_DIR/overlay/$PROJECT_NAME/$ARTIFACT_NAME.hwh"
echo "expected_timing=$OUT_DIR/overlay/$PROJECT_NAME/reports/${ARTIFACT_NAME}_timing_summary.rpt"

if [ ! -f "$HLS_IP_REPO/component.xml" ]; then
  echo "missing packaged HLS IP: $HLS_IP_REPO/component.xml" >&2
  echo "run: scripts/run/run_mode_profile_no_board.sh package $PROFILE" >&2
  exit 3
fi

exec "$VIVADO_BIN" -mode batch -source vivado/scripts/build_mode_profile_bitstream.tcl -tclargs \
  -project_name "$PROJECT_NAME" \
  -artifact_name "$ARTIFACT_NAME" \
  -hls_top "$HLS_TOP" \
  -hls_ip_repo "$HLS_IP_REPO" \
  -out_dir "$OUT_DIR" \
  -jobs "$JOBS"
