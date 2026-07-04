#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$ROOT_DIR"

MODE=${1:-csynth}
PROFILE=${2:-search_par16}

case "$MODE" in
  csim)
    HLS_TCL="vivado/scripts/run_mode_profile_q4w8a_csim.tcl"
    ;;
  csynth)
    HLS_TCL="vivado/scripts/run_mode_profile_q4w8a_csynth.tcl"
    ;;
  package)
    HLS_TCL="vivado/scripts/package_mode_profile_ip.tcl"
    ;;
  *)
    echo "usage: $0 {csim|csynth|package} [profile]" >&2
    echo "profiles: search_par16 search_par32 track_par16 track_par32" >&2
    exit 2
    ;;
esac

case "$PROFILE" in
  search_par16)
    PROJECT="hgtxr_mode_search_par16_no_board"
    TOP="hgtxr_search_profile_top"
    MODE_PAR=16
    TARGET_MS="4.000"
    ;;
  search_par32)
    PROJECT="hgtxr_mode_search_par32_no_board"
    TOP="hgtxr_search_profile_top"
    MODE_PAR=32
    TARGET_MS="4.000"
    ;;
  track_par16)
    PROJECT="hgtxr_mode_track_par16_no_board"
    TOP="hgtxr_track_profile_top"
    MODE_PAR=16
    TARGET_MS="1.000"
    ;;
  track_par32)
    PROJECT="hgtxr_mode_track_par32_no_board"
    TOP="hgtxr_track_profile_top"
    MODE_PAR=32
    TARGET_MS="1.000"
    ;;
  *)
    echo "unknown profile: $PROFILE" >&2
    echo "profiles: search_par16 search_par32 track_par16 track_par32" >&2
    exit 2
    ;;
esac

if [ -n "${HGTXR_MODE_PROFILE_PROJECT_NAME:-}" ]; then
  PROJECT="$HGTXR_MODE_PROFILE_PROJECT_NAME"
fi
if [ -n "${HGTXR_MODE_PROFILE_TOP:-}" ]; then
  TOP="$HGTXR_MODE_PROFILE_TOP"
fi
if [ -n "${HGTXR_MODE_PROFILE_PAR:-}" ]; then
  MODE_PAR="$HGTXR_MODE_PROFILE_PAR"
fi

HLS_BIN=${VITIS_HLS_BIN:-}
if [ -z "$HLS_BIN" ]; then
  if command -v vitis_hls >/dev/null 2>&1; then
    HLS_BIN=$(command -v vitis_hls)
  elif [ -x /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls ]; then
    HLS_BIN=/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls
  else
    echo "vitis_hls not found. Set VITIS_HLS_BIN or source Xilinx environment." >&2
    exit 127
  fi
fi

XILINX_ROOT=${HGTXR_XILINX_ROOT:-/tools/Xilinx}
VITIS_HLS_LIB_DIR="$XILINX_ROOT/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9"
if [ -d "$VITIS_HLS_LIB_DIR" ]; then
  export LD_LIBRARY_PATH="$VITIS_HLS_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

REPORT_DIR="generated/$PROJECT/solution_mode_q4w8a/syn/report"

echo "mode=$MODE"
echo "profile=$PROFILE"
echo "project=$PROJECT"
echo "top=$TOP"
echo "mode_profile_par=$MODE_PAR"
echo "target_ms=$TARGET_MS"
echo "hls_bin=$HLS_BIN"
echo "tcl=$HLS_TCL"
if [ "$MODE" = "csynth" ]; then
  echo "expected_report=$REPORT_DIR/${TOP}_csynth.rpt"
elif [ "$MODE" = "package" ]; then
  echo "expected_ip=generated/$PROJECT/solution_mode_q4w8a/impl/ip/component.xml"
fi

export HGTXR_MODE_PROFILE_PROJECT_NAME="$PROJECT"
export HGTXR_MODE_PROFILE_TOP="$TOP"
export HGTXR_MODE_PROFILE_PAR="$MODE_PAR"

exec "$HLS_BIN" -f "$HLS_TCL"
