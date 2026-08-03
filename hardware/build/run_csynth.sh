#!/bin/sh
# V3 — csynth (SPEC §9). Unlike run_tb.sh this DOES need the tool.
#
#   sh hardware/build/run_csynth.sh                    # every unit
#   sh hardware/build/run_csynth.sh mha_core mlp_core  # some
#   HBTXR_PERIOD=2.5 sh hardware/build/run_csynth.sh rmu_proj
#
# Reports land in hardware/workspace/hls/syn_<unit>/sol/syn/report/ and the log beside them.
# `python3 hardware/tools/report_csynth.py` turns the set into one table.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

# 2023.2 and NOT 2024.1, though both are installed: run_tb.sh proved bit-exactness against
# 2023.2's ap_int, and synthesizing a different header set is not the same design.
HLS_ROOT=${HLS_ROOT:-/tools/Xilinx/Vitis_HLS/2023.2}
[ -x "$HLS_ROOT/bin/vitis_hls" ] || {
  echo "no vitis_hls at $HLS_ROOT" >&2
  echo "  set HLS_ROOT, or run this under WSL where the tool is installed" >&2
  exit 2
}
# NOT `. settings64.sh` -- that script is bash-only (it calls `source`) and this one is sh,
# like run_tb.sh. The bin/vitis_hls launcher sets its own environment, so it is enough.
HLS=$HLS_ROOT/bin/vitis_hls

# One line, no continuation: a backslash-newline INSIDE `${*:-...}` is handled differently
# by dash than by bash, and this script runs under whichever /bin/sh is.
ALL="rmu_proj rmu_qkv rmu_fc1 rmu_fc2 smu_score smu_ctx layernorm softmax gelu_edge requant_edge mha_core mlp_core patch_f patch_e head_box head_ellipse"
UNITS=${*:-$ALL}
OUT=${HBTXR_OUT:-hardware/workspace/hls}
mkdir -p "$OUT"

# Bounded parallelism: each vitis_hls holds a GB or two, so all sixteen at once swaps.
JOBS=${HBTXR_JOBS:-4}
printf '%s\n' $UNITS | xargs -P "$JOBS" -I@ \
  sh -c "HBTXR_TOP=syn_@ HBTXR_OUT='$OUT' '$HLS' -f hardware/build/hls/csynth.tcl \
           >'$OUT/@.log' 2>&1 && echo '  ok   @' || echo '  FAIL @'" || true

# The log's exit status is not the check -- vitis_hls exits 0 on some synthesis failures.
# The report either exists or the unit did not synthesize.
fail=0
for u in $UNITS; do
  [ -f "$OUT/syn_$u/sol/syn/report/syn_${u}_csynth.rpt" ] || {
    echo "no report for $u -- see $OUT/$u.log" >&2
    fail=1
  }
done
[ "$fail" = 0 ] || { echo "SOME UNITS FAILED TO SYNTHESIZE" >&2; exit 1; }
echo "all units synthesized -- $OUT"
