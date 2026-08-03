#!/bin/sh
# V1 — compile and run a testbench against the integer golden (SPEC §9).
#
# g++ with the Vitis HLS headers, not vitis_hls: hls::stream and hls::vector compile and
# run under plain g++, so V1 needs no tool licence and no project. csim/csynth (V2/V3)
# come with S8/S9.
#
#   sh hardware/build/run_tb.sh              # every testbench
#   sh hardware/build/run_tb.sh requant      # one
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

XILINX_INCLUDE=${XILINX_INCLUDE:-/tools/Xilinx/Vitis_HLS/2023.2/include}
[ -d "$XILINX_INCLUDE" ] || {
  echo "no Vitis HLS headers at $XILINX_INCLUDE" >&2
  echo "  set XILINX_INCLUDE, or run this under WSL where they are installed" >&2
  exit 2
}

GOLDEN=hardware/workspace/golden
MODEL=$GOLDEN/model-hbtxr-w4s4h8
[ -d "$GOLDEN/requant" ] && [ -d "$GOLDEN/search-a4" ] && [ -d "$GOLDEN/track-a4" ] &&
  [ -d "$MODEL" ] || {
  echo "golden missing -- generating"
  sh hardware/build/make_golden.sh requant search-4 track-4 model
}

OUT=hardware/workspace/tb
mkdir -p "$OUT"
fail=0

for tb in ${*:-requant gelu layernorm rmu smu softmax patch mha mlp block backbone top}; do
  src="hardware/module/tb/tb_$tb.cpp"
  [ -f "$src" ] || { echo "no such testbench: $src" >&2; exit 2; }
  # -isystem, not -I: the Vitis headers emit -Wall noise of their own (multi-line comment
  # art, signed/unsigned loops) that would bury a warning in OUR code.
  # -Wno-unknown-pragmas: g++ does not know `#pragma HLS`, which is the point.
  # -Wno-unused-label: HLS loop labels are read by directives and the schedule report;
  # to g++ they are dead.
  g++ -std=c++17 -O1 -Wall -Wextra -Wno-unknown-pragmas -Wno-unused-label \
      -isystem "$XILINX_INCLUDE" \
      -Ihardware/config/design -Ihardware/module/include -Ihardware/module/tb \
      "$src" -o "$OUT/tb_$tb"

  # Both token counts, same binary. The backbone is SHARED (SPEC §7): search runs it at
  # N=64 and track at N=16 on the same weights, so "it works at one N" is not the claim.
  case "$tb" in
    requant)  "$OUT/tb_$tb" || fail=1 ;;
    # The backbone runs the eight-block stack, so it needs the MODEL golden -- and it
    # runs search/track/search itself, because state that only matters at the other
    # token count is invisible in one order.
    backbone|top) "$OUT/tb_$tb" "$MODEL" || fail=1 ;;
    *) for g in search-a4 track-a4; do "$OUT/tb_$tb" "$GOLDEN/$g" || fail=1; done ;;
  esac
done

[ "$fail" = 0 ] || { echo "SOME TESTBENCHES FAILED" >&2; exit 1; }
