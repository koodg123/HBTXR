#!/bin/sh
. "$(dirname "$0")/_common.sh"
if command -v vitis_hls >/dev/null 2>&1; then
  exec vitis_hls -f hardware/vivado/scripts/run_csim.tcl
fi
echo "vitis_hls not found; compile tb_hgtxr_top.cpp with g++ for a local syntax smoke."
exec g++ -std=c++17 -Ihardware/hls/include hardware/hls/tb/tb_hgtxr_top.cpp hardware/hls/src/*.cpp -o /tmp/hgtxr_tb && /tmp/hgtxr_tb

