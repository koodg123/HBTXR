#!/bin/sh
. "$(dirname "$0")/_common.sh"
exec vitis_hls -f hardware/vivado/scripts/run_csynth.tcl

