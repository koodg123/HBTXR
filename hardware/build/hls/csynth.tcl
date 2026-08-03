# S9 — csynth ONE unit (SPEC §9).
#
# Driven by the environment so there is one script rather than sixteen that drift apart.
# `run_csynth.sh` is the intended caller; paths are relative to the repo root, which is
# where it puts you.

proc hbtxr_env {name default} {
  if {[info exists ::env($name)]} { return $::env($name) }
  return $default
}

set top [hbtxr_env HBTXR_TOP ""]
if {$top eq ""} {
  puts stderr "HBTXR_TOP is not set -- call this through hardware/build/run_csynth.sh"
  exit 2
}

# ZCU104 = XCZU7EV-2FFVC1156. The speed grade is half the timing answer, so it is named
# here rather than left to whichever board file the tool finds.
set part   [hbtxr_env HBTXR_PART   xczu7ev-ffvc1156-2-e]
set period [hbtxr_env HBTXR_PERIOD 3.333]
set out    [hbtxr_env HBTXR_OUT    hardware/workspace/hls]

open_project -reset $out/$top
set_top $top
# No `-std=c++17`. The 2023.2 front end is a clang old enough that forcing C++17 makes its
# own libstdc++ 8.3 headers fail to parse ("C++ requires a type specifier", in stl_pair.h).
# The default is C++14 and the design needs nothing past it -- `std::decay_t` is the newest
# thing in here. V1 compiles the same headers at C++17 under g++; if the two ever disagree
# it will be a testbench failure, not a silent one.
add_files hardware/module/syn/hbtxr_syn.cpp \
  -cflags "-Ihardware/config/design -Ihardware/module/include"

open_solution -reset sol -flow_target vivado
set_part $part
create_clock -period $period -name default

csynth_design
exit
