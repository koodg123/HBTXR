# Run E2E AXIS Q4W8A HLS RTL co-simulation.
#
# This intentionally reuses the csynth project setup so the cosim build has the
# same source files, target part, clock, scale flags, resource policy, and
# runtime-mode Search/Track testbench coverage as synthesis.

set hw_dir "hardware"
if {![file exists [file join $hw_dir vivado scripts run_e2e_q4w8a_csynth.tcl]]} {
  set hw_dir "."
}

set csynth_script [file join $hw_dir vivado scripts run_e2e_q4w8a_csynth.tcl]
set fp [open $csynth_script r]
set script_text [read $fp]
close $fp

# Host shell build flags can leak into the generated cosim Makefile. In this
# environment DEBUG=release is interpreted as a bare compiler input named
# "release", so keep the HLS cosim build environment explicit and minimal.
foreach name {DEBUG CFLAGS CXXFLAGS CPPFLAGS LDFLAGS LIBRARY_PATH} {
  if {[info exists ::env($name)]} {
    unset ::env($name)
  }
}

set xilinx_root "/tools/Xilinx"
if {[info exists ::env(HGTXR_XILINX_ROOT)]} {
  set xilinx_root $::env(HGTXR_XILINX_ROOT)
}

set xsim_lib_paths [list \
  [file join $xilinx_root Vivado 2023.2 lib lnx64.o Rhel 9] \
  [file join $xilinx_root Vivado 2023.2 lib lnx64.o] \
  [file join $xilinx_root Vitis_HLS 2023.2 lib lnx64.o Rhel 9] \
]
if {[info exists ::env(LD_LIBRARY_PATH)] && $::env(LD_LIBRARY_PATH) ne ""} {
  lappend xsim_lib_paths $::env(LD_LIBRARY_PATH)
}
set ::env(LD_LIBRARY_PATH) [join $xsim_lib_paths ":"]
set ::env(TERM) "xterm"
set ::env(TERMINFO) "/lib/terminfo"

if {![regsub {csynth_design[ \t\r\n]+exit} $script_text {csynth_design
config_cosim -disable_deadlock_detection
cosim_design -rtl verilog -trace_level none
exit} script_text]} {
  error "Failed to patch csynth script into cosim script: $csynth_script"
}

eval $script_text
