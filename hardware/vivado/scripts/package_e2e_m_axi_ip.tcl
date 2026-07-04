# Package the selected A2 E2E memory-mapped HLS top as Vivado IP.
#
# Default run:
#   HGTXR_E2E_SCALE=active196_b6_ff768 HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream \
#     /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/package_e2e_m_axi_ip.tcl

set hw_dir "hardware"
if {![file exists [file join $hw_dir hls src hgtxr_e2e_m_axi_top.cpp]]} {
  set hw_dir "."
}

set axis_script [file join $hw_dir vivado scripts run_e2e_q4w8a_csynth.tcl]
set fd [open $axis_script r]
set script_text [read $fd]
close $fd

if {![info exists ::env(HGTXR_E2E_PROJECT_NAME)] || $::env(HGTXR_E2E_PROJECT_NAME) eq ""} {
  set ::env(HGTXR_E2E_PROJECT_NAME) hgtxr_e2e_m_axi_hls
}
regsub {set_top hgtxr_e2e_axis_top} $script_text {set_top hgtxr_e2e_m_axi_top} script_text
regsub {add_files -cflags \$cxx_flags \[file join \$hw_dir hls src hgtxr_e2e_axis_top.cpp\]} $script_text {add_files -cflags $cxx_flags [file join $hw_dir hls src hgtxr_e2e_m_axi_top.cpp]} script_text
regsub {add_files -tb -cflags \$cxx_flags \[file join \$hw_dir hls tb tb_hgtxr_e2e_axis_top.cpp\]} $script_text {add_files -tb -cflags $cxx_flags [file join $hw_dir hls tb tb_hgtxr_e2e_m_axi_top.cpp]} script_text
regsub {csynth_design[ \t\r\n]+exit} $script_text {csynth_design
export_design -format ip_catalog
exit} script_text

eval $script_text
