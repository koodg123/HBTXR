set hw_dir "hardware"
if {![file exists [file join $hw_dir hls src hgtxr_e2e_m_axi_top.cpp]]} {
  set hw_dir "."
}
set axis_script [file join $hw_dir vivado scripts run_e2e_q4w8a_csim.tcl]
set fd [open $axis_script r]
set script_text [read $fd]
close $fd
if {![info exists ::env(HGTXR_E2E_PROJECT_NAME)] || $::env(HGTXR_E2E_PROJECT_NAME) eq ""} {
  set ::env(HGTXR_E2E_PROJECT_NAME) hgtxr_e2e_m_axi_hls
}
regsub {set_top hgtxr_e2e_axis_top} $script_text {set_top hgtxr_e2e_m_axi_top} script_text
regsub {add_files -cflags \$cxx_flags \[file join \$hw_dir hls src hgtxr_e2e_axis_top.cpp\]} $script_text {add_files -cflags $cxx_flags [file join $hw_dir hls src hgtxr_e2e_m_axi_top.cpp]} script_text
regsub {add_files -tb -cflags \$cxx_flags \[file join \$hw_dir hls tb tb_hgtxr_e2e_axis_top.cpp\]} $script_text {add_files -tb -cflags $cxx_flags [file join $hw_dir hls tb tb_hgtxr_e2e_m_axi_top.cpp]} script_text
eval $script_text
