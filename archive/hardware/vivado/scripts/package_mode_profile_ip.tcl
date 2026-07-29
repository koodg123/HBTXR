# Package the selected Search/Track no-board HLS profile top as Vivado IP.

set hw_dir "hardware"
if {![file exists [file join $hw_dir hls src hgtxr_mode_profile_top.cpp]]} {
  set hw_dir "."
}

set mode_script [file join $hw_dir vivado scripts run_mode_profile_q4w8a_csynth.tcl]
set fd [open $mode_script r]
set script_text [read $fd]
close $fd

if {![info exists ::env(HGTXR_MODE_PROFILE_PROJECT_NAME)] || $::env(HGTXR_MODE_PROFILE_PROJECT_NAME) eq ""} {
  set ::env(HGTXR_MODE_PROFILE_PROJECT_NAME) hgtxr_mode_profile_ip
}

regsub {csynth_design[ \t\r\n]+exit} $script_text {csynth_design
export_design -format ip_catalog
exit} script_text

eval $script_text
