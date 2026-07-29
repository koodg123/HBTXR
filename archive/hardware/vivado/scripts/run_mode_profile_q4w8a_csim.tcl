set hw_dir "hardware"
if {![file exists [file join $hw_dir hls src hgtxr_mode_profile_top.cpp]]} {
  set hw_dir "."
}
set hls_include [file normalize [file join $hw_dir hls include]]
set mode_config [file normalize [file join $hw_dir configs zcu104_e2e_q4w8a_defines.h]]

proc hgtxr_env_default {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default
}

set top_name [hgtxr_env_default HGTXR_MODE_PROFILE_TOP hgtxr_search_profile_top]
if {$top_name ne "hgtxr_search_profile_top" && $top_name ne "hgtxr_track_profile_top"} {
  error "Unsupported HGTXR_MODE_PROFILE_TOP: $top_name"
}

set project_name [hgtxr_env_default HGTXR_MODE_PROFILE_PROJECT_NAME hgtxr_mode_profile_csim]
if {![regexp {^[A-Za-z0-9_]+$} $project_name]} {
  error "Unsupported HGTXR_MODE_PROFILE_PROJECT_NAME: $project_name"
}

set mode_par [hgtxr_env_default HGTXR_MODE_PROFILE_PAR 16]
if {$mode_par ne "4" && $mode_par ne "8" && $mode_par ne "16" && $mode_par ne "32"} {
  error "Unsupported HGTXR_MODE_PROFILE_PAR: $mode_par"
}

set xilinx_root [hgtxr_env_default HGTXR_XILINX_ROOT "/tools/Xilinx"]
set gcc_include [hgtxr_env_default HGTXR_GCC_INCLUDE "/usr/lib/gcc/x86_64-linux-gnu/13/include"]
set gcc_lib [hgtxr_env_default HGTXR_GCC_LIB "/usr/lib/gcc/x86_64-linux-gnu/13"]
set sys_include [hgtxr_env_default HGTXR_SYS_INCLUDE "/usr/include/x86_64-linux-gnu"]
set sys_lib [hgtxr_env_default HGTXR_SYS_LIB "/usr/lib/x86_64-linux-gnu"]

puts "HGTXR_MODE_PROFILE_PROJECT=$project_name"
puts "HGTXR_MODE_PROFILE_TOP=$top_name"
puts "HGTXR_MODE_PROFILE_PAR=$mode_par"

open_project -reset [file join $hw_dir generated $project_name]
set_top $top_name
set cxx_flags "-std=c++17 -I$hls_include -I$sys_include -I$gcc_include -DHGTXR_MODE_PROFILE_PAR=$mode_par -DHGTXR_PARALLELISM_FACTOR=$mode_par -include $mode_config"
add_files -cflags $cxx_flags [file join $hw_dir hls src hgtxr_mode_profile_top.cpp]
add_files -cflags $cxx_flags [file join $hw_dir hls src search_head.cpp]
add_files -cflags $cxx_flags [file join $hw_dir hls src track_head.cpp]
add_files -cflags $cxx_flags [file join $hw_dir hls src fusion.cpp]
add_files -tb -cflags $cxx_flags [file join $hw_dir hls tb tb_hgtxr_mode_profile_top.cpp]
open_solution -reset solution_mode_q4w8a
set_part xczu7ev-ffvc1156-2-e
create_clock -period 5.0
set ::env(LIBRARY_PATH) $sys_lib:$gcc_lib:/lib/x86_64-linux-gnu
set ::env(LD_LIBRARY_PATH) $sys_lib:$gcc_lib:/lib/x86_64-linux-gnu
set ::env(PATH) /usr/bin:/bin:$xilinx_root/Vitis_HLS/2023.2/bin:$xilinx_root/Vivado/2023.2/bin
csim_design
exit
