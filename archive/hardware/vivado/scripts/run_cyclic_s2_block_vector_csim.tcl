# Vitis HLS C-simulation gate for the S2 block vector comparator.
# This is a debug/validation target, not a synthesis target.

set project_name "hardware/generated/hgtxr_s2_block_vector_hls"
set solution_name "solution_s2_block_vector_csim"
set top_name "hgtxr_s2_block_vector_debug"
set part_name "xczu7ev-ffvc1156-2-e"
set clock_period 5.0

set hw_dir "hardware"
if {![file exists [file join $hw_dir hls include]]} {
  set hw_dir "."
}
set root_dir [file normalize [file dirname $hw_dir]]
proc hgtxr_env_default {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default
}
set gcc_include [hgtxr_env_default HGTXR_GCC_INCLUDE "/usr/lib/gcc/x86_64-linux-gnu/13/include"]
set gcc_lib [hgtxr_env_default HGTXR_GCC_LIB "/usr/lib/gcc/x86_64-linux-gnu/13"]
set sys_include [hgtxr_env_default HGTXR_SYS_INCLUDE "/usr/include/x86_64-linux-gnu"]
set sys_lib [hgtxr_env_default HGTXR_SYS_LIB "/usr/lib/x86_64-linux-gnu"]

open_project -reset [file join $hw_dir generated hgtxr_s2_block_vector_hls]
set_top $top_name

set config_header [file normalize [file join $hw_dir configs zcu104_cyclic_s2_block_q4w8a_defines.h]]
set common_includes "-I[file normalize [file join $hw_dir hls include]] -I$sys_include -I$gcc_include"
set cxx_flags "-std=c++17 -ffunction-sections -fdata-sections $common_includes -DHGTXR_S2_BLOCK_VECTOR_NO_MAIN=1 -include $config_header"
set tb_flags "-std=c++17 $common_includes -include $config_header"

add_files -cflags $cxx_flags [file join $hw_dir hls tb tb_cyclic_s2_block_vector.cpp]
add_files -tb -cflags $tb_flags [file join $hw_dir hls tb tb_cyclic_s2_block_vector_hls_main.cpp]

open_solution -reset $solution_name
set_part $part_name
create_clock -period $clock_period -name default

set env(LIBRARY_PATH) "$gcc_lib:$sys_lib"
set env(LD_LIBRARY_PATH) "$gcc_lib:$sys_lib"
set env(PATH) "/usr/bin:$env(PATH)"

csim_design
exit
