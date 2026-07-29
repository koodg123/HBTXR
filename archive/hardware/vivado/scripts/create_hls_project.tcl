# HGTXR Vitis HLS project creation script.

proc hgtxr_hls_arg {name default} {
  set idx [lsearch -exact $::argv $name]
  if {$idx < 0} { return $default }
  set value_idx [expr {$idx + 1}]
  if {$value_idx >= [llength $::argv]} { error "Missing value for $name" }
  return [lindex $::argv $value_idx]
}

proc hgtxr_env_default {name default} {
  if {[info exists ::env($name)] && $::env($name) ne ""} {
    return $::env($name)
  }
  return $default
}

set enable_cyclic_transformer [hgtxr_hls_arg "-enable_cyclic_transformer" "0"]
set define_file [hgtxr_hls_arg "-define_file" ""]
set solution_name [hgtxr_hls_arg "-solution" "solution"]
set reset_project [hgtxr_hls_arg "-reset_project" "0"]
set csim_env [hgtxr_hls_arg "-csim_env" "0"]
set xilinx_root [hgtxr_env_default HGTXR_XILINX_ROOT "/tools/Xilinx"]
set gcc_include [hgtxr_env_default HGTXR_GCC_INCLUDE "/usr/lib/gcc/x86_64-linux-gnu/13/include"]
set gcc_lib [hgtxr_env_default HGTXR_GCC_LIB "/usr/lib/gcc/x86_64-linux-gnu/13"]
set sys_include [hgtxr_env_default HGTXR_SYS_INCLUDE "/usr/include/x86_64-linux-gnu"]
set sys_lib [hgtxr_env_default HGTXR_SYS_LIB "/usr/lib/x86_64-linux-gnu"]
set cxx_flags "-std=c++17 -Ihardware/hls/include -I$sys_include -I$gcc_include"

if {$enable_cyclic_transformer ne "0"} {
  append cxx_flags " -DHGTXR_ENABLE_CYCLIC_TRANSFORMER_TOP=1"
}
if {$define_file ne ""} {
  set define_file_abs [file normalize $define_file]
  if {![file exists $define_file_abs]} { error "Define file not found: $define_file_abs" }
  append cxx_flags " -include $define_file_abs"
}

if {$csim_env ne "0"} {
  set ::env(LIBRARY_PATH) $sys_lib:$gcc_lib:/lib/x86_64-linux-gnu
  set ::env(LD_LIBRARY_PATH) $sys_lib:$gcc_lib:/lib/x86_64-linux-gnu
  set ::env(PATH) /usr/bin:/bin:$xilinx_root/Vitis_HLS/2023.2/bin:$xilinx_root/Vivado/2023.2/bin
}

if {$reset_project ne "0"} {
  open_project -reset hardware/generated/hgtxr_hls
} else {
  open_project hardware/generated/hgtxr_hls
}
set_top hgtxr_top
foreach src [list   hardware/hls/src/hgtxr_top.cpp   hardware/hls/src/frame_patch_embed.cpp   hardware/hls/src/event_patch_embed.cpp   hardware/hls/src/matmul.cpp   hardware/hls/src/attention.cpp   hardware/hls/src/mlp.cpp   hardware/hls/src/fusion.cpp   hardware/hls/src/search_head.cpp   hardware/hls/src/track_head.cpp   hardware/hls/src/runtime_fsm.cpp   hardware/hls/src/rmu_smu.cpp   hardware/hls/src/nonlinear.cpp   hardware/hls/src/weight_prefetcher.cpp   hardware/hls/src/controller.cpp   hardware/hls/src/noc.cpp   hardware/hls/src/global_buffer.cpp ] {
  add_files -cflags $cxx_flags $src
}
add_files -tb -cflags $cxx_flags hardware/hls/tb/tb_hgtxr_top.cpp
open_solution $solution_name
set_part xczu7ev-ffvc1156-2-e
create_clock -period 5.0
