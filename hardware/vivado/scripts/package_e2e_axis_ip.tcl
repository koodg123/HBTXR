# Package the selected A1 E2E AXI-Stream HLS top as Vivado IP.
#
# Default run:
#   HGTXR_E2E_SCALE=active196_b6_ff768 HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream \
#     /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/package_e2e_axis_ip.tcl
#
# Isolated C/PAR package example:
#   HGTXR_E2E_PROJECT_NAME=hgtxr_e2e_axis_par16_hls HGTXR_E2E_PAR=16 \
#     HGTXR_E2E_SCALE=active196_b6_ff768 HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream \
#     /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/package_e2e_axis_ip.tcl

set hw_dir "hardware"
if {![file exists [file join $hw_dir hls src hgtxr_e2e_axis_top.cpp]]} {
  set hw_dir "."
}

proc hgtxr_should_guard_hls_module {module_name mode} {
  if {$mode eq "all"} {
    return [regexp {^hgtxr_e2e_axis_top} $module_name]
  }
  if {$mode eq "top"} {
    return [expr {$module_name eq "hgtxr_e2e_axis_top"}]
  }
  if {$mode eq "compute" || $mode eq "compute_keep"} {
    return [regexp {^hgtxr_e2e_axis_top_(hgtxr_e2e_controller_run|hgtxr_e2e_attn_unit_[0-9]+_s|hgtxr_e2e_mlp_unit_[0-9]+_s|hgtxr_e2e_project_qkv|hgtxr_e2e_attention_core|hgtxr_e2e_output_projection|hgtxr_e2e_mlp_head|grp_hgtxr_conv_patch_embedding)$} $module_name]
  }
  if {$mode eq "1" || $mode eq "datapath"} {
    return [regexp {^hgtxr_e2e_axis_top_(hgtxr_e2e_|grp_hgtxr_e2e_|grp_hgtxr_conv_patch_embedding|grp_hgtxr_global_buffer_load|grp_hgtxr_axis_read_frame|grp_hgtxr_axis_write_state)} $module_name]
  }
  return 0
}

proc hgtxr_guard_hls_verilog_modules {ip_dir mode} {
  if {$mode ne "1" && $mode ne "all" && $mode ne "datapath" && $mode ne "top" && $mode ne "compute" && $mode ne "compute_keep"} {
    return
  }
  set verilog_dir [file join $ip_dir hdl verilog]
  if {![file isdirectory $verilog_dir]} {
    puts "WARNING: HGTXR_E2E_OOC_DONT_TOUCH=$mode but missing Verilog directory $verilog_dir"
    return
  }
  if {$mode eq "compute_keep"} {
    set guard_line {(* keep_hierarchy = "yes" *) // HGTXR physical preservation guard}
  } else {
    set guard_line {(* keep_hierarchy = "yes", dont_touch = "yes" *) // HGTXR physical preservation guard}
  }
  set patched 0
  foreach verilog_file [glob -nocomplain -directory $verilog_dir *.v] {
    set fd [open $verilog_file r]
    set text [read $fd]
    close $fd
    if {[string first "HGTXR physical preservation guard" $text] >= 0} {
      continue
    }
    set changed 0
    set out_lines [list]
    foreach line [split $text "\n"] {
      if {[regexp {^module[ \t]+([^ \t(]+)} $line -> module_name] &&
          [hgtxr_should_guard_hls_module $module_name $mode]} {
        lappend out_lines $guard_line
        set changed 1
      }
      lappend out_lines $line
    }
    if {$changed} {
      set fd [open $verilog_file w]
      puts -nonewline $fd [join $out_lines "\n"]
      close $fd
      incr patched
    }
  }
  puts "HGTXR_E2E_OOC_DONT_TOUCH=$mode patched $patched Verilog module files in $verilog_dir"
}

set axis_script [file join $hw_dir vivado scripts run_e2e_q4w8a_csynth.tcl]
set fd [open $axis_script r]
set script_text [read $fd]
close $fd

if {![info exists ::env(HGTXR_E2E_PROJECT_NAME)] || $::env(HGTXR_E2E_PROJECT_NAME) eq ""} {
  set ::env(HGTXR_E2E_PROJECT_NAME) hgtxr_e2e_axis_hls
}
regsub {csynth_design[ \t\r\n]+exit} $script_text {csynth_design
export_design -format ip_catalog
if {[info exists ::env(HGTXR_E2E_OOC_DONT_TOUCH)]} {
  if {$::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "1" || $::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "all" || $::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "datapath" || $::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "top" || $::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "compute" || $::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "compute_keep"} {
    set hgtxr_ip_dir [file join $hw_dir generated $::env(HGTXR_E2E_PROJECT_NAME) solution_e2e_q4w8a impl ip]
    hgtxr_guard_hls_verilog_modules $hgtxr_ip_dir $::env(HGTXR_E2E_OOC_DONT_TOUCH)
    set hgtxr_ooc_xdc [file join $hw_dir generated $::env(HGTXR_E2E_PROJECT_NAME) solution_e2e_q4w8a impl ip constraints hgtxr_e2e_axis_top_ooc.xdc]
    if {[file exists $hgtxr_ooc_xdc]} {
      set hgtxr_ooc_fd [open $hgtxr_ooc_xdc a]
      puts $hgtxr_ooc_fd ""
      puts $hgtxr_ooc_fd "# HGTXR full-learned physical observability guard."
      if {$::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "all"} {
        puts $hgtxr_ooc_fd "set hgtxr_keep_cells \[get_cells -hier -quiet *\]"
      } elseif {$::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "top"} {
        puts $hgtxr_ooc_fd "set hgtxr_keep_cells \[get_cells -hier -quiet -regexp {.*hgtxr_e2e_axis_top_0$|.*hgtxr_e2e_axis_top_0/inst$}\]"
      } elseif {$::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "compute" || $::env(HGTXR_E2E_OOC_DONT_TOUCH) eq "compute_keep"} {
        puts $hgtxr_ooc_fd "set hgtxr_keep_cells \[get_cells -hier -quiet -filter {REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_controller_run || REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_attn_unit_*_s || REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_mlp_unit_*_s || REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_project_qkv || REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_attention_core || REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_output_projection || REF_NAME =~ hgtxr_e2e_axis_top_hgtxr_e2e_mlp_head || REF_NAME =~ hgtxr_e2e_axis_top_grp_hgtxr_conv_patch_embedding}\]"
      } else {
        puts $hgtxr_ooc_fd "set hgtxr_keep_cells \[get_cells -hier -quiet -regexp {.*(grp_hgtxr_e2e_|hgtxr_e2e_|grp_hgtxr_conv_patch_embedding|grp_hgtxr_global_buffer_load|grp_hgtxr_axis_read_frame|grp_hgtxr_axis_write_state).*}\]"
      }
      puts $hgtxr_ooc_fd "if {\[llength \$hgtxr_keep_cells\] > 0} {"
      if {$::env(HGTXR_E2E_OOC_DONT_TOUCH) ne "compute_keep"} {
        puts $hgtxr_ooc_fd "  set_property DONT_TOUCH true \$hgtxr_keep_cells"
      }
      puts $hgtxr_ooc_fd "  set_property KEEP_HIERARCHY true \$hgtxr_keep_cells"
      puts $hgtxr_ooc_fd "}"
      close $hgtxr_ooc_fd
      puts "HGTXR_E2E_OOC_DONT_TOUCH=$::env(HGTXR_E2E_OOC_DONT_TOUCH) appended $hgtxr_ooc_xdc"
    } else {
      puts "WARNING: HGTXR_E2E_OOC_DONT_TOUCH=$::env(HGTXR_E2E_OOC_DONT_TOUCH) but missing $hgtxr_ooc_xdc"
    }
  }
}
exit} script_text

eval $script_text
