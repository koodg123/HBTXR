# HGTXR A1 E2E AXI-Stream + AXI DMA ZCU104 PYNQ-oriented Vivado BD flow.
# Separate from build_bitstream.tcl and build_e2e_m_axi_bitstream.tcl.

proc hgtxr_log {msg} { puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}] $msg" }
proc hgtxr_fail {msg} { puts stderr "ERROR: $msg"; exit 1 }
proc hgtxr_warn {msg} { puts stderr "WARN: $msg" }

proc hgtxr_arg {name default} {
  set idx [lsearch -exact $::argv $name]
  if {$idx < 0} { return $default }
  set value_idx [expr {$idx + 1}]
  if {$value_idx >= [llength $::argv]} { hgtxr_fail "Missing value for $name" }
  return [lindex $::argv $value_idx]
}

proc hgtxr_connect_clk_rst {clk rst cell} {
  foreach pin [get_bd_pins -quiet -of_objects [get_bd_cells $cell]] {
    set name [string tolower [get_property NAME $pin]]
    set type [get_property TYPE $pin]
    if {[llength [get_bd_nets -quiet -of_objects $pin]] > 0} { continue }
    if {$clk ne "" && $type eq "clk"} { catch {connect_bd_net $clk $pin} }
    if {$rst ne "" && [string match "*resetn*" $name]} { catch {connect_bd_net $rst $pin} }
    if {$rst ne "" && [string match "*aresetn*" $name]} { catch {connect_bd_net $rst $pin} }
  }
}

proc hgtxr_intf_pin {cell candidates} {
  foreach candidate $candidates {
    set pin [get_bd_intf_pins -quiet "$cell/$candidate"]
    if {$pin ne ""} { return $pin }
  }
  hgtxr_fail "No interface pin found on $cell from candidates: $candidates"
}

set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".." ".."]]
set part_name [hgtxr_arg "-part" "xczu7ev-ffvc1156-2-e"]
set board_name [hgtxr_arg "-board" "xilinx.com:zcu104:part0:1.1"]
set project_name [hgtxr_arg "-project_name" "hgtxr_e2e_axis_dma_overlay"]
set bd_name [hgtxr_arg "-bd_name" "hgtxr_e2e_axis_dma_system"]
set jobs [hgtxr_arg "-jobs" "4"]
set artifact_name [hgtxr_arg "-artifact_name" "hgtxr_e2e_axis_dma"]
set report_prefix $artifact_name
if {[string length "${report_prefix}_utilization_hierarchical_implemented.rpt"] > 200} {
  set report_prefix "hgtxr_e2e_axis_dma_impl"
}
set pl_clk_mhz [hgtxr_arg "-pl_clk_mhz" "100.0"]
set impl_strategy [hgtxr_arg "-impl_strategy" "default"]
set hls_ip_repo [file normalize [hgtxr_arg "-hls_ip_repo" [file join $project_root "hardware" "generated" "hgtxr_e2e_axis_hls" "solution_e2e_q4w8a" "impl" "ip"]]]
set output_root [file normalize [hgtxr_arg "-out_dir" [file join $project_root "hardware" "generated" "build" "vivado"]]]
set build_dir [file join $output_root $project_name]
set overlay_dir [file join $output_root "overlay" $project_name]
set pynq_dir [file join $project_root "hardware" "pynq" "hgtxr"]

if {![file exists [file join $hls_ip_repo "component.xml"]]} {
  hgtxr_fail "Packaged A1 E2E AXIS component.xml not found under $hls_ip_repo. Run package_e2e_axis_ip.tcl first."
}

file delete -force $build_dir
file mkdir $build_dir
file mkdir $overlay_dir
file mkdir $pynq_dir

hgtxr_log "Creating project $project_name"
create_project $project_name $build_dir -part $part_name -force
set_property board_part $board_name [current_project]
set_property target_language Verilog [current_project]
set_property ip_repo_paths [list $hls_ip_repo] [current_project]
update_ip_catalog

create_bd_design $bd_name
current_bd_design $bd_name

create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:* psu
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} [get_bd_cells psu]
catch {set_property -dict [list \
  CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ $pl_clk_mhz \
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {1} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__USE__S_AXI_GP1 {1} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
] [get_bd_cells psu]}

create_bd_cell -type ip -vlnv xilinx.com:hls:hgtxr_e2e_axis_top:1.0 hgtxr_e2e_axis_top_0
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_in
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:* axi_dma_out
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_include_mm2s {1} \
  CONFIG.c_include_s2mm {0} \
  CONFIG.c_m_axi_mm2s_data_width {256} \
  CONFIG.c_m_axis_mm2s_tdata_width {256} \
] [get_bd_cells axi_dma_in]
set_property -dict [list \
  CONFIG.c_include_sg {0} \
  CONFIG.c_include_mm2s {0} \
  CONFIG.c_include_s2mm {1} \
  CONFIG.c_m_axi_s2mm_data_width {256} \
  CONFIG.c_s_axis_s2mm_tdata_width {256} \
] [get_bd_cells axi_dma_out]

set ps_clk [get_bd_pins -quiet psu/pl_clk0]
set ps_rst [get_bd_pins -quiet psu/pl_resetn0]
if {$ps_clk eq ""} { hgtxr_warn "PS pl_clk0 not found; clock connection may be incomplete" }
if {$ps_rst eq ""} { hgtxr_warn "PS pl_resetn0 not found; reset connection may be incomplete" }
if {$ps_clk ne ""} {
  foreach p [list \
    psu/maxihpm0_fpd_aclk \
    psu/maxihpm1_fpd_aclk \
    psu/maxihpm0_lpd_aclk \
    psu/saxihpc0_fpd_aclk \
    psu/saxihpc1_fpd_aclk \
    psu/saxihp0_fpd_aclk \
    psu/saxihp1_fpd_aclk \
    psu/saxihp2_fpd_aclk \
    psu/saxihp3_fpd_aclk \
  ] {
    set pin [get_bd_pins -quiet $p]
    if {$pin ne ""} { catch {connect_bd_net $ps_clk $pin} }
  }
}

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_ctrl
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] [get_bd_cells axi_ctrl]
set ctrl_master [get_bd_intf_pins -quiet psu/M_AXI_HPM0_FPD]
if {$ctrl_master eq ""} { set ctrl_master [get_bd_intf_pins -quiet psu/M_AXI_HPM0_LPD] }
if {$ctrl_master eq ""} { hgtxr_fail "No PS HPM master interface found for AXI-Lite control" }
connect_bd_intf_net $ctrl_master [get_bd_intf_pins axi_ctrl/S00_AXI]
connect_bd_intf_net [hgtxr_intf_pin hgtxr_e2e_axis_top_0 {s_axi_control S_AXI_CONTROL}] [get_bd_intf_pins axi_ctrl/M00_AXI]
connect_bd_intf_net [hgtxr_intf_pin axi_dma_in {S_AXI_LITE s_axi_lite}] [get_bd_intf_pins axi_ctrl/M01_AXI]
connect_bd_intf_net [hgtxr_intf_pin axi_dma_out {S_AXI_LITE s_axi_lite}] [get_bd_intf_pins axi_ctrl/M02_AXI]
hgtxr_connect_clk_rst $ps_clk $ps_rst axi_ctrl

connect_bd_intf_net [hgtxr_intf_pin axi_dma_in {M_AXIS_MM2S m_axis_mm2s}] [hgtxr_intf_pin hgtxr_e2e_axis_top_0 {axis_in s_axis_axis_in S_AXIS_AXIS_IN}]
connect_bd_intf_net [hgtxr_intf_pin hgtxr_e2e_axis_top_0 {axis_out m_axis_axis_out M_AXIS_AXIS_OUT}] [hgtxr_intf_pin axi_dma_out {S_AXIS_S2MM s_axis_s2mm}]

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_mem
set hgtxr_master_candidates [list \
  hgtxr_e2e_axis_top_0/m_axi_gmem_e2e_weights \
  hgtxr_e2e_axis_top_0/m_axi_gmem_e2e_runtime \
  axi_dma_in/M_AXI_MM2S \
  axi_dma_out/M_AXI_S2MM \
]
set hgtxr_masters [list]
foreach intf $hgtxr_master_candidates {
  if {[get_bd_intf_pins -quiet $intf] ne ""} {
    lappend hgtxr_masters $intf
  } else {
    hgtxr_log "Skipping absent memory master interface $intf"
  }
}
set_property -dict [list CONFIG.NUM_SI [llength $hgtxr_masters] CONFIG.NUM_MI {1}] [get_bd_cells axi_mem]
set idx 0
foreach intf $hgtxr_masters {
  set si [format "axi_mem/S%02d_AXI" $idx]
  connect_bd_intf_net [get_bd_intf_pins $intf] [get_bd_intf_pins $si]
  incr idx
}

set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HPC0_FPD]
if {$ddr_slave eq ""} { set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HP0_FPD] }
if {$ddr_slave eq ""} { set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HPC1_FPD] }
if {$ddr_slave eq ""} { set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HP1_FPD] }
if {$ddr_slave eq ""} { hgtxr_fail "No PS HP/HPC slave interface found for A1 E2E AXIS/DMA DDR access" }
connect_bd_intf_net [get_bd_intf_pins axi_mem/M00_AXI] $ddr_slave
hgtxr_connect_clk_rst $ps_clk $ps_rst axi_mem

foreach cell [list hgtxr_e2e_axis_top_0 axi_dma_in axi_dma_out] {
  hgtxr_connect_clk_rst $ps_clk $ps_rst $cell
}
if {$ps_clk ne ""} {
  catch {connect_bd_net $ps_clk [get_bd_pins hgtxr_e2e_axis_top_0/ap_clk]}
}
if {$ps_rst ne ""} {
  catch {connect_bd_net $ps_rst [get_bd_pins hgtxr_e2e_axis_top_0/ap_rst_n]}
}

assign_bd_address
validate_bd_design
save_bd_design

set bd_file [get_files -quiet *${bd_name}.bd]
generate_target all $bd_file
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
set_property top [file rootname [file tail $wrapper]] [current_fileset]
update_compile_order -fileset sources_1

launch_runs synth_1 -jobs $jobs
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} { hgtxr_fail "synth_1 did not complete" }

if {$impl_strategy ne "" && $impl_strategy ne "default"} {
  hgtxr_log "Using implementation strategy $impl_strategy"
  set_property strategy $impl_strategy [get_runs impl_1]
}

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { hgtxr_fail "impl_1 did not complete" }

set impl_dir [file join $build_dir "${project_name}.runs" "impl_1"]
open_run impl_1
report_timing_summary -file [file join $impl_dir "${report_prefix}_timing_summary_implemented.rpt"]
report_utilization -file [file join $impl_dir "${report_prefix}_utilization_implemented.rpt"]
report_utilization -hierarchical -file [file join $impl_dir "${report_prefix}_utilization_hierarchical_implemented.rpt"]
report_route_status -file [file join $impl_dir "${report_prefix}_route_status_implemented.rpt"]
report_power -file [file join $impl_dir "${report_prefix}_power_implemented.rpt"]

set bit_files [glob -nocomplain -directory $impl_dir *.bit]
set hwh_files [glob -nocomplain -directory $impl_dir *.hwh]
if {[llength $bit_files] == 0} { hgtxr_fail "No .bit file found in $impl_dir" }
file copy -force [lindex $bit_files 0] [file join $overlay_dir "${artifact_name}.bit"]
file copy -force [lindex $bit_files 0] [file join $pynq_dir "${artifact_name}.bit"]
if {[llength $hwh_files] > 0} {
  file copy -force [lindex $hwh_files 0] [file join $overlay_dir "${artifact_name}.hwh"]
  file copy -force [lindex $hwh_files 0] [file join $pynq_dir "${artifact_name}.hwh"]
} else {
  hgtxr_warn "No .hwh found in $impl_dir. Copying BD handoff if available."
  set bd_hwh [concat \
    [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff"] *.hwh] \
    [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name] *.hwh] \
  ]
  if {[llength $bd_hwh] > 0} {
    file copy -force [lindex $bd_hwh 0] [file join $overlay_dir "${artifact_name}.hwh"]
    file copy -force [lindex $bd_hwh 0] [file join $pynq_dir "${artifact_name}.hwh"]
  }
}

hgtxr_log "A1 E2E AXIS/DMA overlay output: $overlay_dir"
