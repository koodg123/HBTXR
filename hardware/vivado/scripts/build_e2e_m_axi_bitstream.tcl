# HGTXR A2 E2E m_axi ZCU104 PYNQ-oriented Vivado block-design flow.
# This script is separate from build_bitstream.tcl to preserve the legacy hgtxr_top flow.

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
    if {$clk ne "" && $type eq "clk"} { catch {connect_bd_net $clk $pin} }
    if {$rst ne "" && [string match "*resetn*" $name]} { catch {connect_bd_net $rst $pin} }
    if {$rst ne "" && [string match "*aresetn*" $name]} { catch {connect_bd_net $rst $pin} }
  }
}

set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".." ".."]]
set part_name [hgtxr_arg "-part" "xczu7ev-ffvc1156-2-e"]
set board_name [hgtxr_arg "-board" "xilinx.com:zcu104:part0:1.1"]
set project_name [hgtxr_arg "-project_name" "hgtxr_e2e_m_axi_overlay"]
set bd_name [hgtxr_arg "-bd_name" "hgtxr_e2e_m_axi_system"]
set jobs [hgtxr_arg "-jobs" "4"]
set hls_ip_repo [file normalize [hgtxr_arg "-hls_ip_repo" [file join $project_root "hardware" "generated" "hgtxr_e2e_m_axi_hls" "solution_e2e_q4w8a" "impl" "ip"]]]
set output_root [file normalize [hgtxr_arg "-out_dir" [file join $project_root "hardware" "generated" "build" "vivado"]]]
set build_dir [file join $output_root $project_name]
set overlay_dir [file join $output_root "overlay" $project_name]

if {![file exists [file join $hls_ip_repo "component.xml"]]} {
  hgtxr_fail "Packaged A2 E2E m_axi component.xml not found under $hls_ip_repo. Run package_e2e_m_axi_ip.tcl first."
}

file delete -force $build_dir
file mkdir $build_dir
file mkdir $overlay_dir

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
  CONFIG.PSU__USE__M_AXI_GP0 {1} \
  CONFIG.PSU__USE__M_AXI_GP1 {1} \
  CONFIG.PSU__USE__S_AXI_GP0 {1} \
  CONFIG.PSU__USE__S_AXI_GP1 {1} \
  CONFIG.PSU__USE__S_AXI_GP2 {1} \
  CONFIG.PSU__USE__S_AXI_GP3 {1} \
] [get_bd_cells psu]}

create_bd_cell -type ip -vlnv xilinx.com:hls:hgtxr_e2e_m_axi_top:1.0 hgtxr_e2e_m_axi_top_0

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
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {1}] [get_bd_cells axi_ctrl]
set ctrl_master [get_bd_intf_pins -quiet psu/M_AXI_HPM0_FPD]
if {$ctrl_master eq ""} { set ctrl_master [get_bd_intf_pins -quiet psu/M_AXI_HPM0_LPD] }
if {$ctrl_master eq ""} { hgtxr_fail "No PS HPM master interface found for AXI-Lite control" }
connect_bd_intf_net $ctrl_master [get_bd_intf_pins axi_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins hgtxr_e2e_m_axi_top_0/s_axi_control] [get_bd_intf_pins axi_ctrl/M00_AXI]
hgtxr_connect_clk_rst $ps_clk $ps_rst axi_ctrl

create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:* axi_mem
set hgtxr_masters [list \
  m_axi_gmem_frame \
  m_axi_gmem_e2e_weights \
  m_axi_gmem_out_state \
  m_axi_gmem_runtime_state \
]
set_property -dict [list CONFIG.NUM_SI [llength $hgtxr_masters] CONFIG.NUM_MI {1}] [get_bd_cells axi_mem]
set idx 0
foreach intf $hgtxr_masters {
  set si [format "axi_mem/S%02d_AXI" $idx]
  connect_bd_intf_net [get_bd_intf_pins hgtxr_e2e_m_axi_top_0/$intf] [get_bd_intf_pins $si]
  incr idx
}

set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HPC0_FPD]
if {$ddr_slave eq ""} { set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HP0_FPD] }
if {$ddr_slave eq ""} { set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HPC1_FPD] }
if {$ddr_slave eq ""} { set ddr_slave [get_bd_intf_pins -quiet psu/S_AXI_HP1_FPD] }
if {$ddr_slave eq ""} { hgtxr_fail "No PS HP/HPC slave interface found for E2E m_axi DDR access" }
connect_bd_intf_net [get_bd_intf_pins axi_mem/M00_AXI] $ddr_slave
hgtxr_connect_clk_rst $ps_clk $ps_rst axi_mem

if {$ps_clk ne ""} { connect_bd_net $ps_clk [get_bd_pins hgtxr_e2e_m_axi_top_0/ap_clk] }
if {$ps_rst ne ""} { connect_bd_net $ps_rst [get_bd_pins hgtxr_e2e_m_axi_top_0/ap_rst_n] }

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

launch_runs impl_1 -to_step write_bitstream -jobs $jobs
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} { hgtxr_fail "impl_1 did not complete" }

set impl_dir [file join $build_dir "${project_name}.runs" "impl_1"]
set bit_files [glob -nocomplain -directory $impl_dir *.bit]
set hwh_files [glob -nocomplain -directory $impl_dir *.hwh]
if {[llength $bit_files] == 0} { hgtxr_fail "No .bit file found in $impl_dir" }
file copy -force [lindex $bit_files 0] [file join $overlay_dir "hgtxr_e2e_m_axi.bit"]
if {[llength $hwh_files] > 0} {
  file copy -force [lindex $hwh_files 0] [file join $overlay_dir "hgtxr_e2e_m_axi.hwh"]
} else {
  hgtxr_warn "No .hwh found in $impl_dir. Copying BD handoff if available."
  set bd_hwh [concat \
    [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff"] *.hwh] \
    [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name] *.hwh] \
  ]
  if {[llength $bd_hwh] > 0} { file copy -force [lindex $bd_hwh 0] [file join $overlay_dir "hgtxr_e2e_m_axi.hwh"] }
}

hgtxr_log "A2 E2E m_axi overlay output: $overlay_dir"
