# Generate implemented utilization reports from an existing E2E AXIS/DMA checkpoint.

proc hgtxr_log {msg} { puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}] $msg" }
proc hgtxr_fail {msg} { puts stderr "ERROR: $msg"; exit 1 }

proc hgtxr_arg {name default} {
  set idx [lsearch -exact $::argv $name]
  if {$idx < 0} { return $default }
  set value_idx [expr {$idx + 1}]
  if {$value_idx >= [llength $::argv]} { hgtxr_fail "Missing value for $name" }
  return [lindex $::argv $value_idx]
}

set script_dir [file normalize [file dirname [info script]]]
set project_root [file normalize [file join $script_dir ".." ".." ".."]]
set profile "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
set overlay "hgtxr_e2e_axis_dma_${profile}_overlay"
set wrapper "hgtxr_e2e_axis_dma_system_wrapper"
set default_impl_dir [file join $project_root "hardware" "generated" "build" "vivado" $overlay "${overlay}.runs" "impl_1"]
set checkpoint [file normalize [hgtxr_arg "-checkpoint" [file join $default_impl_dir "${wrapper}_postroute_physopt.dcp"]]]
set out_dir [file normalize [hgtxr_arg "-out_dir" $default_impl_dir]]
set artifact_name [hgtxr_arg "-artifact_name" "hgtxr_e2e_axis_dma_${profile}"]

if {![file exists $checkpoint]} {
  hgtxr_fail "Checkpoint not found: $checkpoint"
}
file mkdir $out_dir

hgtxr_log "Opening checkpoint $checkpoint"
open_checkpoint $checkpoint

set util_report [file join $out_dir "${artifact_name}_utilization_implemented.rpt"]
set hier_report [file join $out_dir "${artifact_name}_utilization_hierarchical_implemented.rpt"]
set timing_report [file join $out_dir "${artifact_name}_timing_summary_implemented.rpt"]
set route_report [file join $out_dir "${artifact_name}_route_status_implemented.rpt"]
set power_report [file join $out_dir "${artifact_name}_power_implemented.rpt"]

report_utilization -file $util_report
report_utilization -hierarchical -file $hier_report
report_timing_summary -file $timing_report
report_route_status -file $route_report
report_power -file $power_report

hgtxr_log "Wrote implemented utilization report: $util_report"
hgtxr_log "Wrote implemented hierarchical utilization report: $hier_report"
hgtxr_log "Wrote timing/route/power reports under: $out_dir"
