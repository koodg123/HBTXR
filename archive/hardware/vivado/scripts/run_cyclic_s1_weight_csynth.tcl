set ::argv {-enable_cyclic_transformer 1 -define_file hardware/configs/zcu104_cyclic_s1_weight_defines.h -solution solution_cyclic_s1_weight}
source hardware/vivado/scripts/create_hls_project.tcl
csynth_design
exit
