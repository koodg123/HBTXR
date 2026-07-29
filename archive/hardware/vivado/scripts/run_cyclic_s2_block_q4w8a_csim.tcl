set ::argv {-enable_cyclic_transformer 1 -define_file hardware/configs/zcu104_cyclic_s2_block_q4w8a_defines.h -solution solution_cyclic_s2_block_q4w8a -reset_project 1 -csim_env 1}
source hardware/vivado/scripts/create_hls_project.tcl
csim_design
exit
