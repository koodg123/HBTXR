from pre_syn_process import *
from pst_syn_process import *
import os

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances")

case_list = [
    "PATCH_EMBED",
    *[f"{layer_id % 2 == 0 and 'ATTN' or 'MLP'}{layer_id//2}" for layer_id in range(24)],
    "HEAD"
]

instances_list = ["proj_" + case_name for case_name in case_list]

backup_verilog(INSTANCE_DIR, instances_list=instances_list)
backup_log(INSTANCE_DIR, instances_list=instances_list)

print(case_list)

to_spinal_all_blocks(INSTANCE_DIR)
launch_all_spinal_sim()

for i, case_name in zip(range(-1, 25), case_list):
    print(f"Latency of {case_name:15} is {get_latency(i)}")