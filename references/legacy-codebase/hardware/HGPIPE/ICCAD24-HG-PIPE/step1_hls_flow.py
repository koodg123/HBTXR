"""
Step1: HLS flow
This step is to run HLS flow on all cases: PatchEmbed, Attention*12, MLP*12, Head
These modules are simulated(verified), synthesized, and implemented in parallel to reduce run time.
The compiled results are stored in the 'instances' directory.
"""
from pst_syn_process import *
from pre_syn_process import *

# run HLS flow on all cases: PatchEmbed, Attention*12, MLP*12, Head
case_names = [f"{t}{n}" for t in ("ATTN", "MLP") for n in range(12)] + ["HEAD"] + ["PATCH_EMBED"]

INSTANCE_DIR = os.path.join(ROOT_DIR, "instances")

create_subprojects(INSTANCE_DIR, case_names=case_names, overwrite=True)

# create the tcl files for each subproject
# create_tcls(INSTANCE_DIR, case_names=case_names, do_csim=True, do_csynth=True)
# create_tcls(INSTANCE_DIR, case_names=case_names, do_csim=True, do_csynth=True, do_cosim=True)
create_tcls(INSTANCE_DIR, case_names=case_names, do_csim=True, do_csynth=True, do_cosim=True, do_syn=True)
# create_tcls(INSTANCE_DIR, case_names=case_names, do_csim=True, do_csynth=True, do_cosim=True, do_impl=True, phys_opt="all")

# launch the tcl files
run_instances(INSTANCE_DIR, case_names=case_names, version="2023.2", max_threads=16)