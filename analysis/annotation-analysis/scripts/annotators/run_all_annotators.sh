#!/usr/bin/env bash
# Full 483-anchor accuracy eval for the ready annotator tools, then aggregate.
# CPU by default (does not disturb the GPU training). SAM3-I is GPU-only -> run
# separately when a >=20GB GPU is free. NOT auto-scheduled; launch manually.
cd /home/user/project/PRJXR-HBTXR/HBTXR/analysis/annotation-analysis || exit 1
GS=.venv-gsam2/bin/python
DV=.venv-deepvog/bin/python
LOG=results/annotators/run.log
mkdir -p results/annotators
export TF_CPP_MIN_LOG_LEVEL=3
DEV="${1:-cpu}"
echo "=== annotator full eval start $(date) (device=$DEV) ===" | tee "$LOG"

run() { echo -e "\n>>> $* ($(date +%H:%M:%S))" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

run $GS scripts/annotators/run_ritnet.py       --device "$DEV"
run $GS scripts/annotators/run_ellseg.py       --device "$DEV"
run $GS scripts/annotators/run_edge_guided.py  --device "$DEV"
run $DV scripts/annotators/run_deepvog.py      --device "$DEV"
run $GS scripts/annotators/run_yoloe.py        --device "$DEV" --prompt "black pupil" --conf 0.05

echo -e "\n=== aggregate ===" | tee -a "$LOG"
$GS scripts/15_eval_annotators.py 2>&1 | tee -a "$LOG"
echo "=== done $(date) ===" | tee -a "$LOG"
