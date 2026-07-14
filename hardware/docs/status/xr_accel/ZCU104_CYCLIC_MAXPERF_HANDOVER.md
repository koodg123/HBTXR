# ZCU104 DeiT Tiny Cyclic Maxperf Handover

Date: 2026-06-09
Repository: `/home/user/project/PRJXR/impl_repos/XR_Accel`

## 1. Purpose

This document hands over the current state of the ZCU104 cyclic DeiT-Tiny max-performance work so it can continue in another environment or thread.

Current objective:

- Make `zcu104_deit_tiny_baseline_cyclic` / `deit_tiny_cyclic_zcu104` use ZCU104 resources more aggressively.
- Preserve functional correctness of the cyclic DeiT testbench.
- Obtain HLS and board evidence before making final performance/resource claims.

Current status:

- Functional C simulation for the maxperf preset passes after the condition repair and synthesis-only pruning edit.
- CYCLIC_VIT_TOP HLS synthesis is not complete.
- No final latency/resource/timing claim is validated yet.
- The accidental source-state issue in cyclic_vit.h was repaired before commit: only the fixed-mode pruning block remains synthesis-only.

## 2. Name Mapping

These two names are not two separate accelerator implementations.

- `zcu104_deit_tiny_baseline_cyclic` is the experiment name.
  - Defined in `configs/experiments/zcu104_deit_tiny_baseline_cyclic.json`.
  - It binds model, target, design, and artifact path layout.
  - It points to:
    - model: `configs/models/deit_tiny_baseline_cyclic.json`
    - target: `configs/targets/zcu104.json`
    - design: `configs/designs/deit_tiny_cyclic_zcu104.json`

- `deit_tiny_cyclic_zcu104` is the design name.
  - Defined in `configs/designs/deit_tiny_cyclic_zcu104.json`.
  - It controls cyclic ViT design parameters such as case root, clock, benchmark units, FIFO depth, and HLS compile flags.

Artifact paths combine target/model/design, so both names appear in paths such as:

`workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/`

## 3. Real Sub-Agent Work Already Used

Two read-only exploration sub-agents were used before this handover:

```yaml
task_card:
  task_id: T-001
  sub_agent: "codex-native"
  role: "analyst"
  objective: "Identify cyclic DeiT HLS/source/config knobs for ZCU104 utilization"
  file_ownership: []
  assigned_skill: ["fpga-asic-design-expert", "dnn-accelerator-expert"]
  outputs:
    - "Backbone dominated by imported ATTN/MLP wrappers under workspace/hardware/case"
    - "Candidate knobs: patch/head parallelism, token buffer banking, DSP/LUT policy, FIFO/storage"
  validation: ["read-only source/config review"]
```

```yaml
task_card:
  task_id: T-002
  sub_agent: "codex-native"
  role: "evaluator"
  objective: "Identify validation gates for cyclic DeiT maxperf work"
  file_ownership: []
  assigned_skill: ["fpga-asic-design-expert"]
  outputs:
    - "Functional C-sim is necessary but insufficient"
    - "Completion requires CYCLIC_VIT_TOP csynth report and preferably ZCU104 board implementation report"
  validation: ["read-only artifact/config review"]
```

## 4. Files Changed

Tracked files currently modified:

- `automation/flows/legacy_hls.py`
  - Added design-level HLS cflag normalization/resolution.
  - Extended cyclic-case cflag propagation so target storage flags can apply to `CYCLIC_VIT_TOP`.

- `configs/designs/deit_tiny_cyclic_zcu104.json`
  - Added `hls_cflags` with `-DHGPIPE_ZCU104_CYCLIC_MAXPERF=1`.
  - Added max-resource/performance intent notes.

- `workspace/flow/scripts/utils/deit_cyclic_csim_only.tcl`
  - Added maxperf flag and patch-embed storage flag:
    - `-DHGPIPE_ZCU104_CYCLIC_MAXPERF=1`
    - `-DHGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE=0`

- `workspace/flow/scripts/utils/deit_cyclic_csynth_only.tcl`
  - Same flags as C-sim Tcl.

- `workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp`
  - Added synthesis-only fixed DeiT mode call:
    - In synthesis with maxperf enabled, call `run_cyclic_top(..., MODE_DEIT)`.
    - In non-synthesis, keep runtime modes so C-sim can still smoke-test search/track/deit paths.

- `workspace/hardware/src/cyclic_vit.h`
  - Added `HGPIPE_ZCU104_CYCLIC_MAXPERF` compile profile.
  - Intended maxperf changes:
    - Patch embed PE input/output parallelism increased.
    - DeiT head token/channel parallelism increased.
    - Head FIFOs and local stream depths increased.
    - Head compute switched toward DSP use.
    - Global token buffers more aggressively reshaped/banked.
    - Synthesis-only fixed-mode pruning added near `run_cyclic_top`.
  - Known issue: see Section 7.

Untracked/new status document:

- `docs/status/ZCU104_CYCLIC_MAXPERF_PROGRESS.md`

This handover document:

- `docs/status/ZCU104_CYCLIC_MAXPERF_HANDOVER.md`

## 5. Verification Completed

Passed:

```bash
cd /home/user/project/PRJXR/impl_repos/XR_Accel
python3 -m py_compile automation/flows/legacy_hls.py automation/config.py automation/cyclic_top.py
python3 -m json.tool configs/designs/deit_tiny_cyclic_zcu104.json
```

Cflag resolver smoke passed after the storage-flag fix. Expected effective cflags:

```text
PATCH_EMBED_IMAGE -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1 -DHGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE=0
DEIT_HEAD -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1 -DHGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE=0
CYCLIC_VIT_TOP -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1 -DHGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE=0
```

Standalone maxperf C-sim/g++ verification passed before the final fixed-mode pruning edit:

```bash
g++ -std=c++14 -Wno-unknown-pragmas -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1 \
  -I/tools/Xilinx/Vitis_HLS/2023.2/include \
  -I/tools/Xilinx/Vivado/2023.2/include \
  workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp \
  -o /tmp/xr_accel_deit_cyclic_maxperf_gpp
```

Verification artifact:

- `workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_maxperf_csim_verify.md`
- `workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_maxperf_csim_verify.json`

Reported checks were PASS:

- zero C-sim errors
- search/track/deit smoke modes
- patch/head exact match twice
- ATTN0..ATTN11 exact match twice
- MLP0..MLP11 exact match twice

Important caveat:

- Re-run this verification after fixing Section 7 because `cyclic_vit.h` was changed after that PASS.

## 6. HLS Synthesis Attempts

### Attempt 1

Command:

```bash
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
vitis_hls -f workspace/flow/scripts/utils/deit_cyclic_csynth_only.tcl
```

Result:

- Failed during source synthesis.
- Error cause: patch embed weights were bound to initialized URAM.
- Key error:

```text
Global/static variable 'cyclic_vit::DEIT_PATCH_EMBED_INST (.weight_arr)' with bind_storage implementation 'ram_1p_uram' cannot be initialized.
```

Fix applied:

- Added `-DHGPIPE_PATCH_EMBED_WEIGHT_STORAGE_STYLE=0` to C-sim/C-synth Tcl.
- Updated `legacy_hls.py` so cyclic cases can inherit target cflags.

### Attempt 2

Result:

- Got past the previous URAM init error.
- Reached heavy HLS performance/LTO phase.
- Host memory pressure became severe:
  - clang process around 43 GB RSS.
  - swap nearly exhausted.
- Run was manually interrupted to protect the machine.
- No `top_csynth.rpt` was produced.
- Only partial files were present under `/tmp/xr_accel_deit_cyclic_csynth/solution/syn/report`:
  - `csynth_design_size.rpt`
  - `csynth_design_size.xml`

Mitigation added after this:

- Synthesis-only fixed `MODE_DEIT` pruning in `CYCLIC_VIT_TOP.cpp`.
- Synthesis-only fixed `MODE_DEIT` pruning block in `run_cyclic_top`.

This mitigation has been revalidated at standalone C-sim level, but not at HLS csynth level.

## 7. Source-State Repair Completed

The accidental over-application of synthesis-only maxperf conditionals in cyclic_vit.h has been repaired before this commit.

Current expected condition policy:

- Maxperf profile/tuning definitions use: #if HGPIPE_ZCU104_CYCLIC_MAXPERF
- Only fixed-mode synthesis pruning uses: #if HGPIPE_ZCU104_CYCLIC_MAXPERF && defined(__SYNTHESIS__)

Verified locations after repair:

- workspace/hardware/src/cyclic_vit.h:119,126,191,198,205,211,217,223,229,235,241,247,326,524,703 use #if HGPIPE_ZCU104_CYCLIC_MAXPERF.
- workspace/hardware/src/cyclic_vit.h:711 uses #if HGPIPE_ZCU104_CYCLIC_MAXPERF && defined(__SYNTHESIS__).
- workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp:17 uses #if HGPIPE_ZCU104_CYCLIC_MAXPERF && defined(__SYNTHESIS__).

## 8. Immediate Next Steps

1. Re-run quick checks if continuing from a fresh checkout:

    cd /home/user/project/PRJXR/impl_repos/XR_Accel
    grep -n "HGPIPE_ZCU104_CYCLIC_MAXPERF" workspace/hardware/src/cyclic_vit.h
    python3 -m py_compile automation/flows/legacy_hls.py automation/config.py automation/cyclic_top.py
    python3 -m json.tool configs/designs/deit_tiny_cyclic_zcu104.json

2. Re-run standalone C-sim/g++ verifier after any further source edit:

    cd /home/user/project/PRJXR/impl_repos/XR_Accel
    g++ -std=c++14 -Wno-unknown-pragmas -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1 -I/tools/Xilinx/Vitis_HLS/2023.2/include -I/tools/Xilinx/Vivado/2023.2/include workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp -o /tmp/xr_accel_deit_cyclic_maxperf_gpp
    /tmp/xr_accel_deit_cyclic_maxperf_gpp > /tmp/xr_accel_deit_cyclic_maxperf_gpp.log
    echo "Standalone CSim done with 0 errors" >> /tmp/xr_accel_deit_cyclic_maxperf_gpp.log
    python3 workspace/flow/scripts/utils/verify_deit_cyclic_csim.py --log /tmp/xr_accel_deit_cyclic_maxperf_gpp.log --json workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_maxperf_csim_verify.json --markdown workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_maxperf_csim_verify.md

3. Re-run HLS synthesis:

    cd /home/user/project/PRJXR/impl_repos/XR_Accel
    source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
    vitis_hls -f workspace/flow/scripts/utils/deit_cyclic_csynth_only.tcl

4. Monitor memory during HLS. If clang again grows beyond roughly 40 GB RSS and stalls, do not treat that as a design success or failure. It is a host compile-size/resource issue. Consider a less aggressive profile:

- token buffer reshape factor 16 -> 8
- head TP 4 -> 2
- head CIAP/COAP 4 -> 2
- head CIP 8 -> 4
- keep fixed MODE_DEIT synthesis pruning

## 9. Completion Criteria

Do not mark this work complete until all of the following are true:

- Maxperf C-sim verification remains passing after any further edits.
- `CYCLIC_VIT_TOP` produces a valid `top_csynth.rpt`.
- Timing, latency, and resource numbers are extracted from the HLS report.
- ZCU104 resource utilization is compared against baseline.
- If possible, board synthesis/implementation report is generated.
- Any claim about performance or resource utilization cites the report path and date.

## 10. Uncertainty And Risks

- 확실하지 않음: the current maxperf profile may exceed ZCU104 timing/resource limits; no complete csynth report exists yet.
- 확실하지 않음: fixed-mode synthesis pruning may be enough to reduce host compile memory; it has not been validated yet.
- Known risk: aggressive banking/parallelism may improve throughput but cause timing closure, BRAM/LUT pressure, or HLS compile explosion.
- Known risk: preserving search/track C-sim paths while pruning synthesis to DeiT mode must be verified after every edit.

## 11. Useful Paths

- Experiment config:
  - `configs/experiments/zcu104_deit_tiny_baseline_cyclic.json`
- Design config:
  - `configs/designs/deit_tiny_cyclic_zcu104.json`
- Main cyclic source:
  - `workspace/hardware/src/cyclic_vit.h`
- Top wrapper:
  - `workspace/hardware/case_cyclic_zcu104/CYCLIC_VIT_TOP.cpp`
- C-sim Tcl:
  - `workspace/flow/scripts/utils/deit_cyclic_csim_only.tcl`
- C-synth Tcl:
  - `workspace/flow/scripts/utils/deit_cyclic_csynth_only.tcl`
- Functional verification report:
  - `workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_maxperf_csim_verify.md`
- Partial HLS temp report directory:
  - `/tmp/xr_accel_deit_cyclic_csynth/solution/syn/report`

