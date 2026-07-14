# ZCU104 Cyclic DeiT Max-Performance Progress

Date: 2026-06-09

## Goal

Tune zcu104_deit_tiny_baseline_cyclic / deit_tiny_cyclic_zcu104 toward maximum ZCU104 resource use and performance while preserving the DeiT-Tiny exact-match functional gate.

## Agent Plan

Role: FPGA HLS/Vivado flow lead / ViT accelerator dataflow architect / verification evaluator.

Task cards used:

- T-001: explorer/analyst for cyclic HLS knobs and imported ATTN/MLP evidence.
- T-002: explorer/evaluator for C-sim, C-synth, and board validation gates.
- T-003: main implementer for bounded max-performance profile and fast verification.

## Implemented Changes

- Added HGPIPE_ZCU104_CYCLIC_MAXPERF compile profile in workspace/hardware/src/cyclic_vit.h.
- Increased cyclic DeiT patch-embed parallelism under that profile:
  - DEIT_PE_CIP: 16 -> 32
  - DEIT_PE_COP: 16 -> 32
  - local patch streams: 64/2 -> 128/8
- Increased DeiT head throughput under that profile:
  - DEIT_HEAD_TP: 2 -> 4
  - DEIT_HEAD_CIAP: 1 -> 4
  - DEIT_HEAD_COAP: 1 -> 4
  - DEIT_HEAD_CIP: 1 -> 8
  - DEIT_HEAD_COP: 4 -> 8
  - head FIFOs: 2 -> 8
  - DEIT_HEAD_USE_DSP: false -> true
- Increased ping-pong token-buffer banking:
  - array_reshape channel factor: 4 -> 16 for both global token buffers.
- Added design-level HLS flag in configs/designs/deit_tiny_cyclic_zcu104.json:
  - -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1
- Extended automation/flows/legacy_hls.py so design-level hls_cflags apply to cyclic HLS cases.
- Updated deit_cyclic_csim_only.tcl and deit_cyclic_csynth_only.tcl to use the max-performance profile.

## Validation Completed

- Python compile: PASS
  - python3 -m py_compile automation/flows/legacy_hls.py automation/config.py automation/cyclic_top.py
- JSON validation: PASS
  - python3 -m json.tool configs/designs/deit_tiny_cyclic_zcu104.json
- HLS cflag resolver smoke: PASS
  - PATCH_EMBED_IMAGE: maxperf + ZynqMP storage flag
  - DEIT_HEAD: maxperf flag
  - CYCLIC_VIT_TOP: maxperf flag
- Standalone maxperf compile: PASS
  - g++ -std=c++14 -Wno-unknown-pragmas -DHGPIPE_ZCU104_CYCLIC_MAXPERF=1 ... CYCLIC_VIT_TOP.cpp
- Standalone maxperf functional verifier: PASS
  - workspace/artifacts/reports/zcu104/deit_tiny_baseline_cyclic/deit_tiny_cyclic_zcu104/deit_cyclic_maxperf_csim_verify.md

## Active / Remaining Gates

- Active gate:
  - vitis_hls -f workspace/flow/scripts/utils/deit_cyclic_csynth_only.tcl
  - expected report: /tmp/xr_accel_deit_cyclic_csynth/solution/syn/report/top_csynth.rpt
- 확실하지 않음: final resource use, latency, and maximum performance until full CYCLIC_VIT_TOP csynth completes.
- 확실하지 않음: board-level closure until ZCU104 Vivado post-synth/implementation reports are generated.

## Next Actions

1. Finish or inspect the active maxperf CYCLIC_VIT_TOP csynth run.
2. If csynth fails, classify whether the blocker is syntax/type, memory, port pressure, or resource explosion.
3. If csynth passes, compare maxperf latency/resource against the previous reports and launch a ZCU104 board-level build.
