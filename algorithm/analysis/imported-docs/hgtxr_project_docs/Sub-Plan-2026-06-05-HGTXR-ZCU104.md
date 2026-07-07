# HGTXR Sub-Agent Plan And Task Cards

Date: 2026-06-05

## Sub-Agent Usage

The user requested maximum practical use of GPT-5.3-Codex-Spark. Spark sidecars were used for planning tasks that do not require current filesystem access:

- Architecture sidecar: cyclic Transformer microarchitecture.
- Experiment sidecar: ZCU104 HLS sweep matrix.
- QA sidecar: HLS-to-PYNQ evidence gates.

Filesystem-dependent tasks remain assigned to the main agent until WSL shell access is restored.

## Task Cards

```yaml
task_card:
  task_id: T-001
  sub_agent: "gpt5.3-codex-spark"
  role: "expert"
  objective: "Define cyclic Transformer Block microarchitecture for ZCU104 HLS."
  file_ownership: []
  assigned_skill: ["algo2fpga", "algorithm-hardware-codesign-expert"]
  inputs: ["HGTXR objective", "AXI/PYNQ requirement", "ZCU104 target"]
  outputs: ["docs/Architecture-2026-06-05-Cyclic-Transformer.md"]
  validation: ["Implementable HLS module boundaries", "Parameter surface defined"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-002
  sub_agent: "gpt5.3-codex-spark"
  role: "analyst"
  objective: "Define ZCU104 HLS experiment matrix with tiling, parallelism, bus width, bit width, buffer size, and FIFO depth."
  file_ownership: []
  assigned_skill: ["algo2fpga"]
  inputs: ["HGTXR objective", "ZCU104 target"]
  outputs: ["docs/Experiment-Matrix-2026-06-05-ZCU104.md"]
  validation: ["Resource gates", "Timing gates", "PYNQ gates"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-003
  sub_agent: "gpt5.3-codex-spark"
  role: "evaluator"
  objective: "Define evidence gates for HLS, IP export, Vivado implementation, PYNQ runtime, and paper reproduction."
  file_ownership: []
  assigned_skill: ["algo2fpga"]
  inputs: ["HGTXR objective", "paper reproduction scope"]
  outputs: ["docs/QA-Gates-2026-06-05-HLS-to-PYNQ.md"]
  validation: ["Completion evidence map", "Partial/unverified claim rules"]
  dependencies: []
```

```yaml
task_card:
  task_id: T-004
  sub_agent: "main-agent"
  role: "implementer"
  objective: "Patch Vivado HWH copy flow and copy generated bit/hwh into overlay/PYNQ package."
  file_ownership:
    - "hardware/vivado/scripts/build_bitstream.tcl"
    - "hardware/generated/build/vivado/overlay/hgtxr_overlay"
    - "hardware/pynq/hgtxr"
  assigned_skill: ["algo2fpga"]
  inputs: ["Vivado generated bit/hwh paths"]
  outputs: ["hgtxr.bit", "hgtxr.hwh"]
  validation: ["ls artifact existence", "PYNQ helper unit smoke"]
  dependencies: ["WSL shell access"]
```

```yaml
task_card:
  task_id: T-005
  sub_agent: "main-agent or worker after WSL access"
  role: "implementer"
  objective: "Add parameterized cyclic Transformer HLS modules."
  file_ownership:
    - "hardware/hls/include"
    - "hardware/hls/src"
    - "hardware/hls/tb"
  assigned_skill: ["algo2fpga"]
  inputs: ["Architecture spec", "Experiment matrix"]
  outputs: ["parameter header", "mac_tile", "cyclic scheduler", "tests"]
  validation: ["host smoke", "csim", "csynth"]
  dependencies: ["T-004", "resource reference extraction"]
```

## Conflict Policy

Workers must not revert unrelated edits. HLS implementation workers should have disjoint file ownership once WSL access is restored.

