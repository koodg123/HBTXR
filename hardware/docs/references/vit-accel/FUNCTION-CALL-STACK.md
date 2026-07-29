# Function Call Stack

This document maps the main wrapper scripts to the Python entrypoints and internal flow functions they call.

## 1. VCK190 README Baseline Flow

Primary wrapper:

```text
workspace/flow/scripts/targets/vck190/readme.sh
```

Backend implementation:

```text
workspace/flow/scripts/run/vck190_readme_backend.sh
```

### `step0`

```text
workspace/flow/scripts/targets/vck190/readme.sh step0
  -> workspace/flow/entrypoints/python/step0_case_generation.py
  -> automation.flows.legacy_hls.run_step0_case_generation()
  -> generate_attn() / generate_mlp()
  -> workspace/hardware/case/*.cpp regeneration
```

### `step1`

```text
workspace/flow/scripts/targets/vck190/readme.sh step1
  -> workspace/flow/entrypoints/python/step1_hls_flow.py
  -> automation.flows.legacy_hls.main_step1()
  -> automation.flows.legacy_hls.run_step1_hls_flow()
  -> automation.legacy.pre_syn_process.create_subprojects()
  -> automation.legacy.pre_syn_process.create_tcls()
  -> automation.legacy.pre_syn_process.run_instances()
```

### `step2`

```text
workspace/flow/scripts/targets/vck190/readme.sh step2
  -> workspace/flow/entrypoints/python/step2_print_resource.py
  -> automation.flows.legacy_hls.main_step2()
  -> automation.flows.legacy_hls.run_step2_print_resource()
  -> automation.legacy.pst_syn_process.print_resource_table()
  -> automation.legacy.pst_syn_process.write_resource_reports()
```

### `step3-all`

```text
workspace/flow/scripts/targets/vck190/readme.sh step3-all
  -> run_spinal_server()
  -> workspace/flow/entrypoints/python/step3_spinal_flow.py
  -> automation.flows.spinal_runtime.main_step3()
  -> automation.flows.spinal_runtime.run_step3_spinal_flow()
  -> automation.legacy.pst_syn_process.to_spinal_all_blocks()
  -> automation.legacy.pst_syn_process.launch_all_spinal_sim()
  -> automation.legacy.pst_syn_process.get_latency()
```

### `step4-export`

```text
workspace/flow/scripts/targets/vck190/readme.sh step4-export
  -> sbt "test:runMain network.generate_whole_network_verilog"
  -> workspace/hardware/SPINAL/to_vivado.py
  -> workspace/hardware/SPINAL/vivado/* export bundle
```

### `step4-ooc`

```text
workspace/flow/scripts/targets/vck190/readme.sh step4-ooc
  -> workspace/flow/entrypoints/python/step4_board_impl_flow.py
  -> automation.cli.compat.run_hgpipe_prefixed_command(["board-impl"])
  -> automation.cli.hgpipe.main()
  -> automation.flows.reconstruction.cmd_board_impl()
  -> write impl/run_impl.tcl
  -> automation.vivado.run_vivado_batch()
  -> workspace/hardware/vivado/boards/vck190.tcl
  -> workspace/hardware/vivado/common/... Tcl stack
```

## 2. Config-Driven `hgpipe` Flow

Primary CLI:

```text
python3 workspace/flow/entrypoints/python/hgpipe_flow.py <command>
```

Actual stack:

```text
workspace/flow/entrypoints/python/hgpipe_flow.py
  -> automation.cli.hgpipe.main()
  -> argparse subcommand dispatch
  -> automation.flows.reconstruction.cmd_<subcommand>()
```

Representative subcommands:

- `baseline-report`
  - `cmd_baseline_report()`
- `generate-top`
  - `cmd_generate_top()`
- `prepare-sources`
  - `cmd_prepare_sources()`
- `vivado-sim`
  - `cmd_vivado_sim()`
- `vivado-package`
  - `cmd_vivado_package()`
- `board-impl`
  - `cmd_board_impl()`
- `board-bitstream`
  - `cmd_board_bitstream()`
- `spinal-board-bitstream`
  - `cmd_spinal_board_bitstream()`

## 3. ZynqMP Vivado-Only Flow

Top-level wrapper:

```text
workspace/flow/scripts/targets/zynqmp/vivado.sh
```

It immediately forwards to:

```text
workspace/flow/scripts/run/zynqmp_vivado_backend.sh
```

### `step4-prepare`

```text
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-prepare zcu102
  -> workspace/flow/scripts/run/zynqmp_vivado_backend.sh
  -> python3 -u workspace/flow/entrypoints/python/hgpipe_flow.py board-bitstream ...
  -> automation.flows.reconstruction.cmd_board_bitstream()
  -> write bitstream/run_bitstream.tcl
```

### `step4-validate` / `step4-synth` / `step4-bitstream`

```text
workspace/flow/scripts/targets/zynqmp/vivado.sh step4-bitstream zcu102
  -> workspace/flow/scripts/run/zynqmp_vivado_backend.sh
  -> hgpipe board-bitstream --run
  -> automation.flows.reconstruction.cmd_board_bitstream()
  -> automation.vivado.run_vivado_batch()
  -> workspace/hardware/vivado/common/hgpipe_zynqmp_bitstream.tcl
```

## 4. ZynqMP Spinal-Backed Flow

Top-level wrapper:

```text
workspace/flow/scripts/targets/zynqmp/spinal.sh
```

Backend:

```text
workspace/flow/scripts/run/zynqmp_spinal_backend.sh
```

### `step4-export`

```text
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-export
  -> sbt "test:runMain network.generate_whole_network_verilog"
  -> workspace/hardware/SPINAL/to_vivado.py
```

### `step4-prepare` / `step4-bitstream`

```text
workspace/flow/scripts/targets/zynqmp/spinal.sh step4-bitstream zcu102
  -> workspace/flow/scripts/targets/zynqmp/spinal.sh
  -> python3 -u workspace/flow/entrypoints/python/hgpipe_flow.py spinal-board-bitstream ...
  -> automation.flows.reconstruction.cmd_spinal_board_bitstream()
  -> _prepare_spinal_export_layout()
  -> write spinal_bitstream/run_spinal_bitstream.tcl
  -> automation.vivado.run_vivado_batch()
```

## 5. ZU15EG README-Style Host Flow

Primary wrapper:

```text
workspace/flow/scripts/targets/zu15eg/host.sh
```

Backend:

```text
workspace/flow/scripts/run/zynqmp_vivado_backend.sh
```

Default binding:

- `HGPIPE_TARGETS=zu15eg`
- `HGPIPE_HLS_TARGET=zu15eg`
- `HGPIPE_EXPERIMENT_SUFFIX=full_deit_tiny_fit`

Representative end-to-end path:

```text
workspace/flow/scripts/targets/zu15eg/host.sh all
  -> workspace/flow/scripts/run/zynqmp_vivado_backend.sh all
  -> step0 / step1 / step2
  -> run_target_bitstream_step(step4-bitstream, target=zu15eg)
  -> hgpipe_flow.py board-bitstream --experiment configs/experiments/zu15eg_full_deit_tiny_fit.json --run
```

Unsupported by design in this wrapper:

- `step3-all`
- `step4-export`

Use the generic Spinal-backed wrapper only when you explicitly want that path:

```text
workspace/flow/scripts/targets/zynqmp/spinal.sh ...
```

## 6. ZU15EG Docker Flow

Primary wrapper:

```text
workspace/flow/scripts/targets/zu15eg/docker.sh
```

Mode split:

```text
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal <command>
  -> workspace/flow/scripts/run/zynqmp_vivado_backend.sh
  -> Vivado-running commands are blocked

workspace/flow/scripts/targets/zu15eg/docker.sh spinal <command>
  -> workspace/flow/scripts/run/zynqmp_spinal_backend.sh
  -> Vivado-running commands are blocked
```

Representative Docker-safe paths:

```text
workspace/flow/scripts/targets/zu15eg/docker.sh nonspinal all
  -> step0 / step1 / step2 / step4-prepare

workspace/flow/scripts/targets/zu15eg/docker.sh spinal all
  -> baseline
  -> step0 / step1 / step2 / step3-all / step4-export
  -> step4-prepare
```

Blocked in Docker by policy:

- `step4-validate`
- `step4-synth`
- `step4-bitstream`
- `step4-ooc`

## 7. Docker Launch Path

Wrapper:

```text
workspace/flow/scripts/docker/container_run.sh
```

Backend:

```text
workspace/flow/scripts/docker/container_run.sh
```

Runtime stack:

```text
workspace/flow/scripts/docker/container_run.sh
  -> resolve host UID/GID/user/group
  -> validate Xilinx mount
  -> assemble docker run arguments
  -> launch image with mounted repository + mounted host Xilinx tree
  -> container entrypoint: workspace/flow/scripts/docker/container_entrypoint.sh
```

Attach helper:

```text
workspace/flow/scripts/docker/container_exec.sh
  -> workspace/flow/scripts/docker/container_exec.sh
```

## 6. Recovery Flow For Missing Spinal `.dat`

Wrapper:

```text
workspace/flow/scripts/ops/spinal_dat_recover.sh
```

Backend:

```text
workspace/flow/scripts/run/spinal_dat_recover_backend.sh
```

Call pattern:

```text
workspace/flow/scripts/ops/spinal_dat_recover.sh
  -> selective Step1 HLS replay
  -> HLS-generated .dat copied back into SPINAL export tree
  -> step4-export rerun
```

This flow exists because the exported Spinal bundle depends on generated `.dat` artifacts that are not fully preserved in the checked-in tree.
