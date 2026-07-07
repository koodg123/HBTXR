# ZCU104 HLS Experiment Matrix

Date: 2026-06-05
Toolchain: Vitis/Vivado 2023.2
Target: ZCU104
Source: GPT-5.3-Codex-Spark experiment-planning sidecar, integrated by main agent

## Parameter Knobs

| Symbol | Meaning | Initial Range |
|---|---|---|
| `TF` | Tiling factor | `1, 2, 4, 8` |
| `PF` | Parallelism factor / PE parallelism | `1, 2, 4` |
| `BUS_WIDTH_BITS` | AXI bus width | `64, 128, 256, 512` |
| `BIT_WIDTH` | activation/weight data width | `4, 8, 16` |
| `BUFFER_KB` | local buffer budget | `128, 256, 512, 1024` |
| `FIFO_DEPTH` | inter-stage stream depth | `16, 32, 64, 128` |

## Stage S0: Feasibility Gate

Baseline:

```text
TF=2
PF=2
BUS_WIDTH_BITS=256
BIT_WIDTH=8
BUFFER_KB=512
FIFO_DEPTH=32
```

Optional bandwidth sanity build:

```text
BUS_WIDTH_BITS=128
```

Stop gates:

- Vitis HLS `csynth` succeeds.
- HLS IP export succeeds.
- AXI master and AXI-Lite interfaces compile.
- Estimated Fmax is at least 180 MHz, or a timing-gap note is created.
- No critical warnings that imply invalid hardware.

Expected artifacts:

- `stage0_manifest.csv`
- `s0_csynth_report.json`
- `s0_interface_check.txt`

## Stage S1: One-Dimensional Screening

Sweep one parameter at a time around the baseline:

```text
TF              = 1, 2, 4, 8
PF              = 1, 2, 4
BUS_WIDTH_BITS  = 64, 128, 256, 512
BIT_WIDTH       = 4, 8, 16
BUFFER_KB       = 128, 256, 512, 1024
FIFO_DEPTH      = 16, 32, 64, 128
```

Expected run count: about 20 runs.

Stop gates:

- LUT <= 80%
- FF <= 80%
- BRAM <= 75%
- DSP <= 70%
- HLS estimated clock <= 5 ns, or timing-risk note created.
- Drop dominated points when throughput or II does not improve.

Expected artifacts:

- `s1_sweep_results.csv`
- per-run `csynth.xml`
- per-run `csynth.log`
- `stage1_filter.json`

## Stage S2: Coarse DOE Grid

Run fit-focused DOE after S1 filtering:

```text
TF              = 2, 4
PF              = 1, 2, 4
BUS_WIDTH_BITS  = 128, 256
BIT_WIDTH       = 4, 8
BUFFER_KB       = 256, 512
FIFO_DEPTH      = 32, 64
```

Expected run count: 96.

Additional gates:

- Key compute loops should reach `II <= 1`, or the reason must be recorded.
- Critical warning count is zero.
- If the top 20 Pareto points do not improve beyond S1, stop at S2.

Expected artifacts:

- `s2_pareto_candidates.csv`
- `s2_resource_timing.csv`
- `s2_timing_summary.rpt`
- `s2_area_power_snapshot.csv`

## Stage S3: PYNQ-Compatible Finalization

For the top three S2 Pareto anchors, run local refinement:

- `TF`: anchor minus one, anchor, anchor plus one, clipped to `1..8`
- `PF`: anchor minus one, anchor, anchor plus one, clipped to `1..8`
- `BUS_WIDTH_BITS`: anchor divided by two, anchor, anchor times two, clipped to `64..512`
- `BIT_WIDTH`: `4, 8` and one adjacent value if justified
- `BUFFER_KB`: anchor divided by two, anchor, anchor times two, clipped to `128..1024`
- `FIFO_DEPTH`: anchor divided by two, anchor, anchor times two, clipped to `16..128`

Keep this stage to at most 12 runs per anchor.

PYNQ gates:

- Generate `.bit` and `.hwh`.
- HWH register map matches the Python driver.
- PYNQ overlay loads.
- One-batch smoke transaction completes.
- Fixed input checksum is deterministic across repeated runs.

Expected artifacts:

- `s3_overlay_list.csv`
- `hgtxr.bit`
- `hgtxr.hwh`
- `pynq_smoke.log`
- `final_ranking.json`
- `final_matrix.md`

## Global Resource And Timing Policy

- Keep utilization below saturation to preserve routing margin:
  - LUT <= 80%
  - FF <= 80%
  - BRAM <= 75%
  - DSP <= 70%
- Target 200 MHz when possible.
- Accept 180 MHz only with an explicit timing-risk note.
- Final implementation should have non-negative WNS before claiming timing closure.
- PYNQ overlay compatibility is a hard gate for promoted candidates.

