# C3b Board Measurement Attempt

Date: 2026-06-23

## Summary

| Item | Result |
|---|---|
| Target | ZCU104 PYNQ, C3b `c3b-mem16` AXIS/DMA smoke |
| Measurement requested | board latency, DMA bandwidth, p95/p99 latency, search/track invocation distribution |
| Measurement status | blocked by board host resolution |
| Attempted host | `xilinx@zcu104.local` |
| Failure | `ssh: Could not resolve hostname zcu104.local: Name or service not known` |
| Remote-run evidence | `generated/signoff/zcu104_c3b_smoke_remote_run_2026_06_10.{json,md}` |
| Measurement-capable bundle | `generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz` |
| Bundle sha256 | `generated/signoff/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256` |

## What Was Instrumented

| Metric | JSON field emitted by next board run | Notes |
|---|---|---|
| Measured board latency | `board_latency_ms`, `latency_ms_samples`, `latency_ms_summary` | `latency_ms_summary` includes mean, median, min, max, p95, p99 |
| DMA measured bandwidth | `dma_bandwidth_Bps_samples`, `dma_bandwidth_Bps_summary` | Input DMA wait window based on PYNQ DMA send channel wait |
| Aggregate DMA effective bandwidth | `aggregate_dma_effective_bandwidth_Bps_samples`, `aggregate_dma_effective_bandwidth_Bps_summary` | Includes input wait plus output wait; output wait also includes accelerator compute |
| Throughput | `throughput_fps_summary` | Derived from measured board latency samples |
| p95/p99 latency | `latency_ms_summary.p95`, `latency_ms_summary.p99` | Requires `--repeat N`, preferably `N >= 30` |
| Search/track invocation distribution | `search_track_invocation_distribution` | Runner-label distribution only; C3b hardware does not expose internal search/track counters |

## Exact Command To Measure When Board Is Reachable

```sh
python3 tools/run_zcu104_c3b_smoke_remote.py --profile c3b-mem16 --host <zcu104-ip-or-host> --user xilinx --execute
```

The bundle-local command currently defaults to one measured invocation:

```sh
PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant c3b-mem16 --weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_axis_dma_c3b_mem16_file_smoke.json
```

For percentile-quality timing on the board, run the module directly inside the extracted bundle:

```sh
PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant c3b-mem16 --weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --warmup 3 --repeat 30 --mode-label track --json-out e2e_axis_dma_c3b_mem16_file_smoke.json
```

## Current Blocking Evidence

| Step | Status | Evidence |
|---|---|---|
| Python/PYNQ local import on host | blocked | local Python sees repo `pynq/` namespace, not board PYNQ runtime |
| DNS for `zcu104.local` | failed | SSH returned hostname resolution error |
| SSH remote mkdir | failed | `generated/signoff/zcu104_c3b_smoke_remote_run_2026_06_10.json` command result returncode `255` |
| Board smoke result JSON | not produced | `pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json` still absent |

## Interpretation

The requested values were not physically measurable from this host because the ZCU104 target is not reachable by the configured hostname. The runner and bundle are now measurement-capable; once a reachable ZCU104 host/IP is provided, the same command will produce board latency, DMA bandwidth, p95/p99 latency, and runner-labeled invocation distribution fields in the smoke JSON.
