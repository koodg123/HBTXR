# Legacy ViT/XR/HG-PIPE Experiment Analysis

Date: 2026-06-12
Current workspace: `/home/kjm26/project/PRJXR/XR-VIT/HGTXR/hardware`

## Scope

This note consolidates legacy documents and experiment results from:

- `/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel`
- `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel`
- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs`

The purpose is not to mirror every old file. The purpose is to preserve the
results and design lessons that affect current HGTXR hardware decisions:

- DSP mapping instead of LUT-heavy arithmetic.
- URAM use for large buffers and deep FIFOs.
- LUTRAM use only for small local memories.
- Parallelism-factor scaling after the basic resource policy is stable.
- Clear evidence boundaries between completed reports and partial runs.

## Source Inventory

| Source | Observed role | Files inspected |
|---|---|---:|
| `XR-VIT/ViT_Accel` | Original multi-board DeiT/HG-PIPE flow and docs | 109 files within max depth 3 |
| `XR-VIT/XR_Accel` | ZCU104 cyclic/HBTXR extension and max-performance handoff | 307 files within max depth 4 |
| `Hardware/HG_PIPE_MERGE/ViT_Accel/docs` | Curated historical documentation set | 33 files within max depth 4 |

The HG_PIPE_MERGE docs largely preserve the ViT_Accel document set in a more
structured form. XR_Accel adds the ZCU104 cyclic/HBTXR architecture-freeze and
maxperf material.

## Key Legacy Evidence

### 1. Baseline Step2 HLS Resource Summary

Source:

- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/experiments/deit_tiny_baseline/full_deit_tiny_fit/step2_hls_resource_summary.md`
- Duplicate/rerun:
  `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/experiments/deit_tiny_baseline_ooc_rerun/full_deit_tiny_fit/step2_hls_resource_summary.md`

Observed result:

| Metric | Value |
|---|---:|
| HLS instances | 26 |
| COSIM pass | 26 / 26 |
| Worst timing | `PATCH_EMBED`, 3.397 ns, slack -0.897 ns |
| Max LUT | `ATTN0`, 78,460 |
| Max FF | `ATTN0`, 40,907 |
| Max DSP | `PATCH_EMBED`, 428 |
| Max BRAM18K | `ATTN0`, 123 |
| Max URAM | `PATCH_EMBED`, 18 |

Interpretation for HGTXR:

- The old full-DeiT baseline was functionally credible at HLS/cosim level.
- The baseline was LUT-heavy in attention and MLP. DSP use was concentrated in
  patch embedding rather than spread through dense projections.
- URAM use was also concentrated in patch embedding. Attention/MLP large
  buffers were not broadly moved to URAM in this baseline.
- This supports the current HGTXR policy of forcing arithmetic toward DSP and
  routing large token/intermediate buffers to URAM.

### 2. DSP Probe Evidence

Source:

- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/workspace/artifacts/reports/vck190/attn_mlp_dsp_probe/step2_hls_resource_summary_attn_mlp_dsp_probe.md`

Observed result:

| Case | LUT | FF | DSP | BRAM18K | Cycles | Note |
|---|---:|---:|---:|---:|---:|---|
| `ATTN0` | 65,856 | 38,346 | 748 | 123 | 131,768 | DSP-heavy probe |
| `MLP0` | 12,124 | 15,463 | 682 | 78 | 100,395 | DSP-heavy probe |

Evidence boundary:

- COSIM status was `UNKNOWN`, so this is not functional signoff evidence.
- It is still useful as a resource-direction probe.

Interpretation for HGTXR:

- DSP mapping can move ATTN/MLP away from LUT-dominant arithmetic.
- The probe also increases cycle count versus some smaller-fit experiments, so
  DSP forcing should be combined with parallelism and timing checks rather than
  treated as automatically faster.

### 3. Fit-C / Place-Fit Resource Tradeoff Evidence

Sources:

- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/workspace/artifacts/reports/vck190/fit_c_full_mlp/step2_hls_resource_summary.md`
- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/workspace/artifacts/reports/vck190/place_fit_a_csynth_only/step2_hls_resource_summary.md`

Observed examples:

| Experiment | Case | LUT | FF | DSP | BRAM18K | Cycles |
|---|---|---:|---:|---:|---:|---:|
| `fit_c_full_mlp` | `MLP0` | 10,860 | 8,055 | 10 | 77 | 200,742 |
| `place_fit_a_csynth_only` | `ATTN0` | 74,936 | 38,966 | 32 | 129 | 225,878 |
| `place_fit_a_csynth_only` | `MLP0` | 5,451 | 5,371 | 74 | 133 | 1,354,780 |

Interpretation for HGTXR:

- Pure fit profiles can reduce LUT sharply but may explode latency.
- The current HGTXR `PAR16` direction is justified: resource migration alone is
  not enough; parallelism must be increased to recover throughput.

### 4. Multi-Board PnR Status

Sources:

- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/experiments/pnr/vck190_deit_tiny_full_status_20260410.md`
- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/experiments/pnr/zu15eg_deit_tiny_full_status_20260410.md`
- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/experiments/pnr/zcu102_deit_tiny_full_status_20260410.md`

| Target | Flow result | LUT / logic issue | DSP | URAM | Timing/status |
|---|---|---:|---:|---:|---|
| VCK190 | Routed OOC implementation completed | 644,533 LUT, 567,791 logic LUT | 743 | 287 | WNS -0.973 ns at 2.5 ns |
| ZU15EG | Synth/opt completed, failed before placement | logic LUT 590,531 / 341,280 = 173.0% | 891 | 0 | Capacity blocker |
| ZCU102 | Synth/opt completed, failed before placement | logic LUT 567,104 / 274,080 = 206.9% | 1,254 | 0 | Capacity blocker |

Interpretation for HGTXR:

- The older full-DeiT design was fundamentally too LUT-heavy for ZynqMP-class
  targets without structural reduction.
- VCK190 routed, but timing was not clean at 400 MHz.
- ZU15EG and ZCU102 failures were dominated by LUT pressure. This is directly
  aligned with the current HGTXR requirement to avoid implementing all
  operators in LUTs.
- URAM was underused or absent in ZynqMP results, even when LUTRAM and BRAM were
  over capacity.

### 5. VCK190 Reuse-OOC Handoff

Source:

- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/status/HANDOFF_2026_04_20.md`

Observed result:

- Reuse-OOC black-box DCP injection was repaired.
- A nested clock-buffer placement issue was repaired by disabling OOC top clock
  buffer insertion for VCK190.
- After those fixes, board export advanced to a normal placement-capacity
  blocker.
- Post-opt VCK190 board-export utilization was:
  - CLB LUT: 729,655 / 899,840 = 81.09%
  - Register: 524,410 / 1,799,680 = 29.14%
  - BRAM tile: 651 / 967 = 67.32%
  - URAM: 287 / 463 = 61.99%
  - DSP: 743 / 1,968 = 37.75%

Interpretation for HGTXR:

- Even on VCK190, logic density was the dominant board-level placement limiter.
- DSP and URAM still had headroom relative to LUT pressure.
- This supports the HGTXR direction: use more DSP/URAM and reduce LUT logic
  density before trying to close board placement.

### 6. Parallelism and Memory-Binding Guide

Source:

- `/home/kjm26/project/PRJXR/Hardware/HG_PIPE_MERGE/ViT_Accel/docs/architecture/SRC_CASE_PARALLELISM_TUNING.md`
- `/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel/docs/SRC_CASE_PARALLELISM_TUNING.md`
- `/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel/docs/BOARD_RUNBOOKS.md`

Key legacy policy:

- MAC lane proxy is `TP * COP * CIP`.
- `COP` often reduces output accumulator, bias fanout, and adder pressure.
- `CIP` affects input window/LUTRAM pressure.
- `TP` has the largest latency impact and should be changed carefully.
- For `LUT as Memory`, first consider `CIP`, reshaper buffer storage style,
  QK/RV weight RAM style, and FIFO depth.
- For ZU15EG fit-D:
  - keep deadlock-sensitive ATTN QKV/A/FIFO paths stable,
  - move MLP and ATTN O-path multiplies to DSP,
  - move reshaper, residual, QQ-head buffers, and patch weights to URAM.

Interpretation for HGTXR:

- The old guidance was fit-first. Current HGTXR should adapt it in the opposite
  direction where possible: preserve the already passing PAR16/C3b throughput,
  keep DSP forced, keep large buffers in URAM, and only reduce parallelism when
  timing/resource reports force it.

### 7. ATTN Serialized-Head Experiments

Source:

- `/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel/docs/SRC_CASE_PARALLELISM_TUNING.md`
- `/home/kjm26/project/PRJXR/XR-VIT/ViT_Accel/docs/BOARD_RUNBOOKS.md`

Key reported variants:

| Variant | Resource / latency | Status |
|---|---|---|
| `HEAD_SERIAL_C3_MIX_FIFOLOW_QKVPACK_AQPACK` | 30,761 LUT / 92 BRAM / 10 URAM / 1,588,582 cycles | `COSIM_PASS`, fit baseline |
| `HEAD_SERIAL_C3_MIX_FIFOLOW_QKVPACK_AQBUF_O4X4` | 31,439 LUT / 96 BRAM / 10 URAM / 911,205 cycles | `COSIM_PASS`, low-DSP speed option |
| `HEAD_SERIAL_C3_MIX_FIFOLOW_QKVPACK_AQTILE_O4X4` | 30,593 LUT / 90 BRAM / 10 URAM / 46 DSP | `COSIM_FAIL (stalled)`, do not promote |
| `HEAD_SERIAL_C3_MIX_FIFOLOW_AURAM` | 28,047 LUT / 98 BRAM / 20 URAM / 1,569,752 cycles | `COSIM_PASS`, BRAM-to-URAM option |

Interpretation for HGTXR:

- Legacy ATTN work confirms that memory packing and serialization can reduce
  BRAM/LUT pressure, but unsafe tile replay can stall cosim.
- The current HGTXR C3b approach is more conservative: keep PAR16 and use memory
  banking/URAM without promoting cosim-stalled variants.
- If HGTXR later explores more aggressive ATTN-local restructuring, use
  `QKVPACK_AQPACK` as the safe pattern and treat `AQTILE_O4X4` as a recorded
  failure mode.

### 8. XR_Accel HBTXR Architecture Freeze

Sources:

- `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel/docs/architecture/HBTXR_ARCH_FREEZE.md`
- `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel/docs/architecture/HBTXR_CYCLIC_IMPLEMENTATION.md`
- `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel/docs/validation/DEIT_CYCLIC_VERIFICATION.md`

Key architecture to preserve:

- AXI-connected input/output DMA engines.
- PS-PL bus connection.
- Shared Global Buffer.
- On-chip NoC/interconnect.
- Patch Embedding Core.
- Four cyclic compute cores: `MHA0`, `MLP0`, `MHA1`, `MLP1`.
- Controller-visible mode/status behavior.
- Search mode uses frame input, full-depth traversal.
- Track mode uses event input, reduced-depth traversal to cut point `c`.

Validation anchor:

- DeiT cyclic path exact-match gate covers patch embedding, `attn0..attn11`,
  `mlp0..mlp11`, and head.
- Standalone fallback was reported passing for two repeated exact-match passes.
- HLS csynth was required for acceptance but was not the same as C-sim success.

Interpretation for HGTXR:

- Current HGTXR E2E should keep a clear separation between architecture
  skeleton, numeric model transition, and board evidence.
- C-sim exact-match is necessary but not enough for resource/performance claims.
- The C3b board-smoke path should stay the current practical acceptance path,
  with missing board file evidence treated as a blocker rather than inferred.

### 9. XR_Accel ZCU104 Cyclic Maxperf Handover

Sources:

- `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel/docs/status/ZCU104_CYCLIC_MAXPERF_HANDOVER.md`
- `/home/kjm26/project/PRJXR/XR-VIT/XR_Accel/docs/status/ZCU104_CYCLIC_MAXPERF_PROGRESS.md`

Implemented maxperf intent:

- Patch embed parallelism:
  - `DEIT_PE_CIP`: 16 -> 32
  - `DEIT_PE_COP`: 16 -> 32
- DeiT head throughput:
  - `DEIT_HEAD_TP`: 2 -> 4
  - `DEIT_HEAD_CIAP`: 1 -> 4
  - `DEIT_HEAD_COAP`: 1 -> 4
  - `DEIT_HEAD_CIP`: 1 -> 8
  - `DEIT_HEAD_COP`: 4 -> 8
- Head compute switched toward DSP.
- Token buffers reshaped/banked more aggressively.
- Functional standalone maxperf C-sim verification passed.

Evidence boundary:

- `CYCLIC_VIT_TOP` final HLS synthesis did not complete.
- No valid `top_csynth.rpt` was produced.
- Available files were only partial `csynth_design_size.rpt/xml`.
- HLS compile workload became very large and host memory pressure reached about
  43 GB RSS in one attempt.

Interpretation for HGTXR:

- Increasing parallelism is the right direction after DSP/URAM policies are in
  place, but top-level HLS compile size is a real risk.
- HGTXR should prefer bounded variants like current C3b with completed HLS and
  routed reports over an unbounded maxperf profile without `top_csynth.rpt`.

## Mapping To Current HGTXR State

Current HGTXR resource evidence already aligns with the useful parts of the
legacy lessons:

| Variant | PAR | Mem banks | Latency cycles | DSP | LUT | FF | BRAM18K | URAM | Routed WNS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A2 | 8 | 8 | 81,514,836 | 334 | 84,220 | 46,502 | 326 | 64 | 1.926 |
| A1 | 8 | 8 | 71,316,968 | 332 | 81,168 | 43,804 | 298 | 64 | 4.120 |
| C1 | 16 | 8 | 37,508,072 | 604 | 127,916 | 59,507 | 338 | 64 | 4.723 |
| C3b | 16 | 16 | 37,508,072 | 604 | 126,506 | 59,505 | 332 | 64 | 4.415 |

Current code policy anchors:

- `hls/include/hgtxr_e2e_vit.hpp`
  - `HGTXR_E2E_FORCE_DSP_MUL=1`
  - `HGTXR_E2E_FORCE_URAM_BUFFERS=1`
  - `HGTXR_E2E_SMALL_MEM_LUTRAM=1`
- `hls/include/hgtxr_cyclic_transformer_params.hpp`
  - `HGTXR_PAR_QK`, `HGTXR_PAR_ATTN`, and `HGTXR_PAR_FFN` follow
    `HGTXR_PE_PAR`.

Recommended current interpretation:

- Keep C3b as the selected hardware path because it gives PAR16 latency and DSP
  gain while slightly reducing LUT/BRAM versus C1.
- Treat A1/A2 as lower-risk fallbacks, not the performance target.
- Do not roll back DSP forcing or URAM buffers unless a report shows timing or
  routing failure.
- Small local memories should remain LUTRAM because that avoids wasting BRAM or
  URAM on small control/cache structures.

## Current Actionable Lessons

1. DSP policy:
   - Preserve forced DSP multiplication in E2E dense/QK/attention/FFN paths.
   - Legacy ATTN/MLP probes show DSP can materially change the resource mix.
   - Verify with completed `csynth.xml`, not resource-direction probes alone.

2. URAM policy:
   - Large token, global-buffer, normalization, Q/K/V, attention, and hidden
     buffers should remain URAM-backed where current HGTXR macros enable it.
   - Legacy board failures show LUTRAM/BRAM can be over capacity while URAM is
     still underused or unused.

3. LUTRAM policy:
   - Keep LUTRAM for small memory structures and local scratch/cache paths.
   - Avoid moving small control memories to URAM because it can waste scarce
     URAM ports/capacity.

4. Parallelism policy:
   - PAR16 is the current validated performance point.
   - C3b is preferred over C1 because it keeps PAR16 latency/DSP benefit and
     reduces LUT and BRAM with wider memory banking.
   - Further parallelism increases should be separate experiments with strict
     HLS compile-size monitoring.

5. Evidence policy:
   - C-sim pass is not a resource/performance claim.
   - Partial `csynth_design_size.*` is not a completed HLS result.
   - Board claims require routed timing/power/utilization or explicit smoke
     result artifacts.

## Recommended Follow-Up In HGTXR

| Priority | Work | Expected outcome |
|---|---|---|
| P0 | Keep C3b as active board-smoke candidate | Preserves completed HLS/routed evidence and best latency/DSP balance |
| P0 | Attach this legacy summary to future resource-policy audits | Prevents repeating LUT-heavy ViT_Accel mistakes |
| P1 | Add a small report extractor for HLS/Vivado resource deltas | Makes future A/C/E comparisons reproducible |
| P1 | Run board smoke when exact PYNQ artifact becomes available | Converts C3b from ready-for-board-smoke to board-proven |
| P2 | Explore PAR32 only after C3b smoke or explicit user selection | Potential lower latency, high HLS compile/timing risk |

## Provenance Notes

- All source paths above were inspected read-only.
- The current HGTXR directory is the only write target for this consolidation.
- Spark sub-agent execution was attempted but failed due usage limit; GPT5.5
  sub-agent execution was started for read-only cross-checking.
