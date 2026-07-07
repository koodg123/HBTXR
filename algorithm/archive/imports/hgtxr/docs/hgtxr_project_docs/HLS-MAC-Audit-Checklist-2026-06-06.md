# HGTXR Tiled MAC HLS Audit Checklist

Date: 2026-06-06
Source: GPT-5.3-Codex-Spark HLS review sidecar, integrated by main agent

This checklist applies to `hgtxr_cyclic_mac.hpp` and later cyclic Transformer compute kernels.

## Type Contract

- Every fixed-point typedef has explicit width and integer width.
- Activations, weights, and accumulators are signed when used in MAC paths.
- Arithmetic boundaries use explicit casts instead of relying on C++ promotion.
- Output quantization occurs in one named function.

## Arithmetic Width

- Product width is accounted for before accumulation.
- Accumulator has enough headroom for `ceil(log2(tile_k))` growth.
- Rounding and saturation policy are explicit.
- Worst-case tile sum is checked against accumulator range.

## Accumulator State

- Accumulators reset before a fresh output tile.
- Accumulate mode is explicit when partial sums continue across K tiles.
- No stale tile state crosses sequence/head/layer boundaries.

## Pipeline And Loop Structure

- Pipeline placement matches the actual memory dependency structure.
- Target II is verified in `csynth`, not assumed from pragmas.
- Loop tripcounts match parameter limits.
- Tail tile handling is explicit before claiming arbitrary sequence/model sizes.

## Array Partitioning

- Partition factors match the intended parallel read/write lanes.
- Large arrays are not fully partitioned unless the tile is small enough.
- Partition dimension matches the parallel access dimension.
- BRAM/LUTRAM/FF inference is checked in resource reports.

## ZCU104 Risk Gates

- DSP usage grows approximately with MAC lane count and bit width.
- Wide accumulators and aggressive partitioning are treated as timing risks.
- Adder-chain pressure is reviewed in timing reports.
- A point is not promoted unless LUT/FF/BRAM/DSP and WNS pass the sweep gate.

## Required Tests After Shell Access Is Restored

```bash
cd /home/user/project/PRJXR/HGTXR
python3 -m py_compile hardware/tools/hgtxr_sweep_expand.py hardware/tools/extract_resource_metrics.py hardware/tools/repair_hls_to_pynq.py
python3 hardware/tools/hgtxr_sweep_expand.py \
  --sweep-file hardware/configs/sweeps/zcu104_cyclic_transformer_sweep.yaml \
  --manifest /tmp/hgtxr_sweep_manifest.csv \
  --out-dir /tmp/hgtxr_sweep \
  --emit-format cpp \
  --max-combos 16 \
  --overwrite
```

Then compile a small host or HLS C simulation that instantiates:

- `hgtxr_cyclic_mac.hpp`
- `hgtxr_cyclic_scheduler.hpp`
- `hgtxr_cyclic_transformer_params.hpp`

