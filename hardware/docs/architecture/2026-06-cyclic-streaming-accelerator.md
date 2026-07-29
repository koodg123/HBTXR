> **작성** 2026-06-16 · **갱신** 2026-07-29
> **상태** frozen
> **소유** hardware
>
> **cyclic 가속기 개요.** `archive/hardware/docs/` 에서 복사, 내용 무변경.

# HBTXR Cyclic Streaming Accelerator

This tree keeps the original HG-PIPE style flow, but adds an independent HBTXR
experiment path for the mode-asymmetric cyclic streaming architecture from
`../PAPER_WORKS/Submission/main.tex`.

For the current architecture-freeze milestone and the `DeiT-Tiny`
baseline-equivalence validation plan, see:

- `docs/architecture/HBTXR_ARCH_FREEZE.md`

## Entry Points

- Model config: `configs/models/hbtxr_option_a.json`
- Design config: `configs/designs/hbtxr_cyclic_zcu104.json`
- Experiment config: `configs/experiments/zcu104_hbtxr_option_a_cyclic.json`
- HLS case root: `workspace/hardware/case_hbtxr_zcu104`
- Shared HLS helper: `workspace/hardware/src/hbtxr_cyclic.h`
- Generated RTL wrapper: `automation/hbtxr_top.py`
- ZCU104 command: `workspace/flow/scripts/targets/zcu104/host.sh hbtxr`

## Implemented V1 Shape

- Search mode: frame input `1x128x128`, `T=64`, depth 8, 4 cyclic passes.
- Track mode: event input `2x64x64`, `T=16`, cut point 4, 2 cyclic passes.
- Shared token shape: `[64][192]` with on-chip ping-pong buffers.
- Core order per cycle: `MHA_EVEN -> MLP_EVEN -> MHA_ODD -> MLP_ODD`.
- Output packet: 8 words, 32-bit each.
- AXI-Lite registers:
  - `0x00 cfg_n`
  - `0x10 trigger`
  - `0x20 block_resetn`
  - `0x30 mode`, where `0=search`, `1=track`
  - `0x34 status`
  - `0x38 result_words`
  - `0x3c active_tokens`

## Bring-Up Notes

The current HLS arithmetic is a deterministic v1 scaffold. It validates the
streaming shape, mode routing, cyclic scheduling, token buffer lifetime, packet
length, and ZCU104 automation integration before the final quantized HBTXR
checkpoint is available.

To replace the scaffold with final numerics, export:

- `workspace/hardware/case_hbtxr_zcu104/refs/hbtxr_export.json`
- layer weights, biases, quant scales, LUTs, and per-block golden text files
- search and track terminal packet goldens

Then replace the placeholder transforms in `hbtxr_cyclic.h` with the exported
RMU/SMU, LayerNorm, Softmax, GeLU, MLP, and head parameters while keeping the
same case names and top-level interfaces.

## Quick Checks

```bash
python3 workspace/flow/entrypoints/python/hgpipe_flow.py generate-top \
  --experiment configs/experiments/zcu104_hbtxr_option_a_cyclic.json \
  --output /tmp/XR_HGPIPE_HBTXR_TOP.v

python3 workspace/flow/entrypoints/python/step0_case_generation.py \
  --experiment configs/experiments/zcu104_hbtxr_option_a_cyclic.json
```
