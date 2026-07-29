# VCK190 vs ZU15EG Parameter Comparison

This document records the current `full_deit_tiny_fit` comparison between the
`vck190` and `zu15eg` flows.

Snapshot date: 2026-04-15

Important scope:

- Both experiments keep the same `deit_tiny_baseline` model shape.
- The differences below are from the current `ViT_Accel` working tree.
- `ZU15EG` tuning is fit-oriented and intentionally avoids changing the
  deadlock-sensitive `ATTN QKV/R/A` paths.

## 1. Source Of Truth

Experiment configs:

- `configs/experiments/vck190_full_deit_tiny_fit.json`
- `configs/experiments/zu15eg_full_deit_tiny_fit.json`

Design configs:

- `configs/designs/full_deit_tiny_fit.json`
- `configs/designs/zu15eg_full_deit_tiny_fit.json`

Target configs:

- `configs/targets/vck190.json`
- `configs/targets/zu15eg.json`

Case roots:

- `workspace/hardware/case_vck190`
- `workspace/hardware/case_zu15eg`

## 2. High-Level Config Differences

| Item | VCK190 | ZU15EG |
|---|---:|---:|
| Experiment | `vck190_full_deit_tiny_fit` | `zu15eg_full_deit_tiny_fit` |
| Model | `deit_tiny_baseline` | `deit_tiny_baseline` |
| Design config | `full_deit_tiny_fit` | `zu15eg_full_deit_tiny_fit` |
| Family | `versal` | `zynqmp` |
| Part | `xcvc1902-vsva2197-2MP-e-S` | `xczu15eg-ffvb1156-2-i` |
| Board part | `xilinx.com:vck190:part0:3.2` | empty |
| Clock period | `2.5 ns` | `3.0 ns` |
| Shell mode | `ooc_accelerator` | `ooc_accelerator` |
| Design FIFO depth | `128` | `128` |
| Parallelism scale | `1.0` | `1.0` |

Interpretation:

- `VCK190` remains the fast-clock reference target.
- `ZU15EG` uses a dedicated design and target clock policy so HLS and board
  implementation both target `3.0 ns`.
- Both keep `allow_model_variant=false`, so this is a full-model fit attempt,
  not a reduced model variant.

## 3. Shape Parameters That Stay The Same

The full DeiT-Tiny shape is intentionally preserved:

| Parameter | Value |
|---|---:|
| Token count `T` | `196` |
| Main channel count `C` | `192` |
| Attention heads `H` | `3` |
| Hidden MLP channel count `CH` | `768` |
| Token parallelism `TP` | `2` |
| Main channel adapter `CAP` | `1` |
| Residual channel parallelism `RESI_CP` | `2` |

The resource and latency changes therefore come from HLS scheduling,
parallelism knobs, memory binding, clock policy, and board integration, not
from changing the model dimensions.

## 4. `PATCH_EMBED` Differences

| Parameter | VCK190 | ZU15EG |
|---|---:|---:|
| `TP` | `2` | `2` |
| `CIP` | `16` | `8` |
| `CIAP` | `2` | `1` |
| `COP` | `16` | `8` |
| `COAP` | `1` | `1` |
| `USE_DSP` | `true` | `true` |

Derived compute width:

- VCK190 MAC lane proxy: `TP * CIP * COP = 2 * 16 * 16 = 512`
- ZU15EG MAC lane proxy: `TP * CIP * COP = 2 * 8 * 8 = 128`
- ZU15EG is about `1/4` of the VCK190 `PATCH_EMBED` compute width.

Expected effect:

- Lower `LUT as Logic` and local buffer pressure on ZU15EG.
- Higher `PATCH_EMBED` latency because both input and output channel
  parallelism are reduced.

## 5. `ATTN` Differences

### Stable Paths

These values are intentionally kept the same:

| Parameter | VCK190 | ZU15EG |
|---|---:|---:|
| `MATMUL_QKV_CIP` | `6` | `6` |
| `MATMUL_QKV_COP` | `12` | `12` |
| `MATMUL_R_CIP` | `4` | `4` |
| `MATMUL_R_COP` | `7` | `7` |
| `MATMUL_A_CIP` | `7` | `7` |
| `MATMUL_A_COP` | `4` | `4` |
| `Q_USE_DSP` | `false` | `false` |
| `K_USE_DSP` | `false` | `false` |
| `V_USE_DSP` | `false` | `false` |
| `QK_MATMUL_USE_DSP` | `false` | `false` |
| `RV_MATMUL_USE_DSP` | `false` | `false` |

Rationale:

- `QKV`, `R`, and `A` feed the most backpressure-sensitive path through
  split, reshape, softmax, and merge.
- Previous ZU15EG deadlocks were concentrated around this part of the ATTN
  dataflow, so the current fit profile avoids further reductions here.

### O-Path Changes

| Parameter | VCK190 | ZU15EG |
|---|---:|---:|
| `MATMUL_O_CIP` | `12` | `4` |
| `MATMUL_O_COP` | `6` | `4` |
| `O_MATMUL_USE_DSP` | `false` | `true` |

Derived compute width:

- VCK190 O-path MAC lane proxy: `TP * O_CIP * O_COP = 2 * 12 * 6 = 144`
- ZU15EG O-path MAC lane proxy: `TP * O_CIP * O_COP = 2 * 4 * 4 = 32`
- ZU15EG O-path is about `22%` of the VCK190 O-path compute width.

Expected effect:

- Reduces LUT/register pressure in the output projection.
- Moves output projection multiplication toward DSP on ZU15EG.
- Adds latency on the O-path, but with lower deadlock risk than changing
  `QKV/R/A`.

## 6. `ATTN` FIFO Values

The important elasticity values are currently the same:

| FIFO | VCK190 | ZU15EG |
|---|---:|---:|
| `RESI_FIFO_DEPTH` | `4096 * 3` | `4096 * 3` |
| `QQ_HEAD_FIFO_DEPTH` | `8000` | `8000` |
| `KQ_RESHAPE_HEAD_FIFO_DEPTH` | `512` | `512` |
| `VQ_TRANSPOSE_HEAD_FIFO_DEPTH` | `512` | `512` |
| `R_HEAD_FIFO_DEPTH` | `512` | `512` |
| `RQ_HEAD_FIFO_DEPTH` | `512` | `512` |
| `KQ_HEAD_FIFO_DEPTH` | `64` | `64` |
| `VQ_HEAD_FIFO_DEPTH` | `64` | `64` |
| `A_HEAD_FIFO_DEPTH` | `64` | `64` |

Rationale:

- These FIFO depths are deliberately not reduced in the current ZU15EG fit
  profile.
- Reducing them can help resource fit, but it has a high risk of recreating
  `ATTN8` deadlock.

## 7. `MLP` Differences

| Parameter | VCK190 | ZU15EG |
|---|---:|---:|
| `TP` | `2` | `2` |
| `CHAP` | `2` | `1` |
| `M1_CIP` | `12` | `4` |
| `M1_COP` | `24` | `8` |
| `M1_USE_DSP` | `false` | `true` |
| `M2_CIP` | `24` | `8` |
| `M2_COP` | `12` | `4` |
| `M2_USE_DSP` | `false` | `true` |

Derived compute width:

- VCK190 FC1 proxy: `TP * M1_CIP * M1_COP = 2 * 12 * 24 = 576`
- ZU15EG FC1 proxy: `TP * M1_CIP * M1_COP = 2 * 4 * 8 = 64`
- VCK190 FC2 proxy: `TP * M2_CIP * M2_COP = 2 * 24 * 12 = 576`
- ZU15EG FC2 proxy: `TP * M2_CIP * M2_COP = 2 * 8 * 4 = 64`
- ZU15EG MLP matmul width is about `1/9` of VCK190 for both FC1 and FC2.

Expected effect:

- Large `LUT as Logic` reduction from lower `CIP/COP`.
- Large latency increase in MLP layers.
- DSP usage increases because ZU15EG moves MLP multiplication from LUT logic
  to DSP.

## 8. `HEAD` Differences

`HEAD.cpp` is currently identical between VCK190 and ZU15EG:

| Parameter | Value |
|---|---:|
| `TP` | `2` |
| `CIAP` | `1` |
| `CIP` | `1` |
| `COP` | `4` |
| `ADPT_FIFO_DEPTH` | `2` |
| `WIND_FIFO_DEPTH` | `2` |
| `WGHT_FIFO_DEPTH` | `2` |
| `MACS_FIFO_DEPTH` | `2` |
| `USE_DSP` | `false` |

Rationale:

- `HEAD` is not the first resource-fit target.
- It is small compared with `ATTN` and `MLP`.

## 9. Summary Of Intent

Current ZU15EG profile:

- Preserves full model shape.
- Uses a dedicated `3.0 ns` HLS and board clock target.
- Reduces `PATCH_EMBED` compute width to about `1/4` of VCK190.
- Reduces `MLP` matmul width to about `1/9` of VCK190.
- Reduces only the safer `ATTN O-path`.
- Keeps `ATTN QKV/R/A` and major FIFO depths close to VCK190 to avoid
  recreating deadlock.
- Moves selected ZU15EG `MLP` and `ATTN O-path` multiplication to DSP.
- Moves ZU15EG `ATTN` reshaper / residual / QQ-head buffering and
  `PATCH_EMBED` weights toward URAM for resource fit.

Practical consequence:

- ZU15EG is expected to be much slower than VCK190.
- Resource usage should move away from LUT logic, but DSP usage rises.
- BRAM/FIFO pressure is addressed with storage binding before reducing
  deadlock-sensitive `ATTN` parallelism or FIFO depths.
- A 3ns timing miss is treated as a separate timing-closure issue after
  resource-fit validation.

## 10. VCK190 Packaging And DDRMC Note

The original `HG-PIPE-main/docs/README.original.md` describes a mostly manual
Vivado packaging flow:

1. Generate Spinal whole-network Verilog.
2. Run `SPINAL/to_vivado.py` to create a Vivado source bundle.
3. Use Vivado GUI `Create and Package New IP`.
4. Add all files from the generated `vivado` folder.
5. Infer AXI-Lite and AXIS interfaces.
6. Create the memory map.
7. Re-package the IP.
8. Source `VCK190-bd-base.tcl` to create the base block design.
9. Add the packaged accelerator IP, reconnect DMA, assign addresses, and
   generate the PDI.

The current `ViT_Accel` `step4-board-export` / `step5` board-export path is an
automation of that manual packaging and BD integration flow. It therefore has
to handle generated VCK190 NoC/DDRMC artifacts that were previously hidden
behind the Vivado GUI workflow.

Recent VCK190 failure signature:

- `design_1_noc_ddr4_0.dcp` missing
- `design_1_noc_lpddr4_0.dcp` missing
- `*_ddrmc.elf` missing under generated NoC DDRMC PHY IP directories

Root cause observed in `MemGen.log`:

- Vivado's DDRMC IP generation invokes Xilinx MicroBlaze GCC to build internal
  DDRMC firmware ELF files.
- The HG-PIPE accelerator itself does not use MicroBlaze.
- The MicroBlaze GCC invocation belongs to the Xilinx DDRMC/NoC IP internals.
- Host HLS/CSIM include/library variables such as `C_INCLUDE_PATH`, `CPATH`,
  and `LIBRARY_PATH` can poison that cross-compile and cause missing
  `features.h` / ELF-generation failures.

Mitigation in `automation/vivado.py`:

- Vivado batch runs sanitize host toolchain environment variables by default.
- HLS/CSIM environment setup remains unchanged.
- To opt out for debugging, set:

```bash
export HGPIPE_VIVADO_PRESERVE_HOST_TOOLCHAIN_ENV=1
```

Recent OOC-DCP reuse failure signature:

- `full_board_export_post_impl.log` reported `open_run impl_1` failure.
- The actual `impl_1/runme.log` showed that `link_design` completed, then
  `reuse_ooc_dcp_pre_impl.tcl` failed before loading the accelerator DCP.
- The pre-impl hook could not locate the `HGPIPE_VIVADO_0` hierarchy cell.

Mitigation in `automation/flows/vck190_step5.py`:

- The generated pre-impl hook now searches exact hierarchy paths and fallback
  `NAME` / `REF_NAME` / `ORIG_REF_NAME` filters.
- The hook prints visible HGPIPE candidate cells if lookup still fails.
- This distinguishes hierarchy lookup failures from real place/route, timing,
  or resource failures.

## 11. Recommended Recheck Commands

VCK190 board export after the Vivado environment sanitize patch:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

HGPIPE_STEP4_BOARD_EXPORT_MODE=rebuild-ip \
HGPIPE_STEP5_MAX_THREADS=4 \
bash workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
```

Reuse OOC DCP mode:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

HGPIPE_STEP4_BOARD_EXPORT_MODE=reuse-ooc-dcp \
HGPIPE_STEP5_MAX_THREADS=4 \
bash workspace/flow/scripts/targets/vck190/readme.sh step4-board-export
```

ZU15EG fit profile recheck:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

HGPIPE_STEP1_CASES=ATTN8,MLP0,PATCH_EMBED \
HGPIPE_STEP1_MAX_THREADS=4 \
bash workspace/flow/scripts/targets/zu15eg/spinal.sh step1

bash workspace/flow/scripts/targets/zu15eg/spinal.sh step2
```

Host-side ZU15EG Vivado progression:

```bash
cd /home/kjm26/project/PRJXR/HG_PIPE_MERGE/ViT_Accel

bash workspace/flow/scripts/targets/zu15eg/host.sh step4-prepare
bash workspace/flow/scripts/targets/zu15eg/host.sh step4-synth
```
