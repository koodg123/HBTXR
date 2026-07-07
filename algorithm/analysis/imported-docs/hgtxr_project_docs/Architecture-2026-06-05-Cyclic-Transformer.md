# Cyclic Transformer Block Architecture For HGTXR

Date: 2026-06-05
Target: ZCU104, Vitis/Vivado 2023.2, PYNQ overlay
Source: GPT-5.3-Codex-Spark architecture sidecar, integrated by main agent

## Block-Level Pipeline

One target Transformer layer should map to:

```text
LN1 -> QKV -> Attention(QK^T, online softmax, AV) -> WO -> Add1
    -> LN2 -> MLP(Up projection, GELU, Down projection) -> Add2
```

The implementation should avoid fully unrolling the whole Transformer. It should reuse one or more parameterized MAC clusters across QKV, output projection, and MLP phases.

## Module Boundaries

| Module | Responsibility |
|---|---|
| `ctrl_sch` | Top FSM, phase transitions, runtime sizes, launch/done state |
| `dma_io` | AXI master burst read/write, activation/weight/output staging |
| `stream_ln` | Streaming LayerNorm or documented normalization substitute |
| `mac_tile` | Reusable tiled GEMM/MAC primitive for QKV, WO, W1, W2 |
| `qkv_pack_split` | Bias/scale/quantization and Q/K/V view split |
| `attn_block` | QK tile, online softmax state, and P*V tile |
| `mlp_pack` | MLP activation and projection glue |
| `residual_fma` | Residual add and optional scaling |
| `writeback` | Output packing and AXI writeback |

## Cyclic Scheduler

The top FSM should schedule repeated execution across sequence tiles, K/V tiles, heads, and channel blocks:

```text
IDLE
LOAD_X_TILE
LN1
QKV_COMPUTE
ATTN_QK_TILE
SOFTMAX_ACCUM
ATTN_AV_TILE
WO_PROJECT
RESID_ADD1
LN2
MLP1
ACT_GELU
MLP2
RESID_ADD2
STORE
DONE_OR_WAIT_NEXT
```

The outer cyclic loop is `seq_tile`. The attention inner loop is `kv_tile`. Dense projections reuse the same MAC engine when ZCU104 fit is preferred over maximum throughput.

## Buffering Strategy

- Double-buffer major activation stages:
  - `X_ping/X_pong`
  - `QKV_ping/QKV_pong`
  - `AttnOut_ping/AttnOut_pong`
  - `MLP_mid_ping/MLP_mid_pong`
- Avoid storing the full attention matrix.
- Store a Q tile, K/V tile, online softmax state, and partial context accumulator.
- Online softmax state per row:
  - `row_max`
  - `row_sum`
  - `exp_scale`
  - partial context accumulator

## Parameter Surface

Compile-time parameters:

- `HGTXR_TILE_TOKENS`
- `HGTXR_TILE_CHANNELS`
- `HGTXR_TILE_K`
- `HGTXR_HEAD_PAR`
- `HGTXR_PE_PAR`
- `HGTXR_UNROLL_K`
- `HGTXR_PIPELINE_STAGES`
- `HGTXR_BUS_WIDTH_BITS`
- `HGTXR_DATA_W`
- `HGTXR_DATA_I`
- `HGTXR_ACC_W`
- `HGTXR_ACC_I`
- `HGTXR_LOCAL_BUFFER_DEPTH`
- `HGTXR_STREAM_FIFO_DEPTH`
- `HGTXR_SOFTMAX_APPROX`

Runtime AXI-Lite parameters:

- `seq_len`
- `d_model`
- `n_heads`
- `head_dim`
- `d_ff`
- base pointers for activation, packed QKV weights, WO, W1, W2, output, and optional scratch

## HLS Pragmas

- Top-level AXI:
  - `m_axi` for tensor buffers
  - `s_axilite` for control and pointer registers
  - `ap_ctrl_hs` via `port=return`
- `DATAFLOW` across process-level stages after streams and local buffers are stable.
- `PIPELINE II=1` on MAC and streaming loops where resource pressure allows.
- `ARRAY_PARTITION` or `ARRAY_RESHAPE` on local tile fragments.
- `STREAM depth=HGTXR_STREAM_FIFO_DEPTH` between stage processes.
- `LOOP_TRIPCOUNT` on parameterized loops for stable reports.

## Fit-Oriented Defaults

Start with a conservative reuse-first build:

- `HGTXR_DATA_W=16`, `HGTXR_DATA_I=6`
- `HGTXR_ACC_W=32`, `HGTXR_ACC_I=12`
- `HGTXR_PE_PAR=2`
- `HGTXR_HEAD_PAR=1`
- `HGTXR_BUS_WIDTH_BITS=128` or `256`
- BRAM-backed local buffers before URAM assumptions
- Online softmax approximation disabled or simple LUT-backed until golden vectors exist

## Implementation Order

1. Add a parameter header and config struct.
2. Add `mac_tile` with a small deterministic test.
3. Add cyclic scheduler states without changing the public AXI ABI.
4. Add tiled QKV and WO using `mac_tile`.
5. Add attention QK/AV with online accumulation.
6. Add MLP stages.
7. Run csynth sweeps and promote only ZCU104-fit candidates.

