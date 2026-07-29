> **작성** 2026-06-16 · **갱신** 2026-07-29
> **상태** frozen — 검증 결과
> **소유** hardware
>
> **다중 보드 검증.** `archive/hardware/docs/` 에서 복사, 내용 무변경.

# Multi-Board Validation

The merged `XR_Accel` tree is intended to keep per-target configs and board Tcl entrypoints for:

- `vck190`
- `zu15eg`
- `zcu102`
- `zcu104`
- `ultra96v2`

## Config Layers

Three config layers are expected in the merged flow:

- `configs/models/`
  - baseline DeiT-Tiny
  - board-constrained DeiT-Tiny variants
- `configs/targets/`
  - part / board-part / shell mode / runtime metadata
- `configs/designs/`
  - fit strategy, clock target, FIFO depth, benchmark units

Experiment configs under `configs/experiments/` tie those layers together.

## Profiles

Each target is expected to keep two profiles:

- `full_deit_tiny_fit`
  - keeps the baseline DeiT-Tiny shape
  - explores hardware-only fit strategies
- `deit_tiny_variant_fit`
  - allows smaller DeiT-Tiny variants for constrained devices

## Board-Specific Hardware Tuning

The full-model path keeps the baseline DeiT-Tiny model shape and instead lowers
hardware parallelism per board through the dedicated case roots:

- `workspace/hardware/case_vck190`
- `workspace/hardware/case_zu15eg`
- `workspace/hardware/case_zcu102`
- `workspace/hardware/case_zcu104`
- `workspace/hardware/case_ultra96`

`vck190` remains the baseline reference. The constrained boards reduce only
hardware knobs in `PATCH_EMBED`, `ATTN`, and `MLP`.

### Applied Knob Changes

| Board | PATCH_EMBED | ATTN | MLP |
| --- | --- | --- | --- |
| `vck190` | baseline: `CIP=16`, `CIAP=2`, `COP=16` | baseline: `QKV=6`, `A=7`, `O=12`, `RESI=12288`, `QQ=8000`, `KQ/VQ=512` | baseline: `CHAP=2`, `M1=12`, `M2=24` |
| `zu15eg` | `CIP 16->8`, `CIAP 2->1`, `COP 16->8` | `QKV 6->6`, `A 7->7`, `O_CIP 12->4`, `O_COP 6->4`, `O_USE_DSP false->true`, `RESI 12288->12288`, `QQ 8000->8000`, `KQ/VQ 512->512` | `CHAP 2->1`, `M1_CIP 12->4`, `M1_COP 24->8`, `M1_USE_DSP false->true`, `M2_CIP 24->8`, `M2_COP 12->4`, `M2_USE_DSP false->true` |
| `zcu102` | `CIP 16->8`, `CIAP 2->1`, `COP 16->4` | `QKV 6->4`, `A 7->4`, `O 12->6`, `RESI 12288->1024`, `QQ 8000->2048`, `KQ/VQ 512->256` | `CHAP 2->1`, `M1_CIP 12->8`, `M1_COP 24->12`, `M2_CIP 24->12`, `M2_COP 12->8` |
| `zcu104` | `CIP 16->4`, `CIAP 2->1`, `COP 16->4` | `QKV 6->3`, `A 7->3`, `O 12->4`, `RESI 12288->1024`, `QQ 8000->1024`, `KQ/VQ 512->128` | `CHAP 2->1`, `M1_CIP 12->6`, `M1_COP 24->12`, `M2_CIP 24->12`, `M2_COP 12->8` |
| `ultra96v2` | `CIP 16->4`, `CIAP 2->1`, `COP 16->4` | `QKV 6->3`, `A 7->3`, `O 12->4`, `RESI 12288->512`, `QQ 8000->1024`, `KQ/VQ 512->128`, `QK/RV weight LRAM->BRAM` | `CHAP 2->1`, `M1_CIP 12->4`, `M2_CIP 24->8` |

### Expected Impact

These numbers are engineering estimates from the reduced channel parallelism and
smaller FIFO/LUTRAM buffers. They are not measured post-synth results yet.

| Board | Expected LUT delta | Expected LUTRAM / FIFO delta | Expected BRAM delta | Expected end-to-end latency delta |
| --- | --- | --- | --- | --- |
| `vck190` | reference | reference | reference | reference |
| `zu15eg` | `-20% ~ -30%` | `-25% ~ -35%` | `+5% ~ +10%` | `+10% ~ +20%` |
| `zcu102` | `-35% ~ -45%` | `-40% ~ -55%` | `+10% ~ +20%` | `+20% ~ +35%` |
| `zcu104` | `-45% ~ -55%` | `-50% ~ -60%` | `+10% ~ +25%` | `+30% ~ +45%` |
| `ultra96v2` | `-55% ~ -65%` | `-60% ~ -70%` | `+15% ~ +30%` | `+45% ~ +70%` |

### Tuning Intent

- `zu15eg`
  - uses a ZU15EG-specific 5.0 ns HLS/board clock target
  - preserves the deadlock-sensitive attention `QKV/A/FIFO` path while cutting
    `MLP` and `ATTN O-path` more aggressively
- `zcu102`
  - adds a first `CHAP` reduction because the full baseline overflow was too
    large for FIFO-only tuning
- `zcu104`
  - moves to a clearly lower-throughput but still full-model configuration
    intended to reduce LUT pressure much more aggressively
- `ultra96v2`
  - uses the strongest cuts and should be treated as a feasibility-oriented
    full-model hardware profile

## Validation Outputs

The merged automation is intended to collect:

- prepared RTL bundles
- baseline report
- experiment matrix
- simulation assets
- packaging Tcl
- implementation Tcl
- optional ZynqMP bitstream layouts

## Resolved Issue: ZU15EG Step1 CoSim Deadlock

Earlier `zu15eg` tuning profiles could deadlock during `step1` HLS cosim.

### Observed Failure

- log: `workspace/artifacts/logs/zu15eg/zynqmp-spinal-step1.log`
- failing case: `proj_ATTN8`
- failure signature:
  - `// ERROR!!! DEADLOCK DETECTED at 488645000 ns! SIMULATION WILL BE STOPPED! //`

The deadlock appears in the generated HLS dataflow graph for `ATTN8`, not in
the later Vivado `step4` flow.

### Localized Dependence Cycle

The latest reported cycle is centered on the `head1` attention path:

- `top.do_attn_U0.do_quant_4_U0`
- `top.do_attn_U0.do_split_6_U0`
- `top.do_attn_U0.do_reshape_7_U0`
- `top.do_attn_U0.do_matmul_9_U0`
- `top.do_attn_U0.do_softmax_11_U0`
- `top.do_attn_U0.do_quant_U0`
- `top.do_attn_U0.do_split_U0`
- `top.do_attn_U0.do_reshape_13_U0`
- `top.do_attn_U0.do_matmul_15_U0`

This points to a balance problem between the `KQ/VQ reshape` side and the
`QK/RV` matmul side inside the attention kernel.

### Most Likely Cause

The `zu15eg` board-specific `ATTN` profile reduced both channel parallelism and
several FIFO depths:

- `MATMUL_QKV_CIP 6 -> 4`
- `MATMUL_A_CIP 7 -> 4`
- `MATMUL_O_CIP 12 -> 8`
- `RESI_FIFO_DEPTH 12288 -> 2048`
- `QQ_HEAD_FIFO_DEPTH 8000 -> 4096`
- `KQ_RESHAPE_HEAD_FIFO_DEPTH 512 -> 256`
- `VQ_TRANSPOSE_HEAD_FIFO_DEPTH 512 -> 256`

The FIFO reductions were the first deadlock suspect, and restoring them moved
the failure much later in simulation and away from the residual merge path.
Restoring `MATMUL_QKV_CIP 4 -> 6` alone was not enough; the deadlock remained
on the same `KQ/VQ reshape <-> QK/RV matmul` cycle and moved earlier to
`345315000 ns`. The next correction is to restore `MATMUL_A_CIP 4 -> 7` while
keeping the larger FIFOs and restored `MATMUL_QKV_CIP`.
log already warns that some FIFOs should be enlarged to improve performance or
avoid deadlocks.

### Recommended Immediate Mitigation

If progress is more important than cosim at the moment:

- rerun `step1` with `HGPIPE_STEP1_DO_COSIM=0`

If the goal is to keep cosim enabled:

- revert only the `zu15eg` attention FIFO depths first
- keep the reduced channel parallelism

Recommended first rollback set:

- `RESI_FIFO_DEPTH -> 4096 * 3`
- `QQ_HEAD_FIFO_DEPTH -> 8000`
- `KQ_RESHAPE_HEAD_FIFO_DEPTH -> 512`
- `VQ_TRANSPOSE_HEAD_FIFO_DEPTH -> 512`

### Applied Mitigation On 2026-04-14

The mitigation has been applied to:

- `workspace/hardware/case_zu15eg/ATTN.cpp.template`
- `workspace/hardware/case_zu15eg/ATTN0.cpp` .. `ATTN11.cpp`

Applied changes:

- restored `RESI_FIFO_DEPTH` from `2048` to `4096 * 3`
- restored `QQ_HEAD_FIFO_DEPTH` from `4096` to `8000`
- restored `KQ_RESHAPE_HEAD_FIFO_DEPTH` from `256` to `512`
- restored `VQ_TRANSPOSE_HEAD_FIFO_DEPTH` from `256` to `512`
- restored the deadlock-sensitive compute path:
  - `MATMUL_QKV_CIP = 6`
  - `MATMUL_A_CIP = 7`
- kept the resource-fit cuts on the safer output path:
  - `MATMUL_O_CIP = 4`
  - `MATMUL_O_COP = 4`
  - `O_MATMUL_USE_DSP = true`

Expected effect:

- lower deadlock risk in `step1` cosim by restoring elasticity and the
  `QKV/A` throughput balance in the attention dataflow cycle
- lower LUT logic pressure through `ATTN O-path` and `MLP` cuts
- higher latency than the VCK190 baseline, because the full model shape is kept
  while ZU15EG uses lower parallelism and a 5.0 ns clock target

## Important Constraint

The repository still does not have a general exporter for regenerating weights and golden outputs for arbitrary new model shapes.

Because of that:

- baseline DeiT-Tiny remains the functional oracle
- variant-model configs are best treated as synthesis / resource / cycle exploration first
- full functional validation for arbitrary new shapes remains a follow-on task

## Current Integrated Direction

The current documentation assumes:

- legacy VCK190 baseline remains the main correctness oracle
- Vivado-only automation remains the main structured reconstruction path
- ZynqMP Vivado-only and ZynqMP Spinal-backed flows both remain first-class documentation targets

## Progress Snapshot

Documentation-level merge completed for:

- config-driven model / target / design / experiment layering
- Vivado-only automation overview
- VCK190 baseline runbook
- ZynqMP Vivado-only runbook references
- ZynqMP Spinal-backed runbook references
- SRC/case analysis document set

Code-level merge still pending for:

- actual `workspace/flow/entrypoints/python/` migration
- actual `workspace/flow/scripts/run/` migration
- actual `workspace/hardware/` asset migration
- actual `configs/` tree merge

## Practical Validation Plan After Code Merge

1. revalidate the VCK190 baseline path
2. confirm HLS report generation into `workspace/artifacts/reports/<target>/<model>/<design>/`
3. confirm Vivado-only `board-impl` / `board-bitstream` flows
4. confirm ZynqMP Spinal-backed export and bitstream prepare flow
5. compare results against the legacy Spinal oracle where applicable
