# ZCU104 Full Learned Physical Experiment Plan - 2026-07-01

## Scope And Read Method

- Goal: make the full learned Search/Track cyclic accelerator physically credible on ZCU104, while preserving the single E2E AXIS runtime top, shared ATTN/MLP blocks, learned Conv/Head/Transformer parameters, on-chip nonlinear LUTs, and Search 10% / Track 90% hybrid reporting path.
- Target: PL clock `300 MHz`; user-approved experiment continuation threshold is routed `WNS >= -0.500 ns`; official signoff remains `WNS >= 0.000 ns`.
- Latency target: Search `<= 4.000 ms`; Track `<= 1.000 ms`.
- Read policy for this document: no persistent index artifact is used. The two requested roots were directly opened and read in full:
  - `hardware/analysis`
  - `/home/kjm26/project/PRJXR/References/reports`
- Direct full-content read verification: `548` files, `213,905` lines, `15,649,226` bytes.
- The previous generated inventory/index files were removed and are not used as evidence:
  - `hardware/docs/track/ZCU104_REFERENCE_FILE_INVENTORY_2026_07_01.tsv`
  - `hardware/docs/track/ZCU104_REFERENCE_SECTION_INDEX_2026_07_01.tsv`
  - `hardware/docs/track/ZCU104_REFERENCE_FUNCTION_LINE_INDEX_2026_07_01.tsv`

## Prompt Brief

### Engineering Goal

Build a concrete experiment plan that can close the remaining gap between the current full learned physical implementation and the final ZCU104 target:

1. Preserve full learned computation, not a pruned shell.
2. Preserve runtime Search/Track dispatch inside one E2E AXIS cyclic top.
3. Keep shared ATTN/MLP temporal reuse rather than duplicating Search and Track models.
4. Reach Search `<= 4 ms` and Track `<= 1 ms` at 300 MHz.
5. Keep ZCU104 fit with enough DSP, URAM, BRAM, LUT, and timing margin to survive routed implementation.
6. Produce evidence for the eight requested reporting questions: hybrid latency/power/resource, Search-only, Track-only, active paths, throughput/II/rate/worst-case, measurement methodology, DMA/batch/percentiles/distribution, and major-block utilization.

### Route Selection

- Chosen route: Algorithm-to-Hardware Mapping + DSE/Pareto Search.
- Why: current failures are not single-line bugs. Search latency, DSP pressure, LUT pressure, URAM pressure, scheduler overlap, and measurement evidence are coupled.
- Rejected as primary route: pure RTL timing cleanup. It can improve WNS but cannot close Search latency or prove full learned semantics by itself.

### Expert Perspectives

- FPGA implementation lead: protect routed resource preservation, WNS, and bit/hwh generation.
- Transformer dataflow lead: reduce Search controller/MLP/attention cycles without breaking shared cyclic semantics.
- Measurement lead: separate HLS estimates, Vivado vectorless power, RTL activity, and board-measured distributions.

### Sub-Agent Note

No real sub-agent was launched for this document. The available runtime instruction for the lazy multi-agent tool says not to spawn sub-agents unless the user explicitly asks for sub-agent/delegation work. This document therefore records the required task-card split, but the main agent performed the extraction, synthesis, and documentation directly.

## Current Evidence Baseline

### AQ2 Search/Track Physical Baseline

From `hardware/docs/track/AQ2_REQUESTED_SEARCH_TRACK_METRICS_2026_07_01.md`:

| Item | Search AQ2 | Track AQ2 | Judgment |
|---|---:|---:|---|
| Latency max | `1,709,783 cycles / 5.699277 ms` | `268,710 cycles / 0.895700 ms` | Search fail, Track pass |
| Initial interval | `1,709,784 cycles / 5.699280 ms` | `268,711 cycles / 0.895703 ms` | Search fail |
| Invocation rate from II | `175.46 inv/s` | `1,116.44 inv/s` | Track enough |
| Implemented LUT | `77,494 / 230,400 = 33.63%` | `80,565 / 230,400 = 34.97%` | physically fits |
| Implemented BRAM tile | `191.5 / 312 = 61.38%` | `185.5 / 312 = 59.46%` | fits |
| Implemented URAM | `92 / 96 = 95.83%` | `64 / 96 = 66.67%` | Search has almost no URAM margin |
| Implemented DSP | `1,257 / 1,728 = 72.74%` | `1,270 / 1,728 = 73.50%` | fits |
| Routed WNS | `0.000 ns` | `-0.017 ns` | user threshold pass |
| Total vectorless power | `7.032 W` | `6.777 W` | no board power yet |

Boundary:

- HLS major-block attribution is available, but Vivado power only splits PS8, AXI/DMA/memory, and top E2E IP.
- No board-runtime JSON is available for mean/median/min/max/P95/P99 or measured DMA counters.
- External DDR and sensor I/O rail power are not measured.

### AQ2 Major Latency Bottleneck

| Major block | Search cycles | Track cycles | Immediate implication |
|---|---:|---:|---|
| AXIS read | `16,386` | `4,109` | not dominant |
| Conv/patch embedding | `258,052` | `86,020` | Search conv is material but not the only blocker |
| Global buffer load | `12,292` | `3,076` | small |
| Controller run | `1,397,902-1,397,922` | `168,806` | primary Search bottleneck |
| MLP head | `25,114` | `6,682` | secondary |
| Top | `1,709,763-1,709,783` | `268,710` | Search must remove about `509,783` cycles versus 4 ms target |

Controller internal evidence:

- Search shared attention: `131,388 cycles`.
- Search shared MLP: `211,151 cycles`.
- Search body loop: `1,370,176 cycles`, trip `4`.
- Track body loop: `154,946 cycles`, trip `2`.

### Recent Fastest Search/Track DSE Points

| Candidate | Search | Track | Fit status | Decision |
|---|---:|---:|---|---|
| `qkvbram_tail2 + exppart + w1c2 + AQ8 + hpar8` | `1,288,836 cycles / 4.296 ms` | `229,783 cycles / 0.766 ms` | DSP/LUT over ZCU104 | Fastest pair, not promotable |
| `AQ8 + hpar8 + ACC24` Search-only | `1,284,676 cycles / 4.282 ms` | n/a | DSP/LUT over ZCU104 | best Search-only latency, still `84,676 cycles` over target |
| HPar16 | `2,229,315 cycles / 7.431 ms` | n/a | DSP `195%`, LUT `174%` | rejected; GELU ROM port II bottleneck |
| PatchPar64 | `1,296,963 cycles / 4.323 ms` | n/a | DSP `139%`, LUT `155%` | rejected; patch loop memory-port II pressure |

### Current Active 300 MHz Combined Runtime Candidate

- Active combined profile: `par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16`.
- Routed evidence: WNS `-0.236 ns`, route errors `0`, hold clean, bitgen pass.
- Resource evidence: CLB LUT `73,133/230,400 = 31.74%`, Block RAM Tile `216/312 = 69.23%`, URAM `76/96 = 79.17%`, DSP `1,723/1,728 = 99.71%`.
- Goal-status: experiment-pass but missing board, RTL cosim, official timing clean signoff, and measured distributions.
- Important guardrail: the later `compute_keep` timing-clean profile had WNS `0.064 ns` but collapsed to a pruned shell with DSP `1/1,728`; positive timing alone is invalid unless full learned compute is preserved.

## Direct Line-Level Technical Extraction

The table below is the working transfer map from the directly read files. For codebase reports, "function/module" refers to the function or module named in the report. For paper-only Markdown summaries, it refers to the method block or algorithmic unit documented at those lines.

| Source | Lines | Function/module/method | Transferable method | HGTXR action |
|---|---:|---|---|---|
| `hardware/analysis/NO_BOARD_PERFORMANCE_EXPERIMENT_PLAN_2026_06_26.md` | 14, 57-72 | DSE/resource policy | Avoid LUT-heavy arithmetic; use DSP for MAC-heavy paths; URAM for large buffers; LUTRAM for small tables; define QKV/MLP, memory, quant, nonlinear stress matrix | Keep as global experiment policy |
| `hardware/analysis/NO_BOARD_PERFORMANCE_EXPERIMENT_PLAN_2026_06_26.md` | 94-116 | DSE axes/metrics | Parallelism, bank count, memory primitive, precision, exact/sparse/linear attention axes | Use as candidate matrix shape |
| `hardware/analysis/integrated-2026-06-26/HGTXR_DESIGN_EXPERIMENT_INTEGRATION.md` | 7-13 | Integration goal | Increase DSP in QKV/attention/MLP, use URAM for large buffers, LUTRAM only for small memories/tables, reduce LUT arithmetic | Adopt as fit policy |
| same | 16-24 | HXR-R0/R1/Q0/A0/A1/M0/T0 | Resource rebalance, memory binding, quant calibration, exact attention hardening, optional approximation, mode counters, report automation | Convert to full learned physical experiment stages |
| same | 44-53 | Measurement contract | Do not claim throughput, p95/p99, DMA, quant degradation without evidence | Enforce evidence boundary in final 8-question answers |
| `hardware/analysis/curated.../attention-dataflow-and-sparsity.md` | 14-36 | Attention dataflow | Dense QK/AV dominate; AURA SRAM residency, ViTCoD fixed sparse mask, ViTALiTy linear attention, HG-PIPE FIFO pipeline | Use exact tiled attention first; sparse/linear only SW-first |
| `hardware/analysis/curated.../memory-dataflow-and-uram-policy.md` | 14-17, 64-67 | Memory policy | Large buffers URAM, small score/prob tables LUTRAM, explicit FIFO binding, C3b/current baseline protected | Apply to Search URAM 92/96 pressure and small scratch relocation |
| `hardware/analysis/curated.../quantization-and-nonlinear-operators.md` | 14-38 | Nonlinear/quant evidence | Scale choice, softmax exp/recip, LN/GELU precision; require SHA, scale formula, LUT range/error, bit-accurate CSim | Add operator manifest before replacing LUTs |
| `hardware/analysis/curated.../hls-rtl-integration-and-toolflow.md` | 20-37, 57-60 | Toolflow gate | SW golden -> CSim -> CSynth -> IP -> Vivado -> PYNQ; distinguish HLS, Vivado, board evidence | Promotion gate sequence |
| `hardware/analysis/curated.../sparse-moe-and-routing.md` | 13-24, 42-45 | Routing/sparsity | Dynamic unbounded routing not acceptable; bounded Search/Track heads and fixed sparse token route only | Keep runtime scheduler bounded |
| `hardware/analysis/curated.../xr-eye-tracking-hardware-bridge.md` | 21-34 | XR metric bridge | Need p95/p99, motion/mode split, board power, Search/Track distribution | Board-measurement stage |
| `hardware/docs/track/AQ2_REQUESTED_SEARCH_TRACK_METRICS_2026_07_01.md` | 31-44 | Evidence boundary | HLS deterministic percentiles only; vectorless power not internal Conv/ATTN/MLP split | Do not overclaim measurements |
| same | 70-84 | AQ2 latency | Search 5.699ms fail, Track 0.8957ms pass | Search closure is P0 |
| same | 103-120 | Major latency | Controller dominates Search | Optimize controller/MLP/attention loop, not DMA first |
| same | 124-167 | HLS resource by block | Controller HLS LUT exceeds device; attention and MLP dominate | Need resource-preserving dataflow redesign |
| same | 174-182 | Implemented resources | Search URAM 95.83%, Track URAM 66.67%, DSP ~73% | Search URAM margin must improve |
| same | 216-243 | Power | Search IP dynamic 3.393W, Track IP dynamic 3.129W, PS8 about 2.67W | Need SAIF/board for mode-specific block power |
| `References/.../Transformer-Accel/HG-PIPE.md` | 83-90 | `Adapter` | Pack/unpack/non-divisible width adapter for arbitrary parallelism | Use for variable hpar/attntok/w2 lane boundaries |
| same | 92-104 | `Matmul` | 3-stage dataflow: adapter, cache window, MAC; static/dynamic weight paths; DSP/fabric bind | Rework W2/shared GEMM as bounded shared lanes, not raw duplication |
| same | 106-122 | `LayerNorm`, `Softmax`, `GELU`, `Quant` | LN 3-pass rsqrt LUT, softmax exp+recip LUT, GELU/quant single LUT | Keep nonlinear LUTs but fix porting/replication bottleneck |
| same | 141-156 | `Attn`, `MLP` | Full dataflow composition for LN/QKV/QK/softmax/RV/O and LN/fc1/GELU/fc2/residual | Preserve functional full Transformer block |
| same | 162-187 | Spinal pipeline/on-chip | FIFO + ap_ctrl_chain coarse pipeline; all weights/intermediates on-chip | Good reference, but full layer replication conflicts with cyclic shared objective |
| `References/.../ViT-Accelerator/AURA-FlashAttention-AISC-Accelerator.md` | 103-114 | `PE` | Dot -> running max -> expmul -> vector division elastic pipeline | Use row-wise lifetime/online softmax idea if exactness or calibrated approximation passes |
| same | 116-128 | `dot_product`, reduction tree | Q reuse, K/V stream, tree reduction, V delay alignment | Apply to attention reduction scheduling |
| same | 136-164 | `expmul` | Running max/sum, denominator bundled with vector, shift-only exp approximation | SW-first only unless accuracy contract passes |
| same | 188-201 | memory controller/SRAM | K/V full-sequence retention, Q/O ping-pong, load/compute/store overlap | Use for exact tiled attention buffering |
| `References/.../ViT-Accelerator/efficient-transformer-accelerator.md` | 78-102 | PE/systolic | INT8 outer-product partial-sum streaming, no local PE accumulation | Reference for W2 shared-lane output accumulation |
| same | 112-148 | `accumulator_bank`, `quant_shared` | 32b saturation, shared requantizer: 512 outputs through 64 units over 8 cycles | Use time-multiplexed W2/requant to cut DSP/LUT |
| same | 150-158 | integrated top | systolic -> accumulator -> quant_shared | Candidate microarchitecture template for GEMM subpath only |
| `References/.../ViT-Quantization/P2-ViT.md` | 80-93 | `MinmaxObserver`, `PtfObserver` | Output-aware PoT scale and channel-wise PoT factors | Generate scale manifest and shift-only requant candidate |
| same | 108-124 | `Log2Quantizer`, `QIntLayerNorm`, `QIntSoftmax` | Softmax probability as powers of two; integer LN/softmax | SW/CSim operator stress before hardware replacement |
| same | 167-199 | PoT/PTF/LIS formulas | Requant by shifts, PTF channel shifts, log2 attention | Good for operator cost reduction after functional baseline |
| `References/.../ViT-Quantization/FQ-ViT.md` | 59-80 | PTF/LIS core | Channel `2^k` scale, integer LN, log-int softmax | Secondary confirmation of P2-ViT route |
| same | 128-137 | HGTXR implications | 4-bit attention map, PTF scale ROM, no-float datapath caveats | Add exactness/accuracy gate before replacing current LUTs |
| `References/.../transformer_quant.../03_ViTCoD...md` | 22-30 | Fixed sparse attention | Static mask, Q/K movement reduction, denser/sparser split | Potential Search-only successor, not default |
| same | 110-168 | split/conquer, AE, parser | fixed mask, global tokens, compiler-generated runtime config | SW-first bounded sparse experiment |
| same | 220-225, 239-249 | setup/tradeoff | 512 MACs, 500MHz ASIC sim; memory-bound sparse risk | Do not use reported speedups directly for ZCU104 claims |
| `References/.../04_Integer_Only_Group_Vector...md` | 22-38 | integer-only ViT | LN/Softmax/GELU approximation, unified package, group-vector systolic | Use as long-term nonlinear/data-layout redesign |
| same | 163-169 | methodology | simultaneous mean/variance, K transpose by address reorder, on-chip BRAM feature storage | Apply to buffer/data layout DSE |
| same | 221-238 | setup/results | ZCU102 300MHz, ViT-tiny 4.077ms, board power | Useful measurement/reporting precedent |
| `References/.../xr_eye_tracking.../07_EX_Gaze...md` | 31-55 | Search/Track system analogy | frame relocalization + event tracking; 2KHz; relocalization/event/adaptation runtime split | Good framing for 10/90 hybrid distribution |
| same | 150-166 | architecture | frame init, event patch accumulation, SEPT, adaptation, gaze regression | Helps define active path reporting |

## What To Apply Now Versus Later

### Apply Now

1. Preserve `par32_runtime_full_axi_mem16` and `par32_dsp_mixed_stream_mem16` as historical baselines, but use the current full learned physical candidate family for target closure.
2. Keep the exact full learned Transformer semantics and shared cyclic ATTN/MLP blocks.
3. Optimize Search controller/body loop first because Search miss is dominated by controller/MLP/attention cycles.
4. Treat `hpar8_acc24`-derived Search as the latency lower-bound reference, but do not promote it until DSP/LUT fit is restored.
5. Fix the HPar16 root cause through GELU ROM banking/replication or multiport emulation before increasing hidden-lane parallelism again.
6. Use time-multiplexed/shared W2 and requant paths, inspired by `quant_shared`, instead of direct W2 lane replication.
7. Add a hard resource-preservation gate before accepting any timing-clean result.
8. Add board JSON and SAIF/VCD activity collection to answer the eight reporting questions honestly.

### Apply Later / SW-First Only

1. Fixed sparse attention mask and ViTCoD-style global token split: useful only after software accuracy and bounded route are proven.
2. AURA ExpMul/shift-only softmax replacement: useful only after same-output or accepted accuracy error is defined.
3. P2/FQ-ViT PoT/PTF/LIS: good for reducing multipliers/tables, but requires calibration manifest and CSim operator stress.
4. HG-PIPE full layer replication: not compatible with the stated cyclic accelerator objective unless it is treated as a separate non-cyclic baseline.

### Reject As Default

1. Direct HPar16 without GELU ROM port fix: measured slower and over-resourced.
2. PatchPar64 without memory-port fix: worsens patch latency.
3. Timing-clean `compute_keep` if resource preservation collapses.
4. Dynamic or unbounded MoE routing: violates bounded Search/Track runtime objective.
5. Claims of p95/p99, DMA bandwidth, or internal block power from deterministic HLS or vectorless Vivado alone.

## Prioritized Experiment Plan

### P0 - Evidence And Guardrail Closure

Objective: prevent false wins and make every later result comparable.

| ID | Task | Implementation focus | Success gate | Reject gate |
|---|---|---|---|---|
| FLP-P0-01 | Resource-preservation gate | Parse routed hierarchy for E2E IP, DSP, URAM, BRAM, LUT; fail pruned shell | full learned compute resources present; DSP/BRAM/URAM above expected floor | DSP near zero, URAM zero, or missing compute hierarchy |
| FLP-P0-02 | Mode-specific deterministic functional gate | Search/Track CSim vectors, runtime state, TLAST, strict prefetch trace | same input -> same output for SW/HW CSim; Search state `0`, Track state `1` | vector mismatch or scheduler violation |
| FLP-P0-03 | RTL probe stabilization | Reuse profile-parametric direct XSIM probes; remove waveform-load fragility | direct kernel enters and completes or reaches bounded known-progress timeout | launch failure before kernel entry |
| FLP-P0-04 | Measurement manifest | One JSON/MD per candidate with clock, latency, II, resources, timing, power, evidence level | repeatable candidate ledger | mixed evidence without paths |

Immediate output:

- `candidate_status.json`
- `candidate_status.md`
- `resource_preservation_check.json`
- `mode_vector_check.json`

### P1 - Search Latency Closure Without Direct Resource Explosion

Objective: remove the remaining Search gap while keeping ZCU104 fit.

Measured pressure:

- Fast Search-only best is still about `84,676 cycles / 0.282 ms` above target.
- AQ2 Search is `5.699 ms`, so the physically clean baseline needs larger improvement.
- Direct hpar16 is invalid because GELU ROM becomes `ROM_1P_LUTRAM` and activation loop II becomes `64`.
- Direct lane replication pushes DSP/LUT far over ZCU104.

| ID | Task | Technical method | Expected effect | Gate |
|---|---|---|---|---|
| FLP-P1-01 | GELU ROM banking/replication | Partition/replicate GELU ROM to match hidden-lane read ports; keep table small; verify HLS schedule II | eliminate HPar16 ROM-port regression | activation loop II no worse than hpar8; DSP/LUT not over target |
| FLP-P1-02 | Shared/time-multiplexed W2 path | Replace direct W2 lane duplication with `quant_shared`-style batches and saturated accumulators | reduce DSP/LUT while preserving Search speed | lower DSP than hpar8_acc24 with similar controller cycles |
| FLP-P1-03 | W2 local hgroup cache | Keep W2 reuse local to hgroup, avoid high-fanout global reads | lower route delay and memory-port pressure | routed WNS >= -0.5, no pruned shell |
| FLP-P1-04 | Search controller overlap audit | Ensure dispatcher prefetch ends before prior layer completion; next block starts immediately | reduce idle/control bubbles | strict prefetch trace gaps `0` |
| FLP-P1-05 | Head/Conv micro-closure | only after controller improves: reduce frame conv/head residual cycles | small extra Search margin | no new memory-port II regression |

Ranking:

1. FLP-P1-01, because a known direct experiment exposed the exact ROM port bottleneck.
2. FLP-P1-02, because direct lane replication is not resource-fit.
3. FLP-P1-03, because routed failures are often route/fanout dominated around DSP/memory.
4. FLP-P1-04, because Search is multi-layer and any control bubble multiplies.
5. FLP-P1-05, because patch/head are not the dominant term yet.

### P2 - Attention/Dataflow Refinement

Objective: reduce attention cycles and memory movement without changing semantics first.

| ID | Task | Technical method | Evidence source | Gate |
|---|---|---|---|---|
| FLP-P2-01 | Exact tiled attention | K/V retention, Q/O ping-pong, row-wise lifetime, V-delay alignment | AURA memory/PE patterns; HG-PIPE dynamic matmul | CSim equality and lower QK/RV cycles |
| FLP-P2-02 | Reduction tree refactor | tree reduction for QK and LN reductions where II/fanout dominates | AURA tree_reduce | lower estimated clock pressure without LUT explosion |
| FLP-P2-03 | Fixed sparse mask SW-first | fixed Search mask and global token route | ViTCoD | only promote if software accuracy and bounded hardware schedule pass |
| FLP-P2-04 | Linear/Taylor attention SW-first | ViTALiTy/Glide-style approximation | efficient-transformer report | optional successor, not full learned exact baseline |

Do not replace exact attention until the exact baseline is closed or an explicit accuracy trade-off is accepted.

### P3 - Memory Binding And Resource Fit

Objective: restore margin in Search URAM/DSP/BRAM without creating LUT/timing failure.

| ID | Task | Current evidence | Action | Gate |
|---|---|---|---|---|
| FLP-P3-01 | Search URAM relief | AQ2 Search URAM `92/96`; active combined `76/96`; tail16 LUTBUF reduces URAM to `28/96` but raises LUT/TNS | move only small high-bank buffers to LUTRAM; do not move deep hidden buffers by default | URAM <= 80/96 and WNS >= -0.5 |
| FLP-P3-02 | Balanced DSP relief | tail2_dsppipe4 passes WNS floor with DSP `1,713/1,728`; tail4/tail16 timing worse or LUT heavy | keep tail2_dsppipe4 as balanced pressure-relief row | DSP spare >= 15 and no route collapse |
| FLP-P3-03 | BRAM/LUTRAM threshold | targeted token LUTRAM saved BRAM but missed WNS by 0.060ns | sweep only tokens/score/prob first; freeze norm/Q/K/V unless needed | BRAM relief with WNS >= -0.5 |
| FLP-P3-04 | DONT_TOUCH relaxation | compute_keep pruned shell; full dont_touch creates timing pressure | selective keep hierarchy plus resource-preservation check | timing improvement with compute preserved |

### P4 - Quantization And Nonlinear Operator Calibration

Objective: lower arithmetic/table cost while preserving or explicitly quantifying numerical behavior.

| ID | Task | Method | Gate |
|---|---|---|---|
| FLP-P4-01 | Scale manifest | Generate weight/LN/softmax/GELU LUT metadata: SHA, scale formula, entries, ranges, max error | manifest complete |
| FLP-P4-02 | PoT/PTF candidate | P2/FQ-style PoT scale and PTF LN channel shifts | SW/CSim operator equality or accepted error |
| FLP-P4-03 | Log2 softmax candidate | represent probability as `2^-q`; reduce multiply/storage | SW accuracy and CSim pass |
| FLP-P4-04 | Integer-only LN/GELU stress | stress extreme input ranges and saturation | no overflow/NaN; bounded max error |

This stage should not block P1 exact-path latency closure unless P1 runs out of resource margin.

### P5 - Vivado Promotion Flow

Objective: promote only candidates that survive physical implementation.

Promotion sequence:

1. CSim Search/Track.
2. CSynth Search-only and Track-only, then combined if applicable.
3. IP package and interface audit.
4. Vivado route/bit/hwh at 300 MHz.
5. Resource-preservation gate.
6. Timing gate: experiment `WNS >= -0.500 ns`, official `WNS >= 0.000 ns`.
7. Vectorless power report with hierarchy.
8. PYNQ bundle and dry-run plan.
9. Board JSON import when ZCU104 is available.

Candidate cannot be promoted if it only improves one mode while breaking the combined runtime scheduler, full learned compute preservation, or Track `<= 1 ms`.

### P6 - Board And Power Measurement For The Eight Questions

Objective: produce measured answers, not HLS proxies.

Required measurements:

| Measurement | Required artifact | Why |
|---|---|---|
| Search latency distribution | Search board JSON, repeated samples | mean/median/min/max/P95/P99 |
| Track latency distribution | Track board JSON, repeated samples | same |
| Hybrid 10/90 distribution | interleaved runner JSON | realistic mean, P95/P99, worst-case |
| DMA bandwidth | board-side byte counter/timestamps | effective and useful payload bandwidth |
| Power | Vivado vectorless + SAIF/VCD + board rail readings if available | split methodology |
| Internal block power | preserve deeper hierarchy or separate report hooks | Conv/ATTN/MLP/Head power |
| DDR/sensor I/O | board rail instrumentation | currently not captured by Vivado report |
| Throughput/GOPS | ops model + measured latency/II | comparable throughput |

## Concrete Work DAG

```text
P0 evidence gates
  -> P1 Search latency closure
      -> P3 resource fit cleanup
          -> P5 Vivado promotion
              -> P6 board/hybrid metrics
  -> P2 exact attention refinement
      -> P1/P3 if it improves cycles or fit
  -> P4 quant/nonlinear calibration
      -> optional replacement candidates after SW/CSim proof
```

## Task Cards

```yaml
task_card:
  task_id: FLP-P0
  sub_agent: "not-launched; main-agent fallback"
  role: "evaluator"
  objective: "Build resource-preservation, deterministic functional, and evidence-level gates for full learned candidates."
  file_ownership:
    - "hardware/tools/*goal_status*.py"
    - "hardware/tools/*contract*.py"
    - "hardware/generated/signoff/"
  assigned_skill:
    - "algorithm-hardware-codesign-expert"
  inputs:
    - "CSim logs"
    - "CSynth reports"
    - "Vivado utilization/timing/power reports"
  outputs:
    - "candidate status JSON/Markdown"
  validation:
    - "no pruned shell"
    - "same Search/Track vectors"
    - "evidence paths exist"
  dependencies: []
```

```yaml
task_card:
  task_id: FLP-P1
  sub_agent: "not-launched; main-agent fallback"
  role: "implementer"
  objective: "Close Search latency by fixing GELU ROM ports and replacing direct W2 lane replication with shared/time-multiplexed compute."
  file_ownership:
    - "hardware/hls/src/"
    - "hardware/vivado/scripts/"
    - "hardware/scripts/run/"
  assigned_skill:
    - "algorithm-hardware-codesign-expert"
  inputs:
    - "hpar8_acc24 reports"
    - "hpar16 root-cause reports"
    - "AQ2 latency breakdown"
  outputs:
    - "new HLS profiles"
    - "CSim/CSynth reports"
  validation:
    - "Search <= 4ms candidate"
    - "Track <= 1ms retained"
    - "DSP/LUT not over target"
  dependencies:
    - "FLP-P0"
```

```yaml
task_card:
  task_id: FLP-P3-P5
  sub_agent: "not-launched; main-agent fallback"
  role: "physical evaluator"
  objective: "Turn the best HLS candidates into routed ZCU104 bitstreams without resource collapse."
  file_ownership:
    - "hardware/generated/build/vivado/"
    - "hardware/docs/Validation.md"
    - "hardware/docs/track/PROGRESS.md"
  assigned_skill:
    - "algorithm-hardware-codesign-expert"
  inputs:
    - "best P1/P2/P3 candidates"
  outputs:
    - "bit/hwh"
    - "routed reports"
    - "promotion decision"
  validation:
    - "WNS >= -0.5ns"
    - "route errors 0"
    - "full learned resources preserved"
  dependencies:
    - "FLP-P1"
```

```yaml
task_card:
  task_id: FLP-P6
  sub_agent: "not-launched; main-agent fallback"
  role: "measurement"
  objective: "Collect board/hybrid/power evidence needed for the eight reporting questions."
  file_ownership:
    - "hardware/pynq/hgtxr/"
    - "hardware/generated/pynq/"
    - "hardware/docs/track/"
  assigned_skill:
    - "algorithm-hardware-codesign-expert"
  inputs:
    - "promoted bit/hwh"
    - "ZCU104 runtime"
  outputs:
    - "Search/Track/hybrid JSON"
    - "DMA and latency histograms"
    - "power methodology report"
  validation:
    - "repeated samples"
    - "P95/P99 computed from measured data"
    - "measurement boundary explicit"
  dependencies:
    - "FLP-P5"
```

## Eight-Question Answer Plan

| User question group | Can answer now? | Current answer level | Additional experiment needed |
|---|---|---|---|
| 1. HW Top resource, hybrid latency, hybrid power breakdown | Partially | HLS/Vivado estimates and synthetic 10/90 can be computed; no measured hybrid JSON | interleaved board run, SAIF/VCD or board power |
| 2. Search-mode latency/power breakdown | Partially | AQ2 Search HLS + Vivado vectorless power available | board repeated samples; internal block power hooks |
| 3. Search active blocks/path | Mostly | Search path: AXIS -> Frame Conv -> global buffer -> controller 4 blocks -> head | waveform/trace for exact runtime activity |
| 4. Track-mode latency/power breakdown | Partially | AQ2 Track HLS + Vivado vectorless power available | board repeated samples; internal block power hooks |
| 5. Track active blocks/path | Mostly | Track path: AXIS -> Event Conv -> global buffer -> controller 2 blocks -> head | waveform/trace for exact runtime activity |
| 6. Search/Track only vs Hybrid throughput/II/rate/worst-case | Partially | II and deterministic HLS rates available; hybrid synthetic possible | board hybrid 10/90 JSON |
| 7. Measurement methodology, inclusion boundaries, DMA, batch, throughput, percentiles, distribution | Partially | methodology and HLS proxy values available | measured DMA counters, board samples, distribution import |
| 8. Source utilization absolute/percentage by major block | Partially | HLS major block + Vivado top hierarchy available | deeper preserved hierarchy for Conv/ATTN/MLP/Head implemented split |

## Priority Output

1. P0 guardrails are mandatory before more DSE. The project already has a concrete warning case: `compute_keep` became timing-clean but pruned away full compute. Any future "success" must first prove full learned resources are preserved.
2. P1 Search closure is the highest technical priority. Track already passes 1 ms in AQ2, while Search fails. Search controller/MLP is the dominant target.
3. P1 should not repeat direct hpar16 or PatchPar64. The next useful experiment is GELU ROM banking/replication plus shared/time-multiplexed W2, not more raw lane duplication.
4. P3 resource fit must run with P1, not after it. The promising low-latency Search profiles are currently over DSP/LUT, and AQ2 Search has URAM `92/96`.
5. P2 exact attention tiling is next if P1 stalls. Use K/V retention, Q/O ping-pong, and reduction-tree scheduling before considering sparse or approximate attention.
6. P4 PoT/PTF/LIS is useful but must be SW/CSim-gated. It should not replace exact full learned behavior unless accepted by accuracy/equality evidence.
7. P5 Vivado promotion is only meaningful with route, bit/hwh, timing, resource-preservation, and vectorless power together.
8. P6 board measurement is required for the final eight-question report. P95/P99, measured DMA bandwidth, and board power cannot be honestly produced from current deterministic HLS and vectorless Vivado alone.

## Acceptance Criteria

- Completeness: all eight requested reporting categories have either a current answer path or a named missing experiment.
- Evidence: every technical recommendation is tied to directly read HGTXR or reference report evidence.
- Executability: experiments are ordered as CSim -> CSynth -> IP -> Vivado -> board.
- Consistency: Search/Track terms, 300 MHz clock, WNS threshold, and full learned objective are consistent with current docs.
- Safety: no credentials, hidden reasoning, or private tokens are stored.
- Maintainability: rejected experiments and promotion gates are explicit so future sessions do not repeat known dead ends.

## Immediate Next Actions

1. Implement FLP-P0 resource-preservation gate if not already wired into the candidate runner.
2. Add FLP-P1-01 GELU ROM replication/banking macro and run Search-only CSim/CSynth against the hpar8/hpar16 root case.
3. Add FLP-P1-02 shared W2 time-multiplex profile and compare against `hpar8_acc24`.
4. Promote only candidates that meet both latency and resource-preservation gates into Vivado.
5. Run PYNQ Search/Track/hybrid 10/90 JSON collection on the first promoted candidate.
