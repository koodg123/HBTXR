# No-Board Performance Experiment Plan

Date: 2026-06-26
Scope: HGTXR hardware analysis synthesis. Board inference and PYNQ smoke runs are excluded in this plan.

## 1. Prompt Brief

| Field | Content |
|---|---|
| Goal | Improve HGTXR performance and resource balance without board inference, using analysis-derived HLS/CSim/csynth/software experiments. |
| Inputs | `analysis/vit-accel`, `analysis/curated-2026-06-26`, `analysis/integrated-2026-06-26`, C3b resource/power/latency evidence copies. |
| Assumptions | Current C3b remains protected; new hardware work is suffix-isolated; HLS/Vivado estimates are not board runtime claims. |
| Unknowns | Real board latency, DMA bandwidth, p95/p99, search/track runtime distribution, and physical power remain unavailable by design in this plan. |
| Constraints | Avoid LUT-heavy arithmetic; increase DSP use for MAC-heavy compute; use URAM for large buffers and LUTRAM for small tables. |
| Expected outputs | Experiment manifests, SW/CSim reports, csynth resource/latency matrix, optional routed timing/power reports, no board JSON. |
| Acceptance | Each promoted no-board experiment has reproducible scripts, exact source paths, metrics, and explicit pass/fail or defer decision. |

## 2. Evidence Sources

| Source | Role in this plan |
|---|---|
| `analysis/curated-2026-06-26/categories/memory-dataflow-and-uram-policy.md` | URAM/LUTRAM memory binding policy and C3b successor direction |
| `analysis/curated-2026-06-26/categories/attention-dataflow-and-sparsity.md` | Attention exact/approximation split and safety gates |
| `analysis/curated-2026-06-26/categories/quantization-and-nonlinear-operators.md` | Q4/Q8, PoT scale, LUT/nonlinear calibration requirements |
| `analysis/curated-2026-06-26/categories/hls-rtl-integration-and-toolflow.md` | HLS/csynth/report hygiene and variant isolation |
| `analysis/curated-2026-06-26/categories/sparse-moe-and-routing.md` | Search/track routing constraints and SW-first policy |
| `analysis/vit-accel/experiment_extensions_2026_06_15.md` | Existing P0/P1/P2 reference-derived experiment list |
| `analysis/integrated-2026-06-26/HGTXR_DESIGN_EXPERIMENT_INTEGRATION.md` | Current integrated experiment queue |
| `analysis/curated-2026-06-26/design-assets/hgtxr-local-evidence/c3b_resource_power_latency_tables_2026_06_23.md` | C3b baseline resource/latency/power evidence |

## 3. Current Baseline To Protect

| Metric | Current C3b evidence | Planning implication |
|---|---:|---|
| HLS top latency | 37,508,072 cycles | Transformer controller dominates; target QKV/attention/MLP first |
| HLS latency at 200 MHz | 0.188 s, 5.33 fps estimate | No board claim; use as HLS comparison baseline only |
| LUT | 126,506 / 230,400, 54.91% | LUT pressure is high enough to reject LUT-heavy compute by default |
| DSP | 604 / 1,728, 34.95% | There is room to raise MAC parallelism if timing/resource remain feasible |
| BRAM_18K | 332 / 624, 53.21% | Keep BRAM use controlled; move only truly large buffers to URAM |
| URAM | 64 / 96, 66.67% | URAM is already meaningful; new large buffers must justify added URAM |
| Board metrics | excluded | Do not plan PYNQ smoke, board DMA, board p95/p99 in this phase |

## 4. Experiment Strategy

| Phase | Focus | Why this order |
|---|---|---|
| Phase 0 | Report automation and baseline extraction | Prevent stale or manual resource claims before changing design |
| Phase 1 | Resource rebalance DSE | Directly addresses low DSP use and LUT pressure |
| Phase 2 | Memory binding DSE | Converts analysis guidance into URAM/BRAM/LUTRAM evidence |
| Phase 3 | Quantization and nonlinear calibration | Can improve accuracy or reduce rescale cost without board execution |
| Phase 4 | Exact attention dataflow hardening | Keeps paper-scope behavior while exploring memory/latency improvements |
| Phase 5 | Optional approximation/routing ablations | Higher risk; SW/CSim-only until accuracy and boundedness are proven |
| Phase 6 | Negative controls | Document why rejected methods are not default, with minimal effort |

## 5. Primary Experiment Matrix

| ID | Priority | Experiment | Source basis | What to run, board excluded | Expected outcome | Required metrics | Promotion rule |
|---|---|---|---|---|---|---|---|
| `NB-P0-00` | P0 | Report collector and baseline lock | Prometheus, HLS/RTL toolflow docs | Parse current C3b and successor csynth/routed reports into one Markdown/CSV matrix | Reliable baseline and comparison table | LUT, FF, DSP, BRAM, URAM, latency, II, clock target, evidence path | Required before ranking any new variant |
| `NB-P0-01` | P0 | QKV/MLP parallelism sweep | HG-PIPE, MSD-FCCM23, efficient-transformer-accelerator | Generate `par16`, `par24`, `par32` or feasible equivalents for QKV/MLP lanes; enforce DSP-bound MACs | Higher DSP utilization and lower latency if timing allows | Per-block DSP/LUT/FF/BRAM/URAM, latency cycles, II | Keep if latency improves and LUT does not grow disproportionately |
| `NB-P0-02` | P0 | DSP binding audit | C3b resource tables, HG-PIPE | Add/verify HLS directives or code patterns that prevent multipliers from falling into LUT fabric | Lower accidental LUT arithmetic | HLS operator/resource report by MHA/QKV/MLP | Keep if DSP rises on MAC blocks and LUT pressure drops or stays bounded |
| `NB-P0-03` | P0 | Large-buffer URAM binding sweep | ME-ViT, AURA, TATAA, HG-PIPE | Sweep explicit bindings for Q/K/V, hidden, activation, deep FIFO, score/value buffers | Cleaner URAM use and reduced BRAM/LUTRAM misuse | Memory object mapping, BRAM/URAM/LUTRAM counts, latency | Keep if large buffers are explicit and timing/resource improve or remain acceptable |
| `NB-P0-04` | P0 | Small-buffer LUTRAM threshold sweep | P2-ViT, Edge-MoE, memory policy | Try thresholds for score/prob/scratch/scale/routing tables as LUTRAM | Avoid wasting BRAM/URAM on small structures | Named memory mapping, LUTRAM count, LUT logic delta | Keep only if LUT logic growth is small and BRAM/URAM savings are useful |
| `NB-P1-01` | P1 | PoT scale calibration | P2-ViT, FQ-ViT, PTQ4ViT | Run SW reference and CSim-compatible fixed-point calibration candidates | Lower rescale cost with no accuracy regression | scale manifest, mean/median/p95 error, max error, pass/fail vectors | Promote to HLS only after bit-accurate CSim pass |
| `NB-P1-02` | P1 | Q4/Q8 deterministic coverage | Quantization category, Req5 manifest policy | Generate deterministic vectors for residual, LayerNorm, QKV, attention, MLP, output state | Better confidence in packed weights and fixed-point path | vector count, mismatch count, max abs/rel error | Required before changing precision-sensitive kernels |
| `NB-P1-03` | P1 | Nonlinear operator stress | HG-PIPE, TATAA, Transformer softmax references | Stress Softmax, reciprocal, exp, GELU, LayerNorm LUT/precision variants in SW/CSim | Identify precision-sensitive corners before synthesis | LUT size, input range, precision, max/mean/p95 error | Promote only bounded LUT tables, not LUT-heavy arithmetic |
| `NB-P1-04` | P1 | Exact attention tiled dataflow | AURA, TATAA, HG-PIPE | Keep exact attention math but test tiled/lifetime-aware local buffering | Lower memory movement without accuracy scope change | CSim exactness, csynth latency/resource, buffer mapping | Prefer over approximate attention if it improves latency/resource |
| `NB-P2-01` | P2 | Static sparse attention ablation | ViTCoD | SW and optional CSim test with fixed masks only | Possible attention compute reduction | accuracy/error delta, sparsity, load balance, fallback result | Do not promote if dynamic irregular control is needed |
| `NB-P2-02` | P2 | Linear/Taylor attention ablation | ViTALiTy | SW-only first, then CSim only if exact fallback exists | Estimate latency/resource benefit vs approximation error | mean/median/p95 output error, task accuracy proxy | Paper-scope decision required before HLS promotion |
| `NB-P2-03` | P2 | Static search/track mode counters | Edge-MoE, M3ViT, UbiMoE | Add software/HLS-sim counters or synthetic mode labels, not board runtime | Prepare mode-specific latency analysis | search/track count, per-mode simulated latency, worst-case path | Hardware routing only if bounded and accuracy improves |
| `NB-P3-01` | P3 | LUT-heavy negative control | LUT-GEMM, LUT-LLM, ternaryLLM | Report-only or tiny HLS microbenchmark | Evidence for rejecting LUT-heavy default path | LUT/DSP/resource delta, accuracy proxy if applicable | Keep as reviewer-facing rejection evidence |
| `NB-P3-02` | P3 | Framework automation feasibility | HLS4ML, allo, scalehls, Stream-HLS | Static/toolflow prototype only, no migration | Decide whether a compiler/DSL helps future work | setup cost, generated-code fit, risk list | Do not move C3b successor path unless clear benefit exists |

## 5.1 User-Provided Host/HLS Queue Update

| Experiment ID | 내용 | 현재 상태 | 예상 결과 | No-board plan mapping |
|---|---|---|---|---|
| `P1-PAR32-CSYNTH` | PAR32 exploratory csynth | completed | PAR16 대비 latency `-30.31%`, DSP `604 -> 1,148`; best csynth-fit candidate is `par32_dsp_mixed_stream_mem16` | `NB-P0-01`, after `NB-P0-00` |
| `P1-PAR32-ROUTE` | PAR32 viable candidate route | completed | ZCU104 routed timing pass: WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen completed | optional routed no-board evidence after csynth |
| `P2-SCALE-CALIB` | P2-ViT PoT scale calibration 확장 | planned | accuracy proxy 개선 후보 탐색 | `NB-P1-01` |
| `P2-Q4Q8-COVERAGE` | Q4/Q8 SW-HW test vector 추가 | planned | exact-match confidence 강화 | `NB-P1-02` |
| `P2-OPERATOR-STRESS` | LayerNorm/GeLU/Softmax/Quantization stress | planned | numerical corner case 발견 | `NB-P1-03` |
| `P2-DATASET-PROXY` | DeiT-like image/feature proxy test | planned | full dataset 전 accuracy proxy 확보 | SW-only proxy prerequisite for `NB-P2-*` |

## 5.2 User-Provided Architecture Exploration Queue Update

| Experiment ID | 내용 | 목적 | No-board plan mapping |
|---|---|---|---|
| `P3-URAM-BANKING` | URAM banking layout sweep | BRAM/LUT pressure 감소 | `NB-P0-03` |
| `P3-LUTRAM-SMALLBUF` | small buffer LUTRAM threshold sweep | RMU/SMU/cyclic scratch resource 정책 개선 | `NB-P0-04` |
| `P3-DSP-DENSITY` | dense path DSP utilization 증가 | LUT multiplier 회피, DSP 사용률 증가 | `NB-P0-01`, `NB-P0-02` |
| `P3-MEM16-PLUS` | memory bank count >16 탐색 | bank conflict 감소 가능성 확인 | DSE axis under `NB-P0-03` |

## 6. DSE Axes

| Axis | Candidate values | Applies to | Expected tradeoff |
|---|---|---|---|
| Parallelism factor | 16 baseline, 24, 32, optionally 48 if cost model permits | QKV, attention core, MLP | Latency down, DSP/FF/routing up |
| Memory bank count | 16 baseline, 24, 32 | QKV/global/hidden buffers | Fewer conflicts vs routing/control growth |
| Large memory primitive | BRAM, URAM, mixed | activation/QKV/hidden/deep FIFO | URAM use up, BRAM/LUT pressure down |
| Small memory primitive | LUTRAM, BRAM | score/prob/scratch/scale tables | BRAM down; LUT logic must stay bounded |
| Precision | current Q4/Q8, PoT scale variants, nonlinear higher precision only | quant/rescale/nonlinear | Accuracy up or equal; resource may rise |
| Attention math | exact baseline, exact tiled, fixed sparse, linear/Taylor | attention | Exact tiled is safest; approximation is gated |

## 7. Metrics To Report For Every No-Board Variant

| Metric family | Required fields |
|---|---|
| Functional correctness | CSim pass/fail, mismatch count, max abs error, max relative error |
| Accuracy/proxy | mean, median, p95 error where vectors/dataset exist; no invented subject/motion metrics |
| Latency | HLS min/max/avg cycles if available, II, stage breakdown |
| Resource | LUT, FF, DSP, BRAM_18K, URAM absolute and ZCU104 percentage |
| Block breakdown | RMU, SMU, MHA/QKV, attention core, MLP, nonlinear blocks where available |
| Memory mapping | each large buffer and small table: intended primitive vs reported primitive |
| Power | Vivado estimate only if routed; label as estimate and PL/system scope |
| Provenance | source commit/path, script, config, generated report path, timestamp |

## 8. Excluded In This Phase

| Excluded item | Reason |
|---|---|
| PYNQ board smoke/inference | User explicitly excluded board inference |
| Board DMA bandwidth | Requires board execution |
| Board p95/p99 latency | Requires repeated board/runtime measurement |
| Search/track runtime distribution from deployed hardware | Current top lacks internal counters and board run is excluded |
| Physical rail/wall power | Requires measurement setup |
| Final 3rd-goal signoff | Existing plan requires board evidence; this plan only prepares no-board candidates |

## 9. Recommended Execution Order

| Order | Experiment IDs | Stop/continue decision |
|---:|---|---|
| 1 | `NB-P0-00` | Continue only after baseline/resource parser can reproduce current C3b numbers |
| 2 | `NB-P0-01`, `NB-P0-02` | Choose the best parallelism point by latency/DSP/LUT Pareto |
| 3 | `NB-P0-03`, `NB-P0-04` | Keep memory bindings that improve clarity or reduce pressure without timing regression |
| 4 | `NB-P1-01`, `NB-P1-02`, `NB-P1-03` | Promote only numerically safe quant/nonlinear settings |
| 5 | `NB-P1-04` | Prefer exact tiled attention if it improves memory/latency |
| 6 | `NB-P2-01`, `NB-P2-02`, `NB-P2-03` | Keep as SW/CSim ablation unless accuracy and boundedness are clear |
| 7 | `NB-P3-01`, `NB-P3-02` | Use only to document rejected paths or future automation |

## 10. Success Criteria

| Level | Success definition |
|---|---|
| Minimum useful outcome | Baseline parser plus one new csynth resource matrix row for a DSP/parallelism or memory-binding variant |
| Strong outcome | A Pareto table showing at least one no-board candidate with lower HLS latency and acceptable LUT/DSP/URAM balance |
| Best no-board outcome | Resource-balanced candidate with CSim pass, csynth report, optional routed timing/power, quantization/error report, and no paper-scope violation |

## 11. Promotion Gates

| Gate | Rule |
|---|---|
| Board inference | Excluded; no PYNQ smoke, board DMA, physical p95/p99, or board power required in this plan |
| Functional correctness | CSim or SW deterministic vector mismatch count must be zero unless the experiment explicitly studies approximation error |
| HLS latency | Candidate should improve or explain any regression against C3b HLS top latency `37,508,072` cycles |
| LUT | Candidate should reduce LUT pressure or keep LUT growth justified by measurable latency/accuracy benefit |
| DSP | DSP may increase intentionally for QKV/MHA/MLP MAC parallelism; it is not capped at C3b `604` if timing and total ZCU104 budget remain acceptable |
| URAM | URAM may increase for large buffers only; each URAM object must be named and justified |
| Small memories | Small scale/routing/scratch tables should prefer LUTRAM only when LUT logic growth remains bounded |
| Optional routed evidence | Routed WNS/power are useful no-board evidence, but not required for first-pass csynth DSE |
| Approximation/routing | Attention approximation or mode routing requires exact fallback and explicit accuracy/error report |

## 12. Review Notes

Read-only sub-agent review agreed with the no-board priority order: quantization/resource stability, DSP and memory binding, reporting automation, then SW-first routing and attention approximation. One proposed cap, `DSP <= 604`, was not adopted because the user goal is to raise DSP utilization. The adopted rule is controlled DSP increase on MAC-heavy blocks, with LUT pressure and timing used as the limiting factors.

## 13. Immediate Next Task Cards

```yaml
task_card:
  task_id: NB-T0
  role: implementer
  objective: Build or update a report collector that normalizes C3b and successor HLS/Vivado reports into CSV/Markdown.
  file_ownership:
    - scripts/reporting/
    - analysis/no-board-results/
  validation:
    - Reproduces C3b top resource totals from existing evidence.
    - Emits paths for every parsed artifact.
```

## 14. P0 Execution Status

| Item | Status | Evidence |
|---|---|---|
| `NB-P0-00` report collector | completed | `scripts/report/collect_no_board_reports.py` |
| Existing HLS module collection | completed | `analysis/no-board-results/p0_report_collection_2026_06_26/hls_modules.csv`, `hls_modules.json` |
| Top variant collection | completed | `analysis/no-board-results/p0_report_collection_2026_06_26/hls_top_variants.csv` |
| Markdown summary | completed | `analysis/no-board-results/p0_report_collection_2026_06_26/summary.md` |
| C3b baseline reproduction | completed | `analysis/no-board-results/P0_PROGRESS_2026_06_26.md` |
| `NB-P0-01` / `P1-PAR32-CSYNTH` | completed | Best current csynth-fit candidate is `par32_dsp_mixed_stream_mem16` |
| `P1-PAR32-ROUTE` | completed | `par32_dsp_mixed_stream_mem16` routed on ZCU104 with WNS `3.885 ns`, route errors `0`, bit/hwh generated |
| `NB-P0-02` / `P3-DSP-DENSITY` | audit completed | QKV, attention score/value, WO, MLP FC1/FC2, and head use DSP-bound helpers controlled by `HGTXR_E2E_DENSE_PAR` |
| `NB-P0-03` / `P3-URAM-BANKING` | audit completed, sweep runnable | `tokens/norm/q/k/v/attn/hidden` are named URAM/BRAM macro-controlled buffers |
| `NB-P0-04` / `P3-LUTRAM-SMALLBUF` | audit completed, sweep runnable | `pooled/score/prob/exp_raw` use `HGTXR_E2E_SMALL_MEM_LUTRAM`; RMU/SMU `score/prob` already fixed to LUTRAM |

### P0-00 Result Summary

| Metric | Value |
|---|---:|
| Parsed HLS module reports | 407 |
| Parsed top variants | 11 |
| Reproduced C3b latency cycles | 37,508,072 |
| Reproduced C3b LUT | 126,506 |
| Reproduced C3b DSP | 604 |
| Reproduced C3b BRAM_18K | 332 |
| Reproduced C3b URAM | 64 |

### P0 Execution Profiles Added

| Profile | Command | Expected report | Purpose |
|---|---|---|---|
| `par16_c3b_recheck` | `scripts/run/run_e2e_q4w8a_no_board.sh csynth par16_c3b_recheck` | `generated/hgtxr_e2e_axis_par16_c3b_recheck/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` | Validate wrapper and baseline namespace |
| `par32_dsp_uram_mem16` | `scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_uram_mem16` | `generated/hgtxr_e2e_axis_par32_dsp_uram_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` | First PAR32 exploratory csynth |
| `par32_dsp_uram_mem32` | `scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_uram_mem32` | `generated/hgtxr_e2e_axis_par32_dsp_uram_mem32_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` | PAR32 plus 32-way memory banking |
| `par32_dsp_mixed_stream_mem16` | `scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_mixed_stream_mem16` | `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` | Q/K/V/attention URAM only; current best csynth-fit candidate |
| `par32_dsp_mixed_mem32` | `scripts/run/run_e2e_q4w8a_no_board.sh csynth par32_dsp_mixed_mem32` | `generated/hgtxr_e2e_axis_par32_dsp_mixed_mem32_no_board/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt` | Selective Q/K/V/attention/hidden URAM placement |

### P1-PAR32-CSYNTH Launch Note

| Field | Value |
|---|---|
| Launch sessions | `hgtxr_par32_p0_csynth_20260626`, `hgtxr_par32_mixed_p0_csynth_20260626`, `hgtxr_par32_stream_p0_csynth_20260626` |
| Commands | `par32_dsp_uram_mem16`, `par32_dsp_mixed_mem16`, `par32_dsp_mixed_stream_mem16` via `scripts/run/run_e2e_q4w8a_no_board.sh csynth` |
| Projects | `hgtxr_e2e_axis_par32_dsp_uram_mem16_no_board`, `hgtxr_e2e_axis_par32_dsp_mixed_mem16_no_board`, `hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board` |
| Initial blocker fixed | Vitis HLS needed `libtinfo.so.5`; wrapper now prepends `/tools/Xilinx/Vitis_HLS/2023.2/lib/lnx64.o/Rhel/9` to `LD_LIBRARY_PATH` |
| Current evidence | `python3 scripts/report/collect_no_board_reports.py` parsed `515` HLS module reports and `14` top variants after PAR32 runs |
| Completion action | Run no-board route/timing check for `par32_dsp_mixed_stream_mem16` before any promotion |

### P1-PAR32-ROUTE Continuation

| Item | Status | Evidence / Command |
|---|---|---|
| HLS IP package | completed | `generated/hgtxr_e2e_axis_par32_dsp_mixed_stream_mem16_no_board/solution_e2e_q4w8a/impl/ip/component.xml` |
| package output gate | passed | packaged HLS IP exists |
| route wrapper | ready | `scripts/run/run_e2e_axis_dma_vivado_no_board.sh par32_dsp_mixed_stream_mem16` |
| route result | passed | WNS `3.885 ns`, TNS `0.000 ns`, route errors `0`, bitgen completed |
| current detail note | updated | `analysis/no-board-results/VIVADO_PROGRESS_2026_06_27.md` |

### P1-PAR32-ROUTE Result Matrix

| Item | Value |
|---|---:|
| Routed candidate | `par32_dsp_mixed_stream_mem16` |
| HLS latency cycles | 26,139,644 |
| HLS latency @ 5 ns | 130.698 ms |
| HLS DSP | 1,148 / 1,728, 66.44% |
| HLS LUT | 193,546 / 230,400, 84.00% |
| HLS BRAM_18K | 332 / 624, 53.21% |
| HLS URAM | 64 / 96, 66.67% |
| Routed WNS | 3.885 ns |
| Routed TNS | 0.000 ns |
| Routing errors | 0 |
| Bitstream | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.bit` |
| Handoff | `generated/build/vivado/overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16_overlay/hgtxr_e2e_axis_dma_par32_dsp_mixed_stream_mem16.hwh` |

### Mode-Latency Measurement Gap

| Target | Current status | Required next instrumentation |
|---|---|---|
| Search mode <= 4 ms | HLS/Vivado mode-profile proven: `772,268` cycles / `3.861 ms @ 5 ns`, routed WNS `2.638 ns`, route errors `0`, bitgen pass | URAM margin reduction remains desirable |
| Track mode <= 1 ms | HLS/Vivado mode-profile proven: `99,449` cycles / `0.497 ms @ 5 ns`, routed WNS `1.481 ns`, route errors `0`, bitgen pass | Optional runtime mode counters later |
| Search/Track distribution | unavailable in no-board flow | synthetic mode labels or runtime counters |

### NB-P0-MODE-PROFILE Result

| Mode | Profile | Top | Worst cycles | Latency @ 5 ns | Target | Target met | HLS URAM | Routed WNS | Route errors | Status |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| Search | `search_par32` | `hgtxr_search_profile_top` | 772,268 | 3.861 ms | 4.000 ms | yes | 96 | 2.638 ns | 0 | HLS/IP/Vivado route pass |
| Track | `track_par32` | `hgtxr_track_profile_top` | 99,449 | 0.497 ms | 1.000 ms | yes | 64 | 1.481 ns | 0 | HLS/IP/Vivado route pass |

Evidence:

- `analysis/no-board-results/MODE_PROFILE_RESULTS_2026_06_27.md`
- `analysis/no-board-results/p0_mode_profile_2026_06_27/summary.md`
- `generated/hgtxr_mode_search_par32_no_board/solution_mode_q4w8a/impl/ip/component.xml`
- `generated/hgtxr_mode_track_par32_no_board/solution_mode_q4w8a/impl/ip/component.xml`
- `generated/build/vivado/overlay/hgtxr_mode_search_par32_overlay/reports/hgtxr_mode_search_par32_timing_summary.rpt`
- `generated/build/vivado/overlay/hgtxr_mode_track_par32_overlay/reports/hgtxr_mode_track_par32_timing_summary.rpt`

### P1-PAR32-CSYNTH Result Matrix

| Variant | Latency cycles | LUT | DSP | BRAM_18K | URAM | Decision |
|---|---:|---:|---:|---:|---:|---|
| C3b `par16_c3b_mem16` | 37,508,072 | 126,506 | 604 | 332 | 64 | protected baseline |
| `par32_dsp_uram_mem16` | 26,139,644 | 193,546 | 1,148 | 124 | 160 | reject: URAM exceeds 96 |
| `par32_dsp_mixed_mem16` | 26,139,644 | 193,546 | 1,148 | 204 | 112 | reject: URAM exceeds 96 |
| `par32_dsp_mixed_stream_mem16` | 26,139,644 | 193,546 | 1,148 | 332 | 64 | keep as csynth-fit candidate; LUT 84% route risk |

### P0 Agent Task Cards Executed

```yaml
task_card:
  task_id: P0-A
  sub_agent: gpt5.3-codex-spark
  role: analyst
  objective: Confirm PAR32 no-board csim/csynth commands and output paths.
  file_ownership: [read-only]
  assigned_skill: [caveman, algo2fpga]
  outputs:
    - PAR32 env-variable matrix
    - generated report path convention
  validation:
    - Existing Tcl accepts PAR values 16 and 32.
    - Project names are suffix-isolated.
```

```yaml
task_card:
  task_id: P0-B
  sub_agent: gpt5.3-codex-spark
  role: analyst
  objective: Audit dense DSP-bound multiply path.
  file_ownership: [read-only]
  assigned_skill: [caveman, accelerator-cost-model]
  outputs:
    - QKV/attention/WO/MLP/head DSP helper map
    - PAR16 vs PAR32 comparison points
  validation:
    - DSP helpers use `#pragma HLS bind_op ... impl=dsp` when enabled.
```

```yaml
task_card:
  task_id: P0-C
  sub_agent: gpt5.3-codex-spark
  role: analyst
  objective: Audit URAM/LUTRAM binding candidates.
  file_ownership: [read-only]
  assigned_skill: [caveman, algo2fpga]
  outputs:
    - Large-buffer URAM candidate map
    - Small-buffer LUTRAM candidate map
  validation:
    - Named buffers map to existing macros or fixed pragmas.
```

```yaml
task_card:
  task_id: NB-T1
  role: implementer
  objective: Create QKV/MLP parallelism HLS variants and run CSim/csynth only.
  file_ownership:
    - hls/src/
    - hls/scripts/
    - generated/no_board/
  validation:
    - CSim pass.
    - csynth report exists.
    - RMU/SMU/MHA/MLP resource table generated.
```

```yaml
task_card:
  task_id: NB-T2
  role: analyst
  objective: Create quantization and nonlinear operator calibration report.
  file_ownership:
    - analysis/no-board-results/
  validation:
    - scale manifest exists.
    - deterministic vector report includes mismatch and error distribution.
```
