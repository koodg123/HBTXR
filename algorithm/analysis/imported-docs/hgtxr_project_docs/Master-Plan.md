# HGTXR Hardware Master Plan

Date: 2026-06-16

## Goal
Integrate ViT accelerator paper/codebase findings into the HGTXR third-goal hardware plan without replacing the selected C3b board-smoke path.

## Current Third-Goal State
- Selected path: A2 -> A1 and C path, with C3b as current board-smoke candidate.
- Spec-plan anchor: Path 1 = A2 then A1; Path 2 = C; E pending.
- Live evidence anchor: final manifest `86/86`, consistency checks `347`, spec-plan conformance `86/86`, operator handoff validation `131/131`, bundle validation `152/152`, source audit required `86`, sources `224`, current audit reflected `10`, partial `1`, blocked `1`.
- Current blocker: C3b AXIS/DMA physical smoke result is not captured.
- Policy blocker: exact XR-VITs source or approved replacement policy remains unresolved in the broader HGTXR signoff context.
- Operator unblock readiness: final blocker closure readiness now emits a no-side-effect `operator_unblock_plan` for C3b board JSON and XR-VITs source/policy inputs; final evidence gates the plan and cross-checks final unblock intake `next_inputs`/`operator_sequence` against it.
- XR-VITs policy guard: active replacement-policy writes require a real approver identifier; placeholder approvers are allowed only for dry-run preview.
- Resource direction: increase DSP/URAM utilization, keep large buffers in URAM, keep small tables/FIFOs in LUTRAM or BRAM, avoid LUT-heavy compute mappings. Latest RMU/SMU relation and projection multiply paths use DSP-bound helper functions and are final-evidence gated through the E2E resource-policy audit. The legacy cyclic baseline now also has macro-controlled URAM binding for packed weight tiles and large temporaries, plus LUTRAM binding for small tile scratch. C3b csynth LUT/latency and routed WNS thresholds are also resource-policy gated. Req5 Q4/Q8 packed-weight evidence gates binary file existence, SHA256, byte count, expected runtime state, and C3b raw output.
- Latest successor evidence: VREF-P0-02 QKV weight-cache URAM successor passes HLS CSim/csynth, C3b HLS thresholds, HLS IP package, and routed overlay timing. PYNQ bundle/session artifacts are ready for the QKV variant, but it remains successor-only until physical smoke evidence exists.

## Reference Inputs
- `analysis/vit-accel/README.md`
- `analysis/vit-accel/experiment_extensions_2026_06_15.md`
- `analysis/vit-accel/third_goal_integration_2026_06_15.md`
- `docs/legacy/legacy_experiment_analysis_2026_06_12.md`
- `configs/sweeps/zcu104_cyclic_transformer_sweep.yaml`
- `generated/signoff/third_goal_requirements_trace_2026_06_10.md`
- `generated/signoff/c3b_protection_checklist_2026_06_16.md`

## Team Blueprint
- Master: maintain third-goal scope and non-regression gates.
- Hardware resource lead: preserve C3b, DSP mapping, URAM/LUTRAM memory policy.
- Quantization lead: convert P2-ViT PoT scale ideas into SW-first Q4/Q8 experiments.
- Memory/dataflow lead: convert ME-ViT/HG-PIPE findings into buffer lifetime and FIFO implementation audits.
- Evaluator: require exact-match, csynth, routed timing/power, and PYNQ smoke evidence before signoff promotion.

## Execution DAG
1. Keep C3b board-smoke as immediate unblock path.
2. Add reference-derived VREF experiments to third-goal plan and sweep config.
3. Run P0 software/audit tasks while waiting for board evidence.
4. Keep VREF-P0-02 QKV URAM as HLS-resource/IP/routed-pass successor evidence with physical smoke pending.
5. Promote only bounded successor variants after C3b smoke is captured and the C3b protection checklist passes.
6. Keep P1/P2 attention/MoE/model-structure ideas as software-first ablations until paper-scope review approves them.

## Priority Order
1. `C3b`: physical ZCU104 AXIS/DMA smoke result.
2. `VREF-P0-01`: P2-ViT PoT scale calibration for Q4/Q8.
3. `VREF-P0-02`: ME-ViT single-load/on-chip buffer audit for URAM/global buffers.
4. `VREF-P1-03`: TATAA mixed precision only for nonlinear bottlenecks.
5. `VREF-P1-01`, `VREF-P1-02`: search/track expert or attention approximation ablations, software-first.
6. `VREF-P2/P3`: tooling and negative controls.

## Non-Regression Gates
- C3b evidence must not be overwritten.
- C3b/VREF/QKV board-smoke JSON must report bit/hwh basenames matching the selected preset before import or discovery can clear a gate.
- VREF successor promotion must preserve C3b thresholds: latency `<= 37508072`, WNS `>= 4.415 ns`, DSP `<= 604`, LUT `<= 126506`, URAM `<= 64`.
- HLS hardware promotion requires `csynth.xml` or equivalent csynth report.
- Final resource claims require routed timing/power/utilization.
- Final third-goal signoff requires PYNQ board-smoke JSON.
- Active XR-VITs replacement-policy creation must reject placeholder approver metadata and preserve dry-run preview as no-side-effect review evidence.
- Exact attention or paper-defined behavior replacement requires explicit scope decision.
