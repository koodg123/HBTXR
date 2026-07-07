# HGTXR Third-Goal Future Experiments

Date: 2026-06-16

## Priority Policy

Default final signoff remains tied to:

1. C3b AXIS/DMA physical smoke.
2. XR-VITs exact source or approved replacement policy.

QKV URAM and VREF successors are optional promotion paths unless explicitly selected as hard gates.

## P0 Experiments - Required To Close Third Goal

| ID | Experiment | Goal | Inputs | Expected Result | Risk | Status |
|---|---|---|---|---|---|---|
| P0-C3B-SMOKE | Run C3b E2E AXIS/DMA smoke on ZCU104 | Prove default E2E board execution | ZCU104, C3b bundle, bit/hwh, packed weights | Canonical JSON validates and clears C3b blocker | Board access and result copyback required | ready-for-board |
| P0-XRVITS-EXACT | Restore exact `/home/kjm26/project/PRJXR/XR-VITs` | Clear Req11 with strongest provenance | Exact source checkout | XR-VITs gate clears without replacement policy | Source may be unavailable | pending-external |
| P0-XRVITS-POLICY | Approve XR_Accel replacement policy | Clear Req11 with explicit candidate-bound replacement | User approval metadata, `XR_Accel`, candidate audit | Active policy validates and clears XR-VITs blocker | Weaker than exact source; must avoid placeholder approver | candidate-ready-needs-approval |
| P0-FINAL-RUNNER | Rerun final signoff after unblock | Prove final third-goal pass | C3b JSON plus XR-VITs exact source or policy | Final runner status `pass` | Any stale evidence or wrong smoke JSON path fails | waiting-on-P0 inputs |

## P1 Experiments - Recommended Successor Validation

| ID | Experiment | Goal | Inputs | Expected Result | Risk | Status |
|---|---|---|---|---|---|---|
| P1-QKV-SMOKE | Run QKV URAM successor board smoke | Validate VREF-P0-02 physical behavior | QKV URAM bit/hwh and smoke session | JSON validates under `axis-vref-p0-softmax-input-x2-qkv-uram` | Optional; does not clear default signoff unless selected | ready-for-board |
| P1-VREF-SMOKE | Run VREF-P0-01 `softmax_input_x2` board smoke | Validate DSP mixed-stream successor | VREF bit/hwh and smoke session | JSON validates; successor promotion gate clears | Optional; no default replacement without explicit decision | ready-for-board |
| P1-QKV-PROMOTE | Evaluate QKV URAM as promoted path | Decide whether QKV URAM should become a formal successor | QKV physical smoke, existing route/HLS evidence | Promotion packet with latency/resource/board proof | Must not replace C3b without explicit choice | pending-P1-QKV-SMOKE |
| P1-PAR32-CSYNTH | Run PAR32 exploratory csynth | Test higher parallelism beyond validated PAR16 | PAR32 sweep config | Resource/latency data for PAR32 | Could exceed LUT/DSP/route limits | planned |
| P1-PAR32-ROUTE | Route viable PAR32 candidate | Determine whether PAR32 can beat C3b safely | PAR32 csynth pass | WNS/resource fit evidence | Build time and routing risk | depends-on-P1-PAR32-CSYNTH |

## P2 Experiments - Accuracy And Calibration

| ID | Experiment | Goal | Inputs | Expected Result | Risk | Status |
|---|---|---|---|---|---|---|
| P2-SCALE-CALIB | Extend P2-ViT scale calibration beyond current PoT sweep | Improve accuracy proxy without breaking C3b | Existing PoT sweep, SW golden model | Candidate scale set with accuracy/resource tradeoff | Needs regeneration and CSim before promotion | planned |
| P2-Q4Q8-COVERAGE | Add more Q4/Q8 SW-HW test vectors | Strengthen exact-match confidence | Additional deterministic inputs | More TB/CSim coverage, same expected semantics | Larger test runtime | planned |
| P2-OPERATOR-STRESS | Stress LayerNorm/GeLU/Softmax/Quantization properties | Find numerical corner cases | HG-PIPE operator references | Expanded property audit beyond `211/211` | May expose precision tradeoffs | planned |
| P2-DATASET-PROXY | Add small DeiT-like image/feature proxy tests | Better link to PAPER_PRJXR image reference | Existing image/reference artifacts | Traceable accuracy proxy | Not full dataset accuracy | planned |

## P3 Experiments - Architecture Exploration

| ID | Experiment | Goal | Inputs | Expected Result | Risk | Status |
|---|---|---|---|---|---|---|
| P3-URAM-BANKING | Explore more URAM banking layouts | Reduce BRAM/LUT pressure while preserving timing | C3b/QKV storage macros | New resource matrix rows | Can increase routing pressure | planned |
| P3-LUTRAM-SMALLBUF | Sweep LUTRAM thresholds for small buffers | Reduce LUT pressure from accidental distributed compute/storage | RMU/SMU buffers and cyclic scratch | Better resource-policy recommendations | HLS binding may vary by version | planned |
| P3-DSP-DENSITY | Raise DSP utilization in dense paths | Improve throughput and avoid LUT multipliers | DSP-bound helpers, PAR16/PAR32 configs | Higher DSP use with bounded LUT | Resource or route pressure | planned |
| P3-MEM16-PLUS | Explore memory-bank count above 16 | Test bank conflict reduction | C3b memory-bank refinement | Lower latency or same latency with better timing | More BRAM/control cost | planned |

## P4 Experiments - Negative Controls

| ID | Experiment | Goal | Inputs | Expected Result | Risk | Status |
|---|---|---|---|---|---|---|
| P4-LUT-HEAVY | Quantify LUT-heavy GEMM variants | Show why current DSP policy is preferred | LUT-heavy candidate kernels | Evidence that LUT pressure regresses | Could consume time without promotion value | optional |
| P4-TERNARY | Evaluate ternary/low-bit negative control | Bound accuracy/resource tradeoff | Ternary candidate model/kernels | Rejection or narrow use case | Accuracy likely regresses | optional |
| P4-NONCURRENT-SCALE | Run non-current PoT scale rows as explicit successors | Prove or reject scale alternatives | Sweep rows not recommending current | Successor evidence only | Must not affect default C3b | optional |

## Execution Order

1. P0-C3B-SMOKE.
2. P0-XRVITS-EXACT or P0-XRVITS-POLICY.
3. P0-FINAL-RUNNER.
4. P1-QKV-SMOKE and P1-VREF-SMOKE.
5. P1-PAR32-CSYNTH.
6. P2/P3 only after default signoff or explicit scope decision.

## Promotion Rules

- Do not mark C3b complete without canonical board-produced JSON.
- Do not accept smoke JSON with wrong bit/hwh basenames.
- Do not mark Req11 complete without exact source or approved fingerprint-bound replacement policy.
- Do not promote QKV URAM as default final-signoff blocker unless Q3 is explicitly selected in `docs/CHOICE.md`.
- Do not promote PAR32 without fresh csynth, routed timing, resource-fit, and no-C3b-overwrite evidence.

