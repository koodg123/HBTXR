# HGTXR HLS-To-PYNQ QA Gates

Date: 2026-06-05
Scope: HLS, IP export, Vivado ZCU104 implementation, PYNQ runtime, paper-reproduction direction
Source: GPT-5.3-Codex-Spark QA sidecar, integrated by main agent

## Primary Goals

- P1: Reproduce paper-relevant HGTXR behavior with functional and dataflow correctness.
- P2: Deliver a runnable ZCU104/PYNQ implementation with a stable driver flow.
- P3: Reproduce paper-reported metrics or document justified deltas.

## Secondary Goals

- S1: Keep the build and integration process reproducible and auditable.
- S2: Cover regression and failure-safety cases such as reset, reload, timeout, and repeated runs.

## Gate 0: Baseline Lock

Required evidence:

- Target model/config.
- Precision policy.
- Dataset or fixed-vector input policy.
- Preprocessing order.
- Batch and tensor shapes.
- Paper metric targets and tolerance.
- Explicit list of missing paper details.

Status levels:

- Complete: baseline spec sheet exists and all missing details are classified.
- Partial: shape and precision exist but paper mapping is incomplete.
- Unverified: no baseline spec.

## Gate 1: HLS Functional Correctness

Required evidence:

- HLS C simulation logs.
- RTL cosimulation logs, or a documented reason it is unavailable.
- Fixed input/output vectors.
- Repeated deterministic output check.

Status levels:

- Complete: zero mismatch on smoke vectors and deterministic repeated runs.
- Partial: C simulation only, or only partial datapath covered.
- Unverified: no simulation artifact.

## Gate 2: HLS Implementation Quality

Required evidence:

- `csynth` report.
- Interface summary.
- Latency and II report.
- Resource report.
- Warning/error snapshot.

Status levels:

- Complete: csynth succeeds, no high-severity warning, interface is correct.
- Partial: csynth succeeds but warning/timing risks remain.
- Unverified: no current report.

## Gate 3: IP Packaging And Integration Contract

Required evidence:

- `component.xml`.
- AXI register map.
- IP VLNV.
- Port/bundle list.
- Reset/clock contract.

Status levels:

- Complete: IP package is deterministic and register ABI matches the driver.
- Partial: data ports match but control/reset contract is incomplete.
- Unverified: no package evidence.

## Gate 4: Vivado Synthesis And Implementation

Required evidence:

- BD validation log.
- Synthesis log.
- Implementation log.
- Utilization report.
- Timing report.
- Bitstream generation log.

Status levels:

- Complete: bitstream succeeds and timing/resource gates pass.
- Partial: bitstream succeeds but timing/resource gaps remain.
- Unverified: no implementation reports.

## Gate 5: Bitstream And PYNQ Runtime

Required evidence:

- `.bit`.
- `.hwh`.
- PYNQ overlay load log.
- Register transaction log.
- Buffer allocation log.
- One-batch smoke output.
- Repeated run and reset/reload check.

Status levels:

- Complete: board flow runs repeatedly without deadlock and returns known-good output.
- Partial: overlay loads or one-shot runs, but repeated/reset behavior is not proven.
- Unverified: no live board validation.

## Gate 6: Paper-Reproduction Benchmark

Required evidence:

- Benchmark protocol.
- Raw metric captures.
- Comparison table against paper values.
- Gap justification log.

Status levels:

- Complete: all metric families are reproduced within tolerance.
- Partial: some metrics are reproduced, with unresolved assumptions.
- Unverified: benchmark not run.

## Gate 7: Gap And Exception Audit

Each gap must be classified as one of:

- Resolved.
- Deferred with justification.
- Not tested.

Complete only when no high-impact gap remains without owner and next action.

## Gate 8: Regression And Documentation

Required evidence:

- Smoke regression tests.
- Rebuild commands.
- Toolchain version pins.
- Artifact manifest.
- Environment and report hash log.

Status levels:

- Complete: smoke tests pass and rebuild path is reproducible.
- Partial: hardware or software is covered, but not both.
- Unverified: no repeatability artifact.

## Completion Rule

Do not claim the full objective complete until Gates 0 through 8 have current evidence. A generated bitstream alone proves only part of Gate 4. A Python driver unit test alone proves only part of Gate 5.

