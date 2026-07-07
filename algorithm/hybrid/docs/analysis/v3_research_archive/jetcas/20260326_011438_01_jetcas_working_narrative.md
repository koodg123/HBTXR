# JETCAS Working Narrative For HBTXR_v3_0

Last updated: 2026-03-25 KST

Role: `JETCAS Paper Architect`, `XR System Architect`, `ML/Vision Research Engineer`

## Summary

- This document is the active paper-oriented narrative for `HBTXR_v3_0`.
- It translates the codebase, reference analysis, and design decisions into a manuscript-ready system story.
- The intended paper position is not a pure event detector and not a generic eye-tracking model. It is a deployment-oriented hybrid eye-tracking system for XR with dense supervision, explicit runtime scheduling, and accelerator-aware structure.

## 1. Target Paper Story

The central paper claim should be framed as follows:

- robust on-device XR eye tracking requires more than a strong event model
- it requires a dense-annotation-aware training contract
- a hybrid frame-event model that separates relocalization and residual tracking
- and a runtime structure that can be scheduled on a host side while keeping the backbone accelerator-friendly

This means the paper should present `HBTXR_v3_0` as an integrated system, not as an isolated model block.

## 2. Problem Framing

The working problem statement is:

- event-only pipelines are efficient but can drift, lose global context, or become fragile under sparse or unstable event conditions
- frame-only relocalization is more stable globally but is not sufficient for low-latency continuous tracking
- offline-heavy pipelines are difficult to justify for deployable XR runtime

Therefore, the system should combine:

- dense supervision from annotated eye-region and pupil geometry
- frame-guided search for relocalization and anchor refresh
- event-guided residual tracking for short-horizon online updates
- quality-aware gating and scheduler control

## 3. System Narrative

### 3.1 Annotation And Canonical Contract

The paper should describe the data path as:

```text
raw session
  -> dense annotation source
  -> canonical session package
  -> manifest row
  -> training sample contract
```

The important paper point is that the project does not treat annotation as loose side data. It converts raw inputs into a canonical truth package containing:

- eye region bounding box
- pupil mask
- pupil ellipse / region
- quality and validity flags
- event stream reference
- previous/current annotation linkage

This should be described as a reproducible supervision contract rather than a dataset convenience layer.

### 3.2 Manifest And Dataloader

The paper should highlight that the manifest is not just a split list. It carries experiment-time provenance:

- split
- ROI
- resize policy
- event window policy
- previous annotation reference
- quality metadata

The dataloader then assembles a unified sample contract:

- `frame`
- `event`
- `prev_state`
- `mask_target`
- `eye_target`
- `pupil_search_target`
- `pupil_track_target`
- `constraint_center`
- quality and validity flags

This is important because the paper contribution is partly in the training contract itself, not only in the transformer.

### 3.3 Hybrid Frame-Event Tracking Model

The model story should be:

- frame branch for search and relocalization
- event branch for event-conditioned estimation
- residual tracking branch conditioned on `prev_state`
- shared backbone with decoupled heads

The paper should explicitly keep the distinction between:

- `search`
- `event`
- `track`

This separation is necessary because the runtime scheduler depends on it, and the deployment story becomes weak if all branches are merged into a single undifferentiated estimator.

### 3.4 Scheduler And Runtime

The scheduler is a paper-level component, not just an implementation detail.

The `Search | Track` story should be written as:

- `Search` mode handles relocalization, anchor refresh, and recovery
- `Track` mode handles event-driven residual update from the previous state
- host-side control uses confidence, similarity, event density, and closed-eye or quality signals

This is one of the main points that differentiates the project from a pure FACET-style event estimation pipeline.

### 3.5 Accelerator And Deployment Direction

The accelerator story should remain realistic:

- keep a shared backbone and simple public ABI
- keep runtime scheduling on the host side
- keep model branches and data contracts explicit
- avoid turning the system into an offline-heavy pipeline

The paper should present this as algorithm-accelerator co-design for deployable XR, not as a finalized hardware benchmark paper unless implementation evidence is available.

## 4. Candidate Core Contributions

The current strongest four contribution lines are:

1. Dense-annotation-aware canonical training contract for frame-event eye tracking
2. End-to-end hybrid frame-event transformer with decoupled search and track heads
3. Two-stage `Search | Track` scheduler with host-side runtime control signals
4. FPGA-oriented algorithm-accelerator co-design direction for on-device XR deployment

These should remain aligned with actual code and should not claim functionality that is only partially wired.

## 5. What The Paper Should Emphasize Against FACET

Relative to `FACET`, the paper should emphasize:

- the project is not dataset-centric; it is contract-centric
- the project is not a pure event-only detector; it is a hybrid tracker
- the project separates canonical truth, manifest provenance, sample assembly, model training, and runtime scheduling
- the project includes previous-state-conditioned tracking and scheduler-oriented deployment logic

`FACET` should be cited as an important source for:

- dense supervision structure
- direct square resize policy
- event accumulation options such as `fixed_count`, `time_bin`, and `causal_linear`

But the paper should be clear that `HBTXR_v3_0` does not simply copy FACET as its system shape.

## 6. Code-Grounded Claims That Are Safe To Make

The following claims are currently well supported by the codebase:

- canonical session packaging exists
- manifest-driven sample assembly exists
- unified multimodal dataloader contract exists
- shared-backbone hybrid search/event/track model exists
- stage-wise training structure exists
- runtime scheduler logic exists at library level

## 7. Claims That Need Careful Wording

The following are directionally correct but should be written carefully until fully validated or fully wired:

- `similarity_target` is available as metadata but is not yet fully active as a training signal
- `track_quality` and quality gating need careful wording if the training target path is incomplete
- runtime FSM exists, but the active CLI path should not be overstated as fully integrated if that path is not yet the default execution surface
- hardware co-design should be presented as architecture direction unless backed by implemented accelerator evidence

## 8. Recommended Writing Stance

For the current manuscript stage:

- write strongly about architecture and contract design
- write carefully about end-to-end runtime activation and hardware realization
- avoid claiming a completed pure production stack if the code still contains partially connected paths
- keep the paper centered on the combined value of annotation contract, hybrid search/track learning, and deployment-oriented scheduler design

## 9. Immediate Writing To-Do

- convert this narrative into paper section language for:
  - Introduction
  - Method overview
  - Dataset and supervision contract
  - Runtime and deployment discussion
- align contribution wording with the active implementation
- keep evidence links synchronized with root `docs/` analysis files

## 10. Maintenance Rule

From now on, `JETCAS`-related notes should follow this rule:

- manuscript framing updates go here
- dated history items go to `02_jetcas_update_log.md`
- if a code change modifies the paper story, update both this file and the dated log
