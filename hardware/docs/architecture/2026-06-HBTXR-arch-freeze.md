> **작성** 2026-06-16 · **갱신** 2026-07-29
> **상태** frozen
> **소유** hardware
>
> **아키텍처 동결.** `archive/hardware/docs/` 에서 복사, 내용 무변경.

# HBTXR Architecture Freeze And DeiT Equivalence Plan

This document defines the next implementation milestone for the cyclic
accelerator work in `XR_Accel`.

The goal of this milestone is:

1. freeze the hardware architecture to match the HBTXR paper, and
2. validate that `DeiT-Tiny` still behaves exactly like the baseline hardware
   on top of that frozen architecture.

This is intentionally an architecture-first milestone. Final HBTXR numerics,
scheduler tuning, and board-level performance work come later.

## Purpose

The repository already contains a working `DeiT-Tiny` cyclic path and a
regression that proves exact-match behavior against the checked-in baseline
references. That path is the current functional anchor.

Before replacing the arithmetic with final HBTXR-specific operators and
weights, the hardware structure itself should be fixed to the paper-level
architecture so that later changes do not mix architectural risk with numeric
model risk.

## Current Functional Anchor

The current executable cyclic implementation is centered on:

- `workspace/hardware/case_cyclic_zcu104/`
- `workspace/hardware/src/cyclic_vit.h`
- `automation/cyclic_top.py`

The current validation anchor is:

- `docs/validation/DEIT_CYCLIC_VERIFICATION.md`

That validation already proves:

- `patch_embed` exact match
- `attn0..attn11` exact match
- `mlp0..mlp11` exact match
- `head` exact match

for the `MODE_DEIT` path against the checked-in baseline references.

## Paper-Derived Architecture To Freeze

The hardware architecture to freeze is the one described in
`../PAPER_WORKS/Submission/main.tex`:

- DRAM
- CPU scheduler
- AXI-connected input and output DMA engines
- PS-PL bus connection
- shared Global Buffer
- on-chip NoC/interconnect
- Patch Embedding Core
- four cyclic compute cores
  - `MHA Core0`
  - `MLP Core0`
  - `MHA Core1`
  - `MLP Core1`
- mode-dependent terminal decode/output routing
- FPGA-side controller

The execution policy to freeze with that structure is:

- `search`
  - frame input
  - full-depth traversal
  - search-oriented terminal decode/output
- `track`
  - event input
  - reduced-depth traversal up to fixed cut point `c`
  - track-oriented terminal decode/output

The memory policy to freeze is:

- track-resident operators and weights stay on chip
- search-specific weights are fetched on demand
- search-weight prefetch overlaps with early execution
- token movement stays on chip across the cyclic compute loop

## Scope Of This Milestone

In scope:

- top-level hardware structure
- Global Buffer ownership and interfaces
- on-chip NoC/interconnect structure
- Weight Prefetcher structure
- controller-visible mode and status behavior
- DMA-facing ingress and egress boundaries
- mode-asymmetric traversal policy
- `DeiT-Tiny` equivalence profile on the frozen architecture

Out of scope:

- final HBTXR quantized parameter export
- final search and track head numerics
- scheduler policy quality tuning
- final resource optimization
- final board-level latency or throughput claims

## Phase 1: Freeze The Architecture Spec

First, the repository must define one canonical architecture spec that later
code changes follow.

Required outputs:

- this document
- updates to `docs/architecture/HBTXR_CYCLIC_IMPLEMENTATION.md`
- reconciled naming between current code paths and future HBTXR-target paths

The spec should fix:

- top-level blocks
- mode-specific dataflow
- cut point `c`
- Global Buffer responsibility
- NoC/interconnect responsibility
- Weight Prefetcher responsibility
- terminal decode/output responsibility
- controller register map and observable status
- on-chip versus off-chip residency for tokens and weights

## Phase 2: Implement The Paper Skeleton

After the spec is fixed, the code should be refactored so the top-level shape
matches the paper even if some blocks are still simple first-working versions.

Implementation target:

- input DMA ingress shim
- Global Buffer
- NoC/interconnect
- Patch Embedding Core
- `MHA0 -> MLP0 -> MHA1 -> MLP1`
- terminal decode/output routing
- output DMA egress shim
- controller

Guidelines for the first implementation:

- keep the NoC/interconnect simple
  - crossbar, mux, or staged FIFO routing is acceptable
- keep the Weight Prefetcher simple
  - correctness first, overlap tuning later
- do not change the `DeiT-Tiny` reference numerics in this phase

## Phase 3: Build A DeiT Profile On The Frozen Architecture

Once the structure matches the paper, the next step is to run `DeiT-Tiny`
through that frozen architecture.

This profile is an architecture-validation profile, not the final HBTXR model.

Rules:

- preserve the baseline `DeiT-Tiny` patch embedding
- preserve the baseline `DeiT-Tiny` backbone numerics
- preserve the baseline `DeiT-Tiny` head
- use the frozen paper-style top-level structure around those numerics
- keep `MODE_DEIT` as the exact-match reference path

Expected result:

- any mismatch can then be attributed to architectural changes such as routing,
  buffering, prefetch timing, or controller behavior, not to model changes

## Phase 4: Prove DeiT Equivalence

The frozen architecture is accepted only if `DeiT-Tiny` remains exactly equal
to the baseline hardware.

Required checks:

- `patch_embed` exact match
- `attn0..attn11` exact match
- `mlp0..mlp11` exact match
- `head` exact match
- deterministic repeated passes

Additional architecture-aware checks should also be added for:

- token ordering through the NoC/interconnect
- stale-weight use after prefetch
- buffer-bank ownership in ping-pong or double-buffer structures
- exact output word count and packet termination
- correct mode-controlled source and sink selection

Minimum regression standard:

- two repeated `MODE_DEIT` passes with identical exact-match markers
- no architecture-induced mismatch

## Phase 5: Transition To Final HBTXR Numerics

Only after the frozen architecture passes the `DeiT-Tiny` equivalence gate
should the implementation switch from baseline numerics to final HBTXR-specific
numerics.

That later phase will add:

- HBTXR export format
- HBTXR search and track weights
- HBTXR LUTs and scales
- search and track terminal goldens
- final runtime mode behavior

The architecture should remain fixed while those numeric updates are applied.

## Milestone Completion Criteria

This milestone is complete when all of the following are true:

- the top-level cyclic hardware structure matches the paper-level architecture
- Global Buffer, NoC/interconnect, Weight Prefetcher, and controller all exist
  as real modules or clearly bounded implementation blocks
- `DeiT-Tiny` runs on that frozen architecture
- `DeiT-Tiny` remains bit-exact against the baseline references
- repeated runs are deterministic
- documentation matches the implementation naming and boundaries

## Recommended Immediate Work Order

1. Freeze the architecture spec in docs.
2. Reconcile current code paths with target HBTXR naming.
3. Refactor the top-level skeleton to the paper structure.
4. Add a first-working NoC/interconnect.
5. Add a first-working Weight Prefetcher.
6. Run `DeiT-Tiny` through the frozen architecture.
7. Restore full exact-match regression coverage.
8. Only then start the final HBTXR numeric transition.
