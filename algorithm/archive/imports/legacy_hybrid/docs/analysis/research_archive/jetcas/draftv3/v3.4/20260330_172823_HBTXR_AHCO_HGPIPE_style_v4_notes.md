# HBTXR v4 revision notes

This revision reflects the requested paper-level restructuring and style updates.

## What changed

- Reworked tables so they remain column-safe and use cleaner IEEE/AHCO-style layouts.
  - Switched the dense key-value and comparison tables to `tabularx`-based layouts.
  - Removed the previous awkward hardware-status table structure and replaced it with a cleaner runtime summary table.
  - Kept wide comparison tables as `table*` so they no longer spill across columns.

- Reworked equations so they do not overflow the two-column format.
  - Split long loss equations into Stage-1 and Stage-2 objectives.
  - Broke scheduler equations and architecture equations into multi-line `align` forms.
  - Verified the compiled PDF has no overfull-box warnings from tables/equations.

- Rewrote the hardware section using the requested term system.
  - Removed all explicit HG-PIPE mentions from the body text.
  - Reframed the hardware narrative around:
    - Multi-stage Pipeline
    - Streaming
    - Cross-layer Coupling
    - Intra-layer Dataflow
  - Preserved the underlying hardware intuition while changing the wording and structure to match HBTXR-specific semantics.

- Expanded the model architecture explanation with equations.
  - Added front-end patch embedding equations.
  - Added transformer backbone equations for MHA/MLP.
  - Added back-end head equations.
  - Explicitly stated that Search uses the full backbone and Track uses the first half of the backbone before early exit to the Track head.

- Updated the scheduler and loss design.
  - Search/Track behavior now explicitly reflects full-depth vs. half-depth inference.
  - Loss is split into `L_stage1` and `L_stage2`.
  - Added geometry-aware loss and slimming regularizer expressions.

## Constraints preserved

- Did **not** fabricate post-P&R resource numbers or board-power measurements.
- Kept verified latency / update-rate / accuracy values from the previous draft.

## Output status

- PDF page count: 10 pages
- Body text contains no explicit `HG-PIPE` mention
