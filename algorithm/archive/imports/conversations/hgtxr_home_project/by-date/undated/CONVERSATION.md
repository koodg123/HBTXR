# Conversation Summary

The user requested implementation of `HGTXR` under `/home/user/project/PRJXR`
with both algorithm and hardware code. The package directory should be named
`software` rather than `hgtxr`.



## 2026-06-10 Software Paper-Reproduction Documentation

The user asked to document all work performed so far. Created docs/Software-Paper-Reproduction-Status-2026-06-10.md as the canonical status document. The document states that structure/procedure reproduction is largely implemented, algorithm contract reproduction is partial due to a canonical v3 encoder depth-routing bug, and exact numeric paper reproduction is blocked by missing EV-Eye manifests, final checkpoints, quantization tables, LUTs, golden vectors, and board-level evidence.

## 2026-06-10 Continuation Handover Request

User requested that the received sub-agent outputs be documented and saved, and that progress, remaining plans, and a handover document be created so work can continue on another computer. The saved handover is docs/track/HANDOVER-2026-06-10-E2E.md, with the latest pointer at docs/track/HANDOVER.md.

## 2026-06-10 Next-Direction Choice Rule

After the full active196_b6_ff768 PAR8 QKV-cache point passed CSim/CSynth, the user requested explicit choice whenever multiple directions exist. Added docs/track/NEXT-DECISION-2026-06-10-E2E.md with options for board implementation/timing, HLS timing pre-tune, further parallelism, ZCU104/PYNQ hardware I/O validation, paper-trained weight/LUT alignment, and software paper-reproduction fixes. Do not start long Vivado implementation, broad HLS retune, board runtime validation, or default-path paper-math promotion without user selection.

## 2026-06-10 Option A Preflight Choice

Option A board implementation/timing is not a single command yet. Current scripts still package/build old `hgtxr_top`, while the selected E2E point is `hgtxr_e2e_axis_top`. Spark was attempted for the A1 audit but quota was exhausted until 2026-06-15 23:18, so GPT5.5 sidecars audited A1 and A2. A1 means new E2E AXIS/DMA BD and PYNQ stream path; A2 means a new `hgtxr_e2e_m_axi_top(frame, weights, out_state, runtime_state)` memory-mapped wrapper with fresh CSim/CSynth; A3 means in-place flow replacement with higher regression risk. Ask the user to choose before implementation edits.
