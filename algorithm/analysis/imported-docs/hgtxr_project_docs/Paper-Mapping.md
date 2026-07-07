# HGTXR Paper Mapping

| Paper Concept | Software File | Hardware File | Status |
|---|---|---|---|
| Event representation | `software/event_repr.py` | `hardware/hls/src/event_patch_embed.cpp` | Initial |
| Frame branch | `software/modules/patch_embed.py` | `hardware/hls/src/frame_patch_embed.cpp` | Initial |
| HG-PIPE backbone | `software/modules/hgpipe_backbone.py` | `hardware/hls/src/attention.cpp`, `hardware/hls/src/mlp.cpp` | Placeholder HLS |
| Search head | `software/heads.py` | `hardware/hls/src/search_head.cpp` | Initial |
| Track head | `software/heads.py` | `hardware/hls/src/track_head.cpp` | Initial |
| Runtime FSM | `software/scheduler.py`, `software/runtime.py` | `hardware/hls/src/runtime_fsm.cpp` | Initial |
| HW/SW validation | `hardware/tools/compare_hw_sw.py` | `hardware/refs/` | Initial |

확실하지 않음: final manuscript section and figure numbering still need to be
locked against `PAPER_PRJXR`.


## 2026-06-04 Implementation Alignment Update

- Event path is now represented as a reduced-depth residual path rather than a full independent ellipse predictor.
- Software model exposes `event/residual` and `track/residual`.
- Runtime state now has anchor memory and validity state.
- Hardware tree now includes Global Buffer, NoC, Weight Prefetcher, Controller, RMU/SMU, and nonlinear stage scaffolds.
- Hardware config now reflects the XR_Accel HBTXR cyclic implementation shape: search `1x128x128`, track `2x64x64`, search depth `8`, track cut depth `4`.

확실하지 않음: exact final paper numerics are still blocked by missing trained checkpoints, quant tables, and golden vectors.
