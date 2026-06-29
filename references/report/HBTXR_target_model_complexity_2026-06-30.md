# HBTXR Target Model Complexity

Scope: HBTXR subject-independent 64x64 comparison setting.

| Target | Input Shape | Params | Trainable Params | MACs | FLOPs | Status | Note |
|---|---:|---:|---:|---:|---:|---|---|
| HBTXR | 1x2x64x64 | 4368393 | 4368393 | 1111665456 | 2223330912 | ok |  |
| EPNet_FECET | 1x2x64x64 | 3898280 | 3898280 | 215312088 | 430624176 | ok |  |
| FACET_TennSt | 1x2x50x64x64 | 808771 | 808771 | 922099200 | 1844198400 | ok |  |
| Retina | 1x2x64x64 | 59572 | 59572 | 21492072 | 42984144 | ok |  |
| TDTracker | 1x100x2x64x64 | 3246880 | 3246880 | 23269558144 | 46539116288 | ok |  |
| ERVT | 1x30x3x64x64 | 143938 | 143938 | 1387069440 | 2774138880 | ok |  |
| TENNs_Eye | 1x2x50x64x64 | 808771 | 808771 | 922099200 | 1844198400 | ok |  |
| BRAT | 1x30x2x64x64 | 12892898 | 12892898 | 4509919680 | 9019839360 | ok |  |

Notes:
- MACs are measured with `thop.profile` on CPU dummy inputs.
- FLOPs are reported as `2 * MACs` for multiply-add operations.
- `EPNet_FECET` is used because no separate `FECET` model path was found in `references/codebase/software`.
