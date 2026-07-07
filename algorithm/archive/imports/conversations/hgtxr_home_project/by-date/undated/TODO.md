# TODO

- Lock paper mapping against final manuscript.
- Add real manifest examples from EV-Eye canonical data.
- Add fixed-point calibration.
- Add per-layer HW/SW numerical comparison.
- Add board-specific HLS pragmas.



## Software Paper-Reproduction TODO - 2026-06-10

- [ ] Fix software/src/hbtxr/models/tracker/encoder.py so encode_tokens uses its depth_limit argument.
- [ ] Add a regression test proving search uses depth 8 and track uses depth 4 for the paper config.
- [ ] Fix paper.submitted_version in software/configs/paper/zcu104_option_a.yaml.
- [ ] Create or import EV-Eye train/val/test manifests under software/data/_internal/manifests.
- [ ] Materialize HGTXR-local paths for Grounded-SAM, TimeLens, v2e, canonical data, annotations, and manifests.
- [ ] Import or train final paper checkpoints and export quantization tables, nonlinear LUTs, and golden vectors.
- [ ] Run final paper metric and ZCU104 validation before claiming exact reproduction.

## E2E Q4W/Q8A Continuation TODO - 2026-06-10

- [x] T-E2E-DATA-DRIVEN-001: unify reduced E2E pattern/golden generation between Python and C++.
- [x] T-E2E-DENSE-REDUCED-002: add denser deterministic Q4 reduced SW/HW gate with more nonzero QKV/WO/W1/W2/head taps.
- [x] T-E2E-FULL-SCALE-003: strict active_tokens=196, blocks=6 E2E equivalence, csim/csynth, and resource extraction completed for active196_b6_ff768.
- [x] Tighten pytest/native regression so every output lane is checked for reduced, active8, and active16 gates.

- [x] Raise E2E staged golden specs through active_tokens/patch_grid to active16.
- [x] Raise E2E staged golden specs through final ff_dim=768 at active16 with high hidden-index taps.
- [x] Raise E2E staged golden specs to active_tokens=32 with ff_dim=768.
- [x] Raise E2E staged golden specs to active_tokens=64 with ff_dim=768.
- [x] Raise E2E staged golden specs to active_tokens=128 with ff_dim=768.
- [x] Raise E2E staged golden specs to full active_tokens=196 with ff_dim=768.
- [x] Raise E2E staged golden specs to blocks=6 full numerical equivalence for active196_b6_ff768.

- [ ] Replace deterministic arbitrary Q4 gate and compact math contract with final paper-trained packed weights, calibration tables, and final nonlinear LUTs.
- [ ] Run board-side ZCU104/PYNQ validation for the E2E Q4W/Q8A accelerator package.
- [x] Optimize packed-weight traffic to remove the full PAR8 QKV AXI weight-port `II=3` limit while preserving the selected `dsp_mixed_stream` resource fit.
- [ ] Choose next direction from `docs/track/NEXT-DECISION-2026-06-10-E2E.md` before major execution.
- [ ] Option A: run board-side implementation/timing for full `active196_b6_ff768`, `PAR=8`, `HGTXR_E2E_RESOURCE_POLICY=dsp_mixed_stream`, `HGTXR_E2E_QKV_WEIGHT_CACHE=1`.
- [ ] Before Option A implementation, choose A1 AXIS+DMA, A2 memory-mapped wrapper, or A3 in-place flow replacement from `docs/track/OPTION-A-PREFLIGHT-2026-06-10-E2E-BITSTREAM.md`.
- [ ] Option B: if implementation timing risk should be reduced first, tune QKV cache banking, attention MAC pipeline depth, or clock target before bitstream packaging.

- [x] T-HGPIPE-MATH-CONTRACT-004: add machine-readable compact HG-PIPE nonlinear/quant contract and validation artifact.
- [x] T-HGPIPE-GELUQ64-CURSOR-005: add isolated HG-PIPE `mlp_1_geluq` cursor-table primitive and full-ref validation over 150,528 samples.
- [x] T-HGPIPE-GELUQ-MLP0-005A: add isolated HG-PIPE `mlp_0_geluq` cursor-table primitive and full-ref validation over 150,528 samples.
- [x] T-HGPIPE-GELUQ-MLP10-11-005B: add isolated HG-PIPE `mlp_10_geluq` and `mlp_11_geluq` cursor-table primitives and full-ref validation over 301,056 samples.
- [x] T-HGPIPE-GELUQ-MLP2-9-005C: add isolated HG-PIPE `mlp_2..9_geluq` cursor-table primitives and full-ref validation over 1,204,224 samples.
- [x] T-HGPIPE-GELUQ-E2E-SCAFFOLD-017: add opt-in default-E2E MLP GeLUQ wiring scaffold with matching SW/HW reduced golden validation.
- [ ] T-HGPIPE-GELUQ64-E2E-006: decide whether to wire GeLUQ64 into default E2E; if yes, update SW mirror, regenerate affected golden specs/headers, rerun native/Vitis csim, then csynth.
- [x] T-HGPIPE-QUANT-ATTN0Q-CURSOR-007: add isolated HG-PIPE `attn_0_q_q` quant cursor-table primitive and full-ref validation over 37,632 samples.
- [~] T-HGPIPE-QUANT-LAYERS-008: all `attn_0..11` Q/K/V/A and all `mlp_0..11` GeLUQ cursor-table refs are contracted and validated; remaining work is default E2E wiring or additional LN/Softmax layer coverage.
- [x] T-HGPIPE-QUANT-ATTN0-QKVA-008A: add isolated HG-PIPE `attn_0_{q,k,v,a}_q` quant cursor-table primitives and full-ref validation over 150,528 total samples.
- [x] T-HGPIPE-QUANT-ATTN1-QKVA-008B: add isolated HG-PIPE `attn_1_{q,k,v,a}_q` quant cursor-table primitives and full-ref validation over 150,528 total samples.
- [x] T-HGPIPE-QUANT-ATTN2-QKVA-008C: add isolated HG-PIPE `attn_2_{q,k,v,a}_q` quant cursor-table primitives and full-ref validation over 150,528 total samples.
- [x] T-HGPIPE-QUANT-ATTN3-11-QKVA-008D: add isolated HG-PIPE `attn_3..11_{q,k,v,a}_q` quant cursor-table primitives and full-ref validation over 1,354,752 total samples.
- [x] T-HGPIPE-SOFTMAX-ATTN0-CURSOR-009: add isolated HG-PIPE `attn_0_softmaxq` exp/recip/requant primitives and full rowwise validation over 115,248 samples.
- [x] T-HGPIPE-SOFTMAX-ALL-014: add isolated HG-PIPE `attn_0..11_softmaxq` exp/recip/requant primitives and full rowwise validation over 1,382,976 samples.
- [x] T-HGPIPE-SOFTMAX-E2E-SCAFFOLD-018: add opt-in default-E2E attention softmaxq wiring scaffold with matching SW/HW reduced golden validation.
- [x] T-HGPIPE-MATH-E2E-NAMED-019: add durable `HGTXR_E2E_SCALE=hgpipe_math` named reduced gate combining GeLUQ and SoftmaxQ with generated spec/header, native comparator, Vitis csim, and Vitis csynth evidence.
- [x] T-HGPIPE-LAYERNORM-E2E-020: add durable `HGTXR_E2E_SCALE=hgpipe_math_lnq` named reduced gate combining GeLUQ, SoftmaxQ, and LayerNormQ with generated spec/header, native comparator, Vitis csim, and Vitis csynth evidence.
- [x] T-HGPIPE-MATH-LNQ-ACTIVE8-021: add staged `HGTXR_E2E_SCALE=hgpipe_math_lnq_active8` two-block active8 gate with generated spec/header, native comparator, Vitis csim, and Vitis csynth evidence.
- [ ] T-HGPIPE-SOFTMAX-E2E-010: decide whether to wire integer softmaxq into default E2E; if yes, update SW mirror and regenerate all affected golden specs/headers before Vitis csim/csynth.
- [x] T-HGPIPE-LAYERNORM-ATTN1-CURSOR-011: add isolated HG-PIPE `attn_1_lnq` mean/rsqrt/affine requant primitives and full-ref validation over 37,632 samples.
- [x] T-HGPIPE-LAYERNORM-ALL-013: add isolated HG-PIPE `attn_0..11_lnq` and `mlp_0..11_lnq` mean/rsqrt/affine requant primitives and full-ref validation over 903,168 samples.
- [x] T-HGPIPE-LAYERNORM-HEAD-016: add isolated HG-PIPE `head_lnq` mean/rsqrt/affine requant primitive and full-ref validation over 192 samples.
- [ ] T-HGPIPE-LAYERNORM-E2E-012: decide whether to wire integer LayerNorm into default E2E; if yes, update SW mirror and regenerate affected golden specs/headers before Vitis csim/csynth.
