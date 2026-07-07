# HGTXR Hardware TODO

## P0
- Capture/import C3b AXIS/DMA physical smoke JSON.
- Resolve exact XR-VITs sibling or approved replacement policy.
- Run/import ZCU104 physical smoke JSON for `softmax_input_x2` `dsp_mixed_stream` using either `generated/pynq/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md` or `python3 tools/run_zcu104_c3b_smoke_remote.py --profile vref-p0-softmax-input-x2-dsp-mixed-stream --host <zcu104-ip-or-host> --user xilinx --execute`.
- Copy successor board result back to `pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json` so preflight/projection can promote it.
- Keep C3b as protected baseline until board-smoke and XR-VITs reference gates are closed.
- For additional non-current `VREF-P0-01` scale successors, regenerate a dedicated golden header and pass CSim before promotion.
- Keep `VREF-P0-02` QKV cache URAM promotion as a successor-only experiment; HLS CSim/csynth, HLS IP package, routed overlay timing, and PYNQ bundle/session artifacts are present; physical smoke remains pending.

## P1
- Check nonlinear bottleneck before TATAA mixed-precision path.
- Prepare search/track expert ablation only as bounded/static routing.
- Prepare exact-attention fallback for ViTALiTy/ViTCoD ablations.

## P2/P3
- Keep FlexLLM-style generator as tooling-only.
- Keep LUT-heavy/ternary methods as negative controls.
