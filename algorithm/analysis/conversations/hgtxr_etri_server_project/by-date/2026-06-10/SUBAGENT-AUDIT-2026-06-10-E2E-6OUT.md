# Sub-agent Audit - 2026-06-10 E2E 6-Output Gate

## Provenance

- Dalton: agent 019eae77-eb38-7a12-aaa0-1acfa6e97def, read-only verification explorer for dense/arbitrary reduced E2E expansion.
- Russell: agent 019eae58-3d96-7d82-b00a-ab4c98e95621, read-only audit for the 4-to-6 grouped-state expansion.
- Both agents reported no file edits.

## Current State Confirmed

- Current reduced E2E AXI vector gate is a fixed six-output grouped Q4 pattern.
- C++ testbench: hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp uses kGroupChannels=6, kPatchRaw[6], and kHeadRaw[6].
- Python reference: hardware/tools/validate_e2e_axis_vector.py uses group_channels=6 and checks expected_raw=[18, -3, 0, -2, 0, -4] with runtime_state=2.
- Pytest compiles the C++ gate with reduced overrides: blocks=1, active_tokens=4, patch_grid=1x4, ff_dim=32.
- Output path emits HGTXR_STATE words through hgtxr_axis_write_state; HGTXR_STATE is six.

## Layout Constraints

- Weight layout is static and flattened. hgtxr_e2e_load_weight maps elem_offset to word_idx/lane_idx.
- Patch weights start at kPatchWeightElemBase, block weights at kBlockWeightElemBase, and head weights at kHeadWeightElemBase.
- weights[0].bit(0) is used as the live_weight_bit, affects out_state[0], and sets runtime_state=1+live_weight_bit.
- Output raw packing uses int(value * 16) into the low 16 bits; references must mirror truncation, not rounding.
- num_pixels is currently ignored by hgtxr_e2e_axis_top, so frame-size behavior remains compile-time configured.

## Recommended Next Expansion

1. Keep the architecture unchanged and make the current reduced vector data-driven.
2. Keep six output state words, the existing packing layout, and the live-bit convention.
3. Replace duplicated C++/Python hardcoded arrays with a single seeded deterministic Q4 spec or generated golden.
4. Regenerate expected outputs from Python and feed the golden to the C++ testbench, still using exact raw integer comparison.
5. Increase nonzero support in Wq/Wk/Wv/Wo/W1/W2 while keeping blocks=1.
6. Increase one axis at a time after green validation: active_tokens, then ff_dim, then block count.

## Risks To Avoid

- Diverging constants between Python, C++ testbench, and pytest compile flags.
- Reusing stale expected_raw or runtime_state literals after widening the pattern.
- Writing nonzero values into packed offsets without recalculating layout math.
- Clobbering weights[0].bit(0) while adding arbitrary patch/channel data.
- Assuming full equivalence when the current scope is still reduced deterministic verification.
- Relying only on pytest output lane 0 text checks; every output lane should be parsed or checked.
- Changing group semantics without updating the reference math assumption channels_per_group=embed//group_channels.

## Captured Validation Commands

```bash
cd /home/user/project/PRJXR/HGTXR
python3 hardware/tools/validate_e2e_axis_vector.py --json-out /tmp/e2e_axis_vector_6out_ref.json
./.venv/bin/python -m pytest -s software/tests/test_e2e_axis_vector_csim.py -q
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl
/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls -f hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl
python3 hardware/tools/static_validate_hgtxr.py --root .
```

Native g++ command for the reduced gate:

```bash
g++ -std=c++17 -I/tools/Xilinx/Vitis_HLS/2023.2/include \
  -Ihardware/hls/include -include hardware/configs/zcu104_e2e_q4w8a_defines.h \
  -DHGTXR_E2E_BLOCKS=1 -DHGTXR_E2E_ACTIVE_TOKENS=4 \
  -DHGTXR_E2E_PATCH_GRID_H=1 -DHGTXR_E2E_PATCH_GRID_W=4 -DHGTXR_E2E_FF_DIM=32 \
  hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp hardware/hls/src/hgtxr_e2e_axis_top.cpp \
  -o /tmp/tb_hgtxr_e2e_axis_top && /tmp/tb_hgtxr_e2e_axis_top
```

## Reported Verification Status

- SW reference passed with expected_raw=[18, -3, 0, -2, 0, -4] and runtime_state=2.
- Pytest reported 2 passed.
- Native g++ executable printed all six outputs and runtime_state=2 count=6 last=1 failures=0.
- Vitis csim passed with the same six-word output sequence.
- Static validation passed.

## Integration Decision

Do not jump directly to active_tokens=196 and blocks=6. First make the reduced E2E gate data-driven and denser while preserving the current architecture and output ABI. Then increase one compile-time axis per validation cycle.
