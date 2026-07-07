# HW/SW Interface

Software exports:

- `hardware/refs/inputs/frame.npy`
- `hardware/refs/inputs/event.npy`
- `hardware/refs/inputs/prev_state.npy`
- `hardware/refs/outputs/target_state.npy`
- `hardware/refs/weights/software_initial_weights.pt`

Hardware should emit:

- `hardware/refs/outputs/hls_state.npy`

`hardware/tools/compare_hw_sw.py` compares the hardware output against the software
target state.

