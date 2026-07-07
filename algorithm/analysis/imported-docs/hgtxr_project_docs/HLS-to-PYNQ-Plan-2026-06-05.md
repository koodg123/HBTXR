# HGTXR HLS-to-PYNQ Plan - 2026-06-05

## Goal
Complete the path from HGTXR HLS source to Vitis HLS IP, Vivado bitstream, and PYNQ overlay using Xilinx tools under /tools/Xilinx.

## Confirmed Environment
- WSL project path: /home/user/project/PRJXR/HGTXR
- Vitis HLS 2023.2 is available from /tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls.
- Vivado 2023.2 is available from /tools/Xilinx/Vivado/2023.2/bin/vivado.
- Current encoded board target is ZCU104: xczu7ev-ffvc1156-2-e, xilinx.com:zcu104:part0:1.1.
- VCK190 metadata exists, but the active HLS Tcl currently targets ZCU104.

## Current Verification Evidence
- Host C++ smoke passed with output: state=1 out0=128.000000 out1=128.000000.
- Vitis HLS csynth passed after fixing ambiguous ap_fixed conditionals in hardware/hls/src/nonlinear.cpp and hardware/hls/src/rmu_smu.cpp.
- Main report: hardware/generated/hgtxr_hls/solution/syn/report/hgtxr_top_csynth.rpt.
- HLS target clock: 5.00 ns.
- HLS estimated clock: 10.279 ns, estimated Fmax: 97.29 MHz.
- HLS latency: 10834350 to 10895934 cycles.
- HLS utilization estimate on ZCU104: BRAM_18K 269 of 624, DSP 89 of 1728, FF 14543 of 460800, LUT 20263 of 230400, URAM 0 of 96.

## Current Blockers
- Vitis HLS csim is blocked on Ubuntu 24.04 because Xilinx 2023.2 bundled binutils cannot link the host glibc libm with RELR sections.
- Current HLS top ports synthesize as ap_memory plus ap_ctrl_hs, not as PYNQ-friendly AXI-Lite and AXI master ports.
- hardware/vivado/scripts/build_bitstream.tcl is still a placeholder.
- No .bit, .hwh, .xsa, or PYNQ overlay package exists yet.
- Current 5 ns timing target is too aggressive for the prototype; first board bring-up should use about 10 ns or 100 MHz.

## Required Items
- Confirm final board target, assumed ZCU104 unless changed.
- ZCU104 PYNQ image, SSH access, and working Python pynq package on the board.
- Supported HLS csim environment, preferably Ubuntu 22.04 or a compatibility sysroot, if Vitis csim is mandatory.
- HLS top interface update to AXI-Lite control plus AXI master data buffers.
- Real Vivado block design Tcl for PS, clocks, resets, AXI interconnect, address map, bitstream, and hwh export.
- PYNQ Python driver and test vectors copied from hardware/refs.

## Phase 1 - Stabilize HLS Build
- Keep host g++ smoke as the fast local gate.
- Add or document environment variables for Ubuntu multiarch include paths: C_INCLUDE_PATH and CPLUS_INCLUDE_PATH include /usr/include/x86_64-linux-gnu.
- Use sh hardware/scripts/run_hls_csim.sh and sh hardware/scripts/run_hls_csynth.sh unless executable bits are added.
- Treat csim failure as environment compatibility until a supported sysroot/container is ready.

## Phase 2 - Make HLS IP PYNQ-Friendly
- Add HLS interface pragmas to hgtxr_top.cpp.
- Recommended first interface: s_axilite for control and m_axi for frame, event_pos, event_neg, prev_state, out_state.
- Consider placing runtime_state into out_state or a small status buffer to simplify PYNQ register handling.
- Re-run host smoke and csynth.
- Acceptance: HLS report shows AXI interfaces rather than ap_memory for main data ports.

## Phase 3 - Export HLS IP
- Run Vitis HLS package flow with hardware/vivado/scripts/package_ip.tcl.
- Acceptance: hardware/generated/hgtxr_hls/solution/impl/ip contains a valid component XML and packaged IP output.

## Phase 4 - Build Vivado ZCU104 Design
- Replace hardware/vivado/scripts/build_bitstream.tcl with real automation.
- Create ZCU104 Vivado project and block design.
- Add Zynq UltraScale Plus MPSoC PS.
- Add HGTXR HLS IP repository and instantiate the IP.
- Connect PS HPM master to HGTXR AXI-Lite control.
- Connect HGTXR AXI master ports to PS DDR through SmartConnect and HP or HPC ports.
- Connect clock and reset, assign addresses, validate design, synthesize, implement, and write bitstream.
- Export .hwh next to .bit for PYNQ.

## Phase 5 - Build PYNQ Overlay
- Create hardware/pynq/hgtxr with hgtxr.bit, hgtxr.hwh, hgtxr_overlay.py, test_hgtxr_overlay.py, and data files.
- Driver should load Overlay, allocate buffers, copy test vectors, write physical addresses to registers, start IP, poll done, and compare output.
- Acceptance: overlay loads, IP reaches done, output buffer changes deterministically, and HW/SW comparison is recorded.

## Phase 6 - Optimize After Bring-Up
- First bring-up should relax clock to 10 ns.
- Then remove divisions or replace them with reciprocal/LUT approximations.
- Bank and partition token buffers where memory bandwidth limits II.
- Consider splitting attention and MLP into separate kernels if monolithic top timing remains poor.

## Task Cards
- T-001 HLS validator: own hardware/hls/src and HLS Tcl, keep smoke and csynth passing.
- T-002 Interface implementer: own hgtxr_top.cpp, common.h, HW-SW docs, convert ports to PYNQ-friendly AXI.
- T-003 Vivado integrator: own build_bitstream.tcl and board Tcl, produce .bit and .hwh.
- T-004 PYNQ runtime: own hardware/pynq/hgtxr and docs/Validation.md, produce overlay driver and board smoke test.

## Immediate Next Actions
1. Confirm ZCU104 as the first board target.
2. Update hgtxr_top interface pragmas for AXI-Lite plus AXI master.
3. Re-run host smoke and Vitis HLS csynth.
4. Export HLS IP.
5. Implement real Vivado bitstream Tcl.
6. Generate .bit and .hwh.
7. Create and run PYNQ overlay smoke test on the board.
