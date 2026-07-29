> **작성** 2026-06-16 · **갱신** 2026-07-29
> **상태** active — 구 트리에서 이어받음, 경로 주석 추가
> **소유** hardware
>
> **PYNQ 오버레이 런타임 가이드.** `archive/hardware/pynq/hgtxr/README.md` 에서
> 2026-07-29 복사. **본문 무변경.**
>
> ⚠️ **경로가 바뀝니다.** 본문이 `.bit` / `.hwh` 가 오버레이 `.py` 와 같은
> 디렉토리에 있다고 기술하는데, 새 구조에서는 코드가 `deploy/`, 비트스트림이
> `workspace/` 로 분리됩니다. 구 트리는 실제로 `pynq/hgtxr/` 한 곳에 9개를 섞어
> 두고 있었고, 그것이 분리 이유입니다 ([README](README.md)).

---

# HGTXR PYNQ Overlay

First target: ZCU104.

Expected files after the selected A2 E2E m_axi Vivado build:

- hgtxr_e2e_m_axi.bit
- hgtxr_e2e_m_axi.hwh
- hgtxr_e2e_axis_dma.bit
- hgtxr_e2e_axis_dma.hwh
- hgtxr_e2e_axis_dma_par16.bit
- hgtxr_e2e_axis_dma_par16.hwh
- hgtxr_e2e_axis_dma_c3b_mem16.bit
- hgtxr_e2e_axis_dma_c3b_mem16.hwh
- hgtxr_overlay.py
- e2e_axis_dma_overlay.py
- e2e_m_axi_overlay.py
- e2e_m_axi_weights.py
- run_e2e_axis_dma_smoke.py
- run_e2e_m_axi_smoke.py
- test_hgtxr_overlay.py

Legacy `hgtxr.bit` and `hgtxr.hwh` may also exist in this directory for the
old `hgtxr_top` flow. Do not use those for the selected A2 path.

Typical use on PYNQ:

    from hgtxr_overlay import HgtxrOverlay, RuntimeConfig

    with HgtxrOverlay(RuntimeConfig()) as rt:
        out_state, runtime_state = rt.run(frame, event_pos, event_neg, prev_state)

The runtime uses signed Q6.10-style int16 transport by default because the HLS core is ap_fixed<16,6>.

A2 E2E m_axi helper:

    from hgtxr.e2e_m_axi_overlay import HgtxrE2EMaxiOverlay, E2EMaxiRuntimeConfig

    with HgtxrE2EMaxiOverlay(E2EMaxiRuntimeConfig()) as rt:
        out_state, runtime_state = rt.run(frame, weights_u32)

`weights_u32` is optional for allocation smoke. When omitted, the helper allocates a zero-filled packed Q4 weight buffer, so the HLS wrapper live-weight bit is `0` and `runtime_state` should be `1`. For the live-weight smoke path, pass a packed buffer with `weights_u32[0] = 1`; that sets bit 0 and `runtime_state` should be `2`.

A1 E2E AXIS/DMA helper:

    from hgtxr.e2e_axis_dma_overlay import HgtxrE2EAxisDmaOverlay, E2EAxisDmaRuntimeConfig

    with HgtxrE2EAxisDmaOverlay(E2EAxisDmaRuntimeConfig()) as rt:
        out_state, runtime_state = rt.run(frame, weights_u32)

The A1 helper streams the frame through AXI DMA and keeps the packed Q4 weights plus `runtime_state` on the HLS `m_axi` ports. The frame stream uses the same raw low-8-bit pixel contract as the HLS AXIS testbench: floating input values are converted with `round(value * 128)` and integer input values use the low byte.

C1/C3b AXIS/DMA candidate artifacts use the same helper with explicit bit/HWH paths:

    from pathlib import Path
    from hgtxr.e2e_axis_dma_overlay import HgtxrE2EAxisDmaOverlay, E2EAxisDmaRuntimeConfig

    cfg = E2EAxisDmaRuntimeConfig(
        bitfile=Path("hgtxr_e2e_axis_dma_c3b_mem16.bit"),
        hwhfile=Path("hgtxr_e2e_axis_dma_c3b_mem16.hwh"),
    )

    with HgtxrE2EAxisDmaOverlay(cfg) as rt:
        out_state, runtime_state = rt.run(frame, weights_u32)

Board-side smoke:

    python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode live --json-out e2e_m_axi_live_smoke.json
    python3 -m hgtxr.run_e2e_axis_dma_smoke --weights-mode live --json-out e2e_axis_dma_live_smoke.json

CSim-mirrored packed-weight smoke:

    python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode golden --json-out e2e_m_axi_golden_smoke.json
    python3 -m hgtxr.run_e2e_axis_dma_smoke --weights-mode golden --json-out e2e_axis_dma_golden_smoke.json

Exported packed-weight smoke:

    python3 hardware/tools/export_e2e_m_axi_weights.py
    python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode file --weights-bin hardware/refs/weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_m_axi_file_smoke.json

Transfer-bundle packed-weight smoke:

    ./run_e2e_m_axi_file_smoke.sh

Equivalent command inside the extracted bundle:

    PYTHONPATH=. python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_m_axi_file_smoke.json

C3b AXIS/DMA transfer-bundle packed-weight smoke:

    ./run_e2e_axis_dma_c3b_mem16_file_smoke.sh

Equivalent command inside the extracted C3b bundle:

    PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant c3b-mem16 --weights-mode file --weights-bin weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin --expect-out-raw 32 -13 26 -6 14 -11 --json-out e2e_axis_dma_c3b_mem16_file_smoke.json

C3b AXIS/DMA result validation inside the extracted C3b bundle:

    ./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh

Equivalent command inside the extracted C3b bundle:

    python3 tools/validate_pynq_smoke_result.py e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16 --json-out e2e_axis_dma_c3b_mem16_file_smoke_validation.json

C3b ZCU104 smoke session files generated on the host:

    hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.json
    hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_session.md

PAR32 ROM-only compute 300 MHz Search/Track smoke for the current objective-path bitstream:

    PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant par32-rom-compute-300 --weights-mode zero --mode-profile search --expect-runtime-state 0 --expect-out-raw -1169 -1169 -1169 -1169 -1169 -1133 --warmup 1 --repeat 5 --json-out e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json
    PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_smoke --variant par32-rom-compute-300 --weights-mode zero --mode-profile track --expect-runtime-state 1 --expect-out-raw -235 -235 -235 -235 -235 -226 --warmup 1 --repeat 5 --json-out e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json

PAR32 ROM-only compute result validation:

    python3 tools/validate_pynq_smoke_result.py e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json --preset axis-par32-rom-compute-300-search --json-out e2e_axis_dma_par32_rom_compute_300_search_file_smoke_validation.json
    python3 tools/validate_pynq_smoke_result.py e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json --preset axis-par32-rom-compute-300-track --json-out e2e_axis_dma_par32_rom_compute_300_track_file_smoke_validation.json

PAR32 prefetch-all4 300 MHz measured Search 10% / Track 90% hybrid smoke:

    PYTHONPATH=. python3 -m hgtxr.run_e2e_axis_dma_hybrid_smoke --variant par32-prefetchall4-300 --warmup-cycles 1 --repeat-cycles 5 --json-out e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json

PAR32 prefetch-all4 hybrid result validation:

    python3 tools/validate_pynq_smoke_result.py e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke.json --preset axis-par32-prefetchall4-300-hybrid-10-90 --json-out e2e_axis_dma_par32_prefetchall4_300_hybrid_10_90_file_smoke_validation.json

Zero-weight allocation smoke:

    python3 -m hgtxr.run_e2e_m_axi_smoke --weights-mode zero --json-out e2e_m_axi_zero_smoke.json

Run those commands from the directory that contains the `hgtxr/` package, or set `PYTHONPATH` so Python can import this package. `live` and `zero` modes check `runtime_state` only unless `--expect-out-raw` is also provided. `golden` mode builds the same active196_b6_ff768 packed Q4 test weights as the C++ m_axi CSim testbench and checks output raw values `[32, -13, 26, -6, 14, -11]`. `file` mode loads a raw little-endian `uint32` binary weight artifact. In the generated transfer bundle, the binary is already present under `weights/`.
