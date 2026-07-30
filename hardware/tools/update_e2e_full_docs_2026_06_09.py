from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

summary_row = (
    "2026-06-09,e2e_q4w8a_full_vit,solution_e2e_q4w8a,"
    "hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt,"
    "5.00,3.650,273.97,619214991,619214991,3.096 sec,3.096 sec,"
    "269,48,41885,61369,0,43,2,9,26,0,"
    "csim_passed_smoke_csynth_passed_full_dense_vit_axis\n"
)

csv = ROOT / "docs/resources/hls_csynth_summary_2026_06_06.csv"
if csv.exists():
    text = csv.read_text()
    if "e2e_q4w8a_full_vit" not in text:
        csv.write_text(text.rstrip() + "\n" + summary_row)

resource_doc = ROOT / "docs/resources/e2e_axis_baseline_2026_06_09.md"
section = """

## Full Dense ViT E2E Update - 2026-06-09

The E2E AXI-Stream top now contains a full dense ViT execution path for the
DeiT-Tiny/HG-PIPE dimensions used in the HGTXR target:

- `C=192`, `heads=3`, `head_dim=64`, `MLP hidden=768`.
- Default full target config uses `active_tokens=196`, `blocks=6`,
  `patch_grid=14x14`, `Q4W/Q8A`, `bus_width=128`, `FIFO_DEPTH=64`.
- The controller alternates two physical ATTN units and two physical MLP units:
  `ATTN0 -> MLP0 -> ATTN1 -> MLP1 -> ATTN0 -> MLP0 ...`.
- The top has AXI-Stream input/output, AXI master packed-weight reads, AXI
  master runtime-state write, a controller, and BRAM-backed global buffers.
- The full path performs conv-style patch embedding, pre-LN attention
  (`Q/K/V -> score -> softmax approximation -> value -> output projection`),
  pre-LN MLP (`W1 -> GELU approximation -> W2`), residual updates, and an
  MLP-based output head.

Validation:

- `g++` smoke passed with reduced verification config:
  `runtime_state=1 count=6 last=1`.
- Vitis HLS `csim_design` passed after fixing Ubuntu 24.04/WSL integration:
  multiarch include paths were added, stale testbench files were removed by
  using a reset E2E project, and PATH/LIBRARY_PATH were set so system linker
  and libraries are used instead of the incompatible Vitis 2023.2 bundled
  binutils path.
- Vitis HLS `csynth_design` passed for the full target config in
  `hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a`.
- Full target report:
  - estimated clock: `3.650 ns` (`273.97 MHz`)
  - latency: `619,214,991 cycles`
  - absolute latency at 5 ns target: `3.096 sec`
  - utilization: `269 BRAM_18K`, `48 DSP`, `41,885 FF`, `61,369 LUT`, `0 URAM`
  - utilization percent: `43% BRAM`, `2% DSP`, `9% FF`, `26% LUT`, `0% URAM`

Current limitation: this is the first full dense E2E hardware path. It is
functionally complete at the block-operation level, but it still uses
hardware-friendly approximate LayerNorm denominator, GELU, and softmax math
instead of the final HG-PIPE table math. The next performance step is to
increase datapath parallelism and prefetch/cache packed weights so the design
uses more ZCU104 DSP bandwidth and reduces the current sequential latency.
"""
text = resource_doc.read_text() if resource_doc.exists() else "# E2E AXI-Stream ViT Baseline - 2026-06-09\n"
if "## Full Dense ViT E2E Update - 2026-06-09" not in text:
    resource_doc.write_text(text.rstrip() + section)

updates = {
    "docs/Validation-2026-06-05-Status.md": """

## Full Dense ViT E2E Validation - 2026-06-09

- Fixed Vitis HLS `csim_design` on Ubuntu 24.04/WSL by adding multiarch include
  paths, isolating the E2E project from stale testbench files, and forcing
  system linker/library paths.
- `g++` smoke passed: `runtime_state=1 count=6 last=1`.
- Vitis HLS `csim_design` passed for the reduced E2E smoke config.
- Vitis HLS `csynth_design` passed for the full `196-token, 6-block,
  C=192/H=3/FF=768` target.
- Full report:
  `hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a/syn/report/hgtxr_e2e_axis_top_csynth.rpt`.
""",
    "docs/track/PROGRESS-2026-06-06.md": """

## 2026-06-09 - Full Dense ViT E2E Path

- Completed a full dense E2E AXI-Stream ViT HLS path with conv patch embedding,
  two reused ATTN units, two reused MLP units, controller, global buffer, AXI
  master packed-weight reads, runtime-state write, and MLP head.
- Corrected execution order to ViT-equivalent `ATTN -> MLP` per block.
- `csim_design` and full-target `csynth_design` now pass.
- Current full-target latency is `619,214,991 cycles` (`3.096 sec` at 5 ns),
  which is the unoptimized cyclic dense baseline for the entire model.
""",
    "docs/track/HANDOFF-2026-06-06.md": """

## Latest Continuation: Full Dense ViT E2E AXI-Stream Path

- New/updated files:
  - `hardware/hls/include/hgtxr_e2e_vit.hpp`
  - `hardware/hls/src/hgtxr_e2e_axis_top.cpp`
  - `hardware/hls/tb/tb_hgtxr_e2e_axis_top.cpp`
  - `hardware/configs/zcu104_e2e_q4w8a_defines.h`
  - `hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl`
  - `hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl`
- Full target: `196 tokens`, `6 blocks`, `C=192`, `H=3`, `head_dim=64`,
  `FF=768`, `Q4W/Q8A`, `128-bit AXI`.
- `csim_design` passes with reduced smoke overrides. Full `csynth_design`
  passes in `hardware/generated/hgtxr_e2e_hls/solution_e2e_q4w8a`.
- Current full-target result: `273.97 MHz`, `619,214,991 cycles`, `3.096 sec`,
  `269 BRAM`, `48 DSP`, `41,885 FF`, `61,369 LUT`.
- Next work: replace approximation math with HG-PIPE tables, add packed-weight
  prefetch/cache, and raise channel/head parallelism until ZCU104 DSP/BRAM
  utilization is closer to the target resource budget.
""",
    "docs/track/log.md": """

## 2026-06-09 - Full dense E2E ViT HLS implementation

- Implemented full dense ATTN/MLP/head/patch path inside the E2E AXI-Stream top.
- Fixed Vitis HLS `csim` under Ubuntu 24.04/WSL by adding multiarch includes,
  using a reset E2E HLS project, and forcing system linker/library paths.
- Verified `g++` smoke, Vitis `csim_design`, static artifact validation, and
  full-target `csynth_design`.
""",
    "docs/track/CHANGELOG.md": """

## 0.1.21 - 2026-06-09

- Added full dense ViT E2E execution path for the HGTXR/DeiT-Tiny target.
- Fixed Vitis HLS `csim_design` environment integration on Ubuntu 24.04/WSL.
- Added full-target HLS synthesis result for the E2E AXI-Stream top.
""",
}

for rel, addition in updates.items():
    p = ROOT / rel
    text = p.read_text() if p.exists() else ""
    marker = addition.strip().splitlines()[0]
    if marker not in text:
        p.write_text(text.rstrip() + addition)
