from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

defines = [
    ("HGTXR_TILING_FACTOR", "1"),
    ("HGTXR_PARALLELISM_FACTOR", "2"),
    ("HGTXR_BUS_WIDTH", "128"),
    ("HGTXR_BIT_WIDTH", "8"),
    ("HGTXR_WEIGHT_BIT_WIDTH", "4"),
    ("HGTXR_WEIGHT_INT_WIDTH", "2"),
    ("HGTXR_BUFFER_SIZE", "128"),
    ("HGTXR_FIFO_DEPTH", "64"),
    ("HGTXR_ENABLE_CYCLIC_WEIGHT_PORTS", "1"),
    ("HGTXR_CYCLIC_MODEL_DIM", "192"),
    ("HGTXR_CYCLIC_MLP_RATIO", "4"),
    ("HGTXR_CYCLIC_WEIGHT_BLOCKS", "6"),
    ("HGTXR_E2E_BLOCKS", "6"),
    ("HGTXR_E2E_ATTN_UNITS", "2"),
    ("HGTXR_E2E_MLP_UNITS", "2"),
    ("HGTXR_E2E_ATTN_PAR", "2"),
    ("HGTXR_E2E_WEIGHT_DEPTH", "381024"),
    ("HGTXR_E2E_ACTIVE_TOKENS", "196"),
    ("HGTXR_E2E_PATCH_GRID_H", "14"),
    ("HGTXR_E2E_PATCH_GRID_W", "14"),
    ("HGTXR_E2E_HEADS", "3"),
    ("HGTXR_E2E_HEAD_DIM", "64"),
    ("HGTXR_E2E_FF_DIM", "768"),
    ("HGTXR_E2E_DENSE_PAR", "2"),
    ("HGTXR_E2E_ACC_SCALE", "16"),
]

config = ROOT / "hardware/configs/zcu104_e2e_q4w8a_defines.h"
body = [
    "#ifndef HGTXR_ZCU104_E2E_Q4W8A_DEFINES_H",
    "#define HGTXR_ZCU104_E2E_Q4W8A_DEFINES_H",
    "",
]
for key, value in defines:
    body.extend([f"#ifndef {key}", f"#define {key} {value}", "#endif"])
body.extend(["", "#endif  // HGTXR_ZCU104_E2E_Q4W8A_DEFINES_H", ""])
config.write_text("\n".join(body))

multiarch = "-I/usr/include/x86_64-linux-gnu -I/usr/lib/gcc/x86_64-linux-gnu/13/include"
full_include = "-include [file normalize hardware/configs/zcu104_e2e_q4w8a_defines.h]"
base = f"-std=c++17 -Ihardware/hls/include {multiarch} -DHGTXR_ENABLE_CYCLIC_WEIGHT_PORTS=1"
csim_flags = (
    f'{base} -DHGTXR_E2E_BLOCKS=1 -DHGTXR_E2E_ACTIVE_TOKENS=4 '
    f'-DHGTXR_E2E_PATCH_GRID_H=1 -DHGTXR_E2E_PATCH_GRID_W=4 '
    f'-DHGTXR_E2E_FF_DIM=32 {full_include}'
)
csynth_flags = f"{base} {full_include}"

for rel, flags in [
    ("hardware/vivado/scripts/run_e2e_q4w8a_csim.tcl", csim_flags),
    ("hardware/vivado/scripts/run_e2e_q4w8a_csynth.tcl", csynth_flags),
]:
    p = ROOT / rel
    lines = p.read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("set cxx_flags "):
            lines[i] = f'set cxx_flags "{flags}"'
    p.write_text("\n".join(lines) + "\n")
