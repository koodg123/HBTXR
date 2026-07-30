#!/usr/bin/env python3
"""Repair and verify the HGTXR HLS-to-PYNQ artifact handoff.

This helper is intentionally narrow:
- patch the Vivado Tcl HWH-copy fallback that used unsupported `glob -recursive`
- copy generated bit/hwh artifacts into the overlay and PYNQ package directories
- print the exact local PYNQ smoke commands to run

Run from the HGTXR project root:

    python3 hardware/tools/repair_hls_to_pynq.py --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


TCL_OLD = "set bd_hwh [glob -nocomplain -recursive -directory $build_dir *.hwh]"
TCL_NEW = """set bd_hwh [concat \\
        [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name "hw_handoff"] *.hwh] \\
        [glob -nocomplain -directory [file join $build_dir "${project_name}.gen" "sources_1" "bd" $bd_name] *.hwh] \\
    ]"""


def patch_tcl(root: Path, apply: bool) -> bool:
    script = root / "hardware" / "vivado" / "scripts" / "build_bitstream.tcl"
    if not script.exists():
        print(f"[missing] {script}")
        return False

    text = script.read_text()
    if TCL_NEW in text:
        print(f"[ok] Tcl HWH-copy patch already present: {script}")
        return True
    if TCL_OLD not in text:
        print(f"[warn] expected Tcl pattern not found: {script}")
        return False

    if apply:
        script.write_text(text.replace(TCL_OLD, TCL_NEW))
        print(f"[patched] {script}")
    else:
        print(f"[dry-run] would patch {script}")
    return True


def copy_if_present(src: Path, dst: Path, apply: bool) -> bool:
    if not src.exists():
        print(f"[missing] {src}")
        return False
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[copied] {src} -> {dst}")
    else:
        print(f"[dry-run] would copy {src} -> {dst}")
    return True


def copy_artifacts(root: Path, apply: bool) -> bool:
    bit_src = (
        root
        / "build"
        / "vivado"
        / "hgtxr_overlay"
        / "hgtxr_overlay.runs"
        / "impl_1"
        / "hgtxr_system_wrapper.bit"
    )
    hwh_src = (
        root
        / "build"
        / "vivado"
        / "hgtxr_overlay"
        / "hgtxr_overlay.gen"
        / "sources_1"
        / "bd"
        / "hgtxr_system"
        / "hw_handoff"
        / "hgtxr_system.hwh"
    )

    overlay_dir = root / "build" / "vivado" / "overlay" / "hgtxr_overlay"
    pynq_dir = root / "pynq" / "hgtxr"

    ok = True
    ok &= copy_if_present(bit_src, overlay_dir / "hgtxr.bit", apply)
    ok &= copy_if_present(hwh_src, overlay_dir / "hgtxr.hwh", apply)
    ok &= copy_if_present(bit_src, pynq_dir / "hgtxr.bit", apply)
    ok &= copy_if_present(hwh_src, pynq_dir / "hgtxr.hwh", apply)
    return ok


def print_smoke_commands() -> None:
    print("\n[smoke]")
    print("python3 -m py_compile hardware/pynq/hgtxr/hgtxr_overlay.py hardware/pynq/hgtxr/test_hgtxr_overlay.py")
    print("PYTHONPATH=hardware/pynq/hgtxr python3 hardware/pynq/hgtxr/test_hgtxr_overlay.py")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="HGTXR project root")
    parser.add_argument("--apply", action="store_true", help="modify files and copy artifacts")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"[root] {root}")
    patch_ok = patch_tcl(root, args.apply)
    copy_ok = copy_artifacts(root, args.apply)
    print_smoke_commands()
    return 0 if patch_ok and copy_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

