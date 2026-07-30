#!/usr/bin/env python3
"""Audit Req1 host environment, Xilinx tool paths, and workspace relocation."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, Sequence


DATE_TAG = "2026_06_16"
EXPECTED_HARDWARE_SUFFIX = "project/PRJXR/XR-VIT/HGTXR/hardware"
EXPECTED_HGTXR_SUFFIX = "project/PRJXR/XR-VIT/HGTXR"
EXPECTED_PROJECT_PREFIX = "/home/kjm26/project/"
VITIS_HLS = Path("/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls")
VIVADO = Path("/tools/Xilinx/Vivado/2023.2/bin/vivado")


def normalize_roots(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "hardware":
        return root.parent, root
    return root, root / "hardware"


def read_text(path: Path) -> str:
    return path.read_text(errors="replace") if path.exists() else ""


def parse_os_release(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any) -> None:
    checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})


def path_has_suffix(path: Path, suffix: str) -> bool:
    return path.as_posix().endswith(suffix)


def build_audit(root: Path) -> dict[str, Any]:
    hgtxr, hardware = normalize_roots(root)
    os_release = parse_os_release(read_text(Path("/etc/os-release")))
    proc_version = read_text(Path("/proc/version"))
    platform_system = platform.system()
    legacy_markers = ["/mnt/c", "/mnt/d", "C:\\", "/opt/Xilinx"]
    legacy_hits = [
        marker
        for marker in legacy_markers
        if marker.lower() in hardware.as_posix().lower() or marker.lower() in hgtxr.as_posix().lower()
    ]
    checks: list[dict[str, Any]] = []

    add_check(checks, "platform_linux", platform_system == "Linux", platform_system)
    add_check(checks, "ubuntu_id", os_release.get("ID") == "ubuntu", os_release)
    add_check(checks, "ubuntu_version_22_04", os_release.get("VERSION_ID") == "22.04", os_release.get("VERSION_ID"))
    add_check(checks, "not_wsl_kernel", "microsoft" not in proc_version.lower(), proc_version.splitlines()[0] if proc_version else "")
    add_check(checks, "hgtxr_root_expected_suffix", path_has_suffix(hgtxr, EXPECTED_HGTXR_SUFFIX), str(hgtxr))
    add_check(checks, "hardware_root_expected_suffix", path_has_suffix(hardware, EXPECTED_HARDWARE_SUFFIX), str(hardware))
    add_check(
        checks,
        "hardware_root_expected_prefix",
        hardware.as_posix().startswith(EXPECTED_PROJECT_PREFIX),
        str(hardware),
    )
    add_check(checks, "hardware_root_exists", hardware.exists(), str(hardware))
    add_check(checks, "no_legacy_wsl_or_xilinx_paths", not legacy_hits, legacy_hits)
    add_check(checks, "vitis_hls_tools_xilinx_exists", VITIS_HLS.exists(), str(VITIS_HLS))
    add_check(checks, "vitis_hls_tools_xilinx_executable", os.access(VITIS_HLS, os.X_OK), str(VITIS_HLS))
    add_check(checks, "vivado_tools_xilinx_exists", VIVADO.exists(), str(VIVADO))
    add_check(checks, "vivado_tools_xilinx_executable", os.access(VIVADO, os.X_OK), str(VIVADO))

    fail_count = sum(1 for check in checks if check["status"] == "fail")
    return {
        "status": "pass" if fail_count == 0 else "fail",
        "date_tag": DATE_TAG,
        "root": str(hgtxr),
        "hardware_root": str(hardware),
        "environment": {
            "platform_system": platform_system,
            "platform_release": platform.release(),
            "os_release": os_release,
            "proc_version": proc_version.splitlines()[0] if proc_version else "",
            "vitis_hls": str(VITIS_HLS),
            "vivado": str(VIVADO),
        },
        "policy": {
            "ubuntu_linux_required": True,
            "wsl_legacy_path_rejected": True,
            "xilinx_root": "/tools/Xilinx",
            "expected_project_prefix": EXPECTED_PROJECT_PREFIX,
            "expected_hardware_suffix": EXPECTED_HARDWARE_SUFFIX,
        },
        "checks": checks,
        "pass_count": len(checks) - fail_count,
        "fail_count": fail_count,
        "check_count": len(checks),
        "safety": {
            "executes_tools": False,
            "executes_hls": False,
            "executes_vivado": False,
            "writes_canonical_inputs": False,
        },
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Req1 Environment Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- root: `{audit['root']}`",
        f"- hardware_root: `{audit['hardware_root']}`",
        f"- checks: `{audit['pass_count']}/{audit['check_count']}`",
        f"- fail_count: `{audit['fail_count']}`",
        "",
        "## Environment",
        "",
    ]
    for key, value in audit["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Checks", "", "| Check | Status | Detail |", "|---|---|---|"])
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| {check['name']} | `{check['status']}` | {detail} |")
    lines.extend(["", "## Safety", ""])
    for key, value in audit["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Req1 environment audit.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.root)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(audit))
    print(f"[req1-environment-audit] status={audit['status']} checks={audit['pass_count']}/{audit['check_count']}")
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
