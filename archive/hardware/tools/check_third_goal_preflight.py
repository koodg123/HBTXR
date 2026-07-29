#!/usr/bin/env python3
"""Branch-neutral preflight checks for the active third-goal HGTXR work."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import create_xr_vits_replacement_policy
from validate_pynq_smoke_result import validate_result
from validate_pynq_bundle_package import validate_bundle


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def add(results: list[dict[str, Any]], status: str, name: str, detail: str, path: Path | None = None) -> None:
    row = {"status": status, "name": name, "detail": detail}
    if path is not None:
        row["path"] = str(path)
    results.append(row)


def exists_check(results: list[dict[str, Any]], name: str, path: Path, missing_status: str = "fail") -> None:
    if path.exists():
        add(results, "ok", name, "exists", path)
    else:
        add(results, missing_status, name, "missing", path)


def xr_vits_candidate_detail(root: Path, parent: Path, requested: Path) -> str:
    replacement = parent / "XR_Accel"
    if not replacement.exists():
        return ""

    audit_path = root / "docs" / "resources" / "xr_vits_candidate_audit_2026_06_10.json"
    if not audit_path.exists():
        return f"; XR_Accel candidate exists at {replacement}, approval still required"

    try:
        audit = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError):
        return f"; XR_Accel candidate exists at {replacement}, candidate audit unreadable, approval still required"

    recommendation = audit.get("recommendation", {}) if isinstance(audit, dict) else {}
    recommended_path = recommendation.get("path") if isinstance(recommendation, dict) else ""
    requested_matches = isinstance(audit, dict) and audit.get("requested_path") == str(requested)
    recommended_matches = Path(str(recommended_path)).resolve() == replacement.resolve()
    if requested_matches and recommended_matches:
        return f"; XR_Accel candidate exists and candidate audit recommends it, approval still required: {replacement}"
    return f"; XR_Accel candidate exists at {replacement}, but candidate audit does not fully match, approval still required"


def check_xr_vits_reference(results: list[dict[str, Any]], root: Path, parent: Path, mode: str) -> None:
    requested = parent.parent / "XR-VITs"
    if requested.exists():
        add(results, "ok", "requested XR-VITs sibling", "exists", requested)
        return

    missing_status = final_status(mode)
    candidate_detail = xr_vits_candidate_detail(root, parent, requested)
    policy_path = root / "docs" / "resources" / "xr_vits_replacement_policy.json"
    unblock_hint = (
        "; see generated/signoff/xr_vits_unblock_packet_2026_06_10.md; "
        "dry-run policy command: python3 tools/create_xr_vits_replacement_policy.py "
        "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR --approve --approved-by <approved-by> "
        "--reason \"Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff\" --dry-run"
    )
    if not policy_path.exists():
        add(
            results,
            missing_status,
            "requested XR-VITs sibling",
            "missing; no approved replacement policy" + candidate_detail + unblock_hint,
            requested,
        )
        return

    try:
        policy = json.loads(policy_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(
            results,
            missing_status,
            "requested XR-VITs sibling",
            f"missing; replacement policy invalid JSON: {exc}",
            policy_path,
        )
        return

    errors = []
    if policy.get("approved_replacement") is not True:
        errors.append("approved_replacement is not true")
    replacement = policy.get("replacement_path")
    replacement_path = Path(replacement) if isinstance(replacement, str) else None
    if replacement_path is None:
        errors.append("replacement_path missing")
    elif not replacement_path.exists():
        errors.append(f"replacement_path missing: {replacement_path}")
    if not policy.get("approved_by"):
        errors.append("approved_by missing")
    if policy.get("requested_path") != str(requested):
        errors.append(f"requested_path mismatch: {policy.get('requested_path')!r}")
    integrity = create_xr_vits_replacement_policy.validate_policy_integrity(root, policy)
    if integrity["status"] != "pass":
        errors.extend(integrity.get("errors", []))
        errors.extend(integrity.get("warnings", []))

    if errors:
        add(
            results,
            missing_status,
            "requested XR-VITs sibling",
            "missing; replacement policy rejected: " + "; ".join(errors) + candidate_detail + unblock_hint,
            policy_path,
        )
    else:
        add(
            results,
            "ok",
            "requested XR-VITs sibling",
            f"approved replacement: {replacement_path}; policy_fingerprint validated",
            policy_path,
        )


def is_board_artifact_mode(mode: str) -> bool:
    return mode in {"board-ready", "final-signoff"}


def mode_status(mode: str, neutral_status: str, board_status: str) -> str:
    return board_status if is_board_artifact_mode(mode) else neutral_status


def final_status(mode: str, default_status: str = "warn") -> str:
    return "fail" if mode == "final-signoff" else default_status


def executable_check(results: list[dict[str, Any]], name: str, path: Path, mode: str, board_required: bool) -> None:
    if path.exists():
        add(results, "ok", name, "exists", path)
    else:
        status = "fail" if board_required and is_board_artifact_mode(mode) else "warn"
        add(results, status, name, "not found; long HLS/Vivado runs cannot be launched from this path", path)


def macro_value(text: str, macro: str) -> str | None:
    marker = f"#define {macro} "
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            return stripped[len(marker) :].split()[0]
    return None


def macro_check(results: list[dict[str, Any]], text: str, macro: str, expected: str) -> None:
    value = macro_value(text, macro)
    if value == expected:
        add(results, "ok", f"macro {macro}", f"{value}")
    elif value is None:
        add(results, "fail", f"macro {macro}", f"missing; expected {expected}")
    else:
        add(results, "fail", f"macro {macro}", f"{value}; expected {expected}")


def check_tool_on_path(results: list[dict[str, Any]], tool: str, status_if_missing: str = "warn") -> None:
    found = shutil.which(tool)
    if found:
        add(results, "ok", f"tool {tool}", "found on PATH", Path(found))
    else:
        add(results, status_if_missing, f"tool {tool}", "not found on PATH")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return ""
    return digest.hexdigest()


def resolve_manifest_binary(hardware: Path, manifest_path: Path, manifest: dict[str, Any]) -> Path:
    binary = manifest.get("binary")
    if not isinstance(binary, str) or not binary:
        return hardware / "refs" / "weights" / "e2e_m_axi_active196_b6_ff768_q4_u32.bin"
    path = Path(binary)
    if path.is_absolute():
        if path.exists():
            return path
        return hardware / "refs" / "weights" / path.name
    return (hardware / path).resolve() if binary.startswith("hardware/") else (manifest_path.parent / path).resolve()


def q4_raw_from_file(path: Path, elem_offset: int) -> int | None:
    u32_idx = elem_offset // 8
    shift = (elem_offset % 8) * 4
    try:
        with path.open("rb") as handle:
            handle.seek(u32_idx * 4)
            data = handle.read(4)
    except OSError:
        return None
    if len(data) != 4:
        return None
    raw = (int.from_bytes(data, "little") >> shift) & 0xF
    return raw - 16 if raw & 0x8 else raw


def tail_zero(path: Path, start_u32: int) -> bool:
    try:
        with path.open("rb") as handle:
            handle.seek(start_u32 * 4)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if any(chunk):
                    return False
    except OSError:
        return False
    return True


def check_generated_artifacts(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    generated_roots = [
        ("AXIS E2E", hardware / "generated" / "hgtxr_e2e_hls"),
        ("A1 AXIS/DMA E2E", hardware / "generated" / "hgtxr_e2e_axis_hls"),
        ("C1 PAR16 AXIS/DMA E2E", hardware / "generated" / "hgtxr_e2e_axis_par16_hls"),
        ("C3b PAR16 MEM16 AXIS/DMA E2E", hardware / "generated" / "hgtxr_e2e_axis_par16_c3b_mem16_hls"),
        ("m_axi E2E", hardware / "generated" / "hgtxr_e2e_m_axi_hls"),
        ("VREF-P0-01 softmax_input_x2 CSim", hardware / "generated" / "hgtxr_e2e_axis_vref_p0_softmax_input_x2_csim"),
        ("VREF-P0-01 softmax_input_x2 csynth", hardware / "generated" / "hgtxr_e2e_axis_vref_p0_softmax_input_x2_csynth"),
        (
            "VREF-P0-01 softmax_input_x2 dsp_mixed_stream csynth",
            hardware / "generated" / "hgtxr_e2e_axis_vref_p0_softmax_input_x2_dsp_mixed_stream_csynth",
        ),
    ]
    csim_logs: list[Path] = []
    components: list[Path] = []
    csynth_reports: list[Path] = []
    visible_roots: list[str] = []

    for label, generated in generated_roots:
        if generated.exists():
            visible_roots.append(label)
            csim_logs.extend(sorted(generated.rglob("*csim.log")))
            components.extend(sorted(generated.rglob("component.xml")))
            csynth_reports.extend(sorted(generated.rglob("*csynth.rpt")))

    fallback_root = generated_roots[0][1]

    if csim_logs:
        add(results, "ok", "E2E CSim log", f"{len(csim_logs)} visible", csim_logs[0])
    else:
        add(results, "warn", "E2E CSim log", "not visible under generated E2E trees", fallback_root)

    if components:
        add(results, "ok", "E2E packaged IP component.xml", f"{len(components)} visible", components[0])
    else:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "E2E packaged IP component.xml",
            "not generated yet; A1/A2 packaging required",
            fallback_root,
        )

    if csynth_reports:
        root_note = ", ".join(visible_roots) if visible_roots else "unknown root"
        add(results, "ok", "E2E csynth report", f"{len(csynth_reports)} visible across {root_note}", csynth_reports[0])
    else:
        add(results, "warn", "E2E csynth report", "not visible in generated E2E trees; current final report may be documented/archived separately", fallback_root)


def check_board_flow_mismatch(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    create_hls = hardware / "vivado" / "scripts" / "create_hls_project.tcl"
    build_bitstream = hardware / "vivado" / "scripts" / "build_bitstream.tcl"
    package_e2e_m_axi = hardware / "vivado" / "scripts" / "package_e2e_m_axi_ip.tcl"
    build_e2e_m_axi = hardware / "vivado" / "scripts" / "build_e2e_m_axi_bitstream.tcl"
    e2e_top = hardware / "hls" / "src" / "hgtxr_e2e_axis_top.cpp"

    create_text = read_text(create_hls)
    build_text = read_text(build_bitstream)
    package_e2e_text = read_text(package_e2e_m_axi)
    build_e2e_text = read_text(build_e2e_m_axi)
    e2e_text = read_text(e2e_top)

    has_e2e_board_flow = (
        "set_top hgtxr_e2e_m_axi_top" in package_e2e_text
        and "xilinx.com:hls:hgtxr_e2e_m_axi_top:1.0" in build_e2e_text
        and "hgtxr_e2e_m_axi.bit" in build_e2e_text
    )

    if has_e2e_board_flow:
        add(results, "ok", "board flow selected E2E m_axi", "package/build scripts target hgtxr_e2e_m_axi_top", build_e2e_m_axi)
    elif "set_top hgtxr_top" in create_text and "xilinx.com:hls:hgtxr_top:1.0" in build_text:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "board flow top mismatch",
            "current package/build flow targets old hgtxr_top; selected E2E top needs A1/A2/A3 decision",
            build_bitstream,
        )
    else:
        add(results, "ok", "board flow top mismatch", "old-flow hardcoding not detected", build_bitstream)

    if "void hgtxr_e2e_axis_top" in e2e_text and "#pragma HLS INTERFACE axis" in e2e_text:
        add(results, "ok", "selected E2E AXIS top", "hgtxr_e2e_axis_top exposes AXIS ports", e2e_top)
    else:
        add(results, "fail", "selected E2E AXIS top", "expected AXIS top signature/pragmas not found", e2e_top)

    e2e_m_axi_top = hardware / "hls" / "src" / "hgtxr_e2e_m_axi_top.cpp"
    e2e_m_axi_text = read_text(e2e_m_axi_top)
    if "void hgtxr_e2e_m_axi_top" in e2e_m_axi_text and "#pragma HLS INTERFACE m_axi" in e2e_m_axi_text:
        add(results, "ok", "selected E2E m_axi top", "A2 wrapper exposes memory-mapped ports", e2e_m_axi_top)
    else:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "selected E2E m_axi top",
            "A2 wrapper missing or missing m_axi interface markers",
            e2e_m_axi_top,
        )


def check_pynq_overlay_artifacts(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    legacy_hwh = hardware / "pynq" / "hgtxr" / "hgtxr.hwh"
    legacy_bit = hardware / "pynq" / "hgtxr" / "hgtxr.bit"
    e2e_hwh = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_m_axi.hwh"
    e2e_bit = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_m_axi.bit"
    axis_hwh = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma.hwh"
    axis_bit = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma.bit"
    generated_overlay = hardware / "generated" / "build" / "vivado" / "overlay" / "hgtxr_e2e_m_axi_overlay"
    generated_hwh = generated_overlay / "hgtxr_e2e_m_axi.hwh"
    generated_bit = generated_overlay / "hgtxr_e2e_m_axi.bit"
    generated_axis_overlay = hardware / "generated" / "build" / "vivado" / "overlay" / "hgtxr_e2e_axis_dma_overlay"
    generated_axis_hwh = generated_axis_overlay / "hgtxr_e2e_axis_dma.hwh"
    generated_axis_bit = generated_axis_overlay / "hgtxr_e2e_axis_dma.bit"
    par16_hwh = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_par16.hwh"
    par16_bit = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_par16.bit"
    generated_par16_overlay = hardware / "generated" / "build" / "vivado" / "overlay" / "hgtxr_e2e_axis_dma_par16_overlay"
    generated_par16_hwh = generated_par16_overlay / "hgtxr_e2e_axis_dma_par16.hwh"
    generated_par16_bit = generated_par16_overlay / "hgtxr_e2e_axis_dma_par16.bit"
    c3b_hwh = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.hwh"
    c3b_bit = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.bit"
    generated_c3b_overlay = hardware / "generated" / "build" / "vivado" / "overlay" / "hgtxr_e2e_axis_dma_c3b_mem16_overlay"
    generated_c3b_hwh = generated_c3b_overlay / "hgtxr_e2e_axis_dma_c3b_mem16.hwh"
    generated_c3b_bit = generated_c3b_overlay / "hgtxr_e2e_axis_dma_c3b_mem16.bit"

    bit = e2e_bit if e2e_bit.exists() else generated_bit
    hwh = e2e_hwh if e2e_hwh.exists() else generated_hwh
    exists_check(results, "PYNQ E2E m_axi bit", bit, missing_status=mode_status(mode, "warn", "fail"))
    exists_check(results, "PYNQ E2E m_axi hwh", hwh, missing_status=mode_status(mode, "warn", "fail"))

    axis_bit_candidate = axis_bit if axis_bit.exists() else generated_axis_bit
    axis_hwh_candidate = axis_hwh if axis_hwh.exists() else generated_axis_hwh
    exists_check(results, "PYNQ E2E AXIS/DMA bit", axis_bit_candidate, missing_status=mode_status(mode, "warn", "fail"))
    exists_check(results, "PYNQ E2E AXIS/DMA hwh", axis_hwh_candidate, missing_status=mode_status(mode, "warn", "fail"))
    axis_text = read_text(axis_hwh_candidate)
    if axis_text:
        if "hgtxr_e2e_axis_top" in axis_text:
            add(results, "ok", "PYNQ AXIS/DMA hwh selected E2E IP", "hgtxr_e2e_axis_top found", axis_hwh_candidate)
        else:
            add(
                results,
                mode_status(mode, "warn", "fail"),
                "PYNQ AXIS/DMA hwh selected E2E IP",
                "hgtxr_e2e_axis_top marker not found",
                axis_hwh_candidate,
            )

    par16_bit_candidate = par16_bit if par16_bit.exists() else generated_par16_bit
    par16_hwh_candidate = par16_hwh if par16_hwh.exists() else generated_par16_hwh
    par16_present = par16_bit_candidate.exists() or par16_hwh_candidate.exists()
    if par16_present:
        exists_check(results, "PYNQ C1 PAR16 AXIS/DMA bit", par16_bit_candidate, missing_status=mode_status(mode, "warn", "fail"))
        exists_check(results, "PYNQ C1 PAR16 AXIS/DMA hwh", par16_hwh_candidate, missing_status=mode_status(mode, "warn", "fail"))
        par16_text = read_text(par16_hwh_candidate)
        if "hgtxr_e2e_axis_top" in par16_text:
            add(results, "ok", "PYNQ C1 PAR16 hwh selected E2E IP", "hgtxr_e2e_axis_top found", par16_hwh_candidate)
        elif par16_text:
            add(
                results,
                mode_status(mode, "warn", "fail"),
                "PYNQ C1 PAR16 hwh selected E2E IP",
                "hgtxr_e2e_axis_top marker not found",
                par16_hwh_candidate,
            )

    c3b_bit_candidate = c3b_bit if c3b_bit.exists() else generated_c3b_bit
    c3b_hwh_candidate = c3b_hwh if c3b_hwh.exists() else generated_c3b_hwh
    c3b_present = c3b_bit_candidate.exists() or c3b_hwh_candidate.exists()
    if c3b_present:
        exists_check(results, "PYNQ C3b PAR16 MEM16 AXIS/DMA bit", c3b_bit_candidate, missing_status=mode_status(mode, "warn", "fail"))
        exists_check(results, "PYNQ C3b PAR16 MEM16 AXIS/DMA hwh", c3b_hwh_candidate, missing_status=mode_status(mode, "warn", "fail"))
        c3b_text = read_text(c3b_hwh_candidate)
        if "hgtxr_e2e_axis_top" in c3b_text:
            add(results, "ok", "PYNQ C3b PAR16 MEM16 hwh selected E2E IP", "hgtxr_e2e_axis_top found", c3b_hwh_candidate)
        elif c3b_text:
            add(
                results,
                mode_status(mode, "warn", "fail"),
                "PYNQ C3b PAR16 MEM16 hwh selected E2E IP",
                "hgtxr_e2e_axis_top marker not found",
                c3b_hwh_candidate,
            )

    text = read_text(hwh)
    if text:
        has_old_top = "hgtxr_top_0" in text or "xilinx.com:hls:hgtxr_top:1.0" in text
        has_e2e_axis = "hgtxr_e2e_axis_top" in text
        has_e2e_mm = "hgtxr_e2e_m_axi_top" in text
        if has_e2e_axis or has_e2e_mm:
            add(results, "ok", "PYNQ hwh selected E2E IP", "E2E IP name found", hwh)
        elif has_old_top:
            add(
                results,
                mode_status(mode, "warn", "fail"),
                "PYNQ hwh stale old-flow IP",
                "hgtxr_top_0 found; do not use this hwh/bit as proof of selected E2E top",
                hwh,
            )
        else:
            add(
                results,
                mode_status(mode, "warn", "fail"),
                "PYNQ hwh selected E2E IP",
                "neither selected E2E IP nor old hgtxr_top marker found",
                hwh,
            )

    legacy_text = read_text(legacy_hwh)
    if not legacy_text:
        return
    has_old_top = "hgtxr_top_0" in legacy_text or "xilinx.com:hls:hgtxr_top:1.0" in legacy_text
    has_e2e_axis = "hgtxr_e2e_axis_top" in legacy_text
    has_e2e_mm = "hgtxr_e2e_m_axi_top" in legacy_text
    if has_e2e_axis or has_e2e_mm:
        add(results, "ok", "legacy PYNQ hwh selected E2E IP", "E2E IP name found", legacy_hwh)
    elif has_old_top:
        add(
            results,
            "warn",
            "legacy PYNQ hwh stale old-flow IP",
            "hgtxr_top_0 found; ignored because E2E m_axi artifacts are checked separately",
            legacy_hwh,
        )


def check_e2e_weight_artifacts(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    weights_dir = hardware / "refs" / "weights"
    manifest_path = weights_dir / "e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json"
    missing_status = mode_status(mode, "warn", "fail")
    default_bin_path = weights_dir / "e2e_m_axi_active196_b6_ff768_q4_u32.bin"

    exists_check(results, "E2E m_axi packed Q4 weight manifest", manifest_path, missing_status=missing_status)
    if not manifest_path.exists():
        exists_check(results, "E2E m_axi packed Q4 weight bin", default_bin_path, missing_status=missing_status)
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(results, missing_status, "E2E m_axi weight manifest parse", f"invalid JSON: {exc}", manifest_path)
        return

    bin_path = resolve_manifest_binary(hardware, manifest_path, manifest)
    exists_check(results, "E2E m_axi packed Q4 weight bin", bin_path, missing_status=missing_status)

    required_fields = [
        "name",
        "binary",
        "sha256",
        "bytes",
        "dtype",
        "format",
        "shape",
        "capacity_u32",
        "required_u32",
        "tail_zero_start_u32",
        "layout",
        "known_offset_checks",
    ]
    missing = [field for field in required_fields if field not in manifest]
    if missing:
        add(results, "fail", "E2E m_axi weight manifest fields", f"missing {missing}", manifest_path)
    else:
        add(results, "ok", "E2E m_axi weight manifest fields", "required fields present", manifest_path)

    expected_raw = manifest.get("expected_raw")
    if expected_raw == [32, -13, 26, -6, 14, -11]:
        add(results, "ok", "E2E m_axi weight expected raw", f"{expected_raw}", manifest_path)
    else:
        add(results, missing_status, "E2E m_axi weight expected raw", f"{expected_raw}; expected [32, -13, 26, -6, 14, -11]", manifest_path)

    if manifest.get("expected_runtime_state") == 2:
        add(results, "ok", "E2E m_axi weight runtime state", "2", manifest_path)
    else:
        add(results, missing_status, "E2E m_axi weight runtime state", f"{manifest.get('expected_runtime_state')}; expected 2", manifest_path)

    layout = manifest.get("layout") if isinstance(manifest.get("layout"), dict) else {}
    contract_checks = {
        "format": (manifest.get("format"), "raw-little-endian-uint32"),
        "dtype": (manifest.get("dtype"), "uint32"),
        "layout.bus_width_bits": (layout.get("bus_width_bits"), 256),
        "layout.weight_bits": (layout.get("weight_bits"), 4),
        "layout.u32_per_256b_word": (layout.get("u32_per_256b_word"), 8),
        "layout.q4_lanes_per_256b_word": (layout.get("q4_lanes_per_256b_word"), 64),
        "layout.blocks": (layout.get("blocks"), 6),
        "layout.ff_dim": (layout.get("ff_dim"), 768),
        "layout.embed": (layout.get("embed"), 192),
        "layout.patch": (layout.get("patch"), 16),
        "layout.state": (layout.get("state"), 6),
    }
    bad_contract = [f"{name}={actual} expected {expected}" for name, (actual, expected) in contract_checks.items() if actual != expected]
    if bad_contract:
        add(results, "fail", "E2E m_axi weight contract", "; ".join(bad_contract), manifest_path)
    else:
        add(results, "ok", "E2E m_axi weight contract", "A2 Q4W8A active196_b6_ff768 layout", manifest_path)

    shape = manifest.get("shape")
    capacity_u32 = manifest.get("capacity_u32")
    required_u32 = manifest.get("required_u32")
    tail_start = manifest.get("tail_zero_start_u32")
    size_errors = []
    if not (isinstance(shape, list) and len(shape) == 1 and shape[0] == capacity_u32):
        size_errors.append(f"shape={shape} capacity_u32={capacity_u32}")
    if not isinstance(capacity_u32, int) or manifest.get("bytes") != capacity_u32 * 4:
        size_errors.append(f"bytes={manifest.get('bytes')} capacity_u32={capacity_u32}")
    if (
        isinstance(layout.get("e2e_weight_words_256b"), int)
        and isinstance(layout.get("u32_per_256b_word"), int)
        and capacity_u32 != layout["e2e_weight_words_256b"] * layout["u32_per_256b_word"]
    ):
        size_errors.append("capacity_u32 does not match e2e_weight_words_256b * u32_per_256b_word")
    if (
        isinstance(layout.get("required_weight_words_256b"), int)
        and isinstance(layout.get("u32_per_256b_word"), int)
        and required_u32 != layout["required_weight_words_256b"] * layout["u32_per_256b_word"]
    ):
        size_errors.append("required_u32 does not match required_weight_words_256b * u32_per_256b_word")
    if not (tail_start == required_u32 and isinstance(required_u32, int) and isinstance(capacity_u32, int) and required_u32 <= capacity_u32):
        size_errors.append(f"tail_zero_start_u32={tail_start} required_u32={required_u32} capacity_u32={capacity_u32}")
    if size_errors:
        add(results, "fail", "E2E m_axi weight layout size math", "; ".join(size_errors), manifest_path)
    else:
        add(results, "ok", "E2E m_axi weight layout size math", "shape/bytes/capacity/required/tail values agree", manifest_path)

    if bin_path.exists():
        manifest_bytes = manifest.get("bytes")
        actual_bytes = bin_path.stat().st_size
        if manifest_bytes == actual_bytes:
            add(results, "ok", "E2E m_axi weight bytes", f"{actual_bytes}", bin_path)
        else:
            add(results, missing_status, "E2E m_axi weight bytes", f"{actual_bytes}; manifest {manifest_bytes}", bin_path)

        manifest_hash = manifest.get("sha256")
        actual_hash = sha256_file(bin_path)
        if manifest_hash == actual_hash:
            add(results, "ok", "E2E m_axi weight sha256", actual_hash, bin_path)
        else:
            add(results, missing_status, "E2E m_axi weight sha256", f"{actual_hash}; manifest {manifest_hash}", bin_path)

        if isinstance(tail_start, int) and tail_zero(bin_path, tail_start):
            add(results, "ok", "E2E m_axi weight tail zero", f"from u32 {tail_start}", bin_path)
        elif isinstance(tail_start, int):
            add(results, "fail", "E2E m_axi weight tail zero", f"nonzero data after u32 {tail_start}", bin_path)

        probe_specs = {
            "patch_channel0_elem0": layout.get("patch_weight_elem_base"),
            "patch_channel2_elem0": (
                layout.get("patch_weight_elem_base") + 2 * layout.get("patch") * layout.get("patch")
                if all(isinstance(layout.get(key), int) for key in ["patch_weight_elem_base", "patch"])
                else None
            ),
            "head_diag1": (
                layout.get("head_weight_elem_base") + layout.get("embed") + 1
                if all(isinstance(layout.get(key), int) for key in ["head_weight_elem_base", "embed"])
                else None
            ),
            "block0_wq_diag0": (
                layout.get("block_weight_elem_base") + 4 * layout.get("embed")
                if all(isinstance(layout.get(key), int) for key in ["block_weight_elem_base", "embed"])
                else None
            ),
        }
        known = manifest.get("known_offset_checks") if isinstance(manifest.get("known_offset_checks"), dict) else {}
        probe_errors = []
        for name, elem in probe_specs.items():
            if not isinstance(elem, int):
                probe_errors.append(f"{name}: offset unavailable")
                continue
            actual = q4_raw_from_file(bin_path, elem)
            if actual != known.get(name):
                probe_errors.append(f"{name}: actual {actual} manifest {known.get(name)}")
        if probe_errors:
            add(results, "fail", "E2E m_axi weight Q4 probes", "; ".join(probe_errors), bin_path)
        else:
            add(results, "ok", "E2E m_axi weight Q4 probes", "known packed nibbles match manifest", bin_path)


def check_c3b_axis_dma_smoke_bundle(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    tar_path = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
    run_script = bundle_dir / "run_e2e_axis_dma_c3b_mem16_file_smoke.sh"
    validate_script = bundle_dir / "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh"
    c3b_bit = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.bit"
    c3b_hwh = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.hwh"

    c3b_artifact_present = c3b_bit.exists() or c3b_hwh.exists()
    bundle_present = bundle_dir.exists() or manifest_path.exists() or tar_path.exists()
    if not c3b_artifact_present and not bundle_present:
        return

    missing_status = mode_status(mode, "warn", "fail")
    exists_check(results, "C3b AXIS/DMA PYNQ smoke bundle dir", bundle_dir, missing_status=missing_status)
    exists_check(results, "C3b AXIS/DMA PYNQ smoke bundle manifest", manifest_path, missing_status=missing_status)
    exists_check(results, "C3b AXIS/DMA PYNQ smoke bundle tar", tar_path, missing_status=missing_status)
    exists_check(results, "C3b AXIS/DMA PYNQ smoke bundle run script", run_script, missing_status=missing_status)
    exists_check(results, "C3b AXIS/DMA PYNQ smoke bundle validate script", validate_script, missing_status=missing_status)

    if not manifest_path.exists():
        return

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle manifest parse", f"invalid JSON: {exc}", manifest_path)
        return

    if manifest.get("variant") == "c3b-mem16":
        add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle variant", "c3b-mem16", manifest_path)
    else:
        add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle variant", f"{manifest.get('variant')}; expected c3b-mem16", manifest_path)

    if manifest.get("expected_out_raw") == [32, -13, 26, -6, 14, -11] and manifest.get("expected_runtime_state") == 2:
        add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle expected output", "[32, -13, 26, -6, 14, -11], runtime_state=2", manifest_path)
    else:
        add(
            results,
            missing_status,
            "C3b AXIS/DMA PYNQ smoke bundle expected output",
            f"raw={manifest.get('expected_out_raw')} runtime_state={manifest.get('expected_runtime_state')}",
            manifest_path,
        )

    command = manifest.get("command")
    if isinstance(command, str) and "--variant c3b-mem16" in command and "--weights-mode file" in command:
        add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle command", "variant and file weights selected", manifest_path)
    else:
        add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle command", f"{command}", manifest_path)

    validation_command = manifest.get("validation_command")
    if (
        isinstance(validation_command, str)
        and "tools/validate_pynq_smoke_result.py" in validation_command
        and "--preset axis-c3b-mem16" in validation_command
    ):
        add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle validation command", "axis-c3b-mem16 validator selected", manifest_path)
    else:
        add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle validation command", f"{validation_command}", manifest_path)

    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    bundled_names = {entry.get("bundle") for entry in files if isinstance(entry, dict)}
    required_bundle_files = {
        "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
        "hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
        "hgtxr/run_e2e_axis_dma_smoke.py",
        "weights/e2e_m_axi_active196_b6_ff768_q4_u32.bin",
        "weights/e2e_m_axi_active196_b6_ff768_q4_u32_manifest.json",
        "tools/validate_pynq_smoke_result.py",
        "run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        "validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
    }
    missing_files = sorted(required_bundle_files - bundled_names)
    if missing_files:
        add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle contents", f"missing {missing_files}", manifest_path)
    else:
        add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle contents", "required bit/hwh/helper/weights/run script present", manifest_path)

    if tar_path.exists():
        actual_hash = sha256_file(tar_path)
        manifest_tar = manifest.get("tar") if isinstance(manifest.get("tar"), dict) else {}
        manifest_hash = manifest_tar.get("sha256")
        if actual_hash == manifest_hash:
            add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle tar sha256", actual_hash, tar_path)
        else:
            add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle tar sha256", f"{actual_hash}; manifest {manifest_hash}", tar_path)

    bundle_errors = validate_bundle(bundle_dir, tar_path, "c3b-mem16")
    if bundle_errors:
        add(results, missing_status, "C3b AXIS/DMA PYNQ smoke bundle package", "; ".join(bundle_errors), manifest_path)
    else:
        add(results, "ok", "C3b AXIS/DMA PYNQ smoke bundle package", "tar contents and manifest file hashes validated", manifest_path)


def check_c3b_smoke_session(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle"
    bundle_manifest = bundle_dir / "BUNDLE_MANIFEST.json"
    tar_path = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
    session_json = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.json"
    session_md = hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_session.md"
    c3b_bit = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.bit"
    c3b_hwh = hardware / "pynq" / "hgtxr" / "hgtxr_e2e_axis_dma_c3b_mem16.hwh"

    c3b_artifact_present = c3b_bit.exists() or c3b_hwh.exists()
    bundle_present = bundle_dir.exists() or bundle_manifest.exists() or tar_path.exists()
    session_present = session_json.exists() or session_md.exists()
    if not c3b_artifact_present and not bundle_present and not session_present:
        return

    missing_status = mode_status(mode, "warn", "fail")
    exists_check(results, "C3b ZCU104 smoke session json", session_json, missing_status=missing_status)
    exists_check(results, "C3b ZCU104 smoke session markdown", session_md, missing_status=missing_status)
    if not session_json.exists():
        return

    try:
        session = json.loads(session_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(results, missing_status, "C3b ZCU104 smoke session parse", f"invalid JSON: {exc}", session_json)
        return

    session_errors: list[str] = []
    expected_pairs = {
        "status": "pass",
        "target": "ZCU104 PYNQ",
        "variant": "c3b-mem16",
        "preset": "axis-c3b-mem16",
        "expected_runtime_state": 2,
        "expected_out_raw": [32, -13, 26, -6, 14, -11],
        "result_json": "e2e_axis_dma_c3b_mem16_file_smoke.json",
        "validation_json": "e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
    }
    for key, expected in expected_pairs.items():
        if session.get(key) != expected:
            session_errors.append(f"{key}: {session.get(key)!r} != {expected!r}")

    canonical_path = session.get("canonical_result_path")
    if not isinstance(canonical_path, str) or not canonical_path.endswith("hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"):
        session_errors.append(f"canonical_result_path: {canonical_path!r}")

    tar = session.get("tar") if isinstance(session.get("tar"), dict) else {}
    if tar_path.exists():
        actual_hash = sha256_file(tar_path)
        if tar.get("sha256") != actual_hash:
            session_errors.append(f"tar.sha256: {tar.get('sha256')!r} != {actual_hash!r}")
        if tar.get("bytes") != tar_path.stat().st_size:
            session_errors.append(f"tar.bytes: {tar.get('bytes')!r} != {tar_path.stat().st_size!r}")
    else:
        session_errors.append(f"tar missing: {tar_path}")

    board_steps = session.get("board_steps") if isinstance(session.get("board_steps"), list) else []
    for token in [
        "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
        "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
        "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
    ]:
        if token not in board_steps:
            session_errors.append(f"board_steps missing {token!r}")

    host_steps = session.get("host_steps") if isinstance(session.get("host_steps"), list) else []
    host_text = "\n".join(str(step) for step in host_steps)
    has_direct_import = "tools/import_pynq_smoke_result.py" in host_text and "--preset axis-c3b-mem16" in host_text
    has_runner_import = (
        "tools/run_third_goal_final_signoff.py" in host_text
        and "--import-c3b-smoke-json" in host_text
        and "--dry-run-import-c3b-smoke" in host_text
        and "--allow-blocked" in host_text
    )
    if not (has_direct_import or has_runner_import):
        session_errors.append("host_steps missing C3b import command")
    for token in [
        "tools/check_third_goal_preflight.py",
        "--mode final-signoff",
    ]:
        if token not in host_text:
            session_errors.append(f"host_steps missing {token!r}")

    bundle_errors = session.get("bundle_validation_errors")
    if bundle_errors not in ([], None):
        session_errors.append(f"bundle_validation_errors: {bundle_errors!r}")

    if session_errors:
        add(results, missing_status, "C3b ZCU104 smoke session contract", "; ".join(session_errors), session_json)
    else:
        add(results, "ok", "C3b ZCU104 smoke session contract", "board and host runbook fields validated", session_json)

    if session_md.exists():
        text = session_md.read_text()
        required = [
            "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
            "--import-c3b-smoke-json",
            "--dry-run-import-c3b-smoke",
        ]
        missing = [token for token in required if token not in text]
        if "HGTXR C3b ZCU104 Smoke Session" not in text and "HGTXR ZCU104 Smoke Session" not in text:
            missing.append("HGTXR ZCU104 Smoke Session")
        if missing:
            add(results, missing_status, "C3b ZCU104 smoke session markdown contract", f"missing {missing}", session_md)
        else:
            add(results, "ok", "C3b ZCU104 smoke session markdown contract", "runbook text includes board/import commands", session_md)
    else:
        add(results, "ok", "C3b ZCU104 smoke session markdown contract", "runbook text includes board/import commands", session_md)


def check_vref_successor_smoke_bundle(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    variant = "vref-p0-softmax-input-x2-dsp-mixed-stream"
    preset = "axis-vref-p0-softmax-input-x2-dsp-mixed-stream"
    prefix = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream"
    bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle"
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    tar_path = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle.tar.gz"
    session_json = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.json"
    session_md = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_session.md"
    run_script = bundle_dir / "run_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh"
    validate_script = bundle_dir / "validate_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.sh"
    missing_status = "warn"

    exists_check(results, "VREF-P0 successor PYNQ smoke bundle dir", bundle_dir, missing_status=missing_status)
    exists_check(results, "VREF-P0 successor PYNQ smoke bundle manifest", manifest_path, missing_status=missing_status)
    exists_check(results, "VREF-P0 successor PYNQ smoke bundle tar", tar_path, missing_status=missing_status)
    exists_check(results, "VREF-P0 successor PYNQ smoke run script", run_script, missing_status=missing_status)
    exists_check(results, "VREF-P0 successor PYNQ smoke validate script", validate_script, missing_status=missing_status)
    exists_check(results, "VREF-P0 successor ZCU104 smoke session json", session_json, missing_status=missing_status)
    exists_check(results, "VREF-P0 successor ZCU104 smoke session markdown", session_md, missing_status=missing_status)

    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(results, missing_status, "VREF-P0 successor PYNQ smoke bundle manifest parse", f"invalid JSON: {exc}", manifest_path)
        return

    if manifest.get("variant") == variant:
        add(results, "ok", "VREF-P0 successor PYNQ smoke bundle variant", variant, manifest_path)
    else:
        add(results, missing_status, "VREF-P0 successor PYNQ smoke bundle variant", f"{manifest.get('variant')}; expected {variant}", manifest_path)

    if manifest.get("artifact_prefix") == prefix:
        add(results, "ok", "VREF-P0 successor PYNQ smoke bundle artifact prefix", prefix, manifest_path)
    else:
        add(
            results,
            missing_status,
            "VREF-P0 successor PYNQ smoke bundle artifact prefix",
            f"{manifest.get('artifact_prefix')}; expected {prefix}",
            manifest_path,
        )

    if manifest.get("expected_out_raw") == [58, -51, 42, -28, 36, -41] and manifest.get("expected_runtime_state") == 2:
        add(results, "ok", "VREF-P0 successor PYNQ smoke expected output", "[58, -51, 42, -28, 36, -41], runtime_state=2", manifest_path)
    else:
        add(
            results,
            missing_status,
            "VREF-P0 successor PYNQ smoke expected output",
            f"raw={manifest.get('expected_out_raw')} runtime_state={manifest.get('expected_runtime_state')}",
            manifest_path,
        )

    command = manifest.get("command")
    if isinstance(command, str) and f"--variant {variant}" in command and "--weights-mode file" in command:
        add(results, "ok", "VREF-P0 successor PYNQ smoke command", "successor variant and file weights selected", manifest_path)
    else:
        add(results, missing_status, "VREF-P0 successor PYNQ smoke command", f"{command}", manifest_path)

    validation_command = manifest.get("validation_command")
    if isinstance(validation_command, str) and f"--preset {preset}" in validation_command:
        add(results, "ok", "VREF-P0 successor PYNQ smoke validation command", f"{preset} validator selected", manifest_path)
    else:
        add(results, missing_status, "VREF-P0 successor PYNQ smoke validation command", f"{validation_command}", manifest_path)

    bundle_errors = validate_bundle(bundle_dir, tar_path, variant)
    if bundle_errors:
        add(results, missing_status, "VREF-P0 successor PYNQ smoke bundle package", "; ".join(bundle_errors), manifest_path)
    else:
        add(results, "ok", "VREF-P0 successor PYNQ smoke bundle package", "tar contents and manifest file hashes validated", manifest_path)


def check_qkv_uram_successor_smoke_artifacts(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    variant = "vref-p0-softmax-input-x2-qkv-uram"
    preset = "axis-vref-p0-softmax-input-x2-qkv-uram"
    prefix = "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram"
    bundle_dir = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle"
    manifest_path = bundle_dir / "BUNDLE_MANIFEST.json"
    tar_path = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz"
    session_json = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.json"
    session_md = hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_session.md"
    run_script = bundle_dir / "run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh"
    validate_script = bundle_dir / "validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh"
    bit = hardware / "pynq" / "hgtxr" / f"{prefix}.bit"
    hwh = hardware / "pynq" / "hgtxr" / f"{prefix}.hwh"

    artifact_present = bit.exists() or hwh.exists() or bundle_dir.exists() or manifest_path.exists() or session_json.exists()
    if not artifact_present:
        return

    missing_status = "warn"
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke bit", bit, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke hwh", hwh, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke bundle dir", bundle_dir, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke bundle manifest", manifest_path, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke bundle tar", tar_path, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke run script", run_script, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM PYNQ smoke validate script", validate_script, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM ZCU104 smoke session json", session_json, missing_status=missing_status)
    exists_check(results, "VREF-P0-02 QKV URAM ZCU104 smoke session markdown", session_md, missing_status=missing_status)

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            add(results, missing_status, "VREF-P0-02 QKV URAM PYNQ smoke bundle manifest parse", f"invalid JSON: {exc}", manifest_path)
            return

        if manifest.get("variant") == variant:
            add(results, "ok", "VREF-P0-02 QKV URAM PYNQ smoke bundle variant", variant, manifest_path)
        else:
            add(results, missing_status, "VREF-P0-02 QKV URAM PYNQ smoke bundle variant", f"{manifest.get('variant')}; expected {variant}", manifest_path)

        if manifest.get("artifact_prefix") == prefix:
            add(results, "ok", "VREF-P0-02 QKV URAM PYNQ smoke bundle artifact prefix", prefix, manifest_path)
        else:
            add(
                results,
                missing_status,
                "VREF-P0-02 QKV URAM PYNQ smoke bundle artifact prefix",
                f"{manifest.get('artifact_prefix')}; expected {prefix}",
                manifest_path,
            )

        if manifest.get("expected_out_raw") == [58, -51, 42, -28, 36, -41] and manifest.get("expected_runtime_state") == 2:
            add(results, "ok", "VREF-P0-02 QKV URAM PYNQ smoke expected output", "[58, -51, 42, -28, 36, -41], runtime_state=2", manifest_path)
        else:
            add(
                results,
                missing_status,
                "VREF-P0-02 QKV URAM PYNQ smoke expected output",
                f"raw={manifest.get('expected_out_raw')} runtime_state={manifest.get('expected_runtime_state')}",
                manifest_path,
            )

        command = manifest.get("command")
        if isinstance(command, str) and f"--variant {variant}" in command and "--weights-mode file" in command:
            add(results, "ok", "VREF-P0-02 QKV URAM PYNQ smoke command", "successor variant and file weights selected", manifest_path)
        else:
            add(results, missing_status, "VREF-P0-02 QKV URAM PYNQ smoke command", f"{command}", manifest_path)

        validation_command = manifest.get("validation_command")
        if isinstance(validation_command, str) and f"--preset {preset}" in validation_command:
            add(results, "ok", "VREF-P0-02 QKV URAM PYNQ smoke validation command", f"{preset} validator selected", manifest_path)
        else:
            add(results, missing_status, "VREF-P0-02 QKV URAM PYNQ smoke validation command", f"{validation_command}", manifest_path)

        bundle_errors = validate_bundle(bundle_dir, tar_path, variant)
        if bundle_errors:
            add(results, missing_status, "VREF-P0-02 QKV URAM PYNQ smoke bundle package", "; ".join(bundle_errors), manifest_path)
        else:
            add(results, "ok", "VREF-P0-02 QKV URAM PYNQ smoke bundle package", "tar contents and manifest file hashes validated", manifest_path)

    if session_json.exists():
        try:
            session = json.loads(session_json.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            add(results, missing_status, "VREF-P0-02 QKV URAM ZCU104 smoke session parse", f"invalid JSON: {exc}", session_json)
            return

        session_errors: list[str] = []
        expected_pairs = {
            "status": "pass",
            "target": "ZCU104 PYNQ",
            "variant": variant,
            "preset": preset,
            "expected_runtime_state": 2,
            "expected_out_raw": [58, -51, 42, -28, 36, -41],
            "result_json": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
            "validation_json": "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke_validation.json",
        }
        for key, expected in expected_pairs.items():
            if session.get(key) != expected:
                session_errors.append(f"{key}: {session.get(key)!r} != {expected!r}")

        canonical_path = session.get("canonical_result_path")
        if not isinstance(canonical_path, str) or not canonical_path.endswith("hardware/pynq/hgtxr/e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"):
            session_errors.append(f"canonical_result_path: {canonical_path!r}")

        tar = session.get("tar") if isinstance(session.get("tar"), dict) else {}
        if tar_path.exists():
            actual_hash = sha256_file(tar_path)
            if tar.get("sha256") != actual_hash:
                session_errors.append(f"tar.sha256: {tar.get('sha256')!r} != {actual_hash!r}")
            if tar.get("bytes") != tar_path.stat().st_size:
                session_errors.append(f"tar.bytes: {tar.get('bytes')!r} != {tar_path.stat().st_size!r}")
        else:
            session_errors.append(f"tar missing: {tar_path}")

        board_steps = session.get("board_steps") if isinstance(session.get("board_steps"), list) else []
        for token in [
            "tar -xzf e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle.tar.gz",
            "./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            "./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
        ]:
            if token not in board_steps:
                session_errors.append(f"board_steps missing {token!r}")

        host_steps = session.get("host_steps") if isinstance(session.get("host_steps"), list) else []
        host_text = "\n".join(str(step) for step in host_steps)
        for token in [
            "tools/import_pynq_smoke_result.py",
            "--preset axis-vref-p0-softmax-input-x2-qkv-uram",
            "tools/check_third_goal_preflight.py",
            "--mode final-signoff",
        ]:
            if token not in host_text:
                session_errors.append(f"host_steps missing {token!r}")

        bundle_errors = session.get("bundle_validation_errors")
        if bundle_errors not in ([], None):
            session_errors.append(f"bundle_validation_errors: {bundle_errors!r}")

        if session_errors:
            add(results, missing_status, "VREF-P0-02 QKV URAM ZCU104 smoke session contract", "; ".join(session_errors), session_json)
        else:
            add(results, "ok", "VREF-P0-02 QKV URAM ZCU104 smoke session contract", "board and host runbook fields validated", session_json)

    if session_md.exists():
        text = session_md.read_text()
        required = [
            "./run_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            "./validate_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.sh",
            "--preset axis-vref-p0-softmax-input-x2-qkv-uram",
        ]
        missing = [token for token in required if token not in text]
        if "HGTXR ZCU104 Smoke Session" not in text:
            missing.append("HGTXR ZCU104 Smoke Session")
        if missing:
            add(results, missing_status, "VREF-P0-02 QKV URAM ZCU104 smoke session markdown contract", f"missing {missing}", session_md)
        else:
            add(results, "ok", "VREF-P0-02 QKV URAM ZCU104 smoke session markdown contract", "runbook text includes board/import commands", session_md)


def check_c3b_physical_smoke_result(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    canonical_result = hardware / "pynq" / "hgtxr" / "e2e_axis_dma_c3b_mem16_file_smoke.json"
    noncanonical_candidates = [
        hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_smoke_bundle" / "e2e_axis_dma_c3b_mem16_file_smoke.json",
        hardware / "generated" / "pynq" / "e2e_axis_dma_c3b_mem16_file_smoke.remote.json",
    ]
    discovered = [path for path in noncanonical_candidates if path.exists()]
    if not canonical_result.exists():
        detail = (
            "not captured at canonical path yet; see generated/signoff/final_unblock_commands_2026_06_10.md; "
            "dry-run import: python3 tools/run_third_goal_final_signoff.py "
            "--root /home/kjm26/project/PRJXR/XR-VIT/HGTXR "
            "--import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json "
            "--dry-run-import-c3b-smoke --allow-blocked"
        )
        if discovered:
            detail += "; noncanonical result found but not accepted for final signoff until imported: " + ", ".join(
                str(path) for path in discovered
            )
        add(
            results,
            final_status(mode),
            "C3b AXIS/DMA physical smoke result",
            detail,
            canonical_result,
        )
        return

    try:
        payload = json.loads(canonical_result.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "C3b AXIS/DMA physical smoke result parse",
            f"invalid JSON: {exc}",
            canonical_result,
        )
        return

    errors = validate_result(payload, "axis-c3b-mem16")
    if errors:
        add(results, mode_status(mode, "warn", "fail"), "C3b AXIS/DMA physical smoke result", "; ".join(errors), canonical_result)
    else:
        add(results, "ok", "C3b AXIS/DMA physical smoke result", "validated pass for axis-c3b-mem16 canonical result", canonical_result)


def check_vref_successor_physical_smoke_result(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    preset = "axis-vref-p0-softmax-input-x2-dsp-mixed-stream"
    result_candidates = [
        hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json",
        (
            hardware
            / "generated"
            / "pynq"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_smoke_bundle"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.json"
        ),
        hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream_file_smoke.remote.json",
    ]
    existing = [path for path in result_candidates if path.exists()]
    if not existing:
        add(
            results,
            "warn",
            "VREF-P0 successor physical smoke result",
            "not captured yet; successor promotion remains gated by valid ZCU104 smoke JSON",
            result_candidates[0],
        )
        return

    result_path = existing[0]
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "VREF-P0 successor physical smoke result parse",
            f"invalid JSON: {exc}",
            result_path,
        )
        return

    errors = validate_result(payload, preset)
    if errors:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "VREF-P0 successor physical smoke result",
            "; ".join(errors),
            result_path,
        )
    else:
        add(results, "ok", "VREF-P0 successor physical smoke result", f"validated pass for {preset}", result_path)


def check_qkv_uram_successor_physical_smoke_result(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    preset = "axis-vref-p0-softmax-input-x2-qkv-uram"
    result_candidates = [
        hardware / "pynq" / "hgtxr" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json",
        (
            hardware
            / "generated"
            / "pynq"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_smoke_bundle"
            / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.json"
        ),
        hardware / "generated" / "pynq" / "e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram_file_smoke.remote.json",
    ]
    existing = [path for path in result_candidates if path.exists()]
    if not existing:
        add(
            results,
            "warn",
            "VREF-P0-02 QKV URAM physical smoke result",
            "not captured yet; QKV URAM successor promotion remains gated by valid ZCU104 smoke JSON",
            result_candidates[0],
        )
        return

    result_path = existing[0]
    try:
        payload = json.loads(result_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "VREF-P0-02 QKV URAM physical smoke result parse",
            f"invalid JSON: {exc}",
            result_path,
        )
        return

    errors = validate_result(payload, preset)
    if errors:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            "VREF-P0-02 QKV URAM physical smoke result",
            "; ".join(errors),
            result_path,
        )
    else:
        add(results, "ok", "VREF-P0-02 QKV URAM physical smoke result", f"validated pass for {preset}", result_path)


def check_smoke_candidate_discovery(
    results: list[dict[str, Any]],
    hardware: Path,
    mode: str,
    *,
    name: str,
    rel_path: str,
    expected_preset: str,
) -> None:
    discovery_path = hardware / rel_path
    if not discovery_path.exists():
        add(
            results,
            "warn",
            name,
            "discovery artifact missing; run final-signoff runner to refresh candidate scan",
            discovery_path,
        )
        return

    try:
        discovery = json.loads(discovery_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        add(
            results,
            mode_status(mode, "warn", "fail"),
            f"{name} parse",
            f"invalid JSON: {exc}",
            discovery_path,
        )
        return

    errors: list[str] = []
    if discovery.get("preset") != expected_preset:
        errors.append(f"preset={discovery.get('preset')!r}")
    if discovery.get("status") not in {"found", "missing"}:
        errors.append(f"status={discovery.get('status')!r}")
    safety = discovery.get("safety") if isinstance(discovery.get("safety"), dict) else {}
    for key in ["executes_commands", "creates_board_result", "creates_xr_vits_policy", "writes_canonical_inputs"]:
        if safety.get(key) is not False:
            errors.append(f"safety.{key}={safety.get(key)!r}")
    if errors:
        add(results, mode_status(mode, "warn", "fail"), name, "; ".join(errors), discovery_path)
    else:
        add(
            results,
            "ok",
            name,
            f"status={discovery.get('status')} pass={discovery.get('pass_count')} candidates={discovery.get('candidate_count')}",
            discovery_path,
        )


def check_smoke_candidate_discoveries(results: list[dict[str, Any]], hardware: Path, mode: str) -> None:
    checks = [
        (
            "VREF-P0 successor smoke candidate discovery",
            "generated/signoff/vref_successor_smoke_candidate_discovery_2026_06_10.json",
            "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
        ),
        (
            "C3b generic smoke candidate discovery",
            "generated/signoff/pynq_smoke_candidate_discovery_c3b_2026_06_16.json",
            "axis-c3b-mem16",
        ),
        (
            "VREF-P0 generic smoke candidate discovery",
            "generated/signoff/pynq_smoke_candidate_discovery_vref_p0_2026_06_16.json",
            "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
        ),
    ]
    for name, rel_path, preset in checks:
        check_smoke_candidate_discovery(
            results,
            hardware,
            mode,
            name=name,
            rel_path=rel_path,
            expected_preset=preset,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--json-out", default="")
    parser.add_argument(
        "--mode",
        choices=["neutral", "board-ready", "final-signoff"],
        default="neutral",
        help="neutral warns for choice-gated gaps; board-ready fails stale/missing board artifacts; final-signoff also requires physical smoke/reference inputs",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    hardware = root / "hardware"
    parent = root.parent
    results: list[dict[str, Any]] = []

    exists_check(results, "HGTXR root", root)
    exists_check(results, "hardware root", hardware)
    executable_check(results, "Vitis HLS 2023.2", Path("/tools/Xilinx/Vitis_HLS/2023.2/bin/vitis_hls"), args.mode, board_required=True)
    executable_check(results, "Vivado 2023.2", Path("/tools/Xilinx/Vivado/2023.2/bin/vivado"), args.mode, board_required=True)
    check_tool_on_path(results, "spec-kit")
    check_tool_on_path(results, "specify")

    requested_paper = parent / "PAPER_PRJXR" / "05_RESOURCES" / "DeiT-Tiny C-Syn Results.png"
    hgpipe_paper = parent / "HGPIPE" / "DeiT-Tiny C-Syn Results.png"
    exists_check(results, "requested PAPER_PRJXR DeiT image", requested_paper, missing_status=final_status(args.mode))
    exists_check(results, "HGPIPE substitute DeiT image", hgpipe_paper, missing_status="warn")
    check_xr_vits_reference(results, root, parent, args.mode)
    exists_check(results, "HGPIPE root", parent / "HGPIPE", missing_status="warn")

    config = hardware / "configs" / "zcu104_e2e_q4w8a_defines.h"
    exists_check(results, "ZCU104 E2E Q4W8A config", config)
    config_text = read_text(config)
    for macro, expected in [
        ("HGTXR_PARALLELISM_FACTOR", "8"),
        ("HGTXR_BUS_WIDTH", "256"),
        ("HGTXR_BIT_WIDTH", "8"),
        ("HGTXR_WEIGHT_BIT_WIDTH", "4"),
        ("HGTXR_BUFFER_SIZE", "256"),
        ("HGTXR_FIFO_DEPTH", "128"),
        ("HGTXR_E2E_ACTIVE_TOKENS", "196"),
        ("HGTXR_E2E_BLOCKS", "6"),
        ("HGTXR_E2E_FF_DIM", "768"),
        ("HGTXR_E2E_DENSE_PAR", "8"),
    ]:
        macro_check(results, config_text, macro, expected)

    check_generated_artifacts(results, hardware, args.mode)
    check_board_flow_mismatch(results, hardware, args.mode)
    check_pynq_overlay_artifacts(results, hardware, args.mode)
    check_e2e_weight_artifacts(results, hardware, args.mode)
    check_c3b_axis_dma_smoke_bundle(results, hardware, args.mode)
    check_c3b_smoke_session(results, hardware, args.mode)
    check_vref_successor_smoke_bundle(results, hardware, args.mode)
    check_qkv_uram_successor_smoke_artifacts(results, hardware, args.mode)
    check_smoke_candidate_discoveries(results, hardware, args.mode)
    check_c3b_physical_smoke_result(results, hardware, args.mode)
    check_vref_successor_physical_smoke_result(results, hardware, args.mode)
    check_qkv_uram_successor_physical_smoke_result(results, hardware, args.mode)

    status_order = {"fail": 0, "warn": 1, "ok": 2}
    for row in sorted(results, key=lambda item: (status_order[item["status"]], item["name"])):
        path = f" :: {row['path']}" if "path" in row else ""
        print(f"[{row['status']}] {row['name']}: {row['detail']}{path}")

    summary = {
        "ok": sum(1 for row in results if row["status"] == "ok"),
        "warn": sum(1 for row in results if row["status"] == "warn"),
        "fail": sum(1 for row in results if row["status"] == "fail"),
    }
    print(f"[summary] ok={summary['ok']} warn={summary['warn']} fail={summary['fail']}")

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"mode": args.mode, "summary": summary, "checks": results}, indent=2) + "\n")

    return 1 if summary["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
