#!/usr/bin/env python3
"""Check runtime-mode Search/Track board latency evidence for the E2E AXIS top."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from validate_pynq_smoke_result import load_json, validate_result


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "runtime_mode_board_latency_gate_2026_06_28.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "runtime_mode_board_latency_gate_2026_06_28.md"
PAR32_ROM_COMPUTE_JSON = HARDWARE_ROOT / "generated" / "signoff" / "par32_rom_compute_300_board_latency_gate_2026_06_29.json"
PAR32_ROM_COMPUTE_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "par32_rom_compute_300_board_latency_gate_2026_06_29.md"
PAR32_PREFETCHALL4_JSON = HARDWARE_ROOT / "generated" / "signoff" / "par32_prefetchall4_300_board_latency_gate_2026_06_29.json"
PAR32_PREFETCHALL4_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "par32_prefetchall4_300_board_latency_gate_2026_06_29.md"
PAR32_PATCH32_DTOK4_JSON = HARDWARE_ROOT / "generated" / "signoff" / "par32_patch32_dtok4_300_board_latency_gate_2026_06_30.json"
PAR32_PATCH32_DTOK4_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "par32_patch32_dtok4_300_board_latency_gate_2026_06_30.md"

CASES = {
    "search": {
        "preset": "axis-runtime-mode-par32-search",
        "target_ms": 4.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_runtime_mode_par32_search_file_smoke.json"),
    },
    "track": {
        "preset": "axis-runtime-mode-par32-track",
        "target_ms": 1.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_runtime_mode_par32_track_file_smoke.json"),
    },
}

PAR32_ROM_COMPUTE_CASES = {
    "search": {
        "preset": "axis-par32-rom-compute-300-search",
        "target_ms": 4.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_search_file_smoke.json"),
    },
    "track": {
        "preset": "axis-par32-rom-compute-300-track",
        "target_ms": 1.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_par32_rom_compute_300_track_file_smoke.json"),
    },
}

PAR32_PREFETCHALL4_CASES = {
    "search": {
        "preset": "axis-par32-prefetchall4-300-search",
        "target_ms": 4.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_search_file_smoke.json"),
    },
    "track": {
        "preset": "axis-par32-prefetchall4-300-track",
        "target_ms": 1.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_par32_prefetchall4_300_track_file_smoke.json"),
    },
}

PAR32_PATCH32_DTOK4_CASES = {
    "search": {
        "preset": "axis-par32-patch32-dtok4-300-search",
        "target_ms": 4.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_par32_patch32_dtok4_300_search_file_smoke.json"),
    },
    "track": {
        "preset": "axis-par32-patch32-dtok4-300-track",
        "target_ms": 1.0,
        "path": Path("hardware/pynq/hgtxr/e2e_axis_dma_par32_patch32_dtok4_300_track_file_smoke.json"),
    },
}

PROFILE_SET_CASES = {
    "runtime-mode-par32": CASES,
    "par32-rom-compute-300": PAR32_ROM_COMPUTE_CASES,
    "par32-prefetchall4-300": PAR32_PREFETCHALL4_CASES,
    "par32-patch32-dtok4-300": PAR32_PATCH32_DTOK4_CASES,
}

PROFILE_SET_TARGETS = {
    "runtime-mode-par32": "ZCU104 PYNQ runtime-mode E2E AXIS board latency",
    "par32-rom-compute-300": "ZCU104 PYNQ PAR32 ROM compute 300MHz board latency",
    "par32-prefetchall4-300": "ZCU104 PYNQ PAR32 prefetch-all4 300MHz board latency",
    "par32-patch32-dtok4-300": "ZCU104 PYNQ PAR32 patch32 dense-token4 300MHz board latency",
}


def case_status(root: Path, name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    path = root / cfg["path"]
    result: dict[str, Any] = {
        "case": name,
        "preset": cfg["preset"],
        "target_ms": cfg["target_ms"],
        "path": str(path),
        "exists": path.exists(),
        "status": "missing",
        "errors": [],
        "accelerator_latency_ms_summary": None,
    }
    if not path.exists():
        result["errors"].append(f"missing board result JSON: {path}")
        return result
    try:
        payload = load_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        result["status"] = "fail"
        result["errors"].append(f"invalid board result JSON: {exc}")
        return result
    errors = validate_result(payload, cfg["preset"])
    result["status"] = "pass" if not errors else "fail"
    result["errors"] = errors
    result["accelerator_latency_ms_summary"] = payload.get("accelerator_latency_ms_summary")
    result["runtime_state"] = payload.get("runtime_state")
    result["mode_profile"] = payload.get("mode_profile")
    return result


def build_gate(
    root: Path,
    *,
    cases_cfg: dict[str, dict[str, Any]] | None = None,
    target: str | None = None,
    profile_set: str = "runtime-mode-par32",
) -> dict[str, Any]:
    selected_cases = CASES if cases_cfg is None else cases_cfg
    cases = {name: case_status(root, name, cfg) for name, cfg in selected_cases.items()}
    statuses = [case["status"] for case in cases.values()]
    if all(status == "pass" for status in statuses):
        status = "pass"
    elif any(status == "fail" for status in statuses):
        status = "fail"
    else:
        status = "missing"
    return {
        "status": status,
        "profile_set": profile_set,
        "target": target or PROFILE_SET_TARGETS.get(profile_set, "ZCU104 PYNQ E2E AXIS board latency"),
        "summary": {
            "search_target_ms": selected_cases["search"]["target_ms"],
            "track_target_ms": selected_cases["track"]["target_ms"],
            "cases_total": len(cases),
            "cases_pass": sum(1 for case in cases.values() if case["status"] == "pass"),
            "cases_missing": sum(1 for case in cases.values() if case["status"] == "missing"),
            "cases_fail": sum(1 for case in cases.values() if case["status"] == "fail"),
        },
        "cases": cases,
    }


def render_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# Runtime-Mode Board Latency Gate",
        "",
        f"- status: `{gate['status']}`",
        f"- target: `{gate['target']}`",
        f"- search target: `{gate['summary']['search_target_ms']} ms`",
        f"- track target: `{gate['summary']['track_target_ms']} ms`",
        "",
        "## Cases",
        "",
        "| Case | Status | Target ms | Runtime state | Mode profile | Max accelerator latency ms | Path |",
        "|---|---:|---:|---:|---|---:|---|",
    ]
    for name in ["search", "track"]:
        case = gate["cases"][name]
        summary = case.get("accelerator_latency_ms_summary") if isinstance(case, dict) else None
        max_latency = summary.get("max") if isinstance(summary, dict) else ""
        lines.append(
            "| {case} | `{status}` | {target} | {runtime} | {profile} | {latency} | `{path}` |".format(
                case=name,
                status=case["status"],
                target=case["target_ms"],
                runtime=case.get("runtime_state", ""),
                profile=case.get("mode_profile", ""),
                latency=max_latency,
                path=case["path"],
            )
        )
    errors = []
    for case in gate["cases"].values():
        errors.extend(case.get("errors") or [])
    lines.extend(["", "## Errors", ""])
    if errors:
        lines.extend(f"- `{error}`" for error in errors)
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def write_outputs(gate: dict[str, Any], json_out: Path, markdown_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(gate))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime-mode Search/Track board latency evidence.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--profile-set", choices=sorted(PROFILE_SET_CASES), default="runtime-mode-par32")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_out = args.json_out
    markdown_out = args.markdown_out
    if args.profile_set == "par32-rom-compute-300" and json_out == DEFAULT_JSON:
        json_out = PAR32_ROM_COMPUTE_JSON
    if args.profile_set == "par32-rom-compute-300" and markdown_out == DEFAULT_MARKDOWN:
        markdown_out = PAR32_ROM_COMPUTE_MARKDOWN
    if args.profile_set == "par32-prefetchall4-300" and json_out == DEFAULT_JSON:
        json_out = PAR32_PREFETCHALL4_JSON
    if args.profile_set == "par32-prefetchall4-300" and markdown_out == DEFAULT_MARKDOWN:
        markdown_out = PAR32_PREFETCHALL4_MARKDOWN
    if args.profile_set == "par32-patch32-dtok4-300" and json_out == DEFAULT_JSON:
        json_out = PAR32_PATCH32_DTOK4_JSON
    if args.profile_set == "par32-patch32-dtok4-300" and markdown_out == DEFAULT_MARKDOWN:
        markdown_out = PAR32_PATCH32_DTOK4_MARKDOWN
    gate = build_gate(
        args.root.resolve(),
        cases_cfg=PROFILE_SET_CASES[args.profile_set],
        target=PROFILE_SET_TARGETS[args.profile_set],
        profile_set=args.profile_set,
    )
    write_outputs(gate, json_out, markdown_out)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
