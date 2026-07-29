#!/usr/bin/env python3
"""Run both runtime-mode Search/Track ZCU104 smoke profiles and evaluate latency gate."""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Callable, Sequence

from check_runtime_mode_board_latency_gate import build_gate, render_markdown as render_gate_markdown
from check_runtime_mode_board_latency_gate import PROFILE_SET_CASES, PROFILE_SET_TARGETS
from run_zcu104_c3b_smoke_remote import PROFILE_CONFIGS, build_summary, default_runner, render_markdown


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
HGTXR_ROOT = HARDWARE_ROOT.parent
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "runtime_mode_board_latency_run_2026_06_28.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "runtime_mode_board_latency_run_2026_06_28.md"
RUNTIME_PROFILES = ("runtime-mode-par32-search", "runtime-mode-par32-track")
PAR32_ROM_COMPUTE_PROFILES = ("par32-rom-compute-300-search", "par32-rom-compute-300-track")
PAR32_PREFETCHALL4_PROFILES = ("par32-prefetchall4-300-search", "par32-prefetchall4-300-track")
PAR32_PATCH32_DTOK4_PROFILES = (
    "par32-patch32-dtok4-300-search",
    "par32-patch32-dtok4-300-track",
    "par32-patch32-dtok4-300-hybrid-10-90",
)
PROFILE_SET_PROFILES = {
    "runtime-mode-par32": RUNTIME_PROFILES,
    "par32-rom-compute-300": PAR32_ROM_COMPUTE_PROFILES,
    "par32-prefetchall4-300": PAR32_PREFETCHALL4_PROFILES,
    "par32-patch32-dtok4-300": PAR32_PATCH32_DTOK4_PROFILES,
}
PROFILE_SET_OUTPUTS = {
    "runtime-mode-par32": (DEFAULT_JSON, DEFAULT_MARKDOWN),
    "par32-rom-compute-300": (
        HARDWARE_ROOT / "generated" / "signoff" / "par32_rom_compute_300_board_latency_run_2026_06_29.json",
        HARDWARE_ROOT / "generated" / "signoff" / "par32_rom_compute_300_board_latency_run_2026_06_29.md",
    ),
    "par32-prefetchall4-300": (
        HARDWARE_ROOT / "generated" / "signoff" / "par32_prefetchall4_300_board_latency_run_2026_06_29.json",
        HARDWARE_ROOT / "generated" / "signoff" / "par32_prefetchall4_300_board_latency_run_2026_06_29.md",
    ),
    "par32-patch32-dtok4-300": (
        HARDWARE_ROOT / "generated" / "signoff" / "par32_patch32_dtok4_300_board_latency_run_2026_06_30.json",
        HARDWARE_ROOT / "generated" / "signoff" / "par32_patch32_dtok4_300_board_latency_run_2026_06_30.md",
    ),
}
Resolver = Callable[[str, int | None], Any]


def build_host_preflight(host: str, resolver: Resolver = socket.getaddrinfo) -> dict[str, Any]:
    try:
        infos = resolver(host, None)
    except OSError as exc:
        return {
            "status": "fail",
            "host": host,
            "addresses": [],
            "error": str(exc),
        }

    addresses = sorted({str(info[4][0]) for info in infos if len(info) >= 5 and info[4]})
    return {
        "status": "pass",
        "host": host,
        "addresses": addresses,
        "error": None,
    }


def build_runtime_mode_summary(
    *,
    root: Path,
    execute: bool,
    host: str,
    user: str | None,
    identity_file: Path | None,
    port: int | None,
    runner=default_runner,
    profile_configs: dict[str, dict[str, Any]] | None = None,
    profile_names: Sequence[str] = RUNTIME_PROFILES,
    profile_set: str = "runtime-mode-par32",
    resolver: Resolver = socket.getaddrinfo,
) -> dict[str, Any]:
    configs = PROFILE_CONFIGS if profile_configs is None else profile_configs
    host_preflight = build_host_preflight(host, resolver=resolver)
    execute_profiles = execute and host_preflight["status"] == "pass"
    profile_summaries: dict[str, Any] = {}
    for profile_name in profile_names:
        cfg = configs[profile_name]
        profile_summaries[profile_name] = build_summary(
            root=root,
            execute=execute_profiles,
            host=host,
            user=user,
            remote_dir=cfg["remote_dir"],
            tar_path=Path(cfg["tar"]),
            sha256_path=Path(cfg["sha256"]),
            local_result=Path(cfg["local_result"]),
            identity_file=identity_file,
            port=port,
            profile=profile_name,
            bundle_root=cfg["bundle_root"],
            result_json=cfg["result_json"],
            validation_json=cfg["validation_json"],
            run_script=cfg["run_script"],
            validate_script=cfg["validate_script"],
            preset=cfg["preset"],
            runner=runner,
        )

    gate = build_gate(
        root,
        cases_cfg=PROFILE_SET_CASES[profile_set],
        target=PROFILE_SET_TARGETS[profile_set],
        profile_set=profile_set,
    )
    remote_statuses = {name: summary["status"] for name, summary in profile_summaries.items()}
    errors = []
    if execute and host_preflight["status"] != "pass":
        errors.append(f"host unresolved: {host}: {host_preflight['error']}")
    for summary in profile_summaries.values():
        errors.extend(summary.get("errors") or [])
    if not execute:
        status = "dry-run" if not errors else "blocked-local-inputs"
    elif host_preflight["status"] != "pass":
        status = "host-unresolved"
    elif errors:
        status = "remote-failed"
    elif gate["status"] == "pass":
        status = "pass"
    else:
        status = gate["status"]

    return {
        "status": status,
        "execute": execute,
        "profile_set": profile_set,
        "target": PROFILE_SET_TARGETS[profile_set],
        "host": host,
        "user": user,
        "host_preflight": host_preflight,
        "profiles": profile_summaries,
        "remote_statuses": remote_statuses,
        "board_latency_gate": gate,
        "errors": errors,
    }


def render_combined_markdown(summary: dict[str, Any], profile_names: Sequence[str] | None = None) -> str:
    selected_profiles = list(summary["profiles"]) if profile_names is None else list(profile_names)
    lines = [
        "# Runtime-Mode Board Latency Run",
        "",
        f"- status: `{summary['status']}`",
        f"- execute: `{summary['execute']}`",
        f"- profile set: `{summary['profile_set']}`",
        f"- target: `{summary['target']}`",
        f"- host: `{summary['host']}`",
        f"- user: `{summary['user']}`",
        f"- host preflight: `{summary['host_preflight']['status']}`",
        "",
        "## Host Preflight",
        "",
        f"- host: `{summary['host_preflight']['host']}`",
        f"- addresses: `{', '.join(summary['host_preflight']['addresses']) or 'n/a'}`",
        f"- error: `{summary['host_preflight']['error'] or 'n/a'}`",
        "",
        "## Profile Status",
        "",
        "| Profile | Status | Preset | Remote dir |",
        "|---|---:|---|---|",
    ]
    for profile_name in selected_profiles:
        profile = summary["profiles"][profile_name]
        lines.append(
            f"| {profile_name} | `{profile['status']}` | `{profile['preset']}` | `{profile['remote_dir']}` |"
        )
    lines.extend(["", "## Board Latency Gate", "", render_gate_markdown(summary["board_latency_gate"]).strip(), ""])
    lines.extend(["## Remote Plans", ""])
    for profile_name in selected_profiles:
        lines.append(f"### {profile_name}")
        lines.append("")
        lines.append(render_markdown(summary["profiles"][profile_name]).strip())
        lines.append("")
    lines.extend(["## Errors", ""])
    if summary["errors"]:
        lines.extend(f"- `{error}`" for error in summary["errors"])
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def write_outputs(
    summary: dict[str, Any],
    json_out: Path,
    markdown_out: Path,
    profile_names: Sequence[str] | None = None,
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_combined_markdown(summary, profile_names=profile_names))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run both runtime-mode Search/Track board latency profiles.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--host", default="zcu104.local")
    parser.add_argument("--user", default="xilinx")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--identity-file", type=Path, default=None)
    parser.add_argument("--execute", action="store_true", help="run SSH/SCP commands; otherwise emit dry-run plans only")
    parser.add_argument("--profile-set", choices=sorted(PROFILE_SET_PROFILES), default="runtime-mode-par32")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    json_out = args.json_out
    markdown_out = args.markdown_out
    default_json, default_markdown = PROFILE_SET_OUTPUTS[args.profile_set]
    if args.json_out == DEFAULT_JSON:
        json_out = default_json
    if args.markdown_out == DEFAULT_MARKDOWN:
        markdown_out = default_markdown
    profile_names = PROFILE_SET_PROFILES[args.profile_set]
    summary = build_runtime_mode_summary(
        root=args.root.resolve(),
        execute=args.execute,
        host=args.host,
        user=args.user,
        identity_file=args.identity_file,
        port=args.port,
        profile_names=profile_names,
        profile_set=args.profile_set,
    )
    write_outputs(summary, json_out, markdown_out, profile_names=profile_names)
    print(
        f"[runtime-mode-board-latency] status={summary['status']} "
        f"execute={summary['execute']} gate={summary['board_latency_gate']['status']}"
    )
    return 0 if summary["status"] in {"dry-run", "pass"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
