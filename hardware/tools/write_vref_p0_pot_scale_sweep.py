#!/usr/bin/env python3
"""Evaluate VREF-P0-01 power-of-two scale candidates on reduced SW references."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from validate_e2e_axis_vector import (  # noqa: E402
    E2EMathConfig,
    build_reference,
    load_spec,
)


DATE_TAG = "2026_06_16"
HGTXR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPECS = [
    "refs/e2e_axis_vector_hgpipe_math_spec.json",
    "refs/e2e_axis_vector_hgpipe_math_lnq_spec.json",
    "refs/e2e_axis_vector_hgpipe_math_lnq_active16_spec.json",
]


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def candidate_profiles(base: E2EMathConfig) -> list[tuple[str, E2EMathConfig]]:
    return [
        ("current", base),
        ("gelu_input_x0p5", replace(base, geluq_input_scale=max(1, base.geluq_input_scale // 2))),
        ("gelu_input_x2", replace(base, geluq_input_scale=base.geluq_input_scale * 2)),
        ("gelu_output_x0p5", replace(base, geluq_output_scale=max(1, base.geluq_output_scale // 2))),
        ("gelu_output_x2", replace(base, geluq_output_scale=base.geluq_output_scale * 2)),
        ("softmax_input_x0p5", replace(base, softmax_input_scale=max(1, base.softmax_input_scale // 2))),
        ("softmax_input_x2", replace(base, softmax_input_scale=base.softmax_input_scale * 2)),
        ("softmax_prob_x0p5", replace(base, softmax_prob_scale=max(1, base.softmax_prob_scale // 2))),
        ("softmax_prob_x2", replace(base, softmax_prob_scale=base.softmax_prob_scale * 2)),
        ("layernorm_input_x0p5", replace(base, layernorm_input_scale=max(1, base.layernorm_input_scale // 2))),
        ("layernorm_input_x2", replace(base, layernorm_input_scale=base.layernorm_input_scale * 2)),
        ("layernorm_output_x0p5", replace(base, layernorm_output_scale=max(1, base.layernorm_output_scale // 2))),
        ("layernorm_output_x2", replace(base, layernorm_output_scale=base.layernorm_output_scale * 2)),
        (
            "all_input_x2",
            replace(
                base,
                geluq_input_scale=base.geluq_input_scale * 2,
                softmax_input_scale=base.softmax_input_scale * 2,
                layernorm_input_scale=base.layernorm_input_scale * 2,
            ),
        ),
        (
            "all_output_x2",
            replace(
                base,
                geluq_output_scale=base.geluq_output_scale * 2,
                softmax_prob_scale=base.softmax_prob_scale * 2,
                layernorm_output_scale=base.layernorm_output_scale * 2,
            ),
        ),
    ]


def scale_fields(math_cfg: E2EMathConfig) -> dict[str, int]:
    return {
        "geluq_input_scale": math_cfg.geluq_input_scale,
        "geluq_output_scale": math_cfg.geluq_output_scale,
        "softmax_input_scale": math_cfg.softmax_input_scale,
        "softmax_prob_scale": math_cfg.softmax_prob_scale,
        "layernorm_input_scale": math_cfg.layernorm_input_scale,
        "layernorm_output_scale": math_cfg.layernorm_output_scale,
    }


def score_candidate(
    name: str,
    math_cfg: E2EMathConfig,
    baseline_raw: list[int],
    baseline_head: list[float],
    cfg: Any,
    pattern: dict[str, Any],
) -> dict[str, Any]:
    try:
        ref = build_reference(cfg, math_cfg, pattern)
    except Exception as exc:  # pragma: no cover - defensive report path
        return {
            "candidate": name,
            "status": "fail",
            "error": str(exc),
            "math": asdict(math_cfg),
            "scales": scale_fields(math_cfg),
        }

    raw_delta = [actual - base for actual, base in zip(ref.expected_raw, baseline_raw)]
    head_delta = [actual - base for actual, base in zip(ref.head_float, baseline_head)]
    scales = scale_fields(math_cfg)
    scale_pot = all(is_power_of_two(value) for value in scales.values())
    max_raw_abs_delta = max((abs(value) for value in raw_delta), default=0)
    raw_l1_delta = sum(abs(value) for value in raw_delta)
    head_l1_delta = sum(abs(value) for value in head_delta)
    return {
        "candidate": name,
        "status": "pass" if scale_pot else "fail",
        "math": asdict(math_cfg),
        "scales": scales,
        "all_scales_power_of_two": scale_pot,
        "expected_raw": ref.expected_raw,
        "runtime_state": ref.runtime_state,
        "raw_delta": raw_delta,
        "raw_l1_delta": raw_l1_delta,
        "max_raw_abs_delta": max_raw_abs_delta,
        "head_l1_delta": head_l1_delta,
        "exact_raw_match": ref.expected_raw == baseline_raw,
        "nonzero_counts": ref.nonzero_counts,
    }


def evaluate_spec(spec_path: Path) -> dict[str, Any]:
    cfg, math_cfg, pattern = load_spec(spec_path)
    baseline = build_reference(cfg, math_cfg, pattern)
    candidates = [
        score_candidate(name, candidate, baseline.expected_raw, baseline.head_float, cfg, pattern)
        for name, candidate in candidate_profiles(math_cfg)
    ]
    passing = [row for row in candidates if row["status"] == "pass"]
    exact = [row for row in passing if row.get("exact_raw_match")]
    best = min(
        passing,
        key=lambda row: (
            0 if row.get("candidate") == "current" else 1,
            row.get("raw_l1_delta", 10**9),
            row.get("max_raw_abs_delta", 10**9),
            row.get("candidate", ""),
        ),
    )
    return {
        "spec": str(spec_path),
        "config": asdict(cfg),
        "baseline_math": asdict(math_cfg),
        "baseline_expected_raw": baseline.expected_raw,
        "baseline_runtime_state": baseline.runtime_state,
        "candidate_count": len(candidates),
        "pass_count": len(passing),
        "fail_count": len(candidates) - len(passing),
        "exact_match_count": len(exact),
        "recommended_candidate": best["candidate"],
        "recommendation": "keep_current_scales" if best["candidate"] == "current" else "requires_header_regen_and_csim",
        "candidates": sorted(
            candidates,
            key=lambda row: (
                row.get("status") != "pass",
                row.get("raw_l1_delta", 10**9),
                row.get("max_raw_abs_delta", 10**9),
                row.get("candidate", ""),
            ),
        ),
    }


def build_sweep(root: Path, specs: Sequence[Path] | None = None) -> dict[str, Any]:
    root = root.resolve()
    hardware = root / "hardware" if root.name != "hardware" else root
    selected_specs = list(specs) if specs else [hardware / rel for rel in DEFAULT_SPECS]
    spec_results = [evaluate_spec(path if path.is_absolute() else hardware / path) for path in selected_specs]
    fail_count = sum(result["fail_count"] for result in spec_results)
    exact_specs = sum(1 for result in spec_results if result["recommendation"] == "keep_current_scales")
    return {
        "status": "pass" if fail_count == 0 and exact_specs == len(spec_results) else "review",
        "date_tag": DATE_TAG,
        "root": str(root if root.name != "hardware" else root.parent),
        "hardware": str(hardware),
        "policy": {
            "vref": "VREF-P0-01",
            "source": "P2-ViT",
            "scope": "SW-only power-of-two scale candidate sweep for reduced E2E vector references",
            "promotion_gate": "candidate must regenerate golden header and pass CSim before HLS macro promotion",
            "c3b_protection": "no HLS/Vivado run and no C3b artifact overwrite",
        },
        "spec_count": len(spec_results),
        "specs": spec_results,
        "summary": {
            "specs_recommending_current": exact_specs,
            "total_candidate_count": sum(result["candidate_count"] for result in spec_results),
            "total_fail_count": fail_count,
            "next_action": "keep current PoT scales for C3b; use non-current rows only as explicit successor experiments",
        },
        "safety": {
            "executes_hls": False,
            "executes_vivado": False,
            "writes_hls_source": False,
            "overwrites_c3b_artifacts": False,
        },
    }


def render_markdown(sweep: dict[str, Any]) -> str:
    lines = [
        "# VREF-P0-01 PoT Scale Candidate Sweep",
        "",
        f"- status: `{sweep['status']}`",
        f"- root: `{sweep['root']}`",
        f"- specs: `{sweep['spec_count']}`",
        f"- total candidates: `{sweep['summary']['total_candidate_count']}`",
        f"- total fail_count: `{sweep['summary']['total_fail_count']}`",
        f"- next_action: `{sweep['summary']['next_action']}`",
        "",
        "## Policy",
        "",
    ]
    for key, value in sweep["policy"].items():
        lines.append(f"- {key}: `{value}`")
    for result in sweep["specs"]:
        lines.extend(
            [
                "",
                f"## {Path(result['spec']).name}",
                "",
                f"- baseline_expected_raw: `{result['baseline_expected_raw']}`",
                f"- baseline_runtime_state: `{result['baseline_runtime_state']}`",
                f"- recommended_candidate: `{result['recommended_candidate']}`",
                f"- recommendation: `{result['recommendation']}`",
                f"- candidates: `{result['pass_count']}/{result['candidate_count']} pass`",
                "",
                "| Candidate | Status | Raw L1 Delta | Max Raw Delta | Exact Raw Match | Scales |",
                "|---|---|---:|---:|---|---|",
            ]
        )
        for row in result["candidates"][:8]:
            scales = ", ".join(f"{key}={value}" for key, value in row.get("scales", {}).items())
            lines.append(
                f"| {row['candidate']} | `{row['status']}` | {row.get('raw_l1_delta', 'NA')} | "
                f"{row.get('max_raw_abs_delta', 'NA')} | `{row.get('exact_raw_match', False)}` | {scales} |"
            )
    lines.extend(["", "## Safety", ""])
    for key, value in sweep["safety"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write VREF-P0-01 PoT scale candidate sweep.")
    parser.add_argument("--root", type=Path, default=HGTXR_ROOT)
    parser.add_argument("--spec", type=Path, action="append", default=[])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sweep = build_sweep(args.root, args.spec)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(sweep, indent=2, sort_keys=True) + "\n")
    args.markdown_out.write_text(render_markdown(sweep))
    print(json.dumps({"status": sweep["status"], **sweep["summary"]}, sort_keys=True))
    return 0 if sweep["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
