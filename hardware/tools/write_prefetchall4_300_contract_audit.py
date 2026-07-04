#!/usr/bin/env python3
"""Audit the active 300 MHz prefetch-all4 design against the objective contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


HARDWARE_ROOT = Path(__file__).resolve().parents[1]
PROFILE = "par32_runtime_rom_only_dispatch_dsp3_headraw_addrsw_corefabric_normuram_compute_prefetchall4_300_mem16"
HLS_PROJECT = f"hgtxr_e2e_axis_{PROFILE}_no_board"
SOLUTION = "solution_e2e_q4w8a"
HLS_INCLUDE = HARDWARE_ROOT / "hls" / "include" / "hgtxr_e2e_vit.hpp"
HLS_TOP = HARDWARE_ROOT / "hls" / "src" / "hgtxr_e2e_axis_top.cpp"
HLS_TB = HARDWARE_ROOT / "hls" / "tb" / "tb_hgtxr_e2e_axis_top.cpp"
SYN_ROOT = HARDWARE_ROOT / "generated" / HLS_PROJECT / SOLUTION / "syn"
CSYNTH_REPORT = SYN_ROOT / "report" / "hgtxr_e2e_axis_top_csynth.rpt"
VERILOG_ROOT = SYN_ROOT / "verilog"
COMPONENT_XML = HARDWARE_ROOT / "generated" / HLS_PROJECT / SOLUTION / "impl" / "ip" / "component.xml"
DEFAULT_JSON = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_contract_audit_2026_06_29.json"
DEFAULT_MARKDOWN = HARDWARE_ROOT / "generated" / "signoff" / "prefetchall4_300_contract_audit_2026_06_29.md"


def paths_for_profile(profile: str) -> dict[str, Path]:
    hls_project = f"hgtxr_e2e_axis_{profile}_no_board"
    syn_root = HARDWARE_ROOT / "generated" / hls_project / SOLUTION / "syn"
    return {
        "syn_root": syn_root,
        "csynth_report": syn_root / "report" / "hgtxr_e2e_axis_top_csynth.rpt",
        "verilog_root": syn_root / "verilog",
        "component_xml": HARDWARE_ROOT / "generated" / hls_project / SOLUTION / "impl" / "ip" / "component.xml",
    }


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def add_check(checks: list[dict[str, Any]], name: str, ok: bool, detail: str, sources: list[str]) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if ok else "fail",
            "detail": detail,
            "sources": sources,
        }
    )


def contains_all(text: str, needles: Sequence[str]) -> bool:
    return all(needle in text for needle in needles)


def build_audit(profile: str = PROFILE) -> dict[str, Any]:
    profile_paths = paths_for_profile(profile)
    csynth_report = profile_paths["csynth_report"]
    verilog_root = profile_paths["verilog_root"]
    component_xml_path = profile_paths["component_xml"]
    include = read_text(HLS_INCLUDE)
    top = read_text(HLS_TOP)
    tb = read_text(HLS_TB)
    csynth = read_text(csynth_report)
    component_xml = read_text(component_xml_path)
    generated_top = read_text(verilog_root / "hgtxr_e2e_axis_top.v")
    verilog_files = {path.name for path in verilog_root.glob("*.v")}
    verilog_index = "\n".join(sorted(verilog_files))
    nonlinear_rom_generated = any(
        needle in verilog_index
        for needle in [
            "kGeluRom_ROM_1P_",
            "kExpRom_ROM_1P_",
            "kRsqrtRom_ROM_1P_",
        ]
    )
    full_transformer_generated = (
        all(
            name in verilog_index
            for name in [
                "hgtxr_e2e_layernorm",
                "hgtxr_e2e_project_qkv",
                "hgtxr_e2e_attention_core",
                "hgtxr_e2e_output_projection",
                "hgtxr_e2e_mlp_head",
            ]
        )
        and ("hgtxr_e2e_mlp_unit_0" in verilog_index or "hgtxr_e2e_mlp_unit_0_s" in verilog_index)
        and ("hgtxr_e2e_attn_unit_0" in verilog_index or "hgtxr_e2e_attn_unit_0_s" in verilog_index)
    )

    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        "source_files_available",
        all(file_exists(path) for path in [HLS_INCLUDE, HLS_TOP, HLS_TB, csynth_report]),
        "HLS include, top, testbench, and csynth report are readable.",
        [str(HLS_INCLUDE), str(HLS_TOP), str(HLS_TB), str(csynth_report)],
    )
    add_check(
        checks,
        "runtime_scheduler_present",
        contains_all(
            include,
            [
                "hgtxr_e2e_runtime_mode_from_control",
                "hgtxr_e2e_make_runtime_schedule",
                "schedule.mode",
                "HGTXR_MODE_SEARCH",
                "HGTXR_MODE_TRACK",
            ],
        ),
        "Runtime scheduler and Search/Track mode constants exist in the HLS source.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "frame_event_conv_modules_present",
        "hgtxr_conv_patch_embedding" in verilog_index and "hgtxr_event_conv_patch_embedding" in verilog_index,
        "Generated Verilog contains distinct frame and event convolution patch embedding modules.",
        [str(verilog_root)],
    )
    add_check(
        checks,
        "frame_event_conv_onchip_rom_range",
        (
            contains_all(
                include,
                [
                    "#if HGTXR_E2E_OMIT_WEIGHT_AXI",
                    "return elem_offset >= 0 && elem_offset < kRequiredWeightElems;",
                    "static constexpr int kPatchWeightElemBase = 0;",
                    "static constexpr int kBlockWeightElemBase =",
                ],
            )
            or contains_all(
                include,
                [
                    "const bool patch_weight =",
                    "elem_offset >= kPatchWeightElemBase && elem_offset < kBlockWeightElemBase",
                ],
            )
        )
        and "hgtxr_conv_patch_embedding" in verilog_index
        and "hgtxr_event_conv_patch_embedding" in verilog_index,
        "Active ROM-only path classifies required weights as on-chip ROM; generated frame and event convolution modules exist.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "head_onchip_rom_range",
        contains_all(
            include,
            [
                "const bool head_weight =",
                "elem_offset >= kHeadWeightElemBase",
                "elem_offset < kRequiredWeightElems",
                "hgtxr_e2e_mlp_head",
            ],
        )
        and "hgtxr_e2e_mlp_head" in verilog_index,
        "Head weight range is classified as on-chip ROM and the generated MLP head module exists.",
        [str(HLS_INCLUDE), str(verilog_root)],
    )
    add_check(
        checks,
        "track_transformer_onchip_rom_range",
        contains_all(
            include,
            [
                "hgtxr_e2e_is_track_rom_block_weight",
                "elem_offset < kBlockWeightElemBase",
                "elem_offset >= kHeadWeightElemBase",
                "block_idx < kTrackRomBlockPairs",
                "hgtxr_e2e_is_onchip_param_rom_weight(elem_offset)",
            ],
        ),
        "Track-path Transformer block weights are covered by the on-chip parameter ROM classifier.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "param_rom_storage_bound",
        contains_all(
            include,
            [
                "static const ap_int<HGTXR_WEIGHT_BIT_WIDTH> kParamRom[64]",
                "#pragma HLS bind_storage variable=kParamRom type=rom_1p impl=bram",
                "hgtxr_e2e_onchip_param_rom_raw",
            ],
        ),
        "Parameter pattern ROM is explicitly bound as BRAM ROM and feeds on-chip parameter reads.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "weight_axi_omitted_for_active_profile",
        (
            "gmem_e2e_runtime_m_axi" in csynth
            and "m_axi_gmem_e2e_weights" not in csynth
            and "m_axi_gmem_e2e_weights" not in generated_top
            and ("m_axi_gmem_e2e_weights" not in component_xml if component_xml else True)
        ),
        "Active csynth/top RTL omit the external weight AXI master and retain only the runtime AXI path.",
        [str(csynth_report), str(verilog_root / "hgtxr_e2e_axis_top.v"), str(component_xml_path)],
    )
    add_check(
        checks,
        "three_nonlinear_roms_bound",
        contains_all(
            include,
            [
                "hgtxr_e2e_gelu_rom",
                "hgtxr_e2e_softmax_exp_rom",
                "hgtxr_e2e_layernorm_rsqrt_rom",
                "variable=kGeluRom type=rom_1p impl=bram",
                "variable=kExpRom type=rom_1p impl=bram",
                "variable=kRsqrtRom type=rom_1p impl=bram",
            ],
        ),
        "GeLU, softmax-exp, and layernorm-rsqrt LUT paths are present and bound as BRAM ROMs.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "nonlinear_rom_generated",
        nonlinear_rom_generated,
        "Generated RTL contains at least one nonlinear ROM module; individual exp/rsqrt ROMs may be optimized or inlined depending on profile flags.",
        [str(verilog_root)],
    )
    add_check(
        checks,
        "search_dispatcher_prefetch_storage_uram",
        "gb.dispatch_prefetch" in top
        and "impl=uram" in top
        and "gb_dispatch_prefetch_RAM_2P_URAM_1R1W" in verilog_index,
        "Dispatcher prefetch bank is bound to URAM in HLS and appears as a generated URAM module.",
        [str(HLS_TOP), str(verilog_root)],
    )
    add_check(
        checks,
        "search_dispatcher_prefetch_all4",
        contains_all(
            include,
            [
                "HGTXR_E2E_PREFETCH_ALL_BEFORE_COMPUTE",
                "for (int prefetch_block = 0; prefetch_block < block_pairs; ++prefetch_block)",
                "hgtxr_e2e_search_weight_dispatch_prefetch(weights, gb, prefetch_block",
                "hgtxr_e2e_prefetch_trace_block_begin(block, block_pairs)",
            ],
        ),
        "Search mode supports prefetch-all-before-compute and traces block begin/end ordering.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "immediate_start_trace_gate",
        contains_all(
            include,
            [
                "prefetch_trace_interblock",
                "immediate=%d",
                "g_prefetch_trace_immediate_gaps",
                "HGTXR_E2E_CSIM_STRICT_IMMEDIATE_START",
                "assert(g_prefetch_trace_immediate_gaps == 0)",
            ],
        ),
        "C testbench trace gate asserts zero immediate-start gaps between Transformer blocks.",
        [str(HLS_INCLUDE)],
    )
    add_check(
        checks,
        "full_transformer_modules_generated",
        full_transformer_generated,
        "Generated RTL includes LayerNorm, QKV projection, attention, output projection, shared/runtime MLP, and head modules.",
        [str(verilog_root)],
    )
    add_check(
        checks,
        "csim_search_track_expected_outputs",
        contains_all(
            tb,
            [
                "kExpectedRaw",
                "search_runtime_state=%d expected=%d",
                "track_runtime_state=%d expected=%d",
                "e2e_axis_track_out",
            ],
        ),
        "The C testbench validates both Search and Track runtime states and output vectors.",
        [str(HLS_TB)],
    )
    add_check(
        checks,
        "csynth_active_profile_available",
        contains_all(
            csynth,
            [
                "Target device:  xczu7ev-ffvc1156-2-e",
                "ap_clk",
                "3.33 ns",
                "hgtxr_e2e_controller_run",
            ],
        ),
        "Active profile has a ZCU104-targeted 300 MHz HLS csynth report.",
        [str(csynth_report)],
    )

    failed = [check for check in checks if check["status"] != "pass"]
    return {
        "status": "pass" if not failed else "fail",
        "profile": profile,
        "summary": {
            "checks_total": len(checks),
            "checks_pass": len(checks) - len(failed),
            "checks_fail": len(failed),
        },
        "checks": checks,
        "sources": {
            "hls_include": str(HLS_INCLUDE),
            "hls_top": str(HLS_TOP),
            "hls_tb": str(HLS_TB),
            "csynth_report": str(csynth_report),
            "verilog_root": str(verilog_root),
            "component_xml": str(component_xml_path),
        },
        "interpretation": (
            "The active 300 MHz prefetch-all4 profile satisfies the structural objective contract "
            "for on-chip ROM parameter/nonlinear paths, runtime scheduler, Search dispatcher prefetch, "
            "and full Transformer module generation. Timing-clean, RTL-cosim, and board-measurement gates "
            "remain separate."
            if not failed
            else "One or more structural objective-contract checks failed; inspect failed checks before using this profile as the active objective candidate."
        ),
    }


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Prefetch-All4 300 MHz Objective Contract Audit",
        "",
        f"- status: `{audit['status']}`",
        f"- profile: `{audit['profile']}`",
        f"- checks: `{audit['summary']['checks_pass']}/{audit['summary']['checks_total']}` pass",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---:|---|",
    ]
    for check in audit["checks"]:
        detail = str(check["detail"]).replace("|", "\\|")
        lines.append(f"| `{check['name']}` | `{check['status']}` | {detail} |")
    lines.extend(["", "## Interpretation", "", audit["interpretation"], "", "## Sources", ""])
    for name, path in sorted(audit["sources"].items()):
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit active prefetch-all4 structural objective contract.")
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_audit(args.profile)
    json_out = args.json_out or DEFAULT_JSON
    markdown_out = args.markdown_out or DEFAULT_MARKDOWN
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    markdown_out.write_text(render_markdown(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
