#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import write_third_goal_completion_audit as audit_tool  # noqa: E402
import write_final_evidence_manifest as manifest_tool  # noqa: E402
from test_write_final_evidence_manifest import WriteFinalEvidenceManifestTests  # noqa: E402


def ok_check(name: str, detail: str = "ok") -> dict[str, str]:
    return {"status": "ok", "name": name, "detail": detail}


def fail_check(name: str, detail: str = "missing") -> dict[str, str]:
    return {"status": "fail", "name": name, "detail": detail}


def resource_policy_payload() -> dict[str, object]:
    checks = [
        "force_dsp_macro_default",
        "force_uram_macro_default",
        "small_mem_lutram_macro_default",
        "dsp_bind_op_present",
        "uram_bind_storage_present",
        "lutram_bind_storage_present",
        "zcu104_parallelism_default",
        "zcu104_dense_parallelism_default",
        "zcu104_fifo_depth_default",
        "resource_matrix_pass",
        "resource_matrix_recommends_c3b",
        "c3b_parallelism_16",
        "c3b_memory_banks_16",
        "c3b_dsp_increased_vs_a1",
        "c3b_lut_lower_than_c1",
        "c3b_uram_positive",
        "c3b_latency_not_worse_than_c1",
        "c3b_csynth_xml_exists",
        "c3b_csynth_resources_match_matrix",
        "c3b_csynth_latency_matches_matrix",
        "c3b_csynth_dsp_lte_threshold",
        "c3b_csynth_uram_lte_threshold",
    ]
    return {
        "status": "pass",
        "check_count": len(checks),
        "pass_count": len(checks),
        "fail_count": 0,
        "checks": [{"name": name, "status": "pass", "detail": "pass"} for name in checks],
    }


REQ6_REQUIRED_CHECKS = [
    "legal_bus_width_byte_aligned",
    "legal_bus_width_data_divisible",
    "legal_bus_width_weight_divisible",
    "legal_weight_width_lte_data_width",
    "legal_data_width_lt_acc_width",
    "legal_model_dim_divides_heads",
    "legal_head_dim_matches_model_heads",
    "legal_ff_dim_matches_mlp_ratio",
    "legal_dense_parallelism_matches_req6_parallelism",
    "legal_dense_parallelism_divides_embed",
    "legal_dense_parallelism_divides_ff_dim",
    "legal_weight_lanes_divide_dense_parallelism",
    "legal_buffer_size_positive",
    "legal_fifo_depth_positive",
    "e2e_static_assert_active_tokens_fit",
    "e2e_static_assert_heads_positive",
    "e2e_static_assert_head_dim_covers_embed",
    "e2e_static_assert_dense_par_positive",
    "e2e_static_assert_dense_par_divides_embed",
    "e2e_static_assert_dense_par_divides_ff_dim",
    "e2e_static_assert_weight_lanes_positive",
    "e2e_static_assert_dense_par_divides_weight_lanes",
    "e2e_static_assert_axis_width_matches_cyclic_axi",
    "csim_tcl_supports_par16_par32",
    "csynth_tcl_supports_par16_par32",
]


VREF_BUFFER_REQUIRED_CHECKS = [
    "force_uram_buffers_enabled",
    "small_mem_lutram_enabled",
    "axis_large_gb.q_uram",
    "axis_large_gb.k_uram",
    "axis_large_gb.v_uram",
    "axis_pooled_lutram",
    "attention_small_score_lutram",
    "attention_small_prob_lutram",
    "attention_small_exp_raw_lutram",
    "rmu_smu_small_score_lutram",
    "rmu_smu_small_prob_lutram",
    "qkv_cache_q_weight_cache_uram_successor_branch",
    "qkv_cache_k_weight_cache_uram_successor_branch",
    "qkv_cache_v_weight_cache_uram_successor_branch",
    "qkv_successor_file_present",
    "qkv_successor_status_pass",
    "qkv_successor_macro_uram_enabled",
    "qkv_successor_csim_pass",
    "qkv_successor_csynth_pass",
    "qkv_successor_overlay_routed",
    "qkv_successor_uram_increased_vs_dsp_mixed_stream",
    "qkv_successor_bram_reduced_vs_dsp_mixed_stream",
    "qkv_successor_uram_positive",
    "qkv_successor_physical_smoke_pending_only",
    "c3b_dsp_increased_vs_a1",
    "c3b_lut_lower_than_c1",
]


class WriteThirdGoalCompletionAuditTests(unittest.TestCase):
    def write_json(self, path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    def make_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name) / "XR-VIT" / "HGTXR"
        for rel in [
            "docs/track/PROGRESS.md",
            "docs/track/HANDOVER.md",
            "docs/track/HANDOVER-2026-06-10-E2E.md",
            "docs/track/CHOICE.md",
            "docs/track/log.md",
            "docs/Validation.md",
            "docs/Master-Plan.md",
            "docs/Spec.md",
            "docs/resources/hgpipe_reference_analysis_2026_06_08.md",
            "docs/resources/hgpipe_lut_math_contract_validation_2026_06_10.json",
            "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
            "docs/resources/final_evidence_manifest_2026_06_10.json",
            "docs/resources/c3b_smoke_result_contract_2026_06_10.json",
            "docs/resources/c3b_smoke_result_contract_2026_06_10.md",
            "docs/resources/selected_path_execution_audit_2026_06_10.md",
            "docs/resources/spec_plan_conformance_audit_2026_06_10.md",
            "docs/resources/e2e_resource_policy_audit_2026_06_10.json",
            "docs/resources/e2e_resource_policy_audit_2026_06_10.md",
            "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.bit",
            "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma_c3b_mem16.hwh",
            "hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
            "hardware/pynq/hgtxr/hgtxr_e2e_m_axi.bit",
            "hardware/pynq/hgtxr/hgtxr_e2e_axis_dma.bit",
            "hardware/hls/include/hgtxr_cyclic_norm.hpp",
            "hardware/hls/include/hgtxr_cyclic_math.hpp",
        ]:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}" if path.suffix == ".json" else "fixture")
        self.write_json(
            root / "docs/resources/final_operator_handoff_validation_2026_06_10.json",
            {"status": "pass", "fail_count": 0, "check_count": 52},
        )
        self.write_json(
            root / "docs/resources/req1_environment_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 13,
                "pass_count": 13,
                "fail_count": 0,
                "checks": [
                    {"name": "platform_linux", "status": "pass", "detail": "Linux"},
                    {"name": "ubuntu_id", "status": "pass", "detail": "ubuntu"},
                    {"name": "ubuntu_version_22_04", "status": "pass", "detail": "22.04"},
                    {"name": "not_wsl_kernel", "status": "pass", "detail": "generic"},
                    {"name": "hardware_root_expected_prefix", "status": "pass", "detail": "/home/kjm26/project"},
                    {"name": "no_legacy_wsl_or_xilinx_paths", "status": "pass", "detail": []},
                    {"name": "vitis_hls_tools_xilinx_executable", "status": "pass", "detail": "/tools/Xilinx"},
                    {"name": "vivado_tools_xilinx_executable", "status": "pass", "detail": "/tools/Xilinx"},
                ],
            },
        )
        self.write_json(
            root / "docs/resources/req9_deit_image_reference_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 11,
                "pass_count": 11,
                "fail_count": 0,
                "requested_image": {"exists": True, "sha256": "a" * 64},
                "safety": {
                    "executes_commands": False,
                    "executes_network": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
                "checks": [
                    {"name": "requested_image_exists", "status": "pass", "detail": "image"},
                    {"name": "requested_image_png_magic", "status": "pass", "detail": "89504e470d0a1a0a"},
                    {"name": "hgpipe_substitute_matches_requested_sha256", "status": "pass", "detail": "match"},
                    {"name": "hardware_docs_copy_matches_requested_sha256", "status": "pass", "detail": "match"},
                ],
            },
        )
        self.write_json(
            root / "docs/resources/final_signoff_bundle_validation_2026_06_10.json",
            {"status": "pass", "fail_count": 0, "pass_count": 150, "check_count": 150},
        )
        self.write_json(
            root / "docs/resources/xr_vits_reference_resolution_2026_06_10.json",
            {
                "status": "candidate-ready-needs-approval",
                "safety": {
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        self.write_json(
            root / "docs/resources/xr_vits_replacement_policy_preview_2026_06_10.json",
            {
                "status": "pass",
                "preview_only": True,
                "active_policy_written": False,
                "policy": {
                    "candidate_audit_fingerprint": "abc",
                    "candidate_audit_recommendation_snapshot": {"path": str(root.parent / "XR_Accel")},
                    "candidate_audit_meta": {"sha256": "abc"},
                    "policy_fingerprint": "def",
                },
                "integrity": {"status": "pass"},
                "safety": {
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                    "executes_commands": False,
                },
            },
        )
        (root / "docs/resources/xr_vits_replacement_policy_preview_2026_06_10.md").write_text("# Preview\n")
        self.write_json(
            root / "docs/resources/third_goal_requirements_trace_2026_06_10.json",
            {
                "status": "blocked",
                "blocked_requirement_ids": ["11"],
                "xr_vits_policy_integrity": {
                    "required_policy_fields_complete": True,
                    "legacy_policy_clears_final_signoff": False,
                },
            },
        )
        (root / "docs/resources/third_goal_requirements_trace_2026_06_10.md").write_text("# Trace\n")
        self.write_json(
            root / "docs/resources/final_blocker_closure_readiness_2026_06_10.json",
            {"status": "blocked"},
        )
        self.write_json(
            root / "docs/resources/c3b_smoke_candidate_discovery_2026_06_10.json",
            {"status": "missing", "pass_count": 0},
        )
        self.write_json(
            root / "docs/resources/c3b_smoke_result_contract_2026_06_10.json",
            {
                "status": "pass",
                "preset": "axis-c3b-mem16",
                "safety": {
                    "executes_commands": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        self.write_json(
            root / "docs/resources/final_unblock_closeout_packet_2026_06_10.json",
            {
                "status": "ready-for-operator-unblock",
                "qkv_uram_successor_gate": {
                    "status": "ready-for-physical-smoke",
                    "required_for_final_signoff": False,
                    "physical_smoke_status": "not_captured",
                    "physical_smoke_result_json": "",
                },
                "operator_commands": {
                    "qkv_uram_successor": [
                        "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
                        "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
                    ],
                },
                "safety": {
                    "executes_commands": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        self.write_json(
            root / "docs/resources/final_unblock_intake_2026_06_10.json",
            {
                "status": "blocked",
                "safety": {
                    "executes_commands": False,
                    "creates_board_result": False,
                    "creates_xr_vits_policy": False,
                    "writes_canonical_inputs": False,
                },
            },
        )
        self.write_json(
            root / "docs/resources/final_unblock_closeout_packet_validation_2026_06_10.json",
            {"status": "pass", "check_count": 49, "pass_count": 49, "fail_count": 0},
        )
        self.write_json(
            root / "docs/resources/selected_path_execution_audit_2026_06_10.json",
            {
                "status": "pass",
                "check_count": 22,
                "pass_count": 22,
                "fail_count": 0,
                "selection": {"path_1": "A2 then A1", "path_2": "C", "e": "pending"},
            },
        )
        self.write_json(
            root / "docs/resources/spec_plan_conformance_audit_2026_06_10.json",
            {"status": "pass", "check_count": 32, "pass_count": 32, "fail_count": 0},
        )
        self.write_json(
            root / "docs/resources/e2e_resource_policy_audit_2026_06_10.json",
            resource_policy_payload(),
        )
        self.write_json(
            root / "docs/resources/third_goal_source_audit_2026_06_16.json",
            {"status": "pass", "required_count": 74, "missing_required": []},
        )
        self.write_json(
            root / "docs/resources/third_goal_current_audit_2026_06_16.json",
            {
                "status": "blocked-external",
                "summary": {"requirements": 12},
                "items": [
                    {
                        "id": "2",
                        "status": "reflected",
                        "note": "Req2 gate status=pass-manual-spec-and-model-fallback.",
                    }
                ],
                "blocked_external_input_paths_by_blocker": {
                    "C3b AXIS/DMA physical smoke result": str(
                        root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
                    ),
                    "requested XR-VITs sibling": str(root.parent.parent / "XR-VITs"),
                },
                "xr_vits_policy_integrity": {
                    "consistent": True,
                    "operator_handoff_policy_check_count": 9,
                    "operator_handoff": {
                        "required": True,
                        "policy_exists": False,
                        "validation_status": "pending-policy-creation",
                        "required_policy_fields_complete": True,
                    },
                },
            },
        )
        self.write_json(
            root / "docs/resources/vref_p0_qkv_uram_cache_successor_2026_06_16.json",
            {
                "status": "pass",
                "csynth": {"resources": {"uram": 40}},
                "comparison": {"delta_vs_dsp_mixed_stream": {"bram_18k": -2}},
                "overlay": {
                    "route_status": {"fully_routed_nets": 10, "routable_nets": 10},
                    "checks": [
                        {"name": "timing_wns_gte_c3b", "status": "pass", "detail": "ok"},
                        {"name": "route_errors_zero", "status": "pass", "detail": "ok"},
                    ],
                },
                "checks": [
                    {"name": "latency_lte_c3b", "status": "pass", "detail": "ok"},
                    {"name": "dsp_lte_c3b", "status": "pass", "detail": "ok"},
                    {"name": "lut_lte_c3b", "status": "pass", "detail": "ok"},
                    {"name": "bram_reduced_vs_dsp_mixed_stream", "status": "pass", "detail": "ok"},
                ],
            },
        )
        self.write_json(
            root / "docs/resources/req5_q4q8_swhw_match_audit_2026_06_16.json",
            {"status": "pass", "precision": {"weight_bits": 4, "activation_bits": 8}},
        )
        self.write_json(
            root / "docs/resources/vref_p0_pot_scale_audit_2026_06_16.json",
            {"status": "pass", "check_count": 14, "pass_count": 14, "fail_count": 0},
        )
        (root / "docs/resources/vref_p0_pot_scale_audit_2026_06_16.md").write_text("# PoT audit\n")
        self.write_json(
            root / "docs/resources/vref_p0_pot_scale_sweep_2026_06_16.json",
            {
                "status": "pass",
                "spec_count": 3,
                "summary": {
                    "total_candidate_count": 45,
                    "total_fail_count": 0,
                    "specs_recommending_current": 3,
                    "next_action": "keep current PoT scales for C3b",
                },
            },
        )
        (root / "docs/resources/vref_p0_pot_scale_sweep_2026_06_16.md").write_text("# PoT sweep\n")
        self.write_json(
            root / "docs/resources/req6_parameterization_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 64,
                "fail_count": 0,
                "knobs": {
                    "config_macros": {
                        "HGTXR_TILING_FACTOR": 1,
                        "HGTXR_PARALLELISM_FACTOR": 8,
                        "HGTXR_BUS_WIDTH": 256,
                        "HGTXR_BIT_WIDTH": 8,
                        "HGTXR_WEIGHT_BIT_WIDTH": 4,
                        "HGTXR_BUFFER_SIZE": 256,
                        "HGTXR_FIFO_DEPTH": 128,
                    }
                },
                "checks": [{"name": name, "status": "pass", "detail": "pass"} for name in REQ6_REQUIRED_CHECKS],
            },
        )
        self.write_json(
            root / "docs/resources/hgpipe_operator_audit_2026_06_16.json",
            {
                "status": "pass",
                "contract_summary": {
                    "passed_ref_checks": 97,
                    "total_checked_samples": 5_899_008,
                    "total_ref_checks": 97,
                },
                "property_summary": {
                    "check_count": 211,
                    "pass_count": 211,
                    "fail_count": 0,
                    "status": "pass",
                },
                "operators": [
                    {"operator": "LayerNorm", "status": "pass", "property_fail_count": 0},
                    {"operator": "GeLU", "status": "pass", "property_fail_count": 0},
                    {"operator": "Softmax", "status": "pass", "property_fail_count": 0},
                    {"operator": "Quantization", "status": "pass", "property_fail_count": 0},
                ],
            },
        )
        (root / "docs/resources/hgpipe_operator_audit_2026_06_16.md").write_text("# HG-PIPE operator audit\n")
        self.write_json(
            root / "docs/resources/vref_p0_buffer_lifetime_audit_2026_06_16.json",
            {
                "status": "pass",
                "check_count": 51,
                "fail_count": 0,
                "checks": [
                    {"name": name, "status": "pass", "detail": "ok"}
                    for name in VREF_BUFFER_REQUIRED_CHECKS
                ],
            },
        )
        manifest_fixture = WriteFinalEvidenceManifestTests(methodName="test_build_manifest_hashes_required_artifacts")
        manifest_fixture.populate_required(root)
        self.write_json(
            root / "docs/resources/final_evidence_manifest_2026_06_10.json",
            {
                "status": "pass",
                "required_count": len(manifest_tool.REQUIRED_ARTIFACTS),
                "present_required_count": len(manifest_tool.REQUIRED_ARTIFACTS),
                "missing_required": [],
                "consistency_checks": [
                    {"name": "fixture_manifest_consistency", "status": "pass", "detail": "fixture"}
                ],
                "failed_consistency_checks": [],
            },
        )
        self.write_json(
            root / "docs/resources/third_goal_current_audit_2026_06_16.json",
            {
                "status": "blocked-external",
                "summary": {"requirements": 12},
                "items": [
                    {
                        "id": "2",
                        "status": "reflected",
                        "note": "Req2 gate status=pass-manual-spec-and-model-fallback.",
                    }
                ],
                "blocked_external_input_paths_by_blocker": {
                    "C3b AXIS/DMA physical smoke result": str(
                        root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
                    ),
                    "requested XR-VITs sibling": str(root.parent.parent / "XR-VITs"),
                },
                "xr_vits_policy_integrity": {
                    "consistent": True,
                    "operator_handoff_policy_check_count": 9,
                    "operator_handoff": {
                        "required": True,
                        "policy_exists": False,
                        "validation_status": "pending-policy-creation",
                        "required_policy_fields_complete": True,
                    },
                },
            },
        )
        return tmp, root

    def candidate_audit(self, root: Path) -> dict[str, object]:
        return {
            "recommendation": {
                "role": "candidate-xr-accel",
                "path": str(root.parent / "XR_Accel"),
            }
        }

    def preflight(self, *, fail: bool) -> dict[str, object]:
        checks = [
            ok_check("Vitis HLS 2023.2"),
            ok_check("Vivado 2023.2"),
            {"status": "warn", "name": "tool spec-kit", "detail": "not found on PATH"},
            ok_check("E2E m_axi weight contract"),
            ok_check("E2E m_axi weight expected raw", "[32, -13, 26, -6, 14, -11]"),
            ok_check("requested PAPER_PRJXR DeiT image"),
        ]
        for macro in [
            "macro HGTXR_PARALLELISM_FACTOR",
            "macro HGTXR_BUS_WIDTH",
            "macro HGTXR_BIT_WIDTH",
            "macro HGTXR_WEIGHT_BIT_WIDTH",
            "macro HGTXR_BUFFER_SIZE",
            "macro HGTXR_FIFO_DEPTH",
        ]:
            checks.append(ok_check(macro, "1"))
        if fail:
            checks.extend(
                [
                    fail_check("C3b AXIS/DMA physical smoke result", "not captured yet"),
                    fail_check("requested XR-VITs sibling", "missing; no approved replacement policy"),
                ]
            )
        else:
            checks.extend(
                [
                    ok_check("C3b AXIS/DMA physical smoke result", "validated"),
                    ok_check("requested XR-VITs sibling", "approved replacement"),
                ]
            )
        return {
            "mode": "final-signoff",
            "summary": {"ok": len([check for check in checks if check["status"] == "ok"]), "warn": 1, "fail": 2 if fail else 0},
            "checks": checks,
        }

    def test_build_audit_blocks_on_physical_smoke_and_xr_vits(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)

        audit = audit_tool.build_audit(root, self.preflight(fail=True), self.candidate_audit(root))

        self.assertEqual(audit["status"], "blocked")
        self.assertGreaterEqual(audit["blocked_count"], 2)
        by_id = {item["id"]: item for item in audit["items"]}
        self.assertIn("docs/track/HANDOVER.md: exists", by_id["0"]["evidence"])
        self.assertIn("docs/track/log.md: exists", by_id["0"]["evidence"])
        self.assertEqual(by_id["11"]["status"], "blocked")
        self.assertIn("policy integrity consistent: True", by_id["11"]["evidence"])
        self.assertIn("policy validation status: pending-policy-creation", by_id["11"]["evidence"])
        self.assertEqual(audit["xr_vits_policy_integrity"]["operator_handoff_policy_check_count"], 9)
        self.assertIn("requested XR-VITs sibling", audit["external_blocker_paths"])
        self.assertEqual(by_id["12"]["status"], "pass")
        self.assertEqual(by_id["final"]["status"], "blocked")

    def test_build_audit_can_pass_final_gate_when_failures_are_gone(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)
        (root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json").write_text("{}")

        audit = audit_tool.build_audit(root, self.preflight(fail=False), self.candidate_audit(root))

        self.assertEqual(audit["items"][-1]["status"], "pass")
        self.assertFalse(any(item["status"] == "blocked" for item in audit["items"]))

    def test_build_audit_blocks_on_failed_evidence_manifest_consistency(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)
        manifest_path = root / "docs/resources/final_evidence_manifest_2026_06_10.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["failed_consistency_checks"] = ["final_unblock_closeout_validation_pass"]
        manifest["consistency_checks"] = [
            {"name": "final_unblock_closeout_validation_pass", "status": "fail", "detail": "fixture"}
        ]
        self.write_json(manifest_path, manifest)

        audit = audit_tool.build_audit(root, self.preflight(fail=False), self.candidate_audit(root))

        by_id = {item["id"]: item for item in audit["items"]}
        self.assertEqual(by_id["12"]["status"], "blocked")
        self.assertIn("final_unblock_closeout_validation_pass", by_id["12"]["evidence"][2])
        self.assertIn("final_unblock_closeout_validation_pass", by_id["12"]["evidence"][3])

    def test_completion_self_gate_only_failure_does_not_block_req12(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)
        manifest_path = root / "docs/resources/final_evidence_manifest_2026_06_10.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["status"] = "fail"
        manifest["failed_consistency_checks"] = ["third_goal_completion_audit_req12_manifest_pass"]
        manifest["consistency_checks"] = [
            {"name": "third_goal_completion_audit_req12_manifest_pass", "status": "fail", "detail": "fixture"}
        ]
        self.write_json(manifest_path, manifest)

        audit = audit_tool.build_audit(root, self.preflight(fail=False), self.candidate_audit(root))

        by_id = {item["id"]: item for item in audit["items"]}
        self.assertEqual(by_id["12"]["status"], "pass")
        self.assertIn("external failed consistency checks: []", by_id["12"]["evidence"])

    def test_cli_writes_json_and_markdown(self) -> None:
        tmp, root = self.make_root()
        self.addCleanup(tmp.cleanup)
        preflight_path = root / "hardware/generated/signoff/third_goal_final_signoff_2026_06_10.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(self.preflight(fail=True)))
        candidate_path = root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json"
        candidate_path.write_text(json.dumps(self.candidate_audit(root)))
        json_out = root / "hardware/generated/signoff/third_goal_completion_audit_2026_06_10.json"
        markdown_out = root / "hardware/generated/signoff/third_goal_completion_audit_2026_06_10.md"
        stream = io.StringIO()

        with contextlib.redirect_stdout(stream):
            code = audit_tool.main(
                [
                    "--root",
                    str(root),
                    "--json-out",
                    str(json_out),
                    "--markdown-out",
                    str(markdown_out),
                ]
            )

        self.assertEqual(code, 1)
        self.assertEqual(json.loads(json_out.read_text())["status"], "blocked")
        markdown = markdown_out.read_text()
        self.assertIn("HGTXR Third Goal Completion Audit", markdown)
        self.assertIn("XR-VITs Policy Integrity", markdown)
        self.assertIn("External Blocker Paths", markdown)


if __name__ == "__main__":
    unittest.main()
