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

import write_final_evidence_manifest as manifest_tool  # noqa: E402


class WriteFinalEvidenceManifestTests(unittest.TestCase):
    def write_validator_contract(self, root: Path, *, omit: str = "") -> None:
        prefixes = [
            "hgtxr_e2e_axis_dma_c3b_mem16",
            "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_dsp_mixed_stream",
            "hgtxr_e2e_axis_dma_vref_p0_softmax_input_x2_qkv_uram",
            "hgtxr_e2e_m_axi",
        ]
        text = "\n".join(
            [
                "from pathlib import Path",
                "def expected_artifact_name(preset, field): return preset['artifact_prefix']",
                "def validate_result(payload, preset_name, require_paths=True):",
                "    if require_paths:",
                "        value = payload.get('bitfile')",
                "        actual_name = Path(str(value)).name",
                "        artifact_prefix = 'present'",
            ]
            + [f"        # {prefix}" for prefix in prefixes if prefix != omit]
        )
        path = root / "hardware" / "tools" / "validate_pynq_smoke_result.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")

    def write_runner_contract(self, root: Path, *, omit: str = "") -> None:
        lines = [
            "def build_remaining_blocker_details():",
            "    remaining_blocker_input_paths = {}",
            "    remaining_blocker_details = {}",
            "    blocker_count = len(remaining_blockers)",
            "    remaining_blocker_detail_count = len(remaining_blocker_details)",
            "    kind_a = 'exact-source-or-approved-replacement-policy'",
            "    kind_b = 'canonical-pynq-smoke-json'",
            "    if approve_xr_vits_replacement and not dry_run_xr_vits_replacement and xr_vits_policy_status == \"fail\":",
            "        status = 'blocked'",
            "    summary[\"mirrored_artifacts\"].extend([str(paths[\"doc_summary_json\"]), str(paths[\"doc_summary_md\"])])",
            "    summary[\"mirrored_artifact_count\"] = len(summary[\"mirrored_artifacts\"])",
            "    summary[\"mirrored_artifact_unique_count\"] = len(set(summary[\"mirrored_artifacts\"]))",
            "    summary[\"mirrored_artifact_duplicate_count\"] = summary[\"mirrored_artifact_count\"] - summary[\"mirrored_artifact_unique_count\"]",
            "    doc_summary_json = paths[\"doc_summary_json\"]",
            "    doc_summary_md = paths[\"doc_summary_md\"]",
            "    summary[\"mirrored_artifacts\"] = unique_ordered(summary[\"mirrored_artifacts\"])",
            "def build_mirror_integrity():",
            "    canonical_evidence_root = root / 'docs/resources'",
            "    generated_signoff_root = root / 'hardware/generated/signoff'",
            "    mirrored_artifact_integrity_status = 'pass'",
            "    mirrored_artifact_integrity_checked_count = 1",
            "    sha256_match = True",
        ]
        text = "\n".join(line for line in lines if not omit or omit not in line)
        path = root / "hardware" / "tools" / "run_third_goal_final_signoff.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")

    def write_policy_tool_contract(self, root: Path, *, omit: str = "") -> None:
        lines = [
            "PLACEHOLDER_APPROVER_VALUES = {'<approved-by>', '<name>', 'todo', 'tbd'}",
            "def approver_is_placeholder(value): return value in PLACEHOLDER_APPROVER_VALUES",
            "def validate_inputs(approved_by, dry_run):",
            "    if not dry_run and approver_is_placeholder(approved_by):",
            "        errors.append('--approved-by must be a real approver identifier for active policy writes')",
            "def main(args):",
            "    validate_inputs(args.approved_by, dry_run=args.dry_run)",
        ]
        text = "\n".join(line for line in lines if not omit or omit not in line)
        path = root / "hardware" / "tools" / "create_xr_vits_replacement_policy.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")

    def write_runner_summary(self, root: Path, *, omit_details: bool = False, empty_paths: bool = False) -> None:
        c3b_path = str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json")
        xr_exact = str(root.parent.parent / "XR-VITs")
        xr_policy = str(root / "docs/resources/xr_vits_replacement_policy.json")
        final_manifest = str(root / f"docs/resources/final_evidence_manifest_{manifest_tool.DATE_TAG}.json")
        summary = str(root / f"docs/resources/third_goal_final_signoff_run_{manifest_tool.DATE_TAG}.json")
        input_paths = {
            "C3b AXIS/DMA physical smoke result": [] if empty_paths else [c3b_path],
            "requested XR-VITs sibling": [] if empty_paths else [xr_exact, xr_policy],
        }
        details = {
            "C3b AXIS/DMA physical smoke result": {
                "kind": "canonical-pynq-smoke-json",
                "paths": [c3b_path],
                "status": "missing",
                "would_clear": False,
            },
            "requested XR-VITs sibling": {
                "kind": "exact-source-or-approved-replacement-policy",
                "paths": [xr_exact, xr_policy],
                "mode": "missing",
                "would_clear": False,
            },
        }
        payload = {
            "status": "blocked",
            "remaining_blockers": list(input_paths),
            "blocker_count": len(input_paths),
            "remaining_blocker_input_paths": input_paths,
            "remaining_blocker_detail_count": len(details) if not omit_details else 0,
            "c3b_transfer_manifest_status": "pass",
            "mirrored_artifacts": [
                str(root / f"docs/resources/c3b_board_smoke_readiness_{manifest_tool.DATE_TAG}.json"),
                final_manifest,
                summary,
            ],
        }
        payload["mirrored_artifact_count"] = len(payload["mirrored_artifacts"])
        payload["mirrored_artifact_unique_count"] = len(set(payload["mirrored_artifacts"]))
        payload["mirrored_artifact_duplicate_count"] = (
            payload["mirrored_artifact_count"] - payload["mirrored_artifact_unique_count"]
        )
        payload["mirrored_artifact_integrity"] = [
            {
                "status": "pass",
                "source": str(root / "hardware" / "generated" / "signoff" / Path(path).name),
                "mirror": path,
                "source_exists": True,
                "mirror_exists": True,
                "sha256_match": True,
            }
            for path in payload["mirrored_artifacts"]
        ]
        payload["mirrored_artifact_integrity_status"] = "pass"
        payload["mirrored_artifact_integrity_count"] = len(payload["mirrored_artifacts"])
        payload["mirrored_artifact_integrity_checked_count"] = len(payload["mirrored_artifacts"])
        payload["mirrored_artifact_integrity_excluded_count"] = 0
        payload["mirrored_artifact_integrity_fail_count"] = 0
        payload["mirrored_artifact_integrity_failures"] = []
        if not omit_details:
            payload["remaining_blocker_details"] = details
        path = root / "hardware" / "generated" / "signoff" / f"third_goal_final_signoff_run_{manifest_tool.DATE_TAG}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n")

    def blocker_discovery_summary(self, preset: str) -> dict[str, object]:
        return {
            "status": "missing",
            "preset": preset,
            "candidate_count": 0,
            "pass_count": 0,
            "recommended_candidate_path": "",
            "dry_run_import_command": "",
            "import_command": "",
            "safety": {
                "executes_commands": False,
                "creates_board_result": False,
                "creates_xr_vits_policy": False,
                "writes_canonical_inputs": False,
            },
        }

    def payload_for_artifact(self, artifact_id: str, rel: str, root: Path) -> dict[str, object]:
        payload: dict[str, object] = {"status": "pass", "name": rel}
        if artifact_id == "operator_handoff_validation_json":
            payload.update({"check_count": 131, "pass_count": 131, "fail_count": 0})
        elif artifact_id == "final_bundle_validation_json":
            payload.update({"check_count": 152, "pass_count": 152, "fail_count": 0})
        elif artifact_id == "completion_json":
            items = [
                {"id": str(index), "status": "pass"}
                for index in range(13)
            ] + [{"id": "final", "status": "blocked"}]
            items[4]["status"] = "partial"
            items[11]["status"] = "blocked"
            payload.update(
                {
                    "status": "blocked",
                    "item_count": 14,
                    "pass_count": 11,
                    "partial_count": 1,
                    "blocked_count": 2,
                    "items": items,
                }
            )
        elif artifact_id == "c3b_readiness_json":
            payload.update(
                {
                    "status": "ready-for-board",
                    "target": "ZCU104 PYNQ",
                    "variant": "c3b-mem16",
                    "preset": "axis-c3b-mem16",
                    "tar": {
                        "name": "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "path": "/tmp/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "sha256": "a" * 64,
                        "sha256_file": "/tmp/HGTXR/docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
                        "sha256_line": ("a" * 64) + "  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                    },
                }
            )
        elif artifact_id == "xr_vits_reference_resolution_json":
            payload.update(
                {
                    "status": "candidate-ready-needs-approval",
                    "safety": {
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "xr_vits_policy_preview_json":
            payload.update(
                {
                    "status": "pass",
                    "preview_only": True,
                    "active_policy_written": False,
                    "active_policy_path": str(root / "docs/resources/xr_vits_replacement_policy.json"),
                    "policy": {
                        "approved_by": "<approved-by>",
                        "approved_at": "2026-06-10T00:00:00Z",
                        "reason": "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff",
                        "requested_path": str(root.parent.parent / "XR-VITs"),
                        "replacement_path": str(root.parent / "XR_Accel"),
                        "replacement_role": "candidate-xr-accel",
                        "candidate_audit": "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                        "candidate_audit_fingerprint": "a" * 64,
                        "candidate_audit_recommendation_snapshot": {"path": "/tmp/XR_Accel"},
                        "candidate_audit_meta": {"sha256": "a" * 64},
                        "policy_fingerprint": "b" * 64,
                    },
                    "integrity": {
                        "status": "pass",
                        "candidate_audit_path": str((root / "docs/resources/xr_vits_candidate_audit_2026_06_10.json").resolve()),
                    },
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
            approval_event = {
                "event_type": "xr_vits_replacement_approval",
                "protocol_version": "hgtxr-xr-vits-replacement-v1",
                "approver_id": "<approved-by>",
                "approved_at": "2026-06-10T00:00:00Z",
                "reason_code": "xr_vits_replacement",
                "reason": "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff",
                "replacement_role": "candidate-xr-accel",
                "requested_path": str(root.parent.parent / "XR-VITs"),
                "replacement_path": str(root.parent / "XR_Accel"),
                "candidate_audit": "docs/resources/xr_vits_candidate_audit_2026_06_10.json",
                "candidate_audit_fingerprint": "a" * 64,
                "generator": "tools/create_xr_vits_replacement_policy.py",
            }
            approval_event["event_id"] = manifest_tool.stable_sha256(approval_event)
            payload["policy"]["approval_event"] = approval_event
        elif artifact_id == "final_blocker_closure_json":
            payload["status"] = "blocked"
        elif artifact_id == "c3b_smoke_discovery_json":
            payload.update({"status": "missing", "pass_count": 0})
        elif artifact_id == "pynq_c3b_smoke_discovery_json":
            payload.update(
                {
                    "status": "missing",
                    "preset": "axis-c3b-mem16",
                    "candidate_count": 0,
                    "pass_count": 0,
                    "recommended_candidate": None,
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "pynq_vref_smoke_discovery_json":
            payload.update(
                {
                    "status": "missing",
                    "preset": "axis-vref-p0-softmax-input-x2-dsp-mixed-stream",
                    "candidate_count": 0,
                    "pass_count": 0,
                    "recommended_candidate": None,
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "c3b_smoke_contract_json":
            payload.update(
                {
                    "status": "pass",
                    "preset": "axis-c3b-mem16",
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "c3b_transfer_manifest_json":
            payload.update(
                {
                    "status": "pass",
                    "target": "ZCU104 PYNQ",
                    "variant": "c3b-mem16",
                    "preset": "axis-c3b-mem16",
                    "tar": {
                        "name": "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "path": "/tmp/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "bytes": 1024,
                        "sha256": "a" * 64,
                        "sha256_file": "e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
                        "sha256_line": ("a" * 64) + "  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                    },
                    "transfer_files": [
                        "/tmp/HGTXR/hardware/generated/pynq/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "/tmp/HGTXR/hardware/generated/signoff/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
                    ],
                    "board_verify_commands": [
                        "sha256sum -c e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256",
                    ],
                    "board_run_commands": [
                        "tar -xzf e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz",
                        "cd e2e_axis_dma_c3b_mem16_smoke_bundle",
                        "./run_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                        "./validate_e2e_axis_dma_c3b_mem16_file_smoke.sh",
                    ],
                    "board_expected_outputs": {
                        "expected_runtime_state": 2,
                        "expected_out_raw": [32, -13, 26, -6, 14, -11],
                        "result_json": "e2e_axis_dma_c3b_mem16_file_smoke.json",
                        "validation_json": "e2e_axis_dma_c3b_mem16_file_smoke_validation.json",
                    },
                    "host_copyback": {
                        "canonical_result_path": str(
                            Path("/tmp/HGTXR")
                            / "hardware"
                            / "pynq"
                            / "hgtxr"
                            / "e2e_axis_dma_c3b_mem16_file_smoke.json"
                        ),
                        "import_command": "python3 tools/import_pynq_smoke_result.py /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                        "validate_command": "python3 tools/validate_pynq_smoke_result.py hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json --preset axis-c3b-mem16",
                        "final_signoff_command": "python3 tools/check_third_goal_preflight.py --mode final-signoff",
                    },
                    "bundle_validation_errors": [],
                }
            )
        elif artifact_id == "resource_policy_json":
            payload.update(
                {
                    "status": "pass",
                    "date_tag": manifest_tool.DATE_TAG,
                    "check_count": 36,
                    "pass_count": 36,
                    "fail_count": 0,
                    "checks": [
                        {"name": "force_dsp_macro_default", "status": "pass", "detail": "1"},
                        {"name": "force_uram_macro_default", "status": "pass", "detail": "1"},
                        {"name": "small_mem_lutram_macro_default", "status": "pass", "detail": "1"},
                        {"name": "cyclic_weight_tiles_uram_macro_default", "status": "pass", "detail": "1"},
                        {"name": "cyclic_large_temps_uram_macro_default", "status": "pass", "detail": "1"},
                        {"name": "cyclic_small_tile_lutram_macro_default", "status": "pass", "detail": "1"},
                        {"name": "dsp_bind_op_present", "status": "pass", "detail": "4"},
                        {"name": "rmu_smu_dsp_bind_op_present", "status": "pass", "detail": "2"},
                        {"name": "rmu_smu_dsp_helper_defined", "status": "pass", "detail": "rmu_smu_dsp_mul"},
                        {"name": "rmu_smu_dsp_acc_helper_defined", "status": "pass", "detail": "rmu_smu_dsp_mul_acc"},
                        {"name": "rmu_projection_uses_dsp_helper", "status": "pass", "detail": "RMU projection helper calls"},
                        {"name": "smu_relation_uses_dsp_helper", "status": "pass", "detail": "SMU relation helper calls"},
                        {"name": "uram_bind_storage_present", "status": "pass", "detail": "9"},
                        {"name": "lutram_bind_storage_present", "status": "pass", "detail": "4"},
                        {"name": "cyclic_weight_tiles_uram_pragmas", "status": "pass", "detail": "6"},
                        {"name": "cyclic_large_temps_uram_pragmas", "status": "pass", "detail": "4"},
                        {"name": "cyclic_small_tile_lutram_pragmas", "status": "pass", "detail": "9"},
                        {"name": "zcu104_parallelism_default", "status": "pass", "detail": "8"},
                        {"name": "zcu104_dense_parallelism_default", "status": "pass", "detail": "8"},
                        {"name": "zcu104_fifo_depth_default", "status": "pass", "detail": "128"},
                        {"name": "resource_matrix_pass", "status": "pass", "detail": "pass"},
                        {"name": "resource_matrix_recommends_c3b", "status": "pass", "detail": "C3b"},
                        {"name": "c3b_parallelism_16", "status": "pass", "detail": "16"},
                        {"name": "c3b_memory_banks_16", "status": "pass", "detail": "16"},
                        {"name": "c3b_dsp_increased_vs_a1", "status": "pass", "detail": "C3b=604 A1=332"},
                        {"name": "c3b_lut_lower_than_c1", "status": "pass", "detail": "C3b=126506 C1=127916"},
                        {"name": "c3b_uram_positive", "status": "pass", "detail": "64"},
                        {"name": "c3b_latency_not_worse_than_c1", "status": "pass", "detail": "C3b=37508072 C1=37508072"},
                        {"name": "c3b_csynth_xml_exists", "status": "pass", "detail": "csynth.xml"},
                        {"name": "c3b_csynth_resources_match_matrix", "status": "pass", "detail": "match"},
                        {"name": "c3b_csynth_latency_matches_matrix", "status": "pass", "detail": "match"},
                        {"name": "c3b_csynth_dsp_lte_threshold", "status": "pass", "detail": "604 <= 604"},
                        {"name": "c3b_csynth_uram_lte_threshold", "status": "pass", "detail": "64 <= 64"},
                        {"name": "c3b_csynth_lut_lte_threshold", "status": "pass", "detail": "126506 <= 126506"},
                        {"name": "c3b_csynth_latency_lte_threshold", "status": "pass", "detail": "37508072 <= 37508072"},
                        {"name": "c3b_routed_wns_gte_threshold", "status": "pass", "detail": "4.415 >= 4.415"},
                    ],
                }
            )
        elif artifact_id == "third_goal_source_audit_json":
            payload.update({"status": "pass", "required_count": 86, "source_count": 224, "missing_required": []})
        elif artifact_id == "third_goal_current_audit_json":
            payload.update({"status": "blocked-external", "summary": {"requirements": 12, "blocked": 1}})
        elif artifact_id == "req1_environment_json":
            payload.update(
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
                }
            )
        elif artifact_id == "req9_deit_image_json":
            payload.update(
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
                }
            )
        elif artifact_id == "requirements_trace_json":
            payload.update(
                {
                    "status": "blocked",
                    "blocked_requirement_ids": ["11"],
                    "partial_requirement_ids": ["2", "3", "4", "10"],
                    "xr_vits_policy_integrity": {
                        "required": True,
                        "required_policy_fields": [
                            "candidate_audit_fingerprint",
                            "candidate_audit_recommendation_snapshot",
                            "candidate_audit_meta",
                            "approval_event",
                            "policy_fingerprint",
                        ],
                        "required_policy_fields_complete": True,
                        "legacy_policy_clears_final_signoff": False,
                    },
                }
            )
        elif artifact_id == "qkv_uram_successor_json":
            payload.update(
                {
                    "status": "pass",
                    "csynth": {"resources": {"bram_18k": 114, "dsp": 128, "lut": 43236, "uram": 40}},
                    "comparison": {"delta_vs_dsp_mixed_stream": {"bram_18k": -30, "dsp": 0, "lut": 0, "uram": 8}},
                    "checks": [
                        {"name": "latency_lte_c3b", "status": "pass", "detail": 498485},
                        {"name": "dsp_lte_c3b", "status": "pass", "detail": 128},
                        {"name": "lut_lte_c3b", "status": "pass", "detail": 43236},
                        {
                            "name": "bram_reduced_vs_dsp_mixed_stream",
                            "status": "pass",
                            "detail": {"qkv_uram": 114, "dsp_mixed_stream": 144},
                        },
                    ],
                    "overlay": {
                        "route_status": {"fully_routed_nets": 19846, "routable_nets": 19846, "routing_error_nets": 0},
                        "checks": [
                            {"name": "timing_wns_gte_c3b", "status": "pass", "detail": 4.517},
                            {"name": "route_errors_zero", "status": "pass", "detail": 0},
                        ]
                    },
                }
            )
        elif artifact_id == "qkv_uram_smoke_discovery_json":
            payload.update(
                {
                    "status": "missing",
                    "preset": "axis-vref-p0-softmax-input-x2-qkv-uram",
                    "candidate_count": 0,
                    "pass_count": 0,
                    "recommended_candidate": None,
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "req5_q4q8_swhw_json":
            payload.update(
                {
                    "status": "pass",
                    "precision": {"weight_bits": 4, "activation_bits": 8},
                    "pass_count": 11,
                    "fail_count": 0,
                    "checks": [
                        {"name": "config_q4w_q8a", "status": "pass", "detail": "weight_bits=4 activation_bits=8"},
                        {"name": "packed_weight_manifest", "status": "pass", "detail": "manifest"},
                        {"name": "packed_weight_binary_exists", "status": "pass", "detail": "bin"},
                        {"name": "packed_weight_sha256_matches_manifest", "status": "pass", "detail": "sha"},
                        {"name": "packed_weight_byte_count_matches_manifest", "status": "pass", "detail": "bytes"},
                        {"name": "packed_weight_expected_runtime_state", "status": "pass", "detail": "2"},
                        {"name": "packed_weight_expected_c3b_output", "status": "pass", "detail": "[32, -13, 26, -6, 14, -11]"},
                        {"name": "testbench_strict_golden_compare", "status": "pass", "detail": "strict"},
                        {"name": "c3b_axis_csim_strict_csim", "status": "pass", "detail": "runtime=2 failures=0 outputs=[32, -13, 26, -6, 14, -11]"},
                        {"name": "vref_softmax_input_x2_csim_strict_csim", "status": "pass", "detail": "runtime=2 failures=0 outputs=[58, -51, 42, -28, 36, -41]"},
                        {"name": "qkv_uram_csim_strict_csim", "status": "pass", "detail": "runtime=2 failures=0 outputs=[58, -51, 42, -28, 36, -41]"},
                    ],
                }
            )
        elif artifact_id == "vref_p0_pot_scale_audit_json":
            payload.update(
                {
                    "status": "pass",
                    "check_count": 14,
                    "pass_count": 14,
                    "fail_count": 0,
                }
            )
        elif artifact_id == "vref_p0_pot_scale_sweep_json":
            payload.update(
                {
                    "status": "pass",
                    "spec_count": 3,
                    "summary": {
                        "total_candidate_count": 45,
                        "total_fail_count": 0,
                        "specs_recommending_current": 3,
                        "next_action": "keep current PoT scales for C3b",
                    },
                }
            )
        elif artifact_id == "p2_vit_scale_calibration_json":
            payload.update(
                {
                    "status": "pass",
                    "policy": {"decision": "keep_current_pot_scales_for_c3b"},
                    "precision": {"weight_bits": 4, "activation_bits": 8},
                    "sweep_summary": {
                        "total_candidate_count": 45,
                        "total_fail_count": 0,
                        "specs_recommending_current": 3,
                    },
                    "safety": {
                        "executes_hls": False,
                        "executes_vivado": False,
                        "writes_hls_source": False,
                        "overwrites_c3b_artifacts": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                    },
                }
            )
        elif artifact_id == "req6_parameterization_json":
            payload.update(
                {
                    "status": "pass",
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
                    "pass_count": 71,
                    "check_count": 71,
                    "fail_count": 0,
                    "checks": [
                        {"name": "legal_bus_width_byte_aligned", "status": "pass", "detail": {"bus_width": 256}},
                        {"name": "legal_bus_width_data_divisible", "status": "pass", "detail": {"bus_width": 256, "bit_width": 8}},
                        {"name": "legal_bus_width_weight_divisible", "status": "pass", "detail": {"bus_width": 256, "weight_bit_width": 4}},
                        {"name": "legal_weight_width_lte_data_width", "status": "pass", "detail": {"weight_bit_width": 4, "bit_width": 8}},
                        {"name": "legal_data_width_lt_acc_width", "status": "pass", "detail": {"bit_width": 8, "acc_width": 32}},
                        {"name": "legal_model_dim_divides_heads", "status": "pass", "detail": {"model_dim": 192, "heads": 3}},
                        {"name": "legal_head_dim_matches_model_heads", "status": "pass", "detail": {"head_dim": 64, "expected": 64}},
                        {"name": "legal_ff_dim_matches_mlp_ratio", "status": "pass", "detail": {"ff_dim": 768, "expected": 768}},
                        {"name": "legal_dense_parallelism_matches_req6_parallelism", "status": "pass", "detail": {"dense_par": 8, "parallelism": 8}},
                        {"name": "legal_dense_parallelism_divides_embed", "status": "pass", "detail": {"model_dim": 192, "dense_par": 8}},
                        {"name": "legal_dense_parallelism_divides_ff_dim", "status": "pass", "detail": {"ff_dim": 768, "dense_par": 8}},
                        {"name": "legal_weight_lanes_divide_dense_parallelism", "status": "pass", "detail": {"weight_lanes": 64, "dense_par": 8}},
                        {"name": "legal_buffer_size_positive", "status": "pass", "detail": {"buffer_size": 256}},
                        {"name": "legal_fifo_depth_positive", "status": "pass", "detail": {"fifo_depth": 128}},
                        {"name": "e2e_static_assert_active_tokens_fit", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:139"},
                        {"name": "e2e_static_assert_heads_positive", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:141"},
                        {"name": "e2e_static_assert_head_dim_covers_embed", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:142"},
                        {"name": "e2e_static_assert_dense_par_positive", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:144"},
                        {"name": "e2e_static_assert_dense_par_divides_embed", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:145"},
                        {"name": "e2e_static_assert_dense_par_divides_ff_dim", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:147"},
                        {"name": "e2e_static_assert_weight_lanes_positive", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:149"},
                        {"name": "e2e_static_assert_dense_par_divides_weight_lanes", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:150"},
                        {"name": "e2e_static_assert_axis_width_matches_cyclic_axi", "status": "pass", "detail": "hgtxr_e2e_vit.hpp:152"},
                        {"name": "csim_tcl_supports_par16_par32", "status": "pass", "detail": "vivado/scripts/run_e2e_q4w8a_csim.tcl"},
                        {"name": "csynth_tcl_supports_par16_par32", "status": "pass", "detail": "vivado/scripts/run_e2e_q4w8a_csynth.tcl"},
                        {"name": "sweep_yaml_parsed", "status": "pass", "detail": "configs/sweeps/zcu104_cyclic_transformer_sweep.yaml"},
                        {"name": "parallelism_extension_c3b_par16_validated", "status": "pass", "detail": "validated_resource_matrix"},
                        {"name": "parallelism_extension_c3b_par16_evidence_resource_matrix", "status": "pass", "detail": "docs/resources/e2e_resource_matrix_2026_06_10.json:C3b"},
                        {"name": "parallelism_extension_c3b_par16_expected_result", "status": "pass", "detail": "DSP 604, LUT 126506, URAM 64, latency 37508072 cycles, routed WNS positive."},
                        {"name": "parallelism_extension_par32_exploratory_not_default", "status": "pass", "detail": "exploratory_not_default"},
                        {"name": "parallelism_extension_par32_requires_fresh_reports", "status": "pass", "detail": "Requires fresh csynth, routed timing, resource-fit audit, and no C3b artifact overwrite."},
                        {"name": "parallelism_extension_par32_records_risk", "status": "pass", "detail": "Potential latency reduction with higher DSP/partition pressure and timing risk."},
                    ],
                }
            )
        elif artifact_id == "xr_vits_gate_json":
            payload.update(
                {
                    "status": "blocked",
                    "resolution_mode": "candidate-ready-needs-approval",
                    "exact": {"exists": False, "path": "/home/kjm26/project/PRJXR/XR-VITs"},
                    "replacement_candidate": {"exists": True, "path": "/home/kjm26/project/PRJXR/XR-VIT/XR_Accel"},
                    "active_policy_summary": {"status": "missing", "would_clear": False},
                    "candidate_audit": {"recommendation_matches": True, "candidate_count": 3},
                    "safety": {
                        "executes_commands": False,
                        "executes_network": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "c3b_physical_smoke_gate_json":
            payload.update(
                {
                    "status": "blocked_missing_canonical_physical_smoke_result",
                    "ready_for_board": True,
                    "physical_smoke_pass": False,
                    "canonical_result": {"exists": False, "status": "missing"},
                    "canonical_validation": {"exists": False, "status": "missing"},
                    "bundle": {"status": "pass"},
                    "session": {"status": "pass"},
                    "safety": {
                        "executes_commands": False,
                        "executes_network": False,
                        "creates_board_result": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "vref_p0_buffer_lifetime_json":
            payload.update(
                {
                    "status": "pass",
                    "check_count": 51,
                    "pass_count": 51,
                    "fail_count": 0,
                    "checks": [
                        {"name": "force_uram_buffers_enabled", "status": "pass", "detail": "1"},
                        {"name": "small_mem_lutram_enabled", "status": "pass", "detail": "1"},
                        {"name": "axis_large_gb.q_uram", "status": "pass", "detail": "axis:q"},
                        {"name": "axis_large_gb.k_uram", "status": "pass", "detail": "axis:k"},
                        {"name": "axis_large_gb.v_uram", "status": "pass", "detail": "axis:v"},
                        {"name": "axis_pooled_lutram", "status": "pass", "detail": "axis:pooled"},
                        {"name": "attention_small_score_lutram", "status": "pass", "detail": "score"},
                        {"name": "attention_small_prob_lutram", "status": "pass", "detail": "prob"},
                        {"name": "attention_small_exp_raw_lutram", "status": "pass", "detail": "exp_raw"},
                        {"name": "rmu_smu_small_score_lutram", "status": "pass", "detail": "rmu:score"},
                        {"name": "rmu_smu_small_prob_lutram", "status": "pass", "detail": "rmu:prob"},
                        {"name": "qkv_cache_q_weight_cache_uram_successor_branch", "status": "pass", "detail": "q:uram"},
                        {"name": "qkv_cache_k_weight_cache_uram_successor_branch", "status": "pass", "detail": "k:uram"},
                        {"name": "qkv_cache_v_weight_cache_uram_successor_branch", "status": "pass", "detail": "v:uram"},
                        {"name": "qkv_successor_file_present", "status": "pass", "detail": "successor.json"},
                        {"name": "qkv_successor_status_pass", "status": "pass", "detail": "pass"},
                        {"name": "qkv_successor_macro_uram_enabled", "status": "pass", "detail": "HGTXR_E2E_URAM_QKV_WEIGHT_CACHE=1"},
                        {"name": "qkv_successor_csim_pass", "status": "pass", "detail": "pass"},
                        {"name": "qkv_successor_csynth_pass", "status": "pass", "detail": "pass"},
                        {"name": "qkv_successor_overlay_routed", "status": "pass", "detail": "pass"},
                        {"name": "qkv_successor_uram_increased_vs_dsp_mixed_stream", "status": "pass", "detail": "8"},
                        {"name": "qkv_successor_bram_reduced_vs_dsp_mixed_stream", "status": "pass", "detail": "-30"},
                        {"name": "qkv_successor_uram_positive", "status": "pass", "detail": "40"},
                        {"name": "qkv_successor_physical_smoke_pending_only", "status": "pass", "detail": "pending"},
                        {"name": "c3b_dsp_increased_vs_a1", "status": "pass", "detail": "C3b=604 A1=128"},
                        {"name": "c3b_lut_lower_than_c1", "status": "pass", "detail": "C3b=126506 C1=143124"},
                    ],
                }
            )
        elif artifact_id == "hgpipe_operator_audit_json":
            payload.update(
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
                }
            )
        elif artifact_id == "selected_path_json":
            payload.update(
                {
                    "status": "pass",
                    "check_count": 22,
                    "pass_count": 22,
                    "fail_count": 0,
                    "selection": {"path_1": "A2 then A1", "path_2": "C", "e": "pending"},
                }
            )
        elif artifact_id == "spec_plan_json":
            spec_plan_checks = [
                "spec_records_spec_kit_unavailable",
                "spec_records_zcu104",
                "spec_records_q4w_q8a",
                "spec_records_param_knobs",
                "spec_records_a2_a1_c",
                "master_records_selected_paths",
                "choice_records_e_pending",
                "execution_records_xilinx_root",
            ]
            current_doc_checks = [
                f"{doc}_{suffix}"
                for doc in ["master_plan", "sub_plan", "spec", "validation", "progress", "current_handover", "choice", "log"]
                for suffix in [
                    "records_live_manifest_required",
                    "records_live_manifest_consistency",
                    "records_live_source_count",
                    "records_live_current_summary",
                    "records_live_operator_handoff_validation",
                    "records_live_bundle_validation",
                ]
            ]
            current_doc_checks.extend(
                [
                    "current_handover_records_live_operator_handoff_validation_artifact_summary",
                    "current_handover_records_live_bundle_validation_artifact_summary",
                ]
            )
            payload.update(
                {
                    "status": "pass",
                    "check_count": 86,
                    "pass_count": 86,
                    "fail_count": 0,
                    "checks": [
                        {"name": name, "status": "pass", "detail": "fixture"}
                        for name in spec_plan_checks + current_doc_checks
                    ],
                }
            )
        elif artifact_id == "closeout_packet_json":
            payload.update(
                {
                    "status": "ready-for-operator-unblock",
                    "qkv_uram_successor_gate": {
                        "status": "ready-for-physical-smoke",
                        "required_for_final_signoff": False,
                        "resources": {"bram_18k": 114, "dsp": 128, "ff": 19664, "lut": 43236, "uram": 40},
                    },
                    "operator_commands": {
                        "qkv_uram_successor": [
                            "python3 tools/run_third_goal_final_signoff.py --execute-qkv-uram-smoke --allow-blocked",
                            "python3 tools/run_third_goal_final_signoff.py --import-qkv-uram-smoke-json /tmp/qkv.json --dry-run-import-qkv-uram-smoke --allow-blocked",
                        ]
                    },
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "unblock_intake_json":
            c3b_path = str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json")
            xr_exact = str(root.parent.parent / "XR-VITs")
            xr_policy = str(root / "docs/resources/xr_vits_replacement_policy.json")
            payload.update(
                {
                    "status": "blocked",
                    "blocker_status": {
                        "C3b AXIS/DMA physical smoke result": {
                            "current_status": "missing",
                            "current_would_clear": False,
                            "canonical_path": c3b_path,
                            "candidate_status": "not-supplied",
                            "candidate_would_clear": False,
                            "candidate_path": "",
                            "ready_for_active_unblock": False,
                            "remaining_after_candidates": True,
                            "next_input": "board-produced C3b smoke JSON",
                            "next_input_path": c3b_path,
                            "unblock_action": "capture ZCU104 C3b smoke, dry-run import it, then import to canonical path",
                        },
                        "requested XR-VITs sibling": {
                            "current_status": "missing",
                            "current_mode": "missing",
                            "current_would_clear": False,
                            "exact_path": xr_exact,
                            "policy_path": xr_policy,
                            "candidate_status": "fail",
                            "candidate_mode": "exact",
                            "candidate_would_clear": False,
                            "candidate_replacement_path": "",
                            "ready_for_active_unblock": False,
                            "remaining_after_candidates": True,
                            "next_input": "exact XR-VITs directory or approved replacement policy",
                            "next_input_path": f"{xr_exact} OR {xr_policy}",
                            "unblock_action": "restore exact XR-VITs source or approve fingerprint-bound XR_Accel replacement policy",
                        },
                    },
                    "next_inputs": [
                        {
                            "blocker": "C3b AXIS/DMA physical smoke result",
                            "input": "board-produced C3b smoke JSON",
                            "path": c3b_path,
                            "action": "capture ZCU104 C3b smoke, dry-run import it, then import to canonical path",
                        },
                        {
                            "blocker": "requested XR-VITs sibling",
                            "input": "exact XR-VITs directory or approved replacement policy",
                            "path": f"{xr_exact} OR {xr_policy}",
                            "action": "restore exact XR-VITs source or approve fingerprint-bound XR_Accel replacement policy",
                        },
                    ],
                    "operator_sequence": [
                        {
                            "step": "candidate-audit",
                            "command": "python3 tools/audit_final_unblock_candidates.py --root "
                            + str(root)
                            + " --xr-vits-mode replacement",
                            "side_effects": False,
                        },
                        {
                            "step": "dry-run-final-runner",
                            "command": "",
                            "side_effects": False,
                        },
                        {
                            "step": "active-final-runner",
                            "command": "",
                            "side_effects": True,
                        },
                    ],
                    "safety": {
                        "executes_commands": False,
                        "creates_board_result": False,
                        "creates_xr_vits_policy": False,
                        "writes_canonical_inputs": False,
                    },
                }
            )
        elif artifact_id == "closeout_validation_json":
            payload.update({"status": "pass", "check_count": 49, "pass_count": 49, "fail_count": 0})
        return payload

    def populate_required(self, root: Path) -> None:
        self.write_validator_contract(root)
        self.write_runner_contract(root)
        self.write_policy_tool_contract(root)
        self.write_runner_summary(root)
        for artifact_id, rel, kind in manifest_tool.REQUIRED_ARTIFACTS:
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "json":
                payload = self.payload_for_artifact(artifact_id, rel, root)
                if artifact_id == "final_blocker_closure_json":
                    payload.update(
                        {
                            "status": "blocked",
                            "current": {
                                "c3b_smoke": {
                                    "status": "missing",
                                    "path": str(root / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"),
                                    "would_clear": False,
                                },
                                "xr_vits": {
                                    "status": "missing",
                                    "mode": "missing",
                                    "would_clear": False,
                                    "exact": {
                                        "status": "missing",
                                        "path": str(root.parent.parent / "XR-VITs"),
                                        "would_clear": False,
                                    },
                                    "replacement_policy": {
                                        "status": "missing",
                                        "path": str(root / "docs/resources/xr_vits_replacement_policy.json"),
                                        "would_clear": False,
                                    },
                                },
                            },
                            "pynq_discovery": {
                                "c3b": self.blocker_discovery_summary("axis-c3b-mem16"),
                                "vref_p0": self.blocker_discovery_summary(
                                    "axis-vref-p0-softmax-input-x2-dsp-mixed-stream"
                                ),
                                "qkv_uram": self.blocker_discovery_summary(
                                    "axis-vref-p0-softmax-input-x2-qkv-uram"
                                ),
                            },
                            "operator_unblock_plan": {
                                "status": "needs-external-inputs",
                                "required_inputs": [
                                    {
                                        "blocker": "C3b AXIS/DMA physical smoke result",
                                        "kind": "board-produced-json",
                                        "ready": False,
                                        "canonical_path": str(
                                            root
                                            / "hardware/pynq/hgtxr/e2e_axis_dma_c3b_mem16_file_smoke.json"
                                        ),
                                        "candidate_path": "",
                                        "acceptance": [
                                            "preset axis-c3b-mem16 validates",
                                            "runtime_state is 2",
                                            "out_raw matches expected C3b vector",
                                            "bitfile/hwhfile basenames match hgtxr_e2e_axis_dma_c3b_mem16",
                                        ],
                                    },
                                    {
                                        "blocker": "requested XR-VITs sibling",
                                        "kind": "exact-source-or-approved-policy",
                                        "ready": False,
                                        "exact_path": str(root.parent.parent / "XR-VITs"),
                                        "policy_path": str(root / "docs/resources/xr_vits_replacement_policy.json"),
                                        "replacement_path": "",
                                        "acceptance": [
                                            "exact XR-VITs checkout exists, or",
                                            "replacement policy is generated by create_xr_vits_replacement_policy.py",
                                            "policy has candidate_audit_fingerprint, recommendation snapshot, and policy_fingerprint",
                                            "approved_by is a real non-placeholder identifier",
                                        ],
                                    },
                                ],
                                "dry_run_sequence": [
                                    {
                                        "step": "c3b-import-dry-run",
                                        "ready_to_execute": False,
                                        "command": "python3 tools/run_third_goal_final_signoff.py --root "
                                        + str(root)
                                        + " --import-c3b-smoke-json /path/to/e2e_axis_dma_c3b_mem16_file_smoke.json --dry-run-import-c3b-smoke --allow-blocked",
                                        "side_effects": False,
                                    },
                                    {
                                        "step": "xr-vits-policy-preview",
                                        "ready_to_execute": False,
                                        "command": "python3 tools/run_third_goal_final_signoff.py --root "
                                        + str(root)
                                        + ' --approve-xr-vits-replacement --xr-vits-approved-by USER --xr-vits-replacement-reason "Approve XR_Accel as XR-VITs replacement for HGTXR third-goal signoff" --dry-run-xr-vits-replacement --allow-blocked',
                                        "side_effects": False,
                                    },
                                ],
                                "active_sequence": [
                                    {
                                        "step": "combined-final-runner-active",
                                        "ready_to_execute": False,
                                        "command": "",
                                        "side_effects": True,
                                    }
                                ],
                                "safety": {
                                    "executes_commands": False,
                                    "creates_board_result": False,
                                    "creates_xr_vits_policy": False,
                                    "writes_canonical_inputs": False,
                                    "templates_only_when_inputs_missing": True,
                                },
                            },
                            "safety": {
                                "executes_commands": False,
                                "creates_board_result": False,
                                "creates_xr_vits_policy": False,
                                "writes_canonical_inputs": False,
                                "discovers_pynq_candidates_only": True,
                            },
                        }
                    )
                path.write_text(json.dumps(payload) + "\n")
            elif kind == "sha256-text" and artifact_id == "c3b_bundle_sha256":
                path.write_text(("a" * 64) + "  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz\n")
            else:
                path.write_text(f"{rel}\n")
        generated = root / "hardware" / "generated" / "signoff"
        generated.mkdir(parents=True, exist_ok=True)
        for suffix in ("json", "md"):
            source = root / "docs" / "resources" / f"e2e_resource_policy_audit_{manifest_tool.DATE_TAG}.{suffix}"
            target = generated / source.name
            target.write_text(source.read_text())

    def test_build_manifest_hashes_required_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "pass")
            self.assertEqual(manifest["present_required_count"], manifest["required_count"])
            self.assertEqual(manifest["missing_required"], [])
            self.assertEqual(manifest["required_count"], len(manifest_tool.REQUIRED_ARTIFACTS))
            ids = {entry["id"] for entry in manifest["artifacts"] if entry["required"]}
            self.assertIn("pynq_c3b_smoke_discovery_json", ids)
            self.assertIn("pynq_c3b_smoke_discovery_md", ids)
            self.assertIn("pynq_vref_smoke_discovery_json", ids)
            self.assertIn("pynq_vref_smoke_discovery_md", ids)
            self.assertIn("qkv_uram_smoke_discovery_json", ids)
            self.assertIn("qkv_uram_smoke_discovery_md", ids)
            self.assertIn("vref_p0_pot_scale_audit_json", ids)
            self.assertIn("vref_p0_pot_scale_audit_md", ids)
            self.assertIn("vref_p0_pot_scale_sweep_json", ids)
            self.assertIn("vref_p0_pot_scale_sweep_md", ids)
            self.assertIn("req5_q4q8_swhw_json", ids)
            self.assertIn("req5_q4q8_swhw_md", ids)
            self.assertIn("p2_vit_scale_calibration_json", ids)
            self.assertIn("p2_vit_scale_calibration_md", ids)
            self.assertIn("req6_parameterization_json", ids)
            self.assertIn("req6_parameterization_md", ids)
            self.assertIn("req1_environment_json", ids)
            self.assertIn("req1_environment_md", ids)
            self.assertIn("req9_deit_image_json", ids)
            self.assertIn("req9_deit_image_md", ids)
            self.assertIn("xr_vits_gate_json", ids)
            self.assertIn("xr_vits_gate_md", ids)
            self.assertIn("c3b_physical_smoke_gate_json", ids)
            self.assertIn("c3b_physical_smoke_gate_md", ids)
            self.assertIn("c3b_transfer_manifest_json", ids)
            self.assertIn("c3b_transfer_manifest_md", ids)
            self.assertIn("c3b_bundle_sha256", ids)
            self.assertIn("vref_p0_buffer_lifetime_json", ids)
            self.assertIn("vref_p0_buffer_lifetime_md", ids)
            self.assertIn("hgpipe_operator_audit_json", ids)
            self.assertIn("hgpipe_operator_audit_md", ids)
            self.assertEqual(manifest["failed_consistency_checks"], [])
            self.assertTrue(all(check["status"] == "pass" for check in manifest["consistency_checks"]))
            self.assertEqual(manifest["date_tag_profile"]["canonical"], manifest_tool.DATE_TAG)
            self.assertEqual(manifest["date_tag_profile"]["current_audit"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            self.assertIn(manifest_tool.DATE_TAG, manifest["artifact_date_tags"])
            self.assertIn(manifest_tool.CURRENT_AUDIT_DATE_TAG, manifest["artifact_date_tags"])
            first = manifest["artifacts"][0]
            self.assertEqual(first["status"], "pass")
            self.assertEqual(first["source_date_tag"], manifest_tool.DATE_TAG)
            self.assertEqual(len(first["sha256"]), 64)
            self.assertEqual(first["json_parse_status"], "pass")
            qkv_artifact = next(entry for entry in manifest["artifacts"] if entry["id"] == "qkv_uram_successor_json")
            self.assertEqual(qkv_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            buffer_artifact = next(entry for entry in manifest["artifacts"] if entry["id"] == "vref_p0_buffer_lifetime_json")
            self.assertEqual(buffer_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            hgpipe_artifact = next(entry for entry in manifest["artifacts"] if entry["id"] == "hgpipe_operator_audit_json")
            self.assertEqual(hgpipe_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            req1_artifact = next(entry for entry in manifest["artifacts"] if entry["id"] == "req1_environment_json")
            self.assertEqual(req1_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            req9_artifact = next(entry for entry in manifest["artifacts"] if entry["id"] == "req9_deit_image_json")
            self.assertEqual(req9_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            xr_gate_artifact = next(entry for entry in manifest["artifacts"] if entry["id"] == "xr_vits_gate_json")
            self.assertEqual(xr_gate_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)
            c3b_gate_artifact = next(
                entry for entry in manifest["artifacts"] if entry["id"] == "c3b_physical_smoke_gate_json"
            )
            self.assertEqual(c3b_gate_artifact["source_date_tag"], manifest_tool.CURRENT_AUDIT_DATE_TAG)

    def test_xr_vits_gate_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "docs" / "resources" / f"xr_vits_gate_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["replacement_candidate"]["exists"] = False
            payload["candidate_audit"]["recommendation_matches"] = False
            payload["safety"]["creates_xr_vits_policy"] = True
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("xr_vits_gate_replacement_candidate_present", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_gate_candidate_recommendation_matches", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_gate_no_side_effects", manifest["failed_consistency_checks"])

    def test_c3b_physical_gate_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / (
                f"docs/resources/c3b_physical_smoke_gate_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            payload = json.loads(path.read_text())
            payload["ready_for_board"] = False
            payload["bundle"]["status"] = "fail"
            payload["session"]["status"] = "fail"
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("c3b_physical_smoke_gate_ready_for_board", manifest["failed_consistency_checks"])
            self.assertIn("c3b_physical_smoke_gate_bundle_pass", manifest["failed_consistency_checks"])
            self.assertIn("c3b_physical_smoke_gate_session_pass", manifest["failed_consistency_checks"])

    def test_c3b_transfer_manifest_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/c3b_smoke_transfer_manifest_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["preset"] = "axis-wrong"
            payload["bundle_validation_errors"] = ["bad bundle"]
            payload["tar"]["bytes"] = -1
            payload["tar"]["sha256_line"] = "bad  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
            payload["transfer_files"] = []
            payload["board_expected_outputs"]["expected_runtime_state"] = 1
            payload["board_verify_commands"] = []
            payload["board_run_commands"] = []
            payload["host_copyback"]["import_command"] = "python3 import.py"
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("c3b_transfer_manifest_preset_variant", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_tar_shape", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_files_contract", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_bundle_validation_clean", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_sha256_line_matches", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_expected_outputs_contract", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_board_verify_command", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_board_run_commands", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_copyback_contract", manifest["failed_consistency_checks"])

    def test_c3b_transfer_readiness_alignment_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            sha_path = root / "docs/resources/e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz.sha256"
            sha_path.write_text("bad  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz\n")
            readiness_path = root / f"docs/resources/c3b_board_smoke_readiness_{manifest_tool.DATE_TAG}.json"
            readiness = json.loads(readiness_path.read_text())
            readiness["tar"]["sha256"] = "b" * 64
            readiness["tar"]["sha256_line"] = ("b" * 64) + "  e2e_axis_dma_c3b_mem16_smoke_bundle.tar.gz"
            readiness_path.write_text(json.dumps(readiness) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("c3b_transfer_manifest_sha256_file_format", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_sha_matches_readiness", manifest["failed_consistency_checks"])
            self.assertIn("c3b_transfer_manifest_bundle_to_readiness_alignment", manifest["failed_consistency_checks"])

    def test_pynq_validator_overlay_contract_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            self.write_validator_contract(root, omit="hgtxr_e2e_axis_dma_c3b_mem16")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("pynq_smoke_validator_overlay_prefix_contracts", manifest["failed_consistency_checks"])

    def test_pynq_validator_basename_logic_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "hardware" / "tools" / "validate_pynq_smoke_result.py"
            path.write_text("artifact_prefix = 'present'\nif require_paths:\n    pass\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("pynq_smoke_validator_basename_check", manifest["failed_consistency_checks"])

    def test_req9_deit_image_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "docs" / "resources" / f"req9_deit_image_reference_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["checks"] = [
                {"name": "requested_image_exists", "status": "fail", "detail": "missing"},
            ]
            payload["fail_count"] = 1
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("req9_deit_image_audit_fail_zero", manifest["failed_consistency_checks"])
            self.assertIn("req9_deit_image_requested_image_exists", manifest["failed_consistency_checks"])

    def test_req1_environment_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "docs" / "resources" / f"req1_environment_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["checks"] = [
                {"name": "not_wsl_kernel", "status": "fail", "detail": "microsoft"},
            ]
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("req1_environment_audit_not_wsl_kernel", manifest["failed_consistency_checks"])

    def test_xr_vits_policy_preview_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/xr_vits_replacement_policy_preview_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["preview_only"] = False
            payload["active_policy_written"] = True
            payload["active_policy_path"] = str(root / "docs/resources/alternate_policy.json")
            payload["policy"]["candidate_audit"] = "docs/resources/alternate_candidate_audit.json"
            payload["policy"]["approval_event"]["reason_code"] = "alternate"
            payload["policy"]["approval_event"]["replacement_path"] = "/tmp/alternate"
            payload["policy"]["approval_event"]["event_id"] = "bad-event-id"
            payload["integrity"]["status"] = "fail"
            payload["integrity"]["candidate_audit_path"] = str(root / "docs/resources/alternate_candidate_audit.json")
            payload["safety"]["creates_xr_vits_policy"] = True
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("xr_vits_policy_preview_only", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_integrity_pass", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_active_policy_path", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_candidate_audit_rel", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_integrity_candidate_path", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_approval_event_schema", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_approval_event_matches_policy", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_approval_event_id", manifest["failed_consistency_checks"])
            self.assertIn("xr_vits_policy_preview_no_side_effects", manifest["failed_consistency_checks"])
            excluded = {entry["relative_path"] for entry in manifest["volatile_excluded"]}
            self.assertIn("docs/resources/third_goal_final_signoff_run_2026_06_10.json", excluded)

    def test_xr_vits_active_placeholder_guard_source_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            self.write_policy_tool_contract(root, omit="not dry_run and approver_is_placeholder")
            self.write_runner_contract(root, omit="xr_vits_policy_status == \"fail\"")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "xr_vits_policy_tool_dry_run_placeholder_allowed",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_runner_active_policy_failure_blocks",
                manifest["failed_consistency_checks"],
            )

    def test_spec_plan_fallback_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/spec_plan_conformance_audit_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["checks"] = [
                {"name": "spec_records_spec_kit_unavailable", "status": "fail", "detail": "missing"},
                {"name": "spec_records_zcu104", "status": "pass", "detail": "fixture"},
                {"name": "spec_records_q4w_q8a", "status": "fail", "detail": "missing"},
                {"name": "spec_records_param_knobs", "status": "fail", "detail": "missing"},
                {"name": "spec_records_a2_a1_c", "status": "fail", "detail": "missing"},
                {"name": "master_records_selected_paths", "status": "fail", "detail": "missing"},
                {"name": "choice_records_e_pending", "status": "pass", "detail": "fixture"},
                {"name": "execution_records_xilinx_root", "status": "fail", "detail": "missing"},
            ]
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("spec_plan_manual_fallback_contract", manifest["failed_consistency_checks"])
            self.assertIn("spec_plan_zcu104_q4wq8a_contract", manifest["failed_consistency_checks"])
            self.assertIn("spec_plan_param_knobs_contract", manifest["failed_consistency_checks"])
            self.assertIn("spec_plan_selected_paths_contract", manifest["failed_consistency_checks"])
            self.assertIn("spec_plan_xilinx_root_contract", manifest["failed_consistency_checks"])

    def test_buffer_lifetime_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = (
                root
                / "docs"
                / "resources"
                / f"vref_p0_buffer_lifetime_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            payload = json.loads(path.read_text())
            payload["checks"] = [
                {"name": "small_mem_lutram_enabled", "status": "fail", "detail": "0"},
            ]
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("vref_p0_buffer_lifetime_small_mem_lutram_enabled", manifest["failed_consistency_checks"])

    def test_req6_legality_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "docs" / "resources" / f"req6_parameterization_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["checks"] = [
                {"name": "legal_bus_width_weight_divisible", "status": "fail", "detail": {"bus_width": 250}},
            ]
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "req6_parameterization_legal_bus_width_weight_divisible",
                manifest["failed_consistency_checks"],
            )

    def test_hgpipe_operator_property_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / (
                f"docs/resources/hgpipe_operator_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            payload = json.loads(path.read_text())
            payload["property_summary"]["status"] = "fail"
            payload["property_summary"]["fail_count"] = 1
            payload["operators"][2]["status"] = "fail"
            payload["operators"][2]["property_fail_count"] = 1
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "third_goal_requirements_trace_hgpipe_property_summary_status_pass",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "third_goal_requirements_trace_hgpipe_property_summary_no_fail",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "third_goal_requirements_trace_hgpipe_operators_all_pass",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "third_goal_requirements_trace_hgpipe_operator_property_fail_zero",
                manifest["failed_consistency_checks"],
            )

    def test_runner_blocker_summary_missing_details_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            self.write_runner_summary(root, omit_details=True)

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_runner_remaining_blocker_details_present",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_runner_remaining_blocker_keys_align",
                manifest["failed_consistency_checks"],
            )

    def test_runner_mirrored_artifact_duplicates_fail_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "hardware" / "generated" / "signoff" / f"third_goal_final_signoff_run_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["mirrored_artifacts"].append(payload["mirrored_artifacts"][1])
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_runner_mirrored_artifacts_unique",
                manifest["failed_consistency_checks"],
            )

    def test_runner_summary_count_mismatch_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "hardware" / "generated" / "signoff" / f"third_goal_final_signoff_run_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["blocker_count"] = 99
            payload["remaining_blocker_detail_count"] = 99
            payload["mirrored_artifact_count"] = 99
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_runner_blocker_count_matches_list",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_runner_remaining_blocker_detail_count_matches",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_runner_mirrored_artifact_counts_match",
                manifest["failed_consistency_checks"],
            )

    def test_runner_mirrored_source_contract_missing_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            self.write_runner_contract(root, omit="unique_ordered")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_runner_mirrored_artifacts_dedupe_contract",
                manifest["failed_consistency_checks"],
            )

    def test_blocker_readiness_discovery_side_effect_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/final_blocker_closure_readiness_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["pynq_discovery"]["qkv_uram"]["safety"]["writes_canonical_inputs"] = True
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_blocker_closure_qkv_uram_discovery_safe_no_side_effects",
                manifest["failed_consistency_checks"],
            )

    def test_blocker_readiness_operator_plan_drift_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/final_blocker_closure_readiness_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["operator_unblock_plan"]["dry_run_sequence"][0]["command"] = "python3 unsafe_active_import.py"
            payload["operator_unblock_plan"]["safety"]["writes_canonical_inputs"] = True
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_blocker_closure_operator_plan_dry_run_templates",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_blocker_closure_operator_plan_no_side_effects",
                manifest["failed_consistency_checks"],
            )

    def test_missing_required_artifact_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            missing = root / manifest_tool.REQUIRED_ARTIFACTS[0][1]
            missing.unlink()

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(manifest_tool.REQUIRED_ARTIFACTS[0][1], manifest["missing_required"])

    def test_failed_consistency_check_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/e2e_resource_policy_audit_{manifest_tool.DATE_TAG}.json"
            path.write_text(json.dumps({"status": "pass", "check_count": 26, "pass_count": 26, "fail_count": 0}) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("resource_policy_audit_check_count", manifest["failed_consistency_checks"])

    def test_resource_policy_detail_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/e2e_resource_policy_audit_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            for check in payload["checks"]:
                if check["name"] == "rmu_projection_uses_dsp_helper":
                    check["status"] = "fail"
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("resource_policy_audit_rmu_projection_uses_dsp_helper", manifest["failed_consistency_checks"])

    def test_resource_policy_generated_stale_sibling_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            generated = root / "hardware" / "generated" / "signoff"
            stale_json = generated / "e2e_resource_policy_audit_2026_06_15.json"
            stale_md = generated / "e2e_resource_policy_audit_2026_06_15.md"
            stale_json.write_text(json.dumps({"status": "pass", "date_tag": manifest_tool.DATE_TAG}) + "\n")
            stale_md.write_text("stale\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("resource_policy_generated_singularity", manifest["failed_consistency_checks"])
            self.assertIn("resource_policy_filename_payload_date_match", manifest["failed_consistency_checks"])
            self.assertIn(
                "resource_policy_audit_single_canonical_artifact_set",
                manifest["failed_consistency_checks"],
            )

    def test_resource_policy_docs_stale_sibling_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            resources = root / "docs" / "resources"
            stale_json = resources / "e2e_resource_policy_audit_2026_06_15.json"
            stale_md = resources / "e2e_resource_policy_audit_2026_06_15.md"
            stale_json.write_text(json.dumps({"status": "pass", "date_tag": "2026_06_15"}) + "\n")
            stale_md.write_text("stale\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("resource_policy_docs_singularity", manifest["failed_consistency_checks"])
            self.assertIn(
                "resource_policy_audit_single_canonical_artifact_set",
                manifest["failed_consistency_checks"],
            )

    def test_resource_policy_canonical_mirror_hash_mismatch_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            generated_json = (
                root
                / "hardware"
                / "generated"
                / "signoff"
                / f"e2e_resource_policy_audit_{manifest_tool.DATE_TAG}.json"
            )
            payload = json.loads(generated_json.read_text())
            payload["check_count"] = 37
            generated_json.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("resource_policy_canonical_mirror_sha256", manifest["failed_consistency_checks"])

    def test_req5_weight_sha_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / "docs/resources/req5_q4q8_swhw_match_audit_2026_06_16.json"
            payload = json.loads(path.read_text())
            for check in payload["checks"]:
                if check["name"] == "packed_weight_sha256_matches_manifest":
                    check["status"] = "fail"
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "req5_q4q8_packed_weight_sha256_matches_manifest",
                manifest["failed_consistency_checks"],
            )

    def test_stale_validator_check_counts_fail_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            closeout_path = root / f"docs/resources/final_unblock_closeout_packet_validation_{manifest_tool.DATE_TAG}.json"
            handoff_path = root / f"docs/resources/final_operator_handoff_validation_{manifest_tool.DATE_TAG}.json"
            bundle_path = root / f"docs/resources/final_signoff_bundle_validation_{manifest_tool.DATE_TAG}.json"
            closeout = json.loads(closeout_path.read_text())
            handoff = json.loads(handoff_path.read_text())
            bundle = json.loads(bundle_path.read_text())
            closeout["check_count"] = 48
            handoff["check_count"] = 128
            bundle["check_count"] = 149
            closeout_path.write_text(json.dumps(closeout) + "\n")
            handoff_path.write_text(json.dumps(handoff) + "\n")
            bundle_path.write_text(json.dumps(bundle) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("final_unblock_closeout_validation_check_count", manifest["failed_consistency_checks"])
            self.assertIn("final_operator_handoff_validation_check_count", manifest["failed_consistency_checks"])
            self.assertIn("final_signoff_bundle_validation_check_count", manifest["failed_consistency_checks"])

    def test_missing_bundle_validator_check_count_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            bundle_path = root / f"docs/resources/final_signoff_bundle_validation_{manifest_tool.DATE_TAG}.json"
            bundle = json.loads(bundle_path.read_text())
            bundle.pop("check_count")
            bundle_path.write_text(json.dumps(bundle) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("final_signoff_bundle_validation_check_count", manifest["failed_consistency_checks"])

    def test_unblock_intake_path_drift_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/final_unblock_intake_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["blocker_status"]["C3b AXIS/DMA physical smoke result"]["next_input_path"] = "/tmp/wrong.json"
            payload["next_inputs"][0]["path"] = "/tmp/wrong.json"
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("final_unblock_intake_c3b_next_input_path", manifest["failed_consistency_checks"])
            self.assertIn(
                "final_unblock_intake_next_input_paths_match_operator_plan",
                manifest["failed_consistency_checks"],
            )

    def test_unblock_intake_operator_sequence_drift_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/final_unblock_intake_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["operator_sequence"][1]["command"] = "python3 tools/run_third_goal_final_signoff.py --unsafe"
            payload["operator_sequence"][2]["side_effects"] = False
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_unblock_intake_operator_sequence_matches_operator_plan_commands",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_unblock_intake_operator_sequence_side_effect_profile",
                manifest["failed_consistency_checks"],
            )

    def test_qkv_discovery_wrong_preset_and_side_effects_fail_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = (
                root
                / "docs"
                / "resources"
                / f"qkv_uram_smoke_candidate_discovery_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            payload = json.loads(path.read_text())
            payload["preset"] = "axis-c3b-mem16"
            payload["safety"]["writes_canonical_inputs"] = True
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("qkv_uram_smoke_discovery_preset", manifest["failed_consistency_checks"])
            self.assertIn("qkv_uram_smoke_discovery_safe_no_side_effects", manifest["failed_consistency_checks"])

    def test_source_audit_missing_required_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = (
                root
                / "docs"
                / "resources"
                / f"third_goal_source_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            path.write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "required_count": 66,
                        "source_count": 175,
                        "missing_required": ["docs/track/PROGRESS.md"],
                    }
                )
                + "\n"
            )

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("third_goal_source_audit_missing_zero", manifest["failed_consistency_checks"])

    def test_stale_source_audit_counts_fail_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = (
                root
                / "docs"
                / "resources"
                / f"third_goal_source_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            payload = json.loads(path.read_text())
            payload["required_count"] = 85
            payload["source_count"] = 223
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("third_goal_source_audit_required_count", manifest["failed_consistency_checks"])
            self.assertIn("third_goal_source_audit_source_count", manifest["failed_consistency_checks"])

    def test_current_audit_requirement_count_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = (
                root
                / "docs"
                / "resources"
                / f"third_goal_current_audit_{manifest_tool.CURRENT_AUDIT_DATE_TAG}.json"
            )
            path.write_text(json.dumps({"status": "blocked-external", "summary": {"requirements": 11}}) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("third_goal_current_audit_requirements_count", manifest["failed_consistency_checks"])

    def test_stale_completion_audit_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/third_goal_completion_audit_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["items"][12]["status"] = "blocked"
            payload["pass_count"] = 10
            payload["blocked_count"] = 3
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn("third_goal_completion_audit_req12_manifest_pass", manifest["failed_consistency_checks"])

    def test_requirements_trace_policy_integrity_failure_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/third_goal_requirements_trace_{manifest_tool.DATE_TAG}.json"
            payload = json.loads(path.read_text())
            payload["xr_vits_policy_integrity"]["required_policy_fields_complete"] = False
            payload["xr_vits_policy_integrity"]["legacy_policy_clears_final_signoff"] = True
            path.write_text(json.dumps(payload) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "third_goal_requirements_trace_xr_vits_policy_fields_complete",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "third_goal_requirements_trace_xr_vits_legacy_policy_rejected",
                manifest["failed_consistency_checks"],
            )

    def test_required_qkv_smoke_without_result_fails_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            path = root / f"docs/resources/final_unblock_closeout_packet_{manifest_tool.DATE_TAG}.json"
            packet = json.loads(path.read_text())
            packet["qkv_uram_successor_gate"]["required_for_final_signoff"] = True
            packet["qkv_uram_successor_gate"]["physical_smoke_status"] = "not_captured"
            packet["qkv_uram_successor_gate"]["physical_smoke_result_json"] = None
            path.write_text(json.dumps(packet) + "\n")

            manifest = manifest_tool.build_manifest(root)

            self.assertEqual(manifest["status"], "fail")
            self.assertIn(
                "final_unblock_closeout_qkv_required_smoke_status_consistent",
                manifest["failed_consistency_checks"],
            )
            self.assertIn(
                "final_unblock_closeout_qkv_required_smoke_json_present",
                manifest["failed_consistency_checks"],
            )

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "HGTXR"
            self.populate_required(root)
            json_out = root / "generated" / "manifest.json"
            markdown_out = root / "generated" / "manifest.md"
            stream = io.StringIO()

            with contextlib.redirect_stdout(stream):
                code = manifest_tool.main(
                    [
                        "--root",
                        str(root),
                        "--json-out",
                        str(json_out),
                        "--markdown-out",
                        str(markdown_out),
                    ]
                )

            self.assertEqual(code, 0)
            self.assertEqual(json.loads(json_out.read_text())["status"], "pass")
            self.assertIn("HGTXR Final Evidence Manifest", markdown_out.read_text())
            self.assertIn("Consistency Checks", markdown_out.read_text())


if __name__ == "__main__":
    unittest.main()
