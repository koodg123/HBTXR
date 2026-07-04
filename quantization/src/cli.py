"""Command line interface for HG-PIPE quantization reconstruction."""

from __future__ import annotations

import argparse
import json
import numpy as np
from pathlib import Path

from .artifact_imagenet import evaluate_artifact_image_batch_npy, write_artifact_image_batch_report
from .artifact_patch import evaluate_artifact_patch_batch_npy, evaluate_artifact_patch_matrix_manifest, validate_artifact_patch_matrix_manifest, write_artifact_patch_batch_report, write_artifact_patch_matrix_manifest_template, write_artifact_patch_matrix_manifest_validation_json, write_artifact_patch_matrix_manifest_validation_markdown, write_artifact_patch_matrix_report
from .artifacts import HgPipeSource
from .compare import compare_trace_payloads, write_comparison_markdown
from .completion_audit import audit_completion, write_completion_audit_markdown
from .fake_quant.graph_runner import FakeQuantGraphRunner
from .fake_quant.runner import FakeQuantRunner
from .graph import ArtifactGraphRunner
from .api import HgPipeQuantizationPackage
from .int_infer import TorchIntCaseRunner, TorchIntGraphRunner
from .input_bridge import estimate_scale_from_npy_array, patch_input_contract
from .lut_calibration import calibrate_lut_from_array, write_hgpipe_txt_artifacts, write_lut_payload_json
from .paper_equivalence import scan_paper_equivalence_assets, validate_artifact_imagenet_report, write_artifact_imagenet_validation_json, write_artifact_imagenet_validation_markdown, write_paper_equivalence_assets_json, write_paper_equivalence_assets_markdown
from .pipeline import discover_cases, verify_all
from .quantization_scheme import calibrate_dyadic_scale_kl, quantize_group_vector
from .report import write_json, write_markdown
from .run_result import compare_inference_results, make_inference_result, write_runner_pair_markdown


def default_source() -> Path:
    return Path(__file__).resolve().parents[2] / "ICCAD24-HG-PIPE"


def _refresh_completion_reports(source_path: Path, device: str | None = None) -> dict[str, object]:
    command_specs = [
        (["verify"], ["verification.json", "verification.md"]),
        (["verify-graph"], ["graph_verification.json"]),
        (["verify-e2e"], ["e2e_graph_verification.json"]),
        (["verify-int"], ["torch_int_verification.json"]),
        (["verify-fakequant"], ["fakequant_verification.json"]),
        (["verify-fakequant-graph"], ["fakequant_graph_verification.json"]),
        (["export-contracts"], ["quant_contracts.json"]),
        (["run-compare"], ["run_compare_result.json", "run_compare_result.md"]),
        (["check-paper-equivalence-assets"], ["paper_equivalence_assets.json", "paper_equivalence_assets.md"]),
        (["validate-artifact-imagenet-report"], ["artifact_imagenet_validation.json", "artifact_imagenet_validation.md"]),
        (["validate-artifact-patch-matrix-manifest"], ["artifact_patch_matrix_manifest_validation.json", "artifact_patch_matrix_manifest_validation.md"]),
    ]
    device_commands = {"verify-int", "verify-fakequant-graph", "run-compare"}
    refreshed: list[str] = []
    for command, files in command_specs:
        argv = ["--source", str(source_path), *command]
        if device is not None and command[0] in device_commands:
            argv.extend(["--device", device])
        code = main(argv)
        if code != 0:
            raise RuntimeError("refresh command failed: {}".format(" ".join(command)))
        refreshed.extend(files)
    return {"reports_dir": str(Path.cwd() / "reports"), "device": device or "default", "files": refreshed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct and verify HG-PIPE quantization artifacts.")
    parser.add_argument("--source", type=Path, default=default_source(), help="Path to ICCAD24-HG-PIPE.")
    subparsers = parser.add_subparsers(dest="command", required=True)


    audit_parser = subparsers.add_parser("audit-completion", help="Summarize generated reports against the HG-PIPE quantization completion requirements.")
    audit_parser.add_argument("--json", type=Path, default=Path("reports/completion_audit.json"))
    audit_parser.add_argument("--markdown", type=Path, default=Path("reports/completion_audit.md"))
    audit_parser.add_argument("--refresh", action="store_true", help="Regenerate core reports before auditing.")
    audit_parser.add_argument("--device", default=None, help="Optional torch device for refreshed graph reports, such as cpu or cuda.")


    preflight_parser = subparsers.add_parser("check-paper-equivalence-assets", help="Scan for assets required to claim paper-equivalent HG-PIPE ImageNet accuracy.")
    preflight_parser.add_argument("--json", type=Path, default=Path("reports/paper_equivalence_assets.json"))
    preflight_parser.add_argument("--markdown", type=Path, default=Path("reports/paper_equivalence_assets.md"))

    artifact_validation_parser = subparsers.add_parser("validate-artifact-imagenet-report", help="Validate that artifact ImageNet accuracy covers paper-equivalent 3-model by 3-precision matrix.")
    artifact_validation_parser.add_argument("--report", type=Path, default=Path("reports/artifact_imagenet_accuracy.json"))
    artifact_validation_parser.add_argument("--json", type=Path, default=Path("reports/artifact_imagenet_validation.json"))
    artifact_validation_parser.add_argument("--markdown", type=Path, default=Path("reports/artifact_imagenet_validation.md"))
    artifact_validation_parser.add_argument("--strict", action="store_true", help="Return non-zero when validation fails.")

    list_parser = subparsers.add_parser("list", help="List discovered quantization cases.")
    list_parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of cases to print.")

    verify_parser = subparsers.add_parser("verify", help="Run bit-exact verification over all discovered cases.")
    verify_parser.add_argument("--json", type=Path, default=Path("reports/verification.json"))
    verify_parser.add_argument("--markdown", type=Path, default=Path("reports/verification.md"))

    graph_parser = subparsers.add_parser("verify-graph", help="Run artifact-backed component graph verification.")
    graph_parser.add_argument("--json", type=Path, default=Path("reports/graph_verification.json"))

    e2e_parser = subparsers.add_parser("verify-e2e", help="Run single-input end-to-end artifact graph verification.")
    e2e_parser.add_argument("--json", type=Path, default=Path("reports/e2e_graph_verification.json"))

    int_parser = subparsers.add_parser("verify-int", help="Run single-input end-to-end graph verification with torch integer tensors.")
    int_parser.add_argument("--json", type=Path, default=Path("reports/torch_int_verification.json"))
    int_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")

    fakequant_parser = subparsers.add_parser("verify-fakequant", help="Run LUT-backed FakeQuantizer verification over supported artifact cases.")
    fakequant_parser.add_argument("--json", type=Path, default=Path("reports/fakequant_verification.json"))

    fakequant_graph_parser = subparsers.add_parser("verify-fakequant-graph", help="Run end-to-end artifact graph verification with LUT FakeQuantizer insertion points.")
    fakequant_graph_parser.add_argument("--json", type=Path, default=Path("reports/fakequant_graph_verification.json"))
    fakequant_graph_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")

    run_int_parser = subparsers.add_parser("run-int", help="Run torch integer artifact inference and write final logits/top-k.")
    run_int_parser.add_argument("--json", type=Path, default=Path("reports/run_int_result.json"))
    run_int_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    run_int_parser.add_argument("--topk", type=int, default=5)

    run_fakequant_graph_parser = subparsers.add_parser("run-fakequant-graph", help="Run FakeQuantizer-inserted artifact graph inference and write final logits/top-k.")
    run_fakequant_graph_parser.add_argument("--json", type=Path, default=Path("reports/run_fakequant_graph_result.json"))
    run_fakequant_graph_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    run_fakequant_graph_parser.add_argument("--topk", type=int, default=5)


    run_compare_parser = subparsers.add_parser("run-compare", help="Run torch.int and FakeQuant graph inference, then write a combined comparison report.")
    run_compare_parser.add_argument("--json", type=Path, default=Path("reports/run_compare_result.json"))
    run_compare_parser.add_argument("--markdown", type=Path, default=Path("reports/run_compare_result.md"))
    run_compare_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    run_compare_parser.add_argument("--topk", type=int, default=5)



    estimate_scale_parser = subparsers.add_parser("estimate-input-scale-npy", help="Estimate experimental symmetric patch-input scale from normalized image .npy data.")
    estimate_scale_parser.add_argument("--images-npy", type=Path, required=True)
    estimate_scale_parser.add_argument("--json", type=Path, default=Path("reports/input_scale_estimate.json"))

    group_vector_parser = subparsers.add_parser("quantize-group-vector-npy", help="Apply group-vector quantization: per-token for activation, per-channel for weight.")
    group_vector_parser.add_argument("--input-npy", type=Path, required=True)
    group_vector_parser.add_argument("--tensor-role", choices=["activation", "x", "weight", "w"], required=True)
    group_vector_parser.add_argument("--bits", type=int, default=8)
    group_vector_parser.add_argument("--group-size", type=int, default=None)
    group_vector_parser.add_argument("--unsigned", action="store_true")
    group_vector_parser.add_argument("--quantized-npy", type=Path, required=True)
    group_vector_parser.add_argument("--scales-npy", type=Path, required=True)
    group_vector_parser.add_argument("--json", type=Path, default=Path("reports/group_vector_quantization.json"))

    dyadic_parser = subparsers.add_parser("calibrate-linear-dyadic-npy", help="Calibrate dyadic scaling factor for linear units with KL-divergence.")
    dyadic_parser.add_argument("--input-npy", type=Path, required=True)
    dyadic_parser.add_argument("--bits", type=int, default=8)
    dyadic_parser.add_argument("--unsigned", action="store_true")
    dyadic_parser.add_argument("--histogram-bins", type=int, default=2048)
    dyadic_parser.add_argument("--json", type=Path, default=Path("reports/linear_dyadic_scale.json"))

    calibrate_lut_parser = subparsers.add_parser("calibrate-lut-npy", help="Generate HG-PIPE style LUT/scalar artifacts from calibration .npy samples.")
    calibrate_lut_parser.add_argument("--kind", choices=["requant", "gelu-requant", "rsqrt", "softmax"], required=True)
    calibrate_lut_parser.add_argument("--input-npy", type=Path, required=True)
    calibrate_lut_parser.add_argument("--json", type=Path, default=Path("reports/lut_calibration.json"))
    calibrate_lut_parser.add_argument("--txt-dir", type=Path, default=None, help="Optional directory for HG-PIPE-style scalars/table .txt files.")
    calibrate_lut_parser.add_argument("--stem", default="calibrated", help="Output stem when --txt-dir is used.")
    calibrate_lut_parser.add_argument("--entries", type=int, default=64)
    calibrate_lut_parser.add_argument("--recip-entries", type=int, default=64)
    calibrate_lut_parser.add_argument("--bits", type=int, default=3)
    calibrate_lut_parser.add_argument("--signed", action="store_true", help="Use signed output quantization for requant/gelu/rsqrt.")
    calibrate_lut_parser.add_argument("--scale", type=float, default=1.0, help="ReQuant fixed-point scale.")
    calibrate_lut_parser.add_argument("--zero-point", type=int, default=0)
    calibrate_lut_parser.add_argument("--input-scale", type=float, default=1.0, help="Input real-value scale for GeLU/Softmax samples.")
    calibrate_lut_parser.add_argument("--output-scale", type=float, default=1.0, help="Output real-value scale for GeLU or rsqrt output integerization.")
    calibrate_lut_parser.add_argument("--exp-scale", type=float, default=32768.0)
    calibrate_lut_parser.add_argument("--recip-scale", type=float, default=256.0)
    calibrate_lut_parser.add_argument("--epsilon", type=float, default=1.0)
    calibrate_lut_parser.add_argument("--percentile", type=float, default=100.0)
    calibrate_lut_parser.add_argument("--max-iterations", type=int, default=8)
    calibrate_lut_parser.add_argument("--rounding", action="store_true", help="Add half-step rounding bias to the PoT cursor offset.")

    run_compare_image_parser = subparsers.add_parser("run-compare-image-npy", help="Experimentally run both graph paths from a normalized CHW/HWC image .npy and explicit input scale.")
    run_compare_image_parser.add_argument("--image-npy", type=Path, required=True)
    run_compare_image_parser.add_argument("--scale", type=float, required=True)
    run_compare_image_parser.add_argument("--json", type=Path, default=Path("reports/run_compare_image_result.json"))
    run_compare_image_parser.add_argument("--markdown", type=Path, default=Path("reports/run_compare_image_result.md"))
    run_compare_image_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    run_compare_image_parser.add_argument("--topk", type=int, default=5)


    artifact_image_batch_parser = subparsers.add_parser("run-artifact-image-batch-npy", help="Experimentally evaluate artifact graph accuracy from normalized image batch .npy plus label .npy and explicit input scale.")
    artifact_image_batch_parser.add_argument("--images-npy", type=Path, required=True)
    artifact_image_batch_parser.add_argument("--labels-npy", type=Path, required=True)
    artifact_image_batch_parser.add_argument("--scale", type=float, required=True)
    artifact_image_batch_parser.add_argument("--json", type=Path, default=Path("reports/artifact_imagenet_accuracy.json"))
    artifact_image_batch_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    artifact_image_batch_parser.add_argument("--topk", type=int, default=5)

    artifact_patch_batch_parser = subparsers.add_parser("run-artifact-patch-batch-npy", help="Evaluate artifact graph accuracy from already-quantized HG-PIPE patch input .npy plus label .npy.")
    artifact_patch_batch_parser.add_argument("--patch-inputs-npy", type=Path, required=True)
    artifact_patch_batch_parser.add_argument("--labels-npy", type=Path, required=True)
    artifact_patch_batch_parser.add_argument("--model", required=True)
    artifact_patch_batch_parser.add_argument("--precision", required=True)
    artifact_patch_batch_parser.add_argument("--quantization-flow", choices=["torch_int", "fakequant_graph"], default="torch_int")
    artifact_patch_batch_parser.add_argument("--paper-equivalent-inputs", action="store_true", help="Assert that patch inputs were generated by the paper-equivalent HG-PIPE preprocessing/calibration flow.")
    artifact_patch_batch_parser.add_argument("--append", action="store_true", help="Append or replace this model/precision row in the output report.")
    artifact_patch_batch_parser.add_argument("--json", type=Path, default=Path("reports/artifact_imagenet_accuracy.json"))
    artifact_patch_batch_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    artifact_patch_batch_parser.add_argument("--topk", type=int, default=5)

    artifact_patch_matrix_parser = subparsers.add_parser("run-artifact-patch-matrix-npy", help="Evaluate multiple already-quantized HG-PIPE patch input batches from a JSON manifest.")
    artifact_patch_matrix_parser.add_argument("--manifest", type=Path, required=True)
    artifact_patch_matrix_parser.add_argument("--paper-equivalent-inputs", action="store_true", help="Assert all manifest patch inputs were generated by a paper-equivalent HG-PIPE preprocessing/calibration flow.")
    artifact_patch_matrix_parser.add_argument("--json", type=Path, default=Path("reports/artifact_imagenet_accuracy.json"))
    artifact_patch_matrix_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    artifact_patch_matrix_parser.add_argument("--topk", type=int, default=5)

    artifact_patch_template_parser = subparsers.add_parser("write-artifact-patch-matrix-template", help="Write a JSON manifest template for the 3-model by 3-precision artifact patch-input matrix.")
    artifact_patch_template_parser.add_argument("--json", type=Path, default=Path("configs/artifact_patch_matrix_manifest.template.json"))

    artifact_patch_manifest_validation_parser = subparsers.add_parser("validate-artifact-patch-matrix-manifest", help="Validate manifest coverage and file readiness for the 3-model by 3-precision artifact patch matrix.")
    artifact_patch_manifest_validation_parser.add_argument("--manifest", type=Path, default=Path("configs/artifact_patch_matrix_manifest.template.json"))
    artifact_patch_manifest_validation_parser.add_argument("--json", type=Path, default=Path("reports/artifact_patch_matrix_manifest_validation.json"))
    artifact_patch_manifest_validation_parser.add_argument("--markdown", type=Path, default=Path("reports/artifact_patch_matrix_manifest_validation.md"))
    artifact_patch_manifest_validation_parser.add_argument("--strict", action="store_true", help="Return non-zero when validation fails.")

    compare_run_parser = subparsers.add_parser("compare-run-results", help="Compare two inference result JSON files.")
    compare_run_parser.add_argument("--left", type=Path, required=True)
    compare_run_parser.add_argument("--right", type=Path, required=True)
    compare_run_parser.add_argument("--json", type=Path, default=Path("reports/run_result_comparison.json"))


    export_contracts_parser = subparsers.add_parser("export-contracts", help="Export JSON quantization contracts with scalar, zero-point, and LUT metadata.")
    export_contracts_parser.add_argument("--json", type=Path, default=Path("reports/quant_contracts.json"))
    export_contracts_parser.add_argument("--kind", default=None, help="Optional case kind filter, such as requant_table.")
    export_contracts_parser.add_argument("--include-tables", action="store_true", help="Include full LUT table values instead of table sizes only.")

    trace_fakequant_parser = subparsers.add_parser("trace-fakequant", help="Write LUT-backed FakeQuantizer tensor traces.")
    trace_fakequant_parser.add_argument("--json", type=Path, default=Path("reports/fakequant_trace.json"))
    trace_fakequant_parser.add_argument("--no-values", action="store_true", help="Omit full tensor values from the trace.")

    trace_fakequant_graph_parser = subparsers.add_parser("trace-fakequant-graph", help="Write end-to-end graph traces from LUT FakeQuantizer insertion points.")
    trace_fakequant_graph_parser.add_argument("--json", type=Path, default=Path("reports/fakequant_graph_trace.json"))

    trace_int_parser = subparsers.add_parser("trace-int-cases", help="Write torch integer tensor traces for supported individual cases.")
    trace_int_parser.add_argument("--json", type=Path, default=Path("reports/torch_int_case_trace.json"))
    trace_int_parser.add_argument("--device", default=None, help="Optional torch device, such as cpu or cuda.")
    trace_int_parser.add_argument("--no-values", action="store_true", help="Omit full tensor values from the trace.")

    compare_parser = subparsers.add_parser("compare-traces", help="Compare two tensor trace JSON files.")
    compare_parser.add_argument("--left", type=Path, required=True)
    compare_parser.add_argument("--right", type=Path, required=True)
    compare_parser.add_argument("--json", type=Path, default=Path("reports/trace_comparison.json"))
    compare_parser.add_argument("--markdown", type=Path, default=Path("reports/trace_comparison.md"))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source = HgPipeSource.from_path(args.source)


    if args.command == "audit-completion":
        refresh_payload = None
        if args.refresh:
            refresh_payload = _refresh_completion_reports(args.source, device=args.device)
        payload = audit_completion(Path.cwd())
        if refresh_payload is not None:
            payload["refresh"] = refresh_payload
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        write_completion_audit_markdown(payload, args.markdown)
        status = "complete" if payload.get("fully_complete") else ("partial" if payload["passed"] else "failed")
        refreshed_count = len(refresh_payload["files"]) if refresh_payload is not None else 0
        print(
            "completion_audit={} complete={}/{} incomplete={} missing={} refreshed={}".format(
                status,
                payload["complete"],
                payload["total"],
                payload["incomplete"],
                payload["missing"],
                refreshed_count,
            )
        )
        return 0 if payload["passed"] else 1


    if args.command == "check-paper-equivalence-assets":
        payload = scan_paper_equivalence_assets(source.root, Path.cwd())
        write_paper_equivalence_assets_json(payload, args.json)
        write_paper_equivalence_assets_markdown(payload, args.markdown)
        print(
            "paper_equivalence_assets status={} present={}/{} missing={} ready={}".format(
                payload["status"],
                payload["present"],
                payload["total"],
                payload["missing"],
                payload["paper_equivalent_ready"],
            )
        )
        return 0

    if args.command == "validate-artifact-imagenet-report":
        payload = validate_artifact_imagenet_report(args.report)
        write_artifact_imagenet_validation_json(payload, args.json)
        write_artifact_imagenet_validation_markdown(payload, args.markdown)
        print(
            "artifact_imagenet_validation status={} rows={} expected={} missing_pairs={} paper_equivalent_rows={}".format(
                payload["status"],
                payload["rows"],
                payload["expected_rows"],
                len(payload["missing_pairs"]),
                payload["paper_equivalent_rows"],
            )
        )
        return 0 if payload["passed"] or not args.strict else 1

    if args.command == "list":
        cases = discover_cases(source)
        limit = args.limit or len(cases)
        for case in cases[:limit]:
            print(f"{case.kind:24} {case.name}")
        print(f"cases={len(cases)}")
        return 0

    if args.command == "verify":
        results = verify_all(source)
        write_json(results, args.json)
        write_markdown(results, args.markdown, source.root)
        passed = sum(1 for result in results if result.passed)
        mismatches = sum(result.mismatches for result in results)
        print(f"cases={passed}/{len(results)} passed mismatches={mismatches}")
        return 0 if passed == len(results) else 1

    if args.command in {"verify-graph", "verify-e2e"}:
        runner = ArtifactGraphRunner(source)
        results = runner.verify_graph() if args.command == "verify-graph" else runner.verify_end_to_end()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {
                        "name": result.name,
                        "kind": result.kind,
                        "passed": result.passed,
                        "elements": result.elements,
                        "mismatches": result.mismatches,
                        "max_abs_error": result.max_abs_error,
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            )
        )
        passed = sum(1 for result in results if result.passed)
        mismatches = sum(result.mismatches for result in results)
        label = "graph_cases" if args.command == "verify-graph" else "e2e_graph_cases"
        print(f"{label}={passed}/{len(results)} passed mismatches={mismatches}")
        return 0 if passed == len(results) else 1

    if args.command == "verify-int":
        results = TorchIntGraphRunner(source, device=args.device).verify_end_to_end()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {
                        "name": result.name,
                        "kind": result.kind,
                        "passed": result.passed,
                        "elements": result.elements,
                        "mismatches": result.mismatches,
                        "max_abs_error": result.max_abs_error,
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            )
        )
        passed = sum(1 for result in results if result.passed)
        mismatches = sum(result.mismatches for result in results)
        print(f"torch_int_cases={passed}/{len(results)} passed mismatches={mismatches}")
        return 0 if passed == len(results) else 1

    if args.command == "verify-fakequant":
        results = FakeQuantRunner(source).verify_lut_cases()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {
                        "name": result.name,
                        "kind": result.kind,
                        "passed": result.passed,
                        "elements": result.elements,
                        "mismatches": result.mismatches,
                        "max_abs_error": result.max_abs_error,
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            )
        )
        passed = sum(1 for result in results if result.passed)
        mismatches = sum(result.mismatches for result in results)
        print(f"fakequant_cases={passed}/{len(results)} passed mismatches={mismatches}")
        return 0 if passed == len(results) else 1

    if args.command == "verify-fakequant-graph":
        results = FakeQuantGraphRunner(source, device=args.device).verify_end_to_end()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {
                        "name": result.name,
                        "kind": result.kind,
                        "passed": result.passed,
                        "elements": result.elements,
                        "mismatches": result.mismatches,
                        "max_abs_error": result.max_abs_error,
                    }
                    for result in results
                ],
                indent=2,
                sort_keys=True,
            )
        )
        passed = sum(1 for result in results if result.passed)
        mismatches = sum(result.mismatches for result in results)
        print(f"fakequant_graph_cases={passed}/{len(results)} passed mismatches={mismatches}")
        return 0 if passed == len(results) else 1

    if args.command == "run-int":
        runner = TorchIntGraphRunner(source, device=args.device)
        logits, _ = runner.forward_from_patch_input()
        result = make_inference_result(
            runner="torch_int",
            output_name="head_logits",
            values=logits.detach().cpu().numpy().reshape(-1),
            shape=tuple(logits.shape),
            topk=args.topk,
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True))
        top1 = result.topk[0].index if result.topk else None
        print(f"torch_int_result wrote={args.json} top1={top1}")
        return 0

    if args.command == "run-fakequant-graph":
        runner = FakeQuantGraphRunner(source, device=args.device)
        logits, _ = runner.forward_from_patch_input()
        result = make_inference_result(
            runner="fakequant_graph",
            output_name="head_logits",
            values=logits.detach().cpu().numpy().reshape(-1),
            shape=tuple(logits.shape),
            topk=args.topk,
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True))
        top1 = result.topk[0].index if result.topk else None
        print(f"fakequant_graph_result wrote={args.json} top1={top1}")
        return 0


    if args.command == "run-compare":
        result = HgPipeQuantizationPackage(source, device=args.device).compare_graph_runners(topk=args.topk)
        payload = result.to_json()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        write_runner_pair_markdown(payload, args.markdown)
        top1_int = result.torch_int.topk[0].index if result.torch_int.topk else None
        top1_fake = result.fakequant_graph.topk[0].index if result.fakequant_graph.topk else None
        status = "passed" if result.comparison.passed else "failed"
        print(
            f"run_compare={status} mismatches={result.comparison.mismatches} "
            f"top1_int={top1_int} top1_fakequant={top1_fake} top1_equal={result.comparison.top1_equal}"
        )
        return 0 if result.comparison.passed else 1



    if args.command == "estimate-input-scale-npy":
        images = np.load(args.images_npy)
        payload = estimate_scale_from_npy_array(images)
        payload["images_npy"] = str(args.images_npy)
        payload["array_shape"] = list(images.shape)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(
            "input_scale={} images={} paper_equivalent={} wrote={}".format(
                payload["scale"],
                payload["images"],
                payload["paper_equivalent"],
                args.json,
            )
        )
        return 0

    if args.command == "quantize-group-vector-npy":
        values = np.load(args.input_npy)
        result = quantize_group_vector(
            values,
            tensor_role=args.tensor_role,
            bits=args.bits,
            group_size=args.group_size,
            signed=not args.unsigned,
        )
        args.quantized_npy.parent.mkdir(parents=True, exist_ok=True)
        args.scales_npy.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.quantized_npy, result["quantized"])
        np.save(args.scales_npy, result["scales"])
        payload = {
            key: value
            for key, value in result.items()
            if key not in {"quantized", "scales"}
        }
        payload["input_npy"] = str(args.input_npy)
        payload["input_shape"] = list(values.shape)
        payload["quantized_npy"] = str(args.quantized_npy)
        payload["scales_npy"] = str(args.scales_npy)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(
            "group_vector_quantization granularity={} role={} wrote={} {}".format(
                payload["granularity"],
                payload["tensor_role"],
                args.quantized_npy,
                args.scales_npy,
            )
        )
        return 0

    if args.command == "calibrate-linear-dyadic-npy":
        values = np.load(args.input_npy)
        payload = calibrate_dyadic_scale_kl(
            values,
            bits=args.bits,
            signed=not args.unsigned,
            histogram_bins=args.histogram_bins,
        )
        payload["input_npy"] = str(args.input_npy)
        payload["input_shape"] = list(values.shape)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(
            "linear_dyadic_scale multiplier={} shift={} kl={} wrote={}".format(
                payload["multiplier"],
                payload["shift"],
                payload["kl_divergence"],
                args.json,
            )
        )
        return 0

    if args.command == "calibrate-lut-npy":
        values = np.load(args.input_npy)
        common = {"percentile": args.percentile}
        if args.kind == "requant":
            payload = calibrate_lut_from_array(
                args.kind,
                values,
                entries=args.entries,
                bits=args.bits,
                signed=args.signed,
                scale=args.scale,
                zero_point=args.zero_point,
                max_iterations=args.max_iterations,
                rounding=args.rounding,
                **common,
            )
        elif args.kind == "gelu-requant":
            payload = calibrate_lut_from_array(
                args.kind,
                values,
                entries=args.entries,
                bits=args.bits,
                signed=args.signed,
                input_scale=args.input_scale,
                output_scale=args.output_scale,
                max_iterations=args.max_iterations,
                rounding=args.rounding,
                **common,
            )
        elif args.kind == "rsqrt":
            payload = calibrate_lut_from_array(
                args.kind,
                values,
                entries=args.entries,
                bits=args.bits,
                signed=args.signed,
                output_scale=args.output_scale,
                epsilon=args.epsilon,
                max_iterations=args.max_iterations,
                rounding=args.rounding,
                **common,
            )
        else:
            payload = calibrate_lut_from_array(
                args.kind,
                values,
                exp_entries=args.entries,
                recip_entries=args.recip_entries,
                output_bits=args.bits,
                exp_scale=args.exp_scale,
                recip_scale=args.recip_scale,
                input_scale=args.input_scale,
                **common,
            )
        payload["input_npy"] = str(args.input_npy)
        payload["input_shape"] = list(values.shape)
        write_lut_payload_json(payload, args.json)
        txt_files = []
        if args.txt_dir is not None:
            txt_files = write_hgpipe_txt_artifacts(payload, args.txt_dir, stem=args.stem)
            payload["txt_files"] = [str(path) for path in txt_files]
            write_lut_payload_json(payload, args.json)
        print(
            "lut_calibration kind={} entries={} wrote={} txt_files={} paper_equivalent={}".format(
                args.kind,
                len(next(iter(payload["tables"].values()))),
                args.json,
                len(txt_files),
                payload["paper_equivalent"],
            )
        )
        return 0

    if args.command == "run-compare-image-npy":
        image = np.load(args.image_npy)
        package = HgPipeQuantizationPackage(source, device=args.device)
        result = package.compare_graph_runners_from_image(image, scale=args.scale, topk=args.topk)
        payload = result.to_json()
        payload["input_bridge"] = {
            "mode": "experimental_explicit_scale_npy",
            "image_npy": str(args.image_npy),
            "image_shape": list(image.shape),
            "scale": args.scale,
            "contract": patch_input_contract(),
            "paper_equivalent": False,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True))
        write_runner_pair_markdown(payload, args.markdown)
        status = "passed" if result.comparison.passed else "failed"
        print(
            f"run_compare_image_npy={status} mismatches={result.comparison.mismatches} "
            f"scale={args.scale} paper_equivalent=False"
        )
        return 0 if result.comparison.passed else 1


    if args.command == "run-artifact-image-batch-npy":
        payload = evaluate_artifact_image_batch_npy(
            source,
            images_npy=args.images_npy,
            labels_npy=args.labels_npy,
            scale=args.scale,
            device=args.device,
            topk=args.topk,
        )
        write_artifact_image_batch_report(payload, args.json)
        print(
            "artifact_image_batch samples={} top1={} top5={} paper_equivalent={} runner_comparison_passed={}".format(
                payload["samples"],
                payload["top1"],
                payload["top5"],
                payload["paper_equivalent"],
                payload["runner_comparison_passed"],
            )
        )
        return 0 if payload["runner_comparison_passed"] else 1

    if args.command == "run-artifact-patch-batch-npy":
        payload = evaluate_artifact_patch_batch_npy(
            source,
            patch_inputs_npy=args.patch_inputs_npy,
            labels_npy=args.labels_npy,
            model=args.model,
            precision=args.precision,
            quantization_flow=args.quantization_flow,
            paper_equivalent=args.paper_equivalent_inputs,
            device=args.device,
            topk=args.topk,
        )
        write_artifact_patch_batch_report(payload, args.json, append=args.append)
        print(
            "artifact_patch_batch model={} precision={} samples={} top1={} top5={} paper_equivalent={} flow={} runner_comparison_passed={}".format(
                payload["model"],
                payload["precision"],
                payload["samples"],
                payload["top1"],
                payload["top5"],
                payload["paper_equivalent"],
                payload["quantization_flow"],
                payload["runner_comparison_passed"],
            )
        )
        return 0 if payload["runner_comparison_passed"] else 1

    if args.command == "run-artifact-patch-matrix-npy":
        rows = evaluate_artifact_patch_matrix_manifest(
            source,
            manifest=args.manifest,
            paper_equivalent_inputs=args.paper_equivalent_inputs,
            device=args.device,
            topk=args.topk,
        )
        write_artifact_patch_matrix_report(rows, args.json)
        passed = sum(1 for row in rows if row.get("runner_comparison_passed") is True)
        paper_rows = sum(1 for row in rows if row.get("paper_equivalent") is True)
        print(
            "artifact_patch_matrix rows={} runner_passed={} paper_equivalent_rows={} wrote={}".format(
                len(rows),
                passed,
                paper_rows,
                args.json,
            )
        )
        return 0 if passed == len(rows) else 1

    if args.command == "write-artifact-patch-matrix-template":
        write_artifact_patch_matrix_manifest_template(args.json)
        print("artifact_patch_matrix_template wrote={}".format(args.json))
        return 0

    if args.command == "validate-artifact-patch-matrix-manifest":
        payload = validate_artifact_patch_matrix_manifest(args.manifest)
        write_artifact_patch_matrix_manifest_validation_json(payload, args.json)
        write_artifact_patch_matrix_manifest_validation_markdown(payload, args.markdown)
        print(
            "artifact_patch_matrix_manifest_validation status={} rows={} expected={} missing_pairs={} existing_patch_inputs={} existing_labels={}".format(
                payload["status"],
                payload["rows"],
                payload["expected_rows"],
                len(payload["missing_pairs"]),
                payload["existing_patch_input_files"],
                payload["existing_label_files"],
            )
        )
        return 0 if payload["passed"] or not args.strict else 1

    if args.command == "compare-run-results":
        left = json.loads(args.left.read_text())
        right = json.loads(args.right.read_text())
        result = compare_inference_results(left, right)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True))
        status = "passed" if result.passed else "failed"
        print(
            f"run_result_comparison={status} "
            f"mismatches={result.mismatches} top1_equal={result.top1_equal}"
        )
        return 0 if result.passed else 1


    if args.command == "export-contracts":
        contracts = HgPipeQuantizationPackage(source, device=getattr(args, "device", None)).export_contracts(
            kind=args.kind,
            include_tables=args.include_tables,
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(contracts, indent=2, sort_keys=True))
        print(f"quant_contracts={len(contracts)} wrote={args.json}")
        return 0

    if args.command == "trace-fakequant":
        traces = FakeQuantRunner(source).trace_lut_cases(include_values=not args.no_values)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([trace.to_json() for trace in traces], indent=2, sort_keys=True))
        print(f"fakequant_traces={len(traces)} wrote={args.json}")
        return 0

    if args.command == "trace-fakequant-graph":
        traces = FakeQuantGraphRunner(source).trace_end_to_end()
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([trace.to_json() for trace in traces], indent=2, sort_keys=True))
        print(f"fakequant_graph_traces={len(traces)} wrote={args.json}")
        return 0

    if args.command == "trace-int-cases":
        traces = TorchIntCaseRunner(source, device=args.device).trace_lut_cases(include_values=not args.no_values)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([trace.to_json() for trace in traces], indent=2, sort_keys=True))
        print(f"torch_int_traces={len(traces)} wrote={args.json}")
        return 0

    if args.command == "compare-traces":
        left = json.loads(args.left.read_text())
        right = json.loads(args.right.read_text())
        results = compare_trace_payloads(left, right)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps([result.to_json() for result in results], indent=2, sort_keys=True))
        write_comparison_markdown(results, args.markdown)
        passed = sum(1 for result in results if result.passed)
        mismatches = sum(result.mismatches for result in results)
        print(f"trace_comparisons={passed}/{len(results)} passed mismatches={mismatches}")
        return 0 if passed == len(results) else 1

    parser.error(f"Unhandled command: {args.command}")
    return 2


# Deferred module entrypoint: final wrapped main() is invoked at end of file.

from .artifact_patch import ingest_artifact_patch_matrix_assets as _cli_ingest_artifact_patch_matrix_assets

_cli_original_build_parser = build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _cli_original_build_parser()
    subparsers_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    ingest_parser = subparsers_action.add_parser("ingest-artifact-patch-matrix-assets", help="Ingest canonical artifact patch matrix assets from a source manifest into template destinations.")
    ingest_parser.add_argument("--source-manifest", type=Path, required=True)
    ingest_parser.add_argument("--template-manifest", type=Path, default=Path("configs/artifact_patch_matrix_manifest.template.json"))
    ingest_parser.add_argument("--output-manifest", type=Path, default=Path("configs/artifact_patch_matrix_manifest.json"))
    ingest_parser.add_argument("--report-json", type=Path, default=Path("reports/artifact_patch_matrix_manifest_validation.json"))
    ingest_parser.add_argument("--report-markdown", type=Path, default=Path("reports/artifact_patch_matrix_manifest_validation.md"))
    ingest_parser.add_argument("--no-copy", action="store_true", help="Write the canonical manifest rows without copying files into the template destinations.")
    ingest_parser.add_argument("--assert-paper-equivalent", action="store_true", help="Assert paper_equivalent=true for all matched output rows.")
    return parser


_cli_original_main = main


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest-artifact-patch-matrix-assets":
        payload = _cli_ingest_artifact_patch_matrix_assets(
            source_manifest=args.source_manifest,
            template_manifest=args.template_manifest,
            output_manifest=args.output_manifest,
            copy=not args.no_copy,
            assert_paper_equivalent=args.assert_paper_equivalent,
            report_json=args.report_json,
            report_markdown=args.report_markdown,
        )
        validation = payload["validation"]
        print(
            "artifact_patch_matrix_ingest matched_rows={} copied_files={} validation_status={} output_manifest={}".format(
                payload["matched_rows"],
                payload["copied_files"],
                validation["status"],
                payload["output_manifest"],
            )
        )
        return 0 if validation.get("passed") else 1
    return _cli_original_main(argv)

# Deferred module entrypoint: final wrapped main() is invoked at end of file.

from .artifact_patch import run_artifact_patch_matrix_pipeline as _cli_run_artifact_patch_matrix_pipeline

_cli_pipeline_original_build_parser = build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _cli_pipeline_original_build_parser()
    subparsers_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    pipeline_parser = subparsers_action.add_parser("run-artifact-patch-matrix-pipeline", help="Run the artifact patch matrix pipeline gate from optional ingest through manifest and artifact-report validation.")
    pipeline_parser.add_argument("--source-manifest", type=Path, default=None)
    pipeline_parser.add_argument("--manifest", type=Path, default=Path("configs/artifact_patch_matrix_manifest.json"))
    pipeline_parser.add_argument("--template-manifest", type=Path, default=Path("configs/artifact_patch_matrix_manifest.template.json"))
    pipeline_parser.add_argument("--output-manifest", type=Path, default=None)
    pipeline_parser.add_argument("--matrix-report", type=Path, default=Path("reports/artifact_imagenet_accuracy.json"))
    pipeline_parser.add_argument("--manifest-report-json", type=Path, default=Path("reports/artifact_patch_matrix_manifest_validation.json"))
    pipeline_parser.add_argument("--manifest-report-markdown", type=Path, default=Path("reports/artifact_patch_matrix_manifest_validation.md"))
    pipeline_parser.add_argument("--validation-report-json", type=Path, default=Path("reports/artifact_imagenet_validation.json"))
    pipeline_parser.add_argument("--validation-report-markdown", type=Path, default=Path("reports/artifact_imagenet_validation.md"))
    pipeline_parser.add_argument("--no-copy", action="store_true")
    pipeline_parser.add_argument("--assert-paper-equivalent", action="store_true")
    pipeline_parser.add_argument("--strict", action="store_true")
    return parser


_cli_pipeline_original_main = main


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-artifact-patch-matrix-pipeline":
        source = HgPipeSource.from_path(args.source)
        payload = _cli_run_artifact_patch_matrix_pipeline(
            source,
            source_manifest=args.source_manifest,
            manifest=args.manifest,
            template_manifest=args.template_manifest,
            output_manifest=args.output_manifest,
            copy=not args.no_copy,
            assert_paper_equivalent=args.assert_paper_equivalent,
            matrix_report=args.matrix_report,
            manifest_report_json=args.manifest_report_json,
            manifest_report_markdown=args.manifest_report_markdown,
            validation_report_json=args.validation_report_json,
            validation_report_markdown=args.validation_report_markdown,
            strict=args.strict,
        )
        artifact_status = payload["artifact_report_validation"]["status"] if payload.get("artifact_report_validation") else "none"
        print(
            "artifact_patch_matrix_pipeline status={} ran_matrix={} manifest_validation_status={} artifact_report_validation_status={}".format(
                payload["status"],
                payload["ran_matrix"],
                payload["manifest_validation"]["status"],
                artifact_status,
            )
        )
        return 0 if payload["status"] == "passed" else 1
    return _cli_pipeline_original_main(argv)


# Deferred module entrypoint: final wrapped main() is invoked at end of file.

from .artifact_patch import write_artifact_patch_asset_source_manifest_from_directory as _cli_write_artifact_patch_asset_source_manifest_from_directory

_cli_source_manifest_original_build_parser = build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _cli_source_manifest_original_build_parser()
    subparsers_action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
    source_manifest_parser = subparsers_action.add_parser("write-artifact-patch-source-manifest", help="Scan an asset directory and write the source manifest consumed by artifact patch ingest/pipeline commands.")
    source_manifest_parser.add_argument("--asset-dir", type=Path, required=True)
    source_manifest_parser.add_argument("--output-manifest", type=Path, required=True)
    source_manifest_parser.add_argument("--paper-equivalent", action="store_true")
    source_manifest_parser.add_argument("--quantization-flow", default="torch_int", choices=["torch_int", "fakequant_graph"])
    return parser


_cli_source_manifest_original_main = main


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "write-artifact-patch-source-manifest":
        payload = _cli_write_artifact_patch_asset_source_manifest_from_directory(
            asset_dir=args.asset_dir,
            output_manifest=args.output_manifest,
            paper_equivalent=args.paper_equivalent,
            quantization_flow=args.quantization_flow,
        )
        print(
            "artifact_patch_source_manifest found_pairs={} missing_pairs={} output_manifest={}".format(
                len(payload["found_pairs"]),
                len(payload["missing_pairs"]),
                payload["output_manifest"],
            )
        )
        return 0
    return _cli_source_manifest_original_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
