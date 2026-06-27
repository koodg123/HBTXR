"""High-level package API for HG-PIPE quantization artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifact_imagenet import evaluate_artifact_image_batch_npy
from .artifact_patch import evaluate_artifact_patch_batch_npy, evaluate_artifact_patch_matrix_manifest, validate_artifact_patch_matrix_manifest
from .artifacts import HgPipeSource
from .fake_quant.graph_runner import FakeQuantGraphRunner
from .int_infer import TorchIntGraphRunner
from .int_infer.kernels import as_int_tensor
from .input_bridge import to_hgpipe_patch_input
from .pipeline import QuantizationCase, VerificationResult, discover_cases, verify_all
from .quant_params import OpQuantContract, QuantParamStore
from .run_result import InferenceComparison, InferenceResult, compare_inference_results, make_inference_result
from .trace import TensorTrace


@dataclass(frozen=True)
class RunnerPairResult:
    """Final-output comparison for the two artifact-backed graph runners."""

    torch_int: InferenceResult
    fakequant_graph: InferenceResult
    comparison: InferenceComparison

    def to_json(self) -> dict[str, object]:
        return {
            "torch_int": self.torch_int.to_json(),
            "fakequant_graph": self.fakequant_graph.to_json(),
            "comparison": self.comparison.to_json(),
        }




def _dtype_to_json(dtype):
    if dtype is None:
        return None
    return {"signed": dtype.signed, "bits": dtype.bits, "qmin": dtype.qmin, "qmax": dtype.qmax}


def _range_to_json(range_spec):
    if range_spec is None:
        return None
    return {"min": range_spec.minimum, "max": range_spec.maximum}


def _contract_to_json(contract: OpQuantContract, *, include_tables: bool = False) -> dict[str, object]:
    params = contract.params
    params_json = None
    if params is not None:
        params_json = {
            "name": params.name,
            "scalars": list(params.scalars),
            "offset": params.offset,
            "shift_scale": params.shift_scale,
            "effective_divisor": params.effective_divisor,
            "bound": params.bound,
            "zero_point": params.zero_point,
            "input_dtype": _dtype_to_json(params.input_dtype),
            "output_dtype": _dtype_to_json(params.output_dtype),
            "observed_range": _range_to_json(params.observed_range),
            "table_sizes": [len(table) for table in params.tables],
        }
        if include_tables:
            params_json["tables"] = [list(table) for table in params.tables]
    return {
        "name": contract.name,
        "kind": contract.kind,
        "stat_key": contract.stat_key,
        "input_dtype": _dtype_to_json(contract.input_dtype),
        "output_dtype": _dtype_to_json(contract.output_dtype),
        "observed_range": {name: _range_to_json(spec) for name, spec in contract.observed_range.items()},
        "params": params_json,
    }


class HgPipeQuantizationPackage:
    """Convenience API for scalar, zero-point, LUT, FakeQuant, and torch.int flows.

    The public ICCAD24-HG-PIPE artifacts primarily encode quantization as
    integer scalar tuples plus LUT tables. Affine scale and zero-point
    contracts are represented where a caller supplies them. Recovered HG-PIPE
    LUT contracts expose zero_point=None because no affine zero-point artifact
    is present for those operators.
    """

    def __init__(self, source: HgPipeSource | str | Path, *, device: str | None = None):
        if not isinstance(source, HgPipeSource):
            source = HgPipeSource.from_path(source)
        self.source = source
        self.device = device
        self.quant_params = QuantParamStore(source)

    def cases(self) -> list[QuantizationCase]:
        """Return all discovered quantization reference cases."""

        return discover_cases(self.source)

    def contracts(self, *, kind: str | None = None) -> list[OpQuantContract]:
        """Return structured quantization contracts for discovered cases."""

        contracts: list[OpQuantContract] = []
        for case in self.cases():
            if kind is not None and case.kind != kind:
                continue
            contracts.append(
                self.quant_params.contract_for_table_case(
                    case.name,
                    case.kind,
                    case.scalars_path.name,
                    *(path.name for path in case.table_paths),
                )
            )
        return contracts


    def export_contracts(self, *, kind: str | None = None, include_tables: bool = False) -> list[dict[str, object]]:
        """Return JSON-serializable scalar, zero-point, and LUT contracts."""

        return [_contract_to_json(contract, include_tables=include_tables) for contract in self.contracts(kind=kind)]

    def verify_reference_kernels(self) -> list[VerificationResult]:
        """Verify all scalar and LUT kernels against saved golden refs."""

        return verify_all(self.source)

    def verify_torch_int_graph(self):
        """Verify the artifact graph executed with torch integer tensors."""

        return TorchIntGraphRunner(self.source, device=self.device).verify_end_to_end()

    def verify_fakequant_graph(self):
        """Verify the artifact graph with FakeQuantizers at LUT points."""

        return FakeQuantGraphRunner(self.source, device=self.device).verify_end_to_end()


    def _runner_input(self, input_values):
        if input_values is None:
            return None
        try:
            import torch

            if torch.is_tensor(input_values):
                return input_values.to(dtype=torch.int64, device=self.device)
        except ImportError:
            pass
        return as_int_tensor(np.asarray(input_values).reshape(-1), device=self.device)

    def run_torch_int(self, *, input_values=None, topk: int = 5) -> InferenceResult:
        """Run the torch.int graph and return final head-logit summary."""

        logits, _ = TorchIntGraphRunner(self.source, device=self.device).forward_from_patch_input(self._runner_input(input_values))
        return make_inference_result(
            runner="torch_int",
            output_name="head_logits",
            values=logits.detach().cpu().numpy().reshape(-1),
            shape=tuple(logits.shape),
            topk=topk,
        )

    def run_fakequant_graph(self, *, input_values=None, topk: int = 5) -> InferenceResult:
        """Run the FakeQuantizer-inserted artifact graph and return final logits."""

        logits, _ = FakeQuantGraphRunner(self.source, device=self.device).forward_from_patch_input(self._runner_input(input_values))
        return make_inference_result(
            runner="fakequant_graph",
            output_name="head_logits",
            values=logits.detach().cpu().numpy().reshape(-1),
            shape=tuple(logits.shape),
            topk=topk,
        )

    def compare_graph_runners(self, *, input_values=None, topk: int = 5) -> RunnerPairResult:
        """Run both graph paths and compare final logits exactly."""

        torch_int = self.run_torch_int(input_values=input_values, topk=topk)
        fakequant_graph = self.run_fakequant_graph(input_values=input_values, topk=topk)
        comparison = compare_inference_results(fakequant_graph.to_json(), torch_int.to_json())
        return RunnerPairResult(torch_int=torch_int, fakequant_graph=fakequant_graph, comparison=comparison)


    def patch_input_from_image(self, image, *, scale: float):
        """Convert a normalized image tensor to experimental HG-PIPE patch input."""

        return to_hgpipe_patch_input(image, scale=scale)

    def compare_graph_runners_from_image(self, image, *, scale: float, topk: int = 5) -> RunnerPairResult:
        """Run both graph paths from an explicit-scale image bridge input."""

        patch_input = self.patch_input_from_image(image, scale=scale)
        return self.compare_graph_runners(input_values=patch_input, topk=topk)


    def evaluate_artifact_image_batch_npy(self, *, images_npy, labels_npy, scale: float, topk: int = 5) -> dict[str, object]:
        """Experimentally evaluate artifact graph outputs from image and label .npy files."""

        return evaluate_artifact_image_batch_npy(
            self.source,
            images_npy=images_npy,
            labels_npy=labels_npy,
            scale=scale,
            device=self.device,
            topk=topk,
        )

    def evaluate_artifact_patch_batch_npy(self, *, patch_inputs_npy, labels_npy, model: str, precision: str, quantization_flow: str = "torch_int", paper_equivalent: bool = False, topk: int = 5) -> dict[str, object]:
        """Evaluate already-quantized HG-PIPE patch input .npy files through the artifact graph."""

        return evaluate_artifact_patch_batch_npy(
            self.source,
            patch_inputs_npy=patch_inputs_npy,
            labels_npy=labels_npy,
            model=model,
            precision=precision,
            quantization_flow=quantization_flow,
            paper_equivalent=paper_equivalent,
            device=self.device,
            topk=topk,
        )

    def evaluate_artifact_patch_matrix_manifest(self, *, manifest, paper_equivalent_inputs: bool = False, topk: int = 5) -> list[dict[str, object]]:
        """Evaluate a JSON manifest of already-quantized HG-PIPE patch-input batches."""

        return evaluate_artifact_patch_matrix_manifest(
            self.source,
            manifest=manifest,
            paper_equivalent_inputs=paper_equivalent_inputs,
            device=self.device,
            topk=topk,
        )

    def validate_artifact_patch_matrix_manifest(self, *, manifest) -> dict[str, object]:
        """Validate a patch-input matrix manifest before running artifact graph evaluation."""

        return validate_artifact_patch_matrix_manifest(manifest)

    def trace_fakequant_graph(self) -> list[TensorTrace]:
        """Return observable FakeQuantizer traces from graph LUT insertion points."""

        return FakeQuantGraphRunner(self.source, device=self.device).trace_end_to_end()


__all__ = ["HgPipeQuantizationPackage", "RunnerPairResult"]

from .artifact_patch import ingest_artifact_patch_matrix_assets as _package_ingest_artifact_patch_matrix_assets
from .artifact_patch import load_artifact_patch_asset_source_manifest as _package_load_artifact_patch_asset_source_manifest


def _package_api_load_artifact_patch_asset_source_manifest(self, *, source_manifest):
    return _package_load_artifact_patch_asset_source_manifest(source_manifest)


def _package_api_ingest_artifact_patch_matrix_assets(
    self,
    *,
    source_manifest,
    template_manifest="configs/artifact_patch_matrix_manifest.template.json",
    output_manifest="configs/artifact_patch_matrix_manifest.json",
    copy=True,
    assert_paper_equivalent=False,
    report_json=None,
    report_markdown=None,
):
    return _package_ingest_artifact_patch_matrix_assets(
        source_manifest=source_manifest,
        template_manifest=template_manifest,
        output_manifest=output_manifest,
        copy=copy,
        assert_paper_equivalent=assert_paper_equivalent,
        report_json=report_json,
        report_markdown=report_markdown,
    )


HgPipeQuantizationPackage.load_artifact_patch_asset_source_manifest = _package_api_load_artifact_patch_asset_source_manifest
HgPipeQuantizationPackage.ingest_artifact_patch_matrix_assets = _package_api_ingest_artifact_patch_matrix_assets

from .artifact_patch import run_artifact_patch_matrix_pipeline as _package_run_artifact_patch_matrix_pipeline


def _package_api_run_artifact_patch_matrix_pipeline(
    self,
    *,
    source_manifest=None,
    manifest="configs/artifact_patch_matrix_manifest.json",
    template_manifest="configs/artifact_patch_matrix_manifest.template.json",
    output_manifest=None,
    copy=True,
    assert_paper_equivalent=False,
    matrix_report="reports/artifact_imagenet_accuracy.json",
    manifest_report_json="reports/artifact_patch_matrix_manifest_validation.json",
    manifest_report_markdown="reports/artifact_patch_matrix_manifest_validation.md",
    validation_report_json="reports/artifact_imagenet_validation.json",
    validation_report_markdown="reports/artifact_imagenet_validation.md",
    strict=False,
):
    return _package_run_artifact_patch_matrix_pipeline(
        self.source,
        source_manifest=source_manifest,
        manifest=manifest,
        template_manifest=template_manifest,
        output_manifest=output_manifest,
        copy=copy,
        assert_paper_equivalent=assert_paper_equivalent,
        matrix_report=matrix_report,
        manifest_report_json=manifest_report_json,
        manifest_report_markdown=manifest_report_markdown,
        validation_report_json=validation_report_json,
        validation_report_markdown=validation_report_markdown,
        strict=strict,
    )


HgPipeQuantizationPackage.run_artifact_patch_matrix_pipeline = _package_api_run_artifact_patch_matrix_pipeline

from .artifact_patch import write_artifact_patch_asset_source_manifest_from_directory as _package_write_artifact_patch_asset_source_manifest_from_directory


def _package_api_write_artifact_patch_asset_source_manifest_from_directory(
    self,
    *,
    asset_dir,
    output_manifest,
    models=None,
    precisions=None,
    paper_equivalent=False,
    quantization_flow="torch_int",
):
    return _package_write_artifact_patch_asset_source_manifest_from_directory(
        asset_dir=asset_dir,
        output_manifest=output_manifest,
        models=models,
        precisions=precisions,
        paper_equivalent=paper_equivalent,
        quantization_flow=quantization_flow,
    )


HgPipeQuantizationPackage.write_artifact_patch_asset_source_manifest_from_directory = _package_api_write_artifact_patch_asset_source_manifest_from_directory
