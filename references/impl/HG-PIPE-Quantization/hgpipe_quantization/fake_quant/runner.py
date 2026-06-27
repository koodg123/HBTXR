"""Verification runner for HG-PIPE FakeQuantizer modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..artifacts import HgPipeSource, read_ints
from ..pipeline import discover_cases
from ..trace import TensorTrace, make_trace
from .modules import HGTableFakeQuantizer


@dataclass(frozen=True)
class FakeQuantVerificationResult:
    name: str
    kind: str
    passed: bool
    elements: int
    mismatches: int
    max_abs_error: float


class FakeQuantRunner:
    """Run LUT-backed FakeQuantizer checks over supported artifact cases."""

    SUPPORTED_KINDS = {"requant_table", "gelu_requant_table"}

    def __init__(self, source: HgPipeSource | str | Path):
        if not isinstance(source, HgPipeSource):
            source = HgPipeSource.from_path(source)
        self.source = source

    def verify_lut_cases(self) -> list[FakeQuantVerificationResult]:
        import torch

        results: list[FakeQuantVerificationResult] = []
        for case in discover_cases(self.source):
            if case.kind not in self.SUPPORTED_KINDS:
                continue
            inputs = torch.as_tensor(read_ints(case.input_path), dtype=torch.float32)
            expected = np.asarray(read_ints(case.output_path), dtype=np.float32)
            quantizer = HGTableFakeQuantizer(scalars=read_ints(case.scalars_path), table=read_ints(case.table_paths[0]))
            actual = quantizer(inputs).detach().cpu().numpy().reshape(-1)
            shared = min(actual.size, expected.size)
            mismatches = abs(actual.size - expected.size)
            max_abs_error = 0.0
            if shared:
                diff = actual[:shared] - expected[:shared]
                mismatches += int(np.count_nonzero(diff))
                max_abs_error = float(np.max(np.abs(diff)))
            results.append(
                FakeQuantVerificationResult(
                    name=case.name,
                    kind=f"fakequant_{case.kind}",
                    passed=mismatches == 0,
                    elements=int(expected.size),
                    mismatches=mismatches,
                    max_abs_error=max_abs_error,
                )
            )
        return results


    def trace_lut_cases(self, *, include_values: bool = True) -> list[TensorTrace]:
        import torch

        traces: list[TensorTrace] = []
        for case in discover_cases(self.source):
            if case.kind not in self.SUPPORTED_KINDS:
                continue
            inputs = torch.as_tensor(read_ints(case.input_path), dtype=torch.float32)
            quantizer = HGTableFakeQuantizer(scalars=read_ints(case.scalars_path), table=read_ints(case.table_paths[0]))
            output = quantizer(inputs).detach().cpu().numpy().reshape(-1)
            traces.append(
                make_trace(
                    name=case.name,
                    runner="fakequant",
                    kind=case.kind,
                    values=output,
                    shape=(int(output.size),),
                    include_values=include_values,
                )
            )
        return traces
