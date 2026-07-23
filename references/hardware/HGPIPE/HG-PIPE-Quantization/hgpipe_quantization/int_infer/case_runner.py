"""Torch integer runner for individual quantization artifact cases."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import HgPipeSource, read_ints
from ..pipeline import discover_cases
from ..trace import TensorTrace, make_trace
from .kernels import as_int_tensor, table_quantize_tensor


class TorchIntCaseRunner:
    """Run supported individual cases with torch integer kernels."""

    SUPPORTED_KINDS = {"requant_table", "gelu_requant_table"}

    def __init__(self, source: HgPipeSource | str | Path, *, device: str | None = None):
        if not isinstance(source, HgPipeSource):
            source = HgPipeSource.from_path(source)
        self.source = source
        self.device = device

    def trace_lut_cases(self, *, include_values: bool = True) -> list[TensorTrace]:
        traces: list[TensorTrace] = []
        for case in discover_cases(self.source):
            if case.kind not in self.SUPPORTED_KINDS:
                continue
            inputs = as_int_tensor(read_ints(case.input_path), device=self.device)
            output = table_quantize_tensor(inputs, read_ints(case.scalars_path), read_ints(case.table_paths[0]))
            values = output.detach().cpu().numpy().reshape(-1)
            traces.append(
                make_trace(
                    name=case.name,
                    runner="torch_int",
                    kind=case.kind,
                    values=values,
                    shape=(int(values.size),),
                    include_values=include_values,
                )
            )
        return traces

