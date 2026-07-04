"""Artifact graph runner with HG-PIPE FakeQuantizer insertion points."""

from __future__ import annotations

from pathlib import Path

from ..artifacts import HgPipeSource
from ..int_infer.runner import TorchIntGraphRunner
from ..trace import TensorTrace, make_trace
from .modules import HGTableFakeQuantizer


class FakeQuantGraphRunner(TorchIntGraphRunner):
    """Run the artifact graph while routing LUT quantization through FakeQuantizers."""

    def __init__(self, source: HgPipeSource | str | Path, *, device: str | None = None, collect_traces: bool = False):
        super().__init__(source, device=device)
        self.collect_traces = collect_traces
        self.traces: list[TensorTrace] = []

    @staticmethod
    def _trace_name_from_scalars(scalar_name: str) -> str:
        replacements = {
            "_q_q_scalars.txt": "_qq",
            "_k_q_scalars.txt": "_kq",
            "_v_q_scalars.txt": "_vq",
            "_a_q_scalars.txt": "_aq",
            "_geluq_scalars.txt": "_geluq",
        }
        for suffix, replacement in replacements.items():
            if scalar_name.endswith(suffix):
                return scalar_name[: -len(suffix)] + replacement
        if scalar_name.endswith("_scalars.txt"):
            return scalar_name[: -len("_scalars.txt")]
        return scalar_name

    def table_quant(self, scalar_name: str, table_name: str, input_values):
        import torch

        quantizer = HGTableFakeQuantizer(scalars=self.ints(scalar_name), table=self.ints(table_name)).to(input_values.device)
        output_float = quantizer(input_values.to(torch.float32))
        if self.collect_traces:
            output_np = output_float.detach().cpu().numpy().reshape(-1)
            self.traces.append(
                make_trace(
                    name=self._trace_name_from_scalars(scalar_name),
                    runner="fakequant_graph",
                    kind="lut_fakequant",
                    values=output_np,
                    shape=(int(output_np.size),),
                    include_values=True,
                )
            )
        return torch.round(output_float).to(torch.int64)

    def trace_end_to_end(self) -> list[TensorTrace]:
        self.traces = []
        self.collect_traces = True
        self.forward_from_patch_input()
        return self.traces

