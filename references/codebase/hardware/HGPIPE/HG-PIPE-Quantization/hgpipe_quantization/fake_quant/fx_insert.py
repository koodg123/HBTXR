"""FX graph insertion helpers for FakeQuantizer modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping


def _torch_fx():
    import torch.fx as fx

    return fx


def _torch_nn():
    import torch.nn as nn

    return nn


QuantizerFactory = Callable[[], object]


def _fresh_name(existing: set[str], base: str) -> str:
    candidate = base
    index = 0
    while candidate in existing:
        index += 1
        candidate = f"{base}_{index}"
    existing.add(candidate)
    return candidate


def insert_output_fake_quantizers(model, quantizers: Mapping[str, object | QuantizerFactory]):
    """Insert fake quantizers after selected ``call_module`` nodes.

    ``quantizers`` maps existing module targets, for example ``"blocks.0.mlp"``,
    to either an ``nn.Module`` instance or a zero-argument factory returning one.
    The function returns a ``torch.fx.GraphModule``.
    """

    fx = _torch_fx()
    nn = _torch_nn()
    graph_module = model if isinstance(model, fx.GraphModule) else fx.symbolic_trace(model)
    existing_names = set(dict(graph_module.named_modules()))

    for node in list(graph_module.graph.nodes):
        if node.op != "call_module" or node.target not in quantizers:
            continue

        quantizer_or_factory = quantizers[str(node.target)]
        quantizer = quantizer_or_factory() if callable(quantizer_or_factory) and not isinstance(quantizer_or_factory, nn.Module) else quantizer_or_factory
        if not isinstance(quantizer, nn.Module):
            raise TypeError(f"quantizer for {node.target!r} is not an nn.Module")

        quantizer_name = _fresh_name(existing_names, f"{str(node.target).replace('.', '_')}_fake_quant")
        graph_module.add_module(quantizer_name, quantizer)
        with graph_module.graph.inserting_after(node):
            quantized = graph_module.graph.call_module(quantizer_name, args=(node,))
        node.replace_all_uses_with(quantized)
        quantized.args = (node,)

    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module

