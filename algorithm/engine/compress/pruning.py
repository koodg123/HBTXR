"""Model pruning for the reimplemented HBTXR models (P5.9, P3'-R2 step 10 deferral).

The old codebase only carried pruning as a reference-only flag (see
engine.runspec.run_contract.collect_reference_only_reports); this implements it for
real on the new models with PyTorch's native ``torch.nn.utils.prune``:

- ``prune_model``: global L1-unstructured (default) or per-layer Ln-structured
  pruning over the Linear/Conv weights, optionally made permanent.
- ``remove_pruning``: fold the pruning masks into the weights (drop the reparam).
- ``sparsity_report``: per-layer + global weight sparsity.

Pruning is followed by a fine-tune (reuse engine.train / engine.distill) to recover
accuracy — the entrypoint saves the pruned checkpoint for that downstream step.
"""
from __future__ import annotations

from typing import Any

from torch import nn
import torch.nn.utils.prune as prune

_PRUNABLE = (nn.Linear, nn.Conv1d, nn.Conv2d)


def _prunable_params(model: nn.Module) -> list[tuple[nn.Module, str]]:
    return [(module, "weight") for module in model.modules() if isinstance(module, _PRUNABLE)]


def prune_model(
    model: nn.Module,
    amount: float = 0.3,
    *,
    structured: bool = False,
    n: int = 2,
    dim: int = 0,
    make_permanent: bool = False,
) -> nn.Module:
    """Prune the model's Linear/Conv weights by ``amount`` (fraction removed)."""
    params = _prunable_params(model)
    if not params:
        return model
    if structured:
        for module, name in params:
            prune.ln_structured(module, name=name, amount=amount, n=n, dim=dim)
    else:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    if make_permanent:
        remove_pruning(model)
    return model


def remove_pruning(model: nn.Module) -> nn.Module:
    """Make pruning permanent by removing the reparameterization on each module."""
    for module, name in _prunable_params(model):
        try:
            prune.remove(module, name)
        except ValueError:
            pass  # module was not pruned
    return model


def sparsity_report(model: nn.Module) -> dict[str, Any]:
    """Per-layer and global weight sparsity (fraction of exactly-zero weights)."""
    per_layer: dict[str, float] = {}
    total = 0
    zeros = 0
    for name, module in model.named_modules():
        if isinstance(module, _PRUNABLE):
            weight = module.weight
            count = weight.numel()
            zero = int((weight == 0).sum())
            per_layer[name] = zero / max(1, count)
            total += count
            zeros += zero
    return {
        "global_sparsity": zeros / max(1, total),
        "total_params": total,
        "zero_params": zeros,
        "per_layer": per_layer,
    }


__all__ = ["prune_model", "remove_pruning", "sparsity_report"]
