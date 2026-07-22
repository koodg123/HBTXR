from __future__ import annotations

import torch
from torch import nn

from hybrid.optim.optimizer import (
    build_named_optimizer,
    expand_optimizer_pool_candidates,
    get_optimizer_metadata,
    is_optimizer_implemented,
    list_implemented_optimizer_names,
    list_optimizer_names,
    optimizer_pool_summary,
    write_optimizer_pool_report,
)


def test_registry_names_are_unique_and_non_empty() -> None:
    names = list_optimizer_names()
    assert names
    assert len(names) == len(set(names))


def test_implemented_subset_matches_the_flag() -> None:
    implemented = list_implemented_optimizer_names()
    assert set(implemented) <= set(list_optimizer_names())
    assert all(is_optimizer_implemented(name) for name in implemented)


def test_pool_summary_covers_every_registered_name() -> None:
    summary = optimizer_pool_summary()
    assert len(summary) == len(list_optimizer_names())
    assert {row["name"] for row in summary} == set(list_optimizer_names())


def test_metadata_exposes_required_keys() -> None:
    meta = get_optimizer_metadata("adamw")
    for key in ("name", "implemented", "builder"):
        assert key in meta


def test_build_named_optimizer_injects_the_requested_name() -> None:
    model = nn.Linear(4, 2)
    built = build_named_optimizer(model, "adamw", {"training": {"optimizer": {"lr": 0.001}}})
    assert isinstance(built, tuple) and len(built) == 4
    optimizer, resolved, metadata, summary = built
    assert isinstance(optimizer, torch.optim.Optimizer)
    assert resolved["name"] == "adamw"
    assert resolved["lr"] == 0.001
    assert "builder" in metadata
    assert summary["implemented"] is True


def test_pool_report_helpers_remain_delegated() -> None:
    assert callable(expand_optimizer_pool_candidates)
    assert callable(write_optimizer_pool_report)
