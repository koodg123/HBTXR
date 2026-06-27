from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare_dataset import build_manifests, build_session_index
from prepare_facet_reference_dataset import prepare_facet_reference_dataset


def test_prepare_facet_reference_dataset_builds_sample_txt_and_cache(synthetic_workspace) -> None:
    build_session_index(canonical_root=synthetic_workspace["canonical_root"], indexes_root=synthetic_workspace["indexes_root"])
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        split_scheme="random",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        resize_policy="facet_square_direct",
        event_policy="fixed_count",
        event_count_target=4,
    )

    output_root = synthetic_workspace["project_root"] / "facet_reference"
    summary = prepare_facet_reference_dataset(
        canonical_root=synthetic_workspace["canonical_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        output_root=output_root,
        splits=["train"],
        file_batch_size=5000,
        overwrite=False,
    )

    assert summary["splits"]["train"]["status"] == "prepared"
    assert (output_root / "train" / "data").exists()
    assert (output_root / "train" / "ellipse").exists()
    assert (output_root / "train" / "cached_data" / "events_batch_0.memmap").exists()
    assert (output_root / "train" / "cached_ellipse" / "ellipses_batch_0.memmap").exists()

    event_indices = np.load(output_root / "train" / "cached_data" / "events_indices_0.npy")
    ellipse_indices = np.load(output_root / "train" / "cached_ellipse" / "ellipses_indices_0.npy")
    assert event_indices.shape == (2, 2)
    assert ellipse_indices.shape == (2, 2)
