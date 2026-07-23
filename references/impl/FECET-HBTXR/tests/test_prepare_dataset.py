from __future__ import annotations

from fecet_hbtxr.io import read_json, read_jsonl
from prepare_dataset import build_manifests, build_session_index, build_split_views, prepare_dataset


def test_build_manifests_schema_and_split_views(synthetic_workspace):
    build_session_index(canonical_root=synthetic_workspace["canonical_root"], indexes_root=synthetic_workspace["indexes_root"])
    summary = build_manifests(
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
    split_summary = build_split_views(manifests_root=synthetic_workspace["manifests_root"], splits_root=synthetic_workspace["splits_root"])
    assert summary["counts"]["train"] == 2
    rows = read_jsonl(synthetic_workspace["manifests_root"] / "train_manifest.jsonl")
    assert rows[0]["resize_policy"] == "facet_square_direct"
    assert rows[0]["event_window"]["policy"] == "fixed_count"
    assert rows[0]["annotation_ref"]["ann_id"]
    assert rows[0]["prev_annotation_ref"]["ann_id"]
    manifest_summary = read_json(synthetic_workspace["manifests_root"] / "manifest_summary.json")
    assert manifest_summary["event_count_target"] == 4
    assert split_summary["train"]["count"] == 2
    assert (synthetic_workspace["splits_root"] / "train" / "index.json").exists()


def test_prepare_dataset_end_to_end(synthetic_workspace):
    summary = prepare_dataset(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        splits_root=synthetic_workspace["splits_root"],
        split_scheme="random",
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
        event_count_target=4,
    )
    assert summary["index"]["session_count"] == 1
    assert summary["manifests"]["counts"]["train"] == 2
