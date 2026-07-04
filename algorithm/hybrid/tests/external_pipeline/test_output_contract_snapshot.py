from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.data.dataset import EVEyeHBTXRDataset, Mode2Dataset
from src.loss.metrics import compute_metrics
from src.loss.stage1 import compute_stage1_losses
from src.loss.stage2 import compute_stage2_losses
from src.models.hybrid_tracker import HBTXRTracker
from src.preprocess.build_manifests import build_manifests
from src.preprocess.canonicalize import canonicalize_dataset


SNAPSHOT_PATH = Path(__file__).resolve().parent / "snapshots" / "output_contracts.json"
UPDATE_SNAPSHOTS = os.getenv("HBTXR_UPDATE_SNAPSHOTS") == "1"


def _snapshot_bundle() -> dict[str, Any]:
    if SNAPSHOT_PATH.exists():
        return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    return {
        "dataset_contract": {},
        "model_output_contracts": {},
        "loss_log_contracts": {},
    }


def _write_snapshot_bundle(bundle: dict[str, Any]) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _assert_snapshot(section: str, case_name: str, actual: dict[str, Any]) -> None:
    bundle = _snapshot_bundle()
    section_bundle = bundle.setdefault(section, {})
    if UPDATE_SNAPSHOTS:
        section_bundle[case_name] = actual
        _write_snapshot_bundle(bundle)
    expected = section_bundle.get(case_name)
    assert expected is not None, (
        f"Missing snapshot for {section}.{case_name}. "
        "Run with HBTXR_UPDATE_SNAPSHOTS=1 to refresh the JSON snapshot."
    )
    assert actual == expected


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _symbolic_shape(
    tensor: torch.Tensor,
    *,
    batch_size: int | None = None,
    embed_dim: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> list[Any]:
    shape: list[Any] = []
    rank = tensor.ndim
    for index, dim in enumerate(tensor.shape):
        value: Any = int(dim)
        if batch_size is not None and index == 0 and int(dim) == int(batch_size):
            value = "B"
        elif rank >= 2 and height is not None and index == rank - 2 and int(dim) == int(height):
            value = "H"
        elif rank >= 1 and width is not None and index == rank - 1 and int(dim) == int(width):
            value = "W"
        elif embed_dim is not None and int(dim) == int(embed_dim):
            value = "D"
        shape.append(value)
    return shape


def _tensor_contract(
    tensor: torch.Tensor,
    *,
    batch_size: int | None = None,
    embed_dim: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    return {
        "dtype": _dtype_name(tensor),
        "rank": int(tensor.ndim),
        "shape": _symbolic_shape(
            tensor,
            batch_size=batch_size,
            embed_dim=embed_dim,
            height=height,
            width=width,
        ),
    }


def _tensor_contracts(
    values: dict[str, Any],
    *,
    batch_size: int | None = None,
    embed_dim: int | None = None,
    height: int | None = None,
    width: int | None = None,
) -> dict[str, dict[str, Any]]:
    return {
        key: _tensor_contract(
            value,
            batch_size=batch_size,
            embed_dim=embed_dim,
            height=height,
            width=width,
        )
        for key, value in sorted(values.items())
        if torch.is_tensor(value)
    }


def _non_tensor_contracts(values: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        key: {"type": type(value).__name__}
        for key, value in sorted(values.items())
        if not torch.is_tensor(value)
    }


def _variant_extra_keys(keys: set[str], core_keys: set[str]) -> dict[str, list[str]]:
    extra_keys = keys - core_keys
    groups = {
        "dense_eye": {
            "search/eye_anchor_points",
            "search/eye_boxes",
            "search/eye_obj_logits",
        },
        "bbox_family": {
            "search/pupil_bbox",
            "event/pupil_bbox",
        },
        "obb_family": {
            "search/pupil_obb",
            "event/pupil_obb",
        },
        "roi_bbox_head": {
            "search/roi_boxes_xyxy",
            "search/roi_tokens",
        },
        "proto_mask": {
            "search/mask_coeff",
            "search/mask_coarse_logits",
            "search/mask_proto",
        },
        "mask_guidance": {
            "search/mask_axes",
            "search/mask_center",
        },
        "aux_logits": {
            "search/aux",
            "event/aux",
        },
    }
    return {
        name: sorted(group & extra_keys)
        for name, group in groups.items()
        if group & extra_keys
    }


def _dataset_contract(sample: dict[str, Any]) -> dict[str, Any]:
    frame = sample["frame"]
    meta = sample["meta"]
    return {
        "required_keys": sorted(sample.keys()),
        "optional_keys": [],
        "tensor_contracts": _tensor_contracts(
            sample,
            height=int(frame.shape[-2]),
            width=int(frame.shape[-1]),
        ),
        "non_tensor_contracts": _non_tensor_contracts({k: v for k, v in sample.items() if k != "meta"}),
        "meta_required_keys": sorted(meta.keys()),
        "meta_optional_keys": [],
        "meta_value_types": {key: type(value).__name__ for key, value in sorted(meta.items())},
    }


def _model_contract(
    outputs: dict[str, torch.Tensor],
    *,
    batch_size: int,
    embed_dim: int,
    input_size: tuple[int, int],
    core_keys: set[str] | None = None,
) -> dict[str, Any]:
    keys = sorted(outputs.keys())
    contract = {
        "required_keys": keys,
        "optional_keys": [],
        "tensor_contracts": _tensor_contracts(
            outputs,
            batch_size=batch_size,
            embed_dim=embed_dim,
            height=input_size[0],
            width=input_size[1],
        ),
    }
    if core_keys is not None:
        contract["variant_extra_keys"] = _variant_extra_keys(set(keys), core_keys)
    return contract


def _loss_contract(logs: dict[str, torch.Tensor], metrics: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        "loss_keys": sorted(logs.keys()),
        "metrics_keys": sorted(metrics.keys()),
        "loss_tensor_contracts": _tensor_contracts(logs),
        "metric_tensor_contracts": _tensor_contracts(metrics),
    }


def _mode1_sample(synthetic_workspace: dict[str, Path]) -> dict[str, Any]:
    build_manifests(
        canonical_root=synthetic_workspace["canonical_root"],
        indexes_root=synthetic_workspace["indexes_root"],
        manifests_root=synthetic_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode1",
        manifest_name="manifest1",
    )
    manifest_path = synthetic_workspace["manifests_root"] / "manifest1" / "train_manifest.jsonl"
    dataset = EVEyeHBTXRDataset(
        str(manifest_path),
        canonical_root=str(synthetic_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
        component_cfg={
            "reader": {"variant": "split_v1"},
            "transform": {"variant": "split_v1"},
            "event_builder": {"variant": "split_v1"},
            "roi": {"variant": "split_v1"},
            "targets": {"variant": "split_v1"},
            "assembler": {"variant": "split_v1"},
        },
    )
    return dataset[0]


def _mode2_sample(synthetic_raw_workspace: dict[str, Path]) -> dict[str, Any]:
    canonicalize_dataset(
        raw_root=synthetic_raw_workspace["raw_root"],
        canonical_root=synthetic_raw_workspace["canonical_root"],
        annotation_mode="groundedsam",
        annotation_root=synthetic_raw_workspace["annotation_root"],
        data_mode="mode2",
        canonical_name="canonical2",
        frame_source="interpolated",
        interpolation_alpha=0.5,
        interpolation_model="linear_blend",
        synthetic_overlap_policy="reuse_event_window",
        link_mode="copy",
        num_workers=1,
    )
    build_manifests(
        canonical_root=synthetic_raw_workspace["canonical_root"],
        manifests_root=synthetic_raw_workspace["manifests_root"],
        event_policy="fixed_count",
        event_count_target=4,
        data_mode="mode2",
        canonical_name="canonical2",
        manifest_name="manifest2",
        frame_source="interpolated",
    )
    manifest_path = synthetic_raw_workspace["manifests_root"] / "manifest2" / "train_manifest.jsonl"
    dataset = Mode2Dataset(
        str(manifest_path),
        canonical_root=str(synthetic_raw_workspace["canonical_root"]),
        event_builder={"policy": "fixed_count", "event_count_target": 4},
        resize_policy="facet_square_direct",
        use_cache=False,
        component_cfg={
            "reader": {"variant": "split_v1"},
            "transform": {"variant": "split_v1"},
            "event_builder": {"variant": "split_v1"},
            "roi": {"variant": "split_v1"},
            "targets": {"variant": "split_v1"},
            "assembler": {"variant": "split_v1"},
        },
    )
    return dataset[0]


def _batched_sample(sample: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: value.unsqueeze(0) for key, value in sample.items() if torch.is_tensor(value)}


def _core_tracker() -> HBTXRTracker:
    return HBTXRTracker(
        embed_dim=48,
        depth=2,
        num_heads=3,
        patch_size=16,
        input_size=(256, 256),
        enable_aux_head=False,
    )


def _extended_tracker(*, stage: str) -> HBTXRTracker:
    return HBTXRTracker(
        embed_dim=48,
        depth=2,
        num_heads=3,
        patch_size=16,
        input_size=(256, 256),
        enable_aux_head=True,
        enable_event_head=(stage == "stage2"),
        enable_track_head=(stage == "stage2"),
        enable_search_bbox_aux_head=True,
        enable_event_bbox_aux_head=(stage == "stage2"),
        enable_search_obb_aux_head=True,
        enable_event_obb_aux_head=(stage == "stage2"),
        mask_variant="proto",
        mask_cfg={"num_prototypes": 4},
        search_cfg={
            "eye_detector": {"mode": "dense"},
            "roi_bbox_head": {"enabled": True},
            "mask_cascade": {"enabled": True},
        },
    )


def _loss_cfg() -> dict[str, float]:
    return {
        "search_bbox_aux_weight": 1.0,
        "event_bbox_aux_weight": 1.0,
        "search_obb_aux_weight": 1.0,
        "event_obb_aux_weight": 1.0,
        "aux_weight": 1.0,
        "constraint_center_weight": 1.0,
        "consistency_weight": 1.0,
    }


def test_mode1_dataset_contract_snapshot(synthetic_workspace: dict[str, Path]):
    actual = _dataset_contract(_mode1_sample(synthetic_workspace))
    _assert_snapshot("dataset_contract", "mode1_core_sample", actual)


def test_mode2_dataset_contract_snapshot(synthetic_raw_workspace: dict[str, Path]):
    actual = _dataset_contract(_mode2_sample(synthetic_raw_workspace))
    _assert_snapshot("dataset_contract", "mode2_synthetic_sample", actual)


def test_core_tracker_output_contract_snapshot():
    model = _core_tracker().eval()
    batch = {
        "frame": torch.randn(2, 1, 256, 256),
        "event": torch.randn(2, 2, 256, 256),
        "prev_state": torch.randn(2, 6),
    }
    with torch.no_grad():
        outputs = model(batch)
    actual = _model_contract(outputs, batch_size=2, embed_dim=48, input_size=(256, 256))
    _assert_snapshot("model_output_contracts", "core_tracker_outputs", actual)


def test_extended_tracker_output_contract_snapshot():
    model = _extended_tracker(stage="stage2").eval()
    batch = {
        "frame": torch.randn(2, 1, 256, 256),
        "event": torch.randn(2, 2, 256, 256),
        "prev_state": torch.randn(2, 6),
    }
    with torch.no_grad():
        outputs = model(batch)
    core_keys = set(_core_tracker().eval()(batch).keys())
    actual = _model_contract(
        outputs,
        batch_size=2,
        embed_dim=48,
        input_size=(256, 256),
        core_keys=core_keys,
    )
    _assert_snapshot("model_output_contracts", "extended_tracker_outputs", actual)


def test_loss_log_contract_snapshot(synthetic_workspace: dict[str, Path]):
    sample = _mode1_sample(synthetic_workspace)
    batch = _batched_sample(sample)
    loss_cfg = _loss_cfg()

    stage1_model = _extended_tracker(stage="stage1").eval()
    stage2_model = _extended_tracker(stage="stage2").eval()
    model_inputs = {
        "frame": batch["frame"],
        "event": batch["event"],
        "prev_state": batch["prev_state"],
    }

    with torch.no_grad():
        stage1_outputs = stage1_model(model_inputs)
        stage2_outputs = stage2_model(model_inputs)

    stage1_logs = compute_stage1_losses(batch, stage1_outputs, loss_cfg)
    stage2_logs = compute_stage2_losses(batch, stage2_outputs, loss_cfg)
    stage1_metrics = compute_metrics(batch, stage1_outputs)
    stage2_metrics = compute_metrics(batch, stage2_outputs)

    _assert_snapshot(
        "loss_log_contracts",
        "stage1_loss_logs",
        _loss_contract(stage1_logs, stage1_metrics),
    )
    _assert_snapshot(
        "loss_log_contracts",
        "stage2_loss_logs",
        _loss_contract(stage2_logs, stage2_metrics),
    )
