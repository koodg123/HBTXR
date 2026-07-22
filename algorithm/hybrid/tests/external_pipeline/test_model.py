from __future__ import annotations

import torch

from hybrid.models.hybrid_tracker import HBTXRTracker
from hybrid.models.pruning import normalize_compression_cfg
from hybrid.runtime.tracker import RuntimeHBTXRTracker
from hybrid.training.trainer import build_model


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "frame": torch.randn(batch_size, 1, 256, 256),
        "event": torch.randn(batch_size, 2, 256, 256),
        "prev_state": torch.randn(batch_size, 6),
    }


def _model(*, variant: str = "legacy", pruning_cfg: dict | None = None) -> HBTXRTracker:
    return HBTXRTracker(
        embed_dim=48,
        depth=2,
        num_heads=3,
        patch_size=16,
        input_size=(256, 256),
        pruning_cfg=pruning_cfg,
        patch_embed_cfg={
            "variant": variant,
            "use_mode_affine": True,
            "flatten_tokens": True,
            "search_use_pseudo_edges": True,
            "warmup_epochs": 1,
        },
    )


def _assert_output_abi(outputs: dict[str, torch.Tensor], batch_size: int) -> None:
    assert outputs["search/eye"].shape == (batch_size, 5)
    assert outputs["search/pupil"].shape == (batch_size, 7)
    assert outputs["event/pupil"].shape == (batch_size, 7)
    assert outputs["track/pupil"].shape == (batch_size, 8)
    assert outputs["search/state"].shape == (batch_size, 6)
    assert outputs["track/state"].shape == (batch_size, 6)
    assert outputs["search/mask_logits"].shape == (batch_size, 1, 256, 256)
    assert outputs["search/pooled"].shape == (batch_size, 48)
    assert outputs["pruning/active_width"].shape == (batch_size,)


def test_model_v3_output_abi_legacy():
    model = _model(variant="legacy")
    outputs = model(_batch())
    _assert_output_abi(outputs, 2)


def test_model_v3_output_abi_split1():
    model = _model(variant="split1")
    model.set_epoch_context(1)
    model.train()
    outputs = model(_batch())
    _assert_output_abi(outputs, 2)


def test_model_v3_output_abi_split2():
    model = _model(variant="split2")
    outputs = model(_batch())
    _assert_output_abi(outputs, 2)


def test_model_v3_pruning_width_slicing_changes_active_width():
    model = _model(
        variant="legacy",
        pruning_cfg={"enabled": True, "student_width": 0.5, "width_candidates": [0.5], "sample_strategy": "min"},
    )
    model.eval()
    outputs = model(_batch(batch_size=1))
    assert torch.isclose(outputs["pruning/active_width"][0], torch.tensor(0.5))


def test_split1_dense_reference_matches_cache_plus_event():
    model = _model(variant="split1")
    frontend = model.patch_frontend
    frame = torch.randn(2, 1, 256, 256)
    event = torch.randn(2, 2, 256, 256)
    dense_input = frontend.build_track_3ch(frame, event)
    dense_psum = frontend.forward_dense_reference(dense_input, mode="track", return_psum=True)
    cache = frontend.prime_cache(frame)
    cached_psum = frontend.forward_track(event, cache=cache, return_psum=True)
    assert dense_psum.shape == cached_psum.shape
    assert torch.allclose(dense_psum, cached_psum, atol=1e-5, rtol=1e-5)


def test_runtime_tracker_split2_keeps_patch_cache_state():
    model = _model(variant="split2")
    tracker = RuntimeHBTXRTracker(model)
    batch = _batch(batch_size=1)
    outputs1 = tracker.step(frame=batch["frame"], event=batch["event"])
    assert outputs1["runtime/state"] == "search"
    assert tracker.state.patch_cache is not None
    assert tracker.state.patch_cache.valid is True

    outputs2 = tracker.step(frame=batch["frame"], event=batch["event"])
    assert outputs2["runtime/state"] in {"search", "track"}
    assert tracker.state.patch_cache is not None


def test_structural_student_build_has_smaller_real_dimensions():
    cfg = normalize_compression_cfg(
        {
            "model": {
                "embed_dim": 48,
                "depth": 2,
                "num_heads": 3,
                "mlp_ratio": 2.0,
                "patch_size": 16,
                "input_size": [256, 256],
                "student": {
                    "embed_dim": 24,
                    "depth": 2,
                    "num_heads": 3,
                    "mlp_ratio": 2.0,
                    "patch_size": 16,
                    "adapter_hidden_dim": 24,
                    "prev_state_hidden_dim": 24,
                    "head_hidden_dim": 24,
                    "track_head_hidden_dim": 48,
                    "mask_hidden_dim": 32,
                },
            },
            "pruning": {
                "enabled": True,
                "scheme": "structural_width",
                "export": {"enabled": True},
            },
        }
    )
    teacher = build_model(cfg, role="teacher")
    student = build_model(cfg, role="student")

    assert teacher.embed_dim == 48
    assert student.embed_dim == 24
    assert student.structural_width_ratio == 0.5
    outputs = student(_batch(batch_size=1))
    assert outputs["search/pooled"].shape == (1, 24)
    assert torch.isclose(outputs["pruning/active_width"][0], torch.tensor(0.5))


def test_build_model_accepts_split_component_variants_from_cfg():
    cfg = {
        "model": {
            "embed_dim": 48,
            "depth": 2,
            "num_heads": 3,
            "mlp_ratio": 2.0,
            "patch_size": 16,
            "input_size": [256, 256],
            "patch_embed": {"variant": "legacy"},
            "components": {
                "encoder": {"variant": "split_v1"},
                "head_factory": {"variant": "split_v1"},
                "search_refiner": {"variant": "split_v1"},
                "search_branch": {"variant": "split_v1"},
                "event_branch": {"variant": "split_v1"},
                "track_branch": {"variant": "split_v1"},
                "runtime_policy": {"variant": "split_v1"},
                "track_codec": {"variant": "split_v1"},
            },
        }
    }

    model = build_model(cfg)
    outputs = model(_batch(batch_size=1))
    _assert_output_abi(outputs, 1)
    assert model.encoder.__class__.__name__ == "TrackerTokenEncoder"
    assert model.search_branch.__class__.__name__ == "SearchBranch"
    assert model.track_branch.__class__.__name__ == "TrackStateBranch"
    assert model.runtime_policy.__class__.__name__ == "RuntimeStepPolicy"
    assert model.track_codec.__class__.__name__ == "TrackStateCodec"


def test_tracker_config_normalizer_preserves_search_bool_and_string_forms():
    model = HBTXRTracker(
        embed_dim=48,
        depth=2,
        num_heads=3,
        patch_size=16,
        input_size=(256, 256),
        pruning_cfg={"enabled": True, "width_candidates": [0.5], "sample_strategy": "min"},
        patch_embed_cfg={"variant": "split1", "warmup_epochs": 2, "search_use_pseudo_edges": True},
        search_cfg={
            "eye_detector": "dense",
            "xy_from_mask_centroid": True,
            "roi_bbox_head": True,
            "mask_cascade": True,
        },
    )

    assert model.eye_detector_mode == "dense"
    assert model.search_use_mask_centroid is True
    assert model.search_use_roi_bbox_head is True
    assert model.search_use_mask_cascade is True
    assert model.patch_embed_variant == "split1"
    assert model.patch_warmup_epochs == 2
    assert model.patch_search_use_pseudo_edges is True
    assert model.legacy_width_masking is True


def test_build_model_uses_normalized_tracker_config_for_search_forms():
    cfg = {
        "model": {
            "embed_dim": 48,
            "depth": 2,
            "num_heads": 3,
            "patch_size": 16,
            "input_size": [256, 256],
            "patch_embed": {"variant": "split1", "warmup_epochs": 2, "search_use_pseudo_edges": True},
            "search": {
                "eye_detector": "dense",
                "xy_from_mask_centroid": True,
                "roi_bbox_head": True,
                "mask_cascade": True,
            },
            "mask": {"variant": "legacy"},
        },
        "runtime": {"relocalize_cooldown": 3},
        "pruning": {"enabled": True, "width_candidates": [0.5], "sample_strategy": "min"},
    }

    model = build_model(cfg)

    assert model.eye_detector_mode == "dense"
    assert model.search_use_mask_centroid is True
    assert model.search_use_roi_bbox_head is True
    assert model.search_use_mask_cascade is True
    assert model.patch_embed_variant == "split1"
    assert model.patch_warmup_epochs == 2
    assert model.patch_search_use_pseudo_edges is True
    assert model.runtime_cfg["relocalize_cooldown"] == 3
