from __future__ import annotations

import torch

import hbtxr.loss.metrics as split_metrics
import hbtxr.loss.stage as facade_stage
import hbtxr.loss.stage1 as split_stage1
import hbtxr.loss.stage2 as split_stage2


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


def _batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    cur_state = torch.tensor(
        [[128.0, 120.0, 24.0, 18.0, 1.0, 0.0], [130.0, 118.0, 22.0, 16.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    prev_state = torch.tensor(
        [[126.0, 119.0, 23.0, 17.0, 1.0, 0.0], [131.0, 119.0, 21.0, 15.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    eye_target = torch.tensor(
        [[128.0, 120.0, 160.0, 160.0, 1.0], [130.0, 118.0, 158.0, 158.0, 1.0]],
        dtype=torch.float32,
    )
    pupil_region_target = torch.tensor(
        [[128.0, 120.0, 48.0, 36.0, 1.0], [130.0, 118.0, 46.0, 34.0, 1.0]],
        dtype=torch.float32,
    )
    pupil_search_target = torch.tensor(
        [[128.0, 120.0, 24.0, 18.0, 1.0, 0.0, 1.0], [130.0, 118.0, 22.0, 16.0, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    pupil_track_target = torch.tensor(
        [[0.2, -0.1, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [-0.1, 0.2, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    batch = {
        "frame": torch.randn(batch_size, 1, 64, 64),
        "eye_target": eye_target[:batch_size].clone(),
        "pupil_region_target": pupil_region_target[:batch_size].clone(),
        "cur_state": cur_state[:batch_size].clone(),
        "prev_state": prev_state[:batch_size].clone(),
        "pupil_search_target": pupil_search_target[:batch_size].clone(),
        "pupil_track_target": pupil_track_target[:batch_size].clone(),
        "mask_target": torch.rand(batch_size, 1, 64, 64),
        "constraint_center": cur_state[:batch_size, :2].clone(),
        "annotation_quality": torch.ones(batch_size),
        "closed_eye_flag": torch.zeros(batch_size),
        "mask_valid": torch.ones(batch_size),
        "valid_track": torch.ones(batch_size),
        "aux_target": torch.zeros(batch_size, dtype=torch.long),
    }
    return batch


def _stage1_outputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "search/eye": torch.randn(batch_size, 5),
        "search/pupil": torch.randn(batch_size, 7),
        "search/state": torch.randn(batch_size, 6),
        "search/pupil_bbox": torch.randn(batch_size, 5),
        "search/pupil_obb": torch.randn(batch_size, 6),
        "search/mask_logits": torch.randn(batch_size, 1, 64, 64),
        "search/mask_coarse_logits": torch.randn(batch_size, 1, 64, 64),
        "search/aux": torch.randn(batch_size, 5),
    }


def _stage2_outputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
    outputs = _stage1_outputs(batch_size)
    outputs.update(
        {
            "event/pupil": torch.randn(batch_size, 7),
            "event/state": torch.randn(batch_size, 6),
            "event/pupil_bbox": torch.randn(batch_size, 5),
            "event/pupil_obb": torch.randn(batch_size, 6),
            "event/aux": torch.randn(batch_size, 5),
            "track/pupil": torch.randn(batch_size, 8),
            "track/state": torch.randn(batch_size, 6),
        }
    )
    return outputs


def _assert_tensor_dict_allclose(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> None:
    assert actual.keys() == expected.keys()
    for key in actual:
        assert torch.allclose(actual[key], expected[key], atol=1e-6, rtol=1e-6), key


def test_loss_stage_facade_exports_split_functions():
    assert facade_stage.compute_stage1_losses is split_stage1.compute_stage1_losses
    assert facade_stage.compute_stage2_losses is split_stage2.compute_stage2_losses
    assert facade_stage.compute_metrics is split_metrics.compute_metrics


def test_stage1_module_split_matches_facade_outputs():
    torch.manual_seed(7)
    batch = _batch()
    outputs = _stage1_outputs()
    loss_cfg = _loss_cfg()

    actual = facade_stage.compute_stage1_losses(batch, outputs, loss_cfg)
    expected = split_stage1.compute_stage1_losses(batch, outputs, loss_cfg)

    _assert_tensor_dict_allclose(actual, expected)


def test_stage2_module_split_matches_facade_outputs():
    torch.manual_seed(11)
    batch = _batch()
    outputs = _stage2_outputs()
    loss_cfg = _loss_cfg()

    actual = facade_stage.compute_stage2_losses(batch, outputs, loss_cfg)
    expected = split_stage2.compute_stage2_losses(batch, outputs, loss_cfg)

    _assert_tensor_dict_allclose(actual, expected)


def test_metrics_module_split_matches_facade_outputs():
    torch.manual_seed(13)
    batch = _batch()
    outputs = _stage2_outputs()

    actual = facade_stage.compute_metrics(batch, outputs)
    expected = split_metrics.compute_metrics(batch, outputs)

    _assert_tensor_dict_allclose(actual, expected)
