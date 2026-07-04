from __future__ import annotations

import torch

from hbtxr.loss.bundles.track import (
    center_soft_threshold_loss,
    p10_boundary_loss,
    track_branch_losses,
    track_center_candidate_losses,
    track_center_heatmap_losses,
    track_center_refine_losses,
    track_target_override_losses,
    track_state_aux_losses,
    track_state_simdr_losses,
    track_state_simdr_target_positions,
)
from hbtxr.loss.stage2 import compute_stage2_losses
from hbtxr.models.heads import TrackCenterCandidateHead, TrackCenterRefineHead, TrackStateSimDRHead
from hbtxr.models.tracker.track_branch import TrackStateBranch, decode_heatmap_xy, decode_simdr_xy


def _inputs():
    pred = torch.zeros((2, 8), dtype=torch.float32)
    target = torch.zeros((2, 8), dtype=torch.float32)
    state = torch.tensor(
        [
            [2.0, 0.0, 8.0, 4.0, 0.0, 1.0],
            [0.0, 4.0, 8.0, 4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_state = torch.tensor(
        [
            [0.0, 0.0, 8.0, 4.0, 0.0, 1.0],
            [0.0, 0.0, 8.0, 4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    quality = torch.ones(2, dtype=torch.float32)
    track_geom = torch.ones(2, dtype=torch.float32)
    valid_track = torch.ones(2, dtype=torch.float32)
    return pred, target, state, target_state, quality, track_geom, valid_track


def test_track_center_l2_default_is_zero_weighted():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={},
    )

    assert "track_center_l2" in losses
    assert losses["track_center_l2"].item() == 0.0
    assert "track_center_hinge" in losses
    assert losses["track_center_hinge"].item() == 0.0
    assert "track_center_hinge_sq" in losses
    assert losses["track_center_hinge_sq"].item() == 0.0
    assert "track_p10_boundary" in losses
    assert losses["track_p10_boundary"].item() == 0.0
    assert "track_p10_soft_threshold" in losses
    assert losses["track_p10_soft_threshold"].item() == 0.0
    assert "track_p5_soft_threshold" in losses
    assert losses["track_p5_soft_threshold"].item() == 0.0
    assert "track_axis_log" in losses
    assert losses["track_axis_log"].item() == 0.0
    assert "track_angle_cos" in losses
    assert losses["track_angle_cos"].item() == 0.0


def test_track_center_l2_uses_decoded_state_center_error():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={"track_center_l2_weight": 0.5},
    )

    # Mean squared center distances are (4 + 16) / 2 = 10, then weight 0.5.
    assert torch.isclose(losses["track_center_l2"], torch.tensor(5.0))


def test_track_low_similarity_weighting_emphasizes_hard_rows():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        similarity_target=torch.tensor([0.0, 0.5], dtype=torch.float32),
        loss_cfg={
            "track_center_l2_weight": 1.0,
            "track_low_similarity_threshold": 0.5,
            "track_low_similarity_weight_scale": 1.0,
        },
    )

    # Center squared distances are 4 and 16. Low-sim row gets weight 2, other row weight 1.
    assert torch.isclose(losses["track_center_l2"], torch.tensor(8.0))


def test_track_center_hinge_penalizes_errors_above_margin():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={"track_center_hinge_weight": 2.0, "track_center_hinge_margin_px": 3.0},
    )

    # Center errors are 2 and 4 px. Hinge over 3 px is (0 + 1) / 2, then weight 2.
    assert torch.isclose(losses["track_center_hinge"], torch.tensor(1.0))


def test_track_center_hinge_sq_penalizes_large_errors_more_strongly():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={"track_center_hinge_sq_weight": 2.0, "track_center_hinge_margin_px": 3.0},
    )

    # Center errors are 2 and 4 px. Squared hinge over 3 px is (0^2 + 1^2) / 2, then weight 2.
    assert torch.isclose(losses["track_center_hinge_sq"], torch.tensor(1.0))


def test_p10_boundary_loss_focuses_near_margin_with_softplus_penalty():
    _, _, state, target_state, quality, track_geom, _ = _inputs()
    state = state.clone()
    state[:, :2] = torch.tensor([[10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)
    weights = quality * track_geom

    loss = p10_boundary_loss(
        state,
        target_state,
        weights,
        margin_px=10.0,
        band_px=4.0,
        temperature_px=1.0,
    )

    errors = torch.tensor([10.0, 20.0])
    band = torch.exp(-0.5 * ((errors - 10.0) / 4.0).pow(2))
    expected = (torch.nn.functional.softplus(errors - 10.0) * band).mean()
    assert torch.isclose(loss, expected)


def test_track_p10_boundary_loss_applies_config_weight():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={
            "track_p10_boundary_weight": 0.25,
            "track_p10_boundary_margin_px": 3.0,
            "track_p10_boundary_band_px": 4.0,
            "track_p10_boundary_temperature_px": 1.0,
        },
    )

    errors = torch.tensor([2.0, 4.0])
    band = torch.exp(-0.5 * ((errors - 3.0) / 4.0).pow(2))
    expected = 0.25 * (torch.nn.functional.softplus(errors - 3.0) * band).mean()
    assert torch.isclose(losses["track_p10_boundary"], expected)


def test_center_soft_threshold_loss_is_metric_aligned_bce_positive():
    _, _, state, target_state, quality, track_geom, _ = _inputs()
    state = state.clone()
    state[:, :2] = torch.tensor([[10.0, 0.0], [20.0, 0.0]], dtype=torch.float32)

    loss = center_soft_threshold_loss(
        state,
        target_state,
        quality * track_geom,
        margin_px=10.0,
        temperature_px=2.0,
    )

    errors = torch.tensor([10.0, 20.0])
    expected = torch.nn.functional.softplus((errors - 10.0) / 2.0).mean()
    assert torch.isclose(loss, expected)


def test_track_soft_threshold_losses_apply_config_weights():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={
            "track_p10_soft_threshold_weight": 0.25,
            "track_p10_soft_threshold_margin_px": 3.0,
            "track_p10_soft_threshold_temperature_px": 2.0,
            "track_p5_soft_threshold_weight": 0.5,
            "track_p5_soft_threshold_margin_px": 2.0,
            "track_p5_soft_threshold_temperature_px": 1.0,
        },
    )

    errors = torch.tensor([2.0, 4.0])
    expected_p10 = 0.25 * torch.nn.functional.softplus((errors - 3.0) / 2.0).mean()
    expected_p5 = 0.5 * torch.nn.functional.softplus(errors - 2.0).mean()
    assert torch.isclose(losses["track_p10_soft_threshold"], expected_p10)
    assert torch.isclose(losses["track_p5_soft_threshold"], expected_p5)


def test_track_axis_log_and_angle_cos_use_decoded_state():
    pred, target, state, target_state, quality, track_geom, valid_track = _inputs()
    state = state.clone()
    target_state = target_state.clone()
    state[:, 2:4] = torch.tensor([[16.0, 4.0], [8.0, 2.0]])
    target_state[:, 2:4] = torch.tensor([[8.0, 4.0], [8.0, 4.0]])
    state[:, 4:6] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target_state[:, 4:6] = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    losses = track_branch_losses(
        pred=pred,
        target=target,
        state=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        valid_track=valid_track,
        loss_cfg={"track_axis_log_weight": 2.0, "track_angle_cos_weight": 3.0},
    )

    # Each sample has one log-axis delta of log(2), averaged over two axes, then weighted.
    expected_axis = 0.5 * torch.log(torch.tensor(2.0)).pow(2)
    assert torch.isclose(losses["track_axis_log"], expected_axis)
    # Double-angle unit vectors have cosine distances [1, 0], averaged then weighted by 3.
    assert torch.isclose(losses["track_angle_cos"], torch.tensor(1.5))


def test_track_state_aux_losses_default_to_zero_weighted():
    _, _, state, target_state, quality, track_geom, _ = _inputs()

    losses = track_state_aux_losses(
        state_aux=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={},
    )

    assert losses["track_state_aux_center_l2"].item() == 0.0
    assert losses["track_state_aux_p10_boundary"].item() == 0.0
    assert losses["track_state_aux_p10_soft_threshold"].item() == 0.0
    assert losses["track_state_aux_p5_soft_threshold"].item() == 0.0
    assert losses["track_state_aux_axis_log"].item() == 0.0
    assert losses["track_state_aux_angle_cos"].item() == 0.0


def test_track_state_aux_losses_use_direct_state():
    _, _, state, target_state, quality, track_geom, _ = _inputs()
    state = state.clone()
    state[:, 2:4] = torch.tensor([[16.0, 4.0], [8.0, 2.0]])
    state[:, 4:6] = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    losses = track_state_aux_losses(
        state_aux=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={
            "track_state_aux_center_l2_weight": 0.5,
            "track_state_aux_axis_log_weight": 2.0,
            "track_state_aux_angle_cos_weight": 3.0,
        },
    )

    assert torch.isclose(losses["track_state_aux_center_l2"], torch.tensor(5.0))
    expected_axis = 0.5 * torch.log(torch.tensor(2.0)).pow(2)
    assert torch.isclose(losses["track_state_aux_axis_log"], expected_axis)
    assert torch.isclose(losses["track_state_aux_angle_cos"], torch.tensor(1.5))


def test_track_state_aux_p10_boundary_loss_applies_config_weight():
    _, _, state, target_state, quality, track_geom, _ = _inputs()

    losses = track_state_aux_losses(
        state_aux=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={
            "track_state_aux_p10_boundary_weight": 0.5,
            "track_p10_boundary_margin_px": 3.0,
            "track_p10_boundary_band_px": 4.0,
            "track_p10_boundary_temperature_px": 1.0,
        },
    )

    errors = torch.tensor([2.0, 4.0])
    band = torch.exp(-0.5 * ((errors - 3.0) / 4.0).pow(2))
    expected = 0.5 * (torch.nn.functional.softplus(errors - 3.0) * band).mean()
    assert torch.isclose(losses["track_state_aux_p10_boundary"], expected)


def test_track_state_aux_soft_threshold_losses_apply_config_weights():
    _, _, state, target_state, quality, track_geom, _ = _inputs()

    losses = track_state_aux_losses(
        state_aux=state,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={
            "track_state_aux_p10_soft_threshold_weight": 0.25,
            "track_state_aux_p10_soft_threshold_margin_px": 3.0,
            "track_state_aux_p10_soft_threshold_temperature_px": 2.0,
            "track_state_aux_p5_soft_threshold_weight": 0.5,
            "track_state_aux_p5_soft_threshold_margin_px": 2.0,
            "track_state_aux_p5_soft_threshold_temperature_px": 1.0,
        },
    )

    errors = torch.tensor([2.0, 4.0])
    expected_p10 = 0.25 * torch.nn.functional.softplus((errors - 3.0) / 2.0).mean()
    expected_p5 = 0.5 * torch.nn.functional.softplus(errors - 2.0).mean()
    assert torch.isclose(losses["track_state_aux_p10_soft_threshold"], expected_p10)
    assert torch.isclose(losses["track_state_aux_p5_soft_threshold"], expected_p5)


def test_track_target_override_losses_default_to_zero_weighted():
    _, _, state, target_state, quality, track_geom, _ = _inputs()

    losses = track_target_override_losses(
        state=state,
        override_state=target_state,
        override_weight=torch.ones(2, dtype=torch.float32),
        quality=quality,
        track_geom=track_geom,
        loss_cfg={},
    )

    assert losses["track_target_override_center_l2"].item() == 0.0
    assert losses["track_target_override_axis_log"].item() == 0.0
    assert losses["track_target_override_angle_cos"].item() == 0.0


def test_stage2_track_target_override_loss_is_additive_and_keeps_cur_state():
    _, target, state, cur_state, quality, track_geom, valid_track = _inputs()
    override_state = cur_state.clone()
    override_state[:, :2] = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float32)
    batch = {
        "pupil_track_target": torch.cat([target[:, :6], torch.ones((2, 2), dtype=torch.float32)], dim=-1),
        "cur_state": cur_state,
        "annotation_quality": quality,
        "mask_valid": track_geom,
        "closed_eye_flag": torch.zeros(2, dtype=torch.float32),
        "valid_track": valid_track,
        "track_target_override_state": override_state,
        "track_target_override_weight": torch.ones(2, dtype=torch.float32),
    }
    outputs = {
        "track/pupil": torch.zeros((2, 8), dtype=torch.float32),
        "track/state": state,
    }

    logs = compute_stage2_losses(
        batch,
        outputs,
        {
            "track_center_l2_weight": 1.0,
            "track_target_override_center_l2_weight": 0.5,
        },
    )

    assert torch.isclose(logs["loss_track_center_l2"], torch.tensor(10.0))
    override_errors = ((state[:, :2] - override_state[:, :2]) ** 2).sum(dim=-1)
    assert torch.isclose(logs["loss_track_target_override_center_l2"], 0.5 * override_errors.mean())


def test_track_state_simdr_target_positions_respect_image_bounds():
    target_state = torch.tensor(
        [
            [0.0, 255.0, 8.0, 4.0, 0.0, 1.0],
            [128.0, -4.0, 8.0, 4.0, 0.0, 1.0],
            [300.0, 127.5, 8.0, 4.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    positions = track_state_simdr_target_positions(target_state, bins=64, coordinate_max=255.0)

    assert torch.allclose(positions[0], torch.tensor([0.0, 63.0]))
    assert torch.isclose(positions[1, 0], torch.tensor(128.0 * 63.0 / 255.0))
    assert torch.isclose(positions[1, 1], torch.tensor(0.0))
    assert torch.isclose(positions[2, 0], torch.tensor(63.0))
    assert torch.isclose(positions[2, 1], torch.tensor(31.5))


def test_decode_simdr_xy_maps_axis_logits_to_soft_coordinates():
    logits = torch.full((2, 2, 4), -10.0)
    logits[0, 0, 0] = 10.0
    logits[0, 1, 3] = 10.0
    logits[1, 0, 1] = 10.0
    logits[1, 1, 2] = 10.0

    xy = decode_simdr_xy(logits, coordinate_max=30.0)

    assert torch.allclose(xy[0], torch.tensor([0.0, 30.0]), atol=1.0e-4)
    assert torch.allclose(xy[1], torch.tensor([10.0, 20.0]), atol=1.0e-4)


def test_decode_heatmap_xy_maps_peak_and_offset_to_coordinates():
    logits = torch.full((1, 4, 4), -10.0)
    logits[0, 2, 1] = 10.0
    offset = torch.zeros((1, 2, 4, 4))
    offset[0, 0, 2, 1] = 0.25
    offset[0, 1, 2, 1] = -0.25

    xy = decode_heatmap_xy(logits, offset, coordinate_max=32.0)

    assert torch.allclose(xy[0], torch.tensor([14.0, 18.0]), atol=1.0e-4)


def test_track_center_heatmap_losses_default_to_zero_weighted():
    _, _, _, target_state, quality, track_geom, _ = _inputs()
    logits = torch.zeros((2, 8, 8), dtype=torch.float32)
    offset = torch.zeros((2, 2, 8, 8), dtype=torch.float32)

    losses = track_center_heatmap_losses(
        logits=logits,
        offset=offset,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={},
    )

    assert losses["track_center_heatmap"].item() == 0.0
    assert losses["track_center_heatmap_offset"].item() == 0.0


def test_stage2_includes_track_center_heatmap_loss_log():
    _, target, state, target_state, quality, track_geom, valid_track = _inputs()
    batch = {
        "pupil_track_target": torch.cat([target[:, :6], torch.ones((2, 2), dtype=torch.float32)], dim=-1),
        "cur_state": target_state,
        "annotation_quality": quality,
        "mask_valid": track_geom,
        "closed_eye_flag": torch.zeros(2, dtype=torch.float32),
        "valid_track": valid_track,
    }
    outputs = {
        "track/pupil": torch.zeros((2, 8), dtype=torch.float32),
        "track/state": state,
        "track/center_heatmap_logits": torch.zeros((2, 8, 8), dtype=torch.float32),
        "track/center_heatmap_offset": torch.zeros((2, 2, 8, 8), dtype=torch.float32),
    }

    logs = compute_stage2_losses(
        batch,
        outputs,
        {
            "track_center_heatmap_weight": 0.5,
            "track_center_heatmap_offset_weight": 0.25,
            "track_p10_soft_threshold_weight": 0.1,
            "track_p5_soft_threshold_weight": 0.1,
        },
    )

    assert "loss_track_center_heatmap" in logs
    assert "loss_track_center_heatmap_offset" in logs
    assert "loss_track_p10_soft_threshold" in logs
    assert "loss_track_p5_soft_threshold" in logs
    assert logs["loss_track_center_heatmap"].item() > 0.0


def test_track_state_simdr_head_outputs_axis_logits():
    head = TrackStateSimDRHead(in_dim=4, hidden_dim=8, bins=16)
    logits = head(torch.zeros((3, 4), dtype=torch.float32))

    assert logits.shape == (3, 2, 16)


def test_track_state_simdr_losses_default_to_zero_weighted():
    _, _, _, target_state, quality, track_geom, _ = _inputs()
    logits = torch.zeros((2, 2, 64), dtype=torch.float32)

    losses = track_state_simdr_losses(
        logits=logits,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={},
    )

    assert losses["track_state_simdr"].item() == 0.0


def test_track_state_simdr_losses_apply_weight():
    _, _, _, target_state, quality, track_geom, _ = _inputs()
    logits = torch.zeros((2, 2, 8), dtype=torch.float32)

    losses = track_state_simdr_losses(
        logits=logits,
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={
            "track_state_simdr_weight": 0.5,
            "track_state_simdr_bins": 8,
            "track_state_simdr_sigma": 0.0,
        },
    )

    expected = 0.5 * torch.log(torch.tensor(8.0))
    assert torch.isclose(losses["track_state_simdr"], expected)


def test_stage2_includes_track_state_simdr_loss_log():
    _, target, state, target_state, quality, track_geom, valid_track = _inputs()
    batch = {
        "pupil_track_target": torch.cat([target[:, :6], torch.ones((2, 2), dtype=torch.float32)], dim=-1),
        "cur_state": target_state,
        "annotation_quality": quality,
        "mask_valid": track_geom,
        "closed_eye_flag": torch.zeros(2, dtype=torch.float32),
        "valid_track": valid_track,
    }
    outputs = {
        "track/pupil": torch.zeros((2, 8), dtype=torch.float32),
        "track/state": state,
        "track/state_simdr": torch.zeros((2, 2, 8), dtype=torch.float32),
    }

    logs = compute_stage2_losses(
        batch,
        outputs,
        {
            "track_state_simdr_weight": 0.5,
            "track_state_simdr_bins": 8,
            "track_state_simdr_sigma": 0.0,
        },
    )

    assert "loss_track_state_simdr" in logs
    assert logs["loss_track_state_simdr"].item() > 0.0


def test_track_center_refine_head_initializes_as_identity_delta():
    head = TrackCenterRefineHead(in_dim=4, hidden_dim=8, max_delta_px=3.0)
    out = head(torch.ones((2, 4), dtype=torch.float32))

    assert out["delta"].shape == (2, 2)
    assert torch.allclose(out["delta"], torch.zeros((2, 2), dtype=torch.float32))
    assert torch.allclose(out["gate"], torch.full((2, 1), 0.5, dtype=torch.float32))


def test_track_center_refine_losses_default_to_zero_weighted():
    _, _, _, _, quality, track_geom, _ = _inputs()
    losses = track_center_refine_losses(
        delta=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        quality=quality,
        track_geom=track_geom,
        loss_cfg={},
    )

    assert losses["track_center_refine_delta_l2"].item() == 0.0
    assert losses["track_center_refine_delta_l1"].item() == 0.0


def test_stage2_includes_track_center_refine_loss_log():
    _, target, state, target_state, quality, track_geom, valid_track = _inputs()
    batch = {
        "pupil_track_target": torch.cat([target[:, :6], torch.ones((2, 2), dtype=torch.float32)], dim=-1),
        "cur_state": target_state,
        "annotation_quality": quality,
        "mask_valid": track_geom,
        "closed_eye_flag": torch.zeros(2, dtype=torch.float32),
        "valid_track": valid_track,
    }
    outputs = {
        "track/pupil": torch.zeros((2, 8), dtype=torch.float32),
        "track/state": state,
        "track/center_refine_delta": torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
    }

    logs = compute_stage2_losses(
        batch,
        outputs,
        {
            "track_center_refine_delta_l2_weight": 0.5,
            "track_center_refine_delta_l1_weight": 0.25,
        },
    )

    assert "loss_track_center_refine_delta_l2" in logs
    assert "loss_track_center_refine_delta_l1" in logs
    assert torch.isclose(logs["loss_track_center_refine_delta_l2"], torch.tensor(1.25))
    assert torch.isclose(logs["loss_track_center_refine_delta_l1"], torch.tensor(0.375))


def test_track_center_refine_can_route_into_track_state():
    class ConstantTrackHead(torch.nn.Module):
        def forward(self, fused: torch.Tensor) -> torch.Tensor:
            return torch.zeros((fused.shape[0], 8), dtype=fused.dtype, device=fused.device)

    class ConstantRefineHead(torch.nn.Module):
        def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "delta": torch.tensor([[1.5, -0.5]], dtype=fused.dtype, device=fused.device).expand(fused.shape[0], -1),
                "gate_logit": torch.zeros((fused.shape[0], 1), dtype=fused.dtype, device=fused.device),
                "gate": torch.full((fused.shape[0], 1), 0.5, dtype=fused.dtype, device=fused.device),
            }

    branch = TrackStateBranch(
        prev_state_encoder=torch.nn.Identity(),
        track_head=ConstantTrackHead(),
        track_center_refine_head=ConstantRefineHead(),
        apply_width_mask=lambda tensor, active_dim: tensor,
        apply_track_mask=lambda tensor, active_dim: tensor,
        decode_state=lambda prev_state, logits: prev_state[:, :6],
        refine_as_track_state=True,
        refine_blend=1.0,
    )
    prev_state = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.0, 1.0]], dtype=torch.float32)
    outputs = branch.forward(
        pooled=torch.zeros((1, 2), dtype=torch.float32),
        prev_state=prev_state,
        active_dim=2,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert torch.allclose(outputs["track/state_pre_refine"][:, :2], torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(outputs["track/state"][:, :2], torch.tensor([[11.5, 19.5]]))


def test_track_center_candidate_head_initializes_zero_candidates():
    head = TrackCenterCandidateHead(in_dim=4, hidden_dim=8, num_candidates=3, max_delta_px=5.0)
    out = head(torch.ones((2, 4), dtype=torch.float32))

    assert out["delta"].shape == (2, 3, 2)
    assert out["logits"].shape == (2, 3)
    assert torch.allclose(out["delta"], torch.zeros((2, 3, 2), dtype=torch.float32))
    assert torch.allclose(out["logits"], torch.zeros((2, 3), dtype=torch.float32))


def test_track_center_candidate_losses_default_to_zero_weighted():
    _, _, _, target_state, quality, track_geom, _ = _inputs()
    losses = track_center_candidate_losses(
        candidate_xy=target_state[:, None, :2].repeat(1, 2, 1),
        candidate_logits=torch.zeros((2, 2), dtype=torch.float32),
        candidate_delta=torch.ones((2, 2, 2), dtype=torch.float32),
        target_state=target_state,
        quality=quality,
        track_geom=track_geom,
        loss_cfg={},
    )

    assert losses["track_center_candidate_p10_bce"].item() == 0.0
    assert losses["track_center_candidate_min_soft_threshold"].item() == 0.0
    assert losses["track_center_candidate_delta_l2"].item() == 0.0


def test_stage2_includes_track_center_candidate_loss_log():
    _, target, state, target_state, quality, track_geom, valid_track = _inputs()
    batch = {
        "pupil_track_target": torch.cat([target[:, :6], torch.ones((2, 2), dtype=torch.float32)], dim=-1),
        "cur_state": target_state,
        "annotation_quality": quality,
        "mask_valid": track_geom,
        "closed_eye_flag": torch.zeros(2, dtype=torch.float32),
        "valid_track": valid_track,
    }
    outputs = {
        "track/pupil": torch.zeros((2, 8), dtype=torch.float32),
        "track/state": state,
        "track/center_candidate_xy": target_state[:, None, :2].repeat(1, 2, 1),
        "track/center_candidate_logits": torch.zeros((2, 2), dtype=torch.float32),
        "track/center_candidate_delta": torch.ones((2, 2, 2), dtype=torch.float32),
    }

    logs = compute_stage2_losses(
        batch,
        outputs,
        {
            "track_center_candidate_p10_bce_weight": 0.5,
            "track_center_candidate_min_soft_threshold_weight": 0.25,
            "track_center_candidate_delta_l2_weight": 0.1,
        },
    )

    assert "loss_track_center_candidate_p10_bce" in logs
    assert "loss_track_center_candidate_min_soft_threshold" in logs
    assert "loss_track_center_candidate_delta_l2" in logs
    assert logs["loss_track_center_candidate_p10_bce"].item() > 0.0
    assert logs["loss_track_center_candidate_delta_l2"].item() > 0.0


def test_track_center_candidate_can_route_top_logit_into_track_state():
    class ConstantTrackHead(torch.nn.Module):
        def forward(self, fused: torch.Tensor) -> torch.Tensor:
            return torch.zeros((fused.shape[0], 8), dtype=fused.dtype, device=fused.device)

    class ConstantCandidateHead(torch.nn.Module):
        def forward(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
            batch = fused.shape[0]
            delta = torch.tensor([[[1.0, 0.0], [3.0, -2.0]]], dtype=fused.dtype, device=fused.device).expand(batch, -1, -1)
            logits = torch.tensor([[0.0, 2.0]], dtype=fused.dtype, device=fused.device).expand(batch, -1)
            return {"delta": delta, "logits": logits}

    branch = TrackStateBranch(
        prev_state_encoder=torch.nn.Identity(),
        track_head=ConstantTrackHead(),
        track_center_candidate_head=ConstantCandidateHead(),
        apply_width_mask=lambda tensor, active_dim: tensor,
        apply_track_mask=lambda tensor, active_dim: tensor,
        decode_state=lambda prev_state, logits: prev_state[:, :6],
        candidate_as_track_state=True,
        candidate_blend=1.0,
    )
    prev_state = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.0, 1.0]], dtype=torch.float32)
    outputs = branch.forward(
        pooled=torch.zeros((1, 2), dtype=torch.float32),
        prev_state=prev_state,
        active_dim=2,
        device=torch.device("cpu"),
        batch_size=1,
    )

    assert torch.allclose(outputs["track/center_candidate_selected_xy"], torch.tensor([[13.0, 18.0]]))
    assert torch.allclose(outputs["track/state_pre_candidate"][:, :2], torch.tensor([[10.0, 20.0]]))
    assert torch.allclose(outputs["track/state"][:, :2], torch.tensor([[13.0, 18.0]]))
