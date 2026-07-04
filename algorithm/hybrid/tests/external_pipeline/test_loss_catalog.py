from __future__ import annotations

import torch

from src.training.losses import (
    AdaptiveWingLoss,
    AnchorConsistencyLoss,
    AuxClassificationLoss,
    BinaryFocalLoss,
    CIoULoss,
    CenterHeatmapLoss,
    CenterLoss,
    CenterOffsetLoss,
    ConfidenceAwareRelocalizationLoss,
    ConfidenceWeightedMSELoss,
    ConsistencyLoss,
    ConstraintCenterLoss,
    DiceLoss,
    DisplacementLoss,
    EllipseOverlapLoss,
    EventDensityWeighting,
    EventSearchBranchLosses,
    EventToFrameContrastiveLoss,
    EyeRegionLoss,
    FeatureConsistencyLoss,
    GIoULoss,
    GazeAngularLoss,
    GazeVelocityLoss,
    HBTXRStageLoss,
    MaskBCELoss,
    NLLHeatmapLoss,
    PredictionKDLoss,
    PropagatedStateLoss,
    RelationalKDLoss,
    SearchBranchLosses,
    TemporalDisplacementLoss,
    TrackBranchLosses,
    TrigRotationLoss,
    compute_distillation_losses,
    compute_regularization_ssl_losses,
)


def test_loss_catalog_instantiation_and_basic_forward():
    catalog = [
        AdaptiveWingLoss(),
        AnchorConsistencyLoss(),
        AuxClassificationLoss(),
        BinaryFocalLoss(),
        CIoULoss(),
        CenterHeatmapLoss(),
        CenterLoss(),
        CenterOffsetLoss(),
        ConfidenceAwareRelocalizationLoss(),
        ConfidenceWeightedMSELoss(),
        ConsistencyLoss(),
        ConstraintCenterLoss(),
        DiceLoss(),
        DisplacementLoss(),
        EllipseOverlapLoss(),
        EventDensityWeighting(),
        EventSearchBranchLosses(),
        EventToFrameContrastiveLoss(),
        EyeRegionLoss(),
        FeatureConsistencyLoss(),
        GIoULoss(),
        GazeAngularLoss(),
        GazeVelocityLoss(),
        HBTXRStageLoss({}),
        MaskBCELoss(),
        NLLHeatmapLoss(),
        PredictionKDLoss(),
        PropagatedStateLoss(),
        RelationalKDLoss(),
        SearchBranchLosses(),
        TemporalDisplacementLoss(),
        TrackBranchLosses(),
        TrigRotationLoss(),
    ]
    assert len(catalog) >= 30

    pred_state = torch.randn(2, 6)
    tgt_state = torch.randn(2, 6)
    assert torch.isfinite(CenterLoss()(pred_state[:, :2], tgt_state[:, :2]))
    assert torch.isfinite(DisplacementLoss()(pred_state[:, :2], tgt_state[:, :2]))
    assert torch.isfinite(ConsistencyLoss()(pred_state, tgt_state))


def test_distillation_stage_gating_defaults_block_stage1_event_track_terms():
    batch = {
        "frame": torch.zeros(2, 1, 4, 4),
        "annotation_quality": torch.ones(2),
        "mask_valid": torch.ones(2),
        "closed_eye_flag": torch.zeros(2),
        "valid_track": torch.ones(2),
    }
    search_pooled = torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    event_teacher = torch.tensor([[0.0, 0.5, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0]], dtype=torch.float32)
    event_student = event_teacher + 1.0
    track_teacher = torch.tensor([[0.0, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.0]], dtype=torch.float32)
    track_student = track_teacher + 1.0

    teacher_outputs = {
        "search/pooled": search_pooled,
        "event/pooled": event_teacher,
        "track/fused": track_teacher,
        "search/state": torch.zeros(2, 6),
        "event/state": torch.zeros(2, 6),
        "track/state": torch.zeros(2, 6),
        "search/eye": torch.zeros(2, 5),
        "search/pupil": torch.zeros(2, 7),
        "event/pupil": torch.zeros(2, 7),
        "track/pupil": torch.zeros(2, 8),
        "search/mask_logits": torch.zeros(2, 1, 2, 2),
        "search/aux": torch.zeros(2, 3),
        "event/aux": torch.zeros(2, 3),
    }
    outputs = {
        "search/pooled": search_pooled.clone(),
        "event/pooled": event_student,
        "track/fused": track_student,
        "search/state": torch.zeros(2, 6),
        "event/state": torch.ones(2, 6),
        "track/state": torch.ones(2, 6),
        "search/eye": torch.zeros(2, 5),
        "search/pupil": torch.zeros(2, 7),
        "event/pupil": torch.ones(2, 7),
        "track/pupil": torch.ones(2, 8),
        "search/mask_logits": torch.zeros(2, 1, 2, 2),
        "search/aux": torch.zeros(2, 3),
        "event/aux": torch.ones(2, 3),
    }
    cfg = {
        "enabled": True,
        "feature_similarity": True,
        "feature_weight": 1.0,
        "state_similarity": True,
        "state_weight": 1.0,
        "prediction_similarity": True,
        "prediction_weight": 1.0,
        "mask_similarity": True,
        "mask_weight": 1.0,
        "kd": {"enabled": True, "temperature": 1.0, "weight": 1.0},
        "rkd": {"enabled": True, "distance_weight": 1.0},
    }

    stage1_logs = compute_distillation_losses(batch, outputs, teacher_outputs, cfg, stage="stage1")
    stage2_logs = compute_distillation_losses(batch, outputs, teacher_outputs, cfg, stage="stage2")

    assert torch.isclose(stage1_logs["loss_distillation_total"], torch.zeros((), dtype=stage1_logs["loss_distillation_total"].dtype), atol=1e-6)
    assert float(stage2_logs["loss_distillation_total"]) > 0.0


def test_distillation_stage_gating_can_be_overridden_from_config():
    batch = {
        "frame": torch.zeros(2, 1, 4, 4),
        "annotation_quality": torch.ones(2),
        "mask_valid": torch.ones(2),
        "closed_eye_flag": torch.zeros(2),
        "valid_track": torch.ones(2),
    }
    teacher_outputs = {
        "search/pooled": torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32),
        "event/pooled": torch.zeros(2, 2),
        "search/state": torch.zeros(2, 6),
        "event/state": torch.zeros(2, 6),
        "search/eye": torch.zeros(2, 5),
        "search/pupil": torch.zeros(2, 7),
        "event/pupil": torch.zeros(2, 7),
        "search/mask_logits": torch.zeros(2, 1, 2, 2),
        "search/aux": torch.zeros(2, 3),
        "event/aux": torch.zeros(2, 3),
    }
    outputs = {
        "search/pooled": teacher_outputs["search/pooled"].clone(),
        "event/pooled": torch.ones(2, 2),
        "search/state": torch.zeros(2, 6),
        "event/state": torch.ones(2, 6),
        "search/eye": torch.zeros(2, 5),
        "search/pupil": torch.zeros(2, 7),
        "event/pupil": torch.ones(2, 7),
        "search/mask_logits": torch.zeros(2, 1, 2, 2),
        "search/aux": torch.zeros(2, 3),
        "event/aux": torch.ones(2, 3),
    }
    cfg = {
        "enabled": True,
        "feature_similarity": True,
        "feature_weight": 1.0,
        "state_similarity": True,
        "state_weight": 1.0,
        "prediction_similarity": True,
        "prediction_weight": 1.0,
        "mask_similarity": True,
        "mask_weight": 1.0,
        "targets": {
            "stage1": {
                "feature_keys": ["search/pooled", "event/pooled"],
                "state_keys": ["search/state", "event/state"],
                "prediction_keys": ["search/eye", "search/pupil", "event/pupil"],
                "mask_keys": ["search/mask_logits"],
                "logits_keys": ["search/eye", "search/pupil", "search/aux", "event/pupil", "event/aux"],
                "rkd_keys": ["search/pooled", "event/pooled"],
            }
        },
        "kd": {"enabled": True, "temperature": 1.0, "weight": 1.0},
        "rkd": {"enabled": False, "distance_weight": 1.0},
    }

    stage1_logs = compute_distillation_losses(batch, outputs, teacher_outputs, cfg, stage="stage1")
    assert float(stage1_logs["loss_distillation_total"]) > 0.0


def test_regularization_ssl_stage_gating_defaults_block_stage1_and_enable_stage2():
    batch = {
        "frame": torch.zeros(2, 1, 4, 4),
        "annotation_quality": torch.ones(2),
        "mask_valid": torch.ones(2),
        "closed_eye_flag": torch.zeros(2),
        "valid_track": torch.ones(2),
    }
    outputs = {
        "search/pooled": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "event/pooled": torch.zeros(2, 2),
        "search/state": torch.zeros(2, 6),
        "event/state": torch.ones(2, 6),
        "track/state": torch.ones(2, 6),
        "search/pupil": torch.zeros(2, 7),
        "event/pupil": torch.ones(2, 7),
        "track/pupil": torch.ones(2, 8),
    }
    cfg = {
        "enabled": True,
        "force_with_teacher": False,
        "feature_similarity": True,
        "feature_weight": 1.0,
        "state_similarity": True,
        "state_weight": 1.0,
        "prediction_similarity": True,
        "prediction_weight": 1.0,
        "cross_modal_contrastive": True,
        "contrastive_weight": 1.0,
        "self_consistency": True,
        "self_consistency_weight": 1.0,
    }

    stage1_logs = compute_regularization_ssl_losses(batch, outputs, cfg, stage="stage1")
    stage2_logs = compute_regularization_ssl_losses(batch, outputs, cfg, stage="stage2")

    assert torch.isclose(stage1_logs["loss_regularization_ssl_total"], torch.zeros((), dtype=stage1_logs["loss_regularization_ssl_total"].dtype), atol=1e-6)
    assert float(stage2_logs["loss_regularization_ssl_total"]) > 0.0


def test_regularization_ssl_stage_gating_can_be_overridden_from_config():
    batch = {
        "frame": torch.zeros(2, 1, 4, 4),
        "annotation_quality": torch.ones(2),
        "mask_valid": torch.ones(2),
        "closed_eye_flag": torch.zeros(2),
        "valid_track": torch.ones(2),
    }
    outputs = {
        "search/pooled": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "event/pooled": torch.zeros(2, 2),
        "search/state": torch.zeros(2, 6),
        "event/state": torch.ones(2, 6),
        "track/state": torch.ones(2, 6),
        "search/pupil": torch.zeros(2, 7),
        "event/pupil": torch.ones(2, 7),
        "track/pupil": torch.ones(2, 8),
    }
    cfg = {
        "enabled": True,
        "force_with_teacher": False,
        "feature_similarity": True,
        "feature_weight": 1.0,
        "state_similarity": True,
        "state_weight": 1.0,
        "prediction_similarity": True,
        "prediction_weight": 1.0,
        "cross_modal_contrastive": True,
        "contrastive_weight": 1.0,
        "self_consistency": False,
        "targets": {
            "stage1": {
                "feature_keys": ["search/pooled", "event/pooled"],
                "state_keys": ["search/state", "event/state"],
                "prediction_keys": ["search/pupil", "event/pupil"],
                "contrastive_keys": ["search/pooled", "event/pooled"],
                "self_keys": [],
            }
        },
    }

    stage1_logs = compute_regularization_ssl_losses(batch, outputs, cfg, stage="stage1")
    assert float(stage1_logs["loss_regularization_ssl_total"]) > 0.0
