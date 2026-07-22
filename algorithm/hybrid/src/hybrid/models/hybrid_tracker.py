from __future__ import annotations

from typing import Any
import inspect
from dataclasses import field, fields, make_dataclass

import torch
from torch import nn

from .adapters import ModalityAdapter
from .backbone import PartialDeiTTiny
from .controller import TrackSearchSchedulerFSM
from .patch_embeddings import PatchEmbedCache, build_patch_frontend
from .tracker import (
    EventStateBranch,
    RuntimeStepPolicy,
    SearchBranch,
    SearchMaskGuidanceRefiner,
    TrackerConfigNormalizer,
    TrackerNormalizedConfig,
    TrackStateCodec,
    TrackStateBranch,
    TrackerHeadFactory,
    TrackerTokenEncoder,
    resolve_tracker_component,
)


class HBTXRTracker(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        mlp_hidden_dim: int | None = None,
        patch_size: int = 16,
        input_size: tuple[int, int] = (256, 256),
        frame_input_size: tuple[int, int] | None = None,
        event_input_size: tuple[int, int] | None = None,
        event_cut_depth: int | None = None,
        track_depth: int | None = None,
        dropout: float = 0.0,
        aux_classes: int = 5,
        adapter_hidden_dim: int | None = None,
        prev_state_hidden_dim: int | None = None,
        head_hidden_dim: int | None = None,
        track_head_hidden_dim: int | None = None,
        mask_hidden_dim: int | None = None,
        structural_width_ratio: float = 1.0,
        eye_head_variant: str = "legacy",
        eye_reg_max: int = 1,
        search_head_variant: str = "legacy",
        search_head_residual_hidden_dim: int | None = None,
        enable_eye_head: bool = True,
        enable_search_head: bool = True,
        enable_event_head: bool = True,
        enable_track_head: bool = True,
        enable_mask_head: bool = True,
        enable_aux_head: bool = True,
        enable_search_bbox_aux_head: bool = False,
        enable_event_bbox_aux_head: bool = False,
        enable_search_obb_aux_head: bool = False,
        enable_event_obb_aux_head: bool = False,
        enable_search_center_candidate_head: bool = False,
        enable_track_state_aux_head: bool = False,
        enable_track_state_simdr_head: bool = False,
        enable_track_center_heatmap_head: bool = False,
        enable_track_center_refine_head: bool = False,
        enable_track_center_candidate_head: bool = False,
        track_state_simdr_bins: int = 64,
        track_center_heatmap_grid: int = 32,
        track_center_refine_max_delta_px: float = 4.0,
        track_center_candidate_count: int = 4,
        track_center_candidate_max_delta_px: float = 8.0,
        search_center_candidate_count: int = 4,
        search_center_candidate_max_delta_px: float = 8.0,
        track_state_simdr_as_track_state: bool = False,
        track_state_simdr_coordinate_max: float = 255.0,
        track_state_simdr_blend: float = 1.0,
        track_center_heatmap_as_track_state: bool = False,
        track_center_heatmap_coordinate_max: float = 255.0,
        track_center_heatmap_blend: float = 1.0,
        track_center_refine_as_track_state: bool = False,
        track_center_refine_blend: float = 1.0,
        track_center_candidate_as_track_state: bool = False,
        track_center_candidate_blend: float = 1.0,
        search_center_candidate_as_search_state: bool = False,
        search_center_candidate_blend: float = 1.0,
        mask_variant: str = "legacy",
        runtime_cfg: dict[str, Any] | None = None,
        pruning_cfg: dict[str, Any] | None = None,
        patch_embed_cfg: dict[str, Any] | None = None,
        search_cfg: dict[str, Any] | None = None,
        mask_cfg: dict[str, Any] | None = None,
        component_cfg: dict[str, Any] | None = None,
        tracker_cfg: TrackerNormalizedConfig | None = None,
    ) -> None:
        super().__init__()
        normalized = tracker_cfg or TrackerConfigNormalizer(
            embed_dim=embed_dim,
            structural_width_ratio=structural_width_ratio,
            eye_head_variant=eye_head_variant,
            eye_reg_max=eye_reg_max,
            mask_variant=mask_variant,
            runtime_cfg=runtime_cfg,
            pruning_cfg=pruning_cfg,
            patch_embed_cfg=patch_embed_cfg,
            search_cfg=search_cfg,
            mask_cfg=mask_cfg,
            component_cfg=component_cfg,
        ).build()
        self.embed_dim = int(embed_dim)
        self.input_size = tuple(int(v) for v in input_size)
        self.frame_input_size = tuple(int(v) for v in (frame_input_size or self.input_size))
        self.event_input_size = tuple(int(v) for v in (event_input_size or self.input_size))
        self.event_cut_depth = None if event_cut_depth is None else int(event_cut_depth)
        self.track_depth = int(track_depth if track_depth is not None else event_cut_depth) if (track_depth is not None or event_cut_depth is not None) else None
        self.pruning_cfg = normalized.pruning_cfg
        self.patch_embed_cfg = normalized.patch_embed_cfg
        self.search_cfg = normalized.search_cfg
        self.mask_cfg = normalized.mask_cfg
        self.component_cfg = normalized.component_cfg
        self.runtime_cfg = normalized.runtime_cfg
        self.eye_detector_cfg = normalized.eye_detector_cfg
        self.eye_detector_mode = normalized.eye_detector_mode
        self.search_use_mask_centroid = normalized.search_use_mask_centroid
        self.roi_bbox_cfg = normalized.roi_bbox_cfg
        self.search_use_roi_bbox_head = normalized.search_use_roi_bbox_head
        self.roi_bbox_use_external_eye = normalized.roi_bbox_use_external_eye
        self.mask_cascade_cfg = normalized.mask_cascade_cfg
        self.search_use_mask_cascade = normalized.search_use_mask_cascade
        self.search_mask_cascade_use_xy = normalized.search_mask_cascade_use_xy
        self.search_mask_cascade_use_ab = normalized.search_mask_cascade_use_ab
        self.search_mask_cascade_detach = normalized.search_mask_cascade_detach
        self.search_mask_cascade_xy_scale = normalized.search_mask_cascade_xy_scale
        self.search_mask_cascade_ab_scale = normalized.search_mask_cascade_ab_scale
        self.search_mask_cascade_ab_mode = normalized.search_mask_cascade_ab_mode
        self.search_mask_cascade_ab_limit = normalized.search_mask_cascade_ab_limit
        self.search_mask_cascade_min_ab = normalized.search_mask_cascade_min_ab
        self.structural_width_ratio = normalized.structural_width_ratio
        self.legacy_width_masking = normalized.legacy_width_masking
        self.patch_embed_variant = normalized.patch_embed_variant
        self.patch_warmup_epochs = normalized.patch_warmup_epochs
        self.patch_search_use_pseudo_edges = normalized.patch_search_use_pseudo_edges
        self.current_epoch = 0
        self.eye_head_variant = normalized.eye_head_variant
        self.eye_reg_max = normalized.eye_reg_max
        self.mask_variant = normalized.mask_variant

        self.patch_frontend = build_patch_frontend(
            variant=self.patch_embed_variant,
            embed_dim=embed_dim,
            patch_size=patch_size,
            use_mode_affine=bool(self.patch_embed_cfg.get("use_mode_affine", True)),
            flatten_tokens=bool(self.patch_embed_cfg.get("flatten_tokens", True)),
            alpha=float(self.patch_embed_cfg.get("alpha", 0.5)),
            beta=float(self.patch_embed_cfg.get("beta", 0.5)),
        )
        self.frame_adapter = ModalityAdapter(embed_dim=embed_dim, hidden_dim=adapter_hidden_dim)
        self.event_adapter = ModalityAdapter(embed_dim=embed_dim, hidden_dim=adapter_hidden_dim)
        self.backbone = PartialDeiTTiny(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        encoder_cls, encoder_kwargs = resolve_tracker_component(self.component_cfg, "encoder")
        head_factory_cls, head_factory_kwargs = resolve_tracker_component(self.component_cfg, "head_factory")
        refiner_cls, refiner_kwargs = resolve_tracker_component(self.component_cfg, "search_refiner")
        search_branch_cls, search_branch_kwargs = resolve_tracker_component(self.component_cfg, "search_branch")
        event_branch_cls, event_branch_kwargs = resolve_tracker_component(self.component_cfg, "event_branch")
        track_branch_cls, track_branch_kwargs = resolve_tracker_component(self.component_cfg, "track_branch")
        runtime_policy_cls, runtime_policy_kwargs = resolve_tracker_component(self.component_cfg, "runtime_policy")
        track_codec_cls, track_codec_kwargs = resolve_tracker_component(self.component_cfg, "track_codec")

        self.encoder = encoder_cls(
            embed_dim=self.embed_dim,
            structural_width_ratio=self.structural_width_ratio,
            pruning_cfg=self.pruning_cfg,
            patch_embed_cfg=self.patch_embed_cfg,
            patch_frontend=self.patch_frontend,
            frame_adapter=self.frame_adapter,
            event_adapter=self.event_adapter,
            backbone=self.backbone,
            event_depth_limit=self.event_cut_depth,
            track_depth_limit=self.track_depth,
            **encoder_kwargs,
        )
        head_factory = head_factory_cls(
            embed_dim=embed_dim,
            input_size=self.input_size,
            aux_classes=aux_classes,
            head_hidden_dim=head_hidden_dim,
            track_head_hidden_dim=track_head_hidden_dim,
            mask_hidden_dim=mask_hidden_dim,
            search_head_variant=search_head_variant,
            search_head_residual_hidden_dim=search_head_residual_hidden_dim,
            search_center_candidate_count=search_center_candidate_count,
            search_center_candidate_max_delta_px=search_center_candidate_max_delta_px,
            eye_detector_mode=self.eye_detector_mode,
            eye_head_variant=self.eye_head_variant,
            eye_reg_max=self.eye_reg_max,
            mask_variant=self.mask_variant,
            roi_bbox_cfg=self.roi_bbox_cfg,
            search_use_roi_bbox_head=self.search_use_roi_bbox_head,
            roi_bbox_use_external_eye=self.roi_bbox_use_external_eye,
            mask_cfg=self.mask_cfg,
            track_state_simdr_bins=track_state_simdr_bins,
            track_center_heatmap_grid=track_center_heatmap_grid,
            track_center_refine_max_delta_px=track_center_refine_max_delta_px,
            track_center_candidate_count=track_center_candidate_count,
            track_center_candidate_max_delta_px=track_center_candidate_max_delta_px,
            **head_factory_kwargs,
        )
        head_modules = head_factory.build(
            enable_eye_head=enable_eye_head,
            enable_search_head=enable_search_head,
            enable_event_head=enable_event_head,
            enable_track_head=enable_track_head,
            enable_mask_head=enable_mask_head,
            enable_aux_head=enable_aux_head,
            enable_search_bbox_aux_head=enable_search_bbox_aux_head,
            enable_event_bbox_aux_head=enable_event_bbox_aux_head,
            enable_search_obb_aux_head=enable_search_obb_aux_head,
            enable_event_obb_aux_head=enable_event_obb_aux_head,
            enable_search_center_candidate_head=enable_search_center_candidate_head,
            enable_track_state_aux_head=enable_track_state_aux_head,
            enable_track_state_simdr_head=enable_track_state_simdr_head,
            enable_track_center_heatmap_head=enable_track_center_heatmap_head,
            enable_track_center_refine_head=enable_track_center_refine_head,
            enable_track_center_candidate_head=enable_track_center_candidate_head,
        )
        self.eye_head = head_modules.eye_head
        self.search_head = head_modules.search_head
        self.event_head = head_modules.event_head
        self.search_bbox_aux_head = head_modules.search_bbox_aux_head
        self.event_bbox_aux_head = head_modules.event_bbox_aux_head
        self.search_obb_aux_head = head_modules.search_obb_aux_head
        self.event_obb_aux_head = head_modules.event_obb_aux_head
        self.search_center_candidate_head = head_modules.search_center_candidate_head
        self.roi_bbox_head = head_modules.roi_bbox_head
        self.track_head = head_modules.track_head
        self.track_state_aux_head = head_modules.track_state_aux_head
        self.track_state_simdr_head = head_modules.track_state_simdr_head
        self.track_center_heatmap_head = head_modules.track_center_heatmap_head
        self.track_center_refine_head = head_modules.track_center_refine_head
        self.track_center_candidate_head = head_modules.track_center_candidate_head
        self.mask_head = head_modules.mask_head
        self.aux_head = head_modules.aux_head
        self.search_refiner = refiner_cls(
            use_mask_centroid=self.search_use_mask_centroid,
            use_mask_cascade=self.search_use_mask_cascade,
            cascade_detach=self.search_mask_cascade_detach,
            cascade_use_xy=self.search_mask_cascade_use_xy,
            cascade_use_ab=self.search_mask_cascade_use_ab,
            cascade_xy_scale=self.search_mask_cascade_xy_scale,
            cascade_ab_scale=self.search_mask_cascade_ab_scale,
            cascade_ab_mode=self.search_mask_cascade_ab_mode,
            cascade_ab_limit=self.search_mask_cascade_ab_limit,
            cascade_min_ab=self.search_mask_cascade_min_ab,
            **refiner_kwargs,
        )
        self.search_branch = search_branch_cls(
            input_size=self.input_size,
            eye_head=self.eye_head,
            search_head=self.search_head,
            search_bbox_aux_head=self.search_bbox_aux_head,
            search_obb_aux_head=self.search_obb_aux_head,
            search_center_candidate_head=self.search_center_candidate_head,
            roi_bbox_head=self.roi_bbox_head,
            mask_head=self.mask_head,
            aux_head=self.aux_head,
            eye_detector_mode=self.eye_detector_mode,
            eye_head_variant=self.eye_head_variant,
            roi_bbox_use_external_eye=self.roi_bbox_use_external_eye,
            refiner=self.search_refiner,
            candidate_as_search_state=search_center_candidate_as_search_state,
            candidate_blend=search_center_candidate_blend,
            **search_branch_kwargs,
        )
        self.event_branch = event_branch_cls(
            event_head=self.event_head,
            event_bbox_aux_head=self.event_bbox_aux_head,
            event_obb_aux_head=self.event_obb_aux_head,
            aux_head=self.aux_head,
            state_from_branch=self._state_from_branch,
            **event_branch_kwargs,
        )
        prev_hidden = int(prev_state_hidden_dim) if prev_state_hidden_dim is not None else int(embed_dim)
        self.prev_state_encoder = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, prev_hidden),
            nn.GELU(),
            nn.Linear(prev_hidden, embed_dim),
        )
        self.track_codec = track_codec_cls(**track_codec_kwargs)
        self.track_branch = track_branch_cls(
            prev_state_encoder=self.prev_state_encoder,
            track_head=self.track_head,
            track_state_aux_head=self.track_state_aux_head,
            track_state_simdr_head=self.track_state_simdr_head,
            track_center_heatmap_head=self.track_center_heatmap_head,
            track_center_refine_head=self.track_center_refine_head,
            track_center_candidate_head=self.track_center_candidate_head,
            apply_width_mask=self.encoder.apply_width_mask,
            apply_track_mask=self.encoder.apply_track_mask,
            decode_state=self.track_codec.decode,
            simdr_as_track_state=track_state_simdr_as_track_state,
            simdr_coordinate_max=track_state_simdr_coordinate_max,
            simdr_blend=track_state_simdr_blend,
            heatmap_as_track_state=track_center_heatmap_as_track_state,
            heatmap_coordinate_max=track_center_heatmap_coordinate_max,
            heatmap_blend=track_center_heatmap_blend,
            refine_as_track_state=track_center_refine_as_track_state,
            refine_blend=track_center_refine_blend,
            candidate_as_track_state=track_center_candidate_as_track_state,
            candidate_blend=track_center_candidate_blend,
            **track_branch_kwargs,
        )
        self.scheduler = TrackSearchSchedulerFSM(**self.runtime_cfg)
        self.runtime_policy = runtime_policy_cls(
            scheduler=self.scheduler,
            prime_patch_cache=self.prime_patch_cache,
            invalidate_patch_cache=self.invalidate_patch_cache,
            **runtime_policy_kwargs,
        )

    @staticmethod
    def _state_from_branch(branch_logits: torch.Tensor) -> torch.Tensor:
        return branch_logits[..., :6]

    @staticmethod
    def _normalize_uv(uv: torch.Tensor) -> torch.Tensor:
        return TrackStateCodec.normalize_uv(uv)

    def set_epoch_context(self, epoch: int | None) -> None:
        self.current_epoch = 0 if epoch is None else int(epoch)
        self.encoder.set_epoch_context(epoch)

    def prime_patch_cache(self, frame: torch.Tensor) -> PatchEmbedCache:
        return self.encoder.prime_patch_cache(frame)

    def invalidate_patch_cache(self) -> PatchEmbedCache:
        return self.encoder.invalidate_patch_cache()

    def encode_frame(self, frame: torch.Tensor, *, width_ratio: float | None = None) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], int]:
        return self.encoder.encode_frame(frame, training=self.training, width_ratio=width_ratio)

    def encode_event(self, event: torch.Tensor, *, width_ratio: float | None = None) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int], int]:
        return self.encoder.encode_event(event, training=self.training, width_ratio=width_ratio)

    def decode_track_state(self, prev_state: torch.Tensor, track_logits: torch.Tensor) -> torch.Tensor:
        return self.track_codec.decode(prev_state, track_logits)

    def map_pretrained_state_dict(self, source_state: dict[str, torch.Tensor], *, source: str = "auto") -> dict[str, torch.Tensor]:
        del source
        return self.encoder.map_pretrained_state_dict(source_state)

    def forward_search(self, frame: torch.Tensor, *, width_ratio: float | None = None) -> dict[str, torch.Tensor]:
        tokens, pooled, grid_size, active_dim = self.encode_frame(frame, width_ratio=width_ratio)
        return self.search_branch.forward(
            tokens=tokens,
            pooled=pooled,
            grid_size=grid_size,
            active_dim=active_dim,
            device=frame.device,
            batch_size=frame.shape[0],
        )

    def forward_event(self, event: torch.Tensor, *, width_ratio: float | None = None) -> dict[str, torch.Tensor]:
        if not self.event_branch.enabled:
            return {}
        _, pooled, _, active_dim = self.encode_event(event, width_ratio=width_ratio)
        return self.event_branch.forward(
            pooled=pooled,
            active_dim=active_dim,
            device=event.device,
            batch_size=event.shape[0],
        )

    def forward_track(
        self,
        event: torch.Tensor,
        prev_state: torch.Tensor,
        *,
        cached_frame: torch.Tensor | None = None,
        patch_cache: PatchEmbedCache | None = None,
        width_ratio: float | None = None,
    ) -> dict[str, torch.Tensor]:
        pooled, active_dim = self.encoder.encode_track(
            event,
            cached_frame=cached_frame,
            patch_cache=patch_cache,
            training=self.training,
            width_ratio=width_ratio,
        )
        return self.track_branch.forward(
            pooled=pooled,
            prev_state=prev_state,
            active_dim=active_dim,
            device=event.device,
            batch_size=event.shape[0],
        )

    def forward_train(self, batch: dict[str, Any], *, width_ratio: float | None = None) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        frame = batch.get("frame")
        if frame is not None:
            outputs.update(self.forward_search(frame, width_ratio=width_ratio))

        event = batch.get("event")
        if event is not None and self.event_branch.enabled:
            outputs.update(self.forward_event(event, width_ratio=width_ratio))

        prev_state = batch.get("prev_state")
        cached_frame = batch.get("cached_frame")
        if cached_frame is None and frame is not None:
            cached_frame = frame
        patch_cache = batch.get("patch_cache")
        if not isinstance(patch_cache, PatchEmbedCache):
            patch_cache = None

        if event is not None and prev_state is not None and self.track_branch.enabled:
            outputs.update(
                self.forward_track(
                    event,
                    prev_state,
                    cached_frame=cached_frame,
                    patch_cache=patch_cache,
                    width_ratio=width_ratio,
                )
            )
        reference = frame if frame is not None else event if event is not None else prev_state
        if reference is not None:
            resolved_width = self.encoder.resolve_width_ratio(training=self.training, requested=width_ratio)
            outputs["pruning/active_width"] = torch.full((reference.shape[0],), float(resolved_width), device=reference.device)
        return outputs

    def forward(self, batch: dict[str, Any], *, width_ratio: float | None = None) -> dict[str, torch.Tensor]:
        return self.forward_train(batch, width_ratio=width_ratio)

    @torch.no_grad()
    def runtime_step(
        self,
        *,
        frame: torch.Tensor,
        event: torch.Tensor,
        prev_state: torch.Tensor,
        similarity: torch.Tensor | None = None,
        event_density: torch.Tensor | None = None,
        closed_eye_flag: torch.Tensor | None = None,
        patch_cache: PatchEmbedCache | None = None,
    ) -> dict[str, torch.Tensor | str | PatchEmbedCache]:
        batch = {
            "frame": frame,
            "event": event,
            "prev_state": prev_state,
            "cached_frame": frame,
            "patch_cache": patch_cache,
        }
        outputs = self.forward_train(batch)
        return self.runtime_policy.apply(
            frame=frame,
            outputs=outputs,
            similarity=similarity,
            event_density=event_density,
            closed_eye_flag=closed_eye_flag,
        )


def _build_hbtxr_tracker_config():
    """Build a frozen dataclass mirroring HBTXRTracker.__init__ exactly.

    Introspecting the signature keeps the config in perfect sync with the
    constructor, so the two can never disagree.
    """
    signature = inspect.signature(HBTXRTracker.__init__)
    specs = []
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        annotation = (
            parameter.annotation
            if parameter.annotation is not inspect.Parameter.empty
            else Any
        )
        if parameter.default is inspect.Parameter.empty:
            specs.append((name, annotation))
        else:
            specs.append((name, annotation, field(default=parameter.default)))

    def to_kwargs(self) -> dict[str, Any]:
        """Expand the config back into constructor keyword arguments."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    return make_dataclass(
        "HBTXRTrackerConfig",
        specs,
        frozen=True,
        namespace={"to_kwargs": to_kwargs},
    )


HBTXRTrackerConfig = _build_hbtxr_tracker_config()


def _hbtxr_tracker_from_config(cls, config: HBTXRTrackerConfig) -> HBTXRTracker:
    """Construct an HBTXRTracker from a single HBTXRTrackerConfig object."""
    return cls(**config.to_kwargs())


HBTXRTracker.from_config = classmethod(_hbtxr_tracker_from_config)
