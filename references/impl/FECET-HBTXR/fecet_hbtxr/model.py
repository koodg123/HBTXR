from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .scheduler import TrackSearchSchedulerFSM


def flatten_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def tokens_to_grid(tokens: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    batch, num_tokens, channels = tokens.shape
    if num_tokens != height * width:
        raise ValueError(f"Token count mismatch: got {num_tokens}, expected {height * width}")
    return tokens.transpose(1, 2).contiguous().view(batch, channels, height, width)


class FramePatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature = self.proj(frame)
        return flatten_tokens(feature), (feature.shape[-2], feature.shape[-1])


class EventPatchEmbeddingFACET(nn.Module):
    def __init__(self, embed_dim: int = 192, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv2d(2, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, event: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        feature = self.proj(event)
        return flatten_tokens(feature), (feature.shape[-2], feature.shape[-1])


class ModalityAdapter(nn.Module):
    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.net(tokens)


class HGPipeAttentionStage(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        del mlp_ratio
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = 1.0 / math.sqrt(float(self.head_dim))
        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.shape
        return x.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    @staticmethod
    def _merge_heads(x: torch.Tensor) -> torch.Tensor:
        batch, num_heads, tokens, head_dim = x.shape
        return x.transpose(1, 2).contiguous().view(batch, tokens, num_heads * head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        probs = self.dropout(torch.softmax(scores, dim=-1))
        attn = torch.matmul(probs, v)
        attn = self._merge_heads(attn)
        return residual + self.dropout(self.out_proj(attn))


class HGPipeMLPStage(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        del num_heads
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return residual + self.dropout(x)


class PartialDeiTTiny(nn.Module):
    def __init__(self, embed_dim: int = 192, depth: int = 6, num_heads: int = 3, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn_stages = nn.ModuleList(
            [HGPipeAttentionStage(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.mlp_stages = nn.ModuleList(
            [HGPipeMLPStage(embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = tokens
        for attn_stage, mlp_stage in zip(self.attn_stages, self.mlp_stages):
            x = attn_stage(x)
            x = mlp_stage(x)
        x = self.norm(x)
        return x, x.mean(dim=1)


def _mlp_head(in_dim: int, out_dim: int) -> nn.Sequential:
    hidden = max(in_dim, out_dim)
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class EyeRegionHead(nn.Module):
    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, 5)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class PupilSearchHead(nn.Module):
    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, 7)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class EventSearchHead(nn.Module):
    def __init__(self, embed_dim: int = 192) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, 7)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class PupilTrackHead(nn.Module):
    def __init__(self, in_dim: int = 384) -> None:
        super().__init__()
        self.net = _mlp_head(in_dim, 8)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class AuxStateHead(nn.Module):
    def __init__(self, embed_dim: int = 192, num_classes: int = 5) -> None:
        super().__init__()
        self.net = _mlp_head(embed_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.net(pooled)


class SearchMaskHead(nn.Module):
    def __init__(self, embed_dim: int = 192, output_size: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        hidden = max(embed_dim // 2, 32)
        self.output_size = tuple(int(v) for v in output_size)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_dim, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor, *, grid_size: tuple[int, int]) -> torch.Tensor:
        feature = tokens_to_grid(tokens, height=grid_size[0], width=grid_size[1])
        logits = self.decoder(feature)
        return F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)


class HBTXRTracker(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        input_size: tuple[int, int] = (256, 256),
        dropout: float = 0.0,
        aux_classes: int = 5,
        runtime_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.input_size = tuple(int(v) for v in input_size)

        self.frame_embed = FramePatchEmbedding(embed_dim=embed_dim, patch_size=patch_size)
        self.event_embed = EventPatchEmbeddingFACET(embed_dim=embed_dim, patch_size=patch_size)
        self.frame_adapter = ModalityAdapter(embed_dim=embed_dim)
        self.event_adapter = ModalityAdapter(embed_dim=embed_dim)
        self.backbone = PartialDeiTTiny(
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.eye_head = EyeRegionHead(embed_dim=embed_dim)
        self.search_head = PupilSearchHead(embed_dim=embed_dim)
        self.event_head = EventSearchHead(embed_dim=embed_dim)
        self.track_head = PupilTrackHead(in_dim=embed_dim * 2)
        self.mask_head = SearchMaskHead(embed_dim=embed_dim, output_size=self.input_size)
        self.aux_head = AuxStateHead(embed_dim=embed_dim, num_classes=aux_classes)
        self.prev_state_encoder = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.scheduler = TrackSearchSchedulerFSM(**(runtime_cfg or {}))

    @staticmethod
    def _state_from_branch(branch_logits: torch.Tensor) -> torch.Tensor:
        state = branch_logits[..., :6].clone()
        uv = F.normalize(state[..., 4:6], dim=-1, eps=1e-6)
        return torch.cat([state[..., :4], uv], dim=-1)

    @staticmethod
    def _normalize_uv(uv: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(torch.linalg.norm(uv, dim=-1, keepdim=True), min=1e-6)
        return uv / denom

    def encode_frame(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        tokens, grid_size = self.frame_embed(frame)
        tokens = self.frame_adapter(tokens)
        tokens, pooled = self.backbone(tokens)
        return tokens, pooled, grid_size

    def encode_event(self, event: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
        tokens, grid_size = self.event_embed(event)
        tokens = self.event_adapter(tokens)
        tokens, pooled = self.backbone(tokens)
        return tokens, pooled, grid_size

    def decode_track_state(self, prev_state: torch.Tensor, track_logits: torch.Tensor) -> torch.Tensor:
        dxdy = track_logits[..., 0:2]
        dlogab = track_logits[..., 2:4]
        duv = track_logits[..., 4:6]

        center = prev_state[..., 0:2] + dxdy
        axes = prev_state[..., 2:4] * torch.exp(dlogab)
        uv = self._normalize_uv(prev_state[..., 4:6] + duv)
        return torch.cat([center, axes, uv], dim=-1)

    def forward_search(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, pooled, grid_size = self.encode_frame(frame)
        search_logits = self.search_head(pooled)
        return {
            "search/eye": self.eye_head(pooled),
            "search/pupil": search_logits,
            "search/state": self._state_from_branch(search_logits),
            "search/mask_logits": self.mask_head(tokens, grid_size=grid_size),
            "search/aux": self.aux_head(pooled),
        }

    def forward_event(self, event: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pooled, _ = self.encode_event(event)
        event_logits = self.event_head(pooled)
        return {
            "event/pupil": event_logits,
            "event/state": self._state_from_branch(event_logits),
            "event/aux": self.aux_head(pooled),
        }

    def forward_track(self, event: torch.Tensor, prev_state: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pooled, _ = self.encode_event(event)
        prev_feat = self.prev_state_encoder(prev_state)
        fused = torch.cat([pooled, prev_feat], dim=-1)
        track_logits = self.track_head(fused)
        return {
            "track/pupil": track_logits,
            "track/state": self.decode_track_state(prev_state, track_logits),
        }

    def forward_train(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        frame = batch.get("frame")
        if frame is not None:
            outputs.update(self.forward_search(frame))

        event = batch.get("event")
        if event is not None:
            outputs.update(self.forward_event(event))

        prev_state = batch.get("prev_state")
        if event is not None and prev_state is not None:
            outputs.update(self.forward_track(event, prev_state))
        return outputs

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.forward_train(batch)

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
    ) -> dict[str, torch.Tensor | str]:
        batch = {
            "frame": frame,
            "event": event,
            "prev_state": prev_state,
        }
        outputs = self.forward_train(batch)
        search_conf = torch.sigmoid(outputs["search/pupil"][..., 6]).mean().item()
        track_conf = torch.sigmoid(outputs["track/pupil"][..., 6]).mean().item()
        track_quality = torch.sigmoid(outputs["track/pupil"][..., 7]).mean().item()
        similarity_value = float(similarity.mean().item()) if similarity is not None else 1.0
        density_value = float(event_density.mean().item()) if event_density is not None else 1.0
        closed_eye_value = bool(closed_eye_flag.mean().item() > 0.5) if closed_eye_flag is not None else False
        decision = self.scheduler.step(
            search_conf=search_conf,
            track_conf=track_conf,
            track_quality=track_quality,
            similarity=similarity_value,
            event_density=density_value,
            closed_eye_flag=closed_eye_value,
        )
        outputs["runtime/state"] = decision.state
        outputs["runtime/reason"] = decision.reason
        return outputs
