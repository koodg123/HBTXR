from __future__ import annotations

import torch


class TrackStateCodec:
    @staticmethod
    def normalize_uv(uv: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(torch.linalg.norm(uv, dim=-1, keepdim=True), min=1e-6)
        return uv / denom

    def decode(self, prev_state: torch.Tensor, track_logits: torch.Tensor) -> torch.Tensor:
        dxdy = track_logits[..., 0:2]
        dlogab = track_logits[..., 2:4]
        duv = track_logits[..., 4:6]
        center = prev_state[..., 0:2] + dxdy
        axes = prev_state[..., 2:4] * torch.exp(dlogab)
        uv = self.normalize_uv(prev_state[..., 4:6] + duv)
        return torch.cat([center, axes, uv], dim=-1)
